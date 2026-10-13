# Calc Lifecycle

A calculation compiles a job into factor objects, AAD leaves and a deal tree; then, per batch, the
processes publish paths, the deal tree prices on them and the calculation reduces the result. This
page follows that walk in code order; entry points and output are on
[API Overview](../api_overview.md).

| where | what |
| --- | --- |
| `__init__.py` | `Context.run_job` and the `run_*` dispatch functions |
| `calculation.py` | the four calculations, their states, `batch_seed`, `DealStructure` |
| `utils.py` | `Calculation_State`, `TimeGrid`, `TensorSchedule`, the curve reads, the `BoundarySet` family |
| `pricing.py` | `interpolate`, `oss_stream`, `InnerMCRecompute`, `LogVar2FJKit`, the boundary and kink terms |

**Held by** names the test a broken rule turns red; **Not gated**, that none does.

## Dispatch

`Context.run_job` branches on `Calculation['Object']` to `run_cmc` (through
`Context.Credit_Monte_Carlo`, which shards under `runparallel`), `run_baseval`,
`run_hedgemontecarlo` or `run_simm`, each calling `construct_calculation`. The first three pick the
device and inject only `Run_Date` (and `Time_grid` for the Monte Carlo modes); every other default
comes from the class's `fields` through `declared_defaults`. `run_simm` injects nothing and picks no
device: `SIMM` values each booked trade through `run_baseval`, once per market input moved by ISDA's
definition, all on one seed.

The calculation's `Currency` is required: no default names the currency a book reports in. Before
anything runs, the block - overrides merged, completed as `declared_defaults` completes it - is
judged by its own declarations (`Calculation.judged`, through `schema.validate_block`), and a
`ValueError` names every field it refuses at once: `run_baseval` and `run_cmc` judge the merged
block, `HedgeMonteCarlo`, `SIMM` and the diary where they complete it. `Config.validate` reports the
same messages under `calculation`, and a refused block is not walked for factors.
**Held by** `tests/test_loud_refusals.py::test_a_calculation_stating_no_currency_is_refused_by_name`.

## Compile phase 1 — `calculate_dependencies` {#compile-phase-1-calculate_dependencies}

`Config.calculate_dependencies` discovers the factor universe, sorts it topologically and splits
stochastic from static ([Dependency System](dependency_system.md)); iterating `dependent_factors` is
iterating that order.

!!! note "Invariant — `calculate_dependencies` is idempotent"
    Both halves read the config and write nothing. Discovery classifies an implied model off the
    implied factor's own block and needs no `Price Models` entry, so none is written for it: a
    placeholder would pollute saved market data and tie `plan_hash` to whether a run had happened.
    A process that reads parameters from that entry refuses by name without one -
    `HullWhite2FactorImpliedInterestRateModel` reads its market prices of risk there. **Held by**
    `tests/test_loud_refusals.py::test_an_implied_model_with_no_price_models_block_is_refused_by_name`.

## Compile phase 2 — `_build_factor_state` {#compile-phase-2-_build_factor_state}

1. **Factors.** `construct_process` for each stochastic factor, `construct_factor` for the rest, a
   missing block skipped with a WARNING.
2. **Leaves** ([@capriotti2011]): `stoch_var`, `implied_var` and `static_var`, `Gradient_Variables`
   choosing which require grad. `Calculation.factor_leaf` mints each, the one seam where a
   calibrated factor carries its quotes' graph as `leaf + (θ* − θ*.detach())`
   ([Quote Sensitivities](quote_sensitivities.md#the-attachment)).
3. **Kernels.** `Deterministic_Kernels` is applied both ways on every run, the switch being
   process-global. **Held by**
   `tests/test_recompute_inner_mc.py::test_deterministic_kernels_is_read_every_run_and_set_both_ways`.
4. **State.** `_init_shared_mem` builds the state around `get_cholesky_decomp`, the only place
   `process_ofs` is filled.
5. **`precalculate`**, once `factor_key` is set, fills `z_offset` and `spot0` (its arguments cached
   for a burn-in or a fork); **`calc_references`** follows.

!!! warning "Invariant — `precalculate` before `calc_references`"
    A cross-process link needs what `precalculate` set on the other process. Both loops run in
    topological order, which is also what makes publishing as you go safe.

!!! warning "Invariant — implied-leaf dedupe"
    A factor that is both a static dependent and a spot process's implied factor is one tensor (the
    static branch reuses `implied_leaves`), so `backward()` sums both consumers into it. A second
    leaf under the same name splits the gradient between the pricer path (`t_Static_Buffer`) and the
    scenario path (`implied_tensor`), and desyncs a bump.

### Correlation and `process_ofs` {#correlation}

`correlation_name` returns `(corr_type, [sub-factors])`, one random row per sub-factor, and
`process_ofs[key]` is the first. An undeclared pair in `Correlations` reads 0.0, and a name no
simulated factor answers to is named at INFO. A declared number is the innovations' correlation for
every process; nothing scales it.

!!! warning "Invariant — `num_factors()` equals `len(correlation_name[1])`"
    The first sizes the random block, the second the matrix and the offsets; nothing asserts it. A
    mismatch shifts every later `z_offset`, so a process reads another factor's rows or fails on a
    shape. **Not gated.**

Below an eigenvalue of `1e-8` the matrix is repaired with a WARNING, its size unreported:
`Eigenvalue_Raising` (default) floors eigenvalues at `1e-4` and rescales to a unit diagonal;
`Alternating_Projections` is Higham's method [@higham2002] without Dykstra's correction (100 passes
at most, to a relative `1e-8`), a nearby matrix, not provably the nearest. The matrix is no AAD leaf
([decision 12](../limitations.md#decisions-waiting-on-the-desk)); a base valuation bumps a quanto or
compo correlation under common random numbers (`Correlation_Bump`, 0.025).

## The batch loop

`Credit_Monte_Carlo.execute` compiles the deal tree (`set_deal_structures`), then per batch:

1. Under `deterministic_batches`, `batch_seed(Random_Seed, b)` reseeds and `set_quasi_batch(b)`
   anchors the quasi stream, `b` the global batch.
2. `reset` correlates one `randn(num_factors, n·T)` into `t_random_numbers` `(num_factors, T, n)`,
   resets cashflows and clears `t_Buffer` and `boundary_sets`; `Antithetic` appends the negation of
   a `Batch_Size // 2` draw, pairing path `j` with `j + Batch_Size/2` (an odd `Batch_Size` is one
   column short, unrefused).
3. Each process's `generate` publishes to `t_Scenario_Buffer` in topological order.
4. `resolve_structure` prices each deal, accumulates and runs `post_process`, negating a
   `FLIP`-prefixed set, dropping a NaN one and stamping each boundary registration with its deal.
5. The CollVA, initial-margin, FVA and CVA reductions run in that order
   ([xVA](../theory/valuation_adjustments.md)), do-not-touch
   ([Change Scope](conventions.md#change-scope)).

!!! note "Run controls"
    `Stats` counts deals and structures loaded, `Ignored`, and `Deals Skipped` / `Structs Skipped`
    once each: a deal that fails to compile is dropped (kept at zero under
    `Exclude_Deals_With_Missing_Market_Data: No`), a pricing exception marks zero unless
    `utils.is_fatal_pricing_error` names it (out of memory, an unbound schedule,
    `UnpriceableSchedule`). `Device_Memory` reports the peak, warning past the device; a run that
    valued nothing refuses. **Held by** `tests/test_fork_pricing_failures.py`,
    `tests/test_multi_gpu.py::test_a_run_reports_what_it_held_and_warns_where_it_passed_the_device`.

!!! warning "Invariant — `t_Buffer` per batch, `t_PreCalc` per calculation"
    `t_Buffer` is cleared by `reset` and by every fork. `t_PreCalc` holds what every batch reads
    unchanged (integral precalcs, gathered indices, arrays through `Calculation_State.upload`) and
    nothing carrying a graph.

!!! warning "Invariant — publish as you go, under `(factor_key, kind)`"
    A process publishes its path as `generate` returns and any sufficient statistic under
    `(factor_key, kind)`, so a linked factor reads its parent's path. The calculation and the solver
    never name a regime, belief or variance; they iterate model-agnostic verbs.

## Random streams {#rng-substream-ordering-the-reproducibility-surface}

`process_ofs` follows discovery's canonical order, a function of the factor graph alone;
[Dependency System](dependency_system.md#rng-ordering) says how and what moves it.

| stream | seeded by | read by |
| --- | --- | --- |
| `t_random_numbers` and pseudo-random inner draws: Mersenne Twister on the host [@matsumoto1998], Philox on CUDA [@salmon2011] | `Random_Seed` (default 5120) plus `job_id`, or per batch `batch_seed`, one SplitMix64 round [@steele2014] | every `generate` |
| `quasi_rng`: Sobol [@sobol1967], Joe–Kuo directions [@joe2008], scrambled [@owen1998] | `QUASI_SEED = 1234` from `QUASI_ANCHOR = 1024`, never `Random_Seed` | regime, GARCH and basis-linked spots, LogVar2FJ's extra mixers, `reset_inner` |
| `inner_block`: those points as integers | its shape | pricers' [inner paths](#inner-block) above 16 outer paths |
| a `torch.Generator` per LogVar2FJ walk segment | an integer off the default generator, plus the segment | the daily shocks, redrawn in backward |

`quasi_rng` keeps its position across batches. Under `deterministic_batches` (the `runparallel`
dispatch only) batch `b` reads from `QUASI_ANCHOR + b · span · sample_size` on any worker, and a
second draw of one shape in a batch refuses by name. A sharded run is then bit-identical in the
worker count on one device type, and a different draw of the same law from an unsharded one.
**Held by** `tests/test_multi_gpu.py::test_a_sobol_consuming_world_is_bit_identical_in_the_worker_count`,
`::test_the_unsharded_hmm_path_is_untouched`.

## Deals and schedules

`Factor_dep` is a deal's compiled factor lookup, consumed unchanged by `generate`
([Resolver Layer](resolver_layer.md)); `Time_dep` maps its dates onto the MTM grid, where
`pricing.interpolate` puts its price (zero past its last date) and saves `Calc_res`.

### The schedule lifecycle {#the-schedule-lifecycle}

A `TensorSchedule` is numpy index columns for the compile and a device copy for the arithmetic,
separated in time by `bind`, which `utils.bind_schedules` wraps round `calc_dependencies`. Before it
only compile-time edits write (`carry`, `overwrite_rate`, `set_resets`, `insert_cashflow`); after it
`dual` serves the copy and an edit raises `ScheduleLifecycleError`, since it would not reach the
copy. `BenchmarkInstruments` (last, `bind` splicing in the quote overlay) and the Hull-White
calibration's `create_market_swaps` bind for themselves; `MtMCrossCurrencySwapDeal.post_process`
reopens a leg and binds it again. `Diary` reads schedules by the binding's own walk (**Held by**
`tests/test_diary.py::test_the_diary_reads_the_legs_the_binding_binds`).

!!! warning "Invariant — `derived` is run-scoped"
    A tensor derived from a schedule's copy belongs in `derived`, which `bind` mints fresh, never
    in `Factor_dep` (compile output) or `t_Buffer` (cleared per batch). An unbound schedule raises
    on first touch and is fatal, so no guard turns it into a zero mark.

## Grids {#grids}

| grid | dates |
| --- | --- |
| `Time_Grid` | items `offset` or `offset(step)` in units `D`, `W`, `M`, `Y`, stepping to the next item; default `0d 2d 1w(1w) 3m(1m) 2y(3m)`. A trailing `)` repeats the last step to the book's last date, else later deal dates are clipped. Unadjusted for business days |
| MTM | the parsed grid plus every later date a deal revalues on (a collateralised set adds call, settlement and liquidation dates) |
| scenario | the MTM grid under `Dynamic_Scenario_Dates: Yes` (default; forced by `At_Percentile`), else the parsed grid plus one date past the book; between scenario dates a factor is interpolated linearly |
| deal | each deal's revaluation dates, interpolated onto the MTM grid |

The report rows (`report_index`), on which the xVA reductions run, are each top-level structure's
MTM dates to its last revaluation, a collateralised set's only where its settlement and liquidation
dates are on the grid ([xVA](../theory/valuation_adjustments.md#notation-and-the-grid)).

## Valuation modes {#valuation-modes}

**`Base_Revaluation`** is one date, every factor static, no Cholesky or generate loop:
`resolve_structure` runs once, `pricing.greeks` reports `Greeks_First`, and `Greeks: 'All'` adds the
full Hessian `Greeks_Second`, reported nowhere else. A Monte Carlo pricer still simulates inside
(`MCMC_Simulations` 16,384; 2,048 on the credit Monte Carlo).

**`HedgeMonteCarlo`** runs the credit Monte Carlo engine and harvests its marks. `update_factors`
adds `Scenario_Factors` (factors no deal reaches) and caps the grid at the liability terminal; a leg
that fails to compile kills the run, a skipped leg changing the solver's problem (**Held by**
`tests/test_fork_pricing_failures.py::test_a_hedge_book_leg_that_fails_to_compile_kills_the_run`).
Per batch it publishes print seeds first and drops them once generated (consumed state, not path
series), burns in under `Randomize_Initial_State`, replays
`Observed_Scenario` paths through `reseed_from_path`, and under `solve_hedge` snapshots the outer
buffer for forks. A solve is a stream: a `Bundle` per batch to one `StreamingSolve` (`warmup`
locking the frame, `step` per training batch, `finish` on the held-out last), so
`Simulation_Batches` is a stream length under `solve_hedge` and a path multiplier under
`simulate_only`.

!!! warning "Invariant — a frozen evaluation is the stream of length one"
    Under `DiffV2_Load_Value_Fn` nothing is fitted, so `Simulation_Batches` must be 1, that batch
    both warmup and held-out world; `step` skips a loaded net, and `finish` does not re-fork its own
    bundle, which would advance the Sobol stream and move the verdict. **Held by**
    `tests/test_frozen_policy_is_frozen.py`.

!!! warning "Invariant — `Z.ndim` dispatch"
    One process's `generate()` serves both loops: outer `(T, B)` and inner `(T, B, B2)`, the
    per-outer-path state broadcast on the middle axis.

!!! warning "Invariant — a fork's batch state, and pricing that fails loud"
    `simulation_batch` and `fillvalue` track the fork's batch and are restored in a `finally`, or a
    mid-fork raise fails the next step on shapes instead of its cause. A swallowed failure inside a
    fork is a zero mark that corrupts the solver's labels, so the liability is checked on its flat
    shape and every live tradable on having a `tensor_marks` entry. **Not gated:** no document can
    inject a failure into a fork.

!!! warning "Invariant — `keep_tensor` carries the tradable series"
    `HedgeMonteCarlo` sets `keep_tensor` whatever `Keep_Tensor` says; without it the tradables'
    `Calc_res['tensor']` vanish with no error.

## The inner-MC fork {#inner-mc-subsystem}

`_run_inner_mc_at_t` forks each outer path's state at step `t` on process copies, truncates the grid
and each deal's `Time_dep` to `{t, t+1}`, and runs one pass of `Batch_Size × Inner_Sub_Batch`
samples, its memory set by those two fields. It draws (`reset_inner`), precalculates each
process from `outer_buffer[key][t]` (a leaf under `with_grad`), publishes every seed before any
generate (a seed may condition a process upstream), generates and calls `reseed_inner_state`, then
publishes as a `ScenarioSource` every path series it wrote that the outer path carries — the outer
past at `B_outer`, the forked rows flattened — and prices once. Every series, not every factor: a
process's own `(key, kind)` publication (`basis_mu`) is read like a factor and needs the same grid;
the fork's seeds are excluded by the same rule. **Not gated:** no test forks a process series.

The LogVar2FJ outer publishes its detached state (`lv_ell`, `lv_s`), grid days (`lv_days`)
and a fork's seed (`ell0_inner`, `s0_inner`); `pricing.LogVar2FJKit.carried` reads the seed, else
the state at the row's node, else re-seeds at `(L*(t_row), 0)`, leaving other documents unaffected.

### The LogVar2FJ outer {#logvar2fj-outer}

`LogVar2FJImpliedSpotModel` is the xVA scenario generator for an `EquityPrice` or `FxRate` under LogVar2FJ (the model: [theory](../theory/stochastic_volatility.md#one-walk-three-readers)). What a developer needs from it:

- **One walk.** It is an implied process reading the calibrated `LogVar2FJModelParameters` factor through `implied_tensor`, so a CVA sensitivity reaches the one leaf the pricer reads; and its `draws` and `residual` ARE `pricing.LogVar2FJKit`'s, bound on the class, so outer, fork and pricer cannot disagree. Do not give it a draw or a root of its own.
- **Four rows per name.** `correlation_name` is `('LognormalDiffusionProcess', [(), ('S',), ('L',), ('G',)])`: the return's Gaussian given the mixer, the fast and slow log-variance shocks, and the mixer through its quantile `G = F_IG⁻¹(Φ(Z_G))`, taken in double. Each variance factor steps node to node by its exact OU transition driven by its framework normal, the days inside the interval bridged between the endpoints; whole trading days plus one shorter remainder step span each interval.
- **The day's shocks are a function of the day.** One `torch.Generator` per calendar-anchored segment of `LogVar2FJKit.CHECKPOINT_STEPS` trading days draws them; they are never stored and are redrawn in the checkpoint's recompute, so where the scenario nodes fall moves no draw. The antithetic half takes `-eta`. `Checkpoint_Outer_Walk` (a `CreditMonteCarlo` field, default `Yes`) puts the walk under the checkpoint.
- **No replay.** The variance carries its own shocks, so `(ell, s)` is not a function of realised returns and `reseed_from_path` refuses by name.

## Curve reads {#curve-reads}

A read is a query, logical storage and one physical interpolation, never two in one object.
`CurveTensor`, the query, holds scenario rows (`index`, `index_next`, `alpha`), not a flattened
`row × n_tenors` offset, because a tenor segment has its own stride. A `ScenarioSource` is a
sequence of `ScenarioBlock`s placed by `first_row` and `batch_index`, published only by a fork.
`utils.build_interpolation` builds, recursively, an `Interpolation` (one tensor) or a
`SegmentedInterpolation` (leaves over the tenor axis for a `Near_Interpolation` curve, the far leg
what `Price Factor Interpolation` resolves), both `UnroutedInterpolation`s, or for a source a
`RoutedInterpolation` over its blocks, so a segmented curve in a fork is no special case.

- **Blocks, not a join.** The past is identical across inner draws, so a join would copy it
  `Inner_Sub_Batch` times; `ScenarioBlock.project` widens a read's result, never the stored tensor.
- **Order.** Raw read, time blend, then `combine` (RT scaling, segment select): projection and
  `combine` are linear, so the routed path is the unrouted arithmetic in its order. A blended read
  reaches `index + 1`, so `route` classifies on where a read ends.
- **Write-once.** A source answers only `shape`, `new` and the RT rescale, so a late write fails
  loud rather than materialising the grid.
- **Hermite at the read.** A graph-free curve derives its Hermite pair per read
  (`Interpolation.hermite_at`), equal to the held pair to the bit; a curve carrying a graph holds
  the pair, or a derivative would sum its reads after the Hermite map, moving every greek in the
  last bit.

**Held by** `tests/test_interest_rate_prices.py::test_the_near_splits_far_leg_is_the_curves_own_rule`,
`tests/test_forward_rate_extrapolation.py::test_a_hermite_curve_document_prices_the_cubic_by_hand`.
**Not gated:** routed against joined reads, and the refused late write.

## The canonical inner block and the streamed strip {#inner-block}

**One block.** `CMC_State.inner_block(rows, sims)` returns `(u, 1 − u)`, row `k` fixing `k`, each
inner path one scrambled Sobol point over the strip. Chunk `c` of `INNER_CHUNK = 64` rows is
dimensions `64c` on of a `64(c + 1)`-wide engine, so a longer strip reads more dimensions, never more
points (point `i + 2^k` is point `i` XOR a constant). The complement is taken on the 30-bit integer,
keeping each tail in the job's dtype. Cached per inner path count, prefix-stable and broadcast over
the outer batch, the block makes a trade's value independent of what is booked before it and an
inner-MC book shardable with nothing to anchor; the inner error is common to the outer paths,
and only `MCMC_Simulations` reduces it [@gordy2010].
**Held by** `tests/test_recompute_inner_mc.py::test_a_trade_is_valued_on_its_own_draws_whatever_is_booked_before_it`,
`tests/test_multi_gpu.py::test_an_inner_monte_carlo_book_is_bit_identical_in_the_worker_count`,
`tests/test_barrier_bridge.py::test_a_strip_longer_than_a_block_chunk_prices_to_an_independent_simulation`.

**The strip streams.** The four one-step-survival pricers (discrete barrier, accumulator, TARF,
autocall; [@glasserman2001; @alm2013]) walk their fixings through `pricing.oss_stream`: under a
gradient every `oss_chunk` fixings (4) are one checkpoint, else a plain loop. A step binds its row's
tensors as defaults, since the checkpoint replays it after the loop moved on, and what consumers read
detached (the alive branch, the pending payment, the autocall's counterfactual) is built under
`no_grad`. **Held by**
`tests/test_recompute_inner_mc.py::test_the_streamed_strip_holds_its_peak_and_the_unstreamed_gradient`
(CUDA only).

## Recomputing the inner Monte Carlo {#recompute-inner-mc}

The terminal `backward()` holds every Monte Carlo pricing's graph, while the simulation is cheap.
`Recompute_Inner_MC` (base valuation and credit Monte Carlo, default `No`) trades the tape for a
second forward pass [@griewank2000]: `pricing.InnerMCRecompute` runs forward under `no_grad` and
re-runs the same callable under `enable_grad` in backward, a block at a time. The four adopters
(`pv_MC_Tarf`, `pv_MC_AutoCallSwap`, `pv_MC_Accumulator`, `pv_discrete_barrier_option`) call
`InnerMCRecompute.run`, with no per-pricer flag: what a run can tape is the machine's property, not
the trade's. `run` uses `cls.apply`, so a subclassing gate gets its mutation. An adopter:

- is `simulate(*bound)(*theta)`, the block's shape bound and every graph-carrying tensor in `theta`;
- returns a tuple, the marks then by-products the caller performs once, each graph-carrying one a
  top-level element (nested in a list it is no output, its gradient silently zero);
- is one function called twice, never a fast forward beside a differentiable copy;
- is pure but for its return, since a side effect fires again in the replay;
- reads nothing from a closure, which differentiates as a constant: all four build their vol strips
  at the call site and pass the spot model's scalars, and the autocall hoists its floating leg and
  past fixings (**Not gated:** those two).

**The position is the storage.** The node saves where the plain generator stood (`rng_position`,
host and device), not what it drew; the inner Sobol rows are the canonical block and have none.
**Held by** the `…_bit_identical_with_the_node_on` tests in `tests/test_recompute_inner_mc.py` and
`tests/test_recompute_equity_pricers.py`, by `array_equal`, since a desynchronised stream prices
right and differentiates wrong.

!!! warning "Invariant — settle outside the callable"
    Settled inside, a cashflow books twice, and the credit Monte Carlo saves cashflows after the
    batch's backward, so the report sees both. The forward's booking carries no graph, so a
    collateralised exposure reading the ledger through `C_ts_te` loses that channel of its gradient.
    **Held by**
    `tests/test_recompute_inner_mc.py::test_the_exposure_its_cashflows_and_the_cva_gradient_are_bit_identical_with_the_node_on`,
    `tests/test_recompute_equity_pricers.py::test_a_collateralised_gradient_is_what_the_settle_outside_rule_is_gated_on`
    (whose docstring records the size).

!!! warning "Invariant — a first-order node"
    The replay is rooted at detached inputs, so its `autograd.grad` cannot walk into the outer graph
    and count the first derivative twice; a second derivative through it is severed, and `backward`
    refuses `create_graph`, naming the switch. It asks `torch.is_grad_enabled()` before
    `enable_grad`: the engine runs a backward node in `create_graph` mode, and inside `enable_grad`
    the flag is always True. **Held by**
    `tests/test_recompute_equity_pricers.py::test_the_second_derivative_is_refused_rather_than_reported_wrong`.

## Boundary corrections — the sensitivity subsystem {#boundary-corrections-the-sensitivity-subsystem}

Some deals decide on simulated state: a barrier, an autocall trigger, a swaption exercise, a
collateral transfer clearing its minimum. The price is right; ordinary AAD drops the flux of
scenarios across the trigger, whose indicator has zero derivative almost everywhere.
`pricing.stochastic_boundary_correction` adds the missing term, exactly zero forward
(`gap − gap.detach()`), so nothing reported moves and the derivative reaches every factor in one
sweep. **Held by**
`tests/test_boundary_pricer_events.py::test_asking_for_sensitivities_does_not_move_the_autocall_exposure`,
`tests/test_boundary_tarf_events.py::test_asking_for_sensitivities_does_not_move_the_tarf_price_or_exposure`.

**The estimator.** For a decision $\mathbb 1\{G>0\}$ whose firing changes the objective by $J$, AAD
drops

$$f_G(0)\;\mathbb E\Big[J\,\frac{\partial G}{\partial\theta}\;\Big|\;G=0\Big]$$

([@lyuu2011]; [@glasserman2003], ch. 7). `pricing.boundary_weights` estimates the two factors apart
from $n$ gaps $g_k$: with $h=$ `Boundary_AAD_Bandwidth` $\times\ \mathrm{sd}(g)$,
$K_k=e^{-g_k^2/2h^2}$ and $s_r=\sum_k K_k g_k^r$,

$$\hat f_G(0)=\frac{s_0}{n\sqrt{2\pi}\,h},\qquad w_k=\frac{K_k\,(s_2-g_k s_1)}{s_2s_0-s_1^2},$$

a kernel density and local-linear weights [@fan2018] summing to one, which hold still over a range
of $h$ where a local-constant fit carries an $O(h)$ bias. The term is
$\sum_k (g_k-\bar g_k)\,\hat f_G(0)\,w_kJ_k$, $\bar g$ and the coefficient detached; one gap (a base
valuation's scenario) gives zero width and no correction.

`Boundary_AAD_Bandwidth` is 0.01. `Boundary_AAD_Window_Touch` (`Yes`, credit Monte Carlo) registers
the partial-time barrier's window touch. `Branch_And_Weight` (`Yes`, base valuation) prices the four
fixing-observed knockouts with a smooth estimator that registers nothing (**Held by**
`tests/test_branch_and_weight.py::test_a_registration_the_switch_supersedes_is_not_lost`). The
switch is `boundary_aad = calc_greeks is not None`: no greeks, no cost.

A pricer appends a `BoundarySet`, `gap` keeping its graph (`gap > 0` is fired) and all else
detached, its branches mapped to the MTM grid by the deal's own `pricing.deal_to_mtm_grid`. A
collateralised set's `post_process` stamps its gross-to-net chain onto the registrations beneath it,
on the set because `boundary_sets` spans every set. `pricing.boundary_correction` scores them at the
objective (CVA, FVA, the base value).

!!! warning "Invariant — a counterfactual is scored on the portfolio"
    The objective applies to `resolve_structure`'s root sum over every netting set, so a
    counterfactual is the reported portfolio plus this registration's change, never one set's net.
    A one-set fixture cannot tell them apart, and the error is worst where the machinery is aimed: a
    collateralised net sits at the `relu` kink. **Held by**
    `tests/test_boundary_pricer_events.py::test_a_netting_set_is_scored_on_the_portfolio_not_on_itself`.

The classes differ in reach, so no mode flag: `LatchedBoundarySet` reaches every later row (the
latching pricers, the swaption); `InnerBoundarySet` lives inside an inner Monte Carlo (the TARF's
knock-in) and carries the objective's derivative, an inner path moving a row by `1/n`;
`MTABoundarySet` is a collateral transfer and every later balance.
A decision registers one counterfactual with its whole reach, ledger included, since at a kink two
partial ones score differently from their sum; only booked cash may be declared (`cash_events`), the
collateral chain replaying the ledger.

!!! note "Under a recompute node, a `gap` the simulation decided is an output"
    An untaped forward gives `gap` no graph, so the node returns it and the coefficient arrives as
    its cotangent. A decision on outer state keeps its own graph and registers outside: the TARF's
    knock-in and the autocall's coupon trigger are node outputs, the TARF's redemption, the
    barrier's crossing and the accumulator's latches are not. **Held by**
    `tests/test_recompute_equity_pricers.py::test_the_exposure_its_cashflows_and_the_cva_gradient_are_bit_identical_with_the_node_on`.

!!! warning "A decision whose solve amplifies is refused, not smoothed"
    Local-linear weights are signed, so a kernel admitting two neighbouring points far out cancels
    the first-order term by subtracting two large numbers; $\lVert w\rVert_1$ is that
    amplification, and past `pricing.BOUNDARY_MAX_AMPLIFICATION` (25) the weights go to zero.
    **Not gated:** no document makes the refusal fire.

Acceptance is AAD against a common-random-numbers bump ladder (`tests/crn_ladder.py`): a gate
passes when the ladder's spread is at most 10% of its median, since one that scatters with `h` is
differencing across a jump, and its flattest interior rung is within tolerance (`Ladder.agrees`):

| decision, objective | test | paths × batches | tolerance |
| --- | --- | --- | --- |
| continuous barrier, one-touch, no-touch; CVA | `test_barrier_bridge.py::test_aad_delta_matches_bump_and_reprice` | 4,096 × 1 | 2% |
| TARF knock-in; CVA | `test_boundary_tarf_events.py::test_the_knock_in_gradient_matches_bump_and_reprice_under_exposure` | 512 × 4 | 2% |
| TARF knock-in; base value | `test_boundary_tarf_events.py::test_the_knock_in_gradient_matches_bump_and_reprice_at_base_valuation` | 2 × 65,536 inner | 2% |
| discrete barrier latch; CVA | `test_boundary_pricer_events.py::test_discrete_barrier_latch_gradient_matches_bump_and_reprice` | 1,024 × 4 | 5% |
| the same under a CSA; CVA | `test_boundary_pricer_events.py::test_collateralised_barrier_latch_gradient_matches_bump_and_reprice` | 512 × 8 | 8% |
| discrete barrier latch; FVA | `test_boundary_pricer_events.py::test_fva_gradient_carries_the_boundary_term_too` | 1,024 × 4 | 5% |
| exposure kink; CVA gamma, vanna | `test_cva_gamma_kink.py::test_the_gamma_and_vanna_entries_land_on_crn_ladders_of_the_reported_delta` | 65,536 × 1 | 5%, 10% |

**The CVA Hessian.** Second-order AAD drops $\delta(V)\,V_\theta V_\theta^{\top}$ at the exposure's
$\max(V,0)$; `pricing.exposure_kink_term` adds $\tfrac12 K_\varepsilon(\bar V)(V-\bar V)^2$ under
the CVA's weights, zero in value and at first order, a normalised Gaussian kernel at Silverman's
width [@silverman1986] $\varepsilon=1.06\,\mathrm{sd}(V)\,N^{-1/5}$ per report row, $N$ the run's
paths. A row whose density at zero climbs 2× across $(2,1,\tfrac12,\tfrac14)\varepsilon$ is a point
mass and refuses. A book that registered a boundary correction is refused `Hessian: 'Yes'`, as a
base valuation is refused `Greeks: 'All'`: a first-order correction differentiated twice loses its
density-derivative term. The $O(\varepsilon^2)$ bias shows on volga, so it is a gamma-and-vanna term
([xVA](../theory/valuation_adjustments.md#sensitivities)). **Held by** `tests/test_cva_gamma_kink.py`,
`tests/test_base_valuation_gamma.py::test_a_deal_that_registered_a_boundary_correction_is_refused_by_name`.
**Not gated:** the point-mass refusal.

## Open

The correlation as a leaf and the recompute segment length are
[desk decisions 12 and 20](../limitations.md#decisions-waiting-on-the-desk). Limitations are under
[the xVA outer](../limitations.md#the-xva-outer-and-the-correlation),
[the pricers](../limitations.md#the-autocall-tarf-and-barrier-pricers) and
[the engine](../limitations.md#the-engine); Hessian-vector products through the recompute node are
[designed, not built](roadmap.md#designed-not-built).
