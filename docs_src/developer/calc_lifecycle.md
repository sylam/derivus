# Calc Lifecycle

The internal object walk behind a calculation. Public entry points and output shape: [API Overview](../api_overview.md).

## Dispatch

`Context.run_job` branches on `Calculation['Object']` into `run_cmc` / `run_baseval` / `run_hedgemontecarlo`. Each picks the device, injects the runtime-derived keys only (`Run_Date`, plus `Time_grid` for the two MC modes), then calls `construct_calculation` — `globals().get(calc_type)(config, **kwargs)`. The seed default is declared on the calculation class (`F('Random_Seed', 'Integer', default=5120)`) and applied by `torch.manual_seed` in the state constructor. The three classes are `Credit_Monte_Carlo`, `Base_Revaluation` and `HedgeMonteCarlo` (`calculation.py`).

## Compile phase 1 — `calculate_dependencies`

Discovers the factor universe, wires the dependency DAG, topologically orders it, splits stochastic vs static — see [Dependency System](dependency_system.md). Returns `dependent_factors` (factor → max date), `stochastic_factors` (process-factor → price-factor), `additional_factors` (implied), plus reset/settlement date sets.

!!! note "Invariant — `calculate_dependencies` is idempotent"
    Both halves read the loaded config and write nothing: a second call returns identical output and `params` stays pristine. An implied model needs no `Price Models` entry — `find_models` classifies it off the implied factor's own block, and the process constructor reads `Price Models` with `.get`. Injecting `Price Models[model] = None` dummies pollutes saved market data and makes `plan_hash` a function of whether a run had happened. `Config.validate` / `Config.factor_universe` call `discover_factors` alone: a want-list needs no model resolution.

## Compile phase 2 — `_build_factor_state`

- **Factor objects.** Stochastic → `construct_process(model.type, factor_obj, Price Models[...], implied_obj)`; static = `set(dependent_factors) - stochastic_factors.values()` → `construct_factor`. `all_factors` = stochastic + static + implied.
- **AAD leaves.** Each stochastic factor's `current_value` (offset by `Tenor_Offset`) becomes a `torch.tensor(..., requires_grad=calc_grad)` in `stoch_var`; implied params in `implied_var`; static in `static_var`. Every mint goes through `Calculation.factor_leaf`, the single seam where a factor the library itself calibrated is offered still carrying the graph of the quotes it was solved from, as `leaf + (θ* − θ*.detach())` — worth zero forward; see [Quote Sensitivities](quote_sensitivities.md#the-attachment).
- **Implied-leaf dedupe.** `implied_leaves` makes a factor that is both a static dependent and a spot process's implied factor reuse the one tensor, so `backward()` sums both consumers into it.
- **Sizing.** `num_factors = sum(v.num_factors() for v in stoch_factors.values())` sizes the correlated random block.
- **precalculate, then calc_references.** First loop caches `_factor_precalc_args[key]`, sets `factor_key`, calls `precalculate(..., process_ofs[key], ...)` (which sets `z_offset`, `spot0`). Second loop's `calc_references` resolves cross-process links (`BasisLinkedSpotModel` sets `linked_key` from the name-prefix parent).

!!! warning "Invariant — implied-leaf dedupe"
    A factor that is both a static dependent and a spot process's implied factor must map to **one** tensor. A second leaf under the same scope name splits gradients between the pricer path (`t_Static_Buffer`) and the scenario path (`implied_tensor`), and desyncs a bump.

!!! warning "Invariant — precalculate before calc_references"
    `precalculate` sets `factor_key`, `z_offset`, `spot0` and caches `_factor_precalc_args`; only then can `calc_references` resolve links needing `all_factors`. Both loops iterate `stoch_factors` in topological order, which is also what makes publish-as-you-go safe.

## Compile phase 2a — correlation + cholesky + `process_ofs`

A step *inside* phase 2: `_build_factor_state` → `_init_shared_mem` → `get_cholesky_decomp`, the only place `process_ofs` is populated, ahead of the precalculate loop that reads it. It iterates `stoch_factors.items()` in topological order; per factor `value.correlation_name` returns `(corr_type, [sub_factor_tuples])`, `process_ofs.setdefault(key, len(correlation_factors))` records the row offset of this factor's random substream, and each sub-factor appends a `Factor(corr_type, key.name+sub)`. The symmetric matrix is built from `Correlations`, healed to PSD if needed, and `torch.linalg.cholesky`'d.

!!! warning "Invariant — `num_factors()` must equal `len(correlation_name[1])`"
    The first sizes the rows of the correlated random block; the second sizes the correlation matrix and the `process_ofs` stride. Wired together by convention only — no assert. A process whose two counts disagree silently misaligns every downstream `z_offset`, so each process reads another factor's substream. Wrong numbers, not an error.

## Execute — the per-batch generate loop

`Credit_Monte_Carlo.execute` builds the `DealStructure` tree via `set_deal_structures` (`calc_dependencies` → `Factor_dep`, `calc_time_dependency` → `Time_dep`), then loops `Simulation_Batches`:

1. **`shared_mem.reset(num_factors, time_grid, antithetic)`** — draws `torch.randn(num_factors, sample*T)`, correlates by the cholesky, reshapes to `(num_factors, T, B)` into `t_random_numbers`; resets cashflows; clears `t_Buffer`. Outer draws are pseudo-random.
2. **Publish-as-you-go generate** — `t_Scenario_Buffer[key] = value.generate(shared_mem)` in topological order, so a linked factor reads its parent's already-published path. Each process reads its substream via `t_random_numbers[self.z_offset, ...]`.
3. **`resolve_structure`** — walks the netting tree, calls `deal.calculate` → `generate` + `pricing.interpolate`, accumulates MTM, runs `post_process`; owns cashflow reset/save for accumulating sub-structures and the `FLIP`-prefix sign inversion.
4. CVA/FVA/CollVA/IM adjustments follow. **Do-not-touch** ([Change Scope](conventions.md#change-scope)).

!!! warning "Invariant — `t_Buffer` is the memo table"
    Cleared between batches (`reset`) and between inner forks; never carried across a random-number reset or a batch-size change.

!!! warning "Invariant — publish-as-you-go + the `(factor_key, kind)` convention"
    Every process publishes its own path to `t_Scenario_Buffer[key]` as `generate` returns, and any sufficient statistic under `(factor_key, kind)` (e.g. `(key,'regimes')`, `(key,'garch_log_h')`). The calc/solver never name a regime/belief/variance directly — they iterate the model-agnostic verbs, whose base implementations are inert no-ops.

## RNG-substream ordering (the reproducibility surface)

`process_ofs` is the row a factor reads from `t_random_numbers`; the iteration order is `stoch_factors` insertion order, which is the `topological_sort` output, which tie-breaks equal-depth factors by dict-insertion order (deal-walk order + `factor_fields` order + `dependant_fields` list order). A permutation preserves the correlation structure and moves realized draws bit-for-bit; stated in full on [Dependency System](dependency_system.md#rng-ordering). The HMM regime path draws from a separate Sobol `quasi_rng` stream whose batch counter advances across batches and is never reset by `reset()` — a second, parallel substream-assignment surface.

## Deal `Time_dep` / `Factor_dep` + pricing dispatch

- `Factor_dep` is the compiled factor-offset lookup a deal builds once in `calc_dependencies` (generic `get_*` layer → `calc_factor_code_chain` → `calc_factor_index`), stored verbatim as `DealDataType.Factor_dep` and consumed unchanged by `generate`. See [Resolver Layer](resolver_layer.md).
- `Time_dep` (`DealTimeDependencies`) precomputes interp indices/alphas against the mtm grid; `calculate` prices on the deal grid, `pricing.interpolate` gathers to the mtm grid and saves `Calc_res['Value']` (and `Calc_res['tensor']` when `shared.keep_tensor`).

### The schedule lifecycle {#the-schedule-lifecycle}

A `TensorSchedule` is a **dual**: `.np` for the index columns, checked in kernel-free numpy at compile time, and a device copy for the value arithmetic. `bind` separates the two halves in time and rides the deal walk — `utils.bind_schedules` wraps `calc_dependencies` inside `add_deal_to_structure` / `add_structure_to_structure`, so a schedule is reached wherever a deal filed it. Before `bind` the numpy half is authoritative and only compile-time edits write (`overwrite_rate`, `carry`, `compress_no_compounding`, an inserted principal exchange); after it the device copy is, `dual` is the one accessor, and an edit raises `ScheduleLifecycleError`. `derived` is the run-scoped home for anything a pricer builds off the copy (`pv_fixed_cashflows`' payment vector, a reset's known values); `bind` mints it with the copy, so a re-bind drops it.

Two deals compile outside a `DealStructure` and bind for themselves: `bootstrappers.BenchmarkInstruments` (last, once the quote overlay is on) and the HW2F calibration's `create_market_swaps`. The inner-MC fork windows `Time_dep` and shares `Factor_dep` by reference on the same `shared_mem`, so it prices off schedules the outer walk already bound.

!!! warning "Invariant — `derived` is run-scoped, `t_Buffer` is batch-scoped"
    A tensor derived from a schedule's device copy belongs in `derived`, never in `Factor_dep` (compile output consumed verbatim) and never in `t_Buffer`, cleared per batch. A schedule the walk did not reach raises on first touch and is `is_fatal_pricing_error`, so `Deal.calculate`'s guard cannot turn it into a scalar-0 mark.

## Valuation modes

**`Base_Revaluation`** is the degenerate lifecycle: one time point (`TimeGrid({base_date}, …)`), no stochastic factors, everything a static leaf — no cholesky, no generate loop. `resolve_structure` runs once; greeks via `pricing.greeks`. The compile-plus-single-eval reference for reconciliation.

**`HedgeMonteCarlo`** inherits the full CMC engine and diverges in what happens to the marks:

- **Own dependency assembly** (`update_factors`): merges deal-driven factors with the JSON `Scenario_Factors` list (factors no deal reaches, e.g. a basis consumed only by a composed spot), collapses per-factor tenors to a single horizon, caps the grid at the liability terminal, calls `_build_factor_state` directly.
- **Inner-MC shared state** (`_init_shared_mem`): `CMC_State_Inner` lets one `shared_mem` host outer mode (`reset()`, pseudo-random `(F,T,B)`) and inner mode (`reset_inner()`, Sobol `(F,T,B,B2)`); processes dispatch on `Z.ndim`.
- **Generate loop** adds: a print-seed pass first (`_publish_print_seeds` — calibrated t0 values are the batch's first-step state, a burn-in restart republishes per-path, and the keys are consumed state dropped after the batch generates); optional `Randomize_Initial_State` burn-in; `Observed_Scenario` path substitution + `reseed_from_path` (walk-forward replay); leafing the declared spot for base-delta AAD; snapshotting the outer `t_Scenario_Buffer` for on-demand forking. Marks are **harvested, not aggregated**: liability MTM via `resolve_hedge_structure` (post-process-free, so no per-batch GPU→CPU copy), tradables via `tensor_marks`.
- **Bundle + runtime**: `Bundle.from_batch` + `construct_hedge_runtime` + `run_hedge_execution`; under `solve_hedge` the bundle carries `inner_mc` / `inner_mc_grad` closures forking on demand from the cached outer buffer.
- **A solve is a stream**: one bundle per batch handed to a persistent solver — `StreamingSolve.warmup` on batch 1 (which locks the frame), `step` on each later batch, `finish` on a held-out final batch. Fork width follows `Batch_Size`. So `Simulation_Batches` is a stream length under `solve_hedge` and a path multiplier under `simulate_only` — the one genuinely different meaning between the two verbs. End-to-end gate: `gates/wf_smoke_gate.sh` (trade 202001, 512x5 batches, seed 7).

!!! warning "Invariant — a frozen eval is the stream of length one"
    `DiffV2_Load_Value_Fn` means EVALUATION: nothing to fit, so `Simulation_Batches` must be 1 and that batch is both the warmup bundle and the held-out world. Two defences: the contract refuses `N > 1` with a checkpoint loaded, and `step` refuses to sweep a loaded net. `finish` skips its re-bind/re-fork when handed the bundle it is already bound to — without that the Sobol stream advances and the verdict moves.

!!! warning "Invariant — `Z.ndim` dispatch (outer vs inner MC)"
    `generate()` must handle outer (`Z.ndim==2`, `(T,B)`) and inner (`Z.ndim==3`, `(T,B,B2)`), with the per-outer-path initial state broadcast on the **middle** axis in inner mode. One process instance serves both loops.

!!! warning "Invariant — inner-MC batch state + fail-loud pricing"
    `shared_mem.simulation_batch` and `shared_mem.fillvalue` must track the current flat batch during a fork (set before `reset_inner` and before the pricing pass) and be restored to `B_outer` in a `finally` — otherwise a mid-fork raise leaves the state flat-sized and the *next* chunk fails on shapes instead of the real cause. `fillvalue` is frozen at construction and used as the empty-cat fallback in energy-leg / cash-settle code. Inner-MC pricing must fail loud rather than let `Deal.calculate`'s guard swallow a failure into a scalar-0 mark, which inside a fork corrupts the solver's training labels. Both halves are checked: the liability on its flat shape, and every tradable still live in the fork's dependency list on having produced a `tensor_marks` entry (a missing one is indistinguishable from an expired contract, which the solver's `live` mask retires).

!!! warning "Invariant — `keep_tensor` gates the hedge tradable series"
    `keep_tensor` governs whether `pricing.interpolate` stores `Calc_res['tensor']`. The hedge path passes `keep_tensor=True` unconditionally in `HedgeMonteCarlo._init_shared_mem` and harvests those via `tensor_marks`, independent of the `Keep_Tensor` JSON field, which is `Credit_Monte_Carlo`'s alone. Removing that store breaks the hedge bundle's tradable series with no error — only missing marks.

## Inner-MC subsystem

`_run_inner_mc_at_t` forks the simulator from each outer-path state at outer step `t`: truncates the grid (`TimeGrid.truncate_to`), windows every deal's `Time_dep` to `{t,t+1}` (`copy_window`), and runs ONE pass at `Batch_Size x Inner_Sub_Batch` flat samples — no partition, so peak memory is a function of those two JSON fields and an over-wide config raises CUDA OOM naming the fork. The pass:

1. `reset_inner` (Sobol).
2. Per-process `precalculate` from `outer_buf[key][t]` — a grad leaf under `with_grad`, which also backs the print-seed state so the conditioning rides the AAD tape.
3. **Every seed before any generate** — `inner_fork_seed` off the detached outer buffer, `print_seed` off the leaf-backed day-`t` snapshot — because a seed may condition a topologically-upstream consumer (the fixing bridge conditions its parent's first step).
4. Generate, then `reseed_inner_state`.
5. **Publish every path series the fork produced as a `ScenarioSource`**: the outer-realized past at `B_outer`, then the forked rows flattened `(B,B2)→B*B2`, for one real pricing pass on restricted `DealStructure`s.

*Every series*, not every factor: a process's own `(key, kind)` publication (`BasisLinkedSpotModel`'s `basis_mu`) is read through the same `calc_time_grid_spot_rate` seam as a factor and needs the same logical grid. The seeds a fork publishes (`<kind>_inner`) are excluded by that same rule, because the outer path does not carry them. Publishing the factors alone left such a series at the fork's own two rows while the pricer asked for outer row `t` — not a wrong number but an unrunnable configuration. **OPEN:** its gate (`tests/test_fork_published_state.py`) went in the 2026-08-21 purge and nothing replaces it. The loop is model-agnostic via [the process protocol](dependency_system.md#the-process-protocol).

!!! note "Four objects, one query: rows route by block, tenors route by segment"
    The curve read splits into a query, logical scenario storage and one physical interpolation. Nothing holds two of those jobs.

    - `CurveTensor` — query coordinates: scenario ROWS (`index`, `index_next`, `alpha`), never a flattened `row * n_tenors` offset, because a tenor segment's stride is its own.
    - `ScenarioBlock` / `ScenarioSource` — logical storage: one physical tensor plus `first_row` (where it starts in the logical grid) and `batch_index` (which of ITS columns supplies each logical column). A fork publishes two blocks; ordinary generation publishes a bare tensor and no source.
    - `Interpolation` — one physical tensor and what its kind derives from it. Knows nothing about blocks or logical rows, and flattens rows against its OWN stride. Base valuation, credit MC and the outer hedge loop build only this.
    - `SegmentedInterpolation` — a SIBLING, not a subclass: composes leaves over the TENOR axis for a `Near_Interpolation` curve.
    - `RoutedInterpolation` — composes strategies over the SCENARIO axis for a fork.

    `build_interpolation` is the single recursive constructor: bare tensor + kind → leaf; bare tensor + segment list → segmented; `ScenarioSource` + either → routed, whose per-block children it builds by calling itself. A segmented curve inside a fork needs no special case — the two compositions are orthogonal.

    **Why a fork publishes blocks.** Every realized-past row is identical across the inner draws, so joining them writes the past out `Inner_Sub_Batch` times and drags a same-shaped slab of Hermite coefficients with it. Measured on the production walk-forward book (trade 202001, garch, seed 7) at 1280x64: **6.33 GiB joined against 1.09 GiB blocked**; `peak_alloc = 0.057 GiB + 13.23 kB · B_flat`, against 80.62 kB per flat sample joined. Each block interpolates at its OWN width and `ScenarioBlock.project` takes the RESULT up to the logical width, never the stored tensor.

    **Order is load-bearing.** A read is raw (`read_at`), then blended over time, then `combine`d (RT scaling, segmented tenor select). `combine` and `project` are linear, so they commute with the blend — which is what makes the routed path bitwise identical to an unrouted one. A time-interpolated read reaches `index + 1`, so a row just below a cut reads ACROSS it and names two blocks: `route` classifies on where a read ENDS.

    **A source is write-once.** Built after every process's `generate` has published, and nothing writes into `t_Scenario_Buffer` afterwards. It answers only `shape` / `new` / the RT tenor rescale, so a late write fails loud rather than materializing the grid it exists to avoid.

    **Hermite coefficients are built eagerly.** Blocks store the past at `B_outer`, so eager `g,c` costs single-digit MB: measured 179 s / 0.617 GiB eager against 193 s / 0.595 GiB deferred, the deferral's span bookkeeping adding a device sync per gather. The one shape it would help is a full-horizon fork, and a fork prices exactly `{t, t+1}`.

## Recomputing the inner Monte Carlo {#recompute-inner-mc}

An MC-priced deal builds one autograd graph per pricing, and the terminal `backward()` holds every one at once — every fixing of every reporting row of every deal — while the simulation that built them is cheap. `Recompute_Inner_MC` (a `Calculation` field, `No` by default, on `BaseValuation` and `CreditMonteCarlo`) trades the tape for a second forward pass: `pricing.InnerMCRecompute` runs its forward under `no_grad` leaving nothing but its inputs, and its backward re-runs **the same callable** under `enable_grad`, contracting the cotangent through one graph that dies immediately. Peak is one inner graph rather than all of them.

Four adopters — `pv_MC_Tarf`, `pv_MC_AutoCallSwap`, `pv_MC_Accumulator`, `pv_discrete_barrier_option` — reach it through `InnerMCRecompute.run(shared, simulate, *theta)`: the node when the switch is on, the callable when off, RNG position taken at the call site. No per-pricer flag — what a run can afford to tape is a property of the engine and the machine, not the trade. `run` uses `cls.apply`, so a gate that subclasses the node gets its mutation.

The adopter's shape is the rest of the contract:

- `simulate(*bound)(*theta)` — leading args are the block's SHAPE, bound per block with `partial` (numpy, ints, flags); trailing ones are every tensor that can carry a graph.
- It returns a **tuple**: element 0 is the block's marks, the rest by-products the caller performs once, each with the plain-Python row index that places it. Non-tensors pass through with a `None` cotangent; a pricer with no by-products returns `(mtm,)`.
- Every graph-carrying by-product is a **top-level** element. A tensor nested in a returned list is not an output (`requires_grad` False, no `grad_fn`), so its half of a correction is silently zero.

**The counter is the storage.** What is saved is where each random stream stood (`utils.rng_position`), never what it produced. Sobol draws are memoized per `(dimension, sample_size, batch)`, so rewinding hands the replay the same tensor; the regular generator (≤16 scenarios, and every Heston–Nandi unmonitored sub-step) has no memo, so its state is saved and restored. The live position is put back after the replay — a backward runs long after its forward, and a node leaving the streams where its replay ended would move the next node's draws.

Three ways to be silently wrong:

- **One function, called twice** — not a fast forward beside a differentiable copy. Two spellings agree until one is edited, and the failure is a wrong gradient beside a right price.
- **Pure in everything but its return.** A side effect inside the simulation fires a second time in the backward, so the pricer RETURNS those and the caller performs them once off the forward's result.
- **Its inputs are its whole theta surface.** Autograd returns a gradient only for a tensor passed to `apply`; anything read out of a closure is differentiated as a constant. All four pricers build their vol strip at the call site (`forward_vol_strip` / `forward_vol_rate`), the autocall hoists its floating leg and past equity fixings, and all four pass in the Heston–Nandi scalars — which is what `test_the_heston_nandi_theta_survives_the_node` exists for: reverting the barrier's hoist turns it red and nothing about the GBM fixtures notices.

!!! warning "Invariant — settle outside the callable; the cost is a COLLATERALISED gradient, not cashflows"
    A cashflow settled inside is settled twice, and the reported cashflows cannot see it — the replay runs in `backward()` while `save_cashflows` runs in the forward, so every reported cashflow is bit-identical either way. What moves is the cashflow's GRAPH: booked under the node's `no_grad` forward it reaches `t_Cashflows` carrying nothing, so a collateralised exposure reading that ledger through `C_ts_te` loses **8.7% of the CVA gradient** on the autocall fixture.

!!! warning "Invariant — a FIRST-ORDER node, and ask `torch.is_grad_enabled()` BEFORE `enable_grad`"
    The replay is rooted at *detached* copies of the saved inputs, which stops the inner `autograd.grad` walking back into the outer graph and double-counting — so a second derivative through it is severed and comes back **zero**, a Hessian that looks like a Hessian (`Greeks: 'All'`, TARF fixture: −1.74e6 / −4.03e5 / −1.67e4 taped, all three zero recomputed). `backward` therefore refuses `create_graph`, naming the switch; keeping the inputs attached instead breaks the first derivative outright. The grad-mode flag is how a node learns it is being differentiated twice, and asked from *inside* the `enable_grad` block it always answers True — which does not raise, it retains every recomputed graph for the whole reverse sweep: 5.0 GiB taped against 7.4 GiB recomputed, on a forward that peaked at 0.6.

Measured (`gates/recompute_inner_mc.py`, RTX 3090, float64, CVA gradient on; CVA and gradient bit-identical throughout): peak falls 1.5–6.2× across TARF, autocall and barrier, and at 24 fixings × 512 scenarios × 128 inner the taped path **cannot price the shape on a 24 GiB device at all** while the node does it in 3.86 s at 8.4 GiB. The second forward pass costs 2–41% only on sub-0.2 s runs where neither path is near the device; above that the node is level or 3–5× faster. The whole-run drop (3–4×) is smaller than the forward-pass drop (8.5×) because the backward holds one recomputed block graph live at a time and the boundary registration's per-inner-path `jumps` are retained as before.

!!! warning "A measurement taken beside an out-of-memory run is a measurement of that run"
    `torch.cuda.reset_peak_memory_stats` takes the CURRENT allocation as its floor, and a boundary registration holds `shared → boundary_sets → gap → node → simulate → shared`, a cycle refcounting cannot break. Measured: a 6-coupon autocall read a peak of 20 105 MiB, which was the graduation row above it that had just gone out of memory. `gc.collect()` before the reset, one adopter's shapes per run.

## Boundary corrections — the sensitivity subsystem

Some deals take a decision on **simulated state**: a barrier crosses, an autocall triggers, a swaption exercises, a collateral transfer clears its minimum. The value jump is real, so the price is right; what ordinary AAD drops is the **flux** — as a factor moves, scenarios cross the trigger, and the indicator recording that crossing has zero derivative almost everywhere. Correct number, wrong greek, and no price gate can detect it.

`pricing.stochastic_boundary_correction(gap, jump, bandwidth)` injects the missing term. It is worth **exactly zero in the forward pass** (`gap - gap.detach()` is numerically zero with derivative one), so adding it to the scalar handed to `backward()` reports an unchanged number and still propagates through `gap` to every factor at once. No bump, no second valuation, cost independent of factor count.

Its lifecycle straddles the phases above:

- **During pricing**, a deal appends a `utils.BoundarySet` to `shared.boundary_sets`. `gap` retains its graph and is signed so `gap > 0` means FIRED; everything else is detached, because a counterfactual yields a *coefficient*, not a differentiated quantity.
- **In `NettingCollateralSet.post_process`**, the set stamps its own gross→net chain onto the registrations beneath it. It must ride on the set: `boundary_sets` accumulates across every netting set, so one slot on `shared` would push an uncollateralised set's deal through a collateralised set's collateral scan.
- **At the objective** (CVA, FVA or base valuation), `pricing.boundary_correction` walks the registrations and each converts its own decisions into **portfolio** deltas.

!!! warning "Invariant — a counterfactual is scored on the PORTFOLIO"
    The objective applies to `resolve_structure`'s root sum over *every* netting set, so a counterfactual is `reported_mtm + this registration's change`, never a single set's own net. With one netting set the two coincide, which is what makes the error invisible to a single-set fixture — and it is worst where the machinery is aimed, because a collateralised set's post-collateral net sits at the relu kink by construction.

Two `BoundarySet` subclasses exist, plus `MTABoundarySet`, which shares only `objective_jumps`. They differ in **how far a decision reaches**, which is why one class with a mode flag is the wrong shape:

| subclass | reach | carries |
| --- | --- | --- |
| `LatchedBoundarySet` | every row from the decision onward (barrier, swaption, TARF, autocall, accumulator, extendable forward) | two whole-profile branches shared across decisions; per decision, optionally an own-row fired/survived override, the per-event payment facts (`cash_events`), and per row the survived-weighted pending head (`pending`) |
| `InnerBoundarySet` | a decision inside a pricer's inner MC | the objective's *derivative*, not a difference — one inner path moves the row by `1/n`, a jump the value never takes |
| `MTABoundarySet` (not a subclass) | a netting SET's collateral transfer clearing its minimum, and every later balance | the margin events, the realised balance path, and `replay` / `rescan` callables restarting the collateral recursion from a forced opening balance — nothing re-simulated, re-priced or bumped |

One decision registers **one** counterfactual carrying its whole reach, ledger included: forced ON the latch kills every later payment; forced OFF each later trigger pays iff no earlier decision fired. `objective_jumps` derives those rows per decision from the declared `cash_events`, and `net_from_gross` folds the list into the settlement-risk windows. Splitting one decision across two registrations is exact only while the objective responds linearly — a collateralised net sits at the relu kink, where two partial counterfactuals score differently from the counterfactual of the sum.

!!! note "Under a recompute node, a `gap` is an OUTPUT — when the simulation is what decided it"
    `stochastic_boundary_correction` needs `gap` to carry a graph and an untaped forward has none (see [Recomputing the inner Monte Carlo](#recompute-inner-mc)). The registration SPLITS: the gap's value is computed under `no_grad` for the set to report, the node connects it, and the correction's coefficient — assembled at the objective exactly as before — arrives as that output's cotangent, so the graph-carrying half is built during the backward. What reaches `backward()` differs; what is reported does not.

    The converse is the rule, and it is about **where the graph lives**: a decision taken on OUTER state keeps its own graph whatever the node does to the inner pass, so its registration stays outside. Measured per adopter by dropping the cotangent — the TARF's knock-in and the autocall's coupon trigger are node outputs, where dropping is bit-identical to removing the correction outright; the TARF's redemption latch and the barrier's crossing latch are built outside, where dropping every cotangent reproduces the corrected gradient bit for bit while suppressing the correction moves it 0.30%. **OPEN — `pv_MC_Accumulator` is unplaced**: its `LatchedBoundarySet` is assembled off `block_alive`, one of the node's own outputs, which puts it on the node-output side by construction, but the dropped-cotangent reading has not been taken.

There is **no JSON switch**: `shared_mem.boundary_aad = calc_greeks is not None`. Wanting sensitivities *is* the switch, and registration is gated on it, so the cost is zero when greeks are off. Branch values register on the **pricer's** grid in the **pricer's** currency; `to_mtm` is the deal's own map (`pricing.deal_to_mtm_grid` — fx into reporting, then interpolate), because the deal grid, the MTM grid and the report grid are three different axes and interpolation inserts rows in the middle.

!!! warning "A decision whose solve amplifies is refused, not smoothed"
    `boundary_weights` fits a local **linear** kernel, so its weights are signed and sum to one — and a kernel admitting two neighbouring points a long way out cancels the first-order term by subtracting two enormous numbers. Measured on the Heston-Nandi barrier at 512 paths: two points 1.20 widths out and 0.021 apart, weighted +50.4 / −49.5, supplying 112% of the coefficient and the whole of a 73% gradient error. Since the weights sum to one, `||weights||_1` **is** that amplification, and `pricing.BOUNDARY_MAX_AMPLIFICATION` bounds it; past the bound the weights go to zero. **OPEN:** the gate that proved the refusal *reachable* went in the 2026-08-21 purge, so reachability is unproven — `tests/test_boundary_pricer_events.py` records readings on the repaired guard and asserts nothing about the refusal firing.

Acceptance is AAD against a common-random-numbers bump ladder (`tests/crn_ladder.py`), which reports agreement **and flatness** separately. Agreement at one bump size proves nothing; a ladder that scatters with `h` is differencing across a jump. Flatness measures the *oracle's* spread only — a flat ladder beside an AAD that moves with the path count is not evidence of a residual.
