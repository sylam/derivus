# Calc Lifecycle

The internal object walk behind a calculation. Public entry points and output shape: [API Overview](../api_overview.md).

## Dispatch

`Context.run_job` branches on `Calculation['Object']` into `run_cmc` / `run_baseval` / `run_hedgemontecarlo` / `run_simm`. The first three pick the device and inject the runtime-derived keys only (`Run_Date`, plus `Time_grid` for the two MC modes); `run_simm` injects nothing and picks no device. Each then calls `construct_calculation` — `globals().get(calc_type)(config, **kwargs)`. The seed default is declared on the calculation class (`F('Random_Seed', 'Integer', default=5120)`) and applied by `torch.manual_seed` in the state constructor (see [RNG streams](#rng-substream-ordering-the-reproducibility-surface)). The four classes are `Credit_Monte_Carlo`, `Base_Revaluation`, `HedgeMonteCarlo` and `SIMM` (`calculation.py`).

## Compile phase 1 — `calculate_dependencies`

Discovers the factor universe, wires the dependency DAG, topologically orders it, splits stochastic vs static — see [Dependency System](dependency_system.md). Returns `dependent_factors` (factor → max date), `stochastic_factors` (process-factor → price-factor), `additional_factors` (implied), plus reset/settlement date sets.

!!! note "Invariant — `calculate_dependencies` is idempotent"
    Both halves read the loaded config and write nothing: a second call returns identical output and `params` stays pristine. An implied model needs no `Price Models` entry — `find_models` classifies it off the implied factor's own block, and the process constructor reads `Price Models` with `.get`. Injecting `Price Models[model] = None` dummies pollutes saved market data and makes `plan_hash` a function of whether a run had happened. `Config.validate` / `Config.factor_universe` call `discover_factors` alone: a want-list needs no model resolution.

## Compile phase 2 — `_build_factor_state`

- **Factor objects.** Stochastic → `construct_process(model.type, factor_obj, Price Models[...], implied_obj)`; static = `set(dependent_factors) - stochastic_factors.values()` → `construct_factor`. `all_factors` = stochastic + static + implied.
- **AAD leaves** (reverse-mode differentiation; [@capriotti2011]). Each stochastic factor's `current_value` (offset by `Tenor_Offset`) becomes a `torch.tensor(..., requires_grad=calc_grad)` in `stoch_var`; implied params in `implied_var`; static in `static_var`. Every mint goes through `Calculation.factor_leaf`, the single seam where a factor the library itself calibrated is offered still carrying the graph of the quotes it was solved from, as `leaf + (θ* − θ*.detach())` — worth zero forward; see [Quote Sensitivities](quote_sensitivities.md#the-attachment).
- **Implied-leaf dedupe.** `implied_leaves` makes a factor that is both a static dependent and a spot process's implied factor reuse the one tensor, so `backward()` sums both consumers into it.
- **Sizing.** `num_factors = sum(v.num_factors() for v in stoch_factors.values())` sizes the correlated random block.
- **precalculate, then calc_references.** First loop caches `_factor_precalc_args[key]`, sets `factor_key`, calls `precalculate(..., process_ofs[key], ...)` (which sets `z_offset`, `spot0`). Second loop's `calc_references` resolves cross-process links (`BasisLinkedSpotModel` sets `linked_key` from the name-prefix parent).

!!! warning "Invariant — implied-leaf dedupe"
    A factor that is both a static dependent and a spot process's implied factor must map to **one** tensor. A second leaf under the same scope name splits gradients between the pricer path (`t_Static_Buffer`) and the scenario path (`implied_tensor`), and desyncs a bump.

!!! warning "Invariant — precalculate before calc_references"
    `precalculate` sets `factor_key`, `z_offset`, `spot0` and caches `_factor_precalc_args`; only then can `calc_references` resolve links needing `all_factors`. Both loops iterate `stoch_factors` in topological order, which is also what makes publish-as-you-go safe.

## Compile phase 2a — correlation + cholesky + `process_ofs`

A step *inside* phase 2: `_build_factor_state` → `_init_shared_mem` → `get_cholesky_decomp`, the only place `process_ofs` is populated, ahead of the precalculate loop that reads it. It iterates `stoch_factors.items()` in topological order; per factor `value.correlation_name` returns `(corr_type, [sub_factor_tuples])`, `process_ofs.setdefault(key, len(correlation_factors))` records the row offset of this factor's random substream, and each sub-factor appends a `Factor(corr_type, key.name+sub)`. The symmetric matrix is built from `Correlations` (a pair nothing declares reads 0.0; a declared name no simulated factor answers to is named at INFO), repaired if needed, and `torch.linalg.cholesky`'d. The declared number is the correlation of the innovations for every process alike, and nothing is scaled for it; a multi-factor process declares one row per sub-factor (`PC1..PCn`, `F1, F2`, the LogVar2FJ outer's return row and its `.S`, `.L` and `.G`) and the framework hands it one normal per row per step ([the LogVar2FJ outer](#logvar2fj-outer)).

!!! warning "Invariant — `num_factors()` must equal `len(correlation_name[1])`"
    The first sizes the rows of the correlated random block; the second sizes the correlation matrix and the `process_ofs` stride. Wired together by convention only — no assert. A process whose two counts disagree silently misaligns every downstream `z_offset`, so each process reads another factor's substream. Wrong numbers, not an error.

!!! note "The correlation repair"
    The matrix is repaired while any eigenvalue is below `1e-8`, by `System Parameters.Correlations_Healing_Method`:

    | method | one pass |
    | --- | --- |
    | `Eigenvalue_Raising` (the default) | eigenvalues floored at `1e-4`, the result rescaled to a unit diagonal: $C\leftarrow D^{-1/2}\,Q\max(\Lambda,10^{-4})Q^{\top}D^{-1/2}$ |
    | `Alternating_Projections` | up to 100 alternations of that eigenvalue floor and a reset of the diagonal to one, each diagonal correction added into the next iterate, stopping when two results agree to a relative `1e-8` |

    The second follows Higham's alternating projections [@higham2002] without Dykstra's correction, so its result is a nearby correlation matrix, not provably the nearest. A repair logs one WARNING naming the matrix size; how far it moved the matrix is not reported. The matrix is not an AAD leaf (`requires_grad=False`), so a correlation sensitivity is a bump ([Roadmap](roadmap.md#decisions-waiting-on-the-desk), decision 12).

## Execute — the per-batch generate loop

`Credit_Monte_Carlo.execute` builds the `DealStructure` tree via `set_deal_structures` (`calc_dependencies` → `Factor_dep`, `calc_time_dependency` → `Time_dep`), then loops `Simulation_Batches`:

1. **`shared_mem.reset(num_factors, time_grid, antithetic)`** — draws pseudo-random `torch.randn(num_factors, n*T)`, correlates by the cholesky, reshapes to `(num_factors, T, n)` into `t_random_numbers`; resets cashflows; clears `t_Buffer`. `n` is `Batch_Size`, or under `Antithetic` `Batch_Size // 2` concatenated with its negation along the batch axis, so path `j` and path `j + Batch_Size/2` are a pair and the path count is `Batch_Size × Simulation_Batches` either way (an odd `Batch_Size` loses a column). The quasi-random streams some processes also read are listed [below](#rng-substream-ordering-the-reproducibility-surface).
2. **Publish-as-you-go generate** — `t_Scenario_Buffer[key] = value.generate(shared_mem)` in topological order, so a linked factor reads its parent's already-published path. Each process reads its substream via `t_random_numbers[self.z_offset, ...]`.
3. **`resolve_structure`** — walks the netting tree, calls `deal.calculate` → `generate` + `pricing.interpolate`, accumulates MTM, runs `post_process`; owns cashflow reset/save for accumulating sub-structures and the `FLIP`-prefix sign inversion.
4. The CollVA, initial-margin, FVA and CVA reductions follow, in that order, each specified on [xVA](../theory/valuation_adjustments.md). **Do-not-touch** ([Change Scope](conventions.md#change-scope)).

!!! note "Run controls — what to read on every run"
    Nothing below stops a run; each is counted or logged where it happens. `Stats['Deals loaded']` / `['Structs loaded']`, and `['Deals Skipped']` / `['Structs Skipped']`: a deal whose compile fails is dropped (or, under `Exclude_Deals_With_Missing_Market_Data: No`, kept at zero), and a pricing exception marks the deal a scalar 0 on every row, counted once (a framework fault raises instead). `Stats['Ignored']` counts `Ignore: True` nodes. A factor whose `Price Factors` block is missing is skipped with a WARNING, and one that lost its process is frozen at today's value with a WARNING or ERROR ([Dependency System](dependency_system.md#stochastic-vs-static-split-find_models)). A correlation repair logs a WARNING. `Stats['Device_Memory']` reports the device's peak, with a WARNING where it passed the device. A run that valued nothing refuses.

!!! warning "Invariant — `t_Buffer` is the memo table, `t_PreCalc` the per-calculation one"
    `t_Buffer` is cleared between batches (`reset`) and between inner forks; never carried across a random-number reset or a batch-size change. `t_PreCalc` is what a run reads unchanged every batch — a process's integral precalcs, a curve or skew read's gathered indices, a deal-static array through `Calculation_State.upload`, one copy per calculation keyed by its bytes — and `reset` leaves it alone; nothing carrying a graph goes in it.

!!! warning "Invariant — publish-as-you-go + the `(factor_key, kind)` convention"
    Every process publishes its own path to `t_Scenario_Buffer[key]` as `generate` returns, and any sufficient statistic under `(factor_key, kind)` (e.g. `(key,'regimes')`, `(key,'garch_log_h')`). The calc/solver never name a regime/belief/variance directly — they iterate the model-agnostic verbs, whose base implementations are inert no-ops.

## RNG-substream ordering (the reproducibility surface)

`process_ofs` is the row a factor reads from `t_random_numbers`; the iteration order is `stoch_factors` insertion order, which is the `topological_sort` output, which tie-breaks equal-depth factors by dict-insertion order (deal-walk order + `factor_fields` order + `dependant_fields` list order). A permutation preserves the correlation structure and moves realized draws bit-for-bit; stated in full on [Dependency System](dependency_system.md#rng-ordering).

A run draws from four streams:

| stream | generator | seeded by | consumers |
| --- | --- | --- | --- |
| the correlated normal block `t_random_numbers` | torch's default generator: Mersenne Twister on the host [@matsumoto1998], Philox on CUDA [@salmon2011] | `Random_Seed + job_id` once per worker; under `deterministic_batches`, `batch_seed(Random_Seed, b)` per global batch `b`, one SplitMix64 round [@steele2014] | every process's `generate`, through its `z_offset` |
| `CMC_State.quasi_rng` | torch's Sobol engine [@sobol1967] on Joe and Kuo's direction numbers [@joe2008], scrambled by a random linear matrix and a digital shift [@owen1998] | fixed: scramble seed `QUASI_SEED = 1234`, first point `QUASI_ANCHOR = 1024`, not `Random_Seed` | `MarkovHMMSpotModel`'s regime uniforms, `GARCHSpotModel`, `BasisLinkedSpotModel`, LogVar2FJ's extra bucket-knot mixers, and `reset_inner`'s normals under `Inner_Draws: sobol` |
| `CMC_State.inner_block` | the same Sobol points as 30-bit integers | fixed; a function of the strip's length and the inner path count | the one-step-survival pricers' inner paths when the outer batch exceeds 16 paths (pseudo-random uniforms otherwise) |
| per-segment `torch.Generator`s | one per calendar-anchored segment of the LogVar2FJ walk | an integer drawn off the default generator per batch, plus the segment's index | the LogVar2FJ outer process's and pricer's daily shocks, redrawn in a checkpoint's backward |

`quasi_rng` keeps its position across batches, and `reset()` does not rewind it; it is a second substream-assignment surface beside `process_ofs`. Under `deterministic_batches` (the `runparallel` dispatch only) batch `b` reads from point `QUASI_ANCHOR + b · span · sample_size` whichever worker runs it (`span` the number of engine-width chunks a wide draw takes), and a second draw of one `(dimension, sample_size)` inside a batch refuses by name. A sharded run is then bit-identical in the worker count on one device type; against the unsharded default it is a different draw of the same law — on `tests/test_multi_gpu.py`'s document at 512 × 4 paths, over 20 seeds the expected-exposure means differed by at most 2.89% (1.85 standard deviations of the difference), the base-date row exactly.

## Deal `Time_dep` / `Factor_dep` + pricing dispatch

- `Factor_dep` is the compiled factor-offset lookup a deal builds once in `calc_dependencies` (generic `get_*` layer → `calc_factor_code_chain` → `calc_factor_index`), stored verbatim as `DealDataType.Factor_dep` and consumed unchanged by `generate`. See [Resolver Layer](resolver_layer.md).
- `Time_dep` (`DealTimeDependencies`) precomputes interp indices/alphas against the mtm grid; `calculate` prices on the deal grid, `pricing.interpolate` gathers to the mtm grid and saves `Calc_res['Value']` (and `Calc_res['tensor']` when `shared.keep_tensor`).

### The schedule lifecycle {#the-schedule-lifecycle}

A `TensorSchedule` is a **dual**: `.np` for the index columns, checked in kernel-free numpy at compile time, and a device copy for the value arithmetic. `bind` separates the two halves in time and rides the deal walk — `utils.bind_schedules` wraps `calc_dependencies` inside `add_deal_to_structure` / `add_structure_to_structure`, so a schedule is reached wherever a deal filed it. Before `bind` the numpy half is authoritative and only compile-time edits write (`overwrite_rate`, `carry`, `compress_no_compounding`, an inserted principal exchange); after it the device copy is, `dual` is the one accessor, and an edit raises `ScheduleLifecycleError`. `derived` is the run-scoped home for anything a pricer builds off the copy (`pv_fixed_cashflows`' payment vector, a reset's known values); `bind` mints it with the copy, so a re-bind drops it.

Two deals compile outside a `DealStructure` and bind for themselves: `bootstrappers.BenchmarkInstruments` (last, once the quote overlay is on) and the HW2F calibration's `create_market_swaps`. The inner-MC fork windows `Time_dep` and shares `Factor_dep` by reference on the same `shared_mem`, so it prices off schedules the outer walk already bound.

!!! warning "Invariant — `derived` is run-scoped, `t_Buffer` is batch-scoped"
    A tensor derived from a schedule's device copy belongs in `derived`, never in `Factor_dep` (compile output consumed verbatim) and never in `t_Buffer`, cleared per batch. A schedule the walk did not reach raises on first touch and is `is_fatal_pricing_error`, so `Deal.calculate`'s guard cannot turn it into a scalar-0 mark.

## Grids {#grids}

A Monte Carlo run carries four date axes (`utils.TimeGrid`, built in `Credit_Monte_Carlo.update_time_grid`):

| grid | dates |
| --- | --- |
| `Time_Grid` | the grid text read from the base date. Items are space-separated, each an offset or `offset(step)`; an offset or step is one or more integers each followed by `D`, `W`, `M` or `Y` (`1y6m`). An item places the base date plus its offset and, with a step, every step after it until the next item's date; the default `0d 2d 1w(1w) 3m(1m) 2y(3m)` is 0 and 2 days, weekly to three months, monthly to two years, then quarterly. A grid ending in `)` repeats its last step to the book's last date; otherwise the grid stops at its last offset and deal dates past it are clipped. Dates are not business-day adjusted |
| MTM | the parsed grid plus every date after the base date a deal revalues on (a collateralised set adds its call, settlement and liquidation dates) |
| scenario | the MTM grid under `Dynamic_Scenario_Dates: Yes` (the default, and forced by `Calc_Scenarios: At_Percentile`); otherwise the parsed grid plus the first date past the last reset. A factor read between two scenario dates is interpolated linearly ([Roadmap](roadmap.md#the-engine)) |
| deal | each deal's own revaluation dates, priced there and interpolated onto the MTM grid (`pricing.interpolate`) |

The reported rows (`report_index`) are the MTM dates each top-level structure reports — every MTM date up to its last revaluation date, and for a collateralised set only those whose settlement and liquidation dates are on the grid. The xVA reductions run on those rows ([xVA](../theory/valuation_adjustments.md#notation-and-the-grid)).

## Valuation modes

**`Base_Revaluation`** is the degenerate lifecycle: one time point (`TimeGrid({base_date}, …)`), no stochastic factors, everything a static leaf — no cholesky, no generate loop. `resolve_structure` runs once; greeks via `pricing.greeks`. The compile-plus-single-eval reference for reconciliation.

**`HedgeMonteCarlo`** inherits the full CMC engine and diverges in what happens to the marks:

- **Own dependency assembly** (`update_factors`): merges deal-driven factors with the JSON `Scenario_Factors` list (factors no deal reaches, e.g. a basis consumed only by a composed spot), collapses per-factor tenors to a single horizon, caps the grid at the liability terminal, calls `_build_factor_state` directly.
- **Inner-MC shared state** (`_init_shared_mem`): `CMC_State_Inner` lets one `shared_mem` host outer mode (`reset()`, pseudo-random `(F,T,B)`) and inner mode (`reset_inner()`, Sobol `(F,T,B,B2)`); processes dispatch on `Z.ndim`.
- **Generate loop** adds: a print-seed pass first (`_publish_print_seeds` — calibrated t0 values are the batch's first-step state, a burn-in restart republishes per-path, and the keys are consumed state dropped after the batch generates); optional `Randomize_Initial_State` burn-in; `Observed_Scenario` path substitution + `reseed_from_path` (walk-forward replay); leafing the declared spot for base-delta AAD; snapshotting the outer `t_Scenario_Buffer` for on-demand forking. Marks are **harvested, not aggregated**: liability MTM via `resolve_hedge_structure` (post-process-free, so no per-batch GPU→CPU copy), tradables via `tensor_marks`.
- **Bundle + runtime**: `Bundle.from_batch` + `construct_hedge_runtime` + `run_hedge_execution`; under `solve_hedge` the bundle carries `inner_mc` / `inner_mc_grad` closures forking on demand from the cached outer buffer.
- **A solve is a stream**: one bundle per batch handed to a persistent solver — `StreamingSolve.warmup` on batch 1 (which locks the frame), `step` on each later batch, `finish` on a held-out final batch. Fork width follows `Batch_Size`. So `Simulation_Batches` is a stream length under `solve_hedge` and a path multiplier under `simulate_only` — the one genuinely different meaning between the two verbs. End-to-end gate: `gates/wf_smoke_gate.sh` (512 × 5 batches, seed 7), which reads a walk-forward archive the repository does not carry, so a clone cannot run it.

!!! warning "Invariant — a frozen eval is the stream of length one"
    `DiffV2_Load_Value_Fn` means EVALUATION: nothing to fit, so `Simulation_Batches` must be 1 and that batch is both the warmup bundle and the held-out world. Two defences: the contract refuses `N > 1` with a checkpoint loaded, and `step` refuses to sweep a loaded net. `finish` skips its re-bind/re-fork when handed the bundle it is already bound to — without that the Sobol stream advances and the verdict moves.

!!! warning "Invariant — `Z.ndim` dispatch (outer vs inner MC)"
    `generate()` must handle outer (`Z.ndim==2`, `(T,B)`) and inner (`Z.ndim==3`, `(T,B,B2)`), with the per-outer-path initial state broadcast on the **middle** axis in inner mode. One process instance serves both loops.

!!! warning "Invariant — inner-MC batch state + fail-loud pricing"
    `shared_mem.simulation_batch` and `shared_mem.fillvalue` must track the current flat batch during a fork (set before `reset_inner` and before the pricing pass) and be restored to `B_outer` in a `finally` — otherwise a mid-fork raise leaves the state flat-sized and the *next* chunk fails on shapes instead of the real cause. `fillvalue` is frozen at construction and used as the empty-cat fallback in energy-leg / cash-settle code. Inner-MC pricing must fail loud rather than let `Deal.calculate`'s guard swallow a failure into a scalar-0 mark, which inside a fork corrupts the solver's training labels. Both halves are checked: the liability on its flat shape, and every tradable still live in the fork's dependency list on having produced a `tensor_marks` entry (a missing one is indistinguishable from an expired contract, which the solver's `live` mask retires).

!!! warning "Invariant — `keep_tensor` gates the hedge tradable series"
    `keep_tensor` governs whether `pricing.interpolate` stores `Calc_res['tensor']`. The hedge path passes `keep_tensor=True` unconditionally in `HedgeMonteCarlo._init_shared_mem` and harvests those via `tensor_marks`, independent of the `Keep_Tensor` JSON field, which is `Credit_Monte_Carlo`'s alone. Removing that store breaks the hedge bundle's tradable series with no error — only missing marks.

## Inner-MC subsystem

`_run_inner_mc_at_t` forks the simulator from each outer-path state at outer step `t`: truncates the grid (`TimeGrid.truncate_to`), windows every deal's `Time_dep` to `{t,t+1}` (`restricted`), and runs ONE pass at `Batch_Size x Inner_Sub_Batch` flat samples — no partition, so peak memory is a function of those two JSON fields and an over-wide config raises CUDA OOM naming the fork. The pass:

1. `reset_inner` (Sobol by default; `Inner_Draws: random` draws iid normals).
2. Per-process `precalculate` from `outer_buf[key][t]` — a grad leaf under `with_grad`, which also backs the print-seed state so the conditioning rides the AAD tape.
3. **Every seed before any generate** — `inner_fork_seed` off the detached outer buffer, `print_seed` off the leaf-backed day-`t` snapshot — because a seed may condition a topologically-upstream consumer (the fixing bridge conditions its parent's first step).
4. Generate, then `reseed_inner_state`.
5. **Publish every path series the fork produced as a `ScenarioSource`**: the outer-realized past at `B_outer`, then the forked rows flattened `(B,B2)→B*B2`, for one real pricing pass on restricted `DealStructure`s.

*Every series*, not every factor: a process's own `(key, kind)` publication (`BasisLinkedSpotModel`'s `basis_mu`) is read through the same `calc_time_grid_spot_rate` seam as a factor and needs the same logical grid. The seeds a fork publishes (`<kind>_inner`) are excluded by that same rule, because the outer path does not carry them. Publishing the factors alone left such a series at the fork's own two rows while the pricer asked for outer row `t` — not a wrong number but an unrunnable configuration. **OPEN:** no test holds this; the one that did was removed and nothing replaces it. The loop is model-agnostic via [the process protocol](dependency_system.md#the-process-protocol).

The LogVar2FJ outer process publishes:

| key | what |
| --- | --- |
| `(key, 'lv_ell')`, `(key, 'lv_s')` | its revealed log-variance state, `(T, 1, B)`, detached, as `generate` returns |
| `(key, 'lv_days')` | the scenario grid in days from the base date, how a PRICER row finds the node it inherits its state from |
| `(key, 'ell0_inner')`, `(key, 's0_inner')` | the per-path pair `inner_fork_seed` hands the fork |

A seed key may be read by a PRICER rather than by a process's `generate`. `pricing.LogVar2FJKit.carried` reads the fork's `(ell0_inner, s0_inner)` where the fork published one and the outer path's `(lv_ell, lv_s)` at the row's own node otherwise; where no LogVar2FJ process ran it finds nothing and the row re-seeds at `(L*(t_row), 0)`, which is why every document that does not name the process is bit-identical.

### The LogVar2FJ outer {#logvar2fj-outer}

`LogVar2FJImpliedSpotModel` is the xVA scenario generator for an `EquityPrice` or `FxRate` under LogVar2FJ (the model: [theory](../theory/stochastic_volatility.md#one-walk-three-readers)). What a developer needs from it:

- **One walk.** It is an implied process reading the calibrated `LogVar2FJModelParameters` factor through `implied_tensor`, so a CVA sensitivity reaches the one leaf the pricer reads; and its `draws` and `residual` ARE `pricing.LogVar2FJKit`'s, bound on the class, so outer, fork and pricer cannot disagree. Do not give it a draw or a root of its own.
- **Four rows per name.** `correlation_name` is `('LognormalDiffusionProcess', [(), ('S',), ('L',), ('G',)])`: the return's Gaussian given the mixer, the fast and slow log-variance shocks, and the mixer through its quantile `G = F_IG⁻¹(Φ(Z_G))`, taken in double. Each variance factor steps node to node by its exact OU transition driven by its framework normal, the days inside the interval bridged between the endpoints; whole trading days plus one shorter remainder step span each interval.
- **The day's shocks are a function of the day.** One `torch.Generator` per calendar-anchored segment of `LogVar2FJKit.CHECKPOINT_STEPS` trading days draws them; they are never stored and are redrawn in the checkpoint's recompute, so where the scenario nodes fall moves no draw. The antithetic half takes `-eta`. `Checkpoint_Outer_Walk` (a `CreditMonteCarlo` field, default `Yes`) puts the walk under the checkpoint.
- **No replay.** The variance carries its own shocks, so `(ell, s)` is not a function of realised returns and `reseed_from_path` refuses by name.

---

!!! note "Four objects, one query: rows route by block, tenors route by segment"
    The curve read splits into a query, logical scenario storage and one physical interpolation. Nothing holds two of those jobs.

    - `CurveTensor` — query coordinates: scenario ROWS (`index`, `index_next`, `alpha`), never a flattened `row * n_tenors` offset, because a tenor segment's stride is its own.
    - `ScenarioBlock` / `ScenarioSource` — logical storage: one physical tensor plus `first_row` (where it starts in the logical grid) and `batch_index` (which of ITS columns supplies each logical column). A fork publishes two blocks; ordinary generation publishes a bare tensor and no source.
    - `UnroutedInterpolation` — the base under both leaves: one physical grid answers every row, so there is nothing to route, and a read is blended in time and scaled once.
    - `Interpolation` — one physical tensor and what its kind derives from it. Knows nothing about blocks or logical rows, and flattens rows against its OWN stride. Base valuation, credit MC and the outer hedge loop build only this.
    - `SegmentedInterpolation` — a SIBLING of it, not a subclass: composes leaves over the TENOR axis for a `Near_Interpolation` curve, the near leg the block's own and the far one whatever `Price Factor Interpolation` resolves for that curve.
    - `RoutedInterpolation` — composes strategies over the SCENARIO axis for a fork.

    `build_interpolation` is the single recursive constructor: bare tensor + kind → leaf; bare tensor + segment list → segmented; `ScenarioSource` + either → routed, whose per-block children it builds by calling itself. A segmented curve inside a fork needs no special case — the two compositions are orthogonal.

    **Why a fork publishes blocks.** Every realized-past row is identical across the inner draws, so joining them writes the past out `Inner_Sub_Batch` times and drags a same-shaped slab of Hermite coefficients with it. Measured on a walk-forward hedging book outside the repository (GARCH outer, seed 7) at 1,280 × 64: **6.33 GiB joined against 1.09 GiB blocked**; `peak_alloc = 0.057 GiB + 13.23 kB · B_flat`, against 80.62 kB per flat sample joined. Each block interpolates at its OWN width and `ScenarioBlock.project` takes the RESULT up to the logical width, never the stored tensor.

    **Order is load-bearing.** A read is raw (`read_at`), then blended over time, then `combine`d (RT scaling, segmented tenor select). `combine` and `project` are linear, so they commute with the blend — which is what makes the routed path bitwise identical to an unrouted one. A time-interpolated read reaches `index + 1`, so a row just below a cut reads ACROSS it and names two blocks: `route` classifies on where a read ENDS.

    **A source is write-once.** Built after every process's `generate` has published, and nothing writes into `t_Scenario_Buffer` afterwards. It answers only `shape` / `new` / the RT tenor rescale, so a late write fails loud rather than materializing the grid it exists to avoid.

    **A graph-free curve derives at the read.** `Interpolation.hermite_at` gathers the values at `j - 1` to `j + 2` and runs `hermite_interpolation_tensor`'s operations on them in its order, the end expressions selected where `j` is an end, and an RT kind's tenor is folded in at the gather - so the read is the held pair's to the bit, through device indices and no sync. Held, the pair and the folded copy were half of a credit Monte Carlo batch: on a 58-leg netting set, peak 3.27 GB to 1.71 at 1,024 paths and 12.93 to 6.69 at 4,096. A curve carrying a graph holds both still, because a derivative sums every read into the held pair before the Hermite map, and a pair derived per read would sum them after - a last-bit move in every Greek.

## The inner draws are one canonical block, and the strip streams {#inner-block}

**One block.** Every inner Monte Carlo in a run reads the same quasi-random points: `CMC_State.inner_block(rows, sims)` is a pure function of a strip's length and the inner path count - scrambled Sobol points at a fixed anchor, 30-bit integers made on the device, their antithetic complement taken on the integer before conversion so both tails keep their bits in the job's dtype, and a strip longer than a chunk of 64 rows reads more DIMENSIONS of the one point set, never more points of it, so each inner path is one point in as many dimensions as the strip has fixings. The block is prefix-stable - the first K rows of a longer strip are the K-row block - and broadcast over the outer batch: every product, every outer path, every batch and every worker read it. So a trade's value does not depend on what is booked before it in the job, an inner-MC book shards deterministically with nothing to anchor, and nothing is kept in double or memoized for the run. The outer scenario streams are untouched and keep their anchoring. The inner error is common to the outer paths and is reduced by `MCMC_Simulations` alone - which is what the memory below buys. On the bias and variance of a nested estimator see Gordy and Juneja [@gordy2010].

**The strip streams.** The four one-step-survival pricers (barrier, accumulator, target redemption forward, autocall; [@glasserman2001; @alm2013]) walk their fixings as pure steps through one driver, `pricing.oss_stream`: under a gradient, four fixings are one checkpoint, so a row's tape is its state at the chunk edges and one chunk's intermediates; otherwise it is the plain loop. The alive branch, the pending payment and the autocall's counterfactual, which every consumer reads detached, are built under no gradient. Measured at 512 outer by 8,192 inner paths (RTX 3090): a value run that peaked near 28 GB - and spilled over the bus - peaks at 0.3 to 0.9 GB at 45 to 79% of the card busy, the LogVar2FJ autocall runs where it could not, and a gradient that left the card peaks at 1.8 to 9.0 GB at 58 to 85%, a fifth more wall for the replay.

## Recomputing the inner Monte Carlo {#recompute-inner-mc}

An MC-priced deal builds one autograd graph per pricing, and the terminal `backward()` holds every one at once — every fixing of every reporting row of every deal — while the simulation that built them is cheap. `Recompute_Inner_MC` (a `Calculation` field, `No` by default, on `BaseValuation` and `CreditMonteCarlo`) trades the tape for a second forward pass, checkpointing in the sense of Griewank and Walther [@griewank2000]: `pricing.InnerMCRecompute` runs its forward under `no_grad` leaving nothing but its inputs, and its backward re-runs **the same callable** under `enable_grad`, contracting the cotangent through one graph that dies immediately. Peak is one inner graph rather than all of them.

Four adopters — `pv_MC_Tarf`, `pv_MC_AutoCallSwap`, `pv_MC_Accumulator`, `pv_discrete_barrier_option` — reach it through `InnerMCRecompute.run(shared, simulate, *theta)`: the node when the switch is on, the callable when off, RNG position taken at the call site. No per-pricer flag — what a run can afford to tape is a property of the engine and the machine, not the trade. `run` uses `cls.apply`, so a gate that subclasses the node gets its mutation.

The adopter's shape is the rest of the contract:

- `simulate(*bound)(*theta)` — leading args are the block's SHAPE, bound per block with `partial` (numpy, ints, flags); trailing ones are every tensor that can carry a graph.
- It returns a **tuple**: element 0 is the block's marks, the rest by-products the caller performs once, each with the plain-Python row index that places it. Non-tensors pass through with a `None` cotangent; a pricer with no by-products returns `(mtm,)`.
- Every graph-carrying by-product is a **top-level** element. A tensor nested in a returned list is not an output (`requires_grad` False, no `grad_fn`), so its half of a correction is silently zero.

**The counter is the storage.** What is saved is where the plain generator stood (`Calculation_State.rng_position`), never what it produced, and the live position is put back after the replay. The inner Sobol rows have no position at all: they are the canonical inner block below, a function of their shape, so a replay reads the same rows by construction.

Three ways to be silently wrong:

- **One function, called twice** — not a fast forward beside a differentiable copy. Two spellings agree until one is edited, and the failure is a wrong gradient beside a right price.
- **Pure in everything but its return.** A side effect inside the simulation fires a second time in the backward, so the pricer RETURNS those and the caller performs them once off the forward's result.
- **Its inputs are its whole theta surface.** Autograd returns a gradient only for a tensor passed to `apply`; anything read out of a closure is differentiated as a constant. All four pricers build their vol strip at the call site (`forward_vol_strip` / `forward_vol_rate`), the autocall hoists its floating leg and past equity fixings, and all four pass in the spot model's scalars — the hoist that a closure read would silently differentiate as a constant, in one factor, on a fixture that has that factor.

!!! warning "Invariant — settle outside the callable; the cost is a COLLATERALISED gradient, not cashflows"
    A cashflow settled inside is settled twice, and the reported cashflows cannot see it — the replay runs in `backward()` while `save_cashflows` runs in the forward, so every reported cashflow is bit-identical either way. What moves is the cashflow's GRAPH: booked under the node's `no_grad` forward it reaches `t_Cashflows` carrying nothing, so a collateralised exposure reading that ledger through `C_ts_te` loses **8.7% of the CVA gradient** (`tests/test_recompute_equity_pricers.py`'s autocall, 256 paths, 64 inner, seed 1: max |Δ| 7.59e-05 on a gradient of 8.68e-04).

!!! warning "Invariant — a FIRST-ORDER node, and ask `torch.is_grad_enabled()` BEFORE `enable_grad`"
    The replay is rooted at *detached* copies of the saved inputs, which stops the inner `autograd.grad` walking back into the outer graph and double-counting — so a second derivative through it is severed and comes back **zero**, a Hessian that looks like a Hessian (`Greeks: 'All'`, TARF fixture: −1.74e6 / −4.03e5 / −1.67e4 taped, all three zero recomputed). `backward` therefore refuses `create_graph`, naming the switch; keeping the inputs attached instead breaks the first derivative outright. The grad-mode flag is how a node learns it is being differentiated twice, and asked from *inside* the `enable_grad` block it always answers True — which does not raise, it retains every recomputed graph for the whole reverse sweep: 5.0 GiB taped against 7.4 GiB recomputed, on a forward that peaked at 0.6.

Measured by a harness kept outside the repository (one 24 GiB RTX 3090, float64, CVA gradient on; CVA and gradient bit-identical throughout): peak falls 1.5–6.2× across TARF, autocall and barrier, and at 24 fixings × 512 scenarios × 128 inner the taped path **cannot price the shape on a 24 GiB device at all** while the node does it in 3.86 s at 8.4 GiB. The second forward pass costs 2–41% only on sub-0.2 s runs where neither path is near the device; above that the node is level or 3–5× faster. The whole-run drop (3–4×) is smaller than the forward-pass drop (8.5×) because the backward holds one recomputed block graph live at a time and the boundary registration's per-inner-path `jumps` are retained as before.

!!! warning "A measurement taken beside an out-of-memory run is a measurement of that run"
    `torch.cuda.reset_peak_memory_stats` takes the CURRENT allocation as its floor, and a boundary registration holds `shared → boundary_sets → gap → node → simulate → shared`, a cycle refcounting cannot break. Measured: a 6-coupon autocall read a peak of 20 105 MiB, which was the graduation row above it that had just gone out of memory. `gc.collect()` before the reset, one adopter's shapes per run.

## Boundary corrections — the sensitivity subsystem

Some deals take a decision on **simulated state**: a barrier crosses, an autocall triggers, a swaption exercises, a collateral transfer clears its minimum. The value jump is real, so the price is right; what ordinary AAD drops is the **flux** — as a factor moves, scenarios cross the trigger, and the indicator recording that crossing has zero derivative almost everywhere. Correct number, wrong greek, and no price gate can detect it.

`pricing.stochastic_boundary_correction(gap, jump, bandwidth)` injects the missing term. It is worth **exactly zero in the forward pass** (`gap - gap.detach()` is numerically zero with derivative one), so adding it to the scalar handed to `backward()` reports an unchanged number and still propagates through `gap` to every factor at once. No bump, no second valuation, cost independent of factor count.

**The estimator.** For a decision $\mathbb 1\{G>0\}$ whose firing changes the objective by $J$ (fired minus not fired), the derivative ordinary AAD drops is the surface term

$$f_G(0)\;\mathbb E\Big[J\,\frac{\partial G}{\partial\theta}\;\Big|\;G=0\Big]$$

([@lyuu2011]; [@glasserman2003], ch. 7). `pricing.boundary_weights` estimates its two factors separately from a registration's $n$ gaps $g_k$: with width $h=$ `Boundary_AAD_Bandwidth` $\times\ \mathrm{sd}(g)$, $K_k=e^{-g_k^2/2h^2}$ and $s_r=\sum_k K_k g_k^r$,

$$\hat f_G(0)=\frac{s_0}{n\sqrt{2\pi}\,h},\qquad w_k=\frac{K_k\,(s_2-g_k s_1)}{s_2s_0-s_1^2},$$

a Gaussian-kernel density and local-linear regression weights at zero [@fan2018], which sum to one so the conditional expectation is $\sum_k w_kJ_k\,\partial g_k/\partial\theta$ and holds still over a range of $h$ where a local-constant fit carries an $O(h)$ bias. The term added is $\sum_k (g_k-\bar g_k)\,\hat f_G(0)\,w_kJ_k$, $\bar g$ the detached gap and the coefficient detached. One gap (a base valuation's single scenario) has no spread: zero width, zero correction.

| parameter | default | declared on | meaning |
| --- | --- | --- | --- |
| `Boundary_AAD_Bandwidth` | 0.01 | `CreditMonteCarlo`, `BaseValuation` | $h$ in standard deviations of the gaps |
| `Boundary_AAD_Window_Touch` | `Yes` | `CreditMonteCarlo` | registers the partial-time barrier's window-touch decision; `No` is the unregistered estimator ([Roadmap](roadmap.md#the-autocall-tarf-and-barrier-pricers)) |
| `pricing.BOUNDARY_MAX_AMPLIFICATION` | 25 | a constant | the largest $\sum_k\lvert w_k\rvert$ accepted; above it the decision contributes nothing. Over 2,685 solves in 82 runs every reading carrying density lay in 1.00 to 8.06 but one, the decision that broke a barrier gradient, at 99.9 |
| `Branch_And_Weight` | `Yes` | `BaseValuation` | prices the four fixing-observed knockouts with the smooth estimator, which registers nothing |

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

One decision registers **one** counterfactual carrying its whole reach, ledger included. `cash_events` declares a payment as `(mtm_row, decision, if_fired, if_not, booked)` and one law scores every family: the payment is made iff no decision up to its own fired, and pays according to that decision's state — so a trigger's coupon declares `(amount, 0)` at its own decision, a streamed pricer's fixing `(0, amount)` at the last decision that can kill it, and forcing a decision ON kills every payment gated at or after it while forcing it OFF pays them on the prefix that excludes it.

`objective_jumps` derives those rows per decision, summing payments that share a row before declaring it (`cash_to_C` relu-splits received from paid, and the split of a sum is not the sum of the splits); `net_from_gross` folds the list into the settlement-risk windows. Splitting one decision across two registrations is exact only while the objective responds linearly — a collateralised net sits at the relu kink, where two partial counterfactuals score differently from the counterfactual of the sum.

!!! note "Under a recompute node, a `gap` is an OUTPUT — when the simulation is what decided it"
    `stochastic_boundary_correction` needs `gap` to carry a graph and an untaped forward has none (see [Recomputing the inner Monte Carlo](#recompute-inner-mc)). The registration SPLITS: the gap's value is computed under `no_grad` for the set to report, the node connects it, and the correction's coefficient — assembled at the objective exactly as before — arrives as that output's cotangent, so the graph-carrying half is built during the backward. What reaches `backward()` differs; what is reported does not.

    The converse is the rule, and it is about **where the graph lives**: a decision taken on OUTER state keeps its own graph whatever the node does to the inner pass, so its registration stays outside. Measured per adopter by dropping the cotangent — the TARF's knock-in (under GBM only where a fixing interval straddles a scenario node; one that is a single simulated step is integrated by the conditional-p mixture and registers nothing) and the autocall's coupon trigger are node outputs, where dropping is bit-identical to removing the correction outright; the TARF's redemption latch and the barrier's crossing latch are built outside, where dropping every cotangent reproduces the corrected gradient bit for bit while suppressing the correction moves it 0.30%. `pv_MC_Accumulator` reads the same way at 2.52% of a CVA gradient, and what places it is the registration rather than the mutant: its gaps are the outer scenario's own observed fixings and `block_alive` enters `untriggered` DETACHED, and of the node's five outputs only the marks' ever arrives with a cotangent, so the injection mutant drops nothing there. A base valuation of it registers nothing it can score — one scenario is one gap, which `boundary_weights` gives zero width.

No JSON switch turns it on: `shared_mem.boundary_aad = calc_greeks is not None`. Wanting sensitivities *is* the switch (the fields above only shape it), and registration is gated on it, so the cost is zero when greeks are off. The exception is the four fixing-observed knockouts, which register nothing under a base valuation because `Branch_And_Weight` — its default — swaps their estimator rather than correcting it; `Credit_Monte_Carlo` declares no such field and registers as before. Branch values register on the **pricer's** grid in the **pricer's** currency; `to_mtm` is the deal's own map (`pricing.deal_to_mtm_grid` — fx into reporting, then interpolate), because the deal grid, the MTM grid and the report grid are three different axes and interpolation inserts rows in the middle.

!!! warning "A decision whose solve amplifies is refused, not smoothed"
    `boundary_weights` fits a local **linear** kernel, so its weights are signed and sum to one — and a kernel admitting two neighbouring points a long way out cancels the first-order term by subtracting two enormous numbers. Measured on a spot-model barrier at 512 paths: two points 1.20 widths out and 0.021 apart, weighted +50.4 / −49.5, supplying 112% of the coefficient and the whole of a 73% gradient error. Since the weights sum to one, `||weights||_1` **is** that amplification, and `pricing.BOUNDARY_MAX_AMPLIFICATION` bounds it; past the bound the weights go to zero. **OPEN:** the test that proved the refusal *reachable* was removed, so reachability is unproven — `tests/test_boundary_pricer_events.py` records readings on the repaired guard and asserts nothing about the refusal firing.

Acceptance is AAD against a common-random-numbers bump ladder (`tests/crn_ladder.py`), which reports agreement **and flatness** separately. Agreement at one bump size proves nothing; a ladder that scatters with `h` is differencing across a jump. Flatness measures the *oracle's* spread only — a flat ladder beside an AAD that moves with the path count is not evidence of a residual.

A gate passes when the ladder's spread is at most 10% of its median and its flattest interior rung lies within the tolerance of the AAD (`Ladder.agrees`):

| decision | objective | test | paths × batches | relative rungs | tolerance |
| --- | --- | --- | --- | --- | --- |
| continuously monitored barrier, one-touch, no-touch | CVA | `tests/test_barrier_bridge.py` | 4,096 × 1 | 2e-4 to 2e-3 | 2% |
| TARF knock-in | CVA | `tests/test_boundary_tarf_events.py` | 512 × 4 | 3e-4 to 1e-2 | 2% |
| TARF knock-in | base valuation | same | one scenario, 2 × 65,536 inner | 3e-4 to 1e-2 | 2% |
| discrete barrier latch | CVA | `tests/test_boundary_pricer_events.py` | 1,024 × 4 | 2e-3 to 1e-2 | 5% |
| discrete barrier latch under a CSA, with its transfers | CVA | same | 512 × 8 | 2e-3 to 1e-2 | 8% |
| discrete barrier latch | FVA | same | 1,024 × 4 | 2e-3 to 1e-2 | 5% |
| the exposure kink, second order | CVA gamma, vanna | `tests/test_cva_gamma_kink.py` | 65,536 × 1 | 2e-3 to 5e-2 | 5%, 10% |

**The CVA Hessian** is a second-order relative of the same idea. Second-order AAD drops $\delta(V)\,V_\theta V_\theta^{\top}$ at the exposure's $\max(V,0)$; `pricing.exposure_kink_term` adds $\tfrac12 K_\varepsilon(\bar V)(V-\bar V)^2$ under the CVA's own weights, zero in value and at first order, with a normalised Gaussian kernel at Silverman's width [@silverman1986] $\varepsilon=1.06\,\mathrm{sd}(V)\,N^{-1/5}$ per report row, $N$ the run's paths. It refuses a row whose kernel density at zero climbs 2x or more across $(2,1,\tfrac12,\tfrac14)\varepsilon$, the signature of a point mass, and the run refuses `Hessian: 'Yes'` on any book that registered a boundary correction. Its declared bias is $O(\varepsilon^2)$, visible on volga, so it is a gamma-and-vanna term ([xVA](../theory/valuation_adjustments.md#sensitivities)).
