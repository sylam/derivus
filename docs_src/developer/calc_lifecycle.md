# Calc Lifecycle

The internal object walk behind a calculation. The public entry points and output shape are in [API Overview](../api_overview.md); this page traces what happens *between* `run_job` and the result.

## Dispatch

`Context.run_job` branches on `Calculation['Object']` into `run_cmc` / `run_baseval` / `run_hedgemontecarlo`, which set device/seed defaults then call `construct_calculation`. That constructor is `globals().get(calc_type)(config, **kwargs)` — a class-name `globals()` dispatch. The three classes are `Credit_Monte_Carlo`, `Base_Revaluation`, and `Hedge_Monte_Carlo`, all in `calculation.py`.

## Compile phase 1 — `calculate_dependencies`

Discovers the factor universe, wires the dependency DAG, topologically orders it, and splits stochastic vs static. This is a subsystem in itself — see [Dependency System](dependency_system.md). It returns `dependent_factors` (factor → max date), `stochastic_factors` (process-factor → price-factor), `additional_factors` (implied factors), plus reset/settlement date sets.

!!! danger "Invariant — `calculate_dependencies` is not idempotent"
    It **mutates** `self.params`: `find_models` injects `Price Models[model] = None` for implied models. A second call sees the injected dummies. Do not call it twice expecting identical output, and do not treat `Price Models` as pristine afterward. `find_models` is the only half that writes, which is why it is a separate method — anything that wants the factor universe without disturbing the job (`Context.validate`) calls `discover_factors` instead.

## Compile phase 2 — `_build_factor_state`

Constructs the factor objects, mints the AAD leaves, and builds the processes. Key steps:

- **Factor objects.** Stochastic factors → `construct_process(model.type, factor_obj, Price Models[...], implied_obj)`; static factors = `set(dependent_factors) - stochastic_factors.values()` → `construct_factor`. `all_factors` = stochastic + static + implied.
- **AAD leaves.** Each stochastic factor's `current_value` (offset by `Tenor_Offset`) becomes a `torch.tensor(..., requires_grad=calc_grad)` in `stoch_var`; implied params in `implied_var`; static factors in `static_var`. Every one of those mints goes through `Calculation.factor_leaf`, which is the single seam where a factor the library itself **calibrated** — a bootstrapped zero curve, or one named parameter of an implied model — is offered still carrying the graph of the quotes it was solved from, as `leaf + (θ* − θ*.detach())`. Worth exactly zero forward; see [Quote Sensitivities](quote_sensitivities.md#the-attachment).
- **Implied-leaf dedupe.** A factor that is *both* a static dependent (e.g. a `…ModelParameters` block pulled in by a conditional field) and a spot process's implied factor must share **one** tensor. `implied_leaves` is consulted so the static leaf reuses the single implied tensor; `value.backward()` then sums both consumers' sensitivities into that one leaf.
- **Sizing.** `self.num_factors = sum(v.num_factors() for v in stoch_factors.values())` sizes the correlated random block.
- **precalculate, then calc_references.** First loop: cache `_factor_precalc_args[key]`, set `value.factor_key = key`, call `value.precalculate(..., self.process_ofs[key], ...)` (which sets `z_offset`, `spot0`). Second loop: `value.calc_references(...)` resolves cross-process links (e.g. `BasisLinkedSpotModel.calc_references` sets `linked_key` from the name-prefix parent). Order is load-bearing — references need the keys precalc set.

!!! warning "Invariant — implied-leaf dedupe"
    A factor that is simultaneously a static dependent and a spot process's implied factor must map to **one** `torch` tensor. Minting a second leaf under the same scope name splits gradients between the pricer path (`t_Static_Buffer`) and the scenario path (`implied_tensor`) and desyncs a bump.

!!! warning "Invariant — precalculate before calc_references"
    `precalculate` sets `value.factor_key`, `z_offset`, `spot0` and caches `_factor_precalc_args`; only then can `calc_references` resolve links that need `all_factors`. Both loops iterate `stoch_factors` in the same topological order, which is also what makes publish-as-you-go safe.

## Compile phase 3 — correlation + cholesky + `process_ofs`

`get_cholesky_decomp` iterates `stoch_factors.items()` (topological order). For each, `value.correlation_name` returns `(corr_type, [sub_factor_tuples])`; `self.process_ofs.setdefault(key, len(correlation_factors))` records the **row offset** of this factor's random substream, and each sub-factor appends a `Factor(corr_type, key.name+sub)` to `correlation_factors`. The symmetric correlation matrix is built from `Correlations`, healed to PSD if needed, and `torch.linalg.cholesky`'d.

!!! warning "Invariant — `num_factors()` must equal `len(correlation_name[1])`"
    The first sizes the rows of the correlated random block; the second sizes the correlation matrix and the `process_ofs` stride. They are wired together only by convention — no assert. A new process whose two counts disagree silently misaligns every downstream process's `z_offset`, so each reads another factor's substream. Numbers come out wrong, not erroring.

## Execute — the per-batch generate loop

`Credit_Monte_Carlo.execute` builds the `DealStructure` tree via `set_deal_structures` (each deal's `calc_dependencies` produces `Factor_dep`, `calc_time_dependency` produces `Time_dep`), then loops `Simulation_Batches`:

1. **`shared_mem.reset(num_factors, time_grid, antithetic)`** — draws `torch.randn(num_factors, sample*T)`, correlates by the cholesky, reshapes to `(num_factors, T, B)` into `t_random_numbers`; resets cashflows; **clears `t_Buffer`**. Outer draws are pseudo-random.
2. **Publish-as-you-go generate** — `for key, value in stoch_factors.items(): t_Scenario_Buffer[key] = value.generate(shared_mem)`. Iteration is topological, so a linked factor reads its parent's already-published path from `t_Scenario_Buffer`. Each process reads its substream via `t_random_numbers[self.z_offset, ...]` where `z_offset = process_ofs`.
3. **`resolve_structure`** — walks the netting tree, calls `deal.calculate` → `generate` + `pricing.interpolate`, accumulates MTM, runs `post_process`; owns cashflow reset/save for accumulating sub-structures and the `FLIP`-prefix sign inversion.
4. CVA/FVA/CollVA/IM adjustments follow. This block is **do-not-touch** ([Change Scope](conventions.md#change-scope)).

!!! warning "Invariant — `t_Buffer` is the memo table"
    `t_Buffer` is the memoized eval cache. It must be cleared between batches (`reset`) and between inner forks; never carry it across a random-number reset or a batch-size change.

!!! warning "Invariant — publish-as-you-go + the `(factor_key, kind)` convention"
    Every process publishes its own path to `t_Scenario_Buffer[key]` as `generate` returns, and any sufficient statistic under `(factor_key, kind)` (e.g. `(key,'regimes')`, `(key,'garch_log_h')`). The calc/solver never name a regime/belief/variance directly — they iterate the model-agnostic verbs. Base implementations of those verbs are inert no-ops.

## RNG-substream ordering (the reproducibility surface)

`process_ofs` is the row a factor reads from `t_random_numbers`; the iteration order is `stoch_factors` insertion order, which is the `topological_sort` output, which tie-breaks equal-depth factors by dict-insertion order (deal-walk order + `factor_fields` order + `dependant_fields` list order). The correlation structure is preserved under a permutation (cholesky rows permute consistently), but the **realized draws change bit-for-bit**. This is the single most important reproducibility invariant — stated in full on [Dependency System](dependency_system.md#rng-ordering). The HMM regime path draws from a **separate Sobol `quasi_rng` stream** whose batch counter advances across batches and is never reset by `reset()` — a second, parallel substream-assignment surface.

## Deal `Time_dep` / `Factor_dep` + pricing dispatch

- `Factor_dep` is the compiled factor-offset lookup a deal builds once in `calc_dependencies`, via the generic `get_*` layer → `calc_factor_code_chain` → `calc_factor_index`. It is stored verbatim as `DealDataType.Factor_dep` and consumed unchanged by `generate`. See [Resolver Layer](resolver_layer.md).
- `Time_dep` (`DealTimeDependencies`) precomputes interp indices/alphas against the mtm grid; `calculate` prices on the deal grid, `pricing.interpolate` gathers to the mtm grid and saves `Calc_res['Value']` (and, when `shared.keep_tensor`, `Calc_res['tensor']`).

### The schedule lifecycle {#the-schedule-lifecycle}

A `TensorSchedule` is a **dual**: `.np` for the index columns, which are checked in kernel-free numpy at compile time, and a device copy for the value arithmetic. `bind` is what separates the two halves in TIME, and it rides the deal walk — `utils.bind_schedules` wraps `calc_dependencies` inside `add_deal_to_structure` / `add_structure_to_structure`, walking the compiled output so a schedule is reached wherever a deal filed it. Before it the numpy half is authoritative and only the compile-time edits write (`overwrite_rate`, `carry`, `compress_no_compounding`, an inserted principal exchange); after it the device copy is, `dual` is the one accessor over that one copy, and an edit raises `ScheduleLifecycleError`. `derived` is the run-scoped home for anything a pricer builds off the copy — `pv_fixed_cashflows`' payment vector, a reset's known values — and `bind` mints it with the copy, so a re-bind drops it.

Two deals compile outside a `DealStructure` and therefore bind for themselves: `bootstrappers.BenchmarkInstruments` (last, once the quote overlay is on) and the HW2F calibration's `create_market_swaps`. The inner-MC fork windows `Time_dep` and shares `Factor_dep` by reference on the same `shared_mem`, so it prices off schedules the outer walk already bound.

!!! warning "Invariant — `derived` is run-scoped, `t_Buffer` is batch-scoped"
    A tensor derived from a schedule's device copy belongs in `derived`, never in `Factor_dep` (whose contract is compile output consumed verbatim) and never in `t_Buffer`, which is cleared per batch — the payment vector is deliberately batch-independent and rebuilding it every batch is pure cost. A schedule the walk did not reach raises on first touch and is `is_fatal_pricing_error`, so `Deal.calculate`'s guard cannot turn it into a scalar-0 mark.

## Valuation modes

**`Base_Revaluation`** is the degenerate lifecycle: a single time point (`TimeGrid({base_date}, …)`), no stochastic factors, everything a static leaf — no cholesky, no generate loop. `resolve_structure` runs once; greeks via `pricing.greeks`. It is the compile-plus-single-eval reference for reconciliation.

**`Hedge_Monte_Carlo`** inherits the full CMC scenario engine and diverges in what happens to the marks:

- **Own dependency assembly** (`update_factors`): merges deal-driven factors with the JSON `Scenario_Factors` list (factors no deal reaches, e.g. a basis consumed only by a composed spot), collapses per-factor tenors to a single horizon, caps the time grid at the liability terminal, then calls `_build_factor_state` directly.
- **Inner-MC shared state** (`_init_shared_mem`): builds `CMC_State_Inner` so one `shared_mem` hosts outer mode (`reset()`, pseudo-random `(F,T,B)`) and inner mode (`reset_inner()`, Sobol quasi-random `(F,T,B,B2)`); processes dispatch on `Z.ndim`.
- **Generate loop** adds: optional `Randomize_Initial_State` burn-in; `Observed_Scenario` path substitution + `reseed_from_path` (walk-forward replay); leafing the declared spot (`requires_grad_(True)`) for base-delta AAD; snapshotting the full outer `t_Scenario_Buffer` for on-demand inner forking. Marks are **harvested, not aggregated**: liability MTM via `resolve_hedge_structure` (post-process-free — no per-batch GPU→CPU copy), tradable tensors via `tensor_marks`.
- **Bundle + runtime**: `Bundle.from_batch` + `construct_hedge_runtime` + `run_hedge_execution`; in `solve_hedge` mode the bundle carries `inner_mc` / `inner_mc_grad` closures that fork inner MC on demand from the cached outer buffer.
- **A solve is a stream**: a bundle per batch, handed to a persistent solver as it is built — `StreamingSolve.warmup` on batch 1 (which locks the frame), `step` on each later batch, `finish` on a held-out final batch. Fork width follows `Batch_Size`, not the whole simulation. `Simulation_Batches` is therefore a stream length under `solve_hedge` and a path multiplier under `simulate_only` — the one genuinely different meaning between the two verbs. The end-to-end reproduction gate is `gates/wf_smoke_gate.sh` (trade 202001, 512x5 batches, seed 7).

!!! warning "Invariant — a frozen eval is the stream of length one"
    `DiffV2_Load_Value_Fn` means EVALUATION: the policy is the file's and there is nothing to fit,
    so `Simulation_Batches` must be 1 and that single batch is both the warmup bundle and the
    held-out world (frozen nets saw none of it). Two defences hold it: the contract refuses `N > 1`
    with a checkpoint loaded, so there are no `step` batches to sweep, and `step` refuses to sweep
    a loaded net anyway. `finish` skips its re-bind/re-fork when handed the bundle it is already
    bound to — without that the Sobol stream advances and the verdict moves.

!!! warning "Invariant — `Z.ndim` dispatch (outer vs inner MC)"
    `generate()` must handle both outer (`Z.ndim==2`, `(T,B)`) and inner (`Z.ndim==3`, `(T,B,B2)`) modes, with the per-outer-path initial state broadcast on the **middle** axis in inner mode. This is what lets one process instance serve both loops.

!!! warning "Invariant — inner-MC batch state + fail-loud pricing"
    `shared_mem.simulation_batch` and `shared_mem.fillvalue` must track the current flat batch during an inner fork (set before `reset_inner` and before the pricing pass) and be restored to `B_outer` afterward — in a `finally`, so a mid-fork raise (CUDA OOM, degenerate pricing) cannot leave the state flat-sized and make the *next* chunk fail on shapes instead of the real cause; `fillvalue` is frozen at construction and used as the empty-cat fallback in energy-leg / cash-settle code. Inner-MC pricing must fail loud rather than let `Deal.calculate`'s guard swallow a failure into a scalar-0 mark — inside a fork that silently corrupts the solver's training labels. Both halves are checked: the liability on its flat shape, and every tradable still live in the fork's dependency list on having produced a `tensor_marks` entry (a missing one is indistinguishable from an expired contract, and the solver's `live` mask retires it).

!!! warning "Invariant — `keep_tensor` gates the hedge tradable series"
    `keep_tensor` governs whether `pricing.interpolate` stores `Calc_res['tensor']`; the hedge path sets `Keep_Tensor='Yes'` and harvests those via `tensor_marks`. Removing/altering that store breaks the hedge bundle's tradable series with no error — only missing marks.

## Inner-MC subsystem

`_run_inner_mc_at_t` forks the simulator from each outer-path state at outer step `t`: truncates the grid (`TimeGrid.truncate_to`), windows every deal's `Time_dep` to `{t,t+1}` (`copy_window`), and runs ONE pass at `Batch_Size x Inner_Sub_Batch` flat samples (no partition: peak memory is a function of those two JSON fields, and an over-wide config raises CUDA OOM naming the fork). The pass: `reset_inner` (Sobol), per-process `precalculate` from `outer_buf[key][t]`, `inner_fork_seed` / `reseed_inner_state` for the sufficient statistic, generate, then **publishes every path series the fork produced as a `ScenarioSource`** — the outer-realized past at `B_outer` followed by the forked rows flattened `(B,B2)→B*B2` — for one real pricing pass on restricted `DealStructure`s. *Every* series, not every factor: a process's own `(key, kind)` publication (`BasisLinkedSpotModel`'s `basis_mu`) is read through the same `calc_time_grid_spot_rate` seam as a factor, so a series the OUTER snapshot also carries gets the same logical grid — one expression covers both, and the seeds a fork publishes (`<kind>_inner`) are excluded by that same rule because the outer path does not carry them. Publishing the factors alone left such a series at the fork's own two rows while the pricer asked it for outer row `t`: not a wrong number but an unrunnable configuration (`tests/test_fork_published_state.py`). It uses the model-agnostic verb protocol so the loop is uniform across model worlds — see [The process protocol](dependency_system.md#the-process-protocol).

!!! note "Four objects, one query: rows route by block, tenors route by segment"
    The curve read splits into a **query**, **logical scenario storage** and **one physical
    interpolation**, and nothing holds two of those jobs at once.

    - `CurveTensor` — query coordinates. It keeps scenario ROWS (`index`, `index_next`, `alpha`),
      never a flattened `row * n_tenors` offset, because a tenor segment's stride is its own.
    - `ScenarioBlock` / `ScenarioSource` — logical storage. A block is one physical tensor plus
      `first_row` (where it starts in the logical grid) and `batch_index` (which of ITS columns
      supplies each logical column). A fork publishes two blocks; ordinary generation publishes a
      bare tensor and no source at all.
    - `Interpolation` — one physical tensor and whatever its kind derives from it. It knows
      nothing about blocks, logical rows or batch fan-out, and flattens rows against its OWN
      stride. Base valuation, credit Monte Carlo and the outer hedge loop build only this.
    - `SegmentedInterpolation` — a SIBLING, not a subclass: composes leaves over the TENOR axis
      for a `Near_Interpolation` curve.
    - `RoutedInterpolation` — composes strategies over the SCENARIO axis for a fork.

    `build_interpolation` is the single recursive constructor: bare tensor + kind → leaf; bare
    tensor + segment list → segmented; `ScenarioSource` + either → routed, whose per-block children
    it builds by calling itself. So a segmented curve inside a fork is a `RoutedInterpolation` of
    `SegmentedInterpolation`s and needs no special case — the two compositions are orthogonal.

    **Why a fork publishes blocks.** Every realized-past row is identical across the inner draws,
    so joining them into one tensor writes the past out `Inner_Sub_Batch` times: 98% of the
    stuffed buffer at the production operating point, dragging a same-shaped slab of Hermite
    coefficients with it. Each block interpolates at its OWN width and
    `ScenarioBlock.project` takes the RESULT up to the logical width — never the stored tensor,
    which would hand back exactly the memory the split exists to save.

    **Order is load-bearing.** A read is raw (`read_at`), then blended over time, then `combine`d
    (RT scaling, and the segmented tenor select). `combine` and `project` are both linear, so they
    commute with the blend — which is what lets the routed path be the same arithmetic in the same
    order as an unrouted one, and is why the whole thing is bitwise.

    A time-interpolated read reaches `index + 1`, so a row just below a cut reads ACROSS it and
    names two blocks — `route` classifies on where a read ENDS, not where it starts.

    **Invariant — a source is write-once.** Built after every process's `generate` has published,
    and nothing writes into `t_Scenario_Buffer` afterwards. It answers only `shape` / `new` / the
    RT tenor rescale, so a late write fails loud rather than silently materializing the grid it
    exists to avoid.

    Measured on the production walk-forward book (trade 202001, garch, seed 7), like for like:

    | | 1280x64 | 2048x64 | wall @1280 | kB per flat sample |
    | --- | ---: | ---: | ---: | ---: |
    | joined grid | 6.33 GiB | 10.11 GiB | 116.2 s | 80.62 |
    | block sequence | **1.09 GiB** | **1.71 GiB** | 105.9 s | **13.23** |

    `peak_alloc = 0.057 GiB + 13.23 kB · B_flat` (two-point fit; the 4096x64 rung measured
    3.36 GiB against 3.36 predicted). At a 19.6 GiB allocated ceiling that moves max `B_flat`
    from 254 k to 1.55 M — `Batch_Size` 3 977 → 24 208 at `Inner_Sub_Batch` 64, or
    `Inner_Sub_Batch` 199 → 1 210 at `Batch_Size` 1280.

!!! note "Hermite coefficients are built eagerly, and the block split is why that is affordable"
    An intermediate design deferred the `g,c` pair to the gather that read it, on the argument
    that a fork reads ~11% of a block's rows. That was true of the JOINED grid, where the past's
    coefficients cost `B_flat` columns — 1.03 GiB at 1280x64. Blocks store those rows at
    `B_outer`, so eager costs single-digit MB and the deferral stopped paying for itself.
    Measured both ways on the production walk-forward: deferred 193 s / 0.595 GiB, eager
    **179 s / 0.617 GiB** — 7% faster for 22 MiB of a 0.6 GiB peak, because the build COUNT was
    unchanged (3 737 vs 3 769 — objects widen once) while the span bookkeeping added a device
    sync per gather. On deep credit MC eager wins on both axes (132.4 MiB / 0.68 s against
    152.0 / 0.82). A full-horizon fork was the one shape the deferral would still have helped,
    and that switch is retired: a fork prices exactly `{t, t+1}`, which is every field the
    bootstrap reads. A wider fork should be justified by being the right fork, not by the reader
    hedging against one nobody measured.

## Recomputing the inner Monte Carlo {#recompute-inner-mc}

An MC-priced deal builds one autograd graph per pricing, and the terminal `backward()` holds every
one of them at once — every fixing of every reporting row of every deal — while the simulation that
built them is cheap. `Recompute_Inner_MC` (a `Calculation` field, `No` by default, on
`BaseValuation` and `CreditMonteCarlo`) trades the tape for a second forward pass:
`pricing.InnerMCRecompute` runs its forward under `no_grad` and leaves nothing behind but its
inputs, and its backward re-runs **the same callable** under `enable_grad` and contracts the
cotangent through that one graph, which dies as soon as it has. Peak is one inner graph rather than
all of them.

**One switch, three adopters, one line.** `pv_MC_Tarf`, `pv_MC_AutoCallSwap` and
`pv_discrete_barrier_option` all reach it through `InnerMCRecompute.run(shared, simulate, *theta)`
— the node when the switch is on, the callable when it is off, with the RNG position taken at the
call site. There is no per-pricer flag: which pricings a run can afford to tape is a property of
the valuation engine and the machine, not of the trade. `run` uses `cls.apply`, so a gate that
mutates the node by subclassing it actually gets its mutation.

The **adopter's shape** is the rest of the contract:

- `simulate(*bound)(*theta)` — leading arguments are the block's SHAPE, bound per block with
  `partial` (numpy, ints, flags); trailing ones are every tensor that can carry a graph.
- It returns a **tuple**: element 0 is the block's marks, the rest are by-products the caller
  performs once, each accompanied by the plain-Python row index that places it. Non-tensors pass
  through and take a `None` cotangent. A pricer with no by-products returns `(mtm,)`.
- Every graph-carrying by-product is a **top-level** element of that tuple. A tensor nested inside
  a returned list is *not* an output — measured: `requires_grad False`, no `grad_fn` — so its half
  of a correction would be silently zero.

**The counter is the storage.** What is saved is where each random stream stood
(`utils.rng_position`), never what it produced — the numbers are exactly what is too big to keep.
Sobol draws are memoized per `(dimension, sample_size, batch)`, so rewinding the counter hands the
replay the very same tensor; the regular generator (`torch.rand`, taken at 16 scenarios or fewer
where the Sobol cache is not worth it, and every Heston–Nandi unmonitored sub-step) has no memo, so
its own state is saved and restored. The live
position is put back after the replay: a backward runs long after the forward finished, and a node
that left the streams where its replay ended would move the next node's draws.

Three contracts fall out of it, and each is a way to be silently wrong:

- **One function, called twice** — not a fast forward beside a differentiable copy of it. Two
  spellings of one simulation agree until the day one is edited, and the failure is a wrong
  gradient beside a right price.
- **Pure in everything but its return.** A side effect inside the simulation — a cashflow settled,
  a decision registered — fires a second time in the backward, so the pricer RETURNS those and the
  caller performs them once off the forward's result.
- **Its inputs are its whole theta surface.** Autograd only returns a gradient for a tensor passed
  to `apply`, so anything the simulation reads out of a closure is differentiated as a constant.
  all three pricers build their vol strip at the call site (`forward_vol_strip` differenced by
  `forward_vol_rate`) rather than in the fixing loop; the autocall hoists its
  floating leg and its past equity fixings; all three pass in the Heston–Nandi scalars they used to
  read off `t_Static_Buffer` in the enclosing scope. That last one is what
  `test_the_heston_nandi_theta_survives_the_node` exists for — reverting the barrier's hoist turns
  it red and nothing about the GBM fixtures notices.

!!! warning "Invariant — the settle-outside rule is gated on a COLLATERALISED gradient, not on cashflows"
    A cashflow settled inside the callable is settled twice, and the reported cashflows **cannot
    see it**: the replay runs in `backward()` while `save_cashflows` runs inside `resolve_structure`,
    in the forward pass, so the second settlement lands after the snapshot. Measured with
    `cash_settle` put back inside the autocall's loop — every reported cashflow bit-identical,
    gradient on or off. What is observable is the cashflow's GRAPH: booked under the node's
    `no_grad` forward it reaches `t_Cashflows` carrying nothing, so a collateralised exposure
    reading that ledger through `C_ts_te` loses the channel — 8.7% of the CVA gradient on the
    autocall fixture (max |delta| 7.585e-05 on 8.678e-04).

**It is a FIRST-ORDER node and it says so.** The replay is rooted at *detached* copies of the saved
inputs — that is what stops the inner `autograd.grad` walking back into the outer graph and
double-counting — and a second derivative taken through a detached leaf is severed from the graph
the outer pass holds. The failure is silent: the entries needing that path come back **zero**, which
is a Hessian that looks like a Hessian (measured, `Greeks: 'All'` on the TARF fixture: −1.74e6 /
−4.03e5 / −1.67e4 taped, all three zero recomputed). So `backward` refuses `create_graph` naming the
switch, on the `LeastSquaresSolve` precedent. Keeping the inputs attached instead was tried and
breaks the first derivative outright.

!!! warning "Invariant — ask `torch.is_grad_enabled()` BEFORE `enable_grad`"
    The engine runs a backward node with grad mode set to `create_graph`, so that flag is how a
    node learns it is being differentiated twice — but asked from *inside* the `enable_grad` block
    it always answers True. Getting it the wrong way round here did not raise: it passed
    `create_graph=True` to the inner `autograd.grad`, RETAINED every recomputed graph for the whole
    reverse sweep, and gave the entire saving back and then some — 5.0 GiB taped against 7.4 GiB
    recomputed, on a forward pass that peaked at 0.6.

Measured (`gates/recompute_inner_mc.py`, RTX 3090, float64, CVA gradient on, a never-filling TARF
reported monthly, 3 runs each, CVA and gradient bit-identical throughout):

| fixings × scenarios × inner | taped peak | recomputed peak | taped wall | recomputed wall |
| --- | --- | --- | --- | --- |
| 6 × 128 × 32 | 101.5 MiB | 66.3 MiB (1.5×) | 0.19 s | 0.20 s (1.02×) |
| 12 × 256 × 64 | 1229.6 MiB | 405.1 MiB (3.0×) | 1.13 s | 0.75 s (0.66×) |
| 24 × 256 × 64 | 5266.2 MiB | 1391.0 MiB (3.8×) | 16.39 s | 3.42 s (0.21×) |
| 24 × 512 × 64 | 11481.0 MiB | 3731.4 MiB (3.1×) | 17.66 s | 3.61 s (0.20×) |
| 24 × 512 × 128 | **out of memory** | 8384.5 MiB | — | 3.86 s |

The last row is the number that matters: the taped path cannot price that shape on a 24 GiB device
at all and the node prices it in under four seconds. The second forward pass costs **2%** where
neither path is near the device (top row, which is what the wall column is honestly measuring);
everywhere above that the taped path is memory-bound before it is compute-bound and the node is
3-5× *faster*. The whole-run drop (3-4×) is smaller than the forward-pass drop (8.5×, probed
separately at 24 × 256 × 64: 5023 → 592 MiB) for two reasons, and both are inherent: the backward
holds one recomputed block graph live at a time, and with sensitivities on the per-inner-path
`jumps` are retained exactly as they were before — the boundary registration's own storage, which
this changes nothing about.

The other two adopters, same harness, same question — monthly coupons / monthly barrier
observations reported monthly, CVA gradient on, CVA bit-identical throughout:

| shape | rows × scenarios × inner | taped peak | recomputed peak | taped wall | recomputed wall |
| --- | --- | --- | --- | --- | --- |
| autocall | 6 × 128 × 32 | 45.0 MiB | 39.6 MiB (1.1×) | 0.19 s | 0.26 s (1.41×) |
| autocall | 24 × 256 × 64 | 1093.2 MiB | 281.3 MiB (3.9×) | 2.85 s | 2.91 s (1.02×) |
| autocall | 24 × 512 × 128 | 4693.6 MiB | 1448.7 MiB (3.2×) | 3.61 s | 3.45 s (0.95×) |
| barrier | 6 × 128 × 32 | 68.5 MiB | 45.7 MiB (1.5×) | 0.20 s | 0.21 s (1.07×) |
| barrier | 24 × 256 × 64 | 2227.8 MiB | 361.7 MiB (6.2×) | 2.62 s | 1.35 s (0.52×) |
| barrier | 24 × 512 × 128 | 9239.1 MiB | 1781.4 MiB (5.2×) | 1.66 s | 1.00 s (0.60×) |

Same story as the TARF and for the same reason: the second forward pass is what the top rows are
measuring (41% on a shape too small to care, on a 0.19 s run) and everywhere the tape is large the
node is level or faster. The barrier drops furthest because its OSS keeps nothing per inner path —
no boundary registration rides its node at all — while the autocall's `fired`/`survived` branches
are retained exactly as before.

!!! warning "A measurement taken beside an out-of-memory run is a measurement of that run"
    `torch.cuda.reset_peak_memory_stats` takes the CURRENT allocation as its floor, and a boundary
    registration holds `shared → boundary_sets → gap → node → simulate → shared` — a cycle
    refcounting cannot break, so the previous shape is still resident until the cyclic collector
    runs. Measured: a 6-coupon autocall read a peak of 20 105 MiB, which was the graduation row
    above it that had just gone out of memory. `gc.collect()` before the reset, and `sys.argv[1]`
    to run one adopter's shapes on their own.

## Boundary corrections — the sensitivity subsystem

Some deals take a decision on **simulated state**: a barrier crosses, an autocall triggers, a
swaption exercises, a collateral transfer clears its minimum. The value jump at such a decision is
real — a knocked-out deal *is* worth nothing — so the price is right. What ordinary AAD drops is the
**flux**: as a factor moves, scenarios cross the trigger, and the indicator recording that crossing
has zero derivative almost everywhere. The number is correct and the greek is wrong, which no price
gate can detect.

`pricing.stochastic_boundary_correction(gap, jump, bandwidth)` injects the missing term. It is worth
**exactly zero in the forward pass** — `gap - gap.detach()` is numerically zero with derivative one
— so adding it to the scalar handed to `backward()` reports an unchanged number and still
propagates the term through `gap` to every factor at once. No bump, no second valuation, cost
independent of the factor count.

It has its own lifecycle, and it straddles the phases above:

- **During pricing**, a deal that takes such a decision appends a `utils.BoundarySet` to
  `shared.boundary_sets`. `gap` retains its graph and is signed so `gap > 0` means the trigger
  FIRED; everything else is detached, because a counterfactual yields a *coefficient*, not a
  differentiated quantity.
- **In `NettingCollateralSet.post_process`**, the set stamps its own gross→net chain onto the
  registrations made beneath it. It has to ride on the set: `boundary_sets` accumulates across every
  netting set, so one slot on `shared` would push an uncollateralised set's deal through a
  collateralised set's collateral scan.
- **At the objective** — CVA, FVA or base valuation — `pricing.boundary_correction` walks the
  registrations, and each converts its own decisions into **portfolio** deltas.

!!! warning "Invariant — a counterfactual is scored on the PORTFOLIO"
    The objective is applied to `resolve_structure`'s root sum over *every* netting set, so a
    counterfactual must be `reported_mtm + this registration's change`, never a single set's own
    net level. With exactly one netting set the two coincide — which is what makes the error
    invisible to a single-set fixture, and it is worst precisely where the machinery is aimed,
    because a collateralised set's post-collateral net sits at the relu kink by construction.

Three `BoundarySet` subclasses exist, and they differ in **how far a decision reaches** — which is
why one class with a mode flag would be the wrong shape:

| subclass | reach | carries |
| --- | --- | --- |
| `LatchedBoundarySet` | read by every row from the decision onward (barrier, swaption) | two whole-profile branches, shared across decisions |
| `RowBoundarySet` | lands on its own row and no other (autocall coupon) | a pair of `(B,)` values per decision |
| `InnerBoundarySet` | a decision inside a pricer's inner MC | the objective's *derivative*, not a difference — one inner path moves the row by `1/n`, a jump the value never takes |

!!! note "Under a recompute node, a `gap` is an OUTPUT — when the simulation is what decided it"
    `stochastic_boundary_correction` needs `gap` to carry a graph and an untaped forward pass has
    none to give it (see [Recomputing the inner Monte Carlo](#recompute-inner-mc)). So the
    registration SPLITS: the gap's value is computed under `no_grad` for the set to report, the
    node is what connects it, and the correction's coefficient — assembled at the objective exactly
    as before — arrives as that output's cotangent, so the graph-carrying half of the correction is
    built during the backward, inside the recompute. The contract is unchanged, and it is the same
    contract the whole subsystem has: what reaches `backward()` differs, what is reported does not.

    The converse is the rule, and it is about **where the graph lives** rather than about symmetry.
    A decision taken on OUTER state keeps its own graph whatever the node does to the inner pass,
    so its registration stays outside it. The three adopters split both ways and each way is
    measured: the TARF's knock-in (decided on `Sj`, an inner draw) and the autocall's coupon trigger
    (decided on `Sj` selected by loop state) are node outputs — dropping that cotangent is
    bit-identical to removing the correction outright; the TARF's redemption latch (the block
    loop's own accrual series) and the discrete barrier's crossing latch (`spot_block[-1]` at an
    observation date) are built outside — for the barrier, dropping every cotangent reproduces the
    corrected gradient bit for bit while suppressing the correction moves it 0.30%, which is what
    says the latch is live and simply does not ride the node.

There is **no JSON switch**: `shared_mem.boundary_aad = calc_greeks is not None`. Wanting
sensitivities *is* the switch, and registration is gated on it so the cost is zero when greeks are
off. Branch values register on the **pricer's** grid in the **pricer's** currency; `to_mtm` is the
deal's own map (`pricing.deal_to_mtm_grid` — fx into reporting, then interpolate), because the deal
grid, the MTM grid and the report grid are three different axes and interpolation inserts rows in
the middle.

!!! warning "A decision whose solve amplifies is refused, not smoothed"
    `boundary_weights` fits a local **linear** kernel, so its weights are signed and sum to one —
    and a kernel admitting two neighbouring points a long way out cancels the first-order term by
    subtracting two enormous numbers. Measured on the Heston-Nandi barrier at 512 paths: two points
    1.20 widths out and 0.021 apart, weighted +50.4 / −49.5, supplying 112% of the coefficient and
    the whole of a 73% gradient error. Since the weights sum to one, `||weights||_1` **is** that
    amplification, and `pricing.BOUNDARY_MAX_AMPLIFICATION` bounds it; past the bound the weights
    go to zero and the decision contributes exactly what an empty kernel would.
    Gated by `tests/test_boundary_weight_amplification.py`, whose first job is to prove the refusal
    is *reachable* — the constant it replaced (`1e-30` on a Cauchy-Schwarz ratio pinned into
    `[0, 1]`) never refused anything and could not.

Acceptance is AAD against a common-random-numbers bump ladder (`tests/crn_ladder.py`), which reports
agreement **and flatness** separately. Agreement at one bump size proves nothing; a ladder that
scatters with `h` is differencing across a jump. Note flatness measures the *oracle's* spread only —
a flat ladder beside an AAD that moves with the path count is not evidence of a residual.
