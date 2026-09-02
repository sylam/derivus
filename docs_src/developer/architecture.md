# Architecture

derivus is a **financial virtual machine**. A job is a program; the engine compiles it, then executes it against Monte-Carlo scenarios.

| VM concept | derivus |
| --- | --- |
| program | the job JSON — `Calculation`, `Deals`, market data (`Price Factors`, `Price Models`, `Correlations`) |
| loader | `Context.load_json` |
| compile | `Config.calculate_dependencies` (discover + order factors) + each process's `precalculate` |
| instructions | `StochasticProcess.generate` (per factor) and `Deal.calculate` / `pricing.*` (per deal) |
| execute | the per-batch generate loop in `Calculation.execute` |
| registers / heap | `shared_mem.t_Scenario_Buffer` (simulated paths — a tensor, or a `ScenarioSource` sequence of row blocks inside an inner-MC fork), `t_Static_Buffer` (static leaves) |
| memoized eval cache | `shared_mem.t_Buffer` |

The public surface is documented in [API Overview](../api_overview.md); this section is the internal view. Reading order: Architecture → [Calc Lifecycle](calc_lifecycle.md) → [Dependency System](dependency_system.md) → [Resolver Layer](resolver_layer.md) → [Conventions](conventions.md). (mkdocs sorts the nav alphabetically; follow the prose order.)

## The spine: one `Factor` keys everything

`Factor = namedtuple('Factor', 'type name')` (`utils.Factor`) is the identity used by **every** dict in the pipeline: the discovery graph, `stochastic_factors` / `static_factors`, `all_factors`, `all_tenors`, and the runtime buffers. One key across many dicts is what lets the layers compose without a translation table.

!!! warning "Invariant — the `Factor` identity"
    `Factor = (type:str, name:tuple[str])`. The name is atomic: a dotted market-data name (`"PLATINUM_CME.LME_CME"`) is split into `('PLATINUM_CME','LME_CME')` **only** in `utils.check_rate_name` / `check_tuple_name`, at the [resolver boundary](resolver_layer.md); deal code and processes carry the whole `Factor` by reference and never index into `name`. One `Factor` value keys four dict families identically — the offset maps (`static_factors`/`stoch_factors`), the object graph (`all_factors`), the tenor payloads (`all_tenors`) and the runtime buffers, which are populated under the process's own `factor_key`. Split the name early and this breaks silently.

## The three phases

**1. Compile — `calculate_dependencies`.** Walk the deal tree, discover every price factor a deal touches, wire each to its sub-factors, collect the max date each is needed to, topologically order them, and split **stochastic** (simulated) vs **static** (frozen leaf). Table-driven, not branching code — see [Dependency System](dependency_system.md).

**2. Compile — `_build_factor_state` + `precalculate`.** Construct the factor objects, mint AAD leaves (`torch.tensor(..., requires_grad=…)`), build each stochastic process, assemble the correlation matrix and its cholesky, and assign each process its RNG-substream offset (`process_ofs`). See [Calc Lifecycle](calc_lifecycle.md).

**3. Execute — the generate loop.** Per simulation batch: draw the correlated random block, iterate `stoch_factors` in topological order publishing each path into `t_Scenario_Buffer` as it is produced (so a linked factor reads its parent's already-published path), then price the deal tree, accumulating MTM.

## Why registries, not functions

Extension points are **data**, not control flow. Adding a factor type, a process, a deal or a valuation option means adding a row to a registry (or a class attribute the engine iterates), never editing a dispatcher; the dispatchers are `globals()`-keyed on class name. Mechanics in [Conventions](conventions.md), recipes in [Dependency System](dependency_system.md#extension-recipes).

## Where valuation modes diverge

`Context.run_job` is a 3-way branch on `Calculation['Object']`: `BaseValuation` (single-date static reval), `CreditMonteCarlo` (the full scenario engine — CVA/FVA/exposure) and `HedgeMonteCarlo` (the CMC engine, harvesting raw marks for the diff-ML hedge solver and forking an inner Monte-Carlo). All three share the compile phases; they differ only in what execute does with the priced tensors. See [Calc Lifecycle](calc_lifecycle.md#valuation-modes).

## Design direction

The long-term shape is a stateless streaming compute: JSON events in, stateless compute, JSON results out, state living outside the process. The engine holds no cross-job state beyond the market-data cache, and the job JSON is the whole contract. See [Conventions — JSON is the contract](conventions.md#json-is-the-contract).
