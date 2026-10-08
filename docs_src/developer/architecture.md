# Architecture

derivus is a **financial virtual machine**. A job is a program; the engine compiles it, then executes it against Monte-Carlo scenarios.

| VM concept | derivus |
| --- | --- |
| program | the job JSON — `Calculation`, `Deals`, and the market data it states or names (`Price Factors`, `Price Models`, `Correlations`) |
| loader | `Context.load_json`, onto the config the context keeps for the market data file the job names |
| compile | `Config.calculate_dependencies` (discover + order factors) + each process's `precalculate` |
| instructions | `StochasticProcess.generate` (per factor) and `Deal.calculate` / `pricing.*` (per deal) |
| execute | the per-batch generate loop in `Calculation.execute` |
| registers / heap | `shared_mem.t_Scenario_Buffer` (simulated paths — a tensor, or a `ScenarioSource` sequence of row blocks inside an inner-MC fork), `t_Static_Buffer` (static leaves) |
| memoized eval cache | `shared_mem.t_Buffer` |

The public surface is documented in [API Overview](../api_overview.md); this section is the internal view. Reading order: Architecture → [Calc Lifecycle](calc_lifecycle.md) → [Dependency System](dependency_system.md) → [Resolver Layer](resolver_layer.md) → [Conventions](conventions.md). (mkdocs sorts the nav alphabetically; follow the prose order.) The record around the engine - bookings, markets, closes, and how they become a job - starts at [The Spine](spine.md#how-it-fits).

## Market data files, the context and the job {#market-data-context-job}

**The market data file is the book of record for configuration.** It carries what a bank calibrates and governs — `Market Prices` with their benchmarks and tolerances, `Bootstrapper Configuration`, `Correlations`, `Price Models`, `Model Configuration`, `System Parameters` — beside the `Price Factors` themselves. A job NAMES it (`MergeMarketData.MarketDataFile`) and its calendar (`CalendDataFile`) rather than carrying them; the [JSON reference](../json/index.md#mergemarketdata) has both formats and says what each is for. How many files there are, what each holds and how often each is recalibrated is the implementation's to decide and never the engine's: a real-world file for exposure profiles, a risk-neutral one for the valuation adjustments and a common one for the correlations and system parameters is one layout, and `Config.parse_json` merges a file onto the declared sections, so several can be loaded onto one config, each carrying the sections it has.

**A `Context` is a session.** It parses each market data file a job names once (`config_cache`) and each calendar file once (`holiday_cfg_cache`), and every later `load_json` on that context naming them reuses the parsed objects — thousands of correlations are read once, not once a job. A bootstrap run on the context writes its calibrated parameters into the config's `Price Factors`, where the jobs loaded after it read them.

**The job JSON is a light overlay, and whoever generates it answers for it.** `load_json` merges the job's `ExplicitMarketData` onto the cached config a section at a time (`Config.merge_section`, the last statement of a key winning), sets the job's deals and calculation on it, and that config is what `run_job` prices. So a job states the price factors and the deals it needs and nothing else, and what an earlier job stated STAYS on the context: a later job that restates a factor reads its own, one whose deals never read it is unaffected, and one that reads it without restating it reads the earlier job's. The process that generates a job therefore states every price factor that job depends on — and need not guess them, the engine works them out (`Config.factor_universe`, `calculate_dependencies`; see [Dependency System](dependency_system.md)). This is [JSON is the contract](conventions.md#json-is-the-contract) at the scale of a bank's market data: nothing in the engine empties or guards a context between loads, because what a bootstrap calibrated lives there too.

**One context, one job at a time.** `load_json` sets the config the next `run_job` prices, so loads and runs on a context are sequential, and whoever runs long — a batch, a service's worker — KEEPS the context and loads through it. A data cache is an attribute of the object that owns the data, here the context; it is never a dictionary at the top of a module.

**The files are part of the record.** A valuation adjustment, or an exposure profile as it stood on a past date, is reproducible only against the files it was priced on, so a market data file and a calendar are each a blob addressed by its content like any other the [spine](spine.md) holds - configuration, and so kept in the clear - and each is DECLARED in the record like any other fact. One of each name stands at a time, the latest, and a run at a point in the record reads the ones standing there: nothing carries a version, so nothing can be priced against an older file by choice, and a document is its deals and nothing of the market. What the record and the service do not yet do with a named file is on the [Roadmap](roadmap.md).

## One `Factor` keys everything

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

`Context.run_job` is a 4-way branch on `Calculation['Object']`: `BaseValuation` (single-date static reval), `CreditMonteCarlo` (the full scenario engine — CVA/FVA/exposure), `HedgeMonteCarlo` (the CMC engine, harvesting raw marks for the diff-ML hedge solver and forking an inner Monte-Carlo) and `SIMM` (a CRIF by bump and revaluation, one booked trade at a time, each market move re-bootstrapped and revalued). All four share the compile phases; they differ only in what execute does with the priced tensors. See [Calc Lifecycle](calc_lifecycle.md#valuation-modes).

## Design direction

The long-term shape is a stateless streaming compute: JSON events in, stateless compute, JSON results out, state living outside the process. The engine holds no cross-job state beyond the market-data cache, and the job JSON is the whole contract. See [Conventions — JSON is the contract](conventions.md#json-is-the-contract).
