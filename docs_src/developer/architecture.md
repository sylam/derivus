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

The public surface is documented in [API Overview](../api_overview.md) and the job and market data formats in [JSON Configuration](../json/index.md); those pages carry contracts too. This section is the internal view, in reading order:

| page | read it when you |
| --- | --- |
| [Calc Lifecycle](calc_lifecycle.md) | change anything between a job's load and its result: the compile phases, the generate loop, the random streams, the grids, the valuation modes, the inner Monte Carlo, boundary corrections |
| [Dependency System](dependency_system.md) | add a factor type, a process, a deal or a valuation option |
| [Resolver Layer](resolver_layer.md) | write a deal's `calc_dependencies` or a pricer that reads factors |
| [Conventions](conventions.md) | make any change: the house rules, the change scope, what a landing runs, how a test is built |
| [Market Prices](market_prices.md) | add or change a calibration family, `Bootstrapper Configuration`, or a verb that authors market data |
| [Quote Sensitivities](quote_sensitivities.md) | touch a calibration's derivative, or `dV/dq` |
| [Quote Propagation](quote_propagation.md) | ride a quote between calibrations |
| [Structures](structures.md) | work on the sales structures or the quote runner |
| [MCP](mcp.md) | add or change a tool a model drives the desk with |
| [The Spine](spine.md) | work on the book of record: bookings, markets, closes, and how they become a job |
| [Roadmap](roadmap.md) | propose work: what closes before 0.5.0, what is designed and not built, how a change is verified |
| [Model Limitations](../limitations.md) | need what is known to be wrong or limited, and what waits on a desk decision |

## Market data files, the context and the job {#market-data-context-job}

**The market data file is the book of record for configuration.** It carries what a bank calibrates and governs — `Market Prices` with their benchmarks and tolerances, `Bootstrapper Configuration`, `Correlations`, `Price Models`, `Model Configuration`, `System Parameters` — beside the `Price Factors` themselves. A job NAMES it (`MergeMarketData.MarketDataFile`) and its calendar (`CalendDataFile`) rather than carrying them; the [JSON reference](../json/index.md#mergemarketdata) has both formats and says what each is for. How many files there are, what each holds and how often each is recalibrated is the implementation's to decide and never the engine's: a real-world file for exposure profiles, a risk-neutral one for the valuation adjustments and a common one for the correlations and system parameters is one layout, and `Config.parse_json` merges a file onto the declared sections, so several can be loaded onto one config, each carrying the sections it has.

**A `Context` is a session.** It parses each market data file and each calendar a job names once, keeping the parse under the sha256 of the file's bytes (`Context.cached`), and every later `load_json` on that context naming a file of that content reuses the parsed objects — thousands of correlations are read once, not once a job, at the cost of a read of the bytes. A bootstrap run on the context writes its calibrated parameters into the config's `Price Factors`, where the jobs loaded after it read them. The service's live book owns the parse of the files it names (`service.Book.parses`): a request's context is its own, so one context still runs one job, and it takes each file from that store (`load_json(..., parses=)`), unpickled into a copy of its own, so a file is read and hashed per request and parsed once per content, the XVA tab's long-lived context and the calendar a curve verb rolls against included (that last **Not gated**: the loads beside it have already put the calendar in the store). **Held by** `tests/test_service.py::test_the_book_parses_the_file_it_names_once_per_content`, `::test_every_netting_set_prices_on_one_kept_context_over_the_files_the_book_names`.

**The job JSON is a light overlay, and whoever generates it answers for it.** `load_json` merges the job's `ExplicitMarketData` onto the cached config a section at a time (`Config.merge_section`, the last statement of a key winning), sets the job's deals and calculation on it, and that config is what `run_job` prices. So a job states the price factors and the deals it needs and nothing else, and what an earlier job stated STAYS on the context: a later job that restates a factor reads its own, one whose deals never read it is unaffected, and one that reads it without restating it reads the earlier job's. The process that generates a job therefore states every price factor that job depends on — and need not guess them, the engine works them out (`Config.factor_universe`, `calculate_dependencies`; see [Dependency System](dependency_system.md)). This is [JSON is the contract](conventions.md#json-is-the-contract) at the scale of a bank's market data: nothing in the engine empties or guards a context between loads, because what a bootstrap calibrated lives there too.

**One context, one job at a time.** `load_json` sets the config the next `run_job` prices, so loads and runs on a context are sequential, and whoever runs long — a batch, a service's worker — KEEPS the context and loads through it. A data cache is an attribute of the object that owns the data, here the context; it is never a dictionary at the top of a module.

**The files are part of the record by their content.** A valuation adjustment, or an exposure profile as it stood on a past date, is reproducible only against the files it was priced on. A job names each file by its path, so a job document standing alone always runs off what its paths hold; the record attests content and never replaces the path. A file is known by its content: `plan_hash` covers a market data file's sections and the calendar's weekend rules and holidays, `values_hash` the file's values, and a run names the sha256 of each file it read under `Stats['Files']`, so a file rewritten on disk is a new plan or new values however soon after. The desk DECLARES a file in the [spine](spine.md#markets) by those bytes (`file_declared`, under `mark`): a blob addressed by the sha256 a run reports - configuration, and so kept in the clear - under a name the desk gives it, the latest declaration under a name standing. A standing run's attestation carries the digests of the files it read beside its replay tuple, and a replay hands its context those digests (`spine.reading`): `Context.digest` takes a file's bytes from the record's store where the store holds them (`Context.held`) and off the path where it does not, the replay's own `Stats['Files']` then saying whether it read the same, and every reader of a past marks job - a P&L window's diary and base currency, its explain, a collateral call - reads that job's files the same way. The XVA tab prices the bytes its stamp hashed the same way, so a file rewritten while a set waits moves nothing the set's id names. **Held by** `tests/test_declared_defaults.py::test_the_calendar_a_job_names_is_in_its_plan_by_what_it_holds`, `tests/test_market_prices_partition.py::test_a_market_data_file_rewritten_inside_one_tick_is_read_anew`, `tests/test_service.py::test_a_file_rewritten_inside_one_tick_is_a_new_run`, `::test_an_edit_to_either_named_file_is_a_new_run_and_the_request_parses_neither`, `tests/test_spine_verbs.py::test_a_file_is_declared_by_its_bytes_and_the_latest_under_a_name_stands`, `tests/test_spine_engine.py::test_a_run_says_the_files_it_read_and_a_replay_reads_the_bytes_that_stood`, `tests/test_spine_oracle.py::test_a_run_on_a_file_nobody_declared_is_named_and_passes_once_it_is_declared`, `tests/test_pnl.py::test_a_window_reads_the_files_its_marks_read_as_they_stood`, `::test_the_explain_and_the_call_of_a_past_close_read_the_files_its_marks_read`.

## One `Factor` keys everything

`Factor = namedtuple('Factor', 'type name')` (`utils.Factor`) is the identity used by **every** dict in the pipeline: the discovery graph, `stochastic_factors` / `static_factors`, `all_factors`, `all_tenors`, and the runtime buffers. One key across many dicts is what lets the layers compose without a translation table.

!!! warning "Invariant — the `Factor` identity"
    `Factor = (type:str, name:tuple[str])`. The name is atomic: a dotted market-data name (`"PLATINUM_CME.LME_CME"`) is split into `('PLATINUM_CME','LME_CME')` **only** in `utils.check_rate_name` / `check_tuple_name`, at the [resolver boundary](resolver_layer.md); deal code and processes carry the whole `Factor` by reference and never index into `name`. One `Factor` value keys four dict families identically — the offset maps (`static_factors`/`stoch_factors`), the object graph (`all_factors`), the tenor payloads (`all_tenors`) and the runtime buffers, which are populated under the process's own `factor_key`. Split the name early and this breaks silently.

## The three phases

**1. Compile — `calculate_dependencies`.** Walk the deal tree, discover every price factor a deal touches, wire each to its sub-factors, collect the max date each is needed to, topologically order them, and split **stochastic** (simulated) vs **static** (frozen leaf). Table-driven, not branching code — see [Dependency System](dependency_system.md).

**2. Compile — `_build_factor_state` + `precalculate`.** Construct the factor objects, mint AAD leaves (`torch.tensor(..., requires_grad=…)`; reverse-mode sensitivities as in [@capriotti2011]), build each stochastic process, assemble the correlation matrix and its cholesky, and assign each process its RNG-substream offset (`process_ofs`). See [Calc Lifecycle](calc_lifecycle.md).

**3. Execute — the generate loop.** Per simulation batch: draw the correlated random block, iterate `stoch_factors` in topological order publishing each path into `t_Scenario_Buffer` as it is produced (so a linked factor reads its parent's already-published path), then price the deal tree, accumulating MTM.

## Why registries, not functions

Extension points are **data**, not control flow. Adding a factor type, a process, a deal or a valuation option means adding a row to a registry (or a class attribute the engine iterates), never editing a dispatcher; the dispatchers are `globals()`-keyed on class name. Mechanics in [Conventions](conventions.md), recipes in [Dependency System](dependency_system.md#extension-recipes).

## Where valuation modes diverge

`Context.run_job` runs the class `Calculation['Object']` names through that class's own `run`, one of five: `BaseValuation` (single-date static reval), `CreditMonteCarlo` (the full scenario engine — exposure and the [xVA](../theory/valuation_adjustments.md)), `HedgeMonteCarlo` (the CMC engine, harvesting raw marks for the diff-ML hedge solver and forking an inner Monte-Carlo), `SIMM` (a CRIF by bump and revaluation, the book compiled once and priced again per market input moved with that input written into its leaves, a quote riding its curve's calibration Jacobian, or under `Method: Graph` off one backward pass per trade, and the margin off it in torch) and `SACCR` (the SA-CCR exposure at default per netting set, off the same one compile's values and each trade's declared terms). All five share the compile phases; they differ only in what execute does with the priced tensors. See [Calc Lifecycle](calc_lifecycle.md#valuation-modes).

## Design direction

The long-term shape is a stateless streaming compute: JSON events in, stateless compute, JSON results out, state living outside the process. The engine holds no cross-job state beyond the market-data cache, and the job JSON is the whole contract. See [Conventions — JSON is the contract](conventions.md#json-is-the-contract).
