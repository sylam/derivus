# API

Everything in Derivus is based off a *Context*. All calculations are constructed with reference
to one.

!!! note "Curve convention"
    Interest rate curves typically start one day from now, i.e. $1/365\approx 0.00274$. A curve
    without a knot at time 0 flat-extrapolates its first rate back. A knot at time 0 is harmless
    under the `Linear` and `Hermite` interpolations (the job skeleton carries one, under the default
    `Linear`), but under `LinearRT` and `HermiteRT` the rate is recovered by dividing by the query
    tenor, so a query at exactly 0 on such a curve has no rate (0/0) - see
    [Price Factor Interpolation](json/model_configuration.md#price-factor-interpolation).

## The Context

A `Context` holds the loaded JSON config, market data, deal hierarchy, and calendar metadata. All
calculations read from it. Once a context is loaded, you can reuse it for multiple calculations
(e.g. revalue, then run a Monte Carlo simulation, then run a hedge optimisation) without re-parsing
the JSON.

```python
import derivus as rf

cx = rf.Context()
cx.load_json('fxfwd.json')
```

(The alias is arbitrary and `rf` is the one this page and the rest of `docs_src` were written
under — the same pre-rename initials the `RF_SERVICE_URL` wire variable keeps on purpose. The
README says `dv`; both are `import derivus as`.)

A context can hold *multiple* loaded configurations — `cx.config_cache` holds one `Config` per
`MergeMarketData.MarketDataFile` path a job referenced, written only on a miss for that path. A job
carrying only inline `ExplicitMarketData` builds a fresh `Config` that is never cached, and a job
with no `MergeMarketData` block reuses `cx.current_cfg` and mutates it in place. The most recently
loaded config is always `cx.current_cfg`. All
calculation methods read from `cx.current_cfg`, so to switch which configuration is active you
re-load (or assign `cx.current_cfg` directly to a previously-cached `Config`).

The active configuration exposes:

- `cx.current_cfg.params` — the merged market-data dictionary (`'System Parameters'`,
  `'Price Factors'`, `'Price Models'`, `'Correlations'`, etc.). Mutate this directly to override
  loaded market data.
- `cx.current_cfg.deals` — the deal hierarchy (`'Calculation'`, `'Deals'`, `'Attributes'`).
- `cx.holiday_cfg_cache` — calendar definitions parsed from referenced XML calendar files (this
  one lives on the context itself, since calendars are shared across configurations).

## Running a calculation

Four calculation types are supported. Three have an explicit method, and all four run from the
JSON-driven dispatcher; every field each reads is in the [Calculations](json/calculations.md)
reference:

| Method | JSON `Calculation.Object` | Purpose |
|---|---|---|
| `cx.Base_Valuation(overrides)` | `BaseValuation` | Single-point MTM revaluation |
| `cx.Credit_Monte_Carlo(overrides)` | `CreditMonteCarlo` | Path-dependent simulation (CVA / FVA / PFE) |
| `cx.Hedge_Monte_Carlo(overrides)` | `HedgeMonteCarlo` | Same simulation engine, used to solve a dynamic hedging problem (DiffSolver) |
| `derivus.run_simm(cx.current_cfg, overrides)` | `SIMM` | ISDA SIMM sensitivities (a CRIF) by bump and revaluation, one booked trade at a time |
| `cx.run_job(overrides)` | (any of the above) | Dispatches based on the loaded JSON's `Calculation.Object` |
| `cx.validate()` | (any of the above) | Reports what would stop the loaded job running, without running it |
| `cx.describe()` | (any of the above) | Reports what the engine made of the loaded job, without running it |

Use `run_job()` when the JSON itself fully specifies which calculation to run:

```python
cx = rf.Context()
cx.load_json('BaseValuation.Test1.json')
calc, out = cx.run_job(overrides={})
```

Use the explicit methods when you want to run a different calculation than the JSON specifies (for
example, running a Credit Monte Carlo against a JSON originally written for Base Valuation).

Each method returns a `(calc, out)` tuple — the calculation object (useful for inspecting state
post-run) and the output. `out` is a dict for `Base_Valuation` and `Credit_Monte_Carlo`; for
`Hedge_Monte_Carlo` it is a `HedgeRuntimeExecutionResult` attribute object, not a dict (below).

### Base Valuation

A single-point theoretical price. Cheapest of the three calculations.

```python
calc, out = cx.Base_Valuation(overrides={'Run_Date': '2024-08-01', 'Currency': 'USD'})
out['Results']['mtm']
```

returns a pandas DataFrame:

```
Test	NettingCollateralSet	0.0
	341	FXNonDeliverableForward	-343.123474121
```

So the market value at 1 August 2024 of the forward is -343 USD. The output structure depends on
the deals loaded; `out['Results'].keys()` enumerates everything that's available.

When `Greeks` is set on the calculation, additional dataframes appear in `out['Results']` for
first-order (and optionally second-order) sensitivities by risk factor.

### Credit Monte Carlo

Monte Carlo simulation over a configurable time grid. Used for path-dependent metrics like
exposure profiles, CVA, FVA.

```python
params = {
    'Time_grid': '0d 2d 1w(1w) 3m(1m) 2y(3m)',
    'Run_Date': '2024-08-01',
    'Currency': 'ZAR',
    'Simulation_Batches': 2,
    'Batch_Size': 512,
    'Random_Seed': 6126,
    'Calc_Scenarios': 'No',
    'Generate_Cashflows': 'Yes',
    'Dynamic_Scenario_Dates': 'Yes',
}
calc, out = cx.Credit_Monte_Carlo(overrides=params)
```

```python
out['Results']['exposure_profile']
```

```
             EE        ENE      PFE_95
2024-08-01    0.000000   0.000000    0.000000
2024-08-03    1.809047  -1.712285    0.000000
2024-08-08   25.414378 -23.905112  201.102859
...
```

**EE** is the Expected Exposure (the mean of the mark clipped at 0 below), **ENE** the Expected
Negative Exposure (the mean clipped at 0 above, so never positive). The peak column is named for
its percentile — one `PFE_<p>` per entry in the calculation's `Percentile` field, default `95`,
comma-separated for several — so there is no column called plain `PFE`. The result is a pandas
DataFrame and so can be plotted via `.plot()`.

!!! note "Sharding with `runparallel`"
    `Credit_Monte_Carlo(runparallel=True)` runs one worker process per visible CUDA device (one on a
    box without CUDA); an integer asks for exactly that many workers, sharing devices. Workers take
    contiguous ranges of the batches and every batch is seeded from its global index, so the result
    does not depend on the worker count; a remainder of `Simulation_Batches` that does not divide
    by the worker count is not run. It returns a dict of LISTS, one entry per worker under each of
    `Results`, `Stats`, `Params`, `Reference` and `Job`, rather than `(calc, out)`: pool the `mtm`
    and re-summarize, since `EE`/`ENE` average over equal shards and a `PFE_<p>` percentile does
    not.

### Hedge Monte Carlo

A specialisation of Credit Monte Carlo wired into a differential-ML hedging solver (DiffSolver).
The same scenario engine generates trajectories which are consumed by a backward-DP value-function
solver that hedges a portfolio of liabilities by trading a configured set of futures or other
instruments. See [Hedging](hedging/overview.md) for the `Hedging_Problem` configuration contract.

```python
calc, result = cx.Hedge_Monte_Carlo(overrides={'Random_Seed': 42})
result.policy_artifact
```

`Hedge_Monte_Carlo` returns `(calc, result)` where `result` is a `HedgeRuntimeExecutionResult`
(`hedge_bundle.py`) — a plain attribute object with `bundle`, `runtime`, `evaluation_summary`,
`optimizer_diagnostics`, `policy_artifact` and `metadata`, and **no** `__getitem__`, so
`result['Results']` raises `TypeError`.

When `Execution_Mode` is `solve_hedge`, the solver fits the value function in-process:
`result.policy_artifact` is the fitted value function, `result.evaluation_summary` the held-out
verdict, `result.optimizer_diagnostics` the solver's own numbers. When `Execution_Mode` is
`simulate_only`, only the scenario bundle is computed and the no-trade baseline is run (useful for
offline analysis).

## Checking a job before running it

`cx.validate()` reads the loaded job and returns what would stop it running, as a JSON-serializable
dict. It runs nothing, prices nothing and changes nothing:

```python
cx = rf.Context()
cx.load_json('fxfwd.json')
cx.validate()
{'deals': {'FWD1': ['Sell_Discount_Rate is required']}, 'factors': ['InterestRate.ZAR'],
 'calculation': []}
```

- `'deals'` — the authoring messages of every deal in the book, keyed by the deal's `Reference`
  (the walk position, as `#3`, where that is blank or repeated). These are the rules a deal states
  about itself: a field it cannot price without, a value that is not what that field declares —
  a number, a `{".Timestamp": ...}` date, one of a menu's own strings — or a rule spanning
  several of its fields.
- `'factors'` — every price factor the book names that the market data has no `Price Factors` block
  for, spelled as the key you would add. A factor with no block is never built, so the deals that
  reference it are dropped from the portfolio when the run reaches them.
- `'calculation'` — what the `Calculation` block's own declarations say against it, or that its
  `Object` names no calculation: a field it must state (`Currency is required`) or a value that
  is not what its field declares. A run refuses the block by name for the same messages, and a
  refused block is not walked, so `'factors'` is empty beside it.

All three empty means nothing here can tell you the job will fail. A deal's message never stops it
pricing: the engine still fails exactly where it always failed, and this says so first.

`cx.describe()` answers the other question — not what is wrong with the job, but what the engine
made of it. Also read-only, on the same discovery walk:

```python
cx.describe()
{'deals': {'FXForwardDeal': 2, 'NettingCollateralSet': 1},
 'factors': {'resolved': ['FxRate.USD', 'FxRate.ZAR', 'InterestRate.USD'],
             'missing': ['InterestRate.ZAR']},
 'calculation': {'Object': 'BaseValuation', 'Base_Date': Timestamp('2024-06-28'), ...}}
```

- `'deals'` — the book counted by the `Object` each deal was constructed from. A node whose `Object`
  names no deal type is counted under nothing: it keeps that name and nothing else, and
  `validate()` is where it is reported, by its walk position and the name as authored.
- `'factors'` — the same walk as `validate()`, both halves this time: `resolved` is what a run would
  build, and `missing` is the want-list `validate()` returns on its own. Under a refused
  `Calculation` block both are empty and its messages sit beside them under `'calculation'`.
- `'calculation'` — the `Calculation` block as loaded.

## Patching market values and replaying a run

The market data splits in two. A price factor field declared `bind='value'` is one whose CONTENT
the engine reads and nothing else depends on — a spot, the rate column of a curve, the vol column of
a surface, a calibrated model parameter, a market implied correlation, a recovery rate. Everything
else is STRUCTURAL: change it and the job is a different program. A curve splits *inside* itself,
since its knots size the tenor grid while its rate column is content.

```python
patch = cx.market_patch()          # {factor_name: {field: content}} - the values half, all of it
patch['FxRate.ZAR']['Spot'] = 19.0
cx.patch_market(patch)             # applied in place; anything structural raises, naming it
```

The values half spans BOTH market sections. A `Market Prices` quote block contributes its
`Points` rows' quoted values — the mid, the two-way, the timestamp — as
`{block_name: {'Points': [{...} per row]}}`, index-aligned with the block's own row order
(which is structural, so a changed row count refuses naming both lengths). Per row the patch
is a delta: a named field replaces, an omitted field keeps, a JSON `null` clears a two-way
side or a timestamp and refuses on the mid — a mid is moved, never cleared. A `null` in the
DOCUMENT is an absence, so `patch_market(market_patch())` is the identity: neither hash moves.
A patched quote re-bootstraps nothing — the written factors stand, `values_hash` records the
board honestly, and the consumer that reads quotes at execute time is `Quote_Propagation`'s
ride. The live book is deliberately stricter: `/book/market` refuses a quote-values patch and
names the quote-update path (which bootstraps atomically) as the remedy.

`patch_market` takes what `market_patch` emits. Two verbs hash the two halves:

| Verb | What it hashes |
|---|---|
| `cx.plan_hash()` | the program — `params` and `deals`, less every value-bound field and `Random_Seed` |
| `cx.values_hash()` | exactly what `cx.market_patch()` emits |

Both are sha256 over a canonical dump and pure functions of the loaded config: they run nothing,
price nothing and change nothing. The plan covers the `Calculation` block too, so `Batch_Size` and
`Simulation_Batches` move it — they change the realized numbers.

A reported number is replayable from four coordinates:

```python
(cx.plan_hash(), cx.values_hash(), rf.__version__, calculation['Random_Seed'])
```

`rf.__version__` is the engine version, and the seed is its own coordinate because it belongs to
neither hash. Two runs agreeing on all four report the same numbers, so one plan can be compiled
once and re-run against many value sets — and a run whose engine version differs is not a replay,
because a code change may legitimately reassign the RNG substreams.

## Overrides

Every calculation method accepts an `overrides` dict that updates the JSON's `Calculation` section
just before execution. Common overrides:

- `Run_Date` — switch the valuation date without editing the JSON. The runner sets it from the
  document's `Base_Date` before overrides apply, so overriding `Base_Date` alone does not move the
  date (a `SIMM` run is the exception: it reads `Base_Date`)
- `Currency` — change the reporting currency
- `Random_Seed`, `Batch_Size`, `Simulation_Batches` — control Monte Carlo reproducibility and size;
  on a CUDA device a credit Monte Carlo reports what it held at its peak beside what the device
  has (`Stats.Device_Memory`), and warns where the peak passed the device, since the driver pages
  such a batch over the bus rather than refusing it
- `Greeks` — Base Valuation sensitivities (`'No'` / `'First'` / `'All'`); a Monte Carlo turns
  gradients on per sub-block (`Gradient`) and picks its variables with `Gradient_Variables`
  (`'All'` / `'Factors'` / `'Implied'`)
- `Time_grid` — re-shape the simulation time grid (Credit Monte Carlo reads it as `Time_grid`,
  Hedge Monte Carlo as `Time_Grid`)

Overrides merge **deeply** into the loaded `Calculation` object: a nested dict is merged key by key,
so `{'Hedging_Problem': {'Solver': {'Object': …}}}` changes exactly that entry and leaves the rest
of the block alone. Only non-mapping values replace — a sub-dict never wholesale-replaces its
target, so keys you meant to drop by passing a complete sub-dict survive the merge.

## Inspecting and modifying loaded data

`cx.current_cfg.params` exposes the full market-data tree of the active configuration as nested
dicts. To override a single price factor's spot before running a calculation:

```python
cx.current_cfg.params['Price Factors']['EquityPrice.AAPL']['Spot'] = 200.0
calc, out = cx.run_job(overrides={'Run_Date': '2024-08-01'})
```

Similarly, `cx.current_cfg.deals['Deals']['Children']` is a list of deal definitions you can
append to or mutate. Calling `cx.run_job()` after mutating either of these picks up the changes —
there's no implicit cache that needs to be invalidated.

## Output structure

A base valuation returns `(calc, out)` where `out` is a dict with three top-level keys; a credit
Monte Carlo returns the same three plus a fourth; a `SIMM` run returns `Stats` and `Results` only.
(A hedge run returns a
`HedgeRuntimeExecutionResult` instead of a dict — see [Hedge Monte Carlo](#hedge-monte-carlo).)

- `'Netting'` — the internal `DealStructure` tree. Useful for developers walking the hierarchy.
- `'Stats'` — a dict of timing and counter statistics from the run.
- `'Results'` — the user-facing dataframes / arrays. Keys vary by calculation type; see the
  [Output](output.md) page.
- `'Jacobians'` — credit Monte Carlo only: the harvested Jacobians keyed by variable name, `{}`
  when the run asked for no gradients.

## The same verbs over HTTP

One vocabulary, two bindings. Everything above is the in-process binding; `derivus.service` is the
same verbs over HTTP. The pass-through reading holds for `/schema`, `/schema/job` and `/validate` —
build a `Context` from the posted job, call one of the methods above, serialise what comes back —
but not for the rest: `/describe` adds its own cost estimate, `/execute` its lanes, queue and
attestation, and the book, projection and queue verbs are a layer of logic and state that exists
only here (the plan cache, the compute executor's priority queue and result store, the risk cache,
`DV_HOME/xva.json`, the Bloomberg job, the metronome tick). A browser SPA, an MCP
binding, a marimo notebook and an Excel add-in are clients of the same endpoints, so nothing
specific to any one of them belongs on the surface. `fastapi` and `uvicorn` are the `service`
extra, imported only there, so `import derivus` needs neither:

```
pip install derivus[service]
DV_Service --port 8000
```

Every route, its method and what it does is on [HTTP API](http_api.md), generated from the
service's own declarations; a running service describes each in full at `/docs` and
`/openapi.json`. What follows is the contract those routes share.

THE BLOTTER'S TWO DATA VIEWS are `/book/risk` and `/book/xva`, and they are not the same kind of
thing. Risk is whole-book and counterparty-blind, answered inline and cached on the content of
everything the run reads, so a client polls it on the same beat it polls `/book`. XVA is per
netting set and a CACHED PROJECTION: a credit Monte Carlo is minutes of device time and must never
ride a tick, so `DV_HOME/xva.json` holds the last run of each set — written atomically, one row at
a time — and a desk asks for a FULL or PARTIAL recalc when it wants the file to move. The
projection is a mosaic on purpose: each row carries its own `as_of`, a partial recalc moves only
the rows it names, and staleness is data rather than a failure.

The book's FILE is the source of truth: every client — the web UI, an MCP tool, the Excel add-in —
reads and writes it through these verbs, so a deal booked by one appears to the others on their
next etag poll. Deals are addressed by positional `deal_path` (`"0/2/1"`), because references are
not unique in a book. A write is read-edit-write AGAINST THE ETAG IT READ, and the edit runs
outside the book's lock: a read or a booking never waits behind a tick's bootstrap, and a write
that lands in between costs the edit a redo on the document that now stands rather than costing
the other client its wait — three passes, the last of them under the lock, so an edit that keeps
losing to shorter ones still lands. An edit that APPENDS to the record before it writes — the
acceptance always, `POST /book/deals` under a spine home — takes the lock for the whole act
instead, a redo being a second run of what the first pass already filed.

UNDER A SPINE HOME THE BOOK PRICES AT ITS NET. The verbs that price the live book - `/book/risk`,
`/book/xva`, `/book/price`, `/calculations/run` and a quote's risk-impact step, beside `/execute`
and `/prepare` - price it compiled: every position the record holds written at its net, the same
terms' second node under one set ignored as the first one's clip, and every print the declared
sources order filled in. Every deal type declares which of its fields are amounts (`sized` on the
descriptor, `magnitude` for an amortisation step, which scales and never takes a short's sign),
which says the side a short flips (`side`), and which dates settle cash it builds no schedule for
(`settles`, each payment's currency, amount and sign, which the diary announces); a structure's legs carry its size, and a block stating
none of its type's amounts refuses a position other than one unit by name.

`mapping['Instrument']` also publishes `containers` — the deal types that accept `Children`
(`Deal.accepts_children` emitted into the store), so a client can tell a leaf from a structure
without importing the engine. `mapping['Structure']` publishes the desk's quotable structures —
the vernacular a salesperson says, the parameters, the legs and the recipe — which is what
`POST /book/structure` runs and a client renders a quote ticket from.

A posted job is a job *file* — the same document `load_json` reads, parsed by the same decoder, so
its `.Curve`, `.Timestamp` and `.DateList` tokens travel as themselves. Beside `Calc`, `/execute`
reads four top-level keys — `Patch`, `plan_id` (below), `lane` (`telemetry | curiosity |
standing`, default `curiosity`) and `actor`, the seat the QUEUE admits the job under where a spine
home carries a capabilities document; the lane decides the scope asked for, a standing run being
the only one that files anything. **`actor` rides every verb that queues work for a caller** —
`/book/price`, `/book/solve`, `/book/model`, `/book/xva`, `/book/setup`, `/book/structure`,
`/book/close` and `/book/settlements` — and the poll paths (the diary read, the Bloomberg tick, the
securities verification) name none and run under `DV_SPINE_ACTOR`. An `actor?` is optional only on
a home declaring no capabilities document, where an unnamed request runs under `DV_SPINE_ACTOR`;
under one, a request naming no seat is refused 422:

```json
{"Calc": {"...": "..."}, "Patch": {"FxRate.ZAR": {"Spot": 19.0}}}
```

`Patch` is a values delta, exactly what `patch_market` accepts, and it is applied *before* the
hashes are taken — so `values_hash` describes what actually ran.

**Who is asking.** The reads by node — `/book/positions`, `/book/pnl`, `/book/cash`,
`/book/collateral`, `/book/reconcile`, `/book/diary`, `/book/activity` and `/book/worklist` —
take the seat reading as `?actor=` - a seat that may act at a node sees the rows it acts on,
whichever verb it holds there: the positions and the P&L under its nodes, every total summed over
them alone, the diary and the reconcile rows of the instruments held there - their amounts the
book's own compile at the instrument's net, presentation and not a per-node compile - the calls
of the agreements a position there sits under, and the money and the strip of the books it holds
a grant at, so a node seat reads neither; one reading unnamed on a box
checking no token is the deployment's own view, the whole book. A read runs under no requester:
the diary it compiles is the hub's own cache, admitted under `DV_SPINE_ACTOR`, so no read files
a denial. This is presentation while one class key opens every body. A deployment naming a key
set (`DV_SPINE_JWKS`, with `DV_SPINE_ISSUER` and `DV_SPINE_AUDIENCE`) has every request but the
doorbell carry `Authorization: Bearer <ID token>`, verified locally: the token's subject IS the
seat, a stated `actor` that differs is refused, and none or a bad token is a 401; a key set
that does not read refuses the service at startup and a request by name.

**The envelope, from the service.** `/schema` describes what goes *in* a `Calculation` block, a
`Price Factors` block and a deal; it cannot describe where those blocks go, and that is not
guessable from them — market data lives under `MergeMarketData.ExplicitMarketData` (or behind a
`MarketDataFile` path instead of it), and a deal is a `.Deal` token inside
`Deals.Deals.Children[].Instrument`, nested by each node's own `Children`. `GET /schema/job` serves
a minimal job with exactly that shape, and it is a job rather than a description of one: post it to
`/validate` or `/execute` unedited and it loads, validates clean and prices.

**Plan then patch.** `POST /prepare` parses a job, names it by its `plan_hash` and keeps the parse
under that name, so `/execute` takes either the whole document or the plan:

```json
{"plan_id": "89db21b1…", "Patch": {"FxRate.ZAR": {"Spot": 19.0}}}
```

`/validate` and `/describe` take `{"plan_id": …}` the same way. An unknown `plan_id` is a `404`.
Content addressing does not care how the job arrived: a plan-id execute with no patch reports the
same `result_id` as a full-document execute of the same job. The cache is bounded (32) and
least-recently-used, and every read of it is a deep copy — two executes off one plan cannot
contaminate each other or the plan they came from. What is cached today is the **parse**; caching
the compile arrives behind the same verb with the live refill, and a client written against this
one does not move.

**Always a result_id.** There is no sync/async split at the API level. `/execute` answers
immediately with an id and a status for every calculation, and a base valuation is simply `done` by
the first poll. That is the one contract an Excel RTD cell and a browser poll loop can both be
written against.

**One compute lane.** All pricing goes through a single background worker. There is no cpu lane,
because a base valuation *is* a Monte Carlo for an autocall or a TARF book; device selection stays
where it already is, in the engine. What the queue orders is cost CLASS, read off
`Calculation.Object`: a base valuation jumps a simulation among the jobs still **waiting**, within a
class it is first in first out, and a running job is never preempted. `/schema`, `/schema/job`,
`/validate`, `/describe` and `/prepare` run nothing, so they answer inline and never reach the
queue. `/describe` reports that class under `cost`, with a crude size estimate beside it —
`Batch_Size × Simulation_Batches ×` the segments `Time_Grid` declares, which is a proxy for the
scenario grid and labelled as an estimate because that is what it is.

**One job, one execution.** `result_id` is the content hash of the replay tuple `(plan_hash,
values_hash, engine_version, seed)`. Submitting the same job twice returns the same id without
re-running — while it is still queued, while it is running, and after it has finished — so dedupe
and retry-idempotency are one feature, and two clients patching to the same market share a result.

**Never the whole cube.** A `done` result is a SUMMARY: the four replay coordinates, and every
table the run produced named with its row count and column labels. No cells — a credit Monte Carlo's
exposure is dates by scenarios and does not fit in an answer anyone wants to hold. A client reads
the shapes and fetches the one table it is showing:

```
GET /results/{result_id}
{"status": "done", "plan_hash": …, "values_hash": …, "engine_version": …, "seed": 1,
 "tables": {"mtm": {"rows": 240, "columns": ["Reference", "Object", "Value", …]}}}

GET /results/{result_id}/mtm?offset=0&limit=100
{"name": "mtm", "rows": 240, "columns": [...], "offset": 0, "index": [...], "data": [[...], …]}
```

`rows` and `columns` are the whole table's; `data` is the page. `limit` defaults to the rest of the
table and an offset past the end is an empty page. A group of tables — `cashflows`, `scenarios` —
is flattened to the path that names each one (`cashflows/ZAR`), because a group has no page. An
unknown table, like an unknown result, is a `404`. An `error` result carries the message the run
failed with, and nothing else.

**Generating a client.** The service publishes its own OpenAPI document at
`http://localhost:8000/openapi.json`, with a summary and a description on every endpoint. An SPA
generates its TypeScript client straight from it — `npx openapi-typescript
http://localhost:8000/openapi.json -o src/api/derivus.ts` for types alone, or
[openapi-generator](https://openapi-generator.tech/) (`-g typescript-fetch`) for types plus a
fetch layer. The result payloads are deliberately typed as objects rather than pinned to response
models: a `Results` tree's tables are named by the calculation that ran, so a schema restating them
would be a second copy to keep in step.

**Tokens where named, open CORS.** The service is a **trusted-network** deployment. Unless a key
set is named (above) nothing authenticates a request, and `Access-Control-Allow-Origin` is `*` so a
browser client can call it at all; put it behind something that terminates TLS, or narrow the
origins with `DV_Service --origin https://app.internal` (repeatable). Budget caps are on the
roadmap, not in the service.

---
