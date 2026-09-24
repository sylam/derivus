# API

Everything in Derivus is based off a *Context*. All calculations are constructed with reference
to one.

!!! note "Curve convention"
    Interest rate curves typically start one day from now, i.e. $1/365\approx 0.00274$. A knot at
    time 0 is redundant rather than forbidden — the discount factor there is 1 by identity — so a
    curve may carry one (the job skeleton and the shipped fixtures do), and a curve without one
    flat-extrapolates its first timepoint back.

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

Three calculation types are supported, each with both an explicit method and a JSON-driven
dispatcher:

| Method | JSON `Calculation.Object` | Purpose |
|---|---|---|
| `cx.Base_Valuation(overrides)` | `BaseValuation` | Single-point MTM revaluation |
| `cx.Credit_Monte_Carlo(overrides)` | `CreditMonteCarlo` | Path-dependent simulation (CVA / FVA / PFE) |
| `cx.Hedge_Monte_Carlo(overrides)` | `HedgeMonteCarlo` | Same simulation engine, used to solve a dynamic hedging problem (DiffSolver) |
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
the deals loaded and any tags defined; `out['Results'].keys()` enumerates everything that's
available.

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

!!! danger "`runparallel=True` does not work at HEAD"
    `Credit_Monte_Carlo(runparallel=True)` is meant to shard the simulation across all visible CUDA
    devices and merge the results. It cannot: the worker spawn passes **seven** positional arguments
    to six-parameter `run_cmc`, so every child raises `TypeError` before running and the parent then
    blocks forever on `results.get()`. The branch also returns a dict rather than the `(calc, out)`
    tuple. Do not call it until the stray argument is dropped from the call site.

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
{'deals': {'FWD1': ['Sell_Discount_Rate is required']}, 'factors': ['InterestRate.ZAR']}
```

- `'deals'` — the authoring messages of every deal in the book, keyed by the deal's `Reference`
  (the walk position, as `#3`, where that is blank or repeated). These are the rules a deal states
  about itself: a field it cannot price without, a value that is not what that field declares —
  a number, a `{".Timestamp": ...}` date, one of a menu's own strings — or a rule spanning
  several of its fields.
- `'factors'` — every price factor the book names that the market data has no `Price Factors` block
  for, spelled as the key you would add. A factor with no block is never built, so the deals that
  reference it are dropped from the portfolio when the run reaches them.

Both empty means nothing here can tell you the job will fail. A message never stops a deal pricing:
the engine still fails exactly where it always failed, and this says so first.

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
  build, and `missing` is the want-list `validate()` returns on its own.
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

- `Run_Date` / `Base_Date` — switch the valuation date without editing the JSON
- `Currency` — change the reporting currency
- `Random_Seed`, `Batch_Size`, `Simulation_Batches` — control Monte Carlo reproducibility and size
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
Monte Carlo returns the same three plus a fourth. (A hedge run returns a
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

| | | |
|---|---|---|
| `GET` | `/schema` | `schema.mapping` plus `engine_version` — what a front end renders panels, tables and enums from |
| `GET` | `/schema/job` | the job ENVELOPE those declarations sit inside, as a skeleton that loads |
| `POST` | `/validate` | `cx.validate()` over the posted job, verbatim |
| `POST` | `/describe` | `cx.describe()` plus what the queue would make of the job |
| `POST` | `/prepare` | `{"plan_id": …, "values_hash": …, "engine_version": …}` |
| `POST` | `/execute` | `{"result_id": …, "status": …}`, plus `attested` where the job's lane attests and the result is already `done` |
| `GET` | `/results/{result_id}` | `{"status": …}`, and when done the replay tuple, the run's `stats`, and the SHAPE of each table |
| `GET` | `/results/{result_id}/{table}` | one table, `?offset=&limit=` |
| `GET` | `/ui` | a built web UI - the wheel's own by default, or the `DV_Service --ui <dir>` build |
| `GET` | `/book` | the live job document the service serves — `DV_HOME/book.json` by default (a missing file starts blank), `DV_Service --book <file>` for another, `--no-book` to serve none and 404 every `/book` verb — with the etag naming its state |
| `GET` | `/book/status` | the desk in ONE read, composed from the readers beside it — the book's `base_date`, `base_currency` and calculation, its deal count and netting sets, every curve with the knots it solved on, the scheme it is built under, the latest print its rows carry and any benchmark held out, the vol surfaces with their quote stamps, the spot models calibrated onto it, the last XVA per set trimmed to reference/status/as_of/cva/fva, `spine` — the LSN the book file was hydrated at and how far the record has moved since, in events and in the fills and amendments among them, and nothing that folds — and `terminal` — whether `blpapi` imports on this workstation (no session opened), the `--tick` cadence, and whether a security map has been verified |
| `POST` | `/book/deals` | book, amend or delete one deal — validated BEFORE an atomic write; a refusal is `{"written": false, "refused": […]}` and touches nothing. A `reference` beside the `deal_path` of an amendment or a delete names the deal the client read there, and a path another host's write has moved refuses by name |
| `POST` | `/book/price` | price the book plus an optional candidate deal — a what-if; writes nothing, and the candidate is validated BEFORE it queues, a 422 naming any market data the book lacks rather than a run that drops it |
| `POST` | `/book/solve` | solve one field of a candidate deal to a target value — a root find over base valuations; the solved coordinates arrive under the result's `stats.Solved` |
| `POST` | `/book/market` | tick the book's market: quote blocks installed or value-updated (structure refused), a `patch_market`-shaped values patch, the bootstrap run — one atomic write, refused whole if the bootstrap complains. A values tick bootstraps the blocks it moved and every block that reads one of them, named back under `bootstrapped`; the whole market where a block arrived authored, where the patch names a factor a family reads, or where nothing moved at all |
| `POST` | `/book/curve` | set a curve up from its benchmark rows — `{curve, currency, rows: [{tenor, security?, quote?, use?}], interpolation?}` plus any convention the seed's entry leaves out; each tenor says what its instrument IS (`ON` and the declared front a deposit, `1Mx4M` a FRA, `6M1M` a forward-starting swap, anything else a spot swap), the block is re-authored whole and the market re-bootstrapped in one atomic write, and the answer names the date it was authored on, the block, its knots and the factors the run rewrote. `interpolation` is the curve's OWN scheme, written as its `id` rule in `Price Factor Interpolation` in the same write and before the solve, blank clearing it and an absent one leaving it as it stands. THE SNAP SETS THE DATE: a row priced off the terminal carries the print's own clock, and where that is later than the day the book stands at the book rolls onto it and every other curve block is re-authored there too |
| `GET` | `/book/curve` | the book's curve blocks read back as definitions — rows, conventions and the `interpolation` each is built under with `interpolation_source` saying whether a rule named the curve or it took the type's own, in the shape the POST takes — with `base_date`, the day every block is authored on, and each block's own `snapped`, the latest print its rows carry; and, with no `?curve=`, the seed's own curve entries a desk could set up |
| `POST` | `/book/date` | set the book's calculation date — `{base_date}`, an ISO day or the wire stamp; both spellings of the date move together, every `InterestRatePrices` block is re-authored on the new day from its own rows and conventions with no terminal asked, and the whole market is re-bootstrapped in one atomic write, refused whole if the bootstrap complains. The tick and the curve verb roll the date FORWARD onto the day their quotes were snapped; this is the one that goes anywhere |
| `POST` | `/book/bloomberg` | provision the security map (first use creates `DV_HOME`, copies the packaged seed, verifies every candidate against the terminal), fetch the desk's FX vol surfaces, re-value every curve block's rows off the securities they name, re-cross every `FxRate.<ccy>.Spot` whose currency routes to a verified pair, its print checked for freshness with the surfaces' (answered under `spots`, a currency with no route named under `unrouted` and unmoved, a late or dead print refusing the whole trip by name), roll the book onto the day those prints were snapped where that is later than the day it stands at, and tick the book — a queued job whose `/results/{id}` carries `progress` while it runs |
| `GET` | `/book/dependencies` | what the deals under one node of the book need from the market — `?deal_path=`, no path being the whole book: the engine's own factor walk over that subtree, no model resolution, no pricing, no write. Every factor with its `status` and, for a missing one, the `supply` that would fill it — the seed's `block` and `key`, the `securities` it spells, the `verified` the map holds, `conventions` for a curve — or `supply` null with a `note` where this desk's vocabulary spells nothing for it |
| `POST` | `/book/dependencies` | the same walk over a CANDIDATE — `{deal, parent_reference?}`, spliced the way `/book/price` splices one and answered for that subtree alone, so it is what THIS trade needs rather than what the book happens to lack. The booking's own authoring verdict runs first and refuses 422 in its words, a deal whose `Object` names no type reaching nothing to walk; a missing or null `deal`, and a `deal_path` beside one, refuse by name. Writes nothing |
| `POST` | `/book/setup` | build the market a trade needs — `{pair?, deal?, parent_reference?, deal_path?}`, exactly one or none for the whole book's want-list; a queued job on the tick's executor and cost class. A pair's want-list is the WALK of a vanilla option on it, so each leg reads the curve its own `FxRate` block discounts on. It discovers only the names the map has never heard of for the supplying entries, its rejected ledger asked again — THE MAP IS WRITTEN AS IT IS GATHERED, before the book, so a refused set-up leaves a grown map and an unmoved book, and a map with nothing to add is not rewritten — then checks FRESHNESS for every wanted surface and spot before fetching anything, one late print refusing the whole trip by name, and installs the surface, each new currency's crossed spot and each new curve (a refused benchmark held out by name) in ONE atomic write that the bootstrap judges. WHERE THE PRINTS MOVE THE BOOK'S DATE the market it already carries is re-priced in the same session and lands in the same write, through the one fetch `POST /book/bloomberg` is, so a book never holds two days of quotes under one date. NOTHING LANDS THAT WOULD DEPEND ON A BLOCK THE WRITE WILL NOT CARRY: a curve nothing can supply - no seed entry, no conventions, a strip the screen leaves under the emitter's floor - is a `not_supplied` row carrying the reason, and a NEW CURRENCY IS INSTALLED AS A PAIR OR NOT AT ALL: its curve values its own benchmarks in the book's base, so spot and curve are held with each other and a surface with whichever leg is not coming, each row naming what it waits on, and what is left lands. The outcome carries at least `{written, installed, discovered, held_out, refused, not_supplied, check}` under `stats.Setup`, a write adding the edit's own: `refused` is what refused the WRITE and only ever rides `written: false`, `not_supplied` is `{factor, reason}` for every want that could not be filled, `installed` is exactly what was written, and `written` is true iff something landed. No terminal, no `Bootstrapper Configuration`, a pair the seed does not spell (one stated backwards being told the spelling it carries) and a `deal` the booking verb would refuse each refuse 422 at submission |
| `GET` | `/book/securities` | the desk's ticker vocabulary and its terminal evidence — `seed`, the candidates this desk could quote (the packaged questionnaire with the desk's own file over it); `map`, what a terminal verified, each entry carrying the NAME it answered, its last print and when, with a `rejected` ledger keyed by ticker; `provisioned` and `home`; and `used`, the IPV join — every `InterestRatePrices` row with its tenor, security, quote and timestamp and that security's evidence beside it, the verdict that rejected it, or `unmapped`. `?block=` narrows the seed and the map's blocks |
| `POST` | `/book/securities` | set one entry of the desk's own seed — `{block, key, entry}`, a null `entry` removing it; validated by SPELLING every candidate the merged file now names, so a malformed curve spec refuses by name with the file untouched, written atomically to `DV_HOME/seed.json` with the file it replaced kept as `seed.json.bak_<stamp>`, and the answer naming the tickers that block now spells. The packaged questionnaire is never written |
| `POST` | `/book/securities/verify` | re-verify the named scope against this workstation's terminal — a queued job on the tick's own executor and cost class: every map entry in scope re-probed and its drift named by path, every seed name the map has never heard of probed once and ledgered, every rejected name in scope asked again and one that prices now moved into the map under `revived`, the map rewritten atomically, the outcome under `stats.Securities`. No terminal refuses 422 by name at submission |
| `POST` | `/book/model` | calibrate one pair's spot-model parameters off the surface the book already carries — `{pair, family}`, the family defaulting to the one the runner pins, fitted against that family's own vega-weighted ladder; queued at the heavy cost class, ON REQUEST and never on the tick, because a tick moves the surface and structurally leaves these parameters where they were. Call it after a re-tick and before quoting the TARFs that read them. `/book/hn` is the same verb under the spelling it shipped as, kept for one release |
| `POST` | `/book/structure` | quote a named structure against the book — the declared recipe solved server-side on the live spot where the terminal is up, each leg on the side of any two-way the book carries, the pending trade and its ticket filed under the quote id in `DV_HOME/tmp`. THE RECORD DOES NOT MOVE: a quote is recorded when the client accepts it, so this runs in the curiosity lane and the quotes nobody accepts die in `tmp`. Under a spine home the pending file carries what that acceptance will file — the book's two hashes, the values vector behind them, the `ticket` an approval would sign, the age of the oldest stamped pillar, `quoted_by` and the relayed client `request` |
| `POST` | `/book/quote` | ACCEPT a quote already given and book it — `{quote_id, actor?}`, which books the MIRROR of the pending deal (a quote is client paper; a book holds the bank's position), validated and refused exactly as a booking is; the pending file survives as the audit trail and gains `accepted {lsn, ticket}` and `booked {lsn, deal_path}`, so a second acceptance answers `{written: false, booked, note}` naming where the trade landed rather than reporting the move the booking itself made. Under a spine home, in order, all INSIDE the edit closure against the document the trade lands in and all before anything appends: the desk's own `firm_seconds` (outside it — that one is about the quote), the PLAN the charge was solved against (a moved book refuses), the PILLAR age of the board the quote was struck on against any declared `pillar_seconds`, and the TICKET re-derived from the pending deal and this book; the MARKET is REPORTED as `{pinned, current, moved}` and never refused. Then one write — `quote_filed` under the acceptor's seat, the tier step where a `tiers` policy is in force, the `fill` under the quote id, the file — and a tier that holds the booking back answers `{written: false, accepted, tier, waits_on}`, or, where no tier admits the ticket, `{written: false, accepted, refused: [...]}`. THAT LAST WEARS THE VALIDATION REFUSAL'S SHAPE and is told from it by `accepted`: a validation refusal touched nothing, a tier refusal is a price the client took that the desk's own policy will not book. The acceptance stands either way |
| `POST` | `/book/quote/approve` | sign the ticket an accepted quote minted — `{quote_id, actor}`, filed over the plan this quote would leave the book at, so it reaches this quote and no other; a quote nobody accepted refuses by name and an unscoped seat in the record's own words, with the denial landed. `{recorded: {lsn}, ticket}`, and signing twice is one fact |
| `POST` | `/book/quote/reject` | refuse that ticket with the reason on the row — `{quote_id, reason, actor}`. A verdict is never withdrawn, so what stands is what was filed LAST |
| `GET` | `/book/quote/{quote_id}` | the pending trade filed under that id — the quote as answered, the deal that books it and `quoted_at`, read back from `DV_HOME/tmp`; the id is a file name and never a path out of it, so one carrying a separator refuses 422 and an id nothing stands under 404s naming the directory. The file survives the acceptance and gains `accepted: {lsn, ticket}` from it, so this reads a quote given and a quote booked alike |
| `GET` | `/book/quote/{quote_id}/sheet` | the quote sheet itself, the `.xlsx` streamed under its own media type, so a client not on the service's machine is handed the workbook rather than a path. A quote given where the sheet writer was absent 404s carrying that quote's own `sheet_note`, which names the install |
| `GET` | `/book/risk` | the book's CONSOLIDATED risk — one base valuation with `Greeks: 'First'` over the whole book, counterparty-blind, computed on a miss and cached under the `etag` of everything the run reads; `{as_of, etag, currency, mtm, per_deal, greeks, quotes, quoted, quote_note}` - `quotes` the risk per unit of each quote the book's factors are built from, refitted in the run with `Quote_Sensitivity` on, `greeks` per unit of each factor, an empty book zeros with no run, a book that will not price a 422 naming the cause |
| `POST` | `/book/xva` | recalculate the XVA projection — `{"netting_sets": […] \| null}` queues ONE credit Monte Carlo per netting set, CVA on against the counterparty the set's own `Credit_Support_Amounts` names, each writing its own row; `{"queued": [{"reference", "result_id"}]}`, and an unknown reference refuses by name having queued nothing |
| `GET` | `/book/diary` | every payment, fixing and expiry the book's deals carry — the COMPILE's own schedule re-emitted, one row per leg position with its due date, currency, notional, the amount where the compile determines one (`null` and `determined: false` where it does not), and the derived key a settlement fact names it by; `?due_before=YYYY-MM-DD` trims it, and under a spine home each row carries what the record answers — `settled` against its key, and the source whose print satisfied a fixing. Cached under the etag of everything a compile reads and computed on the compute queue |
| `GET` | `/book/close/check` | `?date=YYYY-MM-DD` — whether a close on that day is legal and what it waits on: a fixing no declared source has printed, a payment no settlement was filed against its key, and an expiry whose terms vest a choice nobody elected; the date is parsed, so a time, a year or a garbage string refuses 422 by name; declares nothing, and 404 where no spine home is configured |
| `GET` | `/book/reconcile` | where the book file and the record disagree — the record folded AT THE HEAD against the file's own deals, so a booking whose file write never landed is visible; `in_record_not_in_file`, `in_file_not_in_record` and `quantity_mismatch` with every row named by its instrument address, beside `events_behind` and `positions_behind` counting how far the record has moved since the file was written; a READING and never a refusal, 404 where no spine home is configured |
| `GET` | `/book/activity` | the record's own strip — one line per event, newest last, beside the `lsn` to ask again from: `{lsn, rows: [{lsn, record_time, effective_time, actor, event_type, book, summary}]}`. The fold opens NO BODY, so it reads every type, a type the declared table has no sentence for renders its own name rather than dropping out of the sequence, and a replica holding no key still answers. A PAGE WALKS FORWARD: with `?since=` it is the OLDEST `?limit=` rows after that position and the `lsn` is the LAST ROW DELIVERED, so a reader asking again with the cursor it was given reaches every event in turn; with no `?since=` it is the NEWEST `?limit=` (200) rows — a strip's first paint — and the `lsn` is the head the fold reached. The fold advances the strip's own `(lsn, state)` pair, and since the log parses its segments to reach that position a page saves the rows and not the read. An empty `?since=` is no `?since=` at all while `?since=0` walks from genesis, one past the head answers the head with no rows, a negative `?limit=` answers none, and anything that is not a position refuses 422 by name. 404 where no spine home is configured, and NO BOOK IS ASKED FOR — a replica carrying a home and no book still reads |
| `GET` | `/book/markets` | the record's markets at the head — `{lsn, names, closes, snapshots}`: the official close standing per market carrying the LSN of the close it RESTATED (a close is superseded by a new close rather than corrected in place), the names a values vector was declared under, and the snapshots registered against a book. A `values_hash` is the address of the vector itself. 404 where no spine home is configured and no book is asked for; a home whose class key is gone answers the record's own sentence as a 422, never a 500 |
| `POST` | `/book/markets` | point a market NAME at the values the live book is carrying — `{name, actor}`, answering `{recorded: {lsn}, name, values_hash}`. Officialness is a property of the name: `official` moves onto a vector only by a declaration from a `mark`-scoped seat, and a `private/<subject>/<name>` board must name the seat declaring it. The record's refusals reach the caller as 422 in its own words, the denial landed as a fact; 404 where no spine home is configured |
| `POST` | `/book/close` | declare the official close over the live book's values — `{market?, date?, actor}`, `market` defaulting to `official` and `date` to the book's own `Calculation.Base_Date`, parsed as `/book/close/check` parses its `?date=`. IT RUNS BEHIND THE CHECK: the same verdict, taken over the document THIS close is struck on in one read of the book and over a diary compiled under the CALLER's seat, so a day the check calls illegal refuses 422 naming what it waits on with nothing appended. A `private/<subject>/…` market is refused for anyone but its own subject, as a mark is; a second close on one market SUPERSEDES the first — `{recorded: {lsn}, market, date, values_hash, supersedes_lsn}`. 404 where no spine home is configured |
| `POST` | `/book/settlements` | the settlement file for one day — `{due_before, actor?}`, with no default on `due_before` because a settlement file is struck FOR a day. IT NAMES NO MARKET: the board it is struck on is the one the `tiers` policy DESIGNATES for `settlement_export`, so a home designating nothing, or designating a name nothing stands under, refuses 422 at submission with the declaration that fixes it and before a row is compiled. The rows are the DIARY's, compiled and cached exactly as `/book/diary` compiles them but admitted under this request's seat BEFORE the cache is read, and an undetermined amount or a row naming no currency refuses by name; answers `{values_hash, market: {name, values_hash, lsn}, due_before, totals, rows, count}`, where `values_hash` is the board the file was struck on and `market` is the name it was resolved under. 404 where no spine home is configured |
| `GET` | `/spine/frames` | the record's own frames, VERBATIM - what a replica pulls: `{head, frames}` with every frame the twelve fields it is on the platter, `body` the base64 ciphertext, and no body opened, so it answers on a crypto-shredded home exactly as it answers on the hub. `?since=` is the LAST LSN DELIVERED, as the strip's cursor is, `?limit=` caps the page at 500 by default, and an empty page is a replica that is up to date. `head` is where this read saw the record and NEVER the next cursor - it is what a ONE-SHOT sync stops at, since a hub that keeps writing never shows an empty page. Not the strip: `/book/activity` serves six of the twelve fields and every one of the six it drops is something `SpineLog.accept` asks for. Anything that is not a position refuses 422 by name; 404 where no spine home is configured, and NO BOOK IS ASKED FOR |
| `GET` | `/spine/blobs/{hash}` | the bytes filed under that address, re-hashed on the way out, as `application/octet-stream` - self-verifying by hash, so the serving side needs no trust and a replica checks what it got whatever route it came by. Served to a seat the capabilities document admits to READ the blob's class (firm for everything while classification is dormant) under `?actor=`, else `DV_SPINE_ACTOR`; a home declaring no document serves everyone, and a seat outside the `read` rows or a read nobody signed is the record's own sentence as a 422, as is an address this home does not hold. 404 where no spine home is configured |
| `GET` | `/spine/doorbell` | a `text/event-stream` carrying `{lsn, head}` and nothing else on every move of the record's head, plus a comment on a fixed cadence so a proxy does not close an idle one. A NOTIFICATION AND NEVER A DELIVERY: the position is the whole payload, so nothing of a body reaches the wire, and a beat dropped is covered by the next, a beat repeated is one empty pull and a beat behind the head is a pull that answers nothing. The trigger is the writer's own append, so nothing here polls the log, and the stream is an ASYNC generator - an open one costs a task rather than one of the worker threads every other verb on this service shares, and a client that goes away is unregistered when its stream closes. 404 where no spine home is configured |
| `GET` | `/book/xva` | the XVA projection as it stands — `DV_HOME/xva.json`'s rows joined with the book's current set list; a set with no row reads `never run`, a row whose set has left the book carries a `note`, and a recalc in flight rides under `recalc` |

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
securities verification) name none and run under `DV_SPINE_ACTOR`:

```json
{"Calc": {"...": "..."}, "Patch": {"FxRate.ZAR": {"Spot": 19.0}}}
```

`Patch` is a values delta, exactly what `patch_market` accepts, and it is applied *before* the
hashes are taken — so `values_hash` describes what actually ran.

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

**No auth, open CORS.** The service is a **trusted-network** deployment. There is no
authentication, and `Access-Control-Allow-Origin` is `*` so a browser client can call it at all;
put it behind something that terminates both, or narrow the origins with `DV_Service --origin
https://app.internal` (repeatable). Auth and budget caps are on the roadmap, not in the service.

---
