## The output dictionary

Every calculation method returns `(calc, out)`. For a base valuation `out` is a dict with three
top-level keys, as do a scenario valuation and a credit Monte Carlo; a `SIMM` or an `SACCR` run
carries `Stats` and `Results` only; and a hedge run returns an attribute object instead (see
[API Overview](api_overview.md#hedge-monte-carlo)):

- **Netting** — the internal `DealStructure` tree, intended for developers walking the hierarchy.
- **Stats** — timing and counter statistics (deals loaded/skipped, execution time), `Files`, the
  sha256 of the bytes of the market data file and the calendar the job named (`MarketDataFile`,
  `CalendDataFile`, None where it named none) and of the `Mapping` and `Parameters` files a SIMM or
  SA-CCR calculation names, under those keys where it names them, `Quasi_Seed`, the seed every quasi-random draw ran
  on, beside the `Random_Seed` a credit or hedging Monte Carlo reports, and — when a calibration was
  ridden rather than refitted — the `Calibrations` entry naming the artifact each curve rode, which
  completes the replay identity.
- **Results** — the user-facing tables. Keys vary by calculation; `out['Results'].keys()`
  enumerates what a run actually produced.

## Base Revaluation

```python
calc, out = cx.Base_Valuation(overrides={})
```

`Results` contains:

- `mtm` — the mark-to-market of all instruments loaded, along with any tagged data; the root row
  first, every structure and deal at the root summed
- `Greeks_First` — AAD sensitivities of the portfolio by risk factor, when `Greeks` is `First` or
  `All`; `Greeks_Second`, the factor Hessian, under `All`
- `Correlation_Bump` — under `Greeks: First` with a non-zero `Correlation_Bump`, each quanto, compo
  or mid-curve correlation delta, by a common-random-numbers bump rather than AAD

## Credit Monte Carlo

```python
calc, out = cx.Credit_Monte_Carlo(overrides={})
```

`Results` may contain, depending on the sub-calculations requested:

- `mtm` — the portfolio mark per scenario per report date
- `exposure_profile` — `EE`, `ENE` and one `PFE_<p>` per `Percentile` entry (defined in
  [Calculations](json/calculations.md#exposure-profile))
- `scenarios` — the simulated factor paths themselves (`Calc_Scenarios`), one table per factor;
  under `At_Percentile` each factor along the path nearest each `Percentile` of the profile at
  every report date, with that path's `mtm` beside them
- `cashflows` — simulated cashflow ledgers per currency (`Generate_Cashflows`)
- `cva` / `grad_cva` — the credit valuation adjustment and its sensitivities; `grad_cva_hessian`
  under `Hessian: Yes`, and `CS01` when `CDS_Tenors` are given (both need `Gradient: Yes`)
- `fva` / `grad_fva` — the funding valuation adjustment and its sensitivities
- `grad_xva` — under `Gradient_Total: Yes`, the sensitivities of the adjustments asking for a
  gradient, summed, in place of `grad_cva` and `grad_fva`
- `collva`, `collva_t` / `grad_collva` and `LCH_Margin` — the experimental collateral valuation
  adjustment and initial margin, not yet specified

## Hedge Monte Carlo

A hedge run carries no `Results`: the fitted value function and the held-out verdict are the
`policy_artifact` and `evaluation_summary` of its `HedgeRuntimeExecutionResult`
([API Overview](api_overview.md#hedge-monte-carlo), [Hedging](hedging/overview.md)).

## ScenarioValuation

`Results['mtm']` holds one row per deal and structure, the root first, each structure before what it
holds: `Parent`, `Reference`, `Object`, `Value` (its value on the job's own market) and one column
per scenario, numbered as `Scenarios` lists them, each in the calculation `Currency`. A deal its
pricer could not value on a scenario reads NaN there. `Stats['Batch_Size']` is the most scenarios
a pass priced.

## SIMM

`Results` contains:

- `CRIF` — one row per sensitivity in ISDA's Common Risk Interchange Format: `TradeID`,
  `Counterparty`, `PostRegulations`, `CollectRegulations`, `ProductClass`, `RiskType`, `Qualifier`,
  `Bucket`, `Label1`, `Label2`, `Amount` (in the calculation `Currency`) and `AmountUSD`
- `Trades` — one row per trade: its netting set, product class, value, moves, rows written and any
  leg the engine skipped, in its own sentence; a trade whose own compile the engine refuses by name
  is set aside, unvalued in that sentence and writing no row
- `Margin` — each netting set's initial margin off its CRIF, as the methodology aggregates it under
  the `Parameters` file's `Horizon`, in each `Direction`: `Posted` on the rows' negation and
  `Collected` on them as written, under the `Regime` its agreement declares for that direction. One
  row per `ProductClass`, `RiskClass` (`InterestRate`, `CreditQ`, `CreditNonQ`, `Equity`,
  `Commodity`, `FX`) and `MarginType` (`Delta`, `Vega`, `Curvature`), each risk class's three
  summed as its `Total`, a product class's risk classes as its `RiskClass` `Total` and the product
  classes as the set's `ProductClass` `Total`; `Amount` in the calculation `Currency` and
  `AmountUSD`. The methodology's add-ons and multipliers are not applied
- `Margin_Rows` — per CRIF row and direction, the `Derivative` of its set's margin in the row's
  amount and the row's `Share`, the derivative times the amount (`ShareUSD` in USD); a trade's share
  is its rows', and the shares sum to the set's margin where no concentration threshold binds

`Stats['Parameters']` records the `version` the SIMM parameters file states, and `Stats` carries the
one compile's counts as a base valuation's does (`Deals loaded`, `Deals Skipped`) and, under
`Method: Bump`, the most moves a pass priced (`Batch_Size`).

## SACCR

`Results['SACCR']` contains, every amount in the calculation `Currency` [@bcbscre52]:

- `netting_sets` — one row per netting set: `Counterparty` (its reference), `Margined`, `MPOR` (in
  business days, blank where unmargined), `V`, `C`, `RC`, the add-on of each asset class
  (`AddOn_IR`, `AddOn_FX`, `AddOn_Credit`, `AddOn_Equity`, `AddOn_Commodity`), `AddOn_Aggregate`,
  `Multiplier`, `PFE`, `EAD` and `Skipped`, every trade the engine skipped inside the set named
  with its own sentence, its EAD written regardless
- `trades` — one row per component of a trade the run valued, a trade of amounts fixed today
  reading none: `TradeID`, `Component` (blank where the row reads the trade whole, else the
  `Reference` of the child of a package it reads, the field stating the last day of the
  component of a type of several, an FX swap's `Near_Settlement_Date` or
  `Far_Settlement_Date`, a caplet's or floorlet's deal `Reference` and its period's end date,
  `CAP_LEG 2027-01-15`, a strip's deal `Reference`, its fixing date and the field naming the leg,
  `TARF 2025-02-15 LeverageNotional`, or an asset-or-nothing digital's deal `Reference` and
  vanilla or digital), `NettingSet`, `AssetClass`, `HedgingSet` (an IR
  trade's currency, an inflation leg's currency and index, `USD InflationRate.CPI`, an FX
  trade's pair, a credit or equity trade's entity, a commodity trade's
  set), `Subclass` (Table 2's row), `Bucket` (an IR trade's maturity bucket, a commodity trade's
  type), `SupervisoryFactor`, `AdjustedNotional`, `Delta`, `MaturityFactor`, `EffectiveNotional`
  (a digital's capped at its payoff over its `SupervisoryFactor`, so not always the product of
  the three before it) and `Value`, the trade's on its first row and blank on the rest, so `V`
  counts each trade once

`Stats['Parameters']` records the `version` the SA-CCR parameters file states, beside the one
compile's counts.

## Over HTTP

The service returns the same tables paged: `GET /results/{id}` answers the run's replay
coordinates, its `Stats` and each table's shape, and `GET /results/{id}/{table}?offset=&limit=`
returns the cells, `format=csv` the same table as a CSV file at full precision — see [the service
verbs](api_overview.md#the-same-verbs-over-http). The web UI links each table it shows to that
file, and the Jupyter workbench writes each to a CSV with a download link (see [Jupyter
Workbench](running_calcs.md#results)).
