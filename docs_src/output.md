## The output dictionary

Every calculation method returns `(calc, out)`, and `out` is a dict with three top-level keys:

- **Netting** — the internal `DealStructure` tree, intended for developers walking the hierarchy.
- **Stats** — timing and counter statistics (deals loaded/skipped, execution time, and — when a
  calibration was ridden rather than refitted — the `Calibrations` entry naming the artifact each
  curve rode, which completes the replay identity).
- **Results** — the user-facing tables. Keys vary by calculation; `out['Results'].keys()`
  enumerates what a run actually produced.

## Base Revaluation

```python
calc, out = cx.Base_Valuation(overrides={})
```

`Results` contains:

- `mtm` — the mark-to-market of all instruments loaded, along with any tagged data
- `Greeks_First` — analytic sensitivities of the portfolio by risk factor, when `Greeks` is on
  (`Greeks_Second` appears when second-order sensitivities are requested)

## Credit Monte Carlo

```python
calc, out = cx.Credit_Monte_Carlo(overrides={})
```

`Results` may contain, depending on the sub-calculations requested:

- `mtm` — theoretical prices per scenario per time point
- `exposure_profile` — percentiles of the mtm calculation (EE and PFE)
- `scenarios` — the simulated factor paths themselves (`Calc_Scenarios`), one table per factor
- `cashflows` — simulated cashflow ledgers per currency (`Generate_Cashflows`)
- `cva` / `grad_cva` — the credit valuation adjustment and its sensitivities
- `fva` / `grad_fva` — the funding valuation adjustment and its sensitivities
- `collva` and initial-margin tables, when the collateral / IM blocks request them

## Hedge Monte Carlo

`Results` carries the solver's outputs: the fitted value-function artifact, the greedy-policy
verdict and the benchmark comparison (see the [Hedging](hedging/overview.md) section).

## Over HTTP

The service returns the same tables paged: `GET /results/{id}` answers the run's replay
coordinates plus each table's shape, and `GET /results/{id}/{table}?offset=&limit=` returns the
cells — see [the service verbs](api_overview.md#the-same-verbs-over-http). A notebook client can
also right-click any rendered table and "Save As" (see [Quick Start](quickstart.md)).
