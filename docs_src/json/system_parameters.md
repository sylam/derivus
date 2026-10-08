# System Parameters

Global parameters that apply across the entire calculation. These live in the market-data section
of the JSON, alongside `Price Factors`, `Price Models`, etc.

- **Base_Currency** — Defines the base currency against which all calculations will internally use
  (usually `USD`). Any FX rates loaded are interpreted against this currency.
- **Base_Date** — Default valuation date (i.e. the calculation `Run_Date` if no override is
  supplied). If `null`, the current system date is used at run time.
- **Exclude_Deals_With_Missing_Market_Data** — `Yes` (default) or `No`. When `Yes`, a deal that
  references price factors not in the market data is dropped from the calculation and counted
  under `Deals Skipped`. When `No`, it is kept and valued at zero on every date, and counted the
  same way; the run never refuses for it.
- **Correlations_Healing_Method** — How non-positive-definite correlation matrices are repaired:
    - `Eigenvalue_Raising` (default) — small negative eigenvalues are floored to a tiny positive
      value, then the matrix is rescaled to keep ones on the diagonal.
    - `Alternating_Projections` — iteratively projects onto the cone of positive-semidefinite
      matrices and onto the unit-diagonal subspace until convergence.

```json
{
  "System Parameters": {
    "Base_Currency": "USD",
    "Base_Date": null,
    "Exclude_Deals_With_Missing_Market_Data": "Yes",
    "Correlations_Healing_Method": "Eigenvalue_Raising"
  }
}
```
