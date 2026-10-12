# System Parameters

Global parameters of the market data, beside `Price Factors`, `Price Models` and the rest.

| Field | Default | Meaning |
|---|---|---|
| `Base_Currency` | blank | the currency the engine values in internally; an `FxRate` of it is never simulated |
| `Base_Date` | blank | the date the market data stands at: the bootstrappers price their quotes as of it, and it is the date a runner falls back to when the context holds no `Calculation` block. Blank or `null` in a loaded market-data file becomes today's date at load. A calculation's own date is its `Base_Date` (see [Calculations](calculations.md#defaults)) |
| `Exclude_Deals_With_Missing_Market_Data` | `Yes` | `Yes` drops a deal that references price factors the market data lacks and counts it under `Deals Skipped`. `No` keeps it, valued at zero on every date, and counts it the same way; the run never refuses for it |
| `Correlations_Healing_Method` | `Eigenvalue_Raising` | how a correlation matrix that is not positive definite is repaired (below) |

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

## Correlation healing {#correlation-healing}

The assembled matrix $C$ (see [Correlations](correlations.md)) is repaired while any eigenvalue is
below $10^{-8}$, with a WARNING line each time; the size of the change is not reported.

| Method | One repair |
|---|---|
| `Eigenvalue_Raising` | $C = V\Lambda V^\top$; floor the eigenvalues, $P = V\max(\Lambda, 10^{-4})V^\top$; rescale to a unit diagonal, $C' = D^{-1/2} P D^{-1/2}$ with $D = \mathrm{diag}(P)$ |
| `Alternating_Projections` | start $B_0 = C_0 = C$. For $k = 0, \dots, 99$: $X_k = V\max(\Lambda, 10^{-4})V^\top$ from the eigendecomposition of $B_k$; $C_{k+1}$ is $X_k$ with its diagonal set to one; $B_{k+1} = B_k + (C_{k+1} - X_k)$. Stop when $\max\lvert C_{k+1} - C_k\rvert < 10^{-8}\max\lvert C_{k+1}\rvert$ or after 100 iterations |

The second follows the alternating-projections idea of Higham (2002) between the positive
semidefinite matrices and the unit-diagonal ones, with three departures: the eigenvalue floor is
$10^{-4}$ rather than 0, the correction carried between iterations is the one above rather than
Higham's Dykstra correction, and the iteration is capped at 100. Neither method is guaranteed to
return the nearest correlation matrix to $C$, and the healed matrix is not differentiated (correlation is not an AAD leaf).

## References

- N. J. Higham, "Computing the nearest correlation matrix--a problem from finance", *IMA Journal of
  Numerical Analysis* 22(3), 329-343, 2002. [doi:10.1093/imanum/22.3.329](https://doi.org/10.1093/imanum/22.3.329)
