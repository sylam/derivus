# Correlations

The correlations between the Gaussian innovations of the simulated processes. Each pair is stated
once, nested by its first name, since the matrix is symmetric:

```json
{
  "Correlations": {
    "HWInterestRate.ZAR-SWAP.F1": {
      "HWInterestRate.USD-MASTER.F1": 0.2,
      "LognormalDiffusionProcess.ZAR": 0.5
    },
    "HWInterestRate.USD-MASTER.F1": {
      "LognormalDiffusionProcess.ZAR": -0.1
    }
  }
}
```

Here the one-factor Hull-White innovation of `ZAR-SWAP` has correlation 0.2 with that of
`USD-MASTER` and 0.5 with the lognormal process of `ZAR` (an `FxRate`), and `USD-MASTER` has -0.1
with `ZAR`.

## Correlation names

A row is named `<correlation name>.<factor name>`, followed by `.<sub-factor>` for a process with
more than one innovation. The name is the process's (`correlation_name` in
`derivus/stochasticprocess.py`), not the factor type's:

| Price model | Correlation name | Rows per factor |
|---|---|---|
| `GBMAssetPriceModel` | `LognormalDiffusionProcess` | one, no suffix |
| `GBMAssetPriceTSModelImplied` | `LognormalDiffusionProcess` | one, no suffix |
| `GBMPriceIndexModel` | `LognormalDiffusionProcess` | one, no suffix |
| `LogVar2FJImpliedSpotModel` | `LognormalDiffusionProcess` | four: no suffix (the return's Gaussian given the mixer), `S` (the fast log-variance shock), `L` (the slow log-variance shock), `G` (the mixer, through its normal quantile) |
| `HullWhite1FactorInterestRateModel` | `HWInterestRate` | one, `F1` |
| `HullWhite2FactorImpliedInterestRateModel` | `HWImpliedInterestRate` | two, `F1` and `F2` |
| `PCAInterestRateModel` | `InterestRateOUProcess` | one per eigenvector the model's parameters carry, `PC1` to `PC`$N$ |
| `HWHazardRateModel` | `HullWhiteProcess` | one, no suffix |
| `CSForwardPriceModel` | `ClewlowStricklandProcess` | one, no suffix |
| `CSImpliedForwardPriceModel` | `ClewlowStricklandProcess` | one, no suffix |
| `LogOUSpotModel` | `OULogSpotProcess` | one, no suffix |
| `MarkovHMMSpotModel` | `MarkovHMMSpotProcess` | one, no suffix |
| `GARCHSpotModel` | `GARCHSpotProcess` | one, no suffix |
| `QuadraticCarryCurveModel` | `QuadraticCarryCurveProcess` | two, `L` and `D` |
| `ChainedBasisModel` | `ChainedBasisProcess` | one, no suffix |
| `FixingBridgeModel` | `FixingBridgeProcess` | one, no suffix |
| `BasisLinkedSpotModel` | `BasisLinkedSpotProcess` | one, no suffix |

So a LogVar2FJ equity `SPX` contributes `LognormalDiffusionProcess.SPX`,
`LognormalDiffusionProcess.SPX.S`, `LognormalDiffusionProcess.SPX.L` and
`LognormalDiffusionProcess.SPX.G`.

## How the matrix is assembled

| Rule | What the code does |
|---|---|
| A pair the section does not state | reads 0.0 |
| A stated name no simulated factor answers to | ignored; one INFO line lists such names |
| Two rows with the same name | correlation 1.0. This happens when two factors of different types share a name under processes with the same correlation name (an `EquityPrice.ZAR` and an `FxRate.ZAR` both lognormal), so give such factors distinct names |
| A number's meaning | the correlation of the two Gaussian innovations. For a process whose return is not a single Gaussian innovation (stochastic volatility, regime switching, a mixed residual), the realised correlation of returns differs from it |
| A matrix that is not positive definite | repaired by `Correlations_Healing_Method` (see [System Parameters](system_parameters.md#correlation-healing)) before its Cholesky factor is taken |
| Sensitivity | the matrix is not an AAD leaf: no run reports a derivative with respect to a correlation through the tape. A base valuation's quanto or compo correlation delta is a common-random-numbers bump (`Correlation_Bump`) |
