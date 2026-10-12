# Historical Calibration

Calibration fits [price model](../json/price_models.md) parameters to historical time series. It is
distinct from [bootstrapping](../bootstrapping/general.md), which fits current market quotes for
risk-neutral simulation: a calibration estimates real-world (P-measure) dynamics, used by PFE,
capital and hedging simulations, and by the `Correlations` section every simulation reads.

This page is the chapter's index and states what every estimator shares. Each estimator has its
own section on the page of its asset class, listed below; the framework's inputs are on
[Config](config.md), the class contract on [Contract](contract.md) and the cross-factor data pulls
on [Cross Factor](cross_factor.md).

## Inputs

- **A historical archive**: a wide table, one row per business day, columns named
  `<Type>.<Name>` or `<Type>.<Name>,<sub-key>`, loaded once from `MarketDataArchiveFile` in
  [calibration_config.json](config.md).
- **A MarketData JSON**: declares the factors and routes each to a process through
  `Model Configuration` (see [Model Configuration](../json/model_configuration.md)).
- **`calibration_config.json`**: names the estimator (`Method`) and its tuning for each process.

## Estimators

| Estimator (`Method`) | Process it writes | Price factor | Archive columns | Estimator | Page |
|---|---|---|---|---|---|
| `GBMAssetPriceCalibration` | `GBMAssetPriceModel` | `EquityPrice`, `FxRate` | one spot | log-return moments, clipped | [Asset Pricing](asset_pricing.md) |
| `LogVar2FJCalibration` | `LogVar2FJImpliedSpotModel` block, read by the LogVar2FJ bootstrap | `EquityPrice`, `FxRate` | close, optional `,OPEN`/`,HIGH`/`,LOW` | Kalman MLE, RTS smoother, ridge regression, NIG MLE | [Asset Pricing](asset_pricing.md) |
| `GBMPriceIndexCalibration` | `GBMPriceIndexModel` | `PriceIndex` | one index | log-return moments | [Inflation](inflation.md) |
| `HWInterestRateCalibration` | `HullWhite1FactorInterestRateModel` | `InterestRate`, `InflationRate`, `DividendRate` | `<factor>,<tenor>` | OU regression per tenor, tenor-averaged | [Interest Rates](interest_rates.md) |
| `PCAInterestRateCalibration` | `PCAInterestRateModel` | `InterestRate`, `InflationRate`, `DividendRate` | `<factor>,<tenor>` | OU regression on log rates, PCA of the covariance | [Interest Rates](interest_rates.md) |
| `HWHazardRateCalibration` | `HWHazardRateModel` | `SurvivalProb` | `<factor>,<tenor>` | OU regression per tenor | [Survival Rates](survival_rates.md) |
| `CSForwardPriceCalibration` | `CSForwardPriceModel` | `ForwardPrice` | `<factor>,<tenor>` | OU regression on the first log column | [Energy Pricing](energy_pricing.md) |
| `QuadraticCarryCurveCalibration` | `QuadraticCarryCurveModel` | `ForwardRate` | two `<factor>,<tau>` | joint AR(1)/ARX(1) Student-t MLE | [Energy Pricing](energy_pricing.md) |
| `LogOUSpotCalibration` | `LogOUSpotModel` | `CommodityPrice` | one spot | OU regression on the log price | [Commodity Spot](commodity_spot.md) |
| `MarkovHMMSpotCalibration` | `MarkovHMMSpotModel` | `CommodityPrice` | one spot | Baum-Welch EM, moment-matched $\nu$ | [Commodity Spot](commodity_spot.md) |
| `GARCHSpotCalibration` | `GARCHSpotModel` | `CommodityPrice` | one spot | GARCH(1,1)-t MLE | [Commodity Spot](commodity_spot.md) |
| `BasisLinkedSpotCalibration` | `BasisLinkedSpotModel` | `ObservedBasis` | basis and its parent | OLS, joint ARX(1)-t MLE, or band reversion with a mixture EM | [Observed Basis](observed_basis.md) |
| `ChainedBasisCalibration` | `ChainedBasisModel` | `ObservedBasis` | basis and its chain source | OLS link residuals, bridge identities | [Observed Basis](observed_basis.md) |
| `FixingBridgeCalibration` | `FixingBridgeModel` | `ObservedBasis` | fixing basis and its parent | pooled bracket OLS | [Observed Basis](observed_basis.md) |

Each process's dynamics are on the Theory page of the same name.

## What every estimator shares

| Convention | Statement |
|---|---|
| Clock | The framework passes `num_business_days = 252` to every class. Every archive row is one business day, $\Delta=1/252$ years, whatever the calendar gap between rows. Only `FixingBridgeCalibration` (1-calendar-day pairs), `ChainedBasisCalibration` (`Max_Session_Gap_Days`) and the `Band_Mixture` fit (gaps of at most 4 days) read the gap. |
| Dating | An innovation is dated at the START of the return it belongs to (`start_of_return`), so every class's `delta` joins on one date index. |
| Measure | P. A drift is estimated only where a section says so; every other drift is the model's own default. |
| Standard errors | Only `LogVar2FJCalibration` writes them (`<name>_SE` beside each estimate). `GARCHSpotCalibration`, `QuadraticCarryCurveCalibration` and the joint `BasisLinkedSpotCalibration` fit log asymptotic errors or t-statistics at INFO and write none. |
| Output | Each calibrated factor's `Price Models` block is REPLACED whole by what the class writes, so a field the class does not write (for example a hand-declared `Drift_States`) is dropped. |
| Tests | What is gated: the innovation dating and a correlation round trip through `GBMAssetPriceCalibration` and `GARCHSpotCalibration` (`tests/test_innovation_clock.py`), the `FixingBridgeCalibration` refusal (`tests/test_loud_refusals.py`) and the declared tuning fields (`tests/test_schema_emission.py`). No test checks that an estimator recovers known parameters. |

## The shared OU regression

Seven classes (the two GBM classes, the two Hull-White classes, `CSForwardPriceCalibration`,
`PCAInterestRateCalibration` and `LogOUSpotCalibration`) read their moments from one function,
`calc_statistics`. It runs per archive column. With $x_t$ the column's level (`Diff`) or
$\log\max(x_t,10^{-4})$ (`Log`; a zero or negative value is read as $10^{-4}$), $\Delta x_t=x_{t+1}-x_t$
and $\Delta=1/252$:

$$\hat b=\frac{\widehat{\mathrm{Cov}}(\Delta x_t,x_t)}{\widehat{\mathrm{Var}}(x_t)},\qquad
\hat\alpha=\mathrm{clip}\big(-252\log(1+\hat b),\;0.001,\;\alpha_{max}\big)$$

$$\hat\sigma^2=\Big(\widehat{\mathrm{Var}}(\Delta x)-\big(e^{-\hat\alpha\Delta}-1\big)^2\widehat{\mathrm{Var}}(x)\Big)\frac{2\hat\alpha}{1-e^{-2\hat\alpha\Delta}},\qquad
\hat\theta=\overline{x}+\frac{\overline{\Delta x}}{1-e^{-\hat\alpha\Delta}}$$

This is the least-squares fit of the exact AR(1) discretisation of an Ornstein-Uhlenbeck process,
$x_{t+\Delta}=\theta(1-e^{-\alpha\Delta})+e^{-\alpha\Delta}x_t+\varepsilon_t$ (Vasicek 1977), mapped
back to $(\alpha,\sigma,\theta)$. The function also returns:

| Output | Formula |
|---|---|
| `Mean Reversion Speed` | $\hat\alpha$; $\alpha_{max}=4$ by default, 5 for `CSForwardPriceCalibration`. A sample with $\hat b\ge0$ (no reversion) lands on the 0.001 floor. |
| `Reversion Volatility` | $\hat\sigma$; at a clipped $\hat\alpha$ the same formula is evaluated there, and is then no longer the residual variance. |
| `Long Run Mean` | $\hat\theta$; under `Log`, $e^{\hat\theta+\hat\sigma^2/(4\hat\alpha)}$, the stationary mean of $e^x$, with infinities and values more than two cross-column standard deviations from the cross-column median set to NaN. |
| `Volatility`, `Drift` | $\mathrm{sd}(\Delta x)\sqrt{252}$ and $252\,\overline{\Delta x}$, the random-walk moments. |
| `delta` | $\Delta x_t$ itself, unstandardised and with no reversion term removed. |
| correlation | Pearson correlation of $\Delta x$ across the factor's columns. |

Limitations: $\hat\alpha$ from a least-squares AR(1) fit is biased upward in short samples
(Tang & Chen 2009); its size on any archive here is unmeasured. No standard error is computed.

## Output

`Config.calibrate_factors` populates two sections of the MarketData JSON in memory:

- `Price Models`: one block per calibrated factor, keyed `<ModelName>.<FactorName>`, holding what
  the class wrote.
- `Correlations`: with `a` the block-diagonal stack of every class's `correlation` matrix and
  $\rho$ the Pearson correlation of every class's `delta` columns, joined on their dates (an outer
  join; each pair over the dates both carry):

$$R = \mathrm{clip}\big(a\,\rho\,a^\top,\,-1,\,1\big)$$

Every off-diagonal entry with $|R_{ij}|$ strictly above `correlation_cuttoff` (default 0.2) is
written; the rest are absent and simulate as zero. This applies to pairs inside one model as well
(the level and shape of `QuadraticCarryCurveModel`, the four sub-factors of a LogVar2FJ history).
Three further properties:

1. $R$ is not guaranteed positive semi-definite after the clip and the cutoff. The simulation
   repairs it when it builds the Cholesky factor, by `Correlations_Healing_Method` in `System
   Parameters`: `Eigenvalue_Raising` (the default) floors the eigenvalues at $10^{-4}$ and rescales
   to a unit diagonal; the alternative runs alternating projections for at most 100 iterations
   (Higham 2002). Both log a warning.
2. `overwrite_correlations=True` (the default) clears the whole `Correlations` section before
   writing, including entries for factors this run did not calibrate.
3. Daily closes taken at different times of day bias a same-day correlation towards zero
   (Burns, Engle & Mezrich 1998). No adjustment is made; the size on any archive here is
   unmeasured.

A subsequent `Config.write_marketdata_json` call serialises the result to disk.

## Spike removal: `smooth`

`calibrate_factors(smooth=s)` with $s>0$ drops every LEVEL with $|x-\mathrm{median}(x)|>s\cdot
\mathrm{sd}(x)$, per column over the whole window, and re-fills it by linear interpolation in the
date index. It filters levels, not returns, so on a trending series it removes valid observations
at both ends of the window. The default 0 turns it off.

## Workflow

```python
from derivus.config import Config
import pandas as pd

aa = Config()
aa.parse_json('./tests/fixtures/data/MarketDataRF_platinum.json')          # static source-of-truth
aa.parse_json('./tests/fixtures/calibration_config.json')        # methods + archive

factors = aa.fetch_all_calibration_factors()                # discovered from archive
keep = {**factors['present'], **factors['absent']}          # routed via Model Configuration

start = aa.archive.index.min()
end   = aa.archive.index.max()
aa.calibrate_factors(start, end, keep, smooth=0.0, correlation_cuttoff=0.1)
aa.write_marketdata_json('./tests/fixtures/data/MarketDataRF_platinum_calibrated.json')
```

The factor list is the **union of two halves**, not a three-way intersection. `present` is every
`Price Factors` entry that `Model Configuration` routes to a model; `absent` is every archive column
prefix not already covered that `Model Configuration` also routes. Neither half checks the other's
source: `present` never tests the archive, `absent` never tests `Price Factors`.

## Skips and refusals

| Condition | Effect |
|---|---|
| A routed factor with no archive column | `KeyError` from the unguarded `self.archive_columns[...]` subscript, before any calibration runs. Trim the list first. |
| No `Method` for the factor's model | The factor is skipped: *"… is not calibrated and is skipped - CalibrationConfig names no Method for …"*. |
| The class raises | The factor is skipped, logged in the class's own words: *"… is not calibrated and is skipped - `<Class>` refused it: …"*. |
| A `correlation` entry outside $[-1,1]$, or a `delta` column with zero standard deviation | The factor is skipped: *"Data errors in factor … resulting in incorrect correlations - skipping"*. |
| The process refuses the written parameters | The framework constructs the process from the written block to read its correlation names, outside the per-factor guard, so a constructor check that fails (`GARCHSpotModel`, `QuadraticCarryCurveModel` assert their parameters) aborts the whole run. |
| Every factor skipped | `ValueError` naming them; nothing is written. |

A skipped factor writes no `Price Models` block and contributes no correlation.

## Iteration order

Factors are calibrated in alphabetical order of their fully-qualified name (`<Type>.<Name>`). A
class that needs a sibling factor's data receives it through `data_frame`, by the archive pulls of
[Cross-Factor Calibration](cross_factor.md), never through iteration order or another class's
fitted parameters.

## References

- O. Vasicek, "An equilibrium characterization of the term structure", *Journal of Financial
  Economics* 5(2), 177-188, 1977. DOI [10.1016/0304-405X(77)90016-2](https://doi.org/10.1016/0304-405X(77)90016-2)
- G. E. Uhlenbeck, L. S. Ornstein, "On the Theory of the Brownian Motion", *Physical Review* 36(5),
  823-841, 1930. DOI [10.1103/PhysRev.36.823](https://doi.org/10.1103/PhysRev.36.823)
- C. Y. Tang, S. X. Chen, "Parameter estimation and bias correction for diffusion processes",
  *Journal of Econometrics* 149(1), 65-81, 2009. DOI [10.1016/j.jeconom.2008.11.001](https://doi.org/10.1016/j.jeconom.2008.11.001)
- N. J. Higham, "Computing the nearest correlation matrix--a problem from finance", *IMA Journal of
  Numerical Analysis* 22(3), 329-343, 2002. DOI [10.1093/imanum/22.3.329](https://doi.org/10.1093/imanum/22.3.329)
- P. Burns, R. F. Engle, J. J. Mezrich, "Correlations and Volatilities of Asynchronous Data",
  *The Journal of Derivatives* 5(4), 7-18, 1998. DOI [10.3905/jod.1998.408000](https://doi.org/10.3905/jod.1998.408000)
