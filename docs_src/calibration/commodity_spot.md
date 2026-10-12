# Commodity Spot

Historical estimators for the three commodity spot processes documented on the Theory page
[Asset Pricing](../theory/asset_pricing.md): `LogOUSpotModel`, `MarkovHMMSpotModel` and
`GARCHSpotModel`. Each reads the factor's single `CommodityPrice.<Name>` column of daily closes.
The conventions every estimator follows are on [Historical Calibration](general.md).

| | `LogOUSpotCalibration` | `MarkovHMMSpotCalibration` | `GARCHSpotCalibration` |
|---|---|---|---|
| Fitted on | log price levels | log returns (or price differences) | log returns |
| Estimator | least-squares OU regression | Baum-Welch EM | Student-t maximum likelihood |
| Volatility clustering | none | regime persistence | GARCH(1,1) recursion |
| Heavy tails | none | regime mixture, optional shared $\nu$ | Student-t $\nu$ |
| Drift written | none (the log level reverts) | per regime | 0 |
