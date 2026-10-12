# Observed Basis

Historical estimators for the three `ObservedBasis` processes documented on the Theory page
[Asset Pricing](../theory/asset_pricing.md). A basis column is named
`ObservedBasis.<parent>.<basis>`; the framework pulls the parent's or the chain source's column into
the same frame by the rules of [Cross-Factor Calibration](cross_factor.md). Basis values are LEVEL
differences in the parent's price units, except where a section says log. The conventions every
estimator follows are on [Historical Calibration](general.md).

| | Link | What the basis is |
|---|---|---|
| `BasisLinkedSpotCalibration` | open: rides its parent's path | an AR(1) level loaded on the parent's same-day move |
| `ChainedBasisCalibration` | closed: the declared `Chained_Basis` | a bridge between consecutive source observations |
| `FixingBridgeCalibration` | open: rides its parent's path | an intraday fixing bridged in the parent's log space |
