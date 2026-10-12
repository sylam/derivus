# Asset Pricing

Historical estimators for the equity and FX spot processes of the Theory page
[Asset Pricing](../theory/asset_pricing.md). The archive holds one column per spot, named after the
factor (`EquityPrice.<Name>`, `FxRate.<Name>`); `LogVar2FJCalibration` also reads the bar columns
`,OPEN`, `,HIGH`, `,LOW` beside the close. The shared OU regression and the conventions every
estimator follows are on [Historical Calibration](general.md).
