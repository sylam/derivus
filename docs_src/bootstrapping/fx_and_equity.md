# Equity and FX

Four classes: `FXVol` constructs an FX volatility surface from delta quotes;
`GBMAssetPriceTSModelParameters` integrates a surface's ATM column into a lognormal term
structure; `OptionQuoteFamily` is the quote preparation the option-fitted families share; and
`LogVar2FJModelParameters` fits the [LogVar2FJ model](../theory/stochastic_volatility.md) to a
ladder of European options. All six families are summarised on [General](general.md).
