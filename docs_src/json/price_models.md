# Price Models

The parameters of the stochastic processes that simulate price factors. An entry is keyed
`<ModelName>.<FactorName>`:

```json
{
  "Price Models": {
    "GBMAssetPriceModel.ZAR": {
      "Vol": 0.171569979672,
      "Drift": -0.0436796256753
    }
  }
}
```

`<FactorName>` is the factor's name without its type (here the `FxRate.ZAR` of `Price Factors`) and
without its `Sub_Type`. Which model a factor runs under is chosen by
[Model Configuration](model_configuration.md); the models themselves are described under
[Theory](../theory/asset_pricing.md).

## When a factor is simulated

A factor the book depends on is simulated when all of these hold, and is otherwise held as a static
factor at its value today:

| Condition | If it fails |
|---|---|
| `Model Configuration` assigns it a model | static, silently |
| it is not the `FxRate` of the `Base_Currency` | static, silently |
| the model has its parameters: a `Price Models` entry, or for an implied model its parameter factor (below) | static, with a log line naming the factor and the model |
| no other factor already took the same `<ModelName>.<FactorName>` key | static, silently |

The last row is a collision of keys that drop the factor type: an `EquityPrice.ZAR` and an
`FxRate.ZAR` both under `GBMAssetPriceModel` share the key `GBMAssetPriceModel.ZAR`, so one entry
parameterises both and only one of them is simulated. Their correlation rows collide too (see
[Correlations](correlations.md)). Give such factors distinct names.

## Implied models {#implied-models}

An implied model takes no `Price Models` entry. It reads the `Price Factors` entry its
[bootstrapper](bootstrapper_configuration.md) writes:

| Implied model | Reads the price factor |
|---|---|
| `CSImpliedForwardPriceModel` | `CSForwardPriceModelParameters.<name>` |
| `GBMAssetPriceTSModelImplied` | `GBMAssetPriceTSModelParameters.<name>` |
| `HullWhite2FactorImpliedInterestRateModel` | `HullWhite2FactorModelParameters.<name>` |
| `LogVar2FJImpliedSpotModel` | `LogVar2FJModelParameters.<name>` |

## Where parameters come from

| Source | How |
|---|---|
| Historical calibration | a `*Calibration` class fits the model to a time series and writes its `Price Models` entry (see [Calibration](../calibration/general.md)) |
| Implied calibration | a bootstrapper fits the model's parameter factor to current quotes ([Market Prices](market_prices.md)) |
| Stated by hand | an entry written directly, for a test or a stress |
