# Model Configuration

Assigns each price factor the stochastic process that simulates it, as a `.ModelParams` rule set
with two parts:

- **modeldefaults** maps a factor type to a model, e.g. `FxRate: GBMAssetPriceModel` simulates every
  FX rate with `GBMAssetPriceModel`.
- **modelfilters** lists exceptions per factor type, each `[[attribute, value], model]`.

```json
{
  "Model Configuration": {
    ".ModelParams": {
      "modeldefaults": {
        "EquityPrice": "GBMAssetPriceModel",
        "FxRate": "GBMAssetPriceModel",
        "InterestRate": "PCAInterestRateModel"
      },
      "modelfilters": {
        "InterestRate": [
          [["Currency", "CHF"], "HullWhite1FactorInterestRateModel"],
          [["Currency", "DKK"], "HullWhite1FactorInterestRateModel"]
        ],
        "InterestRateBasisSpread": [
          [["ID", "ZAR-SWAP.ZAR-USD-BASIS"], "HullWhite1FactorInterestRateModel"]
        ]
      }
    }
  }
}
```

Every equity and FX rate here runs under `GBMAssetPriceModel` and every interest rate under
`PCAInterestRateModel`, except CHF and DKK curves and the `ZAR-SWAP.ZAR-USD-BASIS` basis spread,
which run under `HullWhite1FactorInterestRateModel`.

| Rule | What the code does |
|---|---|
| Lookup key | the factor type, with `BasisSpread` appended for a factor whose `Sub_Type` is `BasisSpread` (so a basis curve is modelled apart from `InterestRate`) |
| Filters first | the type's filter rules are tried in order and the first match wins; with none matching, the type's default applies; with neither, the factor is not simulated |
| A match | the factor's `Price Factors` field named `attribute` (case-insensitive name) equals `value` exactly; `ID` matches the dotted factor name |
| Merging | a later section (a second market-data file, or `ExplicitMarketData`) replaces a type's default and that type's whole filter list |

A model also needs its parameters before the factor is simulated - see
[Price Models](price_models.md#when-a-factor-is-simulated).

## Price Factor Interpolation

Interest-rate, inflation-rate and forward-rate curves take their interpolation from the
`Price Factor Interpolation` section, a `.ModelParams` of the same form: a default per factor type,
and rules matched as above, `ID` naming a single curve (the `Sub_Type` plays no part here).

| Factor type | Methods | When unnamed |
|---|---|---|
| `InterestRate`, `InflationRate` | `HermiteRT`, `Hermite`, `LinearRT`, `Linear` | `Linear` |
| `ForwardRate` | `Linear`, `LinearExtrapolate` | `Linear` |

A name the engine does not recognise is built as `Linear` without a message. `RT` methods
interpolate rate × tenor and divide by the query tenor, so on a curve carrying a knot at tenor 0 a
query at exactly 0 has no rate (0/0); the other methods are unaffected. On an interest-rate or
inflation curve a query below the first knot or past the last reads the end knot's rate (flat
extrapolation). The methods are described under
[Theory](../theory/asset_pricing.md#interest-and-inflation-rate-interpolation).

## Valuation Configuration

Per deal type, options its pricer reads, keyed by the type's `Object`:

```json
{"Valuation Configuration": {"QEDI_CustomAutoCallSwap": {"SpotModel": "LogVar2FJ"}}}
```

`SpotModel` is `None` by default, the type's own pricer. `LogVar2FJ` prices the deal on the
LogVar2FJ spot model; `EquityBarrierBinaryOption`, `EquityBarrierOption`,
`FXAccumulatorOptionDeal`, `FXTARFOptionDeal`, `QEDI_CustomAutoCallSwap` and
`QEDI_CustomAutoCallSwap_V2` accept it. A value a type does not accept refuses when the deal is
built.
