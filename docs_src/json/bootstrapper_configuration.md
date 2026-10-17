# Bootstrapper Configuration

This section names the price families the bootstrap runs. Each entry routes one family to its
[Market Prices](market_prices.md) blocks and may set family-wide values for them. A family runs
only if an entry names it.

```json
{
  "Bootstrapper Configuration": {
    "InterestRate": {"Prices": "InterestRate"},
    "SurvivalProb": {"Prices": "SurvivalProb"},
    "FXVol": {"Prices": "FXVol"},
    "GBMAssetPriceTSModelParameters": {"Prices": "GBMAssetPriceTSModel"},
    "HullWhite2FactorModelParameters": {"Prices": "HullWhite2FactorModel"},
    "CSForwardPriceModelParameters": {"Prices": "CSForwardPriceModel"},
    "LogVar2FJModelParameters": {"Prices": "LogVar2FJModel"}
  }
}
```

## The seven families

| Entry key | `Prices` | Reads blocks | Writes price factor | Prices on (factor types) |
|---|---|---|---|---|
| `InterestRate` | `InterestRate` | `InterestRatePrices.<curve>` | `InterestRate.<curve>` | `InterestRate`, `FxRate` |
| `SurvivalProb` | `SurvivalProb` | `SurvivalProbPrices.<name>` | `SurvivalProb.<name>` | `InterestRate` |
| `FXVol` | `FXVol` | `FXVolPrices.<pair>` | `FXVol.<pair>` | none |
| `GBMAssetPriceTSModelParameters` | `GBMAssetPriceTSModel` | `GBMAssetPriceTSModelPrices.<name>` | `GBMAssetPriceTSModelParameters.<name>` | `FXVol`, `EquityPriceVol`, `CommodityPriceVol` |
| `HullWhite2FactorModelParameters` | `HullWhite2FactorModel` | `HullWhite2FactorModelPrices.<curve>` | `HullWhite2FactorModelParameters.<curve>` | `InterestRate`, `InterestYieldVol`, `GBMAssetPriceTSModelParameters` |
| `CSForwardPriceModelParameters` | `CSForwardPriceModel` | `CSForwardPriceModelPrices.<name>` | `CSForwardPriceModelParameters.<name>` | `ForwardPrice`, `ForwardPriceVol`, `InterestRate` |
| `LogVar2FJModelParameters` | `LogVar2FJModel` | `LogVar2FJModelPrices.<name>` | `LogVar2FJModelParameters.<name>` | `CommodityPrice`, `CommodityPriceVol`, `DividendRate`, `EquityPrice`, `EquityPriceVol`, `FXVol`, `FuturesPrice`, `FxRate`, `InterestRate` |

An entry is keyed by the family, which is named for the factor it writes; an entry keyed
`InterestRateCurveParameters` or `FXVolSurfaceParameters` loads as `InterestRate` or `FXVol`. A block
writes its family's factor under the block's own name. The four `*Parameters` factors are what the
implied models read (see [Price Models](price_models.md#implied-models)). Every field an entry may
carry, per family, is on the family's page under this section.

## Entry form

| Form | What happens |
|---|---|
| An object with `Prices` | `Prices` is the stem of the family's `Market Prices` type (`<stem>Prices`). A stem belonging to another family refuses by name |
| An object without `Prices` | runs, with a warning: in process the family is looked up, but the multiprocessing bootstrap routes without the engine and needs it |
| A string (the older form) | accepted; the string itself is not read, and the family runs on its declared defaults |
| A key naming no family | refused, listing the seven |

## Run order

Entries run in a topological order of what each family writes against what it prices on, so a curve
is solved before a fit that prices on it, whatever order the file gives. Independent entries keep
the file's order. Two entries each reading what the other writes cannot be ordered and refuse by
name.

## What is refused and what is skipped

| Condition | Outcome |
|---|---|
| A `Market Prices` block whose type no family reads | the bootstrap refuses, naming it |
| A block of a family no entry configures | a warning; the block is not read |
| An entry whose family has no block | a warning; nothing to fit |
| An entry the family cannot construct from (a bad dial) | the bootstrap refuses, naming the entry |
| A family that wrote no factor despite having blocks | an error line naming the family; the run continues |

## Precedence {#precedence}

For each block, a field is read from the first of:

1. the block's own `instrument` object;
2. the family's entry here;
3. the field's declared default.

One exception: `HullWhite2FactorModelParameters` reads its boxes and seed (`Sigma_Bounds`,
`Alpha_Bounds`, `Alpha_Seed`, `Correlation_Bounds`) once, from the entry or the default. A block's
own value for those four is ignored.
