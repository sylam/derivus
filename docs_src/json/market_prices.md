# Market Prices

The quotes the bootstrap fits. A block is keyed `<Family>Prices.<name>` and carries the family's
fields under `instrument`; the [Bootstrapper Configuration](bootstrapper_configuration.md) decides
which families run, in what order, and how a block's fields fall back. A block of a type no family
reads refuses the bootstrap by name.

| Block type | What it fits | Quote table | Writes |
|---|---|---|---|
| `InterestRatePrices` | a zero curve, holding its benchmarks (`DepositDeal`, `FRADeal`, `SwapInterestDeal`, `StructuredDeal`, `FXForwardDeal`) at par | `Points` | `InterestRate.<name>` |
| `FXVolPrices` | an FX smile surface from ATM, risk-reversal and butterfly quotes | `Points` | `FXVol.<name>` |
| `GBMAssetPriceTSModelPrices` | the vol curve of `GBMAssetPriceTSModelImplied`, off an ATM column | none: the named surface's ATM column | `GBMAssetPriceTSModelParameters.<name>` |
| `HullWhite2FactorModelPrices` | the two-factor Hull-White model to ATM swaption vols | `Instrument_Definitions` | `HullWhite2FactorModelParameters.<name>` |
| `CSForwardPriceModelPrices` | the Clewlow-Strickland $(\sigma, \alpha)$ to energy futures options | `Energy_Futures_Options` | `CSForwardPriceModelParameters.<name>` |
| `LogVar2FJModelPrices` | the LogVar2FJ spot model to vanilla option quotes | `European_Options` | `LogVar2FJModelParameters.<name>` |

Each family's fields, defaults and table columns are on its page under this section, and its method
on the [Bootstrapping](../bootstrapping/general.md) pages. A quote table's rows are objects keyed by
column name. In `Points` and `European_Options` the values a market patch may move are
`Quoted_Market_Value`, `Quoted_Bid`, `Quoted_Ask` and `Timestamp`; everything else in a block is
structural.

## GBMAssetPriceTSModelPrices

Names the vol surface (`FXVol`, `EquityPriceVol`, `CommodityPriceVol` or `VolatilityGrid`) whose
ATM column becomes the instantaneous vol curve:

```json
{
  "GBMAssetPriceTSModelPrices.AUD": {
    "instrument": {
      "Asset_Price_Volatility": "AUD.USD"
    },
    "Children": []
  }
}
```

## HullWhite2FactorModelPrices

Names a swaption vol surface (`InterestYieldVol`) and the forward-starting swaps the benchmark
swaptions are struck on. Each `Instrument_Definitions` row is one ATM swaption and must carry its
own `Market_Volatility`: a zero and an absent column both refuse by name. The vol is read in the
convention the named surface's `Distribution_Type` declares - a lognormal Black vol by default, an
absolute normal vol where it says `Normal`. A block may state its own `Distribution_Type`; blank,
the default, is unchecked, and one differing from the surface's refuses before anything is priced.
`Weight` weights the row's residual in the objective. `Generate_Instruments` and
`Generation_Parameters` are declared but not built: state the rows.

```json
{
  "HullWhite2FactorModelPrices.ZAR-JIBAR-3M": {
    "instrument": {
      "Swaption_Volatility": "ZAR_SMILE_ICE",
      "Instrument_Definitions": [
        {
          "Start": {".DateOffset": "3M"},
          "Tenor": {".DateOffset": "1Y"},
          "Floating_Frequency": {".DateOffset": "3M"},
          "Fixed_Frequency": {".DateOffset": "3M"},
          "Floating_Day_Count": "ACT_365",
          "Fixed_Day_Count": "ACT_365",
          "Market_Volatility": {".Percent": 21.5},
          "Weight": 1
        },
        {
          "Start": {".DateOffset": "10Y"},
          "Tenor": {".DateOffset": "10Y"},
          "Floating_Frequency": {".DateOffset": "3M"},
          "Fixed_Frequency": {".DateOffset": "3M"},
          "Floating_Day_Count": "ACT_365",
          "Fixed_Day_Count": "ACT_365",
          "Market_Volatility": {".Percent": 18.4},
          "Weight": 1
        }
      ]
    },
    "Children": []
  }
}
```

## InterestRatePrices

Each `Points` entry names an instrument type in `DealType` and carries a block of that type under
`Deal`, authored exactly as [Trade Data](trade_data.md) would author it, with `Quoted_Market_Value`
the number the solve holds the instrument at par at: a rate in percent for a rate benchmark, the
outright in `Buy_Currency` per unit of `Sell_Currency` for an `FXForwardDeal`. A blank `Discount_Rate` builds a
self-discounting curve; naming another block's curve makes this one a projection curve solved after
it. The solved curve carries one knot per used quote, at that quote's last cashflow date, so
holding a quote out with `Use` drops its knot too. An OIS swap is a `StructuredDeal` over a
`Compounding_Method: OIS` floating leg and a fixed leg:

```json
{
  "InterestRatePrices.USD-OIS": {
    "instrument": {
      "Currency": "USD",
      "Day_Count": "ACT_365",
      "Discount_Rate": "",
      "Points": [
        {
          "Use": "Yes",
          "Descriptor": "USD 2Y OIS",
          "DealType": "StructuredDeal",
          "Quote_Type": "Par_Rate",
          "Quoted_Market_Value": 4.1524,
          "Deal": {
            "Reference": "OIS_24M",
            "Currency": "USD",
            "Net_Cashflows": "Yes",
            "Children": ["... the floating and fixed legs ..."]
          }
        }
      ]
    },
    "Children": []
  }
}
```

## FXVolPrices

Builds an `FXVol` log-moneyness surface from the delta quotes an FX smile is dealt in. Each
`Points` row is an `Expiry` in years, a delta `Pillar`, a `Quote_Type` of `ATM`, `RR` or `BF`, the
number, and the `Timestamp` it was seen at; the surface carries the latest of those as its
`Quote_Timestamp`.

| Convention | What the solve assumes |
|---|---|
| Pillar delta | premium-adjusted forward delta [@reiswich2010] (`Delta_Type: Forward`, `Premium_Adjusted: Yes`, the only values offered) |
| ATM | the delta-neutral straddle, $K = F e^{-\sigma^2 T/2}$ (`ATM_Convention: Delta_Neutral_Straddle`) |
| Wings | call $\sigma_{ATM} + BF + RR/2$, put $\sigma_{ATM} + BF - RR/2$; a missing `RR` or `BF` reads as 0 |
| `BF` | the **smile** strangle, $(\sigma_{call} + \sigma_{put})/2 - \sigma_{ATM}$. A broker's market (one-vol) strangle is a different number and must be converted before it is entered |
| `Pillar` 0.5 | refused: a 50-delta pair is quoted as the ATM row |

`Grid_Tolerance` sizes the log-moneyness grid the surface is written on, and that grid is pinned:
re-bootstrapping the same expiries at the same tolerance over a surface this family already wrote
moves the vols and leaves the grid alone, so a vol tick is a value patch rather than a new plan; a
changed tolerance refines a new grid. A `Timestamp` keeps the resolution it was authored at: a plain
date stays a date, an intraday stamp is written in ISO 8601 form [@iso8601] with its time.

```json
{
  "FXVolPrices.USD.ZAR": {
    "instrument": {
      "Currency": "ZAR",
      "Delta_Type": "Forward",
      "Premium_Adjusted": "Yes",
      "ATM_Convention": "Delta_Neutral_Straddle",
      "Grid_Tolerance": 0.0001,
      "Points": [
        {"Use": "Yes", "Expiry": 1.0, "Pillar": 0.0, "Quote_Type": "ATM",
         "Quoted_Market_Value": 0.161, "Timestamp": {".Timestamp": "2026-06-30"}},
        {"Use": "Yes", "Expiry": 1.0, "Pillar": 0.25, "Quote_Type": "RR",
         "Quoted_Market_Value": 0.019, "Timestamp": {".Timestamp": "2026-06-30"}},
        {"Use": "Yes", "Expiry": 1.0, "Pillar": 0.25, "Quote_Type": "BF",
         "Quoted_Market_Value": 0.0041, "Timestamp": {".Timestamp": "2026-06-30"}}
      ]
    },
    "Children": []
  }
}
```

## CSForwardPriceModelPrices

Names the `ForwardPrice` factor, the `ForwardPriceVol` surface the quoted vols are read off and the
discount curve; each `Energy_Futures_Options` row is one option on the future settling at
`Settlement_Date`. `Quote_Type` takes vols only.

```json
{
  "CSForwardPriceModelPrices.BRENT": {
    "instrument": {
      "Energy": "BRENT",
      "Forward_Volatility": "BRENT",
      "Discount_Rate": "USD-OIS",
      "Quote_Type": "Implied_Volatility",
      "Energy_Futures_Options": [
        {"Expiry_Date": {".Timestamp": "2026-11-03"}, "Settlement_Date": {".Timestamp": "2026-12-03"},
         "Option_Type": "Call", "Strike": 74.7965, "Units": 1000.0, "Weight": 1.0,
         "Quoted_Market_Value": 0.3323568397}
      ]
    }
  }
}
```

## LogVar2FJModelPrices

Names the underlying, its vol surface, discount and carry curves (a `*_Type` field says which
factor type a name means where it exists under more than one), and the vanilla quotes in
`European_Options`. Under `Quote_Type: Implied_Volatility`, the default, a quote is a vol the
target premium is priced at off the named surface; under `Premium` the quote is the premium and no
surface is read. The fit's many dials are on the
family's page and its method and limits on the
[Bootstrapping](../bootstrapping/fx_and_equity.md) and
[developer](../developer/market_prices.md#logvar2fj) pages.

```json
{
  "LogVar2FJModelPrices.INDEX_A": {
    "instrument": {
      "Underlying": "INDEX_A", "Underlying_Type": "EquityPrice",
      "Volatility": "INDEX_A.EUR", "Volatility_Type": "EquityPriceVol",
      "Discount_Rate": "EUR", "Discount_Rate_Type": "InterestRate",
      "Yield": "INDEX_A", "Yield_Type": "DividendRate",
      "Quote_Type": "Implied_Volatility",
      "European_Options": [
        {"Expiry_Date": {".Timestamp": "2024-07-26"}, "Strike": 90.103621, "Option_Type": "Put",
         "Units": 1.0, "Weight": 1.0, "Quoted_Market_Value": 0.23237528,
         "Timestamp": {".Timestamp": "2024-06-28"}}
      ]
    }
  }
}
```
