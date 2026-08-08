# Market Prices

All risk neutral models need to derived from observable market prices. This section specifies both the
model used and the necessary data required to correctly simulate a risk-neutral model.

Currently only FX, Equities, Commodities and IR rates may be risk neutral:

- FX may be simulated via *GBMAssetPriceTSModelPrices*  and requires only a corresponding FX vol surface to
  establish average ATM vols used by the corresponding *GBMAssetPriceTSModelImplied* model. It is
  specified as follows:

```json
{
"GBMAssetPriceTSModelPrices.AUD":
  {
    "instrument": {
      "Asset_Price_Volatility": "AUD.USD"
    },
    "Children": []
  }
}
```

- The only risk neutral IR currently available may be simulated via 
  *HullWhite2FactorModelPrices* and requires both a corresponding swaption volatility
   surface and a set of **instrument definitions** that define the forward starting swaps that
   reference the swaption vol surface. Note that again, only ATM vols are used. They are specified as:

```json
{
  "HullWhite2FactorModelPrices.ZAR-JIBAR-3M": {
    "instrument": {
      "Swaption_Volatility": "ZAR_SMILE_ICE",
      "Property_Aliases": null,
      "Generation_Parameters": {
        "Last_Tenor": {
          ".DateOffset": "9Y"
        },
        "Floating_Frequency": {
          ".DateOffset": "6M"
        },
        "First_Tenor": {
          ".DateOffset": "1Y"
        },
        "Day_Count": "ACT_365",
        "Last_Maturity": {
          ".DateOffset": "10Y"
        },
        "First_Start": {
          ".DateOffset": "1Y"
        },
        "Fixed_Frequency": {
          ".DateOffset": "6M"
        },
        "Index_Offset": 0,
        "Last_Start": {
          ".DateOffset": "9Y"
        },
        "First_Maturity": {
          ".DateOffset": "10Y"
        }
      },
      "Generate_Instruments": "No",
      "Instrument_Definitions": [
        {
          "Floating_Frequency": {
            ".DateOffset": "3M"
          },
          "Weight": 1,
          "Floating_Day_Count": "ACT_365",
          "Fixed_Day_Count": "ACT_365",
          "Start": {
            ".DateOffset": "3M"
          },
          "Fixed_Frequency": {
            ".DateOffset": "3M"
          },
          "Tenor": {
            ".DateOffset": "1Y"
          },
          "Market_Volatility": {
            ".Percent": 0
          }
        },
        {
          "Floating_Frequency": {
            ".DateOffset": "3M"
          },
          "Weight": 1,
          "Floating_Day_Count": "ACT_365",
          "Fixed_Day_Count": "ACT_365",
          "Start": {
            ".DateOffset": "3M"
          },
          "Fixed_Frequency": {
            ".DateOffset": "3M"
          },
          "Tenor": {
            ".DateOffset": "2Y"
          },
          "Market_Volatility": {
            ".Percent": 0
          }
        },
        {
          "Floating_Frequency": {
            ".DateOffset": "3M"
          },
          "Weight": 1,
          "Floating_Day_Count": "ACT_365",
          "Fixed_Day_Count": "ACT_365",
          "Start": {
            ".DateOffset": "3M"
          },
          "Fixed_Frequency": {
            ".DateOffset": "3M"
          },
          "Tenor": {
            ".DateOffset": "5Y"
          },
          "Market_Volatility": {
            ".Percent": 0
          }
        },
        {
          "Floating_Frequency": {
            ".DateOffset": "3M"
          },
          "Weight": 1,
          "Floating_Day_Count": "ACT_365",
          "Fixed_Day_Count": "ACT_365",
          "Start": {
            ".DateOffset": "3M"
          },
          "Fixed_Frequency": {
            ".DateOffset": "3M"
          },
          "Tenor": {
            ".DateOffset": "10Y"
          },
          "Market_Volatility": {
            ".Percent": 0
          }
        },
        {
          "Floating_Frequency": {
            ".DateOffset": "3M"
          },
          "Weight": 1,
          "Floating_Day_Count": "ACT_365",
          "Fixed_Day_Count": "ACT_365",
          "Start": {
            ".DateOffset": "6M"
          },
          "Fixed_Frequency": {
            ".DateOffset": "3M"
          },
          "Tenor": {
            ".DateOffset": "1Y"
          },
          "Market_Volatility": {
            ".Percent": 0
          }
        },
        {
          "Floating_Frequency": {
            ".DateOffset": "3M"
          },
          "Weight": 1,
          "Floating_Day_Count": "ACT_365",
          "Fixed_Day_Count": "ACT_365",
          "Start": {
            ".DateOffset": "10Y"
          },
          "Fixed_Frequency": {
            ".DateOffset": "3M"
          },
          "Tenor": {
            ".DateOffset": "2Y"
          },
          "Market_Volatility": {
            ".Percent": 0
          }
        },
        {
          "Floating_Frequency": {
            ".DateOffset": "3M"
          },
          "Weight": 1,
          "Floating_Day_Count": "ACT_365",
          "Fixed_Day_Count": "ACT_365",
          "Start": {
            ".DateOffset": "10Y"
          },
          "Fixed_Frequency": {
            ".DateOffset": "3M"
          },
          "Tenor": {
            ".DateOffset": "5Y"
          },
          "Market_Volatility": {
            ".Percent": 0
          }
        },
        {
          "Floating_Frequency": {
            ".DateOffset": "3M"
          },
          "Weight": 1,
          "Floating_Day_Count": "ACT_365",
          "Fixed_Day_Count": "ACT_365",
          "Start": {
            ".DateOffset": "10Y"
          },
          "Fixed_Frequency": {
            ".DateOffset": "3M"
          },
          "Tenor": {
            ".DateOffset": "10Y"
          },
          "Market_Volatility": {
            ".Percent": 0
          }
        }
      ]
    },
    "Children": []
  }
}
```

Note that although *Generation paramaters* can be specified, instrument definitions are preferred.

- *InterestRatePrices* solves an *InterestRate* zero curve from deposit, FRA and swap quotes. Each
  `Points` entry names an instrument type in `DealType` and carries a block of that type under
  `Deal`, authored exactly as `Trade Data` would author it, with `Quoted_Market_Value` the rate in
  percent the instrument is held at par at. A blank `Discount_Rate` builds a self-discounting
  curve; naming another block's curve makes this one a projection curve solved after it. The
  solved curve carries one knot per used quote, at that quote's last cashflow date, so holding a
  quote out with `Use` drops its knot too. An OIS swap is a `StructuredDeal` over a
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