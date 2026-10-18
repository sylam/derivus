# Derivus JSON data format

A Derivus job is one JSON document (RFC 8259 [@rfc8259]; tagged values such as dates are described in
[General Types](general_types.md)):

```json
{
  "Calc": {
    "Calculation": "<calculation section>",
    "Deals": "<Deals section>",
    "MergeMarketData": "<Market data section>",
    "CalendDataFile": "<name of the xml file defining the business day calendars to use>"
  }
}
```

## Calculation {#calculation}

What to run. `Object` names the calculation type, and every other key is one of that type's
fields; each type's fields, defaults and menus are on its page under
[Calculations](calculations.md), which also says how a block is completed and overridden. The
methods behind them are described in [API & Usage](../api_usage/calculations.md).

```json
{
  "Calc": {
    "Calculation": {
      "Object": "CreditMonteCarlo",
      "Base_Date": {".Timestamp": "2022-12-01"},
      "Currency": "ZAR",
      "Time_Grid": {
        ".Grid": [
          {".DateOffset": "0d"},
          {".DateOffset": "2d"},
          {".DateOffset": "1w", ".Offset": "1w"},
          {".DateOffset": "3m", ".Offset": "1m"},
          {".DateOffset": "2y", ".Offset": "3m"}
        ]
      },
      "Random_Seed": 1,
      "Dynamic_Scenario_Dates": "Yes",
      "Deflation_Interest_Rate": "ZAR-SWAP",
      "Credit_Valuation_Adjustment": {
        "Calculate": "Yes",
        "Counterparty": "Sketchy Traders LTD",
        "Deflate_Stochastically": "Yes",
        "Stochastic_Hazard_Rates": "No",
        "CDS_Tenors": [0.5, 1, 3, 5, 10],
        "Gradient": "No"
      }
    }
  }
}
```

## Deals

Two attributes: `Reference`, the name of this collection of deals (used in the output), and
`Deals`, the deals themselves as a tree.

```json
{
  "Calc": {
    "Deals": {
      "Reference": "WaterfallTrading",
      "Deals": {
        "Children": [
          {
            "Instrument": {
              ".Deal": {
                "Object": "FXNonDeliverableForward",
                "Reference": "12345678",
                "Buy_Currency": "USD",
                "Sell_Currency": "ZAR",
                "Buy_Amount": 10000000,
                "Sell_Amount": 182805000,
                "Settlement_Date": {".Timestamp": "2022-12-28"},
                "Settlement_Currency": "ZAR",
                "Discount_Rate": "ZAR-SWAP"
              }
            }
          }
        ]
      }
    }
  }
}
```

`Deals` holds a single `Children` array of nodes, each an `Instrument` (a `.Deal`); a container
deal (a netting set, a structure) carries its own `Children`, so nested structures form a tree.
This forward is worth $(A\tilde F - BF)\,D$ in its settlement currency, $A$ the `Buy_Amount`, $B$
the `Sell_Amount` and $\tilde F$, $F$ the forward prices of the two currencies: positive amounts
buy USD and sell ZAR, and negating both reverses the trade. The fields of every deal type are under
[Trade Data](trade_data.md).

A deal carries its terms and nothing about where it is held. Where a desk keeps a record, the
portfolio a trade sits in, the agreement it is dealt under and the seat that booked it are carried
by the fill that books it - see [the Spine](../developer/spine.md#where-a-position-sits).

## MergeMarketData

Where the market data comes from. Three attributes:

| Attribute | What it does |
|---|---|
| `MarketDataFile` | a market-data JSON file (a `MarketData` object of sections, below), parsed once per content of its bytes - per `Context`, or per live book serving it |
| `ExplicitMarketData` | sections merged over that file's - or over an empty configuration where no file is named |
| `StressedMarketDataFile` | recorded, not loaded: a `StressedContext` reads it when asked to stress, replacing the `Price Models` entries and implied parameter factors of the factor types it is given, and restores them after |

The sections a market-data file may carry; a file naming any other section is refused by name:

| Section | Page |
|---|---|
| `System Parameters` | [System Parameters](system_parameters.md) |
| `Model Configuration` | [Model Configuration](model_configuration.md) |
| `Price Factors` | [Price Factors](price_factors_overview.md) |
| `Price Factor Interpolation` | [Model Configuration](model_configuration.md#price-factor-interpolation) |
| `Price Models` | [Price Models](price_models.md) |
| `Correlations` | [Correlations](correlations.md) |
| `Market Prices` | [Market Prices](market_prices.md) |
| `Valuation Configuration` | [Model Configuration](model_configuration.md#valuation-configuration) |
| `Bootstrapper Configuration` | [Bootstrapper Configuration](bootstrapper_configuration.md) |

```json
{
  "Calc": {
    "MergeMarketData": {
      "MarketDataFile": "mymarketdatafile.json",
      "ExplicitMarketData": {
        "Price Factors": {
          "InterestRate.ZAR-SWAP": {
            "Day_Count": "ACT_365",
            "Currency": "ZAR",
            "Curve": {".Curve": {"meta": [], "data": [[0.0027397260273972603, 0.069], [10.0, 0.079]]}},
            "Sub_Type": null
          },
          "FxRate.ZAR": {
            "Domestic_Currency": "USD",
            "Interest_Rate": "ZAR-SWAP",
            "Spot": 0.0566318758176227
          },
          "FXVol.USD.ZAR": {
            "Surface": {".Curve": {"meta": [2, "Flat"], "data": [[1.0, 1.0, 0.28]]}}
          }
        }
      }
    }
  }
}
```

The usual split keeps models, correlations and configuration in the file and the day's prices in
`ExplicitMarketData`. How the merge behaves:

| Rule | What the code does |
|---|---|
| Granularity | per entry: an `ExplicitMarketData` entry replaces the file's entry of the same key whole (a whole price factor, a whole model); `Correlations` merge per pair; `Model Configuration` and `Price Factor Interpolation` replace a type's default and its filter list |
| Persistence | the merge writes into the cached configuration, so a later job in the same `Context` naming a `MarketDataFile` of the same content starts from this one's overlay |
| An unknown section in `ExplicitMarketData` | fails on load (a `KeyError` naming it) |

`Context.save_json` writes the loaded job back as a self-contained document: every section of the
loaded market data goes into `ExplicitMarketData`, `MarketDataFile` is left blank, and the calendar
file is still named.

## CalendDataFile

An XML file of business-day calendars, parsed once per `Context` and kept by the sha256 of its
bytes; each location's weekend rule and holidays are part of the job's plan:

```xml
<Calendars>
  <Calendar Location="Johannesburg" Weekends="Saturday and Sunday" Holidays="2016-01-01|New Year's Day, 2016-03-21|Human Rights Day, 2047-12-26|Day of Goodwill" />
  <Calendar Location="Kabul" Weekends="Friday" Holidays="2016-02-15|Liberation Day, 2016-03-20|Noruz (New Year) 2, 2047-10-30|Ashoora*" />
</Calendars>
```

| Attribute | Form |
|---|---|
| `Location` | the calendar's name, as deals and quote blocks refer to it |
| `Weekends` | one of `Saturday and Sunday`, `Friday and Saturday`, `Saturday`, `Sunday`, `Friday` |
| `Holidays` | `YYYY-MM-DD|Name` entries separated by a comma and a space (`, `); a comma without the space does not separate |

```json
{
  "Calc": {
    "CalendDataFile": "calendars.cal"
  }
}
```
