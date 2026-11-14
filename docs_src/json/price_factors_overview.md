#  Price Factors

Price factors representing a snapshot of a market (and hence taken at the same time) are to be
provided here. e.g. for the ZAR FxRate with USD as the **Base Currency**

```json
{
  "Price Factors": {
    "FxRate.USD": {
      "Domestic_Currency": null,
      "Interest_Rate": "USD-MASTER",
      "Spot": 1
    },
    "FxRate.ZAR": {
      "Domestic_Currency": "USD",
      "Interest_Rate": "ZAR-SWAP.ZAR-USD-BASIS",
      "Spot": 0.075475
    }
  }
}
```

Note that the corresponding USD-MASTER and both the ZAR-SWAP and ZAR-USD-BASIS interest rate curves
need to be defined.

Interest-rate curves and volatility surfaces stand alone; an `FxRate` or an `EquityPrice` names in
`Interest_Rate` the curve its carry is read off (blank, the curve named as its currency), and states
in `Settlement_Days` the business days
from a day to its spot date, between which a forward on it carries: 3 on an `EquityPrice` and 2 on
an `FxRate` where omitted, a currency pair reading its underlying currency's. `Holiday_Calendar`
names the calendar those days count on, a pair's joint one on an `FxRate`; blank, a deal's
`Accrual_Calendars`, else Monday to Friday. An `FxRate`'s `Spot` is the price for delivery on its
spot date: a value converts between currencies at that price carried back to the day on the
domestic currency's curve less its own.

Interest Rate price factors uses the .Curve format. e.g.

```json
{
  "Price Factors": {
    "InterestRate.ZAR-JIBAR-3M": {
      "Day_Count": "ACT_365",
      "Currency": "ZAR",
      "Curve": {
        ".Curve": {
          "meta": [],
          "data": [
            [
              0.25,
              0.07
            ],
            [
              15.0,
              0.092
            ]
          ]
        }
      }
    }
  }
}
```

This specifies that the curve starts roughly 3m (0.25 years) from now at 7% and goes to 9.2% 15 years
from now. Curve Interpolation is discussed [here](../theory/asset_pricing.md#interest-and-inflation-rate-interpolation).

Tenors are in years, and rates are continuously compounded unless a field says otherwise.

Basis spreads are defined as InterestRates but have a subtype specified i.e.

```json
{
  "InterestRate.ZAR-SWAP.ZAR-USD-BASIS": {
    "Day_Count": "ACT_365",
    "Currency": "ZAR",
    "Curve": {
      ".Curve": {
        "meta": [],
        "data": [
          [
            0.0027397260273972603,
            0.0001897614337292436
          ],
          [
            30.024657534246575,
            -0.008421900202085705
          ]
        ]
      }
    },
    "Sub_Type": "BasisSpread"
  }
}
```

The **Sub_Type** does not enter any key: the curve above is `InterestRate.ZAR-SWAP.ZAR-USD-BASIS`
under `Price Factors` and `<Model>.ZAR-SWAP.ZAR-USD-BASIS` under `Price Models`. It selects the
`InterestRateBasisSpread` rules of [Model Configuration](model_configuration.md), so a basis curve
can be modelled apart from the curve it is a spread on.
