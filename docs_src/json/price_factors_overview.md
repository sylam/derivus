#  Price Factors

Price factors representing a snapshot of a market (and hence taken at the same time) are to be
provided here. e.g. for the ZAR FxRate with USD as the **Base Currency**

```json
{
  "Price Factors": {
    "FxRate.USD": {
      "Interest_Rate": "USD-MASTER",
      "Spot": 1
    },
    "FxRate.ZAR": {
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
in `Settlement_Days` the business days from a day to its spot date, between which a forward on it
carries: 3 on an `EquityPrice` and 2 on an `FxRate` where omitted, an `FxRate`'s against USD.
`Holiday_Calendar` names the calendar those days count on, on an `FxRate` the one its currency
settles on; blank, Monday to Friday, and a name the calendar file does not hold fails the job.
A currency pair counts the
most days of its currencies other than USD on the calendars of each and settles on a business day
of both and of USD, as the market does, so a pair against USD may pass a New York holiday before
its spot date. An `FxRate`'s `Spot` is the price for delivery on its spot date: a value converts
between currencies at that price carried back to the day on the base currency's curve less its
own. A rate is priced in the base currency: a `Domestic_Currency` it states is not read.

e.g. USDCAD, which settles one business day on, with USD as the **Base Currency**

```json
{
  "Price Factors": {
    "FxRate.USD": {
      "Interest_Rate": "USD-SOFR",
      "Spot": 1,
      "Holiday_Calendar": "New York"
    },
    "FxRate.CAD": {
      "Interest_Rate": "CAD-CORRA",
      "Spot": 0.73,
      "Settlement_Days": 1,
      "Holiday_Calendar": "Toronto"
    }
  }
}
```

Each currency states its own calendar, one `Location` of the
[CalendDataFile](index.md#calenddatafile), and the pair joins them: the one day counts on Toronto's
calendar and the spot date rolls onto a business day of Toronto and of New York. From Wednesday
17 June 2026 the spot date is Thursday 18 June; from Thursday 18 June it is Monday 22 June, the
Friday being a New York holiday; from Tuesday 30 June it is Thursday 2 July, the Wednesday being a
Toronto holiday. USD's own `Settlement_Days` is not read on a pair against it.

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

## A spot model's parameters: `LogVar2FJModelParameters` {#logvar2fjmodelparameters}

The [LogVar2FJ model](../theory/stochastic_volatility.md) reads one price factor,
`LogVar2FJModelParameters.<name>`: `<name>` is the equity, an FX pair's non-base token, or
`<later>.<earlier>` for a cross ([Equity and FX](../theory/stochastic_volatility.md#equity-and-fx)).
A fit writes it from a `LogVar2FJModelPrices` block. Every key, with the values of the synthetic
factor the tests fit (`WORLD_FACTOR` in `tests/test_logvar2fj_json.py`) and the declared defaults
where it states none:

```json
{
  "Price Factors": {
    "LogVar2FJModelParameters.INDEX_A": {
      "Kappa_L": 0.5,
      "Sigma_L": 1.0,
      "Rho_L": -0.4,
      "Kappa_S": 6.0,
      "Cap_A": 3.8329158963830388,
      "Steps_Per_Year": 252.0,
      "C_Min": 0.12,
      "Residual_Law": "NIG",
      "Skew_Gradient": "",
      "On_Guard": "",
      "Stickiness_Band": 0.0,
      "Xi_Curve": {".Curve": {"meta": [], "data": [
        [0.0, 0.04318756980713686], [0.07671232876712329, 0.04702934519853678],
        [0.2493150684931507, 0.050895424812433816], [0.4986301369863014, 0.05447323287603885],
        [0.7479452054794521, 0.05807287748434197]]}},
      "Rho_S": {".Curve": {"meta": [], "data": [[0.0, -0.7939878539386854]]}},
      "Beta": {".Curve": {"meta": [], "data": [[0.0, -2.0308954107502926]]}},
      "Sigma_S": {".Curve": {"meta": [], "data": [[0.0, 2.0536372201266198]]}},
      "Alpha": {".Curve": {"meta": [], "data": [[0.0, 20.916694355454386]]}}
    }
  }
}
```

The first four are scalars. The five curves are rows of `[knot, value]`, the knot in years from the
base date starting the segment or bucket the value holds on, the last value holding beyond its
knot: `Xi_Curve` is the expected forward variance, one segment per ATM expiry of the fit, and
`Rho_S`, `Beta`, `Sigma_S` and `Alpha` share one set of bucket knots, a single knot at 0 being the
constant-parameter model. A curve is authored as a curve, never a bare number. `Cap_A` null is no
corner; this one is the corner the fit wrote. `Skew_Gradient` is two comma-separated numbers where
the fit read a reserve line, and `On_Guard` a sentence where the fit sits on a guard. A fit also
writes `Property_Aliases` null, which the model does not read. What each key means, its units, the
check the factor makes at load and whether it is fitted, declared, derived or written are on
[the model's page](../theory/stochastic_volatility.md#parameters).
