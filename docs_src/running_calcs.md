## In Python

A JSON file defining a calculation is run directly:

```python
import derivus as rf

cx = rf.Context()
cx.load_json('BaseValuation.Test1.json')
calc, out = cx.run_job(overrides={})
```

This returns the calculation object (`calc`, useful for inspecting state post-run) and the
output itself (`out`). `run_job` dispatches on the document's `Calculation.Object` —
`BaseValuation`, `CreditMonteCarlo` or `HedgeMonteCarlo` — and each also has an explicit method
(`cx.Base_Valuation(...)`, `cx.Credit_Monte_Carlo(...)`, `cx.Hedge_Monte_Carlo(...)`) for
running a different calculation than the JSON specifies. Check a document before running it with
`cx.validate()`; the [API Overview](api_overview.md) covers the full verb set, overrides,
market-value patching and the replay identity.

## In Jupyter

We've already seen an example of a *Base_Valuation* calculation, which simply calculates the
theoretical price of a portfolio of derivatives. To calculate expectations of future simulations
of the portfolio, we also need stochastic processes. This is done in the *Settings* tab under
*Model Configuration*. In the clip below, we assign both FxRates and EquityPrices to be simulated
using Geometric Brownian Motion (GBMAssetPriceModel).

Notice that the pairing of Risk Factor with a compatible stochastic process is fixed.

<video width="1000" height="500" controls>
  <source src="credit_monte_carlo.mp4" type="video/mp4">
</video>

The parameters are explained [here](api_usage/calculations.md). There is also the problem of
capturing correlations between risk factors: defining these Correlations and calculating the
stochastic process parameters can be eased by loading a file of historical data.
