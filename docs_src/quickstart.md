## Three lines of Python

A calculation is a JSON document; running one is loading it and asking for the result:

```python
import derivus as rf

cx = rf.Context()
cx.load_json('BaseValuation.Test1.json')
calc, out = cx.run_job(overrides={})
```

`out['Results']` holds the tables — see [Understanding Output](output.md). Before running
anything, `cx.validate()` reports what would stop the job (missing market-data blocks, deals
missing required fields) without pricing a thing, and `cx.describe()` reports what the engine
made of the document. The [API Overview](api_overview.md) covers both, plus patching market
values and replaying a run.

The same document runs over HTTP — `DV_Service` publishes the schema and executes posted jobs —
see [the service verbs](api_overview.md#the-same-verbs-over-http).

## Jupyter

The interactive route is the Workbench: install `derivus[interactive]` and
[riskflow_widgets](https://github.com/sylam/riskflow_widgets). The user interface is
[derivus_jupyter](https://github.com/sylam/derivus/raw/refs/heads/main/derivus_jupyter.py) and is
not included in the core package (it is solely a GUI and not necessary to run any calculation).

Once installed, run the workbench from a cell:

```python
import pandas as pd
import derivus_jupyter

wb = derivus_jupyter.Workbench(default_rundate=pd.Timestamp('2025-04-02'))
```

Here's a quick video showing how an equity option can be booked. Notice that each risk factor
needs to be correctly defined. All interest rates in Derivus are assumed to be NACC (Nominal
Annual Compounded Continuously).

<video width="1000" height="500" controls>
  <source src="quickstart.mp4" type="video/mp4">
</video>

Once the calculation is executed, it can be exported to JSON. The `wb` variable holds the state
of the library and can be used to interrogate the data in the workbench — in particular, its
`context` member is the same `Context` the three-line quickstart above builds by hand.

The GUI is also designed to work with [voila](https://github.com/voila-dashboards/voila) as a
stand-alone dashboard if desired.

## Design Philosophy

The basic idea in derivus is to define all inputs that would normally be associated with pricing
a financial instrument separately as a *price factor*. This includes things like FX rates, equity
prices, volatilities, interest rates etc.

The portfolio of instruments is then declared to reference these price factors. This separates
the definition of the financial instrument from the market variables that would typically be used
to calculate its theoretical (risk neutral) price.

Later, a stochastic process model can be attached to a price factor that specifies how that price
factor changes through time. This is the basis for xVA calculations (as long as you can specify a
risk neutral calibration). Of course, you are free to specify any process model you like to test
the performance of your portfolio of derivatives.
