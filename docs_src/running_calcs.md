# Jupyter Workbench

The Workbench is a notebook interface over a `Context`: it loads a job document, lays out its
settings, portfolio, price factors and calculations as editable panels, runs the calculation and
exports it back to JSON. It is a client only. What it runs is the document
[Getting Started](getting_started.md#three-lines-of-python) runs in three lines, and no calculation
needs it.

## Install and start

Install `derivus[interactive]` (Jupyter and matplotlib) and
[riskflow_widgets](https://github.com/sylam/riskflow_widgets). The Workbench itself is
[derivus_jupyter.py](https://github.com/sylam/derivus/raw/refs/heads/main/derivus_jupyter.py), a
module in the repository rather than in the package. From a cell:

```python
import pandas as pd
import derivus_jupyter

wb = derivus_jupyter.Workbench(default_rundate=pd.Timestamp('2025-04-02'))
```

`default_rundate` fills a calculation's `Base_Date` where the loaded job leaves it blank.
`wb.context` is the `Context` the panels read and write, so whatever the Workbench has loaded can be
inspected or run from the next cell. The same interface runs as a stand-alone dashboard under
[voila](https://github.com/voila-dashboards/voila).

| Tab | What it holds |
|---|---|
| Load JSON | a file drop for a job document, and old-to-new maps for the paths and file names it references, applied while it loads |
| Settings | `System Parameters`, `Model Configuration` and `Price Factor Interpolation` |
| Portfolio | the deal tree |
| Price Factors | every price factor, with the process `Model Configuration` routes it to and that process's parameters |
| Calculations | the calculation's fields; `Execute` runs it and `Export` saves the job as JSON |

## Booking a deal

A deal is added on the Portfolio tab. Every price factor the deal names must be defined on the
Price Factors tab before it prices; curve rates are continuously compounded unless a field says
otherwise ([Price Factors](json/price_factors_overview.md)).

## Simulating through time

A base valuation prices the book at the base date. A Monte Carlo through time also needs a
stochastic process for each simulated factor, assigned on the Settings tab under
`Model Configuration`, for example routing both the FX rates and the equity prices to
`GBMAssetPriceModel`. A factor type is offered only the processes that can simulate it.

The calculation's fields are in the [Calculations](json/calculations.md) reference and its methods
in [API & Usage](api_usage/calculations.md). A process's parameters and the `Correlations` between
factors can be estimated from an archive of historical data rather than typed in:
see [Historical Calibration](calibration/general.md).

## Results

Each result renders under its calculation. A result table is also written beside the notebook to
`./tmp/<calculation>.<table>.csv`, with a download link next to it (right-click and save the link).
The tables themselves are described in [Understanding Output](output.md).
