# Derivus

An xVA quantitative library written in pure python using pytorch, with adjoint algorithmic
differentiation throughout.

A job is a JSON program: market data, deals and a calculation block go in, results come out. The
engine compiles the job — factor discovery, dependency ordering, process construction — then
executes it against Monte-Carlo scenarios. Sensitivities come from AAD rather than bumping, so a
full greek vector costs one backward pass however many factors it covers.

## Installation

```
pip install derivus
```

Requires python >= 3.8 and pytorch >= 2.0. A GPU build of pytorch is strongly recommended — the
scenario engine is written for it.

Optional extras:

```
pip install "derivus[garch]"        # GARCHSpotModel calibration (lazy import; the rest runs without it)
pip install "derivus[interactive]"  # jupyter and matplotlib
pip install "derivus[docs]"         # the mkdocs toolchain DV_Docs generates config for
```

To work on Derivus itself, install from a clone instead:

```
pip install -e .
```

## Usage

```python
import derivus as dv

cx = dv.Context()
cx.load_json('job.json')
calc, results = cx.run_job()
```

The JSON is the whole contract — every feature is reachable from it, and a user script should never
need to import derivus internals. Six console scripts are installed:

| | |
| --- | --- |
| `DV_Batch` | CVA, CollVA and FVA over a folder of netting sets |
| `DV_Bootstrap` | calibration (currently Hull-White 2-factor from swaption vols) |
| `DV_Docs` | builds `./docs` from `./docs_src` |
| `DV_Service` | the HTTP service - the live book, the web UI, the verbs every client rides |
| `DV_MCP` | the MCP binding over stdio, for an LLM host (`derivus[mcp]`) |
| `DV_Bloomberg` | builds and re-verifies a Bloomberg security map on a terminal workstation |

## Layout

| | |
| --- | --- |
| `derivus/` | the library |
| `tests/` | the suite |
| `tests/fixtures/` | every input the suite needs — configs and small calibrated market data |
| `gates/` | acceptance harnesses: end-to-end reproduction and bit-identity |
| `docs_src/` | documentation sources, including the developer section |
| `experiments/` | research and validation drivers; end-user scripts that only use `load_json` / `run_job` |
| `notebooks/` | exploratory notebooks |
| `excel_integration/` | xlwings add-in, and the HTTP client it talks to `DV_Service` through |
| `data/` | *untracked* — where you drop real market data |
| `artifacts/` | *untracked* — run outputs, fits, decks |

Scripts under `experiments/` are run from the repo root, e.g. `python experiments/production_solver.py`.

## Market data

No market data ships with the source. Exchange settlements, open interest and curve history are
licensed by their providers and are not ours to redistribute, so `data/` is gitignored and you
supply your own.

The suite needs nothing from you. Its inputs live in `tests/fixtures/` and are calibrated
*parameters* — HMM transition matrices, VAR coefficients, GARCH fits — because a fitted statistic
is not the series it was fitted to. A fresh clone runs the whole suite green, with two calibration
tests skipping and naming the file they want.

To run those two, put a CSV at `data/pl_exp.csv` with a date index and a `CommodityPrice.PLATINUM`
column. `tests/fixtures/calibration_config.json` shows the wider shape the calibration scripts
expect.

## Documentation

Build it locally with `DV_Docs`, or read the sources under `docs_src/`.
[`docs_src/getting_started.md`](docs_src/getting_started.md) is the working stack in fifteen
minutes — the service with a live book, the web UI, Claude over MCP, and a Bloomberg quote source
ticking the market. The developer section
(`docs_src/developer/`) is the internal view: architecture, the calculation lifecycle, the
dependency system, the resolver layer and the house conventions.

## Licence

[PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0) —
free for any noncommercial purpose, including research, teaching and personal projects. Commercial
use requires a separate licence.

Derivus continues work previously published as RiskFlow under GPL-3.0. Anyone who received that
release keeps their rights under those terms; this repository is licensed separately by the same
copyright holder.
