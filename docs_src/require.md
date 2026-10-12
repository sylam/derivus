# Requirements

Derivus runs on CPU; an NVIDIA GPU (with current drivers and a CUDA-enabled PyTorch build) is
strongly recommended for Monte Carlo work. Installing the package brings the core dependencies with
it:

```
pip install derivus
```

The core dependencies, and what each is for:

- [PyTorch](https://pytorch.org/) >= 2.0 - the computational library that evaluates tensors on
  CPU or GPU, and the automatic-differentiation engine every sensitivity comes from.
- [NumPy](https://numpy.org/) >= 1.16.1 and [SciPy](https://scipy.org/) - array plumbing,
  interpolation, root finding and optimisation.
- [Pandas](https://pandas.pydata.org/) >= 1.0 - every tabular input and result.
- [pyparsing](https://github.com/pyparsing/pyparsing) >= 2.4.7 - the time-grid grammar.

## Python and SciPy versions

The package declares Python 3.8 and SciPy 1.2.2 as its floors, but the two-factor Hull-White
bootstrapper (`HullWhite2FactorModelParameters`) calls `scipy.optimize.basinhopping` with the `rng`
keyword, which SciPy takes from 1.15.0, and SciPy 1.15.0 requires Python 3.10. A library that
bootstraps that family therefore needs **Python >= 3.10 and SciPy >= 1.15**; on an older SciPy that
bootstrap fails with an unexpected-keyword error.

## Extras

Optional capability is grouped into pip extras, so the core library installs lean:

| extra | installs | when you want it |
|---|---|---|
| `derivus[interactive]` | jupyter, matplotlib | the notebook Workbench and plotting; pairs with [riskflow_widgets](https://github.com/sylam/riskflow_widgets) |
| `derivus[garch]` | arch >= 6.0 | calibrating `GARCHSpotModel` with `arch` (below) |
| `derivus[docs]` | mkdocs >= 1.5, mkdocs-material >= 9.0, pymdown-extensions >= 10.0 | building this documentation (`DV_Docs` emits the tree, mkdocs renders it) |
| `derivus[service]` | fastapi, uvicorn | the HTTP service (`DV_Service`) - `derivus/service.py` is the sole importer |
| `derivus[mcp]` | mcp >= 2 and < 3, mcp-types, requests | the MCP binding (`DV_MCP`) - needs Python 3.10+ |
| `derivus[quote]` | xlsxwriter >= 3.1 | the `.xlsx` quote sheet a structure is priced into |
| `derivus[desk]` | service + mcp + quote | the whole working stack of [Getting Started](getting_started.md) in one install |
| `derivus[enterprise]` | cryptography >= 42 | the spine's record: AES-GCM sealing and Ed25519 checkpoints |

`GARCHSpotModel`'s calibration fits a Student-t GARCH(1,1) by maximum likelihood with `arch` when it
is importable. Without it, SciPy's L-BFGS-B minimises the model's own Student-t negative
log-likelihood, its variance path seeded at the sample variance, from the fixed start
$(\omega, \alpha, \beta, \nu) = (0.01, 0.05, 0.90, 8.0)$, so one archive can calibrate to
different parameters in the two environments; the size of that difference is not measured.
Simulation never needs `arch`.

## GPU notes

[CUDA](https://developer.nvidia.com/cuda-zone) execution needs an NVIDIA card with drivers
matching your PyTorch build - `torch.cuda.is_available()` is the check that settles it. On a
CPU-only machine, install the CPU wheel explicitly
(`pip install torch --index-url https://download.pytorch.org/whl/cpu`) to avoid downloading the
much larger CUDA build.
