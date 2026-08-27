# Requirements

Derivus runs on any machine with Python **3.8+**; an NVIDIA GPU (with current drivers and a
CUDA-enabled PyTorch build) is strongly recommended for Monte Carlo work, but everything runs on
CPU. Installing the package brings the core dependencies with it:

```
pip install derivus
```

The core dependencies, and what each is for:

- [PyTorch](https://pytorch.org/) >= 2.0 — the computational library that evaluates tensors on
  CPU or GPU, and the automatic-differentiation engine every sensitivity comes from.
- [NumPy](https://numpy.org/) >= 1.16.1 and [SciPy](https://scipy.org/) >= 1.2.2 — array
  plumbing, interpolation and numerical integration.
- [Pandas](https://pandas.pydata.org/) >= 1.0 — every tabular input and result.
- [pyparsing](https://github.com/pyparsing/pyparsing) >= 2.4.7 — the time-grid grammar.
- [sortedcontainers](https://grantjenks.com/docs/sortedcontainers/) > 2.0 — the ordered
  tenor/expiry maps a volatility surface is built from.

## Extras

Optional capability is grouped into pip extras, so the core library installs lean:

| extra | installs | when you want it |
|---|---|---|
| `derivus[interactive]` | jupyter, matplotlib | the notebook Workbench and plotting; pairs with [riskflow_widgets](https://github.com/sylam/riskflow_widgets) |
| `derivus[garch]` | arch >= 6.0 | CALIBRATING `GARCHSpotModel` — the import is lazy, so simulation never needs it |
| `derivus[docs]` | mkdocs >= 1.5, mkdocs-material, pymdown-extensions | building this documentation (`DV_Docs` emits the tree, mkdocs renders it) |
| `derivus[service]` | fastapi, uvicorn | the HTTP service (`DV_Service`) — `derivus/service.py` is the sole importer |
| `derivus[mcp]` | mcp >= 2, requests | the MCP binding (`DV_MCP`) — needs python 3.10+ |
| `derivus[desk]` | service + mcp | the whole working stack of [Getting Started](getting_started.md) in one install |

## GPU notes

[CUDA](https://developer.nvidia.com/cuda-zone) execution needs an NVIDIA card with drivers
matching your PyTorch build — `torch.cuda.is_available()` is the check that settles it. On a
CPU-only machine, install the CPU wheel explicitly
(`pip install torch --index-url https://download.pytorch.org/whl/cpu`) to avoid downloading the
much larger CUDA build.
