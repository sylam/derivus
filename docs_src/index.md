# Welcome to Derivus

Derivus is a source-available quantitative finance framework built on
[PyTorch](https://pytorch.org). It prices portfolios of derivatives and simulates them through
time — on CPUs or NVIDIA GPUs via [CUDA](https://developer.nvidia.com/cuda-zone) — for
valuation, XVA and hedging, with sensitivities by automatic differentiation carried all the way
back to market quotes.

A calculation is a JSON document in and a set of tables out: the job file declares the market
data, the models and the book, and the engine compiles and runs it. Every run is replayable from
`(plan, values, engine version, seed)`.

## Licence

Derivus is free for **noncommercial** use under the
[PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0):
research, personal projects, education and evaluation are all in scope, and the source is open to
read, run and validate. Commercial use requires a separate licence — contact the author.

## Features

* Full-portfolio Monte Carlo through time: exposure profiles and XVA (CVA / FVA / collateral /
  initial margin) at GPU speed
* Sensitivities via automatic differentiation, including the boundary terms a path-dependent
  decision contributes — a barrier crossing or an autocall trigger moves the value
  discontinuously, and the flux of scenarios across the trigger is carried onto the tape rather
  than dropped
* Sensitivities to market **quotes**, not only to calibrated factors: zero curves solved from
  deposit, FRA and swap benchmarks, swaption-vol model calibration, and FX smiles built from
  ATM / risk-reversal / butterfly quotes all differentiate through their calibration, so one
  backward pass reports `dV/dq` beside `dV/dθ`
* Bootstrapping inside the library: benchmark instruments are priced by the engine's own pricers,
  and between calibrations a moved quote can *ride* the calibration Jacobian to a valuation with
  no re-solve — with the drift of the ridden curve measured against a declared tolerance and the
  ride refused past it
* A hedging stack: a dynamic-programming solver trained against the simulator, with a
  deployment-faithful day-by-day backtest of the frozen policy on observed paths
* An HTTP service over the same engine — the JSON schema is published, jobs post and replay, and
  spreadsheets or notebooks are ordinary clients
* Theoretical documentation beside the code, encouraging independent validation

## Motivation

Similar to other quantitative finance libraries (like [quantlib](http://quantlib.org/)), the
motivations for Derivus are:

- Stop re-inventing the wheel. Robust implementations of standard pricing functions (like Black
  Scholes) have been written multiple times and, as a result of regulation, have had to be
  independently validated as many times.
- Keep the source open to inspection, so a model and its documentation can be validated together.

Libraries like quantlib already do an excellent job of the above. Derivus attempts to also:

- Make use of modern GPUs to perform full portfolio Monte Carlo simulation.
- Provide theoretical documentation as part of the library, thereby encouraging model validation.
- Standardize the way market and trade data is loaded and stored, as JSON — the job document is
  the whole contract, and a run is reproducible from it.
- Offer a simpler alternative by using Python as the main development language.

## Roadmap

What the library should be able to price and report next — engineering work the codebase owes
itself is tracked separately in the [Developer Roadmap](developer/roadmap.md):

- Wrong-way risk during the Monte Carlo simulation
- Initial-margin analytics (SIMM) from the bucketed quote deltas the sensitivity work now
  produces
- A structuring calculation: solve for a strike, a margin or a vol instead of a price
- A web front end over the HTTP service, rendered from the published schema
