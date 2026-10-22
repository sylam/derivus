# Validating Derivus

The evidence for what the Theory, Valuation and Calibration pages claim is the repository's test
suite, and it is meant to be run by the person checking the claim. A test is a document with an
outcome decided before the run, and its docstring names the change to the engine that would turn it
red, so a validator can run it as shipped, then break the engine on purpose and watch it fail. This page
says how to install and run the suite, what kind of evidence a test is, which modules hold which
area, and where what no test holds is written down.

## Install for testing

The tests live in the repository rather than the wheel, so the route is a clone:

```
git clone https://github.com/sylam/derivus && cd derivus
pip install -e ".[desk,enterprise,interactive]" pytest httpx
```

`desk` brings the service, the MCP binding and the quote sheet that the service and structures
tests drive; `enterprise` brings `cryptography`, which the book-of-record tests need; `interactive`
brings the Jupyter kernel the notebook gate executes; and httpx is what the service's test client
runs on. The GPU gates need the CUDA build of PyTorch ([GPU notes](require.md#gpu-notes)).

## The environment

| Variable | Set it to | Why |
|---|---|---|
| `DV_HOME` | a new temporary directory | the service keeps its book, security map, seed and quote files there (`~/.derivus` when unset); a gate that needs a home sets its own, and no run should touch a desk's |
| `DV_SPINE_HOME` | unset | it switches on the book of record; a gate that needs a record creates one in a temporary directory and sets the variable itself |
| `CUDA_VISIBLE_DEVICES` | the card the run may use | a gate that picks a device takes CUDA where one is visible; without one the suite runs on CPU and the CUDA gates skip |

In a POSIX shell:

```
export DV_HOME="$(mktemp -d)"
unset DV_SPINE_HOME
export CUDA_VISIBLE_DEVICES=0
```

In PowerShell:

```
$env:DV_HOME = (New-Item -ItemType Directory -Path (Join-Path $env:TEMP "derivus-tests-$(Get-Random)")).FullName
Remove-Item Env:DV_SPINE_HOME -ErrorAction SilentlyContinue
$env:CUDA_VISIBLE_DEVICES = "0"
```

## Running the suite, one area, one test

```
pytest tests -q -rs                                   # everything; -rs prints the reason each skip names
pytest tests/test_fx_tarf_json.py -q                  # one area
pytest "tests/test_tarf_cash_settle.py::test_a_sold_tarf_books_the_mirror_of_a_bought_one"
pytest tests -q -k barrier                            # every test whose name mentions barrier
pytest tests/test_fx_tarf_json.py -o log_cli=true --log-cli-level=DEBUG
```

The last form shows the engine's debug lines as the gate runs; several gates assert on the line a
pricer logs about what it decided. `tests/*.py` modules not named `test_*` are shared fixtures
(`tests/rates_world.py`, the `trial_*.py` deal sets) and helpers (`tests/crn_ladder.py`).

### What skips, and why

| Gates | Run only with |
|---|---|
| the CUDA arms of `tests/test_multi_gpu.py` | one visible CUDA device, or two for the arms that shard across devices; the same determinism is gated on CPU without a marker |
| `test_deterministic_kernels_is_read_every_run_and_set_both_ways`, `test_the_streamed_strip_holds_its_peak_and_the_unstreamed_gradient`, `test_a_daily_grid_marks_the_weekly_dates_within_its_cap` and `test_a_taped_strip_that_fits_is_one_checkpoint` in `tests/test_recompute_inner_mc.py` | a CUDA device: they read the GPU backward's atomics and the device allocator |
| the last gate of `tests/test_curve_strip_emitter.py`, `tests/test_swaption_vol_emitter.py` and `tests/test_equity_chain.py` | a Bloomberg terminal and `blpapi`; the curve and swaption gates also need a verified security map, and the curve and chain gates `DERIVUS_LIVE_BLOOMBERG` set. Every other gate in those modules runs on canned terminal answers |
| `tests/test_notebooks.py` | `nbformat` and `nbclient`, which the `interactive` extra brings |
| one gate of `tests/test_structures.py` | Python 3.12 or later, since it observes through `sys.monitoring` |
| three cases in `tests/test_service.py` | Windows for the file share-mode gate, `web/src` in the tree for the route gate, and `xlsxwriter` for the two quote-sheet gates |

Everything else runs on CPU or CUDA. Monte Carlo gates run at fixed seeds, and the CPU and CUDA
generators draw different paths from one seed, so a statistical gate reads a different sample on
each device; where a tolerance was set against a seed-to-seed spread, the module docstring records
it. A gate that pins a result by its hash pins it for one device type, since the same bits come
back only on the same kind of device and PyTorch build.

## What a test is

- **A JSON document through the public verbs.** A pricing test authors a job document, runs it with
  `Context.load_json` and `run_job` or the service's verbs, and reads the result tables. It builds
  no internal object by hand and patches nothing, so what it shows is what a user's document would
  show. Book-of-record tests drive real homes in temporary directories and inject faults by
  editing bytes on disk. Where a gate does stand something in (a Bloomberg terminal is the usual
  case), its module docstring says what and why.
- **An outcome from outside the engine.** The expected answer is a closed form (Black on a strip of
  Europeans for a TARF whose target cannot be reached, the cash-or-nothing digital for a one-coupon
  autocall), an oracle written in the test that shares nothing with the engine (a Cox-Ross-Rubinstein
  binomial, a Brownian-bridge Monte Carlo, the Reiner-Rubinstein formulas written out longhand), or a
  financial identity (in-out and put-call parity, a deal and its mirror summing to zero, half a unit
  marking exactly half, a curve recovering the quotes generated off it, an option value that is a
  martingale at zero carry). Where two documents price the same instrument the gate is equality to
  the bit, not a tolerance.
- **Derivatives against differences.** An AAD sensitivity is held to a central difference. Under
  Monte Carlo the difference is taken with common random numbers over a ladder of bump sizes
  (`tests/crn_ladder.py`), and agreement counts only where the ladder is flat, since a difference
  taken across a discontinuity scatters rather than converges.
- **A named killing mutation.** A test's docstring names the change to the engine that turns it
  red, and many module docstrings record how far the mutant moves the number. Making that change in
  a clone and running the test shows the gate can see the defect; a mutant that survives means the
  fixture cannot. For example, `test_a_sold_tarf_books_the_mirror_of_a_bought_one` in
  `tests/test_tarf_cash_settle.py` names "the settled fixing booked without `Buy_Sell`": dropping
  `buy_sell` from the `cash_settle` call in `pricing.pv_MC_Tarf` books the sold TARF's ledger equal
  to the bought one's, and the gate fails.
- **A fixture that does not zero what it tests.** A gate whose fixture sets the quantity under test
  to zero passes and measures nothing: `r = q` removes the forward's carry, a single valuation row
  never reaches a post-event branch, a flat surface hides every wrong-moneyness read. Module
  docstrings carry a degeneracy checklist saying, axis by axis, what the fixture varies and why
  anything it leaves degenerate is harmless. The rule and the checklist are in
  [Conventions](developer/conventions.md#fixture-degeneracy), and
  `python gates/fixture_degeneracy.py tests/<module>.py` tables what a run actually varied, by deal
  type.
- **A refusal by name.** Where a document cannot be priced, the gate asserts that the engine refuses
  it naming the field or factor at fault, rather than pricing a plausible wrong number or dropping
  the deal quietly.

## Where each area is held

Paths are under `tests/`. A module's docstring states its oracle and, where they apply, the
fixture's degeneracy checklist and what the module does not gate; a test's docstring states its claim
and its killing mutation. The [Model Inventory](model_inventory.md) reads the other way, listing for
each instrument, model and calibration the test modules that name it.

| Area | Modules | What they hold |
|---|---|---|
| Rates and curves | `test_interest_rate_prices.py`, `test_fra_timing.py`, `test_rate_leg_conventions.py`, `test_swaption_exercise_boundary.py`, `test_hw2f_analytic.py`, `test_hw2f_composition.py`, `test_base_currency_json.py` | The curve bootstrap recovers the curve its quotes were generated from, on a multi-curve USD world and a single-curve ZAR one, and a cross-currency curve reprices its FX outrights by covered interest parity; a curve quoted off itself bootstraps back to itself to the bit in no Newton iteration, and a ticked one solves cold, to the bit of a solve with no curve standing. FRA payment timings against a hand-derived reference under base valuation and credit Monte Carlo, and a swap leg's `Advance` reset, `Discounted` payment and index publication calendar and a deposit's principal under each timing against hand-derived strips, the payments under credit Monte Carlo too. A physical swaption exercises on its last pre-expiry row, and against Black and Bachelier by hand reads its vol at its duration-equivalent tenor and the level of its own rate where its swap amortises and its mid-curve vol where the swap starts after the expiry. A blank index day count is the floating leg's own on a swap, a cap and a swaption. The Hull-White short rate's mean at reversion speeds through zero against its closed form, a two-factor step that is not positive definite repaired, the two-factor calibration pinned through its job document, and the two-factor composition's measure change bounded in multiples of its own standard error. The base currency's curve simulates when a model is declared. |
| FX and equity options | `test_american_option_json.py`, `test_digital_spread_json.py`, `test_compo_analytic_pricers.py`, `test_fx_double_asian.py`, `test_extendable_forward_json.py`, `test_option_settlement_json.py`, `test_sticky_strike_strip.py`, `test_base_valuation_gamma.py`, `test_equity_swaps.py` | American options between Black and a converged binomial. Digitals as call and put spreads read at their own strikes, exact on collinear smile nodes. Composite-currency barriers, one-touches and Asians against parity with an independent European, the textbook one-touch and a recomputed moment match, which an equity and an FX Asian also meet on a sloped curve under a vol term structure. The FX double Asian against a Monte Carlo off the deal's own terms, flat and on that sloped curve. The extendable forward against Black in its degenerate limits, by dominance where there is no closed form, and its mirror summing to zero. An option paid after its expiry: under simulation every later row is the settlement payoff discounted and the ledger books it on the settlement date; past the base date the mark is the stated print's intrinsic. `Sticky_Strike` equal to the bit to `Sticky_Moneyness` where the reference is the forward. Second-order Greeks: Black-Scholes gamma, the vanna identity across surface nodes, a linear deal's zero gamma and swap convexity against a central difference. An equity swap leg is, to the bit, the swaplet list its schedule's conventions spell, under base valuation and simulation. |
| Barriers and touches | `test_barrier_arms_json.py`, `test_barrier_bridge.py`, `test_barrier_fold_json.py`, `test_partial_barrier_json.py`, `test_commodity_touch_json.py` | All eight Reiner-Rubinstein arms, each in both spellings and with and without a rebate, against the textbook written longhand. Survival between grid dates by the martingale property at zero carry. A crossed barrier folds at compile into the deal it became, equal to the bit to that deal's own document. Partial-time barriers and their rebates against an independent bridge Monte Carlo. A commodity touch prices as the equity touch on the same forward curve. |
| Autocalls | `test_autocall_json.py`, `test_branch_and_weight.py`, `test_recompute_equity_pricers.py` | A one-coupon autocall against the cash-or-nothing digital and its two degenerate limits. The branch-and-weight estimator on autocall, TARF, accumulator and discrete-barrier documents against quadrature references written in the test; it is the default, and the crisp estimator is one switch away. The autocall's and the discrete barrier's inner Monte Carlo replayed in the backward pass, bit-identical. |
| TARFs and accumulators | `test_fx_tarf_json.py`, `test_fx_accumulator_json.py`, `test_tarf_cash_settle.py`, `test_boundary_tarf_events.py`, `test_recompute_inner_mc.py` | Value and FX delta against Black on a strip of Europeans where the target or barrier is out of reach, on a steep curve, with the line the pricer logs. A sold TARF books the mirror of a bought one in the cashflow ledger. The knock-in and target decisions in the gradient against a common-random-numbers ladder. The inner Monte Carlo replayed in the backward pass with a gradient equal to the bit. A base valuation mark the same to the bit in any book, and a second `Quasi_Seed` moving it within its Monte Carlo error. |
| Commodity and energy | `test_energy_forward_components.py`, `test_energy_leg_fx.py`, `test_forward_rate_extrapolation.py`, `test_chained_basis.py` | A floating energy leg's mark in both forward-curve modes against hand values. An energy leg is the sum of its cashflows whatever its curve's currency; an energy option is Black on the moment-matched average; a commodity digital is Black's digital. A carry curve read in front of its first knot against the line's hand value. Observed-basis chains: the partner is pulled in, a same-day link is ordered first, and a chain that cannot close is refused by name. |
| Inflation | `test_inflation_list.py`, `test_inflation_prices.py` | An inflation-linked cashflow list against the linker computed by hand: printed months as printed, later months projected off the inflation curve, under base valuation and simulation. The inflation curve bootstrap recovers the growth curve its zero-coupon swap breakevens were priced off, the closed form where it is the root, and its quote derivative against re-bootstraps, a linker's quote deltas, a swap quote's included, against curves refitted, beside the emitter that authors its block and its index off a scripted terminal. |
| Credit and xVA | `test_credit_json.py`, `test_credit_prices.py`, `test_cva_gamma_kink.py`, `test_boundary_pricer_events.py`, `test_boundary_scoping_dominance.py`, `test_multi_gpu.py`, `test_simm.py`, `test_saccr.py`, `test_fork_pricing_failures.py`, `test_static_curve_risk_neutral.py` | Default swaps (upfront, premium sign and timing, par rate, digital recovery), the nth-to-default basket's limits and its Greeks against central differences, and the cash account. The survival curve bootstrap recovers the hazard curve its par spreads and upfronts were priced off, its par spreads read by the ISDA standard model within its stated conventions, and its quote derivative, a swap quote's included, against re-bootstraps, beside the emitter that authors its block off a scripted terminal. The CVA Hessian's exposure-kink term against a common-random-numbers ladder, on a linear deal whose pathwise gamma is zero. CVA and FVA gradients of deals that take decisions on simulated state against common-random-numbers ladders, including a portfolio where the boundary term dominates and its sign is known in advance. A sharded credit Monte Carlo bit-identical across worker counts. SIMM sensitivities by bump and revaluation: the quotes reprice the book, a swap's PV01 is its annuity, a quote's move ridden on its curve's calibration Jacobian writes the rate rows a re-solve per quote writes but for the second order the ride drops, the CRIF is ISDA's format, a book's CRIF is its trades' own one-trade CRIFs to the bit, in closed form and where a Monte Carlo pricer values them, its moves priced as the scenarios of one scenario valuation write the same CRIF whatever the pass and the worker count and the bump's that priced each move alone to 1e-12 of a trade's largest row, each scenario valued as a base valuation of its market, and a source CRIF's trades revalued off their own documents merge back into it; under `Method: Graph` each row, one backward pass per trade, is the limit of the bump's central difference across every risk type; a rate vega row is the vol-weighted vega off `Greeks: First` on a Lognormal and a Normal space. The SIMM margin by hand on a book of one row per risk class, against the SIMM notebook's aggregation on the synthetic parameters, and its rows' derivatives against central differences of the margin. The SA-CCR exposure at default reproduces the figures of CRE99's five worked examples, and sets built for the purpose reach its floors, a margined set's threshold and cap and the FX, equity and commodity classes by hand. A deal the run cannot value is skipped and counted, while a refusal saying the document is wrong stops the run rather than marking zero. A static curve under `NoModel: RiskNeutral` reads on the base date, and on a flat world on every row, what it reads under `Constant`, for the swap, FRA, equity leg, autocall swap, TARF, barriers, one-touches, extendable forward and compo deals that spelled the read themselves; the FX Asian and the average-price swap price on an unmodelled spot. |
| Collateral | `test_collateral.py` | A CSA's call arithmetic as exact equalities, and a collateralised credit Monte Carlo whose starting balance is the record's and whose transfer on every path is the record's call. |
| Calibrations and bootstraps | `test_innovation_clock.py`, `test_loud_refusals.py`, `test_warm_starts.py`, `test_hw2f_analytic.py` | Every historical estimator dates an innovation at the start of its return, so a drawn correlation is recovered. A calibration or bootstrap that refuses is reported in the class's own words, and a set where every factor is skipped is refused. A written factor warm-starts the Hull-White two-factor and energy fits to where the cold fit lands, counted in objective evaluations. No test checks that a historical estimator recovers known parameters ([Historical Calibration](calibration/general.md#what-every-estimator-shares)). |
| Quote sensitivities | `test_quote_propagation.py`, `test_market_prices_partition.py` | A quote tick carried by the calibration Jacobian: first-order agreement on a bump ladder, second-order drift against a refit, refusal of a tick too large to ride, and a multi-curve set carried whole or not at all. Which keys of a quote block are values and which are structure, by one rule for every family, so a tick moves the values hash and leaves the plan hash. |
| Stochastic volatility (LogVar2FJ) | `test_logvar2fj_json.py`, `test_spot_model_martingale.py`, `test_cross_spot_model.py`, `test_equity_chain.py`, `test_notebooks.py` | The GBM limit in pricing and through the CVA, the quanto loading, the fit repricing its at-the-money strip and fitting its wings, and its refusals by name, on a world authored against the degeneracy checklist. A violent spot model holds the forward, read from both notional sides of one zero-cost strip. A spot model on an FX cross takes the pair's axis, and two books of one economy agree. A listed option chain screened into the family's quote block and bootstrapped. The four model-validation notebooks execute end to end. |
| Hedging | `test_diffml_v2_oos.py`, `test_action_space_and_artifact.py`, `test_frozen_policy_is_frozen.py`, `test_model_interchangeability.py`, `test_spot_history_optional.py`, `test_symlog_smoke.py`, `test_stepper.py` | The solver's value stays bounded and its policy beats no hedge on held-out paths. One action space for the solver, the benchmarks and the stepper; first-step turnover from the opening book; a corridor-trained policy loads only into its corridor. A loaded policy fits nothing. One configuration across two spot-model worlds. The utility scale stated explicitly or derived without a price history, and a misspelt mode refused by name. The stepper's no-trade rollout lands on the framework's own baseline. |
| Structures | `test_structures.py`, `test_quote_sheet.py` | Seven FX structures quoted against a bootstrapped surface and held to financial identities that would be false if any piece of the runner were wrong. The quote workbook carries the market strike, frozen values and deterministic content, and refuses a cell it would truncate. |
| Positions, P&L and the record | `test_position_scaling.py`, `test_pnl.py`, `test_diary.py`, `test_spine.py`, `test_spine_canon.py`, and the other `test_spine_*.py` | For every deal type a position can be held in, half a unit marks exactly half and a short position exactly minus. P&L per position and portfolio against independent valuations and its own additive identity. Every payment the diary announces equals the cashflow a credit Monte Carlo realises. The record's wire formats recomputed independently of the implementation (the canonicaliser against RFC 8785's own vectors, the hashes and seals), tampering detected from bytes edited on disk, capabilities, key custody, replicas, and each of the oracle's invariants made to fail. |
| The service and MCP | `test_service.py`, `test_mcp.py`, `test_bloomberg_discover.py`, `test_bloomberg_session.py`, `test_curve_strip_emitter.py`, `test_swaption_vol_emitter.py` | A job over HTTP gives the numbers `run_job` gives in process; the dispatcher's ordering, deduplication by replay tuple, and survival of engine failures; the book verbs validate before they write. The MCP tools carry no logic of their own and a refusal writes nothing. Bloomberg discovery believes a security only on the terminal's own evidence, and the curve and swaption emitters reject dead prints by name and author blocks in the family's declared fields. |
| Schema and declarations | `test_schema_emission.py`, `test_declared_defaults.py` | The schema is generated from the per-class field declarations, and what a declaration can still get wrong is caught. A declared default is a convention, which completes a read, or a placeholder, which is refused by name at booking; neither enters the plan hash. |

## What is not gated

A test holds only what it was written to hold. Three places say what nothing holds:

- The developer pages mark each rule with the test that holds it (**Held by**) or say **Not gated**;
  [Conventions](developer/conventions.md) states the convention, and
  [Architecture](developer/architecture.md) lists the pages.
- Module docstrings name what the module does not gate and why, often because the defect can be
  reached only by rebinding an engine function rather than through a document
  (`tests/test_boundary_pricer_events.py` and `tests/test_fork_pricing_failures.py` are examples).
- [Model Limitations](limitations.md) lists what is known to be wrong or limited in the models
  themselves.
