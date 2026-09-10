# Market Prices

`Market Prices` is the risk-neutral half of the market data: the **quotes** a risk-neutral model is
fitted to, where `Price Factors` holds the curves and surfaces a historical calibration produces. A
*bootstrapper* turns each block into the factor or model parameters the simulation reads. All eight
families are built.

## A quote is an instrument, a quote type and a number {#a-quote}

| | |
| --- | --- |
| **the instrument** | a reference to an EXISTING instrument type — the thing the number is a price *for* |
| **`Quote_Type`** | what kind of number is quoted, and therefore what "reprices to the quote" means |
| **`Quoted_Market_Value`** | the number |

A quote does not restate an instrument's fields. It names an instrument type and carries a block of
that type, so the `Instrument` store's declarations *are* the quote's schema. `quote_instruments` on
the bootstrapper class names the types a family's quotes may be, and that list is what a `DealType`
dropdown offers. The gate holding quotes to the declared deal types went with the mock-built suite;
the structure registry's `test_the_registry_publishes_exactly_the_declared_structures` is the
surviving copy of the rule.

`Quote_Type` is per family, not global: Clewlow–Strickland takes implied vols, the option family a
vol or a premium, an interest-rate quote a par rate.

## A family maps a quote set to what it calibrates {#a-family}

A bootstrapper class is one price family. It declares:

- `market_factor_type` — the `Market Prices` type string a block is filed under, and the string the
  class selects its own work by. Declared rather than recovered from the class name, because the
  block is `LogVar2FJModelPrices` while the class is `LogVar2FJModelParameters`.
- `price_factor_type` — the `Price Factors` type it writes, which is its key in the
  `Bootstrapper Configuration` section, and `reads` — the factor types its fit reads, which is
  what orders the section (below).
- `fields` — the block's schema, including the quote table or container as the class reads it.
- `quote_instruments`, where the quotes are instruments rather than a fixed option table.

`mapping['MarketPrices']['types']` is `schema.emit_market_prices(bootstrappers)` over those
declarations, and `construct_bootstrapper` resolves the class from its section key — the factor
type it writes, or the class name as an alias.

| Family | Quotes | Writes |
| --- | --- | --- |
| `GBMAssetPriceTSModelPrices` | a vol surface, ATM column only — or, where `FXVolPrices` built that surface, [its ATM rows](#fxvolprices) | `GBMAssetPriceTSModelParameters` — an integrated vol curve |
| `CSForwardPriceModelPrices` | European energy futures options | `CSForwardPriceModelParameters` — sigma, alpha |
| `LogVar2FJModelPrices` | European options on any spot, plus forward-start smiles | `LogVar2FJModelParameters` — four scalars, **a forward-variance curve `ξ` and four bucketed levers** |
| `HullWhite2FactorModelPrices` | forward-starting swaps against a swaption surface | `HullWhite2FactorModelParameters` — two sigma curves, two alphas, a correlation |
| `InterestRatePrices` | deposits, FRAs, swaps and FX forward outrights | an `InterestRate` zero curve |
| `FXVolPrices` | ATM vols, risk reversals and butterflies | an `FXVol` log-moneyness surface |

## `Bootstrapper Configuration` is the book's own default for every dial {#bootstrapper-configuration}

A `Market Prices` block carries the quotes and the factors they price against. Everything else a
family reads — the boxes it fits in, the seeds it starts from, the budgets it stops on, the priors
it falls back to — is a hyperparameter, and a book states those ONCE, in that family's
`Bootstrapper Configuration` entry.

**Two levels, one rule.** A family is constructed with its section entry and completes it from its
own declarations (`schema.declared_defaults`), so every declared key stands whether the book wrote
it or not; each quote block is then read as `dict(section, **block)`, the block winning on
conflict. A dial nobody declares is its `F(...)` default; a dial the section declares is the book's
answer for every block of that family; a dial the block declares is that name's own. A malformed
value refuses by name at construction, before a quote is read — `Sigma_L_Bounds` reversed,
`Psi_Strikes` unordered, a `Seed` of the wrong length, a prior table naming an asset class the
family does not fit — and `Config.bootstrap` logs the refusal and skips the family, so a book with
a bad entry writes no factor rather than a wrong one.

**The section is keyed by what each family WRITES, and `Prices` names what it reads:**

```json
"Bootstrapper Configuration": {
    "InterestRate":             {"Prices": "InterestRate"},
    "FXVol":                    {"Prices": "FXVol"},
    "LogVar2FJModelParameters": {"Prices": "LogVar2FJModel", "Paths": 8192}
}
```

The key is the family's `price_factor_type`, the `Price Factors` block it produces, so a section
reads as the factors a run will write. Four families already spelled that in their class name; the
two that did not are `InterestRate` (was `InterestRateCurveParameters`) and `FXVol` (was
`FXVolSurfaceParameters`), and every old class name stays as an alias, so an older book keeps
working; an unknown key refuses by name and lists both spellings. `Prices` is the STEM of the
`Market Prices` type the entry routes on — the type is that value plus the literal `Prices` — and
is mandatory on the multiprocessing path (`derivus_bootstrap`, which routes blocks to workers off
the section alone, and refuses before a worker is spawned); in process a missing `Prices` is a
warning naming the fix, the family knowing its own type, and one naming another family's type
refuses on both paths. An entry may still be the legacy CSV string an older file carries, which
routes on its first field and declares no hyperparameter. The rename also repairs
`derivus_bootstrap`'s daily carry-over, which matches previously written factors against the
section's keys and so never carried an `FXVol.*` or `InterestRate.*` forward.

**The section drives the loop, and the order is a topological sort.** `Config.bootstrap` and
`derivus_bootstrap.Parent.start` both take `bootstrappers.bootstrap_order`, a
`utils.topological_sort` over what each family writes against what it `reads`, so a curve is
solved before the swaption fit that prices on it and an FX surface before the GBM curve that
integrates its ATM column, whatever order the file gives — where `sorted()` put
`HullWhite2FactorModelParameters` before both spellings of the curve family. A read no configured
family writes carries no edge, that factor being already in `Price Factors`; a family reading what
it writes orders only its own blocks; a cycle refuses by name; independent entries keep the file's
order. Each family is handed only its own blocks, and both empty cases are logged: a configured
family the book carries no block for, and a block type no configured family claims.

Three entry points are reachable without `bootstrap` — the swaption residual closure (`calc_loss`,
`calc_loss_on_ir_curve`), which a gate builds off a hand-authored block, and the curve ride
(`propagate`, `plan_key`), which runs at EXECUTE off a document carrying no bootstrapper. Each
completes its own block from the declarations, so the declared default is the fallback there
rather than a literal repeated in the code: the nineteen inline `.get(key, literal)` reads the
families carried are gone, and eighteen had a literal equal to the declared default. The
nineteenth is the one that mattered. A Table's declared blank is the string `'null'`, and a
completed block handed that string to `sigma_knots`, which took anything truthy as a knot list — one
knot at t = 0 in place of the ten-knot grid, on every Hull-White block that declared none, from
9bfb095 (13:09, 2026-09-08) until the same evening, 65 of the hex set's 4,180 floats. `sigma_knots`
reads a LIST or nothing, which is `schema.quote_rows`' own reading, and the same blank is why the
two prior tables promoted below are Text and not Tables: a Table cannot carry a default table.

Two things are deliberately not declarable. `FXVolSurfaceParameters.grid_tolerance_bounds` is
`Grid_Tolerance`'s own domain rather than a dial — below `1e-8` the refinement does not terminate —
so it is the field's declared `bounds` and the engine's own refusal. `OptionQuoteFamily`'s FX
ladder (`fx_atm_expiries`, `fx_wing_expiries`, `fx_wing_pillars`, `fx_days_per_year`,
`fx_expiry_tolerance`, `fx_minimum_contracts`) stays class attributes: `fx_surface_block` authors
quotes rather than fitting them, is a classmethod with no section in reach, and its ladder is a
family's declaration the partition gate holds the emitted header to. What each family declares is
on its generated page; the fields promoted out of the code on 2026-09-08 —
`LogVar2FJModelParameters`' `Leverage_Prior_Weight`, `Shape_Penalty`, `Residual_Horizon`,
`Spot_Rung_Tolerance`, `Slow_Factor_Prior_Defaults` and `Leverage_Prior_Defaults`;
`HullWhite2FactorModelParameters`' `Sigma_Bounds`, `Alpha_Bounds`, `Correlation_Bounds`,
`Basin_Step`, `Basin_Temperature` and `Basin_Hops`; `CSForwardPriceModelParameters`'
`Sigma_Bounds`, `Alpha_Bounds` and `Seed` — default to the numbers the code carried, so a book
declaring none of them fits what it fit before (the NKY block bit for bit, the four book ladders
within 1e-11).

## `InterestRatePrices` — a curve solved from its quotes {#interestrateprices}

**The block.** `InterestRateCurveParameters` declares the `Currency` of the curve to build, the
`Day_Count` its tenors are expressed in, an optional `Discount_Rate` naming the curve the quotes
discount on, the solver's three knobs (`N_Iter`, `Tol`, `Damping_Halvings`, each read off the
block completed by its declarations and by `Bootstrapper Configuration.InterestRate`; a coupled
set takes the strictest of its members'), the three lifecycle switches (`Quote_Sensitivity`,
`Quote_Propagation`, `Drift_Tolerance` — see [Quote Propagation](quote_propagation.md)), and the
quote `Points`. A blank `Discount_Rate` builds a **self-discounting** curve.

**The quotes.** Each point carries a `Deal` — a deposit, an FRA, a swap, an `FXForwardDeal` or a
`StructuredDeal` over two legs, authored exactly as it would be in `Trade Data` — plus `DealType`,
`Quote_Type`, `Quoted_Market_Value`, a free-text `Descriptor` and a `Use` flag so a quote can be held
out without being deleted. `Quoted_Bid`, `Quoted_Ask` and `Timestamp` are declared and the solve
reads none of them: they are this family's share of `schema.MARKET_QUOTE_VALUES`, the value plane a
tick may move without touching the plan. `DealType` supplies the block's `Object`, and the family
stamps `Discount_Rate` — what an instrument *projects* off is authored in its own deal, what the
quote set *discounts* on belongs to the curve set and is stated once.

`Quoted_Market_Value` is read in its own `DealType`'s unit, and where it lands is a property of the
instrument TYPE — a `FRA_Rate`, a `Swap_Rate`, a pinned `Interest_Rate_Schedule`, a fixed leg's
`Rate` column. That correspondence is a registry (`QUOTE_WRITERS`) rather than a branch, so a new
quotable instrument is a row. Nothing is scaled centrally: a rate benchmark is quoted in **percent**
because that deal's own field semantics divide by 100, so a quote that is not a rate rides untouched.
`Quote_Type` declares the one convention that is built, `Par_Rate`: every benchmark held at PV zero.

**A forward outright as a benchmark.** `FXForwardDeal` is the one quotable type whose quote is not a
rate. Its `Quoted_Market_Value` is the forward **outright** — units of `Buy_Currency` per one unit of
`Sell_Currency` — landing as `Buy_Amount = quote × Sell_Amount` on a benchmark that fixes the sold
amount and both discount-rate names, which keeps the PV affine in the quote the way the par solve and
the drift metric both need. Covered interest parity is written nowhere: the residual is the forward's
own pricer held at PV zero. The residual **crosses currencies**, which the benchmark closure allows
by the rule it applies to every factor a solve is not solving for — the other leg's curve and both
`FxRate` spots are discovered from the deal's declarations and enter as detached constants. Because
the coupling is authored inside the deal rather than in `Discount_Rate`, the **solve order** reads it
there too: a block comes after every block building a curve its benchmark deals name.

`Quote_Sensitivity` on such a block **refuses by name**. The overlay carrying a quote derivative
rides cashflow-schedule value columns; an outright lands in `Buy_Amount`, read as a float off the
deal, so no schedule moves. The refusal is measured rather than a branch on the type, so it stays
silent wherever a quote does reach a column. `Quote_Propagation` meets the same refusal.
**Not built: quote derivatives for an amount-valued quote.**

**The knot rule.** ONE knot per used quote, at that benchmark's last cashflow date, in the block's
`Day_Count`. That is the only placement that makes the system square: a knot with no instrument
maturing at it is unidentified, and two instruments between the same pair of knots leave the curve
under-determined. Below the shortest knot the curve is flat, which `CurveTenor` gives by clipping.
**The output grid is that grid** — writing onto a wider grid would interpolate the result, and an
interpolated curve no longer reprices its quotes. There is therefore no `Zero_Rate_Grid` field, and
holding a quote out with `Use` drops its knot with it.

**The solve.** A damped Newton root find in float64 on the vector of the engine's own pricers held at
par. Two blocks make a multi-curve set and `Discount_Rate` orders them. The Jacobian comes from AAD
through the pricers — one backward pass per quote gives a whole row, no bump loop — which is the same
derivative the [quote-sensitivity](quote_sensitivities.md) thread carries the other way into `dV/dq`.
The residual is written once and differentiated twice.

**Fields that are not declared, on the terms every store here is held to** — a field the engine does
not read is not declared. `Zero_Rate_Grid` (there is no second output grid), `Spot_Offset` (a quote's
own `Deal` states its `Effective_Date`), and `Quote_Type`'s `Rate` and `Price` values, which are
conventions the family would have to author differently.

The curve this writes is an `InterestRate`, not a `<ClassName>` parameter block, so the class
declares `price_factor_type` and `Config.bootstrap`'s "wrote no `<name>.*` price factor" check reads
it. Interpolation of a solved curve comes from `Price Factor Interpolation`, not the block — see
[Conventions](conventions.md#registries-not-functions).

## Retired: the two Heston-Nandi families {#hestonnandi-retired}

`HestonNandiModelPrices` (five parameters off a Fourier-inverted daily GARCH recursion) and
`HestonNandiComponentModelPrices` (its CJOW extension, a long-run component fitted as an **L
curve**) were the desk's spot families until 2026-09-06. Both are gone: the calibrators, the price
factors, the OSS kits, the stride, the two implied spot processes and the whole `utils` half they
alone read. What replaced them is [`LogVar2FJModelPrices`](#logvar2fj), which fits the same
`European_Options` ladder on the same `fx_surface_block` — hoisted here as `OptionQuoteFamily`, the
family-neutral quote preparation — and which the FX gate measured **five times** better on the
wings than the plain family and **six times** better than the component one on the same 22
contracts (0.131 vol points against 0.663 and 0.760), with second-order greeks and quote-space risk
where both Heston-Nandi families refused the switch by name. Everything the two families measured
stands in the roadmap; nothing in the engine reads them, and a book that names one refuses by name.

## `LogVar2FJModelPrices` — a Monte Carlo fit with a forward-skew term {#logvar2fj}

**The block** is `OptionQuoteFamily`'s shared quote preparation plus `Fit_Mode`, the walk, the
structural constants, `Residual_Law`, the two declared guards, the weights, the calendar-time
`Param_Buckets`, `Event_Days`, the forward block with its `Stickiness_Prior` DIFFERENCE pair,
stage 4's `Slow_Factor_Prior` and `Quote_Sensitivity` — each documented where it is declared. `derivus_bloomberg.equity_chain` emits it off a listed chain, and the inherited
`fx_surface_block` authors the FX one, asked here for **two delta pillars** at four wing expiries,
because a `Bootstrap` bucket frees one parameter per wing quote. What follows is what it MEASURED.

**The stored curve is `ξ(t) = E₀[h_t]`, the expected forward variance, and the OU level is
DERIVED from it.** `ℓ + s` is Gaussian, so `E[e^{ℓ+s}] = e^{L* + Var(ℓ+s)/2}`: storing the LEVEL
makes the stored curve not the expected variance, and every change of vol-of-vol drags the ATM
level with it. Storing `ξ` and setting `L*(t) = log ξ(t) − ½Var(ℓ_t + s_t)` makes `E₀[h_t] = ξ(t)`
exactly whatever the vol-of-vol is, so the ATM level is invariant to `σ_s, σ_ℓ` by construction and
the inner solve is near-identity. The variance is measured from wherever the walk STARTS, so a row
the kit re-seeds at `(L*(t_row), 0)` reads `ξ` exactly where a base-date Jensen term would
under-shoot it. `ξ` is piecewise CONSTANT on the segments between ATM expiries — flat forward
variance, the shape var-swap strips are quoted in — so the strip has no recurrence to zig-zag along
and every ATM still reprices exactly. `Event_Days` is then a one-day segment of `ξ` at
`Event_Variance_Prior` times the level enclosing it — unless the day already HAS a segment of its
own, in which case the ordinary pillar bootstrap sets it, the prior is ignored and the report says
so. The curve is comparable to a market object: it should sit on a replicated variance-swap strip up
to the smile's own ATM convexity, and the report prints it beside the market's ATM² forward variance
rather than beside a log level. **No `e^{L*}` appears in any report.**

**The residual is an NIG increment on the variance clock, and its drift is forced.** The part of a
return leverage does not explain spends the clock `A = Σ c_k V_k` with `c = 1 − ρ_s² − ρ_ℓ²`, as
`X_A ~ NIG(α, β, δ_A, μ_A)` with `γ = √(α² − β²)`, `δ_A = A γ³/α²` (which makes `Var(X_A) = A`
exactly) and `μ_A = δ_A(√(α² − (β+1)²) − γ)` (which forces `E[e^{X_A}] = 1`). Both are LINEAR in
the clock, so the residual composes exactly: a month cut into days is one law once the clock is one
number, and a fixing interval straddling a bucket knot splits its clock there — two mixers, one
Gaussian whose `M` and `Σ²` sum. `α` is tail thickness and smile convexity, `β` is skew, and there is
no intensity, no jump size, no compensator to derive per segment and **no parameter the Greeks
cannot reach**: the drift is no-arbitrage's, not the fit's. What replaces the Poisson intensity rule
is nothing at all — that strip existed because vanillas do not identify an intensity apart from the
jump sizes, and `(α, β)` are identified by the 1–3 month wings. The pair is fitted in the
UNCONSTRAINED coordinates `α = ½ + ε + softplus(a)`, `β = −½ + (α − ½ − ε)tanh(b)`, which land
inside `|β| < α` and `|β+1| < α` for every iterate; the transform lives in the calibrator, the model
applies none, and the factor asserts admissibility — `|β| < α`, `|β+1| < α` and the conditioning
share `γ²/α² = 1 − (β/α)² ≥ 0.4` — by name at load.

**`Residual_Law: Gaussian` is a limit/test mode.** It drops the mixer, reads the clock as the
variance and the drift as its own compensator, and leaves `Alpha` and `Beta` unread (they may be
absent). The factor logs it at INFO by name; the calibrator refuses to warm start an NIG block off a
Gaussian factor, or the other way round, because the two are different models and neither seeds the
other.

**The quanto drift lives INSIDE the walk, read off the state's own budget.** A quanto payoff — an
index in one currency paid in another at a fixed rate, which is what the desk's autocall book is —
changes measure to the payoff currency, and the GBM arm applies that as one deal-level carry
`−ρ σ_ATM σ_FX` off a LOGNORMAL implied ATM vol read at the row's expiry (`pricing.calc_vol_adjustment`).
Under a walking kit the equity has no implied vol to read: its instantaneous variance is the
state's own and it moves along the path. So each business day's leverage mean gains
`−ρ_q σ_FX,k √(V_k δ_k)` — the day's own budget `V_k`, so the quanto drift follows the variance
path — and `calc_vol_adjustment` hands the walking arm `(ρ_q, σ_FX)` in place of the carry, which
it returns as zero. `utils.lv_walk` takes the loading `q_k = ρ_q σ_FX,k √δ_k` as one more per-step
tensor and adds `−q_k √V_k`; absent, the arithmetic is bit-identical, and under `Invert_Spot` it is
added the same way, a drift being a drift on either axis.

`ρ_q` is the book's marked `Correlation.EquityPrice.<eq>/FxRate.<pair>` read through the same
`Correlation_Sign` the GBM arm resolves, and it is the desk's number read as a **total-return
correlation** — the ruling of 2026-09-08, and what a desk measures. It is applied to the day's own
return sd `√V_k`, and that IS the un-diluted framework correlation: the framework applies a
correlation to the Gaussian GIVEN THE MIXER, whose sd is `E[Σ_k] = D √V_k` with
`Σ_k² = (ρ_ℓ² + ρ_s²) V_k + G_k`, so a loading on `√V_k` is `ρ_q/D` applied to `Σ_k` and the
normalisation the ruling asked for is the identity. `D` is **1 exactly** under a Gaussian residual
(`Σ_k ≡ √V_k`), **0.9256** at the Q-sized NIG defaults and **0.597** on the desk's NKY fit (`c`
0.84, `α` 7.8). Measured at 2^17 paths on the daily grid, three joints at the same draws: the
joint whose realised total-return correlation IS the marked −0.40 puts the exact quanto forward
`E[SX]/E[X]` within **1.2 SE** of the loading as built, where the loading times `D` sits 8.4 SE
away and the loading over `D` — the ruling's arithmetic taken literally — 11.6 SE away, so the
literal division is wrong-signed. The 91.6–92.0% lane Q measured is the OTHER reading: the marked
ρ applied directly to the Gaussian given the mixer, which is the loading times `D` — worth 1.6% of
the desk's NKY V2 at the Q-sized `D` and **8.4% of the mark** (3.61m ZAR) at the fit's own 0.597.
`σ_FX` is the FX surface's ATM FORWARD
strip on the WALK's own grid — read at every internal step's tenor and differenced by
`pricing.forward_vol_rate` — where the GBM arm reads one expiry ATM and calls it the whole deal's.
On a FLAT FX surface the two are the same number and the walk reproduces the closed form to 8e-16
relative; on a surface sloping 2 vol points of ATM per year they differ by 0.99% of a 2y autocall's
value, 2.72% at 5 points per year. The fx surface's ATM rows are on the tape, so the quanto vega
comes out with the rest (0.000% against its CRN ladder); the CORRELATION delta does not, because
`Correlation` is a `DimensionLessFactor` with no leaf, so `BaseValuation` reports it as a central
CRN bump of the marked value instead (`Correlation_Bump`, default 0.025, `Greeks: 'First'` only,
one re-compile and re-value each side on the job's own seed for each correlation a priced quanto
or compo read, so a book with none pays nothing): the row lands in `Greeks_First` under the
factor's own name and in a `Correlation_Bump` block that names the width and says NOT ON THE
TAPE. The desk's NKY V2 reads −22,418,800 ZAR per unit of ρ at 32,768 paths, flat to 0.003%
between half-widths 0.05 and 0.025 (the value is linear in ρ, as a drift linear in ρ makes it),
and lane Q's 2y SPX autocall −10.95, flat to 0.001%. Only
`QEDI_CustomAutoCallSwap`/`_V2` passes the loading today; `Compo` stays refused by name on every
walking deal, being the product `S·X` and so a second asset the arm does not simulate, and a quanto
naming no `Correlation` factor refuses rather than pricing with a drift of exactly zero.

**The ξ strip is re-bootstrapped at EVERY outer iterate**, so every candidate reprices the ATM term
structure exactly and is judged on the smile alone; the ATM misses the report prints are 1e-16 to
1e-13 rather than the 1e-12 to 1e-9 a between-stage refit left. The level each segment returns is one
Newton step at its own root, so `dξ/dθ` rides the tape and the outer solver keeps its exact Jacobian.
What that costs is one graph pass per segment per iterate: the search itself runs off the tape on
the previous sweep's slope. **The walk is the block's CLOSED FORM, not a scan.** Both factors are
linear in their own shocks, and the OU transition `exp(C_k − C_{j+1})` on the cumulated `−κδ`
factorises, so the whole block's state path is one cumulative sum of the discounted shocks scaled
back by the cumulated decay (`utils.lv_ou_path`) and the clock, the leverage mean and the quanto
drift are elementwise over the step axis with one reduction each — tens of dispatches a block
where the scan spent fifteen a step. At 8192 paths over 690 daily steps a forward pass is **0.156 s against the scan's 0.349 s**, and
with its backward **0.251 s against 0.704 s**; the same walk on an RTX 3090 is **0.0062 s and
0.0099 s**, twenty-five times the CPU, where the scan measured SLOWER on the card than on the
host. **THE FIT RUNS ON THE JOB'S DEVICE**, which on a CUDA box is the card: the constructed
device is no longer ignored, and the two things the old note held the pin for are answered rather
than avoided. THE STREAM IS THE SEED'S AND NOT THE SILICON'S — `draw` generates both streams on
the HOST under `Random_Seed`'s own generator and moves the tensors, so `Pseudo` 8192 on the card
is the draw it was on the CPU and a banked fit re-fits to it; a CUDA generator is a different
stream, and a seed that named a draw only together with the device it was drawn on would make "the
same answer" undefined. THE LEAF LIVES ON THE CALCULATION'S DEVICE — `Calculation.factor_leaf`
already moves the calibrated `theta` with `.to(device=self.device, dtype=self.dtype)` before it
splices `leaf + (theta - theta.detach())`, and that copy is differentiable, so the calculation's
backward reaches the fit's graph across a device boundary and `dV/dq` arrives on the fit's own
quote leaf. A CPU-sharded calculation on a CUDA box is therefore a CPU calculation reading a leaf
fitted on the card, which is what that seam was always shaped for.

**The clock moved with it, and the profile changed shape.** The capped NKY profile
(`artifacts/lv_fast2_20260908/profile_fit.py 3 8192`) is **126.8 s on the CPU against 17.3 s on
the card**, and the item that was 61% of the CPU clock — the batched reverse Jacobian — is 4% of
the card's: the 25-row backward over a 504-step block at 8192 paths costs **6.78 s on the CPU and
0.104 s on an RTX 3090**. What is left is the inverse-Gaussian root at 30% and the pillars' own
backwards at 28%. `ig_root`'s fixed budget of 34 masked Newton steps is roughly 1,200 kernel
launches on a `[paths, blocks]` tensor of a few tens of thousands of doubles, so its cost is
DISPATCH: **34 ms whether the strip is one block or four**, which is two and a half times a whole
690-step walk. One consequence for a reader tuning a block: **the path count is no longer a speed
lever on the card.** An eight-fold cut in `Paths` buys about 1.4x an evaluation where the CPU's
vectorised walk gave most of eight.

**The inner bootstrap is a damped NEWTON, not a chord.** A pillar's pass has to carry a backward
anyway — the level returned is one Newton step at the root, which is what puts `dxi/dtheta` on the
tape — so the slope that step is taken with is already paid for, and a chord off the PREVIOUS
sweep's slope only buys a cheaper step at the price of spending more of them. Measured on the four
book ladders, a sweep costs **2.1–2.5 pillar passes a pillar against the chord's 3.5–4.3**, which
is 1.64–1.70x fewer passes and **1.33x on the wall clock**, at the same evaluation count and the
same RMSE — and it solves the strip TIGHTER, the worst ATM miss falling from 8.8e-11 to 6.5e-12 on
`USD_NDX_INDEX` and from 6.7e-11 to 9.1e-12 on `JPY_NKY_BBG`. A slope of exactly zero now divides
in TENSORS to an infinite step the damping bounds, so the `ZeroDivisionError` a Sobol stream once
reached has no bare float left to raise on.
`Invert_Spot` keeps the recursion: the S-numeraire shift is the state's own `√V`, which makes the
transition state-dependent, and the block sums below it are the one spelling either way. The grid is the QUOTES' own, one trading day
between block ends with a stub landing each block on its `T`: reading the same rung on the
trading-day grid instead costs **0.124 vol points** at the 1m ATM (`jac_check.py`).

**`Sampling` picks the stream, and it is what sets `Paths`.** The draws are fixed for the whole
fit, so `Paths` is not a confidence interval around θ\*, it is the NOISE FLOOR under it: the same
ladder at another `Random_Seed` lands somewhere else, and how far is the only honest reading of
how much of a fitted number is the surface. `Pseudo` is the generator this family drew from
before the field existed and is what a bit-identity gate against a banked fit declares. `Sobol`
is a scrambled sequence over the `2 × steps + blocks` dimensions in the calculation's own
convention (`calculation.CMC_State.quasi_rng` — one engine at `QUASI_ANCHOR`, that clamp margin,
that `norm_icdf`), scrambled off `Random_Seed`; a ladder wider than `SOBOL_MAX_DIMENSION` refuses
by name rather than chunking, the calculation chunking because a scenario grid can be that wide
and a calibration grid that is saying the grid is wrong. **`Pseudo` STAYS THE DEFAULT** and `Paths` stays 8192, and the PRODUCTION objective — priors on
and the forward rows in, which is what a book fits under — is not the cure. Across `Random_Seed`
1–5 at `Pseudo` 8192, `Beta` changes SIGN on `EUR_SX5E`, `USD_NDX_INDEX` and `JPY_NKY_BBG`, and
`Alpha` runs from 110 to its box top of 500 on `JPY_NKY_BBG`, inside RMSE bands 0.03 to 0.25 vol
points wide. **That is an identification failure and not a noise floor**: the ATM strip is pinned
by the inner bootstrap, and what wanders is exactly what the WING quotes are supposed to identify.
No stream is uniformly tighter either — Sobol holds `Alpha` inside 2.1 where `Pseudo` spreads it
over 107.7 on `USD_NDX_INDEX`, is seven times WIDER on `EUR_SD3E`, and is indifferent on
`JPY_NKY_BBG`, where every stream spreads `Alpha` over its whole box. A default cannot be set
against a quantity the objective cannot tell apart, and a same-answer gate on such a ladder proves
determinism rather than agreement.

**The defaults, with the numbers that set them** (lanes S and S2):

| field | default | the number that sets it |
| --- | --- | --- |
| `Sampling` | `Pseudo` | under the PRODUCTION objective `Beta` changes SIGN across `Random_Seed` on three of the four book ladders, and no stream is uniformly tighter (§2) |
| `Paths` | 8192 | the same reading; and on the card the path count is no longer a speed lever, an eight-fold cut buying about 1.4x an evaluation |
| `Tolerance` | 1e-8 | lane S: 1e-6 buys ONE evaluation of forty-four and moves theta\* by 4e-5 relative |
| `LV_IG_EXPAND`, `LV_IG_STEPS` | 3, 34 | lane S: the worst element converges in 26 geometric steps against 53 arithmetic. NOW 30% of the card's clock — the lever left, and a ruling rather than a measurement |
| `LVFit.l_iterations`, `l_damping` | 12, 0.5 | the budget is early-stopped, so it costs nothing unused; measured at 2.1–2.5 passes a pillar |
| `Forward_Smile_Source` | `None` (`Prior` withdrawn) | a sticky-delta TARGET costs 0.4–1.1 vol points of spot fit on all four index ladders and a 0.5y bucket does not repair it; the tie-breaker that capped that cost bound stage 5 alone, and on a one-bucket ladder stage 5 does not run |
| `Residual_Horizon` | 0.25 | since lane L2 it switches no row off: both residual rows stay and the wings outvote them. Its one reader is the report's own line saying whether the ladder's wings reach the residual. The +23 / +13 / +15 the wings alone landed on NKY was the leverage's mis-allocation, not the horizon's; horizon zero was the wrong fix |
| `Alpha_Prior_Defaults` | 44 for every class | brief 2 sizes the residual ONCE, at the index `(44, −22)`, and states no separate FX number; a soft row beside the wings since lane L2, where a walk to the 500 ceiling costs 5.1 standard errors |
| `Alpha_Prior_Sd` | 0.5 on `log α` | brief 5's log-normal spread, a factor of 1.65 either way — and the bar a history's own SE must beat to be called informative |
| `Residual_Skew_Share_Defaults`, `Residual_Skew_Share_Sd` | −0.5 for every class, 0.2 | the −22/44 sizing restated as the share the smile sees (lane L2); `Beta_Prior_Defaults` and `Beta_Prior_Sd` are retired and refuse by name. The 0.2 is asserted, not measured: what would measure it is the estimator's `β^P/α^P` spread on an uncontaminated history, which no index history here is |
| `Contamination_Ratio` | 2.0 | brief 8's `C_Eff > 2c`; the estimator's own report already names the ratio at this threshold |
| `Leverage_Prior_Weight` | 0.02 | one instance of the general rule: `0.01 × 0.2 / 0.1` is one quote-vol-point per tenth of `ρ_s`. Since lane L the row is on the product and the scale is this over `Sigma_S_Reference`, the same statement at the reference vol-of-vol; a history's product carries a delta-method error and is weighted by it. At a fitted `σ_s` of 5 a unit of `ρ_s` is five of product, so the row reads 6.5 quote rows on `ρ_s` where it read 2.3 — open decision 16 |
| `Leverage_Product_Defaults` | index −1.9, FX 0.0 | brief §2: `ρ_s σ_s ≈ −1.9` from VIX-vs-SPX daily co-movement; the ruling's number over the spec's −1.8 |
| `Sigma_S_Reference` | 2.4 | the state's own `Sigma_S` seed, the Q-sized vol-of-vol a declared `Leverage_Prior` is multiplied by to reach the product; refused at or below zero |
| the fit's device | the job's | the walk is 25x on an RTX 3090, the batched Jacobian 65x, and the capped NKY profile 126.8 s → 17.3 s |

**`Tolerance` is `ftol` on a MONTE CARLO objective, and it is not what a stage stops on.** Asking
a fixed-draw sample mean to converge to 1e-8 looks like asking it to converge to its own rounding
— but measured on the NKY chain block at `Pseudo` 8192, two orders of magnitude of `ftol` (1e-8
against 1e-6) buy **one evaluation out of forty-four** and move θ\* by **4e-5 relative**. The
stage is stopping on `xtol` (1e-12, hard-coded beside it), on `gtol`, or on the step. **The
default stays 1e-8** because loosening it is free of cost and free of benefit; a stage that runs
long is a stage with a column the data does not move (see the identification table), not a stage
converging to rounding.

The inverse-Gaussian root's safeguard BISECTS GEOMETRICALLY. Its bracket spans ten decades —
`[m·1e-8, 200m + 200m²/λ]` — so its midpoint is a ratio, not a width: measured over 2e5 uniforms
with the tails out to 1e-300, at clocks 1e-6 to 3 across the whole admissible `(α, β)` box, the
arithmetic halving needs 53 Newton steps against the geometric one's 26, and the bracket never
doubles at all. The fixed budget is therefore `LV_IG_EXPAND, LV_IG_STEPS = 3, 34` where it was
`20, 60` — the same root to `LV_IG_TOL`, reached in 37 CDF evaluations instead of 80.

The CJOW harness tables that stood here were readings of the Poisson residual against a reference
surface priced by the retired Heston-Nandi half of `utils`; the harness could not be re-run after
that retirement and has been deleted with its banked premiums, so the record they carried is the
Poisson residual's and is not re-measured. The flat-segment ruling that preceded them stands on its
own measurement: the piecewise-linear curve's strip alternated across the market's by the `−1`
multiplier of a segment integral that reads both ends, and flat segments took the strip's second
difference from 20.4 vol points RMS to 5.1 on the CJOW surface and from 1.57 to 0.42 on a reduced
USDZAR block at unchanged wing RMSEs.

**Two modes on one banked surface** (`artifacts/lv_nig_20260907/fit.py fx`, the campaign's USDZAR
`FXVol` at `Paths` 8192, daily δ, CPU), against the SAME 22 contracts the Poisson residual was
read on. Six of those rungs are the ATM one of each expiry, which any family carrying a curve
solves to zero, so folding them into one RMSE reports the split rather than the fit: the table
separates them and the headline is the WING RMSE over the other 16.

| fit | ATM rungs | 16 WING rungs, RMSE | worst wing | **one-month wing** | wall clock |
|---|---|---|---|---|---|
| LogVar2FJ v2 `Global` | 6 at **0.0e+00** vol points | 0.177 | +0.403 | **0.296** over 5 | 1,661 s |
| LogVar2FJ v2 `Bootstrap` | 6 at 0.0e+00 | **0.222** | −0.439 | **0.032** over 5 | 435 s |
| *the Poisson residual's record, `Global`* | 6 at 0.0e+00 | *0.132* | *−0.252* | *0.174* | *213 s* |
| *the Poisson residual's record, `Bootstrap`* | 6 at 0.0e+00 | *0.574* | *+1.291* | *0.043* | *222 s* |

and on the banked SPX chain block (`artifacts/hnret/chain_block.py`, six rungs, `Max_Iterations`
4): **0.158** vol points RMSE against the Poisson residual's **0.272**, in 26 s. The two retired
Heston-Nandi families read 0.663 and 0.760 on the same 22 contracts, a record nothing reproduces.

**THE ONE-MONTH WING IS THE NUMBER THE RESIDUAL WAS SWAPPED FOR, and it reads two ways.** On the
equity chain the swap is worth 42% of the fit and on the USDZAR ladder in `Bootstrap` mode 61%; in
`Global` mode on the same ladder it is 34% WORSE. The reason is in the fitted shape: a USDZAR smile
is nearly symmetric, so `Global` has no use for residual SKEW and takes `β → −0.08` while driving
`α` down to **5.45** for convexity instead — where the residual's own shape `α·δ_A` is 7.7e-03 at
one month against the "≈ 1 strongly non-Gaussian" the sizing note gives, and the mixer's
coefficient of variation is 11.4. Every declared guard is satisfied there (`c` 0.311 ≥ 0.12,
`γ²/α²` **1.000** ≥ 0.4). `Bootstrap`, which frees `α` per expiry against that expiry's own wings,
does not reach the corner (`α` 41.6, `γ²/α²` 0.930) and reads an eighth of `Global`'s one-month
wing. **Nothing in the brief bounds `α` from below**; the residual-shape floor below closes it. `Bootstrap` is structurally the mode for a deal read at many
fixings rather than the mode that fits a surface best — bucket `k` acts only on `[E_{k−1}, E_k)`
while the option to `E_k` averages over every bucket before it — and the report says per bucket
which parameters were free and which tied.

**The leverage prior is never absent, it is on the PRODUCT `ρ_s σ_s`, and it is stated on the
ENGINE'S axis.** Vanillas do not separate residual skew from leverage skew: `β` and `ρ_s σ_s` both
bend the spot smile the same way, so a vanilla-only fit gives the skew to whichever is cheaper and
leaves the FORWARD smile — the thing an autocall reads — undetermined. The row is a weak soft term
on the product in every stage's objective, one per fitted bucket, because a prior on `ρ_s` alone
was measured letting the fit buy the product it wanted through the factor nobody priced: on NKY
`ρ_s` −0.59 sat obediently near its −0.7 prior while `σ_s` ran to its 5.0 box, a product of
**−2.97** against the VIX-implied **−1.9** brief §2 sizes it to, and the over-sized leverage
suppressed the call wing until the residual took a POSITIVE skew (`β` +23 / +13 / +15 over three
seeds) to lift it back — two mechanisms compensating along the direction vanillas cannot see, and a
right-skewed residual on an index carried into every terminal put on the book (the owner's ruling
of 2026-09-09). **It is TWO rows** (lane L2, 2026-09-10): one on `ρ_s` and one on the product,
because a product row alone was met on NKY by `ρ_s` falling to −0.37 with `σ_s` still on its 5.0
box, which the log-vol band consents to. Each is read in one order — the block's own declaration
(`Leverage_Prior`, a `ρ_s` on the engine's axis, with an optional `Leverage_Prior_SE`;
`Leverage_Product_Prior` with `Leverage_Product_Prior_SE`, a blank product being
`Leverage_Prior × Sigma_S_Reference`, 2.4), else a LogVar2FJ history's `Rho_S ± Rho_S_SE` and its
`Rho_S × Sigma_S` with the delta-method error, else the class defaults `Leverage_Prior_Defaults`
(an index −0.7, FX 0.0) and `Leverage_Product_Defaults` (an index **−1.9**, FX **0.0**). A
declared error weights its row at one quote per standard error; a blank one takes the nominal
weight, `Leverage_Prior_Weight` on `ρ_s` and that over `Sigma_S_Reference` on the product. **The
block's own numbers come from the IMPLIED REGRESSION** brief §2 sizes the reference by — the daily
log change of the index's implied-vol index on the index's return, `σ_s` the annualised sd of the
log-variance change, with Fisher, `σ/√2N` and delta-method errors
(`artifacts/lv_corrtest_20260909/fetch_vi.py`): SPX/VIX reads ρ_s −0.762 ± 0.012, σ_s 2.48,
product −1.89 ± 0.05, the reference to the second decimal; NDX/VXN −0.712, 1.88, −1.34; NKY against
the Nikkei VI **−0.529 ± 0.025, 3.07, −1.62 ± 0.09** — a weaker correlation and more vol-of-vol
than SPX, which is why NKY's wings wanted a positive residual skew at SPX's −1.85: it is not NKY's
number. `Leverage_Prior_Defaults` also signs the seed (`sign·0.75`) and the slow pair's pin, and a
class whose product and `ρ_s` defaults disagree in sign refuses by name.

That history now has a producer, and where the ladder cannot see the residual pair it is one of the
three things that can anchor them. `Rho_S`, the slow pair, `Alpha` and `Beta` each arrive with their
own standard error and **each enters as a SOFT ROW weighted by it** — never a pin, so the wings
still move every one of them, and each stays in θ\*, in the Jacobian, in the identification table
and in the quote contraction, where a pinned coordinate is in none of them. Every history number is
also **CLIPPED into the box the fit moves that lever in**, and the report names the clip: a prior
the fit cannot reach is a refusal by another name, and an unclipped one is worse than that — a slow
pair pinned at a historical `ρ_ℓ` of −0.83 went straight through `Rho_L_Bounds` (−0.6, 0), crushed
the derived `ρ_s` box from ±0.94 to ±0.43 under a value stage 3 had already fitted in the wider one,
and took stage 5 non-finite. **A history's slow pair is a row like the rest**; what a ladder with no
18-month wing is PINNED at is `Slow_Factor_Prior` where a block declares one, else the class default,
and nothing else.

**Every prior row costs the same, and the price is one quote.** A prior row is
`(x − x^P)/SE_x` scaled so that **one standard error of miss costs exactly what one quote missing by
one vol point costs in the objective on the ladder at hand**. The arithmetic: the quote rows are
decimal-vol misses times weights normalised so `Σ w² = 1 − Forward_Weight`, so one quote missing by
a vol point costs `0.01 × √((1 − share)/N)` — **0.0025** on the sixteen-rung NKY ladder, 0.0021 on
the twenty-two-rung USDZAR one — and that number is the multiplier on every prior row. It is the
same statement `Leverage_Prior_Weight` 0.02 already made, with 0.1 standing in for the standard
error nobody declares on a class default: at the 0.2 quote weight that field's prose quotes, `0.02`
IS `0.01 × 0.2 / 0.1`. A prior with a standard error of its own is weighted by that instead. A row
scaled any other way is a hard pin — the first cut of this rule was a bare z-score, which charged
400 quote-vol-points per standard error and landed `β` on six figures of the history.

Read `α^P` as an UPPER bound on the residual's tail thickness rather than as a measurement of it:
the estimator fits the law of the diffusive REMAINDER, and at an index-sized leverage (`c` 0.27) the
remainder is nearly all leverage the smoothed shocks could not remove — its clock share reads 0.997
against `c` — so the fitted `α` is pulled toward Gaussian. Past `Contamination_Ratio` (2) times the
model's own `c`, `α^P` is **reported and not used** whatever its standard error, and a history block
carrying no `C_Eff` cannot be read at all and counts as contaminated.

**THE HIERARCHY, and what 'informative' means.** Where the ladder carries no wing under
`Residual_Horizon` it does not price the 1–3 month tails at all, and `α` and `β` take priors in a
hierarchy of three, the tier in force named in the report on every fit:

| tier | when | the row |
| --- | --- | --- |
| **the short-dated wings** | the ladder quotes a wing at or under `Residual_Horizon` | the rows stay and the wings OUTVOTE them (lane L2, 2026-09-10): the report says the wings reach the residual, and `Residual_Horizon` reads nothing else |
| **the history** | its own standard error is at or under the class prior's spread, and its `α^P` is not contaminated | `(x − x^P)/SE^P` at that estimate |
| **the class default** | otherwise | `(x − x_class)/SD_class`, reported as *class prior, history uninformative* with the standard error that failed |

**The two residual rows are on the coordinates a smile sees** — `log α` at `Alpha_Prior_Defaults`
44 with spread `Alpha_Prior_Sd` 0.5, and the SKEW SHARE `β/α` at `Residual_Skew_Share_Defaults`
−0.5 with spread `Residual_Skew_Share_Sd` 0.2, a history's `β^P/α^P` with its delta-method error
where informative. A prior has to sit on a coordinate the price depends on, or the optimiser
satisfies it for free: the row lane L put on `β` alone was obeyed on NKY and SX5E by running `α` to
its 500 ceiling, where any `β` is a label on a Gaussian (`|β|/α` 0.044, the `β` row 690 quote rows
against quote rows worth 1.8e-04). `Beta_Prior_Defaults` and `Beta_Prior_Sd` are RETIRED and refuse
by name naming the share; the residual seed is `share × α_seed`, −22 at the defaults to the digit.

**That test IS the definition of informative**, and it is declared: `Beta_Prior_Sd` (10, sized in
brief 5 on the residual's one-month skewness) and `Alpha_Prior_Sd` (0.5 on `log α`, a factor of 1.65
either way) are both the class spread AND the bar a history has to beat. It discriminates the way
the brief's own two measurements do: on a simulated index history `β^P` reads **−0.28 ± 14.4**
against a truth of −22 and FAILS, so the class prior `Beta_Prior_Defaults` **−22** stands; on the
FX-sized test `β^P` reads −20.5 ± 7.4 against −15.4 and PASSES. `α` is a scale, so its prior is on
`log α` and a history's own error is read in the same units. `β^P` crosses only as a prior with its
standard error: an Esscher tilt moves `β` by one unit at most, so the P and Q skews are not the same
number and the sanity table still says so on its row; what the prior states is not that they are
equal but that the ladder has nothing to say.

**`Residual_Horizon` stays 0.25, and the positive skew it left free was the leverage's doing.**
Over `Random_Seed` 1–3 on the NKY chain block at 8,192 paths with no history, the wings alone
landed `β` at **+23.11, +12.68 and +14.85** at `|β|/α` 0.74 / 0.68 / 0.74 against the 0.77 bound —
spec 5.3's own failure mode, reached by the wings — and horizon zero forced it to −13.76 / −20.11 /
−14.35 for 0.43–0.49 vol points while leaving `σ_s` on its box, the wrong fix for the right
symptom. With the product row in force (lane L, 2026-09-10) `β` is negative on every book fit:
NKY **−21.89 / −21.97 / −21.91** with the product −1.85 in place of −2.97, SX5E −21.97, NDX −20.99,
SD3E −19.86, at 1.614 / 1.083 / 1.000 / 1.034 vol points of unweighted RMSE against 1.308 / 0.845 /
0.943 / 0.390, and the three ladders the old row left with `σ_s` on its 0.5 FLOOR (a product a tenth
of Q size) are pulled off it to 3.96 / 4.89 / 1.78. What the class-default tier still cannot do is
put NKY's `σ_s` inside its box: at 5.0 the stationary log-vol sd reads 0.878, inside the 0.4–0.9
band, so the box and the band collude and the ladder wants that vol-of-vol — reported with the
flag below, the box not widened. And the row `β` obeys is on `β` while the residual's skew is
`|β|/α`, so on NKY and SX5E the fit obeys it for free by running `α` to its 500 ceiling
(`|β|/α` 0.044, the residual effectively Gaussian, the whole smile leverage at the Q size) — an open
row. The history tier is where the ruling's expected shape lives: NKY with the index history lands
`α` 7.99, `β` **−2.35**, `|β|/α` 0.29 at **1.171** vol points, better than the clean tree's own
anchored 1.180. On the desk's NKY autocall 229524957 the mark moves −37.49m → **−38.94m ZAR** and
the forward-skew reserve **falls 11.71m → 7.21m** (38%): with the residual near Gaussian its half of
`Skew_Gradient` is nothing and the reserve is the `ρ_s` half alone.

**On-guard fits are reported and flagged, never stood behind.** A fit a box or a floor is holding
is not a fitted one: `LVFit.on_guard` names every guard θ\* sits on — a `Sigma_S`, `Alpha` or
`Sigma_L` on either edge of its declared box, `|β|/α` within `C_Margin` (0.05) of the conditioning
bound 0.7746, `c` within `C_Margin` of its `C_Min` floor, or **a prior row whose column norm exceeds
one quote row's in its own coordinate by more than `LV_PRIOR_RATIO` (100)** at the last stage that
fitted it — the signature of a prior on a coordinate the data cannot see, which is what the
identification line now prints per row (*Rho_S[0y] 38.5x, Beta[0y] 5.92x …*) — in one sentence,
logged as a WARNING by the report, written on the factor as the structural Text field `On_Guard`
(blank where clean), logged at INFO when the factor loads, and collected by `Base_Revaluation` into
`Stats['On_Guard']` per factor it priced off, so a mark carries the flag. Scored on lane L's own
fits the fourth guard names the α escape in one number (`Alpha[0y] 566x … Beta[0y] 551x` on NKY,
SX5E at 532x where the three box rules had read it clean), and on lane L2's it names one thing: NDX
at the regression's own 0.015 error, whose `ρ_s` row is 138 quote rows — a 1.5%-standard-error
desk view is a pin on that ladder, and the mark says so rather than standing behind it.

**With the rows on the smile's coordinates NKY lands where the ruling said** (lane L2): at its own
implied leverage over seeds 1–3, `α` **39.4 / 40.9 / 38.8**, `β` −11.4 / −12.9 / −11.9, the share
−0.29 / −0.32 / −0.31 (the wings outvoting the −0.5 row by about one standard error), `ρ_s` −0.523,
`σ_s` **3.105 / 3.105 / 3.109 inside its box** at the ratio of the two rows, the product −1.62, and
`On_Guard` BLANK — the first clean NKY fit in the programme. `α` is off its ceiling on every ladder
in the book (25.6 to 53.9 across seven fits) and `σ_s` inside its box on every ladder but the
history-anchored NKY. The price is vanilla fit, NKY 1.614 → **1.985** vol points with the miss in the
one-month wing, and the desk's autocall marks **−36.29m ZAR** at 32,768 paths (lane L's −38.94m,
P2's −37.49m) with `Skew_Reserve` **7.64m**, the residual now live in it. At the CLASS defaults
(SPX's −1.9 with a −0.7 `ρ_s`) NKY REFUSES by name at production `Cap_A` on the cap headroom, and
at `Cap_A` 6 lands `σ_s` on its 5.0 box with `ρ_s` −0.397: an index class default is not a shape
this model can carry on NKY, which is the argument for the per-name implied numbers.

THE AXIS IS THE POINT. An `FxRate` is priced in the domestic currency, so `FxRate.ZAR` in a USD book
is *USD per rand* and a desk's `+0.4` on an EM cross quoted USD-per-currency is **−0.4** here. The
declaration is DATA the desk owns: `derivus_bloomberg/seed.json`'s `fx_vol.leverage_prior` states it
per pair on that axis (`USDZAR`/`EURZAR`/`GBPZAR` −0.4, the majors 0.0), `security_map.leverage_prior`
reads it off `$DV_HOME/seed.json` where that exists — a desk seed that predates the key declares no
prior and gets the class default — `/book/model` hands it to `fx_surface_block`, and the emitted
block carries it with `Quote_Source` naming the seed. A block declaring `Leverage_Prior` overrides
all of that and the report says *from the declared Leverage_Prior*.

**With a prior the box is SYMMETRIC.** `ρ_s ∈ [−ρ_max, +ρ_max]`, `ρ_max = √(1 − ρ_ℓ² − c_min)`,
re-derived as `ρ_ℓ` moves. Half a box is an assertion about which way a smile leans, and the prior
is where that assertion belongs; the seed is `sign(prior)·0.75` and a zero prior seeds −0.75 and lets
the box decide. `Model_Priors` governs every soft term at once: the leverage prior on `ρ_s`, the
floor on the residual shape `α·δ_A`, the residual pair's own priors where the ladder does not
identify it, and a history's slow pair where the ladder fits it — all four scaled to the one price
above. `Off` restores the one-sided box with the vanilla-only objective it replaced, and is the
switch a bit-identity gate runs under.

**`Residual_Shape_Floor` is the lower bound on `α` that brief 2 does not have.** `|β|/α ≤ 0.77`
bounds the skew share and nothing bounds `α` from below, so on a nearly symmetric smile the fit has
no use for residual SKEW and buys CONVEXITY instead by walking `α` to the admissible map's own
softplus floor — the 22-rung USDZAR ladder in `Global` landed at `α` 5.45 with `α·δ_A` **7.7e-03** at
one month, on a scale where 1 is strongly non-Gaussian and 15 nearly Gaussian, and with the floor
off it walks on to **0.627**. The guard is one soft term on that shape at the SHORTEST calibrated
expiry, `0.05 · relu(1 − shape/floor)`: relative, so a fit inside the floor pays nothing and one at
the map's floor pays five vol points of residual, against wing misses of a few tenths. Any floor at
or above 0.1 lands the same θ\* (`α` 48.5, shape 1.358), so it excludes a BASIN rather than setting a
value. It is a FIELD and not a box because the number is a modelling choice and `0` is off. With
the floor and the desk's −0.4 together the ladder's one-month wing reads **0.134** against lane 1's
0.305 and the Poisson residual's 0.174, and the stationary log-vol sd comes back inside the 0.4–0.9
band (0.353 → 0.515) without the band moving.

Both modes REFUSE on the default `Stationary_Spread: Refuse`, and that refusal is the reading: with
the residual carrying the skew the fit no longer needs the vol-of-vol to carry it, so the
stationary log-vol sd lands under the 0.4–0.9 the VIX market implies and the guard names the
bucket, the sd and the pair that produced it. The rows above are read under
`Stationary_Spread: Floor`, which takes the same θ\* — the guard is a READING of θ\*, never a
re-solve — and says so. This is the identification statement made concrete: vanillas cannot tell
residual skew from leverage skew, so a vanilla-only fit gives the skew to whichever is cheaper, and
the forward-smile block is what separates them.

**The slow factor's prior is per ASSET CLASS, and the floor is the guard beneath it.** Where the
ladder carries no wing at 18 months or longer, `(ρ_l, σ_l)` are fitted to nothing and are pinned
instead. What they are pinned at is read in one order — `Slow_Factor_Prior` where the block declares
one, else a LogVar2FJ history for the underlying in `Price Models` (the `utils.LV_SLOW_HISTORY`
shape — the one place both lanes declare it — which is `Rho_L`, `Sigma_L`, `Alpha`, `Beta`, `Rho_S`
and `Sigma_S`, EACH WITH ITS OWN `_SE`; a block missing any one of the twelve refuses by name from the write
side. The slow pair is reported with both standard errors and taken to the floor by name where it
sits under it, as a hand-authored `(−0.35, 0.22)` with SEs 0.09 / 0.14 reads back *held at the
history's estimate … its Sigma_L 0.2200 TAKEN TO the 0.3 floor*; the block is written by
`stochasticprocess.LogVar2FJCalibration`, the P-measure estimator, and the same block is where the
leverage prior's `Rho_S` and the residual seed `Alpha` are read from), else
the class default off the factor type `Underlying` resolves to: **FX (0.2, 0.5), an index
(0.4, 1.0)**, whose SIGN is that of the `Rho_S` in force at the pin. The sign is the point: the fit
runs on the `FxRate`'s own axis, so a USDZAR block whose deal convention has vol rising as the rand
weakens fits `ρ_s < 0` on `FxRate.ZAR`, and pinning `ρ_l = −0.4` from the index seed would be the
right sign here but the wrong one on the reciprocal — the rule reads it off the data instead of
assuming it. **On this engine it always reads negative**: spec 5.2 stage 3 boxes
`ρ_s ∈ [−√(1 − ρ_l² − c_min), 0]`, so no fit can produce a positive fast leverage and the FX default
resolves to (−0.2, 0.5) on every ladder — the rule is the right one and it is inert until that box
opens. A pin is applied to every name without an 18-month wing, so a floor-BY-DEFAULT would
understate a whole book's 2–5 year vol in one direction; the floor `σ_l ≥ 0.3` is the BOX beneath
all three instead, a declared prior under it refusing by name.

Where the pair is pinned the report reads the ladder under all three priors, at ONE extra forward
pass each — everything but the slow pair held at θ\*, the `L` strip re-bootstrapped so every ATM
still reprices, the wings re-priced, no outer search — with the wing RMSE and the **5-year log-vol
sd** `½√(σ_s²(1−e^{−10κ_s})/2κ_s + σ_l²(1−e^{−10κ_l})/2κ_l)`, which is the number a phase-3 exposure
row will read:

| pinned ladder | the floor (·, 0.3) | the class default | the index seed (−0.4, 1.0) |
|---|---|---|---|
| reduced USDZAR (`FxRate`), 12 wings | 0.143 / sd 0.515 | **0.132 / sd 0.553** (in force) | 0.324 / sd 0.701 |
| the 22-rung USDZAR ladder, 16 wings | 0.158 / sd 0.521 | **0.135 / sd 0.558** (in force) | 0.422 / sd 0.705 |
| CJOW cut to 4 maturities (`EquityPrice`), `Global` | 1.448 / sd 0.516 | **1.279 / sd 0.701** (in force) = the seed | — |
| the same in `Fit_Mode` `Bootstrap` | 1.330 / sd 0.482 | **1.056 / sd 0.677** (in force) = the seed | — |

Two priors that land on the same pair are ONE pass named for both, which is what an index's class
default IS. And the two classes want opposite things: on FX the floor beats the seed and the class
default beats the floor, while on the index surface the floor is the WORST of the three (1.448
against 1.279) — a floor-by-default would have been the wrong answer for every equity book, which is
the whole reason it is the guard and not the default. The row in force is exact; the others
APPROXIMATE the re-fit they are not, and the table says so, because a re-fit lets the fast pair take
back some of what the slow one gives. The size of that is
measured: the ladder's index-seed row reads 0.422 where the same ladder RE-FITTED at the index seed
banked **0.202**, so a distant prior's cost is overstated about twofold by one pass. What the
comparison is for is the ORDER, and the order is unambiguous — the FX class default costs the
22-rung ladder **0.135 against the 0.132** a slow pair fitted to nothing bought before the
identification rule existed, where the index seed cost 0.202. The whole 0.070 the rule appeared to
cost was the seed being wrong-sized for FX, not the rule.

**Risk in quote space.** `Quote_Sensitivity` **Yes** keeps the written parameters connected to the
numbers quoted. The outer fit is a least-squares minimum, so its half is the Gauss-Newton contraction
at the stationarity point — **`LeastSquaresSolve`, the one node the swaption family also solves
through** — taken over the coordinates `least_squares` did not stop against a bound, and on the
COLUMN-SCALED Jacobian at `Jacobian_Rcond`, which is the same matrix and the same cutoff the
identification table reads. The ξ strip's half is the Newton splice the inner solve already carries,
so `dξ/dq` needs no rule of its own; the residual and its Jacobian are taken once at θ\* and kept, so
a whole `dθ/dq` matrix is one contraction per written parameter rather than one fit's worth of
evaluations each. `Stationarity_Tol` refuses the lot where θ\* is not a stationary point — a stage
that stopped at `Max_Iterations` has no quote derivative to report. Measured on the campaign's 2y SPX
autocall priced off a factor the same `Context` calibrated (`artifacts/logvar2fj/quote_risk.py`,
nine quotes, `Paths` 2048):

| gate | result |
|---|---|
| `dV/dq` one backward vs `dV/dθ · dθ/dq` | **4.4e-16 to 5.8e-14** relative over nine quotes — and a TAUTOLOGY: both route through the same backward, so what closes is the ATTACHMENT |
| `Quote_Sensitivity` Yes vs No, the written factor | **identical on all 14 fields** |
| the fit's own stationarity `‖Jᵀr‖` at θ\* | **1.55e-08** against `‖r‖` 7.65e-05, all seven coordinates free of their box |
| the re-authored central difference, the leg that tests the THEOREM | **does not close**: −39.8 at a half-width of 0.005 and +115.4 at 0.0025, against the backward's 4989 |

The market premium each row measures against carries its quote as a splice worth zero forward, which
is why the fit cannot move when the switch flips.

!!! warning "The third leg is the only one that tests the theorem, and on this document it fails"
    Re-authoring a quote and re-solving goes AROUND the node instead of through it, and it is a
    truth only where `q → θ*` is single valued. Here it is not: the two re-fits either side of a
    0.005 tick land with `Rho_L` **on its −0.6 box** one side and at −0.127 the other, which are two
    different KKT points rather than two points on one smooth manifold, and every coordinate's
    re-fit displacement is two to three orders under the contraction's. **`Jacobian_Rcond` is the
    dial that reaches this**, and it is measured: on a reduced USDZAR polish whose five scaled
    singular values are 1.940 / 0.871 / 0.567 / 0.384 / 0.0861, raising the cutoff from 1e-3 to 0.05
    drops the last direction and takes `Sigma_S` from −430.8 to −232.2 against a re-solved −196.3,
    and `Rho_S` from +11.5 to −13.4 against −15.7 — from sign-wrong to within a fifth. What a desk
    should read first is the identification table.

**Refused under `Fit_Mode` Bootstrap**: bucket `k` is fitted given the buckets before it, so θ\* is
a stationary point of no single objective and the contraction would report the last bucket's
derivative as the whole surface's.

**A traded forward-start is a different contract** from a reference model's slopes, and the block
says which it is being given: **Quotes** prices `E[S_T1(R − k)⁺]/E[S_T1]`, the same per-path gain
under the share measure and about 0.4 vol points of level away from **Reference**'s ratio
expectation `E[(S_T2/S_T1 − k)⁺]`. Either table-reading source with no rows refuses rather than
quietly fitting the vanillas alone.

**The target is a DIFFERENCE, and there is one term whatever the source.** `Δ_skew` is the forward
90–110 slope less the spot slope at maturity Δ and `Δ_bfly` the same for the 90/110 butterfly, both
in vol points, both targeted — residual skew is sticky while its convexity dilutes at the forward
date, leverage skew dilutes while vol-of-vol convexity amplifies, so the slope alone cannot separate
the residual's share from the vol-of-vol. **Prior** DECLARED the pair (`Stickiness_Prior`, `0.0,0.0` being
sticky-delta — withdrawn 2026-09-09 with the `Prior` source, both refusing by name; the
measurements below stand as its record); **Quotes** and **Reference** MEASURE it — the source's own forward
smile less the MARKET's own spot smile at the rung nearest Δ, read off the quoted vols in
log-moneyness, a quote itself wherever the rung carries one at 90/100/110 and the nearest quote
with a log line where it does not. The residual is then the model's difference less the target's,
applied to the model's OWN spot smile per evaluation, and the forward ATM LEVEL is targeted by
neither: the ATM ladder pins it and the ξ strip reprices it exactly.

A RATIO divides by a spot quantity that is within a few tenths of zero wherever the smile is
symmetric — on CJOW `ψ_bfly` read 0.075, 1.37 and −4.1 across three tenors — so it asks the fit to
match noise, and it did: a ratio target of `1.0,1.0` on the 22-rung USDZAR ladder pulled the wings
0.202 → **0.512** and degraded the vanillas at the target maturities by **+0.329** vol points, spec
5.3's own failure mode. The same ladder at `0.0,0.0` — the identical assumption, sticky-delta,
written as differences — moves that guarded RMSE by **−0.012**: it does not reproduce the failure.
What it does not buy is the rest of the surface. The wings still go 0.135 → **0.433** and the spot
RMSE 0.115 → 0.369, because spec 5.3's guard watches the TARGET maturities (0.5y and 1y here) and
the degradation lands on the shorter expiries it does not watch; the forward block is expensive on a
sub-year FX ladder either way, and what changed is that it is no longer expensive for a reason that
was arithmetic. The fit lands at `Δ_skew` −1.84 (6m into 6m) and −2.54 (1y into 1y) against a target
of exactly `+0.00` — sticky-delta is what it is aimed at, not what it reaches — but it reaches
CLOSER than the ratio target did, `ψ_skew` 0.906 / 0.819 against 0.859 / 0.713, on a fit that also
lands a better surface: 0.433 wings against 0.512.

The ratios are still REPORTED beside the differences, but only where the spot side exceeds **0.5
vol points**; where it does not the row prints `no ratio, spot +0.39 inside 0.5 vol points` and no
number, which is `ψ_bfly(1y into 1y)` on that very ladder and two of the three tenors on CJOW.

**Wherever the forward smile was not QUOTED the number is a reserve, and the factor carries its
model half.** `Stickiness_Band` (default **0.5 vol points**, the spread between the sticky-delta and
LSV-like views) is the band; a deal's reserve is `|∂PV/∂Δ_skew| × band`, and with `Prior` withdrawn
this is the ONLY place a desk's forward-smile view is priced. The calibration writes
`Skew_Gradient` — `∂(Δ_skew)/∂β` and `∂(Δ_skew)/∂ρ_s` in the LAST bucket at the nearest forward
tenor, in vol points per unit — and `Stickiness_Band` beside it on the `LogVar2FJModelParameters`
factor, both STRUCTURAL. A deal reporting `Greeks: First` on that factor composes
`bootstrappers.lv_skew_reserve` from them and its own last-bucket `(∂PV/∂β, ∂PV/∂ρ_s)`, and reports
the answer as **`Skew_Reserve`** beside `Value` on the `mtm` frame. Two parameters carry one target,
so the parameter move behind a vol point of `Δ_skew` is the MINIMUM-NORM one, `Jᵀ/(J Jᵀ)` — the
convention the quote contraction takes over its null space. **It is a NETTING-SET number**:
`Base_Revaluation.execute` calls `pricing.greeks` once, on the netting-set object and its total MTM,
so there is one gradient in the calculation and the reserve is composed from it; a per-deal reserve
wants a per-deal gradient this calculation does not produce. Measured on the desk's NKY autocall
229524957 under the fit the landed defaults write: `Skew_Reserve` **11.7m ZAR, 28% of the mark**,
composed identically by hand.

**With the block OFF the rows are REPORTED rather than targeted.** The calibrator can evaluate the
forward window without aiming at it, and does: each `Forward_Tenors` pair is moved onto the two grid
BLOCK ENDS nearest its own, the second strictly beyond the first. Moved, because a window end that
is not a block end is not on the walk's grid at all — putting one there splits a block into two
mixers and moves θ\*, and the reserve would then be quoted off a different fit from the one written.
So the grid, the draws and θ\* are the vanilla-only fit's to the bit, and the tenor the reserve is
read at is a fact about the QUOTES: on a chain quoting 0.51y and 2.74y the 6m-into-6m a desk asked
for is reported as 0.51y into 2.23y, and the log line says so.

**`Forward_Smile_Source` defaults to `None`, and `Prior` is WITHDRAWN.** *Vanillas do not close this
model* — brief 0.6 — but a target is not a source: measured on all four index ladders, a
sticky-delta TARGET on a one-bucket ladder is reached only by driving `α` to its box, flipping `β`
to its admissibility bound and carrying the skew on `ρ_s = −0.8`, at 0.4–1.1 vol points of spot fit,
and a 0.5y bucket does not repair it on a listed chain. **The block exists with a market or a
reference source or not at all.** `Quotes` and `Reference` are SOURCES, read off `Forward_Smiles`,
and enter at stage 3 WITH the leverage prior, because that is the stage that fits `(ρ_s, σ_s)`: the
split between `β` and `ρ_s σ_s` is exactly the direction the vanillas leave flat, and the polish
spectrum says so — its two smallest directions read 0.6699 / 0.2870 with the rows against 0.4135 /
0.0847 without on the USDZAR ladder, 0.4329 / 0.1199 against 0.1499 / 0.0317 on the SPX chain, the
two largest unmoved. A block declaring `Prior` refuses by name and is told which two sources there
are and where a view is carried instead. The tie-breaker form of `Prior` — a declared view
CONSTRAINED to 0.1 vol points of vanilla degradation on stage 5 — was built, measured and
withdrawn: a cap that covers one stage while the damage occurs in another is not a cap, and on a
one-bucket ladder, which is every block in the book, stage 5 does not run at all. What the report
keeps is the measurement the withdrawal rests on: where a source does run, the vanilla RMSE at the
forward rows' own maturities is reported as it moved over stage 5 AND the polish, cumulative, which
is the number a per-stage cap could not see. `Vanilla_Guard`, `Vanilla_Band` and `Stickiness_Prior`
refuse by name from the retired block; a source that names a TABLE still refuses in `Fit_Mode:
Bootstrap` by name.

**A tenor the ladder cannot reach is DROPPED by name.** `Δ_skew` is the forward slope less the SPOT
slope at maturity Δ, read at the quoted rung nearest Δ, so a tenor whose nearest rung is more than a
QUARTER of Δ away is a difference taken at the wrong maturity: on a three-week FX ladder the default
`1y:1y` would put a one-year forward smile beside a three-week spot one, and would walk the grid to
two years to do it. The rule drops it with the ladder's own expiries in the message; where nothing
survives the block is not fitted and the report says so.

**Stage 5's guard is read on the rung NEAREST each forward row's own `T1` and `T1 + Delta`**,
within the same quarter of `Delta` the reachability rule measures a spot rung by. The two rules
asked one question with two tolerances: a row survived reachability because its rung was 1.9% of
`Delta` away and then had no quote within one internal STEP of its maturity, so the guard was read
over an empty set and `torch.stack` raised. A ladder with no rung inside the tolerance now REFUSES
by name rather than judging a stage on no rows at all.

**The lever is the calendar bucket, and it was measured on the Poisson residual against CJOW.** With
ONE bucket a forward target has nothing to move but the vanillas and trips the failure mode by name
(+0.215 and +0.212 vol points on the guarded RMSE for the two sources); with a year-two bucket of
the skew pair to move, the fit reached `ψ_skew` 1.021 / 1.034 / 0.935 against the reference's 0.975
/ 1.008 / 0.552 with a composition residual of 0.104 at 2y and a stage-5 degradation of +0.184. The
reference and its harness are retired, so those are records; the mechanism they measured is
unchanged, the lever now being `β(t)` beside `ρ_s(t)` on the same buckets. A rung that does not
reach 90 or 110 is named in the log, since `Δ` at that tenor is then measured against an understated
spot slope. `Forward_Smile_Source: Reference` stays declared and its report line reads
*unexercised: no reference model wired* until one is.

## `FXVolPrices` — a smile quoted in delta, and where the conversion runs {#fxvolprices}

An FX smile ticks in as DELTA quotes: an ATM vol per expiry and, per delta pillar, the risk reversal
and butterfly around it. The surface the pricers read is a log-moneyness one. Both halves of the
conversion already existed — the strangle pair `vol(call) = ATM + BF + RR/2`,
`vol(put) = ATM + BF − RR/2`, and the delta-to-strike solve `Factor2D` runs for a `Malz` surface — so
this family is the same conversion **moved**, and the move is the point.

**The x-grid is pinned, because refinement is compile-time work.** The delta solve refines a
log-moneyness grid until interpolating total variance between the nodes resolves the smile to
`Grid_Tolerance`, so that grid is a function of the quotes — and at factor-construction time, where
it used to run, every vol tick was potentially STRUCTURAL: a moved node is a new plan and a recompile.
So the refinement runs in the bootstrap once and its grid is part of the written factor. A
re-bootstrap finding a surface already written **for the same expiries at the same `Grid_Tolerance`**
reuses that grid and moves only the vols on it, which is exactly a `bind='value'` patch; changed
expiries, or a different tolerance, refine from scratch. The log says what the pinned grid resolves
the CURRENT quotes to, beside the tolerance it was built at.

`Surface_Type` names the moneyness convention the engine reads a surface at (`calc_moneyness` returns
log(F/K), term interpolation in total variance) and it also said a delta smile is the form the block
was authored in. `Factor2D.solves_delta_surface` separates the two: a `Malz` block **carrying
deltas** is unsolved and gets solved on construction, as before; one carrying only the solved
`Surface` is what this family writes and falls through with its grid intact.

**The conversion is vectorized** — one bisection over the whole grid (`Factor2D.malz_sigma`) rather
than a Python loop with a scalar `brentq` per point, per expiry, per refinement pass. It refines the
identical grid, agrees with the loop to 5e-14 vol (inside `brentq`'s own 2e-12 `xtol`), and is made of
operations an autograd tape can carry, which the scipy call is not. That `brentq` oracle went with the
mock-built suite, so the 5e-14 agreement is a recorded measurement rather than a standing gate.

**The conventions are declared and each offers exactly one value**, because the solve implements
exactly one: `Delta_Type` `Forward`, `Premium_Adjusted` `Yes` (the pillar delta is `(K/F)N(d₂)`) and
`ATM_Convention` `Delta_Neutral_Straddle` (`K = F exp(−σ²T/2)`). A spot delta or an ATMF quote needs
different algebra, and a value the engine cannot honour is the same defect as a field nothing reads.
The gate that held these as maths rather than as strings went with the mock-built suite; today the
single-valued declarations hold the line.

**The ATM row is the surface's ATM vol, and a second family reads it as one.** `malz_skew` places the
±0.5 label's vol at the delta-neutral straddle strike, so this family's written ATM vol IS the quoted
number — nothing is read back off the refined grid to recover it. `FXVolSurfaceParameters.atm_quotes`
is the one reader of that rule, and `GBMAssetPriceTSModelParameters` takes those rows as its ATM
column wherever the surface it integrates is one this family **wrote** — evidenced by the same
fingerprint `pinned_grid` reads back, never by the name. See
[the quote sources](quote_sensitivities.md#the-atm-column), and the defect named beside them.

**The smile differentiates in its own quotes.** `Quote_Sensitivity` — declared here, default `No` —
leaves the written surface connected to the ATM / RR / BF numbers it was built from, so a backward
pass reports `dV/d(risk reversal)` beside `dV/d(surface node)`. The written surface is bit-identical
either way; what this conversion has that the other families do not is a **root find**, and the tape
does not enter it — see [increment 4](quote_sensitivities.md#the-delta-solve).

!!! warning "One ATM quote, two families, two partial derivatives"
    This block's ATM row is also `GBMAssetPriceTSModelPrices`' ATM column, so with **both** blocks
    asking for `Quote_Sensitivity` a single JSON number reaches a valuation through two independent
    maps and its `dV/dq` arrives split over two `quote_leaves` entries — under the *same* descriptor
    string, because a quote is named by what it is and not by which family read it. Each half is
    correct and neither is the total. A consumer must group by descriptor across blocks and **sum**;
    see [the collision](quote_sensitivities.md#the-attachment) for the measured numbers.

**A point may carry a two-way, and the bootstrap never reads it.** `Quoted_Bid` and `Quoted_Ask` are
optional columns beside the mid, written by `derivus_bloomberg` where the terminal answered
`PX_BID`/`PX_ASK`; a mid-only block is byte-identical to the one this family was always handed.
Everything below reads `Quoted_Market_Value` by name, so the surface, the pinned grid and every mark
on the book are built from the **mid**. The one reader is the quote layer:
[`derivus.structures`](structures.md#two-sided) shifts a leg's own copy of the written surface by the
ATM half-spread to quote a client two-sided. *The spread is the quote's; the mid is the book's.*

Both sides are on the value side of `update_market_quote`'s structure guard — a spread widens between
prints and a pillar that starts or stops being quoted two-sided is the same node of the same plan,
while a moved `Pillar` or `Expiry` refuses as it always did. That guard IS the section's plan/values
split, `schema.MARKET_QUOTE_VALUES`, read by the guard, by `plan_hash`, by `market_patch`/
`patch_market` and by the artifact slot alike, so a vol tick moves `values_hash` and leaves
`plan_hash` alone. See [Quote Propagation](quote_propagation.md#protocol) for what that buys and for
the five families whose quotes are not `Points` rows and are therefore wholly plan-side.

**Timestamps are data the engine stores and reports.** Each quote row carries when it was seen; the
written surface carries the latest as `Quote_Timestamp`, its own as-of, `bind='value'` because it
travels with the vols. It enters `values_hash` and therefore the replay identity — the same numbers
read off a different snapshot are a different market event — and **nothing in pricing reads it**.
What counts as too old is the consumer's policy. Resolution is FULL: `CustomJsonEncoder` writes a
midnight `Timestamp` as the plain date it always did (old files re-encode byte-stable) and a
non-midnight one in ISO form with its time, so the 09:15 and 16:30 snapshots of a quote survive a save
as themselves and `values_hash` separates them.
