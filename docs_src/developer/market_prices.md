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
- `fields` — the block's schema, including the quote table or container as the class reads it.
- `quote_instruments`, where the quotes are instruments rather than a fixed option table.

`mapping['MarketPrices']['types']` is `schema.emit_market_prices(bootstrappers)` over those
declarations, and `construct_bootstrapper` resolves the class by name from the
`Bootstrapper Configuration` section.

| Family | Quotes | Writes |
| --- | --- | --- |
| `GBMAssetPriceTSModelPrices` | a vol surface, ATM column only — or, where `FXVolPrices` built that surface, [its ATM rows](#fxvolprices) | `GBMAssetPriceTSModelParameters` — an integrated vol curve |
| `CSForwardPriceModelPrices` | European energy futures options | `CSForwardPriceModelParameters` — sigma, alpha |
| `LogVar2FJModelPrices` | European options on any spot, plus forward-start smiles | `LogVar2FJModelParameters` — four scalars, **a forward-variance curve `ξ` and four bucketed levers** |
| `HullWhite2FactorModelPrices` | forward-starting swaps against a swaption surface | `HullWhite2FactorModelParameters` — two sigma curves, two alphas, a correlation |
| `InterestRatePrices` | deposits, FRAs, swaps and FX forward outrights | an `InterestRate` zero curve |
| `FXVolPrices` | ATM vols, risk reversals and butterflies | an `FXVol` log-moneyness surface |

## `InterestRatePrices` — a curve solved from its quotes {#interestrateprices}

**The block.** `InterestRateCurveParameters` declares the `Currency` of the curve to build, the
`Day_Count` its tenors are expressed in, an optional `Discount_Rate` naming the curve the quotes
discount on, the solver's three knobs (`N_Iter`, `Tol`, `Damping_Halvings`, each read with its
declared default as the engine's fallback), the three lifecycle switches (`Quote_Sensitivity`,
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

**The ξ strip is re-bootstrapped at EVERY outer iterate**, so every candidate reprices the ATM term
structure exactly and is judged on the smile alone; the ATM misses the report prints are 1e-16 to
1e-13 rather than the 1e-12 to 1e-9 a between-stage refit left. The level each segment returns is one
Newton step at its own root, so `dξ/dθ` rides the tape and the outer solver keeps its exact Jacobian.
What that costs is one graph pass per segment per iterate: the search itself runs off the tape on the
previous sweep's slope, and on the walk at 8192 paths over 504 daily steps a forward pass is 0.280 s
against 0.756 s with its backward. The grid is the QUOTES' own, one trading day
between block ends with a stub landing each block on its `T`: reading the same rung on the
trading-day grid instead costs **0.124 vol points** at the 1m ATM (`jac_check.py`).

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
wing. **Nothing in the brief bounds `α` from below**; the seed and pin from history are what close
it, which is the calibrator lane's. `Bootstrap` is structurally the mode for a deal read at many
fixings rather than the mode that fits a surface best — bucket `k` acts only on `[E_{k−1}, E_k)`
while the option to `E_k` averages over every bucket before it — and the report says per bucket
which parameters were free and which tied.

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
shape — the one place both lanes declare it — reported with both standard errors and taken to the
floor by name where it sits under it, as a hand-authored `(−0.35, 0.22)` with SEs 0.09 / 0.14 reads
back *held at the history's estimate … its Sigma_L 0.2200 TAKEN TO the 0.3 floor*; the block is
written by `stochasticprocess.LogVar2FJCalibration`, spec 5.5's estimator), else
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
the residual's share from the vol-of-vol. **Prior** DECLARES the pair (`Stickiness_Prior`, `0.0,0.0` being
sticky-delta and the default); **Quotes** and **Reference** MEASURE it — the source's own forward
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

**The lever is the calendar bucket, and it was measured on the Poisson residual against CJOW.** With
ONE bucket a forward target has nothing to move but the vanillas and trips the failure mode by name
(+0.215 and +0.212 vol points on the guarded RMSE for the two sources); with a year-two bucket of
the skew pair to move, the fit reached `ψ_skew` 1.021 / 1.034 / 0.935 against the reference's 0.975
/ 1.008 / 0.552 with a composition residual of 0.104 at 2y and a stage-5 degradation of +0.184. The
reference and its harness are retired, so those are records; the mechanism they measured is
unchanged, the lever now being `β(t)` beside `ρ_s(t)` on the same buckets. A rung that does not
reach 90 or 110 is named in the log, since `Δ` at that tenor is then measured against an understated
spot slope. `Forward_Smile_Source: Reference` stays as the production path and its report line reads
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
