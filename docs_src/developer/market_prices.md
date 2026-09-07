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
| `LogVar2FJModelPrices` | European options on any spot, plus forward-start smiles | `LogVar2FJModelParameters` — five scalars, **an L curve, a jump-intensity strip and four bucketed levers** |
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
structural constants, the two declared guards, the weights, the calendar-time `Param_Buckets`,
`Event_Days`, spec 5.3's forward block with its `Stickiness_Prior` DIFFERENCE pair, spec 5.2 stage
4's `Slow_Factor_Prior`, `Diffusive_Share` and `Quote_Sensitivity` — each documented where it is
declared. `derivus_bloomberg.equity_chain` emits it off a listed chain, and the inherited
`fx_surface_block` authors the FX one, asked here for **two delta pillars** at four wing expiries,
because a `Bootstrap` bucket frees one parameter per wing quote. What follows is what it MEASURED.

**`L` is piecewise CONSTANT on the segments between ATM expiries** — flat forward variance, the shape
var-swap strips are quoted in. A segment's integral depends on its own level alone, so the strip has
no recurrence to zig-zag along, every ATM still reprices exactly (one level per segment, one ATM
increment per segment, triangular and unique) and anything priced between pillars reads a quoted
forward variance; the OU recursion is a deviation from `L`, so the step at a pillar costs nothing.
`Event_Days` is then a one-day segment at `Event_Variance_Prior` times the level enclosing it —
unless the day already HAS a segment of its own, expiries on the business days either side, in which
case the ordinary pillar bootstrap sets it, the prior is ignored and the report says so.

**`λ(t)` follows the market's own strip, per segment.** A constant intensity makes the jump
variance `λ(μ_J² + σ_J²)` the same number on every segment while the market's forward variance
moves along the strip, so the diffusive share is forced to move *inversely* to the total — measured
at 0.71–0.90 on CJOW and 0.78–0.92 on USDZAR, the market's own shape backwards, a low-vol segment
carrying more crash risk than a high-vol one. So the rule is applied per segment:
`λ(t) = w_J · ξ_mkt(t)/(μ_J² + σ_J²)`, written onto the factor as its own piecewise-constant curve on
the L segments and read by `bucket_at` at absolute times exactly as `L` and the four levers are.
Still STRUCTURAL — never a fitted coordinate, its knots and values both compile-time facts, one knot
being the constant-intensity model — and re-derived from the strip daily; what carries across days is
`w_J`, which a previous factor's own first segment hands back. The jump share is then the SAME number
on every segment by construction, and the one free number left in the split is that level.

**The L strip is re-bootstrapped at EVERY outer iterate**, so every candidate reprices the ATM term
structure exactly and is judged on the smile alone; the ATM misses the report prints are 1e-16 to
1e-13 rather than the 1e-12 to 1e-9 a between-stage refit left. The level each segment returns is one
Newton step at its own root, so `dL/dθ` rides the tape and the outer solver keeps its exact Jacobian.
What that costs is one graph pass per segment per iterate: the search itself runs off the tape on the
previous sweep's slope, and on the walk at 8192 paths over 504 daily steps a forward pass is 0.280 s
against 0.756 s with its backward. The grid is the QUOTES' own, one trading day
between block ends with a stub landing each block on its `T`: reading the same rung on the
trading-day grid instead costs **0.124 vol points** at the 1m ATM (`jac_check.py`).

**On the CJOW harness surface** (spec 8 steps 0–4: 34 premium quotes over five maturities to 2y,
`Paths` 8192, daily δ, CPU). The two right-hand columns are `artifacts/logvar2fj/harness_cjow.py`
before the flat `L`; the left one is `artifacts/lv_split_20260906/split.py cjow | tables`, which
re-fits the same surface off that harness's BANKED premiums — its own step 0 prices them with the
retired Heston-Nandi half of `utils` and no longer runs:

| | vanilla only, `λ(t)` and flat `L` | on the LINEAR `L` at a constant `λ` | with the 5.3 forward target, linear `L` |
|---|---|---|---|
| RMSE, true inversion — unweighted / vega-weighted | **0.930 / 0.319** vol points | 1.020 / 0.331 | 0.995 / 0.337 |
| the 5 ATM rungs / the 29 WINGS, by the same statistic the FX table uses | 0.157 / **1.005** | 0.161 / 1.063 | 0.320 / 1.331 |
| per maturity, 1m … 2y | 2.17 / 0.80 / 0.36 / 0.36 / 0.07 | 2.30 / 0.79 / 0.25 / 0.52 / 0.08 | 2.98 / 0.80 / 0.27 / 0.50 / 0.37 |
| wall clock, evaluations + Jacobians | 1210 s (a contended box), 49 + 40 | **1142 s**, 79 + 61 | 1206 s, 81 + 53 |
| the inner bootstrap, per sweep over 5 pillars | 18.7 pillar passes, **5.4** of them a backward | 18.7, 5.3 | 19.2, 5.5 |
| ψ_skew(1y into 1y), CJOW's being 1.008 | **0.984** | 0.985 | 0.964 |
| ψ_bfly(1y into 1y), CJOW's being 0.374 | 1.365 | — | — |
| `Sigma_L` | fitted, **ON its 0.3 floor** with 2y wings present | **1.5e-06** — the slow factor collapsed | — |
| the campaign's 2y autocall at 2^15, CJOW's −32.068 (SE 0.226) | **−30.855** (SE 0.143) | −32.433 (SE 0.135) | −33.080 (SE 0.061) |

against spec 9's < 90 s on a GPU and spec 8's ≤ 0.2 vol points from one month out. The composition
residual is under spec 5.3's 0.3 failure mode; the forward target's gap, −0.647, is the deal's
forward-skew sensitivity, reported as such rather than as an error. What the per-iterate bootstrap
did NOT buy is the short end: 2.2 vol points at 1m, all of it the 70–80% wing (+6.02 / +3.91) and
the 110–120% convexity, which is what a two-shock structure costs against a one-shock reference and
remains the open number. `SE² × wall time` on the deal value is **2.83e-2** on the `λ(t)` re-fit,
against 1.28e-2 and 2.51e-3 on the linear-`L` fits and CJOW's own 5.29e-2 — the re-fit halves the
SE the FX gate's row 4 read (0.251 → 0.143) and moves the value the other side of the component
family's banked −31.89.

!!! warning "Read the two columns against the sampling floor, not against each other"
    Re-drawing the whole fit at three `Random_Seed`s moves the fitted ATM term structure by 0.10 to
    0.63 vol points and an L pillar by up to 2.6. The 0.930 against 1.020 against 0.995, and the
    ψ_skew 0.984 against 0.985 against 0.964, are all INSIDE that floor: what the forward target
    demonstrably moves is the composition residual and the deal, and what `λ(t)` demonstrably
    moves is the SPLIT, not the spot fit. ψ_bfly WAS the fragile one — the model's own spot
    butterfly at 6m is −0.02 vol points, so its ratio is a division by nothing — which is why the
    target is now the DIFFERENCE and the report prints a ratio only above 0.5 vol points of spot.

**Two modes on one banked surface** (`artifacts/logvar2fj/fx_ladder.py`, the campaign's USDZAR
`FXVol` at `Paths` 8192, daily δ, CPU, four processes contending), against BOTH Heston-Nandi
families fitted to the SAME 22 contracts. The two Heston-Nandi rows are the RECORD of a
measurement taken before the families were [retired](#hestonnandi-retired); nothing reproduces
them today. Six of those rungs are the ATM one of each expiry, which
any family carrying an L curve solves to zero, so folding them into one RMSE reports the split
rather than the fit: the table separates them and the headline is the WING RMSE over the other 16.

| fit | ATM rungs | 16 WING rungs, RMSE | worst wing | wall clock |
|---|---|---|---|---|
| LogVar2FJ `Global`, `λ(t)` and the slow pair PINNED at the FX CLASS DEFAULT (−0.2, 0.5) | 6 at **0.0e+00** vol points | **0.135** | −0.282 | 213 s |
| the same, PINNED at the index seed (−0.4, 1.0) | 6 at **0.0e+00** vol points | 0.202 | −0.415 | 819 s (a contended box) |
| the same at the index seed, `Lambda` pinned flat | 6 at 0.0e+00 | 0.189 | −0.417 | 203 s |
| LogVar2FJ `Global`, the slow pair fitted to nothing | 6 at **0.0e+00** vol points | **0.131** | −0.251 | 254 s |
| LogVar2FJ `Bootstrap` | 6 at 0.0e+00 | 0.575 | +1.291 | 222 s |
| plain Heston-Nandi | 6 at 1.45e-01 | 0.663 | −1.744 | 420 s |
| component Heston-Nandi | residual 4.4e-15 | prints no per-quote record; worst wing **0.760** | +0.760 | 137 s, CAPPED at 300 evaluations |

`Global` fits the wings **five times** better than the plain family and **six times** better than
the component one, which carries an L curve of its own and the same four wing expiries. Rows one and
two are what spec 5.2 stage 4's identification rule COSTS and what the PRIOR under it is worth: this
ladder's longest wing is 1y against the 18 months the rule asks for, so `(ρ_l, σ_l)` are pinned with
the report line *pinned: not identified by this ladder* instead of being fitted to nothing — row
four's 0.131 was bought by a slow factor the box ran to zero, which is the reading that made the
floor a rule. Pinned at the index SEED that rule charged 0.070 of wing RMSE; pinned at the FX CLASS
DEFAULT it charges **0.003** (0.135 against 0.132), so almost all of what the rule appeared to cost
was the seed being wrong-sized for FX. `λ(t)` itself costs 0.013 (row two against row three), inside
the seed spread. `Bootstrap` is worse than any `Global` row, and structurally so: bucket `k` acts
only on `[E_{k−1}, E_k)` while the option
to `E_k` averages over every bucket before it, so a later bucket has progressively less leverage on
the quote that frees it — on a sub-year ladder that is most of the ladder. It is the mode for a deal
read at many fixings, not the mode that fits a surface best, and the report says per bucket which
parameters were free and which tied.

!!! note "The zig-zag was the parametrisation; the level that is left is the decomposition"
    Both tables above were read on the **piecewise-linear** `L` this family carried until
    2026-09-06, whose strip alternated across the market's — `Global` 9.08 / 7.98 / 9.44 / 8.35 /
    10.79 / 9.21% against 9.85 / 10.08 / 10.58 / 11.04 / 12.04 / 12.75%, `Bootstrap` 9.04 / 7.60 /
    9.10 / 7.49 / 9.48 / 8.28%, the CJOW fit 14.56 / 5.51 / 19.95 / 10.76 / 13.37% against 15.97 /
    12.73 / 15.71 / 19.77 / 17.52%. That was the `−1` multiplier of a segment integral that reads
    both ends. Flat segments removed it: the same CJOW surface at `Paths` 2048 reads **14.43 / 9.34
    / 11.91 / 14.37 / 12.44%**, a diffusive share of the market's own strip of 0.90 / 0.73 / 0.76 /
    0.73 / 0.71 where the linear one read 0.91 / 0.43 / **1.27** / 0.54 / 0.76 — it no longer
    crosses one — and the strip's second difference, which IS the alternation, falls from 20.4 vol
    points RMS to 5.1 at a vanilla RMSE of 0.931 against 1.020 (at a quarter of the paths). A
    reduced USDZAR block (3 expiries, 15 quotes, `Paths` 2048) reads 9.11 / 8.75 / 8.82% where the
    linear `L` read 9.11 / 8.38 / 9.23%, 0.42 vol points of second difference against 1.57, at an
    unchanged wing RMSE (0.099 against 0.098). What was left was a LEVEL, and the report now
    prints it as the SPLIT it is (5.4.8) — per segment the market's own forward variance, the
    model's total as MEAN diffusive plus jump, the jump share and the Jensen share:

    | | market | model total | diffusive level | jump share | Jensen share |
    |---|---|---|---|---|---|
    | reduced USDZAR, 3 segments | 9.85 / 10.33 / 11.04% | 10.44 / 11.10 / 11.84% | 9.14 / 8.66 / 8.64% | **11.9%** flat | 14.3 / 33.2 / 41.8% |
    | CJOW, 5 segments | 15.97 / 12.73 / 15.71 / 19.77 / 17.52% | 17.60 / 14.05 / 17.98 / 21.78 / 18.18% | 14.50 / 9.80 / 11.95 / 14.25 / 11.72% | **15.2%** flat | 23.8 / 47.4 / 58.0 / 54.2 / 47.8% |

    So the two reasons the curve level fell away from the market are named apart. `λ(t)` removed
    the jump's — pinning `Lambda` flat on the same ladder puts the jump share back at 18.2 / 17.4 /
    15.8 / 14.5 / 12.2 / 10.9%, the market's shape backwards, for 0.189 wing RMSE against 0.202,
    inside the sampling floor. What is left is the vol-of-vol's own Jensen term growing as the OU
    variance accumulates, plus the gap between matching an ATM PRICE and matching a variance —
    the model's mean total must sit ABOVE the market's ATM² because `E[Black(sd)] < Black(E[sd])`.
    Neither is a miss. The component
    family's `L` is still piecewise-linear — `ω_t = L_{t+1} − ρL_t` differences it, so a step would
    spike `ω` — and still carries its own phase (9.93 / 9.81 / 11.01 / 10.54 / 12.12 / 12.70 /
    13.26% on the same ladder). The tables are not re-measured at their own path counts.

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
identification table reads. The L strip's half is the Newton splice the inner solve already carries,
so `dL/dq` needs no rule of its own; the residual and its Jacobian are taken once at θ\* and kept, so
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
in vol points, both targeted — jump skew is sticky while its convexity dilutes at the forward date,
leverage skew dilutes while vol-of-vol convexity amplifies, so the slope alone cannot separate the
jump share from the vol-of-vol. **Prior** DECLARES the pair (`Stickiness_Prior`, `0.0,0.0` being
sticky-delta and the default); **Quotes** and **Reference** MEASURE it — the source's own forward
smile less the MARKET's own spot smile at the rung nearest Δ, read off the quoted vols in
log-moneyness, a quote itself wherever the rung carries one at 90/100/110 and the nearest quote
with a log line where it does not. The residual is then the model's difference less the target's,
applied to the model's OWN spot smile per evaluation, and the forward ATM LEVEL is targeted by
neither: the ATM ladder pins it and the L strip reprices it exactly.

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

**What the difference target does NOT fix is the lever.** On the CJOW surface at `Paths` 2048 and
ONE bucket, both sources trip spec 5.3's failure mode by name — `Prior` at `0.0,0.0` degrades the
guarded vanilla RMSE **+0.215**, `Reference` against CJOW's own differences **+0.212** — and neither
lands its ψ near the source: 1.034 / 1.034 / 0.982 and 1.055 / 1.038 / 0.969 against CJOW's own
0.975 / 1.008 / **0.552**. With one bucket stages 5a and 5b have nothing to fit, so the forward
block's only lever is `w_J`, and a block with a target and no lever moves the vanillas instead. The
measurement that matters is therefore the bucketed one, and the target's own MECHANISM is visible in
the `Reference` rows: it aims at CJOW's measured `Δ_skew` of −1.01 / +0.19 / −9.98 rather than at
zero, which is what "the source's own difference" means. A rung that does not reach 90 or 110 is
named: CJOW's 3-month row is quoted 70–105% of its forward, so `Δ` at the 1y-into-3m tenor is
measured against an understated spot slope and the log says so.

**Give it the lever and it works.** The SAME `Reference` fit with `Param_Buckets` at 1y — so stages
5a and 5b have a year-two bucket of `μ_J(t)` and `ρ_s(t)` to move — reads `ψ_skew` **1.021 / 1.034 /
0.935** against CJOW's own 0.975 / 1.008 / **0.552** on the independent 2¹⁸-path walk, where one
bucket read 1.055 / 1.038 / 0.969: closer on all three. Its 29 wings go 2.067 → **1.763**, the whole
surface 1.916 → **1.632** and the 5 ATM rungs 0.406 → 0.312, so the second bucket pays for itself on
the spot fit as well. In the fit's own report the 1y-into-3m target `Δ_skew` is −9.98 and the model
reaches −8.10 where the unbucketed fit reached −6.46. Both of spec 5.3's own diagnostics print by
name: the **composition residual is 0.104 vol points** at the 2y rung, the first beyond the bucket
boundary, comfortably inside the 0.3 the spec calls a failure, while the stage-5 vanilla degradation
is **+0.184**, still outside the 0.1 it calls one. The buckets end at (`ρ_s`, `μ_J`) of (−0.718,
−0.216) in year one and (−0.716, −0.207) in year two, which is the smoothness penalty holding a term
structure the 2y vanillas have to tolerate.

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
