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
| `LogVar2FJModelPrices` | European options on any spot, plus forward-start smiles | `LogVar2FJModelParameters` — five scalars, an L curve **and four bucketed levers** |
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
`Event_Days`, spec 5.3's forward block and `Quote_Sensitivity` — each documented where it is
declared. `derivus_bloomberg.equity_chain` emits it off a listed chain, and the inherited
`fx_surface_block` authors the FX one, asked here for **two delta pillars** at four wing expiries,
because a `Bootstrap` bucket frees one parameter per wing quote. What follows is what it MEASURED.

**`L` is piecewise CONSTANT on the segments between ATM expiries** — flat forward variance, the shape
var-swap strips are quoted in. A segment's integral depends on its own level alone, so the strip has
no recurrence to zig-zag along, every ATM still reprices exactly (one level per segment, one ATM
increment per segment, triangular and unique) and anything priced between pillars reads a quoted
forward variance; the OU recursion is a deviation from `L`, so the step at a pillar costs nothing.
`Event_Days` is then a one-day segment at `Event_Variance_Prior` times the level enclosing it.

**The L strip is re-bootstrapped at EVERY outer iterate**, so every candidate reprices the ATM term
structure exactly and is judged on the smile alone; the ATM misses the report prints are 1e-16 to
1e-13 rather than the 1e-12 to 1e-9 a between-stage refit left. The level each segment returns is one
Newton step at its own root, so `dL/dθ` rides the tape and the outer solver keeps its exact Jacobian.
What that costs is one graph pass per segment per iterate: the search itself runs off the tape on the
previous sweep's slope, and on the walk at 8192 paths over 504 daily steps a forward pass is 0.280 s
against 0.756 s with its backward. The grid is the QUOTES' own, one trading day
between block ends with a stub landing each block on its `T`: reading the same rung on the
trading-day grid instead costs **0.124 vol points** at the 1m ATM (`jac_check.py`).

**On the CJOW harness surface** (`artifacts/logvar2fj/harness_cjow.py`, spec 8 steps 0–4: 34
premium quotes over five maturities to 2y, `Paths` 8192, daily δ, CPU, three processes contending):

| | vanilla only | with the 5.3 forward target |
|---|---|---|
| RMSE, true inversion — unweighted / vega-weighted | **1.020 / 0.331** vol points | 0.995 / 0.337 |
| the 5 ATM rungs / the 29 WINGS, by the same statistic the FX table uses | 0.161 / **1.063** | 0.320 / 1.331 |
| per maturity, 1m … 2y | 2.30 / 0.79 / 0.25 / 0.52 / 0.08 | 2.98 / 0.80 / 0.27 / 0.50 / 0.37 |
| wall clock, evaluations + Jacobians | **1142 s**, 79 + 61 | 1206 s, 81 + 53 |
| the inner bootstrap, per sweep over 5 pillars | 18.7 pillar passes, **5.3** of them a backward | 19.2, 5.5 |
| ψ(1y into 1y), CJOW's being 1.008 | 0.985 | 0.964 |
| composition residual beyond the last bucket | — | **0.119** vol points at 2y |
| the campaign's 2y autocall, CJOW's −32.068 (SE 0.226) | −32.433 (SE 0.135) | −33.080 (SE 0.061) |

against spec 9's < 90 s on a GPU and spec 8's ≤ 0.2 vol points from one month out. The composition
residual is under spec 5.3's 0.3 failure mode and the vanilla-only autocall is **1.1%** of CJOW; the
forward target moves it out again to 3.2%, and that gap, −0.647, is the deal's forward-skew
sensitivity, reported as such rather than as an error. What the per-iterate bootstrap did NOT buy is
the short end: 2.3 vol points at 1m, all of it the 70–80% wing (+1.61 RMS, worst +4.68) and the
110–120% convexity (+0.72 RMS), which is what a two-shock structure costs against a one-shock
reference and remains the open number. `SE² × wall time` on the deal value is 1.28e-2 and 2.51e-3
against CJOW's 5.29e-2.

!!! warning "Read the two columns against the sampling floor, not against each other"
    Re-drawing the whole fit at three `Random_Seed`s moves the fitted ATM term structure by 0.10 to
    0.63 vol points and an L pillar by up to 2.6. The 1.020 against 0.995 and the ψ 0.985 against
    0.964 are both INSIDE that floor: what the forward target demonstrably moves is the composition
    residual and the deal, not the spot fit.

**Two modes on one banked surface** (`artifacts/logvar2fj/fx_ladder.py`, the campaign's USDZAR
`FXVol` at `Paths` 8192, daily δ, CPU, four processes contending), against BOTH Heston-Nandi
families fitted to the SAME 22 contracts. Those two rows are the RECORD of a measurement taken
before the families were [retired](#hestonnandi-retired); nothing reproduces them today. Six of those rungs are the ATM one of each expiry, which
any family carrying an L curve solves to zero, so folding them into one RMSE reports the split
rather than the fit: the table separates them and the headline is the WING RMSE over the other 16.

| fit | ATM rungs | 16 WING rungs, RMSE | worst wing | wall clock |
|---|---|---|---|---|
| LogVar2FJ `Global` | 6 at **0.0e+00** vol points | **0.131** | −0.251 | 254 s |
| LogVar2FJ `Bootstrap` | 6 at 0.0e+00 | 0.575 | +1.291 | 222 s |
| plain Heston-Nandi | 6 at 1.45e-01 | 0.663 | −1.744 | 420 s |
| component Heston-Nandi | residual 4.4e-15 | prints no per-quote record; worst wing **0.760** | +0.760 | 137 s, CAPPED at 300 evaluations |

`Global` fits the wings **five times** better than the plain family and **six times** better than
the component one, which carries an L curve of its own and the same four wing expiries. `Bootstrap`
is worse than either, and structurally so: bucket `k` acts only on `[E_{k−1}, E_k)` while the option
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
    unchanged wing RMSE (0.099 against 0.098). What remains is a LEVEL: the diffusive strip sits
    under the market's total forward variance because the jump and the two leverage shocks carry
    the rest, so the ATM is right and the split is a choice the data does not pin. The component
    family's `L` is still piecewise-linear — `ω_t = L_{t+1} − ρL_t` differences it, so a step would
    spike `ω` — and still carries its own phase (9.93 / 9.81 / 11.01 / 10.54 / 12.12 / 12.70 /
    13.26% on the same ladder). The tables are not re-measured at their own path counts.

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
expectation `E[(S_T2/S_T1 − k)⁺]`. **Prior** needs no table and tilts the market's own spot smile at
each Δ by `Stickiness_Prior`; at ψ = 1 the target smile reads the market's own slope back, which
`checks.py prior` gates. Either table-reading source with no rows refuses rather than quietly
fitting the vanillas alone.

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
