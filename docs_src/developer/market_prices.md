# Market Prices

`Market Prices` is the risk-neutral half of the market data. Where `Price Factors` holds the
observed curves and surfaces a historical calibration produces, this section holds the **quotes**
a risk-neutral model is fitted to, and a *bootstrapper* turns each block into the factor or model
parameters the simulation reads.

This page is the design. All seven families are built; `HestonNandiComponentModelPrices` was the
last, and the two sections below are what each was built to.

## A quote is an instrument, a quote type and a number {#a-quote}

Every quote in this section is the same three things:

| | |
| --- | --- |
| **the instrument** | a reference to an EXISTING instrument type — the thing the number is a price *for* |
| **`Quote_Type`** | what kind of number is quoted, and therefore what "reprices to the quote" means |
| **`Quoted_Market_Value`** | the number |

The first is the load-bearing one. A quote does not restate an instrument's fields; it **names an
instrument type and carries a block of that type**, so the `Instrument` store's declarations *are*
the quote's schema. A family names the types its quotes may be — `quote_instruments` on the
bootstrapper class — and that list is what a `DealType` dropdown offers. The gate that held it to
the declared deal types went with the mock-built suite; the structure registry's twin
(`test_the_registry_publishes_exactly_the_declared_structures`) is the surviving copy of the rule.
Nothing about a swap has to be described twice for a swap to be quotable.

`Quote_Type` is per family, not global. The Clewlow–Strickland family takes implied vols only;
Heston-Nandi takes a vol or a premium; an interest-rate quote is a par rate.
Those are three different questions that happen to share a name, which is right — the JSON is per
family, and only a store keyed by field name across all of them was ambiguous.

## A family maps a quote set to what it calibrates {#a-family}

A bootstrapper class is one price family. It declares:

- `market_factor_type` — the `Market Prices` type string a block is filed under, and the string
  the class selects its own work by. It is declared rather than recovered from the class name
  because the block is named `HestonNandiModelPrices` while its class is
  `HestonNandiModelParameters`.
- `fields` — the block's schema, including the quote table or container as the class reads it.
- `quote_instruments`, where the quotes are instruments rather than a fixed option table.

`mapping['MarketPrices']['types']` is `schema.emit_market_prices(bootstrappers)` over those
declarations, and `construct_bootstrapper` resolves the class by name from the
`Bootstrapper Configuration` section.

The seven families, and what each fits:

| Family | Quotes | Writes |
| --- | --- | --- |
| `GBMAssetPriceTSModelPrices` | a vol surface, ATM column only — or, where `FXVolPrices` built that surface, [its ATM rows](#fxvolprices) | `GBMAssetPriceTSModelParameters` — an integrated vol curve |
| `CSForwardPriceModelPrices` | European energy futures options | `CSForwardPriceModelParameters` — sigma, alpha |
| `HestonNandiModelPrices` | European options on any spot | `HestonNandiModelParameters` — omega, alpha, beta, gamma\*, H0 |
| `HestonNandiComponentModelPrices` | the same ladder, wings widened | `HestonNandiComponentModelParameters` — alpha, beta, gamma₁, rho, phi, gamma₂, H0 **and an L curve** |
| `HullWhite2FactorModelPrices` | forward-starting swaps against a swaption surface | `HullWhite2FactorModelParameters` — two sigma curves, two alphas, a correlation |
| `InterestRatePrices` | deposits, FRAs, swaps and FX forward outrights | an `InterestRate` zero curve |
| `FXVolPrices` | ATM vols, risk reversals and butterflies | an `FXVol` log-moneyness surface |

## `InterestRatePrices` — a curve solved from its quotes {#interestrateprices}

The store carried the shape of this one for as long as it existed: a `Points` container holding
`Deal` / `DealType` / `Quote_Type` / `Quoted_Market_Value`, and a `DealType` naming instrument
types. That was the design showing through, and what got built is that design with nothing bolted
on.

**The block.** `InterestRateCurveParameters` declares it: the `Currency` of the curve to build, the
`Day_Count` its tenors are expressed in, an optional `Discount_Rate` naming the curve the quotes
discount on, the solver's three tuning knobs (`N_Iter`, `Tol`, `Damping_Halvings` — the same
vocabulary the calibration classes tune with, and each read with its declared default as the
engine's fallback), the three lifecycle switches (`Quote_Sensitivity`, `Quote_Propagation` and the
`Drift_Tolerance` a ride is refused outside — see [Quote Propagation](quote_propagation.md)), and
the quote `Points`. A blank `Discount_Rate` builds a **self-discounting** curve, which is the
single-curve configuration.

**The quotes.** Each point carries a `Deal` — a deposit, an FRA, a swap, an `FXForwardDeal`, or a
`StructuredDeal` over two legs, authored exactly as it would be in `Trade Data`, because it is the
same declaration — plus `DealType`, `Quote_Type`, `Quoted_Market_Value` and a free-text
`Descriptor`, and a `Use` flag so a quote can be held out without being deleted. Three more
columns are declared and the solve reads none of them: `Quoted_Bid` and `Quoted_Ask`, which a
Bloomberg-authored point carries where the feed printed a two-way, and a `Timestamp`, which it
carries unconditionally. They are this family's share of the value plane every `Points` family has
— `schema.MARKET_QUOTE_VALUES`, the columns a tick may move without touching the plan — and the
curve is solved from `Quoted_Market_Value` alone, the one of the four a patch cannot clear.
`DealType` supplies the block's `Object`, and the family stamps `Discount_Rate`: what an
instrument *projects* off is its own business and it names that curve itself, what the quote set
*discounts* on belongs to the curve set and is stated once.

`Quoted_Market_Value` is read in the unit its own `DealType` reads, and where it lands is a
property of the instrument TYPE — a `FRA_Rate`, a `Swap_Rate`, a pinned `Interest_Rate_Schedule`, a
fixed leg's `Rate` column. That correspondence is the one thing the family knows about a type
beyond the type's own declarations, and it is a registry (`QUOTE_WRITERS`) rather than a branch, so
a new quotable instrument is a row. The family scales nothing centrally: a rate benchmark is quoted
in **percent** because that deal's own field semantics divide by 100 — a `Percent`, a `Basis`, a
schedule the deal scales — so a quote that is not a rate rides the same path untouched.
`Quote_Type` declares the single convention that is built, `Par_Rate`: the solve holds every
benchmark at PV zero.

**A forward outright as a benchmark.** `FXForwardDeal` is the one quotable type whose quote is not
a rate. Its `Quoted_Market_Value` is the forward **outright** — units of `Buy_Currency` per one
unit of `Sell_Currency` — landing as `Buy_Amount = quote × Sell_Amount` on a benchmark that fixes
the sold amount and both discount-rate names, which keeps the PV exactly affine in the quote the
way the par solve and the drift metric both need. A currency's discount curve can then be solved
directly from the points a desk trades, and covered interest parity is written nowhere: the
residual is the forward's own pricer held at PV zero, so the parity is whatever that pricer means.
The residual **crosses currencies**, which the benchmark closure allows by the same rule it applies
to every factor a solve is not solving for — the other leg's curve and both `FxRate` spots are
discovered from the deal's own declarations and enter as detached constants, while the deal
converts to the reporting currency itself, that currency being the solved curve's. Because the
coupling is authored inside the deal rather than in `Discount_Rate`, the **solve order** reads it
there too: a block comes after every block building a curve its benchmark deals name, so a forward
strip authored before the curve its other leg discounts on still solves.

`Quote_Sensitivity` on such a block **refuses**, by name. The overlay that carries a quote
derivative rides cashflow-schedule value columns; an outright lands in `Buy_Amount`, which the
pricer reads as a float off the deal, so no schedule of the compiled forward moves and the refusal
names the benchmark and the type rather than reporting a zero `dV/dq` on the instrument the desk
actually trades. It is measured and not a branch on the type, so it stays silent wherever a quote
does reach a column. `Quote_Propagation` cannot ride one either — a coupled set spanning two
reporting currencies was already refused, and a single-currency block of forwards meets the same
overlay refusal. Full quote-derivative support for an amount-valued quote is not built.

**The knot rule.** ONE knot per used quote, at that benchmark's last cashflow date, in the block's
`Day_Count`. That is the only placement that makes the system square, and squareness is the shape
of a bootstrap: a knot with no instrument maturing at it is unidentified, and two instruments
maturing between the same pair of knots leave the curve under-determined between them. Below the
shortest knot the curve is flat, which `CurveTenor` gives by clipping, so the front stub costs no
extra unknown. **The output grid is that grid** — writing the result onto a second, wider grid
would interpolate it, and an interpolated curve no longer reprices its quotes. There is therefore
no `Zero_Rate_Grid` field; holding a quote out with `Use` drops its knot with it.

**The solve.** Find the curve that reprices every used quote to par. The instruments are priced by
the engine's own pricers — that is the point of quoting them as instruments — so the residual is a
vector of pricing functions of the curve, and the solve is a damped Newton root find on it in
float64. Two blocks make a multi-curve set, and `Discount_Rate` orders them: a projection curve
solved before the discount curve it prices against would be solved against a curve that does not
exist.

**What makes the solve cheap.** The residual's Jacobian in the curve knots comes from AAD through
the pricers — one backward pass per quote gives a whole row, and no bump loop is needed. That is
the same derivative the [quote-sensitivity](quote_sensitivities.md) thread carries the other way,
through the bootstrap's fixed point into `dV/dq`; the residual is written once and differentiated
twice, which is why the two land together.

**Two declared fields are gone**, on the terms every store here is held to: a field the engine does
not read is not declared. `Zero_Rate_Grid` named an output grid there is no longer one of, and
`Spot_Offset` named a spot lag that a quote's own `Deal` block states as an `Effective_Date` —
which is the whole point of a quote carrying an instrument rather than describing one.
`Quote_Type` loses `Rate` and `Price` for the same reason: a futures price and a money-market rate
on a different basis are conventions the family would have to author differently, and a value the
solve does not implement is the same defect as a field nothing reads.

**The loose ends, settled.** The curve this writes is an `InterestRate`, not a `<ClassName>`
parameter block, so the class declares `price_factor_type` — as `FXVolPrices` does for its
`FXVol` — and `Config.bootstrap`'s "wrote no `<name>.*` price factor" check reads it. And the
interpolation of a solved curve comes from `Price Factor Interpolation` rather than from the
block — see the `Interpolation` note in [Conventions](conventions.md#registries-not-functions).

## `HestonNandiModelPrices` — a block authored off a built surface {#hestonnandi-fx}

The family fits five parameters to a set of European options, and for an equity that set is
authored: somebody types the strikes. An FX pair's is not typed, because the desk already has the
answer — the `FXVol` surface [`FXVolPrices`](#fxvolprices) just built. So
`HestonNandiModelParameters.fx_surface_block` READS a pair's built surface and authors the block:
**ten vega-weighted implied vols — ATM at 1M, 2M, 3M, 6M, 9M and 1Y, plus the 25 delta wings at 3M
and 6M**. The term structure is what identifies `H0`, `Beta` and `Omega`; the skew identifies
`Gamma_Star` and the wings' width `Alpha`. `Weight` is the Black vega off the same surface,
normalised, so the front month is not abandoned to the back. Nothing past 1Y: TARFs and
accumulators are sub-year products, and a parameter fitted to the 2Y smile is borrowed against
products nobody quotes.

**The strikes are the surface's own coordinates.** ATM is the delta-neutral straddle
`K = F exp(-σ²T/2)` the surface was BUILT under, and each wing is the strike whose
premium-adjusted forward delta IS the pillar — found by inverting the same delta `Factor2D`'s Malz
solve inverted, off the same vols. An expiry the surface does not carry moves to the **nearest
quoted one at or under a year**, and the block says so in `Quote_Source`; interpolating between two
pillars would put a number nobody quoted into the objective under the name of one somebody did.
The surface's own `Quote_Timestamp` travels onto the block, so staleness stays data here as it does
there.

**Ten rungs are not ten quotes, and the count is what refuses.** Snapping is an argmin, so every
rung the surface does not carry lands on a contract another rung already named — a repeated
contract is a WEIGHT, not an observation. The canned two-pillar USDZAR surface collapses the whole
ladder onto **four** distinct `(expiry, strike)` contracts and a single-expiry surface onto three,
and four observations do not identify five parameters: what identifies `H0`, `Beta` and `Omega` is
the ATM TERM STRUCTURE, which is exactly what a collapse destroys. So the DISTINCT contracts are
counted after snapping and a ladder below **six** refuses by name, with the surface's own pillars
in the message. Four pillars (1M/2M/3M/6M) are the fewest that clear it — measured at eight
contracts, since the two wing pairs then sit on two different expiries.

**And the cap is applied, not hoped for.** An argmin has no ceiling: on a surface quoting 2Y every
rung of the ladder would answer 2Y and a fit of sub-year products would borrow its parameters from
a smile nobody quotes. The candidates are therefore filtered to the ladder's own longest rung plus
a week (the width of the same pillar quoted from a different date), and a rung with nothing
admissible under it is DROPPED — which `Quote_Source` says, and which can drop the ladder below the
six-contract floor and refuse there.

**The vol is read at the PILLAR; the strike hangs off the DATE.** A quote block carries dates, so
the pillar is emitted as the nearest whole day and the fit reads its own accrual back off that date
through the discount curve's day count. That accrual is what the FORWARD is built at — the fit
recomputes exactly it — but it is NOT where the surface is read: under ACT_360 a 1Y pillar resolves
to 1.0139, and reading there walks every rung off its pillar and puts the 1Y rung past the last
expiry the surface carries (measured +0.0036 vol points at the 3M rung on the canned surface with
the USD curve on ACT_360). The day-count conversion belongs to the forward and to the step count.

**`Volatility_Delta` is not folded in.** The block is a QUOTE, and the scenario shift is the FIT's
business: `bootstrap` adds `vol_surface.delta` to every quoted vol it prices a target premium off.
Adding it here as well calibrated a two-vol-point world for a one-vol-point scenario, and said
nothing. Gated as an identity between two real calibrations — the unshifted quotes under a 0.01
scenario land on the same five parameters as the hand-bumped quotes under none, to **7.7e-9
relative**.

**Orientation is the correctness argument, not a detail.** An `FXVol.A.B` surface's x-axis is
`log(F/K)` on the pair *A priced in B*; the 0D factor the family fits is an `FxRate`, which is
priced in the DOMESTIC currency. So the underlying is the token that is not domestic, the strikes
are in that factor's units, and the block declares `Use_Forward` **Yes** with `Invert_Moneyness`
set exactly as an `FXOptionDeal` on that surface sets it. The written
`HestonNandiModelParameters.<underlying>` is then the factor an FX accrual leg resolves by naming
convention off the pair's NON-BASE token (`utils.spot_model_currency`) — the same token this fit
names, describing the same rate the pricer simulates. Inverting the rate flips the sign of the skew
`Gamma_Star` carries, which is why that is worth a paragraph — and it is the same flip
`utils.hn_reciprocal_gamma` performs for a deal written on the reciprocal of this axis, where
`gamma*` becomes `1 − gamma*` under that deal's own numeraire.

**Both signs of the skew live in one box.** `Gamma_Star`'s sign IS the direction of the smile:
positive is the equity leverage shape, vol falling with strike in the underlying's own units, and
USDZAR — whose risk reversal is negative in pair terms — is the other one once read as `FxRate.ZAR`.
The fitted vector used to bound `Gamma_Star` in `[1, 5000]`, strictly positive, so that surface had
no admissible fit and the optimizer answered it by switching the leverage channel off and reporting
a **flat smile** as a converged calibration. It cannot simply be widened across zero:
`Alpha = |l|ψ/Gamma_Star²` is singular there, and the singularity is real — holding the skew
channel fixed while `Gamma_Star → 0` sends the wings' width to infinity. So the **leverage share
carries the sign**: `x[3]` is the magnitude, bounded away from zero, and `x[2]` is a signed share
in `[-1, 1]` whose sign is `Gamma_Star`'s, with `Alpha = |l|ψ/Gamma_Star²` and `Beta = ψ(1-|l|)`.
Stationarity is still a box bound on a fitted parameter and every iterate is still feasible. At
`l = 0` there is no leverage channel and `Gamma_Star` is unidentified, which is what a flat surface
legitimately reports — and the reason refusing a fit that lands ON the bound was the wrong fix.
A cold start seeds the SIGN off the quotes themselves (a smile rising with strike is a negative
`Gamma_Star`), because the objective has a kink at zero leverage and a local optimizer started on
the wrong side of it would have to cross that kink.

**`POST /book/hn` is the verb**, and it is calibrate-on-request, on the XVA mosaic's terms: the fit
is a least squares over a Fourier-inverted daily GARCH recursion, and it is MINUTES — **288 s**
measured on the four-pillar ladder reaching six months (549 s for the same fit with the suite
running beside it, to the same five parameters bit for bit), against 880 s on an earlier 1M/3M one
under the one-signed parametrisation, and still running past 21 minutes on one reaching a year. The
ITERATION count dominates the step count, so neither reading predicts the other; what both share is
that **75–82% of every option price is the adaptive `phi_max` scan** (0.87 s against 3.42 s per
`hn_call` at 252 steps, with the bound pinned). So it rides no tick,
it is queued at the heavy cost class, and a market tick leaves the parameters exactly where they
were. That last part is STRUCTURAL rather than a convention: `Bootstrapper Configuration` names the
families that run on EVERY bootstrap and a tick is a bootstrap, so the verb BORROWS the family
entry for its own run and hands it back. What the book keeps is the block and the fitted factor;
re-fitting is this verb being called again. It drops its own block before re-installing it, because these
strikes are a FUNCTION of the surface: re-emitting after a tick legitimately moves them, which
`update_market_quote` would refuse by name and be right to. There is no GET side — the written
factor in `Price Factors` is the projection, and `GET /book` already serves it.

## `HestonNandiComponentModelPrices` — a term structure fitted as a curve {#hestonnandi-component}

The plain family fits ONE `Omega`, so the model has one long-run variance and the whole ATM term
structure has to be bought with `H0`, `Beta` and that constant. The component model of Christoffersen,
Jacobs, Ornthanalai and Wang splits the variance into a long-run component `q` and a short-run
deviation, and this family fits the long-run component's own PATH:

$$h_{t+1}=q_{t+1}+\beta(h_t-q_t)+\alpha\Big[(z_t-\gamma_1\sqrt{h_t})^2-(1+\gamma_1^2h_t)\Big]$$
$$q_{t+1}=\omega_t+\rho q_t+\phi\Big[(z_t-\gamma_2\sqrt{h_t})^2-(1+\gamma_2^2h_t)\Big]$$

Both bracketed terms are **exactly centered**, so `h − q` is a pure AR(1) at `β` and `E_t[q_{t+k}]`
is driven by `ω` alone. Two things follow, and everything below is one of them.

**`Ω` is not a number here, it is a curve.** Writing `ω_t = L_{t+1} − ρL_t` makes `q_t − L_t` a
homogeneous AR(1), so ANCHORING `q_0 = L(0)` gives `E_0[q_t] = L_t` **exactly**. `L` is therefore
not a reparametrisation trick — it IS the model's expected long-run variance path, and it is
directly comparable to the market's own forward variance strip. It is piecewise-LINEAR in `t`
between pillar knots and flat outside them, so `ω_t` is AFFINE within a pillar - it drifts by
`(B−A)(1−ρ)/n` per step and kinks only AT a pillar, which is also what gives the
declining-variance floor below a closed form.

The stored curve carries a knot at tenor 0 whose value is `H0`: at the base date the two states are
held equal, because no option is quoted at zero maturity to separate them, and that knot is what
makes `q_0 = L(0)` a property of the written factor rather than a convention a reader has to
remember. There is no `Omega` field and no `Q0` field.

**It NESTS the plain family exactly**, which is the whole correctness argument and the spine of its
gates. Set `φ = 0` and hold `L` flat and the h-recursion collapses onto the plain one under

    ω_p = L(1−β) − α,   β_p = β − αγ₁²,   α_p = α,   γ_p = γ₁

whose inverse is `β = ψ_p` (the plain PERSISTENCE, not the plain `Beta`) and `L` = the plain
STATIONARY variance. `utils.hn_component_from_plain` / `hn_component_to_plain` are that map, and
`tests/test_hn_component.py` holds the two closed forms to each other across a strike/expiry grid
at **1.5e-13 relative** — machine precision, not a quadrature tolerance, because the map is exact.
The A/B/C recursion's third coefficient is what makes it close: `A` alone does NOT reduce to the
plain `A`, and it is the anchoring `q_0 = L(0)` that reconciles them.

### The fit is two nested solves

**Inner — a triangular bootstrap.** Given candidate globals, the `L` pillars are solved
SEQUENTIALLY, each against its own ATM expiry's premium, by `brentq` on the pillar level. An option
to `T` reads `L` only on `[0, T]` — the backward recursion consumes exactly `n` intercepts — so the
system is exactly triangular, and the price is monotone in the pillar's level, so a bracketed root
is unique. The ATM ladder therefore reprices to **solver precision** (measured 2.4e-15 relative),
which is a statement about the root find and not about the model.

**Outer — the smile, with `L` concentrated out.** The skew globals are fitted to the WING quotes
with the whole `L` strip re-bootstrapped at every iterate, so every candidate reprices the term
structure exactly and is judged only on the smile. It is DERIVATIVE-FREE (Nelder–Mead): the inner
solve is a root find, so no gradient passes through it. It inherits the plain family's
[sign-free leverage reparametrisation](#hestonnandi-fx) — both skew directions in one box, the
share carrying the sign, `β(1−|l|) ≥ 0` keeping the recursion positive — and adds a second share.

**`α` is a share of the level's own room, and that constraint is not optional.** The plain family
fits `ω` directly in logs, so `h_{t+1} ≥ ω > 0` for free. Here the intercept is DERIVED from `L`,
so an `α` larger than `L(1−β)` makes the variance recursion — and the moment generating function
the pricer inverts — diverge. Measured before this was a share: the adaptive `φ_max` scan ran to
its `2²⁴` cap and every price came back NaN. So `α = a·H0·(1−β)` with `a ∈ (0,1)`, and `γ₁` is
DERIVED as `sgn(l)√(|l|β/α)` rather than fitted. `φ` is a share of `α` (same units, scale-free,
zero is the nested face).

**Two pins, both declared.** `Rho` is **pinned at 0.99 per step** and **refused outside `[0, 1)` at
the read**: the L-parametrisation has
evicted `ρ` from the ATM fit — `L` hits the term structure whatever `ρ` is — into the smile's term
structure alone, and sub-year wings do not identify it. A `ρ ≥ 1` is not merely non-stationary
(`E_0[q_t] = L_t` stops holding, so `L` stops meaning the path it is fitted as) — it turns the
least admissible level below NEGATIVE, and `max(floor, 1e-12)` then admits everything, so the
negative-omega guard is DISABLED rather than tripped. `Tie_Gamma_2` holds `γ₂ = γ₁` by default;
**No** fits the ratio, whose SIGN stays tied because a smile that rises with strike at one horizon
and falls at another is a second kink in an objective that already has one.

**The ladder is the plain one with the wings widened**: the same six ATM rungs, plus 25 delta wings
at 1M, 3M, 6M and 1Y rather than 3M and 6M alone, and `fx_surface_block` is INHERITED unchanged —
same vega weights, same surface-own strikes, same nothing-past-1Y rule, same substitution note. Six
globals reduce to five free ones under the two pins, and five free globals judged on the smile
alone want more than four wing quotes, because the ATM rungs are spent on the `L` pillars and
identify nothing else. The distinct-contract floor rises from six to **eight** for the same reason.

### The negative-omega guard

A pillar demanding `L` to fall FASTER than `ρ` decays it makes `ω_t < 0`, which drives `q` — and
then `h` — negative. On a segment of `n` steps running linearly from `A` to `B`,
`ω_i = A(1−ρ) + (B−A)(1 + i(1−ρ))/n`, so the least admissible level is a closed form:

    B_min = A · (1 − (1−ρ)n / (1 + (n−1)(1−ρ)))

`Declining_Variance` decides what happens there: **Refuse** (default) names the pillar, the level
it wanted, the least admissible one, the premium it can reach and both remedies; **Floor** takes
`B_min` and the note travels into the log beside the fitted parameters. There is no silent third
option. INSIDE the outer search the floor is always taken and its relative miss is added to the
objective at `atm_constraint_weight` (1e4, so a one basis point ATM miss already outweighs the
whole smile residual) — a simplex walks into the corner of a box routinely and an exception there
kills a fit that would have recovered, while `inf` is a wall with no slope out of it.

**Why the guard binds on a RISING term structure, which is worth knowing before reading an `L`.**
A piecewise-linear `L` matched to SEGMENT INTEGRALS is the recurrence `L_k = 2A_k − L_{k−1}`, whose
multiplier is −1: marginally stable, so an error in `L(0)` alternates in sign and never decays.
`H0` therefore sets the PHASE of the whole strip. MEASURED on the four-pillar USDZAR fixture: at
the cold start (`H0` at the front rung's own implied variance, 14.02% annualised) the bootstrapped
strip ZIG-ZAGS — 14.02 / 13.83 / 15.33 / 14.46 / 16.32 percent — and at the converged optimum
(`H0` 13.56%) it is MONOTONE — 13.56 / 14.64 / 14.69 / 15.38 / 15.59. So the oscillation is not
removed by seeding; it is removed by the fit, and what pins `H0` is the smile residual together
with the declining-variance floor rather than any smoothness prior. That is not a defect of the
guard; it is what identifies `H0` here.

The seed still matters, for a DIFFERENT reason and one that is also measured, at the COLD START on
the same ladder: `H0` at the front rung (14.02%) bootstraps 14.02 / 13.85 / 15.30 / 14.48 / 16.31
with nothing floored, while `H0` at the ladder's MEAN (14.44%) bootstraps 14.44 / 13.37 / 15.76 /
14.32 / 16.34 and floors the 3M pillar. Both zig-zag — the seed does not fix that and is not meant
to. What it buys is a FEASIBLE start, which under `Declining_Variance: Refuse` is the difference
between a fit and a refusal.

### Wall time, measured

The outer objective re-bootstraps `L` per iterate and every closed-form call is a per-day backward
recursion, so this family is minutes like its plain sibling. **4.79 s an outer evaluation** on the
four-pillar ladder reaching six months, which puts the declared 300-evaluation cap at **24 minutes**
and 400 at 32 — the default is the largest that fits the half hour, and a fit that stops there
reports itself CAPPED with the residual it actually reached rather than the tolerance it did not.

One reading past the cap, measured on that ladder: at **400** evaluations the fit reads `Alpha`
3.186e-06, `Beta` 0.8271, `Gamma_1` −78.56, `Phi` 1.716e-06, `H0` 7.257e-05, an `L` strip of
13.52 / 14.64 / 14.66 / 15.38 / 15.55 percent annualised, an ATM residual of **0.000e+00** and a
worst wing of 5.66% of premium (0.393 vol points). WHAT FULL CONVERGENCE COSTS AND BUYS IS NOT
MEASURED: the one run taken to Nelder–Mead's own tolerance (1268 evaluations) was taken under the
reused-quadrature-bound defect above, so its numbers are void and it has not been repeated. The cap
is therefore justified on WALL CLOCK, which is measured, and not on a diminishing-returns argument,
which is not.

**It runs on the CPU**, whatever device the job was constructed with, and that is measured rather
than assumed. The A/B/C recursion is `n` SEQUENTIAL steps of about ten elementwise operations over
a 512-element complex vector, which is kernel-launch bound on a GPU: one 126-step price is
**47 ms on the CPU against 186 ms on CUDA** (RTX 3090), and the `φ_max` scan 172 ms against 775 ms.
The gap does not close with panels (16 to 128 measured).

**Every price derives its own `φ_max`, and the shortcut that says otherwise is a trap.** The scan
is 35–184 ms against 8–94 ms for the price itself, so reusing one bound across the ladder is worth
about 4x — and this family shipped that for an evening before it was caught. The reasoning was
that more steps means more variance means faster decay, so the front pillar's bound must cover the
back one. It is right about DECAY and silent about DIVERGENCE: past a parameter- and
step-count-dependent point the A/B/C recursion blows up, so a bound that is too LARGE integrates
garbage. MEASURED at a converged optimum, a 126-step price is 0.7353321384 at `φ_max` 128/256/512
(converged — identical at 64, 256 and 1024 panels), **0.7323069671 at 1024** and **9.4e+55 at
2048**, while the 21-step contract in the same strip wants 512. Carrying the front bound to the
back solved that pillar's `L` against a price 0.4% wrong, and because the ATM ladder is
BOOTSTRAPPED it repriced exactly anyway — the only symptom was the report's own recompute reading a
3.5e-3 residual where it should read 1e-12. Gate:
`test_a_quadrature_bound_is_not_transferable_between_contracts`. The lever is still real, and is
now a [roadmap](roadmap.md) row: a per-pillar bound VERIFIED at the solved level.

### `Quote_Sensitivity` is refused

By name, with the reason: the quote derivative would have to pass through the inner root find by
the implicit function theorem AND through the outer Nelder–Mead. That is real work and it is not
built; a family that answered zeros would be worse than one that says so, and the IFT half is a
[roadmap](roadmap.md) row. **The plain family is NOT the alternative** — the refusal used to name
it, and `HestonNandiModelPrices` declares no `Quote_Sensitivity` field at all, so that sent a desk
to a block which would ignore the switch. What the refusal names now is the quote chains that
really are differentiable: the surface and curve families (`FXVolPrices`, `InterestRatePrices`,
`GBMAssetPriceTSModelPrices`, `HullWhite2FactorModelPrices`), which solve through torch rather than
through `brentq`.

### Positivity is a property of the model, not of the box

Unlike plain Heston-Nandi, the CJOW pair has **no positivity guarantee** for `φ > 0`: the worst
innovation `z = γ₂√h` leaves `q_{t+1} ≥ ω_t + ρq_t − φ − φγ₂²h_t`, whose last term grows with `h`.
No box on the parameters closes this. The simulator therefore floors both states at
`utils.HN_COMPONENT_VARIANCE_FLOOR` (1e-12 per step — 0.16 basis points of annualised vol at the
family's declared 252 steps per year), which is
DECLARED rather than applied quietly, and the calibration reports the worst-case certificate
`worst_case_variance_drift` instead of pretending to enforce one. Measured on the component TARF
gate: **2 of 8192 inner paths** over 248 daily steps at a fitted `φ` share of 0.56 — exactly the
frequency that is easy to miss, and without the floor it is a NaN in an exposure profile and a CVA.
The closed form does NOT floor, so the two agree only where the floor is inactive; the
closed-form-versus-Monte-Carlo gate asserts the margin (measured 2.6e+07 to 3.9e+07 times the
floor) rather than assuming it.

## `FXVolPrices` — a smile quoted in delta, and where the conversion runs {#fxvolprices}

An FX smile ticks in as DELTA quotes: an ATM vol per expiry and, per delta pillar, the risk
reversal and the butterfly around it. The surface the pricers read is a log-moneyness one. Both
halves of the conversion between them already existed — the strangle pair
`vol(call) = ATM + BF + RR/2`, `vol(put) = ATM + BF − RR/2`, and the delta-to-strike solve
`Factor2D` runs for a `Malz` surface — so this family is not new maths. **It is the same
conversion moved, and the move is the point.**

**The x-grid is pinned, because refinement is compile-time work.** The delta solve does not
evaluate a smile at prescribed strikes; it refines a log-moneyness grid until interpolating total
variance between the nodes resolves the smile to `Grid_Tolerance`. That grid is a function of the
quotes. Run at factor-construction time — which is where it ran — every vol tick was potentially
STRUCTURAL: a moved node is a moved tenor grid, a new plan and a recompile, for a number that
changed. So the refinement runs in the bootstrap, once, and the grid it produced is part of the
written factor. A re-bootstrap that finds a surface already written **for the same expiries at the
same `Grid_Tolerance`** reuses its grid and moves only the vols on it, which is exactly a
`bind='value'` patch; a quote set whose expiries have changed is not that plan's, and is refined
from scratch. So is one asking for a different tolerance — the tolerance is what SIZES the grid,
and honouring it on the vols while ignoring it on the nodes would make it a field nothing reads.
It is written onto the surface it built and it is structural, unlike the stamp beside it. What the
pin costs is measured rather than hidden: the log says what the pinned grid resolves the CURRENT
quotes to, beside the tolerance it was built at.

`Surface_Type` was doing two jobs and only one of them is about the solver. It names the moneyness
convention the engine reads a surface at — `calc_moneyness` returns log(F/K), and the term
interpolation runs in total variance — and it said a delta smile is the form the block was authored
in. `Factor2D.solves_delta_surface` separates them: a `Malz` block **carrying deltas** is unsolved
and gets solved on construction, as before; one carrying only the solved `Surface` is what this
family writes, and it falls straight through with its grid intact.

**The conversion is vectorized.** It was a Python loop over the x-grid with a scalar `brentq`
inside it, per point, per expiry, per refinement pass. It is now one bisection over the whole grid
(`Factor2D.malz_sigma`), which refines the identical grid, agrees with the loop to 5e-14 vol —
well inside `brentq`'s own 2e-12 `xtol` — and is made of operations an autograd tape can carry,
which the scipy call is not. The `brentq` loop that was its oracle lived in
`tests/test_fx_vol_prices.py` and went with the mock-built suite, so the 5e-14 agreement is a
recorded measurement rather than a standing gate.

**The conventions are declared, and each offers exactly one value**, because the solve implements
exactly one: `Delta_Type` `Forward`, `Premium_Adjusted` `Yes` — the pillar delta is `(K/F)N(d₂)` —
and `ATM_Convention` `Delta_Neutral_Straddle`, the strike `K = F exp(−σ²T/2)` at which that
convention's straddle is delta neutral. A spot delta or an ATMF quote needs different algebra, and
a value the engine cannot honour is the same defect as a field nothing reads. They WERE gated as
maths rather than as strings — the surface must carry the pillar's vol at the strike whose
premium-adjusted forward delta IS the pillar, and must fail the same statement under the
unadjusted delta — and that gate went with the mock-built suite; today only the single-valued
declarations hold the line.

**The ATM row is the surface's ATM vol, and a second family reads it as one.** `malz_skew` places
the ±0.5 label's vol at the delta-neutral straddle strike, so the ATM vol of the surface this
family writes IS the quoted number — nothing has to be read back off the refined grid to recover
it. `FXVolSurfaceParameters.atm_quotes` is the one reader of that rule (`smile` builds the wings
around the same dict), and `GBMAssetPriceTSModelParameters` takes those rows as its ATM column
wherever the surface it is integrating is one this family **wrote** — evidenced by the same
fingerprint `pinned_grid` reads back, never by the name alone. That is what makes a vol delta land
on the number a desk edits — see
[the quote sources](quote_sensitivities.md#the-atm-column), and the defect named beside them.

**The smile differentiates in its own quotes.** `Quote_Sensitivity` — declared here, default `No` —
leaves the written surface connected to the ATM / RR / BF numbers it was built from, so a backward
pass reports `dV/d(risk reversal)` beside `dV/d(surface node)`. The written surface is bit-identical
either way; what the conversion has that the other families do not is a **root find**, and the tape
does not enter it. See
[increment 4](quote_sensitivities.md#the-delta-solve) for the boundary and what taping the
bisection instead would have reported.

!!! warning "One ATM quote, two families, two partial derivatives"
    The two paragraphs above stack: this block's ATM row is also `GBMAssetPriceTSModelPrices`'
    ATM column, so with **both** blocks asking for `Quote_Sensitivity` a single JSON number reaches
    a valuation through two independent maps and its `dV/dq` arrives split over two `quote_leaves`
    entries — under the *same* descriptor string, because a quote is named by what it is and not by
    which family read it. Each half is correct and neither is the total. A consumer must group by
    descriptor across blocks and **sum**; see
    [the collision](quote_sensitivities.md#the-attachment) for the measured numbers.

**A point may carry a two-way, and the bootstrap never reads it.** `Quoted_Bid` and `Quoted_Ask`
are optional columns beside the mid: `derivus_bloomberg` writes them onto a row when the terminal
answered `PX_BID`/`PX_ASK` for that pillar, and leaves the row exactly as it always was when it did
not — a mid-only surface's block is byte-identical to the one this family has always been handed.
Everything below this line reads `Quoted_Market_Value` by name, so the surface, the pinned grid and
every mark on the book are built from the **mid** whether or not the sides are there. The one
reader is the quote layer: [`derivus.structures`](structures.md#two-sided) shifts a leg's own copy
of the written surface by the ATM half-spread to quote a client two-sided. *The spread is the
quote's; the mid is the book's.*

They tick like the mid, too. Both are on the value side of `update_market_quote`'s structure guard,
because a spread widens between one print and the next and a pillar that starts or stops being
quoted two-sided is still the same node of the same plan; a moved `Pillar` or `Expiry` refuses as
it always did. That guard IS the section's plan/values split — `schema.MARKET_QUOTE_VALUES`, read
by the guard, by `plan_hash`, by `market_patch`/`patch_market` and by the artifact slot alike, so a
vol tick moves `values_hash` and leaves `plan_hash` alone. See
[Quote Propagation](quote_propagation.md#protocol) for what that buys and for the five families
whose quotes are not `Points` rows and are therefore wholly plan-side.

**Timestamps are data the engine stores and reports.** Each quote row carries when it was seen;
the written surface carries the latest of the rows that built it as `Quote_Timestamp`, its own
as-of. It is declared `bind='value'` because it travels with the vols — a tick delivers new numbers
and the time it saw them together, and a stamp that invalidated the plan would recompile on every
tick. It enters `values_hash` and therefore the replay identity, which is right: the same numbers
read off a different snapshot are a different market event. **Nothing in pricing reads it.** What
counts as too old is a policy, and policy belongs to the consumer, not to the surface.

**That identity is FULL resolution.** `CustomJsonEncoder` writes a midnight `Timestamp` as the
plain date it always did — old files re-encode byte-stable — and a non-midnight one in ISO form
with its time, which the decoder always parsed. So the 09:15 and the 16:30 snapshots of a quote
survive a save as themselves and `values_hash` separates them, which closes the capability this
field was first to depend on.
