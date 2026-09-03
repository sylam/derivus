# Market Prices

`Market Prices` is the risk-neutral half of the market data: the **quotes** a risk-neutral model is
fitted to, where `Price Factors` holds the curves and surfaces a historical calibration produces. A
*bootstrapper* turns each block into the factor or model parameters the simulation reads. All seven
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

`Quote_Type` is per family, not global: Clewlow–Strickland takes implied vols, Heston-Nandi a vol or
a premium, an interest-rate quote a par rate.

## A family maps a quote set to what it calibrates {#a-family}

A bootstrapper class is one price family. It declares:

- `market_factor_type` — the `Market Prices` type string a block is filed under, and the string the
  class selects its own work by. Declared rather than recovered from the class name, because the
  block is `HestonNandiModelPrices` while the class is `HestonNandiModelParameters`.
- `fields` — the block's schema, including the quote table or container as the class reads it.
- `quote_instruments`, where the quotes are instruments rather than a fixed option table.

`mapping['MarketPrices']['types']` is `schema.emit_market_prices(bootstrappers)` over those
declarations, and `construct_bootstrapper` resolves the class by name from the
`Bootstrapper Configuration` section.

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

## `HestonNandiModelPrices` — a block authored off a built surface {#hestonnandi-fx}

The family fits five parameters to European options. For an equity somebody types the strikes; for an
FX pair `HestonNandiModelParameters.fx_surface_block` reads the `FXVol` surface
[`FXVolPrices`](#fxvolprices) built and authors the block: **ten vega-weighted implied vols — ATM at
1M, 2M, 3M, 6M, 9M and 1Y, plus the 25 delta wings at 3M and 6M**. The term structure identifies
`H0`, `Beta` and `Omega`; the skew identifies `Gamma_Star`, the wings' width `Alpha`. `Weight` is the
normalised Black vega off the same surface. Nothing past 1Y: TARFs and accumulators are sub-year, and
a parameter fitted to the 2Y smile is borrowed against products nobody quotes.

**The strikes are the surface's own coordinates.** ATM is the delta-neutral straddle
`K = F exp(-σ²T/2)` the surface was built under; each wing is the strike whose premium-adjusted
forward delta IS the pillar, found by inverting the same delta `Factor2D` Malz solve off the same
vols. An expiry the surface does not carry moves to the **nearest quoted one at or under a year**,
named in `Quote_Source`; interpolating between two pillars would put a number nobody quoted into the
objective. The surface's `Quote_Timestamp` travels onto the block.

**Ten rungs are not ten quotes, and the count is what refuses.** Snapping is an argmin, so a rung the
surface does not carry lands on a contract another rung already named — a repeated contract is a
WEIGHT, not an observation. The canned two-pillar USDZAR surface collapses the ladder onto **four**
distinct `(expiry, strike)` contracts and a single-expiry surface onto three, and four observations
do not identify five parameters. So DISTINCT contracts are counted after snapping and a ladder below
**six** refuses by name, with the surface's pillars in the message. Four pillars (1M/2M/3M/6M) are
the fewest that clear it, at eight contracts.

**The cap is applied, not hoped for.** An argmin has no ceiling, so on a surface quoting 2Y every rung
would answer 2Y. Candidates are filtered to the ladder's own longest rung plus a week, and a rung with
nothing admissible under it is DROPPED — which `Quote_Source` says, and which can drop the ladder
below the six-contract floor and refuse there.

**The vol is read at the PILLAR; the strike hangs off the DATE.** The pillar is emitted as the nearest
whole day and the fit reads its accrual back off that date through the discount curve's day count.
That accrual is what the FORWARD is built at, but it is NOT where the surface is read: under ACT_360 a
1Y pillar resolves to 1.0139, and reading there walks every rung off its pillar and puts the 1Y rung
past the surface's last expiry (+0.0036 vol points at the 3M rung on the canned surface with the USD
curve on ACT_360).

**`Volatility_Delta` is not folded in.** The block is a QUOTE; the scenario shift is the FIT's
business — `bootstrap` adds `vol_surface.delta` to every quoted vol it prices a target premium off.
Adding it here as well would calibrate a two-vol-point world for a one-vol-point scenario. Gated as
an identity between two real calibrations, to **7.7e-9 relative**.

**Orientation.** An `FXVol.A.B` surface's x-axis is `log(F/K)` on the pair *A priced in B*; the 0D
factor the family fits is an `FxRate`, priced in the DOMESTIC currency. So the underlying is the token
that is not domestic, the strikes are in that factor's units, and the block declares `Use_Forward`
**Yes** with `Invert_Moneyness` set exactly as an `FXOptionDeal` on that surface sets it. The written
`HestonNandiModelParameters.<underlying>` is the factor an FX accrual leg resolves by naming
convention off the pair's NON-BASE token (`utils.spot_model_currency`). Inverting the rate flips the
sign of the skew: `utils.hn_reciprocal_gamma` maps `gamma*` to `1 − gamma*` for a deal written on the
reciprocal axis under that deal's own numeraire.

**Both signs of the skew live in one box.** `Gamma_Star`'s sign IS the direction of the smile —
positive is the equity leverage shape, and USDZAR is the other one once read as `FxRate.ZAR`. It
cannot be bounded across zero, because `Alpha = |l|ψ/Gamma_Star²` is singular there and the
singularity is real: holding the skew channel fixed while `Gamma_Star → 0` sends the wings' width to
infinity. So the **leverage share carries the sign** — `x[3]` is the magnitude bounded away from zero
and `x[2]` a signed share in `[-1, 1]` whose sign is `Gamma_Star`'s, with `Alpha = |l|ψ/Gamma_Star²`
and `Beta = ψ(1-|l|)`, every iterate feasible. At `l = 0` there is no leverage channel and
`Gamma_Star` is unidentified, which is what a flat surface legitimately reports. A cold start seeds
the SIGN off the quotes (a smile rising with strike is a negative `Gamma_Star`), because the objective
has a kink at zero leverage.

**`POST /book/hn` is the verb**, calibrate-on-request at the heavy cost class. The fit is a least
squares over a Fourier-inverted daily GARCH recursion and it is MINUTES: **288 s** on the four-pillar
ladder reaching six months (549 s with the suite running beside it, to the same five parameters bit
for bit), past 21 minutes on one reaching a year. Iteration count dominates step count, so neither
reading predicts the other; both share that the adaptive `phi_max` scan is **31–38% of every
option price** since the doubling ladder batched. So it rides no tick, and a market tick
leaves the parameters where they were. `Bootstrapper Configuration` names the families that run on
every bootstrap and a tick is a bootstrap, so the verb BORROWS the family entry for its run and hands
it back. It drops its own block before re-installing it, because these strikes are a FUNCTION of the
surface: re-emitting after a tick legitimately moves them, which `update_market_quote` would refuse.
There is no GET side — the written factor is the projection and `GET /book` serves it.

## `HestonNandiComponentModelPrices` — a term structure fitted as a curve {#hestonnandi-component}

The plain family fits ONE `Omega`, so the whole ATM term structure has to be bought with `H0`, `Beta`
and that constant. The Christoffersen–Jacobs–Ornthanalai–Wang component model splits the variance
into a long-run component `q` and a short-run deviation, and this family fits `q`'s expected path:

$$h_{t+1}=q_{t+1}+\beta(h_t-q_t)+\alpha\Big[(z_t-\gamma_1\sqrt{h_t})^2-(1+\gamma_1^2h_t)\Big]$$
$$q_{t+1}=\omega_t+\rho q_t+\phi\Big[(z_t-\gamma_2\sqrt{h_t})^2-(1+\gamma_2^2h_t)\Big]$$

Both bracketed terms are exactly centered, so `h − q` is a pure AR(1) at `β` and `E_t[q_{t+k}]` is
driven by `ω` alone.

**`Ω` is a curve, not a number.** Writing `ω_t = L_{t+1} − ρL_t` makes `q_t − L_t` a homogeneous
AR(1), so anchoring `q_0 = L(0)` gives `E_0[q_t] = L_t` exactly. `L` is therefore the model's expected
long-run variance path, directly comparable to a market forward variance strip. It is
piecewise-LINEAR between pillar knots and flat outside them, so `ω_t` is affine within a pillar —
drifting `(B−A)(1−ρ)/n` per step and kinking only AT a pillar, which is what gives the
declining-variance floor below a closed form. The stored curve carries a knot at tenor 0 whose value
is `H0`, which makes `q_0 = L(0)` a property of the written factor. There is no `Omega` field and no
`Q0` field.

**It nests the plain family exactly.** Set `φ = 0` and hold `L` flat and the h-recursion collapses onto
the plain one under

    ω_p = L(1−β) − α,   β_p = β − αγ₁²,   α_p = α,   γ_p = γ₁

whose inverse is `β = ψ_p` (the plain PERSISTENCE, not the plain `Beta`) and `L` = the plain
STATIONARY variance. `utils.hn_component_from_plain` / `hn_component_to_plain` are that map, and
`tests/test_hn_component.py` holds the two closed forms to each other across a strike/expiry grid at
**1.5e-13 relative** — machine precision, because the map is exact. The A/B/C recursion's third
coefficient is what makes it close: `A` alone does not reduce to the plain `A`, and the anchoring
`q_0 = L(0)` reconciles them.

### The fit is two nested solves

**Inner — a triangular bootstrap.** Given candidate globals, the `L` pillars are solved SEQUENTIALLY,
each by `brentq` against its own ATM expiry's premium. An option to `T` reads `L` only on `[0, T]`, so
the system is exactly triangular, and the price is monotone in the pillar's level, so a bracketed root
is unique. The ATM ladder reprices to **2.4e-15 relative** — a statement about the root find, not the
model.

**Outer — the smile, with `L` concentrated out.** The skew globals are fitted to the WING quotes with
the whole `L` strip re-bootstrapped at every iterate, so every candidate reprices the term structure
exactly and is judged only on the smile. Derivative-free (Nelder–Mead), because the inner solve is a
root find. It inherits the plain family's [sign-free leverage
reparametrisation](#hestonnandi-fx) — with `β(1−|l|) ≥ 0` keeping the recursion positive — and adds a
second share.

**`α` is a share of the level's own room.** The plain family fits `ω` directly in logs, so
`h_{t+1} ≥ ω > 0` for free. Here the intercept is DERIVED from `L`, so an `α` larger than `L(1−β)`
diverges the variance recursion and the moment generating function the pricer inverts — measured
before this was a share: the adaptive `φ_max` scan ran to its `2²⁴` cap and every price came back NaN.
So `α = a·H0·(1−β)` with `a ∈ (0,1)`, `γ₁` is DERIVED as `sgn(l)√(|l|β/α)` rather than fitted, and `φ`
is a share of `α` (same units, scale-free, zero is the nested face).

**Two pins.** `Rho` is **pinned at 0.99 per step** and **refused outside `[0, 1)` at the read**: the
L-parametrisation evicts `ρ` from the ATM fit into the smile's term structure alone, and sub-year
wings do not identify it. A `ρ ≥ 1` also turns the least admissible level NEGATIVE, so
`max(floor, 1e-12)` admits everything and the negative-omega guard is disabled rather than tripped.
`Tie_Gamma_2` holds `γ₂ = γ₁` by default; **No** fits the ratio, whose SIGN stays tied because a smile
that rises with strike at one horizon and falls at another is a second kink in the objective.

**The ladder is the plain one with the wings widened**: the same six ATM rungs plus 25 delta wings at
1M, 3M, 6M and 1Y, with `fx_surface_block` inherited unchanged. Six globals reduce to five free ones
under the two pins, and five free globals judged on the smile alone want more than four wing quotes,
because the ATM rungs are spent on the `L` pillars. The distinct-contract floor rises from six to
**eight**.

### The negative-omega guard

A pillar demanding `L` to fall FASTER than `ρ` decays it makes `ω_t < 0`, driving `q` and then `h`
negative. On a segment of `n` steps running linearly from `A` to `B`, the least admissible level is
closed form:

    B_min = A · (1 − (1−ρ)n / (1 + (n−1)(1−ρ)))

`Declining_Variance` decides what happens there: **Refuse** (default) names the pillar, the level it
wanted, the least admissible one, the premium it can reach and both remedies; **Floor** takes `B_min`
and the note travels into the log beside the fitted parameters. There is no silent third option.
INSIDE the outer search the floor is always taken and its relative miss added to the objective at
`atm_constraint_weight` (1e4, so a one basis point ATM miss outweighs the whole smile residual) — a
simplex walks into the corner of a box routinely, an exception there kills a fit that would have
recovered, and `inf` is a wall with no slope out of it.

**The guard binds on a RISING term structure.** A
piecewise-linear `L` matched to SEGMENT INTEGRALS is the recurrence `L_k = 2A_k − L_{k−1}`, whose
multiplier is −1: marginally stable, so an error in `L(0)` alternates in sign and never decays, and
`H0` sets the PHASE of the whole strip. Measured on the four-pillar USDZAR fixture: at the cold start
(`H0` 14.02% annualised) the strip ZIG-ZAGS — 14.02 / 13.83 / 15.33 / 14.46 / 16.32 percent — and at
the converged optimum (`H0` 13.56%) it is MONOTONE — 13.56 / 14.64 / 14.69 / 15.38 / 15.59. The
oscillation is removed by the fit, not by seeding; what pins `H0` is the smile residual with the
declining-variance floor, and that is what identifies `H0` here. What the seed buys is a FEASIBLE
start (`H0` at the front rung floors nothing; at the ladder's mean, 14.44%, it floors the 3M pillar),
which under `Refuse` is the difference between a fit and a refusal.

### Wall time, measured

**One backward recursion for the bound strip plus one per quadrature bound the evaluation widens
to** — two on the four-pillar ladder, and that is the floor there. `B` and `C` never read the
`ω` strip and are time-homogeneous, and `A` is affine in it, so ONE pass at the longest maturity
carries every maturity (a prefix), every `L` curve (a dot product) and every cost of carry (a
per-step constant) — `utils.hn_component_abc_strip`, held by `bootstrappers.ComponentStrips`, which
lives inside one evaluation and dies with it. The quadrature grid is a NESTED dyadic union
(`utils.gauss_legendre_dyadic`): fixed blocks [0,8], [8,16], [16,32], …, each carrying at least the
panel width `Quadrature_Panels` uniform panels buy over that block's own bound, so a contract at
bound 2ᵏ integrates a PREFIX of one grid — 2,048 nodes at bound 512 against 512, at accuracy at or
above the uniform grid's on every rung by construction, and measurably above it past 512, where 64
uniform panels are under-resolved by up to 2.9e-08 relative.

**0.176 s an outer evaluation** on the four-pillar ladder reaching six months, against 2.21 s
before the strips: the declared 300-evaluation cap is **53 s against 662 s**, not 24 minutes, and a
fit that stops there still reports itself CAPPED with the residual it reached. The profile is 63%
the two recursions (the branch unwrap 26% of the evaluation), 11% the bounds, 12% the quadrature
and its grid, 14% `brentq` and the Python glue. **What full convergence buys, now measured**:
Nelder–Mead to its own tolerance is 1,246 evaluations and 243 s for a wing residual of 1.604e-03
against **2.049e-03** at the 300 cap (the objective's own value, scaled by the mean squared premium;
the fit log's unscaled 1.196e-04 is the same number) — 22% better, in a different basin (`Gamma_1` −72.1 against
−845.1). The cap is a policy call now, not a wall-clock one: the half hour buys about
10,000 evaluations.

**It runs on the CPU**, whatever device the job was constructed with. The evaluation's 126-step
pass over the union grid — 2 contours × 2,048 complex nodes — is **53.9 ms on the CPU against
112.6 ms on CUDA** (RTX 3090); widening to 3,072 nodes closes the gap only to 1.2×, the recursion
being 126 sequential kernel launches whatever the node count. The pin stays.

**Every price derives its own `φ_max`, and now for free.** The scan's criterion is the same affine
form as the price, so the bound is READ off a 21-rung strip (`utils.hn_component_strip_phi_max`)
rather than scanned for; over a replayed fit it lands the scanned bound at all 154 prices, the
nearest rung sitting 0.06 units of the metric from the threshold against a 1e-15 perturbation.
Sharing one would still be wrong: past a parameter- and step-count-dependent point the A/B/C
recursion DIVERGES, so a bound that is too large integrates garbage. Measured at a converged
optimum: a 126-step price is 0.7353321384 at `φ_max` 128/256/512, **0.7323069671 at 1024** and
**9.4e+55 at 2048**, while the 21-step contract in the same strip wants 512. Gate:
`test_a_quadrature_bound_is_not_transferable_between_contracts`.

### `Quote_Sensitivity` is refused

Refused by name: the quote derivative needs IFT through the inner `brentq` plus a rule for the
derivative-free outer search; not built (roadmap). **The plain family is not the alternative** —
`HestonNandiModelPrices` declares no `Quote_Sensitivity` field at all, so naming it would send a desk
to a block that ignores the switch. The refusal names the quote chains that are differentiable
instead: `FXVolPrices`, `InterestRatePrices`, `GBMAssetPriceTSModelPrices` and
`HullWhite2FactorModelPrices`, which solve through torch rather than through `brentq`.

### Positivity is a property of the model, not of the box

Unlike plain Heston-Nandi, the CJOW pair has **no positivity guarantee** for `φ > 0`: the worst
innovation `z = γ₂√h` leaves `q_{t+1} ≥ ω_t + ρq_t − φ − φγ₂²h_t`, whose last term grows with `h`, and
no box on the parameters closes it. The simulator floors both states at
`utils.HN_COMPONENT_VARIANCE_FLOOR` (1e-12 per step — 0.16 basis points of annualised vol at the
family's 252 steps per year), DECLARED rather than applied quietly, and the calibration reports the
worst-case certificate `worst_case_variance_drift` instead of pretending to enforce one. Measured on
the component TARF gate: **2 of 8192 inner paths** over 248 daily steps at a fitted `φ` share of 0.56
— without the floor that is a NaN in an exposure profile and a CVA. The closed form does NOT floor,
so the two agree only where the floor is inactive; the closed-form-versus-Monte-Carlo gate asserts the
margin (2.6e+07 to 3.9e+07 times the floor) rather than assuming it.

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
