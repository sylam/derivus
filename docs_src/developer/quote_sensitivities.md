# Quote Sensitivities

A desk explains P&L in the numbers it quotes, while the engine's tape starts at calibrated factors,
so the library owns each calibration and differentiates through its fixed point: one backward pass
reports `dV/dq` beside `dV/dθ`. Every carrier is a splice worth exactly zero forward, so switching it
on cannot move a mark.

```
q (leaves)  ->  calibration  ->  θ*  ->  factor leaves  ->  scenarios  ->  V
```

The switch is `Quote_Sensitivity` (default `No`) on each `Market Prices` block; the middle arrow is
the family's:

| `Market Prices` type | the arrow | the backward |
| --- | --- | --- |
| `InterestRatePrices` | damped Newton on a square root | [implicit function theorem](#the-ift-contract), exact |
| `SurvivalProbPrices` | the same solve, a block at a time | the same theorem, exact, chained into its discount curve's |
| `InflationRatePrices` | the same solve, a block at a time | the same theorem, exact; its discount curve's term is zero at the root |
| `HullWhite2FactorModelPrices` | least squares, `Objective` `Analytic` by default | Gauss–Newton at a [stationary point](#the-stationarity-contract) |
| `LogVar2FJModelPrices` | staged least squares | the same node, `Fit_Mode: Global` only |
| `GBMAssetPriceTSModelPrices` | an [explicit map](#the-closed-form-map) | autograd through a twin |
| `FXVolPrices` | a [bisection per grid node](#the-delta-solve) | one Newton step at each root |

`CSForwardPriceModelPrices` has none. The methods and their shared maths are on
[Bootstrapping](../bootstrapping/general.md); this page is how the code carries them.

| where | what |
| --- | --- |
| `bootstrappers.BenchmarkInstruments`, `utils.TensorSchedule.carry` | the curve residual and its quote overlay |
| `utils.damped_newton`, `CalibrationSolve`, `residual_jacobians` | the curve solve and its backward |
| `utils.LeastSquaresSolve`, `utils.Residual`, `utils.active_set` | the least-squares node and its operand |
| `SwaptionCalibration`, `utils.create_market_swaps`, `utils.market_swap_class` | the HW2F operand and splice |
| `GBMAssetPriceTSModelParameters.integrated_vol`, `bootstrappers.FXVol.carried_sigma` | the constructions' tapes |
| `Calculation.factor_leaf`, `Config.bootstrap` | the attachment and the harvest |
| `structures.vol_risk`, `service.quote_rows` | the readers of `dV/dq` |

A rule marked **Held by** goes red under the named test if it is broken.

## The curve residual {#the-residual}

A quote names an instrument and carries a block of it ([Market Prices](market_prices.md#a-quote)), so
the residual is the engine's own pricers held at par, $F_i(\theta, q_i) = PV_i(\theta, q_i)|_{t_0}$.
`BenchmarkInstruments` compiles the benchmark nodes once and prices them at t0 off a `{Factor: tensor}`
of curve nodes; a node with `Children` is one benchmark, with no netting rule on top. Every factor not
being solved for — an `FXForwardDeal`'s other curve and `FxRate` included — is a detached constant,
or a [connected](#the-ift-contract) one where another family solved it on the tape.
`held` knots at the front of a solved factor stay where `Price Factors` writes them, a survival
curve's anchor at tenor zero being one, so theta is the knots after them.

### The graph audit {#the-graph-audit}

A severance raises nothing and moves no value; it reports a zero gradient. The closure avoids four:

| where | severs | avoided by |
| --- | --- | --- |
| `_build_factor_state`, `BaseValuation.update_factors` | every leaf is minted from numpy | θ goes straight into `t_Static_Buffer`, where pricers read a static curve |
| `Factor1D.current_value` | numpy end to end | used for constants only; θ follows the knot order `get_tenor` leaves on the constructed factor |
| `Factor1D.check_interpolation` | the Hermite pair from numpy | `utils.Interpolation.build` re-derives it from the tensor (no test per kind) |
| `TensorSchedule.bind` | `new_tensor` copies the fixed rate | [the overlay](#the-quote-side) |

Two silent traps: `t_Buffer` is keyed by factor and time, not tensor identity, so `Benchmark_State` is
built per evaluation (no test checks it); `utils.CurveTenor` caches its grid from the first tensor
asking, so `all_tenors` is rebuilt per closure.

### The quote side {#the-quote-side}

`TensorSchedule.carry` attaches a `{column: tensor}` overlay to a schedule's tensor half, spliced into
the copy `bind` makes ([lifecycle](calc_lifecycle.md#the-schedule-lifecycle)), so
`BenchmarkInstruments` binds last. `_carry_quotes` measures the columns: it compiles the set again one
percent higher and overlays what moved, the difference as slope. Authoring is affine in the quote, so
that difference is exact, and a type's own `quoted` stays the one place it declares where its quote
lands. The splice is

$$\text{column} = \text{base} + \big(q - \texttt{detach}(q)\big)\,\partial\,\text{column}/\partial q .$$

A moved reset column raises, a reset also leaving through `known_resets`, which reads numpy. A
benchmark moving no column — an `FXForwardDeal`, whose outright lands in `Buy_Amount` — raises by name,
measured rather than branched on type so it lapses when such a type grows a schedule. **Held by**
`tests/test_interest_rate_prices.py::test_a_forward_block_refuses_quote_sensitivity`,
`::test_the_refusal_is_measured_and_leaves_the_rate_quotes_carrying`.

!!! warning "A derivative carrier, not a reparameterisation"
    Worth zero forward, the overlay does not follow a quote that moves; a moved quote needs a fresh
    closure. `pv_fixed_cashflows` memoizes in the schedule's `derived`, which `bind` re-mints.

## The curve solve {#the-solve}

`utils.damped_newton` flattens every curve being solved into one system whose Jacobian, by autograd a
row per benchmark, is the one the backward needs too: the residual is written once and differentiated
twice [@dennis1996]. The seed is each par quote as a zero rate. Damping halves the step until the
max-norm residual decreases; convergence is tested on the step before the line search, as a step
inside the linear solve's rounding cannot reduce a residual at noise level. A coupled set takes its
members' strictest knobs.

| field | default | why |
| --- | --- | --- |
| `N_Iter` | 50 | a par seed is basis points off; the cap raises rather than return a half-solved curve |
| `Tol` | 1e-13 | ~1e-11 relative on a zero rate, inside a round trip's 1e-10; a floor on the step, not a target |
| `Damping_Halvings` | 6 | past that the step's length is not what is wrong |

**Held by** `tests/test_interest_rate_prices.py::test_the_bootstrap_recovers_the_curve_its_quotes_came_from`,
`::test_the_solver_knobs_are_read_off_the_block`, `::test_a_tighter_tolerance_still_converges_to_the_same_curve`.
No test world takes a halved step.

### The knot rule {#the-knot-rule}

One knot per used quote, at the benchmark's last cashflow date, and the output grid is that grid
([Market Prices](market_prices.md#interestrateprices)). It makes $\partial F/\partial\theta$ square, so
the quote-delta buckets are the benchmark maturities.

### Curve contracts {#curve-contracts}

- **No knot at tenor zero.** It identifies nothing and Newton meets it as a singular Jacobian, so a
  benchmark maturing on or before the base date refuses by name at seeding. **Held by**
  `tests/test_interest_rate_prices.py::test_a_benchmark_matured_at_the_base_date_refuses_by_name`.
- **An OIS benchmark is a term swap**: at t0 compounded overnight forwards telescope to the period
  forward, so one reset per coupon prices what the daily list prices. **Held by**
  `::test_a_term_ois_benchmark_prices_the_fixing_list_it_replaces`.
- **Several resets per coupon fold by the leg's declaration**: `None` averages, the OIS methods
  compound, undefined combinations refuse; a reset count differing from the cashflow count is the only
  signal. **Held by** `::test_a_coupon_on_several_resets_averages_them_unless_its_leg_compounds_them`.
- **A quoted deposit is pinned whatever its schedule**: `DepositDeal.quote` writes the rate on each
  period's unadjusted start, the day it can name with no calendar, and the deposit reads its rows
  there, so a stub or an accrual calendar leaves the benchmark off the curve it solves. **Held by**
  `::test_a_quoted_deposit_is_pinned_on_the_schedule_its_stub_and_calendar_spell`.

## The IFT contract {#the-ift-contract}

`utils.CalibrationSolve` is the curve solve as one node. **Forward** is `damped_newton` and nothing
else, grad mode re-enabled inside for its own Jacobian; with no quote leaf it is a pass-through.

**The switch changes the set.** With `Quote_Sensitivity` or `Quote_Propagation` on, the family
measures which blocks' residuals read each other's curves (`BenchmarkInstruments.reads`) and solves
each [coupled set](quote_propagation.md#multi-curve) within a currency as one system, the only
Jacobian carrying $d\theta_2/dq_1$. Its one vector leaf is filed under each member that asked,
descriptors prefixed by block.

**Backward** is the implicit function theorem [@christianson1998; @henrard2014], never an unrolled
solver: $(\partial F/\partial\theta)^\top w = v$, then $\partial L/\partial q = -(\partial F/\partial
q)^\top w$. `residual_jacobians` takes both from one backward per benchmark, recomputed at θ\* because
the last Newton step was taken at the iterate before it. Every cotangent at once is
`calibration_jacobian`, the operator [Quote Propagation](quote_propagation.md) rides.

**A connected constant** is a factor another family solved on the tape, which `Config.bootstrap`
hands each family as `Family.connected`: a residual reading one, a survival or inflation curve's
discount curve, reads it through its graph, and the node takes it as a further input $c$ and returns
$-(\partial F/\partial c)^\top w$ for it, so autograd carries the chain rule on through the solve
that wrote $c$, the two solves in turn and never one system.

!!! warning "Every `grad` in the backward retains the graph"
    The residual's subgraph is shared with the forward through `pv_fixed_cashflows`' memo, so freeing
    it would take the forward's graph.

**Held by** `tests/test_quote_propagation.py::test_the_artifact_holds_the_calibration_jacobian_the_solve_would_have_differenced`,
`tests/test_service.py::test_the_consolidated_risk_is_read_on_the_quotes_the_curves_are_built_from`
(red with the blocks solved one at a time),
`tests/test_credit_prices.py::test_a_default_swaps_delta_to_a_swap_quote_carries_the_survival_curves_re_solve`
(red with the connected curve's cotangent not returned).

## The stationarity contract {#the-stationarity-contract}

`utils.LeastSquaresSolve` is the same contract at a stationarity point, $J^\top r = 0$. Its operand,
a `utils.Residual`, supplies `solve()`, the residual `__call__(x)`, `interior(x, g)`, `labels` and
`descriptors`; the HW2F and LogVar2FJ families differ there and nowhere else. HW2F's forward is basin
hopping then least squares, a candidate kept only if it beats the running best with a well-posed
process — so θ\* can be the seed — seeded by the declared `Random_Seed` alone. `Objective` is
`Analytic` (Schrager–Pelsser [@schrager2006]; [why](../limitations.md#model-punchlist)) or
`Monte_Carlo`, which an analytic solve runs once at θ\* to log its worst benchmark.

### The Monte Carlo loss is conditional on its draw {#the-mc-loss}

Under `Monte_Carlo`, $r_i = w_i\big(100(P_i(q_i)/M_i(\theta) - 1)\big)^2$ with $M_i$ priced on Sobol
paths the block's `Quasi_Seed` scrambles. The sample is drawn once per closure and the memos cleared per evaluation, so every
evaluation, Jacobian and backward prices the same paths: the reported derivative is the derivative of
the reported number, where a re-drawn sample would have the optimizer differencing noise.

### The swaption quote splice {#the-swaption-quote-side}

The one severance is the numpy market premium. Each benchmark carries a float64 quote leaf (the vol,
or the premium where a file supplies it) and a `premium` callable through the tensor twin of the
surface's `Distribution_Type` formula, spliced in `market_swap_class.error` as
$\text{error} + (\widetilde{\text{error}} - \texttt{detach}(\widetilde{\text{error}}))$:

- **the model price is detached in the carried half only** — attached, it doubles the calibration
  Jacobian while the residual stays bit-identical;
- **the splice sits at the error, not the price**, as torch divides by a numpy scalar at the scalar's
  precision and would round twice;
- **the premium is rebuilt per evaluation**, as the basin adapter calls `backward()` without
  `retain_graph`.

`get_par_swap_rate` and `set_fixed_amount` stay severed, their upstream being the curve family's
quotes. A `Volatility_Delta` re-strike runs a `brentq`, which carries no derivative, so
`create_market_swaps` refuses `Quote_Sensitivity` with it, before either objective exists. A zero
`Market_Volatility` refuses unless a premium file supplies the target and nothing re-strikes it.
**Held by**
`tests/test_hw2f_analytic.py::test_a_block_the_family_cannot_price_refuses_by_name`; no test holds the
re-strike refusal or the detached model price.

### Gauss–Newton and the dropped terms {#the-dropped-term}

Differentiating $J^\top r = 0$ drops a term on each side ([General](../bootstrapping/general.md);
[@nocedal2006], ch. 10); the backward contracts $w = (J^\top J)^{+} v$, then $\partial L/\partial q =
-(\partial r/\partial q)^\top Jw$. On the Monte Carlo residual, $r_i = w_i f_i^2$ makes each dropped
term half of what it corrects,

$$\textstyle\sum_i r_i\nabla^2_\theta r_i = \tfrac12 J^\top J + O(f^3),
\qquad
\sum_i r_i\,\partial^2_{\theta q} r_i = \tfrac12 J^\top \partial_q r + O(f^3),$$

so the true system is $\tfrac32$ the Gauss–Newton one on both sides and the contraction is exact to
leading order. On the analytic residual the cross term is [absent](#the-analytic-quote-side) and the
other is $O(\|r\|)$.

!!! warning "Correcting one side makes a factor 3/2"
    A reference adding the Hessian term alone is $\tfrac23$ of the truth. The cross term needs a
    finite difference of $J$ in the authored quote: the splice detaches the model price, so autograd
    reports the mixed derivative as exactly zero.

Both Jacobians come through `autograd.grad`, never `.grad`: basin hopping calls `backward()` every
evaluation, so a quote leaf's `.grad` is a plausible path sum. `bootstrap` clears it before publishing.

### Stationarity is checked {#stationarity-checked}

`solve` may return the seed, so the backward raises, naming the norm, when $\|J^\top r\|$ over the free
coordinates exceeds `Stationarity_Tol`. The norm is in the objective's units, hence a block field:

| family | `Stationarity_Tol` | `Jacobian_Rcond` |
| --- | --- | --- |
| HW2F | 1e-3 | 1e-5 |
| LogVar2FJ | 1e-4 | 1e-3 |

The analytic chain reaches the default; the Monte Carlo residual is pre-squared, so its objective is a
quartic, flat enough near its minimum that scipy stops orders above it. LogVar2FJ also refuses
under `Fit_Mode: Bootstrap` (θ\* stationary for no single objective) and after a stage capped at
`Max_Iterations`. **Held by**
`tests/test_warm_starts.py::test_the_warm_hull_white_fit_reprices_a_moved_ladder_without_the_search`
(stationary under the default, cold and warm); no test holds the refusal.

### Rank deficiency, the pseudo-inverse and the box {#rank-deficiency}

On HW2F's default knots $J$ has 23 columns, so four swaptions leave a 19-dimensional null space. The
inverse is the pseudo-inverse [@penrose1955], truncated [@hansen1987] at `Jacobian_Rcond`, and
$d\theta/dq$ along a null direction is the minimum-norm representative; no ridge, which would return
the derivative of another problem. The cutoff is on the singular values of the column-scaled
$J/\|J_{:,j}\|$, the matrix `x_scale='jac'` steps on (LogVar2FJ; the swaption solve is unscaled),
because column norms span orders and minimum-norm needs a metric that is not the parameters' units.
That being a convention, the backward logs each delta's share in the discarded null space. **Held by**
`tests/test_hw2f_analytic.py::test_the_four_quote_job_document_pins_theta_and_its_quote_deltas`: θ\*
and four deltas on `tests/fixtures/hw2f_four_quote_job.json` to 1e-6, red with the unscaled
pseudo-inverse (fourth delta 0.2667 against 0.3014) or `.grad` left on the leaves.

**The box is part of the fixed point.** At θ\* the KKT condition holds, so the contraction runs over
the coordinates `utils.active_set` leaves free. A coordinate is held within 1e-4 of the box width of a
bound **and** with $(J^\top r)_j$ pushing into it — a solver stops a hair short of a floor it is jammed
on, and one resting on a bound the data pulls away from is free. The swaption solve logs every held
bound, a calibration statement. No test holds the active set.

### The analytic objective {#the-analytic-quote-side}

The default differences normal vols: $r_j = w_j\big(\sigma^{SP}_j(\theta) - \sigma^N_j(q_j)\big)$, with
$\sigma^N_j = P_j\sqrt{2\pi/T_{0,j}}/A_j$ the ATM Bachelier inversion [@bachelier1900]
(`market_normal_vol`). Nothing is detached: the annuity comes off the t=0 curve in numpy
(`swaption_legs`), so it carries no θ. The residual is separable, $\partial r/\partial q$ diagonal and
$\partial^2 r/\partial\theta\,\partial q \equiv 0$: one dropped term absent, the other $O(\|r\|)$.

### The re-solve reference is ill-posed {#the-manifold-finding}

Bumping a quote, re-calibrating and differencing θ\* has no limit here: on four quotes θ\* lives on a
19-dimensional manifold, a re-solve lands elsewhere on it, and the quotient grows as the bump shrinks.
The value chain is not what fails — the factor greek contracted with a re-solve's actual displacement
reproduces the value it moved. **Bump-and-recalibrate is ill-posed for the HW2F calibration**: the
solve wanders in the directions the objective is flat in, the ones a derivative is refused along.

### The direction check {#the-identified-fixture}

Full rank does not rescue it: a 25-quote grid against 23 parameters (not in the repository) has no
manifold, and its re-solves, stationary on the analytic objective, still wander along near-flat
directions. The reference is the direction the quotes identify: **step θ by $d\theta/dq \cdot h$ and
re-price without re-solving**, which closes on one linearly in $h$ and negates under a sign flip.
[Quote Propagation](quote_propagation.md#value-space) scores the vol families so. No test re-runs it.

## The closed-form map {#the-closed-form-map}

`GBMAssetPriceTSModelParameters` turns a surface's ATM column into the instantaneous vol curve the GBM
process integrates, one quadratic per step between expiries
([Equity and FX](../bootstrapping/fx_and_equity.md)); nothing is fitted. The walk runs on the quotes'
own tape, float64 on the CPU, so under `Quote_Sensitivity` the curve it writes is the connected
tensor and autograd walks the map.

### The repair is a kink {#the-repair-kink}

A step adding less variance than its one-day ramp, in practice a falling total variance, has no real
root, so $V(t_i)$ is floored and the step's vol is zero. $df_i/dq_i$ is 0 there; autograd reports the
branch's one-sided derivative, and a bump straddling the switch converges to half of it. The quadratic
is written with $c = \text{floor} - V(t_i)$ so the floored root is exactly zero.

!!! danger "The discriminant is guarded"
    A second consecutive repair meets $b = c = 0$: $\sqrt{}$'s infinite derivative times the zero
    $d(b^2)/db$ NaNs every Jacobian entry once a later step reads that σ. So the root at a zero
    discriminant is written as zero and `sqrt` never sees it.

**Held by** `tests/test_gbm_term_structure.py` for the values: every column whose total variance
rises reprices, and a falling one repairs to zero. No test holds the map's quote side.

### Two quote sources {#the-atm-column}

The leaves are the ATM column either way; `atm_column` picks the source by provenance. Where an
`FXVolPrices` block wrote the surface, its ATM rows are the surface's ATM vols by construction —
`malz_skew` puts them at the delta-neutral straddle — so they are taken as quoted rather than read back
off the grid. Otherwise the column is the surface at moneyness 1.

!!! warning "Provenance is evidence, not a name"
    A hand-authored surface can share a quote block's name. The check is the fingerprint
    [`pinned_grid`](market_prices.md#fxvolprices) reads back, the `Malz` type beside its
    `Grid_Tolerance`; a written surface whose quotes left its expiries raises naming both sets.

## The delta solve {#the-delta-solve}

`bootstrappers.FXVol` turns ATM, RR and BF into a log-moneyness surface — the strangle pair
`ATM + BF ± RR/2` (its twin bit-identical), the wing pair, a bisection per pinned node ([Equity and
FX](../bootstrapping/fx_and_equity.md)) — so `dV/d(RR)` and `dV/d(BF)` exist.

### The tape boundary {#the-tape-boundary}

A tape runs straight through `Factor2D.malz_delta`'s 64 halvings and reports the wrong number: a
bisection's iterates are dyadic combinations of the bracket endpoints, so it carries their
derivative, not the root's. So **the tape starts at the converged root**, one Newton step off it in
`carried_sigma`:

$$\delta = \delta^* - \frac{R(\delta^*, q)}{\texttt{detach}(\partial R/\partial\delta)}
\;\Longrightarrow\;
\frac{d\delta}{dq} = -\Big(\frac{\partial R}{\partial\delta}\Big)^{-1}\frac{\partial R}{\partial q} .$$

δ\* being the root makes it the theorem, not the `detach`: $R(\delta^*)$ is a rounding that multiplies
away the slope's graph. `R` mirrors the value path's residual with `utils.norm_cdf` for
`scipy.special.ndtr`, so the splice keeps the written surface the numpy one. The grid is pinned, not
taped, so a tick is a values patch; every discrete decision — ordering, label, wing, bracket,
segment — is read off the value path, a permutation having no derivative.

!!! warning "`delta_atm` is taped"
    The ATM quote sets where the wing grid's ATM nodes sit, $\tfrac12 e^{-\sigma_{ATM}^2T/2}$, so the
    knot positions are a function of a quote; a twin freezing them is silently wrong.

### Four discrete choices {#the-four-choices}

The wing ($x = \sigma_{ATM}^2T/2$), the bracket (the root reaching the endpoint) and the segment (δ\*
crossing a pillar) are kinks: autograd reports the branch's one-sided derivative, and a difference
straddling one converges to the two sides' average, nobody's. The clamp endpoint — where the two endpoint residuals swap which
is smaller — is a jump: no derivative exists there, and either one-sided answer is right. A ladder
fingerprints each node by all four (the endpoint on clamped nodes only) and drops a rung where one
changes.

### The clamp is flat extrapolation {#the-clamp}

A node with no root in its bracket takes the nearer endpoint knot's vol, the smile being flat beyond
its widest quote, so its row is that knot's algebra: 1 in ATM, 1 in BF, ±½ in RR, zero elsewhere.
Assert it as a set: walking non-zero entries never sees a missing column. `dσ/d(ATM) = 1` also marks a
bracketed node on a flat segment.

!!! danger "An ATM-only smile divides zero by zero, only backward"
    Each wing is one knot of zero span; dividing before selecting NaNs the Jacobian of a good flat
    surface. A double `where` guards it, and the Jacobian is the expiry indicator.

### The Jacobian's structure {#the-jacobian-structure}

$J$ is block diagonal in expiry; the RR column is ±½ the BF column wing by wing, exactly; the ATM
column is near one but not on it, as `delta_atm` moves — exactly one is the frozen-`delta_atm`
mutation. **Held by** `tests/test_structures.py::test_each_pillars_charge_is_what_repricing_that_pillar_costs`:
per pillar `dV/dq` against a re-bootstrapped central difference within 2%, signs included. The kinks,
clamp rows and guards are held by no test.

## The attachment {#the-attachment}

`Calculation.factor_leaf` mints every leaf — the three branches of `CreditMonteCarlo._build_factor_state`
and `BaseValuation.update_factors` — and offers a factor with a connected θ\* as
$\text{leaf} + (\theta^* - \texttt{detach}(\theta^*))$, the [boundary
correction](calc_lifecycle.md#boundary-corrections-the-sensitivity-subsystem)'s shape: `retain_grad` on
the sum keeps the factor greek and lands `dV/dq` on the quote leaves in the same pass. A model
attaches per named parameter, split by `utils.Residual.split`, the one place the flat vector comes
apart, so a leaf cannot take the wrong slice. A non-zero `Tenor_Offset` declines, a shifted curve
being another curve. A credit Monte Carlo needs `Gradient_Variables` `Implied` or `All` for a model's
parameters, and reuses `implied_leaves` in its static branch so a gradient cannot split ([the
invariant](calc_lifecycle.md#compile-phase-2-_build_factor_state)).

**The harvest removes as well as adds.** `Config.bootstrap` copies each family's tensors onto the
config, not into JSON `Price Factors`. A whole run first drops the family's keys, or a block flipped
to `No` would leave yesterday's tensor reported as today's; a narrowed run replaces what it covered.
No test flips the switch.

**Two leaf shapes**, `(descriptors, leaves)` either way: HW2F a tuple of scalar leaves, the others one
vector. FX descriptors name the pillar (`RR 0.25 1`).

!!! danger "Descriptors collide across families; the truth is the sum"
    An `FXVolPrices` ATM quote is also the GBM family's ATM column, so with both blocks on, its
    `dV/dq` splits across two leaves under one descriptor. A reader groups by descriptor and sums, as
    `structures.vol_risk` does; `service.quote_rows` files the halves under their two blocks.

## The precision seam {#the-precision-seam}

- **The curve solve is float64 on the host** (`BenchmarkInstruments.dtype`; `construct_bootstrapper`'s
  float32 does not reach it): float32 cannot converge to 1e-10, and the solve is dispatch-bound. The
  GBM and FX tapes are float64 too. **Held by**
  `tests/test_interest_rate_prices.py::test_the_bootstrap_recovers_the_curve_its_quotes_came_from`.
- **θ\* enters the simulation by one differentiable `.to()`** in `factor_leaf`; the splice, taken
  after it, is exactly zero in the simulation's dtype.
- **The swaption residual runs in the bootstrapper's dtype**, float32 by default, because the Monte
  Carlo objective prices the path set thousands of times. The backward promotes $J$ to float64 for the
  pseudo-inverse, contraction and norm, and casts `grad_outputs` back, so what it reports has the
  residual's resolution. The LogVar2FJ family declares double.
- **The LogVar2FJ walk computes every nonlinear function of grid and parameters in double**
  (`utils.LogVar2FJ.wide`) and casts once; its step weight is `−expm1(−2κδ)/(2κ)`, since
  $e^{2\kappa T}$ overflows float32 past 7.4 years at κ = 6.

## The validation triangle {#the-validation-triangle}

Each corner independent of the machinery it checks:

| family | check | test |
| --- | --- | --- |
| curves | round trip, 1e-10 | `test_interest_rate_prices.py::test_the_bootstrap_recovers_the_curve_its_quotes_came_from` |
| curves | $d\theta/dq$ against re-bootstraps, 1e-8 | `test_quote_propagation.py::test_the_artifact_holds_the_calibration_jacobian_the_solve_would_have_differenced` |
| curves | `dV/dq` against a refitted book, 1e-6 | `test_service.py::test_the_consolidated_risk_is_read_on_the_quotes_the_curves_are_built_from` |
| curves | `dV/dq · dq` against the ride, first order | `test_quote_propagation.py::test_the_quote_delta_predicts_the_value_the_ride_reprices` |
| credit | round trip, 1e-10 | `test_credit_prices.py::test_the_bootstrap_recovers_the_survival_curve_its_quotes_came_from` |
| credit | par spreads against the ISDA standard model, 5e-3 | `test_credit_prices.py::test_the_standard_model_reprices_the_solved_curves_quotes_to_its_conventions` |
| credit | $d\theta/dq$ against re-bootstraps, 1e-8 | `test_credit_prices.py::test_the_quote_derivative_is_the_re_bootstrapped_difference` |
| credit | `dV/dq` against the chain rule and a refitted deal, 1e-8 | `test_credit_prices.py::test_a_default_swap_on_the_name_reports_its_quote_delta_beside_its_factor_greek` |
| credit | a swap quote's `dV/dq` against both curves refitted, 1e-6 | `test_credit_prices.py::test_a_default_swaps_delta_to_a_swap_quote_carries_the_survival_curves_re_solve` |
| inflation | round trip, 1e-10; the closed form where it is the root | `test_inflation_prices.py::test_the_bootstrap_recovers_the_inflation_curve_its_breakevens_came_from` |
| inflation | $d\theta/dq$ against re-bootstraps, 1e-8 | `test_inflation_prices.py::test_the_quote_derivative_is_the_re_bootstrapped_difference` |
| inflation | `dV/dq` against the chain rule and a refitted linker, 1e-8 | `test_inflation_prices.py::test_a_linker_reports_its_quote_delta_beside_its_factor_greek` |
| inflation | a swap quote's `dV/dq` against both curves refitted, 1e-6 | `test_inflation_prices.py::test_a_linkers_delta_to_a_swap_quote_is_both_curves_refitted` |
| HW2F | θ\* and deltas pinned, 1e-6 | `test_hw2f_analytic.py::test_the_four_quote_job_document_pins_theta_and_its_quote_deltas` |
| LogVar2FJ | factor hex-identical, switch on or off | `test_logvar2fj_json.py::test_the_quote_switch_writes_the_same_factor_and_publishes_its_leaves` |
| FX vol | per-pillar `dV/dq`, 2% | `test_structures.py::test_each_pillars_charge_is_what_repricing_that_pillar_costs` |
| GBM | none | — |

No test holds the curve θ\* bit-identical on and off, which `lifecycle_fields` assumes.

**The self-delta identity.** A benchmark is at par for every q, so
$\partial_\theta PV_i \cdot d\theta/dq_j = -\partial_{q_i}PV_i\,\delta_{ij}$: the benchmarks priced as
ordinary deals report a diagonal quote-delta matrix, the identity once each row is divided by its own
quote sensitivity — a secant, PV being affine in the quote, so nothing is reused. Under a
pseudo-inverse it is the projector onto the kept directions. No test holds it.

## Open {#open}

- The GBM family reads a wing as a hand-authored `Malz` surface's ATM column: moneyness 1 is a ratio
  surface's ATM, a `Malz` axis is $\log(F/K)$. A surface the FX family wrote takes the [quote
  path](#the-atm-column).
- `Tenor_Offset` declines an FX surface's attachment though `Factor2D.current_value()` ignores the
  offset, dropping a delta for a surface that did not move.
- A model-priced leg's charge reads a lognormal vega; a consolidated risk read refits its blocks
  ([limitations](../limitations.md#the-engine)).

## Non-goals {#non-goals}

Listed on the [roadmap](roadmap.md#designed-not-built).

- **No aggregation off these deltas**; the `SIMM` calculation bumps and revalues on its own, a
  moved quote riding its curve's [calibration](quote_propagation.md#stateless). No wrong-way risk.
- **No recalibration inside the simulation**; quote sensitivities are t0 risk.
- **No new pricers**, and no pricer changes for the t0 closure — so far none.
- **No SABR or SSVI**; a Malz smile is the one delta parameterisation, and a swaption quote is its
  row's number or a premium.
- **No reporting format in the engine**: `dV/dq` is `.grad` on `Config.quote_leaves`, a quote being no
  factor for `make_factor_index`; `service.quote_rows` reads [both shapes](#the-attachment).
- **No second differentiation**: neither backward supports `create_graph`, and `LeastSquaresSolve`
  refuses it, Gauss–Newton carrying no second derivative.
- **No differentiable x-grid**: one following its quotes would make every tick a recompile, which
  [pinning](market_prices.md#fxvolprices) prevents.
