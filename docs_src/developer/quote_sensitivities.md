# Quote Sensitivities

The autograd tape used to start at *calibrated* factors, so a greek came out in zero-curve-node or
model-parameter space. Desks explain P&L in **quote** space — par swap rates, FRA strips, OIS quotes,
swaption vols. This workstream extends the tape one layer upstream by owning the calibration inside
the library and differentiating through its fixed point, so one backward pass yields `dV/dq` beside
`dV/dθ`.

```
q (leaves)  ->  calibration solve  ->  θ*  ->  factor buffers  ->  scenarios  ->  V
```

Only the middle arrow is new; everything downstream of θ\* is the existing engine. All four
increments are built, and the four calibrations put four different things in that arrow — the page
reads in that order:

| increment | family | what is in the arrow |
| --- | --- | --- |
| 1 | `InterestRatePrices` | a damped Newton on a root, differentiated by the [IFT](#the-ift-contract) |
| 2 | `HullWhite2FactorModelPrices` | least squares on a [stationarity point](#the-stationarity-contract); `Objective` picks the residual, `Analytic` the default since 2026-08-31 |
| 3 | `GBMAssetPriceTSModelPrices` | [no solve at all](#the-closed-form-map) — an explicit map |
| 4 | `FXVolPrices` | [a bisection per node the tape refuses to enter](#the-delta-solve) |

So a vol tick arrives as ATM / RR / BF, becomes a log-moneyness surface on a pinned grid, integrates
into a GBM vol curve and reaches `V`, and one backward pass reports `dV/dq` at every coordinate on
the way. `Monte_Carlo` remains fully supported as the engine's own estimator and as this page's
oracle: every comparison here is taken against it.

## The residual is priced by the engine's own pricers {#the-residual}

A benchmark is an ordinary deal. A quote **names an instrument type and carries a block of it** — see
[Market Prices](market_prices.md#a-quote) — so the residual is a vector of the library's own pricing
functions,

$$F_i(\theta, q_i) = PV_i(\theta, q_i) \big|_{t_0}$$

evaluated so a fair instrument prices to zero. `bootstrappers.BenchmarkInstruments` is that
evaluation: it compiles the benchmark deal-tree nodes once — discovery, factor objects, tenor
payloads, each leaf's `Factor_dep` / `Time_dep` — and prices them at t0 from a `{Factor: tensor}` of
curve nodes, returning one PV per benchmark as a `torch` vector whose graph reaches those tensors.

A benchmark is a **node**, not a deal, so a container with `Children` is one benchmark: a deposit and
an FRA are single deals, a par swap is one `SwapInterestDeal`, an OIS swap is a container over an
OIS-compounded floating leg and a fixed leg. The node's PV is the sum of its leaves', each converted
to the reporting currency by its own `pv_*_leg`. No netting or collateral rule applies on top, which
keeps this out of `DealStructure`.

The reporting currency **is** the curve's currency, so a rate benchmark crosses nothing and every
`calc_fx_cross` is the identity. A forward outright is the exception: its other leg discounts on that
currency's own solved curve and converts at a spot the residual reads as a **detached constant**.

## The graph audit {#the-graph-audit}

The factor-construction path severs autograd in four places. A severance here does not raise and does
not change a value — it silently reports a zero gradient — so finding them is the point of increment
1 rather than incidental to it.

| Where | What severs | How the closure avoids it |
| --- | --- | --- |
| `Calculation._build_factor_state`, `Base_Revaluation.update_factors` | every leaf minted as `torch.tensor(factor.current_value(...))` out of a **numpy array** — the only way a curve becomes a tensor on the ordinary path | θ is written straight into `t_Static_Buffer`, where the pricers read a static curve; `current_value` is never called for a curve being solved |
| `riskfactors.Factor1D.current_value` | numpy end to end; handed a tensor it still returns numpy | not on the closure's path. Still used for the **constants** — every factor the solve is not solving for — where a detached leaf is right |
| `Factor1D.__init__` → `check_interpolation` | precomputes the Hermite `(g, c)` pair from the numpy rate column | the pricing path does not read it. `utils.Interpolation.build` re-derives the pair from the buffer **tensor**, and `all_tenors` carries only the kind and the tenor grid. The per-kind gate was culled with the closure suite, so this stands on the code alone |
| `utils.TensorSchedule.bind` | copies the schedule with `new_tensor` — notionals, accruals, margins **and the fixed rate**. Where the **quote** stopped being differentiable | closed by `TensorSchedule.carry`; see [the quote side](#the-quote-side). θ never passed through this seam either way |

Two traps that are not severances and would each be silent:

- **`t_Buffer` is a memo keyed by `(stoch, Factor)` and a time hash, not by tensor identity**, so a
  state reused across two θ answers the second call with the first's discount factors.
  `Benchmark_State` is built fresh per evaluation. The gate went with the closure suite.
- **`utils.CurveTenor` caches its tenor grid from the first tensor that queries it.** `all_tenors` is
  rebuilt per closure instance, so a float64 solve cannot inherit a float32 grid.

## The quote side {#the-quote-side}

A schedule is a **dual**: `.np` for the index columns (everything `np.unique` and `searchsorted` are
asked about) and a device copy for the arithmetic. That split stays. `TensorSchedule.carry` attaches an
optional `{column: tensor}` overlay to the **tensor half only**, spliced into the copy `bind` makes:
absent by default, and a compile-time edit that raises once the copy exists — see [the schedule
lifecycle](calc_lifecycle.md#the-schedule-lifecycle). `BenchmarkInstruments` binds its own schedules
last, once every overlay is on.

Only **value** columns are overlaid, measured rather than assumed: across the four schedule-carried
benchmark shapes at three quotes, exactly one column moves — `CASHFLOW_INDEX_FixedRate`/`FloatMargin`
(the same index 6) on the deposit, the FRA and both fixed legs — and moves *linearly*, second
difference exactly zero. `_carry_quotes` raises if a reset column ever moves, because a reset value
also leaves through `known_resets`, which reads numpy. The fifth quotable shape, an `FXForwardDeal`
outright, moves no column at all — its quote lands in `Buy_Amount` — so `_carry_quotes` refuses it by
name rather than report a zero `dV/dq` row (`test_a_forward_block_refuses_quote_sensitivity`).

`BenchmarkInstruments` therefore takes the benchmark set **twice**, at its quotes and one percent
higher. The difference is the exact ∂(schedule)/∂q, so
[`QUOTE_WRITERS`](market_prices.md#interestrateprices) stays the only place a quotable instrument
declares where its number goes. The splice is

$$\text{column} = \text{base} + \big(q - \texttt{detach}(q)\big)\,\frac{\partial \text{column}}{\partial q}$$

— the [boundary correction](calc_lifecycle.md#boundary-corrections-the-sensitivity-subsystem)'s shape,
for its reason: worth **exactly zero** in the forward pass, derivative one. The tensor half is
bit-identical to the plain copy (`np.array_equal`, not a tolerance), so enabling quote gradients
cannot move a PV.

!!! warning "The overlay is a derivative carrier, not a reparameterisation"
    A moved quote needs a **fresh closure**: the splice is worth zero forward, so a schedule's value
    columns do not follow a quote that moves — only the derivative does. `pv_fixed_cashflows`' payment
    tensor is memoized beside the tensor half in `derived`, which `bind` mints and re-mints with it,
    so the memo cannot outlive the copy it was built from.

## The solve {#the-solve}

`bootstrappers.damped_newton` is the plain solver — the one the IFT wrapper runs unchanged in its
forward pass. The curves being solved are flattened into **one** system, so a projection curve and
the discount curve it prices against are a single Jacobian. That Jacobian is autograd on the
residual: one backward pass per benchmark gives a whole row, no bump loop over the knots, and it is
the same derivative the IFT needs on the other side — *written once, differentiated twice*.

Damping is a backtracking line search on the residual's max-norm: full step first, halved until it
decreases. The three knobs are **declared fields of the block being solved**, not module constants,
each read with the declared default as its `.get` fallback (a gate holds the two together per family).

| Field | Default | |
| --- | --- | --- |
| `N_Iter` | 50 | a par-rate seed is within a few basis points, so a well-posed strip converges in single digits. Reaching the cap raises rather than returning a half-solved curve |
| `Tol` | 1e-14 | a zero rate is O(1e-2), so ~1e-12 relative — inside the 1e-10 a round trip asks for, and where the linear solve's rounding stops the iteration improving |
| `Damping_Halvings` | 6 | below that the step *length* is not what is wrong, so the solve says so rather than creeping |

Convergence is tested on the **step, before the line search**. A step inside the linear solve's own
rounding cannot be asked to reduce a residual already at noise level. The seed is each quote itself,
seeded into `Price Factors` before the closure is built, because the closure constructs the curve
factor out of that section.

!!! note "Honest negative result — the damping never engages here"
    On both round-trip worlds the line search takes the full step every iteration: zero halvings
    across all three solves (USD OIS 4 iterations, USD projection 4, ZAR 5), and removing the damping
    leaves every gate green. It is insurance against a first iterate these fixtures do not produce.

## The knot rule {#the-knot-rule}

ONE knot per used quote, at that benchmark's last cashflow date, in the block's `Day_Count`, and the
output grid is that grid. The reasoning is on [Market
Prices](market_prices.md#interestrateprices); what matters here is that the knot grid is what the
Jacobian is square in, so the quote-delta buckets a desk reads are the benchmark maturities.

## Curve contracts the solve relies on {#curve-contracts}

**No knot at tenor zero.** A curve never carries a 0.0 knot: rates start at T+1, and a rate at tenor
zero is redundant anyway since its discount factor is 1 by identity. `Factor1D.interpolate` divides
by the tenor for the rate-times-time kinds, so a zero knot yields NaN — the contract asserting itself.
The knot rule complies by construction: every benchmark's last cashflow is strictly after t0.

**The compounding leg is a compile-time SHAPE, not a pricer branch.** `pv_float_cashflow_list` routes
an accrual period through geometric compounding when the reset count differs from the cashflow count
(`all_resets.shape[1] != reset_cashflows.np.shape[0]`) — daily resets against quarterly cashflows, the
reshape set up at `calculate_dependencies` by `compress_no_compounding(groupsize=-1)` under
`Compounding_Method='OIS'`. The regular route's `Weight = 1/n` resets are the AVERAGING legs'
arithmetic and must never reach the compounding path. This is why an OIS benchmark is authored as a
floating list with `Compounding_Method='OIS'` (under a `StructuredDeal` for the par swap) and not as a
`SwapInterestDeal`, whose generated legs never pass through the compression. **The shape-difference
check is acknowledged tech debt** — it works, and it is subtle enough to be written down.

## The IFT contract {#the-ift-contract}

`bootstrappers.CalibrationSolve` is the bootstrap as one differentiable node.

**Forward is the ordinary solve.** It calls `damped_newton` and nothing else — same iterations, same
tolerances, same float64 — so enabling quote gradients cannot move a mark by construction. Autograd
runs `forward` with grad mode off and the solve needs it on for its own Jacobian, so it is re-enabled
inside and the iteration's graph dies with the iteration. The block goes through the wrapper whether
or not a quote is on the tape; with `quotes=None` it is a pass-through.

**Backward is the implicit function theorem**, never an unrolled solver. At $F(\theta^*, q) = 0$,
$d\theta/dq = -(\partial F/\partial\theta)^{-1}\partial F/\partial q$, so an incoming cotangent
$v = \partial L/\partial\theta^*$ contracts in two steps:

$$\Big(\frac{\partial F}{\partial \theta}\Big)^{\!\top} w = v
\qquad\text{then}\qquad
\frac{\partial L}{\partial q} = -\Big(\frac{\partial F}{\partial q}\Big)^{\!\top} w$$

Both derivatives come from **autograd on the residual closure itself**, evaluated once at
$(\theta^*, q)$: `residual_jacobians` takes a row of each out of one backward pass per benchmark,
because the $q$ side is another output of the pass the $\theta$ side already needs. $n$ is the
benchmark count, so the linear solve is small and the [knot rule](#the-knot-rule) makes it square. The
Jacobian is recomputed at $\theta^*$ rather than reused from the last Newton step, which was taken at
the iterate *before* it; that costs one iteration's work.

!!! warning "Every `grad` in the backward retains the graph"
    The residual's subgraph is **shared with the forward pass** — `pv_fixed_cashflows` memoizes its
    payment tensor on the schedule — so freeing it in the backward would take the forward's graph too.

!!! note "The same two pieces are also an OPERATOR"
    Solve every cotangent at once and $d\theta/dq$ comes out whole, which is what lets a quote that
    moves between bootstraps carry the curve with it rather than trigger a re-solve. That lifecycle is
    [Quote Propagation](quote_propagation.md), built for this family only, and its unit is the
    **coupled set** rather than the block.

## The stationarity contract — increment 2 {#the-stationarity-contract}

`bootstrappers.LeastSquaresSolve` is the HW2F swaption calibration as one differentiable node: the
contract above with one word changed, the fixed point being the **stationarity** of a least-squares
loss rather than the root of a residual. Everything below is what that word costs.

**Forward is the ordinary solve.** `SwaptionCalibration.solve` runs the chain the bootstrap always ran
— basin hopping, then least squares, `x0` chained, a candidate accepted only if it beats the running
best *and* the process it implies is well posed. The seed is the block's, not the process's:
`Random_Seed` is a declared field and one generator serves both the step taker and the Metropolis
test, which used to draw from the numpy **global** RNG — θ\* moves 0.93 absolute between ambient
seeds on the four-quote fixture. This is therefore the first reproducible HW2F calibration and there
is no earlier baseline to preserve.

`HullWhite2FactorModelParameters.Objective` picks the residual. **`Analytic` is the declared default
since 2026-08-31**, on the four readings in the [Model punchlist](roadmap.md#model-punchlist);
`Monte_Carlo` is the estimator and this page's oracle, and an analytic solve ends by repricing θ\*
through it and logging what it makes of the answer. Everything from here to [the
analytic section](#the-analytic-quote-side) is the Monte Carlo path, unchanged to the bit.

### The loss is a Monte Carlo, and the derivative is conditional on the draw {#the-mc-loss}

There is no par instrument to drive to zero. Each benchmark is a swaption priced by brute-force Monte
Carlo through the engine's own `pv_float_cashflow_list`, and the residual is the weighted squared
relative pricing error against the market premium,

$$r_i(\theta, q_i) = w_i\Big(100\big(\tfrac{P_i(q_i)}{M_i(\theta)} - 1\big)\Big)^2$$

with $M_i$ the model price. Two consequences run through everything below. The residual is **already
squared**, so `least_squares` minimises a quartic in the pricing error and $J = \partial r/\partial
\theta$ carries a factor of that error in every row: at an exact fit $J$ is not small but **zero** and
Gauss–Newton degenerates. And $r$ cannot reach zero anyway on a block quoting more swaptions than the
model has parameters — which is the block this was validated on.

The Sobol engine is seeded once on the state the closure builds, and `reset` clears `t_Buffer` and
`t_PreCalc` (the memo trap) rather than re-drawing. So every evaluation, every Jacobian and the
backward price the **same paths**, and `dθ/dq` is the derivative *conditional on that draw* — the
same philosophy as a pinned `process_ofs`: the number reported is the derivative of the number
reported, and a re-drawn sample would have the optimizer differencing the noise.

### What the quote side closes, and what stays severed {#the-swaption-quote-side}

The severance is at the **market premium** and nowhere else: `create_market_swaps` prices it with
`utils.black_european_option_price`, scipy end to end, so it reaches the residual as a numpy scalar.
The repair is the splice this page uses three times, on `market_swap_class.error`:

$$\text{error} = w\big(100(\tfrac{P}{M}-1)\big)^2
+ \big(\widetilde{\text{error}} - \texttt{detach}(\widetilde{\text{error}})\big)$$

with $\widetilde{\text{error}}$ the same expression off a float64 **twin** of the premium —
`utils.black_european_option`, the engine's own tensor Black, which is what the cap/floor and swaption
pricers value an option with, and bit-identical to the numpy one at the money to 1e-12.

Two details in that line are load-bearing and neither is visible to a price gate:

- **The model price is detached in the carried half, and only there.** Left attached, the carried half
  reaches the *model parameters* as well as the quote and the calibration Jacobian comes out
  **doubled** — the residual stays bit-identical and the optimizer simply walks a different path. The
  quote derivative of the error does not involve the model's own sensitivity.
- **The premium is a callable rebuilt per evaluation, not a compiled subgraph.**
  `make_basin_hopping_loss` calls `total_loss.backward()` with no `retain_graph`, so a subgraph
  compiled once with the benchmark set would be freed by the first evaluation and every one after it
  would raise. Rebuilding costs one scalar Black per benchmark.

Two severances stay **open on purpose**, because their upstream is not a quote of this calibration:
`get_par_swap_rate` prices the strike and the pvbp in numpy off the zero curve, and `set_fixed_amount`
writes that strike into the schedule's numpy half. Both are the calibrated curve — increment 1's
quote. There were three until 2026-09-01, when a zero `Market_Volatility` stopped falling through to
the surface's own ATM read and began refusing by name.

!!! warning "A premium re-struck by `Volatility_Delta` declines the quote side — on either objective"
    That path recovers an implied vol with a `brentq` root find and re-strikes the premium off it, and
    a numerical root find carries no derivative. Reporting zero there is the exact failure this
    workstream exists to prevent, so `Quote_Sensitivity` **raises**. The refusal is in
    `create_market_swaps`, which runs before either objective exists, so `Analytic` inherits it word
    for word rather than declaring a second one.

### Gauss–Newton, and the two terms it drops {#the-dropped-term}

$r(\theta^*, q) \neq 0$ and never will be, so what is held fixed is the gradient of half the sum of
squares, $g = J^\top r = 0$. Differentiating that gives a dropped term on **each** side:

$$\Big(J^\top J + \underbrace{\sum_i r_i \nabla^2_\theta r_i}_{\text{dropped}}\Big)\frac{d\theta}{dq}
= -\Big(J^\top \frac{\partial r}{\partial q} + \underbrace{\sum_i r_i \frac{\partial^2 r_i}{\partial\theta\,\partial q}}_{\text{dropped}}\Big)$$

A cotangent $v$ contracts as $w = (J^\top J)^{+} v$ then
$\partial L/\partial q = -(\partial r/\partial q)^\top (Jw)$, with both derivatives from autograd on
one fresh evaluation at $(\theta^*, q)$ — functionally, through `autograd.grad` and never off `.grad`.
The scipy adapters clear `.grad` per evaluation while the quote leaves accumulate across them, so a
harvested `.grad` is the sum over the optimizer's whole path: on the gate fixture, numbers six orders
out and one `NaN`.

**Neither dropped term is second-order small here.** The textbook argument assumes $r$ is the pricing
error; on this block it is the pricing error **squared**, $r_i = w_i f_i^2$, so each dropped term is
**half** what it corrects at any residual level. **The two halves cancel**: squaring a residual
row-scales $J$ and $\partial r/\partial q$ by the same $\operatorname{diag}(2w_i f_i)$ and the normal
equations are invariant under that, so the true system *is* the Gauss–Newton system and
`LeastSquaresSolve.backward` reports the **exact leading-order derivative**. No correction is owed.

Both halves are measured, because an algebraic cancellation nobody checked is a claim: 0.500064 on the
θ side, 0.4953–0.5115 with cosine 1.000000 on the q side, contracted ratio 0.9908 — in [the identified
fixture](#the-identified-fixture). The *instruments* differ. The θ side is a double backward; the q
side cannot be, because [the splice](#the-swaption-quote-side) detaches the model price in its carried
half, so the closure's mixed second derivative is **structurally zero** and autograd faithfully
reports zero. Measuring it needs a finite difference of $J$ in the **authored** quote at fixed θ\*,
which goes round the splice rather than through it.

!!! warning "Correcting one side is how a factor 3/2 appears"
    Measure the Hessian term, add it, leave the cross term alone, and the reported derivative looks
    3/2 too large — **1.4974** in the top four directions. An artefact of a half-applied correction,
    and the gate's own mutation.

!!! warning "Stationarity is checked, not assumed"
    `solve` accepts whatever the chain returned, which can be the **seed**, and the contraction is
    worthless off the fixed point. So $\|J^\top r\|$ above the declared `Stationarity_Tol` **raises,
    naming the norm**: 3.3e-6 achieved on the four-quote fixture against 2.9e10 for a chain stopped
    after basin hopping. The norm is **absolute and the objective's scale is the block's own**, so a
    tolerance is per block and the identified fixture declares its own.

    **On an over-determined block the Monte Carlo chain does not get there.** It closes most of the
    orders between seed and stationarity — 2.9e11 down to **3.16e2** (8.6e3 before the 2026-09-02
    seed+clock re-mark) — and stops, because a quartic is flat enough near its minimum that the
    relative-improvement test fires long before the gradient does. The analytic objective is the same
    chain over a residual that is not pre-squared and reaches **8.63e-7** on the same block, so this is
    a property of the OBJECTIVE'S SHAPE, measured from both sides. On an under-determined block it
    never shows: the fit interpolates and `‖J'r‖` is small for free. It is why `θ*(q)` is the
    optimizer's stopping point rather than the argmin, and therefore why the classic oracle is
    unavailable.

### Rank deficiency is the problem, not a defect {#rank-deficiency}

$J$ has one row per benchmark and 23 columns — two mean reversions, a correlation, two ten-knot
volatility term structures. A block quoting four swaptions leaves a **19-dimensional null space**:
combinations the quote set does not identify, which no care in the linear algebra can invent.

The inverse is therefore a **pseudo-inverse** at the declared `Jacobian_Rcond`, and `dθ/dq` in a null
direction is the **minimum-norm representative**. No ridge is added: a Tikhonov term returns a
unique-looking number that is the derivative of a different problem.

!!! warning "On an identified block the cutoff is a real decision, not a formality"
    `Jacobian_Rcond` defaults to 1e-8, which on the four-quote block separates four real directions
    from nineteen numerical zeros with four orders of headroom below the smallest real eigenvalue and
    eight above the largest spurious one. An **identified** block has no such gap — 23 real directions
    spanning the conditioning of the swaption grid itself, five orders end to end — and the same cutoff
    keeps 17 of them (18 at the pre-2026-09-02 mark). That is right rather than lossy: the term
    Gauss–Newton drops is the same *size* as the eigenvalues of the last five, so a derivative along
    them would be a derivative of the wrong Hessian. The gate reports the spectrum and how many each
    cutoff keeps.

### The re-solve reference, refuted three times {#the-manifold-finding}

The reference this workstream was briefed to validate against — bump a quote, re-run the calibration,
difference θ\* — is **refuted with evidence**. It is the central result of increment 2 and it is a
negative one.

| refutation | reading |
| --- | --- |
| **θ space, four quotes** | on a 19-dimensional manifold a re-solve lands somewhere else on it: displacement 0.044 at half a percent and 0.065 at a fifth against identified steps of 0.00206 and 0.00082, so the quotient **diverges as 1/h** — cold- and warm-started alike |
| **value space, four quotes** | the CVA change that displacement causes swamps the quote's, by a factor itself unstable in the bump: +10.15 against +4.33 predicted at half a percent, +0.50 against +1.73 at a fifth. On the neighbouring quote the pair **reverses sign** |
| **both, twenty-five quotes** | [the identified fixture](#the-identified-fixture) has `J` 25 × 23 at rank 23 and no manifold, and fails anyway — displacement 0.076 / 0.165 / 0.030 at half, a fifth and a tenth of a vol point, again growing |

**The three have one diagnosis.** The solve wanders in the directions the objective is **flat** in —
on four quotes a true null space, on twenty-five the directions the declared cutoff discards (only a
quarter to seven tenths of the displacement lands inside the subspace the pseudo-inverse keeps).
Either way they are exactly the directions along which a derivative is refused.

Stated plainly: **bump-and-recalibrate P&L explain is ill-posed for this calibration.** The reference
has no limit to converge to, whether or not the quote set identifies the model. All three refutations
are pinned as gates so nobody later "fixes" the derivative against an oracle that is not one; they
pass by *failing* to agree, and if one flips, the comparison has become available. There is a
[fourth](#the-re-solve-fourth-time), on the analytic objective.

What *is* well posed is the direction the quotes **do** identify: step the parameters by
$d\theta/dq \cdot h$ and re-price **without re-solving**. On four quotes that recovers 1.0274 /
1.0121 / 1.0005 / 1.0002 of the predicted move at one percent, a half, a fifth and a tenth, and the
wrong-signed step lands three to four times further out than doing nothing. On the identified fixture
it lands at 1.0382, and the mandated sign-flip mutation turns it into −0.9796. **That check is the
value-space reference**, in place of the ladder.

### The analytic objective's quote side — separable, and that is the whole of it {#the-analytic-quote-side}

`Objective: 'Analytic'` — the declared default, so this is the quote side an undeclared block gets —
differences **normal vols, plain**. There is no second contract: the leaf, the map, the descriptor,
the wrapper, the pseudo-inverse, the stationarity refusal and the `create_graph` refusal are all the
ones above. What is new is the SHAPE of the residual,

$$r_j(\theta, q_j) = w_j\Big(\sigma^{SP}_j(\theta) - \sigma^{mkt}_j(q_j)\Big),
\qquad
\sigma^{mkt}_j = \frac{P_j(q_j)}{A_j}\sqrt{\frac{2\pi}{T_{0,j}}}$$

and every difference below follows from it.

!!! note "The premium construction is convention-aware"
    Since **2026-09-01** `create_market_swaps` reads the surface's declared `Distribution_Type`
    through `Factor3D.get_subtype` and prices the market premium in that convention — Black at
    `K + shift` under `'Lognormal'` (the declared default, so every existing document is
    bit-identical), the general-form Bachelier off an absolute normal vol under `'Normal'` — with the
    float64 tensor twin bound to the *same* pair. The two conventions are **9.7x–11.4x** apart on the
    same numeric ladder, which is 1/F. Under `'Normal'` the inversion below returns the quoted σ_N as
    itself (0.0 to 2.2e-16) and `∂r/∂q` is exactly `−w` at every benchmark, independent of expiry,
    curve and θ; the lognormal diagonal runs −0.10235 … −0.08706, about a tenth, because a lognormal
    vol is a fraction of the forward rather than a rate. (Until 2026-09-02 both carried a spurious
    `√(T_365.25/T_curve)` = 0.99965771 — the premium's 365.25 clock against the inversion's ACT/365;
    the curve's day count won.)

**One severance, one splice, one leaf.** The severance is the scipy market premium, so the repair is
the twin `create_market_swaps` already built (`utils.black_european_option` or
`utils.bachelier_european_option`), carried through the closed-form Bachelier inversion.
`market_swap_class.market_normal_vol` is the whole of it. The splice is worth **exactly zero**
forward: residual, model value, market normal vol, `‖J'r‖` and **θ\* itself** are bit-identical with
the quote side on and off, taken through two whole optimizer chains by `np.array_equal` and hex
comparison rather than a tolerance.

**Nothing is detached, and that is the difference.** `error` divides the twin by the MODEL price and
has to detach it or the Jacobian doubles. `market_normal_vol` divides it by the **annuity**, which
`schrager_pelsser_swaption` builds with `new_tensor` off a numpy curve read, so it carries no
derivative in θ: the market side is a function of $q$ alone and the model side of θ alone.

**The residual is therefore SEPARABLE and the cross term is ABSENT**, not small.
$\partial^2 r/\partial\theta\,\partial q \equiv 0$ exactly, so of the two terms [Gauss–Newton
drops](#the-dropped-term) one is structurally zero and the other is the textbook $O(\|r\|)$ — nothing
pre-squared this residual, so there is no factor of a half and no cancellation to rely on. Measured on
the identified block at its own θ\*, and **around the splice** rather than through it, because a mixed
partial read off the tape is the tape agreeing with itself:

| what | how it was taken | reading |
| --- | --- | --- |
| $\partial r/\partial q$ is DIAGONAL | autograd, off the unstacked residual | **600 of 625** pairs structurally absent; materialised off-diagonal exactly 0.0; diagonal −0.10232 … −0.08704 |
| $\partial^2 r/\partial\theta\,\partial q = 0$ | $J$ across RE-AUTHORED quote rungs, and $\partial r/\partial q$ across θ rungs | `np.array_equal` both ways |
| the annuity is severed | `sp.annuity.requires_grad` | **False** — which is why the two above are exact rather than small |
| the θ-side dropped term | double backward at θ\* | $\|\sum_i r_i\nabla^2 r_i\|_F$ **5.397e-4** against $\|J^\top J\|_F$ **3.604** — a ratio of **1.50e-4** beside a $\|r\|$ of 1.48e-3 |

The Monte Carlo path's θ-side term is 0.500064 of what it corrects at any residual level. This one is
1.50e-4 and **shrinks with the fit** — measured shrinking: 8.75e-4 at the pre-re-mark θ\*, whose fit
was three times worse. Two objectives, one contraction, two different reasons it is the exact
leading-order derivative.

**It runs at the declared `Stationarity_Tol`.** No fixture tolerance is written on this path and none
is allowed: the analytic chain reaches 8.63e-7 on the identified block against the field's own
**1e-3**, where the Monte Carlo path has to declare **1e5** to be differentiated at all.

**The spectrum is this objective's own.** $J$ is 25 × 23 and `Jacobian_Rcond` keeps **15** of the 23
directions against 17 on the squared residual (13 at the pre-re-mark θ\*). Nothing is wrong: σ knots
past the last benchmark expiry are in no swaption's variance integral, and two coordinates of this θ\*
sit ON a bound. Those are directions 25 flat quotes do not identify, and `dθ/dq` along them is the
minimum-norm representative exactly as [rank deficiency](#rank-deficiency) says.

**The triangle closes.** `V` is the four benchmarks priced by the engine's **own Monte Carlo** at θ\*,
so the value chain shares no arithmetic with the residual under test. One backward through
`LeastSquaresSolve` reports `dV/dq`, and three routes reproduce it: the contraction spelled out,
**2.22e-16** relative; the same as $v\cdot d\theta/dq$ with the OPERATOR, **1.088e-14**; the same with
$\partial r/\partial q$ from a RE-AUTHORED central difference, **9.376e-06 / 3.750e-07 / 1.500e-08**
at h = 0.5 / 0.1 / 0.02 vol points — $h^2$ twice. Stepping θ by `dθ/dq · h` and re-pricing without
re-solving closes on 1 from both sides and linearly in h (1.0382 / 0.9561 at a tenth of a vol point on
the worst benchmark, 1.0013 / 0.9997 on the best), and the mandated sign flip negates every one.

!!! warning "`.grad` after an analytic chain is 0.3–2.3% of the answer, which is worse than six orders out"
    Basin hopping calls `total_loss.backward()` per evaluation and the quote leaves accumulate across
    all of them. On the Monte Carlo residual that is six orders out with a `NaN` in it; here it is
    2.11e-4 / 4.72e-3 / 3.65e-4 / 4.05e-3 against one-pass numbers of 0.0258 / 0.2024 / 0.1189 /
    0.2702 — **0.31% to 2.33%**, a plausible-looking number that is nobody's derivative. The backward
    reads `dV/dq` functionally and `bootstrap` clears the leaves before publishing; this measurement
    says the second is load-bearing rather than tidy.

!!! note "Honest negative result — the basin stage alone is already stationary here"
    On the four-quote block the seed reads `‖J'r‖` 9.64e-3 and is refused against the declared 1e-3;
    basin hopping ALONE reaches 5.85e-7 and is accepted; the full chain reaches 6.91e-8 (the Monte
    Carlo path's basin-only reading is 2.9e10). So the gate that reaches the refusal had to change
    instrument.

#### The re-solve reference, refuted a fourth time {#the-re-solve-fourth-time}

The three refutations had a diagnosis with two halves: the solve wanders in the **flat** directions,
*and* on the identified block it **stops short** (`‖J'r‖` 8.24e3), so `θ*(q)` was the optimizer's
stopping point rather than the argmin. The analytic objective removes the second half outright — every
re-solve below lands at `‖J'r‖` between **8.6e-7 and 1.6e-5**, nine orders below the Monte Carlo
chain's stopping point. Same fixture, same chain, same seed.

**It changes nothing.** Quote 12 (3Y × 3Y) of the flat 25-quote grid, cold-started both sides, against
a one-pass `‖dθ/dq‖` of **36.70** in that column:

| h (vol points) | ‖θ(+h) − θ(−h)‖ | ÷ 2h | × 36.70 | cosine with it | in the kept 13 | MC path, same rung |
| --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 0.03872 | 3.872 | 0.1055 | 0.0607 | 0.820 | 0.037, quotient 3.7 |
| 0.2 | 0.02182 | 5.456 | 0.1486 | 0.1657 | 0.576 | 0.021, quotient 5.3 |
| 0.1 | **1.92016** | **960.1** | 26.16 | 0.0197 | 0.807 | 0.013, quotient 6.7 |

The quotient **grows** as h shrinks — two and a half orders at the finest rung, where the up-bumped
solve found a different basin outright. The displacement points **nowhere**: cosines of 0.02 to 0.17
against the 1/√23 = 0.209 a random direction scores, and a fraction inside the kept 13 of 0.58 to 0.82
against a random vector's √(13/23) = 0.752 — so it is not preferentially in the *discarded* directions
either. And every re-solve lands 0.27 to 0.30 from θ\* whatever h is, against bumps worth 0.18 /
0.073 / 0.037 in θ: what sets the displacement is where each search stopped. The two coarse rungs
match the Monte Carlo ladder **to two digits**.

**That settles which half of the diagnosis was load-bearing** — the flat directions; stationarity was
never the obstacle. The classic oracle stays unavailable, `dθ/dq` is not gated against it, and the
value-space direction check remains the reference. Six cold analytic solves of the 25-quote block,
**237 to 333 s each** in CPU float64, is what keeping that refuted costs, and it is why the ladder is
one column rather than the grid.

## The closed-form map — increment 3 {#the-closed-form-map}

`GBMAssetPriceTSModelParameters` turns the ATM column of a vol surface into the integrated vol curve a
risk-neutral GBM reads, and **it does not fit anything** — so it needs no contract at all:

$$V(t_i) = \bar\sigma(t_i)^2 t_i
\qquad
V(t_i) - V(t_{i-1}) = \tfrac{\Delta t}{3}\big(\sigma_{i-1}^2 + \sigma_{i-1}\sigma_i + \sigma_i^2\big)$$

the second being Simpson's rule solved for the instantaneous vol over each step, positive root taken.
No fixed point, so no implicit function theorem, no stationarity tolerance, no pseudo-inverse and no
dropped Gauss–Newton term. `integrated_vol` is the whole of it and autograd walks it.

**A twin, spliced — not a replacement.** `integrated_vol` is the numpy walk this family always
shipped, and every written mark comes out of it; `carried_vol` is the same walk in float64 torch,
riding in as `integrated_vol + (carried - carried.detach())`. One walk is not enough, even though
`+ - * /` and `sqrt` are all correctly rounded in IEEE 754: `torch.sqrt` is one ulp below `np.sqrt` on
**1.4%** of float64 inputs on this box, and a torch walk re-associates the expression tree besides.
Measured on 4000 random ATM columns, letting torch write the curve moves the **shipped** vols on
24.3% of them by up to 2 ulp; with the splice the written curve moves on **none** while the twin still
differs on the same 971.

!!! warning "The map is the IDENTITY wherever forward variance rises"
    Only $\bar\sigma$ is written — $\sigma$ is the walk's own state, sizing the next step's floor and
    never published — so on a well-behaved column the curve that comes back is the column, up to the
    rounding of a square and its root: exactly so on the fixtures gated here, but **not as a property
    of the map** (a round trip returns a different last bit on 5.7% of random rising columns over the
    gated expiries and 23.1% over a ten-point desk grid). Either way `dV/dq` is `dV/dθ` relabelled and
    such a fixture passes whatever the walk does, so every derivative gate here runs on a **declining**
    column and the rising one only pins the identity as the property it is.

### The repair is a kink {#the-repair-kink}

A column implying a *falling* forward variance has no real root, so $V(t_i)$ is floored at the least
variance the step can reach — the one $\sigma_i = 0$ leaves — and the written vol is that floor rather
than the quote. The switch is a **kink**: $d\bar\sigma_i/dq_i$ is 1 on the smooth side and 0 on the
floored one, and autograd reports the one-sided derivative of the branch the column is in, which is
the only quotient a piecewise map has a limit for. Measured at $\pm10^{-3}$ either side: **1.0** and
**0.0**. Straddling the switch reports **exactly 0.5 at every $h$** — one side moves with the quote
and the other does not — so a symmetric bump ladder converges here to nobody's derivative. That is
gated, because it is the reading a ladder would quietly have produced.

The severance is a whole **column** of the Jacobian, not a diagonal entry: the floor is built out of
the walk's state *before* that expiry and $\sigma$ over the floored step is zero, so the next step's
floor does not carry the quote either. The quadratic is written with $c = \text{floor} - V(t_i)$ so
that branch cancels to an exact zero rather than a rounding of one.

!!! danger "The discriminant is guarded, and that cancellation is why it has to be"
    The floored branch leaves $c$ exactly zero, so $\sigma_i = (-b + \sqrt{b^2})/2a$ is exactly zero. A
    *second* consecutive repair arrives with $b = 0$ beside that zero $c$, so the discriminant is
    exactly 0: the forward value is right, the backward is not — $\sqrt{}$ has an infinite derivative
    at zero, $d(b^2)/db$ is zero beside it, and $\infty \times 0$ is NaN on **every** entry of the
    Jacobian, identity rows included. A *third* repair is what pulls a gradient back through the second
    and detonates it; two in a row look clean. So the root at a
    zero discriminant is written as zero and `sqrt` never sees the point. The gate is a five-expiry
    hump column that repairs three steps running.

### Two quote sources, and which one a config gets {#the-atm-column}

The leaves are the **ATM column, one per surface expiry** either way; where those numbers come from is
a property of the surface's PROVENANCE rather than a switch.

**Preferred: the surface's own `FXVolPrices` quotes.** Where the market data carries an
[`FXVolPrices`](market_prices.md#fxvolprices) block for the surface being integrated **and that block
wrote the surface**, its ATM rows **are** the quotes — an identity, because `Factor2D.malz_skew`
places the ±0.5 label's vol at the delta-neutral straddle strike. Reading it back off the refined grid
would recover it to the grid's tolerance and no better, and would put the Malz delta solve on the tape
to say so.

!!! warning "Provenance is evidence, not a name"
    Keying on the name is a **silent desync**: a hand-authored surface can sit under a name a quote
    block also uses, the pricers read the surface, and the integrated curve is built off numbers
    nothing else agrees with — **20–26 vols against a 39–45 vol surface** on the gated fixture, both
    halves individually valid and neither raising. What is checked is the fingerprint
    `FXVolSurfaceParameters` leaves and [`pinned_grid`](market_prices.md#fxvolprices) reads back — the
    `Malz` subtype beside the `Grid_Tolerance` the grid was refined at. A surface the family *did*
    write whose quotes have since moved off its expiries is the other half of the same desync, and it
    **raises, naming both expiry sets**, rather than dying on a `KeyError`.

**Fallback: the surface, at moneyness 1.** Anything else is authored data and the ATM column is what
`np.interp` reads off it. Where the surface carries a node AT moneyness 1 that read *is* the node, so
`dV/d(ATM column)` is `dV/d(surface node)` there.

!!! warning "OPEN DEFECT — a hand-authored `Malz` surface reads a wing"
    Moneyness 1 is the ATM coordinate of a **ratio** surface. A `Malz` surface's axis is $\log(F/K)$,
    whose ATM is at 0 and whose grid stops at ±0.5, so `searchsorted` lands on the last node and the
    "ATM column" is a deep wing — **0.194 against a quoted 0.200** on the gated fixture, and a full vol
    point at three months on a USDZAR-shaped smile. This predates quote derivatives, is a defect of the
    *read*, and is **in nobody's gate**; the preferred path above is what such a surface reaches in
    practice, so it is named rather than moving a shipped forward for it. That the two sources are
    different numbers is itself gated.

### The attachment, and the triangle {#the-gbm-attachment}

Nothing new. `Quote_Sensitivity` is the declared field, default `No`; `Config.bootstrap` harvests
`calibrated` and `quote_leaves` as it does for the other families; θ\* reaches the calculation through
[`factor_leaf`](#the-attachment). `quote_leaves` publishes **one vector leaf** per block — the curve
family's shape, because the whole ATM column enters one map.

`tests/test_gbm_ts_quotes.py` and `tests/test_vol_term_structure_strip.py` were culled in 104bd08, so
these are a record rather than a running check. The forward was `np.array_equal` throughout — the
written curve on vs off on both quote-source paths, the shipped walk against the numpy loop it always
was on 5 fixtures, a reference exposure run's CVA, profile and whole gradient frame — with the torch
twin ≤ **1 ulp** away and the Simpson identity inverted independently to **1.2e-16**. What the
derivative was pinned on:

| gate | result |
| --- | --- |
| `J` vs central FD of the **whole family** | **2.4e-4 / 2.4e-6 / 2.4e-8** at h = 1e-2 / 1e-3 / 1e-4 — $h^2$ |
| three repairs running, on a 5-expiry hump | `J` finite, identity rows intact, FD **1.3e-8 / 1.3e-10** |
| one-sided FD either side of the switch | **1.0** above, **0.0** below; **0.500000000** straddling |
| a surface the family did not write; quotes moved off their own surface | the authored read lands; the second raises naming both expiry sets |
| `dV/dq` vs `J' dV/dθ`, one backward | **1.1e-16** absolute; the repaired quote's delta exactly 0 while its factor delta is 0.524 |

The FD rung is taken through the **whole family** — re-authored surface, re-read ATM column, re-walked
— so what converges is the derivative of the thing the job runs. The identity rows carry no $h^2$
term, so what is left in them is the difference quotient's own rounding, which *grows* as $h$ shrinks
(8.9e-16 to 1.1e-13); they are asserted exact-to-rounding rather than put on the ladder.

## The delta solve — increment 4 {#the-delta-solve}

`FXVolSurfaceParameters` turns a broker's smile into the log-moneyness surface the option pricers
read. Increment 3 took the ATM row; this takes **all** of it, so `dV/d(RR)` and `dV/d(BF)` exist.

```
q  ->  the strangle pair  ->  the Malz wing pair  ->  a bisection per PINNED x-node  ->  surface
```

The first link is `vol(call) = ATM + BF + RR/2`, `vol(put) = ATM + BF − RR/2`: linear, exactly
invertible, and its twin is bit-identical to the shipped `smile` because there is no `sqrt` for two
implementations to disagree over. The second places the ±0.5 label at the delta-neutral straddle and
mirrors a one-sided smile onto the other wing. The third is the design decision.

### The tape boundary, and why it is not where it looks {#the-tape-boundary}

`Factor2D.malz_delta` closes an array of brackets 64 times and `malz_sigma` is the wing lookup over
the root. Every operation in that loop is `+`, `*`, a comparison and a `torch.where`, so **a tape runs
straight through it and reports a number** — which is the trap, because the number is wrong.

A bisection's iterates are **dyadic combinations of the bracket endpoints**: after any number of
halvings `δ_n = α·lo + (1−α)·hi` with α a step function of the data, zero derivative almost
everywhere, so the tape carries `d(endpoint)/dq` rather than `d(root)/dq`. On the call wing `lo` is a
quoted pillar delta and `hi` is `delta_atm`, so the reported derivative is a function of the ATM quote
and nothing else the root actually moves with. Measured: the shipped loop mirrored literally gives the
same forward number to **2.8e-17** and a Jacobian **0.135** out on entries of order one (6.5%
Frobenius), with `dσ/d(ATM)` a plausible **1.000137** where the truth is **0.865559**.

So **the tape starts at the converged root.** δ\* is a constant and the differentiable object is one
Newton step off it:

$$\delta = \delta^* - \frac{R(\delta^*, q)}{\texttt{detach}\big(\partial R/\partial\delta\big)}
\qquad\Longrightarrow\qquad
\frac{d\delta}{dq} = -\Big(\frac{\partial R}{\partial\delta}\Big)^{-1}\frac{\partial R}{\partial q}$$

— the [implicit function theorem](#the-ift-contract) as an expression rather than a `Function`. **What
makes it the theorem is that δ\* is the root**, not the `detach` on the slope: `R(δ*)` is a rounding,
so anything the slope's graph would contribute is multiplied away ([measured](#the-clamp)). `R` is the
residual the value path solves, mirrored once and differentiated by autograd for **both** the slope and
the quote side. The mirror is not the same function — `scipy.special.ndtr` against the engine's own
`utils.norm_cdf` — so the twin's surface sits 2.8e-17 off the shipped one, and the splice
`value + (carried − carried.detach())` keeps the written surface the numpy one bit for bit.

**The grid is not on the tape at all**, which is a property rather than an omission: it is refined
against the quotes when BUILT and [pinned](market_prices.md#fxvolprices) from then on, so a tick is a
values patch instead of a recompile and the twin moves the **vols on frozen nodes**. A rebuild is a
new plan, and a difference quotient across two plans is not one. Everything the solve decides
discretely — ordering, the ±0.5 label mask, which side had its ATM node mirrored in, which wing a node
reads, whether it is bracketed, which linear segment the root sits in — is read off the value path,
for the same reason a permutation has no derivative.

!!! warning "What *is* taped that looks like a coordinate"
    `delta_atm` is. The ATM quote sets the delta-neutral straddle, `|δ| = ½exp(−σ_atm²T/2)`, which is
    **where the two ATM nodes of the wing grid sit** — so the wing's knot *positions* are a function of
    a quote. A twin treating the deltas as coordinates loses that channel silently: forward untouched,
    ladder breaks at h², ATM column comes back looking almost right. Detaching them is a gated mutation.

### Four discrete choices, and the fourth is a jump {#the-four-choices}

Increment 3 had one piecewise switch; this map makes four per node, every one a property of the
shipped conversion rather than of the twin:

| switch | where | what it does to the surface |
| --- | --- | --- |
| the **wing** | `x = σ_atm²T/2` | which wing the node reads — the two agree in value at the straddle strike, not in slope. A kink |
| the **bracket** | where the root arrives AT the wing's endpoint | the flat extrapolation takes over. A kink |
| the **segment** | where δ\* crosses a quoted pillar delta | the wing is piecewise LINEAR in delta. A kink |
| the **clamp endpoint** | where the two endpoint residuals swap which is smaller | a clamped node steps from one endpoint knot's vol to the other's. A **JUMP** |

Across the three kinks autograd reports the one-sided derivative of the branch a node is in, and a
central difference **straddling** one converges to the average of two one-sided derivatives — which is
nobody's: **0.9403** between a clamped 1.0 and a bracketed 0.8806047, at h = 1e-6 and again at 1e-8.

**The fourth is a discontinuity, found by mutating the instrument rather than the code.** A clamped
node takes whichever end of the bracket its residual misses by less (`|f_lo| < |f_hi|`), and the two
ends are *different knots carrying different vols*, so where those magnitudes cross the written number
steps. On a single-expiry smile whose risk reversal is 2.5× its ATM vol, a **2e-6** move in the ATM
quote jumps one node by **0.1199** of vol. The flip starts around RR ≈ 1.5 × ATM, where the same bump
is worth 0.086. Nothing needs repairing — the flat extrapolation is changing its mind about which knot
to extrapolate *from*, no derivative exists at the crossing, and autograd's one-sided answer is correct
either side.

What was wrong was the **measurement**. Wing, bracket and segment are all identical either side of that
flip, so a three-part fingerprint scores the rung and scores a jump divided by 2h: **6.0e+04** at
h = 1e-6, ten times that at 1e-7 — the signature of a step. So the fingerprint carries the endpoint as
a fourth mark, **on the clamped branch alone**, which is the branch that reads it; recording it
unconditionally would exclude rungs where nothing happened. On the gated fixture no clamped node flips
and the census is unchanged: **24** straddles at h = 1e-3, **5** at 1e-4, **none** at 1e-5.

### The clamp is flat extrapolation, and its derivative is exact {#the-clamp}

A third of the grid — **32 of 97 nodes** on the gated fixture, 20–43% per expiry — has no fixed point
inside its wing's bracket. That is not a repair: beyond the widest quoted delta the smile is **flat**,
so the vol at such a node IS the endpoint knot's. Its derivative is therefore that knot's own quote
algebra and nothing else — `1` in the expiry's ATM quote, `1` in the pillar's butterfly, `±½` in its
risk reversal, and exactly zero in everything else, on all 32 rows to the last bit. The taped δ on that
branch is the knot itself, so the `(δ − d_j)·slope` term cancels to an exact zero, the same discipline
[the floored branch](#the-repair-kink) needed.

**The row is asserted as a SET.** Walking the row's non-zero entries visits the columns that *are* live
and can never notice one that is **missing** — a family that dropped the risk reversals from the tape
publishes a shorter row of perfectly correct entries and walks through. That mutation passes the
entry-by-entry form and fails the set.

!!! warning "`dσ/d(ATM) == 1` is not the clamped branch's signature"
    It is the signature of a node whose vol does not move with delta, and a **bracketed** node can be
    one: where two adjacent pillars' wing vols coincide the segment between them is flat, so sliding
    `delta_atm` under the node changes nothing and the level follows the ATM quote exactly.
    `ATM + BF + RR/2` = 0.1396 at both the 0.35 and 0.25 delta of an ordinary mildly-inverted three-year
    does it: **1 of that smile's 16 nodes**, and 5 of 90 on the two-expiry version. The gate saying a
    bracketed row's ATM entry is never 1 was passing on fixture luck; it is now conditioned on the
    segment not being flat, with a second gate reaching the excluded case.

!!! danger "A flat smile divides zero by zero, and only in the backward"
    Quote the ATM row and nothing else — a legitimate config the value path handles without comment.
    `malz_skew` mirrors the single ATM node onto **both** sides, so each wing is one knot: its span is
    exactly zero and so is its slope. Dividing first and selecting after puts a NaN on **every** entry
    of the Jacobian while the written surface is a perfectly good flat one. Guarded with the
    double-where, the flat smile's Jacobian is the **expiry indicator**. Reachable from the schema
    rather than from a pathological fixture.

!!! note "Honest negative results — three no-ops in `carried_sigma`, all measured"
    Each was mutated and every gate stayed green. Recorded so they stay known properties rather than
    being read as tested ones.

    - **The second guard is idiom, not a measured hazard.** The same double-where sits on the Newton
      step's slope, which nothing cancels to zero: a 375-point sweep over x ∈ [−0.6, 0.6] drives
      `∂R/∂δ` no lower than **0.948**.
    - **Detaching the slope is conceptual hygiene, not the mechanism.** `R(δ*)` is ~1e-17, so a graph
      carried through `∂R/∂δ` is multiplied away; `create_graph` changes no reported number.
    - **`base.detach()` is dead** — `base` is minted from a numpy array and carries no graph. Both
      lines stay because they say what the expression *means*.

### What the Jacobian has to look like {#the-jacobian-structure}

The quote algebra is visible in `J` and the gates assert it as arithmetic:

- **Block diagonal in expiry, exactly** — each expiry's smile is built from its own rows onto its own
  nodes, so `max|J|` off the block is **0.0**.
- **`RR = ±½ BF`, wing by wing, exactly** — a node's vol depends on the quotes only through its own
  wing's knot vols (and, through `delta_atm`, on the ATM quote), and those knots are `ATM + BF ± RR/2`,
  so the two columns are in exact ratio and the sign is which wing the node reads (`np.array_equal`,
  not `allclose`). That is "the risk reversal is antisymmetric and the butterfly is symmetric" said so
  that floating point can check it: BF's column is ≥ 0 everywhere, reaching exactly 1.0 on the clamped
  10-delta wing, and RR's flips at the wing boundary — +0.436 … +0.500 on the call side against
  −0.483 … −0.509 on the put side.
- **The ATM column lands near one, not on it** — every wing knot carries the ATM quote with coefficient
  one, but `delta_atm` moves too, sliding the delta each node resolves to. Measured 0.8656 … 1.0537,
  everywhere positive, and asserted *not* to be exactly one, because exactly one is what the
  frozen-`delta_atm` mutation produces.

### The gates {#the-fx-vol-gates}

`tests/test_fx_vol_quotes.py` and `tests/test_fx_vol_prices.py` were culled in 104bd08 and nothing
under `tests/` exercises the Malz lookup as maths today, so these are a record rather than a running
check. The forward was held by `np.array_equal` throughout — written surface on vs off and against
c77740e's SHA-256, a reference FX option valuation's whole `Greeks_First` frame, the carried smile
against `smile`, the carried skew against `malz_skew` (wing deltas ≤ 1 ulp) — and the `market_patch`
round trip held the grid and `J` `array_equal` across a re-bootstrap. What the derivative was pinned
on:

| gate | result |
| --- | --- |
| **the bisection taped literally** | `J` **0.135** out, 6.5% Frobenius, 1.000137 against 0.865559 |
| `J` vs central FD of the **whole family** | **9.8e-5 / 9.8e-7 / 9.8e-9** at h = 1e-3 / 1e-4 / 1e-5 — $h^2$ |
| the same FD on the clamped nodes | exact to rounding, **1.3e-14 → 1.8e-12** — it GROWS as h shrinks |
| every clamped row, as a SET | **32 / 32** exactly `{ATM: 1, BF: 1, RR: ±½}` and nothing else |
| one-sided FD either side of the bracket switch | **1.0** clamped, **0.8806047** bracketed; **0.9403** straddling |
| **a 2.5× risk reversal, bumped 2e-6** | the node steps **0.1199** vol; the fingerprint scores **6.0e+04**, growing as 1/h |
| a wing with a flat segment; an ATM-only smile | a BRACKETED node reads exactly 1.0 (1 of 16); `J` finite and equal to the expiry indicator |
| `dV/dq` vs `J' dV/dθ`, one backward | **1.0e-16** relative; vega chain 7.039e6 against a Black vega of 7.032e6 |
| `Quote_Sensitivity` Yes → No, re-bootstrapped | the connected tensor and its leaf are **gone**, not stale |

!!! note "Honest negative result — the reorder is insurance"
    Three orders are in play: `malz_surface` **emits** expiry-major, `utils.Curve` stores what it is
    handed sorted by **moneyness** first, and `Factor2D` lexsorts back to expiry-major before minting a
    leaf. The twin follows the emission and is paired against it, so the permutation the bootstrap
    applies is the **identity** on every quote set the family can build, and deleting it leaves every
    gate green. It is there because the emission order and the lexsort could drift apart with nothing
    else noticing, and the identity is asserted so the day it stops being one is visible.

### The retraction {#the-retraction}

Increment 3's non-goals closed with **"No differentiable Malz solve"**, on the grounds that a bisection
per node would put that solve on the tape. Retracted: the solve does not go on the tape. The root is
found exactly as it always was, in numpy, and the derivative comes from **one Newton step at the
answer** — two residual evaluations against 64 halvings, value path untouched. What increment 3 said
about the **ATM row** stands: that number is the surface's ATM vol by construction,
`GBMAssetPriceTSModelParameters` still takes it straight off the quote block, and nothing was put on
the tape to recover it.

## The attachment {#the-attachment}

θ\* still carrying its graph has to become the factor leaf a calculation consumes, and there is one
seam where that is possible: `Calculation.factor_leaf`, called from both `_build_factor_state` branches
and from `Base_Revaluation.update_factors` — the three sites that were the first row of [the graph
audit](#the-graph-audit). The leaf offered there is

$$\text{leaf} + \big(\theta^* - \texttt{detach}(\theta^*)\big)$$

the [boundary correction](calc_lifecycle.md#boundary-corrections-the-sensitivity-subsystem)'s shape,
here for its reason: **change what reaches `backward()`, nothing about what is reported**. `leaf` stays
a leaf, so it is still the tensor the pricers read, and `retain_grad` keeps `.grad` populated on the
sum — the factor greek is the same number it always was and `dV/dq` arrives in the *same* pass. Nothing
about discovery ordering or `process_ofs` moves, so a quote bump and its reval are bit-comparable.

The switch is the declared field `Quote_Sensitivity` on the block being solved. `Config.bootstrap`
harvests `calibrated` and `quote_leaves` off the bootstrapper into `calibrated_factors` and
`quote_leaves` on the config; they are tensors, so they cannot live in `Price Factors`, which is data
and gets written back out as JSON.

**The harvest removes as well as adds, and it has to.** Flip `Quote_Sensitivity` to `No`, re-bootstrap
with quotes that have since moved, and an update-only harvest leaves the *previous* connected tensor
standing under the same key — the old surface against the old quotes, reported by a backward as
today's, raising nothing because a splice worth zero forward is invisible to every price gate. So each
family drops its own keys before publishing: the **factor type it writes** and the **Market Prices type
it reads**, both already declared, scoped because `Config.bootstrap` runs one family at a time.

Both an ordinary price factor and a calibrated model's parameters attach at that same seam, which is
why it is a method and not four call sites. An `FXVol` surface is a *static* factor, so increment 4
publishes its connected tensor under `Factor('FXVol', name)`; `Credit_Monte_Carlo._build_factor_state`
mints one leaf per *named* parameter of an implied model, and `SwaptionCalibration.split` is the one
place the flat 23-vector comes apart — used by the residual and the publish alike, so a factor leaf
cannot be handed the wrong slice. Only an equality against `Price Factors` can see that go wrong,
because the splice is worth zero *whatever* is attached. `Gradient_Variables` governs a factor as
`Factors` or `All` and an implied model as `Implied` or `All`.

!!! warning "A `Tenor_Offset` declines the attachment — and there is a PARKED RULING in it"
    A non-zero offset shifts every tenor before the leaf is minted, so the curve the calculation
    consumes is a **different** one and $d\theta_{\text{shifted}}/dq$ is not $d\theta/dq$.

    **The parked ruling:** the decline is unconditional, and a `Factor2D` surface's `current_value()`
    *ignores* the offset — so under a non-zero `Tenor_Offset` an FX vol quote delta is dropped for a
    surface that did not move. Refusing is the conservative direction, so it stays as it is until the
    offset's own semantics per factor type are worth writing down.

!!! warning "`Gradient_Variables` must be `All` or `Implied` for an implied model"
    `Factors` reaches the zero curve and never the calibrated parameters, so a swaption-vol quote delta
    authored under `Factors` reports nothing at all. Not a new switch; a second consumer now depends on
    it.

!!! note "The dedupe invariant survives by construction"
    A `…ModelParameters` factor reachable *both* as a spot process's implied factor and as an ordinary
    static dependent must map to **one** tensor — see [the
    invariant](calc_lifecycle.md#compile-phase-2-_build_factor_state). The static branch reuses
    `implied_leaves`, so it reuses the *connected* tensor and the quote gradient cannot split across
    two leaves. Nothing in the deal tree pulls an HW2F block that way today, so the collision is
    authored in the gate rather than waited for.

**Two `quote_leaves` shapes, and a reporting layer has to honour both.** The value is
`(descriptors, leaves)` in every family, but a curve solve publishes **one vector leaf** whose entries
are the coupled SET's `Points` (block-prefixed where the set has more than one member, filed under
every member's key) while a swaption calibration publishes a **tuple of scalar leaves**, one per
`Instrument_Definitions` row. The shapes are the quotes' own — a curve's quotes enter one residual as a
vector, a swaption's quote carries its own Black preamble — so anything reading `dV/dq` iterates rather
than assumes. There are still two: increments 3 and 4 both publish the vector shape. The FX descriptors
name the pillar as well as the expiry (`ATM 1`, `RR 0.25 1`, `BF 0.1 1`), because a quote there is
identified by three things.

!!! danger "Descriptors collide across families, and the truth is the SUM"
    One JSON number can feed **two** chains: `FXVolPrices` writes the surface an option pricer reads and
    `GBMAssetPriceTSModelPrices` integrates *that surface's ATM column* into the vol curve the FX rate
    is simulated with. With both blocks asking for `Quote_Sensitivity`, a single ATM quote reaches `V`
    twice and its `dV/dq` is **split across two `quote_leaves` entries whose descriptors are the same
    string**. Measured on a CVA over a stacked pair: `ATM 0.5` reads 2.243453e4 on the `FXVolPrices`
    leaf and 8.071709e4 on the `GBMAssetPriceTSModelPrices` leaf, against a bumped total of
    **1.031516e5** to 3.2e-11 relative — reading the surface's leaf alone is 78% short.

    There is nothing to merge in the engine: the two families are genuinely two maps, and which one a
    consumer wants is a reporting question. **A consumer that reports per-quote deltas must group by
    descriptor across blocks and sum.**

## The precision seam {#the-precision-seam}

The curve bootstrap and its Jacobian are **float64 regardless of the simulation's precision**.
`BenchmarkInstruments.dtype` states it once and `construct_bootstrapper`'s own `dtype` — float32 by
default — does not reach it: a solve converging to 1e-10 cannot be done in float32. Gate:
`test_the_solve_is_float64_whatever_the_bootstrapper_was_built_with`.

θ\* crosses back into the simulation as one `.to()` inside `factor_leaf`. The cast is
**differentiable**, so the cotangent is promoted back to float64 before the linear solve — a float32
cube loses the precision of the *cotangent*, not of the contraction. **Bit-identity survives it**:
`theta - theta.detach()` is evaluated after the cast, in the cube's dtype, so it is exactly `0.0`. And
the seam is **one-way** — the residual closure, its Jacobian and the transpose solve are all float64.

**The swaption calibration sits the other way round, and it has to.** Its residual is a Monte Carlo
over the whole path set, so it runs in the job's own dtype — float32 by default — because forcing
float64 would multiply the cost of every evaluation of an optimizer that makes thousands. The seam is
therefore *inside* the backward:

- **The residual is float32; the linear algebra is float64.** $J$ is promoted with `.double()` the
  moment autograd hands it over, and the pseudo-inverse, the transpose contraction and the stationarity
  norm run there. $\partial r/\partial q$ is never materialised — $-(\partial r/\partial q)^\top (Jw)$
  is one autograd VJP — so the promotion buys the linear solve's conditioning, not the residual's
  accuracy.
- **`grad_outputs` casts back**, so what the backward reports carries float32 resolution however
  exactly it was computed: 4.4e-8 absolute against columns of norm 4 on the four-quote fixture. That is
  the floor under every tolerance on this path.
- **The residual's own resolution is the floor under the quote gradient too.** A check of
  $\partial r/\partial q$ against a central difference stalls at about 5e-3 in float32 however small
  the bump — a cancellation of two numbers near a minimum. Re-run in float64 at the float32 solve's
  θ\* it converges as $h^2$ (1e-2, 1e-4, 1e-6). The derivative is right; float32 is what it is
  reported in.

## The validation triangle {#the-validation-triangle}

Three corners, deliberately independent. Of the six rows only **round trip vs θ_true** still runs
(`tests/test_interest_rate_prices.py`, asserting < 1e-10); the CRN ladder, the self-delta matrix, the
one-pass-vs-FD rung and the reference exposure run went with `tests/test_quote_jacobian.py` and
`tests/test_crn_ladder.py` in 104bd08, so the rest is a record.

| gate | result |
| --- | --- |
| θ\* bit-identical, gradients on vs off | `np.array_equal`, max diff **0.0** |
| round trip vs θ_true | **1.7e-15** |
| one-pass `dV/dq` vs `dV/dθ · dθ/dq` (FD) | **8.7e-12** relative |
| CRN quote-bump ladder, re-authored and re-bootstrapped per rung | agreement **3.7e-11 … 8.1e-8**, flatness **8.1e-8** |
| benchmark self-delta matrix | `‖·− I‖∞` = **2.2e-14** |
| reference exposure run, gradients on vs off | `np.array_equal` |

The self-delta identity is the one to read twice. A benchmark is at par, so $PV_i(\theta^*(q), q_i) = 0$
for *every* q, and differentiating that total derivative gives

$$\frac{\partial PV_i}{\partial \theta}\cdot\frac{d\theta}{dq_j}
= -\frac{\partial PV_i}{\partial q_i}\,\delta_{ij}$$

The left-hand side is exactly what a calculation reports when it prices benchmark $i$ as an **ordinary
deal** — the deal carries a number, not a quote — so the reported quote-delta matrix must be diagonal,
and dividing by each instrument's own quote sensitivity makes it the identity. The normaliser is a
*secant* on the fixed solved curve, because a benchmark's PV is affine in its quote, so no part of the
check reuses the machinery it is checking.

### The identified fixture — increment 2 {#the-identified-fixture}

The swaption triangle needs a fixture the classic oracle is **legal on**, and the finding is that it
does not exist for this calibration. The fixture is a 5 × 5 expiry × tenor grid — **25 benchmarks
against 23 parameters**, every row carrying its own quote, so there is no manifold and `θ*(q)` is
locally a function. Expiries reach 10Y because the σ term structures carry knots to 120 months and a
knot past the last expiry is in nobody's variance integral; the quotes are flat because a cube shaped
in expiry and tenor is fittable only by pushing a front σ knot onto its 1e-5 lower bound, where the
solution is not interior and unconstrained stationarity cannot hold at all.

Full column rank makes `θ*(q)` a function only if the solve **reaches** the minimum, and on the Monte
Carlo objective it does not. The swaption suite that held these was culled in 104bd08
(`test_swaption_calibration_solve.py`, `test_swaption_quote_attachment.py`,
`test_swaption_quote_graph.py`, `test_swaption_quote_triangle.py`), so they are a record.

| gate | result |
| --- | --- |
| singular spectrum of `J` (25 × 23) | rank **23**, σ from 5.65e4 down to 0.258, `σ_min/σ_max` 4.57e-6; the declared 1e-8 cutoff keeps **18** |
| stationarity of the accepted θ\* | `‖J'r‖` **8.6e3** at θ\*, 2.9e11 at the seed, `‖r‖` 46.1 — worst benchmark 4.3% out |
| the two dropped Gauss–Newton terms | θ side **0.500064** of `J'J`; q side 0.4785 / 0.5115 / 0.5065 of `J'(∂r/∂q)`, cosine **1.000000** |
| GN against both corrections applied | **1.0022** (cos 0.999940) in the top 4 directions; 1.0136 / 1.1098 / 1.8743 at top 6 / 8 / 12 |
| re-solve at a bumped quote | `‖Δθ‖` 0.037 / 0.021 / 0.013 at h = 0.5 / 0.2 / 0.1 vol points, so the quotient **grows**: 3.7 / 5.3 / 6.7 |
| CRN ladder on those re-solves | **1294 / 2900 / 4189** against a one-pass 1.80e5 |
| where the displacement points | **0.26 / 0.70 / 0.44** of it inside the subspace the cutoff keeps |
| `dV/dθ · Δθ` vs the CVA the re-solve moved | **1.00 / 0.32 / 0.19** relative, converging — so the failure is in the **solve** |
| benchmark self-delta, its **trace** | **17.99964** against the 18 directions the cutoff keeps |
| direction check, nothing re-solved | re-priced move is **1.0333** of the predicted one; the sign-flip mutation gives **−0.9489** |

Three of those are worth reading twice.

**Both dropped terms are half, and they cancel** ([the derivation](#the-dropped-term)); what matters
here is that *both* were measured. The q side cannot be a double backward, because the splice detaches
the model price in its carried half, so the honest instrument is a finite difference of `J` in the
**authored** quote at fixed θ\*: 0.4953 / 0.5115 / 0.4992 on three columns, cosine 1.000000.
Contracted with both corrections the ratio is 0.9908 in the top four directions, degrading to 1.9738
by the twelfth — asserted as a *property*, because that is where the $O(f^3)$ remainder overtakes the
eigenvalue it corrects, the same place `Jacobian_Rcond` already declines to differentiate.

**The trace of the self-delta matrix counts the identified directions.** What replaces the identity is
the orthogonal projector onto the subspace the pseudo-inverse kept, so its trace is that projector's
**rank** and lands on the integer to better than a thousandth. The individual diagonals run 0.05 to
1.04 and must not be asserted near one: the off-diagonal weight scales with the ratio of the two
benchmarks' relative pricing errors.

**The value chain is not what is broken.** The reported factor greek contracted with a re-solve's
actual displacement reproduces the CVA that re-solve actually moved, better as the bump shrinks. So
`dV/dθ`, the attachment and the pricing chain are right; what is not a function of the quotes is `θ*`
itself. Without that row a scattering ladder would be evidence against everything at once.

## Non-goals {#non-goals}

- **No SIMM aggregation or regulatory bucketing** — bucketed quote deltas are the raw material and
  this stops there. No wrong-way risk.
- **No recalibration inside the simulation** — quote sensitivities are t0 risk, and future-dated
  dynamics stay on calibrated parameters.
- **No new pricers**, and no changes to the `instruments.py` pricers beyond what the t0 closure
  strictly requires — so far, none.
- **No vol-surface parameterisation** — SABR and SSVI are out of scope; a Malz smile is the one delta
  parameterisation built, and a swaption quote here is the number on the `Instrument_Definitions` row
  or a premium, never a smile parameter. (It was also the surface's ATM read until 2026-09-01; a zero
  row refuses by name now.)
- **No reporting format** — `dV/dq` lands on the quote leaves in `Config.quote_leaves` paired with each
  quote's descriptor, and no `Greeks_First`-style block is emitted: `make_factor_index` reads a tenor
  grid off `all_factors`, and a quote is not a factor. A consumer would have to honour the [two
  shapes](#the-attachment) and the descriptor collision.
- **No second differentiation** — neither `CalibrationSolve.backward` nor `LeastSquaresSolve.backward`
  supports `create_graph`, and the second refuses it explicitly. A Gauss–Newton contraction carries no
  second derivative, so a quote-space Hessian off that node would be the curvature of a different
  problem.
- **No differentiable x-grid** — the log-moneyness nodes are refined against the quotes ONCE and
  pinned; the tape moves the vols on them. A grid following its quotes would make every tick a
  recompile, which is what [pinning](market_prices.md#fxvolprices) exists to prevent.
- ~~**No differentiable Malz solve.**~~ **BUILT — see [increment 4](#the-delta-solve).** The premise
  was that a bisection per node would have to go on the tape; it does not, and the IFT needs the answer
  rather than the iteration. Kept struck rather than deleted, because it is the reasoning the increment
  refuted.
