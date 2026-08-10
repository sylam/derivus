# Quote Sensitivities

!!! note "Both increments are complete"
    Increment 1 — the t0 benchmark closure and its graph audit, the multi-curve solver, the
    `InterestRatePrices` family, the quote-side graph, the implicit-function-theorem wrapper, the
    factor-buffer attachment and the validation triangle — is built. So is increment 2: the same
    contract around the HW2F swaption-vol calibration, where the fixed point is the
    [stationarity](#the-stationarity-contract) of a least-squares loss rather than a root. What
    remains of the workstream is **FX vol**, which follows the same shape — see
    [the non-goals](#non-goals).

The autograd tape started at *calibrated* factors, so a greek was reported in zero-curve-node or
model-parameter space. Desks explain P&L in **quote** space — par swap rates, FRA strips, OIS
quotes, swaption vols. This workstream extends the tape one layer upstream, by owning the
calibration inside the library and differentiating through its fixed point, so one backward pass
yields `dV/dq` alongside `dV/dθ`.

The chain is

```
q (leaves)  ->  calibration solve  ->  θ*  ->  factor buffers  ->  scenarios  ->  V
```

and the only new arithmetic is the middle arrow. Everything downstream of θ* is the engine that
already exists. The two solves put different things in that arrow — a damped Newton on a root, and
a Monte Carlo least squares on a stationarity point — and the page reads in that order.

## The residual is priced by the engine's own pricers {#the-residual}

A benchmark is an ordinary deal. A quote **names an instrument type and carries a block of it** —
see [Market Prices](market_prices.md#a-quote) — so the residual is a vector of the library's own
pricing functions:

$$F_i(\theta, q_i) = PV_i(\theta, q_i) \big|_{t_0}$$

evaluated so that a fair instrument prices to zero. `bootstrappers.BenchmarkInstruments` is that
evaluation: it compiles a set of benchmark deal-tree nodes once — discovery, factor objects, tenor
payloads, each leaf's `Factor_dep` / `Time_dep` — and then prices them at t0 from a
`{Factor: tensor}` of curve nodes, returning one PV per benchmark as a `torch` vector whose graph
reaches those tensors.

A benchmark is a **node**, not a deal, so a container with `Children` is one benchmark: a deposit
and an FRA are single deals, a par swap is one `SwapInterestDeal`, and an OIS swap is a container
over an OIS-compounded floating leg and a fixed leg. The node's PV is the sum of its leaves', each
already converted to the reporting currency by its own `pv_*_leg`. There is no netting or
collateral rule to apply on top, which is what keeps this out of `DealStructure`.

The reporting currency **is** the curve's currency, which makes every `calc_fx_cross` the identity
and keeps an FX rate out of the residual. A curve solve is single-currency by construction.

## The graph audit {#the-graph-audit}

The factor-construction path severs autograd in four places. Finding them is the point of this
increment, not incidental to it — a severance here does not raise and does not change a value, it
silently reports a zero gradient.

| Where | What it does | How the closure avoids it |
| --- | --- | --- |
| `Calculation._build_factor_state`, `Base_Revaluation.update_factors` | Every leaf is minted as `torch.tensor(factor.current_value(offset=…), requires_grad=…)` — a fresh leaf built out of a **numpy array**. Anything upstream is severed by construction. This is the *only* way a curve becomes a tensor on the ordinary path. | θ is written straight into `t_Static_Buffer`, which is where the pricers read a static curve from. `current_value` is never called for a curve being solved. |
| `riskfactors.Factor1D.current_value` | numpy end to end — `param['Curve'].array[:, 1] + self.delta`, `np.interp`, `np.concatenate`. Handed a tensor it would still return numpy. | Not on the closure's path. It is still used for the **constants** — every factor the solve is not solving for — where a detached leaf is the right answer. |
| `Factor1D.__init__` → `check_interpolation` | Precomputes the Hermite `(g, c)` coefficient pair from the numpy rate column at construction. Those coefficients are constants in θ. | The pricing path does not read them. `utils.Interpolation.build` re-derives the pair from the buffer **tensor** (`hermite_interpolation_tensor`), and `all_tenors` carries only the interpolation *kind* and the tenor grid, both θ-independent. A Hermite curve therefore differentiates, and there is a gate per interpolation kind saying so. |
| `utils.TensorSchedule.bind` | Copies the whole cashflow schedule across with `new_tensor` — notionals, accruals, margins **and the fixed rate**. This is where the **quote** stopped being differentiable. | **Closed.** `TensorSchedule.carry` gives the *tensor* half an optional per-column overlay, spliced into the copy `bind` makes; see [The quote side](#the-quote-side). θ never passed through this seam, so the θ-side closure is unchanged either way. |

Two traps that are not severances, and would each be silent:

- **`t_Buffer` is a memo table keyed by `(stoch, Factor)` and a time hash, not by the tensor's
  identity.** A pricing state reused across two different θ answers the second call with the first
  call's discount factors — a solver built on a reused state converges to whatever it started at.
  `Benchmark_State` is therefore built fresh per evaluation, and a gate holds the trap in place so
  that it stays a known property rather than a surprise.
- **`utils.CurveTenor` caches its tenor grid as a tensor built from the first tensor that queries
  it.** `all_tenors` is rebuilt per closure instance, so a float64 solve cannot inherit a float32
  grid from whatever ran before it.

## The quote side {#the-quote-side}

A schedule is a **dual**: `.np` for the index columns — payment days, accrual days, reset counts,
everything `np.unique` and `searchsorted` are asked about — and a device copy for the arithmetic.
That split is deliberate and it stays: `TensorSchedule.carry` attaches an optional
`{column: tensor}` overlay to the **tensor half only**, spliced into the copy `bind` makes. Absent
by default, so no caller outside a calibration knows it exists; and it is a compile-time edit, so
it happens before the copy exists and raises after — see
[the schedule lifecycle](calc_lifecycle.md#the-schedule-lifecycle). The benchmark set is compiled
outside a calculation, so `BenchmarkInstruments` binds its own schedules, last, once every overlay
is on.

Only **value** columns are overlaid. Measured, not assumed: compiling the four benchmark shapes at
three quotes moves exactly one column, `CASHFLOW_INDEX_FixedRate`/`FloatMargin` (they are the same
index 6), on the deposit, the FRA, the swap's fixed leg and the OIS fixed leg alike — and moves it
*linearly*, second difference exactly zero. No reset column moves, and `_carry_quotes` raises if one
ever does, because a reset value also leaves through `known_resets`, which reads numpy.

`BenchmarkInstruments` therefore takes the benchmark set **twice**: at its quotes, and one percent
higher. The difference between the two compiles is the exact ∂(schedule)/∂q, so
[`QUOTE_WRITERS`](market_prices.md#interestrateprices) stays the only place a quotable instrument
declares where its number goes — the derivative is read off it rather than restated beside it.

The splice is

$$\text{column} = \text{base} + \big(q - \texttt{detach}(q)\big)\,\frac{\partial \text{column}}{\partial q}$$

which is the [boundary correction](calc_lifecycle.md#boundary-corrections--the-sensitivity-subsystem)'s
shape and is there for the same reason: worth **exactly zero** in the forward pass, derivative one.
The tensor half is bit-identical to the plain copy, so enabling quote gradients cannot move a
single PV — asserted with `np.array_equal`, not a tolerance.

!!! warning "The overlay is a derivative carrier, not a reparameterisation"
    A moved quote needs a **fresh closure**. The splice is worth zero in the forward pass by
    construction, so a schedule's value columns do not follow a quote that moves — only the
    derivative does. `pv_fixed_cashflows`' payment tensor is derived from the tensor half and
    memoized beside it in `derived`, which `bind` mints and re-mints together with that half, so
    the memo cannot outlive the copy it was built from.

## The solve {#the-solve}

`bootstrappers.damped_newton` is the plain solver — the one the implicit-function-theorem wrapper
will run unchanged in its forward pass, so that enabling quote gradients cannot move a mark.

The curves being solved are flattened into **one** system, so a projection curve solved against a
discount curve in the same call is a single Jacobian rather than two coupled ones. The Jacobian is
autograd on the residual: one backward pass per benchmark gives a whole row, and no bump loop over
the knots is needed. It is the same derivative the IFT needs on the other side, which is what
"written once, differentiated twice" means.

Damping is a backtracking line search on the max-norm of the residual — full step first, halved
until it decreases. The three tuning knobs are **declared fields of the block being solved**, not
module constants: the JSON is the contract, so a job tightens or loosens the solve without a code
edit, and the `Market Prices` section has fields for exactly this the way `Calibrations` does.

| Field | Default | |
| --- | --- | --- |
| `N_Iter` | 50 | Newton is quadratic near the root and a par-rate seed is already within a few basis points, so a well-posed strip converges in single digits. Reaching the cap raises rather than returning a half-solved curve. |
| `Tol` | 1e-14 | A zero rate is O(1e-2), so this is ~1e-12 relative — inside the 1e-10 a round trip asks for, and where the linear solve's own rounding stops the iteration improving. |
| `Damping_Halvings` | 6 | How many times the line search may halve a step. Below that the step *length* is not what is wrong, so the solve says so rather than creeping. |

Each is read with the declared default as its `.get` fallback, which is the same discipline the
thirteen calibration classes are held to: a declared default and an engine fallback that disagree
mean the same job runs two different ways depending on whether it was hand-written or authored
from the schema, and nothing raises. A gate holds the two together per family.

Convergence is tested on the **step, before the line search**, not on the residual after it. A step
inside the linear solve's own rounding cannot be asked to reduce a residual that is already at
noise level, and a solver that asks converges four iterations in and then raises.

The seed is each quote itself: a par rate is within a few basis points of the zero rate at the same
maturity, and it has to be seeded into `Price Factors` before the closure is built, because the
closure constructs the curve factor out of that section.

!!! note "Honest negative result — the damping never engages here"
    On both round-trip worlds the line search takes the full step at every iteration: zero halvings
    across all three solves (USD OIS 4 iterations, USD projection 4, ZAR 5). Removing the damping
    entirely leaves every gate green. It is insurance against a first iterate these fixtures do not
    produce, and an over-stable fixture is a finding rather than a success — recorded here so that
    nobody reads the passing suite as evidence the line search works.

## The knot rule {#the-knot-rule}

ONE knot per used quote, at that benchmark's last cashflow date, in the block's `Day_Count`, and
**the output grid is that grid**. The reasoning and the consequences are on
[Market Prices](market_prices.md#interestrateprices); what matters here is that the knot grid is
what the Jacobian is square in, so the quote-delta buckets a desk eventually reads are the
benchmark maturities themselves.

## Curve contracts the solve relies on {#curve-contracts}

**No knot at tenor zero — by design, not by accident.** A curve NEVER carries a 0.0 knot: rates
start at T+1 (or wherever the overnight rate settles), and a rate at tenor zero is redundant
anyway — its discount factor is 1 by identity. `Factor1D.interpolate` divides by the tenor for the
rate-times-time kinds, so a zero knot yields NaN, which is the contract asserting itself rather
than a defect to guard against. The knot rule above complies by construction: every benchmark's
last cashflow is strictly after t0.

**The compounding leg is a compile-time SHAPE, not a pricer branch.** `pv_float_cashflow_list`
routes an accrual period through geometric compounding when the reset count differs from the
cashflow count (`all_resets.shape[1] != reset_cashflows.np.shape[0]`) — daily resets against
quarterly cashflows, the reshape set up at `calculate_dependencies` by
`compress_no_compounding(groupsize=-1)` under `Compounding_Method='OIS'`. The regular route's
`Weight = 1/n` resets are the AVERAGING legs' arithmetic and must never reach the compounding
path. This is why an OIS benchmark is authored as a floating list with `Compounding_Method='OIS'`
(composed under a `StructuredDeal` for the par swap) and not as a `SwapInterestDeal`: the swap's
generated legs never pass through the compression. The shape-difference check is acknowledged
tech debt — it works, and it is subtle enough that it is written down here.

## The IFT contract {#the-ift-contract}

`bootstrappers.CalibrationSolve` is the bootstrap as one differentiable node. Its contract is two
sentences:

**Forward is the ordinary solve.** It calls `damped_newton` and nothing else — same iterations,
same tolerances, same float64 — so enabling quote gradients cannot move a mark *by construction*
rather than by a claim someone has to re-check. Autograd runs `forward` with grad mode off, which
the solve needs on for its own Jacobian, so it is re-enabled inside and the iteration's graph is
discarded with the iteration. The block goes through the wrapper whether or not any quote is on the
tape: with `quotes=None` no edge is recorded and the wrapper is a pass-through.

**Backward is the implicit function theorem**, never an unrolled solver. At the fixed point
$F(\theta^*, q) = 0$, so

$$\frac{\partial F}{\partial \theta}\frac{d\theta}{dq} + \frac{\partial F}{\partial q} = 0
\qquad\Longrightarrow\qquad
\frac{d\theta}{dq} = -\Big(\frac{\partial F}{\partial \theta}\Big)^{-1}\frac{\partial F}{\partial q}$$

and an incoming cotangent $v = \partial L/\partial\theta^*$ contracts in two steps:

$$\Big(\frac{\partial F}{\partial \theta}\Big)^{\!\top} w = v
\qquad\text{then}\qquad
\frac{\partial L}{\partial q} = -\Big(\frac{\partial F}{\partial q}\Big)^{\!\top} w$$

Both derivatives come from **autograd on the residual closure itself**, evaluated once at
$(\theta^*, q)$: the $n \times n$ matrix by one backward pass per benchmark, the $q$ side as a
single vector-Jacobian product with $-w$ as its cotangent. The residual is therefore *written once
and differentiated twice*, and the quote derivative cannot drift from the one the solve converged
on. $n$ is the benchmark count, so the linear solve is small by construction — the
[knot rule](#the-knot-rule) is what makes it square.

The Jacobian is recomputed at $\theta^*$ rather than reused from the last Newton step, which was
taken at the iterate *before* it. That costs one iteration's worth of work.

!!! warning "Every `grad` in the backward retains the graph"
    The residual's subgraph is **shared with the forward pass** — `pv_fixed_cashflows` memoizes its
    payment tensor on the schedule — so freeing it in the backward would take the forward pass's
    graph with it.

## The stationarity contract — increment 2 {#the-stationarity-contract}

`bootstrappers.LeastSquaresSolve` is the HW2F swaption calibration as one differentiable node, and
its contract is the one above with a single word changed: the fixed point is the **stationarity** of
a least-squares loss rather than the root of a residual. Everything that follows is what that one
word costs.

**Forward is the ordinary solve.** `SwaptionCalibration.solve` runs the optimizer chain the
bootstrap always ran — basin hopping, then least squares, `x0` chained between them, a candidate
accepted only if it beats the running best *and* the process it implies is well posed. Enabling
quote gradients cannot move θ\* by construction, and the block goes through the wrapper whether or
not a quote is on the tape. Autograd runs `forward` with grad mode off and both optimizers need it
on, so it is re-enabled inside and each evaluation's graph dies with the evaluation.

The seed is the block's, not the process's. Basin hopping's step taker and its Metropolis test used
to draw from the numpy **global** RNG, which made the whole calibration a function of whatever ran
before it in the same interpreter — θ\* moves 0.93 absolute between ambient seeds on the four-quote
fixture. `Random_Seed` is a declared field and one generator serves both, so this is the first
reproducible HW2F calibration; there is no earlier baseline to preserve.

### The loss is a Monte Carlo, and the derivative is conditional on the draw {#the-mc-loss}

There is no par instrument here to drive to zero. Each benchmark is a swaption priced by
**brute-force Monte Carlo** through the engine's own `pv_float_cashflow_list`, and the residual is
the weighted squared relative pricing error against the market premium

$$r_i(\theta, q_i) = w_i\Big(100\big(\tfrac{P_i(q_i)}{M_i(\theta)} - 1\big)\Big)^2$$

with $M_i$ the model price. Two consequences run through everything below. The residual is **already
squared**, so `least_squares` minimises a quartic in the pricing error and $J = \partial r/\partial
\theta$ carries a factor of that error in every row: at an exact fit $J$ is not merely small, it is
**zero**, and the Gauss–Newton system degenerates. And $r$ cannot be driven to zero anyway on a block
quoting more swaptions than the model has parameters, which is exactly the block this was validated
on.

The Sobol engine is seeded once, on the state the closure builds, and `reset` re-draws nothing once
a sample exists — it clears `t_Buffer` and `t_PreCalc`, which is the memo trap, not the sample. So
every evaluation the optimizer makes, every Jacobian, and the backward pass itself price the **same
paths**, and `dθ/dq` is the derivative *conditional on that draw*. This is the same philosophy as a
pinned `process_ofs`: the number reported is the derivative of the number reported, and a sample
re-drawn per evaluation would have the optimizer — and then the ladder — differencing the noise.

### What the quote side closes, and what stays severed {#the-swaption-quote-side}

The severance is at the **market premium** and nowhere else. `create_market_swaps` prices it with
`utils.black_european_option_price`, which is scipy end to end, so it reaches the residual as a numpy
scalar and the vol behind it is a constant by construction. What closes it is the splice this page
already uses twice, on `market_swap_class.error`:

$$\text{error} = \underbrace{w\big(100(\tfrac{P}{M}-1)\big)^2}_{\text{the expression the solve always minimised}}
+ \big(\widetilde{\text{error}} - \texttt{detach}(\widetilde{\text{error}})\big)$$

with $\widetilde{\text{error}}$ the same expression off a float64 **twin** of the premium —
`utils.black_european_option`, the engine's own *tensor* Black, which is what the cap/floor and
swaption pricers value an option with rather than a second opinion of it, and bit-identical to the
numpy one at the money to 1e-12. Worth exactly zero forward, derivative one.

Two details in that line are load-bearing and neither is visible to a price gate.

- **The model price is detached in the carried half, and only there.** The splice is worth zero
  forward but its derivative is not selective: left attached, the carried half reaches the *model
  parameters* as well as the quote and the calibration Jacobian comes out **doubled** — the residual
  stays bit-identical and the optimizer simply walks a different path. The quote derivative of the
  error does not involve the model's own sensitivity, so detaching is what the chain rule says.
- **The premium is a callable rebuilt per evaluation, not a compiled subgraph.**
  `make_basin_hopping_loss` calls `total_loss.backward()` with no `retain_graph`, which frees the
  whole graph the loss was built on; a quote-side subgraph compiled once with the benchmark set would
  be freed with the first evaluation and every one after it would raise. Rebuilding costs one scalar
  Black per benchmark against a Monte Carlo over the whole path set.

Three severances stay **open on purpose**, because their upstream is not a quote of this calibration:
`get_par_swap_rate` prices the strike and the pvbp in numpy off the zero curve, `set_fixed_amount`
writes that strike into the schedule's numpy half, and the surface's ATM read interpolates with
`RectBivariateSpline`. The first two are the calibrated curve — increment 1's quote. The third is the
surface-node-to-ATM map, which is a quote of the *surface* rather than of the swaption.

!!! warning "A premium re-struck by `Volatility_Delta` declines the quote side"
    That path recovers an implied vol with a `brentq` root find and re-strikes the premium off it,
    and a numerical root find carries no derivative. Reporting zero there would be the exact failure
    this workstream exists to prevent, so `Quote_Sensitivity` **raises** instead.

### Gauss–Newton, and the two terms it drops {#the-dropped-term}

$r(\theta^*, q) \neq 0$ and never will be, so what is held fixed is the **gradient** of half the sum
of squares, $g = J^\top r = 0$. Differentiating *that* gives a dropped term on **each** side:

$$\Big(J^\top J + \underbrace{\sum_i r_i \nabla^2_\theta r_i}_{\text{dropped}}\Big)\frac{d\theta}{dq}
= -\Big(J^\top \frac{\partial r}{\partial q} + \underbrace{\sum_i r_i \frac{\partial^2 r_i}{\partial\theta\,\partial q}}_{\text{dropped}}\Big)$$

Gauss–Newton keeps the unbraced term on each side and drops both braced ones. A cotangent $v$
therefore contracts as
$w = (J^\top J)^{+} v$ then $\partial L/\partial q = -(\partial r/\partial q)^\top (Jw)$, with both
derivatives from autograd on **one fresh evaluation** of the residual at $(\theta^*, q)$ —
functionally, through `autograd.grad` and never off `.grad`. The scipy adapters clear `.grad` per
evaluation and the quote leaves accumulate across them, so a harvested `.grad` is the sum over the
optimizer's whole path: on the gate fixture it holds numbers six orders out and one `NaN`.

**Neither dropped term is second-order small here, and that is the interesting part.** The textbook
argument — the term is $O(r)$, so it vanishes as the fit improves — assumes $r$ is the *pricing
error*. On this block it is the pricing error **squared**: `resid` is `x*x`, so $r_i = w_i f_i^2$ with
$f_i = 100(P_i/M_i - 1)$, and

$$J^\top J = \sum_i 4 f_i^2 \nabla f_i \nabla f_i^\top
\qquad
\sum_i r_i \nabla^2_\theta r_i = \sum_i 2 f_i^2 \nabla f_i \nabla f_i^\top + O(f^3)$$

so the dropped Hessian term is **half** what it corrects, at any residual level. The identical
algebra applies to the cross term — $\partial r_i/\partial q = 2 w_i f_i\,\partial f_i/\partial q$
carries the same factor — so the dropped cross term is **half** $J^\top(\partial r/\partial q)$ too.

**The two halves cancel.** The true system is $\tfrac32 J^\top J \cdot d\theta/dq = -\tfrac32
J^\top(\partial r/\partial q)$, which is the Gauss–Newton system. Squaring a residual row-scales
$J$ and $\partial r/\partial q$ by the *same* diagonal $\operatorname{diag}(2w_i f_i)$, and the
normal equations are invariant under that — Gauss–Newton is consistent under per-residual
reparameterisation. So `LeastSquaresSolve.backward` reports the **exact leading-order derivative**,
and no correction is owed.

Both halves are measured, because an algebraic cancellation nobody checked is a claim rather than a
result: 0.500064 on the θ side, 0.4953–0.5115 with cosine 1.000000 on the q side, and the contracted
ratio 0.9908 in the identified directions. The numbers are in
[the identified fixture](#the-identified-fixture); the *instruments* differ, and the reason is a
finding of its own. The θ side is a double backward. The q side cannot be, because
[the splice](#the-swaption-quote-side) detaches the model price in its carried half — the thing that
stops the calibration Jacobian doubling — so the closure's mixed second derivative
$\partial^2 r/\partial\theta\,\partial q$ is **structurally zero** and autograd faithfully reports
zero. Measuring it needs a finite difference of $J$ in the **authored** quote at fixed θ\*, which
goes round the splice rather than through it.

!!! warning "Correcting one side is how a factor 3/2 appears"
    Measure the Hessian term, add it, and leave the cross term alone, and the reported derivative
    looks 3/2 too large — **1.4974** in the top four directions, which is the 3/2 arriving almost
    exactly. That is an artefact of a half-applied correction, not a defect in the backward, and it
    is the gate's own mutation.

!!! warning "Stationarity is checked, not assumed"
    `solve` accepts whatever the chain returned, which can be the **seed** if nothing beat it, and
    the whole contraction above is worthless off the fixed point. So $\|J^\top r\|$ above the
    declared `Stationarity_Tol` **raises, naming the norm**, rather than returning a plausible
    Jacobian of nothing: 3.3e-6 achieved on the four-quote fixture against 2.9e10 for a chain
    stopped after basin hopping.

    The norm is **absolute and the objective's scale is the block's own** — $r_i$ is a weighted
    squared *percentage*, so a block that cannot fit its quotes exactly carries a much larger one at
    the same distance from its minimum. A tolerance is therefore per block, and the identified
    fixture declares its own rather than inheriting the four-quote block's.

    **On an over-determined block the chain does not get there.** It closes seven and a half of the
    eight orders between its seed and stationarity — 2.9e11 down to 8.6e3 — and stops, because it is
    minimising a QUARTIC in the pricing error and a quartic is flat enough near its minimum that the
    relative-improvement test fires long before the gradient does. On a block quoting fewer swaptions
    than the model has parameters this never shows: the fit is then interpolating and `‖J'r‖` is
    small for free. It is the whole reason `θ*(q)` is the optimizer's stopping point rather than the
    argmin, and therefore the reason the classic oracle is unavailable here too.

### Rank deficiency is the problem, not a defect {#rank-deficiency}

$J$ has one row per benchmark and 23 columns — two mean reversions, a correlation, and two
ten-knot volatility term structures. A block quoting four swaptions therefore leaves a
**19-dimensional null space**: those are combinations of model parameters the quote set does not
identify, and no amount of care in the linear algebra can invent them.

The inverse is therefore a **pseudo-inverse** at the declared `Jacobian_Rcond`, and `dθ/dq` in a
null direction is the **minimum-norm representative** — one member of a family the data cannot
choose between. No ridge is added: a Tikhonov term returns a unique-looking number that is the
derivative of a different problem. On the four-quote fixture the cutoff has four orders of headroom
below the smallest real eigenvalue and eight above the largest spurious one.

!!! warning "On an identified block the cutoff is a real decision, not a formality"
    `Jacobian_Rcond` defaults to 1e-8, which on the four-quote block separates *four* real directions
    from nineteen numerical zeros with orders to spare. An **identified** block has no such gap — 23
    real directions spanning the conditioning of the swaption grid itself, five orders end to end —
    and the same cutoff keeps 18 of them. That is right rather than lossy: the term Gauss–Newton
    drops is the same *size* as the eigenvalues of the last five, so a derivative along them would be
    a derivative of the wrong Hessian. The gate reports the spectrum and how many each cutoff keeps.

### The re-solve reference, refuted three times {#the-manifold-finding}

This is the central result of increment 2, and it is a negative one.

The reference this workstream was briefed to validate against — bump a quote, re-run the calibration,
difference θ\* — was **refuted with evidence**, three times.

**In θ space, on four quotes.** On a 19-dimensional solution manifold a re-solve lands *somewhere
else on the manifold*, a roughly fixed displacement of about 0.1 whatever the bump size, so
$(\theta^*(q+h) - \theta^*(q-h))/2h$ **diverges as $1/h$** — measured cold-started and warm-started
alike, with the unidentified part of that difference an order of magnitude larger than the identified
part.

**In value space, on four quotes.** The CVA change that displacement causes swamps the one the quote
causes, and **the sign reverses as the bump shrinks** — +18.1 against +15.6 predicted at half a
percent, then −5.6 against +6.2 at a fifth. No tolerance rescues a sign.

**In both, on twenty-five quotes.** The obvious repair is to quote more swaptions than the model has
parameters, and [the identified fixture](#the-identified-fixture) does exactly that: `J` is 25 × 23
with rank 23 and there is no manifold left. It fails anyway. Full column rank makes `θ*(q)` a
function only if the solve *reaches* the minimum, and this one stops seven and a half orders in. The
displacement is again roughly fixed as the bump shrinks — 0.037, 0.021, 0.013 at half, a fifth and a
tenth of a vol point — so the quotient again grows rather than converging, and the ladder built on
those re-solves scatters against a one-pass number two orders larger.

**The three have one diagnosis.** The solve wanders in the directions the objective is **flat** in.
On four quotes those directions are a true null space, nineteen combinations the quotes cannot see.
On twenty-five they are the ones the declared cutoff discards — measured, only a quarter to seven
tenths of the displacement lands inside the subspace the pseudo-inverse keeps. Either way they are
exactly the directions along which a derivative is refused, which is the pseudo-inverse doing its
job.

Stated plainly, because it is what a desk needs to hear: **bump-and-recalibrate P&L explain is
ill-posed for this calibration.** It is not that the library fails to reproduce a hard reference —
the reference has no limit to converge to, whether or not the quote set identifies the model. The
implicit-function derivative is the better-behaved object of the two, and it is the one that answers
the question the desk was really asking.

What *is* well posed is the direction the quotes **do** identify: step the parameters by
$d\theta/dq \cdot h$ and re-price **without re-solving**. On four quotes that recovers 1.60 of the
predicted move at one percent, 1.32 at a half and 1.10 at a tenth — a second-order remainder
behaving — and the wrong-signed step lands the bumped problem's residual three to four times further
out than doing nothing. On the identified fixture it lands at 1.0333, and the mandated sign-flip
mutation turns it into −0.9489. That check is what the quote delta is gated on in value space, and it
is what the ladder would have been if a ladder were available.

All three refutations are pinned as gates of their own, so they stay known properties rather than
surprises, and so nobody later "fixes" the derivative against an oracle that is not one. They pass by
*failing* to agree; if any flips, the solve has started returning a function of its quotes and the
comparison has become available.

## The attachment {#the-attachment}

θ\* still carrying its graph has to become the `InterestRate` factor leaf a calculation consumes,
and there is exactly one seam where that is possible: `Calculation.factor_leaf`, called from both
`_build_factor_state` branches (static and stochastic) and from `Base_Revaluation.update_factors`.
Those three sites were the first row of [the graph audit](#the-graph-audit) — the
`torch.tensor(current_value(...))` mint. The leaf offered there is

$$\text{leaf} + \big(\theta^* - \texttt{detach}(\theta^*)\big)$$

which is the [boundary correction](calc_lifecycle.md#boundary-corrections--the-sensitivity-subsystem)'s
shape and is here for its reason: **change what reaches `backward()`, nothing about what is
reported**. `leaf` stays a leaf, so it is still the tensor the pricers read, and `retain_grad`
keeps `.grad` populated on the sum — the factor greek reported for that curve is the same number it
always was, and `dV/dq` arrives in the *same* pass. Nothing about discovery ordering or
`process_ofs` moves, which is a feature: a quote bump and its reval are bit-comparable.

The switch is the declared field `Quote_Sensitivity` on the block being solved, not a module
constant. `Config.bootstrap` harvests `calibrated_factors` and `quote_leaves` off the bootstrapper —
they are tensors, so they cannot live in `Price Factors`, which is data and gets written back out as
JSON.

**A calibrated model's parameters attach at the same seam**, which is why the seam is a method and
not four call sites. `Credit_Monte_Carlo._build_factor_state` mints one leaf per *named* parameter of
an implied model — `Alpha_1`, `Alpha_2`, `Correlation`, `Sigma_1`, `Sigma_2` for HW2F — and all
three mints now go through `factor_leaf`, so a bootstrapper that kept its calibration on the tape is
offered where the leaf is born. `SwaptionCalibration.split` is the one place the flat 23-vector comes
apart, used by the residual and by the publish alike, so a factor leaf cannot be handed the wrong
slice of the vector the Jacobian was read off — and only an equality against `Price Factors` can see
that go wrong, because the splice is worth zero *whatever* is attached.

!!! warning "A `Tenor_Offset` declines the attachment"
    A non-zero offset shifts every tenor before the leaf is minted, so the curve the calculation
    consumes is a **different** one and $d\theta_{\text{shifted}}/dq$ is not $d\theta/dq$. Attaching
    anyway would report a plausible number that is the derivative of something nobody priced.

!!! warning "`Gradient_Variables` must be `All` or `Implied`"
    An implied model's leaves are only differentiable under those two — `Factors` reaches the zero
    curve and never the calibrated parameters — so a swaption-vol quote delta authored under
    `Factors` reports nothing at all. That is the same switch that governs the factor greek and it is
    not new; what is new is that a second consumer now depends on it.

!!! note "The dedupe invariant survives by construction"
    A `…ModelParameters` factor reachable *both* as a spot process's implied factor and as an
    ordinary static dependent must map to **one** tensor — see
    [the invariant](calc_lifecycle.md#compile-phase-2--_build_factor_state). The static branch reuses
    `implied_leaves`, so it now reuses the *connected* tensor and the quote gradient cannot split
    across two leaves. Nothing in the deal tree pulls an HW2F block in that way today, so the
    collision is authored in the gate rather than waited for.

**Two `quote_leaves` shapes, and a reporting layer has to honour both.** The value is
`(descriptors, leaves)` in both families, but the second half is not the same object: a curve solve
publishes **one vector leaf** whose entries are the block's `Points`, and a swaption calibration
publishes a **tuple of scalar leaves**, one per `Instrument_Definitions` row. The two shapes are the
quotes' own — a curve's quotes enter one residual as a vector, a swaption's quote is a leaf per
benchmark because each carries its own Black preamble — so anything reading `dV/dq` off them
iterates rather than assumes.

## The precision seam {#the-precision-seam}

The bootstrap and its Jacobian are **float64 regardless of the simulation's precision**.
`BenchmarkInstruments.dtype` states it once, and `construct_bootstrapper`'s own `dtype` — float32 by
default — does not reach it. A solve that has to converge to 1e-10 cannot be done in float32, and
the Jacobian handed to the implicit function theorem is only as good as the residual it came from.
Setting that one attribute to float32 fails eight of the nine round-trip gates.

**θ\* crosses back into the simulation at the `Function` boundary**, and the cast is one `.to()`
inside `factor_leaf`. Three things follow.

- The cast is **differentiable**, so the cotangent arriving at `CalibrationSolve.backward` has been
  promoted back to float64 before the linear solve sees it. A float32 cube therefore still gets a
  float64 calibration Jacobian; what it loses is the *precision of the cotangent*, not of the
  contraction.
- Bit-identity survives it. `theta - theta.detach()` is evaluated **after** the cast, in the cube's
  dtype, so it is exactly `0.0` there and `leaf + 0.0` is `leaf` for every finite value. Rounding
  θ\* to float32 and rounding `current_value()` to float32 cannot disagree, because they are the
  same float64 numbers.
- The seam is **one-way**. Nothing float32 flows back into the solve: the residual closure, its
  Jacobian and the transpose solve are all float64, and `BenchmarkInstruments` rebuilds
  `all_tenors` per instance so a float64 solve cannot inherit a float32 tenor grid from whatever
  ran before it.

**The swaption calibration sits the other way round, and it has to.** Its residual is a Monte Carlo
over the whole path set, so it runs in `construct_bootstrapper`'s own dtype — **float32 by default,
the job's precision** — and forcing it to float64 would multiply the cost of every evaluation of an
optimizer that makes thousands. The seam is therefore *inside* the backward rather than at the
`Function` boundary, and it has three parts.

- **The residual and its two Jacobians are float32; the linear algebra is float64.** $J$ and
  $\partial r/\partial q$ are promoted with `.double()` the moment autograd hands them over, and the
  pseudo-inverse, the transpose contraction and the stationarity norm all run there. The promotion
  buys the linear solve's conditioning, not the residual's accuracy.
- **`grad_outputs` casts back.** $-Jw$ has to be the residual's own dtype before it can be a
  cotangent, so what the backward *reports* carries float32 resolution however exactly it was
  computed — 4.4e-8 absolute against columns of norm 4 on the four-quote fixture. That is the floor
  on every tolerance in the table below, and it is why they are looser than increment 1's.
- **The residual's own resolution is the floor under the quote gradient too.** Stage A's check of
  $\partial r/\partial q$ against a central difference stalls at about 5e-3 in float32 however small
  the bump gets — a cancellation of two numbers near a minimum, the residual's resolution and not
  the derivative's. Re-run in float64 at the float32 solve's θ\* the same check converges as $h^2$:
  1e-2, 1e-4, 1e-6 over three bump sizes. The derivative is right; float32 is what it is reported in.

## The validation triangle {#the-validation-triangle}

Three corners, deliberately independent, plus three identities that need no bump at all.

| gate | what it isolates | result |
| --- | --- | --- |
| θ\* bit-identical, gradients on vs off | the forward pass | `np.array_equal`, max diff **0.0** |
| round trip vs θ_true | the solve | **1.7e-15** |
| one-pass `dV/dq` vs `dV/dθ · dθ/dq` (FD) | the linear solve and the VJP, nothing else | **8.7e-12** relative |
| CRN quote-bump ladder | the whole job, re-authored and re-bootstrapped per rung | agreement **3.7e-11 … 8.1e-8**, flatness **8.1e-8** |
| benchmark self-delta matrix | the IFT equation, through the full chain | `‖·− I‖∞` = **2.2e-14** |
| reference exposure run, gradients on vs off | the stochastic branch and the xVA block | `np.array_equal` |

The self-delta identity is the one worth reading twice. A benchmark is at par, so
$PV_i(\theta^*(q), q_i) = 0$ for *every* q; differentiating that total derivative gives

$$\frac{\partial PV_i}{\partial \theta}\cdot\frac{d\theta}{dq_j}
= -\frac{\partial PV_i}{\partial q_i}\,\delta_{ij}$$

The left-hand side is exactly what a calculation reports when it prices benchmark $i$ as an
**ordinary deal** — the deal carries a number, not a quote — so the reported quote-delta matrix must
be diagonal, and dividing by each instrument's own quote sensitivity makes it the identity. The
normaliser is a *secant* on the fixed solved curve, because a benchmark's PV is affine in its quote,
so no part of the check reuses the machinery it is checking.

### The identified fixture — increment 2 {#the-identified-fixture}

The swaption triangle needs a fixture the classic oracle is **legal on**. Building one is most of the
work, and the finding is that it does not exist for this calibration.

The fixture is a 5 × 5 expiry × tenor grid — **25 benchmarks against 23 parameters**, every row
carrying its own quote, so there is no manifold and `θ*(q)` is locally a function. Expiries reach 10Y
because the σ term structures carry knots to 120 months and a knot past the last expiry is in
nobody's variance integral; the quotes are flat because a cube shaped in expiry and tenor is fittable
only by pushing a front σ knot onto its 1e-5 lower bound, where the solution is not interior and
unconstrained stationarity cannot hold at all.

Full column rank makes `θ*(q)` a function only if the solve **reaches** the minimum, and it does not.
So the re-solve reference fails a third time, and the ladder built on it still scatters. What the
gates hold is everything that does not need it.

| gate | what it isolates | result |
| --- | --- | --- |
| singular spectrum of `J` (25 × 23) | the fixture — is anything unidentified? | rank **23**, σ from 5.65e4 down to **0.258**, `σ_min/σ_max` **4.57e-6**; the declared 1e-8 cutoff keeps **18** of 23 |
| stationarity of the accepted θ\* | the fixed point the theorem needs | `‖J'r‖` **8.6e3** at θ\*, **2.9e11** at the seed, `‖r‖` 46.1 — worst benchmark 4.3% out |
| the two dropped Gauss–Newton terms | the approximation, measured on **both** sides | θ side **0.500064** of `J'J`; q side **0.4953 / 0.5115 / 0.4992** of `J'(∂r/∂q)` on columns 6/12/22, cosine **1.000000** |
| GN against both corrections applied | does the 3/2 cancel? | **0.9908** (cos 0.999996) in the top 4 directions; 0.9941 / 1.0423 / **1.9738** at top 6 / 8 / 12 |
| re-solve at a bumped quote | the classic oracle | `‖Δθ‖` **0.037 / 0.021 / 0.013** at h = 0.5 / 0.2 / 0.1 vol points, so the quotient **grows**: 3.7 / 5.3 / 6.7 |
| CRN quote-bump ladder on those re-solves | the whole job, re-bootstrapped per rung | CRN **1294 / 2900 / 4189** against a one-pass **1.80e5** — agreement 99%, flatness 100% |
| where the displacement points | which cause | **0.26 / 0.70 / 0.44** of it inside the subspace the cutoff keeps |
| `dV/dθ · Δθ` vs the CVA the re-solve moved | the attachment and the greek | **1.00 / 0.32 / 0.19** relative, converging — so the failure is in the **solve** |
| benchmark self-delta, its **trace** | the IFT equation through the full chain | **17.99964** against the 18 directions the cutoff keeps |
| direction check, nothing re-solved | `dV/dq` in value space | re-priced move is **1.0333** of the predicted one |
| sign flip in the backward | can any of it see the subsystem break? | ratio **−0.9489** — fails the direction check |

Three of those rows are worth reading twice.

**Both dropped terms are half, and they cancel.** The derivation is in
[the dropped terms](#the-dropped-term); what matters here is that *both* were measured, because an
algebraic cancellation nobody checked is a claim rather than a result. The θ side is one double
backward per parameter and comes out at 0.500064 of `J'J`. The q side **cannot be taken that way**:
`market_swap_class.error` detaches the model price in the carried half — which is what stops the
calibration Jacobian doubling — so the closure's mixed second derivative is *structurally zero* and
autograd reports it as such. The honest instrument is a finite difference of `J` in the **authored**
quote at fixed θ\*: re-author the block a rung either side, rebuild the residual, difference `J'r`.
That reads 0.4953 / 0.5115 / 0.4992 on three columns with cosine 1.000000 against `J'(∂r/∂q)` — the
same half, pointing the same way.

Contracted with both corrections the ratio is **0.9908** in the top four directions, so Gauss–Newton
is the exact leading-order derivative and nothing is owed. It degrades to 1.9738 by the twelfth
direction, and that degradation is asserted as a *property*: it is where the $O(f^3)$ remainder
overtakes the eigenvalue it corrects, which is the same place `Jacobian_Rcond` already declines to
differentiate. The correction and the cutoff are talking about the same directions.

    Correcting the Hessian and leaving the cross term alone reproduces a spurious 3/2 — **1.4974**
    in the top four directions. That is the gate's own mutation, and it is how the factor was
    briefly believed to be a defect.

**The trace of the self-delta matrix counts the identified directions.** A benchmark held at its own
market number gives $dM_i/dq_j = \delta_{ij}\,\partial P_i/\partial q_i$ only where the model
reproduces every quote exactly *and* the inverse is a true inverse. Neither holds, and what replaces
the identity is not an approximation of it — it is the orthogonal projector onto the subspace the
pseudo-inverse kept. Its trace is therefore the **rank** of that projector, and it lands on the
integer to better than a thousandth. The individual diagonals run from 0.05 to 1.04 and must not be
asserted to be near one: the off-diagonal weight is scaled by the ratio of the two benchmarks' own
relative pricing errors, so a benchmark that fits ten times better than its neighbour has a row ten
times more sensitive to that neighbour's quote.

**The value chain is not what is broken.** Contract the reported factor greek with the displacement
the re-solve actually made and it reproduces the CVA that re-solve actually moved, better as the bump
shrinks. So `dV/dθ`, the attachment and the pricing chain between them are all right; what is not a
function of the quotes is `θ*` itself. Without that row a scattering ladder would be evidence against
everything at once.

!!! note "What replaces the ladder as the value-space reference"
    Step θ by $d\theta/dq \cdot h$ and re-price **without re-solving**. That is the move the quotes
    identify, carried through to a value, and nothing about the manifold or the optimizer's stopping
    point enters it. It lands at 1.0333 of the predicted change, and the mandated sign-flip mutation
    turns that into −0.9489 — so the gate that scores the mutation is a value-space gate, as briefed,
    even though the ladder it was meant to be is unavailable.

## Non-goals {#non-goals}

No SIMM aggregation or regulatory bucketing — bucketed quote deltas are the raw material and this
stops there. No wrong-way risk. **No recalibration inside the simulation**: quote sensitivities are
t0 risk, and future-dated dynamics stay on calibrated parameters. No new pricers, and no changes to
the `instruments.py` pricers beyond what the t0 closure strictly requires — so far, none. **No
vol-surface parameterisation**: SABR and SSVI are out of scope, and a swaption quote here is the
number on the `Instrument_Definitions` row, the surface's ATM read, or a premium — never a smile
parameter.

Two more, specific to what landed here. **No reporting format**: `dV/dq` lands on the quote leaves
in `Config.quote_leaves`, paired with each quote's descriptor, and no `Greeks_First`-style block is
emitted for it — `make_factor_index` reads a tenor grid off `all_factors`, and a quote is not a
factor. Note the [two shapes](#the-attachment) a consumer would have to honour. **No second
differentiation**: neither `CalibrationSolve.backward` nor `LeastSquaresSolve.backward` supports
`create_graph`, and the second refuses it explicitly — a Gauss–Newton contraction carries no second
derivative, so a quote-space Hessian off that node would be the curvature of a different problem.

**FX vol follows the same shape and is not built.** `GBMAssetPriceTSModelParameters` fits an
integrated vol curve to an ATM vol column, which is the same least-squares fixed point with a
smaller residual: the work is a quote leaf per column entry, the same `LeastSquaresSolve` contract,
and the same attachment through `factor_leaf`. Nothing on this page needs to change for it.
