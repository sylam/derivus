# Quote Sensitivities

!!! note "Increment 1, in progress"
    This page is written as the work lands. The **t0 benchmark closure**, its **graph audit**, the
    **plain multi-curve solver** and the `InterestRatePrices` family are built; the
    **implicit-function-theorem contract**, the factor-buffer attachment and the validation
    triangle are not, and are not specified here yet.

The autograd tape starts at *calibrated* factors today, so a greek is reported in zero-curve-node
space. Desks explain P&L in **quote** space — par swap rates, FRA strips, OIS quotes. This
workstream extends the tape one layer upstream, by owning the bootstrap inside the library and
differentiating through its fixed point, so one backward pass yields `dV/dq` alongside `dV/dθ`.

The chain is

```
q (float64 leaves)  ->  bootstrap solve  ->  θ*  ->  factor buffers  ->  scenarios  ->  V
```

and the only new arithmetic is the middle arrow. Everything downstream of θ* is the engine that
already exists.

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
| `utils.TensorSchedule.merged` | Copies the whole cashflow schedule across with `new_tensor` — notionals, accruals, margins **and the fixed rate**. This is where the **quote** stopped being differentiable. | **Closed.** `TensorSchedule.carry` gives the *tensor* half an optional per-column overlay; see [The quote side](#the-quote-side). θ never passed through this seam, so the θ-side closure is unchanged either way. |

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
`{column: tensor}` overlay to the **tensor half only**, spliced into the copy `new_tensor` makes.
Absent by default, so no caller outside a calibration knows it exists; and `carry` clears the
memoized dual, because `dual` and `merged` cache under one key and a plain caller must never be
handed a graph it did not ask for.

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

$$\text{column} = \text{base} + (q - \bar q)\,\frac{\partial \text{column}}{\partial q}$$

which is the [boundary correction](calc_lifecycle.md#boundary-corrections--the-sensitivity-subsystem)'s
shape and is there for the same reason: worth **exactly zero** in the forward pass, derivative one.
The tensor half is bit-identical to the plain copy, so enabling quote gradients cannot move a
single PV — asserted with `np.array_equal`, not a tolerance.

!!! warning "The overlay is a derivative carrier, not a reparameterisation"
    A moved quote needs a **fresh closure**. `pv_fixed_cashflows` memoizes its payment tensor in
    `Factor_dep`, built from the schedule's tensor half, so the first evaluation freezes whatever
    overlay was attached — removing the overlay afterwards does not remove the gradient. That is the
    third memo table in this subsystem, after `t_Buffer` and `CurveTenor`, and it is held in place
    by a gate rather than fixed.

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

## The precision seam {#the-precision-seam}

The bootstrap and its Jacobian are **float64 regardless of the simulation's precision**.
`BenchmarkInstruments.dtype` states it once, and `construct_bootstrapper`'s own `dtype` — float32 by
default — does not reach it. A solve that has to converge to 1e-10 cannot be done in float32, and
the Jacobian handed to the implicit function theorem is only as good as the residual it came from.
Setting that one attribute to float32 fails eight of the nine round-trip gates. Where θ\* crosses
back into the simulation it is cast to the cube's dtype; that boundary is specified with the IFT
contract, which is not built yet.

## Non-goals {#non-goals}

No SIMM aggregation or regulatory bucketing — bucketed quote deltas are the raw material and this
stops there. No wrong-way risk. **No recalibration inside the simulation**: quote sensitivities are
t0 risk, and future-dated dynamics stay on calibrated parameters. No new pricers, and no changes to
the `instruments.py` pricers beyond what the t0 closure strictly requires — so far, none.
