# Hedging Overview

`Execution_Mode` selects one of two verbs on the same simulated world:

- **`simulate_only`** — roll the bundle forward with zero trades and report the terminal P&L
  summary, the unhedged baseline. Callers drive their own policy on top via
  `HedgeRuntimeExecutionResult.create_stepper()`.
- **`solve_hedge`** — fit a value function and report a verdict against benchmark tracks.

## One boundary

`construct_hedge_runtime` (`hedge_bundle.py`) reads and validates the completed `Hedging_Problem`
block into the normalized **runtime** every consumer indexes by key: canonical lowercased modes,
the instrument / cash-account / hedge name sets, per-instrument metadata, the accounting rules
(position limits, turnover cost, spreads, margin funding, corridor), the objective, the solver
config, the portfolio state, and the burn-in and inner-belief switches the calculation and a
process read; the calculation reads the block's deal containers, `Tradable_Instruments` and
`Liabilities`, to price them, and nothing else of it.

Past that boundary the runtime **is** the contract — downstream code indexes it directly rather
than re-checking it. A malformed schedule, a partial spot history or a `Simulation_Batches` that
contradicts the mode fails there, naming the field (an `Execution_Mode` outside its two values is
refused where the block is judged, before it), and a solver refuses at construction, by name, a
combination of declared values it cannot run.

`Simulation_Batches` is the number of batches of `Batch_Size` paths simulated in either mode; its
field says what each mode does with them.

## Objective

`solve_hedge` chooses hedge positions $q$ to maximise expected utility of hedged wealth,

$$\max_q\ \mathbb{E}\big[u(x)\big],\qquad x=\frac{W_T-R}{c},\qquad
W_{t+1}=W_t+\sum_i q_i\,cs_i\,(F_{t+1,i}-F_{t,i})+(L_{t+1}-L_t),$$

where $W_t$ is the cumulative hedge P&L plus the marked liability $L_t$, $F_i$ the hedge
instrument's mark, $cs_i$ its contract size, $R$ = `Reference_Wealth` and $c$
the utility scale. This frictionless law is what the solver fits; the environment adds financing,
margin, costs and rounding ([Bundle](bundle.md)). `Reference_Mode: Running_Wealth` applies $u$ to
each day's wealth increment instead of $W_T$ and sums the step utilities.

The DP recursion lives in utility space, so `solve_hedge` requires one of these `Objective.Object`
values:

| `Objective.Object` | $u(x)$ | parameters |
| --- | --- | --- |
| `AsymmetricUtility_Symlog` | $\mathrm{sign}(x)\log(1+\lvert x\rvert)$ | none |
| `AsymmetricUtility_Huber` | $x-a\,h_\delta(x^-)-a^+h_\kappa(x^+)$, with $h_k(y)=y^2$ for $y\le k$ and $k^2+2k(y-k)$ beyond | `Huber_Aversion` $a$, `Huber_Delta` $\delta$, `Up_Aversion` $a^+$, `Up_Knee` $\kappa$ |
| `AsymmetricUtility_CARA` | $(1-e^{-\gamma x})/\gamma$ | `CARA_Gamma` $\gamma$ |
| `LogWealth` | per step $\log\big((C+W_{t+1})/(C+W_t)\big)$, summed; no $c$ | `Solver.DiffV2_Risk_Aversion`, in [0.1, 10] |

$x^-=\max(-x,0)$ and $x^+=\max(x,0)$. Under `LogWealth` the capital line is
$C=1.1\max(0,-\min W_{bank})/$`DiffV2_Risk_Aversion`, the worst wealth the training bank reaches;
below a ratio of 1e-3 the log continues linearly, it is floored at −20, and a training run whose
worst bank wealth leaves no headroom is refused. The line is measured once, on warmup's bank; a
frozen roll builds no bank and reads its checkpoint's, at the checkpoint's aversion (the run's
`DiffV2_Risk_Aversion` is not read). Held by
`tests/test_action_space_and_artifact.py::test_a_frozen_roll_measures_no_frame_of_its_own`.

The scale $c$ (`Objective.Utility_Scale_Mode`, or the `Utility_Scale_Explicit` override):

| Setting | $c$ |
| --- | --- |
| `vol_scaled_notional` (default) | total leg volume × initial spot × annual vol × $\sqrt\tau$, $\tau$ the business days of the job's calendar from the first simulated row to the last settlement / 252, the market clock the vol is annualised on, a window's squared moves over the market days it holds (`tests/test_action_space_and_artifact.py::test_realised_vol_counts_the_market_days_its_window_holds`); spot and vol are batch medians at the start of the simulation, from the spot history or the calibrated process; a degenerate input (no history and no calibrated vol, no volume, a non-positive vol) refused; the horizon held by `tests/test_action_space_and_artifact.py::test_the_utility_scale_counts_its_horizon_on_the_job_s_calendar` |
| `Utility_Scale_Explicit` | the stated currency amount |
| `conditional_sim` | per row to the close, the cross-sectional standard deviation of the flat-book wealth's move $L_t-L_0$ on the warmup batch, floored at `Utility_Scale_Floor_Frac` of its terminal entry; refuses `Utility_Scale_Explicit` beside it, and `Running_Wealth` except under `LogWealth` |

$c$ is part of the solver's frame, locked at warmup, a loaded checkpoint evaluating in its own
([Solver](solver.md), `DiffSolver`). Under `conditional_sim` the measured schedule sets the level of
$c$, not only its shape, so arming it re-doses every shape parameter, each in units of $c$:
`Huber_Aversion` is then dosed against the measured scale, and the frame lock logs the absolute
risk aversion per unit of money it lands on.

## Privileged factors

What each stochastic process publishes as market state is a naming convention owned here:
`derive_privileged_layout` asks every live process what it emits, and multi-commodity runs prefix
the attribute with the factor name to disambiguate. Adding a process with its own privileged
surface flows through automatically — no registration step.
