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
`Liabilities`, to price them, and nothing else of it. Every knob the block declares is read there,
a stated value reaching the runtime under the knob's own name, and the runtime holds nothing the
block, its deals or the world do not name. **Held by**
`tests/test_action_space_and_artifact.py::test_the_runtime_reads_every_knob_the_hedging_problem_declares_and_none_it_does_not`.

The hedge legs, `Tradable_Instruments` less the cash accounts, are read in the order of their deals'
`Maturity_Date`, ties as stated: the order of `names['hedges']`, so of every per-leg axis
(`Active_Hedge_Indices`, a book, a CSV's columns) and of the consecutive maturities a calendar spread
matches (`turnover_charge`). **Held by**
`tests/test_action_space_and_artifact.py::test_the_hedge_legs_are_read_in_maturity_order_whatever_the_order_stated`.

Past that boundary the runtime **is** the contract — downstream code indexes it directly rather
than re-checking it. A malformed schedule, a partial spot history or a `Simulation_Batches` that
contradicts the mode fails there, naming the field, and a solver refuses at construction, by name, a
combination of declared values it cannot run. Before it the block is judged (`HedgeMonteCarlo.refusals`):
the top level by its declarations, and what `Hedging_Problem` states at every depth by
`schema.stated_messages`, so a menu value outside its field's values (`Accounting_Mode: 'Foo'`) or a
key no field declares is refused by name, as a calculation override is. **Held by**
`tests/test_action_space_and_artifact.py::test_the_hedging_problem_is_judged_at_every_depth`.

`Simulation_Batches` is the number of batches of `Batch_Size` paths simulated in either mode; its
field says what each mode does with them.

## Objective

`solve_hedge` chooses hedge positions $q$ to maximise expected utility of hedged wealth,

$$\max_q\ \mathbb{E}\big[u(x)\big],\qquad x=\frac{W_T-R}{c},\qquad
W_{t+1}=W_t+\sum_i q_i\,cs_i\,(F_{t+1,i}-F_{t,i})+(L_{t+1}-L_t),$$

where $W_t$ is the cumulative hedge P&L plus the marked liability $L_t$, $F_i$ the hedge
instrument's mark per unit of its deal's `Units` (`DealStructure.tensor_marks`), $cs_i$ its
`Contract_Size`, the one spelling of a contract, $R$ = `Reference_Wealth` and $c$
the utility scale. This frictionless law is what the solver fits; the environment adds financing,
margin, costs and rounding ([Bundle](bundle.md)). `Reference_Mode: Running_Wealth` applies $u$ to
each day's wealth increment instead of $W_T$ and sums the step utilities; `LogWealth`, whose
reward is the increment's growth, reads no `Reference_Mode`. A key is read under the mode its
description names and nowhere else, so stating it under another changes nothing (**Held by**
`tests/test_action_space_and_artifact.py::test_a_table_stated_outside_exposure_changes_nothing`,
`::test_a_reference_mode_stated_beside_logwealth_changes_nothing`).

The DP recursion lives in utility space, so `solve_hedge` requires one of these `Objective.Object`
values:

| `Objective.Object` | $u(x)$ | parameters |
| --- | --- | --- |
| `AsymmetricUtility_Symlog` | $\mathrm{sign}(x)\log(1+\lvert x\rvert)$ | none |
| `AsymmetricUtility_Huber` | $x-a\,h_\delta(x^-)-a^+h_\kappa(x^+)$, with $h_k(y)=y^2$ for $y\le k$ and $k^2+2k(y-k)$ beyond | `Huber_Aversion` $a$, `Huber_Delta` $\delta$, `Up_Aversion` $a^+$, `Up_Knee` $\kappa$ |
| `AsymmetricUtility_CARA` | $(1-e^{-\gamma x})/\gamma$ | `CARA_Gamma` $\gamma$ |
| `LogWealth` | per step $\log\big((C+W_{t+1})/(C+W_t)\big)$, summed; $c$ only in its floor | `Solver.DiffV2_Risk_Aversion`, in [0.1, 10] |

$x^-=\max(-x,0)$ and $x^+=\max(x,0)$. Under `LogWealth` the capital line is
$C=1.1\max(0,-\min W_{bank})/$`DiffV2_Risk_Aversion`, the worst wealth the training bank reaches;
below a ratio of 1e-3 the log continues linearly, it is floored at −20, a step launched from
$C+W_t$ at or under $10^{-4}c$ scores at most zero (ruin is absorbing), and a training run whose
worst bank wealth leaves no headroom is refused. The line is measured once, on warmup's bank; a
frozen roll builds no bank and reads its checkpoint's, at the checkpoint's aversion (the run's
`DiffV2_Risk_Aversion` is not read). Held by
`tests/test_action_space_and_artifact.py::test_a_frozen_roll_measures_no_frame_of_its_own`.

The scale $c$ (`Objective.Utility_Scale_Mode`, or the `Utility_Scale_Explicit` override):

| Setting | $c$ |
| --- | --- |
| `vol_scaled_notional` (default) | total leg volume × initial spot × annual vol × $\sqrt\tau$, $\tau$ the business days of the job's calendar from the first simulated row to the last settlement / 252, the market clock the vol is annualised on, a window's squared moves over the market days it holds (`tests/test_action_space_and_artifact.py::test_realised_vol_counts_the_market_days_its_window_holds`); spot and vol are batch medians at the start of the simulation, from the spot history or the calibrated process, the spot in the calculation `Currency` through its FX cross there (`HedgeMonteCarlo._fx`), as the returns state's t0 notional is; a degenerate input (no history and no calibrated vol, no volume, a non-positive vol) refused; the horizon held by `tests/test_action_space_and_artifact.py::test_the_utility_scale_counts_its_horizon_on_the_job_s_calendar` |
| `Utility_Scale_Explicit` | the stated amount of money, read under `vol_scaled_notional` |
| `conditional_sim` | per row to the close, the cross-sectional standard deviation of the flat-book wealth's move $L_t-L_0$ on the warmup batch, floored at `Utility_Scale_Floor_Frac` of its terminal entry; reads no `Utility_Scale_Explicit`, and refuses `Running_Wealth` except under `LogWealth` |

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
