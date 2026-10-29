# Solver

`Solver.Object` is `DiffSolver`, the differential-ML value-function solver, or `DiffSolverV2`, a
separate solver that adds a forward pass constructing the policy its training bank rolls (see
`DiffSolverV2` below); a checkpoint written by one is refused by the other. `HindsightDpSolver` is
not a primary solver: it runs as the clairvoyant benchmark track (`Run_Hindsight_Diagnostic`).

A solve is a **stream**: `StreamingSolve` takes one bundle per simulation batch as the calc builds
it, fits on all but the last, and reports the verdict on the last — a world no fit step saw. Every
track (the solver, the benchmarks, the stepper rollout) optimizes over the same
`HedgeActionSpace`, so they trade the same positions and pay the same frictions.

## Method

The problem is utility-optimal hedging of a liability under trading frictions on simulated paths,
the setting of deep hedging [@buehler2019]. The method is not [@buehler2019]'s end-to-end policy network: `DiffSolver`
runs a backward dynamic program whose continuation value at each decision step is regressed on the
cross-section of simulated states, the regression Monte Carlo of [@longstaff2001], with a neural network fitted
by the differential loss of [@huge2020] (value plus pathwise gradient). Its labels are inner Monte Carlo
estimates of the next step's continuation, and the Bellman maximum is a grid search over target
positions outside the fitted function. The verdict's `textbook` hedge is a one-step
minimum-variance hedge ratio, a simplification of the discrete-time quadratic hedging of [@follmer1988; @schweizer1995]:
unconditional across the paths at each step rather than conditional on the state, and per
instrument rather than joint.

Under `DiffV2_Returns_State` the value state is dimensionless: price columns as log-returns against the t0 spot, basis columns as fractions of it, and wealth as a fraction of the t0 book notional. The capital line of `LogWealth` travels with that frame (below); the utility scale of the other objectives is restored in the training world's dollars, a limitation of their frame. The wealth column has one spelling, `DiffSolver._w_norm`, on every path a net is fitted or read on, so the fit never feeds dollars to a fraction frame. Held by `tests/test_action_space_and_artifact.py::test_the_returns_state_fits_the_wealth_column_it_reads`.

The capital line travels with that frame: a checkpoint stamps the notional its frame divides wealth by (`state_notional`) and a roll scales each member's displacement by its own notional over it (`DiffSolver._capital_line`), so the day's growth is read against the pot the nets were fitted with whatever the price level; a returns-state checkpoint without the stamp is refused. Held by `tests/test_action_space_and_artifact.py::test_the_capital_line_travels_with_the_returns_state_frame`.

An ensemble (`DiffV2_Load_Value_Fn` listing several checkpoints) scores every candidate book as the mean over members of each member's own value: its residual net in its own standardisation frame and, under `LogWealth`, the day's growth at its own capital line (`DiffSolver._reward`), so the greedy book does not depend on the order the checkpoints are listed in; its drift tripwire's null sigma is the mean of the members' stamped nulls. Held by `tests/test_action_space_and_artifact.py::test_an_ensemble_scores_each_member_in_its_own_frame`.

The stepper roll's greedy policy holds the passive book from the first decision day on or after `DiffV2_Passive_From` and the fitted argmax before it, so a trip rule read outside the engine, such as a ledger's realised-vol switch, is read on the engine's own accounting rather than on a replay; the recorded book (`greedy_q_traj` on `greedy_q_dates`) shows the handover, and the drift monitor reads every decision day as before. Held by `tests/test_action_space_and_artifact.py::test_the_stepper_hands_the_greedy_book_to_the_passive_from_the_declared_day`.

A policy trained inside a `Total_Position_Schedule` corridor rolls in that corridor and in any corridor that lies inside it on every decision step, since a narrower fence only restricts the roll to learned support; a wider, different or absent corridor is refused by name, and a corridor-free policy rolls in any. Held by `tests/test_action_space_and_artifact.py::test_a_corridor_trained_policy_rolls_inside_its_own_corridor` and `::test_a_corridor_trained_policy_rolls_in_its_fence_and_loads_only_into_it`.

A saved checkpoint carries the drift monitor's null sigma the training verdict measured, since the policy artifact is built after that verdict, and a frozen roll's tripwire (`DiffV2_Drift_Threshold_Sigmas` null sigmas times the square root of the day count) is scaled by it. An ensemble's reported objective is the mean over members of each member's growth at its own capital line (`DiffSolver._reward`), the same reading the decision uses. Held by `tests/test_action_space_and_artifact.py::test_the_saved_checkpoint_carries_the_drift_null_the_verdict_measured`.

Under `DiffV2_First_Decision_From` 'Passive' the stepper roll's first greedy decision is measured from the passive book at that step, so the trade band and the position state ride the hedge a desk books at inception and the policy's deviation enters through the band; the accounting is untouched, every track entering from the declared opening book at that day's price. Held by `tests/test_action_space_and_artifact.py::test_the_first_decision_can_be_measured_from_the_passive_book`.

## Benchmarks and the verdict

| Track | Policy | Reported in |
| --- | --- | --- |
| `HindsightDpSolver` (`Run_Hindsight_Diagnostic`) | clairvoyant and frictionless: at every step the grid position maximising that step's realized P&L; an upper bound on any deployable policy | `comparison`, ladder |
| `textbook` (`Run_Textbook_Benchmark`) | the single best CONSTANT position held to the horizon, frictionless, chosen by $E[u]$ on the held-out paths; a lower bound | `comparison`, ladder |
| `textbook` (verdict) | per step, per active hedge $i$: $q_i=-\mathrm{Cov}(\Delta L,\Delta F_i)\,/\,(cs_i\,\mathrm{Var}(\Delta F_i)\,n_{active})$ over the outer paths, clamped to the position limits and projected into the corridor | `diagnostics.verdict` |
| `nohedge` | no position | `diagnostics.verdict` |
| `passive` | per step, minus the liability's quantity still to fix after the step, each fixing weighted as the deal weights it, split across the live legs by `Allocation_Weights` (equally where none is declared) in contracts, clamped to the position limits, projected into the corridor and charged the declared turnover cost as the greedy policy is; against a liability stating no fixings, any type but `CommodityAveragePriceSwapDeal`, it holds nothing | `diagnostics.verdict`, `stepper_verdict` |

The two tracks named `textbook` are different policies. Both benchmarks report turnover cost as a
diagnostic and are never charged it.

The verdict rolls the fitted argmax policy over every path of the held-out batch, flat at `T_Min`,
beside the verdict's `textbook`, `nohedge` and `passive`, and reports per policy $E[u]$ (`u_mean`) and the
mean, 5th percentile and CVaR at 5% of terminal wealth (`wT_mean`, `wT_p5`, `wT_cvar5`, the last
the mean of the paths at or below the 5th percentile). It flags `verdict_beats_nohedge_on_utility`
and `verdict_beats_textbook_on_utility` (greedy `u_mean` at least the other's) and
`verdict_tail_competitive_vs_textbook` (greedy CVaR5 no worse than the textbook's less 5% of its
magnitude). The ladder `hindsight >= solver >= textbook` on $V_0$ holds within 1e-6; under
`Reference_Mode = Running_Wealth` the solver's $V_0$ is a sum of step utilities, so the ladder's
order is not an acceptance statement and the run logs a warning.
