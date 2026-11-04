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
positions outside the fitted function. Under `DiffV2_Label_Horizon` k > 1 the value label gains a
correction on the outer path, the k-step bootstrap along the fitted greedy policy (each book chosen
on its step's cached draws, the outer move realised less its repositioning charge under
`DiffV2_Position_State`, with its reward under `Running_Wealth`) less the one-step one, k clipped
at the decision horizon, so a label past the horizon is the utility the greedy policy realises on
the training paths up to its first step's cross-fit; the gradient labels stay the one-step ones,
and k = 1, the default, is the fit as before. Held by
`tests/test_action_space_and_artifact.py::test_a_label_horizon_of_one_is_the_one_step_fit` and
`::test_a_label_past_the_horizon_is_what_its_greedy_policy_realises`.
The verdict's `textbook` hedge is a one-step
minimum-variance hedge ratio, a simplification of the discrete-time quadratic hedging of [@follmer1988; @schweizer1995]:
unconditional across the paths at each step rather than conditional on the state, and per
instrument rather than joint.

The value net's market columns are each simulated factor's revealed state (`reveal_state_at`): a `BasisLinkedSpotModel`'s include its deviation from its slow level where it has one and its log variance under its own GARCH, so a checkpoint trained on another law is refused by the `md` shape check. Held by `tests/test_action_space_and_artifact.py::test_the_value_net_reads_the_band_deviation_the_world_declares`.

Every revealed market column is labelled with its pathwise slope in the coordinates the value net reads it. The gradient fork is seeded from leaves at the coordinates a process reveals beside its level (`diff_state_leaves()`): the spot GARCH's $\log h_t$, and the basis's deviation $D=b-\mu$ and $\log\sigma^2$, $\mu$ the level leaf less $D$, so the deviation's label is $-\partial Y/\partial\mu$ and the level's $\partial Y/\partial b+\partial Y/\partial\mu$, its slope at fixed deviation; under `DiffV2_Returns_State` each label is carried through the column's map (`_reveal_transform`), a price column's times the price, a basis level's and deviation's times the t0 spot, and $\log\sigma^2$'s as it is. The wealth step differences the fork's own futures marks, $F_{t+1}$ and $F_t$ both priced off the leaves, so a column the marks read (the carry, the price, the basis level) is labelled with the slope of the move. A leaf the label does not reach has slope 0, and a column no leaf measures (a factor missing a state leaf) is left to the value term. Held by `tests/test_action_space_and_artifact.py::test_every_revealed_column_is_labelled_in_the_coordinates_the_net_reads` for the basis and `tests/test_action_space_and_artifact.py::test_the_wealth_step_differences_the_fork_s_own_futures_marks` for the marks; the spot GARCH's leaf is Not gated.

Under `DiffV2_Returns_State` the value state is dimensionless: price columns as log-returns against the t0 spot, the basis level and its deviation as fractions of it (its $\log\sigma^2$ as it is), and wealth as a fraction of the t0 book notional. The capital line of `LogWealth` travels with that frame (below); the utility scale of the other objectives is restored in the training world's dollars, a limitation of their frame. The wealth column has one spelling, `DiffSolver._w_norm`, on every path a net is fitted or read on, so the fit never feeds dollars to a fraction frame. Held by `tests/test_action_space_and_artifact.py::test_the_returns_state_fits_the_wealth_column_it_reads` and, for the basis, `::test_the_value_net_reads_the_band_deviation_the_world_declares`.

The capital line travels with that frame: a checkpoint stamps the notional its frame divides wealth by (`state_notional`) and a roll scales each member's displacement by its own notional over it (`DiffSolver._capital_line`), so the day's growth is read against the pot the nets were fitted with whatever the price level; a returns-state checkpoint without the stamp is refused. Held by `tests/test_action_space_and_artifact.py::test_the_capital_line_travels_with_the_returns_state_frame`.

An ensemble (`DiffV2_Load_Value_Fn` listing several checkpoints) scores every candidate book as the mean over members of each member's own value: its residual net in its own standardisation frame and, under `LogWealth`, the day's growth at its own capital line (`DiffSolver._reward`), so the greedy book does not depend on the order the checkpoints are listed in; its drift tripwire's null sigma is the mean of the members' stamped nulls. Held by `tests/test_action_space_and_artifact.py::test_an_ensemble_scores_each_member_in_its_own_frame`.

A fit is held to what its own argmax earns on the held-out batch: a fit whose $V_0$ exceeds the held-out greedy `u_mean`, net of the repositioning charge under `DiffV2_Position_State` as $V_0$ is, by more than `DiffV2_Max_Value_Optimism` (default 1.0) writes no checkpoint and is refused by name with both numbers, since a bootstrap that outruns its own policy has either fed a slope no label supervises back into $V_0$ or ruins out of sample. A checkpoint stamps that gap (`value_optimism`) and its swept mean book per decision step (`q_star_path`); a roll refuses a member stamped above its declared bound, and an ensemble any two of whose members' swept net books lie more than `DiffV2_Max_Member_Disagreement` contracts apart on average (default 8), by name. A checkpoint saved before the stamps loads as before. On the platinum record the healthy members read 0.04 to 0.62 nats of optimism and the three diverged 1.40 to 2.05; the PM-frame pairs agree within 1.7 to 5.2 contracts. Held by `tests/test_action_space_and_artifact.py::test_a_fit_whose_value_outruns_its_held_out_utility_saves_nothing` and `::test_an_ensemble_whose_members_disagree_beyond_the_declaration_is_refused`.

The repositioning charge is on the move from the standing book, as the step that set it left it, to the new book on the live legs, in the argmax, the label and the k-step correction alike, so a position on a leg that dies at the step pays its close there, as the roll and the verdict charge it. Held by `tests/test_action_space_and_artifact.py::test_the_label_charges_the_close_of_a_leg_that_dies_at_the_step` for the label and the correction; the argmax's spelling, which reaches a ranking only where a leg dies beside a live one, is Not gated.

Under `DiffV2_Position_State` the value net reads the standing book $p=\sum q/Q_{max}$, a control the bank samples rather than a simulated coordinate, so the gradient fork measures no slope in it and the value term alone fits that column. Left so, the slope is free: on the platinum AM frame the argmax harvested the nets' own extrapolated premium at the floor of the book, every label that chose the floor then valued it above the bank's book, and the bootstrap ratcheted the premium into $V_0$ (2.05 nats against a held-out 0.003 on one member, the fitted premium 40 times the repositioning charge's slope). `DiffV2_Book_Gradient_Label` 'Yes' makes the standing book a leaf of the fork: by the envelope theorem a label's slope in it is the repositioning charge's at the chosen action, taken along the ladder (a move in $p$ lands on the live legs as the step's `Allocation_Weights` split it, evenly where none is declared, so the gradient in a leg the ladder never moves carries no weight, though under the default bank the books it holds on such legs still shape the charge's slope on the others through the calendar matching; absent a table the live active legs weigh evenly, and a swept step where none can carry a contract labels zero, as the passive holds nothing there), and the fit matches $\partial A/\partial p$ to it beside the wealth and market slopes, normalised by its own variance; each fitted step reports both bank means (`book_slope`, `book_label`). 'No', the default, is the fit without it. Under `DiffV2_Bank_Along_Ladder` 'Yes' the `DiffSolverV2` bank holds only books the policy can: each constructed net spans the ladder's band and is held as its rung (`HedgeActionSpace.ladder`, `HedgeActionSpace.rung`) on the legs the policy's ladder calls live, the inner fork's expired-contract guard at the step (a step before `T_Min` forked for that mask alone); its reserve is priced on the ladder's weights, and its exploration is drawn on the net, `DiffV2_Bank_Noise_Frac` of the band's width, clamped into the band and held as a rung, so no bank book meets a calendar match the policy's books never meet. Absent a table the ladder is the even split over the live active legs, the label's direction, while the Cartesian policy may hold any grid book, so there the bank covers one direction of what the policy can hold. 'No', the default, spans `net_bounds` and water-fills the bank's net onto the replication hedge's shape over the active legs. Held by `tests/test_action_space_and_artifact.py::test_the_book_column_is_fitted_to_the_repositioning_charge_s_slope` and `::test_the_book_label_is_taken_along_the_ladder`; the no-table step by `::test_a_no_table_document_whose_legs_all_die_still_labels`; the ladder bank by `::test_the_bank_is_composed_along_the_ladder`, its live legs by `::test_the_ladder_bank_reads_the_policy_s_live_legs_on_a_replayed_path`.

The stepper roll's greedy policy holds the passive book from the first decision day on or after `DiffV2_Passive_From` and the fitted argmax before it, so a trip rule read outside the engine, such as a ledger's realised-vol switch, is read on the engine's own accounting rather than on a replay; the recorded book (`greedy_q_traj` on `greedy_q_dates`) shows the handover, and the drift monitor reads every decision day as before. Held by `tests/test_action_space_and_artifact.py::test_the_stepper_hands_the_greedy_book_to_the_passive_from_the_declared_day`.

A policy trained inside a `Total_Position_Schedule` corridor rolls in that corridor and in any corridor that lies inside it on every decision step, since a narrower fence only restricts the roll to learned support; a wider, different or absent corridor is refused by name, and a corridor-free policy rolls in any. Held by `tests/test_action_space_and_artifact.py::test_a_corridor_trained_policy_rolls_inside_its_own_corridor` and `::test_a_corridor_trained_policy_rolls_in_its_fence_and_loads_only_into_it`.

A saved checkpoint carries the drift monitor's null sigma the training verdict measured, since the policy artifact is built after that verdict, and a frozen roll's tripwire (`DiffV2_Drift_Threshold_Sigmas` null sigmas times the square root of the day count) is scaled by it. An ensemble's reported objective is the mean over members of each member's growth at its own capital line (`DiffSolver._reward`), the same reading the decision uses. Held by `tests/test_action_space_and_artifact.py::test_the_saved_checkpoint_carries_the_drift_null_the_verdict_measured`.

The drift monitor scores each decision's one-step residual, the fork's mean forecast less the market's move, at the next decision, so a trip, the printed `drift_z` and, under `DiffV2_Drift_Beta`, the fork's shift at a decision read only prices printed before it; the first decision scores nothing and the last decision's residual is never scored, which is what makes a to-date roll's reading the same as the record's. A decision is a row of market time, one carrying a business day of the job's calendar since the row before (`utils.market_days`, the clock `utils.trading_clock` hands the laws), and holds its book to the next one (`Bundle.successors`), in the sweep, the verdict, the bank and the stepper roll alike, and its fork lands there with that interval's market time, so a Friday's residual is its fork's Friday-to-Monday forecast against Monday's move, and every market move is scored once; on a grid coarser than daily a Saturday row carrying its Friday decides and a Sunday row carrying nothing does not. Held by `tests/test_action_space_and_artifact.py::test_the_drift_monitor_scores_a_decision_only_once_its_move_has_printed`, `::test_a_friday_decision_forks_to_monday_and_is_scored_against_monday`, `::test_a_decision_is_a_row_of_market_time_on_a_coarse_grid` and `tests/test_logvar2fj_spot.py::test_a_fork_starts_from_the_outer_path_s_state`.

Under `DiffV2_First_Decision_From` 'Passive' the first greedy decision, the verdict's and the stepper roll's alike (`DiffSolver._first_book`), is measured from the passive book at that step, so the trade band and the position state ride the hedge a desk books at inception and the policy's deviation enters through the band; the accounting is untouched, every track entering from the declared opening book at that day's price and the verdict charging the first move from it. Held by `tests/test_action_space_and_artifact.py::test_the_first_decision_can_be_measured_from_the_passive_book` and `::test_the_verdict_and_the_stepper_close_one_replayed_book_to_one_wealth`.

Under `DiffV2_Risk_Horizon` k > 1 the downside semideviation `DiffV2_Risk_Kappa` penalises is the score's at the decision's k-th successor or the terminal before it (`DiffSolver.landing`): the candidate held there and charged as its one-step wealth is, its continuation read at the book fraction it leaves standing on the legs of the decision before that row, as `_fit_step` fits it, plus under `Running_Wealth` the reward of each decision's move on the fork's own marks, the sum the DP fits; in the roll, the verdict and the sweep's labels alike, the mean, the incumbent's move test and kappa being a horizon of one's. That fork (`DiffSolver._ahead`) draws on streams of its own seeded by the batch's seed and the decision (`CMC_State.own_stream`), so it moves no other draw, and draws its inner paths iid whatever `Inner_Draws` states: one Sobol stream strided over a window's rows hands an inner path nearly the same point on each, which read a constant-variance spot's five-day semideviation at 2.19 √5 times its one-day one. On a `DiffV2_Drift_Beta` trip the stepper roll shifts that fork as it shifts the one-step one, each row's move by the declared drift over the decision intervals to it each leg moves over (`DiffSolver._moving`, the drift monitor's live and still-dispersing legs), so an expired leg is held still and a book parked on it, scored on the live book, earns nothing; held by `::test_a_declared_drift_leaves_an_expired_leg_still`, the per-row count of the risk fork's moving legs Not gated. Held by `tests/test_action_space_and_artifact.py::test_the_risk_horizon_s_semideviation_grows_as_its_root_on_a_constant_variance`, `::test_the_sweep_s_labels_score_the_risk_horizon`, `::test_a_variance_above_its_mean_holds_the_risk_horizon_below_its_root`, `::test_a_risk_horizon_past_the_terminal_lands_on_it`, `::test_a_running_reward_over_the_risk_horizon_sums_its_decisions`, `::test_a_risk_horizon_leaves_the_held_out_world_alone` and `::test_a_declared_drift_shifts_the_risk_horizon_s_fork`; **Not gated**: the landing's leg mask under `DiffV2_Position_State`, which no cheap document makes observable, a fitted net's slope in the book fraction being what it moves.

## Benchmarks and the verdict

| Track | Policy | Reported in |
| --- | --- | --- |
| `HindsightDpSolver` (`Run_Hindsight_Diagnostic`) | clairvoyant and frictionless: at every decision the grid position maximising its realized P&L to the next, the decision rows and the terminal the solver steps; an upper bound on any deployable policy | `comparison`, ladder |
| `textbook` (`Run_Textbook_Benchmark`) | the single best CONSTANT position held to the terminal, the last live liability mark, frictionless, chosen by $E[u]$ on the held-out paths; a lower bound | `comparison`, ladder |
| `textbook` (verdict) | per step, per active hedge $i$: $q_i=-\mathrm{Cov}(\Delta L,\Delta F_i)\,/\,(cs_i\,\mathrm{Var}(\Delta F_i)\,n_{active})$ over the outer paths, booked as the passive book is | `diagnostics.verdict`, `stepper_verdict` |
| `nohedge` | no position from the first decision, the opening book closed there at the declared turnover cost | `diagnostics.verdict`, `stepper_verdict` |
| `passive` | per step, minus the liability's quantity still to fix after the step, each fixing weighted as the deal weights it, split across the live legs by `Allocation_Weights` (equally where none is declared), booked as the execution books it (clamped to the position limits of the legs the step's `Allocation_Weights` knot weighs, the live active legs where none is declared, nothing on any other, its net on those legs inside the cap and the corridor snapped inward to whole contracts, apportioned to whole contracts carrying that net) and charged the declared turnover cost as the greedy policy is; against a liability stating no fixings, any type but `CommodityAveragePriceSwapDeal`, it holds nothing, or the nearest whole-contract edge of a corridor that excludes zero | `diagnostics.verdict`, `stepper_verdict` |

The passive book stays inside its band, in whole contracts inside each leg's box with nothing on a
leg the knot weighs zero, its net inside the corridor, a fractional edge included, and inside the
cap on the signed net; the verdict and the stepper roll hold one textbook book, and on the
Cartesian grid, whose levels are `linspace(Min_Position, Max_Position, levels)` rounded to whole
contracts, one greedy book. Each step's grid keeps the rows whose signed net on the step's live
legs lies inside the cap and the corridor (`HedgeActionSpace.grid_at`), whatever their gross, so a
row parked on an expired leg cannot carry the live legs past the cap, and `Decision_Deadband_Sigma`
holds the standing book only while its live net lies inside them (`HedgeActionSpace._inside`). Held by
`tests/test_action_space_and_artifact.py::test_the_passive_book_stays_inside_a_fractional_corridor`,
`::test_the_verdict_and_the_stepper_hold_one_textbook_book`,
`::test_the_verdict_and_the_stepper_hold_one_greedy_book_on_the_cartesian_grid`,
`::test_the_cartesian_grid_caps_the_signed_net_of_a_mixed_sign_book` and
`::test_the_cap_holds_the_book_the_live_legs_carry`. The constant hold telescopes to the
terminal, where the hindsight bound stops too, held by
`tests/test_action_space_and_artifact.py::test_the_textbook_bound_holds_its_book_to_the_terminal`.

The two tracks named `textbook` are different policies. Both report turnover cost as a diagnostic;
the comparison track is never charged it, the verdict's only under `DiffV2_Position_State`. The
hindsight bound steps the solver's decision rows (`Bundle.decision_steps`), held by
`tests/test_action_space_and_artifact.py::test_the_hindsight_bound_decides_where_the_solver_does`.

The verdict rolls the fitted argmax policy over every path of the held-out batch, flat at `T_Min`,
beside the verdict's `textbook`, `nohedge` and `passive`, and reports per policy $E[u]$ (`u_mean`) and the
mean, 5th percentile and CVaR at 5% of terminal wealth (`wT_mean`, `wT_p5`, `wT_cvar5`, the last
the mean of the paths at or below the 5th percentile). It flags `verdict_beats_nohedge_on_utility`
and `verdict_beats_textbook_on_utility` (greedy `u_mean` at least the other's) and
`verdict_tail_competitive_vs_textbook` (greedy CVaR5 no worse than the textbook's less 5% of its
magnitude). Each traded policy pays turnover cost on the benchmarks' rule, entry from the opening
book, every reposition and the unwind under `Force_Flat_At_End` (`turnover_cost_mean`). Without
`DiffV2_Position_State` its wealth runs cost-free and the `*_net` columns subtract the cost beside
it. Under it every step's wealth is charged the repositioning charge the labels are fitted net of
(`DiffSolver._reposition_charge`), so each greedy decision reads the net wealth its value was
fitted on, the columns are net and there are no `*_net` columns. Held by
`tests/test_action_space_and_artifact.py::test_a_label_past_the_horizon_is_what_its_greedy_policy_realises`.
The ladder `hindsight >= solver >= textbook` on $V_0$ holds within 1e-6; under
`Reference_Mode = Running_Wealth` the solver's $V_0$ is a sum of step utilities, so the ladder's
order is not an acceptance statement and the run logs a warning.

The stepper decides on the solver's decision rows (`Bundle.decision_steps`), the first row whatever
its day and no row at or past the terminal, and the stepper roll closes every book where the verdict
does, at the terminal, the last live liability mark, or under `Force_Flat_At_End` 'No' values it
there, a hedge leg marked at the terminal past it (`BundleStepper._values_at`, and the CSV alike),
so on one replayed path at zero rates the two close one book to one wealth; every track holds the
opening book from the first row to the first
decision in both, the no-hedge track closing it there; at non-zero rates the stepper also earns the
daily interest on cash and margin, the realised accounting the verdict's wealth law leaves out.
Held by
`tests/test_action_space_and_artifact.py::test_the_verdict_and_the_stepper_close_one_replayed_book_to_one_wealth`,
`::test_the_verdict_and_the_stepper_carry_one_opening_book`,
`::test_the_stepper_closes_a_book_held_at_every_decision_at_the_terminal` and
`::test_a_first_row_of_no_market_time_is_one_decision_for_the_verdict_and_the_stepper`.
The stepper's diagnostic CSV (`BundleStepper.write_diagnostic_csvs`) books each trade on the row at
whose mark it trades, the forced close on the terminal whatever the row before it, outright at the
terminal's mark and vol as the stepper and the verdict close a book, so at zero rates its last
`hedge_portfolio_ex_funding` is the stepper's hedge P&L. Held by
`tests/test_action_space_and_artifact.py::test_the_diagnostic_csv_books_the_forced_close_on_the_terminal`.
An opening book (`Portfolio_State.Positions`) is marked at the first row's prices, so the stepper's
P&L since inception and the CSV's both run from that row. Held by
`tests/test_action_space_and_artifact.py::test_an_opening_book_s_pnl_runs_from_the_first_row_in_the_stepper_and_its_csv`.
