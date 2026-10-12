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
the setting of deep hedging [1]. The method is not [1]'s end-to-end policy network: `DiffSolver`
runs a backward dynamic program whose continuation value at each decision step is regressed on the
cross-section of simulated states, the regression Monte Carlo of [2], with a neural network fitted
by the differential loss of [3] (value plus pathwise gradient). Its labels are inner Monte Carlo
estimates of the next step's continuation, and the Bellman maximum is a grid search over target
positions outside the fitted function. The verdict's `textbook` hedge is a one-step
minimum-variance hedge ratio, a simplification of the discrete-time quadratic hedging of [4, 5]:
unconditional across the paths at each step rather than conditional on the state, and per
instrument rather than joint.

## Benchmarks and the verdict

| Track | Policy | Reported in |
| --- | --- | --- |
| `HindsightDpSolver` (`Run_Hindsight_Diagnostic`) | clairvoyant and frictionless: at every step the grid position maximising that step's realized P&L; an upper bound on any deployable policy | `comparison`, ladder |
| `textbook` (`Run_Textbook_Benchmark`) | the single best CONSTANT position held to the horizon, frictionless, chosen by $E[u]$ on the held-out paths; a lower bound | `comparison`, ladder |
| `textbook` (verdict) | per step, per active hedge $i$: $q_i=-\mathrm{Cov}(\Delta L,\Delta F_i)\,/\,(cs_i\,\mathrm{Var}(\Delta F_i)\,n_{active})$ over the outer paths, clamped to the position limits and projected into the corridor | `diagnostics.verdict` |
| `nohedge` | no position | `diagnostics.verdict` |

The two tracks named `textbook` are different policies. Both benchmarks report turnover cost as a
diagnostic and are never charged it.

The verdict rolls the fitted argmax policy over every path of the held-out batch, flat at `T_Min`,
beside the verdict's `textbook` and `nohedge`, and reports per policy $E[u]$ (`u_mean`) and the
mean, 5th percentile and CVaR at 5% of terminal wealth (`wT_mean`, `wT_p5`, `wT_cvar5`, the last
the mean of the paths at or below the 5th percentile). It flags `verdict_beats_nohedge_on_utility`
and `verdict_beats_textbook_on_utility` (greedy `u_mean` at least the other's) and
`verdict_tail_competitive_vs_textbook` (greedy CVaR5 no worse than the textbook's less 5% of its
magnitude). The ladder `hindsight >= solver >= textbook` on $V_0$ holds within 1e-6; under
`Reference_Mode = Running_Wealth` the solver's $V_0$ is a sum of step utilities, so the ladder's
order is not an acceptance statement and the run logs a warning.

## References

1. Buehler, H.; Gonon, L.; Teichmann, J.; Wood, B. "Deep hedging". *Quantitative Finance* 19(8),
   1271-1291, 2019. [doi:10.1080/14697688.2019.1571683](https://doi.org/10.1080/14697688.2019.1571683)
2. Longstaff, F. A.; Schwartz, E. S. "Valuing American Options by Simulation: A Simple
   Least-Squares Approach". *Review of Financial Studies* 14(1), 113-147, 2001.
   [doi:10.1093/rfs/14.1.113](https://doi.org/10.1093/rfs/14.1.113)
3. Huge, B. N.; Savine, A. "Differential Machine Learning". *SSRN Electronic Journal*, 2020.
   [doi:10.2139/ssrn.3591734](https://doi.org/10.2139/ssrn.3591734)
4. Föllmer, H.; Schweizer, M. "Hedging by Sequential Regression: An Introduction to the Mathematics
   of Option Trading". *ASTIN Bulletin* 18(2), 147-160, 1988.
   [doi:10.2143/ast.18.2.2014948](https://doi.org/10.2143/ast.18.2.2014948)
5. Schweizer, M. "Variance-Optimal Hedging in Discrete Time". *Mathematics of Operations Research*
   20(1), 1-32, 1995. [doi:10.1287/moor.20.1.1](https://doi.org/10.1287/moor.20.1.1)
