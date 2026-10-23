# Hedging Overview

`Execution_Mode` selects one of two verbs on the same simulated world:

- **`simulate_only`** — roll the bundle forward with zero trades and report the terminal P&L
  summary, the unhedged baseline. Callers drive their own policy on top via
  `HedgeRuntimeExecutionResult.create_stepper()`.
- **`solve_hedge`** — fit a value function and report a verdict against benchmark tracks.

## One boundary

`construct_hedge_runtime` (`hedge_bundle.py`) is the only place the `Hedging_Problem` JSON block
is read and validated. It returns the normalized **runtime** every consumer indexes by key:
canonical lowercased modes, the instrument / cash-account / hedge name sets, per-instrument
metadata, the accounting rules (position limits, turnover cost, spreads, margin funding, corridor),
the objective, the solver config and the portfolio state.

Past that boundary the runtime **is** the contract — downstream code indexes it directly rather
than re-checking it. A malformed schedule, a partial spot history, an unknown `Execution_Mode` or a
`Simulation_Batches` that contradicts the mode all fail there, naming the field.

!!! warning "`Simulation_Batches` means two different things"
    Under `simulate_only` it is a **path multiplier**: every batch is accumulated into one bundle of
    `Batch_Size x Simulation_Batches` paths. Under `solve_hedge` it is a **stream length**: a bundle
    per batch, `N-1` of them fitted and the last held out for the verdict. Minimum 2. That is a
    genuine semantic difference between two verbs, not a hidden mode flag — size a solve like a
    simulation and you will train on a fraction of the paths you meant to.

## Objective

`solve_hedge` chooses hedge positions $q$ to maximise expected utility of hedged wealth,

$$\max_q\ \mathbb{E}\big[u(x)\big],\qquad x=\frac{W_T-R}{c},\qquad
W_{t+1}=W_t+\sum_i q_i\,cs_i\,(F_{t+1,i}-F_{t,i})+(L_{t+1}-L_t),$$

where $W_t$ is the cumulative hedge P&L plus the marked liability $L_t$, $F_i$ the hedge
instrument's mark, $cs_i$ its contract size, $R$ = `Reference_Wealth` (currency, default 0) and $c$
the utility scale. This frictionless law is what the solver fits; the environment adds financing,
margin, costs and rounding ([Bundle](bundle.md)). `Reference_Mode: Running_Wealth` applies $u$ to
each day's wealth increment instead of $W_T$ and sums the step utilities.

The DP recursion lives in utility space, so `solve_hedge` requires one of these `Objective.Object`
values:

| `Objective.Object` | $u(x)$ | parameters (default) |
| --- | --- | --- |
| `AsymmetricUtility_Symlog` | $\mathrm{sign}(x)\log(1+\lvert x\rvert)$ | none |
| `AsymmetricUtility_Huber` | $x-a\,h_\delta(x^-)-a^+h_\kappa(x^+)$, with $h_k(y)=y^2$ for $y\le k$ and $k^2+2k(y-k)$ beyond | `Huber_Aversion` $a$ 2.5, `Huber_Delta` $\delta$ 1.0, `Up_Aversion` $a^+$ 0, `Up_Knee` $\kappa$ 0.15 |
| `AsymmetricUtility_CARA` | $(1-e^{-\gamma x})/\gamma$ | `CARA_Gamma` $\gamma$ 1.0 |
| `LogWealth` | per step $\log\big((C+W_{t+1})/(C+W_t)\big)$, summed; no $c$ | `Solver.DiffV2_Risk_Aversion` 1.0, in [0.1, 10] |

$x^-=\max(-x,0)$ and $x^+=\max(x,0)$. Under `LogWealth` the capital line is
$C=1.1\max(0,-\min W_{bank})/$`DiffV2_Risk_Aversion`, the worst wealth the training bank reaches;
below a ratio of 1e-3 the log continues linearly, it is floored at −20, and a run whose worst bank
wealth leaves no headroom is refused.

The scale $c$ (`Objective.Utility_Scale_Mode`, or the `Utility_Scale_Explicit` override):

| Setting | $c$ |
| --- | --- |
| `vol_scaled_notional` (default) | total leg volume × initial spot × annual vol × $\sqrt\tau$, $\tau$ the business days to the last settlement / 252; spot and vol are batch medians at the start of the simulation, from the spot history or the calibrated process; refused below 1,000 |
| `Utility_Scale_Explicit` | the stated currency amount |
| `conditional_sim` | per decision step, the cross-sectional standard deviation of the flat-book wealth $L_t$ on the warmup batch, floored at `Utility_Scale_Floor_Frac` (default 0.05) of its terminal entry; refuses `Utility_Scale_Explicit` beside it, and `Running_Wealth` except under `LogWealth` |

$c$ is resolved once when the bundle is built and never recomputed: a per-rollout `c` silently
rescales every reward, so a loaded checkpoint evaluates in **its own** frame.

## Privileged factors

What each stochastic process publishes as market state is a naming convention owned here:
`derive_privileged_layout` asks every live process what it emits, and multi-commodity runs prefix
the attribute with the factor name to disambiguate. Adding a process with its own privileged
surface flows through automatically — no registration step.
