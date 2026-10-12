## xVA: what a credit Monte Carlo reports

A `CreditMonteCarlo` calculation (`Credit_Monte_Carlo` in `calculation.py`) simulates the book's
value along $N$ paths and reduces those paths to an exposure profile and two adjustments: the
credit valuation adjustment (CVA) and the funding valuation adjustment (FVA), each switched on by its
own block in the `Calculation`. The `Collateral_Valuation_Adjustment` and `Initial_Margin` blocks are
experimental and are not specified here. This page states what each reduction computes, as the code computes it, so a
validator can rebuild the number from the simulated paths. The simulation itself (grids, random
numbers, correlation) is on [Calc Lifecycle](../developer/calc_lifecycle.md); the collateral
mechanics are on [Container](../valuation/container.md#nettingcollateralset).

**Scope of one run.** One run is one counterparty. Every top-level structure under `Deals` (normally
one `NettingCollateralSet` per agreement) is summed path by path into one book value before any
reduction, so several sets in one run are netted against each other. A book facing several
counterparties is run once per counterparty. The adjustments are **unilateral**: only the
counterparty defaults; the bank's own default, and with it DVA, is not modelled.

### Notation and the grid

| symbol | meaning |
| --- | --- |
| $t_0<t_1<\dots<t_m$ | the report rows: $t_0$ the base date, then every MTM date up to the last revaluation date of the book ([Grids](../developer/calc_lifecycle.md#grids)); $\Delta_i=t_i-t_{i-1}$ |
| $N$ | paths, `Batch_Size` $\times$ `Simulation_Batches` (with or without `Antithetic`) |
| $V_k(t)$ | the book's value on path $k$ in the reporting currency (`Currency`): the sum of every top-level structure's reported value, a collateralised set's after collateral |
| $X_k(t)$ | the reporting-to-base FX rate on path $k$; $X(0)$ today's |
| $D_k(t)$ | the deflator built from `Deflation_Interest_Rate` (below) |
| $I(t)$ | the counterparty's log-survival curve, $I(t)=-\log S(0,t)$, the `SurvivalProb` factor named by the block's `Counterparty` ([Survival Rates](survival_rates.md)); past its last knot its last hazard continues |
| $R$ | that factor's `Recovery_Rate`, a constant |

Every reported adjustment is a mean over batches of the per-batch estimate; with equal batches this
is the mean over all $N$ paths. Amounts are in the reporting currency at today's FX.

**The deflator.** With `Deflate_Stochastically: No`, $D(t_i)=P(0,t_i)$, today's discount factor
off the deflation curve, the same on every path. With `Yes`,

$$D_k(t_i)=\exp\Big(-\sum_{j=1}^{i}y_k(t_j;\Delta_j)\,\Delta_j\Big),\qquad D_k(t_0)=1,$$

where $y_k(t;\tau)$ is the deflation curve's zero rate for tenor $\tau$ as it stands on path $k$ at
row $t$, read for a tenor equal to the interval's length. A deflation curve with no model is read per the calculation's `NoModel`: `Constant` (the
default) reads today's curve at tenor $\Delta_j$ on every interval, so the curve does not roll;
`RiskNeutral` reads its forward from $t_j$.

### Exposure profile

`Results['mtm']` holds $V_k(t_i)$ for every path and report row; `Results['exposure_profile']`
summarises it per row, undeflated:

$$\mathrm{EE}(t_i)=\frac1N\sum_k\max\big(V_k(t_i),0\big),\qquad
\mathrm{ENE}(t_i)=\frac1N\sum_k\min\big(V_k(t_i),0\big),\qquad
\mathrm{PFE}_q(t_i)=Q_q\big(\{V_k(t_i)\}_k\big),$$

$Q_q$ the empirical $q$-th percentile (linear interpolation) of the **signed** value, one column per
entry of `Percentile` (default `95`). With FVA on, every set's value is multiplied by its
counterparty's survival $S(0,t)$ before it reaches `mtm` (see FVA), so the profile is then
survival-weighted.

### Credit valuation adjustment

$$\widehat{\mathrm{CVA}}=(1-R)\sum_{i=1}^{m}\frac1N\sum_{k=1}^{N}
\frac{E^*_k(t_{i-1})+E^*_k(t_i)}{2}\,\big(H_k(t_{i-1})-H_k(t_i)\big),\qquad
E^*_k(t)=\frac{\max\big(V_k(t)\,X_k(t)\,D_k(t),\,0\big)}{X(0)}.$$

This is the trapezoid discretisation of $(1-R)\,\mathbb E\big[\int_0^T E^*(t)\,(-dH(t))\big]$
[@gregory2020; @brigo2013]: the deflated positive exposure averaged over
each interval's two ends, times the probability of default in the interval. The path mean is taken
per interval and then summed, an order the code keeps so the reported float does not move.

| `Stochastic_Hazard_Rates` | $H_k(t_i)$ |
| --- | --- |
| `No` | $\exp(-I(t_i))$, today's survival, the same on every path |
| `Yes` | $\exp\big(-\sum_{j=1}^{i} I_k(t_j;\Delta_j)\big)$, $I_k(t;\tau)$ the simulated log-survival from $t$ over tenor $\tau$, read as the deflator is |

Default is independent of exposure unless the survival curve is simulated (`HWHazardRateModel`),
`Stochastic_Hazard_Rates` is `Yes` and `Correlations` declares a correlation between the hazard's
innovation and a market factor, or the book's own deals read that hazard (a credit default swap);
those are the only routes to wrong-way risk.

### Funding valuation adjustment

With $U_k(t)=V_k(t)X_k(t)D_k(t)/X(0)$ signed, $U^{+}=\max(U,0)$ and $U^{-}=\max(-U,0)$,

$$\widehat{\mathrm{FVA}}=\frac1N\sum_{k}\sum_{i=1}^{m}\Big[s^{c}_k(i)\,\frac{U^{+}_k(t_{i-1})+U^{+}_k(t_i)}{2}
-s^{b}_k(i)\,\frac{U^{-}_k(t_{i-1})+U^{-}_k(t_i)}{2}\Big]=\mathrm{FCA}-\mathrm{FBA},$$

reported as `fva`, positive for a net funding cost. The interval spreads accrue the funding curve
over the risk-free one, read at the interval's **start**:

| `Deflate_Stochastically` | $s^{c}_k(i)$ (`Funding_Cost_Interest_Curve` $c$ over `Risk_Free_Curve` $f$) |
| --- | --- |
| `No` | $\dfrac{P_c(0,t_{i-1})}{P_c(0,t_i)}-\dfrac{P_f(0,t_{i-1})}{P_f(0,t_i)}$ |
| `Yes` | $e^{y^c_k(t_{i-1};\Delta_i)\Delta_i}-e^{y^f_k(t_{i-1};\Delta_i)\Delta_i}$ |

$s^{b}$ is the same with `Funding_Benefit_Interest_Curve`. FVA weights the exposure by the
counterparty's survival: while the block is on, every netting set multiplies its reported value by
$S(0,t)$ of its own `Credit_Support_Amounts.Counterparty` (a set naming none is left at 1 with a
WARNING). Summed at the root, $U$ is then $\sum_s S_s(0,t)V_s(t)$, the bank-wide form of the
API page ([`Credit_Monte_Carlo`](../api_usage/calculations.md#credit_monte_carlo)). A CVA in the
same run divides its exposure by the CVA counterparty's $S(0,t)$ to undo the scaling - the run's one
counterparty (see the scope above). The block's `Counterparty` only adds that
survival curve to the factor universe; `Bank` and `Stochastic_Funding` are not read. Funding
adjustments of this shape are derived by Burgard and Kjaer [@burgard2011] and Green [@green2015].

### Inputs and defaults

The xVA blocks of the `Calculation`, with the declared defaults. The engine completes the
**top-level** block only: a block that is written takes no sub-field defaults, so a CVA block
must state `Deflate_Stochastically` and `Stochastic_Hazard_Rates` (an omission raises `KeyError`),
an FVA block that omits `Deflate_Stochastically` reads `No`, and an omitted `Gradient` or `Hessian`
reads `No`.

| block | field | default | read as |
| --- | --- | --- | --- |
| `Credit_Valuation_Adjustment` | `Calculate`, `Counterparty` | `No`, blank | the `SurvivalProb` factor's name |
| | `Deflate_Stochastically` | `Yes` | the deflator above |
| | `Stochastic_Hazard_Rates` | `No` | $H$ above |
| | `Gradient`, `Hessian` | `No`, `No` | Sensitivities below |
| | `CDS_Tenors` | `[]` | years; par CDS rates and the CS01 map |
| | `Bank` | blank | not read |
| `Funding_Valuation_Adjustment` | `Funding_Cost_Interest_Curve`, `Funding_Benefit_Interest_Curve`, `Risk_Free_Curve` | blank | $c$, $b$, $f$ |
| | `Deflate_Stochastically`, `Gradient` | `Yes`, `No` | as above |
| | `Counterparty`, `Bank`, `Stochastic_Funding` | blank, blank, `No` | see FVA |
| top level | `Batch_Size`, `Simulation_Batches`, `Random_Seed`, `Antithetic` | 1024, 1, 5120, `No` | paths and seed |
| | `Time_Grid`, `Dynamic_Scenario_Dates` | `0d 2d 1w(1w) 3m(1m) 2y(3m)`, `Yes` | the grids |
| | `MCMC_Simulations` | 2048 | inner paths of a Monte Carlo pricer |
| | `Deflation_Interest_Rate`, `Currency`, `Percentile` | `ZAR-SWAP`, `ZAR`, `95` | the deflator, the reporting currency, the PFE |
| | `NoModel`, `Gradient_Variables` | `Constant`, `All` | unmodelled factors; which leaves are differentiated |
| | `Boundary_AAD_Bandwidth`, `Boundary_AAD_Window_Touch` | 0.01, `Yes` | the boundary correction |

The closeout conventions of a collateralised set are the set's own fields (`Liquidation_Period`,
`Settlement_Period`, integers in calendar days, default 0, so the default margin period of risk is
zero) and three options in the market data's `Valuation Configuration` under
`NettingCollateralSet`: `Cash_Settlement_Risk` (0 received only, the default; 1 paid only; 2 all),
`Forward_Looking_Closeout` and `Exclude_Paid_Today` (both off). Every collateralised set in one job
must share one $(\delta_l,\delta_s)$; a job whose sets differ is refused by name. On the margin
period of risk itself see Andersen, Pykhtin and Sokol [@andersen2017].

### Sensitivities

`Gradient: Yes` on a block differentiates that block's scalar by reverse-mode algorithmic
differentiation [@capriotti2011; @griewank2008]: one backward pass per batch against
every leaf `Gradient_Variables` selects (today's curve knots, vols, spots and calibrated model
parameters; `Implied` or `Factors` narrows it), averaged over batches and reported as `grad_cva`
or `grad_fva`. A leaf the library calibrated carries the graph of the quotes it was
solved from, so the same sweep yields quote-space risk ([Quote Sensitivities](../developer/quote_sensitivities.md)).
A correlation is not a leaf; its delta is a bump ([decision 12](../limitations.md#decisions-waiting-on-the-desk)).

- **Decisions on simulated state.** A barrier crossing, an autocall trigger, an exercise or a
  collateral transfer clearing its minimum adds a term worth exactly zero in value that restores the
  derivative pathwise AAD drops at the jump; it is assembled against the CVA and the FVA objectives
  ([Boundary corrections](../developer/calc_lifecycle.md#boundary-corrections-the-sensitivity-subsystem)).
- **CS01.** With `CDS_Tenors`, the run reports par CDS rates off the counterparty and deflation
  curves under the standard CDS model (piecewise-constant hazard and forward rates, quarterly
  standard coupons, accrual on default; [@okane2008]) and, per tenor, the log-survival curve a 1bp
  bump of that tenor's rate implies (`Results['CS01']`), to map `grad_cva`'s survival rows onto CDS
  quotes.
- **Curve Jacobians.** Where the market data holds an `InterestRateJacobian.<curve>` factor for a
  simulated curve, its value is returned under `Jacobians` beside `grad_cva`.
- **Hessian.** `Hessian: Yes` (with `Gradient: Yes`) reports the CVA's full second-derivative
  matrix, `grad_cva_hessian`, by a second differentiation of the first. Second-order AAD drops
  $\delta(V)\,V_\theta V_\theta^{\top}$ at the exposure's $\max(\cdot,0)$; a term worth zero in value
  and at first order restores it, with a Gaussian kernel at Silverman's width
  $\varepsilon=1.06\,\hat\sigma\,(N)^{-1/5}$ per report row [@silverman1986]. A row whose density
  estimate at zero climbs by 2x or more across widths $(2,1,\tfrac12,\tfrac14)\varepsilon$ is a
  point mass and is refused by name, and the run refuses outright on a book that registered a
  decision correction, a collateralised set included. Gated in `tests/test_cva_gamma_kink.py`: on a
  forward at 65,536 paths the gamma and vanna land on common-random-number ladders of the reported
  delta within 5% and 10%, and asking for the Hessian moves no reported number.

### What is and is not modelled

| modelled | not modelled |
| --- | --- |
| counterparty default, deterministic or simulated hazard | the bank's own default: no DVA, no bilateral CVA |
| wrong-way risk through a declared correlation with a simulated hazard | wrong-way risk with a deterministic hazard, or any jump-at-default |
| netting of everything in the run | several counterparties in one run |
| thresholds, minimum transfers, independent amount, cash, bond and equity collateral, a margin period of risk split into settlement and liquidation | `Haircut_Received`, `Commodity_Collateral`, `Netted`, `Apply_Closeout_When_Uncollateralized`; rating-dependent CSA amounts (the first entry is read); interest on collateral over the closeout |
| funding cost and benefit over a risk-free curve, survival-weighted | funding of initial margin (MVA), capital (KVA) |
| exposure, PFE and CVA off one outer model, the run's `Model Configuration` | a historical outer for PFE beside a risk-neutral one for CVA in the same run ([decision 7](../limitations.md#decisions-waiting-on-the-desk)) |

### Known limitations

| limitation | size |
| --- | --- |
| one haircut, `Haircut_Posted`, is applied whichever side holds the asset | [Model Limitations](../limitations.md#the-engine), UNMEASURED |
| collateral is called only on simulation dates | unmeasured; a finer grid reduces it ([Container](../valuation/container.md#nettingcollateralset)) |
| the collateralised autocall's CVA delta reads short of its ladder below 16,384 paths | 8.0% short at 1,024 paths to within 1.4% at 16,384, four seeds ([Model Limitations](../limitations.md#the-autocall-tarf-and-barrier-pricers)) |
| the boundary correction holds still over its bandwidth only at the declared path count | no dependence resolved over bandwidths 0.00125 to 0.08 at 16,384 paths; a 24% fall at 2,048 ([Model Limitations](../limitations.md#the-autocall-tarf-and-barrier-pricers)) |
| no test holds the CVA or FVA value to an independent closed form | the gates hold derivatives to bump ladders and switches to bit-identity |

The trapezoid's error in the report-grid spacing is unmeasured on the library's grids.
