# Risk-Neutral Bootstrapping

A bootstrapper turns the quotes in `Market Prices` into a market factor or a set of model
parameters, using today's market data only and no history. There are two kinds:

- a **construction** is deterministic: a closed form, or a square system of roots. It has no
  fitted parameters and reprices its quotes exactly;
- an **implied calibration** fits model parameters by least squares. It reprices its quotes only
  as well as the model allows, and its result depends on a seed, a box and a stopping rule.

How a block is configured, ordered and re-run is on
[Market Prices](../developer/market_prices.md); how a quote derivative passes through each solve
is on [Quote Sensitivities](../developer/quote_sensitivities.md).

| `Market Prices` type | Class (page) | Kind | Quotes | Writes | Objective | Solver and stop | Warm start | `dθ/dq` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `InterestRatePrices` | `InterestRate` ([Interest Rates](interest_rates.md)) | construction | deposits, FRAs, swaps, FX forwards | `InterestRate` zero curve | benchmark PVs $F(\theta)=0$, square | damped Newton; step $\le$ `Tol` 1e-13, `N_Iter` 50 | none: the par rates seed it | implicit function theorem at a root, exact |
| `InflationRatePrices` | `InflationRate` ([Inflation](inflation.md)) | construction | zero-coupon inflation swap breakevens | `InflationRate` growth curve | swap PVs $F(\theta)=0$, square | the curve family's damped Newton and knobs | none: the breakevens seed it as continuous rates | implicit function theorem at a root, exact |
| `SurvivalProbPrices` | `SurvivalProb` ([Credit](credit.md)) | construction | CDS par spreads and upfronts on standard contracts | `SurvivalProb` cumulative hazard curve | default swap PVs $F(\theta)=0$, square | the curve family's damped Newton and knobs | none: the quotes seed it as flat hazards | implicit function theorem at a root, exact |
| `FXVolPrices` | `FXVol` ([FX and Equity](fx_and_equity.md)) | construction | ATM, risk reversal, butterfly by delta | `FXVol` log-moneyness surface | a delta fixed point per grid node | 64 bisections per node; grid refined to `Grid_Tolerance` 1e-4 | the previous grid, at the same expiries and tolerance | one Newton step at each root |
| `GBMAssetPriceTSModelPrices` | `GBMAssetPriceTSModelParameters` ([FX and Equity](fx_and_equity.md)) | construction | the ATM column of a surface | instantaneous vol curve | none: an explicit map | none | none | the map itself, with a kink |
| `HullWhite2FactorModelPrices` | `HullWhite2FactorModelParameters` ([Interest Rates](interest_rates.md)) | implied calibration | ATM swaptions | $\sigma_1(t),\sigma_2(t),\alpha_1,\alpha_2,\rho$ | $\tfrac12\sum_jw_j^2(\sigma^{SP}_j-\sigma^N_j)^2$ by default | basin hopping (50 hops), then trust-region least squares at SciPy's 1e-8 tolerances | the previous factor; least squares only | Gauss-Newton at a stationary point, pseudo-inverse |
| `LogVar2FJModelPrices` | `LogVar2FJModelParameters` ([FX and Equity](fx_and_equity.md)) | implied calibration | European options; optional forward-start smiles | four scalars, a $\xi$ curve, four bucketed levers | $\tfrac12\lVert r\rVert^2$: vega-scaled vanilla rows plus prior and penalty rows | staged trust-region least squares; `Tolerance` 1e-8, `Step_Tolerance` 1e-12, `Max_Iterations` 150 evaluations a stage | the previous factor; joint polish only | Gauss-Newton at a stationary point (`Global` mode only) |
| `CSForwardPriceModelPrices` | `CSForwardPriceModelParameters` ([Energy](energy.md)) | implied calibration | energy futures options, as vols | $\sigma,\alpha$ | $\sum_jw_j(P_j-P^{mkt}_j)^2$ | L-BFGS-B at SciPy's defaults | the previous factor, clipped into the box | none |

The two least-squares families run SciPy's trust-region reflective method [@branch1999], a
bound-constrained trust-region method of the Levenberg-Marquardt family
[@levenberg1944; @marquardt1963], on a Jacobian taken by automatic differentiation. Three results recur on
the family pages.

**At a root the implicit function theorem is exact.** If $F(\theta^*,q)=0$ and
$\partial F/\partial\theta$ is invertible, then
$d\theta^*/dq=-(\partial F/\partial\theta)^{-1}\partial F/\partial q$, which a backward pass
contracts with one transposed solve [@christianson1998; @henrard2014].

**At a least-squares minimum it is exact only to the dropped term.** What holds at $\theta^*$ is
$J^\top r=0$. Differentiating it gives
$(J^\top J+\sum_ir_i\nabla^2r_i)\,d\theta/dq=-(J^\top\partial r/\partial q+\sum_ir_i\,\partial^2r_i/\partial\theta\,\partial q)$,
and the Gauss-Newton contraction drops both sums ([@nocedal2006], ch. 10). The
dropped terms are proportional to the residual, so they are small only where the fit is good.
Where the box holds a coordinate, the Karush-Kuhn-Tucker conditions replace $J^\top r=0$ and that
coordinate's derivative is zero.

**Where quotes do not identify every parameter, the derivative is a convention.**
$J^\top J$ is then singular, and the engine takes the Moore-Penrose pseudo-inverse [@penrose1955]
of the column-scaled Jacobian, cutting singular values below `Jacobian_Rcond` times the largest
(a truncated SVD, [@hansen1987]). Along the discarded directions the reported $d\theta/dq$ is the
minimum-norm representative, not something the quotes determine.
