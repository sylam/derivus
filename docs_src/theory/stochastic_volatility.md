## The LogVar2FJ model

LogVar2FJ is a two-factor stochastic log-variance model with a normal inverse Gaussian (NIG)
residual, walked one trading day at a time. One model, on one set of parameters, does three jobs:
it prices the path-dependent equity and FX options that opt into it, generates the spot scenarios
of a credit Monte Carlo, and is fitted to European options and forward-start smiles. It models an
**EquityPrice** and an **FxRate** alike; what changes between the two is the carry and, for FX,
which way round the rate is written (see [Equity and FX](#equity-and-fx)).

### State and return

The instantaneous annualised variance is

$$h_t=\exp\big(\min(\ell_t+s_t,\,a)\big),$$

where $a$ is a corner on log-variance (**Cap_A**, unbounded by default). The fast factor $s$
reverts to zero and the slow factor $\ell$ to a level $L^*(t)$ derived from the variance curve
below:

$$ds_t=-\kappa_s s_t\,dt+\sigma_s(t)\,dW^s_t,\qquad
d\big(\ell_t-L^*(t)\big)=-\kappa_\ell\big(\ell_t-L^*(t)\big)\,dt+\sigma_\ell\,dW^\ell_t,$$

with $\kappa_s>\kappa_\ell$ and $W^s$, $W^\ell$ independent. Both factors are stepped by their exact
OU transitions, $\phi=e^{-\kappa\delta}$ and shock weight $\sigma\sqrt{(1-e^{-2\kappa\delta})/2\kappa}$,
so the factors carry no discretisation error at any step length. The step is one trading day on
the **Steps_Per_Year** clock the parameters were fitted on, and the day is part of the model: each
day's return reads the variance at the day's start.

Over a trading day $k$ of length $\delta_k$, with $V_k=\delta_k h_{t_k}$ the day's variance and
$\eta^s_k$, $\eta^\ell_k$ the standard normals that move $s$ and $\ell$ over the same day, the
log-return is

$$\log\frac{S_{t_{k+1}}}{S_{t_k}}=b_k+\sqrt{V_k}\,\big(\rho_s\eta^s_k+\rho_\ell\eta^\ell_k\big)
-\tfrac12\big(\rho_s^2+\rho_\ell^2\big)V_k+X_k .$$

$b_k$ is the carry over the day. The leverages $\rho_s$, $\rho_\ell$ put the day's own variance
shocks into the return, and the share they leave, $c=1-\rho_s^2-\rho_\ell^2$ (held at or above
**C_Min**), is the variance clock of the residual $X$. The exponential of each day's leverage term
has mean one given the day's variance and the residual's drift is forced (below), so $E[S_T]$ is
the forward with no correction owed.

### The forward-variance curve

The stored curve is **Xi_Curve**, $\xi(t)=E_0[h_t]$, the expected forward variance. The OU level is
derived from it with the Jensen term,

$$L^*(t)=\log\xi(t)-\tfrac12\,\mathrm{Var}\big(\ell_t+s_t\big),$$

the variance measured from wherever the walk starts. $\ell+s$ is Gaussian, so without the corner
$E_0[h_t]=\xi(t)$ exactly, whatever the vol-of-vols: a change of $\sigma_s$ or $\sigma_\ell$ moves the
smile and leaves the forward variance where it is. $\xi$ is strictly positive and piecewise
constant on the segments between the ATM expiries it was fitted to.

### The residual {#the-residual}

The part of a return the leverages do not explain is an NIG increment that spends the clock
$A=\sum_k c_k V_k$. With $\gamma=\sqrt{\alpha^2-\beta^2}$,

$$X_A\sim\mathrm{NIG}(\alpha,\beta,\delta_A,\mu_A),\qquad
\delta_A=\frac{A\gamma^3}{\alpha^2},\qquad
\mu_A=\delta_A\Big(\sqrt{\alpha^2-(\beta+1)^2}-\gamma\Big).$$

$\delta_A$ makes $\mathrm{Var}\,X_A=A$, and $\mu_A$ forces $E[e^{X_A}]=1$: the drift is
no-arbitrage's, not a parameter. $\alpha$ is tail thickness and smile convexity, $\beta$ is skew.
The NIG law is a normal variance-mean mixture with an inverse Gaussian mixer, and it generates a
Lévy process, a Brownian motion subordinated by the inverse Gaussian process
(Barndorff-Nielsen, 1997):

$$X_A=\mu_A+\beta G+\sqrt{G}\,Z,\qquad G\sim\mathrm{IG}\big(\text{mean }\delta_A/\gamma,\ \text{shape }\delta_A^2\big),\qquad Z\sim N(0,1).$$

$\delta_A$ and $\mu_A$ are both linear in $A$, so the residual is that Lévy process run on the
variance clock: a block of days is ONE draw on the block's summed clock, and a month cut into
days is the same law as the month. Given the variance shocks and the block's mixer, the block's
log-return is therefore exactly Gaussian,

$$\log\frac{S_T}{S_t}\;\Big|\;\eta,G\;\sim\;N(M,\,G),\qquad
M=\sum_k\Big(b_k+\sqrt{V_k}\,(\rho_s\eta^s_k+\rho_\ell\eta^\ell_k)-\tfrac12(\rho_s^2+\rho_\ell^2)V_k\Big)+\mu_A+\beta G,$$

which is the whole of the pricing: a survival probability is one $\Phi$, a conditional draw one
$\Phi^{-1}$ and a call a conditional Black, the lognormal arithmetic of
[One-Step Survival](mc_simulation.md) at the block's own $(M,G)$. The factor asserts $|\beta|<\alpha$
and $|\beta+1|<\alpha$, without which there is no forced drift, and a conditioning share
$\gamma^2/\alpha^2\ge0.4$, past which the mixer rather than the Gaussian carries the return.
**Residual_Law** `Gaussian` drops the mixer, $X_A\sim N(-A/2,\,A)$, as a limit and test mode.

### Calendar buckets and the corner

$\kappa_s$, $\kappa_\ell$, $\sigma_\ell$ and $\rho_\ell$ are scalars. $\rho_s$, $\beta$, $\sigma_s$
and $\alpha$ are piecewise constant on calendar-time buckets that start at their (shared) knots, one
knot at zero being the constant-parameter model. A smile at expiry $T$ never reads a bucket later
than $T$, while a forward smile reads nothing else, so the buckets are the lever on the forward
smile; $\rho_s$ and $\beta$ are the forward-skew pair. A bucket knot inside a block splits the
block's clock into two residual draws whose $M$ and $G$ add.

The corner exists for simulation. By default the fit runs uncapped and writes a corner far in the
fitted stationary law's tail, six stationary standard deviations of log-variance above the
starting level (a declared corner is raised to it), which a calibrated walk is not expected to
reach.

## Limits and lineage

### Black-Scholes

Set the vol-of-vols and the leverages to zero and the residual to `Gaussian`. Then
$\mathrm{Var}(\ell_t+s_t)=0$, $h_t=\xi(t)$, $c=1$ and the log-return over $[t,T]$ is Gaussian with
variance $\sum_k\delta_k\xi(t_k)$: Black-Scholes (Black and Scholes, 1973) with a deterministic,
time-dependent variance, and Black-Scholes itself on a flat $\xi=\sigma^2$. The limit is gated
against the GBM pricer on one document, on the mark and through the CVA
(`tests/test_logvar2fj_json.py`).

### Bergomi's forward-variance models {#bergomi}

Bergomi's models specify the joint dynamics of the spot and its forward variance-swap variances
$\xi^T_t$, the forward instantaneous variance for date $T$ seen at $t$ (Bergomi, 2005; restated
in Bergomi, 2008). The initial curve $\xi^T_0$ is an input beside the spot, naturally generated
from variance-swap volatilities. His $N$-factor lognormal form (Bergomi, 2008, eq. 1) is

$$\xi^T_t=\xi^T_0\exp\Big(\omega\sum_n w_n e^{-k_n(T-t)}X^n_t
-\frac{\omega^2}{2}\sum_{n,m}w_nw_m e^{-(k_n+k_m)(T-t)}E\big[X^n_tX^m_t\big]\Big),
\qquad dX^n_t=-k_nX^n_t\,dt+dW^n_t,\ X^n_0=0,$$

with, for two factors, $k_1>k_2$, the factors $X=X^1$ and $Y=X^2$ correlated $\rho$ with each other,
and the spot $dS_t=(r-q)S_t\,dt+S_t\sqrt{\xi^t_t}\,dW_t$ correlated with them by $\rho_{SX}$ and
$\rho_{SY}$ (eq. 10 there).

**Without the corner, with a Gaussian residual and one bucket, LogVar2FJ is that model with
uncorrelated factors, sampled on the trading day.** Write
$y_t=\ell_t-L^*(t)$. From a start at $(\ell,s)=(L^*(0),0)$, $y$ and $s$ are independent zero-mean OU
processes from zero, and the definition of $L^*$ gives

$$h_t=\xi(t)\exp\Big(y_t+s_t-\tfrac12\mathrm{Var}\,y_t-\tfrac12\mathrm{Var}\,s_t\Big).$$

Conditioning on time $t$, with $\tau=T-t$,

$$E_t[h_T]=\xi(T)\exp\Big(e^{-\kappa_s\tau}s_t+e^{-\kappa_\ell\tau}y_t
-\tfrac12e^{-2\kappa_s\tau}\mathrm{Var}\,s_t-\tfrac12e^{-2\kappa_\ell\tau}\mathrm{Var}\,y_t\Big),$$

which is the two-factor form above with $k_1=\kappa_s$, $\omega w_1=\sigma_s$, $k_2=\kappa_\ell$,
$\omega w_2=\sigma_\ell$, $\rho=0$, and $\rho_{SX}=\rho_s$, $\rho_{SY}=\rho_\ell$: the day's return is
correlated $\rho_s$ with the shock that moves $s$, as $W$ is with $W^X$. Its forward variances are
lognormal martingales,

$$\frac{dE_t[h_T]}{E_t[h_T]}=e^{-\kappa_s(T-t)}\sigma_s\,dW^s_t+e^{-\kappa_\ell(T-t)}\sigma_\ell\,dW^\ell_t,$$

so a near date's variance moves with both factors and a far one, $T-t\gg1/\kappa_s$, with the slow
factor alone. The structural defaults put the two half-lives near six weeks ($\kappa_s=6$) and
seventeen months ($\kappa_\ell=0.5$).

**What derivus takes from the framework.**

- *The curve is an input the dynamics leave in place.* $\xi$ is stored and the vol-of-vols move the
  smile, never $E_0[h_t]$. The route to the curve differs: the fit solves $\xi$ against the ATM
  ladder at every candidate parameter set, so the ATM term structure reprices exactly whatever the
  vol-of-vols, and reports it beside the market's ATM$^2$ forward variance; it should sit on a
  replicated variance-swap strip up to the smile's own ATM convexity.
- *Two factors at two speeds*, with the leverages playing $\rho_{SX}$ and $\rho_{SY}$.
- *The skew coordinate.* At first order in the vol-of-vol, the two-factor model's ATM-forward skew
  is the vol-of-vol times a combination of the two spot/volatility correlations whose weights the
  reversion speeds set (Bergomi, 2008, eq. 16). Here the corresponding products are
  $\rho_s\sigma_s$ and $\rho_\ell\sigma_\ell$, and $\rho_s\sigma_s$ is the coordinate one of the
  calibration's two leverage-prior rows sits on.
- *The forward smile is the open quantity.* Bergomi starts from the observation that most recent
  exotic structures, reverse cliquets and Napoleons among them, depend more on the assumed future
  dynamics of implied vols than on today's vanillas (Bergomi, 2004), and that the usual
  stochastic-volatility and jump models tie the short forward skew, the spot/vol correlation and
  the vol-of-vol term structure together, which his 2005 model controls separately (Bergomi,
  2005). Here the vanillas leave the split between residual skew and leverage skew, and with it the
  forward smile, unpinned; see [what the vanillas identify](#what-the-vanillas-identify).

**What derivus does differently.**

- *The variance is specified through its log, not through the forward curve.* The two routes give
  one model in the limit above; the log-variance pair is the state the walk steps, the corner
  bounds and the historical estimator filters.
- *The residual is NIG on the variance clock.* Bergomi's spot is a diffusion given the variance: a
  Brownian one in the continuous form, and in the discrete form, which controls the short forward
  skew on its own, one with a local volatility function of the spot's move since the start of each
  interval (Bergomi, 2008, after Bergomi, 2005). Here $\beta$ and $\alpha$ carry short-dated skew and
  tails beyond what leverage and vol-of-vol give.
- *Calendar buckets.* $\rho_s$, $\beta$, $\sigma_s$ and $\alpha$ move in calendar time, a lever that is
  not one of the factors. Bergomi notes that in a two-factor framework $k_1$ and $k_2$ shape both
  the decay of the skew and the term structure of vol-of-vol, which makes separate handles on the
  two difficult (Bergomi, 2008).
- *Uncorrelated factors.* The walk draws the two factors' shocks independently, where Bergomi's
  carry a correlation $\rho$.

## How derivus uses it

### One walk, three readers

**The pricer.** A deal opts in with the valuation option **SpotModel** `LogVar2FJ`: the equity
barrier and barrier-binary options, the autocall swaps, and the FX TARF and accumulator. The kit
walks one internal-step path per reporting row, and each remaining fixing interval reads its block
law $(M,G)$ off it. A row with no state to inherit re-seeds at $(L^*(t_\text{row}),0)$ with the
variance measured from that row, so it reads $\xi$ exactly; under a credit Monte Carlo whose outer
is LogVar2FJ, the row starts instead from the $(\ell,s)$ the outer path carries.

**The scenario generator.** `LogVar2FJImpliedSpotModel` walks the same days between scenario
nodes, off the same calibrated **LogVar2FJModelParameters** factor, so a CVA sensitivity reaches the
one parameter leaf the pricer reads. It takes four of the framework's correlated normals per name
per scenario step: the return's Gaussian given the mixer, the fast and the slow factor's exact OU
transition over the interval (the days inside it bridged between those endpoints), and the mixer
through its own quantile, $G=F^{-1}_\mathrm{IG}(\Phi(Z_G))$. A declared correlation on any of the
four rows is therefore realised as declared. A book may instead run a GBM term-structure outer
with the LogVar2FJ pricer. Recovering $(\ell,s)$ from an observed price path is a filtering
problem, so the process refuses to replay one. Detail:
[Structures](../developer/structures.md#model-worth).

**The calibration.** A `LogVar2FJModelPrices` block fits the factor to a ladder of Europeans and,
where a source has them, forward-start smiles. A vanilla is a conditional Black averaged over the
block law: by default by Gauss-Hermite quadrature over a lognormal surrogate of the clock matched
on its closed-form moments and over the exact inverse Gaussian mixer, otherwise over fixed
antithetic paths of the walk, which the forward-start rows always read. $\xi$ is the inner solve,
one segment at a time against its own ATM expiry, and the shape parameters the outer least-squares
fit around it. Separately, `LogVar2FJCalibration` estimates the model under the historical measure,
filtering $(\ell,s)$ from a daily realised-variance measurement; it supplies the correlation rows
of the four innovations above (the outer's map from four normals to a day, inverted) and priors,
never the pricing parameters. Detail: [Market Prices](../developer/market_prices.md#logvar2fj) and
[the fit](../bootstrapping/fx_and_equity.md#logvar2fjmodelparameters).

### Equity and FX {#equity-and-fx}

On an **EquityPrice** the carry is the equity's own zero rate less its dividend rate; on an
**FxRate** it is the domestic less the foreign zero rate, as for GBM.

An **FxRate** is a currency priced in the base, and a fitted law describes one token priced in
another. A pair with a base leg is its non-base token priced in the base. A cross is the
alphabetically later token priced in the earlier: EUR/ZAR is the rand priced in the euro, its spot
the ratio of the two base-priced rates (the block's **Priced_In**), filed as
`LogVar2FJModelParameters.ZAR.EUR`, whichever way the deal or the surface is written. Detail:
[the join](../developer/structures.md#the-join).

**A deal paying on the reciprocal axis is a change of numeraire, not a second fit.** A deal whose
underlying is the currency the law is priced in (an FX TARF or accumulator written the other way
up) pays on $1/S$, and the kit prices it under the $S$ numeraire (`Invert_Spot`). With $R_k$ the
day's log-return, the density $e^{R_k-b_k}$ factorises over the day's own draws,
so each variance shock shifts by its loading, $\eta^s_k\sim N(\rho_s\sqrt{V_k},1)$ and
$\eta^\ell_k\sim N(\rho_\ell\sqrt{V_k},1)$, and the residual takes the Esscher transform at one,
$\mathrm{NIG}(\alpha,\beta)\to\mathrm{NIG}(\alpha,\beta+1)$:

$$G\sim\mathrm{IG}\big(\text{mean }\delta_A/\gamma_1,\ \text{shape }\delta_A^2\big),\qquad
\gamma_1=\sqrt{\alpha^2-(\beta+1)^2},\qquad X_A\,|\,G\sim N\big(\mu_A+(\beta+1)G,\ G\big),$$

the same $\gamma_1$ the forced drift already carries. The reciprocal's log-return is the negative
of the walked one. Because the shift is the state's own $\sqrt{V_k}$, the variance path itself
changes under the new measure, so the walk steps that case one day at a time.

**A quanto** (an equity paid in another currency at a fixed rate) changes measure to the payoff
currency. The walk adds the drift on each day's own variance, $-\rho_q\,\sigma_{FX,k}\sqrt{\delta_k}\sqrt{V_k}$,
the continuous-time $-\rho\,\sigma_S\sigma_X\,dt$ with $\sigma_S=\sqrt{h_t}$ in place of an implied ATM
vol. $\sigma_{FX}$ is the FX surface's ATM forward strip read on the walk's grid and $\rho_q$ the
book's marked equity/FX correlation, read as a total-return correlation. Detail:
[Market Prices](../developer/market_prices.md#logvar2fj).

## What the vanillas identify {#what-the-vanillas-identify}

The ATM ladder pins $\xi$, one segment per expiry. The fit reads the residual pair $(\alpha,\beta)$
off the short-dated wings, the fast pair $(\rho_s,\sigma_s)$ off the wings out to a year, and the
slow pair $(\rho_\ell,\sigma_\ell)$ only off wings at eighteen months or longer, holding it at a prior
otherwise. Vanillas alone pin the tail parameter $\alpha$ weakly, and the fit reports for each
parameter how hard its prior pushes against the quotes.

They do not separate residual skew from leverage skew: $\beta$ and $\rho_s\sigma_s$ bend the spot
smile the same way, so a vanilla-only fit hands the skew to whichever is cheaper and leaves the
forward smile, the thing an autocall or a cliquet reads, undetermined. The split is identified by
a forward-start source (**Forward_Smile_Source** `Quotes`, traded forward-starts, or `Reference`, a
reference model's forward smiles), whose targets are differences against the spot smile:
$\Delta_\text{skew}$, the forward 90-110 slope less the spot slope at the window's length, and
$\Delta_\text{bfly}$, the same for the 90/110 butterfly, both in vol points. A window short enough
for the residual to still be non-Gaussian is what moves $(\alpha,\beta)$.

**Where the forward smile is not quoted, it is a reserve.** The fit writes on the factor
**Skew_Gradient**, $J=(\partial\Delta_\text{skew}/\partial\beta,\ \partial\Delta_\text{skew}/\partial\rho_s)$
in the last bucket at the declared forward window, and **Stickiness_Band** $b$ in vol points. A
deal's **Skew_Reserve** takes the minimum-norm parameter move behind one vol point of
$\Delta_\text{skew}$,

$$b\,\frac{\big|g\cdot J\big|}{J\cdot J},\qquad
g=\Big(\frac{\partial PV}{\partial\beta},\frac{\partial PV}{\partial\rho_s}\Big)\ \text{in the same bucket},$$

and is reported per deal and for the portfolio. The priors that hold what the ladder cannot see,
and what each costs against a quote, are on
[Market Prices](../developer/market_prices.md#logvar2fj).

## References {#references}

- Barndorff-Nielsen, O. E. (1997). Normal inverse Gaussian distributions and stochastic volatility
  modelling. *Scandinavian Journal of Statistics*, 24(1), 1-13.
  [doi:10.1111/1467-9469.00045](https://doi.org/10.1111/1467-9469.00045)
- Bergomi, L. (2004). Smile dynamics. *Risk*, September 2004, 117-123.
  [SSRN, doi:10.2139/ssrn.1493294](https://doi.org/10.2139/ssrn.1493294)
- Bergomi, L. (2005). Smile dynamics II. *Risk*, October 2005, 67-73.
  [SSRN, doi:10.2139/ssrn.1493302](https://doi.org/10.2139/ssrn.1493302)
- Bergomi, L. (2008). Smile dynamics III. *Risk*, October 2008, 90-96.
  [SSRN, doi:10.2139/ssrn.1493308](https://doi.org/10.2139/ssrn.1493308);
  [PDF at Risk](https://www.risk.net/media/download/920801/download)
- Bergomi, L. (2016). *Stochastic Volatility Modeling*. Chapman and Hall/CRC. Chapters 3
  (forward-start options), 7 (forward variance models) and 8 (the smile of stochastic volatility
  models). [doi:10.1201/b19649](https://doi.org/10.1201/b19649)
- Black, F. and Scholes, M. (1973). The pricing of options and corporate liabilities. *Journal of
  Political Economy*, 81(3), 637-654. [doi:10.1086/260062](https://doi.org/10.1086/260062)
