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

with $W^s$, $W^\ell$ independent. The fit declares $\kappa_s=6$ and $\kappa_\ell=0.5$, the fast
factor reverting twelve times faster; the factor checks neither that ordering nor the sign of a
speed or a vol-of-vol ([Parameters](#parameters)). Both factors are stepped by their exact OU
transitions, $\phi=e^{-\kappa\delta}$ and shock weight $\sigma\sqrt{(1-e^{-2\kappa\delta})/2\kappa}$,
so the factors carry no discretisation error at any step length. The step is one trading day on
the **Steps_Per_Year** clock the parameters were fitted on, and the day is part of the model: each
day's return reads the variance at the day's start ([The clock](#the-clock)).

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
[@barndorffnielsen1997]:

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

The corner exists for simulation ([moments](#positivity-and-moments)). The fit runs uncapped and
writes

$$a=\log\xi(0)+12\,\max_b\ \tfrac12\sqrt{\frac{\sigma_{s,b}^2}{2\kappa_s}+\frac{\sigma_\ell^2}{2\kappa_\ell}},$$

the square root being the stationary standard deviation of log-variance in bucket $b$ and half of
it that of log-volatility: twelve stationary standard deviations of log-volatility, six of
log-variance, at the widest bucket, above $\log\xi(0)$, which is $L^*(0)$ since the walk starts at
rest. A declared **Cap_A** is raised to it (`LVFit.cap_level`), and the fit refuses a factor whose
own walk puts more than **Cap_Headroom_Max** (1e-5) of its path-days at or above the corner. On
the synthetic factor of `tests/test_logvar2fj_json.py` (`WORLD_FACTOR`), $\log\xi(0)=-3.1422$, the
log-volatility standard deviation is 0.5813 and the written corner 3.8329. **Held by**
`tests/test_logvar2fj_json.py::test_a_null_cap_fits_as_omitted_and_writes_the_corner`.

### The clock {#the-clock}

The model is defined on a clock of **Steps_Per_Year** steps a year, 252 unless the fit's block
states another, written onto the factor with the parameters. A step is $1/$**Steps_Per_Year** of
a year fraction, not a business day of a calendar: no weekend or holiday is modelled, and a year
fraction of one holds 252 steps whatever the calendar. Each step reads the variance at its start,
$V_k=\delta_kh_{t_k}$, so the step is part of the model and not a discretisation to refine: the OU
transitions are exact at any step length and the residual is linear in its clock, but the clock
$\sum_kc_k\delta_kh_{t_k}$ and the leverage sum are sums over the steps taken. Another clock is
another model, and the fitted parameters mean the clock they were fitted on.

Three readers lay the steps over an interval of year fraction $\Delta$, each landing exactly on the
interval's two ends. With $N=\Delta\cdot$**Steps_Per_Year** the interval's length in whole steps
of $1/$**Steps_Per_Year**:

| Reader | Steps over the interval | Code |
| --- | --- | --- |
| the pricer's kit, per remaining fixing interval | $n=\max(\mathrm{round}(N),1)$ equal steps of $\Delta/n$ | `pricing.LogVar2FJKit.grid` |
| the credit Monte Carlo's outer process, per scenario interval | $\lfloor N\rfloor$ whole steps, then one step of the remaining $N-\lfloor N\rfloor$ | `utils.substep_schedule`, read by `LogVar2FJImpliedSpotModel.precalculate` |
| the fit, between consecutive block ends | $n=\max(\mathrm{round}(N),1)$ steps: $n-1$ whole and a last stub of $N-(n-1)$ | `bootstrappers.LogVar2FJModelParameters.prepare` |

The three cuts are one grid where $N$ is a whole number; otherwise the kit spreads the fraction of
a step over all the interval's steps, and the outer and the fit put it in one shorter or longer
step at the interval's end.

**A deal's own dates are grid points.** The kit cuts each remaining fixing interval, measured from
the reporting row, on its own, so every fixing, observation and barrier date is a step end and no
step straddles one; an interval shorter than one and a half steps is one step of its own length,
and a reporting row landing on a fixing opens an interval of no length, which takes no residual
draw. The fit's block ends are every quoted maturity, both ends of every forward window and every
calendar-bucket knot, so the fit reads at a maturity what a pricer reads there.

**The clock is the factor's.** A deal's Valuation Configuration may restate `Steps_Per_Year` only
at the factor's value; another value refuses the deal by name, since one document would value the
same parameters on two grids (`instruments.set_spot_model_index`). **Not gated.** The factor
refuses at load a reversion speed past single precision at its clock,
$\max(|\kappa_s|,|\kappa_\ell|)\cdot31.5/$**Steps_Per_Year** $\ge\log(\text{float32 max})\approx88.72$,
which at 252 is $|\kappa|\ge709.8$ a year: a checkpointed segment of the walk spans 21 steps of up
to a day and a half and exponentiates the reversion across it. **Held by**
`tests/test_logvar2fj_json.py::test_a_reversion_past_single_precision_at_its_clock_refuses_at_load`.
The commodity spot `LogVar2FJSpotModel`, a different model under the historical measure, steps
once a market day of the job's calendar instead ([its page](asset_pricing.md)).

### Parameters {#parameters}

Every parameter is a field of the price factor `LogVar2FJModelParameters.<name>`
([its JSON](../json/price_factors_overview.md#logvar2fjmodelparameters)), and the factor checks
them at load, per bucket and by name. A *leaf* carries a derivative; a *structural* field is read
when the deal is compiled and carries none. *Fitted* is set by the calibration against quotes,
*declared* is stated on the fit's block and copied, *derived* is computed by the fit, and
*written* is a report the fit leaves on the factor.

| Field | Meaning | Units | Default on the factor | Checked at load, and why | Kind |
| --- | --- | --- | --- | --- | --- |
| `Kappa_L` | $\kappa_\ell$, slow reversion speed | per year | 0 (the fit declares 0.5) | only the single-precision bound of [the clock](#the-clock) | declared, never fitted; a leaf |
| `Kappa_S` | $\kappa_s$, fast reversion speed | per year | 0 (the fit declares 6) | as `Kappa_L` | declared, never fitted; a leaf |
| `Sigma_L` | $\sigma_\ell$, slow vol of log-variance | per $\sqrt{\text{year}}$ | 0 | none | fitted where the ladder has a wing at `Slow_Horizon` (1.5y) or longer, box 0.3 to 2.0; else held at a prior; a leaf |
| `Rho_L` | $\rho_\ell$, slow leverage | none | 0 | $c=1-\rho_s^2-\rho_\ell^2\ge$ `C_Min` in every bucket | as `Sigma_L`, box $-0.6$ to 0; a leaf |
| `Xi_Curve` | $\xi(t)=E_0[h_t]$, expected forward variance, constant from each knot to the next and flat past the last | annualised variance; knots in years | none | every value $>0$: the OU level takes its logarithm | fitted exactly, one segment per ATM expiry; values are leaves |
| `Rho_S` | $\rho_s(t)$, fast leverage | none | none | $c\ge$ `C_Min` per bucket; the four bucket curves carry the same knots | fitted; values are leaves |
| `Sigma_S` | $\sigma_s(t)$, fast vol of log-variance | per $\sqrt{\text{year}}$ | none | none | fitted, box 0.5 to 5.0; values are leaves |
| `Alpha` | $\alpha(t)$, residual tail thickness | none | none | $\lvert\beta\rvert<\alpha$, the NIG law; $\lvert\beta+1\rvert<\alpha$, the forced drift; $1-(\beta/\alpha)^2\ge0.4$, the share of the residual's variance left Gaussian once the mixer is drawn, which one-step survival conditions on | fitted, in a coordinate that stays inside the first two at every iterate; values are leaves |
| `Beta` | $\beta(t)$, residual skew | none | none | as `Alpha` | as `Alpha` |
| `Cap_A` | $a$, the corner on $\ell+s$ | log annualised variance | null, no corner | none | derived by the fit (above); structural |
| `Steps_Per_Year` | the clock | steps a year | 252 | the single-precision bound | declared; structural |
| `C_Min` | floor on $c$ | none | 0.12 | the floor itself, in every bucket: one-step survival conditions on the residual, which carries the share $c$ of a step's variance | declared; structural |
| `Residual_Law` | `NIG`, the model, or `Gaussian`, $X_A\sim N(-A/2,A)$ | text | `NIG` | one of the two; under `Gaussian` the factor fills `Alpha` 1 and `Beta` 0 where absent and nothing reads them | declared; structural |
| `Skew_Gradient` | $(\partial\Delta_\text{skew}/\partial\beta,\ \partial\Delta_\text{skew}/\partial\rho_s)$ in the last bucket ([the reserve](#what-the-vanillas-identify)) | vol points per unit | blank | none | written; structural |
| `Stickiness_Band` | the band the reserve is taken over | vol points | 0, no reserve (the fit declares 0.5) | none | declared; structural |
| `On_Guard` | every guard the fit's $\theta^*$ sits on | text | blank | logged at load and carried into a mark's `Stats` | written; structural |

The five curves are authored as curves, a bare number refusing by name (`[[0.0, x]]` is the
one-bucket model). `Rho_S`, `Beta`, `Sigma_S` and `Alpha` are piecewise constant on calendar
buckets that start at their knots; `Rho_S` and `Sigma_S` are read at each step's start, `Alpha`
and `Beta` once per residual draw, at the bucket the draw's piece lies in.

**Not checked:** the signs of `Kappa_L`, `Kappa_S`, `Sigma_L` and `Sigma_S`, and
$\kappa_s>\kappa_\ell$. A reversion speed of exactly zero, the factor's own default, makes the OU
shock weight $0/0$, which evaluates to zero: that factor does not move, where the limit as the
speed falls to zero is a random walk of weight $\sigma\sqrt\delta$. A fit always writes both speeds.

**Retired names refuse by name**, naming what replaced each, on the factor, on a
`LogVar2FJModelPrices` block, in the `Bootstrapper Configuration` entry and on a factor a fit warm
starts from (`utils.LogVar2FJ.retired`): `Cap_Beta`, `Beta_Prior_Defaults`, `Beta_Prior_Sd`,
`Jump_Share`, `Lambda`, `Nu`, `Mu_J`, `Sigma_J`, `L_Curve`, `Wing_Strike`, `Diffusive_Share`,
`Vanilla_Guard`, `Vanilla_Band` and `Stickiness_Prior`. **Held by**
`tests/test_logvar2fj_json.py::test_a_withdrawn_declaration_refuses_by_name`,
`::test_a_warm_start_off_a_retired_factor_refuses_by_name`.

## Theoretical properties {#theoretical-properties}

What the model guarantees by construction, each statement naming the test that holds it or saying
it is held by none. $\mathcal F_k$ is everything known at the start of step $k$.

### The martingale condition {#the-martingale-condition}

$V_k$ is known at the start of step $k$ and the day's shocks $\eta^s_k,\eta^\ell_k$ are independent
of $\mathcal F_k$ and of each other, so the leverage part of a step is compensated given its own
variance,

$$E\Big[\exp\Big(\sqrt{V_k}\,(\rho_s\eta^s_k+\rho_\ell\eta^\ell_k)-\tfrac12(\rho_s^2+\rho_\ell^2)V_k\Big)\,\Big|\,\mathcal F_k\Big]=1 .$$

The residual is drawn given the shocks, on the clock $A$ they set, and its drift is forced. The NIG
cumulant function [@barndorffnielsen1997], $\log E[e^{uX_A}]=\mu_Au+\delta_A\big(\gamma-\sqrt{\alpha^2-(\beta+u)^2}\big)$,
is finite for $\lvert\beta+u\rvert<\alpha$, and at $u=1$ the $\mu_A$ above makes it zero:

$$E\big[e^{X_A}\,\big|\,A\big]=\exp\Big(\mu_A+\delta_A\big(\gamma-\sqrt{\alpha^2-(\beta+1)^2}\big)\Big)=1 .$$

Taking the steps in order, and each residual draw given the shocks, $E_t[S_T]/S_t=\exp\big(\sum_kb_k\big)$,
the forward, at every horizon and for every parameter set the factor admits; a calendar knot inside
an interval splits its clock into draws each forcing its own drift. No pricer normalises its paths
and the outer process adds no drift correction. The corner changes only $V_k$, which the compensator
reads, so the martingale holds with it; what the corner gives up is $E_0[h_t]=\xi(t)$. **Held by**
`tests/test_spot_model_martingale.py::test_a_quoted_forward_strip_solves_one_strike_from_either_side`,
on the pricer's kit under a live NIG factor carrying a declared corner: a zero-cost strip
of monthly forwards over a year solves one strike from either notional side, the direct axis and
the reciprocal, within 1e-3. **Not gated:** $E[S_T]/F(T)$ read directly at several horizons, for
the kit and for the outer process under a live NIG factor.

The fit prices each vanilla off its own estimator's forward, a control variate
(`LVFit.conditional_black`), and the quadrature builds its leverage part so that
$E[e^\text{return}]$ is the forward exactly (`utils.LogVar2FJ.quadrature`).

**The reciprocal axis** is a change of numeraire whose density, a step at a time, is $e^{R_k-b_k}$,
of mean one by the two identities above; under it each shock shifts by its loading and the residual
takes the Esscher transform at one ([Equity and FX](#equity-and-fx)). **Held by**
`tests/test_cross_spot_model.py::test_the_reciprocal_axis_carries_the_cross` and the strip above.

**A quanto** is a change of measure to the payoff currency, under which the spot is not a
martingale: the walk adds $-q_k\sqrt{V_k}$ a step, $q_k=\rho_q\sigma_{FX,k}\sqrt{\delta_k}$
([Equity and FX](#equity-and-fx)). **Held by**
`tests/test_logvar2fj_json.py::test_the_quanto_loading_is_off_at_zero_correlation`,
`::test_the_quanto_gbm_limit_reproduces_the_gbm_quanto_deal`.

### Positivity and moments {#positivity-and-moments}

$h_t=e^{\min(\ell_t+s_t,\,a)}$ is positive by construction, and $\xi>0$ is checked at load, so the
level $L^*$ exists. A step's variance $V_k$ is positive on a step of positive length and zero on
one of none. The mixer $G$ is positive, its quantile computed in logs as $m\,e^{y}$
(`utils.LogVar2FJ.ig_root`); where its uniform rounds to one in double, the root reads the tail at
the smallest double instead ([Model Limitations](../limitations.md#the-xva-outer-and-the-correlation)).

Without the corner $\ell+s$ is Gaussian, so $h$ is lognormal with every moment and
$E_0[h_t]=\xi(t)$ exactly. The spot's moments above the first are another matter. Given
$\mathcal F_k$ a step's $p$-th moment is $E[e^{pR_k}\mid\mathcal F_k]=e^{pb_k+K(p)V_k}$, with
$K(p)$ the step's cumulant per unit of variance, zero at $p=0$ and $p=1$ and convex, so positive
for $p>1$. $V_k$ is lognormal in the shocks of the step before it, which enter $\log S$ only
linearly, and $E[e^{KV_k}]$ is infinite for every $K>0$; so without the corner $E[S_T^p]$ is
infinite for every $p>1$ at every horizon of two steps or more, wherever a vol-of-vol is non-zero.
With it $V_k\le\delta_ke^{a}$, and $E[S_T^p]$ is finite wherever the residual's own moment is,
$\lvert\beta+p\rvert<\alpha$ in every bucket. That is what the corner is for; the fit
places it six stationary standard deviations of log-variance out and refuses a factor whose walk
reaches it on more than 1e-5 of path-days, so it does not shape the smile. Under it
$E_0[h_t]\le\xi(t)$. **Not gated.**

### Limiting cases {#limiting-cases}

| Limit | What it is | Held by |
| --- | --- | --- |
| vol-of-vols and leverages zero, `Residual_Law: Gaussian` | $\mathrm{Var}(\ell_t+s_t)=0$, $h_t=\xi(t)$, $c=1$, and the log-return over $[t,T]$ is Gaussian with variance $\sum_k\delta_k\xi(t_k)$: Black-Scholes [@black1973] with a deterministic, piecewise-constant variance, and Black-Scholes itself on a flat $\xi=\sigma^2$ | on a flat $\xi$, against the GBM arm on the same document within 1e-12 relative: `tests/test_logvar2fj_json.py::test_the_gbm_limit_reproduces_the_gbm_arm` (an autocall, both sides), `::test_a_discrete_barrier_at_the_gbm_limit_is_the_gbm_barrier`, `::test_a_knock_in_every_scenario_has_crossed_is_the_european_on_its_own_walk`, `::test_the_quanto_gbm_limit_reproduces_the_gbm_quanto_deal`; through the CVA to one float32 ulp, `::test_the_gbm_limit_holds_through_the_cva`; the outer process deflating a foreign cashflow to its value today on every row, `tests/test_option_settlement_json.py::test_under_simulation_a_foreign_cashflow_deflates_to_its_value_today_on_every_row`. A $\xi$ with a term structure: **Not gated** |
| $\alpha\to\infty$ at fixed $\beta$ | the residual's skewness and excess kurtosis vanish ([below](#forward-smile)) and its mean tends to $-A/2$: the `Gaussian` residual in law | **Not gated**; no code path takes the limit |
| no corner, `Gaussian` residual, one bucket | Bergomi's two-factor forward-variance model with uncorrelated factors, sampled on the trading day ([below](#bergomi)) | **Not gated** |

## Lineage

### Bergomi's forward-variance models {#bergomi}

Bergomi's models specify the joint dynamics of the spot and its forward variance-swap variances
$\xi^T_t$, the forward instantaneous variance for date $T$ seen at $t$ ([@bergomi2005]; restated
in [@bergomi2008] and in [@bergomi2016], chapter 7). The initial curve $\xi^T_0$ is an input beside the spot, naturally generated
from variance-swap volatilities. His $N$-factor lognormal form ([@bergomi2008], eq. 1) is

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
factor alone. The fit's declared speeds put the two half-lives near six weeks ($\kappa_s=6$) and
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
  reversion speeds set ([@bergomi2008], eq. 16). Here the corresponding products are
  $\rho_s\sigma_s$ and $\rho_\ell\sigma_\ell$, and $\rho_s\sigma_s$ is the coordinate one of the
  calibration's two leverage-prior rows sits on.
- *The forward smile is the open quantity.* Bergomi starts from the observation that most recent
  exotic structures, reverse cliquets and Napoleons among them, depend more on the assumed future
  dynamics of implied vols than on today's vanillas [@bergomi2004], and that the usual
  stochastic-volatility and jump models tie the short forward skew, the spot/vol correlation and
  the vol-of-vol term structure together, which his 2005 model controls separately
  [@bergomi2005]. Here the vanillas leave the split between residual skew and leverage skew, and with it the
  forward smile, unpinned; see [what the vanillas identify](#what-the-vanillas-identify).

**What derivus does differently.**

- *The variance is specified through its log, not through the forward curve.* The two routes give
  one model in the limit above; the log-variance pair is the state the walk steps, the corner
  bounds and the historical estimator filters.
- *The residual is NIG on the variance clock.* Bergomi's spot is a diffusion given the variance: a
  Brownian one in the continuous form, and in the discrete form, which controls the short forward
  skew on its own, one with a local volatility function of the spot's move since the start of each
  interval ([@bergomi2008], after [@bergomi2005]). Here $\beta$ and $\alpha$ carry short-dated skew and
  tails beyond what leverage and vol-of-vol give.
- *Calendar buckets.* $\rho_s$, $\beta$, $\sigma_s$ and $\alpha$ move in calendar time, a lever that is
  not one of the factors. Bergomi notes that in a two-factor framework $k_1$ and $k_2$ shape both
  the decay of the skew and the term structure of vol-of-vol, which makes separate handles on the
  two difficult [@bergomi2008].
- *Uncorrelated factors.* The walk draws the two factors' shocks independently, where Bergomi's
  carry a correlation $\rho$.

## Forward smile and the term structure of skew {#forward-smile}

A forward-start option struck at $kS_{T_1}$ and expiring at $T_2=T_1+\Delta$ reads only the law of
$\log(S_{T_2}/S_{T_1})$, which given the state $(\ell_{T_1},s_{T_1})$ is the model started afresh
at $T_1$ under the buckets in force after it. Its smile is set by the dynamics, not by today's
vanillas, and the vanillas do not pin it ([what the vanillas identify](#what-the-vanillas-identify)).
The statements below are first-order or structural readings of the equations above; no test holds
them. **Not gated.**

**Two speeds for the skew term structure.** At first order in the vol-of-vols, without the corner,
with one bucket, a flat $\xi=\sigma_0^2$ and the `Gaussian` residual, the covariance of the
log-return with the forward variances $E_t[h_T]$ of [the lineage](#bergomi) gives the ATM-forward
skew at maturity $T$ (the expansion of [@bergomi2008], eq. 16, and [@bergomi2016], chapter 8, in
this model's variables):

$$\frac{\partial\sigma_\text{imp}}{\partial\ln K}\Big|_{K=F_T}\approx\tfrac12\Big(\rho_s\sigma_s\,g(\kappa_sT)+\rho_\ell\sigma_\ell\,g(\kappa_\ell T)\Big),\qquad
g(x)=\frac{x-1+e^{-x}}{x^2}.$$

$g(0)=\tfrac12$ and $g(x)\approx1/x$ for large $x$: each factor adds a flat $\rho\sigma/4$ at
maturities short against $1/\kappa$ and decays like $1/(\kappa T)$ past it. At the fit's
$\kappa_s=6$ and $\kappa_\ell=0.5$ the fast pair's share of the skew decays past about two months
and the slow pair's past about two years. Below two months both shares are flat and add, so only
maturities past the fast factor's reach tell the pairs apart, which is why the fit reads the slow
pair only off wings at `Slow_Horizon` (1.5y) or longer. A single factor ties the short and the long
skew to one speed.

**The residual for the short end and the wings.** From the cumulant function above, the residual
on a clock $A$ has skewness and excess kurtosis

$$\frac{3\beta}{\gamma^2\sqrt A}\qquad\text{and}\qquad\frac{3(\alpha^2+4\beta^2)}{\gamma^4A},$$

and a maturity's clock grows about in proportion to it, $E[A]=\sum_kc_k\delta_k\xi(t_k)$. The
residual's skew and tails therefore fade with maturity, carrying the short-dated smile and its
wings and leaving the long end to the leverage of the two factors. At fixed $\beta$ both vanish
as $\alpha\to\infty$, the `Gaussian` limit.

**Why the forward smile does not flatten.** Both factors are stationary OU processes about their
levels and the residual is a Lévy process on the variance clock, so a window of length $\Delta$
starting at $T_1$
has, given the state at $T_1$, the law of a spot interval of length $\Delta$ started from that
state. With one bucket the first-order forward skew of the window is the spot skew at maturity
$\Delta$, $\tfrac12\sum\rho\sigma\,g(\kappa\Delta)$, whatever $T_1$, and the spread of the state
at $T_1$, a mixture over variance levels, adds to the forward smile's curvature. The calendar buckets
make the forward skew move with $T_1$ where a forward-start source asks for it: a window past the
last quoted maturity reads the last bucket's $\rho_s$ and $\beta$ alone.

A volatility that depends on time alone makes every forward-start return lognormal, and its
forward smile is flat. A local volatility $\sigma(t,S)$ fitted to today's smile [@dupire1994]
generates a forward smile from its local volatilities at dates after $T_1$; on a surface whose skew
falls with maturity those are flatter than today's short-dated ones, so its forward smile is
flatter than the spot smile of the same window [@bergomi2004]. Here the forward skew is that of
$\rho\sigma$, $\kappa$ and the residual, which persist, and a forward-start or cliquet structure
priced on the model reads it. Where no source quotes the forward smile, the factor carries the
reserve line for this exposure ([below](#what-the-vanillas-identify)).

## Model choice {#model-choice}

LogVar2FJ is the one stochastic spot model the library prices path-dependent equity and FX deals
with; the other choice a deal has is `SpotModel: None`, lognormal dynamics off the implied surface
per fixing ([One-Step Survival](mc_simulation.md)). No local-volatility, Heston, separate Bergomi or
local-stochastic-volatility dynamics are implemented, and no numerical comparison against any of
them is claimed. What each family is known to do and to cost:

| Family | What it does well | What it costs |
| --- | --- | --- |
| Local volatility $\sigma(t,S)$ [@dupire1994] | reprices every vanilla of an arbitrage-free surface by construction | its forward smile flattens (above) and its smile dynamics are the surface's own; the law of a monitoring interval depends on the spot path inside it, so it has no closed form |
| Heston [@heston1993] | vanillas in semi-closed form; mean-reverting variance with leverage | one factor: the skew's term structure is one speed's, and the short forward skew, the spot/vol correlation and the term structure of vol-of-vol are tied together [@bergomi2005]; the square-root variance $dv=\kappa(\theta-v)\,dt+\sigma_v\sqrt v\,dW$ reaches zero where $2\kappa\theta<\sigma_v^2$, and its exact transition is non-central chi-squared, with no closed-form quantile |
| Bergomi's forward-variance models [@bergomi2005; @bergomi2008] | the forward-variance curve an input; several reversion speeds give the term structure of vol-of-vol and of the forward skew separate handles | vanillas by simulation; given its variance the continuous form's spot is a diffusion, so its short-dated skew is the leverage's alone |
| Local-stochastic volatility [@bergomi2016] | reprices the vanillas exactly and takes its forward smile from the stochastic part | its leverage function is calibrated through the expected variance given the spot, by a particle or PDE method; how much of the smile is left to the stochastic part is a choice the vanillas do not make; like local volatility, no closed-form interval law |

**Why this family, for these deals.** The deals that take the model
([below](#deals-and-calculations)) observe a trigger on dates and are priced by one-step survival,
which needs a closed-form law for each monitoring interval, and they report first and second
derivatives by AAD.

- *A closed-form interval law.* Given each step's two variance shocks and one inverse Gaussian
  mixer per interval, an interval's log-return is exactly Gaussian, so survival is one $\Phi$ and
  the continuing draw one $\Phi^{-1}$, the GBM arm's formulas at $(M,G)$
  ([the residual](#the-residual)). Heston's return is Gaussian given its variance path too; its
  variance's transition is the cost above.
- *Exact, smooth transitions.* Both factors are Ornstein-Uhlenbeck processes [@uhlenbeck1930]
  stepped by their exact Gaussian transitions, linear in their shocks, the variance positive by
  construction. The mixer is drawn by inverting its distribution function, a smooth function of one
  uniform that carries its implicit first and second derivatives (`utils.LogVar2FJ.ig_quantile`),
  and every parameter but the structural fields is a leaf, so AAD carries the estimator's first
  and second derivatives to each.
- *Two speeds and a residual.* The two reversion speeds give the short and the long skew separate
  handles, and the NIG residual, a Lévy process on the variance clock [@barndorffnielsen1998],
  carries short-dated skew and wings without breaking the conditional Gaussian.
- *The trading-day clock.* Every date a deal reads is a step end ([the clock](#the-clock)).
- *One model for three jobs.* The factor that prices a deal also drives the credit Monte Carlo's
  scenarios and is fitted to vanillas, so a CVA sensitivity reaches the leaf the pricer reads.

What it costs: vanillas have no closed form, so the fit prices them on a quadrature over a
surrogate clock or on the walk ([Model Limitations](../limitations.md#the-calibration)); the
vanillas identify neither the forward smile nor $\alpha$ well; the reversion speeds are declared,
not fitted; and the corner is a structural choice.

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
[Calc Lifecycle](../developer/calc_lifecycle.md#logvar2fj-outer).

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

### Deals and calculations {#deals-and-calculations}

A deal type opts in through `SpotModel: LogVar2FJ` in the
[Valuation Configuration](../json/model_configuration.md#valuation-configuration), so the switch
holds in every calculation that prices the deal.

| Deal type | Pricer | Refused under the model, by name | Held by |
| --- | --- | --- | --- |
| `EquityBarrierOption` | `pv_discrete_barrier_option` | the continuously monitored form (no `Barrier_Dates`); a `Quanto` or `Compo` payoff | `tests/test_logvar2fj_json.py::test_a_discrete_barrier_at_the_gbm_limit_is_the_gbm_barrier`, `::test_a_knock_in_every_scenario_has_crossed_is_the_european_on_its_own_walk` |
| `EquityBarrierBinaryOption` | `pv_discrete_barrier_option` | nothing of its own | **Not gated** |
| `QEDI_CustomAutoCallSwap` | `pv_MC_AutoCallSwap` | a `Compo` payoff; a `Quanto` whose `Correlation.EquityPrice.<surface>/FxRate.<pair>` factor is missing | `tests/test_logvar2fj_json.py::test_the_gbm_limit_reproduces_the_gbm_arm`, `::test_the_quanto_loading_is_off_at_zero_correlation` |
| `QEDI_CustomAutoCallSwap_V2` | `pv_MC_AutoCallSwap` | as the autocall above, whose declaration it inherits | **Not gated** |
| `FXTARFOptionDeal` | `pv_MC_Tarf` | nothing of its own; a deal on the reciprocal axis prices by the measure change of [Equity and FX](#equity-and-fx) | `tests/test_fx_tarf_json.py::test_the_seasoned_pot_is_one_number_on_every_arm` |
| `FXAccumulatorOptionDeal` | `pv_MC_Accumulator` | as the TARF | `tests/test_spot_model_martingale.py::test_a_quoted_forward_strip_solves_one_strike_from_either_side`, `tests/test_cross_spot_model.py::test_the_reciprocal_axis_carries_the_cross` |

| Calculation | What the model does there | Held by |
| --- | --- | --- |
| `BaseValuation` | the kit walks from $(L^*(t),0)$ at the base date | the deal rows above |
| `CreditMonteCarlo`, inner pricer | the kit walks per scenario row, from $(L^*(t_\text{row}),0)$, or from the $(\ell,s)$ an outer LogVar2FJ process on the same name carries | `tests/test_logvar2fj_json.py::test_the_gbm_limit_holds_through_the_cva` (a GBM term-structure outer); `tests/test_cross_spot_model.py::test_a_cross_under_an_outer_reads_its_own_law_and_re_seeds` |
| `CreditMonteCarlo`, outer process | `LogVar2FJImpliedSpotModel` simulates an `EquityPrice` or an `FxRate` on four correlated rows a name | `tests/test_logvar2fj_json.py::test_a_logvar2fj_equity_realises_the_rows_its_four_sub_factors_declare`, `tests/test_option_settlement_json.py::test_under_simulation_a_foreign_cashflow_deflates_to_its_value_today_on_every_row` |
| `ScenarioValuation` | the parameters are static: no scenario axis is carried for them, and a scenario stating the factor is priced in a pass of its own | `tests/test_simm.py::test_a_scenario_stating_a_models_parameters_is_priced_alone_on_them` |
| `SIMM` | the mapping states the parameter factor: null writes no row, so a trade priced under the model writes no vega row ([roadmap](../developer/roadmap.md)), and a vega risk type refuses by name | the refusal on a rate factor, `tests/test_simm.py::test_a_point_of_vega_on_what_is_no_vol_surface_is_refused_by_name`; on this factor **Not gated** |

Refused for every deal, by name: a `SpotModel` the deal type does not declare (`Deal.spot_models`)
(**Not gated**); a `Steps_Per_Year` restated at another value ([the clock](#the-clock)); the
reciprocal axis under any other model (`instruments.spot_model_reciprocal_axis`); a retired key
([Parameters](#parameters)); and, in the outer process, a replay of an observed path, since a price
path does not determine $(\ell,s)$. A switch whose factor the market data does not carry skips the
deal, naming the factor, and never prices it on GBM instead. **Held by**
`tests/test_cross_spot_model.py::test_a_key_that_cannot_resolve_skips_the_deal_and_never_kills_the_job`.

### Equity and FX {#equity-and-fx}

On an **EquityPrice** the carry is the equity's own zero rate less its dividend rate; on an
**FxRate** it is the domestic less the foreign zero rate, as for GBM.

An **FxRate** is a currency priced in the base, and a fitted law describes one token priced in
another. A pair with a base leg is its non-base token priced in the base. A cross is the
alphabetically later token priced in the earlier: EUR/ZAR is the rand priced in the euro, its spot
the ratio of the two base-priced rates (the block's **Priced_In**), filed as
`LogVar2FJModelParameters.ZAR.EUR`, whichever way the deal or the surface is written. Detail:
[accrual legs](../developer/structures.md#accrual).

**A deal paying on the reciprocal axis is a change of numeraire, not a second fit** — left out,
the price carries Siegel's convexity [@siegel1972]. A deal whose
underlying is the currency the law is priced in (an FX TARF or accumulator written the other way
up) pays on $1/S$, and the kit prices it under the $S$ numeraire (`Invert_Spot`). With $R_k$ the
day's log-return, the density $e^{R_k-b_k}$ factorises over the day's own draws,
so each variance shock shifts by its loading, $\eta^s_k\sim N(\rho_s\sqrt{V_k},1)$ and
$\eta^\ell_k\sim N(\rho_\ell\sqrt{V_k},1)$, and the residual takes the Esscher transform [@gerber1996] at one,
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

and is reported per deal and for the portfolio. The priors that hold what the ladder cannot see
are on [the fit's page](../bootstrapping/fx_and_equity.md#logvar2fjmodelparameters), with the
weights each carries, and the design rule behind them on
[Market Prices](../developer/market_prices.md#logvar2fj-priors).
