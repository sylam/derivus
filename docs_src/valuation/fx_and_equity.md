## Single Barrier Options

A single barrier option is a European option knocked in or out when the underlying touches the barrier
$H$ before expiry. A knock-out pays its rebate $R$ (**Cash_Rebate**) at the touch; a knock-in never
knocked in pays $R$ at expiry.

### Closed form

The continuously monitored value (Merton [@merton1973]; Reiner-Rubinstein, as tabulated in Haug [@haug]) is a sum of six terms.
With $\phi=+1$ for a call and $-1$ for a put, $\eta=+1$ for a down and $-1$ for an up barrier,
$S=S(t)$, $v=\sigma\sqrt\tau$, $\mu=(b-\sigma^2/2)/\sigma^2$ and $\lambda=\sqrt{\mu^2+2r/\sigma^2}$:

$$A=\phi\big(Se^{(b-r)\tau}\Phi(\phi x_1)-Ke^{-r\tau}\Phi(\phi(x_1-v))\big)$$

$$B=\phi\big(Se^{(b-r)\tau}\Phi(\phi x_2)-Ke^{-r\tau}\Phi(\phi(x_2-v))\big)$$

$$C=\phi\Big(Se^{(b-r)\tau}\big(\tfrac{H}{S}\big)^{2(\mu+1)}\Phi(\eta y_1)-Ke^{-r\tau}\big(\tfrac{H}{S}\big)^{2\mu}\Phi(\eta(y_1-v))\Big)$$

$$D=\phi\Big(Se^{(b-r)\tau}\big(\tfrac{H}{S}\big)^{2(\mu+1)}\Phi(\eta y_2)-Ke^{-r\tau}\big(\tfrac{H}{S}\big)^{2\mu}\Phi(\eta(y_2-v))\Big)$$

$$E=Re^{-r\tau}\Big(\Phi(\eta(x_2-v))-\big(\tfrac{H}{S}\big)^{2\mu}\Phi(\eta(y_2-v))\Big)$$

$$F=R\Big(\big(\tfrac{H}{S}\big)^{\mu+\lambda}\Phi(\eta z)+\big(\tfrac{H}{S}\big)^{\mu-\lambda}\Phi(\eta(z-2\lambda v))\Big)$$

$$x_1=\frac{\log(S/K)}{v}+(1+\mu)v,\quad x_2=\frac{\log(S/H)}{v}+(1+\mu)v,\quad y_1=\frac{\log(H^2/(SK))}{v}+(1+\mu)v,\quad y_2=\frac{\log(H/S)}{v}+(1+\mu)v,\quad z=\frac{\log(H/S)}{v}+\lambda v$$

$E$ is the knock-in's rebate paid at expiry if never touched and $F$ the knock-out's rebate paid at the
touch.

| Barrier type | $K>H$ | $K\le H$ |
|---|---|---|
| Down-and-out call | $A-C+F$ | $B-D+F$ |
| Up-and-out call | $F$ | $A-B+C-D+F$ |
| Down-and-out put | $A-B+C-D+F$ | $F$ |
| Up-and-out put | $B-D+F$ | $A-C+F$ |
| Down-and-in call | $C+E$ | $A-B+D+E$ |
| Up-and-in call | $A+E$ | $B-C+D+E$ |
| Down-and-in put | $B-C+D+E$ | $A+E$ |
| Up-and-in put | $A-B+D+E$ | $C+E$ |

### On the scenario grid

`FXBarrierOption`, and `EquityBarrierOption` without **Barrier_Dates**, apply the closed form on every
scenario date $t_i$ of the deal:

- $\sigma=\max(\sigma(m,\tau),\,0.05)$: the surface vol at the deal's moneyness is **floored at 5%**, a
  model adjustment whose effect is unmeasured (it binds only where the surface is below 5%).
- The touch probability $P_i$ accumulates between scenario dates by the Brownian-bridge crossing
  probability [@beaglehole1997] of the barrier actually priced (after the shift below), formula (B) below.
- A knock-in reports $(1-P_i)V_i^{\text{closed}}+P_iV_i^{\text{vanilla}}$, the vanilla being Black at the
  floored vol, and pays $R(1-P_T)$ at expiry. A knock-out reports $(1-P_i)V_i^{\text{closed}}$ and pays
  $R(P_i-P_{i-1})$ as cash on $t_i$. At expiry the surviving branch is the intrinsic on the forward.
- An FX barrier whose **Payoff_Currency** differs from **Currency** is not priced as a quanto: the
  value is the **Currency** option, and a warning is logged.

The touch probability (B), with $d=\eta\log(S/H)$:

$$P_i=P_{i-1}+(1-P_{i-1})\max\big(\mathbf 1_{\{S_i\text{ beyond }H\}},\,e^{-2d_{i-1}d_i/w_i}\big)$$

"Beyond" is strict: a spot exactly on $H$ has not touched. $w_i$ is the simulation's own variance rate
for the barrier underlying times the days between the two dates over 365.25. Where the underlying publishes no variance rate, and on the first date, only the
endpoints are tested.

### One touch and No touch Binary Options and Rebates

A one-touch pays **Cash_Payoff** if the barrier is touched before expiry, at the touch
(**Payment_Timing** `Touch`) or at expiry (`Expiry`); a no-touch pays it at expiry if the barrier is never
touched. The no-touch value is

$$e^{-r\tau}\Big[\Phi(\eta x_2-\eta\sigma\sqrt\tau)-\Big(\frac{H}{S(t)}\Big)^{2\mu}\Phi(\eta y_2-\eta\sigma\sqrt\tau)\Big]$$

and a one-touch paid at expiry is $e^{-r\tau}$ less it. A one-touch paid at the touch is

$$\Big(\frac{H}{S(t)}\Big)^{\mu+\lambda}\Phi(\eta z)+\Big(\frac{H}{S(t)}\Big)^{\mu-\lambda}\Phi(\eta z-2\eta\lambda\sigma\sqrt\tau)$$

with $z$ and $\lambda$ as above; the argument $\mu^2\sigma^2+2r$ of $\lambda$'s root is floored smoothly at
zero (a quadratic blend over $\pm0.005$). The vol is read at the barrier's moneyness and is not floored.
On the scenario grid the touch probability $P_i$ is the bridge probability above: before a touch the value
is $(1-P_i)$ times the closed form; a touched one-touch paid at expiry carries $P_i\,e^{-r\tau}$, one paid at
the touch settles $P_i-P_{i-1}$ on $t_i$. A no-touch reports $(1-P_i)(e^{-r\tau}-\text{one-touch})$.

### Discontinuous Barrier Sampling

A barrier observed every $\delta$ years rather than continuously is priced as a continuous one against
the shifted level of Broadie, Glasserman and Kou [@broadie1997]:

$$H'=He^{\pm\beta\sigma\sqrt\delta},\qquad \beta=-\frac{\zeta(1/2)}{\sqrt{2\pi}}\approx0.5826$$

$\zeta$ is the Riemann zeta function. The sign follows the barrier TYPE, not the spot: an up barrier is
shifted up and a down barrier down. $\delta$ is the calendar days in **Barrier_Monitoring_Frequency** over
365; `0M` is continuous monitoring. An equity one-touch with **Barrier_Dates** takes $\delta$ as the mean
gap between those dates and stays continuously monitored: the dates are not an observation schedule.

## Partial Barrier Options

`FXPartialTimeBarrierOption` monitors the barrier over a window only. With $t_\ell$ the
**Barrier_Limit_Date**, $\tau_\ell=t_\ell-t$ and $\tau=T-t$, **Barrier_At_Start** `Yes` monitors
$[t,t_\ell]$ and `No` monitors $[t_\ell,T]$. The knock-out call is the Heynen-Kat closed form as tabulated
in Haug [@haug] (type A for a start window, type B2 for an end window), with $M(a,b;\rho)$ the bivariate normal
distribution (Genz [@genz2004]) and $\rho=\sqrt{\tau_\ell/\tau}$. Define $F=Se^{b\tau}$, $\mu'=2b/\sigma^2-1$,
$v=\sigma\sqrt\tau$, $v_\ell=\sigma\sqrt{\tau_\ell}$, $k=\log(S/K)$, $h=\log(S/H)$, $c=b+\sigma^2/2$ and

$$d_1=\frac{k+c\tau}{v},\ f_1=\frac{k-2h+c\tau}{v},\ g_1=\frac{h+c\tau}{v},\ g_3=g_1-\frac{2h}{v},\ e_1=\frac{h+c\tau_\ell}{v_\ell},\ e_3=e_1-\frac{2h}{v_\ell}$$

with $d_2=d_1-v$, $f_2=f_1-v$, $g_2=g_1-v$, $g_4=g_3-v$, $e_2=e_1-v_\ell$, $e_4=e_3-v_\ell$, and

$$\mathcal P(p_1,\dots,p_8;\rho_1,\rho_2)=F\Big[M(p_1,p_2;\rho_1)-\big(\tfrac{H}{S}\big)^{\mu'+2}M(p_3,p_4;\rho_2)\Big]-K\Big[M(p_5,p_6;\rho_1)-\big(\tfrac{H}{S}\big)^{\mu'}M(p_7,p_8;\rho_2)\Big]$$

| Window, barrier | Knock-out call $\times e^{r\tau}$ |
|---|---|
| start, $\eta=\pm1$ | $\mathcal P(d_1,\eta e_1,f_1,\eta e_3,d_2,\eta e_2,f_2,\eta e_4;\eta\rho,\eta\rho)$ |
| end, down, $K<H$ | $\mathcal P(g_1,e_1,g_3,-e_3,g_2,e_2,g_4,-e_4;\rho,-\rho)$ |
| end, down, $K\ge H$ | $\mathcal P(d_1,e_1,f_1,-e_3,d_2,e_2,f_2,-e_4;\rho,-\rho)$ |
| end, up, $K<H$ | $\mathcal P(-g_1,-e_1,-g_3,e_3,-g_2,-e_2,-g_4,e_4;\rho,-\rho)-\mathcal P(-d_1,-e_1,e_3,-f_1,-d_2,-e_2,e_4,-f_2;\rho,-\rho)$ |
| end, up, $K\ge H$ | $0$ |

- A put is the call under the put-call transformation $(S,K,H,r,b,\eta)\to(K,S,SK/H,r-b,-b,-\eta)$.
- A knock-in is the Black vanilla (forward $Se^{b\tau}$, the same $\sigma$) less the knock-out.
- A start window whose spot is already beyond the barrier is touched: the knock-out is 0 and the
  knock-in the vanilla. A start window already closed ($t_\ell\le t$) leaves the knock-out the vanilla and
  the knock-in worthless. Closed takes precedence.
- **Cash_Rebate** is an absolute amount, paid at the touch by a knock-out and at expiry by a knock-in never
  touched. Its legs are first-passage values over the window: for a start window the no-touch
  probability of $E$ (still discounted from expiry) and the touch value of $F$, both with $\tau_\ell$ in
  place of $\tau$; for an end window their bivariate analogues with correlation $\rho$, a path already
  beyond the barrier at $t_\ell$ touching there.
- $\sigma$ is one surface vol at the strike's moneyness and the expiry, used for both the window and the
  remaining life, and is not floored; using one vol for the two periods is an approximation, unmeasured.
- On the scenario grid the bridge touch probability of Single Barrier Options accumulates on the rows
  inside the window only, against the barrier shifted by Broadie-Glasserman-Kou for
  **Barrier_Monitoring_Frequency**. A knock-in reports $P_i$ times the vanilla plus $(1-P_i)$ times the
  closed form; a knock-out $(1-P_i)$ times the closed form plus its rebate increments as cash.
- **Payoff_Currency** names the settlement currency only; a payoff currency other than **Currency** is
  not quanto-adjusted and not refused.
- Sensitivities through an endpoint-tested touch (no bridge variance) register the decision under
  `Boundary_AAD_Window_Touch` (on by default). On one test deal with six live decisions (five seeds,
  32,768 paths) this gives the CVA delta the sign a common-random-number finite-difference ladder
  confirms; the ladder itself moved 13% to 35% over bumps $h=5\times10^{-4}$ to $10^{-2}$, so the magnitude
  is known to about a third.

## Discrete Asian Options

A discrete Asian option pays $\max(\delta(\bar S-K),0)$ at expiry $T$, times **Underlying_Amount** (FX)
or **Units** (equity), with

$$\bar S=\sum_{i=1}^n\omega_iS(t_i),\qquad \omega_i=d_i\Big/\sum_{k=1}^n d_k,$$

$t_i$ and $d_i$ the dates and weights of **Sampling_Data**, $K$ the strike and $\delta=\pm1$ for a call or
put. **Is_Digital** `Yes` pays the amount itself when $\delta(\bar S-K)>0$. The value is discounted from
**Expiry_Date** on **Discount_Rate**.

A sample dated before the base date is the **Price** its row states; one between the base date and $t$
is the simulated spot on that sample's scenario date (today's spot for an unsimulated underlying). A
past sample with no price is logged as an error and not refused.

For valuation date $t$ let $m$ be the number of samples on or before $t$. Then $\bar S-K=A-\bar K$ with

$$A=\sum_{i=m+1}^n\omega_iS(t_i),\qquad \bar K=K-\sum_{i=1}^m\omega_iS(t_i).$$

If $\bar K\le0$ the option is exercised with certainty and valued as $\delta(F-\bar K)^+$ on the forward
$F$ below. Otherwise $A$ is matched to a lognormal on its first two moments (Turnbull-Wakeman [@turnbull1991], Levy [@levy1992]):

$$F=\mathbb E_t[A]=\sum_{i>m}\omega_iF(t,t_i),\qquad F^2e^{v^2}=\sum_{i,j>m}\omega_i\omega_jF(t,t_i)F(t,t_j)e^{\sigma_{i\wedge j}^2\tau_{i\wedge j}}$$

with $\tau_i=t_i-t$ and $i\wedge j=\min(i,j)$, and the option is $\mathcal B_\delta(F,\bar K,v)D(t,T)$ (a
digital $\Phi(\delta d_2)D(t,T)$). The inputs are

- $F(t,t_i)$, the forward to each sample's own date on the discount and dividend (equity) or foreign
  (FX) curves, read as the deal's forward to its expiry is;
- $\sigma_i=\sigma(K',\tau_i)$, each sample's vol at its own tenor and at the moneyness of
  $K'=\bar K/\sum_{i>m}\omega_i$ against that sample's forward.

Past the last sample the value is the intrinsic on the realised average. An equity Asian with
**Payoff_Type** `Quanto` adjusts each sample's carry and vol at its own tenor; `Compo` averages $S\cdot X$
(see [Quanto and Compo Payoffs](definitions.md#quanto-and-compo-payoffs)).

### Discrete Double Asian Options

`FXDiscreteExplicitDoubleAsianOption` is a spread option on two discretely sampled averages,
**Sampling_Data_1** and **Sampling_Data_2** (each normalised as above), paying at expiry, times
**Underlying_Amount**,

$$\max\big(\delta(\alpha_1\bar S_1-\alpha_2\bar S_2-\alpha_0K),0\big)$$

with $\alpha_0$ **Strike_Multiplier**, $\alpha_1$ **Sampling_Multiplier_1** and $\alpha_2$
**Sampling_Multiplier_2**. Realised samples fold into the strike,
$\bar K=\alpha_0K-\alpha_1\sum_{\text{past}}\omega^1_iS(t_i)+\alpha_2\sum_{\text{past}}\omega^2_iS(t_i)$, and each
strip's remaining average is moment-matched as for the single Asian, giving $F_1,v_1$ and $F_2,v_2$, every
sample forwarded to its own date with its vol read at its own tenor and at the strike equal to today's
spot. Their log-correlation is matched from the cross moment, $\sigma_{ij}^2\tau_{ij}$ being the variance
to whichever of $t^1_i$ and $t^2_j$ is earlier:

$$M_\rho=\sum_{i,j}\omega^1_i\omega^2_jF(t,t^1_i)F(t,t^2_j)e^{\sigma_{ij}^2\tau_{ij}},\qquad \rho=\frac{\log M_\rho-\log F_1-\log F_2}{v_1v_2}$$

- Both strips live: the Bjerksund-Stensland spread formula [@bjerksund2014] on $(\alpha_1F_1,\alpha_2F_2,\bar K,v_1,v_2,\rho)$,
  discounted from the expiry.
- One strip fully fixed: Black on the live strip, struck at $\bar K$ (first strip live) or at $-\bar K$ with
  the option type reversed (second strip live), and exercised with certainty where that strike is not
  positive.
- Both fixed, or at expiry: the intrinsic, smoothed within $\pm0.005$ of the strike (a quadratic blend,
  in price units).
- The spread formula needs $\alpha_2F_2+\bar K>0$; outside that domain the value is undefined (not
  guarded).

## Autocallable Swaps

`QEDI_CustomAutoCallSwap` and `QEDI_CustomAutoCallSwap_V2` pay contingent coupons that end the deal
(autocall) and a put that pays on a barrier breach; V2 adds a floating leg. Notation, all from the deal:

| Symbol | Field |
|---|---|
| $N$ | **Units**, signed $+$ for **Buy** and $-$ for **Sell** |
| $K$ | **Strike_Price** |
| $T_j,\ c_j$ | the **Autocall_Coupons** dates and values |
| $h_j$ | the **Autocall_Thresholds** row in the same POSITION (rows pair by position, whatever date they carry) |
| $W_j,\ A_j$ | coupon $j$'s window of **Price_Fixing** dates and the arithmetic mean of the spot over it |
| $q$ | **FixFXRate** under `Payoff_Type: Quanto` with a payoff currency other than **Currency**, else 1 |
| $R_p$ | **Rebate** (default 0) |
| $B$ | the put barrier: **Barrier** $\times K$ on V1, **Barrier** itself on V2; 0 means no put |

### Payoff

On a path not autocalled before the date, in payoff currency:

| Date | Condition | Pays |
|---|---|---|
| $T_j$ | $A_j\ge h_jK$ | $N\,q\,c_j$, and the deal terminates |
| $T_j$ carrying a put barrier date | not autocalled at $T_j$, and $X_j\le B$ | $N\,q\,(A_j/K-1+R_p)$ |
| each **Autocall_Floating** date $d_k$ (V2 only) | always | $-N\,q\,(\ell_k+m)\alpha_k$ |

$X_j$ is what **Barrier_Observation** reads: `Spot` (default) the spot at the window's last fixing,
`Average` the mean $A_j$; with one fixing per window the two agree. The put pays on the average whichever
is observed, does not end the deal, and owes the whole loss below $K$ at $R_p=0$. No principal is
exchanged. A floating payment on a coupon date is paid whatever that date's autocall decides.

- **Windows.** Where **Coupon_Observations** is authored, each coupon observes exactly the one fixing its
  row names. Otherwise coupon $j$ observes every fixing after $T_{j-1}$ up to and including $T_j$; where
  every window after the first then holds a single fixing, the first window keeps only its last fixing
  (the earlier ones are strike observations). Fixings on or before the last coupon dated before the
  base date are dropped.
- **Barrier dates.** A **Barrier_Dates** row dated on a coupon names that coupon; one dated on a fixing
  names the coupon whose window holds it. Either way the put is observed on that window and paid on the
  coupon date.
- **Floating leg (V2).** Periods run between consecutive **Autocall_Floating** dates, the first starting
  **Reset_Frequency** before the first date. $\alpha_k$ is the **Forecast_Rate** curve's day count, $\ell_k$
  the simply compounded forward over the period on **Forecast_Rate** (ACT/365), $m$ the
  **Floating_Margin** in basis points. A period started before the base date pays its table value, the
  period's amount per unit notional with the margin in it. A **Buy** pays the leg.

### Valuation

By one-step-survival Monte Carlo [@glasserman2001] ([One-Step Survival](../theory/mc_simulation.md)), an inner simulation
from every outer scenario row. Coupon $j$ survives with conditional probability
$p_j=\mathbb P(A_j<h_jK\mid\cdot)$, the surviving weight is $L_j=p_jL_{j-1}$ ($L_0=1$), and the inner
estimate is

$$V(t)=N\,\mathbb E_t\Big[\sum_j(1-p_j)L_{j-1}\,q\,c_j\,D(t,T_j)+\sum_{j\in\mathcal B}L_j\,q\Big(\frac{A_j}{K}-1+R_p\Big)\mathbf 1_{\{X_j\le B\}}D(t,T_j)-\sum_kL_{k^-}\,q\,(\ell_k+m)\alpha_k\,D(t,d_k)\Big]$$

with $\mathcal B$ the coupons carrying a barrier date and $L_{k^-}$ the weight before date $d_k$'s decision;
it is converted to the reporting currency at spot. Within a window the later fixing-to-fixing blocks are
drawn first and the survival truncates the return to the window's first fixing, so $p_j$ is exact given
those draws. A fixing already observed is an exact indicator.

- `SpotModel: None` (default): lognormal over each fixing interval. Carry is the interval strip of the
  equity's repo curve less its dividend curve; the vol is the interval forward-variance strip read at each
  coupon's own trigger level $h_jK$, and the put leg walks a second strip read at its own strike, $B$ where
  $R_p=0$ and $K(1-R_p)$ otherwise, off the same uniforms. `Quanto` bends the carry by $-\rho\sigma_S\sigma_X$
  at the expiry's at-the-money vols; `Compo` simulates $S\cdot X$ with each interval vol composed with the
  expiry at-the-money FX vol.
- `SpotModel: LogVar2FJ`: each interval is the model's block law ([Stochastic
  Volatility](../theory/stochastic_volatility.md)), one walk carrying one smile for every leg; `Quanto`
  enters as the per-step drift $-\rho\,\sigma_{FX,k}\sqrt{V_k}$, $\sigma_{FX}$ the FX surface's at-the-money
  forward strip on the walk's grid.
- Inner paths: **MCMC_Simulations** per outer scenario, Sobol where the simulation batch exceeds 16, with
  no antithetic mirror. Under `Branch_And_Weight` (the default, base valuation only) the put leg is
  integrated against the interval's law rather than sampled.

### Refusals and limitations

Refused by name at compile: a stated `BarrierIsHit`; a **Coupon_Observations** table that leaves a coupon
out, names a fixing not in **Price_Fixing** or after its coupon, or does not run in coupon order; fixings
that do not give each coupon, past ones included, a non-empty window of its own (or fall after the last
coupon); a barrier date on no coupon and no fixing; an `Average` barrier whose window holds no fixing; an
**Autocall_Coupons** row of zero or less; a **Steps_Per_Year** differing from the fitted parameter
factor's. Skipped as expired, naming the date: a note a coupon before the base date called, its window's
printed mean at or above $h_jK$. Skipped by name: `Compo` under `LogVar2FJ`, `Quanto` under `LogVar2FJ`
without the `Correlation` factor, `LogVar2FJ` without its parameter factor, an equity with no model.

- Logged and not refused: a threshold of zero or less is replaced by 1; a past fixing of 0 is priced as 0.
- **Option_Type**, **Option_Style**, **Settlement_Style** and **Option_On_Forward** are accepted and not read.
- Under `SpotModel: None` the trigger and put legs read different vol strips, one smile-less world per
  strike; the quanto adjustment reads one at-the-money pair at the expiry. Neither is measured against a
  joint smile model.

## Equity Swaps

Equity swap legs consist of a series of equity swaplets.

### Equity Swaplet

An equity swaplet has:

- a cashflow $A$
- start date $t_0$
- end date $t_1$
- payment date $T$
- a start multiplier $a_0$
- an end multiplier $a_1$
- a dividend multiplier $b$

For cashflow dates $t_0\le t_1\le T$, the equity swaplet pays:

$$A\Big( \frac{a_1 S(t_1)-a_0 S(t_0)+bH(t_0,t_1)}{S(t_0)^a}\Big)$$

at time $T$, where

- $S(t)$ is the equity price
- $H(t_0,t_1)$ is the time-$t_1$ value of dividends per share with ex-dividend date after $t_0$ but on
or before $t_1$
- $a=1$ if the amount type is **Principal** and $a=0$ if it is **Shares**.

The swaplets have $a_0=a_1=1$ and either $b=1$ if **Include Dividends** is **Yes** or $b=0$ otherwise.

The formula used to value the swaplet at $t\le t_0$ is

$$A\Big((a_1-b)\frac{F(t,t_1)}{F(t,t_0)^a}+ \Big( b\frac{D_r(t,t_0)}{D_r(t,t_1)}-a_0\Big)
F(t,t_0)^{1-a}\Big)D(t,T)$$

where:

- $F$ is the forward equity price
- $D$ is the usual discount factor
- $D_r$ is the discount factor from the equity's repo rate

The value of the swaplet at $t_0\le t\lt t_1$ is

$$A\Big(\frac{(a_1-b)F(t,t_1)+ \frac{b(S(t)+H(t_0,t))}{D_r(t,t_1)}-a_0S(t_0)}{S(t_0)^a}\Big)
D(t,T)$$

The value of the swaplet for $t_1\le t\le T$ is the payoff multiplied by $D(t,T)$.

When $t\ge t_0$, the dividend payoff $H(t_0,t\wedge t_1)$ is calculated from **Known Dividends**
and/or the simulated spot and forward prices along the current scenario path.

Dividends are paid at the swaplet's payment date: the swaplet pays $\frac{AbH(t_0,t_1)}{S(t_0)^a}$ at $T$.
**Dividend_Timing** `Continuous` is accepted and priced as `Terminal`.
