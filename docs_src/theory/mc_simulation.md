## One-Step Survival (OSS) Monte Carlo

The path-dependent equity and FX pricers that observe a trigger on discrete fixing dates value
each reporting row by an **inner** Monte Carlo that conditions every simulated step on the
trigger's outcome [@glasserman2001]: at each monitored fixing the probability $p_j$
that the step does *not* fire is computed in closed form, the fired mass $(1-p_j)$ is paid its
payoff analytically, and only the surviving branch is sampled, from the step's law truncated to
the survival region. The path weight then carries the survival probability instead of a 0/1
indicator, so the estimator is unbiased, has no indicator to differentiate, and gives AAD a
pathwise derivative through every decision ([@alm2013], for the
autocall).

| Pricer (`derivus/pricing.py`) | Deals | Trigger $H_j$ | Fired branch pays |
|---|---|---|---|
| `pv_discrete_barrier_option` | `EquityBarrierOption` with `Barrier_Dates`, `EquityBarrierBinaryOption` | `Barrier_Price` | knock-out: `Cash_Rebate` at $t_j$; knock-in: by parity, below |
| `pv_MC_Accumulator` | `FXAccumulatorOptionDeal` | `Barrier_Price` | nothing; later fixings cancelled |
| `pv_MC_Tarf` | `FXTARFOptionDeal` | spot at which the fixing fills the target, $K+\phi R_{j-1}/N_1$ (inverted: $1/(1/K-\phi R_{j-1}/N_1)$) | the remaining target $R_{j-1}$ |
| `pv_MC_AutoCallSwap` | `QEDI_CustomAutoCallSwap`, `QEDI_CustomAutoCallSwap_V2` | coupon threshold $\times$ `Strike_Price`, tested on the coupon window's average | the coupon |
| `pv_MC_ExtendableForward` | `FXExtendableForwardDeal` | the zero-continuation boundary $H_j$ | the terminated branch: nothing further |

Not on this page: the continuously monitored barrier (`EquityBarrierOption` without
`Barrier_Dates`, `FXBarrierOption`) and the partial-time barrier are closed forms weighted by a
Brownian-bridge touch probability [@beaglehole1997], with the Broadie-Glasserman-Kou shift [@broadie1997] for a discrete monitoring
frequency ([FX and Equity](../valuation/fx_and_equity.md)).

### What each reporting row supplies

The inner simulation starts afresh at every reporting row $t$ (the base date in a base valuation,
each scenario date in a `CreditMonteCarlo` or `HedgeMonteCarlo`) and reads only that row's state:

| Input | Source at row $t$ |
|---|---|
| spot $S(t)$ | the outer scenario's spot (today's in a base valuation) |
| discount factors $D(t,T_j)$ | the scenario's discount curve, to each fixing's payment or settlement date |
| zero carry $b_j$ | the scenario's spread of two zero curves to $\tau_j$: repo less dividend yield (equity), domestic less foreign (FX) |
| implied vols | the deal's surface; surfaces are static, so the base-date surface is read at the tenor $\tau_j$ measured from $t$ |
| observed fixings | stated history, and the outer path's spot on fixing dates the grid has reached |
| deal state | barrier hit mask (strictly earlier observations), autocall termination latch, TARF accrued target, accumulator alive state, extendable exercise history |
| LogVar2FJ state $(\ell,s)$ | the outer LogVar2FJ path's carried state where one is published, else re-seeded at $(L^*(t),0)$ |

$\tau_j$ is the time from $t$ to fixing $j$ in the discount curve's day count,
$\Delta\tau_j=\tau_j-\tau_{j-1}$, $\tau_0=0$.

### The interval law under GBM (`SpotModel: None`)

Carry and variance are both cumulative quantities differenced into per-interval rates
(`forward_carry_rate`, `forward_vol_strip`, `forward_vol_rate`):

$$\beta_j=\frac{b_j\tau_j-b_{j-1}\tau_{j-1}}{\Delta\tau_j},\qquad
w_j=\sigma(m_j,\tau_j)^2\,\tau_j,\qquad
v_1=w_1,\quad v_j=\max\big(w_j-w_{j-1},\ \epsilon\big)\ (j\ge2),$$

$$X_j=\ln\frac{S_j}{S_{j-1}}\sim\mathcal N\big(\mu_j,\ v_j\big),\qquad
\mu_j=\beta_j\Delta\tau_j-\tfrac12 v_j,\qquad
\ln\frac{F(t,T_N)}{S(t)}=\sum_{k\le N}\beta_k\Delta\tau_k=b_N\tau_N.$$

$\sigma(m,\tau)$ is the surface read at the deal's strike $K$ and fixing $j$'s tenor; $\epsilon$ is
the job dtype's machine epsilon; the last identity is `total_log_forward`. The strip telescopes to
the surface's own total variance, so every European limit is exact; only the allocation of
variance between fixings is a modelling choice.
Where the surface's cumulative variance falls between two tenors (calendar arbitrage) the floor
$\epsilon$ binds, the strip's total then exceeds $w_N$, and in-out parity against the simulated
vanilla breaks; the remedy is an arbitrage-free surface.

| Pricer | Moneyness $m_j$ the strip reads | Strike the strip is read at |
|---|---|---|
| discrete barrier | as the deal declares (equity: $S/K$ on an explicit surface) | `Strike_Price` |
| accumulator, TARF | forward, $F(t,T_j)/K$ (or $K/F$ under `Invert_Moneyness`) | `Strike_Price` |
| autocall | as the deal declares | each fixing at its own coupon's threshold $\times$ strike; the put leg a second strip at its own strike (the barrier with `Rebate` 0, else strike $\times(1-$`Rebate`$)$) |
| extendable forward | forward | `Extension_Strike` $K_2$ (the smile frozen at $K_2$) |

Quanto: each fixing's carry gains $-\rho\,\sigma_S\sigma_X$, both vols the ATM reads at the deal's
expiry. Compo: the simulation steps $S\cdot X$, each interval vol composed with the FX expiry ATM vol,
$\sqrt{\sigma^2+2\rho\sigma\sigma_X+\sigma_X^2}$, and the local smile read at the strike over the
FX forward to each fixing.

### The interval law under LogVar2FJ (`SpotModel: LogVar2FJ`)

The kit (`LogVar2FJKit`) walks the [LogVar2FJ](stochastic_volatility.md) state once per reporting
row, one trading day per internal step on the parameter factor's `Steps_Per_Year` clock, every
fixing on a grid point. Conditional on the walk's shocks and one inverse-Gaussian mixer per
fixing interval (one more per parameter-curve knot inside it), the interval's log return is
Gaussian, $X_j\mid\mathcal W\sim\mathcal N(M_j,\Sigma_j^2)$, carry $\beta_j\Delta\tau_j$ included.
Every OSS formula below is then the GBM one with $(\mu_j,\sqrt{v_j})\to(M_j,\Sigma_j)$; the implied
surface is not read. Under `Invert_Spot` the walk and mixer change measure to the reciprocal axis.
The walk's normals come from a per-segment `torch.Generator` in float32, never from the Sobol
block; the mixers' uniforms are drawn beside the OSS uniforms. The kit serves the discrete barrier,
accumulator, TARF and autocall; the extendable forward is GBM only.

### One-step survival

With $z_j=(\ln(H_j/S_{j-1})-\mu_j)/\sqrt{v_j}$ and a uniform $u_j$:

| Survival region | $p_j$ | Continuing draw $Z_j$ |
|---|---|---|
| below the trigger ($S_j\le H_j$) | $\Phi(z_j)$ | $\Phi^{-1}(u_j\,p_j)$ |
| above the trigger ($S_j\ge H_j$) | $1-\Phi(z_j)$ | $\Phi^{-1}\big((1-p_j)+u_j\,p_j\big)$ |

$$S_j=S_{j-1}e^{\mu_j+\sqrt{v_j}Z_j},\qquad L_j=p_j\,L_{j-1},\ L_0=1,\qquad
\text{fired mass}_j=(1-p_j)\,L_{j-1}.$$

The argument of $\Phi^{-1}$ is clamped to $[\epsilon,1-\epsilon]$. The fired mass is spelled
$1-p_j$ off the same $p_j$ the draw was truncated with, so per path
$\sum_j(1-p_j)L_{j-1}+L_N=L_0$ to roundoff (`SurvivalLedger` logs the residual at DEBUG under the
smooth estimator). A fired payoff measurable at $t_{j-1}$ (a rebate, the TARF's remaining target,
a coupon) is its own conditional expectation; a leg that depends on $S_j$ over one tail is
integrated over that tail (a digital's terminal step always; the TARF's knock-in and the autocall's
put under the smooth estimator, below) by the partial moments of the interval law,

$$E\big[S_j^{a}\,\mathbf 1\{Z_j>z\}\big]=S_{j-1}^{a}\,e^{a\mu_j+a^2v_j/2}\,\Phi\big(a\sqrt{v_j}-z\big)$$

(`lognormal_partial_moment`, $a=1$, or $a=-1$ for an inverted accrual; the down side reflected).

**Observed fixings.** A fixing on or before the row is data: survival is the exact indicator on
the observed value and no step is simulated. A fixing dated on the row itself is a zero-length
interval resolved off the row's own spot. In the discrete barrier that interval's variance is
floored at $10^{-4}$ under the square root only to keep its gradient finite; drift and vol are
multiplied by zero there, so the floor never reaches a value.

### Discrete barrier

Observation dates are `Barrier_Dates` and expiry; $\mathcal B$ the barrier dates; $R$ the
`Cash_Rebate` divided by the deal's size (`Units`, or `Cash_Payoff` for the digital), so the
rebate is absolute cash; $g$ the call/put payoff or, for `EquityBarrierBinaryOption`, the digital.

$$V_{KO}=E\Big[\sum_{j\in\mathcal B}(1-p_j)L_{j-1}R\,D(t,t_j)+L_N\,g(S_N)\,D(t,T)\Big].$$

A digital's terminal step is integrated rather than sampled: the leg pays
$L_{N-1}\Pr(\text{in the money and survives the last step})$, a difference of two $\Phi$, because an
indicator on a drawn spot carries no density to the tape.

Knock-in, by in-out parity, with the knock-in rebate paid at expiry if never knocked in:

$$V_{KI}=E\Big[V_{\text{van}}-D(t,T)\,L_N\big(g(S_N)-R\big)\Big],\qquad
V_{\text{van}}=D(t,T)\,\mathrm{Black}\big(F(t,T),K,\ \sigma(K,T-t)\sqrt{T-t}\big),$$

$F(t,T)=S(t)\,e^{b_N\tau_N}$. The vanilla is priced at the surface's own vol at $(K,T-t)$ - deliberately
not at the strip's total variance - and the already-hit leg below uses the same forward and the
same vol, so the two Europeans cannot disagree. $T-t$ is floored at $10^{-4}$ years inside the
square root. Under LogVar2FJ, $V_{\text{van}}$ is the walk's own European: the inner average of a
conditional Black [@black1976] on the terminal law $(\sum_jM_j,\sqrt{\sum_j\Sigma_j^2})$, so it carries the
inner noise. The inner mean of each row is floored at zero (an individual parity sample can be
negative).

On the outer grid, a scenario whose spot crossed at a strictly earlier observation is resolved:
a knock-out is worth 0 (its rebate settled on the crossing date), a knock-in is worth
$V_{\text{van}}$. A block where every scenario has resolved runs no inner simulation (except a
LogVar2FJ knock-in, whose vanilla is the walk's). Closes
observed before the base date are folded at compile: a crossed knock-in compiles as its vanilla,
a crossed knock-out as its rebate cash flow.

Under the `Relative_Digital_Spread` valuation option $\varepsilon$, the GBM digital is the
strike spread the European digital takes,
$\phi\big(C(K(1-\varepsilon))-C(K(1+\varepsilon))\big)/(2\varepsilon K)$, each leg $C(K')$ the
barrier call struck at $K'$: the two walk the fixings side by side on the same uniforms, each on
the strip read at its own strike, and each integrates its terminal step, $L_{N-1}$ times the
expectation of $(\phi(S_N-K'))^+$ on the survival region of the last interval's law. The strip
telescopes to $\sigma(K',T-t)$, so the spread is the European's where no path reaches the
barrier. $V_{\text{van}}$ and the already-hit leg are the European spread, each leg the Black at
$\sigma(K',T-t)$; on the expiry row each leg is its intrinsic. Under LogVar2FJ the option is not
read.

### Accumulator, TARF, autocall, extendable forward

- **Accumulator.** Fixing $j$ pays $N_1(\phi(S_j-K))^+-N_2(\phi(K-S_j))^+$ at its settlement date
  with weight $L_{j-1}p_j$; the breaching fixing pays nothing.
- **TARF.** The accrual is $a_j=(\phi(S_j-K))^+$ (`InvertedTarget`: $(\phi(1/K-1/S_j))^+$). The
  fired mass pays the remaining target $R_{j-1}=N_1(\text{TargetLevel}-\sum_{k<j}a_k)$, so the
  filling fixing pays exactly the remainder; the surviving branch pays
  $N_1a_j-N_2(\phi(K-S_j))^+\mathbf 1\{\phi(\text{Barrier}-S_j)\ge0\}$ (no knock-in condition
  when `Barrier` is 0).
- **Autocall.** Coupon $j$ fires when the average $A_j$ of its window's fixings reaches
  $K_j=$ threshold$_j\times$`Strike_Price`. Writing $A_j=c+S\,G\,e^{X}$ ($c$ the observed part, $G$
  the window's sampled later blocks, drawn as plain Gaussians), $\{A_j<K_j\}$ is a half-line in the
  prefix return $X$, which alone is truncated. The fired mass pays the coupon (times `FixFXRate`
  under a quanto payoff). The put on a barrier date pays
  $\big(\text{Rebate}-(1-A_j/\text{Strike})\big)\mathbf 1\{\text{breach}\}$ on the surviving set,
  the breach read on the window's last fixing or its average (`Barrier_Observation`). Under GBM each
  leg walks its own strike's strip off the same uniforms (table above); under LogVar2FJ one walk
  carries both. This loop is not antithetic. A row past an observed autocall is worth 0 from the
  next block on (the termination latch).
- **Extendable forward.** At each decision the holder continues iff the spot is beyond $H_j$,
  where the continuation value $C_j(H_j)=0$: closed form $K_2B/A$ for the strip style, backward
  Gauss-Hermite quadrature for the rolling style (`Boundary_Quadrature` 32 nodes and
  `Boundary_Grid_Size` 501 points by default, half-width $\max(6\sqrt{\text{variance}},\,2)$ in
  log-spot). $H_j$ is detached: by $C_j(H_j)=0$ its first-order effect on value vanishes. Only
  decisions are truncated draws; cash flows between them are exact lognormal expectations.
  Transition variances are total-variance differences at the $K_2$ moneyness; a decreasing total
  variance is refused.

### Sampling

Every calculation, at every outer batch size, draws the same way:

| | the canonical inner block `Calculation_State.inner_block` |
|---|---|
| uniforms $u_j$ | inner path $i$ is point $i$ of a scrambled Sobol sequence from point 1024, fixing $j$ its dimension $j$ (64 dimensions per chunk), clamped to $[1074,\,2^{30}-1074]\times2^{-30}$ |
| shared | every trade, outer path, batch and worker reads the same rows, whatever is booked beside it |
| antithetic half [@hammersley1956; @glasserman2003] | $1-u$, taken on the 30-bit integer |

The Sobol stream [@sobol1967] is `torch.quasirandom.SobolEngine` (Joe-Kuo direction numbers [@joe2008], 30 bits,
random linear matrix scramble with a digital shift [@owen1998], its seed the calculation's `Quasi_Seed`). Because
the Sobol rows are shared, a row's inner error is common to all its outer paths and is not averaged away by more
outer paths; a second `Quasi_Seed` is an independent randomisation of the same points, so the spread of a mark
over seeds measures its inner error. A LogVar2FJ pricer's daily shocks come from a generator per walk segment keyed
by `Quasi_Seed` and the segment, so they too are the trade's own.

Inner paths per row, $N=$ `MCMC_Simulations`:

| Calculation | default $N$ | barrier, accumulator, TARF, extendable forward | autocall |
|---|---|---|---|
| `BaseValuation` | 16,384 | $2N$ = 32,768 ($u$ and $1-u$) | $N$ = 16,384 |
| `CreditMonteCarlo`, `HedgeMonteCarlo` | 2,048 | $2N$ = 4,096 per outer path | $N$ = 2,048 |
| `SIMM` (each pricing) | 16,384 | as `BaseValuation` | as `BaseValuation` |

The mirrored half reuses the drawn uniforms; the variance reduction it gives is not measured.

### Estimators: `Branch_And_Weight`

`Branch_And_Weight` is a `BaseValuation` field, default `Yes`; every other calculation runs the
crisp estimator. Both estimate the same expectation; they differ in variance and in the
derivatives they carry.

| Pricer | `Yes` (smooth) | `No` (crisp) |
|---|---|---|
| discrete barrier, accumulator | value bit-identical to `No`; the terminal (barrier) or per-fixing (accumulator) payoff kink gets its second-order term | with sensitivities on, the observed-crossing decision registers a boundary correction; `Greeks: All` is refused |
| TARF | the knock-in leg integrated (`lognormal_fired_gain`); per-fixing kink term; the redemption and knock-in registrations are not made | knock-in a sampled indicator; its derivatives taken from the integral (`splice_conditional_p`: the crisp value, the integral's derivatives) |
| autocall | the put leg integrated over its tail of the interval law | put a sampled indicator, derivatives spliced as for the TARF |
| extendable forward | not read | not read |

The second-order kink term is $\tfrac12K_h(g)\,(g-\bar g)^2$ with $\bar g$ detached: zero in value
and at first order, it adds the Gaussian-kernel estimate of $f_g(0)\,E[\nabla g\,\nabla g^\top\mid g=0]$
to the Hessian, bandwidth $h=1.06\,\mathrm{sd}(g)\,n^{-1/5}$ over the fixing's inner paths
(Silverman's rule [@silverman1986]); a fixing whose accrual has an atom at the strike refuses by name.

### Measured accuracy

Single runs unless seeds are stated; "MCMC" is `MCMC_Simulations`. A signed reading is
engine/reference $-1$; an unsigned one is the absolute relative difference.

| Product and conditions | Calculation, estimator, MCMC | Reading | Reference |
|---|---|---|---|
| Down-and-out call, $S=K=100$, $H=90$, $\sigma=25\%$, $r=q=0$, 1y, 12 monthly observations | CMC inception row, Sobol arm, 8,192 | 8.469040 ($-0.114\%$); at MCMC 256 / 1,024 / 4,096 / 65,536: $+1.34\%$ / $+0.64\%$ / $-0.12\%$ / $-0.05\%$ | independent simulation, 10.5m paths: $8.4787\pm0.0051$ |
| Same, 121 observations (every 3 days) | as above, 8,192 / 32,768 / 131,072 | $-0.185\%$ / $-0.17\%$ / $-0.11\%$ | independent exact-GBM walk, 294m paths: $7.6649\pm0.0008$ |
| Monthly down-and-out, barrier at 1% of spot (never knocks) | BaseValuation, smooth, 32,768 | value $+0.61\%$, gamma $-1.4\%$ | Black |
| Monthly down-and-out, live | BaseValuation, smooth, 32,768 | gamma within 0.57% | CRN bump ladder of its own delta, flat to 2.1% |
| Two-fixing knock-in TARF: spot = strike 1.1, target 0.05, knock-in 1.05, $N_1/N_2$ 1000/2000, $r$ 4%, $q$ 2%, $\sigma$ 10% | BaseValuation, smooth, 262,144 | value 0.11%, delta 0.006%, vega 0.015%, gamma 0.04%, vanna 0.11% | trapezoid quadrature |
| Same TARF, ten seeds | BaseValuation, 16,384 | seed sd 0.3208 crisp, 0.0910 smooth on a value of $-37.36$: variance ratio 12.4 | |
| TARF, twelve fixings | BaseValuation, smooth, 8,192 | gamma seed spread 1.70% monthly, 21.15% daily; value spread 1.13% and 0.35% | |
| Two-coupon autocall, $S=K=100$, coupon 0.08, put barrier 70%, $r$ 4%, $q$ 1%, $\sigma$ 25% | BaseValuation, smooth, 262,144 | value 0.36%, delta 0.16%, vega 0.15%, gamma 0.15% | region-integral quadrature |
| Autocall with two-fixing coupon windows, 70% put | BaseValuation, 1,048,576 | smooth $-0.14\%$ (spot reading) / $-0.17\%$ (average reading); crisp $-0.36\%$ / $-0.47\%$ | direct simulation of the terms, $2^{24}$ paths |
| Two-fixing accumulator | BaseValuation, 65,536 | smooth gamma 0.07% (the crisp pathwise gamma 11.1% short) | quadrature |
| Extendable forward, strip and one-decision rolling, flat market | BaseValuation, 32,768 | within the gate's 0.5%; the written (counterparty-exercised) side 0.1% | Black on the linear continuation |
| LogVar2FJ kit at its GBM limit (levers zero, Gaussian residual, flat $\xi$), autocall | BaseValuation | $1.3\times10^{-16}$ relative | the GBM arm |

Unmeasured: the strip's allocation of forward variance between fixings on a skewed, term-structured
surface against a smile-consistent model; the autocall's per-strike GBM legs (each leg prices in its
own lognormal world, as a booking of the legs as separate deals would) against a single model;
the extendable forward's frozen-$K_2$ smile; the size of the parity break when the $\epsilon$ floor
binds; the variance reduction of the antithetic pairs.

### Declared limitations

- Discrete monitoring only, on the stated dates; nothing is monitored between them.
- Second-order variance of the conditioning step scales like $s_k^{-3}$, $s_k$ the step's
  standard deviation: fine at monthly and quarterly spacing, rough at daily (TARF row above).
- A decision taken on an observed fixing (under either estimator) or on the outer path (the
  crisp estimator) is data to the inner simulation: its derivative comes from the boundary
  correction (`Boundary_AAD_Bandwidth`) where one is registered, and is otherwise absent.
- Under LogVar2FJ the walk re-seeds at each row unless an outer LogVar2FJ process publishes its
  state. `EquityBarrierOption` refuses a Quanto or Compo payoff under the kit and the autocall a
  Compo; the autocall's Quanto correlation enters the walk per step.
