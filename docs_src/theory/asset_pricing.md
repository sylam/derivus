## Spot

The spot price of an asset $S(t)$ (assuming immediate delivery) is expressed in the *asset currency*.
The asset currency is specified by the corresponding property:

- **Currency** for equity and commodity prices
- **Domestic Currency** for FX rates

The initial price $S(0)$ is given by its **Spot** property.

## FX Rates

There is just one FX rate price factor for each currency pair (including the base currency). FX rate
price factors always have a currency $C$ and a parent currency $p(C)$ which, currently, will always be
the base currency. If $f(C)$ is the *foreign* currency, and $d(C)$ is the domestic currency, then let
$X_{C/A}(t)$ be the spot price of currency $C$ in currency $A$ at time $t$. The asset factor models
evolve the FX rate $S(t)=X_{f(C)/d(C)}(t)$, effectively making the domestic currency of all FX rates
the same as the base currency

## Forward rates

For equity and commodity prices, the forward price at time $t$ for delivery at $T$ is the usual
no-arbitrage formula:

$$F(t,T)=S(t)\frac{Q(t,T)}{D(t,T)}$$

Here $D(t,T)$ is the discount rate from the asset's interest rate price factor (i.e. its repo rate
specified by the **Interest Rate** price factor) and $Q(t,T)$ is the discount rate from the dividend
rate (for equities) or convenience yield (for commodities): the continuous-yield forward of
Merton [@merton1973].

### FX Rates

Forward FX rates for currency $C$ in currency $A$ is given by:

$$F_{C/A}(t,T)=X_{C/A}\frac{D_C(t,T)}{D_A(t,T)}$$

where $D_C(t;T)$ is the discount rate from the interest rate price factor specified by the
**Interest Rate** property on the FX rate price factor for currency $C$ (covered interest parity, the
forward of Garman and Kohlhagen [@garman1983]). This can be extended to handle
the case where a given equity/commodity price $S(t)$ with asset currency $C$ is required in another
currency $A$ as follows

$$F_{S/A}(t,T)=S(t)X_{C/A}\frac{Q(t,T)}{D_A(t,T)}$$

## Stochastic volatility

The spot model the path-dependent equity and FX options can opt into, LogVar2FJ, has a page of its
own: [Stochastic Volatility](stochastic_volatility.md) states the model, its Black-Scholes limit,
its relation to Bergomi's forward-variance models with their references, and how one fit serves
an equity index and an FX pair alike.

## Dividend rate interpolation

A initial dividend rate curve $q(t)$ can be derived from discrete dividends using the no-arbitrage
relationship between spot and forward prices. The spot price is the present-value forward price plus
the dividends the forward purchaser does not get (but the spot purchaser does):

$$S(0)=F(0,t)D(0,t)+\sum_{0 \lt s_i \le t} H_i D(0,t_i),$$

where $H_i$ is the projected/known dividend paid at time $t_i$ with ex-dividend date $s_i$
(with $s_i \le t_i$). With $Q(0,t)=e^{-q(t)t}$,
the implied dividend rate is:

$$q(t)=\frac{1}{t} \log \Biggl( \frac{S(0)}{S(0)-\sum_{0 \lt s_i \le t} H_i D(0,t_i)} \Biggl)$$

Since the curve $q(t)t$ is constant on each interval $[s_i,s_{i+1}]$, $q(t)$ is a piecewise linear
function of $1/t$, and a **DividendRate** curve is interpolated linearly in $1/t$ with flat
extrapolation. If $t_1$ and $t_2$ are points on the curve with $t_1 \le t \le t_2$, then

$$q(t)=q(t_1)+\Biggl(\frac{1/t_1 - 1/t}{1/t_1 - 1/t_2}\Biggl)(q(t_2)-q(t_1))$$

## Interest and Inflation rate interpolation

**InterestRate**, **InflationRate** and **ForwardRate** factors take their method from the
**Price Factor Interpolation** section (default Linear); dividend, survival and forward-price curves
interpolate as their own sections state. Accepted names:

| Name | Method |
|---|---|
| `Linear`, `LinearInterFlatExtrapCurveGetValue` | Linear |
| `LinearRT`, `LogLinearCurveInterpolation` | LinearRT (log-linear in the discount factor) |
| `Hermite`, `HermiteInterpolationCurveGetValue`, `CubicSplineCurveInterpolation`, `NaturalCubicSplineCurveInterLinearExtrap` | Hermite |
| `HermiteRT`, `HermiteRTInterpolationCurveGetValue`, `CubicSplineOnXTimesYCurveInterpolation`, `LogNaturalCubicSplineCurveInterFlatExtrap` | HermiteRT |
| `LinearExtrapolate` | linear, extrapolated along the end segments |
| any other name | Linear |

The cubic-spline names are read as the Hermite scheme below, not as a cubic spline; the difference
from a natural cubic spline is unmeasured.

### Hermite Interpolation

The Hermite method addresses the drawbacks of cubic splines while maintaining smoothness. It begins 
by fixing the slope at each point using a second-degree polynomial between the target point and its 
two neighbours. Once the slope is established, a third-degree polynomial is adjusted to ensure 
continuity up to the second degree. This approach allows changes in one section of the curve to affect 
only its immediate neighbours. However, when using default extrapolation at the long end, a slight 
drop in forward rates may occur before they become flat.

The formulas for the Hermite method are provided below. With knots $(t_i,r_i)$, $i=1,\dots,n$, and
slopes $r'=(r'_1,\dots,r'_n)$, for $t_i\le t\le t_{i+1}$:

$$r(t)=r_i + m_i(t)\big( r_{i+1} - r_i\big) + m_i(t)\big(1-m_i(t)\big)g_i+m_i^2(t)\big(1-m_i(t)\big)c_i$$

where
$$m_i(t) = \frac{t-t_i}{t_{i+1}-t_i} $$

$$g_i = (t_{i+1}-t_i)r'_i - (r_{i+1}-r_i)$$

$$c_i = 2(r_{i+1}-r_i) -(t_{i+1}-t_i)(r'_i+r'_{i+1})$$

The vector $r'$ is the slope at $t_i$ of the parabola through the three neighbouring knots (the
Bessel slope of Hagan and West [@hagan2006]):

$$ r'_i = \frac{1}{t_{i+1} - t_{i-1}} \left[ \frac{(r_i - r_{i-1}) (t_{i+1} - t_i)}{(t_i - t_{i-1})} + 
\frac{(r_{i+1} - r_i) (t_i - t_{i-1})}{t_{i+1} - t_i} \right],\qquad 1<i<n$$

Boundary conditions:

$$ r'_1 = \frac{1}{t_3 - t_1} \left[ \frac{(r_2 - r_1) (t_3 + t_2 - 2t_1)}{(t_2 - t_1)} - 
\frac{(r_3 - r_2) (t_2 - t_1)}{t_3 - t_2} \right]$$

$$ r'_n = \frac{-1}{t_n-t_{n-2}} \left[ \frac{(r_{n-1} - r_{n-2})(t_n - t_{n-1})}{(t_{n-1} - t_{n-2})} - 
\frac{(r_n - r_{n-1})(2t_n-t_{n-1}-t_{n-2})}{t_n - t_{n-1}} \right]$$

### Linear Interpolation

Linear interpolation involves drawing a straight line between two points surrounding the desired rate.
However, linear interpolation is not continuous in its differentials, which means that implied forward
rates can be discontinuous when the curve is interpolated between spot rates.

$$r(t) = \alpha r_{i + 1} + (1-\alpha)r_i$$

where $t_i < t < t_{i+1}$ and

$$ \alpha = \frac{t - t_i}{t_{i + 1} - t_i} $$

The rate is held flat before the first tenor and beyond the last.

### RT interpolation

Both Linear and Hermite interpolation can be applied to the rate multiplied by the tenor instead of just the rate. The variations are HermiteRT and LinearRT and they are exactly as described above except instead of interpolating just the rate, we interpolate the rate x tenor (LinearRT is therefore log-linear in the discount factor)

i.e. instead of interpolating:

$$r = (r_1,, r_i,, r_n)$$

we interpolate:

$$r = (r_1t_1,, r_it_i,, r_nt_n)$$

## Volatility rate interpolation

Equity, FX and commodity surfaces (**Surface_Type**) are static: a simulation reads the base-date
surface at the tenor measured from each row. Between quoted expiries and beyond them:

| Surface_Type | Moneyness coordinate | Across expiry | Beyond the quotes |
|---|---|---|---|
| `Explicit` | ratio $S/K$ or $F/K$ as the deal declares ($K/S$, $K/F$ inverted); ATM $=1$ | linear in vol | flat in moneyness and expiry |
| `Relative_Forward` | $(K-F)/F$; ATM $=0$ | linear in vol | flat |
| `Malz` | $\ln(F/K)$ (or $\ln(S/K)$ as the deal declares); ATM $=0$ | linear in total variance $\sigma^2T$ | flat in moneyness; before the first expiry its vol; after the last its total variance $\sigma_n^2T_n$, so the vol falls as $\sigma_n\sqrt{T_n/T}$ and the forward variance there is zero |
| `SVI` | $x$ of the sticky rule below | linear in variance $\sigma^2$ | expiry flat |
| `Skew` | $x$ of the sticky rule below | linear in vol, between the two expiries' smiles | expiry flat; wings as tabled |

Within one expiry the grid surfaces (`Explicit`, `Relative_Forward`, `Malz`) are linear in
moneyness.

### Sticky rule

**Moneyness_Rule** is read by Skew and SVI surfaces only, which are functions of a log-moneyness $x$:

 - **Sticky_Strike**: $x=\ln(K/\mathrm{ATM}_{\mathrm{ref}})$ (**ATM_Ref**), the reference interpolated linearly in expiry;
 - **Sticky_Moneyness** (default) and **Sticky_Delta**, read alike: $x=\ln(K/F)$.

$K$ is the option strike price and $F$ is the underlying forward price. An `Explicit`, `Malz` or
`Relative_Forward` surface does not read the rule.

### Shifts and normal vols

An interest rate vol surface (caps, floors, swaptions) declares **Distribution_Type**:

- `Lognormal` (default) prices Black [@black1976]. A **Shift** $s$ (default 0) is ignored during
  volatility interpolation and added to both forward and strike: instead of
  $\mathcal B_\delta(F,K,\sigma\sqrt{t_0-t})$ we calculate $\mathcal B_\delta(F+s,K+s,\sigma\sqrt{t_0-t})$.
- `Normal` prices Bachelier [@bachelier1900] on an absolute normal vol: the undiscounted premium is
  $\mu\,\Phi(\mu/v)+v\,\phi(\mu/v)$ with $\mu=\omega(F-K)$, $v=\sigma_N\sqrt{\tau}$; a nonzero shift
  beside it is refused.


### Skew Surfaces

Skews are built up by nine parameters. These parameters are:

 - ATM Volatility $(v_0>0)$
 - Underlying ATM reference quote (**ATM_Ref**, read under Sticky_Strike)
 - the slope $s$
 - the Left curvature $L$
 - the Right curvature $R$
 - the Left cuttoff $C, C<0$
 - the Right cuttoff $D, 0<D$
 - the left wing $\lambda>0$
 - the right wing $\rho>0$

These parameters are used to calculate the volatility $v(x)$ in terms of log-moneyness $x$ as

| Region      | Limits on strike     | Volatility Value    |
| -----------:|:--------------------:| -------------------:|
| I           | $x \lt (1+\lambda)C$ | $\alpha+\beta(1+\lambda)C+\gamma(1+\lambda)^2C^2$ |
| II          | $(1+\lambda)C \lt x \lt C$  | $\alpha+\beta x + \gamma x^2$ |
| III         | $C \lt x \lt 0$    | $v_0+sx+Lx^2$ |
| IV          | $0 \lt x \lt D$    | $v_0+sx+Rx^2$ |
| V           | $D \lt x \lt (1+\rho) D$ | $\alpha'+\beta' x + \gamma' x^2$ |
| VI          | $(1+\rho)D \lt x$         | $\alpha'+\beta'(1+\rho)D+\gamma'(1+\rho)^2D^2$ |

The constants $\alpha, \beta, \gamma$ are derived by solving the following equations:

$$v_0+sC+LC^2=\alpha+\beta C+\gamma C^2$$

$$ s+2LC=\beta + 2\gamma C $$

$$0=\beta + 2\gamma(1+\lambda)C$$

The right side of the curve with coefficients $\alpha', \beta', \gamma'$ are solved analogously but using $D, \rho$ instead of $C, \lambda$.
A wing of zero ($\lambda=0$ or $\rho=0$) holds the smile flat beyond its cutoff.

### SVI Surfaces

The SVI volatility skew is defined by the five parameters $a, b, \rho, m, \sigma$ per expiry; it
gives the variance (not the total variance $\sigma^2T$ of the raw SVI of Gatheral and Jacquier [@gatheral2014]):

$$ \sigma_{SVI}^2(x)=a+b\big(\rho(x-m)+\sqrt{(x-m)^2+\sigma^2}\big)$$

No arbitrage constraint is checked, in particular neither

$$0 \leq b \leq \frac{2}{(1+|\rho|)T}$$

(the wing-slope bound of Lee's moment formula [@lee2004], written for this variance) nor

$$|\rho| \leq 1$$

### Explicit Surfaces

These are moneyness surfaces: the moneyness is spot over strike, or forward over strike, as the
pricing deal declares (strike over spot or forward where it declares the inverse). The moneyness
axis must be strictly positive; volatility is linear in moneyness and flat beyond the grid.

### Relative_Forward Surfaces

A moneyness surface on the relative forward moneyness $(K-F)/F$ (ATM at 0), otherwise read as an
`Explicit` one.

### Malz (Delta-Quoted) Surfaces

FX volatility surfaces are commonly quoted in terms of the option *delta* rather than strike or
log-moneyness [@malz1997]. Each vol quote (**Delta_Surface**) is labelled with a forward delta,
and the surface's **Premium_Adjusted** field names which: `Yes`, the
**percentage-of-foreign-forward delta**, the premium-adjusted forward delta [@reiswich2010] of a
pair whose premium is paid in the foreign currency; `No`, the plain forward delta of an option
whose premium is paid in the pricing currency. `FXVol` defaults to `Yes`, `EquityPriceVol` and
`CommodityPriceVol` to `No`. Derivus converts this delta-quoted surface into a log-moneyness
surface on initialisation; it is then read linearly in log-moneyness, as an `Explicit` one is, but
across expiry in total variance.

#### Delta Convention

Define $x = \ln(F/K)$ as the log-moneyness (positive for ITM calls, negative for OTM calls).
Under the lognormal model, the percentage-of-foreign-forward delta for a **call** is

$$\Delta_{\text{call}}(x,\sigma,T) = \frac{K}{F}\,\mathcal{N}(d_2), \qquad
  d_2 = \frac{x + \tfrac{1}{2}\sigma^2 T}{\sigma\sqrt{T}} - \sigma\sqrt{T}
       = \frac{x - \tfrac{1}{2}\sigma^2 T}{\sigma\sqrt{T}}$$

and for a **put**

$$\Delta_{\text{put}}(x,\sigma,T) = -\frac{K}{F}\,\mathcal{N}(-d_2)$$

Note the $K/F = e^{-x}$ prefactor distinguishes percentage-of-foreign from the more common
spot delta. The plain forward delta drops it and reads $d_1$:

$$\Delta_{\text{call}}(x,\sigma,T) = \mathcal{N}(d_1), \qquad
  \Delta_{\text{put}}(x,\sigma,T) = -\mathcal{N}(-d_1), \qquad
  d_1 = \frac{x + \tfrac{1}{2}\sigma^2 T}{\sigma\sqrt{T}}$$

#### ATM Correction

Vendor surfaces are typically delivered with the ATM node labelled as $\Delta = \pm 0.5$, the
delta-neutral straddle, where $\Delta_{\text{call}} = -\Delta_{\text{put}}$. Premium-adjusted,
that strike is $K = F e^{-\sigma_{\text{ATM}}^2 T/2}$ and its delta

$$\delta_{\text{ATM}} = \tfrac{1}{2}\exp\!\left(-\tfrac{1}{2}\sigma_{\text{ATM}}^2 T\right)$$

rather than exactly $0.5$, so the implementation replaces the vendor $\pm 0.5$ nodes with
the corrected $\pm\delta_{\text{ATM}}$ before any interpolation is performed. Plain, the strike is
$K = F e^{\sigma_{\text{ATM}}^2 T/2}$ at a true $\delta_{\text{ATM}} = 0.5$. ($\sigma_{\text{ATM}}$
is the $+0.5$ quote where both are present; a smile quoted on one side only gets the ATM node on
the other.) Quoted vols are floored at $10^{-4}$.

#### Self-Consistency Inversion

For a given log-moneyness $x$ and expiry $T$, the implied volatility $\sigma^*(x,T)$ must satisfy
the fixed-point condition

$$\Delta\bigl(x,\,\sigma^*(x,T),\,T\bigr) = \delta^*,\qquad
  \sigma^*(\delta^*, T) = \sigma_{\text{interp}}(\delta^*)$$

where $\sigma_{\text{interp}}$ is the piecewise-linear interpolant of the vendor delta-vol quotes
(linear in delta, not Malz's quadratic). This is a scalar nonlinear equation in $\delta^*$ which is
solved by bisection (64 halvings), separately for calls ($x \le x_{\text{ATM}}$) and puts
($x > x_{\text{ATM}}$), bracketed by the available wing deltas. Where the root lies outside the
bracket, $\delta^*$ is the bracket end with the smaller residual, so the smile is flat beyond its
widest quoted delta.

The ATM dividing point in log-moneyness is the straddle's own,

$$x_{\text{ATM}} = \pm\tfrac{1}{2}\sigma_{\text{ATM}}^2 T$$

$+$ premium-adjusted and $-$ plain.

#### Total-Variance Grid and Adaptive Refinement

To avoid repeated root-finding at runtime, the inversion is performed once at initialisation time
on a grid of log-moneyness nodes and stored as **total variance**

$$w(x,T) = \sigma^*(x,T)^2\,T$$

The initial grid $x \in \{-0.5,\,-0.25,\,-0.1,\,0.0,\,0.1,\,0.25,\,0.5\}$ is refined adaptively
as follows. For each consecutive pair $(x_l, x_r)$ the midpoint $x_m = (x_l+x_r)/2$ is tested:

1. Compute the exact implied vol $\sigma^*(x_m, T)$ via the inversion above.
2. Interpolate the stored total-variance grid linearly in $x$ at $x_m$ and convert back to vol:
   $\hat\sigma = \sqrt{w_{\text{interp}}(x_m)/T}$.
3. If $|\hat\sigma - \sigma^*(x_m,T)| > \varepsilon$ (default $\varepsilon = 10^{-4}$, i.e. 1 bp),
   insert $x_m$ into the grid with the exact value $w(x_m,T) = \sigma^{*2}(x_m,T)\,T$.
4. Repeat until all midpoint errors are within tolerance.

The resulting $(x, T, \sigma)$ triples are the surface pricing reads, each expiry on its own
refined grid.

## Spot models

The spot models documented below are geometric Brownian motion [@black1973], with a
term structure of implied vol in the implied variant; a log-spot Ornstein-Uhlenbeck process
[@uhlenbeck1930], Schwartz's one-factor commodity model [@schwartz1997]; a hidden-Markov
regime model [@hamilton1989]; a GARCH(1,1) model [@bollerslev1986] with Student-t innovations
[@bollerslev1987]; basis and fixing models built on them; and LogVar2FJ
([Stochastic Volatility](stochastic_volatility.md)).
