## Conventions

For a valuation date $t$, expiry date $T$ and time to expiry $\tau=T-t$, the interest rate and carry are

$$r=-\frac{1}{\tau}\log D(t,T)$$

$$b=\frac{1}{\tau}\log\Big(\frac{F(t,T)}{S(t)}\Big)$$

Asset prices are lognormal with deterministic rates and yields unless a page names a spot model, so
forwards and European options have the closed forms below.

| Quantity | Convention (vanilla equity and FX options) |
|---|---|
| $\tau$ (vol and Black time) | the **Discount_Rate** curve's day count from $t$ to **Expiry_Date** |
| Forward $F(t,T)$ | to expiry plus 2 business days (Monday-Friday; these deals declare no calendar) |
| Discounting | from the settlement date: **Settlement_Date** (equity) or **Delivery_Date** (FX), the expiry where blank |
| Past the expiry | the underlying is the expiry's print: the scenario's own level, or **Price_Fixing** where the expiry is before the base date |

## Forwards

A forward pays $S(T)-K$ at maturity $T$, $K$ the forward price. Its value at $t$ is

$$(F(t,T)-K)D(t,T)$$

## European Options

A European option pays $\max(\delta(S(T)-K),0)$ at expiry, $\delta=+1$ for a call and $-1$ for a put.
With implied volatility $\sigma$ its value is Black's formula [@black1976] (Black-Scholes [@black1973; @merton1973] on the forward; for FX,
Garman-Kohlhagen [@garman1983]):

$$\mathcal B_\delta(F(t,T),K,\sigma\sqrt\tau)D(t,T)$$

$$\mathcal B_\delta(F,K,v)=\delta(F\Phi(\delta d_1)-K\Phi(\delta d_2)),\quad d_1=\frac{\log(F/K)}{v}+\frac{v}{2},\quad d_2=d_1-v$$

for $F>0$, $K>0$, $v>0$; $\Phi$ is the standard normal distribution function. A cash-or-nothing digital
paying 1 is $\Phi(\delta d_2)D(t,T)$ and an asset-or-nothing digital $F\Phi(\delta d_1)D(t,T)$.

Rate options on a volatility factor declaring `Distribution_Type: Normal` use Bachelier's formula [@bachelier1900] with a
normal (absolute) volatility:

$$\mathcal N_\delta(F,K,v)=\delta(F-K)\Phi(\delta d)+v\,\phi(d),\quad d=\frac{F-K}{v}$$

A `Lognormal` factor with a `Shift` $s$ prices $\mathcal B_\delta(F+s,K+s,v)$.

## Quanto and Compo Payoffs

A deal whose **Payoff_Currency** differs from the underlying's **Currency** reads **Payoff_Type** [@haug]; with
the two currencies equal every value prices as `Standard`. Write $\sigma_S$ and $\sigma_X$ for the
at-the-money vols of the underlying and of the FX pair at the deal's expiry, and $\rho$ for the marked
`Correlation` factor (for an equity `Correlation.EquityPrice.<Equity_Volatility>/FxRate.<pair>`, read
with the sign of the deal's currency order).

| Payoff_Type | Underlying priced | Forward | Vol | Paid |
|---|---|---|---|---|
| `Standard` | local $S$ | $F(t,T)$ | $\sigma$ | the local value, converted at spot FX |
| `Quanto` | local $S$ | $F(t,T)e^{-\rho\sigma_S\sigma_X\tau}$ | $\sigma$ | one unit of payoff currency per unit of local value |
| `Compo` | $S\cdot X$ | $F(t,T)F_X(t,T)$ | $\sqrt{\sigma^2+2\rho\sigma\sigma_X+\sigma_X^2}$ | in payoff currency; the local smile is read at $K/F_X(T)$ |

A pair with no `Correlation` factor reads $\rho=0$ under lognormal dynamics. Under `SpotModel:
LogVar2FJ` `EquityBarrierOption` refuses any `Quanto` or `Compo` payoff by name, and the autocall
refuses `Compo` and a `Quanto` with no correlation factor.
