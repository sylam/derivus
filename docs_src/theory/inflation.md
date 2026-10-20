## Price Index

The Price Index can be monthly or quarterly (**Publication_Period**). It holds the published values
$I(\tau)$ on a discrete set of period start dates $\mathcal T$ (**Index**). If
$\tau_0=$ **Last_Period_Start** is the latest published period, then for $\tau>\tau_0$

$$I(\tau)=I(\tau_0)\,F(\tau-\tau_0),\qquad F(u)=e^{\,r_I(u)\,u},$$

where $r_I$ is the associated **InflationRate** curve (a continuously compounded inflation growth
rate) and $u$ is measured in its day count. No seasonal adjustment is applied.

If $p(\tau)$ is the publication date for period starting $\tau$ (typically $p(\tau)\gt\tau$; the
publication dates run from **Next_Publication_Date** at the publication period) and
$s(t)$ is the greatest $\tau$ for a given date $t$ for which $p(\tau) \le t$, then $p(s(t))\le t$
and $s(p(\tau))=\tau$. The value of the Index at $t$ is $I(t)=I(s(t))$.

## Price Index References

If an inflation contract needs the value of a price index sampled at time $T$, the sampled value is
called a **reference** value $I_R(T)$. The convention is the deal's own **Index_Reference**: a
**Months_Lag** $\ell$ and a **Reference_Type**. With $T^{(1)}$ the first day
of the month of $T$ and $T-\ell m$ the date $\ell$ calendar months before $T$:

- **Interpolated**: the two month-start prints straddling the lagged date, weighted by how far into
  its own month $T$ falls,

$$I_R(T)=I\big((T-\ell m)^{(1)}\big)+w\Big(I\big((T-(\ell-1) m)^{(1)}\big)-I\big((T-\ell m)^{(1)}\big)\Big),
\qquad w=\frac{T-T^{(1)}}{(T^{(1)}+1m)-T^{(1)}}.$$

- **Single** (and **Average**, which is read as **Single**): the print of the month $\ell$ months back,

$$I_R(T)=I\big((T-\ell m)^{(1)}\big).$$

A reference date at or before the last published period reads the published print; one after it
and on or before the base date reads the base-date forward $I(\tau_0)F(\cdot)$ above; a later one
reads the simulated index (or, with no index model, the inflation curve's forward under
`NoModel: RiskNeutral`; see [Definitions](definitions.md)). **Quarters_Lag**,
**Quarter_Reference_Month**, **Reference_Day** and **Days_In_Period** are accepted and not read.

## Inflation Rates

Inflation rate price factors are similar to interest rate price factors [@brigo2006] but have an associated price
index factor (**Price_Index**); their curve is the growth rate $r_I$ above, interpolated as an interest
rate curve ([Asset Pricing](asset_pricing.md#interest-and-inflation-rate-interpolation)). An
`InflationRatePrices` block solves the curve from the index's zero-coupon swap breakevens, a swap
pricing $I_R(T)/I_R(t_0)=(1+b)^T$ whatever it discounts on ([Bootstrapping](../bootstrapping/inflation.md)).
