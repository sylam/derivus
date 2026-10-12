## Discount and zero rates

The discount rate $D(t,T)$ is the price at time $t$ of a unit cashflow paid at $T (T \ge t)$. The zero
rate $r(t,T)$ at time $t$ for maturity $T$ is the continuously compounded interest rate between $t$ and
$T$ where:

$$D(t,T)=\exp\big(-r(t,T)\,\tau(t,T)\big)$$

and $\tau(t,T)$ is the year fraction from $t$ to $T$ in the curve's own **Day_Count** (ACT_365 by
default). The **InterestRate** factor stores $r(0,\cdot)$ at its tenors; between them the curve is
interpolated as **Price Factor Interpolation** declares (Linear by default; Hermite, HermiteRT,
LinearRT: [Asset Pricing](asset_pricing.md#interest-and-inflation-rate-interpolation)).

Let $r_\tau (t)$ denote the $\tau$-tenor zero rate $r(t,t+\tau)$

## Spread Rates

A *spread rate* is an interest rate price factor that is defined relative to a *parent rate*. A
*base rate* is an interest rate price factor without a parent. The parent of a spread rate can be
another spread rate or a base rate.

A spread rate's ID has the form $R.S$, where $S$ is the spread name and $R$ is the ID of the parent rate.
The discount rate of $R.S$ is given by

$$D_{R.S}(t,T)=D_R(t,T)D_S(t,T)$$

where $D_R$ is the discount rate of the parent and $D_S$ is the spread: the zero rates add. Examples of
spreads are ZAR-SWAP.ZAR-USD-BASIS (ZAR-SWAP is the parent and ZAR-USD-BASIS is the spread).

The short-rate models below are the extended Vasicek model of Hull and White [@hull1990] and its
two-factor Gaussian extension ([@hull1994]; the G2++ form of Brigo and Mercurio [@brigo2006]);
the two-factor model's analytic swaption is the affine-model approximation of Schrager and Pelsser
[@schrager2006]. The PCA model drives the zero curve by the principal components of its historical moves
[@litterman1991].
