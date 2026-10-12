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

The short-rate models below are the extended Vasicek model of Hull and White (1990) and its
two-factor Gaussian extension (Hull and White, 1994; the G2++ form of Brigo and Mercurio, 2006);
the two-factor model's analytic swaption is the affine-model approximation of Schrager and Pelsser
(2006). The PCA model drives the zero curve by the principal components of its historical moves
(Litterman and Scheinkman, 1991).

## References

- Brigo, D. and Mercurio, F. (2006). *Interest Rate Models - Theory and Practice*, 2nd edition.
  Springer Finance (the G2++ two-factor Gaussian model). ISBN 9783540221494 (the DOI record lists no
  authors). [doi:10.1007/978-3-540-34604-3](https://doi.org/10.1007/978-3-540-34604-3)
- Hull, J. and White, A. (1990). Pricing interest-rate-derivative securities. *The Review of
  Financial Studies*, 3(4), 573-592. [doi:10.1093/rfs/3.4.573](https://doi.org/10.1093/rfs/3.4.573)
- Hull, J. C. and White, A. D. (1994). Numerical procedures for implementing term structure models
  II. *The Journal of Derivatives*, 2(2), 37-48 (the two-factor model).
  [doi:10.3905/jod.1994.407908](https://doi.org/10.3905/jod.1994.407908)
- Litterman, R. B. and Scheinkman, J. (1991). Common factors affecting bond returns. *The Journal of
  Fixed Income*, 1(1), 54-61. [doi:10.3905/jfi.1991.692347](https://doi.org/10.3905/jfi.1991.692347)
- Schrager, D. F. and Pelsser, A. A. J. (2006). Pricing swaptions and coupon bond options in affine
  term structure models. *Mathematical Finance*, 16(4), 673-694 (online September 2006, the issue
  October 2006). [doi:10.1111/j.1467-9965.2006.00289.x](https://doi.org/10.1111/j.1467-9965.2006.00289.x)
