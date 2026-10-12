## Inflation Linked Instruments

The payoff of inflation linked cashflows involves the ratio of reference values $I_R(T_2)$ and $I_R(T_1)$
with $T_1<T_2$. The following approximation is used:

$$\Bbb E_t^T\Big(\phi\Big(\frac{I_R(T_2)}{I_R(T_1)}\Big)\Big)\approx \phi\Big(\frac{I_R(T_2)}{I_R(T_1)}\Big) $$

where $\phi$ is an approximately linear function, $T$ is the cashflow payment date and $\Bbb E_t^T$ is
a $T$ forward measure conditional at time $t$. This approximation ignores the convexity correction which
is dependent on the underlying inflation model.

## References

- Jarrow, R.; Yildirim, Y. Pricing Treasury Inflation Protected Securities and Related Derivatives using an HJM Model. *The Journal of Financial and Quantitative Analysis* 38(2), first page 337, 2003. [doi:10.2307/4126754](https://doi.org/10.2307/4126754)
- Mercurio, F. Pricing inflation-indexed derivatives. *Quantitative Finance* 5(3), 289-302, 2005 (the convexity correction this page omits). [doi:10.1080/14697680500148851](https://doi.org/10.1080/14697680500148851)
