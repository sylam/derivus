## Forward Curve

Energy prices do not necessarily follow the same behaviour as other financial assets. Let $F(t,T)$
denote the forward price at time $t$ for settlement at time $T$. A **ForwardPrice** factor holds the
initial curve $F(0,T)$ at discrete settlement dates $T_1<\dots<T_m$ (excel dates). Between them the
curve is linear in $T$, and it is flat beyond the first and the last:

$$F(0,T)=\begin{cases}F(0,T_1), & T\le T_1,\\[2pt]
F(0,T_i)+\dfrac{T-T_i}{T_{i+1}-T_i}\big(F(0,T_{i+1})-F(0,T_i)\big), & T_i\le T\le T_{i+1},\\[2pt]
F(0,T_m), & T\ge T_m.\end{cases}$$

A deal on a commodity with no ForwardPrice factor reads its forward from the spot $S(t)$, the
commodity's carry curve $c$ (read at the settlement date) and its repo curve $r$:

$$F(t,T)=S(t)\exp\big(c(T)\,(T-t)+r(t,T)\,(T-t)\big),\qquad F(t,t)=S(t).$$

The forward-curve models below are the one-factor model of Clewlow and Strickland (1999), whose
forward volatility $\sigma e^{-\alpha(T-t)}$ makes the implied spot mean-revert as in Schwartz (1997),
and a quadratic carry-curve model of the futures curve's shape.

## References

- Clewlow, L. and Strickland, C. (1999). Valuing energy options in a one factor model fitted to
  forward prices. SSRN posted content. [doi:10.2139/ssrn.160608](https://doi.org/10.2139/ssrn.160608)
- Schwartz, E. S. (1997). The stochastic behavior of commodity prices: implications for valuation and
  hedging. *The Journal of Finance*, 52(3), 923-973 (July 1997; Wiley's online copy is dated 2012).
  [doi:10.1111/j.1540-6261.1997.tb02721.x](https://doi.org/10.1111/j.1540-6261.1997.tb02721.x)
