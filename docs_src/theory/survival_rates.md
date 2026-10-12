## Hazard Rates

Default is modelled in reduced form: the default time is the first jump of a process with intensity
(hazard rate) $h(t)$ (Jarrow and Turnbull, 1995; Lando, 1998; Duffie and Singleton, 1999). The
relationship between the hazard rate $h(t)$, survival probability $S(t,T)$ and the forward hazard
rate $h(t,T)$ is

$$\exp\Big(-\int_t^T h(t,u)\,du\Big)=S(t,T)=\mathbb{E}_t\Big[\exp\Big(-\int_t^T h(s)\,ds\Big)\Big]$$

where $\mathbb{E}_t$ is the risk-neutral expectation conditional on information at time $t$ and
$h(t)=h(t,t)$.

Initial survival probabilities are represented as a log survival probability curve $I(t)$ defined as

$$I(t)=\int_0^t h(0,s)\,ds=-\log S(0,t)$$

where $I(0)=0$ and $I(u)\ge I(t)$ for $u \ge t$. The **SurvivalProb** factor's **Curve** holds
$I$ at its tenors (ACT_365 year fractions). $I$ is linear between tenors (a piecewise-constant hazard),
held at its first value before the first tenor (so $I(0)=0$ needs a knot at tenor 0) and extended
beyond the last tenor at the last segment's slope (the last hazard continues). **Recovery_Rate** (default 0.4) is the recovery
assumed on default.

## References

- Duffie, D. and Singleton, K. J. (1999). Modeling term structures of defaultable bonds. *The Review
  of Financial Studies*, 12(4), 687-720. [doi:10.1093/rfs/12.4.687](https://doi.org/10.1093/rfs/12.4.687)
- Hull, J. and White, A. (1990). Pricing interest-rate-derivative securities. *The Review of
  Financial Studies*, 3(4), 573-592; the hazard-rate model below is its one-factor model applied to
  $h$. [doi:10.1093/rfs/3.4.573](https://doi.org/10.1093/rfs/3.4.573)
- Jarrow, R. A. and Turnbull, S. M. (1995). Pricing derivatives on financial securities subject to
  credit risk. *The Journal of Finance*, 50(1), 53-85 (March 1995; Wiley's online copy is dated
  2012). [doi:10.1111/j.1540-6261.1995.tb05167.x](https://doi.org/10.1111/j.1540-6261.1995.tb05167.x)
- Lando, D. (1998). On Cox processes and credit risky securities. *Review of Derivatives Research*,
  2(2-3), 99-120. [doi:10.1007/BF01531332](https://doi.org/10.1007/BF01531332)
