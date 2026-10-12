## Hazard Rates

Default is modelled in reduced form: the default time is the first jump of a process with intensity
(hazard rate) $h(t)$ [@jarrow1995; @lando1998; @duffie1999]. The
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
