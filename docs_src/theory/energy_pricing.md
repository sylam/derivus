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

The forward-curve models below are the one-factor model of Clewlow and Strickland [@clewlow1999], whose
forward volatility $\sigma e^{-\alpha(T-t)}$ makes the implied spot mean-revert as in Schwartz [@schwartz1997],
and a quadratic carry-curve model of the futures curve's shape.
