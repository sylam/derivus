## Price Factors

A financial instrument derives its value from several inherently random market variables (e.g. Equity,
commodity, FX, interest rates, etc.). These variables (referred to as *price factors*) will either have
a single value at time $t$ (i.e. $P(t)$,  e.g. Spot Equity or FX rates), or multiple values $P(t;p)$
parametrized by $p$ which represents term, moneyness etc. (e.g. interest rate curves and volatility
 surfaces).

Interest rates have values $D(t,T)$ representing discount factors parametrized by maturity $T$. 
Equity/FX volatility price factors have values $v(t,T,S,K)$ representing implied volatilities for expiry
$T$, spot price $S$ and strike $K$. Valuing a deal at time $t$ involves requesting the set of
corresponding price factor values $\{P(s;p)\}$ for times $s$, where $s \le t$ (in the general case),
and parameters $p$. Note that if there is no path dependency then $s=t$.

## Factor models

A *factor model* calculates future values of a price factor under Monte Carlo simulation. The factor
model attached to a price factor $P$ has a vector of random processes $R_1(t),R_2(t),...,R_l(t)$, which
generates the underlying ($l$-factor) source of randomness. The scenario generator draws independent
standard normals (pseudo-random, seeded from `Random_Seed`), correlates them with the
Cholesky factor of the configured correlation matrix and hands each model its own rows; under
`Antithetic: Yes` the second half of a batch is the first half negated. Some models also draw
uniforms of their own from a fixed scrambled Sobol stream, not seeded from `Random_Seed`:
`MarkovHMMSpotModel`, `GARCHSpotModel`, `BasisLinkedSpotModel` and the LogVar2FJ residual's
mixers.

Under a particular factor model, the corresponding price factor value $P(t;p)$ at time $t$ may depend
on other price factor values, $Q(t;q)$. Typically, an FX rate at time $t$ would depend on the foreign
and domestic interest rates at time $t$.

## Simulation of price factors with no model

At $t=0$, all price factors are read directly from their market data. A factor with no model
attached is read at a later row $t$ as the calculation's `NoModel` field declares
(`CreditMonteCarlo`, `HedgeMonteCarlo`):

| `NoModel` | Static curve read as a discount factor at row $t$ for maturity $t+\tau$ | Spot, FX rate, price index |
|---|---|---|
| `Constant` (default) | today's curve unchanged: $D(t,t+\tau)=D(0,\tau)$ | today's value |
| `RiskNeutral` | today's curve rolled forward to $t$: $D(t,t+\tau)=D(0,t+\tau)/D(0,t)$ | today's value |

`RiskNeutral` is the forward of the base-date curve, so a deterministic curve realises the forwards
the base date implied (Harrison and Pliska, 1981, for pricing under the risk-neutral measure). A
`YieldInflationCashflowListDeal` whose price index has no model is refused under `Constant` on any
grid longer than the base date, and priced under `RiskNeutral` off the inflation curve's forward.

## References

- Harrison, J. M. and Pliska, S. R. (1981). Martingales and stochastic integrals in the theory of
  continuous trading. *Stochastic Processes and their Applications*, 11(3), 215-260.
  [doi:10.1016/0304-4149(81)90026-0](https://doi.org/10.1016/0304-4149%2881%2990026-0)

---
