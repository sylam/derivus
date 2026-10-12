## Containers

Container deals are deals whose primary role is to *aggregate* the results of dependent
sub-deals rather than to price a single financial contract on their own. They form the tree
structure of the portfolio: every leaf is a real instrument (forward, option, swap, ...) and
every interior node is a container that gathers its children's MtMs and applies any
container-specific post-processing (netting, collateral haircuts, structured-product payoff
rules, etc.).

Two container types are used:

- **NettingCollateralSet** — represents an ISDA / CSA boundary. All children are netted into a
  single exposure; if the agreement is collateralised, posted collateral is also accounted for
  here. This is the typical top-level wrapper for a counterparty's portfolio.
- **StructuredDeal** — aggregates multiple instrument legs into a structured product. Used
  when the deal as a whole has payoff rules that combine the underlying legs in non-trivial ways
  (e.g. accumulation/decumulation, conditional cash flows).

In the JSON `Deals` section, containers and instruments are interchangeable inside `Children`
arrays — the difference is whether the entry has its own `Children` (a container) or not (a
leaf instrument). Nesting can go arbitrarily deep, although in practice CSA → leaves is the
common shape.

## References

- Gregory, J. *The xVA Challenge*. Wiley, 2015, ISBN 978-1-119-10941-9 (collateral and close-out). [doi:10.1002/9781119109440](https://doi.org/10.1002/9781119109440)
- Andersen, L.; Pykhtin, M.; Sokol, A. Rethinking the margin period of risk. *The Journal of Credit Risk* 13(1), 2017. [doi:10.21314/jcr.2016.218](https://doi.org/10.21314/jcr.2016.218)
