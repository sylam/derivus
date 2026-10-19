# Calculations

A job's `Calculation` block names one of four calculation types in `Object`, and
`Context.run_job` runs it on the book and the market data the job loads. How each is called and
what it returns is on [API Overview](../api_overview.md), every field each reads and its default on
[Calculations](../json/calculations.md) in the JSON reference, and how a job reaches each and where
they part on [Calc Lifecycle](../developer/calc_lifecycle.md#valuation-modes).

- **`BaseValuation`** ([`BaseValuation`](#basevaluation)) answers what each deal is worth at
  the valuation date: the book at one date on the `Price Factors` it reaches as they stand, none
  simulated, with AAD sensitivities to them under `Greeks`.
- **`CreditMonteCarlo`** ([`CreditMonteCarlo`](#creditmontecarlo)) answers what the book may
  be worth at each future date: it simulates every factor `Price Models` gives a model, correlated
  as `Correlations` declares and the rest held constant, values the book on every path and reduces
  the paths to an exposure profile and, where its blocks ask, the CVA and FVA.
- **`HedgeMonteCarlo`** ([`HedgeMonteCarlo`](#hedgemontecarlo)) runs the same simulation and hands
  the marks to a solver instead of reducing them: it answers how to trade the tradables its
  `Hedging_Problem` block names against a book of liabilities, the solver being on
  [Hedging](../hedging/overview.md).
- **`SIMM`** answers the ISDA SIMM sensitivities of each booked trade as a CRIF [@isdasimm]: it
  moves one market input at a time by ISDA's definition, a benchmark quote's curve ridden on the
  calibration Jacobian its bootstrap from `Market Prices` published and the survival curves solved
  on it re-solved, a CDS quote of a survival curve's block a basis point of spread over and the
  block re-solved, its Risk_CreditQ or Risk_CreditNonQ rows on the credit vertices, a rate vol
  space a point of vol on one vertex's expiry slice, its Risk_IRVol rows on the option's expiry,
  the underlying tenor summed over, a commodity's spot and forward curve a percent together, its
  Risk_Commodity row, and its smile a point per vertex, its Risk_CommodityVol rows, and an
  inflation curve a basis point on the breakevens it is defined by - its block's quotes, re-solved,
  where one solves it, its nodes' own where none does - its Risk_Inflation row. The engine has
  no factor type for ISDA's Risk_InflationVol or Risk_CreditVol, and a mapping naming either is
  refused. The book is compiled once as a `BaseValuation` compiles it and priced again per move
  with the moved factors written into it, each trade read off its own structure, reading the
  regime `Mapping` and the ISDA `Parameters` file the block names. Its tables are on
  [Understanding Output](../output.md#simm).
