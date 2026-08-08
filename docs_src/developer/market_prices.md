# Market Prices

`Market Prices` is the risk-neutral half of the market data. Where `Price Factors` holds the
observed curves and surfaces a historical calibration produces, this section holds the **quotes**
a risk-neutral model is fitted to, and a *bootstrapper* turns each block into the factor or model
parameters the simulation reads.

This page is the design. Four families are built; one — the interest-rate curve — is declared and
not built, and the specification below is what it is to be built to.

## A quote is an instrument, a quote type and a number {#a-quote}

Every quote in this section is the same three things:

| | |
| --- | --- |
| **the instrument** | a reference to an EXISTING instrument type — the thing the number is a price *for* |
| **`Quote_Type`** | what kind of number is quoted, and therefore what "reprices to the quote" means |
| **`Quoted_Market_Value`** | the number |

The first is the load-bearing one. A quote does not restate an instrument's fields; it **names an
instrument type and carries a block of that type**, so the `Instrument` store's declarations *are*
the quote's schema. A family names the types its quotes may be — `quote_instruments` on the
bootstrapper class — and that list is what a `DealType` dropdown offers and what a gate holds to
the declared deal types. Nothing about a swap has to be described twice for a swap to be
quotable.

`Quote_Type` is per family, not global. The Clewlow–Strickland family takes implied vols only;
Heston-Nandi takes a vol or a premium; an interest-rate quote is a par rate, a rate or a price.
Those are three different questions that happen to share a name, which is right — the JSON is per
family, and only a store keyed by field name across all of them was ambiguous.

## A family maps a quote set to what it calibrates {#a-family}

A bootstrapper class is one price family. It declares:

- `market_factor_type` — the `Market Prices` type string a block is filed under, and the string
  the class selects its own work by. It is declared rather than recovered from the class name
  because the block is named `HestonNandiModelPrices` while its class is
  `HestonNandiModelParameters`.
- `fields` — the block's schema, including the quote table or container as the class reads it.
- `quote_instruments`, where the quotes are instruments rather than a fixed option table.

`mapping['MarketPrices']['types']` is `schema.emit_market_prices(bootstrappers)` over those
declarations, and `construct_bootstrapper` resolves the class by name from the
`Bootstrapper Configuration` section.

The four built families, and what each fits:

| Family | Quotes | Writes |
| --- | --- | --- |
| `GBMAssetPriceTSModelPrices` | a vol surface, ATM column only | `GBMAssetPriceTSModelParameters` — an integrated vol curve |
| `CSForwardPriceModelPrices` | European energy futures options | `CSForwardPriceModelParameters` — sigma, alpha |
| `HestonNandiModelPrices` | European options on any spot | `HestonNandiModelParameters` — omega, alpha, beta, gamma\*, H0 |
| `HullWhite2FactorModelPrices` | forward-starting swaps against a swaption surface | `HullWhite2FactorModelParameters` — two sigma curves, two alphas, a correlation |

## `InterestRatePrices` — the family that is designed and not built {#interestrateprices}

The store has carried the shape of this one for as long as it has existed: a `Points` container
holding `Deal` / `DealType` / `Quote_Type` / `Quoted_Market_Value`, and a `DealType` whose values
are `DepositDeal`, `FRADeal` and `SwapInterestDeal`. That is the design showing through, and it is
the design above with nothing bolted on.

**The block.** `InterestRateCurveParameters` declares it: the `Currency` of the curve to build, an
optional `Discount_Rate` naming the curve the quotes discount on where that differs from the one
being built, a `Spot_Offset` in business days, a `Zero_Rate_Grid` of the tenors the result is
written on, and the quote `Points`.

**The quotes.** Each point carries a `Deal` — a deposit, an FRA or a swap, authored exactly as it
would be in `Trade Data`, because it is the same declaration — plus `Quote_Type` and
`Quoted_Market_Value`, and a `Use` flag so a quote can be held out without being deleted.

**The solve.** Find the zero curve on `Zero_Rate_Grid` that reprices every `Use`d quote instrument
to its quote: a deposit and an FRA to their quoted rate, a swap to par. The instruments are priced
by the engine's own pricers — that is the point of quoting them as instruments — so the residual
is a vector of pricing functions of the curve, and the solve is a root find on it. The curve is
written back as an `InterestRate` price factor.

**What makes the solve cheap.** The residual's Jacobian with respect to the curve knots is exactly
what **calibration Jacobians** buy, and they are on the [roadmap](roadmap.md#designed-not-built)
already, listed there beside sensitivity estimators as first-class objects: with AAD through the
pricers, one backward pass per quote gives the whole row, and a bump-and-reprice loop over
twenty-six knots is not needed at all. That thread and this family should land together — the
same derivative that makes bumping a market *quote* flow through bootstrapping is the one that
makes the bootstrap converge.

**Two loose ends, recorded rather than guessed at.** The curve this writes is an `InterestRate`,
not the `<ClassName>` parameter block the other four write, so `Config.bootstrap`'s "wrote no
`<name>.*` price factor" check wants settling when it is built. And a curve solved from quotes
needs a stated interpolation, which comes from `Price Factor Interpolation` rather than from the
block — see the `Interpolation` note in [Conventions](conventions.md#registries-not-functions).
