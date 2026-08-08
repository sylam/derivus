# Market Prices

`Market Prices` is the risk-neutral half of the market data. Where `Price Factors` holds the
observed curves and surfaces a historical calibration produces, this section holds the **quotes**
a risk-neutral model is fitted to, and a *bootstrapper* turns each block into the factor or model
parameters the simulation reads.

This page is the design. All five families are built; the interest-rate curve was the last, and the
section below is what it was built to.

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

The five families, and what each fits:

| Family | Quotes | Writes |
| --- | --- | --- |
| `GBMAssetPriceTSModelPrices` | a vol surface, ATM column only | `GBMAssetPriceTSModelParameters` — an integrated vol curve |
| `CSForwardPriceModelPrices` | European energy futures options | `CSForwardPriceModelParameters` — sigma, alpha |
| `HestonNandiModelPrices` | European options on any spot | `HestonNandiModelParameters` — omega, alpha, beta, gamma\*, H0 |
| `HullWhite2FactorModelPrices` | forward-starting swaps against a swaption surface | `HullWhite2FactorModelParameters` — two sigma curves, two alphas, a correlation |
| `InterestRatePrices` | deposits, FRAs and swaps | an `InterestRate` zero curve |

## `InterestRatePrices` — a curve solved from its quotes {#interestrateprices}

The store carried the shape of this one for as long as it existed: a `Points` container holding
`Deal` / `DealType` / `Quote_Type` / `Quoted_Market_Value`, and a `DealType` naming instrument
types. That was the design showing through, and what got built is that design with nothing bolted
on.

**The block.** `InterestRateCurveParameters` declares it: the `Currency` of the curve to build, the
`Day_Count` its tenors are expressed in, an optional `Discount_Rate` naming the curve the quotes
discount on, the solver's three tuning knobs (`N_Iter`, `Tol`, `Damping_Halvings` — the same
vocabulary the calibration classes tune with, and each read with its declared default as the
engine's fallback), and the quote `Points`. A blank `Discount_Rate` builds a **self-discounting**
curve, which is the single-curve configuration.

**The quotes.** Each point carries a `Deal` — a deposit, an FRA, a swap, or a `StructuredDeal` over
two legs, authored exactly as it would be in `Trade Data`, because it is the same declaration —
plus `DealType`, `Quote_Type` and `Quoted_Market_Value`, and a `Use` flag so a quote can be held out
without being deleted. `DealType` supplies the block's `Object`, and the family stamps
`Discount_Rate`: what an instrument *projects* off is its own business and it names that curve
itself, what the quote set *discounts* on belongs to the curve set and is stated once.

`Quoted_Market_Value` is a rate in percent, and where it lands is a property of the instrument
TYPE — a `FRA_Rate`, a `Swap_Rate`, a pinned `Interest_Rate_Schedule`, a fixed leg's `Rate` column.
That correspondence is the one thing the family knows about a type beyond the type's own
declarations, and it is a registry (`QUOTE_WRITERS`) rather than a branch, so a new quotable
instrument is a row. `Quote_Type` declares the single convention that is built, `Par_Rate`: the
solve holds every benchmark at PV zero.

**The knot rule.** ONE knot per used quote, at that benchmark's last cashflow date, in the block's
`Day_Count`. That is the only placement that makes the system square, and squareness is the shape
of a bootstrap: a knot with no instrument maturing at it is unidentified, and two instruments
maturing between the same pair of knots leave the curve under-determined between them. Below the
shortest knot the curve is flat, which `CurveTenor` gives by clipping, so the front stub costs no
extra unknown. **The output grid is that grid** — writing the result onto a second, wider grid
would interpolate it, and an interpolated curve no longer reprices its quotes. There is therefore
no `Zero_Rate_Grid` field; holding a quote out with `Use` drops its knot with it.

**The solve.** Find the curve that reprices every used quote to par. The instruments are priced by
the engine's own pricers — that is the point of quoting them as instruments — so the residual is a
vector of pricing functions of the curve, and the solve is a damped Newton root find on it in
float64. Two blocks make a multi-curve set, and `Discount_Rate` orders them: a projection curve
solved before the discount curve it prices against would be solved against a curve that does not
exist.

**What makes the solve cheap.** The residual's Jacobian in the curve knots comes from AAD through
the pricers — one backward pass per quote gives a whole row, and no bump loop is needed. That is
the same derivative the [quote-sensitivity](quote_sensitivities.md) thread carries the other way,
through the bootstrap's fixed point into `dV/dq`; the residual is written once and differentiated
twice, which is why the two land together.

**Two declared fields are gone**, on the terms every store here is held to: a field the engine does
not read is not declared. `Zero_Rate_Grid` named an output grid there is no longer one of, and
`Spot_Offset` named a spot lag that a quote's own `Deal` block states as an `Effective_Date` —
which is the whole point of a quote carrying an instrument rather than describing one.
`Quote_Type` loses `Rate` and `Price` for the same reason: a futures price and a money-market rate
on a different basis are conventions the family would have to author differently, and a value the
solve does not implement is the same defect as a field nothing reads.

**The loose ends, settled.** The curve this writes is an `InterestRate`, not the `<ClassName>`
parameter block the other four write, so the class declares `price_factor_type` and
`Config.bootstrap`'s "wrote no `<name>.*` price factor" check reads it. And the interpolation of a
solved curve comes from `Price Factor Interpolation` rather than from the block — see the
`Interpolation` note in [Conventions](conventions.md#registries-not-functions).
