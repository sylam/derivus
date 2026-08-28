# Market Prices

`Market Prices` is the risk-neutral half of the market data. Where `Price Factors` holds the
observed curves and surfaces a historical calibration produces, this section holds the **quotes**
a risk-neutral model is fitted to, and a *bootstrapper* turns each block into the factor or model
parameters the simulation reads.

This page is the design. All six families are built; `FXVolPrices` was the last, and the two
sections below are what each was built to.

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
bootstrapper class — and that list is what a `DealType` dropdown offers. The gate that held it to
the declared deal types went with the mock-built suite; the structure registry's twin
(`test_the_registry_publishes_exactly_the_declared_structures`) is the surviving copy of the rule.
Nothing about a swap has to be described twice for a swap to be quotable.

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

The six families, and what each fits:

| Family | Quotes | Writes |
| --- | --- | --- |
| `GBMAssetPriceTSModelPrices` | a vol surface, ATM column only — or, where `FXVolPrices` built that surface, [its ATM rows](#fxvolprices) | `GBMAssetPriceTSModelParameters` — an integrated vol curve |
| `CSForwardPriceModelPrices` | European energy futures options | `CSForwardPriceModelParameters` — sigma, alpha |
| `HestonNandiModelPrices` | European options on any spot | `HestonNandiModelParameters` — omega, alpha, beta, gamma\*, H0 |
| `HullWhite2FactorModelPrices` | forward-starting swaps against a swaption surface | `HullWhite2FactorModelParameters` — two sigma curves, two alphas, a correlation |
| `InterestRatePrices` | deposits, FRAs, swaps and FX forward outrights | an `InterestRate` zero curve |
| `FXVolPrices` | ATM vols, risk reversals and butterflies | an `FXVol` log-moneyness surface |

## `InterestRatePrices` — a curve solved from its quotes {#interestrateprices}

The store carried the shape of this one for as long as it existed: a `Points` container holding
`Deal` / `DealType` / `Quote_Type` / `Quoted_Market_Value`, and a `DealType` naming instrument
types. That was the design showing through, and what got built is that design with nothing bolted
on.

**The block.** `InterestRateCurveParameters` declares it: the `Currency` of the curve to build, the
`Day_Count` its tenors are expressed in, an optional `Discount_Rate` naming the curve the quotes
discount on, the solver's three tuning knobs (`N_Iter`, `Tol`, `Damping_Halvings` — the same
vocabulary the calibration classes tune with, and each read with its declared default as the
engine's fallback), the three lifecycle switches (`Quote_Sensitivity`, `Quote_Propagation` and the
`Drift_Tolerance` a ride is refused outside — see [Quote Propagation](quote_propagation.md)), and
the quote `Points`. A blank `Discount_Rate` builds a **self-discounting** curve, which is the
single-curve configuration.

**The quotes.** Each point carries a `Deal` — a deposit, an FRA, a swap, an `FXForwardDeal`, or a
`StructuredDeal` over two legs, authored exactly as it would be in `Trade Data`, because it is the
same declaration — plus `DealType`, `Quote_Type`, `Quoted_Market_Value` and a free-text
`Descriptor`, and a `Use` flag so a quote can be held out without being deleted. `DealType`
supplies the block's `Object`, and the family stamps `Discount_Rate`: what an instrument *projects* off is its own business and it names
that curve itself, what the quote set *discounts* on belongs to the curve set and is stated once.

`Quoted_Market_Value` is read in the unit its own `DealType` reads, and where it lands is a
property of the instrument TYPE — a `FRA_Rate`, a `Swap_Rate`, a pinned `Interest_Rate_Schedule`, a
fixed leg's `Rate` column. That correspondence is the one thing the family knows about a type
beyond the type's own declarations, and it is a registry (`QUOTE_WRITERS`) rather than a branch, so
a new quotable instrument is a row. The family scales nothing centrally: a rate benchmark is quoted
in **percent** because that deal's own field semantics divide by 100 — a `Percent`, a `Basis`, a
schedule the deal scales — so a quote that is not a rate rides the same path untouched.
`Quote_Type` declares the single convention that is built, `Par_Rate`: the solve holds every
benchmark at PV zero.

**A forward outright as a benchmark.** `FXForwardDeal` is the one quotable type whose quote is not
a rate. Its `Quoted_Market_Value` is the forward **outright** — units of `Buy_Currency` per one
unit of `Sell_Currency` — landing as `Buy_Amount = quote × Sell_Amount` on a benchmark that fixes
the sold amount and both discount-rate names, which keeps the PV exactly affine in the quote the
way the par solve and the drift metric both need. A currency's discount curve can then be solved
directly from the points a desk trades, and covered interest parity is written nowhere: the
residual is the forward's own pricer held at PV zero, so the parity is whatever that pricer means.
The residual **crosses currencies**, which the benchmark closure allows by the same rule it applies
to every factor a solve is not solving for — the other leg's curve and both `FxRate` spots are
discovered from the deal's own declarations and enter as detached constants, while the deal
converts to the reporting currency itself, that currency being the solved curve's. Because the
coupling is authored inside the deal rather than in `Discount_Rate`, the **solve order** reads it
there too: a block comes after every block building a curve its benchmark deals name, so a forward
strip authored before the curve its other leg discounts on still solves.

`Quote_Sensitivity` on such a block **refuses**, by name. The overlay that carries a quote
derivative rides cashflow-schedule value columns; an outright lands in `Buy_Amount`, which the
pricer reads as a float off the deal, so no schedule of the compiled forward moves and the refusal
names the benchmark and the type rather than reporting a zero `dV/dq` on the instrument the desk
actually trades. It is measured and not a branch on the type, so it stays silent wherever a quote
does reach a column. `Quote_Propagation` cannot ride one either — a coupled set spanning two
reporting currencies was already refused, and a single-currency block of forwards meets the same
overlay refusal. Full quote-derivative support for an amount-valued quote is not built.

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

**The loose ends, settled.** The curve this writes is an `InterestRate`, not a `<ClassName>`
parameter block, so the class declares `price_factor_type` — as `FXVolPrices` does for its
`FXVol` — and `Config.bootstrap`'s "wrote no `<name>.*` price factor" check reads it. And the
interpolation of a solved curve comes from `Price Factor Interpolation` rather than from the
block — see the `Interpolation` note in [Conventions](conventions.md#registries-not-functions).

## `FXVolPrices` — a smile quoted in delta, and where the conversion runs {#fxvolprices}

An FX smile ticks in as DELTA quotes: an ATM vol per expiry and, per delta pillar, the risk
reversal and the butterfly around it. The surface the pricers read is a log-moneyness one. Both
halves of the conversion between them already existed — the strangle pair
`vol(call) = ATM + BF + RR/2`, `vol(put) = ATM + BF − RR/2`, and the delta-to-strike solve
`Factor2D` runs for a `Malz` surface — so this family is not new maths. **It is the same
conversion moved, and the move is the point.**

**The x-grid is pinned, because refinement is compile-time work.** The delta solve does not
evaluate a smile at prescribed strikes; it refines a log-moneyness grid until interpolating total
variance between the nodes resolves the smile to `Grid_Tolerance`. That grid is a function of the
quotes. Run at factor-construction time — which is where it ran — every vol tick was potentially
STRUCTURAL: a moved node is a moved tenor grid, a new plan and a recompile, for a number that
changed. So the refinement runs in the bootstrap, once, and the grid it produced is part of the
written factor. A re-bootstrap that finds a surface already written **for the same expiries at the
same `Grid_Tolerance`** reuses its grid and moves only the vols on it, which is exactly a
`bind='value'` patch; a quote set whose expiries have changed is not that plan's, and is refined
from scratch. So is one asking for a different tolerance — the tolerance is what SIZES the grid,
and honouring it on the vols while ignoring it on the nodes would make it a field nothing reads.
It is written onto the surface it built and it is structural, unlike the stamp beside it. What the
pin costs is measured rather than hidden: the log says what the pinned grid resolves the CURRENT
quotes to, beside the tolerance it was built at.

`Surface_Type` was doing two jobs and only one of them is about the solver. It names the moneyness
convention the engine reads a surface at — `calc_moneyness` returns log(F/K), and the term
interpolation runs in total variance — and it said a delta smile is the form the block was authored
in. `Factor2D.solves_delta_surface` separates them: a `Malz` block **carrying deltas** is unsolved
and gets solved on construction, as before; one carrying only the solved `Surface` is what this
family writes, and it falls straight through with its grid intact.

**The conversion is vectorized.** It was a Python loop over the x-grid with a scalar `brentq`
inside it, per point, per expiry, per refinement pass. It is now one bisection over the whole grid
(`Factor2D.malz_sigma`), which refines the identical grid, agrees with the loop to 5e-14 vol —
well inside `brentq`'s own 2e-12 `xtol` — and is made of operations an autograd tape can carry,
which the scipy call is not. The `brentq` loop that was its oracle lived in
`tests/test_fx_vol_prices.py` and went with the mock-built suite, so the 5e-14 agreement is a
recorded measurement rather than a standing gate.

**The conventions are declared, and each offers exactly one value**, because the solve implements
exactly one: `Delta_Type` `Forward`, `Premium_Adjusted` `Yes` — the pillar delta is `(K/F)N(d₂)` —
and `ATM_Convention` `Delta_Neutral_Straddle`, the strike `K = F exp(−σ²T/2)` at which that
convention's straddle is delta neutral. A spot delta or an ATMF quote needs different algebra, and
a value the engine cannot honour is the same defect as a field nothing reads. They WERE gated as
maths rather than as strings — the surface must carry the pillar's vol at the strike whose
premium-adjusted forward delta IS the pillar, and must fail the same statement under the
unadjusted delta — and that gate went with the mock-built suite; today only the single-valued
declarations hold the line.

**The ATM row is the surface's ATM vol, and a second family reads it as one.** `malz_skew` places
the ±0.5 label's vol at the delta-neutral straddle strike, so the ATM vol of the surface this
family writes IS the quoted number — nothing has to be read back off the refined grid to recover
it. `FXVolSurfaceParameters.atm_quotes` is the one reader of that rule (`smile` builds the wings
around the same dict), and `GBMAssetPriceTSModelParameters` takes those rows as its ATM column
wherever the surface it is integrating is one this family **wrote** — evidenced by the same
fingerprint `pinned_grid` reads back, never by the name alone. That is what makes a vol delta land
on the number a desk edits — see
[the quote sources](quote_sensitivities.md#the-atm-column), and the defect named beside them.

**The smile differentiates in its own quotes.** `Quote_Sensitivity` — declared here, default `No` —
leaves the written surface connected to the ATM / RR / BF numbers it was built from, so a backward
pass reports `dV/d(risk reversal)` beside `dV/d(surface node)`. The written surface is bit-identical
either way; what the conversion has that the other families do not is a **root find**, and the tape
does not enter it. See
[increment 4](quote_sensitivities.md#the-delta-solve) for the boundary and what taping the
bisection instead would have reported.

!!! warning "One ATM quote, two families, two partial derivatives"
    The two paragraphs above stack: this block's ATM row is also `GBMAssetPriceTSModelPrices`'
    ATM column, so with **both** blocks asking for `Quote_Sensitivity` a single JSON number reaches
    a valuation through two independent maps and its `dV/dq` arrives split over two `quote_leaves`
    entries — under the *same* descriptor string, because a quote is named by what it is and not by
    which family read it. Each half is correct and neither is the total. A consumer must group by
    descriptor across blocks and **sum**; see
    [the collision](quote_sensitivities.md#the-attachment) for the measured numbers.

**A point may carry a two-way, and the bootstrap never reads it.** `Quoted_Bid` and `Quoted_Ask`
are optional columns beside the mid: `derivus_bloomberg` writes them onto a row when the terminal
answered `PX_BID`/`PX_ASK` for that pillar, and leaves the row exactly as it always was when it did
not — a mid-only surface's block is byte-identical to the one this family has always been handed.
Everything below this line reads `Quoted_Market_Value` by name, so the surface, the pinned grid and
every mark on the book are built from the **mid** whether or not the sides are there. The one
reader is the quote layer: [`derivus.structures`](structures.md#two-sided) shifts a leg's own copy
of the written surface by the ATM half-spread to quote a client two-sided. *The spread is the
quote's; the mid is the book's.*

They tick like the mid, too. Both are on the value side of `update_market_quote`'s structure guard,
because a spread widens between one print and the next and a pillar that starts or stops being
quoted two-sided is still the same node of the same plan; a moved `Pillar` or `Expiry` refuses as
it always did.

**Timestamps are data the engine stores and reports.** Each quote row carries when it was seen;
the written surface carries the latest of the rows that built it as `Quote_Timestamp`, its own
as-of. It is declared `bind='value'` because it travels with the vols — a tick delivers new numbers
and the time it saw them together, and a stamp that invalidated the plan would recompile on every
tick. It enters `values_hash` and therefore the replay identity, which is right: the same numbers
read off a different snapshot are a different market event. **Nothing in pricing reads it.** What
counts as too old is a policy, and policy belongs to the consumer, not to the surface.

**That identity is FULL resolution.** `CustomJsonEncoder` writes a midnight `Timestamp` as the
plain date it always did — old files re-encode byte-stable — and a non-midnight one in ISO form
with its time, which the decoder always parsed. So the 09:15 and the 16:30 snapshots of a quote
survive a save as themselves and `values_hash` separates them, which closes the capability this
field was first to depend on.
