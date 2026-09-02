# Structures

`derivus/structures.py` is the sales desk's vocabulary as declarations: what a zero-cost collar IS —
its names, its legs, how it composes — lives in the repo rather than in whichever model is driving the
MCP tools. The end user's host may be any LLM, so any finance left in the model's head is finance that
sometimes does not happen. The model's job is a sentence into named parameters; the registry and the
runner own the rest.

## A structure is a class, and everything on it is a declaration

The class name is the registry key (`globals()` dispatch, the house pattern), and it declares:

- **`vernacular`** — the sales names, comma-separated (`'zero-cost collar, range forward, cylinder'`).
  `describe_structure` matches these as well as the class name, so a model that says "cylinder" lands
  on `ZeroCostCollar`.
- **`fields`** — the parameters as `schema.F` descriptors, so `describe_structure` renders exactly like
  `describe_instrument_type` and every client shares one rendering path.
- **`legs`** — named legs, each a `DealType` plus a PARTIAL deal block. This is the [Market Prices quote
  pattern](market_prices.md#a-quote) verbatim: a leg never restates an instrument's fields — the
  `Instrument` store's declarations ARE the leg's schema — it pins what the structure fixes and maps
  parameter slots. A gate holds every leg's type to a declared instrument.
- **`recipe`** — the composition as data: `Price('leg')`, and `Solve('leg', 'Field', target)` where a
  target is a literal or `Premium('other_leg')` (with `__neg__` and `__add__`, so a collar's financing
  leg solves to `-Premium('protection')` and a seagull's to the negative of a sum). Steps run in order;
  each prices the leg ALONE against the book document.

`mapping['Structure']` is `schema.emit_structures(structures)`, assembled with the other stores, so
`GET /schema` publishes the vocabulary and a front end can grow a structures screen for free.

## The runner owns the conventions — all three of them

Parameters arrive in MARKET terms and the engine never shows through. `materialize()` converts once,
for every structure at once:

- a USDZAR strike of 15.50 becomes `Strike_Price = 1/15.50`, because the engine's `FxRate` carries
  REPORTING units per unit of currency and an FX option's strike lives on that axis;
- the option SENSE inverts with the same axis — a market call (the right to buy the base currency) is
  an engine PUT on the quote currency — so no structure declares an orientation and none can get it
  wrong;
- a BARRIER leg crosses on a third axis. `Barrier_Price` inverts exactly as a strike does (it is a level
  on the same pair) and the barrier's DIRECTION inverts with it — the `Up_And_In` a forward extra
  declares on the pair is booked `Down_And_In` on a rand notional — while In/Out describes the PAYOFF
  rather than the axis and never moves. `BARRIER_FLIP` is that map, declared beside `VANILLA` and
  applied exactly where `Option_Type` flips. `Option_Style` is NOT pinned on a barrier leg:
  `FXBarrierOption` declares no such field.

An **accrual** leg asks the axis question a fourth time and gets the first NO. A knock-out is a LEVEL
and crosses exactly as a barrier does, so an accumulator quotes from either side of the pair. A
**target** does not cross at all: it is a cap on a sum of DIFFERENCES rather than a level, and
`1/S − 1/K` is not the reciprocal of `S − K`, so no number in reciprocal units means the client's cap —
two TARFs "capped the same" on the two axes redeem on different paths and are different trades.
`FXTARFOptionDeal`'s `InvertedTarget` flag is not the way out: it moves the whole fixing onto the
reciprocal axis (`eff_intr`, and so `cf_itm` as well as the accrual), which pays `Underlying_Amount`
per unit of MOVE in the pair. That is a coherent product — a rand notional under the flag pays a million
dollars per rand — but not the one `notional_currency` names, since the notional here is an AMOUNT OF
that currency. The two read **0.77% apart** in the solved strike on the gate's book and neither is wrong
about its own product. So `InvertedTarget` is `False` on every leg the runner builds, and
`TargetRedemptionForward` REFUSES a quote-currency notional by name.

All three are gated the same way: a straddle quoted on a ZAR notional and one on the USD it buys must
net the same to machine precision, and a forward extra quoted both ways must solve the SAME barrier in
market terms, travelling opposite paths through the runner. The equivalent notional converts at the
STRIKE, not the spot — `N` rand is `N / K` dollars, which an at-the-money straddle cannot tell apart and
a 0.97-spot forward extra misprices by exactly the moneyness. `notional_currency` IS the option
underlying, which makes "is the notional the quote currency" exactly the discriminator the inversion
needs.

**Strike solves are BRACKETED** (brentq over `(0.25, 4.0) ×` the market spot, crossed to the engine
axis), never the secant — `solve_deal_field`'s secant seed lands in the dead flat region for an
engine-axis strike of 0.06. A zero-cost leg no strike inside the bracket can fund refuses by name.

A **barrier** solve is bracketed on the side its own type lives on, off the same ends with a hair of
buffer at the spot so the level never lands exactly on it: `Down_*` over `(0.25, 0.9999) × spot`,
`Up_*` over `(1.0001, 4.0) × spot`, both read on the ENGINE axis the leg's `Barrier_Type` has already
crossed to. A knock-in's premium is monotone in its barrier, so brentq owns the root or refuses by name.

An **accrual** strike is bracketed over `ACCRUAL_BRACKET` — `(0.5, 2.0) ×` spot — and the reason is
measured. A strip's value is monotone in its strike but SATURATES at the low end: past the point where
every fixing redeems the target at once it is flat at `target × notional` discounted, so moving the end
in gives up no root. What leaving it out gives up is the solve itself — at `0.25 ×` spot on
`fx_tarf_job.json`'s market the TARF prices **NaN** and `brentq` refuses at its own first evaluation,
because a surface quoted over moneyness `[0.8, 1.2]` does not extrapolate to 0.25 as a volatility.
Measured on that fixture: NaN at 0.28, and flat at 99,697.57 — the redeemed target to the cent — from
0.30 through 0.50.

One furnishing that looks like a default and is not: a deal block IS the field dict the pricer reads
(`Deal.__init__` takes it verbatim), so a DECLARED default never reaches it. `pv_barrier_option` asks
for `Barrier_Monitoring_Frequency` and `Cash_Rebate` by name, so `materialize` writes both onto a
barrier leg — `{'.DateOffset': '0M'}`, continuous monitoring in the wire form a Period field decodes
from, and a zero rebate. Without them the deal is SKIPPED at load: an ERROR line in the log, a leg
priced at nothing, and a quote that still returns.

## An accrual leg is one leg and a SCHEDULE {#accrual}

`TargetRedemptionForward` and `Accumulator` are the registry's first MULTI-FIXING structures and its
first legs that are a whole strip. Each declares ONE leg and a recipe of one step —
`Solve('tarf', 'Strike_Price', 0.0)` — because a TARF is dealt at no upfront and the strike IS the
price. `furnish_accrual` is where a leg becomes a strip:

- **the schedule.** `fixing_grid` grows `[[fixing, settlement, observed], ...]` — the untagged row shape
  both declarations read by iterating — from the tenor and `fixing_frequency`, each fixing at
  `base + n × frequency` rather than a step off the last (an offset applied repeatedly from a month end
  walks: 31 Jan + 1M + 1M is 28 Mar), settling `FIXING_LAG` days on, observed 0.0 because a quote is
  struck today. A tenor holding no whole fixing period refuses rather than returning an empty strip.
- **the two ways a strip comes out SHORT**, neither allowed to be silent. A frequency that does not
  DIVIDE the tenor stops at the last fixing that fits — a 1Y ticket at 5M fixes in November and April,
  the `Expiry_Date` becomes the April settlement, and the deal is priced, reported and two-way spread at
  a tenor the ticket does not say. That REFUSES, naming the expiry, the frequency, the last fixing it
  would have produced and both remedies (a frequency that divides, or the broken date quoted directly).
  And a `Base_Date` carrying a TIME (a terminal stamps 16:30) put the final fixing one comparison past a
  midnight expiry and turned twelve monthly fixings into eleven, so the base is normalized to midnight
  before the loop.
- **the expiry.** `FXTARFOptionDeal`'s `Expiry_Date` is the LAST SETTLEMENT — a strip is not over until
  its final cashflow lands — while `FXAccumulatorOptionDeal` declares no such field, so the shared
  block's is REMOVED rather than carried as a key nothing will read. `leg_expiry` reads the last
  settlement where the field is absent, which is the same date either way.
- **the notionals.** `Underlying_Amount` is the notional PER FIXING and `LeverageNotional` is
  `leverage ×` it. `leverage` is the registry's first parameter with a DEFAULT (2.0, the market's own
  gearing), published as the descriptor's `value` and read through `declared()` rather than a `.get`.
- **the model.** Both deals declare `spot_models = ('None', 'HestonNandi', 'HestonNandiComponent')`, of
  which the runner pins only `HestonNandi` (`structures.SPOT_MODEL`) — the component model is the
  autocall ladder's. The switch is a `Valuation Configuration` entry per deal TYPE resolved by naming
  convention off the pair's NON-BASE token — `HestonNandiModelParameters.ZAR` for a USDZAR leg on a USD
  book, whichever side the notional is on, because the base currency is a numeraire and can name no
  block. `spot_model` checks the book for that exact key and pins the model only where it is there: the
  switch on with the factor absent raises inside the engine's dependency loop, which SKIPS the deal and
  logs an ERROR, so a structure that pinned it unconditionally would quote ZERO on every uncalibrated
  book. Where it is absent the leg carries a `note` naming that factor and the verb that installs it.
  The rule needs the BASE as well as the pair, read off `System Parameters.Base_Currency` in the
  EXPLICIT block — the same half `market_data` reads and the only half a quote can write. A book keeping
  its `System Parameters` behind a `MarketDataFile` answers nothing here, and `utils.spot_model_currency`
  REFUSES an unknown base rather than guessing `Underlying_Currency`, which would pin a model the engine
  then looks up under the other name.
- **and the model books with the trade.** The pin is written on the QUOTE's copy of the document,
  because a quote is not a trade and must not touch the book. That copy dies with the answer, so the
  outcome REPORTS what it pinned (`valuation_configuration`), the pending file records it, and
  `/book/quote` merges it into the book inside the same edit closure that splices the deal — one lock,
  one validation, one write. Without that, a leg dealt under a GARCH re-marks as a lognormal on the next
  valuation and nothing says so, because both numbers are plausible. A pin whose parameters the book no
  longer carries REFUSES at the approval rather than booking a switch the engine would raise on.

### The join: one law per pair, and the reader learns the axis {#the-join}

Three keyings met at an accrual leg and did not agree. **They are one rule now**: the pair's NON-BASE
token (`utils.spot_model_currency`, which the engine's lookup, the runner's presence check and the
dependency discovery all call).

| who | keys off | on a USD-base book, for USDZAR |
|---|---|---|
| the engine (`get_spot_model_params_factor`) | the pair's non-base token | `…ModelParameters.ZAR` |
| the calibration (`fx_surface_block`) | the pair's non-base token — the only leg that IS an `FxRate` | writes `…ModelParameters.ZAR` |
| `furnish_accrual` | still forces a TARF onto the pair's BASE, since a target has no reading on the reciprocal | `Underlying_Currency` = USD |

The base currency is a NUMERAIRE, never a rate: `FxRate.<ccy>` is that currency priced in the base, so
`FxRate.USD` is identically one on a USD book and no fit describes it. A USDZAR TARF therefore used to
look up a factor that could not exist and rode GBM however many times the pair was calibrated —
measured on the calibrated book, where its solved strike came out **bit-identical** to its GBM one.
Under the single rule it separates by **3.78%** on that book, against a solve floor of 2.5e-5.

**The forced TARF sits on the reciprocal of the fitted axis, which is a change of NUMERAIRE as well as
of axis** — the deal pays in the other currency. `FxRate.<ccy>` IS the density that changes numeraire,
so the change shifts the innovation by exactly one standard deviation and the fitted
`(omega, alpha, beta, gamma*)` describes the reciprocal as `(omega, alpha, beta, 1 − gamma*)` at the
deal's own carry (`utils.hn_reciprocal_gamma`). One law, two currencies, one parameter — a derivation,
never a second fit, which is why no pricer knows about the axis. Leaving it uncarried leaves one
variance of Siegel drift in the answer: the two orientations of one accumulator then solve strikes
**3.4e-3** apart and the gap does not close with the path count, against **4.2e-6** carried.

The COMPONENT family does not transport — the change puts a state-dependent term in its long-run
intercept, `omega_t + phi(1 − 2·gamma_2)h_t`, and leaves the family — so a component deal on the
reciprocal axis REFUSES by name rather than pricing off a law nobody fitted. A CROSS pair (neither leg
the base) keeps the underlying's own read: both tokens are simulated factors there and the composed
spot's law is out of the ruling's scope.

### What the model is worth, and what it is not {#hn-worth}

The case for Heston-Nandi on an accrual strip is **not "the skew and only the skew"**. On the gate's
book, with the USDZAR parameters `/book/hn` actually fits (`Omega` 2.757e-6, `Alpha` 7.784e-8, `Beta`
1.079e-3, `Gamma_Star` −3529.45, `H0` 7.027e-5; persistence 0.9708, initial vol 13.31% rising to a
long-run 15.64%), an accumulator's zero-cost strike moves **+0.378%** from GBM to Heston-Nandi. Of that
the LEVERAGE CHANNEL alone — `Alpha` to zero with the persistence and the stationary per-step variance
held where the fit put them — is **+0.048%**, about an eighth. The sign of `Gamma_Star` alone is
+0.003%, which is the solve's own Monte Carlo floor at 16,384 paths (2.5e-5 relative) and therefore
*not resolved* at this path count; under a stronger leverage channel at the same persistence (`Alpha`
2e-6, `Gamma_Star` ±474) the same flip is worth **0.43%**.

So what the model is mostly worth on this book is its VARIANCE PATH — a level and a persistence a
lognormal read off the same surface does not have — and the skew is a real but secondary term. The
calibration shows the same asymmetry: the fit reprices its own ten quotes to a worst point of 4.73% and
a weighted residual of 6.21e-5, against **13.13%** and **2.83e-4** with the leverage channel removed.

## The spread is quoted, the mid is booked {#two-sided}

A desk does not sell at the mid and its book does not mark at the offer; both are true at once because
the two numbers live in different places. The `FXVol` surface in `Price Factors` is bootstrapped from
`Quoted_Market_Value` alone and never moves — that is what every mark on the book runs off. The
`FXVolPrices` block beside it may carry each pillar's `Quoted_Bid`/`Quoted_Ask` ([Market
Prices](market_prices.md#fxvolprices)), which is DATA the bootstrap never reads. The runner is its only
reader.

**Where the spread enters.** The ATM rows give a half-spread `(ask − bid) / 2` per quoted expiry, in the
surface's own vol units; a leg's tenor is placed between those pillars linearly and held FLAT past
either end, because a spread extrapolated off the last two pillars is a number the market never quoted.
Each SIDE gets a copy of the book with the written surface shifted flat by that half-spread, and a leg
prices on the one its own side names — legs taking the same shift share it, and every pricing
deep-copies again through `alone()`. Moving the written surface is what a leg prices on because a
pricing run does not bootstrap: the block is not read again inside `run_job`, which the two-sided gate
demonstrates rather than assumes. RR and BF rows carry their own two-way and the leg SHIFT does not
consume it — `atm_two_way` skips every row whose `Quote_Type` is not `ATM`, because a wing spread has to
skew the smile rather than shift it. `quote_two_way` does read the RR/BF halves: it takes every used
pillar's `(ask − bid)/2`, which charges each risk-impact bucket at its own half.

Two refusals sit in that reading, both the same rule. A CROSSED print — a stale bid through a live offer
— reads as ZERO-WIDE rather than as a negative spread, `max(0.0, (ask − bid) / 2)`, because the one
thing a desk must not do with a broken print is pay a client for it. And a leg carrying no `Buy_Sell`
refuses outright wherever the book has a two-way: a leg with no side has no side of the market to be
dealt on, and either guess charges the spread the wrong way round.

**The sign rule.** A leg's `Buy_Sell` is the CLIENT's side. What the client buys is offered at the
**ask** vol (`+half`); what they sell is taken at the **bid** (`−half`). A solve iterates its leg on
that leg's own shifted copy against targets taken from the other legs' shifted copies, so the solved
coordinate is a realistic two-sided quote by construction: the forward extra's barrier comes IN toward
the spot, the collar's cap comes IN toward it, and both are the participation the client gives up for
the spread.

**`net` versus `net_mid`.** `net` is what the client is quoted — zero, for a zero-cost structure, at the
two-sided vols. `net_mid` is one extra pass over the finished legs against the UNSHIFTED book: what the
trade marks at the moment it is booked. Both are in the CLIENT's sign convention, so the desk's captured
edge on a zero-cost structure is `net − net_mid` — measured 113.85 USD on a 1M ZAR one-year forward
extra at a 0.4-vol-point ATM spread, against a barrier that moved from 22.401 to 22.255.

Each leg reports the signed shift it took as `vol_spread` (0.002 is 0.2 vol points), or `None` where the
book quotes no two-way. In that case every shift is zero, `with_vol_shift` hands back the document
ITSELF rather than a copy, and the quote is bit-identical to the one the runner has always given — the
gate compares it float for float against a book carrying a zero-wide two-way, so the presence of the
data cannot move a price. `spread_note` names the absence.

**The mirror.** Everything above is CLIENT paper and a trading book holds the BANK's position, so
`structures.mirror` is the one seam where paper becomes position: every leg's `Buy_Sell` flipped,
nothing else touched, sales margin never entering. `/book/quote` books the mirror, so a two-sided
quote's `edge` lands on the book as positive day-one P&L, and the risk-impact step prices the book PLUS
the same mirror — the risk measured and the trade booked are one object, and a sign cannot disagree
between them. The pending file keeps the client frame it was quoted in; the flip is the booking's act,
by the owner's ruling: quote client-frame, mirror once, book the mirror.

## The risk prices the spread {#risk-impact}

The two-way above is what the MARKET charges for a trade. What a DESK charges is the cost of hedging the
RESIDUAL that trade leaves on its book — a measurement, priced at the market's own two-way, rather than
a bp-per-skew number somebody invented. A trade that nets the book down is quoted tighter; a trade that
piles risk on is quoted at the full spread and no wider.

**The measurement.** The base pass is unchanged — quote two-sided, full half-spread per leg. Then the
composed candidate goes through `structures.mirror` and the book's vol risk is read twice: the book
alone, and the book with the mirror spliced in through `book_node`. Both are `BaseValuation` with
`Greeks: 'First'` — one backward off the ROOT netting set, so a leaf's `.grad` is the whole portfolio's.

**The coordinates are QUOTE space.** `Quote_Sensitivity: 'Yes'` goes onto the `FXVolPrices` block of the
risk run's own copy, bootstrapped in the same `Context` that prices it — the attachment is harvested at
BOOTSTRAP, not at run ([Quote Sensitivities](quote_sensitivities.md#the-attachment)) — and
`Config.quote_leaves` then holds the ATM/RR/BF quote leaves the surface was built from. So a bucket is
`dV/d(ATM 1)`, `dV/d(RR 0.25 1)`, `dV/d(BF 0.25 1)`: what the desk would actually have to trade.
Descriptors are summed across every published block, which is the [collision
rule](quote_sensitivities.md#the-attachment). The switch is worth exactly zero forward, so turning it on
cannot move a price.

**The charge.** `quote_two_way` reads EVERY pillar's half-spread off the same block, keyed by the same
descriptor the leaves are published under (`FXVolSurfaceParameters.descriptor`, reused rather than
re-derived — a second copy of the naming rule is a copy that drifts into pricing no bucket at all). A
bucket's cost is the move in ABSOLUTE risk times that bucket's own half: `dV/dq` is already a vega in
report currency per unit of quote, so the product is money and nothing converts it. Summed, a NEGATIVE
total is a saving. On the gate's book — a desk holding one collar, quoted the same collar back — the
offset moves `ATM 1` by −613.86 at a 0.002 half, `BF 0.25 1` by −626.38 at 0.001 and `RR 0.25 1` by
−8748.99 at 0.001, for a measured saving of **10.6031 USD** against a full charge of **81.2194 USD**.

**The policy** is a declared `Quote Policy` block on `Calc`, beside `Calculation` and `MergeMarketData`
— not inside `ExplicitMarketData`, because `Context.load_json` does `cfg.params[section].update(...)`
and a section `Config` does not declare raises `KeyError` on load (measured), and a mandate is not
market data anyway. Every reader of a job walks `Calc` by name, so an unknown key there travels through
load, pricing and the book file untouched, and `structures.quote` is its only reader. Six fields, each
read with `.get`:

| field | default | what it decides |
| --- | --- | --- |
| `participation` | `0.5` | how much of a measured saving reaches the client |
| `floor` | `'mid'` | the scale never goes below 0 — a quote never crosses the mid automatically |
| `scope` | `'vol'` | all v1 measures; any other value refuses rather than quoting a scope nobody looked at |
| `bucket_limit` | `None` | a cap on `\|risk after\|` per bucket, past which NO tightening applies |
| `min_ticket_bp` | `0.0` | flat bp of notional, the ops floor under the edge (crossed to the report currency on the same `FxRate` ratio everything else here reads) |
| `firm_seconds` | `600` | how long a quote stays approvable — `/book/quote` refuses a pending quote older than this, naming the age, the window and the remedy |

`firm_seconds` is the one field this module carries rather than acts on: the mandate is ONE block a desk
states, and the approval verb reads the clock against the pending file's `quoted_at`.

**The ABSENCE of the block is the off switch**, the same compatibility contract as the two-sided one: a
book declaring no policy never reaches the greeks runs at all and its `risk.scale` is `None` rather than
`1.0` — the difference between "the feature did not run" and "it ran and decided nothing", which a
consumer auditing a quote has to be able to tell apart. A block declaring `participation: 0` runs the
WHOLE layer and lands on the identical floats.

**The scale, and the ceiling.** `charge_effective = min(charge_full, max(min_ticket, max(0,
charge_full − participation × saving)))`, and `scale = charge_effective / charge_full`. Two rulings sit
in that outer `min`. A risk-ADDING trade takes no surcharge — the market's own spread is the ceiling in
v1 — so a positive residual cost is simply no saving and `scale` is exactly 1. And the min ticket is an
ops floor UNDER the tightening, not a second ceiling over it: a ticket above the full spread leaves the
scale at 1 and the REPORTED charge at `charge_full`, rather than lifting the quote through the two-way.

**ONE pass, not a fixed point.** The recipe is re-run once with every leg's half-spread multiplied by
`scale`, threaded through the same two-sided machinery. But the risk was measured on the FULL-SPREAD
candidate and the re-solve moves the solved coordinate, so the tightened structure's residual is not
exactly the one that was priced. The miss is second order and measured on the gate's book at **0.0196
USD on 75.92**, 0.026%: `charge_effective` 75.9178 against a realised `edge` of 75.8983. Iterating to a
fixed point would pay a greeks run per iterate to chase that. The `risk` block is honest about which
candidate it measured: its buckets are the full-spread candidate's.

**What it does to a quote.** Same book, same collar, same policy, opposite sign of the standing
position: the repeat quotes a cap of 19.16949443 (the full-spread cap, to the bit) and the offset quotes
19.17396280 — 6.5% of the way back from the full spread toward the mid cap of 19.23862842.
Client-better, and never through the mid.

**The cache.** The book-alone half of the measurement does not depend on what is being quoted and moves
only when the market ticks or something books — both of which change the book's content etag — so it is
kept in a bounded module dict keyed by that etag, and a desk quoting repeatedly against a standing book
pays one greeks run per quote instead of two. Measured: a miss is **30.5 ms** and a hit **0.115 ms**,
against a cold run of 105.8 ms. Bounded at 16 entries because a book that ticks every 30s would
otherwise leak a vector per tick.

## The quote lifecycle

**The spot is live, the surface is ticked.** Before the recipe runs, `StructureJob` puts this
workstation's terminal spot onto its OWN copy of the book (`service.patch_live_spot` →
`structures.with_live_spots`, the exact inverse of `engine_spot`); the book file is never written by a
quote. Only the spot moves — the vol surface and the curves stay whatever the 30s cadence last ticked
in, which is a convention rather than a shortcut, since a delta-space FX surface is sticky-delta and
meant to be read at whatever spot is standing. Failure is fast and NAMED: one request on a 2s budget,
and a failure that reached the terminal is remembered process-wide for 30s so consecutive quotes skip
the attempt. An unprovisioned `DV_HOME`, a missing blpapi and a pair the security map never verified all
fall back the same way — never an error, never provisioning. Every outcome carries `spot`:
`{value_market, source: 'terminal'|'book', note}`, with `value_market` read back off the document the
legs actually priced against, so what is reported and what was priced cannot disagree.

`POST /book/structure` runs the recipe as one queued job and files TWO artifacts under
`DV_HOME/tmp/<quote_id>`: the pending trade (`.json` — the outcome plus the composed `StructuredDeal`,
ready to book) and its ticket (`.xlsx` via `derivus/quote_sheet.py`, the `quote` extra — legs in market
terms, the market data used, values only, no formulas, `created` pinned to the book's `Base_Date`).

TWO ids name the quoting ACT and they are not the same hash. The runner's `quote_id` — structure, params,
the netting set, the market the document was carrying, and a submission clock — names both files and is
the one an approval quotes. The service's `result_id` — book etag, structure, params, the netting set,
and its own submission clock — names the queued JOB, exactly as `/execute` does. Both carry a clock
because a quote is an ACT: two identical asks are two quotes, never one coalesced result, and a refusal
is never pinned.

`POST /book/quote` is the approval: the pending deal booked through the SAME validate-before-write seam
as any booking, refused in the same wording, against the book as it stands NOW — and the pending file
survives the booking as the audit trail. A missing `xlsxwriter` never refuses a quote; the outcome names
the install under `files['sheet_note']` instead.

The composed deal carries its legs inside the block under `Children`, and `splice_deal` lifts them onto
the node — the engine walks `node['Children']` and never inside a deal block. Before the lift lived in
the splice, a composed candidate priced as an EMPTY container, 0.0 with nothing said against it, on
every verb at once. The gate prices the legs, never the container.

## Testing shape

Every gate is a real document through the real pipeline against a financial identity, nothing
monkeypatched: the straddle equals its call plus its put; the collar, seagull and forward extra net to
zero within the solve's own residual; re-quoting the solved cap as a given strangle reproduces equal
premiums; the whole day (quote → pending file → approve → book marks it at ~0) runs over the served
book; and a hand-authored knock-IN plus knock-OUT call must be the vanilla, which is the first test to
demand a number from `pv_barrier_option`'s analytic knock-in branch (1.1e-16 relative).

**The accrual tolerance is measured, not chosen.** A strip is Monte Carlo priced, so its zero-cost
strike is a root find over an ESTIMATOR — deterministic for a fixed seed, which is what lets `brentq`
own it, but converging on the true root only as the paths grow: the accumulator's two orientations
solve strikes 4.8e-4 apart at 1024 inner paths, 1.3e-4 at 4096, 2.5e-5 at 16384, 3.9e-5 at 65536. The
gates quote at 16384 (about a second) and allow **2e-4**. They pin the axis finding (the accumulator
solves one strike from both sides of the pair while the TARF refuses the second side by name), that
both strips net to zero and land BELOW the forward through `book_node` — not a tautology, since at a
strike of the forward the geared sold leg outweighs the bought one — and that the composed TARF booked
as a `CreditMonteCarlo` reports an exposure profile that is finite, multi-row and DISPERSED, which a
skipped deal cannot be because zero has no spread.

The risk-impact gates are three. A book with no `Quote Policy` and one with `participation: 0` quote
float for float identically. The registry has no sell-side collar, so the opposite SIDE goes on the BOOK
rather than into the quote: one book short the trade and one long it, the offset coming out tighter
while the repeat comes out at exactly the full-spread cap, with the `RR 0.25 1` bucket read from both
sides (`before` equal and opposite, `after` doubled on one and zero on the other), which no sign error
survives. And both limits are made to BIND on a book holding two of the trade: a `bucket_limit` under
the residual suspends the tightening and names the bucket, and a `min_ticket_bp` inside the band the
tightening opens lands the effective charge exactly on the ticket.

## V1 scope, and the named next steps

Seven structures ship: `Straddle`, `Strangle` (no SOLVE — the registry handles recipes that only price),
`ZeroCostCollar` (floor given, cap solved to premium parity), `Seagull` (three legs, two given, one
solved to net zero), `ForwardExtra` (the protected rate given, the BARRIER solved — protection plus a
sold knock-in call at the same strike, so the client keeps the favourable move until the pair trades
through the level and the structure reverts to a plain forward at the protected rate), and the two
[accrual strips](#accrual) — `TargetRedemptionForward` (fixings to the tenor, a target cap in the pair's
own units, the strike solved to zero upfront) and `Accumulator` (the same bargain with a knock-out LEVEL
in place of the cap).

[Risk-impact pricing v1](#risk-impact) ships with them, and its scope is named honestly: the residual is
measured in the VOL book only (`scope: 'vol'`, any other value refuses), in QUOTE coordinates off
`Quote_Sensitivity` on the `FXVolPrices` block — the per-expiry ATM vega fallback was budgeted and never
needed. It is ONE pass rather than a fixed point, and there is no surcharge past the two-way. Named
next, in order of what they exercise:

- **Incremental XVA — the v2 of the same step.** A counterparty on the quote, and the charge grows a
  second term: `CVA(book + mirror(candidate)) − CVA(book)` through the `Credit_Monte_Carlo` engine. The
  seam is already the right shape, so what v2 changes is WHICH calculation those two runs are and what
  the difference is priced at: a credit charge is the number itself rather than a risk times a
  half-spread, and it is a netting-set question rather than a bucket one.
- **A ratio-solve primitive**, for participating forwards — the recipe vocabulary's first step that
  moves a notional fraction rather than a strike.
- **A tenor-vocabulary note**: `expiry` parses `<n><D|W|M|Y>` through the job grammar's own period
  letters, or an ISO date for a broken date; anything else refuses by name.
