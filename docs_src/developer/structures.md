# Structures

`derivus/structures.py` is the sales desk's vocabulary as declarations: what a zero-cost collar
IS — its names, its legs, how it composes — lives in the repo rather than in whichever model is
driving the MCP tools. That is the design brief, stated by the owner: the end user's host may be
any LLM, so any finance left in the model's head is finance that sometimes does not happen. The
model's whole job is a sentence into named parameters; the registry and the runner own the rest.

## A structure is a class, and everything on it is a declaration

The class name is the registry key (`globals()` dispatch, the house pattern), and it declares:

- **`vernacular`** — the sales names, comma-separated (`'zero-cost collar, range forward,
  cylinder'`). `describe_structure` matches against these as well as the class name, so a model
  that says "cylinder" lands on `ZeroCostCollar`.
- **`fields`** — the parameters as `schema.F` descriptors (`pair`, `expiry`, `notional`,
  `notional_currency`, the given strikes), so `describe_structure` renders exactly like
  `describe_instrument_type` and every client shares one rendering path.
- **`legs`** — named legs, each a `DealType` plus a PARTIAL deal block. This is the
  [Market Prices quote pattern](market_prices.md#a-quote) verbatim: a leg never restates an
  instrument's fields — the `Instrument` store's declarations ARE the leg's schema — it pins what
  the structure fixes and maps parameter slots. A gate holds every leg's type to a declared
  instrument, the `quote_instruments` rule made checkable again.
- **`recipe`** — the composition as data: `Price('leg')`, and `Solve('leg', 'Field', target)`
  where a target is a literal or `Premium('other_leg')` (with `__neg__` and `__add__`, so a
  collar's financing leg solves to `-Premium('protection')` and a seagull's to the negative of a
  sum). Steps run in order; each prices the leg ALONE against the book document. Adding a
  structure is adding a class — registries, not functions, at the sales layer.

`mapping['Structure']` is `schema.emit_structures(structures)`, assembled with the other stores,
so `GET /schema` publishes the vocabulary and a front end can grow a structures screen for free.

## The runner owns the conventions — all three of them

Parameters arrive in MARKET terms and the engine never shows through. `materialize()` converts
once, for every structure at once:

- a USDZAR strike of 15.50 becomes `Strike_Price = 1/15.50`, because the engine's `FxRate` carries
  REPORTING units per unit of currency and an FX option's strike lives on that axis;
- the option SENSE inverts with the same axis — a market call (the right to buy the base
  currency) is an engine PUT on the quote currency — so no structure declares an orientation and
  none can get it wrong;
- a BARRIER leg crosses on a third axis at the same time. The `Barrier_Price` inverts exactly as a
  strike does (it is a level on the same pair), and the barrier's DIRECTION inverts with it — the
  `Up_And_In` a forward extra declares on the pair is booked `Down_And_In` on a rand notional —
  while In/Out describes the PAYOFF rather than the axis and never moves. `BARRIER_FLIP` is that
  map, declared beside `VANILLA` and applied exactly where `Option_Type` flips. `Option_Style` is
  NOT pinned on a barrier leg: `FXBarrierOption` declares no such field.

The first two were found the expensive way (a session lost to each) and all three are gated the
same way: a straddle quoted on a ZAR notional and one on the USD it buys must net the same to
machine precision, and a forward extra quoted both ways must solve the SAME barrier in market
terms — travelling opposite paths through the runner. (The equivalent notional converts at the
STRIKE, not at the spot: `N` rand is `N / K` dollars, which an at-the-money straddle cannot tell
apart and a 0.97-spot forward extra misprices by exactly the moneyness.) `notional_currency` IS the option underlying, which
makes "is the notional the quote currency" exactly the discriminator the inversion needs.

Strike solves are BRACKETED (brentq over `(0.25, 4.0) ×` the market spot, crossed to the engine
axis), never the secant — `solve_deal_field`'s secant seed lands in the dead flat region for an
engine-axis strike of 0.06. A zero-cost leg no strike inside the bracket can fund refuses by
name rather than returning a wrong branch.

A BARRIER solve is bracketed on the side its own type lives on, off the same ends with a hair of
buffer at the spot so the level never lands exactly on it: `Down_*` over `(0.25, 0.9999) × spot`,
`Up_*` over `(1.0001, 4.0) × spot`, both read on the ENGINE axis the leg's `Barrier_Type` has
already crossed to. A knock-in's premium is monotone in its barrier — toward the spot is likelier
to knock and so larger in magnitude — so brentq owns the root or refuses by name.

One furnishing the runner does that looks like a default and is not: a deal block IS the field
dict the pricer reads (`Deal.__init__` takes it verbatim), so a DECLARED default never reaches it.
`pv_barrier_option` asks for `Barrier_Monitoring_Frequency` and `Cash_Rebate` by name, so
`materialize` writes both onto a barrier leg — `{'.DateOffset': '0M'}`, continuous monitoring in
the wire form a Period field decodes from, and a zero rebate. Without them the deal is SKIPPED at
load: an ERROR line in the log, a leg priced at nothing, and a quote that still returns.

## The spread is quoted, the mid is booked {#two-sided}

A desk does not sell at the mid, and its book does not mark at the offer. Both are true at once
because the two numbers live in different places. The `FXVol` surface in `Price Factors` is
bootstrapped from `Quoted_Market_Value` alone and never moves — that is what every mark on the book
runs off. The `FXVolPrices` block beside it may carry each pillar's `Quoted_Bid`/`Quoted_Ask`
([Market Prices](market_prices.md#fxvolprices)), which is DATA the bootstrap never reads. The
runner is its only reader.

**Where the spread enters.** The ATM rows give a half-spread `(ask − bid) / 2` per quoted expiry,
in the surface's own vol units; a leg's tenor is placed between those pillars linearly and held
FLAT past either end, because a spread extrapolated off the last two pillars is a number the market
never quoted. Each SIDE of the spread then gets a copy of the book with the written surface
shifted flat by that half-spread, and a leg prices on the one its own side names — legs taking the
same shift share it, and every pricing deep-copies again through `alone()`, so no leg can disturb
another's. Moving the written surface is what a leg prices on because a pricing run does not bootstrap:
the block is not read again inside `run_job`, which the two-sided gate demonstrates rather than
assumes. RR and BF rows carry their own two-way and v1 does not consume it — a wing spread has to
skew the smile rather than shift it.

Two refusals sit in that reading and both are the same rule. A CROSSED print — a stale bid through
a live offer — reads as ZERO-WIDE rather than as a negative spread, `max(0.0, (ask − bid) / 2)`,
because the one thing a desk must not do with a broken print is pay a client for it. And a leg
carrying no `Buy_Sell` at all refuses outright wherever the book has a two-way: a leg with no side
has no side of the market to be dealt on, and either guess charges the spread the wrong way round.
Both are stated where the half-spread is read, not left to the caller.

**The sign rule.** A leg's `Buy_Sell` is the CLIENT's side. What the client buys is offered at the
**ask** vol (`+half`); what they sell is taken at the **bid** (`−half`). A solve then iterates its
leg on that leg's own shifted copy against targets taken from the other legs' shifted copies, so
the solved coordinate is a realistic two-sided quote by construction: the forward extra's barrier
comes IN toward the spot, the collar's cap comes IN toward it, and both are the participation the
client gives up for the spread.

**`net` versus `net_mid`.** `net` is what the client is quoted — zero, for a zero-cost structure,
at the two-sided vols. `net_mid` is one extra pass over the finished legs against the UNSHIFTED
book: what the trade marks at the moment it is booked. Both are in the same sign convention as
every premium here, the CLIENT's, so the desk's captured edge on a zero-cost structure is
`net − net_mid` — measured 113.85 USD on a 1M ZAR one-year forward extra at a 0.4-vol-point ATM
spread, against a barrier that moved from 22.401 to 22.255.

Each leg reports the signed shift it took as `vol_spread` (0.002 is 0.2 vol points), or `None`
where the book quotes no two-way at all. In that case every shift is zero, `with_vol_shift` hands
back the document ITSELF rather than a copy, and the quote is bit-identical to the one the runner
has always given — the gate compares it float for float against a book carrying a zero-wide
two-way, so the presence of the data cannot move a price. `spread_note` names the absence.

**The mirror.** Everything above is CLIENT paper — the legs carry the client's side, `net` is what
the client pays — and a trading book holds the BANK's position, so `structures.mirror` is the one
seam where paper becomes position: every leg's `Buy_Sell` flipped, nothing else touched, sales
margin never entering (a mirror is a pure change of side). `/book/quote` books the mirror, so a
two-sided quote's `edge` lands on the book as positive day-one P&L, and the risk-impact step
prices the book PLUS the same mirror — the risk measured and the trade booked are one object, and
a sign cannot disagree between them. The pending file keeps the client frame it was quoted in; the
flip is the booking's act, by the owner's ruling: quote client-frame, mirror once, book the mirror.

## The risk prices the spread {#risk-impact}

The two-way above is what the MARKET charges for a trade. What a DESK charges is the cost of
hedging the RESIDUAL that trade leaves on its book — and the whole point of the design is that this
is a measurement, priced at the market's own two-way, rather than a bp-per-skew number somebody
invented. A trade that nets the book down costs the desk less to carry and is quoted tighter; a
trade that piles risk on is quoted at the full spread and no wider.

**The measurement.** The base pass is unchanged — quote two-sided, full half-spread per leg. Then
the composed candidate is put through `structures.mirror` (the same verb `/book/quote` books
through, so the risk measured and the trade booked are ONE object and a sign cannot disagree
between them) and the book's vol risk is read twice: the book alone, and the book with the mirror
spliced in through `book_node`. Both are `BaseValuation` with `Greeks: 'First'` — one backward off
the ROOT netting set, so a leaf's `.grad` is the whole portfolio's.

**The coordinates are QUOTE space.** `Quote_Sensitivity: 'Yes'` goes onto the `FXVolPrices` block
of the risk run's own copy, which is bootstrapped in the same `Context` that prices it — the
attachment is harvested at BOOTSTRAP, not at run
([Quote Sensitivities](quote_sensitivities.md#the-attachment)) — and `Config.quote_leaves` then
holds the ATM/RR/BF quote leaves the surface was built from. So a bucket is `dV/d(ATM 1)`,
`dV/d(RR 0.25 1)`, `dV/d(BF 0.25 1)`: what the desk would actually have to trade, not a node of a
log-moneyness grid. Descriptors are summed across every published block, which is the
[collision rule](quote_sensitivities.md#the-attachment) — one JSON number can feed two chains and
each family's partial is correct while neither is the answer. The switch is worth exactly zero in
the forward pass, so turning it on cannot move a price.

**The charge.** `quote_two_way` reads EVERY pillar's half-spread off the same block, keyed by the
same descriptor the leaves are published under (`FXVolSurfaceParameters.descriptor`, reused rather
than re-derived — a second copy of the naming rule is a copy that drifts into pricing no bucket at
all). A bucket's cost is the move in ABSOLUTE risk times that bucket's own half: `dV/dq` is already
a vega in report currency per unit of quote, so the product is money and nothing converts it.
Summed, a NEGATIVE total is a saving. On the gate's book — a desk holding one collar, quoted the
same collar back — the offset moves `ATM 1` by −613.86 at a 0.002 half, `BF 0.25 1` by −626.38 at
0.001 and `RR 0.25 1` by −8748.99 at 0.001, for a measured saving of **10.6031 USD** against a full
charge of **81.2194 USD**.

**The policy** is a declared `Quote Policy` block on `Calc`, beside `Calculation` and
`MergeMarketData`. Not inside `ExplicitMarketData` beside `Market Prices`: `Context.load_json`
does `cfg.params[section].update(...)` and a section `Config` does not declare raises `KeyError` on
load — measured, not assumed — and a mandate is not market data anyway. Every reader of a job walks
`Calc` by name, so an unknown key there travels through load, pricing and the book file untouched,
and `structures.quote` is its only reader exactly as it is the only reader of `Quoted_Bid`. Five
fields, each read with `.get`:

| field | default | what it decides |
| --- | --- | --- |
| `participation` | `0.5` | how much of a measured saving reaches the client |
| `floor` | `'mid'` | the scale never goes below 0 — a quote never crosses the mid automatically |
| `scope` | `'vol'` | all v1 measures; any other value refuses rather than quoting a scope nobody looked at |
| `bucket_limit` | `None` | a cap on `\|risk after\|` per bucket, past which NO tightening applies |
| `min_ticket_bp` | `0.0` | flat bp of notional, the ops floor under the edge (crossed to the report currency on the same `FxRate` ratio everything else here reads) |

**The ABSENCE of the block is the off switch**, and that is the compatibility contract in the same
shape as the two-sided one: a book declaring no policy never reaches the greeks runs at all and its
`risk.scale` is `None` rather than `1.0` — the difference between "the feature did not run" and "it
ran and decided nothing", which a consumer auditing a quote has to be able to tell apart. A block
declaring `participation: 0` runs the WHOLE layer and lands on the identical floats.

**The scale, and the ceiling.** `charge_effective = max(min_ticket, max(0, charge_full −
participation × saving))`, and `scale = charge_effective / charge_full` clipped into [0, 1]. Two
rulings sit in that clip. A risk-ADDING trade takes no surcharge — the market's own spread is the
ceiling in v1 — so a positive residual cost is simply no saving and `scale` is exactly 1. And the
min ticket is an ops floor UNDER the tightening, not a second ceiling over it: a ticket above the
full spread leaves the scale at 1 rather than lifting the quote through the two-way.

**ONE pass, not a fixed point**, and it is stated rather than hidden. The recipe is re-run once with
every leg's half-spread multiplied by `scale`, threaded through the same two-sided machinery rather
than a second shift path. But the risk was measured on the FULL-SPREAD candidate, and the re-solve
moves the solved coordinate, so the tightened structure's residual is not exactly the one that was
priced. The miss is second order — a strike shifts by the spread, the risk shifts by the strike —
and measured on the gate's book it is **0.0196 USD on 75.92**, 0.026%: `charge_effective` 75.9178
against a realised `edge` of 75.8983. Iterating to a fixed point would pay a greeks run per iterate
to chase that. The `risk` block is honest about which candidate it measured: its buckets are the
full-spread candidate's.

**What it does to a quote.** Same book, same collar, same policy, opposite sign of the standing
position: the repeat quotes a cap of 19.16949443 (the full-spread cap, to the bit) and the offset
quotes 19.17396280 — 6.5% of the way back from the full spread toward the mid cap of 19.23862842.
Client-better, and never through the mid.

**The cache.** The book-alone half of the measurement does not depend on what is being quoted and
moves only when the market ticks or something books — both of which change the book's content
etag — so it is kept in a bounded module dict keyed by that etag, and a desk quoting repeatedly
against a standing book pays one greeks run per quote instead of two. Measured: a miss is
**30.5 ms** and a hit **0.115 ms**, against a cold run (bootstrap plus a `Greeks: 'First'`
`BaseValuation`) of 105.8 ms. Bounded at 16 entries because a book that ticks every 30s would
otherwise leak a vector per tick.

## The quote lifecycle

**The spot is live, the surface is ticked.** Before the recipe runs, `StructureJob` puts this
workstation's terminal spot onto its OWN copy of the book (`service.patch_live_spot` →
`structures.with_live_spots`, the exact inverse of `engine_spot`) — the book file is never written
by a quote, because a spot is `bind='value'` data. Only the spot moves: the vol surface and the
curves stay whatever the 30s cadence last ticked in, and that is a convention rather than a
shortcut — an FX surface quoted in delta space is sticky-delta, meant to be read at whatever spot
is standing, while the spot itself is stale seconds after a quote is given. Failure is fast and
NAMED: one request on a 2s budget, and a failure that reached the terminal is remembered
process-wide for 30s so consecutive quotes skip the attempt instead of each re-paying it. An
unprovisioned `DV_HOME`, a missing blpapi and a pair the security map never verified all fall back
the same way — never an error, never provisioning. Every outcome carries `spot`:
`{value_market, source: 'terminal'|'book', note}`, where `value_market` is read back off the
document the legs actually priced against, so what is reported and what was priced cannot disagree.

`POST /book/structure` runs the recipe as one queued job and files TWO artifacts under
`DV_HOME/tmp/<quote_id>`: the pending trade (`.json` — the outcome plus the composed
`StructuredDeal`, ready to book) and its ticket (`.xlsx` via `derivus/quote_sheet.py`, the
`quote` extra — legs in market terms, the market data used, values only, no formulas, `created`
pinned to the book's `Base_Date`).

TWO ids name the quoting ACT, and they are not the same hash. The runner's `quote_id` — structure,
params, the market the document was carrying, and a submission clock — is what both files are
named by, and it is the one an approval quotes. The service's `result_id` — book etag, structure,
params, and its own submission clock — names the queued JOB, exactly as `/execute` does. Both
carry a clock for the same reason: a quote is an ACT, so two identical asks are two quotes, never
one coalesced result, and a refusal is never pinned.

`POST /book/quote` is the approval: the pending deal booked through the SAME validate-before-write
seam as any booking, refused in the same wording, against the book as it stands NOW — and the
pending file survives the booking as the audit trail. A missing `xlsxwriter` never refuses a
quote; the outcome names the install under `files['sheet_note']` instead, beside the paths.

The composed deal carries its legs inside the block under `Children`, and `splice_deal` lifts
them onto the node — the engine walks `node['Children']` and never inside a deal block, and
before the lift lived in the splice a composed candidate priced as an EMPTY container, 0.0 with
nothing said against it, on every verb at once (the roadmap's closed-defect row). The gate prices
the legs, never the container.

## Testing shape

Every gate is the owner's rule made concrete — a real document through the real pipeline, results
against financial identities, nothing monkeypatched: the straddle equals its call plus its put,
the collar, seagull and forward extra net to zero within the solve's own residual, re-quoting the
solved cap as a given strangle reproduces equal premiums, and the whole day (quote → pending file
→ approve → book marks it at ~0) runs over the served book. The forward extra brought one gate of
its own that is not about structures at all: a hand-authored knock-IN plus knock-OUT call must be
the vanilla, which is the first test to demand a number from `pv_barrier_option`'s analytic
knock-in branch (measured 1.1e-16 relative). The declared registry reproduced the first
hand-composed collar's solved cap to the digit, which is the check that the recipe encodes the
composition and not an approximation of it.

The risk-impact gates are three and the same shape. A book with no `Quote Policy` and one with
`participation: 0` quote float for float identically, so the presence of the data cannot move a
price. The registry has no sell-side collar — a collar's client always buys the put and sells the
call — so the opposite SIDE is put on the BOOK rather than into the quote: one book short the trade
and one long it, the same structure quoted into both, and the offset comes out tighter while the
repeat comes out at exactly the full-spread cap. The `RR 0.25 1` bucket is read from both sides in
that gate (`before` equal and opposite, `after` doubled on one and zero on the other), which no
sign error survives. And the two limits are made to BIND on a book holding two of the trade, where
the offset leaves a real residual: a `bucket_limit` under it suspends the tightening and names the
bucket, and a `min_ticket_bp` set inside the band the tightening opens lands the effective charge
exactly on the ticket.

## V1 scope, and the named next steps

Five structures ship: `Straddle`, `Strangle` (no SOLVE — the registry handles recipes that only
price), `ZeroCostCollar` (floor given, cap solved to premium parity), `Seagull` (three legs,
two given, one solved to net zero) and `ForwardExtra` (the protected rate given, the BARRIER
solved — protection plus a sold knock-in call at the same strike, so the client keeps the
favourable move until the pair trades through the level and the structure reverts to a plain
forward at the rate they were protected at). It is the recipe vocabulary's first solved
coordinate that is not a strike, and the first leg that is not an `FXOptionDeal`. Named next, in
order of what they exercise:

RISK-IMPACT PRICING v1 ships with them ([above](#risk-impact)), and its scope is named honestly:
the residual is measured in the VOL book only (`scope: 'vol'`, and any other value refuses), in
QUOTE coordinates off `Quote_Sensitivity` on the `FXVolPrices` block — the per-expiry ATM vega
fallback was budgeted and never needed. It is ONE pass rather than a fixed point, and there is no
surcharge past the two-way. Named next, in order of what they exercise:

- **Incremental XVA — the v2 of the same step.** A counterparty on the quote, and the charge grows
  a second term: `CVA(book + mirror(candidate)) − CVA(book)` through the `Credit_Monte_Carlo`
  engine. The seam is already the right shape — `risk_impact` measures the book with and without
  the mirror and prices the difference — so what v2 changes is WHICH calculation those two runs
  are and what the difference is priced at: a credit charge is the number itself rather than a risk
  times a half-spread, and it is a netting-set question rather than a bucket one.
- **A ratio-solve primitive**, for participating forwards — the recipe vocabulary's first step
  that moves a notional fraction rather than a strike.
- **A tenor-vocabulary note**: `expiry` parses `<n><D|W|M|Y>` through the job grammar's own
  period letters, or an ISO date for a broken date; anything else refuses by name.
