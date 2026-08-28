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

## The quote lifecycle

`POST /book/structure` runs the recipe as one queued job and files TWO artifacts under
`DV_HOME/tmp/<quote_id>`: the pending trade (`.json` — the outcome plus the composed
`StructuredDeal`, ready to book) and its ticket (`.xlsx` via `derivus/quote_sheet.py`, the
`quote` extra — legs in market terms, the market data used, values only, no formulas, `created`
pinned to the book's `Base_Date`). The quote id names the ACT — book etag, structure, params,
and the submission clock — so two identical asks are two quotes and a refusal is never pinned.

`POST /book/quote` is the approval: the pending deal booked through the SAME validate-before-write
seam as any booking, refused in the same wording, against the book as it stands NOW — and the
pending file survives the booking as the audit trail. A missing `xlsxwriter` never refuses a
quote; the outcome names the install under `sheet_note` instead.

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

## V1 scope, and the named next steps

Five structures ship: `Straddle`, `Strangle` (no solve — the registry handles recipe-free
entries), `ZeroCostCollar` (floor given, cap solved to premium parity), `Seagull` (three legs,
two given, one solved to net zero) and `ForwardExtra` (the protected rate given, the BARRIER
solved — protection plus a sold knock-in call at the same strike, so the client keeps the
favourable move until the pair trades through the level and the structure reverts to a plain
forward at the rate they were protected at). It is the recipe vocabulary's first solved
coordinate that is not a strike, and the first leg that is not an `FXOptionDeal`. Named next, in
order of what they exercise:

- **The risk-impact step.** Price the BOOK plus the candidate under `Greeks` and read the skew
  delta in RR/BF coordinates — does the structure add skew or shed it — so a risk-reducing trade
  is charged near mid. The runner already takes the whole document, `dV/d(risk reversal)` is an
  ordinary number ([Quote Sensitivities](quote_sensitivities.md#the-delta-solve)), and with
  `FXForwardDeal` benchmarks every hedge layer now speaks tradable coordinates. What it needs
  from the desk is POLICY — bp per unit of skew, the mid threshold — which is mandate, not code.
- **A ratio-solve primitive**, for participating forwards — the recipe vocabulary's first step
  that moves a notional fraction rather than a strike.
- **A tenor-vocabulary note**: `expiry` parses `<n><D|W|M|Y>` through the job grammar's own
  period letters, or an ISO date for a broken date; anything else refuses by name.
