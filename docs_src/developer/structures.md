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

## The runner owns the conventions — both of them

Parameters arrive in MARKET terms and the engine never shows through. `materialize()` converts
once, for every structure at once:

- a USDZAR strike of 15.50 becomes `Strike_Price = 1/15.50`, because the engine's `FxRate` carries
  REPORTING units per unit of currency and an FX option's strike lives on that axis;
- the option SENSE inverts with the same axis — a market call (the right to buy the base
  currency) is an engine PUT on the quote currency — so no structure declares an orientation and
  none can get it wrong.

Both conversions were found the expensive way (a session lost to each) and are gated: a straddle
quoted on a ZAR notional and one on the USD it buys must net the same to machine precision,
travelling opposite paths through the runner. `notional_currency` IS the option underlying, which
makes "is the notional the quote currency" exactly the discriminator the inversion needs.

Strike solves are BRACKETED (brentq over `(0.25, 4.0) ×` the market spot, crossed to the engine
axis), never the secant — `solve_deal_field`'s secant seed lands in the dead flat region for an
engine-axis strike of 0.06. A zero-cost leg no strike inside the bracket can fund refuses by
name rather than returning a wrong branch.

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
the collar and seagull net to zero within the solve's own residual, re-quoting the solved cap as
a given strangle reproduces equal premiums, and the whole day (quote → pending file → approve →
book marks it at ~0) runs over the served book. The declared registry reproduced the first
hand-composed collar's solved cap to the digit, which is the check that the recipe encodes the
composition and not an approximation of it.

## V1 scope, and the named next steps

Four structures ship: `Straddle`, `Strangle` (no solve — the registry handles recipe-free
entries), `ZeroCostCollar` (floor given, cap solved to premium parity) and `Seagull` (three legs,
two given, one solved to net zero). Named next, in order of what they exercise:

- **The risk-impact step.** Price the BOOK plus the candidate under `Greeks` and read the skew
  delta in RR/BF coordinates — does the structure add skew or shed it — so a risk-reducing trade
  is charged near mid. The runner already takes the whole document, `dV/d(risk reversal)` is an
  ordinary number ([Quote Sensitivities](quote_sensitivities.md#the-delta-solve)), and with
  `FXForwardDeal` benchmarks every hedge layer now speaks tradable coordinates. What it needs
  from the desk is POLICY — bp per unit of skew, the mid threshold — which is mandate, not code.
- **Barrier legs**, for the forward-extra family — the most-sold structured FX hedge. The barrier
  pricers exist; the registry needs to name a barrier deal as a leg, and the first forward-extra
  quote will drive real traffic through analytic knock-in branches the census only recently saw
  executed.
- **A ratio-solve primitive**, for participating forwards — the recipe vocabulary's first step
  that moves a notional fraction rather than a strike.
- **A tenor-vocabulary note**: `expiry` parses `<n><D|W|M|Y>` through the job grammar's own
  period letters, or an ISO date for a broken date; anything else refuses by name.
