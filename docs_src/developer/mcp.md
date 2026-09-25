# MCP Binding

`derivus_mcp/server.py` is the derivus verbs as MCP tools, for a model to book instruments in
plain language. It is the third client of the service — the web UI and the Excel add-in are the
others — and owns **no logic**: every tool is a thin adapter onto a `DV_Service` endpoint, so
anything a tool needs that an endpoint cannot answer is a missing verb on the service, never code
here. It sits outside the `derivus` package on purpose, `derivus_bloomberg`'s shape: importing the
engine pulls torch (~3 s) into a process that only talks HTTP. `tests/test_mcp.py` holds the module
to `requests`, `mcp` and `mcp_types`.

## Running it

```
pip install 'derivus[desk]'                     # service + binding + quote sheet; mcp needs python 3.10+
DV_Service --book path/to/job.json &            # the service does the work
claude mcp add derivus -- DV_MCP
# a service somewhere else:
claude mcp add derivus --env RF_SERVICE_URL=http://host:8000 -- DV_MCP
```

`RF_SERVICE_URL` is the variable the Excel add-in reads — one setting configures every client.
There is deliberately no tracked `.mcp.json`: it would pin one machine's paths into the repo, and
`DV_MCP` takes the path out of the command altogether.

**The server's instructions are the desk's orientation.** A host reads them once per session and
shows them to the model before it calls anything, so they are `INSTRUCTIONS` in
`derivus_mcp/server.py`: what the desk is, to start with `desk_status`, the shape of a working
day, the wire forms a deal is written in, the FX strike axis, what a refusal means, that quoting is
not booking and where the second seat comes in, that bootstrapping dials and ticker codes are
configured once in the web UI, and — where this desk keeps a record — the five verbs that read it:
what the book owes, whether a close is legal, where the file and the record disagree, the strip of
what has been recorded, and the official closes. The module docstring stays the maintainer's.

## The tools

| | |
| --- | --- |
| `desk_status` | `GET /book/status` — the desk in ONE read: the book's date and currency, its curves and surfaces with when each was snapped, the calibrated models, the netting sets and the last XVA per set, and whether a terminal is present |
| `list_instrument_types` | every bookable type, the create-menu grouping, and `containers` |
| `describe_instrument_type` | one type's fields as declared — required, defaults, valid values |
| `describe_structure` | the structures the desk quotes — the sales names, the parameters, the legs or the variations each is dealt as, the recipe |
| `describe_calculation_type` / `describe_factor_type` | the same for calculations and factors |
| `describe_configuration` | every dial the book's bootstrap can be set with, per section, at the default it stands at |
| `job_skeleton` | the envelope, as a job that loads |
| `read_book` / `read_deal` | the live book summarised per deal; one deal verbatim |
| `amend_deal` | merge fields into the deal at a path — the same validate-delta as a booking |
| `book_deal` / `delete_deal` | write verbs onto `POST /book/deals`; `book_deal` carries the `quantity`, `execution_reference` and `actor` a recorded desk requires, and a delete records nothing so it takes no seat |
| `price_candidate` / `execute_book` | `POST /book/price` — the what-if; waits, then hands back the id |
| `describe_calculations` / `configure_calculation` / `run_calculation` | `/calculations` — the desk's own named calculations: listed, saved one at a time (judged against the type's declarations), and run over the live book or one subtree in the curiosity lane |
| `solve_deal` | `POST /book/solve` — solve one field to a target, get the deal back ready to book |
| `solve_structure` | `POST /book/structure` — quote a declared structure: legs priced at the MID, strikes solved with the two-way charged on them, the mid and the edge said, the pending trade filed under its id. Records nothing |
| `book_quote` | `POST /book/quote` — the ACCEPTANCE: the client took the price, so the quote is recorded and its mirror booked, refused exactly as a booking is |
| `approve_quote` / `reject_quote` | `POST /book/quote/approve` \| `/reject` — a seat's decision over an accepted quote's ticket, where the desk's tiers policy wants a second pair of eyes |
| `update_market_quotes` / `patch_market_values` | `POST /book/market` — quote blocks in (values-only updates, bootstrap judging the write), spot/vol values patched |
| `configure_book` | `POST /book/configure` — one bootstrapping dial merged into its entry, built to be judged, then the market re-bootstrapped |
| `describe_curve` | the book's curves as definitions — rows, conventions, and the interpolation each is built under with the source that named it — and, with none named, the seed's own entries a desk can set up |
| `configure_curve` | `POST /book/curve` — a curve's benchmark rows stated, its own interpolation with them, the block authored from them and the curve solved |
| `set_base_date` | `POST /book/date` — the day the book is valued as of, both spellings of it, with every curve re-rolled onto it |
| `tick_market_from_bloomberg` | `POST /book/bloomberg` — today's surfaces, curve rows and spots off this workstation's terminal; provisions the desk on first use, reporting progress while it waits |
| `book_dependencies` | `GET`/`POST /book/dependencies` — what a candidate, a netting set or the whole book needs from the market, and which seed entry would supply each missing factor |
| `setup_market` | `POST /book/setup` — build that market: discover only what is unknown, install the surface, spots and curves, bootstrap, one write |
| `describe_securities` | `GET /book/securities` — the ticker vocabulary a desk could quote, the map of what a terminal verified, and the IPV join: every curve row with the print behind it, the verdict that rejected it, or `unmapped` |
| `configure_securities` | `POST /book/securities` — one entry of the desk's own seed set or removed, validated by spelling the candidates it now names |
| `verify_securities` | `POST /book/securities/verify` — the named scope re-verified against the terminal, drift named per entry and new names ledgered, reporting progress while it waits |
| `calibrate_spot_model` | `POST /book/model` — fit one pair's spot-model parameters off its built surface and land them in the book, under the family the runner pins unless one is named; the expensive one, on request and never on the tick, and what a TARF or accumulator quote reads |
| `book_risk_summary` | `GET /book/risk` — the whole book's mark and its biggest rows, counterparty-blind: per quote the book's factors are built from, and per factor |
| `xva_view` / `recalc_xva` | `GET`/`POST /book/xva` — the cached XVA projection per netting set, and the only thing that moves it |
| `book_diary` | `GET /book/diary` — every payment, fixing and expiry the book carries, with the amount where the compile determines one and the key a settlement fact names the row by |
| `close_check` | `GET /book/close/check` — whether a close on a day is legal, and the rows it waits on |
| `book_reconcile` | `GET /book/reconcile` — where the book file and the book of record disagree, named by instrument |
| `book_activity` | `GET /book/activity` — the record's strip, one line per event with the head to page from; `since` is that head and `limit` keeps the newest |
| `book_markets` | `GET /book/markets` — the official close standing per market with the close each superseded, the declared names, the snapshots |
| `declare_market` | `POST /book/markets` — the desk's mark: the book's own values filed under a name, `official` wanting a `mark` seat and a `private/` board naming the seat that declares it |
| `declare_close` | `POST /book/close` — the official close over those values, behind `close_check`'s own verdict, a second close superseding the first |
| `export_settlements` | `POST /book/settlements` — the settlement file for one day, struck on the market the desk DESIGNATED for the export and on no other, refusing an undetermined amount by name |
| `file_status` | `POST /book/transition` — the back office's half: a payment settled or a confirmation matched, against the row's own derived key, which is what a close then waits on |
| `validate_book` / `describe_book` | the read verbs over the live document |
| `poll_result` / `fetch_table` / `deal_values` | results: status, one paged table, `{reference: value}` |

## Prompts and resources

A host offers **prompts** as commands a user picks and **resources** as documents it can open, so
both are contract the way a tool schema is. Three prompts, each a short numbered walk the model
follows with the tools: `quote_a_structure` (describe, solve, report the legs at market terms,
accept only on the client's word, and where a tier waits, the second seat's approval and then
accept again), `import_a_legacy_book` (the deals wrapped as one
`NettingCollateralSet` and booked in one call, then marked), and `morning_desk_check` (status,
tick where there is a terminal, the mark, and every XVA row older than today).

Four resources, each one `service().call`: `derivus://book` is the live job document itself
(`read_book` stays the summary a model should hold); `derivus://schema/{store}` is one store of
`/schema` whole, an unknown one refused naming the seven; `derivus://quote/{quote_id}` is the
pending trade a quote filed; and `derivus://quote/{quote_id}/sheet` is the sheet as the `.xlsx`
itself. A quote answers with those last two URIs under `resources`, beside the `files` paths.

**The blotter's two data views, said out loud in the docstrings.** Risk is whole-book and
counterparty-blind: one base valuation with first-order Greeks, cached service-side on the book's
content. XVA is per netting set and a cached projection: a credit Monte Carlo takes minutes, so it
never rides a tick — `xva_view` reads the last run of each set off `DV_HOME/xva.json` with its own
`as_of`, and `recalc_xva` (full, or the sets it names) is the only thing that moves it. Staleness
there is data, not a failure, and the docstrings say so, which is what stops a model treating a
three-hour-old CVA as live or paying for a book-wide Monte Carlo to answer "what's my delta".

## The contracts that matter

**Validate-before-write.** `book_deal` never writes a deal something is said against. The rule is
what the booking NEWLY says: its own authoring messages, any deal the book was not already carrying
a message about — which is how a misspelt `Object` is caught, a node that never became a deal being
keyed by its walk position rather than by a reference it no longer has — and market data the book
did not already lack. The messages are the declarations read back: a field the type cannot price
without, an amount authored as text, a date that is not `{".Timestamp": "YYYY-MM-DD"}`, a value
outside the menu its field declares. A refusal is a **normal return** (`{written: false, refused:
[...]}`), not a tool error, because the model's next move is to read the messages and fix what they
name. Tool errors are reserved for *cannot proceed*: service down (named, with how to start it),
unknown type (with close matches), a parent that takes no children. `price_candidate` and
`solve_deal` run the same check before the what-if or the solve queues, in the same words: a
candidate the book has no market data for is refused there rather than dropped from the run it
asked for.

**A refusal names the thing and the remedy, wherever the model gets it wrong.** `solve_deal` refuses
a field the deal's type does not declare and a `bounds` that is not a pair of numbers before
anything prices, and states the field, the target and the distance left where the search will not
converge. `calculation_overrides` are judged against the calculation's own declarations, so a dial
it does not declare and a value outside that dial's menu refuse rather than being ignored.
`patch_market_values` refuses a value whose type is not the declared field's. `solve_structure`
refuses parameters the structure declares and the client did not state, a pair the book does not
quote (naming the ones it does), an expiry on or before the base date and a notional that is not
positive. A credit Monte Carlo over a book with no model, no loaded deal or no date after the base
date refuses naming which, rather than dying inside the simulation. A book carrying a node whose
`Object` names no deal type — a legacy import, a hand edit — is named by position at the first verb
that compiles it instead of making the whole book unpriceable.

**A RECORDED DESK BOOKS THROUGH THESE TOOLS.** Under a configured `DV_SPINE_HOME` the booking
endpoint additionally requires a signed `quantity`, an `execution_reference` and an enclosing
`NettingCollateralSet` naming a counterparty (`spine_fill`), and every write verb attributes its
fact to a seat: `book_deal` takes all four, `amend_deal` and `solve_structure` take `actor`, and a
delete records nothing so it takes no seat. A desk that keeps no record ignores them all, which is
why they are optional rather than required — the tool schema is one contract for both postures, and
the service's own refusal is what names a missing one.

**Deals are addressed positionally.** `deal_path` (`"0/2/1"`) is the identity everywhere, as in the
web UI's tree, because references are not unique in a book. Another host's booking moves every
position under a path, so `amend_deal` and `delete_deal` take the `reference` the model read at
that path and refuse by name when it no longer holds it, rather than acting on whoever sits there
now — measured under three hosts booking at once, where a delete by position alone removed other
hosts' deals.

**Answers are summaries and pointers; the model's context is a budget.** A run comes back as its
replay tuple, its stats and one line per table (`"250 rows x 4 columns"`); a booking outcome is about
*that* booking, the rest of the book's troubles being counts pointing at `validate_book`;
`fetch_table` pages at most 200 rows and refuses a table wider than 60 columns by name; and
`deal_values` checks a result's shape **before** fetching anything. Drill-down is available, never
the default.

**Par, margin and strikes are `solve_deal`'s job.** The root find runs server-side (brentq inside
bounds, else a secant — exact in two pricings for an amount) and the model receives the solved
coordinates with the deal ready to book, so no pricing loop runs through the conversation. A collar
or seagull composes from 1D solves under its conventions: fix one strike, solve the other, margin
last.

**Structures are declared, not composed by the model.** A structure is a class in
`derivus/structures.py` — its `vernacular` (the sales names a desk says), its legs or the
[variations](structures.md#variations) it is dealt as, and a recipe —
served off `/schema` by `describe_structure`, so a model reads what a zero-cost collar IS instead of
inventing it. It fills the structure's own parameters, strikes in **market terms** (a USDZAR strike
is 15.50; the runner puts it on the engine's axis, and that inversion is the one thing never done by
hand). Where a structure is dealt more than one way it fills the level the client named, or
`buy_currency` / `sell_currency`, or both: the runner selects the one variation consistent with what
was stated, refuses rather than guessing, and the quote reports which it dealt. The menu says which
variations there are, the extra parameter each takes, and whether a direction has to be stated at all.
Every FX `Strike_Price` and `Barrier_Price` declaration carries the axis it lives on, so
`describe_instrument_type` says it before a model books an FX option directly instead.
`solve_structure` runs the recipe server-side and answers with the composed deal, the per-leg
premiums and the net — plus, where the book's `FXVolPrices` carries a two-way, each leg's
`spread_charge` with the pillar rows it was levied on, the `net_mid` the trade will mark at and the
`edge` between them, which IS that charge (`spread_note` says so where there is no two-way, and a leg
no vega reaches carries a null rather than a zero). It writes nothing into the book: the quote lands in
`DV_HOME/tmp/<quote_id>.json` as one pending trade, its sheet beside it when the `quote` extra is
installed (a missing `xlsxwriter` names the install in `files.sheet_note` and never refuses a quote).
Both read back by id — `GET /book/quote/{id}` and the `/sheet` beside it, offered to a host as
`derivus://quote/<id>` and `derivus://quote/<id>/sheet` — so the workbook a client is sent travels
as the file rather than as a path only the service's own machine can open.
A quote prices on the LIVE spot when this workstation's terminal is up and the book's last ticked one
when it is not, the outcome's `spot` block naming which and why; surface and curves are always the
book's.

**QUOTING IS NOT BOOKING, and the instructions say the walk.** `solve_structure` records nothing —
a desk quotes many times a day and the record holds the one that comes back — so the model quotes as
often as the client asks. `book_quote(quote_id, actor)` is the ACCEPTANCE: called on the client's
word, it records the quote and books the MIRROR of the pending deal (a quote is client paper, a book
holds the bank's position) over the same validate-before-write seam, and the file stays afterwards.
Between the two the market moves, which the answer REPORTS under `market` and never refuses. Where
the desk's `tiers` policy wants a second seat the acceptance comes back `{written: false}` with
`accepted`, `tier` and `waits_on`: that seat calls `approve_quote(quote_id, actor)` — or
`reject_quote(quote_id, reason, actor)` — and the model calls `book_quote` again. The `quote_a_structure`
prompt walks the same five steps.

**A waiting tool speaks while it waits.** A desktop host cuts a tool call that stays quiet for about
a minute and resets that clock on every progress notification, and the runs behind these verbs are
measured in minutes: a credit Monte Carlo, a spot-model fit, a first Bloomberg use. So every tool
that sits on a run — `price_candidate`, `execute_book`, `solve_deal`, `solve_structure`,
`recalc_xva`, `calibrate_spot_model`, `tick_market_from_bloomberg`, `verify_securities` — is `async`, puts every blocking
HTTP call through `asyncio.to_thread`, and waits in the one poll loop (`_await_result`), which
notifies the injected `Context` on every poll: `done` the seconds waited, `total` the `wait_seconds`
asked for, and the note the run's status carrying the job's own `progress` note where it publishes
one (the provisioning and the spot-model fit do). The poll is a quarter second for the first two and
a second after that, so a host hears from the call about once a second however long the run takes —
which is what lets a five-minute first use finish instead of timing out. `ctx` is injected by the
SDK and never appears in an advertised schema (a gate reads every listed tool's properties to prove
it). Past `wait_seconds` the answer is the pointer it always was, `{result_id, status, hint}`, the
run carrying on service-side.

**First use provisions the desk.** `tick_market_from_bloomberg` is the one verb a model calls for
today's market, and on a fresh machine it is also the setup: `DV_HOME` created, the packaged seed
copied in, every candidate the seed spells verified against *this* workstation's terminal (what it
is, whether it prices, when it last printed), and only then the surfaces fetched through the same
quote-block tick. That verification is the minutes the notifications exist for, and its answer is
the provisioning's own — what installed, what updated, what was refused — where every other waiting
tool answers the run's summary.

**A ticking service refreshes itself, so the verb is for forcing and for provisioning.**
`DV_Service --tick SECONDS` (30 with no value) runs a metronome thread submitting the *same* queued
Bloomberg job `POST /book/bloomberg` submits — same job class, same single-worker queue — so a beat's
write serialises with pricings and lands atomically. It never stacks (a beat whose predecessor is
still queued or running is skipped), never provisions (an unprovisioned `DV_HOME` refuses by name and
the cadence carries on; verifying a workstation is a person's decision), and never dies (a failure is
one warning line, book untouched; three in a row stretch the interval fivefold). `--tick` refuses at
startup where `blpapi` does not import, and `--tick --no-book` is refused by name.

**A market is set up from what a trade needs.** A booking refused for market data names the factors
the book lacks, and the two verbs that follow are the model's whole move: `book_dependencies` walks
a candidate — spliced the way the what-if splices one, so the answer is that trade's want-list and
not the book's — and hands back each missing factor with the seed entry that would supply it, the
securities that entry spells and how many a terminal has verified; `setup_market` then builds it.
It discovers only the names the map has never heard of, checks every wanted print for freshness
before it fetches anything, and installs the surface, each new currency's spot crossed onto the
engine's axis and each new curve's seeded benchmarks in one atomic write that the bootstrap judges.
Nothing lands that would depend on a block the write will not carry: a new currency is installed as
a pair or not at all, its spot and its curve held with each other and any surface on that leg with
them, each named with what it waits on, and what is left still lands. A model reads three independent facts off the answer — whether the book
moved, exactly what was installed, and what could not be supplied and why (`not_supplied`, beside a
`refused` that only ever rides a write that did not land) — and gets the job's own outcome rather
than the result envelope it arrived in. What lands is ordinary blocks, which is the point: the Curves and Market
Prices screens edit them from then on, and `check` names what to look at there. A candidate the
booking verb would refuse is refused by both verbs in its words, so a misspelt type is never
answered "nothing is missing". No model has to chain five verbs in the right order to get a market,
and none of this is a second installer — both verbs ride the curve verb's own edit closures.

**A curve is set up from its instruments, not from a curve.** `configure_curve` takes the benchmark
rows a desk quotes — a tenor, the security it prints on and a number — and the TENOR is what says
what each instrument is, so a model never authors a deposit or a FRA. The block is re-authored whole
and the market re-bootstrapped in one write, which is why this is a separate verb from
`update_market_quotes`: a benchmark set is structure, and structure is never a tick. A row priced
off the terminal carries the print's own clock, so both this verb and the tick roll the book's date
FORWARD onto the day their quotes were snapped and re-author every other curve there; `set_base_date`
is the one that puts it anywhere, which is how a model back-values a book.

**Market data moves on the engine's terms.** `patch_market_values` rides the `bind='value'` seam —
the engine's own `patch_market` refuses a structural key by name. `update_market_quotes` installs or
ticks whole `Market Prices` blocks (the shape `derivus_bloomberg.to_market_prices_block` emits): an
update may move only each point's quoted value and timestamp — a changed pillar, expiry or convention
is a re-authoring, refused — and the bootstrap judges the whole write, so any error it reports
refuses everything, messages verbatim. Structure (a new factor, a moved node) is authoring and stays
outside these tools.

## Testing

`tests/test_mcp.py` drives the tool functions directly against the in-process service
(`configure(session=TestClient(service.app))` — the seam the Excel client uses), so the gates run with
no stdio and no sockets: the import discipline, the registry's contracts and read-only hints, schema
tools equal to the declarations, a booking that prices to the closed form, a refusal that writes
nothing, the byte-identical book-then-delete round trip, and the quoting day end to end — a structure
named, its collar solved to a zero net, the pending trade filed under `DV_HOME/tmp` and approved into
a book that marks the desk's mirror of it at zero.
