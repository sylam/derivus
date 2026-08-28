# MCP Binding

`derivus_mcp/server.py` is the derivus verbs as MCP tools, for a model to book instruments in
plain language. It is the third client of the service — the web UI and the Excel add-in are the
other two — and it owns **no logic**: every tool is a thin adapter onto a `DV_Service` endpoint,
so anything a tool needs that an endpoint cannot answer is a missing verb on the service, never
code in this layer. It lives outside the `derivus` package on purpose — a sibling package in the
same wheel, `derivus_bloomberg`'s shape: importing any of the engine's package pulls the whole
engine (torch included, ~3s) into a process that only talks HTTP, and the import gate in
`tests/test_mcp.py` holds the module to `requests`, `mcp` and `mcp_types` and nothing else.

## Running it

```
pip install 'derivus[desk]'                     # service + binding + quote sheet; mcp needs python 3.10+
DV_Service --book path/to/job.json &            # the service does the work
claude mcp add derivus -- DV_MCP
# a service somewhere else:
claude mcp add derivus --env RF_SERVICE_URL=http://host:8000 -- DV_MCP
```

`RF_SERVICE_URL` is the same variable the Excel add-in reads — one setting configures every
client. There is deliberately no tracked `.mcp.json`: it would pin one machine's paths into the
repo — and `DV_MCP` takes the path out of the command altogether.

## The tools

| | |
| --- | --- |
| `list_instrument_types` | every bookable type, the create-menu grouping, and `containers` |
| `describe_instrument_type` | one type's fields as declared — required, defaults, valid values |
| `describe_structure` | the structures the desk quotes — the sales names, the parameters, the legs, the recipe |
| `describe_calculation_type` / `describe_factor_type` | the same for calculations and factors |
| `job_skeleton` | the envelope, as a job that loads |
| `read_book` / `read_deal` | the live book summarised per deal; one deal verbatim |
| `amend_deal` | merge fields into the deal at a path - the same validate-delta as a booking |
| `book_deal` / `delete_deal` | write verbs onto `POST /book/deals` |
| `price_candidate` / `execute_book` | `POST /book/price` — the what-if; waits, then hands back the id |
| `solve_deal` | `POST /book/solve` — solve one field to a target, get the deal back ready to book |
| `solve_structure` | `POST /book/structure` — quote a declared structure: legs priced at the client's side of a two-way, strikes solved, the mid and the edge said, the pending trade filed under its id |
| `book_quote` | `POST /book/quote` — approve a quote by id and book its mirror, refused exactly as a booking is |
| `update_market_quotes` / `patch_market_values` | `POST /book/market` — quote blocks in (values-only updates, bootstrap judging the write), spot/vol values patched |
| `tick_market_from_bloomberg` | `POST /book/bloomberg` — today's surfaces off this workstation's terminal; provisions the desk on first use, reporting progress while it waits |
| `book_risk_summary` | `GET /book/risk` — the whole book's mark and its biggest gradient rows, counterparty-blind |
| `xva_view` / `recalc_xva` | `GET`/`POST /book/xva` — the cached XVA projection per netting set, and the only thing that moves it |
| `validate_book` / `describe_book` | the read verbs over the live document |
| `poll_result` / `fetch_table` / `deal_values` | results: status, one paged table, `{reference: value}` |

**The blotter's two data views, said out loud in the docstrings.** `book_risk_summary` and
`xva_view` answer two different questions and a model has to know which is which. Risk is
**whole-book and counterparty-blind** — one base valuation with first-order Greeks over everything
the desk holds, cached service-side on the book's own content, so asking again after nothing moved
costs nothing and a booking or a tick moves it. XVA is **per netting set and a cached projection**:
a credit Monte Carlo takes minutes, so it never rides a tick, `xva_view` reads the last run of each
set off `DV_HOME/xva.json` with each row carrying its own `as_of`, and `recalc_xva` — full, or the
sets it names — is the only thing that moves them. Staleness there is data, not a failure, and the
tool docstrings say so, which is what stops a model treating a three-hour-old CVA as live or
paying for a whole book's Monte Carlo to answer "what's my delta".

## The contracts that matter

**Validate-before-write.** `book_deal` never writes a deal something is said against — its own
authoring messages, or market data the book did not already lack. A refusal is a **normal return**
(`{written: false, refused: [...]}`), not a tool error, because the model's next move is to read
the messages and fix exactly what they name. Tool errors are reserved for *cannot proceed*: the
service down (named, with how to start it), an unknown type (with close matches), a parent that
takes no children.

**Deals are addressed positionally.** `deal_path` (`"0/2/1"`) is the identity everywhere —
the same one the web UI's tree uses — because references are not unique in a book.

**Answers are summaries and pointers — the model's context is a budget.** The model needs to know
a deal booked or a calculation ran, never to hold a simulation matrix. So a run comes back as its
replay tuple, its stats and one line per table (`"250 rows x 4 columns"`); a booking outcome is
about *that* booking, with the rest of the book's troubles as counts pointing at `validate_book`;
`fetch_table` pages at most 200 rows and refuses a table wider than 60 columns by name — a
scenario cube belongs in the web UI; and `deal_values` checks a result's shape **before** fetching
anything (the gate records the transport to prove nothing travelled). Drill-down is always
available; it is never the default.

**Par, margin and strikes are `solve_deal`'s job.** The root find runs server-side (brentq inside
bounds, else a secant — exact in two pricings for an amount), the model receives the solved
coordinates and the deal ready to book, and no pricing loop ever runs through the conversation. A
collar or seagull composes from 1D solves under its conventions — fix one strike, solve the
other, margin last — which is exactly why it is not left to the conversation.

**Structures are declared, not composed by the model.** A structure is a class in
`derivus/structures.py`: its `vernacular` (the sales names a desk actually says), its legs, and a
recipe — price this leg, solve that one to the other's premium. `describe_structure` serves
that declaration off `/schema`, so a model reads what a zero-cost collar IS instead of inventing
it, and fills the structure's own parameters — strikes in **market terms** (a USDZAR strike is
15.50; the runner puts it on the engine's axis, and that inversion is the one thing never done
by hand). `solve_structure` runs the recipe server-side and answers with the composed deal, the
per-leg premiums and the net — and where the book's `FXVolPrices` carries a two-way, each leg's
`vol_spread`, the `net_mid` the trade will mark at, and the `edge` between them (`spread_note`
says so where there is no two-way) — writing nothing into the book: the quote lands in
`DV_HOME/tmp/<quote_id>.json` as one pending trade, its sheet beside it when the `quote` extra is
installed (a missing `xlsxwriter` names the install in `files.sheet_note` and never refuses a
quote). A quote prices on the LIVE spot when this workstation's terminal is up and on the book's
last ticked one when it is not, with the outcome's `spot` block naming which and why — the
surface and the curves are always the book's. `book_quote(quote_id)` is the approval that makes
it a trade — booking the MIRROR of the
pending deal, because a quote is client paper and a book holds the bank's position — the same
validate-before-write seam a booking rides — and the file stays afterwards, because what was
quoted at what market is why the book carries what it carries.

**First use provisions the desk, and progress is what keeps that call alive.**
`tick_market_from_bloomberg` is the one verb a model calls for today's market, and on a fresh
machine it is also the setup: `DV_HOME` created, the packaged seed copied in, every candidate the
seed spells verified against *this* workstation's terminal (what it is, whether it prices, when it
last printed), and only then the surfaces fetched and installed through the same quote-block tick.
That verification is minutes, not seconds — so the tool is `async`, every blocking HTTP call goes
through `asyncio.to_thread`, and each poll that carries a `progress` dict is forwarded to the
injected `Context` as `report_progress(done, total, note)`. The notifications are not decoration: a
host resets its call timeout on each one, so they are what lets a five-minute first use finish
instead of timing out, and they are what tells the user which candidate the terminal is on. `ctx`
is injected by the SDK and never appears in the advertised schema (a gate reads the listed tool's
properties to prove it). Past `wait_seconds` the answer is `execute_book`'s pointer —
`{result_id, status, hint}` — because the provisioning carries on service-side either way.

**A ticking service refreshes itself, so the verb is for forcing and for provisioning.**
`DV_Service --tick SECONDS` (30 with no value) runs a metronome thread that submits the *same*
queued Bloomberg job `POST /book/bloomberg` submits — same job class, same single-worker queue —
so a beat's write serialises with pricings and lands atomically exactly as a posted tick does. It
never stacks (a beat whose predecessor is still queued or running is skipped), never provisions
(an unprovisioned `DV_HOME` refuses by name and the cadence carries on — verifying a workstation
is minutes of terminal time and a person's decision), and never dies (a failure is one warning
line naming the cause with the book untouched; three in a row stretch the interval fivefold until
one succeeds). `--tick` on a box where `blpapi` does not import refuses at startup, and `--tick`
with `--no-book` is refused by name. So on a ticking desk `tick_market_from_bloomberg` is what a
model calls to **force** a refresh between beats, or to do the **first-use provisioning** the
cadence will not — which the tool's own docstring says first.

**Market data moves on the engine's terms, never freely.** `patch_market_values` rides the
`bind='value'` seam — the engine's own `patch_market` refuses a structural key by name.
`update_market_quotes` installs or ticks whole `Market Prices` blocks (the shape
`derivus_bloomberg.to_market_prices_block` emits): an update may move only each point's quoted
value and timestamp — a changed pillar, expiry or convention is a re-authoring, refused — and the
bootstrap judges the whole write: any error it reports refuses everything, messages verbatim.
Structure (a new factor, a moved node) is authoring, and stays outside these tools.

## Testing

`tests/test_mcp.py` drives the tool functions directly against the in-process service
(`configure(session=TestClient(service.app))` — the same seam the Excel client uses), so the gates
run with no stdio and no sockets: the import discipline, the registry's contracts and read-only
hints, schema tools equal to the declarations, a booking that prices to the closed form, a
refusal that writes nothing, the byte-identical book-then-delete round trip, and the quoting day
end to end — a structure named, its collar solved to a zero net, the pending trade filed under
`DV_HOME/tmp` and approved into a book that marks the desk's mirror of it — zero-cost, so the
quoted net and the booked mark meet at zero.
