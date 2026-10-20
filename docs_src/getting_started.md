# Getting Started

From an install to a working desk: the library in three lines of Python, then one service holding
a live book, a web UI over it, Claude wired in over MCP, and a quote source ticking the market —
everything a desk needs to book, price, structure and re-mark, all through the same verbs.

## Install

```
pip install derivus                    # the engine alone, enough for the three lines below
pip install "derivus[desk]"            # the engine, DV_Service, the MCP binding and the quote ticket
pip install torch                      # the CUDA build if you have an NVIDIA GPU
```

Derivus needs Python 3.10 or later; [Requirements](require.md) lists every extra and the GPU
notes. A release wheel serves its own web UI, so there is nothing to build.

Working on derivus itself is the clone route instead:

```
git clone https://github.com/sylam/derivus && cd derivus
pip install -e ".[desk,enterprise]" pytest httpx
cd web && npm ci && npm run build && cd ..     # a source tree serves the UI it builds
```

Optional, for a Bloomberg-enabled workstation:

```
pip install --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple/ blpapi
```

Sanity check: start the service and `GET /schema` answers with `engine_version`. Running the test
suite yourself, and reading what each test holds, is [Validating Derivus](validating.md).

## Three lines of Python

A calculation is a JSON document; running one is loading it and asking for the result:

```python
import derivus as rf

cx = rf.Context()
cx.load_json('BaseValuation.Test1.json')
calc, out = cx.run_job(overrides={})
```

`run_job` dispatches on the document's `Calculation.Object` (`BaseValuation`, `CreditMonteCarlo`,
`HedgeMonteCarlo`, `SIMM` or `SACCR`) and returns the calculation object and its output, whose tables are
described in [Understanding Output](output.md). [JSON Configuration](json/index.md) describes every
block of the document, and the service's `GET /schema/job` (below) serves a minimal one that loads
and prices unchanged.

Before running anything, `cx.validate()` reports what would stop the job (a missing market-data
block, a deal missing a required field, a value that is not what its field declares) without
pricing anything, and `cx.describe()` reports what the engine made of the document. The
[API Overview](api_overview.md) covers both, the explicit method for each calculation, patching
market values and replaying a run. The notebook interface over the same `Context` is the
[Jupyter Workbench](running_calcs.md).

## How a job is put together

Every market input a price depends on (an FX rate, an equity price, a curve, a volatility surface)
is a *price factor* of its own, and a deal names the factors it reads rather than carrying their
values. The instrument's terms and the market it is priced on are therefore separate blocks of one
document, and one market can price any book whose factors it holds. A simulation attaches a stochastic process to a price
factor (`Price Models`, routed by `Model Configuration`): an xVA run needs a risk-neutral
calibration of that process, and any other process can be declared to see how a book behaves under
it. Curve rates are continuously compounded zero rates unless a field says otherwise.

## Start the stack

Point the service at a job file — the **live book**, the one document every client reads and
writes:

```
DV_Service --book path/to/book.json
```

With no `--book` at all, the service serves `book.json` from **`DV_HOME`** (`~/.derivus` unless
the env var says otherwise) — the same directory `DV_Bloomberg` keeps its security map and seed
in, so `DV_Service` alone is a working desk. `--no-book` serves the verbs with no book.

On a Bloomberg workstation, add `--tick 30` and the book re-marks itself off the terminal every
30 seconds — the same queued job `POST /book/bloomberg` submits, so a tick serialises with your
pricings; it never provisions (run the fetch once by hand first), and a failure is one warning
line with the book untouched, three in a row stretch the interval fivefold until one lands, and a
beat whose last tick is still in flight is skipped rather than queued.

A wheel serves the UI it shipped with at `/ui` by itself; from a clone, add `--ui web/dist` to
serve the build. (If `DV_Service` is not on your PATH — pip's per-user installs on Windows —
`python -m derivus.service` is the same program.)

The path need not exist: a missing file starts as a **blank book** — no deals, dated today, with
the skeleton's USD market data aboard so the first booking has something to validate against. An
existing file is never touched.

Windows note: book writes are atomic (write-then-replace), which fails if another program holds
the file locked — Excel does, most editors don't. View the book through the UI rather than
keeping it open in Excel.

Open `http://127.0.0.1:8000/ui/`. The portfolio tree, the market data (curves and surfaces
plotted), the calculation and its results all render from the engine's own schema, and the page
follows the book: a deal booked or a value changed by *any* client appears within a couple of
seconds. `GET /schema/job` serves a minimal job document if you need a starting point.

## Wire in Claude

```
claude mcp add derivus -- DV_MCP
```

(`RF_SERVICE_URL` points it at a service somewhere else; it is the same variable the Excel
add-in reads.) Claude now has the full tool set — discovery, booking, amending, pricing,
structuring, market ticks — and things like this work as plain sentences:

> book a 3m par USD ZAR fx forward with a 2m USD nominal and a 200k sales margin

> make OPT1's notional 3m

> solve the strike so this option marks at 500k, then book it

Booking is **validate-before-write**: nothing lands in the book unless the engine has nothing to
say against it, and a refusal comes back as the engine's own messages. The full tool contract is
in [the MCP page](developer/mcp.md).

## The same verbs from Python or Excel

Every client is a thin binding of the same HTTP verbs — see the
[API Overview](api_overview.md) for the full table (the structure and quote verbs are MCP- and
HTTP-side today; `ServiceClient` binds the book, market, price and solve half). From a script or
notebook (`ServiceClient` rides the **clone** — `excel_integration` is an add-in in the repo, not
in the wheel; a wheel user talks the same verbs with plain `requests`):

```python
from excel_integration.service_client import ServiceClient

desk = ServiceClient('http://127.0.0.1:8000')
desk.book()                                   # the live document and its etag
desk.book_deal({...})                         # validate, then atomic write
desk.amend_deal('0', {'Amount': 750_000.0})   # deals are addressed by positional path
desk.solve_deal({...}, 'Strike_Price', target=500_000.0, bounds=[12.0, 30.0])
desk.update_market(quotes={...})              # quote blocks in, bootstrap judged, one write
desk.submit(job); desk.poll(result_id); desk.fetch_table(result_id, 'mtm')
```

## Tick the market from Bloomberg

On a workstation with `blpapi`, the loop from terminal to book is (`ServiceClient` again from
the clone — the Bloomberg half is all in the wheel, and `DV_Bloomberg discover` is how the
security map behind `definition` gets built and verified; see `derivus_bloomberg/README.md`):

```python
from derivus_bloomberg import fetch_fx_vol, to_market_prices_block
from derivus_bloomberg.session import BloombergSession
from excel_integration.service_client import ServiceClient

with BloombergSession() as bloomberg:
    snapshot = fetch_fx_vol(bloomberg, definition)   # your verified security map
ServiceClient('http://127.0.0.1:8000').update_market(
    {'FXVolPrices.' + snapshot.surface_name: to_market_prices_block(snapshot)})
```

The service installs the quote block, runs the bootstrap, and writes the solved surface into the
book in one atomic step — refused whole, with the bootstrap's own messages, if anything
complains. A re-post may move only the quoted values and timestamps: a changed pillar, expiry or
convention is a re-authoring, refused by name. Spot-level ticks go through
`desk.update_market(patch={'FxRate.ZAR': {'Spot': 19.25}})` — the engine's `bind='value'`
declaration decides what may move, and the same fields are directly editable in the UI's Market
Data tab.

## Structure against it

The desk's vocabulary is **declared**, not composed by whoever is driving. `derivus/structures.py`
ships seven structures — `Straddle`, `Strangle`, `ZeroCostCollar`, `Seagull`, `ForwardExtra`,
`TargetRedemptionForward` and `Accumulator` — each a class naming its sales names, its legs and a
recipe (price this leg, solve that one to the other's premium). `POST /book/structure` runs the
recipe server-side and files the pending trade plus its `.xlsx` ticket under `DV_HOME/tmp`, named
by the quote id; `POST /book/quote` is the acceptance, called when the client takes the price. Over
MCP the same three acts are `describe_structure` (what a zero-cost collar is, and the parameters it
takes — strikes in **market** terms, a USDZAR floor of 17.50), `solve_structure` and
`book_quote(quote_id)`, so this is a sentence:

> quote a one-year zero-cost collar on 1m USD ZAR with the floor at 17.50

A quote is **client paper**. Every leg prices at the mid, and where the book's `FXVolPrices` carry
`Quoted_Bid`/`Quoted_Ask` what the market charges for that spread is levied on the coordinate the
recipe solves — each leg's vega read per quoted pillar and charged that pillar's own half. The
outcome carries `net` (the price quoted), `net_mid` (what the trade marks at once booked), the
`edge` between them, each leg's `spread_charge` and the pillar rows behind it, and a `spread_note`
where the book quotes no two-way at all. The quote prices on this workstation's **live** spot when
the terminal is up and on the book's last ticked one, with the reason named, when it is not — the
`spot` block says which was used. `book_quote` books the **mirror** of the pending deal, the
desk's side, because a book holds the bank's position; the pending file survives as the audit
trail. Quoting records nothing and accepting records everything — where the desk keeps a book of
record, the quote is filed at the acceptance and the desk's own workflow decides whether the trade
books then or waits for a second seat ([The Spine](developer/spine.md)). The full contract is on
[the structures page](developer/structures.md).

Underneath all of it `solve_deal` is still the single-field verb: a root find over ordinary base
valuations, run server-side. Par forwards (target 0), sales margins (target the margin), a strike
to a premium (bounds around spot). The answer carries the solved coordinate and the deal ready to
book; a linear payoff solves exactly in two pricings.

Greeks ride the same requests: `calculation_overrides={'Greeks': 'First'}` returns the AAD delta
vector, `{'Greeks': 'All'}` the full second-order block — `Greeks_Second` is the cross-gamma
matrix a trading read of an options book needs, spot gamma on the diagonal and vanna against the
bootstrapped surface's own nodes beside it. `First` is one backward pass however many factors;
`All` adds one more backward pass through that gradient per factor node the book depends on.

## Where next

- [Understanding Output](output.md) — what a calculation returns.
- [API Overview](api_overview.md) — every verb, the replay tuple, patching and hashing.
- [Jupyter Workbench](running_calcs.md) — the notebook interface over a `Context`.
- [Validating Derivus](validating.md) — running the test suite and reading what each test holds.
- [Developer docs](developer/architecture.md) — the internal view, and
  [the MCP binding](developer/mcp.md) in particular.
