# Getting Started

The working stack in fifteen minutes: one service holding a live book, a web UI over it, Claude
wired in over MCP, and a quote source ticking the market — everything a desk needs to book, price,
structure and re-mark, all through the same verbs.

The [Quick Start](quickstart.md) covers the three-line in-process route (`load_json` /
`run_job`); this page is the served route those three lines grew into.

## Install

```
git clone https://github.com/sylam/derivus && cd derivus
pip install -e ".[service]" requests mcp
pip install torch                      # the CUDA build if you have an NVIDIA GPU
cd web && npm ci && npm run build && cd ..
```

Optional, for a Bloomberg-enabled workstation:

```
pip install --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple/ blpapi
```

Sanity check: `pytest tests -q -rs` from the repo root. On CPU expect a handful of skips that
name their reasons (licensed market data; CUDA-calibrated oracles).

## Start the stack

Point the service at a job file — the **live book**, the one document every client reads and
writes — and at the UI build:

```
DV_Service --book path/to/book.json --ui web/dist
```

The path need not exist: a missing file starts as a **blank book** — no deals, dated today, with
the skeleton's USD market data aboard so the first booking has something to validate against. An
existing file is never touched.

Open `http://127.0.0.1:8000/ui/`. The portfolio tree, the market data (curves and surfaces
plotted), the calculation and its results all render from the engine's own schema, and the page
follows the book: a deal booked or a value changed by *any* client appears within a couple of
seconds. `GET /schema/job` serves a minimal job document if you need a starting point.

## Wire in Claude

```
claude mcp add derivus -- python mcp_integration/server.py
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
[API Overview](api_overview.md) for the full table. From a script or notebook:

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

On a workstation with `blpapi`, the loop from terminal to book is:

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

`solve_deal` is the structuring verb: a root find over ordinary base valuations, run server-side.
Par forwards (target 0), sales margins (target the margin), a strike to a premium (bounds around
spot), a zero-cost collar (fix one strike, solve the other). The answer carries the solved
coordinates and the deal ready to book; a linear payoff solves exactly in two pricings.

## Where next

- [Running Calculations](running_calcs.md) and [Understanding Output](output.md) — the job
  document and what comes back.
- [API Overview](api_overview.md) — every verb, the replay tuple, patching and hashing.
- [Developer docs](developer/architecture.md) — the internal view, and
  [the MCP binding](developer/mcp.md) in particular.
