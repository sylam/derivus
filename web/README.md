# derivus web UI

A client over the derivus HTTP service: the portfolio tree, the **blotter** (the same tree read
as a desk list — one row per deal, containers holding their legs, sorted by days-to-roll against
the book's `Base_Date`, with a roll-off window filter), the blotter's two data views — **risk**
and **XVA** — the market data (curves and surfaces plotted), the calculation, and a run's results.
**View + run + scalar edit**: over the live book,
declared scalar fields (amounts, dates, rates, dropdowns) edit in place — saved through the
service's validate-before-write `amend` verb, refusals rendered verbatim, the etag poll doing the
repaint, so there is no client-side edit state at all. Tables, curves and market data stay
read-only for now; booking new deals goes through the `/book` verbs (the MCP tools, or Excel).

**The two data views are reads, and neither has a run button.** *Risk* is `GET /book/risk`: the
book's consolidated mark and its whole-book gradient, counterparty-blind — a headline strip, the
aggregate greeks (sortable by size of exposure or read in curve order), and the per-deal marks,
one row per top-level trade, whose total is the headline. The verb computes on a cache miss and
answers from the cache after, so this view rides the etag the book poll already moves and says
plainly when the book has moved out from under the numbers on screen. *XVA* is `GET /book/xva`:
one row per netting set, the last run over what the book says the set is now — a CVA, its age, its
status (done / failed with the engine's wording / never run / a recalc in flight), and the replay
tuple in an expander. Recalcs are asked for through the MCP verbs (`recalc_xva`), never from this
screen: a credit Monte Carlo is minutes of device time, and a surface that could start one on a
click would start one by accident.

The arithmetic behind both — ages, sorts, totals, staleness — is `src/desk.ts`, pure and free of
React, the way `src/blotter.ts` is for the blotter.

The UI is optional to the core library. Nothing here enters the `derivus` package: it is a client
of the same endpoints every other client uses, built separately and handed to the service as a
directory of static files.

```
npm ci
npm run build                                        # -> dist/
DV_Service --ui web/dist --book path/to/job.json     # serve API + UI + a live book
# open http://127.0.0.1:8000/ui/
```

Development: `DV_Service --book path/to/job.json` in one terminal, `npm run dev` in another —
vite proxies every API path to `127.0.0.1:8000`.

Design notes: the app renders **from the schema** (`GET /schema`) — panels, dropdown values and
table columns are the engine's own declarations, so a new deal type or calculation appears here by
being declared on its class. Views are entries in a workspace registry (`src/registry.ts`); a
future SACCR / backtest / market-archive screen is a new entry, not a refactor. Results render by
SHAPE (date-indexed frame → chart, frame → table, scalar → stat), never by result name.
