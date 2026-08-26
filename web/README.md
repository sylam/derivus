# derivus web UI

A viewer over the derivus HTTP service: the portfolio tree, the market data (curves and surfaces
plotted), the calculation, and a run's results. Slice 1 is **view + run** — fields are read-only;
booking goes through the service's `/book` verbs (the MCP tools, or Excel).

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
