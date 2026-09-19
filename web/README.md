# derivus web UI

A client over the derivus HTTP service: the portfolio tree, the **blotter** (the same tree read
as a desk list — one row per deal, containers holding their legs, sorted by days-to-roll against
the book's `Base_Date`, with a roll-off window filter), the blotter's two data views — **risk**
and **XVA** — the **market prices** (the quote ladders, edited and ticked), the **curves** (the
benchmark rows one is set up from), the **bootstrapper** (the dials each price family is fitted
with), the market data (curves plotted, a vol surface read four ways), the calculation, and a
run's results.
**View + run + scalar edit**: over the live book,
declared scalar fields (amounts, dates, rates, dropdowns) edit in place — saved through the
service's validate-before-write `amend` verb, refusals rendered verbatim, the etag poll doing the
repaint, so there is no client-side edit state at all. A quote ladder's VALUE columns edit the same
way — the schema publishes which columns those are (`MarketPrices.values`) and the tick verb posts
the whole block, so structure stays where the engine refuses it. Everything else stays read-only;
booking new deals goes through the `/book` verbs (the MCP tools, or Excel).

A `Surface`-shaped price factor renders as four views over one pure module (`src/vols.ts`): the
**smiles**, one line per expiry over the x coordinate; the **term structure** at one x; the
**surface**, a rotatable 3-D mesh whose `echarts-gl` code is a lazy chunk the base bundle does not
carry; and the **heatmap**, on numeric axes. The axis names are the factor's own declaration — the
tuple its descriptor opens with — so nothing here knows what a moneyness is.

**Curves** is a quote ladder one level up: the instruments a curve is solved from. One card per
curve block the service reads back as the definition it is — the rows under `tenor / security /
quote / use`, the conventions they were authored under as a descriptor panel off the family's own
declarations plus the curve's own interpolation, which is a rule in a section rather than a column
of the block and so comes off what the answer names, and that factor beside them where the market
data store holds it. The other pane sets one up: a seeded curve pre-fills the rows and the
conventions, rows edit in place (add, remove, hold out, state a quote), the conventions edit
through the Bootstrapper screen's panel, and one button posts `/book/curve` — which authors the
block, solves it and bootstraps the market in one atomic write, answering with the knots and the
price factors the run rewrote, or refusing in its own words with the file untouched. The row
edits and the request — the rows, the curve's own scheme, and only the conventions the desk moved
off the seeded ones, since the verb completes the rest — are `src/curves.ts`, checked by
`scripts/curves_check.mjs`.

**Securities** is the vocabulary those rows are quoted off, one read (`GET /book/securities`) in
three panes. *The vocabulary* is what this desk CLAIMS it could quote — the packaged questionnaire
with the desk's own file over it — one card per key each block files an entry by, every field
edited in the shape it already stands in and merged by one button, a refusal rendered verbatim with
the form standing. *The evidence* is what a terminal ANSWERED: each verified entry under the path a
drift is named by, the ledger of what was rejected and why, and a verify button per block and one
for the whole map, the queued job's progress off the results poll and its verdicts rendered where
it lands; a workstation with no terminal refuses at submission, and that sentence is a notice
rather than an error, since the vocabulary reads perfectly well without one. *Every knot* is the
IPV read: one row per quote row the book carries — tenor, security, quote and the print's own
timestamp — with the name a terminal gave that security and when it was verified, the verdict that
rejected it, or `unmapped`, coloured by the service's word and never by a rule of the client's. The
entry edits, the two requests and the join are `src/securities.ts`, checked by
`scripts/securities_check.mjs`.

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
