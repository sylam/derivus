# Roadmap

The build plan and the engineering status of derivus, for whoever builds on it next: what is to be
closed before 0.5.0, what is designed but not built, and how a change is verified. What was built,
and the numbers it was measured to, is in the commit messages; the tests that hold each rule are
named on the developer page that states it.

Open limitations that move a number a validator reads, with their measured sizes, and the decisions
waiting on a desk are on [Model Limitations](../limitations.md), the validator's page. A fix that
lands deletes its row there and here.

## Before 0.5.0 {#before-0-5-0}

The close-out list. Each row names its fix and sizes it (S, M, L); a landing deletes it, and a row
found later is added here or, where it moves a validator's number, on Model Limitations.

**The model rows**, on [Model Limitations](../limitations.md): three rows marked S land as fixes -
the FX option's expiry print table, the credit Monte Carlo's write into its config and, at S-M, the
Skew vega tent - and four rows a synthetic document measures in an afternoon each get their numbers
and then close: the Asian's flat carry, the cash swaption's annuity, the rate leg's conventions and
the haircut's side. The declared fields not yet read are features kept ahead of their build, not
rows to close.

**The record and the service.**

- **The files a job names declared in the record** (2026-10-05, S). Market data and calendar files
  are named by path, so a replay reads whatever the file holds then, and outside the XVA tab the
  calendar is in no hash (two jobs differing only in a calendar priced 1.1e-3 apart under one result
  id). The fix uses what exists ([the design](architecture.md#market-data-context-job)): the hub
  declares each file under a name and a job names the standing blob. Meanwhile the XVA tab
  re-hashes a file only when its size or modified time moves, missing a rewrite inside one tick, and
  a set refused for a file that moved in the queue keeps the refusal if the file is put back.
- **The service's readers through a loaded context** (2026-10-05, M). Some forty readers take the
  base currency and date, factors, prices and bootstrapper configuration from the book's explicit
  block, against [the design](architecture.md#market-data-context-job). A book naming a market data
  file therefore answers 500 on its status and curves, refuses survival curves and date rolls on a
  bare key, and writes the whole file's factors into itself when one spot is patched; every request
  but the XVA tab's builds its own context, and a market edit makes the XVA tab re-parse the file.
- **Two reads under admission** (2026-10-11, S). `GET /book/risk` and `GET /book/pnl?explain=` run
  valuations on the request thread for any caller, seated or not, and `/book/bloomberg` and
  `/book/securities/verify` queue under the hub's seat whoever asks.
- **A settlement names its agreement** (2026-09-28, about twenty lines over seven files). A payment
  the diary cannot determine is shared across its instrument's positions by holdings: exact when
  every position's settlement is filed, smeared when one is filed alone, named and unshared over
  positions netting to zero. One filed against a payment the diary has dropped is matched within
  the business week before its value date; past that, its day and a window reaching back over it
  stop summing. UNMEASURED. The transition names its agreement, required where the instrument sits
  under more than one.
- **Two live deals under one reference get a diary key** (2026-09-28, M). `POST /book/deals`
  accepts a second live deal under an existing reference, the diary keys neither, and the close
  waits on them for good. The key needs the deal's path beside its reference.
- **The tier recorded on an approval** (2026-09-29, M). Under a policy with both an automatic and a
  four-eyes tier, a copy forging an approval of a ticket only the four-eyes tier would admit is
  caught only by copies agreeing with the hub. Signed history ends at the last checkpoint either
  way.
- **The verb map in force declared at an LSN** (2026-09-29). The oracle judges old frames by today's
  verb map, so a type moved between verbs (a settlement to `settle`) makes every frame filed under
  the old verb read as outside its seat, while the desk runs unchanged. No home outside the gates
  predates a move. The map read like a policy, one read in `verb_for`.
- **A checkpointed index beside `log/`** (2026-09-23, M). `SpineLog.__init__` streams every segment
  to build the head, the tag index and the per-LSN offsets, linear in the log, once per read (twice
  on a strip's first paint; once per submitted job for admission, under a fiftieth of the cheapest
  job). Derived and verified like a seed, so an open scans only the tail and the
  doctored-middle-line refusal moves to `verify_home`.
- **The legacy migration, a fill of one under every legacy node** (2026-09-26). A book-file node no
  fill booked prices as written, one unit; closing it through the verbs files a fill of -1, and the
  compile writes the node at that net, the mirror trade. Until the migration such a node sits
  outside the desk's P&L.
- **A consolidated risk read cached by the market's hash** (2026-09-24). `/book/risk` refits, with
  `Quote_Sensitivity` on, every block the book's factors stand on, cached on the deals and the
  market, so a spot-model block pays its warm polish on every uncached read; one block that cannot
  carry a quote derivative (an FX forward outright, a polish above `Stationarity_Tol`) returns the
  whole book in factor space, named. UNMEASURED on a desk book. The fix is the connected bootstrap
  cached by the market's hash, and a per-block retry with the refusing block off prices.

**Speed and memory.**

- **Each SIMM benchmark move solved once** (2026-10-07, M). The `SIMM` calculation re-solves the
  curve or surface each move feeds, per trade and per worker: on one autocall structure the curve
  re-solves are 60 of 67 seconds of wall, 32 damped Newton solves on the host beside 3 seconds of
  pricing, and four workers give 1.3x over one on a 19-trade set. The dependency walk also hands
  each trade its FX chains, about seven repricings per CRIF row.
- **The inner Monte Carlo's step fused** (2026-10-07, re-measured 2026-10-15, M). A one-step-survival
  step is about fifty elementwise kernels on `[batch, 2 x inner]`, and one batch of a 58-leg credit
  Monte Carlo keeps the device 38% busy at 1,024 paths and 63% at 4,096, with 343,000 kernel
  launches, 7,200 host syncs and 6,200 pageable host-to-device copies per batch. A CUDA graph of the
  pure step replays bit-identically to eager and is worth 7.6x on one row of 32,768 paths, 1.9x at
  256 rows of 4,096 and 1.1x at 2,048 rows, so the ceiling is 2.6x at 1,024 paths and 1.6x at 4,096,
  and the syncs and copies go first; `torch.compile` is not bit-identical (5e-6 relative) and is out.
- **The LogVar2FJ walk's cumsum as a matmul** (2026-10-07, S). It takes about a fifth of a value
  batch on its 21-step axis; a matmul by a 21 x 21 upper-triangular ones matrix is about thirty
  times faster and equal to 2.3e-7 relative, so the LogVar2FJ documents would re-bank. The
  inverse-Gaussian root runs in double, about 45% of the batch on a GPU with 1/64-rate double (M).
  Under a gradient the walk holds 79% of peak memory in its checkpoint segments.
- **One row per block on a dense reporting grid** (2026-10-07, S). The streamed strip holds every
  time-grid row of a block together, so memory grows with the rows per fixing interval and a daily
  grid can exceed a device that holds a weekly one; one row per block past about twelve rows per
  interval.

**The code.**

- `Context.run_job`'s four-way branch and the `run_*` functions fold into the calculation classes
  they dispatch to, one entry per class.
- The retired family keys (`config.RETIRED_FAMILY_KEYS`: `InterestRateCurveParameters` and
  `FXVolSurfaceParameters` read as `InterestRate` and `FXVol`) are carried through this release and
  go the release after. A book keyed by them hashes to a new plan once its keys move, so a quote
  pinned before the move asks for a re-quote.
- `derivus_jupyter.py` (tracked, not in the wheel) stays until the web UI exports a calculation as
  a job and links each result table for download, as it does; it then retires. It raises on three
  field names outside its write allowlist and on multi-column Tables outside a four-name allowlist,
  and fourteen output-shaped descriptors have no widget.
- `pv_float_cashflow_list` decides a coupon's resets need folding by comparing their count with the
  cashflows', a shape; an explicit mark on the compiled cashflows would replace it.

**The tests.**

- **What is left of the test fold** (2026-10-09). The record and service modules (635 tests) are
  `fsync`-bound: most mint a home per test where a copy of the session's (`recorded`,
  `tests/conftest.py`) would do, as the diary, P&L and engine modules do. 146 of these tests name a
  mutation nobody has run red, and fourteen, twelve in `test_service.py` and two in `test_mcp.py`,
  patch `session.BloombergSession` or the `blpapi` module where a declared canned-terminal entry
  would do (S); `test_bloomberg_session.py` already drives a scripted fake of the SDK as data.
  `test_notebooks` needs a world file that is not in the repository.
- **The pricer branch census names 38 arcs no test executes** (2026-10-11). The LogVar2FJ
  discrete-barrier arcs are worth a fixture; the rest are diagnostics, completeness elses and new
  code. `gates/pricer_branch_census.py` lists them.
- **Three checks are ungated** (2026-08-21): the vol term structure reaching the monitoring, the
  payoff's forward against the vol surface's, and rate units
  ([Conventions](conventions.md#what-holds-today-and-what-the-purge-left-open)).
- **A declared-versus-read gate on the `Hedging_Problem` knobs**, lost with the mock-built suite.

## Designed, not built

- **The density recursion**: one FFT convolution per monitored date against the block Gaussian, as
  an alternative inner estimator. The daily walk's tape at a desk's path counts does not fit one
  device, hence the per-block checkpointing; no coarser chain is licensed (a 5-day gap read 3.4
  standard errors off on the coupon leg).
- **Barrier state as a fold over fixings, what remains.** `Barrier_Dates` and `Price_Fixing` fill
  from the record at compile, but continuous monitoring (one-touch, partial-time) still prices from
  terms: it needs daily `(low, high)` bars, which nothing files yet; `utils.bars_touched` is the
  predicate, with no caller and, since the test fold, no test. The rule: a window runs from the
  booking's effective time to the expiry cut; a bar inside it decides, touch being weak; a bar that
  does not touch decides "no" even across an edge; a touching bar across an edge refuses by name
  unless finer bars or a filed `determination` decide it. The same rule serves partial-time windows
  and a migrated trade, whose window opens at its fill's booking time. The TARF's and accumulator's
  remaining decisions are folded parameters, not a substituted deal.
- **A payoff-shaped settlement amount in the diary.** An option's settlement row is due with
  `amount: null`: no field holds `Units × max(S−K, 0)`, which is a pricer's answer rather than a
  schedule's. Beside it, the diary's cache key covers the deals and the calculation but not the
  record, so a print filed after a read does not recompile (the rows stay right, `answered`
  resolving prints per request); keying on the record's last fill, amendment or fixing would.
- **A fold of the deal panel's conventions.** A web panel shows all 47 declared keys of a swap where
  7 are the trade; folding the 40 conventions by default ships once something under `web/scripts`
  drives `FieldView`.
- **The deal tree hydrated from the fold** rather than reconciled against it. The record holds
  positions by agreement and portfolio, instrument terms and each agreement's netting set, but the
  book file stays the materialisation, checked by `/book/reconcile`. One composer closes it: a view
  (a portfolio, the book, or an agreement netted across portfolios) and a point in the record give
  the job. The XVA tab reads netting sets off the file until then.
- **Prints as one fact, the archive as their fold.** A rate history, a close, a vendor correction
  and a reset are all cells under `(index, date, source)`: one fact, a source and a blob of dated
  rows (close, low, high), filed at any time. One fold resolves the latest print per cell, sources
  ordered by the `fixings` policy; the diary reads it over a short window and a real-world
  calibration's archive over a long one, materialised as a table named by its hash. The official
  close is already a blob, so the archive grows daily with no vendor file, and this replaces day
  one's declared archive file the first time a recalibration needs the closes since. M.
- **Fixings as a price factor, the deal keeping its dates.** The compile writes prints into each
  deal's table and `plan_hash` hashes deals whole, so a monitored barrier is a new plan every day. A
  static `Fixings` factor per index, fed by the prints fold, moves `values_hash` alone: deals stay as
  booked, two deals on one index cannot disagree, and a missing print is a missing factor by name.
  It closes two rows of the diary as it lands. Six types announce fixings under no index
  (2026-09-30): the double Asian, the equity swap leg and swaplet list and the two energy deals (and
  a composite equity Asian) read a table the fill cannot write yet, the inflation list M-L, and a
  swaption announces no expiry of its own, a design call; the rate deals name their index, but a
  blank one still announces under none, and a past reset their table does not state fixes at 0%
  with a warning. A row naming none blocks `close_check` for ever, so its day gets no close, marks or
  P&L; `UNOBSERVED` in `tests/test_diary.py` lists the open types. And a floating coupon settled
  for another amount books the difference a day late (2026-09-28): its known-rate table is not
  filled from the record, so the coupon is never determined, a different settled amount lands the
  next day, and a swap bought ex-coupon on its coupon day carries the engine's coupon for that day,
  each month summing right. Every pricer reading a past fixing changes (barrier, TARF, accumulator,
  autocall, the Asians, the swap family's known rates, inflation), the banked numbers
  bit-identical. Two steps: fill today's tables from the factor, then move the reads and drop the
  value columns. L.
- **What a product controller's P&L carries beside the marks.** `GET /book/pnl` explains the held
  positions' moves (carry, market per factor, residual) but carries no reserves, no day-one sales
  margin transfer, no interest earned on collateral and no futures variation margin, which no future
  declares. The residual is reported as a share of P&L, with no declared threshold watching it.
- **Sensitivity estimators as first-class objects**: a `SensitivityProfile` per pricer, telling a
  pathwise derivative from one carrying a boundary term.
- **SIMM in-graph, phase 2.** Phase 1, the `SIMM` calculation, writes a CRIF by bump and
  revaluation. Phase 2 replaces its execute with one backward pass per netting set, the CRIF
  projection and aggregation in-graph so the margin's gradient survives, `dSIMM/dtheta` as one
  Hessian-vector product, and the margin in quote space through the calibration. It waits on the row
  below and on second derivatives in quote space, which the calibration refuses backwards; the SIMM
  notebook aggregating the phase 1 CRIF is its oracle meanwhile.
- **Hessian-vector products** instead of materialised Hessians: a `jvp` rule on the recompute node,
  forward-over-reverse, first for `dSIMM/dtheta` and FVA's splits.
- **Incremental XVA as risk-impact v2**: `CVA(book + mirror) − CVA(book)` through
  `CreditMonteCarlo` on the same two-run seam, with a ratio-solve primitive for participating
  forwards.
- **Service layer, what remains**: progress over SSE, a cost estimate from the real grid, per-seat
  budget caps, and screens for the dependency walk and the set-up, which only the MCP binding serves
  today. The Securities screen stays read-only: it names a knot quoted off a drifted or unmapped
  ticker, fixed by a curve row or a seed entry elsewhere.
- **Excel through the service, not an add-in**: the xlwings add-in in `excel_integration/` retires,
  and a desk reaches Excel through worksheets the service writes on demand and the MCP binding and
  the web UI hand over, the quote sheet (`quote_sheet.py`, `derivus://quote/{id}/sheet`) being the
  first. The writer is a service verb, the binding staying `requests` and `mcp`; generalising it to
  Gridcraft, a spreadsheet written in Rust that also runs on Linux, comes after.
- **`bind=` for payoff-only deal fields**: every deal field is structural today, so `/book/solve`
  recompiles on every iterate, even for a strike. Four candidates stay declined with a citation
  (reset rows and value-dependent leaf sets).
- **The `System` store audit**: the last hand-written store, never audited declared-versus-read
  (`Volatility_Delta`, `Master_Curves`, `Swaption_Premiums` read but undeclared; three shipped keys
  read by nothing).
- **Quote-sensitivity non-goals** ([the page](quote_sensitivities.md#non-goals)): no report format
  for a quote delta, no second derivative in quote space, no SABR or SSVI.

## How a change is verified

- **What a change reaches**: `python gates/reach.py <Symbol|Class.method>` lists the consumer
  classes, the callers, the JSON documents that executed the symbol (cheapest first) and the holes
  no document reaches; `--from <Consumer>` inverts it and `--dirty` covers the changed symbols with a
  minimal document set. Its static half is an AST call graph over `derivus/`; its dynamic half, a
  document map traced from `tests/fixtures/` and a maintainers' store outside the repository and
  keyed to the engine commit, cannot be rebuilt from a clone. It is blind to string dispatch outside
  the registries, callables passed as values, virtual dispatch out of inherited bodies, branches no
  data takes and `derivus_bloomberg/`. At its last build (2026-09-17) no document reached 32 of 50
  deals, 18 of 34 pricers, 2 of 6 bootstrapper families and 9 of 17 processes.
- **Which tests a change reaches**: `python gates/impacted.py --dirty --run` selects the tests that
  executed each changed line, off an execution map built at a campaign boundary, and fails open on
  `derivus/__init__.py`, `conftest.py` and unmapped paths. A line most tests run selects most of the
  suite (the curve read's blend reaches 83%). The full suite runs at campaign boundaries. A rename or
  a deletion also runs `python gates/doc_symbols.py`
  ([Conventions](conventions.md#what-a-landing-runs)).
- **The standing readings every landing runs**: sixteen banked documents of the autocall, TARF and
  accumulator campaigns under the lognormal and Hull-White laws, every float they report bit for
  bit, and one lognormal target redemption
  forward to the bit (`-0x1.2b36cda3bf2d4p+5` under the default estimator; the crisp path, one
  `'No'` away, reads `-0x1.2c48f36318e38p+5`). The documents, banks and scripts live outside the
  repository, so a clone cannot rerun them; moving them under `gates/` waits on a check that none
  carries desk data.
- **The LogVar2FJ module, `tests/test_logvar2fj_json.py`** (2026-09-10): 23 tests on a synthetic
  world (`tests/fixtures/data/logvar2fj_world.json`: one EUR index on a USD book, a five-expiry
  skewed ladder, a GBM sibling), mostly one shared fit, holding the GBM limit, the flat surface, the
  fit's contract, refusals by name, the priors-off and quanto twins, the reserve, the four sub-factor
  rows and 76 floats over six repository documents hex for hex. Every killing mutation goes red
  except the wing-RMSE bound's (0.789 against 1.0), whose mutation stalls the fit. The banked floats
  are the GPU's (a CPU-only box re-banks the walk's), and the only autocall fixture fixes once, at
  maturity, so a vol-strip term-structure error is invisible to it.
- **A fixture must not zero the quantity its gate tests** (the checklist and plugin are on
  [Conventions](conventions.md#fixture-degeneracy)): a surviving mutant means the fixture is wrong,
  not that the code is right.

## What this list is for

Most rows here and on [Model Limitations](../limitations.md) were found auditing work that already
passed its tests. The usual cause was a gate exercising one point of a parameter (a bought deal
only, the default monitoring frequency, one netting set); the next, a unification that left siblings
outside its seam ([Conventions](conventions.md#unification-siblings)). When picking up a row, vary
the parameter the defect lives in, and see the mutant die before trusting the test.
