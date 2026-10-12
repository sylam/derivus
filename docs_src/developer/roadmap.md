# Roadmap

The build plan and the engineering status of derivus, for whoever builds on it next: what is
designed but not built, how a change is verified, and the tidy-ups the code owes itself. What was
built, and the numbers it was measured to, is in the commit messages; the tests that hold each rule
are named on the developer page that states it.

Open limitations, with their measured sizes, and the decisions waiting on a desk are on
[Model Limitations](../limitations.md), the validator's page; a fix that lands deletes its row
there.

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
  booked, two deals on one index cannot disagree, a missing print is a missing factor by name, and
  the six-types row of [the engine's limitations](../limitations.md#the-engine) retires. Every
  pricer reading a past fixing changes (barrier, TARF, accumulator, autocall, the Asians, the swap
  family's known rates, inflation), the banked numbers bit-identical. Two steps: fill today's
  tables from the factor, then move the reads and drop the value columns. L.
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
  `Credit_Monte_Carlo` on the same two-run seam, with a ratio-solve primitive for participating
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
- **A declared-versus-read gate on the `Hedging_Problem` knobs**, lost with the mock-built suite
  (`test_hmc_declared_knobs`).

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
- **The standing readings every landing runs**: sixteen banked autocall-campaign documents under the
  lognormal and Hull-White laws, 5,471 floats bit for bit, and one lognormal target redemption
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

## Tidy-ups

- **What is left of the test fold** (2026-10-09). The record and service modules (625 tests) are
  `fsync`-bound: each test mints its own home, because `spine.advancing` keeps its fold pairs
  process-wide and homes copied from one template would read each other's folds. That module-level
  cache belongs on the log handle or the served book
  ([Architecture](architecture.md#market-data-context-job)): S. 146 of these tests name a mutation
  nobody has run red, and 34 patch `session.BloombergSession` where a declared canned-terminal entry
  would do (S). `test_notebooks` needs a world file that is not in the repository.
- `gates/impacted.py` fails open on any change to `derivus/__init__.py`; read line by line it would
  still select nine tests in ten.
- `derivus_jupyter.py` (tracked, not in the wheel) stays until the web UI exports a calculation as
  a job and links each result table for download, as it does; it then retires. It raises on three
  field names outside its write allowlist and on multi-column Tables outside a four-name allowlist,
  and fourteen output-shaped descriptors have no widget.
- `pv_float_cashflow_list` decides a coupon's resets need folding by comparing their count with the
  cashflows', a shape; an explicit mark on the compiled cashflows would replace it.

## What this list is for

Most rows here and on [Model Limitations](../limitations.md) were found auditing work that already
passed its tests. The usual cause was a gate exercising one point of a parameter (a bought deal
only, the default monitoring frequency, one netting set); the next, a unification that left siblings
outside its seam ([Conventions](conventions.md#unification-siblings)). When picking up a row, vary
the parameter the defect lives in, and see the mutant die before trusting the test.
