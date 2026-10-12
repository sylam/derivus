# Roadmap

The live status of the library for a model-validation reader: what is known to be wrong or limited
and by how much, what waits on a desk decision, and what is designed but not built. What was built,
and the numbers it was measured to, is in the commit messages. The methods a validator prices
against are on the Theory, Valuation and Bootstrapping pages
([the LogVar2FJ fit](../bootstrapping/fx_and_equity.md#logvar2fjmodelparameters) among them), and
the tests that hold each rule are named on the developer page that states it.

## Known limitations, open

Grouped by where a validator meets them. Each row says what the piece does, what goes wrong, how big
it is and what would fix it; a row without a measured size says so, since a limitation without a
number is absolution, not documentation ([Conventions](conventions.md#unification-siblings)). Dates
are when the row was found; S, M and L size the fix, never the error.

### The xVA outer and the correlation

- **A bucket knot inside a scenario interval gives the second piece an uncorrelated mixer**
  (2026-09-11). The LogVar2FJ outer takes one correlated normal per scenario interval for its
  residual mixer; where a calendar bucket knot splits an interval, the second piece's mixer comes
  off the process's own stream (logged at INFO). UNMEASURED: no fitted factor on hand has two
  buckets. Separately, past 8.3 standard deviations (one draw in 1e16) the mixer's distribution
  function rounds to 1 in double and the inverse-Gaussian root returns its bracket's top; named,
  not guarded.
- **A quanto correlation has two bases, and nothing checks which.** The Hull-White 2F writer emits
  `Quanto_FX_Correlation_1/2` against each rate factor's own Brownian, while the correlation
  section's rows are the independent normals the Cholesky consumes. Copying one into the other
  silently changes the rate covariance: 0.5 in the second factor's row realises 0.58. The fix is
  one matrix on the driving innovations: the drift reads the section through the loadings (M) and
  the writer emits one basis (L). UNMEASURED beside it: a calibrated market can carry the quanto
  number with no section row, so the simulated FX ignores the rates the drift reads.
- **The quasi-random draws ignore `Random_Seed`** (2026-10-11). One Sobol sequence at a fixed seed
  (`Calculation_State.QUASI_SEED`) feeds the inner block, the Markov-regime, GARCH and basis-linked
  spot models, the LogVar2FJ knot mixers and the hedging fork (unless `Inner_Draws: random`). A
  study across seeds cannot see that error, and the inner block's error is common to every outer
  path rather than averaged over them. UNMEASURED.
- **A Hull-White 2F step that is not positive definite reuses the previous step's Cholesky factor**
  (2026-10-11), silently. UNMEASURED, and how often it fires is not reported.

### The calibration

- **The default LogVar2FJ fit prices its vanillas on a surrogate clock** (2026-09-14).
  `Vanilla_Pricer: Quadrature` replaces the variance clock with a two-moment lognormal, so θ\* is
  fitted to a stand-in and then used by pricers that walk. On the fifteen-quote ladder of
  `tests/fixtures/data/logvar2fj_world.json` its worst quote sat within 0.085 vol points of a
  32,768-path walk and 0.098 of a 1,048,576-path one. The walk's own ratio-estimator bias, 0.36 vol
  points between 8,192 and 32,768 paths, rides the forward-start rows at the default `Paths` of
  8,192. No test re-checks either; UNMEASURED on other ladders. `Walk` is the alternative.

### The autocall, TARF and barrier pricers

- **The collateralised autocall's CVA delta under-reads below 16,384 outer paths** (2026-10-10). On
  a two-year lognormal index autocall under a zero-threshold CSA, the boundary correction is 0.26 to
  0.61 of the pathwise term and carries the scatter. Over four seeds the delta reads short of its
  common-random-number ladder by 8.0% at 1,024 paths (2% over to 25% short), 5.0% at 2,048, 3.7% at
  4,096, 1.7% at 8,192 and within 1.4% at 16,384, short on 12 of the 16 readings below that: a bias
  fading with paths, not noise. UNMEASURED under LogVar2FJ.
- **The TARF's target pin is a kink the crisp estimator cannot see.** The pin fires on 41% to 60% of
  paths. Under the base valuation's default `Branch_And_Weight` the value's CRN ladder is flat to
  0.05% and meets the reported delta; under `'No'` it does not converge (0.24% to 1.41%), though the
  delta is the same either way. The credit Monte Carlo has no such switch and prices its exposure
  grid crisp: UNMEASURED there.
- **The extendable forward's backward pass smooths its payoff kink.** The 32-node Gauss-Hermite rule
  (`Boundary_Quadrature`) carries a one-signed second-order bias: 128 nodes raise the mark by 2e-7
  to 7e-6 of its value on three two-decision documents, the CVA by up to 0.02% and its delta by
  -0.02% to +0.09%. A single rolling decision has no kink.
- **The window-touch registration fixes the delta's sign, not its size.**
  `Boundary_AAD_Window_Touch` (default `Yes`, credit Monte Carlo only) registers a partial-time
  barrier's window touch for the boundary correction. On a monthly grid with six live decisions,
  five seeds read −1.915 registered at 32,768 paths against +1.318 unregistered; all 70 CRN readings
  are negative and the pooled oracle sits 0.5% from the registered delta. The ladder still moves
  13% to 35% over h ≥ 5e-4, so the magnitude is known to about a third. `No` is the unregistered
  estimator.
- **The boundary correction is stable over its bandwidth only at the declared path count.** At
  16,384 and 32,768 paths six seeds resolve no dependence on `Boundary_AAD_Bandwidth` from 0.00125
  to 0.08 (within ±2% of the correction and ±0.5% of the CVA delta); at 2,048 paths the correction
  falls 24% across that range. Its scoping has no public seam a mutation gate could reach.
- **An already-hit at-the-money knock-in carries time value on its expiry row** (2026-10-04).
  `pv_discrete_barrier_option` clamps the remaining expiry to 1e-4 years, so the row reads 0.0997 on
  a spot of 100 where the payoff is 0. S.
- **The American equity option's approximation is to be retired, not patched** (2026-09-16).
  `pv_american_option` (Bjerksund-Stensland) never calls `calc_vol_adjustment`, so a composite or
  quanto American prices as the local asset. UNMEASURED, as is its error against a lattice. The fix
  is a different pricer.
- **The closed-form barrier floors its vol at 5%** (2026-10-11). `pv_barrier_option` (the FX barrier,
  and an equity barrier without `Barrier_Dates`) lifts any lower vol to 5% before pricing and before
  the monitoring shift. UNMEASURED on low-vol pairs.

### The other pricers

- **An American FX option prices as a European** (2026-10-11). `FXOptionDeal` and
  `FXTARFOptionDeal` declare `Option_Style: American` and neither reads it, so any early-exercise
  premium is lost. UNMEASURED; refusing it by name is S.
- **An Asian carries one flat rate, and the double Asian one vol** (2026-10-11). The discrete Asian
  forwards every sample at the carry $\log(F(t,T)/S)/\tau$; the double Asian reads one ATM vol at
  expiry for both strips. UNMEASURED on a sloped curve or a term-structured surface.
- **A cash-settled swaption is priced on the physical annuity** (2026-10-11), without the cash
  annuity's convexity; an amortising or forward-start swaption prices on the vanilla expiry and
  tenor, unrefused. UNMEASURED.

### The engine

- **The FX option has no table for its expiry print** (2026-10-09). The FX option and binary hold
  their forward at the expiry's print until `Delivery_Date`, as the equity options do, but declare
  no `Price_Fixing`, so one expired before the base date refuses by name for a print it cannot
  state. The table and its `observes`: S.
- **A rate leg ignores eight of its conventions** (2026-09-30). `Reset_Type` (`Arrears`, `Advance`),
  `Payment_Timing`, `Payment_Offset`, `Index_Day_Count`, `Index_Offset`, the calendars and
  `First_Coupon_Date` are declared on the swap, cap, floor and swaption and move no leg: the swap
  reads only its payment calendars, the swaption reads `Index_Day_Count` only for its vol tenor.
  UNMEASURED. The fix is reading each in the schedule generator the four share, which the deposit's
  and the equity swap leg's `First_Coupon_Date` also wait on.
- **A fitted structure leg's two-way charge uses its lognormal vega** (2026-09-21). A leg under a
  fitted spot model publishes no FX vol quote sensitivity, so the charge reads the same leg as a
  lognormal (`spread_source` says so): a vega a desk could hedge, but not the quoted price's.
  UNMEASURED, and not measurable from a quote: the fit's own `dV/dq` needs `Quote_Sensitivity`
  through `LogVar2FJModelParameters`, the calibration live in the session.
- **Haircuts are read on one side, and a floating list refuses its settlement** (2026-09-22). The
  engine and the collateral call apply `Haircut_Posted` whichever side holds the asset, and the
  single-asset haircut cancels in the engine's own formula. UNMEASURED on a collateralised exposure.
  The fix (M, with the spine's collateral kit): held collateral credited at one less
  `Haircut_Received`, posted collateral delivered at face over one less `Haircut_Posted`, and a cash
  row with no `Haircut_Posted` (a `KeyError` at compile today) read the same way. A floating
  cashflow list refuses a stated `Settlement_Date` or `Settlement_Amount` until it reads them: M.
- **Rating-dependent credit support is read at its first row** (2026-10-11).
  `utils.CreditSupportList.value` returns the first row's threshold, minimum transfer or independent
  amount whatever the counterparty's rating, in the engine and the spine's collateral call. Exact
  for one row; UNMEASURED otherwise.
- **Declared fields nothing reads** (2026-10-11): `Commodity_Collateral` (netting set),
  `Net_Cashflows` (structure), `Dividend_Timing` (equity swap leg; `Continuous` prices as
  `Terminal`), `Average_FX` (floating energy deal, energy option), `Risk_Premium` (GBM
  term-structure model) and `Distribution_Type` (PCA rate model, always lognormal). Refusing or
  dropping each: S.
- **A factor read between scenario dates is interpolated** (2026-10-04). A fixing off the grid reads
  the factor linearly, 5e-7 of an FX level on a monthly grid; a price-index month in force on no
  scenario date reads between prints, a linker's payment row 1e-6 to 6e-6 high on a quarterly grid.
  Exactness wants the day on the grid.
- **A host curve read depends on its batch position at the last bit** (2026-10-05). Fused blends
  round differently in vector lanes and scalar tails, so a CPU read moves by an ulp with the batch
  width and the thread count (within 4 ulps, bit-identical on a GPU, seen in no end-to-end result).
  A replay across boxes compares for equality, so a home replaying host-priced results needs a
  declared tolerance policy.
- **The record does not declare the files a job names** (2026-10-05). Market data and calendar files
  are named by path, so a replay reads whatever the file holds then, and outside the XVA tab the
  calendar is in no hash (two jobs differing only in a calendar priced 1.1e-3 apart under one result
  id). The fix uses what exists ([the design](architecture.md#market-data-context-job)): the hub
  declares each file under a name and a job names the standing blob: S. Meanwhile the XVA tab
  re-hashes a file only when its size or modified time moves, missing a rewrite inside one tick, and
  a set refused for a file that moved in the queue keeps the refusal if the file is put back.
- **The service reads the market off the book's own page** (2026-10-05). Some forty readers take the
  base currency and date, factors, prices and bootstrapper configuration from the book's explicit
  block, against [the design](architecture.md#market-data-context-job). A book naming a market data
  file therefore answers 500 on its status and curves, refuses survival curves and date rolls on a
  bare key, and writes the whole file's factors into itself when one spot is patched; every request
  but the XVA tab's builds its own context, and a market edit makes the XVA tab re-parse the file.
  M: the readers through a loaded context.
- **A credit Monte Carlo writes into the config it runs on** (2026-10-06). `run_cmc` re-knots the
  counterparty's survival curve for `CDS_Tenors` and writes flat `.FUNDING` and `.COLLATERAL`
  curves into the price factors over any of that name the market file carries: a later set on a
  kept context read a fifth of a fresh context's funding. The XVA tab drops its context after such a
  run. The write belongs on the run's own copy: S.
- **A SIMM run re-solves each benchmark move in every worker** (2026-10-07). The `SIMM` calculation
  re-solves the curve or surface each move feeds, per trade and per worker, so four workers give
  1.3x over one on a 19-trade set; solving each move once is estimated at 2.5x. The dependency walk
  also hands each trade its FX chains, about seven repricings per CRIF row. M.
- **A SIMM vega vertex on a Skew surface moves its base** (2026-10-07). A vertex bump inserts a
  pillar where the surface lacks one, read off its own interpolation. Explicit and Malz surfaces
  price unchanged, but no smile at an inserted pillar equals a Skew surface's expiry blend (14.5 vol
  points apart deep in one wing), so two of five structures move, by 4.8e-4 and 1.5e-2 of the
  trade. The rows agree with each other, not with the book's base. S-M: the tent through the blend.
- **A one-point vol move carries cross-vertex convexity** (2026-10-07). On the autocalls the twelve
  vertex rows differ from one parallel point by -0.8% to +13% (0.1% to 1.2% on FX), stable across
  seeds and path counts, a tenth of a point cutting it tenfold. It is the regulation's definition,
  so stated, not fixed. Rate vega, credit, commodity and inflation sensitivities refuse by name.
- **The inner Monte Carlo is launch-bound** (2026-10-07). A one-step-survival step is about fifty
  elementwise kernels on `[batch, 2 x inner]`; a value batch kept the device 45% busy on the
  autocall and 64% to 79% elsewhere, measured before the per-batch host syncs were removed and not
  re-measured since. Fusing the step is the lever past about 16k inner paths (`torch.compile` needs
  Triton). M.
- **The LogVar2FJ walk's cumsum is slow on its 21-step axis** (2026-10-07). It takes about a fifth
  of a value batch; a matmul by a 21 x 21 upper-triangular ones matrix is about thirty times faster
  and equal to 2.3e-7 relative, so the LogVar2FJ documents would re-bank (S). The inverse-Gaussian
  root runs in double, about 45% of the batch on a GPU with 1/64-rate double (M). Under a gradient
  the walk holds 79% of peak memory in its checkpoint segments.
- **A dense reporting grid holds a block of rows at once** (2026-10-07). The streamed strip holds
  every time-grid row of a block together, so memory grows with the rows per fixing interval and a
  daily grid can exceed a device that holds a weekly one. One row per block past about twelve rows
  per interval: S.
- **A legacy trade closed before migration prices short** (2026-09-26). A book-file node no fill
  booked prices as written, one unit; closing it through the verbs files a fill of -1, and the
  compile writes the node at that net, the mirror trade. The designed migration, a fill of one under
  every legacy node, makes the close net to nothing; until then such a node sits outside the desk's
  P&L.
- **A consolidated risk read refits its quote blocks, and one refusal drops it to factors**
  (2026-09-24). `/book/risk` refits, with `Quote_Sensitivity` on, every block the book's factors
  stand on, cached on the deals and the market, so a spot-model block pays its warm polish on every
  uncached read; one block that cannot carry a quote derivative (an FX forward outright, a polish
  above `Stationarity_Tol`) returns the whole book in factor space, named. UNMEASURED on a desk book.
  The fix is the connected bootstrap cached by the market's hash, and a per-block retry with the
  refusing block off prices.
- **Every read of the record scans the log** (2026-09-23). `SpineLog.__init__` streams every segment
  to build the head, the tag index and the per-LSN offsets, linear in the log, once per read (twice
  on a strip's first paint; once per submitted job for admission, under a fiftieth of the cheapest
  job). The fix is a checkpointed index beside `log/`, derived and verified like a seed, so an open
  scans only the tail and the doctored-middle-line refusal moves to `verify_home`: M.
- **The oracle judges old frames by today's verb map** (2026-09-29). A type moved between verbs (a
  settlement to `settle`) makes every frame filed under the old verb read as outside its seat, while
  the desk runs unchanged. No home outside the gates predates a move. The fix is the map in force
  declared on the record at an LSN, read like a policy, one read in `verb_for`.
- **A forged hub approval passes the oracle under a mixed tiers policy** (2026-09-29). Under a policy
  with both an automatic and a four-eyes tier, a copy forging an approval of a ticket only the
  four-eyes tier would admit is caught only by copies agreeing with the hub. Recording the tier on
  the approval closes it: M. Signed history ends at the last checkpoint either way.
- **Two reads skip the queue's admission** (2026-10-11). `GET /book/risk` and
  `GET /book/pnl?explain=` run valuations on the request thread for any caller, seated or not, and
  `/book/bloomberg` and `/book/securities/verify` queue under the hub's seat whoever asks. An
  admission on each: S.
- **A checkpoint key rotation is trusted on chain integrity alone** (2026-10-11). `verify_home`
  accepts any `checkpoint_verifying_key` declaration on an intact chain without asking that the key
  then in force signed it, and the signing key sits in the hub's `keys/`. Against the hub's own
  operator, tamper-evidence holds only with a genesis hash and key obtained out of band, or an
  agreeing replica ([Spine](spine.md#trust-limits)). A design call.
- **The replay tuple pins neither build nor device** (2026-10-11). `engine_version` is a hand-set
  string, no commit, torch, CUDA or device is recorded, and `Deterministic_Kernels` defaults to
  `No`, so bit-for-bit replay holds only on one build, stack and device. UNMEASURED across devices
  beyond the host-curve row.
- **A settlement names no agreement** (2026-09-28). A payment the diary cannot determine is shared
  across its instrument's positions by holdings: exact when every position's settlement is filed,
  smeared when one is filed alone, named and unshared over positions netting to zero. One filed
  against a payment the diary has dropped is matched within the business week before its value
  date; past that, its day and a window reaching back over it stop summing. UNMEASURED. The fix: the
  transition names its agreement, required where the instrument sits under more than one; about
  twenty lines over seven files.
- **Two live deals under one reference get no diary key** (2026-09-28). `POST /book/deals` accepts a
  second live deal under an existing reference, the diary keys neither, and the close waits on them
  for good. The key needs the deal's path beside its reference: M.
- **A floating coupon settled for another amount books the difference a day late** (2026-09-28).
  Its known-rate table is not filled from the record, so the coupon is never determined: a different
  settled amount lands the next day, and a swap bought ex-coupon on its coupon day carries the
  engine's coupon for that day. Each month sums right. The fix is fixings as market data
  ([below](#designed-not-built)): M.
- **The market, the date and the file move under no seat** (2026-09-30). `POST /book/market`,
  `/book/date`, `/book/configure`, `/book/curve`, `/book/bloomberg`, `/book/securities`,
  `/calculations`, `/book/setup` (admitted under `validate`, yet it re-solves the market) and a
  delete take no `actor`, so under a capabilities document any client can move the board a close is
  declared over, roll the date or drop a deal. The tick is the deployment's poll path and
  `delete_deal` takes no seat by design, so this is a design call first. M-L: an `actor` and an
  admission on each (`mark` for the market and the date, `book` for the file).
- **Six types announce fixings under no index** (2026-09-30). A fixing row is answered by a print
  under the index its type's `observes` names; a row naming none blocks `close_check` for ever, so
  its day gets no close, marks or P&L. The double Asian, the equity swap leg and swaplet list and the
  two energy deals (and a composite equity Asian) read a table the fill cannot write yet (M), the
  inflation list M-L; a swaption announces no expiry of its own, a design call. The rate deals name
  their index, but a blank one still announces under none, and they refuse a past reset their table
  does not state until fixings are market data ([below](#designed-not-built)). `UNOBSERVED` in
  `tests/test_diary.py` lists the open types.
- **The pricer branch census names 38 arcs no test executes** (2026-10-11). The LogVar2FJ
  discrete-barrier arcs are worth a fixture; the rest are diagnostics, completeness elses and new
  code. `gates/pricer_branch_census.py` lists them.
- **Three checks are ungated** (2026-08-21): the vol term structure reaching the monitoring, the
  payoff's forward against the vol surface's, and rate units
  ([Conventions](conventions.md#what-holds-today-and-what-the-purge-left-open)).

## Decisions waiting on the desk

Nothing here is blocked on work. Commit messages and the model pages cite these numbers, so closed
decisions (2, 3, 4, 5, 10, 11, 13, 15) keep theirs and are not listed.

1. **The per-fixing smile read**: sticky-forward moneyness or the deal's declared moneyness. Both are
   defensible; the choice lands as a switch, with the six removed gates rebuilt.
6. **The Hull-White seed's worst benchmark**: the honesty reprice reads −6.25% against the retired
   seed's −4.64%, while the RMS miss fell from 2.71% to 2.39% and the count outside 3% from ten to
   three. One order statistic, anti-correlated with the fit; a desk's eye wanted.
7. **Exposure versus credit-valuation measure.** CVA wants the risk-neutral, market-calibrated
   outer; PFE a real-world, historically estimated one. One run reports both off one outer measure,
   so a book wanting each in its own runs twice under two model configurations.
8. **Two single-caller wrappers around the implied-correlation read**, kept against the rule of no
   abstraction ahead of a second caller until a third correlation pair appears.
9. **Flagged, not authorised**: the hedge runtime's free functions over the bundle (two clusters, a
   duplicated utility table) and the deal structure's recursions, the shape
   [Conventions](conventions.md#registries-not-functions) calls a class waiting to happen.
12. **The correlation as a leaf.** A correlation mints no leaf, so a quanto's correlation delta is a
    CRN bump of 0.025, flat to 0.003% between half-widths on an index autocall. The leaf is three
    edits with a tree-wide blast radius: every document carrying a correlation gains a first-order
    row and a Hessian row and column. The bump is its oracle.
14. **`Prices` in process: warning or refusal.** Mandatory on the multiprocessing path; a refusal
    means editing about twenty test and gate sites that write an empty family entry.
16. **The nominal leverage weight.** `Leverage_Prior_Weight` 0.02 counts as three to four quote rows
    on the ladders measured, not one, since it assumes a quote weight and a standard error no ladder
    states; a block's `Leverage_Prior_SE` and `Leverage_Product_Prior_SE` supersede it. It matters:
    a history's leverage of −0.20 ± 0.09 and product −1.05 ± 0.67 move a desk index autocall's mark
    by 11.7% through those rows, so the leverage is the estimator's most consequential number and
    its standard error a sampling error, not a desk's spread.
17. **Jacobian column scaling in the Hull-White solve** (2026-09-14). The LogVar2FJ fit scales each
    step by its Jacobian column; the Hull-White chain does not (its backward forms the scaled matrix
    itself), so nothing is wrong. Switching moves every Hull-White fit slightly; read iterations,
    the stationarity norm and the marks on the four-quote fixture and a desk ladder first.
18. **Netting a package's legs before the two-way charge** (2026-09-21). Each leg pays its own
    spread, so a collar is charged as two tickets. Netting per pillar cuts the gate collar's edge by
    85% (the ATM row 95%, the butterfly 88%, the risk reversal not at all).
19. **One `settle` verb for the back office** (2026-09-30). Settlements, confirmations and collateral
    share it, so each worklist shows the others' items and confirmations can pay or post collateral;
    separation of duties wants narrower verbs.
20. **The recompute segment length.** The LogVar2FJ walk checkpoints every 21 steps, trading memory
    for a second forward pass; the mixer's per-draw checkpoint raised peak memory by about a fifth.
    The residual's granularity is the dial if that headroom is wanted.

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
  the six-types row above retires. Every pricer reading a past fixing changes (barrier, TARF,
  accumulator, autocall, the Asians, the swap family's known rates, inflation), the banked numbers
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
  `Credit_Monte_Carlo` on the same two-run seam, with a ratio-solve primitive for participating
  forwards.
- **Service layer, what remains**: progress over SSE, a cost estimate from the real grid, per-seat
  budget caps, and screens for the dependency walk and the set-up, which only the MCP binding serves
  today. The Securities screen stays read-only: it names a knot quoted off a drifted or unmapped
  ticker, fixed by a curve row or a seed entry elsewhere.
- **Excel end-state**: `RF_*_PORTFOLIO` moved to `GET /schema`, after which nothing in
  `excel_integration/` imports the engine.
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

## Hull-White 2F, decided {#model-punchlist}

`Objective: 'Analytic'` (Schrager-Pelsser) is the default because it is accurate enough, stationary,
deterministic and differentiable in the quotes. On the identified fixture it sits inside one Monte
Carlo evaluation's noise at 22 of 25 benchmarks, its annuity-freezing bias (−0.13 to +2.17 bp) below
the Monte Carlo's own numeraire bias (0.6 to 3.0 bp). Its residual is the vol difference itself, so
`‖J'r‖` at θ\* reaches 8.63e-7 against `Stationarity_Tol`'s 1e-3, where the Monte Carlo's squared
residual stops at 3.16e2. It simulates nothing, so two solves at one seed agree to the bit, and its
quote side exists ([the analytic quote side](quote_sensitivities.md#the-analytic-quote-side)).
`Monte_Carlo` remains the oracle and reprices each analytic θ\* once. `tests/test_hw2f_analytic.py`
holds the α→0 series branches, the declared seed pair and the domestic-measure correction.

**Warm start.** A block whose `HullWhite2FactorModelParameters` factor already exists skips the
basin search and runs the least squares from it (6 objective evaluations against 421 cold on the
four-quote block). It adds no re-marking event.

**Re-marking events.** A desk holding an older θ\* re-baselines or re-solves; carrying it forward is
neither.

- A foreign-curve θ\* solved before the domestic-measure fix re-solves to a different θ\*.
- A θ\* solved before 2026-09-02 re-marks on the seed and premium-clock change.
- A θ\* solved before 2026-09-17 re-marks on the basin search's generator (numpy's default in place
  of the legacy one, same seed): on the four-quote document the mean reversions move 14% and 8%, the
  correlation from −0.04 to −0.16 and the 1Y×1Y quote delta 4%, along directions four quotes do not
  identify.

`Quanto_FX_Correlation_1/2` and every locus recorded downstream of a fit move with θ\*. The Monte
Carlo's numeraire bias comes from the curve's tenor grid, not discretisation: adding 1D, 1M, 3M and
6M nodes collapses it from −1.6e-2 to −1.1e-3.

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
- `derivus_jupyter.py` (tracked, not in the wheel, superseded by the web UI) raises on three field
  names outside its write allowlist and on multi-column Tables outside a four-name allowlist, and
  fourteen output-shaped descriptors have no widget. Retire or repair is a desk call.
- `pv_float_cashflow_list` decides a coupon's resets need folding by comparing their count with the
  cashflows', a shape; an explicit mark on the compiled cashflows would replace it.

## What this list is for

Most rows were found auditing work that already passed its tests. The usual cause was a gate
exercising one point of a parameter (a bought deal only, the default monitoring frequency, one
netting set); the next, a unification that left siblings outside its seam
([Conventions](conventions.md#unification-siblings)). When picking up a row, vary the parameter the
defect lives in, and see the mutant die before trusting the test.
