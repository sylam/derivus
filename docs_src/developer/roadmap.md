# Roadmap

The live status of the library, written for a model-validation reader: what is known to be wrong or
limited and how big it is, what is waiting on a desk decision, and what is designed but not built.
What was built, and every number it was measured to, is in the commit messages — `git log` reads
as the ledger, one landing per commit — and the long-form record that stood on this page until
2026-09-10 is in the repository's history before that date. The model pages carry the numbers a validator prices
against: [Market Prices](market_prices.md#logvar2fj) for the LogVar2FJ calibration,
[Structures](structures.md#model-worth) for what the model is worth on a book and how the xVA
outer is chosen, [Calc Lifecycle](calc_lifecycle.md) for the engine.

## Known limitations, open

Grouped by where a validator meets them. Every row carries its measured size or says it is
unmeasured — a limitation without a number is absolution, not documentation
([Conventions](conventions.md#unification-siblings)). Dates are when the row was found.

### The xVA outer and the correlation

- **A calendar bucket knot inside one scenario interval gives the second piece a private mixer**
  (2026-09-11). The outer process takes one framework normal per scenario interval for its mixer,
  so where a bucket knot cuts an interval's clock into two residual draws, the second draw's mixer
  comes from the process's own stream rather than the framework's, which the process names at
  INFO. Every factor in the book carries one bucket, so nothing reaches it today. Beside it, the
  normal distribution function of the mixer normal saturates in double past 8.3 standard
  deviations, where the inverse-Gaussian root answers the top of its bracket: one draw in 1e16,
  named rather than guarded.
- **A quanto correlation can be written in two bases, and nothing checks which.** The parameter
  writer emits a quanto correlation as the correlation between the FX Brownian and each rate
  factor's own, while the correlation section's rows are the independent normals the Cholesky
  consumes, a different basis related by the two-factor rotation. Copying the writer's second
  number into the section gives a world whose rate covariance disagrees with its declaration,
  silently - 0.5 copied into the second factor's row realises 0.58. One matrix on the driving
  innovations is the source: the drift reads the section's rows through the loadings (M) and the
  writer stops emitting a second basis (L). Found beside it, UNMEASURED: a calibrated market can
  carry the quanto number on its Hull-White factors with no row in the section at all, so the
  simulated FX ignores the rates the drift reads.

### The autocall, TARF and barrier pricers

- **The autocall's observation arm starts a lagged block's walk at the last fixing.** A declared
  table pairing a coupon with a fixing on or before the previous coupon is refused by name; what
  remains is the prefix, where a block whose fixings lag its coupon dates walks the next coupon
  from the fixing's print rather than the row's own spot. M, the fixing schedule being the deal's
  to author.
- **The collateralised autocall's CVA delta is the boundary estimator's own variance.** Under a
  zero-threshold credit-support annex the boundary correction supplies two and a half times the
  pathwise term and scatters with the path count: 51% short at 256 outer paths, 18% over at 1,024,
  10% at 2,048; a lognormal autocall under the same annex reads 12% short at 2,048. The path
  count, not the deal, is what the number depends on. Beside it, the autocall's floating leg and
  its terminal put register no settled cash, so the reported cashflows carry the coupons alone.
- **The target redemption forward's target pin is a kink the crisp estimator is blind to.** The
  pin fires on 41% to 60% of paths. Under the default (`Branch_And_Weight`) a common-random-number
  ladder of the switch's own value surface is flat to 0.05% and lands on the reported delta to
  0.0002%; one `'No'` away the same ladder does not converge, 0.24% to 1.41% flat and 0.14% to
  0.63% off, and the delta itself is the same number under both, the fired branch paying a
  constant both estimators differentiate through the same analytic probability. What stays open is
  the exposure grid, which a credit valuation always prices crisp, declaring no such field: under
  a base valuation there is no boundary correction to be uncorrected, one row resolving no fixing.
- **The extendable forward's rolling backward pass carries a smoothing bias over the payoff's
  kink** from its 32-node Gauss-Hermite rule, second order: at 128 nodes on the same seed the mark
  rises 2e-7 to 7e-6 of its value on three documents with two rolling decisions, one-signed as a
  mislocated exercise boundary is, the CVA by up to 0.02% and the CVA delta by -0.02% to +0.09%;
  64 nodes sits with 256, and a single rolling decision has no kink to cross. `Boundary_Quadrature`
  is the dial.
- **The window-touch registration's ladder is not flat, and it is the default on its sign**: on a
  grid carrying a row a month — seven inside the window, six of them live — five seeds read a
  registered −2.136 at 8,192 paths and −1.915 at 32,768 against an unregistered +1.318, every one
  of the 70 CRN readings negative and the pooled oracle 0.5% from the registered delta against
  168% from the unregistered one. The ladder is still not FLAT, 13% to 35% over h ≥ 5e-4, so the
  registered delta's magnitude is known to about a third where the unregistered sign is known to
  be wrong; `Boundary_AAD_Window_Touch: No` is the unregistered estimator, one value away. The
  switch is the credit Monte Carlo's alone: a base valuation's single date never reaches the
  observed-spot branch the latch lives in, so it declares no such field.
- **The boundary correction holds still over its bandwidth only at the declared path count.** At
  16,384 and at 32,768 paths six seeds paired against the declared 0.01 resolve no dependence from
  0.00125 to 0.08 - no rung past 1.7 standard errors, ±2% of the correction and ±0.5% of the CVA
  delta over 0.005 to 0.08 - where three seeds' spread is noise at either count; at 2,048 paths
  the correction falls monotonically by 24%. The correction's scoping has no public seam a
  mutation gate could reach.
- **An already-hit knock-in struck at the money reads a day's time value on its expiry row**
  (2026-10-04): 0.0997 on a spot of 100 where the payoff is 0, the remaining expiry being clamped
  off zero. S.
- **The American option's approximation is to be retired, not patched** (2026-09-16).
  `pv_american_option`, which an `EquityOptionDeal` carrying `Option_Style: American` reaches,
  never calls `calc_vol_adjustment`, so a composite or quanto American prices as the local asset
  whatever the payoff currency says. The owner's ruling: leave it as it is and replace the
  approximation with a different pricer rather than adjust this one.

### The engine

- **A rate leg stated by its terms reads eight of its conventions not at all** (2026-09-30).
  `Reset_Type` in `Arrears` or `Advance`, `Payment_Timing`, `Payment_Offset`, `Index_Day_Count`,
  `Index_Offset`, the calendars and `First_Coupon_Date` are declared on the swap, the cap, the
  floor and the swaption and read by none of them - the swap reads its payment calendars alone -
  so each leaves the mark where it stands, 6,016.17 on the trial cap, where the list form states
  its dates. UNMEASURED beyond that they move nothing; the schedule generator all four share is
  where each is read, and the deposit's and the equity swap leg's `First_Coupon_Date` wait on it.
- **A model-priced leg's two-way charge is read off its LOGNORMAL vega, not the fit's own**
  (2026-09-21). A leg walking a fitted law publishes no FX vol quote sensitivity at all, so the
  charge comes from the same leg at the same terms read as a lognormal — a real vega, and the one a
  desk would hedge in, but not the sensitivity of the price that was quoted. UNMEASURED, and it
  cannot be measured from a quote: the fit's own `dV/dq` needs `Quote_Sensitivity` published through
  `LogVar2FJModelParameters`, which means the calibration live in the same session rather than a
  factor read off the book. Until then the substitution is named on every leg it is made for
  (`spread_source`).
- **A haircut is read on one side, and a floating list's settlement is refused rather than read**
  (2026-09-22). The engine and the collateral call take `Haircut_Posted` whichever side holds the
  asset, and the engine's single-asset haircut cancels against its own documented formula, which
  credits the unhaircut value: collateral HELD is credited to exposure at one less
  `Haircut_Received` and collateral POSTED delivered at face over one less `Haircut_Posted`,
  standing at full value as the bank's asset - the model change M, the spine's kit with it, and a
  cash row stating no `Haircut_Posted`, which fails the set's compile on a `KeyError` today, read
  under the same ruling. A floating cashflow list's `Settlement_Date` and `Settlement_Amount` are
  refused by name where stated until the list reads them as the fixed list does in four places: M.
- **A factor read on a day the scenario grid does not hold is interpolated between its
  neighbours** (2026-10-04). A fixing between two scenario dates reads the factor linearly across
  them, 5e-7 of an FX level on a monthly grid, and a simulated price index's month whose print is
  in force on no scenario date reads on the index's clock between the prints either side, a
  linker's payment row 1e-6 to 6e-6 high on a quarterly grid. Exactness wants the day on the
  scenario grid.
- **A curve read on the host depends on its place in the batch at the last bit** (2026-10-05).
  The fused blends round differently in a vector lane and a scalar tail, so a host read moves by
  an ulp with the batch width and, at some thread counts, the thread count - seen in no
  end-to-end result, a batched curve within 4 ulps of its looped columns, and to the bit on the
  card. A replay pinned from another box is compared for equality unless a tolerance policy
  stands, so a home replaying host-priced results across boxes wants one declared.
- **Nothing declares the files a job names in the record yet** (2026-10-05). A job
  names each by its path, the evidence the record stores does too, so a replay reads whatever the
  file holds then; and outside the XVA tab the calendar is in no hash at all - two jobs differing
  only in a calendar's content priced 1.1e-3 apart under one result id. Nothing is missing to do it
  ([the design](architecture.md#market-data-context-job)): a market declaration files any bytes
  under a name, the store keeps them as a plain file at their address, and the `markets` fold
  resolves the latest of a name - so the hub declares each file and a job composed at a point in
  the record names the standing blob's path. Nothing cites a version. Until then the XVA tab reads
  a path's digest by its size and modified time, so a rewrite inside one timer tick is missed, and
  a set refused for a file that moved in the queue keeps that refusal under its id if the file is
  put back byte for byte. S.
- **The service reads and writes the market on the book's own page** (2026-10-05). It was built on
  a book carrying its whole market inline, against
  [the design](architecture.md#market-data-context-job): 43 readers take the base currency, the
  base date, price factors, market prices and bootstrapper configuration straight off the book's
  explicit block, so a book that names a market data file answers 500 on its status and curves
  with nothing explicit, refuses a survival curve and a date roll on a bare key, and writes the
  whole file's factors into the book when one spot is patched; and every request but the XVA
  tab's loads into a context of its own. The XVA tab is moved - its worker keeps one context, a
  named file parsed once, a ten-set recalc 12.2 s to 2.0 s - and its rows' `plan_hash` and
  `values_hash` hold the set document's hash and the two files', where every other run's hold the
  engine's; a market edit writes `Market Prices` into the book, so the first recalc after a tick
  parses the file again. M: the readers through a loaded context.
- **A credit Monte Carlo writes into the config it runs on** (2026-10-06). `Context.run_cmc`
  re-knots the counterparty's survival curve where `CDS_Tenors` is set - idempotent, to the bit
  across a moved base date and changed tenors - and writes a collateral valuation adjustment's flat
  `.FUNDING` and `.COLLATERAL` curves into the price factors, over a curve of that name a market
  data file carries: on a kept context a later set funding off the file's curve read a fifth of
  what a fresh context does. The XVA tab drops its context after such a run, a parse per set; the
  write belongs on the run's own copy. S.
- **A legacy trade closed before it is migrated prices short** (2026-09-26). A node the file carries
  that no fill ever booked prices as written, one unit; a close-out of it booked through the verbs
  files a fill of -1, and the compile writes the node at that net, the mirror, where nothing should
  stand - one probe read -85.38 against 0, where the tree before read +170.76. The migration the
  design names, a fill of one under every legacy node, is what makes the close net to nothing -
  and until it runs, a node no fill booked is outside the desk's P&L, which reads its positions
  off the record.
- **A consolidated risk read refits the quote blocks its book prices, and one refusal reads all of
  it on factors** (2026-09-24). `/book/risk` refits, with `Quote_Sensitivity` on, every block
  writing a factor the book reads and every block those stand on, so a spot-model block pays its
  warm polish on every uncached read; and a bootstrap raises at its first refusal, so one block that
  cannot carry a quote derivative - an FX forward outright, a polish stopping above
  `Stationarity_Tol` - returns the whole book's risk in factor space with the refusal named.
  UNMEASURED on a desk book; a two-trade FX book reads in 5.1 s, its three blocks' refit
  included. A calibration is a function of the market snapshot, not the book, so the remedy for
  the first is the connected bootstrap cached by the market's hash until a tick; for the second a
  retry with the refusing block alone off prices - per block, an outright being a row of the curve
  family rather than a family of its own.
- **Opening a log scans it, and every read of the record opens one** (2026-09-23). The seek closed
  the READING half of the desk's beat - a page of ten at the head of a 2,005-event log is 0.20 ms
  where it was 7.50, and one two-second beat is 28 ms where it was 44 - and what is left is
  `SpineLog.__init__`, which streams every segment to build the head, the tag index and the offset
  per LSN: 11.3 ms at 2,005 events, paid once per read because a reader never claims the home and
  so cannot hold a handle across one. A strip's FIRST paint pays it twice over, since a page with
  no `?since=` folds from genesis by design. Queue admission rides the same open once per submitted
  job (the capability fold, 1.2 ms at 21 events and 20.5 ms at 1,994), which is still under a
  fiftieth of the cheapest job it gates, so nothing caches it. The remedy is a checkpointed index
  beside `log/` - derivable, disposable and verified the way a seed is; an append invalidates
  nothing, bytes only ever being appended and a new segment only the next number - so an open
  checks the sizes and the head line and scans the tail alone, and the doctored-middle-line
  refusal moves to `verify_home`, where the chain is checked on demand. M.
- **The oracle judges history by the verb map deployed now** (2026-09-29). A type moved between
  verbs - a settlement to `settle`, a run's replay tuple to the writer's own voice - reads every
  frame filed under the old verb as outside its seat, so a record written before the move is named
  by the oracle while the desk on it runs unchanged. No home outside the gates predates it. The
  remedy is the map in force declared on the record at an LSN, read the way a policy is, so a frame
  is judged by the grammar it was filed under: one reserved policy and one read in `verb_for`.
- **A copy forging the hub's approval under a policy declaring both an automatic and a four-eyes
  tier passes the oracle alone** (2026-09-29). The oracle holds a writer-filed approval to a tiers
  policy in force with an automatic tier and a fill carrying its ticket after it; under a policy
  with both kinds, a forged approval of a ticket only the four-eyes tier would admit is caught only
  by copies agreeing with the hub. Recording the tier on the approval closes it: M. Signed history
  ends at the last checkpoint either way.
- **A settlement names no agreement, so the P&L shares what it moved by holdings** (2026-09-28). A
  payment the diary cannot determine falls to the positions of its instrument by what each held
  when it fell due, which is exact while every position's settlement is filed and spreads one filed
  alone across all of them; over positions netting to nothing it cannot be shared at all and is
  named. One filed against a payment the book's diary has dropped is looked for in the marks of the
  business week before its value date; beyond it the day it lands in names it while a window
  reaching back over the payment places it, so those days no longer sum to that window.
  UNMEASURED; settlements are per agreement, one agreement's filing never discharging another's,
  so the transition names its agreement, required where the instrument sits under more than one:
  about twenty lines over the seven files a movement passes through.
- **Two live terms under one reference get no diary key** (2026-09-28). `POST /book/deals` books a
  second deal under a reference the book already carries live, and the diary gives no key to a
  reference two live terms share, so both rows can be settled by nothing and the close waits on
  them for good. The row's key wants the deal's own path beside its reference, which the
  calculation that names rows does not carry: M.
- **A floating coupon settled for another amount than the engine's books the difference the next
  day** (2026-09-28). A determined payment settled for another amount is a break the P&L names; a
  floating coupon is never determined - its known-rate table is not filled from the record at
  compile - so one settled on time for another amount books the difference the next day, and a
  floating-coupon swap bought on its coupon day, ex-coupon, carries the engine's coupon in that
  day's value and gives it back the next. Each month sums right. Closing it is the known-rate
  tables filled from the record as an observation table is: M.
- **The market, the book's date and the file move under no seat** (2026-09-30). The tick and a
  values patch (`POST /book/market`), `/book/date`, the bootstrapping dials (`/book/configure`),
  `/book/curve`, `/book/bloomberg`, `/book/securities`, a saved calculation (`/calculations`),
  `/book/setup` - admitted under `validate`, it installs and re-solves the market - and a delete
  take no `actor` or file nothing, so under a capabilities document any client of the box moves
  the board a close is then declared over, rolls the book to another day or drops a deal from the
  file, which only `/book/reconcile` then names. The tick is the deployment's own poll path and
  `delete_deal` says it takes no seat, so this is a design call before it is a patch. Size M-L: an
  `actor` on each and an admission - `mark` for the market and the date, `book` for the file - with
  the metronome's grants and the web's calls counted.
- **Six types announce fixings that name no index** (2026-09-30). A fixing row is answered by a
  print filed under the index it names, which a type declares as `observes`; a row naming none is
  answered by nothing, so `close_check` waits on it for ever and the day it falls on gets no close,
  no marks and no P&L. The swap, the floating list, the cap and the floor, the FRA and the deposit
  declare theirs, and read a reset the record has printed off the curve until their known-rate
  tables are filled from the record at compile as an observation table is - a day's move on one
  reset; a blank index field, which the compile resolves to the currency's curve, still announces
  its fixings under none. The double Asian, the equity swap leg and swaplet list, the two energy
  deals and the composite equity Asian read fixings off a table the fill cannot write yet (M), the
  inflation list M-L - a month its base date has not printed reading observed at its forward
  meanwhile - and a swaption announces no expiry of its own, its legs reading as a live swap's - a
  design call. The census in `tests/test_diary.py` names every type still open.
- **The pricer branch census names 48 arcs no test executes** (2026-10-04; 59 on 2026-09-02),
  the LogVar2FJ arcs on the discrete barrier and the TARF's pathwise arm the ones a fixture would
  reach; the rest are diagnostics, completeness elses and code added since.
- **Ungated since the 2026-08-21 purge**: three modules named on
  [Conventions](conventions.md#what-holds-today-and-what-the-purge-left-open) - the vol term
  structure reaching the monitoring, the payoff's forward against the vol surface's, and the
  rate-units pass.

## Decisions waiting on the desk

Nothing here is blocked on work. Numbers are stable — commit messages and the model pages cite
them — so closed decisions (2, 3, 4, 5, 10, 11, 13, 15) keep their numbers and are not listed.

1. **The per-fixing smile read.** Sticky-forward moneyness or the deal's declared moneyness; both
   defensible, one can be the pricer's own quote. A switch, not a revert, with the six removed gates
   rebuilt.
6. **The Hull-White seed's worst benchmark**: the honesty reprice reads −6.25% against the retired
   seed's −4.64% while the root-mean-square miss improved from 2.71% to 2.39% and the count outside
   3% fell from ten to three. One order statistic, anti-correlated with the fit; a desk's eye
   wanted.
7. **Exposure versus credit-valuation measure policy.** A credit valuation is a risk-neutral
   expectation wanting the market-calibrated outer; a potential future exposure is a real-world
   quantile wanting a historically estimated one, the pricing kit staying market-implied. One run
   reports both off one outer measure, so a book wanting each in its own measure runs twice under
   two model configurations.
8. **Two single-caller wrappers around the implied-correlation read**, held against the rule of no
   abstraction ahead of a second caller until a third correlation pair appears.
9. **Flagged, not authorised**: the hedge runtime's free functions over the bundle, two clusters
   with a duplicated utility table, and the deal structure's recursions — the shape
   [Conventions](conventions.md) calls a class waiting to happen.
12. **The correlation as a leaf.** A correlation mints no leaf today, so a quanto's correlation
    delta is reported as a common-random-number bump of 0.025: on an index autocall it reads
    flat to 0.003% between half-widths, the value linear in ρ. The leaf is three edits with
    a tree-wide blast radius, every document carrying a correlation gaining a first-order row and a
    Hessian row and column, lognormal ones included. The bump is the leaf's oracle.
14. **`Prices` in process: warning or refusal.** Mandatory on the multiprocessing path; a refusal
    costs an edit at about twenty test and gate sites that write an empty family entry.
16. **The nominal leverage weight.** `Leverage_Prior_Weight` 0.02 reads three to four quote rows on
    these ladders, not one, because it assumes a 0.2 quote weight and a 0.1 standard error no
    ladder states; a block's own `Leverage_Prior_SE` and `Leverage_Product_Prior_SE` supersede it
    where declared. At a fitted vol-of-vol of 5 it reads several quote rows, and at the
    class-default tier it is what the Nikkei cannot carry. A history's leverage of −0.20 ± 0.09
    and product −1.05 ± 0.67 move an NKY autocall's mark by 11.7% through those rows: the
    estimator's leverage is the single most consequential number it produces, and its standard
    error is a sampling error, not a desk's spread.
17. **Whether the Hull-White solve should scale its steps by the Jacobian's columns** (2026-09-14).
    The LogVar2FJ fit runs its least-squares stage with each parameter's step scaled by the size
    of its own Jacobian column, the better-conditioned solve; the Hull-White chain does not, and
    its backward forms that scaled matrix for itself, so nothing is wrong today. Switching the
    chain on to the same scaling changes where the solve stops, so every Hull-White fit in every
    book moves by a small amount. A solver-tuning decision, not a defect, wanting its own reading
    before it is taken: iterations, the stationarity norm at the stopping point, and what the
    marks do, on the four-quote fixture and one desk ladder.
18. **Netting the legs of one package before the two-way charge** (2026-09-21). Each leg pays its
    own spread, so a collar's bought put and sold call are charged as two tickets rather than the
    one position the desk deals; netting per pillar first takes the gate collar's edge from
    1,985.57 to 297.25, an 85% cut - the ATM row from 1,441.26 to 71.47, the butterfly from 362.87
    to 44.34, the risk reversal unmoved. The next dial on this charge.
19. **One `settle` verb for the whole back office** (2026-09-30). Settlements, confirmations and
    collateral each hold `settle`, so each worklist lists the others' payments, clips and calls,
    and confirmations may pay or post collateral; separation of duties inside the back office
    wants a narrower verb or none.
20. **The recompute segment length**, measured for the walk alone: the walk is checkpointed every
    21 internal steps to trade memory for a second forward pass, and the mixer's per-draw
    checkpoint beside it took peak memory from 8,392 to 9,990 MiB at 2,048 scenarios by 2,048
    paths; the residual's granularity is the dial if that headroom is wanted.

## Designed, not built

- **The density recursion** — one FFT convolution per monitored date against the block Gaussian,
  as an alternative inner estimator. The daily walk's tape at 2,048 × 2,048 × 509 does not fit a
  24 GiB card in either direction (the draws alone are 3 × 7.95 GiB), which is what the per-block
  checkpointing exists for; no coarser chain is licensed (a 5-day gap read 3.4 SE on the coupon
  leg).
- **Barrier state as a fold over fixings, what remains.** `Barrier_Dates` and `Price_Fixing` are
  filled from the record at compile; still terms-only are continuous monitoring, which reads daily
  `(low, high)` bars - `utils.bars_touched` is the predicate and is gated, and nothing files a bar
  yet: the prints fact below carries low and high beside the close - so the one-touch and
  partial-time barriers price from terms alone. A bar has a span and a monitoring window has
  edges, open at the booking's effective time and closed at the expiry cut: a bar inside the
  window decides, touch being weak and a bar bracketing every print of its span; a bar that does
  not touch decides "no" even straddling an edge; a bar straddling an edge that does touch is
  undecidable from bars, and the plan refuses by name unless finer bars inside the window decide
  it or a `determination` stands, filed by the agent the contract vests it in. One rule serves the
  trade date, the expiry cut and the partial-time windows, a migrated trade's window opening at
  the booking time its fill carries. The autocall's coupon and threshold ladders fold with the put
  barrier and the pricer's barrier-hit read (it tests for presence, so it fires on a declared
  `'No'`) retires with them; the TARF's and accumulator's decisions-remain arm is folded
  parameters, not a substituted deal.
- **A payoff-shaped settlement amount in the diary.** An option's settlement row is due with
  `amount: null` because no field holds `Units × max(S−K, 0)`; the amount wants the expiry fixing
  and the payoff read together, which is a pricer's answer rather than a schedule's. One smaller
  row beside it: the diary's cache key covers the deals and the calculation but not the record its
  compile reads, so a print filed after a read does not recompile (the rows stay right, `answered`
  resolving prints per request); the key that would is the position of the record's last fill,
  amendment or fixing, the head itself moving on every append.
- **A fold of the deal panel's conventions.** A web panel renders all 48 declared keys of a swap
  where 7 are the trade, the other 41 conventions the store marks as such; a fold would hide them
  by default, and ships once something under `web/scripts` drives `FieldView`.
- **The deal tree hydrated from the fold** rather than reconciled against it. Every piece is in the
  record now - positions keyed by agreement and portfolio, each instrument's terms and each
  agreement's netting set by address - but the book file stays the materialisation and
  `/book/reconcile` is how it is checked. One composer closes it: a view - a portfolio node, the
  book, or an agreement, the credit view netted across portfolios - and a point in the record give
  the light job, its deals the instruments under the agreement's own terms. The XVA tab lists its
  netting sets off the file until then.
- **Prints as one fact, the archive as their fold.** A history for a rate, a close's row, a vendor
  correction and a day's reset are all cells under `(index, date, source)`, so they are ONE fact:
  a source and a blob of rows, each row a date with whichever of close, low and high the source
  has, filed with any dates - a fact's content carries its own dates and the LSN only says when it
  was filed, so a three-year history filed at take-on is ordinary. One fold resolves a cell, the
  latest print per `(index, date)` with sources ordered by the `fixings` policy; the diary reads it
  at a short window, and the ARCHIVE a real-world calibration reads is the same fold over a long
  window, materialised as a table named by its hash. The calibrated file is then a result over two
  hashes, the archive's and the authorised calibration JSON's, pinned and cached by them. The day's
  official close is already a blob, flattened to a row by the engine's column convention, so the
  archive grows a row a day with no vendor file for the live period; a split is the adjusted
  series re-declared, its cells superseding, raw prints and a corporate-action fact only if raw
  must be kept. Not on the take-on path: on day one the archive is a declared file version, and
  this replaces it the first time a recalibration wants the closes since. M: the fact, the fold
  reading batches, the materialiser, the close's flatten.
- **Fixings as a price factor, the deal keeping its dates.** The compile writes a print into the
  deal's own table, and `plan_hash` hashes the deals whole, so a barrier under monitoring is a new
  plan every day; on a factor's value-bound field the same print moves `values_hash` alone, the
  side a market number belongs on, and the instrument stays as booked - terms and a schedule. A
  static `Fixings` factor per index, dates to close, low and high, sourced from the prints fold:
  one source per index, so two deals observing one index cannot disagree and the document carries
  no copy of the history; the compile fills one block per index instead of cells across every
  type's table, and a print the record lacks is a missing factor by name, the engine's own
  refusal; the per-type table writing and the six-types row above retire with it. It is the price
  index's own pattern, history in the factor and dates on the deal, and an FX cross stays the
  ratio of two base-relative histories. Deals keep their schedules and per-date terms - a barrier
  level on its date, an Asian's weights - and lose the observed-value column. The cost is every
  pricer that reads a past fixing: barrier, target redemption forward, accumulator, autocall, the
  Asians, the swap family's known rates, inflation; the banked documents all carry table fixings,
  so the numbers must come out identical. Two steps: the compile fills today's tables from the
  block, pricers untouched and bit-identical by construction; then the reads move and the value
  columns retire. L.
- **What a product controller's P&L carries beside the marks.** `GET /book/pnl` says what the book
  made and its explain why the held positions moved - carry, market per risk factor, residual - but
  not the reserves beside the mid, a new deal's sales margin transferred to sales on day one, the
  interest collateral earns, collateral moving being no P&L, or a future's variation margin, which
  settles daily and which no future declares. The residual is reported with its share of the held
  positions' P&L, and no threshold a desk declares watches it.
- **Sensitivity estimators as first-class objects** — a `SensitivityProfile` per pricer, so a
  consumer can tell a pathwise derivative from one carrying a boundary term.
- **Hessian-vector products** instead of materialised Hessians: a `jvp` rule on the recompute node,
  forward-over-reverse. First consumers: the SIMM calc's dSIMM/dθ, FVA's splits.
- **Incremental XVA as risk-impact v2** — `CVA(book + mirror) − CVA(book)` through
  `Credit_Monte_Carlo`, the same two-run seam with a different calculation in it; a ratio-solve
  primitive for participating forwards beside it.
- **Service layer, what remains** — SSE for progress, a cost estimate that reads the real grid,
  budget caps per seat, and the two market-building verbs served to the MCP binding alone: the
  dependency walk and the set-up have no screen, so a desk reads a refused booking's want-list
  through a model rather than beside the book. The Securities screen's join stays read-only too: it
  names the knot quoted off a drifted or unmapped ticker, and the fix is a curve row or a seed
  entry on another pane.
- **Excel end-state** — `RF_*_PORTFOLIO` migrated to `GET /schema`, after which nothing in
  `excel_integration/` imports the engine.
- **`bind=` for payoff-only deal fields** — a strike moves no discovery, but a deal field is
  structural today, so `/book/solve` recompiles every iterate. Four candidates stay declined with
  a citation (reset rows and value-dependent leaf sets).
- **The `System` store audit** — the last hand-written store, never audited declared-versus-read
  (`Volatility_Delta`, `Master_Curves`, `Swaption_Premiums` read and undeclared; three shipped
  keys reaching no read).
- **Quote-sensitivity non-goals** on [the page](quote_sensitivities.md#non-goals): no report
  format for a quote delta, no second derivative in quote space, no SABR/SSVI.
- **Also owed**: `test_hmc_declared_knobs` (declared-versus-read on the `Hedging_Problem` knobs)
  went with the mock-built suite; batching Schrager–Pelsser across the benchmark set (25 scalar
  calls lose to one batched kernel, 0.158 s against 0.140 s on CUDA).

## Hull-White 2F, decided {#model-punchlist}

`Objective: 'Analytic'` is the default (2026-08-31) on four readings: **accuracy** — the
Schrager–Pelsser price is inside one MC evaluation's noise at 22 of 25 benchmarks on the identified
fixture (the Schrager–Pelsser annuity-freezing bias −0.13 to +2.17 bp against the MC's own
numeraire bias 0.6–3.0 bp, systematic); **stationarity** — `‖J'r‖` at θ\* is 8.63e-7 on the
analytic residual inside `Stationarity_Tol`'s 1e-3 default, against 3.16e2 on the MC quartic;
**determinism and cost** — two analytic solves at one seed agree to the bit and the four-quote
chain is 13.4 s against 75.1 s; and **the quote side exists**
([the analytic quote side](quote_sensitivities.md#the-analytic-quote-side)). `Monte_Carlo` is
unchanged to the bit and remains the oracle. The α→0 series branches, the declared seed pair and
the domestic-measure correction are in `tests/test_hw2f_analytic.py`.

A block whose `HullWhite2FactorModelParameters` factor already stands is a **warm start** — the
basin search is skipped and the least squares runs from that factor — which is 421 objective
evaluations against 6 on the four-quote block and adds no re-marking event, a document carrying no
such factor running the chain it always ran.

**Three standing re-marking events.** Every foreign-curve HW2F θ\* solved before the domestic-measure
fix re-solves to a different θ\*, every θ\* solved before 2026-09-02 re-marks on the seed and
premium-clock change, and every θ\* solved before 2026-09-17 re-marks on the basin search's
generator, numpy's default generator in place of the legacy one under the same seed: on the
four-quote document the two mean reversions move 14% and 8%, the correlation from −0.04 to −0.16
and the 1Y×1Y quote delta 4%, a walk along the directions four quotes do not identify. A desk naming an old θ\* re-baselines or re-solves; carrying one forward
looks like the first and is neither. Emissions that move with it: `Quanto_FX_Correlation_1/2` and
every locus recorded downstream of a fit. The MC's numeraire bias is the curve's tenor grid, not
discretisation (adding 1D/1M/3M/6M nodes collapses it from −1.6e-2 to −1.1e-3) — a fixture lesson
every risk-neutral calibration inherits.

## How a change is verified

- **What a change reaches**: `python gates/reach.py <Symbol|Class.method>` prints, in a second, the
  consumer classes, the callers, the JSON documents that executed the symbol fastest-first and the
  HOLES no document reaches; `--from <Consumer>` is the inverse, `--dirty` diffs symbols by AST and
  greedy-covers them by document. The static half is an AST call graph over `derivus/` with every
  registry read as data; the dynamic half is a document map, every job JSON under `tests/fixtures/`
  and in the maintainers' artifacts store run once under a tracer, keyed to the engine commit
  (`STALE` otherwise; `--build-map --repo <clean checkout>`, about 16 minutes). What it cannot
  see: string-keyed dispatch outside the registries, callables passed as values, virtual dispatch
  out of an inherited body, a branch no data takes, a document over the 180 s cap,
  `derivus_bloomberg/`. At the map's last build (2026-09-17): 956 of 2,209 symbols executed by
  some document; no document reaches 32 of 50 deals, 18 of 34 pricers, 2 of 6 bootstrapper
  families and 9 of 17 processes. The store was swept the same day of every document that
  named a retired model or declaration or a market file no longer on disk, so the 61 documents
  the map names are the ones that load, 53 pricing and 8 refusing by name on purpose.
- **Which tests a change reaches**: `gates/impacted.py --dirty --run` reads an execution map built
  at a campaign boundary, line by line - a changed statement selects the tests that executed it -
  and fails open loudly on `derivus/__init__`, `conftest` and a path the map has not seen. A
  respelling of a line most tests run still selects most of the suite: the curve read's blend
  reaches 83% of it. The full suite runs at campaign boundaries with the tree held still, 75
  minutes instrumented; the record and service sets are bound by `fsync`, 15 minutes between
  them, a home minted per test where one per module would do.
- **The standing readings every landing runs**: sixteen banked documents of the autocall
  validation campaign under the lognormal and Hull-White laws, 5,471 floats compared bit for bit
  against their bank, and one lognormal target redemption forward compared to the bit
  (`-0x1.2b36cda3bf2d4p+5`). That document declares no estimator, so it pins the default; the
  crisp path it used to pin reads `-0x1.2c48f36318e38p+5`, one `'No'` away. The documents, their
  banks and the two scripts live in the maintainers' artifacts store outside the repository;
  moving them under `gates/` waits on a check that no banked document carries desk data.
- **The LogVar2FJ module is `tests/test_logvar2fj_json.py`** (2026-09-10): 38 gates over
  a synthetic world (`tests/fixtures/data/logvar2fj_world.json`, one index quoted in EUR on a USD
  book, a five-expiry skewed ladder, a GBM sibling) — the GBM limit at 1.3e-16 and through the CVA,
  the flat-surface residual at 1.4e-12 vol points, the calibration's contract (five ATM pillars
  under 1e-10, wing RMSE 0.789 against a 1.0 bound), the retired declarations refusing by name,
  the on-guard flag on the factor and in `Stats`, `Model_Priors: Off` bit-identical to its banked
  floats, the quanto arm at 0 ULP against its single-currency twin at ρ = 0, the reserve composed
  to 1e-9, the four sub-factor rows (a declared 0.3 on all four realises 0.2040 on returns against
  0.1520 with the sibling on GBM, 6.6 path-level se apart), and 76 floats over six repo documents
  hex for hex. Most of it is one shared five-expiry fit; every gate's killing mutation went red
  except the wing-RMSE row, whose mutation stalls the fit and whose bound is set at 3.2× below the
  unfitted seed. Two fixture facts: the banked floats are the device's (a CPU-only box re-banks
  the walk's; the quadrature fit already runs on the host), and the repo's only autocall fixture
  has ONE fixing at maturity, so a vol-strip term-structure error is invisible to it (the
  cumulative variance × 1.000001 leaves both autocall hex rows green).
- **A fixture must not zero the quantity its gate is sensitive to** — the checklist and the
  plugin are on [Conventions](conventions.md#fixture-degeneracy). A mutant that survives a gate
  means the fixture is wrong, not that the code is right.

## Tidy-ups

- `gates/impacted.py` fails open to the whole suite on any change to `derivus/__init__.py`, which
  carries the context's verbs: read line by line it would still select nine tests in ten.
- `derivus_jupyter.py`, tracked but not in the wheel and superseded for viewing by the web UI,
  raises on three field names its write allowlist does not carry and on any multi-column Table
  outside a four-name allowlist, and fourteen output-shaped descriptors have no widget; loading,
  pricing, the generated docs and the MCP descriptors are unaffected. Retire or repair is a desk
  call.
- `pv_float_cashflow_list` learns that a coupon's resets fold from their count against the
  cashflows' — a shape — and which fold from the leg's declared compounding; the count is the one
  signal left that an explicit mark on the compiled cashflows would replace.

## What this list is for

Most rows here were found by auditing work that already had passing tests. The recurring failure
was a gate exercising one point of a parameter — only a bought deal, only the default monitoring
frequency, only one netting set, only the default valuation option — and the second was a
unification that absorbed N call sites into one seam and left their siblings outside it
([Conventions](conventions.md#unification-siblings)). So when picking a row up: vary the parameter
the defect would live in, and check the mutant dies before believing the test.
