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

### The LogVar2FJ calibration

These are defects in the engine: each has a change to this library that closes it.

- **One prior is applied to the first bucket only** (2026-09-10). A ladder may fit its residual
  tail per calendar bucket, and the priors on the skew share and the leverage follow the buckets,
  but the prior on the tail parameter `Alpha` sits on the first bucket alone. Nothing moves today,
  because every book ladder fits one bucket, and a multi-bucket fit with priors on has never been
  run.
- **A fitted block cannot be read back to see what it was struck at** (2026-09-11). A quote row may
  leave its strike at zero to mean the forward. The quote preparation resolves that to the forward
  for the fit but no longer writes it back onto the row, so the block reads zero after the fit. One
  line writes it back; because that mutates every block that round-trips through a file, the
  blast radius comes before the line.
- **A Gaussian residual reports two sensitivities the model never reads.** Under `Residual_Law:
  Gaussian` the tail parameters `Alpha` and `Beta` are filled with defaults and reported as
  sensitivities with identically zero rows. Dropping them would make the set of reported
  sensitivities depend on the mode, which is why they are left.
- **The recompute segment length was measured for the walk alone.** The walk is checkpointed every
  21 internal steps to trade memory for a second forward pass; the mixer's per-draw checkpoint
  beside it took peak memory from 8,392 to 9,990 MiB at 2,048 scenarios by 2,048 paths. The
  residual's granularity is the dial if that headroom is wanted.
- **Three calibration dials are module constants rather than declared fields**: the tail
  parameter's seed pair (0.5, 0.05), the solver's step tolerance of 1e-12, and the ten-knot default
  grid of the Hull-White sigma term structure. A desk cannot change them from a document.
- **Under pseudo-random sampling the mixer's uniform has 24 bits** (2026-09-10). The uniform behind
  the inverse-Gaussian mixer is drawn in single precision under `Sampling: Pseudo`, and one minus
  it at the clamp's margin carries 3% error; widening the draw changes how much of the stream each
  step consumes, and so what a double-precision document reproduces. Production takes the
  low-discrepancy sequence above 16 scenarios, where the uniform is drawn in double.
- **One primitive of the walk overflows single precision past κT = 88** (2026-09-10). The
  Ornstein-Uhlenbeck path is spelled with an integrating factor that grows as the exponential of
  the reversion speed times elapsed time. Every caller keeps it in range, the walk through its
  21-step segments and the state variance in double, but the primitive would overflow a
  single-precision caller. The fix is a chunked rescale rather than a cast. Beside it, the clock
  is still summed across blocks in the job's precision, and widening it cascades into the mixer
  and every pricer that reads the law.

#### What the quotes cannot say

These are properties of the market data available, not of the engine. No change to this
library closes one; each is a limit on what a fit of that data can be asked to identify, and
is recorded so a reader knows which readings rest on it.

- **A declared standard error tight enough is a pin, and the guard says so** (2026-09-10). The
  Nasdaq at its regression's own 0.015 reads 138 quote rows on the leverage and is flagged; the
  Nikkei at 0.025 reads 38.5 and is not. Whether a 1,149-day regression's sampling error is the
  right spread for a risk-neutral prior is a modelling question; `Leverage_Prior_SE` is where a
  desk states its own.
- **The width of the residual's skew prior is asserted, not measured** (2026-09-10). At 0.2 the
  wings outvote it by about one standard error on every index ladder, −0.29 to −0.49 against a
  prior of −0.5. The historical estimator's own spread on an uncontaminated history would measure
  it, and no index history here is uncontaminated: the Nikkei's estimated clock share reads 0.9967
  against a fitted 0.2671.
- **A ladder of vanillas alone does not pin the residual's tail parameter** (2026-09-15). The fit
  reports, for each parameter, how hard its prior pushes compared with the quotes; on a
  Nikkei block the tail parameter `Alpha` reads about ten times one quote row, so its fitted value
  is the prior's as much as the market's. Under the walk this showed as two fitted values from
  different seeds; the quadrature pricer, the default since 2026-09-14, writes the same bytes on
  every run of the same document, so what remains is identification, not the pricer. A block of
  forward-starting options identifies it at a tenor short enough for the residual to still be
  non-Gaussian: measured on a world the model owns, nine ONE-MONTH rows take `Alpha` from the prior
  44 to 20.27 against the world's 20.92 and the prior row from 6.74 to 2.35 quote rows, where the
  same rows at the declared default windows move neither. The vendor's chain quotes no
  forward-start.
- **The Nikkei's implied-volatility surface answers one maturity** (2026-09-15), so its long end
  comes from the listed chain, pulled in the Tokyo session, or the file's own surface; a Nikkei
  autocall's mark moves 4.6% between a chain-only fit and one carrying the file's long-dated
  at-the-money point. The S&P 500, Nasdaq 100 and Euro Stoxx 50 surfaces answer three to
  twenty-four months at 90 to 110 percent moneyness, and the listed chain carries every expiry
  past that.
- **A closes-only price archive attenuates the historical estimator's shock rows** (2026-09-11).
  Two names simulated from the four-factor process with every correlation row at 0.60 and
  re-estimated from daily closes read 0.55 on the return row, 0.32 and 0.53 on the two
  volatility-shock rows and 0.35 on the mixer, each to about ±0.015. A smoothed shock is a linear
  functional of one name's noisy observations, so its cross-name correlation is the state's share
  of that functional's variance, driven down by the measurement noise; the mixer column is further
  compressed by a fit whose tail parameter reads 258 to 380 against a truth of 44. The estimator
  logs both shock standard deviations by name. A range-bar archive is the measurement that would
  close it; the script for it is written and has not been run.
- **A ladder shorter than the slow horizon cannot reach the model's flat limit** (2026-09-10). The
  slow factor's horizon is 1.5 years, and a ladder with no wing that far injects skew and convexity
  a flat surface does not want, so a flat 20% ladder fitted with priors off lands at 0.21 vol
  points on one rung and 0.44 on two rather than at zero, with a fitted leverage product of −0.001
  beside the pinned −0.400. Every ladder a short-dated FX desk quotes is such a ladder, and a
  flat-limit gate needs one reaching 1.5 years.

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

- **A stale-fixing read in the autocall's observation arm.** A second consecutive coupon whose
  observation window is already wholly in the past reads the first coupon's last fixing under spot
  observation, the same staleness the window's prefix already carries; no document here reaches
  it. A block whose fixings lag its coupon dates is a booking error, not the engine's: the
  fixing schedule is the deal's to author.
- **The European leg of the observed-spot pricers still branches on the model family** in two
  places, a step count and a scalar carry against the walked block, where one spelling should
  serve both families.
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
- **The extendable forward under a credit-support annex does not register its settled cash**: a
  quarter of a percent across four amplifying documents, against ladders that resolve no finer.
  Its rolling backward pass also carries a one-signed smoothing bias over the payoff's kink from
  the Gauss-Hermite rule it uses.
- **The partial-time barrier's rebate settlement is audited but ungated**, for want of a
  collateralised partial-barrier document.
- **The window-touch registration's ladder is not flat, and it is the default on its sign**: on a
  grid carrying a row a month — seven inside the window, six of them live — five seeds read a
  registered −2.136 at 8,192 paths and −1.915 at 32,768 against an unregistered +1.318, every one
  of the 70 CRN readings negative and the pooled oracle 0.5% from the registered delta against
  168% from the unregistered one. The ladder is still not FLAT, 13% to 35% over h ≥ 5e-4, so the
  registered delta's magnitude is known to about a third where the unregistered sign is known to
  be wrong; `Boundary_AAD_Window_Touch: No` is the unregistered estimator, one value away. The
  switch is the credit Monte Carlo's alone: a base valuation's single date never reaches the
  observed-spot branch the latch lives in, so it declares no such field.
- **The boundary correction's bandwidth plateau holds at 16,384 to 20,480 paths** over bandwidths
  of 0.005 to 0.08, the correction spreading 2.4% to 3.9% and the CVA delta 0.6% to 0.2%; at 2,048
  paths the correction falls monotonically by 24%. The declared default now sits at 16,384, the
  bottom of that plateau, and the acceptance re-read at 32,768 is pending. The correction's scoping
  has no public seam a mutation gate could reach.
- **The accumulator's latch is measured OUTSIDE the recompute node, and no gate holds it there.**
  Dropping every node cotangent but the marks' reproduces the corrected CVA gradient bit for bit
  while suppressing the correction moves it 2.52% — the barrier's side, not the autocall's, the
  gaps being the outer scenario's own observed fixings. The injection mutant cannot fail on this
  pricer: of the node's five outputs only the marks' ever arrives with a cotangent.
- **The American option's approximation is to be retired, not patched** (2026-09-16).
  `pv_american_option`, which an `EquityOptionDeal` carrying `Option_Style: American` reaches,
  never calls `calc_vol_adjustment`, so a composite or quanto American prices as the local asset
  whatever the payoff currency says. The owner's ruling: leave it as it is and replace the
  approximation with a different pricer rather than adjust this one.

### The engine

- **Four compounding methods refuse a leg with several resets per coupon** (2026-09-30). `Flat`,
  `Include_Margin`, `Exclude_Margin` and `Exponential` compile the same schedule the averaging leg
  does, its resets at one over n, so their arithmetic is not there: where they paid one over n of
  the interest they refuse by name. `OIS` compounds and `None` averages. Size: the fold of each in
  `pv_float_cashflow_list`, a branch apiece.
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
- **The legs of one package are not netted against each other** (2026-09-21). Each leg pays its own
  spread, so a collar's bought put and sold call are charged as two tickets rather than as the one
  position the desk actually has to deal. Measured on the gate's 1m USD collar: netting the legs
  per pillar before charging takes the edge from 1,985.57 to 297.25, an 85% cut — the ATM row from
  1,441.26 to 71.47 and the butterfly from 362.87 to 44.34, while the risk reversal, which both
  legs read the same way, does not move at all. A desk ruling rather than a defect, and the next
  dial on this charge.
- **Four declared fields carry stated values nothing reads** (2026-09-22, 2026-09-26). A floating
  cashflow list's `Settlement_Date` and `Settlement_Amount`, which its pricer never reads where the
  fixed list's does in four places; and a `NettingCollateralSet`'s `Haircut_Received`, on every
  collateral row, where the engine and the collateral call take `Haircut_Posted` whichever side
  holds the asset, and its `Independent_Amount_Reference`, a positive independent amount being
  support the bank receives whichever party the field names. A desk that states one is silently
  ignored, which is the opposite failure to the one the convention/placeholder split closes:
  UNMEASURED, because there is no reading to compare against. Each is either wired to the branch
  it names or deleted with the branch; the list's pair is M, the set's two move collateral numbers.
- **A structure or a swaption outside a netting set breaks a credit Monte Carlo's dates**
  (2026-09-30). The root takes its report dates from its sub-structures alone
  (`DealStructure.finalize_struct`), so a `StructuredDeal` or a `SwaptionDeal` at the top of the
  tree raises `Shape of passed values is (20, 256), indices imply (15, 256)`; the same deals under a
  netting set price.
- **A Hull-White on the EUR curve beside a lognormal EURUSD reads NaN on every EURUSD deal**
  (2026-09-30); with EURUSD static instead, `FXOneTouchOption` trips a CUDA device-side gather
  assert that poisons the process. Both measured on the fx trial family; the cause is not.
- **`index_reference` clamps an unpublished month to the last print** (2026-09-30), which a credit
  Monte Carlo then reads as that month's level, and `calc_index` assumes references arrive in time
  order. Read from the code, not measured.
- **An equity swap leg skips in two natural spellings and pays no dividend** (2026-09-26). A blank
  `Payoff_Currency` is not read as the leg's own currency and a leg started on or before the base
  date with no known price at its start carries a `None` FX rate - each skips the leg - and
  compiled, `Known_Dividends` reaches only the reset's `Weight` slot, which `pv_equity_cashflows`
  never reads. The leg's generator rework is where they close.
- **A legacy trade closed before it is migrated prices short** (2026-09-26). A node the file carries
  that no fill ever booked prices as written, one unit; a close-out of it booked through the verbs
  files a fill of -1, and the compile writes the node at that net, the mirror, where nothing should
  stand - one probe read -85.38 against 0, where the tree before read +170.76. The migration the
  design names, a fill of one under every legacy node, is what makes the close net to nothing -
  and until it runs, a node no fill booked is outside the desk's P&L, which reads its positions
  off the record.
- **Three guards read the skip switch not at all** (2026-09-30). Under
  `System Parameters.Exclude_Deals_With_Missing_Market_Data: No` the compile guard and the pricer's
  guard refuse by name; `add_structure_to_structure`, `resolve_structure`'s `post_process` guard
  and the netting set dropped for holding a NaN skip under `No` as under `Yes`. Size: the switch
  read at each, a line apiece.
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
- **A private market has no surveillance or admin read** (2026-09-23). `spine.resolve_market`
  resolves a `private/<subject>/<name>` market for the subject its name names and refuses everyone
  else at the verb, without minting a fact. The other read the design names — surveillance and
  admin — is a second entitlement class and a `read` row per subject, which is the reclassification
  `vocabulary.classify` ships dormant for; a per-market rule instead would be the per-object ACL the
  design forbids by name. Size: the class and the rows are a desk-two decision and the code is a
  second branch in one function, UNMEASURED because nothing has asked for the read yet. Nothing is
  blocked by it: a designated process resolves the firm's own market, and a private one is its
  declaring seat's.
- **A blob is served by the hub and by nobody else** (2026-09-23). A replica holds `blobs/` and
  every byte in it is self-verifying by hash, so a follower could serve another follower and neither
  would have to trust the other. It does not: a peer server is a second entitlement-evaluating
  surface on a box that is not the writer, and this phase has one deployment, so the read
  (`GET /spine/blobs/{hash}`) is the hub's alone and a replica that wants bytes asks it. UNMEASURED,
  and the number that would decide it is a hub's outbound cost per follower, which at one deployment
  is a hub answering itself. Closing it means the entitlement evaluation running where the bytes
  are, which is the same `read` rows over the same fold - and the question it asks is whose document
  a replica evaluates under when its own chain is behind the hub's.
- **A material market move is not a refusal, and a desk cannot ask for one** (2026-09-23). Between
  a quote and the client's word the board moves, and the booking REPORTS it — the values struck on,
  the ones standing, and that they differ — because the desk's own `Quote Policy.firm_seconds` is
  the promise that bounds it. A desk that wanted a refusal would declare a tolerance per field:
  which values-plane fields it cares about and how far each may move before a booking is refused
  rather than reported. UNMEASURED, and measuring it takes the thing that does not exist yet — a
  comparison between two values vectors that answers WHICH numbers moved and by how much, where
  today the record compares two 64-hex addresses. That is a field-level diff over
  `market_patch`'s own shape, per-field epsilons declared like the tolerance policy's, and a gate
  on a tick that moves one pillar inside the epsilon and one outside it.
- **No rejection is filed automatically** (2026-09-23). A ticket that falls in no tier answers
  `refused` with every sentence of the route it took, and the acceptance stands, but nothing files
  a `rejection` against it: the hub's own voice signs an automatic tier's approval and says nothing
  against a ticket. A desk wanting the refusal on the record calls `POST /book/reject` under a seat
  of its own. Size: UNMEASURED and not measurable — it is a document decision rather than a
  number. Closing it means the writer's own voice filing a `rejection` where the route admits no
  tier, one type in that voice and one branch in the tier step, and the question it asks is whether
  a desk wants the hub's signature on a "no".
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
- **Twelve types announce fixings that name no index** (2026-09-30). A fixing row is answered by
  a print filed under the index it names, which a type declares as `observes`; a row naming none
  is answered by nothing, so `close_check` waits on it for ever and the day it falls on gets no
  close, no marks and no P&L. The swap, the floating list, the cap and the floor, the FRA and the
  deposit declare theirs, and read a reset the record has printed off the curve until their known-rate tables are
  filled from the record at compile as an observation table is - a day's move on one reset. A
  blank index field, which the compile resolves to the currency's curve, still announces its
  fixings under none. Undeclared: the FX accumulator, TARF, extendable forward and both Asians,
  the equity Asian, the equity swap leg and swaplet list, the inflation list, and the floating
  energy deal, energy option and commodity average-price swap - each reads its fixings off a table
  of its own that decides its payoff, so a close passing on a print its mark ignores would be the
  worse failure. Beside them a swaption announces no expiry of its own, its legs reading as a live
  swap's, and an equity binary no expiry fixing where the FX one does. Size: a declaration per
  type once its table is filled from the record, the barrier-state row under Designed, not built;
  the census in `tests/test_diary.py` names every type still open.
- **One `settle` verb is the whole back office** (2026-09-30). Settlements, confirmations and
  collateral each hold `settle`, so each worklist lists the others' payments, clips and calls, and
  confirmations may pay or post collateral: separation of duties inside the back office is not
  expressible with the verbs the record has. A design decision - a narrower verb or none - not a
  number.
- **The pricer branch census read 59 unexecuted arcs on 2026-09-02** and has not been re-taken.
- **Ungated since the 2026-08-21 purge**: five modules named on
  [Conventions](conventions.md#what-holds-today-and-what-the-purge-left-open), the
  already-hit barrier leg's value the expensive one.

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

## Designed, not built

- **The density recursion** — one FFT convolution per monitored date against the block Gaussian,
  as an alternative inner estimator. The daily walk's tape at 2,048 × 2,048 × 509 does not fit a
  24 GiB card in either direction (the draws alone are 3 × 7.95 GiB), which is what the per-block
  checkpointing exists for; no coarser chain is licensed (a 5-day gap read 3.4 SE on the coupon
  leg).
- **Barrier state as a fold over fixings, what remains.** `Barrier_Dates` and `Price_Fixing` are
  filled from the record at compile; still terms-only are continuous monitoring, which reads daily
  `(low, high)` bars under `(index, date, source)` — `utils.bars_touched` is the predicate and is
  gated — so the one-touch and partial-time barriers price from terms alone. The autocall's coupon
  and threshold ladders fold with the put barrier and the pricer's barrier-hit read (it tests for
  presence, so it fires on a declared `'No'`) retires with them; the TARF's and accumulator's
  decisions-remain arm is folded parameters, not a substituted deal.
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
  `/book/reconcile` is how it is checked.
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
- **Which tests a change reaches**: `gates/impacted.py --dirty --run` joins an execution-coverage
  map (built at a campaign boundary) with a static fixture map; file-granular, fails open loudly;
  `derivus/__init__`, `utils`, `calculation` and `conftest` are whole-suite modules by construction.
  The full suite runs at campaign boundaries with the tree held still.
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

- `gates/impacted.py --dirty` fails open to the whole suite on a fixture the map has not seen and
  on a `.md` at the repo root, so a change that adds a fixture cannot use the selector until the next
  boundary run rebuilds the map.
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
