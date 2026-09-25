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
  silently.

### The autocall, TARF and barrier pricers

- **A structure's TICKET must spell its pair the way the book stores the surface.** A spot model's
  key and the fit that writes it are spelling-blind - one pair, one law, `FXVol.EUR.ZAR` or
  `FXVol.ZAR.EUR` - but `structures.materialize` looks the surface up under the ticket's own order
  and refuses `EURZAR` on a book carrying `ZAREUR`, naming what the book does quote. So a desk
  quoting a cross states the pair as its own market data spells it.
- **A parametric FX surface never mints its parameter sub-factors** (2026-09-15). The equity
  surface lookup has a branch for a surface declared by its skew or by its SVI parameters and the
  FX surface lookup has none, so an FX surface declared that way fails at the lookup with an
  attribute error, under either moneyness rule, before any strip is read. Two documents in the
  artifacts store reproduce it.
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
- **Two simulating pricers read a composite's smile at the untranslated strike.** The OSS discrete
  barrier's expiry read (`pv_discrete_barrier_option`) and the autocall's (`pv_MC_AutoCallSwap`)
  take the payoff-currency strike against the LOCAL forward, where the declared coordinate is that
  strike translated by the fx forward. Their per-fixing strips already land on the declared
  coordinate under a forward moneyness rule and differ only under a spot one.
- **A deal's fallback to a sibling's factor may name one discovery never fetched.** Safe at the 34
  sites where a discount rate falls back to a currency, because the interest rate arrives
  transitively; the one cross-leg instance is fixed.

### The engine

- **The volatility state a hedge critic reads differs in precision across processes**
  (2026-09-10). The LogVar2FJ outer publishes its two log-variance factors in the job's precision
  while three older spot models cast theirs to single. Inert while the hedge Monte Carlo runs in
  single precision whatever the job says; a double-precision hedge solve would hand the critic one
  double block among single ones.
- **A floating leg with several resets per coupon does not price** (2026-09-18). A term swap whose
  index tenor is shorter than its coupon — a semi-annual leg on a 3M index, a quarterly leg on a
  1M or 1W index, a daily leg — generates its resets at weight one over n, the shape the list
  pricer reads as OIS compounding, and the deal marks NaN under a base valuation; a cashflow list
  hand-authored in that shape, one item per coupon carrying every fixing's reset, pays one over n
  of the interest, 289.08 against 76,967.94 on a two-year annual leg of 266 fixings. One reset
  spanning each coupon, the par swap's default, is exact, and [Quote
  Sensitivities](quote_sensitivities.md#curve-contracts) carries the readings.
- **A solved zero-cost strike moved between two landings of 2026-09-06 on documents that carry no
  LogVar2FJ factor**: 15.32196559 to 15.31624884, up to 3.7e-4 and fifteen times the solver's
  Monte Carlo floor, while the same documents at a fixed strike are bit-identical. The earlier tree
  is in no checkout any more, so the move cannot be pinned to a line.
- **The notebook write path** raises on three field names its hard-coded allowlist does not carry,
  and fourteen output-shaped descriptors have no widget. Superseded for viewing by the web UI.
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
- **Six declared fields carry stated values nothing reads** (2026-09-22). `DealDefaultSwap`'s
  `Is_Digital` and `Digital_Recovery` select a branch that was never wired; its `Upfront` is a real
  payment no pricer discounts; `QEDI_CustomAutoCallSwap`'s `Units` is the deal's notional and
  neither the V1 nor the V2 pricer reads it, the strip being unitless today; `Rate_Currency` on
  `DepositDeal` and `CFFixedInterestListDeal` names a reset currency on a leg with no quanto path;
  and `Averaging_Method` on `CapDeal`/`FloorDeal` is declared `Average_Rate` while the only reader
  in the tree is a cashflow LIST's own container key, falling back to `None` on another class.
  A desk that states one is silently ignored, which is the opposite failure to the one the
  convention/placeholder split closes: UNMEASURED, because there is no reading to compare against.
  Each is either wired to the branch it names or deleted with the branch.
- **A deal the compile could not read still lets the job report success** (2026-09-22). A
  placeholder a document does not say — the key absent, or carrying a `null` — is refused by name
  at booking, so it cannot reach a pricer through
  `POST /book/deals`; a document loaded from disk, an emitter's benchmark or a legacy file can
  still carry one, and `DealStructure.add_deal_to_structure` logs `<type> <reference> <key> -
  Skipped`, counts it under `Deals Skipped` and the run finishes green. `Stats` carries the count
  and the diary files the deal as unreadable, but nothing makes the run itself fail:
  `System Parameters.Exclude_Deals_With_Missing_Market_Data` is declared `Yes`/`No` and documented
  as raising on `No`, and no module reads it.
- **A blank table has two wire spellings and they are not one value** (2026-09-22). A widget writes
  an empty Table as JSON `null`, which the loader reads as `None`; the same table written as its
  own container (`{".DateList": []}`) reads as an empty `DateList`, and a `utils` container defines
  no `__bool__`, so an empty one is TRUTHY. Every `if self.field['<table>']:` guard therefore
  branches differently on two documents that say the same nothing, and a convention completed from
  its declaration takes the container arm. Live example, pre-existing: `EquitySwapLeg`'s dividend
  read takes that arm and calls `DateEqualList.sum_range` with two of its three arguments
  (`instruments.py:5437`), so no equity swap leg whose dividends are a table compiles at all.
  The fix is one rule for the blank — either the loader's `None` or an empty container that is
  falsy — and the arity bug goes with it. UNMEASURED in price, and nothing in the tree can measure
  it: no document carries an `EquitySwapLeg`, and no deal in any fixture or job file states a blank
  table in either spelling.
- **Every book carrying a netting set moves its plan hash once, at this landing** (2026-09-22).
  `NettingCollateralSet.__init__` used to `setdefault` `Settlement_Period`, `Liquidation_Period`
  and `Opening_Balance` INTO the authored block, and the plan hashes the block; the declaration
  says all three, so the same set is three keys shorter and the same program hashes differently.
  Nothing priced moves — `commodity_aps_world.json` is 65,584 reported floats and 0 mismatches
  across the move, and its factor universe is unchanged. What moves is a PIN: under a spine every
  booked deal sits beneath a netting set, and a pending quote pins `plan_hash` beside
  `values_hash`, so a quote pinned before the deploy and accepted after is refused on the PLAN —
  the book moved under the solve, which is what that equality is for. The remedy is
  the ordinary one: drain the pending quotes before deploying, or re-quote. `/book/status`
  publishes no plan hash, so nothing else surfaces it.
- **The deal panel shows every convention beside the terms** (2026-09-22). A web panel renders all
  48 declared keys of a swap where 7 are the trade, the other 41 being conventions the store now
  marks as such (`convention` on the descriptor). A fold would hide them by default; it is not
  built because nothing under `web/scripts` drives `FieldView`, so it would ship ungated.
- **A consolidated risk read refits the quote blocks its book prices, and one refusal reads all of
  it on factors** (2026-09-24). `/book/risk` refits, with `Quote_Sensitivity` on, every block
  writing a factor the book reads and every block those stand on, so a spot-model block pays its
  warm polish on every uncached read; and a bootstrap raises at its first refusal, so one block that
  cannot carry a quote derivative - an FX forward outright, a polish stopping above
  `Stationarity_Tol` - returns the whole book's risk in factor space with the refusal named.
  UNMEASURED on a desk book; a two-trade FX book reads in 5.1 s, its three blocks' refit
  included. The remedy for the second is a retry with the refusing block's switch off.
- **Opening a log scans it, and every read of the record opens one** (2026-09-23). The seek closed
  the READING half of the desk's beat - a page of ten at the head of a 2,005-event log is 0.20 ms
  where it was 7.50, and one two-second beat is 28 ms where it was 44 - and what is left is
  `SpineLog.__init__`, which streams every segment to build the head, the tag index and the offset
  per LSN: 11.3 ms at 2,005 events, paid once per read because a reader never claims the home and
  so cannot hold a handle across one. A strip's FIRST paint pays it twice over, since a page with
  no `?since=` folds from genesis by design. Queue admission rides the same open once per submitted
  job (`capability.state_at`, 1.2 ms at 21 events and 20.5 ms at 1,994), which is still under a
  fiftieth of the cheapest job it gates, so nothing caches it. The remedy is a checkpointed index
  beside `log/` - derivable, disposable and verified the way a seed is - and the question it asks
  is what invalidates one when a second process appends.
- **A private market has no surveillance or admin read** (2026-09-23). `spine.resolve_market`
  resolves a `private/<subject>/<name>` market for the subject its name names and refuses everyone
  else at the verb, without minting a fact. The other read the design names — surveillance and
  admin — is a second entitlement class and a `read` row per subject, which is the reclassification
  `vocabulary.classify` ships dormant for; a per-market rule instead would be the per-object ACL the
  design forbids by name. Size: the class and the rows are a desk-two decision and the code is a
  second branch in one function, UNMEASURED because nothing has asked for the read yet. Nothing is
  blocked by it: a designated process resolves the firm's own market, and a private one is its
  declaring seat's.
- **`spine.quotes()` opens every quote the record holds** (2026-09-23). The `quotes` fold is the one
  projector whose rows grow with the desk's own activity: the envelope filter keeps it off every
  other event, and it opens a body per quote at about 0.16 ms, so a two-thousand-event home holding
  three quotes folds in 7.7 ms and one holding 1,978 in 327 ms. A desk quoting a hundred a day
  reaches the second reading in about three weeks. It is a READING and nothing on a booking path
  calls it: a decision seeks to the one frame the acceptance wrote down (`spine.quote_at`), so the
  cost is one body whatever the desk has quoted. The log's own seek answered the READING half of
  this - a page of quotes now reaches its first row in 0.20 ms rather than 7.50 - and the 0.16 ms
  per body stands, so what a paged `quotes` read wants is a `?since=` on the verb and a seed at the
  close, not a faster walk.
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
  a `rejection` against it: a verdict is a SEAT's decision and the tiers policy names no seat for
  one. A desk wanting the refusal on the record calls `POST /book/quote/reject` under a seat of its
  own. Size: UNMEASURED and not measurable — it is a document decision rather than a number. Closing
  it means the tiers document declaring who signs a refusal, one field on a tier and one branch in
  the tier step, and the question it asks is whose signature a desk wants on a "no".
- **A booking's tier step advances a fold this process holds, and it is still linear in the rows it
  mints** (2026-09-23). `spine.route_ticket` folds `decisions` and `markets` on every acceptance,
  and both open a body per row: folding from genesis costs 0.164 ms per decision filed, so a desk
  two thousand decisions in would pay about 330 ms inside the write closure. The pair `fold` already
  takes is held per projector and advanced instead, which is 0.026 ms per decision — 6.3× cheaper,
  about 57 ms at two thousand — but NOT flat: 0.011 of it is the envelope walk every fold pays per
  event, and 0.015 is `projections._from_seed`'s canonical copy of a state that grows with the
  decisions, the same copy that makes the `activity` strip 219 ms where folding it costs 38. And the
  pair lives in the PROCESS, so the first acceptance after a restart pays the whole history and a
  second service on the same home pays it again. One remedy answers all three: a seed minted at the
  official close (`projections.seed_at`, which `positions` and `blotter` already use), so a fresh
  process starts where the day started and the state copied is the day's rather than the record's.
  Size: one `read_seed` in `advancing` and a close that mints for these two projectors.
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
  and the payoff read together, which is a pricer's answer rather than a schedule's. Two smaller
  rows beside it: a swap's `Fixed_Compounding` is injected into a copy of `Factor_dep` at pricing
  time, so the diary reads `False` whatever the deal says — harmless today, both branches
  coinciding on a one-row-per-pay-day leg, and wrong the day that stops being true; and the diary's
  cache key covers the deals and the calculation but not the record its compile now reads, so a
  print filed after a read does not recompile (the rows stay right, `answered` resolving prints per
  request).
- **The deal tree hydrated from the fold** rather than reconciled against it. Every piece is in the
  record now - positions keyed by agreement and portfolio, each instrument's terms and each
  agreement's netting set by address - but the book file stays the materialisation and
  `/book/reconcile` is how it is checked.
- **Sensitivity estimators as first-class objects** — a `SensitivityProfile` per pricer, so a
  consumer can tell a pathwise derivative from one carrying a boundary term.
- **Hessian-vector products** instead of materialised Hessians: a `jvp` rule on the recompute node,
  forward-over-reverse. First consumers: the SIMM calc's dSIMM/dθ, FVA's splits.
- **Incremental XVA as risk-impact v2** — `CVA(book + mirror) − CVA(book)` through
  `Credit_Monte_Carlo`, the same two-run seam with a different calculation in it; a ratio-solve
  primitive for participating forwards beside it.
- **Service layer, what remains** — SSE for progress, a cost estimate that reads the real grid,
  auth with budget caps, and the two market-building verbs served to the MCP binding alone: the
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
  validation campaign under the lognormal and Hull-White laws, 5,487 floats compared bit for bit
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

- `gates/reach.py --dirty` died on the Windows box decoding `git`'s output as cp1252 (2026-09-08)
  and ran clean there on 2026-09-10; if it recurs, decode the diff as UTF-8.
- `gates/impacted.py --dirty` fails open to the whole suite on a fixture the map has not seen and
  on a `.md` at the repo root, so a change that adds a fixture cannot use the selector until the next
  boundary run rebuilds the map.
- One banked reading of a two-name correlation document from 2026-09-08 no longer reproduces at
  double precision (one term reads 4.163e-17 against 2.776e-17 banked; the single-precision half
  reproduces exactly). Re-bank it.
- `derivus_jupyter.set_repr` raises on any multi-column Table outside a four-name allowlist, which
  now includes `EquityBarrierBinaryOption.Barrier_Dates` and `QEDI_CustomAutoCallSwap.Coupon_Observations`;
  loading, pricing, the generated docs and the MCP descriptors are unaffected.
- Inline comment density: about twelve blocks of 4–11 comment lines from the boundary-correction
  work (the discrete barrier's hit-mask and rebate blocks, the observed-spot walk's terminal
  digital, the net-from-gross helper); house style is 2–3 lines.
- `pv_float_cashflow_list` selects the compounded-in-arrears path by comparing reset count to
  cashflow count — a shape encoding of intent that an explicit signal on the compiled cashflow
  object would replace.

## What this list is for

Most rows here were found by auditing work that already had passing tests. The recurring failure
was a gate exercising one point of a parameter — only a bought deal, only the default monitoring
frequency, only one netting set, only the default valuation option — and the second was a
unification that absorbed N call sites into one seam and left their siblings outside it
([Conventions](conventions.md#unification-siblings)). So when picking a row up: vary the parameter
the defect would live in, and check the mutant dies before believing the test.
