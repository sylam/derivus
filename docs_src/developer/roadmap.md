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
- **A fit warns with a meaningless number when the quotes say nothing about a parameter**
  (2026-09-11). The calibration reports, for each fitted parameter, how hard a declared prior
  belief pushes it compared with the market quotes, and warns when the prior is doing most of the
  work, so that a reader can tell a fitted value from an assumed one. That comparison divides by
  how much the quotes constrain the parameter. When they do not constrain it at all, which is what
  happens to the skew parameters on a surface quoting only at-the-money options and no wings, the
  divisor is zero and the warning reports ratios above ten trillion. Its conclusion is right, since
  those parameters really are held by the prior, but the number is an artefact of dividing by zero
  rather than a reading, and it is written onto the calibrated factor and republished in every
  valuation priced off it. A reader cannot then separate a prior a hundred times stronger than the
  quotes, which is worth investigating, from quotes that are silent, which is a different
  situation with a different remedy. Either floor the divisor or detect the silent case and say so
  in words.
- **The LogVar2FJ family accepts an FX ladder the equity emitter would refuse** (2026-09-11). A
  ladder that collapses onto too few distinct contracts cannot identify the model, so a floor on
  their count refuses it by name. The family lost its own floor of eight when its former parent
  class was retired and now inherits the plain option family's six, sized for a five-parameter
  model, while the Bloomberg equity emitter still refuses under eight. A ladder collapsing onto six
  or seven contracts is accepted where it was declared refused; no document in the repository
  changes side. One line restores the eight, and whether a thin desk surface that passes at six
  should refuse at eight is the call to make first.
- **A fitted block cannot be read back to see what it was struck at** (2026-09-11). A quote row may
  leave its strike at zero to mean the forward. The quote preparation resolves that to the forward
  for the fit but no longer writes it back onto the row, so the block reads zero after the fit. One
  line writes it back; because that mutates every block that round-trips through a file, the
  blast radius comes before the line.
- **The skew reserve is one number per calculation** rather than per deal. It is composed from the
  single gradient the base valuation takes and read at the two block ends nearest the declared
  forward tenors, for the Nikkei 0.51 years into 2.23 for a declared six months into six, which
  is honest but is not the tenor a two-year autocall is exposed to. A per-deal reserve needs a
  per-deal gradient, and the declared tenor needs a post-fit reading on its own grid.
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
  reports, for each parameter, how hard its prior pushes compared with the quotes; on the desk's
  Nikkei block the tail parameter `Alpha` reads about ten times one quote row, so its fitted value
  is the prior's as much as the market's. Under the walk this showed as two fitted values from
  different seeds; the quadrature pricer, the default since 2026-09-14, writes the same bytes on
  every run of the same document, so what remains is identification, not the pricer. A block of
  forward-starting options identifies it at a tenor short enough for the residual to still be
  non-Gaussian: measured on a world the model owns, nine ONE-MONTH rows take `Alpha` from the prior
  44 to 20.27 against the world's 20.92 and the prior row from 6.74 to 2.35 quote rows, where the
  same rows at the declared default windows move neither. The vendor's chain quotes no
  forward-start.
- **The vendor's implied-volatility grid answers at 30, 60 and 90 days only**, so the long end of
  every equity fit comes from the file's own surface or the listed chain; the desk's Nikkei mark
  moves 4.6% between a chain-only fit and one carrying the file's 2.74-year at-the-money point.
  Whether the workstation is entitled to a longer grid is a question for Bloomberg support.
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
- **The newer historical estimators index an innovation one business day away from the incumbent**
  (2026-09-07): `calc_statistics`, which every ordinary factor is estimated with, places a return's
  innovation on the date the return STARTS; the GARCH, regime-switching, basis and LogVar2FJ
  estimators place it on the date the return ends, and the factor calibration correlates the two
  families as they stand. Measured on one simulated pair drawn at 0.6: a GARCH factor reads 0.009
  against a lognormal sibling and 0.335 against the LogVar2FJ estimator's own residual. The shift
  belongs to the four newer estimators, not to `calc_statistics` — every correlation a desk has
  banked was estimated under the incumbent's convention, and moving it would silently re-date all of
  them. Aligning the four leaves banked market data untouched and needs no re-read.
- **A quanto correlation can be written in two bases, and nothing checks which.** The parameter
  writer emits a quanto correlation as the correlation between the FX Brownian and each rate
  factor's own, while the correlation section's rows are the independent normals the Cholesky
  consumes, a different basis related by the two-factor rotation. Copying the writer's second
  number into the section gives a world whose rate covariance disagrees with its declaration,
  silently.

### The autocall, TARF and barrier pricers

- **An energy or commodity floating leg with fixings already set fails against a static forward
  curve** (2026-09-10). Such a leg joins its known resets to the ones still to forecast, and a
  curve that is not simulated answers with one scenario column where the known block carries the
  full width; the join refuses the shape with a tensor error naming dimensions rather than the leg
  or the curve. The rate legs had the same failure until their join was replaced by a broadcasting
  one, which exists and is proven; the two energy and commodity sites still use the plain join.
  One call at each, unmade because no document in the repository reaches it and an unmeasured
  change is not a fix.
- **A parametric FX surface never mints its parameter sub-factors** (2026-09-15). The equity
  surface lookup has a branch for a surface declared by its skew or by its SVI parameters and the
  FX surface lookup has none, so an FX surface declared that way fails at the lookup with an
  attribute error, under either moneyness rule, before any strip is read. Two documents in the
  artifacts store reproduce it.
- **Two stale-fixing reads in the autocall's observation arm.** A second consecutive coupon whose
  observation window is already wholly in the past reads the first coupon's last fixing under spot
  observation, the same staleness the window's prefix already carries; no document here reaches
  it. And a block whose fixings lag its coupon dates prices the final coupon crisply off a fixing
  the reporting row has not yet reached: on the campaign book two rows in mid 2028 read a fixing
  dated a week later.
- **The European leg of the observed-spot pricers still branches on the model family** in two
  places, a step count and a scalar carry against the walked block, where one spelling should
  serve both families.
- **The collateralised autocall's CVA delta is the boundary estimator's own variance.** Under a
  zero-threshold credit-support annex the boundary correction supplies two and a half times the
  pathwise term and scatters with the path count: 51% short at 256 outer paths, 18% over at 1,024,
  10% at 2,048; a lognormal autocall under the same annex reads 12% short at 2,048. The path
  count, not the deal, is what the number depends on. Beside it, the autocall's floating leg and
  its terminal put register no settled cash, so the reported cashflows carry the coupons alone.
- **The target redemption forward's target pin is a kink the crisp default is blind to.** The pin
  fires on 27% to 61% of paths and the delta is 27% short uncorrected. It is exact behind
  `Branch_And_Weight: 'Yes'`; the default keeps the declared blindness, since neither the bandwidth
  estimator at 13% spread nor the oracle at 9% flatness does better than about ten percent.
- **Two seasoned target redemption forwards no fixture reaches.** One whose settlements began
  before the base date discards the settled fixing outright, which is bit-identical to deleting
  it, so the deal prices against its full original target. One valued between two settlements
  marks not-a-number on every tree measured.
- **The extendable forward under a credit-support annex does not register its settled cash**: a
  quarter of a percent across four amplifying documents, against ladders that resolve no finer.
  Its rolling backward pass also carries a one-signed smoothing bias over the payoff's kink from
  the Gauss-Hermite rule it uses.
- **The partial-time barrier's rebate settlement is audited but ungated**, for want of a
  collateralised partial-barrier document.
- **The window-touch switch decides the sign of a boundary term and its magnitude is
  unestablished**: −2.25 with the window registered against +0.52 without, on an oracle that
  scatters 88% of its own median. Decision 10.
- **The boundary correction's bandwidth plateau holds at 16,384 to 20,480 paths** over bandwidths
  of 0.005 to 0.08, the correction spreading 2.4% to 3.9% and the CVA delta 0.6% to 0.2%; at 2,048
  paths the correction falls monotonically by 24%. Acceptance names 32,768 paths and that re-read
  is pending. The correction's scoping has no public seam a mutation gate could reach.
- **The accumulator's boundary placement under the recompute node is unmeasured.** Its latch is
  assembled off a node output, which puts it on the right side by construction, but the reading
  that would show a dropped cotangent has not been taken.
- **Three analytic pricers adjust the volatility for a quanto and nothing else**: the barrier, the
  one-touch and the discrete Asian. A composite-currency barrier, one-touch or Asian therefore
  prices half-adjusted without raising. The composite smile coordinate is decision 2.
- **A deal's fallback to a sibling's factor may name one discovery never fetched.** Safe at the 34
  sites where a discount rate falls back to a currency, because the interest rate arrives
  transitively; the one cross-leg instance is fixed.

### The engine

- **The volatility state a hedge critic reads differs in precision across processes**
  (2026-09-10). The LogVar2FJ outer publishes its two log-variance factors in the job's precision
  while three older spot models cast theirs to single. Inert while the hedge Monte Carlo runs in
  single precision whatever the job says; a double-precision hedge solve would hand the critic one
  double block among single ones.
- **`NettingCollateralSet`'s backward is not bit-reproducible on the GPU**: one gradient entry can
  differ in its last bits between runs of bit-identical inputs. Isolated on this card
  (2026-09-12): the backward of `gather` and of `index_select` accumulates atomically wherever
  indices collide, and five runs of one such backward differ by up to 5.3e-05 absolute, about
  forty float32 epsilons over the accumulated terms; the same backward under
  `torch.use_deterministic_algorithms(True)` is bit-identical across runs and torch accepts it,
  so a deterministic kernel exists for both. `cumsum` is not implicated: its backward is
  bit-identical over five runs on this version and raises nothing under the same switch. The
  effect is far below the 1% a desk reads, so the switch becomes a DECLARED FIELD on the
  calculation rather than a default: off is the faster arithmetic, on pins the gradient for a run
  that has to reproduce, and the document records which it was. It is set beside the cuBLAS pin so
  the dispatch workers inherit it, and it pins one machine and one build, not results across cards
  or versions. Unread: what the deterministic kernels cost on this workload.
- **The exposure profile is reported undeflated.** The deflation curve is applied only inside the
  CVA and FVA scalars, so a deflated expected exposure at an expiry row cannot be read from the
  tables. Publish the discount factor beside the mark.
- **Three books the credit Monte Carlo cannot frame, and dies on without a name.** A book whose
  only deal folded to a static value, a single scalar against the time-by-scenario grid; a book
  whose only deal was skipped; and a book whose deals reach no stochastic factor or no date after
  the base date, which dies on an empty random block or an empty maximum.
- **The kernel bandwidth is chosen per batch.** The Silverman rule sizes the boundary correction's
  bandwidth from one batch's paths, so a run with more than one batch oversmooths against its true
  path count.
- **A swaption's quoting convention lives on the surface, which the Bloomberg emitter does not
  author.** A factor declared lognormal therefore gets a lognormal fit of normal quotes, and the
  two conventions are ten to eleven times apart in premium.
- **36 of the 148 declared field reads disagree with their declaration, three fatally**, decision
  3; one structured deal reads a key spelled with a space where the declaration spells it with an
  underscore, so the declared key reaches no read.
- **A solved zero-cost strike moved between two landings of 2026-09-06 on documents that carry no
  LogVar2FJ factor**: 15.32196559 to 15.31624884, up to 3.7e-4 and fifteen times the solver's
  Monte Carlo floor, while the same documents at a fixed strike are bit-identical. The earlier tree
  is in no checkout any more, so the move cannot be pinned to a line.
- **Two plan-hash pins are a function of the calendar day**, on the platinum hedge shipping
  fixture and the simulate-only policy fixture: both load a market-data file whose `Base_Date` is
  null, and a config loaded without one takes the wall clock, so today's date sits in their plan.
  Reading the job's own `Calculation.Base_Date` instead is not a swap: one market-data file is
  cached across the jobs that name it, and they need not share a base date.
- **The notebook write path** raises on three field names its hard-coded allowlist does not carry,
  and fourteen output-shaped descriptors have no widget. Superseded for viewing by the web UI.
- **The pricer branch census read 59 unexecuted arcs on 2026-09-02** and has not been re-taken.
- **Ungated since the 2026-08-21 purge**: five modules named on
  [Conventions](conventions.md#what-holds-today-and-what-the-purge-left-open), the
  already-hit barrier leg's value the expensive one.

## Decisions waiting on the desk

Nothing here is blocked on work. Numbers are stable — commit messages and the model pages cite
them — so closed decisions (4, 13, 15) keep their numbers and are not listed.

1. **The per-fixing smile read.** Sticky-forward moneyness or the deal's declared moneyness; both
   defensible, one can be the pricer's own quote. A switch, not a revert, with the six removed gates
   rebuilt.
2. **The composite smile coordinate**, undeclared because every fixture is flat. Same class as 1.
3. **The 36 disagreeing field reads** (three fatal): hold a surviving fallback to its declaration,
   or leave the reads as they are. Enumerated in `tests/test_declared_defaults.py`.
5. **Two rates-emitter questions**: an OIS block is about 14 MB live, some 26,000 authored floats
   on a 30-year strip — accept it or build a term-authored variant; and neither side rolls a
   business day, so a two-year USD OIS pays on a Saturday.
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
10. **The window-touch switch's magnitude.** The switch decides the sign; grid dates can now be
    added, so the enriched fixture and the re-measurement are possible.
11. **The `Branch_And_Weight` default.** The family question is closed, the surviving spot model
    handing each fixing interval its own Gaussian block law; what remains is a rule for averaging
    payoffs falling back to the crisp pricer, since the averaging arms refuse under the switch.
    Values re-mark within their own Monte Carlo noise at twelve to twenty-three times less
    variance; the greeks are the prize.
12. **The correlation as a leaf.** A correlation mints no leaf today, so a quanto's correlation
    delta is reported as a common-random-number bump of 0.025: −22.42m ZAR per unit of correlation
    on the desk's Nikkei autocall, flat to 0.003% between half-widths. The leaf is three edits with
    a tree-wide blast radius, every document carrying a correlation gaining a first-order row and a
    Hessian row and column, lognormal ones included. The bump is the leaf's oracle.
14. **`Prices` in process: warning or refusal.** Mandatory on the multiprocessing path; a refusal
    costs an edit at about twenty test and gate sites that write an empty family entry.
16. **The nominal leverage weight.** `Leverage_Prior_Weight` 0.02 reads three to four quote rows on
    these ladders, not one, because it assumes a 0.2 quote weight and a 0.1 standard error no
    ladder states; a block's own `Leverage_Prior_SE` and `Leverage_Product_Prior_SE` supersede it
    where declared. At a fitted vol-of-vol of 5 it reads several quote rows, and at the
    class-default tier it is what the Nikkei cannot carry. A history's leverage of −0.20 ± 0.09
    and product −1.05 ± 0.67 move the desk mark to −41.9m through those rows: the estimator's
    leverage is the single most consequential number it produces, and its standard error is a
    sampling error, not a desk's spread.
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
- **Barrier state as a fold over fixings, the remaining half.** Continuous monitoring reads daily
  `(low, high)` bars under `(index, date, source)` — `utils.bars_touched` is the predicate and is
  gated; the SOURCE is spine increment 4's, so the one-touch and partial-time barriers still price
  from terms alone. The autocall's `Barrier_Dates` ride `Price_Fixing`'s observed value; a called
  autocall is its coupon at that fixing's settlement, folding the coupon and threshold ladders with
  the put barrier, and the autocall pricer's barrier-hit read (it tests for presence, so it fires
  on a declared `'No'`) retires with it. The TARF's and accumulator's decisions-remain arm: folded
  parameters, not a substituted deal.
- **Spine increments 4–7** — projections and the diary, tier policy, the doorbell, the generated
  binding; the book file rehomed as an LSN-pinned projection and the plan compiler as a fold over
  fixings supersession are increment 4's ([The Spine](spine.md)).
- **Sensitivity estimators as first-class objects** — a `SensitivityProfile` per pricer, so a
  consumer can tell a pathwise derivative from one carrying a boundary term.
- **Hessian-vector products** instead of materialised Hessians: a `jvp` rule on the recompute node,
  forward-over-reverse. First consumers: the SIMM calc's dSIMM/dθ, FVA's splits.
- **Incremental XVA as risk-impact v2** — `CVA(book + mirror) − CVA(book)` through
  `Credit_Monte_Carlo`, the same two-run seam with a different calculation in it; a ratio-solve
  primitive for participating forwards beside it.
- **Service layer, what remains** — SSE for progress, a cost estimate that reads the real grid,
  auth with budget caps, the web UI's edit surface and the blotter's two screens.
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

**Two standing re-marking events.** Every foreign-curve HW2F θ\* solved before the domestic-measure
fix re-solves to a different θ\*, and every θ\* solved before 2026-09-02 re-marks on the seed and
premium-clock change. A desk naming an old θ\* re-baselines or re-solves; carrying one forward
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
  `derivus_bloomberg/`. At the map's last build (2026-09-05): 852 of 2,133 symbols executed by
  some document; no document reaches 40 of 50 deals, 24 of 34 pricers and 3 of 8 bootstrapper
  families.
- **Which tests a change reaches**: `gates/impacted.py --dirty --run` joins an execution-coverage
  map (built at a campaign boundary) with a static fixture map; file-granular, fails open loudly;
  `derivus/__init__`, `utils`, `calculation` and `conftest` are whole-suite modules by construction.
  The full suite runs at campaign boundaries with the tree held still.
- **The standing readings every landing runs**: sixteen banked documents of the autocall
  validation campaign under the lognormal and Hull-White laws, 4,180 floats compared bit for bit
  against their bank, and one crisp lognormal target redemption forward compared to the bit
  (`-0x1.2c48f36318e38p+5`). The documents, their banks and the two scripts live in the
  maintainers' artifacts store outside the repository; moving them under `gates/` waits on a check
  that no banked document carries desk data.
- **The LogVar2FJ module is `tests/test_logvar2fj_json.py`** (2026-09-10): 32 gates over
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
- `gates/reach.py`'s document map is stale in substance as well as in commit: of the 43 autocall
  documents it names, 31 no longer load on the head (16 carry the Poisson-era factor block, 14
  the retired component family, one is gone), so a bit-identity claim over "every document the
  map names" is over the 12 that price. Rebuild the map (`--build-map`) at the next campaign
  boundary.
- One banked reading of a two-name correlation document from 2026-09-08 no longer reproduces at
  double precision (one term reads 4.163e-17 against 2.776e-17 banked; the single-precision half
  reproduces exactly). Re-bank it.
- `derivus_jupyter.set_repr` raises on any multi-column Table outside a four-name allowlist, which
  now includes `EquityBarrierBinaryOption.Barrier_Dates` and `QEDI_CustomAutoCallSwap.Coupon_Observations`;
  loading, pricing, the generated docs and the MCP descriptors are unaffected.
- An early calibration pack in the artifacts store still walks the retired Poisson residual and no
  longer imports; delete it or re-spell its scripts against the NIG residual.
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
