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

- **The index defaults cannot fit the Nikkei** (2026-09-10). A fit starts every index from one set
  of class defaults for the leverage, the correlation between an index's return and its own
  volatility, sized on the S&P 500. On the Nikkei those defaults land the fast vol-of-vol on the
  top of its box with the leverage at −0.40, a fit stopped at a wall rather than at a minimum,
  while the Nikkei's own implied pair, −0.53 for the leverage and −1.62 for its product with the
  vol-of-vol, lands clean. The Bloomberg emitter should write per-name numbers from each index's
  own volatility index; a wider box is not the fix.
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
- **A reference-model source for the forward smile is declared and unwired.** The schema accepts
  `Forward_Smile_Source: Reference`; a fit that declares it says at INFO that no reference model is
  wired and prices its forward rows as if none were declared.
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
- **The Nikkei chain block's vanilla-only objective is bimodal under the walk** (2026-09-08):
  thirteen fits over three seeds and four path counts land between 0.976 and 1.016 vol points of
  residual but split into a tail parameter of 0.6 to 2.1 or 7.6 to 8.0, and the same seed lands in
  one mode at one path count and the other at another. The quadrature pricer for the vanilla rows,
  the default since 2026-09-14, takes the seed and the path count out of the objective and lands
  the fixture ladder in one basin from every seed; whether the Nikkei block's two modes survive it
  is unmeasured, and the forward block is still the identification it lacks.
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

- **The cut-interval second mixer is still a private die** (2026-09-11): the outer takes one
  framework normal per scenario interval for its mixer, so where a calendar `Alpha` bucket knot
  cuts an interval's clock into two residual draws the second piece's mixer comes from the
  process's own stream, named at INFO; every factor in the book carries one `Alpha` bucket, so
  nothing reaches it. Beside it, `Φ` of the mixer normal in double saturates past 8.3σ, where
  `ig_root` answers its bracket's top — one draw in 1e16, named rather than guarded.
- **Lane 4's banked `out_g23.json` no longer reproduces at float64** on any tree since
  2026-09-08 (`M_lev` 4.163e-17 banked against 2.776e-17), the float32 half reproducing exactly;
  a re-bank at the head.
- **A correlation declared under a process name no simulated factor answers to is silent**
  (2026-09-09): the lookup reads 0.0 for a missing pair, right for an undeclared one and wrong for
  one filed under a stale key. Name it at INFO.
- **`Correlations` cannot be authored in `ExplicitMarketData`** (2026-09-10): `Config`
  keys the section by a `(name, name)` tuple and builds it only on the `MarketDataFile` path, while
  `Context.load_json` merges an explicit section by `dict.update`, so a correlation written there
  lands under a string key that `get_cholesky_decomp` never looks up — a silent zero. Every
  correlated document needs a market-data file today.
- **The newer historical estimators index an innovation one business day away from the incumbent**
  (2026-09-07): `calc_statistics`, which every ordinary factor is estimated with, places a return's
  innovation on the date the return STARTS; the GARCH, regime-switching, basis and LogVar2FJ
  estimators place it on the date the return ends, and the factor calibration correlates the two
  families as they stand. Measured on one simulated pair drawn at 0.6: a GARCH factor reads 0.009
  against a lognormal sibling and 0.335 against the LogVar2FJ estimator's own residual. The shift
  belongs to the four newer estimators, not to `calc_statistics` — every correlation a desk has
  banked was estimated under the incumbent's convention, and moving it would silently re-date all of
  them. Aligning the four leaves banked market data untouched and needs no re-read.
- **`calibrate_factors` reports a calibration class's own refusal as "Data errors in factor"**, and
  raises `AttributeError: 'NoneType' object has no attribute 'corr'` where every factor is skipped
  (2026-09-07). The same swallow one layer up: a bootstrapper family that refuses at CONSTRUCTION
  (the leverage tables' sign refusal) is caught by `Config.bootstrap`, logged, and the factor left
  unwritten, so a caller that does not read the log sees a bootstrap that succeeded and a
  `KeyError` at the first pricer that wants the factor (2026-09-10).
- **`Correlations` vs `save_params` author a quanto in different bases, unchecked**: `save_params`
  emits `ρ̄ᵢ = corr(dW, dWᵢ)` while the section's rows are the independent normals the Cholesky
  consumes (`a = ρ̄₁`, `b = (ρ̄₂ − ρ·ρ̄₁)/√(1−ρ²)`); copying `ρ̄₂` in gives a world whose drift and
  covariance disagree, silently.

### The autocall, TARF and barrier pricers

- **The energy and commodity floating legs concatenate their resets the way the rate legs did**
  (2026-09-10): `pricing.py`'s two `torch.cat`s of known against forecast resets off a
  `ForwardPrice` curve refuse a static curve in the words the rate legs refused with until
  `utils.concat_resets` — one call at each site, unmade because no document in the packs reaches
  it and an unmeasured change is not a fix.
- **`EquityPriceVol` under `Sticky_Strike` cannot reach a fixing strip** (2026-09-08):
  `calc_moneyness` returns the bare strike for a parametric (`Skew`/`SVI`) surface, one number with
  no fixing axis, and `forward_vol_strip` indexes it on a fixing axis it does not have — at one
  reporting row an `IndexError` at `pricing.py:425`, at more than one the reshape one line below
  dies with `shape '[1]' is invalid for input of size N`, N the block's row count. The desk's book
  declares `Skew` with `Sticky_Strike` on every equity surface, so its ten binary-leg and autocall
  skips ARE this row (reproduced on one leg, 2026-09-11): a book of skew-parameterised surfaces
  cannot price a discrete barrier, an accumulator, a TARF or an autocall. The fix is a broadcast
  of the moneyness onto the fixing axis at the one site that knows the strip's shape.
- **A second consecutive coupon whose window is wholly observed reads the first one's last fixing**
  under `'Spot'` on the OSS arm — the same staleness its prefix already carries; no document here
  reaches it. And a lagged block's terminal rows price the final coupon crisply off a fixing the row
  has not reached (the campaign book's 2028-06-03 and 2028-09-02 rows read the 2028-09-08 fixing).
- **The OSS pricers' European leg still branches on the family in two places** (a step count and
  a scalar carry against the walked block).
- **The collateralised autocall CVA delta is the kernel flux estimator's own variance**: under a
  zero-threshold CSA the correction supplies 2.5× the pathwise term, and it scatters with the path
  count — 51.2% short at 256 outer paths, 18.2% over at 1,024, 10.2% at 2,048; GBM under the same
  CSA reads 11.8% short at 2,048. The path count, not the deal. Beside it the float and the
  terminal put reach no `cash_settle`, so `Results['cashflows']` carries the coupons alone.
- **The TARF's target pin** fires on 27–61% of paths and is 27% short uncorrected; exact behind
  `Branch_And_Weight: 'Yes'`, the crisp default keeps the declared blindness (estimator 13%
  bandwidth spread, oracle 8.9% flatness — neither better than ~10%).
- **Seasoned TARFs with pre-base settlements discard the settled fixing outright** (bit-identical
  to deleting it, so the deal prices against its full original target), and **a TARF valued between
  two settlements marks NaN** on every tree measured. No fixture reaches either.
- **`pv_MC_ExtendableForward`**: the settled-cash channel under a CSA is not registered (+0.25% /
  −0.02% / +0.26% / +0.03% across four amplifying documents, against ladders that resolve no finer),
  and the rolling backward pass carries a one-signed Gauss–Hermite smoothing bias over the relu kink.
- **`pv_partial_barrier_option` rebate settlement completeness** is audited off-gate and ungated,
  for want of a collateralised partial-barrier document.
- **`Boundary_AAD_Window_Touch`** decides the sign and its magnitude is unestablished (−2.2467692
  registered against +0.5207422 unregistered, on an oracle that scatters 88% of its own median).
- **The bandwidth plateau** holds at 16,384–20,480 paths over 0.005–0.08 (correction spread 2.41% /
  3.87%, CVA delta 0.60% / 0.24%); at 2,048 paths the correction falls monotonically 23.76%.
  Acceptance names 32,768 and that re-read is pending. The correction's *scoping* is not
  mutation-gated — a mis-scoping mutant has no public seam.
- **`pv_MC_Accumulator`'s boundary placement under the recompute node is unmeasured** — its latch
  is assembled off a node output, which puts it on that side by construction, but the
  dropped-cotangent reading has not been taken.
- **`calc_vol_adjustment`'s analytic consumers** (`pv_barrier_option`, `pv_one_touch_option`,
  `pv_discrete_asian_option`) adjust the vol only, so a compo barrier, one-touch or asian prices
  half-adjusted without raising. The compo smile coordinate is decision 2.
- **A sibling fallback in a deal's `calc_dependencies` may name a factor discovery never fetched**:
  safe at 34 `Discount_Rate ← Currency` sites (the `InterestRate` comes transitively), the one
  cross-leg instance fixed.

### The engine

- **The privileged surface's dtype differs across processes** (2026-09-10): the LogVar2FJ
  outer publishes `(ell, s)` in the job's dtype while `LogOUSpotModel`, `MarkovHMMSpotModel` and
  `GARCHSpotModel` cast theirs to float32 — inert while `HedgeMonteCarlo` runs at float32 whatever
  the job says; a float64 `solve_hedge` would hand the critic one float64 block among float32.
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
- **The exposure profile is reported undeflated**, `Deflation_Interest_Rate` applied only inside the
  CVA/FVA scalars, so a deflated expiry-row EPE cannot be read from the tables; publish `Dt_T`
  beside `mtm`.
- **`Credit_Monte_Carlo.report` does not frame a book whose only deal folded to a static root**
  (a `(1, 1)` root against the `(T, B)` grid); a lone SKIPPED deal meets the same failure. And **a
  book whose deals reach no stochastic factor, or no date after the base date, dies unnamed**
  (a zero-wide random block; an empty `max()`).
- **`Hessian: 'Yes'` with `Gradient: 'No'` is a silent no-op**, and the Silverman bandwidth is per
  batch, so `Simulation_Batches > 1` oversmooths against the run's true path count.
- **`HullWhite2FactorImpliedInterestRateModel.precalculate` reads `Lambda_1` off a `Price Models`
  block an implied model does not need**, so omitting it raises a `TypeError` naming neither field
  nor factor; `FXVolSurfaceParameters` subscripts `point['Timestamp']` the same way.
- **`create_market_swaps`' `Distribution_Type` lives on the surface**, which the Bloomberg emitter
  does not author, so a lognormally-declared factor gets a lognormal fit of normal quotes — the two
  conventions are 9.7–11.4× apart in premium.
- **36 of 148 declared `.field.get` sites disagree with their declaration, three fatally**
  (decision 3); `StructuredDeal.post_process` reads `'Net Cashflows'` where `Net_Cashflows` is
  declared, so the declared key reaches no read.
- **Solved accrual strikes moved across a landing on documents carrying no LogVar2FJ** — the GBM
  TARF's zero-cost strike 15.32196559 → 15.31624884 between 48f4779 and 7ed3faf, up to 3.7e-4 and
  15× the runner's 2.5e-5 MC floor, while the same documents at a fixed strike are hex-identical.
  48f4779 is in no checkout any more, so the move is unpinned.
- **A `Market Prices` block with no quote table raises a `TypeError`** where it raised a `KeyError`
  (the completed blank `'null'` iterated as a string); one shared `quote_table` refusing by name.
- **`config.CustomJsonEncoder`'s `.DateOffset` string** takes its key order from a set iteration
  for a multi-unit period (`'6M2D'` or `'2D6M'`, 4:1 over five processes). Both parse back; what is
  not byte-stable is a written market-data file and any hash over it.
- **Two plan-hash pins** (`platinum_hedge_shipping.json`, `policy_test_simulate_only.json`) have
  hashed differently since 91c29de; whether that is a declared plan change or a values-plane field
  leaking into the plan is unclassified.
- **The Jupyter write path**: `set_value_from_widget`'s hardcoded whitelist raises on `Names`,
  `Sampling_Data_*` and `Barrier_Dates`, and fourteen output-shaped descriptors have no widget.
  Superseded for viewing by the web UI.
- **`gates/pricer_branch_census.py`** reads 59 unexecuted arcs at 1ed927a, not re-taken since.
- **Ungated since the 2026-08-21 purge** — five modules named on
  [Conventions](conventions.md#what-holds-today-and-what-the-purge-left-open), the
  already-hit barrier leg's value the expensive one.

## Decisions waiting on the desk

Nothing here is blocked on work. Numbers are stable — commit messages and the model pages cite
them — so closed decisions (4, 13, 15) keep their numbers and are not listed.

1. **The per-fixing smile read.** Sticky-forward moneyness or the deal's declared moneyness; both
   defensible, one can be the pricer's own quote. A switch, not a revert, with the six removed gates
   rebuilt.
2. **The compo smile coordinate**, undeclared because every fixture is flat. Same class as 1.
3. **The 36 disagreeing `.field.get` sites** (three fatal): hold a surviving fallback to its
   declaration, or leave the reads as they are. Enumerated in `tests/test_declared_defaults.py`.
5. **Two rates-emitter questions**: an OIS block is ~14 MB live (~26,000 authored floats on a 30Y
   strip) — accept it or build a term-authored variant; and neither side rolls a business day (a 2Y
   USD OIS pays on a Saturday).
6. **The HW2F α-seed's worst benchmark**: the honesty reprice reads −6.25% against the retired
   seed's −4.64% while rms improved 2.71% → 2.39% and the outside-3% count fell 10 → 3. One order
   statistic, anti-correlated with the fit; owner's eye wanted.
7. **PFE vs CVA measure policy.** CVA is a Q-expectation wanting the market-calibrated outer; PFE a
   P-quantile wanting a historically-estimated one, the pricing kit staying market-implied. One run
   reports both off one outer measure, so a book wanting each in its own measure runs twice under
   two `Model Configuration`s.
8. **`get_implied_correlation`'s two single-caller wrappers**, held against the
   no-abstraction-ahead-of-a-second-caller rule until a third correlation pair appears.
9. **Flagged, not authorised**: `runtime`'s free functions over the hedge bundle (two clusters,
   `_UTILITY_OBJECTS` duplicated) and `DealStructure`'s recursions — the shape Conventions calls a
   class waiting to happen.
10. **`Boundary_AAD_Window_Touch`'s magnitude.** The switch decides the sign; `add_grid_dates`
    landed, so the enriched fixture and the re-measurement are now possible.
11. **The `Branch_And_Weight` default.** The family question is closed (the surviving spot model
    hands each fixing interval its own Gaussian block law); what remains is an
    averaging-falls-back-to-crisp rule, since the averaging arms refuse under the switch. Values
    re-mark within their own MC noise at 12–23× less variance; the greeks are the prize.
12. **The correlation as a leaf.** `Correlation` is a `DimensionLessFactor` and mints no leaf, so a
    quanto's correlation delta is reported as a CRN bump (`Correlation_Bump`, 0.025: −22.42m ZAR
    per unit of ρ on the desk's NKY V2, flat to 0.003% between half-widths). The leaf is three edits
    with a tree-wide blast radius — every document carrying a correlation gains a `Greeks_First`
    row and a Hessian row and column, GBM ones included. The bump is the leaf's oracle.
14. **`Prices` in process: warning or refusal.** Mandatory on the multiprocessing path; a refusal
    costs an edit at about twenty test and gate sites that write `{'FXVolSurfaceParameters': {}}`.
16. **The nominal leverage weight.** `Leverage_Prior_Weight` 0.02 reads three to four quote rows on
    these ladders, not one, because it assumes a 0.2 quote weight and a 0.1 standard error no
    ladder states; a block's own `Leverage_Prior_SE` / `Leverage_Product_Prior_SE` supersede it
    where declared. At a fitted `σ_s` of 5 it reads several quote rows, and at the class-default
    tier it is what NKY cannot carry. A history's `ρ_s` −0.20 ± 0.09 and product −1.05 ± 0.67 move
    the desk mark to −41.9m through those rows: the estimator's leverage is the single most
    consequential number it produces, and its SE is a sampling error, not a desk's spread.

## Designed, not built

- **Whether the Hull-White solve should scale its steps by the Jacobian's columns** (2026-09-14).
  The LogVar2FJ fit runs its least-squares stage with each parameter's step scaled by the size of
  its own Jacobian column, which is the better-conditioned solve; the Hull-White chain does not,
  and its backward forms that scaled matrix for itself, so nothing is wrong today. Switching the
  chain on to the same scaling changes where the solve stops, so every Hull-White fit in every book
  moves by a small amount. It is a solver-tuning decision, not a defect, and it wants its own
  reading before it is taken: iterations, the stationarity norm at the stopping point, and what
  the marks do, on the four-quote fixture and one desk ladder.
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
  the put barrier, and the `BarrierIsHit` read at `pricing.py:4807` (it tests `is not None`, so it
  fires on `'No'`) retires with it. The TARF's and accumulator's decisions-remain arm: folded
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
fixture (SP's annuity-freezing bias −0.13 to +2.17 bp against the MC's own numeraire bias 0.6–3.0 bp,
systematic); **stationarity** — `‖J'r‖` at θ\* is 8.63e-7 on the analytic residual inside
`Stationarity_Tol`'s 1e-3 default, against 3.16e2 on the MC quartic; **determinism and cost** — two
analytic solves at one seed agree to the bit and the four-quote chain is 13.4 s against 75.1 s; and
**the quote side exists** ([the analytic quote side](quote_sensitivities.md#the-analytic-quote-side)).
`Monte_Carlo` is unchanged to the bit and remains the oracle. The α→0 series branches, the
declared `ALPHA_SEED = (0.5, 0.05)` and the domestic-measure correction are in
`tests/test_hw2f_analytic.py`.

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
  registry read as data; the dynamic half is `artifacts/reach/document_map.json`, every job JSON
  under `tests/fixtures/` and `artifacts/` run once under a tracer, keyed to the engine commit
  (`STALE` otherwise; `--build-map --repo <clean checkout>`, ~16 min). What it cannot see:
  string-keyed dispatch outside the registries, callables passed as values, virtual dispatch out of
  an inherited body, a branch no data takes, a document over the 180 s cap, `derivus_bloomberg/`.
  At e475bee: 852 of 2,133 symbols executed by some document; no document reaches 40 of 50 deals,
  24 of 34 pricers and 3 of 8 bootstrapper families.
- **Which tests a change reaches**: `gates/impacted.py --dirty --run` joins an execution-coverage
  map (built at a campaign boundary) with a static fixture map; file-granular, fails open loudly;
  `derivus/__init__`, `utils`, `calculation` and `conftest` are whole-suite modules by construction.
  The full suite runs at campaign boundaries with the tree held still.
- **The standing hex gates every landing runs**: `artifacts/lv_nig_20260907/hexcheck.py` (4,180
  floats over 16 GBM and Hull-White documents, diffed by `hexdiff.py`) and the crisp GBM TARF
  `artifacts/autocall_model_validation_20260904/campaign/tarf_hex.py` (`-0x1.2c48f36318e38p+5`).
- **The LogVar2FJ module is `tests/test_logvar2fj_json.py`** (2026-09-10): 32 gates over
  a synthetic world (`tests/fixtures/data/logvar2fj_world.json`, one index quoted in EUR on a USD
  book, a five-expiry skewed ladder, a GBM sibling) — the GBM limit at 1.3e-16 and through the CVA,
  the flat-surface residual at 1.4e-12 vol points, the calibration's contract (five ATM pillars
  under 1e-10, wing RMSE 0.789 against a 1.0 bound), the retired declarations refusing by name,
  the on-guard flag on the factor and in `Stats`, `Model_Priors: Off` bit-identical to its banked
  21 floats, the quanto arm at 0 ULP against its single-currency twin at ρ = 0, the reserve
  composed to 1e-9, the four sub-factor rows (a declared 0.3 on all four realises 0.2040 on returns against
  0.1520 with the sibling on GBM, 6.6 path-level se apart), and 76 floats over six repo documents hex for hex. About five minutes on the card, 213 s
  of it one shared five-expiry fit; every gate's killing mutation went red except the wing-RMSE
  row, whose mutation stalls the fit and whose bound is set at 3.2× below the unfitted seed. Two
  fixture facts: the banked floats are the CARD's (device reductions; a CPU-only box re-banks),
  and the repo's only autocall fixture has ONE fixing at maturity, so a vol-strip term-structure
  error is invisible to it (the cumulative variance × 1.000001 leaves both autocall hex rows
  green).
- **A fixture must not zero the quantity its gate is sensitive to** — the checklist and the
  plugin are on [Conventions](conventions.md#fixture-degeneracy). A mutant that survives a gate
  means the fixture is wrong, not that the code is right.

## Tidy-ups

- `gates/reach.py --dirty` died on the Windows box decoding `git`'s output as cp1252 (2026-09-08)
  and ran clean there on 2026-09-10; if it recurs, decode the diff as UTF-8.
- `gates/impacted.py --dirty` fails open to the whole suite on a fixture the map has not seen and
  on a `.md` at the repo root, so a lane that adds a fixture cannot use the selector until the next
  boundary run rebuilds the map.
- `gates/reach.py`'s document map is stale in substance as well as in commit: of the 43 autocall
  documents it names, 31 no longer load on the head (16 carry the Poisson-era factor block, 14
  the retired component family, one is gone), so a bit-identity claim over "every document the
  map names" is over the 12 that price. Rebuild the map (`--build-map`) at the next campaign
  boundary.
- `derivus_jupyter.set_repr` raises on any multi-column Table outside a four-name allowlist, which
  now includes `EquityBarrierBinaryOption.Barrier_Dates` and `QEDI_CustomAutoCallSwap.Coupon_Observations`;
  loading, pricing, the generated docs and the MCP descriptors are unaffected.
- `artifacts/logvar2fj/`'s second spelling no longer imports (it walks the Poisson residual); delete
  the pack or re-spell its G-gate scripts against the NIG residual.
- Inline comment density: ~12 blocks of 4–11 comment lines from the boundary-correction work
  (`pv_discrete_barrier_option`'s hit-mask and rebate blocks, `sim_spot_oss`'s terminal digital,
  `net_from_gross`); house style is 2–3 lines.
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
