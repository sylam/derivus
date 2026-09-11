# Roadmap

The live status of the library, written for a model-validation reader: what is known to be wrong or
limited and how big it is, what is waiting on a desk decision, and what is designed but not built.
What was built, and every number it was measured to, is in the commit messages — `git log` reads
as the ledger, one landing per commit — and the long-form record that stood on this page until
2026-09-10 is in the tree at 978861f. The model pages carry the numbers a validator prices
against: [Market Prices](market_prices.md#logvar2fj) for the LogVar2FJ calibration,
[Structures](structures.md#model-worth) for what the model is worth on a book and how the xVA
outer is chosen, [Calc Lifecycle](calc_lifecycle.md) for the engine.

## Known limitations, open

Grouped by where a validator meets them. Every row carries its measured size or says it is
unmeasured — a limitation without a number is absolution, not documentation
([Conventions](conventions.md#unification-siblings)). Dates are when the row was found.

### The LogVar2FJ calibration

- **The index class defaults are a shape this model cannot carry on NKY** (2026-09-10). At the
  class leverage (`ρ_s` −0.7, product −1.9, SPX's numbers) NKY refuses at production `Cap_A`
  (4.2e-05 of path-days within `5·Cap_Beta` against the 1e-05 allowed) and at `Cap_A` 6 lands
  `σ_s` on its 5.0 box with `ρ_s` −0.397. NKY's own implied pair (−0.529 / −1.62) lands clean, so
  the emitter has to write per-name numbers; a bigger cap is not the fix.
- **A declared standard error tight enough is a pin, and the guard says so** (2026-09-10). NDX at
  the regression's own 0.015 reads 138 quote rows on `ρ_s` and is flagged; NKY at 0.025 reads 38.5x
  and is not. Whether a 1,149-day regression's sampling error is the right spread for a Q-measure
  prior is a modelling question; `Leverage_Prior_SE` is where a desk states its own.
- **`Residual_Skew_Share_Sd` 0.2 is asserted, not measured** (2026-09-10). The wings win by about
  one standard error on every index ladder (−0.29 to −0.49 against −0.5). The estimator's own
  `β^P/α^P` spread on an uncontaminated history would measure it, and no index history here is
  uncontaminated (NKY's `C_Eff` 0.9967 against `c` 0.2671).
- **`Alpha`'s prior row is on bucket 0** while the share's and the leverage's are per bucket
  (2026-09-10). Nothing moves today — every book ladder is one bucket — and a multi-bucket
  `Bootstrap` fit with priors on is unexercised.
- **The NKY chain block's vanilla-only objective is bimodal** (2026-09-08): thirteen fits over three
  seeds and four path counts land between RMSE 0.976 and 1.016 but split into `Alpha` 0.6–2.1 or
  7.6–8.0, and `Pseudo` 8192 lands in one mode at seed 1 and the other at seeds 2 and 3. A
  same-answer gate on that ladder proves determinism, not agreement; the forward block is the
  identification it lacks.
- **`LVFit.cap_level` is not re-evaluated after the polish** (2026-09-09). A polish that raises
  `Sigma_S` leaves `Cap_A` where a smaller one put it: NKY seed 1 converged and then refused at
  6.6e-05 of path-days within the cap; declaring `Cap_A` 9.0 lets it land. Re-derive after the
  polish, or let the guard raise it once and re-walk.
- **`Skew_Reserve` is one number per calculation**, composed from the one gradient
  `Base_Revaluation` takes, and read at the two block ends nearest the declared `Forward_Tenors`
  (NKY: 0.51y into 2.23y for a declared 6m-into-6m) — honest, and not the tenor a 2y autocall is
  exposed to. A per-deal reserve needs a per-deal gradient; the declared tenor needs a post-fit
  reading on its own grid.
- **`Forward_Smile_Source: Reference` is declared and unexercised** — no reference model is wired
  and the report line says so.
- **`Residual_Law: Gaussian` reports a greek at leaves the model never reads** (`Alpha`, `Beta`
  filled with their defaults, an identically-zero row each). Dropping them would make the leaf set
  mode-dependent, which is why it is not done.
- **`LV_CHECKPOINT_STEPS` 21 was measured for the walk alone**; the mixer's per-draw checkpoint
  beside it took peak memory 8,392 → 9,990 MiB at 2,048 × 2,048. The residual's granularity is the
  dial if that headroom is wanted.
- **The vendor's implied-vol grid answers at 30, 60 and 90 days only**, so the long end of every
  equity fit is the file's own surface or the listed chain; the desk's NKY mark moves 4.6% between
  a chain-only fit and one carrying the file's 2.74y ATM. Whether the workstation is entitled to a
  longer grid is one question to Bloomberg support, the owner's.
- **Three dials stayed module constants**: `bootstrappers.ALPHA_SEED` `(0.5, 0.05)`, `xtol=1e-12`
  in `LVFit.solve`, and `Sigma_Knots`' ten-knot default grid.
- **Under `Sampling: Pseudo` the mixer uniform is 24 bits at float32** (2026-09-10, lane F):
  `torch.rand` at float32 is the draw, and widening it changes the generator's consumption and so
  the stream a float64 document reproduces; `1 − u` at the clamp margin carries 2.98% there.
  Production takes Sobol above 16 scenarios, where the uniform is drawn in double.
- **`lv_ou_path`'s integrating factor is range-limited at float32** past κT = 88 (2026-09-10):
  every caller keeps it safe — the walk's 21-step segments, the state variance in double — but the
  primitive would overflow a float32 caller. The fix is a chunked rescale, not a cast. Beside it
  the clock `A` is still summed across blocks in the job's dtype; widening it cascades into `G`
  and every pricer reading the law, a design change.
- **A ladder shorter than `Slow_Horizon` cannot reach the model's own flat limit** (2026-09-10,
  lane T): with no wing at 1.5 years the slow pair is pinned at the class default (−0.4, 1.0),
  which injects skew and convexity a flat surface does not want, so a flat 20% ladder fitted with
  `Model_Priors: Off` lands at 0.21 vol points RMSE on one rung and 0.44 on two rather than at
  zero, at a fitted `ρ_s σ_s` of −0.001 beside the pinned −0.400. Every ladder a short-dated FX
  desk quotes is such a ladder; the flat-limit gate needs one reaching 1.5 years.

### The xVA outer and the correlation

- **The LogVar2FJ outer is not designed for a netting set of several names, and the book does not
  use it as one** (the owner's ruling, 2026-09-09). The marginal law is clock-free; the cross-name
  law is not, because each name's mixer and its two variance shocks are private, so a declared
  correlation realises `E[√G]/sd(R)` of itself on the scenario interval (NKY 0.34 on a day, 0.70 at
  a year) and moves with the fit. The engine reports that share at INFO and never scales for it;
  the book runs the GBM term-structure outer with the LogVar2FJ pricer, and the LogVar2FJ outer is
  the opt-in for a single-name exposure or a PFE study. The four-sub-factor process below closes
  it. Beside it: `correlation_dilution` reads one bucket per interval and the uncapped law, and
  reads 8–10% high on quarterly intervals for want of the leverage's serial covariance.
- **A correlation declared under a process name no simulated factor answers to is silent**
  (2026-09-09): the lookup reads 0.0 for a missing pair, right for an undeclared one and wrong for
  one filed under a stale key. Name it at INFO.
- **The float32 bias of the LogVar2FJ outer at production size is unmeasured** (2026-09-10, lane
  F): the desk's NKY autocall under the LogVar2FJ outer reads −14.09% between float32 and float64
  at 32 outer × 256 pricing paths, reproduced to the digit on the base and the landed tree — two
  samples of a nonlinear walk, not rounding, since the two precisions take different mixer roots
  and walk different worlds. The 2,048 × 2,048 table over three seeds per precision is the
  reading (`artifacts/lv_precision_20260910/batch.sh`, resumable row by row; about 2.5 hours a
  tree on a free card); the expectation is that the gap is sampling and no bias survives.
- **`Correlations` cannot be authored in `ExplicitMarketData`** (2026-09-10, lane T): `Config`
  keys the section by a `(name, name)` tuple and builds it only on the `MarketDataFile` path, while
  `Context.load_json` merges an explicit section by `dict.update`, so a correlation written there
  lands under a string key that `get_cholesky_decomp` never looks up — a silent zero. Every
  correlated document needs a market-data file today.
- **Every correlation between a `calc_statistics` factor and a newer estimator is a business day
  out** (2026-09-07): `calc_statistics` indexes an innovation at the return's start date, the
  GARCH, HMM, basis and LogVar2FJ estimators at its end, and `calibrate_factors` correlates the two
  families as they stand. On one simulated series GARCH against GBM reads 0.009 and against the
  LogVar2FJ estimator's `eps` 0.335 (GBM −0.008) where the sibling was drawn at 0.6. One line in
  `calc_statistics` and a re-read of every banked `Correlations` block.
- **`calibrate_factors` reports a calibration class's own refusal as "Data errors in factor"**, and
  raises `AttributeError: 'NoneType' object has no attribute 'corr'` where every factor is skipped
  (2026-09-07). The same swallow one layer up: a bootstrapper family that refuses at CONSTRUCTION
  (the leverage tables' sign refusal) is caught by `Config.bootstrap`, logged, and the factor left
  unwritten, so a caller that does not read the log sees a bootstrap that succeeded and a
  `KeyError` at the first pricer that wants the factor (2026-09-10, lane T).
- **`Correlations` vs `save_params` author a quanto in different bases, unchecked**: `save_params`
  emits `ρ̄ᵢ = corr(dW, dWᵢ)` while the section's rows are the independent normals the Cholesky
  consumes (`a = ρ̄₁`, `b = (ρ̄₂ − ρ·ρ̄₁)/√(1−ρ²)`); copying `ρ̄₂` in gives a world whose drift and
  covariance disagree, silently.

### The autocall, TARF and barrier pricers

- **A coupon with no `Autocall_Thresholds` row prices a NEGATIVE trigger** (2026-09-11, lane A):
  the positional read fills a missing row with −1, so `K = −strike` and the coupon fires on
  every path, and the `min(tl.values()) <= 0` guard cannot see it because the −1 never enters the
  table. Since lane A the fixing-level read is `tl[c]`, so such a document skips on a `KeyError`
  naming the deal; it should refuse by name. Every booking in the packs has one row per coupon.
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

- **The privileged surface's dtype differs across processes** (2026-09-10, lane F): the LogVar2FJ
  outer publishes `(ell, s)` in the job's dtype while `LogOUSpotModel`, `MarkovHMMSpotModel` and
  `GARCHSpotModel` cast theirs to float32 — inert while `HedgeMonteCarlo` runs at float32 whatever
  the job says; a float64 `solve_hedge` would hand the critic one float64 block among float32.
- **A document's numbers depend on how many documents ran before it in the process**
  (2026-09-11, lane A): a GBM autocall CVA document moves 9,146 of its 12,250 floats — the CVA by
  12% — between running first and running forty-third in one process, at the same seed on the
  same tree, while two runs at the same position agree to the bit. Process-global random-number
  state that a job's `reset()` does not reset (the quasi-RNG batch counter the RNG-ordering note
  names). A document-set gate has to fix its ORDER as well as its list, and a mark quoted from a
  batch run is not the mark the document prices to alone. Measured on one document; the extent
  across the other draw paths is unmeasured.
- **`NettingCollateralSet`'s backward is nondeterministic on the GPU**: one gradient entry takes two
  distinct float64 values from bit-identical inputs (a reduction order, not a graph defect). It
  bounds how tightly any collateralised sensitivity gate can be pinned.
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
- **The HW2F solve's debug block writes `ZAR.aap` in the CWD** — inert today, an artifact where the
  no-artifacts rule forbids one.
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

- **The LogVar2FJ outer as a process the one-step logic applies to the PAIR** (2026-09-09). A
  step rolls four dice and the framework hands the process one, so the other three are private to
  each name, which is why they are uncorrelated across names. Ask for four sub-factors on the
  `PC1…PCn` pattern: the return innovation, the two variance shocks (each factor's node-to-node
  transition driven by one framework normal per interval), and the mixer drawn through its own
  quantile from a framework normal, `G = F_IG⁻¹(Φ(Z))`. The estimator returns the four innovation
  columns under the inverse map, so every marginal stays the fitted NIG and a historically estimated
  row needs no conversion. Measured by simulation at the Q-sized truth over 5,040 days with the
  return rows declared at 0.60: 0.017 realised as built, 0.100 with the mixers coupled by one
  uniform, 0.454 with all four rows at 0.60, 0.701 with the variance rows at 0.8 and the mixer at 1
  (se 0.009) — the return correlation is a function of the declared rows only once nothing in a
  step is private. A shared-mixer component alone (the earlier design) raises the pair ceiling on
  the Barclays grid from 0.344 / 0.209 / 0.216 to 0.433 / 0.435 / 0.366 — enough for two of the
  three declared pairs and not for SX5E/SD3E's 0.804.
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
- **The LogVar2FJ module is `tests/test_logvar2fj_json.py`** (lane T, 2026-09-10): 32 gates over
  a synthetic world (`tests/fixtures/data/logvar2fj_world.json`, one index quoted in EUR on a USD
  book, a five-expiry skewed ladder, a GBM sibling) — the GBM limit at 1.3e-16 and through the CVA,
  the flat-surface residual at 1.4e-12 vol points, the calibration's contract (five ATM pillars
  under 1e-10, wing RMSE 0.789 against a 1.0 bound), the retired declarations refusing by name,
  the on-guard flag on the factor and in `Stats`, `Model_Priors: Off` bit-identical to its banked
  21 floats, the quanto arm at 0 ULP against its single-currency twin at ρ = 0, the reserve
  composed to 1e-9, the correlation key (0.1379 realised against 0.3 × 0.4525 declared-times-share,
  0.5 se), and 76 floats over six repo documents hex for hex. About five minutes on the card, 213 s
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
