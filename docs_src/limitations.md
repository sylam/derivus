# Model Limitations

What derivus is known to get wrong or leave approximate, by how much and what would fix it, and the
modelling choices it leaves to a desk. The page is for a model validator coming to the library cold.
The methods a validator prices against are on the Theory, Valuation and Bootstrapping pages
([the LogVar2FJ fit](bootstrapping/fx_and_equity.md#logvar2fjmodelparameters) among them), and the
tests that hold each rule are named on the developer page that states it. A row leaves this page
when the fix that closes it lands; what was built, and the numbers it was measured to, is in the
commit messages. What is designed but not built, and what the record, the service and the engine's
speed owe before the next release, is on the [roadmap](developer/roadmap.md): this page holds only
what moves a number a validator reads.

**Reading a row.** The bold lead says what goes wrong; the text says what the piece is for, how big
the error is and what would fix it. A size is a measurement, stated with what it was measured on,
or the row says UNMEASURED: a limitation without a number is absolution, not documentation
([Conventions](developer/conventions.md#unification-siblings)). S, M and L size the fix, never the
error. A date in brackets is when the row was found.

## Known limitations, open {#known-limitations-open}

Grouped by where a validator meets them.

### The xVA outer and the correlation {#the-xva-outer-and-the-correlation}

- **A bucket knot inside a scenario interval gives the second piece an uncorrelated mixer**
  (2026-09-11). The LogVar2FJ outer takes one correlated normal per scenario interval for its
  residual mixer; where a calendar bucket knot splits an interval, the second piece's mixer comes
  off the process's own stream (logged at INFO). UNMEASURED: no fitted factor on hand has two
  buckets. Separately, past 8.3 standard deviations (one draw in 1e16) the mixer's distribution
  function rounds to 1 in double and the inverse-Gaussian root returns its bracket's top; named,
  not guarded.
- **The Hull-White 2F quanto drift assumes an FX covariance the paths carry only where it is
  authored.** The `Correlation.FxRate.<pair>/InterestRate.<currency>` price factor is the desk's
  mark of the curve against the currency, from which the bootstrap derives
  `Quanto_FX_Correlation_1/2`, the drift's change of measure in each factor's own Brownian; the
  paths correlate the FX with the curve through the `Correlations` rows on
  `HWImpliedInterestRate.<curve>.F1` and `.F2`, innovation correlations the bootstrap does not
  write. [The process](theory/interest_rates.md#hullwhite2factorimpliedinterestratemodel) states
  the rows that carry what the drift assumes. Without them, on the composition gate's world (a 0.8
  mark, a 55% FX vol, 6M and 1Y expiries into 1Y swaps, 32,768 paths), the deflated expected
  positive exposure at expiry reads 12.2% and 16.6% of the analytic price above its reading with
  them, 10 and 12 standard errors, while the payoff-free numeraire identity moves within its
  noise; near linear in the mark times the FX vol, 1.2% and 1.5% at a 0.3 mark and a 15% vol.
- **The quasi-random draws ignore `Random_Seed`** (2026-10-11). One Sobol sequence at a fixed seed
  (`Calculation_State.QUASI_SEED`) feeds the inner block, the Markov-regime, GARCH and basis-linked
  spot models, the LogVar2FJ knot mixers and the hedging fork (unless `Inner_Draws: random`). A
  study across seeds cannot see that error, and the inner block's error is common to every outer
  path rather than averaged over them. UNMEASURED.

### The calibration {#the-calibration}

- **The default LogVar2FJ fit prices its vanillas on a surrogate clock** (2026-09-14).
  `Vanilla_Pricer: Quadrature` replaces the variance clock with a two-moment lognormal, so θ\* is
  fitted to a stand-in and then used by pricers that walk. On the fifteen-quote ladder of
  `tests/fixtures/data/logvar2fj_world.json` its worst quote sat within 0.085 vol points of a
  32,768-path walk and 0.098 of a 1,048,576-path one. The walk's own ratio-estimator bias, 0.36 vol
  points between 8,192 and 32,768 paths, rides the forward-start rows at the default `Paths` of
  8,192. No test re-checks either; UNMEASURED on other ladders. `Walk` is the alternative.

### The autocall, TARF and barrier pricers {#the-autocall-tarf-and-barrier-pricers}

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
- **The American equity option's approximation is to be retired, not patched** (2026-09-16).
  `pv_american_option` (Bjerksund-Stensland) never calls `calc_vol_adjustment`, so a composite or
  quanto American prices as the local asset. UNMEASURED, as is its error against a lattice. The fix
  is a different pricer.
- **The closed-form barrier floors its vol at 5%** (2026-10-11). `pv_barrier_option` (the FX barrier,
  and an equity barrier without `Barrier_Dates`) lifts any lower vol to 5% before pricing and before
  the monitoring shift. UNMEASURED on low-vol pairs.

### The engine {#the-engine}

- **A rate leg reads `Advance` as `Standard` and `Discounted` as `End`** (2026-09-30). Measured on
  a two-year quarterly swap of a million on a curve rising from 2% to 6%: `Reset_Type: Advance`
  fixes at the period start, so if it means a reset one period earlier the swap is 77.3 bp of
  notional high; `Payment_Timing: Discounted` values from the period end, as the FRA's does, but
  books its cash there rather than discounted at the start, and against $L\alpha/(1+L\alpha)$ paid
  at the start the floating leg is 0.50 bp high under a 25 bp basis, exact on one curve.
  `Index_Publication_Calendars` is unread, so a leg resetting daily fixes on every calendar day:
  0.03 bp low on a one-year averaged leg, exact compounded.
- **A fitted structure leg's two-way charge uses its lognormal vega** (2026-09-21). A leg under a
  fitted spot model publishes no FX vol quote sensitivity, so the charge reads the same leg as a
  lognormal (`spread_source` says so): a vega a desk could hedge, but not the quoted price's.
  UNMEASURED, and not measurable from a quote: the fit's own `dV/dq` needs `Quote_Sensitivity`
  through `LogVar2FJModelParameters`, the calibration live in the session.
- **A floating cashflow list refuses its settlement** (2026-09-22). `CFFloatingInterestListDeal`
  refuses a stated `Settlement_Date`, `Settlement_Amount`, `Repo_Rate` or `Settlement_Rate` by name
  at booking, so a forward-settled floating note cannot be booked; nothing is mispriced. Reading
  them is a pricer change, not a diary row: the leg carried to the settlement date on repo, the
  amount paid there and the cashflows before it dropped, as `pricing.pv_fixed_cashflows` and
  `utils.TensorCashFlows.fixed` do for the fixed list, through the plain, cap and floor pricers: M.
- **Rating-dependent credit support is read at its first row** (2026-10-11).
  `utils.CreditSupportList.value` returns the first row's threshold, minimum transfer or independent
  amount whatever the counterparty's rating, in the engine and the spine's collateral call. Exact
  for one row; UNMEASURED otherwise.
- **Declared fields not yet read** (2026-10-11), each a feature declared ahead of its build and kept
  so: `Average_FX` (floating energy deal, energy option) will average the fx and the reference
  separately and multiply the two averages in place of the sum of products; `Net_Cashflows`
  (structure) will convert each leg's cashflows to the structure's currency and net them;
  `Risk_Premium` (GBM term-structure model) is the market price of risk a real-world measure adds to
  a risk-neutral calibration; `Distribution_Type` (PCA rate model) will offer a normal process
  beside the lognormal; `Dividend_Timing` (equity swap leg) will price `Continuous` apart from
  `Terminal`; `Commodity_Collateral` (netting set) will join the collateral assets. Until each is
  built a stated value prices as the default, unrefused.
- **A factor read between scenario dates is interpolated** (2026-10-04). A fixing off the grid reads
  the factor linearly, 5e-7 of an FX level on a monthly grid; a price-index month in force on no
  scenario date reads between prints, a linker's payment row 1e-6 to 6e-6 high on a quarterly grid.
  Exactness wants the day on the grid.
- **A host curve read depends on its batch position at the last bit** (2026-10-05). Fused blends
  round differently in vector lanes and scalar tails, so a CPU read moves by an ulp with the batch
  width and the thread count (within 4 ulps, bit-identical on a GPU, seen in no end-to-end result).
  A replay across boxes compares for equality, so a home replaying host-priced results needs a
  declared tolerance policy.
- **A SIMM vega vertex on a Skew surface moves its base** (2026-10-07). A vertex bump inserts a
  pillar where the surface lacks one, read off its own interpolation. Explicit and Malz surfaces
  price unchanged, but no smile at an inserted pillar equals a Skew surface's expiry blend (14.5 vol
  points apart deep in one wing), so two of five structures move, by 4.8e-4 and 1.5e-2 of the
  trade. The rows agree with each other, not with the book's base. S-M: the tent through the blend.
- **A one-point vol move carries cross-vertex convexity** (2026-10-07). On the autocalls the twelve
  vertex rows differ from one parallel point by -0.8% to +13% (0.1% to 1.2% on FX), stable across
  seeds and path counts, a tenth of a point cutting it tenfold. It is the regulation's definition,
  so stated, not fixed. Rate vega, credit, commodity and inflation sensitivities refuse by name.
- **The replay tuple pins neither build nor device** (2026-10-11). `engine_version` is a hand-set
  string, no commit, torch, CUDA or device is recorded, and `Deterministic_Kernels` defaults to
  `No`, so bit-for-bit replay holds only on one build, stack and device, and for a GPU gradient only
  under `Yes`: on one device a base valuation's first-order greeks table moved run to run at 1e-14
  relative while every value was bit-identical (2026-10-15). UNMEASURED across devices beyond the
  host-curve row.

## Decisions waiting on the desk {#decisions-waiting-on-the-desk}

Choices the library leaves to a desk; nothing here is blocked on work. Commit messages and the model
pages cite these numbers, so closed decisions (2, 3, 4, 5, 8, 10, 11, 13, 15) keep theirs and are not
listed.

- **1. The per-fixing smile read**: sticky-forward moneyness or the deal's declared moneyness. Both
  are defensible; the choice lands as a switch, with the six removed gates rebuilt.
- **6. The Hull-White seed's worst benchmark**: the honesty reprice reads −6.25% against the retired
  seed's −4.64%, while the RMS miss fell from 2.71% to 2.39% and the count outside 3% from ten to
  three. One order statistic, anti-correlated with the fit; a desk's eye wanted.
- **7. Exposure versus credit-valuation measure.** CVA wants the risk-neutral, market-calibrated
  outer; PFE a real-world, historically estimated one. One run reports both off one outer measure,
  so a book wanting each in its own runs twice under two model configurations.
- **9. Flagged, not authorised**: the hedge runtime's free functions over the bundle (two clusters,
  a duplicated utility table) and the deal structure's recursions, the shape
  [Conventions](developer/conventions.md#registries-not-functions) calls a class waiting to happen.
- **12. The correlation as a leaf.** A correlation mints no leaf, so a quanto's correlation delta is
  a CRN bump of 0.025, flat to 0.003% between half-widths on an index autocall. The leaf is three
  edits with a tree-wide blast radius: every document carrying a correlation gains a first-order
  row and a Hessian row and column. The bump is its oracle.
- **14. `Prices` in process: warning or refusal.** Mandatory on the multiprocessing path; a refusal
  means editing about twenty test and gate sites that write an empty family entry.
- **16. The nominal leverage weight.** `Leverage_Prior_Weight` 0.02 counts as three to four quote
  rows on the ladders measured, not one, since it assumes a quote weight and a standard error no
  ladder states; a block's `Leverage_Prior_SE` and `Leverage_Product_Prior_SE` supersede it. It
  matters: a history's leverage of −0.20 ± 0.09 and product −1.05 ± 0.67 move a desk index
  autocall's mark by 11.7% through those rows, so the leverage is the estimator's most consequential
  number and its standard error a sampling error, not a desk's spread.
- **17. Jacobian column scaling in the Hull-White solve** (2026-09-14). The LogVar2FJ fit scales
  each step by its Jacobian column; the Hull-White chain does not (its backward forms the scaled
  matrix itself), so nothing is wrong. Switching moves every Hull-White fit slightly; read
  iterations, the stationarity norm and the marks on the four-quote fixture and a desk ladder first.
- **18. Netting a package's legs before the two-way charge** (2026-09-21). Each leg pays its own
  spread, so a collar is charged as two tickets. Netting per pillar cuts the gate collar's edge by
  85% (the ATM row 95%, the butterfly 88%, the risk reversal not at all).
- **19. One `settle` verb for the back office** (2026-09-30). Settlements, confirmations and
  collateral share it, so each worklist shows the others' items and confirmations can pay or post
  collateral; separation of duties wants narrower verbs.
- **20. The recompute segment length.** The LogVar2FJ walk checkpoints every 21 steps, trading
  memory for a second forward pass; the mixer's per-draw checkpoint raised peak memory by about a
  fifth. The residual's granularity is the dial if that headroom is wanted.
- **21. Who may move the market, the date and the file** (2026-09-30). `POST /book/market`,
  `/book/date`, `/book/configure`, `/book/curve`, `/book/bloomberg`, `/book/securities`,
  `/calculations`, `/book/setup` (admitted under `validate`, yet it re-solves the market) and a
  delete take no `actor`, so under a capabilities document any client can move the board a close is
  declared over, roll the date or drop a deal. The tick is the deployment's poll path and
  `delete_deal` takes no seat by design, so which verbs take a seat is a desk's call before the
  build (M-L: an `actor` and an admission on each, `mark` for the market and the date, `book` for
  the file).
- **22. A checkpoint key rotation trusted on chain integrity alone** (2026-10-11). `verify_home`
  accepts any `checkpoint_verifying_key` declaration on an intact chain without asking that the key
  then in force signed it, and the signing key sits in the hub's `keys/`. Against the hub's own
  operator, tamper-evidence holds only with a genesis hash and key obtained out of band, or an
  agreeing replica ([Spine](developer/spine.md#trust-limits)). Whether a rotation must be signed by
  the key it retires is a desk's call.

## Hull-White 2F, decided {#model-punchlist}

`Objective: 'Analytic'` (Schrager-Pelsser) is the default because it is accurate enough, stationary,
deterministic and differentiable in the quotes. On the identified fixture it sits inside one Monte
Carlo evaluation's noise at 22 of 25 benchmarks, its annuity-freezing bias (−0.13 to +2.17 bp) below
the Monte Carlo's own numeraire bias (0.6 to 3.0 bp). Its residual is the vol difference itself, so
`‖J'r‖` at θ\* reaches 8.63e-7 against `Stationarity_Tol`'s 1e-3, where the Monte Carlo's squared
residual stops at 3.16e2. It simulates nothing, so two solves at one seed agree to the bit, and its
quote side exists
([the analytic quote side](developer/quote_sensitivities.md#the-analytic-quote-side)). `Monte_Carlo`
remains the oracle and reprices each analytic θ\* once. `tests/test_hw2f_analytic.py` holds the α→0
series branches, the declared seed pair and the domestic-measure correction.

**Warm start.** A block whose `HullWhite2FactorModelParameters` factor already exists skips the
basin search and runs the least squares from it (6 objective evaluations against 421 cold on the
four-quote block). It adds no re-marking event.

**Re-marking events.** A desk holding an older θ\* re-baselines or re-solves; carrying it forward is
neither.

- A foreign-curve θ\* solved before the domestic-measure fix re-solves to a different θ\*.
- A θ\* solved before 2026-09-02 re-marks on the seed and premium-clock change.
- A θ\* solved before 2026-09-17 re-marks on the basin search's generator (numpy's default in place
  of the legacy one, same seed): on the four-quote document the mean reversions move 14% and 8%,
  the correlation from −0.04 to −0.16 and the 1Y×1Y quote delta 4%, along directions four quotes do
  not identify.

`Quanto_FX_Correlation_1/2` and every locus recorded downstream of a fit move with θ\*. The Monte
Carlo's numeraire bias comes from the curve's tenor grid, not discretisation: adding 1D, 1M, 3M and
6M nodes collapses it from −1.6e-2 to −1.1e-3.
