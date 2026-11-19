# Model Limitations

What derivus is known to get wrong or leave approximate, by how much and what would fix it, and the
modelling choices it leaves to a desk. The page is for a model validator coming to the library cold.
The methods a validator prices against are on the Theory, Valuation and Bootstrapping pages
([the LogVar2FJ fit](bootstrapping/fx_and_equity.md#logvar2fjmodelparameters) among them), and the
tests that hold each rule are named on the developer page that states it. A row leaves this page
when the fix that closes it lands; what was built, and the numbers it was measured to, is in the
commit messages. What is designed but not built, and what the code, the tests and the engine's
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
  function rounds to 1 in double and the inverse-Gaussian root reads its tail at the smallest
  double instead, 37.5 standard deviations out; named, not guarded.
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
- **A margin call's transfer decision carries no boundary term unless the document asks.** A
  minimum transfer amount makes the balance jump, a decision ordinary AAD differentiates as
  frozen. `Boundary_AAD_Margin_Calls` (default `No`, credit Monte Carlo only) registers each call
  for the boundary correction, at the cost of a counterfactual balance path a call; the value is
  the same either way. On one synthetic document of twelve sets under a minimum transfer large
  against their exposure (16,384 paths a seed) the unregistered spot delta reads 78% short of its
  bump on the CVA and 47% on the funding benefit. Registered at a bandwidth of 0.04 the CVA's
  reads within 1% of its bump and the funding benefit's 2.7% under, over 256 seeds; at 0.08,
  within 2% and 8% under. On four one-set documents at 0.04 the CVA's reads 1% to 5% under over
  48 seeds, the most where a set outlives its deal, at three standard errors. A call is met from
  the side that held alone, so the registered estimate is the noisier: at 0.04 a seed's standard
  error is 8% and 15% of the two deltas against a bump's 3% and 7%, and on a one-set document 2.3
  times that at the declared 0.01.

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
  The credit Monte Carlo registers a partial-time barrier's window touch for the boundary
  correction. On a monthly grid with six live decisions,
  five seeds read −1.915 registered at 32,768 paths against +1.318 unregistered; all 70 CRN readings
  are negative and the pooled oracle sits 0.5% from the registered delta. The ladder still moves
  13% to 35% over h ≥ 5e-4, so the magnitude is known to about a third.
- **The boundary correction is stable over its bandwidth only at the declared path count.** At
  16,384 and 32,768 paths six seeds resolve no dependence on `Boundary_AAD_Bandwidth` from 0.00125
  to 0.08 (within ±2% of the correction and ±0.5% of the CVA delta); at 2,048 paths the correction
  falls 24% across that range. That is a decision solved over its whole gap: a registered margin
  call, met from one side, reads lower as its bandwidth widens, the row above. The correction's
  scoping has no public seam a mutation gate could reach.
- **The American equity option's approximation is to be retired, not patched** (2026-09-16).
  `pv_american_option` (Bjerksund-Stensland) never calls `calc_vol_adjustment`, so a composite or
  quanto American prices as the local asset. UNMEASURED, as is its error against a lattice. The fix
  is a different pricer.
- **The partial-time barrier is not a number at a vol under 1%** (2026-11-14).
  `getpartialbarrierpayoff` multiplies each power of $H/S$ onto its bivariate tail, where the single
  barrier and the touch add the two in logs. An up barrier 12% from the spot reads NaN at a 0.5% vol
  in single precision and at 0.2% in double; every reading at 1% is finite.
- **A unit paid at the touch reads $\lambda$ floored at zero under a negative rate** (2026-11-14).
  `touch_root` floors the argument $\mu^2\sigma^2+2r$ of $\lambda$'s root smoothly at zero, binding
  below 0.005, where the exact value has no real root. It reaches a knock-out's rebate, the one-touch
  paid at the touch and the partial-time knock-out's rebate. Against the first-passage integral on a
  one-year touch 12% from the spot at a 15% vol and a 2% carry: exact at a rate of 0.1% and above,
  0.003% low at zero, 0.014% at -0.1%, 0.15% at -0.5%, 0.36% at -1% and 0.8% at -2%.
- **A commodity touch or barrier is watched on its scenario dates alone under simulation**
  (2026-11-18). The forward-price process publishes no bridge variance, so the touch probability
  tests each interval's endpoints and misses a crossing that returns. The trial's knock-out, rebate
  aside, under a Clewlow-Strickland forward at a 20% vol marks about 4% high nine months in on a
  quarterly grid against a daily one (four seeds of 4,096 paths: 3.8% on the card, 4.2% on the
  CPU, each past four standard errors), within noise at three and six months. The fix is that process publishing its fixing's variance rate.
- **The FX range accrual's simulated fixings carry no boundary correction.** A fixing after the
  base date is the path's own inclusive indicator, a step on outer state that `pv_fx_range_accrual`
  registers no `BoundarySet` for, so a credit Monte Carlo's sensitivities drop its flux on every row
  after a fixing; a base valuation has no simulated fixing and its derivative is exact. UNMEASURED:
  measuring it is a CRN ladder of a CMC delta on a row after a fixing (`tests/crn_ladder.py`).

### The engine {#the-engine}

- **The LogVar2FJ spot's filter reads the bar and the return as independent** (2026-11-03). The bar
  of day t-1 and the fix's return into day t share day t-1's afternoon, their innovations
  correlating 0.186 (se 0.035) on the platinum archive, so `LogVar2FJCalibration.step` counts part of
  one measurement twice and its posterior is overconfident by an UNMEASURED amount. The fix is the
  two measurements' covariance in the step.
- **The LogVar2FJ spot filter's Laplace step leaves its log variance low** (2026-11-03): by 0.017 of
  its posterior sd on the test world and 0.087 with bars as noisy as `Sigma_U` 1.5, so the mixed
  forecast reads realised variance 0.9% and 6% low. The fix is a second-order update of the step.
- **The HMM spot runs calendar time** (2026-11-03). `MarkovHMMSpotModel` steps every calendar row,
  its moments annualised on the row's calendar years and its transition matrix re-discretised to
  them, where the GARCH spot, the basis, the carry, the fixing bridges and the LogVar2FJ spot step
  once a market day and hold on a weekend or holiday: a world mixing the HMM with them moves it on a
  row the others hold. The fix is its emission and chain on `utils.trading_clock`'s market days,
  its belief filter holding on a row of none.
- **The hedging engine reads one underlying, commodity futures legs, one calendar spread rate and
  gross initial margin** (2026-11-04). `HedgeMonteCarlo` and its bundle take the primary commodity
  spot as the one underlying (`hedge_bundle.primary_spot`: the spot the utility scale, the step vol,
  the returns-state coordinates, the bundle's width and the diagnostic CSV read; a world whose
  primary is not unique is refused by name), its hedge legs are commodity futures deals, the one
  type declaring the `Contract_Size` a position scales a mark by (a leg of another type fails on
  that key), `Calendar_Spread_Bps` is one scalar rate on every listed pair, so a per-maturity spread
  ladder is inexpressible, and the initial-margin funding charge posts each leg's margin gross,
  over-margining a calendar spread. UNMEASURED: no document carries two underlyings, a leg of
  another type or a spread ladder.
- **A container in the hedging engine's liability is valued at nothing** (2026-11-18). The
  liability's marks and cash are accumulated off its leaves (`DealStructure.resolve_hedge_structure`),
  which runs no container's `post_process`, so a structure's legs and a price swap's children and
  settlement are left out. UNMEASURED, found by reading. The fix is the accumulation running each
  container's `post_process` as a valuation does: S.
- **The LogVar2FJ spot fit writes no standard errors** (2026-11-03). `LogVar2FJSpotCalibration`
  maximises on numerical gradients in float (1.3 s for a five-year fit against the history fit's
  AAD path's 32 s), so nothing downstream can weight its parameters. The fix is the AAD Hessian the
  history fit already takes.
- **A LogVar2FJ month world's start state goes stale under `--recal-months` above 1**
  (2026-11-03). The walk-forward chain stamps a world's `S0`, `Ell0` and `State_Cov` at its
  calibration and re-filters the GARCH and basis seeds to each later trade date, not the LogVar2FJ
  spot's, so the second and later months start from an older state. UNMEASURED; the fix is the
  LogVar2FJ filter run to the trade date in the chain's restamp.
- **`--carry-drift Yes` is a no-op for a LogVar2FJ world** (2026-11-03). `LogVar2FJSpotModel`
  drifts at its block's `Mu` (0 as calibrated) and declares no carry switch, so under the flag its
  futures still roll down the carry curve. UNMEASURED; the fix is a carry drift on the spot.

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
- **A cashflow list held to a horizon cannot be bought forward** (2026-11-18). A fixed, floating or
  inflation-linked list stating `Settlement_Date` beside `Investment_Horizon` is refused by name at
  booking and by the run, so a bond bought forward and held to a horizon cannot be booked; nothing
  is mispriced. Reading both is the forward's carry on repo to its settlement date ahead of the
  holding's close-out on its horizon, in each list's pricer: M.
- **A share held to a horizon settles its dividends on the simulation grid** (2026-11-18).
  `EquityDeal` books each grid interval's dividends off the share's dividend yield, carried at repo to
  the interval's later date, $N S(t_0) e^{r\tau}(1-e^{-q\tau})$, not on the share's own ex-dates and
  amounts, and the diary announces neither them nor its close-out. UNMEASURED: measuring it wants a
  stated dividend schedule against the yield on the same grid. Reading a stated schedule, as the
  equity swap leg reads its known dividends: M.
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
- **A one-point vol move carries cross-vertex convexity** (2026-10-07). On the autocalls the twelve
  vertex rows differ from one parallel point by -0.8% to +13% (0.1% to 1.2% on FX), stable across
  seeds and path counts, a tenth of a point cutting it tenfold. It is the regulation's definition,
  so stated, not fixed; under `Method: Graph` the rows are the shock's limit, the derivative, and
  sum to the parallel point's. ISDA's inflation and credit vol sensitivities refuse by name, the
  engine carrying no factor type for either.
- **The SIMM margin reads a one-day horizon under the ten-day constants** (2026-10-21). The vol a
  vega is weighted by, its delta weight times sqrt(365/14) over the 99th percentile, and the
  curvature's scale, half of 14 days' share of an expiry, are the ten-day methodology's (its
  paragraphs 10(b) and 11(a)), so a `Horizon` naming a file's one-day section weighs that section's
  parameters under them. Unmeasured: the one-day methodology is not in the repository, and
  measuring it takes its constants and one book's margin stated under each horizon.
- **SA-CCR holds a volatility trade, and a basis swap booked as one swap, in its ordinary hedging
  set** (2026-10-20). CRE52.46-52.47 hold a trade on the basis between two risk factors of one
  currency, or on a risk factor's volatility, in a hedging set of its own at half, or five times,
  its class's supervisory factor (CRE52.73). `SACCR` reads a package of two floating legs of a
  currency on two indices in its basis set; a `SwapInterestDeal` booked over two such legs is read
  off its own terms, a fixed leg against a floating one in its currency's ordinary set, offsetting
  the outright rate trades its own set would offset none of, and a volatility trade would be
  charged a fifth of its own set's factor. UNMEASURED: the swap booked beside outright trades of
  its currency, read off its legs against its own terms, would measure the first; the engine has
  no variance swap.
- **SA-CCR reads a cash-settled FX option past its expiry as the forward it would deliver**
  (2026-11-18). From its exercise date an option's delta is the limit of CRE52.40's, plus or minus
  one in the money and nothing out of it, which is the forward a physically settled option leaves
  until it delivers; CRE52 states no T for an option it has reached. A cash-settled option's payoff
  is fixed by then and owes nothing on the pair, yet until it pays its pair's effective notional
  counts it in the money at its whole notional. `SACCR` reads no `Settlement_Style`, as the
  option's price reads none.
- **SA-CCR charges a quanto or a compo for its underlying alone** (2026-11-18). A trade paid in
  another currency than its underlying's carries that pair's risk beside its underlying's, which
  CRE52.28 lets a supervisor allocate to the FX class too; `SACCR` holds a trade in the one class
  its type maps to (CRE52.26-52.27), so the pair's risk of a quanto or a compo enters no add-on.
  Its unit is the underlying at its own price (CRE52.36) and its delta's P the underlying's forward,
  a compo's that of S X in the currency it is struck in: CRE52 states neither, a convention.
- **SA-CCR reads a barrier option as its vanilla** (2026-11-18). CRE52 states no barrier's delta,
  and the regulators who speak to it hold a path-dependent option to CRE52.40's supervisory delta
  (the US SA-CCR rule's preamble, 85 FR 4385-4386; the UAE central bank's SA-CCR guidance, Q&A D.5),
  so `SACCR` reads a barrier on its strike, expiry and underlying with its barrier unread: a
  knock-out near its barrier carries the vanilla's delta where its own falls away, a knock-in far
  from its barrier the vanilla's where its own is nearly none. An equity barrier its stated closes
  have knocked in is the vanilla it became; one they knocked out is the rebate it pays on the
  crossing day, read as the vanilla on that day alone and after it not valued, its set's `Skipped`
  naming it.
- **A cap booked over legs is its legs** (2026-11-19). A cap or floor holding legs is valued as
  their sum, each leg on its own `Buy_Sell`, and its own terms are not read: a document stating
  `Sell` on the cap and `Buy` on every leg beneath it, as a legacy export writes a sold cap, is
  valued as bought, and its legs are what to correct. SA-CCR reads it as its value does, a cap or
  floor being its caplets or floorlets (CRE52.43), each long where its value is. A caplet's T is
  the day its rate is known, its last fixing, where its pricer's Black time runs to its first; a
  period capped fixing by fixing (`Pre_Aggregation`) is read as one caplet on the period's rate. A
  leg paying digital optionlets refuses the run by name, CRE52.42(3)'s collar not built.
- **The replay tuple pins neither build nor device** (2026-10-11). `engine_version` is a hand-set
  string, no commit, torch, CUDA or device is recorded, and `Deterministic_Kernels` defaults to
  `No`, so bit-for-bit replay holds only on one build, stack and device, and for a GPU gradient only
  under `Yes`: on one device a base valuation's first-order greeks table moved run to run at 1e-14
  relative while every value was bit-identical (2026-10-15). UNMEASURED across devices beyond the
  host-curve row.
- **`ACT_365_ISDA` and `ACT_ACT_ICMA` accrue as `ACT_365`** (2026-11-13). `DayCount.accrual` takes
  days over 365 for both, where Actual/Actual (ISDA) splits a period at each year end over that
  year's length and Actual/Actual (ICMA) takes a regular period as one over its frequency
  ([Day counts](json/general_types.md#day-counts)). On deposits over a year spanning 29 February
  2028, the annual ISDA period accrues 1.36e-3 of itself high and the semi-annual one after the
  year end 2.7e-3 high; the two semi-annual ICMA periods accrue 8.2e-3 high and 2.7e-3 low. The fix
  is the two conventions in `DayCount.accrual`, the ICMA one reading the period's frequency.

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
- **20. The recompute segment length.** The LogVar2FJ walk checkpoints every 21 steps, trading
  memory for a second forward pass; the mixer's own checkpoint holds one clock per fixing interval.
  The residual's granularity is the dial if that headroom is wanted.
- **21. Who may move the market, the date and the file** (2026-09-30). `POST /book/market`,
  `/book/date`, `/book/configure`, `/book/curve`, `/book/securities`, `/calculations` and a delete
  take no `actor`, and `/book/setup` and `/book/bloomberg` are admitted under `validate` yet
  re-solve and re-mark the market, so under a capabilities document any client can move the board
  a close is declared over, roll the date or drop a deal. The tick is the deployment's poll path and
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
