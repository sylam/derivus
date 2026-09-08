# Developer Roadmap

The [user-facing roadmap](../index.md#roadmap) lists what the library should be able to *price*.
This is the status board for what the codebase owes itself: open defects, decisions still open,
designed-not-built work, and what is built with the gate that holds it.

Two rules from [Conventions](conventions.md) apply throughout: **no abstraction ahead of a second
caller** (several items below are deliberately not started), and **look before you write**.

## Known defects

### Open

- **CVA vega through the LogVar2FJ outer walk is `nan` at float32** (2026-09-08, lane 4) - The carried state
  hands the pricer clocks the per-row re-seed never produced: the re-seed always started at
  `sqrt(xi)`, a 21% vol, where the outer path's fifth percentile is 5.5% at three years and lower on
  a fitted `Sigma_S` of 5. Below a 4% annualised vol on a 21-day block the SECOND of `ig_quantile`'s
  two Newton steps loses float32 in its BACKWARD — `ig_root`'s residual is pinned at 1.79e-07 by the
  resolution of `norm_cdf` near one while the smallest root falls two decades faster than the clock,
  so the correction is ten times the root and the backward's `1/f^2` chain leaves range. Values are
  untouched (`G` finite everywhere, the CVA identical with the checkpoint on and off); what comes
  back `nan` is every LogVar2FJ leaf of `grad_cva`, which is exactly the CVA vega the implied leaf
  exists to carry. Float64 is clean at every clock measured. The fix is a guard on that one division,
  in the `sqrt_or_zero` style that file already uses; until it lands, a CVA gradient on a book under
  this outer process is a float64 run.
- **`utils.ig_quantile` at a ZERO clock is `nan`, and the kit reaches it whenever an MTM row lands
  exactly on a remaining fixing** (2026-09-08, lane 4) - `ig_root`'s bracket `hi = 200m + 200m^2/lam`
  is 0/0 there; `LogVar2FJKit.grid` makes one step of length zero, `pieces` keeps it and `mixers`
  counts it, so the row draws a residual on an empty clock and the netting set reports *contains
  NANS*. Reproduced by any European whose expiry is a reporting row. The outer process guards its
  own version (`if step[a:b].sum() > 0.0` when building its pieces); the same guard in
  `LogVar2FJKit.pieces` is the shape of the fix.
- **The NKY chain block's vanilla-only objective is BIMODAL, and a `nan` strip walks past `verify`**
  (2026-09-08, lane S) - thirteen fits of `logvar2fj_block_JPY_NKY_BBG` at `Forward_Smile_Source:
  None` over three seeds and four path counts all land between RMSE 0.976 and 1.016 but split into
  two modes, `Alpha` 0.6-2.1 or 7.6-8.0, and `Pseudo` 8192 itself lands in one mode at seed 1 and
  the other at seeds 2 and 3 (`artifacts/lv_fast_20260908/`). So a same-answer gate on that ladder
  proves determinism, not agreement, and its seed spread is a mode switch rather than a noise
  floor - which is the identification the forward block was built to supply and, on a one-bucket
  ladder, costs a vol point of spot fit to supply (the open ruling above). Beside it: `verify`'s
  `atm_miss_max` compares with `<`, and `nan < x` is False, so a fit whose xi bootstrap has gone
  to `nan` still writes a factor (a Sobol 8192 run reported `+nan, +nan` among its ATM misses, a
  CAPPED polish and RMSE 17.8, and was `ok`); a `nan` miss should refuse by name.
- **The autocall's one-step-survival arm under GBM reads ONE moneyness for the whole path, so a
  compact V2's terminal put is priced at the ATM vol** (2026-09-08) - measured on the desk's NKY
  structures: the six-leg booking with its contra repaired (discrete up-and-in barriers on the five
  observation dates, which IS the autocall condition, each put leg reading the 70% strike's vol)
  against the folded V2 reads -64.2m vs -42.8m ZAR on 229524957 and +64.9m vs +86.6m on 220276683
  (33% apart, 32,768 paths, one seed), while on a FLAT NKY surface the two agree within 1%
  (-39.3m vs -39.6m) and on SD3E's near-flat long end to 0.02% (`artifacts/remark_20260908/
  fold_check.py`). `pricing.pv_discrete_barrier_option`'s docstring names the per-fixing smile
  convention as open; this is what it costs. The LogVar2FJ arm carries the smile in the model and
  is untouched, but no quote pins a 5y 70% vol on any index here (the vendor stops at 90 days,
  the chain at 2.7y), so the model's long-dated skew is extrapolated and the same trade reads
  -42m (GBM V2), -64m (GBM six legs on the file's parametric skew) and -42m (LogVar2FJ). A GBM
  mark of a compact V2 on a skewed surface should not be stood behind until the arm reads the put
  at its own strike.
- **The autocall's fixing-to-coupon alignment is a guess the booking never states** (2026-09-08) -
  `QEDI_CustomAutoCallSwap.calc_dependencies` drops fixings more than a month before the first
  unpaid coupon and pairs the rest with the coupons POSITIONALLY (`one_each`: equal counts, each
  fixing on or before its coupon); that pairing is what admits the OSS arm, and where it fails the
  full-path arm reads windows reaching back to the predecessor coupon. The Barclays V2s fix a few
  business days before each payment, so the pairing holds and the trigger reads the payment's own
  window - but a deal fixing five weeks before a long settlement silently falls out of `one_each`
  and prices on the other arm. The fix is a schedule that declares it: an observation-date column
  on `Autocall_Coupons` (or the fixing table naming its coupon), the month heuristic retired. A
  row because every autocall document in the book reads the heuristic today.
- **`EquityPriceVol` under `Sticky_Strike` cannot reach a fixing strip** (2026-09-08) -
  `pricing.calc_moneyness` returns the bare strike for a parametric (`Skew`/`SVI`) surface, which
  carries no fixing axis, and every fixing-strip pricer then raises `IndexError: index 1 is out of
  bounds for dimension 1 with size 1` at `pricing.py:343`. The book works around it with `Explicit`
  surfaces and NKY's `Moneyness_Rule` moved to `Sticky_Moneyness` (its `ATM_Ref` re-anchored and
  inert there). The fix is to broadcast the strike onto the fixing axis as the `Sticky_Moneyness`
  branch already does through the forward.
- **A simulated equity and a static rate curve cannot share a floating-leg deal** (2026-09-08) -
  `utils.get_simulated_resets` (utils.py:1650) through `pricing.pv_float_cashflow_list` returns the
  KNOWN resets shaped by the scenario count and the forecast curve's own unknown resets shaped by
  the static factor, and the concatenation refuses with *Sizes of tensors must match except in
  dimension 0. Expected size 128 but got size 1* - 13 deals skipped in the book's credit Monte
  Carlo under a traceback that reads like a deal fault, so the profile reported was the option legs
  and the NDFs, not the netting set's. Either a static unknown reset broadcasts onto the scenario
  axis, or every forecast curve on a book needs a process and the refusal says so by name.
- **A digital contra booked with `Cash_Payoff` 0.0 prices to zero without a word** (2026-09-08) -
  two of the Barclays structures (229079480 and 229524957) carry one, worth 2.9e8 and 2.85e8 of
  payoff in the legs it was meant to cancel; the engine accepts the leg and the six-leg documents
  price. `derivus_compact_autocalls.py` names it (`DEFECT digital contra Cash_Payoff is 0.0`) and the
  fold removes it, but the SOURCE booking is still wrong and a zero-payoff digital is never a deal:
  the load should refuse it by name.
- **The vendor's implied-vol grid answers only at 30, 60 and 90 days** (2026-09-08) -
  `<n>DAY_IMPVOL_<mny>%MNY_DF` at n = 30/60/90 and five moneyness points is the whole grid; the
  `<tenor>MTH_IMPVOL_` family is a 90-day alias whatever the tenor (six override values leave it at
  24.1058 / 24.3242) and SD3E answers nothing. So the long end of every equity fit is the file's own
  surface or the listed chain, and the LogVar2FJ mark on 229524957 moves 4.6% between a chain-only
  fit and one carrying the file's 2.74y ATM. Recorded so no pilot probes it again; whether the
  workstation is entitled to a longer grid or it lives under another field family is one question
  to Bloomberg support, which is the owner's.
- **`Residual_Law: Gaussian` reports a greek at a leaf the model never reads** (2026-09-08) -
  `Alpha` and `Beta` are filled with their declared defaults (1.0, 0.0) where a Gaussian document
  omits them, so a sensitivity run prints an identically-zero row at each. Dropping them from
  `curve_names` under Gaussian would make the factor's leaf set mode-dependent, which is why it is
  not done; the calibrator already refuses to FIT the Gaussian law unless it is declared.
- **`LV_CHECKPOINT_STEPS = 21` was measured for the walk alone** (2026-09-08) - the mixer now
  checkpoints per residual draw beside the walk's 21-step segments, and peak memory went 8,392 to
  9,990 MiB at 2,048 x 2,048 (`artifacts/lv_nig_20260907`). If the collateral lane wants that headroom
  back, the residual's checkpoint granularity is the dial, not the walk's.
- **Every correlation between a `calc_statistics` factor and a newer one is a business day out**
  (2026-09-07) - `utils.calc_statistics` builds its `delta` as `transformed.diff(1).shift(-1)`, so
  each innovation is indexed at the return's START date, while `GARCHSpotCalibration`,
  `MarkovHMMSpotCalibration`, the two basis classes and `LogVar2FJCalibration` all index at the
  END. `Config.calibrate_factors` then concatenates the two families and takes a Pearson
  correlation of series that are one day apart. MEASURED on ONE simulated series read by both
  classes: GARCH against GBM reads **0.009** - a series uncorrelated with itself - and against the
  LogVar2FJ estimator's own `eps`, **0.335** (GARCH) against **-0.008** (GBM), where the sibling
  was drawn at rho 0.6 (`artifacts/lv_hist_20260907/hist.py recover fine`). Five shipped classes
  read `calc_statistics` (GBM asset, GBM index, CS forward, HW rate, HW hazard), so the fix is one
  line in `calc_statistics` and a re-read of every banked `Correlations` block, which is why it is
  a row and not a fix.
- **`calibrate_factors` reports a calibration class's own refusal as "Data errors in factor"**, and
  raises `AttributeError: 'NoneType' object has no attribute 'corr'` where EVERY factor is skipped:
  the bare `except:` around `rate_value.calibration.calibrate` swallows the message the class
  wrote, and `consolidated_df` is then still `None` when the correlation step runs. Both measured
  2026-09-07 (`hist.py sector`, whose blank-`Implied_Values` arm refuses by name inside the class
  and reads as a data error outside it).

- **Solved accrual strikes moved across the checkpoint landing on documents carrying no LogVar2FJ**
  - the FX gate read the GBM TARF's zero-cost strike at 15.32196559 on 48f4779 and 15.31624884 on
  7ed3faf (the GBM ZAR accumulator 15.68600904 -> 15.68756069), up to 3.7e-4 and 15x the runner's
  own 2.5e-5 MC floor, while the same documents priced at a FIXED strike are hex-identical across
  7ed3faf -> 678af12 (`artifacts/fx_gate/hexcheck.py`). The estimator is bit-stable where it was
  checked and the root moved where it was not; 48f4779 is not in any checkout any more, so the
  move is unpinned. Read the 48f4779 -> 7ed3faf pair at a fixed strike before the next hex claim
  on an accrual document's ROOT.
- **`pv_partial_barrier_option` settlement completeness** — the rebate leg's per-decision `cash_events` are declared and audited off-gate (booked rows {1, 2}, declared {0, 1, 2}, support exact over 1024 paths) but no shipped gate forces completeness, for want of a collateralised partial-barrier document. The safety half is gated (`test_a_rebated_knock_out_registers_the_rebate_it_books_row_by_row`).
- **`pricing.stochastic_boundary_correction` (`gates/boundary_bandwidth_plateau.py`)** — The bandwidth plateau holds and is carried, not closed: the 32768-path operating point became runnable when the Sobol chunking closed (2026-09-03) and the re-read there is pending. The declared `Boundary_AAD_Bandwidth` default 0.01 sits one rung inside the plateau's lower edge. The suppression seam is that same field at 1e-12, bit-identical to deleting the correction. *Measured:* At 16384 and 20480 paths the estimate holds over 0.005–0.08: seed-mean correction spread 2.41% (discrete barrier) / 3.87% (HN), reported CVA delta 0.60% / 0.24%, against seed floors 13.69% / 28.52%. No single seed sees it — per-seed spreads 12.7–23.1%, seeds disagreeing on the drift's sign, and the seed floor bounds part of the noise only because the Sobol stream is not derived from `Random_Seed`. Lower edge at 0.0025: 9–20% low, per-rung seed spread to 101% (kernel starvation). At 2048 paths the correction falls monotonically 23.76% and nothing holds still. Acceptance names 32768. Re-baselined onto the declared grid: the HN barrier gate is 1.18% against a 6.19% suppression mutant, and the discrete-barrier profile gate is a bit-exact rebate ledger.
- **`tests/test_boundary_scoping_dominance.py`** — The correction is mutation-gated; its *scoping* is not — a mis-scoping mutant (set-level `portfolio_delta`) has no public seam, since every registration names its class directly. Said in the gate's own docstring. *Measured (re-recorded 2026-09-03, after the ledger fix):* the fixture is authored because correction/smooth is 3.80 on a down-and-out digital struck at ~zero. At 16384 paths the live lane agrees with its CRN oracle at **2.74%** (flatness 2.83%; the suppressed half bit-identical across the fix, so the whole move was the correction), tolerance 0.10; the bandwidth-suppression mutant reads 366.61%, and the same mutant on the old two-set fixture survives at 0.23%.
- **`NettingCollateralSet` backward, recompute node off** — One gradient entry takes two distinct float64 values from bit-identical inputs — a nondeterministic GPU reduction, not a graph defect. It bounds how tightly any collateralised sensitivity gate can be pinned. Nothing done.
- **A collateralised set reading a knock-out rebate's 14 declared settlement dates** — the shape no gate exercises, left from the `add_grid_dates` closure.
- **`pricing` (TARF block)** — The target pin fires on 27–61% of paths, 27% short uncorrected, and is gated structurally with no tolerance asserted because nothing resolves it better. Exact behind `Branch_And_Weight: 'Yes'`; the crisp default keeps the declared blindness. *Measured:* Estimator 13% bandwidth spread, oracle 8.9% flatness — neither better than ~10%. Do not tune on the oracle: it cannot see it either.
- **`pv_MC_AutoCallSwap`** — The averaging coupon carries the termination latch under `LogVar2FJ` and cannot under the daily kits or GBM. A window of fixings is priced by sampling the window's own blocks and truncating the PREFIX return (spec 2.4.1), so the termination is a crisp per-scenario decision again and `Branch_And_Weight: 'Yes'` is admitted; GBM and the daily families keep the full-path branch, whose termination is a smoothed per-inner-path weight with no decision to stamp, and refuse by name. *Measured (2026-09-06):* the five-fixing document reads 0.76 combined SE against a brute-force oracle and a whole-interval window (26 weekly fixings, prefix 3.8–7.4% of the interval) 0.04; the smooth value is 0.67 SE from the crisp; spot delta 0.035% and gamma 0.0093% off their CRN ladders. A lagging-payment schedule (coupon paying after its fixing) would still have its pending window zeroed by the carry — `pending` remains unused here. *Closed 2026-09-06:* the deal says which - `Barrier_Observation`, `'Spot'` (default) or `'Average'`, read by both arms.
- **`Credit_Monte_Carlo` × the autocall's delta, what is left** — the collateralised residual had
  two parts. The FIRST was a ledger the counterfactual replayed and the value never had: the float
  leg was declared in `cash_events` but never `cash_settle`d, so `cash_to_C` moved cash against a
  `Cf_Rec` holding the four coupons alone (nine declared rows against four settled on the 2y USD SPX
  book; the row they share declared −2.45 where +0.94 booked) — found by diffing that replay against
  the coupon-only gate's, where the two lists coincide digit for digit. The declaration is gone
  (values bit-identical; `LatchedBoundarySet.cash_events` states the law: only settled cash may be
  declared), the CSA row 69.4% → **51.2%**, GBM under the CSA 6.0% → **1.2%**. The SECOND is the
  kernel flux estimator's own variance, open: under a zero-threshold CSA the correction must supply
  2.5× the pathwise term and at 256 outer paths the local-linear kernel holds one to three gaps
  (amplification 1.0 — starved, not refused); it scatters with the path count, 51.2% at 256, 18.2%
  over at 1,024, 10.2% at 2,048, and the coupon-only gate itself reads 0.14% at 1,024 × 4 against
  5.5% at 256 × 1. Widening the bandwidth to 0.05 reads 13.5% here and costs the uncollateralised
  row 2.13% → 6.85%. The path count, not the deal. Beside it, unchanged: the float and the terminal
  put reach no `cash_settle`, so `Results['cashflows']` carries the coupons alone — settling the
  float moves the collateralised CVA (component 0.170177 → 0.147908) and buys nothing in the delta
  (52.6% against its own ladder), a change of its own. The barrier-free lag-0 11.0% is a third
  thing, untouched (2026-09-04). NOT the component model's alone: GBM under the same CSA at
  2,048 outer reads 11.8% short of a ladder flat to 2.2%, where its 0.8% at 256 sat on a ladder
  12.1% non-flat (2026-09-04 night, the put leg's flux analytic by then).
- **`pv_MC_AutoCallSwap` × a lagged block's terminal rows** — a block whose rows lie between a
  fixing and its settlement sets `last_fixing` off its LAST row, so a row dated before that fixing
  prices the coupon crisply off a spot it has not reached: on the campaign's book rows 2028-06-03
  and 2028-09-02 price the final coupon off the 2028-09-08 fixing. A value property (look-ahead),
  identical before and after the reach fix, which now forks exactly the value computed.
- **`pv_MC_ExtendableForward`** — Two declared limits remain after the flux registration and the mirror booking. (1) The settled-cash channel under a CSA is not registered, on the measurement below: the flipped payments' exposure rides the value side until settlement, then one hazard-weighted margin window. The `cash_alive` design is recorded in `test_a_collateralised_cva_delta_carries_the_surviving_cash` and waits for a document that can falsify it. (2) The rolling backward pass carries a one-signed Gauss–Hermite smoothing bias over the relu kink. *Measured:* (1) +0.25% / −0.02% / +0.26% / +0.03% across four amplifying documents, against CRN ladders that resolve no finer. (2) Single-decision inside 5e-3 of Black; multi-decision bounded by the dominance gates only — the `Boundary_*` valuation options are the dials.
- **Seasoned TARFs with pre-base settlements are discarded outright** — `instruments` drops settlement dates before the base date and then any declared fixing older than a month before the earliest survivor, so a fixing at −18d settled at −8d logs `fixings=1 resolved=0` and the deal marks bit-identical to the same document with that fixing deleted — priced against its full original target. The accrual netting (closed below) bites only at later profile rows. *Measured:* mark `0x3ffa9945323ddaf2` with and without the settled fixing.
- **A TARF valued between two settlements marks NaN** — the lagged schedule at a base date with settlements at −2/+2/+88 days (`fixings=3 resolved=2`) marks `nan` on every tree measured. Exactly the one-settled-one-not geometry the seasoned work is about; no fixture reaches it.
- **`pricing.barrier_touched` (window-touch registration)** — `Boundary_AAD_Window_Touch` (declared, default `No`, bit-identical off) registers a `LatchedBoundarySet` where `get_fx_barrier_underlying` publishes no rate and the test collapses to a 0/1 endpoint indicator. The switch decides the *sign*; its magnitude is unestablished — `add_grid_dates` (closed 2026-09-03) unblocks enriching the fixture, and the re-measurement is pending. *Measured:* −2.2467692 registered against +0.5207422 unregistered. The CRN oracle scatters 88% of its own median and is not sign-unanimous across path counts. Where the bridge lives there is nothing to register: `maximum(beyond, crossed) == crossed` identically, and the CVA delta already agrees with its ladder at 0.44%.
- **`pricing.calc_vol_adjustment`'s analytic consumers** — `pv_barrier_option`, `pv_one_touch_option` and `pv_discrete_asian_option` adjust the vol only — no fx scale, no fx carry — so a continuous-monitored compo barrier, a compo one-touch and a compo asian price half-adjusted numbers without raising. `pv_european_option`'s repair is untested; no compo European fixture exists. *Measured:* The OSS half is closed below. The compo smile coordinate is undeclared — an open decision.
- **each deal's `calc_dependencies`** — A sibling fallback may name a factor discovery never fetched: discovery iterates `factor_fields` over the raw field and `get_fieldname` drops blanks, so a blank reference loads no factor. A fallback is safe only if it names something else already pulled in. *Measured:* `Discount_Rate ← Currency` is safe at 34 sites because `Currency` is an `FxRate` whose `InterestRate` comes transitively. The one cross-leg instance (`FXForwardDeal.Sell_Discount_Rate ← Buy_Currency`) is fixed — both rates `default=REQUIRED`, no fallback.
- **`instruments.StructuredDeal.post_process`** — The class declares `F('Net_Cashflows', default='Yes')` and `post_process` reads `field.get('Net Cashflows', 'No')` — the same words with a space. The declared key reaches no read, so a document authoring `Net_Cashflows: 'Yes'` silently takes the un-netted path. Reconciling flips the effective default for every `StructuredDeal` omitting it, so it needs its own measurement first.
- **`schema.DealFields` leftover** — Of 148 declared `.field.get` sites, **36 disagree with their declaration**, three fatally (`config.py`'s two `is not None` discovery guards would mint garbage factors off a blank default; `calculation.py:2088`'s Reference stamp). The calculation store's "hold a surviving fallback to its declaration" half therefore has no deal-side twin. Enumerated in `tests/test_declared_defaults.py`; the decision is open.
- **`Credit_Monte_Carlo` output keys** — The exposure profile is reported undeflated and `Deflation_Interest_Rate` applied only inside the CVA/FVA scalars, so a deflated expiry-row EPE cannot be read from the reported tables. Remedy: publish `Dt_T` beside `mtm`. The composition harness gets it by booking a unit base-currency cashflow whose row-zero mark is `D(0,T)`.
- **the `Correlations` section vs `save_params`** — The two halves of a quanto are authored in different bases and nothing checks they agree: `save_params` emits `ρ̄ᵢ = corr(dW, dWᵢ)`, while the section's rows are the independent normals the cholesky consumes. The section needs `a = ρ̄₁`, `b = (ρ̄₂ − ρ·ρ̄₁)/√(1−ρ²)`; copying `ρ̄₂` in gives a world whose drift and covariance disagree, silently. *Measured:* `a² + b² = C²` exactly.
- **`HullWhite2FactorImpliedInterestRateModel.precalculate`** — Reads `self.param['Lambda_1']` unguarded off the `Price Models` block an implied model does not need, so omitting it raises `TypeError: 'NoneType' object is not subscriptable` naming neither field nor factor. Smaller sibling: `FXVolSurfaceParameters` subscripts `point['Timestamp']`, so a block without one dies on a `KeyError`.
- **`bootstrappers.create_market_swaps`** — The `Distribution_Type` declaration lives on the *surface*, which the Bloomberg emitter does not author, so a desk pointing `Swaption_Volatility` at a lognormally-declared factor still gets a lognormal fit of normal quotes. The block's `Quote_Source` line is where that is said. *Measured:* The two conventions are 9.7×–11.4× apart in premium.
- **`config.CustomJsonEncoder`** — The `.DateOffset` string is built by walking `DateOffset.kwds`, whose key order for a multi-unit period is a set iteration: `DateOffset(months=6, days=2)` encodes `'6M2D'` or `'2D6M'` by interpreter. Both parse back to the same offset, so nothing reads wrong; what is not byte-stable is `write_marketdata_json`'s output and any hash over such a block. *Measured:* 4:1 over five fresh processes. Every offset the emitters write is single-unit, which is why no determinism gate has met it.
- **`derivus_jupyter.set_value_from_widget`** — `set_repr` picks a deserializer from the `obj` token and for an untagged table falls to a hardcoded whitelist of field names; `Names`, `Sampling_Data_1`, `Sampling_Data_2` and `Barrier_Dates` are outside it and raise. The fix is to render from the declaration, not to add a fifth token. Superseded for *viewing* by the web UI; the whitelist still bites the Jupyter write path.
- **`stochasticprocess`, `calculation`, `bootstrappers`** — Fourteen descriptors have no widget — `Transition_Matrix`, `Sigma_By_State`, `States`, `Tradable_Instruments`, the `Hedging_Problem` maps, `CDS_Tenors`, `Scenario_Factors`, and a quote's own `Deal`. Every one is an output shape, while `Table` declares fixed columns and `Container` fixed named children; `define_input` reads `col_names` / `sub_fields` unchecked, so the Workbench raises rendering any platinum-world process or the hedging problem. Wants a widget, not a schema change. Superseded for viewing. *Measured:* Pinned in both directions by `test_the_descriptors_with_no_widget_are_exactly_these`.
- **`gates/pricer_branch_census.py`** — **59 unexecuted arcs at 1ed927a**, not re-taken over the 2026-09-02 edits, so 59 is not pinned (arcs are line numbers — an edit mid-run re-attributes every arc below it). Named leftovers: `pv_MC_Tarf`'s put arm has left the census's scope, its flag having moved into `oss_truncated_draw` outside `FAMILY`, and `coverage` is an undeclared dev dependency of `gates/impacted.py --build-map`. Still unstruck: the knock-out formulas other than the Down-and-Out / Call / K > H arm every non-structure fixture takes. *Measured:* Not comparable to the historical 64/65 — different instrument, and a pricer that starts being called trades one `never-called` line for its interior arcs.
- **`calculation` (CVA Hessian)** — `Hessian: 'Yes'` with `Gradient: 'No'` is a silent no-op — the second-order block rides the first-order tape and should refuse by name. The Silverman bandwidth is per-batch, so `Simulation_Batches > 1` oversmooths relative to the run's true path count.
- **`utils.spot_model_currency` (declared scope reduction)** — The component family does not transport to the reciprocal axis — the change puts a state-dependent term in its long-run intercept (`ω_t + φ(1 − 2γ₂)h_t`) and leaves the family — so a component deal whose underlying *is* the base currency refuses by name (`UnpriceableSchedule`, fatal), crisp arm included, naming the plain family and the other orientation as remedies.
- **`bootstrappers.py` (HW2F solve debug block)** — Overwrites `debug.deals` and attempts `write_trade_file('ZAR.aap')` in the CWD. Inert today, but a run from a writable CWD drops an artifact where the no-artifacts rule forbids one. Delete the block or route it to a declared output directory.
- **`tests/test_declared_defaults.py` (two plan-hash pins)** — `platinum_hedge_shipping.json` and
  `policy_test_simulate_only.json` have hashed to different plans since 91c29de (the two-sided wings
  and the FVA column, 2026-09-02), whose targeted run did not reach the pin; stable across fresh
  processes, so not the `.DateOffset` defect below. Whether the move is a declared plan change or a
  values-plane field leaking into the plan is unclassified; re-pin once it is. Neither fixture
  carries a barrier or an accumulator (bisected 2026-09-03).
- **`Credit_Monte_Carlo.report` × a folded static root** — a book whose only deal folded to a
  `FixedCashflowDeal` (a knock-out crossed on the base date) while the factor the original barrier
  discovered is still simulated does not frame: a (1, 1) static root against the (T, B) grid,
  `Shape of passed values is (1, 1), indices imply (2, 1)`. Beside any simulated deal it reports
  and is gated bit-identical to the longhand cashflow; alone it is new reachability the fold
  opened (2026-09-03). Base valuation is unaffected. A SKIPPED deal alone meets the same frame
  failure (a quanto autocall refused by the component arm, 2026-09-04).
- **`HestonNandiComponentImpliedSpotModel` × a fit with `Beta` above `Rho`** — the all-pillar SPX
  fit of 2026-09-03 (Beta 0.99502, Rho 0.99, Phi 0, its Dec-27 pillar floored) marks a finite base
  valuation and **171 NaN of 3,328 credit-MC cells over 89 of 256 scenarios** (first at the 6M
  rows), CVA and FVA NaN; the same book fitted with Dec-27 dropped (Beta 0.982) is clean on every
  cell. Open decision 5's live instance: with `Beta > Rho` the long-run term `(Rho − Beta) q_t`
  is negative and the coarse-grid walk finds no floor before the square root. Nothing refuses a
  fit that lands there.
- **`HestonNandiComponentModelParameters.bootstrap` × a floored ATM pillar** — a listed chain
  whose ATM ladder carries a calendar-arbitrage pillar (a later expiry quoting less total variance
  than an earlier one: SX5E Sep-27 at 15.0% behind Mar-27's 18.9%, SPX Dec-27 at 15.6% behind
  Sep-27's 17.4%, both 2026-09-03) floors that pillar in the L bootstrap, and the floor's relative
  miss squared at `atm_constraint_weight` 1e4 is then the whole objective: Nelder–Mead walks `H0`
  from the correct front-ATM seed (17.03%) down to 7.04% because that shaves the shortfall from
  0.101 to 0.026, and the wings never enter — converged at 344 evaluations with a worst wing of
  234% and the strip 7.0 → 23.3 → 18.2 → 12.1%. `Rho` 0.97 does not help (the search runs to the
  box corner with the pillar still floored). Dropping the arbitrage expiry from the ladder solves
  every ATM row exactly and lands the same basin from three starts (worst wing 86%, the far call
  wing); it is a JSON remedy, not a fix. Open decision 16. *Measured:* the seed trace in the
  session scratchpad (`calib_scratch/seed_SX5E.log`, scaffold lines removed).
- **`Credit_Monte_Carlo` × nothing to simulate** — a book whose deals reach no stochastic factor
  dies in `shared_mem.reset` on a zero-wide random block, and one whose deals have no date after
  the base date dies in `update_time_grid` on an empty `max()`; neither names the document. Both
  should refuse by name (found 2026-09-03 probing a lone `FixedCashflowDeal`).

### Closed

- **Stage 5's vanilla guard refuses with a bare tensor error where no quote sits within ONE STEP
  of a forward maturity** (2026-09-08) - CLOSED by lane S2 the same day: `LVFit.guard_rungs` reads the rung nearest
  each target's `T1` and `T1 + Delta` within `spot_rung_tol * Delta`, and refuses by name where
  there is none; the guarded set is built only where stage 5 runs, so a one-bucket ladder pays
  neither the extra evaluation nor a degradation line for a stage that did not happen. Every fit
  whose guarded set was non-empty before is bit-identical, and there is none in the book: all four
  ladders re-fit to the LAST BIT under the production defaults, same evaluations and same RMSE.
  **First measurement of the calendar bucket on a listed chain**: the second-bucket NKY document
  runs to the end in 346.9 s where it died at 940 s, and says what the lever costs — the 0.5y bucket
  moves `Rho_S` from −0.7962 to −0.7857 and `Beta` not at all, while the vanilla RMSE at the forward
  rows' own maturities DEGRADES by +0.119 vol points, past spec 5.3's 0.1 failure mode. The joint
  polish stops CAPPED at `Max_Iterations` 150, so theta\* there is where the budget ran out; and the
  guard binds stages 5a/5b, while the +0.119 is read after the unguarded polish.
  As it stood: `LVFit.global_stages` builds `guarded` as the quotes within
  `self.delta` = `1/Steps_Per_Year` of a target's `T1` or `T1 + Delta`, and stage 5a/5b's guard then
  `torch.stack`s their misses: on the NKY chain block with `Param_Buckets` at 0.5y the 6m:6m row
  survives the reachability rule (its 0.5096y rung is 1.9% of Delta away) but is 2.4 steps from
  0.5y, so `guarded` is EMPTY and the fit dies at `bootstrappers.py:1919` with *RuntimeError: stack
  expects a non-empty TensorList* under both `Stationary_Spread` arms, 940 s in
  (`artifacts/remark_20260908/repro_bucket.py`). Two tolerances for one question - the reachability
  rule reads the rung nearest Delta within a quarter of it, the guard reads one day - and the guard
  is the one that should follow the rule: judge the failure mode on the rungs nearest each target
  maturity, and refuse BY NAME where there is none rather than stacking nothing. Until then the
  calendar bucket, which is the brief's own lever for the forward block, cannot be measured on any
  ladder whose rungs miss the forward horizons by more than a day - which is every listed chain.
- **`utils.LatchedBoundarySet` × a lagged settlement** — a block opening after a fixing and before
  its settlement (`last_fixing` set, `fixing_aligned` False) prices the coupon crisply off the one
  observed fixing on EVERY row, but the decision's own-row fork carried the `tau == 0` row alone
  and `obs_before` is block-granular, so the block's earlier rows sat in neither branch and their
  jump was zero: five of the 2y USD SPX book's 23 reporting rows, the uncollateralised CVA delta
  14.6% off its ladder against 0.78% with the coupons on their fixing dates. Closed 2026-09-04:
  `own_row` is a list of forks per decision and a lagged block forks every row it decides;
  14.6% → **2.13%**, the barrier-free lagged book 11.7% → 2.21%, every value bit-identical, the
  rebated knock-out's one-fork entry unchanged. Found by diffing the lagged and lag-0 documents'
  debug flows.
- **`pv_MC_AutoCallSwap` (crisp arm) × the terminal put barrier** — the expiry-row
  `torch.where(Sj <= putBarrier, ...)` carried no boundary set, so every sensitivity the put leg
  touched was short its flux: on the 2y USD SPX `QEDI_CustomAutoCallSwap_V2` under the component
  model the base-valuation spot delta read 12.3% off its ladder, `H0` 24.8%, the 2.29y `L` knot
  175%, every one closing to under 1% with `Barrier: 0.0`. Closed 2026-09-04: the crisp line
  registers an `InnerBoundarySet` beside the coupon latch — one decision per inner path, gap
  `log(B/S)`, jump `L·D·fx·(rebate − (1 − S/K))`, node outputs under the recompute node — only where
  the conditional-p splice did not take it, so the stride and `Branch_And_Weight` register nothing
  extra. Under `HN_Stride: 'Yes'` the same document reads spot 0.16% off its own ladder at first
  order and gamma 0.02% off a ladder of the AAD delta at second (`Greeks: 'All'`, 2,642 s for the
  26 × 26 Hessian at 16,384 sims), the stride's conditional-p mixture carrying both orders.
  On the crisp arm: spot 12.3% → **0.97%**, `H0` → 5.8%, the knot → **2.9%**; the rates delta moves 354.2 →
  447.3, which a seven-rung ladder shows is right (no path crosses at 1e-4, 1e-3…1e-2 read
  436.5…467.3). The float rows declare their `cash_events` on the streamed law (twelve payments
  where there were four, `ledger_max` 0.0). Every value bit-identical; `Greeks: 'All'` now refuses
  on this deal as on any registered decision. The credit-MC residual is the Open row above.
- **`config.find_models` (what the base currency's name excludes)** — every factor named by the
  base currency was kept static, its curve included; a base-currency curve to simulate carried
  another name. Decided 2026-09-03: only `FxRate.<base>` is excluded (identically one), so a curve
  named as the base currency simulates once a model is declared for it; the naming convention
  still works and no document declaring no such model moves. `test_base_currency_json.py`.
- **`pricing.getpartialbarrierpayoff` (start-window arm)** — the option leg priced the live side of a live window and nothing else, so a seasoned knock-out marked **−6.141370** where the vanilla is 85.23. `start_window_state` — the same signed-limit and spot-vs-level tests the rebate resolution computes, one spelling — resolves on the knock-out closed form so the parity carries the knock-in: touched, the KO is dead (rebate at the hit) and the KI is the vanilla; closed untouched, the KO is the vanilla and the KI exactly nothing (+ its discounted expiry rebate). Eight of twelve resolution documents land on one float across directions and sides; live-window marks bit-identical, hexed. THE WALK AGREED ONLY AFTER ITS OWN FIX: the CMC's row-0 window mask forced a point test on every `Barrier_At_Start` document (`at_start or …`), so the mark and the scenario walk read opposite states on the closed-and-beyond corners — one spelling now, all eight agree to the walk's float32, bit-identical on every live-window profile. `test_partial_barrier_json.py::test_a_resolved_start_window_is_the_state_it_resolved_to`, `::test_a_closed_start_window_reads_the_same_state_its_own_scenario_walk_does`. Residual: the deal's own Black sits 0.137% from `FXOptionDeal`'s on agreeing inputs, inside the 2e-3 gate that carries it.
- **`pv_MC_Tarf` (`accumulation`)** — the opening accrual netted every declared reset, settled or not, while the loop banked observed-unsettled fixings again, and the index sat one off per declared reset. The schedule is walked from zero by position (declared prefix off `TensorSchedule.declared_values`, one spelling), so each observed fixing banks its own accrual and the pot opens right. *Measured:* the lagged fixture banks 0.2 + 0.2 and caps the live fixing at the 0.1 they leave — 497.1924 against a three-leg closed-form oracle's 497.1905, discriminated against never-netted and netted-twice both; and **a TARF that crossed on its first declared fixing marked identically zero on every row and scenario** — it now pays the clamped target exactly (20.0, both directions). No-declared-reset documents bit-identical, hexed. `test_fx_tarf_json.py::test_two_observed_fixings_in_one_settlement_lag_bank_their_own_accruals`, `test_tarf_cash_settle.py::test_a_target_its_declared_fixing_exhausts_pays_that_target_and_stops`.
- **`utils.MTABoundarySet` × a held balance** — one collateral decision was registered at every remaining margin call once a dead deal froze the balance below the MTA: one scenario was charged the whole counterfactual 77 times, 97% of the phantom on five frozen scenarios, hidden only while the relu truncated the knocked-out paths. `MTABoundaryEvent.live` marks the binding call per constant-balance run, per side (pooling the sides suppresses one half of an exactly-cancelling zero-MTA pair — measured, which is why zero-MTA books are bit-identical); an unstamped `live` refuses by name. Live-exposure gate 1.90% → 0.01%; zero-CVA gate exactly 0.0 both ways. Structural gate, no tolerance: max live events per run per side is 1, against 81 through the gate's own assertion on today's engine. `test_boundary_pricer_events.py::test_a_run_of_held_balance_registers_its_transfer_decision_once`.
- **`utils.LatchedBoundarySet` × the settled ledger (the barrier family)** — a deal whose row pays out everything it is still worth had the *realised* payment folded into both counterfactual branches while the collateral balance followed the branch, manufacturing exposure equal to the deal's own payoff (+2.19859 per row, exactly). `settles` declares `(mtm_row, booked)` and each branch settles `booked + branch[row]`; `pv_discrete_barrier_option` and `pv_partial_barrier_option` declare it, and both rebate legs gain the per-decision `cash_events` they never had. *Measured:* two collateralised digitals, cushion 300, 432.54% → 12.03%; un-cushioned 23.10% → 1.78%; the un-lifted lane 21.54% → 2.74% with its suppressed half bit-identical; the collateralised barrier gate 6.71% → 1.08%, its deleted-correction mutant 42× clear. Bit-identical wherever no collateral chain exists, hexed per pricer. `test_boundary_pricer_events.py::test_a_latched_registration_declares_every_settlement_in_its_reach`, `::test_a_ledger_row_declared_at_what_was_booked_moves_the_net_by_nothing`, `test_boundary_scoping_dominance.py::test_the_lifted_portfolio_reports_a_delta_its_own_oracle_agrees_with` (the settlement mutant dies on the dominance guard at 1.04×; off-gate it reads 424.08%, sign-flipped).
- **`utils.LatchedBoundarySet` × the settled ledger (the streamed pricers)** — `pv_MC_Accumulator`, `pv_MC_Tarf` and `pv_MC_ExtendableForward` settle a STREAM of per-fixing cashflows, which `settles` cannot state — it reads a knocked-out accumulator as settling 1476 at a row it settles nothing at — so under a CSA their counterfactual cash stayed at what the realised world paid. `cash_events` carries the facts instead, on one law both families now share: a payment declares `(mtm_row, decision, if_fired, if_not, booked)` and is made iff no decision up to its own fired, a trigger's coupon declaring `(amount, 0)` at its own decision and a fixing `(0, amount)` at the last decision that can kill it; payments sharing a row are summed before `cash_to_C`'s relu split, the split of a sum not being the sum of the splits. Each stream DERIVES its amount from the `pending` head it already builds — the accumulator's per-fixing payoff, the TARF's strip head, the extendable's bucket — and each declaration replayed at the booked flags reproduces the book exactly (`ledger_max` 0 on all three, logged beside `recon_max`). *Measured* (collateralised CVA delta against its own CRN ladder; the collateralised documents are probe-built, not in the suite — what the tree holds is the structure: `ledger_max` asserted in the accumulator and extendable organ gates, the five-pricer completeness gate, and the currency gate): accumulator 20.16% → 0.68% (flatness 3.1%); TARF 5.12% → 0.27% at zero settlement lag and 6.61% → 2.10% at 25 days (1.5% / 1.1%); extendable 0.25% → 0.27% (0.1%) — its residual was never this channel's, which is what building the channel shows. Autocall and both barrier declarations re-spelled onto the same law, bit-identical; every document with no collateral chain bit-identical, hexed per pricer. `test_boundary_pricer_events.py::test_a_latched_registration_declares_every_settlement_in_its_reach` now runs on all five latched pricers — today's-engine mutant, undeclared rows inside the first decision's reach: accumulator 6, TARF 6, extendable 3. Test debt, open: the whole-row summing law has no falsifying document (every current document settles one payment per row) — an extendable schedule with two fixings on one settlement row is the natural home; and the declared amounts convert at the deal-currency cross while `cash_settle` books `SettleCurrency` — coincident on every pricer here, parting on a reciprocal-axis accumulator no collateralised document reaches.
- **`NettingCollateralSet.cash_to_C` (the ledger's currency)** — declared amounts are reporting-currency while the balance is base-currency; a declared unit moved the net by exactly the spot (0.80 at 1.25) on a foreign-reporting book. `cash_to_C` scales by the captured cross at the settlement row; one declared unit now moves the net by one in both currencies. The identity is exact only where the cross is flat between settlement and reporting rows — with live vol the chain correctly owes `fx(t)/fx(j)`, measured 1.0183–1.0419 — so the gate runs a (T, B) cross at zero vol to keep every row gather live and the unit exact. `test_boundary_pricer_events.py::test_a_declared_ledger_row_is_read_in_the_currency_it_was_declared_in`.
- **`pv_MC_Tarf` (`use_past_fixing` arm)** — An observed fixing clamped its accrual at the block's remaining target and never zeroed the alive weight, so two observed-but-unsettled fixings each banked the whole remainder and a crossed deal kept paying its OTM leg. Decided 2026-09-02: the observed branch follows the simulated one — a per-fixing running remainder, the crossing fixing paying exactly `R`, an exact 0/1 survival — and the redemption is a *registered* dense decision chain (each row reports the decision at the last fixing its own strip observed; latch reconstruction `max|d| = 0.0` on every no-declared-reset schedule — on the declared-reset shape it read 40.3 on a scale of 1.04e3 until the accrual netting below closed the index offset, 6.1e-05 since). A block's second observed fixing also now reads its own level (the index was end-anchored through a broadcast row: 149.98 against an oracle of 349.94, fixed forward-indexed and bound through the recompute replay). Two lagged fixings read 99.98989 where they read 199.9566, and a later settlement date cannot move a redeemed deal. `test_fx_tarf_json.py::test_two_observed_fixings_in_one_settlement_lag_redeem_at_the_first`, `::test_a_redeemed_deal_pays_nothing_after_the_crossing_fixing`, `::test_the_second_observed_fixing_in_a_block_reads_its_own_level`, `test_boundary_tarf_events.py::test_a_row_reports_the_redemption_its_own_strip_took`.
- **`Credit_Monte_Carlo` (CVA under FVA)** — Turning FVA on scaled the set's MTM by survival and the scaling reached the CVA integrand, moving the shipped `cva` −0.97% / −2.38%. One `unscale` — the exact reciprocal of what `post_process` applied — at the three readings of the integrand (the reported exposure, the boundary objective, the Hessian kink): `cva` is now bit-equal with FVA on or off (120.845161437988 / 297.741149902344, 0 ULP on both mosaic sets), `fva` unchanged. Scope: the divisor is the block's own counterparty — a set naming a different one, or the survival-not-found warning path, is unmeasured. `test_service.py::test_the_cva_column_reads_the_same_whether_or_not_fva_ran`.
- **`calculation.CMC_State.quasi_rng` (the Sobol dimension cap)** — A draw wider than 21201 refused inside the pricer and killed the run downstream. Wide draws now chunk at successive positions (`SOBOL_MAX_DIMENSION`), the position ledger is keyed by the width the caller asked for, and the anchored arm strides by `span × sample_size` — successive draws distinct, zero cross-batch collisions at 32768/42407/63603, historical and anchored arms agreeing draw for draw. At or below the cap the single-chunk path writes exactly what it always wrote (hex-identical fixtures, standing position included). 32768-path OSS runs now price, unblocking the bandwidth plateau's operating point. `test_multi_gpu.py::test_a_draw_wider_than_the_engine_is_its_chunks_at_successive_positions`, `::test_a_wide_draw_advances_the_stream_by_every_chunk_it_took`.
- **`pricing.sim_spot_oss` / `sim_spot` (`dt == 0`)** — A reporting row on an observation date was simulated as a σ=1% kick instead of resolved exactly. The step now applies an exact indicator (`survives`: an up barrier is crossed strictly above the level) with the draw still consumed, so every downstream draw is unmoved; a digital read at an observation-date row is exact where it scattered over 127 distinct marks. `test_barrier_bridge.py::test_a_rebate_read_at_an_observation_date_row_is_exact`, `::test_a_row_that_is_not_an_observation_date_is_untouched`.
- **`instruments.FXPartialTimeBarrierOption.add_grid_dates`** — The deal now contributes the reporting grid the way its siblings do, gated on a knock-out rebate (a knock-in's rebate pays at expiry, which `reset` already declares): KO no-rebate 2/2 dates, KI+rebate 2/2, KO+rebate 14/14, with every existing profile and CVA hex-identical. `test_partial_barrier_json.py::test_only_a_knock_out_rebate_settles_on_the_reporting_grid`.
- **autocall × collateral chain** — The per-decision ledger flipped only the decision's own payment, so later coupons' booked cash sat in the wrong margin windows. `LatchedBoundarySet` now derives each decision's ledger reach from its declared `cash_events`: six-coupon gate 0.14% against its CRN ladder, own-row-only mutant +7.73%. `test_autocall_json.py::test_a_collateralised_cva_delta_carries_the_settled_coupon`.
- **`pv_MC_ExtendableForward`** — Registration off the pricer's own `value = fixed + state·live` split closes the CVA delta from −3.17% to −0.07%. `Exercised_By` splits the payoff's `forward_sign` from `decide_sign`, so a deal and its mirror are one exact negation — all four style×side pairs sum to 0.0 across the CMC profile. `test_extendable_forward_json.py::test_the_cva_delta_carries_the_extension_flux`, `::test_the_mirror_booking_sums_to_zero`. Fixed alongside: `_job` shared `FACTORS` by reference, so file-order readings predating the fix are suspect.
- **`pricing.pv_partial_barrier_option`** — Three structural defects (a non-positive window into `sqrt`; `touched` ignoring the window, −87% on realised payoffs; down + end-window + continuous never pricing) and two formula defects in the `eta == 0` B1 branch. Bridge-corrected MC oracle, worst 0.43% over eight configurations — `test_partial_barrier_json.py`.
- **`pricing.partial_window_rebate`** — The rebate moved the mark by 0.0000 against an oracle putting it at 14–34; both legs are now valued over the window, start-window resolved on both sides of the level. Oracle worst 0.56%, guard-removal mutant dies by 61–214%; `test_partial_barrier_json.py::test_a_touched_or_a_closed_window_pays_its_rebate_exactly_once_or_not_at_all`. Sibling correction: `pv_barrier_option`'s full-window KI rebate does carry pre-expiry value (41.65 vs 21.28).
- **`pricing.forward_carry_rate`** — `carry * dt` was the interval integral only on a flat curve, and it drove the barrier's own simulated drift: 4.276e-02 → 2.220e-16 once the cumulative integrals are differenced. Un-gated: `test_payoff_forward_survives_a_sloped_carry_curve` went with the mock-built suite and has no replacement; the measurement stands.
- **`schema.DealFields` × `Deal.__init__`** — A declared default never reached a deal's field dict, so a schema-valid block could price as zero silently. Apply-on-load at one seam, restricted to `schema.COMPLETABLE`; the blanket version was killed by measurement — an `FXBarrierOption` omitting `Strike_Price` priced 741.53 where HEAD said `nan`. Only authored keys are visible to `get`, `in`, iteration and JSON, so `plan_hash` is byte-identical. `tests/test_declared_defaults.py`.
- **`config.splice_deal`** — Gave every container an empty `Children`, so a composed candidate loaded hollow and priced 0.0 silently on two verbs. `test_a_composed_candidate_prices_its_legs_not_an_empty_container` requires the container to equal the sum of legs, each leg nonzero.
- **`pricing` (analytic barrier/option family)** — Census re-anchored — ledger inlined in `gates/pricer_branch_census.py` (`--anchors`), tracer now stdlib `sys.monitoring`. `pv_american_option` closed by `tests/test_american_option_json.py` as a bound (Black ≤ engine ≤ binomial, worst gap 2.07%, since Bjerksund–Stensland prices a sub-optimal policy); the knock-out arms by `tests/test_barrier_arms_json.py` (1.4e-14 against longhand Reiner–Rubinstein, an arm slip dying by ≥8 orders). `pv_MC_Tarf.bs_call_put_fwd` was dead and is deleted.
- **`QEDI_CustomAutoCallSwap` fixtures** — No fixture had ever priced the autocall at a non-zero rate or carry — 119 runs, counted by `gates/fixture_degeneracy.py` — and the first live-carry run caught the strip reading −8.27% off its oracle. Closed by `test_autocall_json.py`'s credit-MC exposure grid at r = 4%, q = 1% with block splitting and the `terminationDate` latch carried across rows.
- **`pricing.forward_vol_strip`** — The strip hard-coded `use_forward=True` while two of its three adopters declare `use_forwards = False`, so a smiley surface priced a different law from the quote the same pricer marks its Europeans with. The deal's flag is threaded in: the digital reads 0 ULP against its declared quote (was −0.2529%) and every repo fixture is 0 ULP. `test_autocall_json.py::test_a_single_coupon_autocall_is_a_digital` holds the 0-ULP read; the six smiley gates went with `test_vol_term_structure_strip.py`. The convention itself is an open decision.
- **`pricing.calc_vol_adjustment` (OSS half)** — The Compo `b_adj` was a python `0.0` handed to `torch.unsqueeze` — TypeError, deal skipped, so no compo OSS deal had ever priced — and `s_adj` passed a tenor where every other site passes absolute days. Compo now simulates the product `S·X`; a one-coupon compo autocall lands on its closed-form digital at 4.8e-16, both correlation signs (`test_autocall_json.py`).
- **`pricing.pv_one_touch_option`** — `Payment_Timing` tested two values with no `else`, so a third priced as whatever the last assignment left. Both deals refuse at construction. `test_barrier_bridge.py::test_an_unknown_payment_timing_is_refused`.
- **`pv_MC_AutoCallSwap` (no-averaging loop)** — The settled cashflow was booked per coupon, per unit and unsigned — the ledger read 0.24 / 0.8 / 0.8 where 0.8 / 0 / 0 pays. Four causes closed at once (the settle in the coupon loop not the row loop, booking accumulated value not the payment, `nominal` scaling only the mark, `terminationDate` never returned so every block re-paid). One `LatchedBoundarySet` carries both reaches: CVA disagreement 1.68%, each half suppressed alone +73.83% / −72.65%. `test_autocall_json.py::test_the_ledger_mirrors_and_scales_with_the_deal`, `::test_each_booked_date_carries_the_coupon_that_pays`.
- **`QEDI_CustomAutoCallSwap.calc_dependencies`** — A zero `Autocall_Coupons` row left `coupon_index` un-advanced, so the next coupon took its interval (4.41%), and a barrier dated on the row read a stale spot (24.2%). Decided 2026-09-01: refuse the document, do not complicate the loop: `utils.UnpriceableSchedule`, fatal. `test_a_zero_coupon_row_refuses_by_name`.
- **`pv_MC_Accumulator`** — `triggered = zeros` omitted the fixings a knocked deal accrued before the breach (1.07% of the profile at a 45-day lag). `LatchedBoundarySet.pending` applies the survived-weighted pending payoffs: reconstruction 7.6e-8, zero-branch mutant 156 against a 1.6e-3 bound. `test_fx_accumulator_json.py::test_a_knocked_deal_still_carries_its_pending_settlements`.
- **`HullWhite1FactorInterestRateModel`** — The λ and quanto drift legs decayed twice (63% attenuation at 10y for α = 0.10) and the quanto-vol curve was read through `.array.T`. The rule for any HW-family process: the cumsum increment carries `e^{+αs}`, and the single `e^{−αt}` lands once, at assembly. Un-gated: `test_hw1f_lambda_and_quanto_legs_decay_once` went with the mock-built suite; the rule stands, the brute-force check does not.
- **`HestonNandiModelParameters.reparam` / `bounds`** — `Gamma_Star/1000` was bounded strictly positive, so a rising smile fitted to the bound and reported convergence. Refusing at the bound is refuted — at zero leverage `Gamma_Star` is genuinely unidentified — so the leverage *share* carries the sign instead: `x[2]` signed in `[−1, 1]`, `x[3]` the magnitude, with a cold start seeding the sign off the quotes. Four-pillar USDZAR fits `Gamma_Star` −3529.45, all five strictly interior. `test_service.py::test_the_hn_verb_lands_a_fitted_factor_that_reprices_its_own_quotes`.
- **`utils.spot_model_currency` × `utils.hn_reciprocal_gamma`** — Three keyings disagreed, so a USDZAR TARF looked up `HestonNandiModelParameters.USD` while the calibration wrote `.ZAR` and it priced GBM — the forced TARF's strike separating by exactly 0.0 under model and GBM, the defect's signature, and by 3.78% after. Decided 2026-09-02: one rule at four keyings, and a base currency is a numeraire naming no block, so a book declaring none refuses. The reciprocal axis is a change of numeraire as well as of axis — `(ω, α, β, γ*)` for `s` describes `1/s` as `(ω, α, β, 1 − γ*)`, carried as `HN_Invert`, two orientations agreeing to 4.2e-6 and the map pinned in closed form at 1.4e-12. Found alongside: `R = relu(R − accr)` clamps the observed-fixing branch of both TARF arms. Gates in `test_structures.py`, `test_fx_accumulator_json.py`, `test_fx_tarf_json.py`, `test_hn_component.py`.
- **`RiskNeutralInterestRateModel.calc_loss_on_ir_curve`** — `shared_mem.reset` ran once before the batch loop, so every batch after the first re-priced batch 0's paths — bit-identical at 1, 4 and 8 batches for N× the wall clock. `t_Buffer` clears at the top of the loop and `Simulations` (8192) / `Batches` (1) are declared; (2048 × 4) and (8192 × 1) now agree to one ULP. Two gates.
- **`RiskNeutralInterestRateModel.implied_process`** — The Monte Carlo objective simulated under the base measure's quanto drift while quoting and deflating domestically: forcing ρ to zero moved simulated premiums +5.96% to +12.42% while the analytic price did not move by a bit. Decided 2026-08-31: calibrate domestically, simulate globally: the objective's process suppresses the two FX inputs, so `K ≡ 0` and the invariance table is exactly 0.0. The emission is unchanged. Five gates including the mutation.
- **`HestonNandiModelParameters` bootstrap** — `Volatility` was REQUIRED but never read under `Quote_Type` Premium, and a missing reference skipped instead of refusing. The quote type now declares which references it reads and a missing REQUIRED one refuses by name: a chain-sourced `Premium` block fits in a book carrying no `EquityPriceVol` at all, ATM residual 4.441e-16. `tests/test_equity_chain.py`.
- **`HestonNandiModelParameters` × `utils.calc_eq_forward`** — One `Discount_Rate` did two jobs and the pricer's equity forward read a third curve. With `Funding_Rate` and the dividend reference declared, the worst relative miss is 0.000e+00 at every pillar on a 125bp repo spread, against 3.68% undeclared. `test_equity_chain.py::test_the_calibrated_forward_is_the_priced_forward_at_every_pillar`. Carried: declare both on a spread market, since `equity_chain.forward.rate` feeds parity, strike placement and vega at once.
- **`create_market_swaps` × `riskfactors.InterestYieldVol`** — The HW2F calibration priced every premium lognormal and read neither `Distribution_Type` nor the declared `Shift`, leaving one ladder read both ways **9.7×–11.4× apart** in premium. Decided 2026-09-01: the premium reads `get_subtype()` — the deal path's own read — and picks a matched pair out of `PREMIUM_CONVENTIONS`, with `displacement` reading declared `Shift` first. The 365.25-vs-ACT/365 clock closed 2026-09-02, the curve's day count winning, so the round trip returns σ_N as itself. Re-marks every θ\*. Five gates plus `tests/test_swaption_vol_emitter.py`.
- **`create_market_swaps`** — A zero `Market_Volatility` fell through to `vol_surface.ATM(...)`, calibrating a blank cell against whatever the book's surface held. Decided 2026-09-01, retiring the fallthrough: an authored 0.0 and an absent column both refuse by name, and `InterestYieldVol.ATM` now has no engine consumer. `test_a_quoted_zero_refuses_and_so_does_an_absent_one`.
- **`create_market_swaps` (re-strike)** — The `Volatility_Delta` bracket assumed a lognormal scale, so a normal ladder under ~78bp was fatal rather than degraded — at F = X the Bachelier premium does not mention the strike, so the fallback was the same function. `IMPLIED_VOL_BRACKETS` is co-keyed with `PREMIUM_CONVENTIONS` and brackets multiplicatively; the re-strike round-trips to 3.3e-16 and the lognormal arm is hex-identical. `test_hw2f_analytic.py::test_the_normal_re_strike_brackets_in_its_own_scale_and_the_lognormal_arm_is_unmoved`.
- **`HullWhite2FactorModelParameters.fields` × `InterestRateCurveParameters.Points`** — Machine-fetched blocks had nowhere declared to state provenance or a two-way, so the emitters wrote undeclared keys. HW2F gains `Quote_Timestamp` / `Quote_Source` and `Points` gains `Quoted_Bid` / `Quoted_Ask` / `Timestamp`; the emitters' bytes did not move. `test_curve_strip_emitter.py::test_the_block_writes_only_fields_the_family_declares`, `test_swaption_vol_emitter.py::test_the_row_is_the_committed_schemas_own_declaration`. Block-level provenance on `InterestRateCurveParameters` is still absent by decision.
- **`config.CustomJsonEncoder` × `Config.parse_json`** — `.DateOffset` had two incompatible wire spellings, so a `MarketData.json` this engine wrote could not be read back through its other decoder. `Config.parse_period` is the one spelling both decoders call; the kwargs dict is still accepted for bytes already on disk. 38 sites on the ZAR strip; `test_market_prices_partition.py::test_one_dateoffset_wire_spelling_and_both_decoders_read_it`, `::test_the_kwargs_dict_still_reads_because_old_bytes_are_on_disk`.
- **`derivus/service.py` (`market_edit`)** — `CapturedErrors` was a plain root-logger handler, so a concurrent run's CRITICAL turned a good tick into a refusal — three runs wrote 9, 7 and 8 of 25 ticks. Decided 2026-09-01: filter on `record.thread`, single-threaded behaviour byte-identical; 25 of 25 over five runs against ~4,100 foreign CRITICAL records. Two gates.
- **`utils.make_float_cashflows` × the `DealStructure` compile guards** — `cashflow['Rate_Tenor']` was read when a rate window collapsed; no `Row` declares it, and the result was a skipped deal, root mtm 0.0 and a **succeeding** job on a leg whose healthy twin prices 4948.879641. Decided 2026-09-01: refuse by name, do not derive the tenor — `UnpriceableSchedule`, fatal at four guards. Two gates.
- **`pricing.sim_spot_oss`, `pricing.sim_spot` (averaging)** — `drift` came from the unclamped variance while `vol` used the clamped one, so a `dt = 0` step was a σ=1% lognormal kick with no Itô correction. Both consumers read one `var` and the floor is conditioned on `dt == 0`: barrier profile −0.0215%, its CVA −0.0240%, gradient entries to 7.5e-4 — and no gate in the repo could see any of it, `test_a_daily_monitored_barrier_is_not_priced_at_the_variance_floor` having gone with the mock-built suite. What holds the clamp now is the CVA-gradient gates: deleting it sends 11 of 13 gradient entries to NaN and six of them red.

## Open decisions

Every decision the board is waiting on, collected. Nothing below is blocked on work.

1. **The per-fixing smile read.** Sticky-forward moneyness or the deal's declared moneyness — both
   are defensible and only one can be the pricer's own quote. Whoever picks it up picks up a switch,
   not a revert, and rebuilds the six gates the removed read had.
2. **The compo smile coordinate**, undeclared because every fixture is flat. Same class as (1).
3. **The 36 disagreeing `.field.get` sites** (three fatal): hold a surviving fallback to its
   declaration, or leave the reads as they are. Enumerated in `tests/test_declared_defaults.py`.
4. **Which state an OSS row inherits.** RETIRED 2026-09-08 by lane 4's outer process (Built): the kit
   seeds its walk at the row's own offset and the state comes off the outer path. As it stood: the fix is shallow — the kit seeds its walk at the row's
   own offset and the state comes off the outer path — but it is a decision. LogVar2FJ re-seeds
   `(L*(t_row), 0)` per MTM row with the Jensen term measured from the row, so the row reads the
   market's `ξ` exactly; what it still lacks is the carried state, which is phase 3's.
5. **Two rates-emitter design questions**, recorded rather than decided: an OIS block is ~14 MB
    live (~26,000 authored floats on a 30Y strip, bounded by `CurveScreen.maximum_fixings`) —
    accept it through `/book/market` or build a term-authored OIS variant; and neither side rolls a
    business day (a 2Y USD OIS pays on a Saturday) — one convention on both sides, gated.
6. **The α-seed's worst benchmark.** The honesty reprice reads −6.25% against the retired seed's
    −4.64% while its rms improved 2.71% → 2.39% and the outside-3% count fell 10 → 3. The max is one
    order statistic, anti-correlated with the fit on this flat-quoted cube; gated rather than
    absorbed, owner's eye wanted.
7. **PFE vs CVA measure policy.** CVA is a Q-expectation wanting the market-calibrated outer; PFE
    is a P-quantile wanting a historically-estimated one, with the pricing kit staying
    market-implied. One run reports EE and PFE off one outer measure, so a book wanting each metric
    in its own measure runs twice under two `Model Configuration`s. Desk policy to state.
8. **`get_implied_correlation`'s two single-caller wrappers.** They would stop two callers building
    type-prefixed correlation-name tuples, but they brush the no-abstraction-ahead-of-a-second-caller
    rule. Held until a third correlation pair appears or the rule is judged to outrank it; no gate
    covers tuple literals.
9. **Flagged, not authorised.** `runtime` carries free functions over the hedge bundle in two
    clusters (Objective, Accounting) with `_UTILITY_OBJECTS` duplicated — the shape
    [Conventions](conventions.md) calls a class waiting to happen. `DealStructure`'s recursions are
    the same category.
10. **`Boundary_AAD_Window_Touch`'s magnitude.** The switch decides the sign; `add_grid_dates`
    landed 2026-09-03, so the enriched fixture and the re-measurement are now possible.
11. **The `Branch_And_Weight` default.** Prerequisites now in hand except one: the averaging arms
    refuse under the switch, so a blanket flip needs an averaging-falls-back-to-crisp rule first.
    The family question is closed — the one surviving spot model hands each fixing interval its own
    Gaussian block law, so it is admitted on the same terms as GBM. Values re-mark within their own
    MC noise at 12–23× less variance; the greeks are the prize.
12. **What the desk's quanto correlation means under a non-Gaussian residual.** The quanto arm
    applies the marked ρ to the return's total diffusive sd and leaves the NIG mixer untilted, so a
    brute-force joint simulation realises 91.6–92.0% of the applied drift at the Q-sized defaults
    (the analytic Gaussian-shocked share is 0.9306). Return-level ρ as marked, or the loading
    divided by that share so the marked ρ is realised exactly — one line either way, and the
    desk's reading of its own mark.
13. **The correlation as a leaf.** `Correlation` is a `DimensionLessFactor` and mints no leaf, so
    the quanto correlation delta reports as zero against a ladder of −10.95 per unit of ρ on a 2y
    autocall, flat to 0.001%. Three edits with a tree-wide blast radius (every document carrying a
    correlation gains a Greeks row) and its own hex gate.

## Designed, not built

**LogVar2FJ, what remains** (designed 2026-09-04; phase 1, the three pricers and the calibrator
with spec 5.4's engine bootstrapper built 2026-09-05/06 - see Built and `logvar2fj_spec.md`). In
lanes as of 2026-09-06: **the L curve piecewise-constant between ATM expiries** (spec 5.4.3, the
owner's ruling on the strip that zig-zags 14.6 / 5.5 / 20.0 / 10.8 / 13.4 vol against the market's
16.0 / 12.7 / 15.7 / 19.8 / 17.5 - a segment integral that depends on both ends is a recurrence with
multiplier -1, a flat segment's depends on its own level alone); **the averaging arm** by 2.4.1
(sample the window, truncate the prefix); **per-block checkpointing with regenerated draws** (§6)
and the deletion of the step setting (§12: the day is the model); **the reciprocal axis** (§2.8),
all four now Built above. Not yet cut: the **density recursion** (one
FFT convolution per monitored date against the block Gaussian, as an alternative inner estimator)
and the **xVA outer generator carrying `(S, l, s)`** (phase 3), which retires the per-row re-seed
phase 1 declares - the kit seeds `l = L(t_row)`, `s = 0` at every MTM row - through the
`reveal_state_at` / `inner_fork_seed` pattern the HMC fork already uses, behind a declared switch,
default off and bit-identical, flipping it re-marking every LogVar2FJ exposure profile and CVA;
it serves PFE at least as much as CVA, since a tail quantile lives on the high-variance paths a
re-seed flattens (open decision 4). Measured 2026-09-05 (`artifacts/logvar2fj/harness_cjow.py`,
`probe_rho_prime.py`): at the spec's P-sized defaults the 1y-into-1y forward slope read 5.3 vol
points against CJOW's 12.0 whether the state was re-seeded or carried, and the SPOT 1y slope
was half CJOW's too - the deficit was the (rho, sigma) sizing, not the re-seed. At the Q-sized
defaults (rho_s -0.75, sigma_s 2.4, rho_l -0.4, sigma_l 1.0, lambda 0.21) the spot 90-110 slope
ratio reads 0.57 / 0.82 / 0.86 / 0.92 at 3m / 6m / 1y / 2y, the stickiness ratio 0.96 against
CJOW's 1.01, the 1y skew -1.47 against -1.48 (marginal KS 0.015-0.055 against CJOW's exact CDF at the fitted
parameters, 0.057-0.109 at the seeded curve), the autocall -34.07 against
-32.07 at 32,768 inner, and the vanilla RMSE 0.82 / 0.33 vol points at 1y / 2y before any fit;
the curve mapping is first order in the vol-of-vol (the 1y variance lands 8% low, the 2y 9%
high), stage 1's iteration. NONE OF THE SPEC'S LEVEL-DEPENDENT LEVERS MOVES THE
STICKINESS RATIO on this surface (`artifacts/logvar2fj/probe_levers.py`, state carried, 2^17
paths): rho_s' 0 to -4 reads psi 0.96 to 0.94 (and its rho_max tanh read rho_s -0.75 as -0.60,
since removed); sigma_s' 0 to 0.8 raises the spot AND forward slopes together, 20.4 to 24.8 and
19.6 to 23.8, psi 0.96 throughout; mu_J' 0 to 1 raises the spot slope 20.4 to 24.3 and the
forward 19.6 to 22.2, psi 0.96 to 0.91; the jump share is the one lever with the right sign,
lambda 0 / 0.1 / 0.21 / 0.42 / 0.8 reading psi 0.93 / 0.95 / 0.96 / 0.98 / 0.99 at fixed jump
sizes - it cannot pass 1. A forward smile in this model is its spot smile averaged over the
state, and a parameter that moves with the state moves both; what separates the two horizons
is a parameter that moves with TIME, and the spec's fifth revision makes `rho_s(t)` and
`mu_J(t)` piecewise constant on calendar buckets. MEASURED (`probe_buckets.py`, year-two
bucket, spot 1y slope 20.38 unmoved to the digit in every row): `rho_s` -0.80 / -0.85 in year
two reads psi 0.99 / 1.02 (c 0.200 / 0.118, the second below `c_min`), `mu_J` -0.20 / -0.25
reads 1.01 / 1.09 with the forward ATM 18.5% -> 20.3% (bigger jumps add variance the curve's
year-two segment must absorb), both together 1.04 / 1.14. The lever levers; the 2y vanillas'
composition check and the autocall's forward-skew sensitivity (§8) decide how much of it to
carry. The daily walk's tape at 2,048 x 2,048 x 509 does not fit a 24 GiB card in either
direction (the draws alone are 3 x 7.95 GiB, and `Recompute_Inner_MC` replays one block's graph,
not one step's), which is what the checkpoint lane above exists for: no coarser chain is licensed
(G9 reads the 5-day gap at 3.4 SE on the coupon leg).

**Barrier state as a fold over fixings — the REMAINING half** (decided 2026-09-02, built through
2026-09-03; see Built). What is still designed rather than built is the rest of the fold's reach.
The **continuous** side: monitoring reads daily (low, high) bars under `(index, date, source)` — a
bar is a fact that brackets every intraday print, and a disputed determination is a superseded bar,
never an edited flag. `utils.bars_touched` is the predicate and is gated; the SOURCE is increment
4's, so `EquityOneTouchOption`, `FXBarrierOption`, `FXOneTouchOption` and
`FXPartialTimeBarrierOption` still price from terms alone. The **autocall**: `Barrier_Dates` on
`QEDI_CustomAutoCallSwap` is monitored discretely but its state rides `Price_Fixing`'s own observed
value, and the transformation is "a called autocall is its coupon at that fixing's settlement" —
which folds the coupon and threshold ladders together, not just the put barrier, and wants the
`BarrierIsHit` read at `pricing.py:4807` retired with it (it tests `is not None`, so it fires on
`'No'`). The **TARF and accumulator's** decisions-remain arm: folded parameters (remaining target,
`pending`) rather than a substituted deal. The fixing index is derived from the underlying factor
name; only `Fixing_Source` is declarable. Increment 4's: hydrating both fact kinds from the log,
where the book file becomes a projection.

**Spine increments 4–7** — projections plus the diary, tier policy, the doorbell, the generated
binding. The book file's rehoming as an LSN-pinned projection and the plan compiler as a fold over
fixings supersession are increment 4's.

**Sensitivity estimators as first-class objects** — a `SensitivityProfile` per pricer, so a consumer
can tell a pathwise derivative from one carrying a boundary term.

**Hessian-vector products** instead of materialising full Hessians. The named task is a `jvp` rule
on the recompute node: forward-over-reverse HVPs are tape-free and `InnerMCRecompute`'s
one-function-called-twice discipline extends to forward mode. Until then, directional CRN bumps of
the now-smooth delta, whose ladder going flat is this suite's own definition of a derivative that
exists. First consumers: the SIMM calc's dSIMM/dθ (one HVP with ∂SIMM/∂s as the cotangent), FVA's
splits.

**Incremental XVA as risk-impact v2** — a counterparty on the quote and
`CVA(book + mirror) − CVA(book)` through `Credit_Monte_Carlo`, the same two-run seam with a
different calculation in it; `service.xva_document` already composes that job over one netting set.
Also named: a ratio-solve primitive for participating forwards.

**Service layer, what remains** — SSE for progress (`/book/bloomberg`'s progress field is the shape
it would stream), a cost estimate that reads the real grid rather than a segment count, auth with
budget caps (load-bearing sooner: a cloud-hosted MCP host needs streamable-HTTP plus a real auth
story), and the web UI's edit surface — tables and curves (an editable grid) and creating deals. The
blotter's two views are owed their screens, and an `xva.json` row that also carries the exposure
profile a client would chart.

**Excel end-state** — `RF_*_PORTFOLIO` builds its job through `portfolio_service`, which still reads
`schema.mapping` directly; migrating it to `GET /schema` is the last step, after which nothing in
`excel_integration/` imports the engine. `RF_SOLVE_*` stays in process: a deal field is structural,
so a round trip per iterate would buy nothing. Solace returns later as a second transport in front
of the same verbs, not as a second queue.

**`bind=` for payoff-only deal fields.** A strike moves no discovery, but a deal field is structural
today, so `/book/solve` recompiles every iterate — which is where the case gets measured. Four
candidates stay declined with a citation: `ReferencePrice.Fixing_Curve` and `PriceIndex.Index`
become reset rows building a compiled structure, and `GBMAssetPriceTSModelParameters`'
`get_tenor_indices` / `get_quanto_fx` are value-dependent code paths, so the leaf set depends on a
number. `VolatilityGrid.Delta_Surface` is a third of that kind — `Factor2D.update` runs the Malz
solver on a grid refined against the values.

**The `System` store audit** — the last hand-written store, never audited declared-versus-read:
`Volatility_Delta`, `Master_Curves` and `Swaption_Premiums` are read by the bootstrappers and
declared by nothing; `Grouping_File`, `Proxying_Rules_File` and `Script_Base_Scenario_Multiplier`
ship in the market data and reach no read. The one piece of the schema work still owed. It stays
hand-written because its single "type" is a UI panel name and the class consuming
`System Parameters` is `Config` itself.

**Quote-sensitivity non-goals**, recorded on [the page](quote_sensitivities.md#non-goals): no report
*format* for a quote delta — it lands on `Config.quote_leaves` in one of
[two shapes](quote_sensitivities.md#the-attachment), and where two families read the same JSON
number its `dV/dq` arrives as two partials under one descriptor a consumer must sum
(2.243453e4 + 8.071709e4 measured; `structures.vol_risk` obeys it). Neither backward supports
`create_graph`, so there is no second derivative in quote space. No SABR/SSVI parameterisation is in
scope — a Malz smile is the one delta parameterisation built.

**Also owed**: `test_hmc_declared_knobs` (declared-versus-read on the `Hedging_Problem` knobs) went
with the mock-built suite and has no replacement; batching Schrager–Pelsser across the benchmark
set; and five model items in the punchlist below.

## Built

- **LogVar2FJ v2, lane 4: the xVA outer process** (2026-09-08) — `stochasticprocess.LogVar2FJImpliedSpotModel`, the xVA outer process (spec 6.1, brief
  7). An implied process on the calibrated factor the OSS kit prices off, walking `utils.lv_walk` on
  the trading day between scenario nodes with the fractional remainder as one exact shorter step;
  `(ell, s)` revealed per node, handed to the pricer's row as the carried state, and handed to an
  inner fork by `inner_fork_seed`. The framework's correlated Gaussian multiplies `sqrt(G)` exactly.
  `Checkpoint_Outer_Walk` (default `Yes`) checkpoints the walk on calendar-anchored 21-day segments
  with `use_reentrant=False`.
  
  *Measured, on the CPU.* **G3-1** outer marginal against the kit's own European at 1m/1y/3y, calls
  and puts: within `0.22`, `0.43`, `0.80`, `0.78`, `0.21` and `1.21` standard errors at 2,048 outer
  paths against 4,096 pricing paths, with `E[S_T]/S_0` reading `0.99962` / `0.99849` / `1.00372`.
  **G3-2** the shock stream and the mixer are the pricer's own function objects, and a fork seeded at
  an outer node reproduces the whole walk's `M_lev`, `A` and end state within `0.9` ulp at float32
  (`1.49e-08`) and `1.3` ulp at float64 (`4.16e-17`). **G3-3** the same three numbers survive a
  re-cut AT the fractional step (`0.728953` of a trading day) within `1.3` ulp. **G3-4** the
  checkpointed backward is the full-tape one: every one of 892 finite CVA numbers on the autocall book
  bit-identical, and on a book with no decision product 1,293 at first order and 1,468 at second order
  (`Hessian: Yes`, through `exposure_kink_term`) bit-identical; the autocall book refuses second order
  for its own boundary registration, which this switch does not reach. **G3-5** the correlation
  dilution above. **G3-6** the replay refusal is the sentence. *The tape*, on the desk deal's 5-year
  grid (1,322 daily steps, 112 nodes) at 2,048 paths: **8.3 MiB** checkpointed against **236.1 MiB**
  un-checkpointed, a factor of **28.4**, for `1.84x` the forward and `5.0x` the backward.
  **Retires open decision 4 ("Which state an OSS row inherits").** The kit seeds its walk at the row's
  own offset and the state comes off the outer path: `LogVar2FJKit.carried` reads the outer process's
  revealed `(ell, s)` at the node this row lands on, and keeps the `(L*(t_row), 0)` re-seed only where
  no such process ran. What the decision asked for is now what the engine does.
- **LogVar2FJ v2, lane S2 — the calibration on the card, and the inner bootstrap as a Newton**
  (2026-09-08). The `LogVar2FJModelParameters.device = cpu` pin is GONE: the family takes the
  device it is constructed with, which on a CUDA box is the card, and the two things the pin stood
  for are answered rather than avoided — `draw` generates BOTH streams on the host under
  `Random_Seed`'s own generator and moves them, so a seed names a draw and not a
  draw-and-a-device, and `Calculation.factor_leaf`'s existing `theta.to(device, dtype)` is a
  differentiable copy, so a CPU-sharded calculation reads a card-fitted leaf and its backward
  still reaches the fit's quote leaf (proved as a document: a `Quote_Sensitivity: Yes` ladder
  fitted on `cuda:0` whose leaf a CPU `CreditMonteCarlo` consumes, theta\* bit-identical to the
  CUDA-calculation run and `dCVA/dq` back on the CUDA leaf). `LVFit.solve_l` is a damped NEWTON at
  the pillar's own slope — the pass that prices a pillar carries a backward anyway, so the exact
  slope is free and a chord off a stale one only spends more passes: 1.64–1.70x fewer passes,
  1.33x on the clock, and a strip solved an order TIGHTER on two ladders. `verify`'s ATM gate
  compares with `not <=` and names a NaN pillar, which a `>` let through — reproduced on the
  `Sobol` 8192 run lane S found, which now refuses by name and writes no factor. **The four book
  blocks re-fit to the banked theta\* within 1e-11 absolute on three ladders and 2.6e-9 RELATIVE
  on `EUR_SD3E`'s unidentified `Alpha` valley, evaluation for evaluation and RMSE for RMSE**;
  every GBM and Hull-White document stays hex-identical (4,180 floats over 16 documents) and so
  does the crisp GBM TARF. **`JPY_NKY_BBG`: 3385.2 s at the pre-lane-S base, 742.2 s after lane S,
  and 70.5 s here** alone on the box — the designed back-to-back pair reads **588.4 s → 127.2 s**
  under an eight-process load at the same 44 evaluations and the same RMSE 0.978. The four ladders
  fit in **70–98 s** at that objective and **107–192 s** at the production defaults, alone. What
  is left is priced: `utils.ig_root`'s FIXED 34-step budget is 30% of the card's profile and is
  pure dispatch (~1,200 launches a call, the same whether the strip is one block or four), so the
  next lever is `LV_IG_STEPS`' determinism ruling — which lane S set for the PRICER's checkpoint
  recompute and not for the calibrator, which never checkpoints.
- Measured and NOT taken: **forward-mode AD for the Jacobian**. Lane S priced it at ~1.6x when the
  batched reverse backward was 61% of the CPU clock. On the card that backward is **0.104 s a call
  against the CPU's 6.78 s**, so `LVFit.jacobian` is 87% its own FORWARD residual and 13% its
  backward — `jacfwd` would ride `nx` tangents on the expensive half and cost MORE than the 25
  cotangents it replaces. Reverse stays, and `solve_l` keeps the one Python scalar its stop reads.
- **LogVar2FJ v2, lane S — the calibration in seconds.** `utils.lv_walk` is the block's closed form:
  `lv_ou_path` solves both OU factors over the whole block at once (the transition factorises, so
  the path is one cumulative sum), `lv_state_variance` is the same closed form at twice the
  reversion, and the clock, the leverage mean and the quanto drift are elementwise with one
  reduction each — the per-step scan leaves the engine and lives in
  `artifacts/lv_fast_20260908/oracle.py` as `lv_walk_scan`, agreeing with the engine to **1.5e-14**
  over 360 cases (every block length 1…562, `invert` both ways, with and without a quanto loading
  and with bucketed levers) and to 4e-13 at a 20-year horizon. `Invert_Spot` keeps the recursion,
  the measure change being state-dependent, and shares the block sums. `LVFit.walk` draws the whole
  strip's mixers in ONE `ig_quantile` over `[paths, blocks]` — the variance path owes the residual
  nothing, so every clock is known before any mixer is — and `ig_root`'s fallback bisects
  geometrically, which halves its fixed budget. `cap_headroom` reads the same closed form instead of
  spelling the scan a second time. **The four book blocks re-fit to the banked θ\* within 1e-11,
  evaluation for evaluation and RMSE for RMSE**; every GBM and Hull-White document is hex-identical,
  the NIG base valuation is hex-identical, the two LogVar2FJ limit documents move one ulp, and the
  collateralised CVA — the one number that passes through a threshold in the float32 scenario walk —
  moves 9.0e-7. Walk 2.2x forward and 2.8x with its backward, the batched Jacobian 3.2x, and a full
  `JPY_NKY_BBG` ladder **3385.2 s → 742.2 s** back to back on one box at the same 44 evaluations and
  the same RMSE 0.978. The closed form is also what makes `Paths` a lever at all — the scan's cost
  was 10,350 dispatches whatever they carried, where the vectorised walk is very nearly linear in
  the path count. `Sampling` is a declared field defaulting to `Pseudo`, and every default this lane
  touched or left is in the table below with the number that set it. One incidental fix: `solve_l`'s
  chord step divided by a stale slope guarded only against `None`, and a slope of exactly zero
  raised — a Sobol stream reached it.
- **LogVar2FJ v2, lane 3: the residual's P-law, and no outlier mask** (2026-09-08, brief 8) — the
  diffusive remainder's law is now `NIG(α^P, β^P, δ_A, μA)` fitted by maximum likelihood on the
  day's own variance budget, and **the 4σ outlier mask is gone**. WHY: an NIG law has tails, so a
  day the Gaussian filter would have thrown away is a day the residual's own law explains; only
  declared `Event_Days` are excluded, and the count a 4σ threshold *would* have flagged is
  reported beside the likelihood the tails buy. The mask's fixed-point iteration goes with it, so
  the filter is fitted ONCE. `torch` registers no derivative for its Bessel ops, so `BesselRatio`
  (`K₀/K₁`, whose own derivative `r² + r/z − 1` is a function of itself, hence differentiable to
  every order off one evaluation) and `LogBesselK1` carry the density and its Hessian on the same
  tape the Kalman half uses; the SAME ratio is the posterior mean of the mixer, `E[G|x] =
  (q/α)K₀(αq)/K₁(αq)`, which is what `eps` is now standardised by — brief 7's *Gaussian given the
  mixer*, the object the framework's correlation is applied to.

  **Two things the build had to get right.** The budget the return is standardised by is the
  filter's ONE-STEP PREDICTION and not its filtered value (`lv_predicted`): a range measurement is
  built from the DAY'S OWN path, so standardising by the filtered `h` shrinks exactly the days
  whose residual was large, and it turned a left-skewed residual into a right-skewed remainder —
  the standardised remainder's skew reads **+0.835 against the filtered budget and −0.107 against
  the predictable one**, and an oracle remainder built from the shocks that were actually drawn
  **+3.83 against −0.681**, where the law's own is negative. That is the endogeneity brief 1's
  predictable budget exists to forbid, measured. And the residual's clock
  is `A = c_eff·V` with the SHARE FITTED, not `c = 1 − ρ_s² − ρ_ℓ²` imposed: the smoothed shocks
  are attenuated, so the remainder keeps what the leverage regression could not take out, and with
  `c` imposed `Var(X_A) = A` is an assumption the data contradicts by a factor of three. `c_eff`
  against `c` is that attenuation reported.

  **RECOVERY** (`artifacts/lv_estimator_20260908/estimate.py recover fine`), five years simulated
  from `utils.lv_walk` and the engine's own NIG draw at the Q-sized defaults, ONE MIXER PER DAY (a
  day is a block), 1,300 intraday prints: the particle gate passes at **1.23%** and every parameter
  is inside two standard errors of its truth except the leverage SPLIT and `α` — `L` **+0.17**,
  `Drift` +0.28, `κ_ℓ` +1.10, `σ_ℓ` +1.25, `κ_s` +1.26, `σ_s` +1.15, `μ` −0.26, `β^P` **+1.51**,
  against `ρ_ℓ` −2.81, `ρ_s` **+6.20** and `α^P` **+4.45**. The leverage's MAGNITUDE comes back to
  0.7%: `|ρ|` **0.856** against the truth 0.850, with `c` 0.267 against 0.278 — what is not
  identified is which factor carries it, which is 5.5.2 as a number. `α^P` reads 200 against 44,
  biased TOWARD GAUSSIAN, because the remainder's clock share is **0.997 against the model's
  `c` 0.267**: at the Q-sized leverage the remainder is nearly all leverage the smoother could not
  remove, so what the NIG sees is the remainder's law and not the residual's, and the report says
  so by name.

  **WHY THAT IS NOT A BUG, in one number.** A day's own fast log-variance shock is **0.149**
  against a measurement noise of **0.932**: one day is **0.16 of a standard error**, so the
  smoothed shocks correlate **0.196 / 0.141** with the shocks that were actually drawn, and an
  oracle remainder built from those drawn shocks keeps 0.182 of the budget where the smoothed one
  keeps 0.874. Part of that noise is IRREDUCIBLE: under this residual a day's quadratic variation
  is the leverage's share plus the MIXER, whose own dispersion is **0.286 in logs** (CV 5.50) and
  falls with no number of intraday prints — only the range estimator's share does. That is why the
  gate REFUSES the same world on a 260-print bar at **10.99%** h-path RMS, naming the measurement
  (`σ_u` 0.932) rather than a threshold, where the 5.5 lane's Poisson residual passed the same bar
  at 1.54%.

  **The residual's law where the leverage is small.** On a history simulated from lane 2's own
  banked USDZAR fit (`ρ_s` −0.394, `ρ_ℓ` −0.200, so `c` 0.805) on a 260-print bar, the gate passes
  at **1.19%** and everything is inside 1.4 standard errors but `α`: `β^P` **−20.47 ± 7.40 against
  the truth −15.40 (−0.69 SE)**, `ρ_s` +1.22, `ρ_ℓ` −0.58, `σ_s` +0.97, `σ_ℓ` +0.86, `L` +0.26,
  `α^P` +4.39 (130.8 against 72.5, a ratio of **1.80**, under the ×2 flag). So the residual's law
  is identified in proportion to how little of the return the leverage takes, and `α^P` carries a
  systematic bias toward Gaussian in every arm.

  **THE TAILS ARE REAL ON THE ONE REAL SERIES.** 15 years of platinum LME closes, `log r²` mode,
  3,855 days: the NIG remainder reads log-likelihood **11,227.45 against the Gaussian's 11,104.79 —
  a gain of 122.65 nats over two parameters** — with **13** days a 4σ mask would have thrown away
  and none masked. The gate passes at **0.63%** (5.5's 0.65%) and the fast factor still disappears
  (`σ_s` on its 0.05 bound), the whole log-variance dynamic being the slow factor's: `κ_ℓ`
  **0.700 ± 0.384** and `σ_ℓ` **0.5666 ± 0.1208** at a level of **−3.071 ± 0.197** (21.5% vol),
  against the masked fit's 0.672 / 0.5355 / −3.096. The same gain reads 191.76 nats on the USDZAR
  arm, 16.37 on the recovery and 18.64 on the closes-only fallback.

  **`log r²` costs the fast factor outright**, not 46% of its standard error: on the SAME series
  with the bar columns dropped, `σ_s` sits on its 0.05 box bound (−2.67 SE) with `κ_s`'s standard
  error infinite and `ρ_s`'s 11.09 — at a measurement noise of 2.22 a factor reverting at 6/yr is
  invisible — while the slow pair and the level are unmoved (`σ_ℓ` −0.25, `L` +0.32) and the gate
  passes at 0.96%.

  **The correlation object is the Gaussian GIVEN THE MIXER.** `eps = (u − μA − βĜ)/√Ĝ` with `Ĝ`
  the mixer's posterior mean, against the alternative of standardising by the law's own sd: on the
  simulated history the two read correlations of **+0.165 and +0.166** against a drawn 0.60, and
  the choice is made on SHAPE — excess kurtosis **+0.12 against +1.11**, sd 0.946 against 1.005 —
  because a Pearson correlation on a fat-tailed series is a few days' arithmetic. Through the
  framework on the fine bar, `eps` against a sibling drawn at 0.60 on that same object reads
  **0.184** (GARCH, which indexes its delta at the return's END date) and **0.002** (GBM, whose
  `utils.calc_statistics` indexes at the START — the one-day shift the 5.5 lane opened and this
  lane still does not fix). The dilution from 0.60 to 0.18 is the leverage the smoothed shocks
  cannot remove; it is the same 0.16-of-a-standard-error statement.

  **THE SEAM CLOSES ON THREE READERS.** `utils.LV_SLOW_HISTORY` grows from `(Rho_L, Sigma_L,
  Rho_L_SE, Sigma_L_SE)` to `(Rho_L, Sigma_L, Alpha, Beta, Rho_S)` — each with its own `_SE`, which
  is now the shape rather than a list — and the USDZAR ladder cut to 6m and beyond, fitted over the
  block this lane writes, reads all three: *Alpha 200.4857 is held at the history's alpha^P*, *the
  leverage prior on Rho_S −0.2020 from the history's estimate, Rho_S SE 0.0884*, and *Rho_L −0.8319
  and Sigma_L 1.3895 held at the history's estimate, Rho_L SE 0.1535 and Sigma_L SE 0.3119*, at a
  wing RMSE of **0.768** vol points unweighted over 11 quotes. The same job with `Alpha_SE` stripped
  refuses in **0.0 s** by name, from the WRITE side, naming the whole shape.

  **Not built:** the joint QML spec 5.5.2 defers, which is what would identify the leverage split
  and with it the residual's own law — the return as a second measurement whose loading `√ĥ_t` is
  known at `t`; and the deconvolution that would read `α^P` off a remainder the leverage still
  contaminates (the remainder is `NIG(A) ⊛ N(0, kV)`, whose density is one quadrature over the
  mixer's own quantile and whose `k` the filter already computes). Until either lands, `α^P` is an
  UPPER bound on the residual's tail thickness, which is the conservative direction for a seed and
  a sanity check and is why brief 8 crosses it as neither a value nor a belief. Engine net **+165
  lines**. Proof documents: `artifacts/lv_estimator_20260908/estimate.py recover | recover fine |
  choice | closes | banked | pvq | seam | refuse`, `lv_nig_20260907/hexcheck.py` +`hexdiff.py`,
  `campaign/tarf_hex.py`.
- **LogVar2FJ v2, lane 2: the calibrator's priors, the symmetric box and the forward block as
  routine** (2026-09-08) — the fit now carries a LEVERAGE PRIOR it can never be without, a FLOOR on
  the residual's own shape, a SYMMETRIC `ρ_s` box, the forward-smile block at stage 3 by default,
  `α` seeded and pinned from a history, and the model half of a forward-skew reserve. WHY: vanillas
  do not close this model. `β` and `ρ_s σ_s` bend the spot smile the same way, so lane 1's
  vanilla-only fit gave the skew to whichever was cheaper — on a symmetric FX smile that meant
  `β → −0.08` and `α` walked to 5.45, an `α·δ_A` of 7.7e-03 at one month where nothing in brief 2
  bounds `α` from below. The prior states the split the vol market believes, the floor states that
  the residual is a TAIL and not a convexity dial, and the forward rows at stage 3 are the only data
  that sees the split at all. THE AXIS is declared where the desk owns it: `derivus_bloomberg/seed.json`
  carries `leverage_prior` per pair on the ENGINE's axis, so a `+0.4` view on an EM cross quoted
  USD-per-currency is `−0.4` on `FxRate.ZAR` in a USD book, `security_map.leverage_prior` reads the
  desk's `$DV_HOME/seed.json` first (a seed predating the key declares none), and `fx_surface_block`
  writes it into the block with `Quote_Source` naming the seed. MEASURED on the banked 22-rung USDZAR
  ladder at `Paths` 2048: with the floor OFF the fit walks `α` to **0.627**, the admissible map's own
  floor, reading `α·δ_A` **7.3e-05** at one month — lane 1's 5.45 was a way station and there is no
  bottom; any floor at or above **0.1** lands the same θ\* (`α` 48.5, shape 1.358), so the floor
  excludes a BASIN rather than setting a value; and with the desk's own **−0.4** beside it the
  ladder's 16 wings go **0.196 → 0.098** and its one-month wing **0.305 → 0.134**, under the Poisson
  residual's 0.174 for the first time in `Global` and still 4× `Bootstrap`'s 0.032. The stationary
  log-vol sd comes back INSIDE the 0.4–0.9 band with it (0.353 → 0.413 → 0.515), which answers lane
  1's open question 18.2 on this ladder without moving the band. A block declaring `+0.4` overrides
  the seed, lands `ρ_s` at **+0.4272** — a sign the one-sided box could not reach — and pays 0.098 →
  0.595 on the wings, so the prior picks the sign and the data keeps the last word. The forward rows
  are what IDENTIFY the split: the polish spectrum's two smallest directions read 0.6699 / 0.2870
  with them against 0.4135 / 0.0847 without on the FX ladder and 0.4329 / 0.1199 against 0.1499 /
  0.0317 on the SPX chain, the two largest unmoved. `Model_Priors: Off` is the objective AND the box
  lane 1 fitted on, and a document declaring none of the new fields refits to the bit under it (58
  floats over two fits, 0 mismatches). The fit reports the efficiency factor, `α·δ_A` at 1m and 1y
  against the floor, the variance-swap strip beside `ξ`, the prior in force with its source, `α^P`
  against `α^Q` flagged past 2×, and the reserve line per forward tenor. A forward tenor more than a
  quarter of Δ from its nearest rung is DROPPED by name, which is what keeps a three-week ladder from
  walking to two years. NOT BUILT: `Forward_Smile_Source: Reference` is still unexercised, no
  reference model being wired; `α^P` has no producer until lane 3 writes it; the floor is one number
  for every asset class where the two ladders here read three orders of magnitude apart at one month.
- **LogVar2FJ v2, lane Q: a quanto payoff prices under the walking kit** (2026-09-08) — the
  payoff-currency measure change enters `utils.lv_walk` as a per-step drift `−ρ_q σ_FX,k √(V_k δ_k)`
  read off the state's own variance budget, and the deal-level lognormal carry
  `pricing.calc_vol_adjustment` derived from an implied ATM vol is handed back as zero on that arm.
  WHY: `QEDI_CustomAutoCallSwap.calc_dependencies` refused every Quanto payoff under a non-GBM
  `SpotModel` because the carry was a lognormal quantity no leg of the walk reads — and a quanto
  autocall is the deal the desk actually books (the Barclays ISDA book: NKY, SX5E, SD3E paid in
  ZAR). MEASURED: the quanto GBM limit reads **7.6e-16 / 9.8e-16** relative at ρ = ∓0.4 and the
  ρ = 0 document is **hex-identical** to its single-currency twin; a brute-force joint simulation
  of the walk and a correlated lognormal fx at 2^17 paths puts the kit's own quanto-adjusted
  forward within **0.03–0.26 SE** at a flat state and **0.68–1.70 SE** at the Q-sized state over
  1m/1y/2y/3y under a Gaussian residual, where an expiry ATM vol sits 5.2–30.2 SE out and the
  leverage sd alone 17–133 SE — the oracle pins `√V_k` per step; under the NIG residual the
  untilted mixer realises **91.6–92.0%** of the applied drift against the analytic
  Gaussian-shocked share `ρ_ℓ² + ρ_s² + E[G]/V = 0.9306`, which is the design's stated choice
  measured and an open decision (below); the smooth CRN ladders on a quanto document read
  **0.15%** on spot, 0.01% on `ξ`, 0.26–1.15% on `β`/`ρ_s` and **0.00%** on the fx surface's ATM
  rows — the quanto vega comes out with the rest — while the CORRELATION delta is reported as zero
  because `utils.DimensionLessFactors` excludes `Correlation` from the leaf set, its ladder reading
  −10.95 per unit of ρ flat to 0.00%; **7,812 floats over 22 documents hex-identical**, the one
  intended change being a quanto LogVar2FJ document that refused before. THE DESK'S DEAL: the
  rebooked NKY V2 229524957 (quanto ZAR) off a NIG fit of the banked chain block reads
  **−44,084,518 ZAR under LogVar2FJ against −42,084,088 under GBM** (4.75% more negative), the
  quanto drift's own contribution +9.26m against +11.72m — the walk applies 21% less drift than
  the lognormal expiry-ATM read — and with the drift off the two models sit within 0.85%, so on
  this deal the model difference is almost entirely the quanto arm; quanto vega 70.4m against
  87.2m per unit of fx vol. NOT built: the barrier/binary siblings still refuse a quanto under a
  walking kit (only the autocall passes the loading), and the reciprocal axis has no quanto
  document to gate on — `FXAccumulatorOptionDeal` and `FXTARFOptionDeal` carry no `Payoff_Type`
  field, so `invert` and quanto cannot meet on any deal in the tree. Engine net +67 lines. PROOF
  DOCUMENTS: `artifacts/lv_quanto_20260908/` (`quanto.py limit | slope | greeks | gamma | refuse |
  cva`, `oracle.py`, `hex.py`, `deal.py refit | price | cva`).
- **LogVar2FJ v2, lane 1: the NIG residual replaces the Poisson co-jump and `ξ` replaces `L`**
  (2026-09-07) — the part of a return leverage does not explain is now a normal-inverse-Gaussian
  increment on the variance clock, sampled as a Gaussian given one inverse-Gaussian mixer per
  residual draw, and the stored curve is the EXPECTED FORWARD VARIANCE `ξ = E[h]` with the OU level
  `L* = log ξ − ½Var(ℓ+s)` derived inside the model from the ROW's own start. WHY: the Poisson
  intensity `λ` was not identified by vanillas, so it had to be fixed by a rule, re-derived per
  segment and excluded from the Greeks; NIG is the same conditional-Gaussian idea with a
  *continuous* mixer, infinitely divisible in its clock (so it composes exactly), with two
  identified shape parameters, a martingale drift that is FORCED rather than fitted, a closed-form
  European oracle and no parameter the Greeks cannot reach. The price is one monotone scalar root
  per residual draw, `utils.ig_quantile`, whose value is `ig_root`'s off the tape and whose
  derivative is two Newton steps taken at that root with `ig_cdf`/`ig_pdf` on it — the IFT exactly
  at first order and the Newton map's own at second. MEASURED: the v2 notebook's `walk` reproduced
  to **1.1e-16** on `M_lev` and **1.7e-18** on the clock with the end state BIT-IDENTICAL, and its
  block law to 2.1e-13 / 9.5e-15 when both mixers are run to one tolerance; the quantile round trip
  **< 6.6e-15** at every clock from 1e-6 to 0.2 against the 1e-10 gated, in at most 38 Newton steps
  of a fixed 60; the martingale within **0.06–0.47 SE** at 1m/1y/2y/3y and the IG-MGF identity
  1e-16 to 4e-15 in closed form; composability bit-identical on the state and 2.2e-16 / 1.4e-17 on
  the two sums; the Esscher oracle **0.16 SE** from the tilted-mixer MC and the reciprocal axis's
  own `E~[1/S] = 1` **0.17 SE**; OSS against a brute force drawing every step **−0.87 SE** (one
  fixing per coupon) and **−0.34** (five), the averaging arm −0.13 to −0.75 over both window
  lengths and both `Barrier_Observation` values; the smooth CRN ladders **0.01–0.05%** on spot, the
  curve, `σ_s`, `σ_ℓ` and rates, 0.13–0.61% on `α`, `β` and `ρ_ℓ`, 1.64% on `ρ_s`, gamma **0.01%**
  and the `(α, β)` and `(α, ξ)` Hessian cells 0.28% and 0.25%; the Gaussian limit **8.2e-16**
  relative to GBM with the CVA one float32 ulp away; the NIG limit reproducing a flat 20% to
  **1.1e-05 vol points**; **4,180 floats over 16 GBM and Hull-White documents hex-identical before
  and after**; the credit MC at 2,048 × 2,048 uncollateralised in **9,990 MiB / 110 s** against the
  Poisson walk's 8,392 / 70.5, with `Recompute_Inner_MC` on and off hex-identical on the CVA and on
  its spot gradient; and the fit itself in `market_prices.md`'s own table, where the one-month wing
  reads both ways and `Global` on FX lands at an `α` nothing bounds from below — the calibrator
  lane's first item. RETIRED with their prose: the Poisson residual, `lv_counts`, `LV_MAX_JUMPS`,
  the λ strip and the 5.1 rule, `Nu`, `Mu_J`, `Sigma_J`, `Lambda`, `L_Curve`, `Jump_Share`,
  `Diffusive_Share`, `Wing_Strike`, the jump-share box and its bounded scalar search, the split
  report's jump and Jensen columns, the estimator's jump outputs and the particle gate's
  Poisson-mixed measurement. Every retired name REFUSES BY NAME at load, on the factor and on the
  block, naming its replacement. CLOSES the reciprocal axis's `dV/dMu_J` row by construction; the
  forward-skew lever is now `Beta(t)` on the calendar buckets `Mu_J(t)` carried. Engine and tests
  net +95 lines. PROOF DOCUMENTS: `artifacts/lv_nig_20260907/` (`oracle.py mixer | walk |
  identity`, `trials.py limit | nig | base | greeks | buckets | refuse | cva`, `hexcheck.py` /
  `hexdiff.py`, `fit.py fx | quotes | seam | refuse`, `averaging.py`, `reciprocal.py`,
  `recompute_parity.py`).
- **The CJOW harness is retired** (2026-09-07) — its step 0 priced its reference premiums with the
  Heston-Nandi half of `utils`, retired 2026-09-06, so nothing reproduced it, and a reference that
  was never market-validated is another model, not evidence. Deleted from `artifacts/`:
  `logvar2fj/harness_cjow.py`, `bootstrap_cjow.py`, `c_min_dial.py`, `jac_check.py`, `roundtrip.py`,
  `cjow_documents/`, the nine banked `fit_*.json` CJOW premium documents and their outputs;
  `lv_forward_20260907/` and `fx_gate/row45_autocall.py` whole; the `cjow` verbs of
  `logvar2fj/flat_l.py` and `lv_split_20260906/split.py` with every helper they alone reached.
  `Forward_Smile_Source: Reference` stays as code and its report line says *unexercised: no
  reference model wired*; the model document's second reserve line is re-based to the deal's own
  sensitivity over the prior's declared band, which the calibrator lane writes.
- **Spec 5.5's historical estimator: `stochasticprocess.LogVar2FJCalibration`** (2026-09-07) - the
  P-measure half of LogVar2FJ, whose purpose is the correlation matrix and the priors and NOT the
  pricing parameters. `log RV_t = l_t + s_t + u_t` is a linear measurement of the model's OWN
  two-factor OU state stepped by `utils.lv_ou_step_weights`, so the Kalman likelihood is exact;
  `scipy` L-BFGS-B maximises it on the AAD gradient and the standard errors come off the SAME
  tape's inverse Hessian - AAD and not finite differences precisely because the standard errors ARE
  the deliverable (stage 4 pins the slow pair by them) and a differenced Hessian of a 1,260-step
  filter has no step both large enough to see curvature and small enough not to be noise. The
  measurement is chosen by the archive's own columns from ONE registry, `LV_MEASURES` - Yang-Zhang
  where the bar carries opens, Garman-Klass without them, `log r^2` at the log-chi^2 mean -1.27 and
  variance 4.93 otherwise - and the range modes carry the lognormal offset `-sigma_u^2/2`; on the
  simulated bar `log r^2` reads bias **-1.34** and sd **2.18** against the law's -1.27 and 2.22.
  Three things the build had to get right and the documents measured wrong first: the jump mask
  and its fixed point were retired with the Poisson residual (lane 3, 2026-09-08) - what the fixed
  point measured stands as the reason a masked fit and its reported jump set must be one series,
  and the NIG law removes the need for either; the leverages regress the diffusive remainder on
  the SMOOTHED state shocks, because the FILTERED increment is `k_j v` in both components, exactly
  proportional, so it cannot separate the two leverages at all; and the pair is ridged onto the
  model's own `utils.LV_C_MIN` box, which is now one constant both `riskfactors` and this class
  read. `Rho_L`, `Sigma_L` and both standard errors are written under `utils.LV_SLOW_HISTORY`,
  which is where the surface's stage-4 pin reads them, beside `Kappa_S` / `Kappa_L` as priors, the
  sanity table's P side, and `Lambda` / `Mu_J` / `Sigma_J` / the level, which never cross.

  **RECOVERY** on a five-year daily bar simulated from `utils.lv_walk` itself at the Q-sized
  defaults, 260 intraday prints a day, calibrated through `Config`'s own `Calibrations` entry
  (`artifacts/lv_hist_20260907/hist.py recover fine`): `Kappa_S` -0.39, `Sigma_S` -0.90, `Rho_S`
  -0.46, `Kappa_L` +0.28, `Sigma_L` -0.32, `Rho_L` +0.70 standard errors from their truths, and
  the slow pair's standard errors are 248% and 339% of the estimates - 5.5.2's "reported with its
  standard error rather than trusted" as a number. The level lands 3.2 SE low, and that SE is
  itself conditional on a fitted `Kappa_L` of 1.39 against the true 0.5, so a five-year sample
  cannot say where the level is either; 5.5.3 already forbids it crossing. The particle gate
  passes at **1.54%** h-path RMS with the log-likelihood **-1649.23 (replicate sd 0.60)** against
  the Kalman's -1648.07.

  **THE CROSS-CHECK GATE IS NOT DECORATION.** The same truth on the contract's 26-print bar
  refuses BY NAME at 11.0%: 26 prints a day give the range estimator `sigma_u 1.13` against a
  260-print bar's 0.83, and at that noise `z = (r - drift)/sqrt(h_hat*delta)` has tails a 4 sigma
  threshold reads as 2 jumps in a history containing none - which is the measurement noise the
  refusal names. A jump sample of two diffusive days is a `lambda` of 0.4/yr at `sigma_J` 0.027,
  a 27x variance inflation the particle filter's Poisson mixture can then blame every high range
  on. The reduced-USDZAR round trip refuses the same way and RAISING the threshold does not
  clear it - the same two days cross at 5 sigma as at 4 (17.6% against 18.5%) - because that
  surface's `Sigma_S` of 3.42 is a stationary log-vol spread of 1.11 and the filter lags the
  true variance by more than a Gaussian tail allows. No threshold on `z` separates a jump from a
  day the filter has not caught up with, which is a statement about a history with the
  vol-of-vol of a fitted FX surface and not about the threshold.

  **What history identifies, measured.** What replaced the jump sample is the residual's own
  `(alpha^P, beta^P)`, identified in proportion to how little of the return the leverage takes
  (lane 3, 2026-09-08, which also re-read the `log r^2` fallback: the fast factor is lost outright
  there, not 46% of a standard error). The P-vs-Q round trip on the banked
  reduced-USDZAR factor reads `Sigma_S` **0.998**, `Rho_S` 1.27, `Kappa_S` 1.35 - and `Sigma_L`
  1.75, `Kappa_L` 2.02, `Rho_L` **-1.64** (the sign flipped, standard error 0.53) - so the fast
  factor round-trips and the slow one does not, which is exactly the split 5.5.2 and 5.5.3 draw.

  **THE SEAM CLOSES.** A job carrying the estimator's own `Price Models` block for ZAR beside the
  reduced USDZAR ladder fits with the pin reading it: *pinned: not identified by this ladder -
  Rho_L +0.2928 and Sigma_L 0.5544 are held at the history's estimate, Rho_L SE 0.9913 and Sigma_L
  SE 1.3743 (spec 5.5.3)*, at a ladder RMSE of 0.062 vol points unweighted over 15 quotes, and the
  three-way table beside it reads wing RMSE 0.271 / 0.303 / 0.610 and 5-year log-vol sd 0.489 /
  0.528 / 0.682 under the floor, the class default and the index seed. The same job with
  `Sigma_L_SE` stripped from the block refuses in 0.0 s by name, from the WRITE side this time.
  On the one real series - 15 years of platinum LME closes, the `log r^2` mode - the gate passes at
  0.65% and the fast factor DISAPPEARS: `Sigma_S` sits on its 0.05 box bound (which the report
  names, with `Kappa_S`'s standard error infinite beside it) and the whole log-variance dynamic is
  the slow factor's, `Kappa_L` 0.672 +- 0.367 and `Sigma_L` 0.5355 +- 0.1138 at a level of -3.096
  +- 0.193 (21.3% vol) with 13 jumps at 4 sigma in 3,855 days. Squared returns at a measurement
  noise of 2.22 cannot see a factor reverting at 6/yr, which is the efficiency loss stated in the
  currency that matters.

  **Found, not this lane's to fix.** `utils.calc_statistics` indexes its `delta` at the return's
  START date while `GARCHSpotCalibration`, the HMM and the basis classes index at the END, so every
  correlation between the two families is one business day out: ONE simulated sibling series read
  by both classes reads **0.009 against itself**, and against this estimator's `eps` **0.335**
  (GARCH) versus **-0.008** (GBM), where the sibling was drawn at rho 0.6 on the LogVar2FJ return's
  own idiosyncratic Gaussian. The dilution from 0.6 to 0.335 is the leverage the SMOOTHED shocks
  cannot remove - the slow shock comes back with a standard deviation of 0.083 against the
  unit one it has - and since the shocks are already the posterior means, no estimator on this
  data does better. FIXED here (one line): `Config.parse_json` normalised the
  archive index by `dtype == object`, which pandas 3's string dtype is not, so every calibration
  sliced on date strings.

  **Not built:** spec 5.5.1's intraday realised-variance measurement and 5.5.2's bipower variation
  (no archive convention carries intraday returns; `LV_MEASURES` is where a fourth mode goes), and
  v2's joint QML, which the spec defers. The framework cannot finish the Correlations half until
  6.1: `calibrate_factors` constructs the process to NAME the correlation and
  `LogVar2FJImpliedSpotModel` is phase 3, so every calibration document drives the framework path
  to that refusal and finishes the two steps after it itself. Engine net **+385 lines**. Proof
  documents: `campaign/lv_hex.py lvhist`, `campaign/tarf_hex.py` (both hex-identical),
  `artifacts/lv_hist_20260907/hist.py recover | recover fine | recover pjump | closes | sector |
  banked | pvq | seam | refuse`.

- **The forward block on the autocall, re-read** (2026-09-07, spec 8 steps 2-4 / 5.3 as patched)
  - the flat-L / λ(t) fit's 3.8% is not the forward block's to close, and the block makes it
  worse. `Forward_Smile_Source: Reference` on CJOW's own forward smiles as DIFFERENCE targets,
  `Paths` 8192, at ONE bucket and at a YEAR-TWO bucket, moves the 2y SPX autocall from **-30.855
  (+3.78% of CJOW's -32.068)** to **-29.817 (+7.02%)** and **-29.891 (+6.79%)** - the second
  bucket worth 0.23% of the deal - and spec 5.3's failure mode fires BY NAME on the vanilla guard
  in both (+0.258 and +0.287 vol points over stage 5, the 29 wings 1.005 -> 2.034 / 2.040) while
  the composition check does NOT (0.031 at 2y against 0.3). The block reaches its target where it
  is identified - the year-two bucket takes `Δ_skew` at 1y-into-1y from -0.38 to +0.32 against
  CJOW's +0.19 and `Δ_bfly` to +0.04 against +0.04, `Rho_S[1y]`'s column norm 1.31e-02 with the
  forward rows against 3.11e-03 without - and OVERSHOOTS at 6m-into-6m (-1.92 -> +0.89 against
  -0.96), one `w_J` and one year-one pair serving both sub-year tenors. It pays for it with a less
  leveraged, more jump-driven law (`ρ_s` -0.63, `σ_s` on its 5.0 box, realised `w_J` 35.1%) whose
  70-80% wing - the one the knock-in put reads - degrades +3.15 vol points RMS: a cheaper put is a
  less negative swap. So step 8.4's stated expectation (within ~1% with the target) is FALSIFIED
  on this surface. TWO RESERVE LINES for the model document: the SPLIT reserve **±4%** (-33.651 /
  -32.875 / -32.800 / -30.855 against -32.068, a band of 8.72%), unchanged and written down
  whatever step 4 shows, becoming ±6% for a book that uses the block; and the
  SURVIVAL-CONDITIONING residual **+6.79%** - a percent or more, therefore by the owner's rule the
  difference between the forward-start law (matched at 1y-into-1y to 0.13 vol points of `Δ_skew`
  and 0.00 of `Δ_bfly`) and the survival-conditional law the coupon strip and the knock-in put
  read, a property of the two models' conditional structures and NOT a bucket refinement -
  carried with the qualification that the vanilla guard fired, so part of it is a worse marginal
  fit rather than conditional structure. Found, not blocking: `Reference` measures the 1y-into-3m
  target difference against a 3m rung quoted only to 105%, so its target `Δ_skew` reads -9.98
  where an independent walk of CJOW reads -33.79; the engine warns by name at load. No engine
  change. PROOF DOCUMENTS: `artifacts/lv_forward_20260907/forward.py step2 | one | two |
  autocall one | autocall two | step4`.

- **`Slow_Factor_Prior` by asset class, the floor as the guard, stickiness targets that are
  DIFFERENCES** (2026-09-07, spec 5.2 stage 4 / 5.3 / 5.4.4 / 5.5.3) - the two questions the split
  lane left open, closed by measurement. THE PIN'S LEVEL IS A FIELD WITH A CLASS DEFAULT: where the
  ladder carries no wing at 18 months or longer, `(ρ_l, σ_l)` are read in one order -
  `Slow_Factor_Prior` where the block declares one, else a LogVar2FJ history for the underlying in
  `Price Models` (the shape declared once in `utils.LV_SLOW_HISTORY`, reported with both standard
  errors; the ESTIMATOR is the next lane's, only its reader is built), else the default off the
  factor type `Underlying` resolves to - a REGISTRY keyed by that type: **FX (0.2, 0.5), an index
  (0.4, 1.0)**, the magnitude, whose SIGN is that of the `Rho_S` in force at the pin so the slow
  skew is never set against the fast one stage 3 has just fitted. The floor is the BOX beneath all
  three and never their default (a floor-by-default understates a whole book's 2-5 year vol in one
  direction); a DECLARED prior under it refuses by name, a history under it is floored and named.
  MEASURED: the FX class default takes the 22-rung USDZAR ladder's wings to **0.135** where the
  index seed cost 0.202, against the 0.132 a slow factor fitted to nothing bought - so the
  identification rule costs **0.003**, not the 0.070 the seed charged it; the reduced block reads
  0.132 the same way. WHERE THE PAIR IS PINNED THE REPORT READS THE LADDER UNDER ALL THREE PRIORS -
  the floor, the class default, the index seed - at ONE extra forward pass each with everything but
  the slow pair at θ*, the L strip re-bootstrapped and the wings re-priced, no outer search,
  printing the wing RMSE and the 5-year log-vol sd `½√(σ_s²(1−e^{−10κ_s})/2κ_s +
  σ_l²(1−e^{−10κ_l})/2κ_l)`, the number a phase-3 exposure row will read: on the ladder 0.158 / sd
  0.521, **0.135 / 0.558**, 0.422 / 0.705; on CJOW's index surface 1.448 / 0.516 under the floor
  against **1.279 / 0.701** under the class default, which there IS the seed - the ordering
  REVERSES by class (on FX the floor beats the seed, on the index the floor is the worst of the
  three), which is why it is the guard and not the default. The row in force is exact and the
  other two approximate a re-fit, overstating a distant prior's cost about twofold (the seed row
  reads 0.422 where a re-fit at the same prior banked 0.202). THE STICKINESS TARGETS ARE
  DIFFERENCES, both: `Δ_skew` and `Δ_bfly` in vol points, `Stickiness_Prior` defaulting to
  `0.0,0.0` (sticky-delta), `Quotes` / `Reference` measuring their own difference off the source's
  forward smile less the market's spot smile at the rung nearest Δ. The ratio prior's failure is NOT
  reproduced: the same sticky-delta assumption written as differences moves the guarded vanilla
  RMSE by **−0.012** where `1.0,1.0` degraded it +0.329, the wings still 0.135 → 0.433 at expiries
  spec 5.3's guard does not watch; CJOW's own forward smiles as difference targets trip 5.3's
  failure mode BY NAME at one bucket (+0.212 at the target maturities, no lever but `w_J`, no
  composition residual exists) and at a year-two bucket read a composition residual of 0.104 and a
  degradation of +0.184 - the forward-block lane's row. `ψ_skew` / `ψ_bfly` stay in the report only
  above 0.5 vol points of spot. FOUND: the sign rule is INERT today - stage 3's box keeps
  `ρ_s ≤ 0`, so no fit can produce a positive fast leverage and the FX default resolves to
  (−0.2, 0.5) on every ladder; on `FxRate.ZAR`'s own axis USDZAR fits `ρ_s = −0.276`, right-signed
  there - what the index seed got wrong was the SIZE; the rule waits for the box (an owner's
  call). `Param_Buckets` should be required when `Forward_Smile_Source` is set. Every one-value-λ
  document hex-identical. Engine net +242 lines. PROOF DOCUMENTS: `lv_hex.py`, `lv_trials.py
  limit`, `tarf_hex.py`, `lv_deals.py hex`, `artifacts/lv_split_20260906/split.py slow reduced |
  ladder | refuse | prior | cjowprior | cjowref | cjowref2 | index | indexboot | history`,
  `artifacts/logvar2fj/fit_bootstrap.json`.

- **λ(t) follows the market's own strip, and the report prints the split** (2026-09-06, spec 5.1 /
  5.2 / 5.3 / 5.4.3 / 5.4.8) - a constant intensity makes the jump variance `λ(μ_J² + σ_J²)` the
  same number on every segment while the market's forward variance moves along the strip, so the
  diffusive share was forced to track the market's shape BACKWARDS. `Lambda` is now a STRUCTURAL
  CURVE on the ATM segments, `λ(t) = w_J ξ_mkt(t)/(μ_J² + σ_J²)`, read by `bucket_at` at absolute
  times as `L` and the four levers are - never a leaf, knots and values compile-time facts, one
  knot the constant-intensity model (`LV_STRUCTURAL_CURVES`), re-derived from the strip daily with
  `w_J` rather than λ carried across days off a previous factor's first segment; it flows through
  the reciprocal axis's tilt and the compensator per step. MEASURED: the jump share is now the
  SAME number on every segment - **11.9%** on the reduced USDZAR block, **14.8%** on the 22-rung
  ladder, **15.2%** on CJOW - where the same ladder with `Lambda` pinned flat reads 18.2 / 17.4 /
  15.8 / 14.5 / 12.2 / 10.9%, the market's shape backwards, for 0.189 wing RMSE against 0.202
  (inside the seed spread). THE REPORT PRINTS THE SPLIT per segment - the market's forward
  variance, the model's total as MEAN diffusive plus jump, the jump share and the Jensen share -
  and does NOT claim the spec's "equal at the pillars by construction": the pillar matches the
  ATM PRICE, `E[Black(sd)] < Black(E[sd])`, so the model's mean total sits 4-23% above the
  market's ATM² and the Jensen share (14-58%) is what sizes it. THE SLOW FACTOR'S RULE: `(ρ_l,
  σ_l)` fitted only where the ladder carries wing quotes at 18 months or longer, else held at
  their priors with "pinned: not identified by this ladder", and `σ_l ∈ [0.3, 2.0]`. That costs
  the USDZAR wings **0.132 -> 0.202** and the reduced block 0.099 -> 0.202: the banked 0.132 was
  bought by a `Sigma_L` the old box ran to **1.5e-06**, the collapse the floor exists to stop; on
  CJOW, where 2y wings identify it, it is fitted and lands ON the 0.3 floor, named by the
  active-bound line and reported, never refused. TWO STICKINESS RATIOS: ψ_skew 0.950 / 0.984 /
  0.858 against CJOW's 0.975 / 1.008 / 0.552, ψ_bfly a division by nothing on this surface (the
  model's own spot butterfly is within 0.3 vol points of zero at 3m and 6m); `Forward_Smile_Source:
  Prior` targets both on the model's OWN spot smiles per evaluation (the forward block is now a
  vol-point residual, spec 5.3's own term, the implied vol one Newton splice at Black's vega), and
  on a ONE-bucket ladder at `1.0,1.0` it reaches 0.859 / 0.713 for +0.329 vol points of vanilla
  degradation and 0.202 -> 0.512 on the wings - spec 5.3's failure mode, reported by name.
  `Diffusive_Share` drives the jump share to exactly its declared 5.0% at 0.245 wing RMSE. An
  event day whose segment is one internal step is IDENTIFIED and the prior is ignored, which the
  report states; λ's knots are the ATM segments, not the tent's. THE GATE'S ROW 4, RE-READ off
  the CJOW surface re-fitted at `Paths` 8192 on the flat L with λ(t) (spot RMSE 0.930 / 0.319
  against 1.020 / 0.331, 29 wings 1.005 against 1.063): the 2y SPX autocall reads **-30.855 (SE
  0.143)** at `SE² x time` **2.83e-02**, against its own stale-fit -32.80 (SE 0.251) at 9.28e-02
  and the retired component family's banked -31.89 (SE 0.088) at 1.31e-02 - the efficiency 3.3x
  better, now 2.2x the record rather than 7.1x, the remainder the model's own SE on this deal and
  not the clock; the value sits 3.8% from CJOW's -32.068 where the linear-L vanilla fit read 1.1%,
  at an unchanged-to-better spot fit - the forward-skew ambiguity 5.3 names, now with a bigger
  number on it, and whether the forward block or a second calendar bucket pulls it back is the next
  lane's call. CLOSED 2026-09-07 by the slow-prior lane above: the prior is a FIELD with a
  per-asset-class default (FX (0.2, 0.5) signed by the fitted `Rho_S`), not the floor and not the
  seed - the floor stays the guard beneath it, and the ladder's wings read 0.135 rather than
  0.202; ψ_bfly's denominator problem is answered by targeting the DIFFERENCE and printing the
  ratio only above 0.5 vol points of spot. Engine net +238 lines. PROOF DOCUMENTS: `lv_hex.py`,
  `lv_trials.py limit`, `lv_deals.py hex`, `tarf_hex.py` (all four hex-identical),
  `artifacts/lv_split_20260906/split.py reduced | ladder | pinlam | prior | share | events | cjow
  | tables | autocall`.

- **The retirement** (2026-09-06, licensed outright by the owner: "we didn't actually validate HN or
  TARFs or accumulators, so we can remove HN any time we like in favour of LogVar2FJ") — both
  Heston-Nandi families are GONE from the engine, in four commits off the desk pin above, **net
  −9,103 lines** with 786 added — −4,262 in the engine, −4,841 in the gates, tests and docs. What went: the two calibrators and `ComponentStrips`; the two price
  factors; the two OSS kits and the whole STRIDE (`HN_Stride`, its three consumers, the phi_max
  scan, the Esscher tilts, the carried-state loadings); the two implied spot processes; the
  Heston-Nandi half of `utils` and the model-agnostic Fourier inversion it was the last reader of;
  seventeen test files' worth of arms and six whole ones. What stayed: GBM as the limit and the
  reference, Hull-White, the curve families, `FXVolPrices`, `LeastSquaresSolve`, `InnerMCRecompute`,
  the boundary sets.

  **The quote preparation was HOISTED before anything was deleted.** `HestonNandiModelParameters`
  carried the ladder every option family reads — `resolve_block`, `prepare_quotes`, `quote_trailer`,
  `fx_surface_block`, the reference fields, the moneyness dispatch — and `LogVar2FJModelParameters`
  inherited it. That half is now `bootstrappers.OptionQuoteFamily`, family-neutral, with the model's
  own half (the GARCH box, `reparam`, the Fourier price, the objective) deleted from under it;
  LogVar2FJ's MRO is `LogVar2FJModelParameters -> OptionQuoteFamily -> object` and carries nothing
  Heston-Nandi. The hoist is worth **zero to the bit**: the banked USDZAR FX ladder re-fitted on it
  in both `Global` and `Bootstrap` modes reads every written parameter and every L knot hex-identical
  to the pre-hoist fit, the 22 emitted rungs identical, the wing RMSE 0.131682 and 0.573582 the same
  digits. The four-quote HW2F pin document is bit-identical too — θ\* and all four `dV/dq`
  (+4.601481694837237, −14.607195319955192, −8.728049049402394, +39.370686531805916).

  **`OSS_SPOT_MODEL_KITS` is a registry of one, and every branch that asked which kind of kit it
  held collapsed.** `kit.daily` and `kit.strides` had one value left, so the four OSS pricers' daily
  arms went — the barrier's HN closed-form parity leg and its batched-carry refusal, the TARF's
  sub-step loop and its stride, the accumulator's, the autocall's — and `pricing.branch_and_weight`
  folded into the flag it read: the one surviving family hands each fixing interval its own Gaussian
  block law, so the smooth estimator is admitted on the same terms as GBM and the daily-kit refusal
  it existed for has nothing left to refuse. The `factor_dep` keys stopped lying: `HN_Params`,
  `HN_Steps_Per_Year` and `HN_Invert` are `Spot_Model`, `Steps_Per_Year` and `Invert_Spot`.

  **Every LogVar2FJ and GBM document is hex-identical across all four stages.** `lv_hex.py`
  −0x1.aad8d5d75f8c9p+5; `lv_trials.py limit` at −0x1.a1f306d03a5a5p+5 (GBM) and
  −0x1.a1f306d03a59fp+5 (the LogVar2FJ limit) with CVA 0x1.5057040000000p-4 / 0x1.5057060000000p-4;
  `tarf_hex.py`'s crisp GBM TARF −0x1.2c48f36318e38p+5 at delta 1814.77 with the same one
  `LatchedBoundarySet`; `lv_deals.py hex`'s eight surviving GBM keys unmoved to the digit. The TARF
  credit MC under LogVar2FJ reads CVA 0.07946623117 with a profile identical to main's, the
  reciprocal-axis accumulator credit MC 0.2118722349 over seven finite dispersed rows. A document
  that DECLARES a retired family refuses by name at load: *FXTARFOptionDeal does not honour
  SpotModel='HestonNandi'; it accepts ('None', 'LogVar2FJ')*.

  **Six `lv_deals.py hex` keys went by construction**, and one of them was mislabelled: `gbm barrier
  credit mc` walked its equity under `HestonNandiImpliedSpotModel` — only its DEAL was GBM — so it
  died with the scenario process like the five openly Heston-Nandi ones. A `Model Configuration`
  naming a retired process does not refuse by name: the factor drops out of the simulated set with a
  per-deal WARNING and a credit MC then dies on a zero-factor reshape. That is the pre-existing
  behaviour of ANY unknown process name, verified identical on main with a nonsense one, and it is
  an open row rather than a regression.

- **The desk pin moves to LogVar2FJ** (2026-09-06, licensed by the FX gate's addendum) -
  `structures.SPOT_MODEL` is `LogVar2FJ` and `/book/hn` becomes the family-neutral `/book/model`
  (`{pair, family}`, the family defaulting to the pin so a desk that calibrates and a runner that
  pins cannot name two different models); `HestonNandiJob` is `SpotModelJob`, the MCP tool
  `calibrate_spot_model` with a `family` argument (no alias tool: a tool list is re-read on every
  connect and has no deployed callers), and `/book/hn` stays one release as an alias meaning the
  family it is named for. `HN_FAMILY` and `hn_factor` fold into `structures.SPOT_MODEL_FACTOR`, the
  one key the pin, the verb and the engine's lookup now share; the runner's `spot_model` reads the
  BOOK's own `Valuation Configuration` switch where it declares one, so the note that said "priced
  GBM - looked up `HestonNandiModelParameters.ZAR`" while pricing an authored LogVar2FJ pin cannot
  be told again, and on a book carrying no factor it names the pinned family and the verb that
  installs it. One `/book/model` call lands `LogVar2FJModelParameters.ZAR` off the banked USDZAR
  surface in 206.5 s, reproducing the banked ladder fit digit for digit (0.084 vega-weighted vol
  points, ATM misses at 1e-14). On the banked USDZAR book under the runner's own pin, re-taken on
  main with the reciprocal axis landed: the ZAR accumulator 15.63403002 against 15.68600904 GBM
  and 15.63101970 plain HN (-0.331%); the USD-notional accumulator 15.64067534; the TARF - the
  product the desk quotes, forced onto the base by `furnish_accrual` - **15.31369426** against
  15.32196559 GBM and 15.32078223 plain HN, where before the reciprocal lane it refused by name;
  a straddle on the same book hex-unmoved (`40b5b5b769f3949e`); `GET /schema` still lists all
  eight families and the alias verb answers. The run's Stats key is the constant `SpotModel`. The
  Heston-Nandi families are NOT retired: their retirement waits on the axis gap, the credit MC's
  per-document clock on the GPU and the autocall's CJOW re-fit (the FX gate below), and until
  then a book may author `SpotModel: 'HestonNandi'` on a deal type and the runner honours it.
  Net +18 lines (-6 in the engine). PROOF DOCUMENTS: `artifacts/hnpin/verb.py` (the one
  `/book/model` call), `artifacts/hnpin/pin.py` (the eleven quotes above, re-taken on main).

- **LogVar2FJ on the reciprocal axis** (2026-09-06, spec §2.8, `HN_Invert`'s analogue) - the carry
  is a measure change inside the walk, not a parameter map. Under the `S`-numeraire the step's
  density `exp(R_k - b_k delta_k)` is one in expectation and factorises over its own draws, so the
  two shocks take their mean shifts INSIDE the step (`eta_l ~ N(rho_l sqrt(V), 1)`,
  `eta_s ~ N(rho_s sqrt(V), 1)` - the leverage feeding back into the variance path as `1 - gamma*`
  does for Heston-Nandi), the counts arrive at `lambda delta exp(mu_J + sigma_J^2/2)` (the kit's
  `draws` converts the uniforms at the tilted intensity, the inverse-CDF spelling unchanged), and
  the block law handed back is `-(M + Sigma^2)` at the deal's own carry - `utils.lv_walk` takes
  `invert` as a REQUIRED positional, since a wrong value there is a quiet bias and not a shape
  error. `spot_model_reciprocal_axis` is an ALLOW-LIST the family joins (`'HestonNandi',
  'LogVar2FJ'`; the component family still refuses by name), and the plain family's own carry
  moved out of `reciprocal_spot_scalars` (deleted) into `PlainHestonNandiKit.__init__` so
  `oss_model_kit` has one spelling of invert - its three reciprocal documents hex-identical
  across the move. MEASURED: a strip of forwards on `1/S` reads its own market forward to
  **-2.3e-05** against the direct axis' +2.4e-05 at 2^18 paths; the GBM limit on the reciprocal
  lands at **4 ulp** (-7.8e-16); one accumulator solved from both orientations off the banked
  USDZAR fit reads a mean gap of -1.0e-4 / +2.2e-4 / -1.2e-4 / **-4.2e-05** at 4k / 16k / 65k /
  262k paths, sign-changing and inside the same-side seed spread at every rung (6.1e-4 down to
  1.3e-4). UNCARRIED it is a bias, measured on a mutant that walks the fitted law and reads `1/s`
  off it: the forward strip +4.1e-3 (the Siegel drift), the two orientations **+3.7e-3** at every
  count beside plain Heston-Nandi's own 3.4e-3, and the reciprocal GBM limit **16% wrong**, which
  pins the `+Sigma^2` factor on its own. Every non-inverted document is hex-identical (`lv_hex`
  `-0x1.aad8d5d75f8c9p+5`, `lv_deals` 14 keys, `tarf_hex`, `hn_hex`), the calibrator's fifteen
  fitted leaves hex-identical on a 2,048-path fit (it walks the `FxRate`'s own axis), the TARF's
  second side still refuses by name, and one inverted document prices under the credit MC at
  256 x 2,048 with the gradient on (CVA 0.2167, seven finite rows). OPEN: on the reciprocal axis
  (CLOSED 2026-09-07 by the NIG residual, which has no count law: the tilted mixer is on the tape)
  `dV/dMu_J` and `dV/dSigma_J` lose the term through the tilted count law (integer counts carry
  no graph) - a gap the direct axis does not have, unmeasured (a CRN ladder on `Mu_J` of an
  inverted document sizes it); the banked FX fits carry no slow factor (`Sigma_L = Rho_L = 0`),
  so the `rho_l` shift is exercised only on authored parameters. Net +17 lines. PROOF DOCUMENTS:
  `lv_hex.py`, `lv_deals.py hex`, `tarf_hex.py`, `lv_phase1/hn_hex.py`,
  `artifacts/lv_reciprocal_20260906/lvinv.py hnrecip | limit | forward | orient | refuse | cmc`,
  `logvar2fj/fit_doc.py fit_lvl2048.json` on both trees.

- **The FX gate: LogVar2FJ beside plain Heston-Nandi on the desk's own products** (2026-09-06,
  `artifacts/fx_gate/`, CPU, both engines stamped in `engine_stamp.json`). On the banked USDZAR
  world the model wins every row it is allowed to price and is blocked out of the one the desk
  actually quotes. THE LADDER (22 contracts, `Paths` 8192, daily): `Global` fits the 16 wings at
  **0.132** vol points against the plain family's 0.663 and the component family's worst wing
  0.760 (CAPPED), `Bootstrap` 0.574, in 205 / 169 s against 513 / 93 s; the flat-L diffusive strip
  reads 9.08 / 8.54 / 8.68 / 8.89 / 9.54 / 9.97% against the market's 9.85 / 10.08 / 10.58 / 11.04 /
  12.04 / 12.75 - a share of 0.92 down to 0.78, monotone, where the linear strip alternated. THE
  ACCUMULATOR solves to -0.309% of GBM against plain HN's -0.351% and the component's -0.308%, the
  two orientations 4.0e-6 (GBM) and 7.6e-6 (HN) apart; **the USD-notional side and the TARF
  ENTIRELY refuse by name**, `furnish_accrual` forcing the base-currency notional onto the
  reciprocal of the fitted axis, which is spec 2.8 and the reason `structures.SPOT_MODEL` cannot
  move until the reciprocal lane lands. FIRST ORDER on the accumulator: spot **0.084%** against
  HN's 0.067% and the component's 0.010% on ladders flat to 0.2-0.3%, `Mu_J` 0.024%; no spot model
  reports a vol-surface row (the model's parameters replace the surface; the quote-space row is
  the vega). SECOND ORDER is LogVar2FJ's alone: gamma **0.32%** off the AAD delta's ladder under
  `Branch_And_Weight`, where both Heston-Nandi families refuse the switch by name and the crisp
  arm reads 73-88% off. CREDIT MC at 256 x 2,048 agrees across families (CVA 97.5 / 103.0 / 103.2
  uncollateralised, 49.9 / 49.4 / 49.5 under the CSA) and no family's CVA spot delta is resolved
  there - 10.6-12.6% against ladders 14.1-14.8% non-flat. QUOTE SPACE is the widest gap:
  `Quote_Sensitivity` on the FX block publishes 22 `dV/dq` over a 0-dimensional null space at
  `||J^T r||` 1.1e-8 with `Nu` and `Sigma_J[0y]` held by the KKT box; the 16 wing rows match the
  node's contraction to the digit and the 6 ATM rows carry a SECOND route beside it - an ATM quote
  moves the L strip directly through the per-iterate inner bootstrap's Newton splice, so the
  total `dV/dq` (1y ATM -160,095) is the node's route (-56,455) plus `dV/dL . dL/dq` at fixed
  theta, which is the whole derivative and not a disagreement; the plain family publishes **no
  quote leaf**. THE 2y SPX AUTOCALL reads spot **0.502%** and rates 0.493% against the component
  family's 2.26% and 3.43%, with gamma at 0.017% where the component refuses second order - but
  its value is -32.80 (SE 0.251) against -31.89 (SE 0.088) at a `SE^2 x time` of 9.28e-02 against
  1.31e-02, because the CJOW factors it prices off were fitted on the LINEAR L and now walk
  alternating flat segments; a re-fit is what that row waits on. WHAT FAILS, named: the TARF under
  LogVar2FJ (all of it, on §2.8 - the reciprocal lane), the accumulator's base-currency
  orientation (same), the per-document clock on CPU (3.7x a quote, 7.3x a credit MC, 4.3-5.2x on
  the autocall, against a 2x rule; the GPU reading is the checkpoint lane's 70 s at 2,048 x
  2,048 and the rule is re-read there), and the autocall's efficiency until the CJOW re-fit. WHAT
  PASSES: everything else. THE ADDENDUM, on the reciprocal-axis engine (678af12), every FX row
  re-taken on one engine: the TARF under LogVar2FJ PRICES - zero-cost strike 15.30473115 against
  GBM's 15.31624884 and plain HN's 15.31754457 (-0.075% against +0.0085%: the model moves the strike
  the desk quotes nine times further than plain HN, the other way); in the GBM LIMIT (every lever
  zero, `L` flat, the surface flat at the same vol) the reciprocal-axis TARF and both accumulator
  orientations are **bit-identical** to GBM (relative 0.0), so the measure change costs nothing
  where nothing should be lost; the two accumulator orientations solve **2.01e-4** apart against
  the plain family's 1.44e-5 and GBM's 4.05e-5 on the same engine and seeds - the shocks'
  estimator error at 16,384 paths, on the runner's 2e-4 lognormal band and 2x outside the 1e-4
  the plain family is held to, the one number the carry still owes a path count or a variate;
  the TARF's ladders read no worse (spot 0.710% against 1.124% on ladders 2.3% and 2.7% non-flat;
  `Mu_J` 7.2% on a ladder scattering 5.0%, a tail lever on a crisp latch), its gamma lands 3.26%
  off the AAD delta's ladder under `Branch_And_Weight` where plain HN refuses the switch by name,
  and its quote-space risk is 22 identified `dV/dq` concentrated on the front ATM pillars
  (0.167y ATM -550,142) over a 0-dimensional null space with both active bounds named; its credit
  MC agrees with the other families to 0.9% uncollateralised and 1.4% under the CSA. THE CLOCK:
  tick-to-price 211 s against 519 s (2.5x faster) and the quote itself equal (5.9 s against 5.8 s);
  the credit MC per document 78 / 232 s against 15 / 37 s on CPU (5.1x / 6.4x), the one clause
  still failing. THE VERDICT, revised: the desk pin MOVES (the quote path passes every clause);
  the retirement of the families does not - the axis gap, the per-document credit-MC clock (a GPU
  reading owed) and the autocall's stale-fit row are the named blockers. Found beside it, not
  blocking: the structures runner's leg note says "priced GBM - looked up
  `HestonNandiModelParameters.ZAR`" on a book carrying only another family's factor while the
  deal priced under the authored pin (the family is hard-coded in three places); and
  `torch.compile`'s CPU backend without MSVC on PATH raises inside the fused Heston-Nandi substep
  and the deal is SKIPPED rather than refused (surfacing downstream as a frame-shape error).

- **LogVar2FJ per-block checkpointing, and the day as the model** (2026-09-06, spec §6 and §12) -
  `utils.lv_walk` walks ONE block and returns `(M, var, l, s)`; `LogVar2FJKit.blocks` loops the
  fixing blocks as checkpointed segments of `LV_CHECKPOINT_STEPS` (21) internal steps
  (`use_reentrant=False`, `preserve_rng_state=False`), each regenerating its own draws from a
  generator keyed on the row's `base` - one int64 off the plain stream `utils.rng_position`
  replays - plus the segment index, so nothing of the whole grid's shape is ever held. The segment
  length is measured, not chosen: at 2,048 x 2,048 the semi-annual block OOMs, 42 steps read
  11,184 MiB / 74 s, **21 read 8,392 MiB / 71 s**, 10 read 8,112 / 110, 5 read 10,896 / 138 - the
  forward pays boundary states in 1/seg, the backward tape in seg. On the 2y SPX autocall at
  256 x 2,048 daily with the CVA gradient on, peak falls **20,287 -> 1,083 MiB for 1.47x the wall
  clock**, and the contracted target prices: 2,048 x 2,048 daily, uncollateralised,
  `Recompute_Inner_MC: 'Yes'`, **8,392 MiB in 70.5 s**, where the un-checkpointed walk asks for
  7.95 GiB it cannot have. THE CVA DELTA ROW, read on the step the model offers: the
  uncollateralised spot delta is **+4.64e-05** against a ladder +3.96 / +4.22 / +4.49e-05 flat to
  12.5%, between the two finest Richardson extrapolations (+4.58e-05 in h^2, +4.75e-05 in h) -
  superseding the 21-day +5.524e-05 against +5.55..+5.82e-05, a measurement under a step the
  model no longer offers. The collateralised row is read for the first time, at 256 x 2,048
  (-1.223e-04 against a ladder scattering 16.0%) and priced to 1,024 x 2,048 (11.8 GiB); at 2,048
  it still OOMs and THE WALK IS NOT WHAT BINDS IT - the CSA adds 1,909 MiB at 256 outer and 15.0
  GiB at 2,048 against the walk's own 1,083 -> 8,392, so the next memory lane is the collateral
  recursion. Second order is VERIFIED, not assumed: the non-reentrant checkpoint carries a double
  backward, `Greeks: 'All'` reads gamma **0.016%** off the AAD delta's own ladder, with spot
  0.32%, rates 0.06% and the L curve 0.92% at 65,536 paths; CVA gamma through `InnerMCRecompute`
  stays out of scope (the node is first order by contract). `Internal_Step_Days` is GONE - from
  the four deals' valuation options and documentation, from `set_spot_model_index` (two keys
  remain), from the kit (`steps = max(round(dt * spy), 1)`), from the calibrator's fields and
  grid, and from the campaign's authoring; `checks.py` G9 keeps its 1/2/5/21-day table on its own
  test-only grids, reported and no longer gated (the day is the model; the table documents what a
  coarser chain would change - 12.4 SE at 21 days, 3.4 SE at 5). Every GBM and Heston-Nandi
  document is hex-identical (`tarf_hex.py`, `hn_hex.py`, `run_trials.py base`, the 14 keys of
  `lv_deals.py hex`); every LogVar2FJ document RE-MARKS within its own MC error (rev3 -52.397 ->
  -53.356, 0.65 SE; Q-sized -59.884 -> -60.746, 0.53 SE), the draw stream being per-segment now,
  with the GBM limit still at rounding on all five documents (8.2e-16 on the autocall) and the
  9-month deal blind to a year-two bucket to the bit; the limit's CVA reads one float32 ulp off
  GBM's (9.1e-8) where phase 1 read it bit-identical, because the row's key consumes one draw of
  the plain stream the GBM document does not. The calibrator's fitted leaves move 7e-12..3.2e-08:
  the walk is bitwise identical on identical inputs on CPU and CUDA and the fit is run-to-run
  reproducible, so this is the Jacobian reassociating over a curve leaf's per-block reductions in
  the flat `(Sigma_L, Rho_L)` valley, not a changed model. Net **-6** tracked lines. PROOF
  DOCUMENTS: `tarf_hex.py`, `lv_phase1/hn_hex.py`, `lv_deals.py hex | limit | values | guards`,
  `lv_hex.py`, `lv_trials.py limit | base | buckets | greeks | refuse`, `lv_remark.py`,
  `lv_cva_daily.py`, `lv_recompute_parity.py`, `checks.py g9 g10 buckets`, `fit_doc.py`.

- **The LogVar2FJ L curve is piecewise constant on the segments between ATM expiries** (2026-09-06,
  spec 5.4.3, the owner's ruling). Matching segment integrals with a curve whose integral reads
  both ends is a recurrence with multiplier -1, so the fitted strip alternated with a phase nothing
  pinned: on CJOW it read 14.56 / 5.51 / 19.95 / 10.76 / 13.37 vol against the market's
  forward-variance strip 15.97 / 12.73 / 15.71 / 19.77 / 17.52. Flat on each segment the integral
  depends on that segment's own level alone, so the multiplier is zero, every ATM still reprices
  exactly (one level per segment, one ATM increment per segment, triangular and unique) and
  anything priced between pillars reads a quoted forward variance; the OU recursion is a deviation
  from L, so the step at a pillar costs nothing. The same CJOW surface at `Paths` 2048 now reads
  14.43 / 9.34 / 11.91 / 14.37 / 12.44 - a diffusive share of the market's strip of 0.90 / 0.73 /
  0.76 / 0.73 / 0.71 where the linear one read 0.91 / 0.43 / **1.27** / 0.54 / 0.76 and crossed one
  twice - the strip's second difference falling **20.4 -> 5.1** vol points RMS at a vanilla RMSE of
  0.931 against 1.020 at four times the paths; a reduced USDZAR block reads 9.11 / 8.75 / 8.82
  against 9.11 / 8.38 / 9.23, 0.42 against 1.57, at an unchanged wing RMSE (0.0988 / 0.0984).
  `L_Curve`'s knots are the segment starts (tenor 0, then every ATM expiry but the last) and its
  values the levels; `L(0)` is the first segment's level and the phase that tied it to the first
  pillar is gone with the pillar; `Event_Days` is a one-day segment at `Event_Variance_Prior`
  times the enclosing level, measured at exactly 3.000000x and restored to the bit; the 2.6 seed
  maps each segment off its own forward-variance increment and is EXACT for the unconditional
  mapping, since the OU deviation from L does not depend on L. The kit, the calibrator and the
  pack read L with `bucket_at`, the levers' spelling; `curve_at` keeps its one remaining reader,
  the component Heston-Nandi L, which must stay linear because omega differences it. Every
  one-value-L document is hex-identical (`-0x1.a32dcde94c00ap+5`), the GBM limit unmoved
  (1.90e-15, CVA hex-identical), and a document carrying a fitted multi-knot curve RE-MARKS: the
  campaign's V2 autocall off B2's reduced-USDZAR fit moves **1.53%** in MTM (-14.7517 -> -14.5260)
  and 0.41% in spot delta. What is left is the LEVEL: the diffusive strip sits at 0.71-0.90 of the
  market's total forward variance and does not track its shape - the jump plus the two leverage
  shocks, a decomposition the vanillas do not pin, not a phase. The two `Paths` 8192 tables in
  `market_prices.md` are readings of the linear L until re-taken. Engine net +15 lines. PROOF
  DOCUMENTS: `lv_hex.py`, `lv_trials.py limit`, `lv_deals.py hex` (14 keys unmoved),
  `artifacts/logvar2fj/flat_l.py fit lane | fit b2 | cjow lvl2048 | remark b2 | remark b2:lane |
  fit events`.

- **LogVar2FJ on the autocall's averaging arm** (2026-09-06, spec 2.4.1) - a coupon whose decision
  reads the arithmetic average of a window of fixings is priced by SAMPLING THE WINDOW AND
  TRUNCATING THE PREFIX: the window's fixing-to-fixing blocks are ordinary blocks of the daily walk
  (the pricer already hands `kit.blocks` the whole fixing strip, so the block plan and the
  `blocks()` signature are untouched), their returns are drawn as plain Gaussians off the
  dimensions the row's Sobol block already carried, `G = (1/n) sum exp(Delta_i)` with the observed
  fixings' share as a constant, and the survival event is a half-line in the prefix return - one
  `Phi`, one `Phi^-1`, the put leg one `lognormal_fired_gain` at `spot -> S G`,
  `strike -> strike(1 - rebate) - c`. A window of one collapses to `c = 0`, `G = 1` bitwise, which
  is why the deal as authored reads `-0x1.a32dcde94c00ap+5` to the bit and the GBM five-fixing
  document is hex-identical across the change (`-0x1.bb53368e105d2p+5`), as are the CMC's CVA, its
  spot gradient and the ledger on the as-authored deal. The termination is a crisp per-scenario
  decision again, so the averaging coupon carries the latch (`Greeks: 'First'` on and off
  hex-identical) and `Branch_And_Weight: 'Yes'` is ADMITTED for a walking kit; GBM and the daily
  kits keep the full-path branch and refuse by name; `no_averaging` is `oss_windows`, since the
  flag now means "prices on the OSS arm", averaging included. Spec 2.4.1's whole-interval root
  (treatment i) is not built because a zero-length prefix is unreachable here: a fixing on the
  previous coupon's date belongs to that coupon and enters as a constant, so the conditioning step
  is always the first unobserved interval - measured at the limit (26 weekly fixings, prefix
  3.8-7.4% of the interval) at 0.04 combined SE against a brute-force oracle. MEASURED at 65,536
  paths, five seeds: the five-fixing document -37.62 +/- 0.10 against the oracle's -37.06 +/- 0.72
  (0.76 combined SE); on the calibrated market the averaging is worth +0.58 (0.95%) and cuts the
  estimator's SE 3.7x; the smooth value 0.67 SE from the crisp; spot delta 0.035% off its CRN
  ladder (flatness 0.13%) and gamma 0.0093% off the AAD delta's ladder under `Greeks: 'All'`;
  under the credit MC at 256 x 2,048 the profile is dispersed and the ledger settles the four
  coupons on their settlement dates, CVA 0.0732. Net +84 tracked lines. PROOF DOCUMENTS:
  `artifacts/lv_averaging_20260906/lvav.py hex | arms | first | values | oracle | greeks | full |
  cmc | cmcone` (twelve documents, the three hex rows re-taken on main at landing).

- **`Barrier_Observation`: the autocall's put barrier reads the barrier date's spot or the coupon
  window's average** (2026-09-06, owner's ruling) - two conventions, both real products, neither
  derivable from the other, and the DEAL now declares which: one field on
  `QEDI_CustomAutoCallSwap` (`_V2` by inheritance), `'Spot'` | `'Average'`, DEFAULT `'Spot'` - a
  barrier is a level on the spot and that is the plain reading of a term sheet. ONE registry,
  `pricing.BARRIER_OBSERVATION`, hands each caller the two readings of its own quantity and the
  field picks one, so neither arm gains a branch. On the OSS arm under a walking kit both readings
  are half-lines in the SAME prefix return - the average's `(K - c)/S G` and the spot's `B/S W`
  with `W` the window's own cumulative return to its LAST FIXING, the nearest boundary the walk
  has - so the breach is their INTERSECTION and the put leg stays one `lognormal_fired_gain` at a
  shifted forward, no second draw and no new closed form; the stride carries its bound in the
  strip's own measure the same way, and an interval is now built only where a barrier date reads
  it. The full-path branch reads the barrier AFTER the fixing accumulates and compares the
  window's running SUM against `putBarrier * count`, so `Spot` takes no divide at all. A window of
  ONE is the same bit under both values and unmoved: `campaign/lv_hex.py`
  **-0x1.aad8d5d75f8c9p+5** under the default, under `'Spot'` and under `'Average'`; the repo's
  own `autocall_job.json` `0x1.87f8285dd41fcp-2`; the component family and its stride
  `-0x1.ffdf759c774dap+4` / `-0x1.f9ab6bb9189f5p+4`, both re-taken on main; the GBM five-fixing
  full-path document `-0x1.bb53368e105d2p+5`. On the calibrated market the five-fixing LogVar2FJ
  document under `'Average'` reproduces main's five seeds BIT FOR BIT, crisp and smooth, and under
  `'Spot'` the lane's own default. MEASURED at 65,536 paths, five seeds, CRN-paired: at a
  five-business-day window and a 70% barrier the field is worth nothing measurable (-0.005 at 0.1
  paired SE crisp, -0.008 at 1.4 smooth, +0.045 at 1.0 on the full-path arm); at a WHOLE-INTERVAL
  window (26 weekly fixings) it is worth **+0.37** (5.5 SE, 1.2%) on the zero-rate twin, where the
  brute force re-told to read the same thing agrees at 0.02 and -0.15 combined SE (0.36 / 0.42 on
  the five-day), and **+1.87** (29.9 SE, 3.8%) at the GBM limit - where the FULL-PATH arm makes
  the same move, +1.84, the two **0.029 apart** on their own 0.063 / 0.094, which is what gates
  that branch's own reading. `Greeks: 'First'` on and off hex-identical under `Spot`
  (`-0x1.e1d116592091fp+5`), spot delta 0.03% off its CRN ladder (flatness 0.12%) and gamma
  0.008%; the credit MC at 256 x 2,048 reads CVA 0.0731208 with a dispersed profile and the four
  coupons on their settlement dates. OPEN: on the OSS arm a SECOND consecutive coupon whose window
  is wholly observed reads the FIRST one's last fixing under `'Spot'`, the block entering with
  that `Sj` - the same staleness its prefix already carries, and no document here reaches it.
  REFUSED BY NAME rather than left silent: an `'Average'` barrier date whose coupon window holds
  no fixing would compare a mean of nothing, so `calc_dependencies` raises `UnpriceableSchedule`
  naming the date and both remedies, on either arm - a barrier dated on the first float payment,
  a quarter before the first observation, refuses, and the same document under `'Spot'` prices at
  `-0x1.5116d53b3209ap+4`. Net +54 tracked lines. PROOF DOCUMENTS: `campaign/lv_hex.py`,
  `artifacts/lv_averaging_20260906/lvav.py obs_hex | obs_stride | arms | values | obs_values |
  obs_oracle | obs_full | obs_limit | obs_refuse | first | greeks | cmc`,
  `tests/fixtures/autocall_job.json`,
  `tests/fixtures/policy_test_simulate_only.json`,
  `campaign/documents/9500/hex_tarf_True.json` (the last three the `gates/reach.py --dirty` cover,
  re-taken on main and on the lane).

- **LogVar2FJ phase 2 - spec 5.4's engine bootstrapper: the L strip re-bootstrapped at every
  iterate, the implicit function theorem as an expression, and risk in quote space through ONE
  node** (2026-09-06) - the inner triangular bootstrap runs at EVERY outer iterate, so every
  candidate reprices the ATM term structure exactly (misses 1e-16 to 1e-13) and is judged on the
  smile alone; the level each pillar returns is one Newton step at its own root,
  `L_k* - F_k/detach(dF_k/dL_k)`, so `dL/dtheta` rides the tape and the outer solver keeps its
  exact vmapped Jacobian; the search is a chord off the previous sweep's slope, off the tape,
  three rounds to `Pillar_Tolerance` (0.280 s a forward pass against 0.756 s with its backward at
  8,192 paths over 504 daily steps; 18.7 pillar passes a sweep, 5.3 carrying a backward).
  `Quote_Sensitivity` is SUPPORTED and its contraction is `LeastSquaresSolve` - the swaption
  family's node, which absorbed `LVQuoteSolve` rather than acquiring a sibling: one
  `autograd.Function` over `(calibration, rcond, stationarity, *quotes)`, the vmapped Jacobian in
  place of the per-row loop (bit-identical on both HW2F objectives, 0.22 s against 1.21 s on the
  identified block), the contraction on the COLUMN-SCALED Jacobian at `Jacobian_Rcond` over the
  coordinates the KKT active set leaves free (`active_set`: within 1e-4 of the box width of a
  bound AND the gradient pointing into it - scipy's `active_mask` is a step-length report and read
  EMPTY with `Nu` at a denormal above its floor; on the reduced USDZAR fit the KKT set holds `Nu`
  and `Sigma_J[0y]` and the contraction then reads `Sigma_S[0y]` 14.10 against a re-solve flat at
  13.82 / 13.82 / 13.82 over h = 2e-4 / 1e-4 / 5e-5, where contracting the whole vector read 216
  and `Nu` -3791 against 0). That fold fixed two defects the swaption path had carried - a relative
  cutoff on an unscaled `J^T J` whose column norms span orders, and a contraction over coordinates
  the box holds - and re-set HW2F's `Jacobian_Rcond` from 1e-8 to **1e-5**: on sigma of the scaled
  matrix 1e-8 keeps a direction 1.4e-6 of the largest and amplifies `dtheta/dq` 7e5-fold; the
  identified block's one spectral gap is 2.97e-4 against 1.43e-6 and 1e-5 sits in it, keeping 16
  of 23 where the old sigma^2 cutoff kept 15, with `Correlation` held on its -0.95 floor. HW2F's
  fitted parameters and identified directions are unchanged to the digit; in the null space the
  minimum-norm representative is now the one in the metric the solver steps in, so on the
  four-quote block the fourth benchmark's `dV/dq` reads 0.1356 where the unscaled spelling read
  0.2704, and every quote delta is logged beside its NULL-SPACE SHARE (0.99-1.00 there, 0.70
  typical on the identified block, 0.90 on a LogVar2FJ autocall fit) - the part that is convention
  (`tests/fixtures/hw2f_four_quote_job.json` pins the four numbers with that sentence). Every
  binding bound is named in both families' reports whether or not quote sensitivities were asked
  for. `Fit_Mode` Global | Bootstrap makes the ladder's wing expiries the calendar buckets, which
  makes `Sigma_S` and `Sigma_J` bucketed curves beside `Rho_S` and `Mu_J`: `LV_BUCKET_NAMES` is
  four, the factor's curves five, one knot at 0 reads every existing document to the bit.
  `Internal_Step_Days` is NOT written onto the factor - the field, its structural entry and the
  kit's load refusal came off; the deal's own declaration is what the kit walks until the
  checkpoint lane deletes the setting (spec 2.1: the day is the model) - and the trading-day-grid
  cost is measured again at **0.124 vol points** at the 1m ATM. `Stationarity_Tol` is declared and
  the node raises on it, and a stage that stopped CAPPED at `Max_Iterations` refuses
  `Quote_Sensitivity` by name (a converged polish reads `||J^T r||` 2.19e-6 against a capped one's
  6.13e-6, so the norm alone could not have said it). On the banked USDZAR ladder six of the 22
  rungs are the ATM rung of an expiry any curve family solves to zero, so the statistic is the
  per-contract table and its WING RMSE, one spelling shared with the CJOW harness: LogVar2FJ
  `Global` reads **0.131** over the 16 wings against the plain family's **0.663** and the
  component family's worst wing **0.760** (CAPPED at 300 evaluations), `Bootstrap` 0.575 -
  restricting the smoothness rows to the buckets a stage has fitted took it from 0.764/0.704 to
  0.490/0.455 unweighted/vega-weighted. On CJOW the composition residual is 0.119 and the
  vanilla-only autocall 1.1% of CJOW, the forward target moving it to 3.2% for a forward-skew
  sensitivity of -0.647; the short end is unmoved at 2.3 vol points and 1,142 s against spec 9's
  90 s remain the open numbers; the fitted L strip zig-zags (the flat-L lane above). The
  triangle's first two legs are a chain-rule TAUTOLOGY (both route through the same backward;
  their 4.4e-16 to 5.8e-14 tests the attachment); the third leg, the re-authored central
  difference, is the only one that tests the theorem and does NOT close on the autocall document -
  the re-fits either side of a tick land at different active sets (`Rho_L` on its -0.6 box one
  side), so the quotient is a jump, the Hull-White page's finding again; the quote delta is the
  one-sided derivative on the current active set and the direction check is its gate. Deviations
  from the spec, all of them: `Wing_Weight` 1.0 against 5.4.4's 2.0; `Pillar_Tolerance` 1e-10
  against 1e-14; `Max_Iterations` 150 against 60; `Jump_Share` 0.25 against 0.15;
  `Bucket_Smoothness` 0.02 against 1.0; `Forward_Smile_Source` Prior tilts the MARKET's spot slope
  where 5.3 says the model's own, because a target that moves with the iterate is not a target;
  `Event_Days` is a multiplier on the interpolated level, identified by nothing (two straddling
  pillars are spent on their own knots). Net +1,329 tracked lines. PROOF DOCUMENTS: `lv_hex.py`
  (`-0x1.a32dcde94c00ap+5` to the bit), `lv_trials.py limit` (GBM 1.9e-15, CVA hex-identical),
  `lv_deals.py hex` (14 keys, 0 mismatches against every banked file), the reduced USDZAR
  quote-risk document (the KKT table above), the four-quote HW2F document on both engines
  (theta* identical, the moves table), the identified 25-quote document once (733 s).

- **The LogVar2FJ calibrator** (2026-09-05) - `bootstrappers.LogVar2FJModelParameters`, block
  `LogVar2FJModelPrices`, fits the seven scalars, the L curve and the two bucketed levers to
  European premiums and, where a desk has them, to spec 5.3's forward-start smiles. It subclasses
  the plain family, whose quote preparation is now ONE `prepare_quotes` and one `resolve_block`
  where three verbatim copies stood, so the component family's written factor is hex-identical
  through the hoist (`Alpha 0x1.70faeb9ac5818p-22` and every L knot) and so is a chain-emitted
  plain fit; `fx_surface_block` writes only what a family declares, with a gate per family, and
  the chain emitter admits the family through one `FAMILY_HEADER` registry. THE OBJECTIVE IS
  MONTE CARLO AND DETERMINISTIC: one fixed-draw antithetic walk per evaluation on the QUOTES' OWN
  clock - a stub step lands each block exactly on its quote's `T`, so the variance the fit reads at
  a maturity is what a pricer reads at that tenor of the written curve (the trading-day clock
  mis-read a 1m ATM by 0.12 vol points) - each block priced off its sample forward as a martingale
  control variate (at 8,192 paths the fixed draws leave `E[exp(M + V/2)]` 15 bp off the forward,
  which the L bootstrap had absorbed as a fifth of a vol point), the residual
  `(V_model - V_market)/vega_market` in vol space to first order with no root find on the tape, the
  Jacobian one vmapped backward reading a central difference to 1.6e-6, and `L(0)` tied to the
  first pillar (spec 5.4.3) so no free phase remains. `Lambda` never enters a fitted vector: it is
  derived by 5.1 off the wing the desk hedges and re-derived only as the jump share moves, by a
  bounded scalar search, since an inverse-CDF count off a fixed uniform is a step function of it.
  `C_Min` is one declaration read three ways - the block writes it, the factor asserts it, the
  calibrator bounds `Rho_S` with it - with the penalty sized to bite in the last percent of the box
  and the BOX binding a refusal by name (the earlier unsized penalty was a wall every fit stopped
  on, which had read as "the surface wants a one-shock model"); the jump-share box is applied to
  the REALISED share, printed beside the one asked. `verify` runs before `report`, so a refusal is
  a message rather than half a log. THE IDENTIFICATION TABLE publishes the SVD of the DATA rows of
  `least_squares`' Jacobian at each fitted point, columns scaled by `1/||J_:,j||`, with the unscaled
  norms beside it (the penalty rows' leading singular value was an algebraic constant of `C_Min`),
  taken twice for the polish where `Forward_Smiles` is set. MEASURED on the campaign's CJOW surface
  (34 premiums to 2y, 8,192 paths, daily step, CPU; `artifacts/logvar2fj/harness_cjow.py`, spec 8
  steps 0-4): the spot fit reads **0.99 vol points unweighted and 0.34 vega-weighted** in 144 s and
  47 evaluations with no stage capped, against spec 9's 90 s on a GPU and spec 8's 0.2 vol points
  from one month out - the short end carries all of it (2.33 at 1m: the 70-80% wing +1.56 RMS and
  the 110-120% convexity +0.80, which is what two shocks and a co-jump cost against a one-shock
  reference), the 90-110 slopes 68.9 / 52.0 / 36.7 / 24.3 / 13.7 against CJOW's 83.7 / 75.4 /
  38.0 / 23.8 / 13.7, the marginal KS 0.015-0.055 against CJOW's exact CDF, every ATM pillar to
  8e-15. With `Forward_Smiles` at CJOW's three tenors and buckets [0.5, 1.0] the composition
  residual at 2y - the first rung BEYOND the last boundary, which is 5.3's check - is 0.108 vol
  points, the vanillas at the target maturities improve by 0.196, and psi reads 0.964 / 0.956 /
  0.865 against 0.973 / 1.008 / 0.620: on this surface the forward block does not buy the
  1y-into-1y ratio, the vanilla-only fit already reading 0.983, and what it buys is
  identification (`Rho_S[1y]`'s column norm 1.59e-2 with the forward rows against 4.90e-3
  without). The 2y SPX autocall reads -32.875 off the vanilla-only factor and -33.651 off the
  forward-target one against CJOW's -32.068: **the deal's forward-skew sensitivity is -2.4%**, and
  `SE^2 x wall time` on the deal value is a third to a half of CJOW's across two runs (spec 9
  states the target on the coupon leg, which nothing yet measures). `Nu` is unidentified on this
  surface - 1e-12 in every fit, a column norm of 9.7e-4 against `Sigma_S`'s 7.8e-3. The noise
  floor at 8,192 paths is 0.1-0.6 vol points on the ATM term structure and 2.6 on an L pillar; the
  fast trio holds to 2% across seeds and the slow pair (`Sigma_L`, `Rho_L`) does not, which is the
  valley a warm start slides along to a point 3% worse in the objective and 7% better unweighted.
  NEXT (spec 5.4, contracted): the inner bootstrap at every iterate with the IFT through the root
  and `Quote_Sensitivity` through both solves, `Fit_Mode: Bootstrap` with four bucketed curves,
  `Internal_Step_Days` written onto the factor and asserted at pricing, the `Floor` arm and
  `Stationary_Spread`, the share-measure forward-start (`E[S_T1 (R - k)^+]/E[S_T1]`, 0.4 vol points
  of level from what the block prices, so market quotes are not yet a legal source), and the fit
  object the per-iterate bootstrap wants. `tests/test_market_prices_partition.py` 38,
  `tests/test_equity_chain.py` 39.

- **LogVar2FJ phase 2, the three pricers - TARF, accumulator and discrete barrier on the (m, s)
  substitution** (2026-09-05) - the three OSS pricers take the walking kit's block law where GBM
  reads its own step: `hn` means `kit.daily`, `walks` a kit with no per-step state, `kit is None`
  GBM, one spelling per pricer; a walk per MTM row after the row's uniforms, antithetic in `-eta`
  and `1 - u_N` as `oss_uniforms` mirrors, hands each interval its `(M, Sigma)`, which the bound,
  the truncated draw, `otm_analytic` and the advance all read. A barrier observed on the internal
  grid is a block of one internal step and exact; off it the intervals are sub-divided
  independently, so the exactness is the grid's. Both European legs are one kit verb,
  `LogVar2FJKit.european` - the strip summed to expiry, `lognormal_fired_gain` per path or the
  block's Phi, averaged - with the division by Sigma guarded (spec 2.8): a terminal row has no
  strip left, and an unguarded 0/0 at the money took an exposure matrix to NaN. ONE ESTIMATOR PER
  DECISION: an already-knocked-in barrier IS the in-out-parity vanilla, so it leaves `sim_spot_oss`
  as a node by-product instead of being drawn a second time outside the recompute node and on the
  tape - the leg that made `Greeks: 'First'` move a live down-and-in's reported value 3.1%, now
  hex-identical with greeks on and off, the CVA and its L-curve, jump and spot gradients
  bit-identical with `Recompute_Inner_MC` on and off. The four deals declare the model and one
  `set_spot_model_index` writes `Internal_Step_Days` beside `HN_Steps_Per_Year` at all five
  compile sites, where three spelt it and four did not. Every existing reading is unmoved: 14 GBM
  and Heston-Nandi documents hex for hex against a pristine 454f457 tree, 17 more the reviewer
  authored (observed fixings, same-day, partial-time, quanto, compo, digital), `tarf_hex.py`,
  `hn_hex.py`, `run_trials.py base` digit for digit, the GBM limit at rounding on all four
  documents (1.7e-15 / 1.7e-15 / 2.4e-16 / 1.7e-14, the last the vanilla leg's pairwise-sum
  rounding). Every Q-sized value lands within 1.3 SE of a brute-force oracle that draws every
  internal step's return, once the engine's own seed error is counted beside the oracle's, and
  in-out parity reads 79.702795 from both sides against a 79.754 +/- 0.195 vanilla. On the smooth
  value at 65,536 paths the spot and `Mu_J` deltas agree with CRN ladders to 0.4% and 0.09%,
  `Greeks: 'All'` flows on both smooth arms with gamma 1.09% and 0.28% off the AAD delta's own
  ladder, and the crisp splice's delta is the smooth arm's to the bit; the `Rho_S` rows are a
  consistency check on shared draws, not a resolved sensitivity - their seed scatter at 65,536
  paths is 16% and 38%. The accumulator's crisp and smooth values coincide under every model:
  that loop already integrates the knock-out. OPEN: the European leg still branches on the family
  in two places, since the HN closed form wants a step count and a scalar carry where LogVar2FJ
  wants the walked block; the collateralised CVA spot delta stays 5.13% off a ladder at 9.73%
  flatness (the open row above). `campaign/lv_deals.py` and `artifacts/logvar2fj/oracles.py` are
  the documents and the oracles; `lv_deals.py guards` holds the two regressions. Net +80 tracked
  lines, `utils.py` untouched.

- **LogVar2FJ, phase 1 - the kit, the factor and the autocall's arm** (2026-09-05) - two
  mean-reverting log-variance factors and a co-jump, walked on an INTERNAL step and handed to the
  autocall as each fixing interval's own Gaussian block law. The model is free functions in
  `utils` (`lv_walk` and the OU filter both ways, `lv_counts`, `lv_cap`, the curve mapping, the
  cumulants), the factor is `LogVar2FJModelParameters` (seven scalar leaves, three STRUCTURAL
  scalars - the counts' law is not on the tape and the cap is a guard - and three curves whose
  knots are structural and values leaves), the kit is `pricing.LogVar2FJKit`, and
  `Internal_Step_Days` is a
  NUMERICAL setting beside `Steps_Per_Year`. The three kits now DECLARE what `oss_model_scalars` /
  `oss_model_kit` used to branch on the subtype string for - `param_names`, `curve_names`, `daily` -
  so the readers are registry lookups and every Heston-Nandi document is hex-identical
  (`run_trials.py base` digit for digit, `greeks`'s rows and every ladder, the GBM TARF, a
  component TARF and a plain-HN barrier). The GBM arm's `else` becomes the `(m, s)` branch and
  `branch_and_weight` ADMITS a non-daily kit, its refusal text kept for the daily ones. **The GBM
  limit is GBM**: every shock and jump off and the cap out of reach, the 2y SPX autocall reads
  -52.24366533926075 against GBM's -52.24366533926085 (1.9e-15 relative, `-0x1.a1f306d03a597p+5`
  against `-0x1.a1f306d03a5a5p+5`) and the credit MC's CVA is bit-identical at
  `0x1.5057040000000p-4` - the walk drawing from the PLAIN generator, which is the row's own `u`
  stream only where `sobol` is off, so at 16 scenarios or fewer the two documents share no seed and
  the reading degrades to MC error (1.9e-3 at a batch of 8). At the spec's test defaults on a flat
  L: base valuation -52.686 in 1.0 s, CVA 0.0817 / FVA 0.1089 uncollateralised and 0.2256 / 0.3036
  under the CSA, the profile dispersed and the ledger's four coupons settled. Against CRN ladders
  at 65,536 paths: spot **0.54%** (ladder flat 1.05%), rates **0.81%** (2.06%), `Mu_J` **1.97%**
  (0.76%), and - nothing registering on this arm, the put leg being spliced - `Greeks: 'All'` FLOWS
  where the component arm refuses, the spot Hessian cell landing **0.013%** off the AAD delta's own
  ladder. THE TAIL PARAMETERS' LADDERS ARE THE CRISP VALUE'S NOISE, not the tape's: on the crisp
  arm `Sigma_J`, `Nu`, `Rho_S` and `Sigma_S` read 3-28% off ladders that scatter 2-112%, because
  a finite difference in a parameter acting on the tails counts knock-in flips in a region few
  paths reach; the same AAD (identical on both arms, the splice) against ladders of the SMOOTH
  value (`Branch_And_Weight: 'Yes'`, the put integrated) reads **0.02% / 0.01% / 0.02% /
  0.00%**, ladders flat to 0.07%. At the 2026-09-05 defaults (sigma_l 0.4, sigma_s 1.5,
  lambda 0.35, mu_J -0.15, beta 0.25), since superseded by the Q-sized set, the crisp document's
  spot reads 0.53%, rates 1.17%, the L curve 1.11%, `Mu_J` 0.28% on a ladder flat to 0.03%,
  gamma 0.02%. THE CVA DELTA IS THE OPEN ROW: at 2,048 outer the authored daily
  step does not fit the card at all, and at a 21-day step the uncollateralised spot delta reads
  +5.524e-5 against a ladder extrapolating to +5.55e-5 by three points or +5.82e-5 by the two
  finest - 0.5% to 5.1% - while the collateralised row runs out of memory there and is unread.
  `Recompute_Inner_MC` on and off agree BIT FOR BIT on the CVA and its spot gradient, on the limit
  document and the default one, the walk living inside `sim_spot`; the model's own leaves agree to
  1e-7, which is float32 reassociation under the node rather than a dropped cotangent.
  The pack's `checks.py` is the model's gate until `tests/test_logvar2fj.py` lands (67 checks with
  one deliberate miss: at the Q-sized defaults a weekly internal step is not licensed on the coupon
  leg, the 5-day gap 3.4 SE against the daily walk), and the block-summing scan the pricer walks - `lv_walk(blocks=...)`, which never materialises
  `m_x` or `var` - matches the matmul filter to the gate's 1e-12 and takes 183 s against 244 s at
  1024 x 4096 x 1260 on CPU.
  THE TWO FORWARD-SKEW LEVERS NOW CARRY CALENDAR BUCKETS (2026-09-05): `Rho_S` and `Mu_J` are
  `bind='value'` curves whose knots ARE the buckets' start times in years, the kit reads both at
  the ABSOLUTE step-start times as it reads `L`, and the factor refuses at load a bucket whose
  `c = 1 - Rho_S^2 - Rho_L^2` falls under the factor's own `C_Min` (default 0.12), two curves
  whose knots differ, or a lever still spelt as a Float - so
  one knot at 0 is the constant-parameter model, which reads phase 1's MTM to the bit
  (`-0x1.a32dcde94c00ap+5`) with the GBM limit still `-0x1.a1f306d03a597p+5` and its CVA
  `0x1.5057040000000p-4`, while a second bucket at 1y (`rho_s` -0.80, `mu_J` -0.20) leaves a
  9-month deal at `-0x1.8228e7d1645d2p+3` under both markets and gives its own year-two leaves a
  gradient that ladders **0.02%** (`Rho_S`) and **0.12%** (`Mu_J`) against CRN on the smooth value.
  A boundary is matched to the walk time within `utils.BUCKET_TOL`: 252 daily steps accumulate to
  1 - 3.1e-15, and without it a knot authored at exactly 1y would start its bucket one step late.

- **Conditional-p at a jump, the GBM arm** (2026-09-04) — the crisp GBM autocall's put leg and
  TARF's knock-in take `splice_conditional_p` off the fixing interval's own lognormal law, which
  under GBM IS the simulated step, so the mixture is verbatim with `p = Φ` and the
  `InnerBoundarySet` each registered is superseded (one estimator per decision). Two clauses;
  values bit-identical, hexed on both deals and the campaign's four base documents. The 2y USD SPX
  autocall under base valuation at 65,536 paths against CRN ladders: spot **0.05%**, vol 0.83%
  (ladder flat 0.46%), rates **0.07%** at the crossing rungs 5e-3–1e-2 — at 1e-4 a 16,384-path
  ladder counts single path flips, one worth 90 on 655 — and, `Greeks: 'All'` now flowing where it
  refused, gamma **0.02%** and vanna **0.02%** against the AAD delta's own ladders. The two-fixing
  TARF against the quadrature table: delta 0.71% → **0.006%**, vega **0.015%**, the smooth arm's
  own digits (the derivative IS the smooth arm's, the two spellings differing only in what they
  report forward); second order stays behind `Branch_And_Weight` there, the target latch still
  registering. The GBM CVA delta moves in the sixth digit; at 2,048 outer the uncollateralised
  ladder rises linearly in h (5.11 / 5.48 / 5.84e-05 at 2% / 1% / 0.5%, the relu's O(h)) and the
  AAD sits 0.5% from its h→0 extrapolation, while under the CSA it reads **11.8% short of a ladder
  flat to 2.2%** — the open row's collateralised residual is GBM's too.
- **The component outer fit by autograd Levenberg–Marquardt** (2026-09-04) — `Outer_Search`,
  declared beside the simplex, which stays the default (hex-identical to what shipped) until the
  divergence wall below is fixed; the flip is one word. The residual
  VECTOR is one row per ATM pillar (an exact zero where it solved, its relative miss at
  `atm_constraint_weight` where it floored) plus one per wing contract, and its sum of squares is
  the scalar the simplex read — one set of prices, both spellings, so the two searches minimise the
  same number. **The inner root find is on the tape**: `brentq` still finds `L_k*`, and the level
  RETURNED is `L_k* − F_k/detach(∂F_k/∂L_k)`, so autograd carries `dL_k/dθ` and `dL_k/dL_j` exactly
  — the implicit function theorem as an expression, `quote_sensitivities.md`'s delta-solve splice.
  A floored pillar's level is `admissible_level(L_(k-1), ρ)` and differentiates through it.
  **Measured** against Nelder–Mead at its own tolerance, `Floor` on the bank books: USDZAR
  1,246 evaluations / 231 s → 75 + 42 Jacobians / 103 s; SX5E 1,883 / 312 s → 60 + 39 / 94 s; NKY
  1,614 / 864 s → 52 + 32 / 354 s; SPX 1,351 / 1,053 s → 48 + 25 / 819 s, where LM is also the
  BETTER fit (5.696e+02 against 5.784e+02) and floors no pillar where Nelder–Mead floors one.
  Elsewhere the residuals are within 1.4% either way, the worst wing the same or better (NKY
  33.70% → 27.17%), every ATM row 1e-15. The Jacobian agrees with a central
  difference to **5.3e-09** per column at the seed, the disagreement growing as `h` shrinks.
  **What LM does NOT buy**: it stops where the MGF diverges. A 1e-5 step in `β` at the fixture's
  optimum caps the `φ_max` scan on the 126-step pillar's floor probe, and LM started at
  Nelder–Mead's converged θ\* cannot take one feasible trust-region step — a simplex slides along
  that wall, a trust region only scales into it. `Nelder_Mead` stays reachable for exactly that.

- **Barrier state as a fold over fixings, the buildable half** (2026-09-03) —
  `tests/test_barrier_fold_json.py`, 29 gates, base valuation and credit Monte Carlo. **No deal stores a consequence**: `Barrier_Hit` is
  gone from `FXAccumulatorOptionDeal` and from `structures.py`'s accumulator leg, and a document
  carrying it refuses `UnpriceableSchedule` by name whatever it says — measured first, the flag and
  the fact it asserts give the same bit-identical zero, so the OR could only change an answer where
  nothing backed the claim. The accumulator's prefix is now `Prefix_Breached`, the fold alone; the
  fixture reprices at 62.428908447906807, unmoved. **The `Observed` column**: a `Barrier_Dates` row
  on the two discrete deals is `[date, Observed]`, blank by default. Blank on a future date is
  nothing; blank on a date ≤ base date refuses — absence of a fixing is not absence of a hit, which
  is what the one-column table said silently, since `get_start_index` walks past every resolved row.
  Nine documents not using the column reprice byte-identical across the change, plan hash and factor
  universe with them. **The fold** runs at `Deal.resolve_history`, called once per deal inside
  `add_deal_to_structure`'s own guard: a crossing leaves no decisions, so the deal compiles as the
  deal it BECAME — a hit KI as `EquityOptionDeal`/`EquityBinaryOption`, a hit KO as the
  `FixedCashflowDeal` paying its rebate on the crossing date — built through `construct_instrument`
  under that type's own valuation block. Construct-time substitution rather than a `torch.where` in
  the pricer, because `pv_discrete_barrier_option`'s existing `hit_value` leg is the argument: a
  second spelling of the same European that shipped once marking already-hit rows at +1432%. A t0
  fact is a scalar, needs no per-scenario branch, and registers no boundary set — gated through the
  document, `Greeks: 'All'` refusing on the unfolded barrier and answering on the folded one, to the
  bit of the vanilla's own second-order block; under a credit Monte Carlo a folded document
  walks its own document's profile bit for bit beside the unaffected ones. What the fold is worth: the knocked-in call reads
  +74.9% unfolded, the Up-and-In +193.0%, and the knock-outs mark 18.23–30.98 of option value they
  no longer own against a rebate of 40. Five mutations, every one caught: strict crossing (2), blank
  Observed walked past (1), either substitution dropped (6, 4), the fold never run (11).
- **Composition harness** — `gates/hw2f_composition.py` + `tests/test_hw2f_composition.py`, 14
  gates, run on live prints 2026-09-01. Curves reprice their own quotes to 1.07e-12 / 1.69e-13 bp
  and the 54-row ZAR normal-vol ladder fits to 1.348 bp rms; the fit is ρ-invariant on the
  *residual* at max |Δ| exactly 0.000e+00 while the emission moves, and the composition is
  ρ-invariant at ≤ 0.08σ. `K` is load-bearing: zeroing the emitted correlations moves identity 1 by
  −4.99 / −6.58 / −9.61 pp. Identity 1's own miss is a numeraire and step effect, not a measure
  error — 5Y×5Y reads −23.9% quarterly to −9.4% weekly — and the **static** leg's error grows as the
  grid refines, since a frozen t=0 curve accumulates `T·r(0,Δt)` instead of `−log D(0,T)`. The
  canned twin is authored flat-based and short-dated for that reason and closes at +0.26% / +1.13%.
- **The stride** (2026-09-02) — the k-step conditional law of `ln S` given `(h, q)` under
  component HN, cached and exactly differentiable, additive to `utils.py` with the plain path
  AST-untouched: cached A/B/C strips per (fixing interval × quadrature node), Gil-Pelaez Φ over the
  cache, survival-truncated inversion, and carried state by quadratic conditional matching with
  autograd mixed moments. The step returns the spot un-shifted into the deal's own carry — without
  it survivor quantiles sat 26.8× outside the walk's band, with it 0.44×. 69 oracle gates against
  the exact 2D conditional sampler, error scanned in k (peak at k = 24–25). **The speed claim stays
  refuted** (re-measured 2026-09-03 after the batched Fourier layer): order 100× slower than the
  daily walk — quote the strided wall clock, not the ratio: 2.58–3.58 s against 2.96–4.74 s before,
  13–24% off the fixed cost, while the daily leg is 25–38 ms and carries ±20% of any ratio on its
  own. Still no crossover; it stays because it is the smooth estimator's conditioning law.
- **The stride's three consumers, four pricers** (2026-09-02) — HN branch-and-weight on the
  no_averaging paths (crisp mixture and smooth arm report the same delta bit for bit on values 2%
  apart), the conditional-p jump gamma with Φ as `p` (supersession gated on the registered
  BoundarySet type list, one estimator per decision), and fixing-to-fixing stepping behind
  `HN_Stride`, default off and hex-identical (0.08% at 0.22 SE against the daily walk).
  `tests/test_hn_stride.py`, `tests/test_hn_stride_consumers.py`.
- **Branch and weight for TARFs and autocalls, all four products** — `Branch_And_Weight: 'Yes'` on `BaseValuation`, off
  byte-identical; `tests/test_branch_and_weight.py`, 110 gates. At each fixing the fired branch
  closes analytically and the continuing branch draws from the truncated law, so the estimator is
  unbiased rather than smoothed: the two-fixing TARF lands on an independent Gauss–Legendre region
  integral at 1.1e-3 / 4.8e-5 / 3.6e-4 / 9.7e-4 on value / delta / gamma / vanna, variance ratio
  11.9× at 4096 paths, and the target pin becomes exact. The autocall's put leg is one
  `lognormal_fired_gain` conditional on survival — written without that division it reads 61.8% out
  on the strike — closing an 18–22% ladder miss to 0.16%. The averaging coupon is admitted under
  `LogVar2FJ` (the window sampled, the prefix truncated); GBM and the daily kits refuse by name.
  `TargetAdjustment` is retired: it repriced a 'Full Gain' deal 44% on a flag documented as variance
  reduction.
- **Exposure gamma at a kink — the ½Ku² term** (`Hessian: 'Yes'` on the CVA block;
  `pricing.exposure_kink_term`; `tests/test_cva_gamma_kink.py`, 11 gates). What second-order AAD
  drops at a relu is `δ(V)·V_θV_θᵀ`; with `u = V − V.detach()`, `T = ½·K_ε(V.detach())·u²` beside
  `relu(V)` recovers it, and since value is an exact zero and first order accumulates `+0.0` bit for
  bit, admission is byte-identity at both orders with the term on or off. Pathwise gamma
  0.0 → 4.2419e-04 against a CRN ladder at 4.2609e-04; vanna +4.964e-03, wrong sign, → −1.2844e-02
  against −1.3044e-02. The ATOM refusal is re-gated on a bandwidth ladder whose f̂(0) climbs as 1/ε
  (8.000 across a factor-8 ladder against 1.03–1.06 healthy). v1 CVA gamma is for uncollateralised
  books of smooth products — collateralised sets are preempted by the decision-product refusal.
- **Conditional-p at a jump, the HN arm** — the latch's two whole-profile branches are the mixture
  components, `P_vib = p·fired + (1−p)·not`, spliced so the reported value is the crisp estimator's
  bit for bit while every derivative is the mixture's: first order Rao–Blackwellised, second order
  analytic, no bandwidth. Where the mixture takes a decision it replaces that decision's kernel-flux
  estimator at every order, never both, or the flux double-counts. The kernel form it retires worked
  at 27% noise with a ladder that never plateaued.
- **Calibration Jacobians, all four increments** — `Quote_Sensitivity`;
  [Quote Sensitivities](quote_sensitivities.md). `dV/dq` beside `dV/dθ` for a zero curve from
  deposits/FRAs/swaps, HW2F from swaption vols, the integrated vol curve, and the Malz FX surface,
  with solved numbers bit-identical either way. Increment 2's fixed point is stationarity, not a
  root, and the two terms Gauss–Newton drops — half `J'J` (0.500064) and half `J'(∂r/∂q)`
  (0.4953–0.5115) — **cancel**, so no correction is owed; correcting one side alone invents a 3/2,
  now the gate's mutation. Increment 3 needed no solver: the vol curve is a closed-form
  forward-variance walk, identity wherever variance rises, the declining repair a kink
  ([the closed-form map](quote_sensitivities.md#the-closed-form-map)). Increment 4 takes one Newton
  step off the converged root, because differentiating a bisection reports `d(bracket)/dq` — a
  plausible 1.000137 for a truth of 0.865559 — and the x-grid stays
  [pinned](market_prices.md#fxvolprices), since a grid following its quotes is a recompile per tick
  ([the delta solve](quote_sensitivities.md#the-delta-solve)); four things are discrete per node,
  three kinks and one jump, and a flat smile divides zero by zero in the backward unless guarded.
  **Bump-and-recalibrate is ill posed here** and was refuted three times — on four quotes the
  solution is a 19-dimensional manifold, and on 25 quotes at full column rank the optimizer stops
  seven and a half of eight orders short of stationarity — with every refutation pinned as a gate
  ([the manifold finding](quote_sensitivities.md#the-manifold-finding)). Increment 1's two traps are
  unrepresentable rather than fixed, since `TensorSchedule.bind` gives the tensor half a birthday
  ([the schedule lifecycle](calc_lifecycle.md#the-schedule-lifecycle)).
- **Two-sided wings and the FVA column** (2026-09-02) — each side's copy of the surface skews by the
  wing halves through the same strangle algebra, a wing-less book byte-identical, the reported
  `wing_spread` the half the quote dealt at. `fva` lands beside `cva` off one run per set, a live
  run missing the table filing `failed` rather than a null.
- **The schema is emitted from the declarations.** `derivus/fields.py` is a 22-line deprecation
  shim; `schema.py` holds the vocabulary, the emitters and the assembly. Seven stores are views of
  per-class declarations — Instrument, Factor, Process, Calculation, Calibration, MarketPrices,
  Interpolation — each with a declared-versus-read audit and a both-directions gate. A section owns
  its descriptors (`sections[S]` is `{json_key: descriptor}`), so `ALIASED_KEYS` is gone and one
  field name can mean different things on two deals. Authoring rules are stated by
  `schema.validate_instrument`: `default=REQUIRED` for presence, a class's own `validate()` for the
  16 cross-field rules over 6 predicate forms an audit found. What it settled: **the
  declaration is the single source of an omitted field's default** — `schema.declared_defaults`
  completes the params dict at the top of every `execute`, with an AST gate holding any surviving
  fallback to its declaration. Four disagreements settled: `Random_Seed` 5120, `MCMC_Simulations`
  32768, and `Dynamic_Scenario_Dates` / `Generate_Cashflows` `'Yes'` — a results-changing flip and
  measurably the more accurate grid, since `'No'` misses 2.56% of knock-outs and biases the profile
  +0.71% at 175σ. Kept as unbuilt functionality: four declared-and-unread calculation fields plus
  `Generate_Instruments` / `Generation_Parameters`. Undeclared by decision: `DealLevel` (a bool where
  the vocabulary spells flags as strings) and `LegacyFVA`. Two carried notes: `check_interpolation`
  falls through to `Linear` for anything it does not know, so an unimplemented method offered is a
  curve silently interpolated the wrong way; and `SurvivalProb` overrides it and is not routed, so it
  always extrapolates — declared by its absence from the menu.
- **PREPARE / EXECUTE — the hashes.** `cx.plan_hash()` and `cx.values_hash()` are sha256 over a
  canonical dump: PLAN is `params` and `deals` less every `bind='value'` field and `Random_Seed`,
  VALUES is exactly `cx.market_patch()`; the replay tuple is documented in
  [API Overview](../api_overview.md#patching-market-values-and-replaying-a-run). `bind=` and the
  partition are built — 39 fields over 23 types, each declared from its consumption site, a
  shape-valued field splitting inside itself. Quotes are on the values plane
  (`schema.MARKET_QUOTE_VALUES`, `partition_market_price` / `apply_market_values`), so a vol tick
  moves `values_hash` and leaves `plan_hash` bit-identical — the disjointness the spine's quote
  firmness needed. One rule, all seven families: the two carrying `Points` have a values half and the
  five that do not are asserted empty by name in `tests/test_market_prices_partition.py`.
  `/book/market` is stricter than the engine and refuses a quote-values patch by name. The
  market-data half of the split is [Quote Propagation](quote_propagation.md).
- **The trading spine, increments 1–3 of 7** — [The Spine](spine.md). Increment 1 is the append-only
  book of record: a vendored RFC 8785 canonicaliser gated on the RFC's vectors, sealed chained log
  segments (AES-GCM bodies, a blinded idempotency tag so no plaintext hash is a dictionary oracle),
  a content-addressed blob store with no verb for forgetting, Ed25519 checkpoints resolved by
  fold-at-LSN, an enforced single writer, and a two-mode verifier honest about what a keyless
  replica cannot assess — 103 gates, every fault injected as data on disk. Increment 2 is identity,
  attribution and key custody: local OIDC verification, capabilities as one hashed policy document
  plus a pure evaluator, writer enforcement logging every denial as a fact, an unreadable-fold
  sentinel, revocable break-glass, per-seat X25519 wrapping, escrow recovery. Increment 3 is the
  booking verbs, the attestation lanes and two-dimensional quote firmness on the increment-1 wire
  formats untouched — 225 gates, twice-run deterministic. One rule carries the lanes: a run is
  recorded iff its output will be cited by a fact. `derivus/spine.py` is the one module under
  `derivus/` that knows the record exists, lazily imported and refusing without the `enterprise`
  extra; `DV_SPINE_HOME` unset is bit-identical to the edge as it was. `DV_Spine` is the CLI.
- **Service layer, slices 1–4**, plus the book, the web UI and the [MCP binding](mcp.md).
  `derivus/service.py` is the HTTP binding of the Context verbs and owns no logic — one vocabulary,
  two bindings, and anything a client needs that the verbs cannot answer is a missing verb on
  `Context`. `/schema`, `/schema/job` (the envelope, a skeleton that loads), `/validate`,
  `/describe`, `/prepare` (a 32-entry LRU of pristine parses handing out deep copies), `/execute`
  (document or `{plan_id, Patch}`), and `/results/{id}` as a summary with cells paged from
  `/results/{id}/{table}` — never return the exposure cube. Pricing goes through one worker thread on
  a cost-class priority queue, and `result_id` is the content hash of the replay tuple, so an
  identical submission coalesces. The decisive gate is parity: a job over HTTP prices identically,
  table for table, to the same job through `run_job`. The book surface serves one live job document
  whose file is the source of truth — `/book`, `/book/deals`, `/book/price`, `/book/amend`,
  `/book/market`, `/book/structure`, `/book/quote`, `/book/bloomberg`, `/book/risk`, `/book/xva` —
  every write validated before an atomic rewrite in the file's own indent, so book-then-delete is
  byte-identical. `DV_HOME` names where a desk's files live; `DV_Service --tick SECONDS` runs a
  metronome submitting the same queued Bloomberg job, skipping a beat whose predecessor is in flight
  and stretching the interval fivefold after three failures. Three clients sit on the one surface:
  the web UI (`web/`, rendering entirely from `/schema` and the document, value-first field
  dispatch, views in a workspace registry), the Excel add-in (a plain `requests` client importing
  neither `xlwings` nor `derivus`), and `derivus_mcp/server.py`, whose docstrings are the
  model-facing contract and whose import surface is held by a gate.
- **Structuring is a solve verb, not a calculation type** (decided 2026-08-26). `POST /book/solve`:
  brentq inside declared bounds else a secant (exact in two pricings for an affine field), the
  candidate priced alone on the book's market data, the seed fixed so a Monte-Carlo objective is
  deterministic, the residual checked against a declared tolerance rather than clamped, and the
  result's tables being the run *at* the solved value. Multi-strike structures compose from 1D
  solves; no N-D optimizer until a coupled case arrives. Structures are declarations
  (`derivus/structures.py`, emitted as `mapping['Structure']`) carrying sales names, legs and a
  recipe the runner solves server-side, and the runner owns every market-axis conversion — a strike,
  a barrier level (which inverts like a strike, its Up/Down direction with it), and the
  market-call-is-an-engine-put flip — so the finance never depends on which model drives the tools.
  `FXForwardDeal` is a quotable `InterestRatePrices` benchmark
  ([Market Prices](market_prices.md#interestrateprices)), so a currency's curve solves from forward
  outrights with CIP written nowhere.
- **Risk-impact pricing v1** — [Structures](structures.md#risk-impact). A trade's charge is the cost
  of hedging the residual it leaves, at the market's own two-way: the composed candidate is mirrored
  and the book's vol risk read with and without it under `Greeks: 'First'` in quote coordinates,
  each bucket's move charged that bucket's half-spread. V1 scope: vol only, quote-space, one
  re-solve rather than a fixed point (0.026% on the gate's book), no surcharge past the two-way, and
  the `Quote Policy` block's absence as the off switch. Measured: a desk quoted the collar it already
  holds tightens from 81.2194 USD to 75.9178 on a saving of 10.6031. The spread input rides the vol
  pillars as `Quoted_Bid`/`Quoted_Ask` beside the mid — mid for the book, two-sided for the quote
  ([Structures](structures.md#two-sided)).
- **The blotter's two data views**, deliberately different kinds of thing. `GET /book/risk` is one
  base valuation with `Greeks: 'First'` over the whole book, counterparty-blind, cached under a
  content etag (338 ms cold, 3 ms warm). `POST`/`GET /book/xva` is a cached projection in
  `DV_HOME/xva.json` — a credit Monte Carlo is minutes of device time and must never ride a tick —
  one queued job per set at the CMC's cost class, each row carrying its own `as_of` and replay
  tuple, staleness as data rather than a failure. The three MCP tools say the distinction in their
  docstrings.
- **VALIDATE** — `cx.validate()` returns `{'deals': {reference: [message]}, 'factors': [name]}` and
  nothing else, reusing `discover_factors` rather than `calculate_dependencies` (which mints the
  `Price Models` dummies that make that method non-idempotent). The discovery half is
  `Config.factor_universe`, returning both `resolved` and `missing`.

## Test selection

`gates/reach.py` answers, in a second, what a change reaches and which documents read it: `python gates/reach.py <Symbol|Class.method>` prints the consumer classes (deals, pricers, processes, factors, bootstrapper families, calculations, models), the callers, the JSON documents that executed the symbol fastest-first and the HOLES no document reaches; `--from <Consumer>` is the inverse; `--dirty` (or `--since <rev>`) diffs SYMBOLS by AST on both sides, reaches each and greedy-covers them by document. The static half is an AST call graph over `derivus/` (2,182 symbols, 0.7 s) whose exact edges come from import tables, `self`/MRO and every registry read as data - the five `construct_*`, `spot_models`, and any module-level `UPPER = {key: symbol}` dict found by shape - with duck-typed calls kept as guesses that are not chained (`--loose` chains them; chained, 45 of 50 deals reach `lv_walk` through `.blocks` on an unrelated object). The dynamic half is `artifacts/reach/document_map.json`: every job JSON under `tests/fixtures/` and `artifacts/` run once through `derivus.Context` under a `sys.monitoring` tracer (a document carrying `Market Prices` is bootstrapped first), one representative per shape, 180 s cap, keyed to the engine commit so a query on another commit prints `STALE` first; rebuilt at a campaign boundary or after a rename (`--build-map --repo <clean checkout>`, ~16 min, resumable). Measured at e475bee (2026-09-06): 74 representatives for 869 job JSONs, 50 ran, 22 errored (LogVar2FJ documents authored before the levers became curves), 2 over the cap; 852 of 2,133 symbols executed by some document; no document reaches 40 of 50 deals, 24 of 34 pricers and 3 of 8 bootstrapper families; `LeastSquaresSolve` was reached by NO document in the corpus until the four-quote HW2F fixture was written the same day, which is the tool's own account of a minutes-long calibration having been run as a test. The B2 diff's minimal cover was 6 documents in 236 s. What it cannot see: string-keyed dispatch outside the registries, callables passed as values, `getattr`, virtual dispatch out of an inherited body (a calculation does not reach a pricer through `Deal.calculate`; the deal does), a branch no data takes (executed is not observed), a document over the cap, and `derivus_bloomberg/`. It is a sibling of `gates/impacted.py`, not a composition: one names documents and consumer classes, the other names test files.

`gates/impacted.py` selects the tests a change can reach: an execution-coverage map (per-test
contexts recorded at a campaign-boundary run via `pytest --cov=derivus --cov-context=test`, then
`--build-map`) joined with a static, always-fresh fixture map, because tests depend on JSON fixtures
at runtime and no import graph can see that. Selection is file-granular and fails open, loudly;
`derivus/__init__`, `utils`, `calculation` and `conftest` are whole-suite modules by construction.
The full suite still runs at campaign boundaries, and the tree must hold still while it runs: two
full runs each failed one mutation gate purely because source was edited mid-run, since
`inspect`-based gates read the file on disk at imported line numbers. Inner loop: `--dirty --run`.

## Tidy-ups

- **`artifacts/logvar2fj/`'s second spelling no longer imports** (2026-09-08) - `logvar2fj.py`,
  `checks.py`, `oracles.py` and `quote_risk.py` re-export `lv_counts` and walk the Poisson residual,
  so `fx_ladder.py`, `kkt_ladder.py` and `flat_l.py` are dead on this tree; lane 1 wrote
  `lv_nig_20260907/oracles_v2.py` and the estimator lane rebuilds `lv_hist_20260907/hist.py`.
  Delete the pack or re-spell the G-gate scripts against the NIG residual; the roadmap rows that
  cite it are records.
- **Inline comment density.** The boundary-correction work left ~12 inline blocks of 4–11 comment
  lines, several outweighing the code beneath them. House style is 2–3 lines maximum inline and
  never more comments than code; the material belongs in the docstring or the commit. Worst
  offenders: `pv_discrete_barrier_option`'s hit-mask and rebate blocks, `sim_spot_oss`'s terminal
  digital, `NettingCollateralSet.post_process`'s `net_from_gross`.
- **The compounding-leg shape check.** `pv_float_cashflow_list` selects the compounded-in-arrears
  path by comparing reset count to cashflow count — a shape encoding of intent. It works and is
  documented ([Quote Sensitivities](quote_sensitivities.md#curve-contracts)), but an explicit signal
  on the compiled cashflow object would say the same thing without the inference.
- **`Flot` and `Three` are gone** (closed). The widget tokens are `Curve` and `Surface`, the legacy
  spellings owned by the Jupyter front end's `LEGACY_WIDGET` map, and `BLANK`, keyed by type, is the
  one definition of a shape's blank. Two gates.

## Model punchlist

**HW2F: the α→0 repair is built, and the analytic price is an objective behind a declared field.**
`tests/test_hw2f_analytic.py` carries the repair, the checker and the wiring.

*The repair.* `hw_calc_IJK` divides by `a³`, `hw_calc_H` by `a²` and the `AtT`/`BtT` assembly by `a`
— all removable singularities, so the failure was silent cancellation rather than a raise. Each
closed form takes a series branch below a threshold measured against a 40-term reference and
re-checked against 60-digit mpmath, the branch jump at each crossover (2.8e-11 / 2.2e-11 / 7.2e-16)
being the closed form's own residual, so the accurate branch is taken on both sides. Reachability:
2.8% of 20,000 simulated basin walks go under the IJK threshold and `least_squares` crosses zero
outright under `alpha_bounds = (-0.5, 2.4)`; `params_ok` read True at `α₂ = 1e-4` where the
benchmark priced 21% wrong; `Alpha_1`/`Alpha_2` default to 0, so an omitting block was NaN
everywhere. Two more the contract did not anticipate: `J[i][j]` is taken at `α_i + α_j`, so
α₁ = 0.1, α₂ = −0.1 is singular with neither reversion speed near zero; and the repo's own
identified fixture engages the branch at its solved point (`Alpha_2 = −0.0179`).

*The checker.* `schrager_pelsser_swaption` — annuity weights frozen at t=0, constant loadings,
Bachelier — built off the `J` array `precalculate` already computes, now retained so there is one
spelling of `J`. Building it corrected the derivation: the `e^{-(α_k+α_l)T₀}` prefactor does not
belong, because freezing the bracket freezes the loading on the scaled martingale
`Y_k = e^{α_k t}x_k` and `J` is `Y`'s own covariance.

*The decomposition*, at θ\* on the 25-quote identified fixture, 2²⁰ paths, pathwise under common
random numbers (closes to 1e-18):

| term | size |
| --- | --- |
| the leg convention (single-curve float-at-par) | ≤ 5.6e-18 — exactly zero; the leg telescopes path by path |
| SP's annuity-freezing bias | −0.13 to +2.17 bp of normal vol |
| the MC's own numeraire bias | −0.35% to −1.61%, i.e. 0.6–3.0 bp — systematic |
| the MC's noise per objective evaluation | 0.7–1.1 bp at its 8192 paths |

So SP is the *more accurate* of the two over most of the grid — inside one MC evaluation's noise at
22 of 25 benchmarks, stepping outside only at 3Y×10Y, 5Y×10Y and 10Y×10Y, and under 0.21 bp at every
expiry for tenors ≤ 3Y.

**`Objective: 'Analytic'` is the default** (decided 2026-08-31) on four readings: accuracy, above;
**stationarity**, which decides it — `‖J'r‖` at θ\* is 8.63e-7 on the plain residual, inside
`Stationarity_Tol`'s own 1e-3 default, against 3.16e2 on the quartic, which is why that gate file
must declare 1e5; determinism and cost — two analytic solves at one seed agree to the bit and the
four-quote chain is 13.4 s against 75.1 s; and the quote side existing, since `Quote_Sensitivity` on
this path used to refuse by name. Two prerequisites had to land first: the domestic-measure
correction, without which the flip would move foreign-curve answers, and the analytic quote side,
without which it would break a declared field. `Monte_Carlo` is unchanged to the bit and remains the
engine's own estimator — the oracle every comparison here is taken against, and what the analytic
solve's honesty reprice runs at θ\*. In repriced vol space the two agree (rms |SP − market| 5.14 bp
at θ\*_MC against 4.51 at θ\*_An); in θ space they share nothing (ρ −0.0046 against −0.9409), which
is [rank deficiency](quote_sensitivities.md#rank-deficiency), not a disagreement. Per-evaluation cost
is 0.437 s at 25 benchmarks against 0.036 s at four — twelve times the cost for six times the
benchmarks, because SP is one scalar call per benchmark. No JSON fixture in the suite bootstraps an
HW2F block, so nothing re-baselined; `derivus_bloomberg/swaption_vol.py` deliberately emits no
`Objective`, so a Bloomberg-emitted ladder now solves analytically and that emitter's docstring says
so.

**Standing consequences — two re-marking events.**
Every foreign-curve HW2F parameter set solved before the domestic-measure fix re-solves to a
different θ\*. Every HW2F θ\* solved before
2026-09-02 re-marks again on the `ALPHA_SEED` and premium-clock change: the seed moves where the
chain starts on a block with no existing parameter factor, and the clock moves the market side of
both objectives on every block. A desk naming an old θ\* has two options and no third — re-baseline,
or re-solve. Carrying one forward looks like the first and is neither: a block still holding its
parameter factor warm-starts off it, so the fit moves but the seed change does not reach it.
Emissions that move with it: `Quanto_FX_Correlation_1/2`, and every locus recorded downstream of a
fit.

**The cornered θ\* was a seed defect and is closed** — `ALPHA_SEED = (0.5, 0.05)` is declared. What
the symmetric seed cost was basin hopping's iteration-0 descent: at an exchange-symmetric point the
gradient carries the same number in both coordinates, so the old seed's first descent landed
α₁ = α₂ = 0.0118 where the declared seed's lands (0.500, 0.0027) at a lower loss. On the identified
fixture rms 4.5125 → 2.9594 bp with zero σ knots on a bound, and the projected gradient is 2.09e-7
against the retired vector's 4.47e-5. What remains at a bound is the fixture's own minimum: ρ runs
monotonically to −0.95 for every seed reaching the good basin. One reading went the other way and is
an open decision above.

**The analytic quote side is built**, off the same quote leaf, the same `black_premium` twin and the
same `LeastSquaresSolve` wrapper; `market_normal_vol` carries the splice through the closed-form
Bachelier inversion and that is the whole engine change. It is worth exactly zero forward
(bit-identical with the switch on and off). What it buys is a residual **separable** in (θ, q),
because the annuity the market half divides by is severed at source: of the two terms Gauss–Newton
drops, one is structurally zero and the other is the textbook `O(‖r‖)` at 1.50e-4 of `J'J`, against
the Monte Carlo residual's 0.500064. `∂r/∂q` is exactly diagonal — 600 of 625 pairs structurally
absent — and the path runs at `Stationarity_Tol`'s own default where the MC path needs 1e5.
Triangle: 2.22e-16 against the spelled-out contraction, 1.088e-14 against the operator form, and
`h²` twice over against a re-authored central difference. Two findings ride along: `.grad` standing
on the quote leaves after an analytic chain is 0.31%–2.33% of the answer — plausible-looking rather
than obviously wrong, so `bootstrap`'s clear is load-bearing in a way no gate could previously see —
and the re-solve oracle still scatters on a solve that does reach its minimum, every re-solve
landing 0.27–0.30 from θ\* whatever the bump. See
[the analytic quote side](quote_sensitivities.md#the-analytic-quote-side). Open build: batching SP
across the benchmark set (25 scalar calls lose to one batched kernel, 0.158 s against 0.140 s on
CUDA).

**The equity chain emitter is built** (`derivus_bloomberg/equity_chain.py`,
`tests/test_equity_chain.py`); the three engine findings it named are closed above. Chain discovery
goes through the package's own session seam with every response screened as untrusted evidence — a
live SPX chain measured 8,000 asked / 3,729 believed / 4,271 refused, led by stale and
no-open-interest. Pillars are matched to listed expiries one-to-one by nearest-claim-wins and a
pillar the chain cannot serve is dropped by name; before that rule the review measured 73% of an
objective landing on a single print. The undeclared-dividend carry is a median over five two-sided
parity pairs with a band refusal, weight is `vega·√OI/(1 + spread/cap)`, the distinct-contract floor
of eight is counted after snapping, and premium quotes carry the two-way. The emitted block fits
through the real bootstrap on a book with no surface in it — 0.272 vol points over six rungs, ATM
misses 2.0e-15 and -1.1e-14 (`artifacts/hnret/chain_block.py`); it read an ATM residual of 4.4e-16
through the component Heston-Nandi bootstrap before that family was retired. Design: equities
calibrate **to the chain**, quoting premiums, because a listed price is a print while its implied
vol is a convention; the target family is `LogVar2FJModelPrices`, since equity autocalls run 3–5Y
and a multi-year ATM term structure is what one flat variance level cannot hold. V1 is **indices
only** — an American single-name chain refuses by name — and discrete cash dividends on single names
are a declared modelling gap. The emitter must declare which curve feeds the carry, or the calibration's
forward disagrees with the pricer's.

**The MC's numeraire bias is the curve's tenor grid, not discretisation** (refining 10-daily to
daily does not move it): the first node is 1Y while `reduce_deflate` asks for a ten-day rate, and
adding 1D/1M/3M/6M collapses it from −1.6e-2 to −1.1e-3. A fixture lesson every risk-neutral
calibration inherits.

**The Heston-Nandi stack is retired** (2026-09-06) and what it measured is in the Built entry
above. Two paragraphs are kept because they are the MODEL CLASS's and not that family's: a deal's
`Steps_Per_Year` agreeing with the factor's calibrated clock by convention rather than by a check
still costs a rescaled variance horizon — the emitted block states the clock it was fitted on and
nothing refuses a deal that declares another — and the coarse-grid scenario walk between exposure
dates still has no accuracy gate against a daily-grid witness on any surviving family.

!!! warning "The cleanup punchlist is stale by design"
    A separate, older sweep list exists outside the docs. Much of its target code was deleted with
    the RL stack and several entries are already done. **Verify any entry against the tree before
    acting on it** — an item naming a file that no longer exists is the normal case.

## A note on what this list is for

Most entries here were found by auditing work that already had passing tests. The recurring failure
was a gate that exercised **one point of a parameter**: only a bought deal, only the default
monitoring frequency, only one netting set, only the default valuation option.

The second is a failure of review rather than of gating: a unification absorbs N call sites into one
seam and leaves their siblings outside it, all the new testing points at the seam, and the sibling
rots holding the only copy of a formula that used to be read beside its twin. Three entries above
are that mechanism at three different adopters;
[Conventions](conventions.md#unification-siblings) states the obligation.

So when picking something up: a mutant that survives your gate means the **fixture** is wrong, not
that the code is right. Vary the parameter the defect would live in, and check the mutant dies
before believing the test.
