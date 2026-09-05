# Developer Roadmap

The [user-facing roadmap](../index.md#roadmap) lists what the library should be able to *price*.
This is the status board for what the codebase owes itself: open defects, decisions still open,
designed-not-built work, and what is built with the gate that holds it.

Two rules from [Conventions](conventions.md) apply throughout: **no abstraction ahead of a second
caller** (several items below are deliberately not started), and **look before you write**.

## Known defects

### Open

- **`hn_component_stride_invert`'s deep-tail bracket** — a target the strip cannot represent (past ~mean−5 sd the quadrature's error exceeds the probability asked; past mean−10 sd it loses monotonicity) has no bracket, the widening loop runs out, and the returned "root" is |x| of 10³–10⁵ — booked by `stride_advance` at ~1e-10 weight per path (396 paths below −100 at 2¹⁷; identical before and after the batched layer, so pre-existing). The convergence refusal now exempts these paths by the `beyond` mark, so they return as they always did rather than killing the valuation. The fix mirrors `stride_cdf`'s saturation — invert to the support edge, never beyond — and re-marks a default path, so it waits for the word. *Measured:* min draw −211,472 on both trees; zero open-and-never-widened stalls across 26 operating bands.
- **`pv_partial_barrier_option` settlement completeness** — the rebate leg's per-decision `cash_events` are declared and audited off-gate (booked rows {1, 2}, declared {0, 1, 2}, support exact over 1024 paths) but no shipped gate forces completeness, for want of a collateralised partial-barrier document. The safety half is gated (`test_a_rebated_knock_out_registers_the_rebate_it_books_row_by_row`).
- **`pricing.stochastic_boundary_correction` (`gates/boundary_bandwidth_plateau.py`)** — The bandwidth plateau holds and is carried, not closed: the 32768-path operating point became runnable when the Sobol chunking closed (2026-09-03) and the re-read there is pending. The declared `Boundary_AAD_Bandwidth` default 0.01 sits one rung inside the plateau's lower edge. The suppression seam is that same field at 1e-12, bit-identical to deleting the correction. *Measured:* At 16384 and 20480 paths the estimate holds over 0.005–0.08: seed-mean correction spread 2.41% (discrete barrier) / 3.87% (HN), reported CVA delta 0.60% / 0.24%, against seed floors 13.69% / 28.52%. No single seed sees it — per-seed spreads 12.7–23.1%, seeds disagreeing on the drift's sign, and the seed floor bounds part of the noise only because the Sobol stream is not derived from `Random_Seed`. Lower edge at 0.0025: 9–20% low, per-rung seed spread to 101% (kernel starvation). At 2048 paths the correction falls monotonically 23.76% and nothing holds still. Acceptance names 32768. Re-baselined onto the declared grid: the HN barrier gate is 1.18% against a 6.19% suppression mutant, and the discrete-barrier profile gate is a bit-exact rebate ledger.
- **`tests/test_boundary_scoping_dominance.py`** — The correction is mutation-gated; its *scoping* is not — a mis-scoping mutant (set-level `portfolio_delta`) has no public seam, since every registration names its class directly. Said in the gate's own docstring. *Measured (re-recorded 2026-09-03, after the ledger fix):* the fixture is authored because correction/smooth is 3.80 on a down-and-out digital struck at ~zero. At 16384 paths the live lane agrees with its CRN oracle at **2.74%** (flatness 2.83%; the suppressed half bit-identical across the fix, so the whole move was the correction), tolerance 0.10; the bandwidth-suppression mutant reads 366.61%, and the same mutant on the old two-set fixture survives at 0.23%.
- **`NettingCollateralSet` backward, recompute node off** — One gradient entry takes two distinct float64 values from bit-identical inputs — a nondeterministic GPU reduction, not a graph defect. It bounds how tightly any collateralised sensitivity gate can be pinned. Nothing done.
- **A collateralised set reading a knock-out rebate's 14 declared settlement dates** — the shape no gate exercises, left from the `add_grid_dates` closure.
- **`pricing` (TARF block)** — The target pin fires on 27–61% of paths, 27% short uncorrected, and is gated structurally with no tolerance asserted because nothing resolves it better. Exact behind `Branch_And_Weight: 'Yes'`; the crisp default keeps the declared blindness. *Measured:* Estimator 13% bandwidth spread, oracle 8.9% flatness — neither better than ~10%. Do not tune on the oracle: it cannot see it either.
- **`pv_MC_AutoCallSwap`** — The averaging branch cannot carry the termination latch — its termination is a smoothed per-inner-path weight with no crisp per-scenario decision. A lagging-payment schedule (coupon paying after its fixing) would have its pending window zeroed by the carry — the no-averaging arm's reach half of that closed 2026-09-04 (Closed, below), `pending` itself still unused here. No fixture reaches the averaging case; the latch marker (the fixing index that killed the path) is the hook an exemption keys on.
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
  2,048 outer reads 11.8% short of a ladder flat to 2.2%, where its 1.2% at 256 sat on a ladder
  12% non-flat (2026-09-04 night, the put leg's flux analytic by then).
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
4. **Component-HN `Quote_Sensitivity`.** Still refused, but the blocker has moved: `∂r/∂θ` is
   built — the outer search's own Jacobian, the inner `brentq` differentiated by one Newton step at
   its root. What is not built is `∂r/∂q`, the rule joining them, and a stationarity check for a
   search that can legitimately stop on the divergence wall rather than at `J'r = 0`.
5. **Component-HN positivity.** `q_{t+1} ≥ ω_t + ρq_t − φ − φγ₂²h_t` has no sign for free once
   `φ > 0`. The simulator floors (`utils.HN_COMPONENT_VARIANCE_FLOOR`, active on 2 of 8192 inner
   paths over 248 steps) while the closed form integrates the unfloored law. Floor, a bound on
   `φγ₂²`, or a different long-run innovation.
6. **Which state an OSS row inherits (F4).** The fix is shallow — the kit's day counter starts at
   the row's own trading-day offset and the state comes off the outer path — but it is a decision,
   and should be taken for the plain and component families at once.
7. **Two rates-emitter design questions**, recorded rather than decided: an OIS block is ~14 MB
    live (~26,000 authored floats on a 30Y strip, bounded by `CurveScreen.maximum_fixings`) —
    accept it through `/book/market` or build a term-authored OIS variant; and neither side rolls a
    business day (a 2Y USD OIS pays on a Saturday) — one convention on both sides, gated.
8. **The α-seed's worst benchmark.** The honesty reprice reads −6.25% against the retired seed's
    −4.64% while its rms improved 2.71% → 2.39% and the outside-3% count fell 10 → 3. The max is one
    order statistic, anti-correlated with the fit on this flat-quoted cube; gated rather than
    absorbed, owner's eye wanted.
9. **PFE vs CVA measure policy.** CVA is a Q-expectation wanting the market-calibrated outer; PFE
    is a P-quantile wanting a historically-estimated one, with the pricing kit staying
    market-implied. One run reports EE and PFE off one outer measure, so a book wanting each metric
    in its own measure runs twice under two `Model Configuration`s. Desk policy to state.
10. **`get_implied_correlation`'s two single-caller wrappers.** They would stop two callers building
    type-prefixed correlation-name tuples, but they brush the no-abstraction-ahead-of-a-second-caller
    rule. Held until a third correlation pair appears or the rule is judged to outrank it; no gate
    covers tuple literals.
11. **Flagged, not authorised.** `runtime` carries free functions over the hedge bundle in two
    clusters (Objective, Accounting) with `_UTILITY_OBJECTS` duplicated — the shape
    [Conventions](conventions.md) calls a class waiting to happen. `DealStructure`'s recursions are
    the same category.
12. **`Boundary_AAD_Window_Touch`'s magnitude.** The switch decides the sign; `add_grid_dates`
    landed 2026-09-03, so the enriched fixture and the re-measurement are now possible.
13. **COS in the stride.** Measured, not shipped: 256 terms (not 64–128) beat the 512-node strip
    by 400×, worth 2× on the stride path — at the cost of a second quadrature family through the
    tilts, partial moments and saturation, on a default-off path already ruled not a speed lever.
14. **The stride's deep-tail saturation.** Invert to the support edge rather than past it (the
    defect row above); exact fix known, re-marks a default path.
15. **The `Branch_And_Weight` default.** Prerequisites now in hand except one: the averaging arms
    refuse under the switch, so a blanket flip needs an averaging-falls-back-to-crisp rule first;
    the HN arm's cost is measured (the stride layer, above). Values re-mark within their own MC
    noise at 12–23× less variance; the greeks are the prize.

16. **A floored pillar's place in the component objective.** `atm_constraint_weight` 1e4 was set
    so a one-basis-point ATM miss outweighs the whole smile residual on an FX surface that never
    floors; a chain that floors turns it into a wall the search climbs by wrecking `H0`. Refuse a
    ladder the floor binds on (the emitter's check, before any fit), fit with the floored pillar
    dropped, or weight the shortfall at the smile's own scale.

## Designed, not built

**LogVar2FJ beyond phase 1** (designed 2026-09-04, phase 1 built 2026-09-05; see Built and
`logvar2fj_spec.md`). The **calibrator** (`bootstrappers.LogVar2FJModelParameters`, spec 5) and
with it the curve's mapping to a real market forward-variance strip: phase 1 authors `L_Curve` by
hand and `utils.lv_curve_from_forward_variance` is the mapping alone, which on the campaign's own
CJOW surface leaves **3.21 vol points** RMSE against spec 8's expected 0.2 - and refuses outright
at the spec's default `lambda = 1.5`, whose jump variance 2.46e-2 EXCEEDS that surface's 3-month
forward variance, so `xi_diff <= 0` and the curve is not a number. Stage 0 fixing `lambda` for the
surface is the missing half. **TARF, accumulator and discrete barrier** by the same `(m, s)`
substitution the autocall's arm now makes - each is the same GBM branch with the interval's law
swapped, and a daily-monitored barrier is a block of one internal step. The **density recursion**
(phase 2): one FFT convolution per monitored date against the block Gaussian, as an alternative
inner estimator. The **xVA outer generator carrying `(S, l, s)`** (phase 3), which retires the
per-row re-seed phase 1 declares - the kit seeds `l = L(t_row)`, `s = 0` at every MTM row, of the
same class as the daily kits' own re-seed. Measured 2026-09-05 (`artifacts/logvar2fj/harness_cjow.py`,
`probe_rho_prime.py`): at the spec's P-sized defaults the 1y-into-1y forward slope read 5.3 vol
points against CJOW's 12.0 whether the state was re-seeded or carried, and the SPOT 1y slope
was half CJOW's too - the deficit was the (rho, sigma) sizing, not the re-seed. At the Q-sized
defaults (rho_s -0.75, sigma_s 2.4, rho_l -0.4, sigma_l 1.0, lambda 0.21) the spot 90-110 slope
ratio reads 0.57 / 0.82 / 0.86 / 0.92 at 3m / 6m / 1y / 2y, the stickiness ratio 0.96 against
CJOW's 1.01, the 1y skew -1.47 against -1.48 (KS 0.011-0.026), the autocall -34.07 against
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
carry. The **reciprocal axis** (`HN_Invert`'s analogue): a mean
shift on the three shocks and a tilt of the jump-size law, still Gaussian, so the FX arm can price
a base-currency underlying. And **G9's delta convergence on a booked deal**: phase 0 resolves the
internal step on its own autocall's coupon leg, but the engine's own ladder in the internal step
has not been taken, and it is what licenses a weekly step for xVA - where it is not optional,
because a daily walk's tape at 2,048 x 2,048 x 509 does not fit a 24 GiB card in either direction
(the draws alone are 3 x 7.95 GiB, and `Recompute_Inner_MC` replays one block's graph, not one
step's).

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

**Honest HN CVA, steps (2) and (3).** A CVA on HN-priced TARFs and autocalls wants the outer
simulation under the same HN family, because the deal's path state — accrued target, the knock
latch, every fixing between reporting rows — is generated by the outer law, so a GBM outer walks the
deal into its states at GBM frequencies however correctly the pricer values each residual. Step (1),
the base-side keying, closed 2026-09-02. **(2)** is the accuracy gate for the component coarse-grid
walk (punchlist). **(3)** is F4 row-state inheritance: the kit consumes the outer path's running
`(h, q)` and the `L`-curve day offset through the `reveal_state_at` / `inner_fork_seed` pattern the
HMC fork already uses, both families at once, behind a declared switch, default off and
bit-identical — flipping it re-marks every HN exposure profile and CVA. The inheritance serves PFE
at least as much as CVA: a tail quantile lives on the high-`h` paths a re-seed flattens.

**The stride's escalation rungs**, recorded with triggers: map the carry residual through the exact
1D `h`-marginal quantiles (the `h`-floor mass from k = 21 onward is what it deletes); adaptive
striding where fixings sit within 10–15 days of a barrier; Lugannani–Rice on the cached CGF where a
consumer needs Φ past the quadrature's 1e-9. The open speed lever is located (2026-09-03): the
origin moment block is 86% of a strided TARF's cost, its 13 autograd chains depend on the ω strip
only through `A`, which is *linear* in it (`A = Σ D_t ω_t + Ã`, `D_t = B_t + C_t`), so the ω-free
half is reusable across every interval of equal length — one matvec per interval, 11 of 12 blocks
on a monthly schedule. Wants forward-mode jets or a per-step `D` strip, re-records the stride, not
built. Second lever, measured smaller: COS needs 256 terms here (not the hoped 64–128; 128 is
*worse* than the strip it would replace) and buys 2× — measured, not shipped, the call is open.

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

- **LogVar2FJ, phase 1 - the kit, the factor and the autocall's arm** (2026-09-05) - two
  mean-reverting log-variance factors and a co-jump, walked on an INTERNAL step and handed to the
  autocall as each fixing interval's own Gaussian block law. The model is free functions in
  `utils` (`lv_walk` and the OU filter both ways, `lv_counts`, `lv_cap`, the curve mapping, the
  cumulants), the factor is `LogVar2FJModelParameters` (nine leaves, three STRUCTURAL scalars -
  the counts' law is not on the tape and the cap is a guard - and an `L_Curve` whose knots are
  structural and values leaves), the kit is `pricing.LogVar2FJKit`, and `Internal_Step_Days` is a
  NUMERICAL setting beside `Steps_Per_Year`. The three kits now DECLARE what `oss_model_scalars` /
  `oss_model_kit` used to branch on the subtype string for - `param_names`, `curve_name`, `daily` -
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
  0.00%**, ladders flat to 0.07%. At the spec's revised defaults (2026-09-05: sigma_l 0.4,
  sigma_s 1.5, lambda 0.35, mu_J -0.15, beta 0.25) the crisp document's spot reads 0.53%, rates
  1.17%, the L curve 1.11%, `Mu_J` 0.28% on a ladder flat to 0.03%, gamma 0.02%. THE CVA DELTA IS THE OPEN ROW: at 2,048 outer the authored daily
  step does not fit the card at all, and at a 21-day step the uncollateralised spot delta reads
  +5.524e-5 against a ladder extrapolating to +5.55e-5 by three points or +5.82e-5 by the two
  finest - 0.5% to 5.1% - while the collateralised row runs out of memory there and is unread.
  `Recompute_Inner_MC` on and off agree BIT FOR BIT on the CVA and its spot gradient, on the limit
  document and the default one, the walk living inside `sim_spot`; the model's own leaves agree to
  1e-7, which is float32 reassociation under the node rather than a dropped cotangent.
  Phase 0's `checks.py` re-runs on the moved functions at the same numbers (60 checks, 0 failures),
  and the block-summing scan the pricer walks - `lv_walk(blocks=...)`, which never materialises
  `m_x` or `var` - matches the matmul filter to the gate's 1e-12 and takes 183 s against 244 s at
  1024 x 4096 x 1260 on CPU.

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
  on the strike — closing an 18–22% ladder miss to 0.16%. The averaging arm refuses by name.
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

`gates/impacted.py` selects the tests a change can reach: an execution-coverage map (per-test
contexts recorded at a campaign-boundary run via `pytest --cov=derivus --cov-context=test`, then
`--build-map`) joined with a static, always-fresh fixture map, because tests depend on JSON fixtures
at runtime and no import graph can see that. Selection is file-granular and fails open, loudly;
`derivus/__init__`, `utils`, `calculation` and `conftest` are whole-suite modules by construction.
The full suite still runs at campaign boundaries, and the tree must hold still while it runs: two
full runs each failed one mutation gate purely because source was edited mid-run, since
`inspect`-based gates read the file on disk at imported line numbers. Inner loop: `--dirty --run`.

## Tidy-ups

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

**Standing consequences — three re-marking events.** Every component Heston–Nandi θ\* solved before
2026-09-03 re-solves on the calibration strips: the quadrature nodes are a dyadic union grid rather
than uniform panels and `A` accumulates as a dot product, so every price moves at rounding and a
derivative-free search over 300 evaluations can amplify that into a different basin. On the
four-pillar USDZAR fixture it did not — θ\* is unmoved in all five fitted coordinates and the
bootstrapped `L` pillars move 1.5e-14 relative — but that is a fixture reading, not a guarantee.
On 2026-09-04 the recursion became one spelling — `hn_component_abc` reads the strip's last row —
so the STRIDE's documents re-mark at rounding too (the tilt and the stride strip run the same
recursion): the three stride-on documents move 1.7e-16, 1.3e-15 and 1.4e-16 relative, the
stride-off marks and the component credit MC are bit-identical, and θ\* is unmoved.
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

**The equity Heston–Nandi chain emitter is built** (`derivus_bloomberg/equity_chain.py`,
`tests/test_equity_chain.py`); the three engine findings it named are closed above. Chain discovery
goes through the package's own session seam with every response screened as untrusted evidence — a
live SPX chain measured 8,000 asked / 3,729 believed / 4,271 refused, led by stale and
no-open-interest. Pillars are matched to listed expiries one-to-one by nearest-claim-wins and a
pillar the chain cannot serve is dropped by name; before that rule the review measured 73% of an
objective landing on a single print. The undeclared-dividend carry is a median over five two-sided
parity pairs with a band refusal, weight is `vega·√OI/(1 + spread/cap)`, the distinct-contract floor
of eight is counted after snapping, and premium quotes carry the two-way. The emitted block fits
through the real component-HN bootstrap at an ATM residual of 4.4e-16. Design: equities calibrate
**to the chain**, quoting premiums, because a listed price is a print while its implied vol is a
convention; the target family is component HN, since equity autocalls run 3–5Y and a multi-year ATM
term structure is what one ω cannot hold. V1 is **indices only** — an American single-name chain
refuses by name — and discrete cash dividends on single names are a declared modelling gap the daily
recursion does not carry. The emitter must declare which curve feeds the carry, or the calibration's
forward disagrees with the pricer's.

**The MC's numeraire bias is the curve's tenor grid, not discretisation** (refining 10-daily to
daily does not move it): the first node is 1Y while `reduce_deflate` asks for a ten-day rate, and
adding 1D/1M/3M/6M collapses it from −1.6e-2 to −1.1e-3. A fixture lesson every risk-neutral
calibration inherits.

**Component Heston–Nandi (CJOW) is built end to end**, as a strict extension of the plain family:
`utils.hn_component_*`, `HestonNandiComponentModelParameters` on both sides (the price factor
carries an **L curve** whose values are `bind='value'` leaves),
`HestonNandiComponentImpliedSpotModel`, and one kit in `pricing.py` that all four OSS pricers walk,
so a third GARCH family is a class and a dict row rather than a fifth branch in four pricers. The
long-run intercept is a curve — `ω_t = L_{t+1} − ρL_t`, anchored `q_0 = L(0)` so `E_0[q_t] = L_t`
exactly — fitted by an inner triangular bootstrap with the skew globals concentrated over it; the
construction, its two pins and its negative-omega guard are in
[Market Prices](market_prices.md#hestonnandi-component). The gate spine is the nesting identity
(φ = 0, flat `L`: the component closed form *is* `hn_call` at 1.5e-13, and the sub-step walks the
plain path on bitwise-identical draws) — `tests/test_hn_component.py`. The calibration runs **one
backward recursion for the bound strip plus one per quadrature bound an evaluation widens to** (two
on the four-pillar ladder): `B` and `C` never read the `ω` strip and are
time-homogeneous, `A` is affine in it, so one pass at the longest maturity carries every maturity,
every `L` curve and every carry (`utils.hn_component_abc_strip` held by
`bootstrappers.ComponentStrips`), and the quadrature grid is a nested dyadic union
(`utils.gauss_legendre_dyadic`) a narrower bound reads a prefix of. **0.176 s an outer evaluation
against 2.21 s** puts the declared 300-evaluation cap at 53 s against 662 s rather than 24 minutes;
the fit still reports itself CAPPED rather than claiming a tolerance it did not reach, and what
full convergence buys is now measured — 1,246 evaluations and 243 s for a wing residual 22% better,
in a different basin. The per-pillar bound row closes with it, subsumed: a bound off the strips
costs one dot product, so every price still derives its own.

*What the component family still owes*, beyond the two open decisions above:

- **The fixing-jump sampler is v2, and the day-step is its oracle.** Every OSS interval walks `n_sub`
  daily sub-steps because the recursion is calibrated per trading day — exact, and most of the
  pricer's cost. A sampler drawing the interval's aggregate return and terminal `(h, q)` from their
  joint law has to reproduce the day-stepped path's distribution, which already exists and is
  already gated. Nothing about it is designed yet.
- **The OSS row re-seeds at day zero**, and the L curve makes that a bigger approximation. The plain
  model's limitation F4 — `h` re-seeds to `H0` at every MTM row — carries over, and the intercept
  strip also restarts at `ω_0`, so a back row prices under the *front* of the L curve. On a term
  structure moving 2 vol points over six months that is a real level error on the back rows of an
  exposure profile. The fix is shallow; which state a row inherits is the open decision above.
- **The coarse-grid walk has no accuracy gate, and its plain sibling has one.**
  `utils.hn_component_correlated_substeps` is what the scenario process walks between exposure
  dates, and nothing measures its distribution against the daily-grid witness.
  `gates/hn_pfe_stepping.py` is that measurement for the plain and GARCH(1,1)-t siblings, and it
  caught exactly the two ingredients this function reimplements: the fractional remainder (a
  `round(f)` truncation cost up to 13% of interval variance) and the forwarded mean that sets the
  correlation weights. The component version forwards a pair and slices a per-sub-step ω strip, so it
  has strictly more to get wrong. Until the gate exists the component exposure profile is gated for
  shape only — rows, dispersion, a finite CVA.

**`Steps_Per_Year` (plain limitation F2) is amplified on this family.** A deal's valuation option
agreeing with the factor's calibrated clock by convention rather than by a check costs the plain
model one rescaled variance horizon; here it rescales two things at once, because the `L` knots are
in years while `ω_t` is a per-step difference — a mismatch moves the number of steps per knot
interval *and* `ρⁿ` over that interval, so one silent disagreement gives a level error and a
persistence error. The emitted block states the clock it was fitted on; nothing refuses a deal that
declares another.

**What remains narrow on the HN stack**: batched-carry `hn_call` (stochastic-rate CVA raises loud
today rather than mispricing silently, and the already-hit leg raises the same refusal — the second
caller that would pay for the fix), and the `Steps_Per_Year` check. The `phi_max` scan is **CLOSED
(2026-09-03)**: the doubling ladder runs as one batched recursion pass — bit-identical as measured
(elementwise on CUDA; on CPU a different complex kernel lands 3–4 rungs in 20 within 1–2 ulp, the
bound surviving because it is a threshold on a power-of-two ladder with 15–42 units of margin to
`HN_STRIDE_PHI_MIN_DECAY`) — verified on 300 pinned keys both devices plus a four-pillar fit
landing the same parameters to the last digit. The scan is now 0.08 / 0.26 / 0.98 s at 21 / 63 /
252 steps against 0.15 / 0.22 / 0.87 s pinned — **31–38% of a price, was 75–82%** — so every
calibration in the stack roughly halves. `hit_value` staying GBM
under HN is closed: both vanillas take the declared model's closed form, worth 15.8% per unit and
25.3% EPE / 26.9% PFE95 on the profile. The Malz surface lookup is gated again by
`test_the_hn_ladder_is_ten_vega_weighted_points_on_the_surfaces_own_strikes`, which puts every strike
the FX calibration emits back through `pricing.calc_moneyness` and demands the built surface answer
the vol the block carries to 1e-9 relative on all ten points across both expiries.

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
