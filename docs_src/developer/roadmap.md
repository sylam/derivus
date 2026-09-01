# Developer Roadmap

The [user-facing roadmap](../index.md#roadmap) lists what the library should be able to *price*.
This one lists what the codebase owes itself: work that is designed, deferred or known-broken, with
enough of the reasoning to judge whether to pick it up.

Two rules apply to everything here, from [Conventions](conventions.md):

- **No abstraction ahead of a second caller.** Several items below are deliberately *not* started.
- **Look before you write.** A new helper is usually a missed search.

## Known defects

| | Where | Status |
| --- | --- | --- |
| Collateralised boundary counterfactual does not resolve | autocall registration x collateral chain | **CLOSED.** One mechanism, found by dumping the engine's counterfactual net rows and diffing them row-by-row against a chain replica on identical branch inputs: the per-decision ledger flipped only the decision's OWN payment, so a trigger forced ON left every later coupon's booked cash sitting in the margin period's `C_ts_te` windows, and forced OFF never paid the coupon the off-world reaches at the path's first later firing. `LatchedBoundarySet` now derives each decision's full ledger reach from its declared per-event `cash_events` facts with the latch algebra the set already owns; `net_from_gross` sums a list of ledger rows. Two-coupon cut: +6.5% mean excess over four seeds closes to +0.4% (inside estimator noise, ±1.4% each seed); the shipped six-coupon gate reads 0.14% against its CRN ladder, own-row-only mutant +7.73%. The previously suspected second channel ("+5.9% in a cash-free world") was the discriminator's own artifact: its stub emptied the forward ledger via `cash_settle` while the counterfactual still flipped `settle_map` cash - an inconsistent world, not an engine term. Gate: `test_autocall_json.py::test_a_collateralised_cva_delta_carries_the_settled_coupon`, xfail marker off. |
| Extendable forward: boundary flux, mirror booking | `pv_MC_ExtendableForward` | The deal ships forward-value complete (Black-oracle gates, both styles). The boundary registration is **DONE**: reconstructed decisions register a `LatchedBoundarySet` whose alive branch is the facts-only world and whose dead branch is the survived-weighted `pending` head, both derived from the pricer's own `value = fixed + state * live` split - the `EXTENDABLE_LATCH` organ reads reconstruction at 5.5e-8 relative, and the CVA delta closes from -3.17% (registration suppressed) to -0.07% against its CRN ladder (`test_the_cva_delta_carries_the_extension_flux`). REMAINING declared limits: (1) the settled-cash channel under a CSA is not registered, BY MEASURED RULING: across four amplifying documents (ITM extension, vol 0.30 / 20-day margin period, a one-fixing zero-lag tail that makes the flipped object nearly pure cash) the residual reads +0.25% / -0.02% / +0.26% / +0.03% against CRN ladders that resolve no finer - the flipped payments' exposure rides the VALUE side (scored exactly by `pending`) until settlement, then one hazard-weighted margin window; the pay-on-surviving `cash_alive` design is recorded in `test_a_collateralised_cva_delta_carries_the_surviving_cash` and waits for a document that can falsify it. (2) The exercise right rides the REPORTED book (`forward_sign` orients payoff and rule together), so the mirror booking — the client side of a bank-exercisable deal, where the exerciser optimises AGAINST the reporter — is unrepresentable; needs an Exercised_By field, min instead of max in the backward pass, and flipped truncation. (3) The rolling backward pass carries a small one-signed smoothing bias (Gauss-Hermite over the relu kink, interpolated); single-decision measures inside 5e-3 of Black and the multi-decision case is bounded by the dominance gates only — the `Boundary_*` valuation options are the accuracy dials. |
| TARF target pin | `pricing` (TARF block) | Material — fires on 27–61% of paths, 27% short uncorrected — but neither the estimator (13% bandwidth spread) nor the **oracle** (8.9% flatness) resolves better than ~10%, so it is gated structurally with no tolerance asserted. Do not tune one on: the oracle cannot see it either. **RESOLVED behind `Branch_And_Weight: 'Yes'`** (Designed-not-built carries the build): the pinned payment is a closed-form conditional expectation on the fired branch — exact, not estimated — gated against an independent quadrature an order inside the old oracle floor. The crisp path (the default) keeps the structural gate and the declared ~10% blindness. |
| ~~`pv_partial_barrier_option`~~ **REVIEWED AND CLOSED** | `pricing` | Three structural defects and two formula defects, all fixed, gated by an independent bridge-corrected Monte Carlo oracle (`test_partial_barrier_json.py`, worst 0.43% over all eight direction/window configurations). Structural: (1) the suspected NaN was real - rows at/past `Barrier_Limit_Date` fed a non-positive window into `sqrt` (inf-inf in the e-terms at zero); the clamp is also the valuation there, both window types collapsing to the right limit of the same formula. (2) The outer `touched` monitoring ignored the window entirely - a crossing outside it knocked anyway, measured -87% on realised payoffs. (3) Down + end-window + continuous monitoring NEVER PRICED: a float barrier made `strike < barrier` a python bool and `torch.where` refused - silent skip, NaN mark. Formula (adjudicated by workflow against exact-BivN + 4M-path MC): the eta==0 B1 branch had an INVERTED strike-vs-barrier selection and wrong e3/e4 signs in the reflected terms (max error 8.69 on 100-spot cases, negative out-option prices) - unreachable from the FX deal's declared Barrier_Type values, fixed anyway; the suspected eta==-1 argument slip was REFUTED (bivariate-normal symmetry). ApproxBivN edge ledger, all currently unreachable from this sole consumer post-clamp (rho strictly inside (0,1)): NaN grads for rho=0 rows in mixed batches, silent zero rho-grad at exact rho=0, rho=+-1 NaN-or-wrong, P/Q=+-inf rho-grad NaN; true worst accuracy ~7e-4 not the claimed 4 decimals. STILL OPEN: (a) the `touched` latch registers no boundary set, so a CVA gradient drops the window-touch flux - the same registration family as the autocall/accumulator/extendable (the bridge-probability `touched` is already smooth, so only the closed-form/knocked branch selection needs the registration); (b) `Cash_Rebate` is entirely UNGATED - every gate here runs rebate 0 - and the knock-in's rebate carries no pre-expiry value in the closed-form rows (it enters only through the expiry pad), so an untouched KI's mark ignores its discounted rebate until the last row. |
| A sibling fallback may name a factor discovery never fetched | each deal's `calc_dependencies` | Discovery iterates `factor_fields` over the RAW field and `get_fieldname` drops blanks, so a blank reference loads no factor. A fallback is only safe if it names one something ELSE already pulled in. `Discount_Rate ← Currency` is safe — 34 sites — because `Currency` is an `FxRate` and `dependant_fields` pulls its `InterestRate` transitively. Adding a fallback to a field whose sibling has no such edge silently resolves to whatever the sibling's chain did load. The one cross-leg instance (`FXForwardDeal.Sell_Discount_Rate ← Buy_Currency`) is fixed: both rates are `default=REQUIRED` with no fallback. |
| Four tables the Workbench cannot save | `derivus_jupyter.set_value_from_widget` | `set_repr` picks a deserializer from the `obj` token, and for an untagged table falls to a hardcoded whitelist of field NAMES. `Names`, `Sampling_Data_1`, `Sampling_Data_2` and `Barrier_Dates` are outside it and raise. The token table is per-field knowledge the `Row` now carries — the fix is to render from the declaration, not to add a fifth token. **SUPERSEDED for viewing**: the web UI (`web/`) renders every table from its declared `col_names` and never re-derives a deserializer; the whitelist only bites the Jupyter app's WRITE path, which the web UI's edit slice will replace rather than repair. |
| Fourteen descriptors have no widget | `stochasticprocess` (Markov / GARCH-drift / basis / quadratic-carry models), `calculation` (`Hedging_Problem` maps, `CDS_Tenors`, `Scenario_Factors`), `bootstrappers` (a quote's own `Deal`) | `Transition_Matrix` is N×N, `Sigma_By_State` is length-N, `States` is a list of per-regime dicts, `Tradable_Instruments` is a deal map keyed by `Object` then by `Reference` — every one of those shapes is an OUTPUT, and `Table` declares fixed columns while `Container` declares fixed named children. `define_input` reads `element['col_names']` / `element['sub_fields']` unchecked, so the Workbench raises the moment it renders any process in the platinum world, or the hedging problem itself. Pinned by an exact-set gate that fails in both directions (the `Objective` / `Evaluator` / `Solver` blocks left the set when they declared their knobs, under the since-removed `test_hmc_declared_knobs`) — `test_the_descriptors_with_no_widget_are_exactly_these`. The fix wants a widget, not a schema change. **SUPERSEDED for viewing**: the web UI dispatches on the VALUE first and renders any shape the vocabulary cannot state as pretty-printed JSON — visible, never a raise. The Jupyter app keeps the defect and is deprecated in the web UI's favour. |
| ~~The OSS carry strip is not the interval integral~~ **CLOSED** | `pricing.forward_carry_rate` | `carry[j]` is the AVERAGE annualised rate from `t` to fixing `T_j` and `dt[j]` the length of the interval **ending** at `T_j`, so `carry * dt` was that interval's integral only on a FLAT curve — and it drove the GBM drift of the barrier's own **simulation**, so `E[S_T] ≠ F(t,T)` on any sloped curve. Measured **4.276e-02** on the sibling fixture's sloped curve, now **2.220e-16**. `forward_carry_rate` differences the cumulative integrals and all four adopters — `sim_spot_oss`, `pv_MC_Accumulator`, `pv_MC_Tarf` and `pv_MC_AutoCallSwap` (one strip, both branches) — take it in `theta` in place of the raw zero rates; the two correct sibling spellings are deleted, bit-identically (measured on the non-zero-carry fixtures). The gate that held it, `test_payoff_forward_survives_a_sloped_carry_curve`, was removed with the mock-built suite; the measurement stands, un-gated. |
| A declared default never reaches a deal's field dict, so a schema-valid block can price as zero silently | `Deal.__init__` x every pricer that reads a field by name | `Deal.__init__` takes the authored block verbatim, so a `fields` declaration's `default=` is schema-only. A pricer reading an unauthored field by name (`pv_barrier_option`'s `Barrier_Monitoring_Frequency` and `Cash_Rebate`) raises `KeyError` inside `calc_dependencies`, and the deal is SKIPPED: an ERROR log line, `Deals Skipped` incremented, the job succeeds, the deal prices at nothing - the hollow-container failure mode in loader clothing. Found by the forward-extra build; `structures.materialize` furnishes both fields explicitly as the local remedy. The general fix belongs to the loader - apply declared defaults on load, or refuse a missing read by name - and is unowned. |
| ~~A spliced container silently dropped its children~~ **CLOSED** | `config.splice_deal` | `splice_deal` gave every container an EMPTY `Children`, so a composed `StructuredDeal` candidate loaded as a hollow container and priced 0.0 with nothing said against it - on `/book/price` and `/book/solve` at once, and the first "the collar nets to zero" check run against it was VACUOUS (the container priced nothing, not a balanced structure). Found by the structures build's own agent; the lift now lives in `splice_deal` itself so every verb prices the same composed deal, and the gate (`test_a_composed_candidate_prices_its_legs_not_an_empty_container`) asserts the container equals the SUM of legs with each leg required nonzero - the hollow pass is unrepresentable. |
| Analytic barrier/option arcs no test executes — the count is stale and the census can no longer take it | `pricing` (analytic barrier/option family) | The measurement is un-repeatable as written: `gates/pricer_branch_census.py` imports `test_pricer_branch_ledger`, and the ledger went with the 2026-08-21 purge, so the census does not run at all until it is re-anchored. Its last reading was **64** branch arcs across the ten-pricer family (the [conventions](conventions.md) page records the same census at 65 and the two can no longer be reconciled) — treat the number as stale rather than current. Two of its named entries have since been struck by work elsewhere: `pv_partial_barrier_option` and `getpartialbarrierpayoff` are exercised by `test_partial_barrier_json.py` over all eight direction/window configurations (the row above), and the analytic knock-**IN** leg is reached now that `ForwardExtra`'s `reversion` leg declares an `Up_And_In` `FXBarrierOption` at continuous monitoring — `tests/test_structures.py` prices `Up_And_In`, `Down_And_In` and `Up_And_Out` through it, with `test_a_knock_in_plus_a_knock_out_is_the_vanilla` demanding a number from `if direction == BARRIER_IN:` to 1.1e-16 relative. What is still named and unstruck: `pv_american_option` (every equity fixture is European), and the knock-OUT formulas other than the Down-and-Out / Call / K > H arm every non-structure fixture takes — the LAST elif of the OUT block. `pv_MC_Tarf.bs_call_put_fwd` was dead rather than uncovered and has been **deleted**, with its ledger row. |
| ~~The autocall has never been priced at a live carry~~ **CLOSED** | `QEDI_CustomAutoCallSwap` fixtures | `gates/fixture_degeneracy.py`: **119 runs** over the barrier/option/HN/TARF/autocall modules, not one at a non-zero rate or a non-zero carry. It is the OTHER adopter of the OSS seam, and its `called`/`knocked` state is the same class of outer path-state override that `hit_value` was — so no fixture that exists could tell a missing `dt` in it from a correct one. One exposure grid at a live carry closes it. First half: base valuations at r = 5%, q = 1% on sloped and smiley surfaces, which caught the strip reading -8.27% off its oracle. Second half now CLOSED by `test_autocall_json.py`'s credit-monte-carlo gates: an exposure grid at r = 4%, q = 1% with block-splitting live and the `terminationDate` latch carried across reporting rows — exactly the state this row said no fixture could see, and it WAS wrong when first priced (the 0.8/0.8/0.8 ledger above). |
| ~~The interval vol strip is read at a moneyness the deal does not declare~~ **CLOSED**; the per-fixing **smile** read is the open question | `pricing.forward_vol_strip` | `forward_vol_strip` read every fixing at its own FORWARD moneyness, hard-coded `use_forward=True`, mirroring `pv_MC_Tarf`. Its two other adopters, unlike the TARF, have a EUROPEAN LIMIT, and they declare `use_forwards = False` — so on any surface with a **smile** the simulation priced a different law from the quote the same pricer marks its European legs with. Measured, both exact where exactness is available: a one-coupon autocall is a closed-form digital and read **0.024428368300 against the declared 0.024490310460, -0.2529%, 0 ULP either side**; a never-knocking `Down_And_Out` read **1174.80 against Black at the declared quote 1163.9626, +0.948%, 8.3 standard errors**, and in-out parity **-11.03** where it now reads **-0.19 ± 1.30**, ten seeds, on both smiley surfaces. **Closed by internal consistency**: the deal's flag is threaded into the strip, the pricer reproduces its own European quote (0 ULP on the digital, +0.016% and 0.1 se on the barrier) and every repo fixture is **0 ULP** on value, profile, CVA and gradient — they are all flat in moneyness with `r = q = 0`. The TERM-STRUCTURE half was separable and is untouched: alternating only this flag in one process on the smile-free surfaces leaves the prices agreeing to **1.1e-15** relative. **What is still open is the modelling question**, and it is not a defect: a desk quoting sticky-*forward* moneyness wants the read that was removed, and reinstating it costs the six gates the mutation matrix in the since-removed `test_vol_term_structure_strip.py` named — the declared-moneyness gate, both smiley European arms, both smiley parity arms and the smiley digital — so whoever picks the switch up re-builds them first. Whoever picks it up picks up a **switch**, not a revert: the two conventions are both defensible and only one of them can be the pricer's own quote. |
| ~~Compo is broken under plain GBM in both OSS pricers~~ **CLOSED**; the ANALYTIC consumers stay half-adjusted | `pricing.calc_vol_adjustment` and its six consumers | TWO defects, named separately. `calc_vol_adjustment` returned the python float `0.0` as its Compo `b_adj` and both OSS call sites handed it to `torch.unsqueeze` — `TypeError`, deal SKIPPED, so no compo OSS deal had ever priced. And its `s_adj` handed `calc_fx_forward` a TENOR where every other call site passes ABSOLUTE days (the function subtracts the grid itself), mis-tenoring the fx forward off t0 and shaping it `(N,N,B)` — so `pv_european_option`, the one `s_adj` consumer, could not have priced a compo either. **Both closed**: compo now simulates the PRODUCT `S*X` at the quanto treatment's own deal-level fidelity — spot scaled by the cross (`spot_scale`), fx carry added per fixing (`carry_adj`, built on the same fixing matrix as the drifts), interval strip composed with the fx expiry vol (`compo_vol`, which also deduped the expiry-vol expression) — and the fx forward is per-row diagonal at absolute days. MEASURED on the JSON contract (`test_autocall_json.py`): a one-coupon compo autocall is a closed-form digital on the converted spot and the pricer lands on it to **4.8e-16 relative, both correlation signs** — the sorted-pair `check_fx_name` flip is in the oracle, so the convention is tested, not assumed. The old gate's fixture was itself degenerate: USD-scale strike/barrier against a compo spot of `S*X` left the down-and-out born dead and the fixed pricer reading 0.0 — a compo strike is a PAYOFF-currency quantity and the fixture now authors it as one. **Still open, silent**: `pv_barrier_option`, `pv_one_touch_option` and `pv_discrete_asian_option` adjust the VOL only — no fx scale, no fx carry — so a CONTINUOUS-monitored compo barrier, a compo one-touch and a compo asian price half-adjusted numbers without raising; and `pv_european_option`'s repair is untested (no compo european fixture exists). The compo SMILE coordinate is also undeclared — every fixture is flat — same class as the sticky-forward question above. |
| ~~`pv_one_touch_option`'s `Payment_Timing` chain falls through silently~~ **CLOSED** | `pricing.pv_one_touch_option` | It tested `'Expiry'` and `'Touch'` with no `else`, so a third value priced as whatever the last assignment left. Both one-touch deals now refuse at CONSTRUCTION - `reset` and `add_grid_dates` read the field before dependencies run, so the earliest seam is the honest one. Gate: `test_barrier_bridge.py::test_an_unknown_payment_timing_is_refused`. |
| ~~The autocall's settled cashflow is a valuation, booked per coupon, per unit and unsigned; the TERMINATION CARRY it exposed~~ **BOTH CLOSED** | `pv_MC_AutoCallSwap` (no-averaging loop) | Measured on a job document: three coupon dates, threshold 0.01 so the deal autocalls at the first with certainty, one coupon of 0.08 on 10 units. The ledger read **0.24 / 0.8 / 0.8** where **0.8 / 0 / 0** pays, and `Units=1`, `Units=10` and `Sell` all booked the same number. Four causes, all **closed**: `tau` is per ROW but the settle sat in the COUPON loop, so it fired once per coupon and `cash_settle` accumulated; it booked `P`, the accumulated VALUE, not the payment; `nominal` scaled the mark while `cash_settle` got `value` raw; and `terminationDate` was stamped inside `sim_spot` and never returned, so every block re-priced and re-paid the deal as though it had never fired. **The carry**: the latch is a by-product of the simulation (stamped where a fixing is observed, off the scenario's own spot), handed to the next block's theta; the accumulators run ALIVE and the latch masks the exits, `P` being homogeneous in its initial weight - bit-identical while nothing fires, and the alive pass IS the `untriggered` branch the registration needs. **The registration is one set carrying two reaches**: the fired/survived fork rides `LatchedBoundarySet`'s `own_row` and the carried latch its `gaps`/`obs_before` - the halves the measurements below suppress one at a time. MEASURED on the legacy CRN gate (1024x16x256): AAD +2.07176e-06 against CRN best +1.89664e-06 - the 87.39% Row-only disagreement falls to **8.45%** on large rungs flat to ~2%. On the JSON CVA gate (`test_autocall_json.py`, 1024x4x256): disagreement **1.68%**, and each half suppressed alone kills it at **+73.83% / -72.65%** - opposite signs, ~43x the corrected residual. Ledger and profile gates now PASS un-xfailed: 0.8/0/0 and 0.79206/0.8/0/0. **Still open**: the AVERAGING branch cannot carry the latch - its termination is a smoothed per-inner-path weight with no crisp per-scenario decision (its dead discarded latch line is deleted); and a LAGGING-payment schedule (coupon paying after its fixing) would have its pending window zeroed by the carry - no fixture reaches either, and the latch marker (the fixing index that killed the path) is the hook a pending-window exemption would key on. |
| ~~A barrier date whose coupon row is ZERO reads the previous fixing's spot, and mis-tenors the coupon after it~~ **CLOSED** | `QEDI_CustomAutoCallSwap.calc_dependencies` | Closed by owner ruling 2026-09-01 — *"why would there be a 0 coupon? that makes no sense - refuse such documents - don't overcomplicate the code"* — which takes this row's SECOND remedy and takes it UNQUALIFIED: the document is RETIRED rather than priced, and the loop is left byte for byte. On the `no_averaging` arm ANY `Autocall_Coupons` row quoted `<= 0` now raises `utils.UnpriceableSchedule`, naming the deal, the row's date, the coupon and the remedy. FATAL by `is_fatal_pricing_error` on the FRA's precedent: a named refusal swallowed into a zero mark on a job that then SUCCEEDS has said nothing at all, so it re-raises out of `add_deal_to_structure`. **THE BARRIER WAS NEVER THE WHOLE DEFECT**, which is why the check does not mention it: the un-run block has two readings and only ONE of them needs a barrier. `coupon_index` never advances, so the coupon after the row takes its interval in place of its own — **4.41%**, 0.466196 against 0.487692 at 2^19, measured against the economically identical deal with the do-nothing row deleted; and where the deal's barrier is dated ON the row, the breach reads a stale spot too — **+0.394809 against +0.317939, 24.2%**, the reading this row was pinned at. The message carries the stale-spot sentence only where a barrier sits on the row; the remedy is `delete the row` either way, dropping the barrier off it no longer buying a price. `pv_MC_AutoCallSwap`'s crisp-indicator branch now has TWO doors, both exact (an OBSERVED fixing, an unaligned block opening), and its docstring says so. Gate: `test_a_zero_coupon_row_refuses_by_name`, parametrized over both documents, each with its own priced control on the same arm. |
| ~~The FX accumulator's dead branch is zero, and a leveraged deal's pending head is not~~ **CLOSED** | `pv_MC_Accumulator` | `triggered = zeros` omitted the fixings a knocked deal accrued before the breach whose settlements had not landed - measured at 1.07% of the profile at a 45-day lag on 20-day fixings, NOT one-signed (+54.06 on a leveraged OTM head), contained to cells the latch calls dead. The claim that this was 'gated' was FALSE - the named gate never existed in the tree; the measurements were session-harness only. **Closed by `LatchedBoundarySet.pending`**: per pricer row, the survived-weighted pending payoffs, applied by `select` wherever the latch state says dead - the dead branch becomes a function of the state rather than one shared profile, which is what the roadmap's 'fourth shape' actually was. The `ACCUMULATOR_LATCH` organ states the reconstruction against the engine's own rows: residual 7.6e-8 relative (float32 roundoff) on a document whose head is 28% of the profile scale; the zero-branch mutant reads 156 against a 1.6e-3 bound. Gate: `test_fx_accumulator_json.py::test_a_knocked_deal_still_carries_its_pending_settlements`, with the CVA delta riding its CRN ladder at the ladder's own 5% resolution. The extendable forward's owed registration should reuse `pending` for its own dead tail. |
| ~~HW1F's λ and quanto legs decayed twice~~ **CLOSED** | `HullWhite1FactorInterestRateModel` | Two defects, one function, both verbatim from the original import and both invisible because `Lambda` and `Quanto_FX_Correlation` are 0.0 in every fixture. The drift legs accumulated `diff(e^{−αs}·K)` — which telescopes to `e^{−αt}K(t)` — and then decayed again at assembly: ratio-to-exact of exactly `e^{−αt}` at every node (12 digits over a 10y grid; 63% attenuation at 10y for α=0.10), where HW2F and the hazard model both decay once. And the quanto-vol curve was read through `.array.T`, handing the integrator the knot PAIRS as (tenors, values) — garbage for any real curve. The rule, for any HW-family process: the increment fed to the cumsum must carry `e^{+αs}`, never `e^{−αs}` — `hw_calc_H`/`hw_calc_IJK` already put the `+α` in the integrands and the single `e^{−αt}` lands once, at assembly. The gate that held it, `test_hw1f_lambda_and_quanto_legs_decay_once` — the suite's first fixture with BOTH knobs live, held to brute-force quadrature (oracle lower limit = the grid's own first row, since the state conditions on x(t₀)=0), each half re-broken alone turning it red — was removed with the mock-built suite: the rule stands, the brute-force check does not. The docstring's `A(t,T)` carried the OPPOSITE typo (`e^{−αT}` for the code's correct `e^{−αt}`, verified to 4.8e-15) and is fixed with it. |
| ~~A one-signed `Gamma_Star` reports a flat smile as a converged calibration~~ **CLOSED**, and the refuse-at-the-bound proposal is REFUTED | `HestonNandiModelParameters.reparam` / `bounds` | The fitted vector's fourth coordinate was `Gamma_Star/1000` bounded `[1e-3, 5.0]` — STRICTLY POSITIVE — so the model could only produce a smile whose vol FALLS with strike in the underlying's own units, and an FX pair read on the `FxRate` axis routinely wants the other sign. On the canned USDZAR surface, whose risk reversal is negative in pair terms and therefore a RISING smile once read as `FxRate.ZAR`, the fit ran to the bound (`Gamma_Star` 4999.99), killed the leverage channel (`Alpha` 0.0, `l → 0`) and produced a flat 14.11% smile against quoted wings of 13.76% and 14.95% — reporting `CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH` with no complaint. **The cheap fix — refuse a fit that lands ON the `Gamma_Star` bound with `Alpha` at the floor — is REFUTED by measurement**: at zero leverage `Gamma_Star` has no effect on the price at all, so it is legitimately UNIDENTIFIED and sits wherever the optimizer left it while the fit reprices its quotes exactly. A flat surface is not a broken calibration and refusing it would be refusing the truth. **What is built is the other proposal**: the LEVERAGE SHARE carries the sign. `Gamma_Star` cannot simply widen across zero (`Alpha = |l|ψ/Gamma_Star²` is singular there, and the singularity is real — the wings' width goes to infinity), so `x[3]` is the magnitude bounded away from zero and `x[2]` is a signed share in `[-1, 1]` whose sign is `Gamma_Star`'s, with `Alpha = |l|ψ/Gamma_Star²` and `Beta = ψ(1−|l|)`. Stationarity is still a box bound on a fitted parameter, every iterate is still feasible, and the price is continuous across `l = 0`. A cold start seeds the SIGN off the quotes (a smile rising with strike is a negative `Gamma_Star`), because the objective has a kink at zero leverage. **MEASURED, on a four-pillar USDZAR surface carrying the canned fixture's negative RR** (the two-pillar one now refuses: ten ladder rungs collapse onto four distinct contracts on it, and four do not identify five parameters — [Market Prices](market_prices.md#hestonnandi-fx)): 288 s on a quiet box and 549 s with the suite running beside it, to the same five numbers BIT FOR BIT, `Omega` 2.757e-6, `Alpha` 7.784e-8, `Beta` 1.079e-3, `Gamma_Star` **−3529.45**, `H0` 7.027e-5 — persistence 0.9708, signed share −0.9989, long-run vol 15.64%, **every one of the five strictly inside its box**. It reprices its own ten quotes to a worst point of 4.73% and a weighted residual of 6.21e-5, against 13.13% and 2.83e-4 for the same parameters with the leverage channel removed. Gate: `tests/test_service.py::test_the_hn_verb_lands_a_fitted_factor_that_reprices_its_own_quotes`, which now holds the SHAPE — negative `Gamma_Star`, interior optimum — rather than flipping the fixture's risk reversal to stay inside an admissible set. |
| A spot model cannot be keyed off the pair's BASE currency, so a USDZAR TARF rides GBM on a calibrated book | `instruments.get_spot_model_params_factor` (`instruments.py:475`, read at `:5716`) | Three keyings meet at an FX accrual leg and they do not agree. The ENGINE resolves a deal's spot-model parameters by naming convention off `Underlying_Currency`. The CALIBRATION (`fx_surface_block`) writes the pair's NON-DOMESTIC token, because an `FxRate` is priced in the domestic currency and that is the only leg of the pair the engine can simulate at all. And `furnish_accrual` forces a TARF onto the pair's BASE currency, because a target is a cap on a sum of DIFFERENCES and `1/S − 1/K` is not the reciprocal of `S − K`. So on a USD-base book a USDZAR TARF looks up `HestonNandiModelParameters.USD` while `POST /book/hn` wrote `.ZAR`, and it prices GBM however many times the pair is calibrated. EURUSD joins (its base IS the non-domestic token) and so does either side of an ACCUMULATOR, which has no target and crosses freely — `test_a_leg_quoted_under_a_model_books_into_a_book_that_marks_it` is the joining arm, priced under the model and booked with it. **Tonight's half is HONESTY**: the leg's absence note names the factor looked up, the factor the book actually carries, and the orientation that works, so a desk does not re-run a calibration that already ran (`test_the_absence_note_names_the_whole_join`). **The open half is the ENGINE's**: spot-model support where `Underlying_Currency` is the DOMESTIC side. It is not a lookup fix — a GARCH on `S` is not a GARCH on `1/S`, so it is the reciprocal-dynamics question (simulate the non-domestic rate and invert per fixing, or fit the reciprocal's own parameters and say which set a factor holds), and whoever picks it up picks up a modelling decision. |
| ~~`num_batches` buys no paths in a risk-neutral IR calibration — it buys wall clock~~ **CLOSED** | `bootstrappers.RiskNeutralInterestRateModel.calc_loss_on_ir_curve` | `shared_mem.reset` was called ONCE, before the batch loop, so it cleared `t_Buffer` once; `calc_time_grid_curve_rate` keys that memo on curve and time grid rather than on the batch, so every batch after the first re-read BATCH 0's curve and re-priced the same paths. MEASURED on the identified fixture: prices **bit-identical** at 1, 4 and 8 batches, for N× the wall clock. Found by the Schrager–Pelsser checker's harness, which needed independent batches to compute a standard error at all — so nothing in the tree had ever asked this class for more than one batch. **The fix and the fields were one decision and were taken together**: `t_Buffer` (and not `t_PreCalc`, which holds per-solve precalculation and is a function of θ rather than of the sample) is cleared at the TOP of the loop, and the two class constants are now declared as `Simulations` (8192) and `Batches` (1), so a job asks for paths in its JSON. The reading that closes it: **(2048 × 4) and (8192 × 1) are now the same estimate to ONE ULP** — `reset` draws `Simulations × Batches` Sobol points once and blocks them, so batches are a blocking of one sample rather than a second sample — while at a fixed `Simulations` the estimate moves with the batch count where it used to be bit-identical. At the declared defaults everything is bit-identical to before: the clear at the top of iteration 0 runs against a buffer `reset` has just emptied, and a full four-quote MC calibration returns the same 23 doubles to the bit. Gated by `test_batches_now_buy_paths_and_shrink_the_estimates_spread` and `test_the_monte_carlo_objective_still_solves_to_this_vector`. |
| ~~A domestic swaption is deflated domestically but simulated under the BASE measure’s drift, so the two swaption objectives disagree on a quanto’d currency~~ **CLOSED** | `bootstrappers.RiskNeutralInterestRateModel.implied_process` (the seam) × `calc_loss_on_ir_curve` (the Monte Carlo objective) × `HullWhite2FactorImpliedInterestRateModel.precalculate`’s `KtT` | The Monte Carlo objective simulated through `precalculate`, which installs the quanto correction `K` whenever the rate currency is not the base currency; the benchmark was then deflated on that same curve and compared to a market premium struck in the rate currency. `schrager_pelsser_swaption` reads `J` alone and prices the DOMESTIC swaption, which is the correct measure for a domestic payoff. MEASURED, on the identified fixture’s ZAR curve made foreign under a USD base with a 15–20% FX vol and a 0.4 FX/IR correlation, at the recorded θ\*, on the four benchmarks the gate prices: forcing the correlation to zero moved the SIMULATED premiums **+5.96% / +8.84% / +11.80% / +12.42%** at 1Y×1Y, 2Y×5Y, 3Y×3Y and 10Y×10Y, while the analytic price did not move by a bit — 10.9 to 24.4 bp of ATM normal vol, five to eleven times the worst corner of SP’s own approximation bias. (The benchmark set is part of the reading: the block’s `Instrument_Definitions` set its `TimeGrid` and therefore its Sobol sample, so a 5Y×5Y in place of the 3Y×3Y read +14.08% there and moved the untouched 10Y×10Y to +12.43%.) **THE RULING WAS TO FIX IT** (owner, 2026-08-31): *calibrate domestically, simulate globally*. Girsanov moves drifts and never quadratic variation, so σ₁, σ₂, α₁, α₂ and ρ are the same numbers under either measure — fit them where the premium lives, and let the scenario run put its own base-measure drift on the invariants it is handed. `implied_process` now builds the objective’s process on an implied object with the two FX inputs (`Quanto_FX_Volatility`, `short_rate_fx_correlation`) SUPPRESSED, which takes `precalculate` down its base-currency branch — `K ≡ 0` through the Monte Carlo loss, its Jacobian and `honesty_reprice`, all three being one process object. **THE INVARIANCE READING: the whole table above is now exactly 0.0, not float noise** — the correlation reaches nothing the objective reads, so the two runs are the same arithmetic on the same sample and the premiums agree to the last bit (0.006407689690 / 0.036378626138 / 0.027227734062 / 0.059192865961, which is the old ρ=0 column: the fix moved the quanto’d column onto the domestic one, which is the price the market quoted). The foreign world now prices **bitwise as the base-currency world does** on every route the checker measures, so MC-vs-SP there decomposes as it always did at home — numeraire error −0.35% to −1.66%, SP freezing 0.08–2.91 bp, sample noise, and no fourth term. **THE EMISSION IS UNCHANGED**: `save_params` still writes `Quanto_FX_Volatility` and `Quanto_FX_Correlation_1/2` (−0.2975394327182203 / −0.26595488985172294 on this fixture) off the un-suppressed object, and a scenario `precalculate` off that factor carries `KtT` = 1.4321e−02 / 5.7542e−03 — **the same drift the mutation reinstates in the objective, to the bit**, which states the fix as an equality rather than two inequalities: it moved, it did not shrink. Base-currency θ\* is untouched (the twin is content-identical where there is no FX factor); `test_the_monte_carlo_objective_still_solves_to_this_vector` green unchanged. Gated by `test_the_calibration_objective_is_measure_free_on_a_quanto_world`, `test_the_foreign_world_now_prices_bitwise_as_the_domestic_one_does`, `test_a_foreign_calibration_still_emits_the_quanto_factor`, `test_the_simulator_still_carries_the_quanto_drift` and `test_re_enabling_the_quanto_drift_in_the_objective_restores_the_recorded_table` (the mutation, built through a public signature and killing the first three). **Standing consequence**: every foreign-curve HW2F parameter set solved before this re-solves to a different θ\*. |
| The HN family's `Volatility` is REQUIRED but never read under `Quote_Type` Premium, and a missing reference SKIPS instead of refusing | `HestonNandiModelParameters` bootstrap | Declared `default=REQUIRED`, resolved and surface-constructed BEFORE the `Quote_Type` branch that never reads it under Premium; remove the factor (or blank the field) and the whole calibration skips — `logging.error('Unable to bootstrap … - skipping')`, no factor written, NO exception reaching the caller. Measured by the equity-chain build's end-to-end gate. Consequence: a chain-sourced premium block cannot fit unless the book already carries somebody's fitted surface — the circularity the chain-not-surface governance ruling exists to avoid. Remedy: make `Volatility` optional under Premium, and refuse rather than skip on a missing required reference (the skip-not-refuse half is the hollow-container failure mode in bootstrap clothing). |
| One `Discount_Rate` does two jobs in the HN fit, and the pricer's equity forward reads a third curve | `HestonNandiModelParameters` bootstrap × `utils.calc_eq_forward` | In the fit, `r = discount.current_value(t)` both builds `forward = spot·exp((r−q)t)` and discounts the premium; the pricer's `calc_eq_forward` reads `EquityPrice.<name>.Interest_Rate` (falling back to `Currency`). On an index with a repo/borrow spread the calibrated forward and the priced forward part company — the `Steps_Per_Year` clock mismatch one axis over, found by the equity-chain build. Whoever picks it up picks up a reference-plumbing decision: the block should declare its funding and dividend references in the fields `resolve` already reads, and the fit should build the forward the pricer builds. |
| A collateralised backward gradient is not reproducible | `NettingCollateralSet` backward, recompute node OFF | One gradient entry takes two distinct float64 values from bit-identical inputs — a nondeterministic GPU reduction, not a graph defect. It bounds how tightly any collateralised sensitivity gate can be pinned, and nothing has been done about it. |
| ~~A zero-length OSS step is not a martingale~~ **CLOSED**; the exact indicator at `times == 0` is what is left | `pricing.sim_spot_oss`, `pricing.sim_spot` (averaging) | `drift` was taken from the **unclamped** variance while `vol = sqrt(var.clamp(min=1e-4))`, and every reporting row that IS an observation date opens with `dt = 0`, so that step was priced as a σ=1% lognormal kick with no Itô correction: `E[S_j \| S_{j−1}] = S_{j−1}·exp(+5e−5)`. Both consumers now read ONE `var`, so the incoherence is unrepresentable rather than detected — measured on the monthly exposure fixture, the barrier profile moves **-0.0215%**, its CVA **-0.0240%** and every one of the 13 CVA-gradient entries by up to **7.5e-4 relative**, and **no gate in the repo could see any of it**. The same edit conditions the floor itself on `dt == 0`: it was unconditional, which was harmless only while the vol-strip defect handed every interval `sigma(T)^2`, and a correct strip lets it bind wherever `sigma_fwd < 0.01/sqrt(dt)` — 114 of 365 daily intervals on an upward 0.12→0.24 surface, **+1.584%** on a `Down_And_Out` and **-6.870%** on an `Up_And_Out` against a floor-free oracle, now +0.183% and +0.403%, measured under the since-removed `test_a_daily_monitored_barrier_is_not_priced_at_the_variance_floor`. The autocall's averaging branch carried the identical expression and is fixed identically: value **-0.0082%**, CVA **-0.0076%**, CVA-gradient sum **-4.83%**. **The attribution is exact, not inferred**: on that fixture the intervals never reach the floor, so conditioning it is bit-exact in value AND gradient, and putting the incoherent drift back on the fixed tree reproduces HEAD's profile and CVA **bitwise** on both deals — the drift is the whole of what this seam moved in value, and the strip is the whole of what it moved in gradient. The `no_averaging` branch never had a floor — it already resolves `dt <= 0` with an exact indicator. **What the `dt == 0` clamp buys is the GRADIENT**, measured by deleting it: every value stays finite, 11 of 13 CVA-gradient entries go NaN (7 of 13 on the averaging deal) and six gates die. **STILL OPEN**: that step is simulated with 1% of lognormal vol instead of being resolved by an exact indicator, which consumes a Sobol draw and so moves every barrier constant. Cannot matter for a down-and-out call; **will** matter for a digital or a cash rebate. |
| The boundary estimator has no measured bandwidth plateau | `pricing.stochastic_boundary_correction` | Local-linear weights are meant to hold the estimate still over a range of bandwidths, and that is the only acceptance criterion the docstring offers, but it has only ever been read at 512 and 1024 paths where it does not settle. The documented operating point is 32768 and nobody has run it there. (The two gates that used to pass with their subject deleted are re-baselined onto the declared grid; the HN barrier gate is now 1.18% against a 6.19% suppression mutant, and the discrete-barrier profile gate is replaced by a bit-exact rebate ledger.) |
| Boundary scoping is not mutation-gated | `tests/test_boundary_pricer_events.py` | The fix is verified by measuring the term directly against CRN, but both two-netting-set gates measure the END-TO-END gradient, where the boundary term is a small fraction — so if it breaks later the suite stays green. Isolating it needs a portfolio where the correction dominates the smooth sensitivity, which is not a portfolio anyone runs. |
| ~~The HW2F calibration prices every market premium LOGNORMAL and reads neither `Distribution_Type` nor the declared `Shift`~~ **CLOSED** | `bootstrappers.create_market_swaps` × `riskfactors.InterestYieldVol` | Found by the rates-emitter build (0de7f12). `create_market_swaps` priced with `utils.black_european_option_price` carrying only `vol_surface.BlackScholesDisplacedShiftValue` — whose property consults ONLY the undeclared `Property_Aliases` and otherwise returns 0.0, never the schema-declared `Shift` — while the DEAL path's `Factor3D.get_subtype` carries `(Distribution_Type, Shift)` into every swaption deal's `Volatility` dependency. So both the distribution AND the displacement diverged between the deal path and the calibration path: a NORMAL-vol ladder (the seeded ZAR `SASN` grid) fitted under a lognormal convention, and a block authoring `Shift: 3.0` calibrated at zero displacement, silently. **Closed by owner ruling 2026-09-01.** The premium construction reads `get_subtype()` — the DEAL path's own read, so there is ONE spelling of what a surface declares — and picks a matched `(numpy pricer, tensor twin)` pair out of `PREMIUM_CONVENTIONS`: `'Lognormal'` (the field's declared default, so every existing document is bit-identical) strikes Black at `K + shift`, `'Normal'` strikes the general-form Bachelier `A[(F−K)Φ(d) + σ_N√T·φ(d)]` off an absolute normal vol. **MEASURED, one numeric ladder read both ways on the four-quote fixture: the premiums are 11.374x / 9.843x / 9.683x / 10.926x apart** — 1/F to a part in 1e5, since the ratio of an ATM Bachelier premium to an ATM Black one is `σ√(T/2π) / (K(2Φ(σ√T/2)−1))` — which is what the defect cost on every benchmark. A Normal four-quote block now solves in 11.5 s to `||J'r||` **3.97e-7** against `||r||` **1.75e-6**, three orders inside the declared `Stationarity_Tol`, and the market side ROUND-TRIPS: a quoted σ_N through the Bachelier premium through the closed-form inversion comes back as itself to **0.0–4.4e-16** — but times **0.99965771007041049**, which is `√(T_365.25/T_curve)`, the premium's 365.25 clock against the inversion's ACT/365 one. That clock is a **standing finding, not fixed**: it is the same 7e-4 years `swaption_schedule_class` names, it is invisible on the lognormal path, and moving it re-solves every recorded θ\* in `tests/test_hw2f_analytic.py`. The displacement half is `InterestYieldVol.displacement`: declared `Shift` first, `Property_Aliases` the documented legacy behind it, a premiums file's own column under that — the two routes reaching the strike through the *same* division (`Percent(2.0).amount` **is** `2.0/100.0`) and therefore bit-identical, gated in all four corners as hex. A **zero** `Shift` is not an instruction (it is the field's own default, so an authored zero cannot shadow a live alias — which is what keeps every existing file bit-identical), and a displacement authored beside `'Normal'` from *either* spelling refuses by name. Bit-identity held to the hex digit: `MC_FOUR_THETA`, `AN_FOUR_THETA` and the 25-quote readings (‖J'r‖ 3.85e-6, ‖r‖ 2.26e-3) all unmoved. Gates: `test_a_normal_surface_calibrates_and_the_market_side_round_trips`, `test_the_two_conventions_are_two_prices_and_the_normal_one_is_the_bachelier_premium`, `test_a_normal_block_carries_the_quote_side_and_its_bachelier_derivative`, `test_a_declared_shift_reaches_the_premium_and_refuses_beside_a_normal_quote`, and the rewritten `tests/test_swaption_vol_emitter.py::test_the_engine_reads_the_declared_distribution_and_this_block_says_which`. **The declaration lives on the SURFACE, which the Bloomberg emitter does not author** — a desk pointing `Swaption_Volatility` at a lognormally-declared factor still gets a lognormal fit of normal quotes, and the block's `Quote_Source` line is where that is said. |
| ~~A zero `Market_Volatility` is a silent instruction, not a bad number~~ **CLOSED** | `bootstrappers.create_market_swaps` | `if instrument['Market_Volatility'].amount:` fell through to `vol_surface.ATM(tenor, expiry)` — a blank cell emitted as zero calibrated against whatever the book's surface held, under the name of a quote nobody gave. The emitter refuses `zero` from a ladder for exactly this; the engine could not distinguish "quoted zero" from "not quoted". **Closed by owner ruling 2026-09-01, and the fallthrough is RETIRED rather than re-plumbed** — the enumeration is what settles that: no JSON in the repository carries an `Instrument_Definitions` row at all, no test or gate authors a zero or omits the column, and the only zero in the tree was the JSON reference's own worked example (`docs_src/json/market_prices.md`, eight rows at `{".Percent": 0}` beside the UNBUILT `Generation_Parameters`), which is documentation of the convention being retired and is re-authored with it. So an authored **0.0 refuses by name** and an **absent column refuses by name** — the latter used to be a bare `KeyError` inside the loop, the hollow-container shape in quote clothing — both naming the benchmark (`Swaption_2Y_5Y: Market_Volatility is quoted ZERO …`) and both remedies (author the vol, or drop the row). Consequence recorded: `InterestYieldVol.ATM` now has no consumer in the engine, and the calibration's list of open severances drops from three to two — the surface-node-to-ATM map cannot be reached from a benchmark any more. Gate: `test_a_quoted_zero_refuses_and_so_does_an_absent_one`. |
| ~~Machine-fetched blocks have nowhere DECLARED to state provenance or their two-way~~ **CLOSED** | `HullWhite2FactorModelParameters.fields` × `InterestRateCurveParameters.Points` | Found by the rates-emitter build (`0de7f12`), whose own docstrings named the gap in three places. HW2F declared neither `Quote_Source` nor `Quote_Timestamp` (both Heston–Nandi families did); `Points` declared none of `Quoted_Bid`/`Quoted_Ask`/`Timestamp` although `schema.MARKET_QUOTE_VALUES` treats all three as the value plane the tick guard admits. The emitters wrote all of them as undeclared keys `bootstrap` reads past — the same shape as the equity chain's undeclared option-row keys. **Closed by owner ruling 2026-09-01, by DECLARING rather than by accepting the keys.** HW2F gains `Quote_Timestamp` and `Quote_Source` in the Heston–Nandi families' own shape — `Date`/`Text`, blank default, stored and reported, read by nothing in the fit, and no `bind` (a market-price family has none: `bind` is `partition_factor`'s vocabulary and `Market Prices` is partitioned by `MARKET_QUOTE_VALUES` instead). `InterestRateCurveParameters.Points` gains the value trio in `FXVolPrices`' shape — the two sides optional because a benchmark the terminal quotes no two-way for stays mid-only rather than borrowing a spread, the stamp a blank-defaulted `Date`. **The emitters' BYTES DID NOT MOVE**, which is the whole reading: `derivus_bloomberg` wrote these keys already, so both emitter suites are green unchanged (42 gates) and what changed is that a schema-driven front end can see them. The two gates that ASSERTED the gap are rewritten to assert its closure — `test_the_block_writes_only_fields_the_family_declares` now says every POINT key is declared (reading the working tree for the columns this same change lands), and `test_the_row_is_the_committed_schemas_own_declaration` says the block-level subtraction is empty. `InterestRateCurveParameters` still declares no BLOCK-level `Quote_Source`/`Quote_Timestamp`, which `ir_curve.quote_census` records: a curve block's only provenance is still the per-point `Descriptor` plus the trio above. The Heston–Nandi option-table row's undeclared keys are the equity fit-side build's own row and are NOT in this ruling. |
| ~~`.DateOffset` has two incompatible wire spellings~~ **CLOSED** | `config.CustomJsonEncoder` × `Config.parse_json` | Found by the rates-emitter build. The encoder writes `{'.DateOffset': '3M'}` and `Config.read_json` parses that string; `Config.parse_json` did `DateOffset(**dct['.DateOffset'])` and needed a kwargs dict — so a `MarketData.json` this engine WROTE could not be read back through its other decoder, and `write_marketdata_json` → `parse_json` is a round trip `derivus_bootstrap.py` actually runs. **Closed by owner ruling 2026-09-01: `parse_json` reads the string, through the SAME parse.** `Config.parse_period` is the one spelling of `'3M' → DateOffset` and both decoders call it, so a second copy cannot drift again; the kwargs dict is still accepted, because reading a spelling nobody writes any more is free and writing two of them was the defect. **THE ENUMERATION.** `parse_json`'s callers are `derivus.load_market_data`, `Context.load_config`, four sites in `derivus_bootstrap.py` (input market data, the reference old output, `MarketDataCal.json`, `--market_file`) and `experiments/calibrate_platinum.py` — every one of them a file on disk. NOTHING in the tree produces the dict form any more: `CustomJsonEncoder` is the only writer of the key and it writes the string, `write_marketdata_json` and `as_json` go through it, and both Bloomberg emitters copy its output (`ir_curve.wire_period`, `swaption_vol.instrument_row`). So the dict arm is for bytes already on disk and nothing else. MEASURED on the real ZAR strip `test_interest_rate_prices` authors: **38 `.DateOffset` sites**, every one a `pd.DateOffset` through both decoders and `==` between them, with the whole block canonically identical. Gates: `test_one_dateoffset_wire_spelling_and_both_decoders_read_it` and `test_the_kwargs_dict_still_reads_because_old_bytes_are_on_disk`. **A SEPARATE FINDING it turned up, not fixed:** the encoder builds the string by walking `DateOffset.kwds`, whose key order for a MULTI-UNIT period is a set iteration — `DateOffset(months=6, days=2)` encodes `'6M2D'` in one interpreter and `'2D6M'` in the next (4:1 over five fresh processes). Both parse back to the same offset, so nothing reads wrong; what is not byte-stable across processes is `write_marketdata_json`'s output and any hash over such a block. Every offset the emitters write is single-unit, which is why no determinism gate has met it. |
| ~~`market_edit` captures the ROOT logger, so a concurrent run's ERROR refuses an innocent tick~~ **CLOSED** | `derivus/service.py` (`market_edit`'s `CapturedErrors` around `context.bootstrap()`) | Found by spine increment 3's verifier running the suite under CPU contention. The capture is a plain `logging.Handler(ERROR)` on the root logger for the bootstrap's duration, and it captured records from ANY thread — a queued `/book/price` or CMC whose pricing logs a CRITICAL (a skipped deal, a missing factor) inside a tick's capture window turned a good tick into `written: False, refused: [the foreign run's message]`, and `Metronome.failed` counted a failed beat; on a live `--tick` desk with pricing in flight that refuses good market data silently. **Closed by owner ruling 2026-09-01 with the second remedy, `record.thread`.** The root logger is the only channel `Config.bootstrap` publishes on, so the capture has to be there; `CapturedErrors` records the constructing thread's ident and `emit` drops anything else. Single-threaded behaviour is byte-identical, because there is only one thread a record can have come from. **MEASURED, on the verifier's recipe promoted into the suite** (a real book: the one-cashflow job, the FX vol bootstrapper, a real USDZAR quote block, and 40 `NettingCollateralSet` deals authored as deals so the pricer's `generate` raises and `Deal.calculate` logs CRITICAL; a background thread keeping `/book/price` queued behind the one worker; 25 real ATM ticks): with the thread test removed, three runs wrote **9, 7 and 8** of 25 ticks — 16 to 18 innocent ticks refused, each carrying a foreign run's `Deal NCS<n> skipped` line. With it in place, **25 of 25, five runs out of five**, over ~4,100 foreign CRITICAL records in ~0.5 s, with 22–24 of the 25 ticks asserted to have carried one inside their own window so the gate cannot pass vacuously. The fix itself is a comparison rather than a race, which is where the determinism comes from. Gates: `test_a_concurrent_runs_critical_does_not_refuse_an_innocent_tick` and `test_the_capture_hears_its_own_thread_and_no_other` (a joined thread, no timing at all, and the same handler required to hear its OWN thread — a capture that heard nothing would refuse nothing, which is the opposite defect). The negative arm is unchanged and unmoved: `test_a_bootstrap_that_complains_writes_nothing` still refuses the whole write on a bootstrap error raised on the tick's own thread. The spine gate that surfaced it keeps its per-iteration drain for its own reason (a run still in flight has not finished minting nothing) and its comment now says the row is closed. |
| The `Volatility_Delta` re-strike bracket assumes a lognormal scale, so a NORMAL ladder below ~78bp cannot be re-struck | `bootstrappers.create_market_swaps` (the two `brentq` re-solve sites) | Found by the six-row closure’s own review. The implied-vol recovery brackets `brentq(…, 0.01, vol + .5)` — under `Lognormal` a 1% floor is safe; under `Normal` 0.01 IS 100 basis points of absolute rate vol, so a quoted normal vol at or below ~78bp leaves both bracket ends the same sign and the solve raises `ValueError` (measured: 145bp and 100bp bracket, 80bp and 40bp refuse) — ordinary levels for EUR or JPY, and the arm is unmeasured by any fixture (the `Volatility_Delta` + premium path under `Normal`). It FAILS LOUD rather than lying, which is why it is a row and not a patch: the remedy (a distribution-aware bracket, e.g. a fraction of the quoted vol) deserves a fixture that reaches the arm, and none exists. |
| ~~A degenerate reset window dies on a key no Row declares~~ **CLOSED** | `utils.make_float_cashflows` × `DealStructure`'s two compile guards | Found by the rates-emitter build. `cashflow['Rate_Tenor']` was read when a reset's rate start equalled its rate end; it is declared by no `Row`, written nowhere, and appeared exactly once in the engine — that read. The emitters never author such a window (`fixing < following` always); a composition-harness or hand-authored document can. **Closed by owner ruling 2026-09-01: REFUSE BY NAME, and do not derive the tenor** — a rate window is not the accrual window, the schedule states no tenor, and a value the author did not give is not the engine's to invent. `utils.UnpriceableSchedule` names the deal, the fixing, the cashflow it pays, the instant the window collapsed to, and the remedy; `make_float_cashflows` takes the deal `reference` for it, at all three call sites. **AND IT IS FATAL**, which is the half that makes it a refusal at all: `is_fatal_pricing_error` gains the class and is now read by FOUR guards rather than two — `Deal.calculate`, `Deal.build_features`, and both `DealStructure` compile guards, which is where an authored schedule is first touched. **MEASURED on one pair of documents differing only in the reset's rate end date.** Before: `ERROR:FLT-DEGENERATE:CFFloatingInterestListDeal ('Rate_Tenor',) - Skipped`, `Stats: {'Deals Skipped': 1}`, root mtm **0.0**, and **the job succeeded** — the hollow-container failure mode in loader clothing, on a leg whose healthy twin prices **4948.879641** on the same market data. After: the named refusal, out of `run_job`, with no table returned. Gates: `test_a_degenerate_reset_window_refuses_by_name_and_the_run_fails_loud` (asserts all three things it must not be — not a `KeyError`, not a skipped deal, not a zero mark) and `test_the_named_refusal_is_fatal_at_the_compile_guard_too` (the predicate at both ends, and a bare `KeyError` still taking the canonical skip, because a portfolio of thousands must still survive one deal it cannot bind). |

## Designed, not built

**The COMPOSITION HARNESS is BUILT and has been RUN on live prints (2026-09-01).**
`gates/hw2f_composition.py` is the instrument and `tests/test_hw2f_composition.py` its canned twin
— an authored four-cell world through the same pipeline functions, 14 gates, 3–6 min by load. The snapshot
is the emitters' own (`derivus_bloomberg/ir_curve.py`, `derivus_bloomberg/swaption_vol.py`): 24/24
USD SOFR OIS points, 10/14 ZAR SASW/JIBAR-3M, 54/63 `SASN` ATM normal vols at 112.3–134.1 bp, the
USDZAR delta surface and a 16.1227 spot.

*Half A, the domestic fit.* Both curves reprice their own quotes at the solved nodes to
**1.07e-12 bp** (USD, 24 quotes) and **1.69e-13 bp** (ZAR, 10). The 54-row ladder fits through the
declared `Analytic` default in **62–120 min by contention** (3720.6 / 4259.6 / 7229.5 s across
three runs of the one deterministic fit; the published tables stand on the 7229.5 s solve's
pickle), to an rms of **1.348 bp of normal vol** — about 1.1%
of the quoted level — worst `3M×1Y` at **+4.125 bp**; the honesty reprice puts the engine's own
Monte Carlo **−3.39%** from market at its worst benchmark (`3M×5Y`). **The convention-aware
premium is confirmed live and exactly**: inverting each quote through the closure's own Bachelier
and back returns `recovered/quoted` = **0.999657710 on every one of the 54 cells**, which is
√(365/365.25) to the digit — the whole round trip is the documented 365.25-vs-ACT_365 clock and
nothing else.

*The fit's ρ-invariance, measured on the RESIDUAL rather than on θ\*.* Differencing the
54-residual VECTOR at a common θ shows the objective is the same FUNCTION, so no solve from any
starting point can move θ\* — and it is 0.05 s a side instead of a second 62-minute solve (no
live re-solve pair was run for exactly that reason; θ\*-bit-identity under a full re-solve is
gated on the canned world). Across ρ = 0.4, ρ = 0, ρ absent and no-FX-factor-at-
all, the residual is **BIT-IDENTICAL, max |Δ| exactly 0.000e+00** — both at θ\* (‖r‖ =
9.907029301907e-04, rms 1.348 bp) and at the family's own seed (rms 22.998 bp), because an
agreement only at the solution would be a coincidence rather than an invariance. **And the
EMISSION does move, which is the seam**: ρ̄₁/ρ̄₂ = −0.397290847914 / −0.391938309549 at ρ = +0.4,
exactly ±0 at ρ = 0, and exactly negated at ρ = −0.4. θ\* bit-identity under a full re-solve is
gated separately by `test_the_fit_does_not_move_with_the_fx_inputs` on the canned world.

*Half B, the composition.* Three par forward-starting ZAR swaps (1Y×2Y, 2Y×5Y, 5Y×5Y), each
struck at the solved curve's own ATM rate to **1.3e-15** of the coupon `create_market_swaps`
wrote, each in its own `NettingCollateralSet`, 131 072 paths, `Random_Seed` 5120, three scenario
grids. Every reading is the engine's own table: the per-set `Calc_res['Value']` blocks sum back to
`Results['mtm']` at **max |diff| 0.000e+00** in all twelve runs. **The composition is ρ-INVARIANT:
+0.047% / +0.003% / +0.035%, all ≤ 0.08σ under one seed** (weekly grid) — the correlation moves
the paths and does not move the price, which is the second half of the architecture. **`K` is live
and load-bearing**: authoring the emitted `Quanto_FX_Correlation_1/2` to zero while keeping the
correlated Brownians moves identity 1 by **−4.99 / −6.58 / −9.61 percentage points (13.2σ / 19.2σ
/ 32.2σ)**, and mis-signing the FX axis moves it by **+25 770 pp (68 002σ)**. Model against market
on the three cells is **+1.354% / −1.637% / −0.973%**, which is the 1.348 bp fit rms and not a
composition reading.

*What identity 1 actually carries, and it is a finding.* At a weekly grid the identity reads
**−0.481% (−1.3σ) / −5.090% / −9.354%**, and the miss is never a MEASURE error — the live
T-forward martingale reads the drift clean at the two shorter cells (+0.27σ at 2Y×5Y) — but the
payoff-free NUMERAIRE factor carries a per-cell share of it: **94% at 1Y×2Y, 30% at 2Y×5Y, 51%
at 5Y×5Y**, the remainder distributional and step-dependent. One unit of ZAR at T is a tradable worth X₀·P(0,T), and that payoff-free identity alone reads
**−0.455% / −1.531% / −4.735%**; divide it out and the residual is **−0.026% / −3.614% /
−4.849%** — at the 1Y expiry the composition lands on the model's own domestic price to
**twenty-six thousandths of a percent**. The FX drift accumulates a discretely-rolled money market
(`GBMAssetPriceTSModelImplied.generate`, `cumsum(r_base·dt − r_rate·dt)`), so the whole reading is
a function of the scenario STEP: 5Y×5Y reads −23.9% quarterly, −14.7% monthly, −9.4% weekly
against an unchanged 0.3% standard error. **Two mechanisms, measured apart.** The simulated leg's
gap shrinks like √Δt and is the 1Y-first-node lesson at scale — the finer-tenor-grid probe read
it as the step and not the node, though that scratch reading did not survive to a log. The
STATIC leg's does the opposite: a non-simulated curve does not roll, so its scenario view is the
frozen t=0 curve at every step and the drift accumulates T·r(0,Δt) instead of −log D(0,T);
flattening the USD curve recovers **1.00 / 1.72 / 2.31 pp** at quarterly / monthly / weekly
(spike11.log, a differently-fitted world of the same shape), i.e. **that error GROWS as the grid refines**, because a
shorter step reads a shorter and lower rate. The canned twin is authored on a FLAT base curve and
short expiries for exactly this reason, and closes identity 1 at **+0.26% (0.4σ) and
+1.13% (1.7σ)**.

*Three engine gaps named by the run, none of them fixed here.* (1) `Credit_Monte_Carlo` reports
the exposure profile UNDEFLATED and applies `Deflation_Interest_Rate` only inside the CVA/FVA
scalars — there is no deflator series or deflated profile among its output keys, so a deflated
expiry-row EPE cannot be read from the reported tables; the harness gets it by booking a unit
base-currency cashflow whose row-zero mark IS D(0,T), and the remedy is to publish `Dt_T` beside
`mtm`. (2) The two halves of a quanto are authored in two different bases and nothing checks they
agree: `save_params` emits ρ̄ᵢ = corr(dW, dWᵢ) on the parameter factor while the `Correlations`
section's rows are the INDEPENDENT normals the process's own cholesky consumes, so the section
needs `a = ρ̄₁`, `b = (ρ̄₂ − ρ·ρ̄₁)/√(1−ρ²)` (which satisfies a² + b² = C² exactly);
copying ρ̄₂ in directly gives a world whose drift and covariance disagree, silently.
(3) `HullWhite2FactorImpliedInterestRateModel.precalculate` reads `self.param['Lambda_1']`
unguarded off the `Price Models` block that `calc_lifecycle.md`'s own invariant says an implied
model does not need — omitting it raises `TypeError: 'NoneType' object is not subscriptable` from
inside a precalculate, naming neither field nor factor. Smaller: `FXVolSurfaceParameters` subscripts
`point['Timestamp']` directly, so a block without one dies on a `KeyError`; and the emitted
`Quote_Source` string in the SNAPSHOT still says `create_market_swaps` "reads no
`Distribution_Type`" — captured before `6c1301d` fixed the engine and the emitter; a fresh fetch
writes the corrected string, and the 0.999657710 round trip above is the live refutation.

*A standing observation on the fit itself, and it is not a defect claim.* θ\* comes back CORNERED:
the factor correlation sits **exactly on its declared bound (+0.95)**, **six of the twenty sigma
knots sit exactly on the 1e-5 floor** (a seventh at 1.44e-5), and α₂ solves **negative**
(−0.0391, inside the declared [−0.5, 2.4] but an anti-reverting factor). A 54-quote ladder against
23 parameters is over-determined and the rms is a creditable 1.348 bp, so the objective is doing
its job — but a corner solution is a different object from an interior one for anything downstream
that differentiates it, and these parameters should not be used for anything but this test until
somebody has looked at why the ladder prefers it. The two emitter design questions are
unchanged and still recorded rather than decided: an OIS block is ~14 MB live (~26 000 authored
float items on a 30Y strip, bounded by `CurveScreen.maximum_fixings` — accept it through
`/book/market` or build a term-authored OIS variant), and neither side rolls a business day (a 2Y
USD OIS pays on a Saturday — one convention on both sides, gated). The harness authors the
`InterestYieldVol` factor declaring `Distribution_Type: 'Normal'` for itself, since the emitter
writes the block and not the surface.

**THE HONEST HN CVA — a program of three existing rows, sequenced and ratified (2026-09-01).**
A CVA on HN-priced TARFs and autocalls wants the OUTER simulation under the same HN family, and
the reason is sharper than consistency taste: the deal's path STATE — accrued target, the
knock latch, every fixing between reporting rows — is generated by the outer law, so a
`GBMAssetPriceTSModelImplied` outer walks the deal into its states at GBM frequencies (no
clustering, no persistence) and the exposure integrates over the wrong state distribution
however correctly the pricer values each residual. The intended architecture already exists
(`HestonNandiComponentImpliedSpotModel` is the outer process; one calibrated parameter set feeds
both layers), and the engine's own posture at the deal level (the `hit_value` ruling — mixed
models inside one pricing were worth 15.8%/unit) extends to the outer/inner split as the last
mixed-law surface. The sequence, each piece its own existing row: (1) the BASE-SIDE KEYING
defect above — until it closes, a USDZAR TARF cannot find its calibrated parameters in either
layer; the equity book keys cleanly and goes first. (2) The component COARSE-GRID WALK'S
ACCURACY GATE (the punchlist's owed item — the plain and GARCH siblings have
`gates/hn_pfe_stepping.py`, the component walk reimplements exactly the two ingredients that
gate caught defects in). (3) The F4 ROW-STATE INHERITANCE, now RULED wanted rather than parked:
the pricing kit consumes the outer path's running `(h, q)` and the `L`-curve day offset through
the `reveal_state_at`/`inner_fork_seed` pattern the HMC fork already uses — BOTH families at
once, behind a declared valuation switch, default off and bit-identical, because flipping it
re-marks every HN exposure profile and CVA and that re-mark is a decision the switch lets a
book take deliberately. Until (3) lands, an HN outer already buys the right state law on day
one; the conditioning arrives with the switch. THE MEASURE SPLITS BY METRIC (owner's refinement,
same date): CVA is a Q-expectation and wants the MARKET-calibrated outer above; PFE is a
P-quantile and wants a HISTORICALLY-estimated outer (`GARCHSpotModel` is that family, and
`gates/hn_pfe_stepping.py` already gates its stepping) with the pricing kit STAYING
market-implied — mixed measures are the design there, not a defect, and the handoff is exact
because the HN risk-neutralization preserves the variance path (`h_t` is measure-invariant; only
the law forward of it shifts). The F4 inheritance therefore serves PFE at least as much as CVA —
a tail quantile lives on the high-`h` paths the re-seed flattens. One run reports EE and PFE off
ONE outer measure, so a book wanting each metric in its own measure runs twice under two
`Model Configuration`s — two plans, deliberately — which is desk policy to state, not an engine
gap to close.

**The trading spine — increments 1, 2 and 3 of 7 are BUILT.** Increment 3 adds the booking verbs,
the attestation lanes and the two-dimensional quote firmness, all on the increment-1 wire formats
untouched. The vocabulary grew by three types and changed none (`run_completed`, `result_pinned`,
`quote_filed` — a fourth mouth beside the trading, custody and writer vocabularies); the verb
logic lives in `derivus_spine` over plain data and `Context` gains five thin delegators through
`derivus/spine.py`, the ONE module under `derivus/` that knows the record exists, imported lazily
and refusing by name without the `enterprise` extra. `DV_SPINE_HOME` unset is bit-identical to the
edge as it was, gated first. Lanes carry one rule — a run is recorded iff its output will be cited
by a fact — so telemetry and curiosity mint nothing and standing attests at birth (including a run
whose numbers already exist, since content addressing dedupes numbers and the lane is about
standing). `result_pinned` re-executes through an INJECTED executor at the recorded engine version,
cache-hits against a prior `run_completed`, and compares within a declared hashed tolerance policy
— the only epsilon in the package, and a class it does not name is refused rather than passed. A
quote pins its values vector and its book plan, and `/book/quote` answers on each dimension
separately with its own remedy, disjoint by measurement through the `Market Prices` partition
rather than by assumption, beside (never instead of) the desk's own `firm_seconds`. 225 gates
total, twice-run deterministic (58 new; four are the mutation pass's own killers). Increment 2 before it bought identity, attribution and key custody:
local OIDC verification (RS256/ES256 allowlist, `azp` for multi-audience tokens), capabilities as
one hashed policy document and one pure evaluator resolved by fold-at-LSN, writer enforcement that
activates by declaration and logs every denial as a fact, an UNREADABLE fold sentinel so a
doctored policy blob cannot brick the break-glass recovery, revocable break-glass, per-seat
X25519 wrapping with the subject bound into the AAD, escrow recovery gated on a shredded copy,
and a third DECLARED residual (the hub-minted seat-key bootstrap); 24 review mutants dead by named
tests. `DV_Spine` is `init | verify | checkpoint | status | enroll | grant | rewrap | name |
whoami`. Details continue on [The Spine](spine.md): `derivus_spine/` is the append-only book of
record's truth layer: a vendored RFC 8785 canonicaliser gated on the RFC's own vectors, sealed
chained log segments (AES-GCM bodies, a blinded idempotency tag in the envelope so no plaintext
hash is a dictionary oracle, `event_hash` chaining from genesis), a content-addressed blob
store with no verb for forgetting, Ed25519 checkpoints whose verifying key is published as a
genesis policy blob and resolved by fold-at-LSN so a logged rotation neither bricks history nor
retro-invalidates it, an enforced single writer, and a two-mode verifier honest about what a
keyless replica cannot assess. Import surface: stdlib + `cryptography`, held by AST and
subprocess gates; `pip install derivus[enterprise]`; `DV_Spine` is the CLI. 103 gates, every
fault injected as data on disk, eleven mutants each killed by a named test. See
[The Spine](spine.md). What remains is increments 4–7 (projections + the diary, tier policy, the
doorbell, the generated binding). The `Market Prices` partition increment 3's firmness rule waited
on LANDED (see PREPARE/EXECUTE below) and the disjointness gate now rides it directly; the book
file's rehoming as an LSN-pinned projection, and the plan compiler as a fold over fixings
supersession, are increment 4's.

**Sensitivity estimators as first-class objects.** Every Greek should carry the estimator that
produced it — a `SensitivityProfile` per pricer — so a consumer can tell a pathwise derivative from
one carrying a boundary term. Related and also unbuilt: **Hessian-vector products** instead of
materialising full Hessians.

**Exposure gamma at a KINK — the ½Ku² term (owner's construction, ratified 2026-08-30) — is
BUILT**, on the CVA-Hessian route (`Hessian: 'Yes'` on the CVA block is the switch;
`pricing.exposure_kink_term`, hooked into `cva_for_aad` beside the boundary correction under
the mirrored trapezoid; `tests/test_cva_gamma_kink.py`, 11 gates). Measured in-house on the
linear-payoff fixture (65536 paths, r 4% / q 1%, non-flat vol): pathwise gamma 0.0 →
corrected 4.2419e-04 against a CRN ladder of `grad_cva` at 4.2609e-04 (0.45% agreement, 3.09%
flatness); pathwise vanna +4.964e-03 (wrong sign AND size) → corrected −1.2844e-02 against
−1.3044e-02, with `|2·vanna − ladder|` at 96.9% pinning the doubling off one run; deep-ITM
control 1e-26 of the live entry; seed spread 0.41%/0.69% at 65536 paths. Admission verified
one order stricter than gated: sha256 over the raw buffers, CVA, profile and the whole
`grad_cva` frame byte-identical with the term on or off, and the term never BUILT under
`Hessian: 'No'` (frame-counted, not assumed). Three findings of the build's own review are
worth keeping: the ATOM refusal is re-gated on this row's own criterion — a bandwidth ladder
whose f̂(0) climbs as 1/ε (measured 8.000 across a factor-8 ladder on pinned rows against
1.03–1.06 on healthy ones, threshold 2.0) refuses by name, while an exactly-zero netted mirror
correctly contributes an exact zero rather than refusing (no v1 document reaches the atom —
collateralised sets are preempted one step earlier by the decision-product refusal, which
covers `MTABoundarySet` too, so v1 CVA gamma is for uncollateralised books of smooth products;
the conditional-p row is the route out for both); the detach on K's argument is SECOND-ORDER INERT
(every K′ term carries a factor of u — the undetached mutant is bit-identical through second
order) and is kept for tape hygiene, pinned by a probe gate asserting `requires_grad False`
and a third derivative that refuses; and the reported Hessian on a grid whose first row is t0
carries pre-existing NaN rows the new switch merely exposes. Declared limits, unfixed and
carried: `Hessian: 'Yes'` with `Gradient: 'No'` is a silent no-op (the second-order block
rides the first-order tape — should refuse by name, small); the Silverman bandwidth is
per-BATCH, so `Simulation_Batches > 1` oversmooths relative to the run's true path count.
Next consumers: the SIMM calc's dSIMM/dθ HVP; FVA's splits. The original construction and
external measurements stand below as the design record. What second-order AAD drops at a relu is `δ(V)·V_θV_θᵀ` — a density at the
boundary times an outer product, the same object the first-order boundary fix already estimates,
so exposure gamma is a first-order-difficulty problem wearing second-order clothes. The term:
with `u = V − V.detach()`, add `T = ½·K_ε(V.detach())·u²` beside `relu(V)` in the objective,
under the same detached discount × ΔPD weights, per row with its own bandwidth. `K`'s argument
is DETACHED, so `K′` never enters the tape — the construction is confined to the density's
VALUE by design. Value is an exact zero; the gradient contribution is `K·u·V_θ` with `u` an
exact IEEE zero, so first order accumulates `+0.0` bit-for-bit; the Hessian contributes
`K_ε(V̄)·V_θV_θᵀ`, whose weighted path sum is the kernel estimate of `f_V(0)·E[V_θV_θᵀ|V=0]` —
exactly the dropped term. The admission test is therefore ONE ORDER STRICTER than the boundary
correction's: `np.array_equal` at value AND at first order, term on versus off. Wanting
second-order sensitivities IS the switch, so the cost is zero otherwise. Measured externally
(JAX prototype, ATM forward-style exposure under GBM, θ = (S₀, σ), 400k paths, Silverman
bandwidth): pathwise gamma 0.0000 and vanna 0.3966 (double the truth) become 2.6415 and 0.1992
against Black 2.6521 / 0.1989; at xVA cross-sections, 1024 paths per row gives gamma
2.59 ± 0.13 and 4096 gives ± 0.07; the O(ε²) bias shows on volga, and the upgrade is reusing
`boundary_weights` (½ Σ wᵢuᵢ²) — which also inherits `BOUNDARY_MAX_AMPLIFICATION`, and that is
load-bearing: a collateralised net can sit AT the kink with an ATOM at zero (collateral matched
inside the threshold), where gamma is genuinely singular and the row must REFUSE by name when
its bandwidth ladder diverges — the margin-period windows are what normally keep `f(0)` finite.
Declared limits: the taped path only — the recompute node's `create_graph` refusal stands, so
inner-MC products' own curvature falls back to a directional bump of the corrected delta.
Gates on entry: the external table re-taken through the real objective (CRN Hessian ladder,
agreement AND flatness), the admission equality at both orders, and the mutation — term off →
gamma to zero, vanna to double, kill magnitudes in the docstring. First consumers: CVA gamma,
then the SIMM calc's dSIMM/dθ (one HVP with ∂SIMM/∂s as the cotangent). The same term belongs
at every max/min on a simulated quantity — the collateral relu, FVA's splits.

**Second-order flux at a JUMP — conditional-p, pinned to the stride (ratified 2026-08-30;
builds WITH the stride, not before).** A jump needs the density's DERIVATIVE, and the kernel
form pays for it: the `½·K′_ε(ḡ)·J̄·u²` estimator works (a digital's pathwise gamma of 0.000
becomes −1.29 ± 0.36 against −1.33) but at 27% noise where the kink term sits at 0.5%, and its
bandwidth ladder never plateaus (−2.06 / −1.16 / −1.38 / −1.46 across a factor of 8). The
answer is to stop estimating the density: the decision is a return past a level, the fired
probability `p` given the prior state is computed analytically, and the latch's two
whole-profile branches are the mixture components — `P_vib = p·fired + (1−p)·not`, spliced as
`P + (P_vib − P_vib.detach()) − (P − P.detach())` so the reported value is the crisp
estimator's bit-for-bit while EVERY derivative is the mixture's: first order
Rao–Blackwellised (lower variance than the kernel), second order analytic, no bandwidth.
Where the mixture takes a decision it REPLACES that decision's kernel-flux estimator at every
order — one estimator per registration, never both, or the flux double-counts. The cost is
variance growing like Δt^(−3/2) in the conditioning step, which decides the build order: under
GBM the fixing-interval conditional is exactly Gaussian, so GBM books get the mixture verbatim
with `p = Φ(·)` — buildable now. Under Heston–Nandi the per-DAY conditional is Gaussian but
the walk is daily and conditioning on the last day is the bad regime; conditioning at the
FIXING interval needs the k-step conditional law `exp(A_k + B_k h + C_k q)` — which is THE
STRIDE's cached Φ, verbatim (`hn_cdf_logret` is the existing half), already required to be
exactly differentiable by that design. So the stride gains a second consumer: it was a speed
lever, and it is also the bandwidth-free jump-gamma estimator for every latched decision. HN
books get this estimator the day the stride lands, as the same registration reuse with
`Φ_stride` as `p`.

**Branch and weight — the GBM half is BUILT, all four products** (`Branch_And_Weight: 'Yes'` on
`BaseValuation`, default off and byte-identical off, per the supersession ruling;
`tests/test_branch_and_weight.py`, 110 gates). What the build found: the OSS loops were ALREADY
the smooth estimator for their knock-out decisions — `oss_truncated_draw` existed with two
adopters, the TARF's moving trigger was already `K + R_{k-1}` — so the landing was the two
missing pieces: the TARF's knock-IN integrated analytically (forced by the
one-estimator-per-decision ruling — skipping the `InnerBoundarySet` without replacing its
estimator would have lost the 35.7% of delta the boundary gates measure), and the per-fixing
accrual kinks at second order through the shared kernel. Measured: the two-fixing TARF lands on
an independent Gauss–Legendre region integral at 1.1e-3 / 4.8e-5 / 3.6e-4 / 9.7e-4 relative on
value / delta / gamma / vanna (thirty configurations, references built from the DEAL); variance
ratio 11.9× at 4096 paths and 23× at 65536 (the Rao–Blackwell claim, measured);
`Greeks: 'All'` runs under the switch where it refuses without it; the target PIN is exact
(closing the Known-defects row above, behind the switch); the moving trigger's R-coupling is
demonstrated against a deliberately severed reference (25.7% apart on vanna — the engine sits
on the coupled one); `p × realised` dies at 11.1× the gate tolerance. The barrier needed NO
code — audit found no indicator remnant on its taped path (its one-scenario registration was
already inert, now gated as byte identity rather than argued). THE SWITCH IS A PURE
RE-ESTIMATION, and the one thing that made it otherwise is RETIRED: the landing also honoured a
declared `TargetAdjustment`, which repriced a 'Full Gain' deal 44% on a flag documented as
variance reduction. Under one-step survival the estimator steps to the path that pays the
remaining target exactly — that IS the convention — so the field, its machinery, its gates and
its declaration are gone, and the smooth and crisp branches now state ONE convention.

THE AUTOCALL IS THE FOURTH PRODUCT and it landed on the deferral's own named remedy. Its
no-averaging arm was already the construction for the coupon trigger; what deferred it was the
second decision per fixing — the put leg beside the trigger, a bare jump indicator reading
18.5–22.3% off its own delta ladder wherever the barrier is off the strike and exact when on it.
That leg is now one `lognormal_fired_gain` call, and THE LOOP ORDER DECIDED ITS FORM: the coupon
block advances `L` to `p·L` **and** draws `S` survival-truncated *before* the barrier block reads
it, so the analytic term is a CONDITIONAL expectation — the partial moments over
`{S ≤ min(B, K)}` divided by `p`, spelled as the weight from before `p` entered `L`. Written
without that division the same code reads 15.0% out at a 70% barrier and 61.8% out on the strike,
which is what the quadrature arbitrated. Measured: the two-coupon autocall with a jumping put
barrier lands on its own differentiable region integral at 3.6e-3 / 1.6e-3 / 1.5e-3 / 2.6e-4 on
value / delta / gamma / vanna (both rebates, references built from the DEAL), the 18–22% ladder
miss closes to 0.16% while the crisp path still reads 16.2%, `Greeks: 'All'` runs on an observed
coupon where it refuses without the switch, and the survival ledger conserves at 0.0 with 37.6%
of the weight alive. THE AVERAGING ARM REFUSES BY NAME rather than no-opping: the distribution of
a MEAN of spots is not one fixing interval's lognormal, and its termination is a smoothed
per-inner-path weight with no crisp per-scenario decision to replace. HN still refuses by name,
citing the stride. THE REVIEW OF THIS LANDING ALSO SPLIT THE SUB-CASES THE PUT LEG KEEPS AS AN
INDICATOR, which had shipped under one blanket "exact": two of the three are (an observed fixing,
and a block opening on an unaligned one — the scenario's own spot and that coupon's own observed
price fixing), but on a barrier date whose coupon row is ZERO the coupon block never runs and the
indicator is exact on a STALE spot. That is the crisp arm's defect, not the switch's — no interval
is built there, so off-is-off holds to the bit — and it is now a Known-defects row above with its
remedy and a reading gate. The design record stands below.

**The original ratification (owner's construction, 2026-08-30; HN pinned to the stride).** Three things change at an
accrual product, and the second is why the kernel route gets ugly fast: the decisions are
sequential and COUPLED (a knock at fixing k reaches every later row, and the TARF trigger moves
with the path — remaining target = T − accrued); the jump and the decision SHARE a fixing, so
the tempting shortcut "p_k × payoff on the realised path" is BIASED — the fired branch must be
the expectation of the payoff GIVEN the decision, never a probability times a sample; and the
accrual `max(S_j − K, 0)` is a kink at every fixing with the pin `min(gain, remaining)` another
at the last. The construction is the discrete-observation sibling of the Brownian-bridge
survival method (which is why the bridge `touched` is already smooth): at each fixing, given
the state one conditioning step before, `zB = (log(B_k/S_prev) − m_k)/s_k`; the fired branch
closes ANALYTICALLY — `p_k = 1 − Φ(zB)`, payoff `E[J_k(S_k) | S_k > B_k]` (a coupon exactly;
the pinned TARF payment a call spread) — and the continuing branch draws `S_k` from the
truncated law by `Φ⁻¹(U_k · Φ(zB))` with weight `(1 − p_k)`, the alive path carrying
`Π(1 − p_j)`. Every indicator is gone, the barrier enters only through `Φ` and `Φ⁻¹`, and the
estimator is UNBIASED — not a smoothing. Once the path is smooth, double backward carries every
cross-decision term itself: the per-decision counterfactual DERIVATIVE algebra retires for
these products — but only that half of the ledger, because the set-level branches are VALUES:
a knock between reporting rows branches `max(V_t, 0)` and the collateral scan (the relu-kink
nonlinearity the boundary pages already flag), so the netting set pays a collateral rescan per
branch — the `MTABoundarySet` replay/rescan callables are that machinery — with the exposure
kink term applied per branch. Measured externally (two-observation autocall against a
differentiable quadrature reference, 65k paths): hard indicators read gamma 0.0000 and get
vanna and volga with the wrong SIGN (−0.1783 / +0.2211 against +0.4915 / −1.0219); the
truncated sample lands 0.024018 / 0.2205 / −0.1902 / −1.4205 / 0.4912 / −1.0210 against
0.024036 / 0.2203 / −0.1901 / −1.4201 / 0.4915 / −1.0219. At 4096 paths (an inner-MC width),
20 seeds: gamma −1.419 ± 0.014, vanna ± 0.007, volga ± 0.005 — 1% noise, no bandwidth, no
bias, against the kernel B-term's 27% at 400k. FOR THE TARF: the continuing-branch trigger is
`K + R_{k−1}` (the target's path-dependence is just a moving barrier) and the fired payment
`E[min(g(S_k), R) | S_k > K + R]` is CLOSED FORM — so the target-pin row in Known defects
above, which neither its estimator nor its oracle resolves better than ~10%, becomes EXACT
rather than estimated; the accrual kinks take ½Ku² per fixing (a density, not its derivative);
the knock-IN stays on the bridge. THE CONDITIONING STEP DECIDES THE BUILD ORDER: second-order
variance scales like `s_k⁻³` per fixing — fine at monthly and quarterly, rough at daily, and a
daily accumulator's economically right gamma is the desk's own call-spread width anyway (a
fixed smoothing converging at the ordinary rate: what gets hedged). Under GBM the conditioning
step is the fixing interval itself (`sim_spot_oss`'s vol-strip lognormal step) — buildable
now. Under Heston–Nandi the step available today is the last DAILY Gaussian sub-step — the
`s⁻³`-at-daily regime even for monthly fixings, since the walk is daily — while the
fixing-interval law and the truncated inversion are, verbatim, THE STRIDE's cached Φ and its
survival-truncated inverse-CDF: the stride's THIRD consumer, and HN branch-and-weight lands
the day it does. SUPERSESSION IS A SWITCH, NOT AN ADDITION: the value estimator changes (same
expectation, lower variance), so it ships as a declared valuation option, default off and
bit-identical, with re-baselined gates behind the switch — or the derivative-only detach
substitution at the cost of running both. THE MEMORY WALL stands until the recompute node
grows a **`jvp` rule** — forward-over-reverse HVPs are tape-free, and `InnerMCRecompute`'s
one-function-called-twice discipline extends to forward mode; engineering rather than maths,
and now the named task under the Hessian-vector-product row above. Until then: directional CRN
bumps of the now-smooth delta, whose ladder going FLAT is this suite's own definition of a
derivative that exists.

**Calibration Jacobians — ALL FOUR increments are BUILT.** Bumping a market *quote* now flows
through the calibration rather than stopping at the calibrated factor: one `backward()` reports
`dV/dq` beside `dV/dθ`, for a zero curve solved from deposits, FRAs and swaps, for the HW2F model
parameters fitted to swaption vols, for the integrated vol curve an ATM vol column walks into, and
for the log-moneyness FX surface a broker's ATM / risk reversal / butterfly quotes convert to.
[Quote Sensitivities](quote_sensitivities.md) is the
page — the graph audit that made it possible, the quote-side overlay, the IFT contract, the
stationarity contract, the attachment, the precision seams, the validation triangles and the
non-goals. Turned on per block by the declared field `Quote_Sensitivity`; the solved numbers are
bit-identical either way.

Increment 1 shipped carrying two known traps, and both are now **unrepresentable** rather than fixed, because
`TensorSchedule.bind` gave the tensor half a birthday — see
[the schedule lifecycle](calc_lifecycle.md#the-schedule-lifecycle). `dual` and `merged` memoized
under one key and could serve each other's copy: `bind` mints the one copy and `merged` is deleted —
`dual` is the accessor, so there is no second memo to collide with. `pv_fixed_cashflows` memoized its payment tensor in
`Factor_dep`, which outlives the copy it was built from and froze the first evaluation's overlay:
it lives in the schedule's `derived`, which `bind` mints and re-mints with that copy. The two gates
that held them in place assert the design instead.

Increment 2 put the same contract around the HW2F swaption-vol calibration, where the fixed point is
the **stationarity** of a least-squares loss rather than a root: backward is Gauss–Newton at
`J'r = 0`, above a declared `Stationarity_Tol` it refuses rather than reporting a Jacobian of
nothing, and the dropped residual-curvature term is
[measured](quote_sensitivities.md#the-dropped-term) exactly rather than assumed away — on **both**
sides, which is what settles it. The block's residual is already a square, so neither dropped term is
second-order small: the Hessian-side one is half `J'J` (0.500064 measured) and the cross-side one is
half `J'(∂r/∂q)` (0.4953–0.5115 measured, cosine 1.000000). **They cancel.** Squaring a residual
row-scales `J` and `∂r/∂q` by the same diagonal and the normal equations are invariant under that, so
Gauss–Newton is the exact leading-order derivative and **no correction is owed**. Correcting one side
only is what makes a spurious factor 3/2 appear — 1.4974 measured — and that is now the gate's
own mutation. The two
answers separate only where the `O(f³)` remainder overtakes the eigenvalue it corrects — the same
directions `Jacobian_Rcond` already declines to differentiate.

It also carried a finding larger than the increment. **Bump-and-recalibrate — the reference this
workstream was briefed to validate against — is ill posed for this calibration**, and was refuted
three times: on four quotes the solution is a 19-dimensional MANIFOLD so the finite-difference `dθ/dq`
diverges as `1/h` and the re-bootstrapped CVA delta reverses sign; on a 25-quote fixture with `J` at
full column rank it fails anyway, because the optimizer stops seven and a half of the eight orders
short of stationarity and `θ*` is its stopping point rather than the argmin. One diagnosis covers all
three: the solve wanders in the directions the objective is flat in, which are exactly the ones the
pseudo-inverse declines to differentiate. The implicit-function derivative is the better-behaved
object, the value-space reference that IS well posed is a step along what the quotes identify with no
re-solve, and every refutation is pinned as a gate so nobody later tunes the derivative against an
oracle with no limit. See [the re-solve reference](quote_sensitivities.md#the-manifold-finding).

Increment 3 is **FX vol, and it needed no solver at all** — which corrects what this page said it
would need. `GBMAssetPriceTSModelParameters` does not *fit* an integrated vol curve to an ATM vol
column; it **computes** one, by a forward-variance walk that is closed form end to end, so there is
no `LeastSquaresSolve` contract in it, no implicit function theorem and no stationarity to check.
Autograd walks the expression — a torch twin spliced in for its derivative alone, so the shipped
curve is still the numpy walk's, bit for bit — and the validation triangle the two solves could not
close, one-pass `dV/dq` against `dV/dθ · J` against a central difference converging as `h²`, closes
here. Two
properties carry the increment: the map is the **identity** wherever forward variance rises (only
`σ̄` is written; the instantaneous vol is the walk's own state), so every gate runs on a fixture that
declines; and the declining-variance repair is a **kink**, where a quote's delta drops from 1 to 0
and is severed from every later expiry. See
[the closed-form map](quote_sensitivities.md#the-closed-form-map).

Increment 4 is **the delta solve**, and it retracts increment 3's last non-goal. That page said a
differentiable Malz solve was out of scope because a bisection per node would have to go on the
tape; the premise was wrong. **Differentiating the bisection is the defect, not the cost** — the
iterates are dyadic combinations of the two bracket ENDPOINTS, so a tape through the loop reports
`d(bracket)/dq` and not `d(root)/dq`: measured, the same forward number to 2.8e-17 beside a
Jacobian 0.135 out, reporting a plausible 1.000137 for a `dσ/d(ATM)` whose truth is 0.865559. So
the tape starts at the converged root and takes one Newton step off it, which is the implicit
function theorem written as an expression, and `dV/d(risk reversal)` is now an ordinary number. The
x-grid stays [pinned](market_prices.md#fxvolprices) and undifferentiated — the twin moves the vols
on frozen nodes, because a grid that followed its quotes is a recompile per tick. The chain closes
end to end: a tick in ATM / RR / BF reaches `V` through the surface *and*, where the two families
are stacked, through the integrated vol curve as well. See
[the delta solve](quote_sensitivities.md#the-delta-solve).

Its own findings are worth carrying. **Four things are discrete here**, not one — which wing a node
reads, whether its root is bracketed or flat-extrapolated, which linear segment it sits in, and
which END of the bracket a clamped node takes — so the FD ladder carries a fingerprint per node and
scores only the rungs it held still across. Three of the four are kinks, where straddling converges
to the average of two one-sided derivatives, 0.9403 between a 1.0 and a 0.8806047; **the fourth is
a JUMP**, because the two endpoints are different knots carrying different vols, and on a steep
smile a 2e-6 bump steps a node 0.1199 of vol while the first three marks hold perfectly still. That
one was found by mutating the instrument: an unmarked endpoint makes the ladder score a step
divided by 2h, 6.0e+04 at h = 1e-6 and growing as 1/h. And **a flat smile divides zero by zero in
the backward alone**: an ATM-only block mirrors one node onto both wings, each wing is a single
knot, its span is exactly zero, and the unguarded Jacobian is NaN everywhere while the written
surface is a perfectly good flat one. Guarded, it comes back as the expiry indicator.

Three smaller ends stay deliberately open and are recorded on the page's non-goals: there is no
report FORMAT for a quote delta (it lands on the leaves in `Config.quote_leaves`, in one of
[two shapes](quote_sensitivities.md#the-attachment) — increments 3 and 4 both reused the vector
one — `make_factor_index` wants a tenor grid a quote does not have, and where two families read the
same JSON number its `dV/dq` arrives as **two partials under one descriptor** that a consumer has to
sum: 2.243453e4 + 8.071709e4 on a stacked FX pair, measured — `structures.vol_risk` is the first
consumer to obey that rule, and it sums rather than reports); neither backward supports
`create_graph`, so there is no second derivative in quote space; and no vol-surface
parameterisation (SABR, SSVI) is in scope — a Malz smile is the one delta parameterisation built.

**`fields.py` is retired.** This started as 1,931 lines and three drifting stores; it ends with
`derivus/fields.py` a 22-line deprecation shim re-exporting `mapping` and `default` from
`schema.py`, kept for one release because `fields.mapping` was the documented surface and the
package is on PyPI. Every store but `System` and the create-deal menu is emitted from the per-class
declarations, `schema.py` holds the vocabulary, the emitters and the assembly, and every in-repo
consumer reads `schema.mapping`. The sections below are the seven migrations that got there, in
order.

The assembly sits at the BOTTOM of `schema.py`, after the vocabulary its declaring modules import
from it — a one-way edge with an ordering question attached, since a declaring module initialised
first would have `emit_*` read a half-initialised module and return an EMPTY store rather than
raising. A submodule import always initialises the package first and `derivus/__init__` imports
`schema` before any declaring module, and a subprocess gate holds that fixed.

**The Instrument store was the first VIEW.** Every deal type carries a per-class
`fields` list (`schema.py`), and `mapping['Instrument']`'s `types` / `sections` / `fields` are built
by `schema.emit_instrument(instruments)` at import. The hand-written dict is gone — 1,283 of
`fields.py`'s 1,931 lines. The engine never reads any of it (`construct_instrument` takes the raw
JSON), so the blast radius is the Workbench, the docs generator and the Excel add-in.

Two defect classes went with it, which is why the strict xfails guarding them are gone. A type
naming no class: `SwapBasisDeal` and `SwapCurrencyDeal` were offered under two menus with 128
descriptors between them and had no class in this repo or the one it came from, so creating one
logged and returned `{}`. And a descriptor no section reaches: fifteen of them, unrenderable and
undocumentable. Neither state is expressible when a class is the only place a field can be declared.

The schema's inheritance turned out to be composition of named field GROUPS, not the class
hierarchy: `FXAdmin` is shared by eight deals with no common base and `Admin` by all of them, so
groups are module-level `Group` constants a class lists. An MRO-based design cannot express that.

`groups` stays hand-written, but it is now ONLY the create-deal menu. It also carried the jsTree
node kind, which is not presentation at all: it says whether a deal breaks down into simpler
instruments, and only `New Structure` was a folder. So `CapDeal`, `FloorDeal`, `SwapInterestDeal`,
`SwaptionDeal` and `MtMCrossCurrencySwapDeal` were files, and the Workbench could not build any of
them with the legs their own `post_process` prices. That is `Deal.accepts_children` now, declared
on the class and gated against `post_process` in both directions. The engine never cared — it
recurses on `Children` being PRESENT, never on the type — which is exactly why nothing failed.

The menu can still drift from the classes, so a gate holds every declared type to appearing in one.
It found `EquityDeal` and `EquityOneTouchOption`, both concrete and documented, in no menu at all.

**A section owns its descriptors.** The store was keyed by field NAME across all 47 deals as they then stood (`mapping['Instrument']['types']` carries 48 today), which
admits one descriptor per name — so a field needing different valid values in two deals had to
invent a key and carry the real one as an alias, and `ALIASED_KEYS` was the running cost.
`sections[S]` is now `{json_key: descriptor}` and a container holds its children inline, so
`Payment_Timing` is `Touch`/`Expiry` on a one-touch and `End`/`Begin`/`Discounted` on a cashflow leg
with no collision. Both were always right — the JSON is per-deal, and only the flat view was
ambiguous. `Option_Payment_Timing` is renamed, the Instrument aliases are gone, and the flat
`fields` key with them.

Three more defect classes became unreachable rather than merely absent: a section naming a field
with no descriptor, a descriptor no section reaches, and one deal's field silently resolving to
another's. Their gates are deleted, not disabled.

**Authoring rules are stated, not just enforced.** `schema.validate_instrument(deal)` returns the
messages for one constructed deal. Two layers, because the rules have two shapes: `default=REQUIRED`
covers a field that must simply be there, and the declaration alone is enough; a rule spanning
several fields has no shape in common with the next one — a value being non-zero, two fields being
alternatives, one column of a table row implying another — so a class states those as code in its
own `validate()`. An audit of the 45 deal types as they then stood found 16 cross-field rules over 6 predicate forms,
which is why they are code rather than a declarative `one_of=`; 17 more key on `base_date`,
`self.options` or a resolved price factor and can never be evaluated at authoring time at all.

Nothing in the valuation path calls it and a message never stops a deal pricing — the engine still
fails where it always failed. Four rules are stated so far (`Settlement_Amount ⇒ Settlement_Date`,
`Collateral_Rate ⇒ Funding_Rate`, the inflation value-or-date pair, and a commodity average-price
swap's settlement not preceding its own sampling window); `Cash_Payoff` on the binaries became a
`default=REQUIRED` and left the code layer. `cx.validate()` now returns them alongside the factor
want-list, keyed by each deal's `Reference`.

**A price factor owns its schema too, and a TYPE owns its descriptors.** Every factor class carries
a flat `fields` list and `mapping['Factor']['types']` is `schema.emit_factor(riskfactors)` — a
per-type `{json_key: descriptor}` map, so the flat `fields` dict and the `Space`→`Surface` alias
with it are gone. A 2D `Surface` and a 3D one are both called `Surface`, which is what the JSON
always said. Two more states became unreachable: a declared type naming no class (`ConvenienceYield`
had two fields, a process-map row and no class anywhere, and `construct_factor` would have called
`None(block)`), and a factor class no schema can author.

The dead `field_desc` prose went into the declarations as each field's `description`, which is what
the generated page publishes. That needed one thing fixing first: the Workbench rebuilt the JSON key
as `description.replace(' ', '_')` at six sites, so a prose description silently wrote to a key
nobody reads — and the `Process` store's prose descriptions (`States`, `Transition_Matrix`, `Mu`, …)
were already doing exactly that. The key is now stamped from the store as `name` when descriptors
are loaded. `CommodityPrice.Forward_Rate` is newly declarable: the class hard-reads it and
`dependant_fields` builds a `ForwardRate` edge from it, and no authored block could carry it.

**`bind=` and the partition are built.** A declaration is STRUCTURAL unless it says `bind='value'`,
and `schema.partition_factor(type, block)` splits a `Price Factors` block into the half a plan pins
and the half a patch carries — 39 fields over 23 types, each declared from its consumption site. A
shape-valued field splits INSIDE itself, because a curve's knots size `all_tenors` when the factor
is constructed while its rate column is content; the structural half keeps the coordinate columns
and only the last one travels. `cx.market_patch()` emits one and `cx.patch_market()` applies it,
refusing anything structural by name. `plan_hash` is the consumer it was built for, and it hashes
the structural projection alongside the deal tree and the calculation block — see PREPARE / EXECUTE
below.

Four candidates remain declined with a citation rather than bound. `ReferencePrice.Fixing_Curve` and
`PriceIndex.Index` become reset rows in `make_energy_cashflows` / `make_index_cashflows`, so the
content is consumed building a compiled structure. The other two are value-dependent CODE PATHS,
which is the trap the partition is most exposed to:
`GBMAssetPriceTSModelParameters.get_tenor_indices` branches on the truthiness of
`Quanto_FX_Correlation`, so the leaf SET depends on a number; and `get_quanto_fx` returns `None`
when the `Quanto_FX_Volatility` array is all zeros, which decides whether the implied tensor is
published at all. `VolatilityGrid.Delta_Surface` is a third of that kind: `Factor2D.update` runs the
Malz solver on it and rebuilds `Surface` on a grid refined against the VALUES.

**`Correlation.Value` and `SurvivalProb.Recovery_Rate` are bound**, because the compile bake was a
SHAPE and not a fact about either number. `get_implied_correlation` and `get_survival_component`
(was `get_recovery_rate`) now return the factor OBJECT, `Factor_dep` carries the reference, and
`utils.implied_correlation` / `SurvivalProb.recovery_rate` do the reading at eval. The reverse-pair
flip a quanto applies is genuinely compile-time — it follows from the two currencies sorting — so it
travels as its own `Correlation_Sign` entry rather than pre-multiplied into a number.

**A process owns its parameter block, and the factor menu is the same declaration inverted.**
Every stochastic-process class carries a flat `fields` list and a `factor_types` tuple naming the
price factors it drives; `mapping['Process']['types']` and `Process_factor_map` are both
`schema.emit_process(stochasticprocess, ...)`. The flat `fields` dict and the last Process alias
(`sigma` → `Sigma`) go with it, because `Sigma` is a scalar on the OU, hazard-rate and
Clewlow–Strickland models and a term-structure curve on Hull-White. Two more names were carrying
two shapes each and only one descriptor: `Phi` is a 3×3 VAR transition matrix on
`VARMixedFactorInterestRateModel` and a scalar AR(1) coefficient on `BasisLinkedSpotModel`, which
the flat store rendered as a matrix table; and `VARMixedFactorInterestRateModel.Sigma` is a
length-3 vector rendered as the Hull-White curve widget.

The map's keys are now the factor types themselves, which is what the consumer needs: the
Workbench indexes it by the type of the factor in front of it, so a missing key takes the page
down rather than showing an empty menu. Inverting it also found four processes the engine
constructs that no menu offered — `GARCHSpotModel` (in no store at all, though calibrated,
shipped in a fixture and documented), `CSImpliedForwardPriceModel`,
`HullWhite2FactorImpliedInterestRateModel`, and `GBMAssetPriceTSModelImplied` on equity, whose own
`calc_references` handles `EquityPrice` and raises for anything but that or `FxRate`.

`BasisLinkedSpotModel.Sigma` and `SingleRegimeOU1FactorKalmanModel.Measurement_Var_Base` are newly
declarable: the first is one half of the exactly-one-of pair the class asserts on and the shipped
platinum market data carries it, the second is written by the calibration into every block it
emits.

**A calculation owns its parameters, and the audit is the point.** The three calculation classes
carry a `fields` list and a `calc_type` naming the `Object` string a job writes, and
`mapping['Calculation']['types']` is `schema.emit_calculation(calculation)`. The type is keyed by
that string rather than the class name — `Base_Revaluation` is authored as `BaseValuation` — and
a gate holds every declared type to reaching a real branch of `run_job`, parsed by AST for the
reason the `MarketPrices` gate is: a hand-kept list would be a fourth store.

`HedgeMonteCarlo` had no schema row at all, which is worse for a calculation than for a deal: the
store is indexed by the block's own `Object`, so `CalculationPage.load_items` raised KeyError and
the Workbench could not open a hedging job.

The declared-versus-read audit is most of what this bought. **`Base_Time_Grid` was declared and
never read** — `run_cmc` and `run_hedgemontecarlo` read `Time_Grid`, and so do every fixture and
the JSON reference's own example — so the Workbench's grid field wrote a key nobody reads and a
Workbench-authored run silently took the hardcoded default grid. Eight more keys were read and
never declared: `Tenor_Offset`, `Keep_Tensor`, `NoModel`, `Gradient_Variables`,
`Boundary_AAD_Bandwidth` (also on base valuation), the whole `Initial_Margin` block, and
`Hessian` / `CDS_Tenors` inside the CVA block — the last already published in the JSON reference's
example. Their descriptor defaults were sourced from the engine's own `.get` defaults — and the
audit's leftover disagreements were then RULED the other way: **the declaration is the single
source of an omitted field's default.** `schema.declared_defaults` completes the params dict at
the top of every `execute`, `run_*` inject only runtime-derived keys, and the fallbacks that could
lie are direct reads now, with an AST gate holding any survivor to equality with its declaration
(`tests/test_schema_emission.py::test_a_declared_calculation_default_is_the_default_the_engine_falls_back_to`).
Four disagreements surfaced and settled: `Random_Seed` (engine said 1, declaration says **5120**), `Dynamic_Scenario_Dates` and `Generate_Cashflows`
(engine said `'No'`, declarations say **`'Yes'`** — a results-changing event for any job omitting
them, and the flip is what exposed four defects — the never-firing `boundary_weights` guard, the
already-hit leg's model mix, two placebo boundary gates, and the zero-length OSS step still in the
table above — none of which the grid caused: `'Yes'` is **measurably the more accurate grid**,
since interpolating a spot to a barrier
observation date is a convex combination of two scenario rows and therefore carries no bridge
variance. Measured against a 2m-path paired reference, `'No'` misses **2.56%** of knock-outs and
biases the profile **+0.71% (175σ)** while `'Yes'` sits at +0.098% against its own ±0.14%; against
4m exact-law paths the crossing probability reads **−0.6 sem** under `'Yes'` and **−6.6 sem** under
`'No'`), and
`MCMC_Simulations` on base valuation, where the declaration moved to the **32768** that
`run_baseval` had always injected past the store's unread 2048. The `Hedging_Problem`
`Objective` / `Evaluator` / `Solver` blocks declare their ~40 knobs on the same terms; the
both-directions gate that held them, `test_hmc_declared_knobs` (declared-vs-read, engine side
included), went with the mock-built suite and has no replacement.

Four are declared and NOT read, and stay declared as unbuilt functionality:
`Credit_Valuation_Adjustment.Bank`, `Funding_Valuation_Adjustment.Bank` and
`Funding_Valuation_Adjustment.Stochastic_Funding` reach no read anywhere, and
`System.Exclude_Deals_With_Missing_Market_Data` likewise. Two calculation options are read and
still undeclared by decision: `DealLevel` is passed straight to `deal_level_mtm=` as a BOOL, and
every boolean-ish descriptor in the vocabulary is a `'True'`/`'False'` STRING, which is truthy in
both positions; and `LegacyFVA`, whose own comment says it is to be removed. Three keys the
shipped market data carries — `Grouping_File`, `Proxying_Rules_File`,
`Script_Base_Scenario_Multiplier` — are in no store and reach no read.

`System` stays hand-written. Its one type is a UI panel name rather than anything the JSON
dispatches on, and the class that consumes `System Parameters` is `Config` itself — the whole
configuration object — so giving it a `fields` list would make "a class that declares fields IS a
type" mean something else in that module.

**`Interpolation_factor_map` is a view too, and the restriction it carried is real.** The last
reading of it was that `Factor1D.check_interpolation` supports all four methods for every curve
factor, so there was nothing on a class to read the two rows off. That is true of the
interpolation code and beside the point. `Interpolation` is not a `Price Factors` key an author
writes at all: `construct_factor` reads it out of the `Price Factor Interpolation` section — a
`ModelParams` mapping factor type → method, exactly like `Model Configuration` — and injects it
into the block, and only for `InterestRate`, `InflationRate` and `ForwardRate`. So what the three
rows restrict is the OPT-IN, which is per-class, and the menu for a type outside it would offer a
setting the engine drops on the floor.

Those three classes therefore declare `interpolation_methods` — `InterestRate` and `InflationRate`
sharing one `INTERPOLATION_METHODS` object, `ForwardRate` declaring the narrower pair it can
honour — and the map is `schema.emit_interpolation(riskfactors)`, the same shape as a process
naming the `factor_types` it drives. Two gates pin it to the engine: the menu's keys are the type
list `construct_factor` routes, parsed by AST in both directions, and
every method offered is one `check_interpolation` implements and `factor_interp_map` accepts.
`check_interpolation` falls through to `Linear` for anything it does not know, so an unimplemented
method offered is not an error — it is a curve silently interpolated the wrong way.

`SurvivalProb` is the one curve factor that overrides `check_interpolation`, with a different pair
(`Linear` / `LinearExtrapolate`), and it is not routed — so it always takes the extrapolating
branch. That is the current behaviour, declared by its absence from the menu rather than by a row
nobody could honour.

**A calibration owns its tuning block, and the store is authored for the first time.** The
enumeration said this was not a migration but an authoring job against thirteen classes whose only
shared surface is `calibrate()`, and that is what it was. Each `*Calibration` class carries a
`fields` list and a `model_type` naming the process it calibrates, and
`mapping['Calibration']['types']` is `schema.emit_calibration(stochasticprocess)`.

The store is keyed by the PROCESS, because that is what a `Calibrations` entry is filed under —
`Config.parse_json` keys `calibration_process_map` by it and `fetch_all_calibration_factors` looks
a factor's model up in that map. The entry's own `Method` is the CALIBRATION class
`construct_calibration_config` dispatches on, and it is stamped from the class name rather than
declared, so the dispatch key cannot drift from the class it dispatches to. That resolves the
three-name tangle without a fourth store: the process/calibration wiring IS this store's key
paired with its `Method`. `model_type` has to be declared for the reason `calc_type` does —
`HWInterestRateCalibration` calibrates `HullWhite1FactorInterestRateModel`, and no rule recovers
one of those names from the other.

Two types became thirteen. The declared-versus-read audit is again most of what it bought, and it
is now a gate in both directions (`test_the_declared_tuning_keys_are_the_ones_the_class_reads`,
which follows a local `p = self.param` alias). Nineteen descriptors were read by nothing and are
gone — the whole `MLE_Parameters` tree, `Data_Retrieval_Parameters`, `Use_Pre_Computed_Statistics`
and `Number_PCA_Factors`, which was also the last `ALIASED_KEYS` entry. Twenty-five tuning keys
were read and declared by nothing: the ten Kalman knobs, the eight HMM knobs, four on GARCH, three
on the basis model and one on the VAR model. Six classes read nothing at all and declare an empty
list, which is the honest statement that their entry needs only a `Method`.

`Boolean` joins the descriptor vocabulary, because `Use_Student_T` and `Log_Price` are read as
`bool(param.get(...))` off a bare JSON `true` — the `'Yes'`/`'No'` string the rest of the
vocabulary spells a flag with is truthy in both positions, so declaring them that way would have
published a schema that authors a wrong calibration.

`PCAInterestRateCalibration`'s three shape choices are `default=REQUIRED`: they are hard-keyed
reads stamped through onto the emitted `Price Models` block, so there is no engine default to
borrow.

Also read by nothing, and therefore not declared: `ID` on every entry, and the `Vol_Shift`,
`Kappa_Max` and `Sigma_Max` the shipped fixture carries on `LogOUSpotModel` — `calibrate_factors`
passes `vol_shift` as its own argument and `num_business_days=252.0` hardcoded, and `LogOUSpot`'s
two clamps are `calibrate()` keyword defaults no caller supplies.

**A price family owns its quote block, and the design is written down.** `mapping['MarketPrices']`
is `schema.emit_market_prices(bootstrappers)` — one key, `types`, where there were five. The
enumeration set the order and it held: hoist `market_factor_type` onto all four classes, settle
where `InterestRatePrices` lives, then move the descriptors.

The design it is now shaped around is its own page, [Market Prices](market_prices.md): a quote is
a reference to an EXISTING instrument type plus a `Quote_Type` and a `Quoted_Market_Value`, so the
`Instrument` store's declarations ARE the quote's schema and a family only names the types its
quotes may be. `InterestRatePrices` was specified there as the designed-unbuilt family — FRA, swap
and deposit quotes solved for the curve that reprices each to par — with the calibration-Jacobians
thread named as what makes that solve cheap. It is built now, and the two landed together as that
note said they should.

`market_factor_type` is a class attribute on all six families and the engine compares against it,
so the two gates that held the declared types to the literals the engine matched would now be
tautologies. What replaces them is the discipline that would make them tautologies: **no
bootstrapper owns the text**, parsed by AST, the same shape as
`test_instruments_call_resolvers_not_factor_types`; and every `quote_instruments` name is a
declared deal type, which is the reuse-by-reference rule made checkable. Both gates went with the
mock-built suite — only the structure-leg twin,
`test_the_registry_publishes_exactly_the_declared_structures`, survives to state the rule.

`InterestRatePrices` is declared by `InterestRateCurveParameters`, which at the time refused to
bootstrap with an error naming the design note rather than being absent from the store or half
hand-written; it solves now. `quote` is gone as a type — it was never a market-factor type, only the shape of one quote, and it
is now the `Points` container's children. `groups`, `sections` and `properties` are gone with it:
the create menu is the type list itself, the point-field grouping is `quote_instruments` (which
lands as the `DealType` dropdown's values, where a UI needs it), and `properties.Locked_Dates`
reached no read anywhere and is enumerated here instead.

Three live defects went with the move, and one of them was material:

  - `Instrument_Definitions` declared ten columns as three parallel hand-written lists, and the
    `obj` list — which is what `set_repr` picks a deserializer from — was in a **different column
    order** from `col_names` and `sub_types`. `Weight` deserialized as a Period, `Start` as Text,
    `Market_Volatility` as Text. A `Row` makes the three lists one declaration, so the state is
    unreachable.
  - The same table declared `Day_Count` while `create_market_swaps` hard-reads
    `Floating_Day_Count` and `Fixed_Day_Count`. A block authored from the schema — or copied from
    the JSON reference's own example, which carried `Day_Count` too — raises `KeyError` before the
    first swaption prices. The engine is the truth: it is two columns, and the reference example
    is corrected.
  - `Weight` is read by the Clewlow–Strickland objective and by Heston-Nandi's, and only
    Heston-Nandi declared it. An energy option quote authored from the schema had no weight column
    at all.

`Points` declared a `Deal` sub-field with no descriptor and no type reached `Points` itself; both
are fixed. `Deal` joins the pinned shapeless set rather than getting a fake shape: it is a whole
deal whose TYPE a sibling field names, which is the same class of thing as a transition matrix or
a deal map keyed by `Object` — the vocabulary cannot state it and the fix wants a widget.

Four columns and one descriptor are declared no longer, having reached no read:
`Instrument_Definitions`' `Holiday_Calendar`, `Market_Volatility_Type` and `Index_Offset`, and the
top-level `Holiday_Calendar` descriptor no type reached. `Generate_Instruments` and
`Generation_Parameters` reach no read either but stay declared as unbuilt functionality, on the
same terms as the four unread calculation fields — the JSON reference documents both.

`Discount_Rate_Type` is newly declarable: `HestonNandiModelParameters.resolve` reads a
`<field>_Type` for every one of its four references and the store declared three.

`System` is the last hand-written store, and stays one: its single "type" is a UI panel name rather
than anything the JSON dispatches on, and the class consuming `System Parameters` is `Config`
itself — the whole configuration object — so a `fields` list on it would make "a class that
declares fields IS a type" mean something else in that module. It has never been audited
declared-versus-read the way the other stores now have been: `Volatility_Delta`, `Master_Curves`
and `Swaption_Premiums` are read by the bootstrappers and declared by nothing, and
`Grouping_File`, `Proxying_Rules_File` and `Script_Base_Scenario_Multiplier` ship in the market
data and reach no read. That audit is the one piece of this work still owed.

The paired naming cleanup settles first, and is now done: the `MarketPrices` types the engine
matches, one `VolatilityGrid` BODY in place of the three asset-class vol twins — with
`FXVol`/`EquityPriceVol`/`CommodityPriceVol` restored over it as alias subclasses, because the
CRIF-style risk-class partition (`utils.FactorRiskClass`) is a pure function of `factor.type` and
one untagged name makes it undecidable — and the IR prefix chain
(`InterestRate.USD.LIBOR`) **grandfathered** by decision rather than migrated to explicit spread
declarations. `Observed_Factor` and the type-switched `ObservedBasis` tail closed themselves —
`nested_fields` already wires FX and equity alongside commodity, and all four
`calc_factor_code_chain` call sites agree.

**PREPARE / EXECUTE — the hashes are built.** `cx.plan_hash()` and `cx.values_hash()` are pure
functions of the loaded config, sha256 over a canonical dump (sorted keys, `CustomJsonEncoder` for
the objects). The PLAN is `params` and `deals` less two coordinates of their own: every
`bind='value'` field, and `Random_Seed`. So `System Parameters`, `Model Configuration`, `Price
Models`, `params['Correlations']` (the simulation matrix — the cholesky is compile, and this is not
the `Correlation` price factor), `Market Prices`, the structural projection of `Price Factors`, the
whole deal tree and the `Calculation` block are all in, `Batch_Size` and `Simulation_Batches`
included because they change the realized numbers. VALUES is exactly `cx.market_patch()`. The
identity `(plan_hash, values_hash, engine_version, seed)` is documented for callers in
[API Overview](../api_overview.md#patching-market-values-and-replaying-a-run); `engine_version` is
`derivus/_version.py` and no second version source was invented for it.

The same plan/values split now has a MARKET-DATA half: a calibration artifact is a slow object
named by plan-side coordinates with a fast path over it, so a quote that moves between bootstraps
reaches a valuation as a matvec rather than a recompile — see
[Quote Propagation](quote_propagation.md). **The one place the split did not reach is CLOSED
(2026-08-30): quotes are on the values plane.** `schema.MARKET_QUOTE_VALUES` is the one
declaration (`Quoted_Market_Value`, `Quoted_Bid`, `Quoted_Ask`, `Timestamp` per `Points` row —
exactly the tick guard's line, which now READS it), `partition_market_price` /
`apply_market_values` are the projection pair on the `partition_factor` precedent (DROPPED
rather than shadowed, per the guard's own ruling that a pillar starting or stopping to be
quoted two-sided is the same node of the same plan), `plan_hash` hashes the structural
projection, `market_patch` emits the quote rows and `patch_market` applies them — so a vol tick
moves `values_hash` and leaves `plan_hash` bit-identical, which is the disjointness the spine's
two-hash quote firmness needed and the flipped composition gate in
`test_quote_propagation.py` now pins. One rule, all seven families at once: the two that carry
`Points` (`InterestRatePrices`, `FXVolPrices`) have a values half; the five that do not are
asserted empty BY NAME in `tests/test_market_prices_partition.py`, never exempted. The BOOK is
stricter than the engine on purpose: `/book/market` refuses a quote-values patch with the
quote-update path as the named remedy, because on a live book the factors must never go
silently stale against their own quotes. A JSON `null` in a document is an ABSENCE, so the
values plane is an identity over it — `patch_market(market_patch())` changes neither hash. The
artifact's `plan_key` and `derivus_bloomberg`'s snapshot guard are the unification's other two
siblings: the first consumes the partition, the second cannot import the engine and is held
equal by an AST parity gate.

What remains is the rest of the service layer, then the live-refill EXECUTE path — in that order, re-sequenced
by decision: the derivus_jupyter successor is a web SPA (Angular/React) rendering from the schema
over a ROBUST API, not another Python-first front end (NiceGUI was considered and rejected: AG
Grid's tree-data mode needs an Enterprise license), and the SPA is only worth building on a solid
service underneath. marimo serves the quant WORKBENCH — the research loop — which is a different
job; derivus_jupyter was the editor app built in notebook clothing, which is exactly why it was
clunky. The verbs are
**calculation-agnostic by design**: PREPARE/EXECUTE carry the same contract for every calculation
type — a CMC run patches and replays exactly like a base valuation, it just runs longer. What is
deprioritised is only the refill's URGENCY, by measurement rather than by taste: recompiling is
relatively quick, and the latency the refill buys back is felt mainly where the run itself is
cheap — base valuation and the **SOLVE** verb (a strike, a margin, a vol).
CMC is compute-heavy enough that compile time is dwarfed by the run, so it loses nothing by waiting;
it must gain the verbs on the same terms when they land. The solve iterates pricing on one changing
DEAL field, and strike and margin are structural today, so every iterate recompiles — which is
where the case for extending `bind=` to payoff-only deal fields will be measured. A vol solve
already patches cleanly. (Structuring as a CALCULATION TYPE is retired — see below; it shipped as
the `/book/solve` verb over ordinary base valuations.)

**Service layer — slice 1 built.** `derivus/service.py` is the HTTP binding of the Context verbs and
owns no logic of its own: `GET /schema` publishes `schema.mapping` with `engine_version`, `POST
/validate` returns `cx.validate()` verbatim, `POST /execute` takes a job document plus an optional
`Patch` delta and always answers `{result_id, status}`, and `GET /results/{id}` returns the run's
`Results` tables stamped with the replay tuple. The ruling it implements is one vocabulary, two
bindings — the SPA, marimo and Excel/xlOil are clients of the same endpoints, and nothing
client-specific may enter the surface; anything a client needs that the verbs cannot answer is a
missing verb on `Context`. A posted job goes through the decoder that reads a job file, so there is
no second parser. `fastapi` is the `service` extra, imported only there. Pricing goes through ONE
worker thread on a cost-class priority queue — no cpu lane, because base valuation IS Monte Carlo
for an autocall or TARF book, and device selection stays in the engine — and a `result_id` is the
content hash of the replay tuple, so an identical submission coalesces onto the queued or running
job rather than enqueueing a second copy. The decisive gate is parity: a job over HTTP prices
identically, table for table, to the same job through `run_job` in process. The one thing that had
to be added elsewhere is `CustomJsonEncoder` learning the two shapes a `Results` tree holds — a
DataFrame (as `split`, with missing cells as `null`) and an ndarray — rather than the service
inventing a result schema of its own.

**Service layer — slice 2 built.** What the SPA needs before it can be written against this at all.
`CORSMiddleware` first, because a browser discards an answer the server did not invite: origins are
`*` by default and narrowed with `DV_Service --origin`, and the service is stated to be a
trusted-network deployment — auth is still the last slice. `GET /schema/job` publishes the job
ENVELOPE, which is the one piece of contract `/schema` cannot state and is not guessable from it;
it is a skeleton that LOADS rather than a description, and the gate posts it back through
`/validate` and `/execute` unedited. `POST /describe` answers what the engine made of a job without
running it — the book counted by `Object`, both halves of the factor universe, the `Calculation`
block as loaded, and the queue's cost class with a crude `Batch_Size x Simulation_Batches x
Time_Grid`-segment estimate. `POST /prepare` names a parsed job by its `plan_hash` and caches it
(bounded 32, LRU), so `/execute` takes `{plan_id, Patch}` in place of the document; the cache holds
a PRISTINE parse and hands out deep copies, and the two gates that matter are that a plan-id execute
reports the same `result_id` as the full-document execute — content addressing does not care how the
job arrived — and that a patched execute leaves the plan as it found it. It is the PARSE that is
cached, not the compile; the compile arrives behind the same verb with the live refill.

Two things went in elsewhere rather than into the service, because the wrapper owns no logic:
`cx.describe()` is a Context verb (`Config.describe`), and the factor half of `validate` became
`Config.factor_universe`, which both callers now read — `validate` takes its `missing` list.

The `/results` contract MOVED, deliberately, before any external client exists. `GET /results/{id}`
is now a summary — status, the replay tuple, and `tables: {name: {rows, columns}}` — and cells come
from `GET /results/{id}/{table}?offset=&limit=`. "Never return the exposure cube" is the design rule
this implements: a CMC's `mtm` is dates by scenarios and does not fit inline, and a client that has
to hold the whole thing to show one page of it is a client that will fall over on the first real
book. A group of tables (`cashflows`, `scenarios`) flattens to the path naming each one. Now was
the moment: slice 1 shipped yesterday, the Excel add-in is the first client and it is migrated in
the same breath, so the break costs one commit and no downstream.

**Service layer — slice 3 built: the book, the web UI, and the MCP binding.** `DV_Service --book
FILE` serves one live job document whose FILE is the source of truth: `GET /book` (document +
content etag), `POST /book/deals` (add/delete one deal, **validated before an atomic write** — a
refusal returns the messages and touches nothing, and only what the booking itself breaks blocks
it: its own authoring messages, or market data the book did not already lack), and `POST
/book/price` (the what-if: the book plus an optional candidate priced on an in-memory copy — the
verb a par/margin solve rides, since a linear payoff's value is affine in its amount). The
document verbs (`splice_deal` / `remove_deal` / `sniff_indent` — positional `deal_path` identity,
container-checked parents, rewrite in the file's own indent so book-then-delete is byte-identical)
live in `config.py` beside the wire format; the service wrapper still owns no logic. Results now
carry the run's `stats`, and `mapping['Instrument']` publishes `containers` so no client imports
the engine to know which types take children. Three clients sit on the surface: the **web UI**
(`web/`, React + Vite + tree-shaken echarts, served by `--ui`, rendering entirely from `/schema`
and the document — value-first field dispatch, `.Curve` branched on row arity, results by SHAPE,
views in a workspace registry so a future SACCR/backtest/archive screen is an entry, not a
refactor; slice 1 is view + run, and it follows the book by etag poll so a booking from any
client appears within a tick), the **Excel add-in** (`ServiceClient` gained `book` / `book_deal` /
`delete_deal` / `price_candidate`), and the **MCP binding** (`derivus_mcp/server.py` — see
[MCP Binding](mcp.md): twenty-four tools today, docstrings as the model-facing contract, a rejected
booking returned as data, an import gate holding it to `requests`, `mcp` and `mcp_types`).

**The structuring calc is RETIRED as a concept (2026-08-26), by ruling**: structuring is a solve
VERB over base valuation, not a calculation type — the bootstrap's own pattern (a root find
around the engine's pricers) applied to a deal field. `POST /book/solve` /
`derivus.solve_deal_field`: brentq inside declared bounds, else a secant (exact in two pricings
for an affine field), the candidate priced alone on the book's market data, the seed fixed so a
Monte-Carlo-priced objective is deterministic (the solved field is conditional on the draw — the
swaption calibration's philosophy), the residual checked against a declared tolerance rather than
clamped, and the result's tables being the run AT the solved value. One queued job, one
result_id, `stats.Solved` carrying the coordinates. Multi-strike structures (collar, seagull)
compose from 1D solves under their conventions; no N-D optimizer until a genuinely coupled case
arrives. A deal field is structural, so every iterate recompiles — the solve is where the case
for extending `bind=` to payoff-only deal fields (a strike moves no discovery) will be MEASURED.

The book surface also carries `amend` (merge fields into the deal at a path, the same
validate-delta and byte-identical-undo discipline as a booking, negative paths refused by name),
reachable from all three clients: `ServiceClient.amend_deal`, the MCP `amend_deal` tool ("make
the notional 3m"), and the web UI's scalar edit — declared scalar fields grow inputs over the
live book, the wire form chosen by the DECLARATION (`encodeScalar` inverts the token map off
`widget`+`obj`, no field-name whitelists anywhere), refusals rendered verbatim, and the etag
refresh doing the repaint so the client holds no edit state.

**The market ticks through the book (2026-08-26)** — the practical loop the Bloomberg package
feeds. `POST /book/market`: quote blocks install or update (`config.update_market_quote` — the
value-only structure guard generalized to every family: only `Quoted_Market_Value`/`Timestamp`
may move, a changed pillar/expiry/convention refuses by name), a `patch_market`-shaped values
patch (the engine's own structural refusal), and the bootstrap — now a `Context` verb — turning
quotes into the factors the pricers read, all in one atomic write that a bootstrap ERROR refuses
whole with its own messages. MCP: `update_market_quotes` / `patch_market_values`; Excel:
`ServiceClient.update_market`; web UI: `bind='value'` fields edit in place (the declaration is
the predicate — structure stays read-only exactly where the engine refuses it). Gated end to end:
canned Bloomberg observations → `derivus_bloomberg`'s own normalization → the book file carries
the `Malz` surface → `/book/solve` lands an FX option's strike on a target premium
(`test_a_bloomberg_snapshot_reaches_a_solved_strike`; the MCP twin runs the whole day in four
tool calls and then moves the mark with a spot tick).

**Service layer — slice 4 built (2026-08-27/28): the desk provisions and quotes itself.** One env
var, `DV_HOME` (`~/.derivus`), names where a desk's own files live — the live book `DV_Service`
now serves zero-arg, the Bloomberg security map and seed, and pending quotes — and the tools
materialize it on first use rather than any setup step. `POST /book/bloomberg` is the terminal
round trip as one queued job (provision the map — every candidate verified against the terminal's
own NAME and last print, a dead benchmark refused on its date however sane its price — then fetch,
install and bootstrap through the same `market_edit` seam `/book/market` rides), with `progress`
readable off `/results/{id}` and forwarded as MCP progress notifications, which is what keeps a
five-minute first use alive. STRUCTURES are declarations (`derivus/structures.py`, emitted as
`mapping['Structure']`): a structure carries its sales names, its legs (the Market Prices
reference-by-instrument pattern), and a declarative recipe the runner solves server-side —
`POST /book/structure` files the outcome as a pending trade plus its Excel ticket
(`derivus/quote_sheet.py`, the `quote` extra) in `DV_HOME/tmp`, and `POST /book/quote` is the
approval, a booking like any other. The runner owns BOTH market-axis conversions (a USDZAR strike
arrives as 15.50; the engine's FxRate carries reporting-per-unit, and a market call is an engine
put on the quote currency), so the finance never depends on which model drives the MCP tools —
the design brief for a host that may be a local LLM. `FXForwardDeal` is a quotable
`InterestRatePrices` benchmark ([Market Prices](market_prices.md#interestrateprices)): a currency's
curve solves directly from forward outrights, CIP written nowhere, quote refusals measured. The
distribution went pip-first alongside: `derivus_mcp` and the built web UI ship in the wheel,
`DV_MCP`/`DV_Bloomberg` are console scripts, and `desk` is the one-line install. The market keeps
itself current between asks: `DV_Service --tick SECONDS` runs a metronome that submits the SAME
queued Bloomberg job `POST /book/bloomberg` does — routine fetch only, never provisioning (an
unprovisioned `DV_HOME` refuses by name and the cadence carries on), a beat whose predecessor is
still in flight skipped rather than queued, and three consecutive failures stretching the interval
fivefold until one lands. A quote reads the spot off the terminal at quote time; the surface and
the curves are whatever the cadence last ticked in.

What remains of the service: SSE for progress (the UI and the book poll today; `/book/bloomberg`'s
progress field is the shape SSE would stream), a cost estimate that reads the real grid rather
than a segment count, auth with budget caps — now load-bearing sooner, since a cloud-hosted MCP
host (M365 Copilot) needs the binding's streamable-HTTP transport plus a real auth story — and
the rest of the web UI's edit surface — tables/curves (an editable grid component) and creating
deals from the UI. Barrier legs SHIPPED with `ForwardExtra` (protected rate given,
the BARRIER solved — the runner's third axis conversion: a barrier level inverts like a strike and
its Up/Down direction inverts with it), which drove the first numbers through the analytic
knock-in branch. That step's SPREAD INPUT landed first: `PX_BID`/`PX_ASK` ride the vol pillars as
`Quoted_Bid`/`Quoted_Ask` beside the mid, the bootstrap still reads only the mid, and the quote
layer shifts each leg's own copy of the written surface by the ATM half-spread on the client's
side — mid for the book, two-sided for the quote ([Structures](structures.md#two-sided)).

**RISK-IMPACT PRICING v1 is BUILT** ([Structures](structures.md#risk-impact)). A trade's charge is
the cost of hedging the RESIDUAL it leaves, at the market's own two-way and not at an invented
bp-per-skew number: the composed candidate is MIRRORED (the verb the booking already uses) and the
book's vol risk is read with it and without it under `Greeks: 'First'`, in QUOTE coordinates —
`dV/d(ATM)`, `dV/d(RR)`, `dV/d(BF)` per pillar off `Quote_Sensitivity` on the `FXVolPrices` block,
which is the first consumer of that switch outside its own gates. Each bucket's move in absolute
risk is charged that bucket's own half-spread; a negative total is a saving and `participation` of
it comes off the quote. The V1 SCOPE, named honestly: vol only (`scope: 'vol'` refuses anything
else), quote-space coordinates (the per-expiry ATM vega fallback was budgeted and never needed),
ONE re-solve rather than a fixed point (the risk is measured on the full-spread candidate and the
re-solve moves the coordinate second order — measured 0.026% on the gate's book), and NO surcharge
past the two-way: the market spread is the ceiling. The POLICY is a declared `Quote Policy` block
on `Calc` — `participation`, `floor`, `scope`, `bucket_limit`, `min_ticket_bp` — and its ABSENCE is
the feature's off switch, so a book without one quotes bit for bit as it always did. Measured on
the gate's book: a desk quoted the same collar it already holds tightens from a full charge of
81.2194 USD to 75.9178 on a saving of 10.6031, and its solved cap moves from 19.16949 to 19.17396
against a mid of 19.23863. Named next for the registry: **incremental XVA as the v2 of that same
step** — a counterparty on the quote and `CVA(book + mirror) − CVA(book)` through the
`Credit_Monte_Carlo` engine, the same two-run seam with a different calculation in it — and a
ratio-solve primitive for participating forwards. The v2's CVA half now has its service seam
already built: `service.xva_document` composes exactly that job over one netting set, so the
incremental step is a second call to it with the mirror spliced in.

**THE BLOTTER'S TWO DATA VIEWS ARE BUILT**, and they are deliberately not the same kind of thing.
`GET /book/risk` is the CONSOLIDATED view's feed — one base valuation with `Greeks: 'First'` over
the whole book, counterparty-blind, computed on a miss and cached under a content etag over the
deals, the whole `MergeMarketData` and the `Calculation` block, so a blotter polls it on the same
beat it polls `/book` (measured on the quoting fixture book: **338 ms cold, 3 ms warm**, and the
warm path is a dict lookup). `POST`/`GET /book/xva` is the XVA view, and it is a CACHED PROJECTION
on purpose: a credit Monte Carlo is minutes of device time and must never ride a tick, so
`DV_HOME/xva.json` holds the last run of each `NettingCollateralSet` — one atomic row-at-a-time
write, the book writer's discipline — and a desk asks for a FULL or PARTIAL recalc when it wants
the file to move. One queued job PER SET at the CMC's own cost class, so quotes and valuations keep
jumping the queue and the projection fills in row by row; each row carries its own `as_of` and the
replay tuple, a partial recalc moves only the rows it names, and staleness is DATA rather than a
failure. A counterparty the market data carries no `SurvivalProb` block for lands `status: 'failed'`
carrying the engine's own wording in its row, never as a lost projection. The three MCP tools
(`book_risk_summary`, `xva_view`, `recalc_xva`) say the distinction out loud in their docstrings,
because a model that treats a three-hour-old CVA as live is the failure mode this shape exists to
prevent. Left for the web half: the two views as screens, and an `xva.json` row that also carries
the exposure profile a client would chart.

**The Excel add-in is the first real client.** `excel_integration/service_client.py` is a plain
`requests` client of the verbs and imports neither `xlwings` nor `derivus` — it is the HTTP binding
a marimo notebook or a plain script uses as readily as the workbook. The gate that read its import
statements went with the mock-built suite; only the MCP binding's twin still holds that line.
`worker.py` and `queue_clients.py` are DELETED, not deprecated: the service is the worker and the
queue, and the file-queue settings went out of `config.py` with them.
Solace returns later as a second transport in front of the same verbs, not as a second queue.

Two things stayed in process, each for a stated reason. `RF_SOLVE_*` iterates a pricing run on one
changing DEAL field — what `/book/solve` now does server-side; a deal field is structural today, so
every iterate is a fresh compile and a round trip per iterate would buy nothing, which is why this
one stayed in process. And `RF_*_PORTFOLIO` builds its job from the sheets through
`portfolio_service`, which still reads `schema.mapping` directly — migrating that to `GET /schema`, which publishes exactly
those declarations, is the remaining end-state for the folder: after it, nothing there imports the
engine and the add-in installs without it.

`xlwings_udfs.py` is a thin uncovered shim by construction — it cannot be imported without xlwings,
so nothing that could be gated is allowed to live in it.

**VALIDATE built**, and only that: `cx.validate()` returns `{'deals': {reference: [message]},
'factors': [name]}` — the authoring messages of every deal in the book, and every price factor it
names that the market data has no block for. It answers the want-list and nothing else, and it reuses
discovery (`discover_factors`) and not `calculate_dependencies`, which is where `find_models` mints the
`Price Models` dummies that make that method non-idempotent. The discovery half is now
`Config.factor_universe`, which returns both sides (`resolved` / `missing`) because `describe` wants
both; `validate` reads its `missing`, and the deal walk `Config.walk_deals` is shared with it.

## Flagged, not authorised

`runtime` still carries free functions over the hedge bundle in two clusters (Objective and
Accounting) with `_UTILITY_OBJECTS` duplicated — the shape [Conventions](conventions.md) calls a
class waiting to happen. `DealStructure`'s recursions are in the same category.

## Test selection

`gates/impacted.py` selects the tests a change can actually reach: an execution-coverage map
(per-test contexts, recorded as a byproduct of a campaign-boundary run via
`pytest --cov=derivus --cov-context=test`, then `--build-map`) joined with a static, always-fresh
fixture map — tests depend on JSON fixtures at runtime, fixtures chain through `MarketDataFile`,
and no import graph can see either. Selection is file-granular and FAILS OPEN, loudly, for
anything the map has not seen; `derivus/__init__`, `utils`, `calculation` and `conftest` are
whole-suite modules by construction. The full suite still runs at campaign boundaries — for
map staleness (the snapshot only the next instrumented run refreshes) and whole-tree
interactions no subset sees — and the TREE HOLDS STILL while it runs: two full runs each
failed one mutation gate purely because source was edited mid-run (`inspect`-based gates read
the file on disk at imported line numbers, so a shifted tree pulls the wrong text under them).
The inner loop uses `--dirty --run`; the boundary run certifies and re-records the map in one
pass.

## Tidy-ups

**`get_implied_correlation` makes its callers own factor types.** The resolver rule — the `get_*`
layer owns every factor-type text key; an instrument knows which resolver to call, never the
literal — was gated by `test_instruments_call_resolvers_not_factor_types`, removed with the
mock-built suite, so the rule stands as convention; and it stands with one held
exception: `get_implied_correlation`'s two callers build type-prefixed correlation-name tuples
(`('EquityPrice',) + …` in `Deal.check_option_data`, `('FxRate',) + …` in
`EnergySingleOption.calc_dependencies`). The fix is two single-caller wrappers
(`get_equity_fx_correlation`, `get_fx_reference_correlation`), which brushes the
no-abstraction-ahead-of-a-second-caller rule — held until a third correlation pair shows up or the
rule is judged to outrank it. The gate did not cover tuple literals, and there is no gate now, so this stays a note.

**`Flot` is a plotting library, not a type — CLOSED**, and `Three` (three.js/k3d) with it: the
widget tokens are now `Curve` and `Surface` (`Surface` covering both shaped types, so a renderer
branches on the value's row arity, never the token), with the legacy spellings owned by the
Jupyter front end's `LEGACY_WIDGET` map. What the pass had to resolve was the `default` dict:
keyed "by widget", it also carried `Surface`/`Space` as TYPE names, so the rename would have made
`default[F.WIDGET['Space']]` a silently wrong blank. The three shape keys were read by nothing —
every call site keys by a table-COLUMN token — and are deleted; `BLANK`, keyed by type, is the one
definition of a shape's blank. Gated by `test_the_widget_vocabulary_names_no_plotting_library` and
`test_the_column_default_map_is_keyed_by_a_column_token`; the one consumer that would have failed
SILENTLY was `excel_integration/portfolio_service._COMPLEX_WIDGETS`, a membership test that would
have started emitting blank curve strings into deal defaults.

**Inline comment density.** The boundary-correction work left ~12 inline blocks of 4-11 comment
lines, several outweighing the code beneath them. House style is detailed docstrings, 2-3 lines
maximum inline, and never more comments than code. The material is right - the reasoning, the trap,
the measurement - it just belongs in the function's docstring or the commit message. Worst
offenders: `pv_discrete_barrier_option`'s hit-mask and rebate blocks, `sim_spot_oss`'s terminal
digital, `NettingCollateralSet.post_process`'s `net_from_gross`.

**A debug write aimed at the repo root.** `bootstrappers.py` (the HW2F solve's debug block)
overwrites `debug.deals` and attempts `write_trade_file('ZAR.aap')` in the CWD - currently
inert (every run logs "Could not write output file"), but a run from a writable CWD drops an
artifact where the no-artifacts rule forbids one. Delete the block or route it to a declared
output directory.

**The compounding-leg shape check.** `pv_float_cashflow_list` selects the compounded-in-arrears
path by comparing reset count to cashflow count — a shape encoding of intent, set up at
`calculate_dependencies`. It works and is now documented
([Quote Sensitivities](quote_sensitivities.md#curve-contracts)), but an explicit signal on the
compiled cashflow object would say the same thing without the inference.

## Model punchlist

**The HW2F swaption calibration: the α→0 repair is BUILT, and the analytic price is now an
OBJECTIVE behind a declared field.** `tests/test_hw2f_analytic.py` carries all of it — the repair,
the checker, and the wiring the checker's own measurement licensed.

*The repair.* `hw_calc_IJK` divides by `a³`, `hw_calc_H` by `a²`, and the `AtT`/`BtT` assembly by
`a` — every one a REMOVABLE singularity, so the failure was never a raise but silent
cancellation. Each closed form now takes a series branch below a threshold MEASURED against a
40-term reference (and re-checked against 60-digit mpmath: the branch jump at each crossover is
2.8e-11 / 2.2e-11 / 7.2e-16 relative, all of it the closed form's own residual, so the accurate
branch is taken on BOTH sides). Three reachability findings, none of them theoretical: the basin
step is multiplicative so it can decay but never cross — yet **2.8% of 20 000 simulated walks go
under the IJK threshold**, and `least_squares` (plus the L-BFGS-B *inside* basin hopping) crosses
zero outright under `alpha_bounds = (-0.5, 2.4)`; `params_ok` read **True** at `α₂ = 1e-4` where
the benchmark priced **21% wrong**; and `Alpha_1`/`Alpha_2` are declared `default=0`, so a block
omitting them was NaN in every benchmark. Two the contract did not anticipate: there is a SECOND
singular locus — `J[i][j]` is taken at `α_i + α_j`, so **α₁ = 0.1, α₂ = −0.1 is singular with
neither reversion speed near zero** (pre-fix: NaN, `params_ok` False, and basin hopping rejecting
a perfectly admissible step) — and **the repo's own identified fixture engages the branch at its
solved point**: θ\*'s `Alpha_2 = −0.0179`, negative and inside the IJK threshold, reached by the
optimizer unaided over a bound that straddles zero.

*The checker.* `HullWhite2FactorImpliedInterestRateModel.schrager_pelsser_swaption` — annuity
weights frozen at t=0, constant loadings, Bachelier — built off the `J` array `precalculate`
already computes and used to drop (now retained, so there is ONE spelling of `J`). Building it
corrected the derivation it came from: the `e^{-(α_k+α_l)T₀}` prefactor does NOT belong, because
freezing the bracket freezes the loading on the SCALED martingale `Y_k = e^{α_k t}x_k` and `J` is
`Y`'s own covariance, so `J` enters bare (`∂S/∂x_k(t) = e^{α_k t}q_k` verified to 1e-15; the
loadings independently finite-differenced to 3e-12 at a ρ where the cross term is observable).

*What the measurement says*, at θ\* on the 25-quote identified fixture, 2²⁰ paths, the difference
decomposed pathwise under common random numbers (the decomposition closes to 1e-18):

| term | size |
| --- | --- |
| the leg convention (single-curve float-at-par) | **≤ 5.6e-18** — exactly zero, and frequencies are irrelevant to it |
| SP's annuity-freezing bias | −0.13 to +2.17 bp of normal vol |
| the MC's own numeraire bias | −0.35% to −1.61%, i.e. **0.6–3.0 bp — systematic** |
| the MC's noise per objective evaluation | 0.7–1.1 bp at its 8192 paths |

So the first question this was built to answer is closed **exactly**: `1 − Σcᵢ P(T₀,Tᵢ)` is not an
approximation on this benchmark — `Forward` and `Discount` are one curve and each float coupon's
pay day sits on its accrual end, so the leg telescopes path by path (measured 1e-18 on 3M/12M and
6M/12M). And the headline is the one the checker existed to make possible: **SP is the more
accurate of the two over most of the grid**, because the simulation's systematic error exceeds
SP's almost everywhere. SP sits inside one evaluation's noise at **22 of 25** benchmarks,
stepping outside only at 3Y×10Y, 5Y×10Y and 10Y×10Y, each by about a fifth of a noise unit. The
bias rides expiry × tenor and ρ together (10Y×10Y: +1.06 → +2.09 → +3.80 bp across
ρ = −0.95 → ρ\* → +0.95) and is under 0.21 bp at every expiry for tenors ≤ 3Y.

*What that licensed, and what it cost.* `Objective` is a declared field — `Analytic` differencing
**vols against vols, plain**, and `Monte_Carlo` bit-identical to what this family always did.
`Analytic` is the **default** as of 2026-08-31; the ruling is recorded at the end of this row. The residual is `Weight × (σ_SP(θ) − σ_market)` with the market side a CLOSED-FORM
Bachelier inversion of the same numpy premium the MC path builds (`P√(2π/T₀)/A` at the money, on
SP's own annuity — so every quoting convention, displacement and `Volatility_Delta` is inherited
through the premium and no root find enters an objective evaluation). **The quartic retires on that
path and the measurement is the point**: on the identified 25-quote block the MC chain stops at
`‖J'r‖` **8.24e3** (`‖r‖` 45.97), down from 2.86e11 at the seed — seven and a half of eleven orders
— which is why the gate file has to declare `Stationarity_Tol` **1e5**; the analytic chain lands at
**3.85e-6** (`‖r‖` 2.26e-3), *inside the field's own 1e-3 default*. Eight orders of declared
tolerance, and the raw ratio is not claimed as like-for-like because the units differ. In repriced
vol space the two answers agree — rms |SP − market| 5.14 bp at θ\*_MC against 4.51 at θ\*_An on a
175–208 bp level, the two 2.52 bp apart rms — while in θ space they share nothing (ρ −0.0046
against −0.9409), which is [rank deficiency](quote_sensitivities.md#rank-deficiency) and not a
disagreement. **The comparison is taken on the four-quote block too, and reads differently there by
construction:** four quotes against 23 parameters is under-determined, so each objective interpolates
exactly in its own metric (`‖r‖` 4.4e-8 Monte Carlo, 4.0e-7 analytic) and the 2.45 bp rms between them
is the *two metrics* apart — SP's freezing bias plus the simulation's own numeraire error, **adding**
at 10Y×10Y to the 4.13 bp printed (≈2.9 bp of it the −1.61% numeraire error at that benchmark, ≈2.2 bp
SP's own worst corner). θ space is 0.152 apart there, again all of it in ρ. **DETERMINISM:** two
analytic solves at one seed agree TO THE BIT; across seeds 5120 / 7 / 99 the Monte Carlo θ\* is
bit-identical — its basin stage contributes nothing its least-squares stage does not — while the
analytic one spreads **0.250**, in ρ, the coordinate four ATM quotes say least about. That spread is
evidence about the FIXTURE and not about the objective.
**Wall clock, same box and precision, over those three seeds:** four-quote fixture 75.1 s Monte Carlo against
13.4 s analytic — 5.6×, bought as 12× per evaluation against 2.2× the evaluations (351–394 against 170;
the chain needs MORE iterations, not fewer, because a smooth deterministic residual is one
`least_squares` can keep making progress on). The 25-quote block solves analytically in **836 s** on
that box, over 1915 evaluations at **0.437 s** each, and that is **not** to be read against the **531 s** the same block took under the Monte
Carlo objective: that figure is float32 on CUDA, which is what `construct_bootstrapper` hands a job,
and this one is float64 on CPU — the four-quote pair is the like-for-like reading. The per-evaluation
figure is the one that generalises, and it does not flatter the analytic path: 0.437 s at 25 benchmarks
against 0.036 s at four is **twelve times the cost for six times the benchmarks**, because SP is one
scalar call per benchmark and the grid its `J` is integrated over grows with the expiry set. SP is likewise not
faster *per evaluation* on CUDA (25 scalar calls lose to one batched kernel — 0.158 s against 0.140 s), so
batching it across the benchmark set is the open build; what it buys today is exactness, a
gradient, and a chain whose two stages minimise the same function. The analytic solve ends with an
**honesty reprice**: one
MC pass at θ\*, logging the worst benchmark's relative premium residual by name (−4.64% at 1Y×5Y on
the 25-quote block), the component-HN "reports itself CAPPED" pattern. The full Gauss–Hermite
quadrature still earns its keep in exactly one place — the long-expiry/long-tenor corner — and
nowhere else.

**THE DEFAULT HAS FLIPPED — ruling taken 2026-08-31, and this row records it rather than arguing
it.** The sentence that stood here said the default does not flip on the accuracy measurement alone,
that flipping is a ruling and not a build, and that was right at the time: accuracy was the only
reading in hand and it is the weakest of the four. The ruling was taken once the other three landed,
and it names them. **(1) Accuracy**, the reading above: SP inside one MC evaluation's own noise at
**22 of 25** benchmarks, and the *more* accurate of the two over most of the grid, because the
simulation's −0.35% to −1.61% numeraire bias exceeds SP's −0.13 to +2.17 bp freezing bias almost
everywhere. **(2) Stationarity**, and this is the one that decides it: `‖J'r‖` at θ\* is **3.85e-6**
on the plain residual against **8.24e3** on the quartic — *inside* `Stationarity_Tol`'s own **1e-3**
declared default rather than seven orders outside it. A family whose default objective cannot clear
its own default tolerance is a schema two ways round, and the quote-side backward refuses on exactly
that norm. **(3) Determinism and cost**: two analytic solves at one seed agree **to the bit**, and
the four-quote chain is **13.4 s against 75.1 s**. **(4) The quote side**, which is the prerequisite
that made the flip a build rather than a regression: `Quote_Sensitivity` on this path used to refuse
by name.

*The two prerequisite landings, by name*, both of which had to be in the tree first and both of
which are recorded in the paragraphs above and below this one. **The domestic-measure correction** —
`implied_process` builds the objective's process on a quanto-suppressed twin, so both objectives
calibrate under the rate currency's own measure and the analytic default cannot silently change what
a *foreign* curve fits (pre-fix the two disagreed by 5.96%–12.42% of premium, 10.9–24.4 bp of normal
vol, on a ZAR curve under a USD base). **The analytic quote side** — built, separable, and gated,
so a `Quote_Sensitivity: 'Yes'` block reaching the new default gets a Jacobian rather than a refusal.
Without the first the flip would move foreign-curve answers; without the second it would break a
declared field. With both, it moves nothing that was declared and everything that was not.

*What the flip is not.* `Monte_Carlo` is fully supported, unchanged to the bit, and stays the
**engine's own estimator**: it is the oracle every comparison in this row is taken against, and it is
what the analytic solve's **honesty reprice** runs at θ\*. Nothing re-baselined — the arithmetic is
unchanged and both recorded θ\* vectors (`ID_THETA`/`MC_FOUR_THETA` on the MC path,
`ID_ANALYTIC_THETA`/`AN_FOUR_THETA` on the analytic one) are bit-identical across the flip. What
moved is which of them an **undeclared block** reaches, and the enumeration of every such site is
`tests/test_hw2f_analytic.py`'s own deliverable: every gate whose subject is the simulation now
declares `Objective: 'Monte_Carlo'` and is verbatim-preserved, three gates whose subject is the
family default are re-based (`test_the_declared_sample_shape_is_the_shape_the_engine_uses`, which
grew a second Monte Carlo block because a sample the analytic closure never draws cannot hold the
`Simulations`/`Batches` half; `test_the_two_spellings_of_the_default_drive_the_adapters_identically`,
re-pointed to absent-against-`'Analytic'`; and
`test_the_monte_carlo_objective_still_solves_to_this_vector`, renamed from
`test_the_default_objective_still_solves_to_this_vector`), and
`test_the_absent_objective_is_the_declared_analytic_one` is new — two four-quote chains, `array_equal`
on all 23 doubles of θ\*. **The enumeration is over BLOCKS, not over calls to the closure helper**,
and it had to be re-taken that way: review caught a FOURTH undeclared site the call-site scan
structurally could not see — the module-scoped `calibration` fixture (`tests/test_hw2f_analytic.py`
`:446`), which authors its block as a dict LITERAL and enters `calc_loss_on_ir_curve` directly. Its
three gates (`test_params_ok_does_not_guard_this` and the two locus ladders, 15 invocations) price
the removable singularities through the SIMULATION, and their prices are that file's own opening
tables; undeclared it built the analytic closure and read **0.026635828 / 0.027167045 / 0.027172548**
at the alpha₂ locus and **0.034311506 / 0.034320574** at the alpha-sum one — +0.10% and +0.60% off
tables nothing in the suite would then have reproduced. It declares `Monte_Carlo` and reads the
recorded **0.026623990 / 0.027139662 / 0.027145006 / 0.034107728 / 0.034116651** bit for bit. (Not a
vacuous-gate defect either way — `analytic_loss` calls `precalculate`, so `params_ok`, the reversion
floors and the H/IJK/B branches were exercised on both paths; what the flip had silently dropped was
the MC BACK half, `generate` / `pv_float_cashflow_list` / the deflation, at the two degenerate loci,
which is what those tables measure.) Three blocks in that file carry `Swaption_Volatility` and
`Instrument_Definitions`, and all three now name a path.
**No JSON fixture anywhere in the suite bootstraps an HW2F block** — the
family is constructed in Python by its gates and nowhere else — so no fixture re-baselined and no
job in this repository moved a number. The one behavioural consequence outside the gates is
`derivus_bloomberg/swaption_vol.py`, which deliberately emits *no* `Objective` because a market
fetch does not decide a job's optimizer: a Bloomberg-emitted ladder therefore now solves
analytically, and that emitter's docstring says so.

**Standing consequence.** Every HW2F block *outside* this repository that omitted `Objective`
re-solves to a different θ\* — the analytic one — and should either name `Monte_Carlo` to keep its
old answer or re-baseline onto the new one. That is the flip, stated as the only thing it does.
It also gets CHEAPER, and the direction is worth stating because the honesty reprice makes it look
otherwise: such a block does now pay one MC pass at θ\* it never paid before, but it stops paying
for a full MC SOLVE — on the four-quote fixture, **about 12 s per analytic chain including that
reprice against about 73 s for the Monte Carlo one**, read twice on a quiet box (24.29 s and 24.91 s
for `test_the_absent_objective_is_the_declared_analytic_one`'s two analytic chains, against 73.14 s
and 73.66 s for `test_the_monte_carlo_objective_still_solves_to_this_vector`; the 13.4 s against
75.1 s above is the same pair of solves read with the suite running beside them). The reprice is the
dominant cost only relative to a sample-free solve, never relative to what an undeclared block paid
before.

**The analytic quote side — BUILT, and the refusal this row carried has retired.**
`Quote_Sensitivity: 'Yes'` with `Objective: 'Analytic'` works, off the SAME quote leaf, the same
`black_premium` twin and the same `LeastSquaresSolve` wrapper — `market_swap_class.market_normal_vol`
carries the splice through the closed-form Bachelier inversion and that is the whole engine change.
It is worth exactly zero forward (residual, model value, market vol and `‖J'r‖` bit-identical with
the switch on and off). What it buys is a residual that is **SEPARABLE** in (θ, q) — a θ-function
minus a q-function, because the annuity the market half divides by is severed at source — so of the
two terms Gauss–Newton drops, one is **structurally zero** (measured around the splice: `J`
bit-identical across re-authored quote rungs, `∂r/∂q` bit-identical across θ rungs) and the other is
the textbook `O(‖r‖)` at **8.75e-4** of `J'J`, against the Monte Carlo residual's **0.500064**.
`∂r/∂q` is exactly diagonal — 600 of 625 pairs structurally absent. The path runs at
`Stationarity_Tol`'s own **1e-3** default where the Monte Carlo path needs a block-declared 1e5.
Triangle: one backward against the spelled-out contraction **2.22e-16**, against the operator form
**1.088e-14**, and against a RE-AUTHORED central difference of the quote side **9.376e-6 / 3.750e-7 /
1.500e-8** at h = 0.5 / 0.1 / 0.02 vol points — `h²` twice over. Value space: step θ by `dθ/dq·h`
and reprice without re-solving, closing on 1 from both sides linearly in h. **Two findings ride
along.** `.grad` standing on the quote leaves after an analytic chain is **0.31% to 2.33%** of the
answer — plausible-looking rather than the Monte Carlo path's six-orders-out-with-a-NaN, so
`bootstrap`'s clear is load-bearing here in a way no gate could previously see. And the
stationarity refusal now has to be reached **from the seed** (9.64e-3 on the four-quote block):
basin hopping ALONE lands at 5.85e-7 on this residual where the Monte Carlo one is at 2.9e10.
**And the re-solve oracle was tried again, on a solve that does reach its minimum — and it still
scatters.** Every re-solve lands inside the declared tolerance (8.6e-7 to 1.6e-5 against 1e-3) where
the Monte Carlo chain stopped at 8.24e3 — and the ladder is the Monte Carlo path's ladder to two
digits at the two coarse rungs (‖Δθ‖ 0.0387 / 0.0218, quotients 3.87 / 5.46, against 0.037 / 0.021
and 3.7 / 5.3), then blows out to ‖Δθ‖ 1.92 and a quotient of 960 at the finest one where a solve
changed basin. Cosines of 0.02–0.17 against a random direction's 0.209, and every re-solve lands
0.27–0.30 from θ\* whatever the bump. That settles that the flat directions were the whole
diagnosis and "stops short" never was. See
[Quote Sensitivities](quote_sensitivities.md#the-analytic-quote-side).

**Equity Heston–Nandi calibration — the CHAIN EMITTER is BUILT (`e159503`); the fit-side
engine work is the named follow-up.** `derivus_bloomberg/equity_chain.py`: chain discovery
through the package's own session seam with every response screened as untrusted evidence
(per-contract verdicts; a live SPX chain measured 8,000 asked / 3,729 believed / 4,271 refused
— 53%, led by stale and no-open-interest, with ZERO American contracts on the index), pillars
matched to listed expiries ONE-TO-ONE by nearest-claim-wins (a pillar the chain cannot serve
is DROPPED by name, never silently duplicated onto a neighbour's expiry — the review measured
73% of an objective landing on a single print before that rule), the undeclared-dividend carry
read as a MEDIAN over five two-sided parity pairs around its own implied forward with a band
refusal, weight = vega·√OI/(1 + spread/cap), the distinct-contract floor of eight counted
after snapping, premium quotes with the two-way carried, and the emitted block fitting through
the real component-HN bootstrap (ATM residual 4.4e-16). THREE ENGINE FINDINGS gate the
follow-up, in Known defects below: the family's `Volatility` reference is REQUIRED but never
read under `Quote_Type` Premium and a missing reference SKIPS rather than refuses — so a
chain-sourced fit cannot run without somebody's fitted surface, the exact circularity the
governance ruling exists to avoid; one `Discount_Rate` funds the calibrated forward AND
discounts the premium while the pricer's `calc_eq_forward` reads `EquityPrice.Interest_Rate`,
so a repo spread parts the calibrated and priced forwards; and the option-table row declares
no `Quoted_Bid`/`Quoted_Ask`/`Timestamp`, so a chain block cannot re-tick as values. The
ratified design stands below.

**The original ratification (2026-08-31).** The FX principle transfers but the SOURCE flips: FX calibrates off the desk's
own built surface because that surface IS the market (OTC delta quotes through `FXVolPrices`);
an equity's market is the LISTED CHAIN, and any equity surface is already somebody's fit to it —
so equities calibrate TO THE CHAIN, quoting PREMIUMS, not implied vols (a listed price is a
print; its implied vol is a convention — which forward, which discounting — and
`HestonNandiModelPrices` already accepts `Quote_Type` premium: "for an equity that set is
authored" was the design waiting for its emitter). The selection discipline transfers verbatim
— snap to actual listed contracts, count DISTINCT contracts after snapping with a refusal
floor, `Quote_Timestamp`/`Quote_Source` travel — with one addition: LIQUIDITY joins the vega
weight (open interest, two-sided quote, a spread cap), because half a chain is dead strikes.
The horizon rule flips with the product: equity autocalls run 3–5Y, so the ladder reaches the
product horizon and the TARGET FAMILY IS COMPONENT HN (a multi-year ATM term structure is
exactly what one ω cannot hold and the L-pillars can); the skew side is easier than FX — index
skew is steep, stable and one-signed, the original positive-`Gamma_Star` box's home market.
The genuinely new work is the FORWARD: the emitter must declare which curve feeds the carry
(futures-implied/dividend yield for indices; DISCRETE cash dividends on single names are a
declared modelling gap the daily recursion does not carry), or the calibration's forward
disagrees with the pricer's — the `Steps_Per_Year` mismatch one axis over. V1 scope: INDICES
ONLY (European exercise) — an American single-name chain REFUSES by name, since an American
premium is not the European premium the fit assumes. Governance note, recorded on purpose:
chain prints are external evidence, where calibrating to the desk's own authored surface would
be trader marks feeding model parameters feeding trader marks — under the spine's provenance
rules the chain-sourced block is the better fact.

*Two things the measurement found in the machinery around it.* The MC's numeraire bias is NOT
discretisation (refining 10-daily → daily does not move it): it is the CURVE's tenor grid, whose
first node is 1Y while `reduce_deflate` asks for a ten-day rate — adding 1D/1M/3M/6M collapses it
from −1.6e-2 to −1.1e-3, which is a fixture lesson every risk-neutral calibration in this file
inherits. And `num_batches` is in the Known-defects table below.

**Component Heston-Nandi (CJOW) is BUILT, end to end**, and it is a strict extension of the plain
family rather than a second one beside it: `utils.hn_component_*` (the pair recursion, its fused
log sub-step, the A/B/C recursion and the European/OSS closed forms),
`HestonNandiComponentModelParameters` on both sides (the price factor carrying an **L curve** whose
values are `bind='value'` leaves, and the calibration family),
`HestonNandiComponentImpliedSpotModel`, and a kit in `pricing.py` that all four OSS pricers walk so
a third GARCH family is a class and a dict row rather than a fifth branch in four pricers. The
long-run intercept is a CURVE — `ω_t = L_{t+1} − ρL_t`, anchored `q_0 = L(0)` so `E_0[q_t] = L_t`
exactly — fitted by an inner triangular bootstrap with the skew globals concentrated over it; the
whole construction, its two pins and its negative-omega guard are in
[Market Prices](market_prices.md#hestonnandi-component). Gates:
`tests/test_hn_component.py`, whose spine is the NESTING identity (φ=0, `L` flat: the component
closed form IS `hn_call`, measured 1.5e-13 relative, and the sub-step walks the plain path on
bitwise-identical draws). **4.79 s an outer evaluation** on the four-pillar USDZAR ladder, so the
declared 300-evaluation cap is 24 minutes and the fit reports itself CAPPED rather than claiming a
tolerance it did not reach.

Six things it owes.

- **`Quote_Sensitivity` is REFUSED by name**, and this is the roadmap row for it. The quote
  derivative would have to pass through the inner `brentq` on each `L` pillar by the implicit
  function theorem AND through the outer derivative-free search. The IFT half is tractable and is
  the same arithmetic `CalibrationSolve.backward` already runs — the residual is written once and
  differentiated twice — but the outer half is not a root find at all, so what a quote tick means
  for the skew globals has to be decided before it can be computed. Real work, and the family says
  so rather than answering zeros.

- **The fixing-jump sampler is v2, and the day-step is its oracle.** Every OSS interval currently
  walks `n_sub` daily sub-steps because the recursion is calibrated per trading day, which is
  exact and is most of the pricer's cost. A sampler that jumped fixing to fixing — drawing the
  interval's aggregate return and its terminal `(h, q)` from their joint law — would be the same
  speed lever the correlated sub-stepping already took on the scenario grid, and the acceptance
  test writes itself: it has to reproduce the DAY-STEPPED path's distribution, which is the thing
  that already exists and is already gated. Nothing about it is designed yet; what is recorded is
  that the oracle is not a closed form but the walk it would replace.

- **The quadrature bound is derived per PRICE, and the 4x for reusing one is still on the table —
  soundly this time.** The adaptive `φ_max` scan is 35–184 ms against 8–94 ms for the price itself,
  so a bound derived once and reused across a ladder is most of the calibration's wall clock. This
  family shipped exactly that for an evening — one scan on the shortest pillar, on the reasoning
  that more steps means more variance means faster decay — and it MISPRICED: past a parameter- and
  step-count-dependent point the component A/B/C recursion diverges rather than decaying, so a
  bound that is too LARGE integrates garbage. Measured, one converged optimum: a 126-step price is
  0.7353321384 at `φ_max` 128/256/512 (converged at 64, 256 and 1024 panels alike), 0.7323069671 at
  1024, 9.4e+55 at 2048, while the 21-step contract in the same strip wants 512. Because the ATM
  ladder is bootstrapped it repriced exactly anyway and the only symptom was the report's own
  recompute reading 3.5e-3 where it should read 1e-12. The trap is gated
  (`test_a_quadrature_bound_is_not_transferable_between_contracts`) and the shortcut is out. What
  would keep the speed and stay sound is a PER-PILLAR bound VERIFIED at the solved level: derive
  once at the bracket, re-derive at the root, re-solve only if it moved — one extra scan per pillar
  instead of one per brentq step. Not built. The same caution applies to the plain family's own
  "cheaper envelope" entry above, which was written before this was known.

- **The OSS row re-seeds at day zero, and the L curve makes that a bigger approximation than it
  was.** The plain model's known limitation F4 — `h` re-seeds to `H0` at every MTM row, so a row
  six months out prices its remaining horizon from the base date's variance state — carries over,
  and the component model adds a second axis to it: the intercept strip restarts at `ω_0`, so that
  row also prices under the FRONT of the `L` curve rather than the part of it the row has reached.
  On a term structure that moves 2 vol points over six months that is a real level error on the
  back rows of an exposure profile. The fix is not deep (the kit's day counter would start at the
  row's own trading-day offset, and the state would come off the outer path the way
  `inner_fork_seed` already does for the process) but it is a decision about WHICH state a row
  inherits, and it should be taken for both families at once rather than for this one alone.

- **Positivity has no certificate**, and that is the model's property rather than the
  parametrisation's: `q_{t+1} ≥ ω_t + ρq_t − φ − φγ₂²h_t` has no sign for free once `φ > 0`. The
  simulator floors at `utils.HN_COMPONENT_VARIANCE_FLOOR` (declared, measured at 2 of 8192 inner
  paths over 248 steps on the TARF gate) while the closed form integrates the unfloored law, so
  the two agree only where the floor is inactive — asserted in the gate rather than assumed.
  Whether the right answer is a floor, a bound on `φγ₂²`, or a different long-run innovation is a
  modelling decision nobody has taken.

- **The COARSE-GRID walk has no accuracy gate, and its plain sibling has one.**
  `utils.hn_component_correlated_substeps` is what the scenario process walks between exposure
  dates — whole trading days plus the fractional remainder, the framework draw riding the
  `sqrt(E[h·dt])`-weighted combination of the sub-step normals — and nothing measures its
  distribution against the daily-grid witness. The plain and GARCH(1,1)-t siblings do have that
  measurement (`gates/hn_pfe_stepping.py`: return quantiles at the PFE the exposure grid reads,
  four ways, oracle / coarse / daily / bridge), and it is exactly the two ingredients this function
  reimplements that it caught: the FRACTIONAL REMAINDER (the previous `round(f)` truncation cost up
  to **13% of interval variance**, −13% on the framework's own default CVA grid) and the FORWARDED
  MEAN that sets the correlation weights. The component version forwards a PAIR
  (`E[q_{j+1}] = ω_j + ρq_j`, `E[h_{j+1}] = E[q_{j+1}] + β(h_j−q_j)`) and slices a per-sub-step ω
  strip, so it has strictly more to get wrong and no oracle would be needed — the daily witness is
  the same comparison the plain gate already runs. Until it exists, the component exposure profile
  is gated for SHAPE (rows, dispersion, a finite CVA) and not for accuracy.

**The `Steps_Per_Year` mismatch — the plain model's known limitation F2, stated in
`pricing.pv_MC_Tarf` — is AMPLIFIED here.** A deal's valuation option agreeing with
the factor's calibrated clock by convention rather than by a check costs the plain model one
rescaled variance horizon; on this family it rescales TWO things at once, because the `L` knots are
in YEARS while `ω_t` is a per-STEP difference. A spy mismatch moves the number of steps per knot
interval (so a pillar's segment is walked at the wrong length) AND moves `ρⁿ` over that interval
(so the long-run component decays over the wrong number of steps) — the level error and the
persistence error, from one silent disagreement. The emitted block states the clock it was fitted
on, read off the field's own declaration; nothing yet refuses a deal that declares another.

`GARCHSpotModel` and the Heston-Nandi stack are built end to end; what remains is narrow:
batched-carry `hn_call` (stochastic-rate CVA raises loud today rather than mispricing silently —
and the already-hit leg raises the same refusal in its own name, which is the second caller that
would pay for the fix), the deal's `Steps_Per_Year` and the HN factor's calibrated clock agreeing
by convention rather than by a check, and the Malz surface lookup untested. `hit_value` staying
GBM under HN is **closed**: both vanillas take the declared model's
closed form on one discretisation, worth 15.8% per unit and 25.3% EPE / 26.9% PFE95 on the profile. That last one has
moved BACK: `test_fx_vol_prices` (an FX option off a `Malz` surface against the hand-authored one
it replaces) and increment 4's vega-chain gate (the vol the pricer reads at the option's own
log-moneyness against an independent Black vega — 7.039e6 against 7.032e6) both went with the
mock-built suite, so those readings are a record. **The Malz lookup is gated again**, from the
other side: `test_the_hn_ladder_is_ten_vega_weighted_points_on_the_surfaces_own_strikes` puts every
strike the FX calibration emits back through `pricing.calc_moneyness` — the pricer's own dispatch,
off the switches the block declares — and demands the built surface answer the vol the block
carries, to **1e-9 relative on all ten points across both expiries**, which is the term
interpolation in total variance as well as the single-expiry read. What it does NOT cover is the
`Steps_Per_Year` agreement above: the emitted block states the clock it was fitted on, read off the
field's own declaration, and a deal's valuation option still has to match it by convention.

**The calibration's wall time is the adaptive `phi_max` scan, and it is the one cheap lever on it.**
`hn_call` leaves `phi_max=None`, so every price re-derives the quadrature bound by doubling from 8
and running the A/B recursion twice per doubling — measured on this workstation at **0.45 / 1.13 /
3.42 s** for 21 / 63 / 252 steps against **0.08 / 0.22 / 0.87 s** with the bound pinned, i.e.
**75–82% of every Heston-Nandi option price**. An L-BFGS-B fit spends its whole life in that
function: `POST /book/hn` measured 288 s on the four-pillar ladder reaching six months (880 s on
an earlier 1M/3M one under the one-signed parametrisation — the iteration count dominates the step
count) and had not converged past 21 minutes on one reaching a year. The bound depends on the parameters and the step
count, so it cannot simply be hoisted out of the solve — but it is derived once per `hn_call` for a
whole strike vector already, and a cheaper envelope (or reuse across a line search, where the
iterate barely moves) is worth roughly a **4x** on every calibration in the stack.

!!! warning "The cleanup punchlist is stale by design"
    A separate, older sweep list exists outside the docs. Much of its target code was deleted with
    the RL stack and several entries are already done. **Verify any entry against the tree before
    acting on it** — an item that names a file which no longer exists is the normal case, not a
    surprise.

## A note on what this list is for

Most entries here were found by auditing work that already had passing tests. The recurring failure
was never bad reasoning — it was a gate that exercised **one point of a parameter**: only a bought
deal, only the default monitoring frequency, only one netting set, only the default valuation
option. Eight instances of that in a single work stream.

The second recurring failure is the one the barrier leg is named for, and it is a failure of
*review* rather than of gating: a unification absorbs N call sites into one seam and leaves their
siblings outside it, all the new testing points at the seam, and the sibling rots holding the only
copy of a formula that used to be read beside its twin. Three of the entries above are the same
mechanism at three different adopters. [Conventions](conventions.md#unification-siblings) states
the obligation that falls out of it.

So when picking something up: a mutant that survives your gate means the **fixture** is wrong, not
that the code is right. Vary the parameter the defect would live in, and check the mutant dies
before believing the test.
