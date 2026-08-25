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
| Collateralised boundary counterfactual does not resolve | `NettingCollateralSet` gross-to-net chain + the boundary estimator | The collateralised CVA spot delta on the autocall fixture misses by double digits with a FLAT oracle (64 batches, CRN <1% spread, truth 5.22e-05): -10.9% without the ledger channel, +13.1% with it - on a bandwidth PLATEAU (0.02/0.05/0.10 read +12.5% flat to 0.25%, so this is not the unmeasured-plateau problem below: the estimator converges, off the truth). The earlier cash attribution is RETRACTED by measurement: the oracle reads the same delta with the engine cash on or off (5.22 vs 5.23e-05 - the relu kills a fired path's later exposure and the one-row-lagged balance offsets the `C_ts_te` window), and the cash-free WORLD overshoots its own oracle by +17.6% - the residual is the MTM-side counterfactual under the collateral chain, fixture-world-dependent in sign. What shipped meanwhile and is correct-by-contract: ONE registration per decision carrying its whole reach (`LatchedBoundarySet` with an own-row fired/survived override - two partial counterfactuals score differently from their sum under a kinked objective, `RowBoundarySet` deleted with its only adopter migrated) plus a per-decision ledger triple `(row, if-fired, if-not, booked)` that `net_from_gross` folds into `C_ts_te` through the declared settlement-risk windows. Uncollateralised gates are unaffected (1.68% / 0.06%). Strict xfail: `test_autocall_json.py::test_a_collateralised_cva_delta_carries_the_settled_coupon`, with the full measurement table. |
| TARF target pin | `pricing` (TARF block) | Material — fires on 27–61% of paths, 27% short uncorrected — but neither the estimator (13% bandwidth spread) nor the **oracle** (8.9% flatness) resolves better than ~10%, so it is gated structurally with no tolerance asserted. Do not tune one on: the oracle cannot see it either. |
| `pv_partial_barrier_option` | `pricing` | Excluded from the sensitivity work by decision; wants its own review. Also carries a suspected NaN — `limit` goes negative past `Barrier_Limit_Date` and is passed to `sqrt` unclamped. **Read, not run.** |
| A sibling fallback may name a factor discovery never fetched | each deal's `calc_dependencies` | Discovery iterates `factor_fields` over the RAW field and `get_fieldname` drops blanks, so a blank reference loads no factor. A fallback is only safe if it names one something ELSE already pulled in. `Discount_Rate ← Currency` is safe — 34 sites — because `Currency` is an `FxRate` and `dependant_fields` pulls its `InterestRate` transitively. Adding a fallback to a field whose sibling has no such edge silently resolves to whatever the sibling's chain did load. The one cross-leg instance (`FXForwardDeal.Sell_Discount_Rate ← Buy_Currency`) is fixed: both rates are `default=REQUIRED` with no fallback. |
| Four tables the Workbench cannot save | `derivus_jupyter.set_value_from_widget` | `set_repr` picks a deserializer from the `obj` token, and for an untagged table falls to a hardcoded whitelist of field NAMES. `Names`, `Sampling_Data_1`, `Sampling_Data_2` and `Barrier_Dates` are outside it and raise. The token table is per-field knowledge the `Row` now carries — the fix is to render from the declaration, not to add a fifth token. |
| Seventeen descriptors have no widget | `stochasticprocess` (Markov / VAR / basis models), `calculation` (`Hedging_Problem` maps, `CDS_Tenors`, `Scenario_Factors`) | `Transition_Matrix` is N×N, `Mean` and `Sigma_By_State` are length-N, `States` is a list of per-regime dicts, `Tradable_Instruments` is a deal map keyed by `Object` then by `Reference` — every one of those shapes is an OUTPUT, and `Table` declares fixed columns while `Container` declares fixed named children. `define_input` reads `element['col_names']` / `element['sub_fields']` unchecked, so the Workbench raises the moment it renders any process in the platinum world, or the hedging problem itself. Pinned by an exact-set gate that fails in both directions (the `Objective` / `Evaluator` / `Solver` blocks left the set when they declared their knobs — see `test_hmc_declared_knobs`). The fix wants a widget, not a schema change. |
| ~~The OSS carry strip is not the interval integral~~ **CLOSED** | `pricing.forward_carry_rate` | `carry[j]` is the AVERAGE annualised rate from `t` to fixing `T_j` and `dt[j]` the length of the interval **ending** at `T_j`, so `carry * dt` was that interval's integral only on a FLAT curve — and it drove the GBM drift of the barrier's own **simulation**, so `E[S_T] ≠ F(t,T)` on any sloped curve. Measured **4.276e-02** on the sibling fixture's sloped curve, now **2.220e-16**. `forward_carry_rate` differences the cumulative integrals and all four adopters — `sim_spot_oss`, `pv_MC_Tarf`, and BOTH branches of `pv_MC_AutoCallSwap` — take it in `theta` in place of the raw zero rates; the two correct sibling spellings are deleted, bit-identically (measured on the non-zero-carry fixtures). The strict xfail is a live assertion, `test_payoff_forward_survives_a_sloped_carry_curve`. |
| Four pricers, and seven of the eight closed-form barrier payoffs, are executed by no test | `pricing` (analytic barrier/option family) | Measured by `gates/pricer_branch_census.py`, ledgered in `tests/test_pricer_branch_ledger.py`: **64** branch arcs across the ten-pricer family that the whole suite never takes. Never run at all: `pv_partial_barrier_option` and `getpartialbarrierpayoff` (no fixture builds an `FXPartialTimeBarrierOption`), and `pv_american_option` (every equity fixture is European). `pv_MC_Tarf.bs_call_put_fwd` was dead rather than uncovered and has been **deleted**, with its ledger row. `getbarrierpayoff` selects on (direction, eta, phi, strike vs H) and every fixture reaching it is Down-and-Out / Call / K > H, the LAST elif of the OUT block, so all four knock-**IN** formulas and three of the four knock-OUT ones have never been evaluated. **Start with `pv_barrier_option`'s `if direction == BARRIER_IN:` body** — the analytic knock-in leg, zero lines executed, the same construction one function away from the leg that shipped at +1432%, and a fixture for it is one `Barrier_Type` string. |
| ~~The autocall has never been priced at a live carry~~ **CLOSED** | `QEDI_CustomAutoCallSwap` fixtures | `gates/fixture_degeneracy.py`: **119 runs** over the barrier/option/HN/TARF/autocall modules, not one at a non-zero rate or a non-zero carry. It is the OTHER adopter of the OSS seam, and its `called`/`knocked` state is the same class of outer path-state override that `hit_value` was — so no fixture that exists could tell a missing `dt` in it from a correct one. One exposure grid at a live carry closes it. First half: base valuations at r = 5%, q = 1% on sloped and smiley surfaces, which caught the strip reading -8.27% off its oracle. Second half now CLOSED by `test_autocall_json.py`'s credit-monte-carlo gates: an exposure grid at r = 4%, q = 1% with block-splitting live and the `terminationDate` latch carried across reporting rows — exactly the state this row said no fixture could see, and it WAS wrong when first priced (the 0.8/0.8/0.8 ledger above). |
| ~~The interval vol strip is read at a moneyness the deal does not declare~~ **CLOSED**; the per-fixing **smile** read is the open question | `pricing.forward_vol_strip` | `forward_vol_strip` read every fixing at its own FORWARD moneyness, hard-coded `use_forward=True`, mirroring `pv_MC_Tarf`. Its two other adopters, unlike the TARF, have a EUROPEAN LIMIT, and they declare `use_forwards = False` — so on any surface with a **smile** the simulation priced a different law from the quote the same pricer marks its European legs with. Measured, both exact where exactness is available: a one-coupon autocall is a closed-form digital and read **0.024428368300 against the declared 0.024490310460, -0.2529%, 0 ULP either side**; a never-knocking `Down_And_Out` read **1174.80 against Black at the declared quote 1163.9626, +0.948%, 8.3 standard errors**, and in-out parity **-11.03** where it now reads **-0.19 ± 1.30**, ten seeds, on both smiley surfaces. **Closed by internal consistency**: the deal's flag is threaded into the strip, the pricer reproduces its own European quote (0 ULP on the digital, +0.016% and 0.1 se on the barrier) and every repo fixture is **0 ULP** on value, profile, CVA and gradient — they are all flat in moneyness with `r = q = 0`. The TERM-STRUCTURE half was separable and is untouched: alternating only this flag in one process on the smile-free surfaces leaves the prices agreeing to **1.1e-15** relative. **What is still open is the modelling question**, and it is not a defect: a desk quoting sticky-*forward* moneyness wants the read that was removed, and reinstating it costs the six gates the mutation matrix in `test_vol_term_structure_strip.py` names — the declared-moneyness gate, both smiley European arms, both smiley parity arms and the smiley digital. Whoever picks it up picks up a **switch**, not a revert: the two conventions are both defensible and only one of them can be the pricer's own quote. |
| ~~Compo is broken under plain GBM in both OSS pricers~~ **CLOSED**; the ANALYTIC consumers stay half-adjusted | `pricing.calc_vol_adjustment` and its six consumers | TWO defects, named separately. `calc_vol_adjustment` returned the python float `0.0` as its Compo `b_adj` and both OSS call sites handed it to `torch.unsqueeze` — `TypeError`, deal SKIPPED, so no compo OSS deal had ever priced. And its `s_adj` handed `calc_fx_forward` a TENOR where every other call site passes ABSOLUTE days (the function subtracts the grid itself), mis-tenoring the fx forward off t0 and shaping it `(N,N,B)` — so `pv_european_option`, the one `s_adj` consumer, could not have priced a compo either. **Both closed**: compo now simulates the PRODUCT `S*X` at the quanto treatment's own deal-level fidelity — spot scaled by the cross (`spot_scale`), fx carry added per fixing (`carry_adj`, built on the same fixing matrix as the drifts), interval strip composed with the fx expiry vol (`compo_vol`, which also deduped the expiry-vol expression) — and the fx forward is per-row diagonal at absolute days. MEASURED on the JSON contract (`test_autocall_json.py`): a one-coupon compo autocall is a closed-form digital on the converted spot and the pricer lands on it to **4.8e-16 relative, both correlation signs** — the sorted-pair `check_fx_name` flip is in the oracle, so the convention is tested, not assumed. The old gate's fixture was itself degenerate: USD-scale strike/barrier against a compo spot of `S*X` left the down-and-out born dead and the fixed pricer reading 0.0 — a compo strike is a PAYOFF-currency quantity and the fixture now authors it as one. **Still open, silent**: `pv_barrier_option`, `pv_one_touch_option` and `pv_discrete_asian_option` adjust the VOL only — no fx scale, no fx carry — so a CONTINUOUS-monitored compo barrier, a compo one-touch and a compo asian price half-adjusted numbers without raising; and `pv_european_option`'s repair is untested (no compo european fixture exists). The compo SMILE coordinate is also undeclared — every fixture is flat — same class as the sticky-forward question above. |
| `pv_one_touch_option`'s `Payment_Timing` chain falls through silently | `pricing.pv_one_touch_option` | It tests `'Expiry'` and `'Touch'` with no `else`, so a third value prices as whatever the last assignment left instead of refusing. Wants a typed refusal, not a fixture. |
| ~~The autocall's settled cashflow is a valuation, booked per coupon, per unit and unsigned; the TERMINATION CARRY it exposed~~ **BOTH CLOSED** | `pv_MC_AutoCallSwap` (no-averaging loop) | Measured on a job document: three coupon dates, threshold 0.01 so the deal autocalls at the first with certainty, one coupon of 0.08 on 10 units. The ledger read **0.24 / 0.8 / 0.8** where **0.8 / 0 / 0** pays, and `Units=1`, `Units=10` and `Sell` all booked the same number. Four causes, all **closed**: `tau` is per ROW but the settle sat in the COUPON loop, so it fired once per coupon and `cash_settle` accumulated; it booked `P`, the accumulated VALUE, not the payment; `nominal` scaled the mark while `cash_settle` got `value` raw; and `terminationDate` was stamped inside `sim_spot` and never returned, so every block re-priced and re-paid the deal as though it had never fired. **The carry**: the latch is a by-product of the simulation (stamped where a fixing is observed, off the scenario's own spot), handed to the next block's theta; the accumulators run ALIVE and the latch masks the exits, `P` being homogeneous in its initial weight - bit-identical while nothing fires, and the alive pass IS the `untriggered` branch the registration needs. **The registration is TWO sets, not a migration**: the decision's reach has two halves - the fired/survived fork on its own row (`RowBoundarySet`, a hard indicator nothing smooths) and the carried latch killing every later row (`LatchedBoundarySet`, alive against zero, obs strictly-before at block granularity, the barrier's spelling). They share their gaps, do not overlap, and sum to the whole jump. MEASURED on the legacy CRN gate (1024x16x256): AAD +2.07176e-06 against CRN best +1.89664e-06 - the 87.39% Row-only disagreement falls to **8.45%** on large rungs flat to ~2%. On the JSON CVA gate (`test_autocall_json.py`, 1024x4x256): disagreement **1.68%**, and each half suppressed alone kills it at **+73.83% / -72.65%** - opposite signs, ~43x the corrected residual. Ledger and profile gates now PASS un-xfailed: 0.8/0/0 and 0.79206/0.8/0/0. **Still open**: the AVERAGING branch cannot carry the latch - its termination is a smoothed per-inner-path weight with no crisp per-scenario decision (its dead discarded latch line is deleted); and a LAGGING-payment schedule (coupon paying after its fixing) would have its pending window zeroed by the carry - no fixture reaches either, and the latch marker (the fixing index that killed the path) is the hook a pending-window exemption would key on. |
| The FX accumulator's dead branch is zero, and a leveraged deal's pending head is not | `pv_MC_Accumulator` | `triggered = zeros` omits the fixings a knocked deal accrued before the breach whose settlements have not landed. Now MEASURED rather than named: exact (reconstruction 4.5e-13) wherever the settlement lag is shorter than the fixing spacing — the T+2 spot-FX default at weekly-or-longer schedules — and 1.07% of the profile at a 45-day lag on 20-day fixings. The claim that it is one-signed is FALSE and was measured false at +54.06: the pending head is `N1*relu(S-K) - N2*relu(K-S)`, negative on an OTM fixing whenever `LeverageNotional > Underlying_Amount`, which is how this product is ordinarily written. What holds is containment — the error lives only in cells the latch calls dead. Both are gated (`test_a_pending_settlement_is_what_the_zero_dead_branch_costs`). An exact branch is a per-decision head profile, i.e. a fourth `BoundarySet` shape, deliberately not built ahead of a caller who needs the accuracy. A grouped-settlement cashflow-ledger assertion is still owed. |
| ~~HW1F's λ and quanto legs decayed twice~~ **CLOSED** | `HullWhite1FactorInterestRateModel` | Two defects, one function, both verbatim from the original import and both invisible because `Lambda` and `Quanto_FX_Correlation` are 0.0 in every fixture. The drift legs accumulated `diff(e^{−αs}·K)` — which telescopes to `e^{−αt}K(t)` — and then decayed again at assembly: ratio-to-exact of exactly `e^{−αt}` at every node (12 digits over a 10y grid; 63% attenuation at 10y for α=0.10), where HW2F and the hazard model both decay once. And the quanto-vol curve was read through `.array.T`, handing the integrator the knot PAIRS as (tenors, values) — garbage for any real curve. The rule, for any HW-family process: the increment fed to the cumsum must carry `e^{+αs}`, never `e^{−αs}` — `hw_calc_H`/`hw_calc_IJK` already put the `+α` in the integrands and the single `e^{−αt}` lands once, at assembly. Gated by `test_hw1f_lambda_and_quanto_legs_decay_once`, the suite's first fixture with BOTH knobs live, held to brute-force quadrature (oracle lower limit = the grid's own first row — the state conditions on x(t₀)=0); each half re-broken alone turns it red. The docstring's `A(t,T)` carried the OPPOSITE typo (`e^{−αT}` for the code's correct `e^{−αt}`, verified to 4.8e-15) and is fixed with it. |
| A collateralised backward gradient is not reproducible | `NettingCollateralSet` backward, recompute node OFF | One gradient entry takes two distinct float64 values from bit-identical inputs — a nondeterministic GPU reduction, not a graph defect. It bounds how tightly any collateralised sensitivity gate can be pinned, and nothing has been done about it. |
| ~~A zero-length OSS step is not a martingale~~ **CLOSED**; the exact indicator at `times == 0` is what is left | `pricing.sim_spot_oss`, `pricing.sim_spot` (averaging) | `drift` was taken from the **unclamped** variance while `vol = sqrt(var.clamp(min=1e-4))`, and every reporting row that IS an observation date opens with `dt = 0`, so that step was priced as a σ=1% lognormal kick with no Itô correction: `E[S_j \| S_{j−1}] = S_{j−1}·exp(+5e−5)`. Both consumers now read ONE `var`, so the incoherence is unrepresentable rather than detected — measured on the monthly exposure fixture, the barrier profile moves **-0.0215%**, its CVA **-0.0240%** and every one of the 13 CVA-gradient entries by up to **7.5e-4 relative**, and **no gate in the repo could see any of it**. The same edit conditions the floor itself on `dt == 0`: it was unconditional, which was harmless only while the vol-strip defect handed every interval `sigma(T)^2`, and a correct strip lets it bind wherever `sigma_fwd < 0.01/sqrt(dt)` — 114 of 365 daily intervals on an upward 0.12→0.24 surface, **+1.584%** on a `Down_And_Out` and **-6.870%** on an `Up_And_Out` against a floor-free oracle, now +0.183% and +0.403% (`test_a_daily_monitored_barrier_is_not_priced_at_the_variance_floor`). The autocall's averaging branch carried the identical expression and is fixed identically: value **-0.0082%**, CVA **-0.0076%**, CVA-gradient sum **-4.83%**. **The attribution is exact, not inferred**: on that fixture the intervals never reach the floor, so conditioning it is bit-exact in value AND gradient, and putting the incoherent drift back on the fixed tree reproduces HEAD's profile and CVA **bitwise** on both deals — the drift is the whole of what this seam moved in value, and the strip is the whole of what it moved in gradient. The `no_averaging` branch never had a floor — it already resolves `dt <= 0` with an exact indicator. **What the `dt == 0` clamp buys is the GRADIENT**, measured by deleting it: every value stays finite, 11 of 13 CVA-gradient entries go NaN (7 of 13 on the averaging deal) and six gates die. **STILL OPEN**: that step is simulated with 1% of lognormal vol instead of being resolved by an exact indicator, which consumes a Sobol draw and so moves every barrier constant. Cannot matter for a down-and-out call; **will** matter for a digital or a cash rebate. |
| The boundary estimator has no measured bandwidth plateau | `pricing.stochastic_boundary_correction` | Local-linear weights are meant to hold the estimate still over a range of bandwidths, and that is the only acceptance criterion the docstring offers, but it has only ever been read at 512 and 1024 paths where it does not settle. The documented operating point is 32768 and nobody has run it there. (The two gates that used to pass with their subject deleted are re-baselined onto the declared grid; the HN barrier gate is now 1.18% against a 6.19% suppression mutant, and the discrete-barrier profile gate is replaced by a bit-exact rebate ledger.) |
| Boundary scoping is not mutation-gated | `tests/test_boundary_pricer_events.py` | The fix is verified by measuring the term directly against CRN, but both two-netting-set gates measure the END-TO-END gradient, where the boundary term is a small fraction — so if it breaks later the suite stays green. Isolating it needs a portfolio where the correction dominates the smooth sensitivity, which is not a portfolio anyone runs. |

## Designed, not built

**Sensitivity estimators as first-class objects.** Every Greek should carry the estimator that
produced it — a `SensitivityProfile` per pricer — so a consumer can tell a pathwise derivative from
one carrying a boundary term. Related and also unbuilt: **Hessian-vector products** instead of
materialising full Hessians.

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
sum: 2.243453e4 + 8.071709e4 on a stacked FX pair, measured); neither backward supports
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
`fields.py`'s 1,932 lines. The engine never reads any of it (`construct_instrument` takes the raw
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

**A section owns its descriptors.** The store was keyed by field NAME across all 47 deals, which
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
own `validate()`. An audit of all 45 deal types found 16 cross-field rules over 6 predicate forms,
which is why they are code rather than a declarative `one_of=`; 17 more key on `base_date`,
`self.options` or a resolved price factor and can never be evaluated at authoring time at all.

Nothing in the valuation path calls it and a message never stops a deal pricing — the engine still
fails where it always failed. Four rules are stated so far (`Cash_Payoff` on the binaries,
`Settlement_Amount ⇒ Settlement_Date`, `Collateral_Rate ⇒ Funding_Rate`, and the inflation
value-or-date pair). `cx.validate()` now returns them alongside the factor want-list, keyed by each
deal's `Reference`.

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
and the half a patch carries — 32 fields over 20 types, each declared from its consumption site. A
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
(`tests/test_calculation_defaults.py`). Four disagreements surfaced and settled: `Random_Seed`
(engine said 1, declaration says **5120**), `Dynamic_Scenario_Dates` and `Generate_Cashflows`
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
`Objective` / `Evaluator` / `Solver` blocks declare their ~40 knobs on the same terms
(`test_hmc_declared_knobs` holds declared-vs-read in both directions, engine side included).

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
into the block, and only for `InterestRate` and `InflationRate`. So what the two rows restrict is
the OPT-IN, which is per-class, and the menu for a type outside it would offer a setting the
engine drops on the floor.

Those two classes therefore declare `interpolation_methods`, one shared `INTERPOLATION_METHODS`
object rather than a row copied onto each, and the map is `schema.emit_interpolation(riskfactors)`
— the same shape as a process naming the `factor_types` it drives. Two gates pin it to the engine:
the menu's keys are the type list `construct_factor` routes, parsed by AST in both directions, and
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

`market_factor_type` is a class attribute on all five families and the engine compares against it,
so the two gates that held the declared types to the literals the engine matched would now be
tautologies. What replaces them is the discipline that makes them tautologies: **no bootstrapper
owns the text**, parsed by AST, the same shape as
`test_instruments_call_resolvers_not_factor_types`. A second gate holds every `quote_instruments`
name to a declared deal type, which is the reuse-by-reference rule made checkable.

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
[Quote Propagation](quote_propagation.md). It also names the one place the split does not yet
reach: `Market Prices` is inside `plan_hash` whole, so a moved QUOTE is a new plan id even though
the engine now carries it without recompiling anything. Closing that means partitioning a market-
price block the way `partition_factor` partitions a factor, for every family at once.

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
cheap — base valuation and a future **STRUCTURING** calc (solve for a strike, a margin or a vol).
CMC is compute-heavy enough that compile time is dwarfed by the run, so it loses nothing by waiting;
it must gain the verbs on the same terms when they land. The structuring calc iterates pricing on
one changing input, and strike and margin are DEAL fields, structural today — such a calc either
accepts a recompile per iterate or `bind=` eventually extends to deal fields. A vol solve already
patches cleanly.

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

What remains of the service: SSE for progress, a cost estimate that reads the real grid rather than
a segment count, and auth with budget caps.

**The Excel add-in is the first real client.** `excel_integration/service_client.py` is a plain
`requests` client of the verbs and imports neither `xlwings` nor `derivus` — it is the HTTP binding
a marimo notebook or a plain script uses as readily as the workbook, and a gate reads its import
statements to keep it that way. `worker.py` and `queue_clients.py` are DELETED, not deprecated: the
service is the worker and the queue, and the file-queue settings went out of `config.py` with them.
Solace returns later as a second transport in front of the same verbs, not as a second queue.

Two things stayed in process, each for a stated reason. `RF_SOLVE_*` iterates a pricing run on one
changing DEAL field, which is the STRUCTURING calc this roadmap has yet to build; a deal field is
structural today, so every iterate is a fresh compile and a round trip per iterate would buy
nothing. And `RF_*_PORTFOLIO` builds its job from the sheets through `portfolio_service`, which
still reads `schema.mapping` directly — migrating that to `GET /schema`, which publishes exactly
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
literal — is now gated (`test_instruments_call_resolvers_not_factor_types`), with one held
exception: `get_implied_correlation`'s two callers build type-prefixed correlation-name tuples
(`('EquityPrice',) + …` in `Deal.check_option_data`, `('FxRate',) + …` in
`EnergySingleOption.calc_dependencies`). The fix is two single-caller wrappers
(`get_equity_fx_correlation`, `get_fx_reference_correlation`), which brushes the
no-abstraction-ahead-of-a-second-caller rule — held until a third correlation pair shows up or the
rule is judged to outrank it. The gate does not cover tuple literals, so this stays a note.

**`Flot` is a plotting library, not a type.** The widget token the schema vocabulary files a
`Curve` descriptor under (`schema.py`) is the jQuery-Flot chart widget's name, surviving from the
`fields.py` era. What it denotes is a curve OBJECT — a 1d list of indexed numbers (an
interest-rate curve, a commodity curve, a vol column) — and the token should say so, with any
legacy spelling owned by the front-end mapping. Lands with the next schema-design pass.

**Inline comment density.** The boundary-correction work left ~12 inline blocks of 4-11 comment
lines, several outweighing the code beneath them. House style is detailed docstrings, 2-3 lines
maximum inline, and never more comments than code. The material is right - the reasoning, the trap,
the measurement - it just belongs in the function's docstring or the commit message. Worst
offenders: `pv_discrete_barrier_option`'s hit-mask and rebate blocks, `sim_spot_oss`'s terminal
digital, `NettingCollateralSet.post_process`'s `net_from_gross`.

**The compounding-leg shape check.** `pv_float_cashflow_list` selects the compounded-in-arrears
path by comparing reset count to cashflow count — a shape encoding of intent, set up at
`calculate_dependencies`. It works and is now documented
([Quote Sensitivities](quote_sensitivities.md#curve-contracts)), but an explicit signal on the
compiled cashflow object would say the same thing without the inference.

## Model punchlist

`GARCHSpotModel` and the Heston-Nandi stack are built end to end; what remains is narrow:
batched-carry `hn_call` (stochastic-rate CVA raises loud today rather than mispricing silently —
and the already-hit leg raises the same refusal in its own name, which is the second caller that
would pay for the fix), `HN_Steps_Per_Year` hardcoded to 252, and the Malz surface lookup thinly
tested. `hit_value` staying GBM under HN is **closed**: both vanillas take the declared model's
closed form on one discretisation, worth 15.8% per unit and 25.3% EPE / 26.9% PFE95 on the profile. That last one has
moved: `test_fx_vol_prices` prices an FX option off a `Malz` surface against the hand-authored one
it replaces, and increment 4's vega-chain gate now holds the vol the pricer reads at the option's
own log-moneyness against an independent Black vega — 7.039e6 against 7.032e6. What is still
untested is the lookup ACROSS expiries, where the term interpolation runs in total variance.

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
