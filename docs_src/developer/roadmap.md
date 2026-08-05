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
| Collateralised autocall boundary correction | `utils.RowBoundarySet` | An autocall settles a coupon when it fires, and a collateralised exposure reads that ledger through `C_ts_te`, but `gross_to_net` takes only an mtm delta. Shipped as a strict xfail naming the reproduction. |
| TARF target pin | `pricing` (TARF block) | Material — fires on 27–61% of paths, 27% short uncorrected — but neither the estimator (13% bandwidth spread) nor the **oracle** (8.9% flatness) resolves better than ~10%, so it is gated structurally with no tolerance asserted. Do not tune one on: the oracle cannot see it either. |
| `pv_partial_barrier_option` | `pricing` | Excluded from the sensitivity work by decision; wants its own review. Also carries a suspected NaN — `limit` goes negative past `Barrier_Limit_Date` and is passed to `sqrt` unclamped. **Read, not run.** |
| Boundary scoping is not mutation-gated | `tests/test_boundary_pricer_events.py` | The fix is verified by measuring the term directly against CRN, but both two-netting-set gates measure the END-TO-END gradient, where the boundary term is a small fraction — so if it breaks later the suite stays green. Isolating it needs a portfolio where the correction dominates the smooth sensitivity, which is not a portfolio anyone runs. |

## Designed, not built

**Sensitivity estimators as first-class objects.** Every Greek should carry the estimator that
produced it — a `SensitivityProfile` per pricer — so a consumer can tell a pathwise derivative from
one carrying a boundary term. Related and also unbuilt: **calibration Jacobians**, so bumping a
market *quote* flows through bootstrapping rather than stopping at the calibrated factor; and
**Hessian-vector products** instead of materialising full Hessians.

**`fields.py` retirement — Instrument store declared, not yet authoritative.** 45 of the 47 deal
types now carry a per-class `fields` list (`schema.py`), and `schema.emit_instrument` rebuilds the
three-level `fields.mapping` shape from them byte-identically, gated per type/section/descriptor.
The engine never reads `fields.py` — `construct_instrument` takes the raw JSON — so none of this can break
valuation; the blast radius is the Workbench, the docs generator and the Excel add-in.

The schema's inheritance turned out to be composition of named field GROUPS, not the class
hierarchy: `FXAdmin` is shared by eight deals with no common base and `Admin` by all 47, so groups
are module-level `Group` constants a class lists. An MRO-based design cannot express that.

Remaining: flip the direction so `mapping['Instrument']` is GENERATED from the classes and the
hand-written dict goes; `SwapBasisDeal`/`SwapCurrencyDeal` have no class to hold a declaration and
need the same ruling as the strict xfail they already sit under; `bind=` (value-versus-structural
patching) is designed but unbuilt; the other eight stores are untouched.

The paired naming cleanup settles first, and is now done: the `MarketPrices` types the engine
matches, one `VolatilityGrid` in place of the three asset-class vol twins, and the IR prefix chain
(`InterestRate.USD.LIBOR`) **grandfathered** by decision rather than migrated to explicit spread
declarations. `Observed_Factor` and the type-switched `ObservedBasis` tail closed themselves —
`nested_fields` already wires FX and equity alongside commodity, and all four
`calc_factor_code_chain` call sites agree.

**PREPARE / EXECUTE.** A content-hashed `plan_id` from the structural projection, `EXECUTE` = plan
plus a values patch, `VALIDATE` returning the whole want-list, and `(plan_hash, values_hash,
engine_version, seed)` making every reported number replayable. Sequencing undecided.

## Flagged, not authorised

`runtime` still carries free functions over the hedge bundle in two clusters (Objective and
Accounting) with `_UTILITY_OBJECTS` duplicated — the shape [Conventions](conventions.md) calls a
class waiting to happen. `DealStructure`'s recursions are in the same category.

## Tidy-ups

**Inline comment density.** The boundary-correction work left ~12 inline blocks of 4-11 comment
lines, several outweighing the code beneath them. House style is detailed docstrings, 2-3 lines
maximum inline, and never more comments than code. The material is right - the reasoning, the trap,
the measurement - it just belongs in the function's docstring or the commit message. Worst
offenders: `pv_discrete_barrier_option`'s hit-mask and rebate blocks, `sim_spot_oss`'s terminal
digital, `NettingCollateralSet.post_process`'s `net_from_gross`.

## Model punchlist

`GARCHSpotModel` and the Heston-Nandi stack are built end to end; what remains is narrow:
batched-carry `hn_call` (stochastic-rate CVA raises loud today rather than mispricing silently),
`hit_value` staying GBM under HN in scenario mode (a documented discontinuity),
`HN_Steps_Per_Year` hardcoded to 252, and the Malz surface lookup untested.

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

So when picking something up: a mutant that survives your gate means the **fixture** is wrong, not
that the code is right. Vary the parameter the defect would live in, and check the mutant dies
before believing the test.
