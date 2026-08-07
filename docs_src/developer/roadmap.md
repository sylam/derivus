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
| A sibling fallback may name a factor discovery never fetched | each deal's `calc_dependencies` | Discovery iterates `factor_fields` over the RAW field and `get_fieldname` drops blanks, so a blank reference loads no factor. A fallback is only safe if it names one something ELSE already pulled in. `Discount_Rate ← Currency` is safe — 34 sites — because `Currency` is an `FxRate` and `dependant_fields` pulls its `InterestRate` transitively. Adding a fallback to a field whose sibling has no such edge silently resolves to whatever the sibling's chain did load. The one cross-leg instance (`FXForwardDeal.Sell_Discount_Rate ← Buy_Currency`) is fixed: both rates are `default=REQUIRED` with no fallback. |
| Four tables the Workbench cannot save | `derivus_jupyter.set_value_from_widget` | `set_repr` picks a deserializer from the `obj` token, and for an untagged table falls to a hardcoded whitelist of field NAMES. `Names`, `Sampling_Data_1`, `Sampling_Data_2` and `Barrier_Dates` are outside it and raise. The token table is per-field knowledge the `Row` now carries — the fix is to render from the declaration, not to add a fifth token. |
| A PARTIAL values patch drops the rest of a factor's value half | `Context.patch_market` | A shape-valued field splits inside itself, so the structural half of a curve is its coordinate columns alone. `apply_values` puts back only the fields the patch names, so a patch omitting one of the factor's value-bound fields leaves that field as its coordinates — a one-column curve that fails on the next read. It is outside the documented contract (the verb takes what `market_patch` emits) but it fails LATE and confusingly rather than by name. |
| Boundary scoping is not mutation-gated | `tests/test_boundary_pricer_events.py` | The fix is verified by measuring the term directly against CRN, but both two-netting-set gates measure the END-TO-END gradient, where the boundary term is a small fraction — so if it breaks later the suite stays green. Isolating it needs a portfolio where the correction dominates the smooth sensitivity, which is not a portfolio anyone runs. |

## Designed, not built

**Sensitivity estimators as first-class objects.** Every Greek should carry the estimator that
produced it — a `SensitivityProfile` per pricer — so a consumer can tell a pathwise derivative from
one carrying a boundary term. Related and also unbuilt: **calibration Jacobians**, so bumping a
market *quote* flows through bootstrapping rather than stopping at the calibrated factor; and
**Hessian-vector products** instead of materialising full Hessians.

**`fields.py` retirement — the Instrument store is now a VIEW.** Every deal type carries a per-class
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

Also remaining: the other seven stores are untouched.

!!! warning "`CreditNthToDefault`'s beta is calibrated at compile off bound content"
    `get_index_hazard_scale` → `utils.calibrate_index_hazard_scale` solves for `beta` at
    `calc_dependencies` reading the index `SurvivalProb.Curve`, the discount `InterestRate.Curve`
    and now `Recovery_Rate` — all three value-bound. A values patch therefore reprices that deal
    against a STALE beta. It predates this work (the two curves were bound in the first pass) and it
    is confined to the one deal type; the fix is either to move the calibration to eval, or to treat
    content that feeds a compiled calibration as structural.

The paired naming cleanup settles first, and is now done: the `MarketPrices` types the engine
matches, one `VolatilityGrid` in place of the three asset-class vol twins, and the IR prefix chain
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

What remains is the live-refill EXECUTE path and the service lift, and the refill is
**deprioritised**, by measurement rather than by taste: recompiling is relatively quick, and the
latency it would buy back is BASE VALUATION's — CMC is compute-heavy enough that compile time is
dwarfed by the run. The case that would change that is a future **STRUCTURING** calc, solving for a
strike, a margin or a vol: it iterates pricing on one changing input, and strike and margin are DEAL
fields, structural today. Such a calc either accepts a recompile per iterate or `bind=` eventually
extends to deal fields. A vol solve already patches cleanly.

**VALIDATE built**, and only that: `cx.validate()` returns `{'deals': {reference: [message]},
'factors': [name]}` — the authoring messages of every deal in the book, and every price factor it
names that the market data has no block for. It answers the want-list and nothing else, and it reuses
discovery (`discover_factors`) and not `calculate_dependencies`, which is where `find_models` mints the
`Price Models` dummies that make that method non-idempotent.

## Flagged, not authorised

`runtime` still carries free functions over the hedge bundle in two clusters (Objective and
Accounting) with `_UTILITY_OBJECTS` duplicated — the shape [Conventions](conventions.md) calls a
class waiting to happen. `DealStructure`'s recursions are in the same category.

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
