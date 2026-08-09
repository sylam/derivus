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
| Eighteen descriptors have no widget | `stochasticprocess` (Markov / VAR / basis models), `calculation` (`Hedging_Problem`, `CDS_Tenors`) | `Transition_Matrix` is N×N, `Mean` and `Sigma_By_State` are length-N, `States` is a list of per-regime dicts, `Tradable_Instruments` is a deal map keyed by `Object` then by `Reference` — every one of those shapes is an OUTPUT, and `Table` declares fixed columns while `Container` declares fixed named children. `define_input` reads `element['col_names']` / `element['sub_fields']` unchecked, so the Workbench raises the moment it renders any process in the platinum world, or the hedging problem itself. Pinned by an exact-set gate that fails in both directions. The fix wants a widget, not a schema change. |
| Boundary scoping is not mutation-gated | `tests/test_boundary_pricer_events.py` | The fix is verified by measuring the term directly against CRN, but both two-netting-set gates measure the END-TO-END gradient, where the boundary term is a small fraction — so if it breaks later the suite stays green. Isolating it needs a portfolio where the correction dominates the smooth sensitivity, which is not a portfolio anyone runs. |

## Designed, not built

**Sensitivity estimators as first-class objects.** Every Greek should carry the estimator that
produced it — a `SensitivityProfile` per pricer — so a consumer can tell a pathwise derivative from
one carrying a boundary term. Related and also unbuilt: **Hessian-vector products** instead of
materialising full Hessians.

**Calibration Jacobians — increment 1 is BUILT.** Bumping a market *quote* now flows through
bootstrapping rather than stopping at the calibrated factor: one `backward()` reports `dV/dq`
beside `dV/dθ`. [Quote Sensitivities](quote_sensitivities.md) is the page — the graph audit that
made it possible, the quote-side overlay, the IFT contract, the attachment, the precision seam, the
validation triangle and the non-goals. Turned on per curve by the declared field
`Quote_Sensitivity`; the solved numbers are bit-identical either way.

It shipped carrying two known traps, and both are now **unrepresentable** rather than fixed, because
`TensorSchedule.bind` gave the tensor half a birthday — see
[the schedule lifecycle](calc_lifecycle.md#the-schedule-lifecycle). `dual` and `merged` memoized
under one key and could serve each other's copy: `bind` mints the one copy and `merged` is deleted —
`dual` is the accessor, so there is no second memo to collide with. `pv_fixed_cashflows` memoized its payment tensor in
`Factor_dep`, which outlives the copy it was built from and froze the first evaluation's overlay:
it lives in the schedule's `derived`, which `bind` mints and re-mints with that copy. The two gates
that held them in place assert the design instead.

What remains is increment 2: the same IFT contract around the HW2F swaption-vol calibration
(`DV_Bootstrap`), where the fixed point is the stationarity of a least-squares loss rather than a
root, so backward needs the Gauss–Newton Hessian and the dropped residual-curvature term has to be
documented with its tolerance. FX vol follows the same shape. Two smaller ends left deliberately
open and recorded on the page's non-goals: there is no report FORMAT for a quote delta (it lands on
the leaf in `Config.quote_leaves`, and `make_factor_index` wants a tenor grid a quote does not
have), and `CalibrationSolve.backward` does not support `create_graph`, so there is no second
derivative in quote space.

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
example. Their descriptor defaults are the engine's own `.get` defaults, which is the only
defensible source for them.

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

**The compounding-leg shape check.** `pv_float_cashflow_list` selects the compounded-in-arrears
path by comparing reset count to cashflow count — a shape encoding of intent, set up at
`calculate_dependencies`. It works and is now documented
([Quote Sensitivities](quote_sensitivities.md#curve-contracts)), but an explicit signal on the
compiled cashflow object would say the same thing without the inference.

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
