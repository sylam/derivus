# Conventions

House rules for the derivus codebase. A rule marked **Held by** goes red under the named test or gate
if it is broken; one marked **Not gated** is held by nothing.

## JSON is the contract {#json-is-the-contract}

The job JSON is the whole program: a script does `import derivus as rf`, `cx.load_json(...)`, `cx.run_job(...)`, with no internal imports and no monkey-patching, since behaviour reached from outside the JSON is behaviour no other job can state. A feature ships behind a switch whose default stays bit-identical to the old path until the desk moves it on measured numbers, leaving the replaced estimator one value away (`Branch_And_Weight: 'No'` is the crisp path). **Held by** `tests/test_branch_and_weight.py::test_the_default_is_the_switch_on_a_real_document`. The format is generated from `schema.mapping` ([JSON Configuration](../json/index.md)).

- **No defensive checks.** Code assumes a contract field is present and fails loud (`KeyError` naming the factor via `check_tuple_name`); a guard turns a malformed job into a plausible wrong number.
- **A variant is a separate JSON document**, not a flag threaded through signatures.
- **MarketData files are data-only** — no `Description` or commentary fields.

## A declared default is a convention or a placeholder {#declared-defaults}

`F(..., convention=True)` says the default is what omission MEANS — `Pay_Timing: End`, a null calendar, a blank `Discount_Rate`. `schema.DealFields` completes a convention on a read by name (`field['Cash_Rebate']`), so an author writes only what differs. Unflagged, a default is a placeholder — `Swap_Rate: 0.0`, a blank date — what a panel shows and nobody means by leaving it out: a deal that does not state one, or states it null or blank as a front end sends an empty box, is refused by name at booking (`"Swap_Rate is not stated"`, `schema.validate_instrument`), and a placeholder reaching a pricer still raises its `KeyError`. A completion answers a read and never enters the block: `.get`, `in`, iteration, the JSON round trip, `plan_hash` and the factor universe see exactly what the author wrote. A field that falls back reads through `DealFields.read(key, otherwise)`, its one spelling: the value stated, else the type's convention, else `otherwise`, an omitted key and a blank alike on a type declaring the key or none (`Deal.check_factor_name`, `payoff_ccy`, `payoff_type`, `float_index`, whose blank `Index_Day_Count` is the floating leg's own); a placeholder is read by name. Anything authoring a deal (a benchmark emitter, the structures runner) writes the terms and the required fields, and a convention only where it differs from the declaration. **Held by** `tests/test_declared_defaults.py::test_a_default_answers_a_read_and_never_enters_the_program`, `::test_an_economic_field_is_refused_by_name_and_does_not_price`, `::test_a_placeholder_stated_blank_is_refused_by_name_and_does_not_price`, `::test_an_equity_option_omitting_its_surface_is_refused_by_name_and_does_not_price`, `::test_a_job_document_keeps_its_plan_hash_and_its_factor_universe`, `tests/test_interest_rate_prices.py::test_a_blank_index_day_count_is_the_floating_legs_own`. A calculation block is judged by the same rule (`schema.validate_block`), completed first as a run completes it, so what it can lack is a `REQUIRED` field such as its `Currency`, which a run refuses by name, or a placeholder declared blank such as its `Base_Date`; a blank that means none, `HedgeMonteCarlo`'s `Calendar`, is a convention. **Held by** `tests/test_loud_refusals.py::test_a_calculation_stating_no_currency_is_refused_by_name`, `tests/test_declared_defaults.py::test_a_calculation_omitting_a_blank_default_is_refused_unless_the_blank_means_none`. What a block STATES is judged by `schema.stated_messages`: a key its owner does not declare, a blank where a menu wants a choice, a value that is not what its field declares, a container's keys against its children. A calculation override or a saved calculation is judged by it alone, decoded as a run reads it, its absences being the merged block's that the run judges whole (`service.judge_calculation`); a structure's ticket by `validate_block` and then by it, against the variation it selects, the other variations' keys admitted (`structures.declared`). **Held by** `tests/test_service.py::test_a_calculation_override_is_judged_against_its_own_declarations`, `tests/test_structures.py::test_what_a_variation_takes_is_required_and_what_no_form_takes_refuses`. Every top-level declaration carries a default or `REQUIRED`, a null one being a convention's (`ChainedBasisModel.Next_Const`, null the bridge premium's intercept), so its reader indexes the completed block; only a table's columns and a container's children declare none, and `schema` refuses at import a store that publishes one. **Held by** `tests/test_schema_emission.py::test_every_top_level_field_declares_a_default`. A container declares no default of its own: it publishes what its children offer (`schema.offered`) in their wire form, a blank table `'null'` and a credit support list its token, a read completes it in engine form (`schema.engine_default`), and a container the author states completes the conventions its children declare, as the deal completes its own and a collateral row the conventions its haircut columns declare (`NettingCollateralSet.haircuts`). **Held by** `tests/test_schema_emission.py::test_a_containers_default_is_what_its_children_offer`, `tests/test_collateral.py::test_a_set_omitting_its_support_amounts_and_assets_holds_a_zero_csa`, `tests/test_declared_defaults.py::test_a_stated_container_completes_the_conventions_its_children_declare`.

## `globals()` dispatch {#globals-dispatch}

Constructors dispatch on class name through the module `globals()` — `construct_calculation`, `construct_instrument`, `construct_factor`, `process_class`, `construct_calibration_config`, `family_class` — so the class name is the dispatch key: **one class per concept in a file**, and a new type is a new class, never an edit to a dispatcher. An unknown deal `Object` loads as `{'Object': name}`, which `Config.validate` reports and the compile refuses.

## Registries, not functions {#registries-not-functions}

Extension points are data: a new type is a row or a class attribute the engine already iterates.

- Factor discovery is three dicts in `config.py` ([Dependency System](dependency_system.md)); process wiring and calibration are registered in `Model Configuration` and `calibration_config.json`.
- Every JSON schema is a per-class `fields` list, and every store but `System` is the view `schema.emit_*` builds from those declarations at import, never a second copy.
- Where a store's key is not the class name, **the class declares the bridge**: `factor_types` on a process, `model_type` on a calibration, `market_factor_type` on a bootstrapper (`FXVol` reads `FXVolPrices`). No rule recovers one name from another.
- `Process_factor_map` and `Interpolation_factor_map` are the same declarations read sideways; a market-price quote references the `Instrument` declarations rather than restating them ([Market Prices](market_prices.md)).

**Held by** `tests/test_schema_emission.py::test_every_store_is_the_emitted_view`, `::test_the_calculation_types_are_what_run_job_dispatches`.

A function switching on a magic-string type, or a parallel dict passed beside a primary operand, is a class or registry waiting to happen: strengthen the existing primitive.

## Documentation and doc generation {#documentation-and-doc-generation}

Developer-facing model/deal/process docs are **class attributes**, harvested at build time by `derivus_docs.py`:

- `documentation` (a `(section_name, [md_lines])` tuple) on classes in `stochasticprocess.py` / `instruments.py` / `bootstrappers.py` / `calculation.py` and the three `hedge_*.py` modules auto-publishes the Theory / Valuation / Bootstrapping / API / Hedging pages. Read **own-attr-only** (`cls.__dict__`, not MRO), so an alias subclass does not re-emit the parent's page.
- `schema.mapping` alone drives the JSON reference tree (`generate_json_docs`): one generated section per store in `ConstructMarkdown.JSON_STORES` — Price Factors, Price Models, Market Prices, Bootstrapper Configuration, Trade Data and Calculations — each reaching it from the classes through its `schema.emit_*` (an instrument type composes the sections that hold its descriptors; a factor or process type IS them).
- A field's `description` is its **prose**, published per field on the generated page. It is not a key: the key is the field's name, what the descriptor is FILED under and stamped as `name` when the store is loaded. Its `type` says what the value is, and the widget and the `obj` token a front end writes are derived from it (`F.descriptor`). **Held by** `tests/test_schema_emission.py::test_a_published_obj_is_the_wire_token_its_type_names`.
- A page cites a work as [&#64;key] (several as [&#64;a; &#64;b]), the key listed once in `docs_src/references.md`. The build numbers the bibliography in file order, writes the References page and links every citation to its number (`number_references`); a key the bibliography lacks fails the build.

So for a deal, a price factor, a stochastic process, a calculation, a calibration or a price family, model math and JSON field docs both live **on the class**. `schema.py` holds the vocabulary (`F`, `Row`, `Group`), the emitters and the assembly. The only hand-written store left is `System`, whose consumer is `Config` itself, plus the create-deal menu. This developer section links to the generated pages; it does not restate them. The build is `ConstructMarkdown.build` in `derivus_docs.py` at the repo root.

## Comment and code style {#code-style}

- Correct > efficient > terse, and no redundant work in a hot path (no stray `.to()` / `.item()` / `.detach()` in the AAD path).
- Numerically stable forms by default: `expm1`/`log1p`, log-domain tails, no difference of near-equal roots, float64 precalculation cast once.
- A comment says why, in at most three lines; a docstring says what the code does; an `F` description is one sentence. History goes in the commit message.
- Operational lines at `logging.info`, build-time checks at `logging.debug` (`LEDGER`, `BOUNDARY`, `KINK`), never behind `os.environ` flags.
- A diff that removes lines usually wins.

## Look before you write {#look-before-you-write}

Before adding a helper, **search `utils.py` and the package for an existing equivalent**; a new primitive is usually a missed search. A `Deal` method is the seam for every family whatever its docstring says (`check_option_data`): extend it with a parameter rather than writing beside it. A second spelling found while working is deleted.

## Change scope {#change-scope}

`CreditMonteCarlo` and `BaseValuation` are **do-not-touch** — above all the CVA/FVA/CollVA/IM block in `CreditMonteCarlo.execute`, specified on [xVA](../theory/valuation_adjustments.md) — because they are the production valuation paths everything downstream reconciles against. The `HedgeMonteCarlo` / solver stack is free to redesign. Inside `Deal` methods, reuse pre-processed data (`field_index['Cashflows']`) rather than re-walking `self.field`.

The one thing wired into that block is [boundary-correction assembly](calc_lifecycle.md#boundary-corrections-the-sensitivity-subsystem), and its terms are the terms for anything else: it changes what reaches `backward()` and **nothing** reported, being worth exactly zero forward. **Held by** `tests/test_boundary_pricer_events.py::test_asking_for_sensitivities_does_not_move_the_autocall_exposure` (exposure `np.array_equal`, CVA `==`, sensitivities on against off) and `tests/test_cva_gamma_kink.py::test_asking_for_the_hessian_moves_nothing_the_run_already_reported`. A per-factor scaling of the correlation matrix before the Cholesky fails those terms: a declared correlation is the correlation of the innovations for every process, and nothing scales it. **Held by**, for LogVar2FJ, `tests/test_logvar2fj_json.py::test_a_logvar2fj_equity_realises_the_rows_its_four_sub_factors_declare`.

## A unification owes its siblings {#unification-siblings}

When you absorb N call sites into one seam, **enumerate the call sites you did not absorb** and either bring each in or gate it against the seam; naming them in the commit message is the floor. The duty is the unifier's because a unification degrades its siblings three ways: **the seam inherits all the new testing** while a sibling's coverage stays put; **the sibling becomes the only copy** of a formula once reviewed beside its twin; and **a reviewer reading a clean seam stops looking**.

In the OSS pricers every sibling `InnerMCRecompute.run` left outside it kept its own spelling of the forward or the variance, and each was wrong. Two failure shapes run through all of them:

- **The wrong allocation telescopes to the right total**, so European limits, in-out parity and CRN gradient ladders stay right and only path-dependent monitoring is biased.
- **At a flat surface or zero carry the value cannot see the strip while the gradient can.**

`pricing.total_log_forward`, `forward_carry_rate`, `forward_vol_strip` and `forward_vol_rate` make divergence **unrepresentable**: every adopter takes them, and none keeps an inline spelling, even a correct one. A vol read belongs to the leg, at the leg's own strike (the autocall's triggers through `trigger_vol_strip`, its put on a strip of its own, a barrier digital's two spread legs each on theirs: **Held by** `tests/test_digital_spread_json.py::test_a_monthly_barrier_binary_no_path_reaches_is_the_european_spread_within_its_noise`). They cannot catch a wrong `cum_t`, which is shape-compatible and prices identically on a flat fixture. A carry is `zero_carry_rate`, the curves' integral over the year fraction its consumer multiplies it by (`deal_carry_rate` where that is the deal's discount day count), never a zero rate read off the curve unscaled: a static curve under `NoModel: RiskNeutral` answers a read from a row as the base-date curve's forward from it (a curve keyed by date being that already), so an unscaled read is the difference of two zero rates, zero on a flat curve. **Held by** `tests/test_static_curve_risk_neutral.py::test_a_static_curve_reads_the_same_on_the_base_date_under_either_no_model`. **Held by** `tests/test_branch_and_weight.py::test_the_two_fixing_tarf_lands_on_the_quadrature_table`, `::test_the_two_coupon_autocall_lands_on_the_quadrature_table` (flat inputs: the totals, not the allocation) and `tests/test_sticky_strike_strip.py::test_the_two_conventions_are_one_coordinate_where_the_reference_is_the_forward`. **Not gated:** the allocation under a sloped curve or a vol term structure.

**A documented limitation must carry its number.** One recorded without its measured size is absolution, not documentation: a reader assumes it is small. If it is unmeasured, say so and say what measuring it would take.

### What is gated, and what is not {#what-holds-today-and-what-the-purge-left-open}

- **Held:** the already-hit barrier leg's value, invisible to an AAD-against-CRN gate since a bump ladder differentiates the same wrong value (`tests/test_barrier_fold_json.py::test_a_row_every_scenario_has_crossed_is_what_the_barrier_became`); pricer branches no test executes (`gates/pricer_branch_census.py`), though a branch executed is not a branch checked.
- **Not gated:** the vol term structure reaching the monitoring (two surfaces agreeing at expiry and differing before it); the payoff's forward against the vol surface's; rate units, an annualised carry reaching `exp` without a year fraction (an AST pass is the cheap check, blind to a units-correct, value-wrong sloped carry).

## A fixture must not zero the quantity its gate is sensitive to {#fixture-degeneracy}

A gate whose fixture zeroes the quantity under test runs, passes and measures a term that is identically zero. Two rules follow.

**Vary the parameter the defect would live in, then check the mutant dies.** A surviving mutant means the fixture is wrong, not that the code is right. Record the kill magnitudes in the docstring (`tests/test_boundary_pricer_events.py` is the template), so a later "simplify this fixture" refuses itself.

**Marginal coverage is not joint coverage.** A branch reached only by a conjunction needs one gate that satisfies all of it.

### The checklist

Before calling a gate done, say for each "varied" or "degenerate **because** …":

| | |
|---|---|
| `r`, `q` | non-zero **and** different — `r = q` kills the forward, `r = 0` kills the discount |
| time rows | more than one, and at least one **after** the first event date, or every post-t0 branch is unexecuted |
| direction | both knock-**In** and knock-**Out**; both `Up` and `Down` |
| side | `Buy` **and** `Sell`, and both signs of `Units` |
| option type | `Call` **and** `Put` |
| rebate / coupon | non-zero, or the knock branch pays nothing and its date and discount are unasserted |
| vol surface | not flat — a flat surface hides every wrong-moneyness and wrong-expiry read |
| correlation | non-zero where a cross-term exists (quanto carry, multi-factor diffusion) |
| netting sets | more than one where the calculation aggregates |
| the conjunction | which combination reaches the branch you are gating, and does one run hold all of it? |

### The plugin

`python gates/fixture_degeneracy.py tests/<modules>` tables every run by **deal type × degeneracy axis**; a cell is *blind* when no run made the quantity non-degenerate. What **enforces** is the short `JOINT` list of conjunctions one run must hold (`EquityBarrierOption`: `single_mtm_row=False`, `barrier_dir='In'`, `carry_zero=False`), since every marginal cell can read `ok` while a conjunction is unmet: an unmet entry or a stale `ACCEPTED` exemption fails the run, and `--degeneracy-strict` fails a blind cell too. A selection pricing no deal of an entry's type skips the entry, so confirm one over the modules that price it (`tests/test_barrier_fold_json.py` meets the barrier's by construction).

## What a landing runs {#what-a-landing-runs}

**Blast radius first:** `python gates/reach.py <Symbol>` (or `--dirty`) names the consumers, callers, documents and holes a change reaches ([limits](roadmap.md#how-a-change-is-verified)).

**A landing runs the tests its diff reaches.** `python gates/impacted.py --dirty --time` names them, `--run` runs them (`--since BASE`, `--commit SHA`), and `--files` rounds up to whole files for a test reading another's cached value. Selection is line-granular off an execution map — a changed statement selects the tests that executed it, a comment nothing — and fails open to the whole suite, each file under its own home, saying why, on `tests/conftest.py`, `derivus/__init__.py`, an unmapped path or no map. Beside it run the standing readings.

**A campaign boundary runs the full suite** with the tree held still, since a selection cannot see an order-sensitive statistical gate or a dependency gained since the map; then `python gates/impacted.py --measure DIR` rebuilds the local map, and `CUDA_VISIBLE_DEVICES=0 python gates/pricer_branch_census.py` traces the tests it says enter a census pricer (`--anchors`: the AST half alone).

**A test** drives a JSON contract, has a defined outcome, covers a named area, runs or fails quickly, and names its killing mutation; it goes only where every mutation it names is shown red under the tests that remain, never on coverage alone.

**A change that renames, moves or deletes code runs `python gates/doc_symbols.py`**, which checks that every function, class, constant, test and file these pages cite in backticks still exists, so a **Held by** line cannot point at a test that is gone. It cannot tell whether a page describes the code correctly: a change to behaviour updates the page that states the rule.

## No overengineering

One-line fixes — a default, a JSON field, a file move — before a helper module. Do not introduce abstraction ahead of a second caller.
