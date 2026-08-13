# Conventions

House rules for the derivus codebase. These are enforced by convention, not by lint.

## JSON is the contract {#json-is-the-contract}

The job JSON is the whole program. End-user scripts do `import derivus as rf`, `cx.load_json(...)`, `cx.run_job(...)` — **no internal imports, no monkey-patching**. Every framework feature ships behind a JSON switch, defaulting to bit-identical-when-off. The JSON format is documented in [JSON Configuration](../json/index.md) and generated from `schema.mapping`.

Consequences:

- **No defensive checks.** The JSON is validated by being the contract; code assumes required fields are present and fails loud (`KeyError` naming the factor via `check_tuple_name`) when they are not. Do not add `if x in d` guards for contract fields.
- **Config per variant, not flags.** A variant is a separate JSON config; scripts stay generic. Do not thread experiment booleans through function signatures.
- **Never monkey-patch internals.** A missing metric or behavior is added to the framework behind a switch — never reached into from a user script.
- **MarketData files are data-only** — no `Description` / commentary fields.

## `globals()` dispatch {#globals-dispatch}

Constructors dispatch on class name through the module `globals()`: `construct_calculation` does `globals().get(calc_type)(...)`; processes, deals, factors, and calibration classes are all resolved the same way from their JSON type string. This is why **one class per concept in a file** is the norm — the class name *is* the dispatch key. Adding a type means adding a class, not editing a dispatcher.

## Registries, not functions

Extension points are data. Factor discovery is three dicts (`dependant_fields`, `nested_fields`, `conditional_fields` — see [Dependency System](dependency_system.md)); process→factor wiring and calibration are registered in `Model Configuration` and `calibration_config.json`; a deal's, a price factor's, a process's, a calculation's, a calibration's or a price family's JSON schema is a per-class `fields` list (`schema.py`), and every store but `System` is the view `schema.emit_*` builds from those declarations at import — not a second copy to keep in step. Where the store's key is not the class name, the class declares the bridge rather than the emitter guessing it: a process declares the `factor_types` it drives and `Process_factor_map` is that read the other way round; a calculation declares the `calc_type` its JSON `Object` writes, because `Base_Revaluation` is authored as `BaseValuation`; a calibration declares the `model_type` it calibrates, because `HWInterestRateCalibration` calibrates `HullWhite1FactorInterestRateModel`; a bootstrapper declares the `market_factor_type` it selects work by, because a block is named `HestonNandiModelPrices` while its class is `HestonNandiModelParameters`. No rule recovers any of those names from the other. Two menus are the same declarations read sideways: `Process_factor_map`, and `Interpolation_factor_map` from the `interpolation_methods` the two curve factors `construct_factor` routes declare. And a schema may REFERENCE another rather than restating it — a market-price quote names an instrument type and carries a block of it, so the `Instrument` declarations are the quote's schema; see [Market Prices](market_prices.md). A function that switches on a magic-string type, or a parallel dict passed alongside a primary operand, is a class-or-registry waiting to happen — flag it. Strengthen the existing primitive; do not bolt on a parallel concept or a magic-string branch.

## Documentation and doc generation {#documentation-and-doc-generation}

Developer-facing model/deal/process docs are **class attributes**, harvested at build time by `derivus_docs.py`:

- `documentation` (a `(section_name, [md_lines])` tuple) on classes in `stochasticprocess.py` / `instruments.py` / `bootstrappers.py` / `calculation.py` auto-publishes the Theory / Valuation / Bootstrapping / API pages. It is read **own-attr-only** (`cls.__dict__`, not MRO) so an alias subclass does not re-emit the parent's page.
- `schema.mapping` alone drives the JSON reference tree (`generate_json_docs`), for the `Factor`, `Process` and `Instrument` stores only. All three reach it from the classes, via `schema.emit_instrument` / `emit_factor` / `emit_process` — an instrument type composes the sections that hold its descriptors, a factor or process type IS them.
- A field's `description` is its **prose**, published per field on the generated page. It used to be a key as well - the Workbench rebuilt the JSON key as `description.replace(' ', '_')` at six sites - so prose in it pointed a widget at a key nobody writes. The key is now the key the descriptor is FILED under, stamped as `name` when the store is loaded, and an alias is the one thing that has to be declared.
- `field_desc` on the `riskfactors.py` classes is **gone**: it was harvested into `_riskfactor_field_desc`, which nothing read, so twenty-four classes carried prose that reached no page. Its content is now each field's `description`.

So for a deal, a price factor, a stochastic process, a calculation, a calibration or a price family, model math and JSON field docs both live **on the class**. `schema.py` holds the vocabulary (`F`, `Row`, `Group`), the emitters and the assembly, and `fields.py` is a deprecated shim re-exporting `mapping` for one release — it was the documented surface and the package is on PyPI. The only hand-written store left is `System`, whose consumer is `Config` itself, and the create-deal menu. This developer section links to those generated pages; it does not restate them. The doc build (`ConstructMarkdown.build`) is described in the section README.

## Comment and code style {#code-style}

- Terse. Correct **>** efficient **>** least-lines, in that order — but no redundant work in hot paths (no stray `.to()` / `.item()` / `.detach()` in the AAD path).
- One-line comments explaining *why*, not *what*. No banner comments.
- Diagnostics at `logging.info`, never behind `os.environ` flags.
- A diff that removes lines/imports usually wins: small public surface, one-way dependency edges.

## Look before you write {#look-before-you-write}

Before adding any helper, **search `utils.py` and the package for an existing equivalent** — this is a hard rule, and every new-code task must do it first. Dedupe on contact when moving code. The name-resolution, curve-gather, date, and tensor primitives you need almost always already exist; a new one is usually a missed search.

## Change scope {#change-scope}

`credit_monte_carlo` and `base_valuation` are **do-not-touch** — the CVA/FVA/CollVA/IM block inside `Credit_Monte_Carlo.execute` in particular. They are the production valuation paths and everything downstream reconciles against the numbers they report. The `HedgeMonteCarlo` / solver stack is free to redesign. Reuse pre-processed deal data (`field_index['Cashflows']` etc.) inside `Deal` methods rather than re-walking `self.field`.

The one thing that *is* wired into that block is [boundary-correction assembly](calc_lifecycle.md#boundary-corrections-the-sensitivity-subsystem), and the terms on which it got there are the terms for anything else: it changes what is handed to `backward()` and **nothing** about what is reported. The correction is worth exactly zero in the forward pass by construction, and that is gated bit-identically — `np.array_equal` on the exposure and `==` on the scalar, with sensitivities on versus off. A change that cannot make that guarantee does not belong here.

## A unification owes its siblings {#unification-siblings}

When you absorb N call sites into one seam, **enumerate the call sites you did not absorb**, and for each one either bring it in or gate it against the seam. Naming them in the commit message is the floor. An unnamed sibling is where the next defect gets written.

The obligation is on the person doing the unification because a unification does not leave its siblings where it found them — it **degrades** them, three ways at once:

- **The seam inherits all the new testing.** Every gate written during the work points at the seam. The sibling's coverage is whatever it had beforehand, which is now the coverage of code nobody is looking at.
- **The sibling keeps a copy of arithmetic that used to be reviewed beside its twin.** Two expressions side by side are read together and fixed together. Move one behind a seam and the other becomes the only copy of a formula with no reader.
- **A reviewer reading a clean seam stops looking.** One helper, one call, obviously right — and that conclusion generalises to the whole pricer, which is exactly wrong.

**The instance.** `InnerMCRecompute.run` absorbed the inner simulation of three pricers into one seam. `hit_value` — the outer path-state override sitting *beside* that seam in `pv_discrete_barrier_option` — was never part of `simulate` or `theta`, so it was not absorbed. It had literally **copied** its forward from the in-out-parity leg; the two were correct together, and then only one was maintained. What the copy held was `(drifts + 0.5*var).sum(dim=1)`, where `drifts` is an annualised **rate** per fixing: rates summed with no `dt`, plus a half-variance whose cancelling subtraction exists only on the branch it was copied from. On a fixture with `r != q` and a long remaining observation strip it marked the option at **+1432%** of its value, on every `all_hit` row, on every `BARRIER_IN` deal in every exposure/CVA/PFE run, GBM and Heston–Nandi alike. The whole suite was green.

**And a third adopter, one layer down.** `sim_spot_oss` built its per-interval carry as `carry * dt`; `pv_MC_Tarf` and `pv_MC_AutoCallSwap` — the other two adopters of that same seam — differenced the cumulative integrals and were right. Three adopters, two maintained, one not, and this one drove the barrier's simulated drift, so `E[S_T] ≠ F(t,T)` under the pricer's own Monte Carlo. Measured **4.276e-02** on the sibling fixture's sloped curve. One instance is a story; three is the mechanism.

It is closed the way the mechanism says to close it, not by fixing the one that was wrong: `pricing.forward_carry_rate` is the strip, every adopter takes it in its `theta` in place of the raw zero rates, and the two correct spellings were **deleted** rather than left as the reference. The primitive reproduces their arithmetic term for term, including the `j == 0` branch where the cumulative window *is* the interval, so the replacement is bit-identical — measured, not assumed, on the fixtures where the carry is non-zero: TARF and autocall prices identical to the last bit under GBM and Heston–Nandi, with 2 ULP on one CVA-gradient entry from backward reassociation.

**A fourth adopter fell out of it**: `pv_MC_AutoCallSwap`'s *averaging* branch has its own `carry * dt`, in the same function, behind the other arm of `no_averaging`, and nothing in the defect report named it. It is fixed by the same theta swap and by nothing else. Its price is unchanged (that fixture is `r = q = 0`) and its **rate sensitivities move by 100%** — which is the shape of this whole class: at zero carry the value cannot see the strip and the gradient can. An enumeration that stops at the sites the defect report can name is not an enumeration.

### What exists now, and what each one catches

| | catches | cannot catch |
|---|---|---|
| `pricing.total_log_forward` | nothing — it makes divergence **unrepresentable** rather than detected. Both barrier legs call it, so a second spelling has to be written deliberately. | — |
| `pricing.forward_carry_rate` | the same, one layer down: the interval carry strip all four adopters simulate on. It is the only place the difference of cumulative integrals is written. | a caller that hands it the wrong `cum_t`. The strip is *shape*-compatible with the raw zero rates, so passing `drifts` where `fwd_drifts` goes still prices — and prices identically on every flat fixture. Only the sloped gate separates them. |
| `tests/test_sibling_forward_agreement.py` | that both legs still read that one expression, and — the assertion that would have killed the original alone — that the forward the **payoff** uses equals the forward the **vol surface** is read at. Genuinely independent routes: one gathers the carry once at `tau`, the other integrates over the fixing strip. No market data, no Monte Carlo, no reference value. | a defect both spellings agree on. Measured: dropping `times` inside the helper leaves the sibling assertion green and dies only on the third route. A consistency gate needs a route that was never a sibling. |
| `tests/test_already_hit_barrier_leg_value.py` | the **value**, against a reference rebuilt from the deal's own dates, on an `all_hit` block where the leg is the entire reported PV. This is the class **no AAD-vs-CRN gate can see** — a bump ladder differentiates the same wrong value, so a wrong number with a consistent derivative reads as a pass. | anything on a row shape it does not price. Its own anti-placebo matrix records that the `r = q = 0` variant goes green-but-blind on two of six mutants. |
| `tests/test_rate_units.py` | a name holding an annualised carry rate reaching `exp` without a year fraction first — an AST taint pass over `derivus/`, 1.5s, no engine import, no fixture. Clean on the current tree; it flags exactly the shipped expression and nothing else on `a87e3b5^`. | the sloped-carry strip above, which is units-**correct** and value-wrong. Costs two word-lists at the top of the file; a legitimate undeclared time name fires until it is added. |
| `tests/test_pricer_branch_ledger.py` + `gates/pricer_branch_census.py` | branch arcs no test executes — 65 across the ten-pricer barrier/option family, each carrying the fixture property that would reach it, held to the pricers by AST so a reworded branch turns the suite red the same day. | wrong arithmetic. Measured on the pre-fix tree: the defective **line** was executed, as a `boundary_aad` counterfactual whose value is discarded, so branch coverage would not have flagged it. `if all_hit:` was executed by nothing, and that is the line the census prints. **Executed is not observed.** |
| `gates/fixture_degeneracy.py` | see the next section. | |

**A guard is a branch too.** `boundary_weights` carried a near-singularity refusal that tested `denominator/(s0·s2)` against `1e-30` — a ratio Cauchy–Schwarz bounds by 1. It refused nothing in the engine's entire history, and nothing could tell: a refusal that never fires and a refusal that is never needed look identical from outside. The docstring stated the intent and the code did not implement it. It now bounds `||weights||_1`, which is what the solve actually produced. This is why the census reports never-called scopes and never-taken guards at all — two Heston–Nandi batch-constant-carry refusals are on that ledger today.

### A documented limitation must carry its number

A "Known limitation" recorded without its measured magnitude is absolution, not documentation. The model half of the barrier defect sat in the pricer's own docstring for years with no number attached; it was **15.8%** per unit, and 25.3% of EPE on the profile. A reader who meets a limitation with no size assumes it is small — that is the work "known" is doing in the sentence — and stops reading. If you have not measured it, write that you have not, and write what measuring it would take.

## A fixture must not zero the quantity its gate is sensitive to {#fixture-degeneracy}

The recurring way work with passing tests turns out to be wrong here is not bad reasoning and not a mis-stated assertion. It is a **fixture that sets the quantity under test to nothing**, so the gate runs, passes, and measures a term that is identically zero. The roadmap counted eight instances in one work stream; the already-hit barrier leg is the expensive one — a **+1432%** mark that three independent degeneracies each hid on their own:

- **`r = q = 0`.** The defect was a drift summed without its `dt`, which is proportional to the carry. At zero carry the same defect reads **+17.5%**, and two of the six mutants that kill it become no-ops (`exp(r_step·n)` is identically 1, and so is the discount factor).
- **Base valuation.** One MTM row means the `all_hit` mask is all-False at row 0, so the leg was not mis-asserted — it was **never executed**.
- **`Down_And_Out` only.** The one barrier that did run an exposure grid took the model-free zeros branch.

Two rules follow, and they are different rules.

**Vary the parameter the defect would live in, then check the mutant dies.** A mutant that survives your gate means the **fixture** is wrong, not that the code is right. Record the kill *magnitudes* in the test's docstring — the mutation matrix in `tests/test_already_hit_barrier_leg_value.py` is the template, and it is what makes a later "simplify this fixture" refuse itself.

**Marginal coverage is not joint coverage.** The suite already had a knock-in barrier and already had an exposure grid and already had a non-zero carry — in *different* tests. The leg needs all three in the **same run**. When a branch is reached only by a conjunction, one gate has to satisfy the whole conjunction.

### The checklist

Before calling a gate done, for each of these say either "varied" or "degenerate **because** …":

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

### The mechanical half

`gates/fixture_degeneracy.py` is a pytest plugin that observes the calculation entry points and folds every run into a **deal type × degeneracy axis** table — no test source parsing, no hand-kept module→family map, so a new deal type appears the day its first test runs. A cell is *blind* when no run of that deal type ever made the quantity non-degenerate. Run it as `python gates/fixture_degeneracy.py tests/...`; measured on the fourteen barrier/option/HN/TARF/autocall modules it costs 6:39 → 7:12, so it rides an existing suite invocation rather than being a run of its own.

What **enforces** is the short `JOINT` list, not the table, and the reason is measured: with the fix's own gates removed from the row set, every *marginal* cell for `EquityBarrierOption` still reads `ok` — knock-ins existed, exposure grids existed, live carries existed. Only the three-way conjunction is unmet. **The marginal table would not have caught this defect; the joint requirement does.** The table is a report until someone triages it into `ACCEPTED` (`--degeneracy-strict` makes it fail), because fifty-odd red cells on day one is a gate nobody reads.

Neither half can tell you a gate's *assertion* is weak — only that its fixture had nothing to assert on.

## No overengineering

One-line fixes — a default, a JSON field, a file move — before a helper module. Do not introduce abstraction ahead of a second caller.
