# Conventions

House rules for the derivus codebase. Enforced by convention, not by lint.

## JSON is the contract {#json-is-the-contract}

The job JSON is the whole program. End-user scripts do `import derivus as rf`, `cx.load_json(...)`, `cx.run_job(...)` — **no internal imports, no monkey-patching**. Every framework feature ships behind a JSON switch, defaulting to bit-identical-when-off. The format is documented in [JSON Configuration](../json/index.md) and generated from `schema.mapping`.

- **No defensive checks.** The JSON is validated by being the contract; code assumes required fields are present and fails loud (`KeyError` naming the factor via `check_tuple_name`). Do not add `if x in d` guards for contract fields.
- **Config per variant, not flags.** A variant is a separate JSON config; scripts stay generic. Do not thread experiment booleans through function signatures.
- **Never monkey-patch internals.** A missing metric or behavior is added to the framework behind a switch, never reached into from a user script.
- **MarketData files are data-only** — no `Description` / commentary fields.

## `globals()` dispatch {#globals-dispatch}

Constructors dispatch on class name through the module `globals()`: `construct_calculation` does `globals().get(calc_type)(...)`; processes, deals, factors and calibration classes resolve the same way from their JSON type string. Hence **one class per concept in a file** — the class name *is* the dispatch key. Adding a type means adding a class, not editing a dispatcher.

## Registries, not functions

Extension points are data.

- Factor discovery is three dicts (`dependant_fields`, `nested_fields`, `conditional_fields` — see [Dependency System](dependency_system.md)); process→factor wiring and calibration are registered in `Model Configuration` and `calibration_config.json`.
- A deal's, price factor's, process's, calculation's, calibration's, sales structure's or price family's JSON schema is a per-class `fields` list (`schema.py`), and every store but `System` is the view `schema.emit_*` builds from those declarations at import — not a second copy to keep in step.
- Where the store's key is not the class name, **the class declares the bridge** rather than the emitter guessing it: a process declares the `factor_types` it drives, a calculation the `calc_type` its JSON `Object` writes (`Base_Revaluation` is authored as `BaseValuation`), a calibration the `model_type` it calibrates, a bootstrapper the `market_factor_type` it selects work by (a block is named `HestonNandiModelPrices` while its class is `HestonNandiModelParameters`). No rule recovers any of those names from the other.
- Two menus are the same declarations read sideways: `Process_factor_map`, and `Interpolation_factor_map` from the `interpolation_methods` the three curve factors declare.
- A schema may REFERENCE another rather than restating it — a market-price quote names an instrument type and carries a block of it, so the `Instrument` declarations are the quote's schema; see [Market Prices](market_prices.md).

A function that switches on a magic-string type, or a parallel dict passed alongside a primary operand, is a class-or-registry waiting to happen — flag it. Strengthen the existing primitive; do not bolt on a parallel concept or a magic-string branch.

## Documentation and doc generation {#documentation-and-doc-generation}

Developer-facing model/deal/process docs are **class attributes**, harvested at build time by `derivus_docs.py`:

- `documentation` (a `(section_name, [md_lines])` tuple) on classes in `stochasticprocess.py` / `instruments.py` / `bootstrappers.py` / `calculation.py` and the three `hedge_*.py` modules auto-publishes the Theory / Valuation / Bootstrapping / API / Hedging pages. Read **own-attr-only** (`cls.__dict__`, not MRO), so an alias subclass does not re-emit the parent's page.
- `schema.mapping` alone drives the JSON reference tree (`generate_json_docs`), for the `Factor`, `Process` and `Instrument` stores only. All three reach it from the classes via `schema.emit_instrument` / `emit_factor` / `emit_process` — an instrument type composes the sections that hold its descriptors; a factor or process type IS them.
- A field's `description` is its **prose**, published per field on the generated page. It is not a key: the key is what the descriptor is FILED under, stamped as `name` when the store is loaded, and an alias is the one thing that has to be declared.

So for a deal, a price factor, a stochastic process, a calculation, a calibration or a price family, model math and JSON field docs both live **on the class**. `schema.py` holds the vocabulary (`F`, `Row`, `Group`), the emitters and the assembly; `fields.py` is a deprecated shim re-exporting `mapping` for one release, because it was the documented surface and the package is on PyPI. The only hand-written store left is `System`, whose consumer is `Config` itself, plus the create-deal menu. This developer section links to the generated pages; it does not restate them. The build is `ConstructMarkdown.build` in `derivus_docs.py` at the repo root.

## Comment and code style {#code-style}

- Terse. Correct **>** efficient **>** least-lines — but no redundant work in hot paths (no stray `.to()` / `.item()` / `.detach()` in the AAD path).
- One-line comments explaining *why*, not *what*. No banner comments.
- Diagnostics at `logging.info`, never behind `os.environ` flags.
- A diff that removes lines/imports usually wins: small public surface, one-way dependency edges.

## Look before you write {#look-before-you-write}

Before adding any helper, **search `utils.py` and the package for an existing equivalent** — a hard rule, and every new-code task must do it first. Dedupe on contact when moving code. The name-resolution, curve-gather, date and tensor primitives you need almost always already exist; a new one is usually a missed search.

## Change scope {#change-scope}

`Credit_Monte_Carlo` and `Base_Revaluation` are **do-not-touch** — the CVA/FVA/CollVA/IM block inside `Credit_Monte_Carlo.execute` in particular. They are the production valuation paths and everything downstream reconciles against the numbers they report. The `HedgeMonteCarlo` / solver stack is free to redesign. Reuse pre-processed deal data (`field_index['Cashflows']` etc.) inside `Deal` methods rather than re-walking `self.field`.

The one thing wired into that block is [boundary-correction assembly](calc_lifecycle.md#boundary-corrections-the-sensitivity-subsystem), and the terms on which it got there are the terms for anything else: it changes what is handed to `backward()` and **nothing** about what is reported — worth exactly zero in the forward pass by construction, gated bit-identically (`np.array_equal` on the exposure, `==` on the scalar, sensitivities on versus off). A change that cannot make that guarantee does not belong here.

## A unification owes its siblings {#unification-siblings}

When you absorb N call sites into one seam, **enumerate the call sites you did not absorb**, and for each either bring it in or gate it against the seam. Naming them in the commit message is the floor. An unnamed sibling is where the next defect gets written.

The obligation is on the unifier, because a unification does not leave its siblings where it found them — it degrades them three ways at once: **the seam inherits all the new testing**, while the sibling's coverage stays what it was; **the sibling becomes the only copy** of a formula that used to be reviewed beside its twin; and **a reviewer reading a clean seam stops looking**, generalising "obviously right" to the whole pricer.

**Measured — four unabsorbed siblings of one seam.** `InnerMCRecompute.run` absorbed three pricers' inner simulation (`pv_MC_Accumulator` joined as a fourth later), and each sibling left beside it was wrong:

| unabsorbed sibling | the defect | measured |
| --- | --- | --- |
| `hit_value`, the outer path-state override in `pv_discrete_barrier_option` | a literal copy of the in-out-parity leg's forward: annualised rates summed with no `dt`, plus a half-variance whose cancelling subtraction exists only on the branch it came from | **+1432%** of value on every `all_hit` row of every `BARRIER_IN` deal, GBM and Heston–Nandi alike, with the whole suite green |
| `sim_spot_oss`'s `carry * dt` | a per-interval carry where the two maintained adopters differenced cumulative integrals, so `E[S_T] ≠ F(t,T)` under the pricer's own Monte Carlo | **4.276e-02** on the sibling fixture's sloped curve |
| `sim_spot_oss`'s `(vols*vols) * dt`, and the same expression in both `pv_MC_AutoCallSwap` branches | ONE implied vol read at the strike's moneyness and the EXPIRY tenor, applied to every monitoring interval; only `pv_MC_Tarf` differenced | `Down_And_In` **+11.53%**, `Up_And_Out` -11.07% on a 0.10–0.32 surface; the autocall **-8.27%, 207 se** against a sloped oracle, the strip -0.081% |
| `pv_MC_AutoCallSwap`'s *averaging* branch, behind the other arm of `no_averaging` | its own `carry * dt`, named by no defect report | price unchanged (`r = q = 0` fixture); **rate sensitivities move 100%** |

Two failure shapes run through all four. The wrong allocation **telescopes to the right total**, so every European limit, in-out parity and every CRN gradient ladder stay right and only the path-dependent monitoring is biased. And **at a flat surface or zero carry the value cannot see the strip while the gradient can** — which is why a flat-surface repo could not see any of it.

Closed the way the mechanism says, not by fixing the one that was wrong: `pricing.forward_carry_rate`, `pricing.forward_vol_strip` and `pricing.forward_vol_rate` are the primitives, every adopter takes them in its `theta`, and the correct inline spellings were **deleted** rather than left as the reference. Bit-identity was measured, not assumed (2 ULP on one CVA-gradient entry from backward reassociation). The moneyness half is a **second** change with the opposite sign, closed by internal consistency: the deal's own `use_forwards` is threaded into the strip, so both pricers read the surface at the moneyness they mark their own Europeans at (0 ULP on the autocall's closed-form digital, +0.016% on the barrier, parity back to -0.19 ± 1.30) and every flat repo fixture is 0 ULP. What remains in [the roadmap](roadmap.md) is the MODELLING question — a desk quoting sticky-*forward* moneyness wants the read that was removed — a switch, not a defect.

**A fix can wake a sleeping guard, and the fixture that proves the fix is the one that hides it.** `sim_spot_oss` floors every interval's variance at `1e-4`, harmless only while the defect above handed it the largest vol on an upward surface. A correct strip makes it bind wherever `sigma_fwd < 0.01/sqrt(dt)`: **114 of 365 daily intervals** on a 0.12→0.24 surface, worth +1.584% on a `Down_And_Out` and -6.870% on an `Up_And_Out`. Every barrier fixture here is MONTHLY, ten times clear of it and bitwise identical with the floor on, off or conditional. The floor is now conditioned on the zero-length first interval it exists for (`dt == 0`, elementwise, because `dt` is a tensor axis), with `drift` and `vol` reading ONE `var` so a floored interval is a martingale under the law it is drawn from; daily readings land at +0.183% and +0.403%. Its two halves partly CANCEL — the floor alone costs -5.576% on the up-and-out but only -0.13% on the down-and-out, so a one-armed gate would have scored it harmless. **Deleting it entirely is how you find out what it was for**: every value stays finite and 11 of 13 CVA-gradient entries go NaN. It was never a value guard; it is a `sqrt`-at-zero guard.

**A documented limitation must carry its number.** A "known limitation" recorded without its measured magnitude is absolution, not documentation. The model half of the barrier defect sat in the pricer's own docstring for years with no number attached; it was **15.8%** per unit and 25.3% of EPE. A reader who meets a limitation with no size assumes it is small. If you have not measured it, write that you have not, and write what measuring it would take.

### What holds today, and what the purge left open

`pricing.total_log_forward`, `forward_carry_rate`, and `forward_vol_strip` / `forward_vol_rate` catch nothing by assertion — they make divergence **unrepresentable** rather than detected, so a second spelling has to be written deliberately. What they cannot catch: a caller handing the wrong `cum_t` (the strip is *shape*-compatible with the raw zero rates, so it still prices, and prices identically on every flat fixture), and a FLAT surface, which is every barrier, TARF and autocall fixture in this repo.

**A guard is a branch too.** `boundary_weights` carried a near-singularity refusal testing `denominator/(s0·s2)` — a ratio Cauchy–Schwarz bounds by 1 — against `1e-30`. It refused nothing in the engine's entire history, and nothing could tell: a refusal that never fires and one that is never needed look identical from outside. The docstring stated the intent and the code did not implement it. It now bounds `||weights||_1`, which is what the solve actually produces.

**OPEN — ungated since the 2026-08-21 purge** (`gates/plat_machine/TEST_PURGE.md`, commit 104bd08; only JSON-contract tests survived). The measurements are kept above; nothing below re-checks them.

- **`tests/test_vol_term_structure_strip.py`** — that a surface's TERM STRUCTURE reaches the monitoring (two surfaces agreeing at expiry must separate, each landing on an independent fine-step oracle), the moneyness half, and the European invariant the fix must not break. It held the repo's only 0-ULP price assertion (the autocall's one-coupon closed-form digital) and its only fixture dense enough to reach the `1e-4` variance floor.
- **`tests/test_sibling_forward_agreement.py`** — that the forward the **payoff** uses equals the forward the **vol surface** is read at — the third route, the only one that was never a sibling. Measured: dropping `times` inside the helper left the sibling assertion green and died only on that route. Nothing replaces it, so the payoff-forward against vol-surface-forward route is ungated.
- **`tests/test_already_hit_barrier_leg_value.py`** — the `hit_value` leg's VALUE against a reference rebuilt from the deal's own dates, on an `all_hit` block where the leg is the entire reported PV. This is the class **no AAD-vs-CRN gate can see** — a bump ladder differentiates the same wrong value, so a wrong number with a consistent derivative reads as a pass. The leg is live and unasserted.
- **`tests/test_rate_units.py`** — an AST taint pass over `derivus/` for a name holding an annualised carry rate reaching `exp` without a year fraction first (1.5 s, no engine import, no fixture). It flagged exactly the shipped expression and nothing else. Nothing runs the pass today. It never could see the sloped-carry strip, which is units-**correct** and value-wrong.
- **`tests/test_pricer_branch_ledger.py`** — branch arcs no test executes. **Replaced**: the ledger is inlined in `gates/pricer_branch_census.py`, the tracer is stdlib `sys.monitoring`, the AST hold is back as `--anchors`, and the reading is **59** arcs at 1ed927a — not comparable to the 64/65 the old ledger and [the roadmap](roadmap.md) record, which never reconciled with each other. Neither catches wrong arithmetic: the defective barrier line WAS executed, as a `boundary_aad` counterfactual whose value is discarded. **Executed is not observed** — `if all_hit:` was executed by nothing, and that is the line the census prints.

## A fixture must not zero the quantity its gate is sensitive to {#fixture-degeneracy}

The recurring way work with passing tests turns out wrong here is not bad reasoning and not a mis-stated assertion. It is a **fixture that sets the quantity under test to nothing**, so the gate runs, passes, and measures a term that is identically zero. Eight instances in one work stream; the already-hit barrier leg is the expensive one — a **+1432%** mark that three independent degeneracies each hid on their own:

- **`r = q = 0`.** The defect was a drift summed without its `dt`, proportional to the carry. At zero carry the same defect reads **+17.5%**, and two of the six mutants that kill it become no-ops.
- **Base valuation.** One MTM row means the `all_hit` mask is all-False at row 0, so the leg was not mis-asserted — it was **never executed**.
- **`Down_And_Out` only.** The one barrier that did run an exposure grid took the model-free zeros branch.

Two rules follow, and they are different rules.

**Vary the parameter the defect would live in, then check the mutant dies.** A mutant that survives your gate means the **fixture** is wrong, not that the code is right. Record the kill *magnitudes* in the test's docstring — the mutation prose in `tests/test_boundary_pricer_events.py` is the template — which is what makes a later "simplify this fixture" refuse itself.

**Marginal coverage is not joint coverage.** The suite already had a knock-in barrier and an exposure grid and a non-zero carry — in *different* tests. When a branch is reached only by a conjunction, one gate has to satisfy the whole conjunction.

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

`gates/fixture_degeneracy.py` is a pytest plugin that observes the calculation entry points and folds every run into a **deal type × degeneracy axis** table — no test source parsing, no hand-kept module→family map, so a new deal type appears the day its first test runs. A cell is *blind* when no run of that deal type ever made the quantity non-degenerate. Run it as `python gates/fixture_degeneracy.py tests/...`; on the fourteen barrier/option/HN/TARF/autocall modules it costs 6:39 → 7:12, so it rides an existing suite invocation rather than being a run of its own.

What **enforces** is the short `JOINT` list, not the table, and the reason is measured: with the fix's own gates removed from the row set, every *marginal* cell for `EquityBarrierOption` still reads `ok` — knock-ins existed, exposure grids existed, live carries existed. Only the three-way conjunction is unmet. **The marginal table would not have caught this defect; the joint requirement does.** The table stays a report until someone triages it into `ACCEPTED` (`--degeneracy-strict` makes it fail), because forty-five red cells on day one is a gate nobody reads.

**OPEN.** `tests/test_already_hit_barrier_leg_value.py`, the gate the `EquityBarrierOption` JOINT entry was verified against, went in the purge. Knock-in equity-barrier fixtures survive (`tests/test_hn_barrier_cmc.py`, `tests/test_hn_oss_pricers.py`), but whether any single surviving run still meets the three-way conjunction the entry requires — `single_mtm_row=False` **and** `barrier_dir='In'` **and** `carry_zero=False` — is a question for a run of the plugin, not a grep, and it has not been re-run since the purge.

Neither half can tell you a gate's *assertion* is weak — only that its fixture had nothing to assert on.

## No overengineering

One-line fixes — a default, a JSON field, a file move — before a helper module. Do not introduce abstraction ahead of a second caller.
