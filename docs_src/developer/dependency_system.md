# Dependency System

`Config.calculate_dependencies` (`config.py`) is the compiler front end: it discovers every price factor the book touches, wires each to its sub-factor dependencies, orders them, and splits stochastic vs static. Discovery is table-driven — three plain-dict **registries** decide which factors exist; the code around them just walks the tables.

It is two methods: `discover_factors` does everything except the split, `find_models` does the split. Both read the loaded config and write nothing (see the idempotence box on [Calc Lifecycle](calc_lifecycle.md#compile-phase-1-calculate_dependencies)), and they stay separate because `Config.validate` / `Config.factor_universe` want the universe without resolving models. Iterating the `dependent_factors` dict `discover_factors` returns is iterating the topological order.

[Cross Factor](../calibration/cross_factor.md) covers the composed-spot name-prefix chain and the sim-time buffer publish/consume from the calibration angle; this page is the discovery/ordering view. Read them together — neither restates the other.

## The three registries

All three are plain dicts defined inside `discover_factors`. Extend by adding a row.

**`dependant_fields`** — `{factor_type: [(price_factor_key, linked_factor_type), …]}`. The edge generator of the DAG: for a factor it reads `Price Factors[name][price_factor_key]` and, if present, builds `Factor(linked_type, check_rate_name(value))` as a dependency, **recursing** if the linked type is itself a `dependant_fields` key. Chains like `CommodityPrice → {Interest_Rate, Forward_Rate, Currency}` and `ReferencePrice → ForwardPrice → FxRate → InterestRate`. There is **no visited-set** — termination relies on the registry being acyclic.

**`nested_fields`** — `{head_type: tail_period_type}`. Governs the positional **name-prefix chain**. Identity for a curve (`InterestRate: InterestRate`); type-switching for a 0D spot (`CommodityPrice`/`EquityPrice`/`FxRate → ObservedBasis`). Consumed by `update_nested_rates`: for each prefix length it registers a period keyed to a single-parent dependency (`head → tail(2) → tail(3) …`). On a genuine type switch it **pops the full-name head key** — so `CommodityPrice.PLATINUM_CME.LME_CME` never exists as a factor; the real pair is `CommodityPrice.PLATINUM_CME` (the spot, carrying the dependant/conditional keys) and `ObservedBasis.PLATINUM_CME.LME_CME` (the tail, depending on the spot). This is what topo-orders `ObservedBasis` **after** its parent, which `BasisLinkedSpotModel.generate` relies on.

**`conditional_fields`** — `{factor_type: lambda(instrument, factor_fields, params) → [Factor, …]}`. Instrument-dependent extra factors (FXVol, Correlation, `<SpotModel>ModelParameters`). Each returned factor is appended as a dependency **and** registered as its own key with an empty dep list. Because the lambda reads the instrument, these are re-evaluated per occurrence.

### `Chained_Basis` — the fourth inclusion mechanism

A declared field on an `ObservedBasis` block naming the next link of a chain that must **close** — the AM/PM session pair is the 2-cycle. An **open** link riding another factor's finished path is the linked-parent family (`BasisLinkedSpotModel`) and does not declare the field. Discovery walks each declared chain and refuses one that does not return to its start or revisits without closing. `add_chained_bases` runs at the factor's **own** chain registration, so whenever any member of a closed chain enters the universe every member follows with its own positional chain, under **every** calculation — a book pricing one session cannot silently drop the loop. Not every type-switched tail gets it: the `dependant_fields` linked-factor branch registers one with no chain walk or member pull, so a composed spot reached *only* as another factor's `Currency`/`Interest_Rate` link does not pull its chain.

Each member also declares **`Chained_Lag`**, the rows back at which its law references its link.

- **Same-row** (lag 0, the default) enters the graph as a real edge, because positional depth cannot order it: `topological_sort` emits whole chains **within a pass in insertion order**, so a member pulled by the declaration (always inserted last) leapfrogs its depth. A book entering from the same-row side alone — the production shape, PM-session tradables only — generated the bridge before its source until the edge existed.
- **Lagged** is the chain's day boundary: the member steps off the link's *previous* row, the acyclic bridge factorization carries the dependency, no edge needed.

A closed chain must lag somewhere; all same-row is a same-instant loop and refuses in discovery, named, before the sort would refuse its cycle namelessly. `ChainedBasisModel` asserts its own link is same-row. Links must stay on one primary and differ from each other. Gated by `tests/test_chained_basis.py`: the cross-branch pair (`LBMA_AM.PM ↔ LBMA_AM.CME`) is the inclusion arm that cannot pass positionally, the same-row entry (`LBMA_AM.PM.CME`) the ordering arm, whose killing mutations are the lag-0 edge dropped and the lag declared on the wrong member.

!!! warning "Invariant — `dependant_fields` must stay acyclic"
    `get_rates` recurses over `dependant_fields` with no visited-set guard. A cyclic entry is unbounded recursion (`RecursionError`), **not** the clean `RuntimeError` `topological_sort` raises for graph cycles.

!!! warning "Invariant — type-switch head/tail rule"
    For a type-switching `nested_fields` entry (`tail_type != head_type`), the **head** period carries the dependant/conditional keys and the tail periods depend positionally on their own prefix. The **virtual full-name head key must be dropped everywhere** — `get_rates` pops it, `get_price_factors` pops its tenor. Code reading `dependent_factors` must not assume the dotted full name is a key; only the head spot and the `ObservedBasis` tail exist.

!!! warning "Invariant — `conditional_fields` types must not overlap `dependant_fields` keys"
    The conditional branch overwrites a factor's dep list with `[]` (via `update`, not `setdefault`), so a type in both registries would have its real dependencies silently erased. Also: any conditional lambda for a type reachable via a bare-`{}` sentinel factor (`FxRate`, `InterestRate`, `SurvivalProb`) must `getattr`-guard `instrument.options` / `.field` — a raw `AttributeError` there is not caught by `add_rates_for_factor`'s `KeyError` handler.

## The walk

`walk_groups` recurses the deal tree **depth-first, children before parent**, skipping `Ignore=='True'` nodes and nodes that never became a `Deal` (an `Object` naming no class loads as `{}`; `Context.validate` reports it by position). Per instrument it runs `instrument.reset(holidays)` and `finalize_dates(...)` (filling `reval_dates` / `settlement_currencies`), then `get_price_factors`, which iterates `instrument.factor_fields` (`{field_name: [factor_type, …]}`); `iter_factors` pulls the field value(s) via `utils.get_fieldname` (handling nested-tuple keys), flattens, and yields `Factor(type, check_rate_name(v))`. Per factor: record its per-deal max date (`max(instrument.get_reval_dates())`) into `dependent_factor_tenors`, and add its rates unless already present or its type is conditional.

`add_rates_for_factor` calls `get_rates`; on a `KeyError` (missing `Price Factors` block) it logs a warning and skips the factor. Nothing self-heals — the one auto-created default block was `DiscountRate`, a retired type.

!!! warning "Invariant — a missing block is a silently skipped factor"
    A factor whose `Price Factors` block is missing is dropped (two log lines), absent from `dependent_factors`, never simulated or valued. If a new derived type needs a default block, extend `add_rates_for_factor` explicitly. `discover_factors` returns the skipped names because nothing else can recover them; `Context.validate` reports them alongside the discovered factors that have no block, which is the *other* way a factor goes missing — those reach `construct_factor` and fail there instead.

!!! warning "Invariant — dates before tenors"
    `get_reval_dates` / `finalize_dates` must run (via `walk_groups`) before tenor collection: the per-factor max date and the reset/settlement sets all come from `instrument.reset()` + `finalize_dates`. A deal whose `reset()` leaves `reval_dates` empty contributes no tenor, and its directly-referenced factors default to `max(reset_dates)`.

The main body seeds base-currency FX first, walks the book, adds report currency (linked to base), then optional CVA `SurvivalProb` and FVA/deflation curves (`add_interest_rate` pins a curve plus all transitive dependents to `reset_dates`).

!!! warning "Invariant — base currency sorts first, stays static"
    Base-currency FX is appended to every other `FxRate`'s dependency list and excluded from the stochastic set (`find_models`). Keep base a static, dependency-of-all-FX anchor.

## Ordering — `topological_sort`

Edges collected as `dependent_factors` (factor → list-of-prerequisites) are `topological_sort`'d (`utils.topological_sort`): a repeated-pass Kahn variant that, **within a pass, emits nodes in dict-insertion order** and moves every node whose edges all point outside the still-unsorted set. Dependencies land first, dependents follow, cycles raise `RuntimeError`.

`traverse_dependents` (`utils.py`) fans a factor's tenor out to all transitive dependents — BFS, `seen`-guarded.

!!! warning "Invariant — throwaway graph, cycle behavior"
    `topological_sort` rejects cycles with `RuntimeError` and **destroys its input dict** — pass a rebuilt/throwaway graph. `traverse_dependents` yields transitive dependents **excluding** the start node; do not rely on the node appearing in its own output.

## Stochastic vs static split — `find_models`

`find_models` walks the topo order, resolving each factor's process via `Model Configuration.search` (`modelfilters` first-match, else `modeldefaults`; subtype-aware). A factor is **stochastic** iff a process was found, `name[0] != Base_Currency`, and the model's parameters are available — a `Factor(stoch_proc, name)` entry in `Price Models`, or for an implied model the `Price Factors` block of the factor it implies off. Implied models also pull their parameter factor as an additional **implied** factor (`additional_factors` → `implied_factors`, leaves minting into `implied_var`), *not* a static one; it lands in `static_factors` only when a conditional field independently discovered it, the case the implied-leaf dedupe covers. The distinction is load-bearing: implied leaves gate on `sensitivities in ['All', 'Implied']`, static ones on `['All', 'Factors']`. Downstream, `static_factors = set(dependent_factors) - stochastic_factors.values()`.

!!! warning "Invariant — a lost process silently becomes static"
    A factor that resolves to a process but whose `Factor(stoch_proc, name)` is **absent** from `Price Models` is not added to `stochastic_factors`; via the set-difference it falls into `static_factors` and is frozen at its current value. Only a warning (`len(name)>1`) / error (`len==1`) log distinguishes "intended static" from "lost its process."

## RNG ordering {#rng-ordering}

!!! danger "Invariant (WARNING) — insertion order into `dependent_factors` is load-bearing for reproducibility"
    `topological_sort` tie-breaks equal-depth factors by **insertion order**. That order flows through `find_models` (`setdefault`) → `stoch_factors` → `get_cholesky_decomp`'s `process_ofs` → the RNG-substream / correlation column each process reads from `t_random_numbers`. Permuting equal-depth factors — by changing **deal-walk order, `factor_fields` dict order, or `dependant_fields` list order** — moves realized results bit-for-bit. **Preserve all three, or treat any reordering as a results-changing event and say so in the commit.**

    A permutation is statistically harmless (the cholesky rows permute consistently, so the joint distribution is identical) and the same JSON reproduces the same order, so user-facing reproducibility holds for free. What ordering stability buys is the developer oracle: bitwise "no reported number moved" verifies a refactor in minutes rather than by statistical comparison, which is why the replay contract keys on `engine_version`. The HMM regime path is a second, parallel surface: a separate Sobol `quasi_rng` stream whose batch counter is never reset by `reset()`.

## The process protocol {#the-process-protocol}

Every stochastic process implements a small verb set, so the calc/solver loop uniformly across model worlds and never branch on model type. Every model-specific buffer key is owned by the process under the `(factor_key, kind)` convention.

**Core verbs**, all processes: `num_factors()`, `precalculate(...process_ofs...)` (sets `z_offset`), `correlation_name`, `generate(shared_mem)` (dispatch on `Z.ndim` — outer/inner contract on [Calc Lifecycle](calc_lifecycle.md#valuation-modes)).

**Extension verbs**, inert no-ops on the base except where noted: `reveal_state_at` (a working generic default — the factor's buffer value at `t`, tagged `REVEAL_CONTINUOUS`), `inner_fork_seed`, `print_seed` (publishes first-step conditioning for a factor this process informs, off a state snapshot; keys are foreign, so callers publish every seed before any generate and drop them after the run generates), `outer_reseed`, `reseed_from_path`, `reseed_inner_state`, `diff_state_leaves`, `calibrated_annual_vol` / `revealed_annual_vol`, `privileged_factors` / `privileged_layout`, `calc_references` / `link_references`, `copy` (returns `copy.copy(self)`).

Two with their own contract:

- `basis_decay` **raises** on the base rather than lending another law's projection: a basis process either states its own `(phi, lam)` pair or its legs do not price.
- `bridge_variance_rate` returns an annualized log-variance RATE against elapsed time, not a per-interval array, because the deal grid and the scenario grid are different axes and an array would be indexed by one and reached with the other. A process whose interval variance is not proportional to elapsed time returns `None` and the pricer observes endpoints.

## Extension recipes {#extension-recipes}

Registries, not functions. Each recipe is "which registry / attribute to touch."

| Add a… | Touch |
| --- | --- |
| **factor type** | If it embeds other factors: a `dependant_fields` row. If its name is a positional chain: a `nested_fields` row (identity = curve, distinct tail = 0D spot) — and the matching head/tail literal at each `calc_factor_code_chain` call site in `instruments.py`. If it is instrument-conditional (vol/correlation/model params): a `conditional_fields` lambda. Register its process in `Model Configuration` (`modeldefaults` / `modelfilters`). |
| **process** | A class in `stochasticprocess.py` implementing the [process protocol](#the-process-protocol), constructed by name via `globals()` dispatch from `Price Models`. Ensure `num_factors() == len(correlation_name[1])`. Pair it with a `*Calibration` class registered in `calibration_config.json` — see [the calibration contract](../calibration/contract.md). |
| **deal** | A `Deal` subclass in `instruments.py` with a `factor_fields` class attribute (field → factor type) so discovery finds its factors, plus `calc_dependencies` (builds `Factor_dep` via the `get_*` resolver layer), `calc_time_dependency` and `calculate`; dispatched by name. Declare its JSON as a per-class `fields = [ADMIN, …, own('<Class>', [F(…), …])]` list (`schema.py`), reusing the shared `Group` constants (see [Conventions](conventions.md#documentation-and-doc-generation)). That declaration IS the schema: nothing else needs editing except one row in `schema.mapping['Instrument']['groups']`, the create-deal menu and the only hand-kept part. Set `accepts_children = True` if the deal breaks down into simpler instruments. Mark a field the deal cannot price without as `default=REQUIRED`; a rule spanning two fields goes in a `validate()` yielding one message per violation (`schema.validate_instrument`). |
| **decision on simulated state** (a barrier crossing, an autocall trigger, an exercise) | A `utils.BoundarySet` subclass registered onto `shared.boundary_sets` from the pricer, gated on `shared.boundary_aad`. Pick the subclass by how far the decision REACHES — later rows (`LatchedBoundarySet`, whose optional `own_row` override carries the decision's own row) or inside an inner MC (`InnerBoundarySet`) — see [Boundary corrections](calc_lifecycle.md#boundary-corrections-the-sensitivity-subsystem). The assembler is model- and deal-agnostic; nothing there should learn about your deal. |
| **valuation option** | A field on the `Calculation` block, honored inside the relevant `Calculation` class. A new whole mode is a class dispatched by `run_job` / `construct_calculation`. Do **not** touch `Credit_Monte_Carlo`'s CVA/FVA block ([Change Scope](conventions.md#change-scope)) — the sole exception, and the bar for any other, is a term that provably cannot move a reported number. |

Before writing any of these, search `utils.py` and the package for an existing equivalent — see [Conventions — look before you write](conventions.md#look-before-you-write).
