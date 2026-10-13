# Resolver Layer

The generic `get_*` layer in `instruments.py` is the **only** place a factor name is decomposed. A deal's `calc_dependencies` runs here to turn `Factor` keys into the flat **code** tuples pricing consumes off the runtime buffers. [Cross Factor](../calibration/cross_factor.md) walks the positional name-prefix chain and the sim-time summation from the calibration side; this page is the resolver-internals view.

## Name normalization

`utils.check_rate_name` (a tuple is upper-cased; a dotted string is split as written) and `check_tuple_name` (its inverse) are the sole string↔tuple bridge; `check_tuple_name` also builds every fail-loud error string and every `Price Factors` / `Price Models` lookup key. `Deal.check_factor_name` reads a field through `check_rate_name`, answering a fallback where the field is blank. `BASIS_COMPOSABLE_TYPES`, derived from `utils.NESTED` as the heads whose tail is `ObservedBasis` (`FxRate`, `EquityPrice`, `CommodityPrice`), marks the 0D spot types whose name may carry a composed `primary + ObservedBasis…` chain.

## The code tuple

`calc_factor_index(field, static_offsets, stochastic_offsets, all_tenors={})` is the atom. It returns `tuple([stoch_bool, field(=Factor), subtype] + all_tenors.get(field, []))`, static checked first, else raises `Exception('Cannot find <name>')`. The slots are named constants in `utils.py`:

| index | constant | meaning |
| --- | --- | --- |
| 0 | `FACTOR_INDEX_Stoch` | `True` stochastic / `False` static |
| 1 | `FACTOR_INDEX_Offset` | the **`Factor` key** (a misnomer — not an int) |
| 2 | `FACTOR_INDEX_SubType` | factor subtype |
| 3 | `FACTOR_INDEX_Tenor_Index` | `CurveTenor` payload (curves) |
| 4 | `FACTOR_INDEX_Daycount` | daycount closure (curves) |

A vol code reads the same tail positions under its own names (`FACTOR_INDEX_Moneyness_Index` 3, `FACTOR_INDEX_Expiry_Index` 4, `FACTOR_INDEX_VolTenor_Index` / `FACTOR_INDEX_Flat_Index` 5).

!!! warning "Invariant — the code-tuple layout is fixed"
    `[0]` stoch bool, `[1]` a single `Factor` key, `[2]` subtype, `[3+]` tenor payload from `all_tenors`. `calc_factor_index` is the sole producer. The stable hashable identity is `x[:2] = (stoch, Factor)`; indices `3+` are unhashable.

## The positional chain — `calc_factor_code_chain`

`calc_factor_code_chain(head, fieldname, …)` resolves a prefix chain, its tail read off `utils.NESTED`: `[calc_factor_index(Factor(head if x==1 else NESTED[head], fieldname[:x]), …) for x in 1..len]`. A single-element name yields a single-element code — the **bit-identical single-factor path**. Five getters pass their head: `get_interest_factor` and `get_inflation_factor` (a curve and its spread chain), and `get_fxrate_factor` / `get_commodity_rate_factor` / `get_equity_rate_factor` (a spot and its `ObservedBasis` tail).

!!! warning "Invariant — the positional chain"
    Period 1 = head type on `name[:1]`; prefix `k>1` = tail type on `name[:k]`. Curve ⇒ `head==tail` (spread chain); 0D spot ⇒ `tail=ObservedBasis`. Discovery and resolution read the pair off one registry, `utils.NESTED`, and the virtual full-name head key for a type-switched spot (`CommodityPrice.X.Y`) must never be looked up (discovery intentionally drops it). **Held by** `tests/test_chained_basis.py::test_discovery_and_the_resolver_read_one_chain_registry`: a one-deal document per head naming it composed, its universe holding the chain and the deal pricing (killing mutation: the resolver reading a tail of its own for one head).

## Object-graph hops — `get_factor_component`

`get_factor_component(componentname, all_factors)` implements the **primary-spot** rule — `primary = name[:1] if type in BASIS_COMPOSABLE_TYPES else name` — returning the underlying factor **object**. Repo/carry/currency read off it (`get_equity_currency_factor`, `get_equity_zero_rate_factor`, `get_commodity_zero_rate_factor` → `.get_repo_curve_name()` / `.get_currency()` → `get_interest_factor`).

!!! warning "Invariant — two distinct parent rules, both correct"
    Object lookups (repo/carry/dividend/currency) resolve off the **ultimate** primary `name[:1]` via `get_factor_component`. The `ObservedBasis` process links off its **immediate** parent `name[:-1]` (`BasisLinkedSpotModel.calc_references`). Both are correct for their jobs; do not conflate them.

## Getter taxonomy

The two trailing args tell you the axis:

- takes `all_tenors` → **curve-shaped**, needs a tenor payload appended (interest / forward-rate / zero / survival / inflation / dividend / all vol getters).
- takes `all_factors` → must inspect a factor **object** to hop to a linked factor (repo, currency, recovery, spot value, model params).
- takes neither → pure 0D **spot** code, no tenor (`get_fxrate_factor`, `get_commodity_rate_factor`, `get_equity_rate_factor`, `get_price_index_factor`).

## The list-at-index-1 idiom — `get_spot_model_params_factor`

`get_spot_model_params_factor(spot_model, name, all_factors, …)`: `spot_model=='None'` → `None` (GBM); switch-on-but-absent → `KeyError`, which the compile turns into that one deal skipped by name, never a GBM fallback. An unknown model never reaches here — `Deal.__init__` refuses it against the type's `spot_models` declaration. Otherwise it returns a single-element code whose **index 1 is a *list* of per-parameter sub-factors**, whose **index 2 (subtype slot) is the model name** — the same shape as the SVI/Skew vol code in `get_vol_factor` — and whose index 3 is the factor's structural facts (`curve_tenors()`: each curve's knots and the non-leaf scalars, `Steps_Per_Year` among them). **Held by** `tests/test_cross_spot_model.py::test_the_trap_is_closed` and `::test_a_key_that_cannot_resolve_skips_the_deal_and_never_kills_the_job`.

!!! warning "Invariant — `t_Buffer` cache-key discipline"
    Cache keys slice `x[:2] = (stoch, Factor)` and exclude indices `3+`, which requires index 1 be a single `Factor`. SVI and Skew vol codes and spot-model parameter codes (LogVar2FJ's) carry a **list** there and are consumed only by the dedicated vol and spot-model paths that tuple-flatten it (`VolSurface.rate`) or read `t_Static_Buffer` directly. Do not route a list-shaped code through the generic `(stoch, Factor)` key path. Not gated.

## `field_index` == `Factor_dep`

!!! warning "Invariant — `field_index` is stored and consumed as returned"
    The `field_index` dict of codes a deal's `calc_dependencies` returns **is** `DealDataType.Factor_dep`: `add_deal_to_structure` stores it after `utils.bind_schedules` binds each `TensorSchedule` in it to the device in place, keys and codes unchanged, and the pricer reads `deal_data.Factor_dep['Commodity']` etc. as stored.

## Pricing consumption

- `calc_time_grid_spot_rate(rate, …)`: spot = **sum** over code elements of the gathered component (stochastic via `gather_scenario_interp`, static via `reshape(1,-1)`), reading `t_*_Buffer[r[FACTOR_INDEX_Offset]]`. Cache key `('spot', tuple(tuple(r[:2]) for r in rate), time_grid[:, :TIME_GRID_ScenarioPriorIndex+1].tobytes())` — the whole read, since two reads of one day may sit at different places on the path.
- `calc_time_grid_curve_rate(code, …, n_batch_dims=1)`: builds a `TensorBlock`; `make_curve_tensor` collapses extra trailing batch axes (inner-MC `(B,B2)→B*B2`) when `n_batch_dims>1`, keeping the interp stack single-trailing-batch. The buffer value is a tensor or a `ScenarioSource` (published only by an inner-MC fork); `build_interpolation` turns either into the strategy that reads it. `n_batch_dims>1` gathers happen during `generate`, before a fork publishes, so they only ever see one block. See [Calc Lifecycle](calc_lifecycle.md#inner-mc-subsystem).
- `TensorBlock.gather_weighted_curve`: sums `zip(curve_tensors, code)` (parent curve + basis spreads); a static component takes the risk-neutral branch when `shared.riskneutral` is set.
- A floating leg's resets: the known resets carry the scenario axis and the forecast curve's unknown ones carry the curve's — one column where the curve is static, the declared "intended static" of `find_models`. An interest-rate forecast (`TensorResets.get_simulated_resets`, the floating leg's and the autocall's) joins through `utils.concat_resets`, which broadcasts the one column, so a frozen forecast curve prices inside a simulated netting set and the leg names the static curve at INFO. An FX-resetting nominal (`pv_float_cashflow_list`) and an energy leg join through `utils.join_resets`, which refuses that shape by name. **Held by** `tests/test_loud_refusals.py::test_an_fx_resetting_nominal_off_a_frozen_pair_is_refused_by_name` and `::test_an_energy_leg_forecasting_off_a_frozen_curve_is_refused_by_name`. Not gated: the interest-rate broadcast.

!!! warning "Invariant — composed value is a sum of gathered components"
    Spot = primary + basis; curve = parent + spreads. The sum is order-agnostic; ordering (primary first) matters only for the single-element bit-identity guarantee and the risk-neutral static-branch check.

!!! warning "Invariant — fail-loud, end to end"
    An unknown/missing factor raises `Exception` / `KeyError` / `ValueError` naming it via `check_tuple_name`. There is no silent GBM/zero fallback anywhere in the resolver or the pricer.

## The compressed contract

```
JSON string
  → check_rate_name → Factor(name tuple)
  → [discovery: nested/dependant/conditional + topo]        (Dependency System)
  → [construct: stoch/static/all_factors/all_tenors]        (Calc Lifecycle)
  → [resolve: get_* → code [stoch, Factor, subtype, *tenor]] (this page)
  → field_index == Factor_dep
  → [generate: t_Scenario_Buffer[Factor] — one row block, or a fork's ScenarioSource sequence]
  → [price: calc_time_grid_*_rate sums components, caches in t_Buffer by (stoch,Factor)]
```
