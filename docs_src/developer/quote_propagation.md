# Quote Propagation

[Quote sensitivities](quote_sensitivities.md) differentiate `q -> calibration -> θ*`; here the same
matrix is an operator. Between two calibrations a moved quote carries the curve to first order,

$$\theta \;\approx\; \theta^* + \frac{d\theta}{dq}\,(q_{\text{now}} - q_0)$$

so a tick reaches a valuation without a re-bootstrap, and the drift from a refit measures how stale
the calibration is. It is built for `InterestRatePrices` alone, whose fixed point is a unique root
with a square Jacobian ([the knot rule](quote_sensitivities.md#the-knot-rule)), so the implicit
function theorem is exact [@christianson1998; @henrard2014].

| where | what |
| --- | --- |
| `Quote_Propagation`, `Drift_Tolerance` | the block's switch (`No`, `Linear`) and tolerance (1e-3 percent of quote) |
| `utils.CalibrationArtifact` | one coupled set's `(θ*, J, q₀, timestamp)`: `ride`, `mispricing`, `replacing` |
| `utils.ArtifactStore` | `Config.artifacts`: 32 slots, least-recently-used |
| `InterestRateCurveParameters` | `coupled_sets`, `solve_set`, `plan_key`, `publish`, `propagate` |

Tests named below are in `tests/test_quote_propagation.py` unless another module is given; the ZAR
and USD worlds are the synthetic curve worlds of `tests/test_interest_rate_prices.py`.

## Three tiers

| tier | what | cost |
| --- | --- | --- |
| **monitor** | `dV/dq · Δq`: P&L explain, forwards | a dot product |
| **ride** | `θ* + J·Δq` reaching the pricers | a matvec, the drift check, the reval |
| **correct** | a re-bootstrap, publishing what the replaced ride was worth | the solve |

## The artifact {#the-artifact}

`utils.CalibrationArtifact` freezes one calibration of one [coupled set](#multi-curve): `θ*` in
`solve_for` order, `J = dθ/dq` there, `q₀` in percent, its `members`, and the compiled
`BenchmarkInstruments`. `timestamp` is reported, never read by the arithmetic, so a wall clock
cannot make two runs disagree.

`J` is no second solve: `calibration_jacobian` differentiates the residual at the fixed point the
forward pass found, through the `residual_jacobians` that `CalibrationSolve.backward` reads, so it
costs one Newton iteration and cannot drift from the job's own `dV/dq`. **Held by**
`test_the_artifact_holds_the_calibration_jacobian_the_solve_would_have_differenced` (against
re-bootstrapped central differences, to 1e-8 relative).

## Content addressing, and the two identities {#content-addressing}

**The slot** is `plan_key`: the set's member blocks completed by the family's declarations, the
base date, `Price Factor Interpolation` and the engine version, less the quote values and the
`lifecycle_fields` - `schema.partition_market_price`'s structural half, as `Context.plan_hash`
takes it, so the two cannot drift. Every tick of a strip lands on one slot, which is what makes a
ride possible, and re-authoring a discount strip moves the slot of every curve solved against it,
so an operator fitted to quotes that no longer exist cannot be found.

!!! warning "Everything the solve reads is in the key, including what the block does not carry"
    `Base_Date` and `Price Factor Interpolation` feed the solve from outside the block; a key
    missing either lets a second curve ride the first's operator. **Held by**
    `test_another_scheme_or_base_date_addresses_another_slot`,
    `test_a_reauthored_partner_block_takes_the_slot_with_it`. The lifecycle switches are shadowed
    out, since they move neither `θ*` nor `J` and in the key a looser tolerance would mean "refit".
    A riding set declares its solver knobs on its blocks, since one in `Bootstrapper Configuration`
    never reaches the key the ride computes.

**The `artifact_id`** is the slot plus the quotes fitted at, so it moves with every refit and names
the calibration a valuation rode. **Held by**
`test_the_artifact_key_is_plan_side_and_its_id_is_not`.

**Where it lives.** `Config.artifacts`, since an artifact holds tensors and a compiled deal tree. A
ride counts as a use, because a tick stream rides one slot while unrelated jobs publish around it;
the caller recomputes each `covering` candidate's slot, since two artifacts can cover one curve.
**Held by** `test_a_ride_counts_as_use_of_the_slot_it_rode`.

## Stateless {#stateless}

There is no `θ_current`: θ is derived per EXECUTE,

```
Config.propagated_factor(factor) -> InterestRateCurveParameters.propagate -> artifact.ride(q_now)
```

reading the market and a content-addressed slot and writing neither. Two EXECUTEs over one
`(artifact, q_now)` are bit-identical, and riding to A, B, then A returns the first answer. The key
survives a JSON round trip, or a reloaded job would miss its slot forever. **Held by**
`test_an_execute_is_a_pure_function_of_the_artifact_and_the_tick`,
`test_the_artifact_survives_a_job_document_round_trip_by_key`.

!!! danger "A miss is a refusal, never a different number"
    With no artifact answering the plan the ride raises `CalibrationStale` rather than pricing the
    curve the last bootstrap wrote, because the replay tuple is blind to which artifact was in the
    store. A run that rides reports the `artifact_id` per curve under
    `out['Stats']['Calibrations']`, the coordinate that tuple lacks. A fresh process holds no
    artifact, so its first tick refuses until the job is bootstrapped. **Held by**
    `test_a_cold_process_refuses_rather_than_pricing_something_else`,
    `test_an_evicted_slot_refuses_instead_of_silently_repricing`.

## The seam {#the-seam}

`Calculation.factor_leaf` is the one place a curve becomes a tensor, and both switches act there:

| | attachment (`Quote_Sensitivity`) | ride (`Quote_Propagation`) |
| --- | --- | --- |
| offers | `leaf + (θ* - θ*.detach())` | the ridden nodes |
| changes | what reaches `backward()` | the value |
| under `Tenor_Offset` | declines | refuses |

Off, the default, is today's path bit for bit. **Held by**
`test_the_switch_off_is_todays_path_bit_for_bit`. Both on is coherent: the splice adds zero to the
ridden value, so the backward contracts `dV/dθ` there with the `J` that put it there.

!!! warning "A `Tenor_Offset` refuses the ride rather than declining it"
    A shifted curve interpolates off coefficients fitted before the ride, so declining would price
    the stale curve, a wrong number rather than a missing derivative. **Held by**
    `test_a_tenor_offset_refuses_the_ride`.

## The drift metric, and the refusal {#drift}

Before anything prices, `mispricing` is every benchmark's residual at the ridden θ and the current
quotes over its row's largest quote sensitivity: the move in percent of quote that would close it.
A benchmark's PV is affine in its own quote at fixed θ, so
$F(\theta, q) = F(\theta, q_0) + (\partial F/\partial q)(q - q_0)$ exactly, provided
$\partial F/\partial q$ is taken at the θ scored; `residual_jacobians` re-differentiates there,
because the one stored at `θ*` reads low on sparse, sign-mixed ticks, the shapes the tolerance
exists to refuse. **Held by**
`test_the_drift_metric_is_exact_and_refuses_the_tick_shape_it_used_to_admit` (the refit's residual
to 1e-12 plus 1e-9 relative).

Since $\|\theta_{\text{ridden}} - \theta_{\text{refit}}\|_\infty \le \|J\|_\infty\,\|r\|_\infty$ to
first order, the refusal states the tolerance in basis points of zero rate. Past `Drift_Tolerance`
the ride raises `CalibrationStale`, so a caller refits rather than loses the run; a set takes its
strictest member's tolerance. A refused ride is still a plausible curve, which is why the threshold
is declared. **Held by** `test_a_small_tick_rides_and_a_large_one_refuses_in_curve_units` (at the
default on the ZAR world a 5bp alternating tick rides and 45bp refuses).

**At the refit**, `replacing` scores the displaced artifact by `θ_refit − θ_ridden` and the ridden
θ's quote-space residual, kept on the replacement's `drift`. **Held by**
`test_the_refit_publishes_the_drift_of_the_ride_it_replaced`.

## What the ride is worth {#what-it-is-worth}

On the ZAR world, seven quotes moved by alternating signs, the drift is pure curvature, so the gate
pins its second-order constants, since a first-order bug looks small too. At a 1e-2 tick, $\|\theta_{\text{refit}} - \theta_{\text{ridden}}\|_\infty/\text{tick}^2 =
8.44\text{e-}4$ and the quote-space drift over tick² is 0.0752; their third-order corrections run
opposite ways, so neither stands in for the other. **Held by**
`test_the_ride_drifts_second_order_and_the_constant_is_pinned`.

Tiers one and two agree to first order on a two-swap book, so the slope of the miss is pinned,
`ratio − 1 = 0.791 × tick`, which a monitor wrong by a constant could not produce. **Held by**
`test_the_quote_delta_predicts_the_value_the_ride_reprices`.

## The coupled set is the unit {#multi-curve}

`USD-3M` discounts on `USD-OIS`, so $d\theta_2/dq_1 \neq 0$, a column a per-block `J₂` lacks, and
the loss is invisible: a per-block `mispricing` prices against a discount curve frozen at the fit,
so it reads zero while the error grows linearly in the tick. So `solve_set` flattens a coupled set
into one Newton system and `calibration_jacobian` inverts one block matrix, `∂θ₂/∂q₁` a column of
the one `J`. On the USD world `J` is 17×17, and at a 10bp OIS tick the ridden projection curve
lands within 1e-7 of the refit and the book within 1% of its move from 9644.61 to 9621.25, the miss
second order (about 0.20 per tick²). **Held by**
`test_a_coupled_set_solves_as_one_system_and_rides_whole`.

A partial ride is unrepresentable: `Quote_Propagation` on some members of a set raises at the
bootstrap. Sets form only where something reads the Jacobian, and `Quote_Sensitivity` stops one at
a currency. **Held by** `test_a_partial_declaration_over_a_coupled_set_refuses`,
`test_the_set_is_only_measured_where_an_operator_is_asked_for`.

### The coupling is measured, not declared {#measured-coupling}

What a benchmark projects off is authored inside its own deal block, so a strip with a blank
`Discount_Rate` can still forecast off a neighbour. `BenchmarkInstruments.reads` puts every
constant on the tape and lets one backward pass say which curves the residual reached;
`coupled_sets` takes the connected components, at a compile and a backward per block, paid only
where an operator is asked for. **Held by** `test_the_coupling_is_measured_not_declared` (a 10bp
tick on one undeclared strip moves the other curve over 500bp, and the ride carries it to within
10bp of the refit).

## Value space, and the families that come next {#value-space}

θ-space scoring needs a unique root. An HW2F swaption calibration has none: its refit wanders where
the pseudo-inverse declines and [the re-solve reference is
refuted](quote_sensitivities.md#the-manifold-finding), so a θ-space gate would raise false alarms.
Every later family scores in value or residual space, asking whether the ride reprices the
benchmarks and the book as the refit does, as [the direction
check](quote_sensitivities.md#the-identified-fixture) does.

## Where this sits in the protocol {#protocol}

The fast/slow split is the service's plan/values split: [`/prepare`](../api_overview.md) names a
plan and `/execute` takes a values patch. An artifact is the market-data half, a slow object named
by plan-side coordinates with a fast path over it, and the ride writes nothing, so a patched spot
and a moved quote in one job is a reval with no re-bootstrap. **Held by**
`test_one_values_patch_carrying_a_spot_and_a_quote_composes_with_the_ride`.

!!! note "A quote is a patchable value: the split is `schema.MARKET_QUOTE_VALUES`"
    On a quote row `Quoted_Market_Value`, `Quoted_Bid`, `Quoted_Ask` and `Timestamp` are values and
    the rest is structure, so a vol tick moves `values_hash` alone and a moved pillar is a new
    plan; a `Timestamp` moves neither hash, a market's identity being its numbers. A table carries
    values when its row declares all four keys: `Points` (`InterestRatePrices`, `FXVolPrices`)
    and `European_Options` (`LogVar2FJModelPrices`). `HullWhite2FactorModelPrices`,
    `CSForwardPriceModelPrices` and `GBMAssetPriceTSModelPrices` declare none and stay wholly
    plan-side. The tick guard, `Context.plan_hash`, `market_patch` and `plan_key` read that one
    tuple; the projection drops the value keys rather than shadowing them, since a pillar that
    starts being quoted two-sided is the same node of the same plan. **Held by**
    `test_market_prices_partition.py::test_every_family_partitions_by_the_one_rule_and_round_trips_exactly`,
    `::test_the_tick_guard_and_the_partition_draw_the_same_line`.

## Non-goals {#non-goals}

- **No second-order ride.** It needs `create_graph` through `CalibrationSolve`, which neither
  supports nor detects it ([quote sensitivities](quote_sensitivities.md#non-goals)).
- **No automatic refit.** When to recalibrate is desk policy.
- **No persistence.** A serialised artifact would be a second definition of the plan, so a restart
  [refuses its first tick](#stateless).
- **No cross-currency set, no ride off an amount-valued quote.** `BenchmarkInstruments` has one
  reporting currency, and an `FXForwardDeal` outright lands in `Buy_Amount`, not a schedule column,
  so its block refuses rather than publish a zero row of `dF/dq`. **Held by**
  `test_interest_rate_prices.py::test_a_forward_block_cannot_publish_a_ride_operator`.
