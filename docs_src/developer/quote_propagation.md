# Quote Propagation

[Quote sensitivities](quote_sensitivities.md) built the arrow `q -> calibration -> θ*` and
differentiated it. There the derivative is a **reporting** object; this page is the same matrix used
as an **operator**. Between two calibrations a quote that moves carries the curve with it to first
order,

$$\theta \;\approx\; \theta^* + \frac{d\theta}{dq}\,(q_{\text{now}} - q_0)$$

so a tick reaches a valuation without re-bootstrapping, and the drift between the ridden curve and a
true refit is a **measurement of how stale the calibration was** rather than a schedule someone
guessed. Predict, ride, correct.

| tier | what it is | cost |
| --- | --- | --- |
| **monitor** | `dV/dq · Δq` — P&L explain, forwards | a dot product, no reval |
| **ride** | `θ* + J·Δq` reaching the pricers | a matvec plus the reval |
| **correct** | a re-bootstrap, publishing what the ride it replaced was worth | the solve |

Measured on the ZAR strip: a refit is **594 ms**, a ride **74.5 ms** (8×), the operator itself
**0.02 ms**.

!!! note "Built for the curve family only"
    `InterestRatePrices` is the family this lands on, deliberately: its fixed point is a unique root,
    its Jacobian is square by [the knot rule](quote_sensitivities.md#the-knot-rule), and the implicit
    function theorem is therefore **exact** rather than an approximation of a manifold. Nothing in the
    artifact assumes squareness, but the vol families' drift has to be scored in [value
    space](#value-space).

    The unit is the **coupled set**, not the block: blocks whose residuals read each other's curves
    solve as one system and ride as one operator, so `∂θ₂/∂q₁` is a column of the published `J`.
    Every block of such a set must declare `Quote_Propagation` or none may —
    [measured](#multi-curve).

!!! note "Honest reading — the operator is free and the SAFETY CHECK is the whole cost"
    **97% of a ride is `mispricing`**, not `θ* + J·Δq`: what a tick pays for is finding out whether the
    ride was allowed — one residual evaluation and one backward pass per benchmark, since the metric is
    [re-differentiated at the ridden θ](#drift) rather than read off a frozen `dF/dq`. The honest number
    costs 71.7 ms against a 594 ms refit, so buying it back would save 8× at the price of a metric that
    [under-reads by up to 11% on some tick shapes](#drift). The next optimisation is therefore checking
    the drift on a schedule rather than at every leaf mint. A coupled set is scored once **per member
    curve** the valuation mints, so an N-curve set pays the check N times.

## The artifact {#the-artifact}

`bootstrappers.CalibrationArtifact` is one calibration of one **coupled set** frozen as an operator:
`(θ*, J, q₀, timestamp)`, its `members`, and the compiled `BenchmarkInstruments` the first two were
read off. `θ*` is the solved node vector in `solve_for` order, `J = dθ/dq` at that fixed point, `q₀`
the quote vector it was fitted at in percent.

`timestamp` is **reported, never read by the arithmetic**: every ride, refusal and refit names the
artifact it rode and when that artifact was fitted, because "how stale" is the question the drift
number answers. It reaches no number and no hash, so a wall clock cannot make two runs disagree.

**Where `J` comes from — the same differentiation, not a second solve.** Nothing is lifted out of
`CalibrationSolve.backward`'s `ctx`; what is shared is the FUNCTION. `calibration_jacobian` calls
`residual_jacobians` once more at the fixed point the forward pass already found, solving the same
`n × n` system against every column of `dF/dq` at once instead of one cotangent. The cost is **one
Newton iteration's worth** and there is **no second root find** — only the residual is differentiated —
so a published `J` cannot drift from the `dV/dq` the same job reports. Measured against two full
re-bootstraps per quote on the ZAR round-trip world: **1.07e-12** absolute on columns of norm 1.3e-2,
the finite difference's own resolution.

## Content addressing, and the two identities {#content-addressing}

The artifact is a **plan-side** object with two names.

**The slot** — `plan_key` — is every member block of the coupled set, the base date, the interpolation
scheme and the engine version, with the quote VALUES and the `lifecycle_fields` projected out. That is
literally `schema.partition_market_price`'s structural half, the same projection `plan_hash` takes over
the section, so the two cannot drift. Every tick of one strip lands on the **same slot**, which is what
makes a ride possible at all; a row that gains a `Quoted_Bid` keeps its slot for the same reason a
moved mid does. The slot names the SET, so re-authoring a discount strip moves the slot of every curve
solved against it — an operator whose `J` was fitted against quotes that no longer exist must not be
findable by a curve it still covers.

!!! warning "Everything the SOLVE reads has to be in the key, including what the block does not carry"
    `Base_Date` and `Price Factor Interpolation` are inputs to the solve and live in `System
    Parameters` and a `ModelParams`, not on the block — and a key missing one is a key two different
    curves share, with the second silently riding the first's operator. Both were measured doing exactly
    that: two jobs 45 days apart shared a slot, and a linearly-interpolated job rode a Hermite solve
    **0.53bp** away from its own answer. Gated in both directions — the same job under the same scheme
    has to *keep* its slot, or the fix is just "never find anything".

**The `artifact_id`** is the slot plus the quotes it was fitted at. It moves with every refit, so a
propagated valuation is labelled by the calibration it rode plus the tick it rode — a replay coordinate
rather than a timestamp anyone has to trust. Two identical refits produce the same id.

!!! warning "A lifecycle switch cannot be in the slot it governs"
    `Quote_Propagation`, `Drift_Tolerance` and `Quote_Sensitivity` are shadowed out of the key
    (`lifecycle_fields`). They are read when an artifact is published or ridden rather than when it is
    fitted, and none can move `θ*` or `J`. Leaving them in would make loosening a tolerance silently
    mean "refit" instead of "allow more".

**A slot is not a plan id, and the sharing is deliberate.** `plan_hash` names a whole job — deals,
calculation, price factors, market prices — while a slot names only the calibration inputs, so two jobs
with different `plan_hash`es (a different book, netting set or reporting currency) legitimately ride
**one** artifact. A calibration is a property of the market data, not of the trades priced against it.

**Where it lives.** `bootstrappers.ARTIFACTS`, in process, bounded and least-recently-used — the plan
cache's discipline. It holds tensors and a compiled deal tree, so neither `Price Factors` (data, written
back out as JSON) nor a file is an option. LRU rather than FIFO is a correctness property and gated as
one: a tick stream rides one slot over and over while unrelated jobs publish around it, and FIFO throws
that slot out on schedule. `covering(factor)` returns **candidates** most-recently-used first and picks
none — the caller recomputes each one's slot against the market data standing now, because two artifacts
can cover one curve at once.

## Stateless {#stateless}

There is no `θ_current` anywhere — the claim is grep-able. `q_now` arrives as the **values patch** and
θ is derived per EXECUTE as a pure function of `(artifact, q_now)`:

```
Config.propagated_factor(factor) -> InterestRateCurveParameters.propagate -> artifact.ride(q_now)
```

reading `Market Prices`, the base date, the interpolation scheme and a content-addressed slot, and
writing none of them. Two EXECUTEs over one `(artifact, q_now)` are bit-identical by construction;
riding to A, then B, then back to A returns A's first answer to the last bit. An artifact is
**replaced, never edited** — a refit publishes a new one into the same slot.

What DOES have to survive a JSON round trip is the **key**, computed off the block the decoder rebuilt:
a `Market Prices` block carries Timestamps, DateOffsets, DateLists and Percents, and a decoder that
rebuilt any of them differently would miss the slot and refuse forever. Gated.

!!! danger "A MISS IS A REFUSAL, never a different number"
    `Quote_Propagation: 'Linear'` with no artifact answering to the plan raises
    `utils.CalibrationStale`. It does not fall back to the curve the last bootstrap wrote: **a plan miss
    is a 404**, the same house rule the plan cache is held to.

    That fallback was the shipped behaviour and it is the one thing the replay tuple cannot describe. An
    artifact evicted by unrelated jobs filling the store repriced a book by **13.4%** while `plan_hash`,
    `values_hash`, `__version__` and `Random_Seed` were all identical across the two runs — no
    coordinate could have told them apart. A refusal is not a number, so it cannot be mistaken for one.

    The other half of closing that hole is that a run which *does* ride **reports the `artifact_id`** it
    rode, in `calc_stats['Calibrations']` keyed by curve, so `out['Stats']['Calibrations']` is the
    missing replay coordinate.

    **The cold start follows from it.** An artifact cannot be serialised, so a fresh process has none
    and the first tick after a restart refuses until the job is bootstrapped — which publishes, and the
    same EXECUTE then runs. Not a gap to close: a re-bootstrap rederives the artifact from the job
    document, and the artifact was never the record of anything.

## The seam {#the-seam}

`Calculation.factor_leaf` — the same one the quote-sensitivity attachment uses, and for the same
reason: it is the ONE place a curve becomes a tensor. The two offers differ in what they change:

| | attachment (`Quote_Sensitivity`) | ride (`Quote_Propagation`) |
| --- | --- | --- |
| offers | `leaf + (θ* - θ*.detach())` | the ridden nodes as `current_val` |
| changes | what reaches `backward()` | **the value** |
| under `Tenor_Offset` | declines | **refuses** |

`Quote_Propagation` defaults to `No`, and off it is today's path bit for bit — same curve, same marks,
store never consulted.

**Both switches on is the coherent combination, not a collision.** The leaf's VALUE is the ridden curve
and the splice contributes zero to it, so what the backward reports is `dV/dθ` at the ridden curve
contracted with `dθ/dq` at `q₀` — the artifact's own `J`, the operator that put the curve there. Value
and derivative are then the same first-order object, which is what makes tier one comparable with tier
two.

!!! warning "A `Tenor_Offset` refuses the ride rather than declining it"
    `current_value(offset)` interpolates off Hermite coefficients fitted on the numpy rate column at
    construction, so a shifted curve cannot be ridden without refitting them. Declining would silently
    price the STALE curve — a wrong number, not a missing derivative — so it raises.

## The drift metric, and the refusal {#drift}

**At EXECUTE**, before anything is priced: `mispricing` is every benchmark's residual at the ridden θ
and the current quotes, divided by that benchmark's own quote sensitivity — a **quote-space** number,
the move in percent that would close it. It costs no refit, because a benchmark's PV is AFFINE in its
own quote (measured in `_carry_quotes`, second difference exactly zero), so
$F(\theta, q) = F(\theta, q_0) + (\partial F/\partial q)(q - q_0)$ needs no re-authoring of the compiled
set. `dF/dq` is diagonal on this family, measured at **0.0** off-diagonal, and the normaliser is a row
max rather than a diagonal so a family whose benchmarks are not one-quote-each stays expressible.

That identity is exact in `q` at **fixed θ** — *provided `dF/dq` is the one at the θ being scored*. So
`residual_jacobians` re-differentiates it here, at the ridden θ, one backward pass per benchmark off the
forward pass the residual comes out of. The number is then the drift, not an estimate of it.

!!! danger "The frozen-`dF/dq` proxy reads LOW, and 'always high' was one sign pattern on one world"
    Reusing the `dF/dq` stored at `θ*` saves one backward pass and misses by `∂²F/∂θ∂q · Δθ · Δq` —
    second order, the **same order as the residual it estimates**. Scanned across tick *shapes* rather
    than sizes, on two worlds, the ratio runs **0.886× to 21.9×** and under-estimates on 8 of 36 rows: a
    parallel move sits in the Jacobian's dominant direction, while a sparse or sign-mixed one excites the
    small singular values, where a frozen `dF/dq` stops describing the residual.

    | tick shape (GBP world, 0.5%) | proxy | truth | proxy/truth |
    | --- | --- | --- | --- |
    | mixed sparse | 2.1933e-2 | 2.4639e-2 | **0.890** |
    | sign-mixed | 7.2094e-3 | 7.7571e-3 | **0.929** |
    | single quote | 3.1709e-4 | 3.1736e-4 | 0.999 |
    | parallel | 1.3761e-2 | 6.3949e-4 | 21.5 |

    An end-to-end case at 0.886 priced **5.8% over the declared tolerance without refusing**, admitting
    a ride 2.59bp of zero rate from the refit. The honest number costs 10.9ms against a 441ms refit.

**What the number is worth in curve units.** `Drift_Tolerance` is declared in percent of quote and felt
in basis points of zero rate, and the conversion is the Jacobian's own induced max-row-sum norm:
$\|\theta_{\text{ridden}} - \theta_{\text{refit}}\|_\infty \le \|J\|_\infty\,\|r\|_\infty$ to first
order. That is published in the refusal message and in the drift log (`at most X bp of zero rate on this
set, ‖J‖∞ Y`), and it is the gate: over every tick shape and size, the measured θ drift is inside
`‖J‖∞ ×` the quote residual.

Past `Drift_Tolerance` the ride raises `utils.CalibrationStale`, the house's typed refusal, named so a
caller can **refit** rather than lose the run — nothing about the job is wrong and no number has been
reported. The tolerance is the coupled set's strictest, so a set rides or refuses whole.

**At the refit**, when the slot is occupied, both coordinates are legal for this family and both are
computed, logged at INFO and published ON the replacement — and this is where the residual is EXACT,
because `quotes == q₀` on the fresh artifact kills the correction term:

- `θ_refit − θ_ridden` in the max norm — legal here because the solve is a unique root;
- the ridden θ's benchmark residual, in quote space, beside the block's own solver `Tol`.

So the record of how stale the last calibration got travels with the calibration that replaced it.

## What the ride is worth {#what-it-is-worth}

Measured on the ZAR round-trip world — seven quotes moved with **alternating signs** so nothing
cancels. The drift is pure curvature, and the gate pins the CONSTANT rather than asserting smallness: a
first-order bug still looks small at a basis point.

| tick (percent) | `‖θ_refit − θ_ridden‖∞` | `/tick²` | quote-space drift | `/tick²` |
| --- | --- | --- | --- | --- |
| 1e-4 | 8.4670e-12 | 8.4670e-4 | 7.5141e-10 | 0.075141 |
| 1e-3 | 8.4665e-10 | 8.4665e-4 | 7.5142e-8 | 0.075142 |
| 1e-2 | 8.4611e-8 | 8.4611e-4 | 7.5157e-6 | 0.075157 |
| 0.1 | 8.4074e-6 | 8.4074e-4 | 7.5308e-4 | 0.075308 |
| 0.25 | 5.1999e-5 | 8.3198e-4 | 4.7225e-3 | 0.075559 |
| 0.5 | 2.0446e-4 | 8.1783e-4 | 1.8994e-2 | 0.075977 |
| 1.0 | 7.9114e-4 | 7.9114e-4 | 7.6808e-2 | 0.076808 |
| 2.0 | 2.9737e-3 | 7.4342e-4 | 3.1378e-1 | 0.078445 |

Both constants hold to better than **one percent over four decades** of tick — flat, but not flat to
three digits, and the difference is the point: their third-order corrections point OPPOSITE ways, the θ
constant falling as the tick grows while the quote-space one rises, which is why both are pinned rather
than one standing in for the other. `Drift_Tolerance` defaults to **1e-3** — a tenth of a basis point of
mispricing — and it bounds the SQUARE of the tick, so on this world it admits **11.5bp** (drift 9.96e-4)
and refuses **12bp** (1.08e-3). `‖J‖∞` is 1.693e-2 here, so that tolerance is at most **0.17bp** of zero
rate.

The refused move is not a blow-up, which is why the threshold has to be declared rather than left to
something raising: at 45bp the ride is a perfectly plausible curve **1.66bp of zero rate** from the
refit, and it prices without complaint.

Tier one against tier two, on a two-swap book worth 20526.52. The agreement is first-order, so what is
pinned is the SLOPE of the miss — `ratio − 1` is **0.791 × tick**, flat to three digits, which a monitor
wrong by a constant could not produce.

| tick | `dV/dq · Δq` | ridden reval | ratio |
| --- | --- | --- | --- |
| 0.05 | −13.834407 | −14.381701 | 1.039560 |
| 0.02 | −5.5337629 | −5.6213125 | 1.015821 |
| 0.01 | −2.7668815 | −2.7887674 | 1.007910 |
| 0.005 | −1.3834407 | −1.3889120 | 1.003955 |

## Multi-curve: the coupled SET is the unit {#multi-curve}

`USD-3M` is solved discounting on `USD-OIS`, so $F_2(\theta_2;\,q_2,\,\theta_1(q_1)) = 0$ and
$d\theta_2/dq_1 \neq 0$ — for which a **per-block** `J₂` has no column. That is not an approximation but
a **first-order term dropped**, and dropped invisibly: a per-block `mispricing` prices a benchmark set
whose discount curve is a constant frozen at the fit, so it reads machine zero however stale the
projection curve gets. Measured on the USD world with the OIS strip ridden and the projection curve left
where the last bootstrap put it, at a 10bp OIS tick: book PV **9829.62** against a base of 9644.61 and a
full refit of 9621.25. The true move is **−23.36** and the partly-ridden set reports **+185.01** — wrong
sign, 8.9× the size, with the OIS block's own drift metric at 4.5e-4 and admitted healthy. The
unreported error is linear in the tick (2090.8 / 2087.6 / 2083.7 per percent over four sizes), so no
tolerance catches it.

**The fix is that the set is the unit.** `solve_for` was always a list and `split_theta`,
`residual_jacobians` and `calibration_jacobian` were always written over it; only `bootstrap` hardcoded
one curve. `solve_set` flattens a coupled set into ONE Newton system — `damped_newton`'s own documented
shape — so `calibration_jacobian` inverts one block matrix and `∂θ₂/∂q₁` is a column of the published
`J`. One artifact covers the set, with one `θ*`, one `q₀` and one drift number.

| | before | after |
| --- | --- | --- |
| artifact | per block, `J₂` is 9×9 | per set, `J` is 17×17 |
| `USD-3M` at a 10bp OIS tick | **0.177bp stale**, drift 2e-15 | **9.3e-5 bp** from the refit |
| book PV | 9829.62 against a refit of 9621.25 | 9621.25**2** against 9621.250**1** |
| order of the residual error | first (`/tick` ≈ 2085) | second (`/tick²` ≈ 0.20) |

**A partial ride is unrepresentable.** Declaring `Quote_Propagation` on some members of a set and not
others raises at the bootstrap, naming both halves — there is one operator over the set, so there is
nothing for a declining member to mean. The tolerance is likewise the set's strictest.

### The coupling is MEASURED, not declared {#measured-coupling}

`Discount_Rate` is the wrong question: what a benchmark **projects** off is authored inside its own deal
block, so a strip declaring a blank `Discount_Rate` — self-discounting, by the declaration — can still
forecast off a neighbour's curve. On a world built exactly that way, a 10bp tick on the projection strip
moved the "independent" curve by **568 basis points**, and a guard reading the declaration passed it
through: both blocks published per-block operators, the ride moved the coupled curve by exactly zero,
and its drift metric read 7.7e-16.

So `BenchmarkInstruments.reads` answers by differentiation instead — `_carry_quotes`'s idiom applied to
the other side of the residual. Every constant goes on the tape, the residual is evaluated once, and one
backward pass says which curves it actually reached; `coupled_sets` takes the connected components of
that relation. Cost is one compile and one backward per block, paid only when `Quote_Propagation`
appears somewhere in the section. Ridden as a measured set, the same world carries 571bp of the 568bp
move, leaving 3.4bp of second-order drift where it used to leave the whole 568.

## Value space, and the families that come next {#value-space}

θ-space scoring is legal HERE and nowhere else yet. A curve solve has a unique root, so
`θ_refit − θ_ridden` is a displacement of the answer. An HW2F swaption calibration does not: the refit
itself wanders in directions the pseudo-inverse declines, and [the re-solve reference is
refuted](quote_sensitivities.md#the-manifold-finding) on that fixture, so a θ-space drift gate there is
a false-alarm generator.

**The rule for every family after this one is score in value or residual space**: does the propagated
calibration reprice the benchmarks and the book the way the refit does? That is the question a desk is
asking anyway, and it is what [the direction
check](quote_sensitivities.md#the-identified-fixture) already had to fall back on. Branch crossings — a
declining-variance repair, a regime shift — are the expected drift spikes, and the tolerance is what
catches them.

## Where this sits in the protocol {#protocol}

The fast/slow split IS the values-vs-plan classification the service already has:
[`/prepare`](../api_overview.md) parses and names a plan, `/execute` takes a values `Patch` and hashes
what actually runs. A calibration artifact is the *market-data* half of the same idea — a slow object
named by plan-side coordinates with a fast path over it. The ride reads `Market Prices` at EXECUTE and
writes nothing, so it composes with `patch_market`: a patched spot and a moved quote in one job is a
reval with no re-bootstrap in it.

!!! note "A quote IS a patchable value — the split is `schema.MARKET_QUOTE_VALUES`"
    Per `Points` row, `Quoted_Market_Value`, the two-way `Quoted_Bid`/`Quoted_Ask` and the `Timestamp`
    are **values**; every other key of the row, and everything else on the block — pillars, expiries,
    conventions, `Use`, `Weight`, `Quote_Type`, the `Deal` a quote is a price for, the solver knobs, the
    lifecycle switches — is **structure**. So a vol tick moves `values_hash` and leaves `plan_hash`
    bit-identical, which is what makes the two staleness dimensions disjoint, and a moved pillar is a
    plan of its own.

    It is ONE rule for **every** family: a boundary that depended on which block you were looking at
    would not be a boundary. Five of the seven families do not quote in `Points` at all —
    `HullWhite2FactorModelPrices` in `Instrument_Definitions`, the two Heston-Nandi families and
    `CSForwardPriceModelPrices` in their own option tables, `GBMAssetPriceTSModelPrices` with no quote
    table whatever — so their values half is **empty** and a tick on one of them is still a new plan.
    `tests/test_market_prices_partition.py` states it per family by name.

    `config.update_market_quote`'s tick guard, `Config.plan_hash`, `market_patch`/`patch_market` and
    `InterestRateCurveParameters.plan_key` all read that one tuple. The projection **drops** the four
    keys where [`partition_factor`](market_prices.md) shadows a value to `None`, because a pillar that
    starts or stops being quoted two-sided is the same node of the same plan — so `Quoted_Bid`
    key-presence is itself value-plane.

    A quote patch does **not** re-bootstrap: the written factors stand and `values_hash` records the
    board standing now. The consumer that turns one into the other is the ride, so `patch_market` and
    the ride compose into one EXECUTE with no solve in it.

## Non-goals {#non-goals}

- **No second-order ride.** The operator is linear and its error is measured, not corrected; a
  curvature term would need `create_graph` through `CalibrationSolve`, which does not support it and
  does not detect the attempt — only `LeastSquaresSolve` [refuses it by
  name](quote_sensitivities.md#non-goals). The tolerance is the answer to a tick too big.
- **No automatic refit.** A refusal names the set and stops, whether the tick was too big or no artifact
  answered at all; scheduling the re-bootstrap is the caller's, because "when to recalibrate" is a desk
  policy and this page's contribution is the *measurement* that policy needs.
- **No persistence.** An artifact holds a compiled deal tree, and a serialised one would be a second
  definition of the plan. The consequence is [a refusal on the first tick after a restart](#stateless).
- **No cross-currency set.** A coupled set spanning currencies raises rather than joint-solving:
  `BenchmarkInstruments` has one reporting currency, and reaching past it is a change to the benchmark
  compiler rather than to this lifecycle.
- **No ride off an amount-valued quote.** An `FXForwardDeal` outright lands in `Buy_Amount` rather than
  a schedule column, so the operator's quote side reaches nothing and the block refuses by name —
  `Quote_Sensitivity` and `Quote_Propagation` both stay at `No` — rather than publishing a `J` with a
  silent zero `dF/dq` row (`test_a_forward_block_cannot_publish_a_ride_operator`, which also shows the
  cross-currency refusal firing first when both blocks solve together).
- **Not generalised past curves.** The artifact shape does not assume a square Jacobian, and no other
  family publishes one yet.
