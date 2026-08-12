# Quote Propagation

!!! note "Built for the curve family only"
    `InterestRatePrices` is the family this lands on, and deliberately: its fixed point is a unique
    root, its Jacobian is square by [the knot rule](quote_sensitivities.md#the-knot-rule), and the
    implicit function theorem is therefore **exact** rather than an approximation of a manifold. The
    vol families are not painted out — nothing in the artifact assumes squareness — but their drift
    has to be scored in value space and that is [a different contract](#value-space).

    The unit is the **coupled set**, not the block: blocks whose residuals read each other's curves
    solve as one system and ride as one operator, so `∂θ₂/∂q₁` is a column of the published `J`.
    Every block of such a set must declare `Quote_Propagation` or none may — [measured, and the
    partial ride is what it is protecting against](#multi-curve).

[Quote sensitivities](quote_sensitivities.md) built one arrow: `q -> calibration -> θ*`, and
differentiated it. That derivative is a **reporting** object there — `dV/dq` in one backward pass.
This page is about the same matrix used as an **operator**.

Between two calibrations, a quote that moves carries the curve with it to first order:

$$\theta \;\approx\; \theta^* + \frac{d\theta}{dq}\,(q_{\text{now}} - q_0)$$

which is a matvec against a solve. So a tick does not have to re-bootstrap to reach a valuation, and
the drift between the ridden curve and a true refit is a **measurement of how stale the calibration
actually was** rather than a schedule someone guessed. That is the whole design: predict, ride,
correct.

| tier | what it is | cost |
| --- | --- | --- |
| **monitor** | `dV/dq · Δq` — P&L explain, forwards | a dot product, no reval |
| **ride** | `θ* + J·Δq` reaching the pricers | a matvec plus the reval |
| **correct** | a re-bootstrap, publishing what the ride it replaced was worth | the solve |

Measured on the ZAR strip: a refit is **594 ms**, a ride is **74.5 ms** — 8× — and the operator
itself is **0.02 ms**, which is 30,000×.

!!! note "Honest reading — the operator is free and the SAFETY CHECK is the whole cost"
    **97% of a ride is `mispricing`**, not `θ* + J·Δq`. The matvec is the part the design is named
    for and it is unmeasurable beside the rest; what a tick actually pays for is finding out whether
    the ride was allowed — one residual evaluation and one backward pass per benchmark, since the
    metric is [re-differentiated at the ridden θ](#drift) rather than read off a frozen `dF/dq`.

    That is the right trade and the margin says so: the honest number costs 71.7 ms against a
    594 ms refit, so buying it back would save 8× at the price of a metric that
    [under-reads by up to 11% on some tick shapes](#drift). But it means the next optimisation is
    not a faster operator — it is checking the drift on a schedule rather than at every leaf mint,
    and 8× is what the current shape is worth. Note also that a coupled set is scored once **per
    member curve** the valuation mints, so an N-curve set pays the check N times.

## The artifact {#the-artifact}

`bootstrappers.CalibrationArtifact` is one calibration of one **coupled set** frozen as an operator:
`(θ*, J, q₀, timestamp)`, its `members`, and the compiled `BenchmarkInstruments` the first two were
read off. `θ*` is the solved node vector in `solve_for` order, `J = dθ/dq` at that fixed point, `q₀`
the quote vector it was fitted at in percent.

`timestamp` is **reported, never read by the arithmetic**: every ride, every refusal and every refit
names the artifact it rode and when that artifact was fitted, because "how stale" is the question
the drift number answers. It reaches no number and no hash, so a wall clock cannot make two runs
disagree — which is the same rule the [default `Base_Date`](../api_overview.md) had to be held to.

**Where `J` comes from — the same differentiation, not a second solve.** Stated precisely, because
"extracted from the backward" would be wrong in a way that matters. `CalibrationSolve.backward`
builds both of the residual's derivatives at `(θ*, q)` and contracts ONE cotangent with them;
autograd's `ctx` is not reachable from outside that call, so nothing is lifted out of it. What is
shared is the FUNCTION: `residual_jacobians` takes `dF/dθ` and `dF/dq` out of the *same* backward
pass — the quote side is another output of the pass the θ side already needs — and
`calibration_jacobian` calls it once more at the fixed point the forward pass already found,
solving the same `n × n` system against every column of `dF/dq` at once instead of one cotangent.

So the cost is **one Newton iteration's worth** — one residual evaluation and `n` backward passes —
and there is **no second root find**: the solve is not re-entered, only the residual is
differentiated. Materialising the full `n × m` costs nothing over reading one row of it, and the
backward is one pass cheaper than it was, because it no longer takes a separate VJP for the quote
side. The residual stays *written once and differentiated twice*, and a published `J` cannot drift
from the `dV/dq` the same job reports.

Measured against two full re-bootstraps per quote on the ZAR round-trip world: **1.07e-12** absolute
on columns of norm 1.3e-2, which is the finite difference's own resolution.

## Content addressing, and the two identities {#content-addressing}

The artifact is a **plan-side** object with two names, and it needs both.

**The slot** — `plan_key` — is every member block of the coupled set, the base date, the
interpolation scheme and the engine version, with the quote NUMBERS shadowed to `None`. That is
exactly [`partition_factor`'s](market_prices.md) split applied to a `Market Prices` block: a value is
shadowed rather than dropped, so the key SET stays structural and adding a quote is a different
plan. Every tick of one strip therefore lands on the **same slot**, which is what makes a ride
possible at all — key it by the numbers and the artifact would move with the quote that moved.

It names the SET, so re-authoring a discount strip moves the slot of every curve solved against it —
an operator whose `J` was fitted against quotes that no longer exist is exactly the one that must not
be findable by a curve it still covers.

!!! warning "Everything the SOLVE reads has to be in the key, including what the block does not carry"
    `Base_Date` and `Price Factor Interpolation` are inputs to the solve and live in
    `System Parameters` and a `ModelParams`, not on the block — and a key missing one is a key two
    different curves share, with the second silently riding the first one's operator. Both were
    measured doing exactly that: two jobs 45 days apart shared a slot, and a linearly-interpolated
    job rode a Hermite solve **0.53bp** away from its own answer. Gated, in both directions — the
    same job under the same scheme has to *keep* its slot, or the fix is just "never find anything".

**The `artifact_id`** is the slot plus the quotes it was fitted at. It **moves with every refit**,
so a propagated valuation is labelled by the calibration it rode plus the tick it rode — a replay
coordinate rather than a timestamp anyone has to trust. Two identical refits produce the same id.

!!! warning "A lifecycle switch cannot be in the slot it governs"
    `Quote_Propagation`, `Drift_Tolerance` and `Quote_Sensitivity` are shadowed out of the key
    (`lifecycle_fields`). They are read when an artifact is published or ridden rather than when it
    is fitted, and none of them can move `θ*` or `J` — `Quote_Sensitivity`'s bit-identity gate says
    so. Leaving them in means loosening a tolerance silently means "refit" instead of "allow more",
    which is a knob hiding the artifact it governs.

**A slot is not a plan id, and the sharing is deliberate.** `plan_hash` names a whole job — deals,
calculation, price factors, market prices — while a slot names only the calibration inputs, so two
jobs with different `plan_hash`es (a different book, a different netting set, a different reporting
currency) legitimately ride **one** artifact. A calibration is a property of the market data and not
of the trades priced against it; keying it by the job would refit per book.

**Where it lives.** `bootstrappers.ARTIFACTS`, in process, bounded and least-recently-used — the
plan cache's discipline for the other half of a prepared job. It holds tensors and a compiled deal
tree, so `Price Factors` is not an option (that section is data, and gets written back out as JSON)
and neither is a file. LRU rather than FIFO is a correctness property and gated as one: a tick
stream rides one slot over and over while unrelated jobs publish around it, and FIFO throws that
slot out on schedule. `covering(factor)` returns **candidates** most-recently-used first and picks
none — the caller recomputes each one's slot against the market data standing now, because two
artifacts can cover one curve at once and a lookup returning the first would hide the second.

## Stateless {#stateless}

There is no `θ_current` anywhere — the claim is grep-able. `q_now` arrives as the **values patch**,
and θ is derived per EXECUTE as a pure function of `(artifact, q_now)`:

```
Config.propagated_factor(factor) -> InterestRateCurveParameters.propagate -> artifact.ride(q_now)
```

reading `Market Prices`, the base date, the interpolation scheme and a content-addressed slot, and
writing none of them. Two EXECUTEs over one `(artifact, q_now)` are bit-identical by construction
rather than by care; riding to A, then B, then back to A returns A's first answer to the last bit.
An artifact is **replaced, never edited** — a refit publishes a new one into the same slot.

What DOES have to survive a JSON round trip is the **key**, computed off the block the decoder
rebuilt — a `Market Prices` block carries Timestamps, DateOffsets, DateLists and Percents, and a
decoder that rebuilt any of them differently would miss the slot and refuse forever. Gated.

!!! danger "A MISS IS A REFUSAL, never a different number"
    `Quote_Propagation: 'Linear'` and no artifact answering to the plan raises
    `utils.CalibrationStale`. It does not fall back to the curve the last bootstrap wrote, and that
    is the house rule for a plan the cache cannot answer applied to the other half of a prepared
    job: **a plan miss is a 404**.

    The fallback was the shipped behaviour and it is the one thing the replay tuple cannot describe.
    An artifact evicted by unrelated jobs filling the store repriced a book by **13.4%** while
    `plan_hash`, `values_hash`, `__version__` and `Random_Seed` were all **identical** across the
    two runs — nothing in the replay identity is a function of which artifact was in the store, so
    there was no coordinate that could have told them apart. A refusal is not a number, so it cannot
    be mistaken for one.

    The other half of closing that hole is that a run which *does* ride **reports the
    `artifact_id`** it rode, in `calc_stats['Calibrations']` keyed by curve — so
    `out['Stats']['Calibrations']` is the missing replay coordinate, and a propagated valuation is
    reproducible rather than merely repeatable.

    **The cold start follows from it and is the honest half.** An artifact cannot be serialised, so
    a fresh process has none and the first tick after a restart refuses until the job is
    bootstrapped — which publishes, and the same EXECUTE then runs. That is not a gap to close: a
    re-bootstrap rederives the artifact from the job document, and the artifact was never the record
    of anything.

## The seam {#the-seam}

`Calculation.factor_leaf` — the same one the quote-sensitivity attachment uses, and for the same
reason: it is the ONE place a curve becomes a tensor, on both `_build_factor_state` branches and on
`Base_Revaluation.update_factors`. The two offers differ in what they change:

| | attachment (`Quote_Sensitivity`) | ride (`Quote_Propagation`) |
| --- | --- | --- |
| offers | `leaf + (θ* - θ*.detach())` | the ridden nodes as `current_val` |
| changes | what reaches `backward()` | **the value** |
| under `Tenor_Offset` | declines | **refuses** |

The ride is the one that moves a mark, which is the point. `Quote_Propagation` defaults to `No` and
off it is today's path bit for bit — same curve, same marks, and the store is never consulted.

**Both switches on is the coherent combination, not a collision.** The leaf's VALUE is the ridden
curve and the splice contributes zero to it, so what the backward reports is `dV/dθ` at the ridden
curve contracted with `dθ/dq` at `q₀` — the artifact's own `J`, which is the operator that put the
curve there. Value and derivative are then the same first-order object, which is what makes tier
one comparable with tier two at all.

!!! warning "A `Tenor_Offset` refuses the ride rather than declining it"
    `current_value(offset)` interpolates off Hermite coefficients fitted on the numpy rate column at
    construction, so a shifted curve cannot be ridden without refitting them. Declining would
    silently price the STALE curve — a wrong number, not a missing derivative — so it raises.

## The drift metric, and the refusal {#drift}

**At EXECUTE**, before anything is priced: `mispricing` is every benchmark's residual at the ridden
θ and the current quotes, divided by that benchmark's own quote sensitivity — a **quote-space**
number, the move in percent that would close it. It costs no refit, because a benchmark's PV is
AFFINE in its own quote (measured in `_carry_quotes`, second difference exactly zero), so

$$F(\theta, q) = F(\theta, q_0) + \frac{\partial F}{\partial q}(q - q_0)$$

needs no re-authoring of the compiled set. `dF/dq` is diagonal on this family, measured at **0.0**
off-diagonal — each benchmark is authored at its own quote — and the normaliser is a row max rather
than a diagonal so a family whose benchmarks are not one-quote-each stays expressible.

That identity is exact in `q` at **fixed θ** — *provided `dF/dq` is the one at the θ being scored*.
So `residual_jacobians` re-differentiates it **here**, at the ridden θ, one backward pass per
benchmark off the forward pass the residual itself comes out of. The number is then the drift, not
an estimate of it, and it is gated against a refit's own measurement rather than against a ratio.

!!! danger "The earlier proxy read LOW, and 'always high' was one sign pattern on one world"
    Reusing the `dF/dq` stored at `θ*` is cheaper by one backward pass and its miss is
    `∂²F/∂θ∂q · Δθ · Δq` — second order, the **same order as the residual it estimates**. That was
    shipped as a conservative bound pinned at "1.084 to 1.096× the truth, always high". Scanned
    across tick *shapes* rather than sizes, on two worlds, it ran from **0.886× to 21.9×** — a
    spread of 25 — and under-estimated on 8 of 36 rows. A parallel move sits almost entirely in the
    Jacobian's dominant direction; a sparse or sign-mixed one excites the small singular values,
    which is exactly where a frozen `dF/dq` stops describing the residual.

    | tick shape (GBP world, 0.5%) | proxy | truth | proxy/truth |
    | --- | --- | --- | --- |
    | mixed sparse | 2.1933e-2 | 2.4639e-2 | **0.890** |
    | sign-mixed | 7.2094e-3 | 7.7571e-3 | **0.929** |
    | single quote | 3.1709e-4 | 3.1736e-4 | 0.999 |
    | parallel | 1.3761e-2 | 6.3949e-4 | 21.5 |

    An end-to-end case at 0.886 priced **5.8% over the declared tolerance without refusing** — the
    admitted ride was 2.59bp of zero rate from the refit. The cost of the honest number is 10.9ms
    against a 441ms refit, so the trade was never the one the note claimed.

**What the number is worth in curve units.** `Drift_Tolerance` is declared in percent of quote and
felt in basis points of zero rate, and the conversion is the Jacobian's own norm:

$$\|\theta_{\text{ridden}} - \theta_{\text{refit}}\|_\infty \;\le\; \|J\|_\infty\,\|r\|_\infty$$

to first order — `‖J‖∞` being the induced max-row-sum norm. That is published in the refusal message
and in the drift log (`at most X bp of zero rate on this set, ‖J‖∞ Y`), and it is the gate: over
every tick shape and size, the measured θ drift is inside `‖J‖∞ ×` the quote residual.

Past `Drift_Tolerance` the ride raises `utils.CalibrationStale`, the house's typed refusal
(`SecondOrderRefused` is the precedent): named so a caller can **refit** rather than lose the run,
because nothing about the job is wrong and no number has been reported. The tolerance is the
coupled set's strictest, so a set rides or refuses whole.

**At the refit**, when the slot is occupied, both coordinates are legal for this family and both are
computed, logged at INFO and published ON the replacement — and this is where the residual is
EXACT, because `quotes == q₀` on the fresh artifact kills the correction term:

- `θ_refit − θ_ridden` in the max norm — legal here because the solve is a unique root;
- the ridden θ's benchmark residual, in quote space, beside the block's own solver `Tol`.

So the record of how stale the last calibration got travels with the calibration that replaced it.

## What the ride is worth {#what-it-is-worth}

Measured on the ZAR round-trip world — seven quotes, moved with **alternating signs** so nothing
cancels. The drift is pure curvature, and the gate pins the CONSTANT rather than asserting
smallness: a first-order bug still looks small at a basis point.

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

Both columns are exact measurements now, and both constants hold to better than **one percent over
four decades** of tick — flat, but not flat to three digits, and the difference is the point: their
third-order corrections point OPPOSITE ways, the θ constant falling as the tick grows while the
quote-space one rises. That is why both are pinned rather than one standing in for the other.
`Drift_Tolerance` defaults to **1e-3** — a tenth of a basis point of mispricing — and it is a bound
on the SQUARE of the tick, so on this world it admits **11.5bp** (drift 9.96e-4) and refuses
**12bp** (1.08e-3). `‖J‖∞` is 1.693e-2 here, so that tolerance is at most **0.17bp** of zero rate.

The refused move is not a blow-up, which is why the threshold has to be declared rather than left
to something raising: at 45bp the ride is a perfectly plausible curve **1.66bp of zero rate** away
from the refit, and it prices without complaint.

Tier one against tier two, on a two-swap book worth 20526.52: `dV/dq · Δq` against the value the
ride reprices. The agreement is first-order, so what is worth pinning is the SLOPE of the miss —
`ratio − 1` is **0.791 × tick**, flat to three digits, which a monitor wrong by a constant could
not produce.

| tick | `dV/dq · Δq` | ridden reval | ratio |
| --- | --- | --- | --- |
| 0.05 | −13.834407 | −14.381701 | 1.039560 |
| 0.02 | −5.5337629 | −5.6213125 | 1.015821 |
| 0.01 | −2.7668815 | −2.7887674 | 1.007910 |
| 0.005 | −1.3834407 | −1.3889120 | 1.003955 |

## Multi-curve: the coupled SET is the unit {#multi-curve}

`USD-3M` is solved discounting on `USD-OIS`, so

$$F_2(\theta_2;\,q_2,\,\theta_1(q_1)) = 0
\qquad\Longrightarrow\qquad
\frac{d\theta_2}{dq_1} = -\Big(\frac{\partial F_2}{\partial\theta_2}\Big)^{-1}
\frac{\partial F_2}{\partial\theta_1}\frac{d\theta_1}{dq_1} \neq 0$$

and a **per-block** `J₂` has no column for it. That is not an approximation, it is a **first-order
term dropped** — and worse, dropped invisibly, because a per-block `mispricing` prices a benchmark
set whose discount curve is a constant frozen at the fit and therefore reads machine zero however
stale the projection curve gets. Measured on the USD world with the OIS strip ridden and the
projection curve left where the last bootstrap put it:

| | base | ridden (per block) | full refit |
| --- | --- | --- | --- |
| book PV at a 10bp OIS tick | 9644.61 | **9829.62** | 9621.25 |

The true move is **−23.36** and the partly-ridden set reports **+185.01** — wrong sign, 8.9× the
size, with the OIS block's own drift metric at 4.5e-4 and admitted healthy. The unreported error is
linear in the tick (2090.8 / 2087.6 / 2083.7 per percent over four sizes), so no tolerance catches
it and shrinking the tick does not help.

**The fix is that the set is the unit.** `solve_for` was always a list and `split_theta`,
`residual_jacobians` and `calibration_jacobian` were always written over it; only `bootstrap`
hardcoded one curve. `solve_set` flattens a coupled set into ONE Newton system — which is
`damped_newton`'s own documented shape — so `calibration_jacobian` inverts one block matrix and
`∂θ₂/∂q₁` is a column of the published `J` rather than a term nobody carried. One artifact covers
the set, with one `θ*`, one `q₀` and one drift number.

| | before | after |
| --- | --- | --- |
| artifact | per block, `J₂` is 9×9 | per set, `J` is 17×17 |
| `USD-3M` at a 10bp OIS tick | **0.177bp stale**, drift 2e-15 | **9.3e-5 bp** from the refit |
| book PV | 9829.62 against a refit of 9621.25 | 9621.25**2** against 9621.250**1** |
| order of the residual error | first (`/tick` ≈ 2085) | second (`/tick²` ≈ 0.20) |

**A partial ride is now unrepresentable.** Declaring `Quote_Propagation` on some members of a set
and not others raises at the bootstrap, naming both halves — there is one operator over the set, so
there is nothing for a declining member to mean. The tolerance is likewise the set's strictest, so a
set rides or refuses whole.

### The coupling is MEASURED, not declared {#measured-coupling}

`Discount_Rate` is the wrong question. What a benchmark **projects** off is authored inside its own
deal block, so a strip declaring a blank `Discount_Rate` — self-discounting, by the declaration —
can still forecast off a neighbour's curve. On a world built exactly that way, a 10bp tick on the
projection strip moved the "independent" curve by **568 basis points**, and a guard reading the
declaration passed it straight through: both blocks published per-block operators, the ride moved
the coupled curve by **exactly zero**, and its drift metric read 7.7e-16.

So `BenchmarkInstruments.reads` answers by differentiation instead — `_carry_quotes`'s idiom applied
to the other side of the residual. Every constant goes on the tape, the residual is evaluated once,
and one backward pass says which curves it actually reached. `coupled_sets` takes the connected
components of that relation. Cost is one compile and one backward per block, paid only when
`Quote_Propagation` appears somewhere in the section; without it every block is its own group solved
in dependency order, which is what a bootstrap has always done.

Ridden as a measured set, the same world carries 571bp of the 568bp move, leaving 3.4bp of
second-order drift where it used to leave the whole 568.

## Value space, and the families that come next {#value-space}

θ-space scoring is legal HERE and nowhere else yet. A curve solve has a unique root, so
`θ_refit − θ_ridden` is a displacement of the answer. An HW2F swaption calibration does not: the
refit itself wanders in directions the pseudo-inverse declines, and
[the re-solve reference is refuted three times](quote_sensitivities.md#the-manifold-finding) on that
fixture. A θ-space drift gate there is a false-alarm generator.

The rule for every family after this one is therefore **score in value or residual space**: does the
propagated calibration reprice the benchmarks and the book the way the refit does? That is the
question a desk is asking anyway, and it is the one
[the direction check](quote_sensitivities.md#the-identified-fixture) already had to fall back on
when the ladder was unavailable. Branch crossings — a Malz declining-variance repair, a regime shift
— are the expected drift spikes, and the tolerance is what catches them.

## Where this sits in the protocol {#protocol}

The fast/slow split IS the values-vs-plan classification the service already has:
[`/prepare`](../api_overview.md) parses and names a plan, `/execute` takes a values `Patch` and
hashes what actually runs. A calibration artifact is the *market-data* half of the same idea — a
slow object named by plan-side coordinates, with a fast path over it. The ride reads `Market Prices`
at EXECUTE and writes nothing, so it composes with `patch_market`: a patched spot and a moved quote
in one job is a reval with no re-bootstrap in it.

!!! warning "A quote is not a patchable VALUE yet, and that is a gap with a name"
    `market_patch` covers `Price Factors` only, and the whole `Market Prices` section is inside
    `plan_hash`. So a moved quote today changes the **plan id** rather than the **values hash** —
    even though the engine now carries it without recompiling anything, which is the exact
    condition `bind='value'` describes. Closing it means partitioning a `Market Prices` block the
    way `partition_factor` partitions a factor, and it is a change to the split for **every**
    family rather than this one; doing it per-family would make the plan/values boundary depend on
    which block you are looking at. The current contract is pinned by a gate rather than left
    implicit, so the day quotes move to the values half nothing goes quiet.

## Non-goals {#non-goals}

**No second-order ride.** The operator is linear and its error is measured, not corrected; a
curvature term would need `create_graph` through `CalibrationSolve`, which
[is refused](quote_sensitivities.md#non-goals). The tolerance is the answer to a tick too big.

**No automatic refit.** A refusal names the set and stops — whether the tick was too big or no
artifact answered at all; scheduling the re-bootstrap is the caller's, because "when to recalibrate"
is a desk policy and this page's contribution is the *measurement* that policy needs.

**No persistence.** See [the miss rule](#stateless) — an artifact holds a compiled deal tree, and a
serialised one would be a second definition of the plan. The consequence is a refusal on the first
tick after a restart, which is the correct one: a plan the store cannot answer is a 404.

**No cross-currency set.** A coupled set spanning currencies raises rather than joint-solving:
`BenchmarkInstruments` has one reporting currency, and reaching past it is a change to the benchmark
compiler rather than to this lifecycle.

**Not generalised past curves.** The artifact shape does not assume a square Jacobian, and no other
family publishes one yet.
