# Quote Sensitivities

!!! note "Increment 1, in progress"
    This page is written as the work lands. The **t0 benchmark closure** and its **graph audit**
    are built; the **plain multi-curve solver** and the `InterestRatePrices` family are next; the
    **implicit-function-theorem contract**, the factor-buffer attachment and the validation
    triangle are not built and are not specified here yet.

The autograd tape starts at *calibrated* factors today, so a greek is reported in zero-curve-node
space. Desks explain P&L in **quote** space — par swap rates, FRA strips, OIS quotes. This
workstream extends the tape one layer upstream, by owning the bootstrap inside the library and
differentiating through its fixed point, so one backward pass yields `dV/dq` alongside `dV/dθ`.

The chain is

```
q (float64 leaves)  ->  bootstrap solve  ->  θ*  ->  factor buffers  ->  scenarios  ->  V
```

and the only new arithmetic is the middle arrow. Everything downstream of θ* is the engine that
already exists.

## The residual is priced by the engine's own pricers {#the-residual}

A benchmark is an ordinary deal. A quote **names an instrument type and carries a block of it** —
see [Market Prices](market_prices.md#a-quote) — so the residual is a vector of the library's own
pricing functions:

$$F_i(\theta, q_i) = PV_i(\theta, q_i) \big|_{t_0}$$

evaluated so that a fair instrument prices to zero. `bootstrappers.BenchmarkInstruments` is that
evaluation: it compiles a set of benchmark deal-tree nodes once — discovery, factor objects, tenor
payloads, each leaf's `Factor_dep` / `Time_dep` — and then prices them at t0 from a
`{Factor: tensor}` of curve nodes, returning one PV per benchmark as a `torch` vector whose graph
reaches those tensors.

A benchmark is a **node**, not a deal, so a container with `Children` is one benchmark: a deposit
and an FRA are single deals, a par swap is one `SwapInterestDeal`, and an OIS swap is a container
over an OIS-compounded floating leg and a fixed leg. The node's PV is the sum of its leaves', each
already converted to the reporting currency by its own `pv_*_leg`. There is no netting or
collateral rule to apply on top, which is what keeps this out of `DealStructure`.

The reporting currency **is** the curve's currency, which makes every `calc_fx_cross` the identity
and keeps an FX rate out of the residual. A curve solve is single-currency by construction.

## The graph audit {#the-graph-audit}

The factor-construction path severs autograd in four places. Finding them is the point of this
increment, not incidental to it — a severance here does not raise and does not change a value, it
silently reports a zero gradient.

| Where | What it does | How the closure avoids it |
| --- | --- | --- |
| `Calculation._build_factor_state`, `Base_Revaluation.update_factors` | Every leaf is minted as `torch.tensor(factor.current_value(offset=…), requires_grad=…)` — a fresh leaf built out of a **numpy array**. Anything upstream is severed by construction. This is the *only* way a curve becomes a tensor on the ordinary path. | θ is written straight into `t_Static_Buffer`, which is where the pricers read a static curve from. `current_value` is never called for a curve being solved. |
| `riskfactors.Factor1D.current_value` | numpy end to end — `param['Curve'].array[:, 1] + self.delta`, `np.interp`, `np.concatenate`. Handed a tensor it would still return numpy. | Not on the closure's path. It is still used for the **constants** — every factor the solve is not solving for — where a detached leaf is the right answer. |
| `Factor1D.__init__` → `check_interpolation` | Precomputes the Hermite `(g, c)` coefficient pair from the numpy rate column at construction. Those coefficients are constants in θ. | The pricing path does not read them. `utils.Interpolation.build` re-derives the pair from the buffer **tensor** (`hermite_interpolation_tensor`), and `all_tenors` carries only the interpolation *kind* and the tenor grid, both θ-independent. A Hermite curve therefore differentiates, and there is a gate per interpolation kind saying so. |
| `utils.TensorSchedule.merged` | Copies the whole cashflow schedule across with `new_tensor` — notionals, accruals, margins **and the fixed rate**. This is where the **quote** stops being differentiable. | θ does not pass through it, so the closure is unaffected. A quote-side derivative cannot come from autograd through this seam and has to be obtained another way; that is the increment's open question, recorded here rather than guessed at. |

Two traps that are not severances, and would each be silent:

- **`t_Buffer` is a memo table keyed by `(stoch, Factor)` and a time hash, not by the tensor's
  identity.** A pricing state reused across two different θ answers the second call with the first
  call's discount factors — a solver built on a reused state converges to whatever it started at.
  `Benchmark_State` is therefore built fresh per evaluation, and a gate holds the trap in place so
  that it stays a known property rather than a surprise.
- **`utils.CurveTenor` caches its tenor grid as a tensor built from the first tensor that queries
  it.** `all_tenors` is rebuilt per closure instance, so a float64 solve cannot inherit a float32
  grid from whatever ran before it.

## The precision seam {#the-precision-seam}

The bootstrap and its Jacobian are **float64 regardless of the simulation's precision**.
`BenchmarkInstruments.dtype` states it once. A solve that has to converge to 1e-10 cannot be done
in float32, and the Jacobian handed to the implicit function theorem is only as good as the
residual it came from. Where θ\* crosses back into the simulation it is cast to the cube's dtype;
that boundary is specified with the IFT contract, which is not built yet.

## Non-goals {#non-goals}

No SIMM aggregation or regulatory bucketing — bucketed quote deltas are the raw material and this
stops there. No wrong-way risk. **No recalibration inside the simulation**: quote sensitivities are
t0 risk, and future-dated dynamics stay on calibrated parameters. No new pricers, and no changes to
the `instruments.py` pricers beyond what the t0 closure strictly requires — so far, none.
