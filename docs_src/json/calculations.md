# Calculations

The `Calc.Calculation` block names what to run and how. `Object` selects the type and
`Context.run_job` runs that class's `run`; any other `Object` is refused by name. Every field of
every type, with its type, default and menu, is on the type's own page under this section.

| `Object` | Runner | In-process method | What it reports |
|---|---|---|---|
| `BaseValuation` | `run_baseval` | `cx.Base_Valuation` | each deal's mark at the valuation date, and AAD factor sensitivities (`Greeks`) |
| `CreditMonteCarlo` | `run_cmc` | `cx.Credit_Monte_Carlo` | the simulated portfolio mark, exposure profile, CVA and FVA (CollVA and an initial margin experimental) |
| `HedgeMonteCarlo` | `run_hedgemontecarlo` | `cx.Hedge_Monte_Carlo` | the dynamic-hedging solve or simulation of [Hedging](../hedging/overview.md) |
| `SIMM` | `run_simm` | none: `cx.run_job` or `derivus.run_simm(cx.current_cfg)` | an ISDA SIMM sensitivity file (CRIF) [@isdasimm] by bump and revaluation, the book compiled once and priced again per market move, or under `Method: Graph` each row the move's derivative off one backward pass per trade, and each netting set's margin off it |
| `SACCR` | `run_saccr` | none: `cx.run_job` or `derivus.run_saccr(cx.current_cfg)` | the SA-CCR exposure at default of each netting set [@bcbscre52], off one base valuation of the book and each trade's declared terms |

A runner and a method are each one line onto the type's `run`.

## How a block is completed {#defaults}

Before a run, the block is assembled in this order, each step overwriting the last:

1. **Runner keys.** `BaseValuation`, `CreditMonteCarlo` and `HedgeMonteCarlo` add `Run_Date`, the
   document's `Base_Date` as `YYYY-MM-DD`. `CreditMonteCarlo` and `HedgeMonteCarlo` also add
   `Time_grid`, the document's `Time_Grid` (or its default) as text. `SIMM` and `SACCR` add nothing.
2. **The document's own keys.**
3. **`overrides`**, merged deeply: a nested mapping is merged key by key, anything else replaces.
4. **Declared defaults.** The calculation fills every top-level key still missing from its
   declaration (`schema.declared_defaults`).
5. **Judged.** The completed block is held to its declarations before anything runs
   (`schema.validate_block`): a required field still missing, or a value that is not what its
   field declares, refuses the run with every message at once. `cx.validate()` reports the same
   messages under `calculation`.

What follows from that order:

| Rule | Why |
|---|---|
| `Base_Date` must be stated, as a date | the runners read it before any default applies; its declared default is blank, which `cx.validate()` names as not stated |
| `Currency` must be stated | it is the currency the run reports in, and no default names one: a block stating none is refused at step 5 |
| The valuation date of `BaseValuation`, `CreditMonteCarlo` and `HedgeMonteCarlo` is `Run_Date` | step 1 copies it from the document's `Base_Date`, so an override must name `Run_Date`: overriding `Base_Date` alone does not move the date. `SIMM` and `SACCR` read `Base_Date` themselves |
| `CreditMonteCarlo` reads its grid from `Time_grid`, `HedgeMonteCarlo` from `Time_Grid` | so an override of the grid names `Time_grid` for the first and `Time_Grid` for the second |
| `SIMM` refuses a blank `Base_Date`, `Mapping` or `Parameters`, a zero shift and a key the margin reads that the `Parameters` file lacks under the `Horizon`, naming each | checked before anything is valued |
| `SACCR` refuses a blank `Base_Date`, `Mapping` or `Parameters` and a parameters file missing or malforming an entry CRE52 reads, naming each | checked before anything is valued |

## Containers {#containers}

A container the block states (`Credit_Valuation_Adjustment`, `Funding_Valuation_Adjustment`,
`Collateral_Valuation_Adjustment`, `Initial_Margin`, `Hedging_Problem`) replaces its declared
default whole. Step 4 does not reach inside it, so an omitted sub-field reads as whatever the
consuming code falls back to. For the `CreditMonteCarlo` containers that is:

| Sub-fields | Read when the container omits them |
|---|---|
| every `Calculate`, `Gradient` and `Hessian`; `CDS_Tenors` | the declared default |
| `Funding_Valuation_Adjustment.Deflate_Stochastically` | **`No`**, where the declaration says `Yes` |
| `Credit_Valuation_Adjustment`: `Counterparty`, `Deflate_Stochastically`, `Stochastic_Hazard_Rates` | no fallback: a stated block with `Calculate: Yes` must carry them |
| `Funding_Valuation_Adjustment`: the three curves | no fallback, as above |
| `Initial_Margin`: every field but `Calculate` | no fallback, read under `Calculate: Yes` |
| `Bank` (CVA and FVA), `Stochastic_Funding` (FVA) | read by nothing |

`Hessian: Yes` without `Gradient: Yes` in the same CVA block is refused.

## Monte Carlo set-up {#monte-carlo}

For `CreditMonteCarlo` and `HedgeMonteCarlo`:

| Item | What the code does |
|---|---|
| Outer paths | `Batch_Size × Simulation_Batches`. Sharded over `n` workers, each runs `Simulation_Batches // n` batches, so a remainder is not run |
| Outer normals | `torch.randn`, correlated through the Cholesky factor of the matrix [Correlations](correlations.md) assembles. The generator is seeded once per worker with `Random_Seed + worker index`; a sharded run (`runparallel`) reseeds every batch from a SplitMix64 mix of `Random_Seed` and the batch's global index (`calculation.batch_seed`), so its batches do not depend on the worker count |
| `Antithetic: Yes` | draws `Batch_Size // 2` correlated normals and appends their negatives. The path count is unchanged: the pairs are the two halves of the batch axis [@glasserman2003] |
| Quasi-random stream | a scrambled Sobol sequence [@sobol1967; @owen1998; @joe2008] (`torch.quasirandom.SobolEngine`, its scramble seed `Quasi_Seed`, read from point 1024). Its consumers are the regime and mixer uniforms of `MarkovHMMSpotModel`, `GARCHSpotModel`, `BasisLinkedSpotModel` and `LogVar2FJImpliedSpotModel`, the inner normals of `HedgeMonteCarlo` under `Inner_Draws: sobol`, and the inner-path block of the simulated pricers at every outer batch size, which with a LogVar2FJ pricer's walk, keyed by `Quasi_Seed` and its segment, is all a `BaseValuation`, a `SIMM` or an `SACCR` draws. It is not derived from `Random_Seed` |
| Inner paths | `MCMC_Simulations` per simulated pricing: 2,048 by default here, 16,384 under `BaseValuation`, `SIMM` and `SACCR` |
| Precision | float32 by default (`CreditMonteCarlo`, `HedgeMonteCarlo`); `BaseValuation` defaults to float64 |
| Model time | a scenario date's year fraction is its calendar days from `Base_Date` / 365.25 |

### Grids {#grids}

`Time_Grid` is a text grid (grammar in [General Types](general_types.md#time-grids)). Each item
dates `Base_Date + period`; an item `period(step)` repeats every `step` up to the next item's date,
the last item up to the book's last date. Dates are calendar dates, not business-day adjusted.

| Grid | Dates |
|---|---|
| Report (mtm) dates | the grid's dates and every deal reset and settlement date after `Base_Date`. A grid ending in a bare period stops the run at the earlier of the book's last date and `Base_Date` + that period, and drops later deal dates |
| Scenario dates | the report dates when `Dynamic_Scenario_Dates` is `Yes` (the default, forced by `Calc_Scenarios: At_Percentile` and always so under `HedgeMonteCarlo`); otherwise the grid's dates to the book's last date, plus the first one past it |

A book with no date after `Base_Date` is refused: the grid is grown from the dates the deals reach.

### Exposure profile {#exposure-profile}

`exposure_profile` summarises the simulated portfolio mark $V(t)$ over paths at each report date:
`EE` $=\mathbb{E}[\max(V,0)]$, `ENE` $=\mathbb{E}[\min(V,0)]$, and one `PFE_<p>` per entry of
`Percentile` (comma-separated, default `95`), the $p$-th percentile of $V$ itself, not of
$\max(V,0)$, by NumPy's default linear interpolation.
