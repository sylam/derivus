# Calc Lifecycle

A calculation compiles a job into factor objects, AAD leaves and a deal tree; then, per batch, the
processes publish paths, the deal tree prices on them and the calculation reduces the result. This
page follows that walk in code order; entry points and output are on
[API Overview](../api_overview.md).

| where | what |
| --- | --- |
| `__init__.py` | `Context.run_job` and the `run_*` functions, each one line onto a calculation's `run` |
| `calculation.py` | the five calculations and how each runs, their states, `batch_seed`, `DealStructure` |
| `utils.py` | `Calculation_State`, `TimeGrid`, `TensorSchedule`, the curve reads, the `BoundarySet` family |
| `pricing.py` | `interpolate`, `oss_stream`, `InnerMCRecompute`, `LogVar2FJKit`, the boundary and kink terms |

**Held by** names the test a broken rule turns red; **Not gated**, that none does.

[A Job Through the Machine](../animations/job_through_the_machine.html) animates this page, from
the dispatch to the inner fork and its recompute node: play it here or open it full-page.

<iframe src="../../animations/job_through_the_machine.html" title="A Job Through the Machine" loading="lazy"
        style="width:100%;height:860px;border:1px solid var(--md-default-fg-color--lightest);border-radius:6px"></iframe>

## Dispatch

`Context.run_job` is `calculation_class(Calculation['Object']).run`: each calculation class owns
how it runs, and `cx.Base_Valuation`, `cx.Credit_Monte_Carlo`, `cx.Hedge_Monte_Carlo` and the
`run_*` functions are one line onto the same `run`. **Held by**
`tests/test_schema_emission.py::test_the_calculation_types_are_what_run_job_dispatches`. A run
reads the block `Calculation.parameters` completes: the keys its class's `runner_keys` derive
(`Run_Date`, and `Time_grid` for the Monte Carlo modes), the document's block over them and the
overrides merged deeply; every other default comes from the class's `fields` through
`declared_defaults`. `BaseValuation`, `ScenarioValuation`, `CreditMonteCarlo` and
`HedgeMonteCarlo` pick the device and take a precision, `CreditMonteCarlo.run` sharding under
`runparallel`. `SIMM` and `SACCR` inject nothing, `SACCR` running through the base
`Calculation.run`, their calculations sharing `NettingSetCalculation`'s setup and its `base`, which
compiles the book once as a `ScenarioValuation` whose scenarios its valuation states
(`update_factors`, `set_deal_structures`; a worker of `SIMM.run` compiles its own once, over its
own moves, `SIMM.shard`), each surface a vega move reaches given its vertex pillars so the base and
every move share one grid, and prices it on the standing market alone. Under `Method: Bump` the
moves are that valuation's scenarios (`SIMM.values`), a vega move's surface or space carried on
the scenario axis as a curve's is, priced `Batch_Size` at a time, or as many as half the device's
free memory holds at what a scenario took in the standing market's pass, each pass beside the
standing market, so a trade's change is the difference of two columns of one pass
(`ScenarioValuation.passes`). A trade a pass leaves unvalued is priced again a
move a pass, so the refusal names its own move. Every pass prices the whole set, the trades it does
not reach included. A trade's product class is read off its walk: the one class among Credit,
Equity and Commodity its factors are of, RatesFX where they are of none, every trade reading its
curves and currencies, and its own `Product_Class`, a field of the admin group, where they are of
more than one, refused by name where it states none (`SIMM.product_class`); the mapping states
the factors' coordinates alone. **Held by**
`tests/test_simm.py::test_a_trades_product_class_is_its_walks_or_its_own_statement`. A base
valuation draws a trade's Monte Carlo paths off `Quasi_Seed` alone,
whatever is booked or priced beside it, and the inner block stays built through the run, so a trade
draws the same paths through every pricing and its difference is one of common random numbers.
**Held by** `tests/test_simm.py::test_the_set_is_compiled_once_and_a_move_writes_the_leaves_it_moves`,
`::test_a_move_its_pricer_refuses_is_refused_by_name_never_read_off_the_last_pricing`,
`::test_a_netting_set_s_crif_is_its_trades_own_to_the_bit`,
`::test_a_monte_carlo_set_s_crif_is_its_trades_own_to_the_bit`,
`::test_the_sharded_run_equals_the_serial_run_to_the_bit`,
`::test_the_batched_run_sharded_equals_the_serial_run_to_the_bit`.

`base` compiles each trade alone under its set, so a trade whose own compile refuses its document
by name (`utils.UnpriceableSchedule`, which `DealStructure.add_deal_to_structure` counts under
`Deals Skipped` and stamps on the deal before it raises) is set aside: reported unvalued in that
sentence whatever `Exclude_Deals_With_Missing_Market_Data` says, writing no row, the book compiled
once without it and every other trade's rows what a book without it writes, to the bit. Nothing
marks it zero, and every other refusal stops the run - a pricing's, a move's, the mapping's, a
quote set's, a family's - as a compile's refusal by name stops every other calculation. **Held by**
`tests/test_simm.py::test_a_trade_the_compile_refuses_is_set_aside_by_name_and_the_run_goes_on`.
**Not gated:** `SACCR`'s set-aside, the same `base`'s.

A pass's arithmetic is its own: a column is the same to the bit whatever stands beside it, so the
CRIF is the same whatever the chunk and the worker count. A pass of one is the exception: it sums a
deal's terms with the scenario axis contiguous where a pass of several strides it, so a bump
pricing each move in a pass of its own reassociates against the batch, a row departing by at most
3.0e-12 of its trade's largest (a linker's rate rows, two ulps of its value), a rate vega row by
1.5e-13 (the amortising swaption's, fourteen ulps of its value over its hundredth of each vol); a
row being a change over its shift, a shift a ten-thousandth of ISDA's reads the same ulps ten
thousand times larger. A mean over 16,384 inner paths the device splits across blocks by the
pass's width, so on the Monte Carlo set a pass of five departs from a pass of two by 3.7e-15 of a
trade's largest row. **Held by**
`tests/test_simm.py::test_the_scenario_batch_writes_the_crif_the_row_at_a_time_bump_wrote`,
`::test_the_chunked_batch_writes_the_same_crif`.

Under `Method: Graph` nothing moves: the set is priced once on the tape, every curve, survival and
inflation solve kept there (`Quote_Sensitivity`), and a trade's rows are one backward pass of its
value along each move's derivative at a zero shift (`SIMM.tangent`), split as the bump splits it -
along the leaves the move writes, to first order where it is not linear; along the calibrated
curves a quote rides, whose tape chains the survival curves solved on them; and along the quotes of
a block the move re-solves, its family's [Jacobian](quote_sensitivities.md#the-ift-contract). A
sharded run deals whole trades, its parent valuing the base off no tape, which crosses no process.
**Held by** `tests/test_simm.py::test_the_graph_crif_is_the_limit_of_the_bump_crif`,
`::test_a_par_swaps_graph_pv01_is_its_annuity`,
`::test_method_omitted_is_bump_and_graph_is_one_value_away`,
`::test_graph_vertex_rows_sum_to_the_parallel_derivative`,
`::test_the_graph_run_sharded_equals_the_serial_run_to_the_bit`.

A price reads a factor's value off its leaf at pricing, where a move's write and the tape both see
it, never off the factor at the compile: an equity swaplet's level fixing on the base date and
stating no price is a sample the pricer reads off the spot (`utils.TensorCashFlows.equity_swaplet`),
a compo leg's pair there likewise off the fx rates (`EquitySwapLeg.calc_dependencies`), and an
amortising swaption's vol tenor and Lognormal level scale are `SwaptionDeal.duration_tenor`'s
tensors off the curves its pricer reads, a leg that does not amortise reading its stated tenor and
1.0. So the SIMM row of each is the central difference of two full compiles and `Greeks: First`
the same derivative, to 1e-6. The compile's level for an unprinted inflation month is the diary's,
the pricer projecting it off the index and curve leaves; a correlation is no leaf
([decision 12](../limitations.md#decisions-waiting-on-the-desk)) and a basis decay a process's
declared float. **Held by** `tests/test_simm.py::test_a_level_the_base_date_fixes_moves_with_the_spot`,
`::test_a_compo_level_the_base_date_fixes_moves_with_the_pair`,
`::test_an_amortising_swaptions_vol_moves_with_its_curves`.

A move writes what two full compiles would solve. A curve quote rides its coupled set off the
calibration the run's bootstrap published (`SIMM.requoted`), and every survival curve the book
reads that its family solves on a curve the ride moved is re-solved on the ridden curves
(`SIMM.resolved`), so a default swap's Risk_IRCurve rows carry its survival curve's re-solve as
the AAD's quote deltas do. A survival block reads the curve its default swaps discount on and no
other (`bootstrappers.block_readers`), so a default swap takes that curve's quotes and its own and
a curve move re-solves only the survival curves solved on it. A credit quote - Risk_CreditQ or
Risk_CreditNonQ, as the mapping states the survival curve - is one row of its `SurvivalProbPrices`
block a basis point of spread over and the block re-solved by its family, no ride published: a
`Par_Spread` row by the point, an `Upfront` row to what its contract quotes on the solved curve at
a coupon the point under its own.
The rows land on the methodology's credit vertices (`SIMM.CREDIT_VERTICES`) by the quote's `Tenor`,
the parameters file weighing credit by bucket alone, and the knots are the quotes' maturities,
which no spread moves. An upfront-quoted curve writes its par-spread twin's rows to 2.6e-3 of the
largest: the others hold their upfronts where spreads hold spreads, and the conversion is on the
standing curve, both first order in a spread's distance from the coupon. A survival curve a trade
reads that no configured family solves is refused by name, as an unquoted curve is. **Held by**
`tests/test_simm.py::test_a_default_swaps_credit_rows_are_its_survival_curve_re_solved_a_spread_point_apart`,
`::test_a_curve_quote_re_solves_the_survival_curve_a_default_swap_reads`,
`::test_a_default_swap_on_a_survival_curve_no_quotes_solve_is_refused_by_name`,
`::test_a_survival_curve_reads_the_curve_its_default_swaps_discount_on_and_no_other`.

A point of vega is a vertex's tent over a vol surface's expiry pillars, the surface given a pillar
at each vertex expiry it lacks off its own interpolation, a Skew or an SVI one only past its ends
(`Factor2D.augmented`); a commodity option's Risk_CommodityVol rows land so on the expiries its
samples read its `CommodityPriceVol` at. A vol space is written as the grid the engine reads, every
node of each of its three columns, with a slice at each vertex expiry it lacks read off that grid
linearly in expiry and flat past either end (`Factor3D.augmented`), so it prices as it stood; a
vertex moves every node of its slice a percent of itself, and a swaption's or a caplet's
Risk_IRVol rows land on its expiry, the underlying tenor summed over, each row the sum over the
slice of each vol times the vega to it - ISDA's vol-weighted vega, the amount its rate vega risk
reads as it stands (the methodology's paragraph 10(c)) - by one rule whichever convention the space
declares, the methodology admitting a normal vol for rates (paragraphs 10(a) and 30). An FX, equity
or commodity vertex moves its vols a point, the methodology weighing those rows by a vol it derives
from the delta weight. The grid is read by its columns, a `ForwardPriceVol`'s running delivery,
expiry, moneyness where a rate space's run moneyness, expiry, tenor; no trade reads a
`ForwardPriceVol`, the `CSForwardPriceModelParameters` family's fit alone does, so no walk names
one and no row of it is written. A factor mapped to a vega risk type that is no vol surface is
refused by name: a model's parameters move by their family's fit, which a move re-solves for
curves and credit alone.
**Held by** `tests/test_simm.py::test_a_rate_vega_row_is_the_vol_weighted_vega`,
`::test_a_rate_vol_space_given_its_vertex_slices_prices_as_it_stood`,
`::test_a_commodity_vega_row_is_a_point_on_its_vertex_pillar`,
`::test_a_point_of_vega_on_what_is_no_vol_surface_is_refused_by_name`. **Not gated:** a
`ForwardPriceVol`'s grid, which no trade reads.

A percent of a commodity is a percent of every value each price factor the mapping states under
its qualifier holds (`schema.partition_factor`), its spot (`CommodityPrice`) and every delivery of
its forward curve (`ForwardPrice`) at once, one Risk_Commodity row a trade and no vertex. A basis
(`ObservedBasis`) is part of its metal's price and is mapped under the metal's qualifier and
bucket, a cross-class type taking the class it is mapped to (`SIMM.coordinates`), so a future on
the platinum chain's CME level moves by a percent of that level, its LME spot and the chain's
bases together. A carry curve (`ForwardRate`) is a rate, and a reference price, the delivery a
fixing reads, and a sampling rule carry no price, so each is mapped null. A basis point of
inflation is one on the breakevens the trade's
`InflationRate` curve is defined by, one row: where an `InflationRatePrices` block solves the
curve, its quotes, every one a basis point up and the block re-solved by its family (`reinflated`,
nothing on the config written), the block a writer `bumps` admits and lists no quote of; where none
does, the nodes' own, each a continuously compounded growth rate r whose breakeven is exp(r) - 1,
written r + log1p(s exp(-r)). An index's prints (`PriceIndex`) are mapped null. The two agree where
the curve is its quotes' strip, each knot its swap's whole years from the last print and the base
reference that print, and part elsewhere: on the inflation module's six swaps a linker's row is
4.8266 on the curve its block solves and 4.5499 on that curve pasted under the three-month
interpolated rule, 4.5467 against 4.5499 under a single two-month one. ISDA's Risk_InflationVol and
Risk_CreditVol name vols the engine has no factor type for, and a mapping naming either is refused
as one naming any risk type no move writes. **Held by**
`tests/test_simm.py::test_a_commodity_row_is_every_price_of_the_commodity_a_percent_up`,
`::test_an_inflation_row_is_a_basis_point_on_every_breakeven_of_its_curve`,
`::test_a_quoted_inflation_curve_moves_on_its_blocks_breakevens_re_solved`,
`::test_an_inflation_curve_that_is_its_own_strip_moves_alike_quoted_or_pasted`,
`::test_a_risk_type_no_factor_of_the_engine_carries_is_refused_by_name`.

The margin is the methodology's aggregation of each netting set's CRIF as one torch function of its
rows' USD amounts (`SIMM.margin`), the thresholds being in USD and the amounts crossed off the
`FxRate` factors as the CRIF's `AmountUSD` is, read off the `Parameters` file under the block's
`Horizon`, every key it reads refused by name where the file lacks it. Per product class and risk
class (`SIMM.RISK_CLASSES`, paragraphs 5 and 6) a delta, a vega and a curvature margin
(`SIMM.aggregated`, off `SIMM.exposures`): a factor's weighted exposure is concentrated over its
group, a rate currency's curves and inflation and never its basis, a credit issuer's vertices,
which are weighed apart under its same-name correlation (paragraphs 8(b) and 42), any other factor
alone; a rate vega is its row, the vol-weighted vega, and an FX, equity or commodity vega its row
times the vol its delta weight implies and the class's HVR, a pair's off the groups of its two
currencies in order (paragraph 10); a curvature scales a vega by its expiry, correlated squared,
equity's bucket 12 taking none and a rate's over HVR² (paragraph 11); the risk classes of a product
class are correlated by the class correlations and the product classes summed. A set's margin is
read in two directions, collected on its rows as written and posted on their negation, which moves
the curvature alone, each under the regime the agreement declares for it. A norm at zero is zero
(`utils.sqrt_or_zero`) and a tie takes the branch a strict comparison selects, a concentration of
one, a curvature's theta of zero and its floor, a bucket's own sum, so nothing is smoothed;
`Margin_Rows` is each row's derivative of its set's margin, by autograd, and its share. The
methodology's add-ons and multipliers (its section L) are not applied. **Held by**
`tests/test_simm.py::test_a_one_row_book_margins_by_hand`,
`::test_the_margin_is_the_notebooks_on_the_synthetic_parameters`,
`::test_the_margin_rows_gradient_is_the_margins_own_derivative`,
`::test_a_parameters_file_lacking_what_the_margin_reads_is_refused_by_name`. **Not gated:** the
residual buckets, the Credit Non-Qualifying class, a cross-currency basis row and an equity
volatility index, which no gated document writes; a vega concentration that binds, within a bucket
or across rate currencies; the shares summing to the margin below every threshold; a one-day
section of a file, read under the ten-day constants
([limitations](../limitations.md#the-engine)).

`SACCR` reads the one compile and moves nothing: each netting set's exposure at default under
CRE52, `1.4 (RC + PFE)`, off its trades' values and their terms as each type declares them
(`schema.Exposure`, `exposure` on the class, `schema.EXPOSURES` holding each type's own as
`schema.OBSERVES` does `observes`): the fields its notional and its currency, or its units and the
factor pricing a unit, its start and its long side are read off, and an option's strike, expiry and
call; its end and maturity are the last day `schema.tenor_fields` names. A trade's asset class is
the `Asset_Class` the `Mapping`'s `SACCR` section states for its type, a credit entity's rating
the `Rating` under its survival curve's `SACCR` entry, an
equity an index where its SIMM bucket is 11 or 12, a commodity's type its SIMM bucket, whose row in
the `Parameters`' `commodity_sets` is its hedging set and subclass; a type declaring no exposure, one
the mapping classes in none and an unrated entity each refuse by name, and the parameters file is
held to CRE52's shape at setup. An option's P is read off the compiled factors - a swaption's par
rate off the legs it is written on (`TensorCashFlows.get_par_swap_rate`), an FX option's outright
(`utils.calc_fx_forward`) - and a date in years of 30/360 from the base date. A set is margined
where it is `Collateralized`, its MPOR its `Liquidation_Period` in business days floored at ten,
its C its `Opening_Balance` as the engine's balance counts it (`instruments.collateral_credit`), its
TH, MTA and NICA its credit support amounts at their first values, and its EAD is capped at the
set's unmargined one. **Held by** `tests/test_saccr.py::test_the_rates_example_is_cre99s_to_its_printed_figures`,
`::test_the_credit_example_is_cre99s_to_its_printed_figures`,
`::test_the_commodity_example_is_cre99s_to_its_printed_figures`,
`::test_the_rates_and_credit_example_is_cre99s_to_its_printed_figures`,
`::test_the_margined_example_is_cre99s_to_its_printed_figures`,
`::test_a_trade_inside_ten_business_days_is_floored_there_twice`,
`::test_a_margined_set_past_its_threshold_is_capped_at_its_unmargined_exposure`,
`::test_a_margined_sets_collateral_is_its_balance_as_the_engine_counts_it`,
`::test_an_over_collateralised_set_keeps_its_add_on_at_the_five_percent_floor`,
`::test_the_fx_equity_and_commodity_classes_and_an_amortising_notional_read_by_hand`,
`::test_a_trade_whose_exposure_the_declarations_do_not_state_is_refused_by_name`,
`::test_a_parameters_file_stating_an_entry_amiss_is_refused_by_name`. **Not gated:** the MPOR floor
of a set of more than 5,000 trades.

The calculation's `Currency` is required: no default names the currency a book reports in. Before
anything runs, the block - overrides merged, completed as `declared_defaults` completes it - is
judged by its own declarations (`Calculation.judged`, through `schema.validate_block`), and a
`ValueError` names every field it refuses at once: `BaseValuation.run` and `CreditMonteCarlo.run`
judge the merged block, `HedgeMonteCarlo`, `SIMM`, `SACCR` and the diary where they complete it. `Config.validate` reports the
same messages under `calculation`, and a refused block is not walked for factors. A service override
is judged on what it states before it queues (`schema.stated_messages`), an undeclared key refused
([Conventions](conventions.md#declared-defaults)).
**Held by** `tests/test_loud_refusals.py::test_a_calculation_stating_no_currency_is_refused_by_name`.

## Compile phase 1 — `calculate_dependencies` {#compile-phase-1-calculate_dependencies}

`Config.calculate_dependencies` discovers the factor universe, sorts it topologically and splits
stochastic from static ([Dependency System](dependency_system.md)); iterating `dependent_factors` is
iterating that order.

!!! note "Invariant — `calculate_dependencies` is idempotent"
    Both halves read the config and write nothing. Discovery classifies an implied model off the
    implied factor's own block and needs no `Price Models` entry, so none is written for it: a
    placeholder would pollute saved market data and tie `plan_hash` to whether a run had happened.
    A process that reads parameters from that entry refuses by name without one -
    `HullWhite2FactorImpliedInterestRateModel` reads its market prices of risk there. **Held by**
    `tests/test_loud_refusals.py::test_an_implied_model_with_no_price_models_block_is_refused_by_name`.

## Compile phase 2 — `_build_factor_state` {#compile-phase-2-_build_factor_state}

1. **Factors.** `construct_process` for each stochastic factor, `construct_factor` for the rest, a
   missing block skipped with a WARNING.
2. **Leaves** ([@capriotti2011]): `stoch_var`, `implied_var` and `static_var`, `Gradient_Variables`
   choosing which require grad. `Calculation.factor_leaf` mints each, the one seam where a
   calibrated factor carries its quotes' graph as `leaf + (θ* − θ*.detach())`
   ([Quote Sensitivities](quote_sensitivities.md#the-attachment)).
3. **Kernels.** `Deterministic_Kernels` is applied both ways on every run, the switch being
   process-global. **Held by**
   `tests/test_recompute_inner_mc.py::test_deterministic_kernels_is_read_every_run_and_set_both_ways`.
4. **State.** `_init_shared_mem` builds the state around `get_cholesky_decomp`, the only place
   `process_ofs` is filled.
5. **`precalculate`**, once `factor_key` is set, fills `z_offset` and `spot0` (its arguments cached
   for a burn-in or a fork); **`calc_references`** follows.

!!! warning "Invariant — `precalculate` before `calc_references`"
    A cross-process link needs what `precalculate` set on the other process. Both loops run in
    topological order, which is also what makes publishing as you go safe.

!!! warning "Invariant — implied-leaf dedupe"
    A factor that is both a static dependent and a spot process's implied factor is one tensor (the
    static branch reuses `implied_leaves`), so `backward()` sums both consumers into it. A second
    leaf under the same name splits the gradient between the pricer path (`t_Static_Buffer`) and the
    scenario path (`implied_tensor`), and desyncs a bump.

### Correlation and `process_ofs` {#correlation}

`correlation_name` returns `(corr_type, [sub-factors])`, one random row per sub-factor, and
`process_ofs[key]` is the first. An undeclared pair in `Correlations` reads 0.0, and a name no
simulated factor answers to is named at INFO. A declared number is the innovations' correlation for
every process; nothing scales it.

!!! warning "Invariant — `num_factors()` equals `len(correlation_name[1])`"
    The first sizes the random block, the second the matrix and the offsets; nothing asserts it. A
    mismatch shifts every later `z_offset`, so a process reads another factor's rows or fails on a
    shape. **Not gated.**

Below an eigenvalue of `1e-8` the matrix is repaired with a WARNING, its size unreported:
`Eigenvalue_Raising` (default) floors eigenvalues at `1e-4` and rescales to a unit diagonal;
`Alternating_Projections` is Higham's method [@higham2002] without Dykstra's correction (100 passes
at most, to a relative `1e-8`), a nearby matrix, not provably the nearest. The matrix is no AAD leaf
([decision 12](../limitations.md#decisions-waiting-on-the-desk)); a base valuation bumps a quanto,
compo or mid-curve correlation under common random numbers (`Correlation_Bump`, 0.025).

## The batch loop

`CreditMonteCarlo.execute` compiles the deal tree (`set_deal_structures`), then per batch:

1. Under `deterministic_batches`, `batch_seed(Random_Seed, b)` reseeds and `set_quasi_batch(b)`
   anchors the quasi stream, `b` the global batch.
2. `reset` correlates one `randn(num_factors, n·T)` into `t_random_numbers` `(num_factors, T, n)`,
   resets cashflows and clears `t_Buffer` and `boundary_sets`; `Antithetic` appends the negation of
   a `Batch_Size // 2` draw, pairing path `j` with `j + Batch_Size/2` (an odd `Batch_Size` is one
   column short, unrefused).
3. Each process's `generate` publishes to `t_Scenario_Buffer` in topological order.
4. `resolve_structure` prices each deal, accumulates and runs `post_process`, negating a
   `FLIP`-prefixed set, dropping a NaN one and stamping each boundary registration with its deal.
5. The CollVA, initial-margin, FVA and CVA reductions run in that order
   ([xVA](../theory/valuation_adjustments.md)), do-not-touch
   ([Change Scope](conventions.md#change-scope)).

!!! note "Run controls"
    `Stats` counts deals and structures loaded, `Ignored`, and `Deals Skipped` / `Structs Skipped`
    once each: a deal that fails to compile is dropped (kept at zero under
    `Exclude_Deals_With_Missing_Market_Data: No`), a pricing exception marks zero unless
    `utils.is_fatal_pricing_error` names it (out of memory, an unbound schedule,
    `UnpriceableSchedule`). `Device_Memory` reports the peak, warning past the device; a run that
    valued nothing refuses. **Held by** `tests/test_fork_pricing_failures.py`,
    `tests/test_multi_gpu.py::test_a_run_reports_what_it_held_and_warns_where_it_passed_the_device`.

!!! warning "Invariant — `t_Buffer` per batch, `t_PreCalc` per calculation"
    `t_Buffer` is cleared by `reset` and by every fork. `t_PreCalc` holds what every batch reads
    unchanged (integral precalcs, gathered indices, arrays through `Calculation_State.upload`) and
    nothing carrying a graph.

!!! warning "Invariant — publish as you go, under `(factor_key, kind)`"
    A process publishes its path as `generate` returns and any sufficient statistic under
    `(factor_key, kind)`, so a linked factor reads its parent's path. The calculation and the solver
    never name a regime, belief or variance; they iterate model-agnostic verbs.

## Random streams {#rng-substream-ordering-the-reproducibility-surface}

`process_ofs` follows discovery's canonical order, a function of the factor graph alone;
[Dependency System](dependency_system.md#rng-ordering) says how and what moves it.

| stream | seeded by | read by |
| --- | --- | --- |
| `t_random_numbers`: Mersenne Twister on the host [@matsumoto1998], Philox on CUDA [@salmon2011] | `Random_Seed` (credit and hedging Monte Carlo, default 5120) plus `job_id`, or per batch `batch_seed`, one SplitMix64 round [@steele2014] | every `generate` |
| `quasi_rng`: Sobol [@sobol1967], Joe–Kuo directions [@joe2008], scrambled [@owen1998] | `Quasi_Seed` (every calculation, its convention `schema.QUASI_SEED`) from `QUASI_ANCHOR = 1024` | regime, GARCH and basis-linked spots, LogVar2FJ's extra mixers, `reset_inner` |
| `inner_block`: those points as integers | `Quasi_Seed` and its shape | pricers' [inner paths](#inner-block) at every outer batch size |
| a `torch.Generator` per LogVar2FJ walk segment | the scenario walk: an integer off `Random_Seed`'s generator; a pricer's walk: `Quasi_Seed`; each plus the segment | the daily shocks, redrawn in backward |
| a fit's fixed Sobol paths | the fitted block's own `Quasi_Seed`, the same convention | the Hull-White two-factor Monte Carlo pricing of its benchmarks, a LogVar2FJ fit under `Sampling: Sobol` |

A base valuation has no scenario stream and declares no `Random_Seed`: every draw it reaches is
`Quasi_Seed`'s, so a trade's mark is a function of the trade, the market and that seed whatever is
booked beside it. **Held by**
`tests/test_recompute_inner_mc.py::test_a_base_valuation_values_a_trade_on_its_own_draws_in_any_book`,
`tests/test_logvar2fj_json.py::test_a_logvar2fj_autocall_is_valued_on_its_own_walk_in_any_book`,
`tests/test_recompute_inner_mc.py::test_a_second_quasi_seed_moves_the_mark_by_its_monte_carlo_error`.

`quasi_rng` keeps its position across batches. Under `deterministic_batches` (the `runparallel`
dispatch only) batch `b` reads from `QUASI_ANCHOR + b · span · sample_size` on any worker, and a
second draw of one shape in a batch refuses by name. A sharded run is then bit-identical in the
worker count on one device type, and a different draw of the same law from an unsharded one.
**Held by** `tests/test_multi_gpu.py::test_a_sobol_consuming_world_is_bit_identical_in_the_worker_count`,
`::test_the_unsharded_hmm_path_is_untouched`.

## Deals and schedules

`Factor_dep` is a deal's compiled factor lookup, consumed unchanged by `generate`
([Resolver Layer](resolver_layer.md)); `Time_dep` maps its dates onto the MTM grid, where
`pricing.interpolate` puts its price (zero past its last date) and saves `Calc_res`. A deal
booked over `Children` whose `post_process` prices them (a swap, cap, floor or swaption on its
legs) compiles no legs off its own terms, which nothing would read (`Deal.has_children`, stamped by
`set_deal_structures`), so a reset they leave unstated warns nobody; a swaption reads off them only
its vol's geometry, the periods its tenor and its rate's level are measured on, which fix nothing,
and, starting after its expiry, the gap's legs, which reset after the base date. A leaf's
unstated reset is fixed at 0% with
a warning naming it (`utils.unstated_reset`). A swap, cap or floor on its legs revalues on their
days (`Deal.children_dates`) and is their sum, each priced on the parent's grid
(`Deal.sum_children`).
**Held by**
`tests/test_declared_defaults.py::test_a_cap_on_its_optionlet_list_states_no_principal_or_known_rates`,
`::test_a_swaption_on_its_legs_compiles_no_swap_of_its_own`,
`::test_a_swap_on_its_legs_is_their_sum_and_compiles_no_legs_of_its_own`,
`::test_a_swap_on_its_legs_revalues_on_their_days`,
`tests/test_swaption_exercise_boundary.py::test_an_amortising_swap_reads_its_vol_at_its_duration_equivalent_tenor`,
`::test_a_forward_start_swap_reads_the_mid_curve_vol_of_two_co_starting_swaps`.

### The schedule lifecycle {#the-schedule-lifecycle}

A `TensorSchedule` is numpy index columns for the compile and a device copy for the arithmetic,
separated in time by `bind`, which `utils.bind_schedules` wraps round `calc_dependencies`. Before it
only compile-time edits write (`carry`, `overwrite_rate`, `set_resets`, `insert_cashflow`); after it
`dual` serves the copy and an edit raises `ScheduleLifecycleError`, since it would not reach the
copy. `BenchmarkInstruments` (last, `bind` splicing in the quote overlay) and the Hull-White
calibration's `create_market_swaps` bind for themselves; `MtMCrossCurrencySwapDeal.post_process`
reopens a leg and binds it again. `Diary` reads schedules by the binding's own walk (**Held by**
`tests/test_diary.py::test_the_diary_reads_the_legs_the_binding_binds`).

!!! warning "Invariant — `derived` is run-scoped"
    A tensor derived from a schedule's copy belongs in `derived`, which `bind` mints fresh, never
    in `Factor_dep` (compile output) or `t_Buffer` (cleared per batch). An unbound schedule raises
    on first touch and is fatal, so no guard turns it into a zero mark.

## Grids {#grids}

| grid | dates |
| --- | --- |
| `Time_Grid` | items `offset` or `offset(step)` in units `D`, `W`, `M`, `Y`, stepping to the next item; default `0d 2d 1w(1w) 3m(1m) 2y(3m)`. A trailing `)` repeats the last step to the book's last date, else later deal dates are clipped. Unadjusted for business days |
| MTM | the parsed grid plus every later date a deal revalues on (a collateralised set adds call, settlement and liquidation dates) |
| scenario | the MTM grid under `Dynamic_Scenario_Dates: Yes` (default; forced by `At_Percentile`), else the parsed grid plus one date past the book; between scenario dates a factor is interpolated linearly |
| deal | each deal's revaluation dates, interpolated onto the MTM grid |

The report rows (`report_index`), on which the xVA reductions run, are each top-level structure's
MTM dates to its last revaluation, a collateralised set's only where its settlement and liquidation
dates are on the grid ([xVA](../theory/valuation_adjustments.md#notation-and-the-grid)).

## Valuation modes {#valuation-modes}

**`BaseValuation`** is one date, every factor static, no Cholesky or generate loop:
`resolve_structure` runs once, `pricing.greeks` reports `Greeks_First`, and `Greeks: 'All'` adds the
full Hessian `Greeks_Second`, reported nowhere else. A Monte Carlo pricer still simulates inside
(`MCMC_Simulations` 16,384; 2,048 on the credit Monte Carlo). Every deal revalues on the base date
alone, a collateralised set whatever its `Collateral_Call_Frequency`, so a set reports its trades'
net uncollateralised and `SIMM` and `SACCR`, pricing by one, value every trade it holds. **Held by**
`tests/test_collateral.py::test_a_base_valuation_values_a_set_whatever_it_calls_at`,
`tests/test_saccr.py::test_the_margined_example_is_cre99s_to_its_printed_figures`.

**`ScenarioValuation`** is one date against B given markets: a base valuation whose `Scenarios`
are data, each the price factors it states over the job's own, which `SIMM` fills from its moves
and a historical or stressed market would fill from an archive or a file. Where it has scenarios,
every factor of a type that carries a scenario axis (`SuppliedValuesModel.factor_types`: spots,
bases, price indices, every curve and every vol surface and space) but the base currency's price is
a `SuppliedValuesModel`, which draws nothing and publishes its value on each scenario of a pass as
one path on the one row, a Skew or SVI surface a leaf a parameter (`Calculation.split`); a model's
parameters and a correlation stay static, so a scenario stating one is priced in a pass of its own,
the factor written into its leaf and put back. A pass is
`Scenario_State.reset`, each process's `generate` and `resolve_structure`, its state the base
valuation's with a scenario axis, so a deal prices on each scenario as a base valuation of that
market does: the smooth estimator, a collateralised set's net, a Monte Carlo deal's inner paths
the canonical block broadcast over the scenarios, and an amortising swaption's vol tenor read off
each scenario's own curves. `Valuation_Date` ages the deals to a later date as a credit Monte Carlo
prices a later grid point, the scenarios the markets there. **Held by**
`tests/test_simm.py::test_a_scenario_is_the_move_it_was_filled_with`,
`::test_a_scenario_stating_a_models_parameters_is_priced_alone_on_them`,
`::test_a_later_valuation_date_values_each_market_as_a_base_valuation_on_that_date`,
`tests/test_collateral.py::test_a_scenario_valuation_values_a_collateralised_set_as_a_base_valuation_does`.
**Not gated:** the leaf a pass of its own writes put back before the next pass.

`utils.VolSurface` reads a surface as `[nodes, paths]`, a static one on one path and a supplied
one on each scenario, and every gather keeps the trailing path axis: the flat grid's node gather and
its Malz variance blend, a vol space's row gather in expiry and its blend in vol tenor, the Skew and
SVI parameters at each expiry, and the moneyness read, each path its own node off its own surface; a
Skew wing whose `lam` or `rho` is zero on a pillar the read blends is held flat on each path whose
own pillar holds it. So a scenario stating a surface or a space rides the pass with the others, and
a static one reads as it did, to the bit. A surface published on more than one row, a simulated
one, is refused by name, no vol read taking the time grid a curve's row gather reads. **Held by**
`tests/test_simm.py::test_every_kind_of_surface_reads_its_own_market_in_a_pass_of_several`,
`::test_a_scenario_is_the_move_it_was_filled_with`. **Not gated:** the refusal, no process
publishing a surface on more than one row.

**`HedgeMonteCarlo`** runs the credit Monte Carlo engine and harvests its marks. `update_factors`
adds `Scenario_Factors` (factors no deal reaches) and caps the grid at the liability terminal; a leg
that fails to compile kills the run, a skipped leg changing the solver's problem (**Held by**
`tests/test_fork_pricing_failures.py::test_a_hedge_book_leg_that_fails_to_compile_kills_the_run`).
Per batch it publishes print seeds first and drops them once generated (consumed state, not path
series), burns in under `Randomize_Initial_State`, replays
`Observed_Scenario` paths through `reseed_from_path`, and under `solve_hedge` snapshots the outer
buffer for forks. A solve is a stream: a `Bundle` per batch to one `StreamingSolve` (`warmup`
locking the frame, `step` per training batch, `finish` on the held-out last), so
`Simulation_Batches` is a stream length under `solve_hedge` and a path multiplier under
`simulate_only`.

!!! warning "Invariant — a frozen evaluation is the stream of length one"
    Under `DiffV2_Load_Value_Fn` nothing is fitted, so `Simulation_Batches` must be 1, that batch
    both warmup and held-out world; `step` skips a loaded net, and `finish` does not re-fork its own
    bundle, which would advance the Sobol stream and move the verdict. **Held by**
    `tests/test_frozen_policy_is_frozen.py`.

!!! warning "Invariant — `Z.ndim` dispatch"
    One process's `generate()` serves both loops: outer `(T, B)` and inner `(T, B, B2)`, the
    per-outer-path state broadcast on the middle axis.

!!! warning "Invariant — a fork's batch state, and pricing that fails loud"
    `simulation_batch` and `fillvalue` track the fork's batch and are restored in a `finally`, or a
    mid-fork raise fails the next step on shapes instead of its cause. A swallowed failure inside a
    fork is a zero mark that corrupts the solver's labels, so the liability is checked on its flat
    shape and every live tradable on having a `tensor_marks` entry. **Not gated:** no document can
    inject a failure into a fork.

!!! warning "Invariant — `keep_tensor` carries the tradable series"
    `HedgeMonteCarlo` sets `keep_tensor` whatever `Keep_Tensor` says; without it the tradables'
    `Calc_res['tensor']` vanish with no error.

## The inner-MC fork {#inner-mc-subsystem}

`_run_inner_mc_at_t` forks each outer path's state at step `t` on process copies, truncates the grid
and each deal's `Time_dep` to `{t, t+1}`, and runs one pass of `Batch_Size × Inner_Sub_Batch`
samples, its memory set by those two fields. It draws (`reset_inner`), precalculates each
process from `outer_buffer[key][t]` (a leaf under `with_grad`), publishes every seed before any
generate (a seed may condition a process upstream), generates and calls `reseed_inner_state`, then
publishes as a `ScenarioSource` every path series it wrote that the outer path carries — the outer
past at `B_outer`, the forked rows flattened — and prices once. Every series, not every factor: a
process's own `(key, kind)` publication (`basis_mu`) is read like a factor and needs the same grid;
the fork's seeds are excluded by the same rule. **Not gated:** no test forks a process series.

The LogVar2FJ outer publishes its detached state (`lv_ell`, `lv_s`), grid days (`lv_days`)
and a fork's seed (`ell0_inner`, `s0_inner`); `pricing.LogVar2FJKit.carried` reads the seed, else
the state at the row's node, else re-seeds at `(L*(t_row), 0)`, leaving other documents unaffected.

### The LogVar2FJ outer {#logvar2fj-outer}

`LogVar2FJImpliedSpotModel` is the xVA scenario generator for an `EquityPrice` or `FxRate` under LogVar2FJ (the model: [theory](../theory/stochastic_volatility.md#one-walk-three-readers)). What a developer needs from it:

- **One walk.** It is an implied process reading the calibrated `LogVar2FJModelParameters` factor through `implied_tensor`, so a CVA sensitivity reaches the one leaf the pricer reads; and its `draws` and `residual` ARE `pricing.LogVar2FJKit`'s, bound on the class, so outer, fork and pricer cannot disagree. Do not give it a draw or a root of its own.
- **Four rows per name.** `correlation_name` is `('LognormalDiffusionProcess', [(), ('S',), ('L',), ('G',)])`: the return's Gaussian given the mixer, the fast and slow log-variance shocks, and the mixer through its quantile `G = F_IG⁻¹(Φ(Z_G))`, taken in double. Each variance factor steps node to node by its exact OU transition driven by its framework normal, the days inside the interval bridged between the endpoints; whole trading days plus one shorter remainder step span each interval.
- **The day's shocks are a function of the day.** One `torch.Generator` per calendar-anchored segment of `LogVar2FJKit.CHECKPOINT_STEPS` trading days draws them; they are never stored and are redrawn in the checkpoint's recompute, so where the scenario nodes fall moves no draw. The antithetic half takes `-eta`. `Checkpoint_Outer_Walk` (a `CreditMonteCarlo` field, default `Yes`) puts the walk under the checkpoint.
- **No replay.** The variance carries its own shocks, so `(ell, s)` is not a function of realised returns and `reseed_from_path` refuses by name.

## Curve reads {#curve-reads}

A read is a query, logical storage and one physical interpolation, never two in one object.
`CurveTensor`, the query, holds scenario rows (`index`, `index_next`, `alpha`), not a flattened
`row × n_tenors` offset, because a tenor segment has its own stride. A `ScenarioSource` is a
sequence of `ScenarioBlock`s placed by `first_row` and `batch_index`, published only by a fork.
`utils.build_interpolation` builds, recursively, an `Interpolation` (one tensor) or a
`SegmentedInterpolation` (leaves over the tenor axis for a `Near_Interpolation` curve, the far leg
what `Price Factor Interpolation` resolves), both `UnroutedInterpolation`s, or for a source a
`RoutedInterpolation` over its blocks, so a segmented curve in a fork is no special case.

- **Blocks, not a join.** The past is identical across inner draws, so a join would copy it
  `Inner_Sub_Batch` times; `ScenarioBlock.project` widens a read's result, never the stored tensor.
- **Order.** Raw read, time blend, then `combine` (RT scaling, segment select): projection and
  `combine` are linear, so the routed path is the unrouted arithmetic in its order. A blended read
  reaches `index + 1`, so `route` classifies on where a read ends.
- **Write-once.** A source answers only `shape`, `new` and the RT rescale, so a late write fails
  loud rather than materialising the grid.
- **Hermite at the read.** A graph-free curve derives its Hermite pair per read
  (`Interpolation.hermite_at`), equal to the held pair to the bit; a curve carrying a graph holds
  the pair, or a derivative would sum its reads after the Hermite map, moving every greek in the
  last bit.

**Held by** `tests/test_interest_rate_prices.py::test_the_near_splits_far_leg_is_the_curves_own_rule`,
`tests/test_forward_rate_extrapolation.py::test_a_hermite_curve_document_prices_the_cubic_by_hand`.
**Not gated:** routed against joined reads, and the refused late write.

## The canonical inner block and the streamed strip {#inner-block}

**One block.** `Calculation_State.inner_block(rows, sims)` returns `(u, 1 − u)`, row `k` fixing `k`,
each inner path one Sobol point over the strip scrambled by `Quasi_Seed`, taken by the five
one-step-survival pricers under every calculation at every outer batch size. Chunk `c` of
`INNER_CHUNK = 64` rows is dimensions `64c` on of a `64(c + 1)`-wide engine, so a longer strip
reads more dimensions, never more points (point `i + 2^k` is point `i` XOR a constant). The
complement is taken on the 30-bit integer, keeping each tail in the job's dtype. Cached per inner
path count, prefix-stable and broadcast over the outer batch, the block makes a trade's value
independent of what is booked beside it and an inner-MC book shardable with nothing to anchor;
the inner error is common to the outer paths, and only `MCMC_Simulations` reduces it
[@gordy2010].
**Held by** `tests/test_recompute_inner_mc.py::test_a_trade_is_valued_on_its_own_draws_whatever_is_booked_before_it`,
`tests/test_multi_gpu.py::test_an_inner_monte_carlo_book_is_bit_identical_in_the_worker_count`,
`tests/test_barrier_bridge.py::test_a_strip_longer_than_a_block_chunk_prices_to_an_independent_simulation`.

**The strip streams.** The four one-step-survival pricers (discrete barrier, accumulator, TARF,
autocall; [@glasserman2001; @alm2013]) walk their fixings through `pricing.oss_stream`. Under a
gradient on the device a strip goes `ceil(sqrt(n))` fixings at a time, each stretch one checkpoint
holding the state at its edges. A taped run holds those edges to its batch's end, so there a strip
is one checkpoint where its projected tape, its fixings times what a row-step of its pricer tapes,
fits `pricing.device_budget`, half of what the process may still allocate. In a replay
(`Recompute_Inner_MC`), whose tape dies with its block's backward, a stretch walks plain, the
fastest walk, while the tape the strip has laid and its last stretch's for every fixing still to
walk fit the budget; the projection is re-read at every stretch, a strip's first fixings (an
observed one, the first step) laying less than the rest. On the host or with no gradient it is the
plain loop, and so is a smoothed row, its ledger mutated step by step. A step binds its row's
tensors as defaults, since the checkpoint replays it after the loop moved on, and what consumers
read detached (the alive branch, the pending payment, the autocall's counterfactual) is built
under `no_grad`. **Held by**
`tests/test_recompute_inner_mc.py::test_the_streamed_strip_holds_its_peak_and_the_unstreamed_gradient`,
`::test_a_taped_strip_that_fits_is_one_checkpoint` (CUDA only).

**A dense grid is priced in pieces.** A pricer prices the time-grid rows sharing a fixing interval
together, and a recompute's backward replays a node's rows at once. Under a gradient on the device
`pricing.oss_rows` measures what a row-step of each pricer tapes and what its replay peaks at, once
per run on the first row of the first block whose strip streams, the longest in its block, walked
on detached copies; under a recompute it hands the node a block in pieces
of as many rows as the budget replays at the block's own fixings, so a longer strip is cut on its
own length, and a block that fits is one piece, as every block is without a recompute. Each
piece's by-products are performed as a block's. A row's strip reads only its own row of the
block's arrays and draws what it would in the whole block, so the pieces price its rows, and under
`Deterministic_Kernels` their gradient, to the bit. Both decisions log at debug (`OSS_STREAM`,
`OSS_ROWS`), naming the pricer, what was measured and the budget. **Held by**
`tests/test_recompute_inner_mc.py::test_a_daily_grid_marks_the_weekly_dates_within_its_cap`
(CUDA only). **Not gated:** the gradient.

## Recomputing the inner Monte Carlo {#recompute-inner-mc}

The terminal `backward()` holds every Monte Carlo pricing's graph, while the simulation is cheap.
`Recompute_Inner_MC` (base valuation and credit Monte Carlo, default `No`) trades the tape for a
second forward pass [@griewank2000]: `pricing.InnerMCRecompute` runs forward under `no_grad` and
re-runs the same callable under `enable_grad` in backward, a block at a time. The four adopters
(`pv_MC_Tarf`, `pv_MC_AutoCallSwap`, `pv_MC_Accumulator`, `pv_discrete_barrier_option`) call
`InnerMCRecompute.run`, with no per-pricer flag: what a run can tape is the machine's property, not
the trade's. `run` uses `cls.apply`, so a subclassing gate gets its mutation. An adopter:

- is `simulate(*bound)(*theta)`, the block's shape bound and every graph-carrying tensor in `theta`;
- returns a tuple, the marks then by-products the caller performs once, each graph-carrying one a
  top-level element (nested in a list it is no output, its gradient silently zero);
- is one function called twice, never a fast forward beside a differentiable copy;
- is pure but for its return, since a side effect fires again in the replay;
- reads nothing from a closure, which differentiates as a constant: all four build their vol strips
  at the call site and pass the spot model's scalars, and the autocall hoists its floating leg and
  past fixings (**Not gated:** those two).

**Nothing drawn is stored.** The node saves its inputs, not its draws: the inner rows are the
canonical block's and a LogVar2FJ walk's shocks are keyed by `Quasi_Seed` and the segment, so the
replay draws again what the forward drew. **Held by** the `…_bit_identical_with_the_node_on` tests in `tests/test_recompute_inner_mc.py` and
`tests/test_recompute_equity_pricers.py`, by `array_equal`, since a desynchronised stream prices
right and differentiates wrong.

!!! warning "Invariant — settle outside the callable"
    Settled inside, a cashflow books twice, and the credit Monte Carlo saves cashflows after the
    batch's backward, so the report sees both. The forward's booking carries no graph, so a
    collateralised exposure reading the ledger through `C_ts_te` loses that channel of its gradient.
    **Held by**
    `tests/test_recompute_inner_mc.py::test_the_exposure_its_cashflows_and_the_cva_gradient_are_bit_identical_with_the_node_on`,
    `tests/test_recompute_equity_pricers.py::test_a_collateralised_gradient_is_what_the_settle_outside_rule_is_gated_on`
    (whose docstring records the size).

!!! warning "Invariant — a first-order node"
    The replay is rooted at detached inputs, so its `autograd.grad` cannot walk into the outer graph
    and count the first derivative twice; a second derivative through it is severed, and `backward`
    refuses `create_graph`, naming the switch. It asks `torch.is_grad_enabled()` before
    `enable_grad`: the engine runs a backward node in `create_graph` mode, and inside `enable_grad`
    the flag is always True. **Held by**
    `tests/test_recompute_equity_pricers.py::test_the_second_derivative_is_refused_rather_than_reported_wrong`.

## Boundary corrections — the sensitivity subsystem {#boundary-corrections-the-sensitivity-subsystem}

Some deals decide on simulated state: a barrier, an autocall trigger, a swaption exercise, a
collateral transfer clearing its minimum. The price is right; ordinary AAD drops the flux of
scenarios across the trigger, whose indicator has zero derivative almost everywhere.
`pricing.stochastic_boundary_correction` adds the missing term, exactly zero forward
(`gap − gap.detach()`), so nothing reported moves and the derivative reaches every factor in one
sweep. **Held by**
`tests/test_boundary_pricer_events.py::test_asking_for_sensitivities_does_not_move_the_autocall_exposure`,
`tests/test_boundary_tarf_events.py::test_asking_for_sensitivities_does_not_move_the_tarf_price_or_exposure`.

**The estimator.** For a decision $\mathbb 1\{G>0\}$ whose firing changes the objective by $J$, AAD
drops

$$f_G(0)\;\mathbb E\Big[J\,\frac{\partial G}{\partial\theta}\;\Big|\;G=0\Big]$$

([@lyuu2011]; [@glasserman2003], ch. 7). `pricing.boundary_weights` estimates the two factors apart
from $n$ gaps $g_k$: with $h=$ `Boundary_AAD_Bandwidth` $\times\ \mathrm{sd}(g)$,
$K_k=e^{-g_k^2/2h^2}$ and $s_r=\sum_k K_k g_k^r$,

$$\hat f_G(0)=\frac{s_0}{n\sqrt{2\pi}\,h},\qquad w_k=\frac{K_k\,(s_2-g_k s_1)}{s_2s_0-s_1^2},$$

a kernel density and local-linear weights [@fan2018] summing to one, which hold still over a range
of $h$ where a local-constant fit carries an $O(h)$ bias. The term is
$\sum_k (g_k-\bar g_k)\,\hat f_G(0)\,w_kJ_k$, $\bar g$ and the coefficient detached; one gap (a base
valuation's scenario) gives zero width and no correction.

`Boundary_AAD_Bandwidth` is 0.01. `Boundary_AAD_Window_Touch` (`Yes`, credit Monte Carlo) registers
the partial-time barrier's window touch. `Branch_And_Weight` (`Yes`, base valuation) prices the four
fixing-observed knockouts with a smooth estimator that registers nothing (**Held by**
`tests/test_branch_and_weight.py::test_a_registration_the_switch_supersedes_is_not_lost`). The
switch is `boundary_aad = calc_greeks is not None`: no greeks, no cost.

A pricer appends a `BoundarySet`, `gap` keeping its graph (`gap > 0` is fired) and all else
detached, its branches mapped to the MTM grid by the deal's own `pricing.deal_to_mtm_grid`. A
collateralised set's `post_process` stamps its gross-to-net chain onto the registrations beneath it,
on the set because `boundary_sets` spans every set. `pricing.boundary_correction` scores them at the
objective (CVA, FVA, the base value).

!!! warning "Invariant — a counterfactual is scored on the portfolio"
    The objective applies to `resolve_structure`'s root sum over every netting set, so a
    counterfactual is the reported portfolio plus this registration's change, never one set's net.
    A one-set fixture cannot tell them apart, and the error is worst where the machinery is aimed: a
    collateralised net sits at the `relu` kink. **Held by**
    `tests/test_boundary_pricer_events.py::test_a_netting_set_is_scored_on_the_portfolio_not_on_itself`.

The classes differ in reach, so no mode flag: `LatchedBoundarySet` reaches every later row (the
latching pricers, the swaption); `InnerBoundarySet` lives inside an inner Monte Carlo (the TARF's
knock-in) and carries the objective's derivative, an inner path moving a row by `1/n`;
`MTABoundarySet` is a collateral transfer and every later balance.
A decision registers one counterfactual with its whole reach, ledger included, since at a kink two
partial ones score differently from their sum; only booked cash may be declared (`cash_events`), the
collateral chain replaying the ledger.

!!! note "Under a recompute node, a `gap` the simulation decided is an output"
    An untaped forward gives `gap` no graph, so the node returns it and the coefficient arrives as
    its cotangent. A decision on outer state keeps its own graph and registers outside: the TARF's
    knock-in and the autocall's coupon trigger are node outputs, the TARF's redemption, the
    barrier's crossing and the accumulator's latches are not. **Held by**
    `tests/test_recompute_equity_pricers.py::test_the_exposure_its_cashflows_and_the_cva_gradient_are_bit_identical_with_the_node_on`.

!!! warning "A decision whose solve amplifies is refused, not smoothed"
    Local-linear weights are signed, so a kernel admitting two neighbouring points far out cancels
    the first-order term by subtracting two large numbers; $\lVert w\rVert_1$ is that
    amplification, and past `pricing.BOUNDARY_MAX_AMPLIFICATION` (25) the weights go to zero.
    **Not gated:** no document makes the refusal fire.

Acceptance is AAD against a common-random-numbers bump ladder (`tests/crn_ladder.py`): a gate
passes when the ladder's spread is at most 10% of its median, since one that scatters with `h` is
differencing across a jump, and its flattest interior rung is within tolerance (`Ladder.agrees`):

| decision, objective | test | paths × batches | tolerance |
| --- | --- | --- | --- |
| continuous barrier, one-touch, no-touch; CVA | `test_barrier_bridge.py::test_aad_delta_matches_bump_and_reprice` | 4,096 × 1 | 2% |
| TARF knock-in; CVA | `test_boundary_tarf_events.py::test_the_knock_in_gradient_matches_bump_and_reprice_under_exposure` | 512 × 4 | 2% |
| TARF knock-in; base value | `test_boundary_tarf_events.py::test_the_knock_in_gradient_matches_bump_and_reprice_at_base_valuation` | 2 × 65,536 inner | 2% |
| discrete barrier latch; CVA | `test_boundary_pricer_events.py::test_discrete_barrier_latch_gradient_matches_bump_and_reprice` | 1,024 × 4 | 5% |
| the same under a CSA; CVA | `test_boundary_pricer_events.py::test_collateralised_barrier_latch_gradient_matches_bump_and_reprice` | 512 × 8 | 8% |
| discrete barrier latch; FVA | `test_boundary_pricer_events.py::test_fva_gradient_carries_the_boundary_term_too` | 1,024 × 4 | 5% |
| exposure kink; CVA gamma, vanna | `test_cva_gamma_kink.py::test_the_gamma_and_vanna_entries_land_on_crn_ladders_of_the_reported_delta` | 65,536 × 1 | 5%, 10% |

**The CVA Hessian.** Second-order AAD drops $\delta(V)\,V_\theta V_\theta^{\top}$ at the exposure's
$\max(V,0)$; `pricing.exposure_kink_term` adds $\tfrac12 K_\varepsilon(\bar V)(V-\bar V)^2$ under
the CVA's weights, zero in value and at first order, a normalised Gaussian kernel at Silverman's
width [@silverman1986] $\varepsilon=1.06\,\mathrm{sd}(V)\,N^{-1/5}$ per report row, $N$ the run's
paths. A row whose density at zero climbs 2× across $(2,1,\tfrac12,\tfrac14)\varepsilon$ is a point
mass and refuses. A book that registered a boundary correction is refused `Hessian: 'Yes'`, as a
base valuation is refused `Greeks: 'All'`: a first-order correction differentiated twice loses its
density-derivative term. The $O(\varepsilon^2)$ bias shows on volga, so it is a gamma-and-vanna term
([xVA](../theory/valuation_adjustments.md#sensitivities)). **Held by** `tests/test_cva_gamma_kink.py`,
`tests/test_base_valuation_gamma.py::test_a_deal_that_registered_a_boundary_correction_is_refused_by_name`.
**Not gated:** the point-mass refusal.

## Open

The correlation as a leaf and the recompute segment length are
[desk decisions 12 and 20](../limitations.md#decisions-waiting-on-the-desk). Limitations are under
[the xVA outer](../limitations.md#the-xva-outer-and-the-correlation),
[the pricers](../limitations.md#the-autocall-tarf-and-barrier-pricers) and
[the engine](../limitations.md#the-engine); Hessian-vector products through the recompute node are
[designed, not built](roadmap.md#designed-not-built).
