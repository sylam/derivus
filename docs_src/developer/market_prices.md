# Market Prices

`Market Prices` holds the quotes a market factor or a model is fitted to; `Price Factors` holds what
the fits write. A *price family*, one class in `derivus/bootstrappers.py`, turns each of its blocks
into one price factor, and `Config.bootstrap` runs the configured families in the order their reads
demand. A family is a set of declarations because the same declarations publish the schema, drive
the web screens and order the run, so the three cannot drift. Each family's method is on its
[Bootstrapping](../bootstrapping/general.md) page; this page is how the framework works and why.

| where | what |
| --- | --- |
| `bootstrappers.py` | the families, `FAMILIES`, `WRITERS`, `bootstrap_order`, `bootstrap_dependents` |
| `config.py` | `Config.bootstrap(only, wanted)` |
| `schema.py` | `emit_market_prices`, `emit_configuration`, `MARKET_QUOTE_VALUES`, `update_market_quote` |
| `service.py` | the verbs that author market data ([below](#verbs)) |
| `derivus_bloomberg/` | the seed, the security map, the curve strip, the surface and chain fetch |
| `derivus_bootstrap.py` | the multiprocessing launcher |

A rule marked **Held by** goes red under that test if it is broken.

## A quote is an instrument, a quote type and a number {#a-quote}

| | |
| --- | --- |
| **the instrument** | a declared instrument type, the thing the number is a price *for* |
| **`Quote_Type`** | what kind of number is quoted, and so what "reprices to the quote" means |
| **`Quoted_Market_Value`** | the number |

A quote never restates an instrument's fields: it names a declared type and carries a block of it,
so the `Instrument` declarations are its schema and the two cannot drift. `Quote_Type` is per family
because one name would ask three questions: the energy family takes vols, the option family a vol
or a premium, a curve a par rate.
**Held by** `test_schema_emission.py::test_a_quote_type_means_different_things_to_different_families`.

**Values and plan.** A quote row's `Quoted_Market_Value`, `Quoted_Bid`, `Quoted_Ask` and `Timestamp`
(`schema.MARKET_QUOTE_VALUES`) are values; everything else in a block is plan. A row joins the value
plane by declaring all four: `Points` and `European_Options` do, so the curve, FX smile and
LogVar2FJ families tick as values and the other three are wholly plan-side. An absent value key is
dropped rather than shadowed, because a pillar that starts being quoted two-sided is the same node
of the same plan. `schema.update_market_quote` refuses an update moving anything plan-side, a moved
node being a new plan and a recompile, and `values_hash` is taken with the clocks projected out
(`schema.without_clocks`), a market's identity being its numbers. **Held by**
`test_market_prices_partition.py::test_the_tick_guard_and_the_partition_draw_the_same_line`,
`test_market_prices_partition.py::test_a_quote_tick_moves_the_values_hash_and_leaves_the_plan_bit_identical`.
What the split buys is on [Quote Propagation](quote_propagation.md#protocol).

## A family maps a quote set to what it calibrates {#a-family}

A family subclasses `Construction` (a closed form or a square system of roots, repricing its quotes
exactly) or `ImpliedCalibration` (a least-squares fit); both are `Family`, which completes the
configured entry from the declarations. The six are tabled on [General](../bootstrapping/general.md).
A family declares:

| attribute | what |
| --- | --- |
| `market_factor_type` | the `Market Prices` type it reads; declared because it differs from the class name |
| `price_factor_type` | the `Price Factors` type it writes under each block's name, and its `Bootstrapper Configuration` key |
| `reads` | the factor types it prices on, which order a run |
| `fields` | the block's schema as `schema.F`, the quote table included |
| `quote_instruments` | the instrument types a quote may be, where quotes are instruments |
| `documentation` | its generated methodology page |

`mapping['MarketPrices']` carries the families and the value plane, so a client ticks what the guard
admits without spelling it; `mapping['Configuration']` is one entry per family, its dials the
declarations less the quote block's Tables and Containers, its class name under `aliases`. **Held by**
`test_schema_emission.py::test_the_configuration_store_is_the_families_own_declarations`.

**Extending.** Declare the attributes, implement `bootstrap(...)` writing
`price_factors['<price_factor_type>.<name>']`, and add the class to `FAMILIES`, which the registry
and the schema read. Read the family's own written factor as its [warm start](#warm-starts); declare
the four value keys on the quote row (`schema.QUOTE_TWO_WAY`) if a tick should move it without a
recompile; for `Quote_Sensitivity`, leave the connected parameters in `calibrated` and the leaves in
`quote_leaves` for `Config.bootstrap` to harvest. The schema gates are parametrised over families
(`test_schema_emission.py::test_the_quote_block_declares_what_the_bootstrapper_reads`); the
partition gate spells them by hand and fails until the new one is added
(`test_market_prices_partition.py::test_every_family_partitions_by_the_one_rule_and_round_trips_exactly`).

## Warm starts {#warm-starts}

A family's own written factor is its warm start, and no field declares it: a written factor is the
evidence that a basin was found, so an implied calibration skips the search and solves from it.

| family | cold | warm |
| --- | --- | --- |
| Hull-White | basin hopping, then least squares | the least squares alone |
| Clewlow-Strickland | L-BFGS-B from `Seed` | from the factor, clipped into the boxes |
| LogVar2FJ | the staged search | the joint polish alone ([below](#logvar2fj)) |
| `FXVolPrices` | a refined grid | the [pinned grid](#fxvolprices) |
| `InterestRatePrices` | the par rates | none: the par rates move with the tick, a previous θ\* does not |

**Held by** `test_warm_starts.py::test_the_written_hull_white_factor_skips_the_search`,
`test_warm_starts.py::test_the_written_energy_factor_seeds_the_minimisation`.

## `Bootstrapper Configuration` {#bootstrapper-configuration}

A quote block carries quotes and the factors they price against; every other dial a family reads
is a hyperparameter, stated once per book in the family's entry. The author's rules are on
[the JSON page](../json/bootstrapper_configuration.md); this is the mechanism.

**Two levels.** A family is built from its entry completed by its declarations
(`schema.declared_defaults`), and each block is read as `dict(section, **block)`, the block winning.
Construction types every dial, so a malformed value refuses by name before a quote is read and
`Config.bootstrap` raises naming the entry: a bad entry writes no factor rather than a wrong one.
**Held by** `test_loud_refusals.py::test_a_family_refusing_at_construction_is_refused_by_name`.

**Keyed by what a family writes** (`InterestRate`, `FXVol`, `LogVar2FJModelParameters`), so a
section reads as the factors a run writes; class names stay aliases in `WRITERS`. `Prices` is the
stem of the `Market Prices` type an entry routes on: `derivus_bootstrap` routes blocks to workers
off it and refuses an entry without it, while in process a missing one is a warning, the family
knowing its type.

**The order is a topological sort.** `bootstrap_order` sorts the entries over what each writes
against what it `reads`, so a curve is solved before the swaption fit on it and a surface before the
fits reading it, whatever the file's order; both paths take it. Independent entries keep the file's
order and a cycle refuses. The curve family solves each block after every block building a curve
its `Discount_Rate` or its benchmark deals name. **Held by**
`test_interest_rate_prices.py::test_the_blocks_are_solved_in_dependency_order`,
`test_interest_rate_prices.py::test_a_forward_block_authored_before_the_curve_its_other_leg_needs_still_solves`.

**Each family is handed only its own blocks** (`market_prices_for`), and a block no family reads
refuses the bootstrap. A block missing a named factor is logged and skipped by the
Clewlow-Strickland, GBM and Hull-White families and refused by the others; on the service an ERROR
refuses the write ([below](#verbs)).

**A run narrows to what moved.** `Config.bootstrap(only=…)` covers the blocks whose numbers moved
plus every block reading one of them (`bootstrap_dependents`): a family's `reads`, and for curves the
very curve a block names, so a curve re-solves the curves discounting on it and no others. `wanted=`
narrows to what named factors stand on, so a calculation bootstraps only what its book reads.
**Held by** `test_service.py::test_a_moved_curve_re_solves_what_reads_it_and_nothing_else`.

**Outside `bootstrap`**, the swaption residual (`calc_loss`) and the curve ride (`propagate`,
`plan_key`) complete their block from the declarations, so the declared default is the fallback
everywhere.

## The verbs that author market data {#verbs}

The [MCP tools](mcp.md#the-tools) and the web screens call these same verbs. A writing verb holds
three invariants:

- **One atomic write.** Blocks, entries, spots, dates and the bootstrap land in one `Book.mutate`
  edit, so the file never holds quotes and factors from two states.
- **The bootstrap judges.** `market_edit` captures the bootstrap's ERROR channel
  (`utils.CapturedErrors`) and anything on it refuses the whole write: a book never carries a market
  its own bootstrap complained about. **Held by**
  `test_service.py::test_a_bootstrap_that_complains_writes_nothing`,
  `test_service.py::test_a_set_up_a_bootstrap_complains_about_writes_nothing`.
- **A terminal trip is an act.** The tick, a verification and a set-up are queued jobs whose id
  hashes the submission clock, so two asks are two trips.

| verb | what it does | its own rule |
| --- | --- | --- |
| `POST /book/market` | installs or value-updates quote blocks, bootstraps what moved | a quote moves under `quotes`, never as a values patch, which nothing would re-solve |
| `POST /book/configure` | merges `fields` into one entry of either configuration section | validated by constructing the family; the market re-bootstraps |
| `POST /book/curve` | authors an `InterestRatePrices` block from benchmark rows | the tenor says what a row is |
| `POST /book/date` | moves both base dates, re-authoring every curve block | goes back as well as forward |
| `POST /book/bloomberg`, `DV_Service --tick` | the tick: surfaces, curve rows and routed spots | freshness first; a routine beat never provisions |
| `/book/securities` | the ticker vocabulary, the verified map, each knot's print | only a terminal writes the map |
| `/book/dependencies` | the factor walk, each missing factor with what would supply it | writes nothing |
| `POST /book/setup` | builds the market a pair, a deal or a subtree needs | nothing lands that depends on a block the write will not carry |
| `POST /book/model` | fits one pair's spot model off its built surface | on request, never on the tick |

**The tick** re-prices every curve block's rows as values and re-crosses each routed
`FxRate.<ccy>.Spot` onto the engine's axis, never triangulating a cross. Every wanted security is
screened by `security_map.stale` before anything is fetched, and one late print refuses the trip,
because a dead series keeps answering with a plausible number. A curve row the screen refuses is
held out with `Use` No, by name: one dead ticker is one knot fewer. **Held by**
`test_service.py::test_a_stale_quote_refuses_the_tick_by_name`,
`test_service.py::test_a_spot_print_the_terminal_will_not_stand_behind_refuses_the_tick`,
`test_service.py::test_the_tick_values_the_books_curve_rows_and_holds_a_dead_one_out`.

**The snap sets the date.** A row priced off a terminal carries the print's clock, so the curve
verb, the tick and a set-up author on the latest print; where that is after the book's day, both
base dates roll onto it and every other curve block is re-authored there, two curves dated
differently being one solved for a day nobody asked about. A snap never rolls a book back: an old
print is evidence about a quote, not a valuation date. **Held by**
`test_service.py::test_a_later_print_rolls_the_books_date_through_the_tick_and_an_older_one_does_not`,
`test_service.py::test_a_tick_landing_while_the_terminal_priced_the_rows_refuses_rather_than_rolling_back`,
`test_service.py::test_the_date_verb_sets_both_dates_and_re_rolls_every_curve_onto_them`.

**Vocabulary and evidence.** The seed (`$DV_HOME/seed.json`) names what a desk could quote; the map
(`$DV_HOME/security_map.json`) is what a terminal verified, beside a `rejected` ledger verification
re-asks, a rejection being one day's answer; `used` joins each curve row to that evidence. **Held by**
`test_service.py::test_a_vocabulary_entry_is_merged_kept_and_refused_by_name`,
`test_service.py::test_every_knot_names_the_print_it_was_solved_from`.

**Set-up.** The want-list is the dependency walk; for a pair a vanilla on it is walked, so each leg's
spot reads the curve its own `FxRate` discounts on. The map is written as it is gathered, before the
book, so a refused set-up leaves a grown map and an unmoved book. Where the prints roll the date, the
market already carried is re-priced in the same session (`market_fetch`), so the book never holds
two days of quotes. **Nothing lands that depends on a block the write will not carry**: a new
currency's spot and curve land together or not at all, the curve's fit reading the `FxRate` that
discounts on it, and a surface is held with whichever leg is not coming; the rest lands.
**Held by** `test_service.py::test_a_set_up_installs_no_curve_the_book_does_not_discount_on`,
`test_service.py::test_what_the_write_can_complete_still_lands`,
`test_service.py::test_a_surface_is_held_with_the_leg_it_cannot_stand_on`,
`test_service.py::test_a_roll_re_prices_the_market_the_book_already_carried`.

## `InterestRatePrices` — a curve solved from its quotes {#interestrateprices}

The residual, solver and refusals are on
[Interest Rates](../bootstrapping/interest_rates.md#interestratecurveparameters).

**The block is the curve's definition.** Beside the quotes it declares the conventions they were
authored under (`Calendar`, `Spot_Days`, each leg's frequency and day count, `Compounding`), and
each `Points` row its `Tenor` (`ON`, `3M`, `1Mx4M` a FRA, `6M1M` a swap starting in six months) and
`Security`. The solve reads none of it, but it lets a strip re-roll on a later date from the block
alone. A blank `Discount_Rate` builds a self-discounting curve.

**The quotes.** Each point carries a `Deal` authored as in `Trade Data`, its `DealType`, `Quote_Type`
(`Par_Rate`: held at PV zero), the number and `Use`, which holds a quote out without deleting it. The
family stamps `Discount_Rate` on every leg, what the set discounts on being stated once. Where the
number lands is the instrument type's own declaration (`Deal.quoted`, `Deal.quote`), so a new
quotable instrument declares itself.

**A forward outright as a benchmark.** An `FXForwardDeal`'s quote is the outright, landing as
`Buy_Amount = quote × Sell_Amount`, which keeps PV affine in the quote. Covered interest parity is
written nowhere: the residual is the forward's own pricer held at zero, the other leg's curve and
both spots entering as constants. `Quote_Sensitivity` and `Quote_Propagation` refuse such a block,
because a quote derivative rides cashflow-schedule columns and an outright moves none; the refusal
is measured, not a branch on the type. **Held by**
`test_interest_rate_prices.py::test_a_forward_curve_reprices_the_outrights_it_was_solved_from`,
`test_interest_rate_prices.py::test_a_forward_block_refuses_quote_sensitivity`,
`test_interest_rate_prices.py::test_the_refusal_is_measured_and_leaves_the_rate_quotes_carrying`.

**The knot rule.** One knot per used quote, at its last cashflow date in the block's `Day_Count`: the
only placement that makes the system square, since a knot nothing matures at is unidentified and two
instruments between two knots leave the curve under-determined. A quote held out drops its knot, one
at the base date refuses, and the output grid is the knot grid, because an interpolated curve no
longer reprices its quotes. **Held by** `test_interest_rate_prices.py::test_a_held_out_quote_leaves_the_solve`,
`test_interest_rate_prices.py::test_a_benchmark_matured_at_the_base_date_refuses_by_name`.

**Interpolation** comes from `Price Factor Interpolation`, the rule naming this curve's `id` else the
`InterestRate` method ([Conventions](conventions.md#registries-not-functions)), except
`Near_Interpolation` up to `Near_Tenor`, which the block owns for a near end quoted in another
instrument; the solve reads both segments. **Held by**
`test_interest_rate_prices.py::test_the_near_split_is_written_through_and_every_quote_still_reprices`,
`test_interest_rate_prices.py::test_a_curves_own_rule_builds_it_and_leaves_its_neighbour_on_the_default`.

**The solve** is a damped Newton in float64 on the host, from the par rates [@dennis1996], its
Jacobian AAD through the engine's own pricers: the derivative
[Quote Sensitivities](quote_sensitivities.md#the-ift-contract) carries the other way into `dV/dq`,
so the residual is written once and differentiated twice. A discount curve is solved before the
projection curve on it [@ametrano2013]; where a quote derivative or a ride asks, blocks reading each
other's curves solve as one system ([multi-curve](quote_propagation.md#multi-curve)). **Held by**
`test_interest_rate_prices.py::test_the_bootstrap_recovers_the_curve_its_quotes_came_from`.

## `LogVar2FJModelPrices` — a smile fit with a forward-skew term {#logvar2fj}

The model is on [the theory page](../theory/stochastic_volatility.md); the stages, objective rows,
prior weights and report on [its Bootstrapping page](../bootstrapping/fx_and_equity.md#logvar2fjmodelparameters);
this is why it is built so.

**The block** is `OptionQuoteFamily`'s quote preparation plus this family's dials.
`derivus_bloomberg.equity_chain` emits one off a listed chain as premiums; `fx_surface_block` authors
the FX one off a built surface: two delta pillars at four wing expiries, because a `Bootstrap`
bucket frees one parameter per wing quote, and at least eight distinct contracts after snapping,
because a repeated contract is a weight, not an observation. **Held by**
`test_market_prices_partition.py::test_the_ladder_is_the_entrys_and_the_contract_floor_is_a_declared_lever`.

**ξ is stored and the level derived.** The factor stores `ξ(t) = E₀[h_t]` and derives
`L*(t) = log ξ(t) − ½Var(ℓ_t + s_t)`, so `E₀[h_t] = ξ(t)` whatever the vol-of-vol; a stored level
would drag the ATM level with every vol-of-vol move. ξ is flat forward variance between ATM expiries,
the shape a variance-swap strip is quoted in [@demeterfi1999], so every ATM reprices exactly
([theory](../theory/stochastic_volatility.md#the-forward-variance-curve)).

**The residual is NIG on the variance clock, its drift forced**
([theory](../theory/stochastic_volatility.md#the-residual); [@barndorffnielsen1998]): no intensity,
no compensator, no parameter the greeks cannot reach. `Alpha` and `Beta` are fitted in unconstrained
coordinates that stay admissible at every iterate, and the factor asserts admissibility at load.
`Residual_Law: Gaussian` is a limit and test mode whose `Alpha` and `Beta` report zero rows, so the
sensitivity set does not depend on the mode. **Held by** `test_logvar2fj_json.py::test_the_gbm_limit_reproduces_the_gbm_arm`.

**ξ is re-solved at every outer iterate**, segment by segment against each ATM premium (triangular:
an option to T never reads ξ beyond T), so every candidate reprices the ATM term structure exactly
and is judged on the smile alone. Each segment is a damped Newton (`Xi_Solve_Iterations`,
`Xi_Solve_Damping`, `Pillar_Tolerance`) returning one Newton step at its root,
`x* − F/detach(∂F/∂x)`, so `dξ/dθ` and `dξ/dq` ride the tape and the outer `least_squares`
[@branch1999] keeps an exact Jacobian. **Held by**
`test_logvar2fj_json.py::test_the_fit_reprices_its_atm_strip_fits_its_wings_and_writes_every_field`.

**The walk is a closed form on the quotes' own grid.** Both factors are linear in their shocks, so a
block's path is one cumulative sum (`utils.LogVar2FJ.ou_path`); under `Invert_Spot` the measure shift
is the state's own `√V`, so that case keeps the recursion. A block ends at every quoted maturity,
forward window end and bucket knot, so the fit reads at a maturity what a pricer reads there.

**`Vanilla_Pricer: Quadrature`, the default, is a surrogate**: Gauss-Hermite [@golub1969] over a
lognormal clock matched on its moments and the exact mixer, with no seed and no sampling noise. θ\*
fitted under it is used by pricers that walk; the gap is on the Bootstrapping page. Forward-start
rows walk either way. Quadrature refuses `Cap_A` and a second residual bucket, its moments needing
an uncapped exponential and one mixer. The walk's draws are fixed for the fit, so `Paths` is the
noise floor under θ\*, not a confidence interval.

**Where it runs.** A fit that walks nothing (quadrature, no forward block) runs on the host, its
tensors too small for a card; a walk runs on the job's device, its draws made on the host so a seed names one draw on any device.
`Calculation.factor_leaf` moves θ onto the calculation's device before splicing it, so `dV/dq`
crosses the boundary. **Held by**
`test_logvar2fj_json.py::test_the_quote_switch_writes_the_same_factor_and_publishes_its_leaves`.

**A previous factor is the warm start**: every parameter and the ξ strip seed at it and the fit is
the joint polish alone, the staged search existing to find a basin a written factor names.
`Fit_Mode: Bootstrap` stays cold, having no joint stage; a factor carrying a retired key or the other
`Residual_Law` refuses by name. **Held by** `test_logvar2fj_json.py::test_a_previous_factor_makes_the_fit_the_polish_alone`,
`test_logvar2fj_json.py::test_a_warm_start_off_the_other_residual_law_refuses_by_name`.

### Priors and guards {#logvar2fj-priors}

Whatever the ladder cannot identify
([what the vanillas identify](../theory/stochastic_volatility.md#what-the-vanillas-identify)) carries
a prior row.

| choice | why |
| --- | --- |
| a prior is a soft row, never a pin | a pin leaves θ\*, the Jacobian and the quote contraction |
| one standard error costs one quote missing by one vol point | scaled otherwise, a row is a pin or nothing |
| the leverage prior is two rows, `ρ_s` and `ρ_s σ_s`, on the engine's axis | one row alone is evaded through the other factor |
| a prior reads the block, else a history, else the class default | a desk's view, then an estimate, then a convention |
| the residual rows are on `log α` and `β/α` | a row on β alone is obeyed by running α to its ceiling |
| a history counts only where its SE beats the class spread | otherwise it says less than the default |
| a history number is clipped into the fit's box | outside it, it crushes the derived boxes |
| `Residual_Shape_Floor` softly floors `α·δ_A` | nothing else bounds α below |
| the `ρ_s` box is symmetric with priors on | which way a smile leans is the prior's statement |
| the slow pair is pinned without a wing at `Slow_Horizon`, signed by the fitted `ρ_s` | the slow skew must not oppose the fast |
| `Model_Priors: Off` is the vanilla-only fit | the switch a bit-identity gate runs under |

**Held by** `test_logvar2fj_json.py::test_two_class_tables_disagreeing_in_sign_refuse`,
`test_logvar2fj_json.py::test_a_withdrawn_declaration_refuses_by_name`,
`test_logvar2fj_json.py::test_a_priors_off_fit_is_bit_identical_to_its_banked_reading`.

**Guards are readings, never re-solves.** `On_Guard` names every guard θ\* sits on (a box edge, a
share near its bound, a prior row past `LVFit.PRIOR_RATIO` quote rows, a coordinate the quotes are
silent on) and is collected into the mark's `Stats`, so a price carries the flag. No guard moves
θ\*, which keeps the quote contraction where it was taken. **Held by**
`test_logvar2fj_json.py::test_a_fit_forced_onto_a_box_names_it_and_the_mark_reports_it`,
`test_logvar2fj_json.py::test_a_ladder_with_no_wings_reads_its_skew_as_silent`,
`test_logvar2fj_json.py::test_the_stationary_guard_refuses_the_same_surface_by_name`.

### Crosses and the axis {#logvar2fj-crosses}

An `FxRate` is a currency priced in the base, so a desk's leverage view can carry the other sign
here; `security_map.leverage_prior` turns the seed's view onto the axis the book fits. A cross is
fitted on its alphabetically later token priced in the earlier, however the deal or surface is
written, so one pair has one law ([the pair's key](structures.md#accrual)). The block's `Priced_In`
is a resolved `FxRate` reference and the spot is `FxRate.<Underlying> / FxRate.<Priced_In>`, blank
being the base's rate of one, so the ratio is the only spelling. Each leg's carry is the curve its
own `FxRate` names, so the fit's forward is the deal's. The block is filed `<later>.<earlier>`, never
as a two-token `FxRate`, which discovery reads as a spot plus a basis tail. **Held by**
`test_cross_spot_model.py::test_the_fit_is_the_same_on_either_base`,
`test_cross_spot_model.py::test_the_axis_is_the_pairs_not_the_surfaces_spelling`,
`test_cross_spot_model.py::test_the_seed_prior_arrives_on_the_fitted_axis`.

### The forward block and the reserve {#logvar2fj-forward}

The split between β and `ρ_s σ_s` is what a forward smile sees and an autocall reads.
`Forward_Smile_Source` is `None`, `Quotes` (traded forward-starts) or `Reference` (a reference
model's forward smiles), both off `Forward_Smiles`. A view fitted as a target is paid for in vanilla
fit, so `Prior` is withdrawn. A source enters at stage 3, which fits `(ρ_s, σ_s)`, and moves the
later buckets of `β(t)` and `ρ_s(t)`: a smile at T never reads a bucket past T while a forward smile
reads nothing else.

- **The targets are differences**, forward slope and butterfly less the market's spot ones. A ratio
  would divide by a spot quantity near zero on a symmetric smile and fit noise.
- **An unreachable tenor is dropped by name**: a row whose nearest rung is more than
  `Spot_Rung_Tolerance` of Δ away would difference against the wrong maturity.
- **A short Δ identifies the residual pair**: over a quarter the residual is nearly Gaussian.
  **Held by** `test_logvar2fj_json.py::test_the_forward_block_identifies_the_residual_pair_the_vanillas_leave_at_its_prior`.

**Where the forward smile is not quoted, the factor carries a reserve line**, `Skew_Gradient` and
`Stickiness_Band`, from which a deal reporting `Greeks: First` composes `Skew_Reserve`, the parameter
move being the minimum-norm one, per deal and for the set, the set's being its deals contracted
rather than summed. The fit's own reported rows sit on grid block ends, since a window end off one
splits a mixer and moves θ\*; the factor's line is read again at the declared window off θ\*.
**Held by** `test_logvar2fj_json.py::test_the_skew_reserve_is_the_minimum_norm_contraction`,
`test_logvar2fj_json.py::test_the_reserve_is_per_deal_beside_the_portfolio`.

### Quote-space risk {#logvar2fj-quote-risk}

`Quote_Sensitivity: Yes` is the Gauss-Newton contraction at the stationary point through
`utils.LeastSquaresSolve`, over the coordinates no box holds, on the column-scaled Jacobian cut at
`Jacobian_Rcond` [@hansen1987]. Each market premium carries its quote as a splice worth zero
forward, so the switch cannot move a fitted digit. It refuses where a stage stopped off stationarity
and under `Fit_Mode: Bootstrap`, whose θ\* is a stationary point of no single objective
([the stationarity contract](quote_sensitivities.md#the-stationarity-contract)). **Held by**
`test_logvar2fj_json.py::test_the_quote_switch_writes_the_same_factor_and_publishes_its_leaves`.

### The quanto drift is inside the walk {#logvar2fj-quanto}

The GBM arm applies a quanto as one carry off an implied ATM vol; a walking kit has none, so it is
handed the correlation and the walk adds `−q_k √V_k` a day, `q_k = ρ_q σ_FX,k √δ_k`, the drift
following the variance path ([theory](../theory/stochastic_volatility.md#equity-and-fx)). `ρ_q` is the marked equity/FX
correlation read as a total-return correlation; applied to the day's total sd it already is the
framework's correlation given the mixer, so it is not divided by the mixer's dilution. Only the
autocall passes the loading; `Compo` is refused, and so is a quanto naming no correlation. A
correlation mints no leaf, so its delta is a common-random-number bump (`Correlation_Bump`).
**Held by** `test_logvar2fj_json.py::test_the_quanto_loading_is_off_at_zero_correlation`,
`test_logvar2fj_json.py::test_the_quanto_gbm_limit_reproduces_the_gbm_quanto_deal`.

### What the quotes cannot say {#what-the-quotes-cannot-say}

Limits of the market data available, not of the engine:

- **A tight declared standard error is a pin**; `On_Guard` flags it, and `Leverage_Prior_SE` is
  where a desk states its own spread.
- **The skew-share prior's width is asserted, not measured**: an uncontaminated history would measure
  it, and an index history is contaminated by its leverage.
- **Vanillas alone do not pin α**: on the fixture's model-owned ladder the fit lands on the class
  prior, and one-month `Reference` rows recover the world's α. A vendor chain quotes no forward-start.
- **A closes-only archive attenuates the estimator's cross-name shock correlations**, a smoothed
  shock being a functional of one name's noisy closes.
- **A ladder shorter than `Slow_Horizon` cannot reach the flat limit**, the pinned slow pair
  injecting skew a flat surface does not want.
- **`Kappa_L` and `Kappa_S` are declared, never fitted**; their influence on a mark is unmeasured.
- **The default vanilla pricer is a surrogate**, and the forward-start rows carry the walk's
  small-sample bias at `Paths` ([limitations](../limitations.md#the-calibration)).

## `FXVolPrices` — a smile quoted in delta {#fxvolprices}

An FX smile ticks in as ATM, risk-reversal and butterfly quotes by delta, and the pricers read a
log-moneyness surface; this family converts one into the other
([specification](../bootstrapping/fx_and_equity.md#fxvolsurfaceparameters)).

**The grid is pinned, because refinement is compile-time work.** The solve refines a log-moneyness
grid until it resolves the smile to `Grid_Tolerance`, so the grid is a function of the quotes;
refined at factor construction, every vol tick would move a node, a new plan and a recompile. So the
bootstrap refines once and writes the grid with the factor, and a re-bootstrap finding its own
surface at the same expiries and tolerance reuses it (`pinned_grid`), moving only the vols.
`Factor2D.solves_delta_surface` is what lets a solved surface keep its grid. Below 1e-8 the
refinement does not terminate, so that bound is `Grid_Tolerance`'s declared domain, not a dial.

**Each convention offers one value**, because the solve implements one: a premium-adjusted forward
delta and the delta-neutral straddle ATM [@reiswich2010]. `BF` is the smile strangle; a broker's
market strangle is a different number [@clark2011]. "Malz" is a label: the wings are piecewise linear
in delta, not Malz's quadratic [@malz1997].

**The ATM row is the surface's ATM vol.** `Factor2D.malz_skew` places it at the delta-neutral
straddle strike, so the written ATM vol is the quoted number, and `GBMAssetPriceTSModelPrices` takes
those rows as its ATM column wherever this family wrote the surface, evidenced by the fingerprint
`pinned_grid` reads, never by the name ([the two sources](quote_sensitivities.md#the-atm-column)).
`Quote_Sensitivity` leaves the surface connected to its quotes, bit-identical either way
([the delta solve](quote_sensitivities.md#the-delta-solve)).

!!! warning "One ATM quote, two families, two partial derivatives"
    This block's ATM row is also `GBMAssetPriceTSModelPrices`' ATM column, so with **both** blocks
    asking for `Quote_Sensitivity` one JSON number reaches a valuation through two maps and its
    `dV/dq` arrives split over two `quote_leaves` entries under the *same* descriptor. Each half is
    correct and neither is the total: a consumer must group by descriptor across blocks and **sum**
    ([the attachment](quote_sensitivities.md#the-attachment)).

**A point may carry a two-way, and the bootstrap never reads it.** Every read here is of
`Quoted_Market_Value`, so the surface and every mark are the mid's; the one reader of
`Quoted_Bid`/`Quoted_Ask` is [`derivus.structures`](structures.md#two-sided), which charges each leg
the pillar's half-spread times its vega to that pillar. *The spread is the quote's; the mid is the
book's.* **Held by** `test_service.py::test_a_two_way_ticks_beside_the_mid_and_a_moved_pillar_still_refuses`,
`test_structures.py::test_a_book_with_no_two_way_quotes_exactly_as_it_always_did`.

**Timestamps are stored, never priced**; what counts as too old is the consumer's policy.
`config.CustomJsonEncoder` writes a midnight stamp as a date and any other in ISO form, so intraday
snapshots survive a save.

## Open

The surrogate vanilla pricer ([limitations](../limitations.md#the-calibration)); the outright that
cannot carry a quote derivative ([limitations](../limitations.md#the-engine)); whether a missing
`Prices` refuses in process, the nominal leverage weight, the correlation as a leaf
([desk decisions](../limitations.md#decisions-waiting-on-the-desk)).
