# Structures

A structure is a sales product — a collar, a forward extra, a TARF — declared as data in
`derivus/structures.py`, and `structures.quote` is the runner that prices one against the book. The
finance lives in the declarations because the model driving the [MCP binding](mcp.md) may be any
LLM, and finance left in a model's head sometimes does not happen: the model turns a sentence into
named parameters, and the registry and the runner do the rest.

| where | what |
| --- | --- |
| `structures.py` | the declarations (`Structure` subclasses) and the runner (`quote`) |
| `schema.emit_structures` | publishes `mapping['Structure']`, read by `GET /schema` and `describe_structure` |
| `service.py` | `StructureJob`, `POST /book/structure` (quote), `POST /book/quote` (accept) |
| `quote_sheet.py` | the `.xlsx` ticket, the optional `quote` extra |

Tests named below are in `tests/test_structures.py` unless another module is given. Each is a real
document priced through the real pipeline against a financial identity; a rule marked **held by**
goes red under that test if it is broken.

## A structure is a declaration

A structure subclasses `Structure`. Its class name is the registry key (`Structure.__subclasses__()`,
so declaring one registers it), and it declares these — a subclass with no `vernacular` raises at
import — plus, for one solving a [ticket parameter](#parameter-solve), the classmethods that answer
it:

| attribute | what |
| --- | --- |
| `vernacular` | the sales names, comma-separated; `describe_structure` matches them as well as the class name, so "cylinder" lands on `ZeroCostCollar` |
| `fields` | the parameters, as `schema.F`. Shared ones come from the base: `Structure.QUOTED` (`pair`, `expiry`, `notional`, `notional_currency`), the two `SELECTORS`, and `Structure.strike(name, description)` for a level in market terms |
| `legs` | named `Leg(role, deal_type, pinned, slots)`: a declared instrument type, the partial deal block the structure fixes, and `{deal field: parameter}` for the rest. A leg never restates an instrument's fields — the `Instrument` declarations are its schema, as with a [market-price quote](market_prices.md#a-quote), so the two cannot drift. `Leg.vanilla` is a European `FXOptionDeal` |
| `variations` | in place of `legs`, for a structure dealt [more than one way](#variations) |
| `recipe` | ordered steps: `Price(role)`, and `Solve(role, field, target)` where `target` is a number or a `Premium` combination of legs already priced (`-Premium('protection')`, `Premium('a') + Premium('b')`); targeting a leg not yet priced refuses as a mis-ordered recipe. `SolveUnknown(target)` instead solves the one ticket parameter the structure finds blank to the parameter `target` names |

The roster:

| structure | legs | the recipe solves | held by |
| --- | --- | --- | --- |
| `Straddle`, `Strangle` | two bought vanillas | nothing — both strikes are the client's | `test_a_straddle_is_exactly_its_two_legs` |
| `ZeroCostCollar` | protection + financing vanilla | the financing strike, to `-Premium('protection')` | `test_a_zero_cost_collar_costs_nothing` |
| `Seagull` | protection + participation + financing | the financing strike, to the negative of the other two | `test_a_seagull_nets_to_zero` |
| `ForwardExtra` | protection vanilla + sold knock-in (`FXBarrierOption`) at the same strike | the barrier, to `-Premium('protection')` | `test_a_forward_extra_costs_nothing_and_solves_its_barrier` |
| `TargetRedemptionForward` | one `FXTARFOptionDeal` strip | the strike, to zero | `test_an_accrual_strip_costs_nothing_and_strikes_better_than_the_forward` |
| `Accumulator` | one `FXAccumulatorOptionDeal` strip | the strike, to zero | the same |
| `RangeAccrual` | one `FXRangeAccrualDeal` per stated tier (one to four), each paying the increment over the tier outside it | the one blank tier cell, or the tiers' width, to `premium`; nothing, quoting the premium, where every cell is stated | `test_a_wedding_cake_is_its_nested_range_accruals_with_incremental_payoffs`, `test_the_one_blank_tier_parameter_solves_to_the_stated_premium` |

`test_the_registry_publishes_exactly_the_declared_structures` holds the published store to the
classes and every leg type to a declared instrument.

**To add a structure:** subclass `Structure` with the attributes above, using only the leg types the
runner furnishes (`FXOptionDeal`, `FXBarrierOption`, `FXTARFOptionDeal`, `FXAccumulatorOptionDeal`,
`FXRangeAccrualDeal` — `materialize` refuses any other, since it is what fills their shared fields),
derive a mirror form
with `reflected` where the product is dealt both ways, and add the identity it must satisfy to
`tests/test_structures.py` beside the ones in the table.

## Variations {#variations}

Most structures are two different bookings depending on the client's side, not two prices of one
trade. A forward extra for an exporter floors the pair (a bought put funded by a sold up-and-in
call); for an importer it caps the pair with the mirror (a bought call funded by a sold down-and-in
put). A `Variation(buys, fields, legs)` is one form: `buys` is the side of the pair the client buys
(`'base'` or `'quote'`), `fields` the parameters only it takes. Only the straddle and strangle have
one form.

`reflected(variation, rename, fields)` derives the mirror form so it is never written twice: every
leg's `Option_Type` and `Barrier_Type` direction flip (`schema.flipped`, read off the leg type's own
declaration), `rename` maps the form's parameters to their mirror names (`{'floor': 'cap'}`), and
`fields` is declared by hand because the prose of a level depends on the client's side. Two things
do not flip: In/Out (what the payoff does on touch, the same to either client) and `Buy_Sell` (the
client's own side on both sheets — `mirror` stays the one place client paper becomes the bank's
position). **Held by** `test_a_reflected_variation_is_the_legs_written_out_by_hand`,
`test_each_variation_is_the_structure_it_says_it_is`, `test_every_variation_books_the_deals_it_declares`.

`variation_for` is one selection rule for every structure: the variation is the unique one
consistent with everything the ticket states. `buy_currency` / `sell_currency` must be sides of the
pair and fix which side the client buys; a stated parameter that is some variation's own admits only
the variations declaring it. None consistent refuses naming the contradiction; more than one refuses
naming what to state — a structure quoted on the desk's guess of which way round it went is a
booking nobody agreed. "Stated" is one predicate, `stated()`, the falsy rule a `REQUIRED` field is
held to: a blank (`0.0`, `''`) is not a statement, because a front end round-trips unfilled fields
that way. **Held by**
`test_the_selection_rule_answers_every_way_a_ticket_can_state_it`, `test_a_blank_is_not_a_statement`.

The selectors are published with no value and `selector: 'required' | 'optional'`, computed in
`emit_structures`: required where the variations' own parameters cannot tell the forms apart (the
two strips both name one level, so only the direction can say which is meant), optional otherwise.
The outcome reports the `variation` priced and the client's two cashflows (`client`) read off the
variation rather than the ticket, so the booking and the account of it agree.

## The runner

`quote(document, structure_name, params, spot_source, netting_set, margin, cache)` in order:

1. `check_netting_set` — a quote is for a client, and a client is a `NettingCollateralSet` (the CSA
   and the CVA projection live there), so a set the book does not carry refuses before anything is
   priced; then `margin_value`; then `declared()` completes the parameters from their declared
   defaults and refuses what the schema's judges say against them, as they say it of a deal: a
   missing `REQUIRED` one (`schema.validate_block`: a strike the client did not name is a strike
   nobody agreed), then a value that is not what its field declares or a parameter no form takes
   (`schema.stated_messages`: a misspelt gearing would quote at a default nobody agreed).
2. `materialize` turns each leg into a wire-form deal ([axes](#axes), [accruals](#accrual)).
3. `run_recipe` runs the steps at mid. Each step prices one leg alone (`alone`: a deep copy of the
   book with its deal tree emptied and the calculation set by `priced_job`) — a leg's base-valuation
   row does not depend on its siblings, and a lone deal compiles fast enough to sit inside a solve.
   `Solve` is `derivus.solve_deal_field`, [bracketed](#solves); `SolveUnknown` is the
   [parameter solve](#parameter-solve).
4. Where the book quotes a [two-way](#two-sided), each leg's vega is read and the charge levied; the
   [risk-impact step](#risk-impact) may scale it; a second, seeded pass re-solves with the charge.
5. `compose` builds the bookable `StructuredDeal`; the outcome carries the legs, `net`, `net_mid`,
   `edge`, `charged_on`, `spot`, `risk`, `notes`, `valuation_configuration` and the `deal`.

A quote prices simulated legs at the inner path count `BaseValuation` declares or the book's,
whichever is larger (`priced_job`): a booked strip is later marked by an ordinary base valuation, and
the quote and the mark have to be one estimator. A book stating fewer gets a `notes` entry
(`thin_paths`).

### Market axis and engine axis {#axes}

Parameters arrive in market terms (USDZAR 15.50 is ZAR per USD) and `materialize` converts once, so
no structure declares an orientation and none can get it wrong. An `FXOptionDeal` is an option on
`Underlying_Currency` settled in `Currency`, and the runner makes `notional_currency` the underlying —
which is exactly the discriminator the conversion needs. A leg type declaring no `Underlying_Amount`
(`FXRangeAccrualDeal`) is instead PAID in its notional, so `notional_currency` is its `Currency`
and the pair's other side its underlying, the axis read off the declaration and never the type's
name; the shared block carries only what the leg's type declares. When the leg's underlying is the
pair's quote currency the deal's axis is the reciprocal of the market's:

- `Strike_Price` and `Barrier_Price` become `1/K`;
- the option sense flips (a market call is an engine put) and a barrier's direction flips with the
  level (`Up_And_In` becomes `Down_And_In`); In/Out never moves;
- a range's bounds cross and SWAP — the reciprocal of a floor is a cap, so `Lower_Barrier` is
  `1/cap` and `Upper_Barrier` `1/floor`;
- when the underlying is the base currency nothing converts.

`Materialized.to_market` reports solved levels back in market terms. The book's surface may be
stored either way round (`FXVol.USD.ZAR` or `FXVol.ZAR.USD`); the legs name it as the book carries
it. **Held by** `test_a_market_strike_reaches_the_engine_axis`,
`test_the_same_trade_quotes_the_same_from_either_side_of_the_pair`,
`test_a_forward_extra_quotes_the_same_from_either_side_of_the_pair`,
`test_a_ticket_quotes_alike_whichever_way_round_the_book_carries_its_surface`,
`test_a_range_accrual_on_the_base_currency_crosses_its_levels_and_swaps_them`.

### Accrual legs {#accrual}

A leg whose type observes a fixing table (`accrues`, off `schema.OBSERVES`) is a strip, furnished by
`furnish_accrual`:

- **Schedule.** `fixing_grid` writes `[[fixing, settlement, 0.0], ...]` into the table the type
  observes, the observed column 0.0 because a quote is struck today. Each fixing is
  `base + n × fixing_frequency`, never a step off the previous date (31 Jan + 1M + 1M is 28 Mar),
  settling `Structure.fixing_lag` days later, from a base normalised to midnight (a time on
  `Base_Date` would drop the final fixing). A tenor holding no fixing refuses, and so does one the
  frequency does not divide, which would end the strip short of the tenor the ticket says.
  **Held by** `test_a_strip_ends_on_its_own_expiry_or_refuses`.
- **Expiry.** A TARF's `Expiry_Date` is its last settlement — a strip is not over until its last
  cashflow lands; `FXAccumulatorOptionDeal` declares no such field, so its block carries none. A
  type declaring a `Settlement_Date` (`FXRangeAccrualDeal`) pays the whole strip once, on the last
  settlement, its fixings written `[fixing, None]` since none has printed.
- **Notionals.** `Underlying_Amount` is the notional per fixing; `LeverageNotional`, on a type
  declaring it, is `leverage × notional`. Neither converts: the notional is in its own currency and
  gearing is a ratio.
- **Ranges.** A type declaring `Lower_Barrier` and `Upper_Barrier` takes its tier's bounds
  (`RangeAccrual.tier`, innermost first, crossed as [above](#axes)) and a `Payoff` per fixing of
  `notional × increment / 100 × T / N`: the tier's coupon less the next wider one's, a rate a year
  earned in full where every fixing lands, `T` the ACT/365 years to the last fixing and `N` the
  fixings. **Held by** `test_a_wedding_cake_is_its_nested_range_accruals_with_incremental_payoffs`.
- **Axis.** An accumulator's knock-out is a level and crosses like a barrier. A TARF's target is a
  sum of differences, and `1/S − 1/K` is not the reciprocal of `S − K`, so no reciprocal number
  means the same cap: a TARF on the quote currency refuses by name and `InvertedTarget` is always
  `False` (that flag is a different product — the whole fixing on the reciprocal axis). **Held by**
  `test_an_accumulator_crosses_both_axes_and_a_tarf_refuses_the_second`.
- **Model.** A type whose `spot_models` do not admit `Structure.spot_model` pins nothing and has
  no note. Otherwise `spot_model` pins `Structure.spot_model` (`LogVar2FJ`) for the deal type in the copy's
  `Valuation Configuration` where the book carries that pair's calibration, and otherwise leaves the
  leg on GBM with a `note` naming the missing factor and `/book/model`. The check is made here
  because a switch whose factor is absent makes the engine skip the deal, and the quote would come
  back priced at nothing. A book that already declares a `SpotModel` for the type keeps it, and the
  check is made on that family, or a note would disagree with the number beside it. **Held by**
  `test_the_absence_note_names_the_factor_the_book_would_need`,
  `test_a_tarf_on_a_fitted_pair_stops_riding_gbm`.

**The pair's key** is `utils.spot_model_currency`, the one rule the engine's lookup, this check,
discovery and the calibration all call — a second keying lets a book that calibrated one pair price
another off it with no note. It is the non-base token for a pair with a base leg
(`LogVar2FJModelParameters.ZAR` for USDZAR on a USD book), and for a cross the alphabetically later
token priced in the earlier (`…ZAR.EUR` for EURZAR), however the deal or the surface is written, so
one pair has one law. The base currency is the market's (`structures.base_currency` over
`structures.market_of`, the file a book names with its explicit block merged over it), the engine's
own where nothing states one, so the key a quote pins is the one the engine's lookup takes (**held
by** `test_a_book_that_declares_no_base_currency_quotes_on_the_one_the_engine_prices_in`). A
base-leg pair is fitted
on its `FxRate`'s own axis, so two books with different base currencies fit reciprocal laws of one
pair; only a cross is canonical.

**The reciprocal axis.** A deal whose underlying is the currency the law is priced in pays on `1/S`.
`instruments.spot_model_reciprocal_axis` decides that and sets `Invert_Spot`; it is an allow-list
holding `LogVar2FJ` alone, so a model added without the measure change refuses by name rather than
pricing a fit on an axis nobody fitted. The change of numeraire is on the
[theory page](../theory/stochastic_volatility.md#equity-and-fx). **Held by**
`test_the_accumulator_solves_one_strike_from_either_axis_under_the_model`.

**The pin books with the trade.** The quote's copy dies with the answer, so it reports what it
pinned under `valuation_configuration`, and `/book/quote` merges that into the book in the same edit
that splices the deal (`pin_models`) — otherwise a leg dealt under LogVar2FJ re-marks as a lognormal
at the next valuation and nothing says so. A pin whose factor is no longer on the book refuses.
**Held by** `test_service.py::test_a_leg_quoted_under_a_model_books_into_a_book_that_marks_it`.

### Refusals before anything prices

A number a client could be handed for a trade nobody can book is worse than a refusal, so
`materialize`, `declared` and `quote` refuse by name: a missing required or unknown parameter, or
one that is not what it declares; a pair the book quotes no surface for (naming those it does —
otherwise every leg is dropped at load); an expiry that does not parse or lands on or before the
base date (a zero-day option); a non-positive notional (premiums with the sign reversed); a stated
barrier on or through the spot on its own side (a level on the spot is through already, and a
decumulator knocking out above the market solves an attractive rate for a strip that is dead where
it stands); a TARF on the quote currency; a strip whose frequency does not divide its tenor; an
unknown netting set; a margin in a currency with no `FxRate`; and a margin on a recipe solving more
than one coordinate. **Held by**
`test_what_a_variation_takes_is_required_and_what_no_form_takes_refuses`,
`test_what_a_client_cannot_be_quoted_refuses_before_a_leg_is_priced`,
`test_an_unparsed_tenor_refuses_rather_than_expiring_today`,
`test_a_stated_level_on_the_dead_side_of_its_own_direction_refuses`,
`test_service.py::test_a_quote_for_a_client_the_book_never_opened_refuses_at_the_ask`.

### Solves are bracketed {#solves}

`run_solve` brackets around the market spot, crossed to the engine axis and sorted, and `brentq`
owns the root — a secant seeded at the spot lands in a strike's flat region on a small engine-axis
strike:

| field | bracket (× spot) | why |
| --- | --- | --- |
| a strike | `Solve.STRIKE_BRACKET` (0.25, 4.0) | a vanilla is monotone in its strike |
| an accrual strike | `Solve.ACCRUAL_BRACKET` (0.5, 2.0) | a strip is flat once every fixing redeems, and a surface quoted over a narrow moneyness range prices NaN far outside it |
| a `Down_*` barrier | (0.25, 0.9999) | a knock-in is monotone in its barrier, on its own side, never on the spot |
| an `Up_*` barrier | (1.0001, 4.0) | the same |
| anything else | a secant from its current value | exact in two pricings for an affine field |

A re-solve seeded on an earlier root first tries `±Solve.SEED_BRACKET` (2%) around it — a charge moves
the coordinate by about a spread's width — and falls back to the full bracket where that does not
straddle, so the answer is the full bracket's either way. No root in the bracket refuses with the
economics stated ("cannot be financed at these parameters"). **Held by**
`test_a_seeded_solve_costs_less_and_a_missed_seed_still_answers_the_full_bracket`.

### The parameter solve {#parameter-solve}

`SolveUnknown(target)` moves a TICKET parameter rather than a deal field: a cake's coupon enters
two legs (its own increment and the one inside it) and its width enters no field at all. The
structure answers `unknown(params, spot)` — `(path, bracket)`, or `(None, None)` — and
`completed(params, path, value, forward)`, and the runner knows nothing of tiers: `brentq` over the
bracket, every leg rebuilt from the completed ticket and priced together (`run_prices`, one base
valuation, a leg's row not depending on its siblings), the residual the legs' sum less the stated
`premium` (crossed from `notional_currency` as a margin is) less the charge. A charged pass seeds
`±Solve.SEED_BRACKET` around the first root and falls back to the full bracket. The solved value
comes back written into `params` in market terms, so the completed ticket is what the desk re-runs;
`charged_on` and each leg's `solved` name the path (`'tiers[2].cap'`, `'width'`). `Structure.dealt`
says which declared legs a ticket deals — every one, or for `RangeAccrual` one per stated tier — so
the leg count follows the ticket.

`RangeAccrual`'s rule: tiers sort by width where every bound is stated, else by coupon, and refuse
by name where they are not a cake (more than four, a negative cell, a range admitting nothing, a
band not strictly inside the next wider, a coupon not above the next wider's). Then

| the ticket states | the unknown | bracket |
| --- | --- | --- |
| every cell, no `premium` | none: the legs are priced and `net_mid` is the premium, `charged_on` `'premium'` | — |
| a `premium` and one blank cell | that cell | a coupon between its neighbours' coupons (else 0 and 100); a floor between the next wider floor (else `0.25 ×` spot) and the next narrower (else its own cap); a cap the mirror — each opened by `OPEN` (1e-9 of the span), an end being a tier the engine refuses |
| a `premium` and coupons alone | `width` `w`: the k-th innermost tier at `F e^{−kw}` to `F e^{kw}`, `F` the engine's forward to the expiry (`forward_rate`: one valuation of a unit of each currency received that day, priced only when asked for) | `(1e-4, log 4)` |

Two blanks, a blank beside a blank premium, a tier stating no bound beside one that does, and a
ticket stating every cell and a premium each refuse by name; no root refuses with the economics
said, as `run_solve` does. The value is monotone in each parameter, so a root is unique where it
exists. **Held by** `test_the_one_blank_tier_parameter_solves_to_the_stated_premium`,
`test_coupons_alone_solve_the_widest_tiers_centred_on_the_forward`.

## The two-way charge {#two-sided}

A desk quotes at a spread but its book marks at the mid, and both hold because the two numbers live
in different places: every leg prices at the mid, so what the finished legs are worth is what the
trade marks at once booked, and the market's spread is a charge on the coordinate the recipe solves.
The book's `FXVol` surface is bootstrapped from `Quoted_Market_Value` alone; the `FXVolPrices` block
may carry `Quoted_Bid`/`Quoted_Ask` per pillar ([Market Prices](market_prices.md#fxvolprices)), and
the runner is that data's only reader.

- `quote_two_way` reads each used pillar's half-spread `max(0, (ask − bid)/2)` — a crossed print
  reads zero-wide, a desk never paying a client for a broken print — keyed by
  `bootstrappers.FXVol.descriptor`, the name its `dV/dq` leaf is published under. A pillar is a
  bucket; nothing is interpolated.
- `leg_vega` runs one first-order greeks valuation of the leg alone, on the client's paper, with
  `Quote_Sensitivity` on the `FXVolPrices` block (`vol_risk`). A leg walking a fitted spot model
  reads no surface and publishes no quote leaves, so it is read on a copy with its spot model dropped
  (`without_spot_model`) — the lognormal reading at the same terms, the vega a desk would hedge in —
  and `spread_source` says `'surface'` or `'lognormal reading'`. Without that substitution a fitted
  strip would capture nothing.
- A leg's charge is `Σ_pillars |vega_p| × half_p`. The absolute value is the rule: a pillar is dealt
  on the side the risk puts it, not the side the leg's `Buy_Sell` says — a geared strip booked `Buy`
  is short vol at every ATM pillar, and signing by the label quotes the desk paying the client. The
  structure's charge is the sum over legs; legs are not netted against each other first (a desk
  [decision](../limitations.md#decisions-waiting-on-the-desk)).
- The charge moves the solved coordinate in a second pass seeded on the first. A recipe that solves
  nothing (straddle, strangle) charges the premium instead; `charged_on` names which.

Outcome fields, all in the client's sign convention: `net_mid` (the legs at mid), `net` (plus
everything charged), `edge` (the two-way charge alone, non-negative by construction), and per leg
`spread_charge`, `spread_source` and `spread` rows `{pillar, vega, half, cost}`, where `half` is always
the market's half-spread and `cost` the money charged. A leg no reading reaches carries `null`, never
`0.0`, since a zero reads as a spread measured and found to be nothing. A book with no two-way runs
no greeks at all and quotes the mid, with `spread_note` saying why.

**Held by** `test_a_two_sided_quote_charges_the_spread_and_leaves_the_book_at_mid`,
`test_each_pillars_charge_is_what_repricing_that_pillar_costs` (each pillar's cost against a central
difference of that pillar's quote, re-bootstrapped), `test_a_strip_pays_the_spread_rather_than_being_paid_it`,
`test_a_puts_risk_reversal_is_charged_on_its_own_side`, `test_a_model_priced_strip_is_charged_off_its_lognormal_reading`,
`test_a_leg_the_two_way_cannot_reach_is_charged_nothing_and_says_so`,
`test_every_form_charges_a_non_negative_edge_that_is_its_legs_own`,
`test_a_book_with_no_two_way_quotes_exactly_as_it_always_did`.

## The sales margin {#margin}

`margin` is `{'amount': 50000.0, 'currency': 'ZAR'}`, money in any currency, as a client agrees it.
`margin_value` crosses it to the calculation's currency (else the base) on the document's own
`FxRate.<ccy>.Spot` ratio, so the margin and the legs see one market. It is charged on the same
coordinate as the two-way — a financing leg's target becomes the premiums it finances plus the
charge, a strip targets minus it, a solved ticket parameter makes the legs worth the stated
premium less it, and a ticket solving nothing charges the premium — so one convention covers every
structure. On the client's paper `net` reads the margin back negative; the
booked mirror marks the bank at plus the margin and the edge. `compose` records it as `Sales_Margin`
and `Sales_Margin_Currency` (declared on the shared `Admin` group, a record nothing prices off).
`edge` excludes it, so a policy tightening the market's spread never shaves an agreed margin.
**Held by** `test_a_collar_at_a_margin_is_minus_it_on_paper_and_plus_it_on_the_book` (both ends,
since one end alone cannot catch a sign), `test_a_margin_and_a_two_way_compose_on_one_coordinate`,
`test_a_structure_that_solves_nothing_charges_its_premium`,
`test_coupons_alone_solve_the_widest_tiers_centred_on_the_forward`,
`test_a_margin_survives_a_policy_that_tightens_the_two_way`, `test_a_margin_refuses_where_it_cannot_be_valued`.

## Risk-impact pricing {#risk-impact}

The two-way is what the market charges a trade; this step charges what hedging the residual the
trade leaves on the book would cost, at the market's own two-way — so a trade that nets the book down
quotes tighter and one that piles risk on quotes at the full spread. It runs only where the job
declares a `Quote Policy` section beside `Calculation` (`read_policy`; not in `ExplicitMarketData`,
since a mandate is not market data). Its absence is the off switch, and `risk.scale` is then `null`
rather than `1.0`, so a consumer can tell "did not run" from "ran and decided nothing".

| field (`Structure.POLICY`) | default | meaning |
| --- | --- | --- |
| `participation` | 0.5 | the share of a measured saving passed to the client |
| `floor` | `'mid'` | where tightening stops; the only value accepted |
| `scope` | `'vol'` | the book the residual is measured in; the only value accepted |
| `bucket_limit` | none | an `abs(risk after)` per bucket past which nothing is tightened |
| `min_ticket_bp` | 0.0 | a floor under the edge, in bp of notional, crossed to the report currency |
| `firm_seconds` | 600 | how long a quote stays acceptable; read by `/book/quote` |

- **Measurement.** The mid candidate is composed and `mirror`ed — the seam the booking uses, so the
  risk measured and the trade booked cannot disagree by a sign — and the book's vol risk is read with
  it (`vol_risk`) and without it (`book_risk`, cached on the served book's `vol_risk_cache` by
  content hash, since the book alone does not depend on what is quoted). `risk_document` sets
  `Greeks: 'First'` and `Quote_Sensitivity: 'Yes'` on the `FXVolPrices` block, and the copy is
  bootstrapped and priced in one `Context`, so the [attachment](quote_sensitivities.md#the-attachment)
  leaves carry `dV/dq`; descriptors are summed across published blocks.
- **Cost.** `risk_buckets`: `Σ (|after| − |before|) × half` over quoted buckets; negative is a saving.
- **Scale.** `risk_scale`: `charge_effective = min(charge_full, max(min_ticket, max(0, charge_full −
  participation × saving)))`, `scale = charge_effective / charge_full`. The market spread is the
  ceiling (a risk-adding trade takes no surcharge), the mid is the floor, the minimum ticket is a
  floor under the tightening and never lifts a quote past the full spread, and a bucket past
  `bucket_limit` suspends tightening and is named.
- **One pass.** Vegas and buckets are read at the mid solution while the quote sits at the charged
  coordinate — a stated first-order approximation; a fixed point would cost a greeks run per leg per
  iterate.
- `scale` stays `null` with a note when there is no policy, no two-way, no readable leg, a
  non-positive charge, or no vol quote leaves.

**Held by** `test_a_book_with_no_quote_policy_quotes_exactly_as_it_always_did` (no policy and
`participation: 0` quote identically), `test_an_offset_quotes_tighter_than_a_repeat`,
`test_the_cap_and_the_floor` (both limits made to bind).

## The quote lifecycle

**Quote** (`POST /book/structure` → `StructureJob.run_job`). Queued at a base valuation's cost class
in the curiosity lane: a desk quotes many times and the record holds only the one a client accepts.

1. Stamp `quoted_at`; under a spine home, pin the book's `plan_hash`, `values_hash` and board age
   before anything changes the copy, since what an acceptance asks is whether the book has moved.
2. `patch_live_spot` writes the terminal's spot onto the job's own copy through `with_live_spots`
   (`crossed_spots`, the inverse of `engine_spot`), the moved `FxRate` block written into the copy's
   explicit block. Only the spot moves: a delta-quoted FX surface is sticky-delta,
   read at whatever spot stands, and the surface and curves belong to the tick cadence. Any failure
   falls back to the book's spot with the reason in `spot.note` — one request bounded by
   `SPOT_TIMEOUT_MS`, a failure that reached the terminal remembered for `SPOT_BACKOFF_SECONDS` so
   consecutive quotes skip it. `spot.value_market` is read back off the priced document, so the
   reported spot and the priced one cannot disagree.
3. `structures.quote`.
4. File the pending trade as `DV_HOME/tmp/<quote_id>.json` (the outcome, the composed deal,
   `quoted_at`, and under a home what the acceptance will file), with the sheet beside it when
   `xlsxwriter` is installed, else `files['sheet_note']` — a missing writer never refuses a quote.

Two ids, both hashing a clock because a quote is an act — two identical asks are two quotes, never
one coalesced result: the runner's `quote_id` (structure, parameters, netting set, margin, the
market carried) names the files; the service's `result_id` (book etag and the request) names the
queued job. **Held by** `test_a_quote_is_an_act_not_a_lookup`,
`test_a_live_cross_lands_exactly_where_engine_spot_reads_it_back`,
`test_service.py::test_a_quote_with_no_map_prices_on_the_book_and_names_the_spot_it_used`,
`test_spine_engine.py::test_a_quote_pins_the_book_before_the_live_spot_lands_on_its_copy`.

**Accept** (`POST /book/quote {quote_id, actor}`): reads the pending file, refuses a `portfolio`
(stated when quoting), answers `booked` for a quote already on the book, refuses a quote older than
`firm_seconds` where the book declares a policy, and books the `mirror` of the pending deal through
`deal_edit` — the same validate-before-write seam as `/book/deals` — under the quote's netting set,
merging its `valuation_configuration`. The file stays as the audit trail. Under a spine home the
acceptance also checks the plan and the board's age and reports the market's move; that half is on
[The Spine](spine.md). **Held by** `test_service.py`'s `test_a_quoted_collar_is_filed_pending_and_books_at_zero`,
`test_a_quote_for_a_client_books_under_the_clients_netting_set`,
`test_a_quote_is_firm_only_for_the_window_the_book_declares`.

**Composition.** `compose` puts the legs inside the deal block under `Children`, because a quote is
filed and hashed whole; `book_node` lifts them onto the node, which is the shape the engine walks
(`splice_deal`). A container spliced with its children still inside the block loads empty and prices
at zero, silently. **Held by** `test_a_composed_tarf_carries_an_exposure_profile` (a skipped or empty
strip has no spread to show).

## Writing a new gate

- **Use an identity, not a number.** Every gate above compares two routes to one value (a structure
  against its legs, two orientations, a margin from both ends, a charge against a re-bootstrapped
  difference), so it survives a model change that moves every price.
- **Comparing orientations.** An equivalent notional converts at the strike (`N` rand is `N/K`
  dollars), not the spot; at the money the two cannot be told apart.
- **Strip tolerances.** A strip's solved strike is a root over a Monte Carlo estimator, deterministic
  for a seed and converging only with the path count. Compare at the declared count; the gates allow
  `AXIS_TOLERANCE` (2e-4).
- **A book with no two-way and no policy must quote the mid exactly**, float for float. Any new
  layer gets the same gate, which is what lets it ship off by default.

## Open

The two-way on a fitted leg reads the lognormal vega, not the fit's
([limitations](../limitations.md#the-engine)); netting a package's legs before charging is a desk
decision; incremental XVA as the second risk-impact term and a ratio-solve primitive for
participating forwards are designed, not built ([roadmap](roadmap.md#designed-not-built)).
