# Structures

A structure is a sales product — a collar, a forward extra, a TARF — declared as data in
`derivus/structures.py`, and `structures.quote` is the runner that prices one against the book. The
finance lives in the declarations, so whichever model drives the [MCP binding](mcp.md) only turns a
sentence into named parameters; the registry and the runner do the rest.

| where | what |
| --- | --- |
| `structures.py` | the declarations (`Structure` subclasses) and the runner (`quote`) |
| `schema.emit_structures` | publishes `mapping['Structure']`, read by `GET /schema` and `describe_structure` |
| `service.py` | `StructureJob`, `POST /book/structure` (quote), `POST /book/quote` (accept) |
| `quote_sheet.py` | the `.xlsx` ticket, the optional `quote` extra |

## A structure is a declaration

A structure subclasses `Structure`. Its class name is the registry key (`Structure.__subclasses__()`,
so declaring one registers it), and it declares these and nothing else — a subclass with no
`vernacular` raises at import:

| attribute | what |
| --- | --- |
| `vernacular` | the sales names, comma-separated; `describe_structure` matches them as well as the class name |
| `fields` | the parameters, as `schema.F`. Shared ones come from the base: `Structure.QUOTED` (`pair`, `expiry`, `notional`, `notional_currency`), the two `SELECTORS`, and `Structure.strike(name, description)` for a level in market terms |
| `legs` | named `Leg(role, deal_type, pinned, slots)`: a declared instrument type, the partial deal block the structure fixes, and `{deal field: parameter}` for the rest. A leg never restates an instrument's fields — the `Instrument` declarations are its schema, as with a [market-price quote](market_prices.md#a-quote). `Leg.vanilla` is a European `FXOptionDeal` |
| `variations` | in place of `legs`, for a structure dealt [more than one way](#variations) |
| `recipe` | ordered steps: `Price(role)`, and `Solve(role, field, target)` where `target` is a number or a `Premium` combination of legs already priced (`-Premium('protection')`, `Premium('a') + Premium('b')`) |

The roster:

| structure | legs | the recipe solves |
| --- | --- | --- |
| `Straddle`, `Strangle` | two bought vanillas | nothing — both strikes are the client's |
| `ZeroCostCollar` | protection + financing vanilla | the financing strike, to `-Premium('protection')` |
| `Seagull` | protection + participation + financing | the financing strike, to the negative of the other two |
| `ForwardExtra` | protection vanilla + sold knock-in (`FXBarrierOption`) at the same strike | the barrier, to `-Premium('protection')` |
| `TargetRedemptionForward` | one `FXTARFOptionDeal` strip | the strike, to zero |
| `Accumulator` | one `FXAccumulatorOptionDeal` strip | the strike, to zero |

To add a structure: subclass `Structure` with the four attributes, using only furnished leg types
(`FXOptionDeal`, `FXBarrierOption`, `FXTARFOptionDeal`, `FXAccumulatorOptionDeal` — `materialize`
refuses any other), and add its identity gate to `tests/test_structures.py`.

## Variations {#variations}

Most structures are two different bookings depending on the client's side. A forward extra for an
exporter floors the pair (a bought put funded by a sold up-and-in call); for an importer it caps the
pair with the mirror (a bought call funded by a sold down-and-in put). A
`Variation(buys, fields, legs)` is one form: `buys` is the side of the pair the client buys
(`'base'` or `'quote'`), `fields` the parameters only it takes. Only the straddle and strangle have
one form.

`reflected(variation, rename, fields)` derives the mirror form: every leg's `Option_Type` and
`Barrier_Type` direction flip (`schema.flipped`, read off the leg type's own declaration), `rename`
maps the form's parameters to their mirror names (`{'floor': 'cap'}`), and `fields` is declared by
hand because the prose of a level depends on the client's side. Two things do not flip: In/Out (what
the payoff does on touch) and `Buy_Sell` (the client's own side on both sheets — `mirror` is the one
place client paper becomes the bank's position).

`variation_for` is one selection rule for every structure: the variation is the unique one
consistent with everything the ticket states. `buy_currency` / `sell_currency` must be sides of the
pair and fix which side the client buys; a stated parameter that is some variation's own admits only
the variations declaring it. None consistent refuses naming the contradiction; more than one refuses
naming what to state. "Stated" is one predicate, `stated()`: a blank (`0.0`, `''`) is not a
statement, because a front end round-trips unfilled fields that way.

The selectors are published with no value and `selector: 'required' | 'optional'`, computed in
`emit_structures`: required where the variations' own parameters cannot tell the forms apart (the two
strips), optional otherwise. The outcome reports the `variation` priced and the client's two
cashflows (`client`) read off it.

## The runner

`quote(document, structure_name, params, spot_source, netting_set, margin, cache)` in order:

1. `check_netting_set` (a quote is for a client, which is a `NettingCollateralSet`), `margin_value`,
   then `declared()` completes the parameters from their declared defaults and refuses a missing
   `REQUIRED` one or a parameter no form takes.
2. `materialize` turns each leg into a wire-form deal ([axes](#axes), [accruals](#accrual)).
3. `run_recipe` runs the steps at mid. Each step prices one leg alone (`alone`: a deep copy of the
   book with its deal tree emptied and the calculation set by `priced_job`); `Solve` is
   `derivus.solve_deal_field`, [bracketed](#solves).
4. Where the book quotes a [two-way](#two-sided), each leg's vega is read and the charge levied; the
   [risk-impact step](#risk-impact) may scale it; a second, seeded pass re-solves with the charge.
5. `compose` builds the bookable `StructuredDeal`; the outcome carries the legs, `net`, `net_mid`,
   `edge`, `charged_on`, `spot`, `risk`, `notes`, `valuation_configuration` and the `deal`.

A quote prices simulated legs at the inner path count `Base_Revaluation` declares or the book's,
whichever is larger (`priced_job`), because a booked strip is later marked by an ordinary base
valuation and the two must be one estimator; a book stating fewer gets a `notes` entry
(`thin_paths`).

### Market axis and engine axis {#axes}

Parameters arrive in market terms (USDZAR 15.50 is ZAR per USD) and `materialize` converts once. An
`FXOptionDeal` is an option on `Underlying_Currency` settled in `Currency`, and the runner makes
`notional_currency` the underlying. When that is the pair's quote currency the deal's axis is the
reciprocal of the market's:

- `Strike_Price` and `Barrier_Price` become `1/K`;
- the option sense flips (a market call is an engine put) and a barrier's direction flips with the
  level (`Up_And_In` becomes `Down_And_In`); In/Out never moves;
- when the notional is the base currency nothing converts.

No structure declares an orientation. A leg's `Materialized.to_market` reports solved levels back in
market terms. The book's surface may be stored either way round (`FXVol.USD.ZAR` or
`FXVol.ZAR.USD`); the legs name it as the book carries it.

### Accrual legs {#accrual}

A leg whose type observes a fixing table (`accrues`, off `schema.OBSERVES`) is a strip, furnished by
`furnish_accrual`:

- **Schedule.** `fixing_grid` writes `[[fixing, settlement, 0.0], ...]` into the table the type
  observes: each fixing at `base + n × fixing_frequency` (never a step off the previous date, which
  walks off month ends), settling `Structure.fixing_lag` days later, from a base normalised to
  midnight. A tenor holding no fixing, or one the frequency does not divide, refuses by name.
- **Expiry.** A TARF's `Expiry_Date` is its last settlement; `FXAccumulatorOptionDeal` declares no
  such field, so it is removed.
- **Notionals.** `Underlying_Amount` is the notional per fixing; `LeverageNotional` is
  `leverage × notional`.
- **Axis.** An accumulator's knock-out is a level and crosses like a barrier. A TARF's target is a
  sum of differences, which has no reading on the reciprocal axis, so a TARF on the quote currency
  refuses by name and `InvertedTarget` is always `False` (that flag is a different product: the
  whole fixing on the reciprocal axis).
- **Model.** `spot_model` pins `Structure.spot_model` (`LogVar2FJ`) for the deal type in the copy's
  `Valuation Configuration` where the book carries that pair's calibration, and otherwise leaves the
  leg on GBM with a `note` naming the missing factor and `/book/model`. A book that already declares
  a `SpotModel` for the type keeps it, and the check is made on that family. The pin is made here
  because a switch whose factor is absent makes the engine skip the deal.

**The pair's key** is `utils.spot_model_currency`, the one rule the engine's lookup, this check,
discovery and the calibration all call: the non-base token for a pair with a base leg
(`LogVar2FJModelParameters.ZAR` for USDZAR on a USD book), and for a cross the alphabetically later
token priced in the earlier (`…ZAR.EUR` for EURZAR), however the deal or surface is written. The
base currency is read from the explicit `System Parameters`; a book declaring none refuses. A
base-leg pair is fitted on its `FxRate`'s own axis, so two books with different base currencies fit
reciprocal laws of one pair; only a cross is canonical.

**The reciprocal axis.** A deal whose underlying is the currency the law is priced in pays on `1/S`.
`instruments.spot_model_reciprocal_axis` decides that and sets `Invert_Spot`; it is an allow-list
holding `LogVar2FJ` alone, so a model without the measure change refuses by name. The change of
numeraire is on the [theory page](../theory/stochastic_volatility.md#equity-and-fx).

**The pin books with the trade.** The quote reports what it pinned under
`valuation_configuration`; `/book/quote` merges it into the book in the same edit that splices the
deal (`pin_models`), refusing where the pinned factor is no longer on the book.

### Refusals before anything prices

`materialize`, `declared` and `quote` refuse by name: a missing required or unknown parameter, a
pair the book quotes no surface for (naming those it does), an expiry that does not parse or lands
on or before the base date, a non-positive notional, a stated barrier on or through the spot on its
own side, a TARF on the quote currency, a strip whose frequency does not divide its tenor, an
unknown netting set, a margin in a currency with no `FxRate`, and a margin on a recipe solving more
than one coordinate.

### Solves are bracketed {#solves}

`run_solve` brackets around the market spot, crossed to the engine axis and sorted:

| field | bracket (× spot) |
| --- | --- |
| a strike | `Solve.STRIKE_BRACKET` (0.25, 4.0) |
| an accrual strike | `Solve.ACCRUAL_BRACKET` (0.5, 2.0): a strip is flat once every fixing redeems, and a surface does not extrapolate to deep moneyness |
| a `Down_*` barrier | (0.25, 0.9999) |
| an `Up_*` barrier | (1.0001, 4.0) |
| anything else | a secant from its current value (exact in two pricings for an affine field) |

A re-solve seeded on an earlier root first tries `±Solve.SEED_BRACKET` (2%) around it and falls
back to the full bracket where that does not straddle. No root in the bracket refuses with the
economics stated ("cannot be financed at these parameters").

## The two-way charge {#two-sided}

Every leg prices at the mid and the market's spread is a charge on the coordinate the recipe solves.
The book's `FXVol` surface is bootstrapped from `Quoted_Market_Value` alone; the `FXVolPrices` block
may carry `Quoted_Bid`/`Quoted_Ask` per pillar ([Market Prices](market_prices.md#fxvolprices)), and
the runner is that data's only reader.

- `quote_two_way` reads each used pillar's half-spread `max(0, (ask − bid)/2)` — a crossed print
  reads zero-wide — keyed by `FXVolSurfaceParameters.descriptor`, the name its `dV/dq` leaf is
  published under. A pillar is a bucket; nothing is interpolated.
- `leg_vega` runs one first-order greeks valuation of the leg alone, on the client's paper, with
  `Quote_Sensitivity` on the `FXVolPrices` block (`vol_risk`). A leg walking a fitted spot model
  reads no surface, so it is read on a copy with its spot model dropped (`without_spot_model`), the
  lognormal reading at the same terms; `spread_source` says `'surface'` or `'lognormal reading'`.
- A leg's charge is `Σ_pillars |vega_p| × half_p` — the absolute value, because a pillar is dealt on
  the side the risk puts it, not the side the leg's `Buy_Sell` says. The structure's charge is the
  sum over legs; legs are not netted against each other first (a desk decision on the
  [roadmap](roadmap.md#decisions-waiting-on-the-desk)).
- The charge moves the solved coordinate in a second pass seeded on the first. A recipe that solves
  nothing (straddle, strangle) charges the premium instead; `charged_on` names which.

Outcome fields, all in the client's sign convention: `net_mid` (the legs at mid, what the trade
marks at once booked), `net` (plus everything charged), `edge` (the two-way charge alone,
non-negative), and per leg `spread_charge`, `spread_source` and `spread` rows
`{pillar, vega, half, cost}`, where `half` is always the market's half-spread and `cost` the money
charged. A leg no reading reaches carries `null`, never `0.0`. A book with no two-way runs no greeks
at all and quotes the mid, with `spread_note` saying why.

## The sales margin {#margin}

`margin` is `{'amount': 50000.0, 'currency': 'ZAR'}`, money in any currency. `margin_value` crosses
it to the calculation's currency (else the base) on the document's own `FxRate.<ccy>.Spot` ratio,
the market the quote is struck on. It is charged on the same coordinate as the two-way: a financing
leg's target becomes the premiums it finances plus the charge, a strip targets minus it, and a recipe
solving nothing charges the premium. On the client's paper `net` reads the margin back negative; the
booked mirror marks the bank at plus the margin and the edge. `compose` records it as `Sales_Margin`
and `Sales_Margin_Currency` (declared on the shared `Admin` group, a record nothing prices off).
`edge` excludes it, so a policy scaling the two-way never touches an agreed margin.

## Risk-impact pricing {#risk-impact}

The two-way is what the market charges a trade; this step charges what the trade leaves on the
book, priced at the market's own two-way. It runs only where the job declares a `Quote Policy`
section beside `Calculation` (`read_policy`); its absence is the off switch, and the outcome's
`risk.scale` is then `null` rather than `1.0`.

| field (`Structure.POLICY`) | default | meaning |
| --- | --- | --- |
| `participation` | 0.5 | the share of a measured saving passed to the client |
| `floor` | `'mid'` | where tightening stops; the only value accepted |
| `scope` | `'vol'` | the book the residual is measured in; the only value accepted |
| `bucket_limit` | none | an `abs(risk after)` per bucket past which nothing is tightened |
| `min_ticket_bp` | 0.0 | a floor under the edge, in bp of notional, crossed to the report currency |
| `firm_seconds` | 600 | how long a quote stays acceptable; read by `/book/quote` |

- **Measurement.** The mid candidate is composed and `mirror`ed (the same seam the booking uses) and
  the book's vol risk is read with it (`vol_risk`) and without it (`book_risk`, cached on the served
  book's `vol_risk_cache` by content hash). `risk_document` sets `Greeks: 'First'` and
  `Quote_Sensitivity: 'Yes'` on the `FXVolPrices` block, and the copy is bootstrapped and priced in
  one `Context`, so the [attachment](quote_sensitivities.md#the-attachment) leaves carry `dV/dq`;
  descriptors are summed across published blocks.
- **Cost.** `risk_buckets`: `Σ (|after| − |before|) × half` over quoted buckets; negative is a saving.
- **Scale.** `risk_scale`: `charge_effective = min(charge_full, max(min_ticket, max(0, charge_full −
  participation × saving)))`, `scale = charge_effective / charge_full`. A risk-adding trade stays at
  the full spread (no surcharge), and a bucket past `bucket_limit` suspends tightening and is named.
- **One pass.** Vegas and buckets are read at the mid solution while the quote sits at the charged
  coordinate — a stated first-order approximation, not a fixed point.
- `scale` stays `null` with a note when there is no policy, no two-way, no readable leg, a
  non-positive charge, or no vol quote leaves.

## The quote lifecycle

**Quote** (`POST /book/structure` → `StructureJob.run_job`, queued at a base valuation's cost class
in the curiosity lane, so the record does not move):

1. stamp `quoted_at`; under a spine home, pin the book's `plan_hash`, `values_hash` and board age
   before anything changes the copy;
2. `patch_live_spot` writes the terminal's spot onto the job's own copy through `with_live_spots`
   (the inverse of `engine_spot`); only the spot moves, and any failure falls back to the book's
   spot with the reason in `spot.note` — one request bounded by `SPOT_TIMEOUT_MS`, a failure that
   reached the terminal remembered for `SPOT_BACKOFF_SECONDS`;
3. `structures.quote`;
4. file the pending trade as `DV_HOME/tmp/<quote_id>.json` (the outcome, the composed deal,
   `quoted_at`, and under a home what the acceptance will file), with the sheet beside it when
   `xlsxwriter` is installed, else `files['sheet_note']`.

Two ids, both hashing a clock because a quote is an act and two identical asks are two quotes: the
runner's `quote_id` (structure, parameters, netting set, margin, the market carried) names the files;
the service's `result_id` (book etag and the request) names the queued job.

**Accept** (`POST /book/quote {quote_id, actor}`): reads the pending file, refuses a `portfolio`
(stated when quoting), answers `booked` for a quote already on the book, refuses a quote older than
`firm_seconds` where the book declares a policy, and books the `mirror` of the pending deal through
`deal_edit` — the same validate-before-write seam as `/book/deals` — under the quote's netting set,
merging its `valuation_configuration`. The file stays as the audit trail. Under a spine home the
acceptance also checks the plan and the board's age and reports the market's move; that half is on
[The Spine](spine.md).

**Composition.** `compose` puts the legs inside the deal block under `Children`, because a quote is
filed and hashed whole; `book_node` lifts them onto the node, which is the shape the engine walks
(`splice_deal`). A container spliced with its children inside the block loads empty and prices at
zero.

## Testing

`tests/test_structures.py` holds each structure to a financial identity through real documents: a
straddle is its call plus its put; collar, seagull and forward extra net to zero within the solve's
residual; a knock-in plus a knock-out is the vanilla; one trade quotes the same from either side of
the pair; a margin reads minus on paper and plus on the booked mirror; each pillar's charge matches a
re-bootstrapped central difference of that pillar; a book with no two-way or no policy quotes the
mid exactly; every variation books the deals it declares. The strips' deal types are gated on
their own in `tests/test_fx_tarf_json.py` and `tests/test_fx_accumulator_json.py`, the sheet in
`tests/test_quote_sheet.py`.

Two things to get right in a new gate:

- **Comparing orientations.** An equivalent notional converts at the strike (`N` rand is `N/K`
  dollars), not the spot.
- **Strip tolerances.** A strip's solved strike is a root over a Monte Carlo estimator: deterministic
  for a seed, converging only with the path count. Compare at the declared count; the gates allow
  `AXIS_TOLERANCE` (2e-4).

## Open

The two-way on a fitted leg reads the lognormal vega, not the fit's
([roadmap](roadmap.md#the-engine)); netting a package's legs before charging is a desk decision;
incremental XVA as the second risk-impact term and a ratio-solve primitive for participating
forwards are designed, not built ([roadmap](roadmap.md#designed-not-built)).
