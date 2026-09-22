# Structures

`derivus/structures.py` is the sales desk's vocabulary as declarations: what a zero-cost collar IS —
its names, its legs, how it composes — lives in the repo rather than in whichever model is driving the
MCP tools. The end user's host may be any LLM, so any finance left in the model's head is finance that
sometimes does not happen. The model's job is a sentence into named parameters; the registry and the
runner own the rest.

## A structure is a class, and everything on it is a declaration

The class name is the registry key (`globals()` dispatch, the house pattern), and it declares:

- **`vernacular`** — the sales names, comma-separated (`'zero-cost collar, range forward, cylinder'`).
  `describe_structure` matches these as well as the class name, so a model that says "cylinder" lands
  on `ZeroCostCollar`.
- **`fields`** — the parameters as `schema.F` descriptors, so `describe_structure` renders exactly like
  `describe_instrument_type` and every client shares one rendering path.
- **`legs`** — named legs, each a `DealType` plus a PARTIAL deal block. This is the [Market Prices quote
  pattern](market_prices.md#a-quote) verbatim: a leg never restates an instrument's fields — the
  `Instrument` store's declarations ARE the leg's schema — it pins what the structure fixes and maps
  parameter slots. A gate holds every leg's type to a declared instrument.
- **`variations`** — named ways the structure is dealt, in place of `legs` where there is more than
  one: each carries the side of the pair its CLIENT buys, the parameters only it takes, and its own
  legs. The `recipe` is variation-neutral, roles being the same either way.
- **`recipe`** — the composition as data: `Price('leg')`, and `Solve('leg', 'Field', target)` where a
  target is a literal or `Premium('other_leg')` (with `__neg__` and `__add__`, so a collar's financing
  leg solves to `-Premium('protection')` and a seagull's to the negative of a sum). Steps run in order;
  each prices the leg ALONE against the book document.

`mapping['Structure']` is `schema.emit_structures(structures)`, assembled with the other stores, so
`GET /schema` publishes the vocabulary and a front end can grow a structures screen for free. An entry
carries `legs` or `variations` and never both, so a consumer reads which shape it has off the entry.

## One structure, more than one booking {#variations}

A forward extra operates two ways — an exporter FLOORS the pair and an importer CAPS it — and they are
not two prices of one trade. **They are two different bookings**: a vanilla and a barrier each, the
vanilla a put or a call and the barrier an up-and-in or a down-and-in, and the book holds the mirror of
whichever was dealt. The same is true of every structure here bar the straddle and the strangle, so a
variation is DECLARED data rather than a branch in the runner.

**`reflected(variation, rename, fields)` derives the mirror image** where there is one. Every leg's
`Option_Type` swaps and every `Barrier_Type` crosses through `BARRIER_FLIP`, a level said about the pair
reading the other way round for a client standing the other side of it. Two things do not move: In and
Out describe what the payoff does on touch and mean the same to either client, and `Buy_Sell` is the
CLIENT's own side on both sheets — an importer buys their protection exactly as an exporter buys theirs
— so client paper becoming the bank's position stays [`mirror`](#two-sided)'s one seam. `rename` carries
the variation's own parameters under their mirror names (`{'floor': 'cap'}`), applied to the leg slots
that fill from them, and `fields` is those parameters as the declarer WRITES them: prose says what a
level means to the client on that side, which is not something a rename can derive.

**`variation_for` is ONE selection rule for every structure.** The variation is the unique one
consistent with everything the ticket states: a stated `buy_currency` or `sell_currency` must be a side
of the pair and fixes which side the client buys; a stated parameter that is some variation's OWN admits
only the variations declaring it. So "a forward extra, cap 16.90" quotes with no direction stated, a
strip — the same shape whichever way it is dealt — must state one, and stating both means stating them
consistently. NOTHING consistent refuses naming what contradicts what ("a client selling USD and buying
ZAR deals floor, which states floor, not cap"); MORE than one refuses naming what to state. "Stated" is
ONE predicate — a blank is not a statement, since a front end round-tripping an unfilled `Float` sends
`0.0` and the store publishes `"value": ""` for every required field.

The two currencies are SELECTORS rather than parameters — they choose a form instead of filling a leg —
so they are optional to state, never defaulted, and published with no value under `selector`, which
says whether one is `required` or `optional`. That is COMPUTED from the declarations rather than
listed: where some variation's own parameters do not tell it apart from another's, only the direction
can say which is meant, so the two accrual strips — whose forms name one `knockout` or one `target`
between them — require one while a forward extra, a collar and a seagull do not. A parameter no form
takes refuses by name against the roster that does.

The outcome then says which variation it quoted and the client's own two cashflows read off it
(`variation`, `client`), so what is reported and what was priced cannot disagree.

## The runner owns the conventions — all three of them

Parameters arrive in MARKET terms and the engine never shows through. `materialize()` converts once,
for every structure at once:

- a USDZAR strike of 15.50 becomes `Strike_Price = 1/15.50`, because the engine's `FxRate` carries
  REPORTING units per unit of currency and an FX option's strike lives on that axis;
- the option SENSE inverts with the same axis — a market call (the right to buy the base currency) is
  an engine PUT on the quote currency — so no structure declares an orientation and none can get it
  wrong;
- a BARRIER leg crosses on a third axis. `Barrier_Price` inverts exactly as a strike does (it is a level
  on the same pair) and the barrier's DIRECTION inverts with it — the `Up_And_In` a forward extra
  declares on the pair is booked `Down_And_In` on a rand notional — while In/Out describes the PAYOFF
  rather than the axis and never moves. `BARRIER_FLIP` is that map, declared beside `VANILLA` and
  applied exactly where `Option_Type` flips. `Option_Style` is NOT pinned on a barrier leg:
  `FXBarrierOption` declares no such field.

An **accrual** leg asks the axis question a fourth time and gets the first NO. A knock-out is a LEVEL
and crosses exactly as a barrier does, so an accumulator quotes from either side of the pair. A
**target** does not cross at all: it is a cap on a sum of DIFFERENCES rather than a level, and
`1/S − 1/K` is not the reciprocal of `S − K`, so no number in reciprocal units means the client's cap —
two TARFs "capped the same" on the two axes redeem on different paths and are different trades.
`FXTARFOptionDeal`'s `InvertedTarget` flag is not the way out: it moves the whole fixing onto the
reciprocal axis (`eff_intr`, and so `cf_itm` as well as the accrual), which pays `Underlying_Amount`
per unit of MOVE in the pair. That is a coherent product — a rand notional under the flag pays a million
dollars per rand — but not the one `notional_currency` names, since the notional here is an AMOUNT OF
that currency. The two read **0.77% apart** in the solved strike on the gate's book and neither is wrong
about its own product. So `InvertedTarget` is `False` on every leg the runner builds, and
`TargetRedemptionForward` REFUSES a quote-currency notional by name.

All three are gated the same way: a straddle quoted on a ZAR notional and one on the USD it buys must
net the same to machine precision, and a forward extra quoted both ways must solve the SAME barrier in
market terms, travelling opposite paths through the runner. The equivalent notional converts at the
STRIKE, not the spot — `N` rand is `N / K` dollars, which an at-the-money straddle cannot tell apart and
a 0.97-spot forward extra misprices by exactly the moneyness. `notional_currency` IS the option
underlying, which makes "is the notional the quote currency" exactly the discriminator the inversion
needs.

**What cannot be quoted refuses before a leg is built.** A parameter the structure declares REQUIRED
and the client did not state is named against its own `fields` — the SELECTED variation's own included,
so a form chosen by the direction alone is held to the level it takes before a quote id is hashed. A
pair whose surface the book does not carry is named with the pairs it does, because otherwise every leg
is dropped at load and the quote comes back `priced but reported no mtm row`. An expiry on or before the
base date and a notional that is not positive refuse there too — a zero-day option quoted as if live,
and premiums with the sign reversed, are both numbers a client could be handed.

**A level the client STATED sits on the live side of its own direction**, read on the engine axis both
are on by then, so an `Up_*` is strictly above the spot and a `Down_*` strictly below it — a level ON
it is through already, `pv_MC_Accumulator`'s own survival being strict whichever way the barrier faces.
A decumulator knocking out ABOVE the market solves a spectacular-looking rate for a strip that is dead
where it stands. The comparison is against the SPOT and the refusal says so, naming the level, the spot
and the direction in the pair's own terms: a knock-out is observed on the FIXING dates, so on a carried
pair a level through today's spot can still be live at the first of them, and this reading refuses that
ticket rather than booking a dead one — loudly, and with the remedy. A SOLVED level is the recipe's own
and is already bracketed on that side, so it is not checked twice.

**Strike solves are BRACKETED** (brentq over `(0.25, 4.0) ×` the market spot, crossed to the engine
axis), never the secant — `solve_deal_field`'s secant seed lands in the dead flat region for an
engine-axis strike of 0.06. A zero-cost leg no strike inside the bracket can fund refuses by name.

A **barrier** solve is bracketed on the side its own type lives on, off the same ends with a hair of
buffer at the spot so the level never lands exactly on it: `Down_*` over `(0.25, 0.9999) × spot`,
`Up_*` over `(1.0001, 4.0) × spot`, both read on the ENGINE axis the leg's `Barrier_Type` has already
crossed to. A knock-in's premium is monotone in its barrier, so brentq owns the root or refuses by name.

An **accrual** strike is bracketed over `ACCRUAL_BRACKET` — `(0.5, 2.0) ×` spot — and the reason is
measured. A strip's value is monotone in its strike but SATURATES at the low end: past the point where
every fixing redeems the target at once it is flat at `target × notional` discounted, so moving the end
in gives up no root. What leaving it out gives up is the solve itself — at `0.25 ×` spot on
`fx_tarf_job.json`'s market the TARF prices **NaN** and `brentq` refuses at its own first evaluation,
because a surface quoted over moneyness `[0.8, 1.2]` does not extrapolate to 0.25 as a volatility.
Measured on that fixture: NaN at 0.28, and flat at 99,697.57 — the redeemed target to the cent — from
0.30 through 0.50.

One furnishing that looks like a default and is not: a deal block IS the field dict the pricer reads
(`Deal.__init__` takes it verbatim), so a DECLARED default never reaches it. `pv_barrier_option` asks
for `Barrier_Monitoring_Frequency` and `Cash_Rebate` by name, so `materialize` writes both onto a
barrier leg — `{'.DateOffset': '0M'}`, continuous monitoring in the wire form a Period field decodes
from, and a zero rebate. Without them the deal is SKIPPED at load: an ERROR line in the log, a leg
priced at nothing, and a quote that still returns.

## An accrual leg is one leg and a SCHEDULE {#accrual}

`TargetRedemptionForward` and `Accumulator` are the registry's first MULTI-FIXING structures and its
first legs that are a whole strip. Each declares ONE leg and a recipe of one step —
`Solve('tarf', 'Strike_Price', 0.0)` — because a TARF is dealt at no upfront and the strike IS the
price. `furnish_accrual` is where a leg becomes a strip:

- **the schedule.** `fixing_grid` grows `[[fixing, settlement, observed], ...]` — the untagged row shape
  both declarations read by iterating — from the tenor and `fixing_frequency`, each fixing at
  `base + n × frequency` rather than a step off the last (an offset applied repeatedly from a month end
  walks: 31 Jan + 1M + 1M is 28 Mar), settling `FIXING_LAG` days on, observed 0.0 because a quote is
  struck today. A tenor holding no whole fixing period refuses rather than returning an empty strip.
- **the two ways a strip comes out SHORT**, neither allowed to be silent. A frequency that does not
  DIVIDE the tenor stops at the last fixing that fits — a 1Y ticket at 5M fixes in November and April,
  the `Expiry_Date` becomes the April settlement, and the deal is priced, reported and two-way spread at
  a tenor the ticket does not say. That REFUSES, naming the expiry, the frequency, the last fixing it
  would have produced and both remedies (a frequency that divides, or the broken date quoted directly).
  And a `Base_Date` carrying a TIME (a terminal stamps 16:30) put the final fixing one comparison past a
  midnight expiry and turned twelve monthly fixings into eleven, so the base is normalized to midnight
  before the loop.
- **the expiry.** `FXTARFOptionDeal`'s `Expiry_Date` is the LAST SETTLEMENT — a strip is not over until
  its final cashflow lands — while `FXAccumulatorOptionDeal` declares no such field, so the shared
  block's is REMOVED rather than carried as a key nothing will read.
- **the notionals.** `Underlying_Amount` is the notional PER FIXING and `LeverageNotional` is
  `leverage ×` it. `leverage` is the registry's first parameter with a DEFAULT (2.0, the market's own
  gearing), published as the descriptor's `value` and read through `declared()` rather than a `.get`.
- **the model.** Both deals declare `spot_models = ('None', 'LogVar2FJ')`, and the runner pins
  `LogVar2FJ` (`structures.SPOT_MODEL`). The switch is a `Valuation Configuration` entry per deal
  TYPE resolved by naming
  convention off [the pair's key](#the-join) — `LogVar2FJModelParameters.ZAR` for a USDZAR leg on a
  USD book, whichever side the notional is on, because the base currency is a numeraire and can name
  no block. `spot_model` checks the book for that exact key and pins the model only where it is there: the
  switch on with the factor absent raises inside the engine's dependency loop, which SKIPS the deal and
  logs an ERROR, so a structure that pinned it unconditionally would quote ZERO on every uncalibrated
  book. Where it is absent the leg carries a `note` naming that factor and the verb that installs it.
  **A book that already declares the switch for that type keeps it**, and both the check and the note
  are taken on THAT family: a leg priced under a book's own pin and noted as GBM is a note disagreeing
  with the number beside it.
  The rule needs the BASE as well as the pair, read off `System Parameters.Base_Currency` in the
  EXPLICIT block — the same half `market_data` reads and the only half a quote can write. A book keeping
  its `System Parameters` behind a `MarketDataFile` answers nothing here, and `utils.spot_model_currency`
  REFUSES an unknown base rather than guessing `Underlying_Currency`, which would pin a model the engine
  then looks up under the other name.
- **and the model books with the trade.** The pin is written on the QUOTE's copy of the document,
  because a quote is not a trade and must not touch the book. That copy dies with the answer, so the
  outcome REPORTS what it pinned (`valuation_configuration`), the pending file records it, and
  `/book/quote` merges it into the book inside the same edit closure that splices the deal — one lock,
  one validation, one write. Without that, a leg dealt under a GARCH re-marks as a lognormal on the next
  valuation and nothing says so, because both numbers are plausible. A pin whose parameters the book no
  longer carries REFUSES at the approval rather than booking a switch the engine would raise on.

### The join: one law per pair, and the reader learns the axis {#the-join}

Three keyings met at an accrual leg and did not agree. **They are one rule now**: the pair's key
(`utils.spot_model_currency`, which the engine's lookup, the runner's presence check and the
dependency discovery all call) — its NON-BASE token where the pair has a leg on the base, and for a
CROSS the alphabetically LATER token priced in the EARLIER.

| who | keys off | USDZAR on a USD book | EURZAR on a USD book |
|---|---|---|---|
| the engine (`get_spot_model_params_factor`) | the pair's key | `…ModelParameters.ZAR` | `…ModelParameters.ZAR.EUR` |
| the calibration (`fx_surface_block`) | the same key, which is what it writes | writes `…ModelParameters.ZAR` | writes `…ModelParameters.ZAR.EUR` |
| `furnish_accrual` | still forces a TARF onto the pair's BASE, since a target has no reading on the reciprocal | `Underlying_Currency` = USD | `Underlying_Currency` = EUR |

**The cross's axis is a property of the two CURRENCIES and of nothing else** — not of the deal's
orientation, and not of the order a desk happens to store its surface in. `EUR/ZAR` is the rand
priced in the euro whether the book carries `FXVol.EUR.ZAR` or `FXVol.ZAR.EUR`, and the fit finds
the surface in whichever order it is stored and lets `Invert_Moneyness` absorb the difference,
exactly as it already does for a base-leg pair quoted the other way up. Keying off the stored
spelling instead is what that costs: measured on one economy, a EUR-in-ZAR fit against the ZAR-in-EUR
one the other spelling gave read **1.0e+00** apart at `Rho_S` — the sign of the skew — and the two
books priced the same trade **1.0e-3** apart, five times the solve's own axis band, with both books
pinning the model and neither noting anything.

The second token is the currency the law is PRICED IN, and it is a token on the parameter block's own
name rather than on an `FxRate`'s: a two-token `FxRate` name is a primary spot plus an
`ObservedBasis` tail in discovery (`config.nested_fields`), which is a different object entirely, so
a cross is never spelled that way. A pair with a base leg keeps the one-token key, so no document or
factor written before this moves. A name that is not ONE currency has no pair to be a leg of and
REFUSES by name — and because discovery runs outside the per-deal guard, `conditional_fields`
answers `[]` rather than letting that refusal out: the engine's own lookup raises the `KeyError` the
dependency loop turns into one skipped deal, which is the contract a portfolio of thousands survives.

The base currency is a NUMERAIRE, never a rate: `FxRate.<ccy>` is that currency priced in the base, so
`FxRate.USD` is identically one on a USD book and no fit describes it. A USDZAR TARF therefore used to
look up a factor that could not exist and rode GBM however many times the pair was calibrated —
measured on the calibrated book, where its solved strike came out **bit-identical** to its GBM one.
Under the single rule it separates by **3.78%** on that book, against a solve floor of 2.5e-5.

**The forced TARF sits on the reciprocal of the fitted axis, which is a change of NUMERAIRE as well as
of axis** — the deal pays in the other currency, and a law that cannot be carried there prices on an
axis nobody fitted. Leaving the carry out leaves one variance of Siegel drift in the answer: the two
orientations of one accumulator then solve strikes **3.7e-3** apart and the gap does not close with
the path count.

**LogVar2FJ transports as a measure change rather than a parameter.** Under the `S`-numeraire the
step's density factorises over its own draws, so each shifts by its own loading —
`eta_l ~ N(rho_l sqrt(V), 1)`, `eta_s ~ N(rho_s sqrt(V), 1)` — and the residual's MIXER is drawn
from its Esscher-tilted law: tilting the joint law by `exp(X)` leaves the inverse Gaussian's shape
`delta_A^2` alone and moves its mean from `delta_A/gamma` to `delta_A/gamma_1` with
`gamma_1 = sqrt(alpha^2 - (beta+1)^2)`, which is the same `gamma_1` the forced drift already
carries. Given the mixer the Gaussian's mean then gains `+G`, which IS the block law's existing
`-(M + Sigma^2)` spelling at the deal's own carry (`utils.LogVar2FJ.walk`, `pricing.LogVar2FJKit`). One
law, two currencies, no second fit, and one line under `invert`. In the Gaussian-residual limit the
reciprocal-axis accumulator is GBM's to **7.8e-16** relative (the direct axis 1.9e-15); a strip of
forwards on `1/S` at 2^18 paths reads **7.9e-05** relative against `Σ_j D_j N S_0 exp(carry t_j)`
(the direct axis 2.0e-05); and the tilted mixer satisfies the Esscher identity directly,
`E~[1/S] = 1` within 0.17 SE at 400,000 draws, which is what pins the `δ_A/γ₁` mixer rather than
`δ_A/γ`. Uncarried, the two orientations of one accumulator solved 3.7e-3 apart on the Poisson
residual and the gap did not close with the path count; carried, they solve inside the seed spread.

The COMPONENT family does not transport — the change puts a state-dependent term in its long-run
intercept, `omega_t + phi(1 − 2·gamma_2)h_t`, and leaves the family — so a component deal on the
reciprocal axis REFUSES by name rather than pricing off a law nobody fitted; `spot_model_reciprocal_axis`
is the allow-list (`LogVar2FJ`) a family joins. A CROSS asks the same question with the token its
law is priced in standing in for the base, so a EURZAR strip whose `Underlying_Currency` is EUR is
carried exactly as a USDZAR one whose underlying is USD.

**A CROSS is fitted and keyed exactly as the pair would be on a book whose base is its EARLIER
token.** `EUR/ZAR` is the rand priced in the euro — the orientation USDZAR has on a USD book — so
every existing "the domestic is in the name" path is reused with `EUR` standing in for the domestic:
the underlying is ZAR, the moneyness inverts, the discount curve is the euro's and the carry the
rand's, each read off its own `FxRate`'s `Interest_Rate` as the pricer reads them. What a cross adds
is the SPOT: a ratio of the two base-priced rates, declared as `Priced_In`. Before the key was the
pair's, a USD book that had calibrated EURUSD answered `…ModelParameters.EUR` for a EURZAR strip,
found it, pinned the model and priced the cross off the other pair's law with no note — measured
**+2.56%** on the solved strike of a six-month accumulator whose whole model effect is a sixth of
that. The OUTER is untouched: each base-priced rate simulates under whatever the book's `Model
Configuration` says, and the pricer re-seeds the pair's law at each node exactly as it does for any
pair with no process of its own (`LogVar2FJKit.carried` looking for `FxRate.ZAR.EUR`, which nothing
publishes, and finding nothing).

**WHAT THIS DOES NOT REMOVE, and did not introduce: a base-leg pair is fitted on the `FxRate`'s own
axis.** EURZAR on a EUR book is the rand priced in the euro and on a ZAR book it is the euro priced
in the rand, because that is what an `FxRate` IS on each of them — so two books of DIFFERENT bases
fit reciprocal laws of one pair, and their strikes agree only to the family's own reciprocal-axis
difference, measured **1.0e-3** on one economy against a solve band of 2e-4. Only the CROSS case is
canonical; a pair with a leg on the base follows its book. Two books of the same base agree exactly.

### What the model is worth, and what it is not {#model-worth}

The case for a spot model on an accrual strip is **not "the skew and only the skew"**. Measured on
the gate's book under the plain Heston-Nandi family - retired 2026-09-06, the reading kept because
the decomposition is the model class's and not that family's - an accumulator's zero-cost strike
moved **+0.378%** from GBM, of which the LEVERAGE CHANNEL alone (the ARCH coefficient to zero with
the persistence and the stationary per-step variance held where the fit put them) was **+0.048%**,
about an eighth, and the sign of the leverage alone +0.003%, inside the solve's own Monte Carlo
floor at 16,384 paths.

So what a spot model is mostly worth on this book is its VARIANCE PATH — a level and a persistence a
lognormal read off the same surface does not have — and the skew is a real but secondary term. What
LogVar2FJ adds over that is the FORWARD skew, a lever in calendar time rather than in the state
([Market Prices](market_prices.md#logvar2fj)), plus the second-order greeks and the quote-space risk
neither retired family carried.

**The scenario generator is the pricer's own walk.** `LogVar2FJImpliedSpotModel` is the xVA outer
process: an implied process reading the same calibrated `LogVar2FJModelParameters` factor the OSS
kit prices off — through `implied_tensor`, so CVA vega reaches ONE leaf and not two — and stepping
`utils.LogVar2FJ.walk` on the trading day between scenario nodes, whole days plus the remainder as one
shorter step. Outer, inner and pricer are one walk and one mixer STRUCTURALLY:
`LogVar2FJImpliedSpotModel.draws is LogVar2FJKit.draws` and `.residual is LogVar2FJKit.residual`
are the same function objects, so a fork seeded at an outer node cannot disagree with the stride it
continues.

Given the day's two shocks and the block's mixer the interval return is `N(M, G)` exactly, and
the process declares FOUR sub-factors per name — the return's Gaussian given the mixer (the row
every book already marks, under the key GBM's estimator answers), the fast and the slow
log-variance shocks (`.S`, `.L`) and the mixer (`.G`) — so the framework hands it one normal per
row per scenario step and a declared correlation on each is realised (2026-09-11). Each variance
factor's node-to-node step is the exact OU transition on the interval driven by its framework
normal; the daily path inside the interval is the OU bridge between the endpoints, spelled as a
conditioning of the free shocks (`e = z − u(u'z) + uZ` with `Σu² = 1`), so the walk is untouched,
the endpoint is a function of the previous endpoint and the framework normal alone, and a daily
grid leaves `u = 1` exactly. The mixer is the fourth normal through its own quantile,
`G = F_IG⁻¹(Φ(Z_G))`, `Φ` taken in double. Measured on two names at the Q-sized truth against a
truth simulated outside the engine at the document's own clock: the return row alone at 0.60
realises **0.028** on returns (the private-dice behaviour it replaces), all four rows at 0.60
**0.483** against 0.482, the variance rows at 0.8 with the mixer at 1 **0.732** against 0.729;
put one die back to a private draw and the four-row reading collapses to 0.432. The historical
estimator returns the four innovation columns under the inverse map — `eps`, the two smoothed
shocks standardised, `Φ⁻¹(F_IG(Ĝ))` — so an estimated row is the row the outer realises; on a
closes-only archive the Kalman smoother attenuates the shock columns (0.32 and 0.53 against a
drawn 0.60, where the return and mixer columns read 0.55 and 0.35), a property of the measurement
the estimator states by name.

The dilution `D` and the INFO line that reported it are gone with the private dice. Which outer
a book runs — the GBM term-structure outer with the LogVar2FJ pricer, or the LogVar2FJ outer
across a netting set — is the desk's `Model Configuration`; both mean what they say under the
Cholesky. A calendar bucket knot cutting one scenario interval's clock into two residual draws
takes the second piece's mixer from the process's own stream, there being one framework normal
per interval, which `precalculate` names at INFO; every factor in the book carries one `Alpha`
bucket, so nothing reaches it.

**The day's shocks are a function of the day.** They come from a `torch.Generator` per
CALENDAR-ANCHORED segment of `LogVar2FJKit.CHECKPOINT_STEPS` trading days, never stored and redrawn inside
the checkpoint's recompute, so where the scenario nodes fall moves no draw; `-eta` on the antithetic
half, as the framework mirrors its own normals, and the mixer uniform per scenario step is
`quasi_rng`'s with `1 - u` on that half. `Checkpoint_Outer_Walk` (default `Yes`) is the outer tape's
own switch beside `Recompute_Inner_MC`, the pricer's: two tapes, two named switches.

**Replay is refused.** The log-variance carries its own two shocks, so `(ell, s)` is not a function
of the realised returns and `reseed_from_path` answers a sentence saying so — recovering the state
from an observed price path is a filtering problem, not a replay.

---

## The spread is quoted, the mid is booked {#two-sided}

A desk does not sell at the mid and its book does not mark at the offer; both are true at once because
the two numbers live in different places. The `FXVol` surface in `Price Factors` is bootstrapped from
`Quoted_Market_Value` alone and never moves — that is what every mark on the book runs off. The
`FXVolPrices` block beside it may carry each pillar's `Quoted_Bid`/`Quoted_Ask` ([Market
Prices](market_prices.md#fxvolprices)), which is DATA the bootstrap never reads. The runner is its only
reader.

**The two-way is a CHARGE, not a second surface.** Every leg prices at the MID, and what the market
charges for the spread is levied on the coordinate the recipe already solves — the shape the [sales
margin](#margin) and the [risk-impact charge](#risk-impact) both have. `quote_two_way` reads every
quoted pillar's `(ask − bid) / 2` off the block, keyed by the descriptor `dV/dq` publishes that quote
under, and nothing is interpolated: a bucket IS a quoted pillar or it is not a bucket. A CROSSED print —
a stale bid through a live offer — reads ZERO-WIDE rather than negative, `max(0.0, …)`, because the one
thing a desk must not do with a broken print is pay a client for it.

**What is read, per leg and per pillar.** At the mid solution each leg goes through ONE first-order
greeks run of its own — through `alone()`, so the leg is valued exactly as `run_price` values it, on the
CLIENT's paper, with `Quote_Sensitivity` on the `FXVolPrices` block of that run's own copy. The leg's
charge is then the sum over quoted pillars of `|dV/dq_p| × half_p`, and the structure's charge is the
SUM OVER LEGS.

The absolute value is the whole ruling: **a pillar is dealt on the side the RISK puts it, not the side
the leg's label does.** A geared accrual strip is one leg booked `Buy` and is net SHORT vol at every ATM
and butterfly pillar, so signing by the label quoted a NEGATIVE edge — the desk paying a client 6.5k to
25.5k on a 1m ticket to take the trade. Charged per pillar on the absolute vega the same magnitudes come
back the right way round: on the gate's own two-way book at a 1m USD ticket, the TARF **+6,627.45**
buying and **+7,316.06** selling, the accumulator **+22,415.98** and **+25,294.53**, against the collar's
**+1,985.57**. Nor is a vanilla single-signed: a risk reversal follows the WING, so every leg holding a
PUT reads a `dV/d(RR)` opposite in sign to its own ATM vega, and one side per leg mis-charged about 4%
of a collar's spread and 12–14% of a strip's.

**A leg priced under a fitted spot model** reads nothing off the written surface — it walks the fitted
law — so it would publish no quote leaves at all and there would be nothing to charge against. The PIN
therefore decides which book the run is made against: a pinned leg is read on a copy carrying NO spot
model for its deal type, the LOGNORMAL reading of the same leg at the same terms, which is the vega a
desk would hedge in the quotes it actually trades. `spread_source` says which answered, `'surface'` or
`'lognormal reading'`. Reading the surface first and falling back cost a fitted strip **3.8 s** of a
~19 s quote on a run whose emptiness the pin already predicted. Without the substitution at all, a
model-priced strip solves the MID strike, captures exactly nothing, and reports a spread anyway.

**Each leg pays its own spread.** The legs of one package are NOT netted against each other before the
charge — a collar's bought put and sold call are charged as two tickets rather than as the one position
the desk has to deal. That is the desk's next dial rather than a defect: netting per pillar first takes
the gate collar's edge from 1,985.57 to 297.25, an 85% cut, and the numbers are in [the
roadmap](roadmap.md).

**`net` versus `net_mid`.** The finished legs are at mid, so `net_mid` — what they are worth — is what
the trade marks at the moment it is booked, and `net` is that plus everything the desk charged. A
zero-cost structure is quoted at zero, or at minus the margin where one was agreed, and MARKS at minus
the margin and the edge together. A recipe that solves NOTHING — a straddle, a strangle — has no
coordinate to move, so the PREMIUM carries both instead: the client's payment moves against them by the
margin and the charge, the legs book and mark at mid, and the desk's take lives in the cash rather than
on the coordinate. `charged_on` names which it was, the solved field or `premium`, so one convention
covers every structure and a reader never has to infer it. Both readings are in the CLIENT's sign
convention, `edge` is the two-way charge alone, and it is non-negative by construction.

Per leg the outcome carries `spread_charge` (money, in the pricing currency), `spread_source`, and
`spread` — one row per quoted pillar, `{pillar, vega, half, cost}`. **One meaning per name**: `half` is
always the MARKET's own half-spread, the same number `risk.buckets` prices a residual at, while `cost`
and `spread_charge` are money the desk charged — so a policy's tightening is stated once, under
`risk.scale`, and carried in the money rather than in the quote it is a fraction of. A leg no reading
reaches carries `spread_charge` NULL with a note saying so, never a zero, and so does the structure's own
`risk.charge_full` where NO leg could be read: a zero reads as a spread the desk measured and found to be
nothing, which a consumer auditing the quote could not tell from a leg nobody priced. Where
the book quotes no two-way at all, not one greeks run is made, the three keys are null, `spread_note`
names the absence, and the quote is bit-identical to the one the runner has always given — the gate
compares it float for float against a book carrying a ZERO-WIDE two-way, which exercises the whole
layer, so the presence of the data cannot move a price.

**Two passes, and the second is the quote.** Pass 0 solves at the mid against the margin alone; the
vegas are read there; pass 1 re-solves against the margin plus the charge. So the vegas are the MID
solution's while the quote sits at the charged coordinate — the same declared one-pass approximation the
risk-impact step carries, one step earlier. The solved coordinate lands CLIENT-WORSE by construction:
the forward extra's barrier comes IN toward the spot, the collar's cap comes IN, a buyer's strip strikes
UP and a seller's DOWN.

Pass 1 is SEEDED on pass 0's root rather than searching the whole bracket again: a charge moves a
coordinate by a spread's width — under half a percent on every form here — so the second solve brackets
±2% around the first answer and falls back to the full ends where that does not straddle. Measured, the
same root to nine significant figures and **7 engine runs against 16** on a collar's cap, **9 against
16** on a TARF's strike and **6.1 s of a ~12 s** fitted one. Pass 0 takes no seed, which is what keeps
every no-two-way quote bit-identical to the one the runner has always given.

## The sales margin rides the same coordinate the two-way does {#margin}

The two-way is what the market charges and the risk-impact step below is what the residual costs. What
the DESK adds on top is a sales margin, and it is quoted the way a client agrees one: `margin` on
`/book/structure` and `solve_structure` is `{'amount': 50000.0, 'currency': 'ZAR'}` — money, in whatever
currency it was negotiated in, which need not be a currency of the pair. `/book/solve` and `solve_deal`
take the same form in place of their float `target`.

**The conversion.** `structures.margin_value` crosses it to the run's reporting currency on the ratio of
the document's own two `FxRate.<ccy>.Spot` blocks — the same read `engine_spot` makes for every strike
bracket, off the same copy the live tick has already moved, so the margin and the legs see one market. A
currency the book carries no `FxRate` for refuses BY NAME at the verb (422), with the client still on
the phone. The answer states both halves: `margin` carries the amount as declared and its `value` in the
`pricing_currency`.

**The charge, and its sign.** It goes where the [two-way's own charge](#two-sided) goes, which is the
ONE coordinate the quote has. Where the recipe SOLVES: a financing leg's target becomes the premiums it
finances plus the charge, and a single-solve strip targets minus the charge instead of zero. Where it
solves nothing — a straddle, a strangle — the PREMIUM carries it instead: the client simply pays more,
the legs book and mark at mid, and `charged_on` says which coordinate it was. ONE convention, so a
margin is never refused for want of a strike to move it onto; a recipe solving MORE than one coordinate
still refuses by name, since the charge would be levied once per solve.

Because the quote is client paper, a solving structure's `net` reads the margin back NEGATIVE: the
client holds a structure worth minus what they paid for it, and the mirror the approval books marks the
bank at plus the margin and the edge together. Both readings are gated on the same quote — `net`
converted at the quote's own spot, and the mirror priced against the book. A collar's cap comes IN and a
TARF's strike moves UP, each by more than any solve tolerance.

`edge` is left alone: it is the two-way's charge and nothing else, so a policy that tightens the
MARKET's spread scales `edge` and leaves an agreed margin exactly where the client agreed it. The
composed `StructuredDeal` records `Sales_Margin` and `Sales_Margin_Currency` — declared
on the shared `Admin` group, since a margin is a property of a TICKET rather than of an asset class, so
the container, an FX leg and a cashflow solved to a margin target all record it one way. Nothing prices
off the field; the charge is already inside the terms. With no `margin` asked for there is no `margin`
in the answer and no `Sales_Margin` on the deal, and the quote is the one the runner always gave.

## The risk prices the spread {#risk-impact}

The two-way above is what the MARKET charges for a trade. What a DESK charges is the cost of hedging the
RESIDUAL that trade leaves on its book — a measurement, priced at the market's own two-way, rather than
a bp-per-skew number somebody invented. A trade that nets the book down is quoted tighter; a trade that
piles risk on is quoted at the full spread and no wider.

**The measurement.** The base pass is the MID solve, the same one the charge's own vegas are read at.
Its composed candidate goes through `structures.mirror` and the book's vol risk is read twice: the book
alone, and the book with the mirror spliced in through `book_node`. Both are `BaseValuation` with
`Greeks: 'First'` — one backward off the ROOT netting set, so a leaf's `.grad` is the whole portfolio's.

**The coordinates are QUOTE space.** `Quote_Sensitivity: 'Yes'` goes onto the `FXVolPrices` block of the
risk run's own copy, bootstrapped in the same `Context` that prices it — the attachment is harvested at
BOOTSTRAP, not at run ([Quote Sensitivities](quote_sensitivities.md#the-attachment)) — and
`Config.quote_leaves` then holds the ATM/RR/BF quote leaves the surface was built from. So a bucket is
`dV/d(ATM 1)`, `dV/d(RR 0.25 1)`, `dV/d(BF 0.25 1)`: what the desk would actually have to trade.
Descriptors are summed across every published block, which is the [collision
rule](quote_sensitivities.md#the-attachment). The switch is worth exactly zero forward, so turning it on
cannot move a price.

**The residual's cost.** The buckets are the ones [the two-way](#two-sided) is quoted in and they are
priced the same way, off the same `quote_two_way` read: a bucket's cost is the move in ABSOLUTE risk
times that bucket's own half, `dV/dq` being a vega in report currency per unit of quote, so the product
is money and nothing converts it. What differs is the quantity — the two-way charges a LEG's own vega
while this charges what the trade leaves on the BOOK. Summed, a NEGATIVE total is a saving. On the
gate's 1m ZAR collar, a desk holding one and quoted the same one back, the offset moves `ATM 1` by
−456.19 at a 0.002 half, `BF 0.25 1` by −45.48 at 0.001 and `RR 0.25 1` by −8281.52 at 0.001, for a
measured saving of **9.2394 USD** against a full charge of **105.9701 USD**.

**The policy** is a declared `Quote Policy` block on `Calc`, beside `Calculation` and `MergeMarketData`
— not inside `ExplicitMarketData`, because `Context.load_json` does `cfg.params[section].update(...)`
and a section `Config` does not declare raises `KeyError` on load (measured), and a mandate is not
market data anyway. Every reader of a job walks `Calc` by name, so an unknown key there travels through
load, pricing and the book file untouched, and `structures.quote` is its only reader. Six fields, each
read with `.get`:

| field | default | what it decides |
| --- | --- | --- |
| `participation` | `0.5` | how much of a measured saving reaches the client |
| `floor` | `'mid'` | the scale never goes below 0 — a quote never crosses the mid automatically |
| `scope` | `'vol'` | all v1 measures; any other value refuses rather than quoting a scope nobody looked at |
| `bucket_limit` | `None` | a cap on `\|risk after\|` per bucket, past which NO tightening applies |
| `min_ticket_bp` | `0.0` | flat bp of notional, the ops floor under the edge (crossed to the report currency on the same `FxRate` ratio everything else here reads) |
| `firm_seconds` | `600` | how long a quote stays approvable — `/book/quote` refuses a pending quote older than this, naming the age, the window and the remedy |

`firm_seconds` is the one field this module carries rather than acts on: the mandate is ONE block a desk
states, and the approval verb reads the clock against the pending file's `quoted_at`.

**Five ways out leave `scale` null with the reason named**, rather than a scale of 1 nobody can tell
from a decision: no policy; no two-way at all; NO leg readable, where `charge_full` stays null too;
a charge that is not positive; and no vol quote leaves published at all.

**The ABSENCE of the block is the off switch**, the same compatibility contract as the two-sided one: a
book declaring no policy never reaches the BOOK's greeks runs at all and its `risk.scale` is `None`
rather than `1.0` — the difference between "the feature did not run" and "it ran and decided
nothing", which a consumer auditing a quote has to be able to tell apart. A block declaring
`participation: 0` runs the WHOLE layer and lands on the identical floats.

**The scale, and the ceiling.** `charge_effective = min(charge_full, max(min_ticket, max(0,
charge_full − participation × saving)))`, and `scale = charge_effective / charge_full`. Two rulings sit
in that outer `min`. A risk-ADDING trade takes no surcharge — the market's own spread is the ceiling in
v1 — so a positive residual cost is simply no saving and `scale` is exactly 1. And the min ticket is an
ops floor UNDER the tightening, not a second ceiling over it: a ticket above the full spread leaves the
scale at 1 and the REPORTED charge at `charge_full`, rather than lifting the quote through the two-way.

**ONE pass, not a fixed point.** `scale` multiplies the charge, and the one solve that levies it targets
`−(margin + charge_effective)`, so the effective charge and the realised `edge` are now the SAME number
— measured 0.0000 on the gate's book against the 0.026% the two-reprice scheme carried. The
approximation did not go away; it moved. Both the legs' vegas and the book's buckets are read at the MID
solution while the quote sits at the charged coordinate, so what is charged is a first-order reading of
a structure one step away. The `risk` block is honest about which candidate it measured: its buckets are
the mid candidate's. Iterating to a fixed point would pay a greeks run per leg per iterate to chase it.

**What it does to a quote.** Same book, same collar, same policy, opposite sign of the standing
position: the repeat quotes a cap of 19.14881898 (the full-charge cap, to the bit) and the offset quotes
19.15266449 — 4.3% of the way back from the full charge toward the mid cap of 19.23862842.
Client-better, and never through the mid.

**The cache.** The book-alone half of the measurement does not depend on what is being quoted and moves
only when the market ticks or something books — both of which change the book's content etag — so it is
kept in a bounded module dict keyed by that etag, and a desk quoting repeatedly against a standing book
pays one greeks run per quote instead of two. Measured: a miss is **30.5 ms** and a hit **0.115 ms**,
against a cold run of 105.8 ms. Bounded at 16 entries because a book that ticks every 30s would
otherwise leak a vector per tick.

## The quote lifecycle

**The spot is live, the surface is ticked.** Before the recipe runs, `StructureJob` puts this
workstation's terminal spot onto its OWN copy of the book (`service.patch_live_spot` →
`structures.with_live_spots`, the exact inverse of `engine_spot`); the book file is never written by a
quote. Only the spot moves — the vol surface and the curves stay whatever the 30s cadence last ticked
in, which is a convention rather than a shortcut, since a delta-space FX surface is sticky-delta and
meant to be read at whatever spot is standing. Failure is fast and NAMED: one request on a 2s budget,
and a failure that reached the terminal is remembered process-wide for 30s so consecutive quotes skip
the attempt. An unprovisioned `DV_HOME`, a missing blpapi and a pair the security map never verified all
fall back the same way — never an error, never provisioning. Every outcome carries `spot`:
`{value_market, source: 'terminal'|'book', note}`, with `value_market` read back off the document the
legs actually priced against, so what is reported and what was priced cannot disagree.

`POST /book/structure` runs the recipe as one queued job and files TWO artifacts under
`DV_HOME/tmp/<quote_id>`: the pending trade (`.json` — the outcome plus the composed `StructuredDeal`,
ready to book) and its ticket (`.xlsx` via `derivus/quote_sheet.py`, the `quote` extra — legs in market
terms, the market data used, values only, no formulas, `created` pinned to the book's `Base_Date`).

TWO ids name the quoting ACT and they are not the same hash. The runner's `quote_id` — structure, params,
the netting set, the market the document was carrying, and a submission clock — names both files and is
the one an approval quotes. The service's `result_id` — book etag, structure, params, the netting set,
and its own submission clock — names the queued JOB, exactly as `/execute` does. Both carry a clock
because a quote is an ACT: two identical asks are two quotes, never one coalesced result, and a refusal
is never pinned.

`POST /book/quote` is the approval: the pending deal booked through the SAME validate-before-write seam
as any booking, refused in the same wording, against the book as it stands NOW — and the pending file
survives the booking as the audit trail. A missing `xlsxwriter` never refuses a quote; the outcome names
the install under `files['sheet_note']` instead.

The composed deal carries its legs inside the block under `Children`, and `splice_deal` lifts them onto
the node — the engine walks `node['Children']` and never inside a deal block. Before the lift lived in
the splice, a composed candidate priced as an EMPTY container, 0.0 with nothing said against it, on
every verb at once. The gate prices the legs, never the container.

## Testing shape

Every gate is a real document through the real pipeline against a financial identity, nothing
monkeypatched: the straddle equals its call plus its put; the collar, seagull and forward extra net to
zero within the solve's own residual; re-quoting the solved cap as a given strangle reproduces equal
premiums; the whole day (quote → pending file → approve → book marks it at ~0) runs over the served
book; and a hand-authored knock-IN plus knock-OUT call must be the vanilla, which is the first test to
demand a number from `pv_barrier_option`'s analytic knock-in branch (1.1e-16 relative).

**The margin is gated from both ends**, because one end alone cannot catch a sign: a collar quoted at
50,000 rand nets minus 50,000 rand at the quote's own spot on the client's paper, and the mirror of that
same deal, priced against the book and then booked through the service and marked, holds plus its dollar
value. Beside them: the strip's solved strike moving UP by 9.2e-4 relative — 37 times its estimator
noise — a quote with no margin carrying no `margin` and no `Sales_Margin` at all, a margin of zero
reproducing the zero-cost quote to the bit, and the three refusals (a currency with no rate, a recipe
with nothing to solve, an amount with no currency).

**The accrual tolerance is measured, not chosen.** A strip is Monte Carlo priced, so its zero-cost
strike is a root find over an ESTIMATOR — deterministic for a fixed seed, which is what lets `brentq`
own it, but converging on the true root only as the paths grow: the accumulator's two orientations
solve strikes 4.8e-4 apart at 1024 inner paths, 1.3e-4 at 4096, 2.5e-5 at 16384, 3.9e-5 at 65536. The
gates quote at 16384 (about a second) and allow **2e-4**. They pin the axis finding (the accumulator
solves one strike from both sides of the pair while the TARF refuses the second side by name), that
both strips net to zero and land BELOW the forward through `book_node` — not a tautology, since at a
strike of the forward the geared sold leg outweighs the bought one — and that the composed TARF booked
as a `CreditMonteCarlo` reports an exposure profile that is finite, multi-row and DISPERSED, which a
skipped deal cannot be because zero has no spread.

**The two-way is gated on what it CHARGED, not on a shift.** The sharpest of these puts `vol_risk` out
of the loop entirely: for the collar's bought put, each quoted pillar's own MID is moved by its half
both ways, the surface RE-BOOTSTRAPPED from the moved quotes and the leg repriced alone, and the
reported cost has to be that central difference within **2%**, with the side that hurts the client the
one `−sign(vega)` names. Measured across every leg of every form: 0.00% to 1.12%, and 0 sign
disagreements in 48 non-zero pillar rows.

The edge is held non-negative and equal to
the sum of the legs' own `spread_charge` over every form of every structure, on a lognormal book and on
a fitted one — twelve forms, the ten variations and the two structures that solve nothing. The four
strips are held to the vega-signed readings within 5% (+6,627.45 / +7,316.06 / +22,415.98 / +25,294.53
at a 1m USD ticket; signed by `Buy_Sell` instead, all four come back negative), and the solved strike
must move AGAINST the client in each. Each of the collar's two put legs must read a risk reversal
opposite in sign to its own ATM vega and still be charged a POSITIVE cost there. On the fitted book each
strip's `spread_source` is `'lognormal reading'` where the same book without the fit reads `'surface'`,
and its solved strike differs from the mid one — skipping that second run leaves the edge at 0.0. The
legs are proved to be at MID by valuing the composed two-sided deal against the unshifted book and
matching each leg's premium. And a book that carries a two-way but publishes no quote leaves — the
bootstrapper dropped, the surface it wrote left standing — charges NOTHING and says so per leg, while
quoting the mid quote to the float — and where such a book DECLARES a policy, so the risk step really
runs, `risk.charge_full` is null too rather than a zero nobody measured.

**The margin is gated against a policy**, because the two charges do not scale together: on a book whose
policy tightens and one whose policy does not, a quote at an agreed margin comes back at minus exactly
that margin while `edge` is `scale × charge_full`, and the booked mirror holds both. Shaving the margin
by the scale is otherwise silent — every other identity stays true. A structure that solves NOTHING is
gated from both sides of the pair: `charged_on` reads `premium`, what the client pays moves against them
by the margin and the edge together, and the booked legs still mark at mid. And a FITTED strip is booked
end to end through the service, which is the one path where the pin is load-bearing for the mark: only
once `/book/quote` merges the quote's own `valuation_configuration` does the book report the bank at
plus the margin and the edge.

The risk-impact gates are three. A book with no `Quote Policy` and one with `participation: 0` quote
float for float identically. The registry has no sell-side collar, so the opposite SIDE goes on the BOOK
rather than into the quote: one book short the trade and one long it, the offset coming out tighter
while the repeat comes out at exactly the full-charge cap, with the `RR 0.25 1` bucket read from both
sides (`before` equal and opposite, `after` doubled on one and all but zero on the other — 4.4%, the
one-pass approximation's own size, since the book holds the collar at the cap it was dealt at while the
candidate is measured at the mid one). And both limits are made to BIND on a book holding two of the
trade: a `bucket_limit` under the residual suspends the tightening and names the bucket, and a
`min_ticket_bp` inside the band the tightening opens lands the effective charge exactly on the ticket.

## V1 scope, and the named next steps

Seven structures ship, five of them with [variations](#variations): `Straddle` and `Strangle` (one form
each, and no SOLVE — the registry handles recipes that only price), `ZeroCostCollar` (`floor` given and
the cap solved to premium parity, or `cap` given and the floor solved), `Seagull` (three legs, two given
and one solved to net zero: `floor` + `lower_floor`, or `cap` + `upper_cap`), `ForwardExtra` (the
protected rate given, the BARRIER solved — protection plus a sold knock-in at the same strike, so the
client keeps the favourable move until the pair trades through the level and the structure reverts to a
plain forward at that rate: `floor` is a bought put funded by a sold up-and-in call, for a client
selling the base currency, and `cap` its mirror, a bought call funded by a sold down-and-in put), and
the two [accrual strips](#accrual) — `TargetRedemptionForward` (fixings to the tenor, a target cap in
the pair's own units, the strike solved to zero upfront) and `Accumulator` (the same bargain with a
knock-out LEVEL in place of the cap), each dealt `buy` or `sell` and neither told apart by a level, so
both take a direction.

[Risk-impact pricing v1](#risk-impact) ships with them, and its scope is named honestly: the residual is
measured in the VOL book only (`scope: 'vol'`, any other value refuses), in QUOTE coordinates off
`Quote_Sensitivity` on the `FXVolPrices` block — the per-expiry ATM vega fallback was budgeted and never
needed. It is ONE pass rather than a fixed point, and there is no surcharge past the two-way. Named
next, in order of what they exercise:

- **Incremental XVA — the v2 of the same step.** A counterparty on the quote, and the charge grows a
  second term: `CVA(book + mirror(candidate)) − CVA(book)` through the `Credit_Monte_Carlo` engine. The
  seam is already the right shape, so what v2 changes is WHICH calculation those two runs are and what
  the difference is priced at: a credit charge is the number itself rather than a risk times a
  half-spread, and it is a netting-set question rather than a bucket one.
- **A ratio-solve primitive**, for participating forwards — the recipe vocabulary's first step that
  moves a notional fraction rather than a strike.
- **A tenor-vocabulary note**: `expiry` parses `<n><D|W|M|Y>` through the job grammar's own period
  letters, or an ISO date for a broken date; anything else refuses by name.
