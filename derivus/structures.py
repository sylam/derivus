########################################################################
# Copyright (C)  Shuaib Osman (vretiel@gmail.com)
# This file is part of Derivus.
#
# Derivus is free for noncommercial use under the terms of the PolyForm
# Noncommercial License 1.0.0. You should have received a copy of the license
# along with Derivus. If not, see
# <https://polyformproject.org/licenses/noncommercial/1.0.0>.
#
# Derivus is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
########################################################################

"""What a sales desk sells, declared - and the runner that turns one into a priced quote.

A STRUCTURE is a class in this module. Its name is the registry key, and it states four things and
no logic at all:

  - `vernacular`, the names a salesperson says out loud
  - `fields`, `schema.F` descriptors for the PARAMETERS a client quotes in
  - `legs`, each naming a declared `Instrument` type plus the PARTIAL deal block the structure
    pins, and the parameter SLOTS mapped onto the rest. The Instrument store's declarations already
    ARE the leg's field schema, so a leg restates no deal fields
  - `recipe`, an ordered list of `Price` and `Solve` steps

A structure dealt MORE THAN ONE WAY declares `variations` in place of `legs` - an exporter flooring
the pair and an importer capping it are one product and two bookings - each with the side of the
pair its client buys, the parameters only it takes and its own legs. `variation_for` is one rule
for every structure: it selects the one variation everything the ticket states is consistent with.

`quote()` is the runner, and it owns every conversion, once.

MARKET AXIS vs ENGINE AXIS. A desk quotes USDZAR 15.50 - ZAR per USD - while `FXOptionDeal` prices
an option on `Underlying_Currency` settled in `Currency`. When the notional is the pair's QUOTE
currency the deal's axis is the reciprocal of the quoted one, so `Strike_Price = 1/K` and the
option sense inverts with it: a market Call is an engine Put. A barrier's `Barrier_Price` inverts
as a strike does and its DIRECTION flips with it, while In/Out never moves. When the notional is
the BASE currency the two axes agree and nothing is converted. No structure knows any of this.

An ACCRUAL leg asks the axis question once more, and one answer is that a TARGET does not cross:
it is a sum of DIFFERENCES, and `1/S - 1/K` is not the reciprocal of `S - K`, so a TARF is quoted
on the pair's BASE currency and refuses the other side by name. A LEVEL crosses exactly as a
barrier does, and a leverage is a ratio that never converts. See `furnish_accrual`.

PARAMETERS vs A DEAL. The runner fills the shared block from the parameters, then the leg's pinned
block, then its slots. `expiry` is a tenor the job grammar parses, or an ISO date for a broken
one; anything else refuses by name rather than landing on today. Every step prices ONE leg against
a deep copy of the whole book document with the deal tree emptied. `Price` is a plain base
valuation; `Solve` is `derivus.solve_deal_field`, bracketed.

A SPREAD IS QUOTED; A MID IS BOOKED. The book's `FXVol` surface is bootstrapped from
`Quoted_Market_Value` alone, while the `FXVolPrices` block beside it may carry each pillar's
`Quoted_Bid`/`Quoted_Ask` as data, and this module is that data's only reader. Every leg prices at
the MID and the two-way is a CHARGE on the coordinate the recipe solves, the shape the sales margin
already has: each leg's vega is read per QUOTED PILLAR at the mid solution and charged that
pillar's own half-spread, `sum |vega| x half`, so the side a pillar is dealt on follows the RISK
rather than the leg's label. `net_mid` is the finished legs at mid and `net` is that plus the
charge. A book carrying no two-way quotes exactly as it always has, to the bit.

THE RISK PRICES THE SPREAD. A trade's charge is the cost of hedging the RESIDUAL it leaves on the
book, at the market's own two-way. The composed candidate is MIRRORED - the verb the booking uses -
and the book's vol risk is read with it and without it in QUOTE space, off `Quote_Sensitivity`.
Each bucket's move in ABSOLUTE risk is charged that bucket's own half-spread, and `participation`
of any saving comes off the spread. The `Quote Policy` block declares the mandate and its ABSENCE
is the feature's off switch.

THE SPOT IS LIVE, THE SURFACE IS TICKED. `with_live_spots` is the inverse of `engine_spot` and the
one seam a caller writes a terminal's number through: a spot is `bind='value'` data that moves
between prints, while a delta-quoted surface is read at whatever spot is standing. Every quote says
which it used under `spot`, read off the document it priced.
"""

import copy
import json
import time

from . import utils
from . import schema
from .schema import F, REQUIRED

#: How wide the runner brackets a strike solve, as a multiple of the market spot. A vanilla's value
#: is monotone in its strike, so any bracket spanning deep in- and out-of-the-money holds the root;
#: `brentq` refuses by name where a zero-cost leg does not sit inside these ends.
STRIKE_BRACKET = (0.25, 4.0)

#: The same bracket for an ACCRUAL strike, moved in. A strip's value saturates at the low end -
#: flat at the discounted `target x notional` once every fixing redeems at once - so nothing is
#: given up, while `0.25 x spot` prices NaN off a surface quoted over moneyness [0.8, 1.2].
ACCRUAL_BRACKET = (0.5, 2.0)

#: How far a re-solve looks either side of a root it already has, as a fraction of it. A charge
#: moves a coordinate by a spread's width - measured under half a percent on every form here - so
#: this is four times the move it has to hold, and a bracket that does not straddle falls back.
SEED_BRACKET = 0.02

#: A fixing settles on its own spot value date, two days on. CALENDAR days rather than business:
#: the runner holds no calendar, and a settlement date is a cashflow date rather than an
#: observation, so a weekend costs two days of discounting and nothing else.
FIXING_LAG = 2

#: The accrual deals a leg may name beside the two vanilla ones. Each carries a fixing SCHEDULE the
#: runner grows from the tenor, rather than the single expiry a vanilla is struck to.
ACCRUAL_DEALS = ('FXTARFOptionDeal', 'FXAccumulatorOptionDeal')

#: Where each accrual deal files that schedule. The field name is the deal's own; the ROW is one
#: shape either way - `[fixing date, settlement date, observed fixing]`, untagged.
SCHEDULE_FIELD = {'FXTARFOptionDeal': 'TARF_ExpiryDates',
                  'FXAccumulatorOptionDeal': 'Accumulator_ExpiryDates'}

#: `<SpotModel>ModelParameters.<the pair's key>`, the naming convention
#: `get_spot_model_params_factor` resolves the parameters by (`utils.spot_model_currency` picks the
#: token) - so the presence check here and the engine's own lookup are one key.
SPOT_MODEL_FACTOR = '{}ModelParameters.{}'

#: The model an accrual leg is priced under WHERE THE BOOK CARRIES A CALIBRATION. The switch lives
#: in `Valuation Configuration` per deal TYPE, not on a deal, and both accrual deals declare it in
#: their own `spot_models`.
SPOT_MODEL = 'LogVar2FJ'

#: Every vanilla leg is European. Pinned per leg rather than injected by the runner: it is an
#: `FXOptionDeal` field, and an `FXBarrierOption` declares no such field.
VANILLA = {'Option_Style': 'European'}

#: Where a pair's two-way lives: the `Market Prices` block a surface is filed under, which is the
#: leg's own `FX_Volatility` name with the family in front of it. The price factor is `FXVol.<n>`.
FX_VOL_PRICES = 'FXVolPrices.{}'
FX_VOL_FACTOR = 'FXVol.{}'

#: Where a desk's quoting MANDATE lives: a section of the JOB, beside `Calculation` and
#: `MergeMarketData`, not inside `ExplicitMarketData` - `Context.load_json` raises `KeyError` on a
#: section `Config` does not declare. Every reader of a job walks `Calc` by name, so an unknown key
#: there travels through load, pricing and the book file untouched.
QUOTE_POLICY = 'Quote Policy'

#: What the policy means where the block is silent, read with `.get` so a desk states only what it
#: is changing. The ABSENCE OF THE BLOCK is the off switch, not these values. `firm_seconds` is the
#: one field this module does not act on - the approval verb reads that clock.
POLICY_DEFAULTS = {'participation': 0.5, 'floor': 'mid', 'scope': 'vol',
                   'bucket_limit': None, 'min_ticket_bp': 0.0, 'firm_seconds': 600}

#: `min_ticket_bp` is bp of notional, and a bp is this.
BASIS_POINT = 1e-4

#: The book-alone risk vector by the book's own content etag, bounded. The book's risk moves only
#: when the market ticks or something books, and both change the etag, so a repeat quote on a
#: standing book pays one greeks run instead of two. Bounded because the dict would otherwise leak.
RISK_CACHE = {}
RISK_CACHE_LIMIT = 16

#: What a leg says where neither reading of it publishes a quote sensitivity: there is no vega to
#: price the two-way against, so nothing was charged on it and the row says so rather than a zero.
NO_VEGA = ('the two-way could not reach this leg - neither the surface nor a lognormal reading of '
           'it publishes an FX vol quote sensitivity, so no spread was charged on it')

#: A barrier's DIRECTION is a statement about the PAIR, so it crosses to the engine axis with the
#: strike: a barrier above USDZAR 18.50 is below 1/18.50 dollars per rand. In/Out says what the
#: payoff does on touch and means the same on either axis, so only Up/Down moves.
BARRIER_FLIP = {'Up_And_In': 'Down_And_In', 'Down_And_In': 'Up_And_In',
                'Up_And_Out': 'Down_And_Out', 'Down_And_Out': 'Up_And_Out'}

#: The option SENSE, read twice: crossing to the engine axis (a market Call is the right to buy the
#: base currency, so it is a Put on the quote one) and reflecting a variation into the trade the
#: other side of the pair deals.
OPTION_FLIP = {'Call': 'Put', 'Put': 'Call'}

#: The other side of the pair, which is the side a variation's mirror image serves.
OPPOSITE = {'base': 'quote', 'quote': 'base'}

#: The parameters every FX structure quotes in, shared as module constants for the reason the
#: schema's field groups are: a copy per class is a copy that drifts.
PAIR = F('pair', 'Text', default=REQUIRED,
         description='The market pair, base then quote - USDZAR is ZAR per USD')
EXPIRY = F('expiry', 'Period', default=REQUIRED,
           description="Tenor from the book's Base_Date - 3M, 1Y - or an ISO date for a broken one")
NOTIONAL = F('notional', 'Float', default=REQUIRED,
             description='The amount, in notional_currency, each leg is struck on')
NOTIONAL_CURRENCY = F('notional_currency', 'Text', default=REQUIRED,
                      description='Which side of the pair the notional is in; it becomes the '
                                  'option underlying, and naming the quote currency is what '
                                  'inverts the strike axis')

#: WHICH WAY a structure with more than one variation is dealt, stated as the client's own two
#: cashflows. They are SELECTORS rather than parameters - they choose a form instead of filling a
#: leg - so they are optional to state and never defaulted: `variation_for` reads them beside the
#: level a ticket names, which selects on its own wherever the variations differ in one.
SELL_CURRENCY = F('sell_currency', 'Text',
                  description='Currency the client sells: one side of pair')
BUY_CURRENCY = F('buy_currency', 'Text',
                 description='Currency the client buys: the other side of pair')
SELECTORS = (SELL_CURRENCY, BUY_CURRENCY)

#: What a strike-like parameter means, said once. A structure's strikes are the client's numbers.
MARKET_STRIKE = 'In MARKET terms, as the pair is quoted (USDZAR 15.50)'


def strike(name, description):
    """A strike-like parameter: a market-terms number the runner converts to the engine axis."""
    return F(name, 'Float', default=REQUIRED,
             description='{}. {}'.format(description, MARKET_STRIKE))


class Leg(object):
    """One named leg of a structure: a declared `Instrument` type, what the structure PINS on it,
    and which parameter fills each remaining slot.

    `pinned` is a partial deal block - only the fields the structure itself decides. `Option_Type`
    is pinned on the PAIR (a Call is the right to buy the base currency at the strike) and the
    runner puts it on the engine's axis along with the strike. `slots` maps a DEAL field to a
    PARAMETER name, so `{'Strike_Price': 'floor'}` strikes this leg at what the client called the
    floor. A field named by neither takes the shared block's or the Instrument declaration's value.
    """
    __slots__ = ('role', 'deal_type', 'pinned', 'slots')

    def __init__(self, role, deal_type, pinned=None, slots=None):
        self.role = role
        self.deal_type = deal_type
        self.pinned = dict(pinned or {})
        self.slots = dict(slots or {})

    def descriptor(self):
        """This leg as a `mapping['Structure'][...]['legs']` entry."""
        return {'deal_type': self.deal_type, 'pinned': dict(self.pinned), 'slots': dict(self.slots)}


class Variation(object):
    """One way a structure is dealt: whose side of the pair it serves, the parameters only it
    takes, and its legs.

    A forward extra is one structure and two BOOKINGS - an exporter flooring the pair and an
    importer capping it - so a structure dealt more than one way declares `variations` in place of
    `legs` and keeps `fields` for what every form shares. `buys` is 'base' or 'quote', the side the
    CLIENT buys, which is what a stated direction selects on; a variation's own `fields` are
    required exactly as the shared ones are.
    """
    __slots__ = ('buys', 'fields', 'legs')

    def __init__(self, buys, fields, legs):
        self.buys, self.fields, self.legs = buys, list(fields), list(legs)

    def descriptor(self):
        """This variation as a `mapping['Structure'][...]['variations']` entry."""
        return {'buys': self.buys,
                'fields': {f.key: f.descriptor() for f in self.fields},
                'legs': {leg.role: leg.descriptor() for leg in self.legs}}


class Premium(object):
    """The premium of legs already priced, as a solve TARGET.

    `Premium('protection')` is that leg's own value; the arithmetic (`-Premium('a')`,
    `Premium('a') + Premium('b')`) makes a financing leg's target the negative of everything bought
    so far, which is what "zero cost" means once a sold leg's value carries its own sign.
    """
    __slots__ = ('terms',)

    def __init__(self, *roles, **kwargs):
        terms = kwargs.pop('terms', None)
        if kwargs:
            raise TypeError('Premium takes roles and terms, not {}'.format(sorted(kwargs)))
        self.terms = dict(terms) if terms else {role: 1.0 for role in roles}

    def __neg__(self):
        return Premium(terms={role: -weight for role, weight in self.terms.items()})

    def __add__(self, other):
        terms = dict(self.terms)
        for role, weight in other.terms.items():
            terms[role] = terms.get(role, 0.0) + weight
        return Premium(terms=terms)

    def value(self, premiums):
        """This combination against `{role: premium}`. A role not yet priced refuses by name: a
        recipe targeting a leg it has not reached is mis-ordered, not empty."""
        missing = sorted(set(self.terms) - set(premiums))
        if missing:
            raise ValueError('{} is targeted before it is priced'.format(', '.join(missing)))
        return sum(weight * premiums[role] for role, weight in self.terms.items())

    def __str__(self):
        parts = ['{}Premium({}){}'.format(
            '-' if weight < 0 else '+', role,
            '' if abs(weight) == 1.0 else ' * {:g}'.format(abs(weight)))
            for role, weight in self.terms.items()]
        return ' '.join(parts).lstrip('+')


class Price(object):
    """Value one leg as it stands - a base valuation of that leg alone against the book."""
    __slots__ = ('role',)

    def __init__(self, role):
        self.role = role

    def describe(self):
        return 'Price {}'.format(self.role)


class Solve(object):
    """Move one field of one leg until that leg's own value lands on `target`.

    `target` is a float literal or a `Premium` combination over legs the recipe has already priced.
    The field is the DEAL's, so a strike solve moves `Strike_Price` on the engine axis and the
    runner reports the market reading beside it.
    """
    __slots__ = ('role', 'field', 'target')

    def __init__(self, role, field, target):
        self.role = role
        self.field = field
        self.target = target

    def describe(self):
        return 'Solve {}.{} to {}'.format(
            self.role, self.field,
            str(self.target) if isinstance(self.target, Premium) else '{:g}'.format(self.target))


def reflected(variation, rename, fields):
    """`variation`'s mirror image: the same structure dealt for the other side of the pair.

    Every leg's `Option_Type` swaps and every `Barrier_Type` crosses through `BARRIER_FLIP`, a
    level said about the pair reading the other way round for a client standing the other side of
    it. Two things do NOT move: In and Out describe what the payoff does on touch and mean the same
    to either client, and `Buy_Sell` is the CLIENT's side on both sheets - an importer buys their
    protection exactly as an exporter buys theirs - so paper becoming the bank's position stays
    `mirror`'s one seam.

    `rename` is the variation's own parameters under their mirror names (`{'floor': 'cap'}`),
    applied to the leg slots that fill from them, and `fields` is those mirror parameters as the
    declarer writes them: prose says what a level MEANS to the client on that side, which is not
    something a rename can derive.
    """
    legs = []
    for leg in variation.legs:
        pinned = dict(leg.pinned)
        for field, flip in (('Option_Type', OPTION_FLIP), ('Barrier_Type', BARRIER_FLIP)):
            if field in pinned:
                pinned[field] = flip[pinned[field]]
        legs.append(Leg(leg.role, leg.deal_type, pinned,
                        {field: rename.get(slot, slot) for field, slot in leg.slots.items()}))
    return Variation(OPPOSITE[variation.buys], fields, legs)


class Straddle:
    """Both wings at one strike, both bought - the way volatility itself is traded."""
    vernacular = 'straddle, at-the-money volatility, vol trade'
    fields = [PAIR, EXPIRY, NOTIONAL, NOTIONAL_CURRENCY,
              strike('strike', 'The one strike both wings are struck at')]
    legs = [Leg('call', 'FXOptionDeal', dict(VANILLA, Option_Type='Call', Buy_Sell='Buy'),
                {'Strike_Price': 'strike'}),
            Leg('put', 'FXOptionDeal', dict(VANILLA, Option_Type='Put', Buy_Sell='Buy'),
                {'Strike_Price': 'strike'})]
    recipe = [Price('call'), Price('put')]


class Strangle:
    """The straddle's wings pulled apart: both bought, each at its own strike, so the client pays
    less and needs a bigger move. Both strikes are the client's - nothing is solved."""
    vernacular = 'strangle, wide straddle, bought cylinder'
    fields = [PAIR, EXPIRY, NOTIONAL, NOTIONAL_CURRENCY,
              strike('floor', 'The lower strike, bought as a put on the pair'),
              strike('cap', 'The upper strike, bought as a call on the pair')]
    legs = [Leg('floor', 'FXOptionDeal', dict(VANILLA, Option_Type='Put', Buy_Sell='Buy'),
                {'Strike_Price': 'floor'}),
            Leg('cap', 'FXOptionDeal', dict(VANILLA, Option_Type='Call', Buy_Sell='Buy'),
                {'Strike_Price': 'cap'})]
    recipe = [Price('floor'), Price('cap')]


class ZeroCostCollar:
    """Protection paid for by giving up the other side. The client names the level they want; the
    other one is whatever strike makes the sold wing fund the bought one exactly, which is why it
    is solved rather than quoted.

    An exporter floors the pair and an importer caps it, so the level the ticket names IS which of
    the two is being dealt.
    """
    vernacular = 'zero-cost collar, range forward, cylinder'
    fields = [PAIR, EXPIRY, NOTIONAL, NOTIONAL_CURRENCY, SELL_CURRENCY, BUY_CURRENCY]
    variations = {'floor': Variation(
        'quote', [strike('floor', 'The protected level; the cap is solved to fund it')],
        [Leg('protection', 'FXOptionDeal', dict(VANILLA, Option_Type='Put', Buy_Sell='Buy'),
             {'Strike_Price': 'floor'}),
         Leg('financing', 'FXOptionDeal', dict(VANILLA, Option_Type='Call', Buy_Sell='Sell'))])}
    variations['cap'] = reflected(
        variations['floor'], {'floor': 'cap'},
        [strike('cap', 'The protected level; the floor is solved to fund it')])
    recipe = [Price('protection'),
              Solve('financing', 'Strike_Price', -Premium('protection'))]


class Seagull:
    """A collar cheapened by selling a second wing. The client names the level they want protected
    and the one past which they are willing to be unprotected again; the third strike is solved so
    the three legs sum to nothing."""
    vernacular = 'seagull, three-way, participating collar'
    fields = [PAIR, EXPIRY, NOTIONAL, NOTIONAL_CURRENCY, SELL_CURRENCY, BUY_CURRENCY]
    variations = {'floor': Variation(
        'quote', [strike('floor', 'The protected level; the cap is solved against it'),
                  strike('lower_floor', 'Where protection stops, sold back against the floor')],
        [Leg('protection', 'FXOptionDeal', dict(VANILLA, Option_Type='Put', Buy_Sell='Buy'),
             {'Strike_Price': 'floor'}),
         Leg('participation', 'FXOptionDeal', dict(VANILLA, Option_Type='Put', Buy_Sell='Sell'),
             {'Strike_Price': 'lower_floor'}),
         Leg('financing', 'FXOptionDeal', dict(VANILLA, Option_Type='Call', Buy_Sell='Sell'))])}
    variations['cap'] = reflected(
        variations['floor'], {'floor': 'cap', 'lower_floor': 'upper_cap'},
        [strike('cap', 'The protected level; the floor is solved against it'),
         strike('upper_cap', 'Where protection stops, sold back against the cap')])
    recipe = [Price('protection'), Price('participation'),
              Solve('financing', 'Strike_Price', -Premium('protection', 'participation'))]


class ForwardExtra:
    """A zero-cost forward extra, in the client's stated cashflow direction.

    A client selling the pair's quote currency and buying its base currency caps the pair: they buy
    a call and fund it by selling a down-and-in put. A client selling the base and buying the quote
    floors the pair: they buy a put and fund it by selling an up-and-in call. Once the sold wing
    knocks in, either form reverts to a forward at the named cap or floor.
    """
    vernacular = 'forward extra, forward plus, at-worst forward'
    fields = [PAIR, EXPIRY, NOTIONAL, NOTIONAL_CURRENCY, SELL_CURRENCY, BUY_CURRENCY]
    variations = {'floor': Variation(
        'quote', [strike('floor', 'The floor: the rate the client is protected at, and the '
                                  'forward the structure reverts to once the barrier trades')],
        [Leg('protection', 'FXOptionDeal', dict(VANILLA, Option_Type='Put', Buy_Sell='Buy'),
             {'Strike_Price': 'floor'}),
         Leg('reversion', 'FXBarrierOption',
             {'Option_Type': 'Call', 'Buy_Sell': 'Sell', 'Barrier_Type': 'Up_And_In'},
             {'Strike_Price': 'floor'})])}
    variations['cap'] = reflected(
        variations['floor'], {'floor': 'cap'},
        [strike('cap', 'The cap: the rate the client is protected at, and the forward the '
                       'structure reverts to once the barrier trades')])
    recipe = [Price('protection'),
              Solve('reversion', 'Barrier_Price', -Premium('protection'))]


class TargetRedemptionForward:
    """A better rate than the forward at every fixing, bought with gearing and a redemption cap.

    At each fixing to the tenor the client deals `notional` at the solved strike: they accrue the
    whole of a favourable move and take `leverage` times the notional on an unfavourable one, and
    the strip ends the moment their cumulative accrual reaches `target`. The STRIKE is therefore
    the solved coordinate and the premium is zero.

    ONE leg, because the deal itself is the strip: `FXTARFOptionDeal` prices every fixing, the
    knock-out and the partial accrual that fills the target, so the recipe has no `Price` step.

    `notional` is PER FIXING here - a 1M notional on a 1M-fixing 1Y TARF deals a million twelve
    times - and always in the pair's BASE currency, since a target has no reading on the reciprocal
    axis. `furnish_accrual` refuses the other side by name.

    It is dealt BOTH WAYS, and neither is a level a ticket could name it by: the client buying the
    base at every fixing accrues as the pair rises and the one selling it accrues as the pair
    falls, so the direction is the only thing that tells the two apart and a TARF states one.
    """
    vernacular = 'tarf, target redemption forward, target forward'
    fields = [PAIR, EXPIRY, NOTIONAL, NOTIONAL_CURRENCY, SELL_CURRENCY, BUY_CURRENCY,
              F('fixing_frequency', 'Period', default=REQUIRED,
                description='How often the strip fixes - 1M, 3M - counted off the book\'s '
                            'Base_Date to the tenor'),
              F('target', 'Float', default=REQUIRED,
                description='The accrual cap that redeems the strip, in the PAIR\'s own units - '
                            '1.5 on USDZAR is 1.50 rand of cumulative favourable move'),
              F('leverage', 'Float', default=2.0,
                description='The loss-side gearing: how many notionals the client deals on an '
                            'unfavourable fixing, against one on a favourable one')]
    variations = {'buy': Variation('base', [], [
        Leg('tarf', 'FXTARFOptionDeal',
            {'Option_Type': 'Call', 'Buy_Sell': 'Buy',
             'Settlement_Style': 'Cash', 'Option_Style': 'European'},
            {'TargetLevel': 'target'})])}
    variations['sell'] = reflected(variations['buy'], {}, [])
    recipe = [Solve('tarf', 'Strike_Price', 0.0)]


class Accumulator:
    """The same bargain with a LEVEL instead of a cap: accumulate at a better-than-forward strike
    until the pair trades through the knock-out.

    The client deals `notional` at each fixing at the solved strike, geared `leverage` times against
    them on an unfavourable one, and the whole strip cancels at the first fixing that observes the
    pair at or beyond `knockout`. That knock-out is a LEVEL on the spot, so it crosses to the engine
    axis exactly as a barrier does and an accumulator quotes from either side of the pair.

    The knock-out is observed ON THE FIXING DATES rather than continuously -
    `FXAccumulatorOptionDeal`'s own declaration.

    The DECUMULATOR is the same strip sold: the client deals the base away at each fixing, accrues
    as the pair falls and knocks out below. Both forms name their level `knockout`, so like the
    TARF it is the DIRECTION that says which is being dealt.
    """
    vernacular = 'accumulator, decumulator, accumulator forward, accu'
    fields = [PAIR, EXPIRY, NOTIONAL, NOTIONAL_CURRENCY, SELL_CURRENCY, BUY_CURRENCY,
              F('fixing_frequency', 'Period', default=REQUIRED,
                description='How often the strip fixes - 1M, 3M - counted off the book\'s '
                            'Base_Date to the tenor'),
              strike('knockout', 'The level that cancels the strip when a fixing observes it'),
              F('leverage', 'Float', default=2.0,
                description='The loss-side gearing: how many notionals the client deals on an '
                            'unfavourable fixing, against one on a favourable one')]
    variations = {'buy': Variation('base', [], [
        Leg('accumulator', 'FXAccumulatorOptionDeal',
            {'Option_Type': 'Call', 'Buy_Sell': 'Buy', 'Barrier_Type': 'Up_And_Out'},
            {'Barrier_Price': 'knockout'})])}
    variations['sell'] = reflected(variations['buy'], {}, [])
    recipe = [Solve('accumulator', 'Strike_Price', 0.0)]


def registry():
    """`{name: class}` for every structure declared here - the same scan `emit_structures` makes.

    A structure IS a class in this module carrying `vernacular`, so the key is the class name.
    """
    return {name: cls for name, cls in globals().items()
            if isinstance(cls, type) and 'vernacular' in vars(cls)}


def structure_named(name):
    """The structure class `name` refers to, or a refusal carrying the roster."""
    found = globals().get(name)
    if not (isinstance(found, type) and 'vernacular' in vars(found)):
        raise ValueError('{!r} is not a structure - the roster is {}'.format(
            name, ', '.join(sorted(registry()))))
    return found


def split_pair(pair):
    """A quoted pair as `(base, quote)`. 'USDZAR', 'USD/ZAR' and 'USD.ZAR' (the factor's own
    spelling) are the same pair; anything that is not two three-letter codes refuses, because
    guessing the split mis-books a trade."""
    text = str(pair).strip()
    codes = text.replace('.', '/').split('/') if '/' in text or '.' in text else [text[:3], text[3:]]
    if len(codes) != 2 or not all(len(c) == 3 and c.isalpha() for c in codes):
        raise ValueError('{!r} is not a quoted pair - USDZAR or USD/ZAR'.format(pair))
    return codes[0].upper(), codes[1].upper()


def timestamp(value):
    """A wire-form date as a `pd.Timestamp` - `{'.Timestamp': ...}`, a plain ISO string, or one
    already parsed."""
    import pandas as pd
    if isinstance(value, dict) and '.Timestamp' in value:
        value = value['.Timestamp']
    return pd.Timestamp(value)


def expiry_date(base_date, expiry):
    """`Base_Date` plus a quoted tenor, as the wire form a deal's `Expiry_Date` carries.

    A tenor is read through `utils.parse_period`, the grammar a job's date grid is read with, so
    the letters mean here what they mean there. An ISO date passes through for a broken date, and
    anything else refuses by name - an unparsed tenor landing on the base date is a zero-day
    option. So does a tenor that lands ON or BEFORE the base date: nothing is left to price and
    every premium the recipe quotes would be a premium for no optionality.
    """
    import pandas as pd
    try:
        offset = utils.parse_period(str(expiry))
    except ValueError:
        offset = None
    if offset is not None:
        expires = pd.Timestamp(base_date) + offset
    else:
        try:
            expires = pd.Timestamp(expiry)
        except (ValueError, TypeError):
            raise ValueError('{!r} is not a tenor (3M, 1Y) or a date'.format(expiry))
    if expires <= pd.Timestamp(base_date):
        raise ValueError('an expiry of {!r} is {} - on or before the base date {}, where there is '
                         'no optionality left to quote'.format(
                             expiry, expires.strftime('%Y-%m-%d'),
                             pd.Timestamp(base_date).strftime('%Y-%m-%d')))
    return {'.Timestamp': expires.strftime('%Y-%m-%d')}


def fixing_grid(base_date, expiry, frequency):
    """An accrual deal's fixing SCHEDULE, in the wire form both declarations read.

    `[[fixing, settlement, observed], ...]` - the row shape `TARF_ExpiryDates` and
    `Accumulator_ExpiryDates` share, untagged, with the observed fixing 0.0 because a quote is
    struck today. Fixings run from the book's `Base_Date` at `frequency` up to and including the
    tenor; each settles `FIXING_LAG` days later.

    Each fixing is `base + n x frequency` rather than a step off the previous one, an offset
    applied repeatedly from a month end walking (31 Jan + 1M + 1M is 28 Mar, not 31 Mar).

    A tenor holding no whole fixing period refuses rather than returning an empty strip, and so
    does a frequency that does not DIVIDE the tenor - the strip would end short of the tenor that
    was quoted while the ticket still said 1Y.

    The base is normalized to MIDNIGHT first, as `expiry_date` normalizes its answer: a `Base_Date`
    carrying a time would put the final fixing one comparison past a midnight last date.
    """
    import pandas as pd
    try:
        period = utils.parse_period(str(frequency)).kwds
    except ValueError:
        raise ValueError('{!r} is not a fixing frequency - 1M, 3M, 1W'.format(frequency))
    base_date = pd.Timestamp(base_date).normalize()
    last, rows, step = timestamp(expiry_date(base_date, expiry)), [], 1
    while True:
        fixing = base_date + pd.DateOffset(**{unit: n * step for unit, n in period.items()})
        if fixing > last:
            break
        rows.append([{'.Timestamp': fixing.strftime('%Y-%m-%d')},
                     {'.Timestamp': (fixing + pd.DateOffset(days=FIXING_LAG)).strftime('%Y-%m-%d')},
                     0.0])
        step += 1
    if not rows:
        raise ValueError('a {} tenor holds no {} fixing - the strip would be empty and the '
                         'structure would price at nothing'.format(expiry, frequency))
    if timestamp(rows[-1][0]) != last:
        raise ValueError(
            'a {0} tenor is not a whole number of {1} periods - the last fixing would be {2} '
            'rather than {3}, so the strip would end SHORT of the {0} that was quoted and the '
            'deal\'s Expiry_Date and its two-way spread would both be read off {2}. Quote a '
            'fixing frequency that divides the tenor, or quote the broken expiry {2} '
            'directly'.format(expiry, frequency, rows[-1][0]['.Timestamp'],
                              last.strftime('%Y-%m-%d')))
    return rows


def stated(params, key):
    """Whether a ticket STATES a parameter - ONE predicate, for selecting a variation and for
    completing the parameters alike.

    A blank is not a statement. A front end round-tripping an unfilled Float sends 0.0 and the
    store publishes `"value": ""` for every required field, so a level arriving empty is a level
    nobody named rather than a level contradicting the one that was; and a rate of zero is not a
    rate. Two predicates twenty lines apart make a ticket stated for one and unstated for the other.
    """
    return bool(params.get(key))


def client_buys(params, side):
    """Which side of the pair the ticket says the CLIENT buys - 'base', 'quote', or `None` where it
    states no direction at all.

    A stated currency is one of the pair's two sides or it is not a direction on this pair, and the
    two stated together ARE the two sides: buying and selling one currency is not a trade.
    """
    sides = {currency: name for name, currency in side.items()}
    buys = None
    for field, bought in ((BUY_CURRENCY, True), (SELL_CURRENCY, False)):
        if not stated(params, field.key):
            continue
        # read as `split_pair` reads the pair it is checked against, off the same ticket
        currency = str(params[field.key]).strip().upper()
        if currency not in sides:
            raise ValueError('{} {!r} is not a side of {}{} - the client deals the two currencies '
                             'of the pair being quoted'.format(
                                 field.key, currency, side['base'], side['quote']))
        named = sides[currency] if bought else OPPOSITE[sides[currency]]
        if buys is not None and buys != named:
            raise ValueError('a client cannot buy {0} and sell {0} - buy_currency and '
                             'sell_currency are the two sides of {1}{2}'.format(
                                 currency, side['base'], side['quote']))
        buys = named
    return buys


def variation_for(structure, params):
    """Which variation of `structure` a ticket means, as `(name, Variation)` - `(None, None)` for a
    structure declaring one form only.

    ONE rule for every structure: the variation is the UNIQUE one consistent with everything the
    ticket states. A stated direction fixes which side of the pair the client buys and admits the
    variations serving it; a stated parameter that is some variation's OWN admits the variations
    declaring it. So a level word alone selects where the forms differ in one, a direction alone
    selects where they do not, and stating both means stating them consistently. NOTHING consistent
    refuses naming what contradicts what and MORE than one refuses naming what to state, a
    structure quoted on the desk's guess of which way round it went being a booking nobody agreed.
    """
    variations = getattr(structure, 'variations', None)
    if not variations:
        return None, None
    side = dict(zip(('base', 'quote'), split_pair(params['pair'])))
    buys = client_buys(params, side)
    own = {name: {f.key for f in variation.fields} for name, variation in variations.items()}
    given = {key for keys in own.values() for key in keys if stated(params, key)}
    found = [name for name, variation in variations.items()
             if (buys is None or variation.buys == buys) and given <= own[name]]
    if len(found) == 1:
        return found[0], variations[found[0]]
    if found:
        levels = sorted(set().union(set(), *(own[name] for name in found)))
        raise ValueError('{}: state {}which currency the client buys, {} or {} - it deals as '
                         '{}'.format(structure.__name__,
                                     'the {}, or '.format(' or the '.join(levels)) if levels
                                     else '', side['base'], side['quote'], ' or '.join(found)))
    if buys is None:
        raise ValueError('{}: no variation of it states {} together - {}'.format(
            structure.__name__, ' and '.join(sorted(given)),
            ', '.join('{} states {}'.format(name, ', '.join(sorted(keys)) or 'no level of its own')
                      for name, keys in own.items())))
    dealt = [name for name, variation in variations.items() if variation.buys == buys]
    takes = sorted(set().union(set(), *(own[name] for name in dealt)))
    raise ValueError(
        '{}: a client selling {} and buying {} deals {}, which states {}, not {}. buy_currency and '
        'sell_currency are the CLIENT\'s own side of the trade - not the desk\'s, and not the side '
        'the notional is quoted in'.format(
            structure.__name__, side[OPPOSITE[buys]], side[buys],
            ' or '.join(dealt) or 'nothing at all', ', '.join(takes) or 'no level of its own',
            ', '.join(sorted(given.difference(takes)))))


def declared(structure, params):
    """`params` completed by the structure's OWN declared defaults, or a refusal naming what is
    missing beside everything the structure takes.

    Almost every parameter is `REQUIRED` - a strike a client did not name is a strike nobody
    agreed - and the SELECTED variation's own are required exactly as the shared ones are, here,
    before a quote is hashed or a leg is built. A market CONVENTION is the exception, and the
    number belongs on the `F` descriptor where `describe_structure` publishes it rather than in a
    `.get` inside the runner. A SELECTOR is neither: it chooses a form rather than filling a leg,
    so it is never defaulted into the parameters the quote reports back.

    A parameter NO form of the structure takes refuses by name against the roster it does: a
    gearing misspelt is a strip quoted at the default gearing nobody agreed, and silence is what
    makes that invisible. A selector is exempt only where the structure DECLARES one - there is
    nothing to select on a structure dealt one way. One belonging to another VARIATION is not
    unknown: stated, it is the selection rule's to refuse in its own words; blank, it is the slot a
    front end sent back and it is DROPPED, so it rides neither the parameters the quote reports,
    nor the id they are hashed into, nor the pending file, nor the ticket.
    """
    fields = list(structure.fields)
    missing = [f.key for f in fields if f.default is REQUIRED and not stated(params, f.key)]
    if not missing:
        variation = variation_for(structure, params)[1]
        fields += variation.fields if variation else []
        missing = [f.key for f in fields if f.default is REQUIRED and not stated(params, f.key)]
    if missing:
        raise ValueError('{} states no {} - a structure is quoted on its declared parameters, and '
                         'these carry no default'.format(structure.__name__, ', '.join(missing)))
    others = {f.key for form in getattr(structure, 'variations', {}).values()
              for f in form.fields}.difference(f.key for f in fields)
    unknown = sorted(set(params).difference(others, (f.key for f in fields)))
    if unknown:
        raise ValueError('{} takes no {} - it is quoted on {}'.format(
            structure.__name__, ', '.join(unknown),
            ', '.join(sorted(others.union(f.key for f in fields)))))
    declarations = {f.key: f.default for f in fields
                    if f.default is not REQUIRED and f not in SELECTORS}
    return {key: value for key, value in dict(declarations, **params).items()
            if stated(params, key) or key not in others}


def spot_model(document, deal_type, underlying, settlement):
    """Pin `SPOT_MODEL` on `deal_type` where THIS book carries a calibration for the leg's pair.
    Returns the leg's note, or `None` when the model was pinned.

    The switch is a `Valuation Configuration` entry per deal TYPE, and the parameters resolve by
    naming convention off the pair's NON-BASE token - `utils.spot_model_currency`, the rule the
    engine's own lookup takes - so the check here and that lookup are one key. It has to be made
    here: the switch on with the factor absent raises inside the engine's dependency loop, which
    SKIPS the deal and logs an ERROR, so the quote would return its only leg priced at nothing.

    A CROSS is keyed `<later>.<earlier>` alphabetically, one law per pair of currencies however
    a desk spells its surface. A book declaring no `Base_Currency` REFUSES rather
    than guessing a token, since the engine would then look the switch up under the other name.

    THE BOOK'S OWN SWITCH WINS. A book that already declares `SpotModel` for this deal type is
    marked under that family whatever the runner would have pinned, so the presence check and the
    note are taken on the family the ENGINE will look up - a leg priced under a book's own pin and
    noted as GBM is a note disagreeing with the number beside it.

    Writes IN PLACE on the document the runner holds; every pricing deep-copies it through `alone`,
    so one write reaches every iterate of the solve.
    """
    factors = market_data(document)
    market = document['Calc']['MergeMarketData']['ExplicitMarketData']
    standing = (market.get('Valuation Configuration', {}).get(deal_type) or {}).get('SpotModel')
    model = standing or SPOT_MODEL
    token = utils.spot_model_currency(underlying, settlement, base_currency(document))
    factor = SPOT_MODEL_FACTOR.format(model, token)
    if factor in factors:
        market.setdefault('Valuation Configuration', {}).setdefault(
            deal_type, {})['SpotModel'] = model
        return None
    return ('{} - a spot model is keyed off the pair, so this leg looked up {}, '
            'which this book does not carry. Calibrate {} under {} (/book/model)'.format(
                'SKIPPED, this book pinning {} itself'.format(standing) if standing
                else 'priced GBM', factor, token, model))


def pinned_models(document):
    """The `Valuation Configuration` a quote's own passes pinned on this document, or `None`.

    `spot_model` writes the switch onto the document the runner holds; this reads it back so the
    outcome can REPORT it. A leg priced under a spot model that books into a book marking it GBM is
    a mark disagreeing with the price it was dealt at, and the approval is where that is fixed.
    """
    pinned = document.get('Calc', {}).get('MergeMarketData', {}).get(
        'ExplicitMarketData', {}).get('Valuation Configuration')
    return copy.deepcopy(pinned) if pinned else None


def pin_models(document, deal, pinned):
    """`pinned` merged into a document's `Valuation Configuration`, per deal TYPE, in place.

    The booking half of `pinned_models`: an approval carries the quote's own pins onto the BOOK, so
    a leg dealt under a spot model re-marks under it. Merged per type and per key rather
    than assigned, because the block is the whole book's and a quote owns only what it pinned.

    REFUSES where a pinned model's parameters are no longer on the book. A switch pinned over a
    factor since dropped raises inside the engine's dependency loop - the deal skipped, an ERROR
    logged, the trade marked at nothing - which is the outcome this pin exists to prevent.
    """
    factors = market_data(document)
    base = base_currency(document)
    for leg in deal.get('Children') or []:
        block = leg['Instrument']['.Deal']
        entry = pinned.get(block.get('Object'))
        if not entry:
            continue
        # the SAME token rule the engine's lookup takes, or the approval checks a key the engine
        # will not ask for
        token = utils.spot_model_currency(
            block['Underlying_Currency'], block['Currency'], base)
        factor = SPOT_MODEL_FACTOR.format(entry['SpotModel'], token)
        if factor not in factors:
            raise ValueError(
                'this quote was priced under {} and the book no longer carries {} - booking the '
                'switch would skip the deal at the next valuation and mark it at nothing. Re-run '
                'the {} calibration for {}, or re-quote'.format(
                    entry['SpotModel'], factor, entry['SpotModel'], token))
    configuration = document['Calc']['MergeMarketData']['ExplicitMarketData'].setdefault(
        'Valuation Configuration', {})
    for deal_type, entry in pinned.items():
        configuration.setdefault(deal_type, {}).update(entry)
    return configuration


def market_data(document):
    """The document's `Price Factors`, or a refusal naming what a quote needs them for."""
    factors = document.get('Calc', {}).get('MergeMarketData', {}).get(
        'ExplicitMarketData', {}).get('Price Factors')
    if not factors:
        raise ValueError('the book carries no explicit Price Factors - a quote reads the spot off '
                         'them to bracket its solves')
    return factors


def engine_spot(document, underlying_currency, settlement_currency):
    """Today's rate on the DEAL's axis: units of `Currency` per unit of `Underlying_Currency`.

    The engine's `FxRate.<ccy>.Spot` is that currency in base-currency units, so the cross is the
    ratio - `calc_fx_cross`'s own arithmetic, read off the document instead of a tensor. It is the
    seed and the bracket centre for a strike solve; no price is taken from it.
    """
    factors = market_data(document)
    spots = {}
    for currency in (underlying_currency, settlement_currency):
        name = 'FxRate.{}'.format(currency)
        if name not in factors:
            raise ValueError('{} is missing - a {} quote cannot be struck on this book'.format(
                name, currency))
        spots[currency] = float(factors[name]['Spot'])
    return spots[underlying_currency] / spots[settlement_currency]


def margin_value(document, margin):
    """A declared sales margin as money in the currency the legs price in, or `None` for no margin.

    `{'amount': 50000.0, 'currency': 'ZAR'}` is what a desk charges: an AMOUNT and the currency it
    is stated in, which need not be a currency of the pair. It crosses to the run's reporting
    currency on the ratio of the two `FxRate.<ccy>.Spot` blocks this document carries - the same
    read every strike bracket takes, off the same document a live tick has already moved, so the
    margin converts at the market the quote is struck on. A currency the book carries no rate for
    refuses by name, since a margin nobody can value is a price nobody can quote.

    The reporting currency is the `Calculation` block's, which is what every premium here is in,
    falling back to the book's base where a calculation states none.
    """
    if margin is None:
        return None
    try:
        amount, currency = float(margin['amount']), str(margin['currency']).upper()
    except (KeyError, TypeError, ValueError):
        raise ValueError("a margin is {{'amount': 50000.0, 'currency': 'ZAR'}} - an amount and the "
                         'currency it is stated in, not {!r}'.format(margin)) from None
    pricing = document['Calc']['Calculation'].get('Currency') or base_currency(document)
    return {'amount': amount, 'currency': currency, 'pricing_currency': pricing,
            'value': amount * engine_spot(document, currency, pricing)}


def base_currency(document):
    """The reporting currency every `FxRate.<ccy>.Spot` in this document is quoted in, or `None`
    where the document does not declare one.

    Read off the EXPLICIT block alone, the half `market_data` reads and the only half a quote can
    write, so a `MarketDataFile` not repeated here answers `None` rather than the engine's number.
    None never resolves into a token (`utils.spot_model_currency` refuses it), so the two reads can
    differ only into a refusal.
    """
    return document.get('Calc', {}).get('MergeMarketData', {}).get(
        'ExplicitMarketData', {}).get('System Parameters', {}).get('Base_Currency')


def with_live_spots(document, crosses):
    """Live market crosses written onto the document's own `FxRate.<ccy>.Spot` blocks, IN PLACE -
    the exact inverse of `engine_spot`, and the only thing a quote ever takes off a terminal.

    `crosses` is `{PAIR: value}` as the MARKET quotes each one - `'USDZAR': 16.31` is ZAR per USD.
    `FxRate.<ccy>.Spot` is one unit of that currency in the document's own `Base_Currency` units,
    so a cross pins the leg the base does not: against a USD base, USDZAR writes `FxRate.ZAR` at
    1/16.31 and leaves `FxRate.USD` at 1.0. A pair NEITHER of whose legs is the base refuses by
    name rather than being triangulated - inventing a leg is a market view, not a tick.

    A spot is `bind='value'` data, so this moves a number on a block that already exists and never
    authors one. The caller owns the copy. Returns `{currency: spot}` for what it wrote.
    """
    base = base_currency(document)
    factors = market_data(document)
    written = {}
    for pair, cross in sorted(crosses.items()):
        left, right = split_pair(pair)
        if right == base:
            currency, spot = left, float(cross)
        elif left == base:
            currency, spot = right, 1.0 / float(cross)
        else:
            raise ValueError('{} prices neither of its legs against {} - a live spot is written '
                             "against the book's own base currency".format(pair, base))
        name = 'FxRate.{}'.format(currency)
        if name not in factors:
            raise ValueError('{} is missing - a live spot moves a block the book already carries, '
                             'it does not author one'.format(name))
        factors[name]['Spot'] = spot
        written[currency] = spot
    return written


def quote_points(document, surface):
    """The quotes `surface` was built from - `FXVolSurfaceParameters.used`'s filter over the
    `FXVolPrices` block, and empty where the book carries no such block at all.

    One reader for the pillar-keyed halves and the skew's own rebuild: a spread charged off a row
    the surface was not built from is a spread charged on nothing, and `Use` holds a quote out of
    the quote layer for the same reason it holds one out of the bootstrap.

    The filter is spelled here for the DEFAULT alone: `Use` is stated to hold a row out, so its
    absence is 'Yes' and a hand-authored mid-only block must quote rather than raise.
    """
    prices = document.get('Calc', {}).get('MergeMarketData', {}).get(
        'ExplicitMarketData', {}).get('Market Prices', {})
    block = (prices.get(FX_VOL_PRICES.format(surface)) or {}).get('instrument')
    return [point for point in block['Points'] if point.get('Use', 'Yes') == 'Yes'] \
        if block and block.get('Points') else []


def quote_two_way(document, surface):
    """EVERY quoted pillar's half-spread for `surface`, keyed by the descriptor `dV/dq` reports
    that quote under - `{'ATM 1': 0.002, 'RR 0.25 1': 0.001, ...}`, in the surface's own vol units,
    and empty where the block carries no two-way at all.

    A PILLAR is the unit and nothing is interpolated: a bucket IS a quoted pillar or it is not a
    bucket, and both readers here - the charge a quote levies and the residual the risk step prices
    - deal that pillar's own risk at that pillar's own spread.

    A row missing either side is not a two-way and is skipped; a CROSSED one reads zero-wide rather
    than negative, a stale bid through a live offer being a broken print a desk must never pay a
    client for.

    The descriptor comes from `FXVolSurfaceParameters`, so these keys are the ones the bootstrap's
    leaves are published under; a second copy of the naming rule would drift into silently pricing
    no bucket at all.
    """
    from .bootstrappers import FXVolSurfaceParameters
    halves = {}
    for point in quote_points(document, surface):
        bid, ask = point.get('Quoted_Bid'), point.get('Quoted_Ask')
        if bid is None or ask is None:
            continue
        halves[FXVolSurfaceParameters.descriptor(point)] = max(
            0.0, 0.5 * (float(ask) - float(bid)))
    return halves


def without_spot_model(document, deal_type):
    """The book with no spot model declared for `deal_type`, or the document ITSELF where none is.

    A leg walking a FITTED law never reads the written FX surface, so it publishes no quote
    sensitivity and the two-way has nothing to charge against. The LOGNORMAL reading of the same
    leg at the same terms does, and it is the vega a desk would deal in the quotes it trades. The
    quote's own pin and the book's own are both dropped, on this copy alone.
    """
    configuration = document['Calc']['MergeMarketData']['ExplicitMarketData'].get(
        'Valuation Configuration') or {}
    if 'SpotModel' not in (configuration.get(deal_type) or {}):
        return document
    bare = copy.deepcopy(document)
    del bare['Calc']['MergeMarketData']['ExplicitMarketData'][
        'Valuation Configuration'][deal_type]['SpotModel']
    return bare


def leg_vega(document, leg, surface):
    """`(vega, source, note)` for one leg at its own terms: `{descriptor: dV/dq}` in the quote
    coordinates the two-way is quoted in, where that reading came from, and the leg's note.

    ONE first-order greeks run over the leg ALONE against the book, on the CLIENT's paper, exactly
    as `run_price` values it - so the vega priced and the premium quoted are the same deal at the
    same terms. The PIN decides which book that run is made against: a leg walking a fitted law
    reads nothing off the written surface and would publish no quote leaves, so it is read
    `without_spot_model` straight away rather than after a run whose emptiness the pin already
    predicts. A leg no reading reaches carries no vega, and its NOTE says so: a quote that charged
    nothing on a leg must not report a spread it did not levy.
    """
    run = alone(document, leg.deal)[0]
    bare = without_spot_model(run, leg.deal['Object'])
    vega = vol_risk(bare, [], surface)
    return (vega, 'surface' if bare is run else 'lognormal reading', leg.note) if vega \
        else (None, None, ' '.join(filter(None, (leg.note, NO_VEGA))))


def spread_block(vega, source, halves, scale=1.0):
    """What a leg's row says about the two-way: `{spread_charge, spread_source, spread}`.

    `spread` is one row per quoted pillar - `{pillar, vega, half, cost}`. `half` is the MARKET's own
    half-spread, the same number `risk.buckets` prices a residual at, and `cost` is `scale` times
    `|vega| x half`: money, what the desk charged to deal that pillar's risk. A policy's tightening
    lives in `risk.scale` and in the money, never in the quote it is a fraction of.

    The ABSOLUTE value is the whole ruling - a pillar's side follows the RISK the leg leaves rather
    than the leg's Buy_Sell label, so a geared strip short vol at every ATM pillar pays the spread
    instead of being paid it, and a put's risk reversal is charged on its own side.

    A leg with no vega carries nothing at all rather than a zero.
    """
    if vega is None:
        return {'spread_charge': None, 'spread_source': None, 'spread': None}
    rows = [{'pillar': pillar, 'vega': vega.get(pillar, 0.0), 'half': half,
             'cost': scale * abs(vega.get(pillar, 0.0)) * half}
            for pillar, half in sorted(halves.items())]
    return {'spread_charge': sum(row['cost'] for row in rows), 'spread_source': source,
            'spread': rows}


class Materialized(object):
    """One leg turned into a deal: the wire block, its role, and how its axis relates to the quoted
    one - which is what lets the runner report a solved strike back in market terms."""
    __slots__ = ('role', 'deal', 'inverted', 'note')

    def __init__(self, role, deal, inverted, note=None):
        self.role, self.deal, self.inverted = role, deal, inverted
        # what the runner had to decide about this leg that the client would not otherwise see
        self.note = note

    def to_market(self, engine_strike):
        """An engine-axis strike as the client reads it."""
        return 1.0 / engine_strike if self.inverted else engine_strike


def materialize(structure, params, document):
    """Every leg of `structure` as a wire-form deal, in declaration order.

    The shared block comes from the parameters - the two currencies off the pair, the surface named
    for it, the settlement currency doing the discounting, the notional as the underlying amount,
    and the expiry as a date. Then the leg's pinned block, then its slots. Strike-like slots and
    `Option_Type` cross to the engine axis together, exactly once, here.

    An ACCRUAL leg is furnished the same way plus a schedule - see `furnish_accrual`. It may WRITE
    to `document` to pin the spot model on the deal type, so the caller owns the copy exactly as it
    does for `with_live_spots`.

    THE PAIR IS THE BOOK'S. A surface is quoted one way round, so a pair stated backwards names a
    block the book does not carry, every leg is dropped at load and the quote comes back priced at
    nothing; it refuses here naming the pairs the book does quote. A notional is a positive amount
    - which side of the trade the desk is on is the structure's own statement, not the sign.
    """
    params = declared(structure, params)
    base, quote_ccy = split_pair(params['pair'])
    underlying = str(params['notional_currency']).upper()
    if underlying not in (base, quote_ccy):
        raise ValueError('notional_currency {!r} is not a side of {}'.format(
            underlying, params['pair']))
    notional = float(params['notional'])
    if notional <= 0.0:
        raise ValueError('a notional is a positive amount, not {:g} - the side the desk takes is '
                         "the structure's own".format(notional))
    factors = market_data(document)
    if FX_VOL_FACTOR.format('{}.{}'.format(base, quote_ccy)) not in factors:
        raise ValueError('{} is not a pair this book quotes - it carries {}'.format(
            params['pair'], ', '.join(name.split('.', 1)[1].replace('.', '')
                                      for name in sorted(factors)
                                      if name.startswith('FXVol.')) or 'no FX surface'))
    settlement = quote_ccy if underlying == base else base
    # the quoted axis is the deal's own only when the notional is the pair's BASE currency
    inverted = underlying == quote_ccy
    base_date = timestamp(document['Calc']['Calculation']['Base_Date'])
    shared = {'Currency': settlement, 'Discount_Rate': settlement,
              'Underlying_Currency': underlying, 'Underlying_Amount': notional,
              'FX_Volatility': '{}.{}'.format(base, quote_ccy),
              'Expiry_Date': expiry_date(base_date, params['expiry'])}
    seed = engine_spot(document, underlying, settlement)

    out = []
    variation = variation_for(structure, params)[1]
    for leg in (variation.legs if variation else structure.legs):
        if leg.deal_type not in ('FXOptionDeal', 'FXBarrierOption') + ACCRUAL_DEALS:
            raise ValueError('{}: the runner furnishes FXOptionDeal, FXBarrierOption and the '
                             'accrual deals {}, not {}'.format(
                                 leg.role, ', '.join(ACCRUAL_DEALS), leg.deal_type))
        deal = dict(shared, Object=leg.deal_type)
        deal.update(leg.pinned)
        for field, slot in leg.slots.items():
            if slot not in params:
                raise ValueError('{}: leg {} needs the {!r} parameter'.format(
                    structure.__name__, leg.role, slot))
            value = float(params[slot])
            deal[field] = 1.0 / value if inverted and field in (
                'Strike_Price', 'Barrier_Price') else value
        # senses and directions convert AFTER pinned and slots merge, so a structure letting the
        # client choose either still crosses the axis exactly once
        if inverted:
            # a call on the pair is a put on the quote currency
            for field, flip in (('Option_Type', OPTION_FLIP), ('Barrier_Type', BARRIER_FLIP)):
                if field in deal:
                    deal[field] = flip[deal[field]]
        # a level the CLIENT stated must sit on the LIVE side of its own direction, both being on
        # the engine axis by now, and a level ON the spot is through it already - the pricer's own
        # survival is strict both ways. A SOLVED level is the recipe's, bracketed on that side
        if 'Barrier_Price' in leg.slots and not (
                deal['Barrier_Price'] > seed if deal['Barrier_Type'].startswith('Up')
                else deal['Barrier_Price'] < seed):
            raise ValueError(
                '{}: a {} at {:g} is the wrong side of the market SPOT {:g}, which is what this is '
                'checked against - an Up level is quoted above the spot and a Down one below it, '
                'and a level on it is through already. State one the pair has to travel to, or '
                'quote the structure the other way round'.format(
                    leg.role, BARRIER_FLIP[deal['Barrier_Type']] if inverted
                    else deal['Barrier_Type'],
                    1.0 / deal['Barrier_Price'] if inverted else deal['Barrier_Price'],
                    1.0 / seed if inverted else seed))
        # an unsolved strike still has to be a number the splice can price - the solve replaces it
        deal.setdefault('Strike_Price', seed)
        if leg.deal_type == 'FXBarrierOption':
            # the direction is the structure's own statement: the Instrument declaration's default
            # would ride the axis unflipped, so a barrier leg that names none refuses
            if 'Barrier_Type' not in deal:
                raise ValueError('{}: barrier leg {} declares no Barrier_Type'.format(
                    structure.__name__, leg.role))
            # and an unsolved barrier has to be a number on the live side of its own direction,
            # read off the ENGINE axis the type now sits on
            deal.setdefault('Barrier_Price',
                            seed * (0.75 if deal['Barrier_Type'].startswith('Down') else 1.25))
            # a deal block IS the field dict the pricer reads, so a declared default never reaches
            # it: the two fields `pv_barrier_option` asks for by name are written out, or the deal
            # is SKIPPED at load and the leg prices at nothing
            deal.setdefault('Barrier_Monitoring_Frequency', {'.DateOffset': '0M'})
            deal.setdefault('Cash_Rebate', 0.0)
        note = furnish_accrual(deal, params, document, base_date, underlying, inverted) \
            if leg.deal_type in ACCRUAL_DEALS else None
        out.append(Materialized(leg.role, deal, inverted, note))
    return out


def furnish_accrual(deal, params, document, base_date, underlying, inverted):
    """The rest of an accrual leg: its fixing strip, its geared notional, and the one axis question
    a strip asks that a single expiry does not. Returns the leg's note.

    THE SCHEDULE. `fixing_grid` grows it from the tenor and `fixing_frequency`, filed under the
    deal's own field name. `FXTARFOptionDeal` also declares an `Expiry_Date`, set to the LAST
    SETTLEMENT rather than to the tenor; `FXAccumulatorOptionDeal` declares no such field, so the
    shared block's is REMOVED - a deal block is the field dict the pricer reads.

    THE NOTIONALS. `Underlying_Amount` is the notional per fixing, already in `notional_currency`;
    `LeverageNotional` is `leverage` times it. Neither has an axis: `notional_currency` IS the
    underlying, and a gearing is a pure ratio.

    THE TARGET does not cross. It is a sum of DIFFERENCES, and `1/S - 1/K` is not the reciprocal of
    `S - K`, so no number in reciprocal units means the same accrual cap. `InvertedTarget` is not
    the answer either - it moves the whole fixing onto the reciprocal of the deal axis, a different
    product (0.77% apart in the solved strike on the gate's book) - so it is False on every leg the
    runner builds and a TARF quoted on the pair's QUOTE currency REFUSES by name, in this function's
    FIRST statement, since a refusal firing later has already written the block the caller holds. An
    accumulator has no target and crosses freely.

    THE MODEL. `spot_model` pins `SPOT_MODEL` where the book carries a calibration for this leg's
    PAIR, and hands back the note where it does not.
    """
    if inverted and deal['Object'] == 'FXTARFOptionDeal':
        raise ValueError(
            'a target redemption forward is quoted on the pair\'s BASE currency: the target is '
            'an accrual cap in the pair\'s own units, the deal accrues on the axis its notional '
            'puts it on, and a sum of differences has no reading on the reciprocal - '
            '{} would cap a move nobody quoted'.format(underlying))
    schedule = fixing_grid(base_date, params['expiry'], params['fixing_frequency'])
    deal[SCHEDULE_FIELD[deal['Object']]] = schedule
    deal['LeverageNotional'] = float(params['leverage']) * float(params['notional'])
    if deal['Object'] == 'FXTARFOptionDeal':
        deal['Expiry_Date'] = dict(schedule[-1][1])
        # the deal accrues and pays on its OWN axis - see the docstring for what the flag would do
        deal['InvertedTarget'] = False
        # the OTM knock-in this desk does not sell; `> 0.0` is the pricer's own switch
        deal.setdefault('Barrier', 0.0)
    else:
        deal.pop('Expiry_Date', None)
    return spot_model(document, deal['Object'], underlying, deal['Currency'])


def alone(document, deal):
    """A deep copy of the book carrying only `deal`, plus that deal's path.

    The book's market data, calendars, bootstrappers and calculation block travel; its deal tree
    does not. A deal's own base-valuation row does not depend on its siblings, and a lone deal
    compiles faster per iterate of a solve.
    """
    iterate = copy.deepcopy(document)
    schema.job_children(iterate)[:] = []
    iterate['Calc']['Calculation']['Object'] = 'BaseValuation'
    return iterate, schema.splice_deal(iterate, deal)


def own_value(out, reference):
    """One deal's own row off a run's `mtm` frame."""
    frame = out['Results']['mtm']
    row = frame[frame['Reference'] == reference]
    if not len(row):
        raise ValueError('{} priced but reported no mtm row'.format(reference))
    return float(row['Value'].iloc[0])


def run_price(document, deal):
    """One leg's value as an ordinary base valuation, which already reports the number."""
    from . import Context
    _, out = Context().load_json((json.dumps(alone(document, deal)[0]), 'quote')).run_job()
    return own_value(out, deal['Reference'])


def run_solve(document, leg, field, target, spot, seed=None):
    """`derivus.solve_deal_field` over one leg, bracketed, writing the answer back onto the leg.

    A strike is bracketed around the market spot by `STRIKE_BRACKET` - `ACCRUAL_BRACKET` for a
    strip - and crossed to the engine axis, where inverting swaps the ends, so they are sorted
    rather than assumed. A BARRIER is bracketed on the side its own type lives on, off the same
    ends. Any other field is left to the secant from its current value, which is exact in two
    pricings for anything the value is affine in. Returns `(solved, premium at the solved value)`.

    `seed` is a root already found for this field on this leg - the mid pass's, where the charge
    then moves the target by a spread's width - and narrows the bracket to `SEED_BRACKET` around it,
    clipped into the full ends so a barrier keeps the side its own type lives on. A narrow bracket
    that does not STRADDLE falls back to the full one and costs one evaluation, so the answer is the
    full bracket's either way. A solve with no seed runs the ends it always did.
    """
    from . import solve_deal_field
    iterate, deal_path = alone(document, leg.deal)
    bounds = None
    if field == 'Strike_Price':
        ends = ACCRUAL_BRACKET if leg.deal['Object'] in ACCRUAL_DEALS else STRIKE_BRACKET
        bounds = sorted([spot / end if leg.inverted else spot * end for end in ends])
    elif field == 'Barrier_Price':
        # `spot` and the leg's Barrier_Type are both already on the ENGINE axis, so the direction
        # names the side directly - with a hair of buffer so the barrier never lands on the spot
        bounds = sorted([spot * STRIKE_BRACKET[0], spot * 0.9999]) \
            if leg.deal['Barrier_Type'].startswith('Down') \
            else sorted([spot * 1.0001, spot * STRIKE_BRACKET[1]])
    ends = [bounds] if bounds is None or seed is None else [
        [max(bounds[0], seed * (1.0 - SEED_BRACKET)),
         min(bounds[1], seed * (1.0 + SEED_BRACKET))], bounds]
    for attempt, span in enumerate(ends):
        try:
            solved, _, _, out = solve_deal_field(
                iterate, deal_path, field, target=target, bounds=span)
            break
        except ValueError as error:
            if span is None or 'different signs' not in str(error):
                raise
            if attempt + 1 < len(ends):
                continue
            # brentq's sign check speaks in f(a) and f(b); a desk needs the economics said out loud
            raise ValueError(
                '{}: no {} in [{:.6g}, {:.6g}] lets this leg reach {:.6g} - the structure cannot '
                'be financed at these parameters'.format(
                    leg.deal['Reference'], field, span[0], span[1], target))
    leg.deal[field] = solved
    return solved, own_value(out, leg.deal['Reference'])


def compose(reference, legs, margin=None):
    """The priced legs as ONE bookable deal: a `StructuredDeal` whose `Children` are the legs with
    their solved values in place, exactly as they were priced. Settled in the legs' own settlement
    currency, so the container nets what the parts report without a cross of its own.

    The children sit INSIDE the deal here, because a quote is filed, hashed and read back whole.
    The deal TREE holds a container's children one level out - see `book_node`.

    A `margin` is RECORDED on the container as `Sales_Margin` in `Sales_Margin_Currency`, in the
    currency it was declared in rather than the one it was converted to - what was agreed is what
    the ticket says. The charge is already inside the solved coordinate, so this changes no value;
    a quote with no margin composes the block it always did.
    """
    deal = {'Object': 'StructuredDeal', 'Reference': reference,
            'Currency': legs[0].deal['Currency'], 'Net_Cashflows': 'Yes',
            'Children': [{'Instrument': {'.Deal': dict(leg.deal)}} for leg in legs]}
    if margin:
        deal.update(Sales_Margin=margin['amount'], Sales_Margin_Currency=margin['currency'])
    return deal


def book_node(deal):
    """A composed deal as the NODE a job document's deal tree holds.

    A container's children hang off the NODE - beside `Instrument`, not inside the deal block -
    which is the shape `Context.load_json` walks and `splice_deal` builds. Lifting them out
    happens exactly once, here: a populated container spliced flat loads with no children and
    prices at ZERO, silently.
    """
    node = {'Instrument': {'.Deal': {k: v for k, v in deal.items() if k != 'Children'}}}
    if 'Children' in deal:
        node['Children'] = copy.deepcopy(deal['Children'])
    return node


def mirror(deal):
    """The desk's side of a quoted deal: every leg's `Buy_Sell` flipped, nothing else touched.

    A quote is CLIENT paper while a trading book holds the BANK's position, so this is the one seam
    where paper becomes position. Both consumers read it - the approval that books the trade and
    the risk-impact step that prices the book plus the candidate - so a sign cannot disagree
    between them. A recorded sales margin rides along unchanged: the client's paper is worth minus
    the margin and the bank's side plus it, which is the flip doing its work, not a second entry.
    """
    flipped = copy.deepcopy(deal)
    blocks = [flipped] + [child['Instrument']['.Deal'] for child in flipped.get('Children', [])]
    for block in blocks:
        if 'Buy_Sell' in block:
            block['Buy_Sell'] = 'Sell' if block['Buy_Sell'] == 'Buy' else 'Buy'
    return flipped


def netting_set_references(document):
    """Every `NettingCollateralSet` Reference the book carries, sorted - the set names a quote may
    be booked under. A set nested inside another container is still a set, so the whole tree is
    walked rather than the top level."""
    return sorted(node['Instrument']['.Deal'].get('Reference')
                  for _, node in schema.walk_job_deals(document)
                  if node['Instrument']['.Deal'].get('Object') == 'NettingCollateralSet')


def check_netting_set(document, reference):
    """Refuse `reference` unless the book carries a `NettingCollateralSet` by that name.

    A CLIENT IS A NETTING SET: the counterparty and the CSA live on the set, and a trade booked
    anywhere else is invisible to the CVA projection that netting set is the unit of. So the set is
    checked at QUOTE time - a quote given under a set that does not exist is a quote nobody can
    approve. The refusal is worded as the XVA verb's, since both ask the same question.
    """
    if reference is None:
        return
    found = netting_set_references(document)
    if reference not in found:
        raise ValueError('the book carries no NettingCollateralSet called {!r} - its sets are '
                         '{}'.format(reference, ', '.join(found) or 'none'))


def read_policy(document):
    """The desk's quoting mandate off `Calc['Quote Policy']`, or `None` where the book declares
    none - and `None` turns the whole risk-impact feature off, which is the compatibility contract.

    Six fields, each read with `.get` against `POLICY_DEFAULTS` so a block may state one of them:

      - `participation` - how much of a measured hedge-cost SAVING is passed to the client
      - `floor` - 'mid': the scale never goes below zero, so a quote never crosses the mid
      - `scope` - 'vol', all v1 measures; anything else refuses
      - `bucket_limit` - a per-bucket cap on `|risk after|` in the bucket's own vega units, past
        which no tightening applies however good the saving looks
      - `min_ticket_bp` - flat bp of notional, the ops floor under the edge
      - `firm_seconds` - how long a quote stays approvable; the approval verb reads it, and this
        module carries it through so a desk states its mandate in ONE block

    A field that will not read refuses HERE rather than at the later verb that compares against it.
    """
    policy = document.get('Calc', {}).get(QUOTE_POLICY)
    if policy is None:
        return None
    read = {name: policy.get(name, default) for name, default in POLICY_DEFAULTS.items()}
    try:
        read['firm_seconds'] = float(read['firm_seconds'])
        if read['firm_seconds'] < 0.0:
            raise ValueError('negative')
    except (TypeError, ValueError):
        raise ValueError('{}: firm_seconds {!r} - a quote is firm for a NUMBER of seconds, and a '
                         'window that cannot be read is one no approval could be measured '
                         'against'.format(QUOTE_POLICY, policy.get('firm_seconds'))) from None
    if read['scope'] != 'vol':
        raise ValueError('{}: scope {!r} - v1 measures the vol book and nothing else, so any '
                         'other scope would quote a residual it never looked at'.format(
                             QUOTE_POLICY, read['scope']))
    if read['floor'] != 'mid':
        raise ValueError('{}: floor {!r} - the only floor v1 implements is the mid, which is the '
                         'ruling that a quote never goes through it automatically'.format(
                             QUOTE_POLICY, read['floor']))
    return read


def risk_document(document, nodes, surface):
    """The book as a GREEKS run, with `nodes` added to its deal tree and the vol quotes connected.

    Three edits and no others: the calculation becomes a `BaseValuation` with `Greeks: 'First'` -
    one backward off the ROOT netting set, so a leaf's `.grad` is the whole portfolio's - the
    candidate's nodes are appended, and `Quote_Sensitivity` goes to Yes on the `FXVolPrices` block,
    which is what leaves the surface connected to the quotes it was built from.

    That switch is worth exactly zero in the forward pass, so turning it on cannot move a price.

    `Use` is written out per point on the same copy: the bootstrap reads the field BY NAME while
    this module's own contract is that its absence is 'Yes' (`quote_points`), so a hand-authored
    mid-only block that quotes must also be one the quote's own greeks run can read.
    """
    run = copy.deepcopy(document)
    run['Calc']['Calculation'] = dict(run['Calc']['Calculation'],
                                      Object='BaseValuation', Greeks='First')
    schema.job_children(run).extend(nodes)
    block = run['Calc']['MergeMarketData']['ExplicitMarketData'][
        'Market Prices'][FX_VOL_PRICES.format(surface)]
    points = [dict(point, Use=point.get('Use', 'Yes'))
              for point in block['instrument'].get('Points') or []]
    block['instrument'] = dict(block['instrument'], Points=points, Quote_Sensitivity='Yes')
    return run


def vol_risk(document, nodes, surface):
    """`{descriptor: dV/dq}` for the book plus `nodes` - the vol book in QUOTE coordinates.

    The attachment is harvested at BOOTSTRAP rather than at run, so this bootstraps its own copy
    and prices it in the SAME `Context`: the leaves `Config.quote_leaves` publishes are the tensors
    `Calculation.factor_leaf` was offered, and one backward off the root leaves `.grad` on them.

    Descriptors are SUMMED across every published block - one JSON number can feed two chains, and
    each family's partial is correct while neither is the answer.

    An EMPTY deal tree has no value to differentiate and its risk is a zero vector, so it never
    reaches a run.
    """
    from . import Context
    from .config import CustomJsonEncoder
    run = risk_document(document, nodes, surface)
    if not schema.job_children(run):
        return {}
    context = Context().load_json((json.dumps(run, cls=CustomJsonEncoder), 'risk'))
    context.bootstrap()
    context.run_job()
    risk = {}
    for descriptors, leaves in context.current_cfg.quote_leaves.values():
        if leaves.grad is None:
            continue
        for descriptor, value in zip(descriptors, leaves.grad.detach().cpu().numpy().ravel()):
            risk[descriptor] = risk.get(descriptor, 0.0) + float(value)
    return risk


def book_risk(document, surface):
    """The book's OWN vol risk, cached on the book's content etag.

    The book alone is the half of the measurement that does not depend on what is being quoted, so
    a desk quoting repeatedly against a standing book pays for one greeks run rather than two per
    quote. The etag is over everything the run reads - a rolled `Base_Date` over an unmoved book is
    a different risk vector.
    """
    from . import content_hash
    etag = content_hash({'deals': document['Calc']['Deals']['Deals'],
                         'market': document['Calc']['MergeMarketData'],
                         'calculation': document['Calc']['Calculation'],
                         'surface': surface})
    if etag not in RISK_CACHE:
        if len(RISK_CACHE) >= RISK_CACHE_LIMIT:
            RISK_CACHE.pop(next(iter(RISK_CACHE)))
        RISK_CACHE[etag] = vol_risk(document, [], surface)
    return RISK_CACHE[etag]


def risk_buckets(before, after, halves):
    """Per-bucket rows and the RESIDUAL HEDGE COST they add up to, in the report currency.

    A bucket's cost is the move in ABSOLUTE risk times that bucket's own half-spread: what it would
    cost, at the market's own two-way, to put the residual back flat. `dV/dq` is already a vega in
    report currency per unit of quote, so the product is money and nothing converts it. A NEGATIVE
    total is the trade shedding risk, and the policy decides how much of that saving is passed on.

    Only buckets the book quotes a two-way for are priced: a bucket with no quoted spread has no
    market price for its risk.
    """
    rows, cost = [], 0.0
    for bucket in sorted(halves):
        was, now, half = before.get(bucket, 0.0), after.get(bucket, 0.0), halves[bucket]
        delta = abs(now) - abs(was)
        rows.append({'bucket': bucket, 'before': was, 'after': now,
                     'delta': delta, 'half_spread': half})
        cost += delta * half
    return rows, cost


def risk_scale(rows, cost, policy, charge_full, min_ticket):
    """The policy applied: `(scale, saving, charge_effective, note)`.

    `scale` multiplies every leg's half-spread on the re-quote and lives in [0, 1]. Three rulings
    are in this arithmetic:

      - a risk-ADDING trade stays at the full two-way. The market spread is the CEILING - there is
        no surcharge past it - so a positive residual cost is simply no saving.
      - the mid is the FLOOR. The effective charge never goes below zero.
      - the min ticket is the ops floor UNDER the tightening, not a second ceiling over it: a
        min_ticket above the full spread leaves the scale at 1 rather than lifting the quote.

    A bucket standing past `bucket_limit` after the trade suspends the tightening entirely and is
    NAMED, however good the saving elsewhere looks.
    """
    limit = policy['bucket_limit']
    capped = next((row['bucket'] for row in rows
                   if limit is not None and abs(row['after']) > float(limit)), None)
    saving = 0.0 if capped is not None else max(0.0, -cost)
    effective = max(0.0, charge_full - float(policy['participation']) * saving)
    effective = max(effective, min_ticket)
    scale = 1.0 if charge_full <= 0.0 else min(1.0, effective / charge_full)
    note = None if capped is None else (
        '{} stands at {:.6g} after the trade, past bucket_limit {:g} - no tightening '
        'applies'.format(capped, next(row['after'] for row in rows if row['bucket'] == capped),
                         float(limit)))
    return scale, saving, scale * charge_full, note


def run_recipe(document, structure, params, reference, spot, charge=0.0, seed=None):
    """One whole pass of the recipe at MID, from fresh legs.

    Materializes the legs and runs the steps in order against the book as it stands. EVERY leg
    prices at the mid surface: what the market charges for the two-way is levied as a charge on the
    solved coordinate rather than by pricing the legs on two different books at once, so what the
    finished legs are worth IS what the trade marks at once booked.

    `charge` is what the desk levies in the pricing currency - the sales margin and the two-way's
    own charge together - and it moves the SOLVED coordinate: on the client's paper the structure
    is worth minus what the desk charges for it.

    `seed` is an earlier pass's `solved`, and each solve brackets narrowly around its own root in
    it; the charge moves a coordinate by a spread's width, so the second pass need not search the
    whole bracket again.

    Returns `(legs, premiums, solved)`. Legs are FRESH because `run_solve` writes the solved value
    back onto the leg it moved, so a second pass over the first pass's legs would seed itself from
    the answer it is meant to find.
    """
    legs = materialize(structure, params, document)
    for leg in legs:
        leg.deal['Reference'] = '{}_{}'.format(reference, leg.role)
        leg.deal['Structure_Reference'] = reference
    by_role = {leg.role: leg for leg in legs}

    premiums, solved = {}, {}
    for step in structure.recipe:
        if step.role not in by_role:
            raise ValueError('{}: the recipe names leg {!r}, which is not declared'.format(
                structure.__name__, step.role))
        leg = by_role[step.role]
        if isinstance(step, Price):
            premiums[leg.role] = run_price(document, leg.deal)
        elif isinstance(step, Solve):
            # the target is the legs already priced, and the desk's charge with them: the client's
            # paper is worth minus what they are charged for it
            target = step.target.value(premiums) if isinstance(step.target, Premium) \
                else float(step.target)
            value, premiums[leg.role] = run_solve(
                document, leg, step.field, target - charge, spot,
                (seed or {}).get(leg.role, {}).get(step.field))
            solved.setdefault(leg.role, {})[step.field] = value
        else:
            raise ValueError('{}: {!r} is not a recipe step'.format(structure.__name__, step))

    unpriced = [leg.role for leg in legs if leg.role not in premiums]
    if unpriced:
        raise ValueError('{}: the recipe never prices {}'.format(
            structure.__name__, ', '.join(unpriced)))
    return legs, premiums, solved


def risk_impact(document, params, reference, surface, legs, halves, charge_full):
    """The whole risk-impact step over a candidate the two-way has already been charged on.

    Measures the book with the candidate's MIRROR on it and without, prices the difference at the
    market's own two-way, applies the policy, and hands back the `risk` block the outcome carries -
    `scale` included, which is what the quote's charge is multiplied by before the coordinate is
    re-solved.

    Five ways out, each leaving `scale` at None with the reason NAMED rather than reported as a
    scale of 1 nobody can distinguish from a decision: the book declares no `Quote Policy`; it
    quotes no two-way at all, so there is no spread to tighten and no half-spread to price a bucket
    at; NO LEG could be read, so there is no charge at all and `charge_full` stays null rather than
    reporting a zero nobody measured; the charge is not positive; or no vol quote leaves are
    published, so there are no coordinates to measure in.
    """
    policy = read_policy(document)
    empty = {'coordinates': 'quote-space', 'buckets': [], 'saving': None, 'charge_full': None,
             'charge_effective': None, 'scale': None, 'policy': policy}
    if policy is None:
        return dict(empty, note='the book declares no {} block - the quote is the full two-way '
                                'spread, exactly as it was before'.format(QUOTE_POLICY))
    if not halves:
        return dict(empty, note='{} carries no two-way - there is no spread to tighten'.format(
            FX_VOL_PRICES.format(surface)))
    if charge_full is None:
        return dict(empty, note='no leg of this structure publishes an FX vol quote sensitivity - '
                                'the two-way reached none of them, so nothing was charged and '
                                'there is no charge to tighten')
    if charge_full <= 0.0:
        return dict(empty, charge_full=charge_full,
                    note='the two-way captured {:.6g} - there is nothing to give back'.format(
                        charge_full))

    # the MIRROR is the desk's side, and the same verb the approval books through, so the risk
    # measured and the trade booked cannot disagree by a sign
    candidate = book_node(mirror(compose(reference, legs)))
    before = book_risk(document, surface)
    after = vol_risk(document, [candidate], surface)
    if not (before or after):
        return dict(empty, charge_full=charge_full,
                    note='no FX vol quote leaves were published - the book carries no '
                         'FXVolSurfaceParameters bootstrap, so there are no quote coordinates to '
                         'measure the residual in and no tightening applies')

    rows, cost = risk_buckets(before, after, halves)
    # a bp of NOTIONAL in the report currency: the notional is in its own currency and the charge
    # is in the run's, so the ops floor crosses on the same FxRate ratio `engine_spot` reads
    min_ticket =float(policy['min_ticket_bp']) * BASIS_POINT * float(params['notional']) * \
        engine_spot(document, str(params['notional_currency']).upper(),
                    document['Calc']['Calculation']['Currency'])
    scale, saving, effective, note = risk_scale(rows, cost, policy, charge_full, min_ticket)
    return {'coordinates': 'quote-space', 'buckets': rows, 'saving': saving,
            'charge_full': charge_full, 'charge_effective': effective, 'scale': scale,
            'policy': policy, 'note': note}


def quote(document, structure_name, params, spot_source=None, netting_set=None, margin=None):
    """Price a structure against a book, and hand back the quote plus the deal it would book.

    `document` is a wire-form job document - the book - and travels whole, never a patch. `params`
    are the client's numbers, in the market's own terms, and come back COMPLETED by the structure's
    own declared defaults. The answer carries `quote_id`, `structure`, `params`, the `variation`
    selected and the `client` cashflows read off it, `netting_set`, a row per leg (reference, role,
    deal type, side, market-terms strike and barrier, premium, whatever was solved, the two-way
    `spread_charge` levied on it with its `spread_source` and pillar rows, and any `note`), `net`,
    `net_mid`, `edge`, `charged_on`, `spot`, `risk`, `valuation_configuration`, and `deal` - the
    composed `StructuredDeal` in wire form, ready for the booking verb. `quote_id` hashes the structure, the
    parameters, the market the book was carrying AND a submission clock: a quote is an ACT, so two
    identical asks minutes apart are two quotes.

    THE TWO-SIDED HALF IS A CHARGE. Every leg prices at the MID, and where the book's `FXVolPrices`
    block carries a two-way each leg's VEGA is read at the mid solution in the quote coordinates
    those sides are quoted in - one first-order greeks run per leg, on the client's paper - and
    charged `sum over pillars of |vega| x half`. The side a pillar is dealt on therefore follows the
    RISK rather than the leg's `Buy_Sell` label, which is why a geared strip short vol at every ATM
    pillar pays the spread instead of being paid it. A leg priced under a pinned spot model reads
    nothing off the written surface, so its vega comes from the LOGNORMAL reading of the same leg at
    the same terms and `spread_source` says so. Each leg pays its own spread; the legs of one
    package are not netted against each other. The charge moves the coordinate the recipe SOLVES,
    so `net_mid` - the finished legs at mid - is what the trade marks at once booked and `net` is
    that plus the charge. `edge` IS the charge, non-negative by construction. A leg no reading
    reaches carries `spread_charge` null with a note, never a zero, and so does the structure's own
    `risk.charge_full` where no leg could be read; `spread_note` names a book that quotes no
    two-way at all. A pillar's `half` is always the MARKET's half-spread, the same number
    `risk.buckets` prices a residual at - a policy's tightening is stated once under `risk.scale`
    and carried in the money, never in the quote it is a fraction of.

    `spot` names the market this quote was struck on: `value_market` is the pair as the client
    quotes it, READ off the document the legs priced against rather than taken from the caller,
    beside the caller's `source` ('terminal' or 'book') and its `note`.

    THE SALES MARGIN. `margin` is `{'amount': 50000.0, 'currency': 'ZAR'}`, what the desk charges,
    stated as money in whatever currency it was agreed in. It crosses to the pricing currency at
    this document's own spots (`margin_value`) and is charged the way the two-way is, on the ONE
    coordinate the quote has: the financing leg raises the premiums it finances plus the charge, a
    single-solve strip targets minus the charge rather than zero, and a recipe that SOLVES NOTHING
    charges the PREMIUM instead - the client simply pays more, the booked legs stay at mid, and
    `charged_on` names which it was. So on a solving structure `net` reads the margin back NEGATIVE
    - the client holds paper worth minus what they paid for it - and the booked `mirror` marks the
    bank's side at plus the margin and the edge together; on a premium-charged one the client's
    payment moves against them by both and the mirror marks at mid. A recipe that solves MORE than
    one coordinate refuses by name: every strike of it is the client's own. The composed deal
    records the amount as declared. Absent, the answer carries no `margin` at all and every number
    is what it was.

    THE RISK-IMPACT HALF, off unless the book declares a `Calc['Quote Policy']` block. Where it
    does, the mid candidate is MIRRORED into the desk's side and the book's vol risk is measured
    with it and without it in quote coordinates. Each bucket's move in ABSOLUTE risk times that
    bucket's own half-spread is what hedging the residual costs; a negative total is a SAVING, and
    `participation` of it comes off the charge, `charge_effective = scale x charge_full`.

    ONE PASS, NOT A FIXED POINT - a stated approximation. Both the vegas and the risk buckets are
    the MID solution's while the charge moves the solved coordinate, so the quoted structure's vega
    and residual are not exactly the ones priced; the move is second order and the reported buckets
    are the mid candidate's. A risk-ADDING trade quotes at the full spread, so `scale` stays in
    [0, 1].

    `valuation_configuration` is what THIS quote's passes pinned, never what the book already
    declared; the pin lives on the quote's copy and dies with it, so `/book/quote` merges it into
    the book as part of the booking act. `netting_set` names an existing `NettingCollateralSet` -
    the CLIENT, that being where the counterparty and the CSA are declared - and the approval books
    the mirror UNDER that node, inside the subtree the CVA projection prices. It is checked against
    THIS document before anything is priced; `None` is the root booking.
    """
    from . import content_hash
    from .config import CustomJsonEncoder
    # authored objects and a file's wire form become one shape here, so every copy below is plain
    # JSON and a solve's own re-serialisation cannot trip over a Timestamp
    document = json.loads(json.dumps(document, cls=CustomJsonEncoder))
    # what the BOOK already pinned, so what is reported below is what THIS quote pinned
    already = pinned_models(document) or {}
    # the client is checked before the price: a quote nobody could approve is not worth the solves
    check_netting_set(document, netting_set)
    structure = structure_named(structure_name)
    margin = margin_value(document, margin)
    solves = [step for step in structure.recipe if isinstance(step, Solve)]
    if margin is not None and len(solves) > 1:
        raise ValueError(
            'a margin is charged by moving the ONE coordinate a recipe solves, and {} solves {} of '
            'them - every strike of it is the client\'s own. Quote it at strikes the margin is '
            'already in'.format(structure_name, len(solves)))
    # a declared default is part of what was quoted, so it is filled in before the id is hashed and
    # before the outcome reports the parameters, rather than inside `materialize` alone
    params = declared(structure, params)
    variation, dealt = variation_for(structure, params)
    side = dict(zip(('base', 'quote'), split_pair(params['pair'])))
    quote_id = content_hash({
        'structure': structure_name, 'params': params, 'netting_set': netting_set,
        'margin': margin,
        'market': document.get('Calc', {}).get('MergeMarketData', {}).get('ExplicitMarketData', {}),
        'at': time.perf_counter()})
    reference = '{}-{}'.format(structure_name, quote_id[:8])

    probe = materialize(structure, params, document)[0]
    spot = engine_spot(document, probe.deal['Underlying_Currency'], probe.deal['Currency'])
    surface = probe.deal['FX_Volatility']
    halves = quote_two_way(document, surface)

    # the desk's own charge rides the coordinate the recipe SOLVES, or the premium where it solves
    # nothing; either way the client pays the margin and the two-way on top of the mid
    charge = margin['value'] if margin else 0.0
    legs, premiums, solved = run_recipe(document, structure, params, reference, spot,
                                        charge if solves else 0.0)
    # the MID solution's own vegas, one greeks run per leg, and never made at all on a book that
    # quotes no two-way: there is nothing to charge them at
    vegas = [leg_vega(document, leg, surface) for leg in legs] if halves \
        else [(None, None, leg.note) for leg in legs]
    blocks = [spread_block(vega, source, halves) for vega, source, _ in vegas]
    priced = [block['spread_charge'] for block in blocks
              if block['spread_charge'] is not None]
    # a charge NO leg could be read for is not a charge of nothing; it is a charge nobody measured
    full = sum(priced) if priced else None
    risk = risk_impact(document, params, reference, surface, legs, halves, full)
    # what the charge was levied AT: the base pass takes the full two-way, a tightened quote the
    # policy's own scale, and the rows below are rebuilt at the money that was really charged
    charged = 1.0 if risk['scale'] is None else risk['scale']
    levied = charged * (full or 0.0)
    if levied and solves:
        legs, premiums, solved = run_recipe(
            document, structure, params, reference, spot, charge + levied, solved)
    if charged != 1.0:
        blocks = [spread_block(vega, source, halves, charged) for vega, source, _ in vegas]

    outcome = {
        'quote_id': quote_id, 'structure': structure_name, 'params': dict(params),
        # WHICH WAY it was dealt, and the client's own two cashflows read off the variation that
        # was priced rather than off the ticket, so the booking and the account of it agree. Both
        # null for a structure declaring one form
        'variation': variation,
        'client': {'buys': side[dealt.buys], 'sells': side[OPPOSITE[dealt.buys]]}
        if dealt else None,
        # WHO the quote is for, always said: a null is the root booking, never an unanswered
        # question
        'netting_set': netting_set,
        'legs': [dict({'reference': leg.deal['Reference'], 'role': leg.role,
                       'deal_type': leg.deal['Object'], 'buy_sell': leg.deal.get('Buy_Sell'),
                       'strike_market': leg.to_market(leg.deal['Strike_Price'])
                       if 'Strike_Price' in leg.deal else None,
                       'barrier_market': leg.to_market(leg.deal['Barrier_Price'])
                       if 'Barrier_Price' in leg.deal else None,
                       'premium': premiums[leg.role], 'solved': solved.get(leg.role),
                       # what the runner decided about this leg that the parameters did not say
                       'note': note}, **block)
                 for leg, (_, _, note), block in zip(legs, vegas, blocks)],
        # the finished legs are AT MID, so what they are worth is what the trade marks at, and what
        # the client is quoted is that plus everything the desk charged on the coordinate below
        'net': sum(premiums.values()) + levied + (0.0 if solves else charge),
        'net_mid': sum(premiums.values()),
        # WHICH coordinate carried the desk's charge - the field the recipe solved, or the premium
        # where it solves nothing and the client simply pays more
        'charged_on': solves[0].field if solves else 'premium',
        # the spot the legs were ACTUALLY struck on, read back off the document rather than taken
        # from the caller; the caller owns only the account of where it came from
        'spot': dict({'source': 'book', 'note': None}, **(spot_source or {}),
                     value_market=legs[0].to_market(spot)),
        # every number above is CLIENT-frame; the desk's capture is said once under its own name,
        # and it IS the charge, so it is non-negative by construction
        'edge': levied,
        'spread_note': None if halves else '{} {} - there is no two-way to charge, so the quote '
        'is the mid'.format(FX_VOL_PRICES.format(surface),
                            'carries no Quoted_Bid/Quoted_Ask'
                            if quote_points(document, surface) else 'is not on this book'),
        # what the residual this trade leaves on the book costs to hedge, and what the policy did
        # with it. `scale` is None where the feature never ran; the note says why
        'risk': risk,
        # the MODEL these legs were priced under where it is not the book's own default - the pin
        # `spot_model` wrote on the quote's copy, which `/book/quote` merges into the book
        'valuation_configuration': {
            deal_type: entry for deal_type, entry in (pinned_models(document) or {}).items()
            if entry != already.get(deal_type)} or None,
        'deal': compose(reference, legs, margin)}
    if margin:
        # the charge as declared and as converted. `net` above reads it back NEGATIVE, that being
        # the client's paper; the booked mirror marks the bank's side at plus it
        outcome['margin'] = margin
    return outcome
