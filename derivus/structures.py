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

`quote()` is the runner, and it owns every conversion, once.

MARKET AXIS vs ENGINE AXIS. A desk quotes USDZAR 15.50 - ZAR per USD - while `FXOptionDeal` prices
an option on `Underlying_Currency` settled in `Currency`. When the notional is the pair's QUOTE
currency the deal's axis is the reciprocal of the quoted one, so `Strike_Price = 1/K` and the
option sense inverts with it: a market Call is an engine Put. A barrier's `Barrier_Price` inverts
as a strike does and its DIRECTION flips with it, while In/Out never moves. When the notional is
the BASE currency the two axes agree and nothing is converted. No structure knows any of this.

An ACCRUAL leg asks the axis question once more, and one answer is that a TARGET does not cross:
it is a sum of DIFFERENCES, and `1/S - 1/K` is not the reciprocal of `S - K`, so a TARF is quoted
on the pair's BASE currency and refuses the other side by name. A LEVEL - an accumulator's
knock-out - crosses exactly as a barrier does, and a leverage is a ratio that never converts. See
`furnish_accrual`.

PARAMETERS vs A DEAL. The runner fills the shared block from the parameters - the two currencies
off the pair, the vol surface named for it, the discounting currency, the notional, and
`Expiry_Date` as the book's `Base_Date` plus the quoted tenor - then the leg's pinned block, then
its slots. `expiry` is `<n><D|W|M|Y>` read through `Config.offset_lookup`, or an ISO date for a
broken one; anything else refuses by name rather than landing on today.

Every step prices ONE leg against a deep copy of the whole book document with the deal tree
emptied, since a deal's base valuation does not depend on its siblings and a lone deal compiles
faster per iterate. `Price` is a plain base valuation; `Solve` is `derivus.solve_deal_field`,
bracketed. The runner takes the WHOLE document rather than a market patch, and hands each step
that document.

A SPREAD IS QUOTED; A MID IS BOOKED. The book's `FXVol` surface is bootstrapped from
`Quoted_Market_Value` alone, while the `FXVolPrices` block beside it may carry each pillar's
`Quoted_Bid`/`Quoted_Ask` as data - and this module is that data's only reader. Each leg prices on
its own copy of the book, whose written surface is moved by BOTH of the spreads that block quotes,
signed by the CLIENT's side: the ATM half-spread at the leg's expiry shifts it FLAT, and the RR and
BF halves SKEW it - a wing is composed as `ATM + BF +- RR/2`, so it widens by `BF_half + RR_half/2`
while the ATM node does not, and the smile changes shape rather than level. A solved coordinate
therefore comes out where a desk would deal it. The finished legs are then priced once more against
the unmoved book, and that is `net_mid`. A book carrying no two-way moves every leg by nothing and
quotes exactly as it always has, to the bit.

THE RISK PRICES THE SPREAD. A trade's charge is the cost of hedging the RESIDUAL it leaves on the
book, at the market's own two-way. The composed candidate is MIRRORED - the same verb the booking
uses, so the risk measured and the trade booked are one object - and the book's vol risk is read
with it and without it in QUOTE space, off `Quote_Sensitivity` on the risk run's own copy of the
`FXVolPrices` block. Each bucket's move in ABSOLUTE risk is charged that bucket's own half-spread,
and `participation` of any saving comes off the spread. The `Quote Policy` block declares the
mandate and its ABSENCE is the feature's off switch.

THE SPOT IS LIVE, THE SURFACE IS TICKED. `with_live_spots` is the inverse of `engine_spot` and the
one seam a caller writes a terminal's number through: a spot is `bind='value'` data that moves
between prints, while a delta-quoted surface is read at whatever spot is standing. Every quote says
which it used under `spot`, read off the document it priced.
"""

import copy
import json
import re
import time

from . import utils
from .schema import F, REQUIRED

#: A tenor as the job grammar spells one: a count and a period letter. Anchored and whitespace
#: tolerant, because '3M ' out of a spreadsheet cell is the same tenor.
TENOR = re.compile(r'^\s*(\d+)\s*([DWMY])\s*$', re.IGNORECASE)

#: How wide the runner brackets a strike solve, as a multiple of the market spot. A vanilla's value
#: is monotone in its strike, so any bracket spanning deep in- and out-of-the-money holds the root;
#: `brentq` refuses by name where a zero-cost leg does not sit inside these ends.
STRIKE_BRACKET = (0.25, 4.0)

#: The same bracket for an ACCRUAL strike, moved in. A strip's value saturates at the low end -
#: flat at the discounted `target x notional` once every fixing redeems at once - so nothing is
#: given up, while `0.25 x spot` prices NaN off a surface quoted over moneyness [0.8, 1.2].
ACCRUAL_BRACKET = (0.5, 2.0)

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

#: `<SpotModel>ModelParameters.<non-base token>`, the naming convention
#: `get_spot_model_params_factor` resolves the parameters by (`utils.spot_model_currency` picks the
#: token) - so the presence check here and the engine's own lookup are one key.
SPOT_MODEL_FACTOR = '{}ModelParameters.{}'

#: The model an accrual leg is priced under WHERE THE BOOK CARRIES A CALIBRATION. The switch lives
#: in `Valuation Configuration` per deal TYPE, not on a deal, and both accrual deals declare it in
#: their own `spot_models`.
SPOT_MODEL = 'HestonNandi'

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

#: A year, as the expiry axis of a quote block counts one. Only ever used to place a leg's tenor
#: between two quoted pillars of the spread curve, never as a day count anything is priced on.
DAYS_IN_YEAR = 365.0

#: A barrier's DIRECTION is a statement about the PAIR, so it crosses to the engine axis with the
#: strike: a barrier above USDZAR 18.50 is below 1/18.50 dollars per rand. In/Out says what the
#: payoff does on touch and means the same on either axis, so only Up/Down moves.
BARRIER_FLIP = {'Up_And_In': 'Down_And_In', 'Down_And_In': 'Up_And_In',
                'Up_And_Out': 'Down_And_Out', 'Down_And_Out': 'Up_And_Out'}

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
    """Protection paid for by giving up the other side. The client names the floor they want; the
    cap is whatever strike makes the sold call fund the bought put exactly, which is why it is
    solved rather than quoted."""
    vernacular = 'zero-cost collar, range forward, cylinder'
    fields = [PAIR, EXPIRY, NOTIONAL, NOTIONAL_CURRENCY,
              strike('floor', 'The protected level, bought as a put on the pair')]
    legs = [Leg('protection', 'FXOptionDeal', dict(VANILLA, Option_Type='Put', Buy_Sell='Buy'),
                {'Strike_Price': 'floor'}),
            Leg('financing', 'FXOptionDeal', dict(VANILLA, Option_Type='Call', Buy_Sell='Sell'))]
    recipe = [Price('protection'),
              Solve('financing', 'Strike_Price', -Premium('protection'))]


class Seagull:
    """A collar cheapened by selling a second wing. The client names the floor they want and the
    level below which they are willing to be unprotected again; the cap is solved so the three
    legs sum to nothing."""
    vernacular = 'seagull, three-way, participating collar'
    fields = [PAIR, EXPIRY, NOTIONAL, NOTIONAL_CURRENCY,
              strike('floor', 'The protected level, bought as a put on the pair'),
              strike('lower_floor', 'Where protection stops, sold as a put on the pair')]
    legs = [Leg('protection', 'FXOptionDeal', dict(VANILLA, Option_Type='Put', Buy_Sell='Buy'),
                {'Strike_Price': 'floor'}),
            Leg('participation', 'FXOptionDeal', dict(VANILLA, Option_Type='Put', Buy_Sell='Sell'),
                {'Strike_Price': 'lower_floor'}),
            Leg('financing', 'FXOptionDeal', dict(VANILLA, Option_Type='Call', Buy_Sell='Sell'))]
    recipe = [Price('protection'), Price('participation'),
              Solve('financing', 'Strike_Price', -Premium('protection', 'participation'))]


class ForwardExtra:
    """Protection with the upside left on, paid for by a level rather than by a strike. The client
    is protected at the rate they name and still participates in a favourable move - until the pair
    trades through the barrier, where the sold call knocks in and the whole thing reverts to a plain
    forward at that same protected rate. Nothing is given up at a strike, so the solved coordinate
    is the BARRIER: the level at which the knock-in the client sells funds the put they buy."""
    vernacular = 'forward extra, forward plus, at-worst forward'
    fields = [PAIR, EXPIRY, NOTIONAL, NOTIONAL_CURRENCY,
              strike('protected_rate', 'The protected level, bought as a put on the pair - and the '
                                       'forward the structure reverts to if the barrier trades')]
    legs = [Leg('protection', 'FXOptionDeal', dict(VANILLA, Option_Type='Put', Buy_Sell='Buy'),
                {'Strike_Price': 'protected_rate'}),
            Leg('reversion', 'FXBarrierOption',
                {'Option_Type': 'Call', 'Buy_Sell': 'Sell', 'Barrier_Type': 'Up_And_In'},
                {'Strike_Price': 'protected_rate'})]
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
    """
    vernacular = 'tarf, target redemption forward, target forward'
    fields = [PAIR, EXPIRY, NOTIONAL, NOTIONAL_CURRENCY,
              F('fixing_frequency', 'Period', default=REQUIRED,
                description='How often the strip fixes - 1M, 3M - counted off the book\'s '
                            'Base_Date to the tenor'),
              F('target', 'Float', default=REQUIRED,
                description='The accrual cap that redeems the strip, in the PAIR\'s own units - '
                            '1.5 on USDZAR is 1.50 rand of cumulative favourable move'),
              F('leverage', 'Float', default=2.0,
                description='The loss-side gearing: how many notionals the client deals on an '
                            'unfavourable fixing, against one on a favourable one')]
    legs = [Leg('tarf', 'FXTARFOptionDeal',
                {'Option_Type': 'Call', 'Buy_Sell': 'Buy',
                 'Settlement_Style': 'Cash', 'Option_Style': 'European'},
                {'TargetLevel': 'target'})]
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
    """
    vernacular = 'accumulator, accumulator forward, accu'
    fields = [PAIR, EXPIRY, NOTIONAL, NOTIONAL_CURRENCY,
              F('fixing_frequency', 'Period', default=REQUIRED,
                description='How often the strip fixes - 1M, 3M - counted off the book\'s '
                            'Base_Date to the tenor'),
              strike('knockout', 'The level that cancels the strip when a fixing observes it'),
              F('leverage', 'Float', default=2.0,
                description='The loss-side gearing: how many notionals the client deals on an '
                            'unfavourable fixing, against one on a favourable one')]
    legs = [Leg('accumulator', 'FXAccumulatorOptionDeal',
                {'Option_Type': 'Call', 'Buy_Sell': 'Buy',
                 'Barrier_Type': 'Up_And_Out', 'Barrier_Hit': 'No'},
                {'Barrier_Price': 'knockout'})]
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
    """A quoted pair as `(base, quote)`. 'USDZAR' and 'USD/ZAR' are the same pair; anything that is
    not two three-letter codes refuses, because guessing the split mis-books a trade."""
    codes = str(pair).split('/') if '/' in str(pair) else [str(pair)[:3], str(pair)[3:]]
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

    A tenor is `<n><D|W|M|Y>` read through `Config.offset_lookup`, so the letters mean here what
    they mean in a job's date grid. An ISO date passes through for a broken date, and anything else
    refuses by name - an unparsed tenor landing on the base date is a zero-day option.
    """
    import pandas as pd
    from .config import Config
    found = TENOR.match(str(expiry))
    if found:
        offset = pd.DateOffset(**{Config.offset_lookup[found.group(2).upper()]: int(found.group(1))})
        return {'.Timestamp': (base_date + offset).strftime('%Y-%m-%d')}
    try:
        return {'.Timestamp': pd.Timestamp(expiry).strftime('%Y-%m-%d')}
    except (ValueError, TypeError):
        raise ValueError('{!r} is not a tenor (3M, 1Y) or a date'.format(expiry))


def fixing_grid(base_date, expiry, frequency):
    """An accrual deal's fixing SCHEDULE, in the wire form both declarations read.

    `[[fixing, settlement, observed], ...]` - the row shape `TARF_ExpiryDates` and
    `Accumulator_ExpiryDates` share, untagged, with the observed fixing 0.0 because a quote is
    struck today. Fixings run from the book's `Base_Date` at `frequency` up to and including the
    tenor; each settles `FIXING_LAG` days later.

    Each fixing is `base + n x frequency` rather than a step off the previous one, because an
    offset applied repeatedly from a month end walks (31 Jan + 1M + 1M is 28 Mar, not 31 Mar).

    A tenor holding no whole fixing period refuses rather than returning an empty strip, and so
    does a frequency that does not DIVIDE the tenor: the loop would stop at the last fixing that
    fits, leaving the strip short of the tenor that was quoted while the ticket still said 1Y.

    The base is normalized to MIDNIGHT first, as `expiry_date` normalizes its answer: a `Base_Date`
    carrying a time would put the final fixing one comparison past a midnight last date, and a
    twelve-fixing year would quietly be eleven.
    """
    import pandas as pd
    from .config import Config
    found = TENOR.match(str(frequency))
    if not found:
        raise ValueError('{!r} is not a fixing frequency - 1M, 3M, 1W'.format(frequency))
    period, count = Config.offset_lookup[found.group(2).upper()], int(found.group(1))
    base_date = pd.Timestamp(base_date).normalize()
    last, rows, step = timestamp(expiry_date(base_date, expiry)), [], 1
    while True:
        fixing = base_date + pd.DateOffset(**{period: count * step})
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


def declared(structure, params):
    """`params` completed by the structure's OWN declared defaults.

    Almost every parameter is `REQUIRED` - a strike a client did not name is a strike nobody
    agreed. A market CONVENTION is the exception, and the number belongs on the `F` descriptor
    where `describe_structure` publishes it rather than in a `.get` inside the runner.
    """
    stated = {f.key: f.default for f in structure.fields if f.default is not REQUIRED}
    return dict(stated, **params)


def spot_model(document, deal_type, underlying, settlement):
    """Pin `HestonNandi` on `deal_type` where THIS book carries a calibration for the leg's pair.
    Returns the leg's note, or `None` when the model was pinned.

    The switch is a `Valuation Configuration` entry per deal TYPE rather than a deal field, and the
    parameters resolve by naming convention off the pair's NON-BASE token - the same
    `utils.spot_model_currency` rule the engine's own lookup takes - so the presence check here and
    that lookup are one key. It has to be made here: the switch on with the factor absent raises
    inside the engine's dependency loop, which SKIPS the deal and logs an ERROR, so the quote would
    return with its only leg priced at nothing.

    A CROSS - neither leg the book's base - keeps the underlying's name and is out of the ruling's
    scope, both legs being simulated factors whose composed law nothing fits. A book that declares
    no `Base_Currency` at all REFUSES here rather than guessing a token: pinning the wrong one is a
    switch the engine then looks up under the other name, which is that same leg priced at nothing.

    Writes IN PLACE on the document the runner holds. Every pricing deep-copies that document
    through `alone`, so one write reaches every iterate of the solve.
    """
    factors = market_data(document)
    token = utils.spot_model_currency(underlying, settlement, base_currency(document))
    factor = SPOT_MODEL_FACTOR.format(SPOT_MODEL, token)
    if factor in factors:
        document['Calc']['MergeMarketData']['ExplicitMarketData'].setdefault(
            'Valuation Configuration', {}).setdefault(deal_type, {})['SpotModel'] = SPOT_MODEL
        return None
    return ('priced GBM - a spot model is keyed off the pair\'s non-base token, so this leg looked '
            'up {}, which this book does not carry. The HestonNandiModelParameters bootstrapper '
            'installs it: calibrate {} (/book/hn)'.format(factor, token))


def pinned_models(document):
    """The `Valuation Configuration` a quote's own passes pinned on this document, or `None`.

    `spot_model` writes the switch onto the document the runner holds; this reads it back so the
    outcome can REPORT it. A leg priced under Heston-Nandi that books into a book marking it GBM is
    a mark disagreeing with the price it was dealt at, and the approval is where that is fixed.
    """
    pinned = document.get('Calc', {}).get('MergeMarketData', {}).get(
        'ExplicitMarketData', {}).get('Valuation Configuration')
    return copy.deepcopy(pinned) if pinned else None


def pin_models(document, deal, pinned):
    """`pinned` merged into a document's `Valuation Configuration`, per deal TYPE, in place.

    The booking half of `pinned_models`: an approval carries the quote's own pins onto the BOOK, so
    a leg dealt under Heston-Nandi re-marks under Heston-Nandi. Merged per type and per key rather
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


def base_currency(document):
    """The reporting currency every `FxRate.<ccy>.Spot` in this document is quoted in, or `None`
    where the document does not declare one.

    Read off the EXPLICIT block alone, which is the same half of the book `market_data` reads and
    the only half a quote can write; the engine reads the SAME declaration off its merged params,
    so a `MarketDataFile` that is not repeated here answers `None` rather than the engine's number.
    None is never resolved into a token - `utils.spot_model_currency` refuses it - so the two reads
    can differ only into a refusal, never into a different token.
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

    One reader for the pillar-keyed halves and the skew's own rebuild, because a spread charged off
    a row the surface was not built from is a spread charged on nothing: `Use` holds a quote out of
    the bootstrap without deleting it, and it has to hold it out of the quote layer by the same
    token.

    The filter is SPELLED here rather than borrowed, for the default alone: `Use` is an optional
    field a block states to hold a row OUT, so its absence is 'Yes', and a hand-authored mid-only
    block that omits it must quote rather than raise. That is `atm_two_way`'s own reading of the
    same field, and the two are now one rule.
    """
    prices = document.get('Calc', {}).get('MergeMarketData', {}).get(
        'ExplicitMarketData', {}).get('Market Prices', {})
    block = (prices.get(FX_VOL_PRICES.format(surface)) or {}).get('instrument')
    return [point for point in block['Points'] if point.get('Use', 'Yes') == 'Yes'] \
        if block and block.get('Points') else []


def atm_two_way(document, surface):
    """The ATM half-spread the book carries for `surface`, as sorted `[(expiry, half), ...]` in the
    surface's own vol units - `(ask - bid) / 2` off the quote block's ATM rows, and empty when the
    block carries no two-way at all.

    The block is `Market Prices` DATA: the bootstrap reads `Quoted_Market_Value` by name and
    nothing else, so the written surface is the mid one whether or not these sides are there.

    A row missing either side is not a two-way and is skipped. A CROSSED one reads as zero-wide
    rather than as a negative spread - a stale bid through a live offer is a broken print, and a
    desk must not pay a client for it.

    ATM rows ONLY, because this is the half that moves the surface FLAT. The RR and BF rows carry
    their own two-way and it skews the smile instead - `wing_two_way` composes that one.
    """
    prices = document.get('Calc', {}).get('MergeMarketData', {}).get(
        'ExplicitMarketData', {}).get('Market Prices', {})
    block = prices.get(FX_VOL_PRICES.format(surface)) or {}
    rows = []
    for point in block.get('instrument', {}).get('Points', []):
        if point.get('Quote_Type') != 'ATM' or point.get('Use', 'Yes') != 'Yes':
            continue
        bid, ask = point.get('Quoted_Bid'), point.get('Quoted_Ask')
        if bid is None or ask is None:
            continue
        rows.append((float(point['Expiry']), max(0.0, 0.5 * (float(ask) - float(bid)))))
    return sorted(rows)


def half_spread(rows, expiry):
    """The ATM half-spread at `expiry` years: linear between quoted pillars, FLAT past either end.

    Flat rather than extrapolated: a spread continued as a straight line off the last two pillars
    is a number the market never quoted, and the ends are where that line goes furthest wrong.
    """
    if not rows:
        return 0.0
    if expiry <= rows[0][0]:
        return rows[0][1]
    for (left, low), (right, high) in zip(rows, rows[1:]):
        if expiry <= right:
            span = right - left
            return high if not span else low + (high - low) * (expiry - left) / span
    return rows[-1][1]


def quote_two_way(document, surface):
    """EVERY quoted pillar's half-spread for `surface`, keyed by the descriptor `dV/dq` reports
    that quote under - `{'ATM 1': 0.002, 'RR 0.25 1': 0.001, ...}`, in the surface's own vol units.

    `atm_two_way`'s reading widened to the whole block, and the two are not redundant: that one is
    an ATM curve read BETWEEN pillars, for what a vanilla deals on at a tenor, while this is a
    lookup per quote and never interpolated - a bucket IS a quoted pillar or it is not a bucket.

    The descriptor and the used-quote filter both come from `FXVolSurfaceParameters`, so these keys
    are the ones the bootstrap's own leaves are published under; a second copy of the naming rule
    would drift into silently pricing no bucket at all. A row missing either side is skipped and a
    CROSSED one reads zero-wide, for the reason `atm_two_way` states.
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


def wing_two_way(document, surface):
    """What a WING costs over the ATM half-spread, as `{(expiry, pillar): half}` in the surface's
    own vol units - and empty where the block quotes no RR or BF two-way at all.

    Composed through the strangle algebra the surface was built from, which fixes both the number
    and the fact that there is ONE per pillar rather than one per wing. A wing vol is
    `ATM + BF +- RR/2`, so its offered side is the offered side of every term it is made of and the
    SUBTRACTED term takes its bid: `put_ask = ATM_ask + BF_ask - RR_bid/2`. Half the ask-less-bid
    of a linear combination is each term's own half times the size of its coefficient, summed, so
    both wings widen by `BF_half + RR_half/2` - the risk reversal's spread reaches both wings, its
    SIGN does not.

    The ATM half is deliberately not in here: `with_vol_shift` already carries that one FLAT across
    every node, so what this returns is the part that changes the smile's SHAPE.

    A row missing either side is not a two-way and is skipped, a CROSSED one reads zero-wide for
    the reason `atm_two_way` states, and a pillar composing to zero is DROPPED rather than carried -
    a book quoting no wing spread has to price down the path it always did, byte for byte.
    """
    weight = {'BF': 1.0, 'RR': 0.5}
    halves = {}
    for point in quote_points(document, surface):
        bid, ask = point.get('Quoted_Bid'), point.get('Quoted_Ask')
        if point['Quote_Type'] not in weight or bid is None or ask is None:
            continue
        pillar = (float(point['Expiry']), float(point['Pillar']))
        halves[pillar] = halves.get(pillar, 0.0) + weight[point['Quote_Type']] * max(
            0.0, 0.5 * (float(ask) - float(bid)))
    return {pillar: half for pillar, half in sorted(halves.items()) if half > 0.0}


def leg_expiry(document, deal):
    """A leg's tenor in years, on the quote block's own expiry axis - the coordinate the spread
    curve is read at, never a day count a price comes off.

    An accumulator declares no `Expiry_Date`, so a strip's tenor is its LAST SETTLEMENT, the same
    date the TARF writes into the field it does declare. The ATM half-spread is read once per leg,
    so a strip takes the one at its longest fixing - the widest of the ones it spans - rather than
    being quoted tighter than any of them.
    """
    end = deal['Expiry_Date'] if 'Expiry_Date' in deal \
        else deal[SCHEDULE_FIELD[deal['Object']]][-1][1]
    days = (timestamp(end) - timestamp(document['Calc']['Calculation']['Base_Date'])).days
    return max(0.0, days / DAYS_IN_YEAR)


def wing_skew(factor, rows, quotes, wings):
    """The vol move each row of a written log-moneyness surface takes when the WINGS widen by
    `wings` - `{(expiry, pillar): signed half}` - and the ATM row does not.

    ONE spelling of the delta-to-log-moneyness conversion, and it is the shipped one. The widening
    goes onto the QUOTES as a butterfly per pillar, and `smile` -> `malz_skews` -> `malz_surface`
    then runs twice over the surface's OWN x-grid: once on the quotes as they stand, once on the
    widened ones. What comes back is the DIFFERENCE, so a written surface these quotes did not
    build keeps whatever else was done to it and only the skew moves.

    A BUTTERFLY is where the widening goes because the strangle algebra puts it there:
    `vol(call) = ATM + BF + RR/2` and `vol(put) = ATM + BF - RR/2`, so `BF` enters both wings with a
    coefficient of one and the ATM row with none - a widening that spares the ATM node, which is
    what leaves that node carrying the flat shift and nothing else. The RISK REVERSAL is not
    touched: its own spread is already inside `wings`, and moving that quote would tilt one wing up
    and the other down rather than widening both. A pillar quoting a two-way on the risk reversal
    ALONE has no butterfly row to widen, so one is authored at the composed half - on the quote
    copy, which nothing outside this function reads.

    An expiry the written surface carries and the quotes do not REFUSES by name: a skew is composed
    from the quotes the surface was built from, and there is no smile there to widen.
    """
    import numpy as np
    from .bootstrappers import FXVolSurfaceParameters
    from .riskfactors import Factor2D

    # the surface's own x-grid, expiry by expiry, and where each of its rows sits in the written
    # block - `malz_surface` walks this dict in insertion order, so the two travel together
    grid, index = {}, {}
    for position, row in enumerate(rows):
        grid.setdefault(float(row[1]), []).append(float(row[0]))
        index.setdefault(float(row[1]), []).append(position)

    widened = [dict(point) for point in quotes]
    butterflies = {(float(point['Expiry']), float(point['Pillar'])): point
                   for point in widened if point['Quote_Type'] == 'BF'}
    for pillar, half in sorted(wings.items()):
        point = butterflies.get(pillar)
        if point is None:
            widened.append({'Use': 'Yes', 'Expiry': pillar[0], 'Pillar': pillar[1],
                            'Quote_Type': 'BF', 'Quoted_Market_Value': half, 'Timestamp': ''})
        else:
            point['Quoted_Market_Value'] = float(point['Quoted_Market_Value']) + half

    def evaluate(points):
        smile = FXVolSurfaceParameters.smile(points)
        skews = Factor2D.malz_skews(smile, np.unique(smile[:, 1]))
        missing = [expiry for expiry in grid if expiry not in skews]
        if missing:
            raise ValueError(
                '{} carries expiry {:g}, which its quote block does not - a wing spread is '
                'composed from the quotes the surface was built from, so re-bootstrap the surface '
                'or drop the wing two-way'.format(factor, missing[0]))
        return Factor2D.malz_surface(
            skews, {expiry: np.array(nodes) for expiry, nodes in grid.items()})

    moves = [0.0] * len(rows)
    for position, was, now in zip([position for expiry in grid for position in index[expiry]],
                                  evaluate(quotes), evaluate(widened)):
        moves[position] = now[2] - was[2]
    return moves


def with_vol_shift(document, factor, shift, quotes=None, wings=None):
    """The book with `FXVol.<pair>` moved by `shift` vol points flat AND its smile skewed by the
    signed wing half-spreads `wings` - the copy ONE SIDE of the spread prices on, since the two
    sides of a structure need the book at two different vols at once.

    TWO moves, because the block quotes two kinds of spread. The ATM half-spread widens the whole
    surface FLAT, which is its level; the wing halves widen the wings and spare the ATM node, which
    is its shape - `wing_skew` composes that one off `quotes`, the rows the surface was built from,
    through the same strangle algebra it was built by. `wings` and `quotes` travel together: there
    is no skewing a surface without the quotes that made it.

    With NEITHER move to make this hands back the document ITSELF, uncopied: a book carrying no
    two-way prices down the identical path it always did and pays nothing, not even a deep copy.
    Where there is no skew each row takes `shift + 0.0`, which is the float `shift` was.

    Moving the WRITTEN surface is what a leg then prices on, because `run_job` does not bootstrap -
    the block that built this surface is not read again inside a pricing run.
    """
    if not shift and not wings:
        return document
    moved = copy.deepcopy(document)
    factors = market_data(moved)
    if factor not in factors:
        raise ValueError('{} is missing - a two-sided quote moves the written surface, so there '
                         'has to be one'.format(factor))
    surface = factors[factor]['Surface']
    # the wire form is `{'.Curve': {'data': [[moneyness, expiry, vol], ...]}}`; a hand-authored
    # surface is the bare list. The vol is the last column of either
    rows = surface['.Curve']['data'] if isinstance(surface, dict) else surface
    skew = wing_skew(factor, rows, quotes, wings) if wings else [0.0] * len(rows)
    for row, move in zip(rows, skew):
        row[-1] += shift + move
    return moved


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
    """
    params = declared(structure, params)
    base, quote_ccy = split_pair(params['pair'])
    underlying = str(params['notional_currency']).upper()
    if underlying not in (base, quote_ccy):
        raise ValueError('notional_currency {!r} is not a side of {}'.format(
            underlying, params['pair']))
    settlement = quote_ccy if underlying == base else base
    # the quoted axis is the deal's own only when the notional is the pair's BASE currency
    inverted = underlying == quote_ccy
    base_date = timestamp(document['Calc']['Calculation']['Base_Date'])
    shared = {'Currency': settlement, 'Discount_Rate': settlement,
              'Underlying_Currency': underlying, 'Underlying_Amount': float(params['notional']),
              'FX_Volatility': '{}.{}'.format(base, quote_ccy),
              'Expiry_Date': expiry_date(base_date, params['expiry'])}
    seed = engine_spot(document, underlying, settlement)

    out = []
    for leg in structure.legs:
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
            if 'Option_Type' in deal:
                # a call on the pair is a put on the quote currency
                deal['Option_Type'] = 'Put' if deal['Option_Type'] == 'Call' else 'Call'
            if 'Barrier_Type' in deal:
                deal['Barrier_Type'] = BARRIER_FLIP[deal['Barrier_Type']]
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
    SETTLEMENT rather than to the tenor, since a strip is not over until its final cashflow lands.
    `FXAccumulatorOptionDeal` declares no such field, so the shared block's is REMOVED - a deal
    block is the field dict the pricer reads.

    THE NOTIONALS. `Underlying_Amount` is the notional per fixing, already in `notional_currency`;
    `LeverageNotional` is `leverage` times it. Neither has an axis: `notional_currency` IS the
    underlying, and a gearing is a pure ratio.

    THE TARGET does not cross. It is a sum of DIFFERENCES, and `1/S - 1/K` is not the reciprocal of
    `S - K`, so no number in reciprocal units means the same accrual cap. `InvertedTarget` is not
    the answer either - it moves the whole fixing onto the reciprocal of the deal axis, so the deal
    would pay `Underlying_Amount` per unit of MOVE in the pair, which is a different product (0.77%
    apart in the solved strike on the gate's book). So `InvertedTarget` is False on every leg the
    runner builds and a TARF quoted on the pair's QUOTE currency REFUSES by name. An accumulator
    has no target and crosses both axes freely.

    THE MODEL. `spot_model` pins Heston-Nandi where the book carries a calibration for this leg's
    PAIR - keyed off its non-base token, whichever side the leg is written from - and hands back the
    note where it does not.

    The axis refusal is this function's FIRST statement, before the schedule and the notionals: a
    refusal firing later has already written the block the caller holds.
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
    from .config import splice_deal
    iterate = copy.deepcopy(document)
    iterate['Calc']['Deals']['Deals']['Children'] = []
    iterate['Calc']['Calculation']['Object'] = 'BaseValuation'
    return iterate, splice_deal(iterate, deal)


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


def run_solve(document, leg, field, target, spot):
    """`derivus.solve_deal_field` over one leg, bracketed, writing the answer back onto the leg.

    A strike is bracketed around the market spot by `STRIKE_BRACKET` - `ACCRUAL_BRACKET` for a
    strip - and crossed to the engine axis, where inverting swaps the ends, so they are sorted
    rather than assumed. A BARRIER is bracketed on the side its own type lives on, off the same
    ends. Any other field is left to the secant from its current value, which is exact in two
    pricings for anything the value is affine in. Returns `(solved, premium at the solved value)`.
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
    try:
        solved, _, _, out = solve_deal_field(iterate, deal_path, field, target=target, bounds=bounds)
    except ValueError as error:
        if bounds is None or 'different signs' not in str(error):
            raise
        # brentq's sign check speaks in f(a) and f(b); a desk needs the economics said out loud
        raise ValueError(
            '{}: no {} in [{:.6g}, {:.6g}] lets this leg reach {:.6g} - the structure cannot be '
            'financed at these parameters'.format(
                leg.deal['Reference'], field, bounds[0], bounds[1], target))
    leg.deal[field] = solved
    return solved, own_value(out, leg.deal['Reference'])


def compose(reference, legs):
    """The priced legs as ONE bookable deal: a `StructuredDeal` whose `Children` are the legs with
    their solved values in place, exactly as they were priced. Settled in the legs' own settlement
    currency, so the container nets what the parts report without a cross of its own.

    The children sit INSIDE the deal here, because a quote is filed, hashed and read back whole.
    The deal TREE holds a container's children one level out - see `book_node`.
    """
    return {'Object': 'StructuredDeal', 'Reference': reference,
            'Currency': legs[0].deal['Currency'], 'Net_Cashflows': 'Yes',
            'Children': [{'Instrument': {'.Deal': dict(leg.deal)}} for leg in legs]}


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
    between them. Sales margin never enters: a mirror is a pure change of side.
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
    from .config import walk_job_deals

    try:
        children = document['Calc']['Deals']['Deals']['Children']
    except (KeyError, TypeError):
        return []
    return sorted(node['Instrument']['.Deal'].get('Reference')
                  for _, node in walk_job_deals(children)
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
      - `scope` - 'vol', which is all v1 measures; anything else refuses rather than silently
        pricing a scope nobody implemented
      - `bucket_limit` - a per-bucket cap on `|risk after|` in the bucket's own vega units, past
        which NO tightening applies however good the saving looks
      - `min_ticket_bp` - flat bp of notional, the ops floor under the edge
      - `firm_seconds` - how long a quote stays approvable; the approval verb reads it, and this
        module only carries it through so a desk states its mandate in ONE block

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

    Three edits and no others. The calculation becomes a `BaseValuation` asking for
    `Greeks: 'First'` - one backward off the ROOT netting set, so a leaf's `.grad` is the whole
    portfolio's. The candidate's nodes are appended. And `Quote_Sensitivity` goes to Yes on the
    `FXVolPrices` block, which is what makes `Config.bootstrap` leave the surface behind still
    connected to the ATM/RR/BF quotes it was built from.

    That switch is worth exactly ZERO in the forward pass - `leaf + (theta - theta.detach())` - so
    turning it on cannot move a price.
    """
    run = copy.deepcopy(document)
    run['Calc']['Calculation'] = dict(run['Calc']['Calculation'],
                                      Object='BaseValuation', Greeks='First')
    children = list(run['Calc']['Deals']['Deals'].get('Children') or [])
    run['Calc']['Deals']['Deals']['Children'] = children + list(nodes)
    prices = run['Calc']['MergeMarketData']['ExplicitMarketData'].get('Market Prices', {})
    block = prices.get(FX_VOL_PRICES.format(surface))
    if not block:
        raise ValueError('{} is missing - quote-space risk is read off the block the surface was '
                         'bootstrapped from'.format(FX_VOL_PRICES.format(surface)))
    block['instrument'] = dict(block['instrument'], Quote_Sensitivity='Yes')
    return run


def vol_risk(document, nodes, surface):
    """`{descriptor: dV/dq}` for the book plus `nodes` - the vol book in QUOTE coordinates.

    The attachment is harvested at BOOTSTRAP rather than at run, so this bootstraps its own copy
    and prices it in the SAME `Context`: the leaves `Config.quote_leaves` publishes are the tensors
    `Calculation.factor_leaf` was offered, and one backward off the root leaves `.grad` on them.

    Descriptors are SUMMED across every published block, which is the collision rule - one JSON
    number can feed two chains, and each family's partial is correct while neither is the answer.

    An EMPTY deal tree has no value to differentiate - `backward()` on a constant refuses - and its
    risk is a zero vector, so it never reaches a run.
    """
    from . import Context
    from .config import CustomJsonEncoder
    run = risk_document(document, nodes, surface)
    if not run['Calc']['Deals']['Deals']['Children']:
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
    quote. The etag is over everything the greeks run reads - the deal tree, the WHOLE market
    section and the Calculation block - because a rolled `Base_Date` or a changed report currency
    over an unmoved book is a different risk vector.
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


def run_recipe(document, structure, params, reference, two_way, wings, surface, spot, scale):
    """One whole pass of the recipe at `scale` times the book's half-spreads, from fresh legs.

    Materializes the legs, signs each one's spread by the CLIENT's side, builds the shifted and
    skewed book each side needs, runs the steps in order, and marks the finished legs once more at
    mid. `scale` is the ONE thing that differs between the base pass and a re-quote: it multiplies
    the ATM half-spread AND the wing halves, and rides the same two-sided machinery rather than a
    second shift path.

    Both spreads take the client's OWN sign, and it is the same sign: what the client buys is
    offered at the ask of every quote the vol is composed of, what they sell is taken at the bid of
    each. So a leg's copy of the book is named by `(shift, sign)` and legs sharing both share one
    copy - every pricing deep-copies again through `alone`, so nothing here is mutated.

    Returns `(legs, spreads, premiums, solved, mid)`. Legs are fresh because `run_solve` writes the
    solved value back onto the leg it moved, so a second pass over the first pass's legs would seed
    itself from the answer it is meant to find.
    """
    legs = materialize(structure, params, document)
    for leg in legs:
        leg.deal['Reference'] = '{}_{}'.format(reference, leg.role)
        leg.deal['Structure_Reference'] = reference
    by_role = {leg.role: leg for leg in legs}

    quotes = quote_points(document, surface) if wings else None
    spreads, books, by_side = {}, {}, {}
    for leg in legs:
        # a leg's Buy_Sell is the CLIENT's side - what they buy is offered at the ask vol and what
        # they sell is taken at the bid - so the client's side IS the sign of the shift, and a leg
        # stating no side refuses rather than defaulting: either guess charges it the wrong way
        side = leg.deal.get('Buy_Sell')
        if (two_way or wings) and side not in ('Buy', 'Sell'):
            raise ValueError('{}: leg {} carries no Buy_Sell, so which side of the two-way it '
                             'deals on is not stated'.format(structure.__name__, leg.role))
        sign = 1.0 if side == 'Buy' else -1.0
        spreads[leg.role] = scale * sign * half_spread(
            two_way, leg_expiry(document, leg.deal)) if (two_way or wings) else None
        shift = spreads[leg.role] or 0.0
        # a scale of zero is a quote AT the mid, so the skew empties rather than widening by zero
        skewed = {pillar: scale * sign * half for pillar, half in wings.items()
                  if scale * half} if wings else None
        side_of = (shift, sign if skewed else 0.0)
        if side_of not in by_side:
            by_side[side_of] = with_vol_shift(
                document, FX_VOL_FACTOR.format(surface), shift, quotes, skewed)
        books[leg.role] = by_side[side_of]

    premiums, solved = {}, {}
    for step in structure.recipe:
        if step.role not in by_role:
            raise ValueError('{}: the recipe names leg {!r}, which is not declared'.format(
                structure.__name__, step.role))
        leg = by_role[step.role]
        if isinstance(step, Price):
            premiums[leg.role] = run_price(books[leg.role], leg.deal)
        elif isinstance(step, Solve):
            # the target is the legs already priced ON THEIR OWN SIDES, so a solved coordinate
            # finances the structure at the vols it was really quoted at
            target = step.target.value(premiums) if isinstance(step.target, Premium) \
                else float(step.target)
            value, premiums[leg.role] = run_solve(books[leg.role], leg, step.field, target, spot)
            solved.setdefault(leg.role, {})[step.field] = value
        else:
            raise ValueError('{}: {!r} is not a recipe step'.format(structure.__name__, step))

    unpriced = [leg.role for leg in legs if leg.role not in premiums]
    if unpriced:
        raise ValueError('{}: the recipe never prices {}'.format(
            structure.__name__, ', '.join(unpriced)))

    # one more pass, at MID, over the legs as finally solved: the number the trade marks at once
    # booked, since the spread belongs to the quote and the mid belongs to the book
    mid = {leg.role: run_price(document, leg.deal) for leg in legs}
    return legs, spreads, premiums, solved, mid


def risk_impact(document, params, reference, sided, surface, legs, premiums, mid):
    """The whole risk-impact step over a candidate already quoted at the full two-way.

    Measures the book with the candidate's MIRROR on it and without, prices the difference at the
    market's own two-way, applies the policy, and hands back the `risk` block the outcome carries -
    `scale` included, which is what the re-quote multiplies every half-spread by.

    Four ways out, each leaving `scale` at None - nothing to re-quote - with the reason NAMED
    rather than reported as a scale of 1 nobody can distinguish from a decision:

      - the book declares no `Quote Policy`, so the feature is off
      - the book quotes no two-way of ANY kind - `sided` - so there is no spread to tighten and no
        half-spread to price a bucket at either
      - the full charge is not positive - a two-way that captured nothing has nothing to give back
      - the book publishes no vol quote leaves, so there are no coordinates to measure in
    """
    policy = read_policy(document)
    empty = {'coordinates': 'quote-space', 'buckets': [], 'saving': None, 'charge_full': None,
             'charge_effective': None, 'scale': None, 'policy': policy}
    if policy is None:
        return dict(empty, note='the book declares no {} block - the quote is the full two-way '
                                'spread, exactly as it was before'.format(QUOTE_POLICY))
    if not sided:
        return dict(empty, note='{} carries no two-way - there is no spread to tighten'.format(
            FX_VOL_PRICES.format(surface)))
    charge_full = sum(premiums.values()) - sum(mid[leg.role] for leg in legs)
    if charge_full <= 0.0:
        return dict(empty, charge_full=charge_full,
                    note='the two-way captured {:.6g} - there is nothing to give back'.format(
                        charge_full))

    # the MIRROR is the desk's side, and the same verb the approval books through, so the risk
    # measured and the trade booked cannot disagree by a sign
    candidate = book_node(mirror(compose(reference, legs)))
    before = book_risk(document, surface)
    after = vol_risk(document, [candidate], surface)
    halves = quote_two_way(document, surface)
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


def quote(document, structure_name, params, spot_source=None, netting_set=None):
    """Price a structure against a book, and hand back the quote plus the deal it would book.

    `document` is a wire-form job document - the book - and travels whole, never a patch. `params`
    are the client's numbers, in the market's own terms, and come back COMPLETED by the structure's
    own declared defaults. The answer carries `quote_id`, `structure`, `params`, `netting_set`, a
    row per leg (reference, role, deal type, side, market-terms strike and barrier, premium,
    whatever was solved, the vol spread it took and any `note`), `net`, `net_mid`, `edge`, `spot`,
    `risk`, `valuation_configuration`, and `deal` - the composed `StructuredDeal` in wire form,
    ready for the booking verb.

    `quote_id` hashes the structure, the parameters, the market the book was carrying AND a
    submission clock: a quote is an ACT, so two identical asks minutes apart are two quotes.

    THE TWO-SIDED HALF. Where the book's `FXVolPrices` block carries a two-way, each leg prices on
    its own copy of it, moved by the two spreads that block quotes. The ATM half-spread at the
    leg's expiry moves the written surface FLAT - `+half` where the CLIENT buys, `-half` where they
    sell - and the RR/BF halves SKEW it, widening each quoted pillar's wings by `BF_half +
    RR_half/2` on the same side and sparing the ATM node. `net` is therefore the two-sided price
    the client is quoted. Each leg reports its signed flat shift as `vol_spread` in the surface's
    own units (0.002 is 0.2 vol points), the composed wing halves are reported once as
    `wing_spread`, and BOTH are reported at the scale they were actually charged at - a tightened
    re-quote describes the quote it made rather than the two-way the book carries.

    `wing_spread` is None where the block quotes no wing two-way. `vol_spread` is None only where
    the book quotes NO two-way at all, in which case every leg prices on the document itself and
    `spread_note` names the absence: a WINGS-ONLY book has a real skew and a flat half of zero, so
    its legs report a signed zero - the shift that was applied - rather than an absence.

    `net_mid` is the finished legs priced once more against the UNSHIFTED book - what the trade
    marks once booked - in the same sign convention, so the desk's captured `edge` is
    `net - net_mid`, positive when a spread was charged.

    WHICH SPOT. `spot` names the market this quote was struck on: `value_market` is the pair as the
    client quotes it, READ off the document the legs priced against rather than taken from the
    caller, beside the caller's `source` ('terminal' or 'book') and its `note`.

    THE RISK-IMPACT HALF, off unless the book declares a `Calc['Quote Policy']` block. Where it
    does, the base pass is quoted at the full two-way, its composed candidate is MIRRORED into the
    desk's side, and the book's vol risk is measured with it and without it in quote coordinates.
    Each bucket's move in ABSOLUTE risk times that bucket's own half-spread is what hedging the
    residual costs; a negative total is a SAVING, and `participation` of it comes off the charge.
    The re-quote then runs the whole recipe again with every leg's half-spread multiplied by
    `scale = charge_effective / charge_full`.

    ONE PASS, NOT A FIXED POINT - a stated approximation. The risk was measured on the FULL-SPREAD
    candidate while the re-solve moves the solved coordinate, so the tightened structure's residual
    is not exactly the one priced. The move is second order, and the `risk` block's buckets are the
    full-spread candidate's. A risk-ADDING trade quotes at the full spread: the market's own spread
    is the ceiling and the mid is the floor, so `scale` stays in [0, 1].

    WHAT MODEL IT WAS PRICED UNDER. `valuation_configuration` is what THIS quote's passes pinned -
    never what the book already declared - or `None`. The pin lives on the quote's copy of the
    document and dies with it, so `/book/quote` merges it into the book as part of the booking act
    and a leg dealt under a GARCH does not re-mark as a lognormal.

    WHO IT IS FOR. `netting_set` names an existing `NettingCollateralSet` - the CLIENT, since that
    is where the counterparty and the CSA are declared - and the approval books the mirror UNDER
    that node, which is what puts the trade inside the subtree the CVA projection prices. It is
    checked against THIS document before anything is priced. `None` is the root booking.
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
    # a declared default is part of what was quoted, so it is filled in before the id is hashed and
    # before the outcome reports the parameters, rather than inside `materialize` alone
    params = declared(structure, params)
    quote_id = content_hash({
        'structure': structure_name, 'params': params, 'netting_set': netting_set,
        'market': document.get('Calc', {}).get('MergeMarketData', {}).get('ExplicitMarketData', {}),
        'at': time.perf_counter()})
    reference = '{}-{}'.format(structure_name, quote_id[:8])

    probe = materialize(structure, params, document)[0]
    spot = engine_spot(document, probe.deal['Underlying_Currency'], probe.deal['Currency'])
    surface = probe.deal['FX_Volatility']
    two_way, wings = atm_two_way(document, surface), wing_two_way(document, surface)

    legs, spreads, premiums, solved, mid = run_recipe(
        document, structure, params, reference, two_way, wings, surface, spot, 1.0)
    risk = risk_impact(document, params, reference, bool(two_way or wings), surface,
                       legs, premiums, mid)
    # what the halves were CHARGED at, which is what the outcome below has to describe: the base
    # pass deals at the full two-way, a tightened re-quote at the policy's own scale
    charged = 1.0
    if risk['scale'] is not None and risk['scale'] < 1.0:
        charged = risk['scale']
        legs, spreads, premiums, solved, mid = run_recipe(
            document, structure, params, reference, two_way, wings, surface, spot, charged)

    return {
        'quote_id': quote_id, 'structure': structure_name, 'params': dict(params),
        # WHO the quote is for, always said: a null is the root booking, never an unanswered
        # question
        'netting_set': netting_set,
        'legs': [{'reference': leg.deal['Reference'], 'role': leg.role,
                  'deal_type': leg.deal['Object'], 'buy_sell': leg.deal.get('Buy_Sell'),
                  'strike_market': leg.to_market(leg.deal['Strike_Price'])
                  if 'Strike_Price' in leg.deal else None,
                  'barrier_market': leg.to_market(leg.deal['Barrier_Price'])
                  if 'Barrier_Price' in leg.deal else None,
                  'premium': premiums[leg.role], 'solved': solved.get(leg.role),
                  'vol_spread': spreads[leg.role],
                  # what the runner decided about this leg that the parameters did not say
                  'note': leg.note}
                 for leg in legs],
        'net': sum(premiums.values()),
        'net_mid': sum(mid[leg.role] for leg in legs),
        # the spot the legs were ACTUALLY struck on, read back off the document rather than taken
        # from the caller; the caller owns only the account of where it came from
        'spot': dict({'source': 'book', 'note': None}, **(spot_source or {}),
                     value_market=legs[0].to_market(spot)),
        # every number above is CLIENT-frame; the desk's capture is said once under its own name
        'edge': sum(premiums.values()) - sum(mid[leg.role] for leg in legs),
        # what the WINGS cost over the flat one, per quoted pillar as '<pillar> <expiry>', on the
        # side each leg's own `buy_sell` names - times the scale they were CHARGED at, the same one
        # every leg's `vol_spread` already carries, so the two halves of one quote agree. None
        # where the block quotes no wing two-way
        'wing_spread': {'{:g} {:g}'.format(pillar, expiry): half * charged
                        for (expiry, pillar), half in wings.items()} or None,
        'spread_note': None if two_way or wings else
        '{} carries no Quoted_Bid/Quoted_Ask - every leg is quoted at the mid surface, '
        'unshifted'.format(FX_VOL_PRICES.format(surface)),
        # what the residual this trade leaves on the book costs to hedge, and what the policy did
        # with it. `scale` is None where the feature never ran; the note says why
        'risk': risk,
        # the MODEL these legs were priced under where it is not the book's own default - the pin
        # `spot_model` wrote on the quote's copy, which `/book/quote` merges into the book
        'valuation_configuration': {
            deal_type: entry for deal_type, entry in (pinned_models(document) or {}).items()
            if entry != already.get(deal_type)} or None,
        'deal': compose(reference, legs)}
