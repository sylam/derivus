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

A client asks for a zero-cost collar, not for two `FXOptionDeal` blocks with a root find between
them. This module is where that gap is declared rather than coded per request: a STRUCTURE is a
class, its name is the registry key (the `globals()` dispatch the rest of the house uses), and it
states four things and no logic at all -

  - `vernacular`, the names a salesperson says out loud, so a search for "range forward" finds it
  - `fields`, `schema.F` descriptors for the PARAMETERS a client quotes in - the pair, the tenor,
    the notional, the strikes they name
  - `legs`, each naming a declared `Instrument` type plus the PARTIAL deal block the structure
    fixes. This is the `Market Prices` quote pattern: the Instrument store's declarations already
    ARE the leg's field schema, so a leg never restates deal fields - it pins the handful the
    structure decides (`Option_Type`, `Buy_Sell`) and maps its parameter SLOTS onto the rest
  - `recipe`, an ordered list of `Price` and `Solve` steps

`quote()` is the runner, and it owns every conversion, once. Two of them matter.

MARKET AXIS vs ENGINE AXIS. A desk quotes USDZAR 15.50 - ZAR per USD. `FXOptionDeal` prices an
option on `Underlying_Currency` settled in `Currency`, so its `Strike_Price` is in units of
`Currency` per `Underlying_Currency`. When the notional is the pair's QUOTE currency (ZAR of
USDZAR), the deal's axis is the reciprocal of the quoted one, and BOTH the strike and the option
sense invert: `Strike_Price = 1/K`, and a market Call (the right to buy the base currency) is an
engine Put on the quote currency. A leg therefore pins `Option_Type` on the PAIR and the runner
flips it in step with the strike. A barrier leg crosses on a third axis of its own: its
`Barrier_Price` inverts exactly as a strike does, and its DIRECTION flips with it - an Up barrier
on the pair is a Down barrier on the quote currency - while In/Out, which is about the payoff
rather than the axis, never moves. When the notional is the BASE currency the two axes agree and
nothing is converted. No structure knows any of this.

PARAMETERS vs A DEAL. The runner fills the shared block from the parameters - the two currencies
off the pair, the vol surface named for it, the discounting currency, the notional, and
`Expiry_Date` as the book's `Base_Date` plus the quoted tenor - then the leg's pinned block, then
its slots. `expiry` is parsed as `<n><D|W|M|Y>` ('3M', '1Y') against `Config.offset_lookup`, the
same period vocabulary the job grammar reads; an ISO date is also accepted for a broken date, and
anything else refuses by name rather than silently landing on today.

Every step prices ONE leg against a deep copy of the whole book document with the deal tree
emptied - `/book/solve`'s own discipline, since a deal's base valuation does not depend on its
siblings and a lone deal compiles faster per iterate. `Price` is a plain base valuation, which is
the cheapest honest path: pinning a field and solving for it would pay a root find to learn a
number one run already reports. `Solve` is `derivus.solve_deal_field`, bracketed.

The runner takes the WHOLE document rather than a market patch, and hands each step that document.
That is deliberate: a later recipe step is meant to price the book PLUS the candidate with Greeks
and read the risk impact into the quote, and nothing here has to change for it - only what a step
does with the document it is already given.

A SPREAD IS QUOTED; A MID IS BOOKED. A desk does not sell at the mid, and its book does not mark
at the offer. Both are true at once here because the two live in different places: the book's
`FXVol` surface is bootstrapped from `Quoted_Market_Value` alone and never moves, while the
`FXVolPrices` block beside it may carry each pillar's `Quoted_Bid`/`Quoted_Ask` as DATA. This
module is the only reader of that data. Each leg is priced on its own copy of the book whose
written surface is shifted flat by the ATM half-spread at that leg's expiry, signed by the
CLIENT's side - what the client buys is offered at the ask vol, what they sell is taken at the
bid - so a solved coordinate comes out where a desk would actually deal it. Then the finished
legs are priced ONCE more against the unshifted book, and that is `net_mid`: what the trade marks
at the moment it is booked. A book carrying no two-way shifts every leg by zero and quotes
exactly as it always has, to the bit.

THE RISK PRICES THE SPREAD. A trade's charge is the cost of hedging the RESIDUAL it leaves on the
book, at the market's own two-way - never a bp-per-skew number somebody invented. So the composed
candidate is MIRRORED (`mirror`, the same verb the booking uses, so the risk measured and the trade
booked are one object) and the book's vol risk is read twice, with it and without it, in QUOTE
space: `dV/d(ATM)`, `dV/d(RR)`, `dV/d(BF)` per pillar, off `Quote_Sensitivity` on the risk run's own
copy of the `FXVolPrices` block. Each bucket's move in ABSOLUTE risk is charged that bucket's own
half-spread, and a trade that sheds risk saves the desk that much hedge cost - so `participation` of
it comes off the spread. The `Quote Policy` block declares the mandate and its ABSENCE is the
feature's off switch: a book without one quotes bit for bit as it always did.

THE SPOT IS LIVE, THE SURFACE IS TICKED. `with_live_spots` is the inverse of `engine_spot` and the
one seam a caller writes a terminal's number through: a spot is `bind='value'` data and moves
between two prints, while a delta-quoted vol surface is meant to be read at whatever spot is
standing, so a quote may price on a live spot over the book's own ticked surface. Every quote says
which it used under `spot`, and the runner reads that value off the document it priced.
"""

import copy
import json
import re
import time

from .schema import F, REQUIRED

#: A tenor as the job grammar spells one: a count and a period letter. Anchored and whitespace
#: tolerant, because '3M ' out of a spreadsheet cell is the same tenor.
TENOR = re.compile(r'^\s*(\d+)\s*([DWMY])\s*$', re.IGNORECASE)

#: How wide the runner brackets a strike solve, as a multiple of the market spot. A vanilla's value
#: is monotone in its strike, so any bracket spanning deep in- and out-of-the-money contains the
#: root; these ends are wide enough that a zero-cost leg sits inside them for any premium the other
#: legs can carry, and `brentq` refuses by name when it does not.
STRIKE_BRACKET = (0.25, 4.0)

#: Every vanilla leg is European. Pinned per leg rather than injected by the runner: it is an
#: `FXOptionDeal` field, and the runner furnishes only what the PARAMETERS decide. An
#: `FXBarrierOption` declares no such field, so a barrier leg does not carry it.
VANILLA = {'Option_Style': 'European'}

#: Where a pair's two-way lives: the `Market Prices` block `derivus_bloomberg` files a surface
#: under, which is the leg's own `FX_Volatility` name with the family in front of it. The written
#: price factor is `FXVol.<name>`; the quote block is this.
FX_VOL_PRICES = 'FXVolPrices.{}'
FX_VOL_FACTOR = 'FXVol.{}'

#: Where a desk's quoting MANDATE lives: a section of the JOB, beside `Calculation` and
#: `MergeMarketData`. Not inside `ExplicitMarketData` beside `Market Prices`, which was the first
#: choice and is not available - `Context.load_json` does `cfg.params[section].update(...)` and a
#: section `Config` does not declare raises `KeyError` on load, measured rather than assumed. A
#: policy is not market data anyway; every reader of a job walks `Calc` by NAME, so an unknown key
#: there travels through load, pricing and the book file untouched. This module is its only reader.
QUOTE_POLICY = 'Quote Policy'

#: What the policy means where the block is silent, read with `.get` so a desk states only what it
#: is changing. The ABSENCE OF THE BLOCK is the off switch, not these values: `participation` here
#: is what a declared block defaults to, and a book carrying no block never reaches this dict.
POLICY_DEFAULTS = {'participation': 0.5, 'floor': 'mid', 'scope': 'vol',
                   'bucket_limit': None, 'min_ticket_bp': 0.0}

#: `min_ticket_bp` is bp of notional, and a bp is this.
BASIS_POINT = 1e-4

#: The book-alone risk vector by the book's own content etag, bounded. The book's risk moves only
#: when the market ticks or something books, and both change the etag - so a repeat quote on a
#: standing book pays one greeks run instead of two. Bounded because a desk quotes all day against
#: a book that ticks every 30s, and an unbounded dict of vectors keyed by etag is a slow leak.
RISK_CACHE = {}
RISK_CACHE_LIMIT = 16

#: A year, as the expiry axis of a quote block counts one. Only ever used to place a leg's tenor
#: BETWEEN two quoted pillars of the spread curve, so it is a reading of the same axis rather
#: than a day count anything is priced on.
DAYS_IN_YEAR = 365.0

#: A barrier's DIRECTION is a statement about the PAIR, so it crosses to the engine axis with the
#: strike: a barrier above USDZAR 18.50 is below 1/18.50 dollars per rand. In/Out says what the
#: payoff does on touch and means the same on either axis, so only Up/Down moves.
BARRIER_FLIP = {'Up_And_In': 'Down_And_In', 'Down_And_In': 'Up_And_In',
                'Up_And_Out': 'Down_And_Out', 'Down_And_Out': 'Up_And_Out'}

#: The parameters every FX structure quotes in. Shared as module constants for the reason the
#: schema's field groups are: ten legs across five structures read the same four slots, and a
#: copy per class is a copy that drifts.
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
    is pinned on the PAIR (a Call is the right to buy the base currency at the strike); the runner
    puts it on the engine's axis along with the strike. `slots` maps a DEAL field to a PARAMETER
    name, so `{'Strike_Price': 'floor'}` says this leg is struck at whatever the client called the
    floor. A field named by neither is the shared block's or the Instrument declaration's default -
    a leg restates nothing.
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

    A structure that costs nothing is one whose legs sum to zero, so the number a solve aims at is
    almost never a literal - it is whatever the rest of the structure came to. `Premium('protection')`
    is that leg's own value; the arithmetic (`-Premium('a')`, `Premium('a') + Premium('b')`) makes a
    financing leg's target the negative of everything bought so far, which is what "zero cost"
    means once a sold leg's value carries its own sign.
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
        """This combination against `{role: premium}`. A role not yet priced refuses by name - a
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


def registry():
    """`{name: class}` for every structure declared here - the same scan `emit_structures` makes.

    A structure IS a class in this module carrying `vernacular`, so the registry key is the class
    name and there is no second list to keep in step with it.
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

    A tenor is `<n><D|W|M|Y>` read through `Config.offset_lookup`, so the letters mean here exactly
    what they mean in a job's date grid. An ISO date passes through for a broken date. Anything
    else refuses by name - an unparsed tenor landing on the base date is a zero-day option that
    prices without complaining.
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
    seed and the bracket centre for a strike solve and nothing more; no price is taken from it.
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
    """The reporting currency every `FxRate.<ccy>.Spot` in this document is quoted in."""
    return document.get('Calc', {}).get('MergeMarketData', {}).get(
        'ExplicitMarketData', {}).get('System Parameters', {}).get('Base_Currency')


def with_live_spots(document, crosses):
    """Live market crosses written onto the document's own `FxRate.<ccy>.Spot` blocks, IN PLACE -
    the exact inverse of `engine_spot`, and the only thing a quote ever takes off a terminal.

    `crosses` is `{PAIR: value}` as the MARKET quotes each one - `'USDZAR': 16.31` is ZAR per USD.
    `FxRate.<ccy>.Spot` is one unit of that currency in the document's own `Base_Currency` units,
    so a cross pins the leg the base does not: against a USD base, USDZAR writes `FxRate.ZAR` at
    1/16.31 and leaves `FxRate.USD` at the 1.0 it is by definition. A pair NEITHER of whose legs is
    the base needs a second cross to place it and refuses by name rather than being triangulated -
    inventing a leg is a market view, not a tick.

    A spot is `bind='value'` data, so this moves a NUMBER on a block that already exists and never
    authors one: a book carrying no `FxRate` for a leg could not have priced the quote anyway, and
    a new price factor is a re-authoring. The caller owns the copy - the runner is handed a
    document, never a file. Returns `{currency: spot}` for what it wrote.
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


def atm_two_way(document, surface):
    """The ATM half-spread the book carries for `surface`, as sorted `[(expiry, half), ...]` in the
    surface's own vol units - `(ask - bid) / 2` off the quote block's ATM rows, and empty when the
    block carries no two-way at all.

    The block is `Market Prices` DATA: the bootstrap reads `Quoted_Market_Value` by name and
    nothing else, so the written surface is the mid one whether or not these sides are there. This
    is the only reader.

    A row missing either side is not a two-way and is skipped. A CROSSED one reads as zero-wide
    rather than as a negative spread: a stale bid through a live offer is a broken print, and the
    one thing a desk must not do with it is pay a client for it.

    RR and BF rows carry their own two-way and are deliberately NOT read here. v1 widens the whole
    surface by the ATM spread at the leg's expiry, which is the spread a vanilla is dealt on; a
    wing spread would have to skew the smile rather than shift it, so that data waits for the
    version that does.
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

    Flat rather than extrapolated on purpose. A spread continued as a straight line off the last
    two pillars is a number the market never quoted, and the ends - a broken date inside a week, a
    tenor past the longest pillar - are exactly where that line goes furthest wrong.
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

    `atm_two_way`'s reading widened to the whole block, and the two are not redundant: that one
    answers "what does a VANILLA deal on at this tenor", an ATM curve read between pillars, while
    this one answers "what does hedging THIS bucket cost", which is a lookup per quote and never
    interpolated - a bucket IS a quoted pillar or it is not a bucket.

    The descriptor and the used-quote filter both come from `FXVolSurfaceParameters`, because the
    identity that matters is that these keys are the ones the bootstrap's own leaves are published
    under. A second copy of the naming rule here is a copy that drifts, and it would drift into
    silently pricing no bucket at all. A row missing either side is not a two-way and is skipped; a
    CROSSED one reads zero-wide, for the reason `atm_two_way` states.
    """
    from .bootstrappers import FXVolSurfaceParameters
    prices = document.get('Calc', {}).get('MergeMarketData', {}).get(
        'ExplicitMarketData', {}).get('Market Prices', {})
    block = (prices.get(FX_VOL_PRICES.format(surface)) or {}).get('instrument')
    halves = {}
    for point in FXVolSurfaceParameters.used(block) if block and block.get('Points') else []:
        bid, ask = point.get('Quoted_Bid'), point.get('Quoted_Ask')
        if bid is None or ask is None:
            continue
        halves[FXVolSurfaceParameters.descriptor(point)] = max(
            0.0, 0.5 * (float(ask) - float(bid)))
    return halves


def leg_expiry(document, deal):
    """A leg's tenor in years, on the quote block's own expiry axis - the coordinate the spread
    curve is read at, and nothing else. Not a day count a price comes off."""
    days = (timestamp(deal['Expiry_Date']) - timestamp(
        document['Calc']['Calculation']['Base_Date'])).days
    return max(0.0, days / DAYS_IN_YEAR)


def with_vol_shift(document, factor, shift):
    """The book with `FXVol.<pair>` moved by `shift` vol points, flat across the whole surface -
    the copy ONE SIDE of the spread prices on, since the two sides of a structure need the book at
    two different vols at once.

    A zero shift hands back the document ITSELF, uncopied. That is the compatibility contract: a
    book carrying no two-way prices down the identical path it always did, to the bit, and pays
    nothing - not even a deep copy - for a feature it is not using.

    Moving the WRITTEN surface is what a leg then prices on because `run_job` does not bootstrap:
    the block that built this surface is not read again inside a pricing run, so the vols here are
    the vols the pricer sees. Verified rather than assumed - see the two-sided gate, which is
    false if a run were to rebuild the surface from `Market Prices`.
    """
    if not shift:
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
    for row in rows:
        row[-1] += shift
    return moved


class Materialized(object):
    """One leg turned into a deal: the wire block, its role, and how its axis relates to the quoted
    one - which is what lets the runner report a solved strike back in market terms."""
    __slots__ = ('role', 'deal', 'inverted')

    def __init__(self, role, deal, inverted):
        self.role, self.deal, self.inverted = role, deal, inverted

    def to_market(self, engine_strike):
        """An engine-axis strike as the client reads it."""
        return 1.0 / engine_strike if self.inverted else engine_strike


def materialize(structure, params, document):
    """Every leg of `structure` as a wire-form deal, in declaration order.

    The shared block comes from the parameters - the two currencies off the pair, the surface named
    for it, the settlement currency doing the discounting, the notional as the underlying amount,
    and the expiry as a date. Then the leg's pinned block, then its slots. Strike-like slots and
    `Option_Type` cross to the engine axis together, exactly once, here.
    """
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
        if leg.deal_type not in ('FXOptionDeal', 'FXBarrierOption'):
            raise ValueError('{}: the runner furnishes FXOptionDeal and FXBarrierOption legs, '
                             'not {}'.format(leg.role, leg.deal_type))
        deal = dict(shared, Object=leg.deal_type)
        deal.update(leg.pinned)
        for field, slot in leg.slots.items():
            if slot not in params:
                raise ValueError('{}: leg {} needs the {!r} parameter'.format(
                    structure.__name__, leg.role, slot))
            value = float(params[slot])
            deal[field] = 1.0 / value if inverted and field in (
                'Strike_Price', 'Barrier_Price') else value
        # senses and directions convert AFTER pinned and slots merge, so a structure that one day
        # lets the client choose either still crosses the axis exactly once
        if inverted:
            if 'Option_Type' in deal:
                # a call on the pair is a put on the quote currency - the sense inverts with the axis
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
            # it: the two fields `pv_barrier_option` asks for by name are written out, continuous
            # monitoring in the wire form a Period field is decoded from. Absent, the deal is
            # SKIPPED at load and the leg quietly prices at nothing
            deal.setdefault('Barrier_Monitoring_Frequency', {'.DateOffset': '0M'})
            deal.setdefault('Cash_Rebate', 0.0)
        out.append(Materialized(leg.role, deal, inverted))
    return out


def alone(document, deal):
    """A deep copy of the book carrying only `deal`, plus that deal's path.

    The book's market data, calendars, bootstrappers and calculation block travel; its deal tree
    does not. `/book/solve`'s discipline: a deal's own base-valuation row does not depend on its
    siblings, and a lone deal compiles faster per iterate of a solve.
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
    """The cheapest honest valuation of one leg: an ordinary base valuation, which already reports
    the number. Pinning the field and solving for it would pay a root find to learn it twice."""
    from . import Context
    _, out = Context().load_json((json.dumps(alone(document, deal)[0]), 'quote')).run_job()
    return own_value(out, deal['Reference'])


def run_solve(document, leg, field, target, spot):
    """`derivus.solve_deal_field` over one leg, bracketed, writing the answer back onto the leg.

    A strike is bracketed around the market spot by `STRIKE_BRACKET` and crossed to the engine
    axis - inverting swaps the ends, so they are sorted rather than assumed. A BARRIER is bracketed
    on the side its own type lives on, off the same ends. Any other field is left to the secant
    from its current value, which is exact in two pricings for anything the value is affine in.
    Returns `(solved, premium at the solved value)`.
    """
    from . import solve_deal_field
    iterate, deal_path = alone(document, leg.deal)
    bounds = None
    if field == 'Strike_Price':
        bounds = sorted([spot / end if leg.inverted else spot * end for end in STRIKE_BRACKET])
    elif field == 'Barrier_Price':
        # `spot` and the leg's Barrier_Type are both already on the ENGINE axis, so the direction
        # names the side directly - with a hair of buffer so the barrier never lands exactly on the
        # spot. A knock-in's premium is monotone in its barrier (toward spot = more likely to knock
        # = larger magnitude), so brentq owns the root or refuses by name.
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
    their solved values in place and their sell sides carrying `Buy_Sell` Sell, exactly as they
    were priced. Settled in the legs' own settlement currency, so the container nets what the parts
    report without a cross of its own.

    ONE object, carrying its children inside it, because a quote is filed, hashed and read back
    whole. The deal TREE holds a container's children one level out - see `book_node`, which is
    where the two forms meet.
    """
    return {'Object': 'StructuredDeal', 'Reference': reference,
            'Currency': legs[0].deal['Currency'], 'Net_Cashflows': 'Yes',
            'Children': [{'Instrument': {'.Deal': dict(leg.deal)}} for leg in legs]}


def book_node(deal):
    """A composed deal as the NODE a job document's deal tree holds.

    A container's children hang off the NODE - beside `Instrument`, not inside the deal block -
    which is the shape `Context.load_json` walks and `splice_deal` builds. So whatever books a
    quote lifts them out exactly once, here, rather than each caller rediscovering the hard way
    that a populated container spliced flat loads with no children and prices at ZERO, silently.
    """
    node = {'Instrument': {'.Deal': {k: v for k, v in deal.items() if k != 'Children'}}}
    if 'Children' in deal:
        node['Children'] = copy.deepcopy(deal['Children'])
    return node


def mirror(deal):
    """The desk's side of a quoted deal: every leg's `Buy_Sell` flipped, nothing else touched.

    A quote is CLIENT paper - its legs carry the client's side and its net is what the client
    pays - while a trading book holds the BANK's position, so the one seam where paper becomes
    position flips the sign. Both consumers read this one verb: the approval that books the
    trade, and the risk-impact step that prices the book PLUS the candidate - the risk measured
    and the trade booked are the same object, so a sign cannot disagree between them. Sales
    margin never enters here: a mirror is a pure change of side.
    """
    flipped = copy.deepcopy(deal)
    blocks = [flipped] + [child['Instrument']['.Deal'] for child in flipped.get('Children', [])]
    for block in blocks:
        if 'Buy_Sell' in block:
            block['Buy_Sell'] = 'Sell' if block['Buy_Sell'] == 'Buy' else 'Buy'
    return flipped


def read_policy(document):
    """The desk's quoting mandate off `Calc['Quote Policy']`, or `None` where the book declares
    none - and `None` turns the whole risk-impact feature off, which is the compatibility contract.

    Five fields, each read with `.get` against `POLICY_DEFAULTS` so a block may state one of them:

      - `participation` - how much of a measured hedge-cost SAVING is passed to the client
      - `floor` - 'mid': the scale never goes below zero, so a quote never crosses the mid
      - `scope` - 'vol', which is all v1 measures; anything else refuses rather than silently
        pricing a scope nobody implemented
      - `bucket_limit` - a per-bucket cap on `|risk after|` in the bucket's own vega units, past
        which NO tightening applies however good the saving looks
      - `min_ticket_bp` - flat bp of notional, the ops floor under the edge
    """
    policy = document.get('Calc', {}).get(QUOTE_POLICY)
    if policy is None:
        return None
    read = {name: policy.get(name, default) for name, default in POLICY_DEFAULTS.items()}
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

    Three edits and no others. The calculation becomes a `BaseValuation` asking for `Greeks: 'First'`
    - one backward off the ROOT netting set, so a leaf's `.grad` is the whole portfolio's. The
    candidate's nodes are appended (through `book_node` at the call site, which is where a populated
    container learns its children hang off the node). And `Quote_Sensitivity` goes to Yes on the
    `FXVolPrices` block, which is what makes `Config.bootstrap` leave the surface behind still
    connected to the ATM/RR/BF quotes it was built from.

    The switch is worth exactly ZERO in the forward pass - `leaf + (theta - theta.detach())` - so
    turning it on cannot move a price, and this document is a copy in any case.
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

    Descriptors are SUMMED across every published block, which is the documented collision rule -
    one JSON number can feed two chains (`FXVolPrices` writes the surface an option reads, and
    `GBMAssetPriceTSModelPrices` integrates that surface's ATM column into the curve the FX rate is
    simulated with), and each family's partial is correct while neither is the answer. A book that
    only asks the one block still lands here; the sum is over what the run published.

    An EMPTY deal tree has no value to differentiate - `backward()` on a constant refuses - and its
    risk is a zero vector by inspection, so it never reaches a run.
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

    The book alone is the half of the measurement that does not depend on what is being quoted, and
    it moves only when the market ticks or something books - both of which change the etag - so a
    desk quoting repeatedly against a standing book pays for one greeks run rather than two per
    quote. The etag is over everything the greeks run reads: the deal tree, the WHOLE market
    section (a MarketDataFile path is market data too) and the Calculation block - a rolled
    Base_Date or a changed report currency with an unmoved book is a different risk vector, and a
    key missing either would serve yesterday's.
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
    total is the trade shedding risk - the desk saves that much hedge cost, and the policy decides
    how much of the saving the client sees.

    Only buckets the book quotes a two-way for are priced: a bucket with no quoted spread has no
    market price for its risk, and charging it something would be the invented number this whole
    design exists to avoid.
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
    are in this arithmetic and each is one line:

      - a risk-ADDING trade stays at the full two-way. The market spread is the CEILING in v1 -
        there is no surcharge past it - so a positive residual cost is simply no saving.
      - the mid is the FLOOR. The effective charge never goes below zero, so a quote is never
        automatically pushed through the mid however much risk it sheds.
      - the min ticket is the ops floor UNDER the tightening, not a second ceiling over it: a
        min_ticket above the full spread leaves the scale at 1 rather than lifting the quote.

    A bucket standing past `bucket_limit` after the trade suspends the tightening entirely and is
    NAMED - the trade may net down some other bucket, but a book already over its limit somewhere
    does not get to quote tighter on the strength of it.
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


def run_recipe(document, structure, params, reference, two_way, surface, spot, scale):
    """One whole pass of the recipe at `scale` times the book's half-spread, from fresh legs.

    Everything a pass owns: materializing the legs, signing each one's shift by the CLIENT's side,
    building the shifted book each shift needs, running the steps in order, and marking the
    finished legs once more at mid. `scale` is the ONE thing that differs between the base pass and
    a re-quote - it multiplies the half-spread and rides the same two-sided machinery, rather than
    forking a second shift path that could disagree with it about a sign.

    Returns `(legs, spreads, premiums, solved, mid)`. Legs are fresh because `run_solve` writes the
    solved value back onto the leg it moved, so a second pass over the first pass's legs would seed
    itself from the answer it is meant to find.
    """
    legs = materialize(structure, params, document)
    for leg in legs:
        leg.deal['Reference'] = '{}_{}'.format(reference, leg.role)
        leg.deal['Structure_Reference'] = reference
    by_role = {leg.role: leg for leg in legs}

    spreads, books, by_shift = {}, {}, {}
    for leg in legs:
        # a leg's Buy_Sell is the CLIENT's side: what they buy is offered at the ask vol and what
        # they sell is taken at the bid, so the client's side IS the sign of the shift - and a leg
        # that states no side has no side of the market to be dealt on, which is a refusal rather
        # than a default, because either guess charges the spread the wrong way round
        side = leg.deal.get('Buy_Sell')
        if two_way and side not in ('Buy', 'Sell'):
            raise ValueError('{}: leg {} carries no Buy_Sell, so which side of the two-way it '
                             'deals on is not stated'.format(structure.__name__, leg.role))
        spreads[leg.role] = scale * (1.0 if side == 'Buy' else -1.0) * half_spread(
            two_way, leg_expiry(document, leg.deal)) if two_way else None
        # legs taking the SAME shift share one copy - the shift is the whole difference between
        # them, and every pricing deep-copies again through `alone`, so nothing here is mutated.
        # A structure's legs usually share an expiry, which makes this two books rather than five
        shift = spreads[leg.role] or 0.0
        if shift not in by_shift:
            by_shift[shift] = with_vol_shift(document, FX_VOL_FACTOR.format(surface), shift)
        books[leg.role] = by_shift[shift]

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

    # ONE more pass, at MID, over the legs as they were finally solved: the spread belongs to the
    # quote and the mid belongs to the book, and this is the number the trade marks at once booked
    mid = {leg.role: run_price(document, leg.deal) for leg in legs}
    return legs, spreads, premiums, solved, mid


def risk_impact(document, params, reference, two_way, surface, legs, premiums, mid):
    """The whole risk-impact step over a candidate already quoted at the full two-way.

    Measures the book with the candidate's MIRROR on it and without, prices the difference at the
    market's own two-way, applies the policy, and hands back the `risk` block the outcome carries -
    `scale` included, which is what the re-quote multiplies every half-spread by.

    Four ways out, and each leaves `scale` at None (nothing to re-quote) with the reason NAMED
    rather than reported as a scale of 1 nobody can distinguish from a decision:

      - the book declares no `Quote Policy` - the feature is off and this is today's quote
      - the book quotes no two-way - there is no spread to tighten and no half-spread to price a
        bucket at either
      - the full charge is not positive - a two-way that captured nothing has nothing to give back
      - the book publishes no vol quote leaves, so there are no quote coordinates to measure in.
        Conservative on purpose: no coordinates means no tightening, never a guessed one
    """
    policy = read_policy(document)
    empty = {'coordinates': 'quote-space', 'buckets': [], 'saving': None, 'charge_full': None,
             'charge_effective': None, 'scale': None, 'policy': policy}
    if policy is None:
        return dict(empty, note='the book declares no {} block - the quote is the full two-way '
                                'spread, exactly as it was before'.format(QUOTE_POLICY))
    if not two_way:
        return dict(empty, note='{} carries no two-way - there is no spread to tighten'.format(
            FX_VOL_PRICES.format(surface)))
    charge_full = sum(premiums.values()) - sum(mid[leg.role] for leg in legs)
    if charge_full <= 0.0:
        return dict(empty, charge_full=charge_full,
                    note='the two-way captured {:.6g} - there is nothing to give back'.format(
                        charge_full))

    # the MIRROR is the desk's side, which is what a book would carry - the same verb the approval
    # books through, so the risk measured and the trade booked cannot disagree by a sign
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
    min_ticket = float(policy['min_ticket_bp']) * BASIS_POINT * float(params['notional']) * \
        engine_spot(document, str(params['notional_currency']).upper(),
                    document['Calc']['Calculation']['Currency'])
    scale, saving, effective, note = risk_scale(rows, cost, policy, charge_full, min_ticket)
    return {'coordinates': 'quote-space', 'buckets': rows, 'saving': saving,
            'charge_full': charge_full, 'charge_effective': effective, 'scale': scale,
            'policy': policy, 'note': note}


def quote(document, structure_name, params, spot_source=None):
    """Price a structure against a book, and hand back the quote plus the deal it would book.

    `document` is a wire-form job document - the book - and travels whole, never a patch. `params`
    are the client's numbers, in the market's own terms. The answer is
    `{quote_id, structure, params, legs, net, deal}`: one row per leg carrying its reference, role,
    deal type, side, market-terms strike and barrier, premium and whatever was solved on it - a leg
    carrying no barrier reports None for it; `net` as the sum of the legs; and `deal` the composed
    `StructuredDeal`, wire form, ready for the booking verb.

    `quote_id` hashes the structure, the parameters, the market the book was carrying AND a
    submission clock. The clock is the point: a quote is an ACT, so two identical asks minutes
    apart are two quotes and must not coalesce into one - the same reason `/book/bloomberg` stamps
    its result id.

    THE TWO-SIDED HALF. Where the book's `FXVolPrices` block carries a two-way, each leg prices on
    its own copy of it with the written surface shifted flat by the ATM half-spread at that leg's
    expiry - `+half` where the CLIENT buys, `-half` where they sell, so every leg is dealt on the
    side of the market the desk would actually give. `net` is therefore the two-sided price the
    client is quoted, and a solved coordinate is where the structure genuinely finances. Each leg
    reports the signed shift it took as `vol_spread`, in the surface's own units (0.002 is 0.2 vol
    points), or None where the book quotes no two-way at all - in which case every shift is zero
    and this is, to the bit, the quote the runner has always given. `spread_note` names that
    absence when it happens.

    `net_mid` is the finished legs priced once more against the UNSHIFTED book: what the trade
    marks at the moment it is booked. Read it beside `net`, in the same sign convention - a leg's
    `Buy_Sell` is the CLIENT's side, so a bought leg is a positive premium here and the desk's
    captured edge on a zero-cost structure is `net - net_mid`, positive when a spread was charged.

    WHICH SPOT. `spot` is always there and always names the market this quote was actually struck
    on: `value_market` is the pair as the client quotes it, read off the document the legs priced
    against, beside the caller's `source` ('terminal' or 'book') and its `note`. The runner reads
    the value rather than being told it, so what is reported and what was priced cannot disagree;
    a caller that patched a live spot in says so, and the default is the plain reading - the
    document's own spot, nothing tried, nothing to name.

    THE RISK-IMPACT HALF, and it is OFF unless the book declares a `Calc['Quote Policy']` block.
    Where it does: the base pass above is quoted at the full two-way, its composed candidate is
    MIRRORED into the desk's side, and the book's vol risk is measured with it and without it in
    quote coordinates - `dV/d(ATM)`, `dV/d(RR)`, `dV/d(BF)` per pillar. Each bucket's move in
    ABSOLUTE risk, times that bucket's own half-spread, is what hedging the residual costs at the
    market's own two-way; a negative total is a SAVING, and `participation` of it comes off the
    charge. The re-quote then runs the whole recipe again with every leg's half-spread multiplied
    by `scale = charge_effective / charge_full`, threaded through the same two-sided machinery.

    ONE PASS, NOT A FIXED POINT, and that is a deliberate approximation stated rather than hidden:
    the risk was measured on the FULL-SPREAD candidate, and the re-solve moves the solved
    coordinate, so the tightened structure's residual is not exactly the one that was priced. The
    move is second order - a strike shifts by the spread, and the risk shifts by the strike - and
    iterating to a fixed point would pay a greeks run per iterate to chase it. The quote is honest
    about which candidate it measured: the `risk` block's buckets are the full-spread candidate's.

    A risk-ADDING trade quotes at the full spread. There is no surcharge past the two-way in v1 -
    the market's own spread is the ceiling, by the owner's ruling - so a trade that piles risk on
    is simply not tightened. `scale` is never below zero either: the mid is the floor, and a quote
    is never automatically pushed through it. The `risk` block reports the buckets, the saving, the
    full and effective charges, the scale, the policy as READ, and a note where one is warranted.
    """
    from . import content_hash
    from .config import CustomJsonEncoder
    # authored objects and a file's wire form become one shape here, so every copy below is plain
    # JSON and a solve's own re-serialisation cannot trip over a Timestamp
    document = json.loads(json.dumps(document, cls=CustomJsonEncoder))
    structure = structure_named(structure_name)
    quote_id = content_hash({
        'structure': structure_name, 'params': params,
        'market': document.get('Calc', {}).get('MergeMarketData', {}).get('ExplicitMarketData', {}),
        'at': time.perf_counter()})
    reference = '{}-{}'.format(structure_name, quote_id[:8])

    probe = materialize(structure, params, document)[0]
    spot = engine_spot(document, probe.deal['Underlying_Currency'], probe.deal['Currency'])
    surface = probe.deal['FX_Volatility']
    two_way = atm_two_way(document, surface)

    legs, spreads, premiums, solved, mid = run_recipe(
        document, structure, params, reference, two_way, surface, spot, 1.0)
    risk = risk_impact(document, params, reference, two_way, surface, legs, premiums, mid)
    if risk['scale'] is not None and risk['scale'] < 1.0:
        legs, spreads, premiums, solved, mid = run_recipe(
            document, structure, params, reference, two_way, surface, spot, risk['scale'])

    return {
        'quote_id': quote_id, 'structure': structure_name, 'params': dict(params),
        'legs': [{'reference': leg.deal['Reference'], 'role': leg.role,
                  'deal_type': leg.deal['Object'], 'buy_sell': leg.deal.get('Buy_Sell'),
                  'strike_market': leg.to_market(leg.deal['Strike_Price'])
                  if 'Strike_Price' in leg.deal else None,
                  'barrier_market': leg.to_market(leg.deal['Barrier_Price'])
                  if 'Barrier_Price' in leg.deal else None,
                  'premium': premiums[leg.role], 'solved': solved.get(leg.role),
                  'vol_spread': spreads[leg.role]}
                 for leg in legs],
        'net': sum(premiums.values()),
        'net_mid': sum(mid[leg.role] for leg in legs),
        # the spot the legs were ACTUALLY struck on, in the pair's own terms - read back off the
        # document rather than taken from the caller, so the reported market and the priced one are
        # one number. The caller owns only the account of where it came from
        'spot': dict({'source': 'book', 'note': None}, **(spot_source or {}),
                     value_market=legs[0].to_market(spot)),
        # every number above is CLIENT-frame; the desk's capture is the one derived reading, said
        # once under its own name rather than left as arithmetic for every consumer to re-derive
        'edge': sum(premiums.values()) - sum(mid[leg.role] for leg in legs),
        'spread_note': None if two_way else
        '{} carries no Quoted_Bid/Quoted_Ask - every leg is quoted at the mid surface, '
        'unshifted'.format(FX_VOL_PRICES.format(surface)),
        # what the residual this trade leaves on the book costs to hedge, and what the policy did
        # with it. `scale` is None where the feature never ran; the note says why
        'risk': risk,
        'deal': compose(reference, legs)}
