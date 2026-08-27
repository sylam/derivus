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


def quote(document, structure_name, params):
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

    legs = materialize(structure, params, document)
    for leg in legs:
        leg.deal['Reference'] = '{}_{}'.format(reference, leg.role)
        leg.deal['Structure_Reference'] = reference
    by_role = {leg.role: leg for leg in legs}
    spot = engine_spot(document, legs[0].deal['Underlying_Currency'], legs[0].deal['Currency'])

    premiums, solved = {}, {}
    for step in structure.recipe:
        if step.role not in by_role:
            raise ValueError('{}: the recipe names leg {!r}, which is not declared'.format(
                structure_name, step.role))
        leg = by_role[step.role]
        if isinstance(step, Price):
            premiums[leg.role] = run_price(document, leg.deal)
        elif isinstance(step, Solve):
            target = step.target.value(premiums) if isinstance(step.target, Premium) \
                else float(step.target)
            value, premiums[leg.role] = run_solve(document, leg, step.field, target, spot)
            solved.setdefault(leg.role, {})[step.field] = value
        else:
            raise ValueError('{}: {!r} is not a recipe step'.format(structure_name, step))

    unpriced = [leg.role for leg in legs if leg.role not in premiums]
    if unpriced:
        raise ValueError('{}: the recipe never prices {}'.format(
            structure_name, ', '.join(unpriced)))

    return {
        'quote_id': quote_id, 'structure': structure_name, 'params': dict(params),
        'legs': [{'reference': leg.deal['Reference'], 'role': leg.role,
                  'deal_type': leg.deal['Object'], 'buy_sell': leg.deal.get('Buy_Sell'),
                  'strike_market': leg.to_market(leg.deal['Strike_Price'])
                  if 'Strike_Price' in leg.deal else None,
                  'barrier_market': leg.to_market(leg.deal['Barrier_Price'])
                  if 'Barrier_Price' in leg.deal else None,
                  'premium': premiums[leg.role], 'solved': solved.get(leg.role)}
                 for leg in legs],
        'net': sum(premiums.values()),
        'deal': compose(reference, legs)}
