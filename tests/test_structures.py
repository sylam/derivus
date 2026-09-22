"""Seven FX structures and one runner, gated as REAL quotes: a real book document with a real
bootstrapped USDZAR vol surface, `quote()` against it, and a financial identity that would be false
if any piece of the runner were wrong.

The book is `test_service`'s `job()` and `FACTORS`, with its canned Bloomberg observations
bootstrapped into `FXVol.USD.ZAR` - the `/book/market` pipeline done as fixture authoring, since the
runner is a library verb over a document.

ONE deliberate change: `FxRate.ZAR.Spot` is `1/SPOT` here. The engine's `FxRate` spot is a currency
in BASE-currency units, so `test_service`'s 18.5 says one rand buys 18.5 dollars - fine for a
cashflow, but a strike quoted USDZAR 18.50 against it is 340 standard deviations from the money, and
this file's whole subject is the market axis against the engine's.

Both curves are flat at the same rate, so the FX forward IS the spot - asserted by the straddle's
two wings pricing equal at it, and leaned on by the collar gate.

The book quotes MID. `two_sided_book` is the same book with a desk's two-way authored onto the quote
block and nothing else changed - the surface stays the one the mid built, which that fixture proves
by re-bootstrapping.
"""
import copy
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest

import derivus
from derivus import schema, structures, utils
from test_service import BASE, FACTORS, SPOT, dump, fx_vol_quotes, job

#: The client's numbers, in the market's own terms - USDZAR, a rand notional, a one-year tenor.
PAIR = 'USDZAR'
EXPIRY = '1Y'
NOTIONAL = 1_000_000.0
NOTIONAL_CURRENCY = 'ZAR'

#: The names the registry must carry, and nothing else.
ROSTER = {'Straddle', 'Strangle', 'ZeroCostCollar', 'Seagull', 'ForwardExtra',
          'TargetRedemptionForward', 'Accumulator'}

#: Inner paths for an ACCRUAL structure. A TARF and an accumulator are Monte Carlo priced, so a
#: zero-cost solve is a root find over an estimator, converging on the true root only as the paths
#: grow. MEASURED: the accumulator's two orientations solve strikes 4.8e-4 apart at 1024 paths,
#: 1.3e-4 at 4096, 2.5e-5 at 16384, 3.9e-5 at 65536. 16384 keeps the identity sharp at ~1 s a quote.
ACCRUAL_SIMS = 16384

#: The cross-axis band: eight times the measured 2.5e-5, which no axis error survives (the smallest
#: of them, a barrier level inverted twice, moves the solved strike by percent).
AXIS_TOLERANCE = 2e-4

#: The same band under the fitted spot model, MEASURED on the FX gate's own book: 3.1e-4 apart at
#: 16,384 paths against the lognormal's 2.5e-5 - the two shocks' estimator error, both orientations
#: walking one law rather than reading a surface at two moneynesses.
MODEL_AXIS_TOLERANCE = 5e-4

#: Every parameter in the store that is NOT required, and the value it must publish. A market
#: convention is the only reason a sales parameter carries a default, so this is the one place a new
#: one has to be argued for.
DECLARED_DEFAULTS = {'leverage': 2.0}

#: The only two fields that may be published as SELECTORS - they choose which variation is being
#: dealt rather than filling a leg, which is why they carry no value and no client states one.
SELECTOR_KEYS = {'sell_currency', 'buy_currency'}

#: A calibrated LogVar2FJ factor for the rand, as `/book/model` writes one - the JOINING side of
#: the pair. THIS FILE'S OWN SURFACE, fitted once: the ladder `fx_surface_block` authors off the
#: bootstrapped `FXVol.USD.ZAR` (22 quotes), run through `Config.bootstrap` at 2,048 paths, and the
#: written factor pasted here. `On_Guard` is blank, so every number below is the data's.
MODEL_PARAMS = {
    'Property_Aliases': None, 'Kappa_L': 0.5, 'Sigma_L': 0.5, 'Rho_L': 0.2, 'Kappa_S': 6.0,
    'Cap_A': 4.605170185988092, 'Steps_Per_Year': 252.0, 'C_Min': 0.12,
    'Residual_Law': 'NIG', 'On_Guard': '', 'Stickiness_Band': 0.5,
    'Skew_Gradient': '-0.0222277544361,-2.12183436316',
    'Xi_Curve': utils.Curve([], [[0.0, 0.020733491013238004],
                                 [0.2493150684931507, 0.02438547422614177]]),
    'Rho_S': utils.Curve([], [[0.0, 0.08094527234766719]]),
    'Beta': utils.Curve([], [[0.0, -10.934407669066678]]),
    'Sigma_S': utils.Curve([], [[0.0, 2.256886996797385]]),
    'Alpha': utils.Curve([], [[0.0, 60.24372735960779]])}

#: The strip: monthly fixings to the tenor, and a cap of 1.50 rand of cumulative favourable move on
#: a spot of 18.50 - reachable enough that the redemption is part of the price.
FIXING_FREQUENCY = '1M'
TARGET = 1.5

#: `solve_deal_field`'s own default, in report currency - the tolerance every net-zero assertion is
#: entitled to, and no more.
SOLVE_TOLERANCE = 0.01


def params(**extra):
    return dict({'pair': PAIR, 'expiry': EXPIRY, 'notional': NOTIONAL,
                 'notional_currency': NOTIONAL_CURRENCY}, **extra)


def forward_extra_params(**extra):
    return params(sell_currency='USD', buy_currency='ZAR', **extra)


@pytest.fixture(scope='module')
def book():
    """`test_service`'s job document with a USDZAR spot the market would recognise, the canned FX
    vol quotes installed and bootstrapped, so the file carries the `FXVol.USD.ZAR` surface a pricer
    reads - what `/book/market` leaves behind."""
    factors = dict(FACTORS)
    factors['FxRate.ZAR'] = dict(FACTORS['FxRate.ZAR'], Spot=1.0 / SPOT)
    document = json.loads(dump(job(
        deals=(), factors=factors,
        sections={'Bootstrapper Configuration': {'FXVolSurfaceParameters': {}},
                  'Market Prices': json.loads(dump(fx_vol_quotes()))})))
    context = derivus.Context().load_json((json.dumps(document), 'book'))
    context.bootstrap()
    market = document['Calc']['MergeMarketData']['ExplicitMarketData']
    market['Price Factors'] = json.loads(dump(context.current_cfg.params['Price Factors']))
    assert 'FXVol.USD.ZAR' in market['Price Factors'], 'the surface never reached the book'
    return document


#: The desk's ATM two-way here: 0.4 vol points, so a leg shifts by half of it. Wide enough that the
#: solved coordinate moves by more than any solve tolerance, narrow enough to be a real spread.
ATM_SPREAD = 0.004

#: The RR and BF rows' width as a fraction of that - each half as wide as the ATM row, the shape a
#: terminal prints.
WING_FRACTION = 0.5

#: What those two rows compose to at a pillar - `BF_half + RR_half/2`, and what each wing of the
#: smile widens by over and above the flat ATM half. 0.0015 here, against an ATM half of 0.002.
WING_HALF = 0.5 * WING_FRACTION * ATM_SPREAD + 0.5 * (0.5 * WING_FRACTION * ATM_SPREAD)


def two_way(document, spread=ATM_SPREAD, wings=WING_FRACTION):
    """`document` with a two-way authored around the mid its `FXVolPrices` block carries. The
    written surface is not touched: the mid is what built it.

    `spread` is the ATM row's width; `wings` is the RR and BF rows', as a fraction of it. Three
    values: a fraction is a desk's real wing quote, `None` authors no sides on those rows (the
    ATM-only book the skew is measured against), a NEGATIVE fraction crosses those prints.
    """
    out = copy.deepcopy(document)
    block = out['Calc']['MergeMarketData']['ExplicitMarketData'][
        'Market Prices']['FXVolPrices.USD.ZAR']
    for point in block['instrument']['Points']:
        if point['Quote_Type'] != 'ATM' and wings is None:
            continue
        half = 0.5 * (spread if point['Quote_Type'] == 'ATM' else wings * spread)
        point['Quoted_Bid'] = point['Quoted_Market_Value'] - half
        point['Quoted_Ask'] = point['Quoted_Market_Value'] + half
    return out


@pytest.fixture(scope='module')
def two_sided_book(book):
    """The same book quoted two-sided, plus the proof that the bootstrap never reads the two-way:
    re-bootstrapping the block that now carries bid and ask writes the IDENTICAL `FXVol.USD.ZAR`
    surface the mid alone wrote. Were it false the book would mark at the spread."""
    document = two_way(book)
    context = derivus.Context().load_json((json.dumps(document), 'two-sided'))
    context.bootstrap()
    rebuilt = json.loads(dump(context.current_cfg.params['Price Factors']))
    factors = document['Calc']['MergeMarketData']['ExplicitMarketData']['Price Factors']
    assert rebuilt['FXVol.USD.ZAR'] == factors['FXVol.USD.ZAR'], (
        'the bootstrap read the two-way - the written surface is no longer the mid one')
    return document


def option(reference, market_strike, market_type, buy_sell='Buy'):
    """One vanilla leg authored BY HAND on the engine's axis, with the market-to-engine conversion
    done in the gate rather than read from the runner - which is what makes the straddle identity a
    test of the runner and not a restatement of it.

    A rand-notional USDZAR option is an option on ZAR settled in USD, so the strike is USD per ZAR
    (`1/K`) and the market's Call is the engine's Put on rand.
    """
    return {'Object': 'FXOptionDeal', 'Reference': reference, 'Currency': 'USD',
            'Underlying_Currency': 'ZAR', 'Underlying_Amount': NOTIONAL,
            'Strike_Price': 1.0 / market_strike, 'Buy_Sell': buy_sell,
            'Option_Type': 'Put' if market_type == 'Call' else 'Call',
            'Option_Style': 'European', 'FX_Volatility': 'USD.ZAR', 'Discount_Rate': 'USD',
            'Expiry_Date': {'.Timestamp': (BASE + pd_offset()).strftime('%Y-%m-%d')}}


def pd_offset():
    import pandas as pd
    return pd.DateOffset(years=1)


def values(book, deals):
    """`{Reference: value}` for a base valuation of exactly these deals against the book, through
    `structures.book_node` - the one place that knows a container's children hang off the NODE
    rather than inside the block.
    """
    document = copy.deepcopy(book)
    document['Calc']['Deals']['Deals']['Children'] = [
        structures.book_node(deal) for deal in deals]
    _, out = derivus.Context().load_json((json.dumps(document), 'gate')).run_job()
    frame = out['Results']['mtm']
    return {row['Reference']: float(row['Value']) for _, row in frame.iterrows()}


def leg(outcome, role):
    return next(row for row in outcome['legs'] if row['role'] == role)


def test_the_registry_publishes_exactly_the_declared_structures():
    """The store is the front end's whole source: a menu, its parameters, its legs or its
    VARIATIONS, and its recipe. A leg names a declared `Instrument` type and nothing else - that
    type's store entry IS the leg's field schema, so a leg naming a type no class declares cannot
    be built.

    A structure carries `legs` where it is dealt one way and `variations` where it is dealt more,
    never both and never neither. Each variation serves one side of the pair and takes the
    parameters no other form of it takes, and the two together have to make it TELLABLE APART:
    two variations with the same `buys` and the same own parameters are a structure no ticket
    could ever select between, so the runner would be guessing.

    Every rule below runs over each form's WHOLE parameter set - the shared fields plus that
    variation's own - which is what a ticket fills in and what the runner requires.
    """
    store = schema.mapping['Structure']['types']
    assert set(store) == ROSTER
    assert set(store) == set(structures.registry()), (
        'the emitted store and the runner\'s registry disagree about what exists')

    instruments = schema.mapping['Instrument']['types']
    for name, entry in store.items():
        # the invariant lives on the CLASS: the emitter publishes one of the two whatever is
        # declared, so a class carrying BOTH would ship a dead `legs` nothing prices, and one
        # carrying an empty `variations` would fail as a KeyError rather than by name
        cls = structures.structure_named(name)
        assert ('legs' in cls.__dict__) != bool(cls.__dict__.get('variations')), (
            '{} is dealt ONE way or several, and says which - it declares {}'.format(
                name, sorted(set(cls.__dict__) & {'legs', 'variations'}) or 'neither'))
        form = {'legs', 'variations'}.intersection(entry)
        assert form in ({'legs'}, {'variations'}), (
            '{} publishes {} - a structure is dealt ONE way or several, and says which'.format(
                name, sorted(form) or 'neither legs nor variations'))
        assert set(entry) == {'vernacular', 'fields', 'recipe'} | form
        assert entry['vernacular'] and entry['recipe']
        assert {'pair', 'expiry', 'notional', 'notional_currency'} <= set(entry['fields'])

        # a single-form structure reads as one nameless variation taking no parameters of its own,
        # so there is one set of rules rather than two
        dealt = entry.get('variations') or {
            None: {'buys': None, 'fields': {}, 'legs': entry['legs']}}
        # a DIRECTION has to be stated where some variation's own parameters do not tell it apart
        # from another's - the rule the store publishes, computed here from the same declarations
        needed = any(set(one['fields']) <= set(other['fields'])
                     for one in dealt.values() for other in dealt.values() if one is not other)
        # and a selector is the declared field OBJECT, never a parameter sharing its name
        selects = {f.key for f in cls.__dict__['fields'] if f in structures.SELECTORS}
        told_apart = []
        for word, variation in dealt.items():
            assert variation['buys'] in (None, 'base', 'quote'), (name, word, variation['buys'])
            assert not set(variation['fields']) & set(entry['fields']), (
                '{}.{} restates a parameter every variation shares'.format(name, word))
            assert (variation['buys'], sorted(variation['fields'])) not in told_apart, (
                '{}.{} deals the same side of the pair on the same parameters as another '
                'variation - no ticket could select between them'.format(name, word))
            told_apart.append((variation['buys'], sorted(variation['fields'])))

            fields = dict(entry['fields'], **variation['fields'])
            assert fields and variation['legs']
            for key, descriptor in fields.items():
                if descriptor.get('selector'):
                    # a selector CHOOSES a form rather than filling a leg, so it publishes no
                    # value, and a structure with one form has nothing to choose between
                    assert key in SELECTOR_KEYS and 'value' not in descriptor, (name, key)
                    assert key in selects, '{}.{} is published as a selector and is not one'.format(
                        name, key)
                    assert word is not None, '{} selects between one form'.format(name)
                    assert descriptor['selector'] == ('required' if needed else 'optional'), (
                        '{}.{} publishes {!r} where its variations {} told apart by their own '
                        'parameters'.format(name, key, descriptor['selector'],
                                            'are not' if needed else 'are'))
                elif descriptor.get('required') is not True:
                    # a parameter is REQUIRED unless the market has a convention for it, and a
                    # declared default must be PUBLISHED as the value or a front end makes the
                    # client state a number the desk already assumed
                    assert key in DECLARED_DEFAULTS, (
                        '{}.{} has a default a client cannot mean'.format(name, key))
                    assert descriptor['value'] == DECLARED_DEFAULTS[key], (name, key, descriptor)
            for role, declared in variation['legs'].items():
                assert declared['deal_type'] in instruments, '{}.{} is a {}, which no class '\
                    'declares'.format(name, role, declared['deal_type'])
                assert set(declared) == {'deal_type', 'pinned', 'slots'}
                unknown = set(declared['slots'].values()) - set(fields)
                assert not unknown, '{}.{}.{} maps slots to parameters {} it does not take'.format(
                    name, word, role, sorted(unknown))


def test_an_unknown_structure_refuses_with_the_roster():
    """A typo is a sales enquiry that cannot be answered, not an empty answer."""
    with pytest.raises(ValueError) as refusal:
        structures.quote({}, 'RangeAccrual', params())
    assert 'ZeroCostCollar' in str(refusal.value)


def test_a_straddle_is_exactly_its_two_legs(book):
    """The runner's arithmetic against the same two options priced by hand, to 1e-9 relative: the
    shared block, the tenor parse, the strike inversion and the option-sense flip all landed on the
    same contract, with no solve in the way.

    The two wings priced at spot come out EQUAL, which is the forward statement this file needs: a
    call and a put agree in value only at the forward, so here the forward is the spot.

    And the composed `StructuredDeal` prices to the quoted net - the deal riding the quote is the
    thing that was quoted.
    """
    outcome = structures.quote(book, 'Straddle', params(strike=SPOT))
    by_hand = values(book, [option('CALL', SPOT, 'Call'), option('PUT', SPOT, 'Put')])

    assert leg(outcome, 'call')['premium'] == pytest.approx(by_hand['CALL'], rel=1e-9)
    assert leg(outcome, 'put')['premium'] == pytest.approx(by_hand['PUT'], rel=1e-9)
    assert outcome['net'] == pytest.approx(by_hand['CALL'] + by_hand['PUT'], rel=1e-9)
    assert outcome['net'] > 0.0, 'a bought straddle costs money'
    assert by_hand['CALL'] == pytest.approx(by_hand['PUT'], rel=1e-9), (
        'the wings disagree at spot - the forward is not the spot and this world is not flat')

    priced = values(book, [outcome['deal']])
    assert priced[outcome['deal']['Reference']] == pytest.approx(outcome['net'], rel=1e-9)
    assert [priced[row['reference']] for row in outcome['legs']] == pytest.approx(
        [row['premium'] for row in outcome['legs']], rel=1e-9), (
        'the container reports legs the quote does not')
    assert [row['solved'] for row in outcome['legs']] == [None, None], 'a straddle solves nothing'


def test_a_zero_cost_collar_costs_nothing(book):
    """The client names a floor; the cap is whatever strike makes the sold call pay for the bought
    put.

    Three claims. The net is zero to the solve's own tolerance. The solved cap is above the forward
    and the floor below it - a solver on the wrong branch would still net to zero and fail here. And
    re-quoting the two strikes as a bought `Strangle` prices the legs to equal premiums, the same
    statement from the other side.
    """
    floor = SPOT * 0.95
    outcome = structures.quote(book, 'ZeroCostCollar', params(floor=floor))
    protection, financing = leg(outcome, 'protection'), leg(outcome, 'financing')
    cap = financing['strike_market']

    assert abs(outcome['net']) <= SOLVE_TOLERANCE, outcome['net']
    assert protection['premium'] > 0 > financing['premium'], 'the sides are the wrong way round'
    assert floor < SPOT < cap, 'floor {} cap {} straddle the forward {}'.format(floor, cap, SPOT)
    assert protection['solved'] is None
    assert financing['solved'] == {'Strike_Price': pytest.approx(1.0 / cap, rel=1e-12)}

    strangle = structures.quote(book, 'Strangle', params(floor=floor, cap=cap))
    assert leg(strangle, 'floor')['premium'] == pytest.approx(
        leg(strangle, 'cap')['premium'], abs=SOLVE_TOLERANCE)
    assert leg(strangle, 'floor')['premium'] == pytest.approx(
        protection['premium'], rel=1e-9), 'the same floor repriced differently'


#: The desk's charge, as a client agrees one: an amount and the currency it is stated in. Against
#: the notional below it is a quarter of a percent - a real sales margin, and hundreds of times any
#: solve tolerance. The book prices in DOLLARS and this is rand, so nothing reads it unconverted.
MARGIN = {'amount': 50_000.0, 'currency': 'ZAR'}
MARGIN_NOTIONAL = 20_000_000.0

#: The same quarter of a percent against the file's OWN notional, for the gates quoted at it - a
#: margin a 1m collar's solved cap can actually fund.
SMALL_MARGIN = {'amount': 0.0025 * NOTIONAL, 'currency': 'ZAR'}


def test_a_collar_at_a_margin_is_minus_it_on_paper_and_plus_it_on_the_book(book):
    """The sign of a sales margin, taken from the client-paper convention and read from BOTH ends.

    A quote is client paper: the cap the recipe solves has to fund the bought put AND the charge,
    so what the client holds is worth MINUS the margin - `net`, converted back at the quote's own
    spot, is -50,000 rand. The trading book holds the bank's position, which is the MIRROR of that
    paper, and pricing the mirror against this book marks it at PLUS the margin's dollar value.
    One number, and the sign is the mirror's doing.

    Nothing else moves. The bought leg is priced before the solve and comes back bit-identical, and
    the cap comes IN: a call that has to raise more is struck nearer the money.
    """
    asked = params(notional=MARGIN_NOTIONAL, floor=SPOT * 0.95)
    plain = structures.quote(book, 'ZeroCostCollar', dict(asked))
    outcome = structures.quote(book, 'ZeroCostCollar', dict(asked), margin=MARGIN)
    charge = outcome['margin']

    assert charge == {'amount': 50_000.0, 'currency': 'ZAR', 'pricing_currency': 'USD',
                      'value': pytest.approx(50_000.0 / SPOT, rel=1e-12)}
    assert outcome['net'] * outcome['spot']['value_market'] == pytest.approx(
        -50_000.0, abs=SOLVE_TOLERANCE * SPOT), 'the client is not holding paper worth -50,000 rand'

    priced = values(book, [structures.mirror(outcome['deal'])])
    assert priced[outcome['deal']['Reference']] == pytest.approx(
        charge['value'], abs=SOLVE_TOLERANCE), 'the bank is not marked at the margin it charged'

    assert leg(outcome, 'protection')['premium'] == leg(plain, 'protection')['premium']
    assert leg(outcome, 'financing')['strike_market'] < leg(plain, 'financing')['strike_market'], (
        'a cap that funds the margin too is not struck nearer the money')
    assert (outcome['deal']['Sales_Margin'],
            outcome['deal']['Sales_Margin_Currency']) == (50_000.0, 'ZAR')


def test_a_margin_and_a_two_way_compose_on_one_coordinate(book, two_sided_book):
    """Two charges, one coordinate, and neither displaces the other.

    A sales margin and the two-way's own charge are the same shape - money the desk takes, levied
    on the coordinate the recipe solves - so the cap funds the bought put, the margin AND the
    spread. What the client is QUOTED is still minus the margin alone, the spread being inside the
    terms rather than on the ticket, while what the trade MARKS at is minus both. The mirror the
    bank books holds the two together, read off the engine rather than off the runner.

    And the cap comes in further than either charge moves it alone, which is what says they add.
    """
    asked = params(notional=MARGIN_NOTIONAL, floor=SPOT * 0.95)
    priced = structures.quote(two_sided_book, COLLAR, dict(asked), margin=MARGIN)
    spread_only = structures.quote(two_sided_book, COLLAR, dict(asked))
    margin_only = structures.quote(book, COLLAR, dict(asked), margin=MARGIN)
    charge = priced['margin']['value']

    assert priced['edge'] > 0.0
    assert priced['edge'] == pytest.approx(
        sum(row['spread_charge'] for row in priced['legs']), rel=1e-12)
    assert priced['net'] == pytest.approx(-charge, abs=SOLVE_TOLERANCE), (
        'the client was quoted something other than the margin they agreed')
    assert priced['net_mid'] == pytest.approx(-(charge + priced['edge']), abs=SOLVE_TOLERANCE)
    assert cap_of(priced) < min(cap_of(spread_only), cap_of(margin_only)), (
        'the two charges did not add on one coordinate')

    marked = values(book, [structures.mirror(priced['deal'])])
    assert marked[priced['deal']['Reference']] == pytest.approx(
        charge + priced['edge'], abs=SOLVE_TOLERANCE), (
        'the bank is not marked at the margin and the spread together')


def test_a_quote_with_no_margin_is_the_quote_it_always_was(book):
    """The compatibility contract, stated twice. Absent, the feature is not there at all: no
    `margin` block in the answer and no `Sales_Margin` on the deal it books. And a margin of ZERO
    - where every line of the arithmetic DID run - is the same quote to the bit."""
    plain = structures.quote(book, 'ZeroCostCollar', params(floor=SPOT * 0.95))
    zero = structures.quote(book, 'ZeroCostCollar', params(floor=SPOT * 0.95),
                            margin={'amount': 0.0, 'currency': 'ZAR'})

    assert 'margin' not in plain
    assert not {'Sales_Margin', 'Sales_Margin_Currency'}.intersection(plain['deal'])
    assert (zero['net'], zero['net_mid'], zero['edge']) == (
        plain['net'], plain['net_mid'], plain['edge'])
    assert [row['strike_market'] for row in zero['legs']] == [
        row['strike_market'] for row in plain['legs']]
    assert zero['deal']['Sales_Margin'] == 0.0, 'a zero margin is still what was agreed'


def test_a_margin_refuses_where_it_cannot_be_valued(book):
    """Two refusals, each by name. A currency the book carries no `FxRate` for cannot be crossed
    into the price, and a quote is not struck at a rate somebody guessed. And a bare number is not
    a margin: an amount with no currency is exactly the ambiguity this form exists to remove.

    What is NOT refused any more is a recipe that solves nothing: it charges the PREMIUM instead,
    the way a solving one charges its coordinate, and the gate below holds that. A recipe solving
    MORE than one coordinate still refuses - the charge would be levied once per solve - and the
    registry declares no such structure, so the refusal stands on its own statement.
    """
    with pytest.raises(ValueError) as unpriced:
        structures.quote(book, 'ZeroCostCollar', params(floor=SPOT * 0.95),
                         margin={'amount': 50_000.0, 'currency': 'JPY'})
    assert 'FxRate.JPY' in str(unpriced.value)

    with pytest.raises(ValueError) as shapeless:
        structures.quote(book, 'ZeroCostCollar', params(floor=SPOT * 0.95), margin=50_000.0)
    assert "'currency'" in str(shapeless.value)


@pytest.mark.parametrize('currency', ('ZAR', 'USD'))
def test_a_structure_that_solves_nothing_charges_its_premium(book, two_sided_book, currency):
    """ONE convention, not two. A recipe with a coordinate charges the coordinate; a recipe with
    none charges the PREMIUM, and the outcome says which under `charged_on`.

    A straddle is two bought wings at a strike the client named - there is nothing to move - so the
    margin and the two-way are levied on what the client PAYS. What they pay moves against them by
    both, the booked legs stay at MID and the mirror marks there, `edge` is still the two-way charge
    and still the sum of the legs' own, and the desk's whole take is the cash difference
    `net - net_mid`.

    There is no branch on the package's sign: `net` is `net_mid` plus the margin plus the edge
    whichever way round the legs are booked, so a sold form - which the registry does not declare -
    would move the client's receipt by the same amount in the same direction. Both sides of the
    PAIR are quoted here, which is the axis the runner can get wrong.
    """
    ask = params(strike=SPOT, notional_currency=currency,
                 notional=NOTIONAL if currency == 'ZAR' else NOTIONAL / SPOT)
    at_mid = structures.quote(book, 'Straddle', dict(ask))
    charged = structures.quote(two_sided_book, 'Straddle', dict(ask), margin=MARGIN)
    value = charged['margin']['value']

    assert at_mid['charged_on'] == charged['charged_on'] == 'premium'
    assert charged['edge'] > 0.0
    assert charged['edge'] == pytest.approx(
        sum(row['spread_charge'] for row in charged['legs']), rel=1e-12)
    assert charged['net_mid'] == pytest.approx(at_mid['net'], rel=1e-9), (
        'the booked legs are not the mid ones')
    assert charged['net'] == pytest.approx(
        charged['net_mid'] + value + charged['edge'], rel=1e-12)
    assert charged['net'] - charged['net_mid'] == pytest.approx(
        value + charged['edge'], rel=1e-12), 'the desk takes something other than what it charged'
    assert charged['net'] > charged['net_mid'] > 0.0, 'the premium moved toward the client'

    # the legs book and mark at MID - the desk's take on a premium-charged structure is cash the
    # book never sees, which is what `charged_on` is there to say
    marked = values(book, [structures.mirror(charged['deal'])])
    assert marked[charged['deal']['Reference']] == pytest.approx(
        -charged['net_mid'], rel=1e-9)
    assert charged['deal']['Sales_Margin'] == 50_000.0


def test_a_seagull_nets_to_zero(book):
    """Three legs, two strikes named and one solved. The solve targets the sum of TWO already-priced
    legs, so a runner reading only the last priced leg produces a plausible cap and fails here."""
    outcome = structures.quote(book, 'Seagull', params(floor=SPOT * 0.98, lower_floor=SPOT * 0.90))

    assert len(outcome['legs']) == 3
    assert abs(outcome['net']) <= SOLVE_TOLERANCE, outcome['net']
    assert leg(outcome, 'protection')['premium'] > 0
    assert leg(outcome, 'participation')['premium'] < 0
    assert leg(outcome, 'financing')['strike_market'] > SPOT, 'the cap is not above the forward'
    assert [row['buy_sell'] for row in outcome['legs']] == ['Buy', 'Sell', 'Sell']
    assert outcome['deal']['Object'] == 'StructuredDeal'
    assert [child['Instrument']['.Deal']['Buy_Sell']
            for child in outcome['deal']['Children']] == ['Buy', 'Sell', 'Sell']


def test_a_forward_extra_costs_nothing_and_solves_its_barrier(book):
    """The registry's first solved coordinate that is not a strike: protected at the named rate,
    keeping the favourable move, paying nothing - a trade through the barrier knocks the sold call
    in and the whole thing reverts to a forward at that same protected rate.

    Four claims. The net is zero to the solve's tolerance. Both legs are struck at the ONE rate the
    client named, which is what makes the knocked state a forward rather than a spread. The solved
    barrier sits ABOVE the market spot, which is not a tautology: on a rand notional that leg is a
    `Down_And_In` bracketed BELOW the engine spot, so a flipped direction or a doubly-inverted level
    reports a barrier under spot or refuses inside the bracket. And the composed `StructuredDeal`
    reprices to the quoted net leg for leg.
    """
    protected = SPOT * 0.97
    outcome = structures.quote(book, 'ForwardExtra', forward_extra_params(floor=protected))
    protection, reversion = leg(outcome, 'protection'), leg(outcome, 'reversion')

    assert abs(outcome['net']) <= SOLVE_TOLERANCE, outcome['net']
    assert protection['premium'] > 0 > reversion['premium'], 'the sides are the wrong way round'
    assert protection['strike_market'] == pytest.approx(protected, rel=1e-12)
    assert reversion['strike_market'] == pytest.approx(protected, rel=1e-12), (
        'the knock-in reverts to a forward at a rate the client never named')
    assert protection['barrier_market'] is None, 'a vanilla leg reports no barrier'
    assert reversion['barrier_market'] > SPOT, 'the barrier is not on the participation side'
    assert reversion['solved'] == {'Barrier_Price': pytest.approx(
        1.0 / reversion['barrier_market'], rel=1e-12)}
    assert protection['solved'] is None

    booked = outcome['deal']['Children'][1]['Instrument']['.Deal']
    assert booked['Object'] == 'FXBarrierOption' and booked['Buy_Sell'] == 'Sell'
    assert booked['Barrier_Type'] == 'Down_And_In', 'up on the pair is down on the rand'
    assert booked['Option_Type'] == 'Put', 'the sense crosses with the axis on a barrier leg too'
    assert 'Option_Style' not in booked, 'an FXBarrierOption declares no Option_Style to pin'

    priced = values(book, [outcome['deal']])
    assert [priced[row['reference']] for row in outcome['legs']] == pytest.approx(
        [row['premium'] for row in outcome['legs']], rel=1e-9), (
        'the container reports legs the quote does not')
    assert priced[outcome['deal']['Reference']] == pytest.approx(outcome['net'], abs=1e-6)


def test_a_book_with_no_two_way_quotes_exactly_as_it_always_did(book):
    """The compatibility contract as an identity. A book carrying no `Quoted_Bid`/`Quoted_Ask` has
    no two-way to charge, so not one greeks run is made and the quote is the mid one.

    The sharp half is the comparison: a ZERO-WIDE two-way exercises the entire layer - block found,
    every quoted pillar's half read, a vega run per leg, the charge summed, the coordinate left
    where it was - and must land on the IDENTICAL floats. So the presence of the data cannot move a
    price; only a real spread can.

    A QUOTE NEVER REPORTS A SPREAD IT DID NOT CHARGE, and the two books say different things. With
    no two-way at all the three spread keys are null and `spread_note` names the absence. Zero-wide,
    the spread was read and IS zero, and the rows say so pillar by pillar - which is a statement
    about the market rather than about the book.
    """
    ask = forward_extra_params(floor=SPOT * 0.97)
    mid = structures.quote(book, 'ForwardExtra', ask)
    zero_wide = structures.quote(two_way(book, spread=0.0), 'ForwardExtra', ask)

    assert [row['spread_charge'] for row in mid['legs']] == [None, None]
    assert [row['spread_source'] for row in mid['legs']] == [None, None]
    assert [row['spread'] for row in mid['legs']] == [None, None]
    assert 'Quoted_Bid' in mid['spread_note'] and 'FXVolPrices.USD.ZAR' in mid['spread_note']
    assert mid['edge'] == 0.0 and abs(mid['net']) <= SOLVE_TOLERANCE

    assert [row['spread_charge'] for row in zero_wide['legs']] == [0.0, 0.0]
    assert [row['spread_source'] for row in zero_wide['legs']] == ['surface'] * 2
    assert all(cell['half'] == 0.0 and cell['cost'] == 0.0
               for row in zero_wide['legs'] for cell in row['spread'])
    assert zero_wide['spread_note'] is None, 'a two-way was found; there is no fallback to name'
    for row, same in zip(mid['legs'], zero_wide['legs']):
        assert (row['premium'], row['strike_market'], row['barrier_market'], row['solved']) == (
            same['premium'], same['strike_market'], same['barrier_market'], same['solved']), row
    assert mid['net'] == zero_wide['net'] and mid['net_mid'] == zero_wide['net_mid']
    assert mid['edge'] == zero_wide['edge'] == 0.0


def test_a_two_sided_quote_charges_the_spread_and_leaves_the_book_at_mid(book, two_sided_book):
    """The ruling, priced: the spread belongs to the quote and the mid belongs to the book. Three
    DIRECTIONS per structure - a magnitude would restate the charge rather than test it.

    THE LEGS ARE PRICED AT MID, and that is asserted against the engine rather than assumed: the
    composed two-sided deal, valued against the UNSHIFTED book, reports each leg at exactly the
    premium the quote did. A leg priced on its own shifted copy of the surface cannot do that.

    The solved coordinate lands CLIENT-WORSE, because the charge is levied on it: the forward
    extra's financing barrier sits closer to the spot than the mid-solved one and the collar's cap
    comes in. Both strictly between the mid answer and the spot, so a charge with the wrong sign or
    on the wrong side of the target fails here.

    And the desk keeps exactly what it charged: `net - net_mid` is the edge, it IS the sum of the
    legs' own charges, and the structure still costs the client nothing - `net` is zero, the price
    of a zero-cost structure, while what it MARKS at is minus the edge. `charged_on` names the
    coordinate that carried it, and the two structures here solve DIFFERENT fields.
    """
    ask = forward_extra_params(floor=SPOT * 0.97)
    mid, two_sided = (structures.quote(document, 'ForwardExtra', ask)
                      for document in (book, two_sided_book))
    barrier = (leg(mid, 'reversion')['barrier_market'],
               leg(two_sided, 'reversion')['barrier_market'])
    marked = values(book, [two_sided['deal']])

    assert abs(two_sided['net']) <= SOLVE_TOLERANCE, two_sided['net']
    assert [marked[row['reference']] for row in two_sided['legs']] == pytest.approx(
        [row['premium'] for row in two_sided['legs']], rel=1e-9), (
        'a leg was priced on something other than the mid book')
    assert marked[two_sided['deal']['Reference']] == pytest.approx(
        two_sided['net_mid'], abs=1e-6)
    assert SPOT < barrier[1] < barrier[0], (
        'the two-sided barrier {} is not inside the mid one {}'.format(*reversed(barrier)))
    assert two_sided['edge'] > 0.0, two_sided['net_mid']
    assert two_sided['edge'] == pytest.approx(two_sided['net'] - two_sided['net_mid'], rel=1e-9)
    assert two_sided['edge'] == pytest.approx(
        sum(row['spread_charge'] for row in two_sided['legs']), rel=1e-12)

    floor = params(floor=SPOT * 0.95)
    mid_collar, two_sided_collar = (structures.quote(document, 'ZeroCostCollar', floor)
                                    for document in (book, two_sided_book))
    cap = (leg(mid_collar, 'financing')['strike_market'],
           leg(two_sided_collar, 'financing')['strike_market'])

    assert abs(two_sided_collar['net']) <= SOLVE_TOLERANCE, two_sided_collar['net']
    assert SPOT < cap[1] < cap[0], 'the two-sided cap {} is not inside the mid one {}'.format(
        *reversed(cap))
    assert two_sided_collar['net_mid'] == pytest.approx(
        -two_sided_collar['edge'], rel=1e-9) and two_sided_collar['edge'] > 0.0

    # a SOLVING recipe names the coordinate that carried the charge, and the two here are different
    # fields, so a `charged_on` fixed at one string or at 'premium' fails on both
    assert (two_sided['charged_on'], two_sided_collar['charged_on']) == (
        'Barrier_Price', 'Strike_Price'), 'the charge is reported on a coordinate nothing solved'


def with_pillar_moved(document, pillar, delta):
    """The book with ONE quoted pillar's MID moved by `delta`, re-bootstrapped - so the written
    surface is the one those quotes build and the move is the market's, not a shift."""
    from derivus.bootstrappers import FXVolSurfaceParameters
    out = copy.deepcopy(document)
    points = out['Calc']['MergeMarketData']['ExplicitMarketData'][
        'Market Prices']['FXVolPrices.USD.ZAR']['instrument']['Points']
    hit = [point for point in points
           if FXVolSurfaceParameters.descriptor(point) == pillar]
    assert len(hit) == 1, (pillar, len(hit))
    hit[0]['Quoted_Market_Value'] = float(hit[0]['Quoted_Market_Value']) + delta
    context = derivus.Context().load_json((json.dumps(out), 'moved'))
    context.bootstrap()
    market = out['Calc']['MergeMarketData']['ExplicitMarketData']
    market['Price Factors'] = json.loads(dump(context.current_cfg.params['Price Factors']))
    return out


def test_each_pillars_charge_is_what_repricing_that_pillar_costs(book, two_sided_book):
    """THE CHARGE AGAINST THE ENGINE, with `vol_risk` out of the loop entirely.

    One leg - the collar's bought put, at the terms the mid pass solved. For each quoted pillar the
    book's own MID for that pillar is moved by its half, both ways, the surface RE-BOOTSTRAPPED from
    the moved quotes, and the leg repriced alone. `(up - down)/2` is `vega x half` with the second
    order cancelled, and it never touches the reader the charge is built on.

    Two claims per pillar. The reported cost is that reading within **2%** - measured at 0.00% to
    1.12% across every leg of every form, the gap being second order in the spread's own width. And
    the SIGN: the direction that hurts the client is the one `-sign(reported vega)` names, which is
    the whole ruling the absolute value implements. A charge signed by the label instead agrees with
    the reprice on this leg's ATM and butterfly rows and disagrees on its risk reversal.

    This is the tight check `STRIP_CHARGE`'s 5% band is not: at 5% a 4% error in the charge passes.
    """
    solved = structures.quote(book, COLLAR, params(floor=SPOT * 0.95))
    charged = structures.quote(two_sided_book, COLLAR, params(floor=SPOT * 0.95))
    deal = solved['deal']['Children'][0]['Instrument']['.Deal']
    row = leg(charged, 'protection')
    base = structures.run_price(book, deal)

    assert row['role'] == 'protection' and row['buy_sell'] == 'Buy'
    live = [cell for cell in row['spread'] if cell['half'] > 0.0 and cell['vega']]
    assert {cell['pillar'] for cell in live} == {'ATM 1', 'BF 0.25 1', 'RR 0.25 1'}, live

    for cell in live:
        up = structures.run_price(
            with_pillar_moved(book, cell['pillar'], cell['half']), deal)
        down = structures.run_price(
            with_pillar_moved(book, cell['pillar'], -cell['half']), deal)
        repriced = abs(0.5 * (up - down))
        assert cell['cost'] == pytest.approx(repriced, rel=0.02), (cell, repriced)
        assert (up > down) == (cell['vega'] > 0.0), (
            '{} is charged on the side the reprice says pays the client'.format(cell['pillar']))
        assert base - min(up, down) > 0.0, 'neither side of this pillar hurts the client'


def no_quote_leaves(document):
    """The book with its FX vol BOOTSTRAPPER dropped - the surface it already wrote stays, so every
    leg prices, but nothing publishes a quote leaf for the two-way to be charged against."""
    out = copy.deepcopy(document)
    out['Calc']['MergeMarketData']['ExplicitMarketData']['Bootstrapper Configuration'] = {}
    return out


def test_a_leg_the_two_way_cannot_reach_is_charged_nothing_and_says_so(book, two_sided_book):
    """A quote NEVER reports a spread it did not charge, and an unknown is null rather than zero.

    The book here carries a real two-way on quotes that build no leaves any more: the surface is
    written in `Price Factors` so every leg prices exactly as it did, while the bootstrapper that
    published `dV/dq` is gone. There is then no vega to charge the spread against - neither off the
    surface nor off a lognormal reading of it - so the leg's `spread_charge` is NULL, its `spread`
    and `spread_source` with it, and the leg's own NOTE says the two-way could not reach it.

    A zero would be a lie in the other direction: it reads as a spread the desk measured and found
    to be nothing, and a consumer auditing the quote cannot tell that from a leg nobody priced.

    Nothing else moves: an unchargeable two-way quotes the mid quote, to the float.
    """
    ask = params(floor=SPOT * 0.95)
    blind = structures.quote(with_policy(no_quote_leaves(two_sided_book)), COLLAR, ask)
    mid = structures.quote(book, COLLAR, ask)

    assert [row['spread_charge'] for row in blind['legs']] == [None, None]
    assert [row['spread_source'] for row in blind['legs']] == [None, None]
    assert [row['spread'] for row in blind['legs']] == [None, None]
    assert all('could not reach' in row['note'] for row in blind['legs']), blind['legs']
    assert blind['spread_note'] is None, 'the book quotes a two-way; there is no absence to name'
    assert blind['edge'] == 0.0
    # the book DECLARES a policy here, so the risk step really runs and really has to answer: a
    # charge no leg could be read for is null, not a zero somebody could mistake for a measurement
    assert blind['risk']['policy'] is not None, 'the risk step never ran, so nothing is proved'
    assert blind['risk']['charge_full'] is None and blind['risk']['scale'] is None
    assert 'quote sensitivity' in blind['risk']['note'], blind['risk']['note']

    for row, same in zip(blind['legs'], mid['legs']):
        assert (row['premium'], row['strike_market'], row['solved']) == (
            same['premium'], same['strike_market'], same['solved']), row
    assert (blind['net'], blind['net_mid']) == (mid['net'], mid['net_mid'])


COLLAR = 'ZeroCostCollar'


def cap_of(outcome):
    return leg(outcome, 'financing')['strike_market']


#: Every pillar the gate's quote block carries a two-way on - one bucket each, and what a leg's
#: `spread` rows are keyed by.
PILLARS = {'ATM 0.25', 'ATM 1', 'BF 0.25 0.25', 'BF 0.25 1', 'RR 0.25 0.25', 'RR 0.25 1'}


def pillars_of(outcome):
    return {row['pillar'] for leg_row in outcome['legs'] for row in leg_row['spread']}


def test_the_wing_pillars_are_charged_beside_the_atm_ones(book, two_sided_book):
    """Every quoted pillar is a bucket of its own, so a book quoting a two-way on its RR and BF rows
    charges the leg's risk-reversal and butterfly vega as well as its ATM vega.

    Three books order the cap: the mid one furthest out, the ATM-only one inside it, the fully
    quoted one inside that - and the fully quoted charge is the larger by exactly the wing pillars'
    own rows. A pillar charged at the wrong half, or dropped, lands outside that ordering.

    The risk reversal does not WIDEN both wings here, which is the shape this replaces. It is a
    bucket, charged at its own half against the leg's own `dV/d(RR)`, which is what lets a put's
    risk reversal be charged on the side its wing puts it rather than the side its label does.

    The SEAGULL is held to the same ordering, because three legs is where a per-leg charge could go
    wrong in a way two cannot: its extra sold put finances less, so its cap comes in further still,
    and the sum over three legs is what the coordinate has to absorb.
    """
    atm_only = two_way(book, wings=None)
    floor = params(floor=SPOT * 0.95)
    winged, flat, mid = (structures.quote(document, COLLAR, floor)
                         for document in (two_sided_book, atm_only, book))

    assert abs(winged['net']) <= SOLVE_TOLERANCE, winged['net']
    assert pillars_of(winged) == PILLARS, pillars_of(winged)
    assert pillars_of(flat) == {'ATM 0.25', 'ATM 1'}, 'the ATM-only book charged a wing pillar'
    assert all(row['cost'] > 0.0 for leg_row in winged['legs'] for row in leg_row['spread']
               if row['pillar'].endswith(' 1')), 'a one-year pillar was charged nothing'
    assert winged['edge'] > flat['edge'] > 0.0, 'the wing pillars captured nothing'
    assert SPOT < cap_of(winged) < cap_of(flat) < cap_of(mid), (
        'the wing-charged cap {} is not inside the ATM-only cap {}'.format(
            cap_of(winged), cap_of(flat)))

    bird = params(floor=SPOT * 0.98, lower_floor=SPOT * 0.90)
    winged, flat, mid = (structures.quote(document, 'Seagull', bird)
                         for document in (two_sided_book, atm_only, book))

    assert abs(winged['net']) <= SOLVE_TOLERANCE, winged['net']
    assert len(winged['legs']) == 3 and pillars_of(winged) == PILLARS
    assert winged['edge'] == pytest.approx(
        sum(row['spread_charge'] for row in winged['legs']), rel=1e-12)
    assert winged['edge'] > flat['edge'] > 0.0
    assert SPOT < cap_of(winged) < cap_of(flat) < cap_of(mid), (
        'the seagull cap {} is not inside the ATM-only cap {}'.format(
            cap_of(winged), cap_of(flat)))


def test_a_book_with_no_wing_two_way_quotes_exactly_as_it_always_did(book):
    """The compatibility contract for the wings, in the zero-wide precedent's shape.

    THREE books that quote no wing spread must be one quote to the float. One carries no sides on
    its RR and BF rows. One quotes them ZERO-WIDE, exercising the whole reading. One CROSSES them,
    and reads zero-wide because a desk must never pay a client for a broken print.

    What they do not agree about is what they REPORT, and that is the point. A zero-wide wing is a
    row at a half of nothing, because it was read and it is nothing; no sides at all is NO ROW,
    because there was nothing to read. Every number that moves a price is identical, and all three
    still charge the ATM pillars: this is the WING layer's absence, not the two-way's.
    """
    ask = params(floor=SPOT * 0.95)
    absent, zero_wide, crossed = (structures.quote(two_way(book, wings=wings), COLLAR, ask)
                                  for wings in (None, 0.0, -WING_FRACTION))

    assert [outcome['spread_note'] for outcome in (absent, zero_wide, crossed)] == [None] * 3
    assert pillars_of(absent) == {'ATM 0.25', 'ATM 1'}
    assert pillars_of(zero_wide) == pillars_of(crossed) == PILLARS
    for outcome in (zero_wide, crossed):
        for row, same in zip(absent['legs'], outcome['legs']):
            assert (row['premium'], row['strike_market'], row['solved'],
                    row['spread_charge']) == (same['premium'], same['strike_market'],
                                              same['solved'], same['spread_charge']), row
            assert all(cell['cost'] == 0.0 for cell in same['spread']
                       if not cell['pillar'].startswith('ATM')), 'a wing pillar was charged'
        assert (absent['net'], absent['net_mid'], absent['edge']) == (
            outcome['net'], outcome['net_mid'], outcome['edge'])


def test_a_book_that_never_states_use_quotes_rather_than_raising(book):
    """`Use` is an OPTIONAL field, so its absence is a quote that counts.

    `quote_two_way` walks the block on EVERY quote - that is how it learns which pillars there are
    to charge - so a strict `point['Use']` made a `KeyError` of a book that prices perfectly well,
    inside `quote()` rather than at a refusal seam.

    The gate is the whole quote, not the reader: the same collar off the same surface, the same
    pillars charged, the same cap to the float as the book that states the field.
    """
    stated = two_way(book, wings=None)
    useless = copy.deepcopy(stated)
    points = useless['Calc']['MergeMarketData']['ExplicitMarketData'][
        'Market Prices']['FXVolPrices.USD.ZAR']['instrument']['Points']
    assert all(point.pop('Use') == 'Yes' for point in points), 'the fixture states no Use to drop'

    ask = params(floor=SPOT * 0.95)
    quoted = structures.quote(useless, COLLAR, ask)

    assert structures.quote_points(useless, 'USD.ZAR') == points
    assert pillars_of(quoted) == {'ATM 0.25', 'ATM 1'}, 'a pillar this block does not quote'
    # the charged pass re-solves a bracket seeded on the mid root, so the cap is pinned to the
    # solve's own tolerance rather than to the bit
    assert cap_of(quoted) == pytest.approx(19.171481557, rel=1e-9), cap_of(quoted)
    assert cap_of(quoted) == cap_of(structures.quote(stated, COLLAR, ask))


def engine_runs(work):
    """`(what work returned, how many engine runs it made)`.

    Counted with `sys.monitoring` local events on `Context.run_job` itself - an OBSERVER, so the
    run is the real one and nothing is replaced. The tool id is released either way.
    """
    monitor, runs = sys.monitoring, []
    tool, code = monitor.PROFILER_ID, derivus.Context.run_job.__code__

    def seen(*_):
        runs.append(1)

    monitor.use_tool_id(tool, 'engine runs')
    try:
        monitor.register_callback(tool, monitor.events.PY_START, seen)
        monitor.set_local_events(tool, code, monitor.events.PY_START)
        answer = work()
    finally:
        monitor.set_local_events(tool, code, 0)
        monitor.register_callback(tool, monitor.events.PY_START, None)
        monitor.free_tool_id(tool)
    return answer, len(runs)


@pytest.mark.skipif(sys.version_info < (3, 12), reason='sys.monitoring is the observer')
def test_a_seeded_solve_costs_less_and_a_missed_seed_still_answers_the_full_bracket(book):
    """WHAT THE SEEDED BRACKET BUYS, AND WHAT A MISS COSTS - counted, not asserted in prose.

    The charged pass re-solves a coordinate a spread's width from the mid one, so it brackets
    `SEED_BRACKET` around that root and falls back to the full ends where the narrow span does not
    straddle. Neither half of that is visible in a price, so both are counted here: the collar's
    financing leg solved to its zero-cost target three times against the same book, once with no
    seed, once seeded on the answer, once seeded 50% away from it.

    A HIT costs under HALF the unseeded solve and lands on the same root to 1e-9 - so widening the
    span until the seeding buys nothing fails here, and so does dropping the seed at the call site.

    A MISS returns the unseeded root BIT FOR BIT at no more than two runs over it - the fall-back's
    whole contract, since a narrow bracket that kept its own answer would be a quote off a span
    nobody chose. Deleting the fall-back makes this raise.
    """
    document = copy.deepcopy(book)
    structure = structures.structure_named(COLLAR)
    asked = structures.declared(structure, params(floor=SPOT * 0.95))

    def fresh():
        """Legs of their own per solve - `run_solve` writes its answer back onto the leg it moved."""
        legs = {leg.role: leg for leg in structures.materialize(structure, asked, document)}
        for role, leg in legs.items():
            leg.deal['Reference'] = 'bracket_{}'.format(role)
        return legs

    protection = fresh()['protection']
    spot = structures.engine_spot(document, protection.deal['Underlying_Currency'],
                                  protection.deal['Currency'])
    target = -structures.run_price(document, protection.deal)

    def solve(seed):
        return engine_runs(lambda: structures.run_solve(
            document, fresh()['financing'], 'Strike_Price', target, spot, seed))

    (unseeded, _), wide = solve(None)
    (on_the_root, _), narrow = solve(unseeded)
    (past_it, _), over = solve(unseeded * 1.5)

    assert 2 * narrow < wide, 'a seeded solve cost {} engine runs against {}'.format(narrow, wide)
    assert on_the_root == pytest.approx(unseeded, rel=1e-9), (on_the_root, unseeded)
    assert past_it == unseeded, 'the fall-back answered something the full bracket does not'
    assert over <= wide + 2, 'a missed seed cost {} engine runs against {}'.format(over, wide)


#: The desk's mandate, as a book declares one; the gates below vary one field at a time off this.
POLICY = {'participation': 0.5, 'floor': 'mid', 'scope': 'vol',
          'bucket_limit': None, 'min_ticket_bp': 0.0, 'firm_seconds': 600}


def with_policy(document, **stated):
    """`document` with a `Quote Policy` block on it - the whole risk-impact feature's on switch."""
    out = copy.deepcopy(document)
    out['Calc'][structures.QUOTE_POLICY] = dict(POLICY, **stated)
    return out


def holding(document, deals):
    """`document` with exactly `deals` standing in its deal tree, through `book_node`."""
    out = copy.deepcopy(document)
    out['Calc']['Deals']['Deals']['Children'] = [structures.book_node(deal) for deal in deals]
    return out


@pytest.fixture(scope='module')
def standing(two_sided_book):
    """One collar quoted at the full two-way - the trade the books below already carry."""
    return structures.quote(two_sided_book, COLLAR, params(floor=SPOT * 0.95))


def test_a_book_with_no_quote_policy_quotes_exactly_as_it_always_did(two_sided_book, standing):
    """The compatibility contract for the risk-impact half: absence is the identical code path.

    A book declaring no `Quote Policy` never reaches the BOOK's greeks runs at all - `risk.scale` is
    None rather than 1.0, the difference between "did not run" and "ran and decided nothing". A book
    declaring one with `participation` at ZERO runs the WHOLE layer and must land on the identical
    floats, so only a stated participation can move a price. Each leg's own vega run is made either
    way: that is the charge, not the tightening.

    Both quotes are given against a book already carrying a position, so the silence is a decision
    rather than an empty book's default.
    """
    held = holding(two_sided_book, [structures.mirror(standing['deal'])])
    plain = structures.quote(held, COLLAR, params(floor=SPOT * 0.95))
    zero = structures.quote(with_policy(held, participation=0.0), COLLAR,
                            params(floor=SPOT * 0.95))

    assert plain['risk']['scale'] is None and plain['risk']['buckets'] == []
    assert plain['risk']['policy'] is None and structures.QUOTE_POLICY in plain['risk']['note']
    assert zero['risk']['scale'] == 1.0 and zero['risk']['buckets'], 'the layer never ran'
    assert zero['risk']['policy'] == dict(POLICY, participation=0.0)

    for row, same in zip(plain['legs'], zero['legs']):
        assert (row['premium'], row['strike_market'], row['solved'], row['spread_charge']) == (
            same['premium'], same['strike_market'], same['solved'], same['spread_charge']), row
    assert (plain['net'], plain['net_mid'], plain['edge']) == (
        zero['net'], zero['net_mid'], zero['edge'])


def renamed(deal, suffix):
    """The same trade under its own reference - the mtm frame is keyed by reference."""
    copied = copy.deepcopy(deal)
    copied['Reference'] += suffix
    for child in copied.get('Children', []):
        child['Instrument']['.Deal']['Reference'] += suffix
        child['Instrument']['.Deal']['Structure_Reference'] = copied['Reference']
    return copied


def test_an_offset_quotes_tighter_than_a_repeat(book, two_sided_book, standing):
    """The ruling, priced: what a trade costs is what hedging the RESIDUAL it leaves costs, at the
    market's own two-way.

    The registry has no sell-side collar, so the opposite SIDE goes on the BOOK rather than into the
    quote. Two books hold the same trade the two ways a desk can: SHORT it (the mirror of a collar
    it quoted) and LONG it. Quoting into the first piles the risk on again; into the second it nets
    flat. One structure, one policy, one set of parameters, opposite signs.

    Four claims. The offsetting quote is TIGHTER (`scale` strictly inside 1) and the repeat is not
    (`scale` exactly 1 - the market's spread is the ceiling in v1, so no surcharge). The charges
    order the same way. The offset's solved cap lands strictly between the full-spread cap and the
    MID cap, the floor the policy declares. And every scale is in [0, 1].

    The buckets say why: the mirror doubles `RR 0.25 1` on one book and all but zeroes it on the
    other, charged at that pillar's own half-spread. ALL BUT, and that residual is the declared
    one-pass approximation's own size - the book holds the collar at the cap it was DEALT at while
    the candidate is measured at the MID one, so the two do not cancel to the bit. Measured here at
    4.4% of the bucket, against the 200% the repeat piles on.
    """
    ask = params(floor=SPOT * 0.95)
    short_book = with_policy(holding(two_sided_book, [structures.mirror(standing['deal'])]))
    long_book = with_policy(holding(two_sided_book, [standing['deal']]))
    adding = structures.quote(short_book, COLLAR, ask)
    reducing = structures.quote(long_book, COLLAR, ask)
    full_spread = structures.quote(
        holding(two_sided_book, [structures.mirror(standing['deal'])]), COLLAR, ask)

    assert adding['risk']['scale'] == 1.0, 'a risk-adding trade was surcharged past the two-way'
    assert adding['risk']['saving'] == 0.0
    assert 0.0 < reducing['risk']['scale'] < 1.0, reducing['risk']['scale']
    assert reducing['risk']['saving'] > 0.0
    assert reducing['risk']['charge_effective'] < adding['risk']['charge_effective']
    assert adding['risk']['charge_effective'] == pytest.approx(adding['risk']['charge_full'])
    assert reducing['risk']['coordinates'] == 'quote-space'

    # the mirror doubles the skew bucket on one book and cancels it on the other - one number read
    # from both sides, which no sign error survives
    skew = {row['bucket']: row for row in reducing['risk']['buckets']}['RR 0.25 1']
    piled = {row['bucket']: row for row in adding['risk']['buckets']}['RR 0.25 1']
    assert skew['before'] == pytest.approx(-piled['before'], rel=1e-9)
    assert abs(skew['after']) < 0.05 * abs(skew['before']), 'the offset left skew standing'
    assert abs(skew['after']) < 0.05 * abs(piled['after']), 'the two books read the same residual'
    assert piled['after'] == pytest.approx(2.0 * piled['before'], rel=0.05)
    assert skew['half_spread'] > 0.0

    # ONE meaning per name. A pillar's `half` is the MARKET's, the same number the bucket beside it
    # prices a residual at, and it does not move when a policy tightens; the SCALE is said once
    # under `risk.scale` and carried in the money, so `cost` and `spread_charge` are what was
    # charged and a consumer can read both the quote and what the desk did to it
    scale = reducing['risk']['scale']
    buckets = {row['bucket']: row['half_spread'] for row in reducing['risk']['buckets']}
    assert reducing['edge'] == pytest.approx(reducing['risk']['charge_effective'], rel=1e-12)
    assert reducing['edge'] == pytest.approx(
        sum(row['spread_charge'] for row in reducing['legs']), rel=1e-12)
    for row, full in zip(reducing['legs'], adding['legs']):
        assert row['spread_charge'] == pytest.approx(scale * full['spread_charge'], rel=1e-9)
        for cell, wide in zip(row['spread'], full['spread']):
            assert cell['half'] == wide['half'] == buckets[cell['pillar']], cell
            assert cell['cost'] == pytest.approx(scale * wide['cost'], rel=1e-12)
    assert adding['edge'] == pytest.approx(adding['risk']['charge_full'], rel=1e-12)

    assert cap_of(adding) == cap_of(full_spread), 'the repeat is not the full-spread quote'
    assert cap_of(adding) < cap_of(reducing) < cap_of(structures.quote(book, COLLAR, ask)), (
        'the tightened cap {} is not between the full-spread cap {} and the mid one'.format(
            cap_of(reducing), cap_of(adding)))
    assert all(0.0 <= outcome['risk']['scale'] <= 1.0 for outcome in (adding, reducing))


def test_a_margin_survives_a_policy_that_tightens_the_two_way(two_sided_book, standing):
    """THE TWO CHARGES DO NOT SCALE TOGETHER. The policy tightens the MARKET's spread, which the
    desk measured and can give back; it has no opinion at all about a margin the client agreed.

    One book long the trade, so the policy tightens (`scale < 1`), and one short it, so it does not
    (`scale == 1`, the market's own spread being the ceiling). On both, a quote at a 50,000 rand
    margin has to come back at MINUS that margin exactly - not minus the scaled margin - while
    `edge` is `scale x charge_full` and nothing else, and the bank's booked mirror holds the margin
    and the edge together.

    The margin is a quarter of a percent of THIS gate's notional, so the book already standing
    against it is the one the trade offsets exactly. Shaving the margin by the scale would cost the
    client 4.4% of it here, silently: every leg, every premium and every identity but this one
    stays true.
    """
    asked = params(floor=SPOT * 0.95)
    tightens = with_policy(holding(two_sided_book, [standing['deal']]))
    stands = with_policy(holding(two_sided_book, [structures.mirror(standing['deal'])]))

    for label, document in (('tightens', tightens), ('stands', stands)):
        priced = structures.quote(document, COLLAR, dict(asked), margin=SMALL_MARGIN)
        value, risk = priced['margin']['value'], priced['risk']

        assert value == pytest.approx(SMALL_MARGIN['amount'] / SPOT, rel=1e-12)
        assert (risk['scale'] < 1.0) == (label == 'tightens'), (label, risk['scale'])
        assert priced['net'] == pytest.approx(-value, abs=SOLVE_TOLERANCE), (
            '{}: the client was quoted {} against the {} they agreed'.format(
                label, priced['net'], -value))
        assert priced['edge'] == pytest.approx(
            risk['scale'] * risk['charge_full'], rel=1e-12), label
        assert priced['edge'] == pytest.approx(
            sum(row['spread_charge'] for row in priced['legs']), rel=1e-12), label
        assert priced['net_mid'] == pytest.approx(
            -(value + priced['edge']), abs=SOLVE_TOLERANCE), label

        marked = values(two_sided_book, [structures.mirror(priced['deal'])])
        assert marked[priced['deal']['Reference']] == pytest.approx(
            value + priced['edge'], abs=SOLVE_TOLERANCE), (
            '{}: the bank is not marked at the margin and the spread together'.format(label))


def test_the_cap_and_the_floor(two_sided_book, standing):
    """The two limits the policy declares, each made to bind.

    THE CAP. A book holding TWO of the trade still has one standing after the offset, so the
    residual is real - a `bucket_limit` under it suspends the tightening entirely and NAMES the
    bucket. A book already over its limit somewhere does not get to quote tighter on the strength of
    netting down elsewhere.

    THE FLOOR. `min_ticket_bp` is flat bp of NOTIONAL, in rand while the charge is in report
    currency, so it crosses on the same `FxRate` ratio. Set between the tightened charge and the
    full one it binds exactly: the effective charge lands ON the ticket, and the quote is wider than
    the unfloored one and no wider than the two-way.
    """
    ask = params(floor=SPOT * 0.95)
    twice = holding(two_sided_book, [renamed(standing['deal'], '_a'),
                                     renamed(standing['deal'], '_b')])
    free = structures.quote(with_policy(twice), COLLAR, ask)
    residual = max(abs(row['after']) for row in free['risk']['buckets'])
    assert free['risk']['scale'] < 1.0 and residual > 0.0, 'nothing was tightened to cap'

    capped = structures.quote(with_policy(twice, bucket_limit=residual / 2.0), COLLAR, ask)
    assert capped['risk']['scale'] == 1.0
    assert capped['risk']['saving'] == 0.0
    assert 'bucket_limit' in capped['risk']['note'] and 'RR 0.25 1' in capped['risk']['note']
    assert cap_of(capped) == cap_of(structures.quote(
        holding(two_sided_book, [structures.mirror(standing['deal'])]), COLLAR, ask)), (
        'a capped quote is not the full-spread quote')

    # bp of notional, crossed to the report currency exactly as the runner crosses it
    per_bp = structures.BASIS_POINT * NOTIONAL / SPOT
    tight, full = free['risk']['charge_effective'], free['risk']['charge_full']
    ticket_bp = 0.5 * (tight + full) / per_bp
    floored = structures.quote(with_policy(twice, min_ticket_bp=ticket_bp), COLLAR, ask)

    assert tight < ticket_bp * per_bp < full, 'the ticket does not bind between the two charges'
    assert floored['risk']['charge_effective'] == pytest.approx(ticket_bp * per_bp, rel=1e-12)
    assert tight < floored['risk']['charge_effective'] < full
    assert free['risk']['scale'] < floored['risk']['scale'] < 1.0
    assert cap_of(capped) < cap_of(floored) < cap_of(free), (
        'the floored cap is not between the full-spread one and the unfloored one')


def test_a_knock_in_plus_a_knock_out_is_the_vanilla(book):
    """In-out parity through the engine: a bought up-and-IN and a bought up-and-OUT call on the same
    strike and barrier are between them the vanilla. Three deals by hand, one run, no solve.

    The identity the forward extra's sold leg rests on, and the census records the analytic knock-IN
    branch of `pv_barrier_option` as executed by almost no test. Closed forms both sides, so the
    tolerance is analytic: measured 1.1e-16 relative, one ULP, against a 1e-9 gate.

    That exactness bounds the claim: the two branches are complementary term sets of one
    Merton-Reiner-Rubinstein decomposition (`B - C + D` and `A - B + C - D`, `A` the vanilla), so
    the sum cancels term by term in floating point. What this holds is that the IN branch is REACHED
    and composes to that decomposition - not that the formula is accurate.
    """
    knock = SPOT * 1.10
    common = {'Currency': 'ZAR', 'Discount_Rate': 'ZAR', 'Underlying_Currency': 'USD',
              'Underlying_Amount': NOTIONAL / SPOT, 'FX_Volatility': 'USD.ZAR',
              'Buy_Sell': 'Buy', 'Option_Type': 'Call', 'Strike_Price': SPOT,
              'Expiry_Date': {'.Timestamp': (BASE + pd_offset()).strftime('%Y-%m-%d')}}
    # a deal block IS the pricer's field dict, so the barrier's two fields are written out -
    # continuous monitoring, no rebate - exactly as the runner writes them
    barrier = dict(common, Object='FXBarrierOption', Cash_Rebate=0.0,
                   Barrier_Monitoring_Frequency={'.DateOffset': '0M'}, Barrier_Price=knock)
    priced = values(book, [
        dict(common, Object='FXOptionDeal', Reference='VANILLA', Option_Style='European'),
        dict(barrier, Reference='KNOCK_IN', Barrier_Type='Up_And_In'),
        dict(barrier, Reference='KNOCK_OUT', Barrier_Type='Up_And_Out')])

    assert priced['KNOCK_IN'] > 0 and priced['KNOCK_OUT'] > 0, (
        'a bought barrier is worth something, or the barrier is unreachable and this is vacuous')
    assert priced['KNOCK_IN'] + priced['KNOCK_OUT'] == pytest.approx(priced['VANILLA'], rel=1e-9)


def test_a_market_strike_reaches_the_engine_axis(book):
    """The conversion on its own: a floor quoted USDZAR 15.50 on a rand notional is a deal struck at
    `1/15.50` dollars per rand, and the leg reads back 15.50. The option SENSE crosses with it, and
    both live in the runner so a structure declares neither.
    """
    outcome = structures.quote(book, 'ZeroCostCollar', params(floor=15.50))
    protection = outcome['deal']['Children'][0]['Instrument']['.Deal']

    assert protection['Strike_Price'] == 1.0 / 15.50
    assert leg(outcome, 'protection')['strike_market'] == pytest.approx(15.50, rel=1e-12)
    assert protection['Option_Type'] == 'Call', 'a floor on the pair is a call on the rand'
    assert protection['Underlying_Currency'] == 'ZAR' and protection['Currency'] == 'USD'
    assert protection['Expiry_Date'] == {'.Timestamp': '2025-06-28'}, 'the 1Y tenor mis-parsed'
    assert protection['Structure_Reference'] == outcome['deal']['Reference']


def test_a_live_cross_lands_exactly_where_engine_spot_reads_it_back(book):
    """The live-spot conversion, on a real book with no terminal near it.

    `with_live_spots` is the exact inverse of `engine_spot`, so this is the round trip: a market
    cross written in against the book's `Base_Currency`, read back out on the DEAL axis, and read a
    third time in MARKET terms off a real quote's `spot` block. The runner reports the spot it
    PRICED on, so the reading survives the axis inversion a rand notional puts it through.

    Two refusals. A pair with NEITHER leg against the base cannot be placed without triangulating,
    which is a market view rather than a tick; a currency the book carries no `FxRate` for is a new
    price factor, which is authoring rather than the `bind='value'` seam a spot moves through.
    """
    moved = copy.deepcopy(book)
    written = structures.with_live_spots(moved, {'USDZAR': 16.31})
    factors = moved['Calc']['MergeMarketData']['ExplicitMarketData']['Price Factors']

    assert written == {'ZAR': 1.0 / 16.31}
    assert factors['FxRate.USD']['Spot'] == 1.0, 'the base currency prices itself and is not moved'
    assert structures.engine_spot(moved, 'USD', 'ZAR') == pytest.approx(16.31, rel=1e-15)

    outcome = structures.quote(moved, 'Straddle', params(strike=16.31))
    assert outcome['spot'] == {'value_market': pytest.approx(16.31, rel=1e-15),
                               'source': 'book', 'note': None}

    with pytest.raises(ValueError, match='neither of its legs'):
        structures.with_live_spots(copy.deepcopy(book), {'EURJPY': 160.0})
    with pytest.raises(ValueError, match='FxRate.GBP is missing'):
        structures.with_live_spots(copy.deepcopy(book), {'GBPUSD': 1.27})


def test_the_same_trade_quotes_the_same_from_either_side_of_the_pair(book):
    """A straddle on 1,000,000 rand and one on the 54,054 dollars that buys at 18.50 are the SAME
    trade, quoted at the same money.

    They travel opposite paths. The rand notional is the pair's quote currency, so its legs are
    options on rand settled in dollars, struck at `1/18.50`, the market's call written as an engine
    put. The dollar notional is the base currency and nothing inverts. One number out of both, which
    no single-orientation runner can fake.
    """
    both_ways = {'pair': PAIR, 'expiry': EXPIRY, 'strike': SPOT}
    in_rand = structures.quote(book, 'Straddle', dict(
        both_ways, notional=NOTIONAL, notional_currency='ZAR'))
    in_dollars = structures.quote(book, 'Straddle', dict(
        both_ways, notional=NOTIONAL / SPOT, notional_currency='USD'))

    assert in_dollars['net'] == pytest.approx(in_rand['net'], rel=1e-12)
    call = in_dollars['deal']['Children'][0]['Instrument']['.Deal']
    assert call['Strike_Price'] == SPOT, 'a base-currency notional inverts nothing'
    assert call['Option_Type'] == 'Call' and call['Underlying_Currency'] == 'USD'
    assert call['Currency'] == 'ZAR' and call['Discount_Rate'] == 'ZAR'
    assert leg(in_dollars, 'call')['strike_market'] == SPOT


def test_a_forward_extra_quotes_the_same_from_either_side_of_the_pair(book):
    """The straddle's statement with THREE conversions instead of two.

    What the rand side crosses that the straddle's did not is the BARRIER: its level inverts like a
    strike and its DIRECTION inverts with it (`Up_And_In` on the pair is `Down_And_In` on the rand),
    while the dollar side crosses nothing. Get either half wrong and the two orientations solve
    different barriers for one trade, so the sharp claim is the solved LEVELS in market terms.

    The equivalent notional converts at the STRIKE, not the spot: a rand-notional option on the
    reciprocal pays `max(1/S - 1/K, 0) x N`, which is `(N/K) x max(K - S, 0)` in rand. The
    straddle's version is struck at the money and cannot tell the two divisors apart; away from it,
    `N / SPOT` misprices by exactly the moneyness (measured 3.09% at 0.97 spot).
    """
    protected = SPOT * 0.97
    both_ways = {'pair': PAIR, 'expiry': EXPIRY, 'sell_currency': 'USD',
                 'buy_currency': 'ZAR', 'floor': protected}
    in_rand = structures.quote(book, 'ForwardExtra', dict(
        both_ways, notional=NOTIONAL, notional_currency='ZAR'))
    in_dollars = structures.quote(book, 'ForwardExtra', dict(
        both_ways, notional=NOTIONAL / protected, notional_currency='USD'))

    assert in_dollars['net'] == pytest.approx(in_rand['net'], abs=1e-6)
    assert leg(in_dollars, 'protection')['premium'] == pytest.approx(
        leg(in_rand, 'protection')['premium'], rel=1e-9)
    assert leg(in_dollars, 'reversion')['barrier_market'] == pytest.approx(
        leg(in_rand, 'reversion')['barrier_market'], rel=1e-6), (
        'the two orientations solved different barriers for one trade')

    booked = {side: quote['deal']['Children'][1]['Instrument']['.Deal']
              for side, quote in (('rand', in_rand), ('dollars', in_dollars))}
    assert booked['dollars']['Barrier_Type'] == 'Up_And_In', 'a base-currency notional flips nothing'
    assert booked['dollars']['Strike_Price'] == pytest.approx(protected, rel=1e-12)
    assert booked['rand']['Barrier_Type'] == 'Down_And_In'
    assert booked['dollars']['Barrier_Price'] > SPOT
    assert booked['rand']['Barrier_Price'] < 1.0 / SPOT, 'the engine barrier is not the reciprocal'
    assert booked['rand']['Barrier_Price'] == pytest.approx(
        1.0 / leg(in_rand, 'reversion')['barrier_market'], rel=1e-12)


def test_a_forward_extra_importer_caps_the_pair_and_solves_a_lower_barrier(book):
    """A client selling rand to buy dollars buys the USDZAR call and sells the lower knock-in put.

    The cap is the rate paid for dollars. If USDZAR drops through the solved barrier, the sold put
    knocks in and the package becomes a forward at that cap; otherwise the client keeps the lower
    spot. This is the forward extra's importer form, not a second product.
    """
    cap = SPOT * 1.03
    outcome = structures.quote(book, 'ForwardExtra', params(
        sell_currency='ZAR', buy_currency='USD', cap=cap))
    protection, reversion = leg(outcome, 'protection'), leg(outcome, 'reversion')

    assert abs(outcome['net']) <= SOLVE_TOLERANCE, outcome['net']
    assert protection['premium'] > 0 > reversion['premium']
    assert protection['strike_market'] == pytest.approx(cap, rel=1e-12)
    assert reversion['strike_market'] == pytest.approx(cap, rel=1e-12)
    assert reversion['barrier_market'] < SPOT
    booked = outcome['deal']['Children'][1]['Instrument']['.Deal']
    # A rand notional prices on reciprocal ZARUSD, so the market put/down-in crosses once.
    assert booked['Option_Type'] == 'Call'
    assert booked['Barrier_Type'] == 'Up_And_In'


def test_a_quote_is_an_act_not_a_lookup(book):
    """Two identical asks are two quotes: content addressing is right for a computation and wrong
    for an event, so the id carries a submission clock and never coalesces - while the PRICE is the
    same, because the book has not moved."""
    first = structures.quote(book, 'Straddle', params(strike=SPOT))
    second = structures.quote(book, 'Straddle', params(strike=SPOT))

    assert first['quote_id'] != second['quote_id']
    assert first['deal']['Reference'] != second['deal']['Reference']
    assert first['net'] == pytest.approx(second['net'], rel=1e-12)


def test_what_a_client_cannot_be_quoted_refuses_before_a_leg_is_priced(book):
    """Four asks nobody can be quoted on, each named against the structure's own declarations: no
    parameters at all, the pair stated backwards, an expiry on the base date, and a notional the
    client would be paid to take.

    KILLING MUTATION: `KeyError: 'pair'` for the first; for the backwards pair a quote that got as
    far as `ZeroCostCollar-..._protection priced but reported no mtm row`, every leg having been
    dropped at load for a surface the book does not carry; and two QUOTES for the last two - the
    zero-day collar priced as if live, the negative notional priced with negative premiums.
    """
    refusals = {}
    for label, ask in [('none', {}),
                       ('backwards', params(pair='ZARUSD', floor=1.0 / (SPOT * 0.95))),
                       ('expired', params(floor=SPOT * 0.95, expiry='0D')),
                       ('negative', params(floor=SPOT * 0.95, notional=-NOTIONAL))]:
        with pytest.raises(ValueError) as refusal:
            structures.quote(book, 'ZeroCostCollar', ask)
        refusals[label] = str(refusal.value)

    assert refusals['none'].startswith('ZeroCostCollar states no pair, expiry, notional')
    assert refusals['backwards'].startswith('ZARUSD is not a pair this book quotes')
    assert 'USDZAR' in refusals['backwards']
    assert 'on or before the base date' in refusals['expired']
    assert refusals['negative'].startswith('a notional is a positive amount, not -1e+06')


def test_an_unparsed_tenor_refuses_rather_than_expiring_today(book):
    """A tenor that does not parse must never fall through to the base date - a zero-day option
    prices at zero without complaining, quoting the client nothing for something."""
    with pytest.raises(ValueError) as refusal:
        structures.quote(book, 'Straddle', dict(params(strike=SPOT), expiry='three months'))
    assert 'three months' in str(refusal.value)


# --------------------------------------------------------------------------------------------
# the accrual strips: a TARF and an accumulator, which are one leg and a SCHEDULE
# --------------------------------------------------------------------------------------------
@pytest.fixture(scope='module')
def accrual_book(book):
    """The same book at `ACCRUAL_SIMS` inner paths. `MCMC_Simulations` is 1 on `test_service`'s job,
    right for a cashflow's arithmetic and one path for a TARF.
    """
    document = copy.deepcopy(book)
    document['Calc']['Calculation']['MCMC_Simulations'] = ACCRUAL_SIMS
    return document


def accrual_params(**extra):
    """A strip is dealt BOTH ways and neither is a level, so every accrual ask states its
    direction: `buy_currency` USD is the client buying the base at each fixing, today's form."""
    return params(fixing_frequency=FIXING_FREQUENCY, buy_currency='USD', **extra)


def only_leg(outcome):
    """The one deal a strip composes to - the container's single child."""
    return outcome['deal']['Children'][0]['Instrument']['.Deal']


def test_an_accumulator_crosses_both_axes_and_a_tarf_refuses_the_second(accrual_book):
    """The axis gate for the strips, in two halves because the two structures answer differently.

    THE ACCUMULATOR CROSSES: one on 1,000,000 rand and one on the dollars that buys at the solved
    strike are the SAME trade. The rand side crosses everything at once - the strike inverts, the
    market Call is written as an engine Put, the knock-out LEVEL inverts with the strike and its
    DIRECTION with it (`Up_And_Out` on the pair is `Down_And_Out` on the rand) - while the dollar
    side crosses nothing. The equivalent notional converts at the SOLVED strike, which costs nothing
    because a zero-cost strike is scale-invariant: the sizing only has to make the two NETS
    comparable.

    THE TARF DOES NOT, and refuses. Its target caps the ACCRUAL, a sum of differences rather than a
    level, and `1/S - 1/K` is not the reciprocal of `S - K` - so no number in reciprocal units means
    the client's cap. `InvertedTarget` is not the way out: it moves the whole fixing onto the
    reciprocal axis, paying the notional per unit of MOVE, a coherent product and not the one
    `notional_currency` names (measured 0.77% apart in the solved strike). So the refusal names the
    currency, and `InvertedTarget` is False on every leg the runner builds.
    """
    both_ways = {'pair': PAIR, 'expiry': EXPIRY, 'fixing_frequency': FIXING_FREQUENCY,
                 'buy_currency': 'USD', 'knockout': SPOT * 1.10}
    in_rand = structures.quote(accrual_book, 'Accumulator', dict(
        both_ways, notional=NOTIONAL, notional_currency='ZAR'))
    strike = leg(in_rand, 'accumulator')['strike_market']
    in_dollars = structures.quote(accrual_book, 'Accumulator', dict(
        both_ways, notional=NOTIONAL / strike, notional_currency='USD'))

    assert abs(in_rand['net']) <= SOLVE_TOLERANCE and abs(in_dollars['net']) <= SOLVE_TOLERANCE
    assert leg(in_dollars, 'accumulator')['strike_market'] == pytest.approx(
        strike, rel=AXIS_TOLERANCE), 'the two orientations solved different strikes for one trade'

    rand, dollars = only_leg(in_rand), only_leg(in_dollars)
    assert rand['Option_Type'] == 'Put' and dollars['Option_Type'] == 'Call'
    assert rand['Barrier_Type'] == 'Down_And_Out', 'up on the pair is down on the rand'
    assert dollars['Barrier_Type'] == 'Up_And_Out', 'a base-currency notional flips nothing'
    assert rand['Barrier_Price'] == pytest.approx(1.0 / (SPOT * 1.10), rel=1e-12)
    assert dollars['Barrier_Price'] == pytest.approx(SPOT * 1.10, rel=1e-12)
    assert rand['Strike_Price'] == pytest.approx(1.0 / strike, rel=1e-12)

    with pytest.raises(ValueError) as refusal:
        structures.quote(accrual_book, 'TargetRedemptionForward', accrual_params(target=TARGET))
    assert 'ZAR' in str(refusal.value) and 'accrual cap' in str(refusal.value)


def test_an_accrual_strip_costs_nothing_and_strikes_better_than_the_forward(accrual_book):
    """The zero-cost identity for both strips, and the DIRECTION that says the client got the
    bargain they pay gearing and a knock-out for.

    A TARF is dealt at no upfront (`Solve('tarf', 'Strike_Price', 0.0)`, no funding leg), so the net
    is zero to the solve's tolerance and the composed deal reprices to it leg for leg.

    The SIDE of the forward is not a tautology. The forward IS the spot here, and at a strike of the
    forward the bought and sold legs are worth the SAME per fixing - so `leverage` times the sold
    one outweighs the bought one and the package is negative. The strike comes DOWN until they
    balance, and a client accruing `(S - K)+` is better off the lower it goes. A runner that
    inverted a strike twice, or fed the pricer a leverage of one, lands above the forward.

    The rest is what the parameters became: twelve monthly fixings settling two days on, the geared
    notional at `leverage x notional` off a default the client never stated, and the target copied
    through UNCONVERTED with `InvertedTarget` False.

    The model pin is exercised by its ABSENCE, the only arm this repo can reach: no fixture carries
    a `LogVar2FJModelParameters` factor for an FX underlying, so the leg prices GBM and SAYS so.
    Pinning the model on a book with no calibration instead raises inside the dependency loop, which
    SKIPS the deal and marks the quote's only leg at nothing.
    """
    tarf = structures.quote(accrual_book, 'TargetRedemptionForward', dict(
        accrual_params(target=TARGET), notional_currency='USD'))
    row, deal = leg(tarf, 'tarf'), only_leg(tarf)

    assert abs(tarf['net']) <= SOLVE_TOLERANCE, tarf['net']
    assert row['strike_market'] < SPOT, (
        'a zero-cost TARF strike is not better than the forward for the client')
    assert row['solved'] == {'Strike_Price': pytest.approx(row['strike_market'], rel=1e-12)}
    assert row['barrier_market'] is None, 'a TARF leg carries no barrier'

    assert len(deal['TARF_ExpiryDates']) == 12, 'a 1Y tenor holds twelve monthly fixings'
    fixing, settlement, observed = deal['TARF_ExpiryDates'][0]
    assert fixing == {'.Timestamp': '2024-07-28'} and settlement == {'.Timestamp': '2024-07-30'}
    assert observed == 0.0, 'a quote is struck today and nothing in it has fixed'
    assert deal['Expiry_Date'] == deal['TARF_ExpiryDates'][-1][1], (
        'a strip expires with its last cashflow, not with its last fixing')
    assert deal['LeverageNotional'] == 2.0 * NOTIONAL, 'the declared leverage never reached the deal'
    assert deal['TargetLevel'] == TARGET and deal['InvertedTarget'] is False, (
        'the target is the client number, on the pair own axis')
    assert 'LogVar2FJModelParameters' in row['note'], row['note']
    assert not accrual_book['Calc']['MergeMarketData']['ExplicitMarketData'].get(
        'Valuation Configuration'), 'a model was pinned on a book that cannot price it'

    priced = values(accrual_book, [tarf['deal']])
    assert priced[row['reference']] == pytest.approx(row['premium'], rel=1e-9), (
        'the container reports a leg the quote does not')
    assert priced[tarf['deal']['Reference']] == pytest.approx(tarf['net'], abs=1e-6)

    accumulator = structures.quote(accrual_book, 'Accumulator', dict(
        accrual_params(knockout=SPOT * 1.10), notional_currency='USD'))
    accrued = leg(accumulator, 'accumulator')
    assert abs(accumulator['net']) <= SOLVE_TOLERANCE, accumulator['net']
    assert accrued['strike_market'] < SPOT, 'the same bargain, and the same side of the forward'
    assert accrued['barrier_market'] == pytest.approx(SPOT * 1.10, rel=1e-12)
    assert only_leg(accumulator)['LeverageNotional'] == 2.0 * NOTIONAL
    assert 'Expiry_Date' not in only_leg(accumulator), (
        'FXAccumulatorOptionDeal declares no Expiry_Date, so the block must not carry one')


def test_a_strip_at_a_margin_strikes_further_from_the_client(accrual_book):
    """The single-solve half of the margin. A strip has no financing leg to move, so the charge
    goes onto the TARGET itself: zero becomes minus the margin, and the one coordinate the recipe
    solves - the strike - absorbs it.

    The client accrues `(S - K)+`, so the desk's side is UP, and the direction is the claim: a sign
    error here quotes the client a BETTER strike for paying a margin. Measured on this book at
    these paths: 9.2e-4 relative, against the 2.5e-5 the two zero-cost orientations differ by, so
    the assertion is made at `AXIS_TOLERANCE` and clears it four times over.
    """
    asked = dict(accrual_params(target=TARGET), notional_currency='USD')
    plain = structures.quote(accrual_book, 'TargetRedemptionForward', dict(asked))
    outcome = structures.quote(accrual_book, 'TargetRedemptionForward', dict(asked), margin=MARGIN)
    struck, was = leg(outcome, 'tarf')['strike_market'], leg(plain, 'tarf')['strike_market']

    assert outcome['net'] * outcome['spot']['value_market'] == pytest.approx(
        -50_000.0, abs=SOLVE_TOLERANCE * SPOT)
    assert outcome['net'] == pytest.approx(-outcome['margin']['value'], abs=SOLVE_TOLERANCE)
    assert (struck - was) / was > AXIS_TOLERANCE, 'the margin never reached the solved strike'
    assert (outcome['deal']['Sales_Margin'],
            outcome['deal']['Sales_Margin_Currency']) == (50_000.0, 'ZAR')


def test_a_strip_ends_on_its_own_expiry_or_refuses(book):
    """The two ways a fixing strip comes out SHORT of the tenor it was quoted at, neither silent.

    A FREQUENCY THAT DOES NOT DIVIDE. A 1Y ticket at 5M fixes twice and stops, and nothing
    downstream can tell - a TARF's `Expiry_Date` is the last SETTLEMENT, so the deal is priced,
    reported and spread at the SHORT tenor while the ticket says 1Y. So the strip refuses, naming
    the expiry, the frequency, the last fixing and the two ways out.

    A BASE DATE CARRYING A TIME. `expiry_date` normalizes to a date and a terminal snapshot stamps
    `Base_Date` at 16:30, so the final fixing landed one comparison past midnight and a
    twelve-fixing year became eleven. The base is normalized before the loop.
    """
    import pandas as pd

    with pytest.raises(ValueError) as refusal:
        structures.fixing_grid(BASE, '1Y', '5M')
    assert '2025-04-28' in str(refusal.value), 'the refusal must name the fixing it would end on'
    assert '5M' in str(refusal.value) and '1Y' in str(refusal.value)
    assert 'divides the tenor' in str(refusal.value), 'a refusal without a remedy'

    stamped = structures.fixing_grid(pd.Timestamp(BASE) + pd.Timedelta(hours=16, minutes=30),
                                     '1Y', '1M')
    assert len(stamped) == len(structures.fixing_grid(BASE, '1Y', '1M')) == 12
    assert stamped[-1][0] == {'.Timestamp': '2025-06-28'}, 'the last fixing IS the expiry'

    # and the quote refuses through the runner, not just the helper
    with pytest.raises(ValueError) as quoted:
        structures.quote(book, 'TargetRedemptionForward', dict(
            accrual_params(target=TARGET), fixing_frequency='5M', notional_currency='USD'))
    assert '5M' in str(quoted.value)


def test_the_axis_refusal_fires_before_the_deal_is_furnished(book):
    """A refusal must not leave a half-built deal behind. `furnish_accrual` writes the schedule and
    the geared notional onto the caller's deal block IN PLACE, so the axis refusal has to be the
    function's FIRST statement or a caller that catches it holds a strip and a leverage for a trade
    that was never quoted.

    The ordering shows in the wording: an inverted TARF whose frequency also does not divide reports
    the AXIS rather than the schedule it never got to build.
    """
    deal = {'Object': 'FXTARFOptionDeal', 'Currency': 'USD', 'Underlying_Currency': 'ZAR'}
    with pytest.raises(ValueError) as refusal:
        structures.furnish_accrual(
            deal, params(target=TARGET, fixing_frequency='5M'), book, BASE, 'ZAR', True)

    assert 'accrual cap' in str(refusal.value) and 'ZAR' in str(refusal.value)
    assert '5M' not in str(refusal.value), 'the schedule was built before the axis was checked'
    assert deal == {'Object': 'FXTARFOptionDeal', 'Currency': 'USD', 'Underlying_Currency': 'ZAR'}


def calibrated(document):
    """The book with the pair's spot-model fit installed under its NON-BASE token, which is the
    only leg of the pair an `FxRate` can be."""
    document = copy.deepcopy(document)
    document['Calc']['MergeMarketData']['ExplicitMarketData']['Price Factors'][
        'LogVar2FJModelParameters.ZAR'] = dict(MODEL_PARAMS)
    return document


def test_the_absence_note_names_the_factor_the_book_would_need(accrual_book):
    """What the leg says when the model is NOT pinned, and once the pair IS fitted.

    A spot model's parameters are named for the pair's NON-BASE token - the only leg the engine
    simulates and the only one a calibration can write - so `structures.spot_model` and
    `get_spot_model_params_factor` make the same lookup. Unfitted, the note names that factor and
    the verb that installs it; fitted, both orientations pin and there is no note.
    """
    tarf = structures.quote(copy.deepcopy(accrual_book), 'TargetRedemptionForward', dict(
        accrual_params(target=TARGET), notional_currency='USD'))
    note = leg(tarf, 'tarf')['note']

    assert 'LogVar2FJModelParameters.ZAR' in note, 'the factor looked up is unnamed'
    assert 'LogVar2FJModelParameters.USD' not in note, (
        'the base currency is a numeraire, never a rate - it can name no block')
    assert 'keyed off the pair' in note and '/book/model' in note, 'a note without a remedy'
    assert tarf['valuation_configuration'] is None, 'a model was pinned that cannot be resolved'

    # and on the fitted book BOTH orientations join: the TARF forced onto the base currency, and
    # the accumulator that crosses freely
    document = calibrated(accrual_book)
    joined = structures.quote(document, 'TargetRedemptionForward', dict(
        accrual_params(target=TARGET), notional_currency='USD'))
    assert leg(joined, 'tarf')['note'] is None, 'the pinned arm still carries a note'
    assert joined['valuation_configuration'] == {
        'FXTARFOptionDeal': {'SpotModel': 'LogVar2FJ'}}

    accumulator = structures.quote(document, 'Accumulator', dict(
        accrual_params(knockout=SPOT * 1.10), notional=NOTIONAL, notional_currency='ZAR'))
    assert leg(accumulator, 'accumulator')['note'] is None
    assert accumulator['valuation_configuration'] == {
        'FXAccumulatorOptionDeal': {'SpotModel': 'LogVar2FJ'}}


def test_the_token_rule_answers_the_same_token_in_either_spelling():
    """One rule, four callers, two dialects. The engine, discovery and the calibration speak
    `check_rate_name` TUPLES; the runner speaks flat names off the document. The rule compares on
    the checked form, so a mixed call cannot answer `underlying` by falling through an equality that
    was never going to hold - which is the PRE-RULE token wearing the fix's clothes.
    """
    spellings = (('USD', 'ZAR', 'USD'), (('USD',), ('ZAR',), ('USD',)),
                 ('USD', 'ZAR', ('USD',)), (('USD',), ('ZAR',), 'USD'))
    for underlying, currency, base in spellings:
        token = utils.spot_model_currency(underlying, currency, base)
        assert utils.check_rate_name(token) == ('ZAR',), (underlying, currency, base, token)

    # a CROSS is keyed on the two CURRENCIES - the later priced in the earlier - so one pair has
    # one law whichever way the deal is written and whichever way a desk spells its surface
    for underlying, currency in (('EUR', 'GBP'), ('GBP', 'EUR')):
        assert utils.spot_model_currency(underlying, currency, 'USD') == 'GBP.EUR'
        assert utils.spot_model_currency(
            (underlying,), (currency,), ('USD',)) == ('GBP', 'EUR')
    # and a name that is not one currency has no pair to be a leg of
    with pytest.raises(ValueError, match='ONE rate of a pair'):
        utils.spot_model_currency(('EUR', 'BASIS'), ('GBP',), ('USD',))


def test_a_book_that_declares_no_base_currency_refuses_instead_of_pinning(accrual_book):
    """The base is the OTHER half of the token rule, and an unknown one may not be guessed.

    A quote reads the base off the EXPLICIT block, so a book keeping its `System Parameters` behind
    a `MarketDataFile` answers nothing here. Falling back to `Underlying_Currency` is the PRE-RULE
    read: where the underlying is not the base the two agree by luck, and where it IS the base the
    quote PINS a model the engine looks up under the other name - a dependency-loop raise, a skipped
    deal, the trade marked at nothing on a job reporting success. So it refuses, naming the missing
    declaration, and pins nothing.
    """
    document = calibrated(accrual_book)
    market = document['Calc']['MergeMarketData']['ExplicitMarketData']
    del market['System Parameters']

    with pytest.raises(ValueError) as refusal:
        structures.quote(document, 'Accumulator', dict(
            accrual_params(knockout=SPOT * 1.10), notional=NOTIONAL, notional_currency='ZAR'))
    assert 'Base_Currency' in str(refusal.value), 'a refusal that does not name what is missing'
    assert 'NON-BASE' in str(refusal.value), 'a refusal that does not name the rule it could not run'
    assert 'Valuation Configuration' not in market, 'a model was pinned off a guessed base'


def test_a_tarf_on_a_fitted_pair_stops_riding_gbm(accrual_book):
    """THE JOINING GATE, reading the defect's own signature.

    A USDZAR TARF is forced onto the pair's BASE currency, so it used to look up a factor named for
    a NUMERAIRE, which can name no block - and rode GBM however many times the pair was calibrated.
    That showed as BIT-IDENTITY: the solved strike under the declared model and under none were the
    same float, separation exactly 0.0. MEASURED under `MODEL_PARAMS`: 16.832891643626950 against
    the same GBM 16.774620757621133, 0.347% apart - 139 times the solve floor of 2.5e-5 at this
    path count, and the band below is forty times that floor.

    THE ACCUMULATOR'S ORIENTATION ALREADY JOINED: its notional is the rand, so its underlying was
    already the non-base token. Its strike is pinned to the digit at 17.390492998425863, so the
    arm that never had the defect cannot lose the fit either.
    """
    gbm = structures.quote(copy.deepcopy(accrual_book), 'TargetRedemptionForward', dict(
        accrual_params(target=TARGET), notional_currency='USD'))
    modelled = structures.quote(calibrated(accrual_book), 'TargetRedemptionForward', dict(
        accrual_params(target=TARGET), notional_currency='USD'))

    lognormal = leg(gbm, 'tarf')['strike_market']
    garch = leg(modelled, 'tarf')['strike_market']
    assert lognormal != garch, 'the declared model priced the lognormal, to the bit'
    assert abs(garch / lognormal - 1.0) > 1e-3, (garch, lognormal)
    assert abs(gbm['net']) <= SOLVE_TOLERANCE and abs(modelled['net']) <= SOLVE_TOLERANCE

    accumulator = structures.quote(calibrated(accrual_book), 'Accumulator', dict(
        accrual_params(knockout=SPOT * 1.10), notional=NOTIONAL, notional_currency='ZAR'))
    # the band is the accumulation order's, not the claim's - what a lost fit costs this arm is
    # 5.1e-2, and the strike it solves under the factor is pinned here to the digit
    assert leg(accumulator, 'accumulator')['strike_market'] == pytest.approx(
        17.390492998425863, rel=1e-9), 'the orientation that already joined moved'


def test_the_accumulator_solves_one_strike_from_either_axis_under_the_model(accrual_book):
    """RECIPROCAL CONSISTENCY under the fitted law - the axis gate's shape, one model on.

    The rand orientation rides the fit as written; the dollar orientation is on the RECIPROCAL of
    the fitted axis and settles in the other currency, so the law is carried to that numeraire
    (`utils.LogVar2FJ.walk`'s own measure change). Uncarried - walking the fitted law and reading `1/s` -
    the two solve 3.7e-3 apart and the gap does NOT close with the path count: a Siegel drift, not
    noise.

    MEASURED on the FX gate's book: 3.1e-4 apart at 16,384 paths against the lognormal's 2.5e-5 -
    the shocks' estimator error rather than the numeraire. The band is 5e-4.
    """
    document = calibrated(accrual_book)
    both_ways = {'pair': PAIR, 'expiry': EXPIRY, 'fixing_frequency': FIXING_FREQUENCY,
                 'buy_currency': 'USD', 'knockout': SPOT * 1.10}
    in_rand = structures.quote(copy.deepcopy(document), 'Accumulator', dict(
        both_ways, notional=NOTIONAL, notional_currency='ZAR'))
    strike = leg(in_rand, 'accumulator')['strike_market']
    in_dollars = structures.quote(copy.deepcopy(document), 'Accumulator', dict(
        both_ways, notional=NOTIONAL / strike, notional_currency='USD'))

    assert leg(in_rand, 'accumulator')['note'] is None
    assert leg(in_dollars, 'accumulator')['note'] is None
    assert abs(in_rand['net']) <= SOLVE_TOLERANCE and abs(in_dollars['net']) <= SOLVE_TOLERANCE
    assert leg(in_dollars, 'accumulator')['strike_market'] == pytest.approx(
        strike, rel=MODEL_AXIS_TOLERANCE), 'the two orientations solved different strikes under one law'


def test_a_composed_tarf_carries_an_exposure_profile(tmp_path):
    """The CMC bar: a quoted TARF, booked, run as an exposure simulation - the failure mode a base
    valuation cannot see.

    A deal the engine cannot resolve is SKIPPED. On a base valuation that is a missing `mtm` row,
    which the zero-cost gate reads; on a Credit Monte Carlo it does not show up at all - the profile
    is still a frame of the right shape, and a book of one skipped deal is a floor of zeros that
    looks like a trade deep out of the money. So: more than one reporting row, every value finite,
    and DISPERSION across each row's scenarios, which zero cannot have.

    The market is `test_fx_tarf_json`'s, with the FX rate given `GBMAssetPriceModel` (the only edit)
    because a Credit Monte Carlo simulates what a base valuation only discounts.
    """
    import numpy as np

    template = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'fixtures', 'fx_tarf_job.json')
    with open(template) as source:
        document = json.load(source)
    document['Calc']['Deals']['Deals']['Children'] = []
    document['Calc']['Calculation']['MCMC_Simulations'] = ACCRUAL_SIMS
    factors = document['Calc']['MergeMarketData']['ExplicitMarketData']['Price Factors']

    outcome = structures.quote(document, 'TargetRedemptionForward', {
        'pair': 'EURUSD', 'expiry': '6M', 'notional': 1_000_000.0, 'notional_currency': 'EUR',
        'buy_currency': 'EUR', 'fixing_frequency': FIXING_FREQUENCY, 'target': 0.10})
    assert abs(outcome['net']) <= SOLVE_TOLERANCE, outcome['net']
    assert leg(outcome, 'tarf')['strike_market'] < factors['FxRate.EUR']['Spot'] * 1.05

    run = copy.deepcopy(document)
    run['Calc']['Deals']['Deals']['Children'] = [structures.book_node(outcome['deal'])]
    # the grid runs PAST the last settlement: a horizon stopping inside the strip reports one mtm
    # row more than its own report index and refuses on the shape
    run['Calc']['Calculation'] = {
        'Object': 'CreditMonteCarlo', 'Base_Date': document['Calc']['Calculation']['Base_Date'],
        'Currency': 'USD', 'Time_grid': '0d 7m(1m)', 'Batch_Size': 512, 'Simulation_Batches': 2,
        'Random_Seed': 1, 'MCMC_Simulations': 512, 'Deflation_Interest_Rate': 'USD'}
    market = run['Calc']['MergeMarketData']['ExplicitMarketData']
    market['Price Models'] = {'GBMAssetPriceModel.EUR': {
        'Vol': factors['FXVol.EUR.USD']['Surface']['.Curve']['data'][0][2], 'Drift': 0.0}}
    market['Model Configuration'] = {'.ModelParams': {
        'modeldefaults': {'FxRate': 'GBMAssetPriceModel'}, 'modelfilters': {}}}

    path = os.path.join(str(tmp_path), 'tarf_cmc.json')
    with open(path, 'w') as target:
        json.dump(run, target, default=str)
    context = derivus.Context()
    context.load_json(path)
    _, out = context.run_job()

    profile = np.asarray(out['Results']['mtm'].values, dtype=float)
    spread = profile.std(axis=1)
    assert profile.shape[0] > 1, 'one reporting row is not a profile'
    assert np.isfinite(profile).all(), 'the exposure profile carries a non-finite row'
    assert (spread > 0.0).sum() > 1, (
        'a profile with no dispersion across scenarios is a deal the run skipped')
    assert spread[-1] == 0.0, (
        'the grid deliberately outlives the strip, so the last row has nothing left to be worth')


# --------------------------------------------------------------------------------------------
# the variations: one structure, several BOOKINGS, and the rule that selects between them
# --------------------------------------------------------------------------------------------

#: The engine axis, written out here rather than read from the runner. A notional in the pair's
#: QUOTE currency makes each leg an option on that currency, so a market Call is an engine Put and
#: a barrier's DIRECTION turns over with the level it sits on, while In and Out never move.
CROSSED_TYPE = {'Call': 'Put', 'Put': 'Call'}
CROSSED_BARRIER = {'Up_And_In': 'Down_And_In', 'Down_And_In': 'Up_And_In',
                   'Up_And_Out': 'Down_And_Out', 'Down_And_Out': 'Up_And_Out'}

#: The coordinate the recipe MOVES, which no ticket names and no table can state.
SOLVED = 'solved'

#: What each variation BOOKS, on the PAIR's own axis, written out leg by leg rather than derived:
#: `(role, deal type, sense, barrier direction, the CLIENT's side, the parameter the strike is
#: struck at, the parameter the barrier sits at)`. The owner's four forward-extra bookings are the
#: first rows - exporter paper is a bought put plus a sold up-and-in call at the one protected
#: rate, importer paper a bought call plus a sold down-and-in put at the one capped rate.
BOOKINGS = {
    ('ForwardExtra', 'floor'): (
        ('protection', 'FXOptionDeal', 'Put', None, 'Buy', 'floor', None),
        ('reversion', 'FXBarrierOption', 'Call', 'Up_And_In', 'Sell', 'floor', SOLVED)),
    ('ForwardExtra', 'cap'): (
        ('protection', 'FXOptionDeal', 'Call', None, 'Buy', 'cap', None),
        ('reversion', 'FXBarrierOption', 'Put', 'Down_And_In', 'Sell', 'cap', SOLVED)),
    ('ZeroCostCollar', 'floor'): (
        ('protection', 'FXOptionDeal', 'Put', None, 'Buy', 'floor', None),
        ('financing', 'FXOptionDeal', 'Call', None, 'Sell', SOLVED, None)),
    ('ZeroCostCollar', 'cap'): (
        ('protection', 'FXOptionDeal', 'Call', None, 'Buy', 'cap', None),
        ('financing', 'FXOptionDeal', 'Put', None, 'Sell', SOLVED, None)),
    ('Seagull', 'floor'): (
        ('protection', 'FXOptionDeal', 'Put', None, 'Buy', 'floor', None),
        ('participation', 'FXOptionDeal', 'Put', None, 'Sell', 'lower_floor', None),
        ('financing', 'FXOptionDeal', 'Call', None, 'Sell', SOLVED, None)),
    ('Seagull', 'cap'): (
        ('protection', 'FXOptionDeal', 'Call', None, 'Buy', 'cap', None),
        ('participation', 'FXOptionDeal', 'Call', None, 'Sell', 'upper_cap', None),
        ('financing', 'FXOptionDeal', 'Put', None, 'Sell', SOLVED, None)),
    ('TargetRedemptionForward', 'buy'): (
        ('tarf', 'FXTARFOptionDeal', 'Call', None, 'Buy', SOLVED, None),),
    ('TargetRedemptionForward', 'sell'): (
        ('tarf', 'FXTARFOptionDeal', 'Put', None, 'Buy', SOLVED, None),),
    ('Accumulator', 'buy'): (
        ('accumulator', 'FXAccumulatorOptionDeal', 'Call', 'Up_And_Out', 'Buy', SOLVED,
         'knockout'),),
    ('Accumulator', 'sell'): (
        ('accumulator', 'FXAccumulatorOptionDeal', 'Put', 'Down_And_Out', 'Buy', SOLVED,
         'knockout'),)}

#: What each variation is quoted on, in the market's own terms: the LEVEL a client names where a
#: level tells the two forms apart, the DIRECTION where it does not. A seller's knock-out sits
#: below the spot as a buyer's sits above it, the strip cancelling on the move that would pay them.
ASKS = {
    ('ForwardExtra', 'floor'): {'floor': SPOT * 0.97},
    ('ForwardExtra', 'cap'): {'cap': SPOT * 1.03},
    ('ZeroCostCollar', 'floor'): {'floor': SPOT * 0.95},
    ('ZeroCostCollar', 'cap'): {'cap': SPOT * 1.05},
    ('Seagull', 'floor'): {'floor': SPOT * 0.98, 'lower_floor': SPOT * 0.90},
    ('Seagull', 'cap'): {'cap': SPOT * 1.02, 'upper_cap': SPOT * 1.10},
    ('TargetRedemptionForward', 'buy'): {'buy_currency': 'USD', 'target': TARGET,
                                         'fixing_frequency': FIXING_FREQUENCY},
    ('TargetRedemptionForward', 'sell'): {'sell_currency': 'USD', 'target': TARGET,
                                          'fixing_frequency': FIXING_FREQUENCY},
    ('Accumulator', 'buy'): {'buy_currency': 'USD', 'knockout': SPOT * 1.10,
                             'fixing_frequency': FIXING_FREQUENCY},
    ('Accumulator', 'sell'): {'sell_currency': 'USD', 'knockout': SPOT * 0.90,
                              'fixing_frequency': FIXING_FREQUENCY}}

#: Every variation from each side of the pair it takes. A TARF is quoted on the pair's BASE alone,
#: a sum of differences having no reading on the reciprocal, and refuses the other side by name.
SIDES = [(name, word, currency) for name, word in ASKS for currency in ('USD', 'ZAR')
         if (name, currency) != ('TargetRedemptionForward', 'ZAR')]

#: The structures whose legs are struck at ONE rate, which is what makes a notional on either side
#: of the pair the same trade: the forward extra's two legs share the protected rate and a strip is
#: one leg. A collar and a seagull are NOT - they solve across TWO strikes, so each leg's notional
#: divides by a different level and the two orientations are genuinely different trades.
ONE_RATE = ('ForwardExtra', 'Accumulator')

#: Which way round each variation is dealt, in the client's own two cashflows. The outcome must say
#: this, and it must be read off the VARIATION rather than off the side the notional is quoted in -
#: the same words for every structure, since 'floor' and 'sell' are one client and 'cap' and 'buy'
#: the other.
CLIENT = {'floor': {'buys': 'ZAR', 'sells': 'USD'}, 'sell': {'buys': 'ZAR', 'sells': 'USD'},
          'cap': {'buys': 'USD', 'sells': 'ZAR'}, 'buy': {'buys': 'USD', 'sells': 'ZAR'}}


@pytest.fixture(scope='module')
def quoted(accrual_book):
    """Every variation of every structure that declares them, quoted once from each side of the
    pair it takes - one sweep the gates below read, rather than a solve per claim."""
    return {(name, word, currency): structures.quote(
        accrual_book, name, dict(params(notional_currency=currency), **ASKS[(name, word)]))
        for name, word, currency in SIDES}


def solved_market(outcome):
    """The one coordinate the recipe moved, in the market's own terms."""
    return next(row['strike_market'] if 'Strike_Price' in row['solved'] else row['barrier_market']
                for row in outcome['legs'] if row['solved'])


#: Every form the two-way is charged on: the ten variations, plus the two structures dealt ONE way,
#: whose recipes solve NOTHING - there is no coordinate for the charge to move, so the client pays
#: it as premium instead.
SPREAD_ASKS = ASKS | {('Straddle', None): {'strike': SPOT},
                      ('Strangle', None): {'floor': SPOT * 0.95, 'cap': SPOT * 1.05}}

#: Which forms a FITTED book can tell apart: a spot model is pinned per DEAL TYPE and only the
#: accrual deals declare one, so a vanilla quotes identically on both books and is swept once.
MODELLED = ('TargetRedemptionForward', 'Accumulator')

#: What the vega-signed side charges each strip on the gate's own two-way book at a 1m USD ticket,
#: MEASURED: `sum |vega| x half` over the six quoted pillars, at the mid solution. The quote's own
#: solve then moves the coordinate under that reading, so the band is 5%; the four land at 0.5% or
#: better. Signed by `Buy_Sell` instead, every one of them comes back NEGATIVE.
STRIP_CHARGE = {('TargetRedemptionForward', 'buy'): 6657.26,
                ('TargetRedemptionForward', 'sell'): 7361.99,
                ('Accumulator', 'buy'): 22405.15,
                ('Accumulator', 'sell'): 25294.60}


@pytest.fixture(scope='module')
def charged(accrual_book):
    """Every form of every structure quoted at the MID and at a two-way, on the lognormal book and
    on the fitted one - one sweep the four charge gates read, rather than a solve per claim.

    A 1m USD ticket throughout: the side a TARF takes, and the one the charges above were measured
    at. The fitted book is swept for the accrual deals alone, a vanilla pinning no spot model.
    """
    fitted = calibrated(accrual_book)
    books = {'surface': two_way(accrual_book), 'surface mid': accrual_book,
             'model': two_way(fitted), 'model mid': fitted}
    return {(name, word, label): structures.quote(
        copy.deepcopy(document), name,
        dict(params(notional_currency='USD'), **SPREAD_ASKS[(name, word)]))
        for name, word in SPREAD_ASKS for label, document in books.items()
        if name in MODELLED or not label.startswith('model')}


@pytest.mark.parametrize('name,word', sorted(SPREAD_ASKS, key=str))
def test_every_form_charges_a_non_negative_edge_that_is_its_legs_own(charged, name, word):
    """THE RULING as an identity over every form of every structure, on a lognormal book and on a
    fitted one: the desk's edge IS the charge it levied, so it has no other sign.

    Three claims per form. The edge is NON-NEGATIVE, a sum of `|vega| x half` having nowhere else
    to go. It is the sum of the legs' own charges, so a quote reports exactly the spread it took and
    no other. And it lands on the coordinate the recipe SOLVES: a zero-cost structure is quoted at
    nothing and MARKS at minus the edge, while a structure that solves nothing has no coordinate to
    move and the client pays the charge as premium on top of the mid.

    Signing the charge by `Buy_Sell` instead - the rule this replaces - turns all four strips
    negative here, the desk paying a client 6.6k to 25.3k on a 1m ticket to take the trade.
    """
    solves = [step for step in structures.structure_named(name).recipe
              if isinstance(step, structures.Solve)]
    for label in ('surface', 'model'):
        outcome = charged.get((name, word, label))
        if outcome is None:
            continue
        levied = sum(row['spread_charge'] for row in outcome['legs']
                     if row['spread_charge'] is not None)
        assert outcome['edge'] >= 0.0, (label, outcome['edge'])
        assert outcome['edge'] == pytest.approx(levied, rel=1e-12), label
        assert outcome['net'] == outcome['net_mid'] + outcome['edge'], label
        if solves:
            assert abs(outcome['net']) <= SOLVE_TOLERANCE, (label, outcome['net'])
            assert outcome['net_mid'] == pytest.approx(-outcome['edge'], abs=SOLVE_TOLERANCE)
        else:
            assert outcome['net'] > outcome['net_mid'] > 0.0, label


@pytest.mark.parametrize('name,word', sorted(STRIP_CHARGE, key=str))
def test_a_strip_pays_the_spread_rather_than_being_paid_it(charged, name, word):
    """A geared strip is SHORT vol at every ATM and butterfly pillar although its one leg is booked
    `Buy`, so a side taken from the LABEL quoted the client a negative edge - the desk paying 6.5k
    to 25.5k on a 1m ticket for the privilege of the trade. Charged per pillar on the ABSOLUTE vega
    it is the same magnitude back, with the sign a desk needs.

    And it lands the right way round. The client buying the base accrues as the pair rises, so a
    higher strike is worse for them and the solved one comes UP; the client selling it accrues as
    the pair falls and theirs comes DOWN. Dropping the absolute value keeps the magnitude and
    reverses both.
    """
    outcome, at_mid = charged[(name, word, 'surface')], charged[(name, word, 'surface mid')]
    solved, was = solved_market(outcome), solved_market(at_mid)

    assert outcome['edge'] > 0.0, outcome['edge']
    assert outcome['edge'] == pytest.approx(STRIP_CHARGE[(name, word)], rel=0.05)
    assert abs(solved / was - 1.0) > AXIS_TOLERANCE, 'the charge never reached the strike'
    assert (solved > was) == (word == 'buy'), (
        'the solved strike moved TOWARD the client: {} against a mid {}'.format(solved, was))


def test_a_puts_risk_reversal_is_charged_on_its_own_side(charged):
    """Not even a vanilla is single-signed, which is the second thing one side per LEG got wrong.

    A risk reversal follows the WING rather than the label: every leg holding a PUT reads a
    `dV/d(RR)` opposite in sign to its own ATM and butterfly vega - the collar's bought put on the
    floor and its sold put on the cap alike - while the legs holding a call agree with theirs.
    Widening both wings on one side therefore charged about 5% of a collar's spread the wrong way.

    Charged per pillar the sign does not matter: the risk reversal's row costs the client something
    POSITIVE on all four legs, each dealt on the side its own risk puts it.
    """
    puts = {('floor', 'protection'), ('cap', 'financing')}
    for word in ('floor', 'cap'):
        for row in charged[('ZeroCostCollar', word, 'surface')]['legs']:
            pillars = {cell['pillar']: cell for cell in row['spread']}
            atm, reversal = pillars['ATM 1'], pillars['RR 0.25 1']
            assert reversal['cost'] > 0.0, (word, row['role'])
            opposite = atm['vega'] * reversal['vega'] < 0.0
            assert opposite == ((word, row['role']) in puts), (
                '{} {} reads its risk reversal {} its ATM vega'.format(
                    word, row['role'], 'against' if opposite else 'with'))


@pytest.mark.parametrize('name,word', sorted(STRIP_CHARGE, key=str))
def test_a_model_priced_strip_is_charged_off_its_lognormal_reading(charged, name, word):
    """A strip walking a FITTED law never reads the written FX surface, so it publishes no quote
    sensitivity at all and the two-way had nothing to charge against: the quote solved the MID
    strike, captured exactly nothing, and reported a vol spread all the same.

    The charge comes off the LOGNORMAL reading of the same leg at the same terms - the vega a desk
    would deal in the quotes it actually trades - and `spread_source` says so, against a `surface`
    on the same book with no fit installed. Skipping that second run leaves the edge at 0.0 and the
    solved strike on the mid, which is what this refuses.
    """
    outcome, at_mid = charged[(name, word, 'model')], charged[(name, word, 'model mid')]
    row = outcome['legs'][0]
    solved, was = solved_market(outcome), solved_market(at_mid)

    assert row['spread_source'] == 'lognormal reading', row['spread_source']
    assert charged[(name, word, 'surface')]['legs'][0]['spread_source'] == 'surface', (
        'the same book with no fit read its vega somewhere else')
    assert row['note'] is None, 'the leg is priced under the fit and must not say otherwise'
    assert outcome['edge'] > 0.0 and row['spread_charge'] == pytest.approx(
        outcome['edge'], rel=1e-12)
    assert abs(solved / was - 1.0) > MODEL_AXIS_TOLERANCE, (
        'a model-priced strip still quotes the mid strike')
    assert (solved > was) == (word == 'buy'), (solved, was)


@pytest.mark.parametrize('name,word,currency', SIDES)
def test_every_variation_books_the_deals_it_declares(quoted, name, word, currency):
    """THE BOOKING, leg by leg, against a table written out by hand - because two variations can
    price alike and book wrong, and a price alone proves nothing about what was dealt.

    Each leg's type, sense, barrier direction, the CLIENT's side and the currency it is an option
    on are held to the table; the strike and the barrier are read back in MARKET terms and must be
    the rate the ticket named, the engine block carrying its reciprocal where the notional is the
    pair's quote currency. A reflection that flipped `Buy_Sell` too, skipped `Barrier_Type` or
    turned In into Out lands somewhere else on every row of it.

    And the MIRROR is one flip and nothing else: the bank's position is the client's paper with
    every side turned over, so a leg that moved anything more would book a trade nobody priced.
    """
    outcome = quoted[(name, word, currency)]
    inverted = currency == 'ZAR'
    booked = [child['Instrument']['.Deal'] for child in outcome['deal']['Children']]
    assert len(booked) == len(BOOKINGS[(name, word)])
    # the outcome's account of which way the trade went, on every one of these - read off the
    # variation that was priced, never off the side the notional happens to be quoted in
    assert outcome['client'] == CLIENT[word], (name, word, currency)

    for block, (role, kind, sense, barrier, side, struck, level) in zip(
            booked, BOOKINGS[(name, word)]):
        assert block['Object'] == kind
        assert block['Option_Type'] == (CROSSED_TYPE[sense] if inverted else sense)
        assert block.get('Barrier_Type') == (
            barrier if barrier is None or not inverted else CROSSED_BARRIER[barrier])
        assert (block['Buy_Sell'], block['Underlying_Currency']) == (side, currency)
        row = leg(outcome, role)
        assert row['strike_market'] == pytest.approx(
            1.0 / block['Strike_Price'] if inverted else block['Strike_Price'], rel=1e-12)
        for reading, stated in (('strike_market', struck), ('barrier_market', level)):
            if stated is None:
                assert row[reading] is None, '{} reports a {} it has not got'.format(role, reading)
            elif stated is not SOLVED:
                assert row[reading] == pytest.approx(outcome['params'][stated], rel=1e-12), (
                    '{} is not struck at the {} the client named'.format(role, stated))

    for block, flipped in zip(
            booked, [child['Instrument']['.Deal']
                     for child in structures.mirror(outcome['deal'])['Children']]):
        assert flipped['Buy_Sell'] == {'Buy': 'Sell', 'Sell': 'Buy'}[block['Buy_Sell']]
        assert {key: value for key, value in flipped.items() if key != 'Buy_Sell'} == {
            key: value for key, value in block.items() if key != 'Buy_Sell'}, (
            'the mirror moved more than a side')


def test_each_variation_is_the_structure_it_says_it_is(quoted):
    """What each variation is CALLED, held to what it does - the half a booking table cannot say.

    Every one of them is zero cost, and the outcome names the variation it priced. The collar
    names one level and SOLVES the other on the far side of the forward: an exporter's floor buys
    a cap above it, an importer's cap buys a floor below it. A strip's client buying the base
    accrues as the pair rises and strikes BELOW the forward; the one selling it accrues as the
    pair falls and strikes ABOVE - which is not a tautology, the forward being the spot here and
    the geared sold leg outweighing the bought one at a strike of it.

    And where a structure's legs are struck at ONE rate, the two sides of the pair are one trade
    and must solve one coordinate, travelling opposite paths through the runner to reach it.
    """
    for (name, word, currency), outcome in quoted.items():
        assert abs(outcome['net']) <= SOLVE_TOLERANCE, (name, word, currency, outcome['net'])
        assert outcome['variation'] == word, (name, word, currency)
        # a SELECTOR chooses a form rather than filling a leg, so it is never defaulted into the
        # parameters the quote reports, files and hashes: exactly what the ticket stated is there
        assert SELECTOR_KEYS.intersection(outcome['params']) == SELECTOR_KEYS.intersection(
            ASKS[(name, word)]), (name, word, currency, sorted(outcome['params']))

    assert ASKS[('ZeroCostCollar', 'floor')]['floor'] < SPOT < solved_market(
        quoted[('ZeroCostCollar', 'floor', 'USD')]), 'the floor variation solves no cap above it'
    assert solved_market(quoted[('ZeroCostCollar', 'cap', 'USD')]) < SPOT < ASKS[
        ('ZeroCostCollar', 'cap')]['cap'], 'the cap variation solves no floor below it'

    for name in ('TargetRedemptionForward', 'Accumulator'):
        assert solved_market(quoted[(name, 'buy', 'USD')]) < SPOT, (
            '{}: a client buying the base is not struck better than the forward'.format(name))
        assert solved_market(quoted[(name, 'sell', 'USD')]) > SPOT, (
            '{}: a client selling the base is not struck above the forward'.format(name))

    for name, word, currency in SIDES:
        if currency == 'ZAR' and name in ONE_RATE:
            assert solved_market(quoted[(name, word, 'ZAR')]) == pytest.approx(
                solved_market(quoted[(name, word, 'USD')]),
                rel=AXIS_TOLERANCE if name == 'Accumulator' else 1e-6), (
                '{}.{} solved two coordinates for one trade'.format(name, word))


def written_out(variation):
    """A variation as plain data, for comparing against one written out by hand."""
    return (variation.buys, [(f.key, f.description) for f in variation.fields],
            [(leg.role, leg.deal_type, leg.pinned, leg.slots) for leg in variation.legs])


def test_a_reflected_variation_is_the_legs_written_out_by_hand():
    """`reflected` derives the mirror image, and this is that image spelled out instead - field for
    field, so a rename that missed the slots or a flip that reached `Buy_Sell` shows here rather
    than as a price.

    The three shapes it has to get right: a level renamed and named again in its own prose, TWO
    levels renamed at once without `lower_floor` reading as a `floor` inside it, and a strip with
    no level of its own at all, where only the sense and the knock-out direction turn over.
    """
    assert written_out(structures.ForwardExtra.variations['cap']) == (
        'base',
        [('cap', 'The cap: the rate the client is protected at, and the forward the structure '
                 'reverts to once the barrier trades. ' + structures.MARKET_STRIKE)],
        [('protection', 'FXOptionDeal',
          {'Option_Style': 'European', 'Option_Type': 'Call', 'Buy_Sell': 'Buy'},
          {'Strike_Price': 'cap'}),
         ('reversion', 'FXBarrierOption',
          {'Option_Type': 'Put', 'Buy_Sell': 'Sell', 'Barrier_Type': 'Down_And_In'},
          {'Strike_Price': 'cap'})])

    assert written_out(structures.Seagull.variations['cap']) == (
        'base',
        [('cap', 'The protected level; the floor is solved against it. '
                 + structures.MARKET_STRIKE),
         ('upper_cap', 'Where protection stops, sold back against the cap. '
                       + structures.MARKET_STRIKE)],
        [('protection', 'FXOptionDeal',
          {'Option_Style': 'European', 'Option_Type': 'Call', 'Buy_Sell': 'Buy'},
          {'Strike_Price': 'cap'}),
         ('participation', 'FXOptionDeal',
          {'Option_Style': 'European', 'Option_Type': 'Call', 'Buy_Sell': 'Sell'},
          {'Strike_Price': 'upper_cap'}),
         ('financing', 'FXOptionDeal',
          {'Option_Style': 'European', 'Option_Type': 'Put', 'Buy_Sell': 'Sell'}, {})])

    assert written_out(structures.Accumulator.variations['sell']) == (
        'quote', [],
        [('accumulator', 'FXAccumulatorOptionDeal',
          {'Option_Type': 'Put', 'Buy_Sell': 'Buy', 'Barrier_Type': 'Down_And_Out'},
          {'Barrier_Price': 'knockout'})])


def test_the_selection_rule_answers_every_way_a_ticket_can_state_it(book, accrual_book):
    """ONE rule, every arm of it, and the refusals that are the point of having a rule at all.

    A LEVEL WORD ALONE selects where the forms differ in one: "a forward extra, cap 16.90" is the
    importer's, and nobody had to say which way round it was dealt. A DIRECTION ALONE selects
    where they do not: a strip is the same shape either way, so it is the only thing that can.
    Both together must AGREE. A single-form structure declares none of this and answers nothing.

    Then the four refusals. A cap from a client selling the base names BOTH facts, exactly as the
    hand-written validation it replaces did. An undirected strip names what to state rather than
    picking one. A currency outside the pair is not a direction on it, and buying and selling the
    same currency is not a trade.
    """
    for structure, stated, word in (
            (structures.ForwardExtra, {'floor': SPOT * 0.97}, 'floor'),
            (structures.ForwardExtra, {'cap': SPOT * 1.03}, 'cap'),
            (structures.Accumulator, {'buy_currency': 'USD'}, 'buy'),
            (structures.Accumulator, {'sell_currency': 'USD'}, 'sell'),
            (structures.Seagull, {'sell_currency': 'USD', 'buy_currency': 'ZAR',
                                  'floor': SPOT * 0.98}, 'floor'),
            (structures.ZeroCostCollar, {'buy_currency': 'USD', 'cap': SPOT * 1.05}, 'cap')):
        assert structures.variation_for(structure, params(**stated))[0] == word, (structure, stated)
    assert structures.variation_for(structures.Straddle, params(strike=SPOT)) == (None, None), (
        'a structure dealt one way has nothing to select')

    # a direction stated in the shape a ticket arrives in - the pair is read the same way
    assert structures.variation_for(
        structures.Accumulator, params(buy_currency=' usd '))[0] == 'buy'

    with pytest.raises(ValueError) as contradicted:
        structures.quote(book, 'ForwardExtra', forward_extra_params(cap=SPOT * 1.03))
    assert 'selling USD and buying ZAR' in str(contradicted.value)
    assert 'states floor, not cap' in str(contradicted.value), str(contradicted.value)
    assert "CLIENT's own side of the trade" in str(contradicted.value), (
        'a model that read "sell 10m USD" as the direction is not told which reading to fix')

    # the refusal names the DIFFERENCE, never the level the dealt variation does state: a desk
    # told "deals cap, which states cap, not cap, floor" is being told a cap is not a cap
    with pytest.raises(ValueError) as both_and_a_side:
        structures.quote(book, 'ForwardExtra', params(
            buy_currency='USD', cap=SPOT * 1.03, floor=SPOT * 0.97))
    assert 'deals cap, which states cap, not floor' in str(both_and_a_side.value), (
        str(both_and_a_side.value))

    with pytest.raises(ValueError) as undirected:
        structures.quote(accrual_book, 'TargetRedemptionForward', dict(
            params(fixing_frequency=FIXING_FREQUENCY, target=TARGET), notional_currency='USD'))
    assert 'state which currency the client buys, USD or ZAR' in str(undirected.value)
    assert 'deals as buy or sell' in str(undirected.value)

    with pytest.raises(ValueError, match='not a side of USDZAR'):
        structures.variation_for(structures.Accumulator, params(buy_currency='JPY'))
    with pytest.raises(ValueError, match='cannot buy USD and sell USD'):
        structures.variation_for(structures.Accumulator,
                                 params(buy_currency='USD', sell_currency='USD'))
    with pytest.raises(ValueError, match='states cap and floor together'):
        structures.variation_for(structures.ForwardExtra,
                                 params(cap=SPOT * 1.03, floor=SPOT * 0.97))


def test_a_blank_is_not_a_statement(book):
    """ONE predicate for "stated", and it is truthiness - for selecting a variation and for
    completing the parameters alike.

    A front end round-tripping an unfilled Float sends 0.0, and the store publishes `"value": ""`
    for every required field, so the shape that arrives is a level with nothing in it. Read as a
    statement it makes a perfectly good exporter ticket a CONTRADICTION because the unused slot
    came back as a zero; read as a statement for selection and not for completion, a ticket is
    stated for one and missing for the other twenty lines apart.
    """
    assert structures.variation_for(
        structures.ForwardExtra, params(floor=SPOT * 0.97, cap=0.0))[0] == 'floor'
    quoted = structures.quote(book, 'ForwardExtra', params(floor=SPOT * 0.97, cap=0.0))
    assert quoted['variation'] == 'floor' and abs(quoted['net']) <= SOLVE_TOLERANCE

    # and it does not RIDE the ticket either: the slot a front end sent back is not part of what
    # was quoted, so it reaches neither the reported parameters, nor the id they are hashed into,
    # nor the pending file, nor the sheet - two asks differing only in it are one ticket
    assert 'cap' not in quoted['params'], quoted['params']
    assert quoted['params'] == structures.quote(
        book, 'ForwardExtra', params(floor=SPOT * 0.97))['params']

    # and a level stated as nothing is a level nobody named, in all three shapes it arrives in
    for blank in (0.0, '', None):
        with pytest.raises(ValueError) as refusal:
            structures.quote(book, 'ForwardExtra', params(cap=blank))
        assert 'state the cap or the floor' in str(refusal.value), (blank, str(refusal.value))


def test_what_a_variation_takes_is_required_and_what_no_form_takes_refuses(book):
    """`declared` is where a ticket is held to the form it selected, BEFORE a quote id is hashed,
    a netting set is checked or a leg is built.

    A variation's own parameters are required exactly as the shared ones are, so a cap variation
    selected by the DIRECTION alone with no cap refuses by name here rather than reaching a leg and
    coming back as "leg participation needs the 'upper_cap' parameter" from inside `materialize`.

    And a parameter no form of the structure takes refuses against the roster it does: a gearing
    misspelt is a strip quoted at the default gearing nobody agreed, and silence is what makes that
    invisible. A parameter belonging to the OTHER variation is the selection rule's to refuse, in
    its own words, which is a different sentence about a different mistake.
    """
    with pytest.raises(ValueError, match='Seagull states no upper_cap'):
        structures.declared(structures.Seagull, params(cap=SPOT * 1.02))
    with pytest.raises(ValueError, match='ForwardExtra states no cap'):
        structures.quote(book, 'ForwardExtra', params(buy_currency='USD'))

    with pytest.raises(ValueError) as unknown:
        structures.quote(book, 'Accumulator', dict(
            accrual_params(knockout=SPOT * 1.10), leverage_ratio=3.0))
    assert 'Accumulator takes no leverage_ratio' in str(unknown.value)
    assert 'leverage' in str(unknown.value), 'a refusal that never says what it does take'

    # and the roster is the whole vocabulary, the OTHER variation's level included - a client on a
    # floor ticket who meant the cap reads the word they wanted in the answer
    with pytest.raises(ValueError, match='cap'):
        structures.quote(book, 'ForwardExtra', dict(
            params(floor=SPOT * 0.97), collar_level=1.0))

    # a SELECTOR is exempt only where the structure declares one: a straddle is dealt one way and
    # there is nothing to select, so a direction on it is a parameter it does not take
    with pytest.raises(ValueError) as pointless:
        structures.quote(book, 'Strangle', dict(
            params(floor=SPOT * 0.95, cap=SPOT * 1.05), buy_currency='JPY'))
    assert 'Strangle takes no buy_currency' in str(pointless.value)
    assert 'floor' in str(pointless.value) and 'cap' in str(pointless.value)


def test_a_stated_level_on_the_dead_side_of_its_own_direction_refuses(accrual_book):
    """A knock-out behind the spot is a strip that dies at its first fixing, and it reads as a
    spectacular rate: a decumulator knocking out ABOVE the market solves 22.42 for a client selling
    dollars for a year, and an accumulator knocking out BELOW it solves 14.97. Both are numbers
    nobody can deal, quoted without a word.

    The check is the runner's and GENERIC - a level the CLIENT stated sits on the live side of its
    own `Barrier_Type`, read on the engine axis both are on by then - so it reaches the accumulator
    either way round, the rand orientation where the level and the direction have both crossed, and
    any later structure that lets a client state a barrier. A SOLVED level is the recipe's own and
    is bracketed on that side already, which is why the forward extra's barrier is not checked.

    It is checked against the SPOT, which the refusal says out loud: a strip observes at its
    fixings, so on a carried pair a level through today's spot can be live at the first of them.
    The reading fails SAFE - a live trade refused loudly rather than a dead one booked - and the
    sentence states what was compared rather than predicting what the strip would do.

    ON the spot is dead for BOTH directions, and the boundary is the half a one-sided comparison
    gets wrong: `pv_MC_Accumulator`'s `survives` is strict either way (`s < barrier` up, `s > barrier`
    down), so equality knocks out whichever way the barrier faces. This book's spot is exactly
    18.50, so both arms are reachable rather than theoretical.
    """
    for direction, dead in (('buy_currency', SPOT * 0.90), ('sell_currency', SPOT * 1.10),
                            ('buy_currency', SPOT), ('sell_currency', SPOT)):
        with pytest.raises(ValueError) as refusal:
            structures.quote(accrual_book, 'Accumulator', dict(
                params(fixing_frequency=FIXING_FREQUENCY, notional_currency='USD'),
                knockout=dead, **{direction: 'USD'}))
        said = str(refusal.value)
        assert '{} at {:g}'.format(
            'Up_And_Out' if direction == 'buy_currency' else 'Down_And_Out', dead) in said, said
        assert 'wrong side of the market SPOT {:g}'.format(SPOT) in said, said
        assert 'checked against' in said and 'through already' in said, said

    # the rand orientation reads the same market and SAYS so in market terms: the live level quotes
    # and the dead one refuses naming the pair's own numbers, never the engine's reciprocals
    live = structures.quote(accrual_book, 'Accumulator', dict(
        params(fixing_frequency=FIXING_FREQUENCY), sell_currency='USD', knockout=SPOT * 0.90))
    assert live['variation'] == 'sell' and abs(live['net']) <= SOLVE_TOLERANCE
    assert only_leg(live)['Barrier_Type'] == 'Up_And_Out', 'down on the pair is up on the rand'
    with pytest.raises(ValueError) as crossed:
        structures.quote(accrual_book, 'Accumulator', dict(
            params(fixing_frequency=FIXING_FREQUENCY), sell_currency='USD',
            knockout=SPOT * 1.10))
    assert 'Down_And_Out at {:g}'.format(SPOT * 1.10) in str(crossed.value), str(crossed.value)
    assert 'SPOT {:g}'.format(SPOT) in str(crossed.value), 'the refusal is not in market terms'
    assert '{:g}'.format(1.0 / SPOT) not in str(crossed.value), (
        'the refusal quotes the engine axis at a client who never sees it')


def test_a_selector_is_the_declared_field_itself_and_not_its_name():
    """`emit_structures` flags a SELECTOR by the declared field OBJECT, never by its key.

    A structure declaring its OWN parameter that merely shares the name is an ordinary parameter:
    published with its value and required as it was declared, so a front end asks a client for it.
    Flagged by name it would be stripped of its value and published as an OPTIONAL selector - a
    required parameter nobody is ever asked for - and the registry's own roster of the two names
    would agree with the mistake rather than catch it.

    The emitter is a pure function of the module it reads, so this hands it one declared here.
    """
    class Impostor:
        vernacular = 'impostor'
        fields = [structures.PAIR, structures.SELL_CURRENCY,
                  schema.F('buy_currency', 'Text', default=schema.REQUIRED,
                           description='its own parameter, not the one the runner selects on')]
        variations = {'only': structures.Variation('base', [], [
            structures.Leg('leg', 'FXOptionDeal', {'Option_Type': 'Call'})])}
        recipe = [structures.Price('leg')]

    emitted = schema.emit_structures(SimpleNamespace(
        SELECTORS=structures.SELECTORS, Impostor=Impostor))['Impostor']['fields']

    assert emitted['sell_currency']['selector'] == 'optional'
    assert 'value' not in emitted['sell_currency'], 'a selector has no value to offer'
    assert 'selector' not in emitted['buy_currency'], (
        'a parameter that only shares a selector\'s name was published as one')
    assert emitted['buy_currency']['required'] is True and emitted['buy_currency']['value'] == ''
