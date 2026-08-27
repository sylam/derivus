"""A structure is only sold once, and it either costs what it says it costs or it does not.

`derivus.structures` declares four FX structures and one runner. There is nothing to unit-test
about a declaration, so every gate here is a REAL quote: a real book document with a real
bootstrapped USDZAR vol surface, `quote()` run against it, and the answer checked against a
financial identity that would be false if any piece of the runner were wrong.

The book is `tests/test_service.py`'s own `job()` with its own `FACTORS`, its own canned Bloomberg
observations normalized by `derivus_bloomberg` and bootstrapped into the `FXVol.USD.ZAR` surface -
the same pipeline `/book/market` runs, done here as fixture authoring rather than over HTTP,
because the runner is a library verb over a document and has no service in it.

ONE deliberate change to that fixture, and it is the whole reason these numbers mean anything.
`test_service`'s world reads `FxRate.ZAR.Spot = 18.5`, and the engine's `FxRate` spot is a currency
in BASE-currency units, so that world says one rand buys 18.5 dollars. That is fine where it is
used - a cashflow's closed form does not care which way round the quote is - but a structure's
whole subject is the difference between the market's axis and the engine's, and a strike quoted
USDZAR 18.50 against it would be 340 standard deviations from the money. So `FxRate.ZAR.Spot` is
`1/SPOT` here: the same number, read the way the market quotes it, and USDZAR 18.50 is the money.

Both curves are flat at the same rate, so the FX forward IS the spot - and that is asserted rather
than assumed, by the straddle's two wings pricing equal at it. The collar gate leans on it to say
the solved cap sits above the forward.
"""
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest

import derivus
from derivus import schema, structures
from test_service import BASE, FACTORS, SPOT, dump, fx_vol_quotes, job

#: The client's numbers, in the market's own terms - USDZAR, a rand notional, a one-year tenor.
PAIR = 'USDZAR'
EXPIRY = '1Y'
NOTIONAL = 1_000_000.0
NOTIONAL_CURRENCY = 'ZAR'

#: The four names the registry must carry, and nothing else.
ROSTER = {'Straddle', 'Strangle', 'ZeroCostCollar', 'Seagull'}

#: `solve_deal_field`'s own default, in report currency. A zero-cost structure is zero-cost to
#: whatever the solve was allowed to leave on the table, so this is the tolerance every net-zero
#: assertion is entitled to - and no more.
SOLVE_TOLERANCE = 0.01


def params(**extra):
    return dict({'pair': PAIR, 'expiry': EXPIRY, 'notional': NOTIONAL,
                 'notional_currency': NOTIONAL_CURRENCY}, **extra)


@pytest.fixture(scope='module')
def book():
    """The book: `test_service`'s job document with a USDZAR spot the market would recognise, the
    canned FX vol quotes installed, and the bootstrap run so the file carries the `FXVol.USD.ZAR`
    surface a pricer reads - exactly what `/book/market` leaves behind, and the state a desk's book
    is actually in when a quote arrives."""
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


def option(reference, market_strike, market_type, buy_sell='Buy'):
    """One vanilla leg authored BY HAND on the engine's axis - `test_service`'s `FX_OPTION` shape,
    with the market-to-engine conversion done here in the gate rather than read from the runner.

    A rand-notional USDZAR option is an option on ZAR settled in USD, so the deal's strike is USD
    per ZAR (`1/K`) and the market's Call - the right to buy dollars - is the engine's Put on rand.
    Deriving it twice, independently, is what makes the straddle identity a test of the runner and
    not a restatement of it.
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
    """`{Reference: value}` for a base valuation of exactly these deals against the book.

    Every deal goes in through `structures.book_node`, which is the one place that knows a
    container's children hang off the NODE rather than inside the block - so the composed deal is
    gated through the same lift whatever books it will use.
    """
    document = copy.deepcopy(book)
    document['Calc']['Deals']['Deals']['Children'] = [
        structures.book_node(deal) for deal in deals]
    _, out = derivus.Context().load_json((json.dumps(document), 'gate')).run_job()
    frame = out['Results']['mtm']
    return {row['Reference']: float(row['Value']) for _, row in frame.iterrows()}


def leg(outcome, role):
    return next(row for row in outcome['legs'] if row['role'] == role)


def test_the_registry_publishes_exactly_the_four_structures():
    """The store is the front end's whole source: a menu, its parameters, what each one is made of
    and what pricing it does. A leg names a declared `Instrument` type and nothing else - that type's
    entry in the Instrument store IS the leg's field schema, the same reuse-by-reference a
    bootstrapper's `quote_instruments` makes, so a leg naming a type no class declares is a leg that
    cannot be built."""
    store = schema.mapping['Structure']['types']
    assert set(store) == ROSTER
    assert set(store) == set(structures.registry()), (
        'the emitted store and the runner\'s registry disagree about what exists')

    instruments = schema.mapping['Instrument']['types']
    for name, entry in store.items():
        assert set(entry) == {'vernacular', 'fields', 'legs', 'recipe'}
        assert entry['vernacular'] and entry['legs'] and entry['recipe']
        assert {'pair', 'expiry', 'notional', 'notional_currency'} <= set(entry['fields'])
        for key, descriptor in entry['fields'].items():
            assert descriptor['required'] is True, '{}.{} has a default a client cannot mean'.format(
                name, key)
        for role, declared in entry['legs'].items():
            assert declared['deal_type'] in instruments, (
                '{}.{} is a {}, which no class declares'.format(name, role, declared['deal_type']))
            assert set(declared) == {'deal_type', 'pinned', 'slots'}
            unknown = set(declared['slots'].values()) - set(entry['fields'])
            assert not unknown, '{}.{} maps slots to undeclared parameters {}'.format(
                name, role, sorted(unknown))


def test_an_unknown_structure_refuses_with_the_roster():
    """A typo is a sales enquiry that cannot be answered, not an empty answer."""
    with pytest.raises(ValueError) as refusal:
        structures.quote({}, 'RangeAccrual', params())
    assert 'ZeroCostCollar' in str(refusal.value)


def test_a_straddle_is_exactly_its_two_legs(book):
    """The runner's arithmetic against the same two options priced by hand.

    `quote()` reports each wing and their sum; the gate authors the same two deals on the engine's
    axis itself and prices them in one ordinary run. Agreement to 1e-9 relative says the shared
    block, the tenor parse, the strike inversion and the option-sense flip all landed on the same
    contract - a straddle has no solve, so nothing else is in the way.

    The two wings priced at spot come out EQUAL, and that is the forward statement this file needs:
    a call and a put agree in value only at the forward, so in this world (both curves flat at one
    rate) the forward is the spot. The collar gate reads its bracket against that.

    Last, the composed `StructuredDeal` prices to the quoted net, which is the claim that the deal
    riding the quote is the thing that was quoted.
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
    """The structure the verb exists for: the client names a floor, and the cap is whatever strike
    makes the sold call pay for the bought put.

    Three claims. The net is zero to the solve's own tolerance - not approximately a collar, a
    collar. The solved cap is above the forward and the given floor below it, which is what a
    collar IS; a solver that converged on the wrong branch would still net to zero and would fail
    here. And re-quoting the two strikes as a bought `Strangle` prices the legs to equal premiums,
    which is the same statement read from the other side - the cap was found precisely where its
    value matches the floor's, so buying both costs twice one of them.
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


def test_a_seagull_nets_to_zero(book):
    """Three legs, two strikes the client names and one solved - the collar cheapened by selling
    the level below which protection stops. The identity is the same and the arithmetic is not:
    the solve targets the sum of TWO already-priced legs, so a runner that read only the last
    priced leg would still produce a plausible cap and would not net to zero here."""
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


def test_a_market_strike_reaches_the_engine_axis(book):
    """The conversion, on its own, exactly: a floor quoted USDZAR 15.50 on a rand notional is a
    deal struck at `1/15.50` dollars per rand, and the leg the client reads back says 15.50 again.

    The option SENSE crosses with it - the market's put on the pair is the engine's call on the
    rand - and both live in the runner, so a structure declares neither.
    """
    outcome = structures.quote(book, 'ZeroCostCollar', params(floor=15.50))
    protection = outcome['deal']['Children'][0]['Instrument']['.Deal']

    assert protection['Strike_Price'] == 1.0 / 15.50
    assert leg(outcome, 'protection')['strike_market'] == pytest.approx(15.50, rel=1e-12)
    assert protection['Option_Type'] == 'Call', 'a floor on the pair is a call on the rand'
    assert protection['Underlying_Currency'] == 'ZAR' and protection['Currency'] == 'USD'
    assert protection['Expiry_Date'] == {'.Timestamp': '2025-06-28'}, 'the 1Y tenor mis-parsed'
    assert protection['Structure_Reference'] == outcome['deal']['Reference']


def test_the_same_trade_quotes_the_same_from_either_side_of_the_pair(book):
    """The conversion's other half, and the sharpest test of it: a straddle on 1,000,000 rand and a
    straddle on the 54,054 dollars that buys at 18.50 are the SAME trade, and the desk must quote
    them at the same money.

    The two travel opposite paths through the runner. The rand notional is the pair's quote
    currency, so its legs are options on rand settled in dollars, struck at `1/18.50`, with the
    market's call written as an engine put. The dollar notional is the pair's base currency, so
    nothing inverts: the leg is an option on dollars settled in rand, struck at 18.50, a call on
    the pair AND a call on the deal. One number comes out of both, which no single-orientation
    runner can fake.
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


def test_a_quote_is_an_act_not_a_lookup(book):
    """Two identical asks are two quotes. `/book/bloomberg`'s precedent: content addressing is
    right for a computation and wrong for an event, so the id carries a submission clock and the
    same request twice never coalesces into one quote id - while the PRICE it carries is the same,
    because the book has not moved."""
    first = structures.quote(book, 'Straddle', params(strike=SPOT))
    second = structures.quote(book, 'Straddle', params(strike=SPOT))

    assert first['quote_id'] != second['quote_id']
    assert first['deal']['Reference'] != second['deal']['Reference']
    assert first['net'] == pytest.approx(second['net'], rel=1e-12)


def test_an_unparsed_tenor_refuses_rather_than_expiring_today(book):
    """A tenor that does not parse must never fall through to the base date - a zero-day option
    prices at zero without complaining, and a client would be quoted nothing for something."""
    with pytest.raises(ValueError) as refusal:
        structures.quote(book, 'Straddle', dict(params(strike=SPOT), expiry='three months'))
    assert 'three months' in str(refusal.value)
