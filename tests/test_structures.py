"""A structure is only sold once, and it either costs what it says it costs or it does not.

`derivus.structures` declares five FX structures and one runner. There is nothing to unit-test
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

That book quotes MID: the canned observations carry no `Quoted_Bid`/`Quoted_Ask`, so every gate
above the two-sided pair is the runner as it has always been. `two_sided_book` is the same book
with a desk's two-way authored onto the quote block and nothing else changed - the surface stays
the one the mid built, which that fixture proves by re-bootstrapping rather than assuming.
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

#: The names the registry must carry, and nothing else.
ROSTER = {'Straddle', 'Strangle', 'ZeroCostCollar', 'Seagull', 'ForwardExtra'}

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


#: How wide the desk's ATM two-way is in the gates below, in the surface's own units: 0.4 vol
#: points, so a leg is shifted by half of it. Wide enough that the solved coordinate moves by more
#: than any solve tolerance, narrow enough to be a spread a desk would actually show.
ATM_SPREAD = 0.004


def two_way(document, spread=ATM_SPREAD):
    """`document` with a two-way authored around the mid its `FXVolPrices` block already carries -
    the block `derivus_bloomberg.to_market_prices_block` writes when the terminal answers PX_BID
    and PX_ASK. The written surface is not touched: the mid is what built it.

    The wings get a two-way too, half as wide, and no gate below reads them. That is the point:
    v1 charges the ATM spread and carries the smile's own, so a wing spread that started being
    consumed would move these numbers and be caught.
    """
    out = copy.deepcopy(document)
    block = out['Calc']['MergeMarketData']['ExplicitMarketData'][
        'Market Prices']['FXVolPrices.USD.ZAR']
    for point in block['instrument']['Points']:
        half = 0.5 * (spread if point['Quote_Type'] == 'ATM' else 0.5 * spread)
        point['Quoted_Bid'] = point['Quoted_Market_Value'] - half
        point['Quoted_Ask'] = point['Quoted_Market_Value'] + half
    return out


@pytest.fixture(scope='module')
def two_sided_book(book):
    """The same book, quoted two-sided - and the proof, taken here rather than assumed, that the
    bootstrap never reads the two-way: re-bootstrapping the block that now carries bid and ask
    writes the IDENTICAL `FXVol.USD.ZAR` surface the mid alone wrote, key for key and float for
    float. Were it false the owner's ruling would be too: the book would mark at the spread."""
    document = two_way(book)
    context = derivus.Context().load_json((json.dumps(document), 'two-sided'))
    context.bootstrap()
    rebuilt = json.loads(dump(context.current_cfg.params['Price Factors']))
    factors = document['Calc']['MergeMarketData']['ExplicitMarketData']['Price Factors']
    assert rebuilt['FXVol.USD.ZAR'] == factors['FXVol.USD.ZAR'], (
        'the bootstrap read the two-way - the written surface is no longer the mid one')
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


def test_the_registry_publishes_exactly_the_declared_structures():
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


def test_a_forward_extra_costs_nothing_and_solves_its_barrier(book):
    """The desk's most-sold hedge, and the registry's first solved coordinate that is not a strike:
    the client is protected at the rate they name, keeps the favourable move, and pays nothing -
    the price of the upside being that a trade through the barrier knocks the sold call in and the
    whole thing reverts to a plain forward at that same protected rate.

    Four claims. The net is zero to the solve's own tolerance. Both legs are struck at the ONE rate
    the client named, which is what makes the knocked state a forward and not a spread. The solved
    barrier sits ABOVE the market spot - the participation side, and not a tautology: on a rand
    notional that leg lives on the reciprocal axis as a `Down_And_In` bracketed BELOW the engine
    spot, so a runner that flipped the direction the wrong way or inverted the level twice would
    report a barrier under spot or refuse inside the bracket. And the composed `StructuredDeal`,
    spliced back through `book_node`, reprices to the quoted net leg for leg - the deal riding the
    quote is the thing that was quoted.
    """
    protected = SPOT * 0.97
    outcome = structures.quote(book, 'ForwardExtra', params(protected_rate=protected))
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
    """The compatibility contract, stated as an identity rather than as a promise.

    A book whose quote block carries no `Quoted_Bid`/`Quoted_Ask` has no spread to charge, so every
    leg's shift is zero, `with_vol_shift` hands back the document ITSELF rather than a shifted copy,
    and the quote that comes out is the one this file's other gates pin - the same code path, not a
    similar one.

    The sharp half is the comparison. A block carrying a ZERO-WIDE two-way exercises the entire new
    layer - the block is found, the ATM rows are read, a half-spread is computed and interpolated,
    a shift is signed per leg - and must land on the identical floats, not close ones. So the
    presence of the DATA cannot move a price; only a real spread can.

    `net_mid` is the finished legs repriced at mid, and at zero spread that is the same pricing
    twice: it agrees to the bit, which is also what says the solve reports its root's own valuation
    rather than a nearby iterate's.
    """
    ask = params(protected_rate=SPOT * 0.97)
    mid = structures.quote(book, 'ForwardExtra', ask)
    zero_wide = structures.quote(two_way(book, spread=0.0), 'ForwardExtra', ask)

    assert [row['vol_spread'] for row in mid['legs']] == [None, None]
    assert 'Quoted_Bid' in mid['spread_note'] and 'FXVolPrices.USD.ZAR' in mid['spread_note']
    assert structures.with_vol_shift(book, 'FXVol.USD.ZAR', 0.0) is book, (
        'a zero shift copied the book - the mid path is no longer the path it was')

    assert [row['vol_spread'] for row in zero_wide['legs']] == [0.0, 0.0]
    assert zero_wide['spread_note'] is None, 'a two-way was found; there is no fallback to name'
    for row, same in zip(mid['legs'], zero_wide['legs']):
        assert (row['premium'], row['strike_market'], row['barrier_market'], row['solved']) == (
            same['premium'], same['strike_market'], same['barrier_market'], same['solved']), row
    assert mid['net'] == zero_wide['net'] and mid['net_mid'] == zero_wide['net_mid']
    assert mid['net_mid'] == mid['net'], 'the same legs on the same book priced two ways'


def test_a_two_sided_quote_charges_the_spread_and_leaves_the_book_at_mid(book, two_sided_book):
    """The ruling, priced: the spread belongs to the quote and the mid belongs to the book.

    Three statements per structure, and they are DIRECTIONS - a magnitude here would be a
    restatement of the shift rather than a test of what it did.

    The structure still costs nothing. It nets to zero AT THE TWO-SIDED VOLS, which is what a
    zero-cost structure means when a desk quotes one: the client pays no premium, and the price of
    that is where the solved coordinate lands.

    The solved coordinate lands CLIENT-WORSE. The forward extra's client buys protection at the
    offered vol and sells the knock-in at the bid, so the barrier that finances it has to sit
    closer to spot than the mid-solved one - a level more likely to trade, which is exactly what
    the client gives up for the spread. The collar says the same in strikes: the cap comes in.
    Both are strictly between the mid answer and the spot, so a shift applied with the wrong sign,
    to the wrong leg, or to a copy nothing priced would fail here rather than pass quietly.

    And the desk keeps the difference. The finished legs marked at MID come out below what the
    client was quoted, and `net - net_mid` is that gap: the edge, positive, in the report currency.
    Note the frame - a leg's `Buy_Sell` is the CLIENT's side, so the booked package marks NEGATIVE
    on a two-sided quote and the desk's edge is the difference rather than `net_mid` itself.

    This gate is also the empirical answer to "does a pricing run rebuild the surface from
    `Market Prices`?". It does not: the only thing separating these two quotes is a shift applied
    to the written `FXVol` surface of each leg's own copy, and every number below moves.
    """
    ask = params(protected_rate=SPOT * 0.97)
    mid, two_sided = (structures.quote(document, 'ForwardExtra', ask)
                      for document in (book, two_sided_book))
    barrier = (leg(mid, 'reversion')['barrier_market'],
               leg(two_sided, 'reversion')['barrier_market'])

    assert abs(two_sided['net']) <= SOLVE_TOLERANCE, two_sided['net']
    assert leg(two_sided, 'protection')['vol_spread'] == pytest.approx(0.5 * ATM_SPREAD)
    assert leg(two_sided, 'reversion')['vol_spread'] == pytest.approx(-0.5 * ATM_SPREAD)
    assert SPOT < barrier[1] < barrier[0], (
        'the two-sided barrier {} is not inside the mid one {}'.format(*reversed(barrier)))
    assert two_sided['edge'] == two_sided['net'] - two_sided['net_mid'] > 0, two_sided['net_mid']

    floor = params(floor=SPOT * 0.95)
    mid_collar, two_sided_collar = (structures.quote(document, 'ZeroCostCollar', floor)
                                    for document in (book, two_sided_book))
    cap = (leg(mid_collar, 'financing')['strike_market'],
           leg(two_sided_collar, 'financing')['strike_market'])

    assert abs(two_sided_collar['net']) <= SOLVE_TOLERANCE, two_sided_collar['net']
    assert SPOT < cap[1] < cap[0], 'the two-sided cap {} is not inside the mid one {}'.format(
        *reversed(cap))
    assert two_sided_collar['net'] - two_sided_collar['net_mid'] > 0, two_sided_collar['net_mid']


def test_a_knock_in_plus_a_knock_out_is_the_vanilla(book):
    """In-out parity, straight through the engine: a bought up-and-IN call and a bought up-and-OUT
    call on the same strike and the same barrier are between them the vanilla, because exactly one
    of them pays on every path. Three deals authored by hand, one ordinary run, no structure and no
    solve in the way.

    This is the identity the forward extra's sold leg rests on, and it is worth stating alone: the
    census records the analytic knock-IN branch of `pv_barrier_option` as executed by almost no
    test, so the leg the client is being charged for travels code nothing has demanded a number
    from. Closed forms on both sides, so the tolerance is analytic and not statistical: measured
    1.1e-16 relative, one ULP, against a 1e-9 gate.

    That exactness is not luck and it bounds the claim. The two branches are complementary term
    sets of the same Merton-Reiner-Rubinstein decomposition - the knock-in here is `B - C + D`,
    the knock-out `A - B + C - D`, and `A` is the vanilla - so the sum cancels term by term in
    floating point. What this holds is that the IN branch is REACHED and that its terms and signs
    compose to that decomposition, which is exactly the family of defect the partial-time barrier's
    inverted branch turned out to be. It is not an independent check of the formula's accuracy.
    """
    knock = SPOT * 1.10
    common = {'Currency': 'ZAR', 'Discount_Rate': 'ZAR', 'Underlying_Currency': 'USD',
              'Underlying_Amount': NOTIONAL / SPOT, 'FX_Volatility': 'USD.ZAR',
              'Buy_Sell': 'Buy', 'Option_Type': 'Call', 'Strike_Price': SPOT,
              'Expiry_Date': {'.Timestamp': (BASE + pd_offset()).strftime('%Y-%m-%d')}}
    # a deal block IS the pricer's field dict, so the barrier's own two fields are written out -
    # continuous monitoring, no rebate - exactly as the runner writes them onto a barrier leg
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


def test_a_forward_extra_quotes_the_same_from_either_side_of_the_pair(book):
    """The straddle's statement again, with THREE conversions in the way instead of two: a forward
    extra on 1,000,000 rand and one on the dollars that buys are the same trade, and the desk must
    quote them at the same money.

    What the rand side crosses that the straddle's did not is the BARRIER. Its level inverts like a
    strike, and its DIRECTION inverts with it - the `Up_And_In` the structure declares on the pair
    is booked `Down_And_In` on the rand - while the dollar side crosses nothing at all. Get either
    half wrong and the two orientations solve different barriers for one trade, so the sharp claim
    is that the solved LEVELS agree in market terms; the nets agree at all only because `brentq`
    runs to its own `xtol` rather than stopping at the solve's reporting tolerance.

    The equivalent notional converts at the STRIKE, not at the spot: a rand-notional option on the
    reciprocal pays `max(1/S - 1/K, 0) x N`, which is `(N/K) x max(K - S, 0)` read in rand, so the
    same trade is `N / protected_rate` dollars. The straddle's version of this gate is struck at
    the money and cannot tell the two divisors apart; struck away from it, `N / SPOT` misprices by
    exactly the moneyness (measured 3.09% on the protection leg at 0.97 spot).
    """
    protected = SPOT * 0.97
    both_ways = {'pair': PAIR, 'expiry': EXPIRY, 'protected_rate': protected}
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
