"""A structure is only sold once, and it either costs what it says it costs or it does not.

`derivus.structures` declares seven FX structures and one runner. There is nothing to unit-test
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

#: How many inner paths an ACCRUAL structure is quoted on here, and the reason it is not the
#: book's own 1. A TARF and an accumulator are Monte Carlo priced, so a zero-cost solve is a root
#: find over an estimator - deterministic for a fixed seed, which is what lets `brentq` own it at
#: all, but converging on the true root only as the paths grow. The cross-axis gate is what reads
#: that convergence, and it is MEASURED here rather than chosen: the accumulator's two orientations
#: solve strikes 4.8e-4 apart at 1024 paths, 1.3e-4 at 4096, 2.5e-5 at 16384 and 3.9e-5 at 65536.
#: 16384 is where the identity is sharp and a quote still takes about a second.
ACCRUAL_SIMS = 16384

#: The gap the cross-axis gate allows, at that path count: eight times the measured 2.5e-5, which
#: is a band no axis error survives (the smallest of them, a barrier level inverted twice, moves
#: the solved strike by percent).
AXIS_TOLERANCE = 2e-4

#: The same band under the fitted Heston-Nandi, MEASURED rather than inherited: the two
#: orientations solve 2.6e-4 apart at 1024 paths, 2.3e-5 at 4096, 4.2e-6 at 16384 and 1.3e-5 at
#: 65536 - converging, and TIGHTER at the gate's own path count than the lognormal's floor, because
#: both sides run the same daily recursion rather than reading a surface at two moneynesses.
HN_AXIS_TOLERANCE = 1e-4

#: Every parameter in the store that is NOT required, and the value it must publish. A market
#: convention is the only reason a sales parameter carries a default at all, so this list is the
#: one place a new one has to be argued for.
DECLARED_DEFAULTS = {'leverage': 2.0}

#: A calibrated Heston-Nandi factor for the rand, as `/book/hn` writes one - the JOINING side of
#: the pair. Nothing here is fitted: the gates that read it are about which factor the engine looks
#: up and whether the pin reaches the book, so what these five numbers have to be is stationary
#: (persistence 0.90) and roughly the surface's own vol, not a fit of it.
HN_PARAMS = {'Property_Aliases': None, 'Omega': 1e-12, 'Alpha': 2.0e-6, 'Beta': 0.45,
             'Gamma_Star': -474.34, 'H0': 7.8e-5}

#: The strip a TARF and an accumulator are quoted on here: monthly fixings to the tenor, and a cap
#: of 1.50 rand of cumulative favourable move on a spot of 18.50 - reachable enough that the
#: redemption is part of the price rather than decoration.
FIXING_FREQUENCY = '1M'
TARGET = 1.5

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

#: How wide the RR and the BF rows are quoted, as a fraction of that: each of them half as wide as
#: the ATM row, which is the shape a terminal actually prints - a wing quote is a tighter number
#: than the level it hangs off.
WING_FRACTION = 0.5

#: What those two rows compose to at a pillar - `BF_half + RR_half/2`, the ONE number per pillar
#: the strangle algebra makes of them, and what each wing of the smile widens by over and above the
#: flat ATM half. 0.0015 here, against an ATM half of 0.002.
WING_HALF = 0.5 * WING_FRACTION * ATM_SPREAD + 0.5 * (0.5 * WING_FRACTION * ATM_SPREAD)


def two_way(document, spread=ATM_SPREAD, wings=WING_FRACTION):
    """`document` with a two-way authored around the mid its `FXVolPrices` block already carries -
    the block `derivus_bloomberg.to_market_prices_block` writes when the terminal answers PX_BID
    and PX_ASK. The written surface is not touched: the mid is what built it.

    `spread` is the ATM row's own width; `wings` is the RR and BF rows', as a fraction of it. Three
    values of `wings` are three books the gates below need: a fraction is a desk's real wing quote,
    `None` authors no sides on those rows at all - an ATM-only two-way, the book the skew is
    measured against - and a NEGATIVE fraction crosses those prints, a stale bid through a live
    offer.
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
            # a parameter is REQUIRED unless the market itself has a convention for it - a TARF's
            # loss-side gearing is 2.0 unless the client says otherwise - and a declared default
            # has to be PUBLISHED as the value, because a front end that cannot see it is a front
            # end that makes the client state a number the desk already assumed
            if descriptor.get('required') is not True:
                assert key in DECLARED_DEFAULTS, (
                    '{}.{} has a default a client cannot mean'.format(name, key))
                assert descriptor['value'] == DECLARED_DEFAULTS[key], (name, key, descriptor)
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

    The book carries a wing two-way as well as an ATM one, so what moves the coordinate here is the
    whole spread: the flat shift and the skew together, each signed by the same side. The two are
    told apart by the gates below this one, which quote the ATM-only book beside this one.

    This gate is also the empirical answer to "does a pricing run rebuild the surface from
    `Market Prices`?". It does not: the only thing separating these two quotes is what was done to
    the written `FXVol` surface of each leg's own copy, and every number below moves.
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


COLLAR = 'ZeroCostCollar'


def cap_of(outcome):
    return leg(outcome, 'financing')['strike_market']


def surface_of(document):
    """The written `FXVol.USD.ZAR` surface as `{(moneyness, expiry): vol}` - what a leg actually
    prices on, read off the document it prices against rather than off the quotes."""
    rows = document['Calc']['MergeMarketData']['ExplicitMarketData']['Price Factors'][
        'FXVol.USD.ZAR']['Surface']['.Curve']['data']
    return {(row[0], row[1]): row[2] for row in rows}


def test_a_wing_two_way_skews_the_smile_rather_than_shifting_it(two_sided_book):
    """What the RR and BF rows do to a side's copy of the surface, read off the surface itself.

    THE COMPOSITION FIRST. A wing vol is `ATM + BF +- RR/2`, so its offered side is the offered
    side of every term it is made of and the SUBTRACTED one takes its bid - half the ask-less-bid
    of a linear combination is each term's own half times the size of its coefficient, summed. Both
    wings of a pillar therefore widen by `BF_half + RR_half/2`: the risk reversal's spread reaches
    both wings and its SIGN does not, which is why a two-way on the skew quote does not tilt the
    smile. That is `WING_HALF`, and it is asserted as the number rather than as a direction.

    THEN THE SHAPE, which is the whole ruling. A flat shift moves every node of the surface by ONE
    number; this moves the widest quoted nodes by the composed half and the money by almost
    nothing, so what comes out is a different smile rather than the same smile at a different
    level. Measured on this book, the node nearest the money moves 5% of the wing's own widening at
    three months and 10% at a year, against 100% at either end.

    And EVERY node moves the client's way - the minimum move over the whole surface is positive.
    That is the same rule the crossed print states: a widening that went negative somewhere on the
    grid would be paying the client for the desk's own uncertainty about the skew.
    """
    wings = structures.wing_two_way(two_sided_book, 'USD.ZAR')
    assert set(wings) == {(0.25, 0.25), (1.0, 0.25)}, wings
    assert all(half == pytest.approx(WING_HALF, rel=1e-12) for half in wings.values()), wings

    mid = surface_of(two_sided_book)
    skewed = surface_of(structures.with_vol_shift(
        two_sided_book, 'FXVol.USD.ZAR', 0.0,
        structures.quote_points(two_sided_book, 'USD.ZAR'), wings))
    shifted = surface_of(structures.with_vol_shift(
        two_sided_book, 'FXVol.USD.ZAR', 0.5 * ATM_SPREAD))
    skew = {node: skewed[node] - vol for node, vol in mid.items()}
    flat = {node: shifted[node] - vol for node, vol in mid.items()}

    assert min(skew.values()) > 0.0, 'the skew pays the client somewhere on the grid'
    assert max(flat.values()) - min(flat.values()) < 1e-15 < max(skew.values()) - min(skew.values())
    for expiry in (0.25, 1.0):
        nodes = sorted(node for node in mid if node[1] == expiry)
        assert skew[nodes[0]] == pytest.approx(WING_HALF, rel=1e-9), nodes[0]
        assert skew[nodes[-1]] == pytest.approx(WING_HALF, rel=1e-9), nodes[-1]
        money = min(nodes, key=lambda node: abs(node[0]))
        assert skew[money] < 0.2 * WING_HALF, 'the money moved with the wings'


def test_a_wing_two_way_moves_the_solved_coordinate_further_in(book, two_sided_book):
    """The skew, priced - and the sign argument is one argument for every leg of both structures.

    A leg struck away from the money prices off the WINGS, and the wings of that leg's own copy of
    the surface have moved by `ATM_half + WING_HALF` where the flat shift alone would have moved
    them by `ATM_half`. The sign is the client's side, the same sign the flat shift takes: what
    they BUY costs more than the ATM-only book says, what they SELL is worth less. Because the
    widening is symmetric across a pillar it does not matter which wing a leg lands on - the
    asymmetry between two legs comes from their SIDES, never from their strikes.

    So every leg of both structures pushes the solved coordinate the same way. The collar's client
    buys the put (dearer) and sells the call that funds it (cheaper), so the cap has to come IN to
    balance. The seagull adds a sold put, cheaper again, which finances less - in further still.
    Three books say it: the mid cap, the ATM-only cap inside it, and the wing cap inside that,
    with every one of them still above the forward. A skew consumed with the wrong sign, or on the
    wrong leg, lands outside that ordering rather than passing quietly.

    `wing_spread` is what the outcome says about it: one composed half per quoted pillar, or None
    on the ATM-only book, which is how a consumer tells a skewed quote from a shifted one.
    """
    atm_only = two_way(book, wings=None)
    floor = params(floor=SPOT * 0.95)
    winged, flat, mid = (structures.quote(document, COLLAR, floor)
                         for document in (two_sided_book, atm_only, book))

    assert abs(winged['net']) <= SOLVE_TOLERANCE, winged['net']
    assert winged['wing_spread'] == {'0.25 0.25': pytest.approx(WING_HALF, rel=1e-12),
                                     '0.25 1': pytest.approx(WING_HALF, rel=1e-12)}
    assert flat['wing_spread'] is None and mid['wing_spread'] is None
    assert leg(winged, 'protection')['vol_spread'] == leg(flat, 'protection')['vol_spread'], (
        'the flat shift moved when the wings did')
    assert SPOT < cap_of(winged) < cap_of(flat) < cap_of(mid), (
        'the skewed cap {} is not inside the ATM-only cap {}'.format(
            cap_of(winged), cap_of(flat)))
    assert winged['edge'] > flat['edge'] > 0.0, 'the skew captured nothing'

    bird = params(floor=SPOT * 0.98, lower_floor=SPOT * 0.90)
    winged, flat, mid = (structures.quote(document, 'Seagull', bird)
                         for document in (two_sided_book, atm_only, book))

    assert abs(winged['net']) <= SOLVE_TOLERANCE, winged['net']
    assert SPOT < cap_of(winged) < cap_of(flat) < cap_of(mid), (
        'the seagull cap {} is not inside the ATM-only cap {}'.format(
            cap_of(winged), cap_of(flat)))
    assert winged['edge'] > flat['edge'] > 0.0


def test_a_book_with_no_wing_two_way_quotes_exactly_as_it_always_did(book):
    """The compatibility contract for the skew, in the zero-wide precedent's own shape.

    THREE books that quote no wing spread, and they have to be one quote to the float. One carries
    no `Quoted_Bid`/`Quoted_Ask` on its RR and BF rows at all, so `wing_two_way` never sees them.
    One quotes them ZERO-WIDE, which exercises the whole reading - the rows are found, both sides
    are read, a half is composed per pillar - and must still compose to nothing rather than to a
    widening of zero, because a widening of zero would rebuild the smile and land on floats that
    are merely close. And one CROSSES them, a stale bid through a live offer, which reads zero-wide
    for the reason a crossed ATM print does: the one thing a desk must not do with a broken print
    is pay a client for it.

    All three still charge the ATM spread - this is the WING layer's absence, not the two-way's -
    so the gate is that the skew alone did nothing, and the quote that comes out is the one
    `test_a_two_sided_quote_charges_the_spread_and_leaves_the_book_at_mid` pins.
    """
    ask = params(floor=SPOT * 0.95)
    absent, zero_wide, crossed = (structures.quote(two_way(book, wings=wings), COLLAR, ask)
                                  for wings in (None, 0.0, -WING_FRACTION))

    assert [outcome['wing_spread'] for outcome in (absent, zero_wide, crossed)] == [None] * 3
    assert [outcome['spread_note'] for outcome in (absent, zero_wide, crossed)] == [None] * 3
    for outcome in (zero_wide, crossed):
        for row, same in zip(absent['legs'], outcome['legs']):
            assert (row['premium'], row['strike_market'], row['solved'], row['vol_spread']) == (
                same['premium'], same['strike_market'], same['solved'], same['vol_spread']), row
        assert (absent['net'], absent['net_mid'], absent['edge']) == (
            outcome['net'], outcome['net_mid'], outcome['edge'])


def test_a_book_that_never_states_use_quotes_rather_than_raising(book):
    """`Use` is an OPTIONAL field, so its absence is a quote that counts - and the quote layer has
    to read it the way the rest of the file does.

    `to_market_prices_block` writes the field, so every book above carries it; a hand-authored
    block, or one from a source that only ever states what it wants held OUT, does not. The wing
    reader walks the block on EVERY quote - a book whose RR and BF rows carry a mid alone included,
    since walking it is how the layer learns there is no wing two-way to charge - so a strict
    `point['Use']` there made a `KeyError` of a book that prices perfectly well, and made it inside
    `quote()` rather than at any refusal seam a desk could read.

    The gate is the whole quote, not the reader: the same collar off the same surface, charging the
    same ATM spread, solving the same cap to the float as the book that states the field.
    `atm_two_way` already read it leniently, so what this pins is that the two readers agree about
    what an absent `Use` means.
    """
    stated = two_way(book, wings=None)
    useless = copy.deepcopy(stated)
    points = useless['Calc']['MergeMarketData']['ExplicitMarketData'][
        'Market Prices']['FXVolPrices.USD.ZAR']['instrument']['Points']
    assert all(point.pop('Use') == 'Yes' for point in points), 'the fixture states no Use to drop'

    ask = params(floor=SPOT * 0.95)
    quoted = structures.quote(useless, COLLAR, ask)

    assert structures.quote_points(useless, 'USD.ZAR') == points
    assert quoted['wing_spread'] is None, 'the wing reader found sides this block does not quote'
    assert cap_of(quoted) == pytest.approx(19.16949442549199, rel=1e-12), cap_of(quoted)
    assert cap_of(quoted) == cap_of(structures.quote(stated, COLLAR, ask))


#: The desk's mandate, as a book declares one. `participation` at a half is the default a declared
#: block carries; the gates below vary one field at a time off this.
POLICY = {'participation': 0.5, 'floor': 'mid', 'scope': 'vol',
          'bucket_limit': None, 'min_ticket_bp': 0.0, 'firm_seconds': 600}


def with_policy(document, **stated):
    """`document` with a `Quote Policy` block on it - the whole risk-impact feature's on switch."""
    out = copy.deepcopy(document)
    out['Calc'][structures.QUOTE_POLICY] = dict(POLICY, **stated)
    return out


def holding(document, deals):
    """`document` with exactly `deals` standing in its deal tree, through the one lift that knows a
    container's children hang off the NODE."""
    out = copy.deepcopy(document)
    out['Calc']['Deals']['Deals']['Children'] = [structures.book_node(deal) for deal in deals]
    return out


@pytest.fixture(scope='module')
def standing(two_sided_book):
    """One collar quoted at the full two-way - the trade the books below already carry."""
    return structures.quote(two_sided_book, COLLAR, params(floor=SPOT * 0.95))


def test_a_book_with_no_quote_policy_quotes_exactly_as_it_always_did(two_sided_book, standing):
    """The compatibility contract for the risk-impact half, and it is the same shape as the
    two-sided one: absence is not a small effect, it is the identical code path.

    A book declaring no `Quote Policy` never reaches the greeks runs at all - `risk.scale` is None
    rather than 1.0, which is the difference between "the feature did not run" and "it ran and
    decided nothing", and a consumer that cannot tell those apart cannot audit a quote. A book
    declaring one with `participation` at ZERO runs the WHOLE layer - both greeks runs, the
    buckets, the charge - and must land on the identical floats, so the presence of the policy
    cannot move a price and only a stated participation can.

    Both quotes are given against a book already carrying a position, so the measurement has
    something real to say and its silence is a decision rather than an empty book's default.
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
        assert (row['premium'], row['strike_market'], row['solved'], row['vol_spread']) == (
            same['premium'], same['strike_market'], same['solved'], same['vol_spread']), row
    assert (plain['net'], plain['net_mid'], plain['edge']) == (
        zero['net'], zero['net_mid'], zero['edge'])


def renamed(deal, suffix):
    """The same trade under its own reference - a book may hold two of one structure, and the mtm
    frame is keyed by reference."""
    copied = copy.deepcopy(deal)
    copied['Reference'] += suffix
    for child in copied.get('Children', []):
        child['Instrument']['.Deal']['Reference'] += suffix
        child['Instrument']['.Deal']['Structure_Reference'] = copied['Reference']
    return copied


def test_an_offset_quotes_tighter_than_a_repeat(book, two_sided_book, standing):
    """The whole ruling, priced: what a trade costs is what hedging the RESIDUAL it leaves costs,
    at the market's own two-way.

    The registry has no sell-side collar - a collar's client always buys the put and sells the call -
    so the opposite SIDE is put on the BOOK rather than into the quote. Two books carry the same
    trade the two ways a desk can hold it: SHORT it (the mirror of a collar it quoted, the ordinary
    case) and LONG it (what it holds when a client sold it one). Quoting that collar into the first
    piles the same risk on again; into the second it nets the book flat. One structure, one policy,
    one set of parameters, opposite signs - so nothing but the residual can be producing the
    difference below.

    Four claims. The offsetting quote is TIGHTER (`scale` strictly inside 1) and the repeat is not
    (`scale` exactly 1 - the market's own spread is the ceiling in v1, so a risk-adding trade takes
    no surcharge). The charges order the same way. The offset's solved cap lands strictly between
    the full-spread cap and the MID cap - client-better than the full spread, and never through the
    mid, which is the floor the policy declares. And every scale is in [0, 1], on all three.

    The buckets say why: the same trade's mirror doubles `RR 0.25 1` on one book and zeroes it on
    the other, and the RR pillar's own half-spread is what that move is charged at.
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

    # the mirror doubles the skew bucket on one book and cancels it on the other - one number,
    # read from both sides, which no sign error survives
    skew = {row['bucket']: row for row in reducing['risk']['buckets']}['RR 0.25 1']
    piled = {row['bucket']: row for row in adding['risk']['buckets']}['RR 0.25 1']
    assert skew['before'] == pytest.approx(-piled['before'], rel=1e-9)
    assert abs(skew['after']) < 1e-6 * abs(skew['before']), 'the offset left skew standing'
    assert piled['after'] == pytest.approx(2.0 * piled['before'], rel=1e-9)
    assert skew['half_spread'] > 0.0

    # the OUTCOME describes the quote it charged, both halves of it. Every leg's `vol_spread` is
    # already the scaled one - the re-quote multiplies the ATM half by `scale` - and the composed
    # wing halves are reported at the same scale, so a consumer reading the skew off the answer
    # reads what the client dealt on rather than the untightened two-way the book quotes
    assert reducing['wing_spread'] == {
        pillar: pytest.approx(WING_HALF * reducing['risk']['scale'], rel=1e-12)
        for pillar in ('0.25 0.25', '0.25 1')}, reducing['wing_spread']
    assert all(half < WING_HALF for half in reducing['wing_spread'].values())
    assert adding['wing_spread'] == {pillar: pytest.approx(WING_HALF, rel=1e-12)
                                     for pillar in ('0.25 0.25', '0.25 1')}
    assert abs(leg(reducing, 'protection')['vol_spread']) == pytest.approx(
        reducing['risk']['scale'] * abs(leg(adding, 'protection')['vol_spread']), rel=1e-12), (
        'the flat half and the wing halves are not on one scale')

    assert cap_of(adding) == cap_of(full_spread), 'the repeat is not the full-spread quote'
    assert cap_of(adding) < cap_of(reducing) < cap_of(structures.quote(book, COLLAR, ask)), (
        'the tightened cap {} is not between the full-spread cap {} and the mid one'.format(
            cap_of(reducing), cap_of(adding)))
    assert all(0.0 <= outcome['risk']['scale'] <= 1.0 for outcome in (adding, reducing))


def test_the_cap_and_the_floor(two_sided_book, standing):
    """The two limits the policy declares, each made to bind on a book where it matters.

    THE CAP. A book holding TWO of the trade still has one of them standing after the offset, so
    the residual is real - a `bucket_limit` under it suspends the tightening entirely and NAMES the
    bucket. That is the conservative direction and it is the point of the field: a book already
    over its limit somewhere does not get to quote tighter on the strength of netting down
    elsewhere, however good the saving looks.

    THE FLOOR. `min_ticket_bp` is flat bp of NOTIONAL, and the notional is in rand while the charge
    is in the report currency, so it crosses on the same `FxRate` ratio every other conversion here
    reads. Set inside the band the tightening opens - between the tightened charge and the full
    one - it binds exactly: the effective charge lands ON the ticket rather than below it, and the
    quote is wider than the unfloored one and still no wider than the two-way.
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


def test_a_live_cross_lands_exactly_where_engine_spot_reads_it_back(book):
    """The live-spot conversion, on a real book and with no terminal anywhere near it.

    `with_live_spots` is the exact inverse of `engine_spot`, so the gate is the round trip: a market
    cross written in against the book's own `Base_Currency`, the same cross read back out on the
    DEAL axis, and the same number a third time in MARKET terms off a real quote's `spot` block -
    which is what a salesperson reads. The runner reports the spot it PRICED on rather than one it
    was handed, so the reading has to survive the axis inversion a rand notional puts it through.

    The two refusals are the ones that keep a wrong number out of a book. A pair with NEITHER leg
    against the base cannot be placed without a second cross, and triangulating one would be a
    market view rather than a tick; a currency the book carries no `FxRate` for is a new price
    factor, which is authoring rather than the `bind='value'` seam a spot moves through.
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


# --------------------------------------------------------------------------------------------
# the accrual strips: a TARF and an accumulator, which are one leg and a SCHEDULE
# --------------------------------------------------------------------------------------------
@pytest.fixture(scope='module')
def accrual_book(book):
    """The same book quoted at `ACCRUAL_SIMS` inner paths.

    The only edit is the path count, and it is not tuning: every other gate here prices a closed
    form, while a strip is Monte Carlo and its zero-cost strike is a root find over an estimator.
    `MCMC_Simulations` is 1 on `test_service`'s job, which is the right number for a cashflow's
    arithmetic and one path for a TARF.
    """
    document = copy.deepcopy(book)
    document['Calc']['Calculation']['MCMC_Simulations'] = ACCRUAL_SIMS
    return document


def accrual_params(**extra):
    return params(fixing_frequency=FIXING_FREQUENCY, **extra)


def only_leg(outcome):
    """The one deal a strip composes to - the container's single child."""
    return outcome['deal']['Children'][0]['Instrument']['.Deal']


def test_an_accumulator_crosses_both_axes_and_a_tarf_refuses_the_second(accrual_book):
    """The axis gate for the strips, and it has two halves because the two structures answer
    differently - which is the whole finding.

    THE ACCUMULATOR CROSSES. An accumulator on 1,000,000 rand and one on the dollars that buys at
    the solved strike are the SAME trade, and the desk must quote them at the same money. The rand
    side crosses everything at once: the strike inverts, the market Call is written as an engine
    Put, the knock-out LEVEL inverts with the strike and its DIRECTION inverts with it - the
    `Up_And_Out` the structure declares on the pair is booked `Down_And_Out` on the rand - while
    the dollar side crosses nothing at all. Get any of them wrong and the two orientations solve
    different strikes for one trade. The equivalent notional converts at the STRIKE, exactly as the
    forward extra's gate states, and the strike here is the SOLVED one - which costs nothing to
    use, because a zero-cost strike is scale-invariant (both sides of the payoff carry the
    notional) and the sizing therefore only has to make the two NETS comparable, never the strike.

    THE TARF DOES NOT, and refuses rather than quoting something else. Its target is a cap on the
    ACCRUAL, which is a sum of differences rather than a level: `1/S - 1/K` is not the reciprocal
    of `S - K`, so no number in reciprocal units means the client's cap and the two orientations
    would be different trades wearing one parameter. The deal's own `InvertedTarget` flag is not
    the way out - it moves the whole fixing onto the reciprocal axis, paying the notional per unit
    of MOVE, which is a coherent product and not the one `notional_currency` names (measured 0.77%
    apart in the solved strike, and neither wrong about its own product). So the refusal names the
    currency, and `InvertedTarget` is False on every leg the runner builds.
    """
    both_ways = {'pair': PAIR, 'expiry': EXPIRY, 'fixing_frequency': FIXING_FREQUENCY,
                 'knockout': SPOT * 1.10}
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
    bargain they are paying gearing and a knock-out for.

    A TARF is dealt at no upfront - `Solve('tarf', 'Strike_Price', 0.0)` is the whole recipe, with
    no leg to fund it - so the net is zero to the solve's own tolerance and the composed deal
    reprices to it through `book_node`, leg for leg.

    The SIDE of the forward is the claim that is not a tautology. Both curves are flat at one rate
    in this world, so the forward IS the spot (the straddle gate asserts it), and at a strike of
    the forward the client's bought leg and their sold leg are worth the SAME per fixing - at which
    point `leverage` times the sold one outweighs one of the bought one and the package is
    negative. The strike therefore has to come DOWN until the two balance, and a client accruing
    `(S - K)+` is better off the lower it goes. So a zero-cost accrual strike sits BELOW the
    forward, and that discount is exactly what the 2x gearing and the redemption cap are sold for.
    A runner that inverted a strike twice, or fed the pricer a leverage of one, lands above it.

    The rest is what the parameters became: twelve monthly fixings to the tenor, each settling two
    days on; the geared notional at `leverage x notional` off a default the client never stated;
    and the target copied through UNCONVERTED, on the pair's own axis, with `InvertedTarget` False.

    The HN pin is exercised here by its ABSENCE, which is the only arm this repo can reach: no
    fixture carries a `HestonNandiModelParameters` price factor for an FX underlying, so the leg is
    priced GBM and SAYS so on its own row. The alternative is what makes that worth a gate -
    pinning the model on a book with no calibration raises inside the engine's dependency loop,
    which SKIPS the deal and logs an ERROR, and the quote comes back with its only leg priced at
    nothing.
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
    assert 'HestonNandiModelParameters' in row['note'], row['note']
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


def test_a_strip_ends_on_its_own_expiry_or_refuses(book):
    """The two ways a fixing strip can come out SHORT of the tenor it was quoted at, and neither
    is allowed to be silent.

    A FREQUENCY THAT DOES NOT DIVIDE. A 1Y ticket at a 5M frequency fixes twice - November and
    April - and the loop's only stopping rule is `fixing > expiry`, so it stops there. Nothing
    downstream can tell: a TARF's `Expiry_Date` is set to the last SETTLEMENT, so the deal is
    priced, reported and two-way spread at the SHORT tenor, and the ticket still says 1Y. So the
    strip refuses, naming the expiry, the frequency, the last fixing it would have produced and
    the two ways out - a frequency that divides, or the broken date quoted directly.

    A BASE DATE CARRYING A TIME. `expiry_date` normalizes its answer to a date, and a terminal
    snapshot stamps `Base_Date` at 16:30, so the final fixing landed one comparison past midnight
    on the expiry and a twelve-fixing year quietly became eleven. The base is normalized before
    the loop, and the count is the same from either stamp.
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
            params(target=TARGET, fixing_frequency='5M'), notional_currency='USD'))
    assert '5M' in str(quoted.value)


def test_the_axis_refusal_fires_before_the_deal_is_furnished(book):
    """A refusal is not allowed to leave a half-built deal behind.

    `furnish_accrual` writes the schedule and the geared notional onto the deal block the caller
    holds - IN PLACE, because a deal block IS the field dict the pricer reads. So the TARF's axis
    refusal has to be the function's FIRST statement: fired after those writes, it hands a caller
    that catches it a block carrying a strip and a leverage for a trade that was never quoted.

    The ordering is visible in the wording too. An inverted TARF with a frequency that also does
    not divide reports the AXIS - the thing the client has to change - rather than the schedule
    it would never have got to build.
    """
    deal = {'Object': 'FXTARFOptionDeal', 'Currency': 'USD', 'Underlying_Currency': 'ZAR'}
    with pytest.raises(ValueError) as refusal:
        structures.furnish_accrual(
            deal, params(target=TARGET, fixing_frequency='5M'), book, BASE, 'ZAR', True)

    assert 'accrual cap' in str(refusal.value) and 'ZAR' in str(refusal.value)
    assert '5M' not in str(refusal.value), 'the schedule was built before the axis was checked'
    assert deal == {'Object': 'FXTARFOptionDeal', 'Currency': 'USD', 'Underlying_Currency': 'ZAR'}


def calibrated(document):
    """The book with the pair's Heston-Nandi fit installed, as `/book/hn` writes it - under the
    pair's NON-BASE token, which is the only leg of it an `FxRate` can be."""
    document = copy.deepcopy(document)
    document['Calc']['MergeMarketData']['ExplicitMarketData']['Price Factors'][
        'HestonNandiModelParameters.ZAR'] = dict(HN_PARAMS)
    return document


def test_the_absence_note_names_the_factor_the_book_would_need(accrual_book):
    """What the leg says when the model is NOT pinned, and what it says once the pair IS fitted.

    THE KEYING IS ONE RULE NOW. A spot model's parameters are named for the pair's NON-BASE token,
    that being the only leg of the pair the engine simulates and the only one the calibration can
    write, so `structures.spot_model` and `get_spot_model_params_factor` make the same lookup. On a
    book that has never been fitted the note names THAT factor and the verb that installs it - a
    salesperson can act on it - and on a book that has been fitted, both orientations pin and there
    is no note left to write. Which is the whole of the old note's second and third sentences
    deleted rather than reworded: they existed to explain a disagreement that no longer exists.
    """
    tarf = structures.quote(copy.deepcopy(accrual_book), 'TargetRedemptionForward', dict(
        accrual_params(target=TARGET), notional_currency='USD'))
    note = leg(tarf, 'tarf')['note']

    assert 'HestonNandiModelParameters.ZAR' in note, 'the factor looked up is unnamed'
    assert 'HestonNandiModelParameters.USD' not in note, (
        'the base currency is a numeraire, never a rate - it can name no block')
    assert 'non-base' in note and '/book/hn' in note, 'a note without a remedy'
    assert tarf['valuation_configuration'] is None, 'a model was pinned that cannot be resolved'

    # and on the fitted book BOTH orientations join: the TARF forced onto the base currency, and
    # the accumulator that crosses freely
    document = calibrated(accrual_book)
    joined = structures.quote(document, 'TargetRedemptionForward', dict(
        accrual_params(target=TARGET), notional_currency='USD'))
    assert leg(joined, 'tarf')['note'] is None, 'the pinned arm still carries a note'
    assert joined['valuation_configuration'] == {
        'FXTARFOptionDeal': {'SpotModel': 'HestonNandi'}}

    accumulator = structures.quote(document, 'Accumulator', dict(
        accrual_params(knockout=SPOT * 1.10), notional=NOTIONAL, notional_currency='ZAR'))
    assert leg(accumulator, 'accumulator')['note'] is None
    assert accumulator['valuation_configuration'] == {
        'FXAccumulatorOptionDeal': {'SpotModel': 'HestonNandi'}}


def test_the_token_rule_answers_the_same_token_in_either_spelling():
    """One rule, four callers, two dialects.

    The engine (`instruments`), discovery (`Config.calculate_dependencies`) and the calibration all
    speak `check_rate_name` TUPLES; the runner here speaks flat names off the document. The rule
    compares on the checked form, so a mixed call cannot answer `underlying` by falling through an
    equality that was never going to hold. That answer is the PRE-RULE token, which on the wire is
    the defect this row closes wearing the fix's clothes.
    """
    spellings = (('USD', 'ZAR', 'USD'), (('USD',), ('ZAR',), ('USD',)),
                 ('USD', 'ZAR', ('USD',)), (('USD',), ('ZAR',), 'USD'))
    for underlying, currency, base in spellings:
        token = utils.spot_model_currency(underlying, currency, base)
        assert utils.check_rate_name(token) == ('ZAR',), (underlying, currency, base, token)

    # a CROSS keeps the underlying, in every spelling and byte for byte
    assert utils.spot_model_currency('EUR', 'GBP', 'USD') == 'EUR'
    assert utils.spot_model_currency(('EUR',), ('GBP',), ('USD',)) == ('EUR',)


def test_a_book_that_declares_no_base_currency_refuses_instead_of_pinning(accrual_book):
    """The base is the OTHER half of the token rule, and an unknown one may not be guessed.

    The rule needs two things: the pair, which is on the deal, and the base, which is on the book.
    A quote reads that off the EXPLICIT block - the same half `market_data` reads and the only half
    a quote can write - while the engine reads its own merged params, so a book that keeps its
    `System Parameters` behind a `MarketDataFile` answers nothing here.

    NOTHING is what this must never resolve into a token. Falling back to `Underlying_Currency` is
    the PRE-RULE read, and its answer is right or wrong depending on a base this layer just said it
    cannot see: where the underlying is not the base the two agree by luck, and where it IS the base
    the quote PINS a model the engine then looks up under the other name - a dependency-loop raise,
    a skipped deal, and the trade marked at nothing on a job reporting success. A guess that is
    sometimes right is the worst of the three, so the token is not guessed: it refuses, naming the
    declaration that is missing, and pins nothing on the way out.
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
    """THE JOINING GATE, and the defect's own signature is what it reads.

    A USDZAR TARF is forced onto the pair's BASE currency - a target is a cap on a sum of
    differences and has no reading on the reciprocal - so it used to look up a factor named for the
    base, which is a NUMERAIRE and can name no block at all. It therefore rode GBM however many
    times the pair was calibrated, and the way that showed was BIT-IDENTITY: the solved strike
    under the declared model and under no model at all were the same float, to the last bit, on a
    book carrying the fit. MEASURED on this book, before: 16.774620757621133 both ways, separation
    exactly 0.0. After: 17.409375992866366 against the same GBM 16.774620757621133, 3.78% apart,
    against a solve floor of 2.5e-5 at this path count.

    THE ACCUMULATOR'S EXISTING HIT IS UNMOVED, and that is the other half. It was already joining -
    its notional is the rand, so its underlying was already the non-base token - so the keying
    change must not move it by one bit. It does not: 18.15503015327775 before and after.
    """
    gbm = structures.quote(copy.deepcopy(accrual_book), 'TargetRedemptionForward', dict(
        accrual_params(target=TARGET), notional_currency='USD'))
    modelled = structures.quote(calibrated(accrual_book), 'TargetRedemptionForward', dict(
        accrual_params(target=TARGET), notional_currency='USD'))

    lognormal = leg(gbm, 'tarf')['strike_market']
    garch = leg(modelled, 'tarf')['strike_market']
    assert lognormal != garch, 'the declared model priced the lognormal, to the bit'
    assert abs(garch / lognormal - 1.0) > 1e-2, (garch, lognormal)
    assert abs(gbm['net']) <= SOLVE_TOLERANCE and abs(modelled['net']) <= SOLVE_TOLERANCE

    accumulator = structures.quote(calibrated(accrual_book), 'Accumulator', dict(
        accrual_params(knockout=SPOT * 1.10), notional=NOTIONAL, notional_currency='ZAR'))
    # BIT-IDENTICAL across the change when measured directly (18.15503015327775 either side); the
    # band here is the accumulation order's, not the claim's - the hit it must not lose is 5.1e-2
    assert leg(accumulator, 'accumulator')['strike_market'] == pytest.approx(
        18.15503015327775, rel=1e-9), 'the orientation that already joined moved'


def test_the_accumulator_solves_one_strike_from_either_axis_under_the_model(accrual_book):
    """RECIPROCAL CONSISTENCY under the fitted law - the axis gate's shape, one model on.

    The rand orientation rides the fit as written; the dollar orientation is on the RECIPROCAL of
    the axis it was fitted on and settles in the other currency, so the law is carried to that
    numeraire (`utils.hn_reciprocal_gamma`). If it were not carried - if the pricer simply walked
    the fitted law and read `1/s` - the two orientations would solve strikes 3.4e-3 apart and the
    gap would NOT close with the path count, because it is a Siegel drift and not noise.

    THE TOLERANCE IS MEASURED HERE AND NOT INHERITED. Under the model the two solve 2.6e-4 apart at
    1,024 paths, 2.3e-5 at 4,096, 4.2e-6 at 16,384 and 1.3e-5 at 65,536 - converging, and TIGHTER
    at the gate's own path count than the lognormal's own 2.5e-5 floor, because both orientations
    now run the same daily recursion. The band is 1e-4.
    """
    document = calibrated(accrual_book)
    both_ways = {'pair': PAIR, 'expiry': EXPIRY, 'fixing_frequency': FIXING_FREQUENCY,
                 'knockout': SPOT * 1.10}
    in_rand = structures.quote(copy.deepcopy(document), 'Accumulator', dict(
        both_ways, notional=NOTIONAL, notional_currency='ZAR'))
    strike = leg(in_rand, 'accumulator')['strike_market']
    in_dollars = structures.quote(copy.deepcopy(document), 'Accumulator', dict(
        both_ways, notional=NOTIONAL / strike, notional_currency='USD'))

    assert leg(in_rand, 'accumulator')['note'] is None
    assert leg(in_dollars, 'accumulator')['note'] is None
    assert abs(in_rand['net']) <= SOLVE_TOLERANCE and abs(in_dollars['net']) <= SOLVE_TOLERANCE
    assert leg(in_dollars, 'accumulator')['strike_market'] == pytest.approx(
        strike, rel=HN_AXIS_TOLERANCE), 'the two orientations solved different strikes under one law'


def test_a_composed_tarf_carries_an_exposure_profile(tmp_path):
    """The CMC bar: a quoted TARF, booked, run as an exposure simulation - which is what the deal
    is FOR, and the failure mode a base valuation cannot see.

    A deal the engine cannot resolve is SKIPPED: an ERROR in the log, and every number downstream
    computed over a portfolio that quietly lost a trade. On a base valuation that shows up as a
    missing `mtm` row, which the zero-cost gate already reads. On a Credit Monte Carlo it does not
    show up at all - the profile is still a frame of the right shape, and a book of one skipped
    deal is a floor of zeros that looks like a trade deep out of the money. So the claim is made
    unrepresentable: more than one reporting row, every value finite, and DISPERSION across the
    scenarios of each row - which a skipped deal cannot have, because zero has no spread.

    The market is `test_fx_tarf_json`'s own, so the composed deal is priced against the world that
    file's closed-form gates pin, with the FX rate given a simulation model (`GBMAssetPriceModel`,
    the only edit) because a Credit Monte Carlo simulates what a base valuation only discounts.
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
        'fixing_frequency': FIXING_FREQUENCY, 'target': 0.10})
    assert abs(outcome['net']) <= SOLVE_TOLERANCE, outcome['net']
    assert leg(outcome, 'tarf')['strike_market'] < factors['FxRate.EUR']['Spot'] * 1.05

    run = copy.deepcopy(document)
    run['Calc']['Deals']['Deals']['Children'] = [structures.book_node(outcome['deal'])]
    # the grid runs PAST the last settlement on purpose: a Credit Monte Carlo whose horizon stops
    # inside the strip reports one mtm row more than its own report index and refuses on the shape
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
