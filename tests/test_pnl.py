"""The desk's P&L, on a real home through the service: two closes marked, the trades, payments,
fees and a partial unwind between them, and the P&L read back per position, per portfolio and
live - held to independent valuations and to its own identity, nothing monkeypatched.
"""
import json
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from derivus import service
from derivus.config import as_json
from derivus_spine import SpineLog, init_home, policy

import rates_world
from test_spine_engine import (
    ACTOR, CLIENT, CLIENT_SET, EQUITY, FACTORS, INDEX, JSON, WORKER_SECONDS, drained, dump,
    holding, job, netting_set, observe)

START, END, THIRD, FOURTH, FIFTH, SIXTH = (pd.Timestamp(day) for day in (
    '2024-06-28', '2024-07-01', '2024-07-02', '2024-07-03', '2024-07-04', '2024-07-05'))
#: The book, named with the '@' a marks job stamps the position it read the book at with.
BOOK = 'spine@desk'


def cashflow(reference, amount, paid):
    return {'Object': 'FixedCashflowDeal', 'Reference': reference, 'Currency': 'ZAR',
            'Discount_Rate': 'ZAR', 'Calendars': None, 'Amount': amount, 'Payment_Date': paid}


LONG = cashflow('CF-A', 1_000_000.0, START + pd.DateOffset(years=2))
EXPIRING = cashflow('CF-P', 100_000.0, pd.Timestamp('2024-06-30'))
NEW = cashflow('CF-B', 500_000.0, START + pd.DateOffset(years=1))
UNPRICED = cashflow('CF-C', 250_000.0, START + pd.DateOffset(years=1))
SPLIT = cashflow('CF-D', 50_000.0, START + pd.DateOffset(years=1))

#: A digital on the equity expiring inside the first window and settling at T+2, inside the second.
BINARY = {'Object': 'EquityBarrierBinaryOption', 'Reference': 'EQ-BIN', 'Currency': 'USD',
          'Discount_Rate': 'USD', 'Equity': 'EQ', 'Equity_Volatility': 'EQ', 'Buy_Sell': 'Buy',
          'Option_Type': 'Call', 'Strike_Price': 100.0, 'Cash_Payoff': 10_000.0,
          'Barrier_Type': 'Up_And_Out', 'Barrier_Price': 1_000.0,
          'Expiry_Date': pd.Timestamp('2024-06-30'), 'Settlement_Date': pd.Timestamp('2024-07-02')}
#: The same digital out of the money at the print, so its settlement moves nothing.
WORTHLESS = dict(BINARY, Reference='EQ-OTM', Strike_Price=110.0)


def document(base=START, spot=18.5, deals=()):
    """`BOOK` as a document: one client netting set, the fixture's market at `spot`, both of the
    book's dates on `base`."""
    factors = dict(FACTORS, **EQUITY, **{'FxRate.ZAR': dict(FACTORS['FxRate.ZAR'], Spot=spot)})
    body = json.loads(dump(job(deals=deals, factors=factors, Base_Date=base)))
    body['Calc']['MergeMarketData']['ExplicitMarketData']['System Parameters']['Base_Date'] = \
        json.loads(dump({'day': base}))['day']
    body['Calc']['Deals']['Reference'] = BOOK
    return body


def unit(deal, base, spot):
    """ONE UNIT of `deal` valued alone on the book's market at `base` and `spot` - the oracle a
    mark is held to."""
    _, out = service.load(document(base, spot, (deal,))).run_job()
    table = as_json(out['Results']['mtm'])['.DataFrame']
    column = {name: position for position, name in enumerate(table['columns'])}
    return next(float(row[column['Value']]) for row in table['data']
                if row[column['Reference']] == deal['Reference'])


@pytest.fixture
def recorded(tmp_path, monkeypatch):
    """A minted spine home with an actor for the appends."""
    home = tmp_path / 'spine'
    init_home(home, ACTOR)
    monkeypatch.setenv('DV_SPINE_HOME', str(home))
    monkeypatch.setenv('DV_SPINE_ACTOR', ACTOR)
    monkeypatch.setenv('DV_HOME', str(tmp_path))
    yield home


@pytest.fixture
def desk(tmp_path):
    body = document()
    body['Calc']['Deals']['Deals']['Children'] = [netting_set(CLIENT_SET, 'CPTY_A')]
    path = tmp_path / 'book.json'
    path.write_text(json.dumps(body, indent=2), newline='\n')
    service.BOOK = service.Book(str(path))
    yield path
    service.BOOK = None
    service.BOOK_DIARY_CACHE.clear()


def booked(deal, quantity, reference, portfolio, price=None, currency=None, parent=CLIENT_SET):
    body = {'action': 'add', 'deal': deal, 'parent_reference': parent, 'quantity': quantity,
            'execution_reference': reference, 'portfolio': portfolio}
    for field, value in (('price', price), ('price_currency', currency)):
        if value is not None:
            body[field] = value
    answer = CLIENT.post('/book/deals', content=dump(body), headers=JSON).json()
    assert answer['written'] is True, answer
    return answer


def settle(body):
    answer = CLIENT.post('/book/transition', content=dump(dict(
        body, status='settled', actor=ACTOR)), headers=JSON)
    assert answer.status_code == 200, answer.text


def fills(home):
    """Every fill body the record holds, read off the platter."""
    log = SpineLog(home)
    try:
        return [log.open_body(frame) for frame in log.frames() if frame['event_type'] == 'fill']
    finally:
        log.close()


def rolled(path, base, spot):
    """The book moved to `base` with the rand at `spot` - both dates and the one value, as a desk's
    morning roll and tick leave them."""
    body = json.loads(path.read_text())
    stamp = json.loads(dump({'day': base}))['day']
    body['Calc']['Calculation']['Base_Date'] = stamp
    market = body['Calc']['MergeMarketData']['ExplicitMarketData']
    market['System Parameters']['Base_Date'] = stamp
    market['Price Factors']['FxRate.ZAR']['Spot'] = spot
    path.write_text(json.dumps(body, indent=2), newline='\n')


def marked():
    answer = CLIENT.post('/book/marks', content=dump({'actor': ACTOR}), headers=JSON)
    assert answer.status_code == 200, answer.text
    assert drained(answer.json())['status'] == 'done'
    return answer.json()


def designated(home, fixings=True, indices=(INDEX,)):
    """The desk's workflow on the record: the close P&L is marked at, and the fixings' source."""
    log = SpineLog(home)
    try:
        policy.declare(log, ACTOR, policy.TIERS_POLICY,
                       {'tiers': [{'name': 'desk', 'four_eyes': True}],
                        'designations': {'pnl': 'official'}})
        if fixings:
            policy.declare(log, ACTOR, policy.FIXINGS_POLICY,
                           {'sources': {index: ['EXCHANGE'] for index in indices}})
    finally:
        log.close()


def closed(date=None):
    answer = CLIENT.post('/book/close', content=dump(dict(
        {'actor': ACTOR}, **({} if date is None else {'date': date}))), headers=JSON)
    assert answer.status_code == 200, answer.text


def closed_and_marked():
    closed()
    return marked()


def with_set(path, reference, counterparty):
    """The book with a second client's netting set, as the file holds one."""
    body = json.loads(path.read_text())
    body['Calc']['Deals']['Deals']['Children'].append(netting_set(reference, counterparty))
    path.write_text(json.dumps(body, indent=2), newline='\n')


def due(reference, day, leg=None):
    """The diary row of `reference` falling due on `day` - the key a settlement names."""
    return next(row for row in CLIENT.get('/book/diary').json()['rows']
                if row['kind'] == 'payment' and row['instrument'] == instrument_of(reference)
                and row['due_date'] == day and (leg is None or row['leg'] == leg))


def instrument_of(reference):
    """The address the record books `reference` under - its terms in the book file, hashed as the
    service hashes them, which a position that has rolled off still has."""
    return service.booked_instruments(service.BOOK.read()[0])[reference]


def held():
    return {row['reference']: row for row in CLIENT.get('/book/positions').json()['positions']}


#: The figures a longer window is the sum of its days in. `existing` and `trading` are not among
#: them: a trade is new against the end of the window it was done in, which a month's end is not.
ADDITIVE = ('premiums', 'payments', 'fees', 'pnl', 'realised', 'unrealised')


def between(start, end=None, **scope):
    answer = CLIENT.get('/book/pnl', params=dict(
        {'start': start}, **({} if end is None else {'end': end}), **scope))
    assert answer.status_code == 200, answer.text
    return answer.json()


def keyed(answer):
    return {(row['instrument'], row['portfolio']): row for row in answer['rows']}


def summed(whole, days, *keys):
    """A longer window's figures against the sum of its days', for the positions named."""
    for key in keys:
        for figure in ADDITIVE:
            assert keyed(whole)[key][figure] == pytest.approx(
                sum(keyed(day)[key][figure] for day in days if key in keyed(day)),
                rel=1e-12, abs=1e-6), (key, figure)


def test_the_book_makes_its_marks_move_plus_its_cash_and_a_month_is_its_days(recorded, desk):
    """THE P&L IS THE MARKS' MOVE PLUS THE CASH, per position, and a longer window is the sum of
    its days. Over three marked closes the book holds a two-year cashflow and a short one paying
    inside the first window, buys a new cashflow for a premium agreed in rand and pays a fee on it,
    unwinds half the long one, pays a rand fee on that, and buys the new cashflow again in another
    portfolio: every unit mark equals the instrument valued ALONE on its close's market; every row
    is its values' move plus its premiums, payments and fees; a premium in rand is crossed at the
    booking's own board and filed with its currency and rate; a payment and a fee are reported at
    the official close standing on their own day; the partial unwind realises against the average
    cost; the expiring cashflow closes at its settlement value in the window its last day falls in,
    its basis released to realised, and stands no more after it; a digital expiring in the first
    window and settling at T+2 in the second is still held between the two and rolls off on its
    settlement day, its payoff counted in the window it settles in, and its twin out of the money
    settles with nothing moved; a fee filed late, dated inside the first window, lands in the second
    and leaves the first as struck, placed by who held the instrument on its own day - a portfolio
    unwound since included - and a restated one counts its correction where it was filed; the two
    portfolios sum to the book; the two days sum to the window over both in every figure but the
    new-deal split; the explain of a day on which only the rand moved leaves, beyond its carry and
    the rand's move off the start's own sensitivity, exactly the carry's own move with the rand;
    live, a move of the rand is the held quantity times the move of its unit; and a fill booked
    with no price, and a settlement against a payment nothing announces, are NAMED rather than read
    as zero.

    Killing mutations: the premium's sign; the rand premium taken at face value; the payment at
    the end's close rather than its own day's; the expiry's basis never released; an expired
    position still standing after its window; a fee shared by the window's trading rather than its
    own day's, which breaks the month; a payoff settling after its window dropped with the position,
    which breaks it again; an expired position read back until it settles, dropped at once, or kept
    past its settlement day; cash placed by its value date rather than its filing; a restatement
    ignored; a late fee shared by the window's holdings, or by its trading, rather than its own
    day's; a position holding nothing dropped with its share; a bare settlement left unknown; an
    unplaceable settlement dropped; an explained move read backwards, the carry never rolled, and
    the sensitivities taken at the end rather than the start.
    """
    refused = CLIENT.post('/book/marks', content=dump({'actor': ACTOR}), headers=JSON)
    assert refused.status_code == 422 and "designates no market for 'pnl'" in refused.text

    log = SpineLog(recorded)
    try:
        policy.declare(log, ACTOR, policy.TIERS_POLICY,
                       {'tiers': [{'name': 'desk', 'four_eyes': True}],
                        'designations': {'pnl': 'official'}})
        policy.declare(log, ACTOR, policy.FIXINGS_POLICY, {'sources': {INDEX: ['EXCHANGE']}})
    finally:
        log.close()

    price_long, price_short, price_new_zar, price_unwind, price_binary, price_split = (
        17_800_000.0, 3_690_000.0, 500_000.0, 9_050_000.0, 5_000.0, 2_500.0)
    booked(LONG, 1.0, 'EXEC-A', BOOK + '/Rates', price_long)
    booked(EXPIRING, 2.0, 'EXEC-P', BOOK + '/FX', price_short)
    booked(BINARY, 1.0, 'EXEC-Q', BOOK + '/FX', price_binary)
    booked(WORTHLESS, 1.0, 'EXEC-O', BOOK + '/FX', 1_000.0)
    booked(SPLIT, 1.0, 'EXEC-D', BOOK + '/Rates', price_split)
    booked(SPLIT, 1.0, 'EXEC-D2', BOOK + '/FX', price_split)
    assert {row['reference']: (row['expired'], row['settles']) for row in held().values()} == {
        reference: (None, None) for reference in ('CF-A', 'CF-P', 'EQ-BIN', 'EQ-OTM', 'CF-D')}
    assert CLIENT.post('/book/close', content=dump({'actor': ACTOR}),
                       headers=JSON).status_code == 200
    first = marked()
    assert first['instruments'] == 5 and first['market'] == 'official'

    rolled(desk, END, 18.0)
    before = desk.read_bytes()
    stranger = CLIENT.post('/book/deals', content=dump({
        'action': 'add', 'deal': NEW, 'parent_reference': CLIENT_SET, 'quantity': 1.0,
        'execution_reference': 'EXEC-EUR', 'portfolio': BOOK + '/FX', 'price': 1.0,
        'price_currency': 'EUR'}), headers=JSON)
    assert stranger.status_code == 422 and 'FxRate.EUR' in stranger.text
    assert desk.read_bytes() == before, 'a refused consideration moved the file'
    booked(NEW, 1.0, 'EXEC-B', BOOK + '/FX', price_new_zar, 'ZAR')
    booked(LONG, -0.5, 'EXEC-A2', BOOK + '/Rates', price_unwind)
    booked(SPLIT, -1.0, 'EXEC-D3', BOOK + '/FX', price_split)
    settle({'subject': instrument_of('CF-B'), 'amount': -250.0, 'asset': 'USD', 'kind': 'fee',
            'reference': 'FEE-1', 'value_date': '2024-07-01'})
    paid = next(row for row in CLIENT.get('/book/diary').json()['rows']
                if row['kind'] == 'payment' and row['due_date'] == '2024-06-30')
    settle({'subject': paid['key'], 'amount': 200_000.0, 'asset': 'ZAR', 'kind': 'payment',
            'reference': 'PAY-1', 'value_date': '2024-06-30'})
    observe(recorded, 104.0, index=INDEX, date=pd.Timestamp('2024-06-30'))
    closed = CLIENT.post('/book/close', content=dump({'actor': ACTOR}), headers=JSON)
    assert closed.status_code == 200, closed.text
    marked()
    book = held()
    assert 'CF-P' not in book, 'expired and paid on its day, it has rolled off'
    assert (book['EQ-BIN']['expired'], book['EQ-BIN']['settles']) == (
        '2024-06-30', '2024-07-02'), 'expired, it is held until the day it settles'
    assert (book['CF-A']['expired'], book['CF-A']['settles']) == (None, None)

    pnl = between('2024-06-28', '2024-07-01')
    assert pnl['complete'] is True and pnl['unknown'] == [] and pnl['currency'] == 'USD'
    assert pnl['realised_method'] == 'average cost'
    rows = keyed(pnl)
    long, short, new = (rows[(instrument_of(reference), BOOK + portfolio)] for reference, portfolio
                        in (('CF-A', '/Rates'), ('CF-P', '/FX'), ('CF-B', '/FX')))

    assert long['unit_start'] == pytest.approx(unit(LONG, START, 18.5), rel=1e-12)
    assert long['unit_end'] == pytest.approx(unit(LONG, END, 18.0), rel=1e-12)
    assert new['unit_end'] == pytest.approx(unit(NEW, END, 18.0), rel=1e-12)
    assert short['unit_start'] == pytest.approx(unit(EXPIRING, START, 18.5), rel=1e-12)
    assert short['unit_end'] is None and short['value_end'] == 0.0, 'paid away, worth nothing'
    for row in pnl['rows']:
        assert row['pnl'] == pytest.approx(row['value_end'] - row['value_start'] + row['premiums']
                                           + row['payments'] + row['fees'], rel=1e-12)
        assert row['pnl'] == pytest.approx(row['existing'] + row['trading'] + row['payments']
                                           + row['fees'], rel=1e-12)

    filed = next(fill for fill in fills(recorded) if fill['execution_reference'] == 'EXEC-B')
    assert (filed['price'], filed['currency'], filed['rate']) == (price_new_zar, 'ZAR', 18.0)
    assert new['premiums'] == pytest.approx(-price_new_zar * 18.0), 'crossed at its own board'
    assert new['fees'] == -250.0
    assert new['trading'] == pytest.approx(new['unit_end'] - price_new_zar * 18.0, rel=1e-12)
    assert (long['quantity_start'], long['quantity_end']) == (1.0, 0.5)
    assert long['premiums'] == pytest.approx(0.5 * price_unwind)
    assert long['realised'] == pytest.approx(0.5 * (price_unwind - price_long)), 'average cost'
    assert long['existing'] == pytest.approx(long['unit_end'] - long['unit_start'], rel=1e-12)
    assert short['payments'] == pytest.approx(2 * 100_000.0 * 18.5), 'at the close of its day'
    assert short['realised'] == pytest.approx(short['payments'] - 2 * price_short), \
        'the expiry realises the basis'
    binary = rows[(instrument_of('EQ-BIN'), BOOK + '/FX')]
    assert (binary['value_end'], binary['payments']) == (0.0, 0.0), 'expired, its payoff not due'
    assert binary['realised'] == pytest.approx(-price_binary)
    assert pnl['total']['pnl'] == pytest.approx(sum(row['pnl'] for row in pnl['rows']))

    by_portfolio = [between('2024-06-28', '2024-07-01', portfolio=BOOK + portfolio)
                    for portfolio in ('/Rates', '/FX')]
    assert [len(part['rows']) for part in by_portfolio] == [2, 5]
    assert sum(part['total']['pnl'] for part in by_portfolio) == pytest.approx(
        pnl['total']['pnl'], rel=1e-12)

    rolled(desk, THIRD, 17.5)
    booked(NEW, 1.0, 'EXEC-B2', BOOK + '/Rates', price_new_zar, 'ZAR')
    settle({'subject': instrument_of('CF-A'), 'amount': -1_000.0, 'asset': 'ZAR', 'kind': 'fee',
            'reference': 'FEE-2', 'value_date': '2024-07-02'})
    payoff, nothing = (next(row for row in CLIENT.get('/book/diary').json()['rows']
                            if row['kind'] == 'payment' and row['instrument'] == instrument_of(ref))
                       for ref in ('EQ-BIN', 'EQ-OTM'))
    settle({'subject': payoff['key'], 'amount': 10_000.0, 'asset': 'USD', 'kind': 'payment',
            'reference': 'PAY-2', 'value_date': '2024-07-02'})
    settle({'subject': nothing['key']})
    settle({'subject': instrument_of('CF-D'), 'amount': -600.0, 'asset': 'USD', 'kind': 'fee',
            'reference': 'FEE-5', 'value_date': '2024-06-29'})
    assert CLIENT.post('/book/close', content=dump({'actor': ACTOR}),
                       headers=JSON).status_code == 200
    marked()
    assert 'EQ-BIN' not in held(), 'on the day it settles it rolls off'
    assert between('2024-06-28', '2024-07-01') == pnl, 'a day struck never moves'

    second, whole = between('2024-07-01', '2024-07-02'), between('2024-06-28', '2024-07-02')
    assert second['complete'] is True and whole['complete'] is True
    assert (instrument_of('CF-P'), BOOK + '/FX') not in keyed(second), 'expired, it stands no more'
    settled = keyed(second)[(instrument_of('EQ-BIN'), BOOK + '/FX')]
    assert (settled['value_start'], settled['value_end']) == (0.0, 0.0)
    assert settled['payments'] == settled['realised'] == pytest.approx(10_000.0), \
        'its payoff counted in the window it settles in'
    assert keyed(second)[(instrument_of('CF-A'), BOOK + '/Rates')]['fees'] == pytest.approx(
        -1_000.0 * 17.5)
    assert [keyed(second)[(instrument_of('CF-D'), BOOK + part)]['fees'] for part in (
        '/Rates', '/FX')] == [-300.0, -300.0], 'filed late, placed by who held it on its own day'
    assert keyed(second)[(instrument_of('CF-D'), BOOK + '/FX')]['reference'] == 'CF-D'

    assert CLIENT.get('/book/marks').json() == {'market': 'official', 'days': [
        {'day': day, 'lsn': lsn} for day, lsn in (
            ('2024-06-28', pnl['start']['lsn']), ('2024-07-01', pnl['end']['lsn']),
            ('2024-07-02', second['end']['lsn']))]}
    told = between('2024-07-01', '2024-07-02', explain=True)['explain']
    assert told['existing'] == pytest.approx(sum(row['existing'] for row in second['rows']),
                                             rel=1e-12)
    rand = next(row for row in told['factors'] if row.get('factor') == 'FxRate.ZAR')
    assert (rand['start'], rand['end']) == (18.0, 17.5)
    assert told['market'] == pytest.approx(rand['pnl'], rel=1e-12), 'nothing else moved'
    assert told['carry'] > 0.0 and told['unknown'] == []
    assert told['residual'] == pytest.approx(told['carry'] * (17.5 - 18.0) / 18.0, rel=1e-6), \
        "what is left is the carry's own move with the rand - second order"
    days = [keyed(pnl), keyed(second)]
    for key, row in keyed(whole).items():
        for figure in ADDITIVE:
            assert row[figure] == pytest.approx(
                sum(day[key][figure] for day in days if key in day), rel=1e-12, abs=1e-6), \
                (key, figure)
        assert row['value_start'] == pytest.approx(days[0].get(key, {}).get('value_start', 0.0))
        assert row['value_end'] == pytest.approx(days[1].get(key, {}).get('value_end', 0.0))

    flat = CLIENT.get('/book/pnl').json()
    assert flat['start']['day'] == '2024-07-02' and flat['end']['live'] is True
    assert flat['complete'] is True
    assert flat['total']['pnl'] == pytest.approx(0.0, abs=1e-6), 'nothing moved since the marks'
    rolled(desk, THIRD, 17.0)
    settle({'subject': instrument_of('CF-B'), 'amount': -100.0, 'asset': 'USD', 'kind': 'fee',
            'reference': 'FEE-4', 'value_date': '2024-07-02'})
    settle({'subject': instrument_of('CF-D'), 'amount': -700.0, 'asset': 'USD', 'kind': 'fee',
            'reference': 'FEE-5', 'value_date': '2024-06-29'})
    moved = keyed(CLIENT.get('/book/pnl').json())
    assert moved[(instrument_of('CF-B'), BOOK + '/FX')]['pnl'] == pytest.approx(
        unit(NEW, THIRD, 17.0) - unit(NEW, THIRD, 17.5), rel=1e-9)
    assert moved[(instrument_of('CF-B'), BOOK + '/Rates')]['fees'] == -100.0, \
        'on the portfolio that traded it that day'
    assert [moved[(instrument_of('CF-D'), BOOK + part)]['fees'] for part in (
        '/Rates', '/FX')] == [-50.0, -50.0], 'a restatement counts its correction where filed'

    booked(UNPRICED, 1.0, 'EXEC-C', BOOK + '/FX')
    settle({'subject': 'f' * 64, 'amount': 5.0, 'asset': 'USD', 'kind': 'payment',
            'reference': 'PAY-X', 'value_date': '2024-07-02'})
    blind = CLIENT.get('/book/pnl').json()
    assert blind['complete'] is False
    assert {'instrument': instrument_of('CF-C'),
            'what': 'fill EXEC-C was booked with no price'} in blind['unknown']
    assert {'instrument': None, 'what': 'settlement PAY-X was filed against a payment this '
            "window's diary does not announce"} in blind['unknown']
    assert next(row for row in blind['rows'] if row['instrument'] == instrument_of('CF-C'))[
        'pnl'] is None

    missing = CLIENT.get('/book/pnl', params={'start': '2024-06-27'})
    assert missing.status_code == 422 and '2024-06-28, 2024-07-01, 2024-07-02' in missing.text


#: A rand strip quoted by hand, as a desk states one - the tenor says what each benchmark is.
STRIP = [{'tenor': '3M', 'quote': 7.41}, {'tenor': '1Y', 'quote': 7.62},
         {'tenor': '2Y', 'quote': 7.94}, {'tenor': '5Y', 'quote': 8.55}]


def test_the_explain_reads_the_market_in_the_quotes_its_curve_is_built_from(recorded, desk):
    """THE EXPLAIN IN QUOTE SPACE. A rand cashflow on a curve solved from its quotes, marked on two
    days between which the book rolled onto the second at its own quotes and the two-year quote
    moved ten basis points: the move is explained on that quote alone - the curve it builds
    explained through it and never beside it - the roll is the carry, and what is left is second
    order in the move.

    Killing mutations: the curve's own factor rows counted beside its quotes, whose knots a roll
    moves and so cannot be matched from one day to the next.
    """
    designated(recorded, fixings=False)

    def curve(rows):
        answer = CLIENT.post('/book/curve', content=dump(
            {'curve': 'ZAR', 'currency': 'ZAR', 'rows': rows}), headers=JSON)
        assert answer.status_code == 200 and answer.json()['written'], answer.text

    curve(STRIP)
    booked(cashflow('CF-Q', 1_000_000.0, START + pd.DateOffset(years=2)), 1.0, 'EXEC-Q',
           BOOK + '/Rates', 15_600_000.0)
    closed_and_marked()
    rolled = CLIENT.post('/book/date', content=dump({'base_date': '2024-07-01'}), headers=JSON)
    assert rolled.status_code == 200, rolled.text
    curve([dict(row, quote=row['quote'] + 0.10) if row['tenor'] == '2Y' else row
           for row in STRIP])
    closed_and_marked()

    told = between('2024-06-28', '2024-07-01', explain=True)
    explain = told['explain']
    assert told['complete'] is True and explain['unknown'] == [] and explain['note'] is None
    assert [(row.get('block'), row.get('quote'), row.get('factor')) for row in
            explain['factors']] == [('InterestRatePrices.ZAR', 'ZAR 2Y', None)], \
        'the one quote that moved, and the curve it builds only through it'
    (moved,) = explain['factors']
    assert moved['move'] == pytest.approx(0.10, abs=1e-12)
    assert explain['market'] == moved['pnl'] < 0.0, 'a higher rate discounts the cashflow more'
    assert explain['existing'] == pytest.approx(told['rows'][0]['existing'], rel=1e-12)
    assert abs(explain['residual']) < 1e-2 * abs(explain['market']), 'second order in the move'


#: A rand strip paying a coupon on the second marked day, on the day lived after the last marks,
#: and a year on.
STRIP_COUPONS = {'Object': 'CFFixedListDeal', 'Reference': 'CFL', 'Currency': 'ZAR',
                 'Discount_Rate': 'ZAR', 'Buy_Sell': 'Buy', 'Cashflows': {'Items': [
                     {'Payment_Date': END, 'Fixed_Amount': 100_000.0},
                     {'Payment_Date': FIFTH, 'Fixed_Amount': 100_000.0},
                     {'Payment_Date': pd.Timestamp('2025-06-30'), 'Fixed_Amount': 100_000.0}]}}

#: The index a rand swap's floating coupons fix on - printed, since a close waits on its resets.
RAND = 'InterestRate.ZAR'


def seasoned(reference, struck=pd.Timestamp('2024-01-01')):
    """A two-year rand swap struck in January, receiving a quarterly floating coupon whose amount is
    the floating pricer's and nobody else's - one of them due on the second marked day."""
    return dict(rates_world.par_swap(reference, 'ZAR', 'ZAR', 'ZAR', 2, 7.5),
                Effective_Date=struck, Maturity_Date=struck + pd.DateOffset(years=2))


def test_a_payment_on_a_marked_day_is_that_day_s_cash_and_out_of_its_marks(recorded, desk):
    """THE CLOSE IS THE END OF THE DAY. The engine values a payment into the marks of the day it
    falls due, so a coupon paid on a marked day would count twice that day - in the mark and as cash
    - and come back out of the next. A strip's coupon, which the diary determines, leaves the day's
    value as the cash it becomes, and a strip bought that day is bought without it. A swap's
    floating coupon leaves the value once a settlement filed by the close says what it moved:
    settled on time it is that day's cash; marked settled with nothing moved it stays in the value
    and is gone the next day; settled late it lands in the day it was filed - three windows on and
    dated a day after the payment included, after the book's diary has dropped the row - and is in
    no later mark. The day being lived rates what it pays as the book as it stands prices it, any
    day before it - marked or not - at the close standing on it, and a coupon falling due on it that
    nothing has settled is still in its value. The explain's carry takes out what the values take out, so what is left is
    second order in the day's move.

    Killing mutations: what a day paid left in its marks; the start's value corrected by what was
    filed by the END; a position traded on a coupon day paid the coupon; a floating coupon's missing
    money named in the window it falls due in rather than the one it leaves the value in; a
    settlement filed late looked for only in the latest diary before it; one against a payment of
    days past still taken out of a later mark; the day being lived rated at the last close, or a
    day before it inside a window ending now at the book as it stands; the explain's carry not
    corrected.
    """
    designated(recorded, indices=(RAND,))
    for day in ('2024-04-01', '2024-04-04', '2024-07-01'):
        observe(recorded, 0.02, index=RAND, date=pd.Timestamp(day))
    booked(STRIP_COUPONS, 1.0, 'EXEC-L', BOOK + '/Rates', 3_600_000.0)
    for number in range(1, 5):
        booked(seasoned('SW-{}'.format(number)), 1.0, 'EXEC-S{}'.format(number), BOOK + '/Rates',
               0.0)
    booked(seasoned('SW-5', pd.Timestamp('2024-01-04')), 1.0, 'EXEC-S5', BOOK + '/Rates', 0.0)
    closed_and_marked()

    rolled(desk, END, 18.0)
    booked(STRIP_COUPONS, 1.0, 'EXEC-L2', BOOK + '/FX', 1_750_000.0)
    floating = {number: due('SW-{}'.format(number), '2024-07-01', 'FloatCashflows')['key']
                for number in range(1, 5)}
    received = {number: 5_000.0 + 100.0 * number for number in range(1, 5)}

    def moved(number, day='2024-07-01'):
        settle({'subject': floating[number], 'amount': received[number], 'asset': 'ZAR',
                'kind': 'payment', 'reference': 'PAY-S{}'.format(number), 'value_date': day})

    settle({'subject': due('CFL', '2024-07-01')['key']})
    moved(1)
    for number in (2, 3, 4):
        settle({'subject': floating[number]})
    closed_and_marked()
    rolled(desk, THIRD, 18.0)
    moved(4)
    closed_and_marked()
    rolled(desk, FOURTH, 18.0)
    closed_and_marked()

    first, second = (between(*window, explain=True) for window in (
        ('2024-06-28', '2024-07-01'), ('2024-07-01', '2024-07-02')))
    third, whole = between('2024-07-02', '2024-07-03'), between('2024-06-28', '2024-07-03')
    where = {name: (instrument_of(name), BOOK + '/Rates')
             for name in ('CFL', 'SW-1', 'SW-2', 'SW-3', 'SW-4')}
    bought = (instrument_of('CFL'), BOOK + '/FX')
    coupon = 100_000.0 * 18.0
    for answer in (first, second, third, whole):
        assert answer['complete'] is True and answer['unknown'] == [], answer['unknown']

    strip = keyed(first)[where['CFL']]
    assert strip['unit_end'] == pytest.approx(unit(STRIP_COUPONS, END, 18.0), rel=1e-12), \
        "the marks value the day's own coupon"
    assert strip['paid_end'] == strip['payments'] == pytest.approx(coupon)
    assert strip['value_end'] == pytest.approx(strip['unit_end'] - coupon, rel=1e-12)
    assert keyed(first)[bought]['payments'] == 0.0, 'bought on its coupon day, without the coupon'
    assert keyed(first)[bought]['value_end'] == pytest.approx(strip['value_end'], rel=1e-12)
    on_time = keyed(first)[where['SW-1']]
    assert on_time['unit_end'] == pytest.approx(unit(seasoned('SW-1'), END, 18.0), rel=1e-12)
    assert on_time['paid_end'] == on_time['payments'] == pytest.approx(received[1] * 18.0)
    for number in (2, 3, 4):
        row = keyed(first)[where['SW-{}'.format(number)]]
        assert (row['paid_end'], row['payments']) == (0.0, 0.0), 'nothing moved by the close'
        assert row['value_end'] == pytest.approx(row['unit_end'], rel=1e-12)

    following = keyed(second)
    assert following[where['CFL']]['value_start'] == pytest.approx(strip['value_end'], rel=1e-12)
    assert following[where['CFL']]['payments'] == following[where['SW-1']]['payments'] == 0.0
    assert following[where['SW-4']]['value_start'] == pytest.approx(
        keyed(first)[where['SW-4']]['value_end'], rel=1e-12), 'filed after the close it opens on'
    assert following[where['SW-4']]['payments'] == pytest.approx(received[4] * 18.0), \
        'lands in the day it was filed'
    assert following[where['SW-2']]['payments'] == following[where['SW-3']]['payments'] == 0.0
    summed(whole, [first, second, third], *where.values(), bought)

    told = first['explain']
    assert told['unknown'] == []
    assert told['residual'] == pytest.approx(
        (told['carry'] + sum(row['quantity_start'] * row['paid_end'] for row in first['rows']))
        * (18.0 - 18.5) / 18.5, rel=1e-6), "what is left is the carry's own move with the rand"
    assert abs(second['explain']['residual']) < 1e-9 * coupon, 'nothing moved the next day'

    rolled(desk, FIFTH, 17.5)
    moved(2, '2024-07-02')
    lived = CLIENT.get('/book/pnl').json()
    assert lived['complete'] is True and lived['unknown'] == [], lived['unknown']
    now = keyed(lived)
    assert now[where['SW-2']]['payments'] == pytest.approx(received[2] * 18.0), \
        'dated a day after its payment and filed three windows late, found and placed'
    assert now[where['SW-2']]['value_end'] == pytest.approx(now[where['SW-2']]['unit_end'],
                                                             rel=1e-12), 'in no later mark'
    assert now[where['CFL']]['paid_end'] == now[where['CFL']]['payments'] == pytest.approx(
        100_000.0 * 17.5), 'the day being lived rated as the book as it stands prices it'
    today = now[(instrument_of('SW-5'), BOOK + '/Rates')]
    assert today['value_end'] == pytest.approx(
        unit(seasoned('SW-5', pd.Timestamp('2024-01-04')), FIFTH, 17.5), rel=1e-12), \
        'due on the day being lived and settled by nothing, it is still in the value'
    assert keyed(between('2024-07-01'))[where['SW-2']]['payments'] == pytest.approx(
        received[2] * 18.0), 'a marked day inside a window ending now is rated at its own close'
    rolled(desk, SIXTH, 17.0)
    assert keyed(CLIENT.get('/book/pnl').json())[where['CFL']]['payments'] == pytest.approx(
        100_000.0 * 18.0), 'a day between the last marks and the one being lived: its own close'


def structure(reference, legs):
    return {'Object': 'StructuredDeal', 'Reference': reference, 'Currency': 'ZAR',
            'Children': [{'Instrument': {'.Deal': leg}} for leg in legs]}


#: One structure paying a leg inside the first window and another in two years; one paying its
#: two legs across the first window's weekend and on its second marked day.
LEGS = {'LEG-A': cashflow('LEG-A', 100_000.0, pd.Timestamp('2024-06-30')),
        'LEG-B': cashflow('LEG-B', 50_000.0, START + pd.DateOffset(years=2)),
        'LEG-C': cashflow('LEG-C', 70_000.0, pd.Timestamp('2024-06-30')),
        'LEG-D': cashflow('LEG-D', 30_000.0, END)}
PAIR = structure('PAIR', [LEGS['LEG-A'], LEGS['LEG-B']])
DONE = structure('DONE', [LEGS['LEG-C'], LEGS['LEG-D']])
#: A structure one of whose legs is restruck while the other, left as it was, pays the day after.
RESTRUCK = structure('RESTRUCK', [cashflow('LEG-E', 40_000.0, THIRD),
                                  cashflow('LEG-F', 20_000.0, pd.Timestamp('2026-06-30'))])


def test_a_structure_held_whole_is_paid_through_its_legs_and_ends_with_the_last(recorded, desk):
    """A STRUCTURE HELD WHOLE IS ONE INSTRUMENT, and what its legs pay is its own. Every leg's
    payment reaches the structure holding it, keyed as the book's own diary keys the leg so a
    settlement filed against it is placed; a structure closes on its LAST leg's day, its basis
    released there, and rolls off the positions once every leg has paid, while one with a leg still
    to pay stands. A leg restruck makes the structure another, and the leg it left as it was is paid
    once, to the structure that held it when it fell due, in any window over the restrike.

    Killing mutations: the legs' rows left with the legs; a structure's last day taken as its
    first leg's; a structure rolled off once any leg has expired; a marks job's own units not read
    as the structures holding the legs; a restruck structure read in one compile with the one it
    replaced; a movement against the leg they share weighed by one of them alone.
    """
    designated(recorded, fixings=False)
    booked(PAIR, 1.0, 'EXEC-PAIR', BOOK + '/Rates', 3_000_000.0)
    booked(DONE, 1.0, 'EXEC-DONE', BOOK + '/Rates', 1_800_000.0)
    booked(RESTRUCK, 1.0, 'EXEC-RE', BOOK + '/FX', 1_100_000.0)
    closed_and_marked()

    rolled(desk, END, 18.0)
    for leg, day in (('LEG-A', '2024-06-30'), ('LEG-C', '2024-06-30'), ('LEG-D', '2024-07-01')):
        settle({'subject': due(leg, day)['key'], 'amount': LEGS[leg]['Amount'], 'asset': 'ZAR',
                'kind': 'payment', 'reference': 'PAY-' + leg, 'value_date': day})
    assert {name: row['expired'] for name, row in held().items()} == {
        'PAIR': None, 'DONE': None, 'RESTRUCK': None}, 'a leg is still paying today'
    replaced = instrument_of('RESTRUCK')
    restruck = CLIENT.post('/book/deals', content=dump({
        'action': 'amend', 'deal_path': held()['RESTRUCK']['deal_paths'][0] + '/1',
        'fields': {'Amount': 25_000.0}, 'reference': 'LEG-F'}), headers=JSON).json()
    assert restruck['written'] is True, restruck
    closed_and_marked()

    first = between('2024-06-28', '2024-07-01')
    assert first['complete'] is True and first['unknown'] == [], first['unknown']
    pair, done = (keyed(first)[(instrument_of(name), BOOK + '/Rates')] for name in ('PAIR', 'DONE'))
    assert pair['value_start'] == pytest.approx(
        unit(LEGS['LEG-A'], START, 18.5) + unit(LEGS['LEG-B'], START, 18.5), rel=1e-12)
    assert pair['value_end'] == pytest.approx(unit(LEGS['LEG-B'], END, 18.0), rel=1e-12)
    assert pair['payments'] == pytest.approx(100_000.0 * 18.5), "its leg's payment is its own"
    assert done['value_end'] == 0.0, 'its last leg paid on its last day'
    assert done['payments'] == pytest.approx(70_000.0 * 18.5 + 30_000.0 * 18.0)
    assert done['realised'] == pytest.approx(done['payments'] - 1_800_000.0), \
        'its basis released there'

    rolled(desk, THIRD, 17.5)
    settle({'subject': due('LEG-E', '2024-07-02')['key'], 'amount': 40_000.0, 'asset': 'ZAR',
            'kind': 'payment', 'reference': 'PAY-LEG-E', 'value_date': '2024-07-02'})
    closed_and_marked()
    standing = held()
    assert 'DONE' not in standing, 'every leg paid, it has rolled off'
    assert standing['PAIR']['expired'] is None, 'a leg still to pay'
    second, both = between('2024-07-01', '2024-07-02'), between('2024-06-28', '2024-07-02')
    assert (instrument_of('DONE'), BOOK + '/Rates') not in keyed(second)
    after = (instrument_of('RESTRUCK'), BOOK + '/FX')
    assert keyed(second)[after]['payments'] == pytest.approx(40_000.0 * 17.5)
    assert both['complete'] is True and both['unknown'] == [], both['unknown']
    summed(both, [first, second], (replaced, BOOK + '/FX'), after)


#: A structure over a rand swap's floating leg and a cashflow.
FLOATING = structure('ST-2', [seasoned('ST2-SW'),
                              cashflow('ST2-CF', 70_000.0, pd.Timestamp('2026-06-30'))])


def test_a_leg_settled_late_is_its_structure_s_and_money_nobody_held_is_named(recorded, desk):
    """WHAT A LEG PAYS IS ITS STRUCTURE'S, late or not. A structure over a floating swap and a
    cashflow has its swap's coupon marked settled with nothing moved at its close, and the money
    filed two windows later, after the book's diary has dropped the row: the earlier diary it is
    found in maps the leg to the structure holding it, so the money is the structure's. And a swap
    bought on the day its coupon falls due, with a settlement saying the book received it, is named
    as money no position here held when it fell due - not shared as a position netting to nothing.

    Killing mutations: the leg the look-back finds left unmapped; a movement nobody held read as a
    flat net in the day's marks.
    """
    designated(recorded, indices=(RAND,))
    for day in ('2024-04-01', '2024-07-01'):
        observe(recorded, 0.02, index=RAND, date=pd.Timestamp(day))
    booked(FLOATING, 2.0, 'EXEC-ST2', BOOK + '/FX', 0.0)
    closed_and_marked()

    rolled(desk, END, 18.0)
    booked(seasoned('SW-CUM'), 1.0, 'EXEC-CUM', BOOK + '/Rates', 0.0)
    settle({'subject': due('SW-CUM', '2024-07-01', 'FloatCashflows')['key'], 'amount': 5_000.0,
            'asset': 'ZAR', 'kind': 'payment', 'reference': 'PAY-CUM', 'value_date': '2024-07-01'})
    leg = due('ST2-SW', '2024-07-01', 'FloatCashflows')['key']
    settle({'subject': leg})
    closed_and_marked()
    rolled(desk, THIRD, 18.0)
    closed_and_marked()
    rolled(desk, FOURTH, 18.0)
    settle({'subject': leg, 'amount': 10_000.0, 'asset': 'ZAR', 'kind': 'payment',
            'reference': 'PAY-ST2', 'value_date': '2024-07-01'})
    closed_and_marked()

    first, late = between('2024-06-28', '2024-07-01'), between('2024-07-02', '2024-07-03')
    assert first['unknown'] == [{'instrument': instrument_of('SW-CUM'), 'what': 'settlement '
                                 'PAY-CUM moved a payment no position held when it fell due'}]
    assert late['complete'] is True and late['unknown'] == [], late['unknown']
    assert keyed(late)[(instrument_of('ST-2'), BOOK + '/FX')]['payments'] == pytest.approx(
        10_000.0 * 18.0), "the leg's late money is its structure's"


PAYING = cashflow('CF-G', 100_000.0, END)
DUE_MONDAY = cashflow('CF-J', 10_000.0, END)


def test_an_amendment_restrikes_the_position_from_the_start_of_its_day(recorded, desk):
    """AN AMENDMENT RESTRIKES THE TRADE FROM THE START OF ITS DAY. A cashflow restruck on the
    morning it pays pays on the terms it became - the payment the book's diary and its settlement
    file announce - and the old terms pay nothing; a restructuring fee on the new terms, dated
    before the day they were struck, falls to the position holding them when it was filed; and the
    month is the sum of its days. A fee dated forward is placed and rated as the record stood when
    it was filed, and its restatement takes back what it placed, where and at what it placed it, so
    neither a position unwound before its date nor the close of that date moves the day it landed
    in; closes restated after their day move no day struck on them; and a day is marked on its own
    close and no other.

    Killing mutations: the amended terms left out of the window's diary; what was held read at the
    window's start, blind to the amendment; the amendment read as that day's trading, the old terms
    paid; a movement rated at the close standing at the end of the window being read; a payment
    the diary determines rated at the latest marks, or at the marks after its day; a fee placed by
    its value date where that is after its filing; a restatement taken back as of the correction;
    marks taken on another day's close.
    """
    designated(recorded, fixings=False)
    booked(LONG, 1.0, 'EXEC-A', BOOK + '/Rates', 17_800_000.0)
    booked(LONG, 1.0, 'EXEC-A2', BOOK + '/FX', 17_800_000.0)
    booked(PAYING, 1.0, 'EXEC-G', BOOK + '/Rates', 1_850_000.0)
    booked(DUE_MONDAY, 1.0, 'EXEC-J', BOOK + '/Rates', 185_000.0)
    booked(EXPIRING, 2.0, 'EXEC-P', BOOK + '/FX', 3_690_000.0)
    closed_and_marked()

    rolled(desk, END, 18.0)
    settle({'subject': due('CF-P', '2024-06-30')['key']})
    settle({'subject': due('CF-J', '2024-07-01')['key']})
    old = instrument_of('CF-G')
    amended = CLIENT.post('/book/deals', content=dump({
        'action': 'amend', 'deal_path': held()['CF-G']['deal_paths'][0],
        'fields': {'Amount': 150_000.0}, 'reference': 'CF-G'}), headers=JSON).json()
    assert amended['written'] is True, amended
    new = instrument_of('CF-G')
    settle({'subject': due('CF-G', '2024-07-01')['key']})
    settle({'subject': new, 'amount': -500.0, 'asset': 'USD', 'kind': 'fee',
            'reference': 'FEE-AM', 'value_date': '2024-06-29'})
    settle({'subject': instrument_of('CF-A'), 'amount': -1_000.0, 'asset': 'ZAR', 'kind': 'fee',
            'reference': 'FEE-FWD', 'value_date': '2024-07-02'})
    closed_and_marked()
    first = between('2024-06-28', '2024-07-01')

    rolled(desk, THIRD, 17.5)
    booked(LONG, -1.0, 'EXEC-A3', BOOK + '/FX', 17_400_000.0)
    settle({'subject': instrument_of('CF-A'), 'amount': -1_200.0, 'asset': 'ZAR', 'kind': 'fee',
            'reference': 'FEE-FWD', 'value_date': '2024-07-02'})
    closed('2024-06-28')
    closed('2024-07-01')
    refused = CLIENT.post('/book/marks', content=dump({'actor': ACTOR}), headers=JSON)
    assert refused.status_code == 422 and \
        'is for 2024-07-01 and the book stands at 2024-07-02' in refused.text, refused.text
    closed_and_marked()

    assert between('2024-06-28', '2024-07-01') == first, 'a day struck never moves'
    second, whole = between('2024-07-01', '2024-07-02'), between('2024-06-28', '2024-07-02')
    for answer in (first, second, whole):
        assert answer['complete'] is True and answer['unknown'] == [], answer['unknown']
    rows = keyed(first)
    assert rows[(new, BOOK + '/Rates')]['payments'] == pytest.approx(150_000.0 * 18.0), \
        'the payment the restruck terms make'
    assert rows[(old, BOOK + '/Rates')]['payments'] == 0.0, 'the old terms pay nothing that day'
    assert rows[(new, BOOK + '/Rates')]['fees'] == -500.0, 'dated before the terms it was for'
    assert [rows[(instrument_of('CF-A'), BOOK + part)]['fees'] for part in ('/Rates', '/FX')] == [
        -9_000.0, -9_000.0], 'dated forward: placed and rated as the record stood when filed'
    assert rows[(instrument_of('CF-P'), BOOK + '/FX')]['payments'] == pytest.approx(
        2 * 100_000.0 * 18.5)
    assert rows[(instrument_of('CF-J'), BOOK + '/Rates')]['payments'] == pytest.approx(
        10_000.0 * 18.0), "a marked day's payment at its own close, restated or not"
    assert [keyed(second)[(instrument_of('CF-A'), BOOK + part)]['fees'] for part in (
        '/Rates', '/FX')] == [9_000.0, -12_000.0], \
        'restated: what it placed taken back where and at what, the correction as filed'
    summed(whole, [first, second], *keyed(whole))


DAYTRADE = cashflow('CF-E', 300_000.0, START + pd.DateOffset(years=1))
CLOSED = cashflow('CF-F', 200_000.0, START + pd.DateOffset(years=1))


def test_a_fee_falls_to_who_dealt_it_and_a_position_traded_to_nothing_is_worth_nothing(recorded,
                                                                                      desk):
    """A FEE FALLS TO WHOEVER DEALT IT. Brokerage dated the day a trade was done, when the trade
    was booked after that day's marks; a day trade's brokerage at T+1, beside a buy in another
    portfolio that day; the invoice for a position closed out the day before; and a custody fee on
    an instrument held long in one portfolio and short in another: each falls to the positions that
    traded what it was filed on - the day's trading, gross, else its holding, whatever its side,
    else what traded it on the record up to the fee - and none is lost. A position traded to
    nothing inside a longer window is worth nothing at its end, its P&L known, and only the new-deal
    split, which wants a mark the end does not carry, is not.

    Killing mutations: a fee nobody traded or held on its day dropped; a day's trading netted; a
    holding weighed by its sign; a position traded to nothing read as unknown at the end of a
    window its marks do not reach; the split counted against the window being complete; a total
    summing what it can.
    """
    designated(recorded, fixings=False)
    booked(LONG, 1.0, 'EXEC-A', BOOK + '/Rates', 17_800_000.0)
    booked(LONG, -1.0, 'EXEC-A4', BOOK + '/Hedge', 17_900_000.0)
    booked(CLOSED, 1.0, 'EXEC-F', BOOK + '/Rates', 3_500_000.0)
    closed_and_marked()

    rolled(desk, END, 18.0)
    booked(NEW, 1.0, 'EXEC-B', BOOK + '/FX', 9_000_000.0)
    settle({'subject': instrument_of('CF-B'), 'amount': -250.0, 'asset': 'USD', 'kind': 'fee',
            'reference': 'FEE-A', 'value_date': '2024-06-28'})
    booked(CLOSED, -1.0, 'EXEC-F2', BOOK + '/Rates', 3_600_000.0)
    closed_and_marked()

    rolled(desk, THIRD, 17.5)
    booked(DAYTRADE, 1.0, 'EXEC-E1', BOOK + '/FX', 5_400_000.0)
    booked(DAYTRADE, -1.0, 'EXEC-E2', BOOK + '/FX', 5_410_000.0)
    booked(DAYTRADE, 1.0, 'EXEC-E3', BOOK + '/Rates', 5_405_000.0)
    settle({'subject': instrument_of('CF-E'), 'amount': -300.0, 'asset': 'USD', 'kind': 'fee',
            'reference': 'FEE-E', 'value_date': '2024-07-03'})
    settle({'subject': instrument_of('CF-F'), 'amount': -400.0, 'asset': 'USD', 'kind': 'fee',
            'reference': 'FEE-F', 'value_date': '2024-07-02'})
    settle({'subject': instrument_of('CF-A'), 'amount': -200.0, 'asset': 'USD', 'kind': 'fee',
            'reference': 'FEE-X', 'value_date': '2024-07-02'})
    closed_and_marked()

    first, second = between('2024-06-28', '2024-07-01'), between('2024-07-01', '2024-07-02')
    whole = between('2024-06-28', '2024-07-02')
    for answer in (first, second, whole):
        assert answer['complete'] is True and answer['unknown'] == [], answer['unknown']
    assert keyed(first)[(instrument_of('CF-B'), BOOK + '/FX')]['fees'] == -250.0
    assert [keyed(second)[(instrument_of('CF-E'), BOOK + part)]['fees'] for part in (
        '/FX', '/Rates')] == [pytest.approx(-200.0), pytest.approx(-100.0)], 'gross, not netted'
    assert keyed(second)[(instrument_of('CF-F'), BOOK + '/Rates')]['fees'] == -400.0
    assert [keyed(second)[(instrument_of('CF-A'), BOOK + part)]['fees'] for part in (
        '/Rates', '/Hedge')] == [-100.0, -100.0], 'held long and short alike'
    out = keyed(whole)[(instrument_of('CF-F'), BOOK + '/Rates')]
    assert (out['quantity_end'], out['value_end']) == (0.0, 0.0), 'traded to nothing'
    assert out['pnl'] == pytest.approx(3_600_000.0 - out['value_start'] - 400.0, rel=1e-12)
    assert (out['existing'], out['trading']) == (None, None), 'the split wants a mark at the end'
    assert whole['total']['existing'] is None and whole['total']['pnl'] is not None
    summed(whole, [first, second], *keyed(whole))


#: A cashflow paying inside the first window, bought after it paid.
LATE = cashflow('CF-H', 50_000.0, pd.Timestamp('2024-06-30'))


def test_money_nobody_can_place_is_named(recorded, desk):
    """NOTHING IS PLACED ON A GUESS. The same digital long with one client and short with another,
    and the same short cashflow held both ways: the cashflow's payment, which the diary determines,
    falls to each position by what it held, but the digital's payoff is two settlements against
    ONE row of an instrument netting to nothing, and a settlement names no agreement - whose each
    is cannot be known, so it is named on the positions rather than dropped. A fee filed on an
    instrument the book never held falls to nobody, and money settled against a payment no position
    held when it fell due belongs to nobody here: each is named.

    Killing mutations: a payoff over a flat net dropped as nothing; a fee nobody holds dropped; a
    settlement nobody held dropped.
    """
    designated(recorded)
    with_set(desk, 'CLIENT_B', 'CPTY_B')
    for sign, parent, portfolio in ((1.0, CLIENT_SET, '/FX'), (-1.0, 'CLIENT_B', '/Hedge')):
        booked(BINARY, sign, 'EXEC-Q' + portfolio, BOOK + portfolio, 5_000.0, parent=parent)
        booked(EXPIRING, sign, 'EXEC-P' + portfolio, BOOK + portfolio, 1_845_000.0, parent=parent)
    closed_and_marked()

    rolled(desk, END, 18.0)
    settle({'subject': due('CF-P', '2024-06-30')['key']})
    observe(recorded, 104.0, index=INDEX, date=pd.Timestamp('2024-06-30'))
    settle({'subject': 'e' * 64, 'amount': -100.0, 'asset': 'USD', 'kind': 'fee',
            'reference': 'FEE-C', 'value_date': '2024-07-01'})
    closed_and_marked()

    rolled(desk, THIRD, 17.5)
    payoff = due('EQ-BIN', '2024-07-02')['key']
    for amount, reference in ((10_000.0, 'PAY-A'), (-10_000.0, 'PAY-B')):
        settle({'subject': payoff, 'amount': amount, 'asset': 'USD', 'kind': 'payment',
                'reference': reference, 'value_date': '2024-07-02'})
    booked(LATE, 1.0, 'EXEC-H', BOOK + '/FX', 0.0)
    settle({'subject': due('CF-H', '2024-06-30')['key'], 'amount': 50_000.0, 'asset': 'ZAR',
            'kind': 'payment', 'reference': 'PAY-H', 'value_date': '2024-06-30'})
    closed_and_marked()

    first, second = between('2024-06-28', '2024-07-01'), between('2024-07-01', '2024-07-02')
    assert first['unknown'] == [
        {'instrument': 'e' * 64, 'what': 'fee FEE-C falls to no position - none under the book '
         'traded or held what it was filed on'}]
    assert [keyed(first)[(instrument_of('CF-P'), BOOK + part)]['payments'] for part in (
        '/FX', '/Hedge')] == [pytest.approx(100_000.0 * 18.5), pytest.approx(-100_000.0 * 18.5)]
    assert second['complete'] is False
    assert {'instrument': instrument_of('EQ-BIN'), 'what': 'settlement PAY-A moved a payment of '
            'positions netting to nothing - a settlement names no agreement, so whose it is is not '
            'known'} in second['unknown']
    assert [keyed(second)[(instrument_of('EQ-BIN'), BOOK + part)]['payments'] for part in (
        '/FX', '/Hedge')] == [None, None]
    assert {'instrument': instrument_of('CF-H'), 'what': 'settlement PAY-H moved a payment no '
            'position held when it fell due'} in second['unknown']


#: An FX forward maturing on the second marked day; the same forward as a leg of a structure
#: paying for two more years; and one maturing a day later that the desk closes out the day before.
FORWARD = {'Object': 'FXForwardDeal', 'Reference': 'FWD', 'Buy_Currency': 'USD',
           'Buy_Amount': 2_000_000.0, 'Buy_Discount_Rate': 'USD', 'Sell_Currency': 'ZAR',
           'Sell_Amount': 100_000.0, 'Sell_Discount_Rate': 'ZAR', 'Settlement_Date': END}
MIXED = structure('ST-F', [dict(FORWARD, Reference='FWD-LEG'),
                           cashflow('ST-F-CF', 50_000.0, pd.Timestamp('2026-06-30'))])
EARLY = dict(FORWARD, Reference='FWD-2', Settlement_Date=THIRD)


def test_a_forward_is_paid_the_legs_its_settlement_date_declares(recorded, desk):
    """A FORWARD SETTLES WHAT ITS TERMS STATE. Its settlement date declares the leg it receives in
    one currency and the leg it pays in the other, so the diary announces both, determined, and the
    day it settles is paid what it settles for - its own value that day, valued alone - held alone
    or as the leg of a structure still standing, that day's mark without it. One closed out the day
    before pays nothing, and no settlement of it is waited on.

    Killing mutations: the forward's legs undeclared, which closes it at nothing with no money in
    its place; the paid leg declared received.
    """
    designated(recorded, fixings=False)
    booked(FORWARD, 1.0, 'EXEC-FWD', BOOK + '/FX', 0.0)
    booked(MIXED, 1.0, 'EXEC-MIX', BOOK + '/FX', 1_000_000.0)
    booked(EARLY, 1.0, 'EXEC-F2', BOOK + '/FX', 0.0)
    closed_and_marked()

    rolled(desk, END, 18.0)
    booked(EARLY, -1.0, 'EXEC-F3', BOOK + '/FX', 180_000.0)
    for reference in ('FWD', 'FWD-LEG'):
        for leg in ('Buy_Amount', 'Sell_Amount'):
            settle({'subject': due(reference, '2024-07-01', leg)['key']})
    closed_and_marked()
    rolled(desk, THIRD, 17.5)
    closed_and_marked()

    first, whole = between('2024-06-28', '2024-07-01'), between('2024-06-28', '2024-07-02')
    for answer in (first, whole):
        assert answer['complete'] is True and answer['unknown'] == [], answer['unknown']
    settles = unit(FORWARD, END, 18.0)
    alone = keyed(first)[(instrument_of('FWD'), BOOK + '/FX')]
    assert alone['payments'] == pytest.approx(settles, rel=1e-9) and alone['value_end'] == 0.0
    held = keyed(first)[(instrument_of('ST-F'), BOOK + '/FX')]
    assert held['payments'] == pytest.approx(settles, rel=1e-9), "its leg's money is its own"
    assert held['value_end'] == pytest.approx(held['unit_end'] - settles, rel=1e-9)
    early = keyed(whole)[(instrument_of('FWD-2'), BOOK + '/FX')]
    assert (early['quantity_end'], early['payments'], early['value_end']) == (0.0, 0.0, 0.0)


def test_a_dead_position_realises_its_cash_alone_and_a_lot_nobody_priced_names_its_own_window(
        recorded, desk):
    """WHAT WAS REALISED IS REALISED ONCE. A cashflow's basis is released on its last day; the desk
    closing the dead position out of the record afterwards realises nothing more than the premium
    that close-out moved. A lot booked with no price leaves the window whose reduction closes
    against it unknown and named, and a priced round trip after it realises what it realised. The
    explain carries a position whose last day fell in its window at nothing, which time alone did.

    Killing mutations: a dead position's closing fill realised against the basis its last day
    already released; its premium left out; the costs' realised figure unknown for good after one
    unpriced reduction; the explain dropping a position that ended in its window.
    """
    designated(recorded, fixings=False)
    booked(EXPIRING, 2.0, 'EXEC-P', BOOK + '/FX', 3_690_000.0)
    booked(NEW, 1.0, 'EXEC-U', BOOK + '/Rates')
    closed_and_marked()

    rolled(desk, END, 18.0)
    settle({'subject': due('CF-P', '2024-06-30')['key']})
    booked(NEW, -1.0, 'EXEC-U2', BOOK + '/Rates', 8_900_000.0)
    closed_and_marked()

    rolled(desk, THIRD, 17.5)
    booked(EXPIRING, -2.0, 'EXEC-P2', BOOK + '/FX', 500.0)
    booked(NEW, 1.0, 'EXEC-B1', BOOK + '/Rates', 8_600_000.0)
    booked(NEW, -1.0, 'EXEC-B2', BOOK + '/Rates', 8_650_000.0)
    closed_and_marked()

    first = between('2024-06-28', '2024-07-01', explain=True)
    second, whole = between('2024-07-01', '2024-07-02'), between('2024-06-28', '2024-07-02')
    dead, lot = (instrument_of('CF-P'), BOOK + '/FX'), (instrument_of('CF-B'), BOOK + '/Rates')
    assert keyed(first)[dead]['realised'] == pytest.approx(2 * 100_000.0 * 18.5 - 2 * 3_690_000.0)
    assert keyed(first)[lot]['realised'] is None
    assert {'instrument': instrument_of('CF-B'), 'what': 'a reduction closed at no price, or '
            'against a lot booked with none, so what it realised is not known'} in first['unknown']
    told = first['explain']
    assert abs(told['residual']) < 1e-4 * abs(told['existing']), 'the ended position is carry'
    assert second['complete'] is True, second['unknown']
    assert keyed(second)[dead]['realised'] == pytest.approx(1_000.0), 'the close-out premium alone'
    assert keyed(second)[lot]['realised'] == pytest.approx(50_000.0), 'a priced round trip'
    assert keyed(whole)[dead]['realised'] == pytest.approx(
        keyed(first)[dead]['realised'] + 1_000.0)


def test_a_ticket_booked_while_the_marks_wait_is_the_next_day_s_and_marks_run_forward(recorded,
                                                                                     desk):
    """THE MARKS CLOSE THE DAY WHERE THEY READ THE BOOK. A ticket booked while the day's marks
    wait behind another job on the one worker was not in the book they read, so it is the next
    business day's: the window after that close is whole, the ticket's premium in it. The last day
    marked again stands in place of its first marks, closing the day where the second read the
    book. And marks run forward: a day behind the last one marked is not marked again.

    Killing mutations: the business day cut where the run was attested rather than where it read
    the book; the first marks of a day standing over a later one; a day behind the last one marked
    marked again.
    """
    designated(recorded, fixings=False)
    booked(LONG, 1.0, 'EXEC-A', BOOK + '/Rates', 17_800_000.0)
    closed()
    barrier = holding()
    try:
        assert barrier.running.wait(WORKER_SECONDS), 'the worker never picked the barrier up'
        submitted = CLIENT.post('/book/marks', content=dump({'actor': ACTOR}), headers=JSON)
        assert submitted.status_code == 200, submitted.text
        booked(NEW, 1.0, 'EXEC-LATE', BOOK + '/FX', 9_000_000.0)
    finally:
        barrier.released.set()
    assert drained(submitted.json())['status'] == 'done'
    rolled(desk, END, 18.0)
    closed_and_marked()

    first = between('2024-06-28', '2024-07-01')
    assert first['complete'] is True and first['unknown'] == [], first['unknown']
    late = keyed(first)[(instrument_of('CF-B'), BOOK + '/FX')]
    assert (late['quantity_start'], late['quantity_end'], late['premiums']) == (
        0.0, 1.0, -9_000_000.0), "the next business day's trade"
    booked(NEW, 1.0, 'EXEC-AGAIN', BOOK + '/Rates', 9_000_000.0)
    marked()
    assert keyed(between('2024-06-28', '2024-07-01'))[(instrument_of('CF-B'), BOOK + '/Rates')][
        'quantity_end'] == 1.0, 'the last day marked again stands in place of its first marks'

    rolled(desk, START, 18.4)
    closed()
    refused = CLIENT.post('/book/marks', content=dump({'actor': ACTOR}), headers=JSON)
    assert refused.status_code == 422 and 'marks run forward' in refused.text, refused.text


def test_a_seat_marking_one_book_marks_it_and_the_hub_attests_the_marks(recorded, desk):
    """THE MARKS ARE ADMITTED OVER THE BOOK THEY VALUE. A standing run asks `mark` over the book its
    job values, and a marks job is filed under the marks' own name, so `/book/marks` hands the
    queue the book it marks: a seat granted `mark` over this book alone marks it, one granted it
    over another book is turned away naming this one, and the run is attested by the hub in the
    writer's own voice rather than by the seat that asked.

    Killing mutation: the marks job admitted over its own name, which nothing but `*` reaches.
    """
    from derivus_spine.capability import CAPABILITIES_POLICY, canonical_document

    designated(recorded, fixings=False)
    booked(LONG, 1.0, 'EXEC-A', BOOK + '/Rates', 17_800_000.0)
    closed()
    log = SpineLog(recorded)
    try:
        blob = log.store.put(canonical_document({'grants': [
            {'subject': ACTOR, 'verb': 'admin', 'book': '*'},
            {'subject': 'subject-marks', 'verb': 'mark', 'book': BOOK},
            {'subject': 'subject-elsewhere', 'verb': 'mark', 'book': 'elsewhere'}], 'read': []}))
        log.append('policy_declared', {'policy': CAPABILITIES_POLICY, 'blob': blob}, actor=ACTOR,
                   blob_refs=(blob,))
    finally:
        log.close()

    refused = CLIENT.post('/book/marks', content=dump({'actor': 'subject-elsewhere'}),
                          headers=JSON)
    assert refused.status_code == 422, refused.text
    assert "no mark scope over '{}'".format(BOOK) in refused.json()['detail'], refused.text
    answer = CLIENT.post('/book/marks', content=dump({'actor': 'subject-marks'}), headers=JSON)
    assert answer.status_code == 200, answer.text
    assert drained(answer.json())['status'] == 'done'
    log = SpineLog(recorded)
    try:
        assert [frame['actor'] for frame in log.frames()
                if frame['event_type'] == 'run_completed'] == ['writer']
    finally:
        log.close()
    assert [row['day'] for row in CLIENT.get('/book/marks').json()['days']] == ['2024-06-28']
