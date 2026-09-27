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

from test_spine_engine import (
    ACTOR, CLIENT, CLIENT_SET, EQUITY, FACTORS, INDEX, JSON, drained, dump, job, netting_set,
    observe)

START, END, THIRD = (pd.Timestamp(day) for day in ('2024-06-28', '2024-07-01', '2024-07-02'))
BOOK = 'spine-desk'


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
    """The book as a document: one client netting set, the fixture's market at `spot`, both of the
    book's dates on `base`."""
    factors = dict(FACTORS, **EQUITY, **{'FxRate.ZAR': dict(FACTORS['FxRate.ZAR'], Spot=spot)})
    body = json.loads(dump(job(deals=deals, factors=factors, Base_Date=base)))
    body['Calc']['MergeMarketData']['ExplicitMarketData']['System Parameters']['Base_Date'] = \
        json.loads(dump({'day': base}))['day']
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


def booked(deal, quantity, reference, portfolio, price=None, currency=None):
    body = {'action': 'add', 'deal': deal, 'parent_reference': CLIENT_SET, 'quantity': quantity,
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
    new-deal split; live, a move of the rand is the held quantity times the move of its unit; and a
    fill booked with no price, and a settlement against a payment nothing announces, are NAMED
    rather than read as zero.

    Killing mutations: the premium's sign; the rand premium taken at face value; the payment at
    the end's close rather than its own day's; the expiry's basis never released; an expired
    position still standing after its window; a fee shared by the window's trading rather than its
    own day's, which breaks the month; a payoff settling after its window dropped with the position,
    which breaks it again; an expired position read back until it settles, dropped at once, or kept
    past its settlement day; cash placed by its value date rather than its filing; a restatement
    ignored; a late fee shared by the window's holdings, or by its trading, rather than its own
    day's; a position holding nothing dropped with its share; a bare settlement left unknown; an
    unplaceable settlement dropped.
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
