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

"""The diary, the derived key and the settlement exporter, driven through the real compile.

THE DIARY IS THE COMPILE'S OWN SCHEDULE, and the gate that says so is arithmetic rather than
assertion by adjective: the same book is priced under a credit Monte Carlo with `Generate_Cashflows`
on and every payment the diary announces is required to equal the cashflow the run realized on that
day, in float64 where the engine's own accumulation order is the only difference left.

  * ONE WALK. `utils.walk_schedules` is what binds a deal's schedules and what the diary reads them
    at, so the leg names a settlement reference is built from cannot drift from the binding - pinned
    here against a two-leg swap's own legs.
  * ONE SPELLING OF A PAYMENT. The amount is `pricing.fixed_payments`, the line the pricer
    discounts, so a bond's principal, a compounded sub-period and a sold leg all agree with the
    engine's own realized cashflow to float64 rounding.
  * A NULL AMOUNT IS NEVER A ZERO. A floating leg is not determined until its resets fix, and the
    exporter refuses one BY NAME rather than instructing a payment of nothing.
  * THE KEY IS THE ROW'S OWN DATE: a book rolled PAST a coupon keeps every surviving row's key and
    loses only the row that was paid.
  * THE CLOSE CHECK IS THE CATCH-UP RULE as a read - a payment with no settlement transition
    against its key blocks the close, and filing one clears it.
  * THE DIARY NEVER RUNS ON THE POLL PATH: two asks over an unmoved book are one queued job, and a
    booking between them is a second.
"""
import ast
import inspect
import json
import os
import sys

import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rates_world
import derivus
from derivus import diary, service, spine, utils
from derivus.config import CustomJsonEncoder
from derivus_spine import SpineLog, capability, init_home, policy, vocabulary

ACTOR = 'subject-desk-one'
BASE = rates_world.BASE
NOTIONAL, QUOTE = 1_000_000.0, 7.0
JSON = {'content-type': 'application/json'}
CLIENT_SET = 'CLIENT_A'
CLIENT = TestClient(service.app)
CSA = {'.CreditSupportList': [[0.0, 0.0]]}

FACTORS = dict(rates_world.market('ZAR', {'ZAR': ([0.0, 5.0], [QUOTE / 100.0] * 2)}, 'ZAR'),
               **rates_world.market('USD', {'USD': ([0.0, 5.0], [0.02, 0.02])}, 'USD'))
HW1F = {'Alpha': 0.1, 'Lambda': 0.0, 'Quanto_FX_Correlation': 0.0,
        'Sigma': utils.Curve([], [[1.0 / 365.0, 0.01], [5.0, 0.01]]),
        'Quanto_FX_Volatility': utils.Curve([], [[1.0 / 365.0, 0.0], [5.0, 0.0]])}
COUPONS = [BASE + pd.DateOffset(months=months) for months in (6, 12, 18, 24)]


#: Every coupon is PAID two days after it accrues, so the accrual end and the payment day are
#: different columns of the same row and a diary reading the wrong one is caught rather than lucky.
LAG = pd.DateOffset(days=2)


def coupon(start, end, pay, principal=0.0):
    """One `CFFixedInterestListDeal` item: a rate coupon, and a principal repaid WITH it where one
    is named - the two cells of the row a bond's last coupon fills."""
    return {'Payment_Date': pay, 'Notional': NOTIONAL, 'Rate': utils.Percent(QUOTE),
            'Accrual_Start_Date': start, 'Accrual_End_Date': end, 'Accrual_Day_Count': 'ACT_365',
            'Fixed_Amount': principal, 'Discounted': 'No',
            'Accrual_Year_Fraction': utils.DayCount.accrual(
                start, (end - start).days, utils.DayCount.code('ACT_365')),
            'FX_Reset_Date': None, 'Known_FX_Rate': 0.0}


def cashflow_list(reference, items, compounding='No', buy='Buy'):
    return {'Object': 'CFFixedInterestListDeal', 'Reference': reference, 'Currency': 'ZAR',
            'Discount_Rate': 'ZAR', 'Buy_Sell': buy, 'Description': '', 'Settlement_Date': None,
            'Settlement_Amount': 0.0, 'Settlement_Style': 'Physical', 'Is_Defaultable': 'No',
            'Settlement_Amount_Is_Clean': 'Yes', 'Repo_Rate': '', 'Recovery_Rate': '',
            'Survival_Probability': '', 'Investment_Horizon': None, 'Issuer': '',
            'Settlement_Rate': '', 'Calendars': None, 'Rate_Currency': '',
            'Cashflows': {'Compounding': compounding, 'Items': items}}


def fixed_leg(reference='FIXLEG', principal=0.0, compounding='No', buy='Buy'):
    """A fixed cashflow list - four semi-annual coupons quoted by rate, every amount determined by
    the terms alone, which is what makes it comparable to a run's realized cashflows. `principal`
    is repaid on the row the last coupon sits on, which is what a bond does."""
    items = [coupon(start, end, end + LAG) for start, end in zip([BASE] + COUPONS[:-1], COUPONS)]
    items[-1]['Fixed_Amount'] = principal
    return cashflow_list(reference, items, compounding, buy)


def sub_period_leg(reference='SUBS', compounding='No'):
    """Two accrual sub-periods paying on ONE day - a quarterly-accrued, annually-paid coupon. The
    leg pays one payment on that day and the diary has one row to put it on."""
    return cashflow_list(reference, [coupon(start, end, COUPONS[1] + LAG) for start, end in
                                     ((BASE, COUPONS[0]), (COUPONS[0], COUPONS[1]))], compounding)


def swap(reference='SWAP1'):
    return rates_world.par_swap(reference, 'ZAR', 'ZAR', 'ZAR', 2, QUOTE)


def netting_set(reference, counterparty, deals=()):
    return {'Instrument': {'.Deal': {
        'Object': 'NettingCollateralSet', 'Reference': reference, 'Netted': 'True',
        'Collateralized': 'False', 'Agreement_Currency': 'ZAR', 'Balance_Currency': 'ZAR',
        'Funding_Rate': 'ZAR', 'Liquidation_Period': 0.0, 'Settlement_Period': 0.0,
        'Credit_Support_Amounts': {
            'Counterparty': counterparty, 'Received_Threshold': CSA, 'Posted_Threshold': CSA,
            'Independent_Amount': CSA, 'Minimum_Received': CSA, 'Minimum_Posted': CSA}}},
        'Children': [{'Instrument': {'.Deal': deal}} for deal in deals]}


EQUITY = {
    'EquityPrice.EQ': {'Spot': 100.0, 'Currency': 'ZAR', 'Interest_Rate': 'ZAR', 'Issuer': '',
                       'Respect_Default': 'No', 'Jump_Level': 0.0},
    'DividendRate.EQ': {'Currency': 'ZAR', 'Floor': None,
                        'Curve': utils.Curve([], [[0.0, 0.02], [5.0, 0.02]])},
    'EquityPriceVol.EQ': {'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness',
                          'Surface': utils.Curve([], [[m, t, 0.25] for m in (0.6, 1.0, 1.4)
                                                      for t in (0.02, 2.0)])},
    'FXVol.USD.ZAR': {'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness',
                      'Surface': utils.Curve([], [[m, t, 0.15] for m in (0.6, 1.0, 1.4)
                                                  for t in (0.02, 2.0)])}}

#: The day a monitoring row sits on, thirty days behind the book's own.
WATCHED = BASE - pd.DateOffset(days=30)

#: A barrier whose monitoring days sit behind the base date, and a physically settled option - the
#: two shapes whose monitoring and expiry days a reval-date ladder would announce as payments.
BARRIER = {'Object': 'EquityBarrierOption', 'Reference': 'BR', 'Currency': 'ZAR',
           'Payoff_Currency': 'ZAR', 'Equity': 'EQ', 'Dividends': 'EQ', 'Discount_Rate': 'ZAR',
           'Equity_Volatility': 'EQ', 'Buy_Sell': 'Buy', 'Option_Type': 'Call',
           'Strike_Price': 100.0, 'Units': 1.0, 'Cash_Rebate': 0.0, 'Barrier_Price': 115.0,
           'Expiry_Date': BASE + pd.DateOffset(days=365), 'Barrier_Type': 'Up_And_Out',
           'Barrier_Monitoring_Frequency': pd.DateOffset(days=0), 'Barrier_Dates': []}
FX_OPTION = {'Object': 'FXOptionDeal', 'Reference': 'FXO', 'Currency': 'ZAR',
             'Underlying_Currency': 'USD', 'Underlying_Amount': 1_000_000.0, 'Buy_Sell': 'Buy',
             'Option_Type': 'Call', 'Strike_Price': 1.0 / 18.5, 'Option_Style': 'European',
             'Settlement_Style': 'Physical', 'Discount_Rate': 'ZAR', 'FX_Volatility': 'USD.ZAR',
             'Expiry_Date': BASE + pd.DateOffset(days=60), 'Forward_Price_Date': None,
             'Trade_Date': None, 'Delivery_Date': None, 'Structure_Reference': ''}
OPTION = {'Object': 'EquityOptionDeal', 'Reference': 'EQO', 'Currency': 'ZAR', 'Equity': 'EQ',
          'Dividends': 'EQ', 'Discount_Rate': 'ZAR', 'Equity_Volatility': 'EQ', 'Buy_Sell': 'Buy',
          'Option_Style': 'European', 'Option_On_Forward': 'No', 'Forward_Price_Date': None,
          'Option_Type': 'Call', 'Strike_Price': 100.0, 'Units': 1.0, 'Payoff_Type': 'Standard',
          'Settlement_Style': 'Physical', 'Payoff_Currency': 'ZAR',
          'Expiry_Date': BASE + pd.DateOffset(days=60)}


BARRIER_WATCHED = dict(BARRIER, Barrier_Dates=[[WATCHED, 108.5]])


def job(nodes, factors=None, **calculation):
    """A job document as the objects a market data file holds, with a Hull-White model declared so
    the same book runs as a credit Monte Carlo as well as a base valuation."""
    return {'Calc': {
        'Calculation': dict({'Object': 'BaseValuation', 'Base_Date': BASE, 'Currency': 'ZAR',
                             'MCMC_Simulations': 1, 'Random_Seed': 1}, **calculation),
        'Deals': {'Reference': 'diary-desk', 'Deals': {'Children': nodes}},
        'MergeMarketData': {'MarketDataFile': '', 'ExplicitMarketData': {
            'System Parameters': {'Base_Currency': 'ZAR', 'Base_Date': BASE},
            'Model Configuration': {'.ModelParams': {'modelfilters': {}, 'modeldefaults': {
                'InterestRate': 'HullWhite1FactorInterestRateModel'}}},
            'Price Models': {'HullWhite1FactorInterestRateModel.ZAR': dict(HW1F)},
            'Bootstrapper Configuration': {'InterestRateCurveParameters': {}},
            'Valuation Configuration': {}, 'Price Factors': factors or FACTORS}}}}


def dump(document):
    return json.dumps(document, cls=CustomJsonEncoder)


def node(deal):
    return {'Instrument': {'.Deal': deal}}


@pytest.fixture
def unrecorded(tmp_path, monkeypatch):
    """A desk with NO spine home, made explicit so a developer box carrying the variable cannot
    make this file lie."""
    monkeypatch.delenv('DV_SPINE_HOME', raising=False)
    monkeypatch.delenv('DV_SPINE_ACTOR', raising=False)
    monkeypatch.setenv('DV_HOME', str(tmp_path))
    yield tmp_path


@pytest.fixture
def recorded(tmp_path, monkeypatch):
    """A minted spine home with an actor for the appends."""
    home = tmp_path / 'spine'
    init_home(home, ACTOR)
    monkeypatch.setenv('DV_SPINE_HOME', str(home))
    monkeypatch.setenv('DV_SPINE_ACTOR', ACTOR)
    monkeypatch.setenv('DV_HOME', str(tmp_path))
    yield home


def serving(tmp_path, nodes, factors=None):
    """The live book this gate serves, taken down by `closed` after it."""
    path = tmp_path / 'book.json'
    path.write_text(json.dumps(json.loads(dump(job(nodes, factors))), indent=2), newline='\n')
    service.BOOK = service.Book(str(path))
    service.BOOK_DIARY_CACHE.clear()
    return path


@pytest.fixture(autouse=True)
def closed():
    yield
    service.BOOK = None
    service.BOOK_DIARY_CACHE.clear()


def settle(home, key):
    """The settlement fact filed against one diary row.

    `Context.apply_lifecycle` files the three LIFECYCLE facts and refuses this one by name: a
    settlement is an operational fact rather than the holder's act, the world's observation or a
    ruling, so it is appended as itself.
    """
    log = SpineLog(home)
    try:
        return log.append('status_transition', {'subject': key, 'status': diary.SETTLED},
                          actor=ACTOR, book='diary-desk')
    finally:
        log.close()


def at(home, event_type=None):
    """The record's head, or the positions of one event type - what a gate asserts moved, or did
    not."""
    log = SpineLog(home)
    try:
        return (log.head()[0] if event_type is None else
                [frame['lsn'] for frame in log.frames() if frame['event_type'] == event_type])
    finally:
        log.close()


def diary_rows(**params):
    answer = CLIENT.get('/book/diary', params=params)
    assert answer.status_code == 200, answer.text
    return answer.json()


def payments(rows):
    return [row for row in rows if row['kind'] == diary.PAYMENT]


# --------------------------------------------------------------------------------------------
# The diary is the same schedule.

def realized(deal, tmp_path):
    """The book priced under a credit Monte Carlo with `Generate_Cashflows` on, in float64: what
    the engine says it actually paid, per day."""
    run = dict(job([])['Calc']['Calculation'], Object='CreditMonteCarlo', Batch_Size=32,
               Simulation_Batches=1, Time_grid='0d 24m(3m)', Deflation_Interest_Rate='ZAR',
               Generate_Cashflows='Yes')
    context = derivus.Context().load_json((dump(job([node(deal)], **run)), 'diary-cashflows'))
    _, out = derivus.run_cmc(context.current_cfg, prec=torch.float64)
    frame = out['Results']['cashflows']['ZAR']
    return {str(day.date()): float(frame.loc[day].iloc[0]) for day in frame.index}


@pytest.mark.parametrize('name,deal', [
    ('plain', fixed_leg('PLAIN')),
    ('a principal repaid with the last coupon', fixed_leg('BOND', principal=NOTIONAL)),
    ('two sub-periods on one pay day', sub_period_leg('SUBS')),
    ('two sub-periods COMPOUNDED', sub_period_leg('COMP', compounding='Yes')),
    ('sold', fixed_leg('SOLD', buy='Sell'))])
def test_every_payment_the_diary_announces_is_the_run_s_own_cashflow(name, deal, unrecorded,
                                                                     tmp_path):
    """GATE 5. The diary and the pricer spell a payment ONCE - `pricing.fixed_payments` - so the
    assertion is arithmetic on every shape a fixed leg comes in: a coupon carrying a rate AND a
    principal, two accrual sub-periods paying on one day, the same two compounded, and a sold leg.
    The engine's own realized cashflow and the diary's announced amount agree to float64 rounding.

    Killing mutations, both of which the plain fixture cannot see: the fixed amount REPLACING the
    rate coupon rather than summing with it (a bond's last payment reads as its principal alone),
    and the leg's `Compounding` flag ignored (two sub-periods read as their simple sum).
    """
    serving(tmp_path, [node(deal)])
    rows = payments(diary_rows()['rows'])
    assert rows and all(row['determined'] for row in rows), 'a fixed leg determines every coupon'
    assert len({row['due_date'] for row in rows}) == len(rows), 'one PAYMENT per pay day'

    paid = realized(deal, tmp_path)
    announced = {row['due_date']: row['amount'] for row in rows}
    assert set(paid) <= set(announced), (sorted(paid), sorted(announced))
    worst = max(abs(announced[day] - amount) for day, amount in paid.items())
    assert len(paid) == len(rows) and worst < 1e-9, (name, len(paid), len(rows), worst)


def test_the_diary_reads_the_legs_the_binding_binds(unrecorded, tmp_path):
    """The `leg` half of a derived key is a WALK ORDER, so it is pinned: a two-leg swap's schedules
    are reached at the names the binding reaches them at, and at no others.

    Killing mutation: `walk_schedules` recursing dict values without their keys, which renames
    every leg to the empty string and silently renumbers every settlement reference ever filed.
    """
    serving(tmp_path, [node(swap())])
    legs = {row['leg'] for row in diary_rows()['rows']}
    assert legs == {'FixedCashflows', 'FloatCashflows', 'FloatCashflows.Resets', diary.EXPIRY_LEG}

    context = service.load(service.BOOK.read()[0])
    compiled, _ = diary._compiled(context)
    walked = {leg for deal in compiled.netting_sets.deals()
              for leg, _ in utils.walk_schedules(deal.Factor_dep)}
    assert walked == {'FixedCashflows', 'FloatCashflows'}
    assert all(schedule.bound is not None for deal in compiled.netting_sets.deals()
               for _, schedule in utils.walk_schedules(deal.Factor_dep)), 'the walk missed a bind'


# --------------------------------------------------------------------------------------------
# The derived key.

def test_a_derived_key_is_the_same_key_thirty_days_later(unrecorded, tmp_path):
    """GATE 6. The key is the row's OWN DATE and never its position, so a book rolled PAST a coupon
    keeps every surviving row's key on the payment it named and loses only the one that was paid.

    Killing mutation: the row's position in the schedule as the key's third input. A schedule drops
    the rows it has paid, so the positions renumber under the settlement references already filed -
    a transition settling February reads, after the roll, as settling August.
    """
    serving(tmp_path, [netting_set(CLIENT_SET, 'CPTY_A', [fixed_leg()])])
    at_base = {row['key']: (row['leg'], row['kind'], row['due_date'])
               for row in diary_rows()['rows'] if row['key']}
    assert at_base, 'nothing was keyed - the record canonicaliser is not installed'
    first = str((COUPONS[0] + LAG).date())
    assert first in {shown[2] for shown in at_base.values()}

    moved = CLIENT.post('/book/date', content=dump(
        {'base_date': BASE + pd.DateOffset(months=7)}), headers=JSON)
    assert moved.status_code == 200, moved.text
    at_moved = {row['key']: (row['leg'], row['kind'], row['due_date'])
                for row in diary_rows()['rows'] if row['key']}

    assert len(at_moved) == len(at_base) - 1, 'the roll passed exactly one coupon'
    assert set(at_moved) < set(at_base), 'a surviving row was renamed by the roll'
    assert all(at_base[key] == at_moved[key] for key in at_moved), 'a key moved to another payment'
    gone = set(at_base) - set(at_moved)
    assert [at_base[key][2] for key in gone] == [first], 'the paid coupon is what left'


def test_a_derived_key_attaches_a_settlement_fact(recorded, tmp_path):
    """GATE 7. A cashflow key is 64 lowercase hex through the record's own canonicaliser, so a
    `status_transition` names one with NO new field kind - and the diary read after it shows the
    row settled.

    Killing mutation: the key built by string formatting rather than `canonical_bytes`, which
    changes under key order and stops passing `vocabulary.is_hash`.
    """
    serving(tmp_path, [netting_set(CLIENT_SET, 'CPTY_A')])
    booked = CLIENT.post('/book/deals', content=dump(
        {'action': 'add', 'deal': fixed_leg(), 'parent_reference': CLIENT_SET,
         'quantity': NOTIONAL, 'execution_reference': 'EXEC-DIARY-1'}), headers=JSON).json()
    assert booked['written'] is True, booked

    row = sorted(payments(diary_rows()['rows']), key=lambda found: found['due_date'])[0]
    assert vocabulary.is_hash(row['key']) and row['state'] == diary.DUE
    assert row['key'] == spine.cashflow_key(
        row['instrument'], row['leg'], row['kind'], row['due_date'])

    settle(recorded, row['key'])
    after = {found['key']: found['state'] for found in diary_rows()['rows']}
    assert after[row['key']] == diary.SETTLED
    assert sum(1 for state in after.values() if state == diary.SETTLED) == 1


# --------------------------------------------------------------------------------------------
# What is not determined, and what the exporter will not say.

def test_a_floating_amount_is_null_and_the_exporter_refuses_it_by_name(unrecorded, tmp_path):
    """GATE 8. A floating leg's coupon is not determined until its resets fix, so its row carries
    `amount: null` and the exporter REFUSES it - a settlement file saying 0.0 is an instruction to
    pay nothing, which is a wrong payment rather than a missing one.

    A settlement file is struck FOR a day, so `due_before` has no default and a payment past it
    never travels: an exporter with no cutoff instructs the whole book.

    Killing mutations: the determined flag taken from the fixed amount rather than from whether the
    schedule has resets left to fix, which exports every floating coupon as a zero; the cutoff
    ignored, which settles a payment years early; and a row naming no currency totalled anyway,
    which is money going nowhere.
    """
    serving(tmp_path, [node(swap())])
    rows = payments(diary_rows()['rows'])
    floating = [row for row in rows if row['leg'] == 'FloatCashflows']
    fixed = [row for row in rows if row['leg'] == 'FixedCashflows']
    assert floating and fixed
    assert all(row['amount'] is None and row['determined'] is False for row in floating)
    assert all(row['amount'] is not None and row['determined'] is True for row in fixed)

    ever = '2099-01-01'
    with pytest.raises(ValueError) as refusal:
        diary.export_settlements(rows, 'a' * 64, ever)
    assert 'not determined' in str(refusal.value) and floating[0]['leg'] in str(refusal.value)

    exported = diary.export_settlements(fixed, 'a' * 64, ever)
    assert exported['values_hash'] == 'a' * 64 and len(exported['rows']) == len(fixed)
    assert exported['totals']['ZAR'] == pytest.approx(sum(row['amount'] for row in fixed))

    with pytest.raises(ValueError) as unnamed:
        diary.export_settlements([dict(fixed[0], currency=None)], 'a' * 64, ever)
    assert 'no currency' in str(unnamed.value)

    cutoff = min(row['due_date'] for row in fixed)
    trimmed = diary.export_settlements(fixed, 'a' * 64, cutoff)
    assert [row['due_date'] for row in trimmed['rows']] == [cutoff], 'a later payment travelled'
    assert trimmed['due_before'] == cutoff
    assert diary.export_settlements(fixed, 'a' * 64, '1999-01-01')['rows'] == []


def test_the_exporter_reads_the_diary_and_the_official_market_and_nothing_else(unrecorded):
    """GATE 9. Provable two ways: the signature takes the rows and ONE market hash and no other
    object, and the module reaches neither the service nor a Context to find anything else.

    Killing mutation: an import of `service` or of `Context` into the diary module - the exporter
    could then read a live book and the file it wrote would no longer be a function of its
    arguments.
    """
    assert list(inspect.signature(diary.export_settlements).parameters) == [
        'rows', 'official_values_hash', 'due_before']

    source = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'derivus', 'diary.py')
    absolute, relative = set(), set()
    for found in ast.walk(ast.parse(open(source, encoding='utf-8').read())):
        if isinstance(found, ast.Import):
            absolute.update(alias.name.split('.')[0] for alias in found.names)
        elif isinstance(found, ast.ImportFrom):
            (relative if found.level else absolute).add((found.module or '').split('.')[0])
    assert absolute <= {'collections', 'numpy', 'pandas'}, absolute
    assert relative <= {'spine', 'pricing', 'instruments', 'utils', 'calculation',
                        'schema'}, relative
    assert 'service' not in relative and 'Context' not in relative


# --------------------------------------------------------------------------------------------
# The close check, and the poll path.

def test_the_close_check_is_the_catch_up_rule(recorded, tmp_path):
    """GATE 13. A close on a day is legal when every diary entry due on or before it has its fact.
    A coupon with no settlement transition against its key blocks it by name; filing one clears it.

    Killing mutation: the check reading only fixings, which passes a close over every payment the
    desk never settled.
    """
    serving(tmp_path, [netting_set(CLIENT_SET, 'CPTY_A')])
    booked = CLIENT.post('/book/deals', content=dump(
        {'action': 'add', 'deal': fixed_leg(), 'parent_reference': CLIENT_SET,
         'quantity': NOTIONAL, 'execution_reference': 'EXEC-CLOSE-1'}), headers=JSON).json()
    assert booked['written'] is True, booked

    day = str((COUPONS[0] + LAG + pd.DateOffset(days=1)).date())
    verdict = CLIENT.get('/book/close/check', params={'date': day}).json()
    assert verdict['legal'] is False and verdict['due'] == 1
    outstanding = verdict['outstanding'][0]
    assert outstanding['kind'] == diary.PAYMENT
    assert outstanding['due_date'] == str((COUPONS[0] + LAG).date())

    settle(recorded, outstanding['key'])
    cleared = CLIENT.get('/book/close/check', params={'date': day}).json()
    assert cleared['legal'] is True and cleared['outstanding'] == []
    assert cleared['due'] == verdict['due'], 'the row is answered, not dropped'


def test_the_close_check_and_reconcile_are_a_404_on_a_box_that_records_nothing(unrecorded,
                                                                               tmp_path):
    """A box with no home has no record to check against, and says so rather than answering an
    empty diary as though every fact were in. The diary itself still reads: it is the book's own
    schedule and needs no record at all."""
    serving(tmp_path, [node(fixed_leg())])
    assert CLIENT.get('/book/close/check', params={'date': '2030-01-01'}).status_code == 404
    assert CLIENT.get('/book/reconcile').status_code == 404
    assert diary_rows()['rows'], 'the diary is the book, not the record'


def test_a_print_nobody_ordered_a_source_for_leaves_the_read_standing(recorded, tmp_path):
    """A READ REFUSES NOTHING. An operations desk prints an index this book names nowhere and the
    `fixings` policy orders no source for; the diary and the close check answer, and a row whose
    own index is unresolved stands DUE with the reason on it rather than taking the verb down.

    Killing mutation: the read asking for every index the log holds prints of - `spine.fixings`
    with no `indices` - which raises the record's own refusal out of a GET and 500s both verbs.
    """
    serving(tmp_path, [node(fixed_leg()), node(swap())])
    log = SpineLog(recorded)
    try:
        policy.declare(log, ACTOR, policy.FIXINGS_POLICY, {'sources': {'FxRate.ZAR': ['ECB']}})
        log.append('fixing_observed', {'index': 'FxRate.ZAR', 'date': str(BASE.date()),
                                       'source': 'SARB', 'value': 18.5}, actor=ACTOR)
    finally:
        log.close()

    rows = diary_rows()['rows']
    unresolved = [row for row in rows if row['kind'] == diary.FIXING and row['reason']]
    assert unresolved, 'the swap names InterestRate.ZAR and no policy orders it'
    assert all('InterestRate.ZAR' in row['reason'] for row in unresolved)
    assert all(row['source'] is None and row['state'] == diary.DUE for row in unresolved)

    verdict = CLIENT.get('/book/close/check', params={'date': '2030-01-01'}).json()
    assert verdict['legal'] is False
    assert {row['key'] for row in unresolved} <= {row['key'] for row in verdict['outstanding']}


def test_a_monitoring_date_is_never_a_payment_and_a_declared_one_always_is(unrecorded, tmp_path):
    """`get_settlement_currencies()` is the REVAL-DATE accumulator - a barrier registers its
    monitoring days in it - so it is not a payment ladder and the diary never reads it as one. A
    deal with no schedule announces a payment only from the `(date, amount)` its TYPE declares.

    Killing mutations: the reval-date set emitted as a payment ladder, which makes a barrier's
    monitoring day a payment that blocks every close; and the declared payment dropped, which makes
    a `FixedCashflowDeal` - a plain bullet - vanish from the diary and from the close check.
    """
    watched = BASE - pd.DateOffset(days=30)
    barrier = dict(BARRIER, Barrier_Dates=[[watched, 108.5]])
    cash = {'Object': 'FixedCashflowDeal', 'Reference': 'CF1', 'Currency': 'ZAR',
            'Discount_Rate': 'ZAR', 'Calendars': None, 'Amount': NOTIONAL,
            'Payment_Date': COUPONS[0]}
    serving(tmp_path, [node(barrier), node(cash)], factors=dict(FACTORS, **EQUITY))

    rows = diary_rows()['rows']
    monitoring = [row for row in rows if row['due_date'] == str(watched.date())]
    assert {row['kind'] for row in monitoring} == {diary.BARRIER, diary.FIXING}, monitoring

    declared = {row['due_date']: row for row in payments(rows)
                if row['leg'] == diary.SETTLEMENT}
    cash = declared[str(COUPONS[0].date())]
    assert cash['amount'] == NOTIONAL and cash['determined'] is True and cash['currency'] == 'ZAR'
    payoff = declared[str((BASE + pd.DateOffset(days=365)).date())]
    assert payoff['amount'] is None and payoff['determined'] is False, 'a payoff is not a field'
    assert set(declared) == {cash['due_date'], payoff['due_date']}
    assert not [row for row in payments(rows) if row['due_date'] == str(watched.date())]


def test_a_close_is_asked_for_a_day_and_nothing_else(recorded, tmp_path):
    """`?date=` is the only question this verb answers, so it is PARSED. A string compare answers
    `legal` for an empty string and for a garbage one, which is a wrong answer rather than a
    refusal; a date before the book's own is answered honestly.

    Killing mutation: the raw string compared against a row's `due_date`, under which `?date=`
    empty declares every close in the book legal.
    """
    serving(tmp_path, [node(fixed_leg())])
    for refused in ('', 'not-a-date', '2026', '2030-01-01T00:00:00', '2030/01/01'):
        answer = CLIENT.get('/book/close/check', params={'date': refused})
        assert answer.status_code == 422, (refused, answer.text)
        assert repr(refused) in answer.json()['detail'], refused

    early = CLIENT.get('/book/close/check', params={'date': '2020-01-01'}).json()
    assert early['legal'] is True and early['due'] == 0, 'nothing is due before the book begins'
    assert early['date'] == '2020-01-01'
    assert CLIENT.get('/book/close/check', params={'date': '2099-01-01'}).json()['due'] > 0


def test_an_expiry_that_vests_a_choice_blocks_a_close_until_somebody_elects(recorded, tmp_path):
    """An `expiry` row whose terms leave a choice in an actor's hands is outstanding until an
    `election` is filed against its instrument; one a fixing determines never blocks a close.

    Killing mutation: the close check ignoring `needs`, under which a physically settled option
    nobody exercised passes a close in silence.
    """
    serving(tmp_path, [netting_set(CLIENT_SET, 'CPTY_A')], factors=dict(FACTORS, **EQUITY))
    booked = CLIENT.post('/book/deals', content=dump(
        {'action': 'add', 'deal': OPTION, 'parent_reference': CLIENT_SET,
         'quantity': 1.0, 'execution_reference': 'EXEC-OPT'}), headers=JSON).json()
    assert booked['written'] is True, booked

    verdict = CLIENT.get('/book/close/check', params={'date': '2030-01-01'}).json()
    vests = [row for row in verdict['outstanding'] if row['needs'] == diary.ELECTION]
    assert len(vests) == 1 and vests[0]['kind'] == diary.EXPIRY
    assert verdict['legal'] is False

    derivus.Context().load_json((dump(job([])), 'elect')).apply_lifecycle(
        'election', {'instrument': vests[0]['instrument'], 'choice': 'exercise'})
    cleared = CLIENT.get('/book/close/check', params={'date': '2030-01-01'}).json()
    assert [row for row in cleared['outstanding'] if row['needs']] == []

    cash = diary_rows()['rows']
    settled = [row for row in cash if row['kind'] == diary.EXPIRY and row['needs']]
    assert settled == [], 'the election answered every expiry that vested one'


def test_a_cash_settled_expiry_never_vests_a_choice(unrecorded, tmp_path):
    """The other half of `needs`: a cash-settled option's payoff is determined by its fixing, so
    its expiry leaves nobody a choice and never blocks a close."""
    serving(tmp_path, [node(dict(OPTION, Reference='EQO_CASH', Settlement_Style='Cash'))],
            factors=dict(FACTORS, **EQUITY))
    rows = [row for row in diary_rows()['rows'] if row['kind'] == diary.EXPIRY]
    assert rows and all(row['needs'] is None for row in rows)


def test_a_read_never_refuses_over_the_books_own_undeclared_index(recorded, tmp_path):
    """THE COMPILE HALF OF A READ REFUSES NOTHING EITHER. A desk that files fixings before it
    declares the order - or declares one index and not another - has a book whose OWN index the
    policy does not order. The plan path refuses that by name; the two read verbs answer, fill what
    the declared orders can fill, and put the reason on the rows that named it.

    Killing mutation: `DiaryJob` compiling through the strict fold, which raises the record's own
    refusal out of a GET and 500s both verbs on a book nothing is wrong with.
    """
    serving(tmp_path, [node(BARRIER_WATCHED)], factors=dict(FACTORS, **EQUITY))
    log = SpineLog(recorded)
    try:
        policy.declare(log, ACTOR, policy.FIXINGS_POLICY, {'sources': {'FxRate.ZAR': ['ECB']}})
        log.append('fixing_observed', {'index': 'EquityPrice.EQ', 'date': str(WATCHED.date()),
                                       'source': 'EXCHANGE', 'value': 108.5}, actor=ACTOR)
    finally:
        log.close()

    answer = CLIENT.get('/book/diary')
    assert answer.status_code == 200, answer.text
    unresolved = [row for row in answer.json()['rows'] if row['reason']]
    assert unresolved and all('EquityPrice.EQ' in row['reason'] for row in unresolved)
    assert all(row['source'] is None for row in unresolved)

    verdict = CLIENT.get('/book/close/check', params={'date': '2099-01-01'})
    assert verdict.status_code == 200, verdict.text
    assert verdict.json()['legal'] is False
    assert {row['key'] for row in unresolved if row['kind'] == diary.FIXING} <= {
        row['key'] for row in verdict.json()['outstanding']}

    # and the PLAN still refuses: strictness is what separates a plan from a read
    document, _ = service.BOOK.read()
    with pytest.raises(spine.SpineRefused):
        spine.compiled_job({'Calc': document['Calc']})


def test_a_diary_that_refused_runs_again_when_the_record_is_fixed(recorded, tmp_path):
    """SUCCESSES ONLY IN THE CACHE. A compile that failed for a reason OUTSIDE the book - here a
    monitoring day whose close the RECORD holds and no policy yet orders - is answered once; the
    next ask re-runs it, so a desk that declares the order is answered rather than refused again.

    Killing mutations: the stored `error` left in the result store, which the executor answers
    without re-enqueueing - the `/book/risk` discipline inverted, and the desk told the same thing
    until the file moves or the service restarts; and `forget` dropping any stored result rather
    than only an error, which recompiles the whole book on every ask and no reading would say so.
    """
    serving(tmp_path, [node(dict(BARRIER, Barrier_Dates=[[WATCHED, '']]))],
            factors=dict(FACTORS, **EQUITY))
    log = SpineLog(recorded)
    try:
        policy.declare(log, ACTOR, policy.FIXINGS_POLICY, {'sources': {'FxRate.ZAR': ['ECB']}})
        log.append('fixing_observed', {'index': 'EquityPrice.EQ', 'date': str(WATCHED.date()),
                                       'source': 'EXCHANGE', 'value': 108.5}, actor=ACTOR)
    finally:
        log.close()

    before = service.BOOK.read()[1]
    refused = CLIENT.get('/book/diary')
    assert refused.status_code == 422, refused.text
    assert 'will not compile a diary' in refused.json()['detail']
    assert 'Observed close is blank' in refused.json()['detail']

    # the desk does exactly what the refusal instructs, in the RECORD - the file does not move
    log = SpineLog(recorded)
    try:
        policy.declare(log, ACTOR, policy.FIXINGS_POLICY,
                       {'sources': {'EquityPrice.EQ': ['EXCHANGE']}})
    finally:
        log.close()

    again = CLIENT.get('/book/diary')
    assert again.status_code == 200, again.text
    assert service.BOOK.read()[1] == before, 'the book file moved'
    assert [row for row in again.json()['rows'] if row['observed'] == 108.5],         'the record filled the close the refusal asked for'

    # SUCCESSES ONLY means only an error is forgotten: a done, a queued and a running result stand
    service.EXECUTOR.results.update({'DONE': {'status': 'done'}, 'QUEUED': {'status': 'queued'},
                                     'RUNNING': {'status': 'running'}, 'BAD': {'status': 'error'}})
    for name in ('DONE', 'QUEUED', 'RUNNING', 'BAD', 'NEVER-FILED'):
        service.EXECUTOR.forget(name)
    kept = {name: service.EXECUTOR.result(name) for name in
            ('DONE', 'QUEUED', 'RUNNING', 'BAD', 'NEVER-FILED')}
    assert [name for name, stored in kept.items() if stored is None] == ['BAD', 'NEVER-FILED']
    for name in ('DONE', 'QUEUED', 'RUNNING'):
        del service.EXECUTOR.results[name]


def test_an_option_waits_for_its_fixing_and_its_settlement(recorded, tmp_path):
    """A CLOSE DOES NOT PASS OVER AN UNSETTLED PAYOFF. A cash-settled option alone in a book is due
    two things after its expiry - the underlying's print on the expiry day, and the settlement of
    the payoff that print decides - and the close check names both until each has its fact.

    Killing mutations: the option announcing its expiry alone, under which the catch-up rule passes
    over a payoff nobody paid and nobody observed; and the settlement day read off the expiry
    before the type's own field, under which an FX option's `Delivery_Date` is ignored entirely.
    """
    serving(tmp_path, [netting_set(CLIENT_SET, 'CPTY_A')], factors=dict(FACTORS, **EQUITY))
    expiry = str((BASE + pd.DateOffset(days=60)).date())
    booked = CLIENT.post('/book/deals', content=dump(
        {'action': 'add', 'deal': dict(OPTION, Settlement_Style='Cash'),
         'parent_reference': CLIENT_SET, 'quantity': 1.0,
         'execution_reference': 'EXEC-CASH'}), headers=JSON).json()
    assert booked['written'] is True, booked

    rows = {(row['kind'], row['due_date']): row for row in diary_rows()['rows']}
    assert (diary.FIXING, expiry) in rows and (diary.PAYMENT, expiry) in rows
    assert rows[(diary.FIXING, expiry)]['index'] == 'EquityPrice.EQ'
    assert rows[(diary.PAYMENT, expiry)]['amount'] is None
    assert rows[(diary.EXPIRY, expiry)]['needs'] is None, 'cash settlement vests no choice'

    # the settlement day is the TYPE'S OWN field where it declares one, not the expiry
    delivery = str((BASE + pd.DateOffset(days=62)).date())
    serving(tmp_path, [node(dict(FX_OPTION, Delivery_Date=BASE + pd.DateOffset(days=62)))],
            factors=dict(FACTORS, **EQUITY))
    paid = [row for row in payments(diary_rows()['rows'])]
    assert [row['due_date'] for row in paid] == [delivery], 'the payment stood on the expiry'
    assert [row['due_date'] for row in diary_rows()['rows']
            if row['kind'] == diary.FIXING] == [expiry], 'the fixing is the expiry day'
    serving(tmp_path, [netting_set(CLIENT_SET, 'CPTY_A', [dict(OPTION,
                                                               Settlement_Style='Cash')])],
            factors=dict(FACTORS, **EQUITY))

    after = str((BASE + pd.DateOffset(days=61)).date())
    verdict = CLIENT.get('/book/close/check', params={'date': after}).json()
    assert verdict['legal'] is False
    assert {row['kind'] for row in verdict['outstanding']} == {diary.FIXING, diary.PAYMENT}

    log = SpineLog(recorded)
    try:
        policy.declare(log, ACTOR, policy.FIXINGS_POLICY,
                       {'sources': {'EquityPrice.EQ': ['EXCHANGE']}})
        log.append('fixing_observed', {'index': 'EquityPrice.EQ', 'date': expiry,
                                       'source': 'EXCHANGE', 'value': 133.0}, actor=ACTOR)
        log.append('status_transition',
                   {'subject': rows[(diary.PAYMENT, expiry)]['key'], 'status': diary.SETTLED},
                   actor=ACTOR, book='diary-desk')
    finally:
        log.close()

    cleared = CLIENT.get('/book/close/check', params={'date': after}).json()
    assert cleared['legal'] is True, cleared['outstanding']
    assert cleared['due'] == verdict['due'], 'the rows are answered, not dropped'


@pytest.mark.parametrize('name,deal', [
    ('an FX option settled in cash', 'FX_CASH'),
    ('an FX option settled physically', 'FX_PHYSICAL'),
    ('an equity option settled in cash', 'EQUITY_CASH')])
def test_an_option_that_expired_yesterday_still_blocks_the_close(name, deal, recorded, tmp_path):
    """NOTHING FALLS OFF THE DIARY BY EXPIRING. A deal the compile drops because its expiry is
    behind the base date is exactly the one a catch-up rule exists for - it fell due and nobody
    cleared it - so the dropped branch announces what the live one announces: the fixing its
    underlying owes and the settlement its type declares, both pure field reads.

    Killing mutation: the dropped branch emitting the expiry row alone, under which a cash-settled
    option that expired yesterday lets a close through with nothing outstanding at all.
    """
    yesterday = BASE - pd.DateOffset(days=1)
    expired = {'FX_CASH': dict(FX_OPTION, Expiry_Date=yesterday, Delivery_Date=yesterday,
                               Settlement_Style='Cash'),
               'FX_PHYSICAL': dict(FX_OPTION, Expiry_Date=yesterday, Delivery_Date=yesterday),
               'EQUITY_CASH': dict(OPTION, Expiry_Date=yesterday,
                                   Settlement_Style='Cash')}[deal]
    serving(tmp_path, [netting_set(CLIENT_SET, 'CPTY_A')], factors=dict(FACTORS, **EQUITY))
    booked = CLIENT.post('/book/deals', content=dump(
        {'action': 'add', 'deal': expired, 'parent_reference': CLIENT_SET, 'quantity': 1.0,
         'execution_reference': 'EXEC-' + deal}), headers=JSON).json()
    assert booked['written'] is True, booked

    day = str(yesterday.date())
    rows = {(row['kind'], row['due_date']): row for row in diary_rows()['rows']}
    assert rows[(diary.EXPIRY, day)]['state'] == diary.EXPIRED
    assert (diary.FIXING, day) in rows and (diary.PAYMENT, day) in rows, sorted(rows)

    verdict = CLIENT.get('/book/close/check', params={'date': '2099-01-01'}).json()
    assert verdict['legal'] is False, name
    assert {row['kind'] for row in verdict['outstanding']} >= {diary.FIXING, diary.PAYMENT}

    log = SpineLog(recorded)
    try:
        policy.declare(log, ACTOR, policy.FIXINGS_POLICY,
                       {'sources': {rows[(diary.FIXING, day)]['index']: ['EXCHANGE']}})
        log.append('fixing_observed', {'index': rows[(diary.FIXING, day)]['index'], 'date': day,
                                       'source': 'EXCHANGE', 'value': 18.0}, actor=ACTOR)
        log.append('status_transition',
                   {'subject': rows[(diary.PAYMENT, day)]['key'], 'status': diary.SETTLED},
                   actor=ACTOR, book='diary-desk')
        if rows[(diary.EXPIRY, day)]['needs']:
            log.append('election', {'instrument': rows[(diary.EXPIRY, day)]['instrument'],
                                    'choice': 'exercise'}, actor=ACTOR, book='diary-desk')
    finally:
        log.close()

    cleared = CLIENT.get('/book/close/check', params={'date': '2099-01-01'}).json()
    assert cleared['legal'] is True, (name, cleared['outstanding'])


def test_a_deal_the_compile_could_not_read_is_named_and_blocks_the_close(recorded, tmp_path):
    """A CLEAN BILL NOBODY EARNED IS WORSE THAN A REFUSAL. A deal skipped for a price factor the
    market data has no block for leaves no schedule, no expiry and no trace, so a close check that
    answered `legal` would be a verdict on a book the engine could not read.

    One `unreadable` row per such deal, carrying where it sits and the engine's own sentence, and
    outstanding on EVERY day - it has no date of its own.

    Killing mutation: the unreadable rows left out of the outstanding count, which is the clean
    bill again with the row rendered beside it.
    """
    serving(tmp_path, [node(dict(FX_OPTION, Reference='NOVOL'))], factors=FACTORS)
    rows = [row for row in diary_rows()['rows'] if row['kind'] == diary.UNREADABLE]
    assert len(rows) == 1, diary_rows()['rows']
    assert rows[0]['leg'] == '0' and rows[0]['state'] == diary.UNREADABLE
    assert rows[0]['due_date'] is None and rows[0]['amount'] is None and rows[0]['key'] is None
    assert 'NOVOL' in rows[0]['reason'] and 'FXOptionDeal' in rows[0]['reason']
    assert 'FXVol' in rows[0]['reason'], rows[0]['reason']

    verdict = CLIENT.get('/book/close/check', params={'date': '2020-01-01'}).json()
    assert verdict['legal'] is False, 'a day with nothing due was still not legal'
    assert [row['kind'] for row in verdict['outstanding']] == [diary.UNREADABLE]


def test_the_diary_never_runs_on_the_poll_path(unrecorded, tmp_path):
    """GATE 14. Two asks over an unmoved book are ONE compile: the result id is the etag's own, so
    the second submission would coalesce onto the first and the cache never reaches it. A booking
    between them moves the etag and queues a second.

    Killing mutations: the cache keyed on anything but what a COMPILE reads - a clock, which puts a
    deal setup over the whole market on every poll tick; and the market folded into the key, which
    throws the compile away on every beat of the cadence for rows that do not move.
    """
    serving(tmp_path, [netting_set(CLIENT_SET, 'CPTY_A', [fixed_leg()])])
    first = diary_rows()
    second = diary_rows()
    assert first['result_id'] == second['result_id'] and first['etag'] == second['etag']
    assert first['as_of'] == second['as_of'], 'the second ask re-ran the compile'

    for spot in (18.75, 19.25):
        ticked = CLIENT.post('/book/market', content=dump(
            {'patch': {'FxRate.ZAR': {'Spot': spot}}}), headers=JSON)
        assert ticked.json()['written'] is True, ticked.text
        after = diary_rows()
        assert after['result_id'] == first['result_id'], 'a market tick threw the compile away'
        assert after['rows'] == first['rows'], 'a market tick moved a row'

    booked = CLIENT.post('/book/deals', content=dump(
        {'action': 'add', 'deal': fixed_leg('SECOND'), 'parent_reference': CLIENT_SET}),
        headers=JSON).json()
    assert booked['written'] is True, booked
    third = diary_rows()
    assert third['result_id'] != first['result_id'] and third['etag'] != first['etag']
    assert len(payments(third['rows'])) == 2 * len(payments(first['rows']))


# --------------------------------------------------------------------------------------------
# The close, and the settlement file.

def test_a_close_is_declared_only_on_a_day_the_check_calls_legal(recorded, tmp_path):
    """THE CLOSE RUNS BEHIND THE CHECK. `POST /book/close` declares what `GET /book/close/check`
    answers, so a day the check calls illegal refuses HERE naming what it waits on and the head does
    not move - a close over a payoff nobody observed is a clean bill nobody earned. A legal day
    files the close over the book's own values vector, `market` defaulting to `official` and `date`
    to the book's own base date, and a SECOND close on one market supersedes the first rather than
    correcting it, the answer naming the position it stands over.

    THE VERDICT IS TAKEN OVER THE DOCUMENT THE CLOSE IS STRUCK ON - one read of the book, since a
    booking landing between two would make the verdict a statement about a book the close was never
    declared over - and its compile is queued under the CALLER'S seat, like the export's, so a close
    a seat is not scoped for costs the box nothing before it is refused.

    Killing mutations: the check called and its verdict not read, which declares a close over an
    unobserved expiry; the date compared as a string rather than parsed, which calls an empty one
    legal and closes the book on nothing; the supersession read off the close just filed rather
    than off the fold, which reports every close as standing over itself; and the verdict's compile
    queued under the deployment's seat, which charges the box for a close it then refuses.
    """
    serving(tmp_path, [netting_set(CLIENT_SET, 'CPTY_A', [dict(OPTION, Settlement_Style='Cash')])],
            factors=dict(FACTORS, **EQUITY))
    head = at(recorded)

    after = str((BASE + pd.DateOffset(days=61)).date())
    refused = CLIENT.post('/book/close', json={'date': after})
    assert refused.status_code == 422 and diary.FIXING in refused.json()['detail'], refused.text
    for garbage in ('', '2024', 'not-a-day', '2024-06-28T16:30'):
        assert CLIENT.post('/book/close', json={'date': garbage}).status_code == 422, garbage
    assert at(recorded) == head, 'a refused close moved the record'

    standing = service.load(service.BOOK.read()[0])
    declared = CLIENT.post('/book/close', json={}).json()
    assert declared['date'] == str(BASE.date()) and declared['market'] == 'official'
    assert declared['values_hash'] == standing.values_hash()
    assert declared['supersedes_lsn'] is None and at(recorded) == head + 1, declared

    ticked = CLIENT.post('/book/market', content=dump({'patch': {'FxRate.ZAR': {'Spot': 19.5}}}),
                         headers=JSON)
    assert ticked.json()['written'] is True, ticked.text
    restated = CLIENT.post('/book/close', json={'date': str(BASE.date())}).json()
    assert restated['values_hash'] != declared['values_hash']
    assert restated['supersedes_lsn'] == declared['recorded']['lsn']
    assert at(recorded, 'official_close_declared') == [declared['recorded']['lsn'],
                                                       restated['recorded']['lsn']]

    # the compile the verdict is read off is the CALLER's job: a seat the document scopes for
    # nothing is refused at the queue, before the close is even a question
    log = SpineLog(recorded)
    try:
        blob = log.store.put(capability.canonical_document(
            {'grants': [{'subject': ACTOR, 'verb': v, 'book': '*'}
                        for v in ('admin', 'validate', 'mark')], 'read': []}))
        log.append('policy_declared', {'policy': capability.CAPABILITIES_POLICY, 'blob': blob},
                   actor=ACTOR, blob_refs=(blob,))
    finally:
        log.close()
    service.BOOK_DIARY_CACHE.clear()
    stranger = CLIENT.post('/book/close', json={'actor': 'subject-nobody-at-all'})
    assert stranger.status_code == 422, stranger.text
    said = stranger.json()['detail']
    assert 'subject-nobody-at-all' in said and 'validate' in said, said
    assert not service.BOOK_DIARY_CACHE, 'the box compiled a diary for a close it then refused'


def test_the_settlement_file_is_struck_on_the_market_the_record_designates(recorded, tmp_path):
    """THE EXPORT NAMES NO MARKET AND CANNOT. Which board a settlement file is struck on is the one
    the `tiers` policy DESIGNATES for the export, read by name off the record, so a caller pointing
    the export at another market is unrepresentable rather than merely refused - the extra key is
    nothing to the verb. A home designating nothing, and a designation nothing stands under, both
    refuse at SUBMISSION, before a row is compiled, with the declaration that fixes it.

    What travels is the DIARY's own rows, exported by `export_settlements` off the resolved hash, so
    the verb composes and spells nothing: the answer is that function's, byte for byte, plus the
    market block and the count. An undetermined row refuses by name rather than instructing a
    payment of zero.

    NOTHING IS COMPILED BEHIND A REFUSAL. The designation is resolved before the diary is asked
    for, so an undesignated home is told what to declare without paying for a whole compile first -
    asserted on the executor's store rather than on the wording.

    THE COMPILE IS ADMITTED UNDER THE REQUEST'S SEAT, and BEFORE the cache is consulted: a warm
    cache reaches no queue, so a settlement file a desk instructs payments from would otherwise be
    a function of who asked first.

    Killing mutations: a `market` taken off the request, which lets a settlement file be struck on
    any board a caller can name; the designation read without resolving it, which strikes the file
    on a name nothing stands under; the designation resolved AFTER the compile, which makes an
    undesignated home pay for the whole diary; the compile queued under the deployment's seat; and
    admission asked only on a cache miss.
    """
    serving(tmp_path, [netting_set(CLIENT_SET, 'CPTY_A', [fixed_leg()])])
    ever = '2099-01-01'
    stored = dict(service.EXECUTOR.results)

    undesignated = CLIENT.post('/book/settlements', json={'due_before': ever})
    assert undesignated.status_code == 422, undesignated.text
    for said in ('settlement_export', policy.DESIGNATIONS_SECTION, policy.TIERS_POLICY):
        assert said in undesignated.json()['detail'], said
    assert service.EXECUTOR.results == stored and not service.BOOK_DIARY_CACHE, \
        'an undesignated home paid for the compile before it was told what to declare'

    log = SpineLog(recorded)
    try:
        policy.declare(log, ACTOR, policy.TIERS_POLICY,
                       {'tiers': [{'name': 'desk', 'four_eyes': True}],
                        'designations': {'settlement_export': 'official'}})
    finally:
        log.close()
    unmarked = CLIENT.post('/book/settlements', json={'due_before': ever})
    assert unmarked.status_code == 422 and 'official' in unmarked.json()['detail'], unmarked.text
    assert CLIENT.post('/book/settlements', json={}).status_code == 422, 'due_before had a default'
    assert not service.BOOK_DIARY_CACHE, 'a refused export compiled a diary'

    marked = CLIENT.post('/book/markets', json={'name': 'official'}).json()
    exported = CLIENT.post('/book/settlements',
                           json={'due_before': ever, 'market': 'dealer'}).json()
    assert exported['market'] == {'name': 'official', 'values_hash': marked['values_hash'],
                                  'lsn': marked['recorded']['lsn']}, 'a named market was resolved'

    direct = diary.export_settlements(diary_rows()['rows'], marked['values_hash'], ever)
    # `values_hash` is the exporter's own statement of the board; `market` is the record's, the
    # name it resolved under and the position that name stands at
    assert (exported['rows'], exported['totals'], exported['due_before'],
            exported['values_hash']) == (direct['rows'], direct['totals'], ever,
                                         direct['values_hash'])
    assert exported['count'] == len(exported['rows']) == len(payments(diary_rows()['rows']))

    serving(tmp_path, [node(swap())])
    undetermined = CLIENT.post('/book/settlements', json={'due_before': ever})
    assert undetermined.status_code == 422
    assert 'not determined' in undetermined.json()['detail'], undetermined.text

    # the export names the seat its COMPILE is queued under, which is the REQUEST's and not the
    # deployment's, and the question is asked whether or not the cache can already answer it
    other = 'subject-desk-two'
    serving(tmp_path, [netting_set(CLIENT_SET, 'CPTY_A', [fixed_leg()])])
    log = SpineLog(recorded)
    try:
        blob = log.store.put(capability.canonical_document(
            {'grants': [{'subject': other, 'verb': 'validate', 'book': 'diary-desk'}], 'read': []}))
        log.append('policy_declared', {'policy': capability.CAPABILITIES_POLICY, 'blob': blob},
                   actor=ACTOR, blob_refs=(blob,))
    finally:
        log.close()
    assert CLIENT.post('/book/settlements',
                       json={'due_before': ever, 'actor': other}).status_code == 200
    assert service.BOOK_DIARY_CACHE, 'nothing was cached, so the next ask proves nothing'
    unnamed = CLIENT.post('/book/settlements', json={'due_before': ever})
    assert unnamed.status_code == 422 and ACTOR in unnamed.json()['detail'], unnamed.text
