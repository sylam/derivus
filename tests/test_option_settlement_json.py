"""An option paid after it expires, through the JSON contract and nothing else.

`Settlement_Date` is the day the payoff is paid, the expiry where blank. Between the two the
underlying is what the expiry fixed: on a simulated grid the path's own level on the expiry row,
and on a base date past the expiry the print `Price_Fixing` states - which an expired deal must
state, since the spot of today is not the level it settled on. A swaption settled in cash is
paid the same way; settled physically it becomes the swap its legs make and books its cash.

THE ORACLES: under a credit Monte Carlo every row from the expiry on is the settlement row's payoff
discounted to it, scenario by scenario, and the ledger books that payoff on the settlement row and
nothing on the expiry's; under a base valuation past the expiry the mark is the print's intrinsic
discounted to the settlement, exact.
"""
import functools
import json
import math
import os
import sys

# reference-derivus shadow-import guard (MEMORY): pin the package under test to THIS repo.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
import torch

import derivus
import test_declared_defaults as book
import trial_rates
from derivus import utils
from derivus.config import CustomJsonEncoder

BASE = pd.Timestamp('2024-06-28')
EXPIRY = pd.Timestamp('2025-06-27')                  # a Friday
SETTLE = EXPIRY + pd.offsets.BDay(2)                 # the Tuesday after
SPOT, STRIKE, UNITS, CASH, PRINT = 100.0, 100.0, 10.0, 1000.0, 112.0
R_USD, Q_EQ, SIGMA = 0.04, 0.01, 0.25
FX_SPOT, R_EUR, FX_SIGMA = 1.25, 0.02, 0.15

FACTORS = {
    'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Spot': 1.0},
    'FxRate.EUR': {'Domestic_Currency': None, 'Interest_Rate': 'EUR', 'Spot': FX_SPOT},
    'InterestRate.USD': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                         'Curve': utils.Curve([], [[0.0, R_USD], [5.0, R_USD]])},
    'InterestRate.EUR': {'Currency': 'EUR', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                         'Curve': utils.Curve([], [[0.0, R_EUR], [5.0, R_EUR]])},
    'FXVol.EUR.USD': {'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness',
                      'Surface': utils.Curve([], [[m, t, FX_SIGMA] for m in (0.6, 1.0, 1.4)
                                                  for t in (0.02, 2.0)])},
    'EquityPrice.EQ': {'Spot': SPOT, 'Currency': 'USD', 'Interest_Rate': 'USD', 'Issuer': '',
                       'Respect_Default': 'No', 'Jump_Level': 0.0},
    'DividendRate.EQ': {'Currency': 'USD', 'Curve': utils.Curve([], [[0.0, Q_EQ], [5.0, Q_EQ]])},
    'EquityPriceVol.EQ': {'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness',
                          'Surface': utils.Curve([], [[m, t, SIGMA] for m in (0.6, 1.0, 1.4)
                                                      for t in (0.02, 2.0)])}}

VANILLA = {'Object': 'EquityOptionDeal', 'Reference': 'EQO', 'Currency': 'USD',
           'Payoff_Currency': 'USD', 'Equity': 'EQ', 'Dividends': 'EQ', 'Discount_Rate': 'USD',
           'Equity_Volatility': 'EQ', 'Buy_Sell': 'Buy', 'Option_Type': 'Call',
           'Strike_Price': STRIKE, 'Units': UNITS, 'Expiry_Date': EXPIRY, 'Settlement_Date': SETTLE}
BINARY = dict({k: v for k, v in VANILLA.items() if k != 'Units'},
              Object='EquityBinaryOption', Reference='EQB', Payoff=CASH)
FX = {'Object': 'FXOptionDeal', 'Reference': 'FXO', 'Currency': 'USD', 'Underlying_Currency': 'EUR',
      'Discount_Rate': 'USD', 'FX_Volatility': 'EUR.USD', 'Buy_Sell': 'Buy', 'Option_Type': 'Call',
      'Strike_Price': FX_SPOT, 'Underlying_Amount': UNITS, 'Expiry_Date': EXPIRY}
CMC = {'Object': 'CreditMonteCarlo', 'Time_grid': '0d 12m(1m)', 'Batch_Size': 256, 'Random_Seed': 1,
       'Simulation_Batches': 1, 'Deflation_Interest_Rate': 'USD', 'Generate_Cashflows': 'Yes'}


def node(deal):
    """A deal-tree node, a netting set's `Children` beneath it."""
    out = {'Instrument': {'.Deal': {k: v for k, v in deal.items() if k != 'Children'}}}
    return dict(out, Children=[node(child) for child in deal['Children']]) if 'Children' in deal else out


def run(deals, base=BASE, calc=None):
    """The results of one job over `deals`: a base valuation at `base`, or `calc` on a GBM spot."""
    market = {'System Parameters': {'Base_Currency': 'USD', 'Base_Date': base},
              'Valuation Configuration': {}, 'Price Factors': FACTORS}
    if calc:
        market['Price Models'] = {'GBMAssetPriceModel.EQ': {'Vol': SIGMA, 'Drift': 0.0}}
        market['Model Configuration'] = {'.ModelParams': {
            'modeldefaults': {'EquityPrice': 'GBMAssetPriceModel'}, 'modelfilters': {}}}
    job = {'Calc': {
        'Calculation': dict({'Object': 'BaseValuation', 'Base_Date': base, 'Currency': 'USD',
                             'MCMC_Simulations': 1}, **(calc or {})),
        'Deals': {'Reference': 'settle', 'Deals': {'Children': [node(d) for d in deals]}},
        'MergeMarketData': {'MarketDataFile': '', 'ExplicitMarketData': market}}}
    cx = derivus.Context()
    cx.load_json((json.dumps(job, cls=CustomJsonEncoder), 'settle'))
    return cx.run_job()[1]['Results']


def marks(deals, base=BASE):
    frame = run(deals, base)['mtm']
    return dict(zip(frame['Reference'], frame['Value'].astype(float)))


def df(days):
    return math.exp(-R_USD * days / 365.0)


def ndtr(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def test_past_the_expiry_every_row_is_the_settlement_payoff_discounted_and_the_ledger_books_it_there():
    """Under a credit Monte Carlo on a simulated spot the grid runs to the settlement, with the
    monthly date between it and the expiry. On every row from the expiry on the mark is the
    settlement row's payoff discounted to it, scenario by scenario - the level is the expiry row's
    own and not that row's spot - and the ledger books that payoff on the settlement row alone.
    The run is float32, so the readings are held to 1e-5.

    Killing mutations: the level not held past the expiry (`held_past_expiry` handing back its
    input), every later row marking its own spot's intrinsic; the settlement not a reval date,
    which leaves no row to book on.
    """
    results = run([VANILLA], calc=CMC)
    mtm, ledger = results['mtm'], results['cashflows']['USD']
    assert np.isfinite(np.asarray(mtm.values, dtype=float)).all()
    assert list(ledger.index) == [SETTLE], ledger.index
    paid, settled = (np.asarray(frame.loc[SETTLE].values, dtype=float) for frame in (ledger, mtm))
    assert np.allclose(paid, settled, rtol=1e-5, atol=1e-6)
    assert settled.mean() > 0.0 and (settled == 0.0).any(), 'in and out of the money across the paths'
    later = [day for day in mtm.index if day >= EXPIRY]
    assert len(later) == 3, later                    # the expiry, the monthly date after it, the settlement
    for day in later:
        marked = np.asarray(mtm.loc[day].values, dtype=float)
        assert np.allclose(marked, settled * df((SETTLE - day).days), rtol=1e-5, atol=1e-6), day


def test_expired_before_the_base_date_the_payoff_is_on_the_print_and_a_missing_one_refuses():
    """On a base date between the expiry and the settlement the vanilla marks its print's intrinsic
    discounted to the settlement, exact; the cash digital its payoff discounted, the asset digital
    its payoff times the print; and a document stating no print - the table absent, or its row
    blank as the record leaves one it has no close for - refuses by name, naming the deal, the
    table and the day.

    Killing mutations: the print unread, today's spot pricing the call at nothing; a blank row read
    as a print.
    """
    base = EXPIRY + pd.offsets.BDay(1)
    fixed = [[EXPIRY, PRINT]]
    marked = marks([dict(VANILLA, Price_Fixing=fixed),
                    dict(VANILLA, Reference='PUT', Option_Type='Put', Price_Fixing=fixed),
                    dict(BINARY, Price_Fixing=fixed),
                    dict(BINARY, Reference='ASSET', Payoff_Style='Asset', Price_Fixing=fixed)], base)
    d = df((SETTLE - base).days)
    assert marked['EQO'] == pytest.approx(UNITS * (PRINT - STRIKE) * d, rel=1e-9)
    assert marked['PUT'] == 0.0
    assert marked['EQB'] == pytest.approx(CASH * d, rel=1e-9)
    assert marked['ASSET'] == pytest.approx(CASH * PRINT * d, rel=1e-9)
    for unstated in (VANILLA, dict(VANILLA, Price_Fixing=[[EXPIRY, '']])):
        with pytest.raises(Exception) as refusal:
            marks([unstated], base)
        message = str(refusal.value)
        assert all(word in message for word in ('EQO', 'Price_Fixing', '2025-06-27')), message


def test_an_fx_option_expired_before_the_base_date_marks_on_its_print_and_a_missing_one_refuses():
    """The FX option and binary declare the `Price_Fixing` table the equity ones do, filled by the
    record's prints of the pair: on a base date between the expiry and the delivery the vanilla
    marks its print's intrinsic discounted to the delivery, exact, the cash binary its payoff
    discounted, and one stating no print refuses by name, naming the deal, the table and the day.

    Killing mutations: the table undeclared, under which every expired FX option refuses; the
    print unread, today's spot pricing the call at nothing.
    """
    base, print_ = EXPIRY + pd.offsets.BDay(1), 1.40
    fixed = [[EXPIRY, print_]]
    binary = {'Object': 'FXBinaryOption', 'Reference': 'FXB', 'Currency': 'USD',
              'Underlying_Currency': 'EUR', 'Discount_Rate': 'USD', 'FX_Volatility': 'EUR.USD',
              'Buy_Sell': 'Buy', 'Option_Type': 'Call', 'Strike_Price': FX_SPOT, 'Payoff': CASH,
              'Expiry_Date': EXPIRY, 'Delivery_Date': SETTLE}
    marked = marks([dict(FX, Delivery_Date=SETTLE, Price_Fixing=fixed),
                    dict(FX, Reference='PUT', Option_Type='Put', Delivery_Date=SETTLE,
                         Price_Fixing=fixed),
                    dict(binary, Price_Fixing=fixed)], base)
    d = df((SETTLE - base).days)
    assert marked['FXO'] == pytest.approx(UNITS * (print_ - FX_SPOT) * d, rel=1e-9)
    assert marked['PUT'] == 0.0
    assert marked['FXB'] == pytest.approx(CASH * d, rel=1e-9)
    for unstated in (dict(FX, Delivery_Date=SETTLE),
                     dict(FX, Delivery_Date=SETTLE, Price_Fixing=[[EXPIRY, '']])):
        with pytest.raises(Exception) as refusal:
            marks([unstated], base)
        message = str(refusal.value)
        assert all(word in message for word in ('FXO', 'Price_Fixing', '2025-06-27')), message


def test_on_the_expiry_the_engine_fixes_the_level_and_before_it_the_lag_only_moves_the_discount():
    """A base date ON the expiry prices off the engine's own spot - the forward to the expiry's
    settlement at zero tenor, intrinsic - and reads no print, as a fixing dated today follows the
    factor; before the expiry the lagged option is Black discounted to the settlement with the vol
    tenor at the expiry, exact, which the lag leaves exactly where it was.
    """
    carry = (SETTLE - EXPIRY).days
    marked = marks([dict(VANILLA, Strike_Price=90.0),
                    dict(VANILLA, Reference='PRINTED', Strike_Price=90.0, Price_Fixing=[[EXPIRY, PRINT]])],
                   EXPIRY)
    intrinsic = UNITS * (SPOT * math.exp((R_USD - Q_EQ) * carry / 365.0) - 90.0) * df(carry)
    assert marked['EQO'] == pytest.approx(intrinsic, rel=1e-9)
    assert marked['PRINTED'] == marked['EQO']
    t, t_settle = (EXPIRY - BASE).days / 365.0, (SETTLE - BASE).days / 365.0
    forward, sd = SPOT * math.exp((R_USD - Q_EQ) * t_settle), SIGMA * math.sqrt(t)
    d1 = math.log(forward / STRIKE) / sd + 0.5 * sd
    black = forward * ndtr(d1) - STRIKE * ndtr(d1 - sd)
    assert marks([VANILLA])['EQO'] == pytest.approx(UNITS * black * math.exp(-R_USD * t_settle), rel=1e-9)


def test_an_fx_option_pays_on_the_delivery_date_its_family_declares():
    """The FX option settles on `Delivery_Date`, the field its family declares it settles on, the
    expiry where blank: Black on the forward to the expiry's own settlement, discounted to the
    delivery, exact. A `Settlement_Date` written on it declares nothing and is not read.

    Killing mutation: the settlement read off a `Settlement_Date` the type does not declare.
    """
    delivery = EXPIRY + pd.DateOffset(days=10)
    marked = marks([FX, dict(FX, Reference='UNDECLARED', Settlement_Date=delivery),
                    dict(FX, Reference='DELIVERED', Delivery_Date=delivery)])
    t, t_fwd = (EXPIRY - BASE).days / 365.0, (SETTLE - BASE).days / 365.0
    forward, sd = FX_SPOT * math.exp((R_USD - R_EUR) * t_fwd), FX_SIGMA * math.sqrt(t)
    d1 = math.log(forward / FX_SPOT) / sd + 0.5 * sd
    black = UNITS * (forward * ndtr(d1) - FX_SPOT * ndtr(d1 - sd))
    assert marked['FXO'] == pytest.approx(black * df((EXPIRY - BASE).days), rel=1e-9)
    assert marked['UNDECLARED'] == marked['FXO']
    assert marked['DELIVERED'] == pytest.approx(black * df((delivery - BASE).days), rel=1e-9)


#: The declared-defaults swaption, settled physically on its legs, and its cash twin paid 40 days
#: after the expiry
SWPT = next(deal for deal in book.BOOK if deal['Reference'] == 'SWPT')
SWPT_PAID = book.WORLD_EXPIRY + pd.DateOffset(days=40)
CASH_SWPT = dict(SWPT, Reference='CSWPT', Settlement_Style='Cash', Settlement_Date=SWPT_PAID,
                 Children=[dict(leg, Reference='C' + leg['Reference']) for leg in SWPT['Children']])
#: The rates trial's amortising swaption settled in cash on the par yield annuity, paid with the twin
AMORT = next(deal for deal in trial_rates.DEALS if deal['Reference'] == 'SWPT_AMORT')
YIELD_SWPT = dict(AMORT, Reference='YSWPT', Settlement_Style='Cash', Settlement_Date=SWPT_PAID,
                  Cash_Settlement_Method='Par_Yield_Curve',
                  Children=[dict(leg, Reference='Y' + leg['Reference']) for leg in AMORT['Children']])


def cash_annuity(rate, items):
    """A fixed leg's coupons discounted at `rate`, compounded period by period, by hand."""
    tau = np.array([row['Accrual_Year_Fraction'] for row in items])
    coupons = np.array([row['Notional'] for row in items]) * tau
    rate = np.asarray(rate, dtype=float)[..., None]
    return (coupons * np.exp(-np.cumsum(np.log1p(tau * rate), axis=-1))).sum(axis=-1)


@functools.lru_cache(maxsize=None)
def swaptions():
    """One float64 credit Monte Carlo of the declared-defaults world, a Hull-White on both curves:
    the swaption, its cash twin, the swap it delivers beside a unit paid on the twin's day, and the
    amortising par yield swaption beside the swap it becomes, each in a netting set of its own -
    `(dates, {set: ledger}, {reference: profile})`."""
    def held(reference, deals):
        return {'Object': 'NettingCollateralSet', 'Reference': reference, 'Netted': 'True',
                'Collateralized': 'False', 'Children': deals}

    swap = [dict(leg, Reference='SWAP' + leg['Reference'][4:]) for leg in SWPT['Children']]
    amortised = [dict(leg, Reference='AMORTISED' + leg['Reference'][10:]) for leg in AMORT['Children']]
    unit = {'Object': 'FixedCashflowDeal', 'Reference': 'UNIT', 'Currency': 'USD',
            'Discount_Rate': 'USD', 'Amount': 1.0, 'Payment_Date': SWPT_PAID}
    calc, out = book.simulated(
        [held('PHYSICAL', [SWPT]), held('CASH', [CASH_SWPT]), held('SWAP', swap + [unit]),
         held('YIELD', [YIELD_SWPT]), held('AMORTISED', amortised)],
        ('USD', 'USD-PROJ'), prec=torch.float64, Generate_Cashflows='Yes'
        )
    sets = calc.netting_sets.sub_structures[0].sub_structures
    priced = [x.obj for s in sets for x in s.sub_structures] + [d for s in sets for d in s.dependencies]
    ledgers = {s.obj.Instrument.field['Reference']: pd.concat(s.obj.Calc_res['cashflows']['USD'], axis=1)
               for s in sets}
    profiles = {x.Instrument.field['Reference']: np.concatenate(x.Calc_res['Value'], axis=-1)
                for x in priced if (x.Calc_res or {}).get('Value')}
    return list(out['Results']['mtm'].index), ledgers, profiles


def test_a_physical_swaption_settles_the_swap_it_becomes_and_nothing_where_it_lapses():
    """Exercised, the swaption IS the swap its legs make: on the paths in the money at the expiry -
    the delivered swap's own mark positive there - its ledger is that swap's ledger, the same two
    legs held alone beside it in the run, on every payment date the swap has, to the bit, and on
    the others it books nothing. Nothing settles on the base date or the expiry: the ledger's rows
    are the swap's payment dates and the cash twin's.

    Killing mutations: the legs' cash unbooked, an empty ledger where the swap's first coupon pays
    11,166.27 on the first path; booked on every path, the lapsed paying the swap's coupons; the
    base date or the expiry registered as a day it settles, a 2026-08-03 or 2027-08-03 row.
    """
    dates, ledgers, profiles = swaptions()
    exercised = (profiles['SWAP_FIXED'] + profiles['SWAP_FLOAT'])[dates.index(book.WORLD_EXPIRY)] > 0.0
    assert 0 < exercised.sum() < exercised.size, 'in and out of the money across the paths'
    pays = sorted({row['Payment_Date'] for leg in SWPT['Children'] for row in leg['Cashflows']['Items']})
    assert list(ledgers['PHYSICAL'].index) == sorted(pays + [SWPT_PAID]), ledgers['PHYSICAL'].index
    for day in pays:
        delivered, booked = (np.asarray(ledgers[name].loc[day].values, dtype=float) for name in ('SWAP', 'PHYSICAL'))
        assert np.abs(delivered).min() > 0.0, day
        assert np.array_equal(booked, delivered * exercised), day


def test_a_cash_swaption_pays_the_expiry_value_on_its_settlement_date_and_holds_it_between():
    """Settled in cash, the swaption pays `max(U(t0), 0)` - the delivered swap's own mark on the
    expiry row, floored - on its settlement date, path by path, and nothing on any other day; from
    the expiry on it marks that amount discounted to the settlement, the unit paid that day beside
    it on the same path, which is the level the expiry fixed and not a later day's swap.

    Killing mutations: the amount not held past the expiry, the row after it marking that day's
    swap; the settlement not booked, an empty ledger.
    """
    dates, ledgers, profiles = swaptions()
    expiry = dates.index(book.WORLD_EXPIRY)
    paid = np.asarray(ledgers['CASH'].loc[SWPT_PAID].values, dtype=float)
    swap = profiles['SWAP_FIXED'][expiry] + profiles['SWAP_FLOAT'][expiry]
    assert np.allclose(paid, np.maximum(swap, 0.0), rtol=1e-12, atol=1e-8)
    assert (paid > 0.0).any() and (paid == 0.0).any(), 'in and out of the money across the paths'
    assert not np.asarray(ledgers['CASH'].drop(SWPT_PAID).values, dtype=float).any()
    later = [row for row, day in enumerate(dates) if book.WORLD_EXPIRY <= day <= SWPT_PAID]
    assert len(later) == 3, later                    # the expiry, the day after it, the settlement
    for row in later:
        assert np.allclose(profiles['CSWPT'][row], paid * profiles['UNIT'][row], rtol=1e-12, atol=1e-8), row


def test_a_cash_swaption_is_discounted_on_to_its_settlement_and_either_style_expired_refuses():
    """Paid 40 days after its expiry, the cash twin marks what it marks settling on the expiry -
    where the blank date reads it - times the forward discount over the 40 days, `exp(-4% 40/365)`
    on the flat USD curve, to 1e-12. On a base date past the expiry the document refuses naming the
    deal, the day and what it needs: in cash the amount the expiry fixed, stated nowhere; settled
    physically the holder's election, the swap it delivered booked or nothing if it lapsed.

    Killing mutations: the forward discount dropped, 10,361.17 where it marks 10,315.85; either
    refusal removed, the swaption skipped and the job run without it.
    """
    marked, _ = book.marks([CASH_SWPT, dict(CASH_SWPT, Reference='ON_EXPIRY', Settlement_Date=None)])
    forward = math.exp(-0.04 * 40 / 365.0)
    assert float.fromhex(marked['CSWPT']) == pytest.approx(float.fromhex(marked['ON_EXPIRY']) * forward, rel=1e-12)
    for deal, needs in ((CASH_SWPT, 'stated nowhere'), (SWPT, 'nothing if it lapsed')):
        expired, base = book.book([deal]), book.WORLD_EXPIRY + pd.DateOffset(days=10)
        expired['Calc']['Calculation']['Base_Date'] = base
        expired['Calc']['MergeMarketData']['ExplicitMarketData']['System Parameters']['Base_Date'] = base
        context = derivus.Context()
        context.load_json((json.dumps(expired, cls=CustomJsonEncoder), 'expired'))
        with pytest.raises(Exception) as refusal:
            context.run_job()
        message = str(refusal.value)
        assert all(word in message for word in (deal['Reference'], 'Option_Expiry_Date', '2027-08-03', needs)), message


def test_a_par_yield_swaption_is_black_on_its_coupons_discounted_at_the_swap_rate():
    """Settled in cash on `Par_Yield_Curve`, the swaption is struck on its fixed coupons discounted
    at the swap rate s, `A_c(s) = sum N tau prod (1 + tau s)^-1`, and marks the market formula
    `D(0, T) A_c(s) Black(s, k, sigma, t0)`: the declared-defaults swaption struck at its forward,
    payer and receiver, paid 40 days after the expiry, against that by hand off the world's flat
    4% discount and 4.5% projection curves and its flat 20% surface, to 1e-12. The projection
    above the discount is what parts the annuities: on one flat curve they agree to 1e-6.

    Killing mutation: the swap's own annuity restored, 9,707.85 for the payer where it marks
    9,625.17.
    """
    years = lambda day: (day - book.WORLD_BASE).days / 365.0
    discount = lambda day: math.exp(-book.DISCOUNT[0] * years(day))
    fixed, floating = (leg['Cashflows']['Items'] for leg in SWPT['Children'])
    forward = sum(row['Notional'] * math.expm1(book.PROJECTION[0] * row['Accrual_Year_Fraction'])
                  * discount(row['Payment_Date']) for row in floating) / sum(
        row['Notional'] * row['Accrual_Year_Fraction'] * discount(row['Payment_Date']) for row in fixed)
    struck = book.fixed_items(100.0 * forward)
    payer = dict(CASH_SWPT, Reference='PAYER', Cash_Settlement_Method='Par_Yield_Curve',
                 Swap_Rate=100.0 * forward, Children=[
                     dict(SWPT['Children'][0], Reference='PAYER_FIXED',
                          Cashflows=dict(SWPT['Children'][0]['Cashflows'], Items=struck)),
                     dict(SWPT['Children'][1], Reference='PAYER_FLOAT')])
    receiver = dict(payer, Reference='RECEIVER', Payer_Receiver='Receiver', Children=[
        dict(leg, Reference='RECEIVER' + leg['Reference'][5:],
             Buy_Sell={'Buy': 'Sell', 'Sell': 'Buy'}[leg['Buy_Sell']]) for leg in payer['Children']])
    marked, _ = book.marks([payer, receiver])
    black = forward * math.erf(0.20 * math.sqrt(years(book.WORLD_EXPIRY)) / math.sqrt(8.0))
    by_hand = discount(SWPT_PAID) * cash_annuity(forward, struck) * black
    for reference in ('PAYER', 'RECEIVER'):
        assert float.fromhex(marked[reference]) == pytest.approx(by_hand, rel=1e-12), reference


def test_an_amortising_par_yield_swaption_pays_the_cash_annuity_of_the_swap_it_becomes():
    """Settled in cash on `Par_Yield_Curve`, the amortising swaption pays `A_c(S) max(S - k, 0)` on
    its settlement date, path by path, and nothing on any other day. S is the swap rate of the swap
    it becomes on the expiry row - its legs held alone beside it, the floating leg's mark over the
    fixed leg's annuity - and A_c the fixed leg's own coupons, stepping down with its principal,
    discounted at S.

    Killing mutation: the swap's own annuity restored, paying `max(U(t0), 0)`, 22,145.52 on the
    first path that exercises where it owes 21,810.59.
    """
    dates, ledgers, profiles = swaptions()
    expiry, strike = dates.index(book.WORLD_EXPIRY), YIELD_SWPT['Swap_Rate'] / 100.0
    annuity = -profiles['AMORTISED_FIXED'][expiry] / strike
    rate = profiles['AMORTISED_FLOAT'][expiry] / annuity
    paid = np.asarray(ledgers['YIELD'].loc[SWPT_PAID].values, dtype=float)
    owed = cash_annuity(rate, YIELD_SWPT['Children'][0]['Cashflows']['Items']) * np.maximum(rate - strike, 0.0)
    assert np.allclose(paid, owed, rtol=1e-12, atol=1e-8)
    assert (paid > 0.0).any() and (paid == 0.0).any(), 'in and out of the money across the paths'
    assert not np.asarray(ledgers['YIELD'].drop(SWPT_PAID).values, dtype=float).any()


#: Four equity types paying in the payoff currency at the expiry, none stating it
BLANK_PAYOFF = [
    dict({k: v for k, v in VANILLA.items() if k not in ('Payoff_Currency', 'Settlement_Date')},
         Reference='AMER', Option_Type='Put', Strike_Price=120.0, Option_Style='American'),
    {'Object': 'EquityDiscreteExplicitAsianOption', 'Reference': 'ASN', 'Currency': 'USD',
     'Equity': 'EQ', 'Dividends': 'EQ', 'Discount_Rate': 'USD', 'Equity_Volatility': 'EQ',
     'Buy_Sell': 'Buy', 'Option_Type': 'Call', 'Strike_Price': 95.0, 'Units': UNITS,
     'Expiry_Date': EXPIRY,
     'Sampling_Data': [[BASE + pd.DateOffset(months=m), 0.0, 1.0] for m in (3, 6, 9)] + [[EXPIRY, 0.0, 1.0]]},
    {'Object': 'EquityBarrierBinaryOption', 'Reference': 'KOBIN', 'Currency': 'USD',
     'Equity': 'EQ', 'Dividends': 'EQ', 'Discount_Rate': 'USD', 'Equity_Volatility': 'EQ',
     'Buy_Sell': 'Buy', 'Option_Type': 'Call', 'Strike_Price': STRIKE, 'Cash_Payoff': CASH,
     'Barrier_Type': 'Up_And_Out', 'Barrier_Price': 130.0, 'Expiry_Date': EXPIRY,
     'Barrier_Dates': [[BASE + pd.DateOffset(months=m)] for m in range(1, 12)] + [[EXPIRY]]},
    {'Object': 'QEDI_CustomAutoCallSwap', 'Reference': 'AC', 'Currency': 'USD', 'Equity': 'EQ',
     'Dividends': 'EQ', 'Discount_Rate': 'USD', 'Equity_Volatility': 'EQ', 'Buy_Sell': 'Buy',
     'Option_Type': 'Call', 'Strike_Price': STRIKE, 'Expiry_Date': EXPIRY, 'Units': UNITS,
     'Price_Fixing': [[EXPIRY, 0.0]], 'Autocall_Coupons': [[EXPIRY, 0.08]],
     'Autocall_Thresholds': [[EXPIRY, 1.0]]}]

#: A zero-threshold CSA closing out over ten days
CSA = {'Object': 'NettingCollateralSet', 'Reference': 'CSA', 'Netted': 'True', 'Collateralized': 'True',
       'Agreement_Currency': 'USD', 'Balance_Currency': 'USD', 'Liquidation_Period': 10,
       'Settlement_Period': 0, 'Credit_Support_Amounts': dict.fromkeys(
           ('Received_Threshold', 'Posted_Threshold', 'Independent_Amount', 'Minimum_Received',
            'Minimum_Posted'), {'.CreditSupportList': [[0.0, 0.0]]})}


def test_a_blank_payoff_currency_is_the_deal_s_own_currency():
    """`Payoff_Currency` declares its blank the deal's own `Currency`, so a document omitting it is
    the one stating USD: under a credit Monte Carlo the American put, the Asian, the barrier binary
    and the autocall mark and book bit for bit alike either way, 570.74 a path on the expiry, and
    a collateralised set holding the American alone prices as it does.

    Killing mutations, each a reader taking the raw field again: the barrier binary registering its
    dates under it, the ledger's thirteen rows down to two; the Asian booking under it, the expiry
    471.89; the autocall naming its factor off it, skipped, the book's t0 mark 756.032 against
    756.415; the American keying its grid dates under it, a '' ledger beside USD - and, held under
    the CSA, the whole set refused ('Cannot find FxRate.').
    """
    stated = [dict(deal, Payoff_Currency='USD') for deal in BLANK_PAYOFF]
    held = tuple([dict(CSA, Children=deals[:1])] for deals in (BLANK_PAYOFF, stated))
    for pair in ((BLANK_PAYOFF, stated), held):
        blank, usd = (run(deals, calc=CMC) for deals in pair)
        assert np.array_equal(blank['mtm'].values, usd['mtm'].values), pair[0][0]['Reference']
        assert set(blank['cashflows']) == set(usd['cashflows']) == {'USD'}, set(blank['cashflows'])
        paid, owed = blank['cashflows']['USD'], usd['cashflows']['USD']
        assert paid.index.equals(owed.index) and np.array_equal(paid.values, owed.values), (
            pair[0][0]['Reference'], paid.loc[EXPIRY].mean(), owed.loc[EXPIRY].mean())
        if len(pair[0]) > 1:
            assert paid.loc[EXPIRY].mean() == pytest.approx(570.74260938, rel=1e-6)
