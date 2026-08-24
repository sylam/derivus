"""A sold FX TARF must book the mirror of a bought one in the CASHFLOW LEDGER, not just in mtm.

`pv_MC_Tarf` signs its mtm (`theo_price = buy_sell * block_mtm`) and booked its settled fixing
UNSIGNED, so a sold TARF settled a receipt where it owed a payment. The deal's own mirror could
not see it - the mtm flips correctly and only the ledger was wrong - and every fixture that
reached the settle path was a Buy.

It needs a CREDIT MONTE CARLO to be visible at all: `cash_settle` is a no-op under base valuation
(`shared.t_Cashflows is None`), so the whole by-product is unexercised there.

Pre-registered, from the document below: one fixing observed at 1.15 against a 1.10 strike on
1,000 units, settling on the base date. Undiscounted, that is exactly **50.00** to the buyer and
**-50.00** to the seller, on every scenario.
"""
import json
import math
import os
import sys

# reference-derivus shadow-import guard (MEMORY): pin the package under test to THIS repo.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import derivus as rf

BASE = '2024-06-28'
SPOT, STRIKE, SIGMA = 1.10, 1.10, 0.10
R_USD, Q_EUR = 0.04, 0.02
N1, N2 = 1000.0, 2000.0
OBSERVED = 1.15
EXPECTED_TODAY = N1 * max(OBSERVED - STRIKE, 0.0)      # 50.00, settling on the base date


def _day(offset):
    import datetime
    return (datetime.date(2024, 6, 28) + datetime.timedelta(days=offset)).isoformat()


def _curve(points):
    return {'.Curve': {'meta': [], 'data': points}}


def _job(buy_sell):
    deal = {
        'Object': 'FXTARFOptionDeal', 'Reference': 'T1',
        'Currency': 'USD', 'Underlying_Currency': 'EUR', 'Discount_Rate': 'USD',
        'FX_Volatility': 'EUR.USD', 'Buy_Sell': buy_sell, 'Option_Type': 'Call',
        'Expiry_Date': {'.Timestamp': _day(152)}, 'Underlying_Amount': N1,
        'Strike_Price': STRIKE, 'Settlement_Style': 'Cash', 'Option_Style': 'European',
        'InvertedTarget': False, 'LeverageNotional': N2, 'TargetAdjustment': '',
        'TargetLevel': 0.5, 'Barrier': 0.0,
        # a plain array of rows: the field declares no `tag`, so it arrives as rows and the deal
        # reads it by iterating them. Authored as `{'.DateEqualList': ...}` - which is what the
        # declaration promised before it was reconciled - the decoder hands the deal an object
        # that is not iterable and the TARF cannot be loaded at all.
        'TARF_ExpiryDates': [[{'.Timestamp': _day(-5)}, {'.Timestamp': BASE}, OBSERVED]] +
                            [[{'.Timestamp': _day(30 * k)}, {'.Timestamp': _day(30 * k + 2)}, 0.0]
                             for k in range(1, 6)],
    }
    return {'Calc': {
        'Calculation': {
            'Object': 'CreditMonteCarlo', 'Base_Date': {'.Timestamp': BASE},
            'Currency': 'USD', 'Time_grid': '0d 2m(2m)', 'Batch_Size': 128,
            'Simulation_Batches': 1, 'Random_Seed': 1, 'MCMC_Simulations': 1 << 10,
            'Deflation_Interest_Rate': 'USD', 'Generate_Cashflows': 'Yes'},
        'MergeMarketData': {'ExplicitMarketData': {
            'System Parameters': {'Base_Currency': 'USD', 'Base_Date': {'.Timestamp': BASE}},
            'Price Factors': {
                'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD',
                               'Priority': 1, 'Spot': 1.0},
                'FxRate.EUR': {'Domestic_Currency': None, 'Interest_Rate': 'EUR',
                               'Priority': 2, 'Spot': SPOT},
                'InterestRate.USD': {'Currency': 'USD', 'Day_Count': 'ACT_365',
                                     'Sub_Type': None,
                                     'Curve': _curve([[0.0, R_USD], [5.0, R_USD]])},
                'InterestRate.EUR': {'Currency': 'EUR', 'Day_Count': 'ACT_365',
                                     'Sub_Type': None,
                                     'Curve': _curve([[0.0, Q_EUR], [5.0, Q_EUR]])},
                'FXVol.EUR.USD': {'Surface_Type': 'Explicit',
                                  'Moneyness_Rule': 'Sticky_Moneyness',
                                  'Surface': _curve([[m, t, SIGMA] for m in (0.8, 1.0, 1.2)
                                                     for t in (0.02, 2.0)])}},
            'Price Models': {'GBMAssetPriceModel.EUR': {'Vol': SIGMA, 'Drift': 0.0}},
            'Model Configuration': {'.ModelParams': {
                'modeldefaults': {'FxRate': 'GBMAssetPriceModel'}, 'modelfilters': {}}},
            'Correlations': {}, 'Valuation Configuration': {}}},
        'Deals': {'Reference': 'test', 'Tag_Titles': '',
                  'Deals': {'Children': [{'Instrument': {'.Deal': deal}}]}}}}


def _ledger(buy_sell, tmp_path):
    path = os.path.join(str(tmp_path), f'tarf_{buy_sell}.json')
    with open(path, 'w') as f:
        json.dump(_job(buy_sell), f, default=str)
    cx = rf.Context()
    cx.load_json(path)
    _, out = cx.run_job()
    return out['Results']['cashflows']['USD']


def test_a_sold_tarf_books_the_mirror_of_a_bought_one(tmp_path):
    buy, sell = _ledger('Buy', tmp_path), _ledger('Sell', tmp_path)
    today = float(buy.values[0].mean())
    assert abs(today - EXPECTED_TODAY) < 1e-3, (
        today, EXPECTED_TODAY, 'the fixing settling today must book its exact payoff')
    assert np.array_equal(sell.values, -buy.values), (
        'a sold TARF books the mirror of a bought one - the ledger carries direction exactly as '
        'the mtm does')
    assert float(np.abs(buy.values).sum()) > 0.0, 'the fixture settled nothing'
