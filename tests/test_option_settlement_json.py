"""An option paid after it expires, through the JSON contract and nothing else.

`Settlement_Date` is the day the payoff is paid, the expiry where blank. Between the two the
underlying is what the expiry fixed: on a simulated grid the path's own level on the expiry row,
and on a base date past the expiry the print `Price_Fixing` states - which an expired deal must
state, since the spot of today is not the level it settled on.

THE ORACLES: under a credit Monte Carlo every row from the expiry on is the settlement row's payoff
discounted to it, scenario by scenario, and the ledger books that payoff on the settlement row and
nothing on the expiry's; under a base valuation past the expiry the mark is the print's intrinsic
discounted to the settlement, exact.
"""
import json
import math
import os
import sys

# reference-derivus shadow-import guard (MEMORY): pin the package under test to THIS repo.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

import derivus
from derivus import utils
from derivus.config import CustomJsonEncoder

BASE = pd.Timestamp('2024-06-28')
EXPIRY = pd.Timestamp('2025-06-27')                  # a Friday
SETTLE = EXPIRY + pd.offsets.BDay(2)                 # the Tuesday after
SPOT, STRIKE, UNITS, CASH, PRINT = 100.0, 100.0, 10.0, 1000.0, 112.0
R_USD, Q_EQ, SIGMA = 0.04, 0.01, 0.25

FACTORS = {
    'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Spot': 1.0},
    'InterestRate.USD': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                         'Curve': utils.Curve([], [[0.0, R_USD], [5.0, R_USD]])},
    'EquityPrice.EQ': {'Spot': SPOT, 'Currency': 'USD', 'Interest_Rate': 'USD', 'Issuer': '',
                       'Respect_Default': 'No', 'Jump_Level': 0.0},
    'DividendRate.EQ': {'Currency': 'USD', 'Curve': utils.Curve([], [[0.0, Q_EQ], [5.0, Q_EQ]])},
    'VolatilityGrid.EQ': {'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness',
                          'Surface': utils.Curve([], [[m, t, SIGMA] for m in (0.6, 1.0, 1.4)
                                                      for t in (0.02, 2.0)])}}

VANILLA = {'Object': 'EquityOptionDeal', 'Reference': 'EQO', 'Currency': 'USD',
           'Payoff_Currency': 'USD', 'Equity': 'EQ', 'Dividends': 'EQ', 'Discount_Rate': 'USD',
           'Equity_Volatility': 'EQ', 'Buy_Sell': 'Buy', 'Option_Type': 'Call',
           'Strike_Price': STRIKE, 'Units': UNITS, 'Expiry_Date': EXPIRY, 'Settlement_Date': SETTLE}
BINARY = dict({k: v for k, v in VANILLA.items() if k != 'Units'},
              Object='EquityBinaryOption', Reference='EQB', Payoff=CASH)
CMC = {'Object': 'CreditMonteCarlo', 'Time_grid': '0d 12m(1m)', 'Batch_Size': 256,
       'Simulation_Batches': 1, 'Deflation_Interest_Rate': 'USD', 'Generate_Cashflows': 'Yes'}


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
                             'MCMC_Simulations': 1, 'Random_Seed': 1}, **(calc or {})),
        'Deals': {'Reference': 'settle',
                  'Deals': {'Children': [{'Instrument': {'.Deal': d}} for d in deals]}},
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
