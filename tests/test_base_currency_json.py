"""Only the base currency's FX rate is static (it is identically one); its curve simulates like any
other. Found 2026-09-03: `find_models` excluded every factor NAMED by the base currency, so a
USD-base book could not simulate USD rates whatever model it declared - a USD float leg under a
credit Monte Carlo read its resets one scenario wide and was skipped.

The oracle is the code's own dispersion: a par swap's exposure profile has zero spread across
scenarios when its curve is static and a positive one when it is simulated. Degeneracy: r = 4% so
the discount is live; the swap is at par so row 0 is near zero and the later rows are the risk;
one netting set, one currency - the axis under test is the base flag alone.
"""
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import derivus
import rates_world as rw
from derivus import utils
from derivus.config import CustomJsonEncoder

BASE = rw.BASE
HW1F = {'Alpha': 0.1, 'Lambda': 0.0, 'Sigma': utils.Curve([], [[1.0 / 365.0, 0.01], [5.0, 0.01]]),
        'Quanto_FX_Correlation': 0.0, 'Quanto_FX_Volatility': utils.Curve([], [[1.0 / 365.0, 0.0], [5.0, 0.0]])}


def job(base, fx_model=False):
    """A 2y USD par swap under a credit Monte Carlo on a book whose base currency is `base`, with a
    Hull-White model declared on USD and, optionally, a GBM model declared on `FxRate.USD`."""
    factors = {'FxRate.USD': {'Domestic_Currency': None if base == 'USD' else base, 'Interest_Rate': 'USD',
                              'Priority': 1, 'Spot': 1.0 if base == 'USD' else 0.92},
               'InterestRate.USD': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                                    'Curve': utils.Curve([], [[1.0 / 365.0, 0.04], [5.0, 0.04]])}}
    if base != 'USD':
        factors['FxRate.' + base] = {'Domestic_Currency': None, 'Interest_Rate': base, 'Priority': 1, 'Spot': 1.0}
        factors['InterestRate.' + base] = {'Currency': base, 'Day_Count': 'ACT_365', 'Sub_Type': None,
                                           'Curve': utils.Curve([], [[1.0 / 365.0, 0.02], [5.0, 0.02]])}
    models = {'HullWhite1FactorInterestRateModel.USD': dict(HW1F)}
    defaults = {'InterestRate': 'HullWhite1FactorInterestRateModel'}
    if fx_model:
        models['GBMAssetPriceModel.USD'] = {'Vol': 0.1, 'Drift': 0.0}
        defaults['FxRate'] = 'GBMAssetPriceModel'
    market = {'System Parameters': {'Base_Currency': base, 'Base_Date': BASE}, 'Valuation Configuration': {},
              'Price Factors': factors, 'Price Models': models,
              'Model Configuration': {'.ModelParams': {'modeldefaults': defaults, 'modelfilters': {}}}}
    calc = {'Object': 'CreditMonteCarlo', 'Base_Date': BASE, 'Currency': base, 'Time_grid': '0d 2y(3m)',
            'Batch_Size': 64, 'Simulation_Batches': 1, 'Random_Seed': 1, 'Deflation_Interest_Rate': base}
    return {'Calc': {'Calculation': calc,
                     'Deals': {'Tag_Titles': '', 'Reference': 'base', 'Deals': {'Children': [
                         {'Instrument': {'.Deal': rw.par_swap('SW', 'USD', 'USD', 'USD', 2, 4.0)}}]}},
                     'MergeMarketData': {'MarketDataFile': '', 'ExplicitMarketData': market}}}


def run(doc, tmp_path, name):
    path = os.path.join(str(tmp_path), name + '.json')
    with open(path, 'w') as f:
        f.write(json.dumps(doc, cls=CustomJsonEncoder))
    cx = derivus.Context()
    cx.load_json(path)
    _, out = cx.run_job()
    profile = out['Results']['mtm'].values
    logging.debug('%s: profile %s, spread across scenarios per row %s', name, profile.shape,
                  [round(float(x), 4) for x in profile.std(axis=1)])
    return cx, profile


def test_the_base_currency_s_curve_simulates_like_any_other(tmp_path):
    """A USD swap on a USD-base book with a Hull-White model on USD walks a DISPERSED profile - the
    same document on an EUR-base book is the witness that dispersion is what a simulated curve
    looks like here. Before the fix the USD-base run had nothing to simulate at all."""
    _, usd_base = run(job('USD'), tmp_path, 'usd_base')
    _, eur_base = run(job('EUR'), tmp_path, 'eur_base')
    assert usd_base.shape[0] > 4 and np.isfinite(usd_base).all()
    assert usd_base[4].std() > 0.0 and eur_base[4].std() > 0.0, (usd_base[4].std(), eur_base[4].std())
    assert usd_base[0].std() == 0.0, 'row 0 is today: one number across scenarios'


def test_the_base_currency_s_fx_rate_stays_static_whatever_model_is_declared(tmp_path):
    """`FxRate.USD` on a USD book is identically one: a GBM model declared for it is ignored and it
    never enters the stochastic set, while the curve beside it does."""
    cx, _ = run(job('USD', fx_model=True), tmp_path, 'usd_base_fx_model')
    cfg = cx.current_cfg
    _, stochastic, *_ = cfg.calculate_dependencies(cfg.deals['Calculation'], pd.Timestamp(BASE), '0d', False)
    factors = {(m.type, f.type, f.name) for m, f in stochastic.items()}
    logging.debug('stochastic set on the USD-base book: %s', sorted(factors))
    assert ('HullWhite1FactorInterestRateModel', 'InterestRate', ('USD',)) in factors, factors
    assert not any(f.type == 'FxRate' and f.name == ('USD',) for _, f in stochastic.items()), factors
