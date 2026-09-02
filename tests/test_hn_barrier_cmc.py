"""A barrier deal priced through Credit_Monte_Carlo — the path that had no test at all.

Every other barrier gate prices under BASE VALUATION, which has one deal-time row - so a shape
correction that tests spot's COLUMNS and repeats its ROWS is inert there and squares the grid at
37 (1369 vs 37). `Deal.calculate` swallowed the RuntimeError into a skipped deal, so a barrier in
an exposure calculation silently produced nothing.

Kept small: it proves a barrier still prices across an exposure grid, the one thing base valuation
structurally cannot check.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
import torch

import derivus
from derivus import utils
from derivus.config import Config
from derivus.instruments import construct_instrument
import hn_reference as hnref
from conftest import needs_hn_fused

BASE = pd.Timestamp('2024-06-28')
DTYPE = torch.float64
SPOT, STRIKE, UNITS = 100.0, 100.0, 100.0
_SP = hnref.hn_params_from_targets(
    ann_vol=0.30, persistence=0.94, gamma=350.0, leverage_share=0.7, steps_per_year=252.0)
HN = {'Omega': float(_SP['omega']), 'Alpha': float(_SP['alpha']), 'Beta': float(_SP['beta']),
      'Gamma_Star': float(_SP['gamma_star']),
      'H0': 1.6 * float(utils.hn_stationary_var(
          _SP['omega'], _SP['alpha'], _SP['beta'], _SP['gamma_star']))}


def _cfg_knocked_in(sig=0.25, r=0.05, q=0.01):
    """An up-and-IN call whose barrier sits at HALF the spot: it knocks in at its one barrier
    date, so every LATER row is `all_hit` and its mtm IS the already-hit leg, alone. Base
    valuation cannot reach that leg - at deal time the hit mask is all-False at row 0.

    The equity rides a zero-vol zero-drift GBM, so spot is 100 on every path and row and the
    expected value is a NUMBER: what is under test is the HN law of the REMAINING horizon, not the
    scenario diffusion. `r != q` deliberately - a zero carry would hide the forward-growth
    factor."""
    field = {
        'Object': 'EquityBarrierOption', 'Reference': 'BARR1', 'Currency': 'USD',
        'Payoff_Currency': 'USD', 'Equity': 'EQ', 'Dividends': 'EQ', 'Discount_Rate': 'USD',
        'Equity_Volatility': 'EQ', 'Buy_Sell': 'Buy', 'Option_Type': 'Call',
        'Strike_Price': STRIKE, 'Expiry_Date': BASE + pd.Timedelta(days=365), 'Units': UNITS,
        'Barrier_Type': 'Up_And_In', 'Barrier_Price': 0.5 * SPOT, 'Cash_Rebate': 0.0,
        'Barrier_Dates': [BASE + pd.Timedelta(days=30)],
        'Barrier_Monitoring_Frequency': pd.DateOffset(days=1),
    }
    val = {'EquityBarrierOption': {'SpotModel': 'HestonNandi'}}
    c = Config()
    c.params['System Parameters']['Base_Currency'] = 'USD'
    c.params['System Parameters']['Base_Date'] = BASE
    c.params['Price Factors'] = {
        'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Priority': 1, 'Spot': 1.0},
        'InterestRate.USD': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                             'Curve': utils.Curve([], [[0.0, r], [5.0, r]])},
        'EquityPrice.EQ': {'Spot': SPOT, 'Currency': 'USD', 'Interest_Rate': 'USD',
                           'Issuer': '', 'Respect_Default': 'No', 'Jump_Level': 0.0},
        'DividendRate.EQ': {'Currency': 'USD', 'Floor': None,
                            'Curve': utils.Curve([], [[0.01, q], [5.0, q]])},
        'VolatilityGrid.EQ': {'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness',
                              'Surface': utils.Curve([], [[m, t, sig] for m in (0.8, 1.0, 1.2)
                                                          for t in (0.02, 2.0)])},
        'HestonNandiModelParameters.EQ': dict(HN, Property_Aliases=None),
    }
    c.params['Price Models'] = {'GBMAssetPriceModel.EQ': {'Vol': 0.0, 'Drift': 0.0}}
    c.params['Model Configuration'].append('EquityPrice', (), 'GBMAssetPriceModel')
    c.params['Valuation Configuration'] = val
    c.deals = {'Attributes': {'Reference': 'test', 'Tag_Titles': ''},
               'Deals': {'Children': [{'Instrument': construct_instrument(field, val)}]},
               'Calculation': {'Base_Date': BASE, 'Currency': 'USD'}}
    return c


def _brute_force_vanilla(T, r, q, paths=1 << 20, seed=7):
    """UNITS * E[(S_T - K)^+] * D(0,T) by the daily HN recursion in tests/hn_reference.py.

    Independent of the pricer's closed form: it steps h and the log-spot path by path, the only
    reference that can tell an HN price from a normal at the same aggregate variance. Returns
    (value, standard_error)."""
    n_total = max(round(T * 252.0), 1)
    p = hnref.as_tensors({'omega': _SP['omega'], 'alpha': _SP['alpha'], 'beta': _SP['beta'],
                          'gamma_star': _SP['gamma_star'], 'r': (r - q) * T / n_total})
    payoff = torch.relu(SPOT * torch.exp(hnref.hn_simulate(p, n_total, HN['H0'], paths, seed=seed))
                        - STRIKE) * (UNITS * np.exp(-r * T))
    return float(payoff.mean()), float(payoff.std() / np.sqrt(paths))


def _cfg(hn):
    """Monthly-monitored down-and-out barrier. Static rate and dividend curves against a simulated
    equity - the combination that triggered the shape bug."""
    bdates = [BASE + pd.Timedelta(days=d) for d in range(30, 366, 30)]
    field = {
        'Object': 'EquityBarrierOption', 'Reference': 'BARR1', 'Currency': 'USD',
        'Payoff_Currency': 'USD', 'Equity': 'EQ', 'Dividends': 'EQ', 'Discount_Rate': 'USD',
        'Equity_Volatility': 'EQ', 'Buy_Sell': 'Buy', 'Option_Type': 'Call',
        'Strike_Price': 100.0, 'Expiry_Date': BASE + pd.Timedelta(days=365), 'Units': 100.0,
        'Barrier_Type': 'Down_And_Out', 'Barrier_Price': 80.0, 'Cash_Rebate': 0.0,
        'Barrier_Dates': [d for d in bdates],
        'Barrier_Monitoring_Frequency': pd.DateOffset(days=1),
    }
    val = {'EquityBarrierOption': {'SpotModel': 'HestonNandi'}} if hn else {}
    c = Config()
    c.params['System Parameters']['Base_Currency'] = 'USD'
    c.params['System Parameters']['Base_Date'] = BASE
    c.params['Price Factors'] = {
        'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Priority': 1, 'Spot': 1.0},
        'InterestRate.USD': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                             'Curve': utils.Curve([], [[0.0, 0.02], [5.0, 0.02]])},
        'EquityPrice.EQ': {'Spot': 100.0, 'Currency': 'USD', 'Interest_Rate': 'USD',
                           'Issuer': '', 'Respect_Default': 'No', 'Jump_Level': 0.0},
        'DividendRate.EQ': {'Currency': 'USD', 'Floor': None,
                            'Curve': utils.Curve([], [[0.01, 0.01], [5.0, 0.01]])},
        'VolatilityGrid.EQ': {'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness',
                              'Surface': utils.Curve([], [[m, t, 0.25] for m in (0.8, 1.0, 1.2)
                                                          for t in (0.02, 2.0)])},
        'HestonNandiModelParameters.EQ': dict(HN, Property_Aliases=None),
    }
    c.params['Price Models'] = {}
    c.params['Model Configuration'].append('EquityPrice', (), 'HestonNandiImpliedSpotModel')
    c.params['Valuation Configuration'] = val
    c.deals = {'Attributes': {'Reference': 'test', 'Tag_Titles': ''},
               'Deals': {'Children': [{'Instrument': construct_instrument(field, val)}]},
               'Calculation': {'Base_Date': BASE, 'Currency': 'USD'}}
    return c


def _profile(hn, seed=1, batch=64, sims=1024):
    params = {'Run_Date': BASE.strftime('%Y-%m-%d'), 'Time_grid': '0d 2d 1w(1w) 3m(1m)',
              'Batch_Size': batch, 'Simulation_Batches': 1, 'Random_Seed': seed,
              'Currency': 'USD', 'MCMC_Simulations': sims, 'Tenor_Offset': 0.0,
              'Deflation_Interest_Rate': 'USD'}
    _, out = derivus.run_cmc(_cfg(hn), prec=DTYPE, overrides=params)
    return out['Results']['mtm']


@pytest.mark.parametrize('hn', [pytest.param(True, marks=needs_hn_fused, id='heston_nandi'),
                                pytest.param(False, id='gbm')])
def test_barrier_prices_across_the_exposure_grid(hn):
    """One row per report date, one column per path. The shape bug produced len(deal_time)**2 rows
    and the deal was skipped. It fired for GBM too, hence both ids."""
    mtm = _profile(hn)
    assert mtm.shape[0] > 1, 'exposure profile collapsed to a single row — deal skipped?'
    assert mtm.shape[1] == 64, f'expected one column per path, got {mtm.shape[1]}'
    assert np.isfinite(mtm.values).all(), 'NaN in the exposure profile'
    assert (mtm.values > 0).any(), 'a bought down-and-out barrier should carry positive exposure'
    # monotone decay is not guaranteed, but the profile must not be constant across time
    assert mtm.values.std(axis=1).min() > 0.0, 'no dispersion across paths at some date'


@needs_hn_fused
@pytest.mark.parametrize('row_index', [2, 4])
def test_already_hit_leg_prices_under_the_declared_model(row_index):
    """The already-hit KI leg is HESTON-NANDI, not Black at the implied surface.

    Both vanillas here value the SAME state - a knocked-in barrier is a European - so a Black one
    beside the parity leg's HN one is two models in one payoff, selected element by element by
    ``torch.where(row_barrier_hit, hit_value, oss_result)``. The leg was 19% off before the mix
    was removed.

    The reference is the brute-force daily recursion in ``tests/hn_reference.py``, never the
    pricer's own closed form. Tolerance 5e-3 against a 1.4e-3 standard error, with Black at the
    identical state asserted OUTSIDE it (15.1% and 16.5% low on the two rows), so the gate is not
    a placebo.

    MUTATION KILL MATRIX, measured on row_index=2 (reference 1245.639, se 1.696):

        mutant                                        leg      reldiff   verdict
        (correct)                                 1244.281   -1.09e-03   pass
        (a) leg reverted to Black + vol strip      1265.569   +1.60e-02   KILLED
        (b) hn_put where hn_call belongs            918.707   -2.62e-01   KILLED
        (c) h1 <- stationary variance, not H0      1219.474   -2.10e-02   KILLED
        (d) forward growth exp(r_step*n) dropped   1202.902   -3.43e-02   KILLED
        (e) n_total + 5 daily steps                1255.534   +7.94e-03   KILLED
        (f) vol strip rebuilt under HN             1244.281   -1.09e-03   pass, BY DESIGN

    (f) is the enumeration, not a miss: nothing correct consumes the implied surface under HN. The
    withholding buys (a) - reverting ONLY the leg dies with ``The size of tensor a (8) must match
    the size of tensor b (0)`` and the deal is skipped, rather than mispricing."""
    r, q, sig = 0.05, 0.01, 0.25
    params = {'Run_Date': BASE.strftime('%Y-%m-%d'), 'Time_grid': '0d 2m(2m)',
              'Batch_Size': 8, 'Simulation_Batches': 1, 'Random_Seed': 1, 'Currency': 'USD',
              'MCMC_Simulations': 256, 'Tenor_Offset': 0.0, 'Deflation_Interest_Rate': 'USD'}
    _, out = derivus.run_cmc(_cfg_knocked_in(sig, r, q), prec=DTYPE, overrides=params)
    mtm = out['Results']['mtm']

    row = mtm.index[row_index]
    T = (365 - (pd.Timestamp(row) - BASE).days) / 365.0
    assert T > 0.0, 'row is not strictly before expiry - the leg would carry no optionality'
    priced = mtm.loc[row].values
    assert priced.std() == 0.0, 'deterministic spot must give a flat row'
    ref, se = _brute_force_vanilla(T, r, q)
    assert priced[0] == pytest.approx(ref, rel=5e-3), (
        'already-hit leg %.6f vs HN brute force %.6f (se %.6f)' % (priced[0], ref, se))

    black = float(utils.black_european_option(
        torch.tensor(SPOT * np.exp((r - q) * T), dtype=DTYPE), STRIKE,
        torch.tensor(sig * np.sqrt(T), dtype=DTYPE), 1.0, 1.0, 1.0,
        type('S', (), {'one': torch.tensor(0.0, dtype=DTYPE)}))) * UNITS * np.exp(-r * T)
    assert abs(black / ref - 1.0) > 5e-2, (
        'fixture is a placebo: Black %.6f and HN %.6f agree' % (black, ref))
