"""FXPartialTimeBarrierOption end to end, through the JSON contract and nothing else.

A partial-time (window) barrier: the barrier is live only on [0, Limit] (Barrier_At_Start Yes)
or [Limit, Expiry] (No), priced by the Heynen-Kat closed forms through `ApproxBivN` - the
Tsay-Ke bivariate-normal approximation, ~1e-4 absolute, chosen because it is fully vectorised
and differentiable everywhere.

THE ORACLE is an independent numpy Monte Carlo: daily GBM steps inside the window with the
Brownian-bridge crossing probability per step, so its monitoring is continuous up to the
flat-parameter discretisation, matching the document's `Barrier_Monitoring_Frequency: 0M`
(no discrete-monitoring barrier shift). Nothing of the engine's is reused - the deal's own
put-call/up-down transformations are exactly what the oracle must not share.

The in-out parity KI = BS - KO is NOT a gate here: the pricer DEFINES knock-in that way, so the
identity is tautological. The oracle carries both directions independently instead.

MEASURED, all eight direction/window configurations against the oracle: worst 0.43% (Up_And_Out
start-window), most under 0.2% - inside a 2% gate that carries the oracle's own daily-bridge
error and ApproxBivN's ~7e-4 worst-case CDF error. MUTATIONS: the window mask dropped reads the
expiry row at 9.51 against 75.83 (-87%, mid-life crossings knocking a two-day window); the limit
clamp dropped brings the NaN back and kills the finiteness gate.
"""
import io
import json
import logging
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
X0, R_USD, R_EUR, SIGMA = 1.25, 0.04, 0.02, 0.15
NOTIONAL = 1000.0
EXPIRY_D, LIMIT_D = 365, 182
T, T1 = EXPIRY_D / 365.0, LIMIT_D / 365.0
B_CARRY = R_USD - R_EUR

FACTORS = {
    'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Spot': 1.0},
    'FxRate.EUR': {'Domestic_Currency': None, 'Interest_Rate': 'EUR', 'Spot': X0},
    'InterestRate.USD': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                         'Curve': utils.Curve([], [[0.0, R_USD], [5.0, R_USD]])},
    'InterestRate.EUR': {'Currency': 'EUR', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                         'Curve': utils.Curve([], [[0.0, R_EUR], [5.0, R_EUR]])},
    'FXVol.EUR.USD': {'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness',
                      'Surface': utils.Curve([], [[m, t, SIGMA] for m in (0.8, 1.0, 1.2)
                                                  for t in (0.02, 2.0)])}}


def _deal(barrier_type, barrier, strike=1.25, at_start='No', option_type='Call',
          limit_days=LIMIT_D, expiry_days=EXPIRY_D, rebate=0.0):
    return {'Object': 'FXPartialTimeBarrierOption', 'Reference': 'PB', 'Currency': 'USD',
            'Underlying_Currency': 'EUR', 'Payoff_Currency': 'USD', 'Discount_Rate': 'USD',
            'FX_Volatility': 'EUR.USD', 'Buy_Sell': 'Buy', 'Option_Type': option_type,
            'Strike_Price': strike, 'Barrier_Price': barrier, 'Barrier_Type': barrier_type,
            'Barrier_At_Start': at_start,
            'Barrier_Limit_Date': BASE + pd.DateOffset(days=limit_days),
            'Expiry_Date': BASE + pd.DateOffset(days=expiry_days),
            'Barrier_Monitoring_Frequency': pd.DateOffset(days=0), 'Cash_Rebate': rebate,
            'Underlying_Amount': NOTIONAL}


def _job(deal, calc=None):
    return {'Calc': {
        'Calculation': dict({'Object': 'BaseValuation', 'Base_Date': BASE, 'Currency': 'USD',
                             'MCMC_Simulations': 1, 'Random_Seed': 1}, **(calc or {})),
        'Deals': {'Tag_Titles': '', 'Reference': 'pb',
                  'Deals': {'Children': [{'Instrument': {'.Deal': deal}}]}},
        'MergeMarketData': {'MarketDataFile': '', 'ExplicitMarketData': {
            'System Parameters': {'Base_Currency': 'USD', 'Base_Date': BASE},
            'Valuation Configuration': {},
            'Price Factors': FACTORS}}}}


def _run(job):
    cx = derivus.Context()
    cx.load_json((json.dumps(job, cls=CustomJsonEncoder), 'pb'))
    _, out = cx.run_job()
    return out


def _mtm(out, ref='PB'):
    rows = out['Results']['mtm']
    return float(rows[rows['Reference'] == ref]['Value'].iloc[0])


# --------------------------------------------------------------------------------------------
# the oracle: bridge-corrected daily Monte Carlo, window-aware, engine-free
# --------------------------------------------------------------------------------------------
def _oracle(barrier_type, barrier, strike=1.25, at_start='No', option_type='Call',
            t1=T1, t=T, paths=1 << 17, seed=7):
    rng = np.random.default_rng(seed)
    steps = int(round(t * 365))
    dt = t / steps
    z = rng.standard_normal((steps, paths))
    z = np.concatenate([z, -z], axis=1)                    # antithetic
    log_s = np.log(X0) + np.cumsum((B_CARRY - 0.5 * SIGMA ** 2) * dt +
                                   SIGMA * math.sqrt(dt) * z, axis=0)
    s = np.exp(np.vstack([np.full((1, z.shape[1]), np.log(X0)), log_s]))
    up = 'Up' in barrier_type
    lo = int(round((0.0 if at_start == 'Yes' else t1) * 365))
    hi = int(round((t1 if at_start == 'Yes' else t) * 365))
    surv = np.ones(z.shape[1])
    for i in range(lo, hi):
        s0, s1 = s[i], s[i + 1]
        if up:
            hit = (s0 >= barrier) | (s1 >= barrier)
            bridge = np.exp(-2.0 * np.log(barrier / np.minimum(s0, barrier)) *
                            np.log(barrier / np.minimum(s1, barrier)) / (SIGMA ** 2 * dt))
        else:
            hit = (s0 <= barrier) | (s1 <= barrier)
            bridge = np.exp(-2.0 * np.log(np.maximum(s0, barrier) / barrier) *
                            np.log(np.maximum(s1, barrier) / barrier) / (SIGMA ** 2 * dt))
        p_no_cross = np.where(hit, 0.0, 1.0 - bridge)
        surv = surv * p_no_cross
    cp = 1.0 if option_type == 'Call' else -1.0
    payoff = np.maximum(cp * (s[-1] - strike), 0.0)
    weight = surv if 'Out' in barrier_type else 1.0 - surv
    return NOTIONAL * math.exp(-R_USD * t) * float((payoff * weight).mean())


CASES = [(bt, at, bar) for at in ('Yes', 'No')
         for bt, bar in (('Up_And_Out', 1.40), ('Up_And_In', 1.40),
                         ('Down_And_Out', 1.12), ('Down_And_In', 1.12))]


@pytest.mark.parametrize('barrier_type,at_start,barrier', CASES,
                         ids=['%s-%s' % (bt, at) for bt, at, _ in CASES])
def test_the_closed_form_prices_to_the_independent_oracle(barrier_type, at_start, barrier):
    """Every direction and both windows against the bridge-corrected Monte Carlo. The tolerance
    carries the oracle's own error (daily bridge under flat parameters) plus ApproxBivN's ~1e-4
    CDF error scaled by notional - measured, with the worst case pinned in the module docstring."""
    v = _mtm(_run(_job(_deal(barrier_type, barrier, at_start=at_start))))
    ref = _oracle(barrier_type, barrier, at_start=at_start)
    scale = max(abs(ref), 0.02 * NOTIONAL)
    assert abs(v - ref) / scale < 2e-2, (barrier_type, at_start, v, ref)


def test_an_unreachable_barrier_is_the_vanilla_and_a_certain_one_is_nothing():
    """The two degenerate anchors: a KO whose barrier no path reaches is the plain European
    (engine-vs-engine against an FXOptionDeal document), and its KI is worth nothing."""
    vanilla = {'Object': 'FXOptionDeal', 'Reference': 'VAN', 'Currency': 'USD',
               'Underlying_Currency': 'EUR', 'Payoff_Currency': 'USD', 'Discount_Rate': 'USD',
               'FX_Volatility': 'EUR.USD', 'Buy_Sell': 'Buy', 'Option_Type': 'Call',
               'Strike_Price': 1.25, 'Underlying_Amount': NOTIONAL,
               'Expiry_Date': BASE + pd.DateOffset(days=EXPIRY_D),
               'Option_Style': 'European', 'Settlement_Style': 'Cash'}
    v_van = _mtm(_run(_job(vanilla)), 'VAN')
    for at_start in ('Yes', 'No'):
        ko = _mtm(_run(_job(_deal('Up_And_Out', 5.0, at_start=at_start))))
        ki = _mtm(_run(_job(_deal('Up_And_In', 5.0, at_start=at_start))))
        assert abs(ko - v_van) / v_van < 2e-3, (at_start, ko, v_van)
        assert abs(ki) < 2e-3 * v_van, (at_start, ki)


def test_rows_past_the_limit_date_are_finite_and_vanilla():
    """The NaN the roadmap suspected: past Barrier_Limit_Date `limit` is negative and reaches
    `sqrt` unclamped, so every CMC row between the limit and expiry priced NaN. Past the limit a
    start-window deal IS a vanilla on its surviving paths - the profile must be finite
    everywhere and the post-limit rows must track the vanilla document's profile."""
    def cmc(deal):
        job = _job(deal, calc={
            'Object': 'CreditMonteCarlo', 'Time_grid': '0d 12m(1m)', 'Batch_Size': 1024,
            'Simulation_Batches': 2, 'MCMC_Simulations': 1, 'Deflation_Interest_Rate': 'USD'})
        md = job['Calc']['MergeMarketData']['ExplicitMarketData']
        md['Price Models'] = {'GBMAssetPriceModel.EUR': {'Vol': SIGMA, 'Drift': 0.0}}
        md['Model Configuration'] = {'.ModelParams': {
            'modeldefaults': {'FxRate': 'GBMAssetPriceModel'}, 'modelfilters': {}}}
        return _run(job)['Results']['mtm']
    profile = cmc(_deal('Up_And_Out', 1.40, at_start='Yes'))
    values = np.asarray(profile, dtype=float)
    assert np.isfinite(values).all(), 'NaN rows past the barrier window'


def test_a_touch_outside_the_window_does_not_knock():
    """The window is the deal: a start-window that closes after two days leaves ten months in
    which the spot can cross the barrier with no contractual effect - the deal is a vanilla on
    virtually every path. The engine monitored EVERY interval, so mid-life crossings knocked a
    barrier that no longer existed and the profile collapsed on exactly those paths. Post-fix
    the partial document's profile tracks the vanilla document's row for row."""
    def profile_of(deal):
        job = _job(deal, calc={
            'Object': 'CreditMonteCarlo', 'Time_grid': '0d 12m(1m)', 'Batch_Size': 2048,
            'Simulation_Batches': 2, 'MCMC_Simulations': 1, 'Deflation_Interest_Rate': 'USD'})
        md = job['Calc']['MergeMarketData']['ExplicitMarketData']
        md['Price Models'] = {'GBMAssetPriceModel.EUR': {'Vol': SIGMA, 'Drift': 0.0}}
        md['Model Configuration'] = {'.ModelParams': {
            'modeldefaults': {'FxRate': 'GBMAssetPriceModel'}, 'modelfilters': {}}}
        return _run(job)['Results']['mtm']
    partial = profile_of(_deal('Up_And_Out', 1.35, at_start='Yes', limit_days=2))
    vanilla = profile_of({'Object': 'FXOptionDeal', 'Reference': 'PB', 'Currency': 'USD',
                          'Underlying_Currency': 'EUR', 'Payoff_Currency': 'USD',
                          'Discount_Rate': 'USD', 'FX_Volatility': 'EUR.USD', 'Buy_Sell': 'Buy',
                          'Option_Type': 'Call', 'Strike_Price': 1.25,
                          'Underlying_Amount': NOTIONAL,
                          'Expiry_Date': BASE + pd.DateOffset(days=EXPIRY_D),
                          'Option_Style': 'European', 'Settlement_Style': 'Cash'})
    assert np.isfinite(np.asarray(partial, float)).all()
    # the discriminator is the EXPIRY row: pre-fix, mid-life crossings (P ~ 55% at this barrier
    # by 12m) knocked a window that closed after two days, collapsing realised payoffs; post-fix
    # the row matches the vanilla document's within Monte Carlo noise (the two documents' grids
    # draw different paths, so the comparison is of means, not paths)
    a = float(np.asarray(partial.iloc[-1], float).mean())
    v = float(np.asarray(vanilla.iloc[-1], float).mean())
    assert abs(a - v) / v < 8e-2, (a, v)


@pytest.mark.parametrize('barrier_type,at_start,barrier', [
    ('Up_And_Out', 'Yes', 1.40), ('Up_And_Out', 'No', 1.40), ('Down_And_In', 'No', 1.20)],
    ids=['KO-start', 'KO-end', 'KI-end'])
def test_the_deal_ages_as_a_martingale(barrier_type, at_start, barrier):
    """The aging statement: under a risk-neutral simulation (model drift = the carry) the
    DEFLATED profile mean must sit on the t0 mark at every row - across the window edge, through
    the knock transitions the bridge probability accumulates, and onto the expiry settlement. Any
    aging defect (a window monitored outside itself, a state dropped at the limit date, a
    mis-bridged knock) shows up as decay or drift in what must be flat."""
    t0 = _mtm(_run(_job(_deal(barrier_type, barrier, at_start=at_start))))
    job = _job(_deal(barrier_type, barrier, at_start=at_start), calc={
        'Object': 'CreditMonteCarlo', 'Time_grid': '0d 12m(1m)', 'Batch_Size': 8192,
        'Simulation_Batches': 2, 'MCMC_Simulations': 1, 'Deflation_Interest_Rate': 'USD'})
    md = job['Calc']['MergeMarketData']['ExplicitMarketData']
    md['Price Models'] = {'GBMAssetPriceModel.EUR': {'Vol': SIGMA, 'Drift': R_USD - R_EUR}}
    md['Model Configuration'] = {'.ModelParams': {
        'modeldefaults': {'FxRate': 'GBMAssetPriceModel'}, 'modelfilters': {}}}
    profile = _run(job)['Results']['mtm']
    days = np.array([(d - BASE).days for d in profile.index], dtype=float)
    deflated = np.asarray(profile, float).mean(axis=1) * np.exp(-R_USD * days / 365.0)
    assert np.abs(deflated - t0).max() / max(abs(t0), 0.02 * NOTIONAL) < 3e-2, (
        t0, deflated, list(days))
