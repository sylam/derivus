"""`LogVar2FJSpotModel` - LogVar2FJ's two log-variance factors on a commodity spot, every parameter on
its `Price Models` block - and `LogVar2FJSpotCalibration`, which fits it on a bar series beside the
fix.

Every gate is a document through the public verbs: a calibration document through
`Config.calibrate_factors` on a synthetic archive, a credit Monte Carlo of one future, and the
simulate-only hedging fixture, replayed on an observed scenario or solved on a small batch. The filter
the replay runs and the fit leaves its state in is read off the `LV_REPLAY` debug line, a fork's
dispersion off `LV_FORK`, and each is held against the filter written out here in numpy.
"""
import ast
import io
import json
import logging
import math
import re

import numpy as np
import pandas as pd
import pytest
import torch

import derivus as rf
import test_action_space_and_artifact as solve
from test_chained_basis import _world
from derivus.config import Config, CustomJsonEncoder, ModelParams
from derivus.stochasticprocess import FILLED_ROWS

#: A world block: a fast factor of a three-week half-life and a slow one of eight months, a return
#: that loads +0.6 on the slow shock, and a start above the level.
LV = {'Kappa_S': 12.0, 'Sigma_S': 2.0, 'Kappa_L': 1.0, 'Sigma_L': 0.8, 'L': -2.6, 'Rho_S': -0.15, 'Rho_L': 0.6,
      'Sigma_U': 0.6, 'Offset': -0.05, 'Drift': 2.0e-4, 'Mu': 0.0, 'Ell0': -2.2, 'S0': 0.3,
      'State_Cov': [0.08, 0.004, 0.02], 'Calibration_DT_Years': 1.0 / 252.0}
GK = 2.0 * math.log(2.0) - 1.0


def _step(th, prior, y, r):
    """One market row of the filter, `experiments/plat_sv_toy.py`'s recursion in numpy: the bar's log
    variance `y`, then the return `r` by a Laplace step on log v, then the leverage into the next row."""
    dt = th['Calibration_DT_Years']
    phs, phl = np.exp(-th['Kappa_S'] * dt), np.exp(-th['Kappa_L'] * dt)
    ws = th['Sigma_S'] * np.sqrt(-np.expm1(-2.0 * th['Kappa_S'] * dt) / (2.0 * th['Kappa_S']))
    wl = th['Sigma_L'] * np.sqrt(-np.expm1(-2.0 * th['Kappa_L'] * dt) / (2.0 * th['Kappa_L']))
    m1, m2, p11, p12, p22 = prior
    if y is not None:
        f = p11 + 2 * p12 + p22 + th['Sigma_U'] ** 2
        v = y - m1 - m2 - th['Offset']
        k1, k2 = (p11 + p12) / f, (p12 + p22) / f
        m1, m2, p11, p12, p22 = m1 + k1 * v, m2 + k2 * v, p11 - k1 * k1 * f, p12 - k1 * k2 * f, p22 - k2 * k2 * f
    ee, ve, c1, c2 = 0.0, 1.0, 0.0, 0.0
    if r is not None:
        mg, vg = m1 + m2, p11 + 2 * p12 + p22
        sd = (r - th['Drift']) / np.sqrt(dt)
        a = sd * sd
        g = mg if np.log(a) <= mg else mg + (np.log(a) - mg) * vg / (vg + 2.0)
        for _ in range(6):
            g -= (-(g - mg) / vg - 0.5 + 0.5 * a * np.exp(-g)) / (-1.0 / vg - 0.5 * a * np.exp(-g))
        vp = 1.0 / (1.0 / vg + 0.5 * a * np.exp(-g))
        kg1, kg2 = (p11 + p12) / vg, (p12 + p22) / vg
        m1, m2 = m1 + kg1 * (g - mg), m2 + kg2 * (g - mg)
        p11, p12, p22 = p11 - kg1 * kg1 * (vg - vp), p12 - kg1 * kg2 * (vg - vp), p22 - kg2 * kg2 * (vg - vp)
        ee = sd * np.exp(-0.5 * g + 0.125 * vp)
        ve = sd * sd * np.exp(-g + 0.25 * vp) * np.expm1(0.25 * vp)
        c1, c2 = kg1 * -0.5 * vp * ee, kg2 * -0.5 * vp * ee
    rs, rl, a1, a2 = th['Rho_S'], th['Rho_L'], ws * th['Rho_S'], wl * th['Rho_L']
    return (phs * m1 + a1 * ee, th['L'] + phl * (m2 - th['L']) + a2 * ee,
            phs * phs * p11 + ws * ws * (1 - rs * rs) + a1 * a1 * ve + 2 * phs * c1 * a1,
            phs * phl * p12 - ws * wl * rs * rl + a1 * a2 * ve + phs * c1 * a2 + a1 * c2 * phl,
            phl * phl * p22 + wl * wl * (1 - rl * rl) + a2 * a2 * ve + 2 * phl * c2 * a2)


def _measured(px, high, low, close):
    """Each row's bar measurement, the previous row's Garman-Klass on the last close before it, and its
    log return; None where none."""
    y, r, last, before = [None] * len(px), [None] * len(px), np.nan, []
    for k in range(len(px)):
        before.append(last)
        last = close[k] if close[k] == close[k] else last
    for k in range(1, len(px)):
        gk = 0.5 * np.log(high[k - 1] / low[k - 1]) ** 2 - GK * np.log(close[k - 1] / before[k - 1]) ** 2
        y[k] = float(np.log(252.0 * gk)) if gk > 0 else None
        r[k] = float(np.log(px[k] / px[k - 1])) if px[k] != px[k - 1] else None
    return y, r


def _bars(rng, n, vol=0.02):
    """A bar series: a close drawn at `vol` a day, a high and a low either side, the eleventh bar flat."""
    close = 2000.0 * np.exp(np.cumsum(rng.normal(0.0, vol, n)))
    spread = np.exp(np.abs(rng.normal(0.0, 0.8 * vol, n)))
    high, low = close * spread, close / spread
    high[10] = low[10] = close[10]
    return high, low, close


def _replayed(tmp_path, caplog, observed, block=LV):
    """The filter rows `LV_REPLAY` publishes replaying `observed` (npz key to array) on the simulate-only
    hedging fixture, rolling daily from Friday 2026-04-10, its spot `block`."""
    path = str(tmp_path / 'observed.npz')
    np.savez(path, **observed)
    cfg = json.load(open(solve.FIXTURE))
    cfg['Calc']['Calculation'].update({'Batch_Size': 2, 'Observed_Scenario': path})
    market = cfg['Calc']['MergeMarketData']['ExplicitMarketData']
    market['Price Models'] = {'LogVar2FJSpotModel.PLATINUM_LME': block,
                              'BasisLinkedSpotModel.PLATINUM_LME.LME_CME': solve.SLOW_BASIS}
    market['Model Configuration'] = {'.ModelParams': {
        'modeldefaults': {'CommodityPrice': 'LogVar2FJSpotModel'}, 'modelfilters': {}}}
    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        rf.Context().load_json((json.dumps(cfg), 'replay.json')).run_job()
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith('LV_REPLAY'))
    return np.array(ast.literal_eval(line.split(' ', 2)[2])['filter'])


def _tape(rng):
    """A fix on 199 business days from Friday 2026-04-10, London's 4 May bank holiday not among them, a
    flat fix on the fourth, and a PL1-like bar beside it, the eleventh flat and none on 3 July, a
    CME holiday, and one on 4 May, which CME trades: `(business days, fix, high, low, close, the
    observed scenario on calendar rows, the rows the London fixing calendar filled flagged)`."""
    days = pd.bdate_range('2026-04-10', periods=200).drop(pd.Timestamp('2026-05-04'))
    spot = 2055.0 * np.exp(np.cumsum(np.concatenate([[0.0], rng.normal(0.0, 0.02, len(days) - 1)])))
    spot[3] = spot[2]
    high, low, close = _bars(rng, len(days))
    high[days.get_loc(pd.Timestamp('2026-07-03'))] = low[days.get_loc(pd.Timestamp('2026-07-03'))] = np.nan
    close[days.get_loc(pd.Timestamp('2026-07-03'))] = np.nan
    calendar = pd.date_range('2026-04-10', periods=200)
    row = np.searchsorted(days, calendar, side='right') - 1
    holiday = calendar.get_loc(pd.Timestamp('2026-05-04'))
    own_day = lambda x: np.where(calendar.isin(days), x[row], np.nan)  # noqa: E731
    cme_day = lambda x: np.where(np.arange(len(calendar)) == holiday, 1.01 * x[row], own_day(x))  # noqa: E731
    return days, spot, high, low, close, row, {
        'CommodityPrice.PLATINUM_LME': spot[row], 'ObservedBasis.PLATINUM_LME.LME_CME': -30.0 + 0.0 * spot[row],
        'CommodityPrice.PLATINUM_LME,HIGH': cme_day(high), 'CommodityPrice.PLATINUM_LME,LOW': cme_day(low),
        'CommodityPrice.PLATINUM_LME,CLOSE': cme_day(close), FILLED_ROWS: ~calendar.isin(days)}


def test_a_replay_is_the_filter_on_the_bars_before_each_fix(tmp_path, caplog):
    """The simulate-only hedging fixture replays a `LogVar2FJSpotModel` spot on 119 calendar rows of an
    observed fix, the weekends and London's 4 May bank holiday flagged forward-filled, a flat fix on the
    fourth business day, with a PL1-like bar beside it on its own business day (NaN on the weekends,
    a CME print on 4 May, the eleventh bar flat, none on 3 July, a CME holiday). On every row the
    published filter state - the
    (s, ell) means and their covariance for the next step - equals, within 1e-6 (5e-15
    measured), `experiments/plat_sv_toy.py`'s recursion written here in numpy on the business days
    alone, as the calibration's archive carries them, from the block's `S0`, `Ell0` and `State_Cov`:
    each market row's bar is the previous market row's, its close-to-close term on the last close the
    bar carries, the first unmeasured, a repeated fix is no return, and a filled row holds the row
    before and its bar is unread. 3 s.

    Killing mutations: bar t read at row t (0.68 off); the returns left out of the filter (0.46);
    the replay started from the stationary law (0.44); the old rule, a repeated fix with no bar
    filled, so the holiday's CME bar is read (0.474); the flag unread (0.469)."""
    days, spot, high, low, close, row, observed = _tape(np.random.default_rng(5))
    replay = _replayed(tmp_path, caplog, observed)
    y, r = _measured(spot.astype(np.float32).astype(np.float64), high, low, close)
    states = [(LV['S0'], LV['Ell0'], *LV['State_Cov'])]
    for k in range(1, row[len(replay) - 1] + 1):
        states.append(_step(LV, states[-1], y[k], r[k]))
    worst = float(np.abs(replay - np.array([states[row[t]] for t in range(len(replay))])).max())
    assert len(replay) == 119 and worst < 1e-6, (len(replay), worst)


def _fit(tmp_path, dates, px, bars, **tuning):
    """The `LogVar2FJSpotModel.PLAT` block a calibration document fits under `tuning` on the archive of
    the fix `px` and the bar columns `bars` dated `dates`."""
    archive = pd.DataFrame({'CommodityPrice.PLAT': px, **{f'CommodityPrice.PLAT,{k}': v for k, v in
                                                          zip(('HIGH', 'LOW', 'CLOSE'), bars)}}, index=dates)
    archive.index.name = 'Date'
    archive.to_csv(tmp_path / 'archive.csv', sep='\t')
    retrieval = {'ID': '', 'Use_Pre_Computed_Statistics': 'Yes', 'Data_Retrieval_Parameters': {
        'Frequency': '1d', 'Calendar': '', 'Business_Days_In_Year': 252}}
    document = {
        'MarketData': {
            'System Parameters': {'Base_Currency': 'USD', 'Base_Date': dates[-1]},
            'Model Configuration': ModelParams(({'CommodityPrice': 'LogVar2FJSpotModel'}, {})),
            'Price Factors': {}, 'Price Models': {}, 'Correlations': {},
            'Price Factor Interpolation': ModelParams(({}, {})),
            'Bootstrapper Configuration': {}, 'Market Prices': {}},
        'Version': ['JSONVersion', '22.05.30'],
        'CalibrationConfig': {
            'MarketDataArchiveFile': {'name': str(tmp_path / 'archive.csv'), 'skiprows': 0, 'index_column': 'Date'},
            'Calibrations': {'LogVar2FJSpotModel': dict(retrieval, Method='LogVar2FJSpotCalibration', **tuning)}}}
    with io.open(tmp_path / 'calibration.json', 'w', encoding='utf-8') as handle:
        handle.write(json.dumps(document, indent=1, cls=CustomJsonEncoder))
    config = Config()
    config.parse_json(str(tmp_path / 'calibration.json'))
    factors = config.fetch_all_calibration_factors()
    config.calibrate_factors(dates[0], dates[-1], dict(factors['present'], **factors['absent']))
    return config.params['Price Models']['LogVar2FJSpotModel.PLAT']


def _drawn(rng, n, th):
    """`n` business days drawn from the model `th`: the fix's returns at the variance e^(s + ell) a year,
    and a bar beside it whose Garman-Klass is that variance times e^(Offset + Sigma_U u), the next row's
    measurement: its close a walk of its own, its range what the estimator then needs."""
    dt, s, ell = 1.0 / 252.0, 0.0, th['L']
    logv, eps = np.empty(n), rng.standard_normal(n)
    for k in range(n):
        logv[k] = s + ell
        z = rng.standard_normal(2)
        s = np.exp(-th['Kappa_S'] * dt) * s + th['Sigma_S'] * np.sqrt(-np.expm1(-2 * th['Kappa_S'] * dt) / (2 * th['Kappa_S'])) * (
            th['Rho_S'] * eps[k] + np.sqrt(1 - th['Rho_S'] ** 2) * z[0])
        ell = th['L'] + np.exp(-th['Kappa_L'] * dt) * (ell - th['L']) + th['Sigma_L'] * np.sqrt(
            -np.expm1(-2 * th['Kappa_L'] * dt) / (2 * th['Kappa_L'])) * (th['Rho_L'] * eps[k] + np.sqrt(1 - th['Rho_L'] ** 2) * z[1])
    px = 1000.0 * np.exp(np.cumsum(np.exp(0.5 * logv) * np.sqrt(dt) * eps))
    close = 2000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.015, n)))
    gk = np.exp(np.roll(logv, -1) + th['Offset'] + th['Sigma_U'] * rng.standard_normal(n)) * dt
    span = np.sqrt(2.0 * (gk + GK * np.log(close / np.roll(close, 1)) ** 2))
    return px, (close * np.exp(0.5 * span), close * np.exp(-0.5 * span), close)


def test_the_fit_writes_the_filter_s_state_after_its_last_row(tmp_path):
    """A calibration document fits `LogVar2FJSpotModel` by `LogVar2FJSpotCalibration` on five years of a
    fix and a bar beside it drawn from the model, a flat fix and twenty blank bars among them. The
    written block is the fit's own law with the state the filter leaves after the last row: `S0`, `Ell0`
    and `State_Cov` equal, within 1e-9 (2.0e-14 measured), the numpy recursion run here from the
    stationary law with the written parameters over every row, the bar the previous row's and the return
    a second measurement. Under `Slow_Window_Years` 5 and `Window_Years` 2 `Kappa_L` and `Sigma_L` are the
    five-year fit's to the bit and the rest, `Kappa_S` with it, is fitted on the last two years; a declared
    `Kappa_S_Max` of 20 is where `Kappa_S` lands (29.8 free). 35 s.

    Killing mutations: the state of the row before the last (0.079); bar t at row t (0.094); the end
    state off the bars alone (0.38); the slow window unread (Kappa_L 1.035 against 0.653); the ceiling
    unread (29.8)."""
    n = 1260
    dates = pd.bdate_range('2021-01-04', periods=n)
    px, (high, low, close) = _drawn(np.random.default_rng(7), n, dict(LV, Kappa_S=40.0, Sigma_S=3.0))
    px[100] = px[99]
    high[200:220] = low[200:220] = close[200:220]
    fitted = _fit(tmp_path, dates, px, (high, low, close))
    y, r = _measured(px, high, low, close)
    state = (0.0, fitted['L'], fitted['Sigma_S'] ** 2 / (2 * fitted['Kappa_S']), 0.0,
             fitted['Sigma_L'] ** 2 / (2 * fitted['Kappa_L']))
    for k in range(n):
        state = _step(fitted, state, y[k], r[k])
    written = (fitted['S0'], fitted['Ell0'], *fitted['State_Cov'])
    assert max(abs(a - b) for a, b in zip(written, state)) < 1e-9, (written, state)
    assert fitted['Mu'] == 0.0 and abs(fitted['Drift'] - np.mean([x for x in r if x is not None])) < 1e-15

    windows = _fit(tmp_path, dates, px, (high, low, close), Window_Years=2.0, Slow_Window_Years=5.0)
    assert (windows['Kappa_L'], windows['Sigma_L']) == (fitted['Kappa_L'], fitted['Sigma_L']), (windows, fitted)
    assert windows['Kappa_S'] != fitted['Kappa_S'] and windows['L'] != fitted['L'], (windows, fitted)
    capped = _fit(tmp_path, dates, px, (high, low, close), Kappa_S_Max=20.0)
    assert abs(capped['Kappa_S'] - 20.0) < 1e-9 < abs(fitted['Kappa_S'] - 20.0), (capped['Kappa_S'], fitted['Kappa_S'])


def test_a_commodity_with_no_drift_is_a_martingale():
    """A credit Monte Carlo of one future on a `LogVar2FJSpotModel` fix at `Mu` 0, its factor stating a
    rate and a carry curve, simulates 4,096 paths a year out on a monthly grid at a 48% vol: the mean of
    S_T / S_0 is within two standard errors of one (1.0033, se 0.0087), since the walk's compensator is
    the variance the rotated rows realise and nothing reads the curves. 1 s.

    Killing mutation: the residual given the variance shocks not divided by its sd sqrt(c) (0.9729,
    3.8 se)."""
    world = _world('LBMA_AM', chained=False)
    world['Calc']['Calculation'].update(Batch_Size=4096, Time_Grid='0d 1m(1m)', Calc_Scenarios='All')
    future = world['Calc']['Deals']['Deals']['Children'][0]['Children'][0]['Instrument']['.Deal']
    future['Maturity_Date'] = {'.Timestamp': '2027-01-15'}
    market = world['Calc']['MergeMarketData']['ExplicitMarketData']
    market['Model Configuration'] = {'.ModelParams': {
        'modeldefaults': {'CommodityPrice': 'LogVar2FJSpotModel'}, 'modelfilters': {}}}
    market['Price Models'] = {'LogVar2FJSpotModel.LBMA_AM': dict(LV, L=-1.7, Ell0=-1.6)}
    cx = rf.Context(path_transform={}, file_transform={})
    cx.load_json((json.dumps(world), 'martingale.json'))
    s = cx.run_job()[1]['Results']['scenarios']['CommodityPrice.LBMA_AM'].values
    ratio = s[:, -1] / s[:, 0]
    se = ratio.std() / np.sqrt(len(ratio))
    assert s.shape[1] == 13 and abs(ratio.mean() - 1.0) < 2.0 * se, (ratio.mean(), se)


def _solved(block, save=None, load=None):
    """The solve fixture at 48 paths swept from step 100, its spot `block` (a `LogVar2FJSpotModel` one
    where it states `Kappa_S`, else a `GARCHSpotModel` one) over the slow-mean basis, from the
    calibrated t0."""
    cfg = solve._cfg(batch=48, save=save, load=load)
    cfg['Calc']['Calculation']['Hedging_Problem']['Randomize_Initial_State'] = 'No'
    model = 'LogVar2FJSpotModel' if 'Kappa_S' in block else 'GARCHSpotModel'
    market = cfg['Calc']['MergeMarketData']['ExplicitMarketData']
    market['Price Models'] = {f'{model}.PLATINUM_LME': block, 'BasisLinkedSpotModel.PLATINUM_LME.LME_CME': solve.SLOW_BASIS}
    market['Model Configuration'] = {'.ModelParams': {'modeldefaults': {'CommodityPrice': model}, 'modelfilters': {}}}
    return solve._run(cfg).evaluation_summary['diagnostics']


GARCH = {'Omega': 1.0e-5, 'Alpha': 0.05, 'Beta': 0.9, 'Nu': 8.0, 'H0': 2.0e-4, 'Calibration_DT_Years': 1.0 / 252.0}


def test_a_fork_starts_from_the_outer_path_s_state(caplog):
    """A solve on a `LogVar2FJSpotModel` world forks each outer path at every swept step from that path's
    own state: across the outer paths and steps, the log of each fork's spread of its first one-day log
    return regressed on half the path's s + ell has slope 1 within 0.2 (1.001 measured, 1,224 forks of
    8 draws), the spread of a Gaussian residual being sqrt(v dt). 6 s.

    Killing mutation: the fork reading the outer run's seed keys, so every fork starts from the block's
    `Ell0` and `S0` (slope 0, the state one value)."""
    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        _solved(LV)
    forks = [ast.literal_eval(r.getMessage().split(' ', 1)[1]) for r in caplog.records
             if r.getMessage().startswith('LV_FORK')]
    state = 0.5 * np.concatenate([f['state'] for f in forks])
    spread = np.log(np.concatenate([f['sd'] for f in forks]))
    slope = np.polyfit(state, spread, 1)[0] if state.std() > 0 else 0.0
    assert len(state) >= 300 and abs(slope - 1.0) < 0.2, (len(state), slope)


def test_the_value_net_reads_the_filtered_state_and_refuses_a_garch_checkpoint(tmp_path):
    """The value net reads a `LogVar2FJSpotModel` spot as the filtered slow factor, the filtered total
    log variance and the price, one column more than a GARCH spot's log h and price: on the same
    document the market dimension is 7 against 6, and a value function fitted in the GARCH world is
    refused in the LogVar2FJ one by the shape check naming 'md'. 15 s.

    Killing mutation: the filtered total dropped from the reveal (5 against 5, the GARCH checkpoint
    loading)."""
    garch = str(tmp_path / 'value_fn_garch.pt')
    plain = _solved(GARCH, save=garch)
    lv = _solved(LV)
    assert lv['market_dim'] == plain['market_dim'] + 1, (lv['market_dim'], plain['market_dim'])
    with pytest.raises(ValueError, match=re.escape(
            f"checkpoint mismatch on 'md': {garch} saved {plain['market_dim']} vs this run {lv['market_dim']}")):
        _solved(LV, load=garch)
    assert torch.load(garch, map_location='cpu', weights_only=False)['md'] == plain['market_dim']


def test_a_replayed_fork_draws_its_state_from_the_filter_s_posterior(tmp_path, caplog):
    """A solve on a replayed `LogVar2FJSpotModel` path, its bars noisy and its fast factor wide so the
    filter's posterior of log v is wide, forks each replayed row at every swept step; a replay has no
    true state, so each fork path draws its own start from the posterior N(mean, P). Pooled over the
    forks, the variance of a fork's first one-day log return is the posterior-mixed forecast
    dt exp(mean + P/2), within 4% (1.0025 measured over 102 forks of 512 draws), where the
    mean's own exp(mean) reads 0.870 of it. 10 s.

    Killing mutation: each fork started from the posterior mean (0.872 of the mixed forecast)."""
    path = str(tmp_path / 'observed.npz')
    np.savez(path, **_tape(np.random.default_rng(5))[-1])
    cfg = solve._cfg(batch=4, inner=512)
    calc = cfg['Calc']['Calculation']
    calc['Observed_Scenario'] = path
    calc['Hedging_Problem']['Randomize_Initial_State'] = 'No'
    market = cfg['Calc']['MergeMarketData']['ExplicitMarketData']
    market['Price Models'] = {'LogVar2FJSpotModel.PLATINUM_LME': dict(LV, Sigma_S=4.0, Sigma_U=1.5),
                              'BasisLinkedSpotModel.PLATINUM_LME.LME_CME': solve.SLOW_BASIS}
    market['Model Configuration'] = {'.ModelParams': {
        'modeldefaults': {'CommodityPrice': 'LogVar2FJSpotModel'}, 'modelfilters': {}}}
    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        solve._run(cfg)
    forks = [ast.literal_eval(r.getMessage().split(' ', 1)[1]) for r in caplog.records
             if r.getMessage().startswith('LV_FORK')]
    mean, var, sd = (np.concatenate([f[x] for f in forks]) for x in ('mean', 'var', 'sd'))
    seen, mixed, point = (sd ** 2).sum(), np.exp(mean + 0.5 * var).sum() / 365.25, np.exp(mean).sum() / 365.25
    assert len(sd) >= 50 and abs(seen / mixed - 1.0) < 0.04 and point / mixed < 0.9, (len(sd), seen / mixed, point / mixed)
