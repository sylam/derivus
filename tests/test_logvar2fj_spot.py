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


def _flagged(filled):
    """`filled` as the forward-fill flag of the spot's column and the basis's."""
    return {f'{n},{FILLED_ROWS}': filled for n in ('CommodityPrice.PLATINUM_LME', 'ObservedBasis.PLATINUM_LME.LME_CME')}


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


def _replayed(tmp_path, caplog, observed, block=LV, batch=2):
    """The filter rows `LV_REPLAY` publishes replaying `observed` (npz key to array) on the simulate-only
    hedging fixture at `Batch_Size` `batch`, rolling daily from Friday 2026-04-10, its spot `block`."""
    path = str(tmp_path / 'observed.npz')
    np.savez(path, **observed)
    cfg = json.load(open(solve.FIXTURE))
    cfg['Calc']['Calculation'].update({'Batch_Size': batch, 'Observed_Scenario': path})
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
        'CommodityPrice.PLATINUM_LME,CLOSE': cme_day(close), **_flagged(~calendar.isin(days))}


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


def test_a_replay_is_the_compressed_tape_s_on_its_market_rows_whatever_the_batch(tmp_path, caplog):
    """The 119-row calendar tape of `test_a_replay_is_the_filter_on_the_bars_before_each_fix` (weekends
    and London's 4 May holiday flagged forward-filled, a CME bar on the holiday, a flat fix on a
    business day with its bar, no bar on 3 July): its filter rows at `Batch_Size` 5 equal those at 2 to the bit, and on its 84
    market rows they equal, to the bit, the rows of the same tape compressed to its business days. 6 s.

    Killing mutation: every calendar row a market row (the calendar's market rows leave the
    compressed tape's at the first weekend)."""
    days, spot, high, low, close, row, observed = _tape(np.random.default_rng(5))
    calendar = _replayed(tmp_path, caplog, observed)
    assert np.array_equal(calendar, _replayed(tmp_path, caplog, observed, batch=5))
    n = len(calendar)
    compressed = _replayed(tmp_path, caplog, {
        'CommodityPrice.PLATINUM_LME': spot[:n], 'ObservedBasis.PLATINUM_LME.LME_CME': -30.0 + 0.0 * spot[:n],
        'CommodityPrice.PLATINUM_LME,HIGH': high[:n], 'CommodityPrice.PLATINUM_LME,LOW': low[:n],
        'CommodityPrice.PLATINUM_LME,CLOSE': close[:n]})
    market = [t for t in range(n) if t == 0 or row[t] != row[t - 1]]
    assert len(market) == 84 and np.array_equal(calendar[market], compressed[:len(market)])


def test_a_scenario_without_bars_replays_as_one_whose_bars_are_blank(tmp_path, caplog):
    """The calendar tape replayed with no `<factor>,HIGH/LOW/CLOSE` keys and with all three stated
    and blank: the filter rows are equal to the bit and finite, the returns alone measured. 4 s.

    Killing mutation: an absent bar series read as a flat bar of 1.0 rather than blank (every row a
    market row where the keys are absent)."""
    observed = _tape(np.random.default_rng(5))[-1]
    bars = [k for k in observed if k.split(',')[-1] in ('HIGH', 'LOW', 'CLOSE')]
    plain = {k: v for k, v in observed.items() if k not in bars}
    blank = dict(plain, **{k: np.full(len(observed[k]), np.nan) for k in bars})
    a, b = _replayed(tmp_path, caplog, plain), _replayed(tmp_path, caplog, blank)
    assert np.isfinite(a).all() and np.array_equal(a, b)


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


#: A fast factor of a ten-day half-life, a slow one of five months, a return loading -0.35 on the fast
#: shock and +0.5 on the slow, a bar 12% under the variance it measures at sd 0.45 in logs.
TRUE = {'Kappa_S': 25.0, 'Sigma_S': 2.5, 'Kappa_L': 1.5, 'Sigma_L': 0.9, 'L': -2.8, 'Rho_S': -0.35, 'Rho_L': 0.5,
        'Sigma_U': 0.45, 'Offset': -0.12}
#: Each parameter's sd over nine seeds of the 2,520-row tape (seeds 1-9, on the CPU).
SPREAD = {'Kappa_S': 5.59, 'Sigma_S': 0.215, 'Kappa_L': 0.46, 'Sigma_L': 0.18, 'L': 0.21, 'Rho_S': 0.044,
          'Rho_L': 0.144, 'Sigma_U': 0.0081, 'Offset': 0.026}


def _law_tape(seed, n, th=TRUE, dt=1.0 / 252.0):
    """`n` business days of the documented law: e^(s + ell) a year drives row k's return, whose
    innovation loads on the shocks into row k + 1 by `Rho_S` and `Rho_L`, the two shocks uncorrelated;
    row k's bar, closing after fix k, has a Garman-Klass of row k + 1's variance times
    e^(Offset + Sigma_U u), its close a walk of its own."""
    rng = np.random.default_rng(seed)
    ps, pl = math.exp(-th['Kappa_S'] * dt), math.exp(-th['Kappa_L'] * dt)
    ws = th['Sigma_S'] * math.sqrt(-math.expm1(-2 * th['Kappa_S'] * dt) / (2 * th['Kappa_S']))
    wl = th['Sigma_L'] * math.sqrt(-math.expm1(-2 * th['Kappa_L'] * dt) / (2 * th['Kappa_L']))
    rs, rl = th['Rho_S'], th['Rho_L']
    s = rng.normal() * th['Sigma_S'] / math.sqrt(2 * th['Kappa_S'])
    ell = th['L'] + rng.normal() * th['Sigma_L'] / math.sqrt(2 * th['Kappa_L'])
    logv, ret = np.empty(n + 1), np.empty(n)
    for k in range(n + 1):
        logv[k] = s + ell
        if k == n:
            break
        e, a, b = rng.standard_normal(3)
        es = rs * e + math.sqrt(1 - rs * rs) * a
        el = rl * e + (math.sqrt(1 - rs * rs - rl * rl) * b - rs * rl * a) / math.sqrt(1 - rs * rs)
        ret[k] = math.exp(0.5 * logv[k]) * math.sqrt(dt) * e
        s, ell = ps * s + ws * es, th['L'] + pl * (ell - th['L']) + wl * el
    px = 1500.0 * np.exp(np.cumsum(ret))
    close = 1600.0 * np.exp(np.cumsum(rng.normal(0.0, 0.012, n)))
    gk = dt * np.exp(logv[1:] + th['Offset'] + th['Sigma_U'] * rng.standard_normal(n))
    prev = np.concatenate([[close[0]], close[:-1]])
    span = np.sqrt(2.0 * (gk + GK * np.log(close / prev) ** 2))
    return px, (close * np.exp(0.5 * span), close * np.exp(-0.5 * span), close)


def test_the_spot_fit_recovers_the_law_it_filters(tmp_path):
    """A calibration document fits `LogVar2FJSpotModel` by `LogVar2FJSpotCalibration` on ten years of a fix
    and a bar beside it drawn from the documented law: every one of the nine parameters lands within
    three of its seed-to-seed sds of the truth, both leverages with their signs (seed 1: Kappa_S 29.2,
    Sigma_S 2.40, Kappa_L 1.26, Sigma_L 0.81, L -2.88, Rho_S -0.300, Rho_L +0.478, Sigma_U 0.448, Offset
    -0.126; over nine seeds every mean within 2.05 standard errors of the truth). 17 s.

    Killing mutation: both leverages flipped in `LogVar2FJCalibration.step` (the fitted rhos of
    the opposite sign: Rho_S +0.300 and Rho_L -0.478, 14.8 and 6.8 spreads off). Not killed: the bar of
    row t read at row t (within every bound), which `test_the_fit_writes_the_filter_s_state_after_its_last_row` holds."""
    n = 2520
    px, bars = _law_tape(1, n)
    fitted = _fit(tmp_path, pd.bdate_range('2006-01-02', periods=n), px, bars)
    off = {k: (fitted[k] - TRUE[k]) / SPREAD[k] for k in TRUE}
    assert all(abs(v) < 3.0 for v in off.values()), off


def test_a_commodity_with_no_drift_is_a_martingale():
    """A credit Monte Carlo of one future on a `LogVar2FJSpotModel` fix at `Mu` 0, its factor stating a
    rate and a carry curve, simulates 4,096 paths a year out on a monthly grid at a 48% vol: the mean of
    S_T / S_0 is within two standard errors of one (0.9989, se 0.0086), since the walk's compensator is
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


def _paths(block, batch, seed=1):
    """`(row market time in years, log returns [paths, rows])` of a credit Monte Carlo of one future on a
    `LogVar2FJSpotModel` fix `block`, on a calendar-daily grid over 95 days, a business day one
    `Calibration_DT_Years` and a weekend row none."""
    world = _world('LBMA_AM', chained=False)
    world['Calc']['Calculation'].update(Batch_Size=batch, Time_Grid='0d 1d(1d)', Calc_Scenarios='All',
                                        Random_Seed=seed)
    future = world['Calc']['Deals']['Deals']['Children'][0]['Children'][0]['Instrument']['.Deal']
    future['Maturity_Date'] = {'.Timestamp': '2026-04-20'}
    market = world['Calc']['MergeMarketData']['ExplicitMarketData']
    market['Model Configuration'] = {'.ModelParams': {
        'modeldefaults': {'CommodityPrice': 'LogVar2FJSpotModel'}, 'modelfilters': {}}}
    market['Price Models'] = {'LogVar2FJSpotModel.LBMA_AM': block}
    cx = rf.Context(path_transform={}, file_transform={})
    cx.load_json((json.dumps(world), 'lv_walk.json'))
    s = cx.run_job()[1]['Results']['scenarios']['CommodityPrice.LBMA_AM']
    dates = pd.DatetimeIndex(s.columns)
    days = np.diff(np.searchsorted(pd.bdate_range(dates[0], dates[-1]), dates, side='right')) * block['Calibration_DT_Years']
    return days, np.diff(np.log(s.values.astype(np.float64)), axis=1)


def _forecast(block, days, cov=True):
    """`E[r_k^2]` per row from the block's state: `delta_k E[e^(s + ell)]` at the row's start, the two OU
    factors Gaussian from `(S0, Ell0)` with the declared `State_Cov` (or none), plus the -v delta / 2
    drift's square; and `E[r_1 r_k^2]`, k >= 2, the leverage the first return loads on the state."""
    t = np.concatenate([[0.0], np.cumsum(days)])
    ks, ss, kl, sl, lv = (block[x] for x in ('Kappa_S', 'Sigma_S', 'Kappa_L', 'Sigma_L', 'L'))
    p11, p12, p22 = block['State_Cov'] if cov else (0.0, 0.0, 0.0)
    m = block['S0'] * np.exp(-ks * t) + lv + (block['Ell0'] - lv) * np.exp(-kl * t)
    var = (ss ** 2 / (2 * ks) * -np.expm1(-2 * ks * t) + sl ** 2 / (2 * kl) * -np.expm1(-2 * kl * t)
           + p11 * np.exp(-2 * ks * t) + 2 * p12 * np.exp(-(ks + kl) * t) + p22 * np.exp(-2 * kl * t))
    ev = np.exp(m + 0.5 * var)
    er2 = days * ev[:-1] + 0.25 * days ** 2 * np.exp(2 * m[:-1] + 2 * var[:-1])
    d1, v0 = days[0], math.exp(block['S0'] + block['Ell0'])
    ws1 = ss * math.sqrt(-math.expm1(-2 * ks * d1) / (2 * ks))
    wl1 = sl * math.sqrt(-math.expm1(-2 * kl * d1) / (2 * kl))
    lag = t[1:-1] - t[1]
    load = block['Rho_S'] * ws1 * np.exp(-ks * lag) + block['Rho_L'] * wl1 * np.exp(-kl * lag)
    return er2, math.sqrt(v0 * d1) * days[1:] * load * ev[1:-1] - 0.5 * v0 * d1 * er2[1:]


@pytest.mark.parametrize('rho', [(-0.15, 0.6), (-0.9, 0.0)])
def test_the_outer_walk_realises_the_variance_and_leverage_its_filter_forecasts(rho):
    """From a point state (`State_Cov` 0) the walk's summed squared log returns over 30 and 91 calendar
    rows are the OU closed form sum_k delta_k E[e^(s + ell)] within 3 standard errors, delta_k a
    row's market days, and the first return times the later squared returns, the leverage the
    return's innovation loads on the next state by `Rho_S` and `Rho_L`, is its closed form within 3
    standard errors (z 0.09 to 0.34 and 0.17 to 0.47 measured at 32,768 paths; at `Rho_S` -0.9 the
    leverage moment is 10.9 standard errors from 0). 4 s each.

    Killing mutations: the leverage's sign flipped in the walk (the -0.9 world's leverage moment 17.7
    standard errors off); the return read at the row's END state (6.7 standard errors off the -0.9
    world's 30-row variance, 11.5 off its leverage)."""
    block = dict(LV, Rho_S=rho[0], Rho_L=rho[1], State_Cov=[0.0, 0.0, 0.0])
    if rho[0] == -0.9:
        block.update(Kappa_S=40.0, Sigma_S=3.0)
    days, r = _paths(block, 32768)
    er2, cross = _forecast(block, days)
    for n in (30, 91):
        x, y = (r[:, :n] ** 2).sum(1), r[:, 0] * (r[:, 1:n] ** 2).sum(1)
        z_var = (x.mean() - er2[:n].sum()) / (x.std() / math.sqrt(len(x)))
        z_lev = (y.mean() - cross[:n - 1].sum()) / (y.std() / math.sqrt(len(y)))
        assert abs(z_var) < 3.0 and abs(z_lev) < 3.0, (n, z_var, z_lev)


def test_the_outer_walk_starts_from_the_filter_state_its_block_declares():
    """The block's `S0`, `Ell0` and `State_Cov` are the filter's state for the first step: the outer
    path's own filter starts from that posterior (`prior0`), and a replayed fork draws its start from
    it, so the walk the paths realise should forecast what that filter forecasts, the 30-row summed
    squared return sum_k delta_k E[e^(s + ell)] with the state N((S0, Ell0), State_Cov), within 3
    standard errors. 4 s.

    Killing mutation: every outer path started at the point (Ell0, S0), `State_Cov` read only by its
    filter (0.970 of the filter's forecast, z -13.2 at 32,768 paths; -0.39 unmutated)."""
    days, r = _paths(dict(LV), 32768)
    er2, _ = _forecast(LV, days)
    x = (r[:, :30] ** 2).sum(1)
    z = (x.mean() - er2[:30].sum()) / (x.std() / math.sqrt(len(x)))
    assert abs(z) < 3.0, (z, x.mean() / er2[:30].sum())


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
    own state, each fork stepping one market day to the next market row (Monday from a Friday):
    across the outer paths and steps, the log of each fork's spread of its log return over that step
    regressed on half the path's s + ell has slope 1 within 0.2 and intercept log sqrt(dt) within
    0.15, dt one `Calibration_DT_Years` (1.008 and -2.791 against -2.765 measured, 864 forks of 8 draws), the
    spread of a Gaussian residual being sqrt(v dt). 6 s.

    Killing mutations: the fork reading the outer run's seed keys, so every fork starts from the block's
    `Ell0` and `S0` (slope 0, the state one value); the fork stepping one calendar row, so a Friday's
    lands on a Saturday with no market time (its dt 0, the slope NaN)."""
    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        _solved(LV)
    forks = [ast.literal_eval(r.getMessage().split(' ', 1)[1]) for r in caplog.records
             if r.getMessage().startswith('LV_FORK')]
    state = 0.5 * np.concatenate([f['state'] for f in forks])
    spread = np.log(np.concatenate([f['sd'] for f in forks]))
    slope, intercept = np.polyfit(state, spread, 1) if state.std() > 0 else (0.0, 0.0)
    assert all(f['dt'] == LV['Calibration_DT_Years'] for f in forks), sorted({f['dt'] for f in forks})
    assert len(state) >= 300 and abs(slope - 1.0) < 0.2, (len(state), slope)
    assert abs(intercept - 0.5 * math.log(LV['Calibration_DT_Years'])) < 0.15, intercept


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
    forks, each one market day to the next market row, the variance of a fork's log return is the
    posterior-mixed forecast dt exp(mean + P/2), within 4% (1.0001 measured over 72 forks of 512
    draws, each fork's second moment about its own mean), where the mean's own exp(mean) reads
    0.870 of it. 10 s.

    Killing mutation: each fork started from the posterior mean (0.870 of the mixed forecast)."""
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
    mean, var, sd, dt = (np.concatenate([np.broadcast_to(f[x], len(f['sd'])) for f in forks])
                         for x in ('mean', 'var', 'sd', 'dt'))
    seen, mixed, point = (sd ** 2).sum(), (dt * np.exp(mean + 0.5 * var)).sum(), (dt * np.exp(mean)).sum()
    assert len(sd) >= 50 and abs(seen / mixed - 1.0) < 0.04 and point / mixed < 0.9, (len(sd), seen / mixed, point / mixed)


def test_the_simulated_filter_is_as_certain_as_a_replay_of_its_own_law(tmp_path, caplog):
    """The outer path's filter steps once a market day, reading the bar its walk prints beside each
    fix and nothing on a weekend, as the calibration's archive and a replay count them: its posterior
    variance of log v at the solve's swept steps is, within 5%, a replay's of a tape drawn from the
    block's own law on business days, over four seeds (0.0720 against 0.0716, the seeds 0.0696 to 0.0733). 15 s.

    Killing mutation: the filter stepping every calendar row its calendar years, reading a bar on
    the weekend's (0.0605, 15.5% under), as the simulated filter did before the market-time clock."""
    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        _solved(LV)
    forks = [ast.literal_eval(r.getMessage().split(' ', 1)[1]) for r in caplog.records
             if r.getMessage().startswith('LV_FORK')]
    simulated, replayed = np.concatenate([f['var'] for f in forks]).mean(), []
    days = pd.bdate_range('2026-04-10', periods=200)
    calendar = pd.date_range(days[0], days[-1])
    row, market = np.searchsorted(days, calendar, side='right') - 1, calendar.isin(days)
    for seed in (1, 2, 3, 4):
        px, bars = _law_tape(seed, len(days), th=LV)
        observed = {'CommodityPrice.PLATINUM_LME': px[row], 'ObservedBasis.PLATINUM_LME.LME_CME': np.full(len(row), -30.0),
                    **_flagged(~market)}
        observed.update({'CommodityPrice.PLATINUM_LME,' + x: np.where(market, b[row], np.nan)
                         for x, b in zip(('HIGH', 'LOW', 'CLOSE'), bars)})
        rows = _replayed(tmp_path, caplog, observed)
        replayed.append((rows[:, 2] + 2.0 * rows[:, 3] + rows[:, 4])[market[:len(rows)]][60:].mean())
    assert abs(simulated / np.mean(replayed) - 1.0) < 0.05, (simulated, replayed)
