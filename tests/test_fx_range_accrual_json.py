"""`FXRangeAccrualDeal` end to end, through the JSON contract and nothing else.

A range accrual pays `Payoff` once, on its settlement, for every observation the pair fixes inside
`[Lower_Barrier, Upper_Barrier]`, so it IS a strip of cash digitals: per observation a bought cash
call at the lower bound and a sold one at the upper, each on the forward to its own date at the vol
its own strike reads, all discounted to the one settlement. Each test holds the engine to that
identity twice - against the portfolio of `FXBinaryOption`s the same document prices, delivered on
the settlement, and against the hand oracle below - and pins the parts the identity cannot see: a
fixing already known counted by its exact inclusive indicator, a past row stating no fixing refused
by name, the sold strip the mirror, and a simulated fixing the path's own.

THE ORACLES ARE EXACT: the FX surface is `test_digital_spread_json`'s, authored on COLLINEAR
moneyness nodes, so every interpolation reproduces the same line and the vol at any strike inside
the nodes is that line's; the curves are flat at 4% and 2%, so the forward and the discount are
closed form. The credit Monte Carlo gate is the one statistical statement, held to three of its
own standard errors.
"""
import io
import json
import logging
import math
import os
import sys

# reference-derivus shadow-import guard (MEMORY): pin the package under test to THIS repo.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import pytest
import torch

import derivus
from derivus import utils
from derivus.config import CustomJsonEncoder
from test_digital_spread_json import (
    BASE, EPS, FX_ATM, FX_SKEW, FX_SPOT, R_EUR, R_USD, RATES, _job, _mtm, _ndtr, _run, _surface)

LOWER, UPPER, PAYOFF = 1.20, 1.30, 1000.0
MONTHLY = [BASE + pd.DateOffset(months=k) for k in range(1, 13)]
SETTLE = MONTHLY[-1] + pd.DateOffset(days=7)
FACTORS = dict(RATES, **{'FXVol.EUR.USD': _surface(FX_ATM, FX_SKEW)})
SPREAD_ON = {'FXRangeAccrualDeal': {'Relative_Digital_Spread': EPS},
             'FXBinaryOption': {'Relative_Digital_Spread': EPS}}

RA = {'Object': 'FXRangeAccrualDeal', 'Reference': 'RA', 'Currency': 'USD',
      'Underlying_Currency': 'EUR', 'Discount_Rate': 'USD', 'FX_Volatility': 'EUR.USD',
      'Buy_Sell': 'Buy', 'Lower_Barrier': LOWER, 'Upper_Barrier': UPPER, 'Payoff': PAYOFF,
      'Settlement_Date': SETTLE, 'Observation_Dates': [[date, None] for date in MONTHLY]}


def _binaries(dates):
    """The strip as the portfolio it is: per date a bought cash call at the lower bound and a sold
    one at the upper, delivered on the range's settlement so both discount to the same day."""
    return [{'Object': 'FXBinaryOption', 'Reference': '{}{}'.format(side, j), 'Currency': 'USD',
             'Underlying_Currency': 'EUR', 'Discount_Rate': 'USD', 'FX_Volatility': 'EUR.USD',
             'Buy_Sell': buy_sell, 'Option_Type': 'Call', 'Strike_Price': strike,
             'Payoff': PAYOFF, 'Expiry_Date': date, 'Delivery_Date': SETTLE}
            for j, date in enumerate(dates)
            for side, buy_sell, strike in (('L', 'Buy', LOWER), ('U', 'Sell', UPPER))]


def _sigma(fwd, strike):
    return FX_ATM + FX_SKEW * (fwd / strike - 1.0)


def _call(fwd, strike, t):
    """Undiscounted Black call at the line's vol at `strike`."""
    sd = _sigma(fwd, strike) * math.sqrt(t)
    d1 = math.log(fwd / strike) / sd + 0.5 * sd
    return fwd * _ndtr(d1) - strike * _ndtr(d1 - sd)


def _in_range(date, base=BASE, spread=False):
    """The probability the fixing on `date` lands inside the range, seen from `base`: the cash
    digital at the lower bound less the one at the upper, each at its own strike's vol on the
    forward to the date itself, or under `spread` each the call spread at its strike."""
    t = (date - base).days / 365.0
    fwd = FX_SPOT * math.exp((R_USD - R_EUR) * t)

    def digital(k):
        if spread:
            return (_call(fwd, k * (1.0 - EPS), t) - _call(fwd, k * (1.0 + EPS), t)) / (2.0 * EPS * k)
        sd = _sigma(fwd, k) * math.sqrt(t)
        return _ndtr(math.log(fwd / k) / sd - 0.5 * sd)
    return digital(LOWER) - digital(UPPER)


def _df(base=BASE):
    return math.exp(-R_USD * (SETTLE - base).days / 365.0)


def _organs(log):
    return [ln for ln in log.splitlines() if 'RANGE_ACCRUAL' in ln]


def _portfolio(out, dates):
    return sum(_mtm(out, '{}{}'.format(side, j)) for j in range(len(dates)) for side in 'LU')


def test_a_range_accrual_is_its_two_digitals_per_observation_settled_together():
    """Twelve monthly observations, the range 1.20 to 1.30 about a 1.25 spot, settled a week after
    the last: the range, the 24 binaries it is made of and the hand oracle
    `Payoff * sum_j (N(d2_j(L)) - N(d2_j(U))) * DF(settle)` agree within 1e-9, each `d2` at the
    line's vol at its own strike on the forward to the observation date.

    MEASURED 5102.684836043783, the range and the binaries each 1.8e-16 off the oracle.

    Killing mutations: each observation discounted to its fixing rather than the settlement,
    5221.71 (+2.3%); the upper leg read at the lower's vol, 4743.00 (-7.0%); the upper leg added
    instead of subtracted, 11884.11.
    """
    out, log = _run(_job([RA] + _binaries(MONTHLY), FACTORS), debug=True)
    assert _organs(log) == ['RANGE_ACCRUAL ref=RA rows=1 known=0 simulated=0 future=12 eps=0'], \
        _organs(log)
    expected = PAYOFF * _df() * sum(_in_range(date) for date in MONTHLY)
    portfolio = _portfolio(out, MONTHLY)
    for value in (_mtm(out, 'RA'), portfolio):
        assert abs(value - expected) / expected < 1e-9, (_mtm(out, 'RA'), portfolio, expected)


def test_the_digital_spread_reads_each_strike_s_own_smile():
    """Under `Relative_Digital_Spread` each of the strip's digitals is the call spread at its own
    strike, each leg at its own strike's vol: the range against the 24 binaries under the same
    option and the hand oracle, within 1e-9, and moved off the single-vol closed form by the skew.

    MEASURED 5110.308073590088, 6.8e-15 off the oracle and the binaries 6.9e-15, the closed form
    5102.68 (+0.15%).

    Killing mutation: both legs of each spread read at the centre strike's vol, 5078.68 (-0.62%).
    """
    out, log = _run(_job([RA] + _binaries(MONTHLY), FACTORS, valuation=SPREAD_ON), debug=True)
    assert _organs(log) == ['RANGE_ACCRUAL ref=RA rows=1 known=0 simulated=0 future=12 eps=0.01'], \
        _organs(log)
    expected = PAYOFF * _df() * sum(_in_range(date, spread=True) for date in MONTHLY)
    portfolio = _portfolio(out, MONTHLY)
    for value in (_mtm(out, 'RA'), portfolio):
        assert abs(value - expected) / expected < 1e-9, (_mtm(out, 'RA'), portfolio, expected)
    closed = PAYOFF * _df() * sum(_in_range(date) for date in MONTHLY)
    assert abs(_mtm(out, 'RA') - closed) / closed > 1e-4, (_mtm(out, 'RA'), closed)


#: Six fixed months: four in the range, two of them ON its bounds, and two out of it.
FIXED = [1.20, 1.25, 1.27, 1.30, 1.19, 1.31]
PAST = [BASE - pd.DateOffset(months=k) for k in range(6, 0, -1)]
FUTURE = [BASE + pd.DateOffset(months=k) for k in range(1, 7)]


def test_known_fixings_count_exactly_and_a_blank_one_refuses_by_name():
    """The base date mid-strip: six months fixed before it, four inside the range with one on
    each bound, and six to come - the value is `(4 + the six digitals) * Payoff * DF`. A past row
    stating no fixing refuses by name, naming the Reference, the table and the day. With every
    observation behind the base date and the settlement ahead the value is the count times the
    payoff discounted, exactly; with the settlement behind it too the compile skips the deal by
    name as expired and it writes no row.

    MEASURED 7250.700315011354, 1.3e-16 off the oracle; the fully fixed strip 3996.932683504693,
    the oracle to the bit.

    Killing mutations: the bounds read exclusive, 5291.91, the two fixings on the bounds out; the
    refusal deleted, the blank read as 0.0 and priced, no refusal raised.
    """
    settle = FUTURE[-1] + pd.DateOffset(days=7)
    rows = [[date, value] for date, value in zip(PAST, FIXED)] + [[date, None] for date in FUTURE]
    out, log = _run(_job([dict(RA, Observation_Dates=rows, Settlement_Date=settle)], FACTORS),
                    debug=True)
    assert _organs(log) == ['RANGE_ACCRUAL ref=RA rows=1 known=6 simulated=0 future=6 eps=0'], \
        _organs(log)
    df = math.exp(-R_USD * (settle - BASE).days / 365.0)
    expected = PAYOFF * df * (4 + sum(_in_range(date) for date in FUTURE))
    assert abs(_mtm(out, 'RA') - expected) / expected < 1e-12, (_mtm(out, 'RA'), expected)

    blank = [list(row) for row in rows]
    blank[2][1] = None
    with pytest.raises(utils.UnpriceableSchedule,
                       match='RA: the Observation_Dates row {:%Y-%m-%d} is behind the base date'.format(
                           PAST[2])):
        _run(_job([dict(RA, Observation_Dates=blank, Settlement_Date=settle)], FACTORS))

    fixed = [[date, value] for date, value in zip(PAST, FIXED)]
    week = BASE + pd.DateOffset(days=7)
    out, log = _run(_job([dict(RA, Observation_Dates=fixed, Settlement_Date=week)], FACTORS),
                    debug=True)
    assert _organs(log) == ['RANGE_ACCRUAL ref=RA rows=1 known=6 simulated=0 future=0 eps=0'], \
        _organs(log)
    assert _mtm(out, 'RA') == 4 * PAYOFF * math.exp(-R_USD * 7 / 365.0), _mtm(out, 'RA')

    out, log = _run(_job([dict(RA, Observation_Dates=fixed, Settlement_Date=BASE - pd.DateOffset(days=7))],
                         FACTORS), debug=True)
    assert 'skipping expired deal RA' in log and not _organs(log), log
    assert 'RA' not in set(out['Results']['mtm']['Reference'])


def test_a_sold_strip_is_minus_a_bought_one_and_an_empty_range_refuses():
    """`Sell` is `-1` times `Buy` to the bit. A range whose lower bound is not below its upper and a
    non-positive bound are each skipped by name: the log says why and the deal writes no row,
    beside a bought strip that prices.

    Killing mutations: `Buy_Sell` read as `Buy` whatever it states, the sold strip +5102.68; the
    empty range's refusal deleted, a row written for it and no skip logged; the positive bound's
    refusal deleted, the same.
    """
    sold = dict(RA, Reference='SOLD', Buy_Sell='Sell')
    out, _ = _run(_job([RA, sold], FACTORS))
    assert _mtm(out, 'SOLD') == -_mtm(out, 'RA') and _mtm(out, 'RA') > 0.0, (
        _mtm(out, 'SOLD'), _mtm(out, 'RA'))

    for terms, said in (({'Lower_Barrier': 1.30, 'Upper_Barrier': 1.30},
                         'EMPTY: Lower_Barrier 1.3 is not below Upper_Barrier 1.3'),
                        ({'Lower_Barrier': 1.30, 'Upper_Barrier': 1.20},
                         'EMPTY: Lower_Barrier 1.3 is not below Upper_Barrier 1.2'),
                        ({'Lower_Barrier': 0.0},
                         'EMPTY: Lower_Barrier and Upper_Barrier must be positive (got 0.0 and 1.3)')):
        out, log = _run(_job([RA, dict(RA, Reference='EMPTY', **terms)], FACTORS), debug=True)
        assert said in log and 'Skipped' in log, (terms, log[-2000:])
        assert 'EMPTY' not in set(out['Results']['mtm']['Reference']) and _mtm(out, 'RA') > 0.0


# --------------------------------------------------------------------------------------------
# a simulated fixing: the path's own spot, under a credit Monte Carlo on a GBM pair
# --------------------------------------------------------------------------------------------
FOUR_WEEKLY = [BASE + pd.DateOffset(days=28 * k) for k in range(1, 13)]
SETTLE_CMC = FOUR_WEEKLY[-1] + pd.DateOffset(days=7)
#: Half way between the sixth observation and the seventh, a row of the fortnightly grid.
ROW = BASE + pd.DateOffset(days=182)
FLAT = dict(RATES, **{'FXVol.EUR.USD': _surface(FX_ATM, 0.0)})
STRIP = dict(RA, Settlement_Date=SETTLE_CMC, Observation_Dates=[[date, None] for date in FOUR_WEEKLY])


def _flat_in_range(date):
    t = (date - BASE).days / 365.0
    fwd = FX_SPOT * math.exp((R_USD - R_EUR) * t)
    sd = FX_ATM * math.sqrt(t)
    return sum(sign * _ndtr(math.log(fwd / k) / sd - 0.5 * sd) for sign, k in ((1, LOWER), (-1, UPPER)))


def test_a_simulated_fixing_is_the_path_s_own_indicator():
    """A credit Monte Carlo on a GBM pair at the flat surface's vol, drifting at the carry, 8,192
    paths on a fortnightly grid that holds every four-weekly observation: the base-date row of the
    exposure is the base valuation within 1e-9, and on the row half way between the sixth
    observation and the seventh the mean exposure is, by iterated expectation, every observation's
    base-date probability times the payoff, discounted from the settlement to that row - the six
    fixed ones counted off each path's own spot, the six to come its digitals. Held to three of the
    exposure's own standard errors on the row, the paths' indicators being correlated so no
    binomial variance is theirs.

    MEASURED the base-date row 0.0 off the base valuation's 5542.78; the mean 5622.34 against
    5654.44, 1.04 standard errors of 30.98 below.

    Killing mutation: the simulated fixing read off the base date's spot carried to it, not the
    path, so every path counts the six fixings in range - the mean 8029.21, 2,375 above.
    """
    base, _ = _run(_job([STRIP], FLAT))
    doc = _job([STRIP], FLAT, Object='CreditMonteCarlo', Time_grid='0d 2w(2w)', Batch_Size=4096,
               Simulation_Batches=2, Random_Seed=1, Deflation_Interest_Rate='USD')
    market = doc['Calc']['MergeMarketData']['ExplicitMarketData']
    market['Price Models'] = {'GBMAssetPriceModel.EUR': {'Vol': FX_ATM, 'Drift': R_USD - R_EUR}}
    market['Model Configuration'] = {'.ModelParams': {
        'modeldefaults': {'FxRate': 'GBMAssetPriceModel'}, 'modelfilters': {}}}
    buf, root = io.StringIO(), logging.getLogger()
    handler, old = logging.StreamHandler(buf), root.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    try:
        cx = derivus.Context()
        cx.load_json((json.dumps(doc, cls=CustomJsonEncoder), 'range_cmc'))
        mtm = derivus.run_cmc(cx.current_cfg, prec=torch.float64)[1]['Results']['mtm']
    finally:
        root.removeHandler(handler)
        root.setLevel(old)
    assert _organs(buf.getvalue()) == [
        'RANGE_ACCRUAL ref=RA rows=26 known=0 simulated=12 future=12 eps=0'] * 2, _organs(buf.getvalue())

    first = mtm.iloc[0].values
    assert mtm.index[0] == BASE and mtm.shape[1] == 8192, mtm.shape
    assert np.abs(first - _mtm(base, 'RA')).max() / _mtm(base, 'RA') < 1e-9, (first[:3], _mtm(base, 'RA'))

    row = mtm.loc[ROW].values
    expected = PAYOFF * math.exp(-R_USD * (SETTLE_CMC - ROW).days / 365.0) * sum(
        _flat_in_range(date) for date in FOUR_WEEKLY)
    error = row.std(ddof=1) / math.sqrt(row.size)
    assert abs(row.mean() - expected) < 3.0 * error, (row.mean(), expected, error)
