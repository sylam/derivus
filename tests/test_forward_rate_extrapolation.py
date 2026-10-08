"""`LinearExtrapolate` on `ForwardRate` - the carry curve read in FRONT of its first knot.

A carry curve linear in tenor (`z = c + a tau`) is identified exactly by two knots and the line
continues outside them, but `CurveTenor.get_index` CLIPS a query to the knot bracket, so the
default `Linear` read of a contract trading in front of the first knot is FLAT in z and the
log-futures curve's curvature term is silently lost. `ForwardRate` declares
`interpolation_methods` and `construct_factor` routes it through `Price Factor Interpolation`, so a
world can ask for the line instead. `LinearExtrapolate` clamps the index to a real end segment and
leaves alpha UNCLIPPED - a two-point blend with alpha outside [0, 1] IS the line - while `Linear`
keeps the clipped read.

KILL MAGNITUDE: routed and unrouted marks differ by 1.75% of the mark on the end-to-end deal, both
written out here from the deal's own algebra; each test's own docstring names what kills it. The
index clamped to `max_index` instead of `max_index - 1` is invisible end to end, the fixture's
fixing sitting in front of the FIRST knot.

ANTI-PLACEBO, and the fixture does not reproduce without it: the knots are SLOPED (0.02 -> 0.01),
because on a flat carry both reads coincide and every gate here passes on a broken clamp; the
fixing is in front of the first knot, the side the clipped read cannot see past; the discount curve
is zero so `D = 1` and the pinned number is the forward alone; and the repo leg is NOT zero (0.02),
so a carry read that had picked up the repo curve instead would miss both hand values.
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

import derivus as rf
from derivus import utils

# ---------------------------------------------------------------------------
# 1. the world: one fixing in front of the first carry knot
# ---------------------------------------------------------------------------

BASE = pd.Timestamp('2026-01-15')
EXCEL0 = float((BASE - utils.excel_offset).days)
#: dated knots at tau ~0.5 and ~1.0, SLOPED - see the anti-placebo note
KNOTS = (EXCEL0 + 183.0, EXCEL0 + 366.0)
Z = (0.02, 0.01)
FIX_DAYS = 91.0                                  # tau ~0.25, in front of the first knot
FIX_DATE = (BASE + pd.Timedelta(days=int(FIX_DAYS))).strftime('%Y-%m-%d')
SPOT, REPO, UNITS, STRIKE = 1600.0, 0.02, 1000.0, 1500.0
#: the carry the line gives at the fixing, and the flat-clip's nearest knot
Z_LINE = Z[0] + (EXCEL0 + FIX_DAYS - KNOTS[0]) / (KNOTS[1] - KNOTS[0]) * (Z[1] - Z[0])


def _curve(data):
    return {'.Curve': {'meta': [], 'data': data}}


def _job(interpolation=None):
    """A single-fixing average-price swap on a plain (uncomposed) spot, discounted on a ZERO curve:
    `V = N (S e^{z tau + r tau} - K)` and nothing else. `interpolation` is the whole variable - it
    is the one entry `Price Factor Interpolation` carries."""
    market = {
        'System Parameters': {'Base_Currency': 'USD',
                              'Base_Date': {'.Timestamp': BASE.strftime('%Y-%m-%d')}},
        'Price Factors': {
            'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD-ZERO', 'Spot': 1.0},
            'InterestRate.USD-ZERO': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                                      'Curve': _curve([[0.0, 0.0], [5.0, 0.0]])},
            'InterestRate.USD-REPO': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                                      'Curve': _curve([[0.0, REPO], [5.0, REPO]])},
            'CommodityPrice.PLATINUM_CME': {'Spot': SPOT, 'Currency': 'USD',
                                            'Interest_Rate': 'USD-REPO',
                                            'Forward_Rate': 'PLATINUM_CARRY'},
            'ForwardRate.PLATINUM_CARRY': {
                'Currency': 'USD', 'Curve': _curve([[KNOTS[0], Z[0]], [KNOTS[1], Z[1]]])}
        }}
    if interpolation is not None:
        market['Price Factor Interpolation'] = {'.ModelParams': {
            'modeldefaults': {'ForwardRate': interpolation}, 'modelfilters': {}}}
    deal = {'Object': 'CommodityAveragePriceSwapDeal', 'Reference': 'APS1', 'Currency': 'USD',
            'Commodity': 'PLATINUM_CME', 'Carry': 'PLATINUM_CARRY', 'Discount_Rate': 'USD-ZERO',
            'Buy_Sell': 'Buy', 'Units': UNITS, 'Fixed_Price': STRIKE,
            'Settlement_Date': {'.Timestamp': FIX_DATE},
            'Sampling_Data': [[{'.Timestamp': FIX_DATE}, 0.0, 1.0]]}
    return {'Calc': {
        'Calculation': {'Object': 'BaseValuation', 'Currency': 'USD', 'Greeks': 'No',
                        'Base_Date': {'.Timestamp': BASE.strftime('%Y-%m-%d')}},
        'Deals': {'Reference': 'carry', 'Deals': {'Children': [{
            'Instrument': {'.Deal': {'Object': 'NettingCollateralSet', 'Reference': 'NS',
                                     'Netted': 'True', 'Collateralized': 'False'}},
            'Children': [{'Instrument': {'.Deal': deal}}]}]}},
        'MergeMarketData': {'ExplicitMarketData': market}}}


def _value(tmp_path, interpolation=None, tag='job'):
    path = str(tmp_path / f'carry_{tag}.json')
    open(path, 'w').write(json.dumps(_job(interpolation)))
    cx = rf.Context(path_transform={}, file_transform={})
    cx.load_json(path)
    _, out = cx.run_job()
    return out['Results']['mtm'].set_index('Reference').loc['APS1', 'Value']


def _hand(z):
    """The deal's own algebra written out: one fixing, weight one, `D = 1` on the zero curve. The
    carry runs on the 365.25 clock (`utils.DayCount.DAYS_IN_YEAR`) and the repo on its ACT_365 day count -
    two clocks, both live in the pinned number."""
    return UNITS * (SPOT * math.exp(
        z * FIX_DAYS / utils.DayCount.DAYS_IN_YEAR + REPO * FIX_DAYS / 365.0) - STRIKE)


def test_the_routed_carry_reads_the_line_and_the_unrouted_one_clips(tmp_path):
    """The feature, end to end, against two hand-computed marks.

    Killed by: reverting the `CurveTenor.get_index` guard (both runs then return the clipped
    116030.48 and the two marks are equal), dropping `'ForwardRate'` from `construct_factor`'s
    routed types or from `update_tenors` (the declared kind never reaches the `CurveTenor`, same
    equality), or `factor_interp_map` losing the identity row (the method maps to `Linear`)."""
    assert 0.0 < EXCEL0 + FIX_DAYS < KNOTS[0] and Z[0] != Z[1], 'the fixture cannot see the fix'
    clipped = _value(tmp_path, tag='clipped')
    routed = _value(tmp_path, 'LinearExtrapolate', tag='routed')

    assert clipped == pytest.approx(_hand(Z[0]), rel=1e-13)          # 116030.47634940389
    assert routed == pytest.approx(_hand(Z_LINE), rel=1e-13)         # 118055.87009144256
    assert routed / clipped - 1.0 == pytest.approx(0.01745, abs=1e-5), (
        f'the two reads differ by {routed - clipped:.2f} - not a measurable feature')


def test_routing_the_carry_to_the_default_is_a_no_op(tmp_path):
    """Default behaviour is unchanged unless a world asks: the same world with no `Price Factor
    Interpolation` section at all and with `{'ForwardRate': 'Linear'}` in it must agree to the BIT.

    Killed by: reading the routed entry as PRESENCE rather than as a value - `construct_factor`
    injecting `'LinearExtrapolate' if interp_method else 'Linear'` splits this pair by the 2025.39
    of the gate above and passes every other gate in this file."""
    assert _value(tmp_path, tag='absent') == _value(tmp_path, 'Linear', tag='linear')


# ---------------------------------------------------------------------------
# 2. the Hermite read, end to end
# ---------------------------------------------------------------------------

HERMITE_KNOTS = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
HERMITE_RATES = np.array([0.031, 0.034, 0.036, 0.033, 0.035, 0.038, 0.037, 0.039])
#: cashflow days between the knots, none on one
HERMITE_DAYS = (100, 140, 290, 555, 900, 1500, 2300, 3100)


def _hermite_by_hand(kind, t):
    """Rate x tenor off the cubic Hermite through the knots - on the rate, or on rate x tenor for
    HermiteRT - each knot's slope that of the parabola through it and its neighbours (an end knot:
    the first or last three), in the standard basis."""
    y = HERMITE_RATES * HERMITE_KNOTS if kind == 'HermiteRT' else HERMITE_RATES
    first = np.clip(np.arange(y.size) - 1, 0, y.size - 3)
    slope = np.array([np.polyval(np.polyder(np.polyfit(HERMITE_KNOTS[j:j + 3], y[j:j + 3], 2)), x)
                      for j, x in zip(first, HERMITE_KNOTS)])
    i = np.searchsorted(HERMITE_KNOTS, t, side='right') - 1
    h = HERMITE_KNOTS[i + 1] - HERMITE_KNOTS[i]
    m = (t - HERMITE_KNOTS[i]) / h
    cubic = ((2 * m ** 3 - 3 * m ** 2 + 1) * y[i] + (m ** 3 - 2 * m ** 2 + m) * h * slope[i]
             + (3 * m ** 2 - 2 * m ** 3) * y[i + 1] + (m ** 3 - m ** 2) * h * slope[i + 1])
    return cubic if kind == 'HermiteRT' else cubic * t


def _hermite_job(kind):
    """Fixed cashflows of a million on a USD curve read under `kind`, one per `HERMITE_DAYS`."""
    deals = [{'Instrument': {'.Deal': {
        'Object': 'FixedCashflowDeal', 'Reference': 'CF%d' % days, 'Currency': 'USD',
        'Discount_Rate': 'USD-H', 'Amount': 1e6,
        'Payment_Date': {'.Timestamp': (BASE + pd.Timedelta(days=days)).strftime('%Y-%m-%d')}}}}
        for days in HERMITE_DAYS]
    market = {
        'System Parameters': {'Base_Currency': 'USD',
                              'Base_Date': {'.Timestamp': BASE.strftime('%Y-%m-%d')}},
        'Price Factors': {
            'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD-H', 'Spot': 1.0},
            'InterestRate.USD-H': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                                   'Curve': _curve(np.stack([HERMITE_KNOTS, HERMITE_RATES], 1).tolist())}},
        'Price Factor Interpolation': {'.ModelParams': {
            'modeldefaults': {'InterestRate': kind}, 'modelfilters': {}}}}
    return {'Calc': {
        'Calculation': {'Object': 'BaseValuation', 'Currency': 'USD', 'Greeks': 'No',
                        'Base_Date': {'.Timestamp': BASE.strftime('%Y-%m-%d')}},
        'Deals': {'Reference': 'hermite', 'Deals': {'Children': [{
            'Instrument': {'.Deal': {'Object': 'NettingCollateralSet', 'Reference': 'NS',
                                     'Netted': 'True', 'Collateralized': 'False'}},
            'Children': deals}]}},
        'MergeMarketData': {'ExplicitMarketData': market}}}


@pytest.mark.parametrize('kind', ['Hermite', 'HermiteRT'])
def test_a_hermite_curve_document_prices_the_cubic_by_hand(tmp_path, kind):
    """A base valuation of cashflows between the knots of a Hermite curve against the cubic through
    them written out by hand: every mark to 1e-13.

    Killed by: the blend's middle term with `t` for `1 - t`, 1.4e-5 of a mark off.

    Killing mutation: the derived Hermite slope at a knot divided by the far span rather than the near one."""
    path = str(tmp_path / 'hermite.json')
    open(path, 'w').write(json.dumps(_hermite_job(kind)))
    cx = rf.Context(path_transform={}, file_transform={})
    cx.load_json(path)
    marks = cx.run_job()[1]['Results']['mtm'].set_index('Reference')['Value']
    for days in HERMITE_DAYS:
        assert marks['CF%d' % days] == pytest.approx(
            1e6 * math.exp(-_hermite_by_hand(kind, days / 365.0)), rel=1e-13), days
