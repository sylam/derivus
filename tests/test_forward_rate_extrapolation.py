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
written out here from the deal's own algebra. Ten one-token mutants are run by exec'ing the edit
onto the module (control: 8 passed, 0 failing); each test's own docstring names what kills it, and
two are killed by one gate only - the index clamped to `max_index` instead of `max_index - 1` (the
unit blend gates alone, invisible end to end because the fixture's fixing sits in front of the
FIRST knot), and the routed kind read as presence rather than value (the no-op gate alone).

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
import torch

import derivus as rf
from derivus import riskfactors, schema, utils

# ---------------------------------------------------------------------------
# 1. the tenor blend
# ---------------------------------------------------------------------------

KNOT_TAUS = np.array([0.5, 1.0])
KNOT_VALUES = np.array([0.03, 0.04])
QUERY = np.array([0.25, 0.75, 1.25])
#: z = 0.03 + 0.02 (tau - 0.5), the line through the two knots, at the three queries
LINE = np.array([0.025, 0.035, 0.045])
#: what the clipping read gives instead: the nearest knot outside the bracket
CLIPPED = np.array([0.03, 0.035, 0.04])

QUERIES = [pytest.param(QUERY, id='numpy'), pytest.param(torch.tensor(QUERY), id='torch')]


def _blend(kind, query):
    """The read `Interpolation.read_at` makes out of `get_index` - a plain two-point blend, which
    is why an unclipped alpha is all extrapolation takes."""
    index, index_next, alpha = utils.CurveTenor(KNOT_TAUS, kind).get_index(query)
    values = torch.tensor(KNOT_VALUES) if isinstance(query, torch.Tensor) else KNOT_VALUES
    blend = values[index] * (1.0 - alpha) + values[index_next] * alpha
    return blend.numpy() if isinstance(query, torch.Tensor) else blend


@pytest.mark.parametrize('query', QUERIES)
def test_the_extrapolating_blend_is_the_line_through_the_knots(query):
    """Killed by: clipping alpha to [0, 1] (i.e. reverting the `'Extrapolate' in self.type` guard),
    or clamping the index to `max_index` rather than `max_index - 1` - the far query then blends
    the last knot with itself and reads flat."""
    assert _blend('LinearExtrapolate', query) == pytest.approx(LINE, rel=1e-15)


@pytest.mark.parametrize('query', QUERIES)
def test_the_default_linear_blend_still_clips_at_the_knots(query):
    """Killed by: the extrapolating branch firing on every kind - `Linear` must be untouched, or
    every curve in every world silently changes its out-of-bracket read."""
    assert _blend('Linear', query) == pytest.approx(CLIPPED, rel=1e-15)
    assert not np.allclose(CLIPPED, LINE), 'the queries do not leave the bracket'


def test_the_numpy_curve_read_extrapolates_the_end_segments():
    """`Factor1D.interpolate` is the static path (`current_value`), which `np.interp` clips the
    same way.

    Killed by: dropping the branch, or EITHER of its two `np.where`s - `np.interp` alone holds the
    end values flat, so each side is a kill of its own."""
    got = riskfactors.Factor1D.interpolate(QUERY, KNOT_TAUS, KNOT_VALUES, ('LinearExtrapolate',))
    assert got == pytest.approx(LINE, rel=1e-15)
    assert riskfactors.Factor1D.interpolate(
        QUERY, KNOT_TAUS, KNOT_VALUES, ('Linear',)) == pytest.approx(CLIPPED, rel=1e-15)


# ---------------------------------------------------------------------------
# 2/3. the world: one fixing in front of the first carry knot
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
        'Deals': {'Tag_Titles': '', 'Reference': 'carry', 'Deals': {'Children': [{
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
    carry runs on the 365.25 clock (`utils.DAYS_IN_YEAR`) and the repo on its ACT_365 day count -
    two clocks, both live in the pinned number."""
    return UNITS * (SPOT * math.exp(
        z * FIX_DAYS / utils.DAYS_IN_YEAR + REPO * FIX_DAYS / 365.0) - STRIKE)


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
# 4. the registry
# ---------------------------------------------------------------------------

def test_the_interpolation_menu_offers_the_carry_curve_both_methods():
    """`emit_interpolation` reads the menu off the class declarations, so this is the row a UI
    offers and the row `construct_factor` will accept.

    Killed by: dropping `interpolation_methods` from `ForwardRate` (the row disappears, and
    `test_schema_emission`'s routed-types gate goes red beside it), or offering a method
    `check_interpolation` / `factor_interp_map` does not implement."""
    assert tuple(schema.mapping['Interpolation_factor_map']['ForwardRate']) == (
        'Linear', 'LinearExtrapolate')
    assert schema.mapping['Interpolation_factor_map'] == schema.emit_interpolation(riskfactors)
