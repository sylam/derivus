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


# ---------------------------------------------------------------------------
# 5. the pricers' read
# ---------------------------------------------------------------------------

READ_KNOTS = np.array([0.25, 1.0, 2.0, 5.0])
READ_RATES = np.array([0.02, 0.025, 0.03, 0.028])
#: two deals' days to their cashflows, one shape
READ_DAYS = (np.array([[30.0, 200.0, 400.0, 1500.0]]), np.array([[45.0, 300.0, 700.0, 1200.0]]))


def _daycount(name):
    code = utils.DayCount.code(name)
    return lambda days: utils.DayCount.accrual(BASE, days, code)


def _read(tenor, daycount, days, rates):
    """`CurveTensor.interpolate_curve` on a static Linear curve: rate x tenor at `days`."""
    values = torch.as_tensor(rates, dtype=torch.float64).reshape(1, -1, 1)
    curve = utils.CurveTensor(utils.Interpolation.build(values, 'Linear', READ_KNOTS),
                              np.zeros(1, dtype=np.int64), None)
    return curve.interpolate_curve((False, 'curve', None, tenor, daycount), days, 1).reshape(-1)


def test_a_static_read_is_indexed_once_and_reads_its_own_day_count_and_points():
    """`CurveTenor.read_index` keeps a read's year fractions and index across batches, keyed on the
    points, the day count, the dtype and the device; the curve's values are read on every batch.
    Two curves on one tenor axis under ACT_365 and ACT_360, two deals of one shape, by hand.

    Killed by: the key dropped to the points' shape (the second deal reads the first's tenors, 20%
    to 50% off), the day count dropped from it (ACT_360 reads ACT_365's, 1.4%), or the dtype."""
    tenor = utils.CurveTenor(READ_KNOTS)
    for name, per_year in (('ACT_365', 365.0), ('ACT_360', 360.0)):
        daycount = _daycount(name)
        for days in READ_DAYS:
            t = days.reshape(-1) / per_year
            for rates in (READ_RATES, 2.0 * READ_RATES):
                assert _read(tenor, daycount, days, rates).numpy() == pytest.approx(
                    np.interp(t, READ_KNOTS, rates) * t, rel=1e-14), (name, days, rates)
    # a later batch reads the same index, which carries no graph, in its own dtype
    like, act365 = torch.zeros(1, dtype=torch.float64), _daycount('ACT_365')
    first = tenor.read_index(act365, READ_DAYS[0], like)
    assert tenor.read_index(act365, READ_DAYS[0].copy(), like) is first
    assert not any(x.requires_grad for x in first)
    assert tenor.read_index(act365, READ_DAYS[0], like.float())[0].dtype == torch.float32


def test_a_static_index_is_copied_to_the_device_once_and_reads_its_own_points():
    """The vol surface's expiry read and a deal's mtm gather keep their index and weight as the
    surface's or the mtm's own tensors across batches, keyed on the points, dtype and device: the
    same read twice is one set of tensors, and two expiry sets of one shape each read their own -
    the expiry blend against `torch.lerp` by hand, the gather against its numpy index, both devices.

    Killed by: the key dropped to the points' shape - the second set reads the first's weights."""
    from types import SimpleNamespace
    tenor = utils.CurveTenor(READ_KNOTS)
    for device in ['cpu'] + ['cuda'] * bool(torch.cuda.device_count()):
        surface = torch.linspace(0.1, 0.3, READ_KNOTS.size, dtype=torch.float64, device=device).reshape(-1, 1)
        for expiry in (np.array([0.3, 1.7]), np.array([2.2, 4.4])):
            index, index_next, alpha = tenor.get_index(expiry)
            want = torch.lerp(surface[index], surface[index_next], surface.new(alpha).reshape(-1, 1))
            got = utils.VolSurface._interp(surface, (None, 'v', None, None, tenor), expiry,
                                           SimpleNamespace(t_Buffer={}), False)
            assert torch.equal(got, want), (device, expiry)
            assert tenor.device_index(expiry.copy(), surface) is tenor.device_index(expiry, surface)
        timing = utils.DealTimeDependencies(np.array([0, 30, 90, 180, 365]), np.array([0, 2, 4]))
        mtm = torch.arange(15.0, dtype=torch.float64, device=device).reshape(5, 3)
        rows = utils.gather_interp_matrix(mtm, timing)
        assert torch.equal(rows, torch.lerp(mtm[timing.index], mtm[timing.index_next], mtm.new(timing.alpha)))
        assert utils.gather_interp_matrix(mtm, timing) is not rows and len(timing.t_read) == 1


def _nodes(out):
    seen, stack = set(), [out.grad_fn]
    while stack:
        node = stack.pop()
        if node is not None and node not in seen:
            seen.add(node)
            stack.extend(f for f, _ in node.next_functions)
    return {type(node).__name__ for node in seen}


@pytest.mark.parametrize('kind', ('Linear', 'LinearRT', 'Hermite'))
def test_a_time_blended_read_is_the_out_of_place_read_to_the_bit(kind):
    """A read blended across scenario rows gathers its rows with `index_select` on the host and the
    advanced index on the card, blends into the first fresh gather and hands back a read nothing
    scales as read: the out-of-place read written out here, its value and its first and second
    derivatives, to the bit, scaled by the tenor and not, on both devices.

    Killed by: the host gathering with the advanced index; an unscaled read multiplied by one."""
    days = np.array([[30.0, 200.0, 400.0, 1500.0, 2200.0]] * 4)
    daycount = _daycount('ACT_365')
    index, alpha = np.array([0, 1, 2, 3]), np.array([0.0, 0.3, 0.7, 1.0]).reshape(-1, 1, 1)
    gen = torch.Generator().manual_seed(5)
    for device in ['cpu'] + ['cuda'] * bool(torch.cuda.device_count()):
        tenor = utils.CurveTenor(READ_KNOTS, kind)
        rates = (0.02 + 0.01 * torch.rand(5, READ_KNOTS.size, 6, generator=gen, dtype=torch.float64)).to(device)

        def engine(leaf, scaled):
            curve = utils.CurveTensor(utils.Interpolation.build(leaf, kind, READ_KNOTS), index, alpha)
            return curve.interpolate_curve((True, 'curve', None, tenor, daycount), days, scaled)

        def written_out(leaf, scaled):
            interp = utils.Interpolation.build(leaf, kind, READ_KNOTS)
            flat, stride = interp.indexed_tensor, interp.shape[1]
            years, i1, i2, w = tenor.read_index(daycount, days, flat)
            rows = torch.as_tensor(index, device=device)

            def at(row):
                i0, i1x = row.reshape(-1, 1) * stride + i1, row.reshape(-1, 1) * stride + i2
                if kind == 'Hermite':
                    g, c = interp.interp_params
                    return utils.Interpolation.calc_hermite_curve(
                        w.unsqueeze(-1), g[i0, ], c[i0, ], flat[i0, ], flat[i1x, ])
                return torch.lerp(flat[i0, ], flat[i1x, ], w.unsqueeze(-1))
            raw = torch.lerp(at(rows), at((rows + 1).clamp(max=4)), flat.new(alpha))
            mult = years.unsqueeze(-1) if scaled else 1.0
            if kind.endswith('RT'):
                mult = mult / years.unsqueeze(-1).clamp(tenor.min, tenor.max)
            return raw * mult

        for scaled in (0, 1):
            leaf = rates.clone().requires_grad_()
            got, want = engine(leaf, scaled), written_out(leaf, scaled)
            assert torch.equal(got, want), (device, scaled)
            nodes = _nodes(got)
            assert ('IndexSelectBackward0' if device == 'cpu' else 'IndexBackward0') in nodes, nodes
            assert scaled or kind != 'Linear' or 'MulBackward0' not in nodes, nodes
            v = torch.linspace(-1.0, 1.0, got.numel(), dtype=torch.float64, device=device).reshape(got.shape)
            firsts = [torch.autograd.grad((x * x * v).sum(), leaf, create_graph=True)[0] for x in (got, want)]
            assert torch.equal(*firsts), (device, scaled)
            seconds = [torch.autograd.grad((g * g).sum(), leaf)[0] for g in firsts]
            assert torch.equal(*seconds), (device, scaled)


def test_a_flat_surface_reads_flat_at_every_weight():
    """Every torch two-point blend is `torch.lerp`, which reads two equal ends as that value at any
    weight: a flat grid read by `interpolate_tensor`, a deal's MtM gathered between scenario dates,
    a vol surface blended in expiry, in vol tenor and in moneyness - 4,096 weights each, in both
    precisions - where `(1 - w) a + w b` is off its value at up to 37% of them. Each read's first
    and second derivatives pass `gradcheck` on a surface that is not flat.

    Killed by: any of these blends spelled `a * (1 - w) + b * w`."""
    from types import SimpleNamespace
    gen = torch.Generator().manual_seed(7)
    grid = np.array([0.0, 30.0, 400.0, 1500.0])
    queries = np.sort(np.concatenate([grid, 1500.0 * torch.rand(4096, generator=gen, dtype=torch.float64).numpy()]))
    segment = grid.searchsorted(queries, side='right').clip(1, 3) - 1
    reads = {
        'interpolate_tensor': lambda s, q: utils.interpolate_tensor(queries[q], grid, s),
        'gather_interp_matrix': lambda s, q: utils.gather_interp_matrix(s, SimpleNamespace(
            index=segment[q], index_next=segment[q] + 1, t_read={}, alpha=(
                (queries[q] - grid[segment[q]]) / np.diff(grid)[segment[q]]).reshape(-1, 1))),
        'expiry': lambda s, q: utils.VolSurface._interp(
            s, (None, 'flat', None, None, utils.CurveTenor(grid)), queries[q],
            SimpleNamespace(t_Buffer={}), False),
        'tenor': lambda s, q: torch.stack([utils.VolSurface._tenor_blend(
            [(False, 'vol', None, None, None, utils.CurveTenor(grid))], tenor,
            SimpleNamespace(t_Static_Buffer={'vol': s})) for tenor in queries[q][:256]]),
        'moneyness': lambda s, q: utils.VolSurface._moneyness(
            s.new(queries[q] / 1500.0).reshape(1, -1), np.zeros(1), 'flat',
            SimpleNamespace(t_Buffer={'flat': (s.reshape(-1), None, utils.CurveTenor(grid / 1500.0))},
                            one=s.new_ones(1)))}
    shapes = {'interpolate_tensor': grid.shape, 'moneyness': grid.shape}
    every, few = slice(None), slice(0, None, 512)

    def svi(strike, q, dtype):
        """One SVI smile at every expiry, read under a sticky strike: its ATM reference and its
        variance blended between expiries."""
        smile = {k: torch.full((grid.size, 1), v, dtype=dtype) for k, v in (
            ('a', 0.01), ('b', 0.1), ('rho', -0.3), ('m', 0.02), ('sigma', 0.2), ('ATM_Ref', 1.25))}
        key = ('vol_time_grid', (('SVI', 'smile'),))
        return utils.VolSurface._moneyness(strike, queries[q], key, SimpleNamespace(
            t_Buffer={key: (smile, (None, None, ('SVI', 'Sticky_Strike'), utils.CurveTenor(grid)), False)},
            one=torch.ones(1, dtype=dtype)))

    for dtype in (torch.float64, torch.float32):
        for flat in (1.0 / 3.0, 0.0123, 0.2345):
            level = torch.tensor(flat, dtype=dtype)
            for name, read in reads.items():
                got = read(level.expand(shapes.get(name, (grid.size, 3))).contiguous(), every)
                assert got.dtype == dtype and (got == level).all(), (name, dtype, flat, int((got != level).sum()))
        smiles = svi(torch.linspace(0.8, 1.6, 9, dtype=dtype).reshape(1, -1), every, dtype)
        assert (smiles == smiles[0]).all(), ('svi', dtype, int((smiles != smiles[0]).sum()))
    for name, read in reads.items():
        surface = torch.rand(shapes.get(name, (grid.size, 3)), generator=gen, dtype=torch.float64,
                             requires_grad=True)
        assert torch.autograd.gradcheck(lambda s: read(s, few), surface), name
        assert torch.autograd.gradgradcheck(lambda s: read(s, few), surface), name
    strike = torch.linspace(0.8, 1.6, 9, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda k: svi(k.reshape(1, -1), few, torch.float64), strike)
    assert torch.autograd.gradgradcheck(lambda k: svi(k.reshape(1, -1), few, torch.float64), strike)


def test_every_blend_is_exact_at_both_ends():
    """The tenor read at its two knots (weight 0, and 1 off the extrapolating last segment), the
    time blend and the whole-row gather at weight 0 and 1, and the Hermite blend at 0 and 1 give
    their ends bit for bit - in both precisions, on values spanning six decades and both signs,
    where `y0 + w (y1 - y0)` is not exact at w = 1.

    Killed by: the tenor read, the time blend or the Hermite blend as `y0 + w (y1 - y0)`."""
    gen = torch.Generator().manual_seed(11)

    def spread():
        return torch.randn(4096, generator=gen, dtype=torch.float64) * 10.0 ** (
            6.0 * torch.rand(4096, generator=gen, dtype=torch.float64) - 3.0)

    for dtype in (torch.float64, torch.float32):
        y0, y1, g, c = (spread().to(dtype) for _ in range(4))
        hermite = utils.Interpolation.calc_hermite_curve
        assert torch.equal(hermite(torch.zeros_like(y0), g, c, y0, y1), y0)
        assert torch.equal(hermite(torch.ones_like(y0), g, c, y0, y1), y1)
        # two scenario rows of a two-knot curve: [[y0, y1], [y1, y0]]
        values = torch.stack([torch.stack([y0, y1]), torch.stack([y1, y0])])
        interp = utils.Interpolation.build(values, 'LinearExtrapolate', np.array([0.0, 1.0]))
        component = (False, 'curve', None, utils.CurveTenor(np.array([0.0, 1.0]), 'LinearExtrapolate'),
                     _daycount('ACT_365'))
        curve = utils.CurveTensor(interp, np.zeros(2, dtype=np.int64), np.array([0.0, 1.0]).reshape(-1, 1, 1))
        assert torch.equal(curve.interpolate_curve(component, np.array([[0.0, 365.0]] * 2), 0), values)
        assert torch.equal(curve.interp_value(), values)
        # and a Hermite curve read at its three knots
        knots = np.array([0.0, 0.5, 1.0])
        values = torch.stack([y0, y1, g]).unsqueeze(0)
        curve = utils.CurveTensor(utils.Interpolation.build(values, 'Hermite', knots),
                                  np.zeros(1, dtype=np.int64), None)
        assert torch.equal(curve.interpolate_curve(
            (False, 'curve', None, utils.CurveTenor(knots, 'Hermite'), _daycount('ACT_365')),
            knots.reshape(1, -1) * 365.0, 0), values)


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

    Killed by: the blend's middle term with `t` for `1 - t`, 1.4e-5 of a mark off."""
    path = str(tmp_path / 'hermite.json')
    open(path, 'w').write(json.dumps(_hermite_job(kind)))
    cx = rf.Context(path_transform={}, file_transform={})
    cx.load_json(path)
    marks = cx.run_job()[1]['Results']['mtm'].set_index('Reference')['Value']
    for days in HERMITE_DAYS:
        assert marks['CF%d' % days] == pytest.approx(
            1e6 * math.exp(-_hermite_by_hand(kind, days / 365.0)), rel=1e-13), days
