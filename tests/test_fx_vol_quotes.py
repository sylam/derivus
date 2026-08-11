"""Does one backward pass report `dV/d(risk reversal)`, and is the map it goes through the right one?

Increment 4 of [Quote Sensitivities](../docs_src/developer/quote_sensitivities.md), and it is the
increment with a ROOT FIND in the middle of it. `FXVolSurfaceParameters` turns a delta-quoted smile
into the log-moneyness surface the option pricers read, and the chain is:

    quotes  ->  the strangle pair, exactly invertible and linear
            ->  the Malz wing pair, whose ATM NODE moves with the ATM quote
            ->  a bisection per pinned x-node, solving sigma = skew(delta(sigma, x))
            ->  the surface values  ->  (factor_leaf)  ->  V

THE TAPE STOPS AT THE BISECTION, and that is the whole design decision. Every operation in the 64
halvings is differentiable, so a tape runs straight through them and reports a number - the wrong
one. `left` and `right` are only ever `lo`, `hi` or a midpoint of two such, so the iterates are
DYADIC combinations of the two bracket endpoints and what a tape differentiates is where the
bracket is rather than where the root is. `test_taping_the_bisection_reports_the_brackets`
`_derivative_and_not_the_roots` is that mutation: the same forward number to 1e-17, a Jacobian
0.135 out on an entry of order 1, and `dsigma/d(ATM)` reported as a plausible-looking 1.0001 where
the truth is 0.866. So the twin starts at the CONVERGED root and takes one Newton step off it,
which is the implicit function theorem written as an expression.

FOUR THINGS ARE DISCRETE HERE, and a gate written without them is a placebo. A node reads one
WING, its fixed point is either inside that wing's bracket or CLAMPED, the root sits in one
SEGMENT of a piecewise linear wing, and a clamped node takes whichever ENDPOINT of the bracket its
residual misses by less. All four switch on the quotes. The first three are kinks - the map is
continuous across them and a central difference straddling one converges to the average of two
one-sided derivatives, which is nobody's - and the fourth is a JUMP, because the two endpoint
knots carry different vols. The ladder below therefore carries a fingerprint per node and scores
only the rungs where all four did not move, reporting how many did.

AND THE TAPE IS NOT THE VALUE. The numpy conversion this family has always shipped is the only
thing a written vol comes out of; the torch twin rides in as `value + (carried - carried.detach())`
for its derivative alone. That is increment 3's lesson and it is load-bearing for the same reason:
the value path's normal CDF is `scipy.special.ndtr` and the twin's is `utils.norm_cdf` over
`torch.erfc`, so the two disagree in the last bits, and an ulp of a shipped vol is a different
number in a report. The twin's own forward is measured against the shipped one and never written.

The world is `test_fx_vol_prices`'s, imported rather than restated: this increment differentiates
that family's own smile, and two copies of one fixture is two things to keep in step.

Run: ``pytest tests/test_fx_vol_quotes.py -q``
"""
import copy
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pytest
import torch
from scipy.stats import norm

import derivus
from derivus import utils
from derivus.bootstrappers import FXVolSurfaceParameters
from derivus.config import Config
from derivus.instruments import construct_instrument
from derivus.riskfactors import Factor2D, construct_factor
from derivus.schema import mapping

from test_fx_vol_prices import (BASE, DEVICE, DTYPE, EXPIRIES, FX_OPTION, INTERP, PILLARS, SEEN,
                                market_prices, quotes)

BLOCK = 'FXVolPrices.USD.ZAR'
VOL_NAME = 'FXVol.USD.ZAR'
VOL_FACTOR = utils.Factor('FXVol', ('USD', 'ZAR'))

#: The surface `c77740e` writes from these quotes, hashed. `Quote_Sensitivity` is a switch over what
#: `backward()` can reach, so the value it cannot reach is the one the parent commit shipped - and a
#: digest is what makes that a standing gate rather than a measurement somebody took once.
HEAD_SURFACE_SHA = '0d2f94df89ac7cc728be54881a3a0c3caaaaad14ef13e48458e4aa5748397019'


# =====================================================================================
# the world
# =====================================================================================

def prices(connect=True, points=None):
    """The family's own quote block, with the switch this increment adds set."""
    block = market_prices(points)
    block[BLOCK]['instrument']['Quote_Sensitivity'] = 'Yes' if connect else 'No'
    return block


def bootstrapped(connect=True, points=None, factors=None):
    """The family run the way `Config.bootstrap` runs it, returning it and what it wrote.

    `factors` is what the grid PINS to: passing back a previously written set is the tick path, and
    the only one a derivative is comparable across - a rebuild refines a different grid, and a
    difference quotient over two plans is not a difference quotient.
    """
    written = {} if factors is None else factors
    family = FXVolSurfaceParameters({}, DEVICE, DTYPE)
    family.bootstrap({'Base_Date': BASE}, {}, written, INTERP,
                     prices(connect, points), {})
    return family, written


def sorted_surface(written):
    """The written surface in the order `Factor2D` publishes its vol column in."""
    surface = written[VOL_NAME]['Surface'].array
    return surface[np.lexsort((surface[:, 0], surface[:, 1]))]


def pinned(written):
    """`{expiry: x nodes}` off a written surface - the grid every twin below evaluates on."""
    surface = written[VOL_NAME]['Surface'].array
    return {T: surface[surface[:, 1] == T][:, 0] for T in np.unique(surface[:, 1])}


def jacobian(family, written):
    """`d(written surface)/dq`, one backward per node OFF THE SPLICE the calculation attaches.

    Taken on the tensor `factor_leaf` is offered rather than on a closure written beside it, so
    what is measured is the derivative a job would actually report.
    """
    theta = family.calibrated[VOL_FACTOR]
    leaves = family.quote_leaves[BLOCK][1]
    rows = np.zeros((theta.shape[0], leaves.shape[0]))
    for i in range(theta.shape[0]):
        leaves.grad = None
        theta[i].backward(retain_graph=True)
        rows[i] = leaves.grad.numpy()
    leaves.grad = None
    return rows


def fingerprint(points, grid):
    """Per node, the four DISCRETE choices: wing, bracket, segment, and the CLAMP ENDPOINT.

    Every one of them is a switch in the quotes. The first three are kinks - the map is continuous
    across them and only its slope jumps - and the fourth is a genuine DISCONTINUITY: `malz_delta`
    picks the endpoint by `|f_lo| < |f_hi|`, those two magnitudes cross, and the two endpoint knots
    carry different vols, so the written number STEPS. A central difference is a difference
    quotient of nothing at all across any of the four.

    The endpoint is recorded on the CLAMPED branch alone, which is the branch that reads it: a
    bracketed node's vol comes off its own root and does not care which end missed by less.
    Recording it unconditionally would exclude rungs nothing happened on - five more at h = 1e-3,
    measured - which is a fingerprint scoring itself rather than the map.
    """
    delta_surface = FXVolSurfaceParameters.smile(points)
    skews = Factor2D.malz_skews(delta_surface, np.array(sorted(grid)))
    marks = []
    for T in sorted(grid):
        delta, is_call, bracketed = Factor2D.malz_delta(skews[T], T, grid[T])
        for k in range(len(grid[T])):
            knots = skews[T]['d_call' if is_call[k] else 'd_put']
            marks.append((bool(is_call[k]), bool(bracketed[k]), int(np.clip(
                np.searchsorted(knots, delta[k], side='right') - 1,
                0, max(knots.size - 2, 0))),
                -1 if bracketed[k] else int(np.abs(knots - delta[k]).argmin())))
    return marks


def clamped_and_call(points, grid):
    """The clamp mask and the wing mask, node for node, in the written surface's order."""
    marks = fingerprint(points, grid)
    return np.array([not m[1] for m in marks]), np.array([m[0] for m in marks])


def flat_segments(points, grid):
    """Per node, whether the wing SEGMENT its root sits in has two equal knot vols.

    A flat segment reads back the same vol from either end, so a BRACKETED node sitting in one
    answers exactly what a clamped node answers - `dsigma/d(ATM)` of exactly 1.0 - without being
    clamped. Ordinary quotes do it: two adjacent pillars whose `ATM + BF +- RR/2` coincide.
    """
    skews = Factor2D.malz_skews(FXVolSurfaceParameters.smile(points), np.array(sorted(grid)))
    marks, flat = iter(fingerprint(points, grid)), []
    for T in sorted(grid):
        for _ in grid[T]:
            is_call, _, seg, _ = next(marks)
            vols = skews[T]['v_call' if is_call else 'v_put']
            flat.append(vols[seg] == vols[min(seg + 1, vols.size - 1)])
    return np.array(flat)


def moved(points, index, shift):
    """One quote of the block moved by `shift`, everything else untouched."""
    bumped = copy.deepcopy(points)
    bumped[index]['Quoted_Market_Value'] += shift
    return bumped


def descriptors():
    return [FXVolSurfaceParameters.descriptor(point) for point in quotes()]


def quote_index(kind, expiry, pillar=None):
    for i, point in enumerate(quotes()):
        if point['Quote_Type'] == kind and point['Expiry'] == expiry and (
                kind == 'ATM' or point['Pillar'] == pillar):
            return i


# =====================================================================================
# the naive twin - the 64 halvings ON the tape. NOT engine code; the mutation's reference.
# =====================================================================================

def taped_bisection(skew, carried, T, x, iterations=64):
    """`Factor2D.malz_sigma` mirrored LITERALLY, bisection and all.

    Every operation here differentiates, and this is what the engine would report if the tape were
    simply run through the shipped loop. It is right forward and wrong backward, which is what
    makes it worth writing down.
    """
    sqrt_t = np.sqrt(T)
    is_call = torch.as_tensor(x <= 0.5 * float(carried['sigma_atm'].detach()) ** 2 * T)
    k_over_f = carried['sigma_atm'].new_tensor(np.exp(-x))
    xt = carried['sigma_atm'].new_tensor(x)

    def interp(delta, knots, values):
        seg = torch.clamp(torch.searchsorted(
            knots.detach().contiguous(), delta.detach().contiguous(), right=True) - 1,
            0, knots.numel() - 2)
        floor = values[seg]
        return floor + (delta - knots[seg]) * (values[seg + 1] - floor) / (
                knots[seg + 1] - knots[seg])

    def wing(delta):
        return torch.where(is_call, interp(delta, carried['d_call'], carried['v_call']),
                           interp(delta, carried['d_put'], carried['v_put']))

    def residual(delta):
        vol = wing(delta)
        d2 = (xt - 0.5 * vol * vol * T) / (vol * sqrt_t)
        sign = torch.where(is_call, torch.ones_like(vol), -torch.ones_like(vol))
        return k_over_f * sign * utils.norm_cdf(sign * d2) - delta

    lo = torch.where(is_call, torch.clamp(carried['d_call'].min(), min=0.0),
                     carried['d_put'].min())
    hi = torch.where(is_call, carried['d_call'].max(),
                     torch.clamp(carried['d_put'].max(), max=0.0))
    f_lo, f_hi = residual(lo), residual(hi)
    clamp = torch.where(f_lo.abs() < f_hi.abs(), lo, hi)

    left, f_left, right = lo, f_lo, hi
    for _ in range(iterations):
        middle = 0.5 * (left + right)
        f_middle = residual(middle)
        below = (f_left * f_middle <= 0.0).detach()
        right = torch.where(below, middle, right)
        left = torch.where(below, left, middle)
        f_left = torch.where(below, f_left, f_middle)

    return wing(torch.where((f_lo * f_hi > 0.0).detach(), clamp, 0.5 * (left + right)))


def twin_jacobian(solver, grid, points=None):
    """`d(surface)/dq` off whichever per-node solver is handed in - the twin, or the naive one."""
    points = quotes() if points is None else points
    delta_surface = FXVolSurfaceParameters.smile(points)
    expiries = np.array(sorted(grid))
    skews = Factor2D.malz_skews(delta_surface, expiries)

    def surface(q):
        carried = FXVolSurfaceParameters.carried_skews(
            delta_surface, expiries, FXVolSurfaceParameters.carried_smile(points, q))
        return torch.cat([solver(skews[T], carried[T], T, grid[T]) for T in expiries])

    leaves = torch.tensor([point['Quoted_Market_Value'] for point in points],
                          dtype=DTYPE, requires_grad=True)
    return surface(leaves).detach().numpy(), torch.autograd.functional.jacobian(
        surface, leaves).numpy()


# =====================================================================================
# (i) the forward - nothing about a value moves when the quote side is switched on
# =====================================================================================

def test_the_written_surface_is_bit_identical_with_quote_gradients_on_and_off():
    """`np.array_equal`, not a tolerance - and STRUCTURAL rather than lucky, because the numbers do
    not come out of the tape at all. The shipped conversion writes the surface either way and the
    torch twin is spliced in worth zero. Held to the PARENT COMMIT too, by digest: the switch is
    over what `backward()` can reach, so what it cannot reach has to be what c77740e shipped."""
    _, off = bootstrapped(False)
    _, on = bootstrapped(True)
    assert np.array_equal(off[VOL_NAME]['Surface'].array, on[VOL_NAME]['Surface'].array)
    assert hashlib.sha256(np.ascontiguousarray(
        off[VOL_NAME]['Surface'].array)).hexdigest() == HEAD_SURFACE_SHA

    # and the comparison can still fail - a vol on one butterfly moves the surface it holds
    _, ticked = bootstrapped(True, quotes({(1.0, 0.25, 'BF'): 0.01}))
    assert not np.array_equal(on[VOL_NAME]['Surface'].array,
                              ticked[VOL_NAME]['Surface'].array)


def test_the_spliced_tensor_is_the_column_the_leaf_is_minted_from():
    """The attachment stands for exactly the factor it is offered against: `Factor2D` sorts what it
    is handed by (expiry, moneyness) and mints a leaf out of THAT column, and the tensor published
    beside it has to be that column and not merely the same numbers.

    Three orders are in play and only two of them agree, which is why the pairing is computed
    rather than assumed: `malz_surface` EMITS expiry-major, `utils.Curve` stores what it is handed
    sorted by MONEYNESS first, and `Factor2D` lexsorts back to expiry-major before minting a leaf.
    The twin follows the emission, so it is paired against the emission.

    HONEST NEGATIVE RESULT, recorded rather than left implied. Emission order and (expiry,
    moneyness) order are the same order - here and on every quote set this family can build, since
    expiries arrive from `np.unique` and nodes from a sorted refinement - so the permutation is the
    IDENTITY and deleting the reorder leaves every gate in this file green. It is insurance against
    those two drifting apart, which nothing else would notice, and the identity is asserted below
    so that the day it stops being one is visible.
    """
    family, written = bootstrapped(True)
    factor = construct_factor(VOL_FACTOR, {VOL_NAME: written[VOL_NAME]}, INTERP)
    assert np.array_equal(family.calibrated[VOL_FACTOR].detach().numpy(), factor.current_value())
    assert np.array_equal(family.calibrated[VOL_FACTOR].detach().numpy(),
                          sorted_surface(written)[:, 2])

    delta_surface = FXVolSurfaceParameters.smile(quotes())
    expiries = np.unique(delta_surface[:, 1])
    emitted = np.array(Factor2D.malz_surface(
        Factor2D.malz_skews(delta_surface, expiries), pinned(written)))
    assert np.array_equal(np.lexsort((emitted[:, 0], emitted[:, 1])), np.arange(len(emitted))), (
        'the emission order and the sorted order have parted - the reorder is now load-bearing '
        'and this gate has to be able to fail')
    # and the Curve is NOT in that order, which is why the twin is paired off the emission
    assert not np.array_equal(written[VOL_NAME]['Surface'].array[:, 2], emitted[:, 2])


def test_a_block_that_did_not_ask_leaves_nothing_behind():
    """`Quote_Sensitivity` is the switch, and off it costs a config nothing to carry."""
    plain, _ = bootstrapped(False)
    assert not plain.calibrated and not plain.quote_leaves

    family, _ = bootstrapped(True)
    assert set(family.calibrated) == {VOL_FACTOR}
    labels, leaves = family.quote_leaves[BLOCK]
    assert labels[:3] == ['ATM 0.0833', 'RR 0.25 0.0833', 'BF 0.25 0.0833']
    assert labels[10:13] == ['ATM 1', 'RR 0.25 1', 'BF 0.25 1']
    # ONE VECTOR LEAF per block, which is the curve family's shape rather than the swaption
    # family's tuple of scalars - the whole quote set enters one conversion
    assert isinstance(leaves, torch.Tensor) and leaves.shape == (len(quotes()),)
    assert leaves.grad is None, 'the leaf was handed over dirty'


# =====================================================================================
# (ii) the twin against the conversion it mirrors
# =====================================================================================

def test_the_carried_smile_is_bit_for_bit_the_strangle_algebra():
    """`smile` is `+`, `*` and a sort, with no `sqrt` in it for two implementations to disagree
    over, so the mirror is EQUAL rather than close. That is what lets everything downstream index
    the twin with structure read off the value path."""
    points = quotes()
    leaves = torch.tensor([point['Quoted_Market_Value'] for point in points], dtype=DTYPE)
    carried = FXVolSurfaceParameters.carried_smile(points, leaves)
    assert np.array_equal(carried.numpy(), FXVolSurfaceParameters.smile(points)[:, 2])


#: An ATM vol whose delta-neutral straddle is a number `np.exp` and `torch.exp` round differently.
#: `0.5 exp(-0.5 sigma^2 T)` at 8.29% / 1y is 0.49828484599801914 in numpy and one ulp below that
#: in torch, and 0.66% of a 200,000-point (sigma, T) census disagrees the same way. Nothing is
#: wrong with either; the numbers are ordinary and the main fixture happens to miss every one of
#: them, which is exactly what makes `array_equal` on the DELTA arrays a red gate lying in wait.
EXP_DESYNC = [{'Use': 'Yes', 'Expiry': 1.0, 'Pillar': 0.0, 'Quote_Type': 'ATM',
               'Quoted_Market_Value': 0.0829, 'Timestamp': SEEN},
              {'Use': 'Yes', 'Expiry': 1.0, 'Pillar': 0.25, 'Quote_Type': 'RR',
               'Quoted_Market_Value': 0.02, 'Timestamp': SEEN},
              {'Use': 'Yes', 'Expiry': 1.0, 'Pillar': 0.25, 'Quote_Type': 'BF',
               'Quoted_Market_Value': 0.004, 'Timestamp': SEEN}]


@pytest.mark.parametrize('points', [None, EXP_DESYNC], ids=['the fixture', 'an exp desync'])
def test_the_carried_skew_is_the_wing_pair_node_for_node(points):
    """The LAYOUT is what the twin has to reproduce, not only the numbers: which node carries the
    +-0.5 label, which side had its ATM node mirrored in, and the order they end up in. Every
    frozen index below addresses this dict, so a layout that drifted would read the wrong knot and
    still return a plausible vol.

    THE WING VOLS ARE EQUAL AND THE DELTAS ARE NOT, and the split is the increment-3 lesson
    recurring on a different function. `smile` is `+`, `*` and a sort, so its mirror is bit-exact;
    the delta grid carries `delta_atm = 0.5 exp(-0.5 sigma^2 T)`, and `np.exp` and `torch.exp` are
    two correctly-rounded implementations that disagree in the last bit on 0.66% of ordinary vols -
    the `sqrt` finding of increment 3, on `exp`. So the deltas are held to ONE ULP and the vols to
    the bit, and the second parametrisation is a smile that actually reaches it rather than a
    tolerance taken on faith. The VALUE path is untouched by any of it: the deltas that differ are
    the twin's own mirror arrays, and what is written comes out of numpy either way.
    """
    points = quotes() if points is None else points
    delta_surface = FXVolSurfaceParameters.smile(points)
    expiries = np.unique(delta_surface[:, 1])
    leaves = torch.tensor([point['Quoted_Market_Value'] for point in points], dtype=DTYPE)
    numpy_skews = Factor2D.malz_skews(delta_surface, expiries)
    carried = FXVolSurfaceParameters.carried_skews(
        delta_surface, expiries, FXVolSurfaceParameters.carried_smile(points, leaves))

    for T in expiries:
        for key in ('v_put', 'v_call'):
            assert np.array_equal(carried[T][key].numpy(), numpy_skews[T][key]), (T, key)
        for key in ('d_put', 'd_call'):
            mirror, value = carried[T][key].numpy(), numpy_skews[T][key]
            assert np.array_equal(np.sign(mirror), np.sign(value)), (T, key)
            assert (np.abs(mirror - value) <= np.spacing(np.abs(value))).all(), (T, key)
        assert float(carried[T]['sigma_atm']) == numpy_skews[T]['sigma_atm']
        assert float(carried[T]['delta_atm']) == pytest.approx(
            numpy_skews[T]['delta_atm'], rel=1e-15)


def test_the_twin_agrees_with_the_shipped_conversion_and_writes_none_of_it():
    """The twin's forward is the solve's own residual away from the value path and no further -
    2.8e-17 here, which is the Newton step being worth nothing at a converged root. It is a
    DIAGNOSTIC on a number that reaches no mark: the splice is what makes the written surface the
    numpy one, and the gate above asserts that separately and exactly."""
    _, written = bootstrapped(True)
    grid = pinned(written)
    value, _ = twin_jacobian(FXVolSurfaceParameters.carried_sigma, grid)
    assert np.abs(value - sorted_surface(written)[:, 2]).max() < 1e-15


# =====================================================================================
# (iii) the tape boundary - the bisection is not on the tape, and here is why
# =====================================================================================

def test_taping_the_bisection_reports_the_brackets_derivative_and_not_the_roots():
    """THE MUTATION THIS INCREMENT TURNS ON. Mirror the shipped loop literally and the forward is
    the same number to 1e-17 while the Jacobian is 0.135 out on entries of order 1.

    A bisection's iterates are dyadic combinations of the two bracket ENDPOINTS - `left` and
    `right` are only ever `lo`, `hi` or a midpoint of two such - so a tape through the loop carries
    `d(endpoint)/dq` and not `d(root)/dq`. On the call wing `lo` is a quoted pillar delta and `hi`
    is `delta_atm`, so the reported number is a function of the ATM quote and of nothing else the
    root actually moves with. It reads 1.0001 for `dsigma/d(ATM)` where the truth is 0.866, which
    is exactly the failure mode this workstream exists to prevent: plausible, and wrong.
    """
    _, written = bootstrapped(True)
    grid = pinned(written)
    value, naive = twin_jacobian(taped_bisection, grid)
    _, ift = twin_jacobian(FXVolSurfaceParameters.carried_sigma, grid)

    assert np.abs(value - sorted_surface(written)[:, 2]).max() < 1e-15, (
        'the naive twin does not even agree forward, so this is not measuring the derivative')
    worst = np.abs(naive - ift).max()
    assert worst > 0.1, 'the two tapes agree - the bisection has stopped being differentiable'
    assert np.linalg.norm(naive - ift) / np.linalg.norm(ift) > 0.05

    # and the entry it is worst on is a headline number, not a corner of the surface
    row, column = np.unravel_index(np.abs(naive - ift).argmax(), ift.shape)
    assert descriptors()[column].startswith('ATM')
    assert ift[row, column] == pytest.approx(0.865559, rel=1e-5)
    assert naive[row, column] == pytest.approx(1.000137, rel=1e-5)


# =====================================================================================
# (iv) what the Jacobian MEANS - the quote algebra has to be visible in it
# =====================================================================================

def test_the_jacobian_is_block_diagonal_in_expiry():
    """Each expiry's smile is built from its own rows and the x-grid is refined per expiry, so a
    quote reaches no node of any other expiry. EXACTLY zero, not small."""
    family, written = bootstrapped(True)
    J, rows = jacobian(family, written), sorted_surface(written)
    for j, point in enumerate(quotes()):
        assert np.abs(J[rows[:, 1] != point['Expiry'], j]).max() == 0.0, descriptors()[j]
    assert np.abs(J).max() > 0.5, 'nothing is on the tape at all'


def test_the_risk_reversal_is_half_the_butterfly_wing_by_wing_with_the_sign_flipped():
    """THE STRUCTURE, as an exact identity rather than a shape somebody eyeballed.

    A node's vol depends on the quotes only through its own wing's knot vols (and, through
    `delta_atm`, on the ATM quote). Those knots are `ATM + BF + RR/2` on the call side and
    `ATM + BF - RR/2` on the put side, so BF and RR enter through the SAME channel with
    coefficients 1 and +-1/2 - and the two Jacobian columns are therefore in exact ratio, with the
    sign given by which wing the node reads. That is what "the risk reversal is antisymmetric and
    the butterfly is symmetric" means, stated so that floating point can check it.
    """
    family, written = bootstrapped(True)
    J, rows = jacobian(family, written), sorted_surface(written)
    _, is_call = clamped_and_call(quotes(), pinned(written))

    for T in EXPIRIES:
        for pillar in PILLARS:
            at = rows[:, 1] == T
            rr = J[at, quote_index('RR', T, pillar)]
            bf = J[at, quote_index('BF', T, pillar)]
            call = is_call[at]
            assert np.array_equal(rr[call], 0.5 * bf[call]), (T, pillar, 'call wing')
            assert np.array_equal(rr[~call], -0.5 * bf[~call]), (T, pillar, 'put wing')
            # and the butterfly is same-signed on both wings, which is the other half of it
            assert (bf >= 0.0).all() and bf.max() > 0.9


def test_the_atm_quote_moves_every_node_of_its_own_expiry():
    """A parallel shift of the smile is not a parallel shift of the SURFACE. Every wing knot
    carries the ATM quote with coefficient one, so the level moves with it; but `delta_atm` moves
    too, which slides the delta each log-moneyness node resolves to, so the column lands NEAR one
    rather than on it. Between 0.86 and 1.06 here, everywhere positive."""
    family, written = bootstrapped(True)
    J, rows = jacobian(family, written), sorted_surface(written)
    for T in EXPIRIES:
        column = J[rows[:, 1] == T, quote_index('ATM', T)]
        assert (column > 0.0).all(), T
        assert 0.85 < column.min() and column.max() < 1.06, (T, column.min(), column.max())
        assert not np.allclose(column, 1.0), (
            'the ATM column is exactly one, so `delta_atm` reached nothing')


# =====================================================================================
# (v) the quote Jacobian against a central difference of the whole family
# =====================================================================================

@pytest.mark.parametrize('h,bracketed_error', [(1e-3, 9.8e-5), (1e-4, 9.8e-7), (1e-5, 9.8e-9)])
def test_the_quote_jacobian_is_the_central_difference_of_the_whole_family(h, bracketed_error):
    """AAD against a re-bootstrap at q +- h, on a ladder that converges as h^2.

    The difference is taken through the WHOLE family - re-authored smile, re-prepared wings,
    re-solved on the SAME pinned grid - so what converges is the derivative of the thing the job
    runs. Three things are scored separately and none of them is a tolerance chosen for comfort:

      bracketed nodes carry the h^2 term and are the ladder, 9.8e-5 / 9.8e-7 / 9.8e-9;
      clamped nodes are EXACTLY LINEAR in the quotes, so their quotient is exact at every h and
        what is left is its own rounding - which GROWS as h shrinks, 1.3e-14 to 1.8e-12;
      a node whose fingerprint moved across the rung straddles one of the four switches and is
        excluded by MEASUREMENT rather than by taste - 24 of them at h=1e-3, 5 at 1e-4, none at
        1e-5.
    """
    _, factors = bootstrapped(True)
    reference = copy.deepcopy(factors)
    family, written = bootstrapped(True, factors=copy.deepcopy(reference))
    J, grid = jacobian(family, written), pinned(written)
    clamped, _ = clamped_and_call(quotes(), grid)
    coordinates = sorted_surface(written)[:, :2]

    worst_live, worst_clamped, straddles = 0.0, 0.0, 0
    for j in range(len(quotes())):
        columns = []
        for shift in (h, -h):
            points = moved(quotes(), j, shift)
            _, bumped = bootstrapped(True, points, copy.deepcopy(reference))
            assert np.array_equal(sorted_surface(bumped)[:, :2], coordinates), 'the pin broke'
            columns.append((points, sorted_surface(bumped)[:, 2]))
        fd = (columns[0][1] - columns[1][1]) / (2.0 * h)
        held = np.array([a == b for a, b in zip(fingerprint(columns[0][0], grid),
                                                fingerprint(columns[1][0], grid))])
        straddles += int((~held).sum())
        error = np.abs(fd - J[:, j])
        worst_live = max(worst_live, error[held & ~clamped].max())
        worst_clamped = max(worst_clamped, error[held & clamped].max())

    assert worst_live < bracketed_error, worst_live
    assert worst_live > 0.02 * bracketed_error, (
        'the rung is far better than h^2 predicts - is the ladder measuring anything?')
    assert worst_clamped < 1e-11, worst_clamped
    assert (straddles == 0) == (h < 1e-4), (h, straddles)


# =====================================================================================
# (vi) the clamp - the branch with no root in it
# =====================================================================================

def test_the_clamp_fires_and_its_rows_are_the_strangle_coefficients_exactly():
    """A third of the grid has no fixed point inside its wing's bracket, and that is not a repair.

    Beyond the widest quoted delta the smile is FLAT: the vol at such a node IS the endpoint knot's,
    so its Jacobian row is that knot's own quote algebra and nothing else - `1` in the expiry's ATM
    quote, `1` in the pillar's butterfly, `+-1/2` in its risk reversal, and EXACTLY zero everywhere
    else. All 32 clamped rows, to the last bit, which is a stronger statement than a tolerance and
    the one the flat extrapolation actually makes.

    THE ROW IS ASSERTED AS A SET rather than entry by entry, and the difference is what a gate can
    see. Walking `flatnonzero(J[i])` visits the columns that ARE live and can never notice one that
    is MISSING - a family that dropped the risk reversals from the tape publishes a shorter row of
    perfectly correct entries and walks straight through. The endpoint here is always the widest
    quoted pillar (`d_call` runs 0.10, 0.25, delta_atm and the clamp lands on 0.10 at every expiry
    and both wings), so the whole row is known and equality is available.
    """
    family, written = bootstrapped(True)
    J, rows = jacobian(family, written), sorted_surface(written)
    grid = pinned(written)
    clamped, is_call = clamped_and_call(quotes(), grid)
    labels = descriptors()

    assert 0.2 < clamped.mean() < 0.5, 'the clamp census moved: {}'.format(clamped.mean())
    for i in np.flatnonzero(clamped):
        T, pillar = rows[i, 1], PILLARS[-1]
        assert {labels[k]: J[i, k] for k in np.flatnonzero(J[i])} == {
            'ATM {:g}'.format(T): 1.0,
            'BF {:g} {:g}'.format(pillar, T): 1.0,
            'RR {:g} {:g}'.format(pillar, T): 0.5 if is_call[i] else -0.5}, (i, rows[i])

    # and the branch it is NOT: a bracketed row's ATM entry is not 1, so the gate above is
    # distinguishing the two rather than describing every row on the surface. Conditioned on the
    # SEGMENT not being flat, because a bracketed node inside one reads back exactly 1.0 too and
    # is nobody's defect - see `flat_segments` and the gate below it
    live = ~clamped & ~flat_segments(quotes(), grid)
    assert live.any()
    assert not any(J[i, quote_index('ATM', rows[i, 1])] == 1.0 for i in np.flatnonzero(live))


#: Two adjacent pillars whose call-wing vols COINCIDE: `ATM + BF + RR/2` is 0.1396 at the 0.35
#: delta and 0.1396 again at the 0.25, so the wing is FLAT between them. Ordinary broker numbers -
#: a mild inverted skew on a three-year - and nothing about the smile is a corner case.
FLAT_WING = [{'Use': 'Yes', 'Expiry': 3.0, 'Pillar': 0.0, 'Quote_Type': 'ATM',
              'Quoted_Market_Value': 0.14, 'Timestamp': SEEN},
             {'Use': 'Yes', 'Expiry': 3.0, 'Pillar': 0.35, 'Quote_Type': 'RR',
              'Quoted_Market_Value': -0.0040, 'Timestamp': SEEN},
             {'Use': 'Yes', 'Expiry': 3.0, 'Pillar': 0.35, 'Quote_Type': 'BF',
              'Quoted_Market_Value': 0.0016, 'Timestamp': SEEN},
             {'Use': 'Yes', 'Expiry': 3.0, 'Pillar': 0.25, 'Quote_Type': 'RR',
              'Quoted_Market_Value': -0.0090, 'Timestamp': SEEN},
             {'Use': 'Yes', 'Expiry': 3.0, 'Pillar': 0.25, 'Quote_Type': 'BF',
              'Quoted_Market_Value': 0.0041, 'Timestamp': SEEN}]


def test_a_flat_wing_segment_answers_the_clamped_number_without_being_clamped():
    """WHY THE ANTI-ASSERTION ABOVE IS CONDITIONED, measured rather than argued.

    `dsigma/d(ATM) == 1.0` was read as the clamped branch's signature. It is not: it is the
    signature of a node whose vol does not move with delta, and a BRACKETED node sitting in a flat
    segment is one of those - the wing reads the same vol from either end of the interval, so
    sliding `delta_atm` under it changes nothing and the level follows the ATM quote exactly.

    On the fixture above no segment is flat and the unconditioned assertion passes by luck. Here
    two ordinary pillars coincide to 0.1396 and it would fail on nodes with no defect in them, so
    the mask is what makes it a statement about the clamp rather than about this quote set.
    """
    family, written = bootstrapped(True, FLAT_WING)
    J, grid = jacobian(family, written), pinned(written)
    clamped, _ = clamped_and_call(FLAT_WING, grid)
    flat = flat_segments(FLAT_WING, grid)

    exactly_one = [i for i in np.flatnonzero(~clamped) if J[i, 0] == 1.0]
    assert exactly_one, 'no bracketed node reads exactly 1.0 - the flat segment stopped firing'
    assert flat[exactly_one].all(), 'a bracketed 1.0 that is NOT a flat segment - a real defect'
    assert (~clamped & ~flat).any(), 'every bracketed node is flat here - the mask excludes all'


def test_a_flat_smile_is_one_knot_a_wing_and_its_jacobian_is_the_expiry_indicator():
    """THE STRUCTURAL ZERO the double-where is for, and it is an ordinary config rather than a
    pathology: quote the ATM row and nothing else.

    `malz_skew` mirrors the single ATM node onto both sides, so each wing is ONE knot, its span is
    exactly zero and so is its slope. Dividing first and selecting after puts a NaN on every entry
    of the Jacobian - measured, all four columns - while the value path is perfectly happy and
    writes a flat surface. That is this increment's version of the guarded discriminant: a forward
    that is right and a backward that is not, on data nobody would call a corner case.

    What comes back is the expiry indicator: a flat smile IS its ATM vol at every node of its own
    expiry, so `dsigma/d(ATM)` is exactly one there and exactly zero everywhere else.
    """
    flat = [point for point in quotes() if point['Quote_Type'] == 'ATM']
    family, written = bootstrapped(True, flat)
    rows = sorted_surface(written)
    assert np.array_equal(np.unique(rows[:, 2]), np.array([0.145, 0.152, 0.161, 0.168]))

    J = jacobian(family, written)
    assert np.isfinite(J).all(), 'the one-knot wing put a NaN on the tape'
    assert np.array_equal(J, np.array([[float(rows[i, 1] == point['Expiry']) for point in flat]
                                       for i in range(len(rows))]))


#: The quote whose move flips a node between the two branches, and the bump that puts it exactly on
#: the switch - located by bisection rather than authored, because where the root first reaches the
#: wing's endpoint is a property of the smile and not a number anyone can write down.
SWITCHING_QUOTE = ('BF', 1.0, 0.10)


def clamp_switch():
    """`(quote, bump, node, grid)` - the butterfly move landing one node exactly on the switch.

    Everything downstream of this pins to the grid built here: a rebuild would refine a grid of its
    own and the node index would name a different x.
    """
    _, reference = bootstrapped(True)
    grid = pinned(reference)
    j = quote_index(*SWITCHING_QUOTE)
    base, _ = clamped_and_call(quotes(), grid)

    # the SMALLEST bump that flips anything, and then the node it flipped: a larger bracket picks
    # up several switches at once and bisecting on one of those lands between two kinks
    lo, hi = 0.0, 0.05
    for _ in range(60):
        middle = 0.5 * (lo + hi)
        if (clamped_and_call(moved(quotes(), j, middle), grid)[0] != base).any():
            hi = middle
        else:
            lo = middle
    node = int(np.flatnonzero(clamped_and_call(moved(quotes(), j, hi), grid)[0] != base)[0])
    return j, lo, node, reference


def on_the_pin(points, reference):
    """The surface those quotes write onto an already-built grid - the tick path."""
    return bootstrapped(True, points, copy.deepcopy(reference))


@pytest.mark.parametrize('offset,expected', [(1e-6, 0.8806047), (-1e-6, 1.0)])
def test_the_clamp_switch_is_a_kink_and_autograd_reports_its_own_branch(offset, expected):
    """The two branches meet where the root arrives AT the wing's endpoint, so the map is
    continuous and its derivative is not. Above the switch the node is bracketed and the reported
    number is the implicit-function one, 0.8806047; below it the node is clamped and the vol IS the
    endpoint knot's, so the derivative is EXACTLY 1. Each side is measured with a one-sided
    difference taken entirely inside its own branch, which is the only quotient a piecewise map has
    a limit for.
    """
    j, switch, node, reference = clamp_switch()
    points = moved(quotes(), j, switch + offset)
    family, written = on_the_pin(points, reference)
    assert clamped_and_call(points, pinned(reference))[0][node] == (offset < 0)
    assert jacobian(family, written)[node, j] == pytest.approx(expected, rel=1e-6)

    step = np.sign(offset) * 1e-9
    here = sorted_surface(written)[node, 2]
    there = sorted_surface(on_the_pin(moved(points, j, step), reference)[1])[node, 2]
    assert (there - here) / step == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize('h', [1e-6, 1e-8])
def test_a_central_difference_across_the_clamp_switch_answers_neither_branch(h):
    """MUTATE the instrument, not the code: straddle the kink and the quotient reports the AVERAGE
    of the two one-sided derivatives, 0.9403 at every h - a number that is nobody's. That is what a
    symmetric bump ladder would have quietly produced here, and it is why the ladder above carries
    a fingerprint per node instead."""
    j, switch, node, reference = clamp_switch()
    up = sorted_surface(on_the_pin(moved(quotes(), j, switch + h), reference)[1])[node, 2]
    down = sorted_surface(on_the_pin(moved(quotes(), j, switch - h), reference)[1])[node, 2]
    assert (up - down) / (2.0 * h) == pytest.approx(0.5 * (1.0 + 0.8806047), abs=1e-5)


#: The expiry and the quotes of a smile steep enough to put the clamp's two ENDPOINTS in
#: competition: the risk reversal is 2.5x the ATM vol, so the put wing's 25-delta vol is -0.05
#: before the family's 1e-4 floor and the residual misses at both ends of the bracket by
#: comparable amounts. The flip is not confined to a smile this steep - it starts around
#: RR ~ 1.5x ATM, where the same bump jumps a node 0.086 - but at 2.5x nobody can call it rounding.
STEEP_EXPIRY = 1.0
STEEP_WING = [{'Use': 'Yes', 'Expiry': STEEP_EXPIRY, 'Pillar': 0.0, 'Quote_Type': 'ATM',
               'Quoted_Market_Value': 0.12, 'Timestamp': SEEN},
              {'Use': 'Yes', 'Expiry': STEEP_EXPIRY, 'Pillar': 0.25, 'Quote_Type': 'RR',
               'Quoted_Market_Value': 0.30, 'Timestamp': SEEN},
              {'Use': 'Yes', 'Expiry': STEEP_EXPIRY, 'Pillar': 0.25, 'Quote_Type': 'BF',
               'Quoted_Market_Value': -0.02, 'Timestamp': SEEN}]


def test_the_clamp_endpoint_is_a_fourth_choice_and_a_jump_the_other_three_cannot_see():
    """THE FOURTH DISCRETE CHOICE, and the only one that moves the VALUE rather than the slope.

    `malz_delta` clamps to `lo` or `hi` by `|f_lo| < |f_hi|` - which endpoint the residual misses by
    less. Those two magnitudes are continuous in the quotes and they CROSS, and the two endpoints
    are different knots carrying different vols, so where they cross the written number steps.
    Here a 2e-6 move in the ATM quote jumps one node by 0.1199 of vol, `delta*` going -0.496413
    (the mirrored ATM knot, 0.12) to -0.250000 (the quoted pillar, floored at 1e-4). The map is
    genuinely DISCONTINUOUS there - this is the flat extrapolation changing its mind about which
    knot to extrapolate from - so no derivative exists at the crossing and autograd's one-sided
    answer is right on each side of it. There is nothing to fix in the engine.

    WHAT WAS WRONG WAS THE INSTRUMENT. Wing, bracket and segment are all identical either side:
    the node is clamped both times, a two-knot wing has only segment 0, and both endpoints are on
    the put side. A three-mark fingerprint therefore SCORES the rung, and scores a jump divided by
    2h - 6.0e+04 at h = 1e-6 and ten times that at 1e-7, which is the signature of a step and not
    of an error in anything. With the endpoint recorded the rung is excluded by measurement and
    what is left on the clamped branch is exact to rounding.
    """
    family, written = bootstrapped(True, STEEP_WING)
    J, rows = jacobian(family, written), sorted_surface(written)
    grid, reference = pinned(written), copy.deepcopy(written)
    clamped, _ = clamped_and_call(STEEP_WING, grid)

    # the jump itself, off the ATM quote, and the two knots it steps between
    up, down = moved(STEEP_WING, 0, 1e-6), moved(STEEP_WING, 0, -1e-6)
    above = sorted_surface(on_the_pin(up, reference)[1])[:, 2]
    below = sorted_surface(on_the_pin(down, reference)[1])[:, 2]
    node = int(np.abs(above - below).argmax())
    assert np.abs(above - below)[node] == pytest.approx(0.119901, abs=1e-5)
    assert clamped[node] and above[node] == pytest.approx(0.12, abs=1e-4)
    assert below[node] == pytest.approx(1e-4, abs=1e-9)

    resolved = []
    for points in (up, down):
        skew = Factor2D.malz_skews(FXVolSurfaceParameters.smile(points),
                                   np.array([STEEP_EXPIRY]))[STEEP_EXPIRY]
        delta, _, _ = Factor2D.malz_delta(skew, STEEP_EXPIRY, grid[STEEP_EXPIRY])
        resolved.append(delta[np.flatnonzero(grid[STEEP_EXPIRY] == rows[node, 0])[0]])
    assert resolved[0] == pytest.approx(-0.496413, abs=1e-6)
    assert resolved[1] == pytest.approx(-0.25, abs=1e-12)

    # and the first three marks are IDENTICAL across it while the fourth is not
    assert fingerprint(up, grid)[node][:3] == fingerprint(down, grid)[node][:3]
    assert fingerprint(up, grid)[node] != fingerprint(down, grid)[node]

    # the ladder, scored both ways: what the four marks exclude is a step, so what the three would
    # have reported grows as 1/h and multiplies back to the jump
    for h in (1e-6, 1e-7):
        worst, blind = 0.0, 0.0
        for j in range(len(STEEP_WING)):
            columns = []
            for shift in (h, -h):
                points = moved(STEEP_WING, j, shift)
                _, bumped = on_the_pin(points, reference)
                assert np.array_equal(sorted_surface(bumped)[:, :2], rows[:, :2]), 'the pin broke'
                columns.append((points, sorted_surface(bumped)[:, 2]))
            marks = [fingerprint(points, grid) for points, _ in columns]
            error = np.abs((columns[0][1] - columns[1][1]) / (2.0 * h) - J[:, j])
            held = np.array([a == b for a, b in zip(*marks)])
            seen = np.array([a[:3] == b[:3] for a, b in zip(*marks)])
            worst = max(worst, error[held & clamped].max(initial=0.0))
            blind = max(blind, error[seen & clamped].max(initial=0.0))
        assert worst < 1e-10, (h, worst)
        assert blind * 2.0 * h == pytest.approx(0.119901, rel=1e-3), (h, blind)


# =====================================================================================
# (vii) the attachment - dV/dq in one backward, beside an unchanged dV/dtheta
# =====================================================================================

def world(connect=True, points=None):
    """The family's own USDZAR call, priced off the surface this family bootstrapped."""
    config = Config(base_currency='ZAR')
    config.params['System Parameters']['Base_Date'] = BASE
    config.params['Price Factors'] = {
        'FxRate.ZAR': {'Domestic_Currency': None, 'Interest_Rate': 'ZAR', 'Priority': 1,
                       'Spot': 1.0},
        'FxRate.USD': {'Domestic_Currency': 'ZAR', 'Interest_Rate': 'USD', 'Priority': 2,
                       'Spot': 18.4},
        'InterestRate.ZAR': {'Currency': 'ZAR', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                             'Curve': utils.Curve([], [[0.0, 0.075], [5.0, 0.079]])},
        'InterestRate.USD': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                             'Curve': utils.Curve([], [[0.0, 0.041], [5.0, 0.039]])}}
    config.params['Market Prices'] = prices(connect, points)
    config.params['Bootstrapper Configuration'] = {'FXVolSurfaceParameters': {}}
    config.bootstrap()
    config.deals = {'Attributes': {'Reference': 'test', 'Tag_Titles': ''},
                    'Deals': {'Children': [{'Instrument': construct_instrument(FX_OPTION, {})}]},
                    'Calculation': {'Base_Date': BASE, 'Currency': 'ZAR'}}
    return config


def run(config, greeks='First'):
    """One base valuation, with the quote leaves cleared first - `.grad` accumulates."""
    for _, leaves in config.quote_leaves.values():
        leaves.grad = None
    _, out = derivus.run_baseval(config, prec=DTYPE, overrides={
        'Greeks': greeks, 'Random_Seed': 1, 'MCMC_Simulations': 1})
    return out['Results']


def factor_grad(results, rows):
    """The reported `FXVol` greek, back on the full node grid - `gradients_as_df` drops the zeros."""
    grad = np.zeros(len(rows))
    for (rate, x, T, _), value in results['Greeks_First']['root'].items():
        if rate == VOL_NAME:
            grad[np.flatnonzero((rows[:, 0] == x) & (rows[:, 1] == T))[0]] = value
    return grad


def test_a_valuation_reports_dV_dq_beside_dV_dtheta_in_one_pass():
    """The whole chain, closed in value space: `dV/dq = J' dV/dtheta`.

    Both halves come out of the SAME backward pass - the factor greek off the reported frame, the
    quote delta off the leaf - and the contraction is the surface Jacobian, taken independently.
    The vega chain is the sanity check beside it: this option's two live surface nodes carry
    7.04e6 of vol sensitivity against a Black-Scholes vega of 7.03e6, and `dV/d(ATM 1y)` is 0.99 of
    that - the ATM quote moves the whole smile, so nearly all of the vega lands on it.
    """
    config = world(True)
    results = run(config)
    rows = sorted_surface(config.params['Price Factors'])
    theta_grad = factor_grad(results, rows)
    quote_grad = config.quote_leaves[BLOCK][1].grad.numpy()

    family, _ = bootstrapped(True)
    J = jacobian(family, config.params['Price Factors'])
    assert np.allclose(quote_grad, J.T @ theta_grad, rtol=1e-12, atol=1e-6)
    assert np.abs(quote_grad).max() > 1e6, 'nothing reached the leaf'

    # the vega chain, as a magnitude rather than an equality
    forward = 18.4 * np.exp((0.075 - 0.041) * 1.0)
    at_expiry = rows[rows[:, 1] == 1.0]
    sigma = np.interp(np.log(forward / FX_OPTION['Strike_Price']), at_expiry[:, 0], at_expiry[:, 2])
    d1 = (np.log(forward / FX_OPTION['Strike_Price']) + 0.5 * sigma ** 2) / sigma
    vega = FX_OPTION['Underlying_Amount'] * np.exp(-0.075) * forward * norm.pdf(d1)
    assert theta_grad.sum() == pytest.approx(vega, rel=5e-3)
    assert quote_grad[quote_index('ATM', 1.0)] / vega == pytest.approx(0.99, abs=0.02)


def test_the_reported_factor_greek_does_not_move_when_the_quote_side_is_switched_on():
    """`dV/dq` arrives in the same pass and changes nothing about what that pass already said - the
    MTM and the whole gradient frame, `np.array_equal` rather than close."""
    off, on = run(world(False)), run(world(True))
    assert np.array_equal(off['mtm']['Value'].values, on['mtm']['Value'].values)
    assert np.array_equal(off['Greeks_First'].values, on['Greeks_First'].values)


def test_the_contraction_gate_fails_against_the_wrong_jacobian():
    """MUTATE the reference. The identity - what a family that had lost the conversion entirely
    would publish, one quote per node - has to stop agreeing."""
    config = world(True)
    results = run(config)
    rows = sorted_surface(config.params['Price Factors'])
    theta_grad = factor_grad(results, rows)
    quote_grad = config.quote_leaves[BLOCK][1].grad.numpy()

    wrong = np.zeros((len(rows), len(quotes())))
    wrong[:len(quotes()), :] = np.eye(len(quotes()))
    assert not np.allclose(quote_grad, wrong.T @ theta_grad, rtol=1e-3)


def test_the_quote_delta_needs_the_factor_leaves():
    """A quote of a surface is downstream of the switch that makes that surface a leaf at all. With
    greeks off the `FXVol` leaf is never asked for a gradient, so the quote delta is not small - it
    is ABSENT, which is the honest answer. Reporting a zero there would be the exact failure this
    workstream exists to prevent."""
    config = world(True)
    run(config, greeks='No')
    assert config.quote_leaves[BLOCK][1].grad is None
    run(config, greeks='First')
    assert config.quote_leaves[BLOCK][1].grad is not None


def test_a_family_that_stops_publishing_takes_its_stale_tensor_back():
    """A HARVEST THAT ONLY EVER ADDS IS A LEAK WITH TEETH, and this is the seam it leaks at.

    `Config.bootstrap` collects `calibrated` and `quote_leaves` off every family it runs. Turn
    `Quote_Sensitivity` off, re-bootstrap with quotes that have since moved, and the connected
    tensor the PREVIOUS run published is still standing under the same factor key - every number in
    it the old surface against the old quotes. Nothing raises: a splice worth zero in the forward
    is invisible to every price gate, so what a backward would report is the last surface's
    Jacobian, silently, as today's.

    So a run drops the keys the family owns before publishing what it built. Scoped to the factor
    type it writes and the Market Prices type it reads - both already declared, and both needed,
    because `Config.bootstrap` runs one family at a time and every other family's entries have to
    survive it.
    """
    config = world(True)
    published = config.calibrated_factors[VOL_FACTOR].detach().numpy().copy()
    assert np.array_equal(published, sorted_surface(config.params['Price Factors'])[:, 2])

    # Yes -> Yes on a ticked quote: the entry is REPLACED, and it is the surface just written
    block = config.params['Market Prices'][BLOCK]['instrument']
    block['Points'] = quotes({(1.0, 0.25, 'BF'): 0.01})
    config.bootstrap()
    ticked = config.calibrated_factors[VOL_FACTOR].detach().numpy()
    assert not np.array_equal(ticked, published)
    assert np.array_equal(ticked, sorted_surface(config.params['Price Factors'])[:, 2])

    # Yes -> No on another tick: GONE, rather than left behind describing the surface before last
    block['Quote_Sensitivity'], block['Points'] = 'No', quotes({(1.0, 0.25, 'BF'): 0.02})
    config.bootstrap()
    assert VOL_FACTOR not in config.calibrated_factors
    assert BLOCK not in config.quote_leaves
    assert not np.array_equal(ticked, sorted_surface(config.params['Price Factors'])[:, 2]), (
        'the surface did not move, so a stale tensor would have been indistinguishable anyway')


# =====================================================================================
# (viii) the pin - a tick is a values patch, and the derivative rides it
# =====================================================================================

def test_the_derivative_survives_a_patch_market_round_trip():
    """The grid is not differentiated - it is PINNED - so everything above rests on the pin holding
    across the verb a tick actually arrives through.

    `Surface_Type` and `Grid_Tolerance` are the fingerprint `pinned_grid` reads back, and both are
    structural, so an identity round trip through `market_patch` / `patch_market` has to leave a
    block the next bootstrap still recognises. If it did not, the tick would refine a NEW grid and
    the Jacobian would be of a different surface with no error raised anywhere.
    """
    config = world(True)
    before = jacobian(*(bootstrapped(True)[0], config.params['Price Factors']))
    context = derivus.Context()
    context.current_cfg = config
    context.patch_market(context.market_patch())

    family, written = bootstrapped(True, factors=config.params['Price Factors'])
    assert np.array_equal(sorted_surface(written)[:, :2],
                          sorted_surface(config.params['Price Factors'])[:, :2])
    assert np.array_equal(jacobian(family, written), before)
    assert BLOCK in family.quote_leaves


def test_a_ticked_quote_moves_the_surface_and_keeps_the_wiring():
    """The tick path end to end: new quotes onto the grid the last bootstrap wrote. The coordinates
    are byte for byte the old ones, the vols move, and the leaves come back wired to the NEW quote
    values - which is what makes a quote delta a thing a streaming job can report per tick."""
    _, factors = bootstrapped(True)
    first = jacobian(*bootstrapped(True, factors=copy.deepcopy(factors)))

    points = quotes({(1.0, 0.25, 'BF'): 0.005})
    family, ticked = bootstrapped(True, points, factors)
    assert np.array_equal(sorted_surface(ticked)[:, :2],
                          sorted_surface(bootstrapped(True)[1])[:, :2])
    assert float(family.quote_leaves[BLOCK][1].detach()[quote_index('BF', 1.0, 0.25)]) == \
        points[quote_index('BF', 1.0, 0.25)]['Quoted_Market_Value']
    assert not np.array_equal(jacobian(family, ticked), first), (
        'the tick did not reach the derivative')


# =====================================================================================
# (ix) the declaration
# =====================================================================================

def test_the_switch_is_declared_and_the_store_offers_it():
    """The family IS its declarations - the store is emitted from them, so what a UI offers and
    what the engine selects work by cannot disagree. The engine's fallback is held to the declared
    default by `test_a_declared_default_is_the_default_the_engine_falls_back_to`, which reads both
    off the source; what is left to say here is that the field reached the store at all."""
    block = mapping['MarketPrices']['types']['FXVolPrices']
    assert block['Quote_Sensitivity']['value'] == 'No'
    assert block['Quote_Sensitivity']['values'] == ['Yes', 'No']

    absent = prices(True)
    del absent[BLOCK]['instrument']['Quote_Sensitivity']
    family = FXVolSurfaceParameters({}, DEVICE, DTYPE)
    family.bootstrap({'Base_Date': BASE}, {}, {}, INTERP, absent, {})
    assert not family.calibrated and not family.quote_leaves
