"""`FXVolPrices` - an FX smile quoted as ATM / risk reversal / butterfly, and the surface it builds.

An FX vol surface ticks in as DELTA quotes: one ATM vol per expiry and, per delta pillar, the risk
reversal and the butterfly around it. The surface the pricers read is a log-moneyness one, and the
conversion between them - the strangle pair, then the delta-to-strike solve - is the `Malz` branch
`Factor2D` has always carried. This family is that conversion moved to BOOTSTRAP TIME, and these
gates are the three claims that move rests on.

  TRANSPARENT   the family produces the same surface, and prices the same deal to the same number,
                as the hand-authored delta surface it replaces. That is what makes it a move rather
                than a second implementation - `test_the_family_is_transparent_to_a_priced_deal`.
  PINNED        the delta solve does not evaluate a smile at prescribed strikes; it REFINES a
                log-moneyness grid until interpolating between the nodes resolves the smile. Run at
                factor-construction time that makes a moved node - and therefore a recompile - a
                possible consequence of any vol tick. Built here, once, the grid is part of the
                written factor and a tick moves values on it. Both branches of that fork are gated.
  VECTORIZED    the conversion was a Python loop over an x-grid with a scalar `brentq` inside it.
                It is now one bisection over the whole grid. The pre-vectorisation code is kept
                BELOW, in this file, as the oracle - it is what the new code has to agree with, and
                it has no business in the engine. The oracle covers the grid and the vols on it;
                the ATM LABEL it reads them off is a third statement that no quote set this family
                builds can test, and it is held to the pre-refactor numbers by hand -
                `test_the_plus_half_label_is_the_atm_vol_when_a_smile_quotes_both`.

WHAT THE ROUND TRIP IS. The quotes are computed OFF a known smile rather than copied from a vendor,
so the strangle algebra is exactly invertible and the recovered surface is the one the smile
produces, to floating point. Nothing here asserts a level.

THE CONVENTIONS ARE GATED AS MATHS, NOT AS STRINGS. `Delta_Type`, `Premium_Adjusted` and
`ATM_Convention` each offer exactly one value because the solve implements exactly one. A test
asserting those strings would pass on any implementation; `test_the_declared_conventions_are_the`
`_ones_the_solve_implements` instead finds the strike whose PREMIUM-ADJUSTED FORWARD delta is the
pillar and requires the surface to carry the pillar's vol there - and requires the same statement
under the UNADJUSTED delta to fail, so the gate is known to tell the two conventions apart.

Run: ``pytest tests/test_fx_vol_prices.py -q``
"""
import copy
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.optimize import brentq
from scipy.stats import norm
from sortedcontainers import sorteddict

from derivus import run_baseval, utils
from derivus.bootstrappers import FXVolSurfaceParameters
from derivus.config import Config, ModelParams
from derivus.instruments import construct_instrument
from derivus.riskfactors import Factor2D, construct_factor
from derivus.schema import mapping

DEVICE = torch.device('cpu')
INTERP = ModelParams()
BASE = pd.Timestamp('2026-06-30')
DTYPE = torch.float64

#: The smile the quotes are computed off. One ATM vol per expiry and a (risk reversal, butterfly)
#: pair per pillar, shaped the way a USDZAR smile is: a positive risk reversal (calls on the dollar
#: bid), a butterfly that widens into the wings, and a term structure that flattens the skew out.
EXPIRIES = (0.0833, 0.25, 1.0, 3.0)
PILLARS = (0.25, 0.10)
ATM = {0.0833: 0.1450, 0.25: 0.1520, 1.0: 0.1610, 3.0: 0.1680}
RISK_REVERSAL = {(0.25, 0.25): 0.0210, (0.25, 0.10): 0.0375, (1.0, 0.25): 0.0190,
                 (1.0, 0.10): 0.0340, (3.0, 0.25): 0.0150, (3.0, 0.10): 0.0270,
                 (0.0833, 0.25): 0.0230, (0.0833, 0.10): 0.0410}
BUTTERFLY = {(0.0833, 0.25): 0.0032, (0.0833, 0.10): 0.0105, (0.25, 0.25): 0.0035,
             (0.25, 0.10): 0.0112, (1.0, 0.25): 0.0041, (1.0, 0.10): 0.0128,
             (3.0, 0.25): 0.0046, (3.0, 0.10): 0.0140}

SEEN = pd.Timestamp('2026-06-30 09:15:00')
LATEST = pd.Timestamp('2026-06-30 16:30:00')

#: The tolerance every block here is authored at. Named because it is now half of what the pin is
#: keyed on - a surface is this plan's only if it was refined against these expiries AT this number.
TOLERANCE = 1e-4


# =====================================================================================
# the pre-vectorisation Malz solver, verbatim, as the parity oracle. NOT engine code.
# =====================================================================================

def oracle_prepare_skew(skew, T):
    d = np.asarray(skew["delta"], dtype=float)
    v = np.asarray(skew["vol"], dtype=float)
    p05_idx = np.where(np.isclose(d, 0.5))[0]
    m05_idx = np.where(np.isclose(d, -0.5))[0]
    if p05_idx.size:
        sigma_atm = float(v[p05_idx[0]])
    elif m05_idx.size:
        sigma_atm = float(v[m05_idx[0]])
    else:
        raise ValueError("Malz skew missing ATM label")
    delta_atm = 0.5 * np.exp(-0.5 * sigma_atm * sigma_atm * T)
    new_d, new_v = [], []
    for di, vi in zip(d, v):
        if np.isclose(di, 0.5):
            new_d.append(+delta_atm)
            new_v.append(vi)
        elif np.isclose(di, -0.5):
            new_d.append(-delta_atm)
            new_v.append(vi)
        else:
            new_d.append(di)
            new_v.append(vi)
    d, v = np.asarray(new_d, dtype=float), np.asarray(new_v, dtype=float)
    if not np.any(d < 0) or not np.any(np.isclose(d, -delta_atm)):
        d, v = np.append(d, -delta_atm), np.append(v, sigma_atm)
    if not np.any(d > 0) or not np.any(np.isclose(d, +delta_atm)):
        d, v = np.append(d, +delta_atm), np.append(v, sigma_atm)
    idx = np.argsort(d)
    d, v = d[idx], v[idx]
    return {"d_put": d[d <= 0.0], "v_put": v[d <= 0.0], "d_call": d[d >= 0.0],
            "v_call": v[d >= 0.0], "sigma_atm": sigma_atm, "delta_atm": delta_atm}


def oracle_sigma_exact(skew, T, x):
    """One point, one `brentq`: the shape the vectorized solve replaces."""
    sqrtT = np.sqrt(T)
    d_put, v_put = skew["d_put"], skew["v_put"]
    d_call, v_call = skew["d_call"], skew["v_call"]
    is_call = (x <= 0.5 * skew['sigma_atm'] * skew['sigma_atm'] * T)

    def sigma_skew(delta):
        return np.interp(delta, d_call, v_call) if is_call else np.interp(delta, d_put, v_put)

    def delta_of(sigma):
        d2 = (x + 0.5 * sigma * sigma * T) / (sigma * sqrtT) - sigma * sqrtT
        k_over_f = np.exp(-x)
        return k_over_f * norm.cdf(d2) if is_call else -k_over_f * norm.cdf(-d2)

    def f(delta):
        return delta_of(sigma_skew(delta)) - delta

    a, b = ((max(0.0, d_call.min()), d_call.max()) if is_call
            else (d_put.min(), min(0.0, d_put.max())))
    fa, fb = f(a), f(b)
    delta_star = (a if abs(fa) < abs(fb) else b) if fa * fb > 0 else brentq(f, a, b)
    return sigma_skew(delta_star)


def oracle_surface(delta_surface, expiries, tol=1e-4):
    """The whole pre-vectorisation path: `[[x, T, vol], ...]`, on the grid it refines."""
    skews = {}
    for T in expiries:
        row = delta_surface[delta_surface[:, 1] == T]
        idx = np.argsort(row[:, 0])
        skews[T] = oracle_prepare_skew(
            {'delta': row[idx, 0], 'vol': row[idx, 2].clip(min=1e-4)}, T)

    x_grid = np.array([-0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5], dtype=float)
    w_table = {T: sorteddict.SortedDict(
        {x: oracle_sigma_exact(skews[T], T, x) ** 2 * T for x in x_grid}) for T in expiries}
    for T in expiries:
        w, x_nodes = w_table[T], x_grid.copy()
        while True:
            candidates = np.array([[0.5 * (xl + xr), oracle_sigma_exact(
                skews[T], T, 0.5 * (xl + xr))] for xl, xr in zip(x_nodes[:-1], x_nodes[1:])])
            x_sorted, y_sorted = zip(*w.items())
            err = np.abs(np.sqrt(np.interp(candidates[:, 0], x_sorted, y_sorted) / T)
                         - candidates[:, 1])
            if err.max() <= tol:
                break
            updates = candidates[np.where(err > tol)[0]]
            x_nodes = np.union1d(x_nodes, updates[:, 0])
            w.update({k: T * v * v for k, v in updates})

    return np.array(sorted([[x, T, v] for T in expiries for x, v in zip(
        np.array(w_table[T].keys()), np.sqrt(np.array(w_table[T].values()) / T))]))


# =====================================================================================
# the world: one smile, the quotes computed off it, and the two ways to reach a surface
# =====================================================================================

def delta_surface():
    """The known smile as a `(delta, expiry, vol)` surface - what a hand-authored block carries."""
    rows = [[0.5, T, ATM[T]] for T in EXPIRIES]
    for T in EXPIRIES:
        for pillar in PILLARS:
            rr, bf = RISK_REVERSAL[(T, pillar)], BUTTERFLY[(T, pillar)]
            rows.append([pillar, T, ATM[T] + bf + 0.5 * rr])
            rows.append([-pillar, T, ATM[T] + bf - 0.5 * rr])
    return np.array(sorted(rows))


def quotes(bumps=None):
    """The `Points` table - the same smile stated the way a broker states it.

    `bumps` moves individual quotes by `{(expiry, pillar, quote_type): shift}`, which is what the
    pinning gates tick with.
    """
    bumps = bumps or {}
    points = []
    for T in EXPIRIES:
        points.append({'Use': 'Yes', 'Expiry': T, 'Pillar': 0.0, 'Quote_Type': 'ATM',
                       'Quoted_Market_Value': ATM[T] + bumps.get((T, 0.0, 'ATM'), 0.0),
                       'Timestamp': SEEN})
        for pillar in PILLARS:
            for quote_type, table in (('RR', RISK_REVERSAL), ('BF', BUTTERFLY)):
                points.append({
                    'Use': 'Yes', 'Expiry': T, 'Pillar': pillar, 'Quote_Type': quote_type,
                    'Quoted_Market_Value': table[(T, pillar)] + bumps.get(
                        (T, pillar, quote_type), 0.0),
                    'Timestamp': LATEST if (T, pillar, quote_type) == (
                        EXPIRIES[-1], PILLARS[-1], 'BF') else SEEN})
    return points


def market_prices(points=None, **overrides):
    block = {'Currency': 'ZAR', 'Delta_Type': 'Forward', 'Premium_Adjusted': 'Yes',
             'ATM_Convention': 'Delta_Neutral_Straddle', 'Grid_Tolerance': TOLERANCE,
             'Points': quotes() if points is None else points}
    block.update(overrides)
    return {'FXVolPrices.USD.ZAR': {'instrument': block, 'Children': []}}


def bootstrapped(prices=None, price_factors=None):
    """Run the family the way `Config.bootstrap` runs it. `price_factors` is what it pins to."""
    factors = {} if price_factors is None else price_factors
    FXVolSurfaceParameters({}, DEVICE, DTYPE).bootstrap(
        {'Base_Date': BASE}, {}, factors, INTERP, market_prices() if prices is None else prices, {})
    return factors


def authored_block():
    """The hand-authored `Malz` block: a delta surface the factor solves when it is CONSTRUCTED."""
    return {'Property_Aliases': None, 'Surface_Type': 'Malz',
            'Moneyness_Rule': 'Sticky_Moneyness', 'Currency': 'ZAR',
            'Delta_Surface': utils.Curve([], delta_surface().tolist())}


def surface_of(block):
    """The log-moneyness surface a block ends up with once the factor is constructed."""
    return construct_factor(utils.Factor('FXVol', ('USD', 'ZAR')),
                            {'FXVol.USD.ZAR': block}, INTERP).param['Surface'].array


def skews():
    return Factor2D.malz_skews(delta_surface(), np.array(EXPIRIES))


# =====================================================================================
# (i) the round trip - quotes computed off a known smile, and the surface recovered
# =====================================================================================

def test_the_quotes_restate_the_smile_they_were_computed_off():
    """The inner half of the round trip: the strangle pair is exactly invertible.

    `vol(call) = ATM + BF + RR/2` and `vol(put) = ATM + BF - RR/2` recover the two wing vols from
    the two numbers a broker quotes, so a quote set built from a smile rebuilds that smile to
    floating point - and everything downstream can then be compared node for node rather than to a
    tolerance somebody chose.
    """
    rebuilt = FXVolSurfaceParameters.smile(quotes())
    expected = delta_surface()
    assert rebuilt.shape == expected.shape
    assert np.array_equal(rebuilt[:, :2], expected[:, :2]), 'the quotes land on different deltas'
    error = np.abs(rebuilt[:, 2] - expected[:, 2]).max()
    assert error < 1e-16, 'the smile came back to {:.3g}, not to floating point'.format(error)


def test_the_bootstrap_recovers_the_surface_the_smile_produces():
    """The outer half: the family's surface IS the one the hand-authored delta surface builds.

    Same nodes - `np.array_equal` on both coordinate columns, not a tolerance - and the same vols
    on them, EXACTLY. The two paths run the same solver over the same numbers; the only difference
    is where. So the criterion is equality rather than a tolerance somebody picked, and any
    divergence at all is a finding rather than a rounding.
    """
    built = bootstrapped()['FXVol.USD.ZAR']['Surface'].array
    authored = surface_of(authored_block())

    assert built.shape == authored.shape, 'the two paths disagree on how many nodes there are'
    assert np.array_equal(built[:, :2], authored[:, :2]), 'the two paths built different grids'
    error = np.abs(built[:, 2] - authored[:, 2]).max()
    assert error == 0.0, 'the two paths are not the same computation: {:.3g}'.format(error)


def test_a_moved_quote_breaks_the_round_trip():
    """MUTATE the answer. A tenth of a vol on one butterfly has to break the comparison above.

    Bootstrapped onto the UNMOVED grid, so the two surfaces are node for node comparable. Left to
    rebuild, the moved quotes refine a grid of their own (95 nodes against 97) and the comparison
    would have nothing to line up - which is the pinning gate's subject and not this one's.
    """
    authored = surface_of(authored_block())
    moved = bootstrapped(market_prices(quotes({(1.0, 0.25, 'BF'): 0.001})), bootstrapped())
    built = moved['FXVol.USD.ZAR']['Surface'].array

    assert np.array_equal(built[:, :2], authored[:, :2]), 'the pin did not hold the grid'
    error = np.abs(built[:, 2] - authored[:, 2]).max()
    assert error > 1e-4, 'a tenth of a vol did not reach the surface ({:.3g})'.format(error)


def test_a_held_out_quote_leaves_the_smile():
    """`Use` is what lets a quote be dropped without being deleted, as it is on the curve family.
    Dropping a butterfly drops the pillar's curvature, so its wings collapse onto the risk reversal
    around the ATM vol - which is a smile with fewer nodes, not a smile with a hole in it."""
    held = quotes()
    for point in held:
        if (point['Expiry'], point['Pillar'], point['Quote_Type']) == (1.0, 0.10, 'BF'):
            point['Use'] = 'No'
    smile = FXVolSurfaceParameters.smile([p for p in held if p['Use'] == 'Yes'])

    row = smile[(smile[:, 1] == 1.0) & (smile[:, 0] == 0.10)]
    assert row[0, 2] == pytest.approx(ATM[1.0] + 0.5 * RISK_REVERSAL[(1.0, 0.10)], rel=1e-15)
    assert smile.shape == delta_surface().shape, 'holding a quote out changed the node COUNT'


# =====================================================================================
# (ii) the vectorized conversion against the loop it replaced
# =====================================================================================

def test_the_vectorized_conversion_agrees_with_the_loop_it_replaced():
    """The engine's one implementation against the one it replaced, on the same inputs.

    The GRID is bit-identical: refinement is a sequence of comparisons against `tol` and both
    implementations make the same ones. The VOLS are not, and cannot be - `brentq` stops at its own
    2e-12 xtol while the bisection closes to the bracket's machine precision - so the criterion is
    that they agree to better than the root find they are replacing was ever accurate to.
    """
    reference = oracle_surface(delta_surface(), np.array(EXPIRIES))
    prepared = skews()
    built = np.array(sorted(Factor2D.malz_surface(
        prepared, Factor2D.malz_grid(prepared, 1e-4))))

    assert built.shape == reference.shape, 'the two grids have different node counts'
    assert np.array_equal(built[:, :2], reference[:, :2]), 'the refined grid moved'
    error = np.abs(built[:, 2] - reference[:, 2]).max()
    assert error < 2e-12, 'the vols differ by {:.3g}, beyond brentq\'s own xtol'.format(error)
    logging.info('vectorized vs loop: %d nodes, max |dsigma| %.3g', len(built), error)


@pytest.mark.parametrize('tol', [1e-3, 1e-4, 1e-5])
def test_the_grid_the_two_implementations_refine_is_the_same_at_every_tolerance(tol):
    """The grid parity is the load-bearing half - a surface that agreed on vols while refining to
    different nodes would be a different PLAN. Walked across tolerances because the node count is
    what changes with it: 41, 97 and 226 nodes here."""
    prepared = skews()
    reference = oracle_surface(delta_surface(), np.array(EXPIRIES), tol)
    built = np.array(sorted(Factor2D.malz_surface(prepared, Factor2D.malz_grid(prepared, tol))))
    assert np.array_equal(built[:, :2], reference[:, :2]), (
        'tol {:g}: {} nodes against {}'.format(tol, len(built), len(reference)))


def test_the_bisection_has_converged_at_the_iteration_count_the_engine_uses():
    """64 halvings is not a number chosen for comfort: it is where the bracket reaches double
    precision. 30 has not converged (1e-11), 64 and 200 are the same answer to the last bit, so
    the count could be cut only by giving up accuracy the oracle gate above then measures."""
    prepared, grid = skews(), Factor2D.malz_grid(skews())

    def solved(iterations):
        return np.concatenate([Factor2D.malz_sigma(prepared[T], T, nodes, iterations)
                               for T, nodes in grid.items()])

    assert np.abs(solved(64) - solved(200)).max() == 0.0, '64 halvings is not converged'
    assert np.abs(solved(30) - solved(200)).max() > 1e-13, (
        '30 halvings agrees too - this is not measuring convergence')


def test_the_loop_is_gone_from_the_engine():
    """The oracle lives in this file and nowhere else. A scalar root find inside a per-point loop
    is what the vectorisation removed, and a `brentq` back in `riskfactors` would be it returning
    without any of these gates noticing - they compare two implementations, not their shapes.

    Read off the AST rather than the text, because both names are still in the docstring that
    explains why they are gone, and a gate that grepped for prose would be pinning the prose."""
    import ast
    import inspect

    from derivus import riskfactors
    tree = ast.parse(inspect.getsource(riskfactors))
    bound = {alias.asname or alias.name.split('.')[0] for node in ast.walk(tree)
             if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
    used = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    for name in ('brentq', 'sorteddict', 'norm'):
        assert name not in bound | used, '{} is back in the engine'.format(name)


# =====================================================================================
# the ATM label, when a smile quotes BOTH sides of it
# =====================================================================================

#: A hand-authored smile carrying both +-0.5 labels at DIFFERENT vols, at one expiry. No quote set
#: this family builds can produce it - `smile` writes the ATM vol at +0.5 alone - and it is the
#: only input that can tell the two readings of the label apart, which is why the reading was free
#: to regress. A wide, deliberately asymmetric skew so the ATM vol moves the whole grid.
BOTH_LABELS = [[-0.5, 1.0, 0.19], [-0.25, 1.0, 0.15], [0.25, 1.0, 0.175], [0.5, 1.0, 0.16]]

#: What the PRE-REFACTOR solver builds from it: `git show a5f0cad:derivus/riskfactors.py`, run
#: through `Factor2D({'Surface_Type': 'Malz', 'Delta_Surface': ...})` and read off the written
#: `Surface`. Reading -0.5 instead gives 68 nodes and 0.161445 - a different plan AND a different
#: number, which is what makes this reproducible rather than a preference.
PREFERRED_NODES, PREFERRED_ATM = 70, 0.16159077102548716


def test_the_plus_half_label_is_the_atm_vol_when_a_smile_quotes_both():
    """+0.5 wins. The ATM vol is not just another node - it sets `delta_atm`, which is where the
    label is placed and therefore where every refined node lands, so which of the two labels is
    read is a question about the GRID and not only about a level.

    Held to the pre-refactor numbers rather than to internal consistency, because the vectorisation
    was meant to move this computation and not to change it. The vols are `approx` and the node
    count is exact for the reason `test_the_vectorized_conversion_agrees_with_the_loop_it_replaced`
    gives: the grid is a sequence of comparisons both implementations make identically, and the
    bisection closes tighter than the `brentq` it replaced.
    """
    prepared = Factor2D.malz_skews(np.array(sorted(BOTH_LABELS)), np.array([1.0]))
    assert prepared[1.0]['sigma_atm'] == 0.16, 'the -0.5 label was read as the ATM vol'

    surface = np.array(sorted(Factor2D.malz_surface(prepared, Factor2D.malz_grid(prepared))))
    assert len(surface) == PREFERRED_NODES, (
        'the +0.5 ATM vol refines {} nodes, not {}'.format(PREFERRED_NODES, len(surface)))
    at_the_money = float(surface[surface[:, 0] == 0.0][0, 2])
    assert at_the_money == pytest.approx(PREFERRED_ATM, abs=2e-12), (
        'at the money: {!r} against the pre-refactor {!r}'.format(at_the_money, PREFERRED_ATM))

    # the -0.5 vol is not discarded - it is the put wing's ATM node, so both quotes reach the
    # surface; only one of them can say where the delta-neutral straddle sits
    put = prepared[1.0]
    assert put['v_put'][np.argmin(put['d_put'])] == 0.19


# =====================================================================================
# (iii) the pinned grid - both branches
# =====================================================================================

def test_a_ticked_quote_moves_the_values_and_not_the_grid():
    """THE POINT OF THE FAMILY. A vol tick must not move a tenor grid.

    The surface is built once, then the same block is bootstrapped again over it with one butterfly
    a full vol wider. The coordinates come back byte for byte identical - `np.array_equal` on both
    of them - and the vols move, which is a `bind='value'` patch and not a new plan.
    """
    factors = bootstrapped()
    first = factors['FXVol.USD.ZAR']['Surface'].array.copy()

    ticked = bootstrapped(market_prices(quotes({(1.0, 0.25, 'BF'): 0.01})), factors)
    second = ticked['FXVol.USD.ZAR']['Surface'].array

    assert np.array_equal(first[:, 0], second[:, 0]), 'the moneyness grid moved on a tick'
    assert np.array_equal(first[:, 1], second[:, 1]), 'the expiry grid moved on a tick'
    moved = np.abs(first[:, 2] - second[:, 2]).max()
    assert moved > 1e-3, 'nothing moved at all - the tick did not reach the surface ({:.3g})'.format(moved)


def test_the_same_tick_built_from_scratch_refines_a_different_grid():
    """The other branch, and the reason the first one is a claim rather than a tautology.

    The refinement follows the numbers: the SAME ticked quotes, bootstrapped into empty price
    factors, resolve to a grid with different nodes in it. So the grid the tick above kept was kept
    on purpose - it is the plan's, not something the refinement would have produced anyway.
    """
    ticked = market_prices(quotes({(1.0, 0.25, 'BF'): 0.01}))
    pinned = bootstrapped(ticked, bootstrapped())['FXVol.USD.ZAR']['Surface'].array
    rebuilt = bootstrapped(ticked)['FXVol.USD.ZAR']['Surface'].array

    assert not (pinned.shape == rebuilt.shape and np.array_equal(pinned[:, :2], rebuilt[:, :2])), (
        'the rebuild produced the pinned grid, so this world cannot tell the two branches apart')
    # the two surfaces still describe the same smile - the grids differ, the function does not
    for T in EXPIRIES:
        on_pinned, on_rebuilt = pinned[pinned[:, 1] == T], rebuilt[rebuilt[:, 1] == T]
        shared = np.intersect1d(on_pinned[:, 0], on_rebuilt[:, 0])
        assert np.abs(on_pinned[np.isin(on_pinned[:, 0], shared), 2] -
                      on_rebuilt[np.isin(on_rebuilt[:, 0], shared), 2]).max() < 1e-12


def test_a_quote_set_that_drops_an_expiry_is_not_pinned_to_the_old_grid():
    """A grid refined against expiries the quotes no longer have is not this surface's plan. It is
    rebuilt rather than stretched - the alternative is an expiry with no nodes at all.

    The block handed to `pinned_grid` is one this family actually WROTE, with only its surface
    swapped, so the expiry set is the only thing that can be refusing. Handing it a bare
    `{'Surface': ...}` instead - which is what this did - exits through the SUBTYPE guard on the
    missing `Surface_Type`, and the expiry guard could then be deleted with every gate still green.
    """
    factors = bootstrapped()
    written = factors['FXVol.USD.ZAR']

    fewer = [point for point in quotes() if point['Expiry'] != 3.0]
    rebuilt = bootstrapped(market_prices(fewer), factors)['FXVol.USD.ZAR']['Surface'].array
    assert np.array_equal(np.unique(rebuilt[:, 1]), np.array(sorted(set(EXPIRIES) - {3.0})))

    shrunk = dict(written, Surface=utils.Curve([], rebuilt.tolist()))
    assert shrunk['Surface_Type'] == 'Malz' and shrunk['Grid_Tolerance'] == TOLERANCE, (
        'the fixture exits through some other guard than the expiry one')
    assert FXVolSurfaceParameters.pinned_grid(shrunk, np.array(EXPIRIES), TOLERANCE) is None


def test_a_quote_set_that_gains_an_expiry_is_not_pinned_either():
    """The GROWING direction, which is the one that says what the guard is for.

    Shrinking is survivable - a pinned grid would carry nodes for an expiry nobody asked about.
    Growing is not: the new expiry selects NO nodes out of the old surface, and an empty node list
    is not a coarse grid, it is nothing to interpolate through. Without the guard this raises
    `ValueError: array of sample points is empty` out of `malz_error`, which is the mutation
    evidence and the reason the guard is not merely tidy.
    """
    factors = bootstrapped()
    written = factors['FXVol.USD.ZAR']
    grown_expiries = np.array(sorted(EXPIRIES + (5.0,)))

    more = quotes() + [
        {'Use': 'Yes', 'Expiry': 5.0, 'Pillar': pillar, 'Quote_Type': quote_type,
         'Quoted_Market_Value': value, 'Timestamp': SEEN}
        for pillar, quote_type, value in [(0.0, 'ATM', 0.1720), (0.25, 'RR', 0.0130),
                                          (0.25, 'BF', 0.0050)]]
    grown = bootstrapped(market_prices(more), factors)['FXVol.USD.ZAR']['Surface'].array

    assert np.array_equal(np.unique(grown[:, 1]), grown_expiries), 'the 5y expiry has no nodes'
    assert FXVolSurfaceParameters.pinned_grid(written, grown_expiries, TOLERANCE) is None


def test_a_surface_on_another_subtype_is_not_a_grid_to_pin_to():
    """The nodes of an `Explicit` surface are S/K and the nodes of this one are log(F/K), so a
    number is not a grid on its own - pinning across the two would read one coordinate system's
    nodes as the other's and produce a surface that is wrong everywhere but the money."""
    explicit = {'Property_Aliases': None, 'Surface_Type': 'Explicit',
                'Moneyness_Rule': 'Sticky_Moneyness',
                'Surface': utils.Curve([], [[m, T, ATM[T]] for T in EXPIRIES
                                            for m in (0.9, 1.0, 1.1)])}
    assert FXVolSurfaceParameters.pinned_grid(explicit, np.array(EXPIRIES), TOLERANCE) is None

    rebuilt = bootstrapped(market_prices(), {'FXVol.USD.ZAR': explicit})['FXVol.USD.ZAR']
    assert rebuilt['Surface_Type'] == 'Malz'
    assert np.array_equal(rebuilt['Surface'].array,
                          bootstrapped()['FXVol.USD.ZAR']['Surface'].array)


def test_a_changed_grid_tolerance_is_structural_and_a_repeated_one_keeps_the_pin(caplog):
    """`Grid_Tolerance` is the one knob over the thing the pin owns, so it is part of what the pin
    is keyed on. Honouring it when a grid is built and ignoring it when one is reused would make
    the knob unreachable on every run after the first - a field nothing reads, on the second tick.

    So it is written onto the surface it built and compared on the way back in: the same number
    reuses the grid, a different one is a different plan and refines. Both branches here, and the
    per-expiry INFO line has to say which tolerance the nodes it is reporting were built at -
    otherwise a resolved error is a number with no scale beside it.
    """
    factors = bootstrapped()
    coarse = factors['FXVol.USD.ZAR']['Surface'].array.copy()
    assert factors['FXVol.USD.ZAR']['Grid_Tolerance'] == TOLERANCE

    # SAME tolerance: the pin holds, coordinates byte for byte
    again = bootstrapped(market_prices(), factors)['FXVol.USD.ZAR']['Surface'].array
    assert np.array_equal(again[:, :2], coarse[:, :2]), 'the same tolerance broke the pin'

    # TIGHTER: a new plan, refined over the surface already sitting there
    with caplog.at_level(logging.INFO):
        finer = bootstrapped(
            market_prices(**{'Grid_Tolerance': 1e-6}), factors)['FXVol.USD.ZAR']
    assert len(finer['Surface'].array) > len(coarse), (
        'a tighter tolerance reused the coarse grid: {} nodes against {}'.format(
            len(finer['Surface'].array), len(coarse)))
    assert finer['Grid_Tolerance'] == 1e-6, 'the surface does not say what it was built at'
    assert 'refined' in caplog.text and 'built at 1e-06' in caplog.text, caplog.text
    logging.info('tolerance %g -> %d nodes, %g -> %d nodes',
                 TOLERANCE, len(coarse), 1e-6, len(finer['Surface'].array))


def test_the_pinned_grid_survives_construction_of_the_factor():
    """The pin is worth nothing if `Factor2D` re-solves the block it is handed.

    A bootstrapped block carries no delta surface, so `solves_delta_surface` is false and the
    surface reaches `get_moneyness` exactly as written. The hand-authored block is the control: it
    DOES carry one, and it does re-solve.
    """
    written = bootstrapped()['FXVol.USD.ZAR']
    factor = construct_factor(utils.Factor('FXVol', ('USD', 'ZAR')),
                              {'FXVol.USD.ZAR': written}, INTERP)
    assert not factor.solves_delta_surface()
    assert np.array_equal(factor.param['Surface'].array, written['Surface'].array)

    authored = Factor2D(authored_block())
    assert authored.solves_delta_surface()


# =====================================================================================
# (iv) the family is transparent to a priced deal
# =====================================================================================

FX_OPTION = {
    'Object': 'FXOptionDeal', 'Reference': 'FXOPT', 'Currency': 'ZAR',
    'Underlying_Currency': 'USD', 'Underlying_Amount': 1_000_000.0, 'Strike_Price': 19.5,
    'Buy_Sell': 'Buy', 'Option_Type': 'Call', 'Option_Style': 'European',
    'Settlement_Style': 'Cash', 'Option_On_Forward': 'No', 'FX_Volatility': 'USD.ZAR',
    'Discount_Rate': 'ZAR', 'Expiry_Date': BASE + pd.Timedelta(days=365)}


def option_cfg(vol_block):
    """The one option, off whichever `FXVol` block it is handed."""
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
                             'Curve': utils.Curve([], [[0.0, 0.041], [5.0, 0.039]])},
        'FXVol.USD.ZAR': vol_block}
    config.deals = {'Attributes': {'Reference': 'test', 'Tag_Titles': ''},
                    'Deals': {'Children': [{'Instrument': construct_instrument(FX_OPTION, {})}]},
                    'Calculation': {'Base_Date': BASE, 'Currency': 'ZAR'}}
    return config


def priced(vol_block):
    _, out = run_baseval(option_cfg(vol_block), prec=DTYPE,
                         overrides={'Greeks': 'No', 'Random_Seed': 1, 'MCMC_Simulations': 1})
    rows = out['Results']['mtm']
    return float(rows[rows['Parent'] == 'root']['Value'].sum())


def test_the_family_is_transparent_to_a_priced_deal():
    """End to end, through `run_baseval`: a one-year USDZAR call off the bootstrapped surface and
    off the hand-authored delta surface it replaces, and the two are the same number.

    This is the statement the whole family rests on. Everything above compares surfaces; this
    compares what a pricer does with them, which is what a moved node or a mis-labelled moneyness
    convention would show up in and a surface comparison would not. The two are EQUAL, not close:
    the surfaces are bit-identical and the valuation is deterministic.
    """
    built = priced(bootstrapped()['FXVol.USD.ZAR'])
    authored = priced(authored_block())
    assert built == authored, (
        'bootstrapped {:.6f} against authored {:.6f}'.format(built, authored))
    assert abs(built) > 1.0, 'the option is worthless - this compares two zeros'


def test_the_priced_deal_notices_a_moved_quote():
    """MUTATE the market. The gate above is only a statement about the family if the price it
    compares is one the quotes can move: a vol wider on the 25 delta butterfly is worth R19,700 of
    a R983,000 option here - two per cent, against an equality asserted to the last bit."""
    base = priced(bootstrapped()['FXVol.USD.ZAR'])
    ticked = priced(bootstrapped(
        market_prices(quotes({(1.0, 0.25, 'BF'): 0.01})))['FXVol.USD.ZAR'])
    assert abs(ticked - base) > 1e-6 * abs(base), (
        'a vol on the butterfly did not reach the price: {:.6f} against {:.6f}'.format(
            ticked, base))


# =====================================================================================
# the declared conventions, gated as maths
# =====================================================================================

def call_strike(sigma, T, pillar, premium_adjusted):
    """The log-moneyness x = log(F/K) of the OTM call whose forward delta is `pillar`.

    Bracketed on the OTM side, `[-2, sigma^2 T / 2]`, because the premium-adjusted delta is NOT
    monotone in the strike - it rises from zero, peaks near the money and falls back to zero, so a
    bracket spanning the peak holds two roots or none and the one wanted is the outer one.
    """
    def residual(x):
        d = (x + (0.5 if not premium_adjusted else -0.5) * sigma * sigma * T) / (sigma * np.sqrt(T))
        return (np.exp(-x) if premium_adjusted else 1.0) * norm.cdf(d) - pillar
    return brentq(residual, -2.0, 0.5 * sigma * sigma * T)


#: The pillar the convention gates read. The smile FLAT-EXTRAPOLATES beyond its widest quoted
#: delta, so at the 10 delta wing every convention lands on the same clamped vol and nothing can
#: be told apart there - measured, and the reason this is a constant rather than `PILLARS`.
DISCRIMINATING_PILLAR = 0.25


def test_the_declared_conventions_are_the_ones_the_solve_implements():
    """`Delta_Type: Forward`, `Premium_Adjusted: Yes`, `ATM_Convention: Delta_Neutral_Straddle` -
    each offering exactly one value, because the solve implements exactly one of each.

    Asserted as maths rather than as strings. Find the strike whose PREMIUM-ADJUSTED FORWARD delta
    is the pillar, at the vol that pillar is quoted at, and the surface has to carry that vol
    there. The ATM leg is the same statement at the delta-neutral straddle strike,
    K = F exp(-sigma^2 T / 2).
    """
    prepared = skews()
    for T in EXPIRIES:
        skew = prepared[T]
        atm_x = 0.5 * ATM[T] * ATM[T] * T
        assert float(Factor2D.malz_sigma(skew, T, atm_x)) == pytest.approx(ATM[T], rel=1e-12), (
            'the ATM quote is not at the delta-neutral straddle strike for T={}'.format(T))

        for pillar in PILLARS:
            vol = ATM[T] + BUTTERFLY[(T, pillar)] + 0.5 * RISK_REVERSAL[(T, pillar)]
            x = call_strike(vol, T, pillar, True)
            assert float(Factor2D.malz_sigma(skew, T, x)) == pytest.approx(vol, rel=1e-10), (
                'the {} delta call vol is not at the premium-adjusted strike for T={}'.format(
                    pillar, T))


def test_the_convention_gate_can_tell_the_two_delta_conventions_apart():
    """The mutation that makes the gate above mean something: repeat it with the UNADJUSTED forward
    delta N(d1), which is the other convention an FX smile is quoted in, and it has to FAIL.

    A gate that passed under both would be measuring that the surface interpolates its own nodes.
    The 10 delta wing is where that would happen and is excluded by measurement rather than by
    taste - see `DISCRIMINATING_PILLAR`.
    """
    prepared, pillar = skews(), DISCRIMINATING_PILLAR
    misses = []
    for T in EXPIRIES:
        vol = ATM[T] + BUTTERFLY[(T, pillar)] + 0.5 * RISK_REVERSAL[(T, pillar)]
        unadjusted = call_strike(vol, T, pillar, False)
        misses.append(abs(float(Factor2D.malz_sigma(prepared[T], T, unadjusted)) - vol))
    assert min(misses) > 1e-4, (
        'the unadjusted delta lands on the same vol - the gate cannot tell the conventions apart')
    logging.info('unadjusted-delta miss per expiry: %s',
                 ['{:.3g}'.format(m) for m in misses])


def test_the_widest_wing_cannot_tell_the_conventions_apart():
    """The negative result behind `DISCRIMINATING_PILLAR`, recorded rather than quietly dropped.

    Beyond the widest quoted delta the smile is flat, so the unadjusted 10 delta strike - which is
    further out than the premium-adjusted one - reads the SAME clamped vol. Every convention agrees
    there, which is a property of the extrapolation and not of the conventions."""
    prepared, pillar = skews(), 0.10
    for T in EXPIRIES:
        vol = ATM[T] + BUTTERFLY[(T, pillar)] + 0.5 * RISK_REVERSAL[(T, pillar)]
        for premium_adjusted in (True, False):
            assert float(Factor2D.malz_sigma(
                prepared[T], T, call_strike(vol, T, pillar, premium_adjusted))) == vol


# =====================================================================================
# timestamps, the store, and the declared defaults
# =====================================================================================

def test_the_surface_carries_the_latest_contributing_timestamp():
    """A quote row carries when it was seen; the surface carries the LATEST of the rows that built
    it, which is the surface's own as-of. Held-out rows do not contribute - a stale quote nobody
    used cannot make the surface look fresh, and one nobody used cannot make it look stale."""
    assert bootstrapped()['FXVol.USD.ZAR']['Quote_Timestamp'] == LATEST

    without = [dict(point, Use='No' if point['Timestamp'] == LATEST else 'Yes')
               for point in quotes()]
    assert bootstrapped(market_prices(without))['FXVol.USD.ZAR']['Quote_Timestamp'] == SEEN

    undated = [dict(point, Timestamp='') for point in quotes()]
    assert bootstrapped(market_prices(undated))['FXVol.USD.ZAR']['Quote_Timestamp'] == ''


def test_an_intraday_timestamp_survives_the_json_round_trip():
    """The encoder writes a midnight stamp as the plain date it always did - old files re-encode
    byte-stable - and a non-midnight one in ISO form with its time, which the decoder always
    parsed. Before this, `%Y-%m-%d` destroyed the hour on save and the 09:15 and 16:30 snapshots
    reloaded as the same market event; the replay-identity claim was only true per DAY."""
    import json
    from derivus.config import Config, CustomJsonEncoder

    def round_trip(stamp):
        encoded = json.dumps(stamp, cls=CustomJsonEncoder)
        return encoded, json.loads(encoded, object_hook=lambda d: (
            pd.Timestamp(d['.Timestamp']) if '.Timestamp' in d else d))

    # a date stays a date, byte-stable against the old format
    encoded, back = round_trip(BASE)
    assert encoded == '{".Timestamp": "2026-06-30"}' and back == BASE

    # intraday and sub-second stamps keep their time exactly
    for stamp in (SEEN, LATEST, pd.Timestamp('2026-06-30 16:30:00.500123')):
        _, back = round_trip(stamp)
        assert back == stamp, f'{stamp} reloaded as {back}'

    # an OLD file's plain-date string still parses to the same midnight stamp
    assert pd.Timestamp('2026-06-30') == BASE


def test_two_intraday_snapshots_are_two_market_events():
    """The point of the capability: two quote sets differing ONLY in the hour of their stamps must
    hash as different values AFTER A SAVE - the save is where the old encoder truncated to the
    day and the 09:15 and 16:30 snapshots reloaded as one market event. In-memory they always
    differed; the round trip is what this gate holds open."""
    import json
    from derivus import content_hash
    from derivus.config import CustomJsonEncoder

    def hash_after_save(stamp):
        factors = {}
        FXVolSurfaceParameters({}, DEVICE, DTYPE).bootstrap(
            {'Base_Date': BASE}, {}, factors, INTERP,
            market_prices([dict(p, Timestamp=stamp) for p in quotes()]), {})
        # the encoded STRING is what a save persists; hashing its parse is the F1 measurement
        return content_hash(json.loads(json.dumps(factors, cls=CustomJsonEncoder)))

    morning, close = hash_after_save(SEEN), hash_after_save(LATEST)
    assert morning != close, 'the 09:15 and 16:30 snapshots hashed identically after a save'
    assert hash_after_save(SEEN) == morning, 'the hash is not a function of the quotes'


def test_the_timestamp_is_a_value_and_not_a_plan():
    """A stamp that recompiled the plan on every tick would be worse than no stamp at all, so it is
    declared `bind='value'` and travels in the values patch beside the vols it dates."""
    from derivus import schema

    written = bootstrapped()['FXVol.USD.ZAR']
    structural, values = schema.partition_factor('FXVol', written)
    assert values['Quote_Timestamp'] == LATEST
    assert structural['Quote_Timestamp'] is None
    # and the surface splits the way every shaped field does: coordinates structural, vols value
    assert structural['Surface'].array.shape[1] == 2
    assert len(values['Surface']) == len(written['Surface'].array)
    assert schema.apply_values('FXVol', structural, values)['Surface'].array.tolist() == \
        written['Surface'].array.tolist()


def test_pricing_never_reads_the_timestamp():
    """Declared as reported-only, so a surface stamped a week later has to price identically."""
    written = bootstrapped()['FXVol.USD.ZAR']
    stale = dict(written, Quote_Timestamp=LATEST - pd.Timedelta(days=7))
    assert priced(stale) == priced(written)


def test_the_family_is_in_the_market_prices_store():
    """The family IS its declarations: the store is emitted from them, so what a UI offers and what
    the engine selects work by cannot disagree."""
    types = mapping['MarketPrices']['types']
    assert FXVolSurfaceParameters.market_factor_type in types
    block = types['FXVolPrices']
    assert set(block) == {'Currency', 'Delta_Type', 'Premium_Adjusted', 'ATM_Convention',
                          'Grid_Tolerance', 'Quote_Sensitivity', 'Points'}
    assert block['Points']['col_names'] == [
        'Use', 'Expiry', 'Pillar', 'Quote_Type', 'Quoted_Market_Value', 'Timestamp']
    # one value per convention, because the solve implements one of each
    for key, value in [('Delta_Type', 'Forward'), ('Premium_Adjusted', 'Yes'),
                       ('ATM_Convention', 'Delta_Neutral_Straddle')]:
        assert block[key]['values'] == [value], '{} offers a convention nothing implements'.format(key)


def test_the_declared_grid_tolerance_is_the_one_the_engine_falls_back_to():
    """`test_a_declared_default_is_the_default_the_engine_falls_back_to` reads the fallback off the
    AST and only sees a literal, and this one is read off `Factor2D` so that the un-bootstrapped
    path and the family share a number rather than copying it. So the seam is gated here instead:
    the declaration, the family's fallback and the factor's own default are one value."""
    declared = mapping['MarketPrices']['types']['FXVolPrices']['Grid_Tolerance']['value']
    assert declared == Factor2D.malz_tol

    no_knob = market_prices()
    del no_knob['FXVolPrices.USD.ZAR']['instrument']['Grid_Tolerance']
    assert np.array_equal(bootstrapped(no_knob)['FXVol.USD.ZAR']['Surface'].array,
                          bootstrapped()['FXVol.USD.ZAR']['Surface'].array)


def test_a_coarser_tolerance_builds_a_coarser_grid():
    """The knob is the one lever over the thing this family owns - the plan's grid - so it has to
    move it. A declared field nothing honours is the defect the whole store exists to prevent."""
    coarse = bootstrapped(market_prices(**{'Grid_Tolerance': 1e-2}))['FXVol.USD.ZAR']
    fine = bootstrapped(market_prices(**{'Grid_Tolerance': 1e-5}))['FXVol.USD.ZAR']
    assert len(coarse['Surface'].array) < len(bootstrapped()['FXVol.USD.ZAR']['Surface'].array)
    assert len(fine['Surface'].array) > len(bootstrapped()['FXVol.USD.ZAR']['Surface'].array)


def test_a_grid_tolerance_outside_its_declared_bounds_is_refused():
    """The knob has a floor, it is DECLARED, and the family reads its own declaration back.

    `bounds=` is a widget hint everywhere else in the schema - nothing loads it, nothing checks it -
    and here it is the difference between a grid and a hang. Refinement halves an interval until
    the midpoint's vol error falls under the tolerance, so at 0.0 no midpoint ever qualifies: one
    expiry reaches 7.6 million nodes after 21 passes and is still doubling. 1e-8 is 4599 nodes for
    the four-expiry smile above, which is a large plan but a plan, so that is where the floor is.

    The check is the bootstrapper's because there is nowhere else: `Config.bootstrap` hands a
    family its block unexamined, which is right - the family owns what its numbers mean.
    """
    declared = mapping['MarketPrices']['types']['FXVolPrices']['Grid_Tolerance']
    assert declared['widget'] == 'BoundedFloat'
    assert (declared['min'], declared['max']) == FXVolSurfaceParameters.grid_tolerance_bounds

    for refused in (0.0, -TOLERANCE, 2.0):
        with pytest.raises(ValueError, match='Grid_Tolerance'):
            bootstrapped(market_prices(**{'Grid_Tolerance': refused}))

    # and the declared ceiling is accepted rather than merely published - the seed grid passes it
    ceiling = bootstrapped(market_prices(
        **{'Grid_Tolerance': declared['max']}))['FXVol.USD.ZAR']['Surface'].array
    assert len(ceiling) == len(EXPIRIES) * len(Factor2D.malz_seed_grid)


def test_a_pillar_of_a_half_is_refused_because_it_is_the_atm_label():
    """0.5 is the one number a `Pillar` cannot be. It is the LABEL the ATM vol is carried at, so a
    wing quoted there does not land beside the ATM row - it lands ON it.

    Left to run it is silent and it wins: measured, the 50 delta call built from a 1 vol risk
    reversal and a 20 bp butterfly puts 0.168 at (0.5, 1y) next to the ATM quote's 0.161, and the
    solve reads 0.168 as the ATM vol - so the surface disagrees with the quote it was built from,
    at the money, with nothing raised. A 50 delta RR/BF pair is authored as the ATM row by
    convention, which is the only place it can be said once.
    """
    collides = quotes() + [
        {'Use': 'Yes', 'Expiry': 1.0, 'Pillar': 0.5, 'Quote_Type': quote_type,
         'Quoted_Market_Value': value, 'Timestamp': SEEN}
        for quote_type, value in [('RR', 0.0100), ('BF', 0.0020)]]

    with pytest.raises(ValueError, match='Pillar 0.5'):
        FXVolSurfaceParameters.smile(collides)
    with pytest.raises(ValueError, match='Pillar 0.5') as raised:
        bootstrapped(market_prices(collides))
    assert 'BF/RR' in str(raised.value) and 'expiry 1' in str(raised.value), (
        'the refusal does not name the row: {}'.format(raised.value))


def test_config_bootstrap_drives_the_family_and_finds_the_surface_it_wrote(caplog):
    """End to end through `Config.bootstrap`, which is how a job reaches this - and the gate on
    `price_factor_type`: the check for a bootstrapper that silently did nothing looks for a
    `<ClassName>.*` price factor and this family writes an ordinary `FXVol`. The check only LOGS,
    so the log is what has to be asserted."""
    config = Config(base_currency='ZAR')
    config.params['System Parameters']['Base_Date'] = BASE
    config.params['Market Prices'] = market_prices()
    config.params['Bootstrapper Configuration'] = {'FXVolSurfaceParameters': {}}
    with caplog.at_level(logging.ERROR):
        config.bootstrap()

    assert 'wrote no' not in caplog.text, caplog.text
    assert 'FXVol.USD.ZAR' in config.params['Price Factors']
