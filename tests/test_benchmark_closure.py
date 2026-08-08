"""Does a benchmark instrument's t0 PV carry a graph back to the curve nodes it was priced off?

This is the first half of the market-quote sensitivity chain: quotes -> bootstrap -> calibrated
nodes -> portfolio value. The bootstrap's residual is the benchmark set's PV, and the solve needs
its Jacobian in the nodes, so before anything can be solved the PV has to be DIFFERENTIABLE in
them. It is not, on the ordinary path: every leaf the engine mints is
`torch.tensor(factor.current_value(), ...)`, a fresh tensor built out of a numpy array, so a curve
handed in as a tensor is severed the moment a factor object is constructed from it.

`BenchmarkInstruments` routes around that by writing the nodes straight into `t_Static_Buffer`,
which is where the pricers read a static curve from. The gates below are one value gate (the OIS
leg against a hand-computed closed form), one derivative gate per instrument against a float64
central difference, and two mutations - one reinstating the severance the audit found, one
reinstating the memo-table trap that would freeze the solve at its first iterate.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
import torch

from derivus import utils
from derivus.bootstrappers import BenchmarkInstruments, Benchmark_State, quote_node
from derivus.config import ModelParams

from rates_world import BASE, market, deposit, fra, par_swap, ois_swap

CCY = 'USD'
OIS = 'USD-OIS'
PROJ = 'USD-3M'
OIS_FACTOR = utils.Factor('InterestRate', (OIS,))
PROJ_FACTOR = utils.Factor('InterestRate', (PROJ,))

#: The short knot is overnight rather than zero. `Factor1D.interpolate` divides by the tenor for
#: both rate*time kinds, so a 0.0 knot makes `current_value` return NaN on a HermiteRT or LinearRT
#: curve - a live defect on the ordinary leaf-minting path, and one more reason the closure does
#: not route theta through `current_value` at all.
TENORS = [1.0 / 365.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
OIS_NODES = [0.0430, 0.0435, 0.0432, 0.0425, 0.0415, 0.0410, 0.0405, 0.0400]
PROJ_NODES = [r + 0.0020 for r in OIS_NODES]

#: A deposit, an FRA, a par swap and an OIS swap - the four shapes the family has to price, and
#: between them every pricer the residual reaches: the fixed leg, the single-reset floating leg,
#: fixed-against-floating in one deal, and the OIS-compounded leg under a container.
def benchmark_nodes():
    return [quote_node(block, {}) for block in (
        deposit('DEPO_6M', CCY, OIS, 6, 4.32),
        fra('FRA_3X6', CCY, PROJ, OIS, 3, 6, 4.55),
        par_swap('IRS_2Y', CCY, PROJ, OIS, 2, 4.20),
        ois_swap('OIS_2Y', CCY, OIS, 24, 4.05))]


def closure(interpolation=None, nodes=None, curves=None):
    interp = ModelParams()
    if interpolation:
        interp.append('InterestRate', (), interpolation)
    curves = curves or {OIS: (TENORS, OIS_NODES), PROJ: (TENORS, PROJ_NODES)}
    return BenchmarkInstruments(
        nodes if nodes is not None else benchmark_nodes(),
        market(CCY, curves, OIS), interp, BASE, CCY, {},
        [f for f in (OIS_FACTOR, PROJ_FACTOR) if f.name[0] in curves],
        torch.device('cpu'))


def theta(bm, requires_grad=False, bump=None):
    """The solved curves' authored nodes as tensors, optionally with one flattened node bumped.

    Read off the CONSTRUCTED factor rather than the literals above, because `Factor1D.get_tenor`
    rewrites `param['Curve']` on construction - so the factor's own order is the one theta is
    indexed by."""
    flat = np.concatenate([bm.factors[factor].current_value() for factor in bm.solve_for])
    if bump is not None:
        flat = flat.copy()
        flat[bump[0]] += bump[1]
    out, ofs = {}, 0
    for factor in bm.solve_for:
        size = bm.tenors[factor].size
        out[factor] = torch.tensor(flat[ofs:ofs + size], dtype=torch.float64,
                                   requires_grad=requires_grad)
        ofs += size
    return out


def aad_jacobian(bm):
    """d(PV_i)/d(node_j) from one backward pass per benchmark."""
    th = theta(bm, requires_grad=True)
    pv = bm(th)
    rows = []
    for i in range(len(pv)):
        grads = torch.autograd.grad(pv[i], list(th.values()), retain_graph=True, allow_unused=True)
        rows.append(np.concatenate([
            (torch.zeros_like(t) if g is None else g).detach().numpy()
            for g, t in zip(grads, th.values())]))
    return pv.detach().numpy(), np.array(rows)


def fd_jacobian(bm, h=1e-6):
    """The same Jacobian by float64 central difference, one column per node."""
    width = sum(bm.tenors[f].size for f in bm.solve_for)
    cols = []
    for j in range(width):
        up = bm(theta(bm, bump=(j, h))).detach().numpy()
        down = bm(theta(bm, bump=(j, -h))).detach().numpy()
        cols.append((up - down) / (2.0 * h))
    return np.array(cols).T


def agreement(aad, fd):
    return np.abs(aad - fd).max() / np.abs(aad).max()


def test_the_pv_vector_is_connected_to_every_curve_it_reads():
    """Finite, and nonzero exactly where the instrument reads. The zeros carry as much as the
    nonzeros: a pinned deposit must not reach the projection curve at all, because a bootstrapper
    solving that curve would otherwise find its own answer in the residual."""
    bm = closure()
    pv, jac = aad_jacobian(bm)
    assert np.isfinite(jac).all(), 'the Jacobian is not finite'

    n = len(TENORS)
    discount, projection = jac[:, :n], jac[:, n:]
    assert (np.abs(discount).max(axis=1) > 0).all(), 'a benchmark that does not discount'
    assert np.abs(projection[0]).max() == 0.0, (
        'the pinned deposit reached the projection curve: {}'.format(projection[0]))
    assert np.abs(projection[3]).max() == 0.0, (
        'the OIS swap reached the projection curve: {}'.format(projection[3]))
    assert (np.abs(projection[[1, 2]]).max(axis=1) > 0).all(), (
        'the FRA and the swap must forecast off the projection curve')

    # a 6m deposit cannot see the 5y node, whichever curve it is on
    assert np.abs(jac[0, TENORS.index(2.0):n]).max() == 0.0, (
        'the 6m deposit reaches past its own maturity: {}'.format(jac[0]))


@pytest.mark.parametrize('interpolation', [None, 'Hermite', 'HermiteRT'])
def test_every_benchmark_agrees_with_a_central_difference(interpolation):
    """The unit gate on the closure. Hermite is the second case because `Factor1D` precomputes its
    `(g, c)` coefficient pair from the numpy rate column at construction - constants in theta - and
    the whole claim is that the pricing path does not use them: `Interpolation.build` re-derives the
    pair from the buffer TENSOR. If it ever read the factor's, a Hermite curve would report a linear
    curve's derivative and the value would still be right."""
    bm = closure(interpolation)
    _, aad = aad_jacobian(bm)
    assert agreement(aad, fd_jacobian(bm)) < 1e-8, (
        'AAD and the central difference disagree:\n{}\n{}'.format(aad, fd_jacobian(bm)))


def test_the_ois_leg_compounds_in_arrears():
    """The value gate, and the one everything downstream rests on.

    On a flat continuously-compounded curve a daily-compounded OIS coupon telescopes exactly:
    `expm1(sum log1p(F_i d_i))` collapses the daily forwards into `D(T0)/D(T1) - 1`, so the coupon
    is `N (D(T0) - D(T1))` at its payment date and the whole floating leg is `N (D(t0) - D(T_N))`.
    That is an identity rather than an approximation, so it is asserted at 1e-9 relative and it
    catches a `1/n` reset weight, a mis-tiled fixing window and a dropped compounding branch alike.
    """
    flat = 0.04
    years = 3
    quote = 4.0
    bm = closure(nodes=[quote_node(ois_swap('OIS', CCY, OIS, 12 * years, quote), {})],
                 curves={OIS: ([0.0, 30.0], [flat, flat])})
    pv = bm(theta(bm))

    coupons = [BASE + pd.DateOffset(years=k) for k in range(years + 1)]
    df = {d: np.exp(-flat * (d - BASE).days / 365.0) for d in coupons}
    floating = 1e6 * sum(df[s] - df[e] for s, e in zip(coupons[:-1], coupons[1:]))
    fixed = -(quote / 100.0) * 1e6 * sum(
        (e - s).days / 360.0 * df[e] for s, e in zip(coupons[:-1], coupons[1:]))

    assert float(pv[0]) == pytest.approx(floating + fixed, rel=1e-9), (
        'OIS swap {} against a hand-computed {}'.format(float(pv[0]), floating + fixed))

    # MUTATE: the same swap quoted 25bp away has to move by one basis-point-value of the annuity,
    # so the agreement above is not a coincidence of two zeros
    moved = closure(nodes=[quote_node(ois_swap('OIS', CCY, OIS, 12 * years, quote + 0.25), {})],
                    curves={OIS: ([0.0, 30.0], [flat, flat])})
    annuity = 1e6 * sum((e - s).days / 360.0 * df[e] for s, e in zip(coupons[:-1], coupons[1:]))
    assert float(moved(theta(moved))[0]) == pytest.approx(
        float(pv[0]) - 0.0025 * annuity, rel=1e-9)


def test_a_leaf_minted_from_current_value_severs_the_graph():
    """MUTATE the audited seam. `_build_factor_state` builds every leaf as
    `torch.tensor(factor.current_value(), ...)`; doing that here - even from the same numbers -
    detaches the PV from theta, and `autograd.grad` says so rather than returning a wrong number.
    """
    bm = closure()
    th = theta(bm, requires_grad=True)
    severed = {factor: torch.tensor(
        bm.factors[factor].current_value(), dtype=torch.float64) for factor in bm.solve_for}
    pv = bm(severed)
    assert float(pv.sum()) == pytest.approx(float(bm(th).sum().detach()), rel=1e-12), (
        'the mutation has to reproduce the VALUE, or it is testing something else')
    with pytest.raises(RuntimeError):
        torch.autograd.grad(pv.sum(), list(th.values()))


def test_a_reused_pricing_state_answers_with_the_first_curve():
    """MUTATE the memo table. `t_Buffer` is keyed by `(stoch, Factor)` and a time hash, not by the
    tensor's identity, so a state reused across two curves hands the second call the first one's
    discount factors. A solver built on a reused state converges to whatever it started at."""
    bm = closure()
    low, high = theta(bm), theta(bm, bump=(TENORS.index(0.5), 0.01))
    assert float(bm(low)[0]) != float(bm(high)[0]), 'a fresh state must see the bump'

    shared = Benchmark_State({**bm.constants, **low}, bm.one, bm.report_currency)
    first = sum(leg.Instrument.generate(shared, bm.time_grid, leg).reshape(())
                for leg in bm.benchmarks[0])
    shared.t_Static_Buffer.update(high)
    second = sum(leg.Instrument.generate(shared, bm.time_grid, leg).reshape(())
                 for leg in bm.benchmarks[0])
    assert float(first) == float(second), (
        'the memo table is not the trap this gate claims it is - re-check the caching')
