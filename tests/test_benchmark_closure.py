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

#: The quotes the four benchmarks below are authored at, in percent.
QUOTES = [4.32, 4.55, 4.20, 4.05]


#: A deposit, an FRA, a par swap and an OIS swap - the four shapes the family has to price, and
#: between them every pricer the residual reaches: the fixed leg, the single-reset floating leg,
#: fixed-against-floating in one deal, and the OIS-compounded leg under a container.
def benchmark_nodes(quotes=None):
    return [quote_node(block, {}) for block in (
        deposit('DEPO_6M', CCY, OIS, 6, (quotes or QUOTES)[0]),
        fra('FRA_3X6', CCY, PROJ, OIS, 3, 6, (quotes or QUOTES)[1]),
        par_swap('IRS_2Y', CCY, PROJ, OIS, 2, (quotes or QUOTES)[2]),
        ois_swap('OIS_2Y', CCY, OIS, 24, (quotes or QUOTES)[3]))]


def closure(interpolation=None, nodes=None, curves=None, quotes=None):
    """The compiled benchmark set. `quotes` additionally puts the quote vector on the tape - the
    bumped set it is measured against is the same four instruments one percent higher."""
    interp = ModelParams()
    if interpolation:
        interp.append('InterestRate', (), interpolation)
    curves = curves or {OIS: (TENORS, OIS_NODES), PROJ: (TENORS, PROJ_NODES)}
    return BenchmarkInstruments(
        nodes if nodes is not None else benchmark_nodes(quotes),
        market(CCY, curves, OIS), interp, BASE, CCY, {},
        [f for f in (OIS_FACTOR, PROJ_FACTOR) if f.name[0] in curves],
        torch.device('cpu'), quotes=quotes,
        bumped_nodes=None if quotes is None else benchmark_nodes([q + 1.0 for q in quotes]))


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


def quote_jacobian(bm):
    """d(PV_i)/d(quote_j), one backward pass per benchmark."""
    pv = bm(theta(bm))
    return np.array([np.zeros(len(QUOTES)) if g is None else g.numpy() for g in (
        torch.autograd.grad(pv[i], bm.quotes, retain_graph=True, allow_unused=True)[0]
        for i in range(len(pv)))])


def test_the_residual_is_differentiable_in_its_quotes():
    """The other half of the tape. `TensorSchedule.merged` copies the schedule across with
    `new_tensor` - notionals, accruals, margins and the FIXED RATE - so until the tensor half
    carried an overlay a quote could not be differentiated through at all.

    The residual is AFFINE in its quotes (a deposit's coupons, an FRA's strike and a fixed leg's
    rate are each linear in one), so the exact derivative is a SECANT and not an approximation of
    one - which makes this an identity rather than a tolerance, and it gates the pricer as well as
    the schedule: the overlay could be right and `pv_fixed_cashflows` still fold it in wrongly.
    """
    aad = quote_jacobian(closure(quotes=QUOTES))
    lo = closure(quotes=None, nodes=benchmark_nodes(QUOTES))
    hi = closure(quotes=None, nodes=benchmark_nodes([q + 1.0 for q in QUOTES]))
    secant = np.diag(hi(theta(hi)).detach().numpy() - lo(theta(lo)).detach().numpy())
    assert np.abs(aad - secant).max() / np.abs(secant).max() < 1e-12, (
        'AAD and the exact secant disagree:\n{}\n{}'.format(aad, secant))


def test_a_quote_reaches_its_own_benchmark_and_no_other():
    """The zeros carry as much as the numbers. A quote is a property of ONE instrument, so
    d(PV)/d(quote) is diagonal - an off-diagonal entry would mean a quote had leaked into a
    benchmark that does not carry it, and the calibration Jacobian would then be solving a system
    nobody authored."""
    jac = quote_jacobian(closure(quotes=QUOTES))
    assert (np.abs(np.diag(jac)) > 0).all(), 'a benchmark that ignores its own quote: {}'.format(jac)
    assert np.abs(jac - np.diag(np.diag(jac))).max() == 0.0, (
        'a quote reached a benchmark that does not carry it:\n{}'.format(jac))


def test_putting_the_quotes_on_the_tape_does_not_move_a_single_pv():
    """Bit-identity, which is what makes the overlay safe to leave on. The splice is
    `base + (q - q.detach()) * slope` - the boundary correction's shape - so it is worth exactly
    zero in the forward pass and `np.array_equal` rather than a tolerance is the right assertion."""
    with_quotes = closure(quotes=QUOTES)
    plain = closure()
    assert np.array_equal(with_quotes(theta(with_quotes)).detach().numpy(),
                          plain(theta(plain)).detach().numpy())


def test_a_schedule_has_a_birthday_and_its_edits_have_a_deadline():
    """The unit gate on `TensorSchedule`, in the order the lifecycle runs.

    An overlay attached after the tensor half was taken used to be silently dropped, and a plain
    caller could be handed a graph it never asked for - the `t_Buffer` shape of trap, from the same
    cause: the copy was minted by whichever call happened to be first. `bind` is that event made
    explicit, so the protocol is CHECKABLE at both ends - a touch before it raises, an edit after it
    raises - and `dual` and `merged` are one accessor over one copy rather than two memos colliding
    under one key.
    """
    one = torch.ones([1, 1], dtype=torch.float64)
    schedule = utils.TensorSchedule([[1.0, 2.0], [3.0, 4.0]], [[0.0], [0.0]])
    with pytest.raises(utils.ScheduleLifecycleError, match='TensorSchedule.* never bound'):
        schedule.merged(one)

    plain = schedule.bind(one).merged(one).tn
    assert plain.grad_fn is None and not plain.requires_grad
    # one copy: the two accessors cannot be served different halves of it
    assert schedule.dual().tn.data_ptr() == plain.data_ptr()
    assert schedule.merged(one, 1).tn.tolist() == plain[1:].tolist()

    column = torch.tensor([7.0, 8.0], dtype=torch.float64, requires_grad=True)
    with pytest.raises(utils.ScheduleLifecycleError, match='must run before bind'):
        schedule.carry({1: column})

    carried = schedule.reopen().carry({1: column}).bind(one).merged(one).tn
    assert carried[:, 1].tolist() == [7.0, 8.0], 'the overlay was not spliced in: {}'.format(carried)
    assert carried[:, 0].tolist() == [1.0, 3.0], 'the splice moved a column it does not own'
    assert torch.autograd.grad(carried.sum(), column)[0].tolist() == [1.0, 1.0]

    # and back: dropping the overlay has to drop the graph, not serve the spliced copy
    assert schedule.reopen().carry(None).bind(one).merged(one).tn.grad_fn is None


def test_binding_drops_what_the_last_binding_derived():
    """`derived` is run-scoped BY BEING minted with the copy it is derived from. A tensor built off
    the tensor half - `pv_fixed_cashflows`' payment vector, a reset's known values - outlives its
    copy otherwise, and a second run is then priced off the first one's numbers."""
    one = torch.ones([1, 1], dtype=torch.float64)
    schedule = utils.TensorSchedule([[1.0, 2.0], [3.0, 4.0]], [[0.0], [0.0]]).bind(one)
    schedule.derived['payments'] = schedule.dual().tn.sum()
    assert not schedule.bind(one).derived, 'a re-bound schedule kept a tensor from the old copy'


def drop_overlays(bm):
    """MUTANT: rebuild every schedule's tensor half with no overlay on it, which is exactly the
    plain `new_tensor` copy `carry` exists to replace."""
    for legs in bm.benchmarks:
        for leg in legs:
            for schedule in leg.Factor_dep.values():
                if isinstance(schedule, utils.TensorCashFlows):
                    schedule.reopen().carry(None).bind(bm.one)
    return bm


def test_a_schedule_copied_without_the_overlay_severs_the_quote():
    """MUTATE the seam this commit exists to close. Dropping the overlay is exactly what
    `new_tensor` did before it: the PV comes out the same to the last bit and the quote gradient
    silently disappears."""
    connected = closure(quotes=QUOTES)
    severed = drop_overlays(closure(quotes=QUOTES))
    assert np.array_equal(severed(theta(severed)).detach().numpy(),
                          connected(theta(connected)).detach().numpy()), (
        'the mutation has to reproduce the VALUE, or it is testing something else')
    with pytest.raises(RuntimeError):
        torch.autograd.grad(severed(theta(severed)).sum(), severed.quotes)


def test_a_priced_closure_does_not_survive_its_own_rebinding():
    """The same mutation as above, run AFTER pricing - which used to be the interesting case.

    `pv_fixed_cashflows` memoized its payment tensor in `Factor_dep`, built from the schedule's
    tensor half but outliving it, so the first evaluation froze whatever overlay was attached and
    removing the overlay afterwards changed nothing: the quote gradient survived a mutation that
    should have destroyed it. The memo lives on the schedule now and is minted by `bind` with the
    copy it is derived from, so the mutation lands whenever it is run.
    """
    bm = closure(quotes=QUOTES)
    priced = bm(theta(bm)).detach().numpy()
    drop_overlays(bm)
    assert np.array_equal(bm(theta(bm)).detach().numpy(), priced), (
        'the mutation has to reproduce the VALUE, or it is testing something else')
    with pytest.raises(RuntimeError):
        quote_jacobian(bm)


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
