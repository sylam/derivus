"""Is the HW2F swaption calibration one differentiable node, and does saying so cost nothing?

Stage B of the contract [Quote Sensitivities](../docs_src/developer/quote_sensitivities.md) puts
around the curve bootstrap, now around the Monte Carlo calibration. Stage A closed the quote side of
the residual and left the solve at the scipy boundary; `LeastSquaresSolve` is that boundary crossed.

    the forward   theta* is BIT-IDENTICAL through the wrapper and around it, quotes on and off -
                  `np.array_equal` on the 23-vector and on the saved `Price Factors` block.
    the seed      the solve is REPRODUCIBLE for the first time. Basin hopping used to step off the
                  process global; `Random_Seed` declares it, and a different one moves theta*.
    the backward  the implicit function theorem at the STATIONARITY fixed point `J'r = 0` under
                  Gauss-Newton, checked against an exact identity and against what the predicted
                  step does to the bumped problem's own residual.

WHY THERE IS NO FINITE-DIFFERENCE RE-SOLVE HERE, which was the reference this stage was briefed to
use. The calibration has 23 parameters and this block quotes four swaptions, so `J` is 4 x 23 and
the solution is a 19-dimensional MANIFOLD, not a point. A re-solve at a bumped quote lands somewhere
else on it, and `(theta*(q+h) - theta*(q-h))/2h` diverges as `1/h` - a fixed displacement of about
0.1 in theta divided by a shrinking bump. Measured, cold-started and warm-started at theta* alike;
`test_a_finite_difference_re_solve_is_not_a_reference_for_this` holds that finding in place. The
gates below are the two references that ARE well posed on a rank-deficient problem.

Run: ``pytest tests/test_swaption_calibration_solve.py -q``
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pytest
import torch

from derivus import bootstrappers, riskfactors, utils
from derivus.config import ModelParams
from test_swaption_quote_graph import (BASE, BENCHMARKS, BLOCK, CCY, CURVE, VOL, definitions,
                                       price_factors)

PARAMS = 'HullWhite2FactorModelParameters.' + CURVE
#: The declared defaults, restated so a gate that reads one of them says which number it is holding.
RCOND, STATIONARITY, SEED = 1e-8, 1e-3, 5120


def block(connect=True, bumps=(), seed=SEED):
    return {'Swaption_Volatility': VOL, 'Generate_Instruments': 'No', 'Random_Seed': seed,
            'Quote_Sensitivity': 'Yes' if connect else 'No',
            'Instrument_Definitions': definitions(bumps)}


def calibration(connect=True, bumps=(), seed=SEED, optimizers=('basin', 'leastsq')):
    """The calibration as an operand, built the way `bootstrap` builds it.

    `optimizers` selects the chain, and a SHORTER chain is how a gate stops the solve early without
    reaching inside anything: basin hopping alone leaves theta* at the seed on this fixture.
    """
    factors, interp = price_factors(), ModelParams()
    boot = bootstrappers.construct_bootstrapper('HullWhite2FactorModelParameters', {}, torch.float32)
    rate = utils.check_rate_name(BLOCK)
    ir_factor = utils.Factor('InterestRate', rate[1:])
    surface = riskfactors.construct_factor(utils.Factor('InterestYieldVol', (VOL,)), factors, interp)
    surface.delta = 0.0
    ir_curve = riskfactors.construct_factor(ir_factor, factors, interp)
    surface.set_premiums(None, ir_curve.get_currency())
    implied_obj, process, vol_tenors = boot.implied_process(CCY, factors, {}, ir_curve, rate)
    instrument = block(connect, bumps, seed)
    mtm_dates = set([BASE + x['Start'] for x in instrument['Instrument_Definitions']])
    time_grid = utils.TimeGrid(mtm_dates, mtm_dates, mtm_dates)
    time_grid.set_base_date(BASE, delta=(10, vol_tenors * utils.DAYS_IN_YEAR))
    loss_fn, optims, implied_var, swaps, _ = boot.calc_loss(
        {'instrument': instrument, 'Children': []}, BASE, time_grid, process, implied_obj,
        ir_factor, surface)
    return bootstrappers.SwaptionCalibration(
        CURVE, loss_fn, implied_var, [o for o in optims if o[0] in optimizers], process, swaps)


def bootstrapped(connect=True, seed=SEED):
    """One whole `bootstrap` call, and the `Price Factors` block it wrote."""
    factors = price_factors()
    boot = bootstrappers.construct_bootstrapper('HullWhite2FactorModelParameters', {}, torch.float32)
    boot.bootstrap({'Base_Date': BASE, 'Base_Currency': CCY}, {}, factors, ModelParams(),
                   {BLOCK: {'instrument': block(connect, seed=seed), 'Children': []}}, {})
    param = factors[PARAMS]
    return np.concatenate([[param['Alpha_1']], [param['Alpha_2']], [param['Correlation']],
                           param['Sigma_1'].array[:, 1], param['Sigma_2'].array[:, 1]])


SOLVED = {}


def solved(connect=True, wrapped=True, seed=SEED, optimizers=('basin', 'leastsq')):
    """`(calibration, theta*)`, cached - the solve is the slow part and every gate wants one.

    `wrapped` is the whole point of gate (a): `False` calls `SwaptionCalibration.solve` directly and
    `True` goes through the autograd Function, whose forward is that call and nothing else.
    """
    key = (connect, wrapped, seed, optimizers)
    if key not in SOLVED:
        cal = calibration(connect, seed=seed, optimizers=optimizers)
        theta = bootstrappers.LeastSquaresSolve.apply(
            cal, RCOND, STATIONARITY, *cal.quotes) if wrapped else cal.solve()
        SOLVED[key] = (cal, theta)
    return SOLVED[key]


def jacobian(cal, theta):
    """`(J, dr/dq, r)` at theta*, in float64 - the three things the backward contracts."""
    x = theta.detach().requires_grad_(True)
    residual = cal(x)
    J = torch.stack([torch.autograd.grad(residual[i], x, retain_graph=True)[0]
                     for i in range(residual.numel())]).double()
    drdq = torch.stack([torch.stack(torch.autograd.grad(residual, cal.quotes, e, retain_graph=True))
                        for e in torch.eye(len(residual), device=residual.device)]).double()
    return J, drdq, residual.detach().double()


QUOTE_JACOBIAN = {}


def quote_jacobian():
    """`dtheta/dq` of the default solve, 23 x quotes, read one ROW at a time OUT OF THE FUNCTION'S
    OWN BACKWARD.

    A cotangent IS a row, so the whole matrix costs one backward per parameter - which is the point
    rather than the cost. Rebuilding it from `J` and `dr/dq` in the gate would restate the
    contraction being gated beside it, and the mandated sign-flip mutation then passes every gate
    that used it - measured, on the first cut of this file. Cached: 23 backward passes re-price the
    benchmark set 23 times.
    """
    cal, theta = solved()
    if not QUOTE_JACOBIAN:
        QUOTE_JACOBIAN['dtheta'] = torch.stack([
            torch.stack(torch.autograd.grad(theta, cal.quotes, grad_outputs=e, retain_graph=True))
            for e in torch.eye(theta.numel(), dtype=theta.dtype, device=theta.device)]).double()
    return QUOTE_JACOBIAN['dtheta']


def test_the_solve_is_bit_identical_through_the_wrapper(connect=True):
    """The forward pass cannot move, and `np.array_equal` is the only honest way to say so.

    Two independent ways it could: the wrapper could run a different chain, or enabling the quote
    side could perturb the residual the chain walks. Both are checked, and the second one twice -
    on the 23-vector and on the `Price Factors` block a job actually consumes, because
    `save_params` is between them and reads the parameter dict back by name.
    """
    assert np.array_equal(solved(True, True)[1].detach().cpu().numpy(),
                          solved(True, False)[1].detach().cpu().numpy())
    assert np.array_equal(solved(False, True)[1].cpu().numpy(),
                          solved(True, True)[1].detach().cpu().numpy())
    assert np.array_equal(bootstrapped(True), bootstrapped(False))


def test_the_solve_is_reproducible_and_the_declared_seed_is_what_makes_it_so():
    """The first deterministic HW2F calibration - there is no earlier baseline to preserve.

    Basin hopping's step taker drew from `np.random`, so theta* was a function of whatever ran
    before it in the same interpreter: on this fixture the ambient seed moves it 0.93 absolute. That
    is not a regression this gate protects against, it is the behaviour being replaced, so what is
    asserted is the property that did not exist - two runs agree, and a different declared seed
    disagrees.

    BOTH SEEDS ARE RUN TWICE and that is not belt and braces. At the declared default basin hopping
    accepts nothing on this fixture and hands its seed straight to least squares, so a step taker
    left on the process global is INVISIBLE there - measured: putting `np.random.uniform` back
    leaves the default-seed gate green. Seed 7 is one where the search does move, and it is the only
    thing standing between that mutation and a passing suite.
    """
    assert np.array_equal(bootstrapped(seed=SEED), bootstrapped(seed=SEED))
    assert np.array_equal(bootstrapped(seed=7), bootstrapped(seed=7))
    assert not np.array_equal(bootstrapped(seed=SEED), bootstrapped(seed=7))


def test_the_quote_side_costs_no_edge_when_the_block_declines_it():
    """`Quote_Sensitivity` No means no quote leaf, so `apply` records no edge and the node is a
    pass-through - the wrapper is on the path of every calibration, not only a differentiated one."""
    cal, theta = solved(False, True)
    assert cal.quotes == ()
    assert not theta.requires_grad
    assert solved(True, True)[1].requires_grad


def test_the_gauss_newton_matrix_is_rank_deficient_and_the_solve_says_which_directions():
    """23 parameters against four quoted swaptions, so `J'J` has rank four and a 19-dimensional
    null space. That is the problem, not a defect: those are combinations the quote set does not
    identify, and `dtheta/dq` there is the MINIMUM-NORM representative - which is exactly what a
    pseudo-inverse returns and what a ridge would quietly replace with something else.

    The declared cutoff has to separate the two, and the gate says by how much: the smallest REAL
    eigenvalue is four orders above `Jacobian_Rcond` and the largest spurious one eight below.

    The minimum-norm half is asserted RELATIVE, and the tolerance is the precision seam rather than
    slack. The linear algebra is float64 throughout, but `-Jw` has to be cast back to the residual's
    own float32 before it can be a `grad_outputs`, so what the backward reports carries float32
    resolution however exactly it was computed: 4.4e-8 absolute against columns of norm 4 here.
    """
    cal, theta = solved()
    J, _, _ = jacobian(cal, theta)
    eigenvalues = torch.linalg.eigvalsh(J.t() @ J).cpu().numpy()
    real, spurious = eigenvalues[-len(BENCHMARKS):], eigenvalues[:-len(BENCHMARKS)]
    assert (real / eigenvalues[-1] > 1e2 * RCOND).all(), real
    assert (np.abs(spurious) / eigenvalues[-1] < 1e-2 * RCOND).all(), spurious
    # the minimum-norm claim, on what the BACKWARD reported: every column lies in the row space
    projector = torch.linalg.pinv(J.t() @ J, hermitian=True, rtol=RCOND) @ J.t() @ J
    dtheta = quote_jacobian()
    assert float((projector @ dtheta - dtheta).abs().max()) < 1e-6 * float(dtheta.abs().max())


def test_the_quote_jacobian_satisfies_the_benchmark_self_delta_identity():
    """The identity that needs no bump at all, and the one a sign flip dies on.

    `J` has full ROW rank - four benchmarks, 23 parameters - so `J (J'J)^+ J' = I` on the residual
    space, and a cotangent chosen as `v = J'u` contracts through the whole backward to

        dL/dq = -(dr/dq)' J (J'J)^+ J' u  =  -(dr/dq)' u

    with no linear algebra surviving. So this reads the pseudo-inverse, the transpose and the sign
    against a closed form: any of the three wrong and the two sides stop agreeing to 1e-10. It is
    the swaption-shaped version of increment 1's benchmark self-delta matrix, and it holds for the
    same reason - a benchmark is held at its own market number, so its quote is the only one that
    moves it.
    """
    cal, theta = solved()
    J, drdq, _ = jacobian(cal, theta)
    u = torch.ones(len(BENCHMARKS), dtype=torch.float64, device=J.device)
    reported = torch.stack(torch.autograd.grad(
        theta, cal.quotes, grad_outputs=(J.t() @ u).to(theta.dtype), retain_graph=True)).double()
    assert torch.allclose(reported, -(drdq.t() @ u), rtol=1e-10, atol=1e-12), (
        reported.cpu().numpy(), -(drdq.t() @ u).cpu().numpy())


@pytest.mark.parametrize('bump', [0.5, 0.2, 0.1])
def test_the_predicted_step_re_solves_the_bumped_calibration(bump):
    """The reference a rank-deficient problem CAN be held to: does `dtheta/dq` do its job?

    Move one quote by `bump` percent and the residual at the old theta* jumps by four orders. Step
    the parameters by `dtheta/dq . bump` and it comes back down by two to four more - and the
    recovery sharpens as the bump shrinks, which is the second-order remainder behaving. Nothing
    here re-solves, so the 19-dimensional manifold the solver wanders on never enters.

    The WRONG-SIGNED step is checked in the same breath and is the loud half: it does not merely
    fail to help, it lands the residual roughly three times further out than doing nothing. So the
    mandated sign-flip mutation fails this gate on every row and every bump size.
    """
    cal, theta = solved()
    dtheta = quote_jacobian()
    for row, (_, _, quoted) in enumerate(BENCHMARKS):
        if not quoted:
            continue
        moved = calibration(bumps=((row, bump),))
        step = dtheta[:, row] * (bump / 100.0)
        stale = float(moved(theta.detach()).detach().double().norm())
        stepped = float(moved((theta.detach().double() + step).to(theta.dtype)).detach().double().norm())
        reversed_ = float(moved((theta.detach().double() - step).to(theta.dtype)).detach().double().norm())
        assert stepped < 0.1 * stale, (row, bump, stepped, stale)
        assert reversed_ > 2.0 * stale, (row, bump, reversed_, stale)


def test_the_backward_refuses_a_theta_that_is_not_stationary():
    """`solve` accepts whatever the chain returned - which can be the seed, if nothing beat it - and
    the implicit function theorem holds only where `J'r` vanishes. So a solve stopped early has to
    RAISE rather than report a plausible Jacobian of nothing.

    Stopped early here means the chain without its least-squares leg: basin hopping alone leaves
    this fixture at its seed, where `||J'r||` is ten orders of magnitude above what the full chain
    reaches. The full chain's own norm is asserted under the declared tolerance in the same gate, so
    the two halves cannot both be satisfied by a check that always fires or never does.
    """
    cal, theta = solved()
    J, _, residual = jacobian(cal, theta)
    assert float((J.t() @ residual).norm()) < STATIONARITY
    torch.autograd.grad(theta, cal.quotes, grad_outputs=torch.ones_like(theta), retain_graph=True)

    early_cal, early = solved(optimizers=('basin',))
    assert np.array_equal(early.detach().cpu().numpy(), early_cal.optimizers[0][1])
    with pytest.raises(Exception, match='not stationary'):
        torch.autograd.grad(early, early_cal.quotes, grad_outputs=torch.ones_like(early))


def test_the_backward_reads_the_graph_and_not_the_accumulated_grad():
    """`.grad` on a quote leaf is the sum over the optimizer's WHOLE PATH, not the derivative at its
    answer - basin hopping calls `backward()` on every evaluation it makes and nothing clears the
    quote side between them. On this fixture that accumulation is six orders out and one entry is
    NaN, so a backward harvesting `.grad` would report confident nonsense.

    The gate is the gap: what the Function returns is finite, order one, and nothing like what is
    standing on the leaves when it is asked.
    """
    cal, theta = solved()
    accumulated = np.array([np.nan if q.grad is None else float(q.grad) for q in cal.quotes])
    reported = np.array([float(g) for g in torch.autograd.grad(
        theta, cal.quotes, grad_outputs=torch.ones_like(theta), retain_graph=True)])
    assert np.isfinite(reported).all() and (np.abs(reported) < 1e3).all(), reported
    assert not np.isfinite(accumulated).all() or (np.abs(accumulated - reported) > 1.0).any()


def test_a_second_differentiation_raises_rather_than_reporting_one():
    """Gauss-Newton drops the term the second derivative would need, so a quote-space Hessian off
    this node would be the curvature of a different problem. `CalibrationSolve` declines it in prose
    and this declines it in code."""
    cal, theta = solved()
    with pytest.raises(Exception, match='create_graph'):
        torch.autograd.grad(theta, cal.quotes, grad_outputs=torch.ones_like(theta),
                            create_graph=True, retain_graph=True)


def test_a_finite_difference_re_solve_is_not_a_reference_for_this():
    """The honest negative result, pinned so that it is a known property rather than a surprise.

    Bump a quote, re-run the same deterministic solve, difference theta*: the answer is dominated by
    the component the quotes do NOT identify. The unidentified part of that difference is an order
    of magnitude larger than the identified part, so the quotient is measuring where the optimizer
    happened to stop on a 19-dimensional flat, and it grows as the bump shrinks. If this ever fails,
    the solve has started returning a function of its quotes and the comparison becomes available.
    """
    cal, theta = solved()
    J, _, _ = jacobian(cal, theta)
    projector = (torch.linalg.pinv(J.t() @ J, hermitian=True, rtol=RCOND) @ J.t() @ J).cpu().numpy()
    bump = 0.2
    moved = [calibration(bumps=((0, shift),)).solve().double().cpu().numpy()
             for shift in (bump, -bump)]
    difference = (moved[0] - moved[1]) / (2.0 * bump / 100.0)
    identified = projector @ difference
    assert np.linalg.norm(difference - identified) > 4.0 * np.linalg.norm(identified)
