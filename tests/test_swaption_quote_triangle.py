"""What does the swaption quote Jacobian close against, once the classic oracle is ruled out twice?

Stage D of the contract [Quote Sensitivities](../docs_src/developer/quote_sensitivities.md) puts
around the HW2F swaption calibration. Stages B and C refuted the reference this workstream was
briefed to validate against: on the four-swaption fixture `J` is 4 x 23, the solution is a
19-dimensional MANIFOLD, `dtheta*/dq` by finite difference diverges as `1/h`, and the re-bootstrapped
CVA delta reverses sign as the bump shrinks.

This file was built to prove the machinery on a fixture where that oracle is well posed - 25
benchmarks against 23 parameters, so no manifold - and the fixture REFUTED IT A THIRD TIME, for a
different reason. Full column rank makes theta*(q) a function only if the solve REACHES the minimum,
and this one does not: the chain closes 7.5 of the 8 orders between the seed and stationarity and
stops at `||J'r|| = 8.6e3`. So a re-solve at a bumped quote still lands a roughly fixed distance
away whatever the bump, `(theta*(q+h) - theta*(q-h))/2h` still has no limit, and the quote-bump
ladder still scatters. The displacement is the discriminating measurement, and it UNIFIES the two
refutations: the solve wanders in the directions the objective is flat in, which on four quotes are a
true null space and here are the ones the declared cutoff discards. Either way they are exactly the
directions the pseudo-inverse declines to report a derivative for.

What the file therefore gates is everything that does NOT need that oracle:

    the fixture   rank 23 and the singular spectrum, the declared cutoff that says how much of it
                  carries a derivative, and how far short of stationarity the solve stops.
    the dropped   the Gauss-Newton term, measured on BOTH sides. The block's residual is already a
      terms       square, so neither half is second-order small - each is HALF what it corrects -
                  and they CANCEL, so Gauss-Newton is the exact leading-order derivative here.
    the chain     dV/dtheta contracted with the displacement the re-solve actually made reproduces
                  the CVA it actually moved - which puts the failure in the SOLVE and nowhere else.
    the identity  the benchmark self-delta, through the full chain and with no bump in it - its
                  TRACE counts the directions the quote set identifies, and lands on the integer.
    the direction the well-posed value check: step theta by what the quotes identify, re-price, and
                  compare against dV/dq. A sign flip in the backward has to fail it.

COST. The six re-solves behind the divergence measurement dominate: a 25-benchmark solve is about
two and a half minutes, and the file runs ten of them. Measured at 24 minutes, of which the
divergence gate alone is eleven.

Run: ``pytest tests/test_swaption_quote_triangle.py -q``
"""
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pytest
import torch

from derivus import bootstrappers, utils

from crn_ladder import ladder
from test_swaption_calibration_solve import RCOND, calibration, jacobian
from test_swaption_quote_attachment import IR, KEYS, PARAMS, cva, pkey, quote_grad, run, saved, world

#: The identified quote set: every (expiry, tenor) of a 5 x 5 grid, quoted on its own row so that a
#: bump moves the quote rather than moving where the quote comes from. Two properties were measured
#: rather than assumed and both drove the choice. The expiries reach 10Y because the sigma term
#: structures carry knots out to 120 months, and a knot past the last expiry is in nobody's variance
#: integral - a grid stopping at 5Y is rank deficient however many rows it has. And the quotes are
#: FLAT: a cube shaped in expiry and tenor is fittable only by pushing a front sigma knot onto its
#: 1e-5 lower bound, where the solution is not interior and unconstrained stationarity cannot hold.
EXPIRIES, TENORS = (1, 2, 3, 5, 10), (1, 2, 3, 5, 10)
GRID = tuple((expiry, tenor, 20.0) for expiry in EXPIRIES for tenor in TENORS)
#: `Stationarity_Tol` is DECLARED on the block and the default is wrong for it: the norm is absolute
#: and this objective's scale is its own. `Jacobian_Rcond` keeps its default, which the rank gate
#: reports the consequence of. Both are held between what the solve achieves and what the seed sits
#: at, so neither is a check that never fires.
IDENTIFIED = {'Stationarity_Tol': 1e5}
#: The rungs, as absolute moves in the decimal vol the quote leaf carries - 0.5, 0.2 and 0.1 vol
#: points, the range stages B and C measured the manifold's sign reversal over.
RUNGS = (5e-3, 2e-3, 1e-3)
#: Measured, each with headroom. `DROPPED_TERM` is the tight one and the point of the file: the
#: dropped Gauss-Newton term is half the Gauss-Newton matrix to 0.15%.
DROPPED_TERM, CANCELLATION, SELF_DELTA, DIRECTION = 0.02, 0.02, 0.05, 1.5
#: The q-side finite difference's rung, in the PERCENT the `Instrument_Definitions`
#: column is authored in - the leaf carries that over a hundred.
DROPPED_BUMP = 1e-3
#: The direction check's step, in the decimal vol the leaf carries - a thousandth of a vol point.
#: `dtheta/dq` has entries of order 60 here, so a bump any larger walks a sigma knot through its own
#: 1e-5 lower bound and the re-price is of a process nobody calibrated.
DIRECTION_BUMP = 1e-5

CACHE = {}


def identified(connect, bumps=()):
    """The stage-C world, re-quoted on the identified grid with its own stationarity tolerance."""
    return world(connect, bumps, GRID, **IDENTIFIED)


def base():
    """`(config, calc, dV/dq)` of the identified world with its quotes connected - one solve."""
    if 'base' not in CACHE:
        config = identified(True)
        calc = run(config)[0]
        CACHE['base'] = (config, calc, quote_grad(config))
    return CACHE['base']


def rung(j, shift):
    """`(theta*, CVA)` of one whole re-authored, re-bootstrapped, re-priced job - cached.

    The expensive object in the file, and the reason three gates read it: they ask three different
    questions of the SAME re-solve - how far theta moved, where that displacement points, and what
    it did to the value.
    """
    key = (j, round(shift, 12))
    if key not in CACHE:
        config = identified(False, ((j, shift),))
        CACHE[key] = (saved(config), cva(run(config, greeks=False)[1]))
    return CACHE[key]


def quotes_of(config):
    return [leaf for _, leaves in config.quote_leaves.values() for leaf in leaves]


def quote_jacobian(config):
    """`dtheta/dq`, 23 x quotes, one ROW at a time out of the published node's own backward.

    A cotangent IS a row, so the matrix costs one backward per parameter, and rebuilding it from `J`
    and `dr/dq` in the gate would restate the contraction being gated beside it. Cached: each of
    those backward passes re-prices the benchmark set.
    """
    if 'dtheta' not in CACHE:
        theta = torch.cat([config.calibrated_factors[pkey(name)] for name in KEYS])
        CACHE['dtheta'] = np.array([
            torch.stack(torch.autograd.grad(theta, quotes_of(config), grad_outputs=e,
                                            retain_graph=True)).double().cpu().numpy()
            for e in torch.eye(theta.numel(), dtype=theta.dtype, device=theta.device)])
    return CACHE['dtheta']


def solved_calibration():
    """`(calibration, theta*, J, dr/dq, r)` of the identified block at the published theta*."""
    if 'calibration' not in CACHE:
        cal = calibration(benchmarks=GRID)
        theta = torch.tensor(saved(base()[0]), dtype=cal.implied_var[cal.keys[0]].dtype,
                             device=cal.implied_var[cal.keys[0]].device)
        CACHE['calibration'] = (cal, theta) + jacobian(cal, theta)
    return CACHE['calibration']


def biggest():
    """The quote the value moves most with - the one every bump below is taken on."""
    return int(np.abs(base()[2]).argmax())


def factor_greek():
    """`dV/dtheta` off the connected run's own leaves, 23 long in the closure's parameter order."""
    if 'dv' not in CACHE:
        calc = base()[1]
        CACHE['dv'] = np.concatenate([
            np.atleast_1d(calc.implied_var[IR][pkey(name)].grad.double().cpu().numpy())
            for name in KEYS])
    return CACHE['dv']


# ---------------------------------------------------------------------------------------------
# The fixture, and how far the solve gets on it
# ---------------------------------------------------------------------------------------------

def test_the_identified_fixture_has_full_column_rank_and_the_cutoff_says_how_much_of_it_counts():
    """25 benchmarks against 23 parameters, and the whole premise of the file measured rather than
    asserted in prose.

    `J` has rank 23, so there is no null space and no minimum-norm convention to invoke - that much
    is what the fixture was built for. What it does NOT have is a GAP. The four-quote block's
    spectrum is four real directions twelve orders above nineteen numerical zeros, and the declared
    `Jacobian_Rcond` of 1e-8 sits in the middle of it with room to spare. Here the 23 real directions
    span the conditioning of the swaption grid itself - five orders end to end - and the same cutoff
    keeps 18 of them. That is not a defect either: the term Gauss-Newton drops is the same SIZE as
    the eigenvalues of the last five, so a derivative along them is a derivative of the wrong
    Hessian. The gate reports the spectrum and how many each cutoff would keep.
    """
    _, _, J, _, _ = solved_calibration()
    singular = torch.linalg.svdvals(J).cpu().numpy()
    kept = {rcond: int(((singular / singular[0]) ** 2 > rcond).sum())
            for rcond in (1e-6, RCOND, 1e-10, 1e-13)}
    logging.info('identified fixture: singular values %s',
                 np.array2string(singular, precision=4, max_line_width=250))
    logging.info('identified fixture: smin/smax %.4g, directions kept per cutoff %s',
                 singular[-1] / singular[0], kept)
    assert J.shape == (len(GRID), 23), J.shape
    assert int(torch.linalg.matrix_rank(J)) == 23, (
        'the identified fixture is rank deficient: {}'.format(singular))
    assert kept[1e-13] == 23 and kept[RCOND] == 18, kept


def test_the_solve_stops_short_of_stationarity_and_the_block_declares_how_short():
    """The property everything downstream turns on, and the reason this file has no closing triangle.

    The implicit function theorem holds where `J'r` vanishes. The optimizer chain closes most of the
    distance from its seed - seven and a half orders - and then stops: it is minimising a QUARTIC in
    the pricing error, because the block's residual is already a square, and a quartic is flat enough
    near its minimum that the relative-improvement test fires long before the gradient does. On a
    block quoting FEWER swaptions than the model has parameters this never shows, because the fit is
    then interpolating and `||J'r||` is small for free - 3.3e-6 on the four-quote fixture. It shows
    here, and it is why `theta*(q)` is the optimizer's stopping point rather than the argmin.

    `Stationarity_Tol` is declared on the block because the norm is ABSOLUTE while the objective's
    scale is the block's own. The gate holds the declared value between what the solve achieves and
    what the seed sits at, six orders apart, so it cannot be satisfied by a check that never fires.
    """
    cal, theta, J, _, r = solved_calibration()
    seed = torch.tensor(cal.optimizers[0][1], dtype=theta.dtype, device=theta.device)
    J0, _, r0 = jacobian(cal, seed)
    gradient, at_seed = float((J.t() @ r).norm()), float((J0.t() @ r0).norm())
    logging.info('||J^T r|| %.6g at theta*, %.6g at the seed, ||r|| %.6g (max rel err %.4g)',
                 gradient, at_seed, float(r.norm()),
                 float(np.sqrt(np.abs(r.cpu().numpy())).max() / 100.0))
    assert gradient < IDENTIFIED['Stationarity_Tol'] < at_seed / 1e2, (gradient, at_seed)


def dropped_terms():
    """`(theta-side, q-side, J'J, J'dr/dq, columns)` at theta* - the two halves Gauss-Newton drops.

    THE THETA SIDE is a double backward: `g = J' r` with `r` held at its value has theta-derivative
    `sum_i r_i grad^2 r_i`, one Hessian-vector product per parameter.

    THE Q SIDE CANNOT BE TAKEN THAT WAY. `market_swap_class.error` detaches the model price in the
    carried half - which is what stops the calibration Jacobian doubling, see the page - so the
    closure's mixed second derivative `d2r/dtheta dq` is STRUCTURALLY ZERO and autograd reports it as
    such. The honest instrument is a finite difference of `J` in the AUTHORED quote at fixed theta*:
    re-author the block one rung either side, rebuild the residual, difference `J' r`. The bump
    argument is in PERCENT and the leaf carries the decimal vol, which is the `/100` below and the
    whole of the units trap - drop it and both halves read 0.005 instead of 0.5.
    """
    if 'dropped' not in CACHE:
        cal, theta, J, drdq, r = solved_calibration()
        x = theta.detach().requires_grad_(True)
        residual = cal(x)
        tape = torch.stack([torch.autograd.grad(residual[i], x, retain_graph=True,
                                                create_graph=True)[0]
                            for i in range(residual.numel())])
        gradient = (tape * residual.detach().unsqueeze(1)).sum(0)
        theta_side = torch.stack([torch.autograd.grad(gradient[k], x, retain_graph=True)[0]
                                  for k in range(x.numel())]).double()

        columns = sorted({int(np.abs(base()[2]).argmax()), int(np.abs(base()[2]).argmin()), 6})
        q_side = []
        for j in columns:
            halves = []
            for shift in (DROPPED_BUMP, -DROPPED_BUMP):
                bumped = calibration(False, ((j, shift),), benchmarks=GRID)
                xb = theta.detach().requires_grad_(True)
                rb = bumped(xb)
                Jb = torch.stack([torch.autograd.grad(rb[i], xb, retain_graph=True)[0]
                                  for i in range(rb.numel())]).double()
                halves.append(Jb.t() @ r)
            q_side.append((halves[0] - halves[1]) / (2.0 * DROPPED_BUMP / 100.0))
        CACHE['dropped'] = (theta_side, torch.stack(q_side, dim=1), J.t() @ J, J.t() @ drdq,
                            columns)
    return CACHE['dropped']


def restricted_ratio(hessian, cross, columns, k, correct_cross=True):
    """`||GN dtheta/dq|| / ||corrected||`, both solved INSIDE the top-`k` eigenspace of `J'J`.

    Restrict-then-invert rather than pinv-then-compare. The dropped Hessian term has Frobenius norm
    4.8e6 against a smallest eigenvalue of 0.066, so a pseudo-inverse of `J'J + D` couples the
    identified subspace to directions where the correction is five orders larger than the thing it
    corrects, and the comparison stops being about the correction at all.
    """
    theta_side, q_side, gn, gn_cross, _ = dropped_terms()
    values, vectors = torch.linalg.eigh(gn)
    U = vectors[:, -k:]
    A = U.t() @ gn_cross[:, columns]
    approx = torch.linalg.solve(U.t() @ gn @ U, -A)
    exact = torch.linalg.solve(U.t() @ hessian @ U, -(A + (U.t() @ cross if correct_cross else 0.0)))
    cosine = float((approx * exact).sum() / (approx.norm() * exact.norm()))
    return float(approx.norm() / exact.norm()), cosine


def test_the_squared_residual_doubles_both_dropped_terms_and_they_cancel():
    """The measurement the brief asked for, taken EXACTLY - and the answer is that nothing is owed.

    Backward drops `sum_i r_i grad^2 r_i` from the Hessian of half the sum of squares. Every textbook
    calls that second-order small in the residual, and on this block that is FALSE: the residual is
    already a square, `r_i = w_i f_i^2` with `f = 100(P/M - 1)`, so

        J'J          = sum_i 4 f_i^2 (grad f_i)(grad f_i)'
        dropped      = sum_i 2 f_i^2 (grad f_i)(grad f_i)' + 2 f_i^3 grad^2 f_i

    and the leading part of the second is exactly HALF the first, at any residual level.

    THE SAME DOUBLING HAPPENS ON THE OTHER SIDE, and that is the point. The implicit function theorem
    needs `dg/dq = J'(dr/dq) + (dJ'/dq) r`, and the identical algebra makes the second term half the
    first. So the true derivative is `-[3/2 J'J]^-1 [3/2 J'(dr/dq)]` - the two halves CANCEL, and
    Gauss-Newton is the exact leading-order answer rather than a 3/2 approximation of it. Squaring
    the residual row-scales `J` and `dr/dq` by the same diagonal, and the normal equations do not
    care. Measuring only the Hessian side and correcting only that is what makes a 3/2 appear, and
    the gate below runs exactly that as its mutation.

    Where the two DO disagree is where the `O(f^3)` remainder overtakes the eigenvalue it corrects -
    the same directions `Jacobian_Rcond` already declines to differentiate, which is this file's
    theme rather than an exception to it. So the ratio is asserted tight in the top four directions
    and its DEGRADATION is asserted as a property further down.
    """
    theta_side, q_side, gn, gn_cross, columns = dropped_terms()
    halves = {j: (float(q_side[:, i].norm() / gn_cross[:, j].norm()),
                  float((q_side[:, i] @ gn_cross[:, j])
                        / (q_side[:, i].norm() * gn_cross[:, j].norm())))
              for i, j in enumerate(columns)}
    logging.info('dropped terms: theta side %.6f of J^T J (residual %.4g); q side per column '
                 '(ratio, cosine) %s', float(theta_side.norm() / gn.norm()),
                 float((theta_side - 0.5 * gn).norm() / gn.norm()),
                 {j: (round(a, 6), round(b, 6)) for j, (a, b) in halves.items()})
    assert abs(float(theta_side.norm() / gn.norm()) - 0.5) < DROPPED_TERM
    assert float((theta_side - 0.5 * gn).norm() / gn.norm()) < DROPPED_TERM
    for j, (ratio, cosine) in halves.items():
        assert abs(ratio - 0.5) < 2 * DROPPED_TERM and cosine > 0.999, (j, ratio, cosine)

    ratios = {k: restricted_ratio(gn + theta_side, q_side, columns, k) for k in (4, 6, 8, 12)}
    logging.info('GN over both-corrected, restricted to the top k directions: %s',
                 {k: (round(a, 4), round(b, 6)) for k, (a, b) in ratios.items()})
    assert abs(ratios[4][0] - 1.0) < CANCELLATION and ratios[4][1] > 1.0 - CANCELLATION, ratios[4]
    assert ratios[12][0] > 1.0 + 10 * CANCELLATION, (
        'the O(f^3) remainder has stopped mattering in the directions the cutoff discards, so the '
        'unification with the rank story no longer holds: {}'.format(ratios))


# ---------------------------------------------------------------------------------------------
# The oracle, refuted a third time - and the measurement that says which cause
# ---------------------------------------------------------------------------------------------

def test_a_re_solve_at_a_bumped_quote_diverges_on_the_identified_fixture_too():
    """The honest negative result this file exists to pin, and the reason it has no closing ladder.

    Full column rank makes `theta*(q)` a function only where the solve reaches the minimum. It does
    not, so two solves at quotes a fifth of a vol point apart differ by a displacement set by where
    each one stopped rather than by the bump: `||theta*(q+h) - theta*(q-h)||` is roughly FIXED as `h`
    shrinks, so the quotient GROWS - the same `1/h` signature stage B measured on the manifold, from
    a different cause. The CRN value ladder built on those re-solves scatters accordingly, and is
    reported here with agreement and flatness rather than collapsed into one number.

    If this ever converges, the solve has started returning a function of its quotes and the
    comparison this whole increment was briefed to use has become available.
    """
    j = biggest()
    quote = float(quotes_of(base()[0])[j].detach())
    steps = {h: np.linalg.norm(rung(j, h * 100.0)[0] - rung(j, -h * 100.0)[0]) for h in RUNGS}
    quotients = {h: steps[h] / (2.0 * h) for h in RUNGS}
    result = ladder(price=lambda q: rung(j, (q - quote) * 100.0)[1], aad=float(base()[2][j]),
                    base=quote, rungs=RUNGS, absolute=True)
    logging.info('re-solve displacement per rung %s, quotient %s',
                 {h: round(v, 5) for h, v in steps.items()},
                 {h: round(v, 3) for h, v in quotients.items()})
    logging.info('identified ladder, quote %d:\n%s', j, result)
    assert quotients[RUNGS[-1]] > 1.5 * quotients[RUNGS[0]], (
        'the finite-difference quotient has stopped growing as h shrinks - the re-solve may now be '
        'a function of its quotes: {}'.format(quotients))
    assert not result.agrees(tol=0.25, flat_tol=0.25), (
        'the identified ladder converged - this gate should be replaced by the comparison it '
        'refuses\n{}'.format(result))


def test_the_displacement_wanders_in_the_directions_the_cutoff_discards():
    """The measurement that says WHICH directions the re-solve wanders in, and it unifies the two
    refutations rather than contrasting them.

    Project the displacement onto the subspace the declared `Jacobian_Rcond` keeps. On the
    four-swaption block stage B measured most of it landing OUTSIDE - there the discarded directions
    are a true null space, nineteen combinations four quotes cannot see. Here `J` has full column
    rank, so the exact-arithmetic projector is the identity and that split says nothing; taken at the
    DECLARED cutoff it says a great deal, because most of the displacement still lands outside.

    Same sentence covers both: the solve wanders in the directions the objective is FLAT in, and
    those are precisely the directions the pseudo-inverse declines to report a derivative for. The
    cutoff is not hiding information, it is refusing to invent it.
    """
    _, _, J, _, _ = solved_calibration()
    projector = (torch.linalg.pinv(J.t() @ J, hermitian=True, rtol=RCOND)
                 @ J.t() @ J).cpu().numpy()
    j = biggest()
    inside = {h: float(np.linalg.norm(projector @ (rung(j, h * 100.0)[0] - rung(j, -h * 100.0)[0]))
                       / np.linalg.norm(rung(j, h * 100.0)[0] - rung(j, -h * 100.0)[0]))
              for h in RUNGS}
    logging.info('fraction of the re-solve displacement inside the kept subspace: %s',
                 {h: round(v, 4) for h, v in inside.items()})
    assert max(inside.values()) < 0.8, inside


def test_the_value_chain_tracks_the_theta_move_the_re_solve_made():
    """The gate that puts the failure in the SOLVE and nowhere else.

    The re-solve moves theta somewhere; contract the reported factor greek with that displacement and
    it reproduces the CVA the job actually moved, better as the bump shrinks. So `dV/dtheta` is right,
    the attachment is right, and the pricing chain between them is right - what is not a function of
    the quotes is `theta*` itself. Without this, a scattering ladder would be evidence against
    everything at once.
    """
    j = biggest()
    errors = {}
    for h in RUNGS:
        up, down = rung(j, h * 100.0), rung(j, -h * 100.0)
        predicted = float(factor_greek() @ (up[0] - down[0]))
        errors[h] = abs(predicted - (up[1] - down[1])) / abs(up[1] - down[1])
    logging.info('value chain vs the re-solve displacement, relative error per rung %s',
                 {h: round(v, 4) for h, v in errors.items()})
    assert errors[RUNGS[-1]] < errors[RUNGS[0]], errors
    assert errors[RUNGS[-1]] < 0.5, errors


# ---------------------------------------------------------------------------------------------
# The references that ARE well posed
# ---------------------------------------------------------------------------------------------

def test_the_benchmark_self_delta_counts_the_directions_the_quotes_identify():
    """The identity that needs no bump at all, and it is exact.

    A benchmark held at its own market number gives `dM_i/dq_j = delta_ij dP_i/dq_i` only where the
    model reproduces every quote exactly and the inverse is a true inverse. Neither holds here, and
    what replaces the identity matrix is not an approximation of it - it is the ORTHOGONAL PROJECTOR
    onto the subspace the pseudo-inverse kept. Writing `f = 100(P/M - 1)` so that `J = diag(c) A`
    with `A = dM/dtheta`, the reported matrix is

        dM_i/dq_j / (dP_j/dq_j) = -P_ij d_j / c_i        with  P = J (J'J)^+ J'

    whose diagonal is `P_ii M_i/P_i`, within the fit's own error of `P_ii`. So its TRACE is the rank
    of `P`: the number of directions the declared `Jacobian_Rcond` keeps, 18 of 23 here. Measured to
    better than a thousandth, which is what makes this the tightest gate in the file - it reads the
    pseudo-inverse, the transpose, the sign and the whole attachment against an integer.

    The individual diagonals are NOT near one and must not be asserted to be: they run from 0.05 to
    1.04, because the off-diagonal weight is scaled by the ratio of the two benchmarks' own relative
    pricing errors and a benchmark that fits ten times better than its neighbour has a row ten times
    more sensitive to that neighbour's quote. They are logged rather than gated, for the same reason
    the trace is gated rather than the entries.
    """
    cal, theta, J, _, r = solved_calibration()
    x = theta.detach().requires_grad_(True)
    model = cal.loss_fn(cal.split(x))[0]
    sensitivity = torch.stack([torch.autograd.grad(price, x, retain_graph=True)[0]
                               for price in model.values()]).double().cpu().numpy()
    # the market premium's own quote sensitivity, as a central secant on the Black preamble - so no
    # part of this check reuses the machinery it is checking
    own = np.array([float((swap.premium(swap.quote + 1e-4) - swap.premium(swap.quote - 1e-4)
                           ).detach()) / 2e-4 for swap in cal.market_swaps.values()])
    reported = (sensitivity @ quote_jacobian(base()[0])) / own.reshape(1, -1)

    singular = torch.linalg.svdvals(J).cpu().numpy()
    kept = int(((singular / singular[0]) ** 2 > RCOND).sum())
    logging.info('self-delta: trace %.6f against %d directions kept; diagonal %s',
                 np.trace(reported), kept, np.array2string(np.diag(reported), precision=4))
    assert abs(np.trace(reported) - kept) < SELF_DELTA, (np.trace(reported), kept)


def priced_at(theta):
    """The book's CVA with the CONSUMED parameters standing at `theta` - written into `Price Factors`
    as plain numpy, quotes off, nothing else moved and NOTHING RE-SOLVED.

    Its own world, so a gate comparing against the connected job does not depend on run order.
    """
    if 'stepper' not in CACHE:
        CACHE['stepper'] = identified(False)
    param = CACHE['stepper'].params['Price Factors'][PARAMS]
    param['Alpha_1'], param['Alpha_2'], param['Correlation'] = (
        float(theta[0]), float(theta[1]), float(theta[2]))
    param['Sigma_1'] = utils.Curve([], list(zip(param['Sigma_1'].array[:, 0], theta[3:13])))
    param['Sigma_2'] = utils.Curve([], list(zip(param['Sigma_2'].array[:, 0], theta[13:])))
    return cva(run(CACHE['stepper'], greeks=False)[1])


def direction_check(dtheta, delta, bump=DIRECTION_BUMP):
    """`dV/dq . h` against re-pricing at `theta* + dtheta/dq . h`, as a signed ratio.

    Nothing re-solves, so neither the manifold nor the optimizer's stopping point enters: this is the
    step the quotes DO identify, carried through to a value. One is perfect agreement.
    """
    config = base()[0]
    j = biggest()
    moved = priced_at(saved(config) + dtheta[:, j] * bump)
    return (moved - priced_at(saved(config))) / (delta[j] * bump)


def test_a_quote_bump_moves_the_value_the_way_the_quote_delta_says():
    """The direction check, on the step a quote bump identifiably takes - the value-space reference
    a non-stationary solve does not spoil, because nothing here re-solves."""
    ratio = direction_check(quote_jacobian(base()[0]), base()[2])
    logging.info('direction check: re-priced move is %.4f of the predicted one', ratio)
    assert 1.0 / DIRECTION < ratio < DIRECTION, ratio


# ---------------------------------------------------------------------------------------------
# The mutation - can the value-space check see the subsystem break?
# ---------------------------------------------------------------------------------------------

class SignFlipped(bootstrappers.LeastSquaresSolve):
    """`LeastSquaresSolve` with the backward's sign reversed - `dL/dq = +(dr/dq)' J w`.

    The forward pass is untouched, so every value in the job is identical and only the reported
    derivative is wrong: the failure mode the whole increment exists to make visible, and the one a
    price gate cannot see.
    """

    @staticmethod
    def backward(ctx, cotangent):
        grads = bootstrappers.LeastSquaresSolve.backward(ctx, cotangent)
        return grads[:3] + tuple(-g for g in grads[3:])


@pytest.mark.parametrize('wrapper,should_agree', [(bootstrappers.LeastSquaresSolve, True),
                                                  (SignFlipped, False)])
def test_a_sign_flip_in_the_backward_fails_the_direction_check(wrapper, should_agree):
    """MUTATE the backward and score it in VALUE space, against a re-price that knows nothing about
    any of this.

    Both arms go through the same path - `dtheta/dq` read out of the wrapper's OWN backward, the
    factor greek from the connected run, then a re-price at the stepped parameters - so the only
    difference between them is the wrapper. The unflipped arm's reconstruction is checked against
    the one-pass `dV/dq` in the same breath, and that check is what stops this being a placebo: a
    reconstruction that had drifted from the number the job reports would pass both arms while
    measuring nothing.
    """
    j, delta = biggest(), base()[2]
    cal = calibration(benchmarks=GRID)
    theta = wrapper.apply(cal, RCOND, IDENTIFIED['Stationarity_Tol'], *cal.quotes)
    dtheta = np.array([torch.stack(torch.autograd.grad(
        theta, cal.quotes, grad_outputs=e, retain_graph=True)).double().cpu().numpy()
        for e in torch.eye(theta.numel(), dtype=theta.dtype, device=theta.device)])
    reconstructed = float(factor_greek() @ dtheta[:, j])
    logging.info('%s: reconstructed dV/dq %.6g against one-pass %.6g',
                 wrapper.__name__, reconstructed, delta[j])
    if should_agree:
        assert abs(reconstructed - delta[j]) < 1e-3 * abs(delta[j]), (
            'the reconstruction is not the one-pass delta, so this gate measures nothing: '
            '{} against {}'.format(reconstructed, delta[j]))

    ratio = direction_check(dtheta, delta)
    logging.info('%s: direction check ratio %.4f', wrapper.__name__, ratio)
    assert (1.0 / DIRECTION < ratio < DIRECTION) == should_agree, (wrapper.__name__, ratio)


