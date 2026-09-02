"""The COMPONENT Heston-Nandi model (Christoffersen-Jacobs-Ornthanalai-Wang), end to end.

THE SPINE IS THE NESTING IDENTITY. Setting phi = 0 and holding the long-run curve L flat collapses
the component recursion onto the plain one under an EXACT parameter map

    omega_p = L(1-beta) - alpha,  beta_p = beta - alpha*gamma_1^2,  alpha_p = alpha,  gamma_p = gamma_1

whose inverse is `beta = psi_p` (the plain PERSISTENCE, not the plain beta) and `L` = the plain
STATIONARY variance. Both the closed form and the daily-step recursion have to land on the plain
model's own numbers there, and a wrong sign, a dropped centering term or a q read one step early
dies in that gate the way in-out parity killed wrong barrier formulas. Everything else in this file
is downstream of it.

WHAT EACH GATE HOLDS, and the numbers it measured on this workstation:

  1  NESTING, closed form      1.5e-13 relative over 5 strikes x 3 expiries, and 1.2e-10 on the one
                               deep out-of-the-money 21-step price of 0.007 (8.1e-13 ABSOLUTE, the
                               inversion's own cancellation floor). Machine precision - the map is
                               exact, so nothing here is a tolerance
     NESTING, sub-step         the DRAWS are bitwise identical; the paths agree to 1.0e-15 on the
                               log-spot and 2.6e-15 on the variance - the centered algebra's own
                               rounding and not a different law; see the gate's docstring for why
                               bitwise is unreachable and why that is correct
     L curve mechanics        piecewise-linear in t, flat outside its knots, omega AFFINE within a
                               pillar at (B-A)(1-rho)/n per step - this gate was written asserting
                               "constant", failed, and corrected the docstring it holds. It also
                               holds the certificate's omega_min against the FLAT TAIL past the
                               last knot, which is the global minimum on a rising strip whose last
                               segment is shorter than 1/(1-rho): 1.006e-6 read off the strip
                               alone against 1.000e-6 true
     the fitted box           feasible ALGEBRA at every corner - both positivity constraints hold
                               by construction, reparam/unreparam round-trips with its sign - and
                               one corner PRICES (1.5619 at the box's 31,623 leverage). Not every
                               point prices: (beta 0.5, l 0.9, a 0.5) is NaN, which the objective
                               reads as +inf
     the zero-length walk     a kit's FIRST call can be omegas(0, 0) - a fixing one trading day
                               out - and answered a slice of None: measured, a NaN Value on a row
                               that still looks priced, the TypeError swallowed on the way
     quadrature bounds        a phi_max is NOT transferable between contracts - past a contract's
                               own bound the recursion DIVERGES, so a larger one is not
                               conservative. This gate is the corpse of a shortcut that shipped
                               inside the calibration for an evening and mispriced a pillar 0.4%
  2  closed form vs MC         3 strikes x 2 expiries on LIVE phi and rho, 2^20 paths, every one
                               inside 1.6 standard errors, and the variance floor 1e+07 clear
  3  the L bootstrap           ATM repriced to 2.4e-15 relative (they are bootstrapped), worst wing
                               6.88% of premium / 0.462 vol points, L within 0.88% of the market's
                               own forward variance strip, q0 == L(0) exactly
  4  the negative-omega guard  a hump-shaped ATM refuses BY NAME, with the pillar, the level it
                               wanted, the least admissible one and both remedies
     the Rho pin               0 <= Rho < 1 refused AT THE READ: above 1 the long-run component is
                               non-stationary AND the admissible-level floor goes negative, which
                               DISABLES the guard above rather than tripping it
  5  BASEVAL + CREDIT MC       a vanilla (a TARF with one fixing and an unreachable target) prices
                               -0.25% from the component closed form; a one-day TARF (which walks
                               ZERO unmonitored sub-steps) prices finite and keeps its row; a
                               six-fixing TARF carries an 18-row exposure profile with dispersion
                               on every row and a CVA of 2,777.04
  6  both leverage signs       the negative-RR fixture lands Gamma_1 NEGATIVE (-845.1) at an
                               interior optimum, which is the plain family's own precedent
  7  Quote_Sensitivity         REFUSED by name, before any work: it names brentq, the implicit
                               function theorem, the roadmap row and the quote chains that ARE
                               differentiable - which are the surface/curve families, NOT the
                               plain HN one, which declares no Quote_Sensitivity field at all

NO HMC ANYWHERE. Gate 5 is base valuation plus credit Monte Carlo, which is this repo's bar for a
model, and both run through real JSON and the real entry points.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json

import numpy as np
import pandas as pd
import pytest
import torch

import derivus
from derivus import run_baseval, utils
from derivus.bootstrappers import HestonNandiComponentModelParameters as COMPONENT
from derivus.config import Config, CustomJsonEncoder
from derivus.instruments import construct_instrument
from derivus.pricing import ComponentHestonNandiKit
import hn_reference as hnref
from conftest import needs_hn_fused

BASE = pd.Timestamp('2024-06-28')
DTYPE = torch.float64
SPY = 252.0


def _tensor(x):
    return torch.tensor(float(x), dtype=DTYPE)


# ======================================================================================
# GATE 1 - the nesting identity
# ======================================================================================

#: A plain HN law with real leverage and a persistence a desk would recognise. `hn_reference`
#: builds it from targets so the numbers are readable; `utils.hn_component_from_plain` is the map
#: under test and is never re-spelled here.
PLAIN = hnref.hn_params_from_targets(
    ann_vol=0.30, persistence=0.94, gamma=350.0, leverage_share=0.7, steps_per_year=SPY)
H0 = 1.6 * float(utils.hn_stationary_var(
    PLAIN['omega'], PLAIN['alpha'], PLAIN['beta'], PLAIN['gamma_star']))


def _nesting_pair(r_step=0.02 / SPY):
    """The plain parameters and their EXACT component image: phi = 0, L flat at the plain
    stationary level, beta = the plain persistence, gamma_2 arbitrary (it multiplies phi)."""
    om, al, be, ga = (_tensor(PLAIN[k]) for k in ('omega', 'alpha', 'beta', 'gamma_star'))
    alpha, beta, gamma1, level = utils.hn_component_from_plain(om, al, be, ga)
    # gamma_2 is deliberately NOTHING like gamma_1: at phi = 0 it multiplies zero, so a component
    # form that leaked it into the h recursion would be caught here rather than in a later gate
    return (om, al, be, ga), (alpha, beta, gamma1, _tensor(0.99), _tensor(0.0),
                              _tensor(-1234.5)), level, _tensor(r_step)


def test_the_component_closed_form_is_the_plain_one_on_its_nested_face():
    """phi = 0, L FLAT at the plain model's implied stationary level, gamma_2 anything: the
    component European price IS `utils.hn_call`, over a strike/expiry grid.

    MACHINE PRECISION, NOT A QUADRATURE TOLERANCE, and that is the point. The parameter map is
    exact (`utils.hn_component_from_plain` / `hn_component_to_plain` are inverses in closed form),
    both pricers hand their log-CF to the SAME `cf_european_probabilities` primitive, and both
    resolve the same adaptive `phi_max` - so the only difference left is the order the two
    recursions accumulate their own arithmetic in. MEASURED: 1.53e-13 worst relative over the grid
    once its deep out-of-the-money corner is set aside, and 1.15e-10 there - a 21-step 120 strike
    worth 0.007, where 8.1e-13 ABSOLUTE is the P1-P2 cancellation's own floor rather than a
    disagreement about the law. The gate therefore carries an `abs` beside its `rel`.

    THE C COEFFICIENT IS WHAT MAKES THIS WORK AT ALL. The component log-CF is A + B*h_0 + C*q_0,
    and A alone does NOT reduce to the plain A - the flat-L intercept `L(1-rho)` is not the plain
    `omega`. It is the ANCHORING q_0 = L(0) that closes it: A + C*L follows the plain A recursion
    exactly. So this gate is a test of the anchoring as much as of the algebra, and dropping the C
    term would fail it at every strike.
    """
    plain, comp, level, r = _nesting_pair()
    h0 = _tensor(H0)
    worst = 0.0
    for n in (21, 63, 252):
        omegas = list(utils.hn_component_omega_path(level.expand(n + 1), comp[3]))
        assert len(omegas) == n
        for strike in (80.0, 95.0, 100.0, 105.0, 120.0):
            a = float(utils.hn_call(100.0, strike, n, h0, *plain, r, panels=64))
            b = float(utils.hn_component_call(
                100.0, strike, omegas, h0, level, *comp, r, panels=64))
            worst = max(worst, abs(b / a - 1.0))
            assert b == pytest.approx(a, rel=1e-9, abs=1e-11), (
                'n=%d K=%g: component %.14f vs plain %.14f' % (n, strike, b, a))
    assert worst < 1e-9, 'nesting agreement degraded to %.3e' % worst
    # and the map is an involution, so the docstring's algebra is held rather than described
    back = utils.hn_component_to_plain(comp[0], comp[1], comp[2], level)
    for got, want in zip(back, plain):
        assert float(got) == pytest.approx(float(want), rel=1e-12)


def test_the_component_substep_walks_the_plain_path_draw_for_draw():
    """The SAME seed through both sub-steps, on the nested face: identical draws, one law.

    BITWISE IS UNREACHABLE HERE AND SHOULD BE. The plain step computes
    `omega + beta*h + alpha*(z - g*sh)^2`; the component step computes
    `q + beta*(h - q) + alpha*[(z - g*sh)^2 - (1 + g^2*h)]` with q held at L. Those are the same
    real number and a DIFFERENT expression tree, so floating point reassociates them - and the
    centered form is the model's own algebra, not a spelling to be bent for a test. What IS
    bitwise is the thing a draw-for-draw claim is actually about: the innovation sequence. The
    gate therefore asserts `torch.equal` on the draws and 1e-14 relative on the paths.

    MEASURED: draws bitwise identical over 40 steps x 4096 paths; worst relative divergence
    1.03e-15 on log S and 2.61e-15 on h, both flat in the step count - the deviation does not
    compound, because the nested recursion is contracting at beta = psi = 0.94.
    """
    plain, comp, level, r = _nesting_pair()
    b_step = torch.full((1, 4096), float(r), dtype=DTYPE)
    h_p = torch.full((1, 4096), H0, dtype=DTYPE)
    h_c, q_c = h_p.clone(), torch.full((1, 4096), float(level), dtype=DTYPE)
    log_p, log_c = torch.zeros_like(b_step), torch.zeros_like(b_step)
    omega_t = float(level) * (1.0 - float(comp[3]))

    g_plain = torch.Generator().manual_seed(20240628)
    g_comp = torch.Generator().manual_seed(20240628)
    worst_s, worst_h = 0.0, 0.0
    for _ in range(40):
        z_p = torch.randn((1, 4096), generator=g_plain, dtype=DTYPE)
        z_c = torch.randn((1, 4096), generator=g_comp, dtype=DTYPE)
        assert torch.equal(z_p, z_c), 'the two streams diverged - this is not draw for draw'
        log_p, h_p = utils.hn_log_substep(log_p, h_p, z_p, b_step, *plain)
        log_c, h_c, q_c = utils.hn_component_log_substep(
            log_c, h_c, q_c, z_c, b_step, omega_t, *comp)
        worst_s = max(worst_s, float((log_c - log_p).abs().max() / log_p.abs().max()))
        worst_h = max(worst_h, float((h_c - h_p).abs().max() / h_p.abs().max()))
        # q is the flat level for ever, which is what makes the face nested
        assert float((q_c - float(level)).abs().max()) < 1e-18

    assert worst_s < 1e-14, 'log-spot paths diverged: %.3e' % worst_s
    assert worst_h < 1e-13, 'variance paths diverged: %.3e' % worst_h


def test_the_l_curve_is_piecewise_linear_in_t_and_flat_outside_its_knots():
    """The interpolation IS the model, so it is held directly rather than only through a price.

    `omega_t = L_(t+1) - rho*L_t` DIFFERENCES this curve, which is why the choice matters and why
    it is stated in `hn_component_l_path`'s own docstring: piecewise CONSTANT L would make omega a
    SPIKE at each pillar, and a spline would let it oscillate between pillars nobody quoted.

    LINEAR IN t MAKES OMEGA AFFINE WITHIN A PILLAR, not constant, and the difference is worth the
    arithmetic. On a segment of n steps from A to B, `omega_i = A(1-rho) + (B-A)(1 + i(1-rho))/n`,
    so it drifts by exactly `(B-A)(1-rho)/n` per step and KINKS only at a pillar. That is a second
    difference of zero inside a segment - which is what this asserts, along with the slope's own
    closed form. This gate was written asserting `constant` and failed, which is how the docstring
    it holds came to say `affine`.

    A DEGENERATE CURVE - one knot - answers that level everywhere, which is the flat-L face the
    nesting gate rides.
    """
    knots = np.array([0.0, 21.0 / SPY, 63.0 / SPY])
    values = torch.stack([_tensor(v) for v in (1.0e-4, 2.0e-4, 2.0e-4 + 42.0 * 1.0e-6)])
    path = utils.hn_component_l_path(knots, values, 100, SPY)

    assert path.numel() == 101, 'the path is L_0..L_n, one more than the step count'
    assert float(path[0]) == pytest.approx(1.0e-4), 'L(0) is the first knot, which IS q0'
    assert float(path[21]) == pytest.approx(2.0e-4)
    assert float(path[63]) == pytest.approx(2.0e-4 + 42.0e-6)
    # linear WITHIN each segment: a constant daily slope on each side of the 21-day knot
    first = np.diff(path[:22].numpy())
    second = np.diff(path[21:64].numpy())
    assert np.allclose(first, first[0]), 'the first segment is not linear in t'
    assert np.allclose(second, second[0]), 'the second segment is not linear in t'
    assert not np.isclose(first[0], second[0]), 'the fixture has no kink, so it tests nothing'
    # FLAT past the last knot - so omega there is L(1-rho) and stays positive
    assert np.allclose(path[63:].numpy(), float(path[63]))

    # and the omega strip is the difference the model is parametrised by, AFFINE within a pillar
    omegas = utils.hn_component_omega_path(path, _tensor(0.99)).numpy()
    assert len(omegas) == 100
    for lo, hi, a, b, span in ((1, 20, 1.0e-4, 2.0e-4, 21.0), (22, 62, 2.0e-4, 2.42e-4, 42.0)):
        steps = np.diff(omegas[lo:hi])
        assert np.allclose(np.diff(steps), 0.0, atol=1e-18), (
            'omega is not AFFINE inside a pillar - the interpolation is not linear in t')
        assert steps[0] == pytest.approx((b - a) * 0.01 / span, rel=1e-9), (
            'omega drifts at the wrong rate inside a pillar: %.6g against (B-A)(1-rho)/n'
            % steps[0])
    assert float(omegas[-1]) == pytest.approx(float(path[-1]) * 0.01), (
        'past the last knot omega must be L(1-rho)')

    single = utils.hn_component_l_path(np.array([0.0]), values[:1], 5, SPY)
    assert np.allclose(single.numpy(), 1.0e-4), 'a one-knot curve is not flat'

    # THE FLAT TAIL IS PART OF THE STRIP and the certificate's omega_min has to see it: on a rising
    # curve whose last segment is shorter than 1/(1-rho) = 100 steps, the GLOBAL minimum is past the
    # last knot, where a strip built out to that knot never looks - 1.006e-6 against 1.000e-6 true.
    rising = torch.stack([_tensor(v) for v in (9.0e-5, 9.9e-5, 1.0e-4)])
    rising_knots = np.array([0.0, 0.25, 0.5])
    to_last_knot = COMPONENT.l_strip(rising_knots, rising, 126, _tensor(0.99), SPY)
    assert min(float(x) for x in to_last_knot) == pytest.approx(1.00587e-6, rel=1e-4)
    assert COMPONENT.omega_floor(rising_knots, rising, _tensor(0.99), SPY) == pytest.approx(
        1.0e-4 * 0.01, rel=1e-12), 'the certificate misses the flat tail past the last knot'
    assert COMPONENT.omega_floor(rising_knots, rising, _tensor(0.99), SPY) < min(
        float(x) for x in to_last_knot), 'the fixture no longer shows the miss - re-derive it'


def test_every_point_of_the_fitted_box_is_a_feasible_model():
    """WHICH HALF THIS ASSERTS: feasible ALGEBRA at every point of the box - both positivity
    constraints hold by construction, so no iterate needs a penalty to keep Alpha, H0 and the
    nested-face intercept positive - AND that one corner of it prices finite. It does NOT assert
    that every point prices: away from the nested face the MGF can diverge, and the objective reads
    that candidate as infeasible (+inf) rather than scoring it. MEASURED, flat L over 21 steps: the
    (beta 0.5, l 0.9, a 0.5) point of this very loop prices NaN, which is the wall working, and the
    (beta 1-1e-6, |l| 1, a 1e-3) corner - the box's extreme leverage - prices 1.5619.

    TWO CONSTRAINTS, ONE PER SHARE, and both are checked at the box's own corners rather than at a
    comfortable interior point:

      * the LEVERAGE share holds `Alpha*Gamma_1^2 = |l|*Beta`, so the plain-equivalent GARCH
        coefficient `Beta(1-|l|)` is non-negative - what keeps the variance recursion positive;
      * the ARCH share holds `Alpha = a*H0*(1-Beta)`, so the nested-face intercept
        `omega_p = H0(1-Beta) - Alpha` is strictly positive. Without it the moment generating
        function the pricer inverts DIVERGES: measured before this was a share, the adaptive
        phi_max scan ran to its 2^24 cap and every price came back NaN.

    `unreparam` is its inverse, and the round trip is asserted rather than assumed - it is what a
    warm start off a previously written factor rides. The SIGN travels with it, in both directions,
    which is the property the negative-risk-reversal fixture needs.
    """
    for beta, share, arch, phi_share in ((0.5, 0.9, 0.5, 0.5), (0.999, -0.999, 0.999, 1.0),
                                         (1e-4, 1e-6, 1e-3, 0.0), (0.9, -0.3, 0.05, 0.7)):
        x = torch.tensor([beta, share, arch, phi_share, np.log(7.3e-5)], dtype=DTYPE)
        alpha, b, gamma1, phi, h0 = COMPONENT.reparam(x)
        alpha, b, gamma1, phi, h0 = (float(v) for v in (alpha, b, gamma1, phi, h0))

        assert alpha > 0.0 and h0 > 0.0 and phi >= 0.0
        assert np.sign(gamma1) == np.sign(share), 'the share did not carry the sign'
        # the leverage constraint, exactly
        assert alpha * gamma1 ** 2 == pytest.approx(abs(share) * b, rel=1e-12)
        assert b - alpha * gamma1 ** 2 >= -1e-15, 'the plain-equivalent Beta went negative'
        # the intercept constraint: the nested-face omega is strictly positive
        assert h0 * (1.0 - b) - alpha > 0.0, (
            'omega_p = H0(1-Beta) - Alpha is not positive at beta=%g share=%g arch=%g'
            % (beta, share, arch))
        assert phi <= alpha + 1e-18, 'phi is a share of alpha and exceeded it'

        back = COMPONENT.unreparam(alpha, b, gamma1, phi, h0)
        assert back == pytest.approx(x.numpy(), rel=1e-10, abs=1e-12), (
            'reparam/unreparam is not a round trip - a warm start would move the fit')

    # AND ONE CORNER PRICES - feasible algebra is not a finite price. Held where it is most at
    # risk: beta at its ceiling, |l| = 1, the ARCH share at its floor, which is Alpha 7.3e-14 and a
    # dimensionless leverage of 31,623 - the box maximum, not the O(30) the docstring used to claim.
    for share in (1.0, -1.0):
        for phi_share in (0.0, 1.0):
            x = torch.tensor([1.0 - 1e-6, share, 1e-3, phi_share, np.log(7.3e-5)], dtype=DTYPE)
            alpha, b, gamma1, phi, h0 = COMPONENT.reparam(x)
            assert abs(float(gamma1)) * np.sqrt(float(h0)) == pytest.approx(31622.8, rel=1e-4)
            omegas = list(utils.hn_component_omega_path(
                utils.hn_component_l_path(np.array([0.0]), h0.reshape(1), 21, SPY), _tensor(0.99)))
            price = float(utils.hn_component_call(
                100.0, _tensor(100.0), omegas, h0, h0, alpha, b, gamma1, _tensor(0.99), phi,
                gamma1, _tensor(0.0), panels=64))
            assert np.isfinite(price) and price > 0.0, (
                'the box corner (beta 1-1e-6, l %g, a 1e-3, phi share %g) does not price: %r'
                % (share, phi_share, price))
            assert price == pytest.approx(1.5619, rel=1e-3)

    # UNTIED, the vector grows a sixth coordinate: Gamma_2 as a RATIO, sign tied to Gamma_1's
    family = COMPONENT({}, torch.device('cpu'), DTYPE)
    tied = family.unpack([0.9, -0.3, 0.05, 0.7, np.log(7.3e-5)], True, _tensor(0.99))
    untied = family.unpack([0.9, -0.3, 0.05, 0.7, np.log(7.3e-5), 2.5], False, _tensor(0.99))
    assert float(tied[5]) == float(tied[2]), 'Tie_Gamma_2 Yes did not tie'
    assert float(untied[5]) == pytest.approx(2.5 * float(untied[2]))
    assert float(untied[5]) < 0.0, 'the untied Gamma_2 lost the smile direction'
    assert float(tied[3]) == 0.99 and float(untied[3]) == 0.99, 'Rho is not the pinned value'
    assert len(COMPONENT.box(False)) == len(COMPONENT.box(True)) + 1


def test_a_quadrature_bound_is_not_transferable_between_contracts():
    """A LARGER phi_max IS NOT CONSERVATIVE for this model, and this gate is the corpse of a
    shortcut that shipped inside it for an evening.

    The calibration used to derive ONE quadrature bound per L bootstrap - on the shortest pillar,
    on the reasoning that more steps means more variance means faster decay, so the front bound
    must cover the back - and reuse it for every later pillar and every wing. That reasoning is
    right about DECAY and silent about DIVERGENCE. Past a parameter- and step-count-dependent
    point the component A/B/C recursion blows up rather than decaying, so integrating beyond a
    contract's own bound integrates garbage.

    WHAT IT COST, and how it hid: carrying the 21-step contract's bound (512) to the 126-step one
    (which wants 256) solved that pillar's L against a price 0.4% wrong. The ATM ladder is
    BOOTSTRAPPED, so it reprices exactly by construction - the only symptom was the report's own
    recompute, at the correct bound, reading a 3.5e-3 residual on a ladder that should read 1e-12.
    A fit that converged looked converged.

    WHAT THIS HOLDS. At one converged parameter set: the 126-step price is CONVERGED at its own
    bound (identical at 128, 256 and 512, and at 64 against 1024 panels - so it is the quadrature's
    answer and not a resolution artifact), it MOVES at 1024, and it is nonsense at 2048. And the
    two contracts in one strip want DIFFERENT bounds, which is the whole point: 512 for the short
    one, 256 for the long one.
    """
    params = tuple(_tensor(v) for v in
                   (3.29666e-06, 0.832622, -55.9419, 0.99, 2.07667e-06, -55.9419))
    knots = np.array([0.0, 21.0 / SPY, 42.0 / SPY, 63.0 / SPY, 126.0 / SPY])
    values = torch.stack([_tensor(v ** 2 / SPY)
                          for v in (0.1307, 0.1512, 0.1417, 0.1592, 0.1523)])
    carry, strike = _tensor(0.0), _tensor(18.6045)

    def price(n, phi_max, panels=64):
        l_path = utils.hn_component_l_path(knots, values, n, SPY)
        omegas = list(utils.hn_component_omega_path(l_path, params[3]))
        return float(utils.hn_component_call(18.5, strike, omegas, values[0], values[0],
                                             *params, carry, panels=panels, phi_max=phi_max))

    def bound(n):
        l_path = utils.hn_component_l_path(knots, values, n, SPY)
        return utils.hn_component_auto_phi_max(
            list(utils.hn_component_omega_path(l_path, params[3])),
            values[0], values[0], *params, carry)

    short, long_ = bound(21), bound(126)
    assert short != long_, (
        'the two contracts want the same bound, so this fixture cannot show the trap: %g / %g'
        % (short, long_))
    assert long_ < short, 'the LONG contract wants the smaller bound - the trap is the other way'

    converged = price(126, long_)
    for phi_max in (128.0, 256.0, 512.0):
        assert price(126, phi_max) == pytest.approx(converged, rel=1e-12), (
            'the price is not converged in phi_max at %g' % phi_max)
    for panels in (256, 1024):
        assert price(126, long_, panels) == pytest.approx(converged, rel=1e-12), (
            'the price moves with PANELS, so 64 is under-resolved and this reads the wrong thing')

    # ... and past the bound it is wrong, then nonsense
    assert abs(price(126, 1024.0) / converged - 1.0) > 1e-4, (
        'a bound past the divergence no longer moves the price - the trap has been fixed '
        'upstream, and this gate should be re-derived rather than deleted')
    assert not np.isfinite(price(126, 4096.0)) or abs(price(126, 4096.0)) > 1e6, (
        'the recursion no longer diverges at 4096 - re-derive the fixture')


# ======================================================================================
# GATE 2 - the closed form against day-stepped Monte Carlo, on LIVE component parameters
# ======================================================================================

def test_the_closed_form_matches_day_stepped_monte_carlo():
    """GENUINELY COMPONENT PARAMETERS - phi live, rho live, L SLOPING - against the brute-force
    daily recursion. This is the gate that would survive the nesting one being satisfied by a
    model that is right only on its own nested face.

    The reference is `tests/hn_reference.hnc_simulate`, which steps h and q path by path through
    `utils.hn_component_variance_step` and knows nothing about A/B/C. The comparison is a European
    call at three strikes and two expiries, 2^20 paths per cell.

    THE FLOOR MUST NOT BIND, and the gate asserts it rather than hoping. The component recursion
    has no positivity guarantee (see `utils.HN_COMPONENT_VARIANCE_FLOOR`), the simulator floors and
    the Fourier inversion does not - so the two are only the same law where the floor is inactive.
    The reference returns the margin (the smallest variance either state reached, over the floor)
    and it has to be enormous, not merely above one.

    MEASURED, 2^20 paths, seed 11: every one of the six cells inside 1.6 standard errors, worst
    |t| = 1.51 at the 21-step 100 strike. The closed forms run 0.284 to 6.225 in value against
    standard errors of 1.1e-3 to 6.1e-3, and the floor margin is 2.2e+07 at 21 steps and 9.6e+06
    at 63 - seven orders of magnitude clear, so the floored simulator and the unfloored inversion
    are comparing the same law.
    """
    alpha, beta, gamma1 = _tensor(3.5e-6), _tensor(0.81), _tensor(-65.0)
    rho, phi, gamma2 = _tensor(0.985), _tensor(2.0e-6), _tensor(-65.0)
    params = (alpha, beta, gamma1, rho, phi, gamma2)
    r = _tensor(0.02 / SPY)
    # a SLOPING L: 13.5% annualised at the base date rising to 16% at a year, so omega_t is
    # neither constant nor the nested face's
    knots = np.array([0.0, 0.25, 1.0])
    values = torch.stack([_tensor(v ** 2 / SPY) for v in (0.135, 0.150, 0.160)])
    h0 = _tensor(0.135 ** 2 / SPY)

    for n in (21, 63):
        l_path = utils.hn_component_l_path(knots, values, n, SPY)
        omegas = list(utils.hn_component_omega_path(l_path, rho))
        assert min(float(x) for x in omegas) > 0.0, 'fixture is infeasible, not the pricer'
        q0 = l_path[0]
        draws, margin = hnref.hnc_simulate(
            omegas, h0, q0, params, r, 1 << 20, seed=11, dtype=DTYPE)
        assert margin > 1e4, (
            'the simulator floored its variance (margin %.3g over the floor) - the closed form '
            'integrates the UNFLOORED law, so this comparison would be two different laws' % margin)
        for strike in (95.0, 100.0, 105.0):
            payoff = torch.relu(100.0 * draws.exp() - strike) * float(torch.exp(-r * n))
            mc, se = float(payoff.mean()), float(payoff.std() / np.sqrt(payoff.numel()))
            closed = float(utils.hn_component_call(
                100.0, strike, omegas, h0, q0, *params, r, panels=64))
            assert abs(closed - mc) < 3.0 * se, (
                'n=%d K=%g: closed form %.6f vs MC %.6f (se %.6f, %.2f sigma)'
                % (n, strike, closed, mc, se, abs(closed - mc) / se))


# ======================================================================================
# the four-pillar desk_smile world, built through the real bootstrap seam
# ======================================================================================

RATE, FX_SPOT = 0.02, 18.5

FACTORS = {
    'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Spot': 1.0},
    'FxRate.ZAR': {'Domestic_Currency': None, 'Interest_Rate': 'ZAR', 'Spot': FX_SPOT},
    'InterestRate.USD': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                         'Curve': utils.Curve([], [[0.0, RATE], [5.0, RATE]])},
    'InterestRate.ZAR': {'Currency': 'ZAR', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                         'Curve': utils.Curve([], [[0.0, RATE], [5.0, RATE]])}}

#: `test_service.desk_smile`'s four pillars, quoted the way a source posts them. FOUR PILLARS
#: because the ladder needs the ATM term structure to carry more than one expiry, and the RISK
#: REVERSAL IS NEGATIVE in pair terms - which on the `FxRate.ZAR` axis the model is fitted on is a
#: smile RISING with strike, i.e. a NEGATIVE leverage. Gate 6 is that shape.
DESK_SMILE = ((1.0 / 12.0, 0.140, -0.010, 0.0030), (2.0 / 12.0, 0.142, -0.011, 0.0032),
              (0.25, 0.145, -0.012, 0.0035), (0.5, 0.150, -0.014, 0.0040))

#: The same four expiries with a HUMP: the ATM rises hard to 3M and falls off a cliff to 6M. A
#: long-run variance that must fall that fast drives omega_t = L_(t+1) - rho*L_t negative, which is
#: what gate 4 reads.
HUMPED = ((1.0 / 12.0, 0.120, -0.010, 0.0030), (2.0 / 12.0, 0.180, -0.011, 0.0032),
          (0.25, 0.190, -0.012, 0.0035), (0.5, 0.105, -0.014, 0.0040))


def _vol_quotes(points):
    return {'FXVolPrices.USD.ZAR': {'instrument': {
        'Currency': 'USD', 'Delta_Type': 'Forward', 'Premium_Adjusted': 'Yes',
        'ATM_Convention': 'Delta_Neutral_Straddle', 'Grid_Tolerance': 1e-4,
        'Quote_Sensitivity': 'No',
        'Points': [{'Use': 'Yes', 'Expiry': expiry, 'Pillar': pillar, 'Quote_Type': quote_type,
                    'Quoted_Market_Value': value,
                    'Timestamp': pd.Timestamp('2024-06-28 16:30')}
                   for expiry, atm, rr, bf in points
                   for pillar, quote_type, value in ((0.0, 'ATM', atm), (0.25, 'RR', rr),
                                                     (0.25, 'BF', bf))]}}}


def _built_surface(points):
    """A book carrying a BUILT `FXVol.USD.ZAR`, through the real bootstrap seam: the document
    declares the surface bootstrapper and carries the quote block, and `Config.bootstrap` runs it.
    What the component gates start from is therefore a surface the engine built."""
    document = {'Calc': {
        'Calculation': {'Object': 'BaseValuation', 'Base_Date': BASE, 'Currency': 'USD',
                        'MCMC_Simulations': 1, 'Random_Seed': 1},
        'Deals': {'Tag_Titles': '', 'Reference': 'hnc', 'Deals': {'Children': []}},
        'MergeMarketData': {'MarketDataFile': '', 'ExplicitMarketData': {
            'System Parameters': {'Base_Currency': 'USD', 'Base_Date': BASE},
            'Price Factors': FACTORS,
            'Bootstrapper Configuration': {'FXVolSurfaceParameters': {}},
            'Market Prices': _vol_quotes(points)}}}}
    config = derivus.Context().load_json(
        (json.dumps(document, cls=CustomJsonEncoder), 'hn_component')).current_cfg
    config.bootstrap()
    assert 'FXVol.USD.ZAR' in config.params['Price Factors'], 'the surface did not build'
    return config


def _fit(points, **knobs):
    """Emit this family's block off the built surface and run its bootstrap - both real entry
    points, no hand-written quotes and no monkeypatching."""
    config = _built_surface(points)
    name, block = COMPONENT.fx_surface_block(
        'USD.ZAR', config.params['Price Factors'], config.params['System Parameters'],
        config.params['Price Factor Interpolation'])
    block['instrument'].update(knobs)
    config.params['Market Prices'] = {name: block}
    config.params['Bootstrapper Configuration'] = {
        'HestonNandiComponentModelParameters': {}}
    config.bootstrap()
    return config, block['instrument']


@pytest.fixture(scope='module')
def fitted():
    """ONE fit, shared by the round-trip gate and the leverage-sign gate - they are two readings of
    the same calibration and re-running it would double an eight-minute gate for nothing.

    `Max_Iterations` is left at its DECLARED default (300), so what these gates measure is what a
    job gets - INCLUDING the wall clock: MEASURED at 300 outer evaluations in 1591 s (the pair of
    gates, 1601 s), which is why the two share one fixture. ATM residual 2.442e-15, worst wing
    6.88% of premium / 0.462 vol points, weighted wing residual 1.196e-04. The fit reports itself
    CAPPED, because Nelder-Mead over five parameters has not converged there and the family says so
    rather than claiming the tolerance it did not reach.

    THE OUTER SEARCH IS DERIVATIVE-FREE AND THE OBJECTIVE HAS FLAT DIRECTIONS, so a different cap
    lands in a different basin rather than nearer the same one. Measured on this fixture: 300
    evaluations land Gamma_1 -845.1, worst wing 6.88%, L within 0.88% of the market strip; 400 land
    Gamma_1 -78.6, worst wing 5.66%, L within 2.11%. Both are legitimate fits of the same quotes -
    which is why the gates below hold SHAPE and BOUNDS wide enough to cover the family of answers,
    and why every number in their docstrings names the run it came from."""
    config, instrument = _fit(DESK_SMILE)
    return config.params['Price Factors']['HestonNandiComponentModelParameters.ZAR'], instrument


# ======================================================================================
# GATE 3 - the L bootstrap round trip
# ======================================================================================

def test_the_l_bootstrap_reprices_its_atm_ladder_and_lands_on_the_forward_variance_strip(fitted):
    """The round trip off a built surface: the ATM pillars are BOOTSTRAPPED, so they reprice to
    solver precision rather than to a fit quality, and L lands on the market's own forward variance
    strip because that is what L means.

    THE ATM RESIDUAL IS THE BOOTSTRAP'S OWN CONVERGENCE. Each pillar is a bracketed `brentq` on its
    level against its own quote's premium, so "how well does it reprice" is a question about the
    root find and not about the model. MEASURED: 2.442e-15 worst relative over the four pillars -
    and it is EXACTLY that only because the solve and this recompute derive the same quadrature
    bound. When the calibration reused one bound across the ladder it read 3.5e-3 here while the
    solve reported success, which is the whole reason this number is asserted at 1e-8 rather than
    admired (see `test_a_quadrature_bound_is_not_transferable_between_contracts`).

    L AGAINST THE MARKET FORWARD VARIANCE STRIP, AS A NUMBER. `E_0[h_t] = L_t` exactly under the
    anchoring, so the model's expected variance over [T_(k-1), T_k] is the sum of L there, and the
    market's is the difference of the ATM total variances. MEASURED on this fixture, segment by
    segment in annualised vol: model 13.925 / 14.555 / 14.924 / 15.404 percent against market
    13.922 / 14.566 / 14.955 / 15.472 - that is +0.05 / -0.16 / -0.41 / -0.88 percent IN VARIANCE.
    The residual is the model paying for a smile it also has to fit.

    THE GATE HOLDS 8%, an order of magnitude above that, and deliberately: the outer search is
    derivative-free and lands in different basins at different caps (the same fixture at 400
    evaluations reads +2.37 / +1.40 / +0.88 / +0.15, worst 2.4%), so the bound has to cover the
    FAMILY of legitimate answers rather than one run's fourth decimal. It is still tight enough
    that an L bootstrapped against the wrong thing fails by multiples - the natural mistake being a
    TOTAL variance rather than a forward one, which on this fixture is 10% out in variance at the
    back.

    THE ANCHORING IS ASSERTED, not described: q_0 IS L(0), read off the written curve's first knot,
    and that knot's tenor is exactly zero. A curve whose first knot drifted off zero would leave
    `HestonNandiComponentImpliedSpotModel` seeding q from an interpolation instead.
    """
    written, instrument = fitted
    curve = written[utils.HN_COMPONENT_CURVE_NAME].array
    knots, levels = curve[:, 0], curve[:, 1]

    # the anchoring, as a property of the STORED factor
    assert knots[0] == 0.0, 'the L curve does not carry a knot at the base date'
    assert levels[0] == written['H0'], (
        'q0 = L(0) is the anchoring and L(0) = H0 is how the two states are tied at a date no '
        'option is quoted at: %r vs %r' % (levels[0], written['H0']))
    assert len(knots) == 5, 'four ATM pillars plus the zero knot'
    assert all(x > 0.0 for x in levels), 'a non-positive long-run variance was written'

    # the ATM ladder, repriced off the WRITTEN factor through the family's own price function
    family = COMPONENT({}, torch.device('cpu'), DTYPE)
    params = tuple(_tensor(written[k]) for k in
                   ('Alpha', 'Beta', 'Gamma_1', 'Rho', 'Phi', 'Gamma_2'))
    values = torch.stack([_tensor(v) for v in levels])
    discount = np.exp(-RATE * np.array([0.0]))          # flat curves: r == q, so carry is zero

    rungs = instrument['European_Options'][:4]          # the four distinct ATM rungs, in order
    atm_resid = 0.0
    for point in rungs:
        t = (point['Expiry_Date'] - BASE).days / 365.0
        n = max(int(round(t * SPY)), 1)
        omegas = family.l_strip(knots, values, n, _tensor(written['Rho']), SPY)
        fitted_premium = float(family.price(
            FX_SPOT, _tensor(point['Strike']), 1.0, _tensor(point['Units']), omegas,
            values[0], values[0], params, _tensor((RATE - RATE) * t / n), 64,
            _tensor(np.exp(-RATE * t))))
        atm_resid = max(atm_resid, abs(fitted_premium / point['Premium'] - 1.0))
    assert atm_resid < 1e-8, (
        'the ATM ladder is BOOTSTRAPPED and did not reprice: worst %.3e relative' % atm_resid)

    # L against the market's own forward variance strip, segment by segment: the model's expected
    # variance over [T_(k-1), T_k] is the sum of L over those steps (E_0[h_t] = L_t exactly, which
    # is what the anchoring buys), the market's is the difference of the ATM total variances
    days = [max(int(round((p['Expiry_Date'] - BASE).days / 365.0 * SPY)), 1) for p in rungs]
    market_totals = [p['Quoted_Market_Value'] ** 2 * ((p['Expiry_Date'] - BASE).days / 365.0)
                     for p in rungs]
    l_full = utils.hn_component_l_path(knots, values, days[-1], SPY)
    worst_strip, readings = 0.0, []
    prior_days, prior_market, prior_model = 0, 0.0, 0.0
    for n, market_total in zip(days, market_totals):
        model_total = float(l_full[:n].sum())
        span = n - prior_days
        model_rate = (model_total - prior_model) / span
        market_rate = (market_total - prior_market) / span
        readings.append((float(np.sqrt(model_rate * SPY)), float(np.sqrt(market_rate * SPY))))
        worst_strip = max(worst_strip, abs(model_rate / market_rate - 1.0))
        prior_days, prior_market, prior_model = n, market_total, model_total

    assert worst_strip < 0.08, (
        'L is the model EXPECTED long-run variance path and has to be the market forward variance '
        'strip: worst %.1f%% over %s (model, market annualised)' % (
            100.0 * worst_strip, ['(%.2f%%, %.2f%%)' % (100 * a, 100 * b) for a, b in readings]))


# ======================================================================================
# GATE 4 - the negative-omega refusal
# ======================================================================================

def test_a_hump_shaped_atm_term_structure_refuses_by_name():
    """A long-run variance demanded to fall FASTER than rho decays it makes omega_t negative, which
    drives q - and then h - negative. The family refuses BY NAME rather than writing a curve whose
    own simulator would produce a negative variance.

    THE FIXTURE IS A HUMP: 12% at 1M, 18% at 2M, 19% at 3M, then 10.5% at 6M. The 6M rung's own
    forward variance is a small fraction of the 3M level, and no admissible L reaches it - the
    least level whose segment keeps omega_t >= 0 from A over n steps is
    `A(1 - (1-rho)n / (1 + (n-1)(1-rho)))`, a closed form, so the refusal can name the number it
    wanted rather than say a solve failed.

    WHAT THE MESSAGE HAS TO CARRY, and each clause is a separate assertion because a refusal that
    does not say what to do is a crash with better manners: which pillar, the least admissible
    level, the identity that makes it least, the premium it can reach against the one it was asked
    for, and BOTH remedies (Declining_Variance -> Floor, or a lower Rho).

    AND FLOOR IS THE OTHER HALF. The same fixture under `Declining_Variance: Floor` completes,
    writes a curve whose every omega is non-negative, and says in the log which pillar was floored
    and by how much it therefore misprices - which is the whole difference between a repair and a
    silence.
    """
    with pytest.raises(ValueError) as refusal:
        _fit(HUMPED, Max_Iterations=40)
    message = str(refusal.value)

    assert 'ATM pillar demands a long-run variance BELOW' in message
    assert 'omega_t = L_(t+1) - rho*L_t' in message, 'the refusal does not name the identity'
    assert 'non-negative' in message
    assert 'Declining_Variance to Floor' in message, 'a refusal without its first remedy'
    assert 'lower Rho' in message, 'a refusal without its second remedy'
    assert 'y ATM pillar' in message, 'the refusal does not name which pillar'

    # ... and the floor completes, names the pillar, and leaves a curve with no negative omega
    config, _ = _fit(HUMPED, Max_Iterations=40, Declining_Variance='Floor')
    written = config.params['Price Factors']['HestonNandiComponentModelParameters.ZAR']
    curve = written[utils.HN_COMPONENT_CURVE_NAME].array
    values = torch.stack([_tensor(v) for v in curve[:, 1]])
    days = max(int(round(curve[-1, 0] * SPY)), 1)
    l_path = utils.hn_component_l_path(curve[:, 0], values, days, SPY)
    omegas = utils.hn_component_omega_path(l_path, _tensor(written['Rho']))
    assert float(omegas.min()) >= 0.0, (
        'Floor wrote a curve with a negative omega: min %.6g' % float(omegas.min()))


def test_quote_sensitivity_refuses_by_name_on_this_family():
    """`Quote_Sensitivity: Yes` is REFUSED, with the reason and the alternative - not answered with
    zeros, and not silently ignored.

    The quote derivative would have to pass through the inner `brentq` on each L pillar by the
    implicit function theorem AND through the outer derivative-free search. The IFT half is
    tractable and is the arithmetic `CalibrationSolve.backward` already runs; the outer half is not
    a root find at all, so what a quote tick MEANS for the skew globals has to be decided before it
    can be computed. That is a roadmap row, and until it is built the family says so.

    THE REFUSAL COMES BEFORE ANY WORK, which is why this gate costs nothing: it is checked on the
    block, before a single factor is resolved, so a job that asked for the impossible finds out
    immediately rather than eight minutes later.
    """
    config = _built_surface(DESK_SMILE)
    name, block = COMPONENT.fx_surface_block(
        'USD.ZAR', config.params['Price Factors'], config.params['System Parameters'],
        config.params['Price Factor Interpolation'])
    block['instrument']['Quote_Sensitivity'] = 'Yes'
    family = COMPONENT({}, torch.device('cpu'), DTYPE)

    with pytest.raises(Exception) as refusal:
        family.bootstrap(config.params['System Parameters'], config.params['Price Models'],
                         config.params['Price Factors'],
                         config.params['Price Factor Interpolation'], {name: block}, {})
    message = str(refusal.value)
    assert 'Quote_Sensitivity' in message
    assert 'brentq' in message, 'the refusal does not name what carries no derivative'
    assert 'implicit function theorem' in message
    # THE REMEDY HAS TO BE TRUE, and the family it used to name is not one: the plain
    # HestonNandiModelPrices block declares no Quote_Sensitivity field at all, so "fit the plain
    # family, which is differentiable" sent a desk to a block that would ignore the switch. Read off
    # the declaration rather than from the message, so the assertion cannot go stale with it.
    from derivus.bootstrappers import HestonNandiModelParameters as PLAIN_FAMILY
    assert 'Quote_Sensitivity' not in [f.name for f in PLAIN_FAMILY.fields], (
        'the plain family now declares Quote_Sensitivity - the refusal may name it again')
    assert 'FXVolPrices' in message, 'a refusal without a quote chain that IS differentiable'
    assert 'roadmap' in message, 'the refusal does not say where the missing half is tracked'
    assert 'HestonNandiComponentModelParameters.ZAR' not in config.params['Price Factors'], (
        'the refusal still wrote a factor')


def test_a_rho_at_or_above_one_refuses_by_name():
    """`Rho` is READ, never fitted, and nothing between the field and the strip constrains it - so
    the read is where it has to be refused.

    WHAT A RHO >= 1 DOES, and why a wrong answer would be silent rather than loud: q_t is an AR(1)
    at Rho, so at Rho >= 1 the long-run component is non-stationary and `E_0[q_t] = L_t` - the
    identity that makes the fitted L the expected variance path - stops holding. Worse, the
    declining-variance floor `A(1 - (1-Rho)n/(1 + (n-1)(1-Rho)))` goes NEGATIVE at Rho > 1, and
    `low = max(floor, 1e-12)` then admits every level the bracket can reach: the negative-omega
    guard is DISABLED rather than tripped, and the family writes a curve whose own simulator drives
    the variance negative. That is the failure a gate has to reach.

    BEFORE ANY WORK, like the Quote_Sensitivity refusal: the read is above the fit, so a block with
    an impossible pin finds out immediately.
    """
    with pytest.raises(ValueError) as refusal:
        _fit(DESK_SMILE, Rho=1.02)
    message = str(refusal.value)

    assert 'Rho' in message, 'the refusal does not name the field'
    assert '1.02' in message, 'the refusal does not carry the value it was given'
    assert 'NON-STATIONARY' in message, 'the refusal does not give the stationarity reason'
    assert '[0, 1)' in message, 'the refusal does not state the admissible range'


# ======================================================================================
# GATE 6 - both leverage signs
# ======================================================================================

def test_the_negative_risk_reversal_lands_a_negative_leverage_at_an_interior_optimum(fitted):
    """The shape, not the numbers - the plain family's own precedent, on the same fixture.

    USDZAR's risk reversal is NEGATIVE in pair terms, which read on the `FxRate.ZAR` axis this
    model is fitted on is a smile whose vol RISES with strike. A one-signed leverage could only
    answer that by switching the channel off and reporting a flat smile it calls converged, which
    is exactly what the plain family did before its leverage share carried the sign. The component
    family inherits that reparametrisation, and this gate is what proves the inheritance is live
    rather than copied.

    INTERIOR, TOO. Every fitted coordinate has to sit strictly inside its box: a fit pinned on a
    bound is the signature of a model that cannot represent its own data. MEASURED at the declared
    300-evaluation cap: Alpha 2.720e-07, Beta 0.96409, Gamma_1 -845.12, Phi 5.438e-08, H0
    7.568e-05 - fitted coordinates (0.96409, -0.20152, 0.10010, 0.19990, -9.48903), whose smallest
    distance to a bound is 3.6% of that bound's own width (the persistence, which sits high because
    the strip it has to reproduce rises). At 400 evaluations the same fixture lands in a different
    basin - Gamma_1 -78.56, smallest margin 17.3% - and the SIGN is the thing both share, which is
    why that is what this holds.
    """
    written, _ = fitted
    assert written['Gamma_1'] < 0.0, (
        'a rising smile fitted a POSITIVE leverage - the sign-free reparametrisation is not live: '
        'Gamma_1 %r' % written['Gamma_1'])
    assert written['Gamma_2'] == written['Gamma_1'], 'Tie_Gamma_2 defaults to Yes'
    assert written['Rho'] == 0.99, 'Rho is PINNED to its declared default'

    x = COMPONENT.unreparam(written['Alpha'], written['Beta'], written['Gamma_1'],
                            written['Phi'], written['H0'])
    for value, (low, high), name in zip(
            x, COMPONENT.bounds, ('beta', 'leverage share', 'ARCH share', 'phi share', 'log H0')):
        margin = min(value - low, high - value) / (high - low)
        assert margin > 1e-6, (
            '%s sat on its bound (%.6g in [%.6g, %.6g]) - a fit that cannot represent its data'
            % (name, value, low, high))
    # and the leverage share is the coordinate carrying the sign
    assert x[1] < 0.0, 'the leverage share did not carry the negative sign'
    assert written['Alpha'] > 0.0 and written['Phi'] >= 0.0
    assert 0.0 < written['Beta'] < 1.0, 'the short-run persistence left the stationary region'


# ======================================================================================
# GATE 5 - BASEVAL + CREDIT MONTE CARLO
# ======================================================================================

TARF_SPOT = TARF_STRIKE = 18.5
TARF_N1 = 1_000_000.0
UNREACHABLE = 1.0e9

#: A fitted-looking component parameter set, authored directly into the gate's JSON. The gate is
#: about the PRICERS reaching this model, so the parameters are stated rather than fitted - an
#: eight-minute calibration inside a pricing gate would be measuring the wrong thing.
COMPONENT_FACTOR = {
    'Property_Aliases': None, 'Alpha': 3.5681e-06, 'Beta': 0.8138, 'Gamma_1': -64.992,
    'Rho': 0.99, 'Phi': 1.9820e-06, 'Gamma_2': -64.992, 'H0': 7.295e-05,
    'L_Curve': utils.Curve([], [[0.0, 7.295e-05], [1.0 / 12.0, 8.510e-05],
                                [2.0 / 12.0, 8.565e-05], [0.25, 9.383e-05], [0.5, 9.647e-05]])}


def _tarf_deal(fix_days, target=UNREACHABLE, buy_sell='Buy'):
    dates = [BASE + pd.Timedelta(days=d) for d in fix_days]
    return {'Object': 'FXTARFOptionDeal', 'Reference': 'TARF1', 'Currency': 'USD',
            'Underlying_Currency': 'ZAR', 'Discount_Rate': 'USD', 'FX_Volatility': 'USD.ZAR',
            'Buy_Sell': buy_sell, 'Expiry_Date': dates[-1], 'Underlying_Amount': TARF_N1,
            'Option_Type': 'Call', 'Strike_Price': TARF_STRIKE, 'Settlement_Style': 'Cash',
            'Option_Style': 'European', 'InvertedTarget': False, 'LeverageNotional': 0.0,
            'TargetLevel': target,
            'TARF_ExpiryDates': [[d, d, None] for d in dates]}


def _tarf_config(deal, counterparty=False, simulate=False):
    """The gate's JSON: the component parameter factor in `Price Factors`, the switch in
    `Valuation Configuration`, and - when the exposure profile is wanted - the component PROCESS
    driving the underlying in `Model Configuration`. Both seams in one document, which is what
    makes this a gate on the model reaching deals rather than on a function."""
    valuation = {'FXTARFOptionDeal': {'SpotModel': 'HestonNandiComponent',
                                      'Steps_Per_Year': SPY}}
    config = Config()
    config.params['System Parameters']['Base_Currency'] = 'USD'
    config.params['System Parameters']['Base_Date'] = BASE
    config.params['Price Factors'] = {
        'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Priority': 1,
                       'Spot': 1.0},
        'FxRate.ZAR': {'Domestic_Currency': 'USD', 'Interest_Rate': 'ZAR', 'Priority': 1,
                       'Spot': TARF_SPOT},
        'InterestRate.USD': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                             'Curve': utils.Curve([], [[0.0, 0.0], [5.0, 0.0]])},
        'InterestRate.ZAR': {'Currency': 'ZAR', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                             'Curve': utils.Curve([], [[0.0, 0.0], [5.0, 0.0]])},
        'VolatilityGrid.USD.ZAR': {
            'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness',
            'Surface': utils.Curve([], [[m, t, 0.14] for m in (0.5, 1.0, 1.5)
                                        for t in (0.02, 2.0)])},
        'HestonNandiComponentModelParameters.ZAR': dict(COMPONENT_FACTOR)}
    config.params['Price Models'] = {}
    if counterparty:
        config.params['Price Factors']['SurvivalProb.CPTY'] = {
            'Recovery_Rate': 0.4, 'Curve': utils.Curve([], [[0.0, 0.0], [10.0, 0.4]])}
    if simulate:
        # the underlying rides the COMPONENT process, which is the second seam: the same implied
        # factor the pricer consumes also drives the outer scenario evolution
        config.params['Model Configuration'].append(
            'FxRate', (), 'HestonNandiComponentImpliedSpotModel')
    config.params['Valuation Configuration'] = valuation
    config.deals = {'Attributes': {'Reference': 'test', 'Tag_Titles': ''},
                    'Deals': {'Children': [{'Instrument': construct_instrument(deal, valuation)}]},
                    'Calculation': {'Base_Date': BASE, 'Currency': 'USD'}}
    return config


@needs_hn_fused
def test_a_vanilla_prices_within_monte_carlo_error_of_the_component_closed_form():
    """BASEVAL against the closed form, on the ONE FX deal type that honours a spot model.

    A VANILLA IN TARF CLOTHING: one fixing, an unreachable target and no leveraged leg, so the
    payoff is exactly `N (S_T - K)^+` and the OSS machinery reduces to a plain daily walk to
    expiry. That is deliberate - `FXOptionDeal` prices Black off the surface and never reaches a
    spot model at all, so a vanilla priced UNDER this model has to be authored as the deal type
    that carries the switch. The closed form is `utils.hn_component_call` over the same daily step
    count from the same state, which is what the pricer's own kit seeds.

    MEASURED: 2^15 inner paths, seed 1 - baseval 432,731.78 against a closed form of 433,814.67,
    a -0.25% difference, and both finite.
    """
    days = 63
    deal = _tarf_deal([days])
    _, out = run_baseval(_tarf_config(deal), prec=DTYPE,
                         overrides={'MCMC_Simulations': 1 << 15, 'Random_Seed': 1})
    rows = out['Results']['mtm']
    priced = float(rows[rows['Reference'] == 'TARF1']['Value'].iloc[0])
    assert np.isfinite(priced), 'the deal was skipped or produced a non-finite value'

    n = max(int(round(days / 365.0 * SPY)), 1)
    params = tuple(_tensor(COMPONENT_FACTOR[k]) for k in
                   ('Alpha', 'Beta', 'Gamma_1', 'Rho', 'Phi', 'Gamma_2'))
    curve = COMPONENT_FACTOR['L_Curve'].array
    values = torch.stack([_tensor(v) for v in curve[:, 1]])
    l_path = utils.hn_component_l_path(curve[:, 0], values, n, SPY)
    omegas = list(utils.hn_component_omega_path(l_path, params[3]))
    closed = float(utils.hn_component_call(
        TARF_SPOT, _tensor(TARF_STRIKE), omegas, _tensor(COMPONENT_FACTOR['H0']), l_path[0],
        *params, _tensor(0.0), panels=64)) * TARF_N1

    # the OSS inner MC's own standard error on this payoff is ~0.4% at 2^15 paths; 2% is that with
    # room, and a pricer that had fallen back to GBM at the surface's 14% flat vol would read
    # ~4% out on this contract, which is what the bound has to separate
    assert priced == pytest.approx(closed, rel=0.02), (
        'baseval %.2f vs component closed form %.2f (%.2f%%)'
        % (priced, closed, 100.0 * (priced / closed - 1.0)))


def test_the_kit_answers_a_zero_length_walk_before_it_has_a_strip():
    """THE FIRST CALL CAN ASK FOR NOTHING, and it must not be the one that skips the deal.

    A fixing interval of one trading day or less walks `n_sub - 1 = 0` unmonitored sub-steps, so
    all four OSS pricers can open a row with `omegas(0, 0)` - and the strip is built LAZILY, on a
    length test (`self._built < day + n_steps`) that `0 < 0` answers False. What came back was a
    slice of None: a TypeError inside `Deal.calculate`, which swallows it, so the symptom is a NaN
    Value on a row that still looks priced rather than a stack trace. An MTM row one day before a
    fixing is enough to reach it.

    The gate holds the kit's own contract: an empty strip out, the strip actually built behind it,
    and the same intercepts a fully warmed kit answers.
    """
    curve = COMPONENT_FACTOR['L_Curve'].array
    scalars = [_tensor(COMPONENT_FACTOR[k]).reshape(-1, 1)
               for k in utils.HN_COMPONENT_PARAM_NAMES] + [
        torch.stack([_tensor(v) for v in curve[:, 1]])]
    kit = ComponentHestonNandiKit(scalars, curve[:, 0], SPY)

    _, _, day = kit.seed()
    assert day == 0, 'the row does not open at trading day zero'
    empty = kit.omegas(0, 0)
    assert empty.numel() == 0, 'a zero-length walk did not answer an empty strip'

    # and the strip behind it is real: the intercepts the curve itself differences to
    expected = utils.hn_component_omega_path(
        utils.hn_component_l_path(curve[:, 0], scalars[-1], 21, SPY),
        _tensor(COMPONENT_FACTOR['Rho']))
    assert torch.equal(kit.omegas(0, 21), expected), (
        'the strip built behind the zero-length call is not omega_t = L_(t+1) - rho*L_t')
    warm = ComponentHestonNandiKit(scalars, curve[:, 0], SPY)
    assert torch.equal(kit.omegas(0, 21), warm.omegas(0, 21)), (
        'the strip built on the zero-length call differs from a longer walk\'s own')


def test_a_tarf_whose_first_fixing_is_one_trading_day_out_prices_finite():
    """THE PRICING HALF of the zero-length walk, through the real entry point.

    A one-fixing TARF one calendar day after the base date rounds to `n_sub = 1` daily step, so the
    pricer asks its kit for ZERO unmonitored sub-steps before it takes the single monitored one.
    MEASURED UNDER THE DEFECT: no raise reaches the caller - `Deal.calculate` swallows the
    TypeError - and the row is still THERE, carrying a Value of NaN. So the gate asserts the row
    AND the number: a deal that dies inside the walk is indistinguishable from one that priced,
    until something downstream sums the frame.

    NO torch.compile PRECONDITION, unlike the 63-day gate beside it: an interval that walks no
    unmonitored sub-steps never reaches the fused log sub-step at all.
    """
    _, out = run_baseval(_tarf_config(_tarf_deal([1])), prec=DTYPE,
                         overrides={'MCMC_Simulations': 1 << 12, 'Random_Seed': 1})
    rows = out['Results']['mtm']
    priced = rows[rows['Reference'] == 'TARF1']['Value']
    assert len(priced) == 1, 'the deal was SKIPPED - the zero-length walk killed the row'
    assert np.isfinite(float(priced.iloc[0])) and float(priced.iloc[0]) > 0.0, (
        'a one-day call struck at the money is worth something positive: %r' % priced.iloc[0])


@needs_hn_fused
def test_a_tarf_carries_an_exposure_profile_and_a_finite_cva_under_the_component_model():
    """CREDIT MONTE CARLO, which is the half base valuation structurally cannot reach: one row per
    report date, one column per path, the component PROCESS driving the underlying and the
    component PRICER valuing the deal on every row.

    THE FAILURE MODE THIS EXISTS TO CATCH IS THE DEAL BEING SKIPPED. `Deal.calculate` swallows a
    pricer exception into a skipped deal, which surfaces only as a missing row much later - so a
    profile that collapsed to one row, or one with no dispersion, or a non-finite CVA, all mean the
    same thing: the model did not actually reach the deal. A six-fixing TARF is used rather than
    the one-fixing vanilla precisely because it walks the OSS truncation and the target latch on
    every row.

    THE DEAL IS BOUGHT, and that is not cosmetic: CVA is an expectation of POSITIVE exposure, so a
    sold TARF (whose profile is negative on every path here) reports a perfectly finite CVA of
    exactly zero and the gate would pass on a model that never ran. Bought, with no leveraged leg,
    the profile is positive and the CVA is a number the model actually produced.

    MEASURED: 18 rows x 64 paths, the smallest per-row dispersion 23,885 on a row-zero mean of
    344,913, and CVA 2,777.04 - finite and non-zero.
    """
    deal = _tarf_deal([60 * (i + 1) for i in range(6)], target=0.5)
    _, out = derivus.run_cmc(
        _tarf_config(deal, counterparty=True, simulate=True), prec=DTYPE,
        overrides={'Run_Date': BASE.strftime('%Y-%m-%d'), 'Time_grid': '0d 1m(1m)',
                   'Batch_Size': 64, 'Simulation_Batches': 1, 'Random_Seed': 1,
                   'Currency': 'USD', 'MCMC_Simulations': 64, 'Tenor_Offset': 0.0,
                   'Deflation_Interest_Rate': 'USD',
                   'Credit_Valuation_Adjustment': {
                       'Calculate': 'Yes', 'Counterparty': 'CPTY',
                       'Deflate_Stochastically': 'No', 'Stochastic_Hazard_Rates': 'No',
                       'Gradient': 'No'}})
    mtm = out['Results']['mtm']

    assert mtm.shape[0] > 1, 'the exposure profile collapsed to one row - deal skipped?'
    assert np.isfinite(mtm.values).all(), 'NaN in the exposure profile'
    dispersion = mtm.values.std(axis=1)
    assert (dispersion[1:] > 0.0).all(), (
        'a row with no dispersion across paths - the component process is not driving the spot')
    cva = float(out['Results']['cva'])
    assert np.isfinite(cva) and cva != 0.0, 'CVA %r' % cva


# ======================================================================================
# THE RECIPROCAL CARRY - the second exact parameter map, and the one an FX book runs on
# ======================================================================================

#: The three laws the carry is read on. The USDZAR-shaped fit is the one an FX book actually
#: carries; the equity-shaped one has the opposite skew sign, so nothing here can be passing on the
#: sign of `gamma_star` alone; the third has no leverage at all, which is the degenerate face where
#: the whole map reduces to the unit shift and nothing else can hide it.
CARRY_LAWS = (('a USDZAR-shaped fit', 1e-12, 2.0e-6, 0.45, -474.34, 7.8e-5),
              ('an equity-shaped fit', 1e-6, 4.0e-6, 0.60, 120.0, 9.0e-5),
              ('no leverage at all', 1e-6, 3.0e-6, 0.70, 0.0, 8.0e-5))


def test_the_reciprocal_carry_is_the_fx_option_symmetry_in_closed_form():
    """`utils.hn_reciprocal_gamma` in CLOSED FORM, at machine precision, with no Monte Carlo in it.

    An `FxRate` is a currency priced in the BASE, so a deal whose underlying IS the base pays on the
    reciprocal of the only leg the calibration can fit AND settles in the other currency. That is a
    change of NUMERAIRE, and the FxRate is itself the density that performs it - which shifts the
    innovation by exactly one standard deviation and turns `(omega, alpha, beta, gamma*)` for `s`
    into `(omega, alpha, beta, 1 - gamma*)` for `1/s`. One law, two currencies, one parameter.

    THE IDENTITY IS THE FX OPTION SYMMETRY. At zero rates both numeraires discount at one, and a put
    on `s` struck `K` is a call on `1/s` struck `1/K` with the notional converted at the strike:

        (1 / K) * E_USD[(K - s)^+]  ==  s0 * E_EUR[(1/s - 1/K)^+]

    Both sides are `utils.hn_call`'s own Fourier inversion, so there is no estimator between the map
    and the claim and the tolerance is arithmetic rather than statistical.

    MEASURED, worst over three laws x three strikes: 1.4e-12 carried. The map is not
    over-determined by luck either - `-gamma*` (the sign flip without the unit shift) misses by up
    to 3.2e-2 and the uncarried `gamma*` by up to 5.2, so both halves of the map are pinned here.
    The Monte Carlo consistency gates cannot do that: on
    `test_fx_accumulator_json.py`'s fixture the unit shift alone is worth 5e-4, under the floor.
    """
    spot, n_steps = 1.1, 63
    # the two WRONG candidates, named for what each one omits: `-gamma*` takes the sign flip and
    # drops the unit shift, `gamma*` takes neither
    worst = worst_unshifted = worst_uncarried = 0.0
    for label, om, al, be, ga, h0 in CARRY_LAWS:
        law = dict(n_steps=n_steps, h1=_tensor(h0), omega=_tensor(om), alpha=_tensor(al),
                   beta=_tensor(be), r=_tensor(0.0))
        for strike in (0.95 * spot, spot, 1.10 * spot):
            direct = float(utils.hn_put(
                _tensor(spot), _tensor(strike), gamma_star=_tensor(ga), **law)) / strike

            def mirror(gamma):
                return spot * float(utils.hn_call(
                    _tensor(1.0 / spot), _tensor(1.0 / strike),
                    gamma_star=_tensor(gamma), **law))

            carried = mirror(float(utils.hn_reciprocal_gamma(_tensor(ga))))
            assert carried == pytest.approx(direct, rel=1e-10), (
                '%s at K/S %.2f: carried %.14e vs direct %.14e' % (label, strike / spot,
                                                                   carried, direct))
            worst = max(worst, abs(carried / direct - 1.0))
            worst_unshifted = max(worst_unshifted, abs(mirror(-ga) / direct - 1.0))
            worst_uncarried = max(worst_uncarried, abs(mirror(ga) / direct - 1.0))

    assert worst < 1e-10, 'the carry degraded to %.3e' % worst
    # and both halves of the map are load-bearing on this grid
    assert worst_unshifted > 1e-3, (
        'the UNIT SHIFT is not resolved by this grid: -gamma*, which takes only the sign flip, '
        'misses by %.3e' % worst_unshifted)
    assert worst_uncarried > 1e-2, (
        'the CARRY is not resolved by this grid: an uncarried gamma* misses by %.3e'
        % worst_uncarried)
