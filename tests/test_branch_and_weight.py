"""Branch-and-weight: the primitives, then the four products on real documents.

The construction: at each fixing the trigger standardises to `zB`; the FIRED branch closes
analytically with weight `1 - Phi(zB)` and payoff `E[J(S_k) | fired]`, the CONTINUING branch draws
`S_k` from the truncated law by `Phi^-1(U * Phi(zB))` carrying weight `Phi(zB)`. Unbiased, not a
smoothing.

Pinned here: the fired branch is a conditional expectation, never `p x realised payoff`; the switch
is declared on `Base_Revaluation` alone and off is the crisp path bit for bit; GBM only, and a
non-GBM spot model refuses by name.

Products: TARF, accumulator, discrete barrier, autocall - each against a differentiable trapezoid
reference written in this file out of `math`/`torch` alone, so an error in the closed form cannot
hide behind the same error in its oracle. Readings are in each gate's own docstring.
"""
import datetime
import inspect
import io
import json
import logging
import math
import os
import sys

# reference-derivus shadow-import guard (MEMORY): pin the package under test to THIS repo.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import pytest
import torch

import derivus as rf
from derivus import calculation, instruments, pricing, schema, utils
from derivus.instruments import construct_instrument

# the barrier's world and deal, borrowed from the file that already gates them
import test_barrier_bridge as bb
from crn_ladder import ladder

TARF_TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'fixtures', 'fx_tarf_job.json')

DT = torch.float64


# ======================================================================================
# References. Written out of `math` alone, so they share nothing with the engine.
# ======================================================================================

def _Phi(x):
    """The standard normal CDF, from `math.erf` - not `utils.norm_cdf`, which is what is on test."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _phi(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _quadrature(f, lo, hi, n=400001):
    """Trapezoid over [lo, hi] on an odd node count."""
    h = (hi - lo) / (n - 1)
    total = 0.5 * (f(lo) + f(hi))
    for i in range(1, n - 1):
        total += f(lo + i * h)
    return total * h


def _gaussian_tail(payoff, z_lo, z_hi):
    """``E[payoff(Z) * 1{z_lo < Z < z_hi}]`` by quadrature; +-12 stands in for infinity (density
    2e-33 there, forty times under the tolerances below)."""
    return _quadrature(lambda z: payoff(z) * _phi(z), max(z_lo, -12.0), min(z_hi, 12.0))


def _spot(S, m, s, z):
    return S * math.exp(m + s * z)


# one fixture for every closed-form gate below, nothing in it degenerate: non-zero drift, trigger
# above the strike (a TARF's moving barrier always is), a fired tail carrying real mass
S0, DRIFT, VOL, STRIKE_K = 1.2500, 0.0170, 0.1400, 1.2000
Z_TRIGGER = (math.log(1.3100 / S0) - DRIFT) / VOL     # B = 1.31, an up trigger


# ======================================================================================
# THE TRUNCATED DRAW - ONE SPELLING (`pricing.oss_truncated_draw`)
# ======================================================================================

def _stratified(n):
    """Midpoint uniforms. Deterministic, so no seed and no tolerance absorbing one: the quantile
    transform of a midpoint grid errs as 1/n^2 rather than 1/sqrt(n)."""
    return torch.arange(n, dtype=DT).add(0.5).div(n)


def _truncated_normal_moments(a, below):
    """Mean and variance of `Z | Z <= a` (below) or `Z | Z >= a`, in closed form."""
    if below:
        lam = -_phi(a) / _Phi(a)
        second = 1.0 - a * _phi(a) / _Phi(a)
    else:
        lam = _phi(a) / (1.0 - _Phi(a))
        second = 1.0 + a * _phi(a) / (1.0 - _Phi(a))
    return lam, second - lam * lam


@pytest.mark.parametrize('below', [True, False])
@pytest.mark.parametrize('a', [-1.7, -0.4, 0.0, 0.9, 2.3])
def test_the_truncated_draw_samples_the_truncated_law(a, below):
    """The draw's SAMPLE moments against the truncated normal's ANALYTIC ones, to 2e-4.

    A draw merely on the right SIDE of the barrier - rejection, clamp, reflection - passes every
    sign test and fails here.
    """
    u = _stratified(200000)
    p, Z = pricing.oss_truncated_draw(u, torch.tensor(a, dtype=DT), below)
    mean, var = float(Z.mean()), float(Z.var(unbiased=False))
    ref_mean, ref_var = _truncated_normal_moments(a, below)
    assert abs(mean - ref_mean) < 2e-4, (a, below, mean, ref_mean)
    assert abs(var - ref_var) < 2e-4, (a, below, var, ref_var)
    # and every drawn point is on the surviving side of the bound, which the moments alone
    # would not catch on a symmetric error
    assert bool((Z <= a).all() if below else (Z >= a).all())


@pytest.mark.parametrize('a', [-1.7, 0.0, 2.3])
def test_the_survival_probability_is_the_bound_s_own_cdf(a):
    """`p` is `Phi(zB)` one side and its complement the other - the weight the alive path carries."""
    z = torch.tensor(a, dtype=DT)
    assert abs(float(pricing.oss_truncated_draw(_stratified(8), z, True)[0]) - _Phi(a)) < 1e-14
    assert abs(float(pricing.oss_truncated_draw(_stratified(8), z, False)[0]) -
               (1.0 - _Phi(a))) < 1e-14


def test_the_barriers_inline_spelling_still_matches_the_seam():
    """SIBLING GATE. `sim_spot_oss` writes the idiom inline; this holds the copy to the seam.

    The UP arm is bit-for-bit. The DOWN arm's base is `(1 - p) + u*p` against the seam's
    `Phi + u*p`, and `1 - (1 - Phi)` recovers `Phi` exactly only at or above a half (Sterbenz) -
    so both the agreement above a half and the disagreement below it are asserted, and absorbing
    the site turns this red rather than passing silently (it re-baselines the barrier fixtures).
    """
    eps = torch.finfo(DT).eps
    u = _stratified(4096).reshape(-1, 1)
    z = torch.linspace(-3.0, 3.0, 97, dtype=DT).reshape(1, -1)

    # --- the UP arm, verbatim from sim_spot_oss ---
    p_inline = utils.norm_cdf(z)
    Z_inline = utils.norm_icdf(torch.clamp(u * p_inline, eps, 1.0 - eps))
    p_seam, Z_seam = pricing.oss_truncated_draw(u, z, True)
    assert torch.equal(p_inline.expand_as(p_seam), p_seam)
    assert torch.equal(Z_inline, Z_seam)

    # --- the DOWN arm, verbatim from sim_spot_oss ---
    p_inline = 1.0 - utils.norm_cdf(z)
    Z_inline = utils.norm_icdf(torch.clamp((1.0 - p_inline) + u * p_inline, eps, 1.0 - eps))
    p_seam, Z_seam = pricing.oss_truncated_draw(u, z, False)
    assert torch.equal(p_inline.expand_as(p_seam), p_seam)

    above = (utils.norm_cdf(z) >= 0.5).expand_as(Z_seam)
    assert torch.equal(Z_inline[above], Z_seam[above]), 'the down arm must agree at or above a half'
    below = ~above
    gap = (Z_inline[below] - Z_seam[below]).abs()
    assert float(gap.max()) > 0.0, (
        'the enumerated last-bit difference has gone: either sim_spot_oss was absorbed into '
        'oss_truncated_draw - in which case re-baseline the barrier fixtures and retire this half '
        'of the gate - or the seam changed spelling')
    # bounded, so what is enumerated is a last-bit difference and not a second estimator
    assert float((gap / Z_seam[below].abs().clamp(min=1e-12)).max()) < 1e-9


def test_the_autocalls_inline_spelling_is_bit_identical_to_the_seam():
    """SIBLING GATE. `pv_MC_AutoCallSwap` writes `clamp(p * u)`, the seam `clamp(u * p)` -
    bit-identical. The enumeration's obstacle is SHAPE: `p` is formed in the coupon branch and
    consumed a screen later under `fixing_aligned`, where the draw is skipped.
    """
    eps = torch.finfo(DT).eps
    u = _stratified(2048).reshape(-1, 1)
    z = torch.linspace(-3.0, 3.0, 61, dtype=DT).reshape(1, -1)
    p_inline = utils.norm_cdf(z)
    Z_inline = utils.norm_icdf(torch.clamp(p_inline * u, min=eps, max=1.0 - eps))
    p_seam, Z_seam = pricing.oss_truncated_draw(u, z, True)
    assert torch.equal(p_inline.expand_as(p_seam), p_seam)
    assert torch.equal(Z_inline, Z_seam)


# ======================================================================================
# THE FIRED BRANCH IS AN EXPECTATION (`pricing.lognormal_partial_moment` / `_fired_gain`)
# ======================================================================================

def _t(x):
    return torch.tensor(x, dtype=DT)


@pytest.mark.parametrize('power', [0.0, 1.0, -1.0])
@pytest.mark.parametrize('fired_above', [True, False])
def test_the_partial_moments_match_a_quadrature_written_here(power, fired_above):
    """`E[S^a 1{fired}]` against a trapezoid over the normal density, to 1e-9 relative.

    `power=1` is a full-gain payment, `power=-1` an `InvertedTarget` accrual, `power=0` the fired
    probability - one expression, one gate.
    """
    got = float(pricing.lognormal_partial_moment(
        _t(S0), _t(DRIFT), _t(VOL), _t(Z_TRIGGER), fired_above, power))
    lo, hi = (Z_TRIGGER, 12.0) if fired_above else (-12.0, Z_TRIGGER)
    ref = _gaussian_tail(lambda z: _spot(S0, DRIFT, VOL, z) ** power, lo, hi)
    assert abs(got - ref) < 1e-9 * max(abs(ref), 1e-3), (power, fired_above, got, ref)


def test_the_zeroth_moment_is_the_fired_probability():
    """`power=0` is `1 - p` to 1e-15. The ledger spells it as the complement instead, which is what
    makes the telescoping identity below exact rather than merely close."""
    z = _t(Z_TRIGGER)
    fired = float(pricing.lognormal_partial_moment(_t(S0), _t(DRIFT), _t(VOL), z, True, 0.0))
    survive = float(pricing.oss_truncated_draw(_stratified(8), z, True)[0])
    assert abs(fired - (1.0 - survive)) < 1e-15
    assert abs(fired - (1.0 - _Phi(Z_TRIGGER))) < 1e-15


@pytest.mark.parametrize('power', [1.0, -1.0])
def test_the_two_tails_sum_to_the_unconditional_moment(power):
    """Fired plus survived is the whole moment, to 1e-14 relative - conservation in moment space,
    and the one identity a sign error in the reflection cannot survive."""
    args = (_t(S0), _t(DRIFT), _t(VOL), _t(Z_TRIGGER))
    both = float(pricing.lognormal_partial_moment(*args, True, power) +
                 pricing.lognormal_partial_moment(*args, False, power))
    whole = S0 ** power * math.exp(power * DRIFT + 0.5 * power * power * VOL * VOL)
    assert abs(both - whole) < 1e-14 * whole, (power, both, whole)


def test_the_full_gain_is_black_when_the_trigger_never_binds():
    """Trigger at minus infinity: the full gain collapses to the interval forward less the strike,
    to 1e-13. The one place a missing `exp(0.5 s^2)` would show."""
    got = float(pricing.lognormal_fired_gain(
        _t(S0), _t(DRIFT), _t(VOL), _t(-40.0), _t(STRIKE_K), True))
    forward = S0 * math.exp(DRIFT + 0.5 * VOL * VOL)
    assert abs(got - (forward - STRIKE_K)) < 1e-13, (got, forward - STRIKE_K)


def test_the_full_gain_matches_a_quadrature_written_here():
    got = float(pricing.lognormal_fired_gain(
        _t(S0), _t(DRIFT), _t(VOL), _t(Z_TRIGGER), _t(STRIKE_K), True))
    ref = _gaussian_tail(lambda z: _spot(S0, DRIFT, VOL, z) - STRIKE_K, Z_TRIGGER, 12.0)
    assert abs(got - ref) < 1e-9 * abs(ref), (got, ref)


def test_p_times_the_realised_payoff_is_biased():
    """THE BIAS TRAP. `p_fired x J(S_sampled)` evaluates the payoff on the SURVIVAL-truncated draw
    and weights it by the FIRED probability. Both halves here are exact, so the gap is bias: it
    does not shrink with paths.

    Fixture spot 1.2500, interval drift 0.0170, vol 0.1400, up trigger 1.3100 (`z = 0.213454`,
    `p_fired = 0.415486`), strike 1.2000:

        FULL GAIN     truth 0.105802   shortcut 0.009525   11.11x LOW
        CAPPED/EXACT  truth 0.024929   shortcut 0.007585    3.29x LOW

    Written with the RAW gain it reads -0.015531 - a SIGN error: truncating the upper 41.5% drags
    the surviving mean to 1.16257, below the strike. These are the two shapes the TARF ships.
    """
    args = (_t(S0), _t(DRIFT), _t(VOL), _t(Z_TRIGGER))
    p_fired = float(pricing.lognormal_partial_moment(*args, True, 0.0))
    survived = 1.0 - p_fired

    def on_the_surviving_draw(payoff):
        """What the shortcut pays: the FIRED probability times the payoff on the surviving draw."""
        return p_fired * _gaussian_tail(payoff, -12.0, Z_TRIGGER) / survived

    # FULL GAIN: the fired payment is the Black-type partial expectation
    truth = float(pricing.lognormal_fired_gain(*args, _t(STRIKE_K), True))
    shortcut = on_the_surviving_draw(lambda z: max(_spot(S0, DRIFT, VOL, z) - STRIKE_K, 0.0))
    assert truth > 0.0 and shortcut > 0.0
    assert truth / shortcut > 5.0, (
        'the full-gain bias trap has stopped biting, so this gate no longer defends the ruling: '
        '{} against {}'.format(shortcut, truth))

    # CAPPED / EXACT: the fired payment is the remaining target, a per-path constant
    remaining = 0.06
    capped = p_fired * remaining
    capped_shortcut = on_the_surviving_draw(
        lambda z: min(max(_spot(S0, DRIFT, VOL, z) - STRIKE_K, 0.0), remaining))
    assert capped / capped_shortcut > 2.0, (capped_shortcut, capped)

    # and the raw-gain spelling does not merely understate, it changes sign
    assert on_the_surviving_draw(lambda z: _spot(S0, DRIFT, VOL, z) - STRIKE_K) < 0.0 < truth


# ======================================================================================
# THE SURVIVAL LEDGER (`pricing.SurvivalLedger`)
# ======================================================================================

def _ledger_walk(probs, alive0):
    ledger = pricing.SurvivalLedger(alive0)
    for p in probs:
        ledger.fire(p)
    return ledger


def test_the_ledger_telescopes_to_its_starting_weight():
    """CONSERVATION: `sum_j alive_{j-1} * (1 - p_j) + alive_final == alive_0`, per path.

    Exact only because the fired mass is `1 - p` off the SAME `p` the draw was truncated with. The
    bar scales with the strip length - one rounding per fixing.
    """
    torch.manual_seed(20260830)
    probs = [torch.rand(8, 64, dtype=DT) for _ in range(12)]
    ledger = _ledger_walk(probs, torch.ones(8, 64, dtype=DT))
    residual = float(ledger.conservation().abs().max())
    assert residual < len(probs) * torch.finfo(DT).eps * 4, residual


def test_the_ledger_telescopes_off_a_partly_resolved_start():
    """A block opening on a partly-killed prefix starts at `prev_alive`, not at one. A ledger that
    hard-codes 1.0 passes the gate above and loses the prefix here."""
    torch.manual_seed(20260830)
    alive0 = (torch.rand(8, 64, dtype=DT) > 0.3).to(DT)   # an observed prefix: exact 0/1
    probs = [torch.rand(8, 64, dtype=DT) for _ in range(6)]
    ledger = _ledger_walk(probs, alive0)
    assert float(ledger.conservation().abs().max()) < len(probs) * torch.finfo(DT).eps * 4


def test_the_ledger_telescopes_through_observed_fixings():
    """An OBSERVED fixing fires with an exact 0/1 indicator rather than a `Phi`; same arithmetic."""
    torch.manual_seed(20260830)
    probs = [(torch.rand(4, 32, dtype=DT) > 0.5).to(DT) if k % 2 else
             torch.rand(4, 32, dtype=DT) for k in range(8)]
    ledger = _ledger_walk(probs, torch.ones(4, 32, dtype=DT))
    assert float(ledger.conservation().abs().max()) < len(probs) * torch.finfo(DT).eps * 4


def test_a_strip_that_never_fires_keeps_all_its_weight():
    ledger = _ledger_walk([torch.ones(4, dtype=DT)] * 5, torch.ones(4, dtype=DT))
    assert torch.equal(ledger.alive, torch.ones(4, dtype=DT))
    assert float(ledger.fired.abs().max()) == 0.0
    assert float(ledger.conservation().abs().max()) == 0.0


def test_a_strip_with_no_fixings_conserves_trivially():
    """A block with no simulated fixing has fired nothing, and must say so without needing a zero
    of the right shape."""
    ledger = pricing.SurvivalLedger(torch.ones(3, dtype=DT))
    assert ledger.fired is None
    assert float(ledger.conservation().abs().max()) == 0.0


def test_the_ledger_reports_its_residual_at_debug():
    """The identity is a DEBUG line, not an assert in a hot path."""
    buf, root = io.StringIO(), logging.getLogger()
    handler = logging.StreamHandler(buf)
    root.addHandler(handler)
    old = root.level
    root.setLevel(logging.DEBUG)
    try:
        _ledger_walk([torch.full((4,), 0.7, dtype=DT)] * 3, torch.ones(4, dtype=DT)).check('gate')
    finally:
        root.removeHandler(handler)
        root.setLevel(old)
    assert 'LEDGER gate conservation=' in buf.getvalue(), buf.getvalue()


# ======================================================================================
# THE PER-FIXING KINK (`pricing.kink_kernel` / `pricing.accrual_kink_term`)
# ======================================================================================

def _inline_kink_reference(Gbar, axis):
    """The kernel as `exposure_kink_term` spelled it before `kink_kernel` was factored out.

    A deliberate copy: editing it to keep the gate green means the factoring changed a number.
    """
    n = Gbar.shape[axis]
    spread = Gbar.std(dim=axis, keepdim=True) if n > 1 else torch.zeros_like(
        Gbar.narrow(axis, 0, 1))
    eps = 1.06 * spread * n ** -0.2
    floor = pricing.KINK_ATOM_BANDWIDTH_FLOOR * Gbar.abs().mean(dim=axis, keepdim=True)
    collapsed = eps <= floor
    unit = torch.ones_like(eps)
    width = torch.where(collapsed, unit, eps)
    return torch.where(collapsed, torch.zeros_like(eps), torch.exp(
        -0.5 * (Gbar / width) ** 2) / (width * math.sqrt(2.0 * math.pi)))


def test_the_factoring_moved_no_bits():
    """`kink_kernel` is `exposure_kink_term`'s machinery lifted out, not a rewrite: an identical
    tensor on every shape that reached the original."""
    torch.manual_seed(11)
    for shape in [(4, 4096), (1, 1024), (7, 2048)]:
        V = torch.randn(*shape, dtype=DT) * torch.linspace(0.2, 3.0, shape[0], dtype=DT).reshape(-1, 1)
        kernel, _, _ = pricing.kink_kernel(V, 1, 'gate')
        assert torch.equal(kernel, _inline_kink_reference(V, 1)), shape


def test_the_exposure_term_is_the_shared_kernel_and_nothing_else():
    """The term IS `0.5 * K * u^2` off the shared kernel, stated bitwise so the two cannot drift
    apart while both stay individually plausible."""
    torch.manual_seed(12)
    V = (torch.randn(5, 4096, dtype=DT) * 1.3).requires_grad_(True)
    kernel, _, _ = pricing.kink_kernel(V.detach(), 1, 'gate')
    u = V - V.detach()
    assert torch.equal(pricing.exposure_kink_term(V), 0.5 * kernel * u * u)


def test_the_accrual_kink_is_zero_at_value_and_bit_identically_zero_at_first_order():
    """Term on versus off: `np.array_equal` at value AND at first order. `u` is an exact IEEE zero,
    so the value is an exact zero and the gradient accumulates `+0.0` bit for bit."""
    torch.manual_seed(13)
    theta = torch.tensor([1.2500, 0.1400], dtype=DT, requires_grad=True)
    draws = torch.randn(3, 4096, dtype=DT)

    def gain(with_term):
        spot = theta[0] * torch.exp(theta[1] * draws)
        g = spot - STRIKE_K
        value = torch.relu(g).mean()
        return value + pricing.accrual_kink_term(g, 0).mean() if with_term else value

    off = gain(False)
    grad_off, = torch.autograd.grad(off, theta)
    on = gain(True)
    grad_on, = torch.autograd.grad(on, theta)
    assert np.array_equal(off.detach().numpy(), on.detach().numpy())
    assert np.array_equal(grad_off.numpy(), grad_on.numpy())


def test_the_accrual_kink_carries_the_missing_curvature():
    """And not zero at SECOND order: the term adds the kernel estimate of
    `f_g(0) * E[g_theta g_theta^T | g = 0]`, which the crisp relu reports as an exact zero at any
    path count."""
    torch.manual_seed(14)
    theta = torch.tensor([1.2500], dtype=DT, requires_grad=True)
    draws = torch.randn(1, 65536, dtype=DT) * 0.14

    def curvature(with_term):
        g = theta[0] * torch.exp(draws) - STRIKE_K
        value = torch.relu(g).mean()
        if with_term:
            value = value + pricing.accrual_kink_term(g, 0).mean()
        first, = torch.autograd.grad(value, theta, create_graph=True)
        return float(torch.autograd.grad(first, theta)[0])

    assert curvature(False) == 0.0, 'the pathwise gamma of a relu is an exact zero, or this gate ' \
                                    'is measuring something else'
    assert curvature(True) > 0.0


def test_the_accrual_kink_refuses_an_atom_by_name():
    """A fixing whose conditional law carries a POINT MASS at the strike is refused by name: the
    density climbs as `1/h` rather than settling. The crisp path's zero there is a smaller number,
    not a smaller error."""
    torch.manual_seed(15)
    gain = torch.ones(1, 65536, dtype=DT)
    gain[0, :64] = 0.0                     # an atom of weight ~1e-3 exactly at the strike
    with pytest.raises(utils.SecondOrderRefused) as refusal:
        pricing.accrual_kink_term(gain, 3)
    message = str(refusal.value)
    assert 'accrual_kink_term' in message and 'ATOM' in message, message
    assert 'fixing 3' in message, message
    assert 'Branch_And_Weight' in message, 'a refusal names its remedy'


def test_the_exposure_refusal_still_names_itself_after_the_factoring():
    """The refusal PROSE stayed with its caller: the remedies at a reporting row are not the
    remedies at a fixing."""
    torch.manual_seed(15)
    row = torch.ones(1, 65536, dtype=DT)
    row[0, :64] = 0.0
    with pytest.raises(utils.SecondOrderRefused) as refusal:
        pricing.exposure_kink_term(row)
    message = str(refusal.value)
    assert 'exposure_kink_term' in message and 'ATOM' in message, message
    assert 'reporting row' in message, message


def test_a_kink_with_no_atom_is_admitted():
    """Anti-placebo: an ordinary density must NOT refuse, or the gates above pass for nothing."""
    torch.manual_seed(16)
    smooth = torch.randn(1, 65536, dtype=DT)
    _, atom, _ = pricing.kink_kernel(smooth, 1, 'gate')
    assert not bool(atom.any())
    assert float(pricing.accrual_kink_term(smooth, 0).abs().max()) == 0.0


# ======================================================================================
# THE SWITCH (`Branch_And_Weight`)
# ======================================================================================

def _declared(cls):
    return {f.key: f for f in cls.fields}


def test_the_switch_is_declared_on_base_valuation_alone():
    """Declared where second derivatives live and nowhere else. `Credit_Monte_Carlo` not declaring
    it is load-bearing: the crisp path's exposure, cashflow and collateral semantics have no key
    an author could write to reach them.
    """
    field = _declared(calculation.Base_Revaluation)['Branch_And_Weight']
    assert field.type == 'Text' and field.values == ['Yes', 'No'], (field.type, field.values)
    assert 'Branch_And_Weight' not in _declared(calculation.Credit_Monte_Carlo)
    assert 'Branch_And_Weight' not in _declared(calculation.HedgeMonteCarlo)
    emitted = schema.mapping['Calculation']['types']
    assert 'Branch_And_Weight' in emitted['BaseValuation']
    assert 'Branch_And_Weight' not in emitted['CreditMonteCarlo']
    assert 'Branch_And_Weight' not in emitted['HedgeMonteCarlo']


def test_the_switch_defaults_to_no():
    """The declaration is the single source of the default, so an omitted key and an explicit 'No'
    are the same run."""
    assert _declared(calculation.Base_Revaluation)['Branch_And_Weight'].default == 'No'
    assert schema.declared_defaults(calculation.Base_Revaluation, {})['Branch_And_Weight'] == 'No'


def test_the_state_declares_it_so_a_pricer_reads_it_without_a_fallback():
    """`Calculation_State` carries the flag, so a pricer reads `shared.branch_and_weight` directly
    rather than through a `getattr` default a second calculation could disagree with."""
    state = utils.Calculation_State(
        {}, torch.ones([1, 1], dtype=DT), 1, None, 'Constant', 1, False)
    assert state.branch_and_weight is False


def test_the_switch_off_is_the_crisp_path(tmp_path):
    """OFF IS OFF on a real document: the key absent and the key written 'No' are the same number
    to the last bit. Every other gate in the suite depends on it."""
    absent = _run_tarf(_tarf_job(), tmp_path, 'absent')
    written = _run_tarf(_tarf_job(Branch_And_Weight='No'), tmp_path, 'written')
    assert absent == written, (absent, written)


# ======================================================================================
# GBM ONLY (`pricing.branch_and_weight`)
# ======================================================================================

def _deal_data(spot_model=None):
    """A real `FXTARFOptionDeal` off the fixture's deal block, with `HN_Params` shaped as
    `instruments.get_hn_factor` shapes it: `(is_stochastic, [factors], SpotModel, curve tenors)`."""
    with open(TARF_TEMPLATE) as f:
        block = json.load(f)['Calc']['Deals']['Deals']['Children'][0]['Instrument']['.Deal']
    instrument = instruments.construct_instrument(dict(block, Object='FXTARFOptionDeal'), {})
    factor_dep = {}
    if spot_model is not None:
        factor_dep['HN_Params'] = [(False, [utils.Factor(
            spot_model + 'ModelParameters', ('EUR.USD', 'Alpha'))], spot_model, {})]
    return utils.DealDataType(
        Instrument=instrument, Factor_dep=factor_dep, Time_dep=None, Calc_res=None)


class _State(utils.Calculation_State):
    """The engine's own state, with the switch set afterwards exactly as
    `Base_Revaluation.__init_shared_mem` sets it."""

    def __init__(self, on):
        super(_State, self).__init__(
            {}, torch.ones([1, 1], dtype=DT), 1, None, 'Constant', 1, False)
        self.branch_and_weight = on


@pytest.mark.parametrize('spot_model', [None, 'HestonNandi', 'HestonNandiComponent'])
def test_the_switch_off_admits_every_model(spot_model):
    """Off, the seam answers False and asks no questions - the refusal below cannot reach a run
    that did not ask for the smooth estimator."""
    assert pricing.branch_and_weight(_State(False), _deal_data(spot_model)) is False


def test_a_gbm_deal_under_the_switch_is_admitted():
    assert pricing.branch_and_weight(_State(True), _deal_data(None)) is True


@pytest.mark.parametrize('spot_model', ['HestonNandi', 'HestonNandiComponent'])
def test_the_switch_refuses_under_heston_nandi(spot_model):
    """GBM ONLY: the conditioning step must be the FIXING interval's own lognormal law. Under HN
    the walk is daily, so the only Gaussian conditional in hand is the last sub-step and a Gaussian
    `p` there would be a wrong number under the right estimator's name. Both flavours refuse.
    """
    with pytest.raises(ValueError) as refusal:
        pricing.branch_and_weight(_State(True), _deal_data(spot_model))
    message = str(refusal.value)
    assert 'Branch_And_Weight' in message, message
    assert spot_model in message, 'the refusal names the MODEL it refused: ' + message
    assert 'stride' in message.lower(), 'the refusal cites where its Phi comes from: ' + message
    assert 'hn_cdf_logret' in message, message
    # a refusal names its remedies
    assert 'GBM' in message and "Branch_And_Weight: 'No'" in message, message


# ======================================================================================
# The document harness, shared by the one gate above that needs a real run.
# ======================================================================================

def _tarf_job(**calc_overrides):
    with open(TARF_TEMPLATE) as f:
        job = json.load(f)
    job['Calc']['Calculation'].update(calc_overrides)
    return job


def _run_tarf(job, tmp_path, name):
    path = os.path.join(str(tmp_path), name + '.json')
    with open(path, 'w') as f:
        json.dump(job, f, default=str)
    cx = rf.Context()
    cx.load_json(path)
    _, out = cx.run_job()
    rows = out['Results']['mtm']
    return float(rows[rows['Reference'] == 'T1']['Value'].iloc[0])


# ======================================================================================
# THE PRODUCTS. Real documents through the JSON contract, against references built here.
# ======================================================================================
#
# ONE WORLD for every product gate. Both curves flat and DIFFERENT (4% against 2%, so the carry is
# live and no gate passes on a degenerate zero drift), the surface flat at 10%. That gives the
# reference an exact interval strip written from the market data rather than from
# `forward_carry_rate`, which is what is on test. Sloped curves have their own gates (roadmap).
R_USD, R_EUR, SIGMA_FLAT = 0.04, 0.02, 0.10
SPOT_FX, STRIKE_FX = 1.1, 1.1
N_ITM, N_OTM = 1000.0, 2000.0
TARGET, KNOCK_IN = 0.05, 1.05          # trigger at K + T = 1.15; knock-in below the strike
ACC_BARRIER = 1.15                      # the accumulator's up-and-out, ~21% at the first fixing
FIX_DAYS, SETTLE_DAYS = (91, 182), (93, 184)
BASE_DAY = datetime.date(2024, 6, 28)

ACC_TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'fixtures', 'fx_accumulator_job.json')


def _flat(rate):
    return {'.Curve': {'meta': [], 'data': [[0.0, rate], [5.0, rate]]}}


def _stamp(day):
    return {'.Timestamp': (BASE_DAY + datetime.timedelta(days=day)).isoformat()}


def _world(template, greeks='No', sims=1 << 16, seed=1, spot=SPOT_FX, **calc):
    """The template document with the flat world written over it - price factors and calculation."""
    with open(template) as f:
        job = json.load(f)
    pf = job['Calc']['MergeMarketData']['ExplicitMarketData']['Price Factors']
    pf['InterestRate.USD']['Curve'] = _flat(R_USD)
    pf['InterestRate.EUR']['Curve'] = _flat(R_EUR)
    pf['FxRate.EUR']['Spot'] = spot
    job['Calc']['Calculation'].update(
        {'Greeks': greeks, 'MCMC_Simulations': sims, 'Random_Seed': seed}, **calc)
    return job


def _tarf_doc(target=TARGET, barrier=KNOCK_IN, fixings=None, **kwargs):
    job = _world(TARF_TEMPLATE, **kwargs)
    deal = job['Calc']['Deals']['Deals']['Children'][0]['Instrument']['.Deal']
    deal.update(Strike_Price=STRIKE_FX, TargetLevel=target, Barrier=barrier,
                Underlying_Amount=N_ITM, LeverageNotional=N_OTM)
    if fixings is not None:
        deal['TARF_ExpiryDates'] = [[_stamp(d), _stamp(d + 2), 0.0] for d in fixings]
        deal['Expiry_Date'] = _stamp(fixings[-1])
    return job


def _acc_doc(same_day=False, **kwargs):
    job = _world(ACC_TEMPLATE, **kwargs)
    deal = job['Calc']['Deals']['Deals']['Children'][0]['Instrument']['.Deal']
    deal.update(Strike_Price=STRIKE_FX, Barrier_Price=ACC_BARRIER,
                Underlying_Amount=N_ITM, LeverageNotional=N_OTM)
    rows = [[_stamp(f), _stamp(s), 0.0] for f, s in zip(FIX_DAYS, SETTLE_DAYS)]
    if same_day:
        # a fixing dated ON the base date resolves off the SIMULATED spot, so the latch has a
        # graph-carrying gap - the only way this pricer reaches the refusal
        rows = [[_stamp(0), _stamp(2), 0.0]] + rows
    deal['Accumulator_ExpiryDates'] = rows
    return job


def _smooth(job, on=True):
    job['Calc']['Calculation']['Branch_And_Weight'] = 'Yes' if on else 'No'
    return job


def _run_doc(job, tmp_path, name, debug=False):
    """JSON in, (value, results, DEBUG log) out. Nothing here reaches past the loader."""
    path = os.path.join(str(tmp_path), name + '.json')
    with open(path, 'w') as f:
        json.dump(job, f, default=str)
    buf, root = io.StringIO(), logging.getLogger()
    handler, old = logging.StreamHandler(buf), root.level
    if debug:
        root.addHandler(handler)
        root.setLevel(logging.DEBUG)
    try:
        cx = rf.Context()
        cx.load_json(path)
        _, out = cx.run_job()
    finally:
        if debug:
            root.removeHandler(handler)
            root.setLevel(old)
    rows = out['Results']['mtm']
    ref = job['Calc']['Deals']['Deals']['Children'][0]['Instrument']['.Deal']['Reference']
    return float(rows[rows['Reference'] == ref]['Value'].iloc[0]), out, buf.getvalue()


def _first(out, factor='FxRate.EUR'):
    frame = out['Results']['Greeks_First']
    column = [c for c in frame.columns if c != 'Value'][0]
    return float(frame.loc[[i for i in frame.index if str(i[0]) == factor][0], column])


def _second(out, spot='FxRate.EUR', vol='FXVol.EUR.USD'):
    """(gamma, vanna) off the reported Hessian.

    VANNA IS A SUM over the surface's knots: the reference differentiates a single scalar vol,
    which on a FLAT surface is a parallel bump of every knot. The same sum against `Greeks_First`
    reproduces the reference's vega, which is what says the identification is right.
    """
    frame = out['Results']['Greeks_Second']
    row, = [i for i in frame.index if str(i[0]) == spot]
    col, = [c for c in frame.columns if str(c[1]) == spot]
    return (float(frame.loc[row, col]),
            sum(float(frame.loc[row, c]) for c in frame.columns if str(c[1]) == vol))


# ======================================================================================
# THE DIFFERENTIABLE QUADRATURE REFERENCES. No Monte Carlo, no closed form, no engine.
# ======================================================================================

def _t_phi(z):
    """The normal DENSITY, the only distributional fact these references know. No `Phi` here on
    purpose: a sign or reflection error in `norm_cdf` cannot be reproduced by its own oracle."""
    return torch.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


Z_INF = 8.5      # the density there is 5e-15; the payoffs are O(1e3), so the tail is 1e-11


def _seg(a, b, f, n):
    """``int_a^b phi(z) f(z) dz`` by trapezoid on a grid whose ENDS MOVE with theta.

    A FIXED grid with the decisions written as indicators has a value but no derivative - it misses
    exactly the flux the estimator under test carries. Mapping each region onto `u` in [0, 1] puts
    the region's own limits on the tape, so autograd differentiates the way Leibniz's rule does.

    `a`/`b` may be scalars or vectors (one per outer node); the sample axis is last, so a nested
    call broadcasts without a reshape. Crossed limits integrate negatively, which matters only
    where the density has already underflowed.
    """
    u = torch.linspace(0.0, 1.0, n, dtype=DT)
    w = torch.full((n,), 1.0 / (n - 1), dtype=DT)
    w[0] = w[-1] = 0.5 / (n - 1)
    a = a if torch.is_tensor(a) else torch.tensor(a, dtype=DT)
    b = b if torch.is_tensor(b) else torch.tensor(b, dtype=DT)
    z = a[..., None] + (b - a)[..., None] * u
    return (b - a) * ((w * _t_phi(z)) * f(z)).sum(-1)


def _intervals():
    """The two intervals' `(dt1, dt2)` and discount factors, from the MARKET DATA: a flat curve's
    interval carry IS its zero rate and a flat surface's interval vol IS its quote, so this needs
    neither `forward_carry_rate` nor `forward_vol_rate`, which are what is on test."""
    dt = (FIX_DAYS[0] / 365.0, (FIX_DAYS[1] - FIX_DAYS[0]) / 365.0)
    disc = tuple(math.exp(-R_USD * d / 365.0) for d in SETTLE_DAYS)
    return dt, disc


def _tarf_reference(spot, sigma, target=TARGET, n_out=600, n_in=400):
    """The two-fixing TARF as a nested region integral, differentiable end to end.

    Written from the DEAL alone. Per fixing the line splits at the knock-in `Bar`, the strike `K`
    and the moving trigger `K + r` (`r` = target left):

        z < zBar         knocked in and OTM        -N2 * (K - S)
        zBar < z < zK    OTM, not knocked in        0
        zK < z < zB      ITM, target not yet full  +N1 * (S - K)
        z > zB           the target FILLS          +N1 * r, the remaining target

    The second fixing's levels ride the first's outcome because `r` does, which is why the
    trapezoid is nested rather than a product grid.
    """
    (dt1, dt2), (D1, D2) = _intervals()
    carry = R_USD - R_EUR
    v1, v2 = sigma * math.sqrt(dt1), sigma * math.sqrt(dt2)
    m1 = (carry - 0.5 * sigma ** 2) * dt1
    m2 = (carry - 0.5 * sigma ** 2) * dt2
    K = torch.as_tensor(STRIKE_FX, dtype=DT)
    knock = torch.as_tensor(KNOCK_IN, dtype=DT)

    def z1_of(level):
        return (torch.log(level / spot) - m1) / v1

    zB1, zK1, zBar1 = z1_of(K + target), z1_of(K), z1_of(knock)

    def tail(z1):
        """Everything fixing two is worth, given fixing one's own draw."""
        s1 = spot * torch.exp(m1 + v1 * z1)
        r1 = target - torch.clamp(s1 - K, min=0.0)
        s1c = s1[..., None]

        def z2_of(level):
            return (torch.log(level / s1) - m2) / v2

        zB2, zK2, zBar2 = z2_of(K + r1), z2_of(K), z2_of(knock)

        def s2(z):
            return s1c * torch.exp(m2 + v2 * z)

        value = _seg(torch.full_like(zBar2, -Z_INF), zBar2, lambda z: -N_OTM * (K - s2(z)), n_in)
        value = value + _seg(zK2, zB2, lambda z: N_ITM * (s2(z) - K), n_in)
        # the filling fixing pays what is LEFT of the target - a per-path constant over this
        # interval, and the coupling that makes a TARF a TARF
        top = torch.full_like(zB2, Z_INF)
        paid = (N_ITM * r1)[..., None]
        value = value + _seg(zB2, top, lambda z: paid * torch.ones_like(z), n_in)
        return D2 * value

    top1 = torch.as_tensor(Z_INF, dtype=DT)
    total = D1 * N_ITM * target * _seg(zB1, top1, torch.ones_like, n_out)

    for lo, hi, leg in ((torch.as_tensor(-Z_INF, dtype=DT), zBar1, lambda s: -N_OTM * (K - s)),
                        (zBar1, zK1, None),
                        (zK1, zB1, lambda s: N_ITM * (s - K))):
        def integrand(z, leg=leg):
            paid = tail(z)
            if leg is None:
                return paid
            return D1 * leg(spot * torch.exp(m1 + v1 * z)) + paid
        total = total + _seg(lo, hi, integrand, n_out)
    return total


def _acc_reference(spot, sigma, n_out=600, n_in=400):
    """The two-fixing accumulator, the same way: two levels per fixing (the strike splits the legs,
    the barrier ends the deal) and no coupling - nothing accrues that moves its own trigger."""
    (dt1, dt2), (D1, D2) = _intervals()
    carry = R_USD - R_EUR
    v1, v2 = sigma * math.sqrt(dt1), sigma * math.sqrt(dt2)
    m1 = (carry - 0.5 * sigma ** 2) * dt1
    m2 = (carry - 0.5 * sigma ** 2) * dt2
    K = torch.as_tensor(STRIKE_FX, dtype=DT)
    B = torch.as_tensor(ACC_BARRIER, dtype=DT)
    zB1 = (torch.log(B / spot) - m1) / v1
    zK1 = (torch.log(K / spot) - m1) / v1

    def tail(z1):
        s1 = spot * torch.exp(m1 + v1 * z1)
        s1c = s1[..., None]
        zB2 = (torch.log(B / s1) - m2) / v2
        zK2 = (torch.log(K / s1) - m2) / v2

        def s2(z):
            return s1c * torch.exp(m2 + v2 * z)

        return D2 * (_seg(torch.full_like(zK2, -Z_INF), zK2,
                          lambda z: -N_OTM * (K - s2(z)), n_in) +
                     _seg(zK2, zB2, lambda z: N_ITM * (s2(z) - K), n_in))

    total = torch.zeros((), dtype=DT)
    for lo, hi, leg in ((torch.as_tensor(-Z_INF, dtype=DT), zK1, lambda s: -N_OTM * (K - s)),
                        (zK1, zB1, lambda s: N_ITM * (s - K))):
        def integrand(z, leg=leg):
            return D1 * leg(spot * torch.exp(m1 + v1 * z)) + tail(z)
        total = total + _seg(lo, hi, integrand, n_out)
    return total


def _reference_table(build, base_spot=SPOT_FX, base_sigma=SIGMA_FLAT, **kwargs):
    """value / delta / vega / gamma / vanna off one reference, by double backward. `base_spot` /
    `base_sigma` are where the derivatives are taken; the FX products default to their shared
    world, the autocall passes its own."""
    spot = torch.tensor(base_spot, dtype=DT, requires_grad=True)
    sigma = torch.tensor(base_sigma, dtype=DT, requires_grad=True)
    value = build(spot, sigma, **kwargs)
    first = torch.autograd.grad(value, (spot, sigma), create_graph=True)
    second = torch.autograd.grad(first[0], (spot, sigma), retain_graph=True)
    return {'value': float(value), 'delta': float(first[0]), 'vega': float(first[1]),
            'gamma': float(second[0]), 'vanna': float(second[1])}


_REFERENCES = {}


def _table(build, **kwargs):
    """The reference tables, memoized: each is a nested double backward through ~1e6 nodes."""
    key = (build.__name__,) + tuple(sorted(kwargs.items()))
    if key not in _REFERENCES:
        _REFERENCES[key] = _reference_table(build, **kwargs)
    return _REFERENCES[key]


def _rel(got, want):
    return abs(got - want) / max(abs(want), 1e-30)


# ======================================================================================
# OFF IS OFF, and what the switch is attributable for
# ======================================================================================

@pytest.mark.parametrize('build,name', [(_tarf_doc, 'tarf'), (_acc_doc, 'accumulator')])
def test_off_is_off_on_a_real_document(build, name, tmp_path):
    """The key ABSENT and the key written 'No' are the same run - value, the whole reported mtm
    frame and every first-order greek, by `np.array_equal` rather than a tolerance.

    Every number in `test_fx_tarf_json`, `test_fx_accumulator_json`, `test_tarf_cash_settle` and
    the two boundary-event files is a regression bar for this build only while this holds.
    """
    absent, out_a, _ = _run_doc(build(greeks='First'), tmp_path, name + '_absent')
    written, out_w, _ = _run_doc(_smooth(build(greeks='First'), on=False), tmp_path, name + '_no')
    assert absent == written, (absent, written)
    # `DataFrame.equals` rather than `array_equal`: the frame carries an all-NaN root row, and NaN
    # is not equal to itself
    assert out_a['Results']['mtm'].equals(out_w['Results']['mtm']), 'the reported frame moved'
    frame_a, frame_w = out_a['Results']['Greeks_First'], out_w['Results']['Greeks_First']
    assert np.array_equal(frame_a.values, frame_w.values), 'a first-order greek moved'


def test_a_tarf_with_no_knock_in_is_bit_identical_under_the_switch(tmp_path):
    """ATTRIBUTION. The one-step-survival loop was ALREADY the smooth estimator for the target -
    the KO-in-step term is the fired branch integrated against the interval's law, the continuation
    is the truncated draw - so with the knock-in off the two estimators agree TO THE BIT. Turn the
    leveraged knock-in on and they part by ~1.3% at 65536 paths (the gate below): that leg was the
    one indicator this pricer still sampled.
    """
    crisp, _, _ = _run_doc(_tarf_doc(barrier=0.0), tmp_path, 'nokick_off')
    smooth, _, _ = _run_doc(_smooth(_tarf_doc(barrier=0.0)), tmp_path, 'nokick_on')
    assert crisp == smooth, (crisp, smooth)


# ======================================================================================
# THE QUADRATURE TABLE, and the estimator it kills
# ======================================================================================

def _erf_Phi(x):
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _erf_Phi_inv(u):
    return math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)


def _pathwise_tarf(spot, sigma, n1=4096, n2=4096):
    """THE FORBIDDEN ESTIMATOR, written here so the gate can kill it: the same construction with
    the knock-in left as a SAMPLED INDICATOR and no kink term. Deterministic (a midpoint product
    grid, no seed), so any gap against the quadrature is bias rather than noise. Its VALUE is
    right; every derivative is not.
    """
    (dt1, dt2), (D1, D2) = _intervals()
    carry = R_USD - R_EUR
    v1, v2 = sigma * math.sqrt(dt1), sigma * math.sqrt(dt2)
    m1 = (carry - 0.5 * sigma ** 2) * dt1
    m2 = (carry - 0.5 * sigma ** 2) * dt2
    K = torch.as_tensor(STRIKE_FX, dtype=DT)

    def leg(s):
        gain = s - K
        return (torch.relu(gain) * N_ITM -
                torch.relu(-gain) * N_OTM * (s <= KNOCK_IN).to(DT))

    zB1 = (torch.log(K + TARGET) - torch.log(spot) - m1) / v1
    p1 = _erf_Phi(zB1)
    z1 = _erf_Phi_inv(torch.clamp(_stratified(n1) * p1, 1e-15, 1.0 - 1e-15))
    s1 = spot * torch.exp(m1 + v1 * z1)
    r1 = TARGET - torch.relu(s1 - K)
    total = D1 * (N_ITM * TARGET * (1.0 - p1) + p1 * leg(s1).mean())

    zB2 = (torch.log(K + r1) - torch.log(s1) - m2) / v2
    p2 = _erf_Phi(zB2)
    z2 = _erf_Phi_inv(torch.clamp(_stratified(n2) * p2[:, None], 1e-15, 1.0 - 1e-15))
    s2 = s1[:, None] * torch.exp(m2 + v2 * z2)
    tail = (1.0 - p2) * N_ITM * r1 + p2 * leg(s2).mean(-1)
    return total + D2 * p1 * tail.mean()


def test_the_two_fixing_tarf_lands_on_the_quadrature_table(tmp_path):
    """THE TABLE. A two-fixing knock-in TARF under the switch at `Greeks: 'All'` against the
    quadrature reference (spot 1.1, strike 1.1, target 0.05, knock-in 1.05, notionals 1000/2000,
    r 4% q 2%, vol 10%, fixings at 91 and 182 days, 262144 inner paths):

                        engine        quadrature      relative
        value          -37.4018        -37.3594         0.113%
        delta        +1814.772       +1814.660          0.006%
        vega         -1131.33        -1131.166          0.015%
        gamma       -28097.69       -28087.200          0.037%
        vanna        +5016.06        +5010.805          0.105%

    The vega row is what says the vol identification is right: the reference differentiates ONE
    scalar vol and the report spreads it over four live knots, so the sum has to reproduce the
    reference's vega before summing them for vanna means anything.
    """
    ref = _table(_tarf_reference)
    value, out, _ = _run_doc(_smooth(_tarf_doc(greeks='All', sims=1 << 18)), tmp_path, 'table')
    gamma, vanna = _second(out)
    frame = out['Results']['Greeks_First']
    column = [c for c in frame.columns if c != 'Value'][0]
    vega = sum(float(frame.loc[i, column]) for i in frame.index
               if str(i[0]) == 'FXVol.EUR.USD')
    got = {'value': value, 'delta': _first(out), 'vega': vega, 'gamma': gamma, 'vanna': vanna}
    tol = {'value': 0.01, 'delta': 0.005, 'vega': 0.005, 'gamma': 0.01, 'vanna': 0.02}
    bad = {k: (got[k], ref[k], _rel(got[k], ref[k])) for k in tol if _rel(got[k], ref[k]) > tol[k]}
    assert not bad, bad


def test_the_pathwise_estimator_reports_the_wrong_vanna_sign():
    """THE KILL. The same construction with the knock-in sampled rather than integrated, on a
    deterministic 4096 x 4096 grid, against the quadrature - both sides quadratures of one
    integral, and the ratios do not move between 2048 and 4096 nodes:

        value    1.0001 x the truth   - the estimator is UNBIASED, which is the trap
        delta    0.7143 x             - 28.6% short
        vega     0.4826 x             - 52% short
        gamma    0.3006 x             - 70% short
        vanna   -0.3413 x             - THE WRONG SIGN

    A right value beside wrong derivatives is the failure mode nothing downstream detects: the mark
    reconciles, the hedge does not.
    """
    ref = _table(_tarf_reference)
    got = _reference_table(_pathwise_tarf)
    assert _rel(got['value'], ref['value']) < 2e-3, (
        'the pathwise estimator has stopped being an unbiased VALUE estimator, so this gate is no '
        'longer measuring what it says it is: {} against {}'.format(got['value'], ref['value']))
    assert got['vanna'] * ref['vanna'] < 0.0, (
        'the pathwise vanna no longer has the wrong SIGN, so the kill this gate records has gone: '
        '{} against {}'.format(got['vanna'], ref['vanna']))
    assert got['gamma'] / ref['gamma'] < 0.5, (got['gamma'], ref['gamma'])
    assert got['delta'] / ref['delta'] < 0.85, (got['delta'], ref['delta'])


# ======================================================================================
# UNBIASED, AND LOWER VARIANCE - the Rao-Blackwell reading
# ======================================================================================

def test_the_switch_is_rao_blackwell_and_not_a_smoothing(tmp_path):
    """Same expectation, less variance. Ten seeds at 16384 inner paths against the quadrature's
    -37.359382:

        crisp    mean -37.5296   seed sd 0.3208
        smooth   mean -37.4098   seed sd 0.0910

    Variance ratio 12.42, and both means sit inside their own spreads of the reference (0.456% and
    0.135% out).
    """
    seeds = range(1, 11)
    crisp = [_run_doc(_tarf_doc(seed=s, sims=1 << 14), tmp_path, 'rb_off')[0] for s in seeds]
    smooth = [_run_doc(_smooth(_tarf_doc(seed=s, sims=1 << 14)), tmp_path, 'rb_on')[0]
              for s in seeds]
    truth = _table(_tarf_reference)['value']
    sd_c, sd_s = float(np.std(crisp, ddof=1)), float(np.std(smooth, ddof=1))
    assert sd_c / sd_s > 2.0, (
        'the smooth estimator is not visibly quieter than the crisp one, so the leg is probably '
        'still being sampled: sd {:.4g} against {:.4g}'.format(sd_c, sd_s))
    for label, sample, spread in (('crisp', crisp, sd_c), ('smooth', smooth, sd_s)):
        drift = abs(float(np.mean(sample)) - truth)
        assert drift < max(3.0 * spread / math.sqrt(len(sample)), 0.01 * abs(truth)), (
            '{} mean {:.6f} is not the quadrature {:.6f} within its own seed spread'.format(
                label, float(np.mean(sample)), truth))


def test_the_target_pin_is_exact_under_the_switch(tmp_path):
    """THE KNOWN-DEFECTS ROW, closed. Uncorrected the pin read 27% short with neither the estimator
    (13% bandwidth spread) nor the oracle (8.9% flatness) resolving better than ~10%, so no
    tolerance could be asserted. Here the payment is a closed-form conditional expectation and the
    oracle resolves to 1e-5, so the tolerance is set an order under that floor.

    Three seeds at 65536 paths, target nearly filled at the first of two fixings (~46% fire):

        target 0.010   quadrature -54.6691   smooth -54.7124 (0.079%)   crisp -54.8276 (0.290%)
        target 0.020   quadrature -50.0300   smooth -50.0769 (0.094%)   crisp -50.2121 (0.364%)

    The crisp readings are recorded, not gated.
    """
    for target in (0.01, 0.02):
        truth = _table(_tarf_reference, target=target)['value']
        smooth = np.mean([_run_doc(_smooth(_tarf_doc(target=target, seed=s)), tmp_path, 'pin')[0]
                          for s in (1, 2, 3)])
        assert _rel(smooth, truth) < 0.01, (
            'target {}: the pinned payment reads {:.6f} against a quadrature of {:.6f} '
            '({:.3%}) - the pin is meant to be EXACT on this path, not estimated'.format(
                target, smooth, truth, _rel(smooth, truth)))


# ======================================================================================
# ONE SETTLEMENT CONVENTION, ON BOTH ESTIMATORS
# ======================================================================================

def test_the_filling_fixing_pays_the_remaining_target_under_both_estimators(tmp_path):
    """The fixing that fills the target pays the remaining target `R` - measurable one fixing back,
    so `(1 - p) * R` IS its conditional expectation and both branches are the same arithmetic.
    `TargetAdjustment` was a declared field nothing read; honouring it repriced a deal 44%, so it
    is gone along with the fork it fed.

    Gated as the property that buys: the SAME document under both estimators lands on the SAME
    quadrature, whose fired branch pays `N1 * r` and nothing else.
    """
    truth = _table(_tarf_reference)['value']
    crisp, _, _ = _run_doc(_tarf_doc(sims=1 << 18), tmp_path, 'conv_off')
    smooth, _, _ = _run_doc(_smooth(_tarf_doc(sims=1 << 18)), tmp_path, 'conv_on')
    for label, got in (('crisp', crisp), ('smooth', smooth)):
        assert _rel(got, truth) < 0.01, (
            'the {} estimator reads {:.6f} against the remaining-target quadrature {:.6f} '
            '({:.3%}) - the two are meant to price ONE deal'.format(
                label, got, truth, _rel(got, truth)))


def test_a_target_adjustment_key_is_inert_wherever_it_survives(tmp_path):
    """A RETIRED FIELD IS NOT A REFUSAL. A document still carrying `TargetAdjustment` loads,
    prices, and reads the blank document's value TO THE BIT under both estimators. The two failure
    modes are opposite and both plausible: a surviving reader moves the number, a strict loader
    rejects a live book.
    """
    for on in (False, True):
        base, _, _ = _run_doc(_smooth(_tarf_doc(), on=on), tmp_path, 'inert_base')
        job = _smooth(_tarf_doc(), on=on)
        job['Calc']['Deals']['Deals']['Children'][0]['Instrument']['.Deal'][
            'TargetAdjustment'] = 'Full Gain'
        stale, _, _ = _run_doc(job, tmp_path, 'inert_stale')
        assert base == stale, (
            "a document carrying a stale 'Full Gain' priced differently with the switch {} - "
            'something still reads the retired field: {} against {}'.format(
                'on' if on else 'off', stale, base))


# ======================================================================================
# GAMMA FLOWS
# ======================================================================================

def test_gamma_flows_under_the_switch_and_is_refused_without_it(tmp_path):
    """`Greeks: 'All'` on a knock-in TARF is REFUSED without the switch and RUNS with it, and the
    Hessian entry is the derivative of the delta the same run reports.

    65536 paths: spot-spot -28120.6 against a CRN ladder of the reported delta whose flattest rung
    reads -28142.6 - 0.08% at 0.87% flatness. A ladder of the CRISP corrected delta does not
    converge: -12339 / -24870 / -26589 / -27145, a 57.5% spread climbing with h, because that delta
    is itself a kernel estimate whose bandwidth comes from the sample's own spread.
    """
    with pytest.raises(utils.SecondOrderRefused) as refusal:
        _run_doc(_tarf_doc(greeks='All'), tmp_path, 'all_off')
    assert 'boundary correction' in str(refusal.value)

    _, out, _ = _run_doc(_smooth(_tarf_doc(greeks='All')), tmp_path, 'all_on')
    gamma, _ = _second(out)

    def delta_at(spot):
        return _first(_run_doc(_smooth(_tarf_doc(greeks='First', spot=spot)),
                               tmp_path, 'ladder')[1])

    result = ladder(price=delta_at, aad=gamma, base=SPOT_FX,
                    rungs=(3e-4, 1e-3, 3e-3, 1e-2))
    assert result.agrees(tol=0.02), (
        'the reported gamma is not the derivative of the reported delta\n{}'.format(result))


def test_the_ledger_conserves_on_a_real_document(tmp_path):
    """CONSERVATION on a live run: per path, the mass fired plus the mass alive is the mass the
    strip opened with. Measured 1.11e-16 with 59.0% surviving both fixings - a ledger that fired
    everything or nothing would conserve trivially."""
    _, _, log = _run_doc(_smooth(_tarf_doc(sims=1 << 12)), tmp_path, 'ledger', debug=True)
    lines = [ln for ln in log.splitlines() if 'LEDGER TARF' in ln]
    assert lines, 'the smooth path ran no ledger at all'
    residual = max(float(ln.split('conservation=')[1].split()[0]) for ln in lines)
    alive = [float(ln.split('alive=')[1].split()[0]) for ln in lines]
    assert residual < 1e-12, (residual, lines)
    assert 0.05 < max(alive) < 0.95, (
        'this document fires on almost none or almost all of its weight, so conservation on it '
        'says nothing: alive={}'.format(alive))


def test_the_recompute_node_replays_the_smooth_callable(tmp_path):
    """COMPOSITION. `Recompute_Inner_MC` re-runs the inner simulation in `backward()` instead of
    taping it, and the smooth path is one function called twice - so value and every first-order
    greek come back BIT-IDENTICAL. Second order still refuses through the node's own refusal: the
    recompute is rooted at a detached input, and `create_graph` through it would be a plausible
    wrong number rather than a failure."""
    taped, out_t, _ = _run_doc(_smooth(_tarf_doc(greeks='First')), tmp_path, 'taped')
    node, out_n, _ = _run_doc(
        _smooth(_tarf_doc(greeks='First', Recompute_Inner_MC='Yes')), tmp_path, 'node')
    assert taped == node, (taped, node)
    assert np.array_equal(out_t['Results']['Greeks_First'].values,
                          out_n['Results']['Greeks_First'].values)
    with pytest.raises(Exception) as refusal:
        _run_doc(_smooth(_tarf_doc(greeks='All', Recompute_Inner_MC='Yes')), tmp_path, 'both')
    assert 'Recompute_Inner_MC' in str(refusal.value), str(refusal.value)


def test_the_daily_fixing_cost_is_declared_and_measured(tmp_path):
    """NOT A REFUSAL - a declaration with a number behind it. Second-order variance in the
    conditioning step scales like `s_k**-3`. Twelve fixings either way, eight seeds, 8192 paths:

        monthly (30d)   value seed spread 1.126%   gamma seed spread  1.696%
        daily   (1d)    value seed spread 0.350%   gamma seed spread 21.154%

    The VALUE spread FALLS at daily spacing while gamma's grows 12.5x. Gated on that shape, so
    nobody reads a daily gamma off this path believing the monthly measurement covers it.
    """
    def spread(days):
        readings = []
        for seed in range(1, 9):
            job = _smooth(_tarf_doc(fixings=days, greeks='All', sims=1 << 13, seed=seed))
            _, out, _ = _run_doc(job, tmp_path, 'spacing')
            readings.append(_second(out)[0])
        return float(np.std(readings, ddof=1) / abs(np.mean(readings)))

    monthly = spread([30 * (i + 1) for i in range(12)])
    daily = spread([i + 1 for i in range(12)])
    assert monthly < 0.05, ('monthly gamma is already noisy at {:.2%}, so the daily reading below '
                            'is not attributable to the conditioning step'.format(monthly))
    assert daily / monthly > 3.0, (
        'daily fixings no longer cost what this declaration says they cost ({:.2%} against '
        '{:.2%}) - re-measure before deleting the sentence'.format(daily, monthly))


# ======================================================================================
# THE ACCUMULATOR
# ======================================================================================

def test_the_accumulator_value_is_untouched_and_its_curvature_is_not(tmp_path):
    """This pricer's loop was already the smooth estimator - analytic survival, truncated
    continuation, a fired branch worth exactly zero - so the switch cannot move its VALUE and does
    not: bit-identical off and on, first order with it. What it changes is the curvature, because
    both legs are `relu`s of one argument whose pathwise gamma is an exact zero. One document,
    65536 paths, the same `Greeks: 'All'` run either way:

                    crisp          smooth        quadrature
        gamma     -24156.07      -27154.08      -27172.47      11.1% short -> 0.07%
        vanna      -1791.03       -220.69        -211.25       8.5x        -> 4.5%
    """
    ref = _table(_acc_reference)
    crisp, out_c, _ = _run_doc(_acc_doc(greeks='All'), tmp_path, 'acc_off')
    smooth, out_s, _ = _run_doc(_smooth(_acc_doc(greeks='All')), tmp_path, 'acc_on')
    assert crisp == smooth, (crisp, smooth)
    assert _first(out_c) == _first(out_s), 'first order moved, and it has nothing to move through'
    g_crisp, v_crisp = _second(out_c)
    g_smooth, v_smooth = _second(out_s)
    assert _rel(g_smooth, ref['gamma']) < 0.01, (g_smooth, ref['gamma'])
    assert _rel(v_smooth, ref['vanna']) < 0.10, (v_smooth, ref['vanna'])
    assert _rel(g_crisp, ref['gamma']) > 10.0 * _rel(g_smooth, ref['gamma']), (
        'the pathwise gamma is no longer visibly wrong, so the kink term is not what is being '
        'measured: {} against {} with a reference of {}'.format(g_crisp, g_smooth, ref['gamma']))
    assert _rel(v_crisp, ref['vanna']) > 5.0 * _rel(v_smooth, ref['vanna']), (
        v_crisp, v_smooth, ref['vanna'])


def test_the_accumulator_registration_is_superseded_not_lost(tmp_path):
    """The accumulator registers only on an OBSERVED fixing, so this document dates one ON the base
    date - resolved off the simulated spot, which gives the latch a graph-carrying gap and makes
    `Greeks: 'All'` refuse.

    Under the switch the registration is skipped and the Hessian flows. What says nothing was LOST
    is the first-order comparison beside it: BIT FOR BIT. Base valuation has one scenario, a
    one-sample gap supports no local-linear fit, and the correction is exactly zero by construction.
    """
    with pytest.raises(utils.SecondOrderRefused):
        _run_doc(_acc_doc(same_day=True, greeks='All'), tmp_path, 'sd_off')
    _, out, _ = _run_doc(_smooth(_acc_doc(same_day=True, greeks='All')), tmp_path, 'sd_on')
    assert np.isfinite(_second(out)[0])
    crisp, out_c, _ = _run_doc(_acc_doc(same_day=True, greeks='First'), tmp_path, 'sd1_off')
    smooth, out_s, _ = _run_doc(
        _smooth(_acc_doc(same_day=True, greeks='First')), tmp_path, 'sd1_on')
    assert crisp == smooth
    assert np.array_equal(out_c['Results']['Greeks_First'].values,
                          out_s['Results']['Greeks_First'].values), (
        'the skipped registration was carrying flux after all - it is not zero here, and the '
        'supersession is losing a derivative rather than replacing an estimator')


# ======================================================================================
# THE DISCRETE BARRIER, and the model refusal on a document
# ======================================================================================

BARRIER_DATES = [bb.BASE + pd.Timedelta(days=d) for d in range(30, 366, 30)]

# The binary sibling, authored here: `test_barrier_bridge`'s digital fixture has an UNREACHABLE
# barrier, and this section walks the live one too. Everything else is that file's world.
BARRIER_BINARY = {
    'Object': 'EquityBarrierBinaryOption', 'Reference': 'BARR1', 'Currency': 'USD',
    'Payoff_Currency': 'USD', 'Equity': 'EQ', 'Dividends': 'EQ', 'Discount_Rate': 'USD',
    'Equity_Volatility': 'EQ', 'Buy_Sell': 'Buy', 'Option_Type': 'Call', 'Strike_Price': 100.0,
    'Expiry_Date': bb.BASE + pd.Timedelta(days=365), 'Cash_Payoff': 100.0,
    'Settlement_Date': bb.BASE + pd.Timedelta(days=365), 'Barrier_Type': 'Down_And_Out',
    'Barrier_Price': 90.0, 'Barrier_Dates': BARRIER_DATES}


def _barrier_config(barrier_price=90.0, digital=False, spot=None, **over):
    """A monthly-monitored barrier in `test_barrier_bridge`'s zero-rate world.

    ZERO RATES ARE LOAD-BEARING: with every discount factor exactly one, the in-out parity gates
    below read the ledger's telescoping identity off two prices rather than a discounted weighting
    of it."""
    cfg = bb._cfg()
    base = BARRIER_BINARY if digital else dict(bb.BARRIER_DEAL, Barrier_Dates=BARRIER_DATES)
    deal = dict(base, Barrier_Price=barrier_price, **over)
    cfg.deals['Deals']['Children'] = [{'Instrument': construct_instrument(deal, {})}]
    if spot is not None:
        cfg.params['Price Factors']['EquityPrice.EQ']['Spot'] = spot
    return cfg


def _barrier_run(barrier_price=90.0, greeks='No', on=None, sims=1 << 15, **cfg_kwargs):
    """(value, results) off one base valuation. `on=None` leaves the switch ABSENT - a third state
    the byte-identity gate needs, not a spelling of 'No'."""
    overrides = {'MCMC_Simulations': sims, 'Random_Seed': 1, 'Greeks': greeks}
    if on is not None:
        overrides['Branch_And_Weight'] = 'Yes' if on else 'No'
    _, out = rf.run_baseval(_barrier_config(barrier_price, **cfg_kwargs), overrides=overrides)
    rows = out['Results']['mtm']
    return float(rows[rows['Reference'] == 'BARR1']['Value'].iloc[0]), out


def _eq_first(out):
    frame = out['Results']['Greeks_First']
    column = [c for c in frame.columns if c != 'Value'][0]
    return float(frame.loc[[i for i in frame.index if str(i[0]) == 'EquityPrice.EQ'][0], column])


def _eq_second(out):
    frame = out['Results']['Greeks_Second']
    row, = [i for i in frame.index if str(i[0]) == 'EquityPrice.EQ']
    col, = [c for c in frame.columns if str(c[1]) == 'EquityPrice.EQ']
    return float(frame.loc[row, col])


def _black_call(vol=None):
    """(value, delta, gamma) of the one-year European this world's barriers all reduce to."""
    sd = bb.VOL if vol is None else vol
    d1 = (math.log(bb.SPOT / 100.0) + 0.5 * sd * sd) / sd
    return (bb.SPOT * _Phi(d1) - 100.0 * _Phi(d1 - sd), _Phi(d1),
            _phi(d1) / (bb.SPOT * sd))


# The deal family, so the bit-identity claim is about the PRICER rather than one document: both
# directions, both barrier sides, both option types, both sides of the trade, the rebate on each
# leg, and the binary sibling. Every row prices the same to the last bit absent, 'No' and 'Yes'.
BARRIER_FAMILY = [
    ('down-and-out call', 90.0, {}, False),
    ('up-and-out call', 130.0, {'Barrier_Type': 'Up_And_Out'}, False),
    ('down-and-in call', 90.0, {'Barrier_Type': 'Down_And_In'}, False),
    ('up-and-in call', 130.0, {'Barrier_Type': 'Up_And_In'}, False),
    ('down-and-out put', 90.0, {'Option_Type': 'Put'}, False),
    ('knock-out with a rebate', 90.0, {'Cash_Rebate': 5.0}, False),
    ('knock-in with a rebate', 90.0, {'Barrier_Type': 'Down_And_In', 'Cash_Rebate': 5.0}, False),
    ('sold knock-out with a rebate', 90.0, {'Buy_Sell': 'Sell', 'Cash_Rebate': 5.0}, False),
    ('binary down-and-out', 90.0, {}, True),
    ('binary up-and-out', 130.0, {'Barrier_Type': 'Up_And_Out'}, True),
    ('binary down-and-in', 90.0, {'Barrier_Type': 'Down_And_In'}, True),
]


@pytest.mark.parametrize('barrier,over,digital', [row[1:] for row in BARRIER_FAMILY],
                         ids=[row[0] for row in BARRIER_FAMILY])
def test_off_is_off_across_the_whole_barrier_family(barrier, over, digital):
    """OFF IS OFF, and ON IS OFF TOO. This pricer's OSS sampler was the smooth estimator already -
    every monitored step an analytic `p` with a survival-truncated continuation, the rebate
    `(1 - p) * L * rebate * D_j` at the fixing it falls due, a digital's terminal step integrated -
    so there is no indicator to remove and the value under the switch is BIT-IDENTICAL.

    Walked across the family because one down-and-out call says nothing about the parity leg, which
    side the truncation reflects on, the rebate's unsigned size, or a digital's branch. Three
    states, not two: a field read through a default is a different code path from a field read.
    """
    absent, _ = _barrier_run(barrier, on=None, sims=1 << 12, digital=digital, **over)
    off, _ = _barrier_run(barrier, on=False, sims=1 << 12, digital=digital, **over)
    smooth, _ = _barrier_run(barrier, on=True, sims=1 << 12, digital=digital, **over)
    assert absent == off, ('the declared default moved the number', absent, off)
    assert absent == smooth, (
        'the switch moved a value its own sampler already computed the smooth way: {!r} absent, '
        '{!r} on'.format(absent, smooth))


@pytest.mark.parametrize('barrier,over,digital', [
    (90.0, {}, False), (90.0, {'Barrier_Type': 'Down_And_In'}, False),
    (90.0, {'Cash_Rebate': 5.0}, False), (90.0, {}, True)],
    ids=['knock-out', 'knock-in', 'rebate', 'binary'])
def test_the_skipped_registration_loses_nothing_at_first_order(barrier, over, digital):
    """WHY skipping the `LatchedBoundarySet` is free here. The registration records an OBSERVED
    scenario crossing, which branch-and-weight has no conditioning step to integrate against. What
    makes the correction exactly zero is not that the decision carries no graph - `b_gaps` is built
    from the scenario spot and does - but that BASE VALUATION HAS ONE SCENARIO, so
    `boundary_weights` finds no spread and returns its empty-kernel branch.

    The observable is byte identity of the WHOLE first-order frame. Knock-out 0.6597964784,
    knock-in -0.1100582535, binary 2.259431076. The day base valuation grows a second scenario this
    gate goes red, and it should.
    """
    v_off, off = _barrier_run(barrier, greeks='First', on=False, sims=1 << 12,
                              digital=digital, **over)
    v_on, on = _barrier_run(barrier, greeks='First', on=True, sims=1 << 12,
                            digital=digital, **over)
    assert v_off == v_on, (v_off, v_on)
    assert np.array_equal(off['Results']['Greeks_First'].values,
                          on['Results']['Greeks_First'].values), (
        'the boundary correction the switch skips was NOT exactly zero: delta reads {!r} with the '
        'registration and {!r} without it'.format(_eq_first(off), _eq_first(on)))


@pytest.mark.parametrize('rebate', [0.0, 5.0, 12.5])
def test_in_out_parity_is_the_survival_ledger_read_off_two_prices(rebate):
    """CONSERVATION on real documents, through prices instead of internals. In a zero-rate world
    the knock-out pays the rebate `sum_j fired_j` times and the knock-in's parity leg `alive_T`
    times, so

        KO + KI == vanilla + rebate

    exactly when `sum_j alive_{j-1} * (1 - p_j) + alive_final == 1`. A lost mass shows up here as
    an error proportional to the rebate and in neither price alone, which is why the rebate is
    walked: at 0 this is plain parity. Residuals 1.78e-15 / 1.78e-15 / 0.00e+00 at 32768 paths.
    """
    ko, _ = _barrier_run(90.0, on=True, Barrier_Type='Down_And_Out', Cash_Rebate=rebate)
    ki, _ = _barrier_run(90.0, on=True, Barrier_Type='Down_And_In', Cash_Rebate=rebate)
    black = _black_call()[0]
    assert abs(ko + ki - black - rebate) < 1e-11, (
        'in-out parity fails by {:.3e} at a rebate of {}: KO {:.12f} + KI {:.12f} against a '
        'vanilla of {:.12f}. The rebate half of that residual IS the survival ledger'.format(
            ko + ki - black - rebate, rebate, ko, ki, black))


def test_parity_survives_to_second_order_on_the_smooth_path():
    """The same identity at DELTA and GAMMA - a statement about WHERE the kink term went.
    `accrual_kink_term` is added to `surv_payoff`, before the in-out parity subtraction, so both
    legs carry one corrected quantity and `KO + KI` telescopes at every order.

    16384 paths, both barrier sides, against Black 9.9476449660 / 0.5497382248 / 0.0158335075:
    residuals 0.0e+00 at value and delta, 0.0e+00 and 3.5e-18 at gamma.

    MUTATION: gating that addition on `direction == BARRIER_OUT` leaves gamma at 0.0097725840 and
    0.0071409703, missing Black by 0.0010800469 - the term's own contribution. That mutant passes
    the ladder gate below (the leg it reads still carries the term) and removing the term from BOTH
    legs passes this one (parity is about symmetry, not presence). Neither alone is enough.
    """
    black, black_d, black_g = _black_call()
    for barrier, out_type, in_type in ((90.0, 'Down_And_Out', 'Down_And_In'),
                                       (130.0, 'Up_And_Out', 'Up_And_In')):
        ko, out_ko = _barrier_run(barrier, greeks='All', on=True, sims=1 << 14,
                                  Barrier_Type=out_type)
        ki, out_ki = _barrier_run(barrier, greeks='All', on=True, sims=1 << 14,
                                  Barrier_Type=in_type)
        assert abs(ko + ki - black) < 1e-11, (barrier, ko, ki, black)
        assert abs(_eq_first(out_ko) + _eq_first(out_ki) - black_d) < 1e-11, (
            barrier, _eq_first(out_ko), _eq_first(out_ki), black_d)
        assert abs(_eq_second(out_ko) + _eq_second(out_ki) - black_g) < 1e-11, (
            'second-order parity fails at H={}: the terminal kink term is reaching one leg and '
            'not the other ({:.12g} + {:.12g} against {:.12g})'.format(
                barrier, _eq_second(out_ko), _eq_second(out_ki), black_g))


def test_the_live_barriers_gamma_lands_on_its_own_ladder():
    """GAMMA FLOWS on a LIVE barrier, and it is the derivative of the delta reported. Without the
    switch the observed crossing registers a latch whose correction is a detached coefficient that
    cannot be differentiated twice.

    A live knock-out has no closed form, so the reading is a CRN ladder of the corrected delta -
    agreement AND flatness, since differencing across a discontinuity scatters with the bump.
    32768 paths: gamma 0.00969411 against a ladder best of 0.00974971, 0.57% at 2.12% flatness.

    The never-knocking control (barrier 1.0) is a European: value 10.0082 against Black 9.9476
    (0.61%), gamma 0.01560843 against 0.01583351 (1.42%) - an exact zero without the terminal kink
    term, which is what makes landing on Black the kill rather than the confirmation.

    MUTATION: with the term never built the live barrier still reports 0.00861379 rather than an
    exact zero, because every barrier decision is a `Phi` whose curvature AAD carries, and misses
    its unchanged ladder by 13.19%. The term is worth 0.00108005 of 0.00969411 - which is why this
    is a ladder rather than a comparison against zero.
    """
    with pytest.raises(utils.SecondOrderRefused):
        _barrier_run(90.0, greeks='All', on=False)

    sims = 1 << 15
    _, live = _barrier_run(90.0, greeks='All', on=True, sims=sims)
    gamma = _eq_second(live)
    rung = ladder(
        price=lambda s: _eq_first(_barrier_run(
            90.0, greeks='First', on=True, sims=sims, spot=s)[1]),
        aad=gamma, base=bb.SPOT, rungs=(1e-3, 2e-3, 5e-3, 1e-2))
    assert rung.agrees(tol=0.05), (
        'the live barrier\'s gamma is not the derivative of its own delta\n{}'.format(rung))

    value, european = _barrier_run(1.0, greeks='All', on=True, sims=sims)
    black, _, black_gamma = _black_call()
    assert _rel(value, black) < 0.02, (value, black)
    assert _rel(_eq_second(european), black_gamma) < 0.04, (
        'the never-knocking barrier reports {:.8f} against Black {:.8f} - without the terminal '
        'kink term this number is an exact zero, so it is the term that is being read'.format(
            _eq_second(european), black_gamma))


@pytest.mark.parametrize('over', [{}, {'Barrier_Type': 'Down_And_In'}], ids=['knock-out', 'knock-in'])
def test_the_recompute_node_replays_the_barriers_smooth_callable(over):
    """RECOMPUTE COMPOSITION: the switch and the node are orthogonal. The node's contract is one
    function called twice, so the smooth path must be that function with everything it reads
    arriving through theta. Value and the whole first-order frame come back bit-identical taped or
    replayed, on both legs; the knock-in row is worth running because its parity vanilla rides
    `sd_to_expiry` through theta.
    """
    taped_v, taped = _barrier_run(90.0, greeks='First', on=True, sims=1 << 13, **over)
    # the same document with the node switched on beside the switch
    _, replay = rf.run_baseval(_barrier_config(90.0, **over), overrides={
        'MCMC_Simulations': 1 << 13, 'Random_Seed': 1, 'Greeks': 'First',
        'Branch_And_Weight': 'Yes', 'Recompute_Inner_MC': 'Yes'})
    rows = replay['Results']['mtm']
    replay_v = float(rows[rows['Reference'] == 'BARR1']['Value'].iloc[0])
    assert taped_v == replay_v, (taped_v, replay_v)
    assert np.array_equal(taped['Results']['Greeks_First'].values,
                          replay['Results']['Greeks_First'].values), (
        'the recompute node did not replay the same smooth callable: {!r} taped, {!r} '
        'replayed'.format(_eq_first(taped), _eq_first(replay)))


HN_PARAMS = {'Omega': 2.757e-06, 'Alpha': 7.784e-08, 'Beta': 1.079e-03,
             'Gamma_Star': -3529.45, 'H0': 7.027e-05, 'Property_Aliases': None}


def test_a_heston_nandi_document_under_the_switch_does_not_price(tmp_path):
    """GBM ONLY, on a document rather than at the seam. The TARF is pinned to a Heston-Nandi spot
    model and asked for the switch: the deal is SKIPPED with a refusal naming the model, the stride
    that owns the conditional law it would need, and both remedies. The same document prices with
    the switch off, which is what makes the refusal the switch's and not the market data's."""
    def hn_job(**calc):
        job = _tarf_doc(**calc)
        market = job['Calc']['MergeMarketData']['ExplicitMarketData']
        market['Price Factors']['HestonNandiModelParameters.EUR'] = HN_PARAMS
        market['Valuation Configuration'] = {'FXTARFOptionDeal': {'SpotModel': 'HestonNandi'}}
        return job

    priced, _, _ = _run_doc(hn_job(), tmp_path, 'hn_off')
    assert np.isfinite(priced) and priced != 0.0, (
        'the Heston-Nandi document does not price with the switch OFF either, so the refusal '
        'below would be attributable to the market data rather than to the switch')
    refused, _, log = _run_doc(_smooth(hn_job()), tmp_path, 'hn_on', debug=True)
    assert math.isnan(refused), 'the deal priced under a model the switch has no law for'
    # the loader logs the exception's ARGS, so every quote arrives escaped
    log = log.replace('\\', '')
    assert 'HestonNandi' in log and 'stride' in log.lower(), log[-1200:]
    assert 'hn_cdf_logret' in log, 'the refusal cites where its Phi would have to come from'
    assert 'GBM' in log and "Branch_And_Weight: 'No'" in log, 'a refusal names its remedies'


# ======================================================================================
# THE AUTOCALL - the fourth product, and the put leg that used to defer it
# ======================================================================================

AUTOCALL_TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'fixtures', 'autocall_job.json')

# THE AUTOCALL'S OWN WORLD, written over the fixture the way `_world` writes the FX one: flat and
# DIFFERENT (4% funding against a 1% dividend, so the carry is live at 3%) and a flat surface,
# which is what makes an interval strip exactly the quote.
AC_R, AC_Q, AC_SIGMA = 0.04, 0.01, 0.25
AUTOCALL_SPOT = AC_STRIKE = 100.0
AC_COUPON, AC_UNITS = 0.08, 10.0
AC_DAYS = (180, 365)


def _ac_surface(vol):
    """The fixture's explicit (moneyness, tenor, vol) grid, held flat."""
    return {'.Curve': {'meta': [],
                       'data': [[m, t, vol] for m in (0.8, 1.0, 1.2) for t in (0.02, 2.0)]}}


# The autocall's own document: the fixture is a single coupon with no put barrier, the one
# configuration that cannot see what this section is about.
def _autocall_doc(put_barrier=0.0, coupon_days=AC_DAYS, threshold=1.0, rebate=None, greeks='No',
                  spot=AUTOCALL_SPOT, sims=1 << 14, seed=1, same_day=False):
    with open(AUTOCALL_TEMPLATE) as f:
        job = json.load(f)
    pf = job['Calc']['MergeMarketData']['ExplicitMarketData']['Price Factors']
    pf['InterestRate.USD']['Curve'] = _flat(AC_R)
    pf['DividendRate.EQ']['Curve'] = _flat(AC_Q)
    pf['VolatilityGrid.EQ']['Surface'] = _ac_surface(AC_SIGMA)
    pf['EquityPrice.EQ']['Spot'] = spot
    deal = job['Calc']['Deals']['Deals']['Children'][0]['Instrument']['.Deal']
    deal.update(Strike_Price=AC_STRIKE, Units=AC_UNITS)
    dates = [_stamp(d) for d in coupon_days]
    deal['Autocall_Coupons'] = [[d, AC_COUPON] for d in dates]
    deal['Autocall_Thresholds'] = [[d, threshold] for d in dates]
    # one price fixing per coupon date IS the no-averaging branch - the only one in scope
    deal['Price_Fixing'] = [[d, 0.0] for d in dates]
    deal['Expiry_Date'] = dates[-1]
    # `Barrier` is declared as a FRACTION of the strike (instruments.py multiplies it through), so
    # 0.7 is a knock-in 30% below the strike and 1.0 sits exactly on it
    deal['Barrier'] = put_barrier
    deal['Barrier_Dates'] = [dates[-1]] if put_barrier else []
    if rebate is not None:
        # NOT a declared field - `pv_MC_AutoCallSwap` reads it with `.get`, so only a document
        # carrying one exercises this term (asserted below)
        deal['Rebate'] = rebate
    if same_day:
        # a coupon dated ON the base date is decided off the SIMULATED spot, giving the latch a
        # graph-carrying gap. Its threshold sits ABOVE the spot: it registers a decision without
        # taking one, so the rest of the deal still prices and the ladder has something to converge on
        d0 = _stamp(0)
        deal['Autocall_Coupons'] = [[d0, AC_COUPON]] + deal['Autocall_Coupons']
        deal['Autocall_Thresholds'] = [[d0, 1.02]] + deal['Autocall_Thresholds']
        deal['Price_Fixing'] = [[d0, spot]] + deal['Price_Fixing']
    job['Calc']['Calculation'].update(
        {'Greeks': greeks, 'MCMC_Simulations': sims, 'Random_Seed': seed})
    return job


def _barrier_on_the_base_date(job):
    """Move the deal's ONLY barrier date onto the coupon the base date observes."""
    job['Calc']['Deals']['Deals']['Children'][0]['Instrument']['.Deal'][
        'Barrier_Dates'] = [_stamp(0)]
    return job


def _averaging(job, by='fixings'):
    """Push the SAME deal onto the averaging arm, the two ways `calc_dependencies` decides it:
    more than one price fixing per coupon, or a barrier date off the coupon dates."""
    deal = job['Calc']['Deals']['Deals']['Children'][0]['Instrument']['.Deal']
    if by == 'fixings':
        deal['Price_Fixing'] = [[_stamp(d), 0.0] for d in (170, 180, 355, 365)]
    else:
        deal['Barrier_Dates'] = [_stamp(300)]
    return job


def _autocall_reference(spot, sigma, put_barrier=0.7, rebate=0.0, threshold=1.0,
                        n_out=1200, n_in=800):
    """The two-coupon autocall with a put barrier as a nested region integral, differentiable end
    to end. Written from the DEAL and the flat world above.

    Per coupon the line splits at the autocall threshold `K = threshold * strike`: above it the
    deal REDEEMS and pays its coupon, below it survives, and a surviving path is worth nothing at
    the end. The put leg pays over `{S <= min(B, K)}` - the second coupon's own survival has
    already truncated the law to `{S <= K}`.

    That intersection is the whole content of the closed form under test, so it is stated as a
    region LIMIT rather than an indicator: `_seg`'s ends move with theta. Nothing here knows a
    lognormal closed form.
    """
    dt1, dt2 = AC_DAYS[0] / 365.0, (AC_DAYS[1] - AC_DAYS[0]) / 365.0
    D1, D2 = (math.exp(-AC_R * d / 365.0) for d in AC_DAYS)
    carry = AC_R - AC_Q
    s1, s2 = sigma * math.sqrt(dt1), sigma * math.sqrt(dt2)
    m1 = (carry - 0.5 * sigma ** 2) * dt1
    m2 = (carry - 0.5 * sigma ** 2) * dt2
    K = torch.as_tensor(threshold * AC_STRIKE, dtype=DT)
    b_eff = torch.minimum(torch.as_tensor(put_barrier * AC_STRIKE, dtype=DT), K)
    z1K = (torch.log(K / spot) - m1) / s1

    def tail(z1):
        """Everything the second coupon is worth, given the first one's own draw."""
        S1 = spot * torch.exp(m1 + s1 * z1)
        S1c = S1[..., None]
        z2K = (torch.log(K / S1) - m2) / s2
        z2B = (torch.log(b_eff / S1) - m2) / s2
        top = torch.full_like(z2K, Z_INF)
        value = AC_COUPON * D2 * _seg(z2K, top, torch.ones_like, n_in)
        return value + D2 * _seg(
            torch.full_like(z2B, -Z_INF), z2B,
            lambda w: rebate - 1.0 + S1c * torch.exp(m2 + s2 * w) / AC_STRIKE, n_in)

    total = AC_COUPON * D1 * _seg(z1K, torch.as_tensor(Z_INF, dtype=DT), torch.ones_like, n_out)
    total = total + _seg(torch.as_tensor(-Z_INF, dtype=DT), z1K, tail, n_out)
    return AC_UNITS * total


AC_ZERO_DAYS = (180, 270, 365)
AC_ZERO_BARRIER = 0.7      # the document reads ONE barrier, from here


def _zero_coupon_doc(coupon=0.0, barrier=True, **kwargs):
    """The autocall with `coupon` on its middle row, and the deal's only barrier date on that row
    or nowhere.

    At `coupon=0.0` this is the refused document either way: nothing about its SHAPE keeps it off
    the fast arm (`ac_dates` counts the zero row and day 270 IS a coupon date), so the pricer's
    `if coup > 0` would be FALSE at that `j`. At `coupon=AC_COUPON` the same dates, barrier and
    fixings price on that arm - the control, with only the number on that row moving.
    """
    job = _autocall_doc(AC_ZERO_BARRIER if barrier else 0.0, coupon_days=AC_ZERO_DAYS, **kwargs)
    deal = job['Calc']['Deals']['Deals']['Children'][0]['Instrument']['.Deal']
    dates = [_stamp(d) for d in AC_ZERO_DAYS]
    deal['Autocall_Coupons'] = [[dates[0], AC_COUPON], [dates[1], coupon], [dates[2], AC_COUPON]]
    deal['Barrier_Dates'] = [dates[1]] if barrier else []
    return job


def _ac_table(**kwargs):
    return _table(_autocall_reference, base_spot=AUTOCALL_SPOT, base_sigma=AC_SIGMA, **kwargs)


def _ac_run(job, tmp_path, name):
    """(value, delta, vega) and, where the run asked for them, (gamma, vanna) - off the equity."""
    value, out, _ = _run_doc(job, tmp_path, name)
    got = {'value': value, 'delta': _first(out, factor='EquityPrice.EQ')}
    frame = out['Results']['Greeks_First']
    column = [c for c in frame.columns if c != 'Value'][0]
    got['vega'] = sum(float(frame.loc[i, column]) for i in frame.index
                      if str(i[0]) == 'EquityPriceVol.EQ')
    if 'Greeks_Second' in out['Results']:
        got['gamma'], got['vanna'] = _second(
            out, spot='EquityPrice.EQ', vol='EquityPriceVol.EQ')
    return got


# ======================================================================================
# THE PUT LEG - the deferral, flipped
# ======================================================================================

@pytest.mark.parametrize('rebate', [0.0, 0.05], ids=['no rebate', 'rebate 0.05'])
@pytest.mark.parametrize('put_barrier', [1.0, 0.7],
                         ids=['put barrier AT the strike', 'put barrier BELOW the strike'])
def test_the_autocall_put_leg_lands_on_its_own_ladder(put_barrier, rebate, tmp_path):
    """THE DEFERRAL, FLIPPED. The blocker was never the averaging branch (the two arms are a plain
    `if factor_dep['no_averaging']` / `else` inside `sim_spot`, sharing no line of pricing) but that
    the deal takes TWO decisions per fixing and only the coupon trigger was integrated: the put
    barrier was a bare indicator whose payoff `rebate - (1 - S/strike)` is zero only ON the strike
    with no rebate and a JUMP anywhere below it.

    Delta against a CRN ladder of the reported value, four rungs 1e-3 to 1e-2, two coupons, spot
    and strike 100, 25% vol:

        put barrier      rebate    CRISP (the old estimator)      SMOOTH (this switch)
        1.0 on strike     0.00      0.15% and flat                0.07% and flat
        1.0 on strike     0.05      0.14% and flat                0.09% and flat
        0.7 below         0.00      16.2%, ladder scattering      0.16% and flat
        0.7 below         0.05      14.6%, ladder scattering      0.15% and flat

    The on-strike rows are the control and were always exact, which is what says the diagnosis was
    the jump. The rebate rows are not decoration: the closed form carries `Rebate` inside the
    effective strike `strike * (1 - rebate)`, and getting that wrong moves the value 14% here.
    """
    def doc(**kw):
        return _smooth(_autocall_doc(put_barrier, rebate=rebate, sims=1 << 16, **kw))

    aad = _first(_run_doc(doc(greeks='First'), tmp_path, 'ac_d')[1], factor='EquityPrice.EQ')
    assert abs(aad) > 1e-6, 'a live autocall must have a spot delta to compare against'
    rung = ladder(price=lambda s: _run_doc(doc(spot=s), tmp_path, 'ac_v')[0],
                  aad=aad, base=AUTOCALL_SPOT, rungs=(1e-3, 2e-3, 5e-3, 1e-2))
    assert rung.agrees(tol=0.02), (
        'the smooth autocall delta is not the derivative of the value the same document reports - '
        'the put leg is the only decision on this deal that is not the coupon trigger, so this is '
        'where a wrong effective strike, a missed `min(B, K)` or a dropped 1/p shows up\n'
        '{}'.format(rung))


def test_the_crisp_put_leg_still_jumps_and_the_switch_is_what_fixes_it(tmp_path):
    """THE KILL, kept as a live measurement. The SAME document under the crisp estimator misses its
    own ladder wherever the put payoff jumps - 16.2% at a 70% barrier - while landing on it where
    that payoff is continuous. Goes red if the crisp path is ever quietly smoothed, which would
    make the row above pass for the wrong reason."""
    def miss(put_barrier, smooth):
        def job(**kw):
            built = _autocall_doc(put_barrier, sims=1 << 16, **kw)
            return _smooth(built) if smooth else built

        aad = _first(_run_doc(job(greeks='First'), tmp_path, 'k_d')[1], factor='EquityPrice.EQ')
        rung = ladder(price=lambda s: _run_doc(job(spot=s), tmp_path, 'k_v')[0],
                      aad=aad, base=AUTOCALL_SPOT, rungs=(1e-3, 2e-3, 5e-3, 1e-2))
        return abs(rung.best - rung.aad) / max(abs(rung.aad), 1e-30), rung

    jump_crisp, rung = miss(0.7, smooth=False)
    assert jump_crisp > 0.10, (
        'the CRISP autocall put leg no longer misses its own bump ladder ({:.2%}) - either the '
        'indicator has been smoothed on the default path, which is a re-baseline nobody asked '
        'for, or this document no longer has a jumping put leg to measure\n{}'.format(
            jump_crisp, rung))
    flat_crisp, rung = miss(1.0, smooth=False)
    assert flat_crisp < 0.02, (
        'the crisp autocall is inexact even where its put payoff is CONTINUOUS, so the diagnosis '
        'this section rests on is blaming the wrong thing\n{}'.format(rung))
    jump_smooth, _ = miss(0.7, smooth=True)
    assert jump_smooth < 0.02 and jump_crisp / max(jump_smooth, 1e-30) > 10.0, (
        'the switch is not what closes the gap: crisp {:.2%} against smooth {:.2%}'.format(
            jump_crisp, jump_smooth))


# ======================================================================================
# THE QUADRATURE TABLE
# ======================================================================================

@pytest.mark.parametrize('rebate', [0.0, 0.05], ids=['no rebate', 'rebate 0.05'])
def test_the_two_coupon_autocall_lands_on_the_quadrature_table(rebate, tmp_path):
    """THE TABLE. A two-coupon autocall with a JUMPING put barrier under the switch at
    `Greeks: 'All'`, against the differentiable region integral (spot and strike 100, thresholds
    1.0, coupon 0.08, put barrier 0.7, 10 units, r 4% q 1%, vol 25%, coupons at 180 and 365 days,
    262144 inner paths):

                        engine        quadrature      relative
        value          +0.2202478     +0.2210455       0.361%
        delta          +0.03690466    +0.03684703      0.156%
        vega           -3.801398      -3.795778        0.148%
        gamma          -0.001890364   -0.001887619     0.145%
        vanna          +0.0956566     +0.09563154      0.026%

    With a 0.05 rebate: +0.2560977 / +0.03408181 / -3.368577 / -0.001699146 / +0.07971977 against
    +0.2568038 / +0.03403085 / -3.363476 / -0.001696697 / +0.07967986. Crisp, recorded not gated:
    value +0.2191984 (0.836%), delta +0.03087349 - 16.2% out.

    The vega row is what says the vol identification is right before vanna means anything.

    THE `1/p` IS WHAT THIS TABLE ARBITRATES: the put leg's sample is drawn from the law truncated
    to `{S <= K}` and already carries that survival in `L`, so the analytic term is a CONDITIONAL
    expectation - the partial moments divided by `p`. Without that division the same code reads
    +0.254135 against +0.2210455 (15.0%) and -0.081302 against -0.2128747 on the on-strike
    document (61.8%).
    """
    ref = _ac_table(rebate=rebate)
    got = _ac_run(_smooth(_autocall_doc(0.7, rebate=rebate, greeks='All', sims=1 << 18)),
                  tmp_path, 'ac_table')
    tol = {'value': 0.01, 'delta': 0.01, 'vega': 0.01, 'gamma': 0.01, 'vanna': 0.02}
    bad = {k: (got[k], ref[k], _rel(got[k], ref[k])) for k in tol if _rel(got[k], ref[k]) > tol[k]}
    assert not bad, bad


def test_the_put_barrier_above_the_threshold_is_the_whole_surviving_set(tmp_path):
    """`min(B, K)` IS LOAD-BEARING. A put barrier at or above the autocall threshold pays on EVERY
    surviving path; integrating the raw `{S <= B}` would count mass the draw was truncated away
    from. On the strike (`B == K`) the two agree; the gate walks both.

    262144 paths, on-strike: value -0.2136573 against -0.2128747 (0.368%), delta +0.05835698
    against +0.0583142 (0.073%), gamma -0.002207117 against -0.002206393 (0.033%).
    """
    for put_barrier in (1.0, 1.2):
        ref = _ac_table(put_barrier=put_barrier)
        got = _ac_run(_smooth(_autocall_doc(put_barrier, greeks='All', sims=1 << 18)),
                      tmp_path, 'ac_above')
        for key, tol in (('value', 0.01), ('delta', 0.01), ('gamma', 0.01)):
            assert _rel(got[key], ref[key]) < tol, (
                'put barrier {}: {} reads {:.8g} against {:.8g} ({:.3%})'.format(
                    put_barrier, key, got[key], ref[key], _rel(got[key], ref[key])))


# ======================================================================================
# OFF IS OFF, and what the switch is attributable for on this product
# ======================================================================================

@pytest.mark.parametrize('kwargs,name', [
    ({}, 'one coupon, no put barrier'),
    ({'put_barrier': 0.7}, 'two coupons, jumping put barrier'),
    ({'put_barrier': 0.7, 'rebate': 0.05}, 'two coupons, put barrier and rebate'),
    ({'coupon_days': (91, 182, 273), 'put_barrier': 0.6}, 'three coupons'),
    ({'same_day': True, 'put_barrier': 0.7}, 'a coupon observed on the base date')])
def test_off_is_off_on_an_autocall_document(kwargs, name, tmp_path):
    """The key ABSENT and the key written 'No' are the same run - value, the whole mtm frame and
    every first-order greek, by `np.array_equal`. The identity stops there: 'Yes' legitimately
    changes the estimator on this product, so it is checked against the quadrature above instead.
    """
    absent, out_a, _ = _run_doc(_autocall_doc(greeks='First', **kwargs), tmp_path, 'ac_absent')
    written, out_w, _ = _run_doc(
        _smooth(_autocall_doc(greeks='First', **kwargs), on=False), tmp_path, 'ac_no')
    assert absent == written, (name, absent, written)
    assert out_a['Results']['mtm'].equals(out_w['Results']['mtm']), 'the reported frame moved'
    assert np.array_equal(out_a['Results']['Greeks_First'].values,
                          out_w['Results']['Greeks_First'].values), 'a first-order greek moved'


def test_an_autocall_with_no_put_barrier_is_bit_identical_under_the_switch(tmp_path):
    """ATTRIBUTION. The no-averaging loop was ALREADY the construction for its coupon trigger -
    fired branch `(1 - p) * L * coup * D_j`, truncated continuation - so with no put barrier the
    two estimators agree TO THE BIT at value and at both greek blocks. The `Greeks: 'All'` half
    also says the registration this document never made is not what is being measured."""
    crisp, out_c, _ = _run_doc(_autocall_doc(greeks='All'), tmp_path, 'ac_nobar_off')
    smooth, out_s, _ = _run_doc(_smooth(_autocall_doc(greeks='All')), tmp_path, 'ac_nobar_on')
    assert crisp == smooth, (crisp, smooth)
    assert np.array_equal(out_c['Results']['Greeks_First'].values,
                          out_s['Results']['Greeks_First'].values)
    assert np.array_equal(out_c['Results']['Greeks_Second'].values,
                          out_s['Results']['Greeks_Second'].values), (
        'the autocall moved at SECOND order on a document whose only decision was already smooth')


def test_the_switch_claims_exactly_four_products_and_no_more(tmp_path):
    """SCOPE. `Branch_And_Weight` is honoured by `pv_MC_Tarf`, `pv_MC_Accumulator`,
    `pv_discrete_barrier_option` and `pv_MC_AutoCallSwap`, and the field's declaration enumerates
    them. Goes red the day a FIFTH pricer reads the switch without also earning a quadrature table,
    a CRN ladder, conservation on a live run and a named refusal for the arm it does not reach."""
    field = {f.key: f for f in calculation.Base_Revaluation.fields}['Branch_And_Weight']
    for named in ('TARF', 'accumulator', 'discrete barrier', 'autocall'):
        assert named in field.description, (
            'the declaration must enumerate the products it actually covers, and it is missing '
            '{!r}: {}'.format(named, field.description))
    readers = sorted(name for name, fn in vars(pricing).items()
                     if name.startswith('pv_') and inspect.isfunction(fn) and
                     'branch_and_weight(shared' in inspect.getsource(fn))
    assert readers == ['pv_MC_Accumulator', 'pv_MC_AutoCallSwap', 'pv_MC_Tarf',
                       'pv_discrete_barrier_option'], (
        'a pricer reads the switch that this declaration does not enumerate, or one that does no '
        'longer reads it: {}'.format(readers))


# ======================================================================================
# THE SECOND-ORDER BLOCK, CONSERVATION, AND THE ARM THE SWITCH DOES NOT REACH
# ======================================================================================

def test_the_autocall_gamma_flows_under_the_switch_and_is_refused_without_it(tmp_path):
    """`Greeks: 'All'` on an autocall whose first coupon is OBSERVED is REFUSED without the switch
    and RUNS with it, and the Hessian entry is the derivative of the delta the same run reports.

    An observed coupon is decided off the scenario spot, so the pricer registers a
    `LatchedBoundarySet`; under the switch the decision is integrated instead of corrected and
    nothing registers.

    65536 paths: the spot-spot entry against a CRN ladder of the reported delta, four rungs 1e-3 to
    1e-2, inside 2% at better than 10% flatness. What says nothing was lost is the next gate, which
    needs a document with NO put barrier - with one the switch legitimately moves first order too
    (delta 3.08 crisp against 3.70 smooth).
    """
    with pytest.raises(utils.SecondOrderRefused) as refusal:
        _run_doc(_autocall_doc(0.7, same_day=True, greeks='All'), tmp_path, 'ac_all_off')
    assert 'boundary correction' in str(refusal.value)

    _, out, _ = _run_doc(_smooth(_autocall_doc(0.7, same_day=True, greeks='All', sims=1 << 16)),
                         tmp_path, 'ac_all_on')
    gamma, _ = _second(out, spot='EquityPrice.EQ', vol='EquityPriceVol.EQ')
    assert np.isfinite(gamma)

    def delta_at(spot):
        return _first(_run_doc(
            _smooth(_autocall_doc(0.7, same_day=True, greeks='First', sims=1 << 16, spot=spot)),
            tmp_path, 'ac_ladder')[1], factor='EquityPrice.EQ')

    result = ladder(price=delta_at, aad=gamma, base=AUTOCALL_SPOT,
                    rungs=(1e-3, 2e-3, 5e-3, 1e-2))
    assert result.agrees(tol=0.02), (
        'the reported autocall gamma is not the derivative of the reported delta\n{}'.format(
            result))


def test_the_autocall_registration_is_superseded_not_lost(tmp_path):
    """THE OTHER HALF OF THE SUPERSESSION, on a document where the switch has nothing else to
    change: an observed coupon, no put barrier. The registration is skipped, the Hessian flows,
    and first order comes back BIT FOR BIT - base valuation has one scenario, so the boundary
    correction is exactly zero by construction.
    """
    with pytest.raises(utils.SecondOrderRefused):
        _run_doc(_autocall_doc(same_day=True, greeks='All'), tmp_path, 'ac_sd_off')
    _, out, _ = _run_doc(_smooth(_autocall_doc(same_day=True, greeks='All')),
                         tmp_path, 'ac_sd_on')
    assert np.isfinite(_second(out, spot='EquityPrice.EQ', vol='EquityPriceVol.EQ')[0])
    crisp, out_c, _ = _run_doc(_autocall_doc(same_day=True, greeks='First'),
                               tmp_path, 'ac_sd1_off')
    smooth, out_s, _ = _run_doc(_smooth(_autocall_doc(same_day=True, greeks='First')),
                                tmp_path, 'ac_sd1_on')
    assert crisp == smooth, (crisp, smooth)
    assert np.array_equal(out_c['Results']['Greeks_First'].values,
                          out_s['Results']['Greeks_First'].values), (
        'the skipped registration was carrying flux after all - it is not zero here, and the '
        'supersession is losing a derivative rather than replacing an estimator')


def test_an_observed_breach_is_data_and_the_switch_does_not_touch_it(tmp_path):
    """WHAT THE SWITCH DOES NOT SMOOTH. A barrier date on an OBSERVED coupon has no conditioning
    step: `Sj` is the scenario's own spot, so the breach is DATA and its indicator stays exact. The
    document puts the deal's only barrier date on the base date's coupon, spot BELOW the strike and
    barrier ABOVE it, so the breach pays a real -0.10 per unit rather than a vacuous zero.

    Under the switch that leg is untouched and the run is BIT-IDENTICAL at value and first order.
    The third door into this branch - a barrier on a ZERO coupon row - no longer exists: the gate
    below refuses that document at the loader.
    """
    def job(**kw):
        return _barrier_on_the_base_date(
            _autocall_doc(1.1, same_day=True, spot=90.0, greeks='First', **kw))

    crisp, out_c, _ = _run_doc(job(), tmp_path, 'ac_obs_off')
    smooth, out_s, _ = _run_doc(_smooth(job()), tmp_path, 'ac_obs_on')
    assert crisp == smooth, (
        'an OBSERVED breach moved under the switch - its `Sj` is the scenario\'s own spot, so '
        'there is no interval to integrate it against and it must stay an exact indicator: '
        '{} against {}'.format(crisp, smooth))
    assert np.array_equal(out_c['Results']['Greeks_First'].values,
                          out_s['Results']['Greeks_First'].values)
    # and the leg is live rather than a zero agreeing with a zero
    without, _, _ = _run_doc(_smooth(_autocall_doc(
        0.0, same_day=True, spot=90.0, greeks='First')), tmp_path, 'ac_obs_none')
    assert _rel(crisp, without) > 0.05, (
        'the observed put leg pays nothing on this document, so the identity above is two zeros '
        'agreeing: {} against {} with no barrier at all'.format(crisp, without))


@pytest.mark.parametrize('barrier', [True, False],
                         ids=['a barrier on the row', 'nothing on the row'])
def test_a_zero_coupon_row_refuses_by_name(barrier, tmp_path):
    """THE THIRD SUB-CASE, RETIRED RATHER THAN REPAIRED - the Known-defects row, closed.

    A row quoted ZERO runs no coupon block on the fast arm, and there are two readings of that:
    `coupon_index` never advances, so the NEXT coupon takes this row's interval (0.466196 against
    0.487692, 4.41%, against the identical deal with the row deleted); and where the barrier is
    dated ON that row its breach indicator reads the PREVIOUS fixing's spot (+0.394809 against a
    deal-written reference's +0.317939, 24.2%). The second needs a barrier, the first does not,
    which is why both documents are walked.

    THE RULING IS THAT THERE IS NO SUCH DEAL. `calc_dependencies` refuses the document by name, and
    FATALLY (`utils.UnpriceableSchedule`, the FRA's precedent) - a refusal swallowed into a zero
    mark on a job that then succeeds has said nothing. Deleting the row is the remedy either way.

    Three assertions: the run FAILS, not a KeyError downstream or a skipped deal; the message names
    the deal, the row's date, the coupon and the remedy, and names the stale spot only where there
    is a barrier to read it; and the same document with a real coupon still prices on the same arm,
    so what is refused is the zero and not the shape.
    """
    with pytest.raises(utils.UnpriceableSchedule) as raised:
        _run_doc(_zero_coupon_doc(barrier=barrier), tmp_path, 'zc_refused')
    said = str(raised.value)
    row_date = (BASE_DAY + datetime.timedelta(days=AC_ZERO_DAYS[1])).isoformat()
    assert 'AC1' in said and row_date in said, (
        'the refusal does not name the deal and the date the author has to go and fix: '
        '{}'.format(said))
    assert 'quoted 0' in said and 'not a coupon' in said, said
    assert 'takes this row\'s interval in place of its own' in said, (
        'the mis-tenor is the reading that holds with or without a barrier, so it is the one the '
        'refusal always states: {}'.format(said))
    assert 'Author a real coupon' in said and 'delete the row' in said, (
        'a refusal names its remedy: {}'.format(said))
    assert ('PREVIOUS fixing' in said) == barrier, (
        'the stale-spot reading needs a barrier dated on the row to read it, so the message must '
        'claim it exactly when there is one: {}'.format(said))

    # THE CONTROL. Same dates, fixings and barrier - only the number on that row moves, and the
    # engine's own line says what it prices is still the fast arm
    priced, _, log = _run_doc(_zero_coupon_doc(coupon=AC_COUPON, barrier=barrier),
                              tmp_path, 'zc_real', debug=True)
    line, = [ln for ln in log.splitlines() if 'AUTOCALL AC1' in ln]
    assert 'averaging=0' in line and 'coupons=3 thresholds=3' in line, (
        'the control no longer reaches the arm the refusal guards, so it says nothing about '
        'what that refusal costs an ordinary document: {}'.format(line))
    assert np.isfinite(priced) and priced != 0.0, (
        'the control does not price either, so the refusal above is attributable to the document '
        'rather than to the zero on its middle row: {}'.format(priced))


def test_the_autocall_ledger_conserves_on_a_real_document(tmp_path):
    """CONSERVATION on a live run: per path, the mass fired plus the mass alive is the mass the
    strip opened with. Measured 0.0 with 37.6% surviving both coupons - a ledger that fired
    everything or nothing would conserve trivially."""
    _, _, log = _run_doc(_smooth(_autocall_doc(0.7, sims=1 << 12)), tmp_path, 'ac_ledger',
                         debug=True)
    lines = [ln for ln in log.splitlines() if 'LEDGER AUTOCALL' in ln]
    assert lines, 'the smooth path ran no ledger at all'
    residual = max(float(ln.split('conservation=')[1].split()[0]) for ln in lines)
    alive = [float(ln.split('alive=')[1].split()[0]) for ln in lines]
    assert residual < 1e-12, (residual, lines)
    assert 0.05 < max(alive) < 0.95, (
        'this document fires on almost none or almost all of its weight, so conservation on it '
        'says nothing: alive={}'.format(alive))


@pytest.mark.parametrize('by', ['fixings', 'barrier'])
def test_an_averaging_autocall_under_the_switch_refuses_by_name(by, tmp_path):
    """THE ARM THE SWITCH DOES NOT REACH, refused rather than quietly no-opped. Its termination is
    a smoothed per-inner-path weight (`smooth_heaviside_up`) with no crisp per-scenario decision to
    replace, and its breach is a hard indicator on the AVERAGE, whose conditioning law is the
    distribution of a mean of spots.

    Both ways `calc_dependencies` puts a deal on that arm are walked: more than one price fixing
    per coupon, and a barrier date off the coupon dates. The same documents price with the switch
    OFF, which is what makes the refusal the switch's and not the document's.
    """
    priced, _, _ = _run_doc(_averaging(_autocall_doc(0.7), by), tmp_path, 'avg_off')
    assert np.isfinite(priced) and priced != 0.0, (
        'the averaging document does not price with the switch OFF either, so the refusal below '
        'would be attributable to the deal rather than to the switch')
    refused, _, log = _run_doc(_smooth(_averaging(_autocall_doc(0.7), by)), tmp_path, 'avg_on',
                               debug=True)
    assert math.isnan(refused), 'the deal priced on an arm the switch has no conditioning law for'
    # the loader logs the exception's ARGS, so every quote arrives escaped
    log = log.replace('\\', '')
    assert 'averages' in log and 'AVERAGING arm' in log, log[-1200:]
    assert 'smooth_heaviside_up' in log, 'the refusal names what it will not put its name on'
    assert 'ONE price fixing per coupon' in log and "Branch_And_Weight: 'No'" in log, (
        'a refusal names its remedies')
