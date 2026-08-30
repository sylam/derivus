"""Branch-and-weight, PRIMITIVES section: the shared pieces every smooth product will stand on.

The construction (roadmap.md, "Branch and weight for TARFs and autocalls"): at each fixing, given
the state one conditioning step back, the trigger standardises to `zB`, the FIRED branch closes
analytically with weight `1 - Phi(zB)` and payoff `E[J(S_k) | fired]`, and the CONTINUING branch
draws `S_k` from the truncated law by `Phi^-1(U * Phi(zB))` carrying weight `Phi(zB)`. Every
indicator is gone, the barrier enters only through `Phi` and `Phi^-1`, and the estimator is
UNBIASED - not a smoothing.

Three rulings are gated here rather than assumed, because each one is a way the construction can be
built wrong and still look right:

1. THE FIRED BRANCH IS A CONDITIONAL EXPECTATION, NEVER `p x realised payoff`. The jump and the
   decision share the fixing, so the shortcut samples the payoff on the SURVIVING side of the
   trigger and weights it by the FIRED probability. Measured below on a one-fixing fixture, both
   halves exact: the full-gain convention reads 0.009525 against a truth of 0.105802 (11.11x low),
   the capped convention 0.007585 against 0.024929 (3.29x low), and written with the raw gain it
   reads -0.015531 - a SIGN error. None of it is variance; it does not shrink with paths.
2. SUPERSESSION IS A SWITCH. `Branch_And_Weight` is declared on `Base_Revaluation` and nowhere
   else, defaults to 'No', and off is the crisp one-step-survival path bit for bit.
3. GBM ONLY. The conditioning step must be the fixing interval's own lognormal law; a non-GBM spot
   model under the switch REFUSES by name, citing the stride as where its `Phi` comes from.

THE PRODUCTS ARE GATED IN THE SECOND HALF of this file, on real documents through
`Context.load_json` + `run_job`: the TARF and the accumulator against differentiable quadrature
references built here, the Rao-Blackwell variance ratio, the exact pin, the `Greeks: 'All'` flow
that used to refuse, conservation on a live run, the recompute composition and the daily-fixing
declaration. The headline readings are in `TARF_TABLE`'s docstring and each gate's own.

THE DISCRETE BARRIER IS THE THIRD PRODUCT and it needed no new estimator at all - its OSS sampler
is where the truncated-draw idiom came from, so the switch is a VERIFICATION rather than a rewrite
and its section reads accordingly: bit-identity walked across the whole deal family, the skipped
registration shown to lose nothing at first order, in-out parity standing in for the survival
ledger at value, delta and gamma, and a LIVE barrier's gamma on a CRN ladder where it used to be
refused outright. THE AUTOCALL IS DEFERRED, and the last gate in the file is the measurement that
defers it rather than an assertion that it is - its put leg is a second decision per fixing which
the construction does not integrate, so a document whose put barrier sits below the strike misses
its own bump ladder by 18-22% while the same document with the barrier ON the strike, where that
payoff is continuous, is exact to 0.00%.

THE REFERENCES ARE WRITTEN HERE AND SCIPY-FREE. Every tail expectation is checked against a
trapezoid quadrature built in this file out of `math.erf` and `math.exp` alone - it shares no line
with `pricing`, so an error in the closed form cannot hide behind the same error in its oracle.
The truncated sample is checked against the truncated normal's own analytic moments the same way.
The product references are the same idea one dimension up: a trapezoid whose LIMITS are the deal's
own decision levels, so `torch.autograd` differentiates through the regions the way Leibniz does
and the reference has a delta, a gamma and a vanna of its own - built out of `torch.exp` and the
normal DENSITY alone, with no lognormal closed form anywhere in it.
"""
import datetime
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

# the barrier's world and its deal, borrowed rather than re-authored: this file has no business
# owning a second EquityBarrierOption fixture when that one is already gated on its own bridge
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
    """Trapezoid over [lo, hi] on an odd node count - the whole oracle, and it knows no finance."""
    h = (hi - lo) / (n - 1)
    total = 0.5 * (f(lo) + f(hi))
    for i in range(1, n - 1):
        total += f(lo + i * h)
    return total * h


def _gaussian_tail(payoff, z_lo, z_hi):
    """``E[payoff(Z) * 1{z_lo < Z < z_hi}]`` by quadrature; +-12 stands in for infinity (the
    normal density there is 2e-33, forty times under the tolerances below)."""
    return _quadrature(lambda z: payoff(z) * _phi(z), max(z_lo, -12.0), min(z_hi, 12.0))


def _spot(S, m, s, z):
    return S * math.exp(m + s * z)


# one fixture, used by every closed-form gate below, chosen so nothing in it is degenerate:
# the drift is non-zero and differs in sign from nothing, the trigger sits above the strike (a
# TARF's moving barrier always does), and the fired tail carries real mass rather than 1e-9 of it
S0, DRIFT, VOL, STRIKE_K = 1.2500, 0.0170, 0.1400, 1.2000
Z_TRIGGER = (math.log(1.3100 / S0) - DRIFT) / VOL     # B = 1.31, an up trigger


# ======================================================================================
# THE TRUNCATED DRAW - ONE SPELLING (`pricing.oss_truncated_draw`)
# ======================================================================================

def _stratified(n):
    """Midpoint uniforms. Deterministic, so a distribution gate has no seed in it and no
    tolerance that has to absorb one: the quantile transform of a midpoint grid is a Riemann sum
    of the quantile function, and its error falls as 1/n^2 rather than 1/sqrt(n)."""
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
    """The draw's SAMPLE moments against the truncated normal's ANALYTIC ones.

    This is the statement the whole construction rests on and the one no product gate can make:
    a continuing branch weighted `Phi(zB)` is only right if what it carries forward is distributed
    as the law conditioned on survival. A draw that is merely on the right SIDE of the barrier -
    a rejection sample, a clamp, a reflected normal - passes every sign test and fails here.
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
    """SIBLING GATE. `sim_spot_oss` (both arms of `pv_discrete_barrier_option`) writes the idiom
    inline and is enumerated rather than absorbed; this holds the copy to the seam while it waits.

    The UP side is bit-for-bit. The DOWN side is not, and the difference is exactly the reason the
    enumeration gives: its base is `(1 - p) + u*p` where the seam's is `Phi + u*p`, and
    `1 - (1 - Phi)` recovers `Phi` exactly only at or above a half (Sterbenz), so below a half the
    two can differ in the last bit of the base. This gate asserts BOTH halves of that - the
    agreement above a half and the existence of the disagreement below it - so absorbing the site
    turns this red rather than passing silently: it is a results-changing event that has to
    re-baseline the barrier's pinned fixtures, and that is what a red gate is for.
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
    """SIBLING GATE. `pv_MC_AutoCallSwap` writes `clamp(p * u)`; the seam writes `clamp(u * p)`.

    An IEEE product is commutative, so this one is bit-identical and the enumeration's obstacle is
    SHAPE rather than arithmetic: `p` is formed in the coupon branch and consumed a screen later
    under `fixing_aligned`, on an iteration where the draw is deliberately skipped. The gate says
    the arithmetic is ready whenever the loop is.
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
    """`E[S^a 1{fired}]` against a trapezoid over the normal density - the closed form's oracle.

    `power=1` is what every full-gain payment reads, `power=-1` is the same for an INVERTED accrual
    (the TARF's `InvertedTarget` writes its payoff in reciprocals), and `power=0` is the fired
    probability. All three are one expression, which is why one gate covers them.
    """
    got = float(pricing.lognormal_partial_moment(
        _t(S0), _t(DRIFT), _t(VOL), _t(Z_TRIGGER), fired_above, power))
    lo, hi = (Z_TRIGGER, 12.0) if fired_above else (-12.0, Z_TRIGGER)
    ref = _gaussian_tail(lambda z: _spot(S0, DRIFT, VOL, z) ** power, lo, hi)
    assert abs(got - ref) < 1e-9 * max(abs(ref), 1e-3), (power, fired_above, got, ref)


def test_the_zeroth_moment_is_the_fired_probability():
    """`power=0` IS `1 - p`, and the ledger still spells it as the complement - which is not the
    same statement. These two agree to roundoff and not to the bit, and the telescoping identity
    below is exact only because the ledger takes the complement rather than this."""
    z = _t(Z_TRIGGER)
    fired = float(pricing.lognormal_partial_moment(_t(S0), _t(DRIFT), _t(VOL), z, True, 0.0))
    survive = float(pricing.oss_truncated_draw(_stratified(8), z, True)[0])
    assert abs(fired - (1.0 - survive)) < 1e-15
    assert abs(fired - (1.0 - _Phi(Z_TRIGGER))) < 1e-15


@pytest.mark.parametrize('power', [1.0, -1.0])
def test_the_two_tails_sum_to_the_unconditional_moment(power):
    """Fired plus survived is the whole distribution - the moment-space form of conservation, and
    the one identity a sign error in the reflection cannot survive."""
    args = (_t(S0), _t(DRIFT), _t(VOL), _t(Z_TRIGGER))
    both = float(pricing.lognormal_partial_moment(*args, True, power) +
                 pricing.lognormal_partial_moment(*args, False, power))
    whole = S0 ** power * math.exp(power * DRIFT + 0.5 * power * power * VOL * VOL)
    assert abs(both - whole) < 1e-14 * whole, (power, both, whole)


def test_the_full_gain_is_black_when_the_trigger_never_binds():
    """With the trigger pushed to minus infinity the fired set is everything, so the full-gain
    expectation collapses to the interval's own forward less the strike - a limit the closed form
    has to hit exactly, and the one place a missing `exp(0.5 s^2)` would show."""
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
    """THE BIAS TRAP, as measured numbers rather than a warning.

    The forbidden shortcut is `p_fired x J(S_sampled)`, where `S_sampled` is the draw the
    simulation actually carries forward - and that draw is SURVIVAL-truncated, so it sits on the
    far side of the trigger from the event being paid for. Both halves here are exact (a
    probability and a conditional expectation, no Monte Carlo anywhere), so the gap is BIAS: it
    does not shrink with paths, and no seed spread will ever cover it.

    Measured on this fixture - spot 1.2500, interval drift 0.0170, interval vol 0.1400, up trigger
    at 1.3100 (`z = 0.213454`, `p_fired = 0.415486`), strike 1.2000:

        FULL GAIN     truth 0.105802   shortcut 0.009525   11.11x LOW
        CAPPED/EXACT  truth 0.024929   shortcut 0.007585    3.29x LOW

    and the shortcut written with the RAW gain rather than its positive part reads **-0.015531**,
    which is not a magnitude error at all but a SIGN one: truncating the upper 41.5% of the
    interval drags the surviving mean to 1.16257, below the strike, so the branch that is supposed
    to pay for knocking out pays a negative number. "Conservative" is not a defence, and neither is
    "small": these are the two conventions the TARF actually ships, and both are wrong by a factor.
    """
    args = (_t(S0), _t(DRIFT), _t(VOL), _t(Z_TRIGGER))
    p_fired = float(pricing.lognormal_partial_moment(*args, True, 0.0))
    survived = 1.0 - p_fired

    def on_the_surviving_draw(payoff):
        """What the shortcut pays: the FIRED probability times the payoff evaluated on the draw
        that survived - the expectation of `p x realised`, exactly."""
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

    Exact to roundoff, and exact only because the fired mass is `1 - p` off the SAME `p` the draw
    was truncated with. The residual accumulates one rounding per fixing, so the bar is scaled by
    the strip length rather than fixed.
    """
    torch.manual_seed(20260830)
    probs = [torch.rand(8, 64, dtype=DT) for _ in range(12)]
    ledger = _ledger_walk(probs, torch.ones(8, 64, dtype=DT))
    residual = float(ledger.conservation().abs().max())
    assert residual < len(probs) * torch.finfo(DT).eps * 4, residual


def test_the_ledger_telescopes_off_a_partly_resolved_start():
    """A block opening on a deal whose observed prefix already killed some paths starts at
    `prev_alive`, not at one - so the identity is stated against the STARTING weight. A ledger that
    hard-codes 1.0 passes the gate above and silently loses the prefix here."""
    torch.manual_seed(20260830)
    alive0 = (torch.rand(8, 64, dtype=DT) > 0.3).to(DT)   # an observed prefix: exact 0/1
    probs = [torch.rand(8, 64, dtype=DT) for _ in range(6)]
    ledger = _ledger_walk(probs, alive0)
    assert float(ledger.conservation().abs().max()) < len(probs) * torch.finfo(DT).eps * 4


def test_the_ledger_telescopes_through_observed_fixings():
    """An OBSERVED fixing fires with an exact 0/1 indicator rather than a `Phi`, and the identity
    has to survive that too - it is the same arithmetic, which is the point of stating it once."""
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
    """A block whose rows hold no simulated fixing at all - `pv_MC_Accumulator` builds those - has
    fired nothing, and the identity must not need a zero of the right shape to say so."""
    ledger = pricing.SurvivalLedger(torch.ones(3, dtype=DT))
    assert ledger.fired is None
    assert float(ledger.conservation().abs().max()) == 0.0


def test_the_ledger_reports_its_residual_at_debug():
    """The identity is a DEBUG line and not an assert in a hot path, the way every other estimator
    in `pricing` reports what it did."""
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
    """The kernel as `exposure_kink_term` spelled it BEFORE `kink_kernel` was factored out.

    A copy, deliberately: the factoring's whole claim is that it moved no bits, and the only thing
    that can check that claim is the arithmetic it replaced, written out where a reader can diff
    it. If this ever has to be edited to keep the gate green, the factoring changed a number.
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
    """ADMISSION. `kink_kernel` is `exposure_kink_term`'s own machinery lifted out, not a rewrite -
    so it has to return the identical tensor on every shape that reached the original."""
    torch.manual_seed(11)
    for shape in [(4, 4096), (1, 1024), (7, 2048)]:
        V = torch.randn(*shape, dtype=DT) * torch.linspace(0.2, 3.0, shape[0], dtype=DT).reshape(-1, 1)
        kernel, _, _ = pricing.kink_kernel(V, 1, 'gate')
        assert torch.equal(kernel, _inline_kink_reference(V, 1)), shape


def test_the_exposure_term_is_the_shared_kernel_and_nothing_else():
    """The term IS `0.5 * K * u^2` off the shared kernel - stated bitwise, so a future edit to
    either side cannot drift them apart while both stay individually plausible."""
    torch.manual_seed(12)
    V = (torch.randn(5, 4096, dtype=DT) * 1.3).requires_grad_(True)
    kernel, _, _ = pricing.kink_kernel(V.detach(), 1, 'gate')
    u = V - V.detach()
    assert torch.equal(pricing.exposure_kink_term(V), 0.5 * kernel * u * u)


def test_the_accrual_kink_is_zero_at_value_and_bit_identically_zero_at_first_order():
    """The admission test the roadmap sets, ONE ORDER STRICTER than the boundary correction's:
    `np.array_equal` at value AND at first order, term on versus off. `u` is an exact IEEE zero,
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
    """And it is not zero at SECOND order - the whole point. The Hessian the term adds is the
    kernel estimate of `f_g(0) * E[g_theta g_theta^T | g = 0]`, which the crisp relu reports as an
    exact zero however many paths it is given."""
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
    """A fixing whose conditional law carries a POINT MASS at the strike is refused, on the same
    ladder that refuses a reporting row: the density does not settle as the bandwidth narrows, it
    climbs as `1/h`. The crisp path answers zero there, which is a smaller number and not a
    smaller error - the refusal says so rather than picking one."""
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
    """The refusal PROSE stayed with its caller, which is why the factoring could be shared at all:
    the remedies at a reporting row are not the remedies at a fixing."""
    torch.manual_seed(15)
    row = torch.ones(1, 65536, dtype=DT)
    row[0, :64] = 0.0
    with pytest.raises(utils.SecondOrderRefused) as refusal:
        pricing.exposure_kink_term(row)
    message = str(refusal.value)
    assert 'exposure_kink_term' in message and 'ATOM' in message, message
    assert 'reporting row' in message, message


def test_a_kink_with_no_atom_is_admitted():
    """The anti-placebo for the two refusals above: an ordinary density must NOT refuse, or the
    ladder is rejecting everything and the gates that pass are passing for nothing."""
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
    it is the LOAD-BEARING half: the exposure, cashflow and collateral semantics of the crisp path
    are structurally untouched there, because there is no key an author could write to reach them.
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
    are the same run - which is what makes every existing gate a regression bar for this build."""
    assert _declared(calculation.Base_Revaluation)['Branch_And_Weight'].default == 'No'
    assert schema.declared_defaults(calculation.Base_Revaluation, {})['Branch_And_Weight'] == 'No'


def test_the_state_declares_it_so_a_pricer_reads_it_without_a_fallback():
    """`Calculation_State` carries the flag False, the way it carries `recompute_inner_mc` and
    `gamma`: a pricer reads `shared.branch_and_weight` directly rather than through a `getattr`
    default that a second calculation could quietly disagree with."""
    state = utils.Calculation_State(
        {}, torch.ones([1, 1], dtype=DT), 1, None, 'Constant', 1, False)
    assert state.branch_and_weight is False


def test_the_switch_off_is_the_crisp_path(tmp_path):
    """OFF IS OFF, on a real document through the JSON contract: the key absent and the key written
    'No' are the same number to the last bit. This is the statement every other gate in the suite
    depends on, since the switch defaults off and none of them writes it."""
    absent = _run_tarf(_tarf_job(), tmp_path, 'absent')
    written = _run_tarf(_tarf_job(Branch_And_Weight='No'), tmp_path, 'written')
    assert absent == written, (absent, written)


# ======================================================================================
# GBM ONLY (`pricing.branch_and_weight`)
# ======================================================================================

def _deal_data(spot_model=None):
    """A real `FXTARFOptionDeal` off the fixture's own deal block, in the engine's own
    `DealDataType`, with the `HN_Params` entry shaped as `instruments.get_hn_factor` shapes it -
    `(is_stochastic, [parameter factors], SpotModel, curve tenors)`."""
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
    """The engine's own state, constructed the way a calculation constructs it - the switch is set
    on it afterwards exactly as `Base_Revaluation.__init_shared_mem` sets it."""

    def __init__(self, on):
        super(_State, self).__init__(
            {}, torch.ones([1, 1], dtype=DT), 1, None, 'Constant', 1, False)
        self.branch_and_weight = on


@pytest.mark.parametrize('spot_model', [None, 'HestonNandi', 'HestonNandiComponent'])
def test_the_switch_off_admits_every_model(spot_model):
    """Off, the seam answers False and asks no questions - the crisp path prices Heston-Nandi
    exactly as it does today, and the refusal below cannot reach a run that did not ask for the
    smooth estimator."""
    assert pricing.branch_and_weight(_State(False), _deal_data(spot_model)) is False


def test_a_gbm_deal_under_the_switch_is_admitted():
    assert pricing.branch_and_weight(_State(True), _deal_data(None)) is True


@pytest.mark.parametrize('spot_model', ['HestonNandi', 'HestonNandiComponent'])
def test_the_switch_refuses_under_heston_nandi(spot_model):
    """GBM ONLY, and the refusal is the ruling: the conditioning step must be the FIXING interval's
    own lognormal law. Under HN the walk is daily, so the only Gaussian conditional in hand is the
    last daily sub-step, and a Gaussian `p` applied there would be a wrong number wearing the right
    estimator's name. Both flavours refuse, because both are the same walk.
    """
    with pytest.raises(ValueError) as refusal:
        pricing.branch_and_weight(_State(True), _deal_data(spot_model))
    message = str(refusal.value)
    assert 'Branch_And_Weight' in message, message
    assert spot_model in message, 'the refusal names the MODEL it refused: ' + message
    assert 'stride' in message.lower(), 'the refusal cites where its Phi comes from: ' + message
    assert 'hn_cdf_logret' in message, message
    # a refusal names a remedy, and both of this one's are things a caller can do today
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
# ONE WORLD, shared by every product gate below, and the flatness in it is deliberate. The
# fixture's own USD curve is steep and its surface flat; here BOTH curves are flat and DIFFERENT
# (4% against 2%, so the carry is live at 2% and no gate can pass on a degenerate zero drift) and
# the surface stays flat at 10%. That buys the reference an exact interval strip - a flat curve's
# interval carry IS its zero rate and a flat surface's interval vol IS its quote - so the quadrature
# below can be written from the market data rather than from `forward_carry_rate`, which is the
# thing on test. The sloped-curve strip has its own gates (roadmap.md, the OSS carry-strip row).
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
        # a fixing dated ON the base date resolves off the SIMULATED spot, so the deal registers a
        # latch with a graph-carrying gap - which is the only way this pricer reaches the refusal
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

    VANNA IS A SUM over the surface's own knots, and that is not a shortcut: the reference
    differentiates a single scalar vol, which on a FLAT surface is exactly a parallel bump of every
    knot, so the matching object in the report is the whole vol row. The same sum against
    `Greeks_First` reproduces the reference's vega, which is what says the identification is right
    rather than convenient.
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
    """The normal DENSITY, and the only distributional fact these references know. There is no
    `Phi` here on purpose: every probability below is that density integrated over a segment, so a
    sign or a reflection error in `norm_cdf` cannot be reproduced by its own oracle."""
    return torch.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


Z_INF = 8.5      # the density there is 5e-15; the payoffs are O(1e3), so the tail is 1e-11


def _seg(a, b, f, n):
    """``int_a^b phi(z) f(z) dz`` by trapezoid on a grid whose ENDS MOVE with theta.

    That is the whole design. A quadrature on a FIXED grid with the deal's decisions written as
    indicators has a value but no derivative - the indicator does not move, so `d/dtheta` misses
    exactly the flux the estimator under test exists to carry, and the reference would confirm the
    defect rather than catch it. Mapping each region onto `u` in [0, 1] and integrating there puts
    the region's own limits on the tape, so `torch.autograd` differentiates the way Leibniz's rule
    does and the reference owns a delta, a gamma and a vanna that are really its value's.

    `a` and `b` may be scalars or vectors (one per outer node); the sample axis is the last, so a
    nested call broadcasts without a reshape. A segment whose limits CROSS integrates negatively,
    which is the right answer for a signed integral and matters only where the density has already
    underflowed - a knock-in bound pushed past `Z_INF` by an extreme outer node.
    """
    u = torch.linspace(0.0, 1.0, n, dtype=DT)
    w = torch.full((n,), 1.0 / (n - 1), dtype=DT)
    w[0] = w[-1] = 0.5 / (n - 1)
    a = a if torch.is_tensor(a) else torch.tensor(a, dtype=DT)
    b = b if torch.is_tensor(b) else torch.tensor(b, dtype=DT)
    z = a[..., None] + (b - a)[..., None] * u
    return (b - a) * ((w * _t_phi(z)) * f(z)).sum(-1)


def _intervals():
    """The two fixing intervals own `(dt1, dt2)` and the two discount factors, from the MARKET DATA
    - a flat curve's interval carry IS its zero rate and a flat surface's interval vol IS its quote,
    so this needs neither `forward_carry_rate` nor `forward_vol_rate`, which are what is on test."""
    dt = (FIX_DAYS[0] / 365.0, (FIX_DAYS[1] - FIX_DAYS[0]) / 365.0)
    disc = tuple(math.exp(-R_USD * d / 365.0) for d in SETTLE_DAYS)
    return dt, disc


def _tarf_reference(spot, sigma, target=TARGET, adjustment='Exact', n_out=600, n_in=400):
    """The two-fixing TARF as a nested region integral, differentiable end to end.

    Written from the DEAL and nothing else. Per fixing the real line splits at three levels the
    deal names - the knock-in `Bar`, the strike `K`, and the moving trigger `K + r` where `r` is
    whatever target is left - and on each piece the payoff is one analytic expression:

        z < zBar         knocked in and OTM        -N2 * (K - S)
        zBar < z < zK    OTM, not knocked in        0
        zK < z < zB      ITM, target not yet full  +N1 * (S - K)
        z > zB           the target FILLS           the declared convention's payment

    The second fixing's three levels ride the FIRST fixing's outcome, because `r` does - that
    coupling is what makes a TARF a TARF, and it is the reason a one-dimensional oracle cannot see
    it. The trapezoid is nested rather than a product grid for the same reason.
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
        top = torch.full_like(zB2, Z_INF)
        if adjustment == 'Full':
            value = value + _seg(zB2, top, lambda z: N_ITM * (s2(z) - K), n_in)
        elif adjustment == 'Exact':
            paid = (N_ITM * r1)[..., None]
            value = value + _seg(zB2, top, lambda z: paid * torch.ones_like(z), n_in)
        return D2 * value

    top1 = torch.as_tensor(Z_INF, dtype=DT)
    if adjustment == 'Full':
        total = D1 * N_ITM * _seg(zB1, top1, lambda z: spot * torch.exp(m1 + v1 * z) - K, n_out)
    elif adjustment == 'Exact':
        total = D1 * N_ITM * target * _seg(zB1, top1, torch.ones_like, n_out)
    else:
        total = torch.zeros((), dtype=DT)

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
    """The two-fixing accumulator, the same way. Two levels per fixing rather than three - the
    strike splits the legs, the barrier ends the deal - and no coupling, because an accumulator
    accrues nothing that moves its own trigger."""
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


def _reference_table(build, **kwargs):
    """value / delta / vega / gamma / vanna off one reference, by double backward."""
    spot = torch.tensor(SPOT_FX, dtype=DT, requires_grad=True)
    sigma = torch.tensor(SIGMA_FLAT, dtype=DT, requires_grad=True)
    value = build(spot, sigma, **kwargs)
    first = torch.autograd.grad(value, (spot, sigma), create_graph=True)
    second = torch.autograd.grad(first[0], (spot, sigma), retain_graph=True)
    return {'value': float(value), 'delta': float(first[0]), 'vega': float(first[1]),
            'gamma': float(second[0]), 'vanna': float(second[1])}


_REFERENCES = {}


def _table(build, **kwargs):
    """The reference tables, memoized. Each one is a nested double-backward through ~1e6 nodes;
    computing them at import would tax every run of this file that never asks for one."""
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
    frame, and every first-order greek, by `np.array_equal` rather than a tolerance.

    This is the statement the entire existing suite depends on. The switch defaults off and no
    other gate in the repo writes it, so every number in `test_fx_tarf_json`,
    `test_fx_accumulator_json`, `test_tarf_cash_settle` and the two boundary-event files is a
    regression bar for this build only while this holds.
    """
    absent, out_a, _ = _run_doc(build(greeks='First'), tmp_path, name + '_absent')
    written, out_w, _ = _run_doc(_smooth(build(greeks='First'), on=False), tmp_path, name + '_no')
    assert absent == written, (absent, written)
    # `DataFrame.equals` rather than `array_equal`: the reported frame carries the deal's tags and
    # an all-NaN root row, and NaN is not equal to itself - the numpy spelling would fail on two
    # frames that ARE the same report
    assert out_a['Results']['mtm'].equals(out_w['Results']['mtm']), 'the reported frame moved'
    frame_a, frame_w = out_a['Results']['Greeks_First'], out_w['Results']['Greeks_First']
    assert np.array_equal(frame_a.values, frame_w.values), 'a first-order greek moved'


def test_a_tarf_with_no_knock_in_is_bit_identical_under_the_switch(tmp_path):
    """ATTRIBUTION, and the sharpest thing this file can say about what the switch actually does.

    The one-step-survival loop was ALREADY the smooth estimator for the target: the KO-in-step term
    is the fired branch integrated against the interval's own law and the continuation is the
    truncated draw. So with the knock-in switched off there is nothing left for the switch to
    change, and the two estimators agree TO THE BIT - not to a tolerance. Turn the leveraged
    knock-in on and they part company by ~1.3% at 65536 paths (the gate below), because that leg
    is the one thing this pricer was still sampling an indicator for.
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
    """THE FORBIDDEN ESTIMATOR, written here so the gate can kill it: the same one-step-survival
    construction with the knock-in left as a SAMPLED INDICATOR and no kink term.

    Deterministic on purpose - both fixings walk a midpoint grid rather than a random one, and the
    grid is a product so there is no seed anywhere in it. Any gap against the quadrature is
    therefore ESTIMATOR BIAS and not noise, which is what makes the readings in the gate below
    quotable numbers rather than one draw's luck. Its VALUE is right; every derivative is not.
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
    """THE TABLE. A two-fixing knock-in TARF, priced under the switch at `Greeks: 'All'`, against
    the differentiable quadrature reference - value, delta, vega, gamma, vanna.

    Measured on this document (spot 1.1, strike 1.1, target 0.05, knock-in 1.05, notionals
    1000/2000, r 4% q 2%, vol 10%, fixings at 91 and 182 days, 262144 inner paths):

                        engine        quadrature      relative
        value          -37.4018        -37.3594         0.113%
        delta        +1814.772       +1814.660          0.006%
        vega         -1131.33        -1131.166          0.015%
        gamma       -28097.69       -28087.200          0.037%
        vanna        +5016.06        +5010.805          0.105%

    THE VEGA ROW IS NOT DECORATION. It is what says the vol identification is right: the reference
    differentiates ONE scalar vol, and the report spreads that across the surface's four live
    knots, so summing them has to reproduce the reference's vega before summing them for vanna
    means anything. Get the identification wrong and vanna would agree by luck or disagree for the
    wrong reason.
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
    """THE KILL, and the reason the switch is a switch rather than a nicety.

    The same construction with the knock-in sampled rather than integrated, on a deterministic
    4096 x 4096 product grid, against the quadrature. Both sides are quadratures of one integral,
    so nothing here is noise - and the ratios below do not move between 2048 and 4096 nodes:

        value    1.0001 x the truth   - the estimator is UNBIASED, which is the trap
        delta    0.7143 x             - 28.6% short
        vega     0.4826 x             - 52% short
        gamma    0.3006 x             - 70% short
        vanna   -0.3413 x             - THE WRONG SIGN

    A value that is right beside derivatives that are wrong is exactly what a sampled indicator
    produces, and it is the failure mode nothing downstream can detect: the mark reconciles, the
    hedge does not. The vanna sign flip is the reading to quote, because a magnitude can be argued
    about and a sign cannot.
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
    """Same expectation, less variance. Ten seeds each at 16384 inner paths, against the
    quadrature's -37.359382:

        crisp    mean -37.5296   seed sd 0.3208
        smooth   mean -37.4098   seed sd 0.0910

    VARIANCE RATIO 12.42 (standard deviations 3.52x apart), and both means sit inside their own
    spreads of the reference - the crisp one 0.456% out, the smooth one 0.135%. That is what
    integrating a leg out of the sample buys: the estimator changed, the number did not.
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
    """THE KNOWN-DEFECTS ROW, closed on its own terms.

    That row records a pin that fires on 27-61% of paths and reads 27% short uncorrected, with
    neither the estimator (13% bandwidth spread) nor the ORACLE (8.9% flatness) resolving better
    than ~10% - so it was gated structurally with no tolerance asserted, and the row named
    branch-and-weight as its designed exact resolution. Here the pinned payment is a conditional
    expectation the loop evaluates in closed form, and the oracle is a deterministic quadrature
    that resolves to 1e-5, so a tolerance CAN be asserted. It is set an order tighter than that
    ~10% floor.

    Measured, three seeds each at 65536 paths, on a pin-heavy TARF (the target nearly filled at the
    first of two fixings, so the trigger fires on ~46% of paths):

        target 0.010   quadrature -54.6691   smooth -54.7124 (0.079%)   crisp -54.8276 (0.290%)
        target 0.020   quadrature -50.0300   smooth -50.0769 (0.094%)   crisp -50.2121 (0.364%)

    The crisp readings are recorded beside them for the history, not gated: they are a different
    estimator of the same number and they are allowed to be noisier.
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
# THE DEAL'S OWN SETTLEMENT CONVENTION
# ======================================================================================

@pytest.mark.parametrize('declared,convention', [
    ('', 'Exact'), ('Exact', 'Exact'), ('Exact Gain', 'Exact'),
    ('Full', 'Full'), ('Full Gain', 'Full'), ('None', 'None'), ('No Gain', 'None')])
def test_the_declared_target_adjustment_reaches_the_fired_branch(declared, convention, tmp_path):
    """`TargetAdjustment` is a declared field nothing has ever read, and each of its three
    conventions is one closed form on the fired branch. Against the quadrature, 262144 paths:

        Exact  engine -37.40   quadrature -37.3594     the remaining target
        Full   engine -21.09   quadrature -21.0697     the whole gain, overshoot included
        None   engine -52.96   quadrature -52.9211     nothing at all

    Ordered as the economics demand - paying the client more on the filling fixing is worth more to
    a buyer - and each spelling a desk writes lands on its own number rather than the nearest one.
    """
    job = _smooth(_tarf_doc(sims=1 << 18))
    job['Calc']['Deals']['Deals']['Children'][0]['Instrument']['.Deal'][
        'TargetAdjustment'] = declared
    value, _, _ = _run_doc(job, tmp_path, 'adj')
    truth = _table(_tarf_reference, adjustment=convention)['value']
    assert _rel(value, truth) < 0.01, (declared, convention, value, truth)


def test_the_crisp_path_ignores_the_convention_and_the_switch_does_not(tmp_path):
    """THE DECLARED ASYMMETRY, gated so it is a recorded decision rather than a surprise.

    The one-step-survival estimator pays the remaining target whatever the field says. Teaching it
    to read the field would move numbers with the switch OFF, which its own contract forbids, so
    the repair belongs to a separate change with its own re-baselined fixtures - and until then a
    deal declaring 'Full' prices as 'Exact' crisp and as itself smooth. Both halves are asserted,
    because only the pair says which way round it is.

    AND THE ASYMMETRY IS AUDIBLE, which is the half a docstring cannot do. `Branch_And_Weight` is
    documented as "same expectation, lower variance"; on this one deal it is neither - -37.54 crisp
    against -21.09 smooth, a 44% reprice on a flag a desk reads as variance reduction. So
    `tarf_target_adjustment` warns by name on every deal it resolves to something other than
    'Exact', and this gate holds that warning in place from both sides: delete the
    `logging.warning` and `assert warned` fails; widen it to fire on every TARF and the 'Exact'
    silence check fails. A warning nothing asserts is a warning the next cleanup deletes.
    """
    def priced(adjustment, on, debug=False):
        job = _tarf_doc()
        job['Calc']['Deals']['Deals']['Children'][0]['Instrument']['.Deal'][
            'TargetAdjustment'] = adjustment
        value, _, log = _run_doc(_smooth(job, on=on), tmp_path, 'asym', debug=debug)
        return value, log

    assert priced('Full', on=False)[0] == priced('Exact', on=False)[0], (
        'the crisp path has started reading TargetAdjustment - that is a numbers-moving change '
        'with the switch OFF, and every pinned TARF fixture in the repo has to be re-baselined')

    full, full_log = priced('Full', on=True, debug=True)
    exact, exact_log = priced('Exact', on=True, debug=True)
    assert full != exact

    warned = [ln for ln in full_log.splitlines() if 'CHANGES THE DEAL' in ln]
    assert warned, (
        'a deal whose price moves 44% with this switch said nothing about it - the whole asymmetry '
        'lives in a docstring again, and the desk that flips the flag on a live book has no way to '
        'attribute the move')
    assert "'Full'" in warned[0] and 'TargetAdjustment' in warned[0], warned
    assert 'Branch_And_Weight' in warned[0], 'the warning must name the flag that caused the move'
    assert not [ln for ln in exact_log.splitlines() if 'CHANGES THE DEAL' in ln], (
        "the warning fired on 'Exact', where the switch really is same-expectation - a warning on "
        'every TARF is a warning on none of them')


def test_an_unknown_target_adjustment_refuses_by_name(tmp_path):
    """A convention this pricer does not know is refused rather than defaulted - the defect
    `pv_one_touch_option`'s `Payment_Timing` chain shipped with, where a third value priced as
    whatever the last assignment left. The refusal names the field, the value, the accepted set and
    the two things a caller can do today."""
    job = _smooth(_tarf_doc())
    job['Calc']['Deals']['Deals']['Children'][0]['Instrument']['.Deal'][
        'TargetAdjustment'] = 'Knockout'
    value, _, log = _run_doc(job, tmp_path, 'badadj', debug=True)
    log = log.replace('\\', '')          # the loader logs the exception's args, quotes escaped
    assert 'TargetAdjustment' in log and 'Knockout' in log, log[-1500:]
    assert "'Exact'" in log and "'Full'" in log, 'the refusal must name its accepted set'
    assert "Branch_And_Weight: 'No'" in log, 'a refusal names a remedy'
    assert math.isnan(value), 'the deal priced anyway, so nothing was actually refused'


# ======================================================================================
# GAMMA FLOWS
# ======================================================================================

def test_gamma_flows_under_the_switch_and_is_refused_without_it(tmp_path):
    """THE POINT OF THE BUILD, in one gate. `Greeks: 'All'` on a knock-in TARF is REFUSED without
    the switch and RUNS with it, and what comes back is the derivative of the delta the same run
    reports.

    Measured, 65536 paths: the reported spot-spot entry -28120.6 against a common-random-numbers
    ladder of the reported delta whose flattest rung reads -28142.6 - 0.08% disagreement at 0.87%
    flatness over four decades of bump.

    A LADDER OF THE CRISP CORRECTED DELTA DOES NOT CONVERGE, and that is worth recording rather
    than hiding: the same four rungs against the boundary-corrected first-order number read
    -12339 / -24870 / -26589 / -27145, a 57.5% spread climbing monotonically with h. The corrected
    delta is itself a kernel estimate whose bandwidth is set from the sample's own spread, so
    bumping the spot moves the estimator as well as the deal - a ladder going FLAT is this suite's
    definition of a derivative that exists, and that one does not. The smooth path's does.
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
    """CONSERVATION, on a live run rather than on a constructed tensor: per path, the mass every
    fixing fired plus the mass still alive is the mass the strip opened with, exactly to float
    roundoff. Measured on this document at 1.11e-16 with 59.0% of the weight surviving both
    fixings - which is the other half of the reading, since a ledger that fired everything or
    nothing would conserve trivially."""
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
    """COMPOSITION. `Recompute_Inner_MC` re-runs the pricer's inner simulation in `backward()`
    instead of taping it, and the smooth path is the same one function called twice - so the value
    and every first-order greek come back BIT-IDENTICAL to the taped run. Second order still
    refuses through the node's own refusal, which the switch does not lift and is not meant to: the
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
    """NOT A REFUSAL - a declaration with a number behind it.

    Second-order variance in the conditioning step scales like `s_k**-3`, so a daily-fixing deal's
    gamma is noisy where a monthly one's is not. Measured, twelve fixings either way, eight seeds,
    8192 inner paths:

        monthly (30d)   value seed spread 1.126%   gamma seed spread  1.696%
        daily   (1d)    value seed spread 0.350%   gamma seed spread 21.154%

    The VALUE spread FALLS at daily spacing and the gamma spread grows by 12.5x - which is the
    signature exactly: the estimator is fine, its second derivative is not. The desk's answer is
    that a daily accumulator's economically right gamma is its own call-spread width anyway, a
    fixed smoothing converging at the ordinary rate, and that is what gets hedged. This gate holds
    the SHAPE of the finding - monthly quiet, daily an order louder - so nobody later reads a
    daily gamma off this path believing the monthly measurement covers it.
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
    """THE VERIFICATION THE CONTRACT PREDICTED, and it comes out as an equality rather than a
    tolerance.

    This pricer's loop was already the smooth estimator - analytic survival, truncated
    continuation, a fired branch worth exactly zero - so the switch cannot move its VALUE and does
    not: bit-identical off and on, and first order with it. What the switch changes is the
    curvature, because both legs are `relu`s of one argument and pathwise AAD answers their gamma
    with an exact zero. On one document, 65536 paths, the SAME `Greeks: 'All'` run either way:

                    crisp          smooth        quadrature
        gamma     -24156.07      -27154.08      -27172.47      11.1% short -> 0.07%
        vanna      -1791.03       -220.69        -211.25       8.5x        -> 4.5%

    The switch is its own mutation here - same document, same seed, one estimator swapped - so
    nothing had to be patched to produce the kill.
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
    """The accumulator only registers when a fixing has been OBSERVED, so this document dates one
    ON the base date - resolved off the simulated spot, which is what gives the latch a
    graph-carrying gap and makes `Greeks: 'All'` refuse.

    Under the switch the registration is skipped and the Hessian flows. What says nothing was LOST
    is the first-order comparison beside it: the two runs agree BIT FOR BIT. Base valuation has one
    scenario, a one-sample gap supports no local-linear fit, and the boundary correction on it is
    exactly zero by construction - so the estimator this skips was contributing exactly nothing
    here, which is the claim the pricer's docstring makes and this is where it is checked rather
    than believed.
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

# The binary sibling, authored here rather than borrowed: `test_barrier_bridge`'s own digital
# fixture is built around an UNREACHABLE barrier, to isolate the terminal step, and this section
# walks the live one too. Everything else is that file's deal, its world and its zero rates.
BARRIER_BINARY = {
    'Object': 'EquityBarrierBinaryOption', 'Reference': 'BARR1', 'Currency': 'USD',
    'Payoff_Currency': 'USD', 'Equity': 'EQ', 'Dividends': 'EQ', 'Discount_Rate': 'USD',
    'Equity_Volatility': 'EQ', 'Buy_Sell': 'Buy', 'Option_Type': 'Call', 'Strike_Price': 100.0,
    'Expiry_Date': bb.BASE + pd.Timedelta(days=365), 'Cash_Payoff': 100.0,
    'Settlement_Date': bb.BASE + pd.Timedelta(days=365), 'Barrier_Type': 'Down_And_Out',
    'Barrier_Price': 90.0, 'Barrier_Dates': BARRIER_DATES}


def _barrier_config(barrier_price=90.0, digital=False, spot=None, **over):
    """A monthly-monitored barrier in `test_barrier_bridge`'s own zero-rate world.

    ZERO RATES ARE LOAD-BEARING HERE, not incidental: with every discount factor exactly one, the
    in-out parity gates below read the survival ledger's telescoping identity directly off two
    prices rather than off a discounted weighting of it."""
    cfg = bb._cfg()
    base = BARRIER_BINARY if digital else dict(bb.BARRIER_DEAL, Barrier_Dates=BARRIER_DATES)
    deal = dict(base, Barrier_Price=barrier_price, **over)
    cfg.deals['Deals']['Children'] = [{'Instrument': construct_instrument(deal, {})}]
    if spot is not None:
        cfg.params['Price Factors']['EquityPrice.EQ']['Spot'] = spot
    return cfg


def _barrier_run(barrier_price=90.0, greeks='No', on=None, sims=1 << 15, **cfg_kwargs):
    """(value, results) off one base valuation. `on=None` leaves the switch ABSENT, which is a
    third state the byte-identity gate needs and not a spelling of 'No'."""
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


# The deal family, walked so the bit-identity claim is a claim about the PRICER rather than about
# one document: both directions, both barrier sides, both option types, both sides of the trade,
# the cash rebate on each leg, and the binary sibling whose terminal step is integrated instead of
# sampled. Every row must price the same to the last bit with the switch absent, 'No' and 'Yes'.
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
    """OFF IS OFF, and ON IS OFF TOO - which for this one pricer is the same statement.

    The third product is the smallest diff because its OSS sampler was the smooth estimator
    already: every monitored step is an analytic `p` with a survival-truncated continuation, the
    knock-out's rebate is `(1 - p) * L * rebate * D_j` at the fixing it falls due, and a digital's
    terminal step is integrated rather than sampled. So there is no indicator on the simulated tape
    to remove, and the value under the switch is not merely close - it is BIT-IDENTICAL.

    That is only worth asserting across the FAMILY. One down-and-out call passing says nothing
    about the parity leg, about which side of the barrier the truncation reflects on, about the
    rebate dividing by an unsigned size, or about the branch a digital takes instead of sampling
    its payoff. Each of those is a place where a switch could plausibly have leaked into the crisp
    path, so each is a row here. Three states, not two: ABSENT and 'No' are compared as well,
    because a field read through a default is a different code path from a field read.
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
    """WHY skipping the `LatchedBoundarySet` is free here, asserted rather than argued.

    The barrier's registration records an OBSERVED scenario crossing - the deal's own date against
    the scenario spot - which is data rather than simulated state, so branch-and-weight has no
    conditioning step to integrate it against and simply does not claim it. The pricer's docstring
    used to justify dropping it by saying the decision carries no graph. That is NOT the reason,
    and the sharper one is the reason: `b_gaps` is built from the scenario spot and does carry a
    graph. What makes the correction exactly zero is that BASE VALUATION HAS ONE SCENARIO, so
    `boundary_weights` finds no spread to set a kernel width from and returns its empty-kernel
    branch - weights zero, correction an exact zero, which is what one scenario means.

    So the observable is byte identity of the WHOLE first-order frame, not just of the value: the
    correction the switch declines to register was contributing nothing to begin with. Measured on
    the knock-out at 0.6597964784 either way, the knock-in at -0.1100582535, the binary at
    2.259431076. The day base valuation grows a second scenario this gate is the one that goes red,
    and it should - the correction would then be a real number the switch is dropping.
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
    """CONSERVATION, on real documents, through prices instead of through internals.

    In a zero-rate world every discount factor is one, so the knock-out pays the rebate
    `sum_j fired_j` times and the knock-in's parity leg pays it `alive_T` times. Their sum is the
    rebate EXACTLY when `sum_j alive_{j-1} * (1 - p_j) + alive_final == 1` - the telescoping
    identity `SurvivalLedger` exists to state, here observed from outside the pricer entirely:

        KO + KI == vanilla + rebate

    A ledger that lost mass could not show up in either price on its own - both would still look
    like plausible barrier values - but it would show up HERE, as an error proportional to the
    rebate. Which is why the rebate is walked rather than fixed: at 0 the identity is the plain
    parity statement and says nothing about the ledger, and only the 5.0 and 12.5 rows put weight
    on it. Measured residuals 1.78e-15 / 1.78e-15 / 0.00e+00 at 32768 paths.
    """
    ko, _ = _barrier_run(90.0, on=True, Barrier_Type='Down_And_Out', Cash_Rebate=rebate)
    ki, _ = _barrier_run(90.0, on=True, Barrier_Type='Down_And_In', Cash_Rebate=rebate)
    black = _black_call()[0]
    assert abs(ko + ki - black - rebate) < 1e-11, (
        'in-out parity fails by {:.3e} at a rebate of {}: KO {:.12f} + KI {:.12f} against a '
        'vanilla of {:.12f}. The rebate half of that residual IS the survival ledger'.format(
            ko + ki - black - rebate, rebate, ko, ki, black))


def test_parity_survives_to_second_order_on_the_smooth_path():
    """The same identity at DELTA and GAMMA, which is a statement about WHERE the kink term went.

    `accrual_kink_term` is added to `surv_payoff` - before the in-out parity subtraction - so both
    legs carry one corrected quantity and `KO + KI` telescopes to the closed-form vanilla at every
    order. Add it one branch later, inside `if direction == BARRIER_OUT`, and value parity still
    holds while second-order parity breaks by exactly the term: a defect no value gate can see.

    Measured at 16384 paths, both barrier sides, against Black 9.9476449660 / 0.5497382248 /
    0.0158335075: residuals 0.0e+00 at value and delta, 0.0e+00 and 3.5e-18 at gamma.

    THE MUTATION, run live and recorded: gating that addition on `direction == BARRIER_OUT` leaves
    the knock-out at 0.0097725840 and the knock-in at 0.0071409703, missing Black's gamma by
    0.0010800469 - which is exactly the kink term's own contribution, the same number the ladder
    gate below measures from the other side. That mutant PASSES the ladder gate, because the leg it
    reads still carries the term; and the mutant that removes the term from BOTH legs passes THIS
    one, because parity is a statement about symmetry rather than presence. Neither gate is
    redundant, and neither alone is enough.
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
    """GAMMA FLOWS, and it is the derivative of the delta actually reported - on a LIVE barrier.

    Without the switch this document refuses: the observed crossing registers a latch whose
    correction is a detached coefficient that cannot be differentiated twice. With it the
    registration is skipped, every barrier decision is already a `Phi` whose curvature pathwise AAD
    carries, the terminal relu gets `accrual_kink_term`, and the whole Hessian comes back.

    A never-knocking control can be checked against Black, and is below; a LIVE knock-out has no
    closed form, so the reading that means anything is a CRN ladder of the corrected delta -
    agreement AND flatness, because differencing across a discontinuity produces readings that
    scatter with the bump rather than converging. Measured, 32768 paths: gamma 0.00969411 against
    a ladder best of 0.00974971, 0.57% agreement at 2.12% flatness.

    The never-knocking control: a barrier at 1.0 on a spot of 100 is a European, value 10.0082
    against Black's 9.9476 (0.61%) and gamma 0.01560843 against 0.01583351 (1.42%) - and WITHOUT
    the terminal kink term that number is an exact zero, the pathwise second derivative of a relu,
    which is what makes landing on Black at all the kill rather than the confirmation.

    THE MUTATION, run live and recorded: with the term never built, the LIVE barrier still reports
    a gamma - 0.00861379 rather than an exact zero, because every barrier decision is a `Phi` whose
    curvature pathwise AAD does carry - and it misses its own unchanged ladder by 13.19%. The term
    is worth 0.00108005 of the 0.00969411 total, about a ninth of it, and a live barrier is
    therefore the case where a missing kink term looks MOST like a plausible number. That is why
    the gate is a ladder rather than a comparison against zero.
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
    """RECOMPUTE COMPOSITION for the third product: the switch and the node are orthogonal.

    `Recompute_Inner_MC` re-runs `sim_spot_oss` inside backward() instead of taping it, and the
    node's contract is one function called twice - so the smooth path has to be that same function,
    with everything it reads arriving through theta and nothing settled inside it. Value and the
    whole first-order frame come back bit-identical taped or replayed, on both legs; the parity
    vanilla rides `sd_to_expiry` through theta, which is what makes the knock-in row worth running
    beside the knock-out one.
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
    model by the same naming convention the engine resolves it with, and asked for the switch: the
    deal is SKIPPED with the refusal logged, and it names the model, the daily walk, the stride
    that owns the conditional law it would need, and both remedies. The same document prices
    perfectly well with the switch off, which is what says the refusal is the switch's and not the
    market data's."""
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
    # the loader logs the raised exception's ARGS, so every quote in the message arrives escaped -
    # the assertions read the unescaped text rather than a spelling of the logging layer's
    log = log.replace('\\', '')
    assert 'HestonNandi' in log and 'stride' in log.lower(), log[-1200:]
    assert 'hn_cdf_logret' in log, 'the refusal cites where its Phi would have to come from'
    assert 'GBM' in log and "Branch_And_Weight: 'No'" in log, 'a refusal names its remedies'


AUTOCALL_TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'fixtures', 'autocall_job.json')


def test_the_switch_claims_exactly_three_products_and_no_more(tmp_path):
    """SCOPE, pinned so it cannot be assumed wider than it is.

    `Branch_And_Weight` is honoured by `pv_MC_Tarf`, `pv_MC_Accumulator` and
    `pv_discrete_barrier_option` - the three pricers whose fixing-observed decisions the
    construction covers - and the field's own declaration enumerates them. `pv_MC_AutoCallSwap`,
    the roadmap's stretch half, is NOT one of them; the gate below this one is the measurement the
    deferral rests on, and it names a reason that has nothing to do with the averaging branch.

    So the switch is a NO-OP on an autocall, at every order: the pricer never reads it, and the
    value and the whole second-order block come back bit-identical with it on and off. That is the
    honest behaviour - a switch that quietly half-applied to a fourth product would be
    indistinguishable from one that applied correctly - and this gate is what goes red the day
    somebody teaches the autocall the construction without also giving it the quadrature table,
    the pin and the variance ratio the TARF and the accumulator have above.
    """
    with open(AUTOCALL_TEMPLATE) as f:
        job = json.load(f)
    job['Calc']['Calculation'].update({'Greeks': 'All', 'MCMC_Simulations': 1 << 13})
    crisp, out_c, _ = _run_doc(job, tmp_path, 'ac_off')
    smooth, out_s, _ = _run_doc(_smooth(job), tmp_path, 'ac_on')
    assert crisp == smooth, (crisp, smooth)
    assert np.array_equal(out_c['Results']['Greeks_First'].values,
                          out_s['Results']['Greeks_First'].values)
    assert np.array_equal(out_c['Results']['Greeks_Second'].values,
                          out_s['Results']['Greeks_Second'].values), (
        'the autocall moved at SECOND order under a switch it does not read')
    field = {f.key: f for f in calculation.Base_Revaluation.fields}['Branch_And_Weight']
    for named in ('TARF', 'accumulator', 'discrete barrier'):
        assert named in field.description, (
            'the declaration must enumerate the products it actually covers, and it is missing '
            '{!r}: {}'.format(named, field.description))


# The autocall's own document, built here because the fixture is a single coupon with no put
# barrier at all - which is exactly the configuration that cannot see what the deferral is about.
def _autocall_doc(put_barrier, coupon_days=(180, 365), threshold=1.0, greeks='No', spot=None,
                  sims=1 << 14, seed=1):
    with open(AUTOCALL_TEMPLATE) as f:
        job = json.load(f)
    deal = job['Calc']['Deals']['Deals']['Children'][0]['Instrument']['.Deal']
    dates = [_stamp(d) for d in coupon_days]
    deal['Autocall_Coupons'] = [[d, 0.08] for d in dates]
    deal['Autocall_Thresholds'] = [[d, threshold] for d in dates]
    # one price fixing per coupon date IS the no-averaging branch - the only one in scope
    deal['Price_Fixing'] = [[d, 0.0] for d in dates]
    deal['Expiry_Date'] = dates[-1]
    # `Barrier` is declared as a FRACTION of the strike (instruments.py multiplies it through), so
    # 0.7 is a knock-in 30% below the strike and 1.0 sits exactly on it
    deal['Barrier'] = put_barrier
    deal['Barrier_Dates'] = [dates[-1]] if put_barrier else []
    job['Calc']['Calculation'].update(
        {'Greeks': greeks, 'MCMC_Simulations': sims, 'Random_Seed': seed})
    if spot is not None:
        job['Calc']['MergeMarketData']['ExplicitMarketData'][
            'Price Factors']['EquityPrice.EQ']['Spot'] = spot
    return job


AUTOCALL_SPOT = 100.0


@pytest.mark.parametrize('put_barrier,jumps', [(1.0, False), (0.7, True)],
                         ids=['put barrier AT the strike', 'put barrier BELOW the strike'])
def test_the_autocall_is_deferred_because_its_put_leg_still_jumps(put_barrier, jumps, tmp_path):
    """THE DEFERRAL, and the measurement it rests on. This is the fourth product's whole file.

    The inherited reason was that the no-averaging loop is entangled with the AVERAGING branch the
    construction excludes. THAT IS NOT TRUE and it is worth saying so plainly: the two arms are a
    plain `if factor_dep['no_averaging']` / `else` inside `sim_spot`, sharing no line of pricing;
    the boundary registration is already gated on `no_averaging`, so the averaging arm registers
    nothing to skip; and `terminationDate` is threaded through both as a VALUE mechanism the switch
    would not touch. On the contract's own scope the no-averaging arm is ALREADY the construction:
    the coupon is constant so its fired branch `(1 - p) * L * coup * D_j` is a conditional
    expectation rather than a probability times a sample, the funding leg pays on the survival
    weight `L`, and the continuation is `Phi^-1(p * U)` - the same truncated draw, spelled inline
    (`oss_truncated_draw`'s third named adopter). Turning the switch on would be a one-line change.

    THE REAL BLOCKER IS THAT THE AUTOCALL TAKES TWO DECISIONS PER FIXING AND ONLY ONE OF THEM IS
    INTEGRATED. Beside the autocall trigger there is the put barrier, and it is a bare indicator on
    the drawn spot:

        breach = torch.where(Sj <= putBarrier, 1.0, 0.0)
        P = P + L * D[j] * fx * breach * (rebate - (1.0 - Sj / strike))

    whose payoff at the barrier is `rebate - (1 - B/K)` - an exact zero when the barrier sits ON
    the strike, and a genuine JUMP of that size anywhere below it, which is where every real
    autocall puts it. Ruling 2 is one estimator per decision; shipping the switch here would put
    the construction's name on a product where one decision is integrated and the other is a step
    function, which is ruling 3's failure - a wrong number wearing the right estimator's name -
    arriving through a different door.

    MEASURED, and the two rows of this gate are the control and the finding. A two-coupon autocall,
    16384 paths, spot and strike at 100, 25% vol, delta against a CRN ladder of the reported value:

        put barrier AT the strike (payoff continuous)   0.00% disagreement, 0.00% flatness
        put barrier at 0.7 (payoff jumps by -0.3)      18-22% disagreement, ladder SCATTERING

    across three seeds each. The scatter is `crn_ladder`'s own signature for differencing over a
    discontinuity, and the first-order reading is already wrong, so the second-order one is not
    worth quoting. THE ESTIMATOR IS EXACT WHEREVER THE PUT LEG DOES NOT JUMP, which is what says
    the diagnosis is the jump and not the loop.

    WHAT WOULD LAND IT, so the deferral names a remedy like any other refusal: the put leg closes
    in the same partial moments the TARF's knock-in does. `{S <= B}` lies wholly inside the
    surviving set `{S <= K}` whenever `B <= K`, so `lognormal_partial_moment` at powers 0 and 1
    against `min(B, K)` integrates it exactly, with no second derivation. What it costs is a loop
    RESTRUCTURE - the barrier is observed a screen after the coupon block that holds the interval's
    `m`, `s` and `S_prev`, and a barrier date that does not align with a coupon date has no
    conditioning step at all and would have to refuse by name. That is outside the stretch's
    declared scope ("the terminal put prices on the surviving truncated sample"), and a clean
    three-product landing beats a smeared four. This gate is what goes red when it is done: the
    jumping row starts agreeing, and the autocall is then owed the quadrature table, the pin and
    the variance ratio before the switch may name it.
    """
    delta_out = _run_doc(_autocall_doc(put_barrier, greeks='First'), tmp_path, 'ac_d')[1]
    aad = _first(delta_out, factor='EquityPrice.EQ')
    assert abs(aad) > 1e-6, 'a live autocall must have a spot delta to compare against'
    rung = ladder(
        price=lambda s: _run_doc(_autocall_doc(put_barrier, spot=s), tmp_path, 'ac_v')[0],
        aad=aad, base=AUTOCALL_SPOT, rungs=(1e-3, 2e-3, 5e-3, 1e-2))
    miss = abs(rung.best - rung.aad) / max(abs(rung.aad), 1e-30)
    if jumps:
        assert miss > 0.10, (
            'the autocall put leg no longer misses its own bump ladder ({:.2%}) - the `breach` '
            'indicator has been integrated, so the deferral this file records is stale and the '
            'autocall is owed the switch AND the gates the other three products carry\n{}'.format(
                miss, rung))
    else:
        assert rung.agrees(tol=0.02), (
            'the autocall is inexact even where its put payoff is CONTINUOUS, so the deferral is '
            'blaming the wrong thing - the diagnosis above needs redoing\n{}'.format(rung))
