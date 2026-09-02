"""THE STRIDE'S THREE CONSUMERS, ACROSS FOUR PRICERS - the Heston-Nandi exotic gammas.

Wave 1 built the stride (`utils.hn_component_stride_*`, `tests/test_hn_stride.py`): the k-step
conditional law of ln S given (h, q), cached per fixing interval and quadrature node, exactly
differentiable, the state carried across the jump by quadratic matching, and the drawn spot returned
UN-SHIFTED into the deal's own carry. This is wave 2 - the pricers that consume it, behind ONE
declared field.

WHY ONE FIELD. `HN_Stride` governs all three consumers because all three consent to the same thing:
the CARRIED STATE. A caller who wants the smooth estimator under Heston-Nandi, or a bandwidth-free
jump gamma, or a fixing-to-fixing sampler, is in every case accepting a law whose state is matched
quadratically rather than walked exactly. One approximation, one consent, one switch - and off, the
daily walk is what it always was, to the bit.

  CONSUMER 1, the branch-and-weight HN arm. `Branch_And_Weight: 'Yes'` refused under Heston-Nandi
    citing the stride; it now prices, for the COMPONENT family, with `HN_Stride: 'Yes'` beside it.
    `p` is the k-step law's own Phi over the WHOLE fixing interval - never the last daily Gaussian,
    which is the Delta_t^(-3/2) regime the design forbids - the fired branch closes in the SAME
    cache's Esscher-tilted partial moments, the continuing branch is the survival-truncated
    inversion, and (h, q) ride `utils.hn_component_stride_carry`.
  CONSUMER 2, the conditional-p jump gamma. On the CRISP path a jump on simulated state - the
    TARF's knock-IN, the autocall's put leg - is spliced `P + (P_vib - P_vib.detach()) -
    (P - P.detach())`: the reported value is the sampled indicator's bit for bit and every
    derivative is the analytic mixture's. Where it takes a decision it REPLACES that decision's
    kernel-flux registration; never both.
  CONSUMER 3, the fixing-to-fixing sampler - BOUGHT as a speed lever, and it is not one. An
    unmonitored interval strides instead of walking. Measured 111x to 147x SLOWER on the
    three-fixing TARF at 2^10 to 2^15 inner paths, the ratio WIDENING with the cube, so there is no
    crossover to reach. The diagnosis is not the draw count: a daily step is cheap elementwise work
    over the whole cube (flat in the path count) while a stride pays a fixed per-interval cache
    build plus a Gil-Pelaez inversion PER PATH at every fixing. A batched Phi across the cube is the
    open lever and is not built. THE STEPPING STAYS - it is the smooth estimator's own conditioning
    law - and the machinery is kept for that, not for a speed claim that died.

FOUR PRICERS REACH IT, and the fourth had no gate until this file grew one. `pv_MC_Tarf`,
`pv_MC_Accumulator` and `pv_MC_AutoCallSwap` stride the WHOLE fixing interval, so consumers 1-3
arrive together on them. `pv_discrete_barrier_option` has no whole-interval branch at all - its
monitored step is an exact OSS truncation at a constant barrier either way - so it reaches the
stride ONLY through `kit.substeps`, which makes it the one document where consumer 3 can be scored
on its own, with the estimator held fixed.

THE HEADLINE IS THE GAMMA TABLE (`test_the_tarf_gamma_lands_on_its_crn_ladder` and its autocall
sibling): `Greeks: 'All'` on a component Heston-Nandi TARF and autocall, gated against CRN Hessian
ladders. Those numbers did not exist before this landing - the crisp estimator registers a boundary
correction and `Greeks: 'All'` is refused outright wherever one is registered.

FOUR THINGS THE BUILD FOUND, each now a gate:

  1 THE INVERSION SATURATES OR THE DEAL IS WRONG BY 280x. A trigger the interval cannot reach comes
    back as `1 - 4e-8` rather than one, because that is the inversion's own resolution - and a TARF
    whose remaining target is 1e15 currency units then books `(1 - p) * R` of 4e7 at every fixing it
    was never going to fill. Measured 2.29e8 against a true 8.21e5, on a deal with no knockout in it
    at all. Outside the law's support the answer is a zero or a one, and both are written rather
    than integrated (`pricing.ComponentHestonNandiKit.stride_cdf`).
  2 THE QUADRATURE BOUND CANNOT BE ASKED FOR A TOLERANCE IT WILL NEVER MEET. `hn_component_auto_phi_max`
    doubles until the integrand has decayed past -40 and returns its CAP when it never does. For the
    seed state it always does; for a CARRIED one it routinely does not, and the scan walks straight
    through the point where the A/B/C recursion turns from decaying to diverging. Measured one
    stride into a live fit: the box reached a long-run level of 4.0e-5, whose metric falls to -26 at
    phi 2048 and then reads +156 at 8192 - and the tolerance-seeking scan returned 2^24, where the
    integrand is e+3.7e6. The whole cube came back NaN at the second fixing. `pricing.hn_stride_phi_max`
    keeps the BEST rung and stops when the metric turns.
  3 THE FLOOR MASS REACHES THE CONSUMER. The stride's own declared finding - a Gaussian residual on
    a floored state puts 1.1% of the cube at 1e-12 at k = 21 - drags the long-run component down
    with it, and at a quarter of its own level the inversion no longer decays before it turns. The
    state is FLOORED into the region the cache can invert, which is a declared approximation of
    exactly the kind the model's own variance floor already is, one level up.
  4 THE PANEL COUNT WAS NEVER THE CONSTRAINT. Against a 2^20-path daily walk, 32 panels and 256
    agree to the last printed digit; the footprint, not the accuracy, is what limits a cube, and the
    default put a six-fixing TARF at 16k paths into a 24 GB device's OOM.
  5 THE DRAW WAS DIFFERENTIABLE ONCE, NOT TWICE - the wave-2-critical catch this landing owed, the
    way the un-shift was wave 1's. The reattachment's ONE Newton step off a detached root is exact
    at first order from any starting point and 3.1% out at second, and downstream that is 36% of an
    autocall's gamma and 5.1% of a TARF's. A second step, which is Newton's own quadratic
    convergence written on the tape, takes those to 0.21% and 0.36% and moves no value.

NO MONKEYPATCHING. Every product gate is a real document through `run_baseval`; every mutation is a
DELIBERATELY WRONG CALL written here against the same public seam the pricer calls, so what dies is
the construction and not a patched library.
"""
import os
import sys

# reference-derivus shadow-import guard (MEMORY): pin the package under test to THIS repo.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ast
import logging
import io
import math
import time

import numpy as np
import pandas as pd
import pytest
import torch

import derivus
from derivus import run_baseval, calculation, pricing, utils
from derivus.config import Config
from derivus.instruments import construct_instrument

# the component world, its parameter factor and its TARF, borrowed rather than re-authored: that
# fixture is already gated on the model reaching deals (`tests/test_hn_component.py`, gate 5)
import test_hn_component as hnc
from crn_ladder import ladder

DT = torch.float64
SPY = hnc.SPY
BASE = hnc.BASE
#: The fixing schedule every product gate here runs on - monthly, which is the operating point the
#: stride's own error scan calls its worst (k = 21 sits at 98% of the peak).
FIX_DAYS = [30, 61, 91]
#: Small on purpose. A differentiable stride holds about six (paths, node) complex128 buffers alive
#: per fixing for the backward pass, and `Greeks: 'All'` runs that under `create_graph`.
SIMS, SEED = 1 << 11, 3


def _t(x):
    return torch.tensor(float(x), dtype=DT)


def test_uses_repo_under_test():
    assert derivus.__file__ == os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'derivus', '__init__.py')


# ======================================================================================
# The kit, off the fixture's own parameter factor - the seam every consumer calls.
# ======================================================================================

def _kit():
    """A `ComponentHestonNandiKit` on the fixture's authored parameters, built the way
    `oss_model_scalars` builds one: scalars (-1, 1) to broadcast against the state, the curve flat
    because it is indexed by knot."""
    f = hnc.COMPONENT_FACTOR
    curve = f['L_Curve'].array
    scalars = [_t(f[k]).reshape(-1, 1) for k in utils.HN_COMPONENT_PARAM_NAMES]
    scalars += [torch.tensor(curve[:, 1], dtype=DT)]
    return pricing.ComponentHestonNandiKit(scalars, curve[:, 0], SPY)


class _Shared(object):
    """Just enough calculation state for the kit's own draws - `stride_normals` reads three fields
    and nothing else does."""

    def __init__(self, sims=1, gamma=False):
        self.one = torch.ones([1, 1], dtype=DT)
        self.simulation_batch = 1
        self.gamma = gamma
        self.hn_stride = True


def _fixing(kit, spot=hnc.TARF_SPOT, day=0, n=21, carry=0.0, sims=1):
    """One interval opened as a stride, off the seed state."""
    b_step = torch.full((1, 1), float(carry), dtype=DT)
    return kit.stride_interval(torch.full((1, sims), float(spot), dtype=DT), kit.seed(), b_step, n)


# ======================================================================================
# THE SWITCH. Declared once, default off, and the state carries it without a fallback.
# ======================================================================================

def test_the_field_is_declared_on_base_valuation_alone():
    """One field for three consumers, on the calculation that owns the estimator switch beside it.
    `Credit_Monte_Carlo` declares neither, so its exposure semantics are structurally untouched."""
    declared = {f.name: f for f in calculation.Base_Revaluation.fields}
    assert 'HN_Stride' in declared, 'the field is not declared on Base_Revaluation'
    field = declared['HN_Stride']
    assert field.default == 'No', 'the switch does not default off'
    assert sorted(field.values) == ['No', 'Yes']
    for other in (calculation.Credit_Monte_Carlo,):
        assert 'HN_Stride' not in {f.name for f in other.fields}, (
            '{} declares the field'.format(other.__name__))


def test_the_state_declares_it_so_a_pricer_reads_it_without_a_fallback():
    state = utils.Calculation_State(
        {}, torch.ones([1, 1], dtype=DT), 1, None, 'Constant', 1, False)
    assert state.hn_stride is False, 'the default is not off on the state itself'


def test_the_declared_default_is_the_fallback_every_pricer_reads():
    """The schema-emission contract: what the field declares and what a pricer falls back to are the
    same answer, so a document that omits the key and a document that writes 'No' are one run."""
    declared = {f.name: f for f in calculation.Base_Revaluation.fields}['HN_Stride'].default
    state = utils.Calculation_State(
        {}, torch.ones([1, 1], dtype=DT), 1, None, 'Constant', 1, False)
    assert (declared == 'Yes') is state.hn_stride


# ======================================================================================
# THE REFUSALS, flipped. Component + the switch prices; everything else still refuses by name.
# ======================================================================================

class _State(utils.Calculation_State):
    def __init__(self, smooth, stride):
        super(_State, self).__init__(
            {}, torch.ones([1, 1], dtype=DT), 1, None, 'Constant', 1, False)
        self.branch_and_weight = smooth
        self.hn_stride = stride


def _deal_data(spot_model):
    """A real `FXTARFOptionDeal` with `HN_Params` shaped as `instruments.get_hn_factor` shapes it."""
    deal = hnc._tarf_deal(FIX_DAYS)
    instrument = construct_instrument(dict(deal), {})
    factor_dep = {}
    if spot_model is not None:
        factor_dep['HN_Params'] = [(False, [utils.Factor(
            spot_model + 'ModelParameters', ('ZAR', 'Alpha'))], spot_model, {})]
    return utils.DealDataType(
        Instrument=instrument, Factor_dep=factor_dep, Time_dep=None, Calc_res=None)


def test_the_component_arm_is_admitted_under_both_switches():
    """The sentence the stride was built to flip: `Branch_And_Weight` under Heston-Nandi."""
    assert pricing.branch_and_weight(
        _State(True, True), _deal_data('HestonNandiComponent')) is True


@pytest.mark.parametrize('spot_model', ['HestonNandi', 'HestonNandiComponent'])
def test_without_the_stride_switch_both_families_still_refuse(spot_model):
    """The refusal is not lifted by the estimator switch alone. It names the model, cites where its
    Phi comes from, and now names the field that consents to the carried state."""
    with pytest.raises(ValueError) as refusal:
        pricing.branch_and_weight(_State(True, False), _deal_data(spot_model))
    message = str(refusal.value)
    for token in ('Branch_And_Weight', spot_model, 'HN_Stride', 'hn_cdf_logret', 'GBM',
                  "Branch_And_Weight: 'No'"):
        assert token in message, 'the refusal does not name {}: {}'.format(token, message)
    assert 'stride' in message.lower()


def test_the_plain_family_refuses_even_with_the_stride_switch_on():
    """THE STRIDE IS THE COMPONENT RECURSION'S CACHE. It carries a C*q state axis the plain law has
    no room for, so a plain deal is not strided - it is REMAPPED, and the refusal says so by naming
    the exact map (`utils.hn_component_from_plain`, gated at 1.5e-13) as the remedy."""
    with pytest.raises(ValueError) as refusal:
        pricing.branch_and_weight(_State(True, True), _deal_data('HestonNandi'))
    message = str(refusal.value)
    assert 'hn_component_from_plain' in message, message
    assert "SpotModel: 'HestonNandiComponent'" in message, message


@pytest.mark.parametrize('stride', [False, True])
def test_the_estimator_switch_off_admits_every_model(stride):
    """Off, the seam answers False and asks no questions - the stride switch alone changes the
    SAMPLER, never the estimator."""
    for model in (None, 'HestonNandi', 'HestonNandiComponent'):
        assert pricing.branch_and_weight(_State(False, stride), _deal_data(model)) is False


# ======================================================================================
# THE SEAM. The verbs the three consumers stand on, against references written here.
# ======================================================================================

def test_the_carry_shift_moves_a_level_into_the_strips_own_measure():
    """THE UN-SHIFT, half one. The strip is built at r = 0 and keyed on calendar position alone, so
    a deal's own carry enters as a move on the moneyness and NOT as a second strip."""
    kit = _kit()
    b = 0.0004
    plain, carried = _fixing(kit, carry=0.0), _fixing(kit, carry=b)
    assert plain.strip is carried.strip, 'the cache was keyed on the carry'
    level = _t(hnc.TARF_SPOT * 1.05)
    assert float(kit.stride_bound(plain, level) - kit.stride_bound(carried, level)) == \
        pytest.approx(b * plain.n_steps, rel=1e-12)


def test_the_stride_returns_the_spot_un_shifted_into_the_deals_carry():
    """THE UN-SHIFT, half two, and the mutation the stride suite's own gate covers one level down.
    A consumer that moves the barrier IN and forgets to move the return OUT keeps the survival
    WEIGHT right while the whole survivor law sits a carry away from where the barrier is.

    Measured here at k = 21 under b = 4r on the fixture's own law: the median survivor sits
    `b * k` low without the un-shift, exactly, because that is what the omission is.
    """
    kit = _kit()
    b, n = 0.0004, 21
    fix = _fixing(kit, carry=b, sims=4096)
    u = torch.rand((1, 4096), dtype=DT)
    e1, e2 = pricing.stride_normals(_Shared(), 4096, False)
    x_cap = kit.stride_bound(fix, _t(hnc.TARF_SPOT * 1.10))
    _, right, _ = kit.stride_advance(fix, torch.full((1, 1), b, dtype=DT), u, e1, e2, x_cap, True)
    # THE MUTATION: the same draw with `b_step` left off `hn_component_stride_step`
    x, _ = utils.hn_component_stride_draw(fix.strip, u, fix.h, fix.q, x_cap)
    wrong = fix.spot * torch.exp(x)
    gap = float(torch.log(right.median() / wrong.median()))
    assert gap == pytest.approx(b * n, rel=1e-6), (
        'the un-shift is not the deal carry over the interval: {:.6g} against {:.6g}'.format(
            gap, b * n))
    assert float(right.max()) <= hnc.TARF_SPOT * 1.10 * (1 + 1e-12), 'the truncation missed'


def test_the_inversion_saturates_outside_the_laws_own_support():
    """FINDING 1. A trigger the interval cannot reach is a probability of exactly one, and the
    inversion cannot say so - it resolves to about 1e-8, and its oscillation aliases long before
    that. Both facts are read here: the raw quadrature at an unreachable bound, and the saturation
    that makes the answer exact."""
    kit = _kit()
    fix = _fixing(kit)
    raw_x = kit.stride_bound(fix, _t(1.0e15))
    assert float(raw_x) > 30.0, 'the fixture no longer has an unreachable bound in it'
    aliased = utils.hn_component_stride_cdf(fix.strip, raw_x, fix.h, fix.q)
    assert float(aliased) > 1.0 + 1e-3, (
        'the raw inversion no longer aliases at an unreachable bound - the saturation may have '
        'stopped being load-bearing: {}'.format(float(aliased)))
    assert float(kit.stride_cdf(fix, fix.strip, raw_x)) == 1.0, 'the saturation is not exact'
    assert float(kit.stride_cdf(fix, fix.strip, -raw_x)) == 0.0
    # and INSIDE the support nothing is touched
    inside = kit.stride_bound(fix, _t(hnc.TARF_SPOT * 1.02))
    assert float(kit.stride_cdf(fix, fix.strip, inside)) == float(
        utils.hn_component_stride_cdf(fix.strip, inside, fix.h, fix.q))


def test_an_unreachable_target_costs_the_deal_nothing():
    """FINDING 1, as the deal reads it. `(1 - p) * R` is the TARF's fired branch and `R` can be
    enormous, so a survival probability off by the inversion's own resolution is off by
    `4e-8 * R` in currency. Written against the SEAM rather than the pricer, so the number is the
    construction's and not a run's."""
    kit = _kit()
    fix = _fixing(kit)
    R = 1.0e9 * 1.0e6                                     # the fixture's unreachable target x N1
    x = kit.stride_bound(fix, _t(1.0e15))
    lo, hi = kit.stride_support(fix)
    unsaturated = utils.hn_component_stride_cdf(fix.strip, x.clamp(min=lo, max=hi), fix.h, fix.q)
    leak = float((1.0 - unsaturated) * R)
    assert leak > 1.0e7, 'the leak this gate exists for is no longer measurable: {}'.format(leak)
    assert float((1.0 - kit.stride_cdf(fix, fix.strip, x)) * R) == 0.0


def test_the_bound_stops_where_the_recursion_turns():
    """FINDING 2. The metric decays, then TURNS - that is the divergence, and a scan chasing a
    tolerance past it returns a bound where the integrand is astronomically large."""
    kit = _kit()
    omegas = list(kit.omegas(21, 21))
    box = kit.params[0].new_tensor
    # a state one quarter of its own long-run level, which is where a carried cube reaches after a
    # stride or two and where the metric turns BEFORE any usable tolerance
    hq = (1.0e-12, 2.125e-5)
    phi_max, best = pricing.hn_stride_phi_max(
        omegas, kit.params, box((hq[0], hq[0])), box((hq[1], hq[1])))
    assert best > pricing.HN_STRIDE_PHI_LOG_TOL, (
        'this state now decays past the stop, so the scan is not being made to turn: '
        '{:.4g}'.format(best))
    # ONE DOUBLING PAST the chosen bound the metric is WORSE, which is what "turned" means - and it
    # is why a tolerance-seeking scan walks all the way to its cap on this state
    z = torch.tensor([2.0 * phi_max], dtype=DT) * 1j
    beyond = float(utils.hn_component_logmgf(
        z, omegas, box((hq[0],)).reshape(-1, 1), box((hq[1],)).reshape(-1, 1),
        *kit.params, _t(0.0)).real.max()) - math.log(2.0 * phi_max)
    assert beyond > best, (
        'the scan stopped somewhere other than the turning point: {:.4g} at {:.0f} against '
        '{:.4g} beyond'.format(best, phi_max, beyond))
    # -11 is still a probability - measured 1.3e-3 against a 2^20-path walk at every decile - so
    # this state is integrated where it stands. What the floor is for is the rung below it
    assert best <= pricing.HN_STRIDE_PHI_MIN_DECAY
    floor, _ = kit.stride_state_floor(omegas, (hq[0], hq[0]), (hq[1], hq[1]))
    assert floor == 0.0, 'a state the cache CAN invert was floored anyway'


def test_the_state_is_floored_into_the_region_the_cache_can_invert():
    """FINDING 3. The floor is expressed in the interval's OWN long-run level, read off the model's
    omega strip - no new constant - and a healthy cube pays nothing for it."""
    kit = _kit()
    omegas = list(kit.omegas(0, 21))
    healthy, _ = kit.stride_state_floor(omegas, (7.0e-5, 8.0e-5), (7.0e-5, 9.0e-5))
    assert healthy == 0.0, 'a healthy cube was floored'
    drifted, phi = kit.stride_state_floor(omegas, (1e-12, 2.0e-4), (2.0e-6, 1.6e-4))
    assert drifted > 0.0, 'a cube at a fiftieth of its own level was not floored'
    level = float(omegas[0]) / (1.0 - float(kit.params[3]))
    assert drifted <= pricing.HN_STRIDE_STATE_FLOORS[-1] * level * (1 + 1e-12)


# ======================================================================================
# THE FIRED BRANCH. Esscher-tilted partial moments off the SAME cache, against two references.
# ======================================================================================

def test_the_tilt_at_one_is_the_share_measure_the_european_price_rides():
    """ONE SPELLING. `E[S_k 1{S_k > B}]` is the forward times the SHARE-measure probability, and
    that probability is `1 - P1` - the very contour `utils.cf_european_probabilities` runs for the
    delta leg of `hn_component_call`. So the stride's partial moment and the component European
    price are the same two probabilities assembled twice, and they agree.

    At r = 0 the martingale property makes `log M(1)` an exact structural zero, which this reads
    rather than assumes: a tilt normaliser that is not zero is a recursion that does not price a
    forward.
    """
    kit = _kit()
    fix = _fixing(kit, n=21)
    tilt, A0, B0, C0 = kit.stride_tilt(fix, 1.0)
    assert abs(float(A0)) < 1e-14 and abs(float(B0)) < 1e-14 and abs(float(C0)) < 1e-14, (
        'log M(1) is not the forward growth at r = 0: {} {} {}'.format(
            float(A0), float(B0), float(C0)))
    K = hnc.TARF_SPOT * 1.03
    x = kit.stride_bound(fix, _t(K))
    # E[(S - K) 1{S > K}] over the interval IS the European call under the same law and state
    got = kit.stride_fired_gain(fix, x, K, True, 1.0)
    want = utils.hn_component_call(
        _t(hnc.TARF_SPOT), _t(K), list(kit.omegas(0, 21)), kit.seed()[0].reshape(-1)[0],
        kit.q0().reshape(-1)[0], *kit.params, _t(0.0))
    assert float(got) == pytest.approx(float(want), rel=2e-6), (
        'the tilted partial moment and the component call disagree: {:.10g} vs {:.10g}'.format(
            float(got), float(want)))


@pytest.mark.parametrize('fired_above', [True, False])
def test_the_partial_moments_match_a_quadrature_over_the_strides_own_density(fired_above):
    """A SECOND, INDEPENDENT reference: the tail integral taken directly against the stride's PDF,
    which shares no line with the tilt - it is the inversion without the `1/(i phi)`, so an error in
    the Esscher construction cannot hide behind the same error in its oracle."""
    kit = _kit()
    fix = _fixing(kit, n=21)
    B = hnc.TARF_SPOT * 1.02
    xb = float(kit.stride_bound(fix, _t(B)))
    mean, var, _, _ = utils.hn_component_stride_cumulants(fix.strip, fix.h, fix.q)
    lo, hi = float(mean) - 8 * float(var) ** 0.5, float(mean) + 8 * float(var) ** 0.5
    a, b = (xb, hi) if fired_above else (lo, xb)
    grid = torch.linspace(a, b, 20001, dtype=DT)
    dens = utils.hn_component_stride_pdf(fix.strip, grid, fix.h, fix.q)
    spot = fix.spot.reshape(-1)[0]
    ref = torch.trapz(dens * spot * torch.exp(grid), grid)
    got = kit.stride_partial_moment(fix, kit.stride_bound(fix, _t(B)), fired_above, 1.0)
    assert float(got) == pytest.approx(float(ref), rel=1e-5), (
        'the tilt and the density quadrature disagree: {:.10g} vs {:.10g}'.format(
            float(got), float(ref)))


def test_p_times_the_realised_payoff_is_biased():
    """THE MUTATION THAT MUST DIE, one family over. The fired branch is a CONDITIONAL expectation;
    `p x (payoff on the surviving sample)` samples the payoff on the wrong side of the trigger and
    weights it by the fired probability. It is not variance - it does not shrink with paths."""
    kit = _kit()
    n_paths = 1 << 15
    fix = _fixing(kit, n=21, sims=n_paths)
    B = hnc.TARF_SPOT * 0.98
    xb = kit.stride_bound(fix, _t(B))
    truth = float(kit.stride_fired_gain(fix, xb, B, False, 1.0).reshape(-1)[0])
    # the surviving branch, drawn exactly as the loop draws it, then weighted by the fired mass
    torch.manual_seed(11)
    u = torch.rand((1, n_paths), dtype=DT)
    e1, e2 = pricing.stride_normals(_Shared(), n_paths, False)
    b_step = torch.zeros((1, 1), dtype=DT)
    p, S, _ = kit.stride_advance(fix, b_step, u, e1, e2, xb, False)
    mutant = float(((1.0 - p) * (S - B)).mean())
    # MEASURED on the fixture at k = 21, a knock-in-shaped gain over the lower tail: the truth is
    # -0.1387 and the shortcut reads +0.2433 - not a bias but a SIGN, because the sample it weights
    # is drawn from the SURVIVING side of the trigger and the payoff is signed across it
    assert mutant * truth < 0.0, (
        'p x realised no longer carries the wrong sign: {:.6g} against {:.6g}'.format(
            mutant, truth))
    assert abs(mutant - truth) > 2.0 * abs(truth), 'the mutation is no longer material'


def test_the_drawn_return_is_differentiable_TWICE():
    """THE WAVE-2-CRITICAL CATCH, and the reason every gamma in this file exists.

    The stride's draw finds its root under `no_grad` and reattaches the gradient by Newton steps
    whose terms carry the graph. ONE step is exact at first order from any starting point - which is
    what wave 1 gated - and 3% out at SECOND, because a detached root goes in with ``dx = 0`` and
    the ``g_xx dx^2 + 2 g_xtheta dx`` terms never appear. A second step starts from an ``x`` whose
    first derivative is already exact and closes it.

    Scored against the ROOT ITSELF, bumped: the inverter's own solve at the bumped truncation level,
    which carries no reattachment at all and so cannot share the defect. Measured k = 21 on the
    fixture's law: d2 -3.94967 against a bumped-root -3.94975, and the VALUE is the root to the last
    bit. The one-step reattachment read -3.8318 there, and 36% of an autocall's gamma downstream.
    """
    kit = _kit()
    fix = _fixing(kit, n=21)
    u = torch.tensor([[0.37]], dtype=DT)

    def root_at(c):
        """The draw's VALUE with no reattachment in it - the inverter's own root."""
        with torch.no_grad():
            phi = utils.hn_component_stride_cdf(fix.strip, c, fix.h, fix.q)
            return float(utils.hn_component_stride_invert(fix.strip, u * phi, fix.h, fix.q))

    c0 = float(kit.stride_bound(fix, _t(hnc.TARF_SPOT * 1.06)))
    cap = torch.tensor(c0, dtype=DT, requires_grad=True)
    x, _ = utils.hn_component_stride_draw(fix.strip, u, fix.h, fix.q, cap)
    d1, = torch.autograd.grad(x.reshape(()), cap, create_graph=True)
    d2, = torch.autograd.grad(d1, cap)
    assert float(x) == pytest.approx(root_at(_t(c0)), abs=1e-14), 'the reattachment moved the value'
    for step in (1e-3, 3e-3):
        up, mid, dn = (root_at(_t(c0 + step)), root_at(_t(c0)), root_at(_t(c0 - step)))
        assert (up - dn) / (2 * step) == pytest.approx(float(d1), rel=2e-3), 'first order'
        assert (up - 2 * mid + dn) / step ** 2 == pytest.approx(float(d2), rel=0.01), (
            'the drawn return is not differentiable twice at bump {}: autograd {:.8g} against a '
            'bumped-root {:.8g}'.format(step, float(d2), (up - 2 * mid + dn) / step ** 2))


# ======================================================================================
# THE SPLICE (consumer 2). Value exactly crisp, derivatives exactly the mixture's.
# ======================================================================================

def test_the_splice_is_worth_an_exact_zero_and_carries_the_mixtures_derivatives():
    """`crisp + (mixture - mixture.detach()) - (crisp - crisp.detach())`. Both parentheses are exact
    IEEE zeros, so the value is bit-identical to the crisp term; the gradient AND the Hessian are
    the mixture's, there being no detached coefficient left anywhere in it."""
    s = torch.tensor(1.7, dtype=DT, requires_grad=True)
    crisp = torch.where(s > 1.5, s * 0.0 + 3.0, s * 0.0)      # a bare jump: no derivative at all
    mixture = s ** 3                                           # the analytic branch
    total = crisp + pricing.splice_conditional_p(crisp, mixture)
    assert float(total) == float(crisp), 'the splice moved the value'
    assert torch.equal(total.detach(), crisp.detach())
    d, = torch.autograd.grad(total, s, create_graph=True)
    assert float(d) == pytest.approx(3.0 * 1.7 ** 2, rel=1e-14), 'first order is not the mixture\'s'
    dd, = torch.autograd.grad(d, s)
    assert float(dd) == pytest.approx(6.0 * 1.7, rel=1e-14), 'second order is not the mixture\'s'


# ======================================================================================
# THE PRODUCTS. Real documents, through `run_baseval`.
# ======================================================================================

def _tarf_cfg(target=2.0, barrier=0.9, leverage=2.0, fixings=None, spot=None):
    deal = hnc._tarf_deal(fixings or FIX_DAYS, target=target)
    deal['Barrier'] = barrier * hnc.TARF_STRIKE if barrier else 0.0
    deal['LeverageNotional'] = leverage * hnc.TARF_N1
    cfg = hnc._tarf_config(deal)
    if spot is not None:
        cfg.params['Price Factors']['FxRate.ZAR']['Spot'] = spot
    return cfg, 'TARF1'


def _acc_cfg(barrier=1.15, spot=None):
    dates = [BASE + pd.Timedelta(days=d) for d in FIX_DAYS]
    deal = {'Object': 'FXAccumulatorOptionDeal', 'Reference': 'ACC1', 'Currency': 'USD',
            'Underlying_Currency': 'ZAR', 'Discount_Rate': 'USD', 'FX_Volatility': 'USD.ZAR',
            'Buy_Sell': 'Buy', 'Option_Type': 'Call', 'Strike_Price': hnc.TARF_STRIKE,
            'Underlying_Amount': hnc.TARF_N1, 'LeverageNotional': 2.0 * hnc.TARF_N1,
            'Barrier_Type': 'Up_And_Out', 'Barrier_Price': barrier * hnc.TARF_STRIKE,
            'Barrier_Hit': 'No',
            'Accumulator_ExpiryDates': [[d, d, 0.0] for d in dates]}
    cfg = hnc._tarf_config(deal)
    valuation = {'FXAccumulatorOptionDeal': {'SpotModel': 'HestonNandiComponent',
                                             'Steps_Per_Year': SPY}}
    cfg.params['Valuation Configuration'] = valuation
    cfg.deals['Deals']['Children'] = [{'Instrument': construct_instrument(deal, valuation)}]
    if spot is not None:
        cfg.params['Price Factors']['FxRate.ZAR']['Spot'] = spot
    return cfg, 'ACC1'


AC_SPOT = AC_STRIKE = 100.0


def _equity_config(deal, valuation, spot):
    """The COMPONENT equity world both equity fixtures here price in - one Price Factors block, the
    parameter factor resolved off the deal's own `Equity` by the naming convention rather than
    declared per deal type (which is why the TARF's own `_tarf_config` cannot serve an equity)."""
    cfg = Config()
    cfg.params['System Parameters']['Base_Currency'] = 'USD'
    cfg.params['System Parameters']['Base_Date'] = BASE
    cfg.params['Price Factors'] = {
        'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Priority': 1,
                       'Spot': 1.0},
        'InterestRate.USD': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                             'Curve': utils.Curve([], [[0.0, 0.0], [5.0, 0.0]])},
        'EquityPrice.EQ': {'Spot': spot if spot is not None else AC_SPOT, 'Currency': 'USD',
                           'Interest_Rate': 'USD', 'Issuer': '', 'Respect_Default': 'No',
                           'Jump_Level': 0.0},
        'DividendRate.EQ': {'Currency': 'USD', 'Floor': None,
                            'Curve': utils.Curve([], [[0.01, 0.0], [5.0, 0.0]])},
        'VolatilityGrid.EQ': {'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness',
                              'Surface': utils.Curve([], [[m, t, 0.14] for m in (0.8, 1.0, 1.2)
                                                          for t in (0.02, 2.0)])},
        'HestonNandiComponentModelParameters.EQ': dict(hnc.COMPONENT_FACTOR)}
    cfg.params['Price Models'] = {}
    cfg.params['Valuation Configuration'] = valuation
    cfg.deals = {'Attributes': {'Reference': 'test', 'Tag_Titles': ''},
                 'Deals': {'Children': [{'Instrument': construct_instrument(deal, valuation)}]},
                 'Calculation': {'Base_Date': BASE, 'Currency': 'USD'}}
    return cfg


#: The FOURTH consumer's schedules. `pv_discrete_barrier_option` is the one pricer that reaches the
#: stride ONLY through `kit.substeps` - it has no whole-interval stride branch - so it is where the
#: case split is readable: monthly barrier dates leave 20 unmonitored days an interval and the
#: stride opens on them; DAILY ones leave zero, and `substeps` walks nothing at all.
BARRIER_MONTHLY, BARRIER_DAILY = ([30, 61, 91], 91), ([1, 2, 3, 4, 5], 5)


def _barrier_cfg(schedule=BARRIER_MONTHLY, spot=None, bprice=85.0):
    """A COMPONENT Heston-Nandi DISCRETE barrier - the stride's fourth consumer, and the only one
    whose whole use of the stride is the unmonitored sub-step.

    The expiry is authored INSIDE `Barrier_Dates` on purpose: `calc_dependencies` unions the expiry
    into the observation schedule and marks anything not in `Barrier_Dates` as a non-barrier date,
    so leaving it out would add a fourth interval on the other branch and the two schedules would
    stop differing in one thing only.
    """
    bdays, horizon = schedule
    dates = [BASE + pd.Timedelta(days=d) for d in bdays]
    deal = {'Object': 'EquityBarrierOption', 'Reference': 'BARR1', 'Currency': 'USD',
            'Payoff_Currency': 'USD', 'Equity': 'EQ', 'Dividends': 'EQ', 'Discount_Rate': 'USD',
            'Equity_Volatility': 'EQ', 'Buy_Sell': 'Buy', 'Option_Type': 'Call',
            'Strike_Price': AC_STRIKE, 'Expiry_Date': BASE + pd.Timedelta(days=horizon),
            'Units': 1000.0, 'Barrier_Type': 'Down_And_Out', 'Barrier_Price': bprice,
            'Cash_Rebate': 0.0, 'Barrier_Dates': dates,
            'Barrier_Monitoring_Frequency': pd.DateOffset(days=1)}
    valuation = {'EquityBarrierOption': {'SpotModel': 'HestonNandiComponent',
                                         'Steps_Per_Year': SPY}}
    return _equity_config(deal, valuation, spot), 'BARR1'


def _autocall_cfg(put_barrier=0.85, rebate=0.0, spot=None):
    """An equity autocall on the COMPONENT model, with a put barrier OFF the strike - which is where
    the crisp arm's put leg is a bare jump and the mixture has something to carry."""
    dates = [BASE + pd.Timedelta(days=d) for d in FIX_DAYS]
    deal = {'Object': 'QEDI_CustomAutoCallSwap', 'Reference': 'AC1', 'Currency': 'USD',
            'Payoff_Currency': 'USD', 'Equity': 'EQ', 'Dividends': 'EQ', 'Discount_Rate': 'USD',
            'Equity_Volatility': 'EQ', 'Buy_Sell': 'Buy', 'Option_Type': 'Call',
            'Strike_Price': AC_STRIKE, 'Expiry_Date': dates[-1], 'Units': 1000.0,
            'Settlement_Style': 'Cash', 'Option_On_Forward': 'No', 'Option_Style': 'European',
            'Barrier': put_barrier * AC_STRIKE, 'Payoff_Type': None, 'Cash_Rebate': rebate,
            'Price_Fixing': [[d, 0.0] for d in dates],
            'Autocall_Coupons': [[d, 0.05] for d in dates],
            'Autocall_Thresholds': [[d, 1.05] for d in dates],
            'Barrier_Dates': [dates[-1]], 'Autocall_Floating': []}
    valuation = {'QEDI_CustomAutoCallSwap': {'SpotModel': 'HestonNandiComponent',
                                             'Steps_Per_Year': SPY}}
    return _equity_config(deal, valuation, spot), 'AC1'


def _run(cfg_ref, sims=SIMS, seed=SEED, **calc):
    cfg, ref = cfg_ref
    over = {'MCMC_Simulations': sims, 'Random_Seed': seed}
    over.update(calc)
    _, out = run_baseval(cfg, prec=DT, overrides=over)
    rows = out['Results']['mtm']
    value = float(rows[rows['Reference'] == ref]['Value'].iloc[0])
    assert np.isfinite(value), 'the deal was skipped or produced a non-finite value'
    return value, out


def _first(out, factor):
    frame = out['Results']['Greeks_First']
    column = [c for c in frame.columns if c != 'Value'][0]
    return float(frame.loc[[i for i in frame.index if str(i[0]) == factor][0], column])


def _second(out, factor):
    frame = out['Results']['Greeks_Second']
    row, = [i for i in frame.index if str(i[0]) == factor]
    col, = [c for c in frame.columns if str(c[1]) == factor]
    return float(frame.loc[row, col])


ON = {'HN_Stride': 'Yes'}
SMOOTH = {'HN_Stride': 'Yes', 'Branch_And_Weight': 'Yes'}


# ======================================================================================
# OFF IS OFF, to the bit, on every product the switch reaches.
# ======================================================================================

@pytest.mark.parametrize('build,name', [(_tarf_cfg, 'tarf'), (_acc_cfg, 'accumulator'),
                                        (_autocall_cfg, 'autocall')])
def test_off_is_off_on_a_real_document(build, name):
    """The key ABSENT and the key written 'No' are one run - value and the whole reported frame, by
    `array_equal` rather than a tolerance. The daily walk is untouched down to its RNG draws."""
    absent, out_a = _run(build())
    written, out_w = _run(build(), HN_Stride='No')
    assert absent == written, '{}: the declared default moved the value'.format(name)
    # `DataFrame.equals` rather than `array_equal`, which reads the frame's own NaN parent cells as
    # unequal to themselves - the comparison is still exact, not a tolerance
    assert out_a['Results']['mtm'].equals(out_w['Results']['mtm']), (
        '{}: the reported frame moved'.format(name))


def test_the_gbm_path_is_untouched_by_the_switch():
    """A deal with no spot model at all cannot reach the stride, and the switch is inert on it."""
    def gbm():
        deal = hnc._tarf_deal(FIX_DAYS, target=2.0)
        deal['Barrier'] = 0.9 * hnc.TARF_STRIKE
        deal['LeverageNotional'] = 2.0 * hnc.TARF_N1
        cfg = hnc._tarf_config(deal)
        # the deal declares NO spot model, so the factor is not a dependency and the walk is the
        # surface's own lognormal - which the stride cannot reach at all
        cfg.params['Valuation Configuration'] = {}
        del cfg.params['Price Factors']['HestonNandiComponentModelParameters.ZAR']
        cfg.deals['Deals']['Children'] = [{'Instrument': construct_instrument(deal, {})}]
        return cfg, 'TARF1'

    off, _ = _run(gbm())
    on, _ = _run(gbm(), **ON)
    assert off == on, 'the switch moved a GBM deal'


def test_the_discrete_barrier_is_the_fourth_consumer_and_the_case_split_is_readable_on_it():
    """THE FOURTH CONSUMER, which had no gate. `pv_discrete_barrier_option` reaches the stride ONLY
    through `kit.substeps` - it has no whole-interval stride branch, its monitored step being an
    exact OSS truncation at a constant barrier - so it is the one pricer where consumer 3 is the
    whole of what the switch does, and where the case split can be read as a VALUE.

    ONE DOCUMENT, TWO SCHEDULES, and they differ in one thing. Monthly barrier dates leave 20
    unmonitored days an interval and the stride opens on them: the value moves 2.0% on one seed at
    2048 paths (0.55% at 8192), which is the carried-state approximation and is what the band gate
    below scores properly, across seeds. DAILY barrier dates leave ZERO - `nj[j] - 1 == 0` - so
    `substeps` strides nothing at all and off/on is BIT-IDENTICAL.

    The pair is the point. A hex identity on its own is what a deal the switch cannot REACH also
    reads (`test_the_gbm_path_is_untouched_by_the_switch` is exactly that reading), so the daily
    number says "the case split handed this shape no unmonitored days" only because the monthly
    sibling off the same builder moves.
    """
    daily_off, _ = _run(_barrier_cfg(BARRIER_DAILY))
    daily_on, _ = _run(_barrier_cfg(BARRIER_DAILY), **ON)
    assert daily_off.hex() == daily_on.hex(), (
        'a daily-monitored discrete barrier strides something: {:.17g} against {:.17g}'.format(
            daily_off, daily_on))
    monthly_off, _ = _run(_barrier_cfg(BARRIER_MONTHLY))
    monthly_on, _ = _run(_barrier_cfg(BARRIER_MONTHLY), **ON)
    moved = abs(monthly_on - monthly_off) / abs(monthly_off)
    assert moved > 1.0e-3, (
        'the monthly schedule no longer strides either, so the daily reading above is vacuous: '
        '{:.17g} against {:.17g}'.format(monthly_off, monthly_on))
    print('\ndiscrete barrier: daily bit-identical, monthly moves {:.2%}'.format(moved))


def test_the_stride_opens_at_k_one_and_is_inert_rather_than_absent():
    """THE WORDING THE CASE SPLIT CARRIED WAS WRONG, and this is the number that corrects it.

    "A daily-monitored contract keeps the daily path and the stride NEVER FIRES there" is true of
    the shape above, where the split hands `substeps` zero unmonitored days. It is NOT true of a
    pricer that strides the WHOLE fixing interval: `pv_MC_Tarf` and `pv_MC_Accumulator` hand a
    daily-monitored contract an interval of ONE DAY, and one day is a stride of length one. The
    stride OPENS.

    It is INERT there rather than absent, and inert for a reason that is exact: the one-step law IS
    the daily law (`tests/test_hn_stride.py::test_the_one_step_stride_is_the_daily_advance` - h_1 is
    quadratic in z and the one-step return affine in it, so the quadratic carry's residual is zero
    and the carry is not an approximation at k = 1, it is the answer), and the same uniform draws
    the same quantile of it.

    NOT BIT-IDENTICAL, and that is the honest half. The quantile is taken by Gil-Pelaez inversion
    rather than by `norm_icdf`, so what agrees is the LAW and the number agrees to the inversion's
    own resolution. Measured 1.8e-11 relative against 2.9e-2 on the monthly schedule the stride is
    actually for - eight orders of magnitude apart, which is the reading that says k = 1 is the
    identity and k = 21 is the approximation.
    """
    daily = _tarf_cfg(fixings=[1, 2, 3])
    off, _ = _run(daily)
    on, _ = _run(_tarf_cfg(fixings=[1, 2, 3]), **ON)
    inert = abs(on - off) / abs(off)
    assert off.hex() != on.hex(), (
        'the k = 1 stride is now bit-identical - either it stopped opening or the draw stopped '
        'going through the inversion, and this gate no longer reads what it says it reads')
    assert inert < 1.0e-9, (
        'the one-step stride is not the daily law: {:.4g} relative ({:.17g} against {:.17g})'.format(
            inert, on, off))
    monthly_off, _ = _run(_tarf_cfg())
    monthly_on, _ = _run(_tarf_cfg(), **ON)
    approx = abs(monthly_on - monthly_off) / abs(monthly_off)
    assert approx > 1.0e-3, 'the monthly schedule no longer strides: {:.4g}'.format(approx)
    print('\nTARF k=1 {:.3g} relative (inversion resolution) against k=21 {:.3g} '
          '(the carried state)'.format(inert, approx))


# ======================================================================================
# DISTRIBUTIONAL ACCEPTANCE: the strided pricer against the DAY-STEPPED one it replaces.
# ======================================================================================

#: What five seeds of the day-stepped pricer and five of the strided one measured on the fixture,
#: 8192 inner paths: the strided TARF reads -2.5% of the daily one crisp and -1.5% smooth on a deal
#: that never knocks, and +2.4% / +1.9% on one that does - inside 3 standard errors of the daily
#: pricer's own seed-to-seed band either way. That is the CARRY, compounded through three to six
#: jumps, and it is the declared approximation carrying its number at the deal level.
STRIDE_VS_DAILY_TOL, STRIDE_VS_DAILY_SE = 0.06, 3.0
BAND_SEEDS, BAND_SIMS = (1, 2, 3, 4), 1 << 13


def _band(build, kwargs, name):
    """Both pricers over the same seeds, and the difference read TWO ways - relative, and in the
    seed-to-seed standard error the two of them carry between them.

    BOTH, NOT EITHER. The two readings were an OR - "a deal whose value is near zero has no
    meaningful relative reading and a deal with a big one has no meaningful absolute one, so passing
    either is passing" - which is a real asymmetry and the wrong conclusion drawn from it: an OR
    passes a deal that fails the reading that IS meaningful for it, because the other one was
    vacuous. Every product here passes both today, so the AND costs nothing and says something. If
    a future reading breaks one arm, the number is what gets reported - not the tolerance.
    """
    daily = np.array([_run(build(), seed=s, sims=BAND_SIMS)[0] for s in BAND_SEEDS])
    strided = np.array([_run(build(), seed=s, sims=BAND_SIMS, **kwargs)[0] for s in BAND_SEEDS])
    gap = abs(strided.mean() - daily.mean())
    se = math.sqrt(daily.var(ddof=1) + strided.var(ddof=1)) / math.sqrt(len(BAND_SEEDS))
    rel = gap / max(abs(daily.mean()), 1e-12)
    assert rel <= STRIDE_VS_DAILY_TOL and gap <= STRIDE_VS_DAILY_SE * se, (
        '{}: strided {:.4f} against day-stepped {:.4f} - {:.2%} apart (tol {:.0%}) and {:.2f} '
        'standard errors (tol {:.1f}), on a daily seed spread of {:.4f}'.format(
            name, strided.mean(), daily.mean(), rel, STRIDE_VS_DAILY_TOL,
            gap / max(se, 1e-30), STRIDE_VS_DAILY_SE, daily.std(ddof=1)))
    return daily.mean(), strided.mean(), rel, gap / max(se, 1e-30)


@pytest.mark.parametrize('kwargs,name', [(ON, 'crisp'), (SMOOTH, 'smooth')])
def test_the_strided_tarf_prices_within_the_day_stepped_pricers_own_band(kwargs, name):
    """THE ORACLE IS THE WALK IT REPLACES - the punchlist's own ruling, and the reason this is a
    band and not a tolerance: neither side is exact, so what is checked is that the difference sits
    inside the day-stepped pricer's own seed-to-seed spread."""
    print('\nTARF {}: daily {:.2f}  strided {:.2f}  {:.2%}  {:.2f} SE'.format(
        name, *_band(_tarf_cfg, kwargs, 'TARF ' + name)))


@pytest.mark.parametrize('build,name', [(_acc_cfg, 'accumulator'), (_autocall_cfg, 'autocall')])
def test_the_other_two_products_price_within_the_day_stepped_band(build, name):
    print('\n{}: daily {:.4f}  strided {:.4f}  {:.2%}  {:.2f} SE'.format(
        name, *_band(build, SMOOTH, name)))


def test_the_strided_discrete_barrier_prices_within_the_day_stepped_band():
    """THE FOURTH CONSUMER'S band, and the only one where the stride's contribution is PURELY the
    unmonitored sub-step: the monitored step is an OSS truncation at a constant barrier either way,
    identical expression, so the whole difference between the two columns is the carried state
    across 20 unmonitored days x three intervals.

    Crisp - `Branch_And_Weight` refuses under both Heston-Nandi families whatever this switch says,
    the discrete barrier's smooth arm being a GBM-only construction - so this reads consumer 3 with
    the estimator held fixed, which none of the other three do. Measured 0.08% and 0.22 standard
    errors on a daily seed spread of 13.9 currency units.
    """
    print('\ndiscrete barrier: daily {:.4f}  strided {:.4f}  {:.2%}  {:.2f} SE'.format(
        *_band(_barrier_cfg, ON, 'discrete barrier')))


def test_an_autocall_with_no_put_barrier_puts_the_two_arms_on_one_number():
    """With no second decision per fixing the smooth arm has nothing to integrate, so the crisp and
    smooth strided autocalls are the SAME estimator on the same draws - bit for bit, which is what
    says the put leg is the only thing the estimator switch changes here."""
    crisp, _ = _run(_autocall_cfg(put_barrier=0.0), **ON)
    smooth, _ = _run(_autocall_cfg(put_barrier=0.0), **SMOOTH)
    assert crisp == smooth, 'the arms differ with no decision between them'


# ======================================================================================
# THE SURVIVAL LEDGER, on the HN arm.
# ======================================================================================

def test_the_survival_ledger_conserves_on_a_strided_document():
    """The GBM arm's own gate pattern, on the stride. `sum_j alive_{j-1}*(1 - p_j) + alive_final`
    is the starting weight, per path, to roundoff - and it is exact ONLY because the fired mass is
    spelled `1 - p` off the same `p` the stride truncated with, which the saturation is part of."""
    buf, root = io.StringIO(), logging.getLogger()
    handler, old = logging.StreamHandler(buf), root.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    try:
        _run(_tarf_cfg(), **SMOOTH)
    finally:
        root.removeHandler(handler)
        root.setLevel(old)
    lines = [l for l in buf.getvalue().splitlines() if 'LEDGER TARF' in l]
    assert lines, 'the ledger logged nothing - the smooth arm did not run'
    worst = max(float(l.split('conservation=')[1].split()[0]) for l in lines)
    assert worst < 1e-12, 'the survival ledger lost mass: {:.3g}\n{}'.format(
        worst, '\n'.join(lines))


# ======================================================================================
# THE GAMMAS. The numbers that did not exist before this landing.
# ======================================================================================

#: THE GAMMA TABLE, measured on this workstation (RTX 3090, float64) at 2048 inner paths, seed 3,
#: under `Branch_And_Weight: 'Yes'` + `HN_Stride: 'Yes'` - the three-fixing component TARF and the
#: three-coupon component autocall with a put barrier 15% off the strike:
#:
#:                     AAD            CRN(best)      apart    ladder flat to
#:     TARF  delta     +1.63133e+06   +1.63350e+06   0.13%    2.76%
#:     TARF  gamma     -1.33197e+06   -1.32723e+06   0.36%    9.20%
#:     AC    delta     +8.128450      +8.126510      0.02%    0.39%
#:     AC    gamma     -0.2958720     -0.2964980     0.21%    3.30%
#:
#: The gamma ladder is a CRN bump of the reported ADJOINT (`Greeks: 'First'` at S+h and S-h), which
#: is this suite's route to a second derivative wherever the tape is the only Hessian in hand. Crisp
#: - and therefore for every HN exotic before this landing - there is no reading at all: the deal
#: registers a boundary correction and `Greeks: 'All'` is refused outright.
#:
#: THESE NUMBERS COST A SECOND NEWTON STEP. Before it the same two gammas read 5.08% and 36.00% out
#: on ladders flat to 9.2% and 3.3% - a stable disagreement, not noise - because the stride's
#: gradient reattachment was exact at first order and 3.1% out at second
#: (`utils.hn_component_stride_draw`). Every delta above was already exact then.
#:
#: THE FLATNESS TOLERANCE IS 0.10 AND WAS 0.12, which is 0.12 chosen to clear a 9.20% reading by
#: 2.8 points rather than by a stated margin. Both ladders are flat to better than 0.10 (9.20% and
#: 3.30% recorded), so the looser number bought nothing and cost something: `crn_ladder` warns on
#: its OWN spread, and a flat_tol wide enough to swallow that warning suppresses the one signal
#: that says a ladder is scattering rather than converging.
GAMMA_TOL, GAMMA_FLAT = 0.05, 0.10


def test_greeks_all_flows_under_the_switch_and_is_refused_without_it():
    """The gate the whole landing is for. The smooth estimator removes the decisions, so the deal
    registers no `BoundarySet` and the full Hessian flows; crisp, the same document is refused."""
    _, out = _run(_tarf_cfg(), Greeks='All', **SMOOTH)
    assert 'Greeks_Second' in out['Results'], 'no second-order block came back'
    assert np.isfinite(_second(out, 'FxRate.ZAR'))
    with pytest.raises(utils.SecondOrderRefused) as refusal:
        _run(_tarf_cfg(), Greeks='All')
    assert 'boundary correction' in str(refusal.value)


def test_the_tarf_gamma_lands_on_its_crn_ladder():
    """THE HEADLINE. Delta against a CRN bump of the value, gamma against a CRN bump of the reported
    adjoint - agreement AND flatness, because a reading off a scattering ladder is not a derivative
    however close it lands."""
    _, out = _run(_tarf_cfg(), Greeks='All', **SMOOTH)
    aad_d, aad_g = _first(out, 'FxRate.ZAR'), _second(out, 'FxRate.ZAR')
    rungs = (2e-3, 5e-3, 1e-2, 2e-2)
    lad_d = ladder(price=lambda s: _run(_tarf_cfg(spot=s), **SMOOTH)[0],
                   aad=aad_d, base=hnc.TARF_SPOT, rungs=rungs)
    assert lad_d.agrees(tol=0.05, flat_tol=0.10), 'TARF DELTA\n' + str(lad_d)
    lad_g = ladder(price=lambda s: _first(_run(_tarf_cfg(spot=s), Greeks='First', **SMOOTH)[1],
                                          'FxRate.ZAR'),
                   aad=aad_g, base=hnc.TARF_SPOT, rungs=rungs)
    assert lad_g.agrees(tol=GAMMA_TOL, flat_tol=GAMMA_FLAT), 'TARF GAMMA\n' + str(lad_g)


def test_the_autocall_gamma_lands_on_its_crn_ladder():
    """The same table on the second product, whose put leg is the deal's SECOND decision per fixing
    and whose coupon trigger is the first."""
    _, out = _run(_autocall_cfg(), Greeks='All', **SMOOTH)
    aad_d, aad_g = _first(out, 'EquityPrice.EQ'), _second(out, 'EquityPrice.EQ')
    rungs = (2e-3, 5e-3, 1e-2, 2e-2)
    lad_d = ladder(price=lambda s: _run(_autocall_cfg(spot=s), **SMOOTH)[0],
                   aad=aad_d, base=AC_SPOT, rungs=rungs)
    assert lad_d.agrees(tol=0.05, flat_tol=0.10), 'AUTOCALL DELTA\n' + str(lad_d)
    lad_g = ladder(price=lambda s: _first(_run(_autocall_cfg(spot=s), Greeks='First', **SMOOTH)[1],
                                          'EquityPrice.EQ'),
                   aad=aad_g, base=AC_SPOT, rungs=rungs)
    assert lad_g.agrees(tol=GAMMA_TOL, flat_tol=GAMMA_FLAT), 'AUTOCALL GAMMA\n' + str(lad_g)


# ======================================================================================
# CONSUMER 2 on a document: the crisp value untouched, the crisp DELTA fixed.
# ======================================================================================

def test_the_mixture_leaves_the_crisp_value_alone_and_lands_its_delta():
    """The autocall's put leg, crisp. Its breach is a bare jump of `rebate - (1 - B/strike)`, which
    pathwise AAD differentiates as if it were not there - so the reported delta misses its own bump
    ladder. The mixture splices the analytic conditional expectation into the derivative alone: the
    value is the sampled indicator's, and the ladder closes."""
    v_plain, _ = _run(_autocall_cfg(), **ON)
    _, out = _run(_autocall_cfg(), Greeks='First', **ON)
    v_greek, aad = _run(_autocall_cfg(), **ON)[0], _first(out, 'EquityPrice.EQ')
    assert v_plain == v_greek, 'asking for greeks moved the value'
    lad = ladder(price=lambda s: _run(_autocall_cfg(spot=s), **ON)[0],
                 aad=aad, base=AC_SPOT, rungs=(2e-3, 5e-3, 1e-2, 2e-2))
    assert lad.agrees(tol=0.06, flat_tol=0.12), 'AUTOCALL CRISP DELTA\n' + str(lad)


def _registered(build, **calc):
    """The BoundarySet type list one priced document left behind, off the registration site's own
    DEBUG line. A refusal names DEALS and never registrations, and no reported frame carries them,
    so this is the only place the supersession is readable - the same capture the survival-ledger
    gate runs one section down."""
    buf, root = io.StringIO(), logging.getLogger()
    handler, old = logging.StreamHandler(buf), root.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    try:
        _run(build, Greeks='First', **calc)
    finally:
        root.removeHandler(handler)
        root.setLevel(old)
    lines = [l for l in buf.getvalue().splitlines() if 'boundary sets:' in l]
    assert len(lines) == 1, 'expected one TARF registration line, got {}'.format(lines)
    return ast.literal_eval(lines[0].split('boundary sets: ')[1])


def test_the_kernel_flux_registration_is_superseded_not_joined():
    """ONE ESTIMATOR PER DECISION. Where the mixture takes the TARF's knock-IN it must REMOVE that
    decision's `InnerBoundarySet`, or the flux is counted twice - once analytically and once by the
    kernel.

    READ ON THE STRIDED-WITH-BARRIER PATH, which is the only path the guard is on. The gate this
    replaces asserted a second-order refusal on `barrier=0.0` under the stride - a document with no
    knock-in in it at all, so the guard the supersession lives in
    (`if barrier > 0.0 and otm_analytic is None`) was never reached and the reading was the LATCH's,
    unchanged, whatever the mixture did. Deleting the supersession clause from that guard left the
    old gate green.

    So: the TYPE LIST, on a live barrier. Crisp registers the redemption latch AND the knock-in's
    inner set; strided, the mixture takes the knock-in's decision and only the latch is left - the
    latch deciding on the accrual of OBSERVED fixings, which has no conditioning step to integrate
    against. A mixture that JOINED rather than superseded shows both, and that is the mutation.
    """
    crisp = _registered(_tarf_cfg())
    assert crisp == ['LatchedBoundarySet', 'InnerBoundarySet'], (
        'the crisp TARF no longer registers both decisions: {}'.format(crisp))
    strided = _registered(_tarf_cfg(), **ON)
    assert strided == ['LatchedBoundarySet'], (
        'the conditional-p mixture did not supersede the knock-in\'s kernel flux - it registered '
        '{}, so the same decision is counted twice'.format(strided))
    # the knock-in is what moved, not the latch: taken out of the document, crisp and strided are
    # the same one registration, which is what says the supersession removed the RIGHT one
    assert _registered(_tarf_cfg(barrier=0.0)) == ['LatchedBoundarySet']
    assert _registered(_tarf_cfg(barrier=0.0), **ON) == ['LatchedBoundarySet']
    # crisp, the pair is still refused at second order and the deal is named
    with pytest.raises(utils.SecondOrderRefused) as refusal:
        _run(_tarf_cfg(), Greeks='All')
    assert 'TARF1' in str(refusal.value), str(refusal.value)
    # and the FIRST-order reading is untouched by the supersession, the removed registration's own
    # contribution on this document being an exact zero
    plain = _first(_run(_tarf_cfg(), Greeks='First')[1], 'FxRate.ZAR')
    assert np.isfinite(plain)


# ======================================================================================
# THE MUTATIONS, each dying by a named gate.
# ======================================================================================

def test_the_daily_gaussian_p_is_the_wrong_regime_at_monthly_fixings():
    """THE MUTATION THE DESIGN NAMES. Under Heston-Nandi the walk is daily, so the only Gaussian
    conditional in hand is the LAST daily sub-step - and using it as the fixing interval's `p` is a
    wrong number wearing the right estimator's name. Measured on the fixture's own law at k = 21:
    the daily Gaussian's survival probability against the k-step law's own."""
    kit = _kit()
    fix = _fixing(kit, n=21)
    B = hnc.TARF_SPOT * 1.05
    stride_p = float(kit.stride_cdf(fix, fix.strip, kit.stride_bound(fix, _t(B))))
    h = fix.h.reshape(-1)[0]
    daily_p = float(utils.norm_cdf(
        (torch.log(_t(B) / fix.spot.reshape(-1)[0]) + 0.5 * h) / torch.sqrt(h)))
    # the daily conditional says a 5% move is impossible in one day, which it very nearly is; the
    # k-step law says it is a real chance over the month the fixing actually spans
    assert 1.0 - daily_p < 1.0e-8, (
        'the daily conditional is no longer degenerate at this bound: {:.12g}'.format(daily_p))
    assert 0.02 < stride_p < 0.99, 'the k-step law has no decision left in it: {}'.format(stride_p)
    assert abs(daily_p - stride_p) > 0.05, (
        'the two regimes are no longer distinguishable: daily {:.6g} against the k-step law\'s '
        '{:.6g}'.format(daily_p, stride_p))


def test_the_tilt_is_verified_on_a_contour_its_bound_was_not_resolved_on():
    """The Esscher tilt inherits the strip's bound, and that is sound BY CONSTRUCTION only for the
    two contours the bound scan reads - `i phi` and `i phi + 1`. A third is VERIFIED at the top
    node rather than assumed.

    THE INVERTED ACCRUAL IS THE THIRD CONTOUR THAT MATTERS: `pv_MC_Tarf`'s `InvertedTarget` writes
    its gain in `1/S` and asks for `a = -1`. Measured at k = 21 on the fixture's law, the strip's
    own bound leaves that contour at -25.4 against the share measure's -25.5, so it rides the same
    cache - which is the reading that says an inverted TARF is priced rather than refused.

    AND THE CHECK CATCHES A NaN, not merely a large number: past the real MGF's own radius the
    recursion returns NaN, and NaN fails every comparison - including, if the test is written the
    obvious way round, the one meant to catch it.
    """
    kit = _kit()
    fix = _fixing(kit, n=21)
    for power in (1.0, -1.0):
        tilt, _, _, _ = kit.stride_tilt(fix, power)
        top = float((tilt.A[..., -1] + tilt.B[..., -1] * fix.h
                     + tilt.C[..., -1] * fix.q).real.max()) - math.log(tilt.phi_max)
        assert top <= pricing.HN_STRIDE_PHI_MIN_DECAY, (
            'the tilt at {} does not decay on this strip: {:.4g}'.format(power, top))
    with pytest.raises(ValueError) as refusal:
        kit.stride_tilt(fix, 100.0)                        # past the real MGF's own radius
    assert 'HN_Stride' in str(refusal.value) and 'decayed' in str(refusal.value)


#: THE STATE BOX A MULTI-ROW REPORT REACHES. One row at the seed state and one that has drifted to
#: the model's own variance floor - which is what a second MTM row of a CVA profile is, and what
#: `stride`'s own box test exists for. On the fixture at k = 21 the seed box resolves phi_max 256
#: and this one resolves 512, so the strip REBUILDS.
WIDE_BOX = (1.0e-12, 3.0e-4)


def _wide(kit, spot=hnc.TARF_SPOT, n=21):
    """The same interval opened on a two-row state box the first strip's bound was NOT resolved on -
    written as a state, not as a patched cache."""
    lo, hi = WIDE_BOX
    col = torch.tensor([[lo], [hi]], dtype=DT)
    return kit.stride_interval(torch.full((2, 1), float(spot), dtype=DT),
                               (col, col.clone(), 0), torch.zeros((2, 1), dtype=DT), n)


def test_the_tilt_cache_is_keyed_on_the_strip_it_is_assembled_onto():
    """THE CACHE COHERENCE THE LANDING WAS BLOCKED ON.

    `stride` REBUILDS its strip whenever the cube's state box has left the one the quadrature bound
    was resolved on - that is the punchlist's own trap, taken at the consumer - and `stride_tilt`
    assembles ONTO that strip's nodes. Keyed on calendar position alone, the two caches disagree:
    the rebuilt strip is served the OLD strip's tilt, silently, on a different quadrature grid.

    UNREACHABLE TODAY AND NOT FOR LONG. `Base_Revaluation` prices one report row, so the box never
    widens under the cache; the moment the switch reaches multi-row CVA it does, every profile.

    Measured on the fixture at k = 21: the seed row resolves phi_max 256, the row at the floor
    resolves 512, and the stale tilt puts the floored row's partial moment 1.04e-3 out - forced at
    the seam, so the number is the construction's and not a run's.
    """
    kit = _kit()
    fix1 = _fixing(kit, n=21)
    tilt1, _, _, _ = kit.stride_tilt(fix1, 1.0)
    fix2 = _wide(kit)
    assert fix2.strip is not fix1.strip, (
        'the state box no longer forces a rebuild - this gate has stopped testing anything')
    assert fix2.strip.phi_max != fix1.strip.phi_max, (
        'the rebuild landed on the same bound: {} against {}'.format(
            fix1.strip.phi_max, fix2.strip.phi_max))

    tilt2, _, _, _ = kit.stride_tilt(fix2, 1.0)
    assert tilt2 is not tilt1, 'the rebuilt strip was served the old strip\'s tilt'
    assert tilt2.phi_max == fix2.strip.phi_max, (
        'the tilt carries a bound its strip does not: {} against {}'.format(
            tilt2.phi_max, fix2.strip.phi_max))
    assert torch.equal(tilt2.nodes, fix2.strip.nodes), (
        'the tilt is assembled onto a different quadrature grid than the strip it inverts with')

    # and what the incoherence COSTS, on the partial moment a fired branch closes in: the same call
    # with the cache emptied, which is the answer by construction
    stale = kit.stride_partial_moment(fix2, kit.stride_bound(fix2, _t(hnc.TARF_SPOT * 1.02)),
                                      True, 1.0)
    kit._tilts.clear()
    fresh = kit.stride_partial_moment(fix2, kit.stride_bound(fix2, _t(hnc.TARF_SPOT * 1.02)),
                                      True, 1.0)
    assert torch.equal(stale, fresh), (
        'the cached tilt and a freshly built one disagree by {:.3g} relative - the cache is not '
        'keyed on what it holds'.format(
            float(((stale - fresh) / fresh).abs().max())))


def test_the_tilt_decay_check_re_runs_on_a_cache_hit():
    """THE OTHER HALF, and it is not covered by keying the cache. The refusal `stride_tilt` carries
    for a third contour reads the STATE BOX, which widens under the caller whether or not the
    quadrature bound moved - so a check that ran once on the seed state is a claim about a box this
    fixing is not in, and a cache hit that skips it is a refusal that has been cached away.

    Forced at the seam: the SAME strip (so a guaranteed cache hit) carried on a state the model's
    own variance floor exists to keep it off. Measured on the fixture at k = 21, the a = -1 contour
    leaves the seed box at -25.4 - which is what the first call verified - and the same strip at an
    unfloored 1e-12 leaves it at -6.4, past the -10 that is the line the bound scan itself calls a
    probability. The floor is exactly what stands between a live deal and this state
    (`stride_interval` clamps), which is why it is written here rather than reached.
    """
    kit = _kit()
    fix = _fixing(kit, n=21)
    kit.stride_tilt(fix, -1.0)                 # passes at -25.4 and caches
    assert kit._tilts, 'the tilt did not cache at all'
    # the same strip, hence the same key, on a state box the check was never run against
    floorless = fix._replace(h=torch.full_like(fix.h, 1.0e-12),
                             q=torch.full_like(fix.q, 1.0e-12))
    assert floorless.strip is fix.strip, 'the mutation changed the strip, not the box'
    with pytest.raises(ValueError) as refusal:
        kit.stride_tilt(floorless, -1.0)
    assert 'HN_Stride' in str(refusal.value) and 'decayed' in str(refusal.value), str(refusal.value)
    # the refusal names the box it refused on, which is the half a cached check cannot say
    assert '1e-12' in str(refusal.value) or '1.0000e-12' in str(refusal.value), str(refusal.value)


# ======================================================================================
# THE SPEED CLAIM, measured and REFUTED.
# ======================================================================================

def test_the_speed_lever_is_measured_and_it_is_not_one():
    """CONSUMER 3, and the report says what it found: IT IS NOT A SPEED LEVER. The claim is
    REFUTED, not merely unproven - measured three times, on two occasions by a second reader, and
    the sign never came close to turning.

    Measured on this workstation (RTX 3090, float64), the three-fixing component TARF, best of two
    runs at each size:

        inner paths     daily      strided     ratio
             1,024      0.027s      3.007s      111x
             4,096      0.025s      3.075s      121x
            16,384      0.027s      3.392s      124x
            32,768      0.031s      4.540s      147x

    THE RATIO WIDENS WITH THE CUBE. That is the whole finding: there is no crossover to reach by
    widening, because the thing that would have to close is going the other way.

    THE DIAGNOSIS IS NOT THE DRAW COUNT (`pricing.stride_normals` states it as one: three normals a
    stride against `n_steps` a walk). A daily step is a handful of cheap ELEMENTWISE operations over
    the whole cube at once, so the walk is flat - 0.027s to 0.031s across a 32x range of paths, and
    the arithmetic is free beside the launch. The stride pays two things the walk does not: a FIXED
    per-interval cache build (3.0s for three intervals, most of it the bound scan's kernel launches
    - it starts at phi = 8 and doubles, on a 4-element state, each rung a 21-step recursion of tiny
    tensor operations), and then a Gil-Pelaez inversion PER PATH at every fixing, which is a
    (paths x node) complex reduction and is the 4.8e-5 s a path the last column is growing on.

    THE OPEN LEVER IS A BATCHED PHI: one inversion across the whole cube rather than a reduction the
    path axis rides through, and the B and C strips reused across intervals of equal length (which
    the stride's own calendar-anchoring note says is sound - omega reaches A alone). Neither is
    built, and until they are this switch is not bought for time.

    WHAT STAYS. All of it. The stepping is the SMOOTH ESTIMATOR's own conditioning law - `p` over
    the fixing interval, the fired branch's partial moments, the survival-truncated draw - and the
    truncated draws are what branch-and-weight IS. The speed claim died; the machinery it was
    written under is load-bearing for the other two consumers and for the gammas at the top of this
    file. Only the claim is deleted.

    PRINTED rather than asserted past a sanity bound - this workstation shares its GPU with the rest
    of the suite.
    """
    times = {}
    for name, kw in (('daily', {}), ('stride', ON)):
        t = time.time()
        _run(_tarf_cfg(), **kw)
        times[name] = time.time() - t
    print('\nSTRIDE WALL CLOCK, 3 fixings x {} inner paths: daily {:.2f}s, strided {:.2f}s '
          '({:.1f}x)'.format(SIMS, times['daily'], times['stride'],
                             times['stride'] / max(times['daily'], 1e-9)))
    assert times['stride'] < 120.0, 'the strided pricer is not merely slower, it has stalled'
