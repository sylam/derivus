"""Branch-and-weight, the base valuation's smooth estimator, on real documents: TARF, accumulator,
discrete barrier and autocall.

The construction: at each fixing the trigger standardises to `zB`; the FIRED branch closes
analytically with weight `1 - Phi(zB)` and payoff `E[J(S_k) | fired]`, the CONTINUING branch draws
`S_k` from the truncated law by `Phi^-1(U * Phi(zB))` carrying weight `Phi(zB)`. Unbiased, not a
smoothing: ten seeds at 16384 inner paths read crisp -37.5296 (sd 0.3208) and smooth -37.4098 (sd
0.0910) against the quadrature's -37.359382, a variance ratio of 12.42. Second-order variance in the
conditioning step scales like `s_k**-3`: over twelve fixings at 8192 paths the gamma's seed spread
is 1.70% monthly and 21.15% daily, where the value's falls from 1.13% to 0.35%.

Each product is held against a differentiable trapezoid reference written here out of
`math`/`torch` alone, so an error in the closed form cannot hide behind the same error in its
oracle. The primitives (`oss_truncated_draw`, `lognormal_partial_moment`, `lognormal_fired_gain`,
`SurvivalLedger`, `accrual_kink_term`) are read through those documents.
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
from derivus import utils
from derivus.instruments import construct_instrument

# the barrier's world and deal, borrowed from the file that already gates them
import test_barrier_bridge as bb
from crn_ladder import ladder

TARF_TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'fixtures', 'fx_tarf_job.json')

DT = torch.float64


def _Phi(x):
    """The standard normal CDF, from `math.erf` - not `utils.norm_cdf`, which is what is on test."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _phi(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


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


def _crisp(job):
    """The crisp estimator DECLARED. The default is the switch, so a gate about the crisp path has
    to say so; an undeclared document is the smooth one."""
    return _smooth(job, on=False)


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
# THE DEFAULT IS THE SWITCH, and what the switch is attributable for
# ======================================================================================

@pytest.mark.parametrize('name', ['tarf', 'autocall'])
def test_the_default_is_the_switch_on_a_real_document(name, tmp_path):
    """The key ABSENT and the key written 'Yes' are the same run - value, the whole reported mtm
    frame and every first-order greek, by `np.array_equal` - on a knock-in TARF and an autocall with
    a jumping put barrier, the two documents on which 'No' legitimately moves the number.

    Killing mutation: `Branch_And_Weight` declared with default 'No'.
    """
    def build():
        return (_tarf_doc(greeks='First') if name == 'tarf' else
                _autocall_doc(0.7, greeks='First'))

    absent, out_a, _ = _run_doc(build(), tmp_path, name + '_absent')
    written, out_w, _ = _run_doc(_smooth(build()), tmp_path, name + '_yes')
    assert absent == written, (absent, written)
    # `DataFrame.equals` rather than `array_equal`: the frame carries an all-NaN root row, and NaN
    # is not equal to itself
    assert out_a['Results']['mtm'].equals(out_w['Results']['mtm']), 'the reported frame moved'
    frame_a, frame_w = out_a['Results']['Greeks_First'], out_w['Results']['Greeks_First']
    assert np.array_equal(frame_a.values, frame_w.values), 'a first-order greek moved'


def test_a_tarf_with_no_knock_in_is_bit_identical_under_the_switch(tmp_path):
    """ATTRIBUTION. The one-step-survival loop was ALREADY the smooth estimator for the target -
    the KO-in-step term is the fired branch integrated against the interval's law, the continuation
    the truncated draw - so with the knock-in off the two estimators agree TO THE BIT. With it on
    they part by ~1.3% at 65536 paths, that leg being the one indicator the crisp TARF sampled.

    Killing mutation: the integrated-leg guard losing its `otm_analytic is not None`, so the smooth
    arm zeroes an unbarriered OTM leg.
    """
    crisp, _, _ = _run_doc(_crisp(_tarf_doc(barrier=0.0)), tmp_path, 'nokick_off')
    smooth, _, _ = _run_doc(_smooth(_tarf_doc(barrier=0.0)), tmp_path, 'nokick_on')
    assert crisp == smooth, (crisp, smooth)


# ======================================================================================
# THE TARF
# ======================================================================================

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
    scalar vol and the report spreads it over four live knots. The same document under the CRISP
    estimator registers the knock-in and refuses `'All'` by name - the pathwise estimator it would
    otherwise report reads delta 0.714x, gamma 0.301x and vanna -0.341x of this table.

    Killing mutation: the TARF's per-fixing `accrual_kink_term` never built.
    """
    with pytest.raises(utils.SecondOrderRefused) as refusal:
        _run_doc(_crisp(_tarf_doc(greeks='All')), tmp_path, 'all_off')
    assert 'boundary correction' in str(refusal.value)

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


@pytest.mark.parametrize('target', [TARGET, 0.01])
def test_the_filling_fixing_pays_the_remaining_target_under_both_estimators(target, tmp_path):
    """The fixing that fills the target pays the remaining target `R` - measurable one fixing back,
    so `(1 - p) * R` IS its conditional expectation - and the SAME document under both estimators
    lands on the SAME quadrature, whose fired branch pays `N1 * r` and nothing else. At a target of
    0.01 the first of two fixings fires on ~46% of paths: three seeds at 65536 read smooth -54.7124
    and crisp -54.8276 against -54.6691.

    Killing mutation: the knocked-out weight's `(1 - p) * L * R * Dj` dropped from the fixing that
    fills the target.
    """
    truth = _table(_tarf_reference, target=target)['value']
    for label, estimator in (('crisp', _crisp), ('smooth', _smooth)):
        got, _, _ = _run_doc(estimator(_tarf_doc(target=target, sims=1 << 18)), tmp_path, label)
        assert _rel(got, truth) < 0.01, (
            'the {} estimator reads {:.6f} against the remaining-target quadrature {:.6f} '
            '({:.3%})'.format(label, got, truth, _rel(got, truth)))


@pytest.mark.parametrize('name', ['tarf', 'autocall'])
def test_the_ledger_conserves_on_a_real_document(name, tmp_path):
    """CONSERVATION on a live run: per path, the mass fired plus the mass alive is the mass the
    strip opened with. Measured 1.11e-16 on the TARF with 59.0% surviving both fixings and 0.0 on
    the autocall with 37.6% surviving both coupons - a ledger that fired everything or nothing would
    conserve trivially.

    Killing mutation: `SurvivalLedger.fire` booking `alive * (1 - p)` a part in a billion short.
    """
    job = (_tarf_doc(sims=1 << 12) if name == 'tarf' else _autocall_doc(0.7, sims=1 << 12))
    _, _, log = _run_doc(_smooth(job), tmp_path, 'ledger', debug=True)
    lines = [ln for ln in log.splitlines() if 'LEDGER ' + name.upper() in ln]
    assert lines, 'the smooth path ran no ledger at all'
    residual = max(float(ln.split('conservation=')[1].split()[0]) for ln in lines)
    alive = [float(ln.split('alive=')[1].split()[0]) for ln in lines]
    assert residual < 1e-12, (residual, lines)
    assert 0.05 < max(alive) < 0.95, (
        'this document fires on almost none or almost all of its weight, so conservation on it '
        'says nothing: alive={}'.format(alive))


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

    Killing mutation: the accumulator's `accrual_kink_term` never built.
    """
    ref = _table(_acc_reference)
    crisp, out_c, _ = _run_doc(_crisp(_acc_doc(greeks='All')), tmp_path, 'acc_off')
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


@pytest.mark.parametrize('name', ['accumulator', 'autocall'])
def test_a_registration_the_switch_supersedes_is_not_lost(name, tmp_path):
    """A decision dated ON the base date is resolved off the simulated spot, so its latch has a
    graph-carrying gap and `Greeks: 'All'` refuses under the crisp estimator - an accumulator fixing
    or an autocall coupon. Under the switch the registration is skipped and the Hessian flows, and
    first order comes back BIT FOR BIT: base valuation has one scenario, a one-sample gap supports no
    local-linear fit, and the correction is exactly zero by construction.

    Killing mutation: the TARF/accumulator/autocall `boundary_aad` taken whatever the switch says,
    so the smooth run registers and refuses.
    """
    def doc(greeks):
        return (_acc_doc(same_day=True, greeks=greeks) if name == 'accumulator' else
                _autocall_doc(same_day=True, greeks=greeks))

    with pytest.raises(utils.SecondOrderRefused):
        _run_doc(_crisp(doc('All')), tmp_path, 'sd_off')
    _, out, _ = _run_doc(_smooth(doc('All')), tmp_path, 'sd_on')
    spot = 'FxRate.EUR' if name == 'accumulator' else 'EquityPrice.EQ'
    vol = 'FXVol.EUR.USD' if name == 'accumulator' else 'EquityPriceVol.EQ'
    assert np.isfinite(_second(out, spot=spot, vol=vol)[0])
    crisp, out_c, _ = _run_doc(_crisp(doc('First')), tmp_path, 'sd1_off')
    smooth, out_s, _ = _run_doc(_smooth(doc('First')), tmp_path, 'sd1_on')
    assert crisp == smooth
    assert np.array_equal(out_c['Results']['Greeks_First'].values,
                          out_s['Results']['Greeks_First'].values), (
        'the skipped registration was carrying flux after all - it is not zero here, and the '
        'supersession is losing a derivative rather than replacing an estimator')


# ======================================================================================
# THE DISCRETE BARRIER
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
# knock directions, the rebate's leg, and the binary sibling.
BARRIER_FAMILY = [
    ('down-and-out call', 90.0, {}, False),
    ('down-and-in call', 90.0, {'Barrier_Type': 'Down_And_In'}, False),
    ('knock-out with a rebate', 90.0, {'Cash_Rebate': 5.0}, False),
    ('binary down-and-out', 90.0, {}, True),
]


@pytest.mark.parametrize('barrier,over,digital', [row[1:] for row in BARRIER_FAMILY],
                         ids=[row[0] for row in BARRIER_FAMILY])
def test_off_is_off_across_the_barrier_family(barrier, over, digital):
    """OFF IS OFF, and ON IS OFF TOO. This pricer's OSS sampler was the smooth estimator already, so
    the value and the whole first-order frame are BIT-IDENTICAL with the switch absent, 'No' and
    'Yes' - three states, a field read through a default being a different code path from a field
    read. The crisp latch registration the switch skips loses nothing: base valuation has one
    scenario, so `boundary_weights` returns its empty-kernel branch and the correction is exactly
    zero. Knock-out delta 0.6597964784, knock-in -0.1100582535, binary 2.259431076.

    Killing mutation: `boundary_weights` taking `std()` of a one-sample gap, whose NaN reaches the
    crisp run's backward.
    """
    absent, out_a = _barrier_run(barrier, greeks='First', on=None, sims=1 << 12, digital=digital,
                                 **over)
    off, out_o = _barrier_run(barrier, greeks='First', on=False, sims=1 << 12, digital=digital,
                              **over)
    smooth, out_s = _barrier_run(barrier, greeks='First', on=True, sims=1 << 12, digital=digital,
                                 **over)
    assert absent == smooth == off, (absent, off, smooth)
    for out in (out_o, out_s):
        assert np.array_equal(out_a['Results']['Greeks_First'].values,
                              out['Results']['Greeks_First'].values), 'a first-order greek moved'


@pytest.mark.parametrize('rebate', [0.0, 5.0, 12.5])
def test_in_out_parity_is_the_survival_ledger_read_off_two_prices(rebate):
    """CONSERVATION on real documents, through prices instead of internals. In a zero-rate world
    the knock-out pays the rebate `sum_j fired_j` times and the knock-in's parity leg `alive_T`
    times, so

        KO + KI == vanilla + rebate

    exactly when `sum_j alive_{j-1} * (1 - p_j) + alive_final == 1`. A lost mass shows up here as
    an error proportional to the rebate and in neither price alone, which is why the rebate is
    walked: at 0 this is plain parity. Residuals 1.78e-15 / 1.78e-15 / 0.00e+00 at 32768 paths.

    Killing mutation: the knock-out's fired rebate weight `(1 - p) * L` a part in a billion short.
    """
    ko, _ = _barrier_run(90.0, on=True, Barrier_Type='Down_And_Out', Cash_Rebate=rebate)
    ki, _ = _barrier_run(90.0, on=True, Barrier_Type='Down_And_In', Cash_Rebate=rebate)
    black = _black_call()[0]
    assert abs(ko + ki - black - rebate) < 1e-11, (
        'in-out parity fails by {:.3e} at a rebate of {}: KO {:.12f} + KI {:.12f} against a '
        'vanilla of {:.12f}'.format(ko + ki - black - rebate, rebate, ko, ki, black))


def test_parity_survives_to_second_order_on_the_smooth_path():
    """The same identity at DELTA and GAMMA - a statement about WHERE the kink term went.
    `accrual_kink_term` is added to `surv_payoff`, before the in-out parity subtraction, so both
    legs carry one corrected quantity and `KO + KI` telescopes at every order.

    16384 paths, both barrier sides, against Black 9.9476449660 / 0.5497382248 / 0.0158335075:
    residuals 0.0e+00 at value and delta, 0.0e+00 and 3.5e-18 at gamma. The ladder gate below
    passes the same mutant, the leg it reads still carrying the term.

    Killing mutation: the terminal kink term added to the knock-out leg alone
    (`direction == BARRIER_OUT`), which leaves gamma 0.00108 short of Black.
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
    """GAMMA FLOWS on a LIVE barrier, and it is the derivative of the delta reported; the crisp
    estimator registers the observed crossing and refuses. A live knock-out has no closed form, so
    the reading is a CRN ladder of the corrected delta - 32768 paths: gamma 0.00969411 against a
    ladder best of 0.00974971, 0.57% at 2.12% flatness. The never-knocking control (barrier 1.0) is
    a European: value 10.0082 against Black 9.9476, gamma 0.01560843 against 0.01583351.

    Killing mutation: the terminal kink term never built - the live barrier reads 0.00861379, 13.19%
    off its unchanged ladder, every barrier decision being a `Phi` whose curvature AAD carries.
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
        _eq_second(european), black_gamma)


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
# THE AUTOCALL - the put leg, its tables, and the arm the switch does not reach
# ======================================================================================

def test_the_crisp_put_leg_lands_on_its_ladder(tmp_path):
    """The crisp estimator's put leg used to miss its own ladder wherever the payoff jumps - 16.2%
    at a 70% barrier. The GBM arm splices the put leg's conditional `p` into the crisp value (one
    estimator per decision), so the crisp delta lands on a CRN ladder of the crisp value: 2.66% at
    65536 paths, the crisp ladder's own flip-counting noise. The smooth estimator is held against
    the quadrature tables below.

    Killing mutation: `splice_conditional_p` contributing nothing, so the crisp delta is the
    indicator's.
    """
    def job(**kw):
        return _crisp(_autocall_doc(0.7, sims=1 << 16, **kw))

    aad = _first(_run_doc(job(greeks='First'), tmp_path, 'k_d')[1], factor='EquityPrice.EQ')
    rung = ladder(price=lambda s: _run_doc(job(spot=s), tmp_path, 'k_v')[0],
                  aad=aad, base=AUTOCALL_SPOT, rungs=(1e-3, 2e-3, 5e-3, 1e-2))
    miss = abs(rung.best - rung.aad) / max(abs(rung.aad), 1e-30)
    assert miss < 0.05, (
        'the CRISP autocall put leg misses its own bump ladder ({:.2%}) - the conditional-p splice '
        'no longer carries the jump\n{}'.format(miss, rung))


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
    +0.2568038 / +0.03403085 / -3.363476 / -0.001696697 / +0.07967986. The rebate rides inside the
    effective strike `strike * (1 - rebate)`. The vega row is what says the vol identification is
    right before vanna means anything.

    Killing mutation: the put leg's interval carrying `L` where it carries `L / p` - the conditional
    expectation's `1/p` dropped, which reads +0.254135 against +0.2210455 (15.0%).
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

    Killing mutation: the put leg's bound read at the barrier alone, `min(B, K)` -> `B`.
    """
    for put_barrier in (1.0, 1.2):
        ref = _ac_table(put_barrier=put_barrier)
        got = _ac_run(_smooth(_autocall_doc(put_barrier, greeks='All', sims=1 << 18)),
                      tmp_path, 'ac_above')
        for key, tol in (('value', 0.01), ('delta', 0.01), ('gamma', 0.01)):
            assert _rel(got[key], ref[key]) < tol, (
                'put barrier {}: {} reads {:.8g} against {:.8g} ({:.3%})'.format(
                    put_barrier, key, got[key], ref[key], _rel(got[key], ref[key])))


def test_an_observed_breach_is_data_and_the_switch_does_not_touch_it(tmp_path):
    """WHAT THE SWITCH DOES NOT SMOOTH. A barrier date on an OBSERVED coupon has no conditioning
    step: `Sj` is the scenario's own spot, so the breach is DATA and its indicator stays exact. The
    document puts the deal's only barrier date on the base date's coupon, spot BELOW the strike and
    barrier ABOVE it, so the breach pays a real -0.10 per unit rather than a vacuous zero. Under the
    switch that leg is untouched and the run is BIT-IDENTICAL at value and first order.

    Killing mutation: the observed breach read through a sigmoid under the switch.
    """
    def job(**kw):
        return _barrier_on_the_base_date(
            _autocall_doc(1.1, same_day=True, spot=90.0, greeks='First', **kw))

    crisp, out_c, _ = _run_doc(_crisp(job()), tmp_path, 'ac_obs_off')
    smooth, out_s, _ = _run_doc(_smooth(job()), tmp_path, 'ac_obs_on')
    assert crisp == smooth, (crisp, smooth)
    assert np.array_equal(out_c['Results']['Greeks_First'].values,
                          out_s['Results']['Greeks_First'].values)
    # and the leg is live rather than a zero agreeing with a zero
    without, _, _ = _run_doc(_smooth(_autocall_doc(
        0.0, same_day=True, spot=90.0, greeks='First')), tmp_path, 'ac_obs_none')
    assert _rel(crisp, without) > 0.05, (crisp, without)


@pytest.mark.parametrize('barrier', [True, False],
                         ids=['a barrier on the row', 'nothing on the row'])
def test_a_zero_coupon_row_refuses_by_name(barrier, tmp_path):
    """A row quoted ZERO runs no coupon block on the fast arm, so the NEXT coupon takes this row's
    interval (0.466196 against 0.487692, 4.41%) and a barrier dated on the row reads the PREVIOUS
    fixing's spot (+0.394809 against +0.317939, 24.2%). `calc_dependencies` refuses the document by
    name and FATALLY; the same document with a real coupon still prices on the same arm, so what is
    refused is the zero and not the shape.

    Killing mutation: the zero-coupon refusal dropped, the document priced.
    """
    with pytest.raises(utils.UnpriceableSchedule) as raised:
        _run_doc(_zero_coupon_doc(barrier=barrier), tmp_path, 'zc_refused')
    said = str(raised.value)
    row_date = (BASE_DAY + datetime.timedelta(days=AC_ZERO_DAYS[1])).isoformat()
    assert 'AC1' in said and row_date in said, said
    assert 'quoted 0' in said and 'not a coupon' in said, said
    assert 'takes this row\'s interval in place of its own' in said, said
    assert 'Author a real coupon' in said and 'delete the row' in said, said
    assert ('PREVIOUS fixing' in said) == barrier, said

    priced, _, log = _run_doc(_zero_coupon_doc(coupon=AC_COUPON, barrier=barrier),
                              tmp_path, 'zc_real', debug=True)
    line, = [ln for ln in log.splitlines() if 'AUTOCALL AC1' in ln]
    assert 'fullpath=0' in line and 'coupons=3 thresholds=3' in line, line
    assert np.isfinite(priced) and priced != 0.0, priced


@pytest.mark.parametrize('by', ['fixings', 'barrier'])
def test_an_averaging_autocall_under_the_switch_falls_back_by_name(by, tmp_path):
    """THE ARM THE SWITCH DOES NOT REACH, priced on the CRISP estimator and said so rather than
    refused - so a mixed book runs under one setting with every deal priced. Its termination is a
    smoothed per-inner-path weight and its breach a hard indicator on the AVERAGE. Both ways
    `calc_dependencies` puts a deal on that arm are walked: more than one price fixing per coupon,
    and a barrier date off the coupon dates. The fallback is the switch-off run BIT FOR BIT, and
    the line names the deal and what its greeks are blind to; the switch-off run carries no line.

    Killing mutation: the fallback dropped, the full-path arm run under the switch unannounced.
    """
    crisp, _, off_log = _run_doc(_crisp(_averaging(_autocall_doc(0.7), by)), tmp_path,
                                 'avg_off', debug=True)
    assert np.isfinite(crisp) and crisp != 0.0, crisp
    assert 'falls back to the CRISP estimator' not in off_log
    fell_back, _, log = _run_doc(_smooth(_averaging(_autocall_doc(0.7), by)), tmp_path, 'avg_on',
                                 debug=True)
    assert fell_back.hex() == crisp.hex(), (fell_back.hex(), crisp.hex())
    assert 'falls back to the CRISP estimator on AC1' in log, log[-1200:]
    assert 'FULL-PATH branch' in log and 'AVERAGE' in log, log[-1200:]
    assert 'smooth_heaviside_up' in log
    assert 'BLIND to the threshold kink' in log
    assert 'ONE price fixing per coupon' in log
