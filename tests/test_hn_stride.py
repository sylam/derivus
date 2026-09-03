"""THE STRIDE - the k-step conditional law of ln S given (h, q), cached, drawn and carried.

The same backward A/B/C recursion run k steps IS the k-step conditional log-CF exactly, and its
coefficients depend on the parameters, the calendar position and the transform node but never on the
state - so they are cached once and every per-path question after that is exponentials and a dot
product. New here: the C*q state axis, the survival-truncated inversion, and the carry.

THE ONE APPROXIMATION IS THE CARRY, quadratic by KIND: E[h_k | x] is the news-impact curve, an
asymmetric U tilted by gamma_1, so a linear carry is wrong in shape however well fitted.
`h_k = a + b*y + c*y^2` in the centered return, (a, b, c) the L2 projection pinned by joint cumulants
off ONE autograd chain on the cached recursion, plus a residual matched in variance and in the h-q
residual correlation.

THE ORACLE HAS TWO ARMS. Arm A: exact E[h|x], Var(h|x), E[q|x], Var(q|x), Cov(h,q|x) by ONE-
dimensional inversion of the u/v-differentiated joint transform, on the PRODUCTION contour and
unwrap differentiated at zero (a 2D (u, v) grid would put the log(1 - 2b) branch on an axis
`complex_log_unwrap` was never anchored for). Arm B: the model's own daily recursion, paired to Arm A
by x, so a difference in any scored functional is the carry's and nothing else. The two agree inside
three standard errors across the bulk of the return distribution; the outer bins differ by the bin
WIDTH and are excluded by name.

THE WORST-K TABLE - the declared approximation carrying its number. k is trading days; every column
is the stride against the oracle, at h_0 = 1.6x the stationary level, phi share 0.35, rho 0.99, 2^18
paired paths, on the 41-node exact quantile grid of the return:

    k   E[h|x] rel      Var(h|x)     floor  trigger rel   KI put   h 5% / 50% / 95% rel   sd_h
         max / rms      / matched     mass     (in SE)       rel                         /E[h]
    1  3.6e-16 3.6e-16     exact     0.0000  0.0e+00 (0)  -7.8e-16  -0.000 +0.000 -0.000  0.000
    3  2.1e-02 3.9e-03  0.87 - 1.10  0.0000 -6.2e-05 (-8) +2.5e-04  -0.000 +0.001 -0.001  0.029
    8  1.0e-01 1.7e-02  0.56 - 1.51  0.0004 -2.5e-04 (-11)+5.8e-05  -0.014 +0.005 -0.003  0.073
   13  2.1e-01 3.4e-02  0.36 - 1.85  0.0029 -4.2e-04 (-11)-1.2e-03  -0.057 +0.010 -0.009  0.122
   21  2.9e-01 4.9e-02  0.22 - 2.27  0.0110 -5.5e-04 (-9) -2.4e-03  -0.181 +0.026 -0.016  0.200
   34  2.7e-01 4.7e-02  0.15 - 2.75  0.0267 -3.4e-04 (-4) -2.0e-03  -0.481 +0.060 -0.029  0.307
   55  1.8e-01 3.2e-02  0.13 - 3.21  0.0475 -4.7e-04 (-4) -7.6e-04  -0.940 +0.108 -0.047  0.426
  126  7.5e-02 1.3e-02  0.15 - 3.31  0.0675 -8.7e-04 (-7) -6.4e-05  -1.000 +0.164 -0.073  0.559

  The long-run component q is carried an order of magnitude better: its conditional-mean error peaks
  at 2.1e-2 and its residual never reaches the floor.

THE ERROR IS NON-MONOTONE IN k, as the design requires: zero at k -> 0, forgetting x as
k -> infinity, worst in between - measured at k = 24-25, with monthly (k = 21) at 98% of the peak.
Monthly autocall fixings ARE the operating point, which is why this gate scans k.

THE FLOOR MASS IS THE FINDING. The residual is Gaussian and the state is floored, and the residual's
coefficient of variation sd_h/E[h] reaches 0.20 at k = 21 and 0.64 at k = 252 - so a Gaussian puts
1.1% and 8.0% of its mass below zero, onto a floor of 1e-12 (0.16 bp of annualised vol - a frozen
path). The SMOOTH functionals barely notice (0.05% on the trigger, 0.24% on the KI put at k = 21):
the floored paths are the low-variance ones those weight least. The h TAIL QUANTILES are destroyed -
the 5% quantile is -18% at k = 21 and the 1% quantile is ON the floor from k = 21 on. That is
vol-of-vol convexity, and the trigger for escalation rung (1): map the residual through the exact 1D
h-marginal quantiles, invertible off the same cached coefficients.

WALL CLOCK, k = 21, RTX 3090 / float64, idle box at 512 nodes: cache build 5.5 ms at a given
phi_max, 64 ms more for the adaptive bound, 245 ms more for the origin block (the microsecond claim
belongs to the per-path Phi); batched Phi 6.8 ms at 16,384 paths / 59 ms at 131,072; survival draw
73 / 555 ms; the daily walk 15 / 18 ms.

SO THE STRIDE IS NOT A WALL-CLOCK LEVER AT THESE SHAPES. The draw costs n_paths x n_nodes complex
work against the walk's n_paths x k real work, so the crossover is near k = 640 daily steps and no
contract reaches it. The OSS consumer's case is SURVIVAL CONDITIONING OF THE WHOLE INTERVAL, not
speed; the Phi-only consumers have no daily-walk equivalent at all.

No HMC and no monkeypatching: every oracle rides `utils.hn_component_variance_step` or
`utils.hn_component_abc` directly.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time

import numpy as np
import pytest
import torch

from derivus import pricing, utils
from crn_ladder import ladder
import hn_reference as hnref

DTYPE = torch.float64
SPY = 252.0
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
#: Paired paths for the oracle. The gate is a distributional comparison, so the count sets the
#: standard error every reading is quoted against; a CPU-only box runs the same gate smaller.
PATHS = (1 << 18) if DEV == 'cuda' else (1 << 15)
#: The scan. NOT a spot check: the carry error is non-monotone in k and its maximum sits in the
#: middle, where monthly fixings land.
K_SCAN = (1, 3, 8, 13, 21, 34, 55, 126)
#: The horizon the scored functionals look forward over from the fork - one monthly fixing.
K_FORWARD = 21

#: THE RECORDED CARRY ERROR, per k: the module table's `E[h|x] rel` column with 25% of headroom.
#: Tight on purpose - deleting the quadratic term roughly DOUBLES the error at every rung (and takes
#: k = 1 from 3.6e-16 to 1.1e-1), so this bound is failed by the mutant everywhere and by nothing
#: else. A looser bound would pass a linear carry.
RECORDED_MEAN_REL = {1: 1e-13, 3: 2.7e-2, 8: 1.3e-1, 13: 2.6e-1, 21: 3.7e-1,
                     34: 3.4e-1, 55: 2.3e-1, 126: 9.4e-2}

#: The MEDIAN of the carried h against the oracle's, per k - the reading that says the bulk of the
#: distribution has not drifted. It drifts UP with k as the Gaussian residual replaces a
#: right-skewed one; the tails have their own gates.
RECORDED_MEDIAN_REL = {1: 1e-12, 3: 0.003, 8: 0.008, 13: 0.014, 21: 0.033,
                       34: 0.075, 55: 0.135, 126: 0.205}


def _t(x):
    return torch.tensor(float(x), dtype=DTYPE, device=DEV)


# ======================================================================================
# The world under test: a LIVE component law (phi > 0, rho < 1, a sloping L curve), built off the
# plain family through `hn_component_from_plain` so nothing here re-spells a parameter map.
# ======================================================================================

PLAIN = hnref.hn_params_from_targets(
    ann_vol=0.30, persistence=0.94, gamma=350.0, leverage_share=0.7, steps_per_year=SPY)
L_KNOTS = np.array([0.0, 0.25, 1.0, 2.0])


def _world(phi_share=0.35, rho=0.99, gamma2=120.0, horizon=400):
    """(hnc_params, omega strip, r, h0, q0) for the live law every gate below runs on."""
    om, al, be, ga = (_t(PLAIN[k]) for k in ('omega', 'alpha', 'beta', 'gamma_star'))
    alpha, beta, gamma1, level = utils.hn_component_from_plain(om, al, be, ga)
    prm = (alpha, beta, gamma1, _t(rho), phi_share * alpha, _t(gamma2))
    lv = torch.tensor([1.3, 1.1, 1.0, 0.95], dtype=DTYPE, device=DEV) * level
    l_path = utils.hn_component_l_path(L_KNOTS, lv, horizon, SPY)
    omegas = utils.hn_component_omega_path(l_path, prm[3])
    # h0 well above the long-run level - after a shock, where the carry has the most work to do
    return prm, omegas, _t(0.02 / SPY), 1.6 * float(level), float(l_path[0])


PRM, OMEGAS, R_STEP, H0, Q0 = _world()
NESTED = (lambda p: (p[0], p[1], p[2], _t(0.99), _t(0.0), _t(-1234.5)))(
    utils.hn_component_from_plain(*(_t(PLAIN[k]) for k in
                                    ('omega', 'alpha', 'beta', 'gamma_star'))))
NESTED_LEVEL = utils.hn_component_from_plain(
    *(_t(PLAIN[k]) for k in ('omega', 'alpha', 'beta', 'gamma_star')))[3]


#: 128 panels = 1024 nodes, against the family's default 2048. Memory is O(paths x nodes) and a
#: 2048-node strip puts 2^18 paths past a 24 GB device, so every cube-driving gate halves the grid
#: (the bitwise gate takes the default, being the grid it must match). The halving costs under 1e-9
#: of probability. Do NOT quarter it: 512 nodes is 6.4e-13 at a bound of 512 but 2.9e-8 at the wider
#: bound a low-variance corner of the state box asks for.
GATE_PANELS = 128


def _strip(k, day=0, panels=GATE_PANELS, **kw):
    """The cached stride for `k` steps anchored at trading `day`, over the box the cube reaches."""
    return utils.hn_component_stride_strip(
        list(OMEGAS[day:day + k]), PRM, R_STEP, _t(0.15 * H0), _t(0.5 * Q0), panels=panels, **kw)


# ======================================================================================
# THE ORACLE, arm B: the exact conditional sampler - the model's OWN daily recursion.
# ======================================================================================

@torch.no_grad()
def _walk(omegas, n, seed=11, h0=None, q0=None, b=None):
    """(aggregate x, h, q) after `len(omegas)` exact daily component steps from (h0, q0), on
    `utils.hn_component_variance_step` directly - the sampler the whole carry is scored against.

    `b` is the per-step cost of carry, a parameter because
    `test_the_stride_step_un_shifts_the_carry` scores a return drawn under one carry against a walk
    taken under ANOTHER, on the model's own sampler rather than the shift theorem it checks.
    """
    b = R_STEP if b is None else b
    h = torch.full((n,), H0 if h0 is None else h0, dtype=DTYPE, device=DEV)
    q = torch.full((n,), Q0 if q0 is None else q0, dtype=DTYPE, device=DEV)
    x = torch.zeros(n, dtype=DTYPE, device=DEV)
    g = torch.Generator(device=DEV).manual_seed(int(seed))
    for omega_t in omegas:
        z = torch.randn(n, generator=g, dtype=DTYPE, device=DEV)
        sh = h.sqrt()
        x = x + (b - 0.5 * h + sh * z)
        h, q = utils.hn_component_variance_step(h, q, sh, z, omega_t, *PRM)
    return x, h, q


# ======================================================================================
# THE ORACLE, arm A: the exact conditional moments, by 1D inversion of the differentiated transform.
# ======================================================================================

def _conditional_moments(omegas, x, h0=None, q0=None, phi_max=None, panels=256):
    """EXACT E[h|x], Var(h|x), E[q|x], Var(q|x), Cov(h,q|x) and the density f(x). NO Monte Carlo.

    ``E[h_k e^{i xi R}]`` is the u-derivative of the joint transform at the origin, so inverting
    over xi and dividing by the density gives the conditional mean; second derivatives give the
    second moments. Autograd on `utils.hn_component_abc` along the PRODUCTION contour and unwrap,
    differentiated at zero and never evaluated away from it, so no branch of log(1 - 2b) moves.

    The bound is doubled off the price's adaptive scan because the differentiated integrand carries
    a polynomial factor in xi and decays slower than the CDF's.
    """
    h0 = H0 if h0 is None else h0
    q0 = Q0 if q0 is None else q0
    if phi_max is None:
        phi_max = 2.0 * utils.hn_component_auto_phi_max(
            omegas, _t(h0), _t(q0), *PRM, R_STEP)
    nodes, wts = utils.gauss_legendre(0.0, phi_max, panels, 8, DTYPE, DEV)
    n = nodes.numel()
    u = torch.zeros(n, dtype=DTYPE, device=DEV, requires_grad=True)
    v = torch.zeros(n, dtype=DTYPE, device=DEV, requires_grad=True)
    A, B, C = utils.hn_component_abc(nodes * 1j, omegas, *PRM, R_STEP, terminal=(u, v))
    lcf = A + B * h0 + C * q0

    def d(y, wrt):
        gr = torch.autograd.grad(y.real.sum(), wrt, create_graph=True, allow_unused=True)[0]
        gi = torch.autograd.grad(y.imag.sum(), wrt, create_graph=True, allow_unused=True)[0]
        return torch.complex(gr, gi)

    lu, lv = d(lcf, u), d(lcf, v)
    luu, luv, lvv = d(lu, u), d(lu, v), d(lv, v)
    psi = torch.exp(lcf).detach()
    lu, lv, luu, luv, lvv = (z.detach() for z in (lu, lv, luu, luv, lvv))
    y = {'f': psi, 'h': lu * psi, 'q': lv * psi, 'hh': (lu * lu + luu) * psi,
         'qq': (lv * lv + lvv) * psi, 'hq': (lu * lv + luv) * psi}
    e = torch.exp(-1j * nodes * x.unsqueeze(-1))
    m = {kk: ((e * vv).real * wts).sum(-1) / np.pi for kk, vv in y.items()}
    f = m['f']
    e_h, e_q = m['h'] / f, m['q'] / f
    return {'f': f, 'Eh': e_h, 'Eq': e_q, 'Vh': m['hh'] / f - e_h ** 2,
            'Vq': m['qq'] / f - e_q ** 2, 'Chq': m['hq'] / f - e_h * e_q}


# ======================================================================================
# The scored functionals - what the BOOK prices off the carried state, not what is easy to compare.
# ======================================================================================

def _forward_bits(k, states):
    """The forward strip and a quadrature bound valid for every state either arm reaches."""
    fwd = list(OMEGAS[k:k + K_FORWARD])
    hs = torch.stack([z.min() for z in states[0]] + [z.max() for z in states[0]])
    qs = torch.stack([z.min() for z in states[1]] + [z.max() for z in states[1]])
    return fwd, utils.hn_component_auto_phi_max(fwd, hs, qs, *PRM, R_STEP)


@torch.no_grad()
def _trigger(fwd_strip, x, h, q, chunk=1 << 13):
    """P(S rises back to its start by the NEXT fixing | the state carried to this one) - the
    compounding channel an autocall reads.

    CHUNKED: Phi over a (2^18, 1024) node strip is 4.3 GB of complex128 per temporary and this gate
    builds several. Memory, O(paths x nodes), is the stride's real constraint."""
    return torch.cat([1.0 - utils.hn_component_stride_cdf(
        fwd_strip, -x[i:i + chunk], h[i:i + chunk], q[i:i + chunk])
        for i in range(0, x.numel(), chunk)])


@torch.no_grad()
def _ki_put(fwd, phi_max, x, h, q, strike=0.70, chunk=1 << 13):
    """The knock-in put's European leg at the fork horizon, off the carried state. Chunked: a
    (paths, node) complex strip at these path counts does not fit whole."""
    s = torch.exp(x)
    out = []
    for i in range(0, s.numel(), chunk):
        sl = slice(i, i + chunk)
        out.append(utils.hn_component_put(
            s[sl], _t(strike), fwd, h[sl], q[sl], *PRM, R_STEP, phi_max=phi_max,
            panels=GATE_PANELS))
    return torch.cat(out)


_SCAN = {}


def _scan(k):
    """One rung of the k-scan: the oracle, the stride carried at the ORACLE'S OWN return, and every
    scored functional on both. Cached - eight tests read the same run."""
    if k in _SCAN:
        return _SCAN[k]
    om = list(OMEGAS[:k])
    strip = _strip(k)
    x, h_o, q_o = _walk(om, PATHS)
    hb = torch.full_like(x, H0)
    qb = torch.full_like(x, Q0)
    ld = utils.hn_component_stride_carry_loadings(strip, hb, qb)
    g = torch.Generator(device=DEV).manual_seed(5)
    e1 = torch.randn(PATHS, generator=g, dtype=DTYPE, device=DEV)
    e2 = torch.randn(PATHS, generator=g, dtype=DTYPE, device=DEV)
    h_s, q_s = (z.detach() for z in utils.hn_component_stride_carry(strip, x, hb, qb, e1, e2, ld))
    ld = ld._replace(**{f: getattr(ld, f).detach() for f in ld._fields})
    raw_h = ld.a_h + ld.b_h * (x - ld.mean_x) + ld.c_h * (x - ld.mean_x) ** 2 + ld.sd_h * e1

    fwd, phi_max = _forward_bits(k, ((h_o, h_s), (q_o, q_s)))
    fwd_strip = utils.hn_component_stride_strip(
        fwd, PRM, R_STEP, _t(H0), _t(Q0), phi_max=phi_max, panels=GATE_PANELS,
        moments=False)
    t_o, t_s = _trigger(fwd_strip, x, h_o, q_o), _trigger(fwd_strip, x, h_s, q_s)
    v_o, v_s = (_ki_put(fwd, phi_max, x, h_o, q_o), _ki_put(fwd, phi_max, x, h_s, q_s))

    # the conditional-mean curve, on the exact quantile grid of the return - noise-free
    ps = torch.linspace(0.01, 0.99, 41, dtype=DTYPE, device=DEV)
    xg = utils.hn_component_stride_invert(
        strip, ps, torch.full_like(ps, H0), torch.full_like(ps, Q0))
    orc = _conditional_moments(om, xg)
    lg = utils.hn_component_stride_carry_loadings(
        strip, torch.full_like(xg, H0), torch.full_like(xg, Q0))
    # the loadings ride the cache's graph (the origin block is built under `create_graph`); the scan
    # reports numbers, so it reads them detached
    lg = lg._replace(**{f: getattr(lg, f).detach() for f in lg._fields})
    yg = xg - lg.mean_x
    fit_h = lg.a_h + lg.b_h * yg + lg.c_h * yg * yg
    fit_q = lg.a_q + lg.b_q * yg + lg.c_q * yg * yg

    qs = torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99], dtype=DTYPE, device=DEV)
    out = {
        'k': k, 'strip': strip, 'x': x, 'h_o': h_o, 'q_o': q_o, 'h_s': h_s, 'q_s': q_s,
        'loadings': ld, 'raw_h': raw_h, 'grid': xg, 'oracle': orc, 'fit_h': fit_h,
        'fit_q': fit_q, 'grid_loadings': lg,
        'mean_rel_h': float(((fit_h - orc['Eh']).abs() / orc['Eh']).max()),
        'mean_rel_q': float(((fit_q - orc['Eq']).abs() / orc['Eq']).max()),
        # the quantile grid is equiprobable, so a plain mean over it IS density-weighted
        'rms_rel_h': float((((fit_h - orc['Eh']) / orc['Eh']) ** 2).mean().sqrt()),
        'var_ratio': (float((orc['Vh'] / lg.sd_h ** 2).min()) if k > 1 else float('nan'),
                      float((orc['Vh'] / lg.sd_h ** 2).max()) if k > 1 else float('nan')),
        'floor_h': float((h_s <= utils.HN_COMPONENT_VARIANCE_FLOOR).double().mean()),
        'floor_q': float((q_s <= utils.HN_COMPONENT_VARIANCE_FLOOR).double().mean()),
        'negative_h': float((raw_h < 0.0).double().mean()),
        'trigger': (float(t_s.mean() - t_o.mean()), float((t_s - t_o).std() / PATHS ** 0.5),
                    float(t_o.mean())),
        'ki_put': (float(v_s.mean() - v_o.mean()), float((v_s - v_o).std() / PATHS ** 0.5),
                   float(v_o.mean())),
        'h_quantile_rel': (torch.quantile(h_s, qs) / torch.quantile(h_o, qs) - 1.0).tolist(),
        'cv': float(lg.sd_h[0] / lg.a_h[0]),
    }
    _SCAN[k] = out
    return out


# ======================================================================================
# GATE 1 - Phi over the cache
# ======================================================================================

@pytest.mark.parametrize('k', [1, 5, 21, 63])
def test_the_cached_phi_is_the_uncached_inversion(k):
    """The cache is the MEMOISED recursion, so bitwise is the right bar and it is reached.

    `hn_component_cdf_logret` builds its grid, runs the recursion on it and assembles 1 - P2; the
    strip does the first two ONCE and the third per path. Same nodes, weights and assembly order, so
    at equal phi_max/panels/order they differ by nothing at all. Measured: `torch.equal` at every k.
    """
    om = list(OMEGAS[:k])
    strip = _strip(k, panels=None, moments=False)          # the FAMILY default grid, deliberately
    # spread the points over the k-step scale: a fixed grid is 19 sd out at k = 1, where the
    # probability rounds to a hard 0 or 1 and tests the float rather than the CDF
    sd = float(np.sqrt(k * H0))
    x = torch.tensor([-2.5, -1.5, -0.5, 0.0, 0.4, 1.2, 2.2], dtype=DTYPE, device=DEV) * sd
    h = torch.tensor([0.3, 0.6, 0.9, 1.0, 1.4, 2.2, 3.4], dtype=DTYPE, device=DEV) * H0
    q = torch.tensor([0.7, 0.8, 0.9, 1.0, 1.1, 1.3, 1.6], dtype=DTYPE, device=DEV) * Q0
    cached = utils.hn_component_stride_cdf(strip, x, h, q)
    direct = utils.hn_component_cdf_logret(
        x, om, h, q, *PRM, R_STEP, phi_max=strip.phi_max)
    assert torch.equal(cached, direct), float((cached - direct).abs().max())
    assert bool(((cached > 0.0) & (cached < 1.0)).all())
    # the reduced grid every gate below drives a cube on: 1024 nodes against the family's 2048,
    # halving the memory for under 1e-9 of probability (512 would be 2.9e-8 - see GATE_PANELS)
    coarse = _strip(k, moments=False)
    assert float((utils.hn_component_stride_cdf(coarse, x, h, q) - cached).abs().max()) < 1e-9


@pytest.mark.parametrize('k', [1, 5, 21, 63, 126])
def test_the_cached_coefficients_nest_on_the_plain_family(k):
    """phi = 0 with L FLAT: the cached B strip IS `hn_ab`'s B, and A + C*L IS its A.

    Stronger than the price-level nesting it inherits (1.5e-13): `A + B*h + C*q` is affine in h and
    h is free, so on the nested face the two families must agree coefficient by coefficient rather
    than merely integrate to the same price. Measured 8.9e-16 on B and 4.0e-15 on A + C*L, every k.
    """
    om, al, be, ga = (_t(PLAIN[key]) for key in ('omega', 'alpha', 'beta', 'gamma_star'))
    omegas = [NESTED_LEVEL * (1.0 - NESTED[3])] * k
    nodes, _ = utils.gauss_legendre(0.0, 512.0, 64, 8, DTYPE, DEV)
    a_c, b_c, c_c = utils.hn_component_abc(nodes * 1j, omegas, *NESTED, R_STEP)
    a_p, b_p = utils.hn_ab(nodes * 1j, k, om, al, be, ga, R_STEP)
    assert float((b_c - b_p).abs().max() / b_p.abs().max()) < 1e-14
    assert float((a_c + c_c * NESTED_LEVEL - a_p).abs().max() / a_p.abs().max()) < 1e-13


@pytest.mark.parametrize('k', [5, 21])
def test_the_cached_phi_is_the_daily_walks_own_cdf(k):
    """The cached Phi against the empirical CDF of the daily recursion it replaces - the gate above
    says the cache is the same INVERSION, this says the inversion is the same LAW. Deciles, 2^21
    paths (2^15 on CPU), against the binomial standard error at each; every one inside 3.5 SE.
    """
    n = (1 << 21) if DEV == 'cuda' else (1 << 15)
    x, _, _ = _walk(list(OMEGAS[:k]), n, seed=101)
    strip = _strip(k, moments=False)
    ps = torch.linspace(0.1, 0.9, 9, dtype=DTYPE, device=DEV)
    xq = torch.quantile(x, ps)
    phi = utils.hn_component_stride_cdf(
        strip, xq, torch.full_like(xq, H0), torch.full_like(xq, Q0))
    se = (ps * (1.0 - ps) / n).sqrt()
    worst = float(((phi - ps).abs() / se).max())
    assert worst < 3.5, (worst, phi.tolist(), ps.tolist())


def test_the_truncated_draw_has_the_truncated_law():
    """Survival-truncated inverse-CDF: the drawn sample IS the law conditioned on staying below.

    Three readings, two about the law. No draw may exceed the cap - the truncation is a hard
    constraint, not a bias. And the DAILY WALK'S OWN SURVIVORS must have the truncated CDF (inside
    3.5 SE), which is the arm the LAW rests on: the model's sampler against the inversion, sharing
    nothing but the parameters.

    THE MIDDLE ARM IS THE INVERTER, NOT THE LAW. `Phi(x_i)/Phi_cap == u_i` holds by construction to
    1e-14, so the round trip reads a returned root that is not the root, a Phi_cap normalisation
    applied twice or not at all, a draw that left its bracket - but not whether the law is the
    model's, because Phi is both sides of it.
    """
    k = 21
    strip = _strip(k)
    x_cap = _t(np.log(1.10))
    n = min(PATHS, 1 << 15)                    # O(paths x nodes): a draw holds six such buffers
    g = torch.Generator(device=DEV).manual_seed(23)
    u = torch.rand(n, generator=g, dtype=DTYPE, device=DEV)
    hb, qb = torch.full((n,), H0, dtype=DTYPE, device=DEV), torch.full((n,), Q0, dtype=DTYPE,
                                                                      device=DEV)
    x, phi_cap = utils.hn_component_stride_draw(strip, u, hb, qb, x_cap=x_cap)
    assert float(x.max()) < float(x_cap)

    ps = torch.linspace(0.05, 0.95, 19, dtype=DTYPE, device=DEV)
    xq = torch.quantile(x, ps)
    law = utils.hn_component_stride_cdf(
        strip, xq, torch.full_like(xq, H0), torch.full_like(xq, Q0)) / phi_cap[0]
    se = (ps * (1.0 - ps) / n).sqrt()
    assert float(((law - ps).abs() / se).max()) < 3.5, (law.tolist(), ps.tolist())

    xw, _, _ = _walk(list(OMEGAS[:k]), n, seed=77)
    surv = xw[xw <= x_cap]
    xq = torch.quantile(surv, ps)
    law = utils.hn_component_stride_cdf(
        strip, xq, torch.full_like(xq, H0), torch.full_like(xq, Q0)) / phi_cap[0]
    se = (ps * (1.0 - ps) / surv.numel()).sqrt()
    assert float(((law - ps).abs() / se).max()) < 3.5, (law.tolist(), ps.tolist())


@pytest.mark.parametrize('k', [1, 21, 63])
def test_the_calendar_enters_a_alone(k):
    """The omega strip touches A and NOTHING else, which is what makes the cache calendar-ANCHORED
    rather than calendar-shaped.

    Two strips of equal length at different anchors on a SLOPING L curve are BITWISE identical in B
    and C, because those recursions never reference omega_t. Consequence (the recorded escalation
    rung): equal-length intervals can share one B/C pair and recompute A alone, and A is affine in
    the strip (dA/domega_n is step n's D), so that is one autograd pass rather than a recursion.

    AND k = 1 IS CALENDAR-FREE ENTIRELY, A included: omega enters A only through D = B + C, and the
    first backward step has D = 0 by the (0, 0) terminal condition. The CARRY still reads all k -
    its terminal condition is (u, v), so D is 1 at that step, which is why E[h_{t+1}] carries
    omega_t.
    """
    # the SAME node grid on both, or B and C differ because the adaptive bound is a function of the
    # omega slice
    first = _strip(k, day=0, phi_max=1024.0, moments=False)
    later = _strip(k, day=120, phi_max=1024.0, moments=False)
    assert torch.equal(first.B, later.B)
    assert torch.equal(first.C, later.C)
    if k == 1:
        assert torch.equal(first.A, later.A)
    else:
        assert float((first.A - later.A).abs().max()) > 1.0   # the anchors really are different
    # the carry reads the calendar at every k, k = 1 included
    near = _strip(k, day=0, phi_max=1024.0)
    far = _strip(k, day=120, phi_max=1024.0)
    assert float((near.mom - far.mom).detach().abs().max()) > 0.0


def test_the_cost_of_carry_is_a_shift():
    """One strip serves any per-step carry, per-path ones included - which is what lets a cache be
    indexed by calendar position alone while `b_step` reaches the OSS pricers as a tensor.

    `r` enters A only as `phi*r*k` against `exp(-i phi x)`, so changing the carry by `d` is EXACTLY
    shifting the moneyness by `-d*k`. Measured 2.2e-16 on Phi over a 25-point grid, 3.6e-15 on the A
    strip, every carry loading BITWISE identical - only the first cumulant moves, by exactly `d*k`.
    """
    k = 21
    om = list(OMEGAS[:k])
    r0, r1 = R_STEP, R_STEP * 3.0
    kw = dict(phi_max=1024.0, panels=GATE_PANELS)
    s0 = utils.hn_component_stride_strip(om, PRM, r0, _t(H0), _t(Q0), **kw)
    s1 = utils.hn_component_stride_strip(om, PRM, r1, _t(H0), _t(Q0), **kw)
    d = float(r1 - r0) * k

    x = torch.linspace(-0.30, 0.20, 25, dtype=DTYPE, device=DEV)
    h, q = torch.full_like(x, H0), torch.full_like(x, Q0)
    assert float((utils.hn_component_stride_cdf(s1, x, h, q)
                  - utils.hn_component_stride_cdf(s0, x - d, h, q)).abs().max()) < 1e-14
    assert float((s1.A - (s0.A + 1j * s0.nodes * d)).abs().max()) < 1e-13

    l0 = utils.hn_component_stride_carry_loadings(s0, _t(H0), _t(Q0))
    l1 = utils.hn_component_stride_carry_loadings(s1, _t(H0), _t(Q0))
    assert float((l1.mean_x - l0.mean_x).detach()) == pytest.approx(d, rel=1e-13)
    for field in l0._fields:
        if field != 'mean_x':
            assert torch.equal(getattr(l0, field), getattr(l1, field)), field
    moved = [utils.HN_STRIDE_MOMENT_KEYS[i] for i in range(len(utils.HN_STRIDE_MOMENT_KEYS))
             if float((s1.mom[i] - s0.mom[i]).detach().abs().max()) > 0.0]
    assert moved == ['p'], moved                       # only the first cumulant sees the carry


#: The un-shift gate's path counts. The signal is a RIGID translation of (b_step - r)*k and does not
#: shrink with the sample while the walk's quantile band does, so the gate is honest at a count the
#: draw's O(paths x nodes) footprint can hold.
UNSHIFT_PATHS = (1 << 17) if DEV == 'cuda' else (1 << 14)
UNSHIFT_REF = (1 << 21) if DEV == 'cuda' else (1 << 18)


def test_the_stride_step_un_shifts_the_carry():
    """The shift runs BOTH ways, and the step verb owes the return leg.

    A deal carrying `b` reads the strip at moneyness `x - (b - r)*k`, so what comes back is a return
    in the STRIP'S measure and `hn_component_stride_step` un-shifts it by `+(b - r)*k` before moving
    a spot. Without that the survival weight and the carry are both still exactly right - the
    loadings centre on the strip's own mean - while the whole survivor law sits `(b - r)*k` too low.

    At k = 21 and b = 4r that is +0.005000 in log-return:

      the barrier      the draw caps at 0.09030968 instead of ln(1.10) = 0.09531018 - a knock-out
                       that can never reach its own trigger
      the law          the median gap to the DAILY WALK's survivor quantiles under the same b goes
                       from -4.919e-3 to +8.1e-5, against the walk's own band of 1.83e-4 over six
                       replicas: 26.8x the band, down to 0.44x
      the weight       phi_cap 0.8143257 against the walk's survival 0.814463, 0.51 SE - unmoved

    The walk takes `b` on its own steps rather than the shift theorem's word for it.
    """
    k = 21
    om = list(OMEGAS[:k])
    strip = _strip(k)
    b_step = 4.0 * R_STEP
    shift = float(b_step - R_STEP) * k
    barrier = float(np.log(1.10))
    x_cap = _t(barrier - shift)                # the barrier, moved into the strip's own measure
    n = UNSHIFT_PATHS
    g = torch.Generator(device=DEV).manual_seed(23)
    u = torch.rand(n, generator=g, dtype=DTYPE, device=DEV)
    e1 = torch.randn(n, generator=g, dtype=DTYPE, device=DEV)
    e2 = torch.randn(n, generator=g, dtype=DTYPE, device=DEV)

    def sample(bs, chunk=1 << 13):
        """The step verb over the whole sample, chunked for the O(paths x nodes) reason. Spot starts
        at 1.0, so the returned spot IS the log-return's exponential."""
        rets, caps = [], []
        for i in range(0, n, chunk):
            sl = slice(i, i + chunk)
            one = torch.ones(u[sl].numel(), dtype=DTYPE, device=DEV)
            s, _, _, cap = utils.hn_component_stride_step(
                strip, one, one * H0, one * Q0, u[sl], e1[sl], e2[sl], x_cap=x_cap, b_step=bs)
            rets.append(s.log())
            caps.append(cap)
        return torch.cat(rets), torch.cat(caps)

    flat, phi_cap = sample(None)               # b_step=None: the strip's own measure, unchanged
    ret, _ = sample(b_step)
    assert float((ret - (flat + shift)).abs().max()) < 1e-14      # the un-shift is that shift

    # the truncation headroom IS the barrier, and without the un-shift it is short by the shift
    assert float(ret.max()) < barrier
    assert barrier - float(ret.max()) < 1e-4, barrier - float(ret.max())
    assert barrier - float(flat.max()) == pytest.approx(shift, abs=1e-4)

    ps = torch.linspace(0.05, 0.95, 19, dtype=DTYPE, device=DEV)
    xw, _, _ = _walk(om, UNSHIFT_REF, seed=77, b=b_step)
    qw = torch.quantile(xw[xw <= barrier], ps)
    # the walk's OWN band, at the stride sample's count: the MEDIAN signed gap over the quantile
    # grid reads a common-mode translation while per-quantile noise cancels
    band = 0.0
    for seed in (101, 202, 303, 404, 505, 606):
        xa, _, _ = _walk(om, n, seed=seed, b=b_step)
        band = max(band, abs(float((torch.quantile(xa[xa <= barrier], ps) - qw).median())))
    d_fix = float((torch.quantile(ret, ps) - qw).median())
    d_bug = float((torch.quantile(flat, ps) - qw).median())
    assert abs(d_fix) < 2.0 * band, (d_fix, band)
    assert abs(d_bug) > 10.0 * band, (d_bug, band)

    surv = float((xw <= barrier).double().mean())
    se = (surv * (1.0 - surv) / xw.numel()) ** 0.5
    assert abs(float(phi_cap[0]) - surv) < 3.5 * se, (float(phi_cap[0]), surv, se)


def test_blocking_the_inversion_solves_the_same_roots():
    """The inverter blocks its path axis to bound the footprint, and a block must solve what it
    would have solved in company.

    NOT BITWISE: the Newton loop's convergence break reads the WORST residual in the batch, so a
    different blocking stops an iteration earlier or later. Measured 7.1e-15 in x between a
    5,000-path solve and the same paths in blocks of seven; the batch shape is preserved.
    """
    strip = _strip(21)
    n = 5000
    g = torch.Generator(device=DEV).manual_seed(1)
    p = torch.rand(n, generator=g, dtype=DTYPE, device=DEV) * 0.98 + 0.01
    h = torch.full((n,), H0, dtype=DTYPE, device=DEV)
    q = torch.full((n,), Q0, dtype=DTYPE, device=DEV)
    whole = utils.hn_component_stride_invert(strip, p, h, q)
    for chunk in (7, 512, 4999):
        part = utils.hn_component_stride_invert(strip, p, h, q, chunk=chunk)
        assert float((part - whole).abs().max()) < 1e-13, (chunk, float((part - whole).abs().max()))
    two_d = utils.hn_component_stride_invert(
        strip, p[:4000].reshape(50, 80), h[:4000].reshape(50, 80), q[:4000].reshape(50, 80),
        chunk=333)
    assert two_d.shape == (50, 80)
    assert float((two_d.reshape(-1) - whole[:4000]).abs().max()) < 1e-13


def test_the_conditional_moment_oracle_has_converged():
    """Arm A is only an oracle if its quadrature has stopped moving, in BOTH directions.

    Doubling the bound at CONSTANT NODE DENSITY reads TRUNCATION (the u-derivative carries a
    polynomial factor in xi and decays slower than the CDF's integrand); doubling the density at a
    constant bound reads RESOLUTION of the e^{-i xi x} oscillation. Measured 1.4e-13 and 4.1e-13 on
    the conditional variance; "double the bound and keep the panels" reads 7.0e-9 and is measuring
    the panels.
    """
    om = list(OMEGAS[:21])
    x = torch.linspace(-0.30, 0.20, 21, dtype=DTYPE, device=DEV)
    base = utils.hn_component_auto_phi_max(om, _t(H0), _t(Q0), *PRM, R_STEP)
    ref = _conditional_moments(om, x, phi_max=2.0 * base, panels=512)
    for pm, pan in ((4.0 * base, 1024), (2.0 * base, 1024)):
        alt = _conditional_moments(om, x, phi_max=pm, panels=pan)
        for key in ('Eh', 'Eq', 'Vh', 'Vq'):
            rel = float(((alt[key] - ref[key]).abs() / ref[key].abs()).max())
            assert rel < 1e-10, (key, pm, pan, rel)


def test_the_two_oracle_arms_agree():
    """The analytic conditional moments against the daily sampler, binned on EXACT quantiles from
    the cached inversion. Every interior bin inside 3 SE.

    ARM A IS SCORED AT THE BIN'S OWN MEAN x, NOT AT ITS CENTRE. Arm B is an AVERAGE of E[h|x] over
    the bin's sample; scoring Arm A at the centre puts that average against a point value, and the
    curvature difference (2.0e-3 to 2.5e-3 relative in h) is a BIAS that does not shrink with the
    sample. At 2^18 paths it hides inside 3 SE, at 2^21 it does not - bin 3 reads 4.6 / 5.2 / 5.3 SE
    at seeds 11 / 12 / 13. The bin-mean form cancels it to first order at any count: 2.5 at 2^18 and
    2.0 at 2^21.

    The outer two bins each side stay excluded by name - there the bin is wide and open-ended and
    its own mean has not converged.
    """
    k = 21
    om = list(OMEGAS[:k])
    strip = _strip(k)
    x, h, q = _walk(om, PATHS, seed=11)
    ps = torch.linspace(0.02, 0.98, 25, dtype=DTYPE, device=DEV)
    edges = utils.hn_component_stride_invert(
        strip, ps, torch.full_like(ps, H0), torch.full_like(ps, Q0))
    idx = torch.bucketize(x, edges)
    bins = range(3, 23)                                   # the interior bins, by name
    # Arm A at the bin's own mean; the excluded bins keep their centre and are never scored
    xs = 0.5 * (edges[1:] + edges[:-1])
    for b in bins:
        xs[b - 1] = x[idx == b].mean()
    orc = _conditional_moments(om, xs)
    worst = 0.0
    for b in bins:
        m = idx == b
        hb = h[m]
        se = float(hb.std() / m.sum() ** 0.5)
        worst = max(worst, abs(float(hb.mean()) - float(orc['Eh'][b - 1])) / se)
    assert worst < 3.0, worst


# ======================================================================================
# GATE 2 - the carried state, scanned in k
# ======================================================================================

def test_the_one_step_stride_is_the_daily_advance():
    """k = 1 IS `hn_component_daily_advance`, exactly - the case split is an IDENTITY.

    h_1 is quadratic in z and the one-step return is affine in z, so E[h_1 | x] is exactly quadratic
    in x and the residual is exactly zero. Measured 3.6e-16 on E[h|x] and E[q|x], every scored
    functional to 1e-15.
    """
    s = _scan(1)
    assert s['mean_rel_h'] < 1e-13, s['mean_rel_h']
    assert s['mean_rel_q'] < 1e-13, s['mean_rel_q']
    assert float(s['grid_loadings'].sd_h.max()) < 1e-16
    assert float(s['grid_loadings'].sd_q.max()) < 1e-16
    assert abs(s['trigger'][0]) < 1e-15, s['trigger']
    assert abs(s['ki_put'][0]) < 1e-15, s['ki_put']
    assert max(abs(z) for z in s['h_quantile_rel']) < 1e-12, s['h_quantile_rel']
    # and the carry is the recursion itself: reconstruct h_1 from the drawn return
    x = s['x']
    z = (x - R_STEP + 0.5 * H0) / np.sqrt(H0)
    h_ref, q_ref = utils.hn_component_variance_step(
        torch.full_like(x, H0), torch.full_like(x, Q0), _t(np.sqrt(H0)), z, OMEGAS[0], *PRM)
    assert float((s['h_s'] / h_ref - 1.0).abs().max()) < 1e-11
    assert float((s['q_s'] / q_ref - 1.0).abs().max()) < 1e-11


@pytest.mark.parametrize('k', K_SCAN)
def test_the_carry_matches_the_exact_conditional_moments(k):
    """The quadratic conditional mean against the EXACT news-impact curve, per k.

    Noise-free: Arm A gives E[h|x] in closed form on the exact quantile grid, so this reads the
    approximation itself rather than a Monte Carlo estimate of it. The bound is the module's
    worst-k table with 25% of headroom.
    """
    s = _scan(k)
    assert s['mean_rel_h'] < RECORDED_MEAN_REL[k], (k, s['mean_rel_h'], RECORDED_MEAN_REL[k])
    # q is the SLOW component and its loadings are small: carried an order of magnitude better
    assert s['mean_rel_q'] < 0.05, (k, s['mean_rel_q'])
    # the free sign check - see the deviation note below
    assert torch.sign(s['grid_loadings'].b_h[0]) == -torch.sign(PRM[2]), float(
        s['grid_loadings'].b_h[0])


def test_the_news_impact_slope_opposes_the_fitted_gamma():
    """DEVIATION. The design note asks for `sign(b) == sign(fitted gamma)`; the news-impact
    algebra gives the OPPOSITE and this gate holds the algebra.

    h_{t+1} carries alpha*(z - gamma_1*sqrt(h))^2, whose z-slope at the origin is
    -2*alpha*gamma_1*sqrt(h), against a return of +sqrt(h)*z - so a POSITIVE gamma_1 makes
    E[h_k | x] decreasing in x and b negative. Both signs are swept because the component family
    fits either.
    """
    for gamma1_sign in (+1.0, -1.0):
        prm = list(PRM)
        prm[2] = PRM[2] * gamma1_sign
        strip = utils.hn_component_stride_strip(
            list(OMEGAS[:21]), tuple(prm), R_STEP, _t(0.15 * H0), _t(0.5 * Q0))
        ld = utils.hn_component_stride_carry_loadings(strip, _t(H0), _t(Q0))
        assert float(ld.b_h) * gamma1_sign < 0.0, (gamma1_sign, float(ld.b_h))


@pytest.mark.parametrize('k', K_SCAN)
def test_the_book_functionals_survive_the_carry(k):
    """What the BOOK prices off the carry, per k: the compounding channel, the KI put, the h tail.

    Paired - the stride carries at the ORACLE'S OWN realised return, so every difference is the
    carry and not the draw. The three are only a gate together: the trigger and the put are SMOOTH
    and barely move (0.05% and 0.19% at k = 21) while the h tail quantiles, where vol-of-vol
    convexity lives, move 18% at the 5% level.
    """
    s = _scan(k)
    bias, se, level = s['trigger']
    assert abs(bias) / level < 1.5e-3, (k, bias, level, bias / se)
    bias, se, level = s['ki_put']
    assert abs(bias) / level < 6e-3, (k, bias, level, bias / se)
    assert abs(s['h_quantile_rel'][2]) < RECORDED_MEDIAN_REL[k], (k, s['h_quantile_rel'])


@pytest.mark.parametrize('k', K_SCAN)
def test_zeroing_the_quadratic_term_breaks_the_gate(k):
    """MUTATION. Delete c and the live gate MUST fail at every rung - scored against the SAME
    `RECORDED_MEAN_REL` bound rather than a synthetic ratio that could be tuned until it passed.

    Scored where the news-impact channel lives: the conditional-mean curve c exists to bend, and the
    h tail quantiles the bend moves. NOT the trigger probability, where zeroing c makes the bias
    SMALLER by cancellation.

    Measured, max relative error of the conditional mean, fitted -> mutant: 3.6e-16 -> 1.1e-1 at
    k = 1, then 2.1e-2 -> 1.5e-1, 1.0e-1 -> 3.1e-1, 2.1e-1 -> 4.7e-1, 2.9e-1 -> 5.8e-1,
    2.7e-1 -> 5.4e-1, 1.8e-1 -> 4.0e-1, 7.5e-2 -> 1.2e-1.
    """
    s = _scan(k)
    lg = s['grid_loadings']
    linear = lg._replace(c_h=torch.zeros_like(lg.c_h), c_q=torch.zeros_like(lg.c_q))
    y = s['grid'] - linear.mean_x
    mutated = float((((linear.a_h + linear.b_h * y) - s['oracle']['Eh']).abs()
                     / s['oracle']['Eh']).max())
    assert mutated > RECORDED_MEAN_REL[k], (k, mutated, RECORDED_MEAN_REL[k], s['mean_rel_h'])
    if k == 1 or s['floor_h'] > 0.005:
        return
    # THE QUANTILE ARM ONLY WHERE IT CAN STILL READ. k = 1 has no residual to move, and past a floor
    # mass of 0.5% the 5% quantile is already ON the floor (-100% relative), so a mutant cannot read
    # as worse - the floor has saturated the instrument. The conditional-mean arm reads at every k.

    g = torch.Generator(device=DEV).manual_seed(5)
    e1 = torch.randn(PATHS, generator=g, dtype=DTYPE, device=DEV)
    e2 = torch.randn(PATHS, generator=g, dtype=DTYPE, device=DEV)
    ld = s['loadings']._replace(c_h=torch.zeros_like(s['loadings'].c_h),
                                c_q=torch.zeros_like(s['loadings'].c_q))
    h_m, _ = utils.hn_component_stride_carry(
        s['strip'], s['x'], torch.full_like(s['x'], H0), torch.full_like(s['x'], Q0), e1, e2, ld)
    qs = torch.tensor([0.05, 0.95], dtype=DTYPE, device=DEV)
    rel_m = (torch.quantile(h_m, qs) / torch.quantile(s['h_o'], qs) - 1.0).abs()
    rel_s = torch.tensor([abs(s['h_quantile_rel'][1]), abs(s['h_quantile_rel'][3])],
                         dtype=DTYPE, device=DEV)
    assert bool((rel_m > 1.4 * rel_s).all()), (k, rel_m.tolist(), rel_s.tolist())


@pytest.mark.parametrize('k', [8, 21, 55])
def test_halving_the_stride_closes_the_error(k):
    """Two strides of k/2 must be closer to the truth than one of k - the convergence that says the
    error is the CARRY and not a bug in it.

    The composition is honest: the first half carries onto the walk's own half-way return, the
    second onto its second-half return but FROM THE CARRIED STATE, so the two compound exactly as a
    real schedule would. At k = 21 the h quantile errors fall from (-18.1%, +2.6%, -1.6%) to
    (-5.0%, +1.0%, -0.4%).
    """
    s = _scan(k)
    k1 = k // 2
    om = list(OMEGAS[:k])
    x_half, _, _ = _walk(om[:k1], PATHS, seed=11)               # the SAME stream as _scan's walk
    hb, qb = torch.full_like(s['x'], H0), torch.full_like(s['x'], Q0)
    g = torch.Generator(device=DEV).manual_seed(5)
    s1 = _strip(k1)
    e1 = torch.randn(PATHS, generator=g, dtype=DTYPE, device=DEV)
    e2 = torch.randn(PATHS, generator=g, dtype=DTYPE, device=DEV)
    h1, q1 = utils.hn_component_stride_carry(s1, x_half, hb, qb, e1, e2)
    s2 = utils.hn_component_stride_strip(
        om[k1:], PRM, R_STEP, torch.stack([h1.min(), h1.max()]),
        torch.stack([q1.min(), q1.max()]))
    e1 = torch.randn(PATHS, generator=g, dtype=DTYPE, device=DEV)
    e2 = torch.randn(PATHS, generator=g, dtype=DTYPE, device=DEV)
    h2, _ = utils.hn_component_stride_carry(s2, s['x'] - x_half, h1, q1, e1, e2)
    qs = torch.tensor([0.05, 0.5, 0.95], dtype=DTYPE, device=DEV)
    ref = torch.quantile(s['h_o'], qs)
    one = (torch.quantile(s['h_s'], qs) / ref - 1.0).abs()
    two = (torch.quantile(h2, qs) / ref - 1.0).abs()
    print('\n  k=%3d  one stride %s  ->  two of %d %s' % (
        k, ['%.4f' % z for z in one], k // 2, ['%.4f' % z for z in two]))
    assert bool((two < one).all()), (k, one.tolist(), two.tolist())


def test_a_whole_schedule_of_strides_reproduces_the_walks_survival():
    """END TO END: six monthly fixings, an up-barrier at 1.10, both schemes side by side.

      the WALK   observes the spot on the fixing dates only and counts paths that never finished a
                 fixing above the barrier - the product's own definition;
      the STRIDE never leaves the survival set: each fixing draws from the law TRUNCATED at the
                 barrier and carries `Phi_cap` in a running product, so `E[prod Phi_cap]` is that
                 same probability by construction.

    So `hn_component_stride_step` runs six times and the carry's error COMPOUNDS through the strip.

    MEASURED at 2^15 paths: walk 0.467010 +/- 0.002756, stride 0.462364 +/- 0.001510 - a bias of
    -1.00%. The stride's standard error is 0.55x the walk's on the same paths, because it never
    leaves the survival set: that is the OSS consumer's case, variance rather than wall clock.

    This gate found the defect `test_a_cache_built_under_no_grad_is_the_same_cache` now pins -
    nothing shorter would have, since the single-stride gates all run in grad mode.
    """
    n, k, n_fix, barrier = min(PATHS, 1 << 15), 21, 6, 1.10
    x_b = float(np.log(barrier))

    alive = torch.ones(n, dtype=DTYPE, device=DEV)                       # ---- the walk
    log_s = torch.zeros(n, dtype=DTYPE, device=DEV)
    h, q = torch.full((n,), H0, dtype=DTYPE, device=DEV), torch.full((n,), Q0, dtype=DTYPE,
                                                                     device=DEV)
    g = torch.Generator(device=DEV).manual_seed(404)
    with torch.no_grad():
        for f in range(n_fix):
            for omega_t in OMEGAS[f * k:(f + 1) * k]:
                z = torch.randn(n, generator=g, dtype=DTYPE, device=DEV)
                sh = h.sqrt()
                log_s = log_s + (R_STEP - 0.5 * h + sh * z)
                h, q = utils.hn_component_variance_step(h, q, sh, z, omega_t, *PRM)
            alive = alive * (log_s <= x_b).to(DTYPE)
    walked = float(alive.mean())
    se = (walked * (1.0 - walked) / n) ** 0.5

    log_s = torch.zeros(n, dtype=DTYPE, device=DEV)                      # ---- the stride
    h, q = torch.full((n,), H0, dtype=DTYPE, device=DEV), torch.full((n,), Q0, dtype=DTYPE,
                                                                     device=DEV)
    weight = torch.ones(n, dtype=DTYPE, device=DEV)
    g = torch.Generator(device=DEV).manual_seed(909)
    with torch.no_grad():
        for f in range(n_fix):
            strip = _strip(k, day=f * k)
            u = torch.rand(n, generator=g, dtype=DTYPE, device=DEV)
            e1 = torch.randn(n, generator=g, dtype=DTYPE, device=DEV)
            e2 = torch.randn(n, generator=g, dtype=DTYPE, device=DEV)
            s_new, h, q, phi_cap = utils.hn_component_stride_step(
                strip, log_s.exp(), h, q, u, e1, e2, x_cap=x_b - log_s)
            log_s = s_new.log()
            weight = weight * phi_cap
            assert bool((log_s <= x_b + 1e-12).all())     # never leaves the survival set
    strode = float(weight.mean())
    strode_se = float(weight.std() / n ** 0.5)

    print('\n  survival over %d fixings of %d days: walk %.6f +/- %.6f   stride %.6f +/- %.6f'
          '   rel %+.4f' % (n_fix, k, walked, se, strode, strode_se, strode / walked - 1.0))
    assert abs(strode / walked - 1.0) < 0.025, (walked, strode, se, strode_se)
    # and the stride's estimator is the tighter one, which is the OSS consumer's actual case
    assert strode_se < se


@pytest.mark.parametrize('k', K_SCAN)
def test_the_gaussian_residual_floor_mass_is_declared(k):
    """THE COST OF THE GAUSSIAN RESIDUAL, held to its measured number.

    The residual is Gaussian and the state is FLOORED (`HN_COMPONENT_VARIANCE_FLOOR`, the floor the
    daily recursion carries because the CJOW pair has no positivity guarantee at phi > 0). But
    sd_h/E[h] grows with k - 0.20 at k = 21, 0.64 at k = 252 - so a Gaussian puts 1.1% and 8.0% of
    its mass below zero onto a floor of 1e-12, which is 0.16 bp of annualised vol: frozen paths.

    Also the trigger for escalation rung (1) - map the residual through the exact 1D h-marginal
    quantiles, invertible off these same cached coefficients - which deletes the floor mass rather
    than bounding it.
    """
    s = _scan(k)
    recorded = {1: 0.0, 3: 0.0, 8: 0.001, 13: 0.005, 21: 0.015, 34: 0.033,
                55: 0.055, 126: 0.075}
    assert s['floor_h'] <= recorded[k], (k, s['floor_h'], recorded[k])
    assert s['floor_h'] == pytest.approx(s['negative_h'], abs=1e-9)   # the floor IS the negatives
    assert s['floor_q'] == 0.0                     # q is slow: its residual never reaches the floor


def test_the_worst_k_table_is_reported():
    """Print the scan, and assert the error is non-monotone with an INTERIOR maximum."""
    rows = [_scan(k) for k in K_SCAN]
    print('\n%5s %10s %10s %16s %8s %14s %10s %16s' % (
        'k', 'E[h|x] max', 'rms', 'Var(h|x)/matched', 'floor', 'trigger rel', 'KI put',
        'h 5% / 95% rel'))
    for s in rows:
        band = 'exact' if s['k'] == 1 else '%6.2f - %6.2f' % s['var_ratio']
        print('%5d %10.1e %10.1e %16s %8.4f %8.1e(%4.0f) %10.1e %+8.3f /%+.3f' % (
            s['k'], s['mean_rel_h'], s['rms_rel_h'], band, s['floor_h'],
            s['trigger'][0] / s['trigger'][2], s['trigger'][0] / max(s['trigger'][1], 1e-300),
            s['ki_put'][0] / s['ki_put'][2], s['h_quantile_rel'][1], s['h_quantile_rel'][3]))
    worst = max(rows, key=lambda s: s['mean_rel_h'])
    half_life = float(np.log(0.5) / np.log(float(PRM[1])))
    print('worst k = %d, against a short-run half-life of %.1f trading days   '
          '(the trigger bias is quoted relative, with its size in standard errors in brackets)'
          % (worst['k'], half_life))
    assert rows[0]['mean_rel_h'] < rows[1]['mean_rel_h']
    assert rows[-1]['mean_rel_h'] < worst['mean_rel_h']
    assert 8 <= worst['k'] <= 55


# ======================================================================================
# GATE 3 - differentiability
# ======================================================================================

def _ladder_world(alpha_value, k=21, phi_max=512.0, panels=64, moments=True):
    prm = (torch.tensor(alpha_value, dtype=DTYPE, device=DEV),) + PRM[1:]
    return utils.hn_component_stride_strip(
        list(OMEGAS[:k]), prm, R_STEP, _t(H0), _t(Q0),
        phi_max=phi_max, panels=panels, moments=moments)


def test_the_phi_gradient_is_the_derivative_of_the_phi():
    """dPhi/d(alpha) through the cache, against CRN central differences. Phi carries no randomness,
    so the ladder reads a clean limit: 0.00% disagreement at 0.00% flatness across six rungs.
    """
    n = 64
    x = torch.linspace(-0.25, 0.15, n, dtype=DTYPE, device=DEV)
    hb, qb = torch.full_like(x, H0), torch.full_like(x, Q0)
    a0 = float(PRM[0])
    alpha = torch.tensor(a0, dtype=DTYPE, device=DEV, requires_grad=True)
    strip = utils.hn_component_stride_strip(
        list(OMEGAS[:21]), (alpha,) + PRM[1:], R_STEP, _t(H0), _t(Q0),
        phi_max=512.0, panels=64, moments=False)
    utils.hn_component_stride_cdf(strip, x, hb, qb).sum().backward()
    lad = ladder(lambda a: float(utils.hn_component_stride_cdf(
        _ladder_world(a, moments=False), x, hb, qb).sum()), float(alpha.grad), a0)
    assert lad.agrees(tol=1e-5), str(lad)


def test_the_drawn_return_gradient_is_the_implicit_function_theorem():
    """d(drawn x)/d(alpha) and d(drawn x)/dh, against CRN central differences.

    THE ROOT IS FOUND UNDER no_grad, so this gates the reattachment: one graph-carrying Newton step
    puts back exactly `-(dQ/dtheta - u dPhi_cap/dtheta)/q`, the IFT for `Q(x) = u*Phi_cap`. Measured
    0.00% on both, ladders flat to 0.00% - a differentiated ITERATION would not do this, and neither
    would a detached draw.
    """
    n = 64
    a0, cap = float(PRM[0]), _t(np.log(1.10))
    g = torch.Generator(device=DEV).manual_seed(3)
    u = torch.rand(n, generator=g, dtype=DTYPE, device=DEV)
    hb, qb = torch.full((n,), H0, dtype=DTYPE, device=DEV), torch.full((n,), Q0, dtype=DTYPE,
                                                                      device=DEV)

    alpha = torch.tensor(a0, dtype=DTYPE, device=DEV, requires_grad=True)
    strip = utils.hn_component_stride_strip(
        list(OMEGAS[:21]), (alpha,) + PRM[1:], R_STEP, _t(H0), _t(Q0), phi_max=512.0, panels=64)
    utils.hn_component_stride_draw(strip, u, hb, qb, x_cap=cap)[0].sum().backward()
    lad = ladder(lambda a: float(utils.hn_component_stride_draw(
        _ladder_world(a), u, hb, qb, x_cap=cap)[0].sum()), float(alpha.grad), a0)
    assert lad.agrees(tol=1e-5), 'd/d(alpha)\n' + str(lad)

    fixed = _ladder_world(a0)
    h = torch.full((n,), H0, dtype=DTYPE, device=DEV, requires_grad=True)
    utils.hn_component_stride_draw(fixed, u, h, qb, x_cap=cap)[0].sum().backward()
    lad = ladder(lambda hv: float(utils.hn_component_stride_draw(
        fixed, u, torch.full((n,), hv, dtype=DTYPE, device=DEV), qb, x_cap=cap)[0].sum()),
        float(h.grad.sum()), H0)
    assert lad.agrees(tol=1e-5), 'd/dh\n' + str(lad)


def test_the_carried_state_moments_are_the_transforms_own_derivatives():
    """The origin block against CENTRAL DIFFERENCES of the transform it is autodiffed from.

    The carry rests on 13 mixed partials of `A + B*h + C*q` at (phi, u, v) = 0, taken by autograd on
    the cached recursion so nothing hand-writes a moment recursion. Measured 5.3e-10 on the phi row
    and 6.0e-9 on the u row, both FD truncation.

    dB/dv is asserted as a HARD zero, not a tolerance: E[q_{t+k}] is driven by omega and rho alone
    and cannot depend on h_t, so 0.0 rather than 1e-19 says the algebra is structural.
    """
    om = list(OMEGAS[:21])
    block = utils._hn_stride_moment_block(om, PRM, R_STEP)
    idx = {key: i for i, key in enumerate(utils.HN_STRIDE_MOMENT_KEYS)}

    def coeffs(p, u, v):
        arg = [torch.tensor([z], dtype=DTYPE, device=DEV) for z in (p, u, v)]
        a, b, c = utils.hn_component_abc(
            arg[0], om, *PRM, R_STEP, unwrap=False, terminal=(arg[1], arg[2]))
        return torch.stack([a[0], b[0], c[0]])

    for key, axis, step in (('p', 0, 1e-4), ('u', 1, 1e-4), ('v', 2, 1e-4)):
        up, dn = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        up[axis], dn[axis] = step, -step
        fd = (coeffs(*up) - coeffs(*dn)) / (2.0 * step)
        auto = block[idx[key]]
        scale = float(auto.abs().max())
        assert float((fd - auto).abs().max()) < 1e-7 * scale, (key, fd.tolist(), auto.tolist())
    assert float(block[idx['v']][1]) == 0.0                       # dB/dv, structurally zero
    assert float(block[idx['u']][1]) != 0.0                       # dB/du is not


def test_a_cache_built_under_no_grad_is_the_same_cache():
    """A VALUATION BUILDS ITS CACHE INSIDE `torch.no_grad()`, and the origin block is obtained by
    DIFFERENTIATING - so the block forces its own mode or comes back as zeros.

    With the mode left to the caller, a six-fixing schedule returned NaN on all 8,192 paths at the
    FIRST stride while its Phi read 0.832, in range: the zeros propagate as mu2 = 0, the carry's
    Gram determinant goes to 0/0, and every loading is NaN.

    Both directions: the block is BITWISE the same under either mode, and comes back DETACHED under
    no_grad.
    """
    om = list(OMEGAS[:21])
    with torch.enable_grad():
        live = utils.hn_component_stride_strip(om, PRM, R_STEP, _t(H0), _t(Q0), panels=GATE_PANELS)
    with torch.no_grad():
        inference = utils.hn_component_stride_strip(
            om, PRM, R_STEP, _t(H0), _t(Q0), panels=GATE_PANELS)
    assert torch.equal(live.mom.detach(), inference.mom)
    assert inference.mom.grad_fn is None and live.mom.grad_fn is not None
    assert bool((inference.mom[1] != 0.0).any())              # mu2 is not a structural zero

    with torch.no_grad():
        ld = utils.hn_component_stride_carry_loadings(inference, _t(H0), _t(Q0))
        assert all(bool(torch.isfinite(getattr(ld, f)).all()) for f in ld._fields), ld


@pytest.mark.parametrize('k', [3, 21, 63])
def test_the_residual_covariance_is_symmetric(k):
    """The h-q residual covariance has two spellings - `Cov(h,q) - (b_q*M_up + c_q*M_upp)` from h's
    side and `Cov(h,q) - (b_h*M_vp + c_h*M_vpp)` from q's - because a projection residual is
    orthogonal to the span whichever variable is projected. Different expressions in different
    cached moments, agreeing to 1e-13 relative: the L2 projection is genuinely orthogonal and not
    merely a linear solve that converged.
    """
    strip = _strip(k)
    ld = utils.hn_component_stride_carry_loadings(strip, _t(H0), _t(Q0))
    m = {key: strip.mom[i, 0] + strip.mom[i, 1] * H0 + strip.mom[i, 2] * Q0
         for i, key in enumerate(utils.HN_STRIDE_MOMENT_KEYS)}
    from_h = float((m['uv'] - (ld.b_q * m['up'] + ld.c_q * m['upp'])).detach())
    from_q = float((m['uv'] - (ld.b_h * m['vp'] + ld.c_h * m['vpp'])).detach())
    assert abs(from_h - from_q) < 1e-13 * abs(from_h), (k, from_h, from_q)
    # and it is the covariance the carry actually draws: corr * sd_h * sd_q
    assert float((ld.corr * ld.sd_h * ld.sd_q).detach()) == pytest.approx(from_h, rel=1e-12)


# ======================================================================================
# GATE 4 - the existing path is untouched
# ======================================================================================

@pytest.mark.parametrize('k', [1, 21, 63])
def test_the_default_recursion_is_what_it_was(k):
    """`hn_component_abc` with no terminal condition is BITWISE what it was before the stride. The
    one edit is two defaulted keywords and a branch the default path does not enter, so an explicit
    zero terminal must reproduce the default bitwise.
    """
    om = list(OMEGAS[:k])
    nodes, _ = utils.gauss_legendre(0.0, 512.0, 64, 8, DTYPE, DEV)
    zero = torch.zeros_like(nodes) * 1j
    default = utils.hn_component_abc(nodes * 1j, om, *PRM, R_STEP)
    explicit = utils.hn_component_abc(nodes * 1j, om, *PRM, R_STEP, terminal=(zero, zero))
    for a, b in zip(default, explicit):
        assert torch.equal(a, b)
    # and the CDF the OSS pricers already call is unmoved by any of it
    x = torch.tensor([-0.2, 0.0, 0.1], dtype=DTYPE, device=DEV)
    assert bool(torch.isfinite(utils.hn_component_cdf_logret(
        x, om, _t(H0), _t(Q0), *PRM, R_STEP)).all())


# ======================================================================================
# GATE 5 - wall clock
# ======================================================================================

def test_the_wall_clocks_are_recorded():
    """Measure and PRINT what the stride costs, including where it loses.

    The cache build is not microseconds: 5.5 ms for the coefficient strip, 64 ms for the adaptive
    bound, 245 ms for the origin block at k = 21. Microseconds is the per-path Phi.

    And the draw is 30x SLOWER than the daily walk at 131,072 paths - n_paths x n_nodes complex work
    against the walk's n_paths x k real work, so the crossover is near k = 640 and no contract
    reaches it. A finding, not a failure: the OSS consumer's case is survival conditioning of the
    whole interval, and the Phi-only consumers have no daily-walk equivalent at all.
    """
    k, n = 21, (1 << 16) if DEV == 'cuda' else (1 << 13)
    om = list(OMEGAS[:k])

    def clock(fn, repeat=2):
        for _ in range(repeat):
            t = time.perf_counter()
            fn()
            if DEV == 'cuda':
                torch.cuda.synchronize()
            out = time.perf_counter() - t
        return 1e3 * out

    bound = utils.hn_component_auto_phi_max(om, _t(0.15 * H0), _t(0.5 * Q0), *PRM, R_STEP)
    t_bound = clock(lambda: utils.hn_component_auto_phi_max(
        om, _t(0.15 * H0), _t(0.5 * Q0), *PRM, R_STEP))
    t_coef = clock(lambda: utils.hn_component_stride_strip(
        om, PRM, R_STEP, _t(H0), _t(Q0), phi_max=bound, panels=GATE_PANELS,
        moments=False))
    t_block = clock(lambda: utils._hn_stride_moment_block(om, PRM, R_STEP))

    strip = utils.hn_component_stride_strip(
        om, PRM, R_STEP, _t(H0), _t(Q0), phi_max=bound, panels=GATE_PANELS)
    g = torch.Generator(device=DEV).manual_seed(7)
    u = torch.rand(n, generator=g, dtype=DTYPE, device=DEV)
    hb = torch.full((n,), H0, dtype=DTYPE, device=DEV)
    qb = torch.full((n,), Q0, dtype=DTYPE, device=DEV)
    xc = torch.full((n,), float(np.log(1.10)), dtype=DTYPE, device=DEV)
    t_phi = clock(lambda: utils.hn_component_stride_cdf(strip, xc, hb, qb))
    t_draw = clock(lambda: utils.hn_component_stride_draw(strip, u, hb, qb, x_cap=xc[0]))
    t_walk = clock(lambda: _walk(om, n, seed=1))

    print('\n%s, k=%d, %d paths, %d nodes' % (DEV, k, n, strip.nodes.numel()))
    print('  adaptive bound      %8.2f ms   (phi_max %g)' % (t_bound, bound))
    print('  coefficient strip   %8.2f ms' % t_coef)
    print('  origin block        %8.2f ms' % t_block)
    print('  batched Phi         %8.2f ms' % t_phi)
    print('  survival draw       %8.2f ms   (%.1f Phi-equivalents)' % (t_draw, t_draw / t_phi))
    print('  the daily walk      %8.2f ms   <- what the draw would replace' % t_walk)
    print('  draw / walk         %8.1fx      crossover near k = %d daily steps' % (
        t_draw / t_walk, int(k * t_draw / t_walk)))
    # THE INVARIANT IS THE ITERATION COUNT, not the clock: two Phi-equivalents an iteration against
    # a cap of 32 puts a non-converging loop at ~64, so this separates "converging" from "running
    # the loop out" and nothing finer. The same draw reads 8.5x idle and 39x under load.
    assert t_draw / t_phi < 50.0, t_draw / t_phi


# ======================================================================================
# GATE 6 - the batched layer: one state factor, one polished root, two routes to second order
# ======================================================================================

def _factor_world(k=21, n=256, panels=64):
    """A strip and a cube of states for the factorisation gates - `n` paths off one fixing."""
    strip = _strip(k, panels=panels)
    g = torch.Generator(device=DEV).manual_seed(19)
    h = _t(H0) * torch.exp(0.3 * torch.randn(n, generator=g, dtype=DTYPE, device=DEV))
    q = _t(Q0) * torch.exp(0.1 * torch.randn(n, generator=g, dtype=DTYPE, device=DEV))
    u = torch.rand(n, generator=g, dtype=DTYPE, device=DEV) * 0.98 + 0.01
    return strip, h, q, u


def test_the_state_factor_is_the_only_state_dependent_object():
    """`exp(A + B h + C q)` built ONCE and handed down is the same tensor as rebuilding it, so
    every contraction over the cache answers bit for bit either way.

    That is the whole content of the factorisation: the survival Phi, the density, each Newton
    residual and both polish steps were five rebuilds of one object per truncated draw. Sharing it
    is a saving and not a re-marking, which is what `torch.equal` here says.
    """
    strip, h, q, u = _factor_world()
    e = utils.hn_component_stride_factor(strip, h, q)
    x = utils.hn_component_stride_invert(strip, u, h, q)
    assert torch.equal(utils.hn_component_stride_cdf(strip, x, h, q),
                       utils.hn_component_stride_cdf(strip, x, h, q, e)), 'the shared Phi moved'
    assert torch.equal(utils.hn_component_stride_pdf(strip, x, h, q),
                       utils.hn_component_stride_pdf(strip, x, h, q, e)), 'the shared density moved'
    assert torch.equal(x, utils.hn_component_stride_invert(strip, u, h, q, factor=e)), \
        'the shared inversion moved'
    cap = _t(np.log(1.10))
    a, _ = utils.hn_component_stride_draw(strip, u, h, q, x_cap=cap)
    b, _ = utils.hn_component_stride_draw(strip, u, h, q, x_cap=cap, factor=e)
    assert torch.equal(a, b), 'the shared draw moved'


def test_the_cumulant_seed_is_what_keeps_the_polish_short():
    """The Cornish-Fisher warm start against a cold one, in standard deviations of the law.

    The cache already holds the exact first four cumulants, so the seed costs three multiply-adds a
    path and lands a fraction of a standard deviation from the root; the mean of the law - the cold
    start a bracket alone would give - is a whole quantile away. What it buys is the iteration
    count, the only part of the stride that is not a single pass over the cache.
    """
    strip, h, q, u = _factor_world()
    root = utils.hn_component_stride_invert(strip, u, h, q)
    mean, var, g1, g2 = utils.hn_component_stride_cumulants(strip, h, q)
    sd = var.sqrt()
    z = utils.norm_icdf(u)
    w = (z + (z * z - 1.0) * g1 / 6.0 + (z ** 3 - 3.0 * z) * g2 / 24.0
         - (2.0 * z ** 3 - 5.0 * z) * g1 * g1 / 36.0)
    warm = float(((mean + sd * w - root).abs() / sd).max())
    cold = float(((mean - root).abs() / sd).max())
    assert warm < 0.35 * cold, (warm, cold)
    print('\nseed distance in sd: Cornish-Fisher %.3f against the law mean %.3f' % (warm, cold))


def test_an_unpolished_root_is_refused_rather_than_returned():
    """The inversion REFUSES a root it has not resolved, because everything downstream
    differentiates that equation AT a solution of it. Starved of iterations, it raises by name.

    AND IT REFUSES THE RIGHT PATHS. The breaks inside the loop are COLLECTIVE, so one stalled path
    holds every other open - and the population that stalls is not a tail root at the quadrature's
    floor, it is a path asked for a quantile the law does not reach, whose bracket the widening loop
    then walks off to an |x| of 1e3 or more. That answer is a pre-existing saturation of the deep
    tail; refusing the WHOLE call for it kills a valuation over paths of weight 1e-10. This gate
    walks the band the pricer itself aims at - `pricing.HN_STRIDE_BOUND_SD` clamps every survival
    cap to mean - 8 sd - and reads that it is returned, that the paths still open there are exactly
    the ones thrown outside the law, and that a starved solve is refused all the same.
    """
    strip, h, q, u = _factor_world()
    utils.hn_component_stride_invert(strip, u, h, q)              # converges at the default 32
    with pytest.raises(ValueError) as exc:
        utils.hn_component_stride_invert(strip, u, h, q, iters=1)
    assert 'did not converge' in str(exc.value) and 'implicit function theorem' in str(exc.value)

    # the reachable band: the deepest cap the pricer can aim, at the production panel count
    deep = _strip(126, panels=pricing.HN_STRIDE_PANELS)
    g = torch.Generator(device=DEV).manual_seed(77)
    n = 1 << 14
    hh = _t(H0) * torch.exp(0.3 * torch.randn(n, generator=g, dtype=DTYPE, device=DEV))
    qq = _t(Q0) * torch.exp(0.1 * torch.randn(n, generator=g, dtype=DTYPE, device=DEV))
    uu = torch.rand(n, generator=g, dtype=DTYPE, device=DEV)
    mean, var, _, _ = utils.hn_component_stride_cumulants(deep, hh, qq)
    sd, worst, thrown = var.sqrt(), 0.0, 0
    for n_sd in (4.0, 5.0, 6.0, pricing.HN_STRIDE_BOUND_SD):
        cap = (mean - n_sd * sd).detach()
        x, phi_cap = utils.hn_component_stride_draw(deep, uu, hh, qq, x_cap=cap)
        assert torch.isfinite(x).all(), 'the draw at mean - %g sd is not finite' % n_sd
        open_ = (utils.hn_component_stride_cdf(deep, x, hh, qq) - uu * phi_cap).abs() >= 1e-14
        out = (x - mean).abs() > 20.0 * sd
        assert not bool((open_ & ~out).any()), \
            'a path INSIDE the law stalled at mean - %g sd, which is the case that must refuse' \
            % n_sd
        worst, thrown = max(worst, int(open_.sum())), max(thrown, int(out.sum()))
    assert worst and thrown, 'no band here stalls at all, so this gate no longer reads what it ' \
                             'is for: %d open, %d thrown' % (worst, thrown)
    print('\nrefusal: the pricer\'s own band down to mean - %g sd is returned; the %d paths still '
          'open there are all outside the law (%d thrown past 20 sd), and none inside it stalls'
          % (pricing.HN_STRIDE_BOUND_SD, worst, thrown))


def _ift_terms(strip, root, h, q):
    """The five partials of `F(x, h)` at the root that the second-order IFT contracts."""
    xv = root.detach().clone().requires_grad_(True)
    hv = h.detach().clone().requires_grad_(True)
    F = utils.hn_component_stride_cdf(strip, xv, hv, q)
    f = torch.autograd.grad(F.sum(), xv, create_graph=True)[0]
    F_h = torch.autograd.grad(F.sum(), hv, create_graph=True)[0]
    F_xx = torch.autograd.grad(f.sum(), xv, retain_graph=True)[0]
    F_xh = torch.autograd.grad(f.sum(), hv, retain_graph=True)[0]
    F_hh = torch.autograd.grad(F_h.sum(), hv)[0]
    return f.detach(), F_h.detach(), F_xx.detach(), F_xh.detach(), F_hh.detach()


def test_the_second_derivative_of_the_draw_is_the_second_order_ift():
    """TWO ROUTES to `d2x/dh2`, and the named one-step mutant between them.

    Route A double-backwards through the drawn return - the two attached polish steps as the pricer
    runs them. Route B contracts the second-order implicit function theorem explicitly off the same
    cache:

        x'  = -F_h / f
        x'' = -(F_xx x'^2 + 2 F_xh x' + F_hh) / f

    THE MUTANT IS ONE STEP. Off a DETACHED root a single attached correction returns `-F_hh/f`,
    the right FIRST derivative with `-F_xx F_h^2/f^3 + 2 F_xh F_h/f^2` dropped at second order - an
    O(1) term, not a small one. Two steps start from an `x` that already carries `dx` and close it.
    This reads all three: A == B, mutant == -F_hh/f, and the gap between them IS the contraction.

    MEASURED: the two routes agree to 4.1e-15 relative, and what one step drops is 349% of the
    answer - the dropped contraction is larger than the term the mutant keeps, so a one-step polish
    does not give a small error in the second derivative, it gives the wrong sign of one.
    """
    strip, h, q, u = _factor_world(n=64)
    hv = h.detach().clone().requires_grad_(True)
    x, _ = utils.hn_component_stride_draw(strip, u, hv, q)
    d1 = torch.autograd.grad(x.sum(), hv, create_graph=True)[0]
    d2 = torch.autograd.grad(d1.sum(), hv)[0]

    root = utils.hn_component_stride_invert(strip, u, h, q)
    f, F_h, F_xx, F_xh, F_hh = _ift_terms(strip, root, h, q)
    xp = -F_h / f
    xpp = -(F_xx * xp ** 2 + 2.0 * F_xh * xp + F_hh) / f
    scale = float(xpp.abs().max())
    assert float((d1 - xp).abs().max()) / float(xp.abs().max()) < 1e-9, 'first order disagrees'
    two = float((d2 - xpp).abs().max()) / scale
    assert two < 1e-6, 'the two routes to second order disagree: %.3g relative' % two

    # the mutant, built from the same public verbs: ONE attached step off the detached root
    hm = h.detach().clone().requires_grad_(True)
    dens = utils.hn_component_stride_pdf(strip, root, hm, q).detach()
    xm = root + (u - utils.hn_component_stride_cdf(strip, root, hm, q)) / dens
    m1 = torch.autograd.grad(xm.sum(), hm, create_graph=True)[0]
    m2 = torch.autograd.grad(m1.sum(), hm)[0]
    assert float((m1 - xp).abs().max()) / float(xp.abs().max()) < 1e-9, \
        'the mutant was supposed to keep first order'
    assert float((m2 + F_hh / f).abs().max()) / scale < 1e-6, \
        'the mutant is not the term it is named for'
    dropped = -F_xx * xp ** 2 / f + 2.0 * F_xh * F_h / f ** 2
    assert float((d2 - m2 - dropped).abs().max()) / scale < 1e-6, \
        'the gap between one step and two is not the contraction it should be'
    size = float(((d2 - m2).abs() / d2.abs().clamp_min(1e-300)).max())
    assert size > 0.1, 'the dropped term is not O(1) on this fixture, so this gate reads nothing'

    # AND THE POLISH CLAMP DOES NOT BIND. Each correction is bounded by one standard deviation of
    # the law, a guard against a density underflowed to zero in a deep tail - but WHERE IT BINDS its
    # gradient is zero and both routes above silently become something else, so the theorem this
    # gate reads holds only while the corrections stay numerically zero.
    corr = ((u - utils.hn_component_stride_cdf(strip, root, h, q)).abs()
            / utils.hn_component_stride_pdf(strip, root, h, q).clamp_min(1e-300))
    bind = float((corr / utils.hn_component_stride_cumulants(strip, h, q)[1].sqrt()).max())
    assert bind < 1e-9, 'the polish clamp is within %.3g of binding, which zeroes its own ' \
                        'gradient and makes both readings above meaningless' % bind
    print('\nsecond order: two routes agree to %.2g relative; one step drops %.0f%% of it; the '
          'polish clamp sits %.1g of a standard deviation from binding' % (two, 100.0 * size, bind))


def test_the_cos_term_count_is_measured_against_the_production_strip():
    """STEP 4, MEASURED. A cosine series on the SAME cached recursion, its range off the same
    cumulants, against the Gauss-Legendre strip the stride actually runs.

    64-128 terms is not enough here: at the operating point (k = 21, monthly fixings) 128 terms read
    1.3e-5 against a production strip whose own error is 6.9e-7. 256 terms read 1.6e-9 - better than
    512 Gauss-Legendre nodes at HALF the count, which is the prize and it is 2x, not the 4-8x a
    64-term series would have been. THE RANGE IS PER CONTRACT: `mean +/- 8 sd` is
    `pricing.HN_STRIDE_BOUND_SD`, the reach the saturation already declares, and widening it to 12
    costs two orders (4.2e-7 at 256 terms) because the top frequency falls with the range.

    Nothing runs on this path. It is the measurement the decision needs, gated so it stays true.
    """
    k = 21
    om = list(OMEGAS[:k])
    ref = utils.hn_component_stride_strip(
        om, PRM, R_STEP, _t(0.15 * H0), _t(0.5 * Q0), panels=256)
    prod = utils.hn_component_stride_strip(
        om, PRM, R_STEP, _t(0.15 * H0), _t(0.5 * Q0), phi_max=ref.phi_max, panels=64)
    n = 2048
    g = torch.Generator(device=DEV).manual_seed(3)
    h = _t(H0) * torch.exp(0.4 * torch.randn(n, generator=g, dtype=DTYPE, device=DEV))
    q = _t(Q0) * torch.exp(0.1 * torch.randn(n, generator=g, dtype=DTYPE, device=DEV))
    with torch.no_grad():
        mean, var, _, _ = utils.hn_component_stride_cumulants(ref, h, q)
        sd = var.sqrt()
        x = mean + sd * torch.linspace(-4.0, 4.0, n, dtype=DTYPE, device=DEV)
        exact = utils.hn_component_stride_cdf(ref, x, h, q)
        gl = float((utils.hn_component_stride_cdf(prod, x, h, q) - exact).abs().max())

        def cos_err(terms, sd_half):
            half = sd_half * float(sd.max())
            a, b = float(mean.min()) - half, float(mean.max()) + half
            kk = torch.arange(terms, dtype=DTYPE, device=DEV)
            uu = kk * np.pi / (b - a)
            A, B, C = utils.hn_component_abc(uu * 1j, om, *PRM, R_STEP)
            logcf = A + B * h.unsqueeze(-1) + C * q.unsqueeze(-1)
            coef = (torch.exp(logcf) * torch.exp(-1j * uu * a)).real * (2.0 / (b - a))
            coef = torch.cat([0.5 * coef[..., :1], coef[..., 1:]], dim=-1)
            t = x.unsqueeze(-1) - a
            integ = torch.where(kk > 0, torch.sin(kk * np.pi * t / (b - a))
                                * (b - a) / (kk.clamp_min(1) * np.pi), t)
            return float(((coef * integ).sum(-1) - exact).abs().max())

        rows = {terms: cos_err(terms, 8.0) for terms in (64, 128, 256)}
        wide = cos_err(256, 12.0)
    print('\nk=21  Gauss-Legendre 512 nodes %.1e | COS at +/-8 sd  ' % gl
          + '  '.join('N=%d %.1e' % (t, e) for t, e in sorted(rows.items()))
          + '  | COS 256 at +/-12 sd %.1e' % wide)
    assert rows[128] > gl, '128 COS terms now beat the production strip - re-read the claim'
    assert rows[256] < gl, '256 COS terms no longer beat 512 GL nodes, which was the whole prize'
    assert wide > rows[256], 'the range stopped being per-contract - a wider one costs terms'
