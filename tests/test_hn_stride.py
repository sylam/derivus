"""THE STRIDE - the k-step conditional law of ln S given (h, q), cached, drawn and carried.

A simulation that observes the spot only on a monthly fixing schedule still walks every trading day
in between, because the daily recursion is the only thing that knows how to move (h, q). The stride
writes the k-step law down instead of walking it: the SAME backward A/B/C recursion run k steps is
the k-step conditional log-CF exactly, its coefficients depend on the parameters, the calendar
position and the transform node but NEVER on the state, so they are cached once and every per-path
question after that is exponentials and a dot product. `hn_component_cdf_logret` was the existing
half of this. What is NEW is the C*q state axis, the survival-truncated inversion, and the one thing
that is not exact - the state carried across the jump.

THE ONE APPROXIMATION IS THE CARRY, and it is quadratic by KIND, not by convenience. E[h_k | x] is
the news-impact curve, an asymmetric U tilted by gamma_1, so a linear carry is wrong in shape
however well it is fitted. `h_k = a + b*y + c*y^2` in the centered return, with (a, b, c) the L2
projection pinned by joint CUMULANTS read off ONE autograd chain on the cached recursion, and a
residual matched in variance and in the h-q residual correlation.

THE ORACLE HAS TWO ARMS, and the contract's single one could not have checked itself:

  ARM A, analytic and noise-free.  The exact E[h|x], Var(h|x), E[q|x], Var(q|x), Cov(h,q|x) by ONE-
    dimensional inversion of the u/v-differentiated joint transform.  The ratified oracle was a 2D
    inversion per path on the x-conditioned CF; a conditional MOMENT is a derivative at the origin,
    not an inversion in (u, v), so the (u, v) grid is not needed to get it - and dropping it drops a
    real hazard with it: an imaginary terminal condition puts the log(1 - 2b) branch on an axis
    `complex_log_unwrap` was never anchored for (its guarantee is a phi axis starting near 1+0j).
    Arm A runs the PRODUCTION contour and the production unwrap, differentiated at zero.
  ARM B, the exact conditional sampler.  The model's own daily recursion, paired to Arm A by
    x: the stride's carry is evaluated at the WALK's realised return, so a difference in any scored
    functional is the carry's fault and nothing else.

  The two agree inside three standard errors across the bulk of the return distribution - most bins
  inside one (the outer bins differ by the bin WIDTH, Arm A being the exact value at a bin centre
  and Arm B its average over a convex curve, and are excluded by name). That agreement is the check
  the ratified single oracle had no way to give itself.

WHAT EACH GATE HOLDS, and the numbers it measured on this workstation (RTX 3090, float64):

  1  Phi over the cache        BITWISE identical to `hn_component_cdf_logret` at equal
                               phi_max/panels/order - not a tolerance: the cache is the memoised
                               recursion and the assembly order is the same primitive's
     coefficient nesting       phi = 0 with L flat collapses the CACHED B strip onto `hn_ab`'s own B
                               to 8.9e-16 relative and A + C*L onto its A to 4.0e-15, at k = 1, 5,
                               21, 63, 126. This is STRONGER than the 1.5e-13 price-level nesting
                               precedent it inherits: the affine representation in h is unique, so
                               the two families must agree coefficient by coefficient, not just
                               after the quadrature has integrated them
     Phi vs the daily walk     the empirical CDF of 2^21 daily paths, every decile inside 3.5 SE
     the truncated draw        the daily walk's own survivors inside 3.5 SE of Phi/Phi_cap, and no
                               draw above the cap. The round trip through Phi is scored too, under
                               its own name: it is the INVERTER's check, not the law's, because
                               Phi(x_i)/Phi_cap == u_i holds by construction to 1e-14
     the un-shift              the step verb returns a spot in the DEAL's carry, not the strip's.
                               Drawn at k = 21 under b = 4r and scored against a daily walk that
                               takes b on its own steps: the survivor quantiles' median gap is
                               0.44x the walk's own band with the un-shift and 26.8x without it,
                               and the truncation reaches ln(1.10) exactly instead of stopping
                               0.005000 short of it. The survival WEIGHT is unmoved either way
     the calendar in A alone   two strips of equal length at different anchors on a SLOPING L curve
                               are BITWISE identical in B and C. k = 1 is calendar-free in A too -
                               omega reaches A only through D = B + C, which the (0, 0) terminal
                               condition makes zero at the first backward step - while the CARRY
                               reads the calendar at every k, its terminal condition being (u, v)
     the carry is a SHIFT      one strip serves any per-step cost of carry, INCLUDING a per-path
                               one, by moving the moneyness: 2.2e-16 on Phi, every carry loading
                               bitwise identical, only the first cumulant moving. Without this the
                               cache would need keying on b_step, which arrives per path
     blocking the inversion    the inverter blocks its path axis to bound the footprint and solves
                               the same roots to 7.1e-15 - not bitwise, because the convergence
                               break reads the worst residual in the batch
  2  k = 1 IS the daily        the one-step conditional mean is EXACTLY quadratic in the return
     advance                   (h_1 is a quadratic in z and x is affine in z), so the carry has zero
                               residual there and the stride reproduces the daily advance to 3.6e-16
                               on E[h|x] and E[q|x] alike, with every scored functional agreeing to
                               1e-15 and the carried state reconstructing
                               `hn_component_variance_step` off the drawn return. The case split is
                               an IDENTITY, not a convention
     the carry, scanned in k   the worst-k table below
     the two oracle arms       the analytic conditional moments against the daily sampler, binned on
                               exact quantiles: every interior bin inside 3 SE
     MUTATION, c := 0          the news-impact channel is visible and the harness sees it: the
                               conditional-mean error roughly DOUBLES at every rung of the scan
                               (2.9e-1 -> 5.8e-1 at k = 21) and goes from 3.6e-16 to 1.1e-1 at
                               k = 1, so the mutant fails the live gate's own recorded bound
                               everywhere. Zeroing c on the TRIGGER PROBABILITY alone makes its bias
                               SMALLER by cancellation, which is why the mutation is scored where
                               the channel lives and not where it is convenient
     stride halving            two strides of k/2 cut every h quantile error, at every k tried
     a WHOLE SCHEDULE          six monthly fixings with an up-barrier, against the walk's own
                               survival count: -1.0%, compounded through six carries. And the
                               stride's standard error is 0.55x the walk's on the same paths,
                               because it never leaves the survival set - which is the OSS
                               consumer's case stated as a number
     the floor mass            declared per k, and it is THE FINDING (below)
  3  differentiability         the reattached gradient of Phi and of the drawn return against CRN
                               central differences: 0.00% disagreement on a ladder flat to 0.00%,
                               through a parameter and through the state. The root is found under
                               no_grad and the graph put back by one Newton step, which is the
                               implicit function theorem written as arithmetic
     the origin block          autograd vs central differences of the transform itself: 5.3e-10 on
                               the phi row, 6.0e-9 on the u row, and dB/dv EXACTLY zero - E[q_{t+k}]
                               does not depend on h_t, a structural fact the block reproduces as a
                               hard zero rather than as a small number
     no_grad                   a cache built inside a valuation's inference block is BITWISE the
                               one built outside it, and comes back detached. IT WAS NOT: the block
                               is obtained by DIFFERENTIATING, so under no_grad every partial came
                               back a structural zero and the carry divided by mu2 = 0. Found by the
                               schedule gate as NaN on all 8,192 paths at the FIRST stride, beside a
                               perfectly healthy Phi
     the residual covariance   symmetric to 1e-13 between its two independent spellings, which is
                               what says the projection is orthogonal and not merely solved
  4  the plain path            `hn_component_abc` with no terminal condition is bitwise what it was;
                               the plain HN family is not touched at all
  5  wall clock                below

THE WORST-K TABLE - the declared approximation carrying its number. k is trading days; every
column is the stride against the oracle, at h_0 = 1.6x the stationary level, phi share 0.35,
rho 0.99, 2^18 paired paths, on the 41-node exact quantile grid of the return:

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

  and the long-run component q, which is carried an order of magnitude better throughout: its
  conditional-mean error peaks at 2.1e-2 and its residual never reaches the floor at any k.

THE ERROR IS NON-MONOTONE IN k, exactly as the design says it must be: it vanishes at k -> 0 (where
the carry is exact), it forgets x as k -> infinity (the stationary law is matched exactly by
construction), and it is WORST in between - measured worst at k = 24-25 (monthly, k = 21, at 98% of the peak - 2.2x the
metric, 11.2-trading-day short-run half-life; the half-life-coincidence claim is withdrawn). Monthly autocall
fixings ARE the operating point, which is why this gate scans k and never spot-checks one stride.

THE FLOOR MASS IS THE FINDING. The residual is Gaussian and the state is floored, per the ratified
design and the declared-floor precedent - and the residual's coefficient of variation sd_h/E[h]
reaches 0.20 at k = 21 and 0.64 at k = 252, so a Gaussian puts 1.1% and 8.0% of its mass below zero
respectively, onto a floor of 1e-12 (0.16 bp of annualised vol - a frozen path). The SMOOTH
functionals barely notice (0.05% on the trigger probability, 0.24% on the KI put at k = 21) because
the floored paths are the low-variance ones a put and a trigger weight least. The h TAIL QUANTILES
are destroyed: the 5% quantile is -18% at k = 21 and the 1% quantile is ON the floor from k = 21
onward. That is vol-of-vol convexity, it is exactly what this gate was told to score, and it is the
trigger for escalation rung (1) - map the residual through the exact 1D h-marginal quantiles, which
are invertible off the same cached coefficients.

WALL CLOCK, k = 21, RTX 3090 / float64:
    cache build          5.5 ms for the Phi coefficient strip at a GIVEN phi_max; 64 ms more to
                         resolve the bound by the adaptive scan; 245 ms more for the origin block.
                         The microsecond claim belongs to the per-path Phi, not to the build
    batched Phi          6.8 ms at 16,384 paths / 59 ms at 131,072, on a 512-node strip
    survival draw        73 ms / 555 ms at the same shapes - about 8 Phi-equivalents, which is what
                         the Cornish-Fisher seed buys (25 before it)
    the daily walk       15 ms / 18 ms for the 21 days the stride replaces

    These are IDLE-BOX readings at 512 nodes; `test_the_wall_clocks_are_recorded` prints its own at
    the gate's 1024-node grid, and prints rather than asserts them because this workstation shares
    its GPU with the rest of the suite - the same draw reads 8.5x one Phi idle and 39x under load.

    SO THE STRIDE IS NOT A WALL-CLOCK LEVER AT THESE SHAPES, and the report says so. The draw costs
    n_paths x n_nodes complex work against the walk's n_paths x k real work: the crossover is near
    k = 640 daily steps and no contract reaches it. The OSS consumer's case is the SURVIVAL
    CONDITIONING OF THE WHOLE INTERVAL rather than of its last day, not speed. The Phi-only
    consumers - the conditional-p jump gamma and HN branch-and-weight - have no daily-walk
    equivalent at all, and for them the cache is pure gain.

NO HMC ANYWHERE, and no monkeypatching: every oracle rides `utils.hn_component_variance_step` or
`utils.hn_component_abc` directly, so the model stays a single source of truth.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time

import numpy as np
import pytest
import torch

from derivus import utils
from crn_ladder import ladder
import hn_reference as hnref

DTYPE = torch.float64
SPY = 252.0
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
#: Paired paths for the oracle. The gate is a distributional comparison, so the count sets the
#: standard error every reading is quoted against; a CPU-only box runs the same gate smaller.
PATHS = (1 << 18) if DEV == 'cuda' else (1 << 15)
#: The scan. NOT a spot check: the carry error is non-monotone in k and its maximum sits in the
#: middle, at the fast component's half-life, which is where monthly fixings land.
K_SCAN = (1, 3, 8, 13, 21, 34, 55, 126)
#: The horizon the scored functionals look forward over from the fork - one monthly fixing.
K_FORWARD = 21

#: THE RECORDED CARRY ERROR, per k: the module table's `E[h|x] rel` column with 25% of headroom.
#: Tight on purpose - it is what the mutation gate breaks. Deleting the quadratic term roughly
#: DOUBLES the conditional-mean error at every rung (and takes k = 1 from 3.6e-16 to 1.1e-1), so a
#: bound sitting 25% above the fitted value is failed by the mutant at every rung and by nothing
#: else. A looser bound would pass a linear carry, which is the whole thing this gate exists to
#: refuse.
RECORDED_MEAN_REL = {1: 1e-13, 3: 2.7e-2, 8: 1.3e-1, 13: 2.6e-1, 21: 3.7e-1,
                     34: 3.4e-1, 55: 2.3e-1, 126: 9.4e-2}

#: The MEDIAN of the carried h against the oracle's, per k. The tails carry their own gates (the
#: floor mass, and the table); the median is the reading that says the bulk of the carried
#: distribution has not drifted, and it drifts UP with k as the Gaussian residual replaces a
#: right-skewed one.
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
    # h0 well above the long-run level, which is where a live book sits after a shock and where the
    # carry has the most work to do
    return prm, omegas, _t(0.02 / SPY), 1.6 * float(level), float(l_path[0])


PRM, OMEGAS, R_STEP, H0, Q0 = _world()
NESTED = (lambda p: (p[0], p[1], p[2], _t(0.99), _t(0.0), _t(-1234.5)))(
    utils.hn_component_from_plain(*(_t(PLAIN[k]) for k in
                                    ('omega', 'alpha', 'beta', 'gamma_star'))))
NESTED_LEVEL = utils.hn_component_from_plain(
    *(_t(PLAIN[k]) for k in ('omega', 'alpha', 'beta', 'gamma_star')))[3]


#: 128 panels = 1024 nodes, against the family's default 2048. The memory is O(paths x nodes) and
#: a 2048-node strip puts 2^18 paths past a 24 GB device, so every gate that drives a whole cube
#: halves the grid; the bitwise gate takes the default, because that is the grid it has to match.
#: The halving costs a probability under 1e-9. Do NOT quarter it: 512 nodes is 6.4e-13 at a bound of
#: 512 but 2.9e-8 at the wider bound a low-variance corner of the state box asks for, and the bound
#: is what the panel width has to resolve.
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
    """(x, h, q) after `len(omegas)` exact daily component steps from (h0, q0).

    Rides `utils.hn_component_variance_step` directly - the pair recursion stays the single source
    of truth, exactly as `hn_reference.hnc_simulate` does, and this is the sampler the whole carry
    is scored against. Returns the AGGREGATE log-return and the state it landed in, per path.

    `b` is the per-step cost of carry, defaulting to the `R_STEP` every other gate runs at. It is a
    parameter because `test_the_stride_step_un_shifts_the_carry` has to score a return drawn under
    one carry against a walk taken under ANOTHER, and it must do that on the model's own sampler
    rather than by applying the shift theorem the gate exists to check.
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

    ``E[h_k e^{i xi R}]`` is the u-derivative of the joint transform at the origin, so inverting it
    over xi and dividing by the density gives the conditional mean; the second derivatives give the
    conditional second moments. Everything is autograd on `utils.hn_component_abc` along the
    PRODUCTION contour with the PRODUCTION unwrap - the terminal condition is differentiated at
    zero, never evaluated away from it, so no branch of log(1 - 2b) moves.

    The bound is doubled off the price's own adaptive scan because the differentiated integrand
    carries a polynomial factor in xi and so decays slower than the CDF's; `test_the_conditional
    _moment_oracle_has_converged` reads the doubling.
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
    compounding channel, and the one an autocall actually reads.

    CHUNKED for the same reason `_ki_put` is: Phi over a (2^18, 1024) node strip is 4.3 GB of
    complex128 per temporary and this gate builds several. That is the stride's real constraint -
    memory, O(paths x nodes) - and a gate that drives a whole cube has to respect it."""
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
    # the loadings ride the cache's graph (the origin block is built under `create_graph`, which is
    # what makes the carry differentiable); the scan reports numbers, so it reads them detached
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

    `hn_component_cdf_logret` builds its Gauss-Legendre grid, runs the A/B/C recursion on it and
    assembles 1 - P2; the strip does the first two ONCE and the third per path. Same nodes, same
    weights, same assembly order, so given the same phi_max/panels/order the two differ by nothing
    at all - not by a quadrature tolerance. Measured: torch.equal is True at every k.
    """
    om = list(OMEGAS[:k])
    strip = _strip(k, panels=None, moments=False)          # the FAMILY default grid, deliberately
    # spread the test points over the k-step scale: a fixed grid is 19 standard deviations out at
    # k = 1 and the probability there rounds to a hard 0 or 1, which tests the float and not the CDF
    sd = float(np.sqrt(k * H0))
    x = torch.tensor([-2.5, -1.5, -0.5, 0.0, 0.4, 1.2, 2.2], dtype=DTYPE, device=DEV) * sd
    h = torch.tensor([0.3, 0.6, 0.9, 1.0, 1.4, 2.2, 3.4], dtype=DTYPE, device=DEV) * H0
    q = torch.tensor([0.7, 0.8, 0.9, 1.0, 1.1, 1.3, 1.6], dtype=DTYPE, device=DEV) * Q0
    cached = utils.hn_component_stride_cdf(strip, x, h, q)
    direct = utils.hn_component_cdf_logret(
        x, om, h, q, *PRM, R_STEP, phi_max=strip.phi_max)
    assert torch.equal(cached, direct), float((cached - direct).abs().max())
    assert bool(((cached > 0.0) & (cached < 1.0)).all())
    # and the reduced grid every gate below drives a cube on: 1024 nodes against the family's 2048,
    # which halves the memory for under 1e-9 of probability (512 would be 2.9e-8 - see GATE_PANELS)
    coarse = _strip(k, moments=False)
    assert float((utils.hn_component_stride_cdf(coarse, x, h, q) - cached).abs().max()) < 1e-9


@pytest.mark.parametrize('k', [1, 5, 21, 63, 126])
def test_the_cached_coefficients_nest_on_the_plain_family(k):
    """phi = 0 with L FLAT: the cached B strip IS `hn_ab`'s B, and A + C*L IS its A.

    STRONGER THAN THE PRICE-LEVEL NESTING it inherits (1.5e-13, test_hn_component gate 1). The
    representation `A + B*h + C*q` is affine in h and h is a free variable, so on the nested face -
    where q is pinned at L for every path - the two families cannot merely integrate to the same
    price, they must agree coefficient by coefficient. Nothing in the quadrature can hide a wrong
    coefficient here. Measured: 8.9e-16 on B and 4.0e-15 on A + C*L, at every k.
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
    """The cached Phi against the empirical CDF of the daily recursion it replaces.

    The bitwise gate above says the cache is the same INVERSION; this says the inversion is the same
    LAW the simulation walks. Deciles, 2^21 paths (2^15 on a CPU box), read against the binomial
    standard error of the empirical CDF at each one.
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

    Three readings, and only two of them are about the law. No draw may exceed the cap - the
    truncation is a hard constraint, not a bias. And the DAILY WALK'S OWN SURVIVORS must have the
    truncated CDF: that is the arm the LAW rests on, because it puts the model's own sampler against
    the inversion with nothing shared between them but the parameters.

    THE MIDDLE ARM IS THE INVERTER, NOT THE LAW, and it is kept under its right name rather than
    dropped. `Phi(x_i)/Phi_cap == u_i` holds BY CONSTRUCTION to the solve tolerance (1e-14), so
    pushing the drawn sample back through Phi and scoring it against the quantile grid is the
    Kolmogorov-Smirnov statistic of `torch.rand` composed with a round trip through the inverter. It
    reads that round trip - a returned root that is not the root, a Phi_cap normalisation applied
    twice or not at all, a draw that silently left its bracket - and those are real ways to get this
    wrong. What it cannot read is whether the law is the model's, because Phi is both sides of it.
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
    """The omega strip touches A and NOTHING else - which is what makes the cache calendar-ANCHORED
    rather than calendar-shaped, and is a property of the recursion worth pinning rather than
    reading off the source.

    Two strips of the same length at different anchors on a SLOPING L curve are BITWISE identical in
    B and C, because those recursions never reference omega_t. The consequence is the recorded
    escalation rung: a schedule of equal-length intervals can share one B/C pair and recompute A
    alone, and A is affine in the strip (dA/domega_n is the D coefficient of step n), so it is one
    autograd pass rather than a second recursion.

    AND k = 1 IS CALENDAR-FREE ENTIRELY, A included, which is not an accident: omega enters A only
    through D = B + C, and the FIRST backward step has D = 0 by the (0, 0) terminal condition. So a
    one-day Phi is the same object wherever it sits on the curve, and the k-step A reads only
    omegas[0:k-1]. The CARRY still reads all k of them - its terminal condition is (u, v), not
    (0, 0), so D is 1 at that first step, which is exactly why E[h_{t+1}] carries omega_t.
    """
    # the SAME node grid on both, or B and C would differ for the trivial reason that the adaptive
    # bound is itself a function of the omega slice
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

    `r` enters A only as `phi*r*k`, and the inversion multiplies that against `exp(-i phi x)`, so
    changing the carry by `d` is EXACTLY shifting the moneyness by `-d*k`. Measured: 2.2e-16 on Phi
    over a 25-point grid, 3.6e-15 on the A strip itself, and every carry loading BITWISE identical -
    only the first cumulant moves, and by exactly `d*k`. Without this the cache would have to be
    keyed on the carry too, or A would have to be rebuilt per deal.
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


#: The un-shift gate's own path counts. The signal is a RIGID translation of (b_step - r)*k, so it
#: does not shrink with the sample while the walk's own quantile band does - which is why this gate
#: can be honest at a count the draw's O(paths x nodes) footprint can hold.
UNSHIFT_PATHS = (1 << 17) if DEV == 'cuda' else (1 << 14)
UNSHIFT_REF = (1 << 21) if DEV == 'cuda' else (1 << 18)


def test_the_stride_step_un_shifts_the_carry():
    """The shift runs BOTH ways, and the step verb owes the return leg of it.

    `test_the_cost_of_carry_is_a_shift` pins the way IN: a deal carrying `b` reads the strip at
    moneyness `x - (b - r)*k`, so one cache serves every carry. What comes back is then a return in
    the STRIP'S measure, and `hn_component_stride_step` un-shifts it by `+(b - r)*k` before it moves
    a spot. Without that the survival WEIGHT is still exactly right and the CARRY is still exactly
    right - the loadings centre on the strip's own mean, which is why the state is carried at the
    unshifted return - while the whole survivor law sits `(b - r)*k` too low.

    At k = 21 and b = 4r that is +0.005000 in log-return, and the readings say what it costs:

      the barrier      the draw is capped at 0.09030968 instead of the ln(1.10) = 0.09531018 it was
                       handed - a truncation that stops 50 bp short of the barrier, and a knock-out
                       that can never reach its own trigger
      the law          the median gap between the survivor quantiles and the DAILY WALK's under the
                       same b goes from -4.919e-3 to +8.1e-5, against a walk's own band of 1.83e-4
                       over six replicas: 26.8x the band, down to 0.44x
      the weight       phi_cap reads 0.8143257 against the walk's own survival 0.814463, 0.51 SE -
                       unmoved, because the weight was never the broken part

    The walk here takes `b` on its own steps rather than the shift theorem's word for it, so the
    thing being checked is not also doing the checking.
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
        """The step verb over the whole sample, chunked for the same O(paths x nodes) reason
        `_ki_put` is. Spot starts at 1.0, so the returned spot IS the log-return's exponential."""
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
    # the walk's OWN band, at the stride sample's count: the MEDIAN signed gap across the quantile
    # grid, which is what reads a common-mode translation while per-quantile noise cancels
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

    NOT BITWISE, and the docstring says why: the Newton loop's convergence break reads the WORST
    residual in the batch, so a different blocking stops one iteration earlier or later. The roots
    agree to the solve tolerance - measured 7.1e-15 in x between a 5,000-path solve and the same
    paths in blocks of seven - and the batch shape is preserved through the blocking.
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
    """Arm A is only an oracle if its own quadrature has stopped moving, in BOTH directions.

    Two doublings, and they are not the same test. Doubling the bound at CONSTANT NODE DENSITY reads
    TRUNCATION - how much of the tail of the differentiated integrand is being thrown away, which
    matters here because the u-derivative carries a polynomial factor in xi and so decays slower
    than the CDF's own integrand. Doubling the density at a constant bound reads RESOLUTION of the
    e^{-i xi x} oscillation. Measured 1.4e-13 and 4.1e-13 on the conditional variance; a naive
    "double the bound and keep the panels" reads 7.0e-9 and is measuring the panels, not the bound.
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
    """The analytic conditional moments against the daily sampler, binned on EXACT quantiles.

    This is the check the ratified single oracle could not have given itself. Bin edges come from
    the cached inversion, so they are exact rather than empirical.

    ARM A IS SCORED AT THE BIN'S OWN MEAN x, NOT AT ITS CENTRE, and that is what makes a standard-
    error bar the right bar. Arm B is an AVERAGE of E[h|x] over the sample that fell in the bin;
    scoring Arm A at the bin centre puts that average against a point value, and the difference is
    the curvature term - measured here at 2.0e-3 to 2.5e-3 RELATIVE in h, and it is a BIAS: it does
    not shrink with the sample while the standard error does. At 2^18 paths it hides inside 3 SE
    (worst bin 2.3 - 2.6 across seeds), at 2^21 it does not - bin 3 reads 4.6 / 5.2 / 5.3 SE at
    seeds 11 / 12 / 13 here, and 6.66 SE on the review probe that found it. That is a gate whose
    verdict is set by its path count and by a convention of the harness, not by the carry's error.

    Evaluating Arm A at the bin's own mean cancels the term to first order at ANY path count - what
    is left is the third moment of x inside a bin - and the same 3 SE bar then holds at both counts:
    2.5 at 2^18 and 2.0 at 2^21, where the bin-centre form fails.

    The outer two bins on each side stay excluded by name. Re-centering does not rescue them: there
    the bin is wide and open-ended, and its own mean is the thing that has not converged.
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
    # Arm A at the bin's own mean. The excluded bins keep their centre - they are never scored, and
    # a bin's mean is only defined where the sampler actually put paths.
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

    h_1 is a QUADRATIC in z and the one-step return is AFFINE in z, so E[h_1 | x] is exactly
    quadratic in x and the residual is exactly zero: the quadratic carry is not an approximation at
    k = 1, it is the answer. That is what makes "daily-monitored contracts keep the daily path" a
    rule rather than a hedge - the two agree where they meet.
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

    Noise-free: Arm A gives E[h|x] in closed form on the exact quantile grid of the return, so this
    reads the approximation itself rather than a Monte Carlo estimate of it. The bound is the
    module's worst-k table, scaled: the error is non-monotone in k and peaks near the short-run
    half-life, and a rung that came in materially ABOVE its recorded value would mean the carry has
    changed, which is exactly what this must catch.
    """
    s = _scan(k)
    assert s['mean_rel_h'] < RECORDED_MEAN_REL[k], (k, s['mean_rel_h'], RECORDED_MEAN_REL[k])
    # q is the SLOW component and its loadings are small: it is carried an order of magnitude better
    assert s['mean_rel_q'] < 0.05, (k, s['mean_rel_q'])
    # the sign the design asks for as a free sanity check - see the deviation note below
    assert torch.sign(s['grid_loadings'].b_h[0]) == -torch.sign(PRM[2]), float(
        s['grid_loadings'].b_h[0])


def test_the_news_impact_slope_opposes_the_fitted_gamma():
    """THE FREE SANITY ASSERT, WITH ITS SIGN CORRECTED. The ratified text asks for
    `sign(b) == sign(fitted gamma)`; the news-impact algebra gives the OPPOSITE and this gate holds
    the algebra.

    h_{t+1} carries alpha*(z - gamma_1*sqrt(h))^2, whose z-slope at the origin is
    -2*alpha*gamma_1*sqrt(h), and the return is +sqrt(h)*z. So a POSITIVE gamma_1 - the standard
    equity leverage, down moves raising variance - makes E[h_k | x] DECREASING in x and b NEGATIVE.
    Reported as a deviation rather than taken as a choice; both signs of gamma_1 are swept here
    because the component family fits either (test_hn_component gate 6 lands gamma_1 at -845.1 on
    its negative-RR fixture).
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

    Paired: the stride carries at the ORACLE'S OWN realised return, so every difference here is the
    carry and not the draw. The three read different things and only together are they a gate - the
    trigger probability and the put are SMOOTH functionals and barely move (0.05% and 0.19% at
    k = 21), while the h tail quantiles, which is where vol-of-vol convexity lives, move by 18% at
    the 5% level. A gate that scored only the smooth two would pass a carry that had lost the tail.
    """
    s = _scan(k)
    bias, se, level = s['trigger']
    assert abs(bias) / level < 1.5e-3, (k, bias, level, bias / se)
    bias, se, level = s['ki_put']
    assert abs(bias) / level < 6e-3, (k, bias, level, bias / se)
    assert abs(s['h_quantile_rel'][2]) < RECORDED_MEDIAN_REL[k], (k, s['h_quantile_rel'])


@pytest.mark.parametrize('k', K_SCAN)
def test_zeroing_the_quadratic_term_breaks_the_gate(k):
    """MUTATION. Delete c - the quadratic term - and `test_the_carry_matches_the_exact_conditional
    _moments` MUST fail at every rung of the scan, which is what is asserted here: the mutant is
    scored against the SAME `RECORDED_MEAN_REL` bound the live gate holds, not against a synthetic
    ratio that could be tuned until it passed.

    Scored where the news-impact channel actually lives: on the conditional-mean curve, which is the
    thing c exists to bend, and on the h tail quantiles, which is what the bend moves. NOT on the
    trigger probability, where zeroing c makes the bias SMALLER by cancellation - that is exactly
    the trap a mutation test is for. A linear carry is not a slightly worse quadratic one, it is
    the wrong SHAPE.

    Measured, max relative error of the conditional mean, fitted -> mutant: 3.6e-16 -> 1.1e-1 at
    k = 1 (where the quadratic carry is EXACT and the linear one is not even close), then
    2.1e-2 -> 1.5e-1, 1.0e-1 -> 3.1e-1, 2.1e-1 -> 4.7e-1, 2.9e-1 -> 5.8e-1, 2.7e-1 -> 5.4e-1,
    1.8e-1 -> 4.0e-1, 7.5e-2 -> 1.2e-1. About a factor of two everywhere the carry is approximate,
    and twenty-seven orders of magnitude where it is not.
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
    # mass of 0.5% the 5% quantile of the carried h is already ON the floor (-100% relative), so a
    # mutant cannot be measured as WORSE there - the floor has saturated the instrument, not fixed
    # the carry. Where it can read - k = 3, 8, 13, 21 - it does, and the conditional-mean arm above
    # reads at every rung regardless.

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
    second onto the walk's own second-half return but FROM THE CARRIED STATE, so the two strides
    compound their approximations exactly as a real schedule would. Measured at k = 21: the h
    quantile errors fall from (-18.1%, +2.6%, -1.6%) to (-5.0%, +1.0%, -0.4%).
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
    """END TO END, and the closest thing here to what a wave-2 OSS pricer will actually price.

    Six monthly fixings, an up-barrier at 1.10, and the two schemes run side by side:

      the WALK   observes the spot on the fixing dates only and counts the paths that never
                 finished a fixing above the barrier - the product's own definition;
      the STRIDE never leaves the survival set at all. Each fixing draws its return from the law
                 TRUNCATED at the barrier and carries the surviving weight `Phi_cap` in a running
                 product, so `E[prod Phi_cap]` is that same probability by construction.

    This exercises `hn_component_stride_step` - draw, carry, weight - six times in a row, so the
    carry's error COMPOUNDS through the strip the way a real schedule makes it.

    MEASURED: walk 0.467010 +/- 0.002756, stride 0.462364 +/- 0.001510 at 2^15 paths - a bias of
    -1.00%, which is the number to quote when anyone asks what the carry costs on a real product.
    Note the OTHER column: the stride's standard error is 0.55x the walk's on the same path count,
    because every stride path is inside the survival set by construction while the walk spends more
    than half of its paths leaving it and reads the answer off the remainder. THAT is the OSS
    consumer's case - variance, not wall clock - and this gate is where it is visible.

    This gate also found the defect `test_a_cache_built_under_no_grad_is_the_same_cache` now pins.
    Nothing shorter than a full schedule would have: the single-stride gates all ran in grad mode.
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
    """THE COST OF THE RATIFIED RESIDUAL, held to its measured number rather than hidden.

    The residual is Gaussian and the state is FLOORED, which is the design and the declared-floor
    precedent (`HN_COMPONENT_VARIANCE_FLOOR`, the same floor the daily recursion carries because the
    CJOW pair has no positivity guarantee at phi > 0). But the residual's coefficient of variation
    sd_h/E[h] grows with k - 0.20 at k = 21, 0.64 at k = 252 - and a Gaussian with that spread puts
    1.1% and 8.0% of its mass below zero, onto a floor of 1e-12 per step, which is 0.16 basis points
    of annualised vol: those paths are frozen, not merely slow.

    This gate exists so the number cannot drift silently. It is also the trigger for escalation
    rung (1) - map the residual through the exact 1D h-marginal quantiles, which are invertible off
    these same cached coefficients - and rung (1) is what deletes the floor mass rather than
    bounding it.
    """
    s = _scan(k)
    recorded = {1: 0.0, 3: 0.0, 8: 0.001, 13: 0.005, 21: 0.015, 34: 0.033,
                55: 0.055, 126: 0.075}
    assert s['floor_h'] <= recorded[k], (k, s['floor_h'], recorded[k])
    assert s['floor_h'] == pytest.approx(s['negative_h'], abs=1e-9)   # the floor IS the negatives
    assert s['floor_q'] == 0.0                     # q is slow: its residual never reaches the floor


def test_the_worst_k_table_is_reported():
    """Print the scan. The declared approximation carries its numbers, and they are here, not in a
    docstring that was true once."""
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
    # non-monotone in k, and the maximum is INTERIOR - the design's central claim about the error
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
    """dPhi/d(alpha) through the cache, against CRN central differences on an h^2 ladder.

    Phi carries no randomness, so 'common random numbers' costs nothing here and the ladder reads a
    clean limit: measured 0.00% disagreement at a flatness of 0.00% across six rungs.
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

    THE ROOT IS FOUND UNDER no_grad, so this is the gate on the reattachment: one graph-carrying
    Newton step at the root puts back exactly `-(dQ/dtheta - u dPhi_cap/dtheta)/q`, the implicit
    function theorem for `Q(x) = u*Phi_cap`. Under CRN the SAME u drives the bumped draws, so the
    difference measures the same derivative. Measured 0.00% on both, ladders flat to 0.00% - a
    differentiated ITERATION would not do this, and neither would a detached draw.
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

    The whole carry rests on 13 mixed partials of `A + B*h + C*q` at (phi, u, v) = 0, taken by
    autograd on the cached recursion so that nothing hand-writes a moment recursion. This checks
    them against the closed form itself, differenced: 5.3e-10 on the phi row and 6.0e-9 on the u
    row, both FD truncation.

    dB/dv is EXACTLY zero and asserted as a hard zero, not a tolerance: E[q_{t+k}] is driven by
    omega and rho alone and cannot depend on h_t, so the block reproducing that as 0.0 rather than
    as 1e-19 says the algebra is structural and not merely small.
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
    DIFFERENTIATING - so the block has to force its own mode or it comes back as a block of zeros.

    Found by the end-to-end schedule gate, not by reasoning: with the mode left to the caller, a
    six-fixing schedule returned NaN on all 8,192 paths at the FIRST stride while its Phi sat there
    looking perfectly healthy (0.832, in range, no NaN). The zeros propagate as mu2 = 0, the carry's
    Gram determinant goes to 0/0, and every loading is NaN. The failure is silent at the point of
    the mistake and loud four layers downstream, which is the shape of defect a gate has to catch
    once and pin forever.

    Both directions are pinned: the block is BITWISE the same under either mode, and it comes back
    DETACHED under no_grad (the caller asked for no graph and must not be handed one).
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
    """THE FREE CONSISTENCY CHECK the carry's own algebra offers, taken.

    The h-q residual covariance can be written two ways - `Cov(h,q) - (b_q*M_up + c_q*M_upp)` from
    h's side and `Cov(h,q) - (b_h*M_vp + c_h*M_vpp)` from q's - because a projection residual is
    orthogonal to the span whichever variable is projected. They are DIFFERENT expressions in
    different cached moments and they agree to 1e-13 relative, which says the L2 projection is
    genuinely orthogonal and not merely a linear solve that happened to converge.
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
    """`hn_component_abc` with no terminal condition is BITWISE what it was before the stride.

    The one edit to an existing function in this build is two defaulted keyword values and a
    branch that the default path does not enter. Passing an explicit zero terminal must reproduce
    the default bitwise (it is the same arithmetic, plus an add of zero), and the plain HN family -
    `hn_ab`, `hn_cdf_logret`, `hn_call` - is not touched at all, which the nesting gate above reads
    from the other side.
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

    The design claimed the cache build in microseconds. It is not: the coefficient strip is 5.5 ms
    at a given bound, the adaptive bound another 64 ms, and the origin block 245 ms more, at k = 21
    on this box. Microseconds is the per-path Phi, which is the thing the consumers repeat.

    And the draw is SLOWER than the daily walk it would replace, by 30x at 131,072 paths - it costs
    n_paths x n_nodes complex work against the walk's n_paths x k real work, so the crossover is
    near k = 640 daily steps and no contract reaches it. That is a finding, not a failure: the OSS
    consumer's case is that the WHOLE interval is survival-conditioned rather than only its last
    day, and the Phi-only consumers have no daily-walk equivalent at all.
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
    # a cap of 32 puts a loop that never converges at ~64, so this separates "converging" from
    # "running the loop out" and nothing finer. It has to: this box shares its GPU with the rest of
    # the suite and the same draw reads 8.5x idle and 39x under load.
    assert t_draw / t_phi < 50.0, t_draw / t_phi
