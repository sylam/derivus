"""The one refusal `pricing.boundary_weights` makes, and whether it can be made at all.

The shipped guard read

    usable = denominator.abs() > 1e-30 * (s0 * s2).abs().clamp_min(1e-300)

and its docstring said it stopped "a near-singular solve turning that into a large arbitrary
number". It never did, and could not: `denominator / (s0 * s2)` is `1 - s1^2 / (s0 * s2)`, which
Cauchy-Schwarz pins into [0, 1] for every population of every scale, so a threshold of 1e-30 sits
twenty-five decades below anything reachable. In float64 the quantity is either EXACTLY zero -
nothing admitted at all - or above the ulp of `s0 * s2`, so the condition was equivalent to
`denominator != 0` and refused only the empty kernel. The large arbitrary numbers went through.

WHAT ACTUALLY GOES WRONG is visible in closed form. A kernel admitting exactly two points, `a` and
`b` widths from the boundary, has local-linear weights `b/(b - a)` and `-a/(b - a)`: they sum to
one, as they must, by cancelling two enormous numbers. Measured on the Heston-Nandi barrier at 512
paths, seed 1: two points 1.20 widths out and 0.021 widths apart, weighted +50.4 and -49.5. Their
jumps differed, so they did not cancel, and that one decision supplied 112% of the coefficient -
the whole of a 73% gradient error. `||weights||_1` is exactly that amplification, and it is what
the guard now bounds (`pricing.BOUNDARY_MAX_AMPLIFICATION`).

Every test below is written so that DELETING the guard fails it. The reachability test is the one
the shipped code could never have passed: it constructs the population, measures the amplification
against its own closed form, and requires the refusal to happen.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pytest
import torch

import derivus
from derivus import pricing, utils
from crn_ladder import ladder

DTYPE = torch.float64
BOUND = pricing.BOUNDARY_MAX_AMPLIFICATION
# the engine default (calculation.py Boundary_AAD_Bandwidth); the populations below are built in
# units of the kernel width it produces, so nothing here depends on its value
BANDWIDTH = 0.01

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason='CRN oracle is float32-precision-limited off CUDA')


def two_point_gap(out, apart, n=512, bandwidth=BANDWIDTH):
    """A gap population whose kernel admits exactly two points, `out` widths from the boundary and
    `apart` widths apart - the measured shape, parameterised.

    The width is `bandwidth * gap.std()`, so the pair's own placement moves the spread that placed
    it; the loop is that fixed point, and it converges in a handful of passes because the pair is
    two points in n. The cloud is held clear of the boundary by 0.2, which is tens of widths at any
    bandwidth this estimator is run at, so it contributes to the SPREAD and not to the kernel."""
    cloud = torch.cat([torch.linspace(-1.0, -0.2, n // 2 - 1, dtype=DTYPE),
                       torch.linspace(0.2, 1.0, n // 2 - 1, dtype=DTYPE)])
    g = torch.cat([cloud, torch.zeros(2, dtype=DTYPE)])
    for _ in range(50):
        width = bandwidth * g.std()
        g = torch.cat([cloud, torch.tensor([out * width, (out + apart) * width], dtype=DTYPE)])
    return g


def cluster_gap(out, apart, per=64, n=512, bandwidth=BANDWIDTH):
    """The same two admitted positions, each holding `per` points instead of one.

    The amplification is unchanged - the solve still differences two means scaled by
    `1 / (b - a)` - while every individual weight is divided by `per`, which is what separates a
    bound on the L1 norm from a bound on the largest weight. The linspace inside each cluster is a
    hair of jitter so the population is not a repeated value."""
    cloud = torch.cat([torch.linspace(-1.0, -0.2, (n - 2 * per) // 2, dtype=DTYPE),
                       torch.linspace(0.2, 1.0, (n - 2 * per) // 2, dtype=DTYPE)])
    g = torch.cat([cloud, torch.zeros(2 * per, dtype=DTYPE)])
    jitter = torch.linspace(0.0, 1e-4, per, dtype=DTYPE)
    for _ in range(80):
        width = bandwidth * g.std()
        g = torch.cat([cloud, (out + jitter) * width, (out + apart + jitter) * width])
    return g


def unguarded(gap, bandwidth=BANDWIDTH, monkeypatch=None):
    """The same solve with the bound lifted - what the engine returned before this guard."""
    monkeypatch.setattr(pricing, 'BOUNDARY_MAX_AMPLIFICATION', float('inf'))
    out = pricing.boundary_weights(gap, bandwidth)
    monkeypatch.undo()
    return out


def amplification(weights):
    return float(weights.abs().sum())


# --------------------------------------------------------------- the fixture is the measured one

def test_the_two_point_kernel_reproduces_the_measured_pathology(monkeypatch):
    """Before asserting anything about the guard: the synthetic population has to be the thing
    that was measured, and its amplification has to be the closed form rather than a number that
    happens to come out. `(a + b) / (b - a)` is what the local-linear solve does to two points, so
    agreeing with it to a part in 1e-9 says the fixture is the mechanism and not a coincidence."""
    out, apart = 1.20, 0.021
    gap = two_point_gap(out, apart)
    _, weights = unguarded(gap, monkeypatch=monkeypatch)

    width = BANDWIDTH * gap.std()
    # the cloud's own kernel values underflow to denormals rather than to zero, so "admitted"
    # is measured against the weights that carry the solve
    carries = weights.abs() > 1e-12 * weights.abs().max()
    assert int(carries.sum()) == 2, (
        f'{int(carries.sum())} points carry weight, not two - the fixture is not the measured shape')
    assert float(weights.sum()) == pytest.approx(1.0, abs=1e-9), (
        'local-linear weights must sum to one; the amplification bound is only meaningful because '
        'they do')
    near, far = sorted(float(x / width) for x in gap[carries])
    assert (near, far) == pytest.approx((out, out + apart), rel=1e-6), (
        f'the admitted pair sits at {near:.4f} and {far:.4f} widths out, not {out} and '
        f'{out + apart}')
    assert amplification(weights) == pytest.approx(
        (2 * out + apart) / apart, rel=1e-9), 'the two-point closed form is not what came back'
    assert amplification(weights) > 100.0, (
        f'the measured pathology amplifies ~100x; this fixture reads '
        f'{amplification(weights):.1f} and is not reproducing it')


def test_the_determinant_the_old_constant_tested_could_never_reach_it():
    """Why the criterion had to change rather than be re-tuned.

    `denominator / (s0 * s2)` is a Cauchy-Schwarz ratio - it cannot leave [0, 1] - and in float64
    a cancelling difference is either exactly zero or an ulp of its operands, so it cannot land
    below ~1e-16 either. A 1e-30 threshold therefore refuses exactly the empty kernel and nothing
    else, which is what twenty-five decades of headroom buys.

    The pathological population is included in the sweep on purpose: it reads 7.5e-05, twenty-five
    decades ABOVE the threshold that was supposed to catch it."""
    populations = [two_point_gap(o, a) for o in (0.5, 1.2, 2.0) for a in (0.021, 0.1, 0.5)]
    torch.manual_seed(0)
    populations += [torch.randn(512, dtype=DTYPE) * scale for scale in (1e-9, 1.0, 1e9)]

    ratios = []
    for gap in populations:
        for bandwidth in (0.005, 0.01, 0.05, 0.2, 1.0):
            g = gap.detach()
            width = bandwidth * g.std()
            k = torch.exp(-0.5 * (g / width) ** 2)
            s0, s1, s2 = k.sum(), (k * g).sum(), (k * g * g).sum()
            ratios.append(float((s2 * s0 - s1 * s1) / (s0 * s2)))

    assert min(ratios) >= 0.0 and max(ratios) <= 1.0, (
        f'the Cauchy-Schwarz ratio left [0, 1]: {min(ratios):.3g}..{max(ratios):.3g}')
    assert min(r for r in ratios if r > 0.0) > 1e-20, (
        f'a non-zero ratio reached {min(r for r in ratios if r > 0.0):.3g} - if float64 can land '
        f'near 1e-30 after all, the old constant was reachable and this test is the wrong story')
    g = two_point_gap(1.20, 0.021)
    width = BANDWIDTH * g.std()
    k = torch.exp(-0.5 * (g / width) ** 2)
    s0, s1, s2 = k.sum(), (k * g).sum(), (k * g * g).sum()
    pathological = float((s2 * s0 - s1 * s1) / (s0 * s2))
    assert pathological > 1e-30 * 1e20, (
        f'the population that broke the gradient reads {pathological:.3g}, which the old guard '
        f'passed - it has to, or there was nothing to fix')


# ------------------------------------------------------------------------ the guard is REACHABLE

@pytest.mark.parametrize('apart,refused', [
    (0.5, False),    # amplification 5.8 - a two-point kernel can be perfectly well conditioned
    (0.2, False),    # 13.0
    (0.11, False),   # 22.8 - the last rung below the bound
    (0.09, True),    # 27.7 - the first rung above it
    (0.05, True),    # 48.0
    (0.021, True),   # 115.3 - the measured shape
    (0.005, True)])  # 481.0
def test_the_amplification_bound_is_reachable(apart, refused, monkeypatch):
    """The assertion the shipped code could not make: a population that trips the guard, and one
    that does not, distinguished by the declared constant and nothing else.

    Refusal is monotone in the amplification because it IS the amplification, so the sweep is a
    crossing rather than a scatter - which is what makes the constant a calibrated level and not a
    number that happened to pass. The two middle rungs bracket the declared constant to within
    11%, so a bound moved by more than that in either direction fails here."""
    gap = two_point_gap(1.20, apart)
    _, loose = unguarded(gap, monkeypatch=monkeypatch)
    _, weights = pricing.boundary_weights(gap, BANDWIDTH)

    assert (amplification(loose) > BOUND) == refused, (
        f'the fixture reads {amplification(loose):.4g} against a bound of {BOUND} - the sweep no '
        f'longer straddles the constant and gates nothing')
    if refused:
        assert bool((weights == 0.0).all()), (
            f'amplification {amplification(loose):.4g} exceeds {BOUND} and the solve was still '
            f'used: ||weights||_1 came back {amplification(weights):.4g}')
    else:
        assert torch.equal(weights, loose), (
            f'amplification {amplification(loose):.4g} is within {BOUND} and the weights moved - '
            f'the guard is refusing a healthy solve')


@pytest.mark.parametrize('per', [16, 64, 128])
def test_the_bound_is_on_the_L1_NORM_and_not_on_the_largest_weight(per, monkeypatch):
    """Why `||weights||_1` and not `max|weight|`, which reads the same thing on the fixture that
    motivated the guard and stops doing so the moment the kernel is populated.

    Spread the two admitted points into two CLUSTERS and the estimator is unchanged - it is still
    `(b * mean_a - a * mean_b) / (b - a)`, amplification and all - but every individual weight is
    divided by the cluster's size. At 64 points a side the population that broke the gradient reads
    `||weights||_1` = 115.3 with a largest weight of 0.91, so a bound on `max|weight|` admits it
    and measures nothing.

    Populated kernels are the normal case, not the contrived one: the TARF registration pools its
    (B, n_inner) gap into 131072 points and its largest weight never exceeded 0.002 across the
    whole calibration sweep, while its `||weights||_1` moved from 1.00 to 2.99."""
    gap = cluster_gap(1.20, 0.021, per=per)
    _, loose = unguarded(gap, monkeypatch=monkeypatch)
    _, weights = pricing.boundary_weights(gap, BANDWIDTH)

    assert float(loose.sum()) == pytest.approx(1.0, abs=1e-9)
    assert amplification(loose) == pytest.approx((2 * 1.20 + 0.021) / 0.021, rel=1e-3), (
        f'clustering changed the amplification to {amplification(loose):.4g} - the two forms are '
        f'no longer being compared on the same solve')
    assert float(loose.abs().max()) < BOUND, (
        f'the largest weight is {float(loose.abs().max()):.4g}, which a bound of {BOUND} would '
        f'also refuse - this population does not separate the two criteria')
    assert bool((weights == 0.0).all()), (
        f'||weights||_1 = {amplification(loose):.4g} and the solve was still used - the bound is '
        f'reading the largest weight rather than the norm')


def test_a_refused_decision_contributes_exactly_zero(monkeypatch):
    """What refusal means downstream. The correction is worth exactly zero in the FORWARD pass
    whatever the weights are, so the only reading that says anything is the backward one: the
    coefficient reaching `gap` has to be exactly 0.0, the same answer an empty kernel gives.

    Measured against the unguarded solve on the same population, which is where the defect lived -
    a coefficient two orders of magnitude larger than any healthy decision's."""
    gap = two_point_gap(1.20, 0.021).requires_grad_(True)
    jump = torch.linspace(-2.5, -0.5, gap.numel(), dtype=DTYPE)

    correction = pricing.stochastic_boundary_correction(gap, jump, BANDWIDTH)
    coefficient, = torch.autograd.grad(correction, gap)
    assert float(correction) == 0.0, 'the correction is not worth zero in the forward pass'
    assert bool((coefficient == 0.0).all()), (
        f'a refused decision still reaches backward() with max|coefficient| '
        f'{float(coefficient.abs().max()):.6g}')

    monkeypatch.setattr(pricing, 'BOUNDARY_MAX_AMPLIFICATION', float('inf'))
    loose, = torch.autograd.grad(
        pricing.stochastic_boundary_correction(gap, jump, BANDWIDTH), gap)
    assert float(loose.abs().max()) > 1.0, (
        'the unguarded solve contributes nothing here either, so this population is not the one '
        'that had to be refused')


# ------------------------------------------------------------------ a healthy solve is UNTOUCHED

@pytest.mark.parametrize('bandwidth', [0.005, 0.01, 0.05, 0.2, 1.0])
@pytest.mark.parametrize('seed', [0, 1, 2, 3, 4])
def test_a_healthy_population_is_bit_identical_to_the_unguarded_solve(bandwidth, seed, monkeypatch):
    """The failure mode to hunt hardest: refusing something legitimate. Bit-identical, not close -
    the guard either fires or it does not, and a healthy population is one where it does not.

    Also pins the identity the bound rests on. `||weights||_1` is 1 exactly while no weight is
    negative, so a reading of 1 is a Nadaraya-Watson kernel and anything above it is the
    first-order correction buying cancellation. Across these twenty-five populations it reads
    1.0000 to 1.0481 - the top of that range is the narrowest bandwidth, where the kernel holds
    fewest points, which is the direction the pathology lies in and nowhere near it."""
    torch.manual_seed(seed)
    gap = torch.randn(512, dtype=DTYPE)
    _, loose = unguarded(gap, bandwidth, monkeypatch=monkeypatch)
    _, weights = pricing.boundary_weights(gap, bandwidth)

    assert amplification(loose) < BOUND, (
        f'this population reads {amplification(loose):.4g} and is not the healthy control it is '
        f'being used as')
    assert torch.equal(weights, loose), 'the guard moved a healthy solve'
    assert 1.0 - 1e-12 <= amplification(weights) < 1.1, (
        f'||weights||_1 = {amplification(weights):.6g}; it cannot fall below one while the '
        f'weights sum to one, and a healthy kernel should sit essentially on it')


def test_the_criterion_is_scale_free():
    """A gap is a log-moneyness here, a swap value there and a collateral shortfall in the third,
    so a criterion carrying units would be calibrated on one registration and meaningless on the
    next. Scaling the population scales the weights' own scale away entirely: the kernel width
    scales with it, and `||weights||_1` is invariant to a part in 1e-12 over eighteen decades."""
    base = two_point_gap(1.20, 0.021)
    readings = [amplification(pricing.boundary_weights(base * scale, BANDWIDTH)[1])
                for scale in (1e-9, 1e-3, 1.0, 1e3, 1e9)]
    assert readings == [0.0] * 5, f'the refusal is not scale-invariant: {readings}'

    healthy = two_point_gap(1.20, 0.5)
    loose = [amplification(pricing.boundary_weights(healthy * scale, BANDWIDTH)[1])
             for scale in (1e-9, 1e-3, 1.0, 1e3, 1e9)]
    assert max(loose) - min(loose) < 1e-9 * max(loose), (
        f'the amplification of an admitted solve moved with the units: {loose}')


@pytest.mark.parametrize('bandwidth', [0.005, 0.01, 0.02, 0.05])
@pytest.mark.parametrize('shape', [(64,), (512,), (8192,), (16, 512)])
def test_the_verdict_does_not_move_with_the_population_s_size_shape_or_bandwidth(
        shape, bandwidth, monkeypatch):
    """The three axes the rest of the file holds fixed, and every one of them is live in the engine.

    A pooled gap is (B,) at a latched barrier and (B, n_inner) at a TARF - 131072 elements against
    512 - and the bandwidth is a JSON knob. `||weights||_1` depends on NONE of them: it is a ratio
    of the same solve, so the same population refused at one size, shape or bandwidth has to be
    refused at all of them. Four criteria that are wrong in exactly that way pass every other test
    here - one scaled by the sample size, one switched off above 1000 points, one with the bound
    scaled by the bandwidth, and one reducing on the last axis so a (B, n_inner) gap is refused per
    ROW while the solve it came from is pooled.

    0.1 is out of the sweep because the fixture stops being a two-point kernel there - the cloud
    sits 3.4 widths out and joins the fit - which is why the closed form is checked and not assumed.
    """
    n = int(np.prod(shape))
    for apart, refused in ((0.5, False), (0.09, True), (0.021, True)):
        gap = two_point_gap(1.20, apart, n=n, bandwidth=bandwidth).reshape(shape)
        _, loose = unguarded(gap, bandwidth, monkeypatch=monkeypatch)
        _, weights = pricing.boundary_weights(gap, bandwidth)

        assert amplification(loose) == pytest.approx((2 * 1.20 + apart) / apart, rel=0.1), (
            f'{shape} at bandwidth {bandwidth} reads {amplification(loose):.4g}, not the two-point '
            f'closed form - this population is no longer the one being compared across the axes')
        assert bool((weights == 0.0).all()) == refused, (
            f'||weights||_1 = {amplification(loose):.4g} against a bound of {BOUND}, and reshaping '
            f'the same population to {shape} at bandwidth {bandwidth} '
            f'{"admitted" if refused else "refused"} it - the criterion is reading the population '
            f'size, its shape or the bandwidth, none of which it is a function of')
        if not refused:
            assert torch.equal(weights, loose), 'the guard moved an admitted solve'


# ------------------------------------------------------------- the defect this was written for

def _hn_barrier(spot=100.0, gradient=False, batch=512, seed=1, mcmc=256):
    """The Heston-Nandi barrier CVA and its equity-spot gradient - the fixture whose gradient the
    unguarded solve broke. Same deal and same overrides as
    test_boundary_pricer_events.test_the_correction_covers_heston_nandi_barriers."""
    import test_hn_barrier_cmc as hb
    c = hb._cfg(True)
    c.params['Price Factors']['EquityPrice.EQ']['Spot'] = spot
    c.params['Price Factors']['SurvivalProb.CPTY'] = {
        'Recovery_Rate': 0.4, 'Curve': utils.Curve([], [[0.0, 0.0], [10.0, 0.4]])}
    _, out = derivus.run_cmc(c, prec=hb.DTYPE, overrides={
        'Run_Date': hb.BASE.strftime('%Y-%m-%d'), 'Time_grid': '0d 3m(3m)', 'Batch_Size': batch,
        'Simulation_Batches': 1, 'Random_Seed': seed, 'Currency': 'USD', 'Tenor_Offset': 0.0,
        'MCMC_Simulations': mcmc, 'Deflation_Interest_Rate': 'USD',
        'Gradient_Variables': 'Factors',
        'Credit_Valuation_Adjustment': {
            'Calculate': 'Yes', 'Counterparty': 'CPTY', 'Deflate_Stochastically': 'No',
            'Stochastic_Hazard_Rates': 'No', 'Gradient': 'Yes' if gradient else 'No'}})
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not gradient:
        return out['Results']['mtm'].values, float(out['Results']['cva']), None
    g = out['Results']['grad_cva']['Gradient']
    return (out['Results']['mtm'].values, float(out['Results']['cva']),
            float(g.loc[[i for i in g.index if 'EquityPrice' in str(i[0])][0]]))


@needs_cuda
def test_the_guard_is_what_closed_the_heston_nandi_barrier_gradient(monkeypatch):
    """End to end, on the run the constant was calibrated against, with the oracle measured beside
    it rather than quoted from a note.

    One decision out of twelve is refused - the two-point kernel above, 1.20 widths out and 0.021
    apart, weighted +50.4 / -49.5 - and it was contributing 112% of the coefficient. Measured:
    AAD +0.842068 unguarded against a five-rung CRN oracle of +1.468004 (42.7% low), +1.469847
    guarded (0.13%). Five rungs, not three: the ladder here moved 4.2% when two were added, so a
    three-rung read is not a converged one.

    The forward pass is asserted bit-identical either way, because the whole subsystem's contract
    is that what reaches backward() differs and what is reported does not."""
    _, cva_off, _ = _hn_barrier()
    mtm, cva, aad = _hn_barrier(gradient=True)
    assert cva == cva_off, f'the cva moved when sensitivities were asked for: {cva_off!r} -> {cva!r}'

    r = ladder(price=lambda s: _hn_barrier(spot=s)[1], aad=aad, base=100.0,
               rungs=(2e-4, 5e-4, 1e-3, 2e-3, 5e-3))
    assert r.agrees(tol=0.02), f'the guarded gradient does not land on its oracle\n{r}'

    monkeypatch.setattr(pricing, 'BOUNDARY_MAX_AMPLIFICATION', float('inf'))
    mtm_loose, cva_loose, loose = _hn_barrier(gradient=True)
    assert np.array_equal(mtm, mtm_loose) and cva == cva_loose, (
        'the exposure itself moved with the bound, so the guard is not confined to the backward '
        'pass the way the correction is')
    assert abs(loose - r.best) / abs(r.best) > 0.20, (
        f'the unguarded solve reads {loose:+.6g} against an oracle of {r.best:+.6g} - it is no '
        f'longer broken here, so this fixture no longer gates the guard')


@needs_cuda
def test_at_2048_paths_nothing_is_refused_and_the_guard_is_bit_identical(monkeypatch):
    """The strongest invariant available: quadruple the paths on the same fixture and the two-point
    kernel does not occur, so the guarded and unguarded runs must agree BIT FOR BIT.

    A threshold that refuses legitimate decisions fails here and nowhere else - the CRN oracle at
    these path counts cannot resolve a few percent, and a correction that quietly dropped a healthy
    decision would still look converged. Measured across the sweep this was calibrated on, no
    decision at 1024 paths or above reaches an amplification of 3.6."""
    _, cva, aad = _hn_barrier(batch=2048, mcmc=64)
    monkeypatch.setattr(pricing, 'BOUNDARY_MAX_AMPLIFICATION', float('inf'))
    _, cva_loose, loose = _hn_barrier(batch=2048, mcmc=64)

    assert cva == cva_loose, f'the cva itself moved: {cva!r} -> {cva_loose!r}'
    assert aad == loose, (
        f'the guard changed a run in which it should have refused nothing: {loose!r} -> {aad!r} '
        f'({abs(aad - loose) / abs(loose):.3%}) - the bound is refusing legitimate decisions')
