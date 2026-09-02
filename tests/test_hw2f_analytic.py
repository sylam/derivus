"""Hull-White near zero reversion speed, the Schrager-Pelsser swaption, and the analytic objective.

THE SERIES BRANCH. `hw_calc_H` divides by `a*a`, `hw_calc_IJK` by `a**3`, `hw_calc_B` by `a`, and
the HW2F `AtT` divides one factor's `B` by the OTHER factor's speed. Every one is a REMOVABLE
singularity, so the failure is silent cancellation and not a raise: pre-fix `IJK` read 2.7e+10 out
at |a| = 1e-8, and a benchmark priced 21% low at alpha_2 = 1e-4 with `params_ok` still True
(it guards the Cholesky, not this). The second locus is `alpha_1 + alpha_2 -> 0` - `J[i][j]` is
taken at `alpha_i + alpha_j` - a hyperplane through the box needing no small speed anywhere.
Zero is reachable without solving: `Alpha_1`/`Alpha_2` declare `default=0`.

THE THRESHOLDS ARE READINGS: relative error crosses 1e-10 at 2e-3 (H), 1.5e-2 (IJK) and 1.3e-4 (B),
so `HW_ALPHA_SERIES_H/IJK/B` are 1e-2 / 3e-2 / 1e-3 - each a rung above its crossing and a step
below the 0.1 anything authored carries. Above its threshold each function is BIT-IDENTICAL to the
expression it always was. IJK's crossing rides the sigma slope (3.5e-2 on a 5x-jagged term
structure), which is what is left on the table at 1.7e-10. The `AtT` cross term is the one division
no series repairs: `HW_ALPHA_FLOOR` is 1e-8, two orders above where the quotient stops carrying
information at all, and the 1-factor `AtT` was all NaN at `Alpha`'s own default of 0.

THE ANALYTIC SWAPTION. `schrager_pelsser_swaption` assembles the ATM premium out of the tensors
`precalculate` already carries. Four things hold before any comparison means anything: the loadings
ARE dS/dx at t=0, the variance IS its own integral, the premium IS Bachelier ATM, and the price is
differentiable in every calibrated parameter. `J` is read bit-exact at a grid node and linearly
blended off one - 0.0013bp across a ten-day step, up to 2.07bp across the two-year steps past the
last benchmark, which is why that read warns and names its step.

THE CHECKER: SP against the brute-force Monte Carlo at theta* on the identified 25-quote fixture
(`docs_src/developer/quote_sensitivities.md#the-identified-fixture`). Three separate numbers.
SP's own freezing bias is -0.13 to +2.17bp of normal vol, signed, growing with expiry TIMES tenor
and roughly doubling at the top of `corr_bounds`. The simulation's own numeraire bias is -0.35% to
-1.61% - `E[D(0,T0) A(T0)] = A(0)` failing, and it is the fixture's 1Y first curve node read by a
ten-day deflator, not discretisation. The Monte Carlo noise is 0.51 to 1.81bp per evaluation. SP
therefore sits inside one evaluation's noise at 22 of 25 benchmarks, outside at the three 10Y-tenor
entries by about a fifth of a standard deviation. The checker also found a defect in its subject:
`Var` carried a spurious `e^{-(alpha_k+alpha_l)T_0}` in front of `J`, worth -18.1% to +13.5% of the
normal vol.

THE OBJECTIVE. `Objective` declares `Analytic` (default since 2026-08-31) and `Monte_Carlo`. The
analytic residual differences SP normal vols against a closed-form Bachelier inversion of the same
numpy premium, `P sqrt(2pi/T0)/A` on SP's own annuity - PLAIN, where the Monte Carlo residual is
already a square and so minimises a quartic. At theta* on the 25-quote block ||J'r|| is 3.16e2
under Monte Carlo and 8.63e-7 under Analytic, against `Stationarity_Tol`'s declared 1e-3 default.
Every gate whose subject is the simulation DECLARES `Objective: 'Monte_Carlo'` rather than reaching
the family default; the three whose subject is the default say so in their own docstrings.
`Simulations` and `Batches` are declared too - the batch loop clears `t_Buffer` per iteration, so
(2048 x 4) and (8192 x 1) are one estimate to an ulp - and the closure captures the shape as a
LOCAL, because `bootstrap` runs every curve through one bootstrapper.

THE PREMIUM CONVENTION. `create_market_swaps` reads `Distribution_Type` off `Factor3D.get_subtype`
and the displacement off `InterestYieldVol.displacement`. One numeric ladder read both ways is
9.68x to 11.37x apart, which is 1/F - what pricing a normal vol as a lognormal one cost. A zero or
absent `Market_Volatility` refuses rather than falling through to the surface's ATM read.

THE SEED AND THE CLOCK: one 2026-09-02 re-marking event, and it moved every recorded theta* here.
The premium expiry now reads the curve's day count rather than 365.25ths (`OLD_CLOCK` is the
0.99965771 bias it carried, and it moved every lognormal premium by +3.4e-4); `ALPHA_SEED` is
asymmetric, where two equal speeds beside two identical sigma curves left basin hopping's
iteration-0 descent inside the symmetric hyperplane.

THE QUANTO RULING: the calibration is DOMESTIC. Parameters are measure-invariant and the quanto
drift is the simulation's own bookkeeping, still emitted off the factor. Forcing the FX/IR
correlation to zero used to move the simulated premiums 5.96% to 12.42%; it now moves them by
exactly 0.0.
"""
import itertools
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import pytest
import scipy.optimize
import scipy.stats
import torch

from derivus import bootstrappers, riskfactors, utils
from derivus.bootstrappers import (HullWhite2FactorModelParameters, LeastSquaresSolve,
                                   RiskNeutralInterestRate_State, SwaptionCalibration)
from derivus.config import ModelParams
from derivus.stochasticprocess import (HW_ALPHA_FLOOR, HW_ALPHA_SERIES_B, HW_ALPHA_SERIES_H,
                                       HW_ALPHA_SERIES_IJK, HullWhite1FactorInterestRateModel,
                                       HullWhite2FactorImpliedInterestRateModel, hw_alpha_floor,
                                       hw_calc_B, hw_calc_H, hw_calc_IJK,
                                       integrate_piecewise_linear)

BASE = pd.Timestamp('2026-08-03')
CCY, CURVE = 'ZAR', 'ZAR-SWAP'
DEVICE = torch.device('cpu')
DTYPE = torch.float64

#: the vol knots `HullWhite2FactorModelParameters.implied_process` builds, in years
VOL_TENOR = np.array([0, 1, 3, 6, 12, 24, 48, 72, 96, 120]) / 12.0
#: four sigma term structures inside `sigma_bounds = (1e-5, 0.09)`. `flat` is benign - the
#: cancelling `mi mj` term is identically zero with no slope - and `humped` is the adversarial one.
SIGMA = {'flat': np.full(VOL_TENOR.size, 0.01),
         'sloped': np.linspace(0.004, 0.02, VOL_TENOR.size),
         'steep': np.linspace(1e-5, 0.09, VOL_TENOR.size),
         'humped': np.array([0.004, .02, .03, .025, .012, .008, .006, .02, .004, .03])}
#: a block field a gate wants ABSENT rather than set - how a declared default is read through
#: helpers that write every knob into the JSON
ABSENT = object()
#: a 5x5 swaption grid's expiries, which is the shape the identified fixture carries
TIME_GRID = np.array([0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
#: an interest rate factor's tenor grid, which is what B is read on
FACTOR_TENOR = np.array([1 / 365.0, .25, .5, 1., 2., 3., 5., 7., 10., 15., 20., 30.])
LADDER = [float(10.0 ** e) for e in np.arange(-8.0, 0.5, 0.5)]
#: 40 terms is exact to the last bit for |a * dt| <~ 1, which covers every rung read here
REF_TERMS = 40


class Shared:
    """The one attribute `integrate_piecewise_linear` reaches for."""

    def __init__(self):
        self.t_PreCalc = {}


# ---------------------------------------------------------------- the references

def taylor_H(a, exp, terms=REF_TERMS):
    """int_t^{t+dt} e^{as}(v + m(s-t))ds as its power series in `a`, grouped by dt^k rather than by
    (a dt)^k - an independent ordering of the same sum, not a copy of the branch."""
    def H(t, v, dt, m):
        acc, term = 0.0 * v, 1.0 + 0.0 * a
        for k in range(terms):
            acc = acc + term * (v * dt ** (k + 1) / (k + 1) + m * dt ** (k + 2) / (k + 2))
            term = term * a / (k + 1)
        return acc * exp(a * t)

    return H, 1.0


def taylor_IJK(a, exp, terms=REF_TERMS):
    """The same, for the product integrand."""
    def IJK(t, vi, vj, dt, mi, mj):
        Q, S, P = mi * mj, mj * vi + mi * vj, vi * vj
        acc, term = 0.0 * vi, 1.0 + 0.0 * a
        for k in range(terms):
            acc = acc + term * (P * dt ** (k + 1) / (k + 1) + S * dt ** (k + 2) / (k + 2)
                                + Q * dt ** (k + 3) / (k + 3))
            term = term * a / (k + 1)
        return acc * exp(a * t)

    return IJK, 1.0


def taylor_B(a, tenor, terms=REF_TERMS):
    acc, term = np.zeros_like(tenor), 1.0
    for k in range(terms):
        acc = acc + term * tenor ** (k + 1) / (k + 1)
        term = term * (-a) / (k + 1)
    return acc


def simpson_IJK(a, t, sigma_i, sigma_j, n=200001):
    """int_0^t e^{as} sigma_i sigma_j ds by Simpson on a dense grid - shares nothing with the closed
    form or the series. Good to about 1e-10; the sigma product kinks at every vol knot."""
    if t == 0.0:
        return 0.0
    s = np.linspace(0.0, t, n)
    f = np.exp(a * s) * np.interp(s, VOL_TENOR, sigma_i) * np.interp(s, VOL_TENOR, sigma_j)
    h = s[1] - s[0]
    return h / 3.0 * (f[0] + f[-1] + 4 * f[1:-1:2].sum() + 2 * f[2:-2:2].sum())


def simpson_cross(ai, aj, t, sigma_i, sigma_j, n=200001):
    """The AtT cross term `(e^{-ai t} I_ij - e^{-(ai+aj) t} J_ij) / aj` with nothing cancelling: the
    difference pulled inside the integral is `(e^{aj t} - e^{aj s})/aj -> (t - s)`, regular at aj=0.
    """
    if t == 0.0:
        return 0.0
    s = np.linspace(0.0, t, n)
    x = aj * (t - s)
    bracket = (t - s) * np.where(np.abs(x) > 0, np.expm1(x) / np.where(x == 0, 1.0, x), 1.0)
    f = (np.exp((ai + aj) * s) * bracket
         * np.interp(s, VOL_TENOR, sigma_i) * np.interp(s, VOL_TENOR, sigma_j))
    h = s[1] - s[0]
    return np.exp(-(ai + aj) * t) * h / 3.0 * (
        f[0] + f[-1] + 4 * f[1:-1:2].sum() + 2 * f[2:-2:2].sum())


# ------------------------------------------------- the expressions as they stood, for bit identity

def prefix_H(a, exp):
    def H(t, v, dt, m):
        return (-a * v + m) * exp(a * t) + (a * m * dt + a * v - m) * exp(a * (dt + t))

    return H, a * a


def prefix_IJK(a, exp):
    def IJK(t, vi, vj, dt, mi, mj):
        a2, dt2, Q, S, P = a * a, dt * dt, mi * mj, mj * vi + mi * vj, vi * vj
        return ((a2 * (dt2 * Q + dt * S + P) + 2 * Q * (1 - a * dt) - a * S) * exp(a * dt)
                - a2 * P + a * S - 2 * Q) * exp(a * t)

    return IJK, a ** 3


def prefix_B(a, tenor):
    return (1.0 - torch.exp(-a * tenor)) / a


# ---------------------------------------------------------------- helpers

def integral(fn_norm, sigma_i, sigma_j=None, grid=TIME_GRID):
    shared = Shared()
    t1 = torch.tensor(sigma_i, dtype=DTYPE)
    if sigma_j is None:
        return integrate_piecewise_linear(fn_norm, shared, grid, VOL_TENOR, t1).numpy()
    return integrate_piecewise_linear(
        fn_norm, shared, grid, VOL_TENOR, t1, VOL_TENOR, torch.tensor(sigma_j, dtype=DTYPE)).numpy()


def rel(got, want):
    """max elementwise relative error, skipping the t=0 node that is exactly zero in both"""
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    keep = np.abs(want) > 0.0
    return float((np.abs(got[keep] - want[keep]) / np.abs(want[keep])).max()) if keep.any() else 0.0


#: the jagged one is kept out: `humped`'s adjacent knots differ by 5x, which multiplies IJK's
#: cancelling term by ~370 and moves its crossing by the cube root of that
SMOOTH = ('flat', 'sloped', 'steep')
ALL_CURVES = ('flat', 'sloped', 'steep', 'humped')


def worst(fn, ref, a, curves=SMOOTH, cross=False):
    ta = torch.tensor(a, dtype=DTYPE)
    if cross:
        pairs = [(SIGMA[c], SIGMA['humped']) for c in curves]
        return max(rel(integral(fn(ta, torch.exp), p, q), integral(ref(ta, torch.exp), p, q))
                   for p, q in pairs)
    if ref is taylor_H:
        return max(rel(integral(fn(ta, torch.exp), SIGMA[c]), integral(ref(ta, torch.exp), SIGMA[c]))
                   for c in curves)
    return max(rel(integral(fn(ta, torch.exp), SIGMA[c], SIGMA[c]),
                   integral(ref(ta, torch.exp), SIGMA[c], SIGMA[c])) for c in curves)


# ---------------------------------------------------------------- the authored world

def authored_world():
    """A `Price Factors` block for one currency: a zero curve and an ATM-flat swaption surface.

    Levels are invented; what is read off it is finiteness and continuity as a reversion speed goes
    to zero. No zero tenor on the curve, because `sim_curve` divides by the factor tenor.
    """
    tenors = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0])
    rates = 0.07 + 0.01 * (1.0 - np.exp(-tenors / 5.0))
    surface = [(1.0, e, t, 0.20) for e in (0.25, 1.0, 2.0, 5.0, 10.0) for t in (1.0, 2.0, 5.0, 10.0)]
    return {
        'FxRate.{}'.format(CCY): {'Domestic_Currency': None, 'Interest_Rate': CURVE,
                                  'Priority': 1, 'Spot': 1.0},
        'InterestRate.{}'.format(CURVE): {
            'Property_Aliases': None, 'Sub_Type': None, 'Currency': CCY, 'Day_Count': 'ACT_365',
            'Curve': utils.Curve([], list(zip(tenors, rates)))},
        'InterestYieldVol.{}'.format(CURVE): {
            'Property_Aliases': None, 'Distribution_Type': 'Lognormal', 'Shift': utils.Percent(0.0),
            'Surface': utils.Curve([], surface)}}


def instrument_definitions():
    """Two benchmarks; the first has DIFFERENT fixed and floating frequencies, so the leg-frequency
    branch of `create_market_swaps` is on the path."""
    return [{'Start': pd.DateOffset(years=1), 'Tenor': pd.DateOffset(years=2),
             'Floating_Frequency': pd.DateOffset(months=3),
             'Fixed_Frequency': pd.DateOffset(months=6),
             'Floating_Day_Count': 'ACT_365', 'Fixed_Day_Count': 'ACT_365',
             'Market_Volatility': utils.Percent(20.0), 'Weight': 1.0},
            {'Start': pd.DateOffset(years=2), 'Tenor': pd.DateOffset(years=5),
             'Floating_Frequency': pd.DateOffset(months=6),
             'Fixed_Frequency': pd.DateOffset(months=6),
             'Floating_Day_Count': 'ACT_365', 'Fixed_Day_Count': 'ACT_365',
             'Market_Volatility': utils.Percent(20.0), 'Weight': 1.0}]


@pytest.fixture(scope='module')
def calibration():
    """`(process, implied_var, loss, market_swaps)` - the calibration's own residual closure, built
    the way `calc_loss` builds it. Nothing patched: parameters move through `implied_var`, the seam
    the scipy adapters use.

    `Objective: 'Monte_Carlo'` is DECLARED: the gates it feeds price the two singular loci through
    `generate`, `pv_float_cashflow_list` and the deflation, the half `analytic_loss` does not run.
    """
    price_factors = authored_world()
    rate = utils.check_rate_name('InterestRate.' + CURVE)
    ir_factor = utils.Factor('InterestRate', rate[1:])
    ir_curve = riskfactors.construct_factor(ir_factor, price_factors, ModelParams())
    vol_surface = riskfactors.construct_factor(
        utils.Factor('InterestYieldVol', rate[1:]), price_factors, ModelParams())
    vol_surface.delta = 0.0
    vol_surface.set_premiums(None, ir_curve.get_currency())

    model = HullWhite2FactorModelParameters({}, DEVICE, DTYPE)
    implied_obj, process, vol_tenors = model.implied_process(CCY, price_factors, {}, ir_curve, rate)

    # both knobs are DECLARED, so the block says them in JSON. Defaulting `Objective` would build
    # the analytic closure and price the loci 0.10% and 0.60% away from the recorded rungs.
    instrument = {'Instrument_Definitions': instrument_definitions(), 'Swaption_Volatility': CURVE,
                  'Objective': 'Monte_Carlo', 'Simulations': 1024}
    mtm_dates = set([BASE + x['Start'] for x in instrument['Instrument_Definitions']])
    time_grid = utils.TimeGrid(mtm_dates, mtm_dates, mtm_dates)
    time_grid.set_base_date(BASE, delta=(10, vol_tenors * utils.DAYS_IN_YEAR))

    implied_var, objective, market_swaps, _ = model.calc_loss_on_ir_curve(
        {'instrument': instrument}, BASE, time_grid, process, implied_obj, ir_factor, vol_surface)
    return process, implied_var, objective.loss, market_swaps


def price_at(calibration, alpha_1, alpha_2, sigma=None):
    process, implied_var, loss, _ = calibration
    implied_var['Alpha_1'].data = torch.tensor([alpha_1], dtype=DTYPE)
    implied_var['Alpha_2'].data = torch.tensor([alpha_2], dtype=DTYPE)
    for key in ('Sigma_1', 'Sigma_2'):
        implied_var[key].data = torch.tensor(
            SIGMA['flat'] if sigma is None else sigma, dtype=DTYPE)
    prices, _ = loss(implied_var)
    return process.params_ok, np.array([float(v.detach()) for v in prices.values()])


# ---------------------------------------------------------------- the gates

def test_the_series_and_the_closed_form_are_the_same_function():
    """The overlap check: where the closed form is healthy the two agree to 1e-12 (H) and 1e-10
    (IJK), or the reference is measuring something else."""
    for a in (0.05, 0.1, 0.3, 1.0):
        assert worst(prefix_H, taylor_H, a, curves=ALL_CURVES) < 1e-12, a
        assert worst(prefix_IJK, taylor_IJK, a, curves=ALL_CURVES) < 1e-10, a
        # 40 terms is not enough past |a| ~ 0.1 on a 30-year tenor - the reference's own domain
        if a <= 0.1:
            assert rel(prefix_B(torch.tensor(a, dtype=DTYPE), torch.tensor(FACTOR_TENOR)).numpy(),
                       taylor_B(a, FACTOR_TENOR)) < 1e-12, a


def test_the_series_agrees_with_a_reference_that_shares_nothing_with_it():
    """Simpson on a dense grid against the repaired `hw_calc_IJK` in its near-zero branch, to 1e-9.
    The Taylor reference could be wrong the same way as the branch; this cannot."""
    for a in (0.0, 1e-8, 1e-4):
        got = integral(hw_calc_IJK(torch.tensor(a, dtype=DTYPE), torch.exp),
                       SIGMA['sloped'], SIGMA['humped'])
        want = np.array([simpson_IJK(a, t, SIGMA['sloped'], SIGMA['humped']) for t in TIME_GRID])
        assert rel(got, want) < 1e-9, 'a={}: {:.3e}'.format(a, rel(got, want))


@pytest.mark.parametrize('a', LADDER)
def test_the_crossover_is_where_the_thresholds_say_it_is(a):
    """Post-fix every function is under 1e-10 at every rung; pre-fix IJK is over 1.0 at |a| <= 1e-6.
    The pre-fix column is asserted too, or the thresholds are a taste rather than a reading.

    The one place 1e-10 is not reached: on the JAGGED sigma the branch reads 1.7e-10 at 3.2e-2,
    because that curve crosses at 4e-2. Raising the threshold to meet it would put the branch
    within a factor of two of the 0.1 every authored speed carries.
    """
    post = {'H': worst(hw_calc_H, taylor_H, a, curves=ALL_CURVES),
            'IJK': worst(hw_calc_IJK, taylor_IJK, a),
            'IJK cross': worst(hw_calc_IJK, taylor_IJK, a, cross=True)}
    if a * FACTOR_TENOR.max() < 3.0:
        # past |a T| ~ 3 the 40-term reference is the inaccurate one, not B
        post['B'] = rel(hw_calc_B(torch.tensor(a, dtype=DTYPE), torch.tensor(FACTOR_TENOR)).numpy(),
                        taylor_B(a, FACTOR_TENOR))
    for name, error in post.items():
        assert error < 1e-10, '{} at a={:g} is {:.3e} after the repair'.format(name, a, error)
    jagged = worst(hw_calc_IJK, taylor_IJK, a, curves=('humped',))
    assert jagged < 2e-10, 'IJK on a jagged sigma at a={:g} is {:.3e}'.format(a, jagged)

    pre = {'H': worst(prefix_H, taylor_H, a, curves=ALL_CURVES),
           'IJK': worst(prefix_IJK, taylor_IJK, a, curves=ALL_CURVES)}
    if a < HW_ALPHA_SERIES_H:
        assert pre['H'] >= post['H'], 'H at a={:g}: the branch has to be an improvement'.format(a)
    if a < HW_ALPHA_SERIES_IJK:
        assert pre['IJK'] >= post['IJK'], 'IJK at a={:g}'.format(a)
    if a <= 1e-6:
        # the defect this repairs, still visible: IJK is 2.7e+04 out at 1e-6 and 2.7e+10 at 1e-8
        assert pre['IJK'] > 1.0, 'IJK at a={:g} reads {:.3e} pre-fix'.format(a, pre['IJK'])


def test_the_thresholds_are_the_readings():
    """The four constants pinned directly: 1e-2 / 3e-2 / 1e-3 and a 1e-8 floor.

    Every other gate READS a threshold, so this is the only assertion that fails when
    `HW_ALPHA_SERIES_IJK` moves 3e-2 -> 3.5e-2 - a mutation that otherwise passes the suite while
    every |a| in [3e-2, 3.5e-2) silently swaps the closed form for the series.
    """
    assert (HW_ALPHA_SERIES_H, HW_ALPHA_SERIES_IJK, HW_ALPHA_SERIES_B) == (1e-2, 3e-2, 1e-3), (
        'the thresholds are readings off `test_the_crossover_is_where_the_thresholds_say_it_is` - '
        'H crosses 1e-10 at 2e-3, IJK at 1.5e-2 and B at 1.3e-4 - so moving one is a re-measurement '
        'and this gate is where it gets recorded')
    assert HW_ALPHA_FLOOR == 1e-8, (
        'the floor is `test_the_atT_cross_term_carries_its_number`\'s reading: two orders above the '
        '1e-10 where the AtT quotient stops carrying information at all')


def test_bit_identity_at_the_reversion_speeds_this_repository_carries():
    """Above its threshold each function is BIT-IDENTICAL to the expression it always was.

    Pinned at FIXED speeds the repository carries - `ALPHA_SEED`, `SP_THETA`, `ID_THETA`'s solved
    pair with their sums and doubles, both ends of `alpha_bounds` - never at a threshold, because a
    guard `if abs(a) >= HW_ALPHA_SERIES_*` switches the gate off instead of failing when the
    constant it exists to pin moves. The last lines drive the branch at an explicit rung, so the
    loop is a statement about where the identity is closed rather than about a branch never taken.
    """
    a1, a2 = ID_THETA['Alpha_1'][0], ID_THETA['Alpha_2'][0]
    authored = tuple(bootstrappers.ALPHA_SEED) + tuple(
        SP_THETA['Alpha_1'] + SP_THETA['Alpha_2']) + (2.4, -0.5, -0.1)
    solved = (a1, a2, 2.0 * a1, 2.0 * a2, a1 + a2)
    tenor = torch.tensor(FACTOR_TENOR)
    for a in authored + solved:
        ta = torch.tensor(a, dtype=DTYPE)
        for name, sigma in SIGMA.items():
            assert np.array_equal(integral(hw_calc_H(ta, torch.exp), sigma),
                                  integral(prefix_H(ta, torch.exp), sigma)), ('H', a, name)
            assert np.array_equal(integral(hw_calc_IJK(ta, torch.exp), sigma, sigma),
                                  integral(prefix_IJK(ta, torch.exp), sigma, sigma)), (
                'IJK', a, name)
        assert torch.equal(hw_calc_B(ta, tenor), prefix_B(ta, tenor)), ('B', a)
        assert torch.equal(hw_alpha_floor(ta), ta), ('floor', a)
    # every speed above was ABOVE the threshold, which is what makes the loop a bit-identity claim
    for a in authored + solved:
        assert abs(a) > HW_ALPHA_SERIES_IJK, (
            '{:.6g} is inside the IJK threshold, so the loop above is not the claim it reads '
            'as'.format(a))
    # and below it the branch IS the series
    inside = torch.tensor(HW_ALPHA_SERIES_IJK / 2.0, dtype=DTYPE)
    assert not np.array_equal(
        integral(hw_calc_IJK(inside, torch.exp), SIGMA['humped'], SIGMA['humped']),
        integral(prefix_IJK(inside, torch.exp), SIGMA['humped'], SIGMA['humped']))


def test_the_branch_never_engages_on_an_authored_reversion_speed():
    """`ALPHA_SEED` is the only HW2F alpha this repository authors, and neither coordinate nor
    their sum engages a series branch. The slow half, 0.05, clears `HW_ALPHA_SERIES_IJK` by 1.7x."""
    price_factors = authored_world()
    rate = utils.check_rate_name('InterestRate.' + CURVE)
    ir_curve = riskfactors.construct_factor(
        utils.Factor('InterestRate', rate[1:]), price_factors, ModelParams())
    implied_obj, _, _ = HullWhite2FactorModelParameters(
        {}, DEVICE, DTYPE).implied_process(CCY, price_factors, {}, ir_curve, rate)

    seeded = implied_obj.current_value()
    for name in ('Alpha_1', 'Alpha_2'):
        alpha = float(seeded[name][0])
        assert abs(alpha) > HW_ALPHA_SERIES_IJK, '{} = {} engages the series branch'.format(
            name, alpha)
        assert abs(alpha) > HW_ALPHA_SERIES_H and abs(alpha) > HW_ALPHA_SERIES_B, name
    # and the pair, because J is taken at alpha_1 + alpha_2
    assert abs(float(seeded['Alpha_1'][0]) + float(seeded['Alpha_2'][0])) > HW_ALPHA_SERIES_IJK


@pytest.mark.parametrize('a', [0.0, 1e-12, -1e-12, 1e-9, -1e-9, 1e-4, 0.02, 0.1, -0.1])
def test_the_gradient_survives_the_branch(a):
    """`torch.where` evaluates BOTH branches, so a NaN in the one not taken poisons the backward
    even where the forward is clean - the half of the repair a price gate cannot see."""
    ta = torch.tensor(a, dtype=DTYPE, requires_grad=True)
    sigma = torch.tensor(SIGMA['humped'], dtype=DTYPE, requires_grad=True)
    out = integrate_piecewise_linear(
        hw_calc_IJK(ta, torch.exp), Shared(), TIME_GRID, VOL_TENOR, sigma, VOL_TENOR, sigma).sum()
    out = out + integrate_piecewise_linear(
        hw_calc_H(ta, torch.exp), Shared(), TIME_GRID, VOL_TENOR, sigma).sum()
    out = out + hw_calc_B(ta, torch.tensor(FACTOR_TENOR)).sum() + hw_alpha_floor(ta)
    d_alpha, d_sigma = torch.autograd.grad(out, [ta, sigma])
    assert torch.isfinite(out) and torch.isfinite(d_alpha) and torch.isfinite(d_sigma).all(), a


def test_the_numpy_leg_takes_the_same_branch():
    """`HullWhite1FactorInterestRateModel` hands these the same integrands with `np.exp` and a
    python float, so the branch is spelled once for both or it is not spelled at all."""
    for a in (0.0, 1e-9, 1e-4, 0.1):
        H = integrate_piecewise_linear(
            hw_calc_H(a, np.exp), Shared(), TIME_GRID, VOL_TENOR, SIGMA['humped'])
        assert np.isfinite(H).all(), a
        assert np.isfinite(hw_calc_B(a, FACTOR_TENOR)).all(), a
    # B(0) = T exactly, which is an analytic identity and not a tolerance
    assert np.array_equal(hw_calc_B(0.0, FACTOR_TENOR), FACTOR_TENOR)


def hw1f_at(alpha, sigma='sloped'):
    """`HullWhite1FactorInterestRateModel` precalculated on the authored world at one `Alpha` - the
    numpy leg's own assembly, so a NaN `AtT` fails here and not three layers downstream."""
    price_factors = authored_world()
    rate = utils.check_rate_name('InterestRate.' + CURVE)
    ir_curve = riskfactors.construct_factor(
        utils.Factor('InterestRate', rate[1:]), price_factors, ModelParams())
    process = HullWhite1FactorInterestRateModel(ir_curve, {
        'Alpha': alpha, 'Lambda': 0.0, 'Quanto_FX_Correlation': 0.0, 'Quanto_FX_Volatility': None,
        'Sigma': utils.Curve([], list(zip(VOL_TENOR, SIGMA[sigma])))})
    mtm = set([BASE + pd.DateOffset(years=y) for y in (1, 2, 5, 10)])
    time_grid = utils.TimeGrid(mtm, mtm, mtm)
    time_grid.set_base_date(BASE, delta=(365, VOL_TENOR * utils.DAYS_IN_YEAR))
    process.precalculate(
        BASE, time_grid, torch.tensor(ir_curve.current_value(), device=DEVICE, dtype=DTYPE),
        RiskNeutralInterestRate_State({'full': None, 'reduced': None}, 8, DEVICE, DTYPE), 0)
    return process


def test_the_numpy_assembly_is_finite_at_the_fields_own_default():
    """A finite `B` is not a finite `A`: this drives `precalculate` itself, at `Alpha`'s default 0.

    `AtT = (B/alpha) e^{-at} (2I - (e^{-at}+e^{-aT})J)` divides by the speed a SECOND time. At
    alpha = 0 the series makes `B` the tenor exactly while the bracket is identically zero and
    `B/alpha` is inf, so `0 * inf` put a NaN through the whole curve at a value the price factor
    reaches by omitting the field. It takes `hw_alpha_floor` now, the same as the HW2F cross term.

    What the floor costs: the +/- pair straddle their own alpha -> 0 limit by 7.0e-7 relative,
    against an all-NaN curve.
    """
    field = next(f for f in HullWhite1FactorInterestRateModel.fields if f.name == 'Alpha')
    assert field.default == 0, 'the default is why exactly zero is reachable without solving at all'

    reading = {}
    for a in (0.1, 0.05, -0.05, 1e-4, 1e-8, -1e-8, 1e-9, 1e-12, -1e-12, 0.0):
        process = hw1f_at(a)
        AtT, BtT = process.AtT.numpy(), process.BtT.numpy()
        assert np.isfinite(BtT).all(), 'B at Alpha = {:g}'.format(a)
        assert np.isfinite(AtT).all(), (
            'AtT at Alpha = {:g} is {} finite - the reversion speed is divided by twice and the '
            'floor is what keeps the second one off zero'.format(a, np.isfinite(AtT).sum()))
        reading[a] = AtT
    # something is being measured: a curve of zeros would pass every isfinite above
    assert np.abs(reading[0.1]).max() > 1e-3, np.abs(reading[0.1]).max()

    # the floor is on the REVERSION SPEED and not on the divisor, so the default reads exactly what
    # the floor reads - clamping the divisor instead would zero the term and leave these apart
    assert np.array_equal(reading[0.0], reading[HW_ALPHA_FLOOR]), (
        'Alpha = 0 does not read as Alpha = the floor')
    for a in (1e-12, 1e-9):
        assert np.array_equal(reading[a], reading[HW_ALPHA_FLOOR]), a
    # and what the floor costs, with its number: the +/- pair straddle the limit by 7.0e-7
    straddle = rel(reading[HW_ALPHA_FLOOR], reading[-HW_ALPHA_FLOOR])
    assert 1e-8 < straddle < 5e-6, (
        'the two signs of the floor read {:.3e} apart against a recorded 7.0e-7 - if that is now '
        'zero the floor has stopped being signed, and if it is large it has moved'.format(straddle))


def test_the_atT_cross_term_carries_its_number():
    """The division `hw_alpha_floor` guards rather than repairs, against a cancellation-free
    reference. One order of error per order of alpha_j: under 1e-8 at 1e-2, under 1e-2 at the 1e-8
    floor, and over 1.0 at 1e-12 - which is why the floor is not lower.
    """
    ai, si, sj = 0.1, SIGMA['sloped'], SIGMA['humped']
    reading = {}
    for aj in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12):
        tai, taj = torch.tensor(ai, dtype=DTYPE), torch.tensor(aj, dtype=DTYPE)
        I = integral(hw_calc_IJK(tai, torch.exp), si, sj)
        J = integral(hw_calc_IJK(tai + taj, torch.exp), si, sj)
        got = (np.exp(-ai * TIME_GRID) * I - np.exp(-(ai + aj) * TIME_GRID) * J) / aj
        want = np.array([simpson_cross(ai, aj, t, si, sj) for t in TIME_GRID])
        reading[aj] = rel(got, want)

    # the shape of it: one order of error per order of alpha, all the way down
    assert reading[1e-2] < 1e-8, '1e-2: {:.3e}'.format(reading[1e-2])
    assert reading[1e-4] < 1e-6, '1e-4: {:.3e}'.format(reading[1e-4])
    assert reading[1e-6] < 1e-4, '1e-6: {:.3e}'.format(reading[1e-6])
    assert reading[HW_ALPHA_FLOOR] < 1e-2, 'at the floor: {:.3e}'.format(reading[HW_ALPHA_FLOOR])
    # and why the floor is not lower: below it the quotient is not small, it is wrong
    assert reading[1e-12] > 1.0, (
        'the quotient reads {:.3e} at 1e-12 - if that is now small, the floor is in the wrong '
        'place and this gate has stopped measuring anything'.format(reading[1e-12]))
    # Simpson's own accuracy, so none of the above is a reading of the reference instead
    assert reading[1e-2] > 1e-10, 'the reference floor is in the way: {:.3e}'.format(reading[1e-2])


def basin_walks(seed, walks=2000, niter=50, rng_seed=5120):
    """`(floors, ratios)` over `walks` runs of `basin_step` as `solve` calls it - a per-coordinate
    lognormal multiplier at step 0.125 clipped to `alpha_bounds`, `niter` times. `floors` is the
    smallest speed each walk stood on, `ratios` the pair's separation at the end.
    """
    rng = np.random.RandomState(rng_seed)
    floors, ratios = [], []
    for _ in range(walks):
        alpha, floor = np.array(seed, dtype=float), 1.0
        for _ in range(niter):  # niter=50, as `SwaptionCalibration.solve` calls it
            alpha = (alpha * np.exp(rng.uniform(-0.125, 0.125, alpha.size))).clip(-0.5, 2.4)
            floor = min(floor, alpha.min())
        floors.append(floor)
        ratios.append(alpha[0] / alpha[1])
    return np.array(floors), np.array(ratios)


def test_the_basin_step_can_decay_but_never_cross():
    """`basin_step` run as written: MULTIPLICATIVE, so a reversion speed keeps its sign for the
    whole search - decay toward zero, never a crossing.

    Reachability at the seed the engine starts from, 2000 walks of 50 steps: `ALPHA_SEED` has a
    median floor of 0.0366 and reaches below `HW_ALPHA_SERIES_IJK` in 28.0% of searches, against
    0.0616 and 2.3% for the retired (0.1, 0.1). Worst floor 7.04e-3, above the B threshold; the
    pure-decay bound off the 0.05 coordinate is 9.65e-5.

    It also SEPARATES an equal pair - one multiplier per coordinate, ratio 1.0064 after one step -
    so the retired symmetric seed cost basin hopping's iteration-0 descent and not the search
    (`test_the_declared_alpha_seed_is_asymmetric_and_the_first_descent_leaves_the_ridge`).
    """
    live, live_ratios = basin_walks(bootstrappers.ALPHA_SEED)
    old, ratios = basin_walks((0.1, 0.1))

    for tag, floors in (('ALPHA_SEED', live), ('(0.1, 0.1)', old)):
        assert (floors > 0.0).all(), (
            '{}: a multiplicative step cannot change sign'.format(tag))
        assert floors.min() < 0.05, (
            '{}: but it decays - worst floor reached {:.3g}'.format(tag, floors.min()))
    # the LIVE seed's own reachability, which is the reading the module docstring quotes
    assert 0.2 < float((live < HW_ALPHA_SERIES_IJK).mean()) < 0.4, (
        'the shipped seed drops its slow factor into the IJK series branch in {:.1%} of searches '
        'against a recorded 28.0% - this is the reading that says which branch the random search '
        'spends its time in'.format(float((live < HW_ALPHA_SERIES_IJK).mean())))
    assert float((live < HW_ALPHA_SERIES_IJK).mean()) > 5.0 * float(
        (old < HW_ALPHA_SERIES_IJK).mean()), (
        'the shipped seed no longer reaches the series branch an order of magnitude more often '
        'than the retired one did: {:.1%} against {:.1%}'.format(
            float((live < HW_ALPHA_SERIES_IJK).mean()),
            float((old < HW_ALPHA_SERIES_IJK).mean())))
    assert live.min() > HW_ALPHA_SERIES_B, (
        'a walk off the shipped seed reached {:.3g}, below the B threshold, where the recorded '
        'worst of 2000 is 7.04e-3'.format(live.min()))
    # the separation: an equal pair NEVER comes back equal, over 2000 independent walks
    assert (ratios != 1.0).all(), (
        'the step returned an equal pair - it draws one multiplier per coordinate and cannot')
    # and widely: the median log-ratio after 50 steps is 0.477, a factor 1.6
    assert np.median(np.abs(np.log(ratios))) > 0.3, (
        'the median separation after a full search is a log-ratio of {:.3g} against a recorded '
        '0.477'.format(float(np.median(np.abs(np.log(ratios))))))
    # and it never UNDOES the declared seed's own separation, which is what carries it to the fit
    assert np.abs(np.log(live_ratios)).min() > 0.15, (
        'a walk off the shipped seed came back {:.3g} apart in log-ratio, against a recorded '
        'closest of 0.177 in 2000'.format(float(np.abs(np.log(live_ratios)).min())))
    # and one step off the declared seed is already the recorded 1.0064
    one = np.array([0.1, 0.1]) * np.exp(np.random.RandomState(5120).uniform(-0.125, 0.125, 2))
    assert abs(one[0] / one[1] - 1.0064054) < 1e-6, one
    # the bound if every draw went the same way, off the seed's own SLOW coordinate
    assert min(bootstrappers.ALPHA_SEED) * np.exp(-0.125 * 50) < HW_ALPHA_SERIES_B


def test_least_squares_can_cross_zero_outright():
    """`alpha_bounds` straddles zero, so basin hopping's inner L-BFGS-B and the `least_squares`
    after it can cross where the multiplicative step cannot - and `Alpha_1` defaults to 0."""
    model = HullWhite2FactorModelParameters({}, DEVICE, DTYPE)
    low, high = model.alpha_bounds
    assert low < 0.0 < high, model.alpha_bounds
    # and the price factor reaches zero without any solve at all
    alpha_field = next(f for f in riskfactors.HullWhite2FactorModelParameters.fields
                       if f.name == 'Alpha_1')
    assert alpha_field.default == 0


def test_the_declared_alpha_seed_is_asymmetric_and_the_first_descent_leaves_the_ridge():
    """THE SEED DEFECT. `Alpha_1 = Alpha_2` beside two identical sigma curves is two copies of one
    factor: the objective is exchange-symmetric there, and `solve` runs basin hopping first, which
    minimises AT `x0` before taking any step - so that entire first descent stays on the symmetric
    hyperplane. The step was never the problem; it separates from iteration 1 onward.

    One L-BFGS-B call each through `calc_loss`'s own adapter on the 25-quote block: the symmetric
    seed reaches (0.0118, 0.0118) at loss 8.95e-6, `ALPHA_SEED` reaches (0.500, 0.00267) at
    5.33e-6 - a factor 1.68 lower. The assertions hold the SHAPE (ridge, separation, ordering),
    because an L-BFGS-B stopping point can move an ulp under a scipy upgrade.
    """
    lo, hi = HullWhite2FactorModelParameters({}, DEVICE, DTYPE).alpha_bounds
    a1, a2 = bootstrappers.ALPHA_SEED
    assert a1 != a2, 'the declared seed is symmetric again - that is the defect itself'
    # all four box inequalities: `bounds_check` tests them strictly
    assert lo < a1 < hi and lo < a2 < hi, (
        'the seed {} is outside the box ({}, {}) that `bounds_check` tests strictly'.format(
            bootstrappers.ALPHA_SEED, lo, hi))
    assert min(abs(a1), abs(a2)) > HW_ALPHA_SERIES_IJK, (
        'the seed opens a small-alpha series branch at evaluation zero: {}'.format(
            bootstrappers.ALPHA_SEED))
    assert abs(a1 + a2) > HW_ALPHA_SERIES_IJK, 'the seed sits on the alpha-sum singular locus'
    assert a1 > 0.0 and a2 > 0.0, (
        'the basin step is multiplicative and preserves sign, so a seed that is to carry its '
        'separation through the random search has to be one-signed')

    calibration, _ = identified_calibration(benchmarks=ID_GRID, Objective='Analytic')
    basin = calibration.optimizers[0]
    x0, fn_grad, bounds = basin[1], basin[2], basin[5]
    assert [float(v) for v in x0[:2]] == list(bootstrappers.ALPHA_SEED), (
        'the chain did not start where the seam says it does: {}'.format(x0[:2]))

    declared = scipy.optimize.minimize(fn_grad, x0, method='L-BFGS-B', jac=True, bounds=bounds)
    symmetric_x0 = x0.copy()
    symmetric_x0[0] = symmetric_x0[1] = 0.1
    symmetric = scipy.optimize.minimize(
        fn_grad, symmetric_x0, method='L-BFGS-B', jac=True, bounds=bounds)

    # the OLD seed's first descent never leaves the ridge, to eight digits
    assert abs(symmetric.x[0] / symmetric.x[1] - 1.0) < 1e-8, (
        'the symmetric seed no longer descends inside the symmetric hyperplane - it reached '
        '{} and this gate has stopped measuring what it was built for'.format(symmetric.x[:2]))
    # the DECLARED seed's does, by an order of magnitude
    assert declared.x[0] / declared.x[1] > 10.0, (
        'the declared seed descended to two reversion speeds a factor {:.3g} apart - it is no '
        'longer buying a fast factor and a slow one'.format(declared.x[0] / declared.x[1]))
    # and it is the better point, which is what the ridge costs
    assert declared.fun < symmetric.fun, (
        'the declared seed reaches {:.6g} against the symmetric seed\'s {:.6g} - the recorded '
        'readings are 5.33e-6 against 8.95e-6, a factor 1.68'.format(declared.fun, symmetric.fun))


def test_params_ok_does_not_guard_this(calibration):
    """`params_ok` is the Cholesky's positive definiteness, read by the acceptance test as if it
    meant well-posed parameters. It is True at the speed that used to move the benchmark 21%."""
    ok, prices = price_at(calibration, 0.1, 1e-4, SIGMA['sloped'])
    assert ok, 'params_ok is True here - that is the point'
    assert np.isfinite(prices).all()


@pytest.mark.parametrize('alpha_2', [1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-9, 0.0, -1e-9, -1e-4])
def test_the_objective_prices_through_zero(calibration, alpha_2):
    """Down the alpha_2 -> 0 locus every rung prices finitely, positively, and within 5% of the
    1e-2 rung - the alpha -> 0 limit is a real one."""
    ok, prices = price_at(calibration, 0.1, alpha_2, SIGMA['sloped'])
    _, limit = price_at(calibration, 0.1, 1e-2, SIGMA['sloped'])
    assert ok, 'params_ok went False at alpha_2 = {:g}'.format(alpha_2)
    assert np.isfinite(prices).all(), 'alpha_2 = {:g} priced {}'.format(alpha_2, prices)
    assert (prices > 0.0).all(), 'alpha_2 = {:g} priced {}'.format(alpha_2, prices)
    assert (np.abs(prices / limit - 1.0) < 0.05).all(), (
        'alpha_2 = {:g} priced {} against {} at 1e-2'.format(alpha_2, prices, limit))


@pytest.mark.parametrize('delta', [1e-2, 1e-4, 1e-6, 1e-8, 0.0])
def test_the_objective_prices_through_the_alpha_sum_locus(calibration, delta):
    """The other locus, with no small speed in it: `J` is read at `alpha_i + alpha_j`, so
    (0.1, -0.1) is singular with both factors ordinary. Same 5% band on the 1e-2 rung."""
    ok, prices = price_at(calibration, 0.1, delta - 0.1, SIGMA['sloped'])
    _, limit = price_at(calibration, 0.1, -0.1 + 1e-2, SIGMA['sloped'])
    assert ok, 'params_ok went False at alpha_1 + alpha_2 = {:g}'.format(delta)
    assert np.isfinite(prices).all(), '{} priced {}'.format(delta, prices)
    assert (prices > 0.0).all(), '{} priced {}'.format(delta, prices)
    assert (np.abs(prices / limit - 1.0) < 0.05).all(), (
        'alpha_1 + alpha_2 = {:g} priced {} against {} at 1e-2'.format(delta, prices, limit))


# ------------------------------------------------- the Schrager-Pelsser swaption, and its own world

def as_float(x):
    """`float()` on a tensor still carrying a graph warns, and every reading below is off one."""
    return float(x.detach())


#: One ATM payer swaption's fixed leg, 2Y into 5Y semi-annual, in the CURVE's own ACT_365 year
#: fractions - the clock `read_cache` builds `time_grid_years` on and `J` is integrated against.
#: The expiry is a grid NODE because the block's 2Y benchmark put it there.
SP_EXPIRY = 2.0
SP_PAY = np.arange(2.5, 7.01, 0.5)
SP_TAU = np.full(SP_PAY.size, 0.5)
#: and one that is NOT a node: the grid is 10-daily, so 1.5 years falls between 540 and 550 days
SP_OFF_NODE = 1.5

#: Asymmetric in EVERY coordinate, so a formula that swapped the factors, dropped a cross term or
#: lost a sign cannot pass by coincidence.
SP_THETA = {'Alpha_1': [0.06], 'Alpha_2': [0.35], 'Correlation': [-0.6],
            'Sigma_1': SIGMA['sloped'], 'Sigma_2': SIGMA['humped']}


def sp_at(calibration, expiry=SP_EXPIRY, **override):
    """`(process, HW2FSwaption)` at one parameter set, through the closure the optimizers call.

    Leaves move with `.data`, the seam the scipy adapters use, and `loss` is what runs
    `precalculate`. The Monte Carlo prices are discarded: these gates read what `precalculate` left.
    """
    process, implied_var, loss, _ = calibration
    for key, value in dict(SP_THETA, **override).items():
        implied_var[key].data = torch.tensor(np.atleast_1d(value), dtype=DTYPE)
    loss(implied_var)
    return process, process.schrager_pelsser_swaption(expiry, SP_PAY, SP_TAU)


def swap_rate_at_state(process, x, expiry=SP_EXPIRY):
    """The forward swap rate as an explicit function of the t=0 state, in numpy. At t=0 the
    convexity term is identically zero, so the discount factor is $P(0,T)e^{-\\sum_k B_k(T)x_k}$.
    Shares nothing with the loading formula, so `dS/dx` off it is evidence about the loadings.
    """
    times = np.concatenate(([expiry], SP_PAY))
    alpha = np.array([as_float(a) for a in process.alpha]).reshape(-1, 1)
    B = (1.0 - np.exp(-alpha * times)) / alpha
    P = np.exp(-process.factor.current_value(times) * times) * np.exp(-np.asarray(x).dot(B))
    annuity = (SP_TAU * P[1:]).sum()
    return (P[0] - P[-1]) / annuity, annuity


def simpson_sp_variance(process, q, expiry=SP_EXPIRY, n=200001, sigma_name=('Sigma_1', 'Sigma_2'),
                        theta=None):
    """$\\mathrm{Var}(\\sum_k q_kY_k(T_0))$ as ONE Simpson of ONE integrand.

    `q` is the loading on the SCALED martingale $Y_k(T_0)=\\int_0^{T_0}e^{\\alpha_ks}\\sigma_k(s)
    dW_k(s)$ (`schrager_pelsser_swaption` has the $B_k(t)$ cancellation), so the covariance is
    written out rather than read off $J_{kl}$. Nothing here touches `J` or the closed forms.
    """
    theta = SP_THETA if theta is None else theta
    s = np.linspace(0.0, expiry, n)
    sigma = [np.interp(s, VOL_TENOR, theta[sigma_name[0]]),
             np.interp(s, VOL_TENOR, theta[sigma_name[1]])]
    alpha = [as_float(a) for a in process.alpha]
    rho = [[1.0, theta['Correlation'][0]], [theta['Correlation'][0], 1.0]]
    f = np.zeros_like(s)
    for k, l in itertools.product(range(2), range(2)):
        f = f + rho[k][l] * q[k] * q[l] * np.exp(
            (alpha[k] + alpha[l]) * s) * sigma[k] * sigma[l]
    h = s[1] - s[0]
    return h / 3.0 * (f[0] + f[-1] + 4 * f[1:-1:2].sum() + 2 * f[2:-2:2].sum())


def test_the_loadings_are_the_swap_rates_own_derivative(calibration):
    """`q_k` is what the approximation FREEZES, so it has to be `dS/dx_k` at t=0: central-differenced
    against a rebuilt swap rate to 1e-8, with the annuity and forward rate held to 1e-14."""
    process, sp = sp_at(calibration)
    rate, annuity = swap_rate_at_state(process, np.zeros(2))
    assert abs(as_float(sp.annuity) / annuity - 1.0) < 1e-14, (as_float(sp.annuity), annuity)
    assert abs(as_float(sp.swap_rate) / rate - 1.0) < 1e-14, (as_float(sp.swap_rate), rate)

    h = 1e-6
    for k in range(2):
        bump = np.zeros(2)
        bump[k] = h
        fd = (swap_rate_at_state(process, bump)[0]
              - swap_rate_at_state(process, -bump)[0]) / (2.0 * h)
        got = as_float(sp.loadings[k])
        assert abs(got) > 1e-3, 'loading {} is {:.3e} - nothing is being measured'.format(k, got)
        assert abs(got / fd - 1.0) < 1e-8, 'q_{} is {:.12g} against dS/dx {:.12g}'.format(k, got, fd)
    # the two are not the same number, which is what makes the pair a test of both
    assert abs(as_float(sp.loadings[0]) / as_float(sp.loadings[1]) - 1.0) > 0.1


def test_the_variance_is_a_direct_integral_of_the_same_integrand(calibration):
    """`Var = sum_kl rho_kl q_k q_l J_kl(T0)` against Simpson on the covariance integrand, to 1e-9.

    The two routes share the loadings and nothing else, so this is what says the correlation matrix
    is in the right place and that NO `e^{-(a_k+a_l)T0}` prefactor belongs in front of `J` - `q` is
    the loading on the scaled martingale whose covariance `J` already is."""
    process, sp = sp_at(calibration)
    want = simpson_sp_variance(process, [as_float(x) for x in sp.loadings])
    got = as_float(sp.variance)
    assert abs(got / want - 1.0) < 1e-9, '{:.12g} against Simpson {:.12g}'.format(got, want)
    # and the correlation is load-bearing: the same q at rho = 0 is a materially different variance
    _, uncorrelated = sp_at(calibration, Correlation=[0.0])
    assert abs(as_float(uncorrelated.variance) / got - 1.0) > 0.05


def test_the_premium_is_the_engines_own_bachelier_at_the_money(calibration):
    """`A(0) sqrt(Var/2pi)` against `utils.bachelier_european_option` at F == X, to 1e-14, and
    `Var == vol^2 T_0` with it. The closed form is spelled out rather than delegated because that
    function clamps its vol at 1e-5 and a reference should not carry a floor its subject lacks."""
    _, sp = sp_at(calibration)
    engine = sp.annuity * utils.bachelier_european_option(
        sp.swap_rate, sp.swap_rate, sp.normal_vol, SP_EXPIRY, 1.0, 1.0, None)
    assert abs(as_float(engine) / as_float(sp.premium) - 1.0) < 1e-14, (
        '{:.15g} against the engine\'s {:.15g}'.format(as_float(sp.premium), as_float(engine)))
    assert abs(as_float(sp.normal_vol) ** 2 * SP_EXPIRY / as_float(sp.variance) - 1.0) < 1e-14
    # the clamp is why the closed form is spelled out: at a vol under 1e-5 the engine's Bachelier
    # stops being the identity and this price does not
    assert as_float(sp.normal_vol) > 1e-5, as_float(sp.normal_vol)


@pytest.mark.parametrize('name', ['Alpha_1', 'Alpha_2', 'Sigma_1', 'Sigma_2', 'Correlation'])
def test_the_price_is_differentiable_in_the_calibrated_parameters(calibration, name):
    """The premium carries a finite gradient back to every calibrated parameter off `implied_var`,
    the leaf the scipy adapters write, matching a central difference to 1e-6. A term structure is
    bumped whole against `grad.sum()` - the directional derivative along ones, one difference
    rather than ten, and still failing if any knot's gradient is wrong."""
    _, implied_var, _, _ = calibration
    _, sp = sp_at(calibration)
    grad = torch.autograd.grad(sp.premium, implied_var[name])[0]
    assert torch.isfinite(grad).all(), '{} gradient is {}'.format(name, grad)
    assert abs(as_float(grad.sum())) > 1e-6, (
        '{} gradient sums to {:.3e} - a central difference would match nothing'.format(
            name, as_float(grad.sum())))

    h = 1e-6
    base = np.atleast_1d(np.asarray(SP_THETA[name], dtype=np.float64))
    up = as_float(sp_at(calibration, **{name: base + h})[1].premium)
    down = as_float(sp_at(calibration, **{name: base - h})[1].premium)
    fd = (up - down) / (2.0 * h)
    assert abs(as_float(grad.sum()) / fd - 1.0) < 1e-6, (
        '{}: autograd {:.12g} against a central difference {:.12g}'.format(
            name, as_float(grad.sum()), fd))


def test_J_is_read_at_the_expiry_and_at_a_node_exactly(calibration):
    """`TimeGrid` is built out of the benchmark starts, so the ordinary read is the node itself and
    not an interpolation landing on it. Rebuilt off `process.J` at the node index, BIT-identical."""
    process, sp = sp_at(calibration)
    grid = process.cache['time_grid_years']
    node = int(np.searchsorted(grid, SP_EXPIRY))
    assert grid[node] == SP_EXPIRY, 'the 2Y benchmark should have put {} on the grid'.format(
        SP_EXPIRY)

    rho, q = process.rho, sp.loadings
    variance = 0.0
    for k, l in itertools.product(range(2), range(2)):
        variance = variance + rho[k][l] * q[k] * q[l] * process.J[k][l][node]
    assert as_float(variance.reshape(())) == as_float(sp.variance), (
        as_float(variance), as_float(sp.variance))


def test_J_off_a_node_is_the_linear_blend_and_carries_its_number(calibration):
    """Off a node the read is `utils.interpolate_tensor`'s linear blend between the two neighbours.
    Held to be exactly that blend, and priced against the exact integral.

    On a ten-day step (1.5Y) the chord over-reads the variance by 3.0e-5 relative, which is
    0.0013bp of normal vol at an 85.1bp level - four orders under the 0.7bp of Monte Carlo standard
    error the objective's 8192 paths carry. Past the last benchmark expiry the grid jumps to the
    vol knots and the step is TWO YEARS: 2.5Y..5.0Y read +0.79 to +2.07bp, the size of the
    approximation error the checker measures. That is why the read warns and names its step.

    The sign is the local curvature's, not a law - at 0.75Y the same read is -4.7e-5 - so the
    assertion holds the sign at 1.5 where it was measured.
    """
    with pytest.warns(UserWarning, match='not a node'):
        process, sp = sp_at(calibration, expiry=SP_OFF_NODE)
    grid = process.cache['time_grid_years']
    hi = int(np.searchsorted(grid, SP_OFF_NODE))
    assert grid[hi - 1] < SP_OFF_NODE < grid[hi], 'this expiry has to be off the grid'

    weight = (SP_OFF_NODE - grid[hi - 1]) / (grid[hi] - grid[hi - 1])
    rho, q = process.rho, sp.loadings
    variance = 0.0
    for k, l in itertools.product(range(2), range(2)):
        blend = process.J[k][l][hi - 1] * (1.0 - weight) + process.J[k][l][hi] * weight
        variance = variance + rho[k][l] * q[k] * q[l] * blend
    assert as_float(variance.reshape(())) == as_float(sp.variance), 'not the linear blend'

    want = simpson_sp_variance(process, [as_float(x) for x in q], expiry=SP_OFF_NODE)
    error = as_float(sp.variance) / want - 1.0
    assert 0.0 < error < 1e-4, (
        'the chord reads {:.3e} relative against the integral of the same integrand - it is a chord '
        'over a convex J, so it over-reads, and a sign change here means the read has stopped being '
        'the blend this gate just asserted'.format(error))
    # what that is worth where a desk reads it, which is the reason it is tolerable
    assert abs(as_float(sp.normal_vol) - np.sqrt(want / SP_OFF_NODE)) * 1e4 < 0.01

    # the two-year gaps past the last benchmark, held to 2% - nothing here is simulated. The
    # warning has to fire at every one of them or the chord is silent again.
    for expiry, var_rel, bp in ((2.5, 6.021e-2, 2.0711), (3.0, 6.286e-2, 1.9250),
                                (3.5, 3.832e-2, 1.0312), (4.5, 4.820e-2, 0.9064),
                                (5.0, 5.385e-2, 0.7907)):
        with pytest.warns(UserWarning, match='not a node'):
            gap_process, gap_sp = sp_at(calibration, expiry=expiry)
        exact = simpson_sp_variance(
            gap_process, [as_float(x) for x in gap_sp.loadings], expiry=expiry)
        got = as_float(gap_sp.variance) / exact - 1.0
        got_bp = (as_float(gap_sp.normal_vol) - np.sqrt(exact / expiry)) * 1e4
        assert abs(got / var_rel - 1.0) < 0.02, (
            '{}Y reads {:.3e} on the variance against a recorded {:.3e}'.format(
                expiry, got, var_rel))
        assert abs(got_bp / bp - 1.0) < 0.02, (
            '{}Y reads {:+.4f}bp against a recorded {:+.4f} - the chord across a two-year step is '
            'the same size as the approximation error the checker measures, and that is the whole '
            'reason this warns'.format(expiry, got_bp, bp))
    # the two stretches are three orders apart, which is the finding rather than the tolerance
    assert error * 1000.0 < 3.8e-2


def test_the_analytic_swaption_refuses_rather_than_extrapolates(calibration):
    """Two refusals plus the un-precalculated one. Reading `J` past the grid would flat-extrapolate
    silently, because `utils.interpolate_tensor` clips."""
    process, _ = sp_at(calibration)
    horizon = process.cache['time_grid_years'][-1]
    with pytest.raises(Exception, match='outside'):
        process.schrager_pelsser_swaption(horizon + 1.0, SP_PAY, SP_TAU)
    with pytest.raises(Exception, match='outside'):
        process.schrager_pelsser_swaption(0.0, SP_PAY, SP_TAU)

    from derivus.stochasticprocess import HullWhite2FactorImpliedInterestRateModel as HW2F
    fresh = HW2F.__new__(HW2F)
    fresh.J = None
    with pytest.raises(Exception, match='precalculate'):
        HW2F.schrager_pelsser_swaption(fresh, SP_EXPIRY, SP_PAY, SP_TAU)


# ---------------------------------------- THE CHECKER: Schrager-Pelsser against the Monte Carlo

#: The repo's identified HW2F fixture: a humped ZAR-shaped zero curve, a cube rising with expiry
#: and falling with tenor, and a 5x5 grid of 25 benchmarks against 23 parameters quoted FLAT at 20
#: vol. `docs_src/developer/quote_sensitivities.md#the-identified-fixture` is the design record.
ID_CCY, ID_CURVE, ID_VOL = 'ZAR', 'ZAR-JIBAR-3M', 'ZAR_SWAPTION'
ID_BLOCK = 'HullWhite2FactorModelPrices.' + ID_CURVE
ID_ZERO = ((1.0, 0.0800), (2.0, 0.0835), (3.0, 0.0880), (5.0, 0.0915), (7.0, 0.0950), (10.0, 0.0905))
ID_SURFACE_E, ID_SURFACE_T, ID_MONEY = (0.25, 1.0, 2.0, 5.0), (1.0, 2.0, 5.0, 10.0), (-0.01, 0.0, 0.01)
ID_STATIONARITY, ID_SEED = 1e5, 5120

#: theta* on the 25-quote grid under `Objective: 'Monte_Carlo'`, AS SOLVED - so a gate reads the
#: calibrated point without paying the 2304 s `SwaptionCalibration.solve` costs at this block's
#: `Random_Seed` in float32 on CUDA. It cost 531 s before the 2026-09-02 seed-and-clock re-mark,
#: which is the squared residual's quartic taking more optimizer iterations from a separated seed.
ID_THETA = {
    'Alpha_1': [0.2236477176394386],
    'Alpha_2': [0.059855648875476106],
    'Correlation': [-0.9411964519647071],
    'Sigma_1': [0.048875401579212434, 0.037597874799372204, 0.01605911877783019,
                0.019955134367379605, 0.017089982593522925, 0.0063462252324984845,
                0.08043282984444651, 0.0816858786914603, 0.06612160740018493,
                0.05477805411987446],
    'Sigma_2': [0.07546553168585883, 0.054077255262949354, 0.022210485581280602,
                0.032087034716934366, 0.0296409668489019, 0.030198948838079873,
                0.05997267405985911, 0.0192699408355007, 0.06273524268924238,
                0.05933545450099554]}
#: the checker's rows: the grid's own corners plus two whose FIXED and FLOATING frequencies differ,
#: which is the pair that settles the single-curve question
CHECKER_BENCHMARKS = ((1, 1, 3, 3), (2, 5, 3, 6), (3, 3, 3, 12), (10, 10, 3, 3))


def identified_world(zero=ID_ZERO):
    """The `Price Factors` block of the identified fixture - one curve and one swaption cube."""
    quads = [[m, e, t, 0.20 + 0.01 * np.log1p(e) - 0.005 * np.log1p(t) + 2.0 * m * m]
             for t in ID_SURFACE_T for e in ID_SURFACE_E for m in ID_MONEY]
    return {
        'FxRate.{}'.format(ID_CCY): {
            'Domestic_Currency': None, 'Interest_Rate': ID_CURVE, 'Priority': 1, 'Spot': 1.0},
        'InterestRate.{}'.format(ID_CURVE): {
            'Currency': ID_CCY, 'Day_Count': 'ACT_365', 'Sub_Type': None,
            'Curve': utils.Curve([], list(zero))},
        'InterestYieldVol.{}'.format(ID_VOL): {
            'Property_Aliases': None, 'Currency': ID_CCY, 'Distribution_Type': 'Lognormal',
            'Shift': utils.Percent(0), 'Surface': utils.Curve([], quads)}}


def identified_definitions(benchmarks):
    """`Instrument_Definitions` from `(expiry, tenor, float months, fixed months)` rows."""
    return [{'Start': pd.DateOffset(years=int(e)), 'Tenor': pd.DateOffset(years=int(t)),
             'Floating_Frequency': pd.DateOffset(months=fm),
             'Fixed_Frequency': pd.DateOffset(months=xm),
             'Floating_Day_Count': 'ACT_365', 'Fixed_Day_Count': 'ACT_365',
             'Market_Volatility': utils.Percent(20.0), 'Weight': 1.0}
            for e, t, fm, xm in benchmarks]


def identified_calibration(benchmarks=CHECKER_BENCHMARKS, **extra):
    """A full `SwaptionCalibration` on the identified fixture, the OPTIMIZER CHAIN included -
    through `calc_loss`, so a gate that solves drives what `bootstrap` drives. Parameters are left
    AT THE SEED, because `calc_loss` captures `x0` off them.
    """
    from derivus.bootstrappers import SwaptionCalibration
    world = identified_closure(benchmarks=benchmarks, theta={}, chain=True, **extra)
    return SwaptionCalibration('gate', world['objective'], world['implied_var'],
                               world['optimizers'], world['process'], world['swaps']), world


def identified_closure(benchmarks=CHECKER_BENCHMARKS, zero=ID_ZERO, batch_size=8192,
                       theta=None, device=DEVICE, dtype=DTYPE, world=None, chain=False,
                       premiums=None, delta=0.0, **extra):
    """The calibration's own residual closure on the identified fixture, standing at `theta`.

    Built the way `RiskNeutralInterestRateModel.bootstrap` builds it, nothing patched: parameters
    move through `implied_var` and every knob is DECLARED in the JSON rather than written onto the
    bootstrapper as an attribute.

    `Objective` is the one this block LEAVES OUT: a caller names the path it means, and a call
    naming neither is a gate reading the family default. Three do that and say so in their own
    docstrings - `test_the_declared_sample_shape_is_the_shape_the_engine_uses` (which POPS the key
    off the block, so a scan of call sites will not see it),
    `test_the_absent_objective_is_the_declared_analytic_one` and
    `test_the_two_spellings_of_the_default_drive_the_adapters_identically`.

    A field passed as `ABSENT` is DELETED rather than set, which is how a gate reads a declared
    default: `Stationarity_Tol` is written here at this fixture's 1e5 for the Monte Carlo path,
    and the analytic quote side is contracted to run at the field's own 1e-3. `premiums` and
    `delta` are the pair whose brentq re-strike the quote side declines.
    """
    from derivus import bootstrappers
    factors, interp = (world or identified_world(zero)), ModelParams()
    boot = HullWhite2FactorModelParameters({}, device, dtype)
    rate = utils.check_rate_name(ID_BLOCK)
    ir_factor = utils.Factor('InterestRate', rate[1:])
    surface = riskfactors.construct_factor(
        utils.Factor('InterestYieldVol', (ID_VOL,)), factors, interp)
    surface.delta = delta
    ir_curve = riskfactors.construct_factor(ir_factor, factors, interp)
    surface.set_premiums(premiums, ir_curve.get_currency())
    implied_obj, process, vol_tenors = boot.implied_process(
        extra.pop('base_currency', ID_CCY), factors, {}, ir_curve, rate)
    block = {'Swaption_Volatility': ID_VOL, 'Generate_Instruments': 'No', 'Random_Seed': ID_SEED,
             'Stationarity_Tol': ID_STATIONARITY, 'Quote_Sensitivity': 'No',
             'Simulations': batch_size, 'Batches': 1,
             'Instrument_Definitions': identified_definitions(benchmarks)}
    block.update(extra)
    block = {k: v for k, v in block.items() if v is not ABSENT}
    mtm = set([BASE + x['Start'] for x in block['Instrument_Definitions']])
    time_grid = utils.TimeGrid(mtm, mtm, mtm)
    time_grid.set_base_date(BASE, delta=(10, vol_tenors * utils.DAYS_IN_YEAR))
    optimizers = None
    if chain:
        objective, optimizers, implied_var, swaps, _ = boot.calc_loss(
            {'instrument': block}, BASE, time_grid, process, implied_obj, ir_factor, surface)
    else:
        implied_var, objective, swaps, _ = boot.calc_loss_on_ir_curve(
            {'instrument': block}, BASE, time_grid, process, implied_obj, ir_factor, surface)
    for name, value in (ID_THETA if theta is None else theta).items():
        implied_var[name].data = torch.tensor(value, dtype=dtype, device=device)
    return dict(model=boot, process=process, implied_var=implied_var, loss=objective.loss,
                objective=objective, swaps=swaps, block=block, time_grid=time_grid,
                curve=ir_curve, ir_factor=ir_factor, implied_obj=implied_obj, factors=factors,
                optimizers=optimizers, surface=surface)


def checker_legs(world):
    """`{benchmark: the FIXED leg}` in the CURVE's own year fractions, CHECKED against the schedule
    the Monte Carlo prices.

    `create_market_swaps` writes the fixed leg into the FLOAT leg's `FixedAmt` column and keeps no
    fixed schedule, so this rebuilds it with the same two generators and holds the rebuild against
    that column - an analytic leg that is not the simulated leg fails here rather than downstream.

    The clock is the CURVE's ACT_365 and not `utils.DAYS_IN_YEAR`: `read_cache` builds
    `time_grid_years` with the day count, so a 365.25ths expiry would miss its node by 7e-4 years.
    """
    from derivus import bootstrappers, instruments
    out, curve = {}, world['curve']
    for instrument in world['block']['Instrument_Definitions']:
        name = 'Swaption_{}_{}'.format(bootstrappers.date_fmt(instrument['Start']),
                                       bootstrappers.date_fmt(instrument['Tenor']))
        effective = BASE + instrument['Start']
        dates = instruments.generate_dates_backward(
            effective + instrument['Tenor'], effective, instrument['Fixed_Frequency'])
        cash = utils.generate_fixed_cashflows(
            BASE, dates, 1.0, None, utils.get_day_count(instrument['Fixed_Day_Count']), 0.0)
        pay_days = cash.schedule[:, utils.CASHFLOW_INDEX_Pay_Day]
        tau = cash.schedule[:, utils.CASHFLOW_INDEX_Year_Frac]
        exp_days = (effective - BASE).days
        T0 = curve.get_day_count_accrual(BASE, exp_days)
        pay_times = curve.get_day_count_accrual(BASE, pay_days)
        P = np.exp(-curve.current_value(pay_times) * pay_times)
        P0 = float(np.exp(-curve.current_value(np.array([T0]))[0] * T0))
        strike = float((P0 - P[-1]) / (tau * P).sum())

        schedule = world['swaps'][name].deal_data.Factor_dep['Cashflows'].schedule
        nz = np.flatnonzero(schedule[:, utils.CASHFLOW_INDEX_FixedAmt] != 0.0)
        assert np.array_equal(schedule[nz, utils.CASHFLOW_INDEX_Pay_Day], pay_days), name
        written = schedule[nz, utils.CASHFLOW_INDEX_FixedAmt]
        assert np.abs(-strike * tau / written - 1.0).max() < 1e-12, name
        out[name] = dict(T0=T0, pay_times=pay_times, tau=tau, K=strike, pay_days=pay_days,
                         exp_days=exp_days, float_pay=schedule[:, utils.CASHFLOW_INDEX_Pay_Day],
                         coupon=-written)
    return out


def checker_mc(world, leg_map, num_batches, batch_size):
    """Per-BATCH means of every route, `{benchmark: {route: array(num_batches)}}`.

    The harness holds its own batch loop because `mean_se` needs the batches apart. It clears
    `t_Buffer` per iteration, as `calc_loss_on_ir_curve` now does, and leaves `t_PreCalc` alone -
    that holds `precalculate`'s integrals and is a function of theta, not of the sample.

    The routes, all off the SAME paths so every difference is a common-random-number one:

        A    the engine       E[ D(0,T0) relu( pv_float_cashflow_list ) ] - what the objective prices
        B    the convention   E[ D(0,T0) relu( 1 - P(T0,Tn) - sum c_i P(T0,Ti) ) ]
        W    the numeraire    E[ D(0,T0) A(T0) ], which is A(0) identically
        WS, WS2               the same weight against S(T0) and S(T0)^2, so the swap rate's
                              annuity-measure mean and variance come off one simulation.
                              `A(T0) S(T0)` is `1 - P(T0,Tn)` identically.
    """
    from derivus import bootstrappers, instruments, pricing
    process, time_grid, model = world['process'], world['time_grid'], world['model']
    ir_factor = world['ir_factor']
    all_tenors = utils.update_tenors(BASE, {ir_factor: process})
    index_keys = {'full': utils.Factor(ir_factor.type, ir_factor.name + ('full',)),
                  'reduced': utils.Factor(ir_factor.type, ir_factor.name + ('reduced',))}
    c_index = instruments.calc_factor_index(ir_factor, {}, {ir_factor: process}, all_tenors)
    reduced = [(c_index[utils.FACTOR_INDEX_Stoch], index_keys['reduced']) + c_index[2:]]
    shared = bootstrappers.RiskNeutralInterestRate_State(
        index_keys, batch_size, model.device, model.prec)
    for md in world['swaps'].values():
        utils.bind_schedules(md.deal_data.Factor_dep, shared.one)

    stoch_var = torch.tensor(process.factor.current_value(), device=model.device, dtype=model.prec)
    shared.reset(num_batches, process.num_factors(), time_grid)
    process.precalculate(BASE, time_grid, stoch_var, shared, 0, implied_tensor=world['implied_var'])
    delta_scen_t = np.diff(time_grid.scen_time_grid).reshape(-1, 1)
    routes = ('A', 'B', 'W', 'WS', 'WS2')
    out = {name: {r: np.zeros(num_batches) for r in routes} for name in leg_map}

    for batch in range(num_batches):
        shared.t_Buffer.clear()
        shared.batch_index = batch
        shared.t_Scenario_Buffer = process.generate(shared)
        deflation = utils.calc_time_grid_curve_rate(
            reduced, time_grid.calc_time_grid(time_grid.scen_time_grid[:-1]), shared
        ).reduce_deflate(delta_scen_t, time_grid.mtm_time_grid, shared)

        for name, md in world['swaps'].items():
            leg, deal_data = leg_map[name], md.deal_data
            DtT = deflation[deal_data.Time_dep.mtm_time_grid[deal_data.Time_dep.deal_time_grid[0]]]
            swap = pricing.pv_float_cashflow_list(
                shared, time_grid, deal_data, pricing.pricer_float_cashflows, settle_cash=False)
            deal_time = time_grid.time_grid[deal_data.Time_dep.deal_time_grid]
            block = utils.calc_time_grid_curve_rate(
                deal_data.Factor_dep['Discount'], deal_time, shared)
            days = np.concatenate((leg['pay_days'], [leg['float_pay'][-1]]))
            P = utils.calc_discount_rate(
                block, days.reshape(1, -1) - deal_time[:, utils.TIME_GRID_MTM].reshape(-1, 1),
                shared)[0]
            coupon = P.new_tensor(leg['coupon']).reshape(-1, 1)
            tau = P.new_tensor(leg['tau']).reshape(-1, 1)
            annuity = (tau * P[:-1]).sum(dim=0)
            convention = 1.0 - P[-1] - (coupon * P[:-1]).sum(dim=0)
            for route, value in (('A', torch.relu(DtT * swap)),
                                 ('B', torch.relu(DtT * convention)),
                                 ('W', DtT * annuity), ('WS', DtT * (1.0 - P[-1])),
                                 ('WS2', DtT * (1.0 - P[-1]) ** 2 / annuity)):
                out[name][route][batch] = float(torch.sum(value.detach()) / batch_size)
    return out


def mean_se(per_batch):
    """`(mean, standard error)` off the batch means. The Sobol sample is drawn ONCE and split into
    consecutive blocks, so this is a batch-mean error bar and never a confidence interval."""
    x = np.asarray(per_batch, dtype=np.float64)
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(x.size))


def checker_readings(world, leg_map, per_batch):
    """`{benchmark: reading}` - the analytic price, the simulated one, and the split between them."""
    out = {}
    for name, leg in leg_map.items():
        pb, T0, K = per_batch[name], leg['T0'], leg['K']
        sp = world['process'].schrager_pelsser_swaption(T0, leg['pay_times'], leg['tau'])
        premium, se = mean_se(pb['A'])
        convention, _ = mean_se(pb['B'])
        weight = mean_se(pb['W'])[0]
        annuity = as_float(sp.annuity)
        mean_S = mean_se(pb['WS'])[0] / weight
        var_S = mean_se(pb['WS2'])[0] / weight - mean_S * mean_S
        sd = float(np.sqrt(max(var_S, 0.0)))
        out[name] = dict(
            sp=as_float(sp.premium), mc=premium, mc_se=se, convention=premium - convention,
            numeraire=weight / annuity - 1.0, sp_nvol=as_float(sp.normal_vol) * 1e4,
            sim_nvol=sd / np.sqrt(T0) * 1e4, drift=(mean_S - K) / K, annuity=annuity, T0=T0,
            sigma=(as_float(sp.premium) - premium) / se, variance=as_float(sp.variance),
            loadings=[as_float(x) for x in sp.loadings])
    return out


@pytest.fixture(scope='module')
def checker():
    """`(world, legs, readings)` at the recorded theta*: ONE simulation for every gate below.
    `Objective: 'Monte_Carlo'` DECLARED - the checker IS the simulation.
    """
    world = identified_closure(Objective='Monte_Carlo')
    world['loss'](world['implied_var'])
    leg_map = checker_legs(world)
    return world, leg_map, checker_readings(
        world, leg_map, checker_mc(world, leg_map, 8, 8192))


def test_the_single_curve_float_leg_is_exact_and_not_an_approximation(checker):
    """NO leg-convention term in the SP-vs-MC gap: SP prices `1 - sum c_i P(T0,Ti)` and the Monte
    Carlo prices `pv_float_cashflow_list`, and they are the SAME number path by path.

    One curve for both `Forward` and `Discount`, each float coupon's pay day on its accrual end and
    its `Year_Frac` its reset's own accrual, so a coupon is `P(t,t_i) - P(t,T_i)` and the leg
    telescopes to `1 - P(T0,Tn)` identically. The frequencies therefore do not matter: 3M/3M,
    3M/6M and 3M/12M are all held. |A - B| is at most 5.6e-18 against premiums of 0.006 to 0.06.

    A second curve would break it; `create_market_swaps` has no basis-curve path.
    """
    _, _, readings = checker
    for name, r in readings.items():
        assert abs(r['convention']) / r['mc'] < 1e-13, (
            '{}: the engine\'s float leg and 1 - sum c_i P differ by {:.3e} relative, so the '
            'single-curve telescoping this gate exists to confirm has stopped holding'.format(
                name, r['convention'] / r['mc']))
    # and the differing-frequency rows are actually on the path, or the gate proves nothing
    assert 'Swaption_2Y_5Y' in readings and 'Swaption_3Y_3Y' in readings


def test_the_analytic_vol_is_the_simulated_vol_and_where_it_stops_being(checker):
    """WHAT SP'S FREEZING COSTS, in basis points of normal vol against the swap rate the simulation
    produces. Vol against vol and not premium against premium, because a premium comparison also
    carries the simulation's numeraire error - the larger of the two here. `sim_nvol` is the
    annuity-measure s.d. of `S(T0)` off the same paths, so both ratios divide that error out.

    Over the 5x5 grid at 1048576 paths the bias runs -0.13bp to +2.17bp against a level of 178 to
    209. It is SIGNED - low at the short end, high at the long, crossing zero along the 3Y expiry -
    and grows with the PRODUCT of expiry and tenor: under 0.21bp at every expiry for tenors to 3Y,
    and the 10Y tenor column carries all of it.

    Against the noise of ONE objective evaluation, read off the SAME statistic over 64
    re-scramblings of the Sobol rule at 8192 paths: 0.51 to 1.81bp. SP therefore sits INSIDE one
    evaluation's noise at 22 of 25 benchmarks and steps outside at 3Y x 10Y, 5Y x 10Y and
    10Y x 10Y, each by about a fifth of a standard deviation. Good enough to BE the objective
    inside a 10Y tenor; marginal on that column, where a Gauss-Hermite quadrature would earn its
    keep. The premium estimator's noise is a DIFFERENT number (0.61 to 0.86bp) because the scramble
    buys 3-4x on the first moment and only 1.2-2.5x on the second, so it cannot stand in for this.

    Taken at this fixture's own correlation; `test_the_bias_rides_the_correlation_axis` sweeps the
    third axis, which is why this grid is not the whole measurement.
    """
    _, _, readings = checker
    for name, bound in (('Swaption_1Y_1Y', 0.5), ('Swaption_2Y_5Y', 0.5), ('Swaption_3Y_3Y', 0.5)):
        gap = readings[name]['sp_nvol'] - readings[name]['sim_nvol']
        assert abs(gap) < bound, '{}: SP is {:+.3f}bp off the simulated vol'.format(name, gap)
    # and the corner, held from BOTH sides - a regression that made it small would mean the
    # simulation had stopped being the reference, not that the approximation had improved
    corner = readings['Swaption_10Y_10Y']
    gap = corner['sp_nvol'] - corner['sim_nvol']
    assert 1.2 < gap < 4.5, (
        '10Y x 10Y reads {:+.3f}bp against a recorded 2.2 - the freezing bias is the one thing this '
        'measurement is for, and it has moved'.format(gap))
    # the swap rate IS a martingale under its own annuity measure, which is what makes sim_nvol a
    # reference. 24 of the 25 read under 5.1e-4; the 10Y x 10Y corner reads -8.9e-3, which is where
    # theta*'s correlation on the -0.95 end puts it and is why the bound is 1.2e-2.
    for name, r in readings.items():
        assert abs(r['drift']) < 1.2e-2, '{}: E^A[S]/K - 1 is {:.3e}'.format(name, r['drift'])
    assert max(abs(r['drift']) for n, r in readings.items()
               if n != 'Swaption_10Y_10Y') < 1e-3, (
        'the martingale miss has spread off the 10Y x 10Y corner, where it is the level rho puts '
        'on the model - the other 24 benchmarks read under 5.1e-4')


#: The correlation sweep's own world: a short benchmark, the mixed-frequency one, and both
#: 10Y-tenor corners. Its own `TimeGrid` and therefore its own Sobol sample, so its numbers are not
#: the 25-benchmark grid's to a basis point.
RHO_BENCHMARKS = ((1, 1, 3, 3), (2, 5, 3, 6), (5, 10, 3, 3), (10, 10, 3, 3))
#: the ends of `HullWhite2FactorModelParameters.corr_bounds`
RHO_ENDS = (-0.95, 0.95)


@pytest.fixture(scope='module')
def rho_sweep():
    """`{rho: (readings, params_ok)}` at the two ends of `corr_bounds`, off ONE world and leg map.

    `Correlation` moves through `implied_var`, so both routes see the same parameter. Its own world
    rather than the checker's: a module fixture whose parameters a gate mutated would hand every
    gate after it a different theta. `Objective: 'Monte_Carlo'` declared.
    """
    world = identified_closure(benchmarks=RHO_BENCHMARKS, Objective='Monte_Carlo')
    world['loss'](world['implied_var'])
    leg_map = checker_legs(world)
    out = {}
    for rho in RHO_ENDS:
        world['implied_var']['Correlation'].data = torch.tensor([rho], dtype=DTYPE)
        out[rho] = (checker_readings(world, leg_map, checker_mc(world, leg_map, 8, 8192)),
                    world['process'].params_ok)
    return out


def test_the_bias_rides_the_correlation_axis(rho_sweep):
    """THE THIRD AXIS, which the expiry x tenor grid cannot see. `rho` enters this price in exactly
    one place - the cross term `2 rho q_1 q_2 J_12` - and the fixture solves it near the -0.95 end,
    where that term is worth a fraction of a percent of the variance.

    SP minus simulated at the two ends of `corr_bounds`, 65536 paths: the 10Y x 10Y corner reads
    +2.66bp at -0.95 and +21.58bp at +0.95, while 1Y x 1Y stays flat (-0.130 to +0.122) - so the
    growth belongs to the tenor and to rho, not to the level. rho moves the level itself 2.4x,
    183-202bp to 430-519bp, because at +0.95 the two factors reinforce.

    At +0.95 THE SIMULATION STOPS BEING A REFERENCE: the swap rate misses its own annuity-measure
    martingale by +11.8% at that corner and `E[D A]/A(0) - 1` reads -16.6%. That column is two
    estimators out of range rather than "SP is 21.58bp wrong", so the martingale bound of 1.2e-2 is
    asserted only where the simulation still meets it. Noise on `sim_nvol` is 0.4 to 2.1bp across
    the axis, which is what makes the gaps readable at all.
    """
    for rho in RHO_ENDS:
        _, ok = rho_sweep[rho]
        assert ok, 'params_ok went False at rho = {:+g} - the Cholesky has to survive the corner ' \
                   'before anything read there means anything'.format(rho)
    low, high = (rho_sweep[rho][0] for rho in RHO_ENDS)

    def gap(readings, name):
        return readings[name]['sp_nvol'] - readings[name]['sim_nvol']

    # the axis is load-bearing: rho moves the level by half again before any bias is read off it
    assert high['Swaption_10Y_10Y']['sp_nvol'] / low['Swaption_10Y_10Y']['sp_nvol'] > 1.4, (
        'rho barely moves the analytic vol, so this sweep is not sweeping anything')
    # the corner, held from both sides at each end
    assert 1.5 < gap(low, 'Swaption_10Y_10Y') < 4.0, (
        '10Y x 10Y at rho = -0.95 reads {:+.3f}bp against a recorded 2.66'.format(
            gap(low, 'Swaption_10Y_10Y')))
    assert 15.0 < gap(high, 'Swaption_10Y_10Y') < 30.0, (
        '10Y x 10Y at rho = +0.95 reads {:+.3f}bp against a recorded 21.58 - the whole point of '
        'this sweep is that the corner is worse than the solved rho says'.format(
            gap(high, 'Swaption_10Y_10Y')))
    # and it GROWS with rho on the 10Y tenor, which is the finding rather than the level
    for name in ('Swaption_10Y_10Y', 'Swaption_5Y_10Y'):
        assert gap(high, name) > 2.0 * gap(low, name), (
            '{}: {:+.3f}bp at +0.95 against {:+.3f} at -0.95'.format(
                name, gap(high, name), gap(low, name)))
    # while the short benchmark is flat in it, so the growth belongs to the tenor and not to rho
    assert abs(gap(high, 'Swaption_1Y_1Y') - gap(low, 'Swaption_1Y_1Y')) < 0.4
    # the martingale, to the tolerance the SIMULATION allows at this end of the axis
    for name, r in low.items():
        assert abs(r['drift']) < 1.2e-2, '{}: E^A[S]/K - 1 is {:.3e}'.format(name, r['drift'])
    # at the other end the simulation is out of range on the 10Y tenor; the short and mid
    # benchmarks still meet the bound, so the breakdown is the corner's and not the sample's
    for name in ('Swaption_1Y_1Y', 'Swaption_2Y_5Y'):
        assert abs(high[name]['drift']) < 1.2e-2, '{}: E^A[S]/K - 1 is {:.3e}'.format(
            name, high[name]['drift'])
    assert high['Swaption_10Y_10Y']['drift'] > 5e-2, (
        'the +0.95 corner no longer breaks the martingale - it read +11.8% and that is what this '
        'sweep found: {:.3e}'.format(high['Swaption_10Y_10Y']['drift']))
    assert high['Swaption_10Y_10Y']['numeraire'] < -5e-2, (
        'and the numeraire with it, at a recorded -16.6%: {:.3e}'.format(
            high['Swaption_10Y_10Y']['numeraire']))


def test_the_monte_carlo_carries_a_bias_of_its_own(checker):
    """`E[D(0,T0) A(T0)] = A(0)` is an identity, so this reads the SIMULATION'S error - -0.35% to
    -1.61% across the grid, the larger of the two on this fixture.

    Not discretisation: refining the scenario grid from ten-daily to daily (375 nodes to 3654)
    moves it by 2%, where a rollover numeraire's O(dt) error would fall by ten. It is the CURVE's
    tenor grid - the first node is at ONE YEAR and `reduce_deflate` asks for a ten-day rate,
    flat-extrapolated off it. Adding 1D/1M/3M/6M nodes collapses it two orders, to -3.1e-5.

    So on the fixture as authored MOST of the SP-vs-MC premium difference is the simulation's: at
    2Y x 5Y the premium gap is 0.67% while the vol gap is 0.05bp out of 193.
    """
    _, _, readings = checker
    for name, r in readings.items():
        assert -0.03 < r['numeraire'] < -1e-3, (
            '{}: E[D A]/A(0) - 1 is {:.3e} against a recorded -0.0035 to -0.0166 - this fixture\'s '
            'curve starts at a 1Y tenor and that is what the number measures'.format(
                name, r['numeraire']))
    # the premium difference is dominated by it, which is the reason the gate above reads vols
    r = readings['Swaption_2Y_5Y']
    assert abs(r['sp_nvol'] - r['sim_nvol']) < abs(r['numeraire']) * r['sp_nvol'], (
        'the numeraire error no longer dominates the vol gap at 2Y x 5Y, so the premium comparison '
        'may now be the honest one and this file\'s reasoning needs re-reading')


def test_reinstating_the_prefactor_is_visible_against_the_simulation(checker):
    """THE DEFECT THE CHECKER FOUND, kept as a mutation so it cannot come back quietly.

    `Var` used to carry `e^{-(alpha_k+alpha_l)T_0}` in front of `J_{kl}(T_0)`. It does not belong:
    `q` is the loading on the SCALED martingale `Y_k = e^{alpha_k t} x_k` and `J` is `Y`'s own
    covariance, so a prefactor scales the state twice.

    Reinstating it moves the normal vol by -18.1% to +13.5% of the level - not one-signed, because
    `e^{-2 alpha T_0}` goes above one where a speed solves negative - and hundreds of Monte Carlo
    standard errors at half a million paths. Held here at 4x the repaired error at every benchmark.
    """
    world, leg_map, readings = checker
    process = world['process']
    grid = process.cache['time_grid_years']
    alpha = np.array([as_float(a) for a in process.alpha])
    for name, r in readings.items():
        T0 = leg_map[name]['T0']
        node = int(np.searchsorted(grid, T0))
        assert grid[node] == T0, name
        q = np.array(r['loadings'])
        rho01 = as_float(process.rho[0][1].reshape(()))
        R = np.array([[1.0, rho01], [rho01, 1.0]])
        J = np.array([[as_float(process.J[k][l][node]) for l in range(2)] for k in range(2)])
        mutated = float((R * np.outer(q, q) * J * np.exp(-np.add.outer(alpha, alpha) * T0)).sum())
        assert abs(np.sqrt(mutated / r['variance']) - 1.0) > 1e-3, name
        mutated_nvol = np.sqrt(mutated / T0) * 1e4
        assert abs(mutated_nvol - r['sim_nvol']) > 4.0 * abs(r['sp_nvol'] - r['sim_nvol']), (
            '{}: with the prefactor back the vol is {:.2f}bp against a simulated {:.2f} and a '
            'repaired {:.2f} - if that is no longer a large multiple of the repaired error, the '
            'mutation has stopped being detectable'.format(
                name, mutated_nvol, r['sim_nvol'], r['sp_nvol']))
    # the corner, with the number
    corner = readings['Swaption_10Y_10Y']
    assert corner['sp_nvol'] > 180.0 and corner['sim_nvol'] > 178.0


def test_batches_now_buy_paths_and_shrink_the_estimates_spread():
    """`Batches` buys paths, and it did not: `calc_loss_on_ir_curve` cleared `t_Buffer` once before
    the loop, and `calc_time_grid_curve_rate` keys its cache on the curve and the time grid rather
    than the batch, so batch 1 onward re-gathered batch 0's simulated curve. Prices were
    BIT-IDENTICAL at 1, 4 and 8 batches at a cost linear in the count.

    Both counts are DECLARED now (`Simulations`, `Batches`) and the loop clears `t_Buffer` - not
    `t_PreCalc`, which is a function of theta - per iteration. Two things are held.

    THE IDENTITY: `reset` draws `Simulations x Batches` Sobol points ONCE and reshapes, so
    (2048 x 4) and (8192 x 1) are the same points in the same order and agree to 2.2e-16, one ulp.
    THE MOVE: at 2048 a batch, four batches read 30bp of relative value away from one, where before
    it was zero to the last bit. Not a root-B law - consecutive blocks of a scrambled Sobol rule
    are not independent draws - so the gate holds the identity and the move, not a variance law.

    `Objective: 'Monte_Carlo'` DECLARED: the analytic closure draws no sample to block.
    """
    estimate = {}
    for batches in (1, 2, 4, 8):
        world = identified_closure(benchmarks=((1, 1, 3, 3), (5, 5, 3, 3)), batch_size=2048,
                                   Objective='Monte_Carlo', Batches=batches)
        prices, _ = world['loss'](world['implied_var'])
        estimate[batches] = {k: as_float(v) for k, v in prices.items()}

    for name in estimate[1]:
        moved = abs(estimate[4][name] / estimate[1][name] - 1.0)
        assert moved > 1e-4, (
            '{} at Batches=4 is {:.12g} against {:.12g} at 1 - identical to that many figures means '
            'the buffer is being cleared once outside the loop again and the batches are re-pricing '
            'batch zero'.format(name, estimate[4][name], estimate[1][name]))

    # the identity: the same 8192 points, four ways of blocking them
    whole = identified_closure(benchmarks=((1, 1, 3, 3), (5, 5, 3, 3)), batch_size=8192,
                               Objective='Monte_Carlo')
    pooled, _ = whole['loss'](whole['implied_var'])
    pooled = {k: as_float(v) for k, v in pooled.items()}
    for sims, batches in ((4096, 2), (2048, 4), (1024, 8)):
        split = identified_closure(benchmarks=((1, 1, 3, 3), (5, 5, 3, 3)), batch_size=sims,
                                   Objective='Monte_Carlo', Batches=batches)
        prices, _ = split['loss'](split['implied_var'])
        for name, value in prices.items():
            assert abs(as_float(value) / pooled[name] - 1.0) < 1e-14, (
                '{} at {} x {} is {:.17g} against {:.17g} at 8192 x 1 - those are the SAME Sobol '
                'points, so a difference means the batch loop is not walking all of them'.format(
                    name, sims, batches, as_float(value), pooled[name]))


def declared_shape_closure(objective, *popped):
    """`(the spelled-out world, the popped world, its implied_var, its objective)` at theta*.

    The absent half POPS `popped` off an otherwise identical block and re-enters
    `calc_loss_on_ir_curve` through its public signature, so the engine's own `.get` fallback is
    what runs. `pop` without a default: a key the block does not carry is a gate measuring nothing.
    """
    spelled = identified_closure(benchmarks=((1, 1, 3, 3),), batch_size=8192, Objective=objective)
    world = identified_closure(benchmarks=((1, 1, 3, 3),), batch_size=8192, Objective=objective)
    for key in popped:
        world['block'].pop(key)
    implied_var, chosen, _, _ = world['model'].calc_loss_on_ir_curve(
        {'instrument': world['block']}, BASE, world['time_grid'], world['process'],
        world['implied_obj'], world['ir_factor'], world['surface'])
    for name, value in ID_THETA.items():
        implied_var[name].data = torch.tensor(value, dtype=DTYPE, device=DEVICE)
    return spelled, world, implied_var, chosen


def test_the_declared_sample_shape_is_the_shape_the_engine_uses():
    """A block omitting a declared field and a block spelling out its default are the same job -
    read off the state the closure built and the residual it returns, where the schema-emission
    suite holds only the AST half (`.get`'s fallback and the `F` default are one number).

    TWO arms. `Objective` absent builds the ANALYTIC closure, told apart by the auditor it carries
    (`reprice` is `None` on the path that IS the estimator). `Simulations` and `Batches` shape a
    sample the analytic closure never draws, so popping them there would be vacuous - that half is
    taken on a second block DECLARING `Monte_Carlo`. Both arms bit-identical.
    """
    for field, value in (('Simulations', 8192), ('Batches', 1), ('Objective', 'Analytic')):
        f = next(x for x in HullWhite2FactorModelParameters.fields if x.name == field)
        assert f.default == value, (field, f.default)

    # the DEFAULT arm: all three absent, and the field absent is the analytic objective
    spelled, world, implied_var, chosen = declared_shape_closure(
        'Analytic', 'Simulations', 'Batches', 'Objective')
    assert (world['model'].batch_size, world['model'].num_batches) == (8192, 1), (
        'a block omitting Simulations and Batches did not fall back to the declared 8192 and 1')
    assert chosen.reprice is not None, (
        'the Objective absent has to BE the Analytic objective, which carries the Monte Carlo '
        'closure as its auditor - a `None` there is the pre-flip default coming back')
    assert spelled['objective'].reprice is not None
    absent = chosen.loss(implied_var)[1]
    said = spelled['loss'](spelled['implied_var'])[1]
    for name in absent:
        assert as_float(absent[name]) == as_float(said[name]), (
            '{}: the field absent reads {:.17g} against {:.17g} spelled Analytic'.format(
                name, as_float(absent[name]), as_float(said[name])))

    # the SAMPLE arm, on the path where a sample exists
    spelled, world, implied_var, chosen = declared_shape_closure(
        'Monte_Carlo', 'Simulations', 'Batches')
    assert (world['model'].batch_size, world['model'].num_batches) == (8192, 1)
    assert chosen.reprice is None, 'the Monte Carlo objective audits nothing - it IS the estimator'
    absent = chosen.loss(implied_var)[1]
    said = spelled['loss'](spelled['implied_var'])[1]
    for name in absent:
        assert as_float(absent[name]) == as_float(said[name]), (
            '{}: the sample shape absent reads {:.17g} against {:.17g} spelled out'.format(
                name, as_float(absent[name]), as_float(said[name])))


def test_the_solved_fixture_no_longer_engages_the_series_branch_and_that_is_the_finding():
    """theta* is not the seed, and this fixture's calibrated point used to sit inside the series
    region - `Alpha_2` solved to -0.017851, so `I[1][*]` took the branch at the repository's own
    solved vector. It solves to +0.059856 since the 2026-09-02 re-mark.

    What the gate records is REACHABILITY, which has not changed: the branch is not engaged at this
    theta* (no gate here relies on it), the bound that made it reachable still straddles zero, and
    the margin is stated - 2.0x the IJK threshold on `Alpha_2`, not ten rungs. The series code is
    driven directly at fixed speeds by the gates above; what this fixture stopped being is the
    live witness.
    """
    alpha_1, alpha_2 = ID_THETA['Alpha_1'][0], ID_THETA['Alpha_2'][0]
    assert alpha_2 > 0.0, (
        'the solved Alpha_2 is {:.6g}: this fixture is back inside the region the gate was '
        're-based off, and the docstring above needs re-taking'.format(alpha_2))
    # nothing this theta* is read at takes a series branch
    for a in (alpha_1, alpha_2, alpha_1 + alpha_2, 2.0 * alpha_1, 2.0 * alpha_2):
        assert abs(a) > HW_ALPHA_SERIES_IJK, (
            'a reversion speed this theta* is read at, {:.6g}, is inside the IJK threshold '
            '{:g}'.format(a, HW_ALPHA_SERIES_IJK))
    # and the margin, stated: the tightest of them against the highest threshold
    tightest = min(abs(a) for a in (alpha_1, alpha_2, alpha_1 + alpha_2))
    assert tightest / HW_ALPHA_SERIES_IJK > 1.5, (
        'the tightest reversion speed is {:.4g}, only {:.2f}x the IJK threshold - the recorded '
        'margin is 2.0x on Alpha_2'.format(tightest, tightest / HW_ALPHA_SERIES_IJK))
    # the reachability claim itself is unchanged and lives on the BOX, not on this vector
    low, high = HullWhite2FactorModelParameters({}, DEVICE, DTYPE).alpha_bounds
    assert low < 0.0 < high, 'the bound that made the series region reachable has moved'


def test_the_recorded_theta_is_a_stationary_point_of_the_recorded_world(checker):
    """`ID_THETA` still prices its own world's benchmarks within 10% of the market, against the 4.4%
    the solve reached. Not a re-solve - that costs 2304 s, and
    `docs_src/developer/quote_sensitivities.md` records that the chain does not reach stationarity
    on an over-determined block anyway - but the checkable claim that vector and world still match.
    """
    world, _, _ = checker
    prices, _ = world['loss'](world['implied_var'])
    for name, value in prices.items():
        market = world['swaps'][name].price
        assert 0.0 < as_float(value) < 1.0, name
        assert abs(as_float(value) / market - 1.0) < 0.10, (
            '{} prices {:.6g} against a market {:.6g} - theta* fitted this grid to 4.4% at its '
            'worst, so a tenth means the recorded vector and the recorded world have parted'.format(
                name, as_float(value), market))


# ------------------------------- THE ANALYTIC OBJECTIVE: the declared field, and what it retires

#: The identified fixture's own 5x5 grid, which is what `ID_THETA` was solved on - the four
#: `CHECKER_BENCHMARKS` rows are a subset plus two mixed-frequency variants.
ID_GRID = tuple((e, t, 3, 3) for e in (1, 2, 3, 5, 10) for t in (1, 2, 3, 5, 10))

#: theta* on that grid under `Objective: 'Analytic'`, AS SOLVED - the analytic twin of `ID_THETA`,
#: so a gate reads it without paying 204 s. `Random_Seed` 5120, CPU float64. Written at `repr`
#: precision, which round-trips a float64: transcribed at sixteen digits the previous vector came
#: back one ulp out. It fits the grid at 2.9594bp rms against `ID_THETA`'s 4.54.
#:
#: That it is a solve output is checkable in seconds:
#: `test_the_analytic_objective_reaches_a_stationarity_the_quartic_cannot` reads `||J'r||` 8.63e-7
#: here, which no authored vector lands on. It carries ONE active bound, `Correlation` on the -0.95
#: end, where the retired vector carried a sigma knot on the 1e-5 floor and one on the ceiling -
#: `test_the_corner_moved_off_the_sigma_floor_and_what_is_left_is_the_fixtures_own_minimum` asserts
#: that and the profile that says the bound is the fixture's own minimum.
ID_ANALYTIC_THETA = {
    'Alpha_1': [0.2813034827193],
    'Alpha_2': [0.073731864540266],
    'Correlation': [-0.9499987110160323],
    'Sigma_1': [0.011751240023735926, 0.01772698571220824, 0.018714771671368008,
                0.027410420015248016, 0.022977367549116015, 0.009740267235151446,
                0.07253821496125087, 0.0899861890705156, 0.08466199615274374,
                0.08023573225258275],
    'Sigma_2': [0.016275786547270594, 0.025608424932383113, 0.02695027130043976,
                0.04443426160914259, 0.03511101780395394, 0.030027986018198727,
                0.05853341042527687, 0.008795354296570166, 0.07069010279253891,
                0.06961435242058203]}

#: theta* on the FOUR-quote fixture under `Objective: 'Monte_Carlo'`, as solved on this box in
#: float64 - the bit-identity baseline `test_the_monte_carlo_objective_still_solves_to_this_vector`
#: re-derives. It is a fast factor beside a nearly driftless one, which is the shape the separated
#: seed was chosen to reach.
MC_FOUR_THETA = {
    'Alpha_1': [0.8838200347991989],
    'Alpha_2': [0.003535909908850363],
    'Correlation': [-0.6420711771679645],
    'Sigma_1': [0.012048204866639782, 0.022289393326153036, 0.012131697402018596,
                0.017336645235464358, 0.017761506195400695, 0.027001440506809755,
                0.03867671946670284, 0.0051682892665788505, 0.02420097457038873,
                0.021719967510674714],
    'Sigma_2': [0.024878038790869982, 0.020798888165221196, 0.020152884562528145,
                0.019816401225002778, 0.022499761487407982, 0.02053355304739493,
                0.01835945160571706, 0.018447529927269692, 0.018423964357339267,
                0.018494757668385337]}

#: theta* on the FOUR-quote fixture under `Objective: 'Analytic'`, AS SOLVED - the analytic twin of
#: `MC_FOUR_THETA`, so the theta-comparison is taken on the under-determined block as well as the
#: identified one. `Random_Seed` 5120, CPU float64, 15.4 s over 403 evaluations against 81.8 s over
#: 133 for the vector above. `||J'r||` 7.66e-9 against `||r||` 2.29e-8: four quotes against 23
#: parameters, so it INTERPOLATES, and no authored vector lands there.
AN_FOUR_THETA = {
    'Alpha_1': [0.2849546414778107],
    'Alpha_2': [0.034205762238456776],
    'Correlation': [-0.038872809713065115],
    'Sigma_1': [0.01477079675178202, 0.007501737216816033, 0.010196504317133285,
                0.008804083972127764, 0.005551965570300839, 0.00851406227509759,
                0.012679137814980049, 0.0040191968265081346, 0.010782787260332296,
                0.01219901242331459],
    'Sigma_2': [0.0076506352073352595, 0.013074727254761297, 0.012667036797553018,
                0.008549497758875578, 0.030606943635628478, 0.01921286010311337,
                0.013954072058773867, 0.02168436210611065, 0.029162520715096445,
                0.03146698326207222]}


def flat_theta(calibration, named):
    """`named` as the flat vector `SwaptionCalibration` speaks in, in ITS parameter order."""
    return torch.tensor(np.concatenate([np.atleast_1d(named[k]) for k in calibration.keys]),
                        dtype=DTYPE)


def stationarity(calibration, theta):
    """`(||J'r||, ||r||)` at a flat theta - the quantity `LeastSquaresSolve.backward` refuses on,
    read the way that backward reads it: one fresh evaluation through
    `SwaptionCalibration.__call__`, one `autograd.grad` per row, promoted to float64.
    """
    x = theta.detach().clone().requires_grad_(True)
    residual = calibration(x)
    jacobian = torch.stack([torch.autograd.grad(residual[i], x, retain_graph=True)[0]
                            for i in range(residual.numel())]).double()
    residual = residual.detach().double()
    return float((jacobian.t() @ residual).norm()), float(residual.norm())


def calibration_at(theta, benchmarks=ID_GRID, **extra):
    """`(SwaptionCalibration, world)` standing at `theta` - no optimizer chain, so nothing solves."""
    world = identified_closure(benchmarks=benchmarks, theta=theta, **extra)
    return SwaptionCalibration('gate', world['objective'], world['implied_var'], None,
                               world['process'], world['swaps']), world


def declared(name):
    """The DECLARED default of one `HullWhite2FactorModelParameters` field. The analytic quote side
    is contracted to run at `Stationarity_Tol`'s own 1e-3, so gates read it off the schema.
    """
    return next(f for f in HullWhite2FactorModelParameters.fields if f.name == name).default


def quoted_definitions(benchmarks, vols):
    """`Instrument_Definitions` with each row's ATM vol AUTHORED, in vol points. RE-AUTHORING is
    how a quote-derivative rung is taken: the block is rebuilt from the JSON up, so the difference
    quotient goes AROUND the splice instead of reading the tape back to itself.
    """
    rows = identified_definitions(benchmarks)
    for row, vol in zip(rows, vols):
        row['Market_Volatility'] = utils.Percent(vol)
    return rows


def quote_calibration(theta=None, benchmarks=ID_GRID, vols=None, chain=False, **extra):
    """`(SwaptionCalibration, world)` on the identified fixture with the ANALYTIC quote side on.
    `Stationarity_Tol` is DELETED rather than declared: this path runs at the field's own 1e-3
    where the Monte Carlo one has to write itself a 1e5.
    """
    world = identified_closure(
        benchmarks=benchmarks, theta=theta, chain=chain, Objective='Analytic',
        Quote_Sensitivity='Yes', Stationarity_Tol=ABSENT,
        Instrument_Definitions=quoted_definitions(
            benchmarks, [20.0] * len(benchmarks) if vols is None else vols), **extra)
    return SwaptionCalibration('gate', world['objective'], world['implied_var'],
                               world['optimizers'], world['process'], world['swaps']), world


def residual_pieces(calibration, theta):
    """`(r, J, dr/dq)` at a flat theta - the three objects `LeastSquaresSolve.backward` reads, one
    fresh evaluation and one `autograd.grad` per row, promoted to float64.
    """
    x = theta.detach().clone().requires_grad_(True)
    residual = calibration(x)
    jacobian = torch.stack([torch.autograd.grad(residual[i], x, retain_graph=True)[0]
                            for i in range(residual.numel())]).double()
    quote_jac = torch.stack([
        torch.cat([g.reshape(1) for g in torch.autograd.grad(
            residual[i], calibration.quotes, retain_graph=True)])
        for i in range(residual.numel())]).double()
    return residual.detach().double(), jacobian, quote_jac


def unconnected_pairs(calibration, theta):
    """How many (benchmark, quote) pairs autograd finds UNCONNECTED, off the UNSTACKED residual.

    A `select` off `SwaptionCalibration.__call__`'s `torch.stack` is downstream of every input, so
    the structural claim has to be taken one step earlier, off `objective.loss`'s dict. Absence is
    stronger than a zero: a zero is arithmetic that cancelled, this is no path at all.
    """
    x = theta.detach().clone().requires_grad_(True)
    errors = calibration.objective.loss(calibration.split(x))[1]
    return sum(grad is None
               for error in errors.values()
               for grad in torch.autograd.grad(error, calibration.quotes, retain_graph=True,
                                               allow_unused=True))


def test_a_second_block_does_not_rescale_the_first_blocks_residual():
    """ONE BOOTSTRAPPER, MANY BLOCKS. `bootstrap` loops over `market_prices` on a single
    `RiskNeutralInterestRateModel`, so a residual closure reading its sample shape off `self` would
    be re-scaled by whatever the NEXT curve declares. The shape is a local per block instead.

    With the mutant in place, block A at 2048 paths after a block B at 8192: every premium x0.25
    and `||J'r||` 7.8e5 -> 7.4e11. Not a stale number nobody reads - `LeastSquaresSolve` keeps
    block A's calibration alive in `ctx` for a backward that runs after the loop ends, so the
    corrupted closure is the one a quote Jacobian comes off. With `Batches` differing it is an
    IndexError out of `t_random_numbers` rather than a rescale.

    The second block is built on the FIRST world's bootstrapper, because `identified_closure`
    builds a fresh one per world. `Objective: 'Monte_Carlo'` on every block: the analytic closure
    divides by no sample count.
    """
    pair = ((1, 1, 3, 3), (5, 5, 3, 3))

    def second_block(world, **fields):
        """Another curve's block, built on THIS world's bootstrapper - `bootstrap`'s own loop."""
        return world['model'].calc_loss_on_ir_curve(
            {'instrument': dict(world['block'], **fields)}, BASE, world['time_grid'],
            world['process'], world['implied_obj'], world['ir_factor'], world['surface'])

    a = identified_closure(benchmarks=pair, batch_size=2048, Objective='Monte_Carlo')
    alone = {k: as_float(v) for k, v in a['loss'](a['implied_var'])[0].items()}
    second_block(a, Simulations=8192)
    after = {k: as_float(v) for k, v in a['loss'](a['implied_var'])[0].items()}
    for name, value in alone.items():
        assert after[name] == value, (
            '{}: a 2048-path block prices {:.17g} on its own and {:.17g} once an 8192-path block '
            'was built on the same bootstrapper - a factor of {:.6f}, which is the closure reading '
            'its sample count off `self`'.format(name, value, after[name], after[name] / value))
    # the attributes are a REPORT of the last block, and the declared-shape gate reads them as one
    assert (a['model'].batch_size, a['model'].num_batches) == (8192, 1), (
        'the mirror onto `self` has to keep tracking the last block built - '
        'test_the_declared_sample_shape_is_the_shape_the_engine_uses reads it')

    # the same seam, in the quantity `LeastSquaresSolve.backward` refuses on
    cal, quoted = calibration_at(ID_THETA, benchmarks=pair, batch_size=2048,
                                 Objective='Monte_Carlo', Quote_Sensitivity='Yes')
    theta = flat_theta(cal, ID_THETA)
    before = stationarity(cal, theta)
    second_block(quoted, Simulations=8192)
    assert stationarity(cal, theta) == before, (
        "||J'r||, ||r|| read {} at theta* and {} after a second block was built - the backward's "
        'own measure, so a quote-side Jacobian there is one of a rescaled residual'.format(
            before, stationarity(cal, theta)))

    # and the `Batches` half, which does not rescale - it indexes off the end of the sample
    c = identified_closure(benchmarks=pair, batch_size=2048, Objective='Monte_Carlo', Batches=1)
    priced = {k: as_float(v) for k, v in c['loss'](c['implied_var'])[0].items()}
    second_block(c, Batches=4)
    assert {k: as_float(v) for k, v in c['loss'](c['implied_var'])[0].items()} == priced, (
        'a Batches=1 block did not survive a Batches=4 block being built beside it')


def test_the_objective_field_declares_two_spellings_and_builds_two_things():
    """The two spellings the menu offers build two different objectives - the analytic one carries
    the Monte Carlo auditor, the Monte Carlo one does not - and hand the chain a different
    reduction: a sum on one path, a sum of SQUARES on the other. The schema-emission suite holds
    the AST half (the declaration and the `.get` fallback are one string).
    """
    field = next(f for f in HullWhite2FactorModelParameters.fields if f.name == 'Objective')
    assert field.default == 'Analytic'
    assert sorted(field.values) == ['Analytic', 'Monte_Carlo']

    mc = identified_closure(benchmarks=((2, 5, 3, 6),), batch_size=2048, Objective='Monte_Carlo')
    an = identified_closure(benchmarks=((2, 5, 3, 6),), batch_size=2048, Objective='Analytic')
    assert mc['objective'].reprice is None, 'the Monte Carlo objective audits nothing'
    assert an['objective'].reprice is not None, (
        'the Analytic objective has to carry the Monte Carlo closure as its auditor')
    # the model VALUE each returns is a premium either way, which is what keeps the solve's log
    # one thing - and the residuals are different quantities, so they must not read the same
    mc_value, mc_error = (list(x.values())[0] for x in mc['loss'](mc['implied_var']))
    an_value, an_error = (list(x.values())[0] for x in an['loss'](an['implied_var']))
    assert 1e-3 < as_float(mc_value) < 1.0 and 1e-3 < as_float(an_value) < 1.0
    assert abs(as_float(mc_error)) > 1.0 and abs(as_float(an_error)) < 1e-2, (
        'a weighted squared percentage and an absolute normal vol should not be the same size')
    # the scalar the chain compares: a sum on one path, a sum of SQUARES on the other
    probe = torch.tensor([2.0, -3.0], dtype=DTYPE)
    assert float(mc['objective'].reduce(probe)) == -1.0
    assert float(an['objective'].reduce(probe)) == 13.0
    assert float(an['objective'].reduce(probe.numpy())) == 13.0, 'numpy is the other caller'


def test_an_unknown_objective_refuses_and_names_the_two_it_knows():
    """An unknown spelling RAISES naming both, rather than falling through to whichever branch the
    `if` ends on."""
    for spelling in ('analytic', 'SchragerPelsser', 'MonteCarlo', ''):
        with pytest.raises(Exception, match='Monte_Carlo'):
            identified_closure(benchmarks=((1, 1, 3, 3),), batch_size=1024, Objective=spelling)
    try:
        identified_closure(benchmarks=((1, 1, 3, 3),), batch_size=1024, Objective='analytic')
    except Exception as e:
        assert 'Analytic' in str(e), 'the refusal has to name both spellings: {}'.format(e)


def test_the_analytic_quote_side_is_built_and_the_splice_is_worth_exactly_zero():
    """The analytic quote side is BUILT, and the first thing it owes is that it changed nothing:
    residual, model value, market normal vol and the backward's own `(||J'r||, ||r||)` are all
    BIT-IDENTICAL with the quote side on and off, with the leaves present and connected.

    Bit-identity is the splice's own claim - `base + (carried - detach(carried))` is `base + 0.0`
    for every finite value - so this is hex comparison rather than a tolerance. Asserted as a
    MEASUREMENT and not as an absence of exceptions: a `market_normal_vol` that stopped carrying a
    splice would run to completion and report zeros, which is what the old refusal named.
    """
    on, _ = quote_calibration(ID_ANALYTIC_THETA)
    off = identified_closure(benchmarks=ID_GRID, theta=ID_ANALYTIC_THETA, Objective='Analytic')
    theta = flat_theta(on, ID_ANALYTIC_THETA)

    assert len(on.quotes) == len(ID_GRID) and all(q.requires_grad for q in on.quotes)
    prices_on, errors_on = on.objective.loss(on.implied_var)
    prices_off, errors_off = off['loss'](off['implied_var'])
    for name in errors_off:
        assert float(errors_on[name].detach()).hex() == float(errors_off[name].detach()).hex(), (
            '{}: the analytic residual moved when the quote side was switched on'.format(name))
        assert float(prices_on[name].detach()).hex() == float(prices_off[name].detach()).hex(), name
        sp = on.process.schrager_pelsser_swaption(
            *[getattr(on.market_swaps[name].schedule, f) for f in ('expiry', 'pay_times', 'accruals')])
        market = [float(swaps[name].market_normal_vol(sp.annuity).detach()).hex()
                  for swaps in (on.market_swaps, off['swaps'])]
        assert market[0] == market[1], '{}: the market normal vol moved'.format(name)
    # and the quantity the backward refuses on, which is what a quote Jacobian is taken through
    assert stationarity(on, theta) == stationarity(
        SwaptionCalibration('gate', off['objective'], off['implied_var'], None, off['process'],
                            off['swaps']), theta), (
        "||J'r|| and ||r|| moved with the quote side on - the recorded readings are 8.63e-7 and "
        '1.48e-3 and this gate is what holds them still')
    # the schema stopped saying it is not built, which is half of what a user reads
    for field in ('Objective', 'Quote_Sensitivity'):
        text = next(f for f in HullWhite2FactorModelParameters.fields
                    if f.name == field).description
        assert 'not built on the analytic' not in text and 'still to build' not in text, field


def test_the_market_normal_vol_is_a_division_and_it_round_trips():
    """At the money `sigma_N = P sqrt(2pi/T0) / A` is a DIVISION - no brentq inside an optimizer
    evaluation. Inverting SP's own premium returns SP's own vol to 1e-15, and `normal_vol_error` is
    the premium residual rescaled by exactly `weight sqrt(2pi/T0) / A` to 1e-12 - which makes
    vols-against-vols and premium-against-premium the same zero, and is why no quoting convention
    can be inherited by half. The annuity is SP's own, held to the pvbp the premium was struck on.
    """
    world = identified_closure(benchmarks=CHECKER_BENCHMARKS, Objective='Analytic', batch_size=2048)
    world['loss'](world['implied_var'])
    for name, swap in world['swaps'].items():
        sp = world['process'].schrager_pelsser_swaption(
            swap.schedule.expiry, swap.schedule.pay_times, swap.schedule.accruals)
        got = as_float(sp.premium * np.sqrt(2.0 * np.pi / swap.schedule.expiry) / sp.annuity)
        assert abs(got / as_float(sp.normal_vol) - 1.0) < 1e-15, (
            '{}: inverting SP\'s own premium reads {:.17g} against its own vol {:.17g}'.format(
                name, got, as_float(sp.normal_vol)))
        scale = float(swap.weight) * np.sqrt(2.0 * np.pi / swap.schedule.expiry) / as_float(
            sp.annuity)
        assert abs(as_float(swap.normal_vol_error(sp)) /
                   (scale * (as_float(sp.premium) - swap.price)) - 1.0) < 1e-12, name
        # something is being measured: a market vol of zero would pass both lines above
        assert 1e-2 < as_float(swap.market_normal_vol(sp.annuity)) < 5e-2, name
    # and the annuity is SP's OWN pvbp, which is why the rescaling above is exact
    swap = world['swaps']['Swaption_2Y_5Y']
    sp = world['process'].schrager_pelsser_swaption(
        swap.schedule.expiry, swap.schedule.pay_times, swap.schedule.accruals)
    assert abs(as_float(sp.annuity) / (swap.price / utils.black_european_option_price(
        as_float(sp.swap_rate), as_float(sp.swap_rate), 0.0, 0.20, swap.schedule.expiry,
        1.0, 1.0)) - 1.0) < 1e-3, 'SP\'s annuity is not the pvbp the market premium was struck on'


def test_a_displaced_surface_reaches_the_bachelier_side_through_the_premium_only():
    """A shifted-lognormal surface strikes its Black premium at `K + shift`, so the shift reaches
    the analytic residual through `swap.price` and NOWHERE else - a normal vol has nothing to
    displace. Held both ways: the market normal vol moves with the shift, and is exactly the
    shifted premium's own inversion on an annuity that did not move at all.
    """
    reading = {}
    for shift in (0.0, 2.0):
        factors = identified_world()
        # the displacement rides on `Property_Aliases`, which is where `create_market_swaps` reads
        # it - not on the `Shift` field, which is the quote's own and never reaches the strike
        factors['InterestYieldVol.' + ID_VOL]['Property_Aliases'] = (
            [{'BlackScholesDisplacedShiftValue': shift}] if shift else None)
        world = identified_closure(benchmarks=CHECKER_BENCHMARKS, world=factors,
                                   Objective='Analytic', batch_size=2048)
        assert world['surface'].BlackScholesDisplacedShiftValue == shift
        world['loss'](world['implied_var'])
        reading[shift] = {}
        for name, swap in world['swaps'].items():
            sp = world['process'].schrager_pelsser_swaption(
                swap.schedule.expiry, swap.schedule.pay_times, swap.schedule.accruals)
            market = as_float(swap.market_normal_vol(sp.annuity))
            assert abs(market / (swap.price * np.sqrt(2.0 * np.pi / swap.schedule.expiry) /
                                 as_float(sp.annuity)) - 1.0) < 1e-15, name
            reading[shift][name] = (market, swap.price, as_float(sp.annuity))
    for name in reading[0.0]:
        flat, shifted = reading[0.0][name], reading[2.0][name]
        assert flat[2] == shifted[2], '{}: the shift moved the ANNUITY, which it cannot'.format(name)
        assert abs(shifted[0] / flat[0] - 1.0) > 1e-3, (
            '{}: a 2% displacement moved the market normal vol by {:.3e} relative - if that is now '
            'zero the shift has stopped reaching the residual at all'.format(
                name, shifted[0] / flat[0] - 1.0))


def test_the_analytic_objective_reaches_a_stationarity_the_quartic_cannot():
    """THE CLAIM THE BUILD WAS FOR, on the identified 25-quote fixture:

        objective        ||J'r|| at theta*     the Stationarity_Tol that accepts it
        Monte Carlo             3.16e+02       1e5   (this file has to declare it)
        Analytic                8.63e-07       1e-3  (the field's own DEFAULT)

    `market_swap_class.error` returns a residual that is ALREADY a square, so `least_squares`
    minimises a QUARTIC and stops where a quartic goes flat - 8.5 of the 11 orders from the seed's
    2.86e11 and no further, which is why `ID_STATIONARITY` is 1e5. `normal_vol_error` returns the
    difference itself, and falls 4.8 orders from the seed's 4.96e-2 to land 8.6e-7 short of zero.

    The raw norms are in different units - a weighted squared PERCENTAGE against an absolute normal
    VOL - so the 8.6 orders between them is not claimed as a ratio. What is like-for-like is the
    tolerance each needs, and that is the reading the default flip was taken on.
    """
    field = next(f for f in HullWhite2FactorModelParameters.fields if f.name == 'Stationarity_Tol')
    assert field.default == 1e-3 and ID_STATIONARITY == 1e5, (field.default, ID_STATIONARITY)

    for tag, extra, theta, bound, resid in (
            ('Monte_Carlo', {'Objective': 'Monte_Carlo'}, ID_THETA, (1e2, 1e5), (15.0, 30.0)),
            ('Analytic', {'Objective': 'Analytic'}, ID_ANALYTIC_THETA, (0.0, 1e-3), (1e-3, 2e-3))):
        calibration, _ = calibration_at(theta, **extra)
        norm, residual = stationarity(calibration, flat_theta(calibration, theta))
        assert bound[0] <= norm < bound[1], (
            '{} at its own theta* reads ||J\'r|| {:.4g} against a recorded band {} - the whole '
            'point of the plain residual is which side of Stationarity_Tol\'s 1e-3 default it '
            'lands on'.format(tag, norm, bound))
        assert resid[0] < residual < resid[1], '{}: ||r|| is {:.4g}'.format(tag, residual)

    # the seed, so the descent has two ends and the stopping point is not read alone
    calibration, world = calibration_at({}, Objective='Analytic')
    seed = torch.cat([v.detach().clone() for v in world['implied_var'].values()]).double()
    norm, _ = stationarity(calibration, seed)
    assert 1e-3 < norm < 1.0, (
        'the analytic gradient at the seed reads {:.4g} against a recorded 6.0e-2 - if it is now '
        'tiny the seed is already stationary and the solve below measures nothing'.format(norm))


#: The retired `ID_ANALYTIC_THETA`, kept because the corner gate below is worthless without the
#: vector whose corner it is: `Sigma_1[5]` sits EXACTLY on the 1e-5 variance floor and `Sigma_1[4]`
#: 7.0e-7 under the 0.09 ceiling, both with an outward gradient - two active bounds, one of them a
#: factor switched off.
RETIRED_ANALYTIC_THETA = {
    'Alpha_1': [0.019636043320534653],
    'Alpha_2': [0.052303947299117401],
    'Correlation': [-0.94086243266563796],
    'Sigma_1': [0.0093430763366823093, 0.0078472524308586793, 0.010534180253796762,
                0.024972435422619753, 0.089999295056638492, 1.0000000000000003e-05,
                0.043491067322455008, 0.082356319833777003, 0.034608001948002336,
                0.028132036904066834],
    'Sigma_2': [0.0063492865790124309, 0.0041844033445664437, 0.0088736732224940721,
                0.0089213613978900604, 0.074641807429185525, 0.02053997401379614,
                0.063694656668906141, 0.074318374104084511, 0.026604763684314203,
                0.021751976358139601]}


def active_bounds(calibration, named):
    """`(||J'r||, ||projected J'r||, [(coordinate, value, side, gradient)])` at a named theta.

    ACTIVE is the KKT test and not a proximity one: within 1e-5 of the box width of an end AND the
    gradient pointing OUT. The PROJECTED norm zeroes those components, and is what says whether
    theta* is a constrained stationary point where the raw norm cannot.
    """
    theta = flat_theta(calibration, named)
    x = theta.detach().clone().requires_grad_(True)
    residual = calibration(x)
    jacobian = torch.stack([torch.autograd.grad(residual[i], x, retain_graph=True)[0]
                            for i in range(residual.numel())]).double()
    gradient = (jacobian.t() @ residual.detach().double()).numpy()
    model = HullWhite2FactorModelParameters({}, DEVICE, DTYPE)
    free, active, k = gradient.copy(), [], 0
    for key in calibration.keys:
        low, high = (model.alpha_bounds if key.startswith('Alpha') else
                     model.corr_bounds if key == 'Correlation' else model.sigma_bounds)
        for i, value in enumerate(np.atleast_1d(named[key])):
            name = '{}[{}]'.format(key, i) if np.atleast_1d(named[key]).size > 1 else key
            width = high - low
            if value - low < 1e-5 * width and gradient[k] > 0.0:
                active.append((name, float(value), 'floor', float(gradient[k])))
                free[k] = 0.0
            elif high - value < 1e-5 * width and gradient[k] < 0.0:
                active.append((name, float(value), 'ceiling', float(gradient[k])))
                free[k] = 0.0
            k += 1
    return (float(np.linalg.norm(gradient)), float(np.linalg.norm(free)), active)


def correlation_profile(calibration, named, grid=(-0.95, -0.94, -0.9, -0.8, -0.5, 0.0, 0.95)):
    """`{rho: half the sum of squares}` along the correlation axis, everything else held at `named`
    - the only thing that can say whether a correlation on its bound is the fixture's minimum or
    where a search stopped.
    """
    theta = flat_theta(calibration, named)
    index = sum(np.atleast_1d(named[k]).size
                for k in calibration.keys[:calibration.keys.index('Correlation')])
    out = {}
    for rho in grid:
        x = theta.detach().clone()
        x[index] = rho
        out[rho] = 0.5 * float((calibration(x).detach().double() ** 2).sum())
    return out


def test_the_corner_moved_off_the_sigma_floor_and_what_is_left_is_the_fixtures_own_minimum():
    """A BOUND-DOMINATED `||J'r||` IS TWO DIFFERENT THINGS and the raw norm cannot tell them apart:
    at a box-constrained minimum the gradient is the sum of the multipliers on the active bounds.
    So stationarity is the PROJECTED gradient and pathology is WHICH bound is active.

        theta*                      ||J'r||    ||projected||   active bounds
        RETIRED_ANALYTIC_THETA      5.75e-5      4.47e-5       Sigma_1[5] on the 1e-5 FLOOR,
                                                               Sigma_1[4] on the 0.09 CEILING
        ID_ANALYTIC_THETA           8.63e-7      2.09e-7       Correlation on the -0.95 end

    A factor 214 in the measure that means constrained stationarity, and the surviving multiplier
    is a different KIND of coordinate: a sigma knot pinned at 1e-5 is a FACTOR SWITCHED OFF, where
    a correlation on its end is the quotes wanting the strongest the authored box allows.

    AND IT IS THE FIXTURE'S OWN MINIMUM, asserted rather than argued: the objective's profile along
    that axis through theta* is monotone from -0.95 outward and minimised AT the bound, the same
    shape at `ID_THETA`. The retired vector is the counter-example - minimised at its own -0.9409
    and rising toward -0.95, which is what a different basin looks like from inside.

    A six-seed sweep (measured, too expensive to assert) puts five seeds on the same rms to four
    digits and the same end; the sixth reaches the other basin at 4.4659bp with NINE knots on the
    floor, so the corner is alive and reachable rather than removed.
    """
    calibration, _ = quote_calibration(ID_ANALYTIC_THETA)
    norm, projected, active = active_bounds(calibration, ID_ANALYTIC_THETA)

    # the corner signature: not one sigma knot pinned at either end of its box
    model = HullWhite2FactorModelParameters({}, DEVICE, DTYPE)
    low, high = model.sigma_bounds
    for name in ('Sigma_1', 'Sigma_2'):
        knots = np.asarray(ID_ANALYTIC_THETA[name], dtype=float)
        assert (knots - low > 1e-5 * (high - low)).all(), (
            '{} has a knot on the {:g} variance floor - {} - which is a FACTOR SWITCHED OFF and is '
            'the corner the 2026-09-02 landing removed'.format(
                name, low, [(i, v) for i, v in enumerate(knots) if v - low <= 1e-5 * (high - low)]))
        assert (high - knots > 1e-5 * (high - low)).all(), (
            '{} has a knot on the {:g} ceiling - {} - which the retired vector also had and this '
            'one does not'.format(
                name, high, [(i, v) for i, v in enumerate(knots) if high - v <= 1e-5 * (high - low)]))

    # ONE active bound, and it is the correlation
    assert [a[0] for a in active] == ['Correlation'] and active[0][2] == 'floor', (
        'the active set at theta* is {} against the recorded single Correlation multiplier - if a '
        'sigma knot is back on a bound the corner has returned'.format(active))
    assert 0.9 < norm / abs(active[0][3]) < 1.1, (
        'the correlation multiplier is {:.4g} of a ||J\'r|| of {:.4g}, against the recorded 97% - '
        'if the rest of the gradient has grown this is no longer a constrained minimum with one '
        'active bound'.format(abs(active[0][3]) / norm, norm))
    # and the PROJECTED gradient, which is the reading that means stationarity
    assert projected < 1e-6, (
        'the projected gradient at theta* is {:.4g} against a recorded 2.09e-7 - the raw ||J\'r|| '
        'is dominated by the correlation multiplier by construction, so THIS is the number that '
        'says the free coordinates are stationary'.format(projected))

    # the retired vector, so none of the above is vacuous
    retired_norm, retired_projected, retired_active = active_bounds(
        calibration, RETIRED_ANALYTIC_THETA)
    assert sorted(a[2] for a in retired_active) == ['ceiling', 'floor'], (
        'the retired vector no longer carries the two sigma multipliers this gate contrasts '
        'against: {}'.format(retired_active))
    assert retired_projected > 100.0 * projected, (
        'the retired vector\'s projected gradient is {:.4g} against theta*\'s {:.4g} - the recorded '
        'readings are 4.47e-5 and 2.09e-7, a factor 214'.format(retired_projected, projected))

    # the correlation end is the fixture's own minimum, on both solved points
    for tag, named in (('ID_ANALYTIC_THETA', ID_ANALYTIC_THETA), ('ID_THETA', ID_THETA)):
        profile = correlation_profile(calibration, named)
        rhos = sorted(profile)
        assert rhos[0] == model.corr_bounds[0] and min(profile, key=profile.get) == rhos[0], (
            '{}: the correlation profile through theta* is minimised at {:+.4g} rather than at the '
            '{:g} end - if that is real the bound has stopped being where this fixture wants to be '
            'and the corner reading above needs re-taking'.format(
                tag, min(profile, key=profile.get), model.corr_bounds[0]))
        assert all(profile[b] > profile[a] for a, b in zip(rhos, rhos[1:])), (
            '{}: the profile is no longer monotone away from the bound: {}'.format(
                tag, {k: '{:.4g}'.format(v) for k, v in profile.items()}))
    # and the retired vector is the counter-example that makes the shape a reading
    retired_profile = correlation_profile(calibration, RETIRED_ANALYTIC_THETA)
    assert (min(retired_profile, key=retired_profile.get) != model.corr_bounds[0]), (
        'the retired vector now wants the correlation end too, so the contrast this gate draws '
        'between the two basins has gone: {}'.format(
            {k: '{:.4g}'.format(v) for k, v in retired_profile.items()}))


def honesty_errors(named, benchmarks=ID_GRID):
    """`{benchmark: relative premium residual}` from the engine's OWN Monte Carlo estimator - the
    whole vector `honesty_reprice` takes its maximum of, off the same closure and frozen sample.
    """
    world = identified_closure(benchmarks=benchmarks, theta=named, Objective='Analytic')
    calibration = SwaptionCalibration('gate', world['objective'], world['implied_var'], None,
                                      world['process'], world['swaps'])
    prices, _ = calibration.objective.reprice(calibration.implied_var)
    return calibration, {name: float(value.detach()) / calibration.market_swaps[name].price - 1.0
                         for name, value in prices.items()}


def test_the_honesty_reprice_is_one_order_statistic_of_a_distribution_that_improved():
    """`honesty_reprice` reports the WORST benchmark by construction - one order statistic of 25 -
    and on this fixture that number ROSE while every other reading of the same distribution fell.
    Both vectors audited through the same estimator, sample and clock:

        reading                              retired theta*     ID_ANALYTIC_THETA
        worst benchmark                    -4.669% (1Y x 5Y)   -6.249% (5Y x 2Y)
        rms over the 25                        2.709%              2.387%
        benchmarks outside 3%                   10 of 25            3 of 25

    The distribution tightened around a mean that moved DOWN rather than toward zero: the
    simulation's numeraire bias on this 1Y-first-node curve is negative almost everywhere and no
    theta* removes it. The max is also ANTI-correlated with the fit here - over the six-seed sweep
    the worst fit reads the best honesty - so "lower rms" and "honesty no worse" point opposite
    ways and which one this is judged on is an open decision. This gate asserts the distribution,
    which is the half that is not ambiguous.
    """
    calibration, now = honesty_errors(ID_ANALYTIC_THETA)
    _, retired = honesty_errors(RETIRED_ANALYTIC_THETA)
    worst = max(now, key=lambda name: abs(now[name]))

    # the library's own instrument agrees with the vector this gate takes apart
    reported = calibration.honesty_reprice(flat_theta(calibration, ID_ANALYTIC_THETA))
    assert reported[0] == worst and abs(reported[1] / now[worst] - 1.0) < 1e-9, (
        'honesty_reprice reports {} against the {} this gate reads off the same closure'.format(
            reported, (worst, now[worst])))
    assert worst == 'Swaption_5Y_2Y' and abs(now[worst] + 0.062493) < 1e-4, (
        'the worst benchmark is {} at {:+.4%} against the recorded Swaption_5Y_2Y at '
        '-6.2493%'.format(worst, now[worst]))

    # the distribution, on every reading but its own extreme
    rms = {tag: float(np.sqrt(np.mean([v * v for v in errors.values()])))
           for tag, errors in (('now', now), ('retired', retired))}
    outside = {tag: sum(abs(v) > 0.03 for v in errors.values())
               for tag, errors in (('now', now), ('retired', retired))}
    assert rms['now'] < rms['retired'], (
        'the audit\'s rms over the 25 benchmarks is {:.4%} against the retired vector\'s {:.4%} - '
        'the recorded readings are 2.387% and 2.709%, and this is the reading that says the whole '
        'distribution improved where its maximum did not'.format(rms['now'], rms['retired']))
    assert outside['now'] <= 4 < outside['retired'], (
        '{} of 25 benchmarks are more than 3% from the auditor against the retired vector\'s {} - '
        'the recorded readings are 3 and 10'.format(outside['now'], outside['retired']))
    # and the max, which is the acceptance criterion the seed change did NOT meet, recorded as such
    assert abs(now[worst]) > abs(retired[max(retired, key=lambda n: abs(retired[n]))]), (
        'the worst benchmark is no longer worse than the retired vector\'s - if the max has come '
        'back under -4.669% the ruling this gate records has been overtaken and the docstring '
        'needs re-taking')


def repriced_vols(theta, benchmarks):
    """`{benchmark: (SP normal vol, the market's own inverted premium)}` in bp, at `theta`. Both
    sides through the ANALYTIC closure on one annuity, so the difference is the premium residual
    rescaled rather than two conventions differenced.
    """
    world = identified_closure(benchmarks=benchmarks, theta=theta, Objective='Analytic')
    world['loss'](world['implied_var'])
    out = {}
    for name, swap in world['swaps'].items():
        sp = world['process'].schrager_pelsser_swaption(
            swap.schedule.expiry, swap.schedule.pay_times, swap.schedule.accruals)
        out[name] = (as_float(sp.normal_vol) * 1e4,
                     as_float(swap.market_normal_vol(sp.annuity)) * 1e4)
    return out


def test_the_two_answers_agree_in_vol_space_and_the_theta_space_half_is_the_fixtures():
    """THE THETA-COMPARISON, on BOTH fixtures, and it comes out as two findings.

    IN REPRICED NORMAL-VOL SPACE THEY AGREE. On the identified 25-quote grid, level 175 to 208 bp:
    rms |SP - market| is 4.54bp at theta*_MC and 2.96bp at theta*_An, and the two answers price the
    whole cube within 6.0bp of each other. So the analytic solve fits the grid 35% better in the
    metric it is fitting, and each objective is measured on the fixture it is solved on.

    IN THETA SPACE the answer is the FIXTURE'S rather than the objective's, which is rank
    deficiency ([Quote Sensitivities](quote_sensitivities.md#rank-deficiency)) and not a
    disagreement. On the 25-quote block both objectives solve the correlation to the -0.95 end,
    0.0088 apart, and the whole vector agrees to 0.059. On the FOUR-quote block - 4 quotes against
    23 parameters, a 19-dimensional null space - they are 0.603 apart in the correlation. Both
    Jacobians are rank deficient: the analytic `J` runs `sigma_min/sigma_max` 5.1e-15 and the
    declared 1e-8 cutoff keeps 15 of 23 directions, the Monte Carlo one 1.76e-6 and 17.

    Which is why the four-quote arm's 4.16bp rms is asserted as a CROSS-METRIC reading and not as a
    fit: both chains interpolate there (`||r||` 1.01e-8 and 2.29e-8), so what it measures is how
    far apart two ESTIMATORS are - SP's freezing bias plus the simulation's numeraire error, adding
    at the 10Y x 10Y corner to 7.82bp. The fit itself is the `||r||` pair, held at the end.
    """
    for tag, benchmarks, mc_theta, an_theta, bound in (
            ('25-quote', ID_GRID, ID_THETA, ID_ANALYTIC_THETA,
             dict(rms_mc=(3.0, 8.0), rms_an=(2.0, 4.0), apart=(5.0, 8.0), theta=0.1,
                  converged=True)),
            ('4-quote', CHECKER_BENCHMARKS, MC_FOUR_THETA, AN_FOUR_THETA,
             dict(rms_mc=(2.0, 6.0), rms_an=(0.0, 0.05), apart=(6.0, 9.0), theta=0.3,
                  converged=False))):
        reading = {'mc': repriced_vols(mc_theta, benchmarks),
                   'analytic': repriced_vols(an_theta, benchmarks)}
        gaps = {k: np.array([model - market for model, market in r.values()])
                for k, r in reading.items()}
        apart = np.array([reading['mc'][n][0] - reading['analytic'][n][0] for n in reading['mc']])
        rms = {k: float(np.sqrt((g * g).mean())) for k, g in gaps.items()}
        level = [market for _, market in reading['mc'].values()]

        assert 170.0 < min(level) and max(level) < 215.0, (
            '{}: the level has moved off the recorded band'.format(tag))
        assert rms['analytic'] < rms['mc'], (
            '{}: the analytic solve fits the vols it is fitting no better than the Monte Carlo one '
            'does: {:.3f}bp against {:.3f}'.format(tag, rms['analytic'], rms['mc']))
        assert bound['rms_mc'][0] < rms['mc'] < bound['rms_mc'][1], (tag, rms)
        assert bound['rms_an'][0] <= rms['analytic'] < bound['rms_an'][1], (tag, rms)
        assert (float(np.sqrt((apart * apart).mean())) < bound['apart'][0]
                and float(np.abs(apart).max()) < bound['apart'][1]), (
            '{}: the two answers price the cube {:.3f}bp apart rms, worst {:.3f}'.format(
                tag, float(np.sqrt((apart * apart).mean())), float(np.abs(apart).max())))
        # the theta-space half, which is the finding rather than the agreement
        gap = abs(mc_theta['Correlation'][0] - an_theta['Correlation'][0])
        if bound['converged']:
            assert gap < bound['theta'], (
                '{}: the two correlations are {:.4g} apart, where the 2026-09-02 re-mark left them '
                'both on the -0.95 end and 0.0088 apart - if they have separated again the '
                'correlation has come off the bound and the docstring needs re-taking'.format(
                    tag, gap))
            for name in mc_theta:
                assert np.abs(np.atleast_1d(mc_theta[name]) -
                              np.atleast_1d(an_theta[name])).max() < 0.1, (
                    '{}: the two answers have come apart in {} - this fixture was identified '
                    'enough that they agreed to 0.059 across the whole vector'.format(tag, name))
        else:
            assert gap > bound['theta'], (
                '{}: the two correlations have converged to {:.4g} - if that is real this fixture '
                'has become identified and the null-space reading above needs '
                're-taking'.format(tag, gap))

    # the four-quote fit in the metric each objective actually minimises - the half the vol-space
    # column cannot see, and the only thing here that says the fit is a fit
    for objective, named, before, now in (('Monte_Carlo', MC_FOUR_THETA, 4.4e-8, 1.0137e-08),
                                          ('Analytic', AN_FOUR_THETA, 4.0e-7, 2.2874e-08)):
        cal, _ = calibration_at(named, benchmarks=CHECKER_BENCHMARKS, Objective=objective)
        norm = stationarity(cal, flat_theta(cal, named))[1]
        assert abs(norm / now - 1.0) < 1e-3, (
            '{}: ||r|| at the recorded four-quote theta* reads {:.4e} against {:.4e} - this is the '
            'metric that objective minimises and the one the landing improved (it read {:.1e} '
            'before the 2026-09-02 re-mark)'.format(objective, norm, now, before))
        assert norm < before, (
            '{}: ||r|| at theta* is {:.4e}, no better than the {:.1e} it read before the re-mark - '
            'the vol-space column above is a cross-metric reading and THIS is the fit'.format(
                objective, norm, before))


def bootstrap_the_block(**extra):
    """One `RiskNeutralInterestRateModel.bootstrap` on the identified fixture, run whole - the
    entry point a job drives, because the honesty reprice and the parameters it writes only exist
    out here. `Objective` is left out of the block; the callers name the path they mean.
    """
    factors = identified_world()
    block = {'Swaption_Volatility': ID_VOL, 'Generate_Instruments': 'No', 'Random_Seed': ID_SEED,
             'Stationarity_Tol': ID_STATIONARITY, 'Quote_Sensitivity': 'No',
             'Simulations': 2048, 'Batches': 1,
             'Instrument_Definitions': identified_definitions(CHECKER_BENCHMARKS)}
    block.update(extra)
    block = {k: v for k, v in block.items() if v is not ABSENT}
    model = HullWhite2FactorModelParameters({}, DEVICE, DTYPE)
    model.bootstrap({'Base_Date': BASE, 'Base_Currency': ID_CCY}, {}, factors, ModelParams(),
                    {ID_BLOCK: {'instrument': block}}, {})
    return model, factors


def test_the_analytic_solve_reports_what_the_engines_own_estimator_makes_of_it(caplog):
    """An analytic solve fits frozen-annuity vols, so `honesty_reprice` runs one pass of the
    engine's own Monte Carlo at theta* and `bootstrap` LOGS the worst benchmark by name - reported
    CAPPED, not checked against a tolerance it did not reach.

    The number is mostly the SIMULATION'S, which is why it is reported and not asserted tightly: on
    the four-quote block it names 10Y x 10Y at -1.67%, against the -1.61% numeraire error
    `test_the_monte_carlo_carries_a_bias_of_its_own` measures at that benchmark. It also moves with
    the block's own path count - -2.29% at the 2048 this gate declares, -1.67% at the default 8192.
    The Monte Carlo objective logs nothing: it IS the estimator.
    """
    with caplog.at_level(logging.INFO, logger=''):
        bootstrap_the_block(Objective='Analytic')
    lines = [r.getMessage() for r in caplog.records if 'Analytic objective' in r.getMessage()]
    assert len(lines) == 1, 'the analytic solve has to report itself exactly once: {}'.format(lines)
    named = [n for n in identified_definitions(CHECKER_BENCHMARKS)
             if 'Swaption_{}Y_{}Y'.format(n['Start'].years, n['Tenor'].years) in lines[0]]
    assert named, 'the reprice has to name a benchmark: {}'.format(lines[0])
    percent = float(lines[0].rsplit(',', 1)[1].split('%')[0])
    assert 0.05 < abs(percent) < 25.0, (
        'the reprice reads {:+.2f}% - against the -0.35% to -1.61% this fixture\'s own numeraire '
        'error carries, a number that small means it is not repricing and one that large means the '
        'analytic answer has stopped being an answer'.format(percent))

    caplog.clear()
    with caplog.at_level(logging.INFO, logger=''):
        bootstrap_the_block(Objective='Monte_Carlo')
    assert not [r for r in caplog.records if 'Analytic objective' in r.getMessage()], (
        'the Monte Carlo objective audited itself - it IS the estimator, so there is nothing to say')


def test_the_analytic_solve_is_deterministic_and_the_seed_moves_what_the_quotes_do_not():
    """DETERMINISM, and the seed spread beside it. The analytic objective draws no sample, so the
    only randomness is basin hopping's own seeded search: two runs at one seed agree TO THE BIT,
    and the answer is `AN_FOUR_THETA` and not the seed.

    THE SEED SPREAD IS LARGER THAN THE MONTE CARLO PATH'S here, which is the opposite of what was
    expected and is not evidence for the objective. Across seeds 5120 / 7 / 99 on the four-quote
    block the Monte Carlo chain returns a bit-identical theta* every time while the analytic one
    spreads 0.643 in `Alpha_1`. Four quotes against 23 parameters leaves theta* a MANIFOLD both
    objectives interpolate exactly (`||r||` 1.01e-8 and 2.29e-8), so what differs is which point of
    it the search reaches: the analytic evaluation is 13x cheaper, the chain makes 3x as many, and
    it actually explores. The evidence for the objective is the stationarity gate.

    Wall clock, CPU float64: 75.3 s against 16.4 s on the four-quote chain, a factor 4.6 bought as
    13x per evaluation against 3.0x the evaluations. The 25-quote block costs 204 s over 849
    evaluations - 0.240 s each against the four-quote block's 0.041, because SP is one scalar call
    per benchmark. Per evaluation on the CUDA float32 a job runs, SP is still the slower of the two
    (0.158 s against 0.140 s); batching it across the benchmark set is the open build.
    """
    first, second = (identified_calibration(Objective='Analytic')[0].solve() for _ in range(2))
    assert [float(v).hex() for v in first.numpy()] == [float(v).hex() for v in second.numpy()], (
        'two analytic solves at one seed disagree - the objective draws no sample, so the only '
        'thing left that could move is the search, and it is seeded')
    # and the answer is not the seed, or determinism would be free
    calibration, world = identified_calibration(Objective='Analytic')
    seed = torch.cat([v.detach().clone() for v in world['implied_var'].values()]).double()
    assert float((first.double() - seed).abs().max()) > 1e-3, 'the chain returned its own seed'
    # and it is the vector the theta-comparison reads, so that constant is a SOLVE OUTPUT
    got = calibration.unflatten(first)
    for name, recorded in AN_FOUR_THETA.items():
        assert [float(v).hex() for v in np.atleast_1d(got[name])] == [
            float(v).hex() for v in recorded], (
            '{} solved to {} against the recorded {}'.format(name, list(got[name]), recorded))
    # the fit is essentially exact, which is what makes theta* a manifold rather than a point
    residual = stationarity(calibration, first)[1]
    assert residual < 1e-5, (
        'the four-quote analytic fit reads ||r|| {:.3e} against a recorded 2.29e-8 - it is 4 quotes '
        'against 23 parameters, so it interpolates'.format(residual))


def test_the_monte_carlo_objective_still_solves_to_this_vector():
    """THE BIT-IDENTITY BASELINE: the whole chain on the four-quote fixture returns
    `MC_FOUR_THETA`'s 23 doubles to the bit - same seed, same frozen Sobol sample, same acceptance
    rule. It pays 70 seconds because nothing cheaper is the claim, and it is why the batch repair
    had to be a no-op at `Batches: 1`.

    If a scipy or torch upgrade moves the last bits, RE-RECORD after checking that the residual at
    a fixed theta has not: the residual is arithmetic and cannot drift, a stopping point can.
    """
    calibration, world = identified_calibration(Objective='Monte_Carlo')
    assert world['block']['Objective'] == 'Monte_Carlo', (
        'this vector is the SIMULATION\'s and the block has to say so - on the family default it '
        'would be solved by the analytic chain and would not be this vector at all')
    theta = calibration.solve()
    got = calibration.unflatten(theta)
    for name, recorded in MC_FOUR_THETA.items():
        assert [float(v).hex() for v in np.atleast_1d(got[name])] == [
            float(v).hex() for v in recorded], (
            '{} solved to {} against the recorded {}'.format(name, list(got[name]), recorded))


def test_the_absent_objective_is_the_declared_analytic_one():
    """WHAT THE DEFAULT SOLVES TO: the same 23 doubles as a block spelling `Objective: 'Analytic'`,
    `array_equal`, off two runs of the whole optimizer chain - and `AN_FOUR_THETA` on the ABSENT
    run, which is what makes this the default's gate rather than a self-comparison.

    The gates upstream hold the fallback at the schema and at the residual; none of them runs a
    SOLVE, and between the fallback and the number written into `Price Factors` sit two optimizer
    stages, a seeded search, an acceptance test and a stopping rule. This pays ~13 s a chain to
    close that gap.
    """
    absent, absent_world = identified_calibration()
    spelled, world = identified_calibration(Objective='Analytic')
    assert 'Objective' not in absent_world['block'], 'the first block has to OMIT the field'
    assert world['block']['Objective'] == 'Analytic'
    assert absent.objective.reprice is not None, (
        'the absent field did not build the analytic objective, so this gate is comparing the '
        'pre-flip default against Analytic and would be a finding rather than a pass')
    solved = absent.solve()
    got, want = solved.detach().numpy(), spelled.solve().detach().numpy()
    assert np.array_equal(got, want), (
        'a block OMITTING Objective solved to a different theta* than one declaring Analytic - '
        'worst entry {:.3e} absolute, over {} of 23 coordinates'.format(
            float(np.max(np.abs(got - want))), int((got != want).sum())))
    recorded = absent.unflatten(solved)
    for name, vector in AN_FOUR_THETA.items():
        assert [float(v).hex() for v in np.atleast_1d(recorded[name])] == [
            float(v).hex() for v in vector], (
            '{} solved to {} on the absent field against the recorded analytic {}'.format(
                name, list(recorded[name]), vector))


def test_the_two_spellings_of_the_default_drive_the_adapters_identically():
    """`Objective` absent and `Objective: 'Analytic'` are one job at the three SEAMS the chain uses
    - the residual handed to the implicit-function wrapper, the scalar-and-gradient pair basin
    hopping reads, the residual-and-Jacobian pair `least_squares` reads - at three parameter
    vectors, all bitwise. Comparing only the residual would miss a `reduce` that differed, and
    `reduce` is the seam the default flip put behind the absent field.
    """
    a_cal, a_world = identified_calibration(batch_size=2048, Objective='Analytic')
    b_cal, b_world = identified_calibration(batch_size=2048)
    assert a_world['block']['Objective'] == 'Analytic'
    assert 'Objective' not in b_world['block'], 'the second block has to OMIT the field'
    # both closures carry the auditor, which is the analytic path's own signature
    assert a_cal.objective.reprice is not None and b_cal.objective.reprice is not None
    x0 = a_cal.optimizers[0][1]
    for step in (0.0, 0.01, -0.005):
        x = x0 * (1.0 + step)
        a_basin, a_grad = a_cal.optimizers[0][2](x)
        b_basin, b_grad = b_cal.optimizers[0][2](x)
        assert float(a_basin) == float(b_basin) and np.array_equal(a_grad, b_grad), step
        assert np.array_equal(a_cal.optimizers[1][2](x), b_cal.optimizers[1][2](x)), step
        assert np.array_equal(a_cal.optimizers[1][3](x), b_cal.optimizers[1][3](x)), step
        theta = torch.tensor(x, dtype=DTYPE)
        assert torch.equal(a_cal(theta), b_cal(theta)), step


# --------------------------- THE ANALYTIC QUOTE SIDE: separable, and gated on the separability

def repriced_value(world, calibration, theta):
    """The block's benchmarks priced by the ENGINE'S OWN Monte Carlo at `theta`, summed - the value
    the triangle and the direction check are taken in, deliberately not the analytic price. It
    shares no arithmetic with the residual under test, where an SP value would be that residual's
    front half read back to itself. Deterministic in the parameters: the Sobol sample is frozen.
    """
    return sum(world['objective'].reprice(calibration.split(theta))[0].values())


@pytest.fixture(scope='module')
def quote_solve():
    """ONE analytic solve with the quote side on, and every number below read off it. The FOUR-quote
    block, because this needs a real `LeastSquaresSolve.forward` and the identified block costs
    250 s a solve; the readings that need its RANK are taken there instead.
    """
    calibration, world = quote_calibration(theta={}, benchmarks=CHECKER_BENCHMARKS, chain=True,
                                           batch_size=2048)
    theta = LeastSquaresSolve.apply(calibration, declared('Jacobian_Rcond'),
                                    declared('Stationarity_Tol'), *calibration.quotes)
    value = repriced_value(world, calibration, theta)
    # read BEFORE anything else asks for a gradient - see the gate for what is standing here
    stale = [None if quote.grad is None else float(quote.grad) for quote in calibration.quotes]
    return dict(
        calibration=calibration, world=world, theta=theta, value=float(value.detach()),
        stale=stale,
        one_pass=np.array([float(g) for g in torch.autograd.grad(
            value, calibration.quotes, retain_graph=True)]),
        cotangent=torch.autograd.grad(value, theta, retain_graph=True)[0].double().numpy())


def test_the_quote_leaves_are_the_monte_carlo_paths_own_leaf_for_leaf():
    """NOTHING NEW AT THE ATTACHMENT SEAM. `create_market_swaps` mints the quote leaf and the
    premium map before anything knows which objective is being solved - `Quote_Sensitivity` is the
    only switch - so the analytic path inherits the leaf, map, descriptors, ordering and dtype
    rather than declaring them again. Held side by side on one block, and again one layer out at
    `bootstrap`, which publishes the same keys and descriptors with `.grad` cleared.
    """
    an, _ = quote_calibration(ID_ANALYTIC_THETA, benchmarks=CHECKER_BENCHMARKS)
    mc, _ = calibration_at(ID_THETA, benchmarks=CHECKER_BENCHMARKS, Objective='Monte_Carlo',
                           Quote_Sensitivity='Yes',
                           Instrument_Definitions=quoted_definitions(CHECKER_BENCHMARKS,
                                                                     [20.0] * 4))
    assert an.descriptors == mc.descriptors == list(an.market_swaps), an.descriptors
    assert len(an.quotes) == len(mc.quotes) == len(CHECKER_BENCHMARKS)
    for name, a, m in zip(an.descriptors, an.quotes, mc.quotes):
        assert (a.dtype, a.shape, a.requires_grad) == (torch.float64, m.shape, True), name
        assert float(a.detach()).hex() == float(m.detach()).hex() == float(0.20).hex(), name

    an_model, _ = bootstrap_the_block(Objective='Analytic', Quote_Sensitivity='Yes',
                                      Stationarity_Tol=ABSENT)
    mc_model, _ = bootstrap_the_block(Objective='Monte_Carlo', Quote_Sensitivity='Yes')
    an_desc, an_leaves = an_model.quote_leaves[ID_BLOCK]
    assert (an_desc, list(map(str, sorted(an_model.calibrated)))) == (
        mc_model.quote_leaves[ID_BLOCK][0], list(map(str, sorted(mc_model.calibrated))))
    assert an_desc == an.descriptors and len(an_leaves) == len(CHECKER_BENCHMARKS)
    assert all(leaf.grad is None for leaf in an_leaves), (
        'the optimizer chain called backward() on every evaluation it made, so a leaf published '
        'with `.grad` standing carries the sum over that whole path - `bootstrap` clears it, and '
        'test_the_quote_triangle_closes... measures what is standing there on this path')


def test_the_analytic_residual_is_separable_and_its_cross_term_is_structurally_zero():
    """THE SHAPE THE WHOLE BUILD RESTS ON, on the identified 25-quote block, four ways.

    `r_j = w_j (sigma_SP,j(theta) - sigma_mkt,j(q))` is a THETA-FUNCTION MINUS A Q-FUNCTION: the
    market half divides the twin premium by SP's annuity, built with `new_tensor` off a numpy curve
    read and so carrying no derivative in theta. Four consequences, held as arithmetic:

        dr/dq is DIAGONAL                600 of the 625 entries are STRUCTURALLY absent - autograd
                                         returns None off the unstacked residual, not a small
                                         number. The diagonal runs -0.10235 to -0.08706, which is
                                         dP/dq . sqrt(2pi/T0)/A
        d2r/dtheta dq == 0               `np.array_equal` both ways round, both taken AROUND the
                                         splice by rebuilding the block
        the annuity is severed           `sp.annuity.requires_grad` is False, which is why the two
                                         above are exact rather than small
        the theta-side dropped term      ||sum_i r_i grad^2 r_i||_F is 1.50e-4 of ||J'J||_F beside
                                         a ||r|| of 1.48e-3, so it is O(||r||)

    That last row is the point of retiring the quartic: on the Monte Carlo residual the same term
    is 0.500064 of `J'J` - a HALF at any residual level, because that residual is already a square,
    cancelled only by a second dropped term on the quote side. Here there is none and none is
    needed, and it is 5.5e-5 to 3.7e-4 of the leading eigenvalues rather than small only in norm.

    THE SPECTRUM IS THIS OBJECTIVE'S OWN: `J` is 25 x 23 with `sigma_min/sigma_max` 5.1e-15, and
    the declared `Jacobian_Rcond` of 1e-8 on `J'J`'s eigenvalues keeps 15 of 23 directions. What is
    unidentified is genuine - sigma knots past the last benchmark expiry are in no variance
    integral - and `dtheta/dq` there is the minimum-norm representative, as
    [Quote Sensitivities](quote_sensitivities.md#rank-deficiency) says.
    """
    calibration, world = quote_calibration(ID_ANALYTIC_THETA)
    theta = flat_theta(calibration, ID_ANALYTIC_THETA)
    residual, jacobian, quote_jac = residual_pieces(calibration, theta)
    n, unused = len(ID_GRID), unconnected_pairs(calibration, theta)
    assert unused == n * n - n, (
        'autograd found {} of {} quote/residual pairs unconnected - the separable residual has '
        'exactly one live quote per row'.format(unused, n * n))
    assert float(quote_jac.diagonal().abs().min()) > 1e-2, 'the diagonal has gone quiet'
    assert np.array_equal(quote_jac.numpy(),
                          np.diag(np.diag(quote_jac.numpy()))), 'dr/dq is not diagonal'
    assert 1e-3 < float(residual.norm()) < 2e-3, float(residual.norm())

    swap = calibration.market_swaps['Swaption_1Y_1Y']
    sp = world['process'].schrager_pelsser_swaption(
        swap.schedule.expiry, swap.schedule.pay_times, swap.schedule.accruals)
    assert not sp.annuity.requires_grad, (
        'the analytic annuity has picked up a graph - the market half of this residual would then '
        'reach theta and the cross term below would stop being zero')

    # the mixed partial, BOTH ways round and both AROUND the splice
    for column, bump in ((12, 0.5), (0, 0.1)):
        vols = [20.0] * n
        vols[column] = 20.0 + bump
        moved, _ = quote_calibration(ID_ANALYTIC_THETA, vols=vols)
        assert np.array_equal(residual_pieces(moved, theta)[1].numpy(), jacobian.numpy()), (
            'J moved when quote {} was re-authored {:+g} vol points - d2r/dtheta dq is not '
            'zero'.format(column, bump))
    for scale in (1.01, 0.97):
        assert np.array_equal(residual_pieces(calibration, theta * scale)[2].numpy(),
                              quote_jac.numpy()), 'dr/dq moved with theta'

    # the theta-side term Gauss-Newton drops, by double backward at the same point
    x = theta.detach().clone().requires_grad_(True)
    fresh = calibration(x)
    gradient = torch.autograd.grad((fresh.detach() * fresh).sum(), x, create_graph=True)[0]
    hessian = torch.stack([torch.autograd.grad(gradient[i], x, retain_graph=True)[0]
                           for i in range(gradient.numel())]).detach().double().numpy()
    gauss_newton = (jacobian.t() @ jacobian).numpy()
    ratio = float(np.linalg.norm(hessian) / np.linalg.norm(gauss_newton))
    assert 5e-5 < ratio < 5e-4, (
        'the dropped Hessian term is {:.4g} of J\'J against a recorded 1.50e-4 - it is O(||r||) on '
        'a residual nothing squared, and 0.5 is what a squared one reads'.format(ratio))
    eigenvalue, direction = np.linalg.eigh(gauss_newton)
    for k in (-1, -2, -4):
        u = direction[:, k]
        assert abs((u @ hessian @ u) / (u @ gauss_newton @ u)) < 5e-3, (
            'direction {}: {:.3g}'.format(k, (u @ hessian @ u) / (u @ gauss_newton @ u)))
    kept = int((eigenvalue > declared('Jacobian_Rcond') * eigenvalue.max()).sum())
    assert kept == 15, (
        'the declared cutoff keeps {} of 23 directions against a recorded 15 - the analytic '
        'residual identifies fewer of them than the squared one\'s 18, which is a property of '
        'this objective and is what dtheta/dq is a minimum-norm representative along'.format(kept))


def test_the_quote_triangle_closes_and_the_re_authored_rung_converges_as_h_squared(quote_solve):
    """THE TRIANGLE. One backward pass reports `dV/dq`; three routes reproduce it. `V` is the four
    benchmarks priced by the engine's own MONTE CARLO at theta*, so the value chain shares nothing
    with the residual under test. theta* is `AN_FOUR_THETA` to the bit, which is the quote side's
    no-op claim taken through a WHOLE optimizer chain rather than at a fixed theta.

        the contraction spelled out here, -(dr/dq)' J (J'J)^+ v          2.22e-16 relative
        the same, as v . dtheta/dq with dtheta/dq the OPERATOR           1.088e-14 relative
        the same with dr/dq from a re-authored central difference        h^2, 25.0x per 5x in h

    The operator form matters because it is what the direction check steps along. The third is the
    only route that is not autograd differentiating itself: the block is rebuilt from the JSON a
    rung either side, so the quotient goes AROUND the splice.

    `.grad` AFTER THE CHAIN STOPS holds 0.31% to 2.33% of the answer - basin hopping backwards on
    every evaluation and the leaves accumulate. The Monte Carlo path's version is six orders out
    with a NaN in it and could not be mistaken for an answer; this one could, which is why the gate
    reads `dV/dq` through `autograd.grad` and `bootstrap` clears the leaves before publishing.

    The second differentiation refuses: a Gauss-Newton contraction carries no second derivative.
    """
    calibration, world = quote_solve['calibration'], quote_solve['world']
    theta, one_pass, v = quote_solve['theta'], quote_solve['one_pass'], quote_solve['cotangent']
    assert 0.1 < quote_solve['value'] < 0.2 and (one_pass > 1e-3).all(), quote_solve['value']
    # the Yes-vs-No bit-identity through two WHOLE chains: this solve had the quote side ON and
    # `test_the_analytic_solve_is_deterministic...` re-derives the same vector with it OFF
    solved = calibration.unflatten(theta.detach())
    for name, recorded in AN_FOUR_THETA.items():
        assert [float(value).hex() for value in np.atleast_1d(solved[name])] == [
            float(value).hex() for value in recorded], (
            '{}: the quote side moved theta*, which the splice cannot do - it solved to {} against '
            'the recorded {}'.format(name, list(solved[name]), recorded))

    residual, jacobian, quote_jac = residual_pieces(calibration, theta.detach())
    norm = float((jacobian.t() @ residual).norm())
    assert norm < declared('Stationarity_Tol'), (
        'the four-quote analytic chain reads ||J\'r|| {:.3g} against the DECLARED 1e-3 this path '
        'runs at - no fixture tolerance is written on it'.format(norm))
    pseudo = torch.linalg.pinv(jacobian.t() @ jacobian, hermitian=True,
                               rtol=declared('Jacobian_Rcond'))
    contraction = -(quote_jac.t() @ (jacobian @ (pseudo @ torch.from_numpy(v)))).numpy()
    assert np.abs(contraction / one_pass - 1.0).max() < 1e-11, (contraction, one_pass)
    operator = (-pseudo @ jacobian.t() @ quote_jac).numpy()
    assert np.abs((v @ operator) / one_pass - 1.0).max() < 1e-11, (v @ operator, one_pass)

    named, reading = calibration.unflatten(theta.detach()), {}
    for bump in (0.5, 0.1, 0.02):
        finite = torch.zeros_like(quote_jac)
        for column in range(len(CHECKER_BENCHMARKS)):
            side = {}
            for sign in (+1, -1):
                vols = [20.0] * len(CHECKER_BENCHMARKS)
                vols[column] = 20.0 + sign * bump
                rung, _ = quote_calibration(named, benchmarks=CHECKER_BENCHMARKS, vols=vols,
                                            batch_size=2048)
                side[sign] = rung(theta.detach()).detach().double()
            finite[:, column] = (side[+1] - side[-1]) / (2.0 * bump / 100.0)
        rebuilt = -(finite.t() @ (jacobian @ (pseudo @ torch.from_numpy(v)))).numpy()
        reading[bump] = (float((finite - quote_jac).abs().max()),
                         float(np.abs(rebuilt / one_pass - 1.0).max()))
        assert np.array_equal(finite.numpy(), np.diag(np.diag(finite.numpy()))), (
            'the re-authored difference is not diagonal either, so the structural zero above is '
            'not an artefact of the tape')
    for coarse, fine in ((0.5, 0.1), (0.1, 0.02)):
        assert 15.0 < reading[coarse][1] / reading[fine][1] < 40.0, (
            'the re-authored rung reads {:.4g} at h={} and {:.4g} at h={} - a ratio of {:.1f} '
            'against the 25 that h^2 owes'.format(reading[coarse][1], coarse, reading[fine][1],
                                                  fine, reading[coarse][1] / reading[fine][1]))
    assert reading[0.02][1] < 1e-7, reading

    # the second differentiation, refused here as on the other residual
    with pytest.raises(Exception, match='create_graph'):
        torch.autograd.grad(theta.sum(), calibration.quotes, create_graph=True)

    stale = np.array([0.0 if g is None else g for g in quote_solve['stale']])
    assert (stale != 0.0).all() and (np.abs(stale / one_pass) < 0.05).all(), (
        'the optimizer chain left {} standing in `.grad` against a one-pass {} - if that is now '
        'zero the chain has stopped touching the quotes and this gate measures nothing; if it is '
        'orders larger the reading below has changed'.format(list(stale), list(one_pass)))


def test_stepping_theta_by_dtheta_dq_reprices_the_move_the_quotes_identify(quote_solve):
    """THE DIRECTION CHECK, in value space. Step theta by `dtheta/dq . h` and re-price WITHOUT
    re-solving, so nothing about where an optimizer stopped enters it. The two halves come from
    different places on purpose - the STEP from the contraction spelled out in this file, the
    PREDICTION from `LeastSquaresSolve.backward` - so a sign flipped inside that backward moves one
    and not the other.

    Every benchmark's repriced-over-predicted ratio closes on 1 from both sides and LINEARLY in h:
    0.956 to 1.039 at a tenth of a vol point, against 0.64 to 1.37 at one. The gap is the curvature
    of `V` along the step, so it halves when h does, which is stronger than any single rung.

    The mandated mutation: flipping the sign of `grad_outputs` in that backward makes every ratio
    read its own negative while nothing in the forward moves and no price gate sees it.
    """
    calibration, world = quote_solve['calibration'], quote_solve['world']
    theta, one_pass = quote_solve['theta'], quote_solve['one_pass']
    _, jacobian, quote_jac = residual_pieces(calibration, theta.detach())
    operator = (-torch.linalg.pinv(jacobian.t() @ jacobian, hermitian=True,
                                   rtol=declared('Jacobian_Rcond'))
                @ jacobian.t() @ quote_jac).numpy()
    flat = theta.detach().double().numpy()

    reading = {}
    for column in range(len(CHECKER_BENCHMARKS)):
        for bump in (1.0, 0.1):
            for sign in (+1, -1):
                moved = torch.tensor(flat + sign * operator[:, column] * (bump / 100.0),
                                     dtype=DTYPE)
                reading[column, bump, sign] = float(repriced_value(
                    world, calibration, moved).detach()) - quote_solve['value']
    for column in range(len(CHECKER_BENCHMARKS)):
        for sign in (+1, -1):
            ratios = [reading[column, bump, sign] / (sign * one_pass[column] * bump / 100.0)
                      for bump in (1.0, 0.1)]
            assert 0.9 < ratios[1] < 1.1, (
                'benchmark {} at h={:+g} vol points reprices {:.4f} of the move the backward '
                'predicted - the recorded band is 0.956 to 1.039'.format(
                    CHECKER_BENCHMARKS[column][:2], sign * 0.1, ratios[1]))
            assert abs(ratios[1] - 1.0) < abs(ratios[0] - 1.0), (
                'benchmark {}: the ratio reads {:.4f} at one vol point and {:.4f} at a tenth, so '
                'it is not converging - a first-order prediction owes a gap linear in h'.format(
                    CHECKER_BENCHMARKS[column][:2], ratios[0], ratios[1]))


def test_the_re_solve_oracle_still_scatters_once_the_solve_does_reach_stationarity():
    """A NEGATIVE RESULT: bump a quote, re-solve, difference theta* - and it is still not
    `dtheta/dq`, on a chain that now reaches stationarity.

    The reference was refuted three times on the Monte Carlo path
    ([Quote Sensitivities](quote_sensitivities.md#the-manifold-finding)) with a two-part diagnosis:
    the solve wanders where the objective is FLAT, and it also STOPS SHORT at `||J'r||` 3.16e2.
    The analytic objective removes the second half - all six re-solves here land between 8.5e-7 and
    9.7e-7, three orders inside the declared 1e-3 - so this ladder tests whether that was the half
    that mattered. It was not.

    Quote 12 (3Y x 3Y) against a one-pass `||dtheta/dq||` of 37.64: the quotient GROWS as h shrinks
    (3.24 at h=0.5 to 5.71 at h=0.1) and is seven to twelve times too small at every rung, and the
    displacement points the WRONG WAY - cosine -0.42 to -0.25 where a derivative owes +1 and a
    random direction +-0.209. An anti-aligned displacement is not a noisy derivative.

    So the classic oracle is unavailable and `dtheta/dq` is NOT gated against it; the triangle and
    the value-space direction check are what gate it. This gate passes by FAILING to agree, and a
    flip would mean the solve had started returning a function of its quotes.

    Six cold 25-quote solves, 210 to 330 s each - which is why the ladder is one column.
    """
    column, bumps = 12, (0.5, 0.2, 0.1)
    calibration, _ = quote_calibration(ID_ANALYTIC_THETA)
    theta = flat_theta(calibration, ID_ANALYTIC_THETA)
    _, jacobian, quote_jac = residual_pieces(calibration, theta)
    gauss_newton = jacobian.t() @ jacobian
    predicted = (-torch.linalg.pinv(gauss_newton, hermitian=True,
                                    rtol=declared('Jacobian_Rcond'))
                 @ jacobian.t() @ quote_jac).numpy()[:, column]
    eigenvalue, direction = np.linalg.eigh(gauss_newton.numpy())
    kept = direction[:, eigenvalue > declared('Jacobian_Rcond') * eigenvalue.max()]
    assert 30.0 < np.linalg.norm(predicted) < 45.0, np.linalg.norm(predicted)

    base, reading = flat_theta(calibration, ID_ANALYTIC_THETA).double().numpy(), {}
    for bump in bumps:
        solved = {}
        for sign in (+1, -1):
            vols = [20.0] * len(ID_GRID)
            vols[column] = 20.0 + sign * bump
            chain, _ = identified_calibration(
                benchmarks=ID_GRID, Objective='Analytic', Stationarity_Tol=ABSENT,
                Instrument_Definitions=quoted_definitions(ID_GRID, vols))
            solved[sign] = chain.solve()
            norm = stationarity(chain, solved[sign])[0]
            assert norm < declared('Stationarity_Tol'), (
                'the re-solve at {:+g} vol points stopped at ||J\'r|| {:.4g}, outside the declared '
                '1e-3 - the recorded band is 8.6e-7 to 1.6e-5, and half of what this gate says is '
                'that the analytic chain gets there'.format(sign * bump, norm))
            solved[sign] = solved[sign].detach().double().numpy()
        moved = solved[+1] - solved[-1]
        quotient = moved / (2.0 * bump / 100.0)
        reading[bump] = dict(
            moved=float(np.linalg.norm(moved)), quotient=float(np.linalg.norm(quotient)),
            fraction=float(np.linalg.norm(quotient) / np.linalg.norm(predicted)),
            cosine=float(quotient @ predicted /
                         (np.linalg.norm(quotient) * np.linalg.norm(predicted))),
            inside=float(np.linalg.norm(kept.T @ moved) / np.linalg.norm(moved)),
            away=[float(np.linalg.norm(solved[s] - base)) for s in (+1, -1)])

    coarse, fine = reading[bumps[0]], reading[bumps[-1]]
    assert fine['quotient'] > 1.5 * coarse['quotient'], (
        'the re-solve quotient reads {:.4g} at h={} and {:.4g} at h={} - it CONVERGED. If that is '
        'real the solve has started returning a function of its quotes and dtheta/dq now has a '
        'classic oracle to be gated against; the recorded readings are 3.236 and 5.714'.format(
            coarse['quotient'], bumps[0], fine['quotient'], bumps[-1]))
    for bump in bumps:
        got = reading[bump]
        assert abs(got['fraction'] - 1.0) > 0.5, (
            'h={}: the re-solve displacement is {:.4g} of the one-pass derivative - the recorded '
            'readings are 0.086 / 0.105 / 0.152, seven to twelve times too small. If this is now '
            'near 1.0 the oracle has become available and dtheta/dq can be gated against '
            'it'.format(bump, got['fraction']))
        # the direction, which is the sharper half: anti-aligned, not merely orthogonal
        assert got['cosine'] < 0.0, (
            'h={}: the displacement now points WITH the one-pass derivative at a cosine of {:+.4f} '
            '- the recorded readings are -0.42 / -0.34 / -0.25 and an anti-aligned displacement is '
            'what says this is not a noisy derivative'.format(bump, got['cosine']))
        assert abs(got['cosine']) < 0.6, (
            'h={}: cosine {:+.4f} against a recorded worst of -0.4172 - if the magnitude is '
            'climbing toward 1 the re-solve is starting to track the derivative'.format(
                bump, got['cosine']))
        assert 0.5 < got['inside'] < 0.995, (
            'h={}: {:.3f} of the displacement lands in the 15 directions the cutoff keeps, against '
            'the sqrt(15/23) = 0.808 a random one would - the recorded readings are 0.578 to 0.981, '
            'which brackets that number rather than sitting to one side of it'.format(
                bump, got['inside']))
        assert got['moved'] > 5e-3 and min(got['away']) > 1e-3, (
            'h={}: the two re-solves land {} from the recorded theta* against a bump worth {:.4g} '
            'in theta - the recorded distances are 0.006 to 0.021'.format(
                bump, got['away'], np.linalg.norm(predicted) * bump / 100.0))
    # the distance falls with h - the chain is stable enough that a smaller bump lands nearer
    # theta*. Still not a derivative; the cosine says so.
    assert min(fine['away']) < min(coarse['away']), (
        'the distance from theta* no longer falls with h - {:.4g} at h={} against {:.4g} at h={}. '
        'Before 2026-09-02 it did NOT fall (0.279 against 0.268, a floor set by where each search '
        'stopped); the re-mark made the chain stable and this is where that shows'.format(
            min(fine['away']), bumps[-1], min(coarse['away']), bumps[0]))
    assert min(coarse['away']) / min(fine['away']) < 10.0, (
        'the coarse rung lands {:.3g}x further from theta* than the fine one - the recorded ratio '
        'is 2.7x, and a large one would mean a re-solve had found another basin again'.format(
            min(coarse['away']) / min(fine['away'])))


def test_a_premium_re_struck_by_volatility_delta_declines_the_analytic_quote_side():
    """THE ONE REFUSAL THAT SURVIVES, on both objectives: a `Volatility_Delta` bump on a
    PREMIUM-quoted block recovers an implied vol with `brentq` and re-strikes off it, and a
    numerical root find carries no derivative. `create_market_swaps` raises BEFORE either objective
    exists, because the severance is at the market premium.

    Both halves of the pair are exercised, so the refusal is known to be on the PAIR: a premium
    block with no bump builds and carries the PREMIUM on its leaf with the identity map, and a bump
    with no premium file builds because there is nothing to re-strike.
    """
    world = identified_closure(benchmarks=CHECKER_BENCHMARKS, batch_size=1024, Objective='Analytic')
    frame = pd.DataFrame([
        {'Currency': ID_CCY, 'Expiry': '{}Y'.format(e), 'UnderlyingTenor': '{}Y'.format(t),
         'Payer': 1e4 * world['swaps']['Swaption_{}Y_{}Y'.format(e, t)].price,
         'Shift': '0%', 'StrikeValue': 8.0} for e, t, _, _ in CHECKER_BENCHMARKS])

    for objective in ('Analytic', 'Monte_Carlo'):
        with pytest.raises(Exception, match='brentq'):
            identified_closure(benchmarks=CHECKER_BENCHMARKS, batch_size=1024,
                               Objective=objective, Quote_Sensitivity='Yes',
                               premiums=frame, delta=0.005)
        # the two halves of the pair, so this is not a refusal of premiums or of a bump
        quoted = identified_closure(benchmarks=CHECKER_BENCHMARKS, batch_size=1024,
                                    Objective=objective, Quote_Sensitivity='Yes', premiums=frame)
        swap = quoted['swaps']['Swaption_1Y_1Y']
        assert abs(float(swap.quote.detach()) / swap.price - 1.0) < 1e-12, (
            'a premium-quoted block carries the PREMIUM on the leaf and the identity map')
        assert abs(float(swap.premium(swap.quote).detach()) / swap.price - 1.0) < 1e-12
        identified_closure(benchmarks=CHECKER_BENCHMARKS, batch_size=1024, Objective=objective,
                           Quote_Sensitivity='Yes', delta=0.005)


def test_a_truncated_chain_refuses_the_analytic_quote_jacobian_and_names_the_norm():
    """STATIONARITY IS CHECKED, NOT ASSUMED, and the refusal is REACHABLE on this path too. `solve`
    accepts whatever the chain returned - the SEED if nothing beat it - and the Gauss-Newton
    contraction is worthless off the fixed point, so `LeastSquaresSolve.backward` raises naming the
    norm. Nothing patched: a `SwaptionCalibration` with an empty `optimizers`.

        chain                    ||J'r|| at what it returned      against the DECLARED 1e-3
        the seed, nothing run              9.64e-03               refused, naming the norm
        basin hopping alone                5.85e-07               ACCEPTED
        the full chain                     6.91e-08               accepted

    The middle row is 2.9e+10 on the Monte Carlo residual - a quartic goes flat enough that the
    relative-improvement test fires thirteen orders short - so the refusal has to be driven from
    the seed here, which is what this does.
    """
    calibration, world = quote_calibration(theta={}, benchmarks=CHECKER_BENCHMARKS, chain=True,
                                           batch_size=2048)

    def stage(optimizers):
        stopped = SwaptionCalibration('gate', calibration.objective, calibration.implied_var,
                                      optimizers, calibration.process, calibration.market_swaps)
        return stopped, LeastSquaresSolve.apply(stopped, declared('Jacobian_Rcond'),
                                                declared('Stationarity_Tol'), *stopped.quotes)

    seeded, theta = stage([])
    norm = stationarity(seeded, theta.detach())[0]
    assert norm > declared('Stationarity_Tol'), (
        'the seed reads ||J\'r|| {:.4g} against a recorded 9.64e-3, which is inside the declared '
        'tolerance - this gate no longer reaches the refusal it exists for'.format(norm))
    with pytest.raises(Exception, match='not stationary') as refusal:
        repriced_value(world, seeded, theta).backward()
    assert '{:.6g}'.format(norm) in str(refusal.value), (
        'the refusal has to name the norm it read: {}'.format(refusal.value))
    assert 'Stationarity_Tol' in str(refusal.value), refusal.value

    # the other end: the basin stage alone is already inside the declared tolerance here
    basin, theta = stage(calibration.optimizers[:1])
    assert stationarity(basin, theta.detach())[0] < declared('Stationarity_Tol'), (
        'basin hopping alone no longer clears the declared tolerance on the analytic residual - '
        'the recorded reading is 5.85e-7 and it is half of why the default is 1e-3')
    repriced_value(world, basin, theta).backward()


# =================================================================================================
# THE PREMIUM CONVENTION - what the surface DECLARES, read by the calibration that prices its quotes
# =================================================================================================

#: normal vols in the family's `Percent` column, ZAR-shaped: 145 basis points is 1.45 here, and
#: `.amount` is then the ABSOLUTE rate move 0.0145 the Bachelier premium wants. The four
#: `CHECKER_BENCHMARKS` rows in order.
NORMAL_VOLS = (1.45, 1.33, 1.26, 1.18)

#: theta* on the four-quote fixture with the surface declaring `Distribution_Type: 'Normal'`, AS
#: SOLVED - the Normal path's bit-identity baseline. `Random_Seed` 5120, CPU float64, 11.5 s.
#: `||J'r||` 1.17e-8 against `||r||` 2.10e-8: it INTERPOLATES, five orders inside the declared 1e-3,
#: and no authored vector lands there. It is NOT `AN_FOUR_THETA` and must not be - the same four
#: numeric quotes read as normal vols are a different market, priced an order of magnitude higher.
NORMAL_FOUR_THETA = {
    'Alpha_1': [0.24966452368249561],
    'Alpha_2': [0.0488512794103916],
    'Correlation': [-0.10980525520167612],
    'Sigma_1': [0.012543554866911631, 0.007566168859012784, 0.014933287188537762,
                0.007721973683818293, 0.004688568766587583, 0.008701363306803303,
                0.012725926961641956, 0.004004880678613152, 0.00803724607640168,
                0.014724211527068123],
    'Sigma_2': [0.004425853435529894, 0.015578208610304378, 0.010456826442251882,
                0.008118947663983501, 0.022538909612556043, 0.007469344759637167,
                0.007904411824524817, 0.019770314249231875, 0.01973294557745891,
                0.024658651123981138]}


def surface_world(**declared):
    """The identified fixture with `declared` written onto its `InterestYieldVol` block - one
    fixture, two declarations, everything else the identified world's. The `Surface` quads are
    re-scaled to normal units under `'Normal'`: they no longer reach the premium, but a surface
    saying 20% beside rows quoting 145bp would describe a market that does not exist.
    """
    factors = identified_world()
    block = factors['InterestYieldVol.' + ID_VOL]
    block.update(declared)
    if declared.get('Distribution_Type') == 'Normal':
        block['Surface'] = utils.Curve([], [
            [m, e, t, 0.0145 + 0.0005 * np.log1p(e) - 0.0002 * np.log1p(t) + 0.05 * m * m]
            for t in ID_SURFACE_T for e in ID_SURFACE_E for m in ID_MONEY])
    return factors


def normal_closure(vols=NORMAL_VOLS, **extra):
    """The four-quote closure on a surface declaring `Normal`, its rows quoting normal vols."""
    return identified_closure(
        benchmarks=CHECKER_BENCHMARKS, world=surface_world(Distribution_Type='Normal'),
        Objective='Analytic',
        Instrument_Definitions=quoted_definitions(CHECKER_BENCHMARKS, vols), **extra)


def normal_calibration(vols=NORMAL_VOLS, **extra):
    """`(SwaptionCalibration, world)` on that block, WITH the optimizer chain."""
    return identified_calibration(
        benchmarks=CHECKER_BENCHMARKS, world=surface_world(Distribution_Type='Normal'),
        Objective='Analytic',
        Instrument_Definitions=quoted_definitions(CHECKER_BENCHMARKS, vols), **extra)


def struck_annuity(swap, curve):
    """The annuity the market premium was STRUCK on, rebuilt from the schedule and the curve - the
    same expression on the same arrays as `get_par_swap_rate`'s pvbp, so bit-identical to it by
    construction and the round trip below is an identity rather than a tolerance.

    Not SP's annuity, which the residual uses and which agrees with this to about 1e-3: the round
    trip is a statement about the quote and the premium.
    """
    t = swap.schedule.pay_times
    return float((np.exp(-curve.current_value(t) * t) * swap.schedule.accruals).sum())


#: THE CLOCK THAT USED TO BE THERE, kept as the number the gates below assert the answer is NOT.
#: `create_market_swaps` measured its expiry in 365.25ths while the Bachelier inversion read
#: `schedule.expiry` in the curve's day count, so a quoted normal vol came back as
#: `sigma * sqrt(T_365.25/T_curve)` - one number for the whole ladder, both clocks being linear.
OLD_CLOCK = np.sqrt(365.0 / 365.25)


def test_a_normal_surface_calibrates_and_the_market_side_round_trips():
    """A NORMAL-VOL LADDER FITS AS A NORMAL-VOL LADDER. `SASN` is quoted in basis points of
    ABSOLUTE rate move and the family priced every premium with the numpy Black whatever the
    surface declared; `create_market_swaps` reads `Distribution_Type` through `Factor3D.get_subtype`
    now - the DEAL path's own read, so one spelling of what a surface says - and this block
    calibrates to `NORMAL_FOUR_THETA`, `||J'r||` under the declared 1e-3 and `||r||` under 1e-4.

    THE MARKET SIDE ROUND-TRIPS, which is the half a price gate cannot see: at the money the
    Bachelier premium is `A sigma_N sqrt(T/2pi)` and `market_normal_vol` inverts exactly that, so a
    quote goes in through the premium and comes back AS ITSELF, 0.0 to 2.2e-16 on the annuity it
    was struck on. It used to come back times `OLD_CLOCK`, a 3.4e-4 bias.

    THE TWO CLOCKS STILL EXIST and that is asserted, because a round trip reading 1.0 is also what
    a fixture with no clock difference would read: `exp_days / DAYS_IN_YEAR` and the curve accrual
    are 6.85e-4 apart here, and what changed is which the premium is priced on.
    """
    calibration, world = normal_calibration()
    theta = calibration.solve()
    solved = calibration.unflatten(theta)
    for name, vector in NORMAL_FOUR_THETA.items():
        assert [float(v).hex() for v in np.atleast_1d(solved[name])] == [
            float(v).hex() for v in vector], (
            '{} solved to {} against the recorded {}'.format(name, list(solved[name]), vector))
    assert [float(v).hex() for v in np.concatenate(
        [np.atleast_1d(AN_FOUR_THETA[k]) for k in calibration.keys])] != [
        float(v).hex() for v in theta.detach().numpy()], (
        'the Normal block solved to the LOGNORMAL four-quote vector - the same four numbers read '
        'under the other convention are a different market, so this would mean the declaration '
        'reached nothing')
    norm, residual = stationarity(calibration, theta.detach())
    assert norm < declared('Stationarity_Tol'), (
        "a Normal block reads ||J'r|| {:.4g} at theta* against a recorded 3.97e-7 and the declared "
        '{:g}'.format(norm, declared('Stationarity_Tol')))
    assert residual < 1e-4, 'the four-quote Normal fit reads ||r|| {:.3e} against 1.75e-6'.format(
        residual)

    # the round trip, on the annuity the premium was struck on - the quote AS ITSELF
    worst = 0.0
    for (name, swap), quoted, row in zip(world['swaps'].items(), NORMAL_VOLS,
                                         CHECKER_BENCHMARKS):
        sigma = utils.Percent(quoted).amount
        recovered = as_float(swap.market_normal_vol(
            torch.tensor(struck_annuity(swap, world['curve']), dtype=DTYPE)))
        assert abs(recovered / sigma - 1.0) < 1e-14, (
            '{}: a quoted sigma_N of {:.6g} came back as {:.17g} - the Bachelier premium and its '
            'closed-form inversion are not each other'.format(name, sigma, recovered))
        worst = max(worst, abs(recovered / sigma - 1.0))
        # and it is NOT the old clock: that bias was 3.4e-4, four orders above what is left
        assert abs(recovered / sigma - OLD_CLOCK) > 1e-4, (
            '{}: the round trip is still carrying sqrt(T_365.25/T_curve)'.format(name))
        # the two clocks are still two numbers, or the identity above passes for no reason
        expiry_365_25 = (BASE + pd.DateOffset(years=int(row[0])) - BASE).days / utils.DAYS_IN_YEAR
        assert abs(swap.schedule.expiry / expiry_365_25 - 365.25 / 365.0) < 1e-15, (
            '{}: the fixture no longer has two clocks in it, so this gate reaches nothing'.format(
                name))
    assert worst < 1e-14, 'the worst round-trip residual is {:.3e} against a recorded 2.2e-16'.format(
        worst)
    assert abs(OLD_CLOCK - 0.99965771007041049) < 1e-15, (
        'the old clock constant moved - DAYS_IN_YEAR or the fixture day count changed')


def test_the_two_conventions_are_two_prices_and_the_normal_one_is_the_bachelier_premium():
    """THE GATE THAT WOULD HAVE CAUGHT THE ORIGINAL DEFECT: one numeric ladder, two declarations,
    two prices, each held to its OWN closed form.

    Read once as a lognormal Black vol and once as an absolute normal vol on one fixture, the
    premiums come out 9.68x to 11.37x apart - which is 1/F to a part in 1e5, the par swap rate here
    running 8.8% to 10.3%. That is what "prices every market premium LOGNORMAL" cost, silently,
    with `Distribution_Type` declared on the surface all along.

    Each side against its own closed form off the pvbp the premium was struck on - Bachelier
    `A sigma sqrt(T/2pi)` and Black `A K (2 Phi(sigma sqrt(T)/2) - 1)`, both to 1e-15 - and the
    TENSOR twin the quote side differentiates held bit-identical to the numpy premium.
    """
    lognormal = identified_closure(
        benchmarks=CHECKER_BENCHMARKS, world=surface_world(Distribution_Type='Lognormal'),
        Objective='Analytic', batch_size=2048,
        Instrument_Definitions=quoted_definitions(CHECKER_BENCHMARKS, NORMAL_VOLS))
    normal = normal_closure(batch_size=2048, Quote_Sensitivity='Yes', Stationarity_Tol=ABSENT)
    factors = []
    for (name, swap), quoted in zip(normal['swaps'].items(), NORMAL_VOLS):
        sigma, other = utils.Percent(quoted).amount, lognormal['swaps'][name]
        # one clock: the closed forms are checked at the year fraction the pricer was handed
        annuity, T = struck_annuity(swap, normal['curve']), swap.schedule.expiry
        assert struck_annuity(other, lognormal['curve']) == annuity, name
        assert abs(swap.price / (annuity * sigma * np.sqrt(T / (2.0 * np.pi))) - 1.0) < 1e-15, (
            '{}: the Normal premium is not the Bachelier one'.format(name))
        # the strike is the par swap rate, which is what the two premiums differ BY
        strike = other.price / (annuity * (2.0 * scipy.stats.norm.cdf(
            sigma * np.sqrt(T) / 2.0) - 1.0))
        assert 0.08 < strike < 0.11, '{}: the implied par rate {:.6g} is off this curve'.format(
            name, strike)
        factors.append(swap.price / other.price)
        assert abs(factors[-1] * strike - 1.0) < 1e-4, (
            '{}: the two conventions differ by {:.6g}x against 1/F of {:.6g}'.format(
                name, factors[-1], 1.0 / strike))
    assert min(factors) > 5.0, (
        'the two conventions price within {:.3g}x of each other - if that is now 1.0 the '
        'declaration has stopped reaching the premium: {}'.format(
            min(factors), ['{:.3f}'.format(f) for f in factors]))

    # the tensor twin is the same formula in the other precision, per convention
    for name, swap in normal['swaps'].items():
        assert float(swap.premium(swap.quote).detach()).hex() == float(swap.price).hex(), (
            '{}: the float64 Bachelier twin is not the numpy premium it splices onto'.format(name))


#: THE LADDER THE OLD BRACKET COULD NOT REACH: 78bp and 40bp of absolute rate move, alternating so
#: both levels are read at a short expiry and a long one. Ordinary EUR and JPY normal vols, sitting
#: UNDER the old fixed floor of `0.01`, which is 100bp in these units.
LOW_NORMAL_VOLS = (0.78, 0.40, 0.78, 0.40)

#: 5bp of ABSOLUTE vol against quotes of 40 and 78 - a tenth of the quote, not a rounding of it
RESTRIKE_DELTA = 0.0005

#: THE LOGNORMAL ARM'S RE-STRUCK PREMIUM, HEX, on the 20% ladder and on the 1.45%..1.18% one - the
#: nearest a lognormal quote gets to the old floor without going under it.
#: `IMPLIED_VOL_BRACKETS['Lognormal']` is the historical `(0.01, vol + .5)` expression itself, so
#: the bracket is unmoved by construction and these eight numbers hold everything else that could
#: reach them. They moved once, by the +3.31e-4 to +3.41e-4 the expiry clock owed.
LOGNORMAL_RESTRUCK = {
    'Swaption_1Y_1Y': '0x1.9cae339f80687p-8',
    'Swaption_2Y_5Y': '0x1.39a1426559910p-5',
    'Swaption_3Y_3Y': '0x1.c61cf04d59a3cp-6',
    'Swaption_10Y_10Y': '0x1.f58c44f50b9c5p-5'}
LOGNORMAL_LOW_RESTRUCK = {
    'Swaption_1Y_1Y': '0x1.3a95b474ea4d1p-11',
    'Swaption_2Y_5Y': '0x1.c183a49f3d69dp-9',
    'Swaption_3Y_3Y': '0x1.3986da4e106bfp-9',
    'Swaption_10Y_10Y': '0x1.4e8d7b3fbb9b6p-8'}


def premium_frame(world):
    """The `Swaption_Premiums` frame carrying THIS world's own priced premiums, so a recovered
    implied vol has a known answer - the row's own quote - and the round trip is an identity.
    `get_premium` divides the `Payer` column by 1e4 and this multiplies by it.

    That scaling is NOT the identity in binary, which is why the gates below hold it to one ulp:
    1e4 is not a power of two, so `x * 1e4 / 1e4` rounds twice and `Swaption_2Y_5Y` comes back one
    ulp low. The file's scaling, not the solve's, and the assertion has to be able to say so.
    """
    return pd.DataFrame([
        {'Currency': ID_CCY, 'Expiry': '{}Y'.format(e), 'UnderlyingTenor': '{}Y'.format(t),
         'Payer': 1e4 * world['swaps']['Swaption_{}Y_{}Y'.format(e, t)].price,
         'Shift': '0%', 'StrikeValue': 8.0} for e, t, _, _ in CHECKER_BENCHMARKS])


def old_bracket_residual(swap, world, premium):
    """The implied-vol residual `create_market_swaps` brackets, rebuilt out of library code - a
    reproduction, nothing patched - so bracketing it with the OLD bounds says what the old code
    would have done on THIS fixture.

    The STRIKE is irrelevant and that is the point: at `F = X` the Bachelier premium is
    `A sigma sqrt(T/2pi)` and does not mention it, so the fallback re-solve at the premium file's
    own strike is the SAME function and a normal ladder under the floor was fatal rather than
    degraded. The expiry is the SCHEDULE's, the clock the premium is struck on.
    """
    annuity = struck_annuity(swap, world['curve'])
    expiry = swap.schedule.expiry
    return lambda v: annuity * utils.bachelier_european_option_price(
        0.08, 0.08, 0.0, v, expiry, 1.0, 1.0) - premium


def test_the_normal_re_strike_brackets_in_its_own_scale_and_the_lognormal_arm_is_unmoved():
    """THE RE-STRIKE BRACKET IS THE QUOTE'S SCALE, and under `Normal` that is not a fixed 1%.

    The bracket was `(0.01, vol + .5)` whatever the surface declared. Under `Lognormal` that floor
    sits under every quoted surface; under `Normal` a vol IS an absolute rate move, so 0.01 is 100
    basis points and an ordinary EUR or JPY quote sits BELOW it - both ends carry the same sign and
    `brentq` raises. No fixture in this repository reached that arm, which is what this gate is.

    The turnover is the floor itself: with the premium file carrying the block's own premiums the
    implied vol IS the quoted vol, so `f(0.01)` changes sign exactly where the quote crosses 100bp.
    At 78bp and 40bp it reads +7.7e-4 to +2.0e-2, three to five orders above the noise on an
    annuity of order one - a SIGN failure, not a near miss.

    `IMPLIED_VOL_BRACKETS` is co-keyed with `PREMIUM_CONVENTIONS` and read off the same declared
    `Distribution_Type`, so the convention that picks the pricer picks its bracket's scale.
    `'Normal'` brackets MULTIPLICATIVELY around the row's own quote, which is scale-free in the
    units the quote is in; at the money Bachelier is linear in the vol, so the bracket has only to
    contain a division. The round trip is then an identity: the recovered vol re-strikes to the
    premium a block QUOTING `sigma + delta` prices outright, 0.0 to 3.3e-16, one ulp.

    THE LOGNORMAL ARM IS UNMOVED, held as eight hex premiums across two ladders.
    """
    from derivus import bootstrappers

    # one vocabulary, and the lognormal entry is the historical literal
    assert sorted(bootstrappers.IMPLIED_VOL_BRACKETS) == sorted(
        bootstrappers.PREMIUM_CONVENTIONS), (
        'the bracket table and the pricer table have drifted apart - a convention can now arrive '
        'carrying one and not the other: {} against {}'.format(
            sorted(bootstrappers.IMPLIED_VOL_BRACKETS),
            sorted(bootstrappers.PREMIUM_CONVENTIONS)))
    for vol in (0.20, 0.0145, 0.0078):
        assert [float(b).hex() for b in bootstrappers.IMPLIED_VOL_BRACKETS['Lognormal'](vol)] == [
            float(b).hex() for b in (0.01, vol + .5)], (
            'the Lognormal bracket is no longer the historical (0.01, vol + .5) at vol={}'.format(
                vol))

    # ------------------------------------------------------------------ the Normal arm, at 40/78bp
    base = normal_closure(vols=LOW_NORMAL_VOLS, batch_size=512)
    frame = premium_frame(base)
    struck = normal_closure(vols=LOW_NORMAL_VOLS, batch_size=512,
                            premiums=frame, delta=RESTRIKE_DELTA)
    # the same block QUOTING the bumped vol, which is where the re-strike has to land
    direct = normal_closure(
        vols=tuple(v + RESTRIKE_DELTA * 100.0 for v in LOW_NORMAL_VOLS), batch_size=512)

    worst = 0.0
    for (name, swap), quoted, row in zip(struck['swaps'].items(), LOW_NORMAL_VOLS,
                                         CHECKER_BENCHMARKS):
        sigma, priced = utils.Percent(quoted).amount, base['swaps'][name].price
        assert sigma < 0.01, (
            '{}: a quote of {:.6g} is NOT under the old 0.01 floor, so this fixture no longer '
            'reaches the arm it was built for'.format(name, sigma))
        picked = frame[(frame['Expiry'] == '{}Y'.format(row[0])) &
                       (frame['UnderlyingTenor'] == '{}Y'.format(row[1]))]
        premium = float(picked['Payer'].values[0]) / 10000.0
        # one ulp, not the bit - see `premium_frame` for the 1e4 scaling
        assert abs(premium / float(priced) - 1.0) <= 2.0 ** -52, (
            '{}: the premium file is not carrying the premium this block priced ({} against {}), '
            'so nothing below is a statement about the solve'.format(
                name, premium.hex(), float(priced).hex()))

        # the round trip: the recovered vol re-strikes to where quoting sigma + delta lands
        landed = swap.price / direct['swaps'][name].price - 1.0
        assert abs(landed) < 1e-14, (
            '{}: a premium struck at sigma_N {:.6g} recovered a vol that re-strikes {:.3e} away '
            'from the premium a block quoting {:.6g} prices outright - the recorded readings are '
            '0.0 to 3.3e-16, one ulp'.format(name, sigma, landed, sigma + RESTRIKE_DELTA))
        # and the bump is the whole of the move: ATM Bachelier is linear in the vol
        ratio = swap.price / priced / ((sigma + RESTRIKE_DELTA) / sigma) - 1.0
        assert abs(ratio) < 1e-14, (
            '{}: the re-struck premium moved by {:.6g}x against the (sigma + delta)/sigma the '
            'linearity requires'.format(name, swap.price / priced))
        worst = max(worst, abs(landed))

        # the old bounds, reproduced in process: they refuse this fixture on the sign
        residual = old_bracket_residual(swap, struck, premium)
        low, high = residual(0.01), residual(sigma + 0.5)
        assert low > 0.0 and high > 0.0, (
            '{}: the OLD bracket ends read {:+.4e} and {:+.4e} - if they now straddle, this '
            'fixture no longer reaches the defect it was built for'.format(name, low, high))
        assert low > 1e-4, (
            '{}: f(0.01) is {:+.4e} under the old bounds - the recorded readings are 7.7e-4 to '
            '2.0e-2, so this is a SIGN failure and not a near miss'.format(name, low))
        with pytest.raises(ValueError, match='different signs'):
            scipy.optimize.brentq(residual, 0.01, sigma + 0.5)
    assert worst < 1e-14, (
        'the worst Normal re-strike lands {:.3e} from the outright quote against a recorded '
        '3.3e-16'.format(worst))

    # ------------------------------------------------------- the Lognormal arm, unmoved to the bit
    for vols, recorded in ((None, LOGNORMAL_RESTRUCK), (NORMAL_VOLS, LOGNORMAL_LOW_RESTRUCK)):
        extra = {} if vols is None else {
            'Instrument_Definitions': quoted_definitions(CHECKER_BENCHMARKS, vols)}
        flat = identified_closure(benchmarks=CHECKER_BENCHMARKS, batch_size=512,
                                  Objective='Analytic', **extra)
        bumped = identified_closure(benchmarks=CHECKER_BENCHMARKS, batch_size=512,
                                    Objective='Analytic', premiums=premium_frame(flat),
                                    delta=0.005, **extra)
        assert {n: float(s.price).hex() for n, s in bumped['swaps'].items()} == recorded, (
            'the LOGNORMAL re-struck premium moved off its recorded reading - the bracket landing '
            'was contracted to leave that arm alone and the clock landing moved it exactly once, '
            'by sqrt(365.25/365): {}'.format(
                {n: float(s.price).hex() for n, s in bumped['swaps'].items()}))


def test_a_normal_block_carries_the_quote_side_and_its_bachelier_derivative():
    """`Quote_Sensitivity` ON A NORMAL BLOCK, held as the lognormal one is - and the derivative it
    reports is a NUMBER this convention pins. theta* is BIT-IDENTICAL with the quote side on and
    off over two whole optimizer chains.

    THE DIAGONAL IS EXACTLY `-w`: `sigma_mkt = P sqrt(2pi/T0) / A` with `P = A q sqrt(T0/2pi)` is
    LINEAR in the quote and the two `T0`s are one year fraction, so `dr/dq` is -1.0 to 2.2e-16 at
    every benchmark, independent of expiry, curve and theta. It read `-w * OLD_CLOCK` before, which
    is what this asserts it is NOT. The lognormal diagonal runs -0.10235 to -0.08706, which is `F`
    times this: a lognormal vol point is a fraction of the forward rather than a rate.

    And it is EXACT under a difference quotient rather than second-order accurate - 4.0e-15 with no
    h-squared ladder, where the lognormal quote side needs one. Taken AROUND the splice.
    """
    off, _ = normal_calibration()
    on, world = normal_calibration(Quote_Sensitivity='Yes', Stationarity_Tol=ABSENT)
    assert len(on.quotes) == len(CHECKER_BENCHMARKS) and all(q.requires_grad for q in on.quotes)
    assert [float(v).hex() for v in off.solve().numpy()] == [
        float(v).hex() for v in on.solve().numpy()], (
        'theta* moved when the quote side was switched on for a Normal block')

    theta = flat_theta(on, NORMAL_FOUR_THETA)
    residual, jacobian, quote_jac = residual_pieces(on, theta)
    n = len(CHECKER_BENCHMARKS)
    assert unconnected_pairs(on, theta) == n * n - n, (
        'the separable residual has exactly one live quote per row on this convention too')
    assert np.array_equal(quote_jac.numpy(), np.diag(np.diag(quote_jac.numpy()))), 'dr/dq'
    for i in range(n):
        assert abs(float(quote_jac[i, i]) + 1.0) < 1e-14, (
            'benchmark {}: dr/dq is {:.17g} against the -w a Bachelier quote side owes once the '
            'premium and its inversion read one clock'.format(i, float(quote_jac[i, i])))
        assert abs(float(quote_jac[i, i]) + OLD_CLOCK) > 1e-4, (
            'benchmark {}: dr/dq is still carrying sqrt(T_365.25/T_curve)'.format(i))

    # one FD rung through a re-authored quote, linear here rather than h-squared
    for column, h in ((0, 0.5), (2, 0.05)):
        vols = list(NORMAL_VOLS)
        vols[column] = NORMAL_VOLS[column] + h
        moved, _ = normal_calibration(vols=vols, Quote_Sensitivity='Yes', Stationarity_Tol=ABSENT)
        quotient = (residual_pieces(moved, theta)[0].numpy() - residual.numpy()) / (h / 100.0)
        assert abs(quotient[column] / float(quote_jac[column, column]) - 1.0) < 1e-13, (
            'quote {} at h={:g}: the difference quotient reads {:.17g} against autograd\'s '
            '{:.17g}'.format(column, h, quotient[column], float(quote_jac[column, column])))
        assert np.array_equal(quotient[np.arange(n) != column],
                              np.zeros(n - 1)), 'a re-authored quote moved another benchmark'
        assert np.array_equal(residual_pieces(moved, theta)[1].numpy(), jacobian.numpy()), (
            'J moved when quote {} was re-authored - the cross term is not zero here'.format(column))


def test_a_declared_shift_reaches_the_premium_and_refuses_beside_a_normal_quote():
    """THE SECOND DEFECT: the schema declares `Shift` and the calibration read `Property_Aliases`,
    so a block authoring the field the schema offers calibrated at ZERO displacement while the DEAL
    path carried that same `Shift` into every swaption's `Volatility` dependency.

    The precedence is gated in all four corners, premiums compared as HEX:

        Property_Aliases 2%, no Shift      the legacy route            <- what a live file carries
        Shift 2%, no Property_Aliases      the declared route          BIT-IDENTICAL to it
        Shift 2% AND Property_Aliases 5%   the declared one wins       BIT-IDENTICAL to both above
        neither                            no displacement             and it MOVES the premium

    The first two agreeing to the bit is the units claim as well as the precedence one: both routes
    reach the strike through one division by 100 rather than a round trip.

    A ZERO SHIFT IS NOT AN INSTRUCTION - zero is the field's own default, so an authored zero
    cannot outrank a legacy alias, which is what keeps every existing file bit-identical. And a
    displacement beside `Normal` refuses BY NAME from either spelling.
    """
    premium = {}
    for label, declared_fields in (
            ('alias', {'Property_Aliases': [{'BlackScholesDisplacedShiftValue': 2.0}]}),
            ('declared', {'Shift': utils.Percent(2.0)}),
            ('both', {'Shift': utils.Percent(2.0),
                      'Property_Aliases': [{'BlackScholesDisplacedShiftValue': 5.0}]}),
            ('neither', {})):
        world = identified_closure(benchmarks=CHECKER_BENCHMARKS, batch_size=2048,
                                   Objective='Analytic',
                                   world=surface_world(**declared_fields))
        assert world['surface'].displacement == (0.02 if declared_fields else 0.0), label
        premium[label] = {name: swap.price for name, swap in world['swaps'].items()}
    for name in premium['alias']:
        assert premium['declared'][name].hex() == premium['alias'][name].hex(), (
            '{}: a declared Shift and the Property_Aliases legacy reach the strike differently - '
            '{:.17g} against {:.17g}'.format(
                name, premium['declared'][name], premium['alias'][name]))
        assert premium['both'][name].hex() == premium['declared'][name].hex(), (
            '{}: the undeclared alias outranked the declared Shift'.format(name))
        assert premium['neither'][name] != premium['declared'][name], (
            '{}: a 2% displacement moved nothing, so it is reaching no premium at all'.format(name))

    # the legacy stays the legacy: a zero Shift beside an alias reads the alias
    world = identified_closure(
        benchmarks=CHECKER_BENCHMARKS, batch_size=2048, Objective='Analytic',
        world=surface_world(Shift=utils.Percent(0),
                            Property_Aliases=[{'BlackScholesDisplacedShiftValue': 2.0}]))
    assert world['surface'].displacement == 0.02, 'an authored ZERO Shift shadowed a live alias'

    for spelling, fields in (
            ('Shift', {'Shift': utils.Percent(3.0)}),
            ("Property_Aliases' BlackScholesDisplacedShiftValue",
             {'Property_Aliases': [{'BlackScholesDisplacedShiftValue': 3.0}]})):
        with pytest.raises(Exception, match='no strike to displace') as refused:
            identified_closure(
                benchmarks=CHECKER_BENCHMARKS, batch_size=2048, Objective='Analytic',
                world=surface_world(Distribution_Type='Normal', **fields),
                Instrument_Definitions=quoted_definitions(CHECKER_BENCHMARKS, NORMAL_VOLS))
        assert spelling in str(refused.value), (
            'the refusal has to name which spelling carried the displacement: {}'.format(
                refused.value))
        assert 'Lognormal' in str(refused.value), 'and the remedy: {}'.format(refused.value)
    # and a Normal surface carrying the field at its own default is not a contradiction
    assert normal_closure(batch_size=2048)['surface'].displacement == 0.0


def test_a_quoted_zero_refuses_and_so_does_an_absent_one():
    """A zero `Market_Volatility` used to fall through to `vol_surface.ATM(tenor, expiry)`, so a row
    quoting zero calibrated against whatever the book's surface held, under the name of a quote
    nobody gave. The fallthrough is RETIRED rather than re-plumbed - no JSON in this repository
    carries such a row - so a zero refuses and so does an absent column, which used to be a
    `KeyError` inside the loop. Both refusals name the BENCHMARK, and an unpriceable
    `Distribution_Type` refuses naming the two it knows.
    """
    rows = quoted_definitions(CHECKER_BENCHMARKS, [20.0] * len(CHECKER_BENCHMARKS))
    rows[1]['Market_Volatility'] = utils.Percent(0.0)
    with pytest.raises(Exception, match='Market_Volatility is quoted ZERO') as refused:
        identified_closure(benchmarks=CHECKER_BENCHMARKS, Objective='Analytic',
                           batch_size=2048, Instrument_Definitions=rows)
    assert 'Swaption_2Y_5Y' in str(refused.value), refused.value
    assert 'drop the row' in str(refused.value), 'the remedy: {}'.format(refused.value)

    rows = quoted_definitions(CHECKER_BENCHMARKS, [20.0] * len(CHECKER_BENCHMARKS))
    del rows[2]['Market_Volatility']
    with pytest.raises(Exception, match='carries no Market_Volatility') as refused:
        identified_closure(benchmarks=CHECKER_BENCHMARKS, Objective='Analytic',
                           batch_size=2048, Instrument_Definitions=rows)
    assert 'Swaption_3Y_3Y' in str(refused.value), refused.value

    # and a distribution this family cannot price, refused where a desk can fix it
    with pytest.raises(Exception, match='not a convention this calibration prices') as refused:
        identified_closure(benchmarks=CHECKER_BENCHMARKS, Objective='Analytic', batch_size=2048,
                           world=surface_world(Distribution_Type='Bachelier'))
    assert 'Lognormal and Normal' in str(refused.value), refused.value


def quanto_world(correlation, zero=ID_ZERO):
    """The identified fixture's ZAR curve made FOREIGN: a USD base, a 15-20% FX vol curve on the
    ZAR/USD rate, and `correlation` between that rate and the ZAR short rate - the world the quanto
    defect was measured on, kept intact. `implied_process` reverses the sign of a correlation quoted
    against the base currency, so `Value: 0.4` reaches the implied object as -0.4.
    """
    factors = identified_world(zero)
    factors['GBMAssetPriceTSModelParameters.{}'.format(ID_CCY)] = {
        'Property_Aliases': None, 'Quanto_FX_Volatility': None, 'Quanto_FX_Correlation': 0.0,
        'Vol': utils.Curve([], [(0.0, 0.15), (1.0, 0.15), (3.0, 0.17), (5.0, 0.18),
                                (10.0, 0.20)])}
    factors['Correlation.FxRate.USD.{}/InterestRate.{}'.format(ID_CCY, ID_CCY)] = {
        'Value': correlation}
    return factors


def foreign_closure(correlation, **extra):
    """`identified_closure` on `quanto_world(correlation)` under a USD base, at 8192 paths.
    `Objective: 'Monte_Carlo'` declared: the quanto drift is installed by `precalculate` and read
    by the SIMULATION, so every reading in this section is off `world['loss']`.
    """
    return identified_closure(benchmarks=CHECKER_BENCHMARKS, world=quanto_world(correlation),
                              base_currency='USD', batch_size=8192, Objective='Monte_Carlo',
                              **extra)


def both_premiums(world):
    """`({benchmark: MC premium}, {benchmark: SP premium})` at the world's standing theta."""
    prices, _ = world['loss'](world['implied_var'])
    analytic = {name: as_float(world['process'].schrager_pelsser_swaption(
        swap.schedule.expiry, swap.schedule.pay_times, swap.schedule.accruals).premium)
        for name, swap in world['swaps'].items()}
    return {name: as_float(value) for name, value in prices.items()}, analytic


def max_KtT(process):
    """The largest quanto drift the process carries, over both factors and the whole grid."""
    return max(float(k.abs().max().detach()) for k in process.KtT)


def loss_and_gradient(objective, implied_var):
    """`({benchmark: premium}, the gradient)` - the scalar the basin adapter reduces, backwarded.
    The value alone is half a reading here: the quanto branch multiplies `K` by a correlation that
    can be exactly zero, so a premium can be bit-identical while the tape behind it is not.
    """
    for leaf in implied_var.values():
        leaf.grad = None
    prices, errors = objective.loss(implied_var)
    objective.reduce(torch.stack(list(errors.values()))).backward()
    return ({name: as_float(value) for name, value in prices.items()},
            torch.cat([leaf.grad for leaf in implied_var.values()]).detach().cpu().numpy())


def quanto_objective(world):
    """THE MUTATION: `(process, premiums, gradient)` for the same residual closure over a process
    handed the EMITTED implied object. Built out of the world's own parts through a public
    signature and nothing patched, so this IS the pre-fix objective rather than an imitation.
    """
    process = HullWhite2FactorImpliedInterestRateModel(
        world['curve'], {'Lambda_1': 0.0, 'Lambda_2': 0.0}, world['implied_obj'])
    implied_var, objective, _, _ = world['model'].calc_loss_on_ir_curve(
        {'instrument': world['block']}, BASE, world['time_grid'], process,
        world['implied_obj'], world['ir_factor'], world['surface'])
    for name, value in ID_THETA.items():
        implied_var[name].data = torch.tensor(value, dtype=DTYPE, device=DEVICE)
    return (process,) + loss_and_gradient(objective, implied_var)


def test_the_calibration_objective_is_measure_free_on_a_quanto_world():
    """CALIBRATE DOMESTICALLY, SIMULATE GLOBALLY. A market swaption premium on this curve is
    E^{Q_dom}[D_dom . payoff] - struck, deflated and quoted in the RATE currency - so the objective
    prices it on domestic-measure paths whatever the job's base currency, and Girsanov moves drifts
    and not quadratic variation, so the fitted parameters are the same numbers either way.
    `implied_process` builds the objective's process on a twin with the two FX inputs SUPPRESSED,
    and `precalculate` takes its base-currency branch: K is identically zero through the loss, its
    Jacobian and `honesty_reprice`.

    Forcing the FX/IR correlation to zero used to move the simulated premiums +5.96% to +12.42%.
    It now moves them by EXACTLY 0.0 - structural rather than lucky, the correlation reaching
    nothing the objective reads - and the surviving column is the old rho=0 one, which is the
    domestic price the market quoted.

    THE WORLD IS STILL QUANTO, which is the fixture-degeneracy half: the object `save_params` emits
    off carries the FX vol curve and an instantaneous correlation of -0.4, and
    `test_the_simulator_still_carries_the_quanto_drift` reads a non-zero K off it.
    """
    reading = {}
    for correlation in (0.4, 0.0):
        world = foreign_closure(correlation)
        mc, sp = both_premiums(world)
        reading[correlation] = (mc, sp, world, max_KtT(world['process']))

    (mc_on, sp_on, world_on, k_on), (mc_off, sp_off, _, k_off) = reading[0.4], reading[0.0]
    # the world is quanto on the side that emits, or the zeros below measure nothing
    assert world_on['implied_obj'].get_quanto_fx() is not None, (
        'the emitted implied object carries no quanto FX vol, so this world is not quanto at all')
    assert float(world_on['implied_obj'].get_instantaneous_correlation()) == -0.4, (
        'the instantaneous FX/IR correlation reached the emitted object as {} against the -0.4 '
        'this fixture declares'.format(world_on['implied_obj'].get_instantaneous_correlation()))
    # and the calibration's own process has neither input, which is the seam
    assert world_on['process'].implied.get_quanto_fx() is None, (
        'the objective\'s process still carries a quanto FX vol - `implied_process` is meant to '
        'build it on a suppressed twin, so the domestic-measure seam is not where it was')
    assert world_on['process'].implied.get_instantaneous_correlation() is None
    assert k_on == 0.0 and k_off == 0.0, (
        'the objective simulated with a quanto drift of {} at rho=0.4 and {} at rho=0 - both must '
        'be exactly zero, because the premium being repriced is a domestic one'.format(k_on, k_off))

    for name in mc_on:
        assert mc_on[name] == mc_off[name], (
            '{}: the SIMULATED premium moved {:+.4%} with the FX/IR correlation against a required '
            'exact 0.0 - the calibration objective is back under the base measure'.format(
                name, mc_on[name] / mc_off[name] - 1.0))
        assert sp_on[name] == sp_off[name], (
            '{}: the analytic price moved with the FX/IR correlation, which it must not - it reads '
            'J alone and J carries no quanto drift'.format(name))
    # the surviving column is the DOMESTIC one, recorded at rho=0
    for name, recorded in (('Swaption_1Y_1Y', 0.006280267807), ('Swaption_2Y_5Y', 0.036982804395),
                           ('Swaption_3Y_3Y', 0.027657448260),
                           ('Swaption_10Y_10Y', 0.058735951675)):
        assert abs(mc_on[name] / recorded - 1.0) < 1e-9, (
            '{}: {:.12f} against the recorded domestic {:.12f}'.format(name, mc_on[name], recorded))


def test_the_foreign_world_now_prices_bitwise_as_the_domestic_one_does():
    """THE COMPARISON THE DEFECT FORBADE: SP against the Monte Carlo on a quanto'd block. With both
    objectives in the domestic measure the FX inputs reach nothing either reads, so the foreign
    world IS the base-currency world - bitwise, on every route this file measures, which is a
    stronger claim than two numbers that agree.

    THREE TERMS AND NO FOURTH in the premium gap of -0.05% to -4.52%: the simulation's numeraire
    error (-0.35% to -1.32% here, the fixture's 1Y first curve node), SP's freezing bias (0.03 to
    2.73bp of normal vol), and the per-evaluation noise of a quasi-random sample. The 10.9 to
    24.4bp quanto term is gone and the leg-convention term was already exactly zero.
    """
    foreign = foreign_closure(0.4)
    domestic = identified_closure(benchmarks=CHECKER_BENCHMARKS, batch_size=8192,
                                  Objective='Monte_Carlo')
    readings = []
    for world in (foreign, domestic):
        world['loss'](world['implied_var'])
        legs = checker_legs(world)
        readings.append(checker_readings(world, legs, checker_mc(world, legs, 8, 8192)))
    got, want = readings

    assert foreign['implied_obj'].get_quanto_fx() is not None, (
        'the foreign world stopped being quanto, so this gate compares two domestic worlds')
    for name in want:
        for route in ('sp', 'mc', 'numeraire', 'sp_nvol', 'sim_nvol', 'variance'):
            assert got[name][route] == want[name][route], (
                '{} {}: the foreign world reads {!r} against the base-currency world\'s {!r} - '
                'with the measure fixed these are the same arithmetic on the same sample and any '
                'difference at all is a quanto term that survived'.format(
                    name, route, got[name][route], want[name][route]))
        # the gap that is left is the two biases this file already measures, not a third thing
        gap = got[name]['mc'] / got[name]['sp'] - 1.0
        assert abs(gap) < 0.055, (
            '{}: MC/SP - 1 is {:+.4%} against a recorded -0.05% to -4.52%, which is the numeraire '
            'error plus SP\'s freezing bias plus sample noise'.format(name, gap))
        assert abs(gap - got[name]['numeraire']) < 0.035, (
            '{}: the gap {:+.4%} is no longer within sample noise of the numeraire error {:+.4%}, '
            'so a third term has appeared in the decomposition'.format(
                name, gap, got[name]['numeraire']))


def test_a_base_currency_block_is_bit_identical_through_the_seam():
    """THE THING THIS FIX MUST NOT MOVE. On a base-currency curve the quanto correlation is already
    zero, so the split is a no-op - but "already zero" is TWO CODE PATHS. With no FX vol factor the
    implied object never had a quanto curve; with one present - the case this gate authors, because
    nobody would think to write it down - `implied_process` reads `C = -0.0` off an absent
    correlation, so the pre-fix assembly took the QUANTO branch and integrated `K` for nothing
    while the shipped one takes the base-currency branch. Same zero, two branches, two tapes.

    Both price all four benchmarks to the LAST BIT with an `array_equal` gradient over all 23
    parameters, and so does the plain base-currency world - which is the statement that no
    base-currency theta* in existence can move.
    """
    factors = quanto_world(0.0)
    factors.pop('Correlation.FxRate.USD.{}/InterestRate.{}'.format(ID_CCY, ID_CCY))
    with_fx = identified_closure(benchmarks=CHECKER_BENCHMARKS, world=factors, batch_size=8192,
                                 Objective='Monte_Carlo')
    plain = identified_closure(benchmarks=CHECKER_BENCHMARKS, batch_size=8192,
                               Objective='Monte_Carlo')

    # the fixture is the awkward one it claims to be: an FX vol curve present, and the correlation
    # that would have scaled K the absent-factor default with its sign reversed
    assert with_fx['implied_obj'].get_quanto_fx() is not None, (
        'no FX vol factor reached the implied object, so this gate is the plain base-currency '
        'world twice and the two branches it exists to compare are never both taken')
    assert float(with_fx['implied_obj'].get_instantaneous_correlation()) == 0.0
    assert plain['implied_obj'].get_quanto_fx() is None

    _, pre_mc, pre_grad = quanto_objective(with_fx)
    post_mc, post_grad = loss_and_gradient(with_fx['objective'], with_fx['implied_var'])
    base_mc, base_grad = loss_and_gradient(plain['objective'], plain['implied_var'])
    assert max_KtT(with_fx['process']) == 0.0

    for name in post_mc:
        assert pre_mc[name] == post_mc[name], (
            '{}: the un-suppressed objective prices {!r} against the shipped {!r} - a '
            'base-currency block is meant to be untouched by the domestic-measure split'.format(
                name, pre_mc[name], post_mc[name]))
        assert base_mc[name] == post_mc[name], (
            '{}: adding an FX vol factor to a base-currency book moved the premium from {!r} to '
            '{!r}'.format(name, base_mc[name], post_mc[name]))
    assert np.array_equal(pre_grad, post_grad), (
        'the gradient moved through the seam on a base-currency block: worst entry {:.3e} '
        'relative'.format(np.max(np.abs(pre_grad / post_grad - 1.0))))
    assert np.array_equal(base_grad, post_grad), (
        'adding an FX vol factor to a base-currency book moved the gradient: worst entry {:.3e} '
        'relative'.format(np.max(np.abs(base_grad / post_grad - 1.0))))
    assert post_grad.size == 23 and np.abs(post_grad).max() > 0.0, (
        'the gradient is {} entries with a maximum of {} - an all-zero or empty gradient would '
        'make both equalities above vacuous'.format(post_grad.size, np.abs(post_grad).max()))


def test_re_enabling_the_quanto_drift_in_the_objective_restores_the_recorded_table():
    """THE MUTATION: hand `calc_loss_on_ir_curve` a process built on the EMITTED implied object -
    which is what `implied_process` did before the split - and the objective is back under the base
    measure, so the invariance gate above goes red. Nothing patched: built out of the world's own
    parts through a public signature, so it cannot rot against a refactor of its subject.

    The four premiums move -0.39% to -4.54% with the correlation, and `KtT` reads 1.2058e-01 /
    1.8051e-02 over the two factors against exactly zero through the shipped seam. The column is
    one-signed and the SIGN is asserted, because a cancellation eating the mutation would show as a
    split column before it showed as a small one - at a theta* whose two emitted quanto
    correlations shared a sign this same column read +5.96% to +12.42%.

    THE FLOOR IS THE PREMIUM'S OWN MATERIALITY, not the reading: 0.1% is a hundred times the 1e-5
    the four rows are pinned to and is what a mark moves by, and the measured 0.39% clears it 3.9x.
    Sweeping `FX_VOL` would raise it - `K` carries one factor of sigma_FX and the kill is exactly
    linear in it - but restoring a 5% floor would take 13x the curve, and `quanto_world` is the
    world the defect was MEASURED on.
    """
    mutated = {}
    for correlation in (0.4, 0.0):
        process, prices, _ = quanto_objective(foreign_closure(correlation))
        mutated[correlation] = (prices, max_KtT(process))
    (mc_on, k_on), (mc_off, k_off) = mutated[0.4], mutated[0.0]

    assert k_off == 0.0, 'the rho=0 leg carries a drift of {}, so this pair is not a sweep'.format(
        k_off)
    assert abs(k_on / 1.205779823689404e-01 - 1.0) < 1e-6, (
        'the re-enabled quanto drift is {:.6e} against the recorded 1.2058e-01'.format(k_on))
    for name, recorded in (('Swaption_1Y_1Y', -0.0039149), ('Swaption_2Y_5Y', -0.0259157),
                           ('Swaption_3Y_3Y', -0.0258340), ('Swaption_10Y_10Y', -0.0453586)):
        moved = mc_on[name] / mc_off[name] - 1.0
        assert abs(moved - recorded) < 1e-5, (
            '{}: the mutated objective moves {:+.4%} with the correlation against the recorded '
            '{:+.4%} - if this has stopped reproducing, the mutation no longer reinstates the '
            'defect and the invariance gate above is unguarded'.format(name, moved, recorded))
    # anti-vacuity: the shipped seam moves by exactly 0.0 on these same two worlds, so there is no
    # noise floor here - only the size below which a reinstated defect stops being one
    moves = {n: mc_on[n] / mc_off[n] - 1.0 for n in mc_on}
    assert min(abs(v) for v in moves.values()) > 1e-3, (
        'the mutated objective\'s smallest move is {:.4%}, under the 0.1% a reinstated quanto drift '
        'has to be worth for this gate to be measuring a defect - the recorded reading is -0.3915% '
        'at 1Y x 1Y and the kill is linear in the FX vol curve if a bigger one is wanted'.format(
            min(abs(v) for v in moves.values())))
    assert len({v > 0.0 for v in moves.values()}) == 1, (
        'the mutation now moves the four premiums in DIFFERENT directions: {} - the two factors\' '
        'quanto corrections opposing is what shrank this kill 15x in 2026-09-02, and a column that '
        'has split sign is one where they have started cancelling benchmark by '
        'benchmark'.format({n: '{:+.4%}'.format(v) for n, v in moves.items()}))


def emitted_factor(correlation=0.4):
    """`(the world, the price-factor dict, the risk factor)` a foreign calibration writes back."""
    world = foreign_closure(correlation)
    price_factors = {}
    factor = world['model'].save_params(
        {name: np.asarray(value) for name, value in ID_THETA.items()},
        price_factors, world['implied_obj'], utils.check_rate_name(ID_BLOCK))
    return world, price_factors[utils.check_tuple_name(utils.Factor(
        'HullWhite2FactorModelParameters', utils.check_rate_name(ID_BLOCK)[1:]))], factor


def test_a_foreign_calibration_still_emits_the_quanto_factor():
    """THE OTHER HALF OF THE SPLIT: what the calibration does not simulate, it still WRITES. The
    fitted parameters are measure-invariant, so the quanto correction is assembled at SIMULATION
    time out of them and the block has to go on emitting the three fields a scenario run reads.

    Off the foreign world at theta* the written factor carries the FX vol curve unchanged and
    `Quanto_FX_Correlation_1/2` = -0.27457788626607527 / +0.16015750955565405, both off
    `get_quanto_correlation`, which reads the instantaneous correlation the suppressed twin does
    not have. The pair can split its sign - `C (s_i + rho s_j) / D` per factor - and theta* itself
    rides through untouched.
    """
    _, param, factor = emitted_factor()
    assert param['Quanto_FX_Volatility'] is not None, (
        'a foreign-curve calibration emitted no Quanto_FX_Volatility, so the scenario run has '
        'nothing to build its quanto drift out of')
    assert np.array_equal(param['Quanto_FX_Volatility'].array,
                          np.array([[0.0, 0.15], [1.0, 0.15], [3.0, 0.17],
                                    [5.0, 0.18], [10.0, 0.20]]))
    for name, recorded in (('Quanto_FX_Correlation_1', -0.27457788626607527),
                           ('Quanto_FX_Correlation_2', 0.16015750955565405)):
        assert param.get(name) is not None, (
            '{} is missing from the emitted factor'.format(name))
        assert abs(float(param[name]) / recorded - 1.0) < 1e-12, (
            '{} emitted as {!r} against the recorded {!r}'.format(name, param[name], recorded))
    # theta* rides through untouched, which is what the split exists to preserve
    assert float(param['Alpha_1']) == ID_THETA['Alpha_1'][0]
    assert np.array_equal(param['Sigma_2'].array[:, 1], np.array(ID_THETA['Sigma_2']))
    # the object handed back is the one a scenario run would construct off that dict
    assert factor.get_quanto_fx() is not None and factor.get_instantaneous_correlation() is None


def test_the_simulator_still_carries_the_quanto_drift():
    """THE SPLIT IS REAL AND NOT A GLOBAL ZERO: precalculate the SCENARIO way - the implied tensor
    carrying `Quanto_FX_Correlation_1/2` and `Quanto_FX_Volatility`, off the factor the gate above
    watched being written - and `KtT` comes back as 1.205779823689404e-01 / 1.805065169779931e-02.

    That is THE SAME DRIFT the mutation gate reinstates in the objective, to the bit: the fix as an
    equality rather than two inequalities. The drift did not shrink and did not go away; it moved
    to the only place a base-measure drift belongs. The calibration's own process carries none.
    """
    world, _, factor = emitted_factor()
    world['loss'](world['implied_var'])
    ir_factor, time_grid = world['ir_factor'], world['time_grid']
    index_keys = {'full': utils.Factor(ir_factor.type, ir_factor.name + ('full',)),
                  'reduced': utils.Factor(ir_factor.type, ir_factor.name + ('reduced',))}
    shared = RiskNeutralInterestRate_State(index_keys, 8192, DEVICE, DTYPE)
    shared.clear()
    process = HullWhite2FactorImpliedInterestRateModel(
        world['curve'], {'Lambda_1': 0.0, 'Lambda_2': 0.0}, factor)
    implied_tensor = {name: torch.tensor(np.atleast_1d(value), dtype=DTYPE, device=DEVICE)
                      for name, value in factor.current_value(include_quanto=True).items()}
    assert {'Quanto_FX_Correlation_1', 'Quanto_FX_Correlation_2',
            'Quanto_FX_Volatility'} <= set(implied_tensor), (
        'the emitted factor no longer publishes the three fields the scenario branch reads, so '
        'this gate would silently drive the calibration branch instead')
    process.precalculate(
        BASE, time_grid, torch.tensor(world['curve'].current_value(), dtype=DTYPE, device=DEVICE),
        shared, 0, implied_tensor=implied_tensor)

    scenario = [float(k.abs().max()) for k in process.KtT]
    for i, recorded in enumerate((1.205779823689404e-01, 1.805065169779931e-02)):
        assert abs(scenario[i] / recorded - 1.0) < 1e-9, (
            'factor {}: the scenario quanto drift reads {:.10e} against the recorded {:.10e} - the '
            'simulator is meant to be untouched by the calibration measure fix'.format(
                i + 1, scenario[i], recorded))
    # the calibration's own process, on the same world, carries none of it
    assert max_KtT(world['process']) == 0.0
