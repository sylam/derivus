"""What the Hull-White closed forms do as the reversion speed goes to zero, measured.

`hw_calc_H` divides by `a*a`, `hw_calc_IJK` by `a**3`, `hw_calc_B` by `a`, and the HW2F `AtT`
assembly divides one factor's `B` by the OTHER factor's reversion speed. Every one of those is a
REMOVABLE singularity - the numerator is O(a^k) assembled out of O(1) terms - so the failure mode is
not a raise. It is silent cancellation: the relative error of `IJK` runs like eps/a^3, the price
degrades quietly, and `params_ok` goes on saying True because it guards the Cholesky's positive
definiteness and not this.

WHAT THE OBJECTIVE WAS REWARDING, before the repair, on the two-benchmark world below with a sloped
sigma term structure - the 2Y5Y model price against the same price computed from the series:

    alpha_2   params_ok   pre-fix price   post-fix price   moved by   what the optimizer saw
    1e-2      True        0.026623990     0.026623990      2.0e-09    nothing wrong
    1e-3      True        0.027090076     0.027091648      5.8e-05    a benchmark 0.006% off
    1e-4      True        0.021458598     0.027139662      2.1e-01    a benchmark 21% off, TAKEN
    1e-6      False       nan             0.027144954      -          a step it had to reject
    1e-9      False       0.000000000     0.027145006      1.0e+00    a swaption worth nothing
    0.0       False       nan             0.027145006      -          the price factor's default

and down the OTHER singular locus, `alpha_1 + alpha_2 -> 0`, which is a hyperplane through the
middle of the admissible box rather than a corner of it - `J[i][j]` is taken at `alpha_i + alpha_j`,
so alpha_1 = 0.1 with alpha_2 = -0.1 is singular with neither reversion speed anywhere near zero:

    a_1 + a_2   params_ok   pre-fix price   post-fix price   moved by
    1e-4        True        0.034107605     0.034107728      3.6e-06
    1e-6        False       0.033787482     0.034116562      9.6e-03
    1e-8        False       0.035115769     0.034116650      2.9e-02
    1e-9        False       0.000000000     0.034116651      1.0e+00
    0.0         False       nan             0.034116651      -

REACHABILITY, measured in `test_the_basin_step_can_decay_but_never_cross` and read off the code in
`test_least_squares_can_cross_zero_outright`: basin hopping's step is MULTIPLICATIVE
(`bootstrappers.py:2845-2846`), so it preserves the sign of every reversion speed and can only decay
toward zero - from 0.1 over 50 steps the median floor is 0.061 and 2.8% of walks reach below 3e-2,
with a pure-decay bound of 1.9e-4. Crossing is done by the OPTIMIZERS: `scipy.least_squares` and
basin hopping's own inner L-BFGS-B both run under `alpha_bounds = (-0.5, 2.4)`. And zero itself is
reachable without solving at all, because `riskfactors.HullWhite2FactorModelParameters` declares
`Alpha_1` and `Alpha_2` with `default=0`.

THE THRESHOLDS ARE READINGS. `test_the_crossover_is_where_the_thresholds_say_it_is` sweeps |a| from
1e-8 to 1 over four sigma term structures inside `sigma_bounds` and puts each threshold where that
function's relative error crosses 1e-10. Pre-fix, elementwise relative, worst over the three SMOOTH
term structures with the jagged one kept in its own column:

    |a|      H         IJK       IJK jagged    B
    1e-08    1.0e+00   1.2e+09   2.7e+10       1.7e-06
    1e-06    2.9e-04   8.9e+02   2.7e+04       8.0e-09
    1e-04    5.6e-08   2.5e-03   5.3e-04       7.0e-11
    1e-03    6.7e-10   1.5e-06   4.3e-06       1.1e-11   <- B threshold 1e-3
    3e-03    4.4e-11   2.1e-08   3.2e-07       1.7e-12
    1e-02    2.2e-12   7.0e-10   4.3e-08       8.5e-13   <- H threshold 1e-2
    3e-02    1.4e-13   4.0e-11   1.7e-10       2.9e-13   <- IJK threshold 3e-2
    1e-01    4.1e-14   4.0e-12   1.3e-11                 <- every authored alpha is here

so B crosses at 1.3e-4 against a threshold of 1e-3, H at 2e-3 against 1e-2, and IJK at 1.5e-2
against 3e-2 - each threshold a rung or so above its crossing, and every one of them a clear step
below the 0.1 that anything authored carries. That is the compatibility contract: above its
threshold each function is BIT-IDENTICAL to the expression it always was, asserted rather than
assumed - at FIXED reversion speeds this repository carries, never at the thresholds themselves,
because an assertion guarded by the constant it exists to pin switches itself off instead of
failing. `test_the_thresholds_are_the_readings` holds the four constants directly, which is the
only thing a moved threshold cannot pass.

IJK's is the threshold with a decision in it, because its crossing moves with the SLOPE of the
sigma term structure - the cancelling term is `2 mi mj` against a result of order `a^3 vi vj dt` -
and a term structure whose adjacent knots differ by 5x crosses at 3.5e-2 rather than 1.5e-2. Its
column above is what is left on the table: 1.7e-10 at the threshold instead of 1e-10. Meeting it
would put the branch within a factor of two of 0.1, and bit-identity there is worth more.

The AtT cross term is the one division no series here repairs; `hw_alpha_floor` says what that
costs and `test_the_atT_cross_term_carries_its_number` measures it against a Simpson reference with
no cancellation in it at all. The 1 FACTOR model divides the same way and now takes the same floor:
its `AtT` was ALL NaN at `Alpha`'s own declared default of 0, behind a `B` the series had already
repaired, and `test_the_numpy_assembly_is_finite_at_the_fields_own_default` drives that assembly
rather than the piece.

THE ANALYTIC SWAPTION, second part of this file. `schrager_pelsser_swaption` is the same arrays read
a second way: `precalculate` already integrates `J` and now retains it, so the ATM premium is
assembled out of the tensors the calibration carries rather than out of a second spelling of the
integral. Four things have to be true before comparing it to anything means anything, and those are
what these gates hold:

    the loadings ARE dS/dx at t=0        against a finite difference of the swap rate in (x,y),
                                         rebuilt in numpy from the model's own discount factor
    the variance IS its own integral     against one Simpson of the whole integrand, which never
                                         factorises through J at all
    the premium IS Bachelier ATM         against `utils.bachelier_european_option`, the engine's
                                         own normal-vol option, at F == X
    the price is DIFFERENTIABLE          a gradient in each of alpha, sigma and rho, finite and
                                         matching a central difference through the same closure

and, because the read of `J` at the expiry is the one place a convention could hide, that the read
is BIT-EXACT at a grid node and the linear blend off one - see
`test_J_off_a_node_is_the_linear_blend_and_carries_its_number` for what interpolating costs.

THE CHECKER, third part of this file, and the measurement the build turned on: the analytic price
against the brute-force Monte Carlo at the CALIBRATED theta*, on the repository's own identified
25-quote fixture (recovered from `104bd08^`, where the suite that held it was culled;
`docs_src/developer/quote_sensitivities.md#the-identified-fixture` is its design record). Three
numbers came out of it and they are not the same number.

    SP's own error          -0.13 to +2.17 basis points of normal vol across the 5x5 grid, signed,
                            growing with expiry TIMES tenor. Under 0.21bp at every expiry for
                            tenors to 3Y; 2.17bp at the 10Y x 10Y corner. And it RIDES THE
                            CORRELATION: at the top of `corr_bounds` the same corner reads 3.80bp
                            against 2.09 at the solved rho, so the third axis roughly doubles it.
    the MC's own error      -0.35% to -1.61% on the premium, systematic and NOT statistical -
                            `E[D(0,T_0) A(T_0)] = A(0)` fails by that much. It does not move when
                            the scenario grid is refined tenfold and it collapses by two orders
                            when the CURVE is given short tenors, so it is the fixture's 1Y first
                            node read by a ten-day deflator.
    the MC's noise          0.51 to 1.81 basis points per evaluation at the objective's own 8192
                            paths, measured over 64 re-scramblings of the Sobol rule ON THE VOL
                            ITSELF - the premium estimator's noise is a different number, 0.61 to
                            0.86bp, because the scramble buys three to four on the first moment and
                            only 1.2 to 2.5 on the second.

so the verdict this file records is that SP sits inside one evaluation's Monte Carlo noise at 22 of
the 25 benchmarks - stepping outside at the three 10Y-tenor entries 3Y x 10Y, 5Y x 10Y and
10Y x 10Y, each by about a fifth of a standard deviation - and that on the fixture as authored MOST
of the SP-vs-MC PREMIUM difference is the simulation's rather than the approximation's: at
2Y x 5Y the premium gap is 0.67% while the vol gap is 0.05bp out of 193. Which
is to say the analytic price is the more accurate of the two over most of this grid. It is also not
faster: 0.040s for a 25-benchmark analytic pass against 0.199s for the Monte Carlo on CPU float64,
but 0.158s against 0.140s on the CUDA float32 a job actually runs, because twenty-five scalar calls
lose to one batched kernel. An analytic objective would be bought for its exactness and its
gradient, not for its wall clock, and batching it across the benchmark set is a build nobody has
done.

AND THE CHECKER FOUND A DEFECT IN THE THING IT WAS CHECKING, which is recorded rather than quietly
removed: `Var` carried a spurious `e^{-(alpha_k+alpha_l)T_0}` in front of `J`, worth -18.1% to
+13.5% of the normal vol at theta*. See `test_reinstating_the_prefactor_is_visible_against_the_
simulation` for the grid of it and `schrager_pelsser_swaption` for why no prefactor belongs.

Two more findings sit beside it, each with its own gate: the identified fixture's solved `Alpha_2`
is -0.0179, so the repository's own calibrated point DOES engage the part-one series branch; and
reading `J` off a node costs whatever the LOCAL grid step does, which is 0.0013 basis points across
ten days and up to 2.07 across the two-year steps past the last benchmark - the size of the
approximation error the whole checker exists to measure, which is why that read now warns and names
its step.

THE OBJECTIVE, fourth part of this file and what the checker's measurement licensed.
`HullWhite2FactorModelParameters` declares `Objective`: `Monte_Carlo` by default and bit-identical
to what this family always did, `Analytic` differencing the Schrager-Pelsser normal vols against the
market's - PLAIN, because the Monte Carlo residual is already a square and that is the pathology the
analytic path retires. The market side is a CLOSED-FORM Bachelier inversion of the same numpy
premium the simulation's residual uses, `P sqrt(2pi/T0) / A` at the money on SP's own annuity, so
every quoting convention rides in through the premium and no root find enters an objective
evaluation.

    the quartic, measured    on the identified 25-quote block, ||J'r|| at theta* is 8.24e3 under
                             the Monte Carlo objective (||r|| 45.97, down from 2.86e11 at the
                             seed) and 3.85e-6 under the analytic one (||r|| 2.26e-3, from
                             6.00e-2). Seven orders OUTSIDE the declared Stationarity_Tol default
                             of 1e-3, against three orders inside it.
    the two answers          agree in repriced vol space - 5.14bp rms against the market at
                             theta*_MC, 4.51 at theta*_An, 2.52 apart on the 25-quote grid - and
                             share nothing in theta space, which is rank deficiency rather than
                             disagreement. On the four-quote block each objective INTERPOLATES in
                             its own metric, and the 2.45bp rms between them there is the
                             simulation's numeraire error plus SP's freezing bias, adding at the
                             10Y x 10Y corner to the 4.13bp printed.
    the wall clock           four-quote chain over three seeds: 75.1s Monte Carlo against 13.4s
                             analytic on this box in float64, a factor of 5.6. 12x per evaluation
                             against 2.2x the evaluations. The 25-quote block solves analytically
                             in 836s over 1915 evaluations - 0.437s each, twelve times the
                             four-quote block's 0.036 for six times the benchmarks, because SP is
                             one scalar call per benchmark against one batched kernel.
    what it owes             Quote_Sensitivity on the analytic path REFUSES by name, and an
                             analytic solve ends by reporting what the engine's own Monte Carlo
                             makes of theta* - -4.64% at its worst benchmark on the 25-quote block.

Riding along with it: `num_batches` and `batch_size` were undeclared class constants and the first
bought NO paths when raised, because `t_Buffer` was cleared once before the batch loop and never
inside it. Both are declared now (`Simulations`, `Batches`) and the loop clears per iteration, so
(2048 x 4) and (8192 x 1) are the same estimate to one ulp where they used to be the 2048-path
answer twice over. Declaring them made the sample shape PER BLOCK, and `bootstrap` runs every curve
in `market_prices` through ONE bootstrapper, so the shape the closure divides by is captured as a
local: through `self` it would be block A's paths over block B's count, in a residual
`LeastSquaresSolve` differentiates after the loop has ended
(`test_a_second_block_does_not_rescale_the_first_blocks_residual` has the factor of four and the
||J'r|| it moves).

AND A FINDING TAKEN RATHER THAN PATCHED: the two objectives do NOT agree on a quanto'd currency.
Schrager-Pelsser prices the domestic swaption, which is correct; the simulation carries
`precalculate`'s quanto drift. On a ZAR curve under a USD base, forcing the FX/IR correlation to
zero moves the simulated premiums 5.96% to 12.42% - 10.9 to 24.4 basis points of normal vol, five
to eleven times the worst corner of the approximation it would be confused with - and the analytic
price not at all.
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
import torch

from derivus import riskfactors, utils
from derivus.bootstrappers import (HullWhite2FactorModelParameters,
                                   RiskNeutralInterestRate_State)
from derivus.config import ModelParams
from derivus.stochasticprocess import (HW_ALPHA_FLOOR, HW_ALPHA_SERIES_B, HW_ALPHA_SERIES_H,
                                       HW_ALPHA_SERIES_IJK, HullWhite1FactorInterestRateModel,
                                       hw_alpha_floor, hw_calc_B, hw_calc_H, hw_calc_IJK,
                                       integrate_piecewise_linear)

BASE = pd.Timestamp('2026-08-03')
CCY, CURVE = 'ZAR', 'ZAR-SWAP'
DEVICE = torch.device('cpu')
DTYPE = torch.float64

#: the vol knots `HullWhite2FactorModelParameters.implied_process` builds, in years
VOL_TENOR = np.array([0, 1, 3, 6, 12, 24, 48, 72, 96, 120]) / 12.0
#: four sigma term structures, every one inside `sigma_bounds = (1e-5, 0.09)`. The flat one is the
#: benign corner - with no slope the cancelling `mi mj` term is identically zero - and `humped` is
#: the adversarial one, because nothing in the objective penalises a jagged term structure.
SIGMA = {'flat': np.full(VOL_TENOR.size, 0.01),
         'sloped': np.linspace(0.004, 0.02, VOL_TENOR.size),
         'steep': np.linspace(1e-5, 0.09, VOL_TENOR.size),
         'humped': np.array([0.004, .02, .03, .025, .012, .008, .006, .02, .004, .03])}
#: a 5x5 swaption grid's expiries, which is the shape the identified fixture carries
TIME_GRID = np.array([0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
#: an interest rate factor's tenor grid, which is what B is read on
FACTOR_TENOR = np.array([1 / 365.0, .25, .5, 1., 2., 3., 5., 7., 10., 15., 20., 30.])
LADDER = [float(10.0 ** e) for e in np.arange(-8.0, 0.5, 0.5)]
#: the reference series is entire but not infinitely long: 40 terms is exact to the last bit for
#: |a * dt| <~ 1, which covers every rung this gate reads it on
REF_TERMS = 40


class Shared:
    """The one attribute `integrate_piecewise_linear` reaches for."""

    def __init__(self):
        self.t_PreCalc = {}


# ---------------------------------------------------------------- the references

def taylor_H(a, exp, terms=REF_TERMS):
    """int_t^{t+dt} e^{as}(v + m(s-t))ds as its power series in `a`, grouped by dt^k rather than by
    (a dt)^k - so it is an independent ordering of the same sum and not a copy of the branch."""
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
    """int_0^t e^{as} sigma_i(s) sigma_j(s) ds by Simpson on a dense grid - a reference that shares
    nothing with the closed form OR with the series, so agreement is evidence and not bookkeeping.
    Slow and only good to about 1e-10 (the sigma product kinks at every vol knot), which is exactly
    the accuracy this needs to be worth reading."""
    if t == 0.0:
        return 0.0
    s = np.linspace(0.0, t, n)
    f = np.exp(a * s) * np.interp(s, VOL_TENOR, sigma_i) * np.interp(s, VOL_TENOR, sigma_j)
    h = s[1] - s[0]
    return h / 3.0 * (f[0] + f[-1] + 4 * f[1:-1:2].sum() + 2 * f[2:-2:2].sum())


def simpson_cross(ai, aj, t, sigma_i, sigma_j, n=200001):
    """The AtT cross term `(e^{-ai t} I_ij - e^{-(ai+aj) t} J_ij) / aj` written so that NOTHING
    cancels: pull the difference inside the integral, where it is
    `(e^{aj t} - e^{aj s})/aj -> (t - s)` and perfectly well behaved at aj = 0."""
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


#: the three term structures a calibrated sigma plausibly has, and the jagged one kept apart from
#: them - `humped`'s adjacent knots differ by 5x, which multiplies IJK's cancelling term by ~370 and
#: moves its crossing by the cube root of that
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

    Every level is invented and none of it has to be real. What the gate reads off this world is
    whether the model prices FINITELY and CONTINUOUSLY as a reversion speed is driven to zero, which
    is a property of the arithmetic rather than of the market it is pointed at. The curve carries no
    zero tenor because `sim_curve` divides by the factor tenor.
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
    """Two benchmarks, deliberately with DIFFERENT fixed and floating frequencies on the first, so
    the leg-frequency branch of `create_market_swaps` is on the path this gate prices through."""
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
    """`(process, implied_var, loss)` - the swaption calibration's own residual closure, built the
    way `calc_loss` builds it and evaluated the way the optimizers evaluate it. Nothing is patched:
    the parameters are moved through `implied_var`, which is the seam the scipy adapters use.
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

    # a gate's path count is its own business, and it is now the BLOCK's business too - `Simulations`
    # is declared, so a gate that wants 1024 paths says so in the JSON rather than writing an
    # attribute the schema knows nothing about. This reads finiteness and continuity, neither of
    # which is a function of the sample size.
    instrument = {'Instrument_Definitions': instrument_definitions(), 'Swaption_Volatility': CURVE,
                  'Simulations': 1024}
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
    """The overlap check. Where the closed form is healthy the two must agree, or the reference is
    measuring something else and every number below it is worthless."""
    for a in (0.05, 0.1, 0.3, 1.0):
        assert worst(prefix_H, taylor_H, a, curves=ALL_CURVES) < 1e-12, a
        assert worst(prefix_IJK, taylor_IJK, a, curves=ALL_CURVES) < 1e-10, a
        # B is read to 30 years, so its series is asked for |a T| up to 30 and 40 terms is not
        # enough past |a| ~ 0.1 - the reference's own domain, not the closed form's
        if a <= 0.1:
            assert rel(prefix_B(torch.tensor(a, dtype=DTYPE), torch.tensor(FACTOR_TENOR)).numpy(),
                       taylor_B(a, FACTOR_TENOR)) < 1e-12, a


def test_the_series_agrees_with_a_reference_that_shares_nothing_with_it():
    """Simpson on a dense grid, against the repaired `hw_calc_IJK` in the branch it takes near zero.
    The Taylor reference could in principle be wrong in the same way as the branch; this cannot."""
    for a in (0.0, 1e-8, 1e-4):
        got = integral(hw_calc_IJK(torch.tensor(a, dtype=DTYPE), torch.exp),
                       SIGMA['sloped'], SIGMA['humped'])
        want = np.array([simpson_IJK(a, t, SIGMA['sloped'], SIGMA['humped']) for t in TIME_GRID])
        assert rel(got, want) < 1e-9, 'a={}: {:.3e}'.format(a, rel(got, want))


@pytest.mark.parametrize('a', LADDER)
def test_the_crossover_is_where_the_thresholds_say_it_is(a):
    """The ladder, pre-fix beside post-fix. The pre-fix column is asserted too, because a threshold
    justified by a defect nobody can still see is a taste rather than a reading.

    THE ONE PLACE 1e-10 IS NOT REACHED, and it carries its number: on the JAGGED sigma the closed
    form still reads 1.7e-10 at 3.2e-2, the rung immediately above IJK's threshold, because that
    curve's crossing is at 4e-2 rather than the 3e-3 a smooth term structure crosses at. Raising the
    threshold to meet it would put the branch within a factor of two of the 0.1 every authored
    reversion speed carries, and bit-identity there is worth more than the remaining 7e-11.
    """
    post = {'H': worst(hw_calc_H, taylor_H, a, curves=ALL_CURVES),
            'IJK': worst(hw_calc_IJK, taylor_IJK, a),
            'IJK cross': worst(hw_calc_IJK, taylor_IJK, a, cross=True)}
    if a * FACTOR_TENOR.max() < 3.0:
        # past |a T| ~ 3 the 40-term reference is the inaccurate one, not B - its own domain
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
    """The four constants, asserted to BE the numbers the ladder measured.

    Every other gate in this file reads a threshold rather than pinning it, so without this one a
    measured crossing can be moved to taste and nothing says so. It is not a tautology: it is the
    only assertion that fails when `HW_ALPHA_SERIES_IJK` moves from 3e-2 to 3.5e-2 - the largest
    value below `2|Alpha_2*| = 0.0357`, which is the only landmark holding that side - a mutation
    that otherwise passes the entire suite 78 green while every |a| in [3e-2, 3.5e-2) silently
    swaps the closed form for the series and the calibration's numbers move with it.
    """
    assert (HW_ALPHA_SERIES_H, HW_ALPHA_SERIES_IJK, HW_ALPHA_SERIES_B) == (1e-2, 3e-2, 1e-3), (
        'the thresholds are readings off `test_the_crossover_is_where_the_thresholds_say_it_is` - '
        'H crosses 1e-10 at 2e-3, IJK at 1.5e-2 and B at 1.3e-4 - so moving one is a re-measurement '
        'and this gate is where it gets recorded')
    assert HW_ALPHA_FLOOR == 1e-8, (
        'the floor is `test_the_atT_cross_term_carries_its_number`\'s reading: two orders above the '
        '1e-10 where the AtT quotient stops carrying information at all')


def test_bit_identity_at_the_reversion_speeds_this_repository_carries():
    """The compatibility contract, pinned at FIXED reversion speeds rather than at the thresholds.

    Above its own threshold each function is the expression it always was, to the bit - not close to
    it, and not to a tolerance. What this gate used to do was guard every assertion with
    `if abs(a) >= HW_ALPHA_SERIES_*`, which is the constant it exists to pin: move the threshold and
    every guard goes false and the gate SWITCHES ITSELF OFF instead of failing.

    So the alphas are read off what the REPOSITORY carries and never off a threshold: the seed
    `implied_process` builds, the identified fixture's solved pair with the sums and doubles `I` and
    `J` are integrated at, the analytic block's own asymmetric theta, and both ends of
    `alpha_bounds`. `ID_THETA` is the one spelling of theta*, so the solved ones are taken from it.

    ONE OF THEM IS DELIBERATELY EXCLUDED FROM THE IJK LIST. `Alpha_2* = -0.017851` is inside
    `HW_ALPHA_SERIES_IJK` and takes the series there, which is the finding
    `test_the_solved_fixture_engages_the_series_branch` records - so IJK is held bit-identical at
    `2|Alpha_2*| = 0.0357` and at the sum `0.1038` instead, both of which J is read at.
    """
    a1, a2 = ID_THETA['Alpha_1'][0], ID_THETA['Alpha_2'][0]
    authored = (0.1,) + tuple(SP_THETA['Alpha_1'] + SP_THETA['Alpha_2']) + (2.4, -0.5, -0.1)
    solved = (a1, a2, 2.0 * a1, 2.0 * a2, a1 + a2)
    tenor = torch.tensor(FACTOR_TENOR)
    for a in authored + solved:
        ta = torch.tensor(a, dtype=DTYPE)
        for name, sigma in SIGMA.items():
            assert np.array_equal(integral(hw_calc_H(ta, torch.exp), sigma),
                                  integral(prefix_H(ta, torch.exp), sigma)), ('H', a, name)
            if a != a2:
                assert np.array_equal(integral(hw_calc_IJK(ta, torch.exp), sigma, sigma),
                                      integral(prefix_IJK(ta, torch.exp), sigma, sigma)), (
                    'IJK', a, name)
        assert torch.equal(hw_calc_B(ta, tenor), prefix_B(ta, tenor)), ('B', a)
        assert torch.equal(hw_alpha_floor(ta), ta), ('floor', a)
    # and the excluded one IS the series, so the skip above is an exclusion and not an oversight
    ta = torch.tensor(a2, dtype=DTYPE)
    assert not np.array_equal(integral(hw_calc_IJK(ta, torch.exp), SIGMA['humped'], SIGMA['humped']),
                              integral(prefix_IJK(ta, torch.exp), SIGMA['humped'], SIGMA['humped']))


def test_the_branch_never_engages_on_an_authored_reversion_speed():
    """Nothing in this repository authors an HW2F alpha except the seed `implied_process` builds
    when the price factor is absent, and that is 0.1 on both factors - a clear step above every
    threshold. A fixture that DID engage the branch would be a finding, so this asserts rather
    than assumes it."""
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
    even where the forward is clean. The calibration differentiates in the reversion speed, so this
    is the half of the repair a price gate cannot see."""
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
    """`HullWhite1FactorInterestRateModel` precalculated on the authored world at one `Alpha`.

    The numpy leg's OWN assembly and not a second spelling of it: what the gate below reads is what
    `precalculate` left on the process, so an `AtT` full of NaN fails there rather than three layers
    downstream in a simulated curve nobody could attribute it from.
    """
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
    """A FINITE `B` IS NOT A FINITE `A`, and the gate above certifies the piece and not the
    assembly - so this one drives `precalculate` itself.

    `HullWhite1FactorInterestRateModel` declares `F('Alpha', 'Float', default=0)`, and its
    `AtT = (B/alpha) e^{-at} (2I(t) - (e^{-at}+e^{-aT})J(t))` divides `B` by the reversion speed a
    SECOND time. At alpha == 0 the series makes `B` exactly the tenor, which reads as repaired,
    while the bracket is identically zero - `I` and `J` are the same integral there - and `B/alpha`
    is inf, so `0 * inf` put a NaN through the whole curve at a value the price factor reaches by
    omitting the field. It is the structural twin of the HW2F cross term at `precalculate`'s
    `second_part`/`third_part`, and it now takes the same floor.

    THE NUMBERS, on the world below - a sloped sigma, tenors to 30Y, `AtT` at the last grid node
    and the 30Y tenor. PRE-FIX it was all NaN at Alpha = 0 exactly and merely noisy beneath the
    floor: 2.75387786 at 1e-9 against 2.75389087 at 1e-12 and 2.75391689 at -1e-12, a 9.5e-6
    relative spread across a sign change worth 1e-12, which is the 1/alpha amplification
    `hw_alpha_floor` exists for. POST-FIX everything under the floor is evaluated AT the floor:
    2.753877041 at +1e-8 against 2.753878968 at -1e-8, so the pair straddle their own alpha -> 0
    limit by 7.0e-7 relative and the value at the declared default sits 3.5e-7 below it. That is
    the whole cost of the repair, and it is paid against a NaN.
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
    """The division `hw_alpha_floor` guards rather than repairs, against a reference with no
    cancellation in it. This is a LIMITATION with a measurement attached, and the measurement is
    the point: the quotient is a first-order difference of two integrals taken SEPARATELY, so its
    error is whatever those carry - the closed form's own 1.3e-11 at alpha_i = 0.1, not machine
    epsilon - amplified by 1/alpha_j. It is what puts `HW_ALPHA_FLOOR` where it is, two orders
    above the alpha at which the quotient stops carrying information at all.
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


def test_the_basin_step_can_decay_but_never_cross():
    """`bootstrappers.py:2842-2849`, run as written. The step is MULTIPLICATIVE, so a reversion
    speed keeps its sign for the whole search however long it runs - decay toward zero, never a
    crossing of it."""
    rng = np.random.RandomState(5120)
    floors = []
    for _ in range(2000):
        alpha, floor = np.array([0.1, 0.1]), 1.0
        for _ in range(50):  # niter=50, as `SwaptionCalibration.solve` calls it
            alpha = (alpha * np.exp(rng.uniform(-0.125, 0.125, 2))).clip(-0.5, 2.4)
            floor = min(floor, alpha.min())
        floors.append(floor)
    floors = np.array(floors)
    assert (floors > 0.0).all(), 'a multiplicative step cannot change sign'
    assert floors.min() < 0.05, 'but it decays: worst floor reached {:.3g}'.format(floors.min())
    # the bound if every draw went the same way, which is what the threshold has to survive
    assert 0.1 * np.exp(-0.125 * 50) < HW_ALPHA_SERIES_B


def test_least_squares_can_cross_zero_outright():
    """The bounds the optimizers actually run under. Basin hopping's step cannot cross zero but its
    inner L-BFGS-B and the `least_squares` that follows it both take these, and both straddle it."""
    model = HullWhite2FactorModelParameters({}, DEVICE, DTYPE)
    low, high = model.alpha_bounds
    assert low < 0.0 < high, model.alpha_bounds
    # and the price factor reaches zero without any solve at all
    alpha_field = next(f for f in riskfactors.HullWhite2FactorModelParameters.fields
                       if f.name == 'Alpha_1')
    assert alpha_field.default == 0


def test_params_ok_does_not_guard_this(calibration):
    """`process.params_ok` is the Cholesky's positive definiteness and the acceptance test reads it
    as if it meant the parameters are well posed. It is True at a reversion speed that used to move
    the benchmark 21%, which is why the repair could not be left to it."""
    ok, prices = price_at(calibration, 0.1, 1e-4, SIGMA['sloped'])
    assert ok, 'params_ok is True here - that is the point'
    assert np.isfinite(prices).all()


@pytest.mark.parametrize('alpha_2', [1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-9, 0.0, -1e-9, -1e-4])
def test_the_objective_prices_through_zero(calibration, alpha_2):
    """A calibration whose reversion speed is driven into the small region prices FINITELY and
    SANELY - the swaption is worth something, it is worth less than the annuity, and it sits where
    the neighbouring rung says it should rather than collapsing to zero or to nan."""
    ok, prices = price_at(calibration, 0.1, alpha_2, SIGMA['sloped'])
    _, limit = price_at(calibration, 0.1, 1e-2, SIGMA['sloped'])
    assert ok, 'params_ok went False at alpha_2 = {:g}'.format(alpha_2)
    assert np.isfinite(prices).all(), 'alpha_2 = {:g} priced {}'.format(alpha_2, prices)
    assert (prices > 0.0).all(), 'alpha_2 = {:g} priced {}'.format(alpha_2, prices)
    # the limit as alpha -> 0 is a real one, so no rung may sit far from the 1e-2 rung
    assert (np.abs(prices / limit - 1.0) < 0.05).all(), (
        'alpha_2 = {:g} priced {} against {} at 1e-2'.format(alpha_2, prices, limit))


@pytest.mark.parametrize('delta', [1e-2, 1e-4, 1e-6, 1e-8, 0.0])
def test_the_objective_prices_through_the_alpha_sum_locus(calibration, delta):
    """The other singular locus, and the one with no small reversion speed anywhere in it: `J` is
    read at `alpha_i + alpha_j`, so alpha_1 = 0.1 against alpha_2 = -0.1 is singular with both
    factors at a perfectly ordinary 0.1. This is a hyperplane through the middle of the admissible
    box and `least_squares` walks across it."""
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


#: One ATM payer swaption's fixed leg, authored in the CURVE's own year fractions - ACT_365, which
#: is the clock `read_cache` builds `time_grid_years` on and therefore the clock `J` is integrated
#: against. 2Y into 5Y, semi-annual. The expiry is a NODE of the calibration's grid because the
#: block's own 2Y benchmark put it there, which is the normal case and the one worth reading first.
SP_EXPIRY = 2.0
SP_PAY = np.arange(2.5, 7.01, 0.5)
SP_TAU = np.full(SP_PAY.size, 0.5)
#: and one that is NOT a node: the grid is 10-daily out to the last expiry, so 1.5 years falls
#: between 540 and 550 days and the read has to interpolate
SP_OFF_NODE = 1.5

#: The parameter set the analytic gates read at. Deliberately asymmetric in EVERY coordinate - two
#: different reversion speeds, two different sigma term structures, a correlation that is neither 0
#: nor +-1 - so a formula that swapped factor 1 for factor 2, dropped a cross term or lost a sign
#: could not pass here by coincidence.
SP_THETA = {'Alpha_1': [0.06], 'Alpha_2': [0.35], 'Correlation': [-0.6],
            'Sigma_1': SIGMA['sloped'], 'Sigma_2': SIGMA['humped']}


def sp_at(calibration, expiry=SP_EXPIRY, **override):
    """`(process, HW2FSwaption)` at one parameter set, through the closure the optimizers call.

    The leaves are moved with `.data`, which is the seam the scipy adapters move them through, and
    `loss` is what runs `precalculate` - so this gate drives the model the way the calibration does
    rather than through a second setup of its own. The Monte Carlo prices `loss` returns are
    discarded: what these gates read is what precalculate LEFT on the process.
    """
    process, implied_var, loss, _ = calibration
    for key, value in dict(SP_THETA, **override).items():
        implied_var[key].data = torch.tensor(np.atleast_1d(value), dtype=DTYPE)
    loss(implied_var)
    return process, process.schrager_pelsser_swaption(expiry, SP_PAY, SP_TAU)


def swap_rate_at_state(process, x, expiry=SP_EXPIRY):
    """The forward swap rate as an explicit function of the t=0 state, in numpy.

    At t=0 the convexity term $A(t,T)$ is identically zero, so the model's own discount factor is
    just $P(0,T)e^{-\\sum_k B_k(T)x_k}$ and the swap rate is what it always is. This shares nothing
    with the loading formula - no B, no annuity weight, no q - so `dS/dx` off it is evidence about
    the loadings rather than a second copy of them.
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
    dW_k(s)$ - see `schrager_pelsser_swaption` for why the $B_k(t)$ terms cancel out of the time-$t$
    loading and leave that scaling exactly - so the covariance is
    $\\rho_{kl}\\int e^{(\\alpha_k+\\alpha_l)s}\\sigma_k\\sigma_l ds$ written out rather than read
    off $J_{kl}$. Nothing here touches `J`, the closed forms, or their series, so this is the
    reading that says the variance is the variance and not just a consistent piece of bookkeeping.
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
    """`q_k` is what the approximation FREEZES, so if it is not `dS/dx_k` at t=0 the whole price is
    a different quantity. Central-differenced against a swap rate rebuilt from the model's own
    discount factor, one state coordinate at a time - and the annuity and the forward rate the same
    expression hands back are checked against the same rebuild at x = 0, because a q that is right
    off an annuity that is wrong would still price wrong."""
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
    """`Var = sum_kl rho_kl q_k q_l J_kl(T0)` against Simpson on the covariance integrand itself.

    The two routes share the loadings and NOTHING else - not `J`, not the closed forms, not the
    series - so this is what says the correlation matrix is in the right place and that NO
    `e^{-(a_k+a_l)T0}` prefactor belongs in front of `J`: the Simpson carries none either, because
    `q` is the loading on the scaled martingale whose covariance `J` already is. The formula is
    written here WITHOUT the prefactor deliberately - `test_reinstating_the_prefactor_is_visible_
    against_the_simulation` is the defect's own record, and a reader reconciling the code to a
    docstring that still carried it is exactly how it would come back."""
    process, sp = sp_at(calibration)
    want = simpson_sp_variance(process, [as_float(x) for x in sp.loadings])
    got = as_float(sp.variance)
    assert abs(got / want - 1.0) < 1e-9, '{:.12g} against Simpson {:.12g}'.format(got, want)
    # and the correlation is load-bearing: the same q at rho = 0 is a materially different variance
    _, uncorrelated = sp_at(calibration, Correlation=[0.0])
    assert abs(as_float(uncorrelated.variance) / got - 1.0) > 0.05


def test_the_premium_is_the_engines_own_bachelier_at_the_money(calibration):
    """`A(0) sqrt(Var/2pi)` is the ATM Bachelier written out, so it is held against
    `utils.bachelier_european_option` - what the engine's own normal-vol pricers value an option
    with - at F == X rather than against a second opinion of it. The closed form is spelled here and
    not delegated because that function clamps its vol at 1e-5 for a batched pricing path, and a
    reference price should not carry a floor the thing it references does not have.

    The normal vol is the other half of the same identity: it is what a desk reads and what a
    vols-against-vols objective would difference, so `Var == vol^2 T_0` is asserted rather than
    left implied."""
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
    """The half that decides whether this could ever BE the objective. The premium has to carry a
    finite gradient back to every calibrated parameter off the same tensors the Monte Carlo residual
    uses - so the leaf differentiated here is `implied_var`'s, the one the scipy adapters write.

    A term structure is bumped whole and differenced against `grad.sum()`, which is the directional
    derivative along ones - one central difference rather than ten, and it still fails if any knot's
    gradient is wrong, because the sum would not match."""
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
    """The expiry is a node whenever the calibration asks - `TimeGrid` is built out of the benchmark
    starts - so the ordinary read must be the node itself and not an interpolation that happens to
    land on it. Rebuilt off `process.J` at the node index and held BIT-identical, not to a
    tolerance."""
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
    """Off a node the read is LINEAR in `t` between the two neighbours, which is
    `utils.interpolate_tensor` - `piecewise_linear`'s own read of a curve on its own grid - and the
    docstring says so. Two things are held here: that it is exactly that blend, and what the blend
    costs against the exact integral.

    THE NUMBER, on the 10-day stretch of the calibration grid where a 1.5 year expiry lands: the
    variance comes out 3.0e-5 relative ABOVE the Simpson of the same integrand - the combination of
    `J`s this variance reads is convex in `t` over that step and a chord lies above a convex
    function - which is 1.5e-5 on the premium and 0.0013 basis points of normal vol at an 85.1bp
    level. FOUR ORDERS below the 0.7 basis points of Monte Carlo standard error the objective's own
    8192 paths carry, which is the whole reason interpolating is acceptable rather than a refusal.

    IT IS THE STEP AND NOT THE HORIZON that sets it: 3.3e-5 at 1.253 years, 3.4e-5 at 1.9, and
    1.6e-7 at 2.001 where the grid happens to carry a one-day step.

    AND THE STEP IS NOT ALWAYS TEN DAYS, which is the half this gate used to leave unnumbered. The
    same grid runs 10-daily to the last benchmark expiry and then jumps to the vol knots -
    2.0027 -> 4.0027 -> 6.0055 -> 8.0055 -> 10.0055 on the two-benchmark world - so an expiry in one
    of THOSE gaps is a chord across two years:

        expiry     step      variance     normal vol      level
        1.50      0.0274     +3.0e-05      +0.0013bp      85.13
        2.50      2.0000     +6.0e-02      +2.0711bp      71.88
        3.00      2.0000     +6.3e-02      +1.9250bp      64.12
        3.50      2.0000     +3.8e-02      +1.0312bp      55.36
        4.50      2.0027     +4.8e-02      +0.9064bp      38.96
        5.00      2.0027     +5.4e-02      +0.7907bp      30.54

    THREE ORDERS above the ten-day reading and the SIZE OF THE THING THE CHECKER MEASURES - SP's own
    approximation error is 2.17bp at the identified fixture's worst corner - so an expiry in the gap
    would carry an interpolation error as large as the approximation error and nothing downstream
    could tell the two apart. That is why the read now WARNS, naming the step it is crossing, and
    why the remedy in the message is to put the expiry on the grid rather than to trust the chord.

    THE SIGN IS THE LOCAL CURVATURE'S AND NOT A LAW: at 0.75 years the same read is -4.7e-5, so the
    combination is concave over that step. What the assertion below holds is the sign AT 1.5, where
    it has been measured; a sign change THERE would mean the read has stopped being the blend.
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

    # and the gaps past the last benchmark, where the same read is worth basis points rather than
    # ten-thousandths of one - the numbers in the table above, held to 2% because nothing here is
    # simulated. The warning has to fire at every one of them or the chord is silent again.
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
    """Two refusals, each naming the thing and the remedy. Reading `J` past the grid would flat-
    extrapolate silently - `utils.interpolate_tensor` clips - and that is exactly the quiet-garbage
    failure the rest of this file exists to remove."""
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

#: The repo's identified HW2F fixture, recovered from `104bd08^` where the swaption suite that held
#: it was culled: a humped ZAR-shaped zero curve, a swaption cube that rises with expiry and falls
#: with the underlying tenor, and a 5x5 expiry x tenor grid of 25 benchmarks against 23 parameters
#: quoted FLAT at 20 vol. `docs_src/developer/quote_sensitivities.md#the-identified-fixture` is the
#: design record and says why both of those choices are forced.
ID_CCY, ID_CURVE, ID_VOL = 'ZAR', 'ZAR-JIBAR-3M', 'ZAR_SWAPTION'
ID_BLOCK = 'HullWhite2FactorModelPrices.' + ID_CURVE
ID_ZERO = ((1.0, 0.0800), (2.0, 0.0835), (3.0, 0.0880), (5.0, 0.0915), (7.0, 0.0950), (10.0, 0.0905))
ID_SURFACE_E, ID_SURFACE_T, ID_MONEY = (0.25, 1.0, 2.0, 5.0), (1.0, 2.0, 5.0, 10.0), (-0.01, 0.0, 0.01)
ID_STATIONARITY, ID_SEED = 1e5, 5120

#: theta* AS SOLVED, so a gate reads the calibrated point without paying 531 seconds for it. The
#: chain is `SwaptionCalibration.solve` at the block's own `Random_Seed`, float32 on CUDA, which is
#: what `construct_bootstrapper` hands a job; `test_the_recorded_theta_is_the_one_the_chain_returns`
#: says what re-deriving it costs and `test_the_solved_fixture_engages_the_series_branch` reads the
#: finding off it.
ID_THETA = {
    'Alpha_1': [0.12163561971138252],
    'Alpha_2': [-0.017850818439802296],
    'Correlation': [-0.004643081221729517],
    'Sigma_1': [2.554538896307349e-03, 1.004817319102585e-03, 1.343422277830541e-03,
                1.076457309536077e-05, 1.478665862232447e-02, 1.380606275051832e-02,
                2.833980973809958e-02, 2.485187910497189e-02, 2.855546073988080e-03,
                5.384350661188364e-03],
    'Sigma_2': [1.873115003108978e-02, 2.103322558104992e-02, 1.769018359482288e-02,
                1.503448002040386e-02, 1.554366014897823e-02, 1.599598675966263e-02,
                2.692321315407753e-03, 1.005611754953861e-02, 2.024488896131515e-02,
                1.966073922812939e-02]}
#: the benchmark rows the checker prices: the identified grid's own corners, plus two whose FIXED
#: and FLOATING frequencies DIFFER, which is the pair that settles the single-curve question
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
    """A full `SwaptionCalibration` on the identified fixture - the OPTIMIZER CHAIN included.

    `identified_closure` stops at the residual, which is what most of this file reads. This goes the
    one step further, through `calc_loss`, so the gates that solve drive exactly what `bootstrap`
    drives: the same two adapters, the same seeded generator, the same acceptance rule. The
    parameters are left AT THE SEED, because `calc_loss` captures `x0` off them.
    """
    from derivus.bootstrappers import SwaptionCalibration
    world = identified_closure(benchmarks=benchmarks, theta={}, chain=True, **extra)
    return SwaptionCalibration('gate', world['objective'], world['implied_var'],
                               world['optimizers'], world['process'], world['swaps']), world


def identified_closure(benchmarks=CHECKER_BENCHMARKS, zero=ID_ZERO, batch_size=8192,
                       theta=None, device=DEVICE, dtype=DTYPE, world=None, chain=False, **extra):
    """The calibration's own residual closure on the identified fixture, standing at `theta`.

    Built the way `RiskNeutralInterestRateModel.bootstrap` builds it and nothing patched: the
    parameters are moved through `implied_var`, which is the seam the two scipy adapters use, and
    every knob the block carries - `Simulations`, `Batches`, `Objective` - is DECLARED and set in
    the JSON rather than written onto the bootstrapper as an attribute.
    """
    from derivus import bootstrappers
    factors, interp = (world or identified_world(zero)), ModelParams()
    boot = HullWhite2FactorModelParameters({}, device, dtype)
    rate = utils.check_rate_name(ID_BLOCK)
    ir_factor = utils.Factor('InterestRate', rate[1:])
    surface = riskfactors.construct_factor(
        utils.Factor('InterestYieldVol', (ID_VOL,)), factors, interp)
    surface.delta = 0.0
    ir_curve = riskfactors.construct_factor(ir_factor, factors, interp)
    surface.set_premiums(None, ir_curve.get_currency())
    implied_obj, process, vol_tenors = boot.implied_process(
        extra.pop('base_currency', ID_CCY), factors, {}, ir_curve, rate)
    block = {'Swaption_Volatility': ID_VOL, 'Generate_Instruments': 'No', 'Random_Seed': ID_SEED,
             'Stationarity_Tol': ID_STATIONARITY, 'Quote_Sensitivity': 'No',
             'Simulations': batch_size, 'Batches': 1,
             'Instrument_Definitions': identified_definitions(benchmarks)}
    block.update(extra)
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
    fixed schedule of its own, so this rebuilds it with the same two generators and holds the
    rebuild against that column - a leg the analytic price reads that is not the leg the simulation
    pays would fail here rather than downstream as a disagreement nobody could attribute.

    The clock is the CURVE's, ACT_365, and not `utils.DAYS_IN_YEAR`: `create_market_swaps` measures
    its own expiry with 365.25 while `read_cache` builds `time_grid_years` with the day count, so an
    expiry taken from the first would miss the grid node it is supposed to land on by 7e-4 years.
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

    THE HARNESS HOLDS ITS OWN BATCH because it reads per-batch means rather than one pooled one -
    `mean_se` needs the batches apart. Its `t_Buffer.clear()` is the same line the library's own
    batch loop now carries: when this was written the library cleared ONCE before the loop and every
    gather after the first read the FIRST batch's cached curve straight back out, so `Batches` cost
    N times the wall clock and bought nothing. That is fixed in `calc_loss_on_ir_curve`, and
    `test_batches_now_buy_paths_and_shrink_the_estimates_spread` is the reading of the repair.
    `t_PreCalc` is left alone on both sides: it holds `precalculate`'s own integrals and is a
    function of theta rather than of the sample.

    The routes, all off the SAME paths so every difference between them is a common-random-number
    one and carries a far smaller error bar than a difference of two independent means would:

        A    the engine       E[ D(0,T0) relu( pv_float_cashflow_list ) ] - what the objective prices
        B    the convention   E[ D(0,T0) relu( 1 - P(T0,Tn) - sum c_i P(T0,Ti) ) ]
        W    the numeraire    E[ D(0,T0) A(T0) ], which is A(0) identically
        WS, WS2               the same weight against S(T0) and S(T0)^2, so the swap rate's own
                              annuity-measure mean and variance come out of the same simulation.
                              `A(T0) S(T0)` is `1 - P(T0,Tn)` identically, so neither needs the swap
                              rate divided back out and multiplied in again.
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
    """`(mean, standard error)` off the batch means. The Sobol sample is drawn ONCE for the whole
    run and split into consecutive blocks, so this is a batch-mean error bar on a scrambled
    quasi-random sample and not an i.i.d. one - reported as such, never as a confidence interval."""
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
    """`(world, legs, readings)` at the recorded theta*: ONE simulation for every gate below."""
    world = identified_closure()
    world['loss'](world['implied_var'])
    leg_map = checker_legs(world)
    return world, leg_map, checker_readings(
        world, leg_map, checker_mc(world, leg_map, 8, 8192))


def test_the_single_curve_float_leg_is_exact_and_not_an_approximation(checker):
    """QUESTION ONE, and it settles with an equality rather than a tolerance.

    Schrager-Pelsser prices the swap as `1 - sum c_i P(T_0,T_i)`; the Monte Carlo prices the actual
    `pv_float_cashflow_list`. The benchmark carries `'Forward': curve_index, 'Discount':
    curve_index` - the SAME curve - and `create_market_swaps` builds each float coupon with its pay
    day ON its accrual end day and its `Year_Frac` equal to its reset's own accrual, so a coupon
    contributes `P(t,T_i) (P(t,t_i)/P(t,T_i) - 1) = P(t,t_i) - P(t,T_i)` and the leg TELESCOPES to
    `P(T_0,T_0) - P(T_0,T_n) = 1 - P(T_0,T_n)` path by path. Not approximately: identically, and at
    every state, because the forward rate and the discount factor are read off one curve at the same
    two points.

    THE FREQUENCIES THEREFORE DO NOT MATTER, which is the half worth measuring rather than arguing.
    A 3M float against a 12M fixed leaves the float leg telescoping to the same two bonds and the
    fixed leg is `c_i` whatever its schedule, so this gate prices 3M/3M, 3M/6M and 3M/12M and holds
    all three. Measured on 524288 paths across the whole 5x5 grid and on the mixed-frequency and
    steep-curve worlds: |A - B| is at most 5.6e-18 against premiums of 0.006 to 0.06, which is
    float64 round-off on the sum and not a difference.

    So there is NO leg-convention component in the SP-vs-MC gap on this benchmark, and every basis
    point of it below belongs to the approximation or to the simulation.

    WHAT WOULD BREAK IT is a second curve. `create_market_swaps` has no basis-curve path - both
    fields are the same `curve_index` - so a tenor-basis benchmark would need building before this
    question could be asked again, and the answer would then be no.
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
    """QUESTION TWO, and the answer that decides the build: what SP's freezing costs, in the units
    a desk reads, against the swap rate the simulation actually produces.

    The comparison is VOL AGAINST VOL and deliberately not premium against premium, because a
    premium comparison also carries the simulation's own numeraire error - see
    `test_the_monte_carlo_carries_a_bias_of_its_own`, which is the larger of the two over most of
    this grid. `sim_nvol` is the annuity-measure standard deviation of `S(T_0)` off the same paths,
    `E[D A S^2]/E[D A] - (E[D A S]/E[D A])^2`, and both of those ratios divide the numeraire error
    out of themselves.

    SP MINUS SIMULATED, in basis points of normal vol, on the identified 5x5 grid at theta* over
    1048576 paths - the whole measurement, because a one-point agreement is not one - with the
    standard error of each entry beneath it:

        expiry \\ tenor      1Y      2Y      3Y      5Y     10Y
        1Y               -0.113  -0.050  -0.030  +0.004  +0.248
        2Y               -0.127  -0.093  -0.044  +0.050  +0.307
        3Y               -0.026  +0.012  +0.055  +0.461  +0.898
        5Y               +0.047  +0.090  +0.106  +0.213  +0.876
        10Y              +0.016  +0.094  +0.208  +0.557  +2.173

        s.e.              0.044   0.044   0.044   0.045   0.049   (1Y)
                          0.067   0.068   0.069   0.071   0.076   (3Y)
                          0.083   0.088   0.094   0.107   0.146   (10Y)

    against a LEVEL of 178 to 209 basis points. The bias is SIGNED - SP reads low at the short end
    and high at the long one, crossing zero along the 3Y expiry - and it grows with the PRODUCT of
    expiry and tenor rather than with either alone, which is what freezing a loading over a longer
    horizon on a longer swap does. Held to a tenor of 3Y it is under 0.21bp at EVERY expiry out to
    ten years; it is the 10Y tenor column that carries all of it.

    AND WHAT ONE EVALUATION OF THE OBJECTIVE CARRIES, so the two are on one scale. The noise is read
    off THE SAME STATISTIC the grid above compares: `sim_nvol`, recomputed from scratch on each of 64
    independent re-scramblings of the Sobol rule at the engine's own single 8192-path evaluation. A
    randomised-QMC reading, unbiased whatever the sequence's internal correlation, and the reason the
    batch-mean error bar `mean_se` returns is NOT what this is read off.

        s.d.(sim_nvol)      1Y      2Y      3Y      5Y     10Y
        1Y                0.512   0.514   0.518   0.528   0.581
        2Y                0.677   0.678   0.681   0.690   0.759
        3Y                0.681   0.677   0.675   0.679   0.723
        5Y                0.641   0.638   0.644   0.660   0.725
        10Y               1.236   1.279   1.327   1.438   1.812

        bias / s.d.         1Y      2Y      3Y      5Y     10Y
        1Y                -0.22   -0.10   -0.06   +0.01   +0.43
        2Y                -0.19   -0.14   -0.07   +0.07   +0.40
        3Y                -0.04   +0.02   +0.08   +0.68   +1.24
        5Y                +0.07   +0.14   +0.16   +0.32   +1.21
        10Y               +0.01   +0.07   +0.16   +0.39   +1.20

    THE PREMIUM'S NOISE IS NOT THE VOL'S, and this column used to be the premium's - the estimator's
    relative s.d. multiplied by the vol level, against a bias that is a difference of standard
    DEVIATIONS. The scramble buys most of its variance reduction on the FIRST moment, which is what
    makes the two different numbers rather than the same one rescaled. Over the same 64 draws, iid
    instead of scrambled, all in basis points:

        benchmark        premium s.d. x level         s.d. of sim_nvol
                        Sobol     iid    ratio     Sobol     iid    ratio
        1Y x 1Y         0.665   2.638      4.0     0.512   1.275      2.5
        3Y x 3Y         0.826   2.438      3.0     0.675   1.462      2.2
        2Y x 5Y         0.739   2.505      3.4     0.690   1.526      2.2
        5Y x 10Y        0.612   1.955      3.2     0.725   1.546      2.1
        10Y x 10Y       0.627   1.890      3.0     1.812   2.180      1.2

    so the scramble is worth three to four in the premium's standard deviation and only 1.2 to 2.5 in
    the vol's - LEAST at the 10Y x 10Y corner, which is exactly where the verdict is decided. The
    premium's noise there, 0.627bp, is 2.89x too small to stand in for the vol's 1.812, and that is
    how this column came to read 3.04 standard deviations where the right denominator gives 1.20. The
    error was conservative for the build decision - SP is better than it was reported to be - and the
    correction is recorded here rather than quietly applied.

    THE VERDICT THIS FILE RECORDS. SP sits INSIDE one evaluation's Monte Carlo noise at 22 of the 25
    benchmarks. It steps outside at three - 3Y x 10Y, 5Y x 10Y and 10Y x 10Y - and at all three by
    about a fifth of a standard deviation, because the noise grows with EXPIRY where the bias grows
    with expiry times tenor. So the approximation is good enough to BE the objective on a grid held
    inside a 10Y tenor; on the 10Y tenor column it is marginal rather than wrong, and that column is
    where a Gauss-Hermite quadrature over the annuity measure would have to earn its keep.

    THE READING IS AT THIS FIXTURE'S CORRELATION, which is -0.0046 and where the cross term is worth
    a quarter of a percent of the variance. It roughly doubles at the top of `corr_bounds` - see
    `test_the_bias_rides_the_correlation_axis`, which is the sweep of that axis and the reason this
    grid is not the whole measurement.
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
    # reference at all rather than a number off an unidentified measure
    for name, r in readings.items():
        assert abs(r['drift']) < 5e-3, '{}: E^A[S]/K - 1 is {:.3e}'.format(name, r['drift'])


#: The correlation sweep's own world: a short benchmark, the mixed-frequency one, and both
#: 10Y-tenor corners. It carries its OWN `TimeGrid` - starts at 1, 2, 5 and 10 years - and therefore
#: its own Sobol sample, so its numbers are not the 25-benchmark grid's to a basis point.
RHO_BENCHMARKS = ((1, 1, 3, 3), (2, 5, 3, 6), (5, 10, 3, 3), (10, 10, 3, 3))
#: the ends of `HullWhite2FactorModelParameters.corr_bounds`
RHO_ENDS = (-0.95, 0.95)


@pytest.fixture(scope='module')
def rho_sweep():
    """`{rho: (readings, params_ok)}` at the two ends of `corr_bounds`, off ONE world and one leg map.

    `Correlation` moves through `implied_var`, which is the seam the scipy adapters write and the one
    `sp_at` already moves it through, so both routes see the same parameter: `checker_mc` runs
    `precalculate` against it and `schrager_pelsser_swaption` reads what that left behind. Its own
    world rather than the checker's, because a module fixture whose parameters a gate mutated would
    hand every gate after it a different theta.
    """
    world = identified_closure(benchmarks=RHO_BENCHMARKS)
    world['loss'](world['implied_var'])
    leg_map = checker_legs(world)
    out = {}
    for rho in RHO_ENDS:
        world['implied_var']['Correlation'].data = torch.tensor([rho], dtype=DTYPE)
        out[rho] = (checker_readings(world, leg_map, checker_mc(world, leg_map, 8, 8192)),
                    world['process'].params_ok)
    return out


def test_the_bias_rides_the_correlation_axis(rho_sweep):
    """THE THIRD AXIS, and the one the expiry x tenor grid cannot see.

    `rho` enters this price in exactly ONE place - the cross term `2 rho q_1 q_2 J_12` - and the
    identified fixture solves it to -0.0046, where that term is worth -0.18% to -0.33% of the total
    variance. So every number in the grid above is taken at a point where the only piece of the
    variance that couples the two FROZEN loadings contributes almost nothing, which is where a
    two-factor freezing approximation is least stressed. `corr_bounds` is (-0.95, 0.95).

    SP MINUS SIMULATED across that range, on this world at 1048576 paths, with the level beside it
    because rho moves the level itself by a factor of 1.9:

        rho          1Y x 1Y          2Y x 5Y         5Y x 10Y        10Y x 10Y
        -0.95    146.39  -0.117   129.17  +0.000   127.04  +0.258   145.32  +1.059
        -0.60    161.00  -0.113   158.21  +0.017   149.40  +0.435   160.33  +1.459
        -0.0046  183.19  -0.111   198.06  +0.053   181.20  +0.845   183.05  +2.094
         0.00    183.35  -0.111   198.34  +0.053   181.43  +0.848   183.21  +2.099
        +0.60    203.26  -0.109   231.61  +0.109   208.60  +1.391   203.54  +3.022
        +0.95    214.02  -0.108   248.98  +0.146   222.92  +1.746   214.51  +3.795

    MONOTONE IN RHO, and it is the 10Y-tenor column that carries it: the corner runs 1.06 -> 3.80
    basis points end to end, so the fixture's own -0.0046 sits at 55% of the worst admissible value.
    5Y x 10Y goes 0.26 -> 1.75 the same way. The short benchmark does not move at all - -0.117 to
    -0.108 across the whole range - because on a one-year swap the two loadings barely differ and
    the cross term has nothing to couple.

    AGAINST THE NOISE, measured the way the grid above measures it - 64 re-scramblings of one
    8192-path evaluation, on this world:

        s.d.(sim_nvol)   1Y x 1Y   2Y x 5Y   5Y x 10Y   10Y x 10Y
        rho = -0.95        0.452     0.400      0.440       0.778
        rho = -0.0046      0.660     0.805      0.960       2.040
        rho = +0.95        0.425     0.665      1.145       2.064

    so the corner reads 1.36 s.d. at -0.95, 1.03 at the solved rho and 1.84 at +0.95, and 5Y x 10Y
    reads 0.59, 0.88 and 1.53. THE VERDICT HOLDS ACROSS THE AXIS rather than only at a point - SP is
    marginal on the 10Y tenor column and inside the noise everywhere else - but the margin at the
    top of `corr_bounds` is half again what it is at the solved point, and that is where a
    Gauss-Hermite quadrature would first pay for itself.

    AND THE SIMULATION'S OWN ERROR RIDES THE SAME AXIS, which is why the martingale bound below is
    1.2e-2 rather than the 5e-3 the solved rho meets. `E[D A]/A(0) - 1` and `E^A[S]/K - 1` at the
    10Y x 10Y corner, on this world at 65536 paths:

        rho          numeraire      drift
        -0.95        -8.4e-03   -1.9e-03
        -0.0046      -1.7e-02   +3.2e-03
        +0.95        -2.4e-02   +7.7e-03

    Both grow with the LEVEL rho puts on the model - the fixture's curve starts at a 1Y tenor and
    the deflator reads a ten-day rate off it, which `test_the_monte_carlo_carries_a_bias_of_its_own`
    measures - so the +0.95 reading is the least clean of the three from BOTH sides, and the
    doubling above is the honest shape of it rather than a number to quote to three figures.

    This gate reads the two ENDS at 65536 paths, which is what it can afford; the tables above are
    the same harness at sixteen times the sample.
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
    assert 0.5 < gap(low, 'Swaption_10Y_10Y') < 2.2, (
        '10Y x 10Y at rho = -0.95 reads {:+.3f}bp against a recorded 1.3'.format(
            gap(low, 'Swaption_10Y_10Y')))
    assert 3.0 < gap(high, 'Swaption_10Y_10Y') < 6.0, (
        '10Y x 10Y at rho = +0.95 reads {:+.3f}bp against a recorded 4.5 - the whole point of this '
        'sweep is that the corner is worse than the solved rho says'.format(
            gap(high, 'Swaption_10Y_10Y')))
    # and it GROWS with rho on the 10Y tenor, which is the finding rather than the level
    for name in ('Swaption_10Y_10Y', 'Swaption_5Y_10Y'):
        assert gap(high, name) > 2.0 * gap(low, name), (
            '{}: {:+.3f}bp at +0.95 against {:+.3f} at -0.95'.format(
                name, gap(high, name), gap(low, name)))
    # while the short benchmark is flat in it, so the growth belongs to the tenor and not to rho
    assert abs(gap(high, 'Swaption_1Y_1Y') - gap(low, 'Swaption_1Y_1Y')) < 0.1
    # the swap rate is still a martingale under its own annuity measure at both ends, to the
    # tolerance the SIMULATION allows there rather than the 5e-3 the solved rho meets - see the
    # docstring's last table for why the bound has to widen with rho and not with the sample
    for readings in (low, high):
        for name, r in readings.items():
            assert abs(r['drift']) < 1.2e-2, '{}: E^A[S]/K - 1 is {:.3e}'.format(name, r['drift'])


def test_the_monte_carlo_carries_a_bias_of_its_own(checker):
    """`E[D(0,T_0) A(T_0)] = A(0)` is an identity, so what this reads is the SIMULATION'S error and
    not the approximation's - and on this fixture it is the larger of the two.

        expiry \\ tenor      1Y      2Y      3Y      5Y     10Y
        1Y                -0.35%  -0.32%  -0.31%  -0.31%  -0.32%
        2Y                -0.86%  -0.85%  -0.83%  -0.83%  -0.95%
        3Y                -0.82%  -0.80%  -0.80%  -0.62%  -0.70%
        5Y                -1.33%  -1.32%  -1.34%  -1.38%  -1.35%
        10Y               -1.56%  -1.56%  -1.57%  -1.58%  -1.61%

    IT IS NOT DISCRETISATION, which is what it looks like and is the first thing to rule out. Refine
    the scenario grid from ten-daily to daily - 375 nodes to 3654 - and it does not move: -3.48e-3
    to -3.56e-3 at 1Y x 1Y and -1.69e-2 to -1.67e-2 at 10Y x 10Y. A rollover numeraire's own O(dt)
    error would have fallen by ten.

    IT IS THE CURVE'S TENOR GRID. This fixture's zero curve carries its first node at ONE YEAR, and
    `reduce_deflate` asks the simulated curve for a ten-day rate - flat-extrapolated off a 1Y node,
    with `B_k(T-t)` evaluated there too. Add short tenors to the same curve, re-read at the same
    theta*, and it collapses:

        curve tenor grid        nodes    1Y x 1Y     5Y x 5Y   10Y x 10Y
        1Y .. 10Y (this one)        6  -3.48e-03   -1.38e-02   -1.64e-02
        + 3M, 6M                    8  -4.76e-04   -3.72e-03   -4.55e-03
        + 1D, 1M, 3M, 6M           10  -3.11e-05   -8.76e-04   -1.08e-03
        + 1D, 1W, 1M, 3M, 6M, 9M   12  -2.24e-05   -8.63e-04   -1.00e-03

    so two of the three orders are the short end alone, and what is left at 1e-3 is the interpolated
    curve between the remaining tenor nodes. The consequence for the decision is direct: on the
    fixture as authored, MOST of the SP-vs-MC premium difference is the Monte Carlo's, not SP's -
    at 2Y x 5Y the premium gap is 0.67% while the vol gap is 0.05 basis points out of 193.
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
    `q` is the loading on the SCALED martingale `Y_k = e^{alpha_k t} x_k` - see
    `schrager_pelsser_swaption` for the cancellation that makes the time-t loading exactly
    `e^{alpha_k t} q_k` - and `J` is `Y`'s own covariance, so a prefactor scales the state twice.

    WHAT IT COST, at theta* on the identified grid, in basis points of normal vol:

        expiry \\ tenor      1Y      2Y      3Y      5Y     10Y
        1Y                +1.06   +1.31   +1.53   +1.88   +2.49
        2Y                -4.01   -2.91   -1.95   -0.41   +2.13
        3Y               -10.63   -8.62   -6.87   -4.04   +0.62
        5Y               -37.90  -33.33  -29.44  -22.74  -11.36
        10Y              +10.59  +13.11  +15.31  +18.90  +24.74

    -18.1% to +13.5% of the level, and NOT SIGNED, because `Alpha_2` solves negative on this
    fixture and `e^{-2 alpha_2 T_0}` is then above one. Hundreds of Monte Carlo standard errors
    wide at half a million paths, so nothing about the earlier agreement was borderline: it was
    never checked against a simulation at all.
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
        # the mutation reproduces what the formula used to be, so this is the same number twice
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
    """THE DEFECT AND ITS REPAIR, in one gate, because the repair is only meaningful beside it.

    `num_batches` and `batch_size` were UNDECLARED class constants on `RiskNeutralInterestRateModel`
    (1 and 8192), and the first did not do what raising it looks like it does.
    `calc_loss_on_ir_curve` cleared `t_Buffer` ONCE, in the `shared_mem.reset` before its batch loop,
    and never again inside it; `calc_time_grid_curve_rate` keys its cache on the curve code and the
    time grid rather than on the batch, so batch 1 onwards gathered batch 0's simulated curve
    straight back out of the buffer. The prices came out BIT-IDENTICAL at 1, 4 and 8 batches and
    the cost was linear in the batch count for no paths at all.

    Both are now DECLARED - `Simulations` and `Batches` - and the loop clears `t_Buffer` (and NOT
    `t_PreCalc`, which holds `precalculate`'s integrals and is a function of theta rather than of
    the sample) at the top of each iteration. Three things are held here, and the first is the one
    that would fail if the clear went back outside the loop.

    THE IDENTITY, which is the decisive half and needs no tolerance to speak of. `reset` draws
    `Simulations x Batches` Sobol points ONCE and reshapes them into blocks, so `Batches` is a
    BLOCKING of one sample and not a second sample: (2048 x 4) and (8192 x 1) are the same 8192
    points in the same order, summed in a different grouping. They agree to ONE ULP - 2.2e-16
    relative, and 0.0 on the second benchmark - which is a statement about the whole loop that no
    error bar can be fudged into. Before the repair (2048 x 4) was the 2048-path answer, four
    times over.

        shape            1Y x 1Y             5Y x 5Y
        8192 x 1      0.00634279527552446  0.0427816484939556
        4096 x 2      same to 2.2e-16      same to 0.0
        2048 x 4      same to 2.2e-16      same to 0.0
        1024 x 8      same to 2.2e-16      same to 1.1e-16

    THE ESTIMATE MOVES with the sample, which is what it could not do before. At 2048 paths a
    batch, the pooled premium at 1 / 2 / 4 / 8 batches:

        benchmark          1 batch       2 batches     4 batches     8 batches
        1Y x 1Y          0.006323968   0.006340595   0.006342795   0.006374788
        5Y x 5Y          0.042794526   0.042843707   0.042781648   0.042638971

    so a four-batch reading is 30 basis points of relative value away from the one-batch one on the
    short benchmark, where before it was zero to the last bit.

    WHAT IT BUYS is one batch's own scatter divided down, and the honest version of that is not a
    root-B law: on sixteen batches of 2048, the batch-mean standard deviation is 6.21e-5 on
    1Y x 1Y and grouping them in twos brings it to 4.32e-5 (a factor of 1.44 against root-two's
    1.41) - but grouping in fours brings it only to 4.25e-5. Consecutive blocks of a SCRAMBLED
    SOBOL rule are not independent draws, so the first halving behaves and the next does not, and
    the estimator's error bar is a batch-mean one rather than a confidence interval. That is the
    same caveat `mean_se` carries, and it is why the gate below holds the identity and the move
    rather than a variance law.
    """
    estimate = {}
    for batches in (1, 2, 4, 8):
        world = identified_closure(benchmarks=((1, 1, 3, 3), (5, 5, 3, 3)), batch_size=2048,
                                   Batches=batches)
        prices, _ = world['loss'](world['implied_var'])
        estimate[batches] = {k: as_float(v) for k, v in prices.items()}

    for name in estimate[1]:
        moved = abs(estimate[4][name] / estimate[1][name] - 1.0)
        assert moved > 1e-4, (
            '{} at Batches=4 is {:.12g} against {:.12g} at 1 - identical to that many figures means '
            'the buffer is being cleared once outside the loop again and the batches are re-pricing '
            'batch zero'.format(name, estimate[4][name], estimate[1][name]))

    # the identity: the same 8192 points, four ways of blocking them
    whole = identified_closure(benchmarks=((1, 1, 3, 3), (5, 5, 3, 3)), batch_size=8192)
    pooled, _ = whole['loss'](whole['implied_var'])
    pooled = {k: as_float(v) for k, v in pooled.items()}
    for sims, batches in ((4096, 2), (2048, 4), (1024, 8)):
        split = identified_closure(benchmarks=((1, 1, 3, 3), (5, 5, 3, 3)), batch_size=sims,
                                   Batches=batches)
        prices, _ = split['loss'](split['implied_var'])
        for name, value in prices.items():
            assert abs(as_float(value) / pooled[name] - 1.0) < 1e-14, (
                '{} at {} x {} is {:.17g} against {:.17g} at 8192 x 1 - those are the SAME Sobol '
                'points, so a difference means the batch loop is not walking all of them'.format(
                    name, sims, batches, as_float(value), pooled[name]))


def test_the_declared_sample_shape_is_the_shape_the_engine_uses():
    """`Simulations` and `Batches` are DECLARED, so a block omitting them and a block declaring the
    declaration's own values have to be the same job.

    The schema-emission suite holds the AST half of this - that the `.get` fallback and the `F`
    default are one number. What it cannot see is whether the number reaches the sample, so this
    reads the shape off the state the closure built and off the residual it returns.
    """
    for field, value in (('Simulations', 8192), ('Batches', 1), ('Objective', 'Monte_Carlo')):
        f = next(x for x in HullWhite2FactorModelParameters.fields if x.name == field)
        assert f.default == value, (field, f.default)

    # a block that declares the declaration's own values, and one that omits all three
    declared = identified_closure(benchmarks=((1, 1, 3, 3),), batch_size=8192)
    assert (declared['model'].batch_size, declared['model'].num_batches) == (8192, 1)
    world = identified_closure(benchmarks=((1, 1, 3, 3),), batch_size=8192)
    for key in ('Simulations', 'Batches', 'Objective'):
        world['block'].pop(key, None)
    implied_var, objective, _, _ = world['model'].calc_loss_on_ir_curve(
        {'instrument': world['block']}, BASE, world['time_grid'], world['process'],
        world['implied_obj'], world['ir_factor'], world['surface'])
    assert (world['model'].batch_size, world['model'].num_batches) == (8192, 1), (
        'a block omitting Simulations and Batches did not fall back to the declared 8192 and 1')
    assert objective.reprice is None, 'the Objective absent has to BE the Monte Carlo objective'
    # and the same residual, to the bit, as the block that spelled all three out
    for name, value in (ID_THETA.items()):
        implied_var[name].data = torch.tensor(value, dtype=DTYPE, device=DEVICE)
    absent = objective.loss(implied_var)[1]
    spelled = declared['loss'](declared['implied_var'])[1]
    for name in absent:
        assert as_float(absent[name]) == as_float(spelled[name]), (
            '{}: the field absent reads {:.17g} against {:.17g} spelled out'.format(
                name, as_float(absent[name]), as_float(spelled[name])))


def test_the_solved_fixture_engages_the_series_branch():
    """A FIXTURE THAT DOES ENGAGE THE BRANCH, which the compatibility contract said to report
    rather than absorb.

    `test_the_branch_never_engages_on_an_authored_reversion_speed` reads the SEED, and the seed is
    0.1 on both factors. `theta*` is not the seed: on the identified 25-quote fixture the chain
    solves `Alpha_2` to -0.017851 - negative, and inside `HW_ALPHA_SERIES_IJK`. `I[i][j]` is
    integrated at `alpha_i` alone, so `I[1][0]` and `I[1][1]` take the SERIES branch at the
    calibrated point of the repository's own identified fixture.

    That is the repair earning its place rather than a problem: pre-fix, `hw_calc_IJK` at 1.8e-2 on
    a jagged sigma reads about 1e-7 relative, which is small - the point is that the optimizer
    walked there on its own, over a bound that straddles zero, and nothing on the path would have
    told anyone if it had walked two orders further.
    """
    alpha_1, alpha_2 = ID_THETA['Alpha_1'][0], ID_THETA['Alpha_2'][0]
    assert alpha_2 < 0.0, 'the solved Alpha_2 is negative, which is the reachability claim'
    assert abs(alpha_2) < HW_ALPHA_SERIES_IJK, (
        'Alpha_2* is {:.6g} against the IJK threshold {:g}'.format(alpha_2, HW_ALPHA_SERIES_IJK))
    # H and B are more forgiving and their thresholds are lower, so those branches stay closed
    assert abs(alpha_2) > HW_ALPHA_SERIES_H and abs(alpha_2) > HW_ALPHA_SERIES_B
    # J is taken at the SUM, and the sum is nowhere near either threshold
    assert abs(alpha_1 + alpha_2) > HW_ALPHA_SERIES_IJK
    assert abs(2.0 * alpha_2) > HW_ALPHA_SERIES_IJK


def test_the_recorded_theta_is_a_stationary_point_of_the_recorded_world(checker):
    """`ID_THETA` is a reading and this is what makes it one: the world it was solved in still
    prices its benchmarks where the solve left them. Not a re-solve - that costs 531 seconds and
    `docs_src/developer/quote_sensitivities.md` records that the chain does not reach stationarity
    on an over-determined block anyway - but the weaker and checkable claim that the recorded vector
    is the one this fixture's closure was built around.
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

#: The identified fixture's own 5x5 grid, which is what `ID_THETA` was solved on - the four rows
#: `CHECKER_BENCHMARKS` prices are a subset of it plus two mixed-frequency variants.
ID_GRID = tuple((e, t, 3, 3) for e in (1, 2, 3, 5, 10) for t in (1, 2, 3, 5, 10))

#: theta* on that grid with `Objective: 'Analytic'`, AS SOLVED - the analytic twin of `ID_THETA`,
#: so a gate reads the analytic answer without paying 836 seconds for it. `Random_Seed` 5120, CPU
#: float64, 8192 paths declared but never drawn (the analytic closure draws none).
#:
#: THAT IT IS A SOLVE OUTPUT IS CHECKABLE IN SECONDS and is not taken on trust:
#: `test_the_analytic_objective_reaches_a_stationarity_the_quartic_cannot` reads `||J'r||` here and
#: gets 3.85e-6, which no authored vector lands on. Two coordinates sit ON a bound - `Sigma_1`'s
#: 5th knot at the 0.09 ceiling and its 6th at the 1e-5 floor - so the point is not interior and
#: unconstrained stationarity is not owed there; that it holds anyway says those directions carry
#: no gradient rather than that the bound is doing the work.
#: AND IT HAS BEEN RE-DERIVED THE LONG WAY: a full analytic chain on this block returns this vector
#: in 836 s over 1915 evaluations, with `||J'r||` 3.853e-6 and `||r||` 2.256e-3 - the numbers below.
#: Six of the twenty sigma knots are written here at SEVENTEEN significant digits because sixteen
#: does not round-trip a float64: transcribed at 16 they came back one ulp out, which is invisible
#: in every reading this file takes and is exactly why the constant is written to round-trip anyway.
ID_ANALYTIC_THETA = {
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

#: theta* on the FOUR-quote fixture under the DEFAULT objective, as solved on this box in float64.
#: It is the bit-identity baseline: the same vector, to the bit, before and after the analytic
#: objective was built - `test_the_default_objective_still_solves_to_this_vector` is the gate and
#: says what re-records it.
MC_FOUR_THETA = {
    'Alpha_1': [0.022545754213621216],
    'Alpha_2': [0.0018553868685500575],
    'Correlation': [-0.26307585773035419],
    'Sigma_1': [0.0059719210783128326, 0.0038731463946331019, 0.0053526691500668669,
                0.01877328396894392, 0.021282629111535927, 0.016109986402646746,
                0.0058307925191234539, 0.016199603181997538, 0.022549554433431374,
                0.01412973457408096],
    'Sigma_2': [0.0083557613189397927, 0.0068394360667839212, 0.0051707900399454295,
                0.010321136905445082, 0.021028822081422899, 0.016774750165862013,
                0.012819327482685442, 0.017221307246261863, 0.016757484356760555,
                0.019917356820708082]}

#: theta* on the FOUR-quote fixture under `Objective: 'Analytic'`, AS SOLVED - the analytic twin of
#: `MC_FOUR_THETA` and the second half of the theta-comparison, so the reading is taken on the
#: under-determined block as well as on the identified one. `Random_Seed` 5120, CPU float64;
#: 12.8 seconds over 351 evaluations, against 76.3 over 170 for the vector above.
#:
#: THAT IT IS A SOLVE OUTPUT IS CHECKABLE IN SECONDS: `||J'r||` here is 6.91e-8 against `||r||`
#: 4.01e-7, which is a fit that INTERPOLATES - four quotes against 23 parameters - and no authored
#: vector lands on it. `test_the_analytic_solve_is_deterministic_and_the_seed_moves_what_the_quotes_
#: do_not` re-solves it at this seed and is what would catch it going stale.
AN_FOUR_THETA = {
    'Alpha_1': [0.048592845601884482],
    'Alpha_2': [0.09440744561345675],
    'Correlation': [-0.11090827716509652],
    'Sigma_1': [1.2465372443084727e-02, 7.2778260503007170e-03, 1.4006085393093904e-02,
                3.5430024521239784e-03, 1.5960553435155459e-02, 9.6506709744549873e-03,
                1.2706915235818215e-02, 6.2807358340067013e-03, 1.8440690247164576e-02,
                2.8377539119097413e-02],
    'Sigma_2': [4.3376162862581984e-03, 1.4350349838542796e-02, 9.0120452652106865e-03,
                4.0131799581242791e-03, 3.4400275670176540e-02, 2.2503998642944115e-02,
                1.0268145640508276e-02, 3.1668568515472173e-02, 3.8415424141668202e-02,
                4.1795126093191949e-02]}


def flat_theta(calibration, named):
    """`named` as the flat vector `SwaptionCalibration` speaks in, in ITS parameter order."""
    return torch.tensor(np.concatenate([np.atleast_1d(named[k]) for k in calibration.keys]),
                        dtype=DTYPE)


def stationarity(calibration, theta):
    """`(||J'r||, ||r||)` at a flat theta - the quantity `LeastSquaresSolve.backward` refuses on.

    Read exactly the way that backward reads it: one fresh evaluation of the residual through
    `SwaptionCalibration.__call__`, one `autograd.grad` per row, promoted to float64. So this is
    the library's own stationarity measure rather than a second opinion of it.
    """
    x = theta.detach().clone().requires_grad_(True)
    residual = calibration(x)
    jacobian = torch.stack([torch.autograd.grad(residual[i], x, retain_graph=True)[0]
                            for i in range(residual.numel())]).double()
    residual = residual.detach().double()
    return float((jacobian.t() @ residual).norm()), float(residual.norm())


def calibration_at(theta, benchmarks=ID_GRID, **extra):
    """`(SwaptionCalibration, world)` standing at `theta` - no optimizer chain, so nothing solves."""
    from derivus.bootstrappers import SwaptionCalibration
    world = identified_closure(benchmarks=benchmarks, theta=theta, **extra)
    return SwaptionCalibration('gate', world['objective'], world['implied_var'], None,
                               world['process'], world['swaps']), world


def test_a_second_block_does_not_rescale_the_first_blocks_residual():
    """ONE BOOTSTRAPPER, MANY BLOCKS. `bootstrap` loops over `market_prices` on a single
    `RiskNeutralInterestRateModel`, so a residual closure that read its sample shape off `self`
    would be re-scaled by whatever the NEXT curve in that loop declares.

    THE MUTANT THIS NAMES is the sample shape reaching the closure through the instance -
    `shared_mem.reset(self.num_batches, ...)`, `range(self.num_batches)` and
    `v / (self.batch_size * self.num_batches)` in place of the locals `calc_loss_on_ir_curve` binds
    per block. Both attributes are written at the top of that call and read at CALL time, while
    `shared_mem` - which carries the drawn Sobol block and the `simulation_batch` it was drawn at -
    is built per block, so block A's closure goes on walking A's 2048 paths and divides them by B's
    8192.

    It is not a stale number nobody reads. `LeastSquaresSolve` keeps block A's calibration alive in
    `ctx` for a backward that runs after the whole loop has finished, so the corrupted closure is
    exactly the one an implicit-function-theorem Jacobian comes off. Measured with the mutant in
    place, block A at `Simulations=2048` against a block B at 8192:

        quantity              A alone         A after B was built
        Swaption_1Y_1Y      0.0063239682      0.0015809920      x 0.250000
        Swaption_5Y_5Y      0.0427945259      0.0106986315      x 0.250000
        ||J'r||               776642.6        7.4013242e+11     x 9.5e+05
        ||r||                 31.928282       113492

    so a quote-side solve either returns the Jacobian of a residual scaled by the wrong sample
    count, or `backward` fires its stationarity refusal naming a norm that belongs to no evaluation
    anyone made. With `Batches` differing rather than `Simulations` it is not a rescale but an
    index: a `Batches=1` block re-priced after a `Batches=4` one was built raises `IndexError: index
    1 is out of bounds for dimension 0 with size 1` out of `t_random_numbers`, because its own
    sample was drawn one block deep.

    Nothing else in this file can see it - `identified_closure` builds a fresh bootstrapper per
    world, which is why this gate builds its second block on the FIRST world's.
    """
    pair = ((1, 1, 3, 3), (5, 5, 3, 3))

    def second_block(world, **fields):
        """Another curve's block, built on THIS world's bootstrapper - `bootstrap`'s own loop."""
        return world['model'].calc_loss_on_ir_curve(
            {'instrument': dict(world['block'], **fields)}, BASE, world['time_grid'],
            world['process'], world['implied_obj'], world['ir_factor'], world['surface'])

    a = identified_closure(benchmarks=pair, batch_size=2048)
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
                                 Quote_Sensitivity='Yes')
    theta = flat_theta(cal, ID_THETA)
    before = stationarity(cal, theta)
    second_block(quoted, Simulations=8192)
    assert stationarity(cal, theta) == before, (
        "||J'r||, ||r|| read {} at theta* and {} after a second block was built - the backward's "
        'own measure, so a quote-side Jacobian there is one of a rescaled residual'.format(
            before, stationarity(cal, theta)))

    # and the `Batches` half, which does not rescale - it indexes off the end of the sample
    c = identified_closure(benchmarks=pair, batch_size=2048, Batches=1)
    priced = {k: as_float(v) for k, v in c['loss'](c['implied_var'])[0].items()}
    second_block(c, Batches=4)
    assert {k: as_float(v) for k, v in c['loss'](c['implied_var'])[0].items()} == priced, (
        'a Batches=1 block did not survive a Batches=4 block being built beside it')


def test_the_objective_field_declares_two_spellings_and_builds_two_things():
    """The declaration IS the menu, and the engine's fallback IS the declared default.

    The schema-emission suite holds the AST half - `Objective` is declared and the `.get` fallback
    beside it is the same string. What this holds is the half a store cannot see: that the two
    spellings the menu offers build two different objectives, and that the scalar each hands the
    optimizer chain is a different reduction.
    """
    field = next(f for f in HullWhite2FactorModelParameters.fields if f.name == 'Objective')
    assert field.default == 'Monte_Carlo'
    assert sorted(field.values) == ['Analytic', 'Monte_Carlo']

    mc = identified_closure(benchmarks=((2, 5, 3, 6),), batch_size=2048)
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
    """The one-touch lesson: a spelling the engine does not know RAISES naming the menu, rather
    than falling through to whichever branch the `if` happens to end on."""
    for spelling in ('analytic', 'SchragerPelsser', 'MonteCarlo', ''):
        with pytest.raises(Exception, match='Monte_Carlo'):
            identified_closure(benchmarks=((1, 1, 3, 3),), batch_size=1024, Objective=spelling)
    try:
        identified_closure(benchmarks=((1, 1, 3, 3),), batch_size=1024, Objective='analytic')
    except Exception as e:
        assert 'Analytic' in str(e), 'the refusal has to name both spellings: {}'.format(e)


def test_the_analytic_quote_side_refuses_and_names_todays_remedy():
    """`Quote_Sensitivity` on the analytic objective is NOT BUILT, and says so rather than handing
    back a quote Jacobian of a residual nothing spliced a quote into.

    The splice `market_swap_class.error` carries is simply absent from `normal_vol_error`, so
    without this refusal a solve would run to completion and the backward would report zeros -
    the quiet-garbage failure the rest of this file exists to remove. The refusal names the thing
    (the market vol enters that residual linearly, through the closed-form Bachelier inversion of
    the premium) and the remedy (Monte_Carlo for a quote-differentiable solve today).
    """
    with pytest.raises(Exception, match='Quote_Sensitivity'):
        identified_closure(benchmarks=((1, 1, 3, 3),), batch_size=1024,
                           Objective='Analytic', Quote_Sensitivity='Yes')
    try:
        identified_closure(benchmarks=((1, 1, 3, 3),), batch_size=1024,
                           Objective='Analytic', Quote_Sensitivity='Yes')
    except Exception as e:
        assert 'Monte_Carlo' in str(e), 'the refusal has to name the remedy: {}'.format(e)
    # the Monte Carlo path with the quote side on still builds, which is what makes it a remedy
    world = identified_closure(benchmarks=((1, 1, 3, 3),), batch_size=1024, Quote_Sensitivity='Yes')
    assert world['swaps']['Swaption_1Y_1Y'].quote is not None
    # and the analytic path with it OFF is the ordinary case, so the refusal is on the pair
    assert identified_closure(benchmarks=((1, 1, 3, 3),), batch_size=1024,
                              Objective='Analytic')['objective'].reprice is not None


def test_the_market_normal_vol_is_a_division_and_it_round_trips():
    """THE CLOSED-FORM INVERSION, held against the pair it inverts.

    At the money the Bachelier premium is `A sigma_N sqrt(T0/2pi)`, so
    `sigma_N = P sqrt(2pi/T0) / A` is a DIVISION - no brentq, no bracket, nothing that could fail
    to converge inside an optimizer evaluation. The round trip is the identity itself: hand
    `market_normal_vol` the analytic price's OWN premium and it gives back that price's own normal
    vol, to float precision.

    Then the thing that matters: hand it the MARKET premium `create_market_swaps` built, and the
    residual `normal_vol_error` returns is the premium residual rescaled by exactly
    `weight sqrt(2pi/T0) / A`. That is what makes vols-against-vols and premium-against-premium the
    same zero rather than two different fits, and it is why no quoting convention can be inherited
    by half: all of them arrive as that premium.
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
    # and the annuity is SP's OWN, not a second spelling: it is the pvbp `create_market_swaps`
    # struck the premium on, which is the whole reason the rescaling above is exact
    swap = world['swaps']['Swaption_2Y_5Y']
    sp = world['process'].schrager_pelsser_swaption(
        swap.schedule.expiry, swap.schedule.pay_times, swap.schedule.accruals)
    assert abs(as_float(sp.annuity) / (swap.price / utils.black_european_option_price(
        as_float(sp.swap_rate), as_float(sp.swap_rate), 0.0, 0.20, swap.schedule.expiry,
        1.0, 1.0)) - 1.0) < 1e-3, 'SP\'s annuity is not the pvbp the market premium was struck on'


def test_a_displaced_surface_reaches_the_bachelier_side_through_the_premium_only():
    """THE CONVENTION INHERITANCE, on the convention most likely to be dropped.

    A shifted-lognormal surface strikes its Black premium at `K + shift`; `create_market_swaps`
    does that and nothing downstream of it knows. So the shift has to reach the analytic residual
    through `swap.price` and NOWHERE else - the Bachelier inversion carries no displacement term,
    because a normal vol has nothing to displace.

    Held both ways: the market normal vol MOVES with the shift, because the premium did, and it is
    exactly the shifted premium's own inversion on an annuity that did not move at all.
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
    """THE CLAIM THE BUILD WAS FOR, on the repository's own identified 25-quote fixture, measured.

    `market_swap_class.error` returns a residual that is ALREADY a square, so `least_squares`
    minimises a QUARTIC in the pricing error and stops where a quartic goes flat. The recorded
    Monte Carlo theta* reads `||J'r||` **8.24e3** against `||r||` **45.97** on this box in float64
    - `docs_src/developer/quote_sensitivities.md` recorded 8.6e3 and 46.1 in float32 on CUDA, the
    same reading - having come down from **2.86e11** at the seed. Seven and a half of the eleven
    orders, and it stops: which is why this file's own `ID_STATIONARITY` is **1e5**, because the
    quote-side backward would otherwise refuse the fixture it was built for.

    `normal_vol_error` returns the difference itself, so `least_squares` minimises the sum of
    squares it was written for. At the analytic theta*, same fixture, `||J'r||` is **3.85e-6**
    against `||r||` **2.26e-3**, down from **6.00e-2** at the seed. In the units of the declared
    field that is the whole finding:

        objective        ||J'r|| at theta*     the Stationarity_Tol that accepts it
        Monte Carlo             8.24e+03       1e5   (this file has to declare it)
        Analytic                3.85e-06       1e-3  (the field's own DEFAULT)

    EIGHT ORDERS of declared tolerance, and the analytic path is the one that clears the default.
    The raw norms are in different units - a weighted squared PERCENTAGE against an absolute
    normal VOL - so the 9.3 orders between them is not a like-for-like ratio and is not claimed as
    one. What IS like-for-like is the tolerance each needs, and the shape of the descent: the
    quartic falls 7.5 orders and stops 8240 short of zero; the quadratic falls 4.2 orders and
    lands 3.9e-6 short of it.
    """
    field = next(f for f in HullWhite2FactorModelParameters.fields if f.name == 'Stationarity_Tol')
    assert field.default == 1e-3 and ID_STATIONARITY == 1e5, (field.default, ID_STATIONARITY)

    for tag, extra, theta, bound, resid in (
            ('Monte_Carlo', {}, ID_THETA, (1e3, 1e5), (40.0, 55.0)),
            ('Analytic', {'Objective': 'Analytic'}, ID_ANALYTIC_THETA, (0.0, 1e-3), (1e-3, 5e-3))):
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


def repriced_vols(theta, benchmarks):
    """`{benchmark: (SP normal vol, the market's own inverted premium)}` in bp, at `theta`.

    Both sides through the ANALYTIC closure, so the model vol and the market vol are read off one
    annuity - `market_normal_vol` takes SP's own - and the difference below is the premium residual
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


def test_the_two_answers_agree_in_vol_space_and_share_nothing_in_theta_space():
    """THE THETA-COMPARISON, on BOTH fixtures, and it comes out as two findings rather than one.

    IN REPRICED NORMAL-VOL SPACE THEY AGREE. Every benchmark's ATM vol at both solved points,
    against the market's own inverted premium, in basis points. On the identified 25-quote grid,
    level 175 to 208:

        rms |SP - market|      at Monte Carlo theta* 5.14bp      at Analytic theta* 4.51bp
        worst benchmark        10.97bp (5Y x 5Y)                  9.43bp (5Y x 5Y)
        rms |the two apart|    2.52bp, worst 6.38bp at 3Y x 10Y

    So the analytic solve fits the grid 12% better in the metric it is fitting, and the two answers
    price the whole cube within 6.4 basis points of each other - the band the measured SP bias
    itself spans (-0.13 to +2.17bp) plus one evaluation of the Monte Carlo's own noise (0.5 to
    1.8bp), which is as close as two objectives on an unidentified block can be asked to come.

    ON THE FOUR-QUOTE BLOCK THE SAME COMPARISON READS DIFFERENTLY, AND THAT IS THE POINT OF TAKING
    IT TWICE. Four quotes against 23 parameters is UNDER-determined, so each objective interpolates
    exactly in ITS OWN metric and the gap between them is the two metrics', not a disagreement
    about the market:

        benchmark        market   SP @ theta*_MC   SP @ theta*_An   the two apart
        1Y x 1Y          175.49       174.86           175.49          -0.63
        2Y x 5Y          202.45       203.89           202.45          +1.44
        3Y x 3Y          205.46       207.58           205.46          +2.12
        10Y x 10Y        180.00       184.13           180.00          +4.13

        rms |SP - market|    at theta*_MC 2.45bp    at theta*_An 0.00bp (it interpolates)

    The analytic column is the market column TO THE PRINTED DIGIT because `||r||` there is 4.0e-7 -
    that is under-determination and not accuracy, and it is why the 25-quote block is the fixture
    the objective is judged on. What the OTHER column measures is worth having: at theta*_MC the
    Monte Carlo premium residual is ~0 (`||r||` 4.4e-8), so 2.45bp rms is the whole SP-versus-MC gap
    at a point where the simulation fits exactly - SP's freezing bias and the simulation's own
    numeraire error, and at 10Y x 10Y they ADD rather than cancel. SP reads 4.13bp HIGH there:
    about 2.9bp of it is the -1.61% numeraire error `test_the_monte_carlo_carries_a_bias_of_its_own`
    measures at that benchmark on this 1Y-first-node curve (the model has to be pumped up to make a
    biased-low estimator hit the market premium) and about 2.2bp is SP's own worst corner. Two
    biases with two names, summing to the number printed, on the block where nothing else is moving.

    IN THETA SPACE THEY SHARE NOTHING ON EITHER FIXTURE, and that is rank deficiency
    ([Quote Sensitivities](quote_sensitivities.md#rank-deficiency)) rather than a disagreement:

        fixture       Correlation           Alpha_1          Alpha_2       max |theta apart|
        25-quote      -0.0046 / -0.9409   0.122 / 0.020   -0.018 / 0.052        0.936
        4-quote       -0.2631 / -0.1109   0.023 / 0.049    0.002 / 0.094        0.152

    The 25-quote block's singular spectrum runs `sigma_min/sigma_max` 4.6e-6 and the declared cutoff
    keeps 18 of 23 directions, so most of theta is a direction 25 flat quotes cannot choose in; the
    four-quote block has a 19-dimensional null space outright. Two objectives landing in different
    parts of that null space is the expected result, and it is why the comparison that means
    anything is the one above, in the space the quotes live in.
    """
    for tag, benchmarks, mc_theta, an_theta, bound in (
            ('25-quote', ID_GRID, ID_THETA, ID_ANALYTIC_THETA,
             dict(rms_mc=(3.0, 8.0), rms_an=(3.0, 7.0), apart=(4.0, 9.0), theta=0.9)),
            ('4-quote', CHECKER_BENCHMARKS, MC_FOUR_THETA, AN_FOUR_THETA,
             dict(rms_mc=(1.2, 4.0), rms_an=(0.0, 0.05), apart=(4.0, 7.0), theta=0.05))):
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
        # and the theta-space half, which is the finding rather than the agreement
        assert abs(mc_theta['Correlation'][0] - an_theta['Correlation'][0]) > bound['theta'], (
            '{}: the two correlations have converged - if that is real this fixture has become '
            'identified and the null-space reading above needs re-taking'.format(tag))


def bootstrap_the_block(**extra):
    """One `RiskNeutralInterestRateModel.bootstrap` on the identified fixture, run whole.

    Every gate above stops at the closure or at `SwaptionCalibration`; this drives the entry point a
    job drives, because the honesty reprice and the parameters it writes only exist out here.
    """
    factors = identified_world()
    block = {'Swaption_Volatility': ID_VOL, 'Generate_Instruments': 'No', 'Random_Seed': ID_SEED,
             'Stationarity_Tol': ID_STATIONARITY, 'Quote_Sensitivity': 'No',
             'Simulations': 2048, 'Batches': 1,
             'Instrument_Definitions': identified_definitions(CHECKER_BENCHMARKS)}
    block.update(extra)
    model = HullWhite2FactorModelParameters({}, DEVICE, DTYPE)
    model.bootstrap({'Base_Date': BASE, 'Base_Currency': ID_CCY}, {}, factors, ModelParams(),
                    {ID_BLOCK: {'instrument': block}}, {})
    return model, factors


def test_the_analytic_solve_reports_what_the_engines_own_estimator_makes_of_it(caplog):
    """THE HONESTY REPRICE, which is the analytic objective's own CAPPED line.

    An analytic solve fits Schrager-Pelsser vols and Schrager-Pelsser freezes the annuity's
    weights, so nothing in that solve has ever asked the Monte Carlo the rest of the library prices
    with what it makes of the answer. `SwaptionCalibration.honesty_reprice` runs one pass of that
    estimator at theta* and `bootstrap` LOGS the worst benchmark's relative premium residual by
    name - not as a check with a tolerance, the way the component Heston-Nandi fit reports itself
    CAPPED rather than claiming a tolerance it did not reach.

    THE NUMBER IS MOSTLY THE SIMULATION'S, which is why it is reported and not asserted tightly -
    and the four-quote block says so almost exactly. There the reprice names **10Y x 10Y** and
    reads **-1.67% / -1.65% / -1.78%** across seeds 5120 / 7 / 99, against the **-1.61%** that
    `test_the_monte_carlo_carries_a_bias_of_its_own` measures for the SIMULATION'S OWN numeraire
    error at that same benchmark on this fixture's 1Y-first-node curve. So the honesty reprice is
    reporting the estimator's own bias with a tenth of a percent of approximation on top, which is
    the honest thing for it to be reporting. On the 25-quote grid it reads -4.64% at 1Y x 5Y.

    THE NUMBER MOVES WITH THE REPRICE'S OWN PATH COUNT, which is the other reason it is reported
    rather than asserted tightly: this gate declares `Simulations: 2048` to stay under a minute and
    reads **-2.29%** at the same benchmark, against -1.67% at the declared default of 8192. The
    reprice prices at the block's own count, so a block that wants a tighter audit buys one.

    The Monte Carlo objective logs NOTHING here, because there is nothing to audit: it IS the
    estimator.
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
        bootstrap_the_block()
    assert not [r for r in caplog.records if 'Analytic objective' in r.getMessage()], (
        'the Monte Carlo objective audited itself - it IS the estimator, so there is nothing to say')


def test_the_analytic_solve_is_deterministic_and_the_seed_moves_what_the_quotes_do_not():
    """DETERMINISM, and the seed spread beside the Monte Carlo path's - measured, not assumed.

    The analytic objective draws no sample at all, so the only randomness left in the chain is
    basin hopping's own search, and `Random_Seed` is the single generator serving its step taker
    and its Metropolis test. Two runs at one seed therefore have to agree TO THE BIT, and they do.

    THE SEED SPREAD IS LARGER THAN THE MONTE CARLO PATH'S ON THIS FIXTURE, and that is recorded
    because the opposite was expected. Across seeds 5120 / 7 / 99 on the four-quote identified
    block, this box, CPU float64:

        objective       evaluations    s/evaluation    max |theta* apart|   where it lives
        Monte Carlo    170/170/170    0.438..0.449            0.0       nowhere: bit-identical
        Analytic       351/358/394    0.0362..0.0364          0.250     Correlation (0.155 in
                                                                        Alpha_2, 0.075 Alpha_1)

    Neither number is noise and both come from the same thing. FOUR quotes against 23 parameters
    leaves a 19-dimensional null space, so theta* is a MANIFOLD and both objectives fit it
    essentially exactly - `||r||` is 4.4e-8 on the Monte Carlo path and 4.0e-7 on the analytic one.
    What differs is which point of that manifold the search reaches. The Monte Carlo chain lands
    0.273 off its seed and lands there for EVERY seed - bit-identical theta*, all three - so its
    basin stage is contributing nothing the least-squares stage does not; the analytic evaluation is
    twelve times cheaper, the chain makes twice as many of them, and it moves 0.121 / 0.368 / 0.371
    off the seed on the three - the search is actually exploring. The correlation carrying all of
    the spread is exactly the coordinate four ATM quotes say least about.

    So this is a DETERMINISM gate and a spread READING, and the spread is not evidence FOR the
    analytic objective - it is evidence that this fixture does not identify its parameters, which
    is the same thing `test_the_two_answers_agree_in_vol_space_and_share_nothing_in_theta_space`
    reads off the 25-quote block. The evidence for the objective is
    `test_the_analytic_objective_reaches_a_stationarity_the_quartic_cannot`.

    WALL CLOCK, this box, CPU float64, nothing else running, over the same three seeds: the
    four-quote chain is 76.3 / 74.4 / 74.6 s under the Monte Carlo objective and 12.8 / 13.0 /
    14.3 s under the analytic one - 75.1 s against 13.4 s on the mean, a factor of 5.6 bought as
    12x per evaluation against 2.2x the evaluations. The chain needs MORE iterations, not fewer,
    which is what a smooth deterministic residual buys: `least_squares` can keep making progress on
    one.

    The 25-quote identified block solves analytically in 836 s on the same box, over 1915
    evaluations at 0.437 s each - and that per-evaluation figure is the one to read, because it is
    TWELVE times the four-quote block's 0.036 for six times the benchmarks. Schrager-Pelsser is
    called once per benchmark and the grid it integrates `J` over grows with the expiry set, so the
    analytic pass scales worse than the simulation's single batched kernel does. Batching it across
    the benchmark set is the open build; what the objective buys today is exactness and a gradient.

    THAT 836 IS NOT TO BE READ AGAINST THE 531 the roadmap records for the Monte Carlo objective on
    the same block: that figure is float32 on CUDA, which is what `construct_bootstrapper` hands a
    job, and this one is float64 on CPU. The two like-for-like readings are the four-quote pair
    above. Per EVALUATION on CUDA float32 the analytic price is still the slower of the two -
    0.158 s against 0.140 s, twenty-five scalar calls against one batched kernel - so batching it
    across the benchmark set is the build this leaves on the table, and what the objective buys
    today is exactness, a gradient, and a chain that agrees with itself.
    """
    first, second = (identified_calibration(Objective='Analytic')[0].solve() for _ in range(2))
    assert [float(v).hex() for v in first.numpy()] == [float(v).hex() for v in second.numpy()], (
        'two analytic solves at one seed disagree - the objective draws no sample, so the only '
        'thing left that could move is the search, and it is seeded')
    # and the answer is not the seed, or determinism would be free
    calibration, world = identified_calibration(Objective='Analytic')
    seed = torch.cat([v.detach().clone() for v in world['implied_var'].values()]).double()
    assert float((first.double() - seed).abs().max()) > 1e-3, 'the chain returned its own seed'
    # and it is the vector the theta-comparison reads, so that constant is a SOLVE OUTPUT rather
    # than an authored one - the same claim `MC_FOUR_THETA` makes, and re-recorded the same way
    got = calibration.unflatten(first)
    for name, recorded in AN_FOUR_THETA.items():
        assert [float(v).hex() for v in np.atleast_1d(got[name])] == [
            float(v).hex() for v in recorded], (
            '{} solved to {} against the recorded {}'.format(name, list(got[name]), recorded))
    # the fit is essentially exact, which is what makes theta* a manifold rather than a point
    residual = stationarity(calibration, first)[1]
    assert residual < 1e-5, (
        'the four-quote analytic fit reads ||r|| {:.3e} against a recorded 4.0e-7 - it is 4 quotes '
        'against 23 parameters, so it interpolates'.format(residual))


def test_the_default_objective_still_solves_to_this_vector():
    """THE BIT-IDENTITY BASELINE: the whole chain, on the four-quote fixture, at the default.

    `MC_FOUR_THETA` was solved before this workstream and re-solved after it, and the 23 doubles
    came back IDENTICAL to the bit - same seed, same frozen Sobol sample, same acceptance rule.
    That is the claim `Objective`'s default is making, and it is the reason the batch repair had to
    be a no-op at `Batches: 1`: the clear at the top of iteration zero runs against a `t_Buffer`
    that `reset` has just emptied and `precalculate` only writes `t_PreCalc`, so it removes nothing.

    This gate pays 70 seconds for it because nothing cheaper is the claim. If a scipy or torch
    upgrade moves the last bits, the honest repair is to RE-RECORD this vector after checking that
    the residual at a fixed theta has not moved - the residual is arithmetic and cannot drift; an
    optimizer's stopping point can.
    """
    calibration, _ = identified_calibration()
    theta = calibration.solve()
    got = calibration.unflatten(theta)
    for name, recorded in MC_FOUR_THETA.items():
        assert [float(v).hex() for v in np.atleast_1d(got[name])] == [
            float(v).hex() for v in recorded], (
            '{} solved to {} against the recorded {}'.format(name, list(got[name]), recorded))


def test_the_two_spellings_of_the_default_drive_the_adapters_identically():
    """`Objective` absent and `Objective: 'Monte_Carlo'` are one job, held at the SEAMS the chain
    uses rather than only at the residual.

    Three of them, at three different parameter vectors: the residual `SwaptionCalibration.__call__`
    hands the implicit-function wrapper, the scalar-and-gradient pair basin hopping reads, and the
    residual-and-Jacobian pair `least_squares` reads. All bitwise. A gate that only compared the
    residual would miss a `reduce` that differed, which is the one new thing in that path.
    """
    a_cal, a_world = identified_calibration(batch_size=2048, Objective='Monte_Carlo')
    b_cal, b_world = identified_calibration(batch_size=2048)
    assert a_world['block']['Objective'] == 'Monte_Carlo'
    assert 'Objective' not in b_world['block'], 'the second block has to OMIT the field'
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


def test_the_analytic_price_is_quanto_free_where_the_simulation_is_not():
    """A FINDING ABOUT THE MONTE CARLO, taken and recorded rather than patched.

    Schrager-Pelsser reads `J` alone - the covariance of the scaled martingales under the RATE
    currency's own risk-neutral measure - so it prices the DOMESTIC swaption, which is the correct
    measure for a domestic payoff whatever the base currency of the job. The Monte Carlo objective
    simulates through the same `precalculate`, and that installs the quanto drift `K` into `KtT`,
    so on a non-base-currency curve THE TWO OBJECTIVES ARE NOT PRICING UNDER THE SAME MEASURE.

    Measured here, on the four benchmarks this gate prices: the identified fixture's ZAR curve made
    foreign under a USD base, with a 15-20% FX vol curve and a 0.4 FX/IR correlation, at the
    recorded theta* on one Sobol sample.

        benchmark      MC premium, rho=0.4       rho=0      moves by     SP moves by
        1Y x 1Y             0.006789635      0.006407690     +5.96%      0 (bitwise)
        2Y x 5Y             0.039594307      0.036378626     +8.84%      0 (bitwise)
        3Y x 3Y             0.030439928      0.027227734    +11.80%      0 (bitwise)
        10Y x 10Y           0.066546269      0.059192866    +12.42%      0 (bitwise)

    THE BENCHMARK SET IS PART OF THE READING. These are `CHECKER_BENCHMARKS`, and the block's
    `Instrument_Definitions` set its `TimeGrid` and therefore its Sobol sample - swapping the third
    row for a 5Y x 5Y moves that row to +14.08% AND the untouched 10Y x 10Y to +12.43%. So a table
    taken on one set is not a reading of another, and this one names its set.

    Which is 10.9 to 24.4 basis points of ATM normal vol - an ATM Bachelier premium is linear in the
    vol, so the two percentages are the same percentage. Against the 0.13 to 2.17bp the
    approximation itself costs, that is FIVE TO ELEVEN TIMES the worst corner of SP's own bias and
    fifty to two hundred times its typical one. The disagreement is the MONTE CARLO'S - a domestic swaption
    deflated domestically but simulated under the base measure's drift - and repairing it moves
    every calibrated foreign-curve parameter set in existence, so it is a Known-defects row and a
    decision rather than a patch. What this gate holds is the shape of the finding: the simulation
    moves with the correlation and the analytic price does not move at all.
    """
    def quanto_world(correlation):
        factors = identified_world()
        factors['GBMAssetPriceTSModelParameters.{}'.format(ID_CCY)] = {
            'Property_Aliases': None, 'Quanto_FX_Volatility': None, 'Quanto_FX_Correlation': 0.0,
            'Vol': utils.Curve([], [(0.0, 0.15), (1.0, 0.15), (3.0, 0.17), (5.0, 0.18),
                                    (10.0, 0.20)])}
        factors['Correlation.FxRate.USD.{}/InterestRate.{}'.format(ID_CCY, ID_CCY)] = {
            'Value': correlation}
        return factors

    reading = {}
    for correlation in (0.4, 0.0):
        world = identified_closure(benchmarks=CHECKER_BENCHMARKS, world=quanto_world(correlation),
                                   base_currency='USD', batch_size=8192)
        prices, _ = world['loss'](world['implied_var'])
        analytic = {}
        for name, swap in world['swaps'].items():
            sp = world['process'].schrager_pelsser_swaption(
                swap.schedule.expiry, swap.schedule.pay_times, swap.schedule.accruals)
            analytic[name] = as_float(sp.premium)
        reading[correlation] = ({k: as_float(v) for k, v in prices.items()}, analytic,
                                float(world['process'].KtT[0].abs().max().detach()))

    (mc_on, sp_on, k_on), (mc_off, sp_off, k_off) = reading[0.4], reading[0.0]
    assert k_on > 0.0 and k_off == 0.0, (
        'the quanto drift is {} with the correlation on and {} with it off - if the first is zero '
        'this world is not quanto at all and the gate measures nothing'.format(k_on, k_off))
    for name in mc_on:
        assert sp_on[name] == sp_off[name], (
            '{}: the analytic price moved with the FX/IR correlation, which it must not - it reads '
            'J alone and J carries no quanto drift'.format(name))
        moved = mc_on[name] / mc_off[name] - 1.0
        assert 0.03 < moved < 0.25, (
            '{}: the simulated premium moves {:+.2%} with the correlation against a recorded +5.96% '
            'to +12.42% - this is the measure inconsistency, and it is the finding'.format(
                name, moved))
    # the size of it against the approximation it would be confused with: SP's own bias is at most
    # 2.17bp on a ~180bp level, and the smallest quanto move here is five times that
    assert min(mc_on[n] / mc_off[n] - 1.0 for n in mc_on) > 4.0 * (2.17 / 180.0)
