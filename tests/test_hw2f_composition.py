"""The HW2F composition, on an authored world, through the harness's own functions.

`gates/hw2f_composition.py` is the instrument; this is its CANNED TWIN at test scale on a world
small enough to gate and invented enough to commit. Every function under test is imported from the
gate, so there is one spelling of the pipeline.

WHAT THE WORLD IS AUTHORED TO BE, every choice load-bearing:

- A **FLAT** USD curve. The base currency's curve is STATIC, and a static curve does not roll: the
  FX drift reads its frozen t=0 view at every step, accumulating `T x r(0, dt)` rather than
  `-log D(0,T)`. Those are two different objects, and identity 1 as posed can only close where they
  agree - which a flat curve is. That takes the error out of the world and leaves the measure
  change to be measured on its own; the live instrument reports it instead of removing it.
- **SHORT expiries** (6M and 1Y) and a **weekly** grid. The composition's numeraire is a discretely
  rolled bond whose gap grows with the horizon and the step: identity 1 read -6.93% at 1Yx1Y
  monthly against +1.13% weekly at an unchanged 0.66% standard error.
- A **55% FX vol and a 0.8 FX/IR correlation**. Neither is realistic and both are deliberate: the
  quanto drift is what the mutation removes, and a world where it is worth less than the Monte
  Carlo noise cannot tell a broken `K` from a working one. The kill is exactly LINEAR in the FX vol
  (`K` carries one factor of `sigma_FX`) - 1.640% at 25%, 3.591% at 55% on the binding 1Yx1Y cell -
  which makes the level a reading rather than a search. It also moves the CLEAN gate the right way:
  identity 1's miss grows slower in this knob than its own error bar does, buying 0.24 sigma of
  headroom while it buys 1.8 on the kill.

  THE OTHER TWO LEVERS WERE MEASURED AND REJECTED. Refining the grid takes the payoff-free
  NUMERAIRE miss to zero (-0.301% weekly, +0.019% daily) and leaves identity 1 at -1.54%, because a
  drift error is not a discretisation one; raising the path count shrinks the band faster than it
  lifts the mutation over it, since identity 1's residual is BIAS - at four times the paths the
  clean gate would fail at 5.2 sigma while the kill only reached 4.1.

EVERY BOUND HERE IS MEASURED AND SPELLED AS A MULTIPLE OF THE READING'S OWN STANDARD ERROR, so a
path count or grid change moves every band with the noise rather than orphaning an assertion. No
absolute tolerance is declared: a chosen constant is a number that was true of one sample.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'gates'))

import numpy as np
import pandas as pd
import pytest
import torch

import hw2f_composition as H
from derivus import utils

BASE = pd.Timestamp('2026-09-01')
CURRENCIES = {'base': 'USD', 'rate': 'ZAR'}
#: One unit of ZAR in USD. Authored on the ENGINE's axis (base per unit), which is what
#: `test_the_fx_axis_is_base_per_unit` exists to hold: the screen quotes the reciprocal.
FX_SPOT = 1.0 / 16.0
#: The FX/IR correlation, on the desk's own pair (ZAR per USD). Strong on purpose - see the module
#: docstring. `implied_process` flips its sign on the way into the calibration.
RHO = 0.8
#: The ATM FX vol the whole surface is authored at, and the knob that sets how big the quanto drift
#: the mutation removes is. Unrealistic on purpose - the module docstring carries the sweep that
#: chose it. It reaches the composition only: the calibration objective runs on `implied_process`'s
#: SUPPRESSED twin, so theta* is not a function of this number and
#: `test_the_fit_does_not_move_with_the_fx_inputs` says so.
FX_VOL = 0.55
#: The grid and sample the composition gates run at, CHOSEN OFF A MEASUREMENT. The composition's
#: numeraire is a discretely-rolled bond, so the miss is a function of the step: identity 1 reads
#: (6Mx1Y / 1Yx1Y) -0.27%/-6.93% monthly, +0.46%/-1.51% fortnightly and +0.26%/+1.13% weekly
#: against a standard error of 0.70%/0.66%, and only weekly is inside three of its own sigmas at
#: both cells.
#:
#: REFINING FURTHER DOES NOT TAKE IT TO ZERO, which is a FINDING rather than a tolerance: the 1Yx1Y
#: cell reads -2.108% weekly, -1.700% at three days and -1.540% daily, plateauing near -1.5%, while
#: the payoff-free numeraire identity - the half that IS a discretisation error - converges cleanly
#: to +0.019%. So identity 1 carries a residual the step size does not explain. It is inside the
#: three-sigma band at this path count and would not be at four times it.
GRID, BATCH, BATCHES, SEED = '0d 1w(1w)', 8192, 4, 5120
#: The band identity 1 is asserted inside, as a multiple of the run's own standard error.
IDENTITY_SIGMA = 3.0
#: The band the payoff-free numeraire identity is asserted inside, as a multiple of THAT reading's
#: own standard error. FIVE rather than three because this reading is bias PLUS noise where
#: identity 1's is noise: the discretely-rolled bond's gap is deterministic and no path count
#: removes it. The multiple is what the SEED SPREAD sets rather than the pinned seed - across six
#: probed seeds the worst 1Yx1Y reading was 3.7 of its own sigmas, which a three-sigma band fails.
#:
#: THE MISS BARELY MOVED across the 2026-09-02 HW2F re-mark and that is worth keeping: this
#: identity has no theta in it at all, so an HW2F re-mark cannot reach it and what grew is its own
#: error bar. Identity 1 moved and this did not, which is the cleanest statement available that the
#: two gates measure two things.
NUMERAIRE_SIGMA = 5.0

#: The composition's benchmarks: short expiries, so the numeraire is not what is being measured.
CELLS = (('6M', '1Y'), ('1Y', '1Y'))
#: Four cells against twenty-three parameters is under-determined and that is fine: an
#: under-determined fit reaches its quotes exactly, which makes the analytic price the only thing
#: identity 1 can be missing by.
LADDER = ((('6M',), ('1Y',), 1.10), (('1Y',), ('1Y',), 1.15),
          (('1Y',), ('2Y',), 1.18), (('2Y',), ('2Y',), 1.20))


# ---------------------------------------------------------------------------------------------
# THE AUTHORED WORLD
# ---------------------------------------------------------------------------------------------

def _deposit(reference, currency, curve, months, quote, day_count='ACT_365'):
    """A money-market deposit at `quote` percent, its rate PINNED through
    `Interest_Rate_Schedule` so the quote cannot depend on the curve it identifies."""
    return {
        'Object': 'DepositDeal', 'Reference': reference, 'Currency': currency,
        'Discount_Rate': curve, 'Interest_Rate': curve, 'Effective_Date': BASE,
        'Maturity_Date': BASE + pd.DateOffset(months=months),
        'Payment_Frequency': pd.DateOffset(months=months),
        'Interest_Frequency': pd.DateOffset(months=months),
        'Accrual_Day_Count': day_count, 'Amount': 1e6, 'Amortisation': None,
        'Compounding': 'No', 'Payment_Timing': 'End', 'Payment_Offset': 0,
        'Accrual_Calendars': None, 'Payment_Calendars': None, 'First_Coupon_Date': None,
        'Penultimate_Coupon_Date': None, 'Rate_Currency': '', 'FX_Reset_Offset': 0,
        'Known_FX_Rates': None, 'Interest_Rate_Schedule': utils.DateList({BASE: quote})}


def _par_swap(reference, currency, curve, years, quote, day_count='ACT_365'):
    """A par swap at `quote` percent - fixed quarterly against a single-reset quarterly float leg on
    one curve. `Index_Tenor` of zero months gives each coupon ONE reset over its accrual period."""
    quarterly = pd.DateOffset(months=3)
    return {
        'Object': 'SwapInterestDeal', 'Reference': reference, 'Currency': currency,
        'Discount_Rate': curve, 'Interest_Rate': curve, 'Effective_Date': BASE,
        'Maturity_Date': BASE + pd.DateOffset(years=years),
        'Pay_Rate_Type': 'Fixed', 'Pay_Frequency': quarterly, 'Pay_Day_Count': day_count,
        'Pay_Interest_Frequency': quarterly, 'Pay_Timing': 'End', 'Pay_Payment_Offset': 0,
        'Pay_Accrual_Calendars': None, 'Pay_Payment_Calendars': None,
        'Pay_First_Coupon_Date': None, 'Pay_Penultimate_Coupon_Date': None,
        'Receive_Frequency': quarterly, 'Receive_Day_Count': day_count,
        'Receive_Interest_Frequency': pd.DateOffset(months=0), 'Receive_Timing': 'End',
        'Receive_Payment_Offset': 0, 'Receive_Accrual_Calendars': None,
        'Receive_Payment_Calendars': None, 'Receive_First_Coupon_Date': None,
        'Receive_Penultimate_Coupon_Date': None,
        'Index_Tenor': pd.DateOffset(months=0), 'Index_Day_Count': day_count,
        'Index_Frequency': pd.DateOffset(months=0), 'Index_Offset': 0,
        'Index_Calendars': None, 'Index_Publication_Calendars': None,
        'Reset_Type': 'Standard', 'Rate_Multiplier': 1.0, 'Rate_Constant': utils.Percent(0.0),
        'Floating_Margin': 0.0, 'Fixed_Compounding': 'No', 'Compounding_Method': 'None',
        'Known_Rates': None, 'Amortisation': None, 'Swap_Rate': quote, 'Principal': 1e6,
        'Interest_Rate_Volatility': '', 'Discount_Rate_Volatility': ''}


def _point(descriptor, deal, quote):
    """A `Points` row: the type NAMED, the block carried, the quote beside it. `Object` and
    `Discount_Rate` are dropped - the row names the type and the block stamps the curve."""
    return {'Use': 'Yes', 'Descriptor': descriptor, 'DealType': deal['Object'],
            'Quote_Type': 'Par_Rate', 'Quoted_Market_Value': quote,
            'Deal': {k: v for k, v in deal.items() if k not in ('Object', 'Discount_Rate')}}


def rates_block(currency, curve, quotes, day_count='ACT_365'):
    """One `InterestRatePrices` block: a 3M deposit and par swaps, single-curve."""
    points = [_point('{} 3M depo'.format(currency),
                     _deposit('DEPO_3M', currency, curve, 3, quotes[0], day_count), quotes[0])]
    for years, quote in quotes[1]:
        points.append(_point(
            '{} {}Y IRS'.format(currency, years),
            _par_swap('IRS_{}Y'.format(years), currency, curve, years, quote, day_count), quote))
    return {'instrument': {'Currency': currency, 'Day_Count': day_count, 'Discount_Rate': '',
                           'Points': points}}


def fx_vol_block(currency, atm, expiries=(0.25, 0.5, 1.0, 2.0, 5.0)):
    """An `FXVolPrices` block: one ATM quote per expiry and a token smile on two delta pillars. The
    smile is small and symmetric on purpose - the composition reads only the ATM column - so the
    wings make the block complete rather than being priced against.

    EVERY ROW CARRIES A `Timestamp`, which is not optional in practice: `FXVolSurfaceParameters`
    reads `point['Timestamp']` by direct subscript, so a block without one raises `KeyError` from
    inside the surface build rather than refusing by name. A finding; authored around here.
    """
    points = []
    for expiry in expiries:
        points.append({'Use': 'Yes', 'Expiry': expiry, 'Pillar': 0.0, 'Quote_Type': 'ATM',
                       'Quoted_Market_Value': atm, 'Timestamp': BASE})
        for pillar in (0.1, 0.25):
            points.append({'Use': 'Yes', 'Expiry': expiry, 'Pillar': pillar, 'Timestamp': BASE,
                           'Quote_Type': 'RR', 'Quoted_Market_Value': 0.01})
            points.append({'Use': 'Yes', 'Expiry': expiry, 'Pillar': pillar, 'Timestamp': BASE,
                           'Quote_Type': 'BF', 'Quoted_Market_Value': 0.002})
    return {'instrument': {
        'Currency': currency, 'Delta_Type': 'Forward', 'Premium_Adjusted': 'Yes',
        'ATM_Convention': 'Delta_Neutral_Straddle', 'Grid_Tolerance': 0.0001,
        'Quote_Sensitivity': 'No', 'Points': points}}


def swaption_block(surface, ladder=LADDER):
    """A `HullWhite2FactorModelPrices` block quoting NORMAL vols. `Objective` IS ABSENT
    deliberately: a block that spelled the declared default would stop reading it the day it moved.
    The rows are 3M/3M ACT_365 weighted flat, which is what the live emitter writes.
    """
    return {'instrument': {
        'Swaption_Volatility': surface,
        'Quote_Source': 'authored - a four-cell NORMAL ladder for the composition gate',
        'Instrument_Definitions': [
            {'Start': pd.DateOffset(**H._period(start[0])),
             'Tenor': pd.DateOffset(**H._period(tenor[0])),
             'Floating_Frequency': pd.DateOffset(months=3),
             'Fixed_Frequency': pd.DateOffset(months=3),
             'Floating_Day_Count': 'ACT_365', 'Fixed_Day_Count': 'ACT_365',
             'Market_Volatility': utils.Percent(vol), 'Weight': 1.0}
            for start, tenor, vol in ladder]}}


def authored_blocks():
    """The whole small world as `Market Prices` blocks, keyed as the section keys them."""
    return {
        # FLAT, at 4% - see the module docstring on why the base curve is flat
        'InterestRatePrices.USD': rates_block('USD', 'USD', (4.0, ((1, 4.0), (2, 4.0), (3, 4.0),
                                                                  (5, 4.0), (10, 4.0)))),
        'InterestRatePrices.ZAR': rates_block('ZAR', 'ZAR', (7.0, ((1, 7.35), (2, 7.45),
                                                                  (3, 7.50), (5, 7.70),
                                                                  (10, 8.10)))),
        'FXVolPrices.USD.ZAR': fx_vol_block('ZAR', FX_VOL),
        'HullWhite2FactorModelPrices.ZAR': swaption_block('ZAR-SWAPTION')}


# ---------------------------------------------------------------------------------------------
# THE FIXTURES - one fit and one pair of runs for the whole file
# ---------------------------------------------------------------------------------------------

@pytest.fixture(scope='module')
def world():
    """Half A on the authored world: both curves solved, the FX surface built, the ladder fitted.
    Module-scoped because the fit is the expensive half and every gate reads the same theta*.
    """
    built = H.build_and_fit(
        BASE, authored_blocks(),
        ir_market_prices=['InterestRatePrices.USD', 'InterestRatePrices.ZAR'],
        swaption_market_price='HullWhite2FactorModelPrices.ZAR',
        vol_surface_name='ZAR-SWAPTION', swaption_currency='ZAR',
        fx_currency=CURRENCIES['rate'], ir_curve='ZAR', fx_spot_base_per_unit=FX_SPOT,
        base_currency=CURRENCIES['base'], base_curve='USD', rho_quote=RHO,
        fxvol_market_price='FXVolPrices.USD.ZAR')
    built['meta'] = {'base_currency': CURRENCIES['base'], 'fx_currency': CURRENCIES['rate'],
                     'ir_curve': 'ZAR',
                     'swaption_market_price': 'HullWhite2FactorModelPrices.ZAR'}
    return built


@pytest.fixture(scope='module')
def legs(world):
    """The composition's par forward swaps, off the solved curve and checked against the coupon
    `create_market_swaps` wrote into the benchmark's leg."""
    return H.benchmark_legs(world['closure'], CELLS, pd.DateOffset(months=3), 'ACT_365')


def _reading(world, legs, rho, **mutation):
    return H.composition_reading(world, legs, FX_SPOT, GRID, BATCH, BATCHES, SEED, rho,
                                 **mutation)


@pytest.fixture(scope='module')
def composed(world, legs):
    """`{label: reading}` - the correlated world, its rho = 0 twin under the SAME seed, and the two
    mutants. One set of runs for every gate below."""
    return {'rho': _reading(world, legs, RHO),
            'zero': _reading(world, legs, 0.0),
            'no_K': _reading(world, legs, RHO, suppress_quanto_drift=True),
            'flipped': _reading(world, legs, RHO, fx_axis_sign=-1.0)}


# ---------------------------------------------------------------------------------------------
# HALF A
# ---------------------------------------------------------------------------------------------

def test_both_curve_solves_reprice_their_own_quotes(world):
    """A bootstrap's only honest gate is that its benchmarks come back at par on the curve it wrote,
    to 1e-6 bp. The residual is reported as a PV and as the par-rate move that would close it; the
    second is the scale-free one."""
    for name, reading in world['curves'].items():
        assert reading['max_abs_bp'] < 1e-6, (
            '{}: worst benchmark is {:.3e} bp from par on the solved curve'.format(
                name, reading['max_abs_bp']))


def test_the_normal_ladder_fits_through_the_declared_analytic_default(world):
    """`Objective` is ABSENT, so this runs whatever the family declares - the Schrager-Pelsser
    normal-vol residual. Two readings: the fit rms is under 0.5bp in NORMAL VOL (an
    under-determined ladder reaches its quotes, so that is a tolerance on the closed form rather
    than on the fit), and the honesty reprice was reported.
    """
    assert 'Objective' not in world['config'].params['Market Prices'][
        'HullWhite2FactorModelPrices.ZAR']['instrument']
    assert world['readings']['n'] == len(LADDER)
    assert world['readings']['rms_bp'] < 0.5, (
        'fit rms {:.4f} bp over {} benchmarks'.format(
            world['readings']['rms_bp'], world['readings']['n']))
    assert world['fit']['honesty'] is not None, (
        'the analytic solve reported no honesty reprice - the Monte Carlo auditor did not run')


def test_the_emitted_factor_carries_the_quanto_it_was_never_fitted_with(world):
    """`save_params` writes the three quanto fields off the UN-suppressed implied object while the
    objective priced on the suppressed twin, so the block the simulator reads carries a drift the
    fit never saw - the whole calibrate-domestic/simulate-global architecture in one price factor.

    THE SIGN IS NOT ASSERTED PER COMPONENT and must not be: `get_quanto_correlation` projects onto
    each factor as `C (s_i + rho s_j) / D`, and one of those brackets can change sign on its own -
    it does here. What IS a sign statement is that the pair is LINEAR in C, so quoting the
    correlation the other way round negates both exactly.
    """
    param = world['fit']['param']
    assert param['Quanto_FX_Volatility'] is not None
    assert param['Quanto_FX_Correlation_1'] and param['Quanto_FX_Correlation_2']
    flipped = H.reemit_quanto(param, CURRENCIES['base'], CURRENCIES['rate'], -RHO)
    for name in ('Quanto_FX_Correlation_1', 'Quanto_FX_Correlation_2'):
        assert flipped[name] == pytest.approx(-float(param[name]), rel=1e-12), name


@pytest.mark.parametrize('label,kwargs', [
    ('rho = 0', dict(rho_quote=0.0)),
    ('no FX factor', dict(rho_quote=None, drop_fx_factor=True))])
def test_the_fit_does_not_move_with_the_fx_inputs(world, label, kwargs):
    """THE FIT'S OWN INVARIANCE, BIT-IDENTICAL rather than close. `implied_process` builds the
    objective's process on a twin with the quanto FX vol and the FX/IR correlation SUPPRESSED, so
    `K = 0` and the correlation reaches nothing the residual reads - two solves over the same
    sample are the same arithmetic. A tolerance here would pass on a world where the correlation
    leaked in at the eighth decimal.
    """
    reference = H.theta_vector(world['fit']['param'])
    other = H.theta_vector(H.refit_at(world, **kwargs)['param'])
    assert np.array_equal(reference, other), (
        '{}: theta* moved by {:.3e}'.format(label, float(np.abs(reference - other).max())))


# ---------------------------------------------------------------------------------------------
# HALF B
# ---------------------------------------------------------------------------------------------

def test_the_per_set_profiles_are_the_reported_table(composed):
    """The reading is `Credit_Monte_Carlo`'s own output or it is nothing: every netting set's
    `Calc_res['Value']` summed back must BE `Results['mtm']`, to the bit."""
    for label, reading in composed.items():
        assert reading['run']['sum_check'] == 0.0, (
            '{}: the netting sets do not sum to the reported mtm ({:.3e})'.format(
                label, reading['run']['sum_check']))


def test_the_swap_is_the_swaptions_own_underlying(legs):
    """A swap struck a basis point off the analytic price's ATM rate would miss identity 1 for a
    reason unrelated to the measure, so the strike rebuilt off the curve is held to 1e-10 against
    the coupon `create_market_swaps` wrote into the benchmark's float leg."""
    for name, leg in legs.items():
        assert leg['strike_check'] < 1e-10, name


def test_the_numeraire_identity_holds_before_any_payoff(composed):
    """One unit of the rate currency at T is a TRADABLE worth `X_0 P_zar(0,T)` today, so
    `E[D_usd(0,T) X_T]` is that number with no option anywhere in it - the half of identity 1 that
    is pure plumbing, gated separately because a miss here is not a measure error. The authored
    world makes it small: a FLAT base curve, so a static curve's frozen roll is exact, and short
    expiries. The band is `NUMERAIRE_SIGMA` times the reading's own `se_rel`.
    """
    for label in ('rho', 'zero'):
        for name, row in composed[label]['numeraire'].items():
            assert abs(row['numeraire_rel']) < NUMERAIRE_SIGMA * row['se_rel'], (
                '{} {}: E[D_base X_T] is {:+.3f}% from X_0 D_rate(0,T) - {:.2f} sigma '
                '(se {:.3f}%)'.format(label, name, 100.0 * row['numeraire_rel'],
                                      row['numeraire_rel'] / row['se_rel'],
                                      100.0 * row['se_rel']))


def test_the_composition_closes_on_the_models_own_domestic_price(composed):
    """IDENTITY 1: the USD-deflated, FX-converted expected positive exposure at the expiry row
    against spot times the fitted model's own Schrager-Pelsser price. Both sides are the same
    theta*, so this is not a question about the market - it is the test of `K`. The band is the
    run's own standard error times `IDENTITY_SIGMA`, and the mutation gates say it is tight enough.
    """
    for label in ('rho', 'zero'):
        for name, row in composed[label]['rows'].items():
            assert abs(row['sigma']) < IDENTITY_SIGMA, (
                '{} {}: deflated EPE {:.10f} against X0 x SP {:.10f} - {:+.3f}%, {:.2f} sigma '
                '(se {:.3f}%)'.format(
                    label, name, row['deflated_epe'], row['spot_x_sp'], 100.0 * row['miss_rel'],
                    row['sigma'], 100.0 * row['deflated_epe_se'] / row['spot_x_sp']))


def test_the_rho_pair_agrees_under_common_random_numbers(composed):
    """Both correlations price the SAME domestic swaption, at one `Random_Seed`, which makes this a
    common-random-numbers comparison rather than a difference of two independent means. NOT
    bit-identical and not asserted to be: the correlation matrix changes the cholesky, so the same
    normals correlate differently. What has to hold is that both land on the same price.
    """
    left, right = composed['rho']['rows'], composed['zero']['rows']
    for name in left:
        spread = np.hypot(left[name]['deflated_epe_se'], right[name]['deflated_epe_se'])
        assert abs(left[name]['deflated_epe'] - right[name]['deflated_epe']) < \
            IDENTITY_SIGMA * spread, (
            '{}: rho = {:g} prices {:.10f} and rho = 0 prices {:.10f} ({:+.3f}%)'.format(
                name, RHO, left[name]['deflated_epe'], right[name]['deflated_epe'],
                100.0 * (left[name]['deflated_epe'] / right[name]['deflated_epe'] - 1.0)))


def test_suppressing_the_quanto_drift_breaks_the_identity(composed):
    """THE MUTATION, a DOCUMENT rather than a patch: `Quanto_FX_Correlation_1/2` authored to zero
    on the emitted block while the `Correlations` section keeps the correlated Brownians - a world
    where FX and rates move together and the measure change is missing, which is what a wrong `K`
    is and what a desk could author by hand. The kill must exceed the band the clean gate passes
    inside.
    """
    clean, dirty = composed['rho']['rows'], composed['no_K']['rows']
    for name in clean:
        se = clean[name]['deflated_epe_se'] / clean[name]['spot_x_sp']
        moved = abs(dirty[name]['miss_rel'] - clean[name]['miss_rel'])
        assert moved > IDENTITY_SIGMA * se, (
            '{}: suppressing K moved identity 1 by only {:+.3f}% ({:.1f} sigma) - this world '
            'cannot tell a broken drift from noise'.format(name, 100.0 * moved, moved / se))
        assert abs(dirty[name]['sigma']) > IDENTITY_SIGMA, (
            '{}: identity 1 still closes with the quanto drift suppressed'.format(name))


def test_the_fx_axis_is_base_per_unit(composed):
    """THE OTHER MUTATION: the FX factor authored on the SCREEN's axis instead of the engine's
    (`FxRate.Spot` is the spot rate IN BASE CURRENCY). A two-hundred-fold error that prices
    perfectly happily, which is why it is gated rather than commented."""
    clean, dirty = composed['rho']['rows'], composed['flipped']['rows']
    for name in clean:
        assert abs(dirty[name]['sigma']) > 10.0 * IDENTITY_SIGMA, name
        assert dirty[name]['miss_rel'] > 10.0, (
            '{}: the flipped axis moved identity 1 by only {:+.1f}%'.format(
                name, 100.0 * dirty[name]['miss_rel']))


def test_the_quanto_correlations_change_basis_on_the_way_to_the_correlations_section(world):
    """The drift's rho-bar and the covariance's rows are stated in TWO DIFFERENT BASES and nothing
    in the library changes between them (`quanto_correlations`). The invariant: the pair `(a, b)`
    the section is authored with satisfies `a^2 + b^2 = C^2`, which is what says the FX Brownian
    keeps exactly `sqrt(1 - C^2)` of its own independent part. Copying rho-bar_2 into the section
    directly does not, and that world has a drift and a covariance that disagree silently.
    """
    param = world['fit']['param']
    first, second, cross = H.quanto_correlations(param)
    assert cross == 0.0, 'the process applies its own rho inside delta_CtT; correlating the two ' \
                         'raw rows as well applies it twice'
    assert first == pytest.approx(float(param['Quanto_FX_Correlation_1']), rel=1e-12)
    assert first ** 2 + second ** 2 == pytest.approx(RHO ** 2, rel=1e-8), (
        'a^2 + b^2 is {:.10f}, not C^2 = {:.10f}'.format(
            first ** 2 + second ** 2, RHO ** 2))


def test_the_composition_reports_its_own_readings(composed, capsys):
    """Not an assertion - the readings, printed, so a reader deciding whether a band is honest has
    the numbers it was computed from. `capsys.disabled()` puts them on the terminal rather than in
    pytest's capture buffer, which only surfaces on a failure: the point is to read them GREEN.
    """
    with capsys.disabled():
        for label, reading in composed.items():
            H.print_identity_table(label, reading['rows'], reading['wiring'], reading['run'],
                                   reading['numeraire'])
            H.print_numeraire_table(reading['numeraire'])
