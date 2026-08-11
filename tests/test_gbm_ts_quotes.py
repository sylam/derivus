"""Does one backward pass report `dV/d(ATM vol)`, and is the map it goes through the right one?

Increment 3 of [Quote Sensitivities](../docs_src/developer/quote_sensitivities.md), and it is the
increment with no solver in it. `GBMAssetPriceTSModelParameters` turns the ATM column of a vol
surface into the integrated vol curve a risk-neutral process reads, and that map is CLOSED FORM -
total variances, a forward-variance walk, and a quadratic root - so there is no fixed point to
differentiate implicitly and no stationarity to check. Autograd walks the expression.

That makes the triangle increments 1 and 2 could not close available here:

    one-pass dV/dq      the number under test - `q.grad` after a single backward()
    dV/dtheta . J       the greek the engine already reported, contracted against the quote
                        Jacobian of the map
    central FD          J itself against a re-bootstrap at q +- h, on an h-ladder that has to
                        converge as h^2

TWO THINGS ABOUT THIS MAP HAVE TO BE SAID OUT LOUD, because a gate written without them is a
placebo.

  IT IS THE IDENTITY WHERE VARIANCE RISES. Only the integrated vol is written; the instantaneous
  vol the walk solves for is its own state and is never published. So on a column implying rising
  variance the written curve is `sqrt(q^2 t / t)` - the quote column back, up to the rounding of a
  square and its root, and EXACTLY it on the fixtures here rather than as a property of the map:
  the round trip returns a different last bit on a bit over half of random columns. Either way a
  round trip on such a fixture passes whatever the walk does, and the whole increment would be
  gated on nothing. THE FIXTURES HERE DECLINE - `DECLINING` repairs at 2y - and the rising column
  is kept only to pin the identity as the property it is.

  IT IS PIECEWISE, AND THE SWITCH IS A KINK. Where the repair fires the written vol is a floor
  that does not involve that expiry's own quote at all, so `d/dq` drops from 1 to 0 across the
  switch and the two one-sided derivatives are both right. Both branches are exercised, and the
  kink is measured from each side rather than averaged across it.

AND THE TAPE IS NOT THE VALUE. `integrated_vol` is the numpy walk this family has always shipped
and `carried_vol` is a torch twin spliced in for its derivative alone, which is why the gates below
compare them at ONE ULP rather than for equality: `torch.sqrt` is a different implementation of a
correctly-rounded operation and is one ulp low on better than one input in a hundred here. Marks do
not go through it - that is what the splice buys - so the tolerance is a DIAGNOSTIC on the twin
rather than a slackened equality on a shipped number.

Run: ``pytest tests/test_gbm_ts_quotes.py -q``
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import pytest
import torch

import derivus
from derivus import utils
from derivus.bootstrappers import FXVolSurfaceParameters, GBMAssetPriceTSModelParameters
from derivus.config import Config
from derivus.instruments import construct_instrument

BASE = pd.Timestamp('2024-06-28')
DTYPE = torch.float64
BLOCK = 'GBMAssetPriceTSModelPrices.EQ'
PARAMS = 'GBMAssetPriceTSModelParameters.EQ'
VOL_FACTOR = utils.Factor('GBMAssetPriceTSModelParameters', ('EQ', 'Vol'))

#: The surface's expiries, and the ATM column read off it at moneyness 1. `DECLINING` implies a
#: FALLING forward variance between 1y and 2y - 0.22^2 x 1 = 0.0484 against 0.14^2 x 2 = 0.0392 -
#: so the repair fires at 2y and the map is not the identity there. That is the fixture every
#: derivative gate below runs on.
EXPIRY = (0.25, 1.0, 2.0, 3.0)
DECLINING = (0.20, 0.22, 0.14, 0.20)
RISING = (0.20, 0.22, 0.24, 0.26)
#: A column repairing THREE STEPS IN A ROW, on its own five-point expiry grid. One repair leaves the
#: walk's instantaneous vol at exactly zero, so the second reaches a discriminant of exactly zero,
#: and the third is what pulls a gradient back through it - which is where `sqrt` has an infinite
#: derivative and the whole Jacobian used to come back NaN.
HUMP_EXPIRY = (0.25, 1.0, 2.0, 3.0, 4.0)
HUMP = (0.10, 0.16, 0.10, 0.10, 0.10)
#: A skew, so the surface is not flat in moneyness and an ATM read off the wrong column would be a
#: different number. Moneyness 1 is a NODE, so `np.interp` returns it exactly.
SKEW = {0.8: 0.02, 1.0: 0.0, 1.2: -0.01}

TIME_GRID = '0d 6m(6m)'
DEAL = {'Object': 'EquityOptionDeal', 'Reference': 'OPT1', 'Currency': 'USD',
        'Payoff_Currency': 'USD', 'Equity': 'EQ', 'Dividends': 'EQ', 'Discount_Rate': 'USD',
        'Equity_Volatility': 'EQ', 'Buy_Sell': 'Buy', 'Option_Type': 'Call', 'Units': 1.0,
        'Strike_Price': 105.0, 'Expiry_Date': BASE + pd.DateOffset(years=3)}
NETTING = {'Object': 'NettingCollateralSet', 'Reference': 'NS1', 'Agreement_Currency': 'USD',
           'Apply_Closeout_When_Uncollateralized': 'No', 'Balance_Currency': 'USD',
           'Opening_Balance': 0.0, 'Collateralized': 'False', 'Netted': 'True', 'Calendars': None}


# =====================================================================================
# the pre-change numpy walk, verbatim, as the parity oracle. NOT engine code.
# =====================================================================================

def oracle_walk(atm_vol, expiry, repair=True):
    """The loop `integrated_vol` replaced, copied out of the shipped bootstrapper.

    Returns the integrated curve the factor is written from AND the instantaneous vols the walk
    solves for on the way, which the engine never publishes - the second is only here because it is
    where an unrepaired column goes wrong.

    `repair=False` is the MUTATION: the same walk with the declining-variance branch removed, which
    is what the parity gate has to be able to tell apart from the real one.
    """
    atm_vol = list(atm_vol)
    expiry = np.asarray(expiry, dtype=float)
    dt = np.diff(np.append(0, expiry))
    var = expiry * np.array(atm_vol) ** 2
    sig = atm_vol[:1]
    vol = atm_vol[:1]
    var_tm1 = var[0]

    for var_t, delta_t, t_i in zip(var[1:], dt[1:] / 3.0, expiry[1:]):
        M = var_tm1 + delta_t * (sig[-1] ** 2)
        if var_t < M and repair:
            var_t = M

        a = delta_t
        b = sig[-1] * delta_t
        c = M - var_t

        sig.append((-b + np.sqrt(b * b - 4.0 * a * c)) / (2.0 * a))
        vol.append(np.sqrt(var_t / t_i))
        var_tm1 = var_t

    return np.array(vol), np.array(sig)


# =====================================================================================
# the worlds
# =====================================================================================

def surface(atm):
    """A `VolatilityGrid` whose moneyness-1 column is `atm` - the FALLBACK quote source."""
    return {'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness',
            'Surface': utils.Curve([], [[m, t, v + s] for m, s in SKEW.items()
                                        for t, v in zip(EXPIRY, atm)])}


def bootstrapped(connect, atm=DECLINING, bumps=()):
    """An equity world whose GBM TS parameters have been built from the surface's ATM column."""
    atm = [v + sum(h for j, h in bumps if j == i) for i, v in enumerate(atm)]
    config = Config()
    config.params['System Parameters']['Base_Currency'] = 'USD'
    config.params['System Parameters']['Base_Date'] = BASE
    config.params['Price Factors'] = {
        'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Priority': 1,
                       'Spot': 1.0},
        'InterestRate.USD': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                             'Curve': utils.Curve([], [[0.0, 0.02], [5.0, 0.03]])},
        'EquityPrice.EQ': {'Spot': 100.0, 'Currency': 'USD', 'Interest_Rate': 'USD', 'Issuer': '',
                           'Respect_Default': 'No', 'Jump_Level': 0.0},
        'DividendRate.EQ': {'Currency': 'USD', 'Floor': None,
                            'Curve': utils.Curve([], [[0.0, 0.01], [5.0, 0.01]])},
        'VolatilityGrid.EQ': surface(atm),
        'SurvivalProb.CPTY': {'Recovery_Rate': 0.4,
                              'Curve': utils.Curve([], [[0.0, 0.0], [10.0, 0.4]])}}
    config.params['Market Prices'] = {BLOCK: {'instrument': {
        'Asset_Price_Volatility': 'EQ',
        'Quote_Sensitivity': 'Yes' if connect else 'No'}, 'Children': []}}
    config.params['Bootstrapper Configuration'] = {'GBMAssetPriceTSModelParameters': {}}
    config.bootstrap()
    config.params['Model Configuration'].append(
        'EquityPrice', (), 'GBMAssetPriceTSModelImplied')
    config.deals = {'Attributes': {'Reference': 'test', 'Tag_Titles': ''},
                    'Deals': {'Children': [{
                        'Instrument': construct_instrument(NETTING, {}),
                        'Children': [{'Instrument': construct_instrument(DEAL, {})}]}]},
                    'Calculation': {'Base_Date': BASE, 'Currency': 'USD'}}
    return config


def written(config):
    """The integrated vol curve as `Price Factors` holds it - what a leaf is minted from."""
    return config.params['Price Factors'][PARAMS]['Vol'].array


def run(config, gradient_variables='All'):
    """One exposure run over the book, with the quote leaves cleared first - `.grad` accumulates."""
    for _, leaves in config.quote_leaves.values():
        leaves.grad = None
    _, out = derivus.run_cmc(config, prec=DTYPE, overrides={
        'Run_Date': BASE.strftime('%Y-%m-%d'), 'Time_grid': TIME_GRID, 'Batch_Size': 256,
        'Simulation_Batches': 1, 'Random_Seed': 1, 'Currency': 'USD', 'MCMC_Simulations': 0,
        'Tenor_Offset': 0.0, 'Deflation_Interest_Rate': 'USD',
        'Gradient_Variables': gradient_variables,
        'Credit_Valuation_Adjustment': {
            'Calculate': 'Yes', 'Counterparty': 'CPTY', 'Deflate_Stochastically': 'No',
            'Stochastic_Hazard_Rates': 'No', 'Gradient': 'Yes'}})
    return out['Results']


def factor_grad(results, name=PARAMS + '.Vol'):
    frame = results['grad_cva']
    return frame.xs(name, level='Rate')['Gradient'].values.astype(np.float64)


def jacobian(atm=DECLINING, expiry=EXPIRY):
    """`d(written curve)/d(ATM column)` by autograd on the twin the splice carries."""
    quotes = torch.tensor(atm, dtype=torch.float64, requires_grad=True)
    return torch.autograd.functional.jacobian(
        lambda q: GBMAssetPriceTSModelParameters.carried_vol(
            q, np.array(expiry, dtype=float)), quotes).numpy()


# =====================================================================================
# (i) the forward - nothing about a value moves when the quote side is switched on
# =====================================================================================

def test_the_written_curve_is_bit_identical_with_quote_gradients_on_and_off():
    """`np.array_equal`, not a tolerance - and STRUCTURAL rather than lucky, because the numbers do
    not come out of the tape at all. The shipped walk writes the curve either way and the torch twin
    is spliced in worth zero, so the switch cannot reach a value. The mutation shows the comparison
    can still fail, because a moved quote moves the curve."""
    connected = bootstrapped(True)
    assert np.array_equal(written(bootstrapped(False)), written(connected))
    assert not np.array_equal(written(connected),
                              written(bootstrapped(True, bumps=[(1, 1e-4)]))), (
        'a basis point on the 1y ATM vol did not reach the written curve')
    # and the tensor a calculation splices in stands for exactly the factor it is offered against
    assert np.array_equal(connected.calibrated_factors[VOL_FACTOR].detach().numpy(),
                          written(connected)[:, 1])


def test_a_reference_run_is_bit_identical_with_quote_gradients_on_and_off():
    """The whole job, not just the block: the CVA, the exposure profile and the WHOLE reported
    gradient frame are the numbers they always were. dV/dq arrives in that same pass."""
    plain, connected = run(bootstrapped(False)), run(bootstrapped(True))
    assert plain['cva'] == connected['cva']
    assert np.array_equal(plain['mtm'].values, connected['mtm'].values)
    assert np.array_equal(plain['grad_cva']['Gradient'].values,
                          connected['grad_cva']['Gradient'].values)


def test_a_block_that_did_not_ask_leaves_nothing_behind():
    """`Quote_Sensitivity` is the switch, and off it costs a config nothing to carry."""
    plain = bootstrapped(False)
    assert not plain.calibrated_factors and not plain.quote_leaves
    connected = bootstrapped(True)
    assert set(connected.calibrated_factors) == {VOL_FACTOR}
    descriptors, leaves = connected.quote_leaves[BLOCK]
    assert descriptors == ['ATM 0.25', 'ATM 1', 'ATM 2', 'ATM 3']
    # ONE VECTOR LEAF per block, which is the curve family's shape rather than the swaption
    # family's tuple of scalars - the whole ATM column enters one map
    assert isinstance(leaves, torch.Tensor) and leaves.shape == (len(EXPIRY),)
    assert leaves.grad is None, 'the leaf was handed over dirty'


# =====================================================================================
# (ii) the map - against the loop it replaced, and against the identity it reduces to
# =====================================================================================

@pytest.mark.parametrize('atm', [
    DECLINING, RISING, (0.30, 0.15, 0.16, 0.17), (0.30, 0.15, 0.10, 0.30), (0.20,)])
def test_the_shipped_walk_is_bit_for_bit_the_numpy_loop_it_always_was(atm):
    """The VALUE path, against the loop copied out of the bootstrapper before this increment.

    `integrated_vol` is that loop - same operations, same order, same numpy - so this is equality
    and any difference at all is a finding. The fourth fixture repairs TWICE IN A ROW and the fifth
    is a single expiry, where the loop never runs at all.
    """
    expiry = np.array(EXPIRY[:len(atm)], dtype=float)
    curve, _ = GBMAssetPriceTSModelParameters.integrated_vol(list(atm), expiry)
    assert np.array_equal(np.array(curve), oracle_walk(atm, expiry)[0])


@pytest.mark.parametrize('atm', [
    DECLINING, RISING, (0.30, 0.15, 0.16, 0.17), (0.30, 0.15, 0.10, 0.30), (0.20,)])
def test_the_torch_twin_agrees_with_the_shipped_walk_to_one_ulp_and_no_further(atm):
    """The DIAGNOSTIC that says why there are two walks rather than one.

    Every operation here is `+ - * /` and `sqrt`, and IEEE-754 requires all of them to be correctly
    rounded - so the tempting conclusion is that torch and numpy cannot disagree and the twin could
    simply BE the shipped walk. They do disagree: `torch.sqrt` is one ulp below `np.sqrt` on better
    than one float64 in a hundred on this box, and a torch expression tree re-associates besides.
    An ulp of a shipped vol is a different number in a report, so the twin is spliced in for its
    derivative and never for its value, and what is asserted here is the SIZE of a difference that
    reaches nothing rather than the absence of one.
    """
    expiry = np.array(EXPIRY[:len(atm)], dtype=float)
    shipped = np.array(GBMAssetPriceTSModelParameters.integrated_vol(list(atm), expiry)[0])
    carried = GBMAssetPriceTSModelParameters.carried_vol(
        torch.tensor(atm, dtype=torch.float64, requires_grad=True), expiry).detach().numpy()
    assert np.abs(carried - shipped).max() <= np.spacing(shipped).max()


def test_the_parity_oracle_can_tell_a_broken_walk_apart():
    """MUTATE the oracle. On a RISING column the repair never fires, so the parity gate above would
    pass on a bootstrapper that had lost the branch entirely - which is why the declining fixture is
    the one everything here runs on.

    WHAT THE REPAIR IS ACTUALLY FOR is visible in the same comparison. Without it the declining step
    has no real instantaneous vol: the discriminant goes negative and `sigma` is NaN from there on.
    The WRITTEN curve does not show it - unrepaired it is just the quote column back again - so the
    damage lands on the process that reads the curve rather than on the factor a gate would look at.
    """
    assert np.array_equal(oracle_walk(RISING, np.array(EXPIRY))[0],
                          oracle_walk(RISING, np.array(EXPIRY), repair=False)[0])

    curve, sigma = oracle_walk(DECLINING, np.array(EXPIRY), repair=False)
    assert np.isnan(sigma).any() and not np.isnan(curve).any()
    assert np.array_equal(curve, np.array(DECLINING))
    assert not np.array_equal(oracle_walk(DECLINING, np.array(EXPIRY))[0], curve)


def test_the_map_is_the_identity_where_forward_variance_rises():
    """The round trip, and the reason it is not enough on its own.

    Author a column off a known integrated-vol curve and bootstrap it: only sigma_bar is written, so
    on a rising column the curve that comes back is the column - up to the rounding of a square and
    its root, `sqrt(q^2 t / t)`, which is EXACTLY `q` here and on most columns but not on all of
    them. So the equality below is a statement about this fixture and not about the map, and it is
    why the same round trip on the declining fixture has to be a different number, which is asserted
    here rather than left implied.
    """
    assert np.array_equal(written(bootstrapped(True, atm=RISING))[:, 1], np.array(RISING))
    recovered = written(bootstrapped(True, atm=DECLINING))[:, 1]
    assert not np.array_equal(recovered, np.array(DECLINING))
    assert (recovered[[0, 1, 3]] == np.array(DECLINING)[[0, 1, 3]]).all(), (
        'the repair reached an expiry whose own variance was never declining')


def test_the_written_curve_satisfies_the_simpson_identity_it_documents():
    """The walk's own algebra, checked by inverting it rather than by re-running it.

    The class documents `V(t_i) - V(t_{i-1}) = (dt/3)(s_{i-1}^2 + s_{i-1} s_i + s_i^2)` for the
    instantaneous vol `s`. `s` is never published, so it is recovered here from the WRITTEN
    integrated curve by solving that quadratic forwards - an independent inversion - and the
    identity is then required to close on the variances the curve implies. It also has to have a
    real, non-negative root at every step, which is exactly what the repair exists to guarantee.
    """
    curve = written(bootstrapped(True))[:, 1]
    expiry = np.array(EXPIRY)
    variance = curve ** 2 * expiry
    assert (np.diff(variance) >= 0).all(), 'the repaired curve still declines in variance'

    third = np.diff(np.append(0.0, expiry)) / 3.0
    sigma = [curve[0]]
    for i in range(1, expiry.size):
        disc = (sigma[-1] * third[i]) ** 2 - 4.0 * third[i] * (
            variance[i - 1] + third[i] * sigma[-1] ** 2 - variance[i])
        assert disc >= 0, 'no real instantaneous vol over step {}'.format(i)
        sigma.append((-sigma[-1] * third[i] + np.sqrt(disc)) / (2.0 * third[i]))
        assert sigma[-1] >= 0, 'a negative instantaneous vol over step {}'.format(i)
        integrated = variance[i - 1] + third[i] * (
            sigma[-2] ** 2 + sigma[-2] * sigma[-1] + sigma[-1] ** 2)
        assert integrated == pytest.approx(variance[i], rel=1e-14, abs=1e-16)


# =====================================================================================
# (iii) the quote Jacobian against a central difference of the map
# =====================================================================================

def test_the_quote_jacobian_is_the_central_difference_of_the_bootstrap():
    """AAD against a re-bootstrap at q +- h, on an h-ladder that has to converge as h^2.

    The finite difference goes through the WHOLE family - re-authored surface, re-read ATM column,
    re-walked - so what is compared is the derivative of the thing the job runs, not of the closure
    the derivative was taken on. The interesting row is the repaired one: rows 0, 1 and 3 are the
    identity and their difference quotients are exact at every h, so the ladder is scored on the
    row where the map has curvature.
    """
    J = jacobian()
    errors = []
    for h in (1e-2, 1e-3, 1e-4):
        fd = np.column_stack([
            (written(bootstrapped(True, bumps=[(j, h)]))[:, 1] -
             written(bootstrapped(True, bumps=[(j, -h)]))[:, 1]) / (2.0 * h)
            for j in range(len(EXPIRY))])
        # the identity rows carry no h^2 term at all, so what is left in them is the difference
        # quotient's own rounding - it GROWS as h shrinks, which is why they are not on the ladder
        assert np.abs(fd[[0, 1, 3]] - J[[0, 1, 3]]).max() < 1e-12
        errors.append(np.abs(fd[2] - J[2]).max())

    assert errors[0] > 1e-5, 'the ladder measured nothing - is the repaired row constant?'
    for coarse, fine in zip(errors, errors[1:]):
        assert 50.0 < coarse / fine < 200.0, (
            'the error fell by {:.0f}x for a 10x smaller h, which is not h^2: {:.3g} then '
            '{:.3g}'.format(coarse / fine, coarse, fine))


def test_the_jacobian_has_the_shape_the_repair_forces():
    """What the numbers MEAN, pinned so a plausible-looking Jacobian cannot replace them.

    A repaired expiry's vol is a floor built out of everything BEFORE it, so its own column is zero
    and the earlier ones are not - and the earliest is NEGATIVE, because more variance spent early
    leaves the step a smaller instantaneous vol to floor at. Every other row is a unit vector.
    """
    J = jacobian()
    assert np.array_equal(J[[0, 1, 3]], np.eye(4)[[0, 1, 3]])
    assert J[2, 2] == 0.0 and J[2, 3] == 0.0
    assert J[2, 0] < 0.0 < J[2, 1]
    assert J[2, 0] == pytest.approx(-0.33638672, rel=1e-8)
    assert J[2, 1] == pytest.approx(1.15311268, rel=1e-8)


# =====================================================================================
# (iv) the kink - both branches, and a one-sided derivative from each side
# =====================================================================================

def switching_column(offset):
    """The declining fixture with 2y moved to `offset` away from the switch.

    At the switch the 2y quote is exactly the one whose own variance equals the floor the walk would
    impose. Above it the map is smooth and the written vol IS the quote; below it the floor bites.
    """
    floor = written(bootstrapped(True))[2, 1]
    return DECLINING[:2] + (floor + offset,) + DECLINING[3:]


@pytest.mark.parametrize('offset,expected', [(1e-3, 1.0), (-1e-3, 0.0)])
def test_the_kink_is_one_sided_and_autograd_reports_the_branch_it_is_in(offset, expected):
    """The derivative of the 2y vol in the 2y quote is 1 just above the switch and 0 just below it.

    A central difference ACROSS the switch would report 0.5 and agree with neither branch, so each
    side is measured with a ONE-SIDED difference taken entirely inside its own branch - which is the
    only quotient a piecewise map has a limit for.
    """
    atm, h = switching_column(offset), 1e-6
    J = jacobian(atm)
    assert J[2, 2] == expected

    inside = np.sign(offset) * h
    one_sided = (written(bootstrapped(True, atm=atm, bumps=[(2, inside)]))[2, 1] -
                 written(bootstrapped(True, atm=atm))[2, 1]) / inside
    assert one_sided == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize('h', [1e-3, 1e-5])
def test_a_central_difference_across_the_switch_answers_neither_branch(h):
    """MUTATE the instrument, not the code: straddle the kink and the quotient reports the AVERAGE.

    Exactly 0.5, at every h, because one side moves with the quote and the other does not - so it
    converges to a number that is nobody's derivative. That is why the gate above is one-sided, and
    it is what a ladder built on a symmetric bump would have quietly reported here.
    """
    atm = switching_column(0.0)
    central = (written(bootstrapped(True, atm=atm, bumps=[(2, h)]))[2, 1] -
               written(bootstrapped(True, atm=atm, bumps=[(2, -h)]))[2, 1]) / (2.0 * h)
    assert central == pytest.approx(0.5, abs=1e-9)


def test_the_floored_branch_severs_that_quote_from_everything_downstream():
    """A repaired expiry's quote does not reach ANY written vol, not just its own.

    The floor is built out of the walk's state at the previous step, so the quote is not an input to
    it - and `sigma` is zero over the floored step, so the next step's floor does not carry it
    either. A whole zero COLUMN of the Jacobian is a stronger statement than a zero diagonal, and it
    is the one the repair actually makes.
    """
    assert np.array_equal(jacobian()[:, 2], np.zeros(4))
    assert not np.array_equal(jacobian(RISING)[:, 2], np.zeros(4)), (
        'the column is zero on a fixture that never repairs, so this gate sees nothing'
    )


def test_both_branches_are_reached_and_the_repair_is_reported():
    """The two branches are a fork in one loop, and a fixture exercising one is half a gate."""
    expiry = np.array(EXPIRY, dtype=float)
    _, floored = GBMAssetPriceTSModelParameters.integrated_vol(list(DECLINING), expiry)
    assert floored == [2.0], 'the declining fixture did not reach the repair'
    _, clean = GBMAssetPriceTSModelParameters.integrated_vol(list(RISING), expiry)
    assert clean == [], 'the rising fixture reached a repair it has no reason to'


def test_a_repair_after_a_repair_keeps_a_finite_jacobian():
    """THREE REPAIRS IN A ROW, which is where the discriminant reaches exactly zero.

    The floored branch cancels `c` to an exact zero, so `sigma` comes back exactly zero - and the
    NEXT floored step then has `b = 0` beside that same zero `c` and takes `sqrt` of nothing.
    Forward that is fine; backward, `sqrt` has an infinite derivative there, `d(b^2)/db` is zero
    beside it, and the NaN that product makes propagates to EVERY entry - the identity rows
    included, which is why this is a gate on the whole Jacobian rather than on the floored rows.
    The third repair is what pulls a gradient back through the second, so two in a row is not
    enough to see it.
    """
    def walk(column):
        return np.array(GBMAssetPriceTSModelParameters.integrated_vol(
            list(column), np.array(HUMP_EXPIRY, dtype=float))[0])

    _, floored = GBMAssetPriceTSModelParameters.integrated_vol(
        list(HUMP), np.array(HUMP_EXPIRY, dtype=float))
    assert floored == [2.0, 3.0, 4.0], 'the hump fixture did not repair three steps in a row'

    J = jacobian(HUMP, HUMP_EXPIRY)
    assert np.isfinite(J).all(), 'the guarded discriminant still puts a NaN on the tape'
    assert np.array_equal(J[[0, 1]], np.eye(5)[[0, 1]]), (
        'the rows above the repair are the identity and have to survive it intact')
    assert np.array_equal(J[:, 2:], np.zeros((5, 3))), 'a repaired quote reached a written vol'

    # and the recovered rows are the derivative of the SHIPPED walk, not of a different map
    for h, tol in ((1e-4, 3e-8), (1e-5, 3e-10)):
        fd = np.column_stack([(walk(HUMP + h * np.eye(5)[j]) -
                               walk(HUMP - h * np.eye(5)[j])) / (2.0 * h)
                              for j in range(len(HUMP))])
        assert np.abs(fd - J).max() < tol


# =====================================================================================
# (v) the attachment - dV/dq in one backward, beside an unchanged dV/dtheta
# =====================================================================================

def test_a_calculation_reports_dV_dq_beside_dV_dtheta_in_one_pass():
    """The whole chain, closed in value space: `dV/dq = J' dV/dtheta`.

    Both halves come out of the SAME backward pass - the factor greek off the reported frame, the
    quote delta off the leaf - and the contraction is the map's Jacobian, taken independently. The
    repaired 2y quote is severed, so its reported delta is exactly zero while its FACTOR delta is
    not: that pair is what says the attachment is carrying the map rather than relabelling a greek.
    """
    config = bootstrapped(True)
    results = run(config)
    theta_grad = factor_grad(results)
    quote_grad = config.quote_leaves[BLOCK][1].grad.numpy()

    assert np.allclose(quote_grad, jacobian().T @ theta_grad, rtol=1e-13, atol=1e-15)
    assert quote_grad[2] == 0.0 and theta_grad[2] != 0.0, (
        'the severed quote and the live factor it sits under are the same number'
    )
    assert not np.allclose(quote_grad, theta_grad), (
        'dV/dq is dV/dtheta, so the map contributed nothing and the gate is vacuous'
    )


def test_the_contraction_gate_fails_against_the_wrong_jacobian():
    """MUTATE the reference. Drop the repair from the map - the identity Jacobian a bootstrapper
    that had lost the branch would publish - and the contraction has to stop agreeing."""
    config = bootstrapped(True)
    theta_grad = factor_grad(run(config))
    quote_grad = config.quote_leaves[BLOCK][1].grad.numpy()
    assert not np.allclose(quote_grad, np.eye(len(EXPIRY)).T @ theta_grad, rtol=1e-3)


@pytest.mark.parametrize('gradient_variables,reported', [
    ('All', True), ('Implied', True), ('Factors', False)])
def test_the_quote_delta_needs_the_implied_leaves(gradient_variables, reported):
    """`Gradient_Variables` governs whether an implied model's parameters are leaves at all, and a
    quote of one is downstream of that switch. Under `Factors` the `Vol` leaf is never minted, so
    the quote delta is not small - it is ABSENT, which is the honest answer. Reporting a zero there
    would be the exact failure this workstream exists to prevent."""
    config = bootstrapped(True)
    run(config, gradient_variables)
    assert (config.quote_leaves[BLOCK][1].grad is not None) is reported


# =====================================================================================
# (vi) the quote source - which numbers a config's leaves ARE, and why
# =====================================================================================

def fx_quotes(atm):
    return [{'Use': 'Yes', 'Expiry': T, 'Pillar': 0.0, 'Quote_Type': 'ATM',
             'Quoted_Market_Value': v, 'Timestamp': ''} for T, v in zip(EXPIRY, atm)] + [
        {'Use': 'Yes', 'Expiry': T, 'Pillar': 0.25, 'Quote_Type': q, 'Quoted_Market_Value': w,
         'Timestamp': ''} for T in EXPIRY for q, w in (('RR', 0.02), ('BF', 0.004))]


def fx_world(connect, atm=DECLINING, quoted=True, between=None):
    """A ZAR world whose `FXVol` surface this same market data BUILT from ATM/RR/BF quotes.

    `quoted=False` drops the quote block after the surface is built, which is the same SURFACE
    reached as authored data - the fork the quote source turns on, with everything else held.
    `between` runs on the config in the seam between the two bootstraps, which is where a config
    that has DESYNCED - a surface and a quote block that no longer describe each other - is made.
    """
    config = Config(base_currency='ZAR')
    config.params['System Parameters']['Base_Date'] = BASE
    config.params['Price Factors'] = {
        'FxRate.ZAR': {'Domestic_Currency': None, 'Interest_Rate': 'ZAR', 'Priority': 1,
                       'Spot': 1.0}}
    config.params['Market Prices'] = {
        'FXVolPrices.USD.ZAR': {'instrument': {
            'Currency': 'ZAR', 'Delta_Type': 'Forward', 'Premium_Adjusted': 'Yes',
            'ATM_Convention': 'Delta_Neutral_Straddle', 'Grid_Tolerance': 1e-4,
            'Points': fx_quotes(atm)}, 'Children': []},
        'GBMAssetPriceTSModelPrices.ZAR': {'instrument': {
            'Asset_Price_Volatility': 'USD.ZAR',
            'Quote_Sensitivity': 'Yes' if connect else 'No'}, 'Children': []}}
    config.params['Bootstrapper Configuration'] = {'FXVolSurfaceParameters': {}}
    config.bootstrap()
    if not quoted:
        del config.params['Market Prices']['FXVolPrices.USD.ZAR']
    if between is not None:
        between(config)
    config.params['Bootstrapper Configuration'] = {'GBMAssetPriceTSModelParameters': {}}
    config.bootstrap()
    return config


def fx_written(config):
    return config.params['Price Factors']['GBMAssetPriceTSModelParameters.ZAR']['Vol'].array


def test_a_family_quoted_surface_is_integrated_off_its_own_atm_rows():
    """The preferred source. A `Malz` surface carries the ATM QUOTE as its ATM vol by construction -
    the +-0.5 label sits at the delta-neutral straddle strike - so the quotes are taken straight off
    the block that built it, exactly, rather than re-read off the grid they were solved onto."""
    curve = fx_written(fx_world(True, atm=RISING))
    assert np.array_equal(curve[:, 0], np.array(EXPIRY))
    assert np.array_equal(curve[:, 1], np.array(RISING))


def test_the_two_quote_sources_are_different_numbers_on_the_same_surface():
    """MUTATE the provenance, hold the surface. Read as authored data the ATM column is
    `np.interp` at moneyness 1 - and a `Malz` surface's axis is log(F/K), whose grid stops at 0.5,
    so that read lands on the LAST log-moneyness node and returns a WING. It is a defect of the
    read and it is named on `atm_column`; what this gate holds is that the preferred path is not
    cosmetic, because the two sources disagree by vol points rather than by rounding."""
    quoted = fx_written(fx_world(True, atm=RISING))[:, 1]
    authored = fx_written(fx_world(True, atm=RISING, quoted=False))[:, 1]
    assert np.abs(quoted - authored).max() > 1e-3, (
        'the two sources agree, so preferring one of them decides nothing'
    )
    assert np.array_equal(quoted, np.array(RISING))


def test_the_atm_rows_are_read_by_the_family_that_declares_them():
    """One reader for the `Quote_Type == 'ATM'` rule, on the class whose schema declares the column.
    `smile` builds the wings around the same dict, so the two cannot drift into disagreeing about
    what an ATM row is."""
    used = FXVolSurfaceParameters.used({'Points': fx_quotes(RISING) + [
        {'Use': 'No', 'Expiry': 5.0, 'Pillar': 0.0, 'Quote_Type': 'ATM',
         'Quoted_Market_Value': 9.9, 'Timestamp': ''}]})
    assert FXVolSurfaceParameters.atm_quotes(used) == dict(zip(EXPIRY, RISING))


def test_the_family_quoted_world_is_bit_identical_with_quote_gradients_on_and_off():
    """THE SAME PROPERTY AS THE EQUITY WORLD, ON THE PATH THAT HAS A SECOND SOURCE TO PICK FROM.

    `Quote_Sensitivity` cannot move a mark - and the equity fixture reads its column off the surface
    whether the switch is on or not, so it cannot see a bootstrapper that chose its SOURCE by the
    switch. Here the two sources are different numbers by vol points, so a source picked on `connect`
    lands as a different curve and this equality is the one that says it was not.
    """
    assert np.array_equal(fx_written(fx_world(False)), fx_written(fx_world(True)))


def strip_the_fingerprint(config):
    """The written surface as data nobody's bootstrapper made: the `Grid_Tolerance` the grid was
    refined at is what this family stamps on what it writes, so a surface without it is authored -
    and every vol is moved so reading the quotes instead is a different number, not a lucky one."""
    written = config.params['Price Factors']['FXVol.USD.ZAR']
    surface = written['Surface'].array.copy()
    surface[:, 2] += 0.2
    config.params['Price Factors']['FXVol.USD.ZAR'] = {
        key: value for key, value in written.items() if key != 'Grid_Tolerance'}
    config.params['Price Factors']['FXVol.USD.ZAR']['Surface'] = utils.Curve([], surface)


def test_a_surface_this_family_did_not_write_is_read_as_the_authored_data_it_is():
    """MUTATE THE PROVENANCE, HOLD THE NAME - the desync a preference keyed on a NAME cannot see.

    A hand-authored surface can sit under a name an `FXVolPrices` block also uses, and then the two
    are simply different market data: the pricers read the surface, so integrating the quotes would
    build a curve off numbers nothing else in the config agrees with - silently, because both halves
    are individually valid. What decides is the FINGERPRINT this family leaves on what it writes,
    the same evidence `pinned_grid` reads back, so with it gone the quote block is not preferred and
    the curve is the one a config with no quote block at all gets.
    """
    desynced = fx_written(fx_world(True, atm=RISING, between=strip_the_fingerprint))
    authored = fx_written(
        fx_world(True, atm=RISING, quoted=False, between=strip_the_fingerprint))
    assert np.array_equal(desynced, authored), (
        'the quote block decided the source of a surface it did not write')
    assert np.abs(desynced[:, 1] - np.array(RISING)).max() > 1e-2, (
        'the desynced curve is the quote column, so this fixture cannot tell the sources apart')


def test_a_quote_block_that_moved_off_its_own_surface_raises_naming_both_expiry_sets():
    """The other half of the desync: the family DID write this surface, and the quotes have moved
    since. There is no ATM row at the new expiry to read, and the honest answer is to say so - the
    two expiry sets, named - rather than the `KeyError` a straight lookup raises out of a
    dict comprehension nobody can locate."""
    def quote_a_fifth_expiry(config):
        block = config.params['Market Prices']['FXVolPrices.USD.ZAR']['instrument']
        block['Points'] = block['Points'] + [
            {'Use': 'Yes', 'Expiry': 5.0, 'Pillar': 0.0, 'Quote_Type': 'ATM',
             'Quoted_Market_Value': 0.3, 'Timestamp': ''}]

    with pytest.raises(ValueError) as raised:
        fx_world(True, atm=RISING, between=quote_a_fifth_expiry)
    assert 'expiries 0.25, 1, 2, 3, 5' in str(raised.value) and \
        'surface over 0.25, 1, 2, 3 ' in str(raised.value), (
        'the refusal names one expiry set or neither: {}'.format(raised.value))


def test_a_family_quoted_world_reports_dV_dq_on_the_quotes_it_was_authored_with():
    """The preferred path all the way to a leaf: the quote leaf IS the authored ATM row, so the
    descriptors and the values line up with the block a desk would edit."""
    config = fx_world(True, atm=RISING)
    descriptors, leaves = config.quote_leaves['GBMAssetPriceTSModelPrices.ZAR']
    assert descriptors == ['ATM 0.25', 'ATM 1', 'ATM 2', 'ATM 3']
    assert np.array_equal(leaves.detach().numpy(), np.array(RISING))
    assert utils.Factor('GBMAssetPriceTSModelParameters', ('ZAR', 'Vol')) in \
        config.calibrated_factors
