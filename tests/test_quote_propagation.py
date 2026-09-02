"""Does the calibration Jacobian carry a tick, and does it say when it stopped being able to?

The operator is `theta ~ theta* + (dtheta/dq)(q_now - q0)` - `CalibrationSolve`'s implicit-function
backward read as a matrix. What is new is the LIFECYCLE, in three tiers:

    monitor   `dV/dq . dq` against the full reval of the ridden theta - a first-order agreement,
              scored on an h-ladder rather than at one bump size
    ride      theta_ridden against theta_refit where the solve is a unique root and the IFT is
              exact, so the drift is SECOND order and the gate pins the constant
    correct   a refit publishes what the ride it replaced was worth, under a NEW artifact id

plus three properties: a ride REFUSES a tick too big for it rather than pricing a plausible wrong
curve, an artifact MISS refuses rather than falling back to a number the replay tuple cannot
distinguish, and the operator is published over a COUPLED SET so a multi-curve tick cannot be
carried half way.

Most worlds are `test_interest_rate_prices.authored_world('zar')`; the coupling gates use `usd` and
one authored here whose coupling no field declares.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import copy
import json
import logging
import re

import numpy as np
import pandas as pd
import pytest
import torch

import derivus
from derivus import utils
from derivus.bootstrappers import ARTIFACTS, InterestRateCurveParameters, quote_knots
from derivus.config import Config, CustomJsonEncoder, ModelParams
from derivus.instruments import construct_instrument

from rates_world import BASE, deposit, par_swap
from test_interest_rate_prices import (authored_world, block_nodes, curve_of, discount_of,
                                       par_quotes, quote_point)

DTYPE = torch.float64
CCY = 'ZAR'
CURVE = 'ZAR-JIBAR-3M'
CURVE_NAME = 'InterestRate.' + CURVE
BLOCK = 'InterestRatePrices.' + CURVE
FACTOR = utils.Factor('InterestRate', (CURVE,))

USD_OIS, USD_PROJ = 'InterestRatePrices.USD-OIS', 'InterestRatePrices.USD-3M'

#: The book the ride has to reach, struck away from par so V is neither zero nor symmetric in the
#: curve - the same two swaps the quote-delta triangle prices.
BOOK = [('IRS_4Y', 4, 9.10), ('IRS_6Y', 6, 9.80)]

#: A tick that is not parallel: alternating signs, so no cancellation makes the ride look better
#: than it is and the whole Jacobian is exercised rather than its row sums.
SIGNS = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])


def market(propagate, tick=0.0, tolerance=1e9, sensitivity=False):
    """The ZAR world's `Market Prices`, every quote moved by `tick` percent with alternating sign.

    `tolerance` defaults to something no ride can exceed, because most gates here are measuring the
    drift rather than the refusal - the refusal has its own gate and its own mutation.
    """
    market_prices, _, _ = authored_world('zar')
    block = market_prices[BLOCK]['instrument']
    block['Quote_Propagation'] = 'Linear' if propagate else 'No'
    block['Quote_Sensitivity'] = 'Yes' if sensitivity else 'No'
    block['Drift_Tolerance'] = tolerance
    for point, sign in zip(block['Points'], SIGNS):
        point['Quoted_Market_Value'] += tick * sign
    return market_prices


def bootstrapped(market_prices, currency=CCY, spot_curve=CURVE, interpolation=None):
    """A `Config` whose curves have been solved from their quotes, the way a job reaches this."""
    config = Config(base_currency=currency)
    config.params['System Parameters']['Base_Date'] = BASE
    config.params['Price Factors'] = {'FxRate.{}'.format(currency): {
        'Domestic_Currency': None, 'Interest_Rate': spot_curve, 'Priority': 1, 'Spot': 1.0}}
    if interpolation is not None:
        config.params['Price Factor Interpolation'] = interpolation
    config.params['Market Prices'] = market_prices
    config.params['Bootstrapper Configuration'] = {'InterestRateCurveParameters': {}}
    config.bootstrap()
    return config


def key_of(config, *names):
    """The artifact SLOT a set of blocks in this config addresses."""
    market_prices = config.params['Market Prices']
    return InterestRateCurveParameters.plan_key(
        [(name, market_prices[name]['instrument']) for name in (names or (BLOCK,))],
        config.params['Price Factor Interpolation'],
        config.params['System Parameters']['Base_Date'])


def artifact_of(config, *names):
    return ARTIFACTS.get(key_of(config, *names))


def with_deals(config, book=None, currency=CCY, curve=CURVE, discount=CURVE,
               day_count='ACT_365'):
    config.deals = {'Attributes': {'Reference': 'test', 'Tag_Titles': ''},
                    'Deals': {'Children': [{'Instrument': construct_instrument(
                        par_swap(ref, currency, curve, discount, years, rate,
                                 day_count=day_count), {})}
                        for ref, years, rate in (BOOK if book is None else book)]},
                    'Calculation': {'Base_Date': BASE, 'Currency': currency}}
    return config


def run(config, greeks='No'):
    return derivus.run_baseval(
        config, prec=DTYPE, overrides={'MCMC_Simulations': 1, 'Random_Seed': 1, 'Greeks': greeks})


def baseval(config, greeks='No'):
    calc, out = run(config, greeks)
    rows = out['Results']['mtm']
    return calc, float(rows[rows['Parent'] == 'root']['Value'].sum())


def curve_values(config, name=CURVE_NAME):
    return config.params['Price Factors'][name]['Curve'].array[:, 1].copy()


def curve_of_block(config, market_price):
    return curve_values(config, curve_of(market_price))


def quotes_of(config, *names):
    """The quote vector in the order the artifact for that set indexes it, on its device."""
    market_prices = config.params['Market Prices']
    names = names or (BLOCK,)
    artifact = next((found for found in ARTIFACTS.artifacts.values()
                     if names[0] in found.members), None)
    return torch.tensor(
        [point['Quoted_Market_Value'] for name in names
         for point in InterestRateCurveParameters.used_quotes(
             market_prices[name]['instrument'], name)],
        dtype=DTYPE, device='cpu' if artifact is None else artifact.quotes.device)


def fitted_quotes(prepared, block=BLOCK):
    """The quotes the artifact covering `block` was FITTED at - that block's slice of the set-wide
    `q0`, which is the origin every tick here is written off."""
    market_prices = prepared.params['Market Prices']
    artifact = next(found for found in ARTIFACTS.artifacts.values() if block in found.members)
    start = 0
    for name in artifact.members:
        used = len(InterestRateCurveParameters.used_quotes(
            market_prices[name]['instrument'], name))
        if name == block:
            return artifact.quotes[start:start + used].cpu().numpy()
        start += used


def ticked(prepared, tick, tolerance=1e9, signs=None, block=BLOCK):
    """A PREPARED config whose quotes have since moved - the plan, patched with values.

    Nothing is re-bootstrapped: the block is the one the artifact was published against and only
    the numbers on it change, which is exactly the shape of a values patch and exactly what the
    ride is for. Written ABSOLUTELY, off the artifact's own q0, so repeated ticks in one test do
    not accumulate into a move nobody authored.
    """
    points = prepared.params['Market Prices'][block]['instrument']
    points['Drift_Tolerance'] = tolerance
    for point, base, sign in zip(points['Points'], fitted_quotes(prepared, block),
                                 SIGNS if signs is None else signs):
        point['Quoted_Market_Value'] = float(base + tick * sign)
    return prepared


@pytest.fixture(autouse=True)
def empty_store():
    """Every gate starts on a COLD process. The store is module-level and content-addressed, so a
    slot filled by the test before this one is exactly the state a restart does not have - and a
    gate that read it would be measuring the suite's order."""
    ARTIFACTS.artifacts.clear()
    yield
    ARTIFACTS.artifacts.clear()


# ---------------------------------------------------------------------------------------------
# The artifact - what is in it, and where J came from
# ---------------------------------------------------------------------------------------------

def test_the_artifact_holds_the_calibration_jacobian_the_solve_would_have_differenced():
    """`J` is the implicit function theorem read as a matrix, so it has to equal the one two full
    re-bootstraps per quote produce - the same reference the one-pass `dV/dq` triangle uses.

    That is the only claim worth making about extraction: `calibration_jacobian` runs no second
    solve, it differentiates the residual at the fixed point the forward pass already found.
    """
    artifact = artifact_of(bootstrapped(market(True)))
    h = 1e-4
    columns = []
    for j in range(len(artifact.quotes)):
        moved = []
        for sign in (1.0, -1.0):
            market_prices = market(False)
            market_prices[BLOCK]['instrument']['Points'][j]['Quoted_Market_Value'] += sign * h
            moved.append(curve_values(bootstrapped(market_prices)))
        columns.append((moved[0] - moved[1]) / (2.0 * h))

    reference = np.array(columns).T
    error = np.abs(artifact.jacobian.cpu().numpy() - reference).max()
    assert error / np.abs(reference).max() < 1e-8, (
        'the published J is not the calibration Jacobian: {:.3g}'.format(error))


def test_the_artifact_key_is_plan_side_and_its_id_is_not():
    """The two identities, and why there are two.

    The SLOT is the quote set with the numbers shadowed out, so a tick lands on it again - without
    that there is nothing to ride, because the artifact would have moved with the quote that moved.
    The artifact's own ID is the slot plus the quotes it was fitted at, so a refit publishes a new
    one and a propagated valuation can name which calibration it rode.
    """
    def key(block, interpolation=None, base_date=BASE):
        return InterestRateCurveParameters.plan_key(
            [(BLOCK, block)], ModelParams() if interpolation is None else interpolation, base_date)

    plain, moved = market(True)[BLOCK]['instrument'], market(True, tick=0.5)[BLOCK]['instrument']
    assert key(plain) == key(moved), (
        'a moved quote changed the slot, so no artifact could ever be found to ride')

    dropped = market(True)[BLOCK]['instrument']
    dropped['Points'][-1]['Use'] = 'No'
    assert key(dropped) != key(plain), (
        'dropping a quote left the slot alone, so a shorter curve would ride the longer one\'s J')

    first = artifact_of(bootstrapped(market(True)))
    second = artifact_of(bootstrapped(market(True, tick=0.5)))
    assert first.key == second.key
    assert first.artifact_id != second.artifact_id, 'the refit reused its predecessor\'s identity'


def test_the_slot_names_everything_the_solve_reads_and_the_block_does_not_carry():
    """A key that does not name an input of the SOLVE is a key two different curves share, and the
    second silently rides the first one's operator. Two are not on the block at all - the base date
    and the interpolation scheme - and both were measured doing exactly that: two jobs 45 days apart
    shared a slot, and a linearly-interpolated job rode a Hermite solve 0.53bp away from its own.
    The mutation is the pairing: the same job at the same date under the same scheme KEEPS the slot.
    """
    def key(interpolation=None, base_date=BASE):
        return InterestRateCurveParameters.plan_key(
            [(BLOCK, market(True)[BLOCK]['instrument'])],
            ModelParams() if interpolation is None else interpolation, base_date)

    hermite = ModelParams()
    hermite.append('InterestRate', (), 'HermiteRT')

    assert key() == key(), 'the slot is not a function of its inputs'
    assert key(interpolation=hermite) != key(), (
        'a Hermite job addresses the linear job\'s slot - it would ride a curve nobody solved')
    assert key(base_date=BASE + pd.DateOffset(days=45)) != key(), (
        'two base dates share a slot - the later job would ride the earlier date\'s theta*')


# ---------------------------------------------------------------------------------------------
# Tier 1 - monitoring: dV/dq . dq against the reval the ride produces
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize('tick', [0.02, 0.01, 0.005])
def test_the_quote_delta_predicts_the_value_the_ride_reprices(tick):
    """Tier one against tier two, which is the only pair that can be compared without a refit.

    `dV/dq . dq` is the desk's P&L explain forwards; the ride is what a reval off the same tick
    actually produces. They agree to FIRST order, so what is pinned is the SLOPE of the miss:
    `ratio - 1` is 0.791 times the tick on this world and book, flat to three digits over four
    sizes. Asserting only that the ratio is near one would pass on a monitor that was wrong by a
    constant, which is the failure worth catching.
    """
    prepared = bootstrapped(market(True, sensitivity=True))
    _, base = baseval(with_deals(prepared), greeks='First')
    predicted = prepared.quote_leaves[BLOCK][1].grad.detach().cpu().numpy() @ (SIGNS * tick)

    _, rode = baseval(with_deals(ticked(prepared, tick)))
    assert abs(rode - base) > 0, 'the tick never reached the value - nothing is being compared'
    ratio = (rode - base) / predicted
    assert abs((ratio - 1.0) / tick - 0.791) < 0.01, (
        'dV/dq . dq {:.6g} against the ridden reval {:.6g}, ratio {:.6f}'.format(
            predicted, rode - base, ratio))


# ---------------------------------------------------------------------------------------------
# Tier 2 - the ride: theta_ridden against theta_refit, and the refusal
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize('tick', [1e-3, 1e-2, 1e-1])
def test_the_ride_drifts_second_order_and_the_constant_is_pinned(tick):
    """The curve family's solve is a unique root and its IFT is exact, so the ride's error is pure
    curvature: `theta_refit - theta_ridden` is O(dq^2) with a constant that belongs to the world.

    Pinning the CONSTANT is what makes this a gate rather than a smallness assertion - a first-order
    bug would still look small at 1bp and would blow the quotient here. Measured on the ZAR world:
    8.4665e-4 / 8.4611e-4 / 8.4074e-4 in theta and 0.075142 / 0.075157 / 0.075308 in quote space
    over the three ticks, so each holds to better than a percent rather than to three digits - the
    bands below are what that is worth.
    """
    prepared = bootstrapped(market(True))
    artifact = artifact_of(prepared)
    refit = curve_values(bootstrapped(market(False, tick=tick)))

    quotes = quotes_of(ticked(prepared, tick))
    ridden = artifact.ride(quotes)
    theta_drift = float(np.abs(ridden.cpu().numpy() - refit).max())
    quote_drift = float(artifact.mispricing(ridden, quotes).abs().max())

    assert abs(theta_drift / tick ** 2 - 8.44e-4) < 5e-6, (
        'theta drift {:.4g} at tick {:g} is not the pinned second-order constant'.format(
            theta_drift, tick))
    assert abs(quote_drift / tick ** 2 - 0.0752) < 3e-4, (
        'quote-space drift {:.4g} at tick {:g} is not the pinned second-order constant'.format(
            quote_drift, tick))


#: Tick SHAPES, not sizes - the scan that found the drift metric reading low. A parallel move sits
#: almost entirely in the Jacobian's dominant direction; a sparse or sign-mixed one excites the
#: small singular values, which is where a `dF/dq` frozen at theta* stops describing the residual.
SHAPES = {'alternating': SIGNS,
          'parallel': np.ones(7),
          'one quote': np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
          'mixed sparse': np.array([0.0, -1.0, -1.0, 0.0, 1.0, -1.0, 1.0]),
          'long end': np.array([0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 1.0])}


@pytest.mark.parametrize('shape', sorted(SHAPES))
@pytest.mark.parametrize('tick', [0.05, 0.2, 0.5])
def test_the_drift_metric_is_exact_and_bounds_the_theta_drift(shape, tick):
    """What the refusal is scored on, and what that score is WORTH.

    EXACT. `mispricing` re-differentiates `dF/dq` at the theta being scored rather than reusing the
    one stored at `theta*`, so `F(theta, q) = F(theta, q0) + (dF/dq)(q - q0)` holds with no
    remainder. The reference is a REFIT at the moved quotes, where no correction term survives; the
    two agree to solver tolerance rather than to a ratio.

    BOUNDING - what `Drift_Tolerance` rests on. The quote-space residual bounds the curve:
    `||theta_ridden - theta_refit||inf <= ||J||inf ||r||inf` to first order, which is the conversion
    the refusal message publishes.

    The shapes are the point. The proxy this replaced reused `dF/dq` at `theta*` and was pinned as
    "always HIGH" on ONE sign pattern on ONE world; scanned across shapes it ran from 0.886x the
    truth to 21.9x, and an end-to-end case at 0.886 priced 5.8% over tolerance without refusing.
    """
    prepared = bootstrapped(market(True))
    artifact = artifact_of(prepared)
    signs = SHAPES[shape]
    quotes = quotes_of(ticked(prepared, tick, signs=signs))
    ridden = artifact.ride(quotes)
    measured = float(artifact.mispricing(ridden, quotes).abs().max())

    moved = market(True)
    for point, sign in zip(moved[BLOCK]['instrument']['Points'], signs):
        point['Quoted_Market_Value'] += tick * sign
    refit = bootstrapped(moved)
    exact = float(artifact_of(refit).mispricing(ridden, quotes).abs().max())

    assert abs(measured - exact) <= 1e-12 + 1e-9 * abs(exact), (
        'the drift metric {:.6g} is not the residual a refit measures, {:.6g}'.format(
            measured, exact))
    theta_drift = float(np.abs(ridden.cpu().numpy() - curve_values(refit)).max())
    assert theta_drift <= artifact.jacobian_norm * measured, (
        'the theta drift {:.4g} is outside ||J||inf {:.4g} x the quote residual {:.4g} - the '
        'conversion the refusal publishes does not bound what it claims to'.format(
            theta_drift, artifact.jacobian_norm, measured))


def test_a_small_tick_rides_and_a_large_one_refuses():
    """The refusal, and the mutation is built in: the SAME tick under a tolerance that admits it
    prices, so a gate that lost the refusal would have to lose this half too.

    The tolerance is in percent of quote, which is what makes 1e-3 a number a desk can set - a tenth
    of a basis point of mispricing on the block's own benchmarks. On this world it admits 11.5bp
    (drift 9.96e-4) and refuses 12bp (1.08e-3).
    """
    prepared = with_deals(bootstrapped(market(True)))
    _, small = baseval(with_deals(ticked(prepared, 0.05, tolerance=1e-3)))
    assert small != 0.0

    with pytest.raises(utils.CalibrationStale, match='Drift_Tolerance'):
        baseval(with_deals(ticked(prepared, 0.45, tolerance=1e-3)))

    # the SAME tick under a tolerance that admits it prices, so the refusal above is the
    # tolerance's doing and not the tick's - and it rode somewhere else, so something did happen
    _, large = baseval(with_deals(ticked(prepared, 0.45, tolerance=1e9)))
    assert large != small


def test_the_refusal_message_converts_its_tolerance_into_curve_units():
    """`Drift_Tolerance` is declared in percent of quote and felt in basis points of zero rate. The
    refusal carries the conversion - `tolerance x ||J||inf` - because a number a desk cannot read in
    the units it thinks in is a number nobody tunes."""
    prepared = with_deals(bootstrapped(market(True)))
    with pytest.raises(utils.CalibrationStale) as refusal:
        baseval(with_deals(ticked(prepared, 0.45, tolerance=1e-3)))
    expected = 1e-3 * artifact_of(prepared).jacobian_norm * 1e4
    assert 'bp of zero rate' in str(refusal.value), str(refusal.value)
    assert '{:.3g}bp'.format(expected) in str(refusal.value), str(refusal.value)


def test_the_tick_shape_that_used_to_be_admitted_over_tolerance_now_refuses():
    """The executed case behind the scorer's replacement, driven end to end.

    On the mixed-sparse shape the old proxy read 0.886 of the true drift, so a tolerance set between
    the two admitted a ride whose real mispricing was 13% past it - the engine priced, no refusal,
    2.59bp of zero rate from the refit. The metric is exact now, so the same tolerance refuses.

    The tolerance is derived from the drift a REFIT measures, never from the scorer under test -
    which is what keeps this a gate on the scorer rather than a tautology. At the refit's own
    quotes the correction term is identically zero, so that reading is the residual whatever
    `dF/dq` the scorer would have used.
    """
    signs, tick = SHAPES['mixed sparse'], 0.5
    prepared = bootstrapped(market(True))
    ridden = artifact_of(prepared).ride(quotes_of(ticked(prepared, tick, signs=signs)))

    moved = market(True)
    for point, sign in zip(moved[BLOCK]['instrument']['Points'], signs):
        point['Quoted_Market_Value'] += tick * sign
    refit = artifact_of(bootstrapped(moved))
    true_drift = float(refit.mispricing(ridden, refit.quotes).abs().max())

    # the same ride through the engine, under a tolerance the OLD proxy cleared (it read 0.886x of
    # the truth on this shape) and the truth does not
    ARTIFACTS.artifacts.clear()
    prepared = with_deals(bootstrapped(market(True)))
    with pytest.raises(utils.CalibrationStale, match='Drift_Tolerance'):
        baseval(with_deals(ticked(prepared, tick, tolerance=0.95 * true_drift, signs=signs)))

    # and a tolerance above the truth still prices, so the refusal is the drift's doing
    _, priced = baseval(with_deals(ticked(prepared, tick, tolerance=1.05 * true_drift,
                                          signs=signs)))
    assert priced != 0.0


def test_what_the_refusal_is_protecting_against():
    """The refusal above is scored on a tick that is genuinely too big - this is how big.

    The refused ride is not a NaN and not an exception waiting to happen: it is a perfectly
    plausible curve, **1.66 basis points** of zero rate away from the one a refit finds, priced
    without complaint. That is the number the refusal exists for, and measuring it here is what
    stops the gate above from passing on a tick nothing was ever wrong with. It is also the
    second-order constant read forwards - 8.3e-4 x 0.45^2 - so the two gates agree on one curvature.
    """
    prepared = bootstrapped(market(True))
    artifact = artifact_of(prepared)
    tick = 0.45
    ridden = artifact.ride(quotes_of(ticked(prepared, tick))).cpu().numpy()
    refit = curve_values(bootstrapped(market(False, tick=tick)))

    assert np.isfinite(ridden).all(), 'a refused ride would have raised on its own'
    error = np.abs(ridden - refit).max()
    assert abs(error * 1e4 - 1.66) < 0.02, (
        'the refused ride is {:.3g} of zero rate from the refit'.format(error))


# ---------------------------------------------------------------------------------------------
# The coupled set - one system, one operator, and no partial rides
# ---------------------------------------------------------------------------------------------

def usd_market(propagation):
    market_prices, _, _ = authored_world('usd')
    for name, mode in propagation.items():
        market_prices[name]['instrument']['Quote_Propagation'] = mode
        market_prices[name]['instrument']['Drift_Tolerance'] = 1e9
    return market_prices


def usd_book(config):
    return with_deals(config, book=[('BK_5Y', 5, 3.90), ('BK_10Y', 10, 4.20)],
                      currency='USD', curve='USD-3M', discount='USD-OIS', day_count='ACT_360')


def usd_config(market_prices):
    return bootstrapped(market_prices, currency='USD', spot_curve='USD-OIS')


def test_a_coupled_set_solves_as_one_system_and_rides_whole():
    """THE MULTI-CURVE RIDE. `USD-3M` is solved discounting on `USD-OIS`, so `theta_2` depends on
    `q_1` through `theta_1` - and a PER-BLOCK Jacobian has no column for it.

    A first-order term dropped behind a drift metric that read machine zero, because `mispricing`
    priced a benchmark set whose discount curve was frozen at the fit. Measured before the fix at a
    10bp OIS tick: the true book move was -23.36 and the partly-ridden set reported +185.01 - wrong
    sign, 8.9x the size, drift 4.5e-4 and admitted healthy.

    The SET is the unit: `coupled_sets` measures which blocks read each other's curves, `solve_set`
    flattens them into ONE Newton system, and `calibration_jacobian` inverts one block matrix, so
    `dtheta_2/dq_1` is a column of the published `J`. From a base of 9644.61 the refit lands at
    9621.25 and the ride has to land there too.
    """
    tick = 0.10
    prepared = usd_book(usd_config(usd_market({USD_OIS: 'Linear', USD_PROJ: 'Linear'})))
    artifact = artifact_of(prepared, USD_OIS, USD_PROJ)
    assert artifact.members == (USD_OIS, USD_PROJ), 'the two blocks did not solve as one set'
    assert ARTIFACTS.covering(utils.Factor('InterestRate', ('USD-3M',))) == [artifact], (
        'the projection curve answers to a different artifact than the one it was solved with')
    assert artifact.jacobian.shape == (17, 17), artifact.jacobian.shape

    _, base = baseval(prepared)
    stale = curve_of_block(prepared, USD_PROJ)

    ticked(prepared, tick, block=USD_OIS, signs=np.ones(8))
    _, rode = baseval(prepared)
    ridden = prepared.propagated_factor(utils.Factor('InterestRate', ('USD-3M',)))[0]

    moved = usd_market({USD_OIS: 'Linear', USD_PROJ: 'Linear'})
    for point in moved[USD_OIS]['instrument']['Points']:
        point['Quoted_Market_Value'] += tick
    ARTIFACTS.artifacts.clear()
    refit_config = usd_book(usd_config(moved))
    _, refit = baseval(refit_config)
    refit_nodes = curve_of_block(refit_config, USD_PROJ)

    assert abs(base - 9644.61) < 0.01 and abs(refit - 9621.25) < 0.01, (base, refit)
    assert np.abs(refit_nodes - stale).max() > 1.7e-5, (
        'the OIS tick did not move the projection curve - this world proves nothing')
    assert np.abs(ridden - refit_nodes).max() < 1e-7, (
        'the ridden projection curve is {:.4g} from the refit - the coupling is not in J'.format(
            np.abs(ridden - refit_nodes).max()))
    assert abs(rode - refit) < 0.01 * abs(refit - base), (
        'the ridden SET is worth {:.4f} where the refit says {:.4f}, against a move of {:.4f}'
        .format(rode, refit, refit - base))


@pytest.mark.parametrize('tick', [0.05, 0.10, 0.20])
def test_the_coupled_ride_is_second_order_in_the_tick_it_used_to_be_first_order_in(tick):
    """The order of the error is what says the coupling is CARRIED rather than merely smaller.

    Partially ridden, the residual PV error was linear in the OIS tick - 2090.8 / 2087.6 / 2083.7
    per percent over four sizes, a first-order term with no tolerance that catches it. Ridden as a
    set it is quadratic, which is the ride's own curvature and nothing else.
    """
    prepared = usd_book(usd_config(usd_market({USD_OIS: 'Linear', USD_PROJ: 'Linear'})))
    _, base = baseval(prepared)
    ticked(prepared, tick, block=USD_OIS, signs=np.ones(8))
    _, rode = baseval(prepared)

    moved = usd_market({USD_OIS: 'Linear', USD_PROJ: 'Linear'})
    for point in moved[USD_OIS]['instrument']['Points']:
        point['Quoted_Market_Value'] += tick
    ARTIFACTS.artifacts.clear()
    _, refit = baseval(usd_book(usd_config(moved)))

    assert abs(refit - base) > 10.0, 'the tick barely moved the book - nothing is being measured'
    assert abs((rode - refit) / tick ** 2 - 0.20) < 0.03, (
        'the ride error {:.6g} at tick {:g} is {:.4g}/tick and {:.4g}/tick^2 - a first-order term '
        'reads about 2085/tick'.format(rode - refit, tick, (rode - refit) / tick,
                                       (rode - refit) / tick ** 2))


def x_world():
    """Two blocks whose `Discount_Rate` is blank on BOTH - so every declaration says the two curves
    stand alone - while `X-DISC`'s swaps FORECAST off `X-3M`.

    That is the configuration a declared dependency graph cannot see: what a benchmark projects off
    is authored inside its own deal block, and `Discount_Rate` is a property of the quote SET.
    """
    truth = {'InterestRate.X-3M': [0.0400, 0.0395, 0.0390, 0.0385, 0.0380],
             'InterestRate.X-DISC': [0.0370, 0.0366, 0.0362, 0.0358, 0.0355]}
    projection = [quote_point('X 3M depo',
                              deposit('XD3', 'XXX', 'X-3M', 3, 0.0, day_count='ACT_365'))]
    projection += [quote_point('X {}Y proj'.format(y),
                               par_swap('XP{}'.format(y), 'XXX', 'X-3M', 'X-3M', y, 0.0,
                                        day_count='ACT_365')) for y in (1, 2, 5, 10)]
    discount = [quote_point('X 3M depo D',
                            deposit('XDD3', 'XXX', 'X-DISC', 3, 0.0, day_count='ACT_365'))]
    discount += [quote_point('X {}Y disc'.format(y),
                             par_swap('XS{}'.format(y), 'XXX', 'X-3M', 'X-DISC', y, 0.0,
                                      day_count='ACT_365')) for y in (1, 2, 5, 10)]
    blocks = {'InterestRatePrices.X-3M': {'Currency': 'XXX', 'Day_Count': 'ACT_365',
                                          'Discount_Rate': '', 'Points': projection},
              'InterestRatePrices.X-DISC': {'Currency': 'XXX', 'Day_Count': 'ACT_365',
                                            'Discount_Rate': '', 'Points': discount}}

    price_factors = {'FxRate.XXX': {'Domestic_Currency': None, 'Interest_Rate': 'X-3M',
                                    'Priority': 1, 'Spot': 1.0}}
    for market_price, block in blocks.items():
        knots = block_nodes(block, discount_of(market_price, block), 0.0)
        price_factors[curve_of(market_price)] = {
            'Property_Aliases': None, 'Sub_Type': None, 'Currency': 'XXX',
            'Day_Count': 'ACT_365', 'Curve': utils.Curve([], list(zip(
                quote_knots(knots, BASE, 'ACT_365', {}), truth[curve_of(market_price)])))}
    for market_price, block in blocks.items():
        for point, quote in zip(block['Points'], par_quotes(
                block, discount_of(market_price, block), price_factors)):
            point['Quoted_Market_Value'] = quote
    for block in blocks.values():
        block['Quote_Propagation'] = 'Linear'
        block['Drift_Tolerance'] = 1e9
    return {name: {'instrument': block, 'Children': []} for name, block in blocks.items()}


def test_the_coupling_is_measured_not_declared():
    """The set is formed by DIFFERENTIATION, not by reading `Discount_Rate`. On this world every
    declaration says the two curves are independent and self-discounting, and a 10bp tick on the
    projection strip moves the other curve by 568 basis points - before the fix both blocks
    published per-block operators, the ride moved `X-DISC` by exactly ZERO and its drift metric read
    7.7e-16. `BenchmarkInstruments.reads` puts every constant on the tape and asks the backward pass.
    """
    prepared = bootstrapped(x_world(), currency='XXX', spot_curve='X-3M')
    artifact, = ARTIFACTS.covering(utils.Factor('InterestRate', ('X-DISC',)))
    assert artifact.members == ('InterestRatePrices.X-3M', 'InterestRatePrices.X-DISC'), (
        'the undeclared coupling was not measured: {}'.format(artifact.members))
    assert all(block['instrument']['Discount_Rate'] == ''
               for block in prepared.params['Market Prices'].values()), (
        'this world declares a dependency after all - it proves nothing')

    stale = curve_of_block(prepared, 'InterestRatePrices.X-DISC')
    ticked(prepared, 0.10, block='InterestRatePrices.X-3M', signs=np.ones(5))
    ridden = prepared.propagated_factor(utils.Factor('InterestRate', ('X-DISC',)))[0]

    moved = x_world()
    for point in moved['InterestRatePrices.X-3M']['instrument']['Points']:
        point['Quoted_Market_Value'] += 0.10
    ARTIFACTS.artifacts.clear()
    refit = curve_of_block(bootstrapped(moved, currency='XXX', spot_curve='X-3M'),
                           'InterestRatePrices.X-DISC')

    assert np.abs(refit - stale).max() > 5e-2, (
        'the undeclared dependency is not material here - this world proves nothing')
    assert np.abs(ridden - stale).max() > 5e-2, 'the ride left the coupled curve where it was'
    assert np.abs(ridden - refit).max() < 1e-3, (
        'the ride carried {:.4g} of a {:.4g} move'.format(np.abs(ridden - stale).max(),
                                                          np.abs(refit - stale).max()))


def test_a_partial_declaration_over_a_coupled_set_refuses():
    """A ride that reaches one curve of a coupled set and not the other is the defect this whole
    section is about, so it is made UNREPRESENTABLE rather than merely discouraged.

    The pairing is the mutation: declaring it on BOTH blocks of the same set publishes, so the
    refusal is the partial declaration's doing and not the world's.
    """
    with pytest.raises(Exception, match='COUPLED SET'):
        usd_config(usd_market({USD_OIS: 'Linear'}))
    with pytest.raises(Exception, match='COUPLED SET'):
        usd_config(usd_market({USD_PROJ: 'Linear'}))

    ARTIFACTS.artifacts.clear()
    both = usd_config(usd_market({USD_OIS: 'Linear', USD_PROJ: 'Linear'}))
    assert artifact_of(both, USD_OIS, USD_PROJ) is not None


def test_the_set_is_only_measured_where_an_operator_is_asked_for():
    """The measurement costs a compile and a backward pass per block, and it buys exactly one thing:
    an operator. Without `Quote_Propagation` anywhere, the blocks are solved one at a time in
    dependency order - which is what a bootstrap has always done, and the store is never touched.

    The two solves agree on the ANSWER but not bit for bit, and that is the honest reading: Newton
    from a par seed on one 17-dimensional system takes a different path from two 8- and
    9-dimensional ones taken in order, and both stop at the same root inside the solver's own
    tolerance. `Tol` is 1e-14 on a rate of order 1e-2, and they agree to 1e-15.
    """
    plain = usd_config(usd_market({}))
    assert not ARTIFACTS.artifacts, 'a section that asked for nothing published an artifact'
    coupled = usd_config(usd_market({USD_OIS: 'Linear', USD_PROJ: 'Linear'}))
    for market_price in (USD_OIS, USD_PROJ):
        moved = np.abs(curve_of_block(plain, market_price) -
                       curve_of_block(coupled, market_price)).max()
        assert moved < 1e-14, (
            '{}: solving the set jointly moved the answer by {:.3g}'.format(market_price, moved))


def test_there_is_no_theta_current():
    """The grep the statelessness claim rests on, as a gate rather than a sentence. Every mutable-
    calibration design this one is NOT has a `theta_current` each tick updates in place; theta is
    derived per EXECUTE from `(artifact, q_now)` and stored nowhere. The word appears in docstrings
    SAYING there is none; an ASSIGNMENT is what would make it real.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assigned = {}
    for name in ('bootstrappers', 'calculation', 'config', 'riskfactors', 'utils'):
        with open(os.path.join(root, 'derivus', name + '.py')) as source:
            found = re.findall(r'^\s*\S*theta_current\s*=.*$', source.read(), re.MULTILINE)
            if found:
                assigned[name] = found
    assert not assigned, 'something now carries a mutable theta: {}'.format(assigned)


def test_the_switch_off_is_todays_path_bit_for_bit():
    """`Quote_Propagation: 'No'` is the default and has to be free: the same job, the same quotes,
    the same numbers - and the artifact store must not have been consulted at all.

    Scored on the VALUE of a book, because a switch that changed a mark is the failure this whole
    lifecycle exists to prevent.
    """
    plain = with_deals(bootstrapped(market(False)))
    assert not ARTIFACTS.artifacts, 'a block that declined the switch published an artifact'
    connected = with_deals(bootstrapped(market(True)))
    assert artifact_of(connected) is not None
    assert np.array_equal(curve_values(plain), curve_values(connected)), (
        'publishing an artifact moved the solve')
    assert baseval(plain)[1] == baseval(connected)[1], 'the switch moved a mark'


def cmc(config, tenor_offset=0.0):
    """The same book under a Hull-White curve, as a netting set - the only path that applies a
    `Tenor_Offset`, since `update_factors` never asks for one. It is also the STOCHASTIC branch of
    `_build_factor_state`, so reaching the refusal from here says the ride is offered where a
    simulated curve is minted and not only where a static one is."""
    config.params['Price Models'] = {
        'HullWhite1FactorInterestRateModel.{}'.format(CURVE): {
            'Alpha': 0.05, 'Lambda': 0.0, 'Sigma': utils.Curve([], [[0.0, 0.01]]),
            'Quanto_FX_Correlation': 0.0, 'Quanto_FX_Volatility': None}}
    config.params['Model Configuration'].append(
        'InterestRate', (), 'HullWhite1FactorInterestRateModel')
    netting = {'Object': 'NettingCollateralSet', 'Reference': 'test', 'Agreement_Currency': CCY,
               'Apply_Closeout_When_Uncollateralized': 'No', 'Balance_Currency': CCY,
               'Opening_Balance': 0.0, 'Collateralized': 'False', 'Netted': 'True',
               'Calendars': None}
    config.deals['Deals'] = {'Children': [{
        'Instrument': construct_instrument(netting, {}),
        'Children': config.deals['Deals']['Children']}]}
    _, out = derivus.run_cmc(config, prec=DTYPE, overrides={
        'Run_Date': BASE.strftime('%Y-%m-%d'), 'Time_grid': '0d 1y(1y) 6y',
        'Batch_Size': 32, 'Simulation_Batches': 1, 'Random_Seed': 1, 'Currency': CCY,
        'MCMC_Simulations': 0, 'Tenor_Offset': tenor_offset, 'Deflation_Interest_Rate': CURVE,
        'Gradient_Variables': 'Factors'})
    return out['Results']['mtm'].values


def test_a_tenor_offset_refuses_the_ride():
    """A shifted curve is interpolated off coefficients fitted before the ride, so riding it would
    price a curve nobody solved for - and DECLINING would silently price the stale one, which is a
    wrong number rather than a missing derivative. So it raises.

    Paired with a zero offset, which prices: the refusal is the offset's doing, and this gate
    cannot pass by never reaching the seam."""
    prepared = with_deals(bootstrapped(market(True)))
    assert np.abs(cmc(with_deals(ticked(prepared, 0.01)))).max() > 0
    with pytest.raises(Exception, match='Tenor_Offset'):
        cmc(with_deals(ticked(prepared, 0.01)), tenor_offset=10.0)


# ---------------------------------------------------------------------------------------------
# Tier 3 - the correct step: the refit publishes what the ride was worth
# ---------------------------------------------------------------------------------------------

def test_the_refit_publishes_the_drift_of_the_ride_it_replaced(caplog):
    """The loop closes here. A refit of a block whose slot is occupied scores the artifact it is
    about to replace - in both coordinates the curve family is entitled to - logs it at INFO and
    publishes it ON the replacement, so how stale the last calibration got travels with the one
    that replaced it.

    The numbers are checked against the ride computed independently, not merely asserted present:
    a drift block full of zeros would pass a presence check and say nothing.
    """
    first = bootstrapped(market(True))
    artifact = artifact_of(first)
    assert artifact.drift is None, 'the first artifact into an empty slot scored something'

    tick = 0.2
    quotes = quotes_of(ticked(first, tick))
    ridden = artifact.ride(quotes)

    with caplog.at_level(logging.INFO):
        refit = bootstrapped(market(True, tick=tick))
    published = artifact_of(refit)

    assert published.drift['rode'] == artifact.artifact_id
    assert published.artifact_id != artifact.artifact_id
    assert abs(published.drift['tick'] - tick) < 1e-12
    assert abs(published.drift['theta'] -
               np.abs(ridden.cpu().numpy() - curve_values(refit)).max()) < 1e-14
    assert abs(published.drift['quote'] -
               float(published.mispricing(ridden, quotes).abs().max())) < 1e-14
    assert published.drift['theta'] > float(refit.params['Market Prices'][BLOCK][
        'instrument'].get('Tol', 1e-14)), 'the drift is at solver tolerance - nothing was measured'
    assert 'rode a' in caplog.text and 'drift of' in caplog.text, caplog.text


# ---------------------------------------------------------------------------------------------
# How a tick arrives - one patch carrying a spot AND a quote
# ---------------------------------------------------------------------------------------------

def test_one_values_patch_carrying_a_spot_and_a_quote_composes_with_the_ride():
    """THE CLOSED CONTRACT. A quote is a patchable VALUE, so the tick and the reval it reaches are
    one `patch_market` call and one EXECUTE with no bootstrap between them.

    Three claims. The plan is UNTOUCHED by a moved quote and `values_hash` carries it, so the two
    staleness dimensions are disjoint. And the ridden reval off the PATCH is bit-for-bit the ridden
    reval off the same tick applied by EDITING the document, which says the patch is a real tick.

    BOTH halves have to MOVE something, or a composition gate is one section wearing the other's
    name: the spot moves off the 1.0 this world bootstraps at and the quote rows move, both
    asserted. The spot cannot reach a mark here - `FxRate.ZAR` is the reporting currency's own rate
    - so the EDITED reference carries the same spot, which makes the bit-equality a statement about
    the whole patch.

    THE HASH CLAIM IS TAKEN TWICE, because on the composed patch it is not attributable: the spot is
    `bind='value'`, so `values_hash` moves on the spot alone and a quote reaching neither hash would
    read as a pass. A second `Context` takes the QUOTE half alone first. MUTANT: delete the
    `Market Prices` loop from `Context.market_patch` and 49 of 49 assertions stay green with quotes
    off the values plane entirely - the quote-only reading is what dies.
    """
    tick, spot, rate = 0.02, 1.25, 'FxRate.{}'.format(CCY)
    edited = with_deals(bootstrapped(market(True)))
    plain = baseval(edited)[1]
    ticked(edited, tick)
    edited.params['Price Factors'][rate]['Spot'] = spot
    rode = baseval(with_deals(edited))[1]
    moved = [{'Quoted_Market_Value': point['Quoted_Market_Value']}
             for point in edited.params['Market Prices'][BLOCK]['instrument']['Points']]

    alone = derivus.Context()
    alone.current_cfg = with_deals(bootstrapped(market(True)))
    quiet = (alone.plan_hash(), alone.values_hash())
    alone.patch_market({BLOCK: {'Points': moved}})

    assert BLOCK in alone.market_patch(), 'the quote never reached the values half at all'
    assert alone.plan_hash() == quiet[0], 'a quote-only patch moved the plan hash'
    assert alone.values_hash() != quiet[1], (
        'a quote-only patch left `values_hash` alone - the quote half of the patch is on neither '
        'plane, which is the replay collision this whole split exists to close')

    context = derivus.Context()
    context.current_cfg = with_deals(bootstrapped(market(True)))
    before = (context.plan_hash(), context.values_hash())
    context.patch_market({rate: {'Spot': spot}, BLOCK: {'Points': moved}})
    after = (context.plan_hash(), context.values_hash())
    patched = context.current_cfg.params

    assert after[0] == before[0], (
        'a moved quote moved the plan hash - the section is plan-side again')
    assert after[1] != before[1], 'the composed patch left the values hash alone'
    assert patched['Price Factors'][rate]['Spot'] == spot, 'the spot half never landed'
    assert [{'Quoted_Market_Value': point['Quoted_Market_Value']} for point in
            patched['Market Prices'][BLOCK]['instrument']['Points']] == moved, (
        'the quote half never landed')
    assert rode != plain, 'the tick never reached the book - nothing is being composed'
    assert baseval(with_deals(context.current_cfg))[1] == rode, (
        'the patched tick and the edited one rode to different marks')


# ---------------------------------------------------------------------------------------------
# Statelessness
# ---------------------------------------------------------------------------------------------

def test_two_executes_over_one_artifact_and_one_tick_are_bit_identical():
    """The property the whole design is built on: theta is DERIVED per EXECUTE from (artifact,
    q_now) and nothing accumulates, so running twice cannot drift. Bit-identical, not close."""
    prepared = with_deals(bootstrapped(market(True)))
    ticked(prepared, 0.03)
    first = baseval(with_deals(prepared))[1]
    second = baseval(with_deals(prepared))[1]
    assert first == second, 'two EXECUTEs over one (artifact, q_now) disagreed'


def test_the_artifact_survives_a_job_document_round_trip_by_key():
    """The artifact holds tensors and cannot be serialised - so what has to survive a JSON round
    trip is its KEY, computed off the block the decoder rebuilt.

    That is not free: a `Market Prices` block carries Timestamps, DateOffsets, DateLists and
    Percents, and the key is a hash of their encoded form. A decoder that rebuilt any of them
    differently would silently miss the slot and REFUSE forever, which is at least loud - but it is
    a refusal nobody can act on, so it is gated here.
    """
    prepared = bootstrapped(market(True))
    key = key_of(prepared)
    ticked(prepared, 0.02)

    document = json.dumps({'Calc': {
        'Calculation': {'Object': 'BaseValuation', 'Base_Date': BASE, 'Currency': CCY},
        'Deals': {'Tag_Titles': '', 'Reference': 'test', 'Deals': {'Children': []}},
        'MergeMarketData': {'MarketDataFile': '', 'ExplicitMarketData': {
            'System Parameters': {'Base_Currency': CCY, 'Base_Date': BASE},
            'Price Factors': prepared.params['Price Factors'],
            'Market Prices': prepared.params['Market Prices']}}}}, cls=CustomJsonEncoder)

    reloaded = derivus.Context().load_json((document, 'posted')).current_cfg
    assert key_of(reloaded) == key, (
        'the decoder rebuilt the block into a different slot, so a reloaded job refuses forever')
    assert np.array_equal(reloaded.propagated_factor(FACTOR)[0],
                          prepared.propagated_factor(FACTOR)[0]), (
        'the reloaded job rode to a different curve')


def test_the_ride_is_a_pure_function_of_the_artifact_and_the_quotes():
    """Riding to A, then to B, then back to A returns the FIRST answer - so no order of ticks can
    leave a residue in the operator. That is what `theta_current` would break, and there is no
    `theta_current`: the grep below is part of the claim."""
    prepared = with_deals(bootstrapped(market(True)))
    artifact = artifact_of(prepared)
    a = artifact.ride(quotes_of(ticked(prepared, 0.02))).cpu().numpy().copy()
    b = artifact.ride(quotes_of(ticked(prepared, 0.10))).cpu().numpy().copy()
    back = artifact.ride(quotes_of(ticked(prepared, 0.02))).cpu().numpy().copy()
    assert not np.array_equal(a, b), 'the two ticks produced one curve - nothing is being tested'
    assert np.array_equal(a, back), 'the ride kept state between calls'
    assert np.array_equal(artifact.theta.cpu().numpy(),
                          artifact_of(prepared).theta.cpu().numpy()), 'the artifact was edited'


def test_a_cold_process_refuses_rather_than_pricing_something_else():
    """A plan the cache cannot answer is a MISS, never a different number. An artifact holds tensors
    and a compiled benchmark set, so a fresh process has none; falling back to `Price Factors` was
    the shipped behaviour and is the one thing the replay tuple cannot describe - `plan_hash`,
    `values_hash`, the version and the seed are all blind to which artifact was in the store.
    """
    prepared = with_deals(bootstrapped(market(True)))
    ticked(prepared, 0.02)
    priced = baseval(with_deals(prepared))[1]

    ARTIFACTS.artifacts.clear()
    with pytest.raises(utils.CalibrationStale, match='no calibration artifact'):
        baseval(with_deals(prepared))

    # and a re-bootstrap of the same document publishes into the same slot, so the SAME execute runs
    assert baseval(with_deals(bootstrapped(prepared.params['Market Prices'])))[1] != priced, (
        'the refit reproduced the ride exactly - the tick never reached anything')


def test_an_evicted_slot_refuses_instead_of_silently_repricing():
    """THE REPLAY HOLE, closed. An artifact is evicted by an unrelated job filling the store, and
    the run that then prices the same document has an identical replay tuple to the one before it:
    same `plan_hash`, same `values_hash`, same version, same seed. Measured on the probe, the two
    marks differed by 13.4%.

    So the eviction cannot be allowed to produce a number. It refuses, and the run that DID ride
    reports the `artifact_id` it rode - which is the coordinate the replay tuple was missing.
    """
    context = derivus.Context()
    context.current_cfg = with_deals(bootstrapped(market(True)))
    ticked(context.current_cfg, 0.02)
    replay = (context.plan_hash(), context.values_hash())

    _, out = run(with_deals(context.current_cfg))
    rode = out['Stats']['Calibrations']
    assert rode == {CURVE_NAME: artifact_of(context.current_cfg).artifact_id}, rode

    ARTIFACTS.artifacts.pop(key_of(context.current_cfg))
    with pytest.raises(utils.CalibrationStale, match='no calibration artifact'):
        run(with_deals(context.current_cfg))
    assert (context.plan_hash(), context.values_hash()) == replay, (
        'the replay tuple moved on its own - this gate is not measuring the hole it names')


def test_a_reauthored_partner_block_takes_the_slot_with_it():
    """The slot names the SET, so re-authoring one member has to move the other's address too - an
    artifact whose `J` was fitted against quotes that no longer exist is exactly the one that must
    not be findable by the curve it still covers."""
    prepared = usd_config(usd_market({USD_OIS: 'Linear', USD_PROJ: 'Linear'}))
    assert ARTIFACTS.covering(utils.Factor('InterestRate', ('USD-3M',)))

    prepared.params['Market Prices'][USD_OIS]['instrument']['Points'][-1]['Use'] = 'No'
    for factor in ('USD-OIS', 'USD-3M'):
        with pytest.raises(utils.CalibrationStale, match='different plan'):
            prepared.propagated_factor(utils.Factor('InterestRate', (factor,)))


class Slot(object):
    """A stand-in artifact: the store reads a key and the factors an entry covers, and nothing
    else. Using one keeps the eviction discipline gate off the back of 33 real bootstraps."""

    def __init__(self, key):
        self.key = key
        self.factors = (utils.Factor('InterestRate', (key,)),)


def test_the_store_evicts_the_least_recently_used_and_not_the_oldest():
    """The store is bounded, so WHICH thing goes is a correctness property. A tick stream rides one
    slot over and over while unrelated jobs publish around it; FIFO throws that slot out on schedule
    and every eviction is a refusal the caller has to refit through. Gated because the mutation
    survived everything else: flipping `move_to_end` off passed the entire suite.
    """
    for index in range(ARTIFACTS.size):
        ARTIFACTS.put(Slot('slot-{}'.format(index)))
    assert len(ARTIFACTS.artifacts) == ARTIFACTS.size

    ARTIFACTS.get('slot-0')
    ARTIFACTS.put(Slot('slot-new'))
    assert ARTIFACTS.get('slot-0') is not None, 'a touched slot was evicted - the store is FIFO'
    assert 'slot-1' not in ARTIFACTS.artifacts, (
        'the untouched least-recently-used slot survived - nothing was evicted at all')


def test_a_ride_counts_as_use_of_the_slot_it_rode():
    """The end-to-end half: `covering` is a SCAN and touches nothing - content addressing picks the
    artifact, and only the pick is a use - so the discipline is gated where the pick happens, which
    is the ride.
    """
    prepared = with_deals(bootstrapped(market(True)))
    ticked(prepared, 0.02)
    key = key_of(prepared)

    for index in range(ARTIFACTS.size - 1):
        ARTIFACTS.put(Slot('slot-{}'.format(index)))
    assert next(iter(ARTIFACTS.artifacts)) == key, 'the real artifact is not the oldest entry'

    assert prepared.propagated_factor(FACTOR) is not None
    ARTIFACTS.put(Slot('slot-new'))
    assert ARTIFACTS.get(key) is not None, 'the slot a tick stream is riding was evicted under it'
    assert 'slot-0' not in ARTIFACTS.artifacts, 'nothing was evicted at all'


def test_the_interpolation_scheme_addresses_a_different_slot():
    """A Hermite solve and a linear one on the same quotes are different curves, and the block does
    not say which - `Price Factor Interpolation` does. Before it was in the key they shared a slot
    and the linear job rode the Hermite solve, 0.53bp away from its own answer.

    Driven through the engine rather than through `plan_key` alone, because what matters is that the
    second job cannot FIND the first one's operator.
    """
    hermite = ModelParams()
    hermite.append('InterestRate', (), 'HermiteRT')
    linear_job = bootstrapped(market(True))
    linear_nodes = linear_job.propagated_factor(FACTOR)[0].copy()

    hermite_job = bootstrapped(market(True), interpolation=hermite)
    assert np.abs(curve_values(hermite_job) - curve_values(linear_job)).max() > 1e-6, (
        'the two schemes solved the same curve - this gate proves nothing')
    assert key_of(hermite_job) != key_of(linear_job)
    assert np.array_equal(linear_job.propagated_factor(FACTOR)[0], linear_nodes), (
        'the linear job now rides the Hermite solve out of a shared slot')
