"""Does one backward pass report dV/dq, and is it the derivative of the number reported?

This is the last arrow of the chain `q -> bootstrap -> theta* -> factor buffers -> scenarios -> V`.
The middle arrow is `CalibrationSolve`: its forward is the ordinary damped Newton and its backward
is the implicit function theorem at the fixed point, so nothing is unrolled and nothing is bumped.
The last arrow is `Calculation.factor_leaf`, which offers the already-connected theta* where the
engine would otherwise mint a fresh leaf out of numpy.

Verification is a TRIANGLE, and its three corners are deliberately independent:

    one-pass dV/dq          the number under test - `q.grad` after a single backward()
    dV/dtheta . dtheta/dq   the greek the engine already reported, contracted against a
                            finite-difference calibration Jacobian (two re-bootstraps per quote)
    CRN quote-bump ladder   re-author, re-bootstrap, re-price at q +/- h, scored on agreement AND
                            flatness by `crn_ladder`

plus three identities that do not need a bump at all: the solved curve is BIT-IDENTICAL with quote
gradients on and off, a reference exposure run is `np.array_equal` either way, and the benchmark
self-delta matrix is the identity - which is the IFT equation itself, read through the full chain.

The last gate is the one that says the rest can see anything: a sign flip in the backward linear
solve has to fail the ladder.
"""
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pytest
import torch

import derivus
from derivus import utils
from derivus.bootstrappers import BenchmarkInstruments, CalibrationSolve, quote_nodes
from derivus.config import Config
from derivus.instruments import construct_instrument

from crn_ladder import ladder
from rates_world import BASE, par_swap
from test_interest_rate_prices import authored_world

DTYPE = torch.float64
CCY = 'ZAR'
CURVE = 'ZAR-JIBAR-3M'
CURVE_NAME = 'InterestRate.' + CURVE
BLOCK = 'InterestRatePrices.' + CURVE

#: The book the quote deltas are for - two par swaps struck away from par, so V is not zero and
#: not symmetric in the curve. 4Y and 6Y both fall BETWEEN knots, so a bucket delta has to be
#: interpolated rather than read off, which is where a mis-ordered theta would show up.
BOOK = [('IRS_4Y', 4, 9.10), ('IRS_6Y', 6, 9.80)]


def market(connect, bumps=None):
    """The ZAR world's `Market Prices`, optionally with quote `j` moved by `h` percent."""
    market_prices, _, _ = authored_world('zar')
    block = market_prices[BLOCK]['instrument']
    block['Quote_Sensitivity'] = 'Yes' if connect else 'No'
    for j, h in (bumps or ()):
        block['Points'][j]['Quoted_Market_Value'] += h
    return market_prices


def bootstrapped(connect, bumps=None, market_prices=None):
    """A `Config` whose curve has been solved from its quotes, the way a job reaches this."""
    config = Config(base_currency=CCY)
    config.params['System Parameters']['Base_Date'] = BASE
    config.params['Price Factors'] = {'FxRate.{}'.format(CCY): {
        'Domestic_Currency': None, 'Interest_Rate': CURVE, 'Priority': 1, 'Spot': 1.0}}
    config.params['Market Prices'] = market_prices or market(connect, bumps)
    config.params['Bootstrapper Configuration'] = {'InterestRateCurveParameters': {}}
    config.bootstrap()
    return config


def with_deals(config, book=None):
    config.deals = {'Attributes': {'Reference': 'test', 'Tag_Titles': ''},
                    'Deals': {'Children': [{'Instrument': construct_instrument(
                        par_swap(ref, CCY, CURVE, CURVE, years, rate, day_count='ACT_365'), {})}
                        for ref, years, rate in (book or BOOK)]},
                    'Calculation': {'Base_Date': BASE, 'Currency': CCY}}
    return config


def baseval(config, greeks='First'):
    """The portfolio value, and the calculation the greeks hang off."""
    calc, out = derivus.run_baseval(
        config, prec=DTYPE, overrides={'MCMC_Simulations': 1, 'Random_Seed': 1, 'Greeks': greeks})
    rows = out['Results']['mtm']
    # the root Aggregation reports no value of its own, so the portfolio IS the sum of its
    # children - which is the scalar `resolve_structure` handed to `backward()`
    return calc, float(rows[rows['Parent'] == 'root']['Value'].sum())


def quote_grad(config):
    """`dV/dq` off the quote leaf, in the order the block authored its `Points`."""
    return config.quote_leaves[BLOCK][1].grad.detach().cpu().numpy().copy()


def curve_of(config):
    return config.params['Price Factors'][CURVE_NAME]['Curve'].array[:, 1].copy()


def value_at(bumps):
    """The book's value with the quotes moved - re-authored, re-bootstrapped and re-priced, so
    nothing about the connected run is reused."""
    return baseval(with_deals(bootstrapped(False, bumps)), greeks='No')[1]


# ---------------------------------------------------------------------------------------------
# The identities - no bump anywhere in them
# ---------------------------------------------------------------------------------------------

def test_the_round_trip_still_recovers_the_curve_and_gradients_do_not_move_it():
    """Two acceptance criteria at once, and the second is the reason the first is repeated here.

    `CalibrationSolve.forward` IS `damped_newton` - same iterations, same tolerances, same float64
    - so theta* cannot depend on whether anything is watching. `np.array_equal` says so at the last
    bit, which is a stronger statement than the 1e-10 the round trip asks for and a different one:
    a solve that drifted by 1e-15 would pass the round trip and fail here.
    """
    _, true_factors, _ = authored_world('zar')
    plain, connected = curve_of(bootstrapped(False)), curve_of(bootstrapped(True))
    assert np.array_equal(plain, connected), (
        'enabling quote gradients moved the solve: {}'.format(np.abs(plain - connected).max()))
    error = np.abs(connected - true_factors[CURVE_NAME]['Curve'].array[:, 1]).max()
    assert error < 1e-10, 'recovered to {:.3g}, not 1e-10'.format(error)


def test_a_reference_exposure_run_is_bit_identical_with_quote_gradients_on():
    """Forward bit-identity through the WHOLE stochastic chain, which is the criterion the
    boundary-correction gate set and the one this attachment has to meet.

    A Hull-White curve makes theta* the drift of every scenario rather than a discount factor at
    t0, so this exercises `_build_factor_state`'s stochastic branch and the xVA block behind it -
    the CVA path is do-not-touch, and the way to keep it that way is to change what reaches
    `backward()` and nothing about what is reported.
    """
    plain, plain_config = cmc_exposure(False)
    connected, config = cmc_exposure(True)
    assert BLOCK not in plain_config.quote_leaves, (
        'the switch was ignored - both sides of this comparison are the same job')
    assert np.abs(quote_grad(config)).max() > 0, (
        'the connected run never reached the quotes, so this compares two identical jobs')
    assert np.abs(plain).max() > 0, 'an all-zero profile compares equal to anything'
    assert np.array_equal(plain, connected), (
        'the exposure profile moved: {:.3g}'.format(np.abs(plain - connected).max()))


def test_the_benchmark_self_delta_matrix_is_the_identity():
    """The IFT equation itself, read through the full chain and without differencing anything.

    A benchmark is at par, so `PV_i(theta*(q), q_i) = 0` for every q, and differentiating that
    total derivative gives `dPV_i/dtheta . dtheta/dq_j = -(dPV_i/dq_i) delta_ij`. The left side is
    what a calculation reports when it prices benchmark i as an ORDINARY DEAL at a frozen rate -
    the deal carries a number, not a quote - so the reported quote delta must be diagonal, and
    dividing by each instrument's own quote sensitivity makes it exactly the identity.

    The normaliser comes from a SECANT on the fixed solved curve, not from the same machinery:
    a benchmark's PV is affine in its quote, so `PV(q+1) - PV(q)` is that derivative exactly.
    """
    config = bootstrapped(True)
    points = config.params['Market Prices'][BLOCK]['instrument']['Points']
    quotes = config.quote_leaves[BLOCK][1]
    interp = config.params['Price Factor Interpolation']

    priced = [BenchmarkInstruments(
        quote_nodes(points, CURVE, shift), config.params['Price Factors'], interp,
        BASE, CCY, {}, [], torch.device('cpu'))({}) for shift in (0.0, 1.0)]
    own = (priced[1] - priced[0]).detach().numpy()

    rows = []
    for point, node in zip(points, quote_nodes(points, CURVE)):
        quotes.grad = None
        config.deals = {'Attributes': {'Reference': 'test', 'Tag_Titles': ''},
                        'Deals': {'Children': [node]},
                        'Calculation': {'Base_Date': BASE, 'Currency': CCY}}
        _, value = baseval(config)
        assert abs(value) < 1e-6, 'benchmark {} is not at par: {}'.format(point['Descriptor'], value)
        rows.append(quote_grad(config))

    identity = np.array(rows) / -own.reshape(-1, 1)
    assert np.abs(identity - np.eye(len(points))).max() < 1e-6, (
        'the self-delta matrix is not the identity:\n{}'.format(np.round(identity, 8)))


def test_a_tenor_offset_declines_the_attachment():
    """The guard, gated rather than reasoned about.

    `Tenor_Offset` shifts every tenor before the leaf is minted, so the curve the calculation
    consumes is a DIFFERENT one and `dtheta_shifted/dq` is not `dtheta/dq`. Quote sensitivities are
    t0 risk. Attaching anyway would report a plausible number that is the derivative of something
    nobody priced, so the attachment declines and the quote leaf is simply never reached.
    """
    _, config = cmc_exposure(True, tenor_offset=10.0)
    assert config.quote_leaves[BLOCK][1].grad is None, (
        'a shifted curve claimed the unshifted curve\'s quote derivative')


# ---------------------------------------------------------------------------------------------
# The triangle
# ---------------------------------------------------------------------------------------------

def finite_difference_calibration_jacobian(h=1e-4):
    """`dtheta/dq` by central difference on the SOLVE - two full bootstraps per quote, and no part
    of the implicit-function path touched."""
    points = market(False)[BLOCK]['instrument']['Points']
    columns = []
    for j in range(len(points)):
        up, down = curve_of(bootstrapped(False, [(j, h)])), curve_of(bootstrapped(False, [(j, -h)]))
        columns.append((up - down) / (2.0 * h))
    return np.array(columns).T


def test_the_triangle_closes():
    """Corner one against corner two: the one-pass quote delta against the greek the engine already
    reported, contracted with a finite-difference calibration Jacobian.

    This is the corner that isolates the IFT. `dV/dtheta` is the ordinary factor greek and is not
    under test; `dtheta/dq` is differenced off the solve and knows nothing about `CalibrationSolve`.
    So a disagreement here is the linear solve or the VJP and nothing else.
    """
    config = with_deals(bootstrapped(True))
    calc, _ = baseval(config)
    one_pass = quote_grad(config)

    dv_dtheta = calc.netting_sets.obj.Calc_res['Greeks_First'][CURVE_NAME]
    contracted = dv_dtheta @ finite_difference_calibration_jacobian()

    scale = np.abs(one_pass).max()
    assert np.abs(one_pass - contracted).max() / scale < 1e-6, (
        'dV/dq and dV/dtheta . dtheta/dq disagree:\n{}\n{}'.format(one_pass, contracted))


@pytest.mark.parametrize('j', [3, 4])
def test_the_quote_bump_ladder_is_flat_and_lands_on_the_one_pass_delta(j):
    """Corner three, and the only one that re-runs the whole job. Agreement and flatness are
    reported separately because agreement at one bump size proves nothing: a ladder that scatters
    with h is differencing across something that is not there.

    Two buckets, because a single one can agree by accident - the 3Y and 5Y knots are the ones a
    4Y and a 6Y swap actually straddle.
    """
    config = with_deals(bootstrapped(True))
    baseval(config)
    aad = float(quote_grad(config)[j])
    base = market(False)[BLOCK]['instrument']['Points'][j]['Quoted_Market_Value']

    result = ladder(price=lambda q: value_at([(j, q - base)]), aad=aad, base=base)
    assert result.flatness < 1e-6, 'the quote ladder is not converging\n{}'.format(result)
    assert result.agrees(tol=1e-6, flat_tol=1e-6), str(result)


# ---------------------------------------------------------------------------------------------
# The mutation - can any of the above see the subsystem break?
# ---------------------------------------------------------------------------------------------

class SignFlipped(CalibrationSolve):
    """`CalibrationSolve` with the backward linear solve's sign reversed - `dL/dq = +(dF/dq)^T w`.

    The forward pass is untouched, so every value in the job is identical and only the reported
    derivative is wrong: exactly the failure mode the whole increment exists to make visible, and
    exactly the one a price gate cannot see.
    """

    @staticmethod
    def backward(ctx, cotangent):
        grads = CalibrationSolve.backward(ctx, cotangent)
        return grads[:-1] + (-grads[-1],)


def solve_through(wrapper):
    """`(theta*, quotes)` from one benchmark set solved through `wrapper`."""
    market_prices = market(True)
    block = market_prices[BLOCK]['instrument']
    price_factors = {'FxRate.{}'.format(CCY): {
        'Domestic_Currency': None, 'Interest_Rate': CURVE, 'Priority': 1, 'Spot': 1.0}}
    config = bootstrapped(False, market_prices=copy.deepcopy(market_prices))
    price_factors[CURVE_NAME] = config.params['Price Factors'][CURVE_NAME]

    factor = utils.Factor('InterestRate', (CURVE,))
    benchmarks = BenchmarkInstruments(
        quote_nodes(block['Points'], CURVE), price_factors,
        Config().params['Price Factor Interpolation'], BASE, CCY, {}, [factor],
        torch.device('cpu'),
        quotes=[point['Quoted_Market_Value'] for point in block['Points']],
        bumped_nodes=quote_nodes(block['Points'], CURVE, 1.0))
    seed = {factor: torch.tensor(benchmarks.factors[factor].current_value(), dtype=DTYPE)}
    return wrapper.apply(benchmarks, seed, 50, 1e-14, 6, benchmarks.quotes), benchmarks.quotes


@pytest.mark.parametrize('wrapper,should_agree', [(CalibrationSolve, True), (SignFlipped, False)])
def test_a_sign_flip_in_the_backward_solve_fails_the_ladder(wrapper, should_agree):
    """MUTATE the linear solve. Reported like every other mutation, and parametrised against the
    unflipped wrapper so the gate cannot pass by measuring nothing.

    Scored on `dtheta/dq` rather than on `dV/dq`, because that is where the sign lives: one row of
    the calibration Jacobian against the same row differenced off two full bootstraps.
    """
    theta, quotes = solve_through(wrapper)
    j = 4
    cotangent = torch.zeros_like(theta)
    cotangent[j] = 1.0
    theta.backward(cotangent)
    row = quotes.grad.detach().cpu().numpy()

    reference = finite_difference_calibration_jacobian()[j]
    scale = np.abs(reference).max()
    agrees = np.abs(row - reference).max() / scale < 1e-6
    assert agrees == should_agree, (
        '{} {} the finite-difference calibration Jacobian:\n{}\n{}'.format(
            wrapper.__name__, 'agrees with' if agrees else 'disagrees with', row, reference))


# ---------------------------------------------------------------------------------------------
# The stochastic-curve run the bit-identity gate uses
# ---------------------------------------------------------------------------------------------

def cmc_exposure(connect, tenor_offset=0.0):
    """A collateral-free netting set of the same two swaps under a Hull-White curve, with CVA on.

    Returned as the mean exposure profile, which is what a reference run compares. `Gradient: Yes`
    on both sides of the comparison: the toggle under test is the QUOTE graph, not whether greeks
    were asked for, and comparing a greeks-on run against a greeks-off one would be measuring the
    boundary machinery instead.
    """
    config = bootstrapped(connect)
    netting = {'Object': 'NettingCollateralSet', 'Reference': 'test', 'Agreement_Currency': CCY,
               'Apply_Closeout_When_Uncollateralized': 'No', 'Balance_Currency': CCY,
               'Opening_Balance': 0.0, 'Collateralized': 'False', 'Netted': 'True',
               'Calendars': None}
    config.deals = {'Attributes': {'Reference': 'test', 'Tag_Titles': ''},
                    'Deals': {'Children': [{
                        'Instrument': construct_instrument(netting, {}),
                        'Children': [{'Instrument': construct_instrument(
                            par_swap(ref, CCY, CURVE, CURVE, years, rate, day_count='ACT_365'), {})}
                            for ref, years, rate in BOOK]}]},
                    'Calculation': {'Base_Date': BASE, 'Currency': CCY}}
    config.params['Price Factors']['SurvivalProb.CPTY'] = {
        'Recovery_Rate': 0.4, 'Curve': utils.Curve([], [[0.0, 0.0], [10.0, 0.3]])}
    config.params['Price Models'] = {
        'HullWhite1FactorInterestRateModel.{}'.format(CURVE): {
            'Alpha': 0.05, 'Lambda': 0.0, 'Sigma': utils.Curve([], [[0.0, 0.01]]),
            'Quanto_FX_Correlation': 0.0, 'Quanto_FX_Volatility': None}}
    config.params['Model Configuration'].append(
        'InterestRate', (), 'HullWhite1FactorInterestRateModel')

    _, out = derivus.run_cmc(config, prec=DTYPE, overrides={
        'Run_Date': BASE.strftime('%Y-%m-%d'), 'Time_grid': '0d 3m(3m) 6y(6m)',
        'Batch_Size': 128, 'Simulation_Batches': 1, 'Random_Seed': 1, 'Currency': CCY,
        'MCMC_Simulations': 0, 'Tenor_Offset': tenor_offset, 'Deflation_Interest_Rate': CURVE,
        'Gradient_Variables': 'Factors',
        'Credit_Valuation_Adjustment': {
            'Calculate': 'Yes', 'Counterparty': 'CPTY', 'Deflate_Stochastically': 'No',
            'Stochastic_Hazard_Rates': 'No', 'Gradient': 'Yes'}})
    return out['Results']['mtm'].values, config
