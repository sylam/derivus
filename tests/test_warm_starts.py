"""A family's own written factor is the warm start: the search is skipped, the polish runs from it.

THE RULING. There either are parameters in the price factors for that name or there are not. No new
field declares it - `HullWhite2FactorModelParameters.ZAR-JIBAR-3M` standing in `Price Factors` is
what makes the next fit of `HullWhite2FactorModelPrices.ZAR-JIBAR-3M` a warm start, and the same for
the energy family. A document carrying no such factor is the cold path, untouched to the bit.

WHAT IT BUYS, MEASURED HERE. The Hull-White chain is basin hopping (50 hops, each one L-BFGS-B) then
least squares; warm it is the least squares alone from the written factor. On the four-quote fixture
that is **421 objective evaluations cold against 6 warm**, landing 9.4e-9 apart in the sup norm over
all 23 coordinates. The energy fit is one bounded `minimize` from the declared `Seed`; warm it starts
at the written (Sigma, Alpha) clipped to the two boxes, and that is **69 evaluations against 15**,
9.3e-11 and 4.6e-10 apart.

WHAT A MOVED LADDER SHOWS, AND IT IS NOT WHAT A BASIN STORY PREDICTS. Every quoted vol x 1.02: the
cold chain lands 0.0012 from the unmoved theta* and the warm fit 0.0432, almost all of it in
`Alpha_1`. Neither displacement means anything - four quotes leave a 19-dimensional null space and
[the re-solve reference is refuted](../docs_src/developer/quote_sensitivities.md#the-manifold-finding)
- and the residual says which fit is better: the warm one reaches loss 1.31e-14 where the cold chain
stops at 2.83e-13, both stationary (||J'r|| 6.4e-8 and 1.2e-7 against `Stationarity_Tol` 1e-3), in
**49 evaluations against 429**. The cold chain does not visibly wander because `Random_Seed` pins its
search and a 2% move barely reshapes the objective, so it retraces its own walk; what the warm fit
does that no chain can is reach a stationary point that reprices the moved ladder without paying for
the search. That is the intraday property.

THE KILLING MUTATIONS, each measured by making the change and reading the number back:

| mutation | what dies |
| --- | --- |
| run the basin hopping despite the factor (`[basin, leastsq]` unconditionally) | warm reads 429 evaluations against cold 421, and 443 against 429 on the moved ladder; both Hull-White gates |
| skip the search on a COLD block too (`[leastsq]` unconditionally) | the cold count goes 421 -> 47 and the moved pair to 32 against 38; three gates, two of them on the cold half |
| ignore the written factor in the energy seed | the warm count goes 15 -> 69, equal to cold, and no warm line is logged |
| read the energy factor transposed (`Alpha` as the sigma seed) | the warm count goes 15 -> 60 - a seed off the manifold costs what the box is wide |

EVALUATIONS, NOT SECONDS. Both counts come from a `sys.setprofile` hook counting calls to one named
engine function - `precalculate` runs once per swaption objective evaluation, `black_european_option_price`
once per quote per energy objective evaluation - so the reading is the same on a loaded box as on an
idle one. Nothing is patched: the hook observes.
"""
import collections
import json as jsonlib
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch

import derivus
from derivus import riskfactors, utils
from derivus.bootstrappers import HullWhite2FactorModelParameters, SwaptionCalibration
from derivus.config import CustomJsonEncoder, ModelParams
from derivus.stochasticprocess import HullWhite2FactorImpliedInterestRateModel

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures')
HW_JOB = os.path.join(FIXTURES, 'hw2f_four_quote_job.json')
HW_BLOCK = 'HullWhite2FactorModelPrices.ZAR-JIBAR-3M'
HW_FACTOR = 'HullWhite2FactorModelParameters.ZAR-JIBAR-3M'
CS_JOB = os.path.join(FIXTURES, 'cs_energy_job.json')
CS_BLOCK = 'CSForwardPriceModelPrices.BRENT'
CS_FACTOR = 'CSForwardPriceModelParameters.BRENT'
#: the (sigma, alpha) `cs_energy_job.json`'s quotes were generated off - a round trip, so the
#: energy fit has a known answer and not only a repeatable one
CS_TRUE = (0.35, 0.25)
#: every quoted vol scaled by this, which is the intraday move both families are fitted through
MOVE = 1.02


def counted(run, *targets):
    """`(answer, {name: calls})` - a profile hook over named engine functions.

    The previous hook is restored rather than cleared, so a coverage or profiling run that wraps
    this suite keeps its own.
    """
    codes = {target.__code__: target.__qualname__ for target in targets}
    tally = collections.Counter()

    def hook(frame, event, arg):
        if event == 'call' and frame.f_code in codes:
            tally[codes[frame.f_code]] += 1

    previous = sys.getprofile()
    sys.setprofile(hook)
    try:
        return run(), tally
    finally:
        sys.setprofile(previous)


def document(path, block, factor_name=None, factor=None, move=1.0, quote='Market_Volatility'):
    """The fixture as JSON text, optionally with a previous factor written into it and every quote
    moved - the two axes every gate below reads."""
    cfg = jsonlib.load(open(path))
    market = cfg['Calc']['MergeMarketData']['ExplicitMarketData']
    if move != 1.0:
        for row in market['Market Prices'][block]['instrument'][
                'Instrument_Definitions' if quote == 'Market_Volatility'
                else 'Energy_Futures_Options']:
            if quote == 'Market_Volatility':
                row[quote]['.Percent'] *= move
            else:
                row[quote] *= move
    if factor is not None:
        market['Price Factors'][factor_name] = factor
    return jsonlib.dumps(cfg, cls=CustomJsonEncoder)


def carried(factor):
    """A written price factor as a document carries it - through the encoder, because a warm start
    is a factor that was SAVED and read back, curves and all."""
    return jsonlib.loads(jsonlib.dumps(factor, cls=CustomJsonEncoder))


def fit(text, factor_name, *targets):
    """`(written factor, {name: calls})` - one document bootstrapped through `derivus.Context`."""
    context = derivus.Context()
    context.load_json((text, 'warm_starts.json'))
    _, tally = counted(context.current_cfg.bootstrap, *targets)
    return context.current_cfg.params['Price Factors'][factor_name], tally


def hw_fit(factor=None, move=1.0):
    written, tally = fit(
        document(HW_JOB, HW_BLOCK, HW_FACTOR, factor, move), HW_FACTOR,
        HullWhite2FactorImpliedInterestRateModel.precalculate)
    return written, tally[HullWhite2FactorImpliedInterestRateModel.precalculate.__qualname__]


def cs_fit(factor=None, move=1.0):
    written, tally = fit(
        document(CS_JOB, CS_BLOCK, CS_FACTOR, factor, move, quote='Quoted_Market_Value'),
        CS_FACTOR, utils.black_european_option_price)
    rows = len(jsonlib.loads(open(CS_JOB).read())['Calc']['MergeMarketData']['ExplicitMarketData'][
        'Market Prices'][CS_BLOCK]['instrument']['Energy_Futures_Options'])
    # one Black per quote prices the market premium and one per quote logs the fit; the rest is
    # the objective, which values every quote per evaluation
    calls = tally[utils.black_european_option_price.__qualname__]
    return written, (calls - 2 * rows) // rows


def hw_theta(factor):
    """theta* flat, in the order `SwaptionCalibration` concatenates it in."""
    return np.concatenate([[factor['Alpha_1']], [factor['Alpha_2']], [factor['Correlation']],
                           factor['Sigma_1'].array[:, 1], factor['Sigma_2'].array[:, 1]])


def hw_named(factor):
    return {'Alpha_1': np.array([factor['Alpha_1']]), 'Alpha_2': np.array([factor['Alpha_2']]),
            'Correlation': np.array([factor['Correlation']]),
            'Sigma_1': factor['Sigma_1'].array[:, 1], 'Sigma_2': factor['Sigma_2'].array[:, 1]}


def hw_calibration(move):
    """The four-quote block's own residual, built the way `calc_loss` builds it, on the MOVED
    quotes - so a landed theta can be scored where it was fitted."""
    context = derivus.Context()
    context.load_json((document(HW_JOB, HW_BLOCK, move=move), 'warm_residual.json'))
    params = context.current_cfg.params
    price_factors = params['Price Factors']
    base_date = params['System Parameters']['Base_Date']
    rate = utils.check_rate_name('InterestRate.ZAR-JIBAR-3M')
    ir_factor = utils.Factor('InterestRate', rate[1:])
    interp = ModelParams()
    ir_curve = riskfactors.construct_factor(ir_factor, price_factors, interp)
    block = params['Market Prices'][HW_BLOCK]['instrument']
    vol = riskfactors.construct_factor(
        utils.Factor('InterestYieldVol', utils.check_rate_name(block['Swaption_Volatility'])),
        price_factors, interp)
    vol.delta = 0.0
    vol.set_premiums(None, ir_curve.get_currency())
    model = HullWhite2FactorModelParameters({}, torch.device('cpu'), torch.float64)
    implied_obj, process, vol_tenors = model.implied_process(
        'ZAR', price_factors, {}, ir_curve, rate)
    mtm = set([base_date + row['Start'] for row in block['Instrument_Definitions']])
    time_grid = utils.TimeGrid(mtm, mtm, mtm)
    time_grid.set_base_date(base_date, delta=(10, vol_tenors * utils.DayCount.DAYS_IN_YEAR))
    objective, optimizers, implied_var, swaps = model.calc_loss(
        {'instrument': block}, base_date, time_grid, process, implied_obj, ir_factor, vol)
    return SwaptionCalibration('warm', objective, implied_var, optimizers, process, swaps)


def scored(calibration, factor):
    """`(loss, ||J'r||)` at that theta on the calibration's own quotes."""
    named = hw_named(factor)
    x = torch.tensor(np.concatenate([named[key] for key in calibration.keys]),
                     dtype=torch.float64).requires_grad_(True)
    residual = calibration(x)
    jacobian = torch.autograd.grad(
        residual, x, torch.eye(residual.numel(), dtype=torch.float64), is_grads_batched=True)[0]
    return (float(calibration.objective.reduce(residual.detach())),
            float(torch.linalg.norm(jacobian.t() @ residual.detach())))


@pytest.fixture(scope='module')
def hull_white():
    """`(cold factor, cold evaluations)` on the four-quote fixture as the bank carries it - no
    parameter factor in the document, so this is the cold chain and the hex set's own reading."""
    return hw_fit()


@pytest.fixture(scope='module')
def energy():
    return cs_fit()


def test_the_written_hull_white_factor_skips_the_search(hull_white, caplog):
    """Warm lands on cold, at a fraction of the evaluations, and says so.

    The cold assertion is the other half: skipping the search on a block with no factor reads 47
    evaluations here, not 421.
    """
    cold, cold_evaluations = hull_white
    assert cold_evaluations > 100, (
        'the cold chain must run the basin search: {} evaluations'.format(cold_evaluations))
    with caplog.at_level(logging.INFO):
        warm, warm_evaluations = hw_fit(factor=carried(cold))
    assert 'warm start off ' + HW_FACTOR in caplog.text, 'the fit must report the warm start'
    assert warm_evaluations * 10 < cold_evaluations, (
        'warm {} evaluations against cold {}'.format(warm_evaluations, cold_evaluations))
    gap = np.abs(hw_theta(warm) - hw_theta(cold)).max()
    assert gap < 1e-7, 'warm landed {:.3g} from cold, not within 1e-7'.format(gap)


def test_the_warm_hull_white_fit_reprices_a_moved_ladder_without_the_search(hull_white):
    """A 2% move in every quoted vol, fitted from the previous theta* and fitted cold.

    What is asserted is the FIT and not the displacement: on four quotes theta* lives on a
    19-dimensional manifold and the two land 0.0432 and 0.0012 from the unmoved answer with no
    ordering between them. The warm fit reaches the lower loss, is stationary, and pays 49
    evaluations where the chain pays 429.
    """
    cold, _ = hull_white
    moved_cold, cold_evaluations = hw_fit(move=MOVE)
    moved_warm, warm_evaluations = hw_fit(factor=carried(cold), move=MOVE)
    assert warm_evaluations * 4 < cold_evaluations, (
        'warm {} evaluations against cold {}'.format(warm_evaluations, cold_evaluations))

    calibration = hw_calibration(MOVE)
    cold_loss, cold_gradient = scored(calibration, moved_cold)
    warm_loss, warm_gradient = scored(calibration, moved_warm)
    tolerance = float(jsonlib.loads(document(HW_JOB, HW_BLOCK))['Calc']['MergeMarketData'][
        'ExplicitMarketData']['Market Prices'][HW_BLOCK]['instrument'].get(
            'Stationarity_Tol', 1e-3))
    assert warm_loss <= cold_loss, (
        'the warm fit must reprice the moved ladder at least as well: {:.3g} against {:.3g}'.format(
            warm_loss, cold_loss))
    assert max(warm_gradient, cold_gradient) < tolerance, (
        'both must be stationary: warm {:.3g}, cold {:.3g}'.format(warm_gradient, cold_gradient))
    # the displacements are RECORDED, not ordered - neither is a well-posed quantity here
    logging.info('moved: ||theta_cold - theta*|| %.4g, ||theta_warm - theta*|| %.4g',
                 np.linalg.norm(hw_theta(moved_cold) - hw_theta(cold)),
                 np.linalg.norm(hw_theta(moved_warm) - hw_theta(cold)))


def test_the_written_energy_factor_seeds_the_minimisation(energy, caplog):
    """The energy fit recovers the (sigma, alpha) its quotes were generated off, and warm gets
    there from the written factor in a fraction of the evaluations."""
    cold, cold_evaluations = energy
    assert abs(cold['Sigma'] - CS_TRUE[0]) < 1e-6 and abs(cold['Alpha'] - CS_TRUE[1]) < 1e-6, (
        'the round trip must recover ({}, {}): got ({}, {})'.format(
            CS_TRUE[0], CS_TRUE[1], cold['Sigma'], cold['Alpha']))
    with caplog.at_level(logging.INFO):
        warm, warm_evaluations = cs_fit(factor=dict(cold))
    assert 'warm start off ' + CS_FACTOR in caplog.text, 'the fit must report the warm start'
    assert warm_evaluations * 2 < cold_evaluations, (
        'warm {} evaluations against cold {}'.format(warm_evaluations, cold_evaluations))
    assert abs(warm['Sigma'] - cold['Sigma']) < 1e-8 and abs(warm['Alpha'] - cold['Alpha']) < 1e-8


def test_a_document_with_no_written_factor_is_the_cold_path(hull_white, energy):
    """The fixtures carry no parameter factor, so both fits above ran the path they always ran -
    which is what the hex set holds to the bit, and what this asserts in the family's own terms."""
    for path, block, factor in ((HW_JOB, HW_BLOCK, HW_FACTOR), (CS_JOB, CS_BLOCK, CS_FACTOR)):
        carried_factors = jsonlib.load(open(path))['Calc']['MergeMarketData'][
            'ExplicitMarketData']['Price Factors']
        assert factor not in carried_factors, '{} must be fitted cold'.format(block)
    assert hull_white[1] > 100 and energy[1] > 40, (
        'both cold fits must run their full search: {} and {}'.format(hull_white[1], energy[1]))
