"""Does theta* reach the calculation still carrying its quotes, and does saying so cost nothing?

Stage C of the contract [Quote Sensitivities](../docs_src/developer/quote_sensitivities.md) puts
around the HW2F swaption calibration. Stage A closed the quote side of the residual, stage B made
the solve one differentiable node; this is THE ATTACHMENT - the arrow from that node to the
`HullWhite2FactorModelParameters` leaves a calculation consumes.

    the forward   a reference exposure run is BIT-IDENTICAL with the quote side on and off, and so
                  is the `Price Factors` block behind it - `np.array_equal`, not a tolerance.
    the greek     dV/dtheta is the number it always was, `np.array_equal` on the whole reported
                  gradient frame, and dV/dq arrives in the SAME backward pass.
    the mapping   the tensor published under a parameter's name is the one its leaf is minted from.
                  Nothing else can see a mis-split: the splice is worth zero WHATEVER is attached.
    the dedupe    a params factor reachable as an implied factor AND as a static dependent gets ONE
                  tensor, and it is the connected one.

WHY THERE IS NO RE-BOOTSTRAPPED BUMP LADDER HERE. `test_a_re_bootstrapped_quote_bump_is_not_a_
direction_reference_for_this` holds the measurement: four quotes against 23 parameters leaves a
19-dimensional solution manifold, so a re-solve at a bumped quote lands a FIXED ~0.09 away in theta
whatever the bump size, and the value change that displacement causes swamps - and reverses - the
one the quote causes. The well-posed direction check steps theta by what the quotes DO identify and
re-prices, which is stage B's predicted-step gate carried through to a value.

Run: ``pytest tests/test_swaption_quote_attachment.py -q``
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pytest
import torch

import derivus
from derivus import utils
from derivus.calculation import construct_calculation
from derivus.config import Config
from derivus.instruments import construct_instrument

from rates_world import par_swap
from test_swaption_quote_graph import BASE, BLOCK, CCY, CURVE, VOL, definitions, price_factors

DTYPE = torch.float64
PARAMS = 'HullWhite2FactorModelParameters.' + CURVE
IR = utils.Factor('InterestRate', (CURVE,))
#: The closure's own parameter order, which is `save_params`' order, which is the 23-vector's order.
KEYS = ('Alpha_1', 'Alpha_2', 'Correlation', 'Sigma_1', 'Sigma_2')
#: The book the quote deltas are for - one swap struck away from par, so CVA is not zero and not
#: symmetric in the volatility term structure the calibration solves for.
BOOK = [('IRS_4Y', 4, 9.10)]
TIME_GRID = '0d 3m(3m) 4y(6m)'


def pkey(name):
    return utils.Factor('HullWhite2FactorModelParameters', (CURVE, name))


def block(connect, bumps=()):
    return {'Swaption_Volatility': VOL, 'Generate_Instruments': 'No', 'Random_Seed': 5120,
            'Quote_Sensitivity': 'Yes' if connect else 'No',
            'Instrument_Definitions': definitions(bumps)}


def world(connect, bumps=()):
    """A bootstrapped ZAR world whose curve is simulated by the HW2F process the block calibrated.

    `Price Models` carries the process explicitly rather than leaning on the dummy entry
    `find_models` injects, because that dummy is `None` and the process reads `Lambda_1` off it.
    """
    config = Config(base_currency=CCY)
    config.params['System Parameters']['Base_Date'] = BASE
    config.params['Price Factors'] = price_factors()
    config.params['Market Prices'] = {BLOCK: {'instrument': block(connect, bumps), 'Children': []}}
    config.params['Bootstrapper Configuration'] = {'HullWhite2FactorModelParameters': {}}
    config.bootstrap()
    config.params['Price Models'] = {
        'HullWhite2FactorImpliedInterestRateModel.' + CURVE: {'Lambda_1': 0.0, 'Lambda_2': 0.0}}
    config.params['Model Configuration'].append(
        'InterestRate', (), 'HullWhite2FactorImpliedInterestRateModel')
    config.params['Price Factors']['SurvivalProb.CPTY'] = {
        'Recovery_Rate': 0.4, 'Curve': utils.Curve([], [[0.0, 0.0], [10.0, 0.3]])}
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
    return config


def overrides(gradient_variables='All', greeks=True):
    return {'Run_Date': BASE.strftime('%Y-%m-%d'), 'Time_grid': TIME_GRID, 'Batch_Size': 64,
            'Simulation_Batches': 1, 'Random_Seed': 1, 'Currency': CCY, 'MCMC_Simulations': 0,
            'Tenor_Offset': 0.0, 'Deflation_Interest_Rate': CURVE,
            'Gradient_Variables': gradient_variables,
            'Credit_Valuation_Adjustment': {
                'Calculate': 'Yes', 'Counterparty': 'CPTY', 'Deflate_Stochastically': 'No',
                'Stochastic_Hazard_Rates': 'No', 'Gradient': 'Yes' if greeks else 'No'}}


def run(config, gradient_variables='All', greeks=True):
    """One exposure run over the book, with the quote leaves cleared first.

    A quote leaf lives on the CONFIG and `.grad` accumulates, so a second run through the same world
    would report the sum of two jobs. Every gate here reads dV/dq of the run it just made.
    """
    for _, leaves in config.quote_leaves.values():
        for leaf in leaves:
            leaf.grad = None
    return derivus.run_cmc(config, prec=DTYPE,
                           overrides=overrides(gradient_variables, greeks))


def saved(config):
    """The 23-vector as `Price Factors` holds it - what a calculation mints its leaves from."""
    param = config.params['Price Factors'][PARAMS]
    return np.concatenate([np.atleast_1d(param[name]) if name in ('Alpha_1', 'Alpha_2',
                                                                  'Correlation')
                           else param[name].array[:, 1] for name in KEYS])


def quote_grad(config):
    """`dV/dq` off the quote leaves, in the order the block authored its benchmarks."""
    return np.array([float(leaf.grad) for _, leaves in config.quote_leaves.values()
                     for leaf in leaves])


def cva(out):
    return float(np.mean(out['Results']['cva']))


JOBS = {}


def job(connect):
    """`(config, out, dV/dq)` of one whole world, cached - the calibration is the slow part.

    The quote delta is captured WITH its run: `.grad` lives on the config and the next job through
    the same world clears it, so a gate reading it later would read whatever ran last.
    """
    if connect not in JOBS:
        config = world(connect)
        out = run(config)[1]
        JOBS[connect] = (config, out, quote_grad(config) if config.quote_leaves else np.array([]))
    return JOBS[connect]


# ---------------------------------------------------------------------------------------------
# The identities - no bump anywhere in them
# ---------------------------------------------------------------------------------------------

def test_the_job_is_bit_identical_with_the_quote_side_on():
    """Forward bit-identity end to end, which is the criterion the boundary correction set and the
    one this attachment has to meet: `leaf + (theta - theta.detach())` is worth EXACTLY zero, so a
    reference exposure run cannot move by a single bit.

    Both ends are checked because they fail differently - the saved `Price Factors` block is what
    the leaf is minted from, and the profile is what the whole chain made of it. The two guards say
    the comparison is between two DIFFERENT jobs: the plain run published no quote leaf, and the
    connected one actually reached its quotes.
    """
    plain_config, plain, _ = job(False)
    config, connected, delta = job(True)
    assert not plain_config.quote_leaves and not plain_config.calibrated_factors, (
        'the switch was ignored - both sides of this comparison are the same job')
    assert np.abs(delta).max() > 0, (
        'the connected run never reached the quotes, so this compares two identical jobs')
    assert np.array_equal(saved(plain_config), saved(config)), (
        'enabling quote gradients moved the calibration: {:.3g}'.format(
            np.abs(saved(plain_config) - saved(config)).max()))
    assert np.abs(plain['Results']['mtm'].values).max() > 0, (
        'an all-zero profile compares equal to anything')
    assert np.array_equal(plain['Results']['mtm'].values, connected['Results']['mtm'].values), (
        'the exposure profile moved: {:.3g}'.format(np.abs(
            plain['Results']['mtm'].values - connected['Results']['mtm'].values).max()))


def test_one_backward_reports_the_factor_greek_unchanged_and_the_quote_delta_beside_it():
    """The whole point of the splice: what reaches `backward()` changes, what is REPORTED cannot.

    So the reported gradient frame - every model parameter, the zero curve and the survival curve -
    is `np.array_equal` with the quote side on and off, and dV/dq is finite and non-zero on every
    benchmark in the same pass. Zero is the failure this workstream is about: a severed quote does
    not raise and does not move a value, it silently reports no sensitivity.
    """
    plain = job(False)[1]
    connected, delta = job(True)[1:]
    assert plain['Results']['grad_cva'].index.equals(connected['Results']['grad_cva'].index)
    assert np.array_equal(plain['Results']['grad_cva'].values,
                          connected['Results']['grad_cva'].values), (
        'the reported greek moved:\n{}\n{}'.format(plain['Results']['grad_cva'],
                                                   connected['Results']['grad_cva']))
    assert np.isfinite(delta).all() and (np.abs(delta) > 0.0).all(), delta


def test_the_published_parameters_are_the_ones_the_leaf_is_minted_from():
    """The split back to named parameters, and the ONLY gate that can see it go wrong.

    `factor_leaf` splices `theta - theta.detach()` onto the leaf, and that is worth zero WHATEVER
    tensor is attached - so a publish that put `Sigma_2`'s slice under `Sigma_1`'s key would leave
    every value in the job bit-identical and report the derivative of the wrong parameter. What
    pins it is the equality below: the tensor published under a name is, to the last bit, the
    number `Price Factors` holds under that name, which is what `current_value` mints its leaf from.

    A shape-preserving permutation - `Sigma_1` against `Sigma_2` - survives every other gate in this
    file, including the direction check, because a permutation is its own inverse on both sides of
    that comparison. Measured as a mutation, not reasoned about.
    """
    config = job(True)[0]
    param = config.params['Price Factors'][PARAMS]
    for name in KEYS:
        published = config.calibrated_factors[pkey(name)].detach().cpu().numpy()
        held = np.atleast_1d(param[name]) if published.size == 1 else param[name].array[:, 1]
        assert np.array_equal(published.astype(held.dtype), held), (name, published, held)
    assert set(config.calibrated_factors) == {pkey(name) for name in KEYS}, (
        'the publish does not name exactly the parameters the solve solved for')


def test_the_quote_side_costs_no_leaf_when_the_block_declines_it():
    """A quote leaf where no job asked for one is a residual graph nobody can see holding memory,
    and `Gradient_Variables` is the second switch: the implied leaves are only differentiable under
    `All` or `Implied`, so `Factors` reaches the curve and never the calibrated parameters."""
    plain_config = job(False)[0]
    assert not plain_config.quote_leaves and not plain_config.calibrated_factors
    config = job(True)[0]
    out = run(config, gradient_variables='Factors')[1]
    assert all(leaf.grad is None for _, leaves in config.quote_leaves.values() for leaf in leaves)
    assert 'InterestRate.' + CURVE in set(out['Results']['grad_cva'].index.get_level_values('Rate'))


# ---------------------------------------------------------------------------------------------
# The dedupe invariant
# ---------------------------------------------------------------------------------------------

def both_consumers():
    """A calculation reaching the params factor BOTH ways: as the HW2F process's implied factor and
    as an ordinary static dependent.

    Nothing in the deal tree pulls a `HullWhite2FactorModelParameters` block in as a dependent today
    - only a `<SpotModel>ModelParameters` conditional field does that, which is how the Heston-Nandi
    OSS pricer collides with its own process - so the collision is authored here, on the documented
    seam `HedgeMonteCarlo` builds its own dependency sets through. The invariant is the same one
    either way, and it is the failure mode the attachment has to be designed against rather than an
    edge case: minting a second leaf splits the quote gradient across two tensors, silently.
    """
    config = job(True)[0]
    params = dict(overrides(), Base_Date=BASE)
    calc = construct_calculation('Credit_Monte_Carlo', config, device=torch.device('cpu'),
                                 prec=DTYPE)
    calc.input_time_grid, calc.batch_size, calc.params = TIME_GRID, 64, params
    dependent, stochastic, implied, resets, settlements = config.calculate_dependencies(
        params, BASE, TIME_GRID)
    dependent[utils.Factor('HullWhite2FactorModelParameters', (CURVE,))] = []
    calc.update_time_grid(BASE, resets, settlements)
    return config, calc, calc._build_factor_state(
        dependent, stochastic, implied, params, BASE, 0, 1)


def test_the_attachment_mints_one_leaf_for_both_consumers():
    """ONE tensor, and it is the CONNECTED one.

    Three things have to hold together. The static dependent reuses the implied leaf rather than
    minting beside it; the pricer's `t_Static_Buffer` resolves to that same object, so a bump moves
    both consumers; and it carries the quote graph, so whichever of them a value reads through, the
    sensitivity sums into one place. The fourth is `all_var`: a leaf reachable twice must be
    differentiated once, or the reported gradient vector doubles.
    """
    config, calc, shared = both_consumers()
    quotes = [leaf for _, leaves in config.quote_leaves.values() for leaf in leaves]
    for name in KEYS:
        leaf = calc.implied_var[IR][pkey(name)]
        assert calc.static_var[pkey(name)] is leaf, '{}: the static leaf is a DUPLICATE'.format(name)
        assert shared.t_Static_Buffer[pkey(name)] is leaf, '{}: the pricer reads another'.format(name)
        assert leaf.grad_fn is not None, '{}: the one leaf is not the connected one'.format(name)
        reached = torch.autograd.grad(leaf.sum(), quotes, retain_graph=True, allow_unused=True)
        assert all(g is not None and float(g) != 0.0 for g in reached), name

    names = [name for name, _ in calc.all_var]
    assert len(names) == len(set(names)), 'a leaf reachable twice is differentiated twice'


# ---------------------------------------------------------------------------------------------
# The direction check, and the reference it is NOT
# ---------------------------------------------------------------------------------------------

QUOTE_JACOBIAN = {}


def quote_jacobian(config):
    """`dtheta/dq`, 23 x quotes, read one ROW at a time out of the published node's own backward.

    A cotangent IS a row, so the matrix costs one backward per parameter. Cached: each of those
    re-prices the benchmark set.
    """
    if not QUOTE_JACOBIAN:
        theta = torch.cat([config.calibrated_factors[pkey(name)] for name in KEYS])
        quotes = [leaf for _, leaves in config.quote_leaves.values() for leaf in leaves]
        QUOTE_JACOBIAN['dtheta'] = np.array([
            torch.stack(torch.autograd.grad(theta, quotes, grad_outputs=e, retain_graph=True)
                        ).double().cpu().numpy()
            for e in torch.eye(theta.numel(), dtype=theta.dtype, device=theta.device)])
    return QUOTE_JACOBIAN['dtheta']


def priced_at(theta):
    """The book's CVA with the CONSUMED parameters standing at `theta` - written into `Price
    Factors` as plain numpy, quotes off, nothing else moved and nothing re-solved.

    Its own world, not `job(False)`'s: this OVERWRITES the parameter block, and a gate comparing
    against the plain job must not depend on whether this ran first.
    """
    JOBS.setdefault('stepper', world(False))
    param = JOBS['stepper'].params['Price Factors'][PARAMS]
    param['Alpha_1'], param['Alpha_2'], param['Correlation'] = (
        float(theta[0]), float(theta[1]), float(theta[2]))
    param['Sigma_1'] = utils.Curve([], list(zip(param['Sigma_1'].array[:, 0], theta[3:13])))
    param['Sigma_2'] = utils.Curve([], list(zip(param['Sigma_2'].array[:, 0], theta[13:])))
    return cva(run(JOBS['stepper'], greeks=False)[1])


@pytest.mark.parametrize('bump', [0.5, 0.1])
def test_a_quote_bump_moves_the_value_the_way_the_quote_delta_says(bump):
    """The direction check, on the step a quote bump IDENTIFIABLY takes.

    `dV/dq . h` against what re-pricing at `theta* + dtheta/dq . h` actually does, for the quote
    whose delta dominates. Nothing re-solves, so the 19-dimensional flat the solver wanders on never
    enters, and the agreement improves LINEARLY as the bump shrinks - which is the second-order
    remainder behaving rather than a tolerance being met: 1.60 of the predicted move at one percent,
    1.32 at a half, 1.10 at a tenth. Measured.
    """
    config, connected, delta = job(True)
    j = int(np.argmax(np.abs(delta)))
    base = priced_at(saved(config))
    assert base == cva(connected), 'standing the plain world at theta* is not the connected world'
    moved = priced_at(saved(config) + quote_jacobian(config)[:, j] * (bump / 100.0))
    predicted = delta[j] * (bump / 100.0)
    assert (moved - base) * predicted > 0, (j, bump, moved - base, predicted)
    assert abs((moved - base) / predicted - 1.0) < 1.5 * bump, (j, bump, moved - base, predicted)


def test_a_re_bootstrapped_quote_bump_is_not_a_direction_reference_for_this():
    """The honest negative result, pinned so that it is a known property rather than a surprise.

    Bump a quote, re-run the same deterministic solve and difference the VALUE: four quotes against
    23 parameters leaves a 19-dimensional solution manifold, so theta lands a fixed ~0.09 away
    whatever the bump size and the value change that displacement causes swamps the one the quote
    causes. Measured on this fixture: +18.1 against +15.6 predicted at half a percent, then -5.6
    against +6.2 at a fifth - the sign REVERSES as the bump shrinks, which no tolerance can rescue.
    Stage B refuted the same comparison in theta space for the same reason.

    If this ever fails, the solve has started returning a function of its quotes and the comparison
    becomes available - or dV/dq is wrong, since a wrong one can make the pair agree by accident.
    Which is why the split back to named parameters is pinned on its own, above.
    """
    config, connected, delta = job(True)
    j = int(np.argmax(np.abs(delta)))
    identified = quote_jacobian(config)[:, j] * (0.2 / 100.0)
    bumped = world(False, ((j, 0.2),))
    assert np.linalg.norm(saved(bumped) - saved(config) - identified) > 10.0 * np.linalg.norm(
        identified), 'the re-solve now moves theta by what the quotes identify'
    predicted = delta[j] * (0.2 / 100.0)
    assert abs(cva(run(bumped, greeks=False)[1]) - cva(connected) - predicted) > abs(predicted), (
        'the re-solved value now agrees with dV/dq to better than itself - the comparison is '
        'available and this gate should be replaced by it')


# ---------------------------------------------------------------------------------------------
# The write-back
# ---------------------------------------------------------------------------------------------

def test_no_tensor_reaches_the_json_write_back(tmp_path):
    """`Price Factors` is DATA and gets written back out as JSON, which is why the connected half
    lives on the `Config` instead. The encoder does not raise on a tensor - it logs and emits
    `.Unknown` - so a publish into the wrong section would survive a round trip as a silently
    corrupted market data file rather than as a failure.
    """
    config = job(True)[0]
    path = str(tmp_path / 'market.json')
    config.write_marketdata_json(path)
    with open(path, 'rt') as f:
        assert '.Unknown' not in f.read(), 'something in Price Factors is not JSON'

    param = Config().read_json(path)['MarketData']['Price Factors'][PARAMS]
    assert np.array_equal(
        np.concatenate([np.atleast_1d(param[name]) if name in ('Alpha_1', 'Alpha_2', 'Correlation')
                        else param[name].array[:, 1] for name in KEYS]), saved(config))
