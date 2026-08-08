"""Does the curve family recover the curve its quotes were priced off?

A bootstrap has exactly one honest gate, and it is a round trip: construct a zero curve, PRICE the
benchmark set off it to GENERATE the quotes, then require the solve to recover the curve it started
from. Nothing here is copied from a vendor file and nothing needs to be - the levels only have to be
plausibly shaped, because the quotes are derived from them rather than asserted against them.

Two worlds, because the two configurations fail differently. A USD SOFR-style world is genuinely
multi-curve: an OIS discount curve solved from compounded-in-arrears OIS quotes, then a projection
curve solved from a FRA strip and par swaps that discount on it, which only works if the two solves
run in dependency order. A ZAR JIBAR-style world is the degenerate single-curve one, where discount
and projection coincide - the harder solve, because the unknown appears on both sides of every
benchmark.

Generating the quotes needs no second pricer and no root find. A benchmark's PV is AFFINE in its
quote - a deposit's coupons, an FRA's strike and a swap's fixed leg are each linear in it - so two
priced sets locate the par rate exactly.
"""
import copy
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch

from derivus import utils
from derivus.bootstrappers import (BenchmarkInstruments, InterestRateCurveParameters, author_quote,
                                   damped_newton, quote_knots, quote_node)
from derivus.config import Config, ModelParams

from rates_world import BASE, deposit, fra, par_swap, ois_swap

DEVICE = torch.device('cpu')
INTERP = ModelParams()


def quote_point(descriptor, deal):
    """A `Points` entry: the instrument type NAMED, a block of it carried, and the number beside it.

    `Object` and `Discount_Rate` are dropped from the block on the way in. The point names the type
    in `DealType` and the family stamps the discount curve from the block it belongs to, so neither
    is authored twice - which is the whole reuse-by-reference rule made concrete.
    """
    return {'Use': 'Yes', 'Descriptor': descriptor, 'DealType': deal['Object'],
            'Quote_Type': 'Par_Rate', 'Quoted_Market_Value': 0.0,
            'Deal': {k: v for k, v in deal.items() if k not in ('Object', 'Discount_Rate')}}


# ---------------------------------------------------------------------------------------------
# A USD SOFR-style world. OIS accrues ACT/360 on business-day fixings compounded in arrears; the
# projection curve is a 3M index quoted as a FRA strip out to a year and par swaps beyond it. The
# zero curves are 2026-shaped: a 4.4% front end easing through a 3.9% belly back up to 4.0%, with
# the projection curve carrying a basis over the discount curve that tightens with maturity.
# ---------------------------------------------------------------------------------------------
USD_OIS_MONTHS = (3, 6, 12, 24, 36, 60, 84, 120)
USD_OIS_TRUE = [0.0448, 0.0442, 0.0430, 0.0412, 0.0402, 0.0396, 0.0397, 0.0400]
USD_PROJ_TRUE = [0.0470, 0.0463, 0.0455, 0.0450, 0.0430, 0.0419, 0.0412, 0.0412, 0.0415]

# ---------------------------------------------------------------------------------------------
# A ZAR JIBAR-style world. One curve, quoted ACT/365 off a 3M deposit and quarterly-resetting par
# swaps, with the humped shape a hiking-then-cutting curve has: 8.0% at the front, 9.5% at five
# years, back to 9.05% at ten.
# ---------------------------------------------------------------------------------------------
ZAR_SWAP_YEARS = (1, 2, 3, 5, 7, 10)
ZAR_TRUE = [0.0800, 0.0835, 0.0880, 0.0915, 0.0950, 0.0935, 0.0905]


def usd_blocks():
    """The two `Market Prices` blocks of the multi-curve world, keyed as the section keys them."""
    ois = [quote_point('USD {}M OIS'.format(m), ois_swap('OIS_{}M'.format(m), 'USD', 'USD-OIS', m, 0.0))
           for m in USD_OIS_MONTHS]
    projection = [
        quote_point('USD FRA {}x{}'.format(a, b),
                    fra('FRA_{}X{}'.format(a, b), 'USD', 'USD-3M', 'USD-OIS', a, b, 0.0))
        for a, b in ((0, 3), (3, 6), (6, 9), (9, 12))]
    projection += [quote_point('USD {}Y IRS'.format(y),
                               par_swap('IRS_{}Y'.format(y), 'USD', 'USD-3M', 'USD-OIS', y, 0.0))
                   for y in (2, 3, 5, 7, 10)]
    return {'InterestRatePrices.USD-OIS': {
                'Currency': 'USD', 'Day_Count': 'ACT_365', 'Discount_Rate': '', 'Points': ois},
            'InterestRatePrices.USD-3M': {
                'Currency': 'USD', 'Day_Count': 'ACT_365', 'Discount_Rate': 'USD-OIS',
                'Points': projection}}


def zar_blocks():
    points = [quote_point('ZAR 3M JIBAR',
                          deposit('DEPO_3M', 'ZAR', 'ZAR-JIBAR-3M', 3, 0.0, day_count='ACT_365'))]
    points += [quote_point('ZAR {}Y IRS'.format(y),
                           par_swap('IRS_{}Y'.format(y), 'ZAR', 'ZAR-JIBAR-3M', 'ZAR-JIBAR-3M', y,
                                    0.0, day_count='ACT_365'))
               for y in ZAR_SWAP_YEARS]
    return {'InterestRatePrices.ZAR-JIBAR-3M': {
        'Currency': 'ZAR', 'Day_Count': 'ACT_365', 'Discount_Rate': '', 'Points': points}}


WORLDS = {
    'usd': (usd_blocks, {'InterestRate.USD-OIS': USD_OIS_TRUE, 'InterestRate.USD-3M': USD_PROJ_TRUE},
            'USD', 'USD-OIS'),
    'zar': (zar_blocks, {'InterestRate.ZAR-JIBAR-3M': ZAR_TRUE}, 'ZAR', 'ZAR-JIBAR-3M'),
}


def curve_of(market_price):
    return utils.check_tuple_name(utils.Factor('InterestRate', utils.check_rate_name(market_price)[1:]))


def block_nodes(block, discount_rate, quote=None):
    """The block's benchmarks as deal-tree nodes: all authored at `quote` percent, or each at its
    own `Quoted_Market_Value` when `quote` is None."""
    nodes = []
    for point in block['Points']:
        deal = copy.deepcopy(dict(point['Deal'], Object=point['DealType']))
        author_quote(deal, point['Quoted_Market_Value'] if quote is None else quote, discount_rate)
        nodes.append(quote_node(deal, {}))
    return nodes


def discount_of(market_price, block):
    """A blank `Discount_Rate` discounts on the curve being built - the single-curve configuration."""
    return block['Discount_Rate'] or '.'.join(utils.check_rate_name(market_price)[1:])


def par_quotes(block, discount_rate, price_factors):
    """The rate, in percent, at which each benchmark is worth exactly zero on `price_factors`.

    PV is affine in the quote, so `PV(0) / (PV(0) - PV(1))` is the root and not an approximation of
    one - which matters, because a quote generated to anything less than machine precision would
    put a floor under what the round trip can recover. Bracketing at 0 and 1 PERCENT returns the
    root in percent, which is the unit every quote field on every one of these deals is read in.
    """
    priced = [BenchmarkInstruments(
        block_nodes(block, discount_rate, quote), price_factors, INTERP, BASE, block['Currency'],
        {}, [], DEVICE)({}).detach().numpy() for quote in (0.0, 1.0)]
    return priced[0] / (priced[0] - priced[1])


def authored_world(world):
    """`(market_prices, price_factors, true_curves)` - the quotes generated off a known curve.

    The knots come from the family's own rule, so the curve is authored on exactly the grid the
    bootstrap will solve on. Without that the round trip could not close to 1e-10 whatever the
    solver did, because there would be no curve on the solved grid that reprices the quotes.
    """
    blocks_fn, true_nodes, currency, spot_curve = WORLDS[world]
    blocks = blocks_fn()
    price_factors = {'FxRate.{}'.format(currency): {
        'Domestic_Currency': None, 'Interest_Rate': spot_curve, 'Priority': 1, 'Spot': 1.0}}

    true_curves = {}
    for market_price, block in blocks.items():
        discount_rate = discount_of(market_price, block)
        knots = quote_knots(block_nodes(block, discount_rate, 0.0), BASE, block['Day_Count'], {})
        assert (np.diff(knots) > 0).all(), 'the quotes must be authored in maturity order'
        true_curves[curve_of(market_price)] = knots
        price_factors[curve_of(market_price)] = {
            'Property_Aliases': None, 'Sub_Type': None, 'Currency': currency,
            'Day_Count': block['Day_Count'],
            'Curve': utils.Curve([], list(zip(knots, true_nodes[curve_of(market_price)])))}

    # in block order, which is dependency order here: the projection quotes discount on the OIS
    # curve, so that curve has to be authored before they can be priced
    for market_price, block in blocks.items():
        for point, quote in zip(block['Points'], par_quotes(
                block, discount_of(market_price, block), price_factors)):
            point['Quoted_Market_Value'] = quote

    market_prices = {name: {'instrument': block, 'Children': []} for name, block in blocks.items()}
    return market_prices, price_factors, true_curves


def bootstrapped(market_prices, currency, spot_curve, dtype=torch.float32):
    """Run the family the way `Config.bootstrap` runs it, into an empty `Price Factors`."""
    price_factors = {'FxRate.{}'.format(currency): {
        'Domestic_Currency': None, 'Interest_Rate': spot_curve, 'Priority': 1, 'Spot': 1.0}}
    InterestRateCurveParameters({}, DEVICE, dtype).bootstrap(
        {'Base_Date': BASE, 'Base_Currency': currency}, {}, price_factors, INTERP, market_prices,
        {})
    return price_factors


@pytest.mark.parametrize('world', sorted(WORLDS))
def test_the_bootstrap_recovers_the_curve_its_quotes_came_from(world):
    """The acceptance criterion: theta_true back to 1e-10 in float64."""
    _, _, currency, spot_curve = WORLDS[world]
    market_prices, true_factors, knots = authored_world(world)
    solved = bootstrapped(market_prices, currency, spot_curve)

    for curve_name in knots:
        expected = true_factors[curve_name]['Curve'].array
        recovered = solved[curve_name]['Curve'].array
        assert np.abs(recovered[:, 0] - expected[:, 0]).max() == 0.0, (
            '{}: the solve placed different knots'.format(curve_name))
        error = np.abs(recovered[:, 1] - expected[:, 1]).max()
        assert error < 1e-10, '{}: recovered to {:.3g}, not 1e-10\n{}\n{}'.format(
            curve_name, error, recovered[:, 1], expected[:, 1])


@pytest.mark.parametrize('world', sorted(WORLDS))
def test_a_perturbed_knot_fails_the_round_trip(world):
    """MUTATE the answer. One knot moved by a basis point has to break the comparison, or the gate
    above is comparing something to itself.

    The second half is what says the gate is not a placebo in the other direction: the SEED is the
    quotes themselves, and a solver that returned its seed unchanged would have to fail too. It
    does, by three or four basis points - a par rate is a good starting guess and nowhere near the
    zero rate at 1e-10.
    """
    _, _, currency, spot_curve = WORLDS[world]
    market_prices, true_factors, knots = authored_world(world)
    solved = bootstrapped(market_prices, currency, spot_curve)

    curve_name = sorted(knots)[0]
    solved[curve_name]['Curve'].array[0, 1] += 1e-4
    assert np.abs(solved[curve_name]['Curve'].array[:, 1] -
                  true_factors[curve_name]['Curve'].array[:, 1]).max() > 1e-10

    for market_price, entry in market_prices.items():
        seed = np.array([point['Quoted_Market_Value'] / 100.0
                         for point in entry['instrument']['Points']])
        assert np.abs(np.sort(seed) - true_factors[curve_of(market_price)]['Curve'].array[:, 1]
                      ).max() > 1e-5, 'the seed is already the answer - this world proves nothing'


def test_the_solve_is_float64_whatever_the_bootstrapper_was_built_with():
    """The precision seam. `construct_bootstrapper` defaults to float32 and a cube may run in it;
    the bootstrap and its Jacobian do not, because a residual carried in float32 cannot be driven
    to a 1e-10 curve."""
    market_prices, true_factors, knots = authored_world('zar')
    solved = bootstrapped(market_prices, 'ZAR', 'ZAR-JIBAR-3M', dtype=torch.float32)
    curve_name = 'InterestRate.ZAR-JIBAR-3M'
    assert solved[curve_name]['Curve'].array.dtype == np.float64
    assert np.abs(solved[curve_name]['Curve'].array[:, 1] -
                  true_factors[curve_name]['Curve'].array[:, 1]).max() < 1e-10


def test_the_blocks_are_solved_in_dependency_order():
    """A projection curve solved before the discount curve it prices against is solved against a
    curve that does not exist. Authoring the two blocks the wrong way round has to change nothing,
    because `Discount_Rate` says which is which and the family reads it."""
    market_prices, true_factors, knots = authored_world('usd')
    reversed_blocks = dict(reversed(list(market_prices.items())))
    assert list(reversed_blocks) == ['InterestRatePrices.USD-3M', 'InterestRatePrices.USD-OIS']

    solved = bootstrapped(reversed_blocks, 'USD', 'USD-OIS')
    for curve_name in knots:
        assert np.abs(solved[curve_name]['Curve'].array[:, 1] -
                      true_factors[curve_name]['Curve'].array[:, 1]).max() < 1e-10


def test_a_held_out_quote_leaves_the_solve():
    """`Use` is what lets a quote be dropped without being deleted. Dropping one has to drop its
    knot, because the knot grid IS the used quotes' maturities - a curve that kept the knot would
    be solving for an unknown no instrument identifies."""
    market_prices, _, _ = authored_world('zar')
    block = market_prices['InterestRatePrices.ZAR-JIBAR-3M']['instrument']
    full = bootstrapped(market_prices, 'ZAR', 'ZAR-JIBAR-3M')

    block['Points'][-1]['Use'] = 'No'
    held_out = bootstrapped(market_prices, 'ZAR', 'ZAR-JIBAR-3M')
    curve_name = 'InterestRate.ZAR-JIBAR-3M'
    assert len(held_out[curve_name]['Curve'].array) == len(full[curve_name]['Curve'].array) - 1
    # the quotes that stayed still reprice, so the shorter curve agrees on every knot it kept
    assert np.abs(held_out[curve_name]['Curve'].array[:, 1] -
                  full[curve_name]['Curve'].array[:-1, 1]).max() < 1e-10


def test_config_bootstrap_drives_the_family_and_finds_the_curve_it_wrote(caplog):
    """End to end through `Config.bootstrap`, which is how a job reaches this.

    It also gates the one loose end the design note recorded: the check for a bootstrapper that
    silently did nothing looks for a `<ClassName>.*` price factor, and this family writes an
    ordinary `InterestRate`. `price_factor_type` is what settles it - and the check only LOGS, so
    the log is what has to be asserted or dropping the declaration would go unnoticed.
    """
    market_prices, true_factors, knots = authored_world('usd')
    config = Config(base_currency='USD')
    config.params['System Parameters']['Base_Date'] = BASE
    config.params['Price Factors'] = {'FxRate.USD': {
        'Domestic_Currency': None, 'Interest_Rate': 'USD-OIS', 'Priority': 1, 'Spot': 1.0}}
    config.params['Market Prices'] = market_prices
    config.params['Bootstrapper Configuration'] = {'InterestRateCurveParameters': {}}
    with caplog.at_level(logging.ERROR):
        config.bootstrap()
    assert 'wrote no' not in caplog.text, caplog.text

    for curve_name in knots:
        assert curve_name in config.params['Price Factors'], (
            '{} was not written'.format(curve_name))
        assert np.abs(config.params['Price Factors'][curve_name]['Curve'].array[:, 1] -
                      true_factors[curve_name]['Curve'].array[:, 1]).max() < 1e-10


@pytest.mark.parametrize('knob,value', [('N_Iter', 1), ('Damping_Halvings', -1)])
def test_the_solver_knobs_are_read_off_the_block(knob, value):
    """The knobs are JSON, so a job tightens or loosens the solve with no code edit - and a declared
    field nothing honours is the defect this whole store exists to make unreachable.

    Each value is chosen to be unsatisfiable: one Newton iteration cannot converge a seven-knot
    curve, and `-1` halvings forbids the line search from trying even the full step. `-1` rather
    than `0` because the full step is what these worlds always take - see the damping finding below
    - so no non-negative value fails here, and pretending otherwise would be a gate that passes for
    the wrong reason.
    """
    market_prices, _, _ = authored_world('zar')
    market_prices['InterestRatePrices.ZAR-JIBAR-3M']['instrument'][knob] = value
    with pytest.raises(Exception, match='Curve bootstrap'):
        bootstrapped(market_prices, 'ZAR', 'ZAR-JIBAR-3M')


def test_a_tighter_tolerance_still_converges_to_the_same_curve():
    """The other direction: `Tol` is a floor on the step, not a target, so asking for less than the
    default cannot move the answer - it can only cost an iteration. A knob that changed the number
    would be a knob nobody could safely turn."""
    market_prices, true_factors, _ = authored_world('zar')
    loose = bootstrapped(market_prices, 'ZAR', 'ZAR-JIBAR-3M')
    market_prices['InterestRatePrices.ZAR-JIBAR-3M']['instrument']['Tol'] = 1e-16
    tight = bootstrapped(market_prices, 'ZAR', 'ZAR-JIBAR-3M')
    curve_name = 'InterestRate.ZAR-JIBAR-3M'
    assert np.abs(tight[curve_name]['Curve'].array[:, 1] -
                  loose[curve_name]['Curve'].array[:, 1]).max() < 1e-14
    assert np.abs(tight[curve_name]['Curve'].array[:, 1] -
                  true_factors[curve_name]['Curve'].array[:, 1]).max() < 1e-10


def test_the_damping_never_engages_on_these_worlds():
    """Honest negative result, recorded rather than dressed up as a gate.

    Newton from a par-rate seed takes the FULL step at every iteration on both worlds - the
    backtracking line search never halves once - so removing the damping entirely would not fail
    anything here. It is insurance against a first iterate these fixtures do not produce, and the
    fixture is too well-posed to exercise it. What IS gated is that the search runs and agrees with
    the undamped step, so a line search that silently rejected good steps would show up.
    """
    market_prices, _, _ = authored_world('zar')
    block = market_prices['InterestRatePrices.ZAR-JIBAR-3M']['instrument']
    nodes = block_nodes(block, 'ZAR-JIBAR-3M')

    price_factors = {
        'FxRate.ZAR': {'Domestic_Currency': None, 'Interest_Rate': 'ZAR-JIBAR-3M',
                       'Priority': 1, 'Spot': 1.0},
        'InterestRate.ZAR-JIBAR-3M': {
            'Property_Aliases': None, 'Sub_Type': None, 'Currency': 'ZAR', 'Day_Count': 'ACT_365',
            'Curve': utils.Curve([], list(zip(
                quote_knots(nodes, BASE, 'ACT_365', {}),
                [p['Quoted_Market_Value'] / 100.0 for p in block['Points']])))}}
    curve = utils.Factor('InterestRate', ('ZAR-JIBAR-3M',))
    benchmarks = BenchmarkInstruments(
        nodes, price_factors, INTERP, BASE, 'ZAR', {}, [curve], DEVICE)

    seed = torch.tensor(benchmarks.factors[curve].current_value(), dtype=torch.float64)
    damped = damped_newton(benchmarks, {curve: seed}, 50, 1e-14, 6)

    # the same iteration with the line search removed - full step every time
    x = seed.clone()
    for _ in range(20):
        x = x.detach().requires_grad_(True)
        f = benchmarks({curve: x})
        jacobian = torch.stack([torch.autograd.grad(f[i], x, retain_graph=True)[0]
                                for i in range(f.numel())])
        x = x.detach() - torch.linalg.solve(jacobian, f.detach())
    assert torch.allclose(damped[curve], x, atol=1e-14), (
        'the damped and undamped iterations disagree, so the line search is doing something')
