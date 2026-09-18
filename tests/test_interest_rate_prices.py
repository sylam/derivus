"""Does the curve family recover the curve its quotes were priced off?

A bootstrap has one honest gate, a round trip: construct a zero curve, PRICE the benchmark set off
it to GENERATE the quotes, then require the solve to recover the curve it started from. The levels
only have to be plausibly shaped, because the quotes are derived from them.

Two rate worlds, because the configurations fail differently. USD SOFR-style is genuinely
multi-curve: an OIS discount curve, then a projection curve solved from a FRA strip and par swaps
that discount on it, which only works in dependency order. ZAR JIBAR-style is the degenerate
single-curve one - the harder solve, because the unknown appears on both sides of every benchmark.

Generating the quotes needs no root find: a benchmark's PV is AFFINE in its quote, so two priced
sets locate the par rate exactly.

A third world is CROSS-CURRENCY, gated differently because its quotes are FX forward outrights and
there is no true curve to recover: a USD curve from its own quotes, a ZAR curve from USDZAR
outrights against it. What stands in for the round trip is the identity covered interest parity IS -
reprice a fresh par forward off the solved pair and the outright comes back - plus its subtleties: a
residual reading another currency's curve and spot as constants, an ordering dependency no
`Discount_Rate` declares, and a quote that cannot reach the `Quote_Sensitivity` overlay and says so.
"""
import copy
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
import torch

from derivus import riskfactors, utils
from derivus.bootstrappers import (BenchmarkInstruments,
                                   InterestRateCurveParameters,
                                   author_quote,
                                   quote_knots,
                                   quote_node)
from derivus.utils import damped_newton
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


# ---------------------------------------------------------------------------------------------
# A ZARONIA-style world: an overnight front, OIS swaps quoted MONTHLY out to the last policy
# meeting anyone has a view on, and ordinary annual swaps beyond. That is two quoting conventions
# on one curve, which is what `Near_Interpolation` is for - LinearRT through the monthly half,
# HermiteRT over the annual one. Its shape is a cutting cycle that stops: 7.05% easing to 6.85% by
# 18M, then steepening back to 7.65% at ten years.
# ---------------------------------------------------------------------------------------------
ZARONIA_MONTHS = (1, 2, 3, 6, 9, 12, 15, 18)
ZARONIA_YEARS = (2, 3, 5, 10)
ZARONIA_TRUE = [0.0702, 0.0700, 0.0697, 0.0694, 0.0692, 0.0690, 0.0687, 0.0686, 0.0685, 0.0685,
                0.0692, 0.0710, 0.0741, 0.0765]


def zaronia_blocks():
    """The monthly half is the near one, and `Near_Tenor` is where it stops - the last monthly
    benchmark, so the split lands ON a knot rather than inside a segment.

    The overnight front is why `Tol` defaults to 1e-13. That benchmark's PV is a difference of two
    numbers of the notional's size, so its residual floors at one ULP of 1e6 - 1.16e-10 - and the
    Newton step that floor implies is `1.16e-10 / (N tau)` with tau a day, which is 4e-14: under a
    1e-14 tolerance the line search ran out of halvings at iteration 4. A three-month front divides
    the same floor by ninety and never sees it.
    """
    points = [quote_point('ZAR ON', deposit('ON', 'ZAR', 'ZAR-ZARONIA', 0, 0.0,
                                            day_count='ACT_365', days=1))]
    points += [quote_point('ZAR {}M OIS'.format(m),
                           par_swap('OIS_{}M'.format(m), 'ZAR', 'ZAR-ZARONIA', 'ZAR-ZARONIA', 0,
                                    0.0, fixed_frequency=12, float_frequency=12,
                                    day_count='ACT_365', compounding='OIS', months=m))
               for m in ZARONIA_MONTHS]
    # THE ONE BENCHMARK THAT READS THE NEAR HALF BETWEEN ITS KNOTS. Every OIS row above pays
    # annually, so each reads the curve at its own maturity and at knots the ladder already
    # carries; a 4x7 FRA reads 4M, which no quote puts a knot at, and is therefore the only row
    # the near interpolation can move. Placed at its own maturity so the grid stays ascending.
    points.insert(5, quote_point('ZAR FRA 4x7', fra('FRA_4X7', 'ZAR', 'ZAR-ZARONIA', 'ZAR-ZARONIA',
                                                    4, 7, 0.0, day_count='ACT_365')))
    points += [quote_point('ZAR {}Y OIS'.format(y),
                           par_swap('OIS_{}Y'.format(y), 'ZAR', 'ZAR-ZARONIA', 'ZAR-ZARONIA', y,
                                    0.0, fixed_frequency=12, float_frequency=12,
                                    day_count='ACT_365', compounding='OIS'))
               for y in ZARONIA_YEARS]
    return {'InterestRatePrices.ZAR-ZARONIA': {
        'Currency': 'ZAR', 'Day_Count': 'ACT_365', 'Discount_Rate': '', 'Points': points,
        'Near_Interpolation': 'LinearRT',
        'Near_Tenor': pd.DateOffset(months=18)}}


WORLDS = {
    'usd': (usd_blocks, {'InterestRate.USD-OIS': USD_OIS_TRUE, 'InterestRate.USD-3M': USD_PROJ_TRUE},
            'USD', 'USD-OIS'),
    'zar': (zar_blocks, {'InterestRate.ZAR-JIBAR-3M': ZAR_TRUE}, 'ZAR', 'ZAR-JIBAR-3M'),
    'zaronia': (zaronia_blocks, {'InterestRate.ZAR-ZARONIA': ZARONIA_TRUE}, 'ZAR', 'ZAR-ZARONIA'),
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


def par_quotes(block, discount_rate, price_factors, interp=None):
    """The rate, in percent, at which each benchmark is worth exactly zero on `price_factors`.

    PV is affine in the quote, so `PV(0) / (PV(0) - PV(1))` is the root and not an approximation of
    one - which matters, because a quote generated to anything less than machine precision would
    put a floor under what the round trip can recover. Bracketing at 0 and 1 PERCENT returns the
    root in percent, which is the unit every quote field on every one of these deals is read in.
    """
    priced = [BenchmarkInstruments(
        block_nodes(block, discount_rate, quote), price_factors, interp or INTERP, BASE,
        block['Currency'], {}, [], DEVICE)({}).detach().numpy() for quote in (0.0, 1.0)]
    return priced[0] / (priced[0] - priced[1])


def authored_world(world, interp=None):
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
        price_factors[curve_of(market_price)] = dict({
            'Property_Aliases': None, 'Sub_Type': None, 'Currency': currency,
            'Day_Count': block['Day_Count'],
            'Curve': utils.Curve([], list(zip(knots, true_nodes[curve_of(market_price)])))},
            # the TRUE curve carries the split the block declares, or the quotes would be generated
            # under one interpolation and recovered under another
            **({'Near_Interpolation': block['Near_Interpolation'],
                'Near_Date': BASE + block['Near_Tenor']}
               if block.get('Near_Interpolation') else {}))

    # in block order, which is dependency order here: the projection quotes discount on the OIS
    # curve, so that curve has to be authored before they can be priced
    for market_price, block in blocks.items():
        for point, quote in zip(block['Points'], par_quotes(
                block, discount_of(market_price, block), price_factors, interp)):
            point['Quoted_Market_Value'] = quote

    market_prices = {name: {'instrument': block, 'Children': []} for name, block in blocks.items()}
    return market_prices, price_factors, true_curves


def bootstrapped(market_prices, currency, spot_curve, dtype=torch.float32, interp=None):
    """Run the family the way `Config.bootstrap` runs it, into an empty `Price Factors`."""
    price_factors = {'FxRate.{}'.format(currency): {
        'Domestic_Currency': None, 'Interest_Rate': spot_curve, 'Priority': 1, 'Spot': 1.0}}
    InterestRateCurveParameters({}, DEVICE, dtype).bootstrap(
        {'Base_Date': BASE, 'Base_Currency': currency}, {}, price_factors, interp or INTERP,
        market_prices, {})
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
    """MUTATE the answer: one knot moved by a basis point has to break the comparison, or the gate
    above compares something to itself. The second half is the other direction - the SEED is the
    quotes themselves, and a solver returning its seed unchanged fails too, by three or four basis
    points.
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


def test_the_solve_runs_on_the_host_whatever_device_the_job_runs_on():
    """A benchmark Jacobian is one small backward per quote per iteration, so the solve is
    dispatch-bound and a card is the slow place for it: three desk curves in 5 s on the host
    against 17 s on an RTX 3090. Killing mutation: take the family's own device back from the
    constructor's, which on a CUDA box puts the solve on the card."""
    family = InterestRateCurveParameters({}, torch.device('cuda'), torch.float64)
    assert family.device.type == 'cpu'


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


def test_a_benchmark_matured_at_the_base_date_refuses_by_name():
    """A quote whose last cashflow is ON the base date wants its knot at tenor zero, and the knot
    rule cannot make that square: the curve is flat below its shortest knot by `CurveTenor`'s
    clipping, so a knot there identifies nothing. What the solve makes of it is a singular matrix
    out of `damped_newton` naming nothing, so the family refuses first - the block, the benchmark,
    its last cashflow, the base date and the two remedies.

    The second arm is one of those remedies taken: the same block with that quote's `Use` off
    solves and recovers its own curve, which is what says the refusal reads the TENOR and not the
    extra quote.
    """
    market_prices, true_factors, _ = authored_world('zar')
    block = market_prices['InterestRatePrices.ZAR-JIBAR-3M']['instrument']
    # a live one-month deposit that pays TODAY - one real cashflow, on the base date itself
    started = BASE - pd.DateOffset(months=1)
    block['Points'].insert(0, quote_point('ZAR O/N', dict(
        deposit('DEPO_ON', 'ZAR', 'ZAR-JIBAR-3M', 1, 7.5, day_count='ACT_365'),
        Effective_Date=started, Maturity_Date=BASE,
        Interest_Rate_Schedule=utils.DateList({started: 7.5}))))

    with pytest.raises(ValueError) as refusal:
        bootstrapped(market_prices, 'ZAR', 'ZAR-JIBAR-3M')
    message = str(refusal.value)
    assert 'ZAR O/N (last cashflow {:%Y-%m-%d})'.format(BASE) in message, message
    assert 'base date {:%Y-%m-%d}'.format(BASE) in message, message
    assert 'tenor zero' in message and 'Use No' in message, message

    block['Points'][0]['Use'] = 'No'
    curve_name = 'InterestRate.ZAR-JIBAR-3M'
    solved = bootstrapped(market_prices, 'ZAR', 'ZAR-JIBAR-3M')[curve_name]['Curve'].array
    assert (solved[:, 0] > 0.0).all(), solved
    assert np.abs(solved[:, 1] - true_factors[curve_name]['Curve'].array[:, 1]).max() < 1e-10


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
    """The knobs are JSON, so a job tightens or loosens the solve with no code edit. Each value here
    is unsatisfiable: one Newton iteration cannot converge a seven-knot curve, and `-1` halvings
    forbids even the full step - `-1` rather than `0` because the full step is what these worlds
    always take, so no non-negative value fails here.
    """
    market_prices, _, _ = authored_world('zar')
    market_prices['InterestRatePrices.ZAR-JIBAR-3M']['instrument'][knob] = value
    with pytest.raises(Exception, match='Curve bootstrap'):
        bootstrapped(market_prices, 'ZAR', 'ZAR-JIBAR-3M')


def test_the_near_split_is_written_through_and_every_quote_still_reprices():
    """THE BLOCK SAYS HOW THE CURVE IT DEFINES IS INTERPOLATED AT THE FRONT, and the seed writes
    that onto the factor: `Near_Interpolation` as declared, `Near_Date` as base date plus
    `Near_Tenor`. A ZARONIA curve is quoted monthly to the last policy meeting anyone has a view on
    and annually beyond, which is two conventions on one curve - LinearRT through the near half,
    the job's own scheme over the far one.

    THE SOLVE READS BOTH SEGMENTS, which is what makes this more than a stored string:
    `BenchmarkInstruments` constructs its factor with the base date, so the residual is priced off
    the stacked interpolation and every benchmark comes back at par. Drop the two keys from `seed`
    and the solve prices under one scheme where the quotes were generated under two: the round trip
    misses by **1.907e-05** with a Linear far leg and 1.879e-05 with a HermiteRT one, against its
    1e-10 bound.

    THE PAR CHECK CANNOT SEE IT AND THE ROUND TRIP CAN. A solve drives its own benchmarks to par
    under whatever scheme it is using, so the residual reads 1.164e-10 either way; only a curve
    AUTHORED under the split separates them. And it separates them only where a benchmark reads the
    near half BETWEEN its knots - which annual-paying OIS rows never do, every coupon of theirs
    landing on a knot the ladder already carries. The 4x7 FRA is in the world for that one reason,
    and without it this gate's own mutation survives.

    A block declaring NEITHER writes exactly the five keys it always wrote.
    """
    interp = ModelParams()
    interp.append('InterestRate', (), 'HermiteRT')
    market_prices, true_factors, _ = authored_world('zaronia', interp)
    solved = bootstrapped(market_prices, 'ZAR', 'ZAR-ZARONIA', interp=interp)
    factor = solved['InterestRate.ZAR-ZARONIA']
    assert factor['Near_Interpolation'] == 'LinearRT'
    assert factor['Near_Date'] == BASE + pd.DateOffset(months=18)

    # the factor the engine builds off it carries TWO segments, split on the 18M knot itself
    built = riskfactors.construct_factor(
        utils.Factor('InterestRate', ('ZAR-ZARONIA',)), solved, interp, base_date=BASE)
    assert [segment[2][0] for segment in built.interpolation] == ['LinearRT', 'HermiteRT']
    split = built.interpolation[0][1]
    assert built.tenors[split] == pytest.approx(
        ((BASE + pd.DateOffset(months=18)) - BASE).days / 365.0)
    # the overnight front, every monthly benchmark and the FRA between them
    assert split == len(ZARONIA_MONTHS) + 1

    # every benchmark reprices at par off the SOLVED curve, read through both segments
    block = market_prices['InterestRatePrices.ZAR-ZARONIA']['instrument']
    priced = BenchmarkInstruments(block_nodes(block, 'ZAR-ZARONIA'), solved, interp, BASE, 'ZAR',
                                  {}, [], DEVICE)({}).detach().numpy()
    assert np.abs(priced).max() < 1e-9, priced

    plain = bootstrapped(*authored_world('zar')[:1], 'ZAR', 'ZAR-JIBAR-3M')
    assert set(plain['InterestRate.ZAR-JIBAR-3M']) == {
        'Property_Aliases', 'Sub_Type', 'Currency', 'Day_Count', 'Curve'}


def test_a_near_interpolation_without_its_tenor_refuses_by_name():
    """The split needs the date it stops at: a block naming the scheme and no tenor is refused
    before a quote is read, rather than dying on a date plus an empty string."""
    market_prices, _, _ = authored_world('zaronia')
    market_prices['InterestRatePrices.ZAR-ZARONIA']['instrument']['Near_Tenor'] = ''
    with pytest.raises(Exception, match='Near_Tenor'):
        bootstrapped(market_prices, 'ZAR', 'ZAR-ZARONIA')


def test_a_term_ois_benchmark_prices_the_fixing_list_it_replaces():
    """WHY THE OIS BENCHMARK IS A TERM SWAP. At t0 the compounded overnight forwards read off a
    curve telescope to the period forward, so a coupon carrying ONE reset over its own accrual
    prices what a list of daily fixings prices - on a flat curve and on a sloped one alike.

    MEASURED, at a 4% quote on a million of notional. On a flat 4% curve the 2Y reads
    483.511030079 both ways off 523 items against one deal, the 10Y 2068.437863819 off 2612; on the
    world's own sloped curve the 5Y reads -355.506266991 both ways and the 10Y 2012.027439223. The
    largest disagreement over the eight readings is 6.4e-10, which is the float64 noise of summing
    2612 items rather than a difference.

    The shape that does NOT work is the one in between: a list authored one item per COUPON, each
    carrying every fixing's reset, which is what the engine's own leg generation produces and which
    `pv_float_cashflow_list` averages at 1/n.
    """
    _, price_factors, _ = authored_world('usd')
    flat = dict(price_factors)
    flat['InterestRate.USD-OIS'] = dict(
        price_factors['InterestRate.USD-OIS'],
        Curve=utils.Curve([], [(t, 0.04) for t in
                               price_factors['InterestRate.USD-OIS']['Curve'].array[:, 0]]))

    for label, factors in (('flat', flat), ('sloped', price_factors)):
        for months in (12, 24, 60, 120):
            listed = quote_node(ois_swap('LIST', 'USD', 'USD-OIS', months, 4.0), {})
            term = quote_node(par_swap('TERM', 'USD', 'USD-OIS', 'USD-OIS', 0, 4.0,
                                       fixed_frequency=12, float_frequency=12, months=months,
                                       compounding='OIS'), {})
            pvs = [float(BenchmarkInstruments([node], factors, INTERP, BASE, 'USD', {}, [],
                                              DEVICE)({}).detach().numpy()[0])
                   for node in (listed, term)]
            assert pvs[0] == pytest.approx(pvs[1], abs=1e-8), (label, months, pvs)


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
    """Honest negative result. Newton from a par-rate seed takes the FULL step at every iteration on
    both worlds, so removing the damping entirely would fail nothing here: it is insurance against a
    first iterate these fixtures do not produce. What IS gated is that the search runs and agrees
    with the undamped step, so a line search rejecting good steps would show up.
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


# ---------------------------------------------------------------------------------------------
# A CROSS-CURRENCY world. The USD curve is solved from its own deposit and swap quotes; the ZAR
# curve is solved DIRECTLY from FX FORWARD OUTRIGHTS against it, which is covered interest parity
# run backwards - the spot, the outright and the USD leg are given and the ZAR discount factor is
# the unknown. Nothing states CIP anywhere: the residual is an `FXForwardDeal` priced by the
# engine's own pricer and held at zero, so the parity relation is whatever that pricer means.
#
# Unlike the rate worlds these quotes are STATED rather than generated, because there is no true
# ZAR curve to recover - the outrights ARE the market data. What replaces the round trip is the
# identity CIP is: price a fresh par forward off the solved pair and the outright has to come back.
# The levels are invented and 2026-shaped - an 18.25 spot carrying about 4.8% of forward points.
# ---------------------------------------------------------------------------------------------
FX_SPOT = 18.25
FX_OUTRIGHTS = ((1, 18.32), (3, 18.47), (6, 18.70), (12, 19.15))
FX_USD_SWAP_YEARS = (1, 2, 3)
FX_USD_TRUE = [0.0448, 0.0430, 0.0412, 0.0402]
USD_CURVE, ZAR_CURVE = 'USD-OIS', 'ZAR-FX-IMPLIED'


def fx_forward(ref, months, outright, sell_amount=1e6):
    """One benchmark FX forward: `sell_amount` of USD sold against ZAR at `months`, bought at
    `outright` ZAR per USD.

    `Sell_Amount` and BOTH discount-rate names are fixed by the authoring, leaving the quote exactly
    one place to land - `Buy_Amount` - and letting the deal name its own two curves. The block's
    `Discount_Rate` is inert for this type, which is why the ordering read cannot stop at it.
    """
    return {'Object': 'FXForwardDeal', 'Reference': ref,
            'Sell_Currency': 'USD', 'Sell_Amount': sell_amount,
            'Buy_Currency': 'ZAR', 'Buy_Amount': outright * sell_amount,
            'Settlement_Date': BASE + pd.DateOffset(months=months),
            'Buy_Discount_Rate': ZAR_CURVE, 'Sell_Discount_Rate': USD_CURVE}


def fx_blocks():
    """The two `Market Prices` blocks of the cross-currency world, USD authored first."""
    usd = [quote_point('USD 3M depo', deposit('USD_DEPO_3M', 'USD', USD_CURVE, 3, 0.0))]
    usd += [quote_point('USD {}Y IRS'.format(y),
                        par_swap('USD_IRS_{}Y'.format(y), 'USD', USD_CURVE, USD_CURVE, y, 0.0))
            for y in FX_USD_SWAP_YEARS]
    forwards = [dict(quote_point('USDZAR {}M outright'.format(months),
                                 fx_forward('FWD_{}M'.format(months), months, outright)),
                     Quoted_Market_Value=outright)
                for months, outright in FX_OUTRIGHTS]
    return {'InterestRatePrices.' + USD_CURVE: {
                'Currency': 'USD', 'Day_Count': 'ACT_365', 'Discount_Rate': '', 'Points': usd},
            'InterestRatePrices.' + ZAR_CURVE: {
                'Currency': 'ZAR', 'Day_Count': 'ACT_365', 'Discount_Rate': '', 'Points': forwards}}


def fx_price_factors():
    """The two FX spots and nothing else. ZAR is the base currency of this world, so `FxRate.USD`
    IS the USDZAR spot - the constant the forward's other leg converts through, and the one
    `BenchmarkInstruments` hands the residual as a detached leaf."""
    return {
        'FxRate.ZAR': {'Domestic_Currency': None, 'Interest_Rate': ZAR_CURVE, 'Priority': 1,
                       'Spot': 1.0},
        'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': USD_CURVE, 'Priority': 1,
                       'Spot': FX_SPOT}}


def fx_world():
    """`market_prices` for the cross-currency world - the USD quotes generated at par off a known
    curve the way the rate worlds do, the ZAR forwards carrying their stated outrights."""
    blocks = fx_blocks()
    usd_block = blocks['InterestRatePrices.' + USD_CURVE]
    price_factors = fx_price_factors()
    price_factors['InterestRate.' + USD_CURVE] = {
        'Property_Aliases': None, 'Sub_Type': None, 'Currency': 'USD', 'Day_Count': 'ACT_365',
        'Curve': utils.Curve([], list(zip(
            quote_knots(block_nodes(usd_block, USD_CURVE, 0.0), BASE, 'ACT_365', {}),
            FX_USD_TRUE)))}
    for point, quote in zip(usd_block['Points'], par_quotes(usd_block, USD_CURVE, price_factors)):
        point['Quoted_Market_Value'] = quote
    return {name: {'instrument': block, 'Children': []} for name, block in blocks.items()}


def fx_bootstrapped(market_prices):
    """Run the family over the cross-currency world into a `Price Factors` holding the two spots
    and no curve at all, so every curve it comes back with was solved here."""
    price_factors = fx_price_factors()
    InterestRateCurveParameters({}, DEVICE, torch.float32).bootstrap(
        {'Base_Date': BASE, 'Base_Currency': 'ZAR'}, {}, price_factors, INTERP, market_prices, {})
    return price_factors


def fx_par_outrights(market_prices, price_factors):
    """Each forward benchmark's PAR outright off `price_factors` - the outright at which a FRESH
    `FXForwardDeal` is worth exactly zero, priced by the same pricer the solve used.

    `par_quotes` is the rate worlds' own affine root and it needs no adjusting to read an amount:
    PV is affine in the quote whatever the quote MEANS, so bracketing at outrights 0 and 1 returns
    the par outright exactly rather than approximately.
    """
    block = market_prices['InterestRatePrices.' + ZAR_CURVE]['instrument']
    return par_quotes(block, ZAR_CURVE, price_factors)


def test_the_outright_reaches_the_deal_unscaled():
    """The quote is the forward outright in `Buy_Currency` per unit of `Sell_Currency`, landing as
    `Buy_Amount = quote * Sell_Amount` with no conversion - which says the family scales nothing
    centrally: a percent-quoted benchmark is scaled by its own field semantics, so an amount-valued
    quote on the same `author_quote` path arrives untouched.
    """
    deal = fx_forward('FWD_6M', 6, 0.0)
    author_quote(deal, 18.70, ZAR_CURVE)
    assert deal['Buy_Amount'] == 18.70 * deal['Sell_Amount']
    assert deal['Sell_Amount'] == 1e6, 'the authored benchmark fixes the sold amount'


def test_a_forward_curve_reprices_the_outrights_it_was_solved_from():
    """THE IDENTITY. Solve a ZAR curve from USDZAR outrights, then price a fresh par forward off the
    solved pair: the par outright is the quote back, to 1e-9 relative.

    Covered interest parity CLOSING THROUGH THE ENGINE'S OWN PRICERS - no formula for the forward is
    written here or in the family. The residual is `FXForwardDeal.generate` held at zero, so
    whatever parity that pricer means is the parity the curve carries.
    """
    market_prices = fx_world()
    solved = fx_bootstrapped(market_prices)
    quoted = np.array([outright for _, outright in FX_OUTRIGHTS])
    par = fx_par_outrights(market_prices, solved)
    assert np.abs(par / quoted - 1.0).max() < 1e-9, (
        'the solved curve does not reprice its own outrights\n{}\n{}'.format(par, quoted))


def test_the_forward_knots_land_on_the_settlement_dates():
    """The knot rule, on this family's newest benchmark: one knot per used quote, at that
    benchmark's last cashflow - which for a forward is its settlement date, in the curve's own day
    count. Strictly increasing, or two forwards would share a knot and leave the curve between
    them unidentified."""
    market_prices = fx_world()
    solved = fx_bootstrapped(market_prices)
    knots = solved['InterestRate.' + ZAR_CURVE]['Curve'].array[:, 0]
    settlement = np.array([utils.DayCount.accrual(
        BASE, ((BASE + pd.DateOffset(months=months)) - BASE).days,
        utils.DayCount.code('ACT_365')) for months, _ in FX_OUTRIGHTS])
    assert (np.diff(knots) > 0).all(), 'the forward knots are not increasing: {}'.format(knots)
    assert np.abs(knots - settlement).max() == 0.0, '{} against {}'.format(knots, settlement)


def test_a_forwards_pv_is_affine_in_its_outright_to_the_last_bit():
    """The property the par solve and the drift metric both rest on, MEASURED rather than argued:
    the PV is affine in the outright at fixed curves, so the second difference across three levels
    carries NO curvature - what `CalibrationArtifact.mispricing` needs to be an exact quote-space
    residual.

    NOT the flat zero `_carry_quotes` reports for a rate quote: it is 3.7252903e-09 identically on
    all four benchmarks, which is one ULP of the BOUGHT LEG rather than curvature. A swap is
    bracketed where its own PV is near zero and cancels bit for bit; a forward's PV is the
    difference of two legs each worth about 2e7, so `np.spacing(20.0 * 1e6)` is its arithmetic
    floor - the machine's own resolution at the scale the sum is formed.

    The FIRST difference is about 9.5e5, so a quadratic term would have to hide fourteen orders of
    magnitude under the linear one to pass here.
    """
    market_prices = fx_world()
    solved = fx_bootstrapped(market_prices)
    block = market_prices['InterestRatePrices.' + ZAR_CURVE]['instrument']
    outrights = (18.0, 19.0, 20.0)
    priced = [BenchmarkInstruments(
        block_nodes(block, ZAR_CURVE, outright), solved, INTERP, BASE, 'ZAR', {}, [], DEVICE
    )({}).detach().numpy() for outright in outrights]
    second = priced[0] - 2.0 * priced[1] + priced[2]
    # the scale the PV's terms are formed at, which is what sets its rounding floor
    bought = max(outrights) * block['Points'][0]['Deal']['Sell_Amount']
    assert (np.abs(second) <= np.spacing(bought)).all(), (
        'PV carries curvature in the outright: {} against one ULP of {}'.format(second, bought))
    first = np.abs(priced[2] - priced[1])
    assert (first > 1e5).all() and (np.abs(second) / first < 1e-14).all(), (
        'the second difference is not at the rounding floor: {}'.format(second / first))


def test_a_held_out_forward_drops_its_knot():
    """`Use` on a forward is `Use` on any other benchmark: the knot grid IS the used quotes'
    settlement dates, so dropping the 1Y outright shortens the curve by one knot and moves none of
    the others - each forward identifies its own discount factor and nothing beyond it."""
    market_prices = fx_world()
    full = fx_bootstrapped(market_prices)
    market_prices['InterestRatePrices.' + ZAR_CURVE]['instrument']['Points'][-1]['Use'] = 'No'
    held_out = fx_bootstrapped(market_prices)

    curve_name = 'InterestRate.' + ZAR_CURVE
    assert len(held_out[curve_name]['Curve'].array) == len(full[curve_name]['Curve'].array) - 1
    assert np.abs(held_out[curve_name]['Curve'].array[:, 1] -
                  full[curve_name]['Curve'].array[:-1, 1]).max() < 1e-10
    par = fx_par_outrights(market_prices, held_out)
    assert np.abs(par[:-1] / np.array([o for _, o in FX_OUTRIGHTS])[:-1] - 1.0).max() < 1e-9


def test_a_forward_block_authored_before_the_curve_its_other_leg_needs_still_solves():
    """THE ORDERING SUBTLETY. This block's `Discount_Rate` is BLANK, yet its residual reads the USD
    curve because each forward names `USD-OIS` in its own `Sell_Discount_Rate` - so a dependency
    read stopping at the block's field orders the ZAR solve first, against a curve that does not
    exist. `benchmark_curves` includes the curves the deals name, so authoring ZAR FIRST changes
    nothing.
    """
    market_prices = fx_world()
    zar_name, usd_name = 'InterestRatePrices.' + ZAR_CURVE, 'InterestRatePrices.' + USD_CURVE
    assert market_prices[zar_name]['instrument']['Discount_Rate'] == '', (
        'the gate is void unless the block declares nothing - the deals have to be what orders it')

    reversed_blocks = dict(reversed(list(market_prices.items())))
    assert list(reversed_blocks) == [zar_name, usd_name]
    family = InterestRateCurveParameters({}, DEVICE, torch.float32)
    assert [name for name, _ in family.in_dependency_order(reversed_blocks)] == [usd_name, zar_name]
    assert family.benchmark_curves(market_prices[zar_name]['instrument']) == {USD_CURVE, ZAR_CURVE}

    solved = fx_bootstrapped(reversed_blocks)
    quoted = np.array([outright for _, outright in FX_OUTRIGHTS])
    assert np.abs(fx_par_outrights(reversed_blocks, solved) / quoted - 1.0).max() < 1e-9


def test_the_ordering_read_does_not_reorder_the_rate_worlds():
    """The extension is only safe if it is a NO-OP where the declaration was already enough. Both
    rate worlds order exactly as they did off `Discount_Rate` alone - a benchmark naming the curve
    its own block builds is the self-discounting case and orders nothing."""
    family = InterestRateCurveParameters({}, DEVICE, torch.float32)
    usd, _, _ = authored_world('usd')
    assert [name for name, _ in family.in_dependency_order(dict(reversed(list(usd.items()))))] == [
        'InterestRatePrices.USD-OIS', 'InterestRatePrices.USD-3M']
    zar, _, _ = authored_world('zar')
    assert family.benchmark_curves(zar['InterestRatePrices.ZAR-JIBAR-3M']['instrument']) == {
        'ZAR-JIBAR-3M'}
    assert [name for name, _ in family.in_dependency_order(zar)] == [
        'InterestRatePrices.ZAR-JIBAR-3M']


def test_a_forward_block_refuses_quote_sensitivity():
    """THE REFUSAL, and the reason it is not a zero. `Quote_Sensitivity` reports dV/dq by overlaying
    the CASHFLOW SCHEDULE COLUMNS a quote writes; an outright writes into `Buy_Amount`, read as a
    float off the deal, so no column moves and the overlay carries nothing. A zero delta on the
    instrument a desk actually trades is the failure this switch exists to prevent, so the block
    refuses by name and says which benchmark and which type could not be carried.
    """
    market_prices = fx_world()
    market_prices['InterestRatePrices.' + ZAR_CURVE]['instrument']['Quote_Sensitivity'] = 'Yes'
    with pytest.raises(Exception, match='Quote_Sensitivity') as refusal:
        fx_bootstrapped(market_prices)
    assert 'FXForwardDeal' in str(refusal.value), str(refusal.value)
    assert 'FWD_1M' in str(refusal.value), str(refusal.value)


def test_the_refusal_is_measured_and_leaves_the_rate_quotes_carrying():
    """The other half of that refusal: it is MEASURED, not a branch on the deal type, so it has to
    stay silent everywhere a quote does reach a schedule column. A rate world asking for
    `Quote_Sensitivity` still solves, still to 1e-10, and still leaves its quote leaf behind."""
    market_prices, true_factors, _ = authored_world('zar')
    block = market_prices['InterestRatePrices.ZAR-JIBAR-3M']['instrument']
    block['Quote_Sensitivity'] = 'Yes'
    family = InterestRateCurveParameters({}, DEVICE, torch.float32)
    price_factors = {'FxRate.ZAR': {'Domestic_Currency': None, 'Interest_Rate': 'ZAR-JIBAR-3M',
                                    'Priority': 1, 'Spot': 1.0}}
    family.bootstrap({'Base_Date': BASE, 'Base_Currency': 'ZAR'}, {}, price_factors, INTERP,
                     market_prices, {})
    curve_name = 'InterestRate.ZAR-JIBAR-3M'
    assert np.abs(price_factors[curve_name]['Curve'].array[:, 1] -
                  true_factors[curve_name]['Curve'].array[:, 1]).max() < 1e-10
    descriptors, quotes = family.quote_leaves['InterestRatePrices.ZAR-JIBAR-3M']
    assert len(descriptors) == len(block['Points']) and quotes.requires_grad


def test_a_forward_block_cannot_publish_a_ride_operator():
    """`Quote_Propagation` wants the same quote side `Quote_Sensitivity` does, and a forward block
    cannot give it either - but which refusal fires depends on what else is in the run.

    Solved TOGETHER the two blocks are one coupled set, MEASURED rather than declared:
    `BenchmarkInstruments.reads` finds the ZAR residual reaching `InterestRate.USD-OIS` through the
    sell leg's constant. A set spanning two reporting currencies cannot be compiled as one system,
    which the family refuses by name.

    Solved ALONE the set is one currency and that refusal has nothing to say; the overlay's does -
    no schedule column moves, so it refuses rather than publishing a `dF/dq` row of zeros.
    """
    market_prices = fx_world()
    for entry in market_prices.values():
        entry['instrument']['Quote_Propagation'] = 'Linear'
    with pytest.raises(Exception, match='Quote_Propagation') as coupled:
        fx_bootstrapped(market_prices)
    assert 'InterestRatePrices.' + USD_CURVE in str(coupled.value), str(coupled.value)
    assert 'InterestRatePrices.' + ZAR_CURVE in str(coupled.value), str(coupled.value)

    price_factors = fx_price_factors()
    price_factors['InterestRate.' + USD_CURVE] = fx_bootstrapped(fx_world())[
        'InterestRate.' + USD_CURVE]
    solo = {'InterestRatePrices.' + ZAR_CURVE: market_prices['InterestRatePrices.' + ZAR_CURVE]}
    with pytest.raises(Exception, match='Quote_Sensitivity') as alone:
        InterestRateCurveParameters({}, DEVICE, torch.float32).bootstrap(
            {'Base_Date': BASE, 'Base_Currency': 'ZAR'}, {}, price_factors, INTERP, solo, {})
    assert 'FXForwardDeal' in str(alone.value), str(alone.value)


def test_a_swap_rolls_the_coupons_it_generates_on_its_payment_calendar():
    """A benchmark's two dates are rolled by whoever authored it; the coupons between them are the
    engine's own schedule, and the leg's payment calendar is what rolls those, Modified Following.
    A deal naming no calendar keeps the tenor arithmetic's dates, which is every book without a
    calendar file. Killing mutation: the calendar not handed to the generator."""
    from derivus import instruments
    deal = par_swap('SWP', 'ZAR', 'ZAR-ZARONIA', 'ZAR-ZARONIA', 2, 7.0, fixed_frequency=6,
                    float_frequency=6)
    coupon = BASE + pd.DateOffset(months=12)
    business = pd.offsets.CustomBusinessDay(holidays=[coupon])
    calendars = {'JHB': {'businessday': business}}
    plain = instruments.SwapInterestDeal(deal, {})
    plain.reset(calendars)
    rolled = instruments.SwapInterestDeal(
        dict(deal, Pay_Payment_Calendars='JHB', Receive_Payment_Calendars='JHB'), {})
    rolled.reset(calendars)

    assert coupon in plain.paydates and coupon in plain.recdates
    assert coupon not in rolled.paydates and coupon not in rolled.recdates
    moved = utils.adjust_date(business, True, coupon)
    assert moved in rolled.paydates and moved in rolled.recdates
    assert len(rolled.paydates) == len(plain.paydates)
