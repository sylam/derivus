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
import json
import logging
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
import torch

import test_declared_defaults as book
import trial_rates
import derivus
from derivus import bootstrappers, utils
from derivus.bootstrappers import author_quote
from derivus.instruments import deal_node
from derivus.schema import completed
from derivus.config import Config, CustomJsonEncoder, ModelParams

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
    own `Quoted_Market_Value` when `quote` is None.

    `completed` first, as `quote_nodes` does it: a quote WRITER reads the block's own conventions
    and a benchmark states only what differs from its declaration."""
    nodes = []
    for point in block['Points']:
        deal = completed(copy.deepcopy(dict(point['Deal'], Object=point['DealType'])))
        author_quote(deal, point['Quoted_Market_Value'] if quote is None else quote, discount_rate)
        nodes.append(deal_node(deal, {}))
    return nodes


def discount_of(market_price, block):
    """A blank `Discount_Rate` discounts on the curve being built - the single-curve configuration."""
    return block['Discount_Rate'] or '.'.join(utils.check_rate_name(market_price)[1:])


def par_quotes(block, discount_rate, price_factors, interp=None):
    """The rate, in percent, at which each benchmark is worth exactly zero on `price_factors` - the
    library's own inverse map, exact rather than searched, which matters because a quote generated
    to anything less than machine precision would put a floor under what the round trip recovers.
    """
    return bootstrappers.par_quotes(block['Points'], discount_rate, block['Currency'],
                                    price_factors, interp or INTERP, BASE, {})


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
        knots = bootstrappers.InterestRate.quote_knots(
            block_nodes(block, discount_rate, 0.0), BASE, block['Day_Count'], {})
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


def bootstrapped(market_prices, currency, spot_curve, interp=None, config=None):
    """`Price Factors` after `Config.bootstrap` over `market_prices`, from one holding the base
    currency's spot alone - the way a job reaches the family."""
    config = config or Config(base_currency=currency)
    config.params['System Parameters']['Base_Date'] = BASE
    config.params['Price Factors'] = {'FxRate.{}'.format(currency): {
        'Domestic_Currency': None, 'Interest_Rate': spot_curve, 'Priority': 1, 'Spot': 1.0}}
    config.params['Price Factor Interpolation'] = interp or INTERP
    config.params['Market Prices'] = market_prices
    config.params['Bootstrapper Configuration'] = {'InterestRate': {}}
    config.bootstrap()
    return config.params['Price Factors']


@pytest.mark.parametrize('world', sorted(WORLDS))
def test_the_bootstrap_recovers_the_curve_its_quotes_came_from(world, caplog):
    """The acceptance criterion: theta_true back to 1e-10, on the knots the quotes name, through
    `Config.bootstrap`. Float64 whatever the bootstrapper was built with - `construct_bootstrapper`
    defaults to float32, and a residual carried in float32 cannot be driven to a 1e-10 curve. The
    SEED is the quotes themselves, three or four basis points off the answer, so a solver returning
    it fails. And the family writes an ordinary `InterestRate`, the type it is named for, or the
    check for a bootstrapper that silently did nothing logs one.

    Killing mutation: the solver returning its seed.
    """
    _, _, currency, spot_curve = WORLDS[world]
    market_prices, true_factors, knots = authored_world(world)
    for market_price, entry in market_prices.items():
        seed = np.array([point['Quoted_Market_Value'] / 100.0
                         for point in entry['instrument']['Points']])
        assert np.abs(np.sort(seed) - true_factors[curve_of(market_price)]['Curve'].array[:, 1]
                      ).max() > 1e-5, 'the seed is already the answer - this world proves nothing'
    with caplog.at_level(logging.ERROR):
        solved = bootstrapped(market_prices, currency, spot_curve)
    assert 'wrote no' not in caplog.text, caplog.text

    for curve_name in knots:
        expected = true_factors[curve_name]['Curve'].array
        recovered = solved[curve_name]['Curve'].array
        assert recovered.dtype == np.float64
        assert np.abs(recovered[:, 0] - expected[:, 0]).max() == 0.0, (
            '{}: the solve placed different knots'.format(curve_name))
        error = np.abs(recovered[:, 1] - expected[:, 1]).max()
        assert error < 1e-10, '{}: recovered to {:.3g}, not 1e-10\n{}\n{}'.format(
            curve_name, error, recovered[:, 1], expected[:, 1])


def keyed_job(families):
    """A job over the ZAR world's curve and the USD.ZAR smile, `families` keying the curve and the
    FX surface families' `Bootstrapper Configuration` entries."""
    from test_service import fx_vol_quotes

    market_prices, _, _ = authored_world('zar')
    return json.dumps({'Calc': {
        'Calculation': {'Object': 'BaseValuation', 'Base_Date': BASE, 'Currency': 'ZAR'},
        'Deals': {'Reference': 'keyed', 'Deals': {'Children': []}},
        'MergeMarketData': {'MarketDataFile': '', 'ExplicitMarketData': {
            'System Parameters': {'Base_Currency': 'ZAR', 'Base_Date': BASE},
            'Price Factors': {'FxRate.ZAR': {'Domestic_Currency': None, 'Priority': 1,
                                             'Interest_Rate': 'ZAR-JIBAR-3M', 'Spot': 1.0}},
            'Market Prices': dict(market_prices, **fx_vol_quotes()),
            'Bootstrapper Configuration': {family: {} for family in families}}}}},
        cls=CustomJsonEncoder)


def test_a_book_keying_a_family_by_its_retired_name_writes_the_same_factors(caplog):
    """A book written before the curve and FX surface families took the names of the types they
    write keys them `InterestRateCurveParameters` and `FXVolSurfaceParameters`. The loader files
    each entry under `InterestRate` and `FXVol` and logs both spellings, so the job bootstraps the
    same curve and surface, bit for bit, as the one keyed by the families' names.

    Killing mutation: the loader's table of retired keys dropped - the bootstrap refuses
    `InterestRateCurveParameters` as no price family."""
    written = []
    for families in (('InterestRate', 'FXVol'),
                     ('InterestRateCurveParameters', 'FXVolSurfaceParameters')):
        caplog.clear()
        with caplog.at_level(logging.INFO):
            config = derivus.Context().load_json((keyed_job(families), 'keyed')).current_cfg
            config.bootstrap()
        assert list(config.params['Bootstrapper Configuration']) == ['InterestRate', 'FXVol']
        written.append(json.dumps(config.params['Price Factors'], cls=CustomJsonEncoder,
                                  sort_keys=True))

    assert {'InterestRate.ZAR-JIBAR-3M', 'FXVol.USD.ZAR'} <= set(config.params['Price Factors'])
    assert written[1] == written[0], 'the retired keys wrote other factors'
    assert 'InterestRateCurveParameters is read as InterestRate' in caplog.text
    assert 'FXVolSurfaceParameters is read as FXVol' in caplog.text


def test_the_blocks_are_solved_in_dependency_order():
    """A projection curve solved before the discount curve it prices against is solved against a
    curve that does not exist. Authoring the two blocks the wrong way round has to change nothing,
    because `Discount_Rate` says which is which and the family reads it.

    Killing mutation: the blocks solved in the order authored."""
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
    be solving for an unknown no instrument identifies.

    Killing mutation: `Use` unread - the held-out quote keeps its knot."""
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

    Killing mutation: the refusal dropped - the solve dies on a singular matrix naming nothing.
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


@pytest.mark.parametrize('knob,value', [('N_Iter', 1), ('Damping_Halvings', -1)])
def test_the_solver_knobs_are_read_off_the_block(knob, value):
    """The knobs are JSON, so a job tightens or loosens the solve with no code edit. Each value here
    is unsatisfiable: one Newton iteration cannot converge a seven-knot curve, and `-1` halvings
    forbids even the full step - `-1` rather than `0` because the full step is what these worlds
    always take, so no non-negative value fails here.

    Killing mutation: the iteration cap read off a constant rather than the block.
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

    Killing mutation: the split left off the seed - the round trip misses by 1.9e-05.
    """
    interp = ModelParams()
    interp.append('InterestRate', (), 'HermiteRT')
    market_prices, true_factors, _ = authored_world('zaronia', interp)
    solved = bootstrapped(market_prices, 'ZAR', 'ZAR-ZARONIA', interp=interp)
    factor = solved['InterestRate.ZAR-ZARONIA']
    assert factor['Near_Interpolation'] == 'LinearRT'
    assert factor['Near_Date'] == BASE + pd.DateOffset(months=18)
    assert np.abs(factor['Curve'].array[:, 1] -
                  true_factors['InterestRate.ZAR-ZARONIA']['Curve'].array[:, 1]).max() < 1e-10

    plain = bootstrapped(*authored_world('zar')[:1], 'ZAR', 'ZAR-JIBAR-3M')
    assert set(plain['InterestRate.ZAR-JIBAR-3M']) == {
        'Property_Aliases', 'Sub_Type', 'Currency', 'Day_Count', 'Curve'}


def test_a_curves_own_rule_builds_it_and_leaves_its_neighbour_on_the_default():
    """TWO CURVES ON ONE BOOK UNDER TWO SCHEMES. `Price Factor Interpolation` names one method per
    routed factor type and, in its filter half, a rule per factor `id` - the dotted curve name -
    which `ModelParams.search` resolves FIRST. So a desk's projection curve is HermiteRT while the
    OIS curve it discounts on stays Linear, which one method per type could not say.

    The solve reads the section the pricers read, so the round trip closes on both curves at once:
    each is recovered to 1e-10 from quotes generated under its OWN scheme, and each block reprices
    at par read the same way.

    The same quotes solved under the type default alone build the projection curve Linear, which
    misses the one its quotes came from by **7.507e-06**, while the OIS curve beside it is the same
    curve to the bit - so the rule is what the solve read. The par check cannot see the difference:
    a solve drives its own benchmarks to par under whatever scheme it is using.

    Killing mutation: the rule ignored and the type default read alone - the two solves are one.
    """
    interp = ModelParams()
    interp.append('InterestRate', (), 'Linear')
    interp.append('InterestRate', ('id', 'USD-3M'), 'HermiteRT')
    market_prices, true_factors, knots = authored_world('usd', interp)
    linear = ModelParams()
    linear.append('InterestRate', (), 'Linear')
    plain = bootstrapped(copy.deepcopy(market_prices), 'USD', 'USD-OIS', interp=linear)
    solved = bootstrapped(market_prices, 'USD', 'USD-OIS', interp=interp)
    for name in knots:
        assert np.abs(solved[name]['Curve'].array[:, 1] -
                      true_factors[name]['Curve'].array[:, 1]).max() < 1e-10, name
    assert np.abs(plain['InterestRate.USD-3M']['Curve'].array[:, 1] -
                  solved['InterestRate.USD-3M']['Curve'].array[:, 1]).max() > 1e-6
    assert np.array_equal(plain['InterestRate.USD-OIS']['Curve'].array,
                          solved['InterestRate.USD-OIS']['Curve'].array)


def test_the_near_splits_far_leg_is_the_curves_own_rule():
    """THE NEAR SPLIT AND THE RULE COMPOSE. `Near_Interpolation` to `Near_Tenor` stays the block's,
    written onto the factor as `Near_Interpolation`/`Near_Date`; the FAR leg is whatever the
    section resolves for this curve, which is now its own rule rather than the type default. A
    ZARONIA curve quoted monthly to the last policy meeting is LinearRT to 18M and HermiteRT
    beyond, on a book whose every other curve is Linear.

    The same quotes solved with the far leg read off the type default build it Linear, which misses
    the curve its quotes came from by **7.879e-05** against the 1e-10 the rule's solve holds.

    Killing mutation: the rule ignored - the far leg is the type default's and the two solves are
    one.
    """
    interp = ModelParams()
    interp.append('InterestRate', (), 'Linear')
    interp.append('InterestRate', ('id', 'ZAR-ZARONIA'), 'HermiteRT')
    market_prices, true_factors, _ = authored_world('zaronia', interp)
    linear = ModelParams()
    linear.append('InterestRate', (), 'Linear')
    plain = bootstrapped(copy.deepcopy(market_prices), 'ZAR', 'ZAR-ZARONIA', interp=linear)
    solved = bootstrapped(market_prices, 'ZAR', 'ZAR-ZARONIA', interp=interp)
    assert np.abs(plain['InterestRate.ZAR-ZARONIA']['Curve'].array[:, 1] -
                  solved['InterestRate.ZAR-ZARONIA']['Curve'].array[:, 1]).max() > 1e-5
    assert np.abs(solved['InterestRate.ZAR-ZARONIA']['Curve'].array[:, 1] -
                  true_factors['InterestRate.ZAR-ZARONIA']['Curve'].array[:, 1]).max() < 1e-10


def test_a_near_interpolation_without_its_tenor_refuses_by_name():
    """The split needs the date it stops at: a block naming the scheme and no tenor is refused
    before a quote is read, rather than dying on a date plus an empty string.

    Killing mutation: the tenor's presence unchecked."""
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
    2612 items rather than a difference - both priced as one base-valuation document.

    Killing mutation: the term coupon's OIS compounding read as averaging its one reset.
    """
    _, price_factors, _ = authored_world('usd')
    flat = dict(price_factors)
    flat['InterestRate.USD-OIS'] = dict(
        price_factors['InterestRate.USD-OIS'],
        Curve=utils.Curve([], [(t, 0.04) for t in
                               price_factors['InterestRate.USD-OIS']['Curve'].array[:, 0]]))

    for label, factors in (('flat', flat), ('sloped', price_factors)):
        deals = []
        for months in (12, 24, 60, 120):
            deals += [ois_swap('LIST{}'.format(months), 'USD', 'USD-OIS', months, 4.0),
                      par_swap('TERM{}'.format(months), 'USD', 'USD-OIS', 'USD-OIS', 0, 4.0,
                               fixed_frequency=12, float_frequency=12, months=months,
                               compounding='OIS')]
        marks, _ = book.marks(deals, {'InterestRate.USD-OIS': factors['InterestRate.USD-OIS']})
        for months in (12, 24, 60, 120):
            pvs = [float.fromhex(marks[kind + str(months)]) for kind in ('LIST', 'TERM')]
            assert pvs[0] == pytest.approx(pvs[1], abs=1e-8), (label, months, pvs)


def test_a_tighter_tolerance_still_converges_to_the_same_curve():
    """The other direction: `Tol` is a floor on the step, not a target, so asking for less than the
    default cannot move the answer - it can only cost an iteration. A knob that changed the number
    would be a knob nobody could safely turn.

    Killing mutation: `Tol` read as a target on the residual - 1e-16 is never reached."""
    market_prices, true_factors, _ = authored_world('zar')
    loose = bootstrapped(market_prices, 'ZAR', 'ZAR-JIBAR-3M')
    market_prices['InterestRatePrices.ZAR-JIBAR-3M']['instrument']['Tol'] = 1e-16
    tight = bootstrapped(market_prices, 'ZAR', 'ZAR-JIBAR-3M')
    curve_name = 'InterestRate.ZAR-JIBAR-3M'
    assert np.abs(tight[curve_name]['Curve'].array[:, 1] -
                  loose[curve_name]['Curve'].array[:, 1]).max() < 1e-14
    assert np.abs(tight[curve_name]['Curve'].array[:, 1] -
                  true_factors[curve_name]['Curve'].array[:, 1]).max() < 1e-10


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
            bootstrappers.InterestRate.quote_knots(
                block_nodes(usd_block, USD_CURVE, 0.0), BASE, 'ACT_365', {}),
            FX_USD_TRUE)))}
    for point, quote in zip(usd_block['Points'], par_quotes(usd_block, USD_CURVE, price_factors)):
        point['Quoted_Market_Value'] = quote
    return {name: {'instrument': block, 'Children': []} for name, block in blocks.items()}


def fx_bootstrapped(market_prices, price_factors=None, config=None):
    """`Config.bootstrap` over the cross-currency world from a `Price Factors` holding the two spots
    and no curve at all, so every curve it comes back with was solved here."""
    config = config or Config(base_currency='ZAR')
    config.params['System Parameters']['Base_Date'] = BASE
    config.params['Price Factors'] = price_factors or fx_price_factors()
    config.params['Market Prices'] = market_prices
    config.params['Bootstrapper Configuration'] = {'InterestRate': {}}
    config.bootstrap()
    return config.params['Price Factors']


def fx_par_outrights(market_prices, price_factors):
    """Each forward benchmark's PAR outright off `price_factors` - the outright at which a FRESH
    `FXForwardDeal` is worth exactly zero, priced by the same pricer the solve used.

    `par_quotes` is the rate worlds' own affine root and it needs no adjusting to read an amount:
    PV is affine in the quote whatever the quote MEANS, so bracketing at outrights 0 and 1 returns
    the par outright exactly rather than approximately.
    """
    block = market_prices['InterestRatePrices.' + ZAR_CURVE]['instrument']
    return par_quotes(block, ZAR_CURVE, price_factors)


def test_a_forward_curve_reprices_the_outrights_it_was_solved_from():
    """THE IDENTITY. Solve a ZAR curve from USDZAR outrights, then price a fresh par forward off the
    solved pair: the par outright is the quote back, to 1e-9 relative.

    Covered interest parity CLOSING THROUGH THE ENGINE'S OWN PRICERS - no formula for the forward is
    written here or in the family. The residual is `FXForwardDeal.generate` held at zero, so
    whatever parity that pricer means is the parity the curve carries.

    Killing mutation: the solver returning its seed.
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
    them unidentified.

    Killing mutation: a benchmark's knot read at its first cashflow rather than its last."""
    market_prices = fx_world()
    solved = fx_bootstrapped(market_prices)
    knots = solved['InterestRate.' + ZAR_CURVE]['Curve'].array[:, 0]
    settlement = np.array([utils.DayCount.accrual(
        BASE, ((BASE + pd.DateOffset(months=months)) - BASE).days,
        utils.DayCount.code('ACT_365')) for months, _ in FX_OUTRIGHTS])
    assert (np.diff(knots) > 0).all(), 'the forward knots are not increasing: {}'.format(knots)
    assert np.abs(knots - settlement).max() == 0.0, '{} against {}'.format(knots, settlement)


def test_a_held_out_forward_drops_its_knot():
    """`Use` on a forward is `Use` on any other benchmark: the knot grid IS the used quotes'
    settlement dates, so dropping the 1Y outright shortens the curve by one knot and moves none of
    the others - each forward identifies its own discount factor and nothing beyond it.

    Killing mutation: `Use` unread - the held-out forward keeps its knot."""
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
    nothing - a no-op where the declaration was already enough, as the rate worlds' order is.

    Killing mutation: the ordering read stopping at the block's `Discount_Rate`.
    """
    market_prices = fx_world()
    zar_name, usd_name = 'InterestRatePrices.' + ZAR_CURVE, 'InterestRatePrices.' + USD_CURVE
    assert market_prices[zar_name]['instrument']['Discount_Rate'] == '', (
        'the gate is void unless the block declares nothing - the deals have to be what orders it')

    reversed_blocks = dict(reversed(list(market_prices.items())))
    assert list(reversed_blocks) == [zar_name, usd_name]

    solved = fx_bootstrapped(reversed_blocks)
    quoted = np.array([outright for _, outright in FX_OUTRIGHTS])
    assert np.abs(fx_par_outrights(reversed_blocks, solved) / quoted - 1.0).max() < 1e-9


def test_a_forward_block_refuses_quote_sensitivity():
    """THE REFUSAL, and the reason it is not a zero. `Quote_Sensitivity` reports dV/dq by overlaying
    the CASHFLOW SCHEDULE COLUMNS a quote writes; an outright writes into `Buy_Amount`, read as a
    float off the deal, so no column moves and the overlay carries nothing. A zero delta on the
    instrument a desk actually trades is the failure this switch exists to prevent, so the block
    refuses by name and says which benchmark and which type could not be carried.

    Killing mutation: the overlay's measurement dropped - the block publishes a zero delta.
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
    `Quote_Sensitivity` still solves, still to 1e-10, and still leaves its quote leaf behind.

    Killing mutation: the refusal made on the deal type rather than measured."""
    market_prices, true_factors, _ = authored_world('zar')
    block = market_prices['InterestRatePrices.ZAR-JIBAR-3M']['instrument']
    block['Quote_Sensitivity'] = 'Yes'
    config = Config(base_currency='ZAR')
    price_factors = bootstrapped(market_prices, 'ZAR', 'ZAR-JIBAR-3M', config=config)
    curve_name = 'InterestRate.ZAR-JIBAR-3M'
    assert np.abs(price_factors[curve_name]['Curve'].array[:, 1] -
                  true_factors[curve_name]['Curve'].array[:, 1]).max() < 1e-10
    descriptors, quotes = config.quote_leaves['InterestRatePrices.ZAR-JIBAR-3M']
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

    Killing mutation: a set spanning two reporting currencies compiled as one system.
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
        fx_bootstrapped(solo, price_factors)
    assert 'FXForwardDeal' in str(alone.value), str(alone.value)


def discount(day, rate=0.04):
    """The declared-defaults world's flat curves by hand: USD 4%, USD-PROJ 4.5%, ACT/365."""
    return math.exp(-rate * (day - BASE).days / 365.0)


def forward(start, end):
    """The simple ACT/360 forward off USD-PROJ."""
    return (discount(start, 0.045) / discount(end, 0.045) - 1.0) / ((end - start).days / 360.0)


def value(flows, at=BASE):
    """The `(pay day, amount)` flows still to pay on `at`, discounted to it."""
    return sum(amount * discount(day) / discount(at) for day, amount in flows if day >= at)


def test_an_amortising_deposit_repays_each_step_on_the_day_its_balance_steps_down():
    """The trial's two amortising deposits: a million lent today for two years in 6M periods, 4.2%
    ACT/360 pinned or forecast off USD-PROJ, 250,000 amortised on the 18-month date. By hand, with
    balances N = (1e6, 1e6, 1e6, 750,000) over the period ends T_1..T_4,

        V = -1,000,000 + sum_i N_i I_i D(T_i) + 250,000 D(T_3) + 750,000 D(T_4),

    I_i being 0.042 a_i pinned and P(T_{i-1}) / P(T_i) - 1 forecast. The control beside them is the
    declared-defaults book's bullet, a million for six months at 4.2%, which repays its whole
    balance with its one coupon: -1,000,000 + 1,000,000 (1 + 0.042 a) D(T).

    KILLING MUTATION: no step written and `add_fixed_payments` back on `Start_Maturity`, the whole
    Amount in the last row: the pinned deposit reads -745.76 against 3,902.87.
    """
    dates = [BASE + pd.DateOffset(months=6 * k) for k in range(5)]
    balances = [1e6, 1e6, 1e6, 750_000.0]
    principal = -1e6 + 250_000.0 * discount(dates[3]) + 750_000.0 * discount(dates[4])
    pinned = principal + sum(n * 0.042 * (e - s).days / 360.0 * discount(e)
                             for n, s, e in zip(balances, dates, dates[1:]))
    forecast = principal + sum(n * (discount(s, 0.045) / discount(e, 0.045) - 1.0) * discount(e)
                               for n, s, e in zip(balances, dates, dates[1:]))
    floating = next(deal for deal in trial_rates.DEALS if deal['Reference'] == 'DEPO_AMORT_FLOAT')
    bullet = next(deal for deal in book.BOOK if deal['Reference'] == 'DEPO')
    marks, _ = book.marks([trial_rates.DEPOSIT, floating, bullet])
    assert float.fromhex(marks['DEPO_AMORT']) == pytest.approx(pinned, rel=1e-12)
    assert float.fromhex(marks['DEPO_AMORT_FLOAT']) == pytest.approx(forecast, rel=1e-12)
    assert float.fromhex(marks['DEPO']) == pytest.approx(
        -1e6 + 1e6 * (1.0 + 0.042 * (dates[1] - BASE).days / 360.0) * discount(dates[1]), rel=1e-12)


def test_a_coupon_on_several_resets_averages_them_unless_its_leg_compounds_them():
    """A two-year swap receiving 6M coupons on a 3M index, a million a side, paying 4.5% ACT/360
    annually, and the same floating leg as a list of one item per coupon carrying both resets. By
    hand, F(s, e) the simple ACT/360 forward off USD-PROJ and a the coupon's ACT/360 accrual:

        averaged:    sum_k 1e6 (F(s_k, m_k) + F(m_k, e_k)) / 2  a_k D(e_k)
        compounded:  sum_k 1e6 (P(s_k) / P(e_k) - 1) D(e_k)

    the swap adding its fixed leg. `None` averages - 164.33 and the list 86,228.29 - and `OIS`
    compounds, which telescopes to the one-reset swap's 652.82. Seasoned three months, its first
    fixing known at 4%, the swap's first coupon averages that fixing with the forecast one. At a
    margin m - 50bp on the swap, 50, -25, 100 and 10bp on the list's coupons - g_j = 1 + F_j a_j,
    the three margin methods compound per coupon as the ISDA meanings place the margin:

        Exclude_Margin:  1e6 (g_1 g_2 - 1 + m (a_1 + a_2))
        Include_Margin:  1e6 ((g_1 + m a_1)(g_2 + m a_2) - 1)
        Flat:            1e6 (g_1 g_2 - 1 + m (a_1 g_2 + a_2))

    and across two coupons the list pays on one date the first two compose exactly, Flat there
    refusing by name; a two-reset coupon paid alone beside two one-reset rows sharing a date is
    nothing Flat refuses, and marks the hand. A `Pre_Aggregation` cap list caps each fixing and
    places no margin, so it reads None and OIS alike and refuses the three. Under a credit Monte
    Carlo with no volatility every path is today's curve, so the averaged swap's value on each date
    of its profile is its flows still to pay, off today's forwards, discounted to that date - a
    fixing taken since the base date read back at its weight as a forecast one is.

    KILLING MUTATIONS: the fold keyed on the shape again, every such row compounded
    (`method == 'None'` read as `False`): the averaged swap reads 652.82 against 164.33; the margin
    left out of Include_Margin's fixings, or read off the first coupon's; Flat's margin compounded
    with them, or left simple; the mixed Flat shape folded, or refused leg-wide; a pre-aggregated
    list's margin method priced. And a fixing counted `known` by the valuation slice rather than
    the base date (`Reset_Day < time_slice.max()`): every base valuation holds, and the Monte Carlo
    misprices every coupon from its first fixing on.
    """
    def legs(start, known=None):
        """The averaged and the compounded floating flows and the fixed ones, `(pay day, amount)`."""
        averaged, compounded = [], []
        for k in range(4):
            begin, end = (start + pd.DateOffset(months=6 * k),
                          start + pd.DateOffset(months=6 * k + 6))
            mid = begin + pd.DateOffset(months=3)
            first = known if begin < BASE else forward(begin, mid)
            averaged.append((end, 1e6 * (first + forward(mid, end)) / 2 * (end - begin).days / 360.0))
            compounded.append((end, 1e6 * (discount(begin, 0.045) / discount(end, 0.045) - 1.0)))
        fixed = [(end, -1e6 * 0.045 * (end - begin).days / 360.0) for begin, end in (
            (start, start + pd.DateOffset(years=1)),
            (start + pd.DateOffset(years=1), start + pd.DateOffset(years=2)))]
        return averaged, compounded, fixed

    swap =dict(par_swap('SWAP', 'USD', 'USD-PROJ', 'USD', 2, 4.5, fixed_frequency=12,
                         float_frequency=6), Index_Tenor=pd.DateOffset(months=3))
    items = []
    for begin, end in ((BASE + pd.DateOffset(months=6 * k), BASE + pd.DateOffset(months=6 * k + 6))
                       for k in range(4)):
        mid = begin + pd.DateOffset(months=3)
        items.append({
            'Payment_Date': end, 'Notional': 1e6, 'Accrual_Start_Date': begin,
            'Accrual_End_Date': end, 'Accrual_Day_Count': 'ACT_360',
            'Accrual_Year_Fraction': (end - begin).days / 360.0, 'Resets': [
                [s, s, e, (e - s).days / 360.0, pd.DateOffset(days=1), 'ACT_360', '0D', 0.0, 'No',
                 utils.Percent(0.0)] for s, e in ((begin, mid), (mid, end))],
            'Margin': utils.Basis(0.0), 'Fixed_Amount': 0.0, 'FX_Reset_Date': None,
            'Known_FX_Rate': 0.0})
    seasoned = BASE - pd.DateOffset(months=3)
    marks, _ = book.marks([
        swap, dict(swap, Reference='OIS', Compounding_Method='OIS'),
        trial_rates.float_list('LIST', 'Buy', items),
        dict(swap, Reference='SEASONED', Effective_Date=seasoned,
             Maturity_Date=seasoned + pd.DateOffset(years=2),
             Known_Rates=utils.DateList({seasoned: 4.0}))])

    averaged, compounded, fixed = legs(BASE)
    assert float.fromhex(marks['SWAP']) == pytest.approx(value(averaged + fixed), rel=1e-12)
    assert float.fromhex(marks['OIS']) == pytest.approx(value(compounded + fixed), rel=1e-12)
    assert float.fromhex(marks['LIST']) == pytest.approx(value(averaged), rel=1e-12)
    averaged_seasoned, _, fixed_seasoned = legs(seasoned, known=0.04)
    assert float.fromhex(marks['SEASONED']) == pytest.approx(
        value(averaged_seasoned + fixed_seasoned), rel=1e-12)
    def margined(method, coupons, bp):
        """`(pay day, amount)` per coupon, `bp[k]` its margin, `coupons` the lists paid together."""
        flows = []
        for paid in coupons:
            pot, simple = 1.0, 0.0
            for k in paid:
                begin, end = BASE + pd.DateOffset(months=6 * k), BASE + pd.DateOffset(months=6 * k + 6)
                mid, m = begin + pd.DateOffset(months=3), bp[k] / 1e4
                a1, a2 = (mid - begin).days / 360.0, (end - mid).days / 360.0
                g1, g2 = 1.0 + forward(begin, mid) * a1, 1.0 + forward(mid, end) * a2
                pot *= {'Exclude_Margin': g1 * g2, 'Include_Margin': (g1 + m * a1) * (g2 + m * a2),
                        'Flat': g1 * g2 + m * (a1 * g2 + a2)}[method]
                simple += m * (a1 + a2) if method == 'Exclude_Margin' else 0.0
            flows.append((end, 1e6 * (pot - 1.0 + simple)))
        return flows

    # a margin of its own on each listed coupon; coupon 1 again as two one-reset rows on its date
    each, bp = [[k] for k in range(4)], (50.0, -25.0, 100.0, 10.0)
    margin_items = [dict(item, Margin=utils.Basis(margin)) for item, margin in zip(items, bp)]
    (begin, mid), (_, end) = [(reset[1], reset[2]) for reset in items[1]['Resets']]
    lone = margin_items[:1] + [
        dict(margin_items[1], Accrual_End_Date=mid, Accrual_Year_Fraction=(mid - begin).days / 360.0,
             Resets=items[1]['Resets'][:1]),
        dict(margin_items[1], Accrual_Start_Date=mid, Accrual_Year_Fraction=(end - mid).days / 360.0,
             Resets=items[1]['Resets'][1:])]
    cap = {'Digital_Payoff_Rate': None, 'Cap_Multiplier': 1.0, 'Cap_Strike': utils.Percent(4.4),
           'Floor_Multiplier': 0.0, 'Floor_Strike': utils.Percent(4.4)}
    for method in ('Exclude_Margin', 'Include_Margin', 'Flat'):
        listed, alone, mixed = (trial_rates.float_list(reference, 'Buy', rows) for reference, rows in (
            ('LISTED', margin_items), ('LONE', lone), ('MIXED', [dict(
                margin_items[0], Payment_Date=items[1]['Payment_Date'])] + margin_items[1:])))
        listed['Cashflows']['Compounding_Method'] = mixed['Cashflows']['Compounding_Method'] = method
        alone['Cashflows']['Compounding_Method'] = method
        marks, _ = book.marks([dict(swap, Compounding_Method=method, Floating_Margin=50.0), listed,
                               alone])
        assert float.fromhex(marks['SWAP']) == pytest.approx(
            value(margined(method, each, (50.0,) * 4) + fixed), rel=1e-12)
        assert float.fromhex(marks['LISTED']) == pytest.approx(value(margined(method, each, bp)),
                                                               rel=1e-12)
        assert float.fromhex(marks['LONE']) == pytest.approx(
            value(margined(method, [[0], [1]], bp)), rel=1e-12)
        if method == 'Flat':
            with pytest.raises(utils.UnpriceableSchedule, match='Compounding_Method Flat on a payment '
                                                                'date gathering several cashflows'):
                book.marks([mixed])
        else:
            assert float.fromhex(book.marks([mixed])[0]['MIXED']) == pytest.approx(
                value(margined(method, [[0, 1], [2], [3]], bp)), rel=1e-12)
    pre = {method: trial_rates.float_list('PRE_' + method, 'Buy', margin_items, optionlet=cap)
           for method in ('None', 'OIS', 'Exclude_Margin', 'Include_Margin', 'Flat')}
    for method, deal in pre.items():
        deal['Cashflows'].update(Compounding_Method=method, Averaging_Method='Pre_Aggregation')
    capped = book.marks([pre['None'], pre['OIS']])[0]
    assert capped['PRE_None'] == capped['PRE_OIS'] and math.isfinite(float.fromhex(capped['PRE_None']))
    for method in ('Exclude_Margin', 'Include_Margin', 'Flat'):
        with pytest.raises(utils.UnpriceableSchedule, match='Compounding_Method {} on a '
                                                            'Pre_Aggregation list'.format(method)):
            book.marks([pre[method]])

    calc, _ = book.simulated([swap], ('USD', 'USD-PROJ'), sigma=0.0, prec=torch.float64,
                             Generate_Cashflows='No')
    profile = next(deal.Calc_res['Value'][0] for deal in calc.netting_sets.deals()
                   if deal.Instrument.field['Reference'] == 'SWAP')
    read = [BASE + pd.Timedelta(days=int(day)) for day in calc.time_grid.mtm_time_grid[:len(profile)]]
    assert {BASE + pd.DateOffset(months=3 * k) for k in range(9)} <= set(read)
    for at, paths in zip(read, profile):
        assert paths.mean() == pytest.approx(value(averaged + fixed, at), rel=1e-9, abs=1e-6), at


def test_a_reset_fixed_by_the_base_date_reads_the_rate_its_document_states(caplog):
    """A reset behind the base date reads the rate its document states and is fixed at 0% with a
    warning naming the deal where it states none; one ON the base date reads the stated rate where
    there is one and the curve where not. By hand, each stated fixing at its rate and the rest
    at the simple ACT/360 forward off USD-PROJ, a coupon's fixings averaged, discounted on USD:

        a one-year 4.5% swap on quarterly coupons seasoned a month, its 6% stated   4,204.67
        the same swap effective today, its 6% stated today                          4,190.41
        six-month coupons on a 3M index seasoned three months, 4% then 6% today     2,797.59
        a 4.3% FRA reset a month ago, Use_Known_Rate Yes at 6%: (6% - 4.3%) a D(T)  4,315.50
        and one resetting today                                                     4,300.86

    The seasoned swap stating no row marks as one stating 0%, and so does the FRA saying
    Use_Known_Rate No, which states no rate whatever its Known_Rate, each warning by name. Under a
    credit Monte Carlo with no volatility the swap effective today reads its flows still to pay, its
    6% fixing among them, on every date.

    Killing mutations: the refusal restored, or the warning dropped;
    a row stated today unread, 303.50 off the curve; the FRA's Known_Rate read whatever its flag;
    a cashflow list's Use_Known_Rate unread today, the FRA at 413.95; a known reset counted by its
    day in the block split, 303.50, or in the coupon fold, 17,977.10; its row read by day, three
    deals skipped.
    """
    start, earlier = BASE - pd.DateOffset(months=1), BASE - pd.DateOffset(months=3)

    def swap(reference, effective, months, rows, index=0):
        """A one-year 4.5% swap from `effective` paying `months`-month coupons on an `index`-month
        index, its known rates `rows`, and its flows by hand."""
        deal = dict(par_swap(reference, 'USD', 'USD-PROJ', 'USD', 1, 4.5, float_frequency=months),
                    Effective_Date=effective, Maturity_Date=effective + pd.DateOffset(years=1),
                    Index_Tenor=pd.DateOffset(months=index), Known_Rates=utils.DateList(rows))
        flows = [(effective + pd.DateOffset(years=1), -1e6 * 0.045 * (
            effective + pd.DateOffset(years=1) - effective).days / 360.0)]
        for k in range(12 // months):
            begin, end = (effective + pd.DateOffset(months=months * k),
                          effective + pd.DateOffset(months=months * (k + 1)))
            fixings = [begin + pd.DateOffset(months=index * j) for j in range(months // (index or months))]
            rates = [rows[day] / 100.0 if day in rows else forward(day, min(day + pd.DateOffset(
                months=index or months), end)) for day in fixings]
            flows.append((end, 1e6 * sum(rates) / len(rates) * (end - begin).days / 360.0))
        return deal, value(flows), flows

    def agreement(reference, months_ago):
        """A three-month 4.3% FRA reset `months_ago` months ago, fixed at 6%, and its mark by hand."""
        deal = dict(fra(reference, 'USD', 'USD-PROJ', 'USD', -months_ago, 3 - months_ago, 4.3),
                    Use_Known_Rate='Yes', Known_Rate=utils.Percent(6.0))
        effective, maturity = deal['Effective_Date'], deal['Maturity_Date']
        return deal, 1e6 * (0.06 - 0.043) * (maturity - effective).days / 360.0 * discount(maturity)

    documents = {
        'SEASONED': swap('SEASONED', start, 3, {start: 6.0}),
        'TODAY': swap('TODAY', BASE, 3, {BASE: 6.0}),
        'AVERAGED': swap('AVERAGED', earlier, 6, {earlier: 4.0, BASE: 6.0}, index=3),
        'FRA': agreement('FRA', 1),
        'FRA_TODAY': agreement('FRA_TODAY', 0)}
    marks, stats = book.marks([document[0] for document in documents.values()])
    assert 'Deals Skipped' not in stats, stats
    for reference, (_, hand, *_) in documents.items():
        assert float.fromhex(marks[reference]) == pytest.approx(hand, rel=1e-12), (reference, hand)

    for unstated, zero in (
            (dict(documents['SEASONED'][0], Reference='UNSTATED', Known_Rates=None),
             dict(documents['SEASONED'][0], Reference='UNSTATED', Known_Rates=utils.DateList({start: 0.0}))),
            (dict(documents['FRA'][0], Reference='FRA_UNSTATED', Use_Known_Rate='No'),
             dict(documents['FRA'][0], Reference='FRA_UNSTATED', Known_Rate=utils.Percent(0.0)))):
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            assert book.marks([unstated])[0] == book.marks([zero])[0]
        assert caplog.text.count('{}: the reset fixing 2026-07-03'.format(unstated['Reference'])) == 1

    deal, _, flows = documents['TODAY']
    calc, _ = book.simulated([deal], ('USD', 'USD-PROJ'), sigma=0.0, prec=torch.float64,
                             Generate_Cashflows='No')
    profile = next(deal.Calc_res['Value'][0] for deal in calc.netting_sets.deals()
                   if deal.Instrument.field['Reference'] == 'TODAY')
    read = [BASE + pd.Timedelta(days=int(day)) for day in calc.time_grid.mtm_time_grid[:len(profile)]]
    assert {BASE + pd.DateOffset(months=3 * k) for k in range(5)} <= set(read)
    for at, paths in zip(read, profile):
        assert paths.mean() == pytest.approx(value(flows, at), rel=1e-9, abs=1e-6), at


#: The rate-leg conventions' world: USD-PROJ's zero rates rising from 2% to 6% over five years and
#: USD 25bp under it, since a flat curve reads a reset taken a period late as the same rate.
SLOPED = {'InterestRate.' + name: {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                                   'Curve': utils.Curve([], [(t, 0.02 - shift + 0.008 * t)
                                                             for t in (0, 0.5, 1, 2, 3, 4, 5)])}
          for name, shift in (('USD-PROJ', 0.0), ('USD', 0.0025))}
START, END, QUARTER = pd.Timestamp('2026-12-16'), pd.Timestamp('2028-12-16'), pd.DateOffset(months=3)
#: the quarterly dates from a Johannesburg holiday, rolled Modified Following on Johannesburg
ROLLED = [pd.Timestamp(day) for day in (
    '2026-12-17', '2027-03-16', '2027-06-17', '2027-09-16', '2027-12-17', '2028-03-16', '2028-06-19',
    '2028-09-18', '2028-12-18')]


def year(start, end, day_count='ACT_360'):
    return utils.DayCount.accrual(start, (end - start).days, utils.DayCount.code(day_count))


def periods(dates, pay=lambda s, e: e, fixing=lambda s, e: s, window=lambda s, e: (s, e)):
    """`(start, end, pay, fixing, window start, window end)` per period of `dates`."""
    return [(s, e, pay(s, e), fixing(s, e)) + window(s, e) for s, e in zip(dates[:-1], dates[1:])]


def listed(reference, rows, index='ACT_360', known=None):
    """A swap paying 4% ACT/360 on a million as the two cashflow lists its `rows` spell, the
    fixings behind the base date reading `known`."""
    fixed = [{'Payment_Date': pay, 'Notional': 1e6, 'Rate': utils.Percent(4.0),
              'Accrual_Start_Date': s, 'Accrual_End_Date': e, 'Accrual_Day_Count': 'ACT_360',
              'Accrual_Year_Fraction': year(s, e), 'Fixed_Amount': 0.0, 'Discounted': 'No',
              'FX_Reset_Date': None, 'Known_FX_Rate': 0.0} for s, e, pay, *_ in rows]
    floating = [{'Payment_Date': pay, 'Notional': 1e6, 'Accrual_Start_Date': s, 'Accrual_End_Date': e,
                 'Accrual_Day_Count': 'ACT_360', 'Accrual_Year_Fraction': year(s, e), 'Resets': [
                     [fixing, w0, w1, year(w0, w1, index), pd.DateOffset(days=1), index, '0D', 0.0,
                      'No', utils.Percent((known or {}).get(fixing, 0.0))]],
                 'Margin': utils.Basis(0.0), 'Fixed_Amount': 0.0, 'FX_Reset_Date': None,
                 'Known_FX_Rate': 0.0} for s, e, pay, fixing, w0, w1 in rows]
    return [trial_rates.fixed_list(reference + '_FIXED', 'Sell', fixed),
            trial_rates.float_list(reference + '_FLOAT', 'Buy', floating)]


def test_a_swap_leg_is_the_schedule_its_conventions_spell():
    """A two-year quarterly swap from 16 December 2026, a million paying 4% ACT/360 against the
    sloped index, each convention stated against the two cashflow lists it means, authored by hand,
    to the bit: `Payment_Offset` 2 pays every coupon two weekdays after its period ends; `Begin`
    timing pays it on its start; a Johannesburg accrual calendar rolls the dates Modified Following
    (eight of nine move) and pays on the rolled ends; with the payment calendar too and an offset of
    five the payment counts Johannesburg business days; a `First_Coupon_Date` of 16 June 2027 makes
    a long first period, one of 16 January rolls the schedule forward from it to a short first and
    a short last period, and a `Penultimate_Coupon_Date` of 16 November 2028 rolls it back from that
    day; `Arrears` fixes each coupon on its period's end over the window of its length from there;
    `Index_Day_Count` ACT_365 accrues the index rate on its own basis; and a swap seasoned from 17
    June 2026 fixing one Johannesburg business day before each start reads the 6% stated for 15 June,
    the 16th being a holiday.

    KILLING MUTATIONS, each the convention unread again, by the mark it reads against the hand's:
    the payment offset, +2.87; the timing, -67.66; the accrual calendar (the dates rolled on the
    payment one alone), -129.47; the offset counted in weekdays, +1.37; the first coupon date, -77.06
    (and +22.97 short); the penultimate, +22.90; arrears, -7,742.34; the index day count, -1,113.95;
    the fixing offset or its calendar, or the known rate looked up by the window's start, each reading
    the 5% stated for the 16th, -2,549.74.
    """
    dates = [END - QUARTER * k for k in range(9)][::-1]
    first, penultimate, seasoned = (pd.Timestamp(day) for day in ('2027-01-16', '2028-11-16', '2026-06-17'))
    jhb = {'Pay_Accrual_Calendars': book.CALENDAR, 'Receive_Accrual_Calendars': book.CALENDAR}
    paid = [pd.Timestamp(day) for day in ('2027-03-24', '2027-06-24', '2027-09-23', '2027-12-24',
                                          '2028-03-24', '2028-06-26', '2028-09-26', '2028-12-27')]
    cases = {
        'OFFSET': ({'Pay_Payment_Offset': 2, 'Receive_Payment_Offset': 2},
                   periods(dates, pay=lambda s, e: e + pd.offsets.BDay(2))),
        'BEGIN': ({'Pay_Timing': 'Begin', 'Receive_Timing': 'Begin'}, periods(dates, pay=lambda s, e: s)),
        'ACCRUAL': (jhb, periods(ROLLED)),
        'PAYMENT': (dict(jhb, Pay_Payment_Calendars=book.CALENDAR, Receive_Payment_Calendars=book.CALENDAR,
                         Pay_Payment_Offset=5, Receive_Payment_Offset=5),
                    periods(ROLLED, pay=lambda s, e: paid[ROLLED.index(e) - 1])),
        'LONG': ({'Pay_First_Coupon_Date': dates[2], 'Receive_First_Coupon_Date': dates[2]},
                 periods([START] + dates[2:])),
        'SHORT': ({'Pay_First_Coupon_Date': first, 'Receive_First_Coupon_Date': first},
                  periods([START] + [first + QUARTER * k for k in range(8)] + [END])),
        'BACK': ({'Pay_Penultimate_Coupon_Date': penultimate,
                  'Receive_Penultimate_Coupon_Date': penultimate},
                 periods([START] + [penultimate - QUARTER * k for k in range(8)][::-1] + [END])),
        'ARREARS': ({'Reset_Type': 'Arrears'},
                    periods(dates, fixing=lambda s, e: e, window=lambda s, e: (e, e + (e - s)))),
        'INDEX': ({'Index_Day_Count': 'ACT_365'}, periods(dates))}
    documents = []
    for reference, (conventions, rows) in cases.items():
        documents += [dict(par_swap(reference, 'USD', 'USD-PROJ', 'USD', 2, 4.0, fixed_frequency=3,
                                    float_frequency=3), Effective_Date=START, Maturity_Date=END,
                           **conventions)]
        documents += listed('HAND_' + reference, rows, conventions.get('Index_Day_Count', 'ACT_360'))
    fixed = {pd.Timestamp('2026-06-15'): 6.0, pd.Timestamp('2026-06-16'): 5.0}
    documents += [dict(par_swap('SEASONED', 'USD', 'USD-PROJ', 'USD', 2, 4.0, fixed_frequency=3,
                                float_frequency=3), Effective_Date=seasoned,
                       Maturity_Date=seasoned + pd.DateOffset(years=2), Index_Offset=1,
                       Index_Calendars=book.CALENDAR, Known_Rates=utils.DateList(fixed))]
    # a fixing still to come forecasts its window whatever day it fixes on
    documents += listed('HAND_SEASONED', periods(
        [seasoned + QUARTER * k for k in range(9)],
        fixing=lambda s, e: pd.Timestamp('2026-06-15') if s == seasoned else s - pd.offsets.BDay(1)),
        known=fixed)
    marks, stats = book.marks(documents, factors=SLOPED)
    assert 'Deals Skipped' not in stats, stats
    for reference in list(cases) + ['SEASONED']:
        hand = float.fromhex(marks['HAND_{}_FIXED'.format(reference)]) + float.fromhex(
            marks['HAND_{}_FLOAT'.format(reference)])
        assert float.fromhex(marks[reference]) == hand, (reference, float.fromhex(marks[reference]) - hand)


#: Each reader of `Index_Day_Count` on ACT/360 legs stating none: the two-year quarterly swap, a
#: cap at 4% on a million from 16 December 2026, and the declared-defaults swaption, whose underlying
#: tenor the index's day count reads, on a surface rising a vol point a year of tenor.
INDEXED = {
    'swap': dict({key: value for key, value in par_swap(
        'SWAP', 'USD', 'USD-PROJ', 'USD', 2, 4.0, fixed_frequency=3, float_frequency=3).items()
        if key != 'Index_Day_Count'}, Effective_Date=START, Maturity_Date=END),
    'cap': {'Object': 'CapDeal', 'Reference': 'CAP', 'Currency': 'USD', 'Discount_Rate': 'USD',
            'Forecast_Rate': 'USD-PROJ', 'Forecast_Rate_Volatility': 'USD-PROJ', 'Buy_Sell': 'Buy',
            'Effective_Date': START, 'Maturity_Date': END, 'Cap_Rate': 4.0, 'Principal': 1e6,
            'Accrual_Day_Count': 'ACT_360'},
    'swaption': dict(next(deal for deal in book.BOOK if deal['Reference'] == 'SWPT'),
                     Receive_Day_Count='ACT_360')}
RISING = {'InterestYieldVol.USD-PROJ': {
    'Property_Aliases': None, 'Distribution_Type': 'Lognormal', 'Shift': utils.Percent(0.0),
    'Surface': utils.Curve([], [(m, e, t, 0.20 + 0.01 * t) for m in (-0.02, 0.0, 0.02)
                                for e in (0.25, 1.0, 2.0, 5.0) for t in (1.0, 3.0, 5.0)])}}


@pytest.mark.parametrize('reader', list(INDEXED))
def test_a_blank_index_day_count_is_the_floating_legs_own(reader):
    """A deal on ACT/360 legs stating no `Index_Day_Count` prices bit-identically to the one stating
    `ACT_360`, on the sloped index, each in a run of its own.

    Killing mutation: a blank read as ACT/365, the swap +1,113.95 (11.1 bp of notional), the cap
    +654.90 and the swaption -10.36 against the stated.
    """
    readings = []
    for deal in (INDEXED[reader], dict(INDEXED[reader], Index_Day_Count='ACT_360')):
        marks, stats = book.marks([deal], factors=dict(SLOPED, **RISING))
        assert 'Deals Skipped' not in stats, stats
        readings.append(float.fromhex(marks[deal['Reference']]))
    assert readings[0] == readings[1], readings[0] - readings[1]


def test_two_swaptions_on_one_surface_and_expiry_each_read_their_own_tenor():
    """The declared-defaults swaption beside a copy restated to read a five-year tenor, on the surface
    rising a vol point a year of tenor, marks in either order as each marks alone, to the bit.

    Killing mutation: the tenor dropped from `VolSurface._interp`'s cache key, every later read on
    the surface and expiry taking the first deal's blend: the five-year beside the three-year
    11,809.53 against 12,771.77 alone, and the three-year beside the five-year 12,771.77 against
    11,809.53.
    """
    three = next(deal for deal in book.BOOK if deal['Reference'] == 'SWPT')
    five = dict(three, Reference='S5Y', Children=[dict(leg, Reference='S5Y' + leg['Reference'])
                                                   for leg in three['Children']],
                Swap_Maturity_Date=three['Swap_Effective_Date'] + pd.DateOffset(years=5))
    alone = {deal['Reference']: book.marks([deal], factors=RISING)[0][deal['Reference']] for deal in (three, five)}
    assert alone['SWPT'] != alone['S5Y']
    for deals in ((three, five), (five, three)):
        marks, _ = book.marks(list(deals), factors=RISING)
        assert {reference: marks[reference] for reference in alone} == alone, marks


def test_a_caplet_expires_on_the_day_its_rate_fixes():
    """A two-year quarterly cap at 4% on a million from 16 December 2026, Black at 20% on the sloped
    index, stated `Arrears` and stated an `Index_Offset` of two against the caplet lists they mean,
    to the bit: each caplet's rate fixes on its period's end over the three months from there, or
    two weekdays before its start, and the caplet expires on that fixing. A list row whose accrual
    starts on its fixing expires there however the expiry is read; and a list whose reset fixes two
    weekdays before its accrual start is that row, its year fraction its own.

    KILLING MUTATIONS: arrears unread, -4,974.97; the fixing offset unread, +10.56; and a caplet
    expiring on its accrual start rather than its fixing, -482.68 on the arrears cap and +10.56 on
    the lagged list.
    """
    optionlet = {'Digital_Payoff_Rate': None, 'Cap_Multiplier': 1.0, 'Cap_Strike': utils.Percent(4.0),
                 'Floor_Multiplier': 0.0, 'Floor_Strike': utils.Percent(4.0)}

    def caplets(reference, rows):
        """Caplets on `(accrual start, pay, fixing, window start, window end, year fraction)`."""
        return trial_rates.float_list(reference, 'Buy', [{
            'Payment_Date': pay, 'Notional': 1e6, 'Accrual_Start_Date': s, 'Accrual_End_Date': w1,
            'Accrual_Day_Count': 'ACT_365', 'Accrual_Year_Fraction': a, 'Resets': [
                [fixing, w0, w1, year(w0, w1, 'ACT_365'), pd.DateOffset(days=1), 'ACT_365', '0D', 0.0,
                 'No', utils.Percent(0.0)]], 'Margin': utils.Basis(0.0), 'Fixed_Amount': 0.0,
            'FX_Reset_Date': None, 'Known_FX_Rate': 0.0} for s, pay, fixing, w0, w1, a in rows],
            optionlet=optionlet)

    cap = {'Object': 'CapDeal', 'Reference': 'CAP', 'Currency': 'USD', 'Discount_Rate': 'USD',
           'Forecast_Rate': 'USD-PROJ', 'Forecast_Rate_Volatility': 'USD-PROJ', 'Buy_Sell': 'Buy',
           'Effective_Date': START, 'Maturity_Date': END, 'Cap_Rate': 4.0, 'Principal': 1e6}
    dates = [END - QUARTER * k for k in range(9)][::-1]
    spans = list(zip(dates[:-1], dates[1:]))
    lag = [s - pd.offsets.BDay(2) for s, _ in spans]
    marks, stats = book.marks([
        dict(cap, Reference='ARREARS', Reset_Type='Arrears'), dict(cap, Reference='LAGGED', Index_Offset=2),
        caplets('HAND_ARREARS', [(e, e, e, e, e + QUARTER, year(s, e, 'ACT_365')) for s, e in spans]),
        caplets('HAND_LAGGED', [(f, e, f, s, e, year(s, e, 'ACT_365')) for (s, e), f in zip(spans, lag)]),
        caplets('LISTED', [(s, e, f, s, e, year(s, e, 'ACT_365')) for (s, e), f in zip(spans, lag)])],
        factors=SLOPED)
    assert 'Deals Skipped' not in stats, stats
    assert marks['ARREARS'] == marks['HAND_ARREARS'], (float.fromhex(marks['ARREARS']) - float.fromhex(
        marks['HAND_ARREARS']))
    assert marks['LAGGED'] == marks['HAND_LAGGED'] == marks['LISTED'], (
        float.fromhex(marks['LAGGED']) - float.fromhex(marks['HAND_LAGGED']),
        float.fromhex(marks['LISTED']) - float.fromhex(marks['HAND_LAGGED']))


def test_a_quoted_deposit_is_pinned_on_the_schedule_its_stub_and_calendar_spell():
    """A year's deposit of a million from the base date paying 4.2% ACT/360 every six months between
    a first coupon on 10 August 2026, a Johannesburg holiday, and a penultimate one on 10 May 2027,
    accruing on Johannesburg, as a curve authors its benchmark at the quote. `quote` pins the rate
    on each period's unadjusted start, the day it can name with no calendar, so the deposit reads no
    forecast curve, and it accrues and pays on its rolled schedule - 3 and 11 August and 10 November
    2026, 10 May and 3 August 2027 - by hand

        -1,000,000 + sum_k 1,000,000 0.042 a_k D(T_k) + 1,000,000 D(T_4)

    on the declared-defaults world's flat 4%.

    KILLING MUTATIONS, by the mark against the hand's 2,230.43: the quote's starts written without
    either stub, or the pinned rows looked up on the rolled starts, each leaving a start unpinned so
    the deposit forecasts off USD-PROJ, +2,784.09; the first coupon date unread, -9.25; the
    penultimate, +2,680.93; the accrual calendar, -1.07.
    """
    block = completed(dict(deposit('STUB', 'USD', 'USD', 12, 0.0), Interest_Rate='USD-PROJ',
                           Payment_Frequency=pd.DateOffset(months=6),
                           Interest_Frequency=pd.DateOffset(months=6),
                           First_Coupon_Date=pd.Timestamp('2026-08-10'),
                           Penultimate_Coupon_Date=pd.Timestamp('2027-05-10'), Accrual_Calendars=book.CALENDAR))
    author_quote(block, 4.2, 'USD')
    dates = [pd.Timestamp(day) for day in ('2026-08-03', '2026-08-11', '2026-11-10', '2027-05-10', '2027-08-03')]
    hand = -1e6 + 1e6 * discount(dates[-1]) + sum(
        1e6 * 0.042 * (e - s).days / 360.0 * discount(e) for s, e in zip(dates[:-1], dates[1:]))
    marks, stats = book.marks([dict(block)])
    assert 'Deals Skipped' not in stats, stats
    assert float.fromhex(marks['STUB']) == pytest.approx(hand, rel=1e-12)
