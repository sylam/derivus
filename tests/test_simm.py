"""The SIMM calculation - ISDA's CRIF by bump and revaluation - gated through job documents on a
world of two curves, a swap and an FX option.

Every level is invented. The curves are authored first and their quotes GENERATED off them by the
library's inverse map (`bootstrappers.knot_prices`), so the bootstrap has a known curve to recover
and every CRIF row is a difference of two valuations of a known book: a USD overnight curve quoted
in deposits and annual swaps, a ZAR term curve in a deposit, FRAs and quarterly swaps, a fixed
spread riding on the ZAR curve, and a USDZAR surface on three expiry pillars. The swap IS the USD
five-year benchmark, struck at its own quote, so its PV01 has a hand figure; the option discounts
on the spread. The vertices are read off a synthetic parameters file carrying no licensed number.
"""
import copy
import json
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

import derivus
from derivus import bootstrappers, calculation, utils
from derivus.config import CustomJsonEncoder, ModelParams

from rates_world import BASE

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures')
MAPPING = os.path.join(FIXTURES, 'simm_mapping.json')
PARAMETERS = os.path.join(FIXTURES, 'simm_parameters.json')

USD_DATES = [BASE + pd.DateOffset(months=m) for m in (1, 3, 6)] + [
    BASE + pd.DateOffset(years=y) for y in (1, 2, 3, 4, 5, 7, 10)]
USD_RATES = [0.0448, 0.0445, 0.0440, 0.0430, 0.0415, 0.0405, 0.0400, 0.0398, 0.0399, 0.0402]
ZAR_DATES = [BASE + pd.DateOffset(months=m) for m in (3, 6, 9, 12)] + [
    BASE + pd.DateOffset(years=y) for y in (2, 3, 5)]
ZAR_RATES = [0.0700, 0.0705, 0.0712, 0.0720, 0.0735, 0.0750, 0.0770]
SPOT, SPREAD, NOTIONAL = 18.25, 0.005, 1e6

#: The block conventions: annual overnight swaps on USD, a quarterly term curve on ZAR quoted in
#: FRAs to a year - with `fras_to`, what `knot_prices` authors each knot as.
OIS = {'Fixed_Frequency': '1Y', 'Float_Frequency': '1Y', 'Compounding': 'OIS'}
TERM = {'Fixed_Frequency': '3M', 'Float_Frequency': '3M', 'Compounding': 'None'}

OPTION = {'Object': 'FXOptionDeal', 'Reference': 'FXO', 'Currency': 'ZAR',
          'Underlying_Currency': 'USD', 'Discount_Rate': 'ZAR-JIBAR.ZAR-JIBAR+50BP',
          'FX_Volatility': 'USD.ZAR', 'Buy_Sell': 'Buy', 'Option_Type': 'Call',
          'Strike_Price': 18.5, 'Underlying_Amount': NOTIONAL,
          'Expiry_Date': BASE + pd.DateOffset(years=1)}

#: The SIMM calculation each document carries, a valuation of one inner path the option needs not.
SIMM = {'Object': 'SIMM', 'Base_Date': BASE, 'Currency': 'USD', 'Mapping': MAPPING,
        'Parameters': PARAMETERS, 'MCMC_Simulations': 1, 'Random_Seed': 1}


def factors():
    """The world's `Price Factors`, the curves as they truly stand."""
    def curve(currency, dates, rates):
        return {'Currency': currency, 'Day_Count': 'ACT_365', 'Sub_Type': None,
                'Curve': utils.Curve([], [[(day - BASE).days / 365.0, rate]
                                          for day, rate in zip(dates, rates)])}

    return {
        'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD-OIS', 'Spot': 1.0},
        'FxRate.ZAR': {'Domestic_Currency': 'USD', 'Interest_Rate': 'ZAR-JIBAR',
                       'Spot': 1.0 / SPOT},
        'InterestRate.USD-OIS': curve('USD', USD_DATES, USD_RATES),
        'InterestRate.ZAR-JIBAR': curve('ZAR', ZAR_DATES, ZAR_RATES),
        'InterestRate.ZAR-JIBAR.ZAR-JIBAR+50BP': {
            'Currency': 'ZAR', 'Day_Count': 'ACT_365', 'Sub_Type': 'FlatSpread',
            'Curve': utils.Curve([], [[0.0, SPREAD], [30.0, SPREAD]])},
        'FXVol.USD.ZAR': {
            'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness',
            'Surface': utils.Curve([], [[m, t, 0.15 + 0.1 * (m - 1.0) ** 2 + 0.005 * t]
                                        for m in (0.8, 1.0, 1.2) for t in (0.5, 1.0, 2.0)])}}


def blocks():
    """The two curves' `Market Prices`, generated on their own knots at par."""
    found = factors()
    return dict([bootstrappers.knot_prices('USD-OIS', found, ModelParams(), BASE, {}, OIS),
                 bootstrappers.knot_prices('ZAR-JIBAR', found, ModelParams(), BASE, {}, TERM,
                                           '1Y')])


def benchmark(prices, years=5):
    """`(position, deal)` - the USD `years` benchmark, and the swap booked as it at its quote."""
    points = prices['InterestRatePrices.USD-OIS']['instrument']['Points']
    position = next(at for at, point in enumerate(points)
                    if point['Deal']['Maturity_Date'] == BASE + pd.DateOffset(years=years))
    return position, dict(points[position]['Deal'], Object='SwapInterestDeal', Reference='SWAP',
                          Discount_Rate='USD-OIS',
                          Swap_Rate=points[position]['Quoted_Market_Value'])


def world(calculation=None, prices=None, option=None, configuration=None, system=None,
          extra=None):
    """A context loaded with the swap and the option under one netting set, on the true curves,
    with `prices` - the generated quote set unless given - and the curve family configured."""
    prices = blocks() if prices is None else prices
    _, swap = benchmark(blocks())
    document = {'Calc': {
        'Calculation': dict(SIMM, **(calculation or {})),
        'Deals': {'Reference': 'simm', 'Deals': {'Children': [{
            'Instrument': {'.Deal': {'Object': 'NettingCollateralSet', 'Reference': 'NS',
                                     'Netted': 'True', 'Collateralized': 'False',
                                     'Post_Regulations': 'SEC', 'Collect_Regulations': 'SEC'}},
            'Children': [{'Instrument': {'.Deal': swap}},
                         {'Instrument': {'.Deal': dict(OPTION, **(option or {}))}}]}]}},
        'MergeMarketData': {'MarketDataFile': '', 'ExplicitMarketData': {
            'System Parameters': dict({'Base_Currency': 'USD', 'Base_Date': BASE}, **(system or {})),
            'Valuation Configuration': {}, 'Price Factors': dict(factors(), **(extra or {})),
            'Market Prices': prices,
            'Bootstrapper Configuration': {'InterestRate': {'Prices': 'InterestRate'}}
            if configuration is None else configuration}}}}
    context = derivus.Context()
    context.load_json((json.dumps(document, cls=CustomJsonEncoder), 'simm'), compress=False)
    return context


def valued(context):
    """`{reference: value}` off one base valuation of the loaded job."""
    frame = context.run_job()[1]['Results']['mtm']
    return {reference: value for reference, value in zip(frame['Reference'], frame['Value'])}


VALUATION = {'Object': 'BaseValuation'}


def rows(output, trade, risk_type):
    crif = output['Results']['CRIF']
    found = crif[(crif['TradeID'] == trade) & (crif['RiskType'] == risk_type)]
    return dict(zip(found['Label1'], found['Amount']))


@pytest.fixture(scope='module')
def serial():
    return world().run_job()


def test_the_quotes_reprice_the_book_they_were_read_off():
    """THE ROUND TRIP. Quotes authored on the curves' own knots and bootstrapped back by the run
    value the book as the curves they were read off did - the swap and the option to 1e-10
    relative - on knots the solve places exactly where the curves had them.

    Killing mutation: `knot_prices` writing each quote rounded to a thousandth of a percent, a quote
    near par rather than at it - the option's mark moves by 1.7e-5 of itself.
    """
    pasted = valued(world(VALUATION))
    context = world()
    curves = {name: context.current_cfg.params['Price Factors'][name]['Curve'].array.copy()
              for name in ('InterestRate.USD-OIS', 'InterestRate.ZAR-JIBAR')}
    trades = context.run_job()[1]['Results']['Trades']
    solved = dict(zip(trades['TradeID'], trades['Value']))
    assert abs(solved['FXO'] / pasted['FXO'] - 1.0) < 1e-10, (solved['FXO'], pasted['FXO'])
    assert abs(solved['SWAP'] - pasted['SWAP']) < 1e-10 * NOTIONAL, (solved['SWAP'], pasted['SWAP'])
    for name, curve in curves.items():
        recovered = context.current_cfg.params['Price Factors'][name]['Curve'].array
        assert np.array_equal(recovered[:, 0], curve[:, 0]), name
        assert np.abs(recovered[:, 1] - curve[:, 1]).max() < 1e-12, name


def test_a_swaps_pv01_is_its_annuity_times_a_basis_point(serial):
    """A PAR SWAP'S PV01 BY HAND. The swap is the five-year benchmark struck at its quote, so a
    basis point on that quote re-solves the curve to price the benchmark at par a basis point up
    and the swap is worth `N A' 1bp`, `A'` its fixed leg's annuity on the re-solved curve - read
    off its knots, every coupon landing on one. Each other quote moves it by nothing a solve can
    see, so its rate rows sum to that figure.

    Killing mutation: the quote moved by the shift taken as a decimal, 1e-4 on a quote in percent -
    the rows sum to a hundredth of the hand figure.
    """
    calc, output = serial
    position, _ = benchmark(calc.config.params['Market Prices'])
    knots = calc.move(('quote', 'InterestRatePrices.USD-OIS', position))[
        'InterestRate.USD-OIS']['Curve'].array
    coupons = [BASE + pd.DateOffset(years=y) for y in range(6)]
    annuity = sum((end - start).days / 365.0 * np.exp(-knots[knots[:, 0] == (
        end - BASE).days / 365.0, 1][0] * (end - BASE).days / 365.0)
                  for start, end in zip(coupons[:-1], coupons[1:]))
    trades = output['Results']['Trades']
    assert abs(trades[trades['TradeID'] == 'SWAP']['Value'].item()) < 1e-6
    assert sum(rows(output, 'SWAP', 'Risk_IRCurve').values()) == pytest.approx(
        NOTIONAL * annuity * 1e-4, rel=1e-9)


def test_a_tenor_is_split_between_the_vertices_either_side_of_it(serial):
    """THE VERTEX ALLOCATION BY HAND, on the twelve vertices the parameters file names: linear in
    years of 365 days between the two vertices either side, a tenor on a vertex landing whole, and
    beyond either end of the grid on the end vertex. On the swap its own quote matures 1826 days
    out, so all but 1/1825 of its PV01 is the 5y row.

    Killing mutation: the two weights swapped, the nearer vertex taking the smaller share - a
    tenor on the 5y vertex lands whole on 3y, and 7y reads 0.4 on 5y; 4y alone cannot tell.
    """
    grid = serial[0].vertices
    assert [label for label, _ in grid] == [
        '2w', '1m', '3m', '6m', '1y', '2y', '3y', '5y', '10y', '15y', '20y', '30y']
    assert [years for _, years in grid] == pytest.approx(
        [14 / 365.0, 1 / 12.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0], rel=1e-15)
    weights = utils.vertex_weights
    assert weights(4.0, grid) == [('3y', 0.5), ('5y', 0.5)]
    assert weights(5.0, grid) == [('5y', 1.0)]
    assert weights(0.01, grid) == [('2w', 1.0)] and weights(40.0, grid) == [('30y', 1.0)]
    assert dict(weights(7.0, grid)) == pytest.approx({'5y': 0.6, '10y': 0.4})
    assert dict(weights(0.75, grid)) == pytest.approx({'6m': 0.5, '1y': 0.5})
    tenor = 1826 / 365.0
    assert dict(weights(tenor, grid)) == pytest.approx({'5y': (10.0 - tenor) / 5.0,
                                                        '10y': (tenor - 5.0) / 5.0})


def test_the_crif_is_isdas_format(serial, tmp_path):
    """THE FILE THE MARGIN CALCULATION READS. Its columns are the ones the margin calculation's
    reader consumes, one row per trade and coordinates, the trade id the booked trade's reference;
    a rate row lands on a vertex of a curve's sub-curve, an FX row on a currency other than the
    calculation's, a vega row on a pair and a vertex; and it reads back tab separated to the bit.

    Killing mutation: the vertex dropped from a row's key (`label or ...` read as the mapping's own
    Label1) - every rate row of a curve folds into one carrying no vertex.
    """
    calc, output = serial
    frame, trades = output['Results']['CRIF'], output['Results']['Trades']
    assert tuple(frame.columns) == (
        'TradeID', 'Counterparty', 'PostRegulations', 'CollectRegulations', 'ProductClass',
        'RiskType', 'Qualifier', 'Bucket', 'Label1', 'Label2', 'Amount', 'AmountUSD')
    assert not frame.duplicated(
        ['TradeID', 'RiskType', 'Qualifier', 'Bucket', 'Label1', 'Label2']).any()
    assert set(frame['TradeID']) == {'SWAP', 'FXO'} == set(trades['TradeID'])
    assert set(frame['RiskType']) == {'Risk_IRCurve', 'Risk_FX', 'Risk_FXVol'}
    vertices = {label for label, _ in calc.vertices}
    rates = frame[frame['RiskType'] == 'Risk_IRCurve']
    assert set(rates['Label1']) <= vertices and len(set(rates['Label1'])) > 6
    assert set(zip(rates['Qualifier'], rates['Bucket'], rates['Label2'])) == {
        ('USD', '1', 'OIS'), ('ZAR', '3', 'Libor3m')}
    fx = frame[frame['RiskType'] == 'Risk_FX']
    assert list(fx['Qualifier']) == ['ZAR'] and set(fx['Label1']) == {''}
    vega = frame[frame['RiskType'] == 'Risk_FXVol']
    assert set(vega['Qualifier']) == {'USDZAR'} and set(vega['Label1']) <= vertices
    assert (frame['AmountUSD'] == frame['Amount']).all()
    assert set(frame['Counterparty']) == {'NS'} and set(frame['ProductClass']) == {'RatesFX'}

    calculation.SIMM.write(frame, tmp_path / 'crif.tsv')
    read = pd.read_csv(tmp_path / 'crif.tsv', sep='\t', dtype={'Bucket': str},
                       keep_default_na=False, float_precision='round_trip')
    assert list(read.columns) == list(frame.columns)
    assert [value.hex() for value in read['Amount']] == [value.hex() for value in frame['Amount']]


def test_the_sharded_run_equals_the_serial_run_to_the_bit(serial):
    """Two workers spawned on contexts of their own, the trades dealt between them, against the
    serial run in-process: the same CRIF to the bit, row for row, and the same trades report.

    Killing mutation: a worker keying each trade by its place in its own share rather than in the
    book - both workers answer for trade 0 and the merge has no trade 1.
    """
    output = serial[1]
    sharded = world({'Workers': 2}).run_job()[1]
    for table in ('CRIF', 'Trades'):
        mine, theirs = sharded['Results'][table], output['Results'][table]
        floats = [column for column in ('Amount', 'AmountUSD', 'Value') if column in mine]
        assert mine.drop(columns=floats).equals(theirs.drop(columns=floats)), table
        for column in floats:
            assert [x.hex() for x in mine[column]] == [x.hex() for x in theirs[column]], column


def test_a_curve_no_quote_set_stands_behind_is_refused_by_name():
    """MARKET PRICES ARE A CONDITION. The ZAR curve's quote set withdrawn, its nodes still pasted
    in, the run refuses before a trade is valued and names the curve: no margin off a curve no
    market price produced.

    Killing mutation: the condition reading every curve as quoted - the run goes on to the mapping
    and refuses there for another reason, never naming the missing quote set.
    """
    prices = blocks()
    del prices['InterestRatePrices.ZAR-JIBAR']
    with pytest.raises(ValueError, match='a configured family claims bootstraps') as refusal:
        world(prices=prices).run_job()
    assert 'InterestRate.ZAR-JIBAR' in str(refusal.value)
    assert 'FXO' in str(refusal.value)


def test_a_fixed_spread_child_of_a_quoted_parent_is_not_refused(serial):
    """The option discounts on a fixed spread over the quoted ZAR curve, which has no market rate
    of its own and rides on its parent: the walk passes, and the option moves with the parent's
    quotes and not with any of the spread's.

    Killing mutation: the condition not reading a child through its parent - the spread is refused
    as a curve no market price stands behind.
    """
    calc = serial[0]
    index = next(at for at, (_, node) in enumerate(calc.trades)
                 if node['Instrument'].field['Reference'] == 'FXO')
    assert 'InterestRate.ZAR-JIBAR.ZAR-JIBAR+50BP' in calc.walk(index)
    moved = {bump.factor for bump in calc.bumps(index)}
    assert 'InterestRate.ZAR-JIBAR' in moved
    assert 'InterestRate.ZAR-JIBAR.ZAR-JIBAR+50BP' not in moved


def test_a_quote_set_no_configured_family_solves_is_refused_by_name():
    """The condition reads what a CONFIGURED family claims, not what the section carries: with the
    quote sets present and no curve entry in `Bootstrapper Configuration`, nothing solves them and
    the curves stand as pasted nodes - refused by name, where the IR risk would otherwise vanish
    from a CRIF that still wrote its FX and vega rows.

    Killing mutation: the condition reading the section's own keys - the run passes the walk and
    refuses later, at the first quote move that re-solves nothing, never naming the condition.
    """
    with pytest.raises(ValueError, match='a configured family claims bootstraps') as refusal:
        world(configuration={}).run_job()
    assert 'InterestRate.USD-OIS' in str(refusal.value)


def test_a_curve_riding_its_calibration_is_refused_by_name():
    """A block declaring `Quote_Propagation: Linear` is valued by riding its calibration to the
    quotes standing, so a re-solve with one quote moved is never read and its delta reads nothing:
    refused by name before a trade is valued.

    Killing mutation: the refusal removed - the run writes the swap a rate delta of -0.08 where its
    hand PV01 is +443.7.
    """
    prices = blocks()
    for block in prices.values():
        block['instrument']['Quote_Propagation'] = 'Linear'
    with pytest.raises(ValueError, match='Quote_Propagation Linear') as refusal:
        world(prices=prices).run_job()
    assert 'InterestRatePrices.USD-OIS' in str(refusal.value)


def test_a_worker_that_raises_surfaces_in_the_parent():
    """A sharded run raises what its serial run raises, within a bound and never hanging: a worker's
    error crosses the queue as its answer and the parent re-raises it, the other worker stopped.
    The error is the engine's own - a spot moved to nothing values the option at NaN, which a book
    excluding no deal refuses by name.

    Killing mutation: the worker's error not put on the queue - the parent names a worker dead
    without answering, never the error the worker raised.
    """
    broken = {'Spot_Shift': -100.0}
    strict = {'Exclude_Deals_With_Missing_Market_Data': 'No'}
    with pytest.raises(utils.UnpriceableSchedule) as alone:
        world(broken, system=strict).run_job()
    found = {}

    def sharded():
        try:
            world(dict(broken, Workers=2), system=strict).run_job()
        except Exception as error:
            found['error'] = error

    runner = threading.Thread(target=sharded, daemon=True)
    runner.start()
    runner.join(300)
    assert not runner.is_alive(), 'the sharded run hangs where the serial run raises'
    assert type(found.get('error')) is type(alone.value), found
    assert str(found['error']) == str(alone.value)


@pytest.mark.parametrize('months, split', [
    (36, {'3y': 1.0 - 1.0 / 730.0, '5y': 1.0 / 730.0}),
    (1, {'1m': (0.25 - 31 / 365.0) / (0.25 - 1 / 12.0), '3m': (31 / 365.0 - 1 / 12.0) / (
        0.25 - 1 / 12.0)}),
    (30, {'2y': 3.0 - 915 / 365.0, '3y': 915 / 365.0 - 2.0})])
def test_vega_lands_on_the_options_own_expiry(months, split):
    """ISDA's vol-tenor is the OPTION's expiry. The surface carries pillars at 6m, 1y and 2y; given
    one at every vertex expiry it lacks, read off its own interpolation - its value unmoved, to the
    rounding - each vertex alone moved by a point is a tent, so a 3Y option's vega is a 3y row
    (a 5y sliver, its expiry a day past 3y), a 1M option's a 1m row, a 2.5Y option's split between
    2y and 3y. Each row is the move of the option's own vol by its tent there - as an independent
    parallel move of that size values it - and the rows sum to the parallel point's vega but for
    that move's convexity.

    Killing mutation: the surface left on its own pillars - the vertex tents land on 6m, 1y and 2y,
    a 3Y option's vega reading on 2y and a 1M option's on 6m.
    """
    expiry = BASE + pd.DateOffset(months=months)
    context = world(option={'Expiry_Date': expiry})
    calc, output = context.run_job()
    vega = rows(output, 'FXO', 'Risk_FXVol')

    index = next(at for at, (_, node) in enumerate(calc.trades)
                 if node['Instrument'].field['Reference'] == 'FXO')
    base = calc.value(index, {})[0]
    assert calc.value(index, calc.move(('augmented', 'FXVol.USD.ZAR')))[0] == pytest.approx(
        base, rel=1e-14)
    surface = context.current_cfg.params['Price Factors']['FXVol.USD.ZAR']

    def shifted(size):
        return calc.value(index, {'FXVol.USD.ZAR': dict(surface, Surface=utils.Curve(
            surface['Surface'].meta, surface['Surface'].array + [0.0, 0.0, size]))})[0] - base

    assert set(vega) == set(split)
    for label, weight in split.items():
        assert vega[label] == pytest.approx(shifted(0.01 * weight), rel=1e-6), label
    assert sum(vega.values()) == pytest.approx(shifted(0.01), rel=5e-3)


def test_a_curve_discounting_on_another_gets_a_fra_in_front():
    """`knot_prices` on a forecast curve discounting on the overnight one: its front knot, inside
    one float period, is a FRA from the base date projecting off the curve being solved - a deposit
    pins its rate and reads only the discount curve, which leaves the knot identified by nothing -
    so the block solves and returns the curve it was read off.

    Killing mutation: the front left a deposit whatever the block discounts on - the solve meets a
    singular matrix.
    """
    found = factors()
    found['InterestRate.USD-TERM'] = {
        'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
        'Curve': utils.Curve([], [[(day - BASE).days / 365.0, rate + 0.0012]
                                  for day, rate in zip(USD_DATES[1:], USD_RATES[1:])])}
    prices = blocks()
    name, block = bootstrappers.knot_prices('USD-TERM', found, ModelParams(), BASE, {},
                                            dict(TERM, Discount_Rate='USD-OIS'), '1Y')
    prices[name] = block
    context = world(VALUATION, prices=prices, extra={'InterestRate.USD-TERM': copy.deepcopy(
        found['InterestRate.USD-TERM'])})
    context.bootstrap()
    solved = context.current_cfg.params['Price Factors']['InterestRate.USD-TERM']['Curve'].array
    assert np.abs(solved[:, 1] - found['InterestRate.USD-TERM']['Curve'].array[:, 1]).max() < 1e-12
    assert block['instrument']['Points'][0]['DealType'] == 'FRADeal'


def test_the_context_keeps_its_base_calibration_tape():
    """A run re-solves every quote it moves, and the config it leaves holds the BASE's calibration
    tape - the quote leaves and the connected curves a quote-space Greeks run reads - as a fresh
    bootstrap of the same document writes it, to the bit.

    Killing mutation: the tape not restored after a re-solve - the quote leaves read a basis point
    off and the curves 1.1e-4.
    """
    prices = blocks()
    for block in prices.values():
        block['instrument']['Quote_Sensitivity'] = 'Yes'
    after = world(prices=copy.deepcopy(prices))
    after.run_job()
    reference = world(prices=copy.deepcopy(prices)).current_cfg
    reference.bootstrap(wanted=set(reference.factor_universe()['resolved']))
    config = after.current_cfg
    assert set(config.quote_leaves) == set(reference.quote_leaves)
    for name, (descriptors, leaf) in reference.quote_leaves.items():
        assert config.quote_leaves[name][0] == descriptors
        assert np.array_equal(config.quote_leaves[name][1].detach().numpy(), leaf.detach().numpy())
    assert set(config.calibrated_factors) == set(reference.calibrated_factors)
    for factor, theta in reference.calibrated_factors.items():
        assert np.array_equal(config.calibrated_factors[factor].detach().numpy(),
                              theta.detach().numpy()), factor
