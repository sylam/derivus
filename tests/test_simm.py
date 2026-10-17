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

#: An index autocall whose coupon a month behind the base date printed a tenth over its trigger,
#: and the index it reads.
CALLED = {'Object': 'QEDI_CustomAutoCallSwap', 'Reference': 'AC', 'Currency': 'USD',
          'Payoff_Currency': 'USD', 'Equity': 'EQ', 'Dividends': 'EQ', 'Discount_Rate': 'USD-OIS',
          'Equity_Volatility': 'EQ', 'Buy_Sell': 'Buy', 'Strike_Price': 100.0, 'Units': 10.0,
          'Barrier': 0.0, 'Expiry_Date': BASE + pd.DateOffset(years=1),
          'Price_Fixing': [[BASE - pd.DateOffset(days=33), 110.0],
                           [BASE + pd.DateOffset(days=362), 0.0]],
          'Autocall_Coupons': [[BASE - pd.DateOffset(days=30), 0.04],
                               [BASE + pd.DateOffset(years=1), 0.04]],
          'Autocall_Thresholds': [[BASE - pd.DateOffset(days=30), 1.0],
                                  [BASE + pd.DateOffset(years=1), 1.0]]}
INDEX = {
    'EquityPrice.EQ': {'Spot': 100.0, 'Currency': 'USD', 'Interest_Rate': 'USD-OIS', 'Issuer': '',
                       'Respect_Default': 'No', 'Jump_Level': 0.0},
    'DividendRate.EQ': {'Currency': 'USD', 'Floor': None,
                        'Curve': utils.Curve([], [[0.0, 0.01], [5.0, 0.01]])},
    'EquityPriceVol.EQ': {'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness',
                          'Surface': utils.Curve([], [[1.0, 2.0, 0.25]])}}

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


def job(calculation=None, prices=None, option=None, configuration=None, system=None,
          extra=None, deals=(), book=None):
    """The job document of the swap, the option and `deals` under one netting set - or under `book`
    the trades each set reference names, under that set - on the true curves, with `prices` - the
    generated quote set unless given - and the curve family configured, in wire form."""
    prices = blocks() if prices is None else prices
    _, swap = benchmark(blocks())
    trades = [swap, dict(OPTION, **(option or {}))] + list(deals)
    book = {'NS': [deal['Reference'] for deal in trades]} if book is None else book
    document = {'Calc': {
        'Calculation': dict(SIMM, **(calculation or {})),
        'Deals': {'Reference': 'simm', 'Deals': {'Children': [{
            'Instrument': {'.Deal': {'Object': 'NettingCollateralSet', 'Reference': reference,
                                     'Netted': 'True', 'Collateralized': 'False',
                                     'Post_Regulations': 'SEC', 'Collect_Regulations': 'SEC'}},
            'Children': [{'Instrument': {'.Deal': deal}} for deal in trades
                         if deal['Reference'] in held]} for reference, held in book.items()]}},
        'MergeMarketData': {'MarketDataFile': '', 'ExplicitMarketData': {
            'System Parameters': dict({'Base_Currency': 'USD', 'Base_Date': BASE}, **(system or {})),
            'Valuation Configuration': {}, 'Price Factors': dict(factors(), **(extra or {})),
            'Market Prices': prices,
            'Bootstrapper Configuration': {'InterestRate': {'Prices': 'InterestRate'}}
            if configuration is None else configuration}}}}
    return json.loads(json.dumps(document, cls=CustomJsonEncoder))


def world(*args, **kwargs):
    """A context loaded with the `job` of `args` and `kwargs`."""
    context = derivus.Context()
    context.load_json((json.dumps(job(*args, **kwargs)), 'simm'), compress=False)
    return context


def valued(context):
    """`{reference: value}` off one base valuation of the loaded job."""
    frame = context.run_job()[1]['Results']['mtm']
    return {reference: value for reference, value in zip(frame['Reference'], frame['Value'])}


VALUATION = {'Object': 'BaseValuation'}


def identical(mine, theirs):
    """Two runs' CRIF and trades report, `{table: frame}`, the same to the bit, row for row."""
    for table in ('CRIF', 'Trades'):
        floats = [column for column in ('Amount', 'AmountUSD', 'Value') if column in mine[table]]
        assert mine[table].drop(columns=floats).equals(theirs[table].drop(columns=floats)), table
        for column in floats:
            assert [x.hex() for x in mine[table][column]] == [
                x.hex() for x in theirs[table][column]], (table, column)


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
    basis point on that quote moves the curve to price the benchmark at par a basis point up and
    the swap is worth `N A' 1bp`, `A'` its fixed leg's annuity on the re-solved curve - the same
    quote set with that quote a basis point up, bootstrapped, and read off its knots, every coupon
    landing on one. Each other quote leaves it at par, so its rate rows sum to that figure but for
    the second order a ride on the calibration Jacobian drops, 1.8e-4 of it.

    Killing mutation: the quote moved by the shift taken as a decimal, 1e-4 on a quote in percent -
    the rows sum to a hundredth of the hand figure.
    """
    output = serial[1]
    prices = blocks()
    position, _ = benchmark(prices)
    prices['InterestRatePrices.USD-OIS']['instrument']['Points'][position][
        'Quoted_Market_Value'] += 0.01
    bumped = world(VALUATION, prices=prices)
    bumped.bootstrap()
    knots = bumped.current_cfg.params['Price Factors']['InterestRate.USD-OIS']['Curve'].array
    coupons = [BASE + pd.DateOffset(years=y) for y in range(6)]
    annuity = sum((end - start).days / 365.0 * np.exp(-knots[knots[:, 0] == (
        end - BASE).days / 365.0, 1][0] * (end - BASE).days / 365.0)
                  for start, end in zip(coupons[:-1], coupons[1:]))
    trades = output['Results']['Trades']
    assert abs(trades[trades['TradeID'] == 'SWAP']['Value'].item()) < 1e-6
    assert sum(rows(output, 'SWAP', 'Risk_IRCurve').values()) == pytest.approx(
        NOTIONAL * annuity * 1e-4, rel=2e-4)


def test_the_crif_is_isdas_format(serial, tmp_path):
    """THE FILE THE MARGIN CALCULATION READS. Its columns are the ones the margin calculation's
    reader consumes, one row per trade and coordinates, the trade id the booked trade's reference;
    a rate row lands on a vertex of a curve's sub-curve, an FX row on a currency other than the
    calculation's, a vega row on a pair and a vertex - the twelve the parameters file names; and it
    reads back tab separated to the bit.

    Killing mutation: the vertex dropped from a row's key (`label or ...` read as the mapping's own
    Label1) - every rate row of a curve folds into one carrying no vertex.
    """
    calc, output = serial
    assert [label for label, _ in calc.vertices] == [
        '2w', '1m', '3m', '6m', '1y', '2y', '3y', '5y', '10y', '15y', '20y', '30y']
    assert [years for _, years in calc.vertices] == pytest.approx(
        [14 / 365.0, 1 / 12.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0], rel=1e-15)
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
    """Two workers spawned on contexts of their own, the book valued here and its moves dealt
    between them, against the serial run in-process: the same CRIF to the bit, row for row, and the
    same trades report.

    Killing mutation: a worker valuing its moves on its context uncalibrated - the first quote
    move finds no calibration to ride and refuses `CalibrationStale` naming the USD curve.
    """
    identical(world({'Workers': 2}).run_job()[1]['Results'], serial[1]['Results'])


@pytest.mark.parametrize('book', [{'NS': ('SWAP', 'FXO')}, {'NS': ('SWAP',), 'NT': ('FXO',)}])
def test_a_netting_set_s_crif_is_its_trades_own_to_the_bit(book):
    """A BOOK IS VALUED PER MOVE. The swap and the option, under one netting set or each under its
    own, are valued as one book once per move, 32 valuations where valuing each trade per move took
    43 (2.1 a CRIF row of 15 against 2.9), and write the CRIF rows and the trades report the two
    write each alone in a one-trade document, to the bit. That holds because both price in closed
    form and says nothing of a Monte Carlo trade: a base valuation draws a Monte Carlo pricer's
    paths off one stream in booking order, so such a trade booked after another is valued as the
    book values it, not as it is valued alone, which the Monte Carlo gate below measures.

    Killing mutations: a trade read off the structure of the next trade the valuation holds - the
    swap valued at the option's 66,209.81 and missing its 5y row of 443.41; a book's trades booked
    under its first set - on the two-set book the option, found under no set of its own, values to
    nothing and writes none of the 10 rows it writes alone.
    """
    output = world(book=book).run_job()[1]['Results']
    alone = [world(book={reference: (trade,)}).run_job()[1]['Results']
             for reference, held in book.items() for trade in held]
    identical(output, {table: pd.concat([run[table] for run in alone], ignore_index=True)
                       for table in ('CRIF', 'Trades')})


def test_a_crif_s_trades_revalued_off_their_own_documents_merge_back_to_the_bit(serial, tmp_path):
    """THE EXTRACT UTILITY. `experiments/simm_from_crif.py` reads this world as a source system
    exports it - each trade in a one-trade document of its own under a set named for nothing, and a
    CRIF of the module's run in the source's layout stating a counterparty and regulations of its
    own - books one netting set of the counterparty's trades off the union of the documents' market
    data, runs it, and writes the module's CRIF to the bit, the counterparty and regulations the
    source's, and merged back into the source CRIF the source's own bytes.

    Killing mutation: the set's regulations taken from the documents' own sets rather than the
    source's rows - every merged row posts under SEC where the source states CFTC.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    'experiments'))
    import simm_from_crif

    stated = {'Counterparty': 'CP', 'PostRegulations': 'CFTC'}
    crif = serial[1]['Results']['CRIF'].assign(**stated)
    folder, output = tmp_path / 'documents', tmp_path / 'output'
    folder.mkdir()
    for trade in crif['TradeID'].unique():
        with open(folder / '{}.json'.format(trade), 'wt') as f:
            json.dump(job(book={'SET-' + trade: (trade,)}), f)
    source = crif.assign(IMModel='SIMM', AmountCurrency='USD', CSANumber='CSA', PortfolioID='BOOK',
                         ValuationDate='{:%Y-%m-%d}'.format(BASE), EndDate='')[
        ['IMModel', 'ProductClass', 'RiskType', 'Qualifier', 'Bucket', 'Label1', 'Label2', 'Amount',
         'AmountCurrency', 'AmountUSD', 'Counterparty', 'CSANumber', 'TradeID', 'PortfolioID',
         'CollectRegulations', 'PostRegulations', 'ValuationDate', 'EndDate']]
    source.to_csv(tmp_path / 'source.tsv', sep='\t', index=False)
    simm_from_crif.main([str(tmp_path / 'source.tsv'), str(folder), '--mapping', MAPPING,
                         '--parameters', PARAMETERS, '--output', str(output)])
    date = '{:%Y-%m-%d}'.format(BASE)
    ran = pd.read_csv(output / 'CRIF_{}.tsv'.format(date), sep='\t', dtype={'Bucket': str},
                      keep_default_na=False, float_precision='round_trip')
    identical({'CRIF': ran, 'Trades': serial[1]['Results']['Trades']},
              {'CRIF': crif, 'Trades': serial[1]['Results']['Trades']})
    assert (output / 'CRIF_REVAL_{}.tsv'.format(date)).read_bytes() == (
        tmp_path / 'source.tsv').read_bytes()


#: Two index autocalls, observed a half year and a year out, the second on an index of its own.
AUTOCALLS = [dict(CALLED, Reference=reference, Equity=index, Dividends=index,
                  Equity_Volatility=index,
                  Price_Fixing=[[BASE + pd.DateOffset(days=days - 3), 0.0] for days in (182, 365)],
                  Autocall_Coupons=[[BASE + pd.DateOffset(days=days), 0.04] for days in (182, 365)],
                  Autocall_Thresholds=[[BASE + pd.DateOffset(days=days), 1.0]
                                       for days in (182, 365)])
             for reference, index in (('AC1', 'EQ'), ('AC2', 'EQ2'))]
INDICES = dict(INDEX, **{name + '2': dict(block) for name, block in INDEX.items()})


def test_monte_carlo_trades_draw_one_stream_through_every_valuation():
    """Two autocalls a Monte Carlo pricer values, under one set, the second on an index the first
    does not read: every valuation of the run books both in one order and a move changes no draw
    count, so each draws the same paths through the base and every move. The second's equity row
    is the difference of two base valuations of the book, its index's spot a percent up and as it
    stands, to 1e-9, and the run with its moves dealt over two workers writes the serial run's CRIF
    and trades report to the bit. A base valuation draws its paths off one stream in booking order,
    so the second, drawing after the first, is valued as the book values it: at 256 inner paths its
    base sits 5.5e-3 off its own one-trade document's and its rows up to 4.7e-3 of its largest,
    where the first writes its own document's to the bit.

    Killing mutation: a move valuing only the trades it writes rows for - the second autocall draws
    from the seed under its own index's moves and after the first at the base, and its equity row
    reads 8.807e-3 against the difference of the book's valuations, 7.488e-3.
    """
    calculation = {'MCMC_Simulations': 256}
    serial, sharded = (world(dict(calculation, **workers), extra=INDICES, deals=AUTOCALLS,
                             book={'NS': ('AC1', 'AC2')}).run_job()[1]['Results']
                       for workers in ({}, {'Workers': 2}))
    identical(sharded, serial)

    def worth(spot):
        moved = dict(INDICES['EquityPrice.EQ2'], Spot=spot)
        return valued(world(dict(VALUATION, **calculation), configuration={}, deals=AUTOCALLS,
                            extra=dict(INDICES, **{'EquityPrice.EQ2': moved}),
                            book={'NS': ('AC1', 'AC2')}))['AC2']

    spot = INDICES['EquityPrice.EQ2']['Spot']
    crif = serial['CRIF']
    row = crif[(crif['TradeID'] == 'AC2') & (crif['RiskType'] == 'Risk_Equity')]['Amount'].item()
    assert row == pytest.approx(worth(spot * 1.01) - worth(spot), rel=1e-9)


def test_an_autocall_a_past_print_called_writes_no_rows():
    """An autocall whose coupon a month behind the base date printed over its trigger was called
    then, and every valuation of it is skipped as expired - so it writes no CRIF row: the trade is
    reported unvalued in the engine's own sentence, and the swap and the option beside it write
    theirs.

    Killing mutation: the compile-time call test removed - the note is valued live and writes its
    equity, vega and rate rows.
    """
    output = world(deals=[CALLED], extra=INDEX).run_job()[1]
    crif, trades = output['Results']['CRIF'], output['Results']['Trades']
    assert set(crif['TradeID']) == {'SWAP', 'FXO'}
    called = trades[trades['TradeID'] == 'AC']
    assert called['Written'].item() == 0 and np.isnan(called['Value'].item())
    assert 'AC: autocalled on {:%Y-%m-%d}'.format(
        BASE - pd.DateOffset(days=30)) in called['Skipped'].item()


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


def test_a_quote_set_no_configured_family_solves_is_refused_by_name():
    """The condition reads what a CONFIGURED family claims, not what the section carries: with the
    quote sets present and no curve entry in `Bootstrapper Configuration`, nothing solves them and
    the curves stand as pasted nodes - refused by name, where the IR risk would otherwise vanish
    from a CRIF that still wrote its FX and vega rows.

    Killing mutation: the condition reading the section's own keys - the run passes the walk and
    refuses later, at the first quote move, which finds no calibration to ride, never naming the
    condition.
    """
    with pytest.raises(ValueError, match='a configured family claims bootstraps') as refusal:
        world(configuration={}).run_job()
    assert 'InterestRate.USD-OIS' in str(refusal.value)


def test_a_quote_move_rides_the_jacobian_and_is_the_re_solve_to_first_order(serial):
    """A BASIS POINT RIDES THE CALIBRATION JACOBIAN. The run's bootstrap solves each coupled set
    once and publishes `theta*` and `J = dtheta/dq`, and a quote's move is `theta* + J (q - q0)`.
    Bootstrapping the block with that quote a basis point up and revaluing the book, quote by quote,
    writes the same rate rows but for the second order the re-solve carries and the ride drops:
    every row within 2e-4 of its trade's largest, the swap's 5y row departing most, by 1.67e-4 of
    itself - 8.4e-5 at half the shift and 3.3e-4 at twice it, the order of the bump's own
    one-sided error. A block declaring `Quote_Propagation: Linear` is valued off the curves the move
    writes, its rows those of one declaring `No` to the bit. The serial run of this world takes
    1.1 seconds where re-solving each quote took 4.5, and the sharded one 10 where it took 13.

    Killing mutations: the Jacobian's sign flipped - every row the negative of the re-solve's, the
    swap's 5y -443.64 against 443.48; the quote moved one place along - the rows land on the next
    quote's vertices, the swap's PV01 halved onto 3y; the move added in basis points to a quote in
    percent - the swap's 5y row 43,203 against 443.48; the run's declaration of no ride not held
    through its valuations - every valuation rides the quotes standing and no rate row is written.
    """
    calc, output = serial
    trades = ('SWAP', 'FXO')

    def rates(output):
        crif = output['Results']['CRIF']
        found = crif[crif['RiskType'] == 'Risk_IRCurve']
        return dict(zip(zip(found['TradeID'], found['Qualifier'], found['Label1']), found['Amount']))

    def bootstrapped(prices):
        context = world(VALUATION, prices=prices)
        context.bootstrap()
        return valued(context)

    ridden, prices = rates(output), blocks()
    for block in prices.values():
        block['instrument']['Quote_Propagation'] = 'Linear'
    linear = rates(world(prices=prices).run_job()[1])
    assert {key: amount.hex() for key, amount in linear.items()} == {
        key: amount.hex() for key, amount in ridden.items()}
    base, solved = bootstrapped(blocks()), {}
    for block, qualifier, dates in (('InterestRatePrices.USD-OIS', 'USD', USD_DATES),
                                    ('InterestRatePrices.ZAR-JIBAR', 'ZAR', ZAR_DATES)):
        for position, date in enumerate(dates):
            prices = blocks()
            prices[block]['instrument']['Points'][position]['Quoted_Market_Value'] += 0.01
            moved = bootstrapped(prices)
            for trade in trades:
                for label, weight in utils.vertex_weights((date - BASE).days / 365.0,
                                                          calc.vertices):
                    key = (trade, qualifier, label)
                    solved[key] = solved.get(key, 0.0) + weight * (moved[trade] - base[trade])
    for trade in trades:
        scale = max(abs(amount) for key, amount in solved.items() if key[0] == trade)
        for key in {key for key in set(ridden) | set(solved) if key[0] == trade}:
            assert abs(ridden.get(key, 0.0) - solved.get(key, 0.0)) < 2e-4 * scale, key


def test_a_worker_that_raises_surfaces_in_the_parent():
    """A sharded run raises what its serial run raises, within a bound and never hanging: a worker's
    error crosses the queue as its answer and the parent re-raises it, the other worker stopped.
    The error is the margin's own - a spot moved to nothing values the option at NaN, off which it
    refuses by name to read a sensitivity.

    Killing mutation: the worker's error not put on the queue - the parent names a worker dead
    without answering, never the error the worker raised.
    """
    broken = {'Spot_Shift': -100.0}
    with pytest.raises(ValueError, match='values the trade to nan') as alone:
        world(broken).run_job()
    found = {}

    def sharded():
        try:
            world(dict(broken, Workers=2)).run_job()
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
    one at every vertex expiry it lacks, read off its own interpolation, each vertex alone moved by
    a point is a tent, so a 3Y option's vega is a 3y row (a 5y sliver, its expiry a day past 3y),
    a 1M option's a 1m row, a 2.5Y option's split between 2y and 3y - linear in years of 365 days
    between the vertices either side. Each row is the move of the option's own vol by its tent
    there - as a base valuation of the same book on the surface moved in parallel by that much
    values it - and the rows sum to the parallel point's vega but for that move's convexity.

    Killing mutations: the surface left on its own pillars - the vertex tents land on 6m, 1y and
    2y, a 3Y option's vega reading on 2y and a 1M option's on 6m; the two vertex weights swapped,
    the nearer vertex taking the smaller share.
    """
    expiry = BASE + pd.DateOffset(months=months)
    vega = rows(world(option={'Expiry_Date': expiry}).run_job()[1], 'FXO', 'Risk_FXVol')
    surface = factors()['FXVol.USD.ZAR']

    def worth(size):
        moved = dict(surface, Surface=utils.Curve(
            surface['Surface'].meta, surface['Surface'].array + [0.0, 0.0, size]))
        return valued(world(VALUATION, option={'Expiry_Date': expiry}, configuration={},
                            extra={'FXVol.USD.ZAR': moved}))['FXO']

    base = worth(0.0)
    assert set(vega) == set(split)
    for label, weight in split.items():
        assert vega[label] == pytest.approx(worth(0.01 * weight) - base, rel=1e-6), label
    assert sum(vega.values()) == pytest.approx(worth(0.01) - base, rel=5e-3)


def parametric(kind, tenors, **pillars):
    """A USDZAR surface of `Surface_Type` `kind`, a pillar per tenor about the spot: `pillars` names
    each other parameter per pillar - on an SVI slice `a`, `b`, `rho`, `m` and `sigma`, the variance
    at log-moneyness x `a + b(rho(x - m) + sqrt((x - m)^2 + sigma^2))`."""
    return dict({'Surface_Type': kind, 'Moneyness_Rule': 'Sticky_Moneyness',
                 'ATM_Ref': utils.Curve([], [[t, SPOT] for t in tenors])}, **{
        name: utils.Curve([], [[t, value] for t, value in zip(tenors, values)])
        for name, values in pillars.items()})


@pytest.mark.parametrize('days, label', [(14, '2w'), (365, '1y'), (1095, '3y')])
def test_a_flat_svi_surface_moves_as_the_explicit_surface_stating_its_vol(days, label):
    """An SVI surface with no smile, `b` 0, at one vol on slices at 6m, 1y and 2y is the Explicit
    surface stating that vol, and a point of vega moves the two alike: a vanilla expiring on a
    vertex - before the first slice, on one, past the last - writes one vega row on that vertex on
    either, the same to 1e-12, the option priced in closed form. Each slice's `a` moves by
    what moves its vol by the point, and the surface gains a slice at each vertex past either
    end, where it reads its edge slice.

    Killing mutations: the level shift dropped - the SVI writes no vega row; no slice added past
    the ends - the 2w and 3y options' vega lands on 6m and 2y.
    """
    tenors, vol = (0.5, 1.0, 2.0), 0.15
    flat = parametric('SVI', tenors, a=[vol ** 2] * 3, b=[0.0] * 3, rho=[0.0] * 3, m=[0.0] * 3,
                      sigma=[0.1] * 3)
    explicit = dict(factors()['FXVol.USD.ZAR'], Surface=utils.Curve(
        [], [[m, t, vol] for m in (0.8, 1.0, 1.2) for t in tenors]))
    option = {'Expiry_Date': BASE + pd.DateOffset(days=days)}
    vega = [rows(world(option=option, extra={'FXVol.USD.ZAR': surface}).run_job()[1], 'FXO',
                 'Risk_FXVol') for surface in (flat, explicit)]
    assert set(vega[0]) == set(vega[1]) == {label}, vega
    assert vega[0][label] == pytest.approx(vega[1][label], rel=1e-12)


def test_an_svi_vertex_moves_the_slices_its_tent_reaches_off_the_standing_surface():
    """Between two slices an SVI surface blends their variances, which no slice is, so a vertex
    there inserts none: its point moves the `a` of each slice its tent reaches by what moves that
    slice's at-the-money vol by the point times the tent, and the row is taken off the surface as
    it stands. On a smile sliced at 0.04, 0.4, 0.8 and 1.5 years, a 0.6-year call struck over the
    forward writes 3m, 6m and 1y rows, each a base valuation of the same book on the surface so
    moved less its standing value, to 1e-9.

    Killing mutation: a slice inserted at the 6m vertex off each parameter's own interpolation,
    as a Skew surface gains one - the call's base moves by 0.24% and its vega lands on 6m and 1y
    alone, the 6m row 1,656 against 1,023.
    """
    tenors = (0.04, 0.4, 0.8, 1.5)
    smile = parametric('SVI', tenors, a=[0.020, 0.018, 0.017, 0.016],
                       b=[0.08, 0.06, 0.05, 0.04], rho=[-0.5, -0.4, -0.35, -0.3],
                       m=[0.02, 0.03, 0.04, 0.05], sigma=[0.10, 0.15, 0.20, 0.25])
    option = {'Expiry_Date': BASE + pd.DateOffset(days=219), 'Strike_Price': 21.0}
    vega = rows(world(option=option, extra={'FXVol.USD.ZAR': smile}).run_job()[1], 'FXO',
                'Risk_FXVol')

    def worth(a):
        moved = dict(smile, a=utils.Curve([], list(zip(tenors, a))))
        return valued(world(VALUATION, option=option, configuration={},
                            extra={'FXVol.USD.ZAR': moved}))['FXO']

    a, b, rho, m, sigma = (smile[name].array[:, 1] for name in ('a', 'b', 'rho', 'm', 'sigma'))
    atm = np.sqrt(a + b * (rho * -m + np.sqrt(m ** 2 + sigma ** 2)))
    vertices = list(zip(('2w', '1m', '3m', '6m', '1y', '2y'),
                        (14 / 365.0, 1 / 12.0, 0.25, 0.5, 1.0, 2.0)))
    base = worth(a)
    assert set(vega) == {'3m', '6m', '1y'}, vega
    for label, amount in vega.items():
        point = 0.01 * np.array([dict(utils.vertex_weights(t, vertices)).get(label, 0.0)
                                 for t in tenors])
        assert amount == pytest.approx(worth((atm + point) ** 2 - atm ** 2 + a) - base,
                                       rel=1e-9), label


#: A USDZAR smile on four pillars, steeper at the short end: at log-moneyness x the `ATM_Vol` plus
#: `x(s + L x)` down to the cut-off `C` and `x(s + R x)` up to `D`, a quadratic wing beyond each
#: flattening `lam` and `rho` of the cut-off further out.
SKEW = {'ATM_Vol': [0.16, 0.15, 0.145, 0.14], 's': [0.10, 0.08, 0.06, 0.05],
        'L': [0.5, 0.4, 0.3, 0.3], 'R': [0.8, 0.6, 0.5, 0.4], 'C': [-0.06, -0.10, -0.14, -0.20],
        'D': [0.06, 0.10, 0.14, 0.20], 'lam': [0.5] * 4, 'rho': [0.8, 0.6, 0.5, 0.5]}


@pytest.mark.parametrize('tenors, labels', [((0.04, 0.4, 0.8, 1.5), {'3m', '6m', '1y'}),
                                            ((0.25, 0.5, 1.0, 2.0), {'6m', '1y'})])
def test_a_skew_vertex_moves_the_pillars_its_tent_reaches_off_the_standing_surface(tenors, labels):
    """Between two pillars a Skew surface blends their vols at the strike, which a pillar
    interpolated in its parameters reproduces in the centre and not in the wings, so a vertex there
    inserts none: its point moves the `ATM_Vol` of each pillar its tent reaches, and with it that
    pillar's whole smile, by the point times the tent, and the row is taken off the surface as it
    stands. A 0.6-year call struck 13% over the forward, in the right wing of a 0.4-year pillar,
    writes on pillars at 0.04, 0.4, 0.8 and 1.5 years 3m, 6m and 1y rows, each a base valuation of
    the same book on the surface so moved less its standing value, to 1e-9; on pillars on the 3m,
    6m, 1y and 2y vertices, which gain none inside the range under either rule, 6m and 1y rows,
    each the move of its own pillar alone, what they wrote before to the bit. The rows sum to the
    parallel point's move but for its convexity, 1.6% and 0.8% under it.

    Killing mutations: a pillar inserted at each vertex inside the range off each parameter's own
    interpolation, as before - the call's base moves by 0.64% and its vega lands on 6m and 1y
    alone, the 6m row 1,622 against 1,003; the two tent weights swapped - the 3m row 599 against
    398, and on the vertex pillars the vega lands on 3m and 6m.
    """
    smile = parametric('Skew', tenors, **SKEW)
    option = {'Expiry_Date': BASE + pd.DateOffset(days=219), 'Strike_Price': 21.0}
    vega = rows(world(option=option, extra={'FXVol.USD.ZAR': smile}).run_job()[1], 'FXO',
                'Risk_FXVol')

    def worth(atm):
        moved = dict(smile, ATM_Vol=utils.Curve([], list(zip(tenors, atm))))
        return valued(world(VALUATION, option=option, configuration={},
                            extra={'FXVol.USD.ZAR': moved}))['FXO']

    atm = np.array(SKEW['ATM_Vol'])
    vertices = {'2w': 14 / 365.0, '1m': 1 / 12.0, '3m': 0.25, '6m': 0.5, '1y': 1.0, '2y': 2.0}
    base = worth(atm)
    assert set(vega) == labels, vega
    for label, amount in vega.items():
        # the tent spelt apart from the engine's: the hat on the vertex grid, one at its vertex
        tent = np.interp(tenors, list(vertices.values()),
                         [float(label == vertex) for vertex in vertices])
        assert amount == pytest.approx(worth(atm + 0.01 * tent) - base, rel=1e-9), label
    assert sum(vega.values()) == pytest.approx(worth(atm + 0.01) - base, rel=2e-2)


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
    """A run rides every quote it moves and writes nothing, so the config it leaves holds the
    BASE's calibration tape - the quote leaves and the connected curves a quote-space Greeks run
    reads - and its curves as a fresh bootstrap of the same document writes them, to the bit.

    Killing mutations: a move re-solved through the config's own loop and its tape left as the
    re-solve wrote it - the quote leaves read a basis point off; the ride's curve written into the
    standing block, as `solve_set` writes one - the USD curve's 10Y node left 1.1bp up.
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
        name = utils.check_tuple_name(factor)
        assert np.array_equal(config.params['Price Factors'][name]['Curve'].array,
                              reference.params['Price Factors'][name]['Curve'].array), name
