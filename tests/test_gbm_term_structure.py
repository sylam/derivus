"""`GBMAssetPriceTSModelParameters`: a vol surface's ATM column as the instantaneous vol curve
`GBMAssetPriceTSModelImplied` integrates, built by a job document's bootstrap and read back as the
process reads it - piecewise linear between knots, flat outside, its square integrated exactly. And
the carry the process drifts an equity on: its dividend curve, in the universe of every credit
Monte Carlo simulating the equity whatever its deals read."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

import derivus
import test_declared_defaults as book
import test_position_scaling as trials
from derivus import utils
from derivus.config import CustomJsonEncoder


def built(expiries, atm):
    """The `Vol` knots a bootstrap writes off an equity surface whose ATM column is `atm`."""
    surface = [[m, t, v * (1.0 + abs(m - 1.0))] for t, v in zip(expiries, atm) for m in (0.8, 1.0, 1.2)]
    base = {'.Timestamp': '2026-01-15'}
    document = {'Calc': {
        'Calculation': {'Object': 'BaseValuation', 'Base_Date': base, 'Currency': 'USD'},
        'Deals': {'Reference': 'gbm', 'Deals': {'Children': []}},
        'MergeMarketData': {'MarketDataFile': '', 'ExplicitMarketData': {
            'System Parameters': {'Base_Currency': 'USD', 'Base_Date': base},
            'Price Factors': {'EquityPriceVol.EQ': {
                'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness',
                'Surface': utils.Curve([], surface)}},
            'Market Prices': {'GBMAssetPriceTSModelPrices.EQ': {
                'instrument': {'Asset_Price_Volatility': 'EQ', 'Quote_Sensitivity': 'Yes'}}},
            'Bootstrapper Configuration': {
                'GBMAssetPriceTSModelParameters': {'Prices': 'GBMAssetPriceTSModel'}}}}}}
    context = derivus.Context()
    context.load_json((json.dumps(document, cls=CustomJsonEncoder), 'gbm'))
    context.bootstrap()
    return context.current_cfg.params['Price Factors']['GBMAssetPriceTSModelParameters.EQ']['Vol'].array


def implied(knots, t):
    """The vol the process implies to `t`: `sqrt(V(t)/t)`, `V` the exact integral of the square."""
    grid = np.union1d([0.0, t], knots[knots[:, 0] < t, 0])
    s = np.interp(grid, knots[:, 0], knots[:, 1])
    return np.sqrt(np.sum(np.diff(grid) / 3.0 * (s[:-1] ** 2 + s[:-1] * s[1:] + s[1:] ** 2)) / t)


@pytest.mark.parametrize('atm', [[0.10, 0.30, 0.25, 0.26], [0.10, 0.20, 0.25, 0.30],
                                 [0.30, 0.25, 0.22, 0.20]], ids=['humped', 'rising', 'falling'])
def test_the_curve_reprices_every_atm_vol_it_was_built_from(atm):
    """Every column whose total variance rises - the humped one included, which a curve knotted
    only at the expiries cannot fit, its 1y vol forcing more variance onto the 2y than the 2y
    quote holds - is matched at every expiry.

    Killing mutations: the column's average vol written where the process reads an instantaneous
    one (the rising column's 1y misses by 5.9 vol points); the curve knotted at the expiries alone
    (the humped column's 2y is repaired to 30.4%)."""
    expiries = [0.25, 1.0, 2.0, 5.0]
    knots = built(expiries, atm)
    assert [implied(knots, t) for t in expiries] == pytest.approx(atm, abs=1e-14)


def test_a_falling_total_variance_is_repaired_to_a_zero_vol():
    """30%, 20%, 10% at 6m, 1y, 2y: total variance falls, so each later step is raised to the least
    variance it can reach - its vol zero after a day's ramp - and the curve misses those quotes.

    Killing mutation: the repair dropped - the walk writes negative vols."""
    knots = built([0.5, 1.0, 2.0], [0.30, 0.20, 0.10])
    assert knots[1:, 1].tolist() == [0.0] * 4
    assert implied(knots, 2.0) == pytest.approx(0.30 * np.sqrt((0.5 + 1.0 / 365.0 / 3.0) / 2.0), rel=1e-12)


@pytest.mark.parametrize('reference,stated', [('EQO', {'Dividends': 'OWN'}), ('EQS', {})],
                         ids=['option stating its dividends', 'holding'])
def test_a_simulated_equity_holds_the_dividend_curve_its_process_reads(reference, stated):
    """The declared-defaults `EQ` simulated under `GBMAssetPriceTSModelImplied`, which drifts it on
    its own dividend curve, in a credit Monte Carlo whose one deal is the trial option valued off the
    `DividendRate.OWN` it states, or the trial holding, which names no dividend curve: the universe
    holds `DividendRate.EQ` beside what the deal names, the booking verdict is clean, and the run
    values the deal.

    Killing mutation: the process's `reads` dropped - neither deal registers `DividendRate.EQ`, and
    the run stops at "Cannot find DividendRate.EQ".
    """
    deal = next(dict(deal, **stated) for family in trials.FAMILIES.values()
                for deal in family.DEALS if deal['Reference'] == reference)
    job = json.loads(json.dumps(book.book([deal]), cls=CustomJsonEncoder))
    job['Calc']['Calculation'] = dict(
        job['Calc']['Calculation'], Object='CreditMonteCarlo', Time_Grid='0d 3m(3m)',
        Batch_Size=256, Deflation_Interest_Rate='USD')
    market = job['Calc']['MergeMarketData']['ExplicitMarketData']
    market['Price Factors'].update({
        'DividendRate.OWN': {'Floor': None, 'Currency': 'USD', 'Curve': {'.Curve': {
            'meta': [], 'data': [[0.0, 0.04], [5.0, 0.04]]}}},
        'GBMAssetPriceTSModelParameters.EQ': {
            'Quanto_FX_Volatility': None, 'Quanto_FX_Correlation': 0.0,
            'Vol': {'.Curve': {'meta': [], 'data': [[0.0, 0.2], [5.0, 0.2]]}}}})
    market['Model Configuration'] = {'.ModelParams': {
        'modeldefaults': {'EquityPrice': 'GBMAssetPriceTSModelImplied'}, 'modelfilters': {}}}
    context = derivus.Context().load_json((json.dumps(job), 'simulated'))
    named = {'DividendRate.EQ', 'DividendRate.' + stated.get('Dividends', 'EQ')}
    assert named <= set(context.describe()['factors']['resolved']), context.describe()
    assert context.validate() == {'deals': {}, 'factors': [], 'calculation': []}, context.validate()
    _, out = context.run_job()
    assert out['Stats'].get('Deals Skipped', 0) == 0 and len(out['Results']['mtm']), out['Stats']
