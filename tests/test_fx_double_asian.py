"""The FX double Asian is a spread option on two discretely sampled averages: each leg's samples still
to fix two-moment matched to a lognormal, the pair by Bjerksund-Stensland, and a leg whose samples
are all fixed a constant, its realised average.

The reference is a Monte Carlo off the deal's own terms: EURUSD a GBM at the surface's 15% with USD
at 4% and EUR at 2%, sampled at the twelve month-ends to the expiry on 1,000,000 antithetic paths,
a print held where the deal states one. The moment match is an approximation of that law: against
16,000,000 paths it sits inside 0.04% of every case below.
"""

import json
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd

import derivus
import test_declared_defaults as book
import trial_fx
from derivus import utils
from derivus.config import CustomJsonEncoder
from test_position_scaling import document, marks

B, E = book.WORLD_BASE, book.WORLD_EXPIRY
#: four fixings printed before the base date
PRINTS = [(-4, 1.22), (-3, 1.26), (-2, 1.23), (-1, 1.27)]


def month(n):
    return B + pd.DateOffset(months=n)


def option(first, second):
    """`trial_fx`'s double Asian - a call on its first average less its second, struck at 0.01 -
    sampling `first` and `second`, each `[(month, print)]` with no print for a sample to come."""
    def table(samples):
        return [[month(m), price or 0.0, 1.0] for m, price in samples]

    deal = dict(next(d for d in trial_fx.DEALS if d['Reference'] == 'FXDASN'),
                Sampling_Data_1=table(first), Sampling_Data_2=table(second))
    return document(SimpleNamespace(DEALS=[deal], FACTORS={}, CONFIGURATION={}))


def monte_carlo():
    """EURUSD at the twelve month-ends on 1,000,000 antithetic paths, seed 11, and the discount from
    expiry: `(spot, df)`."""
    rng = np.random.default_rng(11)
    t = np.array([(month(m) - B).days for m in range(1, 13)]) / 365.0
    steps = np.diff(np.concatenate([[0.0], t]))
    z = rng.standard_normal((500_000, 12))
    z = np.concatenate([z, -z])
    rd, vol = book.DISCOUNT[0], book.SIGMA
    spot = book.X0 * np.exp(np.cumsum(
        (rd - book.R_EUR - 0.5 * vol * vol) * steps + vol * np.sqrt(steps) * z, axis=1))
    return spot, math.exp(-rd * (E - B).days / 365.0)


def test_a_double_asian_is_its_monte_carlo_seasoned_and_with_either_leg_fixed():
    """The base valuation of the trialled option (the last four months' average over the first
    four's), of the same option seasoned - its second average's first fixing a month behind the base
    date, printed at 1.24 - and of each leg in turn fully fixed at four prints averaging 1.245, sits
    within three standard errors of the Monte Carlo of its payoff.

    Reference, by hand: `1000 max(A1 - A2 - 0.01, 0) DF(expiry)` per path, `A` a leg's average of
    its prints and its simulated month-ends.

    The seasoned leg's two fixings weighted against its four weights skips the deal.

    Killing mutation: a fully fixed leg sent through the spread formula, whose empty moment is
    `log 0` and marks NaN.
    """
    spot, df = monte_carlo()
    late, early = spot[:, 8:12].mean(axis=1), spot[:, 0:4].mean(axis=1)
    seasoned = (1.24 + spot[:, 0:3].sum(axis=1)) / 4.0
    printed = np.mean([price for _, price in PRINTS])
    last = [(m, None) for m in (9, 10, 11, 12)]
    for first, second, a1, a2 in (
            (last, [(m, None) for m in (1, 2, 3, 4)], late, early),
            (last, [(-1, 1.24), (1, None), (2, None), (3, None)], late, seasoned),
            (last, PRINTS, late, printed),
            (PRINTS, last, printed, late)):
        paid = 1000.0 * np.maximum(a1 - a2 - 0.01, 0.0) * df
        pairs = 0.5 * (paid[:500_000] + paid[500_000:])
        reference, error = pairs.mean(), pairs.std(ddof=1) / math.sqrt(pairs.size)
        mark = float.fromhex(marks(option(first, second))['FXDASN'])
        assert abs(mark - reference) <= 3.0 * error, (first, second, mark, reference, error)


def test_a_fixed_leg_leaving_a_strike_at_or_below_zero_is_worth_its_forward():
    """The first average half fixed at 2.0 and the second wholly at 0.5 leave the call a strike of
    -0.49 on the first's two samples to come, so it is exercised on every path and is worth the
    discounted forward of the spread.

    Reference, by hand: `1000 DF(expiry) (0.25 (2 + 2 + F_11 + F_12) - 0.5 - 0.01)`, `F_m` the
    EURUSD forward at month-end m.

    Killing mutation: the exercised-for-certain arm taken at the strike clamped to 1e-5, which
    drops the 0.49.
    """
    forward = [book.X0 * math.exp((book.DISCOUNT[0] - book.R_EUR) * (month(m) - B).days / 365.0)
               for m in (11, 12)]
    reference = 1000.0 * math.exp(-book.DISCOUNT[0] * (E - B).days / 365.0) * (
        0.25 * (2.0 + 2.0 + sum(forward)) - 0.5 - 0.01)
    job = option([(-2, 2.0), (-1, 2.0), (11, None), (12, None)], [(m, 0.5) for m, _ in PRINTS])
    mark = float.fromhex(marks(job)['FXDASN'])
    assert abs(mark - reference) <= 1e-12 * reference, (mark, reference)


def test_a_double_asian_prices_every_row_of_a_credit_monte_carlo():
    """The trialled option alone in a netting set, on a quarterly and on a monthly grid with EURUSD
    a GBM at 15%, and on the quarterly grid with EURUSD static and the USD curve a Hull-White, there
    seasoned too: the run completes, every row and path is finite - the monthly rows from December
    to July included, where the second average is fixed and the first is not - and the first row
    is the base valuation's mark. A static EURUSD fixes every sample at its spot, so at expiry the
    spread is -0.01, or -0.0075 seasoned at 1.24, and the option is worth nothing on any path.

    A leg's fixed count read with the block's counter reads `KeyError(1)` on the quarterly grid; a
    static spot's one row left standing for all of its leg's fixings prices the static run's expiry
    at 927.50.

    Killing mutation: a fully fixed leg sent through the spread formula, NaN from December to July
    on the monthly grid.
    """
    gbm = ({'FxRate': 'GBMAssetPriceModel'}, {},
           {'GBMAssetPriceModel.EUR': {'Vol': book.SIGMA, 'Drift': 0.0}})
    hull_white = ({}, {'InterestRate': [[['id', 'USD'], 'HullWhite1FactorInterestRateModel']]},
                  {'HullWhite1FactorInterestRateModel.USD': {
                      'Alpha': 0.05, 'Lambda': 0.0, 'Quanto_FX_Correlation': 0.0,
                      'Quanto_FX_Volatility': utils.Curve([], [[0.0, 0.0], [10.0, 0.0]]),
                      'Sigma': utils.Curve([], [[0.0, 0.01], [10.0, 0.01]])}})
    last = [(m, None) for m in (9, 10, 11, 12)]
    trialled, seasoned = [(m, None) for m in (1, 2, 3, 4)], [(-1, 1.24), (1, None), (2, None),
                                                             (3, None)]
    for second, grid, (defaults, filters, models) in (
            (trialled, '0d 3m(3m)', gbm), (trialled, '0d 1m(1m)', gbm),
            (trialled, '0d 3m(3m)', hull_white), (seasoned, '0d 3m(3m)', hull_white)):
        job = option(last, second)
        base = float.fromhex(marks(job)['FXDASN'])
        deals = job['Calc']['Deals']['Deals']
        deals['Children'] = [{'Instrument': {'.Deal': {
            'Object': 'NettingCollateralSet', 'Reference': 'NS', 'Netted': 'True',
            'Collateralized': 'False'}}, 'Children': deals['Children']}]
        market = job['Calc']['MergeMarketData']['ExplicitMarketData']
        market['Model Configuration'] = {'.ModelParams': {
            'modeldefaults': defaults, 'modelfilters': filters}}
        market['Price Models'] = models
        job['Calc']['Calculation'] = {
            'Object': 'CreditMonteCarlo', 'Base_Date': B, 'Currency': 'USD', 'Time_Grid': grid,
            'Batch_Size': 256, 'Simulation_Batches': 1, 'Random_Seed': 1,
            'Deflation_Interest_Rate': 'USD'}
        context = derivus.Context()
        context.load_json((json.dumps(job, cls=CustomJsonEncoder), 'double_asian'))
        profile = context.run_job()[1]['Results']['mtm']
        assert np.isfinite(profile.values).all(), (grid, profile.mean(axis=1))
        assert abs(profile.iloc[0].mean() - base) <= 1e-5 * base, (grid, profile.iloc[0].mean(), base)
        if models is hull_white[2]:
            assert (profile.loc[E] == 0.0).all(), profile.loc[E].describe()
