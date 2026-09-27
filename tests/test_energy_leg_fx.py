"""A floating energy leg is the sum of its cashflows, whatever currency its forward curve is in.

`pv_energy_cashflows` prices a leg one block of rows at a time - a block per set of cashflows still
unpaid - and converts every forecast fixing into the payoff currency at the forward FX seen from its
own row.
"""

import json

import pandas as pd

import derivus
from derivus import utils
from derivus.config import CustomJsonEncoder

BASE = pd.Timestamp('2026-01-15')
#: (period start, period end, payment) of two delivery months
FEB = ('2026-02-02', '2026-02-27', '2026-03-05')
MAR = ('2026-03-02', '2026-03-31', '2026-04-06')


def rate(currency, rates):
    return {'Currency': currency, 'Day_Count': 'ACT_365', 'Sub_Type': None,
            'Curve': utils.Curve([], [[0.0, rates[0]], [5.0, rates[1]]])}


#: a EUR forward curve under a Clewlow-Strickland model, paying USD at a static EURUSD
MARKET = {
    'System Parameters': {'Base_Currency': 'USD', 'Base_Date': BASE},
    'Model Configuration': {'.ModelParams': {
        'modeldefaults': {'ForwardPrice': 'CSForwardPriceModel'}, 'modelfilters': {}}},
    'Price Models': {'CSForwardPriceModel.FUEL': {'Alpha': 0.35, 'Drift': 0.0, 'Sigma': 0.2}},
    'Price Factors': {
        'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Spot': 1.0},
        'FxRate.EUR': {'Domestic_Currency': 'USD', 'Interest_Rate': 'EUR', 'Spot': 1.1},
        'InterestRate.USD': rate('USD', (0.041, 0.033)),
        'InterestRate.EUR': rate('EUR', (0.020, 0.025)),
        'ReferencePrice.FUEL': {'Fixing_Curve': utils.Curve([], [[45900, 45900], [46600, 46600]]),
                                'ForwardPrice': 'FUEL'},
        'ForwardPrice.FUEL': {'Currency': 'EUR',
                              'Curve': utils.Curve([], [[45900, 1450.0], [46600, 1550.0]])},
        'ForwardPriceSample.DAILY': {'Offset': 0, 'Holiday_Calendar': '',
                                     'Sampling_Convention': 'ForwardPriceSampleDaily'}}}


def leg(reference, *months):
    return {'Instrument': {'.Deal': {
        'Object': 'FloatingEnergyDeal', 'Reference': reference, 'Currency': 'USD',
        'Discount_Rate': 'USD', 'Reference_Type': 'FUEL', 'Sampling_Type': 'DAILY',
        'FX_Sampling_Type': '', 'Payer_Receiver': 'Receiver', 'Payments': {'Items': [
            {'Period_Start': pd.Timestamp(start), 'Period_End': pd.Timestamp(end),
             'Payment_Date': pd.Timestamp(paid), 'Volume': 2500.0, 'Realized_Average': 0.0,
             'FX_Realized_Average': 0.0} for start, end, paid in months]}}}}


def mtm(*legs):
    """The netting set's mark on every row and path of a daily credit Monte Carlo over `legs`."""
    job = {'Calc': {
        'Calculation': {'Object': 'CreditMonteCarlo', 'Base_Date': BASE, 'Currency': 'USD',
                        'Batch_Size': 16, 'Simulation_Batches': 1, 'Random_Seed': 1,
                        'Deflation_Interest_Rate': 'USD', 'Time_Grid': '0d 1d(1d)'},
        'Deals': {'Reference': 'fx', 'Deals': {'Children': [{
            'Instrument': {'.Deal': {'Object': 'NettingCollateralSet', 'Reference': 'NS',
                                     'Netted': 'True', 'Collateralized': 'False'}},
            'Children': list(legs)}]}},
        'MergeMarketData': {'ExplicitMarketData': MARKET}}}
    context = derivus.Context()
    context.load_json((json.dumps(job, cls=CustomJsonEncoder), 'energy_leg_fx'))
    return context.run_job()[1]['Results']['mtm']


def test_a_two_cashflow_leg_on_a_foreign_curve_prices_as_its_two_legs():
    """A leg on a EUR forward curve paying USD, run as a daily credit Monte Carlo, marks on every
    row and path as its two cashflows held as two legs - on the same paths, since both documents
    simulate the same factors on the same grid. The rows after the first payment are the leg's
    second block, where only the second cashflow is left to price.

    Killing mutation: the forward FX read off the whole grid (`discounts.time_grid`) rather than
    the block's own rows, which converts the second cashflow's fixings at the first block's dates
    and misses on every row after the first payment that still has a fixing to come.
    """
    joined, split = mtm(leg('FEB_MAR', FEB, MAR)), mtm(leg('FEB', FEB), leg('MAR', MAR))
    gap = (joined - split).abs().max(axis=1) / split.abs().max(axis=1).clip(lower=1.0)
    assert gap.max() <= 1e-6, (gap.idxmax(), gap.max())
