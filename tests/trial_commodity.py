"""Commodity and energy - forwards, futures, average-price swaps, the energy option and both energy
legs, as `tests/test_position_scaling.py` trials them."""

import pandas as pd

import test_declared_defaults as book
from derivus import utils

B, E = book.WORLD_BASE, book.WORLD_EXPIRY
DAY = pd.DateOffset(days=1)


def month(n):
    return B + pd.DateOffset(months=n)


def period(n):
    """Delivery month `n`, sampled daily."""
    return {'Period_Start': month(n), 'Period_End': month(n + 1) - DAY,
            'FX_Period_Start': month(n), 'FX_Period_End': month(n + 1) - DAY}


def paid(n):
    return month(n + 1) + 5 * DAY


METAL = {'Commodity': 'METAL', 'Currency': 'USD', 'Discount_Rate': 'USD'}
FUEL = {'Currency': 'USD', 'Discount_Rate': 'USD', 'Sampling_Type': 'FUEL',
        'FX_Sampling_Type': '', 'Reference_Type': 'FUEL'}
FUTURE = dict(METAL, Object='CommodityFutureDeal', Maturity_Date=month(6), Repo_Rate='USD',
              Carry='METAL_CARRY')

DEALS = [
    dict(METAL, Object='CommodityForwardDeal', Reference='METAL_FWD', Buy_Sell='Buy',
         Maturity_Date=E, Reference_Type='METAL', Units=50.0),
    dict(FUTURE, Reference='METAL_FUT', Units=20.0),
    # its units omitted, the one contract the convention means, which `scaled` writes at size
    dict(FUTURE, Reference='METAL_FUT_ONE'),
    dict(METAL, Object='CommodityAveragePriceSwapDeal', Reference='METAL_APS', Buy_Sell='Buy',
         Carry='METAL_CARRY', Units=250.0, Fixed_Price=990.0, Settlement_Date=paid(5),
         Sampling_Data=[[month(-1), 1005.0, 1.0]] + [[month(n), 0.0, 1.0] for n in range(1, 7)]),
    dict(FUEL, Object='FloatingEnergyDeal', Reference='FUEL_FLOAT', Payer_Receiver='Receiver',
         Reference_Volatility='', Payments={'Items': [
             dict(period(n), Payment_Date=paid(n), Volume=5000.0 * n, Fixed_Basis=2.5,
                  Price_Multiplier=1.0, Realized_Average=0.0, FX_Realized_Average=0.0)
             for n in (2, 3)]}),
    {'Object': 'FixedEnergyDeal', 'Reference': 'FUEL_FIXED', 'Currency': 'USD',
     'Discount_Rate': 'USD', 'Payer_Receiver': 'Payer', 'Payments': {'Items': [
         {'Payment_Date': paid(n), 'Volume': 5000.0 * n, 'Fixed_Price': 81.0} for n in (2, 3)]}},
    dict(FUEL, Object='EnergySingleOption', Reference='FUEL_OPT', Buy_Sell='Buy',
         Option_Type='Call', Settlement_Date=paid(4), Strike=82.0, Realized_Average=0.0,
         FX_Realized_Average=0.0, Volume=1000.0, Reference_Volatility='FUEL', **period(4)),
]

FACTORS = {
    'CommodityPrice.METAL': {'Spot': 1000.0, 'Currency': 'USD', 'Interest_Rate': 'USD-PROJ',
                             'Forward_Rate': 'METAL_CARRY'},
    'ForwardRate.METAL_CARRY': {'Currency': 'USD',
                                'Curve': utils.Curve([], [[46237.0, 0.010], [46602.0, 0.015]])},
    'ReferencePrice.METAL': {'Fixing_Curve': utils.Curve([], [[40000, 40000], [60000, 60000]]),
                             'ForwardPrice': 'METAL'},
    'ForwardPrice.METAL': {'Currency': 'USD',
                           'Curve': utils.Curve([], [[46237.0, 1000.0], [46602.0, 1025.0]])},
    'ReferencePrice.FUEL': {'Fixing_Curve': utils.Curve([], [[40000, 40000], [60000, 60000]]),
                            'ForwardPrice': 'FUEL'},
    'ForwardPrice.FUEL': {'Currency': 'USD',
                          'Curve': utils.Curve([], [[46237.0, 80.0], [46602.0, 84.0]])},
    'ForwardPriceSample.FUEL': {'Offset': 0, 'Holiday_Calendar': '',
                                'Sampling_Convention': 'ForwardPriceSampleDaily'},
    'ReferenceVol.FUEL': {'ForwardPriceVol': 'FUEL', 'ReferencePrice': 'FUEL'},
    'CommodityPriceVol.FUEL': {
        'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness',
        'Surface': utils.Curve([], [[m, t, 0.30] for m in (0.5, 1.0, 1.5)
                                    for t in (0.1, 1.0, 3.0)])},
}
CONFIGURATION = {}
UNMARKED = set()
