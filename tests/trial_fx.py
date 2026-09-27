"""FX - the Asians, the one-touch, the partial-time barrier and the extendable forward, beside a
barrier paying its rebate, as `tests/test_position_scaling.py` trials them."""

import pandas as pd

import test_declared_defaults as book

E, B = book.WORLD_EXPIRY, book.WORLD_BASE


def dates(*months):
    return [B + pd.DateOffset(months=month) for month in months]


#: Quarterly fixings to the expiry, none printed and no extension decided yet.
FIXINGS = [[date, date + pd.DateOffset(days=2), None, None] for date in dates(3, 6, 9, 12)]

DEALS = [
    book.fx_leg('FXDiscreteExplicitAsianOption', 'FXASN', Expiry_Date=E,
                Sampling_Data=[[date, 0.0, 1.0] for date in dates(3, 6, 9, 12)]),
    # a call on the last four months' average over the first four's
    book.fx_leg('FXDiscreteExplicitDoubleAsianOption', 'FXDASN', Expiry_Date=E, Strike_Price=0.01,
                Sampling_Data_1=[[date, 0.0, 1.0] for date in dates(9, 10, 11, 12)],
                Sampling_Data_2=[[date, 0.0, 1.0] for date in dates(1, 2, 3, 4)]),
    {'Object': 'FXOneTouchOption', 'Reference': 'FXOT', 'Currency': 'USD',
     'Underlying_Currency': 'EUR', 'Discount_Rate': 'USD', 'FX_Volatility': 'EUR.USD',
     'Buy_Sell': 'Buy', 'Expiry_Date': E, 'Cash_Payoff': 10_000.0, 'Barrier_Type': 'Up',
     'Barrier_Price': 1.35},
    # 8% below spot for the first half year, so the knock-out's rebate is paid
    book.fx_leg('FXPartialTimeBarrierOption', 'FXPKO', Expiry_Date=E, Barrier_Type='Down_And_Out',
                Barrier_Price=1.15, Barrier_At_Start='Yes', Barrier_Limit_Date=dates(6)[0],
                Cash_Rebate=50.0),
    # rolling, so the extension boundary is solved rather than closed-form
    {'Object': 'FXExtendableForwardDeal', 'Reference': 'FXEXT', 'Currency': 'USD',
     'Underlying_Currency': 'EUR', 'Discount_Rate': 'USD', 'FX_Volatility': 'EUR.USD',
     'Buy_Sell': 'Buy', 'Option_Type': 'Call', 'Strike_Price': 1.24, 'Extension_Strike': 1.28,
     'Extension_Date': FIXINGS[1][0], 'Extension_Style': 'Rolling', 'Underlying_Amount': 1000.0,
     'Extendable_ExpiryDates': FIXINGS},
    # the declared-defaults book's barrier pays no rebate, so this one does
    book.fx_leg('FXBarrierOption', 'FXKOR', Expiry_Date=E, Barrier_Type='Down_And_Out',
                Barrier_Price=1.15, Cash_Rebate=50.0),
]
FACTORS = {}
CONFIGURATION = {}
UNMARKED = set()
