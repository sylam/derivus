"""Equity - the share itself and its barrier, binary, one-touch and Asian options, as
`tests/test_position_scaling.py` trials them."""

import pandas as pd

import test_declared_defaults as book

E, B = book.WORLD_EXPIRY, book.WORLD_BASE

#: An option on the declared-defaults book's 'EQ' - spot 100 in USD, 1% dividends, 15% flat vol.
OPTION = {'Currency': 'USD', 'Discount_Rate': 'USD', 'Equity': 'EQ', 'Equity_Volatility': 'EQ',
          'Buy_Sell': 'Buy', 'Expiry_Date': E}

#: Twelve monthly closes to the expiry, none fixed yet.
MONTHLY = [[B + pd.DateOffset(months=month), ''] for month in range(1, 13)]

DEALS = [
    {'Object': 'EquityDeal', 'Reference': 'EQS', 'Currency': 'USD', 'Equity': 'EQ',
     'Buy_Sell': 'Buy', 'Investment_Horizon': E, 'Units': 100.0},
    # 10% below spot is reached inside the year at 15% vol, so the knock-out's rebate is paid
    dict(OPTION, Object='EquityBarrierOption', Reference='EQKO', Option_Type='Call',
         Strike_Price=100.0, Units=100.0, Cash_Rebate=500.0, Barrier_Type='Down_And_Out',
         Barrier_Price=90.0, Barrier_Dates=MONTHLY),
    dict(OPTION, Object='EquityBinaryOption', Reference='EQBIN', Option_Type='Call',
         Strike_Price=105.0, Cash_Payoff=10_000.0),
    dict(OPTION, Object='EquityBarrierBinaryOption', Reference='EQKOBIN', Option_Type='Call',
         Strike_Price=100.0, Cash_Payoff=10_000.0, Barrier_Type='Up_And_Out', Barrier_Price=120.0,
         Barrier_Dates=MONTHLY),
    {'Object': 'EquityOneTouchOption', 'Reference': 'EQOT', 'Currency': 'USD',
     'Discount_Rate': 'USD', 'Equity': 'EQ', 'Equity_Volatility': 'EQ', 'Buy_Sell': 'Buy',
     'Expiry_Date': E, 'Cash_Payoff': 10_000.0, 'Barrier_Type': 'Down',
     'Barrier_Price': 90.0},
    dict(OPTION, Object='EquityDiscreteExplicitAsianOption', Reference='EQASN', Option_Type='Call',
         Strike_Price=100.0, Units=100.0,
         Sampling_Data=[[B + pd.DateOffset(months=month), 0.0, 1.0] for month in (3, 6, 9, 12)]),
]
FACTORS = {}
CONFIGURATION = {}
UNMARKED = set()
