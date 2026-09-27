"""Equity swaps - the swap leg, the swaplet list and both autocalls, as
`tests/test_position_scaling.py` trials them."""
import pandas as pd

import test_declared_defaults as book
from derivus import utils

B = book.WORLD_BASE
QUARTERS = [B + pd.DateOffset(months=3 * k) for k in range(1, 5)]
SEASONED = B - pd.DateOffset(months=1)


def equity_leg(reference, amount_type):
    """A seasoned leg paying on `Units` shares or on `Principal`, as `amount_type` says."""
    return {'Object': 'EquitySwapLeg', 'Reference': reference, 'Currency': 'USD',
            'Payoff_Currency': 'USD', 'Discount_Rate': 'USD', 'Equity': 'EQ',
            'Equity_Volatility': 'EQ', 'Equity_Currency': 'USD', 'Buy_Sell': 'Buy',
            'Effective_Date': SEASONED, 'Maturity_Date': book.WORLD_EXPIRY,
            'Principal_Fixed_Variable': amount_type, 'Units': 1e4, 'Principal': 1e6,
            'Equity_Known_Prices': utils.DateEqualList([[SEASONED, 97.0, 1.0]]),
            'Known_Dividends': None}


def swaplet(start, end, known_start):
    return {'Start_Date': start, 'End_Date': end, 'Payment_Date': end, 'Amount': 1e4,
            'Start_Multiplier': 1.0, 'End_Multiplier': 1.0, 'Dividend_Multiplier': 1.0,
            'Known_Start_Price': known_start, 'Known_End_Price': 0.0,
            'Known_Start_FX_Rate': 1.0 if known_start else 0.0, 'Known_End_FX_Rate': 0.0,
            'Quanto_FX_Rate': 0.0}


def autocall(object_type, reference, barrier, **extra):
    return dict({
        'Object': object_type, 'Reference': reference, 'Currency': 'USD',
        'Payoff_Currency': 'USD', 'Equity': 'EQ', 'Dividends': 'EQ', 'Discount_Rate': 'USD',
        'Equity_Volatility': 'EQ', 'Buy_Sell': 'Buy', 'Option_Type': 'Call',
        'Strike_Price': 100.0, 'Expiry_Date': QUARTERS[-1], 'Units': 1e6,
        'Settlement_Style': 'Cash', 'Option_On_Forward': 'No', 'Option_Style': 'European',
        'Barrier': barrier, 'Payoff_Type': 'Standard',
        'Price_Fixing': [[date, 0.0] for date in QUARTERS],
        'Autocall_Coupons': [[date, 0.02] for date in QUARTERS],
        'Autocall_Thresholds': [[date, 1.0] for date in QUARTERS],
        'Barrier_Dates': [QUARTERS[-1]]}, **extra)


DEALS = [
    equity_leg('EQL_SHARES', 'Variable'),
    equity_leg('EQL_PRINCIPAL', 'Principal'),
    {'Object': 'EquitySwapletListDeal', 'Reference': 'EQSL', 'Currency': 'USD',
     'Discount_Rate': 'USD', 'Equity': 'EQ', 'Equity_Currency': 'USD', 'Equity_Volatility': 'EQ',
     'Buy_Sell': 'Buy', 'Amount_Type': 'Shares', 'Cashflows': {'Items': [
         swaplet(SEASONED, QUARTERS[0], 97.0), swaplet(QUARTERS[0], QUARTERS[1], 0.0)]}},
    autocall('QEDI_CustomAutoCallSwap', 'ACS', 0.7),
    autocall('QEDI_CustomAutoCallSwap_V2', 'ACS_V2', 70.0, Forecast_Rate='USD-PROJ',
             Floating_Margin=utils.Basis(50.0), Reset_Frequency=pd.DateOffset(months=3),
             Autocall_Floating=[[date, 0.0125] for date in QUARTERS]),
]
FACTORS = {}
CONFIGURATION = {}
UNMARKED = set()
