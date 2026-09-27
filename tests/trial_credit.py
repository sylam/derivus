"""Credit and cash - the default swap, the nth-to-default basket and the cash account, as
`tests/test_position_scaling.py` trials them."""

import pandas as pd

import rates_world
from derivus import utils

B = rates_world.BASE


def survival(hazard):
    """A flat `hazard`, as the negative log survival probability a `SurvivalProb` curve holds."""
    return {'Recovery_Rate': 0.4, 'Curve': utils.Curve([], [[0.0, 0.0], [10.0, 10.0 * hazard]])}


#: Both credit deals start today, amortise inside their lives and state their coupons as bare
#: percents: a number `scaled` would move were a rate declared sized, which a tagged basis is not.
DEALS = [
    {'Object': 'DealDefaultSwap', 'Reference': 'CDS', 'Currency': 'USD', 'Discount_Rate': 'USD',
     'Name': 'ISSUER_A', 'Buy_Sell': 'Buy', 'Effective_Date': B,
     'Maturity_Date': B + pd.DateOffset(years=5), 'Pay_Frequency': pd.DateOffset(months=3),
     'Pay_Rate': 1.0, 'Principal': 1e7,
     'Amortisation': utils.DateList({B + pd.DateOffset(years=2): 4e6})},
    {'Object': 'CreditNthToDefault', 'Reference': 'NTD', 'Currency': 'USD',
     'Discount_Rate': 'USD', 'Names': ['ISSUER_A', 'ISSUER_B', 'ISSUER_C'],
     'CDS_Index': 'ISSUER_INDEX', 'Effective_Date': B,
     'Maturity_Date': B + pd.DateOffset(years=3), 'Pay_Frequency': pd.DateOffset(months=3),
     'Pay_Rate': 2.5, 'Max_Defaults': 2, 'Correlation': 0.3, 'Buy_Sell': 'Buy',
     'Principal': 5e6, 'Amortisation': utils.DateList({B + pd.DateOffset(months=18): 1e6})},
    {'Object': 'CashAccountDeal', 'Reference': 'CASH', 'Currency': 'USD', 'Discount_Rate': 'USD',
     'Investment_Horizon': B + pd.DateOffset(years=1), 'Units': 1e6},
]
FACTORS = {'SurvivalProb.ISSUER_A': survival(0.02), 'SurvivalProb.ISSUER_B': survival(0.03),
           'SurvivalProb.ISSUER_C': survival(0.015), 'SurvivalProb.ISSUER_INDEX': survival(0.02)}
CONFIGURATION = {}
UNMARKED = set()
