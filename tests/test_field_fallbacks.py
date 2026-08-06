"""A blank payoff currency must mean what leaving it out means.

Every optional field in the deal schema is declared with an empty default - `F('Payoff_Currency',
'Text', default='')`, eleven times - so a deal authored from any UI carries the key PRESENT and
EMPTY. A deal authored by hand omits it. Both say "not specified", and the fallback has to treat
them the same or the same deal prices two ways.

It did not. `reset()` tested PRESENCE (`'Payoff_Currency' in self.field`) while the same class's
`calc_dependencies` tested the VALUE, so a UI-authored equity option registered its settlement
currency under `''` while its payoff factor correctly resolved to Currency. Four idioms were in
use for this one rule; only the value test handles both authorings, and it is now the only one.

These tests pin the equivalence, not a number: absent and blank must price identically, and the
comparison is set up so that getting it wrong gives a different answer.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import pytest
import torch

import derivus
from derivus import utils
from derivus.config import Config
from derivus.instruments import construct_instrument

BASE = pd.Timestamp('2024-06-28')
DTYPE = torch.float64
# EUR differs from USD so a payoff currency resolved to the wrong one changes the answer
USD_RATE, EUR_RATE = 0.05, 0.01


def _price(deal, ref):
    c = Config()
    c.params['System Parameters']['Base_Currency'] = 'USD'
    c.params['System Parameters']['Base_Date'] = BASE
    c.params['Price Factors'] = {
        'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Priority': 1, 'Spot': 1.0},
        'FxRate.EUR': {'Domestic_Currency': 'USD', 'Interest_Rate': 'EUR', 'Priority': 1, 'Spot': 1.1},
        'InterestRate.USD': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                             'Curve': utils.Curve([], [[0.0, USD_RATE], [10.0, USD_RATE]])},
        'InterestRate.EUR': {'Currency': 'EUR', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                             'Curve': utils.Curve([], [[0.0, EUR_RATE], [10.0, EUR_RATE]])},
        'EquityPrice.ACME': {'Issuer': None, 'Respect_Default': 'No', 'Jump_Level': 0.0,
                             'Spot': 100.0, 'Interest_Rate': 'USD', 'Currency': 'USD'},
        'DividendRate.ACME': {'Floor': None, 'Currency': 'USD',
                              'Curve': utils.Curve([], [[0.0, 0.0], [10.0, 0.0]])},
        'VolatilityGrid.ACME': {'Surface': utils.Curve([], [[0.5, 1.0, 0.2], [1.5, 1.0, 0.2]])},
    }
    c.params['Price Models'] = {}
    c.params['Valuation Configuration'] = {}
    c.deals = {'Attributes': {'Reference': 'test', 'Tag_Titles': ''},
               'Deals': {'Children': [{'Instrument': construct_instrument(deal, {})}]},
               'Calculation': {'Base_Date': BASE, 'Currency': 'USD'}}
    _, out = derivus.run_baseval(c, prec=DTYPE, overrides={'MCMC_Simulations': 1, 'Random_Seed': 1})
    rows = out['Results']['mtm']
    return float(rows[rows['Reference'] == ref]['Value'].iloc[0])


def _equity_option(**over):
    deal = {
        'Object': 'EquityOptionDeal', 'Reference': 'EQO1', 'Currency': 'USD',
        'Discount_Rate': 'USD', 'Equity': 'ACME', 'Equity_Volatility': 'ACME',
        'Payoff_Type': 'Standard', 'Option_Type': 'Call', 'Buy_Sell': 'Buy',
        'Option_Style': 'European', 'Strike_Price': 100.0, 'Units': 1000.0,
        'Expiry_Date': BASE + pd.Timedelta(days=365), 'Forward_Price_Date': None,
    }
    deal.update(over)
    return deal


@pytest.mark.parametrize('blank', ['', None])
def test_a_blank_payoff_currency_means_the_deal_currency(blank):
    """Absent and blank must agree. With the presence test, a blank `Payoff_Currency` set
    `payoff_ccy` to `''` and `add_reval_dates` filed the settlement currency under a name no price
    factor has - while `calc_dependencies`, testing the value, resolved the payoff factor to
    Currency. Both `''` (what a UI writes) and None are covered because the two reach the fallback
    by different routes."""
    assert _price(_equity_option(Payoff_Currency=blank), 'EQO1') == pytest.approx(
        _price(_equity_option(), 'EQO1'), rel=1e-12)
