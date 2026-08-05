"""`Cashflows['Properties']` selects the cap/floor pricer, and the guard and the selection disagreed.

`CFFloatingInterestListDeal.calc_dependencies` enters the branch on TRUTHINESS - `Cap_Multiplier`
or `Floor_Multiplier` non-zero - and then chose between the two pricers on PRESENCE:
`first_prop.get('Cap_Multiplier') is not None`. A floor carrying an explicit `Cap_Multiplier: 0.0`
therefore entered as a floor and priced as a cap, struck at `Cap_Strike` instead of `Floor_Strike`.

That is what an authored floor looked like. The Properties table declared 21 columns and the
Workbench writes every column of a table it renders (`set_repr` zips `col_names` against the row),
so `Cap_Multiplier: 0.0` came free with any floor a UI produced - and the table declared sixteen
columns no code has ever read, which is what made the trap wide enough to fall into.

No fixture reached this path at all: the suite's only Properties value is `[]`.
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

BASE = pd.Timestamp('2020-01-01')
DTYPE = torch.float64
RATE = 0.02          # flat continuously-compounded curve, so every forecast reset is ~2%
VOL = 0.20
NOTIONAL = 1_000_000.0
FLOOR_STRIKE = 10.0  # percent - far above the 2% forward, so the floor is deep in the money
CAP_STRIKE = 0.0     # percent - a cap struck here pays essentially the whole floating leg


def _items():
    dates = [BASE + pd.DateOffset(months=m) for m in (3, 6, 9, 12)]
    out = []
    for start, end in zip([BASE] + dates[:-1], dates):
        accrual = (end - start).days / 365.0
        out.append({'Payment_Date': end, 'Accrual_Start_Date': start, 'Accrual_End_Date': end,
                    'Accrual_Year_Fraction': accrual, 'Notional': NOTIONAL, 'Fixed_Amount': 0.0,
                    'Margin': utils.Basis(0.0),
                    'Resets': [[start, start, end, accrual, pd.DateOffset(months=3),
                                'ACT_365', '0D', 0.0, 'No', utils.Percent(0.0)]]})
    return out


def _deal(properties):
    return {
        'Object': 'CFFloatingInterestListDeal', 'Reference': 'FLT1', 'Currency': 'USD',
        'Discount_Rate': 'USD', 'Forecast_Rate': 'USD', 'Buy_Sell': 'Buy',
        'Forecast_Rate_Cap_Volatility': 'USD', 'Discount_Rate_Cap_Volatility': '',
        'Forecast_Rate_Swaption_Volatility': '', 'Discount_Rate_Swaption_Volatility': '',
        'Cashflows': {'Compounding_Method': 'None', 'Averaging_Method': 'None',
                      'Properties': properties, 'Items': _items()}}


def _price(properties):
    c = Config()
    c.params['System Parameters']['Base_Currency'] = 'USD'
    c.params['System Parameters']['Base_Date'] = BASE
    c.params['Price Factors'] = {
        'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Priority': 1, 'Spot': 1.0},
        'InterestRate.USD': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                             'Curve': utils.Curve([], [[0.0, RATE], [5.0, RATE]])},
        'DiscountRate.USD': {'Interest_Rate': 'USD'},
        'InterestRateVol.USD': {
            'Property_Aliases': None,
            'Surface': utils.Curve([], [[m, e, t, VOL] for m in (-0.01, 0.0, 0.01)
                                        for e in (0.25, 0.5, 1.0) for t in (0.25, 1.0)])},
    }
    c.params['Price Models'] = {}
    c.params['Valuation Configuration'] = {}
    c.deals = {'Attributes': {'Reference': 'test', 'Tag_Titles': ''},
               'Deals': {'Children': [{'Instrument': construct_instrument(_deal(properties), {})}]},
               'Calculation': {'Base_Date': BASE, 'Currency': 'USD'}}
    _, out = derivus.run_baseval(c, prec=DTYPE, overrides={'MCMC_Simulations': 1, 'Random_Seed': 1})
    rows = out['Results']['mtm']
    return float(rows[rows['Reference'] == 'FLT1']['Value'].iloc[0])


# a floor, authored the way a UI writes one: every column of the row present, the cap side zeroed
FLOOR_FROM_A_UI = [{'Cap_Multiplier': 0.0, 'Cap_Strike': utils.Percent(CAP_STRIKE),
                    'Floor_Multiplier': 1.0, 'Floor_Strike': utils.Percent(FLOOR_STRIKE),
                    'Digital_Payoff_Rate': None}]
# the same floor, authored by hand with the cap side simply absent
FLOOR_BY_HAND = [{'Floor_Multiplier': 1.0, 'Floor_Strike': utils.Percent(FLOOR_STRIKE),
                  'Digital_Payoff_Rate': None}]


def test_a_zero_cap_multiplier_does_not_turn_a_floor_into_a_cap():
    """The two rows describe the same instrument and must price the same. Before the fix the first
    selected `pricer_cap` and struck it at `Cap_Strike` (0%), pricing a floor as a cap on the whole
    floating leg - roughly the leg's own value rather than the floor's."""
    assert _price(FLOOR_FROM_A_UI) == pytest.approx(_price(FLOOR_BY_HAND), rel=1e-12)


def test_the_floor_is_worth_what_a_deep_in_the_money_floor_is_worth():
    """Guards the comparison above against agreeing on a wrong number: a floor struck 8% above a 2%
    forward is worth about `(K - F) * notional * accrual` summed over the four periods, and a cap
    struck at 0% would instead be worth about the floating leg itself - two very different values,
    which is what makes the first test able to fail."""
    floor = _price(FLOOR_BY_HAND)
    intrinsic = sum(NOTIONAL * ((FLOOR_STRIKE / 100.0) - RATE) * 0.25 for _ in range(4))
    assert floor == pytest.approx(intrinsic, rel=0.10), (
        f'floor {floor:,.0f} is not near its intrinsic {intrinsic:,.0f}')


def test_a_cap_still_prices_as_a_cap():
    """The converse leg: the fix must not turn every Properties row into a floor."""
    cap = _price([{'Cap_Multiplier': 1.0, 'Cap_Strike': utils.Percent(CAP_STRIKE),
                   'Floor_Multiplier': 0.0, 'Floor_Strike': utils.Percent(FLOOR_STRIKE),
                   'Digital_Payoff_Rate': None}])
    # struck at 0% against a 2% forward, a cap is worth about the floating leg
    leg = sum(NOTIONAL * RATE * 0.25 for _ in range(4))
    assert cap == pytest.approx(leg, rel=0.10), f'cap {cap:,.0f} is not near the leg {leg:,.0f}'
