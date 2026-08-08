"""An FX forward discounts each leg on its own curve, and says so rather than guessing.

`Sell_Discount_Rate` fell back to `field['Buy_Currency']` when blank - the BUY leg's currency, with
`field['Sell_Currency']` computed on the line directly above. A blank sell rate therefore discounted
the sell leg on the buy curve and reported a number, with no error and no warning.

It hid for two reasons. Discovery iterates `factor_fields` over the RAW field and `get_fieldname`
drops blanks, so a blank rate loads no curve of its own and only a fallback naming an
already-discovered curve resolves at all - the buy curve always is one. And the obvious fixture
hides it exactly: with `Buy_Amount * spot == Sell_Amount` the two legs cancel when they share a
curve, so the wrong answer is 0.0 and reads as "the deal did not price" rather than "the deal priced
wrongly". The amounts below are deliberately asymmetric.

Both rates are now `default=REQUIRED` with no fallback, so a blank one fails rather than guessing.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import pytest
import torch

import derivus
from derivus import schema, utils
from derivus.config import Config
from derivus.instruments import construct_instrument

BASE = pd.Timestamp('2024-06-28')
DTYPE = torch.float64
USD_RATE, EUR_RATE = 0.05, 0.01   # far apart, so the choice of curve moves the number
SPOT_EUR = 1.1
BUY_AMOUNT, SELL_AMOUNT = 10e6, 9e6   # NOT 10e6 * SPOT_EUR - see the module docstring


def _mtm(drop=None, **over):
    deal = {'Object': 'FXForwardDeal', 'Reference': 'FWD1',
            'Buy_Currency': 'EUR', 'Sell_Currency': 'USD',
            'Buy_Amount': BUY_AMOUNT, 'Sell_Amount': SELL_AMOUNT,
            'Buy_Discount_Rate': 'EUR', 'Sell_Discount_Rate': 'USD',
            'Settlement_Date': BASE + pd.Timedelta(days=730), 'Discount_Rate': 'USD'}
    deal.update(over)
    if drop:
        del deal[drop]
    c = Config()
    c.params['System Parameters'].update({'Base_Currency': 'USD', 'Base_Date': BASE})
    c.params['Price Factors'] = {
        'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Priority': 1, 'Spot': 1.0},
        'FxRate.EUR': {'Domestic_Currency': 'USD', 'Interest_Rate': 'EUR', 'Priority': 1,
                       'Spot': SPOT_EUR},
        'InterestRate.USD': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                             'Curve': utils.Curve([], [[0.0, USD_RATE], [10.0, USD_RATE]])},
        'InterestRate.EUR': {'Currency': 'EUR', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                             'Curve': utils.Curve([], [[0.0, EUR_RATE], [10.0, EUR_RATE]])},
    }
    c.params['Price Models'] = {}
    c.params['Valuation Configuration'] = {}
    c.deals = {'Attributes': {'Reference': 'test', 'Tag_Titles': ''},
               'Deals': {'Children': [{'Instrument': construct_instrument(deal, {})}]},
               'Calculation': {'Base_Date': BASE, 'Currency': 'USD'}}
    _, out = derivus.run_baseval(c, prec=DTYPE, overrides={'MCMC_Simulations': 1, 'Random_Seed': 1})
    rows = out['Results']['mtm']
    return rows[rows['Reference'] == 'FWD1']


def _price(**kw):
    return float(_mtm(**kw)['Value'].iloc[0])


def test_the_sell_leg_discounts_on_the_sell_curve():
    """Closed form: `Buy * D_EUR * spot - Sell * D_USD`, each leg on its OWN curve. Pinning the
    number is what makes the wrong-curve answer a failure rather than just a different number."""
    t = 730 / 365.0
    import math
    expected = (BUY_AMOUNT * math.exp(-EUR_RATE * t) * SPOT_EUR
                - SELL_AMOUNT * math.exp(-USD_RATE * t))
    assert _price() == pytest.approx(expected, rel=1e-3)


def test_discounting_the_sell_leg_on_the_buy_curve_is_a_different_number():
    """Guards the test above from passing on a fixture where the curves cannot be told apart - the
    exact failure that let the defect survive. If these two agree, nothing here proves anything."""
    assert _price() != pytest.approx(_price(Sell_Discount_Rate='EUR'), rel=1e-6)


@pytest.mark.parametrize('name', ['Buy_Discount_Rate', 'Sell_Discount_Rate'])
@pytest.mark.parametrize('how,names', [('absent', True), ('blank', False)])
def test_a_discount_rate_that_is_not_supplied_drops_the_deal(name, how, names, caplog):
    """With the fallback gone the deal cannot price, and `Deal.calculate`'s guard turns that into an
    ERROR naming the cause plus NO mtm row - the deal leaves the portfolio rather than reporting a
    number off the wrong curve. That is the whole point of the change: a dropped deal is visible in
    a way a 26%-wrong one is not.

    Absent names the FIELD (`KeyError: Sell_Discount_Rate`); blank names the FACTOR
    (`Cannot find InterestRate.`), because `check_rate_name('')` resolves to an empty curve name."""
    kw = {'drop': name} if how == 'absent' else {name: ''}
    with caplog.at_level('ERROR'):
        assert _mtm(**kw).empty, 'the deal priced despite an unsupplied discount rate'
    logged = ' '.join(r.getMessage() for r in caplog.records)
    assert 'Skipped' in logged, f'no skip logged: {logged!r}'
    assert (name if names else 'InterestRate') in logged, f'the log does not name the cause: {logged!r}'


def test_both_rates_are_declared_required():
    """The schema has to say what the engine enforces, or a UI keeps offering a blank."""
    f = schema.mapping['Instrument']['sections']['FXForwardDeal.Fields']
    for name in ('Buy_Discount_Rate', 'Sell_Discount_Rate'):
        assert f[name].get('required') is True, f'{name} is not declared required'
        assert f[name]['value'] == '', f'{name} still offers a default'
