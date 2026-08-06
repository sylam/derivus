"""The quanto conditional: an equity paying in a currency other than its own.

`conditional_fields` is how discovery reaches factors no deal field names. The quanto rule is the
awkward one - an equity option settling in a second currency needs an FX vol surface for the pair
and the equity/fx correlation, and NEITHER is reachable from the deal's own `factor_fields`.

Ten deal types declare both `Equity_Volatility` and `Payoff_Currency` and no fixture had ever set
them to different currencies, so this whole branch ran in no test. That mattered when the three
asset-class vol types collapsed into one `VolatilityGrid`: the rule had been keyed on
`EquityPriceVol`, which fired only for equity deals, and the surviving key now sees every vol
surface - so what used to be implied by the factor type has to be asked of the instrument.

The correlation names the equity by its VOLATILITY field, which is correct: a vol surface is named
after the underlying it belongs to (`EquityPrice.EQ` / `VolatilityGrid.EQ`), so the two are the
same string by convention.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import pytest

from derivus import utils
from derivus.instruments import construct_instrument
import test_barrier_bridge as bb

PAYOFF_CCY = 'ZAR'


def _cfg(payoff_currency):
    """bb's equity world plus a second currency, carrying an option that settles in it."""
    c = bb._cfg()
    c.params['Price Factors'].update({
        'FxRate.' + PAYOFF_CCY: {'Domestic_Currency': 'USD', 'Interest_Rate': PAYOFF_CCY,
                                 'Priority': 1, 'Spot': 18.0},
        'InterestRate.' + PAYOFF_CCY: {'Currency': PAYOFF_CCY, 'Day_Count': 'ACT_365',
                                       'Sub_Type': None,
                                       'Curve': utils.Curve([], [[0.0, 0.0], [5.0, 0.0]])},
        # the pair's vol surface and the equity/fx correlation: the two the rule must reach
        'VolatilityGrid.USD.' + PAYOFF_CCY: {
            'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness',
            'Surface': utils.Curve([], [[m, t, 0.15] for m in (0.8, 1.0, 1.2) for t in (0.02, 2.0)])},
        'Correlation.EquityPrice.EQ/FxRate.USD.' + PAYOFF_CCY: {'Value': 0.3},
    })
    deal = {
        'Object': 'EquityOptionDeal', 'Reference': 'QUANTO1', 'Currency': 'USD',
        'Payoff_Currency': payoff_currency, 'Equity': 'EQ', 'Dividends': 'EQ',
        'Discount_Rate': 'USD', 'Equity_Volatility': 'EQ', 'Buy_Sell': 'Buy',
        'Option_Type': 'Call', 'Strike_Price': 100.0, 'Units': 1.0,
        'Expiry_Date': bb.BASE + pd.Timedelta(days=365),
    }
    c.deals['Deals']['Children'] = [{'Instrument': construct_instrument(deal, {})}]
    return c


def _discovered(payoff_currency):
    c = _cfg(payoff_currency)
    dependent_factors = c.calculate_dependencies(
        {'Currency': 'USD', 'Base_Date': bb.BASE}, bb.BASE, '0d 2y(3m)', False)[0]
    return {utils.check_tuple_name(f) for f in dependent_factors}


def test_a_quanto_equity_pulls_the_pair_vol_and_the_correlation():
    """Neither is named by any field of the deal - `factor_fields` gives Equity, Dividends,
    Discount_Rate and Equity_Volatility, and none of them mentions ZAR."""
    found = _discovered(PAYOFF_CCY)
    assert 'VolatilityGrid.USD.' + PAYOFF_CCY in found, sorted(f for f in found if 'Vol' in f)
    assert 'Correlation.EquityPrice.EQ/FxRate.USD.' + PAYOFF_CCY in found, (
        sorted(f for f in found if f.startswith('Correlation')))


def test_a_same_currency_equity_pulls_neither():
    """The discriminating half: the rule must fire on the currencies differing, not merely on the
    deal being an equity option. Without this the test above passes for a rule that always fires."""
    found = _discovered('USD')
    assert 'VolatilityGrid.USD.' + PAYOFF_CCY not in found
    assert not [f for f in found if f.startswith('Correlation')], (
        f'same-currency deal pulled a correlation: {sorted(f for f in found if f.startswith("Correlation"))}')


def test_equity_collateral_reaches_the_rule_without_a_deal_currency():
    """The instrument shape the guard actually protects, and the one a deal-only fixture misses.

    A `NettingCollateralSet` holding equity collateral reaches `EquityPrice` through the tuple key
    `('Collateral_Assets', 'Equity_Collateral', 'Equity')` - but it declares `Agreement_Currency` and
    `Balance_Currency`, no plain `Currency`. So the quanto rule is handed an instrument whose
    `field['Currency']` is a KeyError, and only the `Equity_Volatility` guard stops it being read.

    The failure is SILENT, not a crash: `add_rates_for_factor` catches KeyError, logs, and skips the
    factor - so the collateral equity would simply never be simulated. That also means the netting
    set has to be the ONLY thing naming the equity here; leaving a child deal that references it
    masks the drop and the test passes either way."""
    c = bb._cfg()
    netting = {
        'Object': 'NettingCollateralSet', 'Reference': 'NS1', 'Netted': 'True',
        'Agreement_Currency': 'USD', 'Funding_Rate': 'USD', 'Balance_Currency': 'USD',
        'Liquidation_Period': 10.0, 'Settlement_Period': 0.0, 'Collateralized': 'True',
        'Credit_Support_Amounts': {
            'Received_Threshold': utils.CreditSupportList([[0.0, 0.0]]),
            'Posted_Threshold': utils.CreditSupportList([[0.0, 0.0]]),
            'Independent_Amount': utils.CreditSupportList([[0.0, 0.0]]),
            'Minimum_Received': utils.CreditSupportList([[0.0, 0.0]]),
            'Minimum_Posted': utils.CreditSupportList([[0.0, 0.0]])},
        'Collateral_Assets': {'Equity_Collateral': [{'Equity': 'EQ', 'Units': 1.0}]},
    }
    cash_only = {
        'Object': 'FixedCashflowDeal', 'Reference': 'CF1', 'Currency': 'USD',
        'Discount_Rate': 'USD', 'Buy_Sell': 'Buy', 'Amount': 1.0,
        'Payment_Date': bb.BASE + pd.Timedelta(days=180),
    }
    c.deals['Deals']['Children'] = [
        {'Instrument': construct_instrument(netting, {}),
         'Children': [{'Instrument': construct_instrument(cash_only, {})}]}]
    found = {utils.check_tuple_name(f) for f in c.calculate_dependencies(
        {'Currency': 'USD', 'Base_Date': bb.BASE}, bb.BASE, '0d 2y(3m)', False)[0]}
    assert 'EquityPrice.EQ' in found, (
        'the collateral equity was dropped from discovery: ' + str(sorted(found)))


def test_a_non_equity_quanto_referencing_a_vol_grid_does_not_raise():
    """The regression the collapse could have introduced, and it needs the RIGHT deal to show it.

    One VolatilityGrid now serves every asset class, so the rule is handed FX and commodity surfaces
    too - and reading `Equity_Volatility` off those instruments is a KeyError during discovery. But
    the currency comparison short-circuits first, so any deal whose payoff currency defaults to its
    own currency never reaches that read and cannot expose the defect. It takes a deal that declares
    `Payoff_Currency`, references a vol grid, has NO `Equity_Volatility`, and sets the two
    currencies DIFFERENT. Four types qualify; FXBarrierOption is one."""
    c = bb._cfg()
    c.params['Price Factors'].update({
        'FxRate.' + PAYOFF_CCY: {'Domestic_Currency': 'USD', 'Interest_Rate': PAYOFF_CCY,
                                 'Priority': 1, 'Spot': 18.0},
        'InterestRate.' + PAYOFF_CCY: {'Currency': PAYOFF_CCY, 'Day_Count': 'ACT_365',
                                       'Sub_Type': None,
                                       'Curve': utils.Curve([], [[0.0, 0.0], [5.0, 0.0]])},
        'VolatilityGrid.USD.' + PAYOFF_CCY: {
            'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness',
            'Surface': utils.Curve([], [[m, t, 0.15] for m in (0.8, 1.0, 1.2) for t in (0.02, 2.0)])},
    })
    fx_barrier = {
        'Object': 'FXBarrierOption', 'Reference': 'FXB1', 'Currency': 'USD',
        'Underlying_Currency': PAYOFF_CCY, 'Payoff_Currency': PAYOFF_CCY, 'Discount_Rate': 'USD',
        'FX_Volatility': 'USD.' + PAYOFF_CCY, 'Buy_Sell': 'Buy', 'Option_Type': 'Call',
        'Strike_Price': 18.0, 'Underlying_Amount': 1.0, 'Cash_Rebate': 0.0,
        'Expiry_Date': bb.BASE + pd.Timedelta(days=365), 'Barrier_Type': 'Down_And_Out',
        'Barrier_Price': 16.0, 'Barrier_Monitoring_Frequency': pd.DateOffset(days=0),
    }
    c.deals['Deals']['Children'] = [{'Instrument': construct_instrument(fx_barrier, {})}]
    found = {utils.check_tuple_name(f) for f in c.calculate_dependencies(
        {'Currency': 'USD', 'Base_Date': bb.BASE}, bb.BASE, '0d 2y(3m)', False)[0]}
    assert 'VolatilityGrid.USD.' + PAYOFF_CCY in found
