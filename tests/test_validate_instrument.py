"""Rules the engine enforces by failing, stated where an author can read them first.

Every rule below was already true - the engine raises, or silently prices against the wrong thing,
when it is broken. What was missing is any way to find out before running. `validate_instrument`
returns the messages; nothing in the valuation path calls it, and a message never stops a deal
pricing.

Two layers, because the rules have two shapes. `default=REQUIRED` covers a field that must simply
be there, and the declaration alone is enough. A rule spanning several fields cannot be declared on
any one of them and has no shape in common with the next one - a value being non-zero, two fields
being alternatives, one column of a table row implying another - so a class states those as code in
its own `validate()`.

Each test pairs a violating deal with a conforming one. Asserting only that a bad deal complains
would pass just as well if `validate_instrument` complained about everything.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import pytest

from derivus import instruments, schema

BASE = pd.Timestamp('2024-06-28')


def _v(deal):
    return schema.validate_instrument(instruments.construct_instrument(deal, {}))


BINARIES = ['EquityBinaryOption', 'EquityOneTouchOption', 'FXOneTouchOption', 'FXBinaryOption',
            'EquityBarrierBinaryOption']


@pytest.mark.parametrize('deal_type', BINARIES)
def test_cash_payoff_is_required_on_a_binary(deal_type):
    """Cash_Payoff IS the notional for these - `nominal = self.field['Cash_Payoff']`, and
    pv_barrier_option divides the rebate by it. Zero is not a small deal, it is no deal, and the
    declared default of 0 was offering exactly that."""
    assert 'Cash_Payoff is required' in _v({'Object': deal_type, 'Reference': 'X'})
    assert 'Cash_Payoff is required' not in _v(
        {'Object': deal_type, 'Reference': 'X', 'Cash_Payoff': 1e6})


def test_a_settlement_amount_needs_a_settlement_date():
    """pv_fixed_cashflows reaches `factor_dep['Settlement_Date'] - time_block` under
    `if settlement_amt:`, and calc_dependencies leaves that None when no date is supplied."""
    msg = 'Settlement_Amount is set, so Settlement_Date is required'
    assert msg in _v({'Object': 'CFFixedInterestListDeal', 'Reference': 'X',
                      'Settlement_Amount': 1e6})
    assert msg not in _v({'Object': 'CFFixedInterestListDeal', 'Reference': 'X',
                          'Settlement_Amount': 1e6, 'Settlement_Date': BASE})
    assert msg not in _v({'Object': 'CFFixedInterestListDeal', 'Reference': 'X'}), (
        'a deal with no settlement amount is not asked for a date')


def _netting(**cash_row):
    return {'Object': 'NettingCollateralSet', 'Reference': 'NS1',
            'Collateral_Assets': {'Cash_Collateral': [dict(
                {'Currency': 'USD', 'Amount': 1.0, 'Haircut_Posted': 0.0}, **cash_row)]}}


def test_a_collateral_rate_needs_a_funding_rate():
    """post_process enters the ColVA block on Collateral_Rate alone and then reads Funding_Rate
    unconditionally - there is no collateral valuation adjustment without a funding curve."""
    msg = 'Funding_Rate is required'
    assert any(msg in m for m in _v(_netting(Collateral_Rate='USD')))
    assert not any(msg in m for m in _v(_netting(Collateral_Rate='USD', Funding_Rate='USD')))
    assert not any(msg in m for m in _v(_netting())), (
        'a row asking for no ColVA is not asked to fund it')


def _inflation(**item):
    base = {'Payment_Date': BASE, 'Notional': 1e6, 'Accrual_Year_Fraction': 1.0}
    return {'Object': 'YieldInflationCashflowListDeal', 'Reference': 'X',
            'Cashflows': {'Items': [dict(base, **item)]}}


def test_each_index_cashflow_pins_both_references():
    """A value, or a date to read the index at. With neither, make_index_cashflows measures the
    offset from base_date and the cashflow prices against the wrong index level - silently, which
    is the only reason this rule is worth stating."""
    both_missing = _v(_inflation())
    assert any('Base_Reference_Value or Base_Reference_Date' in m for m in both_missing)
    assert any('Final_Reference_Value or Final_Reference_Date' in m for m in both_missing)

    by_value = _v(_inflation(Base_Reference_Value=100.0, Final_Reference_Value=110.0))
    assert not [m for m in by_value if 'Reference' in m], by_value
    by_date = _v(_inflation(Base_Reference_Date=BASE, Final_Reference_Date=BASE))
    assert not [m for m in by_date if 'Reference' in m], by_date


def test_a_deal_with_nothing_to_say_says_nothing():
    """Guards every test above: they all assert the ABSENCE of a message for a conforming deal, and
    a `validate_instrument` that returned [] unconditionally would satisfy all of them. This one
    fails in that case, because a genuinely incomplete deal has to produce something."""
    assert _v({'Object': 'FXForwardDeal', 'Reference': 'X', 'Buy_Discount_Rate': 'EUR',
               'Sell_Discount_Rate': 'USD'}) == []
    assert _v({'Object': 'FXForwardDeal', 'Reference': 'X'}) == [
        'Sell_Discount_Rate is required', 'Buy_Discount_Rate is required']
