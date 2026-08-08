"""Hand-authored linear-rates worlds for the curve-bootstrap gates.

Every level here is invented. Nothing is copied from a vendor file, and nothing has to be: a
round-trip fixture constructs a curve, prices the benchmark set off it to GENERATE the quotes, and
then requires the bootstrap to recover the curve - so the numbers only have to be plausibly shaped,
not real.

The builders return deal-tree NODES (`{'Instrument': deal}`, or a container plus `Children`), which
is what `Config.set_calculation_children` and `BenchmarkInstruments` both take.

Quotes are quoted in PERCENT throughout, because that is what the engine reads: `DepositDeal`
divides its `Interest_Rate_Schedule` by 100, `SwapInterestDeal` divides `Swap_Rate` by 100, and
`FRADeal` wraps `FRA_Rate` in a `Basis(-100 * rate)` - which is the same scaling by a different
route. A `CFFixedInterestListDeal` row carries a `utils.Percent` instead, and that is the one place
the type says so.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from derivus import utils
from derivus.instruments import construct_instrument

BASE = pd.Timestamp('2026-08-03')


def market(currency, curves, discount_curve, day_count='ACT_365'):
    """A `Price Factors` block holding one `FxRate` and the named `InterestRate` curves.

    `discount_curve` is what the currency's own `FxRate` points at, which is the curve every
    `Discount_Rate`-less deal falls back to and the one discovery reaches through `dependant_fields`.
    """
    factors = {'FxRate.{}'.format(currency): {
        'Domestic_Currency': None, 'Interest_Rate': discount_curve, 'Priority': 1, 'Spot': 1.0}}
    for name, (tenors, rates) in curves.items():
        factors['InterestRate.{}'.format(name)] = {
            'Currency': currency, 'Day_Count': day_count, 'Sub_Type': None,
            'Curve': utils.Curve([], list(zip(tenors, rates)))}
    return factors


def node(deal):
    return {'Instrument': construct_instrument(deal, {})}


def deposit(ref, currency, discount, months, quote, day_count='ACT_360'):
    """A money-market deposit quoted at `quote` percent.

    The rate is pinned through `Interest_Rate_Schedule`, which is what keeps a depo quote off the
    forecast curve entirely - `DepositDeal.reset` drops the `Interest_Rate` dependency when the
    schedule covers every accrual start, so a quote cannot depend on the curve it is solving for.
    """
    maturity = BASE + pd.DateOffset(months=months)
    return node({
        'Object': 'DepositDeal', 'Reference': ref, 'Currency': currency,
        'Discount_Rate': discount, 'Interest_Rate': discount,
        'Effective_Date': BASE, 'Maturity_Date': maturity,
        'Payment_Frequency': pd.DateOffset(months=months),
        'Interest_Frequency': pd.DateOffset(months=months),
        'Accrual_Day_Count': day_count, 'Amount': 1e6, 'Amortisation': None,
        'Compounding': 'No', 'Payment_Timing': 'End', 'Payment_Offset': 0,
        'Accrual_Calendars': None, 'Payment_Calendars': None,
        'First_Coupon_Date': None, 'Penultimate_Coupon_Date': None,
        'Rate_Currency': '', 'FX_Reset_Offset': 0, 'Known_FX_Rates': None,
        'Interest_Rate_Schedule': utils.DateList({BASE: quote})})


def fra(ref, currency, forecast, discount, start_months, end_months, quote, day_count='ACT_360'):
    """A forward rate agreement on the projection curve, quoted at `quote` percent."""
    return node({
        'Object': 'FRADeal', 'Reference': ref, 'Currency': currency,
        'Discount_Rate': discount, 'Interest_Rate': forecast,
        'Effective_Date': BASE + pd.DateOffset(months=start_months),
        'Maturity_Date': BASE + pd.DateOffset(months=end_months),
        'Reset_Date': BASE + pd.DateOffset(months=start_months),
        'Day_Count': day_count, 'Principal': 1e6, 'FRA_Rate': quote,
        'Borrower_Lender': 'Borrower', 'Use_Known_Rate': 'No', 'Known_Rate': 0.0,
        'Payment_Timing': 'End', 'Calendars': None})


def par_swap(ref, currency, forecast, discount, years, quote,
             fixed_frequency=12, float_frequency=3, day_count='ACT_360'):
    """A par interest-rate swap quoted at `quote` percent - fixed against a single-reset floating
    leg. `Index_Tenor` of zero months is what makes each coupon carry ONE reset spanning its own
    accrual period, which is the vanilla shape; a multi-reset period is the OIS one below."""
    return node({
        'Object': 'SwapInterestDeal', 'Reference': ref, 'Currency': currency,
        'Discount_Rate': discount, 'Interest_Rate': forecast,
        'Effective_Date': BASE, 'Maturity_Date': BASE + pd.DateOffset(years=years),
        'Pay_Rate_Type': 'Fixed', 'Pay_Frequency': pd.DateOffset(months=fixed_frequency),
        'Pay_Day_Count': day_count, 'Pay_Interest_Frequency': pd.DateOffset(months=fixed_frequency),
        'Pay_Timing': 'End', 'Pay_Payment_Offset': 0, 'Pay_Accrual_Calendars': None,
        'Pay_Payment_Calendars': None, 'Pay_First_Coupon_Date': None,
        'Pay_Penultimate_Coupon_Date': None,
        'Receive_Frequency': pd.DateOffset(months=float_frequency), 'Receive_Day_Count': day_count,
        'Receive_Interest_Frequency': pd.DateOffset(months=0), 'Receive_Timing': 'End',
        'Receive_Payment_Offset': 0, 'Receive_Accrual_Calendars': None,
        'Receive_Payment_Calendars': None, 'Receive_First_Coupon_Date': None,
        'Receive_Penultimate_Coupon_Date': None,
        'Index_Tenor': pd.DateOffset(months=0), 'Index_Day_Count': day_count,
        'Index_Frequency': pd.DateOffset(months=0), 'Index_Offset': 0,
        'Index_Calendars': None, 'Index_Publication_Calendars': None,
        'Reset_Type': 'Standard', 'Rate_Multiplier': 1.0, 'Rate_Constant': utils.Percent(0.0),
        'Floating_Margin': 0.0, 'Fixed_Compounding': 'No', 'Compounding_Method': 'None',
        'Known_Rates': None, 'Amortisation': None, 'Swap_Rate': quote, 'Principal': 1e6,
        'Interest_Rate_Volatility': '', 'Discount_Rate_Volatility': ''})


def ois_swap(ref, currency, curve, years, quote, day_count='ACT_360'):
    """An OIS swap quoted at `quote` percent, as a container over two legs.

    The floating leg is a `CFFloatingInterestListDeal` with `Compounding_Method='OIS'` and ONE
    cashflow item per fixing, all sharing their coupon's payment date. That shape is the whole
    point: `compress_no_compounding(groupsize=-1)` merges the items of a payment date into one
    cashflow carrying all their resets, each still at `Weight` 1, and only then does
    `pv_float_cashflow_list` compound them geometrically. A leg whose resets arrive already
    weighted `1/n` compounds at `1/n` of the rate.

    Fixings are on business days, which is what an RFR actually publishes on, and each accrues to
    the next fixing so the daily windows tile the coupon exactly.
    """
    coupons = [BASE + pd.DateOffset(years=k) for k in range(years + 1)]
    float_items, fixed_items = [], []
    for start, end in zip(coupons[:-1], coupons[1:]):
        fixings = pd.bdate_range(start, end, inclusive='left')
        for fixing, nxt in zip(fixings, list(fixings[1:]) + [end]):
            accrual = utils.get_day_count_accrual(
                fixing, (nxt - fixing).days, utils.get_day_count(day_count))
            float_items.append({
                'Payment_Date': end, 'Notional': 1e6,
                'Accrual_Start_Date': fixing, 'Accrual_End_Date': nxt,
                'Accrual_Day_Count': day_count, 'Accrual_Year_Fraction': accrual,
                'Resets': [[fixing, fixing, nxt, accrual, pd.DateOffset(days=1), day_count, '0D',
                            0.0, 'No', utils.Percent(0.0)]],
                'Margin': utils.Basis(0.0), 'Fixed_Amount': 0.0,
                'FX_Reset_Date': None, 'Known_FX_Rate': 0.0})
        fixed_items.append({
            'Payment_Date': end, 'Notional': 1e6, 'Rate': utils.Percent(quote),
            'Accrual_Start_Date': start, 'Accrual_End_Date': end, 'Accrual_Day_Count': day_count,
            'Accrual_Year_Fraction': utils.get_day_count_accrual(
                start, (end - start).days, utils.get_day_count(day_count)),
            'Fixed_Amount': 0.0, 'Discounted': 'No',
            'FX_Reset_Date': None, 'Known_FX_Rate': 0.0})

    container = construct_instrument(
        {'Object': 'StructuredDeal', 'Reference': ref, 'Currency': currency,
         'Net_Cashflows': 'Yes'}, {})
    return {'Instrument': container, 'Children': [
        node(_cashflow_leg('CFFloatingInterestListDeal', ref + '_FLOAT', currency, curve, 'Buy',
                           {'Compounding_Method': 'OIS', 'Averaging_Method': 'Average_Interest',
                            'Properties': [], 'Items': float_items},
                           Forecast_Rate=curve, Rate_Adjustment_Method='None',
                           Rate_Sticky_Month_End='Yes', Rate_Offset=0, Rate_Calendars=None,
                           Accrual_Calendars=None, Forecast_Rate_Cap_Volatility='',
                           Forecast_Rate_Swaption_Volatility='', Discount_Rate_Cap_Volatility='',
                           Discount_Rate_Swaption_Volatility='')),
        node(_cashflow_leg('CFFixedInterestListDeal', ref + '_FIXED', currency, curve, 'Sell',
                           {'Compounding': 'No', 'Items': fixed_items},
                           Calendars=None, Rate_Currency=''))]}


def _cashflow_leg(object_type, ref, currency, discount, buy_sell, cashflows, **extra):
    """The `CashflowListDeal` block both interest-cashflow legs share, plus the type's own fields."""
    return dict({
        'Object': object_type, 'Reference': ref, 'Currency': currency,
        'Discount_Rate': discount, 'Buy_Sell': buy_sell, 'Description': '',
        'Settlement_Date': None, 'Settlement_Amount': 0.0, 'Settlement_Style': 'Physical',
        'Settlement_Amount_Is_Clean': 'Yes', 'Is_Defaultable': 'No', 'Repo_Rate': '',
        'Recovery_Rate': '', 'Survival_Probability': '', 'Investment_Horizon': None,
        'Issuer': '', 'Settlement_Rate': '', 'Cashflows': cashflows}, **extra)
