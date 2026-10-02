"""Rates - caps, floors, the fixed cashflow list, the mark-to-market cross-currency swap and the
inflation list, beside amortising and settling variants of the rates types already declared, as
`tests/test_position_scaling.py` trials them."""
import pandas as pd

import rates_world
import test_declared_defaults as book
from derivus import utils

E, B = book.WORLD_EXPIRY, book.WORLD_BASE
#: where every amortising deal here steps its principal down, inside each one's life
STEP = E + pd.DateOffset(months=6)


def months(count):
    return pd.DateOffset(months=count)


def stepped(rows):
    """`rows` with every period accruing from `STEP` on a quarter of a million smaller."""
    return [dict(row, Notional=row['Notional'] - 250_000.0) if row['Accrual_Start_Date'] >= STEP
            else row for row in rows]


def redeeming(rows):
    """`rows` with the principal paid back beside the last coupon."""
    return rows[:-1] + [dict(rows[-1], Fixed_Amount=rows[-1]['Notional'])]


def fixed_list(reference, buy_sell, items, **extra):
    return rates_world._cashflow_leg('CFFixedInterestListDeal', reference, 'USD', 'USD', buy_sell,
                                     {'Compounding': 'No', 'Items': items}, **extra)


def float_list(reference, buy_sell, items, currency='USD', forecast='USD-PROJ', optionlet=None,
               **extra):
    """A floating cashflow list, priced as caplets or floorlets where an `optionlet` row says so."""
    return rates_world._cashflow_leg(
        'CFFloatingInterestListDeal', reference, currency, currency, buy_sell,
        {'Compounding_Method': 'None', 'Averaging_Method': 'Average_Rate',
         'Properties': [optionlet] if optionlet else [], 'Items': items},
        Forecast_Rate=forecast, Forecast_Rate_Cap_Volatility=forecast if optionlet else '', **extra)


def optioned(kind, reference, strike):
    """A cap or floor on its optionlet list, the list amortising as the terms do. The list is
    what prices - it is itself held - and the terms are what a list is generated from."""
    cap = kind == 'CapDeal'
    optionlet = {'Digital_Payoff_Rate': None, 'Cap_Multiplier': float(cap),
                 'Cap_Strike': utils.Percent(strike), 'Floor_Multiplier': float(not cap),
                 'Floor_Strike': utils.Percent(strike)}
    return {'Object': kind, 'Reference': reference, 'Currency': 'USD', 'Discount_Rate': 'USD',
            'Forecast_Rate': 'USD-PROJ', 'Forecast_Rate_Volatility': 'USD-PROJ', 'Buy_Sell': 'Buy',
            'Effective_Date': E, 'Maturity_Date': E + pd.DateOffset(years=3),
            'Cap_Rate' if cap else 'Floor_Rate': strike, 'Principal': 1e6,
            'Amortisation': utils.DateList({STEP: 250_000.0}),
            'Children': [float_list(reference + '_LETS', 'Buy', stepped(book.float_items()),
                                    optionlet=optionlet)]}


def accrual(start, end):
    return utils.DayCount.accrual(start, (end - start).days, utils.DayCount.code('ACT_365'))


#: an inflation index printed monthly to June 2026, and the growth its later prints forecast at
PRINTS = [pd.Timestamp('2024-01-01') + months(count) for count in range(30)]
FACTORS = {
    'PriceIndex.INFL': {
        'Index': utils.Curve([], [[float((day - utils.excel_offset).days), 100.0 * 1.002 ** count]
                                  for count, day in enumerate(PRINTS)]),
        'Last_Period_Start': PRINTS[-1], 'Next_Publication_Date': B + pd.DateOffset(days=12),
        'Publication_Period': 'Monthly', 'Currency': 'USD'},
    'InflationRate.INFL': {
        'Price_Index': 'INFL', 'Reference_Name': 'IndexReferenceInterpolated3M',
        'Day_Count': 'ACT_365', 'Accrual_Calendar': '', 'Currency': 'USD',
        'Curve': utils.Curve([], [[0.0, 0.025], [10.0, 0.025]])}}
CONFIGURATION = {}

COUPONS = [B + months(6 * count) for count in range(5)]
DEPOSIT = dict(rates_world.deposit('DEPO_AMORT', 'USD', 'USD', 24, 4.2),
               Payment_Frequency=months(6), Interest_Frequency=months(6),
               Interest_Rate_Schedule=utils.DateList({start: 4.2 for start in COUPONS[:-1]}),
               Amortisation=utils.DateList({STEP: 250_000.0}))
SWAPTION = next(deal for deal in book.BOOK if deal['Reference'] == 'SWPT')

DEALS = [
    {'Object': 'CFFixedListDeal', 'Reference': 'FIXED_LIST', 'Currency': 'USD',
     'Discount_Rate': 'USD', 'Buy_Sell': 'Buy', 'Cashflows': {'Items': [
         {'Payment_Date': B + months(6), 'Fixed_Amount': 50_000.0},
         {'Payment_Date': E, 'Fixed_Amount': 75_000.0}]}},
    optioned('CapDeal', 'CAP_LIST', 4.5),
    optioned('FloorDeal', 'FLOOR_LIST', 4.5),
    # the euro leg's nominal resets off the pair each quarter; its legs are priced by the swap
    {'Object': 'MtMCrossCurrencySwapDeal', 'Reference': 'XCCY_MTM', 'MtM_Side': 'Pay',
     'Pay_Currency': 'EUR', 'Receive_Currency': 'USD', 'Principal_Exchange': 'Start_Maturity',
     'Effective_Date': E, 'Maturity_Date': E + pd.DateOffset(years=3), 'Children': [
         float_list('XCCY_MTM_EUR', 'Sell', [dict(row, FX_Reset_Date=row['Accrual_Start_Date'])
                                             for row in book.float_items(notional=1.25e6)],
                    currency='EUR', forecast='EUR'),
         fixed_list('XCCY_MTM_USD', 'Buy', book.fixed_items(4.0, months=3, notional=1.25e6))]},
    # a seasoned linker: its base print is known, its final ones forecast
    {'Object': 'YieldInflationCashflowListDeal', 'Reference': 'LINKER', 'Currency': 'USD',
     'Discount_Rate': 'USD', 'Buy_Sell': 'Buy', 'Index': 'INFL',
     'Index_Reference': {'Months_Lag': 3, 'Quarters_Lag': 0, 'Quarter_Reference_Month': 1,
                         'Reference_Type': 'Interpolated', 'Reference_Day': 1, 'Days_In_Period': 0},
     'Cashflows': {'Items': [{
         'Payment_Date': end, 'Notional': 1e6, 'Base_Reference_Date': B - pd.DateOffset(years=1),
         'Base_Reference_Value': 104.0, 'Final_Reference_Date': end, 'Final_Reference_Value': 0.0,
         'Accrual_Start_Date': start, 'Accrual_End_Date': end, 'Accrual_Day_Count': 'ACT_365',
         'Accrual_Year_Fraction': accrual(start, end), 'Yield': utils.Percent(2.0),
         'Margin': utils.Basis(0.0), 'Rate_Multiplier': 1.0, 'Is_Coupon': 'Yes'}
         for start, end in zip(COUPONS[:-1], COUPONS[1:])]}},
    dict(rates_world.par_swap('SWAP_AMORT', 'USD', 'USD-PROJ', 'USD', 3, 4.5),
         Amortisation=utils.DateList({STEP: 250_000.0})),
    DEPOSIT,
    dict(DEPOSIT, Reference='DEPO_AMORT_FLOAT', Interest_Rate='USD-PROJ',
         Interest_Rate_Schedule=None),
    # the legs amortise as the swaption's own schedules say, and the legs are what it prices
    dict(SWAPTION, Reference='SWPT_AMORT', Pay_Amortisation=utils.DateList({STEP: 250_000.0}),
         Receive_Amortisation=utils.DateList({STEP: 250_000.0}), Children=[
             fixed_list('SWPT_AMORT_FIXED', 'Sell', stepped(book.fixed_items(4.5))),
             float_list('SWPT_AMORT_FLOAT', 'Buy', stepped(book.float_items()))]),
    fixed_list('FIXED_FWD', 'Buy', redeeming(book.fixed_items(4.5)),
               Settlement_Date=B + months(3), Settlement_Amount=950_000.0),
    float_list('FLOAT_REDEEMING', 'Buy', redeeming(book.float_items())),
]
UNMARKED = {'SWPT_AMORT_FIXED', 'SWPT_AMORT_FLOAT', 'XCCY_MTM_EUR', 'XCCY_MTM_USD'}
