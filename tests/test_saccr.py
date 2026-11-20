"""The SA-CCR calculation - CRE52's exposure at default of each netting set - gated through job
documents on CRE99's five worked examples, each a netting set booked on a world of the module's
own, and on sets built to reach the floors, the margin terms and the classes the examples leave
out.

Every level is invented, chosen so the examples' terms are what the engine reads: a USD curve flat
at 4% and a EUR one through 6% a year at each anniversary of the base date, so the one-into-ten
receiver's legs have the forward swap rate of 6% CRE99.31 assumes; a euro at 1.25 dollars, a
pound at 1.5 and a dollar at 150 yen, so a notional in any is crossed; three survival curves rated in the module's mapping as
CRE99.41 rates the entities; three commodities whose forward curves price a unit at 80, 100 and 25
dollars, silver with an LBMA basis of -0.5 and WTI sampled daily; an index at 4,000 and a share at
50, each yielding 2%; the FX, equity and commodity vol surfaces flat at 10%, which no exposure
reads; a USD-OIS curve flat at 3.9%, a basis's second index; a USD cap surface flat at 20%,
which the caps and floors price on and no exposure reads; and a price index printed monthly
through 2024 with its inflation curve flat at 2.5%, which the inflation legs price on and no
exposure reads. The swaps' rates and the default swaps' coupons are
solved for the market values the examples state and the receiver's vol for its 50 thousand. A
commodity forward is worth its funded price, ten million and more, which no term moves, so silver
is lent at the rate that makes the three forwards worth the 20 thousand CRE99.59 states together:
each set is worth the example's V.
A date counts in years of 30/360, so every tenor is the example's to the day. The module's mapping
is the fixture's with the world's credit, equity and commodity factors stated beside it.

The examples print their figures rounded, in thousands: each is held to half a unit of the last
digit it prints. Their RC, the hedging sets' effective notionals and add-ons, the multiplier and
the EAD are printed in formulas (CRE99.24-99.40, 99.55-99.58, 99.71-99.76, 99.78-99.79,
99.82-99.97).
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest
from scipy import optimize, stats

import derivus
from derivus import bootstrappers, utils
from derivus.config import CustomJsonEncoder, ModelParams
from test_option_settlement_json import lag

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures')
PARAMETERS = os.path.join(FIXTURES, 'saccr_parameters.json')

BASE = pd.Timestamp('2025-01-15')
SIX_MONTHS = BASE + pd.DateOffset(months=6)
EURO, STERLING, YEN = 1.25, 1.5, 150.0


def years(n):
    return BASE + pd.DateOffset(years=n)


def excel(day):
    return float((day - utils.excel_offset).days)


def tenor(day):
    """`day` in years of ACT/365, the curves' count."""
    return (day - BASE).days / 365.0


def flat(rate, currency='USD'):
    return {'Currency': currency, 'Day_Count': 'ACT_365', 'Sub_Type': None,
            'Curve': utils.Curve([], [[0.0, rate], [40.0, rate]])}


def flat_space(vol):
    """A rate vol space at `vol` on every node."""
    return utils.Curve([], [[m, e, t, vol] for m in (-0.02, 0.0, 0.02) for e in (0.25, 1.0, 5.0)
                            for t in (1.0, 5.0, 10.0, 20.0)])


def commodity(name, price, repo='USD'):
    """A commodity's spot, reference price and forward curve, a unit at `price` today, funded on
    `repo`."""
    return {'CommodityPrice.' + name: {'Spot': price, 'Currency': 'USD', 'Interest_Rate': repo,
                                       'Forward_Rate': 'CARRY'},
            'ReferencePrice.' + name: {
                'Fixing_Curve': utils.Curve([], [[40000, 40000], [60000, 60000]]),
                'ForwardPrice': name},
            'ForwardPrice.' + name: {'Currency': 'USD', 'Curve': utils.Curve(
                [], [[excel(BASE), price], [excel(years(6)), 1.06 * price]])}}


#: The lease that funds silver's forward at two thousandths over its ten million (CRE99.59's 20).
LEASE = 0.04 + math.log1p(2e-3) / tenor(years(5))
#: The first of each month of 2024, when CPI prints, 0.2% a month up from 100.
PRINTS = [pd.Timestamp('2024-01-01') + pd.DateOffset(months=k) for k in range(12)]


def factors(vol):
    """The world's `Price Factors`, the receiver's surface flat at `vol`."""
    surface = utils.Curve([], [[m, t, 0.1] for m in (0.8, 1.0, 1.2) for t in (0.5, 1.0, 2.0)])
    return dict({
        'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Spot': 1.0},
        'FxRate.EUR': {'Domestic_Currency': 'USD', 'Interest_Rate': 'EUR', 'Spot': EURO},
        'FxRate.GBP': {'Domestic_Currency': 'USD', 'Interest_Rate': 'GBP', 'Spot': STERLING},
        'FxRate.JPY': {'Domestic_Currency': 'USD', 'Interest_Rate': 'JPY', 'Spot': 1.0 / YEN},
        'InterestRate.USD': flat(0.04), 'InterestRate.SILVER_LEASE': flat(LEASE),
        'InterestRate.GBP': flat(0.05, 'GBP'), 'InterestRate.JPY': flat(0.01, 'JPY'),
        'InterestRate.USD-OIS': flat(0.039),
        'InterestRate.EUR': {'Currency': 'EUR', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                             'Curve': utils.Curve([], [[tenor(years(k)), k * math.log(1.06) / tenor(
                                 years(k))] for k in range(1, 12)])},
        'InterestYieldVol.EUR': {'Property_Aliases': None, 'Distribution_Type': 'Lognormal',
                                 'Shift': utils.Percent(0.0), 'Surface': flat_space(vol)},
        'InterestRateVol.USD': {'Property_Aliases': None, 'Surface': flat_space(0.2)},
        'PriceIndex.CPI': {'Index': utils.Curve([], [[excel(day), 100.0 * 1.002 ** k] for k, day in enumerate(PRINTS)]),
                           'Last_Period_Start': PRINTS[-1], 'Next_Publication_Date': BASE + pd.DateOffset(days=12),
                           'Publication_Period': 'Monthly', 'Currency': 'USD'},
        'InflationRate.CPI': {'Price_Index': 'CPI', 'Day_Count': 'ACT_365', 'Accrual_Calendar': '', 'Currency': 'USD',
                              'Curve': utils.Curve([], [[0.0, 0.025], [40.0, 0.025]])},
        'ForwardRate.CARRY': {'Currency': 'USD', 'Curve': utils.Curve(
            [], [[excel(BASE), 0.01], [excel(years(10)), 0.01]])}}, **{
        name: {'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness', 'Surface': surface}
        for name in ('FXVol.EUR.USD', 'FXVol.USD.JPY', 'EquityPriceVol.INDEX', 'EquityPriceVol.SHARE')}, **{
        'EquityPrice.' + name: {'Spot': spot, 'Currency': 'USD', 'Interest_Rate': 'USD',
                                'Issuer': '', 'Respect_Default': 'No', 'Jump_Level': 0.0}
        for name, spot in (('INDEX', 4000.0), ('SHARE', 50.0))}, **{
        'DividendRate.' + name: {'Currency': 'USD', 'Floor': None,
                                 'Curve': utils.Curve([], [[0.0, 0.02], [5.0, 0.02]])}
        for name in ('INDEX', 'SHARE')}, **{
        'SurvivalProb.' + name: {'Recovery_Rate': 0.4, 'Curve': utils.Curve(
            [], [[0.0, 0.0], [10.0, 10.0 * hazard]])}
        for name, hazard in (('FIRM_A', 0.005), ('FIRM_B', 0.02), ('CDX_IG', 0.01),
                             ('UNRATED', 0.01))},
        **commodity('WTI', 80.0), **commodity('BRENT', 100.0),
        **commodity('SILVER', 25.0, 'SILVER_LEASE'), **{
            'ObservedBasis.SILVER.LBMA': {'Spot': -0.5}, 'CommodityPriceVol.WTI': {
                'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness', 'Surface': surface},
            'ForwardPriceSample.WTI': {'Offset': 0, 'Holiday_Calendar': '',
                                       'Sampling_Convention': 'ForwardPriceSampleDaily'}})


#: The module's credit, equity and commodity factors as its mapping states them beside the
#: fixture's, the entities rated as CRE99.41 rates them.
STATED = {
    'SurvivalProb.FIRM_A': {'SIMM': None, 'SACCR': {'Rating': 'AA'}},
    'SurvivalProb.FIRM_B': {'SIMM': None, 'SACCR': {'Rating': 'BBB'}},
    'SurvivalProb.CDX_IG': {'SIMM': None, 'SACCR': {'Rating': 'IG'}},
    'SurvivalProb.UNRATED': {'SIMM': None},
    'CommodityPrice.WTI': {'SIMM': {'RiskType': 'Risk_Commodity', 'Qualifier': 'WTI', 'Bucket': '2'}},
    'CommodityPrice.BRENT': {'SIMM': {'RiskType': 'Risk_Commodity', 'Qualifier': 'BRENT',
                                      'Bucket': '2'}},
    'CommodityPrice.SILVER': {'SIMM': {'RiskType': 'Risk_Commodity', 'Qualifier': 'SILVER',
                                       'Bucket': '12'}},
    'ForwardPrice.WTI': {'SIMM': {'RiskType': 'Risk_Commodity', 'Qualifier': 'WTI', 'Bucket': '2'}},
    'EquityPrice.INDEX': {'SIMM': {'RiskType': 'Risk_Equity', 'Qualifier': 'INDEX', 'Bucket': '11'}},
    'EquityPrice.SHARE': {'SIMM': {'RiskType': 'Risk_Equity', 'Qualifier': 'SHARE', 'Bucket': '5'}}}


def swap(reference, maturity, pays, rate=4.0, **terms):
    """A USD swap from the base date to `maturity` on ten million, paying the `pays` leg."""
    return dict({'Object': 'SwapInterestDeal', 'Reference': reference, 'Currency': 'USD',
                 'Discount_Rate': 'USD', 'Interest_Rate': 'USD', 'Principal': 1e7,
                 'Effective_Date': BASE, 'Maturity_Date': maturity, 'Pay_Rate_Type': pays,
                 'Swap_Rate': rate}, **terms)


def default_swap(reference, name, maturity, side, currency='USD'):
    """A default swap on `name` from the base date to `maturity` on ten million dollars' worth."""
    return {'Object': 'DealDefaultSwap', 'Reference': reference, 'Currency': currency,
            'Discount_Rate': currency, 'Name': name, 'Buy_Sell': side,
            'Principal': 1e7 / (EURO if currency == 'EUR' else 1.0), 'Effective_Date': BASE,
            'Maturity_Date': maturity, 'Pay_Rate': utils.Basis(100.0)}


def forward(name, price, notional, maturity, side):
    """A forward on `notional` dollars' worth of `name` at its `price` today."""
    return {'Object': 'CommodityForwardDeal', 'Reference': name, 'Currency': 'USD',
            'Discount_Rate': 'USD', 'Commodity': name, 'Reference_Type': name, 'Buy_Sell': side,
            'Units': notional / price, 'Maturity_Date': maturity}


ANNIVERSARIES = [years(k) for k in range(1, 12)]


def receiver_leg(kind, terms, own, dates):
    """A receiver's leg of `kind`, annual over `dates` on four million euros, its `terms` and `own`
    the columns of the period from `start` to `end`."""
    return {'Instrument': {'.Deal': {
        'Object': kind, 'Reference': 'SWPT_' + kind[2:7], 'Currency': 'EUR', 'Discount_Rate': 'EUR',
        'Forecast_Rate': 'EUR', 'Buy_Sell': 'Buy', 'Cashflows': dict(terms, Items=[dict({
            'Payment_Date': end, 'Notional': 4e6, 'Accrual_Start_Date': start,
            'Accrual_End_Date': end, 'Fixed_Amount': 0.0, 'FX_Reset_Date': None,
            'Known_FX_Rate': 0.0}, **own(start, end)) for start, end in zip(dates[:-1], dates[1:])])}}}


def receiver(expiry, dates, **terms):
    """A bought receiver struck at 5% on four million euros, exercised on `expiry` into the swap
    over `dates`, on its legs: a 30/360 fixed leg, so a year accrues one."""
    return {'Instrument': {'.Deal': dict({
        'Object': 'SwaptionDeal', 'Reference': 'SWPT', 'Currency': 'EUR', 'Discount_Rate': 'EUR',
        'Forecast_Rate': 'EUR', 'Forecast_Rate_Volatility': 'EUR', 'Buy_Sell': 'Buy',
        'Payer_Receiver': 'Receiver', 'Settlement_Style': 'Physical', 'Option_Expiry_Date': expiry,
        'Swap_Effective_Date': dates[0], 'Swap_Maturity_Date': dates[-1], 'Settlement_Date': expiry,
        'Principal': 4e6, 'Swap_Rate': 5.0, 'Index_Tenor': pd.DateOffset(months=12),
        'Pay_Frequency': pd.DateOffset(months=12),
        'Pay_Day_Count': 'ACT_365', 'Receive_Frequency': pd.DateOffset(months=12),
        'Receive_Day_Count': '_30_360'}, **terms)},
        'Children': [
            receiver_leg('CFFixedInterestListDeal', {'Compounding': 'No'}, lambda start, end: {
                'Rate': utils.Percent(5.0), 'Accrual_Day_Count': '_30_360',
                'Accrual_Year_Fraction': 1.0, 'Discounted': 'No'}, dates),
            receiver_leg('CFFloatingInterestListDeal', {
                'Compounding_Method': 'None', 'Averaging_Method': 'Average_Rate', 'Properties': []},
                lambda start, end: {
                    'Accrual_Day_Count': 'ACT_365', 'Accrual_Year_Fraction': (end - start).days / 365.0,
                    'Resets': [[start, start, end, (end - start).days / 365.0, pd.DateOffset(days=1),
                                'ACT_365', '0D', 0.0, 'No', utils.Percent(0.0)]],
                    'Margin': utils.Basis(0.0)}, dates)]}


#: Example 1's European swaption, a bought receiver one year into ten struck at 5% on five million
#: dollars' worth of euros, on its legs (CRE99.22).
RECEIVER = receiver(years(1), ANNIVERSARIES)
#: Table 1's swaption six months into five years, settled physically and in cash (CRE52.32).
SWAPTIONS = [receiver(SIX_MONTHS, [SIX_MONTHS + pd.DateOffset(years=k) for k in range(6)],
                      Reference=style.upper(), Settlement_Style=style) for style in ('Physical', 'Cash')]

#: CRE99.22 and CRE99.41: each linear trade and the market value its rate is solved for, in
#: thousands; CRE99.59's forwards, worth their funded prices.
LINEAR = [(swap('SWAP_10Y', years(10), 'Fixed'), 30.0), (swap('SWAP_4Y', years(4), 'Floating'), -20.0),
          (default_swap('CDS_A', 'FIRM_A', years(3), 'Buy'), 20.0),
          (default_swap('CDS_B', 'FIRM_B', years(6), 'Sell', 'EUR'), -40.0),
          (default_swap('CDX', 'CDX_IG', years(5), 'Buy'), 0.0)]
FORWARDS = [forward('WTI', 80.0, 1e7, BASE + pd.DateOffset(months=9), 'Buy'),
            forward('BRENT', 100.0, 2e7, years(2), 'Sell'),
            forward('SILVER', 25.0, 1e7, years(5), 'Buy')]


def margined(threshold, minimum, independent, held, mpor):
    """A margined agreement's terms in dollars: its received threshold and minimum transfer, the
    independent amount received, the collateral held and the MPOR in business days."""
    return {'Collateralized': 'True', 'Agreement_Currency': 'USD', 'Liquidation_Period': mpor,
            'Opening_Balance': held, 'Credit_Support_Amounts': {
                'Received_Threshold': utils.CreditSupportList([[1, threshold]]),
                'Minimum_Received': utils.CreditSupportList([[1, minimum]]),
                'Independent_Amount': utils.CreditSupportList([[1, independent]])}}


#: Example 5's agreement (CRE99.80): margined weekly, no threshold, a minimum transfer of five, an
#: independent amount of 150 received, 200 of collateral held, and the MPOR of CRE99.83 its weekly
#: margin floors.
EXAMPLE_5 = dict(
    margined(0.0, 5e3, 150e3, 200e3, 14), Collateral_Call_Frequency=pd.DateOffset(weeks=1))
#: A threshold of a million and a minimum transfer of fifty thousand over an independent amount of
#: 200 thousand held, at an MPOR of five, under the bilateral floor; a set holding four million
#: euros of cash cut 8% at a closeout; and one that posted a million dollars cut 10% on delivery.
THRESHOLD = margined(1e6, 50e3, 200e3, 200e3, 5)
OVER = dict(margined(0.0, 0.0, 0.0, 4e6, 10), Balance_Currency='EUR', Collateral_Assets={
    'Cash_Collateral': [{'Currency': 'EUR', 'Amount': 1.0, 'Haircut_Received': utils.Percent(8.0)}]})
POSTED = dict(margined(0.0, 0.0, 0.0, -1e6, 10), Collateral_Assets={
    'Cash_Collateral': [{'Currency': 'USD', 'Amount': 1.0, 'Haircut_Posted': utils.Percent(10.0)}]})

#: A bought euro call in dollars and a sold dollar call in euros on the one pair, forwards on an
#: index and a share, a payer amortising a fifth of its ten million a year, and a short future.
MIXED = [
    {'Object': 'FXOptionDeal', 'Reference': 'EUR_CALL', 'Currency': 'USD', 'Underlying_Currency': 'EUR',
     'Discount_Rate': 'USD', 'FX_Volatility': 'EUR.USD', 'Buy_Sell': 'Buy', 'Option_Type': 'Call',
     'Strike_Price': 1.3, 'Underlying_Amount': 1e6, 'Expiry_Date': years(1), 'Delivery_Date': years(1)},
    {'Object': 'FXOptionDeal', 'Reference': 'USD_CALL', 'Currency': 'EUR', 'Underlying_Currency': 'USD',
     'Discount_Rate': 'EUR', 'FX_Volatility': 'EUR.USD', 'Buy_Sell': 'Sell', 'Option_Type': 'Call',
     'Strike_Price': 0.75, 'Underlying_Amount': 1e6, 'Expiry_Date': SIX_MONTHS, 'Delivery_Date': SIX_MONTHS},
    {'Object': 'EquityForwardDeal', 'Reference': 'INDEX_FWD', 'Currency': 'USD', 'Discount_Rate': 'USD',
     'Equity': 'INDEX', 'Buy_Sell': 'Buy', 'Units': 2500.0, 'Forward_Price': 4000.0,
     'Maturity_Date': years(2)},
    {'Object': 'EquityForwardDeal', 'Reference': 'SHARE_FWD', 'Currency': 'USD', 'Discount_Rate': 'USD',
     'Equity': 'SHARE', 'Buy_Sell': 'Sell', 'Units': 1e5, 'Forward_Price': 50.0,
     'Maturity_Date': SIX_MONTHS},
    swap('AMORTISING', years(5), 'Fixed', Amortisation=utils.DateList(
        {years(k): 2e6 for k in range(1, 5)})),
    {'Object': 'CommodityFutureDeal', 'Reference': 'SILVER_FUT', 'Commodity': 'SILVER',
     'Currency': 'USD', 'Repo_Rate': 'USD', 'Carry': 'CARRY', 'Units': -4e4,
     'Maturity_Date': years(1)}]


def outright(reference, bought, sold, settles):
    """A deliverable forward buying `bought` for `sold`, each `(amount, currency)`, on `settles`."""
    return {'Object': 'FXForwardDeal', 'Reference': reference, 'Buy_Amount': bought[0],
            'Buy_Currency': bought[1], 'Buy_Discount_Rate': bought[1], 'Sell_Amount': sold[0],
            'Sell_Currency': sold[1], 'Sell_Discount_Rate': sold[1], 'Settlement_Date': settles}


#: Euros bought for dollars six months out and sold back a year out, and a non-deliverable forward
#: buying euros for pounds a year out, settled in dollars.
FX_LINEAR = [
    outright('FWD_EUR', (1e6, 'EUR'), (1.3e6, 'USD'), SIX_MONTHS),
    outright('FWD_USD', (1.3e6, 'USD'), (1e6, 'EUR'), years(1)),
    {'Object': 'FXNonDeliverableForward', 'Reference': 'NDF', 'Buy_Amount': 1e6, 'Buy_Currency': 'EUR',
     'Sell_Amount': 0.9e6, 'Sell_Currency': 'GBP', 'Settlement_Currency': 'USD', 'Discount_Rate': 'USD',
     'Settlement_Date': years(1)}]

#: `FX_LINEAR` as a book of record states each trade the other way: both its amounts negative.
FX_SIGNED = [dict(deal, Buy_Amount=-deal['Buy_Amount'], Sell_Amount=-deal['Sell_Amount'])
             for deal in FX_LINEAR]

#: A call and a put on the euro on their expiry day, delivering two business days on, and a call on
#: two million euros that expired the business day before at a print of 1.31, delivering the next.
EXPIRED = [
    dict(MIXED[0], Reference='EXPIRY_CALL', Expiry_Date=BASE, Delivery_Date=BASE + pd.offsets.BDay(2)),
    dict(MIXED[0], Reference='EXPIRY_PUT', Option_Type='Put', Expiry_Date=BASE,
         Delivery_Date=BASE + pd.offsets.BDay(2)),
    dict(MIXED[0], Reference='EXERCISED', Underlying_Amount=2e6, Expiry_Date=BASE - pd.offsets.BDay(1),
         Delivery_Date=BASE + pd.offsets.BDay(1), Price_Fixing=[[BASE - pd.offsets.BDay(1), 1.31]])]

#: The bought euro call written as a dollar put in euros, and a bought dollar call in yen struck at
#: 150 a year out written each way up, each pair's surface named as the market quotes it.
AXES = [
    dict(MIXED[0], Reference='EUR_CALL'),
    dict(MIXED[0], Reference='USD_PUT', Currency='EUR', Underlying_Currency='USD', Discount_Rate='EUR',
         Option_Type='Put', Strike_Price=1 / 1.3, Underlying_Amount=1.3e6),
    dict(MIXED[0], Reference='USD_CALL', Currency='JPY', Underlying_Currency='USD', Discount_Rate='JPY',
         FX_Volatility='USD.JPY', Strike_Price=YEN, Underlying_Amount=1e6),
    dict(MIXED[0], Reference='JPY_PUT', Currency='USD', Underlying_Currency='JPY', Discount_Rate='USD',
         FX_Volatility='USD.JPY', Option_Type='Put', Strike_Price=1 / YEN, Underlying_Amount=1e6 * YEN)]


def equity_option(reference, equity, side, kind, strike, units, expiry, **terms):
    """An option on `units` of `equity` priced in dollars."""
    return dict({'Object': 'EquityOptionDeal', 'Reference': reference, 'Currency': 'USD',
                 'Discount_Rate': 'USD', 'Equity': equity, 'Equity_Volatility': equity, 'Buy_Sell': side,
                 'Option_Type': kind, 'Strike_Price': strike, 'Units': units, 'Expiry_Date': expiry}, **terms)


#: A bought call on a thousand units of the index struck at 4,200 a year out, a sold put on twenty
#: thousand shares struck at 45 six months out, and the index call paid in euros on S X, struck at
#: 3,360 euros.
EQUITY_OPTIONS = [
    equity_option('INDEX_CALL', 'INDEX', 'Buy', 'Call', 4200.0, 1e3, years(1)),
    equity_option('SHARE_PUT', 'SHARE', 'Sell', 'Put', 45.0, 2e4, SIX_MONTHS),
    equity_option('COMPO_CALL', 'INDEX', 'Buy', 'Call', 3360.0, 1e3, years(1), Payoff_Currency='EUR',
                  Payoff_Type='Compo')]


def energy_option(reference, side, kind, strike, volume, start, end, realised=0.0):
    """An option on the average of WTI's daily fixings from `start` to `end`, settled five days on,
    `realised` the average of those already fixed."""
    return {'Object': 'EnergySingleOption', 'Reference': reference, 'Currency': 'USD',
            'Discount_Rate': 'USD', 'Sampling_Type': 'WTI', 'Reference_Type': 'WTI',
            'Reference_Volatility': 'WTI', 'Buy_Sell': side, 'Option_Type': kind, 'Strike': strike,
            'Volume': volume, 'Period_Start': start, 'Period_End': end, 'FX_Period_Start': start,
            'FX_Period_End': end, 'Settlement_Date': end + pd.DateOffset(days=5),
            'Realized_Average': realised, 'FX_Realized_Average': 0.0}


#: A bought call on a thousand barrels of the month's average from a year out struck at 82, and a
#: sold put on two thousand of the average from ten days ago to twenty days on, its fixings so far
#: at 79.
ENERGY = [energy_option('JANUARY', 'Buy', 'Call', 82.0, 1e3, years(1), years(1) + pd.DateOffset(months=1)),
          energy_option('RUNNING', 'Sell', 'Put', 80.0, 2e3, BASE - pd.DateOffset(days=10),
                        BASE + pd.DateOffset(days=20), 79.0)]


#: A bought euro call in dollars on the average of a fixing a month ago at 1.24 and two to come,
#: six months and a year out; and a sold put on a thousand units of the index's average, a fixing
#: ten days ago at 3,950 weighing twice each of two to come, three and six months out.
ASIANS = [
    dict(MIXED[0], Object='FXDiscreteExplicitAsianOption', Reference='FX_AVERAGE', Sampling_Data=[
        [BASE - pd.DateOffset(months=1), 1.24, 1.0], [SIX_MONTHS, 0.0, 1.0], [years(1), 0.0, 1.0]]),
    dict(equity_option('INDEX_AVERAGE', 'INDEX', 'Sell', 'Put', 3900.0, 1e3, SIX_MONTHS),
         Object='EquityDiscreteExplicitAsianOption', Sampling_Data=[
             [BASE - pd.DateOffset(days=10), 3950.0, 2.0], [BASE + pd.DateOffset(months=3), 0.0, 1.0],
             [SIX_MONTHS, 0.0, 1.0]])]


#: The euro call knocked out above 1.6, and sold knocked in below 1.1 watched for six months; the
#: index call knocked in above 4,100 by a close five days ago at 4,150, and the sold share put
#: knocked out below 40 watched monthly; and a bought put on a thousand barrels of WTI a year out
#: struck at 81, knocked out below 60.
BARRIERS = [
    dict(MIXED[0], Object='FXBarrierOption', Reference='FX_OUT', Barrier_Type='Up_And_Out',
         Barrier_Price=1.6),
    dict(MIXED[0], Object='FXPartialTimeBarrierOption', Reference='FX_WINDOW', Buy_Sell='Sell',
         Barrier_Type='Down_And_In', Barrier_Price=1.1, Barrier_Limit_Date=SIX_MONTHS),
    dict(EQUITY_OPTIONS[0], Object='EquityBarrierOption', Reference='INDEX_IN', Barrier_Type='Up_And_In',
         Barrier_Price=4100.0, Barrier_Dates=[[BASE - pd.DateOffset(days=5), 4150.0]]),
    dict(EQUITY_OPTIONS[1], Object='EquityBarrierOption', Reference='SHARE_OUT', Barrier_Type='Down_And_Out',
         Barrier_Price=40.0, Barrier_Dates=[[BASE + pd.DateOffset(months=k), None] for k in range(1, 7)]),
    {'Object': 'CommodityBarrierOption', 'Reference': 'WTI_OUT', 'Currency': 'USD', 'Discount_Rate': 'USD',
     'Reference_Type': 'WTI', 'Reference_Volatility': 'WTI', 'Buy_Sell': 'Buy', 'Option_Type': 'Put',
     'Strike_Price': 81.0, 'Units': 1e3, 'Barrier_Type': 'Down_And_Out', 'Barrier_Price': 60.0,
     'Expiry_Date': years(1)}]


def fra(reference, side, timing):
    """Table 1's forward rate agreement on ten million dollars from six months to a year out."""
    return {'Object': 'FRADeal', 'Reference': reference, 'Currency': 'USD', 'Discount_Rate': 'USD',
            'Interest_Rate': 'USD', 'Principal': 1e7, 'FRA_Rate': 4.0, 'Borrower_Lender': side,
            'Payment_Timing': timing, 'Effective_Date': SIX_MONTHS, 'Reset_Date': SIX_MONTHS,
            'Maturity_Date': years(1)}


#: The agreement borrowed and paid at its end, lent and paid at its start discounted; and a sold
#: swap of a thousand barrels of WTI's average over three monthly fixings, settled four months out.
SINGLES = [fra('PAID_AT_END', 'Borrower', 'End'), fra('PAID_AT_START', 'Lender', 'Discounted'),
           {'Object': 'CommodityAveragePriceSwapDeal', 'Reference': 'WTI_SWAP', 'Currency': 'USD',
            'Discount_Rate': 'USD', 'Commodity': 'WTI', 'Carry': 'CARRY', 'Buy_Sell': 'Sell', 'Units': 1e3,
            'Fixed_Price': 80.0, 'Settlement_Date': BASE + pd.DateOffset(months=4), 'Sampling_Data': [
                [BASE + pd.DateOffset(months=k), 0.0, 1.0] for k in (1, 2, 3)]}]


def node(deal):
    return {'Instrument': {'.Deal': deal}}


def held(prefix, nodes):
    """`nodes` under one set, every reference in them `prefix`ed, so a trade booked in two sets is
    two trades."""
    return [dict(item, Instrument={'.Deal': dict(item['Instrument']['.Deal'], Reference='{}_{}'.format(
        prefix, item['Instrument']['.Deal']['Reference']))}, **(
        {'Children': held(prefix, item['Children'])} if item.get('Children') else {}))
        for item in nodes]


def cashflows(kind, reference, side, dates, notionals, currency='USD', curve=None, **terms):
    """A list of `kind` in `currency`, at 4% where fixed and on `curve` (the currency's own where
    None) where floating, received where `side` is Buy: a period between each two of `dates`, on
    each of `notionals` (or the one), paid at its end, a reset behind the base date fixed at 4%."""
    fixed = kind == 'CFFixedInterestListDeal'

    def item(start, end, notional):
        accrual = (end - start).days / 365.0
        return dict({'Payment_Date': end, 'Notional': notional, 'Accrual_Start_Date': start,
                     'Accrual_End_Date': end, 'Accrual_Day_Count': 'ACT_365',
                     'Accrual_Year_Fraction': accrual, 'Fixed_Amount': 0.0, 'FX_Reset_Date': None,
                     'Known_FX_Rate': 0.0}, **({'Rate': utils.Percent(4.0), 'Discounted': 'No'} if fixed else {
                         'Margin': utils.Basis(0.0), 'Resets': [[start, start, end, accrual, pd.DateOffset(
                             days=1), 'ACT_365', '0D', 0.0, 'No', utils.Percent(4.0)]]}))

    sizes = notionals if isinstance(notionals, list) else [notionals] * (len(dates) - 1)
    return dict({'Object': kind, 'Reference': reference, 'Currency': currency, 'Discount_Rate': currency,
                 'Buy_Sell': side, 'Cashflows': dict({'Compounding': 'No'} if fixed else {
                     'Compounding_Method': 'None', 'Averaging_Method': 'Average_Rate', 'Properties': []},
                     Items=[item(*period) for period in zip(dates[:-1], dates[1:], sizes)])},
                **({} if fixed else {'Forecast_Rate': curve or currency}), **terms)


def package(reference, legs, currency='USD'):
    """A structured deal holding `legs`."""
    return {'Instrument': {'.Deal': {'Object': 'StructuredDeal', 'Reference': reference,
                                     'Currency': currency}}, 'Children': [node(leg) for leg in legs]}


def every(months, count, first=BASE):
    """`count` periods of `months` from `first`, as their `count + 1` dates."""
    return [first + pd.DateOffset(months=months * k) for k in range(count + 1)]


FIXED, FLOATING = 'CFFixedInterestListDeal', 'CFFloatingInterestListDeal'
#: Amounts fixed today: a list of two, a fee, a fixed energy leg received and a fixed leg alone; and
#: a package of the list and the fee.
KNOWN = [{'Object': 'CFFixedListDeal', 'Reference': 'AMOUNTS', 'Currency': 'USD', 'Discount_Rate': 'USD',
          'Buy_Sell': 'Buy', 'Cashflows': {'Items': [{'Payment_Date': SIX_MONTHS, 'Fixed_Amount': 5e4},
                                                     {'Payment_Date': years(1), 'Fixed_Amount': 7.5e4}]}},
         {'Object': 'FixedCashflowDeal', 'Reference': 'FEE', 'Currency': 'USD', 'Discount_Rate': 'USD',
          'Amount': 1e5, 'Payment_Date': years(2)},
         {'Object': 'FixedEnergyDeal', 'Reference': 'FIXED_ENERGY', 'Currency': 'USD', 'Discount_Rate': 'USD',
          'Payer_Receiver': 'Receiver', 'Payments': {'Items': [
              {'Payment_Date': BASE + pd.DateOffset(months=3), 'Volume': 1e3, 'Fixed_Price': 80.0}]}},
         cashflows(FIXED, 'FIXED_LEG', 'Buy', every(12, 3), 1e6)]
#: A floating leg on ten million amortising two a year, its first period begun six months ago.
STARTED = every(12, 5, BASE - pd.DateOffset(months=6))
AMORTISED = [1e7, 8e6, 6e6, 4e6, 2e6]
#: The payer of ten million five years out, quarterly, booked as a package of its two legs.
QUARTERS = every(3, 20)
SWAP_LEGS = [cashflows(FIXED, 'FIXED', 'Sell', QUARTERS, 1e7), cashflows(FLOATING, 'FLOAT', 'Buy', QUARTERS, 1e7)]
#: Two lists of fixed amounts beside them: ten million received at the end, 500 thousand paid in six months.
AMOUNTS = [dict(KNOWN[0], Reference=reference, Buy_Sell=side, Cashflows={'Items': [
    {'Payment_Date': day, 'Fixed_Amount': amount}]}) for reference, side, day, amount in (
        ('PRINCIPAL', 'Buy', years(5), 1e7), ('FEE', 'Sell', SIX_MONTHS, 5e5))]
#: Ten million three years out, yearly, received on USD-OIS against USD and the other way round.
BASIS = [package(reference, [cashflows(FLOATING, 'OIS', ois, every(12, 3), 1e7, curve='USD-OIS'),
                             cashflows(FLOATING, 'USD', usd, every(12, 3), 1e7)])
         for reference, ois, usd in (('OIS_RECEIVED', 'Buy', 'Sell'), ('OIS_PAID', 'Sell', 'Buy'))]
#: Cross-currency swaps three years out, yearly: a floating leg on 1.25 million dollars against a
#: fixed one on a million euros, the dollars received and then the euros; and a floating leg on a
#: million euros received against a fixed one on 900 thousand pounds.
CROSS = [package(reference, [cashflows(FLOATING, 'USD', usd, every(12, 3), 1.25e6),
                             cashflows(FIXED, 'EUR', eur, every(12, 3), 1e6, currency='EUR')])
         for reference, usd, eur in (('USD_RECEIVED', 'Buy', 'Sell'), ('EUR_RECEIVED', 'Sell', 'Buy'))] + [
    package('FOREIGN', [cashflows(FLOATING, 'EUR', 'Buy', every(12, 3), 1e6, currency='EUR'),
                        cashflows(FIXED, 'GBP', 'Sell', every(12, 3), 9e5, currency='GBP')]),
    package('FLOATING', [cashflows(FLOATING, 'USD', 'Buy', every(12, 3), 1.25e6),
                         cashflows(FLOATING, 'EUR', 'Sell', every(12, 3), 1e6, currency='EUR')])]


def fx_swap(reference, near):
    """A swap buying a million euros for 1.3 million dollars on `near` and selling them back for 1.32
    million a year out."""
    return {'Object': 'FXSwapDeal', 'Reference': reference, 'Near_Settlement_Date': near,
            'Far_Settlement_Date': years(1), 'Near_Buy_Far_Sell_Ccy': 'EUR', 'Near_Sell_Far_Buy_Ccy': 'USD',
            'Near_Buy_Far_Sell_Discount_Rate': 'EUR', 'Near_Sell_Far_Buy_Discount_Rate': 'USD',
            'Near_Buy_Amount': 1e6, 'Near_Sell_Amount': 1.3e6, 'Far_Buy_Amount': 1.32e6, 'Far_Sell_Amount': 1e6}


def energy_leg(reference, side):
    """A leg on WTI's monthly average, `side` Receiver receiving it: a period paid a month ago; one
    from ten days ago to twenty days on of a thousand barrels, fixed so far at 79; then two thousand
    at a multiplier of a half and 1,500, each paid five days after its month."""
    def period(start, end, volume, multiplier=1.0, realised=0.0):
        return {'Payment_Date': end + pd.DateOffset(days=5), 'Period_Start': start, 'Period_End': end,
                'Volume': volume, 'Fixed_Basis': 0.0, 'Price_Multiplier': multiplier,
                'Realized_Average_Date': None, 'Realized_Average': realised, 'FX_Period_Start': start,
                'FX_Period_End': end, 'FX_Realized_Average': 0.0}
    month = pd.DateOffset(months=1)
    return {'Object': 'FloatingEnergyDeal', 'Reference': reference, 'Currency': 'USD', 'Discount_Rate': 'USD',
            'Sampling_Type': 'WTI', 'Reference_Type': 'WTI', 'Commodity': 'WTI', 'Payer_Receiver': side,
            'Payments': {'Items': [
                period(BASE - 2 * month, BASE - month - pd.DateOffset(days=10), 1e3),
                period(BASE - pd.DateOffset(days=10), BASE + pd.DateOffset(days=20), 1e3, realised=79.0),
                period(BASE + month, BASE + 2 * month, 2e3, 0.5), period(BASE + 2 * month, BASE + 3 * month, 1.5e3)]}}


def resetting(reference, side):
    """A swap from three months out for two years, quarterly, exchanging its principals at its start
    and end, whose euro leg's notional resets each quarter off 1.25 million dollars, paid where
    `side` is Pay, against a dollar leg fixed at 4% on 1.25 million."""
    dates = every(3, 8, BASE + pd.DateOffset(months=3))
    euro = cashflows(FLOATING, 'EUR', 'Sell' if side == 'Pay' else 'Buy', dates, 1.25e6, currency='EUR')
    euro['Cashflows']['Items'] = [dict(item, FX_Reset_Date=item['Accrual_Start_Date'])
                                  for item in euro['Cashflows']['Items']]
    paid, received = ('EUR', 'USD') if side == 'Pay' else ('USD', 'EUR')
    return {'Instrument': {'.Deal': {
        'Object': 'MtMCrossCurrencySwapDeal', 'Reference': reference, 'MtM_Side': side,
        'Pay_Currency': paid, 'Receive_Currency': received, 'Pay_Discount_Rate': paid,
        'Receive_Discount_Rate': received, 'Pay_Interest_Rate': paid, 'Receive_Interest_Rate': received,
        'Principal_Exchange': 'Start_Maturity',
        'Effective_Date': dates[0], 'Maturity_Date': dates[-1]}}, 'Children': [
        node(euro), node(cashflows(FIXED, 'USD', 'Buy' if side == 'Pay' else 'Sell', dates, 1.25e6))]}


def optioned(kind, reference, side, dates, digital=None):
    """A floating leg on a million over `dates`, received where `side` is Buy, paying a `kind`, Cap
    or Floor, at 4.5% on its rate, each optionlet a digital paying `digital` where one is stated."""
    leg = cashflows(FLOATING, reference, side, dates, 1e6, Forecast_Rate_Cap_Volatility='USD')
    leg['Cashflows']['Properties'] = [{'Digital_Payoff_Rate': digital, 'Cap_Strike': utils.Percent(4.5),
                                       'Floor_Strike': utils.Percent(4.5), 'Cap_Multiplier': float(kind == 'Cap'),
                                       'Floor_Multiplier': float(kind == 'Floor')}]
    return leg


def overnight(leg):
    """`leg` compounding a fixing a day over each of its periods, each period paid at its end."""
    day = pd.DateOffset(days=1)
    leg['Cashflows'] = dict(leg['Cashflows'], Compounding_Method='OIS', Items=[dict(
        item, Accrual_Start_Date=start, Accrual_End_Date=start + day, Accrual_Year_Fraction=1 / 365,
        Resets=[[start, start, start + day, 1 / 365] + item['Resets'][0][4:]])
        for item in leg['Cashflows']['Items']
        for start in pd.date_range(item['Accrual_Start_Date'], item['Accrual_End_Date'] - day)])
    return leg


def by_terms(kind, reference, side='Buy'):
    """A `kind`, CapDeal or FloorDeal, at 4.5% on a million from a year out to two, a period of a
    year, stated by its terms."""
    year = pd.DateOffset(months=12)
    return {'Object': kind, 'Reference': reference, 'Currency': 'USD', 'Discount_Rate': 'USD',
            'Forecast_Rate': 'USD', 'Forecast_Rate_Volatility': 'USD', 'Buy_Sell': side, 'Principal': 1e6,
            'Effective_Date': years(1), 'Maturity_Date': years(2), 'Payment_Interval': year,
            'Reset_Frequency': year, 'Index_Tenor': year, 'Cap_Rate' if kind == 'CapDeal' else 'Floor_Rate': 4.5}


def cap(reference, side, legs):
    """A cap stating `side` over the capped `legs`."""
    return {'Instrument': {'.Deal': by_terms('CapDeal', reference, side)}, 'Children': [node(leg) for leg in legs]}


#: A period of a year from a year out, and three such periods.
ONE_YEAR, THREE_YEARS = [years(1), years(2)], every(12, 3, years(1))
#: A cap of four half-years from three months ago, the first fixed at 5%.
STRIP = optioned('Cap', 'LEG', 'Buy', every(6, 4, BASE - pd.DateOffset(months=3)))
STRIP['Cashflows']['Items'][0]['Resets'][0][-1] = utils.Percent(5.0)
#: Caps on three legs: a year from a year out, a year from two years out and three months from three.
CAP_LEGS = [optioned('Cap', 'FIRST', 'Buy', ONE_YEAR), optioned('Cap', 'SECOND', 'Buy', THREE_YEARS[1:3]),
            optioned('Cap', 'SHORT', 'Buy', every(3, 1, BASE + pd.DateOffset(months=3)))]
#: The quarter from three months out capped at 4.5% on a million and the same leg uncapped, both
#: received, beside a leg paid on a million quarterly for nine months from the base date.
SHORT_LEGS = [optioned('Cap', 'CAPPED', 'Buy', every(3, 1, BASE + pd.DateOffset(months=3))),
              cashflows(FLOATING, 'FLOAT', 'Buy', every(3, 1, BASE + pd.DateOffset(months=3)), 1e6),
              cashflows(FLOATING, 'LONG', 'Sell', every(3, 3), 1e6)]


def binary(reference, kind='Call', expiry=None, payoff='Payoff', **terms):
    """A bought digital `kind` on the share struck at 50, paying 100,000 dollars, the field `payoff`
    states, at `expiry`, a year out where None."""
    return dict({'Object': 'EquityBinaryOption', 'Reference': reference, 'Currency': 'USD', 'Discount_Rate': 'USD',
                 'Equity': 'SHARE', 'Equity_Volatility': 'SHARE', 'Buy_Sell': 'Buy', 'Option_Type': kind,
                 'Strike_Price': 50.0, payoff: 1e5, 'Expiry_Date': expiry or years(1)}, **terms)


def touch(kind, reference, barrier, **terms):
    """A bought `kind` of touch paying 100,000 dollars a year out where `barrier` is touched from below."""
    return dict({'Object': kind, 'Reference': reference, 'Currency': 'USD', 'Discount_Rate': 'USD', 'Buy_Sell': 'Buy',
                 'Cash_Payoff': 1e5, 'Barrier_Type': 'Up', 'Barrier_Price': barrier, 'Expiry_Date': years(1)}, **terms)


#: The share digital, its barrier twin knocked out above 70, the share call struck at 55 and the
#: share's touches there, a WTI digital call struck at 90 paying 100,000 dollars a year out and its
#: touches there, the euro digital call in dollars paying 100,000 at 1.3, and the euro Asian call paying
#: its million dollars where its average ends above 1.3.
DIGITALS = [
    binary('CALL'), binary('BARRIER', payoff='Cash_Payoff', Object='EquityBarrierBinaryOption',
                           Barrier_Type='Up_And_Out', Barrier_Price=70.0, Barrier_Dates=[
                               [BASE + pd.DateOffset(months=k), None] for k in range(1, 13)]),
    binary('CALL_55', Strike_Price=55.0),
    *[touch(kind, reference, 55.0, Equity='SHARE', Equity_Volatility='SHARE')
      for kind, reference in (('EquityOneTouchOption', 'ONE_TOUCH'), ('EquityNoTouchOption', 'NO_TOUCH'))],
    {'Object': 'CommodityDigitalOption', 'Reference': 'WTI', 'Currency': 'USD', 'Discount_Rate': 'USD',
     'Reference_Type': 'WTI', 'Reference_Volatility': 'WTI', 'Buy_Sell': 'Buy', 'Option_Type': 'Call',
     'Strike_Price': 90.0, 'Payoff': 1e5, 'Expiry_Date': years(1)},
    *[touch(kind, reference, 90.0, Reference_Type='WTI', Reference_Volatility='WTI')
      for kind, reference in (('CommodityOneTouchOption', 'WTI_ONE'), ('CommodityNoTouchOption', 'WTI_NO'))],
    dict(MIXED[0], Object='FXBinaryOption', Reference='EURO', Payoff=1e5),
    dict(ASIANS[0], Is_Digital='Yes')]


def document(sets, calculation, vol=0.2):
    """The job of `sets`, `{reference: (agreement terms, [trade node])}`, each trade's reference
    prefixed with its set's, running `calculation` on the world, the receiver's surface at `vol`."""
    return json.loads(json.dumps({'Calc': {
        'Calculation': dict({'Base_Date': BASE, 'Currency': 'USD'}, **calculation),
        'Deals': {'Reference': 'saccr', 'Deals': {'Children': [{
            'Instrument': {'.Deal': dict({'Object': 'NettingCollateralSet', 'Reference': reference,
                                          'Netted': 'True', 'Collateralized': 'False'}, **terms)},
            'Children': held(reference, nodes)} for reference, (terms, nodes) in sets.items()]}},
        'MergeMarketData': {'MarketDataFile': '', 'ExplicitMarketData': {
            'System Parameters': {'Base_Currency': 'USD', 'Base_Date': BASE},
            'Valuation Configuration': {}, 'Price Factors': factors(vol)}}}}, cls=CustomJsonEncoder))


def run(job):
    context = derivus.Context()
    context.load_json((json.dumps(job), 'saccr'), compress=False)
    return context.run_job()[1]


def valued(nodes):
    """`{reference: value}` of `nodes`, held under a set `BOOK`, off one base valuation."""
    frame = run(document({'BOOK': ({}, nodes)}, {'Object': 'BaseValuation'}))['Results']['mtm']
    return dict(zip(frame['Reference'], frame['Value']))


def struck(deal, rate):
    """`deal` at `rate` percent: a swap's fixed rate, a default swap's coupon."""
    return dict(deal, **({'Swap_Rate': rate} if deal['Object'] == 'SwapInterestDeal' else {
        'Pay_Rate': utils.Basis(100.0 * rate)}))


def at_market(trades):
    """Each deal of `trades`, `[(deal, value in thousands)]`, struck where it is worth its value,
    the value being affine in its rate."""
    low, high = (valued([node(struck(deal, rate)) for deal, _ in trades]) for rate in (1.0, 2.0))
    worth = {deal['Reference']: (low['BOOK_' + deal['Reference']], high['BOOK_' + deal['Reference']])
             for deal, _ in trades}
    return [struck(deal, 1.0 + (1e3 * value - worth[deal['Reference']][0]) / (
        worth[deal['Reference']][1] - worth[deal['Reference']][0])) for deal, value in trades]


def receiver_vol():
    """The vol at which the receiver is worth its 50 thousand (CRE99.22): Black on its forward at
    6% over the annuity of its 30/360 leg on the euro curve, crossed at 1.25."""
    annuity = 4e6 * EURO * sum(1.06 ** -k for k in range(2, 12))

    def worth(vol):
        d1 = (math.log(0.06 / 0.05) + 0.5 * vol * vol) / vol
        return annuity * (0.05 * stats.norm.cdf(vol - d1) - 0.06 * stats.norm.cdf(-d1)) - 50e3

    return optimize.brentq(worth, 0.01, 2.0, xtol=1e-15)


def mapping_file(path, drop=()):
    """The fixture's mapping with the world's factors stated beside it, the asset class of each
    type in `drop` unstated, written to `path`."""
    with open(os.path.join(FIXTURES, 'simm_mapping.json'), 'rt', encoding='utf-8') as f:
        mapping = json.load(f)
    mapping['Factors'].update(STATED)
    for kind in drop:
        del mapping['SACCR']['Asset_Class'][kind]
    with open(path, 'wt', encoding='utf-8') as f:
        json.dump(mapping, f)
    return str(path)


def saccr(mapping, parameters=PARAMETERS):
    return {'Object': 'SACCR', 'Mapping': mapping, 'Parameters': parameters,
            'MCMC_Simulations': 1, 'Quasi_Seed': 1}


@pytest.fixture(scope='module')
def world(tmp_path_factory):
    """`(sets, output)`: the five examples beside their hedging sets alone and the module's other
    sets, `{reference: (agreement terms, [trade node])}`, one SA-CCR run over one book."""
    mapping = mapping_file(tmp_path_factory.mktemp('saccr') / 'mapping.json')
    linear = at_market(LINEAR)
    rates, credit = [node(deal) for deal in linear[:2]] + [RECEIVER], [node(deal) for deal in linear[2:]]
    forwards, five = [node(deal) for deal in FORWARDS], [node(swap('SWAP_5Y', years(5), 'Fixed'))]
    sets = {'EX1': ({}, rates), 'EX1_USD': ({}, rates[:2]), 'EX1_EUR': ({}, rates[2:]),
            'EX2': ({}, credit), 'EX3': ({}, forwards), 'EX3_ENERGY': ({}, forwards[:2]),
            'EX3_METALS': ({}, forwards[2:]), 'EX4': ({}, rates + credit),
            'EX5': (EXAMPLE_5, rates + forwards),
            'SHORT': ({}, [node(swap('WEEK', BASE + pd.DateOffset(days=7), 'Fixed'))]),
            'THRESHOLD': (THRESHOLD, five), 'OVER': (OVER, five), 'POSTED': (POSTED, five),
            'MIXED': ({}, [node(deal) for deal in MIXED]),
            'FX_LINEAR': ({}, [node(deal) for deal in FX_LINEAR]),
            'FX_SIGNED': ({}, [node(deal) for deal in FX_SIGNED]),
            'EXPIRED': ({}, [node(deal) for deal in EXPIRED]),
            'AXES': ({}, [node(deal) for deal in AXES]),
            'ALIAS': ({}, [node(dict(MIXED[0], Object='FXEuropeanOption'))]),
            'COMPOSED': ({}, [node(dict(MIXED[5], Commodity='SILVER.LBMA'))]),
            'SETTLED': ({}, SWAPTIONS),
            'EQUITY_OPTIONS': ({}, [node(deal) for deal in EQUITY_OPTIONS]),
            'ENERGY': ({}, [node(deal) for deal in ENERGY]),
            'ASIANS': ({}, [node(deal) for deal in ASIANS]),
            'BARRIERS': ({}, [node(deal) for deal in BARRIERS]),
            'SINGLES': ({}, [node(deal) for deal in SINGLES]),
            'KNOWN': ({}, [node(deal) for deal in KNOWN]),
            'KNOWN_PACKAGE': ({}, [package('PACKAGE', KNOWN[:2])]),
            'RECEIVED_LEG': ({}, [node(cashflows(FLOATING, 'LEG', 'Buy', STARTED, AMORTISED))]),
            'PAID_LEG': ({}, [node(cashflows(FLOATING, 'LEG', 'Sell', STARTED, AMORTISED))]),
            'SWAP_ALONE': ({}, [node(swap('SWAP', years(5), 'Fixed'))]),
            'SWAP_LEGS': ({}, [package('SWAP', SWAP_LEGS)]),
            'SWAP_AMOUNTS': ({}, [package('SWAP', SWAP_LEGS + AMOUNTS)]),
            'SWAP_NETTED': ({}, [node(swap('PAYER', years(5), 'Fixed')), package('RECEIVER', [
                dict(leg, Buy_Sell='Buy' if leg['Buy_Sell'] == 'Sell' else 'Sell') for leg in SWAP_LEGS])]),
            'TWO_ROWS': ({}, [package('PACKAGE', [cashflows(FLOATING, 'FIVE', 'Buy', every(12, 5), 1e7),
                                                  cashflows(FLOATING, 'TWO', 'Sell', every(12, 2), 5e6)])]),
            'BASIS_RECEIVED': ({}, BASIS[:1]), 'BASIS_PAID': ({}, BASIS[1:]), 'BASIS_NETTED': ({}, BASIS),
            'BASIS_BESIDE': ({}, BASIS[1:] + [node(swap('PAYER', years(3), 'Fixed'))]),
            'CCS_USD_RECEIVED': ({}, CROSS[:1]), 'CCS_EUR_RECEIVED': ({}, CROSS[1:2]),
            'CCS_NETTED': ({}, CROSS[:2]), 'CCS_FOREIGN': ({}, CROSS[2:3]), 'CCS_FLOATING': ({}, CROSS[3:]),
            'MTM_PAY': ({}, [resetting('MTM', 'Pay')]), 'MTM_RECEIVE': ({}, [resetting('MTM', 'Receive')]),
            'FX_SWAP': ({}, [node(fx_swap('SWAP', SIX_MONTHS))]),
            'FX_SWAP_SEASONED': ({}, [node(fx_swap('SWAP', BASE - pd.DateOffset(months=1)))]),
            'FX_SWAP_TODAY': ({}, [node(fx_swap('SWAP', BASE))]),
            'ENERGY_SWAP': ({}, [node(energy_leg('FLOATING', 'Receiver')), node(dict(KNOWN[2], Payer_Receiver='Payer'))]),
            'ENERGY_PAYER': ({}, [node(energy_leg('FLOATING', 'Payer'))]),
            'CAPLET': ({}, [node(optioned('Cap', 'LEG', 'Buy', ONE_YEAR))]),
            'CAPLET_SOLD': ({}, [node(optioned('Cap', 'LEG', 'Sell', ONE_YEAR))]),
            'CAPLET_SELL_CAP': ({}, [cap('CAP', 'Sell', [optioned('Cap', 'LEG', 'Buy', ONE_YEAR)])]),
            'FLOORLET': ({}, [node(optioned('Floor', 'LEG', 'Buy', ONE_YEAR))]),
            'OVERNIGHT': ({}, [node(overnight(optioned('Cap', 'LEG', 'Buy', every(3, 1, SIX_MONTHS))))]),
            'STRIP': ({}, [node(STRIP)]),
            'BY_TERMS': ({}, [node(by_terms('CapDeal', 'CAP')), node(by_terms('FloorDeal', 'FLOOR'))]),
            'CAP_LEGS': ({}, [cap('CAP', 'Buy', CAP_LEGS)]),
            'TRADE_PACKAGE': ({}, [package('TRADE', SHORT_LEGS)]),
            'TRADE_CAP': ({}, [cap('TRADE', 'Buy', SHORT_LEGS)]),
            'COLLAR': ({}, [node(optioned('Cap', 'CAP', 'Buy', THREE_YEARS)),
                            node(optioned('Floor', 'FLOOR', 'Sell', THREE_YEARS))]),
            'COLLAR_LEG': ({}, [node(cashflows(FLOATING, 'LEG', 'Buy', THREE_YEARS, 1e6))]),
            'DIGITAL': ({}, [node(binary('CALL'))]),
            'DIGITAL_NEAR': ({}, [node(binary('CALL', expiry=BASE + pd.DateOffset(days=2), Settlement_Date=years(1)))]),
            'DIGITAL_MARGINED': (THRESHOLD, [node(binary('CALL', expiry=BASE + pd.DateOffset(days=2),
                                                         Settlement_Date=years(1)))]),
            'DIGITAL_PAIR': ({}, [node(binary('CALL')), node(binary('PUT', 'Put'))]),
            **{name: ({}, [node(binary('CALL', expiry=BASE - pd.offsets.BDay(1), Settlement_Date=BASE + pd.DateOffset(
                days=7), Price_Fixing=[[BASE - pd.offsets.BDay(1), 50.5]], **terms))]) for name, terms in (
                    ('DIGITAL_FIXED', {}), ('ASSET_FIXED', {'Payoff': 1e3, 'Payoff_Style': 'Asset'}))},
            'ASSET': ({}, [node(binary(reference, kind, Payoff=1e3, Payoff_Style='Asset'))
                           for reference, kind in (('CALL', 'Call'), ('PUT', 'Put'))]),
            'DIGITALS': ({}, [node(deal) for deal in DIGITALS]),
            'TARF': ({}, [node(TARF)]),
            'TARF_EVEN': ({}, [node(dict(TARF, LeverageNotional=1e6, TARF_ExpiryDates=TARF['TARF_ExpiryDates'][1:]))]),
            'FORWARDS_EVEN': ({}, [node(outright('FWD_{:%m}'.format(day), (1e6, 'EUR'), (1.3e6, 'USD'), settles))
                                   for day, settles, _ in TARF['TARF_ExpiryDates'][1:]]),
            'ACCUMULATOR': ({}, [node(dict(STRIPS[1], Buy_Sell='Sell', Option_Type='Put'))]),
            'STRIP_SIGNS': ({}, [node(dict(strip, Reference=strip['Reference'] + legs, Strike_Price=strike, **{amount: 0.0}))
                                 for strip in STRIPS[:2] for legs, strike, amount in (
                                     ('_CALLS', 1.0, 'LeverageNotional'), ('_PUTS', 1.6, 'Underlying_Amount'))]),
            'RANGE': ({}, [node(STRIPS[3])]),
            'INFLATION': ({}, [node(LINKER)]),
            'INFLATION_FIXED': ({}, INFLATION_SWAPS[:1]), 'INFLATION_FLOATING': ({}, INFLATION_SWAPS[1:]),
            'INFLATION_NETTED': ({}, [node(LINKER), node(dict(LINKER, Reference='PAID', Buy_Sell='Sell'))]),
            'INFLATION_AMORTISING': ({}, [node(inflation('LEG', 'Buy', STARTED, AMORTISED))]),
            'SWAPLETS': ({}, [node(deal) for deal in SWAPLETS]), 'SWAPLETS_ENDED': ({}, [node(ENDED)]),
            'SWAPLET_PACKAGE': ({}, [package('TRS', [SWAPLETS[0], cashflows(
                FLOATING, 'FUNDING', 'Sell', HALVES, 4e6)])])}
    return sets, run(document(sets, saccr(mapping), receiver_vol()))


def netting_sets(output):
    return output['Results']['SACCR']['netting_sets'].set_index('Counterparty')


def trades(output, netting_set):
    """The rows of the trades `netting_set` holds, by the reference each is booked under."""
    frame = output['Results']['SACCR']['trades']
    found = frame[frame['NettingSet'] == netting_set]
    return found.set_index(found['TradeID'].str[len(netting_set) + 1:])


def printed(value, figure, places=0):
    """Whether `value` is the `figure` an example prints to `places` decimals."""
    return abs(value - figure) <= 0.5 * 10.0 ** -places


def table(rows, column, figures, scale=1e3, places=0):
    """Each trade of `rows`' `column` over `scale` against the `figures` printed for it."""
    found = {trade: rows.loc[trade, column] / scale for trade in figures}
    assert all(printed(found[trade], figure, places) for trade, figure in figures.items()), (
        column, found, figures)


def single_factor(add_ons, correlations):
    """CRE52.61(4): the systematic and idiosyncratic sums of `add_ons` at their `correlations`."""
    return (sum(rho * add_on for add_on, rho in zip(add_ons, correlations)) ** 2,
            sum((1.0 - rho * rho) * add_on ** 2 for add_on, rho in zip(add_ons, correlations)))


def test_the_rates_example_is_cre99s_to_its_printed_figures(world):
    """EXAMPLE 1 (CRE99.22-99.40), an unmargined set of two USD swaps and a EUR receiver: the
    supervisory durations 7.87, 3.63 and 7.49, the adjusted notionals 78,694, 36,254 and 37,428,
    every maturity factor one, the deltas 1, -1 and -0.2694 - the receiver a bought put at the 50%
    vol on its legs' forward swap rate of 6% against its 5% strike over a year - the effective
    notionals 78,694, -36,254 and -10,083 in buckets 3, 2 and 3 of USD and EUR; the hedging sets'
    effective notionals 59,270 and 10,083 and their add-ons 296 and 50, booked alone, the add-on 347,
    the RC 60 at a multiplier of one and the EAD 569; the run's stats carrying the parameters file's
    version.

    Killing mutations: the 1.4 cross-bucket term of CRE52.57 dropped - USD's add-on 433.2, the
    EAD 761.1; the long side read the other way up - the swap paying fixed at delta -1 and the
    receiver at +0.2694; a call read as a put - the receiver's delta +0.7306, the EAD 690.3.
    """
    _, output = world
    with open(PARAMETERS, 'rt', encoding='utf-8') as f:
        assert output['Stats']['Parameters'] == json.load(f)['version']
    found, rows = netting_sets(output), trades(output, 'EX1')
    notional = {'SWAP_10Y': 1e7, 'SWAP_4Y': 1e7, 'SWPT': 4e6 * EURO}
    durations = {trade: rows.loc[trade, 'AdjustedNotional'] / notional[trade] for trade in notional}
    assert all(printed(durations[trade], figure, 2) for trade, figure in (
        ('SWAP_10Y', 7.87), ('SWAP_4Y', 3.63), ('SWPT', 7.49))), durations
    table(rows, 'AdjustedNotional', {'SWAP_10Y': 78_694, 'SWAP_4Y': 36_254, 'SWPT': 37_428})
    table(rows, 'MaturityFactor', {'SWAP_10Y': 1, 'SWAP_4Y': 1, 'SWPT': 1}, 1.0, 4)
    table(rows, 'Delta', {'SWAP_10Y': 1, 'SWAP_4Y': -1, 'SWPT': -0.2694}, 1.0, 4)
    table(rows, 'EffectiveNotional', {'SWAP_10Y': 78_694, 'SWAP_4Y': -36_254, 'SWPT': -10_083})
    assert list(zip(rows['HedgingSet'], rows['Bucket'])) == [('USD', '3'), ('USD', '2'), ('EUR', '3')]
    usd, eur = (found.loc['EX1_' + currency, 'AddOn_IR'] / 1e3 for currency in ('USD', 'EUR'))
    assert printed(usd / 0.005, 59_270) and printed(eur / 0.005, 10_083), (usd, eur)
    assert printed(usd, 296) and printed(eur, 50), (usd, eur)
    ex1 = found.loc['EX1']
    assert ex1['AddOn_IR'] == ex1['AddOn_Aggregate'] == pytest.approx((usd + eur) * 1e3, rel=1e-15)
    assert printed(ex1['AddOn_Aggregate'] / 1e3, 347) and printed(ex1['V'] / 1e3, 60), ex1
    assert ex1['RC'] == ex1['V'] and ex1['Multiplier'] == 1.0 and printed(ex1['EAD'] / 1e3, 569), ex1


def test_the_credit_example_is_cre99s_to_its_printed_figures(world):
    """EXAMPLE 2 (CRE99.41-99.58), an unmargined set of default swaps, bought on Firm A rated AA,
    sold on Firm B rated BBB and bought on the investment-grade index: the supervisory durations
    2.79, 5.18 and 4.42, the adjusted and effective notionals 27,858, 51,836 and 44,240, signed 1,
    -1 and 1; the entity add-ons 106, -280 and 168 at the factors 0.38%, 0.54% and 0.38%; CRE99.54's
    table - the systematic terms 52.9, -140 and 134.5, the idiosyncratic weights 0.75, 0.75 and 0.36,
    the squares 11,207, 78,353 and 28,261, the idiosyncratic terms 8,405, 58,765 and 10,174 - the
    systematic sum 47.5, its square 2,253, the idiosyncratic sum 77,344 and the add-on 282; on V of
    -20 the RC nothing, the multiplier 0.965 and the EAD 381.

    Killing mutations: the long side read the other way up - Firm A's delta -1; the multiplier's
    exponent read without its floor's share, `(V - C) / (2 AddOn)` - the multiplier 0.96692 and
    the EAD 381.91.
    """
    _, output = world
    found, rows = netting_sets(output).loc['EX2'], trades(output, 'EX2')
    with open(PARAMETERS, 'rt', encoding='utf-8') as f:
        supervisory = json.load(f)['supervisory']['Credit']
    durations = {trade: rows.loc[trade, 'AdjustedNotional'] / 1e7 for trade in rows.index}
    assert all(printed(durations[trade], figure, 2) for trade, figure in (
        ('CDS_A', 2.79), ('CDS_B', 5.18), ('CDX', 4.42))), durations
    table(rows, 'AdjustedNotional', {'CDS_A': 27_858, 'CDS_B': 51_836, 'CDX': 44_240})
    table(rows, 'Delta', {'CDS_A': 1, 'CDS_B': -1, 'CDX': 1}, 1.0)
    table(rows, 'EffectiveNotional', {'CDS_A': 27_858, 'CDS_B': -51_836, 'CDX': 44_240})
    assert list(rows['Subclass']) == ['AA', 'BBB', 'IG'], rows['Subclass']
    add_ons = rows['SupervisoryFactor'] * rows['EffectiveNotional'] / 1e3
    rho = rows['Subclass'].map(lambda rating: supervisory[rating]['correlation'])
    for column, figures, places in (
            (add_ons, (106, -280, 168), 0), (rho * add_ons, (52.9, -140, 134.5), 1),
            (1.0 - rho * rho, (0.75, 0.75, 0.36), 2), (add_ons ** 2, (11_207, 78_353, 28_261), 0),
            ((1.0 - rho * rho) * add_ons ** 2, (8_405, 58_765, 10_174), 0)):
        assert all(printed(value, figure, places) for value, figure in zip(column, figures)), (
            list(column), figures)
    systematic, idiosyncratic = single_factor(add_ons, rho)
    assert printed((rho * add_ons).sum(), 47.5, 1) and printed(systematic, 2_253), systematic
    assert printed(idiosyncratic, 77_344), idiosyncratic
    assert found['AddOn_Credit'] / 1e3 == pytest.approx(math.sqrt(systematic + idiosyncratic),
                                                        rel=1e-14)
    assert printed(found['AddOn_Aggregate'] / 1e3, 282) and printed(found['V'] / 1e3, -20), found
    assert found['RC'] == 0.0 and printed(found['Multiplier'], 0.965, 3), found
    assert printed(found['EAD'] / 1e3, 381), found


def test_the_commodity_example_is_cre99s_to_its_printed_figures(world):
    """EXAMPLE 3 (CRE99.59-99.76), an unmargined set of three forwards whose adjusted notionals are
    their notionals: WTI long nine months at the maturity factor (9/12)^0.5, Brent short two years
    and silver long five, the effective notionals 8,660, -20,000 and 10,000; WTI and Brent the one
    commodity type, crude oil, in the energy set, silver the metals'; the type's effective notional
    -11,340 and silver's 10,000, their add-ons -2,041 and 1,800 at 18%, each set's add-on the one
    type's, 2,041 and 1,800 booked alone, the add-on 3,841, the RC 20 on V of 20 and the EAD 5,406.

    Killing mutation: the supervisory factor read from the wrong subclass, electricity's 40% for a
    metal - the metals set's add-on 4,000, the EAD 8,485.6.
    """
    _, output = world
    found, rows = netting_sets(output), trades(output, 'EX3')
    table(rows, 'AdjustedNotional', {'WTI': 10_000, 'BRENT': 20_000, 'SILVER': 10_000})
    assert rows.loc['WTI', 'MaturityFactor'] == pytest.approx(math.sqrt(9 / 12), rel=1e-15)
    table(rows, 'EffectiveNotional', {'WTI': 8_660, 'BRENT': -20_000, 'SILVER': 10_000})
    assert list(zip(rows['HedgingSet'], rows['Bucket'], rows['SupervisoryFactor'])) == [
        ('Energy', '2', 0.18), ('Energy', '2', 0.18), ('Metals', '12', 0.18)]
    crude = rows.loc[['WTI', 'BRENT'], 'EffectiveNotional'].sum() / 1e3
    assert printed(crude, -11_340) and printed(0.18 * crude, -2_041), crude
    energy, metals = (found.loc['EX3_' + name, 'AddOn_Commodity'] / 1e3 for name in (
        'ENERGY', 'METALS'))
    assert printed(energy, 2_041) and printed(metals, 1_800), (energy, metals)
    ex3 = found.loc['EX3']
    assert ex3['AddOn_Commodity'] == pytest.approx((energy + metals) * 1e3, rel=1e-15)
    assert printed(ex3['AddOn_Aggregate'] / 1e3, 3_841) and printed(ex3['RC'] / 1e3, 20), ex3
    assert ex3['Multiplier'] == 1.0 and printed(ex3['EAD'] / 1e3, 5_406), ex3


def test_the_rates_and_credit_example_is_cre99s_to_its_printed_figures(world):
    """EXAMPLE 4 (CRE99.77-99.79), examples 1 and 2 under one unmargined set: RC 40, the add-ons
    347 and 282 summed across their classes with no diversification, 629, the multiplier one and
    the EAD 936.

    Killing mutation: the aggregate taken as the larger class's add-on - 346.8, the EAD 541.5.
    """
    _, output = world
    found = netting_sets(output)
    ex4 = found.loc['EX4']
    assert (ex4['AddOn_IR'], ex4['AddOn_Credit']) == (found.loc['EX1', 'AddOn_IR'],
                                                      found.loc['EX2', 'AddOn_Credit'])
    assert ex4['AddOn_Aggregate'] == ex4['AddOn_IR'] + ex4['AddOn_Credit']
    assert printed(ex4['RC'] / 1e3, 40) and printed(ex4['AddOn_Aggregate'] / 1e3, 629), ex4
    assert ex4['Multiplier'] == 1.0 and printed(ex4['EAD'] / 1e3, 936), ex4


def test_the_margined_example_is_cre99s_to_its_printed_figures(world):
    """EXAMPLE 5 (CRE99.80-99.97), examples 1 and 3 under a set margined weekly: no threshold, a
    minimum transfer of 5, an independent amount of 150 received and 200 of collateral held, the RC
    max(80 - 200, 0 + 5 - 150, 0) nothing; the MPOR 14, nine business days and the five of its
    weekly margin, the maturity factor 1.5 (14/250)^0.5 for every trade; the rate trades'
    effective notionals 27,934, -12,869 and -3,579 and their add-on 123; the forwards' 3,550,
    -7,100 and 3,550 - the text's Brent 0.355 x 20,000, the factor rounded, held to a unit - crude's
    -3,550 and silver's 3,550, their add-ons -639 and 639 and the commodity add-on 1,278; the
    aggregate 1,401, V - C -120, the multiplier 0.958 and the EAD 1,879.

    Killing mutations: the margined maturity factor without its 1.5 of CRE52.52 - the aggregate
    934.0, the EAD 1,226.3; the set's call days run to the last of its revaluation dates, which a
    base valuation finalises none of - the set skipped, no trade of it valued and the EAD nothing.
    """
    _, output = world
    found, rows = netting_sets(output).loc['EX5'], trades(output, 'EX5')
    assert found['Margined'] and found['MPOR'] == 14, found
    assert (rows['MaturityFactor'] == 1.5 * math.sqrt(14 / 250)).all(), rows['MaturityFactor']
    table(rows, 'EffectiveNotional', {'SWAP_10Y': 27_934, 'SWAP_4Y': -12_869, 'SWPT': -3_579,
                                      'WTI': 3_550, 'SILVER': 3_550})
    assert abs(rows.loc['BRENT', 'EffectiveNotional'] / 1e3 + 7_100) < 1.0, rows.loc['BRENT']
    crude = rows.loc[['WTI', 'BRENT'], 'EffectiveNotional'].sum() / 1e3
    assert printed(crude, -3_550) and printed(0.18 * crude, -639), crude
    assert printed(0.18 * rows.loc['SILVER', 'EffectiveNotional'] / 1e3, 639)
    assert printed(found['C'] / 1e3, 200) and printed(found['V'] / 1e3, 80), found
    assert printed((found['V'] - found['C']) / 1e3, -120) and found['RC'] == 0.0, found
    assert printed(found['AddOn_IR'] / 1e3, 123) and printed(found['AddOn_Commodity'] / 1e3, 1_278)
    assert printed(found['AddOn_Aggregate'] / 1e3, 1_401), found
    assert printed(found['Multiplier'], 0.958, 3) and printed(found['EAD'] / 1e3, 1_879), found


def multiplier(v, c, add_on):
    """CRE52.23 as it is printed: `min(1, F + (1 - F) exp((V - C) / (2 (1 - F) AddOn)))`."""
    return min(1.0, 0.05 + 0.95 * math.exp((v - c) / (2.0 * 0.95 * add_on)))


def supervisory_duration(start, end):
    """CRE52.34 as it is printed, unfloored."""
    return (math.exp(-0.05 * start) - math.exp(-0.05 * end)) / 0.05


def test_a_trade_inside_ten_business_days_is_floored_there_twice(world):
    """CRE52.34 AND CRE52.48. A swap maturing a week out, 7/360 years, has a supervisory duration
    of 0.0194 and a remaining maturity below ten business days: each is floored at 10/250, the
    adjusted notional 400,000 on its ten million and the maturity factor 0.2, its add-on 400 and
    its EAD 1.4 times its value and that.

    Killing mutations: the duration unfloored - the adjusted notional 194,350; the maturity factor
    unfloored - 0.1394.
    """
    _, output = world
    found, rows = netting_sets(output).loc['SHORT'], trades(output, 'SHORT').loc['WEEK']
    assert rows['AdjustedNotional'] == pytest.approx(1e7 * 10 / 250, rel=1e-15), rows
    assert rows['MaturityFactor'] == pytest.approx(math.sqrt(10 / 250), rel=1e-15), rows
    assert found['AddOn_IR'] == pytest.approx(0.005 * 1e7 * 0.04 * 0.2, rel=1e-14), found
    assert found['EAD'] == pytest.approx(1.4 * (max(found['V'], 0.0) + found['AddOn_IR']),
                                         rel=1e-14), found


def test_a_margined_set_past_its_threshold_is_capped_at_its_unmargined_exposure(world):
    """CRE52.18, CRE52.50 AND CRE52.2. A payer five years on ten million under a threshold of a
    million and a minimum transfer of 50 thousand, over 200 thousand of independent amount held: its
    MPOR of five floored at ten business days, its maturity factor 0.3, its RC the 850 thousand of
    TH + MTA - NICA where V - C is -191 thousand; margined its EAD is 1.4 (850,000 + PFE), above
    the 1.4 times the PFE of the set unmargined at a maturity factor of one, so the EAD is that.

    Killing mutations: the margined RC without TH + MTA - NICA - the RC nothing and the EAD 24,057;
    the cap of CRE52.2 dropped - the EAD 1,214,057; the MPOR unfloored - the maturity factor 0.2121.
    """
    _, output = world
    found, rows = netting_sets(output).loc['THRESHOLD'], trades(output, 'THRESHOLD').loc['SWAP_5Y']
    v, c = found['V'], found['C']
    adjusted = 1e7 * supervisory_duration(0.0, 5.0)
    assert found['Margined'] and found['MPOR'] == 10 and c == 200e3, found
    assert rows['AdjustedNotional'] == pytest.approx(adjusted, rel=1e-14), rows
    assert rows['MaturityFactor'] == pytest.approx(0.3, rel=1e-15), rows
    assert found['RC'] == 850e3, found
    margined_pfe, unmargined_pfe = (multiplier(v, c, add_on) * add_on for add_on in (
        0.005 * adjusted * 0.3, 0.005 * adjusted))
    assert 1.4 * unmargined_pfe < 1.4 * (850e3 + margined_pfe)
    assert found['PFE'] == pytest.approx(margined_pfe, rel=1e-13), found
    assert found['EAD'] == pytest.approx(1.4 * unmargined_pfe, rel=1e-13), found


def test_a_margined_sets_collateral_is_its_balance_as_the_engine_counts_it(world):
    """CRE52.10-52.11. The payer under a set holding four million euros of cash, cut 8% at a
    closeout: C is 4,600,000, the euros crossed at 1.25 and held at one less their received cut;
    under one that posted a million dollars cut 10% on delivery, C is -1,111,111.11, posted at face
    over one less its posted cut, and its RC is V - C.

    Killing mutations: the received cut ignored - C 5,000,000; the balance read in the agreement's
    currency, uncrossed - C 3,680,000; the posted cut ignored - C -1,000,000.
    """
    _, output = world
    found = netting_sets(output)
    assert found.loc['OVER', 'C'] == pytest.approx(4e6 * 1.25 * 0.92, rel=1e-15), found.loc['OVER']
    posted = found.loc['POSTED']
    assert posted['C'] == pytest.approx(-1e6 / 0.9, rel=1e-15), posted
    assert posted['RC'] == posted['V'] - posted['C'], posted


def test_an_over_collateralised_set_keeps_its_add_on_at_the_five_percent_floor(world):
    """CRE52.22-52.23. The payer under the set holding 4.6 million of collateral and no margin
    terms: V - C is 69 times its margined add-on below zero, so the multiplier is its floor of 5%
    within an ulp and its PFE 5% of its add-on, the RC nothing and the EAD 1.4 times that.

    Killing mutation: the multiplier's floor dropped - the multiplier 9.5e-16 and the PFE 6.3e-11.
    """
    _, output = world
    found = netting_sets(output).loc['OVER']
    assert found['RC'] == 0.0, found
    assert found['Multiplier'] == pytest.approx(0.05, rel=1e-15), found
    assert found['PFE'] == pytest.approx(0.05 * found['AddOn_Aggregate'], rel=1e-14), found
    assert found['EAD'] == pytest.approx(1.4 * found['PFE'], rel=1e-15), found


def black_delta(forward, strike, vol, expiry, call):
    """CRE52.40's delta of a bought option."""
    d1 = (math.log(forward / strike) + 0.5 * vol * vol * expiry) / (vol * math.sqrt(expiry))
    return stats.norm.cdf(d1) if call else -stats.norm.cdf(-d1)


def between(expiry, days=2):
    """The years from the base date's spot date to `expiry`'s, `days` business days on each."""
    return ((expiry - BASE).days + lag(expiry, days) - lag(BASE, days)) / 365


def test_the_fx_equity_and_commodity_classes_and_an_amortising_notional_read_by_hand(world):
    """CRE52.35, CRE52.36, CRE52.37(2), CRE52.59 AND CRE52.66. One unmargined set: a bought euro
    call in dollars, its notional the million euros crossed, its delta Black on the pair's forward
    a year out between its spot dates, 1.25 e^((0.04 - ln 1.06) 367/365), at the 15% vol; a sold
    dollar call in euros, its foreign leg the 750 thousand euros its strike pays, its delta taken
    on EUR/USD as its surface quotes the pair (52.59 FAQ1), a sold euro put struck at 1/0.75 on
    the forward 1.25 e^((0.04 - ln 1.06) 181/365) six months out, +0.7392; the two summed in the
    one pair at 4%. A forward on an index, Risk_Equity bucket 11, at 20% and a correlation of 80%,
    and a sold one on a share at 32% and 50%, aggregated as CRE52.66(4) states; a payer amortising
    two million a year, its notional averaged over its life, six million; and a future short
    40,000 units of silver at 25, its delta -1 by the sign of its units. Each class's add-on, the
    aggregate, the multiplier on V and the EAD by hand.

    Killing mutations: the pair's ordering ignored - the FX add-on 666.4 against 38,535.1; the index
    bucket rule dropped, the index a single name - the equity add-on 3,116,056 against 1,862,742; the
    amortising principal read whole - its adjusted notional 44,239,843 against 26,543,906.
    """
    _, output = world
    found, rows = netting_sets(output).loc['MIXED'], trades(output, 'MIXED')
    euro = black_delta(1.25 * math.exp((0.04 - math.log(1.06)) * between(years(1))), 1.3, 0.15, 1.0, True)
    dollar = -black_delta(1.25 * math.exp((0.04 - math.log(1.06)) * between(SIX_MONTHS)), 1 / 0.75,
                          0.15, 0.5, False)
    assert rows.loc['EUR_CALL', 'AdjustedNotional'] == 1.25e6, rows.loc['EUR_CALL']
    assert rows.loc['USD_CALL', 'AdjustedNotional'] == 0.75e6 * 1.25, rows.loc['USD_CALL']
    assert rows.loc['EUR_CALL', 'Delta'] == pytest.approx(euro, rel=1e-12), rows.loc['EUR_CALL']
    assert rows.loc['USD_CALL', 'Delta'] == pytest.approx(dollar, rel=1e-12), rows.loc['USD_CALL']
    assert set(rows.loc[['EUR_CALL', 'USD_CALL'], 'HedgingSet']) == {'EUR/USD'}
    fx = 0.04 * abs(1.25e6 * euro + 0.9375e6 * dollar * math.sqrt(0.5))
    assert found['AddOn_FX'] == pytest.approx(fx, rel=1e-12), found
    assert list(rows.loc[['INDEX_FWD', 'SHARE_FWD'], 'Subclass']) == ['Index', 'Single']
    equity = math.sqrt(sum(single_factor([0.2 * 1e7, -0.32 * 5e6 * math.sqrt(0.5)], [0.8, 0.5])))
    assert found['AddOn_Equity'] == pytest.approx(equity, rel=1e-14), found
    amortising = 6e6 * supervisory_duration(0.0, 5.0)
    assert rows.loc['AMORTISING', 'AdjustedNotional'] == pytest.approx(amortising, rel=1e-14)
    assert found['AddOn_IR'] == pytest.approx(0.005 * amortising, rel=1e-14), found
    future = rows.loc['SILVER_FUT']
    assert (future['AdjustedNotional'], future['Delta'], future['HedgingSet']) == (1e6, -1.0,
                                                                                   'Metals')
    assert found['AddOn_Commodity'] == pytest.approx(0.18e6, rel=1e-15), found
    aggregate = fx + equity + 0.005 * amortising + 0.18e6
    assert found['AddOn_Aggregate'] == pytest.approx(aggregate, rel=1e-14), found
    assert found['Multiplier'] == pytest.approx(multiplier(found['V'], 0.0, aggregate), rel=1e-14)
    assert found['EAD'] == pytest.approx(1.4 * (max(found['V'], 0.0) + found['PFE']), rel=1e-15)


def test_an_fx_option_takes_its_delta_on_the_pair_as_its_surface_quotes_it(world):
    """CRE52.59 FAQ1 AND CRE52.40. A call on EUR/USD is a put on USD/EUR, whose delta is not the
    call's; each pair is read in one order, the one its options are quoted in, which a book states
    as the name of the vol surface it quotes them on. A bought euro call in dollars struck at 1.3 a
    year out, and the same written as a bought dollar put in euros struck at 1/1.3 on 1.3 million
    dollars: each Black on EUR/USD's forward 1.25 e^((0.04 - ln 1.06) 367/365) at 1.3, its adjusted
    notional the million euros crossed. A bought dollar call in yen struck at 150 a year out on a
    million dollars, and the same written as a bought yen put in dollars at 1/150 on 150 million
    yen, the surface USD.JPY: each Black on USD/JPY's forward 150 e^((0.01 - 0.04) 367/365) at 150,
    0.4498, short the pair as its hedging set orders it, JPY/USD, so -0.4498, its adjusted notional
    the yen leg's million dollars.

    Killing mutations: the delta on the side of the pair the deal is written on - the dollar put's
    0.3231 against the call's 0.3787; the pair read in the order its hedging set sorts it - the
    dollar call's -0.3912 against -0.4498.
    """
    _, output = world
    rows = trades(output, 'AXES')
    euro = black_delta(1.25 * math.exp((0.04 - math.log(1.06)) * between(years(1))), 1.3, 0.15, 1.0, True)
    dollar = black_delta(YEN * math.exp((0.01 - 0.04) * between(years(1))), YEN, 0.15, 1.0, True)
    assert list(rows['HedgingSet']) == ['EUR/USD', 'EUR/USD', 'JPY/USD', 'JPY/USD'], rows
    assert rows['Delta'].tolist() == pytest.approx([euro, euro, -dollar, -dollar], rel=1e-12), rows
    assert rows['AdjustedNotional'].tolist() == pytest.approx([1e6 * EURO] * 2 + [1e6] * 2,
                                                              rel=1e-15), rows


def test_an_fx_forward_reads_its_foreign_leg_the_larger_where_both_are(world):
    """CRE52.35, CRE52.39, CRE52.48 AND CRE52.59. One unmargined set: a forward buying a million
    euros for 1.3 million dollars six months out, its adjusted notional the euros crossed,
    1,250,000, long the pair EUR/USD at the maturity factor (1/2)^0.5; one buying the dollars back
    for the million euros a year out, its foreign leg the euros it sells, 1,250,000, short the
    pair; and a non-deliverable forward buying a million euros for 900 thousand pounds a year out,
    settled in dollars outside its pair EUR/GBP, both legs foreign and the larger the pounds',
    1,350,000, long. The add-on 4% of |1,250,000 (1/2)^0.5 - 1,250,000| plus 4% of 1,350,000.

    Killing mutations: the smaller leg where both are foreign - the forward's adjusted notional
    1,250,000; the amount's leg alone - the run refused, the dollar buyer reading no foreign leg;
    the side not turned to the pair's order - the dollar buyer long, its effective notional
    +1,250,000.
    """
    _, output = world
    found, rows = netting_sets(output).loc['FX_LINEAR'], trades(output, 'FX_LINEAR')
    assert list(rows['HedgingSet']) == ['EUR/USD', 'EUR/USD', 'EUR/GBP'], rows['HedgingSet']
    assert list(rows['AdjustedNotional']) == [1e6 * EURO, 1e6 * EURO, 0.9e6 * STERLING], rows
    assert list(rows['Delta']) == [1.0, -1.0, 1.0], rows['Delta']
    assert rows['MaturityFactor'].tolist() == pytest.approx([math.sqrt(0.5), 1.0, 1.0], rel=1e-15)
    fx = 0.04 * abs(1.25e6 * math.sqrt(0.5) - 1.25e6) + 0.04 * 1.35e6
    assert found['AddOn_FX'] == pytest.approx(fx, rel=1e-14), found
    assert found['EAD'] == pytest.approx(1.4 * (max(found['V'], 0.0) + multiplier(
        found['V'], 0.0, fx) * fx), rel=1e-14), found


def test_an_fx_forward_stating_both_amounts_negative_is_the_opposite_trade_of_the_same_size(world):
    """CRE52.35 AND CRE52.39. `FX_LINEAR`'s three trades with both amounts negative, as a book of
    record states a trade the other way: each adjusted notional is its twin's, 1,250,000, 1,250,000
    and 1,350,000, and each delta the opposite, short, long and short.

    Killing mutation: the second leg read with its sign - the dollar seller's foreign leg then
    -1,250,000 and the non-deliverable forward's the smaller, 1,250,000.
    """
    _, output = world
    rows, twins = trades(output, 'FX_SIGNED'), trades(output, 'FX_LINEAR')
    assert list(rows['AdjustedNotional']) == list(twins['AdjustedNotional']), rows['AdjustedNotional']
    assert list(rows['Delta']) == [-1.0, 1.0, -1.0], rows['Delta']


def test_an_fx_option_at_or_past_its_expiry_is_its_deltas_limit(world):
    """CRE52.31, CRE52.40 AND CRE52.48. T runs to the latest exercise date, nothing once that is
    reached, and there 52.40's delta is its limit, one in the money and nothing out of it. On their
    expiry day a call and a put struck at 1.3 on a million euros, P the spot of 1.25: the call out
    of the money, its delta nothing, the put in, -1; a call on two million expired the business day
    before at a print of 1.31, P the print its pricer holds it at, +1. Each delivers inside ten
    business days, its maturity factor (10/250)^0.5 = 0.2; the adjusted notionals 1,250,000 and
    2,500,000, and the add-on 4% of |-1,250,000 x 0.2 + 2,500,000 x 0.2|, 10,000.

    Killing mutations: d1 as printed at T nothing - the run refused, a division by zero; P the
    forward for an expiry behind the base date - the exercised call out of the money at 1.25, its
    delta nothing.
    """
    _, output = world
    found, rows = netting_sets(output).loc['EXPIRED'], trades(output, 'EXPIRED')
    assert list(rows['Delta']) == [0.0, -1.0, 1.0], rows['Delta']
    assert list(rows['AdjustedNotional']) == [1e6 * EURO, 1e6 * EURO, 2e6 * EURO], rows
    assert rows['MaturityFactor'].tolist() == pytest.approx([math.sqrt(10 / 250)] * 3, rel=1e-15)
    assert found['AddOn_FX'] == pytest.approx(0.04 * abs(-1.25e6 * 0.2 + 2.5e6 * 0.2), rel=1e-14)


def test_an_fx_european_option_reads_as_the_fx_option_it_is(world):
    """`FXEuropeanOption` is `FXOptionDeal` under another name: the bought euro call booked as one
    reads the row it reads booked as the other, value and all.

    Killing mutation: the alias declaring no exposure - the run refused by name.
    """
    _, output = world
    columns = ['AssetClass', 'HedgingSet', 'AdjustedNotional', 'Delta', 'MaturityFactor',
               'EffectiveNotional', 'Value']
    assert list(trades(output, 'ALIAS').loc['EUR_CALL', columns]) == list(
        trades(output, 'MIXED').loc['EUR_CALL', columns])


def test_a_composed_commodity_reads_its_primarys_type_at_its_own_price(world):
    """CRE52.36 AND CRE52.69-52.71. A future short 40,000 units of `SILVER.LBMA`, silver at 25 and
    its LBMA basis at -0.5: its commodity type is silver's, the one the mapping states for the
    primary, bucket 12 in the metals set, and a unit is the composed price the trade references,
    24.5, its adjusted notional 980,000.

    Killing mutations: the mapping read under the composed name - the run refused for want of
    its SIMM.Bucket; the unit priced at its primary alone - 1,000,000.
    """
    _, output = world
    row = trades(output, 'COMPOSED').loc['SILVER_FUT']
    assert (row['HedgingSet'], row['Bucket']) == ('Metals', '12'), row
    assert (row['AdjustedNotional'], row['Delta']) == (4e4 * 24.5, -1.0), row


def test_a_cash_settled_swaption_matures_at_its_exercise_date(world):
    """CRE52.31, CRE52.32 TABLE 1 AND CRE52.48. Table 1's European swaption six months into a
    five-year swap: settled physically, M is the swap's end, 5.5 years, its maturity factor one;
    settled in cash, M is the exercise date, the start of the swap it fixes, half a year, its
    maturity factor (1/2)^0.5. Each reads S 0.5 and E 5.5 alike, the adjusted notional four million
    euros crossed times SD = (e^(-0.025) - e^(-0.275)) / 0.05, in bucket 3, and the one delta.

    Killing mutation: M read as E whatever the settlement - the cash-settled maturity factor one.
    """
    _, output = world
    rows = trades(output, 'SETTLED')
    assert rows['MaturityFactor'].tolist() == pytest.approx([1.0, math.sqrt(0.5)], rel=1e-15), rows
    assert rows['AdjustedNotional'].tolist() == pytest.approx(
        [4e6 * EURO * supervisory_duration(0.5, 5.5)] * 2, rel=1e-14), rows
    assert list(rows['Bucket']) == ['3', '3'] and rows['Delta'].nunique() == 1, rows


def test_an_equity_option_takes_its_delta_on_the_shares_forward(world):
    """CRE52.36, CRE52.40 AND ITS FAQ1, CRE52.42(1). Each option's adjusted notional is its units at
    the share's price, its delta Black at Table 2's vol on the share's forward to its expiry
    between its spot dates, three business days on, carried at the 4% repo less the 2% dividend
    yield (FAQ1: the forward, for the rate and the dividends). A bought call on a thousand units of
    the index at 4,000, struck at 4,200 a year out at the index's 75%: 4,000,000 and +0.6318. A sold
    put on twenty thousand shares at 50, struck at 45 six months out at a single name's 120%:
    1,000,000, +0.2877, long the share, and its maturity factor (1/2)^0.5. The index call paid in
    euros on S X (a compo), struck at 3,360 euros: its unit still the index at 4,000 dollars
    (52.36), its P the index forward times the dollar's in euros, 0.8 e^((ln 1.06 - 0.04) 367/365),
    +0.6410.

    Killing mutations: P the spot - the index call's delta +0.6217; a compo's P left in the
    share's currency against its strike in euros - +0.7370; the type declaring no exposure - the
    run refused by name.
    """
    _, output = world
    rows = trades(output, 'EQUITY_OPTIONS')
    index = 4000.0 * math.exp(0.02 * between(years(1), 3))
    dollar = math.exp((math.log(1.06) - 0.04) * between(years(1))) / EURO
    expected = [black_delta(index, 4200.0, 0.75, 1.0, True),
                -black_delta(50.0 * math.exp(0.02 * between(SIX_MONTHS, 3)), 45.0, 1.2, 0.5, False),
                black_delta(index * dollar, 3360.0, 0.75, 1.0, True)]
    assert rows['Delta'].tolist() == pytest.approx(expected, rel=1e-12), rows
    assert list(rows['AdjustedNotional']) == [4e6, 1e6, 4e6], rows
    assert rows['MaturityFactor'].tolist() == pytest.approx([1.0, math.sqrt(0.5), 1.0], rel=1e-15)
    assert list(rows['Subclass']) == ['Index', 'Single', 'Index'], rows


def test_an_energy_option_takes_its_delta_on_its_averages_current_value(world):
    """CRE52.36, CRE52.42(1) AND CRE52.31. An energy option is on the average of its reference's
    fixings over its period: its P is that average's current value, every weekday of the period
    weighted alike, one fixed at the realised average stated for those fixed and one to come at the
    forward of its own delivery off WTI's curve, 80 today rising 6% over six years; T is the
    period's end, M the settlement five days on, d its barrels at today's 80, its type crude oil
    at Table 2's 70%. The average over the month from a year out, 80.8326, under a bought call
    struck at 82: +0.6348 on 80,000. The average from ten days ago to twenty days on, its first seven fixings at
    79 and the rest to come, 79.6965, under a sold put struck at 80 over 19/360: +0.4774, long the
    oil, on 160,000 at the maturity factor (24/360)^0.5.

    Killing mutations: P the forward at the period's end - the call's delta +0.6350, the put's
    +0.4666; the fixed samples read at their deliveries' forwards - the put's +0.4675; the
    settlement left undeclared - the run refused, its tenor naming no day.
    """
    _, output = world
    rows = trades(output, 'ENERGY')
    six = (years(6) - BASE).days

    def average(deal):
        days = pd.bdate_range(deal['Period_Start'], deal['Period_End'])
        return sum(deal['Realized_Average'] if day < BASE else 80.0 * (
            1.0 + 0.06 * (day - BASE).days / six) for day in days) / len(days)

    january, running = ENERGY
    assert rows['Delta'].tolist() == pytest.approx([
        black_delta(average(january), 82.0, 0.7, 1.0 + 1.0 / 12.0, True),
        -black_delta(average(running), 80.0, 0.7, 19.0 / 360.0, False)], rel=1e-12), rows
    assert list(rows['AdjustedNotional']) == [8e4, 1.6e5], rows
    assert rows['MaturityFactor'].tolist() == pytest.approx([1.0, math.sqrt(24.0 / 360.0)], rel=1e-15)
    assert set(zip(rows['HedgingSet'], rows['Bucket'], rows['SupervisoryFactor'])) == {
        ('Energy', '2', 0.18)}, rows


def test_an_asian_option_takes_its_delta_on_its_averages_current_value(world):
    """CRE52.35, CRE52.36 AND CRE52.42(1). An option on a discrete average reads P as the average's
    current value: each sample its table states fixed at its price, each to come at the forward to
    its day, by its weight over all the weights; else it reads as its vanilla. The euro call in
    dollars struck at 1.3 a year out, on a fixing at 1.24 and EUR/USD's forwards six months and a
    year out, 1.2353: +0.3954 on its million euros crossed, 1,250,000. The sold put on a thousand
    units of the index struck at 3,900 six months out, on a fixing at 3,950 weighing twice and the
    index's forwards three and six months out, 3,989.69: +0.3790, long the index, on 4,000,000.

    Killing mutations: P the forward at the expiry - the call's delta +0.3787; the weights left
    unnormalised - the put's P 15,958.8 and its delta +0.0017.
    """
    _, output = world
    rows = trades(output, 'ASIANS')
    euro = [1.24] + [1.25 * math.exp((0.04 - math.log(1.06)) * between(day)) for day in (SIX_MONTHS, years(1))]
    index = [4000.0 * math.exp(0.02 * between(day, 3)) for day in (BASE + pd.DateOffset(months=3), SIX_MONTHS)]
    assert rows['Delta'].tolist() == pytest.approx([
        black_delta(sum(euro) / 3.0, 1.3, 0.15, 1.0, True),
        -black_delta((2.0 * 3950.0 + sum(index)) / 4.0, 3900.0, 0.75, 0.5, False)], rel=1e-12), rows
    assert list(rows['AdjustedNotional']) == [1e6 * EURO, 4e6], rows


def test_a_barrier_option_reads_as_its_vanilla(world):
    """CRE52.40 AND CRE52.42(1); A CONVENTION, CRE52 STATING NO BARRIER'S DELTA. A barrier option
    is read by CRE52.40 on its own strike, expiry and underlying, its barrier unread: the US
    agencies hold a path-dependent option to the supervisory delta, no model of its own (the
    SA-CCR final rule's preamble, 85 FR 4385-4386), and the UAE central bank's SA-CCR guidance
    applies the standardised delta to barrier options (Q&A D.5). The euro call knocked out above
    1.6 reads the vanilla euro call's row, +0.3787 on 1,250,000, and sold knocked in below 1.1 over
    a window its negative; the index call knocked in by a close behind the base date is the
    vanilla it became, +0.6318; the sold share put knocked out below 40, +0.2877 at (1/2)^0.5. A
    bought put on a thousand barrels of WTI struck at 81 a year out reads WTI's forward for the
    delivery its fixing curve maps the expiry to, 80 (1 + 0.06 365/2191): -0.3645 on 80,000.

    Killing mutations: an FX barrier declaring no exposure - the run refused by name; P read at
    the reference price's delivery today - the WTI put's delta -0.3698.
    """
    _, output = world
    rows, vanillas = trades(output, 'BARRIERS'), trades(output, 'MIXED').loc['EUR_CALL']
    equities = trades(output, 'EQUITY_OPTIONS')
    expected = [vanillas['Delta'], -vanillas['Delta'], equities.loc['INDEX_CALL', 'Delta'],
                equities.loc['SHARE_PUT', 'Delta'],
                black_delta(80.0 * (1.0 + 0.06 * 365 / (years(6) - BASE).days), 81.0, 0.7, 1.0, False)]
    assert rows['Delta'].tolist() == pytest.approx(expected, rel=1e-13), rows
    assert list(rows['AdjustedNotional']) == [1e6 * EURO] * 2 + [4e6, 1e6, 8e4], rows
    assert rows['MaturityFactor'].tolist() == pytest.approx([1.0] * 3 + [math.sqrt(0.5), 1.0], rel=1e-15)


def test_an_fra_matures_where_it_pays_and_an_average_price_swap_reads_its_units(world):
    """CRE52.32 TABLE 1 AND ITS FAQ2, CRE52.34, CRE52.36 AND CRE52.39. Table 1's forward rate
    agreement for the period six months to a year out: S 0.5 and E 1, its adjusted notional ten
    million times SD = (e^(-0.025) - e^(-0.05)) / 0.05, 4,816,097.5, in bucket 2; borrowed and paid
    at its end, M is E, long the rate, its maturity factor one; lent and paid at its start, M is
    the start (FAQ2), short, at (1/2)^0.5. A sold swap of a thousand barrels of WTI's average: its
    units at today's 80, 80,000, short, its maturity factor its settlement's, (120/360)^0.5.

    Killing mutations: the agreement paid at its start matured at E - its maturity factor one; a
    lender read long - its delta +1.
    """
    _, output = world
    rows = trades(output, 'SINGLES')
    adjusted = 1e7 * supervisory_duration(0.5, 1.0)
    assert rows['AdjustedNotional'].tolist() == pytest.approx([adjusted, adjusted, 8e4], rel=1e-14), rows
    assert list(rows['Delta']) == [1.0, -1.0, -1.0], rows
    assert rows['MaturityFactor'].tolist() == pytest.approx(
        [1.0, math.sqrt(0.5), math.sqrt(120 / 360)], rel=1e-15), rows
    assert list(zip(rows['HedgingSet'], rows['Bucket'])) == [('USD', '2'), ('USD', '2'), ('Energy', '2')]


def test_amounts_fixed_today_read_no_row_and_enter_v_alone(world):
    """CRE52.10 AND CRE52.26: KNOWN CASH IS VALUE ONLY. A list of two fixed amounts, a fee of 100
    thousand two years out, a fixed energy leg received and a fixed leg at 4% standing alone are
    amounts fixed today, which carry no risk factor CRE52 charges: no row, the add-on nothing, the
    multiplier one on V above nothing and the EAD 1.4 V, V their values off one base valuation, the
    fee's 100,000 e^(-0.04 x 730/365) = 92,311.63. A package of the list and the fee the same.

    Killing mutation: the fee's type declaring no exposure - the run refused by name.
    """
    sets, output = world
    found, frame = netting_sets(output), output['Results']['SACCR']['trades']
    assert frame[frame['NettingSet'].isin(['KNOWN', 'KNOWN_PACKAGE'])].empty
    worth = valued(sets['KNOWN'][1] + sets['KNOWN_PACKAGE'][1])
    assert worth['BOOK_FEE'] == pytest.approx(1e5 * math.exp(-0.04 * 730 / 365), rel=1e-12), worth
    for name, value in (('KNOWN', sum(worth['BOOK_' + deal['Reference']] for deal in KNOWN)),
                        ('KNOWN_PACKAGE', worth['BOOK_PACKAGE'])):
        row = found.loc[name]
        assert row['V'] == pytest.approx(value, rel=1e-12) and row['AddOn_Aggregate'] == 0.0, row
        assert row['Multiplier'] == 1.0 and row['EAD'] == pytest.approx(1.4 * value, rel=1e-12), row


def test_a_floating_leg_reads_its_notional_averaged_over_its_periods(world):
    """CRE52.34, CRE52.37(2) AND CRE52.39. A floating leg on ten million amortising two million a
    year, its first period begun six months ago and its last ending 4.5 years out, read off the
    periods its compile binds: S nothing, the period begun counting from today, E and M 4.5, its
    notional averaged over those 4.5 years, (10 x 0.5 + 8 + 6 + 4 + 2) / 4.5 million = 5,555,555.56,
    times SD = (1 - e^(-0.225)) / 0.05 = 4.0296756, 22,387,086.80 in bucket 2 of USD, received long
    and paid short, at a maturity factor one; each set's add-on 0.5% of it, 111,935.43.

    Killing mutations: the notional the first period's whole - 40,296,756.25; the paid leg read long
    off its periods' sizes - its delta +1.
    """
    _, output = world
    adjusted = (10 * 0.5 + 8 + 6 + 4 + 2) * 1e6 / 4.5 * supervisory_duration(0.0, 4.5)
    for name, delta in (('RECEIVED_LEG', 1.0), ('PAID_LEG', -1.0)):
        row, found = trades(output, name).loc['LEG'], netting_sets(output).loc[name]
        assert row['AdjustedNotional'] == pytest.approx(adjusted, rel=1e-14), row
        assert (row['Delta'], row['MaturityFactor'], row['HedgingSet'], row['Bucket'], row['Component']) == (
            delta, 1.0, 'USD', '2', ''), row
        assert found['AddOn_IR'] == pytest.approx(0.005 * adjusted, rel=1e-14), found


def test_a_swap_booked_as_its_legs_reads_as_the_swap(world):
    """CRE52.1 FAQ3, CRE52.34 AND CRE52.39: A LINEAR SWAP IS READ WHOLE. The payer of ten million
    five years out, quarterly at 4%, booked as a SwapInterestDeal and as a package of its fixed leg
    paid and its floating leg received on the same days: the floating leg carries the swap, the
    fixed leg its value alone - S nothing, E and M five, its notional 10,000,000 over every quarter,
    0.25 years of 30/360 each so the average is exact, SD = (1 - e^(-0.25)) / 0.05, 44,239,843.39 in
    bucket 2, long, at a maturity factor one, the add-on 221,199.22: the swap's to the last digit,
    its value the swap's, so the set holding the payer beside the package received the other way
    up has V, each effective notional's sum and the add-on nothing. The package beside two lists of
    fixed amounts, ten million received at its end and 500 thousand paid in six months, reads the
    same one row, the lists their value alone in V.

    Killing mutation: the fixed leg read as a floating one, its paid notional short the rate - the
    package two rows, long and short 44,239,843.39, its add-on nothing.
    """
    sets, output = world
    found, rows, legs = netting_sets(output), trades(output, 'SWAP_ALONE'), trades(output, 'SWAP_LEGS')
    adjusted = 1e7 * supervisory_duration(0.0, 5.0)
    assert rows.loc['SWAP', 'AdjustedNotional'] == pytest.approx(adjusted, rel=1e-14), rows
    columns = ['AssetClass', 'HedgingSet', 'Bucket', 'AdjustedNotional', 'Delta', 'MaturityFactor',
               'EffectiveNotional', 'Value']
    assert list(legs[columns].itertuples(index=False)) == list(rows[columns].itertuples(index=False)), legs
    assert list(legs['Component']) == ['SWAP_LEGS_FLOAT'], legs
    assert found.loc['SWAP_LEGS', 'AddOn_IR'] == found.loc['SWAP_ALONE', 'AddOn_IR'] == pytest.approx(
        0.005 * adjusted, rel=1e-14), found
    netted = found.loc['SWAP_NETTED']
    assert (netted['V'], trades(output, 'SWAP_NETTED')['EffectiveNotional'].sum(), netted['AddOn_IR']) == (
        0.0, 0.0, 0.0), netted
    amounts = trades(output, 'SWAP_AMOUNTS')
    assert list(amounts[columns[:-1]].itertuples(index=False)) == list(rows[columns[:-1]].itertuples(index=False))
    assert found.loc['SWAP_AMOUNTS', 'AddOn_IR'] == found.loc['SWAP_ALONE', 'AddOn_IR'], found.loc['SWAP_AMOUNTS']
    assert found.loc['SWAP_AMOUNTS', 'V'] == pytest.approx(valued(sets['SWAP_AMOUNTS'][1])[
        'BOOK_SWAP'], rel=1e-12), found.loc['SWAP_AMOUNTS']


def test_a_trade_enters_v_once_whatever_rows_it_reads(world):
    """CRE52.10 AND CRE52.42(4): V COUNTS EACH TRADE ONCE. A package of a floating leg received on
    ten million five years out and one paid on five million two years out reads a row per leg, its
    `Component`: 44,239,843.39 long and 5,000,000 (1 - e^(-0.1)) / 0.05 = 9,516,258.20 short, both in
    bucket 2, the add-on 0.5% of 34,723,585.19, 173,617.93; the package's value on its first row
    alone, and V that value off one base valuation. A package of amounts fixed today reads no row,
    its V its value all the same.

    Killing mutation: V summed over the rows, each carrying its trade's value - the first package's
    V twice its value, the second's nothing.
    """
    sets, output = world
    found, rows = netting_sets(output), trades(output, 'TWO_ROWS')
    adjusted = [1e7 * supervisory_duration(0.0, 5.0), 5e6 * supervisory_duration(0.0, 2.0)]
    assert list(rows['Component']) == ['TWO_ROWS_FIVE', 'TWO_ROWS_TWO'], rows
    assert rows['AdjustedNotional'].tolist() == pytest.approx(adjusted, rel=1e-14), rows
    assert list(rows['Delta']) == [1.0, -1.0] and list(rows['Bucket']) == ['2', '2'], rows
    assert found.loc['TWO_ROWS', 'AddOn_IR'] == pytest.approx(0.005 * (adjusted[0] - adjusted[1]), rel=1e-14)
    worth = valued(sets['TWO_ROWS'][1])
    assert found.loc['TWO_ROWS', 'V'] == pytest.approx(worth['BOOK_PACKAGE'], rel=1e-12), found.loc['TWO_ROWS']
    assert rows['Value'].iloc[0] == found.loc['TWO_ROWS', 'V'] and math.isnan(rows['Value'].iloc[1]), rows


def test_two_floating_legs_on_two_indices_are_one_basis_trade(world):
    """CRE52.46, CRE52.72 AND CRE52.73: A BASIS TRADE IS HELD IN A HEDGING SET OF ITS OWN. Basis
    trades, two risk factors in one currency, form a hedging set per basis at half Table 2's
    supervisory factor, 0.25% for rates. A package receiving USD-OIS on ten million three years out
    and paying USD is one row: its received leg's, 10,000,000 (1 - e^(-0.15)) / 0.05 = 27,858,404.70
    in bucket 2, in the set `USD USD/USD-OIS`, the pair in one order so the two ways round net,
    short it where it receives the second, -1; the add-on 0.25% of that, 69,646.01. Received the
    other way round it is long, +1, at the same add-on; the two in one set net to nothing; and
    beside a payer of ten million three years out, whose 27,858,404.70 sits in USD's own set at
    0.5%, offset by none of it, the add-on 0.75% of 27,858,404.70, 208,938.04.

    Killing mutations: the basis left in the currency's own set - each package its two legs, long
    and short, its add-on nothing, and the set beside the payer 139,292.02; the pair named with the
    index received first, every basis row long - the two ways round in two sets, the netted set's
    add-on 139,292.02; the factor not halved - each package's add-on 139,292.02.
    """
    _, output = world
    found, adjusted = netting_sets(output), 1e7 * supervisory_duration(0.0, 3.0)
    for name, delta in (('BASIS_RECEIVED', -1.0), ('BASIS_PAID', 1.0)):
        rows = trades(output, name)
        assert list(zip(rows['HedgingSet'], rows['Bucket'], rows['SupervisoryFactor'], rows['Delta'],
                        rows['Component'])) == [('USD USD/USD-OIS', '2', 0.0025, delta, '')], rows
        assert rows['AdjustedNotional'].tolist() == pytest.approx([adjusted], rel=1e-14), rows
        assert found.loc[name, 'AddOn_IR'] == pytest.approx(0.0025 * adjusted, rel=1e-14), found.loc[name]
    assert found.loc['BASIS_NETTED', 'AddOn_IR'] == 0.0, found.loc['BASIS_NETTED']
    assert found.loc['BASIS_BESIDE', 'AddOn_IR'] == pytest.approx(0.0075 * adjusted, rel=1e-14)


def test_legs_paying_two_currencies_are_one_fx_trade(world):
    """CRE52.35, CRE52.37(5) AND CRE52.39; FOOTNOTE 11 TO CRE52.46: A CROSS-CURRENCY SWAP IS AN FX
    TRADE. A package of a floating leg on 1.25 million dollars and a fixed leg on a million euros,
    three years out, is one row on EUR/USD, nothing beside it in the rate class (CRE52.28 holds a
    trade in one class: a convention): its foreign leg the euros, 1,000,000 x 1.25 = 1,250,000, the
    dollar leg domestic; short the pair, -1, receiving the dollars, long, +1, receiving the euros;
    M three years, its maturity factor one; each set's add-on 4% of 1,250,000, 50,000, the two in
    one set nothing. A floating leg on a million euros received against a fixed one on 900
    thousand pounds has both legs foreign, the larger the pounds', 900,000 x 1.5 = 1,350,000, long
    EUR/GBP, its add-on 54,000. Two floating legs, on 1.25 million dollars received and a million
    euros paid, are no basis but the same FX trade, -1 on 1,250,000, its add-on 50,000 (the
    footnote). A swap whose euro leg's notional resets each quarter off 1.25
    million dollars (an `MtMCrossCurrencySwapDeal`) settles its exposure at each reset, so M runs
    to the next, three months out, its maturity factor (1/4)^0.5 = 0.5; that leg states its
    notional in dollars, 1,250,000, short paying the euros and long receiving them, each set's
    add-on 4% of 1,250,000 x 0.5, 25,000.

    Killing mutations: legs of two currencies read as rate legs - the dollar leg's IR row, no FX
    row; the larger leg dropped for the smaller - the pound swap's 1,250,000, its add-on 50,000;
    the resetting leg's notional read in euros - 1,562,500, its add-on 31,250; M the swap's last
    day despite its resets - its maturity factor one, its add-on 50,000.
    """
    _, output = world
    found = netting_sets(output)
    for name, pair, adjusted, delta, factor in (
            ('CCS_USD_RECEIVED', 'EUR/USD', 1.25e6, -1.0, 1.0), ('CCS_EUR_RECEIVED', 'EUR/USD', 1.25e6, 1.0, 1.0),
            ('CCS_FOREIGN', 'EUR/GBP', 1.35e6, 1.0, 1.0), ('CCS_FLOATING', 'EUR/USD', 1.25e6, -1.0, 1.0),
            ('MTM_PAY', 'EUR/USD', 1.25e6, -1.0, 0.5),
            ('MTM_RECEIVE', 'EUR/USD', 1.25e6, 1.0, 0.5)):
        rows = trades(output, name)
        assert list(zip(rows['AssetClass'], rows['HedgingSet'], rows['Delta'], rows['Component'])) == [
            ('FX', pair, delta, '')], rows
        assert rows['AdjustedNotional'].tolist() == pytest.approx([adjusted], rel=1e-15), rows
        assert rows['MaturityFactor'].tolist() == pytest.approx([factor], rel=1e-15), rows
        assert (found.loc[name, 'AddOn_IR'], found.loc[name, 'AddOn_FX']) == pytest.approx(
            (0.0, 0.04 * adjusted * factor), rel=1e-15), found.loc[name]
    assert found.loc['CCS_NETTED', 'AddOn_FX'] == 0.0, found.loc['CCS_NETTED']


def test_an_fx_swap_is_its_two_forwards_the_near_one_gone_once_settled(world):
    """CRE52.35, CRE52.39 AND CRE52.48; A CONVENTION, CRE52 STATING NO SPLIT OF AN FX SWAP. A swap
    buying a million euros for 1.3 million dollars six months out and selling them back for 1.32
    million a year out is two forwards, each a row with its own M: the near one long EUR/USD on its
    euros, 1,000,000 x 1.25 = 1,250,000, at (1/2)^0.5; the far one short, buying the dollars, on the
    euros it sells, 1,250,000, at one; the add-on 4% of 1,250,000 (1 - (1/2)^0.5), 14,644.66. The
    same swap whose near leg settled a month ago is its far forward alone, its add-on 50,000, and
    one whose near leg settles today still holds it, at the ten business days M is floored at,
    (10/250)^0.5 = 0.2, its add-on 4% of 1,250,000 (1 - 0.2), 40,000; the value of each swap on its
    first row.

    Killing mutations: each forward matured at the swap's last day - the near one's maturity factor
    one, the add-on nothing; the settled near forward kept - two rows, the add-on 40,000; a forward
    settling today dropped - one row, the add-on 50,000.
    """
    _, output = world
    found = netting_sets(output)
    rows, seasoned = trades(output, 'FX_SWAP'), trades(output, 'FX_SWAP_SEASONED')
    assert list(zip(rows['Component'], rows['HedgingSet'], rows['Delta'])) == [
        ('Near_Settlement_Date', 'EUR/USD', 1.0), ('Far_Settlement_Date', 'EUR/USD', -1.0)], rows
    assert list(rows['AdjustedNotional']) == [1.25e6, 1.25e6], rows
    assert rows['MaturityFactor'].tolist() == pytest.approx([math.sqrt(0.5), 1.0], rel=1e-15), rows
    assert found.loc['FX_SWAP', 'AddOn_FX'] == pytest.approx(0.04 * 1.25e6 * (1.0 - math.sqrt(0.5)), rel=1e-14)
    assert list(zip(seasoned['Component'], seasoned['Delta'], seasoned['MaturityFactor'])) == [
        ('Far_Settlement_Date', -1.0, 1.0)], seasoned
    assert found.loc['FX_SWAP_SEASONED', 'AddOn_FX'] == pytest.approx(5e4, rel=1e-15), found
    today = trades(output, 'FX_SWAP_TODAY')
    assert today['MaturityFactor'].tolist() == pytest.approx([0.2, 1.0], rel=1e-15), today
    assert found.loc['FX_SWAP_TODAY', 'AddOn_FX'] == pytest.approx(4e4, rel=1e-14), found
    assert rows['Value'].iloc[0] == found.loc['FX_SWAP', 'V'] and math.isnan(rows['Value'].iloc[1]), rows


def test_a_floating_energy_leg_reads_the_units_it_has_still_to_pay(world):
    """CRE52.36, CRE52.39 AND CRE52.48. A floating leg on WTI's monthly average is read off the
    periods its compile binds still to pay, the one paid a month ago gone: its units every period's
    volume at its price multiplier, fixed or to fix, 1,000 + 2,000 x 0.5 + 1,500 = 3,500, at today's
    80, 280,000; crude oil, bucket 2 of the energy set at 18%; M its last pay day, 95/360 of 30/360,
    its maturity factor 0.513701; long receiving the average and short paying it, the add-on 18% of
    280,000 x (95/360)^0.5, 25,890.54. Beside it a fixed energy leg paid is amounts fixed today, its
    value in V and no row.

    Killing mutations: the price multiplier unread - 4,500 units, 360,000, the add-on 33,287.84; a
    payer read long - its delta +1.
    """
    sets, output = world
    found = netting_sets(output)
    for name, delta in (('ENERGY_SWAP', 1.0), ('ENERGY_PAYER', -1.0)):
        rows = trades(output, name)
        assert list(zip(rows.index, rows['HedgingSet'], rows['Bucket'], rows['Delta'])) == [
            ('FLOATING', 'Energy', '2', delta)], rows
        assert rows['AdjustedNotional'].tolist() == pytest.approx([3.5e3 * 80.0], rel=1e-15), rows
        assert rows['MaturityFactor'].tolist() == pytest.approx([math.sqrt(95 / 360)], rel=1e-15), rows
        assert found.loc[name, 'AddOn_Commodity'] == pytest.approx(0.18 * 2.8e5 * math.sqrt(95 / 360), rel=1e-14)
    worth = valued(sets['ENERGY_SWAP'][1])
    assert found.loc['ENERGY_SWAP', 'V'] == pytest.approx(worth['BOOK_FLOATING'] + worth['BOOK_FIXED_ENERGY'],
                                                          rel=1e-12), found.loc['ENERGY_SWAP']


def test_a_caplet_is_a_call_on_its_periods_rate_and_a_floorlet_a_put(world):
    """CRE52.34, CRE52.40 AND CRE52.43: A CAP IS ITS CAPLETS. A floating leg on a million paying a
    cap at 4.5% on its rate over one period from a year out to two, fixed at its start, is one
    caplet: S = T = 1 and E = 2, its d 1,000,000 SD(1, 2) = 1,000,000 (e^(-0.05) - e^(-0.1)) / 0.05 =
    927,840.13 in bucket 2 of USD; P the period's rate on the flat 4% curve, e^0.04 - 1 =
    4.0810774%, K 4.5%, so d1 = (ln(4.0810774 / 4.5) + 0.5 x 0.5^2 x 1) / (0.5 x 1) = 0.054567 and
    the bought call's delta N(d1) = 0.521758; M two years, its maturity factor one; the add-on 0.5%
    of 484,108.37, 2,420.54. The leg sold is the caplet sold, -0.521758, at the same add-on. A
    floorlet on the same terms is the put, -N(-d1) = -0.478242, its add-on 0.5% of 443,731.76,
    2,218.66. A cap stating Sell over the bought leg is valued as that leg, its pricer summing its
    legs each on its own side (`Deal.sum_children`), and is read so: the bought caplet, V the leg's.

    Killing mutations: T read as the period's end - the caplet's delta 0.585257; the sold leg signed
    as bought - its delta +0.521758.
    """
    _, output = world
    found, adjusted = netting_sets(output), 1e6 * supervisory_duration(1.0, 2.0)
    caplet, floorlet = (black_delta(math.expm1(0.04), 0.045, 0.5, 1.0, call) for call in (True, False))
    for name, delta in (('CAPLET', caplet), ('CAPLET_SOLD', -caplet), ('CAPLET_SELL_CAP', caplet),
                        ('FLOORLET', floorlet)):
        rows = trades(output, name)
        assert list(zip(rows['Component'], rows['HedgingSet'], rows['Bucket'], rows['MaturityFactor'])) == [
            (name + '_LEG 2027-01-15', 'USD', '2', 1.0)], rows
        assert rows['AdjustedNotional'].tolist() == pytest.approx([adjusted], rel=1e-14), rows
        assert rows['Delta'].tolist() == pytest.approx([delta], rel=1e-12), rows
        assert found.loc[name, 'AddOn_IR'] == pytest.approx(0.005 * adjusted * abs(delta), rel=1e-12), found
    assert found.loc['CAPLET_SELL_CAP', 'V'] == found.loc['CAPLET', 'V'] > 0.0, found


def test_a_caplet_compounded_overnight_is_known_on_its_last_fixing(world):
    """CRE52.31, CRE52.40 AND CRE52.43. A cap at 4.5% on a million on a rate compounded overnight from
    six months out to nine, a fixing a day, which its pricer compounds into the period's rate, is
    one caplet: P that rate on the flat 4% curve, (e^(0.04 x 92/365) - 1) / (92/365) = 4.0202323%;
    T the day it is known, its last fixing's, 14 October 2025, 269/360 years of 30/360, so d1 =
    (ln(4.0202323 / 4.5) + 0.5 x 0.5^2 x 269/360) / (0.5 (269/360)^0.5) = -0.044735 and its delta
    0.482159; S 0.5 and E 0.75, its d 1,000,000 SD(0.5, 0.75) = 242,309.89 in bucket 1; M its last
    pay day, 0.75, its maturity factor 0.75^0.5; the add-on 0.5% of 101,179.44, 505.8972.

    Killing mutations: T read as the period's end, 270/360 - the delta 0.482512; T read as its first
    fixing, its pricer's Black time - 0.443503.
    """
    _, output = world
    found, row = netting_sets(output).loc['OVERNIGHT'], trades(output, 'OVERNIGHT').loc['LEG']
    accrual, adjusted = 92 / 365, 1e6 * supervisory_duration(0.5, 0.75)
    delta = black_delta(math.expm1(0.04 * accrual) / accrual, 0.045, 0.5, 269 / 360, True)
    assert (row['Component'], row['Bucket']) == ('OVERNIGHT_LEG 2025-10-15', '1'), row
    assert row['Delta'] == pytest.approx(delta, rel=1e-12) and row['AdjustedNotional'] == pytest.approx(
        adjusted, rel=1e-14), row
    assert row['MaturityFactor'] == pytest.approx(math.sqrt(0.75), rel=1e-15), row
    assert found['AddOn_IR'] == pytest.approx(0.005 * adjusted * delta * math.sqrt(0.75), rel=1e-12), found


def test_a_cap_reads_a_caplet_per_period_still_to_fix_each_in_its_own_bucket(world):
    """CRE52.10, CRE52.43 AND CRE52.57. A cap at 4.5% on a million over four half-years from three
    months ago, each fixed at its start, the first at 5%: that period is known cash, no row; the
    other three a caplet each, S = T 0.25, 0.75 and 1.25 and E 0.75, 1.25 and 1.75, P over 183, 182
    and 183 days 4.0403791%, 4.0401570% and 4.0403791%, the deltas 0.379819, 0.487063 and 0.534577
    on d 487,667.66, 475,627.10 and 463,883.82; M the cap's last day, 1.75, its maturity factor one.
    Bucket 1 holds the first, D1 = 185,225.57, and bucket 2 the others, D2 = 479,641.97: the add-on
    0.5% (D1^2 + D2^2 + 1.4 D1 D2)^0.5 = 3,117.4656. V is the cap's base valuation, the fixed period
    in it at its payoff, 1,000,000 x 182/365 x 0.5% e^(-0.04 x 90/365) = 2,468.68, the cap's value
    less that of the same cap without that period.

    Killing mutations: the fixed period read as an option - four rows; the bucket taken from the
    cap's end - all three in bucket 2, the add-on 3,324.3377.
    """
    sets, output = world
    found, rows = netting_sets(output).loc['STRIP'], trades(output, 'STRIP')
    periods = ((183, 0.25, 0.75), (182, 0.75, 1.25), (183, 1.25, 1.75))
    deltas = [black_delta(math.expm1(0.04 * days / 365) / (days / 365), 0.045, 0.5, start, True)
              for days, start, _ in periods]
    adjusted = [1e6 * supervisory_duration(start, end) for _, start, end in periods]
    assert list(zip(rows['Component'], rows['Bucket'], rows['MaturityFactor'])) == [
        ('STRIP_LEG 2025-10-15', '1', 1.0), ('STRIP_LEG 2026-04-15', '2', 1.0), ('STRIP_LEG 2026-10-15', '2', 1.0)]
    assert rows['Delta'].tolist() == pytest.approx(deltas, rel=1e-12), rows
    assert rows['AdjustedNotional'].tolist() == pytest.approx(adjusted, rel=1e-14), rows
    first, second = adjusted[0] * deltas[0], adjusted[1] * deltas[1] + adjusted[2] * deltas[2]
    assert found['AddOn_IR'] == pytest.approx(0.005 * math.sqrt(
        first * first + second * second + 1.4 * first * second), rel=1e-12), found
    worth = valued(sets['STRIP'][1] + [
        node(optioned('Cap', 'LIVE', 'Buy', every(6, 3, BASE + pd.DateOffset(months=3))))])
    assert found['V'] == pytest.approx(worth['BOOK_LEG'], rel=1e-12), found
    assert worth['BOOK_LEG'] - worth['BOOK_LIVE'] == pytest.approx(
        1e6 * 182 / 365 * 0.005 * math.exp(-0.04 * 90 / 365), rel=1e-12), worth


def test_a_bought_cap_and_a_sold_floor_at_one_strike_read_as_the_floating_leg_they_make(world):
    """CRE52.40 AND CRE52.43: PUT-CALL PARITY. A cap bought and a floor sold at 4.5% on a million over
    three years from a year out, a period a year, receive the rate less 4.5%, the floating leg on
    those periods and amounts fixed today. Row by row the bought caplet's delta N(d1) and the sold
    floorlet's N(-d1) sum to one, so each period's effective notionals sum to its d, 1,000,000
    SD(k, k + 1), all in bucket 2, and over the periods to 1,000,000 SD(1, 4) = 2,649,973.43, the
    floating leg's read whole off those periods, its notional over them times SD(1, 4). The add-ons
    agree to one ulp, 2.2e-16: N(d1) + N(-d1) is one to the bit on each row, and the three d sum to
    2,649,973.428454643 against the leg's 2,649,973.4284546426, the rounding of the telescoped SD.

    Killing mutation: a floorlet read as a call - each sold one's delta -N(d1), the pair's effective
    notional nothing.
    """
    _, output = world
    found, rows, leg = netting_sets(output), trades(output, 'COLLAR'), trades(output, 'COLLAR_LEG').loc['LEG']
    caps, floors = rows.loc['CAP'], rows.loc['FLOOR']
    assert list(caps['Bucket']) == list(floors['Bucket']) == ['2', '2', '2'], rows
    assert (caps['Delta'].to_numpy() + floors['Delta'].to_numpy()).tolist() == pytest.approx([1.0] * 3, rel=3e-16)
    assert leg['EffectiveNotional'] == pytest.approx(1e6 * supervisory_duration(1.0, 4.0), rel=1e-15), leg
    assert rows['EffectiveNotional'].sum() == pytest.approx(leg['EffectiveNotional'], rel=3e-16), rows
    assert found.loc['COLLAR', 'AddOn_IR'] == pytest.approx(found.loc['COLLAR_LEG', 'AddOn_IR'], rel=3e-16)


def test_a_cap_over_its_legs_enters_v_once_and_matures_on_its_last_day(world):
    """CRE52.10, CRE52.43 AND CRE52.48. A cap booked over three capped legs - a year from a year out,
    a year from two years out and three months from three months out - reads a caplet per leg, each
    named by its leg and its period's end, the cap's value on its first row alone and V the cap's
    base valuation. M is the cap's, its last day three years out, so every row's maturity factor is
    one, the short leg's too though that leg alone ends half a year out; its d 1,000,000 SD(0.25,
    0.5) = 245,357.77 in bucket 1.

    Killing mutations: V summed over the rows, each carrying the cap's value - V three times it; M
    read per leg - the short leg's maturity factor 0.5^0.5.
    """
    sets, output = world
    found, rows = netting_sets(output).loc['CAP_LEGS'], trades(output, 'CAP_LEGS')
    assert list(zip(rows['Component'], rows['Bucket'], rows['MaturityFactor'])) == [
        ('CAP_LEGS_FIRST 2027-01-15', '2', 1.0), ('CAP_LEGS_SECOND 2028-01-15', '2', 1.0),
        ('CAP_LEGS_SHORT 2025-07-15', '1', 1.0)], rows
    assert rows['AdjustedNotional'].iloc[2] == pytest.approx(1e6 * supervisory_duration(0.25, 0.5), rel=1e-14)
    assert found['V'] == pytest.approx(valued(sets['CAP_LEGS'][1])['BOOK_CAP'], rel=1e-12), found
    assert rows['Value'].iloc[0] == found['V'] and rows['Value'].iloc[1:].isna().all(), rows


def test_a_cap_or_a_floor_stated_by_its_terms_reads_the_optionlet_it_compiles(world):
    """CRE52.43. A cap and a floor at 4.5% on a million from a year out to two, a period of a year,
    stated by their terms over no legs, compile the one optionlet the capped leg holds, and each
    reads that leg's row, the caplet's delta 0.521758 and the floorlet's -0.478242 on d 927,840.13,
    at the leg's value.

    Killing mutation: the cap's type declaring no exposure - the run refused by name.
    """
    _, output = world
    columns = ['AssetClass', 'HedgingSet', 'Bucket', 'AdjustedNotional', 'Delta', 'MaturityFactor',
               'EffectiveNotional']
    rows = trades(output, 'BY_TERMS')
    for reference, twin in (('CAP', 'CAPLET'), ('FLOOR', 'FLOORLET')):
        leg = trades(output, twin).loc['LEG']
        assert list(rows.loc[reference, columns]) == list(leg[columns]), rows
        assert rows.loc[reference, 'Value'] == pytest.approx(leg['Value'], rel=1e-12), rows


def test_every_row_of_a_trade_matures_on_the_trades_last_day(world):
    """CRE52.31 AND CRE52.48: M IS THE TRADE'S. A package holding a cap at 4.5% on a million over the
    quarter from three months out, the same leg uncapped and a leg paid on a million quarterly for
    nine months ends on the latest day its children reach, the paid leg's last payment nine months
    out; booked under a cap whose own terms run two years out it ends there too, the cap there to
    imply its legs and its own terms unread. Under either every row's M is 270/360 = 0.75 years of
    30/360 and its maturity factor 0.75^0.5 = 0.866025, the capped and the uncapped leg's own half
    year unread. The caplet on d 1,000,000 (e^(-0.0125) - e^(-0.025)) / 0.05 = 245,357.77 in bucket
    1 is a call at 4.5% on P (e^(0.04 x 91/365) - 1) / (91/365) = 4.0200117% known in a quarter,
    0.25 years, d1 -0.326170 and delta 0.372148; the uncapped leg +1 on the same d and the paid leg
    -1 on 1,000,000 (1 - e^(-0.0375)) / 0.05 = 736,111.65: the add-on 0.5% of 0.866025 |245,357.77 x
    1.372148 - 736,111.65|, 1,729.65, the two holders' rows the same to the bit.

    Killing mutations: the parent's own last day read where it has children - the cap's rows at
    one, its add-on 1,997.22; M read per child, each its own last day - the caplet and the uncapped
    leg at (1/2)^0.5 under either holder, each add-on 1,997.16.
    """
    _, output = world
    found, mf = netting_sets(output), math.sqrt(0.75)
    adjusted = [1e6 * supervisory_duration(0.25, 0.5)] * 2 + [1e6 * supervisory_duration(0.0, 0.75)]
    delta = black_delta(math.expm1(0.04 * 91 / 365) / (91 / 365), 0.045, 0.5, 0.25, True)
    for name in ('TRADE_PACKAGE', 'TRADE_CAP'):
        rows = trades(output, name)
        assert list(rows['Component']) == [name + '_CAPPED 2025-07-15', name + '_FLOAT', name + '_LONG'], rows
        assert rows['MaturityFactor'].tolist() == pytest.approx([mf] * 3, rel=1e-15), rows
        assert rows['AdjustedNotional'].tolist() == pytest.approx(adjusted, rel=1e-14), rows
        assert rows['Delta'].tolist() == pytest.approx([delta, 1.0, -1.0], rel=1e-12), rows
        assert found.loc[name, 'AddOn_IR'] == pytest.approx(0.005 * mf * abs(
            adjusted[0] * (1.0 + delta) - adjusted[2]), rel=1e-12), found.loc[name]
    columns = ['AdjustedNotional', 'Delta', 'MaturityFactor', 'EffectiveNotional']
    assert (trades(output, 'TRADE_CAP')[columns].to_numpy() == trades(output, 'TRADE_PACKAGE')[columns].to_numpy()).all()


EQUITY_LEG = {'Object': 'EquitySwapLeg', 'Reference': 'LEG', 'Currency': 'USD',
              'Equity_Currency': 'USD', 'Discount_Rate': 'USD', 'Equity': 'INDEX', 'Buy_Sell': 'Buy',
              'Effective_Date': BASE, 'Maturity_Date': years(1),
              'Payment_Frequency': pd.DateOffset(months=3), 'Principal_Fixed_Variable': 'Variable',
              'Units': 1e4}

#: The four strips on the euro, each fixing monthly for two months and settling two days on.
STRIP_FX = {'Currency': 'USD', 'Underlying_Currency': 'EUR', 'Discount_Rate': 'USD',
            'FX_Volatility': 'EUR.USD', 'Buy_Sell': 'Buy', 'Option_Type': 'Call',
            'Strike_Price': 1.3, 'Underlying_Amount': 1e6}
FIXINGS = [[BASE + pd.DateOffset(months=k), BASE + pd.DateOffset(months=k, days=2), None]
           for k in (1, 2)]
STRIPS = [
    dict(STRIP_FX, Object='FXTARFOptionDeal', Reference='TARF', Expiry_Date=FIXINGS[-1][1],
         LeverageNotional=2e6, TargetLevel=0.1, TARF_ExpiryDates=FIXINGS),
    dict(STRIP_FX, Object='FXAccumulatorOptionDeal', Reference='ACCUMULATOR', LeverageNotional=2e6,
         Barrier_Type='Up_And_Out', Barrier_Price=1.6, Accumulator_ExpiryDates=FIXINGS),
    dict(STRIP_FX, Object='FXExtendableForwardDeal', Reference='EXTENDABLE', Extension_Strike=1.3,
         Extension_Date=FIXINGS[0][0], Extendable_ExpiryDates=[row + [None] for row in FIXINGS]),
    {'Object': 'FXRangeAccrualDeal', 'Reference': 'RANGE', 'Currency': 'USD', 'Buy_Sell': 'Buy',
     'Underlying_Currency': 'EUR', 'Discount_Rate': 'USD', 'FX_Volatility': 'EUR.USD',
     'Lower_Barrier': 1.2, 'Upper_Barrier': 1.3, 'Payoff': 1e4, 'Settlement_Date': FIXINGS[-1][1],
     'Observation_Dates': [row[:1] + [None] for row in FIXINGS]}]

#: A TARF buying a million euros a fixing at 1.3 and geared two million against: a fixing a day ago
#: at 1.35 settling tomorrow, its 0.05 past the 0.04 target, then `FIXINGS`.
TARF = dict(STRIPS[0], TargetLevel=0.04, TARF_ExpiryDates=[
    [BASE - pd.DateOffset(days=1), BASE + pd.DateOffset(days=1), 1.35]] + FIXINGS)


def collar(forward, strike, vol, expiry, call=True):
    """CRE52.42(3)'s delta of a bought digital: its option bought at 95% of `strike` less the one
    sold at 105%, a put's the other way round."""
    bought, sold = (0.95, 1.05) if call else (1.05, 0.95)
    return (black_delta(forward, bought * strike, vol, expiry, call) -
            black_delta(forward, sold * strike, vol, expiry, call))


def test_a_cash_digital_is_its_collar_its_effective_notional_capped_at_its_payoff_over_sf(world):
    """CRE52.36, CRE52.40, CRE52.42(3) AND CRE52.48. A bought digital call on the share struck at 50
    paying 100,000 dollars a year out is a call bought at 47.50 and one sold at 52.50, each on
    100,000 / (0.1 x 50) = 20,000 shares: one row, its d those shares at 50, 1,000,000, P the
    share's forward 50 e^(0.02 x 365/365) = 51.010067 at the single name's 120% over T one, so
    d1 = (ln(51.010067 / 47.5) + 0.72) / 1.2 = 0.659411 and (ln(51.010067 / 52.5) + 0.72) / 1.2 =
    0.576008, its delta N(0.659411) - N(0.576008) = 0.745184 - 0.717695 = 0.027489; MF one, its
    effective notional 27,488.91, inside the cap of its payoff over the factor, 100,000 / 32% =
    312,500, its add-on 32% of it, 8,796.45. The same digital two days from its expiry, settled a
    year out: P 50.005480, T 2/360, d1 0.619423 and -0.499544, its delta 0.732181 - 0.308698 =
    0.423483 on the same million, 423,483.06 past the cap, so its effective notional is 312,500 and
    its add-on the payoff, 100,000. Held under the margined set past its threshold, its MPOR of five
    floored at ten, its maturity factor 1.5 (10/250)^0.5 = 0.3 and its effective notional
    127,044.92, inside the cap, its add-on 32% of that; the set's EAD is its unmargined one
    (CRE52.2), whose add-on the cap holds at 100,000: 1.4 x multiplier(V, C, 100,000) x 100,000, its
    RC nothing, V under the 200,000 held. The cap binds near the band close to the expiry and never
    deep in the money, where both legs read nearly one: over every T the most its effective notional
    reaches is 1.36 times the cap at a forward 6% over the strike, 0.85 at 10%, 0.46 at 20%. A
    digital put on the same terms, a put bought at 52.50 and one sold at 47.50, reads -0.027489: the
    two pay 100,000 for certain and read nothing, N(d1) + N(-d1) one on each leg.

    Killing mutations: the collar struck at K on both legs - every delta nothing; the cap not
    applied - the near digital's effective notional 423,483.06, its add-on 135,514.58; the cap on
    the margined column alone - the margined set's EAD 109,541.22 against 66,907.21; the put's legs
    the call's - the pair's add-on 17,592.90.
    """
    _, output = world
    found = netting_sets(output)
    forward = 50.0 * math.exp(0.02 * between(years(1), 3))
    delta = collar(forward, 50.0, 1.2, 1.0)
    row = trades(output, 'DIGITAL').loc['CALL']
    assert (row['Component'], row['AdjustedNotional'], row['MaturityFactor']) == ('', 1e6, 1.0), row
    assert row['Delta'] == pytest.approx(delta, rel=1e-12), row
    assert found.loc['DIGITAL', 'AddOn_Equity'] == pytest.approx(0.32 * 1e6 * delta, rel=1e-12), found
    near = trades(output, 'DIGITAL_NEAR').loc['CALL']
    assert near['Delta'] == pytest.approx(collar(50.0 * math.exp(0.02 * between(
        BASE + pd.DateOffset(days=2), 3)), 50.0, 1.2, 2 / 360), rel=1e-12), near
    assert near['EffectiveNotional'] == pytest.approx(1e5 / 0.32, rel=1e-15) and 1e5 / 0.32 < 1e6 * near['Delta']
    assert found.loc['DIGITAL_NEAR', 'AddOn_Equity'] == pytest.approx(1e5, rel=1e-15), found
    margined = found.loc['DIGITAL_MARGINED']
    assert margined['AddOn_Equity'] == pytest.approx(0.32 * 0.3 * 1e6 * near['Delta'], rel=1e-12), margined
    assert margined['EAD'] == pytest.approx(1.4 * multiplier(margined['V'], margined['C'], 1e5) * 1e5, rel=1e-12)
    pair = trades(output, 'DIGITAL_PAIR')
    assert pair['Delta'].tolist() == pytest.approx([delta, -delta], rel=1e-12), pair
    assert found.loc['DIGITAL_PAIR', 'AddOn_Equity'] == pytest.approx(0.0, abs=1e-9), found


def test_an_asset_or_nothing_digital_is_its_vanilla_and_a_cash_digital_of_its_units_at_its_strike(world):
    """CRE52.42(3) AND CRE52.42(4). A digital call paying a thousand shares where the share ends
    above 50 a year out pays (S - 50) on them and 50 x 1,000 = 50,000 dollars: two rows, the vanilla
    call on the thousand shares, d 50,000 and delta N((ln(51.010067 / 50) + 0.72) / 1.2) =
    N(0.616667) = 0.731273, and the cash digital paying 50,000, the collar on 50,000 / 5 = 10,000
    shares, d 500,000 and delta 0.027489, under the cap 50,000 / 32% = 156,250. The put pays 50,000
    less (50 - S) on them: the vanilla put sold, +0.268727, and the cash digital put, -0.027489. The
    two pay the thousand shares for certain: their effective notionals sum to 50,000 to an ulp, the
    set's add-on 32% of it, 16,000.

    Killing mutations: the shares read as cash - one row each, d 1,000 / 5 x 50 = 10,000; the put's
    vanilla bought - its delta -0.268727, the pair's sum 23,127.27.
    """
    _, output = world
    forward = 50.0 * math.exp(0.02 * between(years(1), 3))
    rows = trades(output, 'ASSET')
    vanilla, digital = black_delta(forward, 50.0, 1.2, 1.0, True), collar(forward, 50.0, 1.2, 1.0)
    assert list(zip(rows['Component'], rows['AdjustedNotional'])) == [
        ('ASSET_CALL vanilla', 5e4), ('ASSET_CALL digital', 5e5), ('ASSET_PUT vanilla', 5e4),
        ('ASSET_PUT digital', 5e5)], rows
    assert rows['Delta'].tolist() == pytest.approx([vanilla, digital, 1.0 - vanilla, -digital], rel=1e-12)
    assert rows['EffectiveNotional'].sum() == pytest.approx(5e4, rel=3e-16), rows
    assert netting_sets(output).loc['ASSET', 'AddOn_Equity'] == pytest.approx(1.6e4, rel=1e-15)
    assert rows['Value'].notna().tolist() == [True, False, True, False], rows


def test_a_digital_its_expiry_has_fixed_pays_known_cash_and_reads_no_row(world):
    """CRE52.10 AND CRE52.31: AN OPTION PAST ITS EXPIRY IS NO LONGER ONE. The share digital paying
    100,000 dollars, which expired the business day before at a print of 50.5, inside its band from
    47.50 to 52.50, and pays in a week, owes what is known: no row, its value in V alone, 100,000
    e^(-0.04 x 7/365) = 99,923.32. The same digital paying a thousand shares reads its vanilla alone,
    as an expired option does: delta one on what the expiry fixed, 50.5 over 50, on d 50,000, at the
    ten business days its maturity is floored at, (10/250)^0.5 = 0.2; its cash digital of 50,000 is
    known and reads none.

    Killing mutation: the expired digital read as a collar - the cash digital's row at delta one,
    the collar's limit inside its band, D 200,000; the asset's digital row beside its vanilla.
    """
    _, output = world
    found = netting_sets(output)
    assert trades(output, 'DIGITAL_FIXED').empty and found.loc['DIGITAL_FIXED', 'AddOn_Equity'] == 0.0, found
    assert found.loc['DIGITAL_FIXED', 'V'] == pytest.approx(1e5 * math.exp(-0.04 * 7 / 365), rel=1e-12), found
    row = trades(output, 'ASSET_FIXED')
    assert list(zip(row['Component'], row['AdjustedNotional'], row['Delta'])) == [
        ('ASSET_FIXED_CALL vanilla', 5e4, 1.0)], row
    assert row['MaturityFactor'].tolist() == pytest.approx([0.2], rel=1e-15), row


def test_every_digital_type_reads_the_collar_of_its_terms(world):
    """CRE52.42(1) AND CRE52.42(3); A CONVENTION FOR A BARRIER OR A TOUCH, CRE52 STATING NEITHER'S
    DELTA. The share digital knocked out above 70 reads the plain digital's row, its barrier unread.
    A one-touch on the share at 55 paying 100,000 reads the digital call struck there, d 100,000 /
    5.5 x 50 = 909,090.91 and delta 0.028780, and the no-touch its negative. A WTI digital call
    struck at 90 paying 100,000 a year out is the collar on 100,000 / 9 = 11,111.11 barrels at
    today's 80, 888,888.89, P WTI's forward for the expiry's delivery, 80.799635, at crude's 70%:
    0.055889; WTI's touches at 90 read it and its negative. The euro digital call in dollars
    paying 100,000 at 1.3 is on 100,000 / 0.13 = 769,230.77 euros, 961,538.46 dollars, P 1.227248
    at 15%: 0.250197. The euro Asian call paying its million dollars where its average ends above
    1.3 is on 1,000,000 / 0.13 euros, 9,615,384.62 dollars, P the average's current value, 1.235325:
    0.253152.

    Killing mutations: a type declaring no exposure - the run refused by name; the euro digital
    declared on the dollar it pays in - d 1,250,000 and delta -0.001984; the no-touch the
    one-touch's sign - +0.028780.
    """
    _, output = world
    rows, columns = trades(output, 'DIGITALS'), ['Delta', 'AdjustedNotional', 'MaturityFactor', 'EffectiveNotional']
    assert list(rows.loc['BARRIER', columns]) == list(rows.loc['CALL', columns]), rows
    for one, other, sign in (('ONE_TOUCH', 'CALL_55', 1.0), ('NO_TOUCH', 'CALL_55', -1.0), ('WTI_ONE', 'WTI', 1.0),
                             ('WTI_NO', 'WTI', -1.0)):
        assert rows.loc[one, 'Delta'] == sign * rows.loc[other, 'Delta'], rows
        assert rows.loc[one, 'AdjustedNotional'] == rows.loc[other, 'AdjustedNotional'], rows
    share = 50.0 * math.exp(0.02 * between(years(1), 3))
    euro = [1.25 * math.exp((0.04 - math.log(1.06)) * between(day)) for day in (SIX_MONTHS, years(1))]
    expected = {'CALL_55': (collar(share, 55.0, 1.2, 1.0), 1e5 / 5.5 * 50.0),
                'WTI': (collar(80.0 * (1.0 + 0.06 * 365 / (years(6) - BASE).days), 90.0, 0.7, 1.0), 1e5 / 9.0 * 80.0),
                'EURO': (collar(euro[1], 1.3, 0.15, 1.0), 1e5 / 0.13 * EURO),
                'FX_AVERAGE': (collar((1.24 + sum(euro)) / 3.0, 1.3, 0.15, 1.0), 1e6 / 0.13 * EURO)}
    for name, (delta, adjusted) in expected.items():
        assert rows.loc[name, 'Delta'] == pytest.approx(delta, rel=1e-12), rows.loc[name]
        assert rows.loc[name, 'AdjustedNotional'] == pytest.approx(adjusted, rel=1e-15), rows.loc[name]
    assert list(rows.loc[['WTI', 'EURO'], 'HedgingSet']) == ['Energy', 'EUR/USD'], rows


def test_a_tarf_is_its_fixings_to_come_each_an_option_bought_and_one_sold(world):
    """CRE52.40 AND CRE52.43; THE TARGET UNREAD, A CONVENTION. A TARF reads a row per leg of each
    fixing still to come, the strip read to its end as if no target redeemed it, which never
    understates it: a call bought on its million euros and, geared against, a put sold on its two
    million, each at 1.3, as its value signs them - bought, its calls alone struck at 1.0 are worth
    more than nothing and its geared puts alone struck at 1.6 less, an accumulator's alike, each
    leg long the euro. Each fixing is T, P EUR/USD's forward to it between its spot dates:
    1.247999 at T 30/360 and 1.246252 at 60/360, so d1 -0.921102 and -0.658893, the
    deltas N(d1) 0.178499 and 0.254982 on 1,250,000 and N(-d1) 0.821501 and 0.745018 on 2,500,000;
    M the TARF's last day, 62/360, its maturity factor 0.414997, once for the trade. The fixing a day
    ago settles tomorrow: known cash, no row, in V alone - its 0.05 has passed the 0.04 target, so
    the TARF is worth that row's capped payment alone, 1,000,000 x 0.04 e^(-0.04/365) = 39,995.62,
    on its first row, and the strip is still read past it. The add-on 4% of 0.414997 x
    4,458,149.04, 74,004.68.

    Killing mutations: the leveraged leg bought - its deltas -0.821501 and -0.745018; the fixed row
    read as an option - the run refused, a `KeyError` on the print the TARF states none of; the
    target read as an end - no row; T read as the settlement day - 0.186311 and 0.258499.
    """
    _, output = world
    rows, signs = trades(output, 'TARF'), trades(output, 'STRIP_SIGNS')
    for kind in ('TARF', 'ACCUMULATOR'):
        calls, puts = signs.loc[kind + '_CALLS'], signs.loc[kind + '_PUTS']
        assert calls['Value'].iloc[0] > 0.0 and (calls['Delta'].iloc[::2] > 0.5).all(), calls
        assert puts['Value'].iloc[0] < 0.0 and (puts['Delta'].iloc[1::2] > 0.5).all(), puts
    days = [day for day, *_ in FIXINGS]
    deltas = [sign * black_delta(1.25 * math.exp((0.04 - math.log(1.06)) * between(day)), 1.3, 0.15, k / 12, call)
              for k, day in enumerate(days, 1) for sign, call in ((1.0, True), (-1.0, False))]
    assert list(rows['Component']) == ['TARF_TARF {:%Y-%m-%d} {}'.format(day, leg) for day in days
                                       for leg in ('Underlying_Amount', 'LeverageNotional')], rows
    assert rows['Delta'].tolist() == pytest.approx(deltas, rel=1e-12), rows
    assert list(rows['AdjustedNotional']) == [1.25e6, 2.5e6] * 2, rows
    assert rows['MaturityFactor'].tolist() == pytest.approx([math.sqrt(62 / 360)] * 4, rel=1e-15), rows
    adjusted = sum(notional * delta for notional, delta in zip([1.25e6, 2.5e6] * 2, deltas))
    assert netting_sets(output).loc['TARF', 'AddOn_FX'] == pytest.approx(
        0.04 * math.sqrt(62 / 360) * adjusted, rel=1e-12)
    assert rows['Value'].iloc[0] == pytest.approx(4e4 * math.exp(-0.04 / 365), rel=1e-12), rows
    assert rows['Value'].iloc[0] == netting_sets(output).loc['TARF', 'V'] and rows['Value'].iloc[1:].isna().all()


def test_a_tarf_geared_one_to_one_reads_the_forward_of_each_fixing(world):
    """CRE52.39 AND CRE52.40: PUT-CALL PARITY. A TARF whose geared amount is its own million euros
    buys a call and sells a put at 1.3 at each fixing: the forward. Row by row its two deltas,
    N(d1) and N(-d1), sum to one and its effective notional to the d of a forward buying the
    million euros for 1.3 million dollars on the fixing's settlement, 1,250,000, which the forward
    reads at +1: equal to the last digit, N(d1) + N(-d1) rounding to one on each fixing.

    Killing mutation: the geared leg a call - its delta -N(d1), each fixing's sum nothing.
    """
    _, output = world
    rows, forwards = trades(output, 'TARF_EVEN'), trades(output, 'FORWARDS_EVEN')
    legs = (rows['AdjustedNotional'] * rows['Delta']).to_numpy().reshape(2, 2).sum(axis=1)
    assert legs.tolist() == (forwards['AdjustedNotional'] * forwards['Delta']).tolist() == [1.25e6] * 2, rows


def test_an_accumulator_sold_is_its_options_sold_at_each_fixing(world):
    """CRE52.40 AND CRE52.43; THE KNOCK-OUT UNREAD, A CONVENTION. An accumulator is signed as a TARF
    is, by its value (`STRIP_SIGNS`): sold on puts, it is short a put on its million euros and long a
    call on its two million at each fixing, both long the euro, N(-d1) 0.821501 and 0.745018 on
    1,250,000 and N(d1) 0.178499 and 0.254982 on 2,500,000 at the TARF's forwards; its knock-out
    above 1.6 unread.

    Killing mutation: the leveraged leg bought - its deltas -0.178499 and -0.254982.
    """
    _, output = world
    rows = trades(output, 'ACCUMULATOR')
    tarf = trades(output, 'TARF')['Delta'].tolist()
    assert rows['Delta'].tolist() == [tarf[1], tarf[0], tarf[3], tarf[2]], rows
    assert list(rows['AdjustedNotional']) == [1.25e6, 2.5e6] * 2, rows


def test_a_range_accrual_is_a_digital_bought_at_its_lower_bound_and_one_sold_at_its_upper(world):
    """CRE52.42(3) AND CRE52.43. A range accrual earning 10,000 dollars at each of two monthly
    observations the euro fixes inside 1.2 to 1.3 is, at each, a digital call bought at 1.2 and one
    sold at 1.3, each paying 10,000, bought where its value is more than nothing: at
    1.2 the collar on 10,000 / 0.12 = 83,333.33 euros, 104,166.67 dollars, at 1.3 on 76,923.08
    euros, 96,153.85 dollars, each observation T at its forward, the deltas +0.561663 and -0.583618
    a month out and +0.490378 and -0.498175 two; M its settlement, 62/360. Each cap, 10,000 / 4% =
    250,000, is out of reach; the add-on 4% of 0.414997 x 5,569.10, 92.45.

    Killing mutations: the upper digital bought - its deltas +0.583618 and +0.498175; the range
    declared on the dollar it pays in - every row's d 125,000 and its delta under 1e-8.
    """
    _, output = world
    rows = trades(output, 'RANGE')
    days = [day for day, *_ in FIXINGS]
    forwards = [1.25 * math.exp((0.04 - math.log(1.06)) * between(day)) for day in days]
    deltas = [sign * collar(forward, bound, 0.15, k / 12) for k, forward in enumerate(forwards, 1)
              for sign, bound in ((1.0, 1.2), (-1.0, 1.3))]
    assert list(rows['Component']) == ['RANGE_RANGE {:%Y-%m-%d} {}'.format(day, bound) for day in days
                                       for bound in ('Lower_Barrier', 'Upper_Barrier')], rows
    assert rows['Delta'].tolist() == pytest.approx(deltas, rel=1e-12) and rows['Value'].iloc[0] > 0.0, rows
    adjusted = [1e4 / bound * 10.0 * EURO for bound in (1.2, 1.3)] * 2
    assert rows['AdjustedNotional'].tolist() == pytest.approx(adjusted, rel=1e-15), rows
    assert netting_sets(output).loc['RANGE', 'AddOn_FX'] == pytest.approx(0.04 * math.sqrt(62 / 360) * sum(
        notional * delta for notional, delta in zip(adjusted, deltas)), rel=1e-12)


def inflation(reference, side, dates, notionals):
    """A dollar leg paying 2% a year on each of `notionals` (or the one) grown by CPI over its period,
    a period between each two of `dates`, paid at its end, received where `side` is Buy."""
    sizes = notionals if isinstance(notionals, list) else [notionals] * (len(dates) - 1)
    return {'Object': 'YieldInflationCashflowListDeal', 'Reference': reference, 'Currency': 'USD',
            'Discount_Rate': 'USD', 'Buy_Sell': side, 'Index': 'CPI',
            'Index_Reference': {'Months_Lag': 3, 'Reference_Type': 'Interpolated'}, 'Cashflows': {'Items': [{
                'Payment_Date': end, 'Notional': size, 'Base_Reference_Date': start, 'Base_Reference_Value': 0.0,
                'Final_Reference_Date': end, 'Final_Reference_Value': 0.0, 'Accrual_Start_Date': start,
                'Accrual_End_Date': end, 'Accrual_Day_Count': 'ACT_365',
                'Accrual_Year_Fraction': (end - start).days / 365.0, 'Yield': utils.Percent(2.0),
                'Rate_Multiplier': 1.0} for start, end, size in zip(dates[:-1], dates[1:], sizes)]}}


#: A leg on ten million grown by CPI received yearly for seven years, and swaps of it against a fixed
#: and a floating leg paid on its days.
SEVEN = every(12, 7)
LINKER = inflation('LINKER', 'Buy', SEVEN, 1e7)
INFLATION_SWAPS = [package(reference, [LINKER, cashflows(kind, 'RATE', 'Sell', SEVEN, 1e7)])
                   for reference, kind in (('FIXED', FIXED), ('FLOATING', FLOATING))]


def test_an_inflation_leg_is_a_rate_trade_in_a_hedging_set_of_its_own(world):
    """CRE52.34, CRE52.37(2), CRE52.39 AND CRE52.57 FAQ1: INFLATION IS A RATE TRADE IN A SET OF ITS
    OWN. A leg paying 2% a year on ten million dollars grown by CPI, yearly for seven years from the
    base date, received, its value above nothing, is one IR row in the set of its currency and index,
    `USD InflationRate.CPI`, apart from USD's own: S nothing, E and M seven, d = 10,000,000 (1 -
    e^(-0.35)) / 0.05 = 59,062,382.06 in bucket 3 at 0.5%, +1, its maturity factor one; the add-on
    295,311.91. Swapped against a fixed leg paid on its days it reads that row alone, the fixed leg's
    amounts fixed today in V alone; against a floating leg paid on its days, two rows, the floating
    leg's 59,062,382.06 short in USD's set, which the inflation row does not net: the add-on each
    leg's, 590,623.82, and V the swap's on its first row. A received and a paid leg on one schedule,
    the paid one worth less than nothing, net to nothing in their set. A leg on ten million
    amortising two a year, begun six months ago, reads its notional averaged over its 4.5 years
    left, the floating leg's (10 x 0.5 + 8 + 6 + 4 + 2) / 4.5 million x (1 - e^(-0.225)) / 0.05 =
    22,387,086.80, in bucket 2 of its own set.

    Killing mutations: the leg read into its currency's rate set - the floating swap's add-on
    nothing; the leg signed by its field, `long=('Buy_Sell', 'Buy')` declared - the paid leg +1 and
    the netted set's add-on 590,623.82; the leg recomposed with the rate leg beside it, `legs` taking
    a leg on another factor - the run refused, a `KeyError` on the forecast curve it binds none of.
    """
    _, output = world
    found, adjusted = netting_sets(output), 1e7 * supervisory_duration(0.0, 7.0)
    row = trades(output, 'INFLATION').loc['LINKER']
    assert (row['HedgingSet'], row['Bucket'], row['SupervisoryFactor'], row['Delta'], row['MaturityFactor']) == (
        'USD InflationRate.CPI', '3', 0.005, 1.0, 1.0), row
    assert row['AdjustedNotional'] == pytest.approx(adjusted, rel=1e-14) and row['Value'] > 0.0, row
    assert found.loc['INFLATION', 'AddOn_IR'] == pytest.approx(0.005 * adjusted, rel=1e-14), found.loc['INFLATION']
    columns = ['HedgingSet', 'Bucket', 'AdjustedNotional', 'Delta', 'MaturityFactor', 'EffectiveNotional']
    fixed, floating = trades(output, 'INFLATION_FIXED'), trades(output, 'INFLATION_FLOATING')
    assert list(fixed['Component']) == ['INFLATION_FIXED_LINKER'], fixed
    assert list(fixed.loc['FIXED', columns]) == list(row[columns]), fixed
    assert list(zip(floating['Component'], floating['HedgingSet'], floating['Bucket'], floating['Delta'])) == [
        ('INFLATION_FLOATING_LINKER', 'USD InflationRate.CPI', '3', 1.0), ('INFLATION_FLOATING_RATE', 'USD', '3', -1.0)]
    assert floating['AdjustedNotional'].tolist() == pytest.approx([adjusted] * 2, rel=1e-14), floating
    assert found.loc['INFLATION_FLOATING', 'AddOn_IR'] == pytest.approx(0.01 * adjusted, rel=1e-14)
    assert floating['Value'].iloc[0] == found.loc['INFLATION_FLOATING', 'V'] and math.isnan(floating['Value'].iloc[1])
    netted = trades(output, 'INFLATION_NETTED')
    assert list(netted['Delta']) == [1.0, -1.0] and netted.loc['PAID', 'Value'] < 0.0, netted
    assert found.loc['INFLATION_NETTED', 'AddOn_IR'] == 0.0, found.loc['INFLATION_NETTED']
    amortising = trades(output, 'INFLATION_AMORTISING').loc['LEG']
    assert (amortising['HedgingSet'], amortising['Bucket']) == ('USD InflationRate.CPI', '2'), amortising
    assert amortising['AdjustedNotional'] == pytest.approx(
        (10 * 0.5 + 8 + 6 + 4 + 2) * 1e6 / 4.5 * supervisory_duration(0.0, 4.5), rel=1e-14), amortising


def swaplets(reference, side, equity, kind, amounts, dates, known):
    """A dollar swap of `equity`'s price return over each period between two of `dates`, paid at its
    end, on each of `amounts`, shares under `kind` Shares and dollars under Principal, the return
    received where `side` is Buy, a start behind the base date fixed at `known`."""
    return {'Object': 'EquitySwapletListDeal', 'Reference': reference, 'Currency': 'USD', 'Equity_Currency': 'USD',
            'Discount_Rate': 'USD', 'Equity': equity, 'Buy_Sell': side, 'Amount_Type': kind, 'Cashflows': {'Items': [{
                'Start_Date': start, 'End_Date': end, 'Payment_Date': end, 'Amount': amount, 'Start_Multiplier': 1.0,
                'End_Multiplier': 1.0, 'Dividend_Multiplier': 0.0, 'Known_Start_Price': known if start < BASE else 0.0}
                for start, end, amount in zip(dates[:-1], dates[1:], amounts)]}}


#: Two half-years from three months ago: a thousand index units' return received, then two thousand,
#: and a million dollars' worth of the share's paid, then three million; and a thousand index units'
#: from a month ago to yesterday, fixed at 3,900 and 3,990 and paid in a week.
HALVES = every(6, 2, BASE - pd.DateOffset(months=3))
SWAPLETS = [swaplets('SHARES', 'Buy', 'INDEX', 'Shares', [1e3, 2e3], HALVES, 3900.0),
            swaplets('PRINCIPAL', 'Sell', 'SHARE', 'Principal', [1e6, 3e6], HALVES, 49.0)]
ENDED = swaplets('ENDED', 'Buy', 'INDEX', 'Shares', [1e3], [BASE - pd.DateOffset(months=1), BASE - pd.DateOffset(
    days=1)], 3900.0)
ENDED['Cashflows']['Items'][0].update(Payment_Date=BASE + pd.DateOffset(days=7), Known_End_Price=3990.0)


def test_a_swaplet_list_is_a_linear_equity_trade_on_the_period_it_is_in(world):
    """CRE52.36, CRE52.39 AND CRE52.48: A TOTAL RETURN SWAP IS LINEAR. A swaplet list reads one row,
    its amount the period's it is in, its first still to end, which the next resets and does not add
    to: stated in shares, its units at today's price; stated as a principal, that amount in its
    currency, no unit priced; +1 where it receives the return and -1 where it pays, as its value is
    above or below nothing; M its last payment day. A thousand index units received for the half
    from three months ago, two thousand the next: d 1,000 x 4,000 = 4,000,000 in the index's set at
    20%, +1. A million dollars' worth of the share's return paid, three million the next: d
    1,000,000 at 32%, -1. Each M nine months, its maturity factor 0.75^0.5 = 0.866025; the add-on
    CRE52.66's single factor over the index's 692,820.32 at 80% and the share's -277,128.13 at 50%,
    634,980.31. A list whose one return fixed yesterday at 3,990 against 3,900 and pays in a week
    owes what is known: no row, its value in V alone, 1,000 x 90 e^(-0.04 x 7/365) = 89,930.99.

    Killing mutations: the periods summed - d 12,000,000 and 4,000,000; the principal priced as
    shares - 50,000,000; M the first payment day - 0.5; a list whose every return has fixed read on
    a period it is in - the run refused, an `IndexError`.
    """
    _, output = world
    found, rows = netting_sets(output), trades(output, 'SWAPLETS')
    assert list(zip(rows.index, rows['HedgingSet'], rows['Subclass'], rows['AdjustedNotional'], rows['Delta'])) == [
        ('SHARES', 'EquityPrice.INDEX', 'Index', 4e6, 1.0), ('PRINCIPAL', 'EquityPrice.SHARE', 'Single', 1e6, -1.0)]
    assert rows['Value'].iloc[0] > 0.0 > rows['Value'].iloc[1], rows
    assert rows['MaturityFactor'].tolist() == pytest.approx([math.sqrt(0.75)] * 2, rel=1e-15), rows
    add_ons = [0.2 * 4e6 * math.sqrt(0.75), -0.32 * 1e6 * math.sqrt(0.75)]
    assert found.loc['SWAPLETS', 'AddOn_Equity'] == pytest.approx(math.sqrt(sum(single_factor(
        add_ons, [0.8, 0.5]))), rel=1e-14), found.loc['SWAPLETS']
    assert trades(output, 'SWAPLETS_ENDED').empty and found.loc['SWAPLETS_ENDED', 'AddOn_Equity'] == 0.0
    assert found.loc['SWAPLETS_ENDED', 'V'] == pytest.approx(9e4 * math.exp(-0.04 * 7 / 365), rel=1e-12)


def test_a_swaplet_list_beside_its_funding_leg_reads_each_its_own_row(world):
    """CRE52.34, CRE52.36 AND CRE52.42(4). The thousand index units' list booked in a package beside a
    floating leg paid on four million dollars over its days reads two rows, each its child's: the
    list's as alone, 4,000,000 at +1 in the index's set, and the floating leg's, S nothing and E 0.75,
    4,000,000 (1 - e^(-0.0375)) / 0.05 = 2,944,446.58 in bucket 1 of USD at -1; each at the maturity
    factor 0.75^0.5; the equity add-on 20% of 4,000,000 x 0.866025, 692,820.32, and the rate one 0.5%
    of 2,944,446.58 x 0.866025, 12,749.83; V the package's on its first row.

    Killing mutation: the list taken for a rate leg of the package, `legs` taking a leg on another
    factor - the run refused, a `KeyError` on the forecast curve its compile binds none of.
    """
    _, output = world
    found, rows = netting_sets(output).loc['SWAPLET_PACKAGE'], trades(output, 'SWAPLET_PACKAGE')
    funding = 4e6 * supervisory_duration(0.0, 0.75)
    assert list(zip(rows['Component'], rows['AssetClass'], rows['HedgingSet'], rows['Bucket'], rows['Delta'])) == [
        ('SWAPLET_PACKAGE_SHARES', 'Equity', 'EquityPrice.INDEX', '', 1.0),
        ('SWAPLET_PACKAGE_FUNDING', 'IR', 'USD', '1', -1.0)], rows
    assert rows['AdjustedNotional'].tolist() == pytest.approx([4e6, funding], rel=1e-14), rows
    assert rows['MaturityFactor'].tolist() == pytest.approx([math.sqrt(0.75)] * 2, rel=1e-15), rows
    assert (found['AddOn_Equity'], found['AddOn_IR']) == pytest.approx(
        (0.2 * 4e6 * math.sqrt(0.75), 0.005 * funding * math.sqrt(0.75)), rel=1e-14), found
    assert rows['Value'].iloc[0] == found['V'] and math.isnan(rows['Value'].iloc[1]), rows


@pytest.mark.parametrize('held_node, drop, said', [
    (package('PACKAGE', [SWAP_LEGS[0], optioned('Cap', 'CAPPED', 'Buy', every(12, 3), utils.Percent(1.0))]), (),
     'SET_CAPPED is a CFFloatingInterestListDeal paying a digital on its rate, whose collar of '
     'CRE52.42\\(3\\) is not built'),
    (node(optioned('Floor', 'CAPPED', 'Buy', every(12, 3), utils.Percent(1.0))), (),
     'SET_CAPPED is a CFFloatingInterestListDeal paying a digital on its rate'),
    (dict(package('INNER', [SWAP_LEGS[1]]), Instrument={'.Deal': {
        'Object': 'NettingCollateralSet', 'Reference': 'INNER', 'Netted': 'True', 'Collateralized': 'False'}}), (),
     'SET_INNER is a NettingCollateralSet, which declares no exposure CRE52 reads'),
    (dict(package('PRICE_SWAP', [KNOWN[3]]), Instrument={'.Deal': {
        'Object': 'PriceSwapDeal', 'Reference': 'PRICE_SWAP', 'Currency': 'USD', 'Buy_Sell': 'Buy',
        'Strike_Amount': 9e5, 'Maturity_Date': years(1)}}), (),
     'SET_PRICE_SWAP is a PriceSwapDeal, which declares no exposure CRE52 reads'),
    (package('THREE', [cashflows(FLOATING, 'USD', 'Buy', every(12, 3), 1.25e6),
                       cashflows(FIXED, 'EUR', 'Sell', every(12, 3), 1e6, currency='EUR'),
                       cashflows(FIXED, 'GBP', 'Sell', every(12, 3), 9e5, currency='GBP')]), (),
     'SET_THREE is a package whose legs pay in EUR, GBP, USD, no one pair CRE52.35 reads'),
    (node(EQUITY_LEG), (), 'SET_LEG is a EquitySwapLeg, which declares no exposure CRE52 reads'),
    (node(STRIPS[2]), (), 'SET_EXTENDABLE is a FXExtendableForwardDeal, whose extension is an option on a strip of '
                          'forwards CRE52 states no delta for'),
    (node(dict(STRIPS[0], InvertedTarget='Yes')), (), "SET_TARF is a FXTARFOptionDeal whose accrual is on the "
                                                      "reciprocal of its pair, where CRE52's reading at its strike "
                                                      "as stated does not hold"),
    *[(node(touch(kind, 'TOUCH', 1.4, Underlying_Currency='EUR', FX_Volatility='EUR.USD', Delivery_Date=years(1))),
       (), 'SET_TOUCH is a {}, which declares no exposure CRE52 reads'.format(kind))
      for kind in ('FXOneTouchOption', 'FXNoTouchOption')],
    (node(MIXED[2]), ('EquityForwardDeal',), 'SET_INDEX_FWD is a EquityForwardDeal, and the mapping '
                                             'classes it under no SACCR.Asset_Class'),
    (node(default_swap('CDS', 'UNRATED', years(5), 'Buy')), (),
     'SET_CDS reads SurvivalProb.UNRATED, and the mapping states no SACCR.Rating for it')])
def test_a_trade_the_reader_cannot_read_is_refused_by_name(tmp_path, held_node, drop, said):
    """A TRADE IS READ THROUGH ITS DECLARATIONS; CRE52.35, CRE52.42(3) AND CRE52.74-52.76. A
    floating leg paying digital caplets or floorlets, alone or in a package, is a strip of digitals,
    whose collars are not built; a netting set inside a netting set is several sets to one
    agreement, not built; a price swap settles in its children's place, a trade on their value that
    its children's own rows do not read; a package whose legs pay in three currencies is no one
    pair; an equity swap leg's type declares no exposure; an extendable forward's right to extend
    its strip of forwards is an option CRE52 states no delta for; a TARF stating `InvertedTarget`
    pays N1 max(c(1/K - 1/S), 0) at each fixing on the reciprocal of its pair, not the option at its
    strike (its calls alone on a million euros struck at 1.0 are worth 392,309.89 stated so and
    491,715.33 blank, the vanilla's 1,000,000 sum (F - 1) DF); an FX one-touch's and no-touch's
    payout may be in the currency their barrier prices, which no collar of a cash digital reads; an
    equity forward, which does declare one, is booked under a mapping classing its type in no asset
    class; and a default swap is on a survival curve the mapping rates nothing: each refuses the run
    by name.

    Killing mutations: a digital optionlet read as the plain one its strike states - the capped
    package valued and its EAD written; a container read through its children whatever it is - the
    price swap's bond read as amounts fixed today, its add-on nothing; legs of three currencies left
    unrefused - the package read as its rate legs and the run valued; a type declaring no exposure
    skipped - the leg's set valued and its EAD written, nothing refused; a strip of no options read
    as its fixings - a `KeyError` on the expiry its declaration leaves unstated; the reciprocal TARF
    read at its strike - its set valued, nothing refused.
    """
    with pytest.raises(ValueError, match='SACCR: trade ' + said):
        run(document({'SET': ({}, [held_node])}, saccr(mapping_file(tmp_path / 'mapping.json', drop))))


@pytest.mark.parametrize('broken, said', [
    (lambda parameters: parameters['supervisory']['Credit'].pop('CCC'), "'CCC'"),
    (lambda parameters: parameters.update(alpha='one point four'),
     "could not convert string to float: 'one point four'"),
    (lambda parameters: parameters['commodity_sets'].update({'18': ['Energy', 'Coal']}),
     "'a commodity_sets subclass of Table 2: Coal'")])
def test_a_parameters_file_stating_an_entry_amiss_is_refused_by_name(tmp_path, broken, said):
    """THE FILE IS HELD TO CRE52'S SHAPE BEFORE ANYTHING IS VALUED. Table 2 without its CCC row, an
    alpha that is no number and a commodity bucket held at a subclass Table 2 has none of each
    refuse the run at its setup, naming the entry.

    Killing mutation: the file's entries read where a trade reads them - the missing row refused as
    a bare `KeyError` from the first credit trade, after the book is valued.
    """
    with open(PARAMETERS, 'rt', encoding='utf-8') as f:
        parameters = json.load(f)
    broken(parameters)
    path = tmp_path / 'parameters.json'
    with open(path, 'wt', encoding='utf-8') as f:
        json.dump(parameters, f)
    with pytest.raises(ValueError, match='SACCR: .* does not state {} as CRE52 reads it'.format(
            said.replace('(', r'\(').replace(')', r'\)'))):
        run(document({'SET': ({}, [node(swap('SWAP', years(5), 'Fixed'))])}, saccr(
            mapping_file(tmp_path / 'mapping.json'), str(path))))


def test_a_run_dated_after_its_market_file_leaves_the_files_date_standing(tmp_path):
    """SA-CCR a day after its market file's date, then the file's quotes bootstrapped on the same
    context mark a cash flow as a fresh context does, to the bit. Killing mutation: the run's date
    left in the file's `System Parameters`, where the bootstrap solves the quotes a day late."""
    found = dict(factors(0.2), **{'InterestRate.USD-OIS': {
        'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
        'Curve': utils.Curve([], [[tenor(years(n)), 0.04 + 0.002 * n] for n in (1, 2, 5)])}})
    quotes = bootstrappers.knot_prices('USD-OIS', found, ModelParams(), BASE, {}, {
        'Fixed_Frequency': '1Y', 'Float_Frequency': '1Y', 'Compounding': 'OIS'})
    market = tmp_path / 'market.json'
    market.write_text(json.dumps({'MarketData': {
        'System Parameters': {'Base_Currency': 'USD', 'Base_Date': BASE},
        'Valuation Configuration': {}, 'Price Factors': found, 'Market Prices': dict([quotes]),
        'Bootstrapper Configuration': {'InterestRate': {'Prices': 'InterestRate'}}}},
        cls=CustomJsonEncoder))
    cash = {'Object': 'FixedCashflowDeal', 'Reference': 'CF', 'Currency': 'USD',
            'Discount_Rate': 'USD-OIS', 'Amount': 1e6, 'Payment_Date': years(3)}

    def on_file(trade, calculation):
        job = document({'NS': ({}, [node(trade)])}, calculation)
        job['Calc']['MergeMarketData'] = {'MarketDataFile': str(market), 'ExplicitMarketData': {}}
        return json.dumps(job), 'on_file'

    def marked(context):
        context.load_json(on_file(cash, {'Object': 'BaseValuation'}))
        context.bootstrap()
        frame = context.run_job()[1]['Results']['mtm']
        return frame[frame['Reference'] == 'NS_CF']['Value'].item()

    reused = derivus.Context()
    reused.load_json(on_file(swap('SWAP', years(5), 'Fixed'), dict(
        saccr(mapping_file(tmp_path / 'mapping.json')), Base_Date=BASE + pd.DateOffset(days=1))))
    reused.run_job()
    assert marked(reused) == marked(derivus.Context())
