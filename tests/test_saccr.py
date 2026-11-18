"""The SA-CCR calculation - CRE52's exposure at default of each netting set - gated through job
documents on CRE99's five worked examples, each a netting set booked on a world of the module's
own, and on sets built to reach the floors, the margin terms and the classes the examples leave
out.

Every level is invented, chosen so the examples' terms are what the engine reads: a USD curve flat
at 4% and a EUR one through 6% a year at each anniversary of the base date, so the one-into-ten
receiver's legs have the forward swap rate of 6% CRE99.31 assumes; a euro at 1.25 dollars, a
pound at 1.5 and a dollar at 150 yen, so a notional in any is crossed; three survival curves rated in the module's mapping as
CRE99.41 rates the entities; three commodities whose forward curves price a unit at 80, 100 and 25
dollars. The swaps' rates and the default swaps' coupons are solved for the market values the
examples state and the receiver's vol for its 50 thousand. A commodity forward is worth its funded
price, ten million and more, which no term moves, so silver is lent at the rate that makes the
three forwards worth the 20 thousand CRE99.59 states together: each set is worth the example's V.
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
        'InterestRate.EUR': {'Currency': 'EUR', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                             'Curve': utils.Curve([], [[tenor(years(k)), k * math.log(1.06) / tenor(
                                 years(k))] for k in range(1, 12)])},
        'InterestYieldVol.EUR': {'Property_Aliases': None, 'Distribution_Type': 'Lognormal',
                                 'Shift': utils.Percent(0.0), 'Surface': flat_space(vol)},
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
        'Index_Day_Count': 'ACT_365', 'Pay_Frequency': pd.DateOffset(months=12),
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
     'Strike_Price': 1.3, 'Underlying_Amount': 1e6, 'Expiry_Date': years(1)},
    {'Object': 'FXOptionDeal', 'Reference': 'USD_CALL', 'Currency': 'EUR', 'Underlying_Currency': 'USD',
     'Discount_Rate': 'EUR', 'FX_Volatility': 'EUR.USD', 'Buy_Sell': 'Sell', 'Option_Type': 'Call',
     'Strike_Price': 0.75, 'Underlying_Amount': 1e6, 'Expiry_Date': SIX_MONTHS},
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
     'Sell_Amount': 0.9e6, 'Sell_Currency': 'GBP', 'Settlement_Currency': 'USD',
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


def node(deal):
    return {'Instrument': {'.Deal': deal}}


def held(prefix, nodes):
    """`nodes` under one set, every reference in them `prefix`ed, so a trade booked in two sets is
    two trades."""
    return [dict(item, Instrument={'.Deal': dict(item['Instrument']['.Deal'], Reference='{}_{}'.format(
        prefix, item['Instrument']['.Deal']['Reference']))}, **(
        {'Children': held(prefix, item['Children'])} if item.get('Children') else {}))
        for item in nodes]


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
    """`(mapping path, output)`: the five examples beside their hedging sets alone and the
    module's other sets, one SA-CCR run over one book."""
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
            'BARRIERS': ({}, [node(deal) for deal in BARRIERS])}
    return mapping, run(document(sets, saccr(mapping), receiver_vol()))


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


EQUITY_LEG = {'Object': 'EquitySwapLeg', 'Reference': 'LEG', 'Currency': 'USD',
              'Equity_Currency': 'USD', 'Discount_Rate': 'USD', 'Equity': 'INDEX', 'Buy_Sell': 'Buy',
              'Effective_Date': BASE, 'Maturity_Date': years(1),
              'Payment_Frequency': pd.DateOffset(months=3), 'Principal_Fixed_Variable': 'Variable',
              'Units': 1e4}


@pytest.mark.parametrize('deal, drop, said', [
    (EQUITY_LEG, (), 'SET_LEG is a EquitySwapLeg, which declares no exposure CRE52 reads'),
    (MIXED[2], ('EquityForwardDeal',), 'SET_INDEX_FWD is a EquityForwardDeal, and the mapping '
                                       'classes it under no SACCR.Asset_Class'),
    (default_swap('CDS', 'UNRATED', years(5), 'Buy'), (),
     'SET_CDS reads SurvivalProb.UNRATED, and the mapping states no SACCR.Rating for it'),
    (dict(ASIANS[0], Is_Digital='Yes'), (), 'SET_FX_AVERAGE is a FXDiscreteExplicitAsianOption paying '
                                           'a digital, whose collar of CRE52.42\\(3\\) is not built')])
def test_a_trade_whose_exposure_the_declarations_do_not_state_is_refused_by_name(
        tmp_path, deal, drop, said):
    """A TRADE IS READ THROUGH ITS DECLARATIONS. An equity swap leg, whose type declares no
    exposure; an equity forward, which does, under a mapping classing its type in no asset class;
    a default swap on a survival curve the mapping rates nothing; and an Asian paying a digital,
    whose collar is not built: each refuses the run by name.

    Killing mutations: a type declaring no exposure skipped - the leg's set valued and its EAD
    written, nothing refused; a digital read as the vanilla its strike states - the Asian's set
    valued.
    """
    mapping = mapping_file(tmp_path / 'mapping.json', drop)
    with pytest.raises(ValueError, match='SACCR: trade ' + said):
        run(document({'SET': ({}, [node(deal)])}, saccr(mapping)))


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
