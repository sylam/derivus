"""An inflation-linked cashflow list pays its fixed interest grown by the ratio of its final to its
base index reference. A reference states its level, or names a date the level is read off: the
index's prints where published, projected off the last print at the inflation curve otherwise.

The reference is the linker by hand off `trial_rates`' own index: printed monthly at 0.2% a month
to June 2026, then growing at the curve's flat 2.5%, and read three months back, interpolated.
"""

import copy
import json
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd

import derivus
import test_declared_defaults as book
import trial_rates
from derivus import utils
from derivus.config import CustomJsonEncoder
from test_position_scaling import document, marks

B = book.WORLD_BASE
LINKER = next(d for d in trial_rates.DEALS if d['Reference'] == 'LINKER')


def index_level(day):
    """INFL at a month start: its print where published, else the last print grown at 2.5%."""
    prints = trial_rates.FACTORS['PriceIndex.INFL']
    published = prints['Last_Period_Start']
    if day <= published:
        knots = prints['Index'].array
        return float(np.interp((day - utils.excel_offset).days, knots[:, 0], knots[:, 1]))
    return index_level(published) * math.exp(0.025 * (day - published).days / 365.0)


def reference_level(date):
    """The interpolated three-month-lag reference at `date`: the prints three and two months back,
    weighted by how far into its own month `date` sits."""
    start = date.to_period('M').to_timestamp()
    weight = (date - start).days / ((start + pd.DateOffset(months=1)) - start).days
    return sum(share * index_level((date - pd.DateOffset(months=lag)).to_period('M').to_timestamp())
               for lag, share in ((3, 1.0 - weight), (2, weight)))


def linker(items):
    """`trial_rates`' LINKER paying `items`, its one-deal wire document and its value by hand:
    `Notional Rate_Multiplier final / base Yield accrual DF(pay)` summed, USD at 4%."""
    deal = dict(LINKER, Cashflows={'Items': items})
    value = sum(item['Notional'] * item['Rate_Multiplier'] * item['Yield'].amount *
                item['Accrual_Year_Fraction'] * math.exp(-book.DISCOUNT[0] * (
                    item['Payment_Date'] - B).days / 365.0) *
                (item['Final_Reference_Value'] or reference_level(item['Final_Reference_Date'])) /
                (item['Base_Reference_Value'] or reference_level(item['Base_Reference_Date']))
                for item in items)
    return document(SimpleNamespace(DEALS=[deal], FACTORS=trial_rates.FACTORS, CONFIGURATION={})), value


def dated(items, when, rows=slice(None)):
    """`items` with the base reference of each row in `rows` a date, `when(item)`, and no level."""
    items = copy.deepcopy(items)
    for item in items[rows]:
        item.update(Base_Reference_Value=0.0, Base_Reference_Date=when(item))
    return items


def test_a_reference_date_is_read_off_the_index_and_a_stated_level_as_stated():
    """The trialled linker, its base print stated at 104, beside the same linker naming its base
    reference as a date and stating no level - 90 days back, on the base date, and, on its last two
    coupons only, their own accrual starts a year and more ahead - each marks the linker by hand.

    Reference, by hand: each reference's level off the index as the module docstring reads it, or
    the level stated; `sum N A (I_final / I_base) r alpha DF(pay)`.

    Killing mutations: a reference's level read whatever its flag says, which prices a date 90 days
    back or on the base date against a level of nothing; the flag inverted, which reads the mixed
    list's stated levels off the index and its dates as levels.
    """
    items = LINKER['Cashflows']['Items']
    for variant in (items, dated(items, lambda item: B - pd.Timedelta(days=90)),
                    dated(items, lambda item: B),
                    dated(items, lambda item: item['Accrual_Start_Date'], slice(2, None))):
        job, reference = linker(variant)
        mark = float.fromhex(marks(job)['LINKER'])
        assert abs(mark - reference) <= 1e-12 * reference, (mark, reference)


def test_an_inflation_list_on_a_static_index_prices_under_a_credit_monte_carlo():
    """The trialled linker, and the same with its first coupon's final reference on 2026-09-15 -
    read off June's print and July's, which is not published - under a credit Monte Carlo that
    simulates the USD curve and holds the index static: every row and path is finite and the first
    row is the base valuation's mark.

    Killing mutations: the static index's one row joined to the declared prints unbroadcast, the
    256-against-1 refusal that skipped the deal; the straddling rows joined unbroadcast; the
    projection's columns reshaped onto the simulation batch.
    """
    items = copy.deepcopy(LINKER['Cashflows']['Items'])
    straddling = copy.deepcopy(items)
    straddling[0]['Final_Reference_Date'] = pd.Timestamp('2026-09-15')
    for variant in (items, straddling):
        job, _ = linker(variant)
        base = float.fromhex(marks(job)['LINKER'])
        deals = job['Calc']['Deals']['Deals']
        deals['Children'] = [{'Instrument': {'.Deal': {
            'Object': 'NettingCollateralSet', 'Reference': 'NS', 'Netted': 'True',
            'Collateralized': 'False'}}, 'Children': deals['Children']}]
        job['Calc']['Calculation'] = {
            'Object': 'CreditMonteCarlo', 'Base_Date': B, 'Currency': 'USD',
            'Time_Grid': '0d 3m(3m)', 'Batch_Size': 256, 'Simulation_Batches': 1,
            'Random_Seed': 1, 'Deflation_Interest_Rate': 'USD'}
        market = job['Calc']['MergeMarketData']['ExplicitMarketData']
        market['Model Configuration'] = {'.ModelParams': {
            'modeldefaults': {'InterestRate': 'HullWhite1FactorInterestRateModel'},
            'modelfilters': {}}}
        market['Price Models'] = {'HullWhite1FactorInterestRateModel.USD': {
            'Alpha': 0.05, 'Lambda': 0.0, 'Quanto_FX_Correlation': 0.0,
            'Quanto_FX_Volatility': utils.Curve([], [[0.0, 0.0], [10.0, 0.0]]),
            'Sigma': utils.Curve([], [[0.0, 0.01], [10.0, 0.01]])}}
        context = derivus.Context()
        context.load_json((json.dumps(job, cls=CustomJsonEncoder), 'linker_static_index'))
        profile = context.run_job()[1]['Results']['mtm']
        assert np.isfinite(profile.values).all(), profile.mean(axis=1)
        assert abs(profile.iloc[0].mean() - base) <= 1e-6 * base, (profile.iloc[0].mean(), base)
