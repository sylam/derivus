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
import pytest
import torch

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


def rolled(items, at):
    """`items` valued by hand at `at`: `Notional Rate_Multiplier final / base Yield accrual DF(pay)`
    summed over the cashflows paying on or after it, USD at 4%."""
    return sum(item['Notional'] * item['Rate_Multiplier'] * item['Yield'].amount *
               item['Accrual_Year_Fraction'] * math.exp(-book.DISCOUNT[0] * (
                   item['Payment_Date'] - at).days / 365.0) *
               (item['Final_Reference_Value'] or reference_level(item['Final_Reference_Date'])) /
               (item['Base_Reference_Value'] or reference_level(item['Base_Reference_Date']))
               for item in items if item['Payment_Date'] >= at)


def linker(items):
    """`trial_rates`' LINKER paying `items`, its one-deal wire document and its value by hand."""
    deal = dict(LINKER, Cashflows={'Items': items})
    return document(SimpleNamespace(DEALS=[deal], FACTORS=trial_rates.FACTORS, CONFIGURATION={})), \
        rolled(items, B)


def still(job, simulated, grid='0d 1m(1m)'):
    """`job`'s profile on `grid` under a credit Monte Carlo in float64 whose USD curve is a
    Hull-White at zero vol and whose index is static or, `simulated`, a GBM at zero vol drifting at
    the curve's 2.5% on its own 365.25-day clock."""
    deals = job['Calc']['Deals']['Deals']
    deals['Children'] = [{'Instrument': {'.Deal': {
        'Object': 'NettingCollateralSet', 'Reference': 'NS', 'Netted': 'True',
        'Collateralized': 'False'}}, 'Children': deals['Children']}]
    job['Calc']['Calculation'] = {
        'Object': 'CreditMonteCarlo', 'Base_Date': B, 'Currency': 'USD', 'Time_Grid': grid,
        'Batch_Size': 16, 'Simulation_Batches': 1, 'Random_Seed': 1, 'Deflation_Interest_Rate': 'USD'}
    defaults = {'InterestRate': 'HullWhite1FactorInterestRateModel'}
    models = {'HullWhite1FactorInterestRateModel.USD': {
        'Alpha': 0.05, 'Lambda': 0.0, 'Quanto_FX_Correlation': 0.0,
        'Quanto_FX_Volatility': utils.Curve([], [[0.0, 0.0], [10.0, 0.0]]),
        'Sigma': utils.Curve([], [[0.0, 0.0], [10.0, 0.0]])}}
    if simulated:
        defaults['PriceIndex'] = 'GBMPriceIndexModel'
        models['GBMPriceIndexModel.INFL'] = {'Vol': 0.0, 'Drift': 0.025 * 365.25 / 365.0}
    market = job['Calc']['MergeMarketData']['ExplicitMarketData']
    market['Model Configuration'] = {'.ModelParams': {'modeldefaults': defaults, 'modelfilters': {}}}
    market['Price Models'] = models
    context = derivus.Context()
    context.load_json((json.dumps(job, cls=CustomJsonEncoder), 'linker_still'))
    return derivus.run_cmc(context.current_cfg, prec=torch.float64)[1]['Results']['mtm']


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


def test_a_month_not_yet_printed_reads_the_forward_the_base_valuation_projects():
    """The linker's first coupon, its final reference on 2026-09-15 - read off June's print and
    July's, which is printed twelve days after the base date - under a credit Monte Carlo whose index
    is a GBM at zero vol drifting at the curve's 2.5% on its own 365.25-day clock and whose USD
    curve is a Hull-White at zero vol: every row is the base valuation rolled to it by hand, July
    100 x 1.002^29 x exp(0.025 x 30/365) = 106.1833 where it read June's 106.0151 from the row
    printing August on. A level stated with no date reads the base date's references, which after a
    cashflow still forecast the pricer cannot split at a print: it refuses by name. The order is a
    cashflow's, and only among months not yet printed: a principal paid beside the last coupon on its
    final reference, coupons sharing a base reference whose second month is not printed, and printed
    base references running backwards mark by hand.

    Killing mutations: an unprinted month read off the index's last print; the order guard dropped,
    which prices that linker at NaN; the guard reading the printed references too, which refuses
    the backwards ones; the guard reading every month a cashflow interpolates, which refuses the
    redeeming and the forward-based linkers.
    """
    item = dict(LINKER['Cashflows']['Items'][0], Final_Reference_Date=pd.Timestamp('2026-09-15'))
    job, value = linker([item])
    assert float.fromhex(marks(job)['LINKER']) == pytest.approx(value, rel=1e-12)
    profile = still(job, simulated=True)
    assert profile.index[-1] > item['Payment_Date'] and (
        profile.index > pd.Timestamp('2026-09-15')).sum() > 3
    for at, row in profile.iterrows():
        assert row.mean() == pytest.approx(rolled([item], at), rel=1e-12, abs=1e-9), at
    blank = copy.deepcopy(LINKER['Cashflows']['Items'])
    blank[2].update(Final_Reference_Value=106.0, Final_Reference_Date=None)
    with pytest.raises(utils.UnpriceableSchedule, match="Index references fall before an earlier"):
        marks(linker(blank)[0])
    redeeming = copy.deepcopy(LINKER['Cashflows']['Items'])
    redeeming.append(dict(redeeming[-1], Yield=utils.Percent(100.0), Accrual_Year_Fraction=1.0,
                          Is_Coupon='No'))
    for items in (redeeming, dated(LINKER['Cashflows']['Items'], lambda _: pd.Timestamp('2026-09-10')),
                  dated(LINKER['Cashflows']['Items'],
                        lambda item: B - pd.DateOffset(months=3) - (item['Payment_Date'] - B) / 4)):
        job, value = linker(items)
        assert float.fromhex(marks(job)['LINKER']) == pytest.approx(value, rel=1e-12)


def test_a_static_index_rolls_as_the_base_valuation_does():
    """The trialled linker, its four final references forecast, under a credit Monte Carlo whose USD
    curve is a Hull-White at zero vol and whose index is STATIC: a static index prints nothing after
    the base date, so every row projects off the publication in force on the base date at the curve,
    on a grid opening there or a month later, and every row is the base valuation rolled to it by
    hand - 40110.7281684577 at 2026-09-03.

    Killing mutations: a static index's rows reading each row's own last publication, which projects
    the base date's print off a later month - 0.2% under a month on, 4.69e-2 under two years on;
    every row reading the grid's first, which is a month late on the later grid - 2.05e-3.
    """
    items = LINKER['Cashflows']['Items']
    for grid in ('0d 1m(1m)', '1m 1m(1m)'):
        for at, row in still(linker(items)[0], False, grid).iterrows():
            assert row.mean() == pytest.approx(rolled(items, at), rel=1e-12, abs=1e-9), (grid, at)
