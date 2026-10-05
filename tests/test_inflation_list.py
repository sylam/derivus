"""An inflation-linked cashflow list pays its fixed interest grown by the ratio of its final to its
base index reference. A reference states its level, or names a date the level is read off: the
index's prints where published, projected off the last print at the inflation curve otherwise.

The reference is the linker by hand off `trial_rates`' own index: printed monthly at 0.2% a month
to June 2026, then growing at the curve's flat 2.5%, and read three months back, interpolated.
"""

import copy
import itertools
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


def in_force(at):
    """The month whose print is in force on `at`: June's until `Next_Publication_Date`, a month
    later at each monthly publication from it."""
    prints = trial_rates.FACTORS['PriceIndex.INFL']
    return prints['Last_Period_Start'] + pd.DateOffset(months=sum(
        prints['Next_Publication_Date'] + pd.DateOffset(months=k) <= at for k in range(60)))


def index_level(day, at=B, drift=0.025, zero=None):
    """INFL at a month start as a row on `at` reads it: its print where published by the base date,
    the last print grown at `drift` where printed by `at`, and past that the row's print grown on at
    the curve's 2.5% - or along the t0 forward from the row of the curve `zero` (years -> rate)."""
    prints = trial_rates.FACTORS['PriceIndex.INFL']
    published = prints['Last_Period_Start']
    if day <= published:
        knots = prints['Index'].array
        return float(np.interp((day - utils.excel_offset).days, knots[:, 0], knots[:, 1]))
    printed = min(day, in_force(at))
    if zero is None:
        return index_level(published) * math.exp(
            (drift * (printed - published).days + 0.025 * (day - printed).days) / 365.0)
    t, T = (at - B).days / 365.0, (at - B + (day - printed)).days / 365.0
    return index_level(published) * math.exp(
        drift * (printed - published).days / 365.0 + zero(T) * T - zero(t) * t)


def reference_level(date, at=B, drift=0.025, zero=None, lag=3, interpolated=True):
    """The `lag`-month reference at `date`: interpolated, the prints `lag` and `lag - 1` months back
    weighted by how far into its own month `date` sits; single, the print `lag` months back."""
    start = date.to_period('M').to_timestamp()
    weight = (date - start).days / ((start + pd.DateOffset(months=1)) - start).days if interpolated else 0.0
    return sum(share * index_level((date - pd.DateOffset(months=back)).to_period('M').to_timestamp(),
                                   at, drift, zero) for back, share in ((lag, 1.0 - weight), (lag - 1, weight)))


def rolled(items, at, drift=0.025, zero=None, lag=3, interpolated=True):
    """`items` valued by hand at `at`: `Notional Rate_Multiplier final / base Yield accrual DF(pay)`
    summed over the cashflows paying on or after it, USD at 4%."""
    def reference(item, ref):
        return item[ref + '_Reference_Value'] or reference_level(
            item[ref + '_Reference_Date'], at, drift, zero, lag, interpolated)

    return sum(item['Notional'] * item['Rate_Multiplier'] * item['Yield'].amount *
               item['Accrual_Year_Fraction'] * math.exp(-book.DISCOUNT[0] * (
                   item['Payment_Date'] - at).days / 365.0) *
               reference(item, 'Final') / reference(item, 'Base')
               for item in items if item['Payment_Date'] >= at)


def linker(items):
    """`trial_rates`' LINKER paying `items`, its one-deal wire document and its value by hand."""
    deal = dict(LINKER, Cashflows={'Items': items})
    return document(SimpleNamespace(DEALS=[deal], FACTORS=trial_rates.FACTORS, CONFIGURATION={})), \
        rolled(items, B)


def still(job, drift, grid='0d 1m(1m)', vol=0.0, paths=16, nomodel='Constant', curves=('USD',)):
    """`job`'s answer on `grid` under a credit Monte Carlo in float64 reading `nomodel`, its
    `curves` Hull-Whites at zero vol, its index static where `drift` is None, else a GBM at `vol`
    drifting at `drift` on its own 365.25-day clock."""
    deals = job['Calc']['Deals']['Deals']
    deals['Children'] = [{'Instrument': {'.Deal': {
        'Object': 'NettingCollateralSet', 'Reference': 'NS', 'Netted': 'True',
        'Collateralized': 'False'}}, 'Children': deals['Children']}]
    job['Calc']['Calculation'] = {
        'Object': 'CreditMonteCarlo', 'Base_Date': B, 'Currency': 'USD', 'Time_Grid': grid,
        'Batch_Size': paths, 'Simulation_Batches': 1, 'Random_Seed': 1, 'Deflation_Interest_Rate': 'USD',
        'NoModel': nomodel}
    defaults = {kind: 'HullWhite1FactorInterestRateModel' for kind, curve in (
        ('InterestRate', 'USD'), ('InflationRate', 'INFL')) if curve in curves}
    models = {'HullWhite1FactorInterestRateModel.' + curve: {
        'Alpha': 0.05, 'Lambda': 0.0, 'Quanto_FX_Correlation': 0.0,
        'Quanto_FX_Volatility': utils.Curve([], [[0.0, 0.0], [10.0, 0.0]]),
        'Sigma': utils.Curve([], [[0.0, 0.0], [10.0, 0.0]])} for curve in curves}
    if drift is not None:
        defaults['PriceIndex'] = 'GBMPriceIndexModel'
        models['GBMPriceIndexModel.INFL'] = {'Vol': vol, 'Drift': drift * 365.25 / 365.0}
    market = job['Calc']['MergeMarketData']['ExplicitMarketData']
    market['Model Configuration'] = {'.ModelParams': {'modeldefaults': defaults, 'modelfilters': {}}}
    market['Price Models'] = models
    context = derivus.Context()
    context.load_json((json.dumps(job, cls=CustomJsonEncoder), 'linker_still'))
    return derivus.run_cmc(context.current_cfg, prec=torch.float64)[1]


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
    simulates the USD curve, holds the index static and reads its static curves risk-neutral, as a
    deal on a constant index asks: every row and path is finite and the first row is the base
    valuation's mark.

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
            'Random_Seed': 1, 'Deflation_Interest_Rate': 'USD', 'NoModel': 'RiskNeutral'}
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
    profile = still(job, 0.025)['Results']['mtm']
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


def test_a_constant_index_prices_only_where_its_curves_read_forward(caplog):
    """A price index with no model is constant. The trialled linker on one, under a credit Monte
    Carlo whose USD curve is a Hull-White at zero vol: under `NoModel: RiskNeutral`, beside a static
    inflation curve sloped from 1% to 5% over ten years, every row is the constant print in force
    there grown along the curve's t0 forward from the row, by hand off the publication calendar, on
    a grid opening on the base date and on one opening a month later; beside the trial's flat curve
    under a Hull-White at zero vol it prices the same way. Under the default `Constant`, beside a
    fixed list, it is skipped by name and counted, the modelled curve or not, and refused by name
    under `Exclude_Deals_With_Missing_Market_Data: No`; alone, the run valued nothing and says so,
    naming it and why.

    The profile is NOT the base valuation rolled forward: the growth from the base date to a row is
    never realised on a constant print - on the trial linker two years on, 4.69% under it.

    Killing mutations: the rule keyed on the curve's model instead of the switch; the skip removed,
    the deal priced under `Constant`; the base date's publication read on every row; a structure
    holding only skipped deals counted as valued, which frames the run's one scalar and dies.
    """
    items = LINKER['Cashflows']['Items']
    sloped, job = linker(items)[0], linker(items)[0]
    sloped['Calc']['MergeMarketData']['ExplicitMarketData']['Price Factors']['InflationRate.INFL'][
        'Curve'] = utils.Curve([], [[0.0, 0.01], [10.0, 0.05]])
    for grid in ('0d 1m(1m)', '1m 1m(1m)'):
        for at, row in still(copy.deepcopy(sloped), None, grid, nomodel='RiskNeutral')['Results'][
                'mtm'].iterrows():
            assert row.mean() == pytest.approx(rolled(items, at, 0.0, lambda years: np.interp(
                years, (0.0, 10.0), (0.01, 0.05))), rel=1e-12, abs=1e-9), (grid, at)
    modelled = still(copy.deepcopy(job), None, nomodel='RiskNeutral', curves=('USD', 'INFL'))
    for at, row in modelled['Results']['mtm'].iterrows():
        assert row.mean() == pytest.approx(rolled(items, at, 0.0), rel=1e-12, abs=1e-9), at
    pair = document(SimpleNamespace(DEALS=[dict(LINKER, Cashflows={'Items': items}), next(
        deal for deal in trial_rates.DEALS if deal['Reference'] == 'FIXED_LIST')],
        FACTORS=trial_rates.FACTORS, CONFIGURATION={}))
    for curves in (('USD',), ('USD', 'INFL')):
        caplog.clear()
        assert still(copy.deepcopy(pair), None, curves=curves)['Stats']['Deals Skipped'] == 1
        assert 'Deal LINKER skipped' in caplog.text and 'NoModel is Constant' in caplog.text
    pair['Calc']['MergeMarketData']['ExplicitMarketData']['System Parameters'][
        'Exclude_Deals_With_Missing_Market_Data'] = 'No'
    with pytest.raises(utils.UnpriceableSchedule, match='Deal LINKER could not be priced'):
        still(pair, None)
    with pytest.raises(utils.UnpriceableSchedule, match=r'Nothing in the book was valued \(1 Deals '
                                                        r'Skipped\).*LINKER - the price index of INFL'):
        still(copy.deepcopy(job), None)


def test_a_simulated_index_reads_a_month_where_its_print_is_in_force():
    """The trialled linker; the same with every base reference on 2026-09-10, off June's print and
    July's, which the base date has not printed; and its first coupon fixing on 2026-09-15 - under
    a credit Monte Carlo whose USD curve is a Hull-White at zero vol and whose index is a GBM at
    zero vol, which holds at each scenario date the print in force there. Drifting at the curve's
    2.5%, every row is the base valuation rolled to it by hand; at 3%, a month the row has printed
    reads the last print grown at 3% and a later one the row's print grown on at 2.5% - payment
    rows included, on a monthly grid, where every month's print is in force on a scenario date. At
    1% vol row 0 is the base valuation and the first payment row's mean over 4,096 paths its
    expectation by hand within 4 standard errors. On a quarterly grid July is in force on no
    scenario date: it is read on the index's clock between June's print and September's, 30/92 of
    the way, and the first coupon's rows are that read by hand.

    Killing mutations: the lookup back on the month's first day, where the print in force is two
    months older - each payment row short; the lookup a publication period late; a month the base
    date has not printed read at its t0 forward; the rows left in their printed-then-simulated
    order, which the shared base reference interleaves; the clock's weight dropped, which reads
    July as June's print on the quarterly grid.
    """
    items = LINKER['Cashflows']['Items']
    cases = (items, dated(items, lambda _: pd.Timestamp('2026-09-10')),
             [dict(items[0], Final_Reference_Date=pd.Timestamp('2026-09-15'))])
    for case, drift in itertools.product(cases, (0.025, 0.03)):
        for at, row in still(linker(case)[0], drift)['Results']['mtm'].iterrows():
            assert row.mean() == pytest.approx(rolled(case, at, drift), rel=1e-12, abs=1e-9), (drift, at)
    profile = still(linker(items)[0], 0.025, vol=0.01, paths=4096)['Results']['mtm']
    pay = items[0]['Payment_Date']
    assert profile.iloc[0].mean() == pytest.approx(rolled(items, B), rel=1e-12)
    assert abs(profile.loc[pay].mean() - rolled(items, pay)) <= 4 * profile.loc[pay].std() / 64
    june, july, september = (pd.Timestamp('2026-%02d-01' % month) for month in (6, 7, 9))
    read = index_level(june) + (index_level(september) - index_level(june)) * 30 / 92
    exact = reference_level(cases[2][0]['Final_Reference_Date'])
    coarse = exact + 14 / 30 * (read - index_level(july))
    for at, row in still(linker(cases[2])[0], 0.025, '0d 3m(3m)')['Results']['mtm'].iterrows():
        hand = rolled(cases[2], at) * (coarse / exact if at > B else 1.0)
        assert row.mean() == pytest.approx(hand, rel=1e-12, abs=1e-9), at


def test_two_lists_on_one_simulated_index_read_a_shared_month_alike():
    """Two lists on the index simulated as above at 3%, on a one-month single lag and their base
    references dated at their accrual starts: the same reference dates, the second paid two months
    after the first, so the first's last final month - printed after the first's last payment - is
    read by the second alone. Whichever list the netting set prices first, every row on the weekly
    grid is the two lists by hand.

    Killing mutation: the read cached on its reset days alone, which hands the second list the
    first's read clipped at its own payments - the later list's last coupon 2.545e-3 short where the
    first is priced first.
    """
    first = dated(LINKER['Cashflows']['Items'][:3], lambda item: item['Accrual_Start_Date'])
    later = [dict(item, Payment_Date=item['Payment_Date'] + pd.DateOffset(months=2)) for item in first]
    single = dict(LINKER['Index_Reference'], Months_Lag=1, Reference_Type='Single')
    lists = [dict(LINKER, Reference=name, Index_Reference=single, Cashflows={'Items': items})
             for name, items in (('FIRST', first), ('LATER', later))]
    for deals in (lists, lists[::-1]):
        job = document(SimpleNamespace(DEALS=deals, FACTORS=trial_rates.FACTORS, CONFIGURATION={}))
        for at, row in still(job, 0.03, '0d 1w(1w)')['Results']['mtm'].iterrows():
            if (at - B).days % 7 == 0:
                hand = sum(rolled(items, at, 0.03, lag=1, interpolated=False) for items in (first, later))
                assert row.mean() == pytest.approx(hand, rel=1e-12, abs=1e-9), (deals[0]['Reference'], at)
