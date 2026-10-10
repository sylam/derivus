"""Only the base currency's FX rate is static (it is identically one); its curve simulates like any
other. Found 2026-09-03: `find_models` excluded every factor NAMED by the base currency, so a
USD-base book could not simulate USD rates whatever model it declared - a USD float leg under a
credit Monte Carlo read its resets one scenario wide and was skipped.

The oracle is the code's own dispersion: a par swap's exposure profile has zero spread across
scenarios when its curve is static and a positive one when it is simulated. Degeneracy: r = 4% so
the discount is live; the swap is at par so row 0 is near zero and the later rows are the risk;
one netting set, one currency - the axis under test is the base flag alone.
"""
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import pytest
import torch

import derivus
import rates_world as rw
import test_declared_defaults as book
import trial_fx
from derivus import utils
from derivus.config import CustomJsonEncoder

BASE = rw.BASE
HW1F = {'Alpha': 0.1, 'Lambda': 0.0, 'Sigma': utils.Curve([], [[1.0 / 365.0, 0.01], [5.0, 0.01]]),
        'Quanto_FX_Correlation': 0.0, 'Quanto_FX_Volatility': utils.Curve([], [[1.0 / 365.0, 0.0], [5.0, 0.0]])}


def job(base, fx_model=False):
    """A 2y USD par swap under a credit Monte Carlo on a book whose base currency is `base`, with a
    Hull-White model declared on USD and, optionally, a GBM model declared on `FxRate.USD`."""
    factors = {'FxRate.USD': {'Domestic_Currency': None if base == 'USD' else base, 'Interest_Rate': 'USD',
                              'Priority': 1, 'Spot': 1.0 if base == 'USD' else 0.92},
               'InterestRate.USD': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                                    'Curve': utils.Curve([], [[1.0 / 365.0, 0.04], [5.0, 0.04]])}}
    if base != 'USD':
        factors['FxRate.' + base] = {'Domestic_Currency': None, 'Interest_Rate': base, 'Priority': 1, 'Spot': 1.0}
        factors['InterestRate.' + base] = {'Currency': base, 'Day_Count': 'ACT_365', 'Sub_Type': None,
                                           'Curve': utils.Curve([], [[1.0 / 365.0, 0.02], [5.0, 0.02]])}
    models = {'HullWhite1FactorInterestRateModel.USD': dict(HW1F)}
    defaults = {'InterestRate': 'HullWhite1FactorInterestRateModel'}
    if fx_model:
        models['GBMAssetPriceModel.USD'] = {'Vol': 0.1, 'Drift': 0.0}
        defaults['FxRate'] = 'GBMAssetPriceModel'
    market = {'System Parameters': {'Base_Currency': base, 'Base_Date': BASE}, 'Valuation Configuration': {},
              'Price Factors': factors, 'Price Models': models,
              'Model Configuration': {'.ModelParams': {'modeldefaults': defaults, 'modelfilters': {}}}}
    calc = {'Object': 'CreditMonteCarlo', 'Base_Date': BASE, 'Currency': base, 'Time_grid': '0d 2y(3m)',
            'Batch_Size': 64, 'Simulation_Batches': 1, 'Random_Seed': 1, 'Deflation_Interest_Rate': base}
    return {'Calc': {'Calculation': calc,
                     'Deals': {'Reference': 'base', 'Deals': {'Children': [
                         {'Instrument': {'.Deal': rw.par_swap('SW', 'USD', 'USD', 'USD', 2, 4.0)}}]}},
                     'MergeMarketData': {'MarketDataFile': '', 'ExplicitMarketData': market}}}


def priced(doc, tmp_path, name):
    path = os.path.join(str(tmp_path), name + '.json')
    with open(path, 'w') as f:
        f.write(json.dumps(doc, cls=CustomJsonEncoder))
    cx = derivus.Context()
    cx.load_json(path)
    return (cx,) + cx.run_job()


def run(doc, tmp_path, name):
    cx, _, out = priced(doc, tmp_path, name)
    profile = out['Results']['mtm'].values
    logging.debug('%s: profile %s, spread across scenarios per row %s', name, profile.shape,
                  [round(float(x), 4) for x in profile.std(axis=1)])
    return cx, profile


def test_the_base_currency_s_curve_simulates_like_any_other(tmp_path):
    """A USD swap on a USD-base book with a Hull-White model on USD walks a DISPERSED profile - the
    same document on an EUR-base book is the witness that dispersion is what a simulated curve
    looks like here. Before the fix the USD-base run had nothing to simulate at all.

    Killing mutation: every factor named by the base currency held static, its curve included."""
    _, usd_base = run(job('USD'), tmp_path, 'usd_base')
    _, eur_base = run(job('EUR'), tmp_path, 'eur_base')
    assert usd_base.shape[0] > 4 and np.isfinite(usd_base).all()
    assert usd_base[4].std() > 0.0 and eur_base[4].std() > 0.0, (usd_base[4].std(), eur_base[4].std())
    assert usd_base[0].std() == 0.0, 'row 0 is today: one number across scenarios'


def test_the_base_currency_s_fx_rate_stays_static_whatever_model_is_declared():
    """`FxRate.USD` on a USD book is identically one: a GBM model declared for it is ignored and it
    never enters the stochastic set the run simulates, while the curve beside it does.

    Killing mutation: the base currency's FX rate let into the stochastic set by its model."""
    context = derivus.Context()
    context.load_json((json.dumps(job('USD', fx_model=True), cls=CustomJsonEncoder), 'fx_model'))
    calc, _ = derivus.run_cmc(context.current_cfg)
    factors = {(type(process).__name__, key.type, key.name) for key, process in calc.stoch_factors.items()}
    assert ('HullWhite1FactorInterestRateModel', 'InterestRate', ('USD',)) in factors, factors
    assert not any(kind == 'FxRate' for _, kind, _ in factors), factors


def test_a_static_spot_beside_a_simulated_curve_prices_as_a_spot_that_never_moves():
    """A static `FxRate` is one row where a simulated curve beside it is one per date, so the row
    loops of the one-touch, the barrier, the partial-time barrier and the extendable forward ran
    once and gathered past their own end, and the TARF and accumulator read one fixing where the
    schedule holds several. Each takes its spot on its own grid now. The fx trial family, a TARF and
    an accumulator on a static EURUSD beside a Hull-White on USD at zero volatility price, row for
    row and bit for bit, as the same book whose EURUSD is a GBM at zero vol and drift - a simulated
    spot that never moves - and row 0 of the closed-form six is their base valuation to 1e-13.

    Killing mutations: the spot left one row in any of the six types, or the fixings in either
    pricer reading them.
    """
    deals = trial_fx.DEALS + [deal for deal in book.BOOK if deal['Reference'] in ('TARF', 'ACC')]

    def rows(simulated, **still):
        calc, _ = book.simulated(deals, ('USD',), sigma=0.0, prec=torch.float64,
                                 Generate_Cashflows='No', **still)
        assert (('FxRate', ('EUR',)) in {(key.type, key.name) for key in calc.stoch_factors}) == simulated
        return {deal.Instrument.field['Reference']: deal.Calc_res['Value'][0]
                for deal in calc.netting_sets.deals()}

    static = rows(False)
    still = rows(True, models={'FxRate.EUR': ('GBMAssetPriceModel', {'Vol': 0.0, 'Drift': 0.0})})
    assert set(static) == set(still) == {deal['Reference'] for deal in deals}
    for ref, profile in static.items():
        assert profile.shape == still[ref].shape and np.array_equal(profile, still[ref]), ref
    marks, _ = book.marks(trial_fx.DEALS)
    for ref in ('FXASN', 'FXDASN', 'FXOT', 'FXNT', 'FXPKO', 'FXKOR'):
        assert static[ref][0].mean() == pytest.approx(float.fromhex(marks[ref]), rel=1e-13), ref


def rooted(deals, vol, netted=False):
    """`deals` at the root of a quarterly credit Monte Carlo, or under one uncollateralised netting
    set, every curve, the EURUSD and the equity simulated at volatility `vol`: its mtm frame."""
    doc = json.loads(json.dumps(book.book(deals), cls=CustomJsonEncoder))
    calc = doc['Calc']
    calc['Calculation'] = dict(Object='CreditMonteCarlo', Base_Date=calc['Calculation']['Base_Date'],
                               Currency='USD', Time_Grid='0d 3m(3m)', Batch_Size=64, Random_Seed=1,
                               Deflation_Interest_Rate='USD', Generate_Cashflows='No')
    if netted:
        calc['Deals']['Deals']['Children'] = [{'Instrument': {'.Deal': {
            'Object': 'NettingCollateralSet', 'Reference': 'NS', 'Netted': 'True',
            'Collateralized': 'False'}}, 'Children': calc['Deals']['Deals']['Children']}]
    market = calc['MergeMarketData']['ExplicitMarketData']
    market['Model Configuration'] = {'.ModelParams': {'modeldefaults': {
        'InterestRate': 'HullWhite1FactorInterestRateModel', 'FxRate': 'GBMAssetPriceModel',
        'EquityPrice': 'GBMAssetPriceModel'}, 'modelfilters': {}}}
    sigma = json.loads(json.dumps(utils.Curve([], [[0.0, vol], [10.0, vol]]), cls=CustomJsonEncoder))
    market['Price Models'] = dict({'HullWhite1FactorInterestRateModel.' + curve: dict(
        HW1F, Alpha=0.05, Sigma=sigma, Quanto_FX_Volatility=sigma) for curve in ('USD', 'USD-PROJ')},
        **{'GBMAssetPriceModel.' + name: {'Vol': vol, 'Drift': 0.0} for name in ('EUR', 'EQ')})
    market['Price Models'] = json.loads(json.dumps(market['Price Models'], cls=CustomJsonEncoder))
    context = derivus.Context()
    context.load_json((json.dumps(doc), 'rooted'))
    return context.run_job()[1]['Results']['mtm']


def test_a_structure_at_the_root_reports_on_every_date_its_deals_do():
    """A credit Monte Carlo read its report dates off its netting sets, any other container answering
    its own sparse revaluation dates, so a structure or a swaption at the root left the frame short
    of the rows its deals priced and the run died on it. Every deal answers the grid's dates to its
    last now, as a netting set always did, and the root reads its own deals' beside its structures'.
    The core trial family at the root prices, every row finite. A three-month collar, a swaption over
    its legs, an option and a cashflow outliving them read, at zero volatility and on every date the
    root reports, bit for bit what the same tree reads under one uncollateralised netting set - the
    set reporting the day after each settlement besides. What the set's day does not allow: under
    live volatility its added days re-lay the draws and the paths part, and at the root a deal is
    read across a day after that another deal's legs put on the grid, the set revaluing it there.

    Killing mutations: a deal answering its revaluation dates again; the root's own deals left out.
    """
    import test_position_scaling
    core = rooted(test_position_scaling.CORE.DEALS, 0.01)
    assert core.shape[0] > 4 and np.isfinite(core.values).all()
    swaption = next(deal for deal in book.BOOK if deal['Reference'] == 'SWPT')
    expiry = book.WORLD_BASE + pd.DateOffset(months=3)
    collar = {'Object': 'StructuredDeal', 'Reference': 'COLLAR', 'Currency': 'USD', 'Children': [
        book.fx_leg('FXOptionDeal', 'COLLAR_CALL', Expiry_Date=expiry, Strike_Price=1.30),
        book.fx_leg('FXOptionDeal', 'COLLAR_PUT', Expiry_Date=expiry, Strike_Price=1.20,
                    Option_Type='Put', Buy_Sell='Sell')]}
    deals = [collar, swaption, book.fx_leg('FXOptionDeal', 'FXO', Expiry_Date=expiry),
             {'Object': 'FixedCashflowDeal', 'Reference': 'CF', 'Currency': 'USD', 'Discount_Rate': 'USD',
              'Amount': 250_000.0, 'Payment_Date': book.WORLD_EXPIRY + pd.DateOffset(years=5)}]
    root, netted = rooted(deals, 0.0), rooted(deals, 0.0, netted=True)
    assert root.index.isin(netted.index).all() and len(netted) > len(root)
    assert root.index[-1] == book.WORLD_EXPIRY + pd.DateOffset(years=5)
    assert np.array_equal(root.values, netted.loc[root.index].values)


CSA = {'.CreditSupportList': [[0.0, 0.0]]}


def netting_set(reference, children, liquidation=None):
    """An uncollateralised netting set, or with `liquidation` a zero-threshold CSA closing out over
    that many days and no settlement period."""
    terms = {} if liquidation is None else {
        'Collateralized': 'True', 'Agreement_Currency': 'USD', 'Balance_Currency': 'USD',
        'Liquidation_Period': liquidation, 'Settlement_Period': 0, 'Credit_Support_Amounts': dict.fromkeys(
            ('Received_Threshold', 'Posted_Threshold', 'Independent_Amount', 'Minimum_Received',
             'Minimum_Posted'), CSA)}
    return dict({'Object': 'NettingCollateralSet', 'Reference': reference, 'Netted': 'True',
                 'Collateralized': 'False', 'Children': children}, **terms)


def forward(reference, units, years):
    return {'Object': 'EquityForwardDeal', 'Reference': reference, 'Currency': 'USD', 'Equity': 'EQ',
            'Discount_Rate': 'USD', 'Buy_Sell': 'Buy', 'Units': units, 'Forward_Price': 95.0,
            'Maturity_Date': book.WORLD_BASE + pd.DateOffset(years=years)}


def cashflow(reference, days):
    return {'Object': 'FixedCashflowDeal', 'Reference': reference, 'Currency': 'USD',
            'Discount_Rate': 'USD', 'Amount': 1_000.0, 'Payment_Date': book.WORLD_BASE + pd.DateOffset(days=days)}


def test_netting_sets_side_by_side_read_as_each_one_alone():
    """The root adds its netting sets row by row on the global grid and reports its own dates off
    it, so every set side by side reads, on every date it reports alone, what it reads alone - at
    zero volatility, where the dates one set adds cannot move another's paths. An uncollateralised
    set holding a two-year forward beside one paying on days 45 and 120 reads its forward once the
    cashflows are paid; beside a ten-day CSA over a one-year forward, the two add up on the dates both
    report and the plain set reads alone once the CSA has expired. A second CSA closing out over
    another period is refused by name.

    Killing mutations: the uncollateralised set interpolating its total a second time off its own
    dates (100.04 against 77.30 on the slide's case); the CSA set handing back its own report rows,
    which slide past the look-back dates the plain set reports.
    """
    long = netting_set('P1', [forward('FWD2', 10.0, 2)])
    paid = netting_set('P2', [cashflow('CF45', 45), cashflow('CF120', 120)])
    csa = netting_set('C1', [forward('FWD1', 1_000.0, 1)], liquidation=10)
    for pair in ((long, paid), (long, csa)):
        alone = [rooted([deal], 0.0).mean(axis=1) for deal in pair]
        joint = rooted(list(pair), 0.0).mean(axis=1)
        last = max(alone[1].index)
        compared = 0
        for day in joint.index:
            parts = [frame.get(day) for frame in alone]
            if day > last:
                parts[1] = 0.0
            if None not in parts:
                compared += 1
                assert abs(joint[day] - sum(parts)) <= 1e-3 * max(1.0, abs(sum(parts))), (
                    pair[1]['Reference'], day, joint[day], parts)
        assert compared >= 4, (pair[1]['Reference'], compared)
    with pytest.raises(utils.UnpriceableSchedule) as refusal:
        rooted([csa, netting_set('C2', [forward('FWD3', 1.0, 1)], liquidation=5)], 0.0)
    assert all(word in str(refusal.value) for word in ('C1', 'C2', 'Liquidation_Period')), refusal.value


def test_a_deal_settled_before_the_base_date_is_skipped_by_name_under_a_scenario_grid(tmp_path, caplog):
    """A forward whose settlement date is behind the base date has expired, and a credit Monte Carlo
    skips it by name as a base valuation does: the profile is the book without it to the bit and
    the log carries the expiry, never a pricing failure. Before the fix the scenario grid priced
    it at a negative maturity - an index past the curve's rows, which on the card is a device
    assertion that takes the whole run down.

    Killing mutation: the expiry rule read on the single-date grid alone.
    """
    doc = job('EUR')
    settled = {'Object': 'FXForwardDeal', 'Reference': 'SETTLED', 'Buy_Currency': 'USD',
               'Buy_Amount': 1_000_000.0, 'Buy_Discount_Rate': 'USD', 'Sell_Currency': 'EUR',
               'Sell_Amount': 900_000.0, 'Sell_Discount_Rate': 'EUR',
               'Settlement_Date': pd.Timestamp(BASE) - pd.DateOffset(days=7)}
    doc['Calc']['Deals']['Deals']['Children'].append({'Instrument': {'.Deal': settled}})
    with caplog.at_level(logging.WARNING):
        _, with_settled = run(doc, tmp_path, 'eur_base_settled')
    _, alone = run(job('EUR'), tmp_path, 'eur_base_alone')
    assert np.array_equal(with_settled, alone)
    records = [r for r in caplog.records if 'SETTLED' in r.getMessage()]
    assert records and all(r.levelno == logging.WARNING and 'expired' in r.getMessage() for r in records), [
        (r.levelname, r.getMessage()[:80]) for r in records]


def test_at_percentile_reports_each_factor_and_the_mtm_along_the_path_nearest_the_percentile(tmp_path, caplog):
    """`Calc_Scenarios: At_Percentile` is for backtesting: at every REPORT date, the path whose mtm
    sits nearest each `Percentile` of the profile, its factors and its mtm. Every report date is
    therefore simulated - the switch turns `Dynamic_Scenario_Dates` on, named at INFO, the swap's
    reset days joining a five-month grid - and each factor is read at the report date's own scenario
    row: under a collateralised set with a liquidation period the grid carries each date's
    liquidation day too, which reports nothing, so the report rows are a strict subset of the
    scenario rows. Read against the same seed's `All` run: the mtm frames are one, and at each date
    and percentile the reported curve is the `All` run's at the chosen path, whose mtm no other
    path's is nearer the percentile than.

    Killing mutations: the switch not forced (the grids part, the draws with them); a row read by
    its position rather than at its date, the old reading, which the liquidation days expose.
    """
    csa = {'.CreditSupportList': [[0.0, 0.0]]}
    netting = {'Object': 'NettingCollateralSet', 'Reference': 'NS1', 'Netted': 'True', 'Collateralized': 'True',
               'Agreement_Currency': 'USD', 'Funding_Rate': 'USD', 'Balance_Currency': 'USD',
               'Liquidation_Period': 10.0, 'Settlement_Period': 0.0,
               'Credit_Support_Amounts': {name: csa for name in (
                   'Received_Threshold', 'Posted_Threshold', 'Independent_Amount', 'Minimum_Received', 'Minimum_Posted')}}
    docs = [job('USD') for _ in range(2)]
    for doc, scenarios in zip(docs, ('At_Percentile', 'All')):
        deals = doc['Calc']['Deals']['Deals']
        deals['Children'] = [{'Instrument': {'.Deal': netting}, 'Children': deals['Children']}]
        doc['Calc']['Calculation'].update(Time_grid='0d 2y(5m)', Percentile='5, 95', Calc_Scenarios=scenarios,
                                          Dynamic_Scenario_Dates='No' if scenarios == 'At_Percentile' else 'Yes')
    with caplog.at_level(logging.INFO):
        _, calc, out = priced(docs[0], tmp_path, 'at_percentile')
    _, _, every = priced(docs[1], tmp_path, 'all_paths')
    grid = calc.time_grid
    assert sorted(grid.scenario_dates) == sorted(grid.mtm_dates) and set(grid.scenario_dates) > set(grid.base_MTM_dates)
    assert len(grid.report_index) < len(grid.mtm_dates)
    assert any(r.levelno == logging.INFO and 'Dynamic_Scenario_Dates is Yes' in r.getMessage() for r in caplog.records)
    mtm, scen, whole = out['Results']['mtm'], out['Results']['scenarios'], every['Results']['scenarios']
    assert np.array_equal(mtm.values, every['Results']['mtm'].values)
    assert set(scen) == {'mtm', 'InterestRate.USD'} and all(f.columns.equals(mtm.index) for f in scen.values())
    chosen = {}
    for q in ('5', '95'):
        level = np.percentile(mtm.values, float(q), axis=1)[:, np.newaxis]
        chosen[q] = path = np.argmin(np.abs(mtm.values - level), axis=1)
        picked = mtm.values[np.arange(len(mtm)), path]
        assert np.array_equal(scen['mtm'].loc[(0.0, q)].values, picked)
        assert (np.abs(picked[:, np.newaxis] - level) <= np.abs(mtm.values - level)).all()
        curve = scen['InterestRate.USD'].xs(q, level='scenario')
        for i, date in enumerate(mtm.index):
            assert np.array_equal(curve[date].values, whole['InterestRate.USD'].xs(path[i], level='scenario')[date].values)
    assert (chosen['5'] != chosen['95']).any()
