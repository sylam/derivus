"""Seven documents that used to fail without saying what happened.

Each test authors the smallest document that reaches one site, runs it the way a caller does -
`Config.calibrate_factors`, `Config.bootstrap` or a job through `derivus.Context` - and reads what
comes back. What is asserted is the REFUSAL BY NAME, or the number the document now prices: an
engine that goes quiet at one of these sites again fails here.

Nothing is stubbed. The correlated pair runs the commodity world at 128 paths, which is enough to
tell a correlated draw from an uncorrelated one and cheap enough to gate on.
"""
import io
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

import derivus
from derivus import utils
from derivus.config import Config, CustomJsonEncoder, ModelParams

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures')


def fixture(name):
    with io.open(os.path.join(FIXTURES, name), 'rt', encoding='utf-8') as handle:
        return json.load(handle)


def authored(tmp_path, name, document):
    path = str(tmp_path / name)
    with io.open(path, 'w', encoding='utf-8') as handle:
        handle.write(json.dumps(document, indent=1, cls=CustomJsonEncoder))
    return path


def marks(out):
    """The root row of a `Credit_Monte_Carlo` report - one number per document."""
    return float(out['Results']['mtm'].values.sum())


def calibration_world(tmp_path, constant):
    """An archive whose basis column has no parent price (its calibration refuses by name) beside a
    commodity that is a live ramp, or one that never moves - which skips too."""
    archive = str(tmp_path / 'archive.csv')
    dates = pd.bdate_range('2026-01-01', periods=60)
    frame = pd.DataFrame(
        {'CommodityPrice.PLATINUM_CME': np.full(len(dates), 100.0) if constant
         else np.linspace(100.0, 130.0, len(dates)),
         'ObservedBasis.GOLD_LDN.LBMA': np.linspace(-2.0, 2.0, len(dates))}, index=dates)
    frame.index.name = 'Date'
    frame.to_csv(archive, sep='\t')
    retrieval = {'ID': '', 'Use_Pre_Computed_Statistics': 'Yes',
                 'Data_Retrieval_Parameters': {'Frequency': '1d', 'Calendar': '',
                                               'Business_Days_In_Year': 252}}
    path = authored(tmp_path, 'calibration.json', {
        'MarketData': {
            'System Parameters': {'Base_Currency': 'USD', 'Base_Date': pd.Timestamp('2026-03-25')},
            'Model Configuration': ModelParams(({'CommodityPrice': 'GBMAssetPriceModel',
                                                 'ObservedBasis': 'FixingBridgeModel'}, {})),
            'Price Factors': {}, 'Price Models': {}, 'Correlations': {},
            'Price Factor Interpolation': ModelParams(({}, {})),
            'Bootstrapper Configuration': {}, 'Market Prices': {}},
        'Version': ['JSONVersion', '22.05.30'],
        'CalibrationConfig': {
            'MarketDataArchiveFile': {'name': archive, 'skiprows': 0, 'index_column': 'Date'},
            'Calibrations': {
                'GBMAssetPriceModel': dict(retrieval, Method='GBMAssetPriceCalibration'),
                'FixingBridgeModel': dict(retrieval, Method='FixingBridgeCalibration')}}})
    config = Config()
    config.parse_json(path)
    factors = config.fetch_all_calibration_factors()
    return config, dict(factors['present'], **factors['absent'])


def test_a_calibration_that_refuses_is_reported_in_its_own_words(tmp_path, caplog):
    """Row: the skip said 'Data errors in factor' whatever the class had said."""
    config, factors = calibration_world(tmp_path, constant=False)
    with caplog.at_level(logging.ERROR):
        config.calibrate_factors(pd.Timestamp('2026-01-01'), pd.Timestamp('2026-03-25'), factors)
    assert 'no parent price column' in caplog.text and 'FixingBridgeCalibration' in caplog.text
    assert config.params['Price Models'] == {'GBMAssetPriceModel.PLATINUM_CME': pytest.approx(
        config.params['Price Models']['GBMAssetPriceModel.PLATINUM_CME'])}


def test_a_calibration_set_where_every_factor_is_skipped_is_refused(tmp_path):
    """Row: `AttributeError: 'NoneType' object has no attribute 'corr'`."""
    config, factors = calibration_world(tmp_path, constant=True)
    with pytest.raises(ValueError, match='Every one of the 2 factors asked for was skipped'):
        config.calibrate_factors(pd.Timestamp('2026-01-01'), pd.Timestamp('2026-03-25'), factors)


def test_a_family_refusing_at_construction_is_refused_by_name(tmp_path):
    """Row: the refusal was logged and the factor left unwritten for the first pricer to trip on."""
    document = fixture('hw2f_four_quote_job.json')
    market = document['Calc']['MergeMarketData']['ExplicitMarketData']
    market['Bootstrapper Configuration'] = {'LogVar2FJModelParameters': {
        'Leverage_Prior_Defaults': 'FxRate=-0.5', 'Leverage_Product_Defaults': 'FxRate=0.4'}}
    market['Market Prices'] = {'LogVar2FJModelPrices.ZAR': {'instrument': {
        'Currency': 'ZAR', 'Asset_Class': 'FX', 'European_Options': []}}}
    context = derivus.Context()
    context.load_json(authored(tmp_path, 'construction.json', document))
    with pytest.raises(ValueError, match='Bootstrapper Configuration LogVar2FJModelParameters'):
        context.current_cfg.bootstrap()


def test_a_block_with_no_quote_table_is_refused_by_name(tmp_path):
    """Row: the completed blank was iterated as a string."""
    document = fixture('hw2f_four_quote_job.json')
    market = document['Calc']['MergeMarketData']['ExplicitMarketData']
    market['Bootstrapper Configuration'] = {'InterestRateCurveParameters': {}}
    market['Market Prices'] = {'InterestRatePrices.ZAR-JIBAR-3M': {'instrument': {
        'Currency': 'ZAR', 'Day_Count': 'ACT_365', 'Discount_Rate': ''}}}
    context = derivus.Context()
    context.load_json(authored(tmp_path, 'no_quote_table.json', document))
    with pytest.raises(ValueError, match='InterestRatePrices.ZAR-JIBAR-3M carries no Points table'):
        context.current_cfg.bootstrap()


def test_a_quote_carrying_no_timestamp_still_builds_its_surface(tmp_path):
    """Row: `KeyError: 'Timestamp'` on a field declared optional."""
    document = fixture('hw2f_four_quote_job.json')
    market = document['Calc']['MergeMarketData']['ExplicitMarketData']
    market['Bootstrapper Configuration'] = {'FXVolSurfaceParameters': {}}
    market['Price Factors']['FxRate.USD'] = {
        'Domestic_Currency': None, 'Interest_Rate': 'ZAR-JIBAR-3M', 'Priority': 1, 'Spot': 17.5}
    market['Market Prices'] = {'FXVolPrices.USD.ZAR': {'instrument': {
        'Currency': 'ZAR', 'Quote_Sensitivity': 'No', 'Grid_Tolerance': 1e-4,
        'Points': [{'Use': 'Yes', 'Quote_Type': 'ATM', 'Expiry': expiry, 'Pillar': 0.0,
                    'Quoted_Market_Value': vol}
                   for expiry, vol in ((0.25, 0.14), (1.0, 0.16), (2.0, 0.17))]}}}
    context = derivus.Context()
    context.load_json(authored(tmp_path, 'no_timestamp.json', document))
    context.current_cfg.bootstrap()
    surface = context.current_cfg.params['Price Factors']['FXVol.USD.ZAR']
    assert surface['Quote_Timestamp'] == '' and len(surface['Surface'].array) > 3


def correlated_job(tmp_path, name, stale=False):
    document = fixture('commodity_aps_world.json')
    document['Calc']['Calculation'].update({'Batch_Size': 128, 'Simulation_Batches': 1})
    correlations = document['Calc']['MergeMarketData']['ExplicitMarketData']['Correlations']
    if stale:
        correlations['GARCHSpotProcess.PLATINUM_LME'] = correlations.pop(
            'GARCHSpotProcess.PLATINUM_CME')
    return authored(tmp_path, name, document)


def test_correlations_authored_in_a_job_reach_the_cholesky(tmp_path, caplog):
    """Row: an explicit section landed under a string key and read as a silent zero, so a
    correlated document needed a market-data file. The stale-key document is the control: it
    declares the same numbers under a process nothing simulates, and prices differently."""
    context = derivus.Context()
    context.load_json(correlated_job(tmp_path, 'correlated.json'))
    assert ('GARCHSpotProcess.PLATINUM_CME',
            'BasisLinkedSpotProcess.PLATINUM_CME.LBMA') in context.current_cfg.params[
                'Correlations'], 'the section is not keyed by name pair'
    live = marks(context.run_job()[1])

    stale = derivus.Context()
    stale.load_json(correlated_job(tmp_path, 'stale.json', stale=True))
    with caplog.at_level(logging.INFO):
        unread = marks(stale.run_job()[1])
    assert 'GARCHSpotProcess.PLATINUM_LME' in caplog.text, (
        'a correlation filed under a process no factor answers to went unnamed')
    assert live != unread, 'the declared correlations reach no draw'


def test_an_implied_model_with_no_price_models_block_is_refused_by_name(tmp_path):
    """Row: `TypeError: 'NoneType' object is not subscriptable` naming neither field nor factor."""
    document = fixture('commodity_aps_world.json')
    document['Calc']['Calculation'].update({'Batch_Size': 128, 'Simulation_Batches': 1})
    market = document['Calc']['MergeMarketData']['ExplicitMarketData']
    market['Price Factors']['HullWhite2FactorModelParameters.USD-SOFR'] = {
        'Quanto_FX_Correlation_1': 0.0, 'Quanto_FX_Correlation_2': 0.0,
        'Quanto_FX_Volatility': utils.Curve([], [[0.0, 0.0]]),
        'Alpha_1': 0.1, 'Sigma_1': utils.Curve([], [[0.0, 0.005]]),
        'Alpha_2': 0.2, 'Sigma_2': utils.Curve([], [[0.0, 0.004]]), 'Correlation': -0.5}
    market['Model Configuration']['.ModelParams']['modeldefaults'][
        'InterestRate'] = 'HullWhite2FactorImpliedInterestRateModel'
    context = derivus.Context()
    context.load_json(authored(tmp_path, 'implied_no_block.json', document))
    with pytest.raises(ValueError, match='Price Models.HullWhite2FactorImpliedInterestRateModel'
                                         '.USD-SOFR is not declared'):
        context.run_job()


def test_a_hessian_without_a_gradient_is_refused_by_name(tmp_path):
    """Row: the block was a silent no-op - the run reported neither."""
    document = fixture('commodity_aps_world.json')
    document['Calc']['Calculation'].update({
        'Batch_Size': 128, 'Simulation_Batches': 1,
        'Credit_Valuation_Adjustment': {
            'Calculate': 'Yes', 'Counterparty': 'CPTY', 'Bank': '', 'Gradient': 'No',
            'Hessian': 'Yes', 'Deflate_Stochastically': 'No', 'Stochastic_Hazard_Rates': 'No'}})
    context = derivus.Context()
    context.load_json(authored(tmp_path, 'hessian.json', document))
    with pytest.raises(ValueError, match="Hessian: 'Yes' needs Gradient: 'Yes'"):
        context.run_job()


@pytest.mark.parametrize('period', [pd.DateOffset(months=6, days=2),
                                    pd.DateOffset(days=2, months=6)])
def test_a_two_unit_period_is_written_in_one_order(tmp_path, period):
    """Row: the units came off a set iteration, so the file's bytes were the process's."""
    config = Config()
    config.params['Price Factors'] = {'InterestRate.ZAR-JIBAR-3M': {
        'Currency': 'ZAR', 'Day_Count': 'ACT_365', 'Sub_Type': None, 'Reset_Frequency': period}}
    path = str(tmp_path / 'market.json')
    config.write_marketdata_json(path)
    written = json.loads(io.open(path, encoding='utf-8').read())
    spelling = written['MarketData']['Price Factors'][
        'InterestRate.ZAR-JIBAR-3M']['Reset_Frequency']['.DateOffset']
    assert spelling == '6M2D', 'the units are not in the declared order'
    assert dict(config.parse_period(spelling).kwds) == {'months': 6, 'days': 2}
