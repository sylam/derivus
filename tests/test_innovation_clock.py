"""ONE CLOCK: every historical estimator dates an innovation at the START of the return it
belongs to - `calc_statistics`' `shift(-frequency)`, stated once as `start_of_return`.

The consumer is `config.Config.calculate_correlations`, which concatenates each estimator's
`delta` frame on its DATE index and reads `.corr()` of the result. Two estimators that disagree
about which end of a return an innovation sits on therefore correlate day t's move in one factor
against day t+1's in the other, and a daily correlation survives that at about zero.

Gate: two daily closes drawn at 0.60 in an archive document, one under the lognormal estimator
and one under GARCH - the two cheapest families; the HMM's EM and the LogVar2FJ particle filter
are the others on this clock - calibrated through `Config.calibrate_factors`, and the correlation
it writes is the drawn pair's own.
"""
import io
import json

import numpy as np
import pandas as pd

from derivus.config import Config, CustomJsonEncoder, ModelParams

N = 1008                                       # four business years
DATES = pd.bdate_range('2016-01-04', periods=N)


def test_two_estimators_read_the_correlation_the_pair_was_drawn_at(tmp_path):
    """`EquityPrice.A` under `GBMAssetPriceModel` and `EquityPrice.B` under `GARCHSpotModel`, their
    log returns drawn at 0.60 over four years: the calibrated correlation is the sample
    correlation of the two draws, 0.5886 here, to 1e-2 (measured 0.5888 - GARCH's filtered
    variance standardising one side). 0.3 s.

    Killing mutations: `calc_statistics` dating its innovation at the end of the return; GARCH's
    `delta` dated at the end - either pair then reads one day apart, about zero.
    """
    z = np.random.default_rng(20260916).standard_normal((2, N - 1))
    draws = {'A': z[0], 'B': 0.6 * z[0] + 0.8 * z[1]}
    archive = pd.DataFrame({'EquityPrice.' + name: 100.0 * np.exp(np.concatenate(
        [[0.0], np.cumsum(0.05 / 252.0 + 0.2 / np.sqrt(252.0) * draw)]))
        for name, draw in draws.items()}, index=DATES)
    archive.index.name = 'Date'
    archive.to_csv(tmp_path / 'archive.csv', sep='\t')
    retrieval = {'ID': '', 'Use_Pre_Computed_Statistics': 'Yes', 'Data_Retrieval_Parameters': {
        'Frequency': '1d', 'Calendar': '', 'Business_Days_In_Year': 252}}
    document = {
        'MarketData': {
            'System Parameters': {'Base_Currency': 'USD', 'Base_Date': DATES[-1]},
            'Model Configuration': ModelParams(({'EquityPrice': 'GBMAssetPriceModel'},
                                                {'EquityPrice': [(('ID', 'B'), 'GARCHSpotModel')]})),
            'Price Factors': {}, 'Price Models': {}, 'Correlations': {},
            'Price Factor Interpolation': ModelParams(({}, {})),
            'Bootstrapper Configuration': {}, 'Market Prices': {}},
        'Version': ['JSONVersion', '22.05.30'],
        'CalibrationConfig': {
            'MarketDataArchiveFile': {'name': str(tmp_path / 'archive.csv'), 'skiprows': 0,
                                      'index_column': 'Date'},
            'Calibrations': {
                'GBMAssetPriceModel': dict(retrieval, Method='GBMAssetPriceCalibration'),
                'GARCHSpotModel': dict(retrieval, Method='GARCHSpotCalibration')}}}
    with io.open(tmp_path / 'calibration.json', 'w', encoding='utf-8') as handle:
        handle.write(json.dumps(document, indent=1, cls=CustomJsonEncoder))
    config = Config()
    config.parse_json(str(tmp_path / 'calibration.json'))
    factors = config.fetch_all_calibration_factors()
    config.calibrate_factors(DATES[0], DATES[-1], dict(factors['present'], **factors['absent']))

    assert set(config.params['Price Models']) == {'GBMAssetPriceModel.A', 'GARCHSpotModel.B'}
    rho = config.params['Correlations'].get(('LognormalDiffusionProcess.A', 'GARCHSpotProcess.B'))
    assert rho is not None, ('the pair wrote no correlation - under the 0.2 cutoff, the reading of '
                             'two estimators a day apart', config.params['Correlations'])
    drawn = np.corrcoef(draws['A'], draws['B'])[0, 1]
    assert abs(rho - drawn) < 1e-2, (rho, drawn)
