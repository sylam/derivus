"""ONE CLOCK: every historical estimator dates an innovation at the START of the return it
belongs to - `calc_statistics`' `shift(-frequency)`, stated once as `start_of_return`.

The consumer is `config.Config.calculate_correlations`, which concatenates each estimator's
`delta` frame on its DATE index and reads `.corr()` of the result. Two estimators that disagree
about which end of a return an innovation sits on therefore correlate day t's move in one factor
against day t+1's in the other, and a daily correlation survives that at about zero.

Gate: two daily series drawn at 0.60, one estimator per side, every cross pair reads 0.60 back.
Killing mutation: `start_of_return` dropped from any estimator's `delta` - the pairs it enters
then read 0.05 on this draw, which is what one business day does to a daily correlation.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.signal import lfilter

from derivus import stochasticprocess as sp

N = 2016                                       # eight business years
RHO = 0.60
SE = 1.0 / np.sqrt(N)                          # the sample's own standard error, 0.0223
INDEX = pd.bdate_range('2016-01-04', periods=N)
MU, VOL = 0.05 / 252.0, 0.20 / np.sqrt(252.0)

#: `(class, param, frame, the delta column carrying the return's own innovation)`. The HMM's EM
#: and the LogVar2FJ particle gate are capped short: the clock under test is not the fit.
ESTIMATORS = {
    'calc_statistics': (sp.GBMAssetPriceCalibration, {}, 'spot', 'EquityPrice.S'),
    'MarkovHMM': (sp.MarkovHMMSpotCalibration, {'N_Iter': 25}, 'spot', 'EquityPrice.S'),
    'GARCH': (sp.GARCHSpotCalibration, {}, 'spot', 'EquityPrice.S'),
    'QuadraticCarry': (sp.QuadraticCarryCurveCalibration, {}, 'carry', 'ForwardRate.C,L'),
    'BasisLinked': (sp.BasisLinkedSpotCalibration, {}, 'basis', 'ObservedBasis.P.B'),
    'LogVar2FJ': (sp.LogVar2FJCalibration, {
        'Jump_Threshold': 4.0, 'Event_Days': '', 'Particle_Count': 200, 'Random_Seed': 0,
        'Implied_Values': '', 'Scale_To_Sector': 'No'}, 'spot', 'EquityPrice.S'),
}


def _ar1(u, phi, scale):
    """An AR(1) whose one-step innovation between t and t+1 IS `u[t]`, so the estimator that
    regresses it out hands `u` back as its residual."""
    return lfilter([scale], [1.0, -phi], np.concatenate([[0.0], u]))


def frames(u, nuisance):
    """One frame per estimator, each drawn so its primary innovation column is `u` - a lognormal
    close for the three spot estimators, an AR(1) basis against an unrelated linked spot, and a
    carry pair whose level leg is that same AR(1)."""
    level, spread = _ar1(u, 0.8, 0.01), _ar1(nuisance, 0.5, 0.005)
    return {
        'spot': pd.DataFrame({'EquityPrice.S': 100.0 * np.exp(
            np.concatenate([[0.0], np.cumsum(MU + VOL * u)]))}, index=INDEX),
        'basis': pd.DataFrame({'ObservedBasis.P.B': _ar1(u, 0.7, 1.0),
                               'CommodityPrice.P': 1000.0 * np.exp(
                                   np.concatenate([[0.0], np.cumsum(0.01 * nuisance)]))},
                              index=INDEX),
        'carry': pd.DataFrame({'ForwardRate.C,1.0': level - 0.5 * spread,
                               'ForwardRate.C,2.0': level + 0.5 * spread}, index=INDEX)}


def sides():
    """The drawn pair, as `{side: {estimator: CalibrationInfo}}`. Each side gets its own nuisance
    draw, so nothing but the 0.60 is shared."""
    z = np.random.default_rng(20260916).standard_normal((4, N - 1))
    out = {}
    for side, u, nuisance in (('a', z[0], z[2]),
                              ('b', RHO * z[0] + np.sqrt(1.0 - RHO ** 2) * z[1], z[3])):
        built = frames(u, nuisance)
        out[side] = {name: cls(None, param).calibrate(built[frame], 0.0)
                     for name, (cls, param, frame, _) in ESTIMATORS.items()}
    return out


@pytest.fixture(scope='module')
def innovations():
    """The primary innovation column each estimator emits, per side."""
    return {side: {name: fit.delta[ESTIMATORS[name][3]] for name, fit in fits.items()}
            for side, fits in sides().items()}


def test_every_estimator_dates_its_first_innovation_on_the_first_observation(innovations):
    """The first return starts on the first row of the frame; under the return's END convention
    the first innovation lands on the second row instead."""
    for side, columns in innovations.items():
        for name, column in columns.items():
            assert column.index[0] == INDEX[0], (side, name, column.index[0])


def test_every_pair_of_estimators_reads_the_correlation_the_pair_was_drawn_at(innovations):
    """Every estimator on one side against every estimator on the other, joined on the dates the
    two share - 36 readings of one number. This draw realises 0.6363, 1.6 standard errors above
    the 0.60 it was drawn at, and every family reads that back: the SPREAD across the table is
    0.0024, a tenth of a standard error, which is the assertion the clock owns. One estimator off
    the clock puts 0.05 in eleven of the 36 cells."""
    table = {(left, right): a.corr(b) for left, a in sorted(innovations['a'].items())
             for right, b in sorted(innovations['b'].items())}
    assert max(abs(rho - RHO) for rho in table.values()) < 2.0 * SE, table
    assert max(table.values()) - min(table.values()) < 0.25 * SE, table
