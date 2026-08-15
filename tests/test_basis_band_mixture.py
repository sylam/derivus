"""`BasisLinkedSpotModel.Reversion_Model = 'Band_Mixture'` — the Q-Q-ruled basis law: dead-band
reversion around the slow level with 2-state Markov mixture innovations, in place of the linear
AR / GARCH-t deviation. The mixture is the law's entire fat tail (Gaussian components, no chi2);
the regime chain draws off the independent quasi stream, the HMM's own convention.

Gates and their killing mutations:

1. OFF-IDENTITY — the key absent and the key at 'Linear' generate bit-identical paths under one
   seed. Killed by a truthiness read of the field or a band expression evaluated on the off path.
2. THE LEVEL LAW — simulated deviation stats land on the fitted law's own: bounded extremes (the
   reason this law exists — the linear AR + GARCH gave $495+ where the data max is $73), the
   mixture's innovation scale, and fat tails. Killed by: the pull dropped (extremes blow past the
   gate), pull applied inside the band (kurtosis and scale die), mixture scales swapped, or the
   regime chain frozen quiet.
3. CALIBRATION ROUND TRIP — data generated under known (kappa, beta, mixture) is recovered within
   loose bounds by `Reversion_Model='Band_Mixture'`, and the same frame under the default returns
   the linear GARCH block untouched. Killed by the fit reading the unlagged level or the EM
   collapsing to one state.
4. THE REGIME RIDES THE VERBS — generate publishes `(key,'basis_regime')` (0/1, near-stationary
   occupancy); `inner_fork_seed` carries `regime0_inner`; `reseed_from_path` republishes a
   filtered 0/1 regime on a replayed path. Killed by dropping any of the three publications.

Sim harness copied from tests/test_basis_slow_mean_garch.py (the unit idiom for this process).
"""
import types

import numpy as np
import pandas as pd
import pytest
import torch

from derivus import utils
from derivus.calculation import CMC_State
from derivus.stochasticprocess import BasisLinkedSpotCalibration, BasisLinkedSpotModel

DEVICE = torch.device('cpu')
REF_DATE = pd.Timestamp('2026-04-10')
DT_C = 1.0 / 252.0

BAND = {'A': 0.0075, 'Sigma': 5.79, 'Nu': 0.0, 'Mu': 0.0, 'Phi': 0.78,
        'Reversion_Model': 'Band_Mixture', 'Band_Kappa': 3.5, 'Band_Beta': 0.24,
        'Mix_Q_Stress': 0.26, 'Mix_Sigma_Quiet': 2.22, 'Mix_Sigma_Stress': 9.63,
        'Mix_Stay_Stress': 0.45, 'Mix_P0_Stress': 0.26,
        'Slow_Mean_Lambda': 1.0 - 2.0 / 64.0, 'Mu_0': -9.57, 'Calibration_DT_Years': DT_C}
LINEAR = {'A': 0.0075, 'Sigma': 5.79, 'Nu': 5.31, 'Mu': 0.0, 'Phi': 0.78,
          'Slow_Mean_Lambda': 1.0 - 2.0 / 64.0, 'Mu_0': -9.57, 'Calibration_DT_Years': DT_C}


def _time_grid(T):
    days = np.arange(1, T + 1, dtype=np.float64)
    tg = types.SimpleNamespace()
    tg.scen_time_grid = days
    tg.time_grid_years = days * DT_C
    tg.CurrencyMap = {}
    scen = np.zeros((T, 3), dtype=np.float64)
    scen[:, utils.TIME_GRID_MTM] = days
    scen[:, utils.TIME_GRID_ScenarioPriorIndex] = np.arange(T)
    tg.scenario_grid = scen
    return tg


def _shared(B, T, seed=42, dtype=torch.float32):
    one = torch.ones(1, 1, dtype=dtype, device=DEVICE)
    s = CMC_State(cholesky=torch.eye(1, dtype=dtype), static_buffer={}, batch_size=B, one=one,
                  mcmc_sims=0, report_currency=None, seed=seed, job_id=0, num_jobs=1)
    s.reset(num_factors=1, time_grid=_time_grid(T))
    return s


def _linked(shape, dtype, seed=123):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    return 1700.0 + torch.randn(shape, generator=g, dtype=dtype, device=DEVICE).cumsum(0) * 6.0


def _basis(param, shared, T, b0=-7.35):
    p = BasisLinkedSpotModel(factor=None, param=dict(param))
    p.factor_key = utils.Factor('ObservedBasis', ('PLATINUM_CME', 'LBMA'))
    p.linked_key = utils.Factor('CommodityPrice', ('PLATINUM_CME',))
    p.precalculate(REF_DATE, _time_grid(T), torch.tensor([b0]), shared, process_ofs=0)
    Z = shared.t_random_numbers[0, :T]
    shared.t_Scenario_Buffer[p.linked_key] = _linked(Z.shape, Z.dtype)
    return p


def _run(param, T=110, B=4096, seed=42):
    shared = _shared(B, T, seed=seed)
    p = _basis(param, shared, T)
    path = p.generate(shared)
    return p, shared, path


def test_the_absent_key_and_the_declared_linear_are_the_same_run():
    base = {k: v for k, v in LINEAR.items()}
    _, _, a = _run(base)
    declared = dict(LINEAR, Reversion_Model='Linear')
    _, _, b = _run(declared)
    assert torch.equal(a, b)


def test_the_level_law_is_the_fitted_laws_own():
    p, shared, path = _run(BAND)
    mu = shared.t_Scenario_Buffer[(p.factor_key, 'basis_mu')]
    d = (path - mu).cpu().numpy()
    innov = np.diff(path.cpu().numpy(), axis=0)
    mix_sd = np.sqrt(0.74 * 2.22 ** 2 + 0.26 * 9.63 ** 2)
    assert 0.75 * mix_sd < innov.std() < 1.35 * mix_sd
    assert np.abs(d).max() < 120.0
    assert np.percentile(np.abs(d), 99.9) < 60.0
    from scipy import stats as st
    assert st.kurtosis(innov.ravel()) > 3.0


def test_the_calibration_recovers_the_band_and_the_default_stays_linear():
    rng = np.random.default_rng(3)
    n, kappa, beta, q, s1, s2, stay = 3000, 4.0, 0.3, 0.3, 1.5, 8.0, 0.5
    b = np.zeros(n)
    lvl, s = 0.0, False
    enter = q * (1 - stay) / (1 - q)
    for t in range(1, n):
        s = (rng.random() < stay) if s else (rng.random() < enter)
        dev = b[t - 1] - lvl
        pull = -beta * np.sign(dev) * max(abs(dev) - kappa, 0.0)
        b[t] = b[t - 1] + pull + (s2 if s else s1) * rng.standard_normal()
        lvl = 0.96875 * lvl + 0.03125 * b[t]
    idx = pd.bdate_range('2015-01-01', periods=n)
    frame = pd.DataFrame({'ObservedBasis.X.Y': b,
                          'CommodityPrice.X': 1000 + rng.standard_normal(n).cumsum()}, index=idx)
    cal = BasisLinkedSpotCalibration(
        'BasisLinkedSpotModel', {'Slow_Mean_Span': 63, 'Reversion_Model': 'Band_Mixture'})
    p = cal.calibrate(frame, 0.0).param
    assert p['Reversion_Model'] == 'Band_Mixture'
    assert 1.0 < p['Band_Kappa'] < 8.0 and 0.1 < p['Band_Beta'] < 0.6
    assert 0.8 < p['Mix_Sigma_Quiet'] < 2.5 and 5.5 < p['Mix_Sigma_Stress'] < 11.0
    assert 0.15 < p['Mix_Q_Stress'] < 0.5
    linear = BasisLinkedSpotCalibration(
        'BasisLinkedSpotModel', {'Slow_Mean_Span': 63, 'GARCH_Innovation': 'Yes'})
    q_lin = linear.calibrate(frame, 0.0).param
    assert 'Reversion_Model' not in q_lin and 'G_Omega' in q_lin


def test_the_regime_rides_generate_fork_and_replay():
    p, shared, path = _run(BAND)
    key = p.factor_key
    buf = shared.t_Scenario_Buffer
    buf[key] = path
    regime = buf[(key, 'basis_regime')]
    vals = set(np.unique(regime.cpu().numpy()))
    assert vals <= {0.0, 1.0}
    assert 0.1 < regime.cpu().numpy().mean() < 0.45           # near the stationary 0.26
    seeds = p.inner_fork_seed(key, buf, 5)
    assert (key, 'regime0_inner') in seeds
    p.reseed_from_path(path, shared)
    replayed = buf[(key, 'basis_regime')]
    assert replayed.shape == regime.shape
    assert set(np.unique(replayed.cpu().numpy())) <= {0.0, 1.0}
