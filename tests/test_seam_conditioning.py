"""Seam conditioning: the t0 row is DATA, and the first simulated step conditions on the
observed later-session prints (`Bridge_T0_Fix`/`Bridge_T0_Premium` on `GARCHSpotModel`,
`Chain_T0_Basis`/`Chain_T0_Premium` on `BasisLinkedSpotModel`) instead of reverting them
through downstream factors — the bridge laws place the session print ON the path to the
next own-session value, so row 1 re-anchors at the print less the premium.

Gates and their killing mutations (all RUN):

1. SPOT ROW 1 IS THE CONDITIONAL — with the field stamped, every path is the unstamped run
   times exp(shift) from row 1 on, shift = log(fix_t0/S0) − premium; row 0 is untouched.
   EXACT (same seeds, additive log drift), so a mutant that drops the shift, flips the
   premium, or smears it over later rows dies here.
2. ABSENT IS BITWISE OFF — no field, identical tensors to the pre-feature law.
   (Carried by gate 1's unstamped arm being the baseline of an exact ratio.)
3. A FORK IS NOT THE SEAM — a grid anchored past t0 (scen_time_grid[0] > 0) never applies
   the shift, stamped or not. Kills a mutant that drops the t0-anchor test.
4. BAND ROW 1 RE-ANCHORS AT THE PARTNER — out[1] shifts by exactly
   (Chain_T0_Basis − b0 − premium) against the unstamped run (same draws); row 0 untouched.
   Kills a mutant that drops the t==1 ds bias.
5. FORWARD == REPLAY — reseed_from_path along the stamped run's own path republishes the
   forward run's slow mean (the bias lives in `_ds_linked`, shared by both). Kills a mutant
   that biases only `generate`.
"""
import types

import numpy as np
import pandas as pd
import torch

from derivus import utils
from derivus.calculation import CMC_State
from derivus.stochasticprocess import BasisLinkedSpotModel, GARCHSpotModel

DEVICE = torch.device('cpu')
DTYPE = torch.float32
REF_DATE = pd.Timestamp('2026-04-10')
DT_C = 1.0 / 252.0

SPOT = {'Omega': 8.028e-07, 'Alpha': 0.0328, 'Beta': 0.9639, 'Nu': 7.50,
        'Mu': 0.0, 'H0': 7.671e-04, 'Log_Price': True, 'Calibration_DT_Years': DT_C}
BAND = {'A': 0.0075, 'Sigma': 5.79, 'Nu': 0.0, 'Phi': 0.78,
        'Reversion_Model': 'Band_Mixture', 'Band_Kappa': 3.5, 'Band_Beta': 0.24,
        'Mix_Q_Stress': 0.26, 'Mix_Sigma_Quiet': 2.22, 'Mix_Sigma_Stress': 9.63,
        'Mix_Stay_Stress': 0.45, 'Mix_P0_Stress': 0.26,
        'Slow_Mean_Lambda': 1.0 - 2.0 / 64.0, 'Mu_0': -9.57, 'Calibration_DT_Years': DT_C}
S0, FIX_T0, PREM = 2000.0, 1960.0, -5.0e-4
B0, PARTNER, PREM_B = -7.35, -29.0, 0.26


def _time_grid(T, start=0.0):
    days = start + np.arange(T, dtype=np.float64)
    tg = types.SimpleNamespace()
    tg.scen_time_grid = days
    tg.time_grid_years = days * DT_C
    tg.CurrencyMap = {}
    scen = np.zeros((T, 3), dtype=np.float64)
    scen[:, utils.TIME_GRID_MTM] = days
    scen[:, utils.TIME_GRID_ScenarioPriorIndex] = np.arange(T)
    tg.scenario_grid = scen
    return tg


def _shared(B, T, seed, tg):
    one = torch.ones(1, 1, dtype=DTYPE, device=DEVICE)
    s = CMC_State(cholesky=torch.eye(1, dtype=DTYPE), static_buffer={}, batch_size=B, one=one,
                  mcmc_sims=0, report_currency=None, seed=seed, job_id=0, num_jobs=1)
    s.reset(num_factors=1, time_grid=tg)
    return s


def _spot_run(param, T=4, B=4096, seed=42, start=0.0):
    tg = _time_grid(T, start=start)
    sh = _shared(B, T, seed=seed, tg=tg)
    p = GARCHSpotModel(factor=types.SimpleNamespace(param={}), param=dict(param))
    p.factor_key = utils.Factor('CommodityPrice', ('TEST',))
    p.precalculate(REF_DATE, tg, torch.tensor([S0], dtype=DTYPE), sh, process_ofs=0)
    torch.manual_seed(7)
    return p.generate(sh)


def _band_run(param, T=4, B=4096, seed=42, start=0.0, replay=None):
    tg = _time_grid(T, start=start)
    sh = _shared(B, T, seed=seed, tg=tg)
    p = BasisLinkedSpotModel(factor=types.SimpleNamespace(param={}), param=dict(param))
    p.factor_key = utils.Factor('ObservedBasis', ('LBMA_AM', 'CME'))
    p.linked_key = utils.Factor('CommodityPrice', ('LBMA_AM',))
    p.precalculate(REF_DATE, tg, torch.tensor([B0], dtype=DTYPE), sh, process_ofs=0)
    Z = sh.t_random_numbers[0, :T]
    sh.t_Scenario_Buffer[p.linked_key] = torch.full(Z.shape, 2000.0, dtype=Z.dtype)  # flat ΔS=0
    torch.manual_seed(7)
    if replay is not None:
        p.reseed_from_path(replay, sh)
        return sh.t_Scenario_Buffer[(p.factor_key, 'basis_mu')]
    out = p.generate(sh)
    return out, sh.t_Scenario_Buffer[(p.factor_key, 'basis_mu')].clone()


def test_spot_row1_is_the_conditional_and_absent_is_off():
    base = _spot_run(SPOT)
    stamped = _spot_run({**SPOT, 'Bridge_T0_Fix': FIX_T0, 'Bridge_T0_Premium': PREM})
    shift = np.log(FIX_T0 / S0) - PREM
    ratio = (stamped / base).log()
    assert torch.allclose(ratio[0], torch.zeros_like(ratio[0]), atol=1e-6)
    for t in (1, 2, 3):                        # the shift lands ONCE, at row 1, and stays
        assert torch.allclose(ratio[t], torch.full_like(ratio[t], shift), atol=1e-5), \
            f'row {t}: expected uniform log-shift {shift:.6f}'


def test_a_fork_grid_is_not_the_seam():
    base = _spot_run(SPOT, start=5.0)
    stamped = _spot_run({**SPOT, 'Bridge_T0_Fix': FIX_T0, 'Bridge_T0_Premium': PREM}, start=5.0)
    assert torch.equal(stamped, base)


def test_band_row1_reanchors_at_the_partner():
    base, _ = _band_run(BAND)
    stamped, _ = _band_run({**BAND, 'Chain_T0_Basis': PARTNER, 'Chain_T0_Premium': PREM_B})
    shift = PARTNER - B0 - PREM_B
    d = stamped - base
    assert torch.allclose(d[0], torch.zeros_like(d[0]), atol=1e-6)
    assert torch.allclose(d[1], torch.full_like(d[1], shift), atol=1e-4), \
        f'row 1: expected exact re-anchor shift {shift:.4f}'
    # off a fork grid the stamped field is inert
    b2, _ = _band_run({**BAND, 'Chain_T0_Basis': PARTNER, 'Chain_T0_Premium': PREM_B}, start=5.0)
    b1, _ = _band_run(BAND, start=5.0)
    assert torch.equal(b2, b1)


def test_band_forward_equals_replay_under_the_seam():
    param = {**BAND, 'Chain_T0_Basis': PARTNER, 'Chain_T0_Premium': PREM_B}
    out, mu_fwd = _band_run(param)
    mu_rep = _band_run(param, replay=out)
    assert torch.allclose(mu_fwd, mu_rep, atol=1e-5), \
        'replayed slow mean must ride the same row-1 bias as the forward sim'
