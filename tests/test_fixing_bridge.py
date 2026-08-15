"""`FixingBridgeModel` + `FixingBridgeCalibration` — the intraday fixing bridged between
consecutive parent observations (the linked-parent family's open link; the closed-chain case is
`ChainedBasisModel`). The class exists for ONE measured conditional: knowing the fixing must
reduce the variance of the next parent observation by the bracket share, as the data does
(sd ratio 0.868, slope 1.006) — an independent draw of the same marginal scores zero there.

Gates and their killing mutations:

1. THE LAW IS THE BRACKET, h-SCALED — pooled OLS recovers W; the per-row residual tracks
   √(h_t·f) against a deliberately ROW-SHARP alternating h (a smooth ramp hides a one-row h
   mis-index — a mutant survived that fixture once). Killed by a flat-σ read or an h/f shift.
2. THE FIXING REDUCES THE NEXT PARENT'S VARIANCE — the named conditional: regressing the
   parent's next move on the intraday move gives slope ≈ 1 and residual sd ≈ √(1−W) of the
   unconditional. Killed by the independent-draw law (same marginal, zero placement) — the
   mutant that motivated the class.
3. BRACKET AVAILABILITY DECIDES THE OPEN ROWS — a parent outliving the grid bridges the last
   row too; an equal-length parent leaves it open at √(W·h·f_bd).
4. INNER MODE BROADCASTS; ROW 0 IS THE SPOT.
5. A COARSE GRID REFUSES LOUD — h·f is not the interval variance across sub-steps.
6. THE PARENT RESOLVES LOUD — the name minus its last period, under exactly one composable
   type (this family's own documented convention).
7. THE CALIBRATION ROUND-TRIPS — W and the premium from the pooled 1-day-pair OLS.
"""
import types

import numpy as np
import pandas as pd
import pytest
import torch

from derivus import utils
from derivus.calculation import CMC_State
from derivus.stochasticprocess import FixingBridgeCalibration, FixingBridgeModel

DEVICE = torch.device('cpu')
REF_DATE = pd.Timestamp('2026-04-10')
DT_C = 1.0 / 252.0
F_BD = (1.0 / utils.DAYS_IN_YEAR) / DT_C
W = 0.247
PREM = -0.0006
BRIDGE = {'Bridge_Weight': W, 'Bridge_Premium': PREM, 'Calibration_DT_Years': DT_C}
B0 = -0.98                                               # level basis at t0 (~-6bp of 1700)


def _time_grid(T, coarse=False):
    days = np.arange(T, dtype=np.float64) * (5.0 if coarse else 1.0)
    tg = types.SimpleNamespace()
    tg.scen_time_grid = days
    tg.time_grid_years = days / utils.DAYS_IN_YEAR
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


def _world(shared, T, rows_extra=0, seed=7):
    """Parent price path whose per-step variance IS the published h·f — row-sharp h so a
    one-row mis-index misassigns every row by 2.2x."""
    key = utils.Factor('CommodityPrice', ('LBMA_AM',))
    Z = shared.t_random_numbers[0, :T]
    n = T + rows_extra
    h = 0.012 ** 2 * np.where(np.arange(n) % 2 == 0, 0.5, 2.4)
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    steps = torch.randn((n,) + tuple(Z.shape[1:]), generator=g, dtype=Z.dtype)
    scale = torch.tensor(np.sqrt(h * F_BD), dtype=Z.dtype).view((-1,) + (1,) * (Z.ndim - 1))
    steps = steps * scale
    steps[0] = 0.0
    P = 1700.0 * steps.cumsum(0).exp()
    shared.t_Scenario_Buffer[key] = P
    shared.t_Scenario_Buffer[(key, 'garch_log_h')] = torch.tensor(
        np.log(h), dtype=Z.dtype).view((n, 1) + (1,) * (Z.ndim - 1)).expand(
        (n, 1) + tuple(Z.shape[1:]))
    return key, P, h


def _proc(shared, T, key, coarse=False):
    p = FixingBridgeModel(factor=types.SimpleNamespace(param={}), param=dict(BRIDGE))
    p.factor_key = utils.Factor('ObservedBasis', ('LBMA_AM', 'PM'))
    p.linked_key = key
    p.precalculate(REF_DATE, _time_grid(T, coarse), torch.tensor([B0]), shared, process_ofs=0)
    return p


def _run(T=120, B=8192, rows_extra=0):
    shared = _shared(B, T)
    key, P, h = _world(shared, T, rows_extra)
    p = _proc(shared, T, key)
    return p, shared, P, h, p.generate(shared)


def test_the_law_is_the_bracket_h_scaled():
    p, shared, P, h, out = _run()
    b, Pn = out.cpu().numpy(), P.cpu().numpy()
    assert np.allclose(b[0], B0)
    y = np.log((Pn[1:-1] + b[1:-1]) / Pn[1:-1])
    x = np.log(Pn[2:120] / Pn[1:-1])
    beta, *_ = np.linalg.lstsq(
        np.column_stack([np.ones(y.size), x.ravel()]), y.ravel(), rcond=None)
    assert abs(beta[1] - W) < 0.02, beta
    resid = y - beta[0] - beta[1] * x
    pred = np.sqrt(W * (1 - W) * h[1:119] * F_BD)
    ratio = resid.std(axis=1) / pred
    assert np.abs(ratio - 1.0).max() < 0.12, ratio


def test_the_fixing_reduces_the_next_parents_variance():
    """The named conditional this class exists for: in the SIMULATED world,
    Var(P(t+1) | P(t), fixing(t)) = (1−W)·Var(P(t+1) | P(t)) with unit slope on the intraday
    move — the data's 0.868 sd ratio. The independent-draw law (same marginal) reads slope 0
    and ratio 1.00 here."""
    p, shared, P, h, out = _run()
    b, Pn = out.cpu().numpy(), P.cpu().numpy()
    r_next = np.log(Pn[2:120] / Pn[1:-1]).ravel()
    u = np.log((Pn[1:-1] + b[1:-1]) / Pn[1:-1]).ravel()
    X = np.column_stack([np.ones(u.size), u])
    beta, *_ = np.linalg.lstsq(X, r_next, rcond=None)
    resid = r_next - X @ beta
    assert abs(beta[1] - 1.0) < 0.05, beta                # unit slope - purely mechanical
    ratio = resid.std() / r_next.std()
    assert abs(ratio - np.sqrt(1.0 - W)) < 0.03, ratio    # sqrt(0.753) = 0.868


def test_bracket_availability_decides_the_open_rows():
    p, shared, P, h, out = _run(rows_extra=3)
    b, Pn = out.cpu().numpy(), P.cpu().numpy()
    y_last = np.log((Pn[119] + b[-1]) / Pn[119]) - PREM - W * np.log(Pn[120] / Pn[119])
    assert 0.9 * np.sqrt(W * (1 - W) * h[119] * F_BD) < y_last.std() \
        < 1.1 * np.sqrt(W * (1 - W) * h[119] * F_BD)      # still the true bridge

    p2, shared2, P2, h2, out2 = _run()                    # equal length: open half
    y_open = np.log((P2.cpu().numpy()[119] + out2.cpu().numpy()[-1])
                    / P2.cpu().numpy()[119]) - PREM
    target = np.sqrt(W * h2[119] * F_BD)
    assert 0.9 * target < y_open.std() < 1.1 * target


def test_inner_mode_broadcasts():
    T, B, B2 = 40, 64, 32
    shared = _shared(B, T, seed=3)
    shared.t_random_numbers = torch.randn(1, T, B, B2)
    key, P, h = _world(shared, T)
    p = _proc(shared, T, key)
    p.b0 = torch.full((B,), B0)
    out = p.generate(shared)
    assert out.shape == (T, B, B2)
    assert torch.allclose(out[0], torch.full((B, B2), B0))


def test_a_coarse_grid_refuses_loud():
    shared = _shared(8, 20, seed=1)
    key, P, h = _world(shared, 20)
    with pytest.raises(ValueError, match='coarser'):
        _proc(shared, 20, key, coarse=True)


def test_the_parent_resolves_loud():
    p = FixingBridgeModel(factor=types.SimpleNamespace(param={}), param=dict(BRIDGE))
    fix = utils.Factor('CommodityPrice', ('LBMA_AM',))
    p.calc_references(utils.Factor('ObservedBasis', ('LBMA_AM', 'PM')),
                      None, None, None, {fix: object()})
    assert p.linked_key == fix
    with pytest.raises(Exception, match='exactly one'):
        p.calc_references(utils.Factor('ObservedBasis', ('LBMA_AM', 'PM')),
                          None, None, None, {})


def test_the_calibration_round_trips():
    rng = np.random.default_rng(11)
    n = 3000
    idx = pd.bdate_range('2014-01-01', periods=n)
    S = 1700.0 * np.exp(np.cumsum(rng.standard_normal(n)) * 0.012)
    lpm = np.empty(n)
    lpm[:-1] = np.log(S[:-1]) + W * np.diff(np.log(S)) + PREM \
        + 0.005 * rng.standard_normal(n - 1)
    lpm[-1] = np.log(S[-1]) + PREM
    frame = pd.DataFrame({'ObservedBasis.X.PM': np.exp(lpm) - S, 'CommodityPrice.X': S},
                         index=idx)
    cal = FixingBridgeCalibration('FixingBridgeModel', {})
    res = cal.calibrate(frame, 0.0)
    assert abs(res.param['Bridge_Weight'] - W) < 0.02
    assert abs(res.param['Bridge_Premium'] - PREM) < 0.0005
    assert res.delta.shape[1] == 1 and cal.num_factors == 1
