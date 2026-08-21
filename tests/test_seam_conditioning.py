"""Session-print conditioning from STATE: the prints a run's initial state carries (the bridge
and chain factors' own values) condition the FIRST simulated step of the factor they inform —
`StochasticProcess.print_seed` publishes the per-path shift, the consumer folds it at its one
shared drift site so forward and replay agree structurally. Present in state ⇒ condition;
absent ⇒ off. No stamped parameter, no grid predicate: the calibrated t0, a burn-in restart
and an inner fork all ride the same protocol on their own state.

Gates and their killing mutations (all RUN):

1. SPOT ROW 1 IS THE CONDITIONAL — with the bridge's print in state, every path is the
   unseeded run times exp(shift) from row 1 on, shift = log((P0+b0)/P0) − premium; row 0
   untouched. EXACT (same seeds, additive log drift), so a mutant that drops the shift, flips
   the premium, or smears it over later rows dies here.
2. ABSENT IS BITWISE OFF — nothing published, identical tensors to the unconditioned law.
   (Carried by gate 1's blind arm being the baseline of an exact ratio.)
3. A FORK CONDITIONS ON ITS OWN DAY'S PRINTS, PER PATH — inner shapes, per-path prints, the
   parent's row 1 sits on each path's own print. Kills a mutant that drops the drift fold or
   the seed maths.
4. BAND ROW 1 RE-ANCHORS AT THE PARTNER — with the chain's premium-adjusted print published,
   out[1] shifts by exactly (partner − premium − b0) against the blind run (same draws);
   row 0 untouched. Kills a mutant that drops the t==1 ds bias or `_chain_shift`.
5. FORWARD == REPLAY UNDER THE SHIFT — reseed_from_path along the conditioned run's own path
   republishes the forward run's slow mean (the bias lives in `_ds_linked`, shared by both).
   Kills a mutant that biases only `generate`.
"""
import types

import numpy as np
import pandas as pd
import torch

from derivus import utils
from derivus.calculation import CMC_State, CMC_State_Inner
from derivus.stochasticprocess import (BasisLinkedSpotModel, ChainedBasisModel,
                                       FixingBridgeModel, GARCHSpotModel)

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
S0, PRINT_B0, PREM = 2000.0, -40.0, -5.0e-4          # print level = S0 + b0 = 1960
B0, PARTNER, PREM_B = -7.35, -29.0, 0.26

SPOT_KEY = utils.Factor('CommodityPrice', ('TEST',))
BRIDGE_KEY = utils.Factor('ObservedBasis', ('TEST', 'PM'))
BAND_KEY = utils.Factor('ObservedBasis', ('LBMA_AM', 'CME'))
CHAIN_KEY = utils.Factor('ObservedBasis', ('LBMA_AM', 'PM', 'CME'))


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


def _bridge(b0=PRINT_B0, prem=PREM):
    br = FixingBridgeModel(factor=types.SimpleNamespace(param={}, name=('TEST', 'PM')),
                          param={'Bridge_Weight': 0.31, 'Bridge_Premium': prem})
    br.linked_key = SPOT_KEY
    return br


def _spot_run(param, T=4, B=4096, seed=42, print_b0=None):
    """Generate the GARCH spot; when `print_b0` is given, publish the bridge's print
    conditioning off a t0-state snapshot first — the way the calc's print pass does."""
    tg = _time_grid(T)
    sh = _shared(B, T, seed=seed, tg=tg)
    p = GARCHSpotModel(factor=types.SimpleNamespace(param={}), param=dict(param))
    p.factor_key = SPOT_KEY
    p.precalculate(REF_DATE, tg, torch.tensor([S0], dtype=DTYPE), sh, process_ofs=0)
    if print_b0 is not None:
        state = {SPOT_KEY: torch.tensor([[[S0]]], dtype=DTYPE),
                 BRIDGE_KEY: torch.tensor([[[print_b0]]], dtype=DTYPE)}
        for k, v in _bridge(print_b0).print_seed(BRIDGE_KEY, state, 0).items():
            sh.t_Scenario_Buffer[k] = v
    torch.manual_seed(7)
    return p.generate(sh)


def _band_run(param, T=4, B=4096, seed=42, replay=None, partner=None, chain_param=None):
    tg = _time_grid(T)
    sh = _shared(B, T, seed=seed, tg=tg)
    p = BasisLinkedSpotModel(factor=types.SimpleNamespace(param={}), param=dict(param))
    p.factor_key = BAND_KEY
    p.linked_key = utils.Factor('CommodityPrice', ('LBMA_AM',))
    p.precalculate(REF_DATE, tg, torch.tensor([B0], dtype=DTYPE), sh, process_ofs=0)
    if partner is not None:
        chain = ChainedBasisModel(factor=types.SimpleNamespace(param={}, name=CHAIN_KEY.name),
                                  param=chain_param or {'Bridge_Premium': PREM_B})
        chain.source_key = BAND_KEY
        state = {CHAIN_KEY: torch.tensor([[[partner]]], dtype=DTYPE),
                 BAND_KEY: torch.tensor([[[B0]]], dtype=DTYPE)}
        for k, v in chain.print_seed(CHAIN_KEY, state, 0).items():
            sh.t_Scenario_Buffer[k] = v
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
    seeded = _spot_run(SPOT, print_b0=PRINT_B0)
    shift = np.log((S0 + PRINT_B0) / S0) - PREM
    ratio = (seeded / base).log()
    assert torch.allclose(ratio[0], torch.zeros_like(ratio[0]), atol=1e-6)
    for t in (1, 2, 3):                        # the shift lands ONCE, at row 1, and stays
        assert torch.allclose(ratio[t], torch.full_like(ratio[t], shift), atol=1e-5), \
            f'row {t}: expected uniform log-shift {shift:.6f}'


def test_a_fork_parent_first_step_conditions_on_the_current_print():
    """Inner shapes, per-path prints: the parent's row 1 sits on each path's own print. The
    fork's state at day t carries that day's prints; `print_seed` conditions on them exactly
    as the calibrated start conditions on its own — one protocol, no grid predicate."""
    T, B, B2 = 2, 4, 4096
    tg = _time_grid(T)
    one = torch.ones(1, 1, dtype=DTYPE, device=DEVICE)
    sh = CMC_State_Inner(cholesky=torch.eye(2, dtype=DTYPE), static_buffer={}, batch_size=B,
                         one=one, mcmc_sims=0, report_currency=None, seed=11, job_id=0,
                         num_jobs=1, simulation_sub_batch=B2)
    sh.reset_inner(num_factors=2, time_grid=tg)
    parent = GARCHSpotModel(factor=types.SimpleNamespace(param={}), param=dict(SPOT))
    parent.factor_key = SPOT_KEY
    P_t = torch.tensor([2000.0, 1900.0, 2100.0, 2000.0], dtype=DTYPE)
    parent.precalculate(REF_DATE, tg, P_t, sh, process_ofs=0)
    b_t = torch.tensor([40.0, -30.0, 0.0, 15.0], dtype=DTYPE)
    state = {SPOT_KEY: P_t.view(1, 1, B), BRIDGE_KEY: b_t.view(1, 1, B)}
    torch.manual_seed(3)
    blind = parent.generate(sh)                                  # nothing published
    bridge = _bridge(prem=0.0)
    for k, v in bridge.print_seed(BRIDGE_KEY, state, 0).items():
        sh.t_Scenario_Buffer[k] = v
    torch.manual_seed(3)
    seeded = parent.generate(sh)
    prints = P_t + b_t
    assert float(((seeded[1].mean(-1) - prints) / prints).abs().max()) < 5e-3, \
        f'seeded fork row 1 must sit on the print: {seeded[1].mean(-1)} vs {prints}'
    assert float(((blind[1].mean(-1) - P_t) / P_t).abs().max()) < 5e-3, \
        'unseeded fork must stay at its own spot'


def test_band_row1_reanchors_at_the_partner():
    base, _ = _band_run(BAND)
    seeded, _ = _band_run(BAND, partner=PARTNER)
    shift = PARTNER - PREM_B - B0
    d = seeded - base
    assert torch.allclose(d[0], torch.zeros_like(d[0]), atol=1e-6)
    assert torch.allclose(d[1], torch.full_like(d[1], shift), atol=1e-4), \
        f'row 1: expected exact re-anchor shift {shift:.4f}'


def test_band_row1_partial_loading_is_the_calibrated_carrythrough():
    """With Next_* stamped, row 1 re-anchors at a·print + k + c·b(0) EXACTLY — the measured
    partial inheritance — and the unit-loading default is its (a=1, c=0, k=−premium)
    degeneration (the gate above). Kills a mutant that drops a loading or the intercept."""
    a, c, k = 0.5, 0.4, 2.0
    base, _ = _band_run(BAND)
    seeded, _ = _band_run(BAND, partner=PARTNER,
                          chain_param={'Next_Print_Loading': a, 'Next_Self_Loading': c,
                                       'Next_Const': k, 'Bridge_Premium': PREM_B})
    shift = (a * PARTNER + k) + (c - 1.0) * B0
    d = seeded - base
    assert torch.allclose(d[0], torch.zeros_like(d[0]), atol=1e-6)
    assert torch.allclose(d[1], torch.full_like(d[1], shift), atol=1e-4), \
        f'row 1: expected partial-loading shift {shift:.4f}'


def test_band_forward_equals_replay_under_the_shift():
    out, mu_fwd = _band_run(BAND, partner=PARTNER)
    mu_rep = _band_run(BAND, replay=out, partner=PARTNER)
    assert torch.allclose(mu_fwd, mu_rep, atol=1e-5), \
        'replayed slow mean must ride the same row-1 bias as the forward sim'


def test_print_conditioning_rides_the_aad_tape():
    """The differential labels' print columns must SEE the conditioning: when the state
    snapshot handed to `print_seed` is built from grad leaves (as `_run_inner_mc_at_t` builds
    it under `with_grad`), d(parent row 1)/d(print basis) must be the bridge law's unit
    loading — elasticity ≈ 1 through log-space. A seed computed off a detached buffer makes
    the bump a graph-free constant and this gradient ~0: the bug this gate pins dead."""
    T, B, B2 = 2, 4, 2048
    tg = _time_grid(T)
    one = torch.ones(1, 1, dtype=DTYPE, device=DEVICE)
    sh = CMC_State_Inner(cholesky=torch.eye(2, dtype=DTYPE), static_buffer={}, batch_size=B,
                         one=one, mcmc_sims=0, report_currency=None, seed=5, job_id=0,
                         num_jobs=1, simulation_sub_batch=B2)
    sh.reset_inner(num_factors=2, time_grid=tg)
    parent = GARCHSpotModel(factor=types.SimpleNamespace(param={}), param=dict(SPOT))
    parent.factor_key = SPOT_KEY
    P_leaf = torch.tensor([2000.0, 1900.0, 2100.0, 2000.0], dtype=DTYPE, requires_grad=True)
    b_leaf = torch.tensor([40.0, -30.0, 0.0, 15.0], dtype=DTYPE, requires_grad=True)
    with torch.enable_grad():
        parent.precalculate(REF_DATE, tg, P_leaf, sh, process_ofs=0)
        state = {SPOT_KEY: P_leaf.view(1, 1, B), BRIDGE_KEY: b_leaf.view(1, 1, B)}
        for k, v in _bridge(prem=0.0).print_seed(BRIDGE_KEY, state, 0).items():
            sh.t_Scenario_Buffer[k] = v
        path = parent.generate(sh)
        path[1].mean().backward()
    g = b_leaf.grad
    assert g is not None and bool((g > 0).all()), f'print leaf grad missing/nonpositive: {g}'
    # elasticity: d S1 / d b = E[S1]/(P+b) per path; loss averaged over B and B2
    elast = g * (P_leaf.detach() + b_leaf.detach()) * B / path[1].mean(-1).detach()
    assert float((elast - 1.0).abs().max()) < 0.1, f'unit-loading elasticity broken: {elast}'
