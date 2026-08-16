"""`ChainedBasisModel` + `ChainedBasisCalibration` — the closed-chain bridge: a basis whose
`Chained_Basis` declarations walk back to their start (the AM/PM session pair is the 2-cycle),
drawn as a bridge between consecutive observations of its declared link's finished path.
Memoryless given that path (the data killed the reversal term), so the class carries no loop,
no extra fork seed and no replay recursion. An open link is `BasisLinkedSpotModel`'s territory
and this class refuses to exist without its declaration.

Gates and their killing mutations:

The fixture triple is the archive's own (3y window): ID 5.06 / ON 5.58 / daily 5.99, which is
ANTI-correlated (ρ(ID,ON) ≈ −0.41 — the AM transient partially reverting overnight). On an
independent triple (s² = σ_ID²+σ_ON²) the exact identities and the ρ=0 recomposition coincide
and a reverted-derivation mutant SURVIVES — the anti-correlation is what makes every gate
below able to see the derivation at all.

1. THE LAW IS THE DECLARED BRIDGE — pooled regression on (P_t, P_{t+1}) recovers (1−w, w) with
   the residual the identities derive; the open last row carries σ_ID alone; row 0 is the
   declared Spot exactly. Killed by a wrong weight derivation, swapped link sigmas, a drawn
   row 0, or the open row reading the bridge σ.
2. THE LINKS CLOSE AT THE DECLARED SOURCE SCALE — the simulated ID and ON link residuals
   reproduce the DECLARED Link_ID/ON sigmas when the source steps at Link_Daily_Sigma. The
   independence derivation under-sizes both ~10% here (sim 4.62/4.99 against 5.06/5.58) and
   dies at the 3% tolerance.
3. THE NEWS CHANNEL IS THE CONSTRUCTION — corr(b−P, the NEXT source move) sits where w and the
   scales put it; a same-row-only law (what an R-only world expresses) scores ≈ 0 here, which
   is this class's reason to exist. Killed by the bracket read dropped to same-row.
4. BRACKET AVAILABILITY DECIDES THE OPEN ROWS — a source outliving the grid bridges every row
   (the last against the row past the grid); an equal-length source leaves the last row open.
   Killed by keying the test on the factor's own horizon.
5. INNER MODE BROADCASTS — (T, B, B2) end to end, with the fork's per-path (B,) row 0.
6. THE SOURCE IS THE DECLARED LINK AND NOTHING ELSE — no positional fallback (a naming
   convention is not a contract); absent declaration and absent-from-universe both raise.
7. THE CALIBRATION ROUND-TRIPS — a panel drawn from the model's own law at the anti-correlated
   triple returns all three sigmas, the derived weight and the premium; and its delta (the
   standardized bridge residual) is uncorrelated with the source's step, so an estimated
   correlation matrix cannot double-count the news channel.

Sim harness copied from tests/test_basis_band_mixture.py (the unit idiom for basis processes).
"""
import types

import numpy as np
import pandas as pd
import pytest
import torch

from derivus import utils
from derivus.calculation import CMC_State
from derivus.stochasticprocess import ChainedBasisCalibration, ChainedBasisModel

DEVICE = torch.device('cpu')
REF_DATE = pd.Timestamp('2026-04-10')
SIG_ID, SIG_ON, SIG_D = 5.06, 5.58, 5.99          # the archive's 3y triple: rho(ID,ON) = -0.41
W_LVL = 0.5 + (SIG_ID ** 2 - SIG_ON ** 2) / (2.0 * SIG_D ** 2)
SIG_BR = np.sqrt(SIG_ID ** 2 - W_LVL ** 2 * SIG_D ** 2)
CHAIN = {'Link_ID_Sigma': SIG_ID, 'Link_ON_Sigma': SIG_ON, 'Link_Daily_Sigma': SIG_D,
         'Bridge_Premium': -0.8}
B0 = -10.65


def _time_grid(T):
    days = np.arange(T, dtype=np.float64)
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


def _proc(param, shared, T, source_key, b0=B0):
    p = ChainedBasisModel(factor=types.SimpleNamespace(param={}), param=dict(param))
    p.factor_key = utils.Factor('ObservedBasis', ('LBMA_AM', 'PM', 'CME'))
    p.source_key = source_key
    p.precalculate(REF_DATE, _time_grid(T), torch.tensor([b0]), shared, process_ofs=0)
    return p


def _run(T=120, B=8192, rows_extra=0, seed=42):
    shared = _shared(B, T, seed=seed)
    key = utils.Factor('ObservedBasis', ('LBMA_AM', 'CME'))
    Z = shared.t_random_numbers[0, :T]
    g = torch.Generator(device=DEVICE).manual_seed(7)
    # the source walks at the DECLARED daily scale, so the closure gate can hold the sim
    # link residuals to the declared targets
    steps = torch.randn((T + rows_extra - 1,) + tuple(Z.shape[1:]), generator=g,
                        dtype=Z.dtype) * SIG_D
    P = -7.0 + torch.cat([torch.zeros((1,) + tuple(Z.shape[1:]), dtype=Z.dtype),
                          steps.cumsum(0)])
    shared.t_Scenario_Buffer[key] = P
    p = _proc(CHAIN, shared, T, key)
    return p, shared, P, p.generate(shared)


def test_the_law_is_the_declared_bridge():
    p, shared, P, out = _run()
    b, Pn = out.cpu().numpy(), P.cpu().numpy()
    assert np.allclose(b[0], B0)                          # row 0 = the Spot, never drawn
    y = (b[1:-1] - (-0.8)).ravel()
    X = np.column_stack([Pn[1:-1].ravel(), Pn[2:120].ravel()])
    beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(y.size), X]), y, rcond=None)
    assert abs(beta[1] - (1.0 - W_LVL)) < 0.02 and abs(beta[2] - W_LVL) < 0.02, beta
    resid = y - beta[0] - X @ beta[1:]
    assert 0.95 * SIG_BR < resid.std() < 1.05 * SIG_BR
    open_sd = (b[-1] - Pn[119] - (-0.8)).std()
    assert 0.95 * SIG_ID < open_sd < 1.05 * SIG_ID        # the open row is the ID half alone


def test_the_links_close_at_the_declared_source_scale():
    """Sim ID/ON link residuals reproduce the DECLARED sigmas when the source steps at
    Link_Daily_Sigma — the whole point of the exact identities. Mutation: the independence
    derivation (w = ID²/(ID²+ON²), σ = ID·ON/√(ID²+ON²)) puts them at 4.62/4.99 against
    5.06/5.58 declared, −9%/−11%, killed at the 3% tolerance."""
    p, shared, P, out = _run()
    b, Pn = out.cpu().numpy(), P.cpu().numpy()
    id_link = (b[1:-1] - Pn[1:-1]).ravel().std()
    on_link = (Pn[2:120] - b[1:-1]).ravel().std()
    assert abs(id_link - SIG_ID) / SIG_ID < 0.03, id_link
    assert abs(on_link - SIG_ON) / SIG_ON < 0.03, on_link


def test_the_news_channel_is_the_construction():
    p, shared, P, out = _run()
    b, Pn = out.cpu().numpy(), P.cpu().numpy()
    dev = (b[1:-1] - Pn[1:-1]).ravel()
    nxt = (Pn[2:120] - Pn[1:-1]).ravel()
    rho = np.corrcoef(dev, nxt)[0, 1]
    s = nxt.std()
    target = W_LVL * s / np.sqrt(W_LVL ** 2 * s ** 2 + SIG_BR ** 2)
    assert abs(rho - target) < 0.05, (rho, target)
    assert rho > 0.4                                      # the channel R-only worlds score 0 on


def test_bracket_availability_decides_the_open_rows():
    p, shared, P, out = _run(rows_extra=3)                # source outlives the grid
    b, Pn = out.cpu().numpy(), P.cpu().numpy()
    last = (b[-1] - Pn[119] - (-0.8) - W_LVL * (Pn[120] - Pn[119])).std()
    assert 0.95 * SIG_BR < last < 1.05 * SIG_BR           # the true bridge, not the open half


def test_inner_mode_broadcasts():
    T, B, B2 = 40, 64, 32
    shared = _shared(B, T, seed=3)
    shared.t_random_numbers = torch.randn(1, T, B, B2)
    key = utils.Factor('ObservedBasis', ('LBMA_AM', 'CME'))
    g = torch.Generator(device=DEVICE).manual_seed(5)
    shared.t_Scenario_Buffer[key] = -7.0 + torch.randn(T, B, B2, generator=g) * 6.0
    p = _proc(CHAIN, shared, T, key)
    p.b0 = torch.full((B,), B0)                           # the fork's per-path (B,) contract
    out = p.generate(shared)
    assert out.shape == (T, B, B2)
    assert torch.allclose(out[0], torch.full((B, B2), B0))


def test_the_source_is_the_declared_link_and_nothing_else():
    me = utils.Factor('ObservedBasis', ('LBMA_AM', 'CME', 'PM'))
    cme = utils.Factor('ObservedBasis', ('LBMA_AM', 'CME'))

    p = ChainedBasisModel(
        factor=types.SimpleNamespace(param={'Chained_Basis': 'LBMA_AM.CME'}), param=dict(CHAIN))
    p.calc_references(me, None, None, None, {cme: object()})
    assert p.source_key == cme

    with pytest.raises(Exception, match='exactly one'):
        p.calc_references(me, None, None, None, {})       # declared but absent from the universe
    bare = ChainedBasisModel(factor=types.SimpleNamespace(param={}), param=dict(CHAIN))
    with pytest.raises(Exception, match='no Chained_Basis'):
        bare.calc_references(me, None, None, None, {cme: object()})   # NO positional fallback
    lagged = ChainedBasisModel(factor=types.SimpleNamespace(
        param={'Chained_Basis': 'LBMA_AM.CME', 'Chained_Lag': 1}), param=dict(CHAIN))
    with pytest.raises(Exception, match='SAME row'):
        lagged.calc_references(me, None, None, None, {cme: object()})  # a bridge never lags

    # an ungenerated source fails loud at generate, naming both
    shared = _shared(4, 8, seed=1)
    q = _proc(CHAIN, shared, 8, cme)
    with pytest.raises(Exception, match='has not generated'):
        q.generate(shared)


def test_the_chain_refuses_the_decay_projection():
    """One model per payoff: this basis re-anchors to its declared link's own path (E[b]
    tracks the source basis plus the premium), which the own-level (phi, lam) decay cannot
    state — so a deal pricing the PM-session composed name refuses loud instead of pricing
    under another law's projection. The PM-session marks increment defines it, if ever."""
    from derivus.instruments import get_observed_basis_decay
    p = ChainedBasisModel(factor=types.SimpleNamespace(param={}), param=dict(CHAIN))
    key = utils.Factor('ObservedBasis', ('LBMA_AM', 'PM', 'CME'))
    code = [(True, utils.Factor('ObservedBasis', ('LBMA_AM', 'CME')), None), (True, key, None)]
    with pytest.raises(Exception, match='basis-decay'):
        get_observed_basis_decay(code, {key: p})


def test_the_calibration_round_trips():
    # a panel drawn from the model's own law at the anti-correlated triple: the source walks
    # at SIG_D, and Var(ID) = w²s² + σ² = σ_ID², Var(ON) = (1−w)²s² + σ² = σ_ON² close by the
    # same identities the model derives with — no constraint between s and the link sigmas
    rng = np.random.default_rng(11)
    n = 3000
    idx = pd.bdate_range('2014-01-01', periods=n)
    P = -7.0 + np.cumsum(rng.standard_normal(n)) * SIG_D
    b = np.empty(n)
    b[:-1] = P[:-1] + W_LVL * (P[1:] - P[:-1]) - 0.8 + SIG_BR * rng.standard_normal(n - 1)
    b[-1] = P[-1] - 0.8 + SIG_ID * rng.standard_normal()
    frame = pd.DataFrame({'ObservedBasis.X.PM.CME': b, 'ObservedBasis.X.CME': P}, index=idx)
    cal = ChainedBasisCalibration('ChainedBasisModel', {})
    res = cal.calibrate(frame, 0.0)
    sid, son, sd = (res.param['Link_ID_Sigma'], res.param['Link_ON_Sigma'],
                    res.param['Link_Daily_Sigma'])
    assert abs(sid - SIG_ID) < 0.25
    assert abs(son - SIG_ON) < 0.25
    assert abs(sd - SIG_D) < 0.25
    assert abs(0.5 + (sid ** 2 - son ** 2) / (2 * sd ** 2) - W_LVL) < 0.03
    assert abs(res.param['Bridge_Premium'] - (-0.8)) < 0.3
    assert res.delta.shape[1] == 1 and cal.num_factors == 1
    # the delta is the BRIDGE residual: uncorrelated with the source's step, so a correlation
    # estimate off it cannot double-count the news channel (the ID-link residual would)
    dP = pd.Series(np.diff(P), index=idx[:-1]).reindex(res.delta.index)
    assert abs(np.corrcoef(res.delta.values[:, 0], dP.values)[0, 1]) < 0.05
