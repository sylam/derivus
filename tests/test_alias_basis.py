"""`AliasBasisModel` — a basis whose simulated path IS another basis's, under a second
composable name. Pure name composition: no randomness, no law; the declared `Alias_Of` is the
whole contract. Exists so one simulated series can price two positional chains (a session basis
entering both the AM-anchored and PM-anchored composed spots — the PM-session futures marks).

One scenario, several assertions (the density rule): the alias's path is the source's BITWISE
(outer and inner mode), it consumes ZERO randomness (every other factor's draws are unmoved by
its presence — the RNG-ordering invariant), and its refusals are loud: no declaration, an
undeclared source, an ungenerated source, a source grid shorter than its own. Killed by an
alias that draws (its own Z row would shift the ordering), re-simulates (paths diverge), or
resolves positionally (the no-declaration arm)."""
import types

import numpy as np
import pandas as pd
import pytest
import torch

from derivus import utils
from derivus.calculation import CMC_State
from derivus.stochasticprocess import AliasBasisModel

DEVICE = torch.device('cpu')
REF_DATE = pd.Timestamp('2026-04-10')
SRC = utils.Factor('ObservedBasis', ('LBMA_AM', 'CME', 'PM'))
ME = utils.Factor('ObservedBasis', ('LBMA_AM', 'PM', 'CME'))


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


def _shared(B, T, seed=42):
    one = torch.ones(1, 1, dtype=torch.float32, device=DEVICE)
    s = CMC_State(cholesky=torch.eye(1, dtype=torch.float32), static_buffer={}, batch_size=B,
                  one=one, mcmc_sims=0, report_currency=None, seed=seed, job_id=0, num_jobs=1)
    s.reset(num_factors=1, time_grid=_time_grid(T))
    return s


def _alias(shared, T, declared='LBMA_AM.CME.PM'):
    p = AliasBasisModel(factor=types.SimpleNamespace(param={'Alias_Of': declared} if declared
                                                    else {}), param={})
    p.factor_key = ME
    p.calc_references(ME, None, None, None, {SRC: object()})
    p.precalculate(REF_DATE, _time_grid(T), torch.tensor([-7.22]), shared, process_ofs=0)
    return p


def test_the_alias_is_the_source_and_draws_nothing():
    T, B = 60, 512
    shared = _shared(B, T)
    g = torch.Generator(device=DEVICE).manual_seed(7)
    src = -7.0 + torch.randn(T, B, generator=g).cumsum(0)
    shared.t_Scenario_Buffer[SRC] = src
    p = _alias(shared, T)
    assert p.num_factors() == 0 and p.correlation_name[1] == []
    out = p.generate(shared)
    assert torch.equal(out, src)                       # the path IS the source, bitwise
    draws = shared.t_random_numbers.clone()
    p.generate(shared)
    assert torch.equal(shared.t_random_numbers, draws)  # zero randomness consumed

    # inner mode: (T, B, B2) verbatim, same read
    shared.t_Scenario_Buffer[SRC] = src.unsqueeze(-1).expand(T, B, 4).contiguous()
    inner = p.generate(shared)
    assert inner.shape == (T, B, 4) and torch.equal(inner, shared.t_Scenario_Buffer[SRC])

    # a source outliving the grid truncates to the alias's own horizon
    shared.t_Scenario_Buffer[SRC] = torch.cat([src, src[-1:]])
    assert p.generate(shared).shape[0] == T


def test_the_refusals_are_loud():
    T = 8
    shared = _shared(4, T)
    with pytest.raises(Exception, match='no Alias_Of'):
        _alias(shared, T, declared=None)
    bad = AliasBasisModel(factor=types.SimpleNamespace(param={'Alias_Of': 'NOT.THERE'}), param={})
    with pytest.raises(Exception, match='exactly one'):
        bad.calc_references(ME, None, None, None, {SRC: object()})
    p = _alias(shared, T)
    with pytest.raises(Exception, match='has not generated'):
        p.generate(shared)                              # source missing from the buffer
    shared.t_Scenario_Buffer[SRC] = torch.zeros(T - 2, 4)
    with pytest.raises(Exception, match='may not outlive'):
        p.generate(shared)                              # source grid shorter than the alias's
