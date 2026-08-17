"""`Max_Trade_Per_Step` — the per-leg |Δq| rate limit at the argmax (execution policy only;
training labels pass no `q_prev` and never see it). The defect it exists for is measured:
the realized 202201 roll flapped between full cover and none on consecutive days (grid-step
multiples of 45 contracts), turning ~50% average cover into a whipsaw that cost ~4x what a
steady half-cover would have. A noisy value surface must produce ramps, not teleports.

Gates: (1) absent/zero = off, bit-identical argmax; (2) the argmax lands on the best action
INSIDE the band around the standing book — killed by the penalty dropped; (3) when nothing is
inside the band (a corridor-forced jump), the ordering falls back to the unrestricted best
rather than garbage — the penalty is a large finite constant, not -inf.
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from derivus.hedge_solver import DiffSolver, HedgeActionSpace


def _aspace(max_trade=0.0, levels=9, lo=-60, hi=0):
    hedges = ["A", "B", "C"]
    runtime = {
        "names": {"hedges": hedges},
        "tradables": {r: {"contract_size": 1.0} for r in hedges},
        "solver": {"training_action_grid_levels_per_axis": levels,
                   "active_hedge_indices": None},
        "accounting": {
            "position_limits": {r: {"min_position": lo, "max_position": hi} for r in hedges},
            "total_position_abs_limit": 0.0,
            "total_position_schedule": None,
            "max_trade_per_step": max_trade,
        },
    }
    return HedgeActionSpace(runtime, torch.device("cpu"))


def _fake_solver(aspace):
    """The smallest object `_decide` runs on: continuation value = total position (favours the
    flat book), so the unrestricted argmax is [0,0,0] and every preference is deterministic."""
    s = types.SimpleNamespace(
        aspace=aspace, chunk=64, risk_kappa=0.0, churn_lambda=0.0, position_state=False,
        # Flat marks: `_calendar_kappa` reads them to price a matched leg (None here — no rate).
        tradables_sim={r: torch.zeros(4, 2) for r in aspace.hedges},
        _wealth_step=lambda W, q, dF, dL: (W + q.sum(-1)).expand(
            W.shape[0], q.shape[1], dF.shape[-1]),
        _continuation=lambda nets, m, W1, t, p: W1,
        _decide=DiffSolver._decide,
    )
    s._unwind_kappa = types.MethodType(DiffSolver._unwind_kappa, s)
    s._calendar_kappa = types.MethodType(DiffSolver._calendar_kappa, s)
    s._reposition_charge = types.MethodType(DiffSolver._reposition_charge, s)
    return s


def _decide(aspace, q_prev):
    s = _fake_solver(aspace)
    B, Bi, md = 2, 3, 1
    market_t1 = torch.zeros(B, Bi, md)
    dF = torch.zeros(B, Bi)
    dL = torch.zeros(B, Bi)
    W = torch.zeros(B)
    q, _ = s._decide(s, None, market_t1, dF, dL, W, 0, q_prev=q_prev, kappa=None, live=None)
    return q


def test_absent_is_off_and_the_unrestricted_argmax_stands():
    a = _aspace()
    assert a.max_trade == 0.0
    q = _decide(a, q_prev=torch.full((2, 3), -45.0))
    assert torch.equal(q, torch.zeros(2, 3))         # flat book is the global best, 45 away


def test_the_argmax_obeys_the_band():
    q = _decide(_aspace(max_trade=7.5), q_prev=torch.full((2, 3), -45.0))
    assert torch.equal(q, torch.full((2, 3), -37.5)), (
        'the best action within one grid step of the standing book is a ramp toward flat')


def test_an_empty_band_falls_back_to_the_unrestricted_ordering():
    """q_prev between grid points with a band too narrow to reach any action: every action is
    penalised by the same constant, the ordering survives, and the argmax is the corridor's
    own best rather than an arbitrary row."""
    q = _decide(_aspace(max_trade=1.0), q_prev=torch.full((2, 3), -3.0))
    assert torch.equal(q, torch.zeros(2, 3))
