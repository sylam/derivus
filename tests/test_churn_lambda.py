"""`DiffV2_Churn_Lambda` — the quadratic repositioning charge λ·Σ(q−q_prev)² subtracted from
the wealth entering the continuation, at every argmax that knows the standing book: the
verdict/stepper argmax AND the training-label argmax (threaded the bank's own book, with the
rate-limit band explicitly NOT armed there — the band stays execution-only).

The charge is quadratic on purpose: a 45-contract flip costs 36x a 7.5-contract step, which
is the anti-bang-bang shape, while small adjustments stay nearly free — and unlike a hard cap
it cannot freeze the book (a big move stays available when the value difference earns it).

Gates: (1) λ=0 is off — bit-identical argmax; (2) the charge moves a near-tie to the nearer
action but lets a large value gap pay for a large move (killed by the charge dropped or made
linear); (3) `rate_limit=False` disables the band while keeping the charge (killed by the
label call arming the band).
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


def _fake_solver(aspace, churn_lambda=0.0, value_scale=1.0):
    """Continuation value = value_scale * total position (favours the flat book), so the
    unrestricted argmax is [0,0,0] and the charge's geometry is exactly computable."""
    return types.SimpleNamespace(
        aspace=aspace, chunk=64, risk_kappa=0.0, churn_lambda=churn_lambda,
        _wealth_step=lambda W, q, dF, dL: (W + q.sum(-1) * value_scale).expand(
            W.shape[0], q.shape[1], dF.shape[-1]),
        _continuation=lambda nets, m, W1, t: W1,
        _decide=DiffSolver._decide,
    )


def _decide(aspace, q_prev, churn_lambda=0.0, value_scale=1.0, rate_limit=True):
    s = _fake_solver(aspace, churn_lambda, value_scale)
    B, Bi, md = 2, 3, 1
    q, _ = s._decide(s, None, torch.zeros(B, Bi, md), torch.zeros(B, Bi), torch.zeros(B, Bi),
                     torch.zeros(B), 0, q_prev=q_prev, kappa=None, live=None,
                     rate_limit=rate_limit)
    return q


def test_zero_lambda_is_off():
    q = _decide(_aspace(), q_prev=torch.full((2, 3), -45.0), churn_lambda=0.0)
    assert torch.equal(q, torch.zeros(2, 3))          # global best, 45 away, uncharged


def test_the_charge_prices_the_move():
    """From a standing book of -45 per leg on the [-60,0] 9-level grid (step 7.5), value
    rises 1.0 per contract of total position. One 7.5-step toward flat gains 22.5 of value
    for 3·λ·7.5² of charge; the full 45-contract flip gains 135 for 3·λ·45². At λ=0.1 the
    flip costs 607.5 > 135 while one step costs 16.9 < 22.5 — the argmax walks. At λ=0.001
    every charge is dwarfed and it jumps to flat. Killed by the charge dropped (always
    jumps) or made linear (λ=0.1 linear charge 13.5 < gain — would still jump)."""
    prev = torch.full((2, 3), -45.0)
    q = _decide(_aspace(), prev, churn_lambda=0.1)
    assert torch.equal(q, torch.full((2, 3), -37.5)), 'expensive flip: walk one step'
    q = _decide(_aspace(), prev, churn_lambda=0.001)
    assert torch.equal(q, torch.zeros(2, 3)), 'cheap flip: jump to the optimum'


def test_label_path_charges_without_the_band():
    """rate_limit=False (the label call): the max-trade band must NOT bind, the charge must.
    With a band of 7.5 armed the far action is unreachable regardless of λ; unarmed, tiny λ
    jumps and large λ walks — the discrimination the label semantics need."""
    prev = torch.full((2, 3), -45.0)
    a = _aspace(max_trade=7.5)
    q = _decide(a, prev, churn_lambda=0.001, rate_limit=False)
    assert torch.equal(q, torch.zeros(2, 3)), 'band must not reach the label path'
    q = _decide(a, prev, churn_lambda=0.1, rate_limit=False)
    assert torch.equal(q, torch.full((2, 3), -37.5)), 'the charge alone shapes the label'
