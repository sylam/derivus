"""`Decision_Deadband_Sigma` — the statistical no-trade band at the argmax (execution policy
only; the label path passes `rate_limit=False` and never sees it).

The defect it exists for: `_decide` picks by argmax of E_inner[C] over candidate books, and
two candidates whose estimates differ by less than the noise in the estimates swap the
winner from one day to the next, so the realized book teleports (0 <-> 45 contracts) on a
difference that was never there. The band makes the STANDING book the incumbent and moves
only when the winner beats holding by k standard errors of the difference PAIRED across the
common inner draws - pairing is the whole point, since a common per-draw shock (here `dL`)
dominates each candidate's own dispersion while cancelling exactly in the difference.

The fake solver is the real wealth law with identity continuation, so a candidate's per-draw
value is `q*dF + dL`: the mean gap is set by `mean(dF)`, the paired se by `std(dF)`, and the
UNPAIRED se by `std(dL)` - three dials that move independently.

Kill matrix (each mutant applied alone to `DiffSolver._decide`, then reverted):

  mutant                                                   kills
  se from `best_f.std(-1)` (unpaired, not the difference)  test_a_significant_gap_moves
  `move | ~hold_ok` dropped (holding never re-checked)     test_an_infeasible_incumbent_moves
  `dead` drops the `rate_limit and` guard                  test_the_label_path_is_untouched
  `dead` armed at `>= 0.0` (knob 0 no longer off)          test_the_knob_at_zero_is_off
  `dead = False` (band never applied)                      test_an_insignificant_gap_holds
                                                           + test_an_infeasible_incumbent_moves
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from derivus.hedge_solver import DiffSolver, HedgeActionSpace

B, BI = 2, 16
#: Mean-zero, +-1 per-draw pattern - the noise dial every series below is built from.
P = torch.tensor([1.0, -1.0] * (BI // 2))


def _aspace(deadband=0.0, schedule=None, levels=9, lo=-60, hi=0):
    runtime = {
        "names": {"hedges": ["A"]},
        "tradables": {"A": {"contract_size": 1.0}},
        "solver": {"training_action_grid_levels_per_axis": levels,
                   "active_hedge_indices": None},
        "accounting": {
            "position_limits": {"A": {"min_position": lo, "max_position": hi}},
            "total_position_abs_limit": 0.0,
            "total_position_schedule": schedule,
            "max_trade_per_step": 0.0,
            "decision_deadband_sigma": deadband,
        },
    }
    return HedgeActionSpace(runtime, torch.device("cpu"))


def _fake_solver(aspace):
    """The smallest object `_decide` runs on: the real frictionless wealth law at
    contract_size 1 with an identity continuation, so a candidate's per-draw value is
    exactly `q*dF + dL` and every quantity the band computes is analytic."""
    s = types.SimpleNamespace(
        aspace=aspace, chunk=64, risk_kappa=0.0, churn_lambda=0.0, position_state=False,
        _wealth_step=lambda W, q, dF, dL: W + (q * dF).sum(-1) + dL,
        _continuation=lambda nets, m, W1, t, p: W1,
        _decide=DiffSolver._decide,
    )
    s._unwind_kappa = types.MethodType(DiffSolver._unwind_kappa, s)
    s._reposition_charge = types.MethodType(DiffSolver._reposition_charge, s)
    return s


def _decide(aspace, q_prev, dF_noise=1.0, dF_mean=-0.1, dL_scale=100.0, rate_limit=True):
    """One decision on a one-leg [-60, 0] 9-level grid (step 7.5). `dF` carries the drift
    that orders the candidates and the noise the PAIRED se sees; `dL` is the common per-draw
    shock every candidate shares - it cancels in the paired difference and dominates any
    unpaired dispersion."""
    dF = (dF_mean + dF_noise * P).expand(B, BI)[..., None].contiguous()      # (B,Bi,1)
    dL = (dL_scale * P).expand(B, BI).contiguous()                           # (B,Bi)
    s = _fake_solver(aspace)
    return s._decide(s, None, torch.zeros(B, BI, 1), dF, dL, torch.zeros(B), 0,
                     q_prev=q_prev, kappa=None, live=None, rate_limit=rate_limit)


def test_the_knob_at_zero_is_off():
    """Default 0.0 is bit-identically the pre-change argmax, in both returns: the standing
    book cannot change the answer (identical to passing no `q_prev` at all), and a candidate
    that only TIES the incumbent still wins, because at 0 there is no incumbent."""
    a = _aspace()
    assert a.deadband_sigma == 0.0
    q, v = _decide(a, q_prev=torch.full((B, 1), -45.0))
    q0, v0 = _decide(a, q_prev=None)
    assert torch.equal(q, q0) and torch.equal(v, v0)
    assert torch.equal(q, torch.full((B, 1), -60.0))
    # mean(dF)=0: every candidate ties, and the unrestricted argmax takes the first row.
    q, _ = _decide(a, q_prev=torch.full((B, 1), -45.0), dF_mean=0.0)
    assert torch.equal(q, torch.full((B, 1), -60.0))


def test_an_insignificant_gap_holds():
    """The winner beats holding by 1.5 of value on a paired se of 3.87 (t=0.39): inside a
    2-sigma band, so the book stands at -45 instead of teleporting 15 contracts on noise."""
    q, _ = _decide(_aspace(deadband=2.0), q_prev=torch.full((B, 1), -45.0), dF_noise=1.0)
    assert torch.equal(q, torch.full((B, 1), -45.0))


def test_a_significant_gap_moves():
    """The SAME 1.5 of value gap on a paired se of 0.039 (t=39) clears the same 2-sigma band
    and the book moves to -60 - the pair with the test above pins the decision on the paired
    se, not on the mean gap, which is identical in both. The candidates' own per-draw
    dispersion is ~103 here (the common `dL` shock), so an unpaired se rejects this move."""
    q, _ = _decide(_aspace(deadband=2.0), q_prev=torch.full((B, 1), -45.0), dF_noise=0.01)
    assert torch.equal(q, torch.full((B, 1), -60.0))


def test_the_label_path_is_untouched():
    """`rate_limit=False` (the training-label argmax, handed the bank's book for the churn
    charge): the band must not arm, so the insignificant move above is taken anyway."""
    q, _ = _decide(_aspace(deadband=2.0), q_prev=torch.full((B, 1), -45.0), dF_noise=1.0,
                   rate_limit=False)
    assert torch.equal(q, torch.full((B, 1), -60.0))


def test_an_infeasible_incumbent_moves():
    """Holding is only on offer while the standing book is corridor-feasible at t. Under a
    [-60,-50] corridor a book of -45 projects to -50, so holding is not an action and the
    path moves on the insignificant gap; a book of -52.5, inside the same corridor, holds."""
    a = _aspace(deadband=2.0, schedule=[(0, -60.0, -50.0)])
    q, _ = _decide(a, q_prev=torch.full((B, 1), -45.0), dF_noise=1.0)
    assert torch.equal(q, torch.full((B, 1), -60.0)), 'infeasible incumbent must move'
    q, _ = _decide(a, q_prev=torch.full((B, 1), -52.5), dF_noise=1.0)
    assert torch.equal(q, torch.full((B, 1), -52.5)), 'feasible incumbent still holds'
