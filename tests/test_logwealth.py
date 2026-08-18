"""`Objective 'LogWealth'` — the per-step growth objective: reward = log(W1/W0), scale-free,
step-sum by construction. Gates and their killing mutations (all RUN):

1. THE REWARD IS THE GROWTH RATIO — a perfectly hedged step scores exactly 0; log(e·W0/W0) = 1;
   the Running_Wealth arm is untouched (dispatch, not replacement). Kills a diff-for-ratio mutant.
2. THE FLOOR IS THE DOMAIN — the log continues LINEARLY below eps (continuous at the joint,
   nonzero gradient below it, so a breached path still learns), and a step LAUNCHED at or under
   the floor can only score <= 0 (ruin is absorbing — no rebirth against a clamped denominator).
   Kills a dropped-extension mutant and a dropped-launch-cap mutant.
3. THE READER COUPLES THE DIALS — 'LogWealth' + Reference_Mode='Fixed' refuses;
   'LogWealth' alone lands reference_mode='running_wealth'; conditional_sim beside it is
   ACCEPTED (the schedule is the capital line). Kills a dropped-coupling mutant (the silent
   result would be a terminal-utility run wearing a growth objective's name).
4. THE TERMINAL WRAP REFUSES BY NAME — any caller asking LogWealth for a terminal utility
   (benchmark tracks, terminal verdicts) gets a named error, not dollars.
"""
import math
import types

import pytest
import torch

from derivus import hedge_runtime
from derivus.hedge_bundle import _utility_wrap_signed
from derivus.hedge_solver import DiffSolver


def _s(log_ratio=True, w_floor=1.0, capital=0.0):
    s = types.SimpleNamespace(log_ratio=log_ratio, w_floor=w_floor,
                              utility_scale_schedule=None, utility_scale=capital,
                              _u=lambda x, t=None: x * 2.0)     # a marked stand-in for the wrap
    s._u_step = types.MethodType(DiffSolver._u_step, s)
    s._capital = types.MethodType(DiffSolver._capital, s)
    return s


def test_the_reward_is_the_growth_ratio():
    s = _s()
    W0 = torch.tensor([100.0, 250.0, 3.0])
    assert torch.allclose(s._u_step(W0, W0, 1), torch.zeros(3), atol=1e-7)
    assert torch.allclose(s._u_step(W0 * math.e, W0, 1), torch.ones(3), atol=1e-6)
    assert torch.allclose(s._u_step(W0 * 0.5, W0, 1), torch.full((3,), math.log(0.5)), atol=1e-6)
    # the CAPITAL line: wealth in the ratio is capital + MTM — an MTM move of +capital from a
    # flat book is a doubling; and the schedule entry wins over the scalar when locked
    sc = _s(capital=100.0)
    assert abs(float(sc._u_step(torch.tensor([100.0]), torch.tensor([0.0]), 1))
               - math.log(2.0)) < 1e-6
    sc.utility_scale_schedule = [40.0, 60.0]
    assert abs(float(sc._u_step(torch.tensor([60.0]), torch.tensor([0.0]), 1))
               - math.log(2.0)) < 1e-6                       # c_1 = 60
    # the Running_Wealth arm dispatches to the wrap untouched
    off = _s(log_ratio=False)
    assert torch.allclose(off._u_step(torch.tensor([5.0]), torch.tensor([2.0]), 1),
                          torch.tensor([6.0]))                   # _u(5-2) = 2*(3)


def test_the_floor_is_the_domain():
    s = _s(w_floor=1.0)
    W0 = torch.tensor([100.0])
    eps = 1.0e-3
    # continuity at the joint and a LIVE gradient below it
    lo = s._u_step(W0 * eps * 1.0001, W0, 1)
    hi = s._u_step(W0 * eps * 0.9999, W0, 1)
    assert abs(float(lo - hi)) < 1e-3
    r = torch.tensor([0.0002], requires_grad=True)
    u = s._u_step(r * 100.0, W0, 1)
    u.backward()
    assert float(r.grad.abs()) > 0.0, 'a breached path must keep gradient'
    # a deeply negative step is a large penalty, monotone in depth
    assert float(s._u_step(-W0, W0, 1)) < float(s._u_step(W0 * eps, W0, 1))
    # ruined launch: W0 under the floor can only score <= 0, however large W1
    assert float(s._u_step(torch.tensor([1.0e6]), torch.tensor([0.5]), 1)) <= 0.0
    assert float(s._u_step(torch.tensor([1.0e6]), torch.tensor([-50.0]), 1)) <= 0.0


def _hedge_json(**objective):
    return {"Hedging_Problem": {
        "Tradable_Instruments": {"CommodityFutureDeal": {"F1": {"Currency": "USD"}}},
        "Evaluator": {},
        "Objective": dict({"Object": "LogWealth"}, **objective)}}


def test_the_reader_couples_the_dials():
    with pytest.raises(ValueError, match="LogWealth.*Reference_Mode|Reference_Mode.*LogWealth"):
        hedge_runtime.construct_hedge_runtime(_hedge_json(Reference_Mode="Fixed"))
    rt = hedge_runtime.construct_hedge_runtime(_hedge_json())
    assert rt["objective"]["reference_mode"] == "running_wealth"
    assert rt["objective"]["object"] == "logwealth"
    rt2 = hedge_runtime.construct_hedge_runtime(_hedge_json(Reference_Mode="Running_Wealth"))
    assert rt2["objective"]["reference_mode"] == "running_wealth"
    # conditional_sim beside LogWealth is the DESIGNED pair: the schedule is the capital line
    rt3 = hedge_runtime.construct_hedge_runtime(_hedge_json(Utility_Scale_Mode="conditional_sim"))
    assert rt3["objective"]["utility_scale_mode"] == "conditional_sim"


def test_the_terminal_wrap_refuses_by_name():
    rt = hedge_runtime.construct_hedge_runtime(_hedge_json())
    with pytest.raises(ValueError, match="no terminal utility"):
        _utility_wrap_signed(torch.tensor([1.0]), rt)
