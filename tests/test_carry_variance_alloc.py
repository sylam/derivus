"""`Allocation_Mode 'Carry_Variance'` — the solver-derived contract split: given the net the
delta/utility logic chose, the triple of futures solves min variance-of-the-hedged-book minus
expected carry, at the log objective's ½/capital exchange rate, subject to the weights summing
to one and staying non-negative.

Gates and their killing mutations (all RUN):

1. TRACKING WINS AT ZERO CARRY — with equal-vol legs and no drift, a liability co-moving
   like the full net takes ~all the weight onto the tracking leg. Kills a dropped-c-term
   mutant.
2. THE CARRY DEMAND IS EXPRESSED AND CAPITAL-SCALED — in quantity space the optimum is
   fixed demands (tracking −Σ⁻¹c, carry C·Σ⁻¹μ) plus min-variance spreading of the rest,
   so a short-favourable drift buys its leg weight, and MORE capital (cheaper variance)
   buys it more. Kills a dropped-mu-term mutant. (The carry/tracking RATIO is Q-invariant
   by design — the marginal carry contract stops paying once its own variance cost catches
   up, so "favour carry as the short grows" saturates at the C·Σ⁻¹μ lump.)
3. A DEAD LEG GETS ZERO — a frozen-mark leg (variance ~0) is excluded by the active set.
4. THE INSTALL IS ONE-SHOT AND MODE-GUARDED — install outside 'carry_variance' refuses;
   install over a standing table refuses. Kills a dropped-guard mutant.
5. THE READER REFUSES THE CONTRADICTIONS — an unknown mode, a declared table beside
   'Carry_Variance', and 'Carry_Variance' without a Solver each fail by name.
"""
import types

import pytest
import torch

from derivus import hedge_runtime
from derivus.hedge_solver import DiffSolverV2, HedgeActionSpace

torch.manual_seed(4)


def _solve(mu, Sig, c, Q=-40.0, C=1.0e5):
    return DiffSolverV2._carry_variance_solve(
        torch.tensor(mu), torch.tensor(Sig), torch.tensor(c), Q, C)


def test_tracking_wins_at_zero_carry():
    # leg 0 covaries with the liability like the FULL net (c ~ |Q|·sigma²), leg 1 is noise
    Sig = [[1.0e6, 0.0], [0.0, 1.0e6]]
    w = _solve([0.0, 0.0], Sig, c=[4.0e7, 0.0], Q=-40.0)
    assert float(w[0]) > 0.95, f'the tracking leg must dominate; got {w.tolist()}'
    assert abs(float(w.sum()) - 1.0) < 1e-6 and float(w.min()) >= 0.0


def test_the_carry_demand_is_expressed_and_capital_scaled():
    # tension: leg 0 tracks better (c 4e7 vs 1e7), leg 1 carries (mu -300 favours the short)
    mu, Sig, c = [0.0, -300.0], [[1.0e6, 0.0], [0.0, 1.0e6]], [4.0e7, 1.0e7]
    w_carry = _solve(mu, Sig, c, Q=-60.0, C=1.0e5)
    w_nocarry = _solve([0.0, 0.0], Sig, c, Q=-60.0, C=1.0e5)
    assert float(w_carry[1]) > float(w_nocarry[1]) + 0.05, \
        'a short-favourable drift must buy its leg weight'
    w_rich = _solve(mu, Sig, c, Q=-60.0, C=1.0e6)
    assert float(w_rich[1]) > float(w_carry[1]) + 0.05, \
        'more capital cheapens variance, so the carry demand grows'


def test_a_dead_leg_gets_zero():
    s = types.SimpleNamespace(
        n_hedge=2, hedges=["A", "B"], T_dec=2,
        contract_size=torch.tensor([1.0, 1.0]),
        liability_sim=None, tradables_sim=None,
    )
    B = 4096
    g = torch.Generator().manual_seed(9)
    dF = torch.randn(2, B, generator=g)
    FA = torch.cat([torch.zeros(1, B), dF.cumsum(0)])
    s.tradables_sim = {"A": FA, "B": torch.zeros(3, B)}          # B is frozen — dead
    L = torch.zeros(3, B)
    L[1] = 2.0 * dF[0]
    L[2] = L[1] + 2.0 * dF[1]
    s.liability_sim = L
    s._replication_hedge = lambda t: torch.tensor([-2.0, 0.0])
    s._capital = lambda t: 10.0
    s._carry_variance_solve = DiffSolverV2._carry_variance_solve
    sched = DiffSolverV2._carry_variance_weights(s)
    assert sched and float(sched[0][1][1]) < 1e-6, f'the dead leg must get 0; got {sched}'
    assert abs(sum(sched[0][1]) - 1.0) < 1e-6


def _runtime(mode):
    return {"names": {"hedges": ["A"]}, "tradables": {"A": {"contract_size": 1.0}},
            "portfolio_state": {"positions": {}},
            "solver": {"training_action_grid_levels_per_axis": 3,
                       "training_action_chunk_size": 8, "active_hedge_indices": None},
            "accounting": {
                "position_limits": {"A": {"min_position": -4.0, "max_position": 0.0}},
                "total_position_abs_limit": 4.0, "total_position_schedule": None,
                "allocation_weights": None, "allocation_mode": mode,
                "max_trade_per_step": 0.0, "decision_deadband_sigma": 0.0,
                "force_flat_at_end": False, "transaction_cost_per_unit": 0.0,
                "bid_offer_spread_bps": 0.0, "bid_offer_spread_spec": None}}


def test_the_install_is_one_shot_and_mode_guarded():
    a = HedgeActionSpace(_runtime("exposure"), torch.device("cpu"))
    with pytest.raises(ValueError, match="outside Allocation_Mode"):
        a.install_weights(((0, (1.0,)),))
    b = HedgeActionSpace(_runtime("carry_variance"), torch.device("cpu"))
    b.install_weights(((0, (1.0,)),))
    assert b.weights == ((0, (1.0,)),)
    with pytest.raises(ValueError, match="already stands"):
        b.install_weights(((0, (1.0,)),))


def _hedge_json(**evaluator):
    return {"Hedging_Problem": {
        "Tradable_Instruments": {"CommodityFutureDeal": {"F1": {"Currency": "USD"}}},
        "Evaluator": dict(evaluator),
        "Objective": {"Object": "AsymmetricUtility_Huber"}}}


def test_the_reader_refuses_the_contradictions():
    with pytest.raises(ValueError, match="Allocation_Mode"):
        hedge_runtime.construct_hedge_runtime(_hedge_json(Allocation_Mode="CarryVariance"))
    with pytest.raises(ValueError, match="contradiction"):
        hedge_runtime.construct_hedge_runtime(_hedge_json(
            Allocation_Mode="Carry_Variance",
            Allocation_Weights=[{"Step": 0, "Instrument": "F1", "Weight": 1.0}]))
    with pytest.raises(ValueError, match="needs a Solver"):
        hedge_runtime.construct_hedge_runtime(_hedge_json(Allocation_Mode="Carry_Variance"))
    rt = hedge_runtime.construct_hedge_runtime(_hedge_json(Allocation_Mode="Exposure"))
    assert rt["accounting"]["allocation_mode"] == "exposure"
    assert hedge_runtime.construct_hedge_runtime(
        _hedge_json())["accounting"]["allocation_mode"] == "exposure"
