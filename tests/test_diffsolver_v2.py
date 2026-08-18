"""`DiffSolverV2` — the forward-backward solver: a CONSTRUCTED moving-strike put-delta bank
(breach probability against TODAY'S book measured ONCE on the flat liability, aversion
multiplier, affine range map onto the net position band — fully deterministic) that the
standard backward sweep then fits and improves on.

Gates and their killing mutations (all RUN):

1. PHI IS THE CONDITIONAL AND IT IS ISOTONIC — the bucketed curve tracks the oracle
   P(y|x) and never increases in x, including through a deliberate non-monotone blip that
   raw bucket means would keep. Kills the dropped-cummax mutant.
2. THE DIAL IS THE SLOPE AT THE MONEY — tilt(0.5) == atm EXACTLY (linear slope scale);
   aversion 1.0 is the identity on the measured delta. Kills a scale-ignoring mutant.
3. THE STRIKE RIDES THE BOOK — on a driftless world the curve reads ~0.5 at EVERY wealth,
   including the top decile after a gain: release-at-profit is structurally dead (the old
   static-zero strike read ~0 there). Kills a reverted-static-strike mutant.
4. THE FORWARD PASS IS DETERMINISTIC AND SINGLE-PASS — two calls agree bitwise and the
   policy equals the hand-rolled single recursion (measure once on the flat liability,
   apply at the rolling hedged wealth, range map). Kills a reintroduced-iteration mutant.
5. THE BANK ROLLS THE CONSTRUCTED POLICY — noise 0: q_list[t+1] == q*[t] bitwise and W
   telescopes; the RANGE MAP drives the net to the box ends at the dial's extremes (a
   rep-scaled policy cannot reach the box floor). Kills dropped-range-map and
   re-centered-bank mutants.
6. A FOREIGN CHECKPOINT IS REFUSED BY NAME — a 'DiffSolver' stamp under a DiffSolverV2 run
   raises Solver.Object mismatch (a pre-feature stamp defaults to 'DiffSolver'); a matching
   stamp gets PAST that check. Kills a dropped-refusal mutant.
7. THE NAME IS A CLASS, NOT AN ALIAS.
"""
import types

import pytest
import torch

from derivus.hedge_solver import DiffSolver, DiffSolverV2, HedgeActionSpace

torch.manual_seed(11)


def _runtime(lo=-4.0, hi=0.0):
    hedges = ["A"]
    return {
        "names": {"hedges": hedges},
        "tradables": {r: {"contract_size": 1.0} for r in hedges},
        "portfolio_state": {"positions": {}},
        "solver": {"training_action_grid_levels_per_axis": 5,
                   "training_action_chunk_size": 64, "active_hedge_indices": None},
        "accounting": {
            "position_limits": {r: {"min_position": lo, "max_position": hi} for r in hedges},
            "total_position_abs_limit": 4.0,
            "total_position_schedule": None,
            "max_trade_per_step": 0.0,
            "decision_deadband_sigma": 0.0,
            "force_flat_at_end": False,
            "transaction_cost_per_unit": 0.0,
            "bid_offer_spread_bps": 0.0,
            "bid_offer_spread_spec": None,
        },
    }


def _v2(T_dec=3, B=2048, aversion=1.0, noise=0.0, seed=3, hi=0.0):
    """A DiffSolverV2 stand-in with the REAL forward-pass methods bound. The world: one leg,
    dF ~ N(0,1) per step, liability dL = +2·dF + noise (a LONG book, so the short-only box
    [-4, 0] hedges it at ~-2 per step) and terminal book sign genuinely varies across paths."""
    rt = _runtime(hi=hi)
    aspace = HedgeActionSpace(rt, torch.device("cpu"))
    g = torch.Generator().manual_seed(seed)
    dF = torch.randn(T_dec, B, generator=g)
    L = torch.zeros(T_dec + 1, B)
    for t in range(T_dec):
        L[t + 1] = L[t] + 2.0 * dF[t] + 0.3 * torch.randn(B, generator=g)
    F = torch.cat([torch.zeros(1, B), dF.cumsum(0)])
    s = types.SimpleNamespace(
        aspace=aspace, hedges=["A"], n_hedge=1, B_outer=B, T_dec=T_dec,
        device=torch.device("cpu"), liability_sim=L, tradables_sim={"A": F},
        contract_size=aspace.contract_size, q_lo=aspace.q_lo, q_hi=aspace.q_hi,
        active=aspace.active, n_active=aspace.n_active,
        aversion=aversion, noise_frac=noise, phi_curves=None,
        log_ratio=False, w_floor=1.0,
    )
    for name in ("_phi_curve", "_phi_apply"):
        setattr(s, name, getattr(DiffSolverV2, name))
    for name in ("_tilt", "_constructed_policy", "_build_bank"):
        setattr(s, name, types.MethodType(getattr(DiffSolverV2, name), s))
    s._wealth_step = types.MethodType(DiffSolver._wealth_step, s)
    s._replication_hedge = types.MethodType(DiffSolver._replication_hedge, s)
    return s


def test_phi_is_the_conditional_and_isotonic():
    g = torch.Generator().manual_seed(7)
    x = torch.randn(8192, generator=g) * 2.0
    y = (x + torch.randn(8192, generator=g)) < 0.0          # oracle P = Phi_normal(-x)
    bx, bp = DiffSolverV2._phi_curve(x, y)
    assert (bp[1:] <= bp[:-1] + 1e-6).all(), 'Phi must be decreasing in the book mark'
    import math as _m
    for q, tol in ((-2.0, 0.06), (0.0, 0.06), (2.0, 0.06)):
        oracle = 0.5 * (1.0 + _m.erf(-q / _m.sqrt(2.0)))
        got = float(DiffSolverV2._phi_apply((bx, bp), torch.tensor([q]))[0])
        assert abs(got - oracle) < tol, f'Phi({q}) = {got} vs oracle {oracle}'
    y2 = y.clone()
    y2[(x > 0.5) & (x < 0.7)] = True                        # a planted non-monotone blip
    bx2, bp2 = DiffSolverV2._phi_curve(x, y2)
    assert (bp2[1:] <= bp2[:-1] + 1e-6).all(), 'the blip survived — cummax dropped'
    xf = torch.zeros(512)
    bxf, bpf = DiffSolverV2._phi_curve(xf, y[:512])
    assert torch.allclose(bpf, torch.full_like(bpf, float(y[:512].float().mean())))


def test_the_dial_is_a_multiplier_on_the_measured_delta():
    for gamma in (1.0, 1.2, 1.5, 2.0):
        s = types.SimpleNamespace(aversion=gamma)
        tilt = types.MethodType(DiffSolverV2._tilt, s)
        assert abs(float(tilt(torch.tensor([0.5]))[0]) - min(1.0, 0.5 * gamma)) < 1e-6
        p = torch.linspace(0.0, 1.0, 21)
        out = tilt(p)
        assert (out[1:] >= out[:-1] - 1e-6).all() and float(out[-1]) <= 1.0 + 1e-6
    s1 = types.SimpleNamespace(aversion=1.0)
    p = torch.rand(64)
    assert torch.allclose(types.MethodType(DiffSolverV2._tilt, s1)(p), p, atol=1e-6)
    s15 = types.SimpleNamespace(aversion=1.5)
    assert abs(float(types.MethodType(DiffSolverV2._tilt, s15)(torch.tensor([0.2]))[0])
               - 0.3) < 1e-6                                # linear: g = γ·d


def test_the_strike_rides_the_book():
    """Driftless world: P(end below TODAY'S book | book) ~ 0.5 at every wealth — including
    after a big gain. The old static-zero strike read ~0 at the top decile (the crash
    month's release-at-profit); the moving strike must not."""
    s = _v2(aversion=1.0)
    q, curves, WT = s._constructed_policy()
    book1 = s.liability_sim[1]
    for pt in (book1.quantile(0.1), book1.quantile(0.5), book1.quantile(0.9)):
        d = float(s._phi_apply(curves[1], pt.reshape(1))[0])
        assert 0.32 < d < 0.68, f'moving strike must stay near-ATM everywhere; got {d}'


def test_the_forward_pass_is_deterministic_and_single_pass():
    """The user's ruling: measure once on the flat liability, construct once — the contracts
    held are known exactly, by construction. Two calls agree bitwise (no RNG anywhere in the
    pass), and the policy is exactly the hand-rolled single recursion: curve on the FLAT book,
    aversion-scaled delta applied at the ROLLING hedged wealth, range-mapped, water-filled."""
    s = _v2(seed=5)
    q1, curves1, WT1 = s._constructed_policy()
    q2, curves2, WT2 = s._constructed_policy()
    for a, b in zip(q1, q2):
        assert torch.equal(a, b)
    assert torch.equal(WT1, WT2)
    L = s.liability_sim
    LT = L[s.T_dec]
    W = L[0].clone()
    for t in range(s.T_dec):
        cv = DiffSolverV2._phi_curve(L[t], LT < L[t])
        g = s._tilt(s._phi_apply(cv, W))
        lo_t, hi_t = s.aspace.net_bounds(t)
        net = (hi_t - g * (hi_t - lo_t)).unsqueeze(-1)
        qt = s.aspace.waterfill(
            s._replication_hedge(t)[None].expand(s.B_outer, 1).clone(), net, net)
        qt = torch.minimum(torch.maximum(qt, s.q_lo), s.q_hi)
        assert torch.allclose(q1[t], qt, atol=1e-6), f'hand roll diverges at t={t}'
        dF = (s.tradables_sim["A"][t + 1] - s.tradables_sim["A"][t]).unsqueeze(-1)
        W = s._wealth_step(W, qt, dF, L[t + 1] - L[t])


def test_a_fixed_liability_constructs_a_flat_book():
    """After the last pricing day the replication delta is zero and the below-today comparison
    is float dust — the constructed book must be FLAT there (the box's speculative allowance
    is not a mandate), while the wealth roll still accrues the settling liability rows."""
    s = _v2(T_dec=3, seed=5, hi=1.0)                     # a long allowance: dust would BUY
    L = s.liability_sim
    L[3] = L[2]                                          # final step: liability fully fixed
    s.tradables_sim["A"][3] = s.tradables_sim["A"][2] + 1.0   # marks still move
    q, curves, WT = s._constructed_policy()
    assert torch.equal(q[2], torch.zeros_like(q[2])), 'fixed liability must hold nothing'
    assert torch.allclose(WT, L[3] + sum(
        (q[t] * (s.tradables_sim["A"][t + 1] - s.tradables_sim["A"][t]).unsqueeze(-1)).sum(-1)
        for t in range(3)) - L[0] + L[0], atol=1e-4), 'the settling rows must still accrue'


def test_the_bank_rolls_the_constructed_policy():
    s = _v2(noise=0.0)
    q_star, _, _ = s._constructed_policy()
    gen = torch.Generator().manual_seed(0)
    s.aspace.initial_q = lambda B, dev: torch.zeros(B, 1)
    W_list, q_list = s._build_bank(gen)
    for t in range(1, s.T_dec):
        assert torch.equal(q_list[t], q_star[t - 1]), 'the bank must hold the constructed book'
    L = s.liability_sim
    W = L[0].clone()
    for t in range(1, s.T_dec):
        dF = (s.tradables_sim["A"][t] - s.tradables_sim["A"][t - 1]).unsqueeze(-1)
        W = s._wealth_step(W, q_star[t - 1], dF, L[t] - L[t - 1])
        assert torch.allclose(W_list[t], W, atol=1e-5)


def test_the_range_map_reaches_the_box():
    """The dial's extremes must drive the NET to the box ends — a rep-scaled policy cannot
    reach the box floor (rep ~ -2 on this fixture, the box is [-4, 0])."""
    hi_dial = _v2(aversion=2.0, noise=0.0)         # g = min(1, 2d) ~ 1 at d ~ 0.5
    q_hi, _, _ = hi_dial._constructed_policy()
    assert float(q_hi[1].mean()) < -3.5, 'aversion 2.0 must push the net to the box floor (-4)'
    lo_dial = _v2(aversion=0.1, noise=0.0)         # g ~ 0.05 -> net near the box top
    q_lo, _, _ = lo_dial._constructed_policy()
    assert float(q_lo[1].mean()) > -0.6, 'aversion 0.1 must release toward the box top'


def test_a_foreign_checkpoint_is_refused_by_name():
    v2 = DiffSolverV2.__new__(DiffSolverV2)
    v2.t_min, v2.T_dec, v2.hedges = 0, 3, ["A"]
    v2.position_state = False
    ck = {"t_min": 0, "T_dec": 3, "md": 1, "hedges": ["A"], "solver_object": "DiffSolver"}
    with pytest.raises(ValueError, match="Solver.Object mismatch"):
        DiffSolver._check_load_provenance(v2, ck, "ck.pt", 1)
    ck2 = dict(ck)
    ck2.pop("solver_object")
    with pytest.raises(ValueError, match="Solver.Object mismatch"):
        DiffSolver._check_load_provenance(v2, ck2, "ck.pt", 1)
    ck3 = {**ck, "solver_object": "DiffSolverV2", "position_state": True}
    with pytest.raises(ValueError, match="DiffV2_Position_State mismatch"):
        DiffSolver._check_load_provenance(v2, ck3, "ck.pt", 1)


def test_the_name_is_a_class_not_an_alias():
    assert DiffSolverV2 is not DiffSolver
    assert issubclass(DiffSolverV2, DiffSolver)
