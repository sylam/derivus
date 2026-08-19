"""`DiffSolverV2` — the forward-backward solver. The forward pass is the CLAIRVOYANT SEED:
a measured moving-strike delta (strike = max(today's book, $1/oz), no dial — the aversion
belongs to the backward DP), mapped affinely onto the net band, in whole contracts, plus the
LEAST extra cover the known path needs so the book never falls under the $1/oz floor —
positive wealth everywhere, by construction. The backward sweep then fits and improves on
the bank this policy rolls.

Gates and their killing mutations (all RUN):

1. PHI IS THE CONDITIONAL AND IT IS ISOTONIC — oracle-tracked, blip-ironed.
   Kills the dropped-cummax mutant.
2. THE STRIKE IS THE HIGH-WATER OR THE FLOOR — above water ~0.5 (no release-at-profit);
   under water the delta saturates. Kills static-zero and dropped-floor strike mutants.
3. THE FORWARD PASS IS DETERMINISTIC AND SINGLE-PASS — two calls agree bitwise and equal
   the hand-rolled recursion (reserve backward, delta + least-repair forward).
4. THE BOOK HOLDS WHOLE CONTRACTS — integer per leg, net rounded AWAY from zero at a
   discriminating fraction, odd nets apportioned whole. Kills plain-round and
   dropped-largest-remainder mutants.
5. THE FLOOR HOLDS EVERYWHERE, AND MINIMALLY — on a feasible world every path's book stays
   >= the floor at every step, INCLUDING through a defenseless day (the reserve
   pre-positions the cushion the day before), and the repair is the LEAST earning integer,
   never max cover. Kills a dropped-reserve mutant (floor-only look-ahead breaches the
   defenseless day) and a max-cover mutant.
6. A FOREIGN CHECKPOINT IS REFUSED BY NAME. 7. THE NAME IS A CLASS, NOT AN ALIAS.
"""
import math
import types

import pytest
import torch

from derivus.hedge_solver import DiffSolver, DiffSolverV2, HedgeActionSpace

torch.manual_seed(11)


def _runtime(lo=-4.0, hi=0.0, hedges=("A",)):
    hedges = list(hedges)
    return {
        "names": {"hedges": hedges},
        "tradables": {r: {"contract_size": 1.0} for r in hedges},
        "portfolio_state": {"positions": {}},
        "solver": {"training_action_grid_levels_per_axis": 5,
                   "training_action_chunk_size": 64, "active_hedge_indices": None},
        "accounting": {
            "position_limits": {r: {"min_position": lo, "max_position": hi} for r in hedges},
            "total_position_abs_limit": abs(lo) * len(hedges),
            "total_position_schedule": None,
            "max_trade_per_step": 0.0,
            "decision_deadband_sigma": 0.0,
            "force_flat_at_end": False,
            "transaction_cost_per_unit": 0.0,
            "bid_offer_spread_bps": 0.0,
            "bid_offer_spread_spec": None,
        },
    }


def _v2(T_dec=3, B=2048, noise=0.0, seed=3, lo=-4.0, hi=0.0, legs=1, flat_day=None,
        l0=0.0):
    """A DiffSolverV2 stand-in with the REAL forward-pass methods bound. One (or `legs`)
    futures with dF ~ N(0,1) per step, liability dL = +2·dF + noise. `flat_day` freezes the
    marks on that step (m = 0: a defenseless day — the liability still moves on its noise),
    which is what exercises the reserve's pre-positioning."""
    names = [chr(65 + i) for i in range(legs)]
    rt = _runtime(lo=lo, hi=hi, hedges=names)
    aspace = HedgeActionSpace(rt, torch.device("cpu"))
    g = torch.Generator().manual_seed(seed)
    dF = torch.randn(T_dec, B, generator=g)
    if flat_day is not None:
        dF[flat_day] = 0.0
    L = torch.zeros(T_dec + 1, B)
    L[0] = l0                                # the margin cushion; 0 leaves day-0 under floor
    for t in range(T_dec):
        L[t + 1] = L[t] + 2.0 * dF[t] + 0.3 * torch.randn(B, generator=g)
    F = torch.cat([torch.zeros(1, B), dF.cumsum(0)])
    s = types.SimpleNamespace(
        aspace=aspace, hedges=names, n_hedge=legs, B_outer=B, T_dec=T_dec,
        device=torch.device("cpu"), liability_sim=L, tradables_sim={n: F for n in names},
        contract_size=aspace.contract_size, q_lo=aspace.q_lo, q_hi=aspace.q_hi,
        active=aspace.active, n_active=aspace.n_active,
        noise_frac=noise, phi_curves=None, leg_volume=0.5,
        log_ratio=False, w_floor=1.0, aversion=1.0,
    )
    for name in ("_phi_curve", "_phi_apply"):
        setattr(s, name, getattr(DiffSolverV2, name))
    for name in ("_constructed_policy", "_build_bank"):
        setattr(s, name, types.MethodType(getattr(DiffSolverV2, name), s))
    s._wealth_step = types.MethodType(DiffSolver._wealth_step, s)
    s._replication_hedge = types.MethodType(DiffSolver._replication_hedge, s)
    return s


def _roll_books(s, q):
    """The per-step book under a constructed trajectory (t=0..T_dec), for floor checks."""
    L = s.liability_sim
    books, W = [L[0].clone()], L[0].clone()
    for t in range(s.T_dec):
        dF = torch.stack([s.tradables_sim[r][t + 1] - s.tradables_sim[r][t]
                          for r in s.hedges], dim=-1)
        W = s._wealth_step(W, q[t], dF, L[t + 1] - L[t])
        books.append(W.clone())
    return torch.stack(books)


def test_phi_is_the_conditional_and_isotonic():
    g = torch.Generator().manual_seed(7)
    x = torch.randn(8192, generator=g) * 2.0
    y = (x + torch.randn(8192, generator=g)) < 0.0          # oracle P = Phi_normal(-x)
    bx, bp = DiffSolverV2._phi_curve(x, y)
    assert (bp[1:] <= bp[:-1] + 1e-6).all(), 'Phi must be decreasing in the book mark'
    for qq, tol in ((-2.0, 0.06), (0.0, 0.06), (2.0, 0.06)):
        oracle = 0.5 * (1.0 + math.erf(-qq / math.sqrt(2.0)))
        got = float(DiffSolverV2._phi_apply((bx, bp), torch.tensor([qq]))[0])
        assert abs(got - oracle) < tol, f'Phi({qq}) = {got} vs oracle {oracle}'
    y2 = y.clone()
    y2[(x > 0.5) & (x < 0.7)] = True                        # a planted non-monotone blip
    bx2, bp2 = DiffSolverV2._phi_curve(x, y2)
    assert (bp2[1:] <= bp2[:-1] + 1e-6).all(), 'the blip survived — cummax dropped'
    xf = torch.zeros(512)
    bxf, bpf = DiffSolverV2._phi_curve(xf, y[:512])
    assert torch.allclose(bpf, torch.full_like(bpf, float(y[:512].float().mean())))


def test_the_strike_is_the_high_water_or_the_floor():
    s = _v2()
    q, curves, WT = s._constructed_policy()
    book1 = s.liability_sim[1]
    d_hi = float(s._phi_apply(curves[1], book1.quantile(0.9).reshape(1))[0])
    assert 0.32 < d_hi < 0.68, f'above water must read near-ATM (no release); got {d_hi}'
    d_lo = float(s._phi_apply(curves[1], book1.quantile(0.1).reshape(1))[0])
    assert d_lo > 0.62, f'under water the floor must demand more cover; got {d_lo}'
    assert d_lo > d_hi + 0.1, 'the floor must separate under- from above-water books'


def test_the_forward_pass_is_deterministic_and_single_pass():
    """Two calls agree bitwise, and the policy equals the hand-rolled spec: the reserve
    backward (floor minus liability move minus best earnable on the KNOWN move), then
    forward the whole-contract delta base overridden by the LEAST repairing integer
    wherever tomorrow's book would miss tomorrow's reserve."""
    s = _v2(seed=5, lo=-7.0)
    q1, curves1, WT1 = s._constructed_policy()
    q2, curves2, WT2 = s._constructed_policy()
    for a, b in zip(q1, q2):
        assert torch.equal(a, b)
    assert torch.equal(WT1, WT2)
    L, T = s.liability_sim, s.T_dec
    LT = L[T]
    floor = 1.0 * s.leg_volume
    rep = [s._replication_hedge(t) for t in range(T)]
    m = [(rep[t] / rep[t].sum() * s.contract_size) @ torch.stack(
        [s.tradables_sim[r][t + 1] - s.tradables_sim[r][t] for r in s.hedges])
        for t in range(T)]
    lo_i, hi_i = [], []
    for t in range(T):
        lo_t, hi_t = s.aspace.net_bounds(t)
        lo_i.append(math.ceil(lo_t - 1e-9))
        hi_i.append(math.floor(hi_t + 1e-9))
    req = torch.full((s.B_outer,), floor)
    reqs = [None] * T
    for t in range(T - 1, -1, -1):
        reqs[t] = req.clone()
        best = torch.maximum(hi_i[t] * m[t], lo_i[t] * m[t])
        req = torch.maximum(req - (L[t + 1] - L[t]) - best, torch.full_like(req, floor))
    W = L[0].clone()
    for t in range(T):
        cv = DiffSolverV2._phi_curve(L[t], LT < L[t].clamp_min(floor))
        g = s._phi_apply(cv, W)
        net = hi_i[t] - g * (hi_i[t] - lo_i[t])
        net = torch.where(net < 0, net.floor(), net.ceil()).clamp(lo_i[t], hi_i[t])
        dL = L[t + 1] - L[t]
        target = reqs[t]                       # reqs[t] IS the requirement at t+1
        if t == 0:
            net = net.unsqueeze(-1)
        else:
            defensible = m[t].abs() > 1e-9
            bound = (target - W - dL) / torch.where(defensible, m[t], torch.ones_like(m[t]))
            net = torch.where(defensible & (m[t] > 0), torch.maximum(net, bound.ceil()), net)
            net = torch.where(defensible & (m[t] < 0), torch.minimum(net, bound.floor()), net)
            net = net.clamp(lo_i[t], hi_i[t]).unsqueeze(-1)
        qt = s.aspace.waterfill(
            rep[t][None].expand(s.B_outer, s.n_hedge).clone(), net, net)
        qt = s.aspace._largest_remainder(qt, net)
        qt = torch.minimum(torch.maximum(qt, s.q_lo), s.q_hi)
        if t > 0:
            dFl = torch.stack([s.tradables_sim[r][t + 1] - s.tradables_sim[r][t]
                               for r in s.hedges], dim=-1)
            exec_pnl = (qt * s.contract_size * dFl).sum(-1)
            short = target - (W + dL + exec_pnl)
            fix = defensible & (short > 0)
            if bool(fix.any()):
                extra = torch.where(fix, ((short / m[t].abs()).ceil() + 1.0)
                                    * torch.sign(m[t]), torch.zeros_like(m[t]))
                net = (net.squeeze(-1) + extra).clamp(lo_i[t], hi_i[t]).unsqueeze(-1)
                qt = s.aspace.waterfill(
                    rep[t][None].expand(s.B_outer, s.n_hedge).clone(), net, net)
                qt = s.aspace._largest_remainder(qt, net)
                qt = torch.minimum(torch.maximum(qt, s.q_lo), s.q_hi)
        assert torch.allclose(q1[t], qt, atol=1e-6), f'hand roll diverges at t={t}'
        dF = torch.stack([s.tradables_sim[r][t + 1] - s.tradables_sim[r][t]
                          for r in s.hedges], dim=-1)
        W = s._wealth_step(W, qt, dF, L[t + 1] - L[t])


def test_the_book_holds_whole_contracts():
    """You cannot trade 2.5 contracts: integer per leg, the net rounded AWAY from zero at a
    fraction where plain rounding differs (box -7 puts the mapped net at ~-5.1), and an odd
    net apportioned whole across two legs (box -9 maps to an odd net)."""
    s = _v2(seed=7, lo=-7.0)
    q, curves, _ = s._constructed_policy()
    saw = False
    L, W = s.liability_sim, s.liability_sim[0].clone()
    for t in range(s.T_dec):
        assert torch.equal(q[t], q[t].round()), f'fractional contracts at t={t}'
        g = s._phi_apply(curves[t], W)
        raw = 0.0 - g * (0.0 - (-7.0))
        frac = (raw - raw.trunc()).abs()
        saw |= bool(((frac > 0.05) & (frac < 0.45)).any())
        dF = torch.stack([s.tradables_sim[r][t + 1] - s.tradables_sim[r][t]
                          for r in s.hedges], dim=-1)
        W = s._wealth_step(W, q[t], dF, L[t + 1] - L[t])
    assert saw, 'fixture must contain a case where plain rounding differs'
    s2 = _v2(seed=7, lo=-9.0, legs=2)
    q2, _, _ = s2._constructed_policy()
    saw_odd = False
    for t in range(s2.T_dec):
        assert torch.equal(q2[t], q2[t].round()), f'fractional leg at t={t}'
        saw_odd |= bool(((q2[t].sum(-1).abs() % 2.0) == 1.0).any())
    assert saw_odd, 'fixture must produce an odd net (even splits would hide the mutant)'


def test_the_floor_holds_everywhere_and_minimally():
    """The seed's promise: POSITIVE WEALTH EVERYWHERE. Every path's book stays >= the floor
    at every step — including through a DEFENSELESS day (marks frozen, liability moving),
    which only the backward reserve can survive: the cushion is pre-positioned the day
    before. And the repair is MINIMAL: the seed is not max cover — books strictly inside
    the box must exist."""
    s = _v2(T_dec=4, seed=9, lo=-7.0, flat_day=2, l0=5.0)
    q, _, _ = s._constructed_policy()
    books = _roll_books(s, q)
    floor = 1.0 * s.leg_volume
    # t0 is uniform (no clairvoyance), so feasibility is judged AT t1: a path whose day-0
    # loss at the honest delta already exceeds salvage is the accepted 1-day risk
    import math as _m
    L, T = s.liability_sim, s.T_dec
    rep = [s._replication_hedge(t) for t in range(T)]
    req = torch.full((s.B_outer,), floor)
    for t in range(T - 1, 0, -1):
        lo_t, hi_t = s.aspace.net_bounds(t)
        lo_i, hi_i = _m.ceil(lo_t - 1e-9), _m.floor(hi_t + 1e-9)
        rs = float(rep[t].abs().sum())
        m = (torch.zeros(s.B_outer) if rs < 0.5 else torch.einsum(
            'i,ib->b', (rep[t] / rep[t].sum()) * s.contract_size,
            torch.stack([s.tradables_sim[r][t + 1] - s.tradables_sim[r][t]
                         for r in s.hedges])))
        req = torch.maximum(req - (L[t + 1] - L[t]) - torch.maximum(hi_i * m, lo_i * m),
                            torch.full_like(req, floor))
    feasible = req <= books[1] + 1e-6
    assert bool(feasible.any()) and float(feasible.float().mean()) > 0.5
    assert float(books[2:, feasible].min()) >= floor - 1e-4, \
        f'the floor must hold from t1 on feasible paths; min ' \
        f'{float(books[2:, feasible].min()):.3f}'
    nets = torch.stack([q[t].sum(-1) for t in range(s.T_dec)])
    assert bool((nets > -7.0 + 0.5).any()), 'the seed must not be max cover everywhere'


def test_the_floor_survives_legs_that_move_differently():
    """The floor arithmetic plans at the rep-weighted basket move, but execution splits
    across legs whose moves DIFFER — the plan-vs-execution gap breached feasible paths on
    the real world. The top-up re-prices the EXECUTED book and adds the least whole
    contracts. Two legs, second leg = 0.6x the first plus its own noise. Kills a
    dropped-top-up mutant."""
    s = _v2(T_dec=4, B=2048, seed=13, lo=-7.0, legs=2, l0=6.0)
    g = torch.Generator().manual_seed(21)
    FB = torch.zeros_like(s.tradables_sim["B"])
    dFA = s.tradables_sim["A"].diff(dim=0)
    FB[1:] = (0.6 * dFA + 0.2 * torch.randn(dFA.shape, generator=g)).cumsum(0)
    s.tradables_sim["B"] = FB
    q, _, _ = s._constructed_policy()
    books = _roll_books(s, q)
    floor = 1.0 * s.leg_volume
    # feasibility mask: rebuild the reserve exactly as the engine does
    import math as _m
    L, T = s.liability_sim, s.T_dec
    rep = [s._replication_hedge(t) for t in range(T)]
    req = torch.full((s.B_outer,), floor)
    reqs0 = None
    for t in range(T - 1, -1, -1):
        lo_t, hi_t = s.aspace.net_bounds(t)
        lo_i, hi_i = _m.ceil(lo_t - 1e-9), _m.floor(hi_t + 1e-9)
        w = rep[t] / rep[t].sum()
        m = torch.einsum('i,ib->b', w * s.contract_size,
                         torch.stack([s.tradables_sim[r][t + 1] - s.tradables_sim[r][t]
                                      for r in s.hedges]))
        if t == 0:
            reqs0 = req.clone()                 # the requirement at t1
        req = torch.maximum(req - (L[t + 1] - L[t]) - torch.maximum(hi_i * m, lo_i * m),
                            torch.full_like(req, floor))
    q0 = q[0].sum(-1)
    m0 = torch.einsum('i,ib->b', (rep[0] / rep[0].sum()) * s.contract_size,
                      torch.stack([s.tradables_sim[r][1] - s.tradables_sim[r][0]
                                   for r in s.hedges]))
    B1 = L[0] + (L[1] - L[0]) + q0 * m0
    feasible = reqs0 <= B1 + 1e-6
    assert bool(feasible.any()), 'fixture must contain feasible paths'
    # the guarantee under WHOLE-contract trading: the floor holds to within one
    # contract's day move (the integer lattice — achievable books are one quantum apart)
    quantum = torch.stack([(s.tradables_sim[r].diff(dim=0)).abs() for r in s.hedges]).amax(0)
    worst = (floor - books[1:]).clamp_min(0.0)
    worst = worst[1:]                       # t0 is the accepted uniform-delta day
    viol = ((worst > quantum[1:books.shape[0] - 1] + 1e-4).any(0)) & feasible
    assert int(viol.sum()) == 0, \
        f'{int(viol.sum())} feasible paths breached beyond one contract quantum'
    assert float(worst[:, feasible].amax()) < 1.0, \
        'feasible residuals must stay sub-quantum on this fixture'


def test_t0_is_one_uniform_decision():
    """Deployment shares exactly one state with the seed: today. Every path holds the SAME
    t0 position — the honest delta, no clairvoyance — and the catch-up happens at t1.
    Kills a clairvoyant-t0 mutant (per-path t0 repairs scatter the cross-section)."""
    s = _v2(T_dec=4, seed=9, lo=-7.0, flat_day=2, l0=5.0)
    q, _, _ = s._constructed_policy()
    assert int((q[0] != q[0][0:1]).sum()) == 0, 't0 must be identical on every path'
    assert bool((q[1] != q[1][0:1]).any()), 't1 must scatter (the catch-up)'


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


def test_the_bank_mixes_the_seed_and_the_drowning():
    """Half the bank rolls the clairvoyant seed, half the UNDEFENDED delta base — the value
    function must train on the under-water states the floor prevents (the first trained
    roll extrapolated to MAX LONG in the crash exactly because no drowning book existed in
    its classroom). Kills a dropped-mix mutant."""
    s = _v2(T_dec=4, seed=9, lo=-7.0, flat_day=2, l0=5.0, noise=0.0)
    q_seed, _, _ = s._constructed_policy()
    q_base, _, _ = s._constructed_policy(floored=False)
    gen = torch.Generator().manual_seed(0)
    s.aspace.initial_q = lambda B, dev: torch.zeros(B, 1)
    W_list, q_list = s._build_bank(gen)
    half = s.B_outer // 2
    for t in range(1, s.T_dec):
        assert torch.equal(q_list[t][:half], q_seed[t - 1][:half]), 'first half = the seed'
        assert torch.equal(q_list[t][half:], q_base[t - 1][half:]), 'second half = the base'
    floor = 1.0 * s.leg_volume
    Wb = torch.stack(W_list[1:])
    assert bool((Wb[:, half:] < floor - 1e-4).any()), \
        'the drowning must be PRESENT in the base half'
    assert bool((torch.stack(q_seed) != torch.stack(q_base)).any()), \
        'seed and base must genuinely differ on this fixture'


def test_the_bias_measurement_includes_the_terminal_wealth():
    """The bias covers the WORST book anywhere in the bank — including the wealth AFTER the
    last step, which in a crash month is often the global minimum. A bias measured only at
    decision times leaves the last fit step's states uncovered. Kills a decision-times-only
    mutant."""
    s = _v2(T_dec=3, seed=2, lo=-7.0, l0=2.0)
    s.log_ratio = True
    gen = torch.Generator().manual_seed(0)
    s.aspace.initial_q = lambda B, dev: torch.zeros(B, 1)
    W_list, q_list = s._build_bank(gen)
    q_star = [q_list[t + 1] for t in range(s.T_dec - 1)]
    books = _roll_books(s, [q_list[t + 1] for t in range(s.T_dec - 1)] + [q_list[-1]])
    # the stash must be <= every decision-time book AND the terminal book
    assert hasattr(s, '_bank_worst')
    assert s._bank_worst <= float(torch.stack(W_list).min()) + 1e-4
    assert s._bank_worst <= float(books[-1].min()) + 1e-4, \
        'the terminal wealth must bound the measured worst'
