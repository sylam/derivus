"""`Evaluator.Allocation_Weights` — the FACTORED action space — and `Evaluator.Calendar_Spread_Bps`
— the SPREAD-AWARE charge. Two switches, both default-absent and bit-identical off.

WHY EITHER EXISTS. The 3-leg 9-level Cartesian grid has a 7.5-contract quantum on the NET, while
realistic friction ($23-40 a contract one-way) buys a no-trade band of one or two: no friction can
make trading smooth at that quantum, and measured, the frictional Bellman over it changed the
rolled P&L to the cent. The composition dimension the grid spends its rows on went unused —
composition-only turnover measured 0.0%. So the DP is given ONE number to choose (the net cover,
on a fine ladder) and a deterministic allocator decides where it sits; and because a roll is then
a genuine action, the charge has to know that a matched leg-against-leg move crosses one calendar
quote rather than two outright bid/offers.

ONE RULE PER QUESTION, which is what most of this module pins:

  * WHAT A MOVE COSTS is `hedge_runtime.turnover_charge` — greedy matching across CONSECUTIVE
    maturities, matched volume at one calendar crossing on the pair, every unmatched lot at its
    OWN leg's kappa. The argmax, the regressed target, both benchmark tracks, the verdict's
    diagnostic and the realized `_roll_rebate` all call it, so `decided == realized` for ANY move
    shape. The first cut netted globally in the solver and paired adjacently in the accounting,
    which agreed only on a single-pair roll at a flat mark and disagreed by up to 5× elsewhere.
  * WHICH BOOKS ARE REACHABLE is `grid_at`. Under a schedule the ladder is built on the legs that
    can carry a contract — active AND live — before `net_bounds` and the water-fill, not filtered
    afterwards. Filtering afterwards let 255 contracts sit on an expired leg and made a corridor
    the live legs could satisfy raise.

KILL MATRIX — every mutant applied to the source ALONE, this module run, the death recorded, the
mutant reverted. 32 mutants, 32 deaths, none survived.

| mutant | died at |
| ------ | ------- |
| M1 `_largest_remainder` replaced by `torch.round` | `test_largest_remainder_beats_naive_rounding` (+7) |
| M2 `_weights_at` pins the FIRST knot (no forward fill) | `test_the_weights_forward_fill_between_knots` |
| M3 `_weights_at` pins the LAST knot | `test_the_weights_forward_fill_between_knots` |
| M4 `grid_at` builds the ladder without `live` | `test_the_ladder_is_built_on_the_live_legs` (+3) |
| M5 `grid_at`'s corridor total ignores `live` | `test_absent_weights_leave_the_corridor_filter_untouched` |
| M6 the water-fill runs over every ACTIVE leg, not the weighted ones | `test_the_waterfill_cannot_re_arm_a_zeroed_leg` |
| M7 `_weights_at` drops the inactive-leg zeroing | `test_active_hedge_indices_compose_with_a_schedule` |
| M8 `turnover_charge` nets GLOBALLY instead of pairing adjacently | `test_decided_equals_realized_for_every_move_shape` |
| M9 `turnover_charge` charges the residual at an `abs`-share BLEND of the leg kappas | `test_decided_equals_realized_under_a_per_instrument_spread` |
| M10 `_reposition_charge` drops `kappa_cal` | `test_the_ranking_pays_the_spread_rate` |
| M11 `_fit_step` drops `_calendar_kappa` from the target's charge | `test_the_label_target_reads_the_same_spread_rate` |
| M12 the runtime's roll gate reads `Roll_As_Calendar_Spread` alone | `test_one_key_moves_the_solver_and_the_accounting` |
| M13 `grid_at` builds the ladder whether or not weights are configured | `test_absent_weights_leave_the_cartesian_grid_untouched` (+7) |
| M14 `per_contract_kappa(calendar=True)` charges ONE flat fee | `test_a_matched_roll_pays_both_clearing_fees` |
| M15 the verdict's cost drops `kappa_cal` | `test_the_verdict_cost_prices_a_roll_like_the_argmax` |
| M16 `_policy_artifact` drops the `allocation_weights` stamp | `test_the_artifact_stamps_the_action_universe_and_the_rate` |
| M17 `_policy_artifact` drops the `calendar_spread_bps` stamp | `test_the_artifact_stamps_the_action_universe_and_the_rate` |
| M18 `_check_action_universe` accepts a different ladder | `test_a_foreign_ladder_is_refused_by_name` |
| M19 `_check_calendar_spread` never raises under position state | `test_a_moved_rate_is_refused_under_position_state` |
| M20 the ladder-quantum check deleted | `test_a_ladder_coarser_than_one_contract_is_refused` |
| M21 `universe_size` reports `grid()` under a schedule | `test_the_reported_universe_never_builds_the_meshgrid` |
| M22 the rate-limit band measures per-leg under a schedule | `test_the_rate_limit_band_measures_the_net_under_a_schedule` |
| M23 `static_grid` pins the benchmark to the ENTRY knot ray | `test_the_static_benchmark_sees_every_composition_the_schedule_offers` |
| M24 the ladder's box is not snapped INWARD to whole contracts | `test_a_fractional_box_still_bounds_the_apportioned_rows` |
| M25 no rung dedupe before the ladder cache write | `test_duplicate_rungs_collapse` |
| M26 no sign check on `Calendar_Spread_Bps` | `test_a_rate_that_cannot_mean_anything_is_refused` |
| M27 a rate beside `Roll_As_Calendar_Spread: 'No'` is not refused | `test_a_rate_that_cannot_mean_anything_is_refused` |
| M28 a duplicate (Step, Instrument) row last-wins | `test_a_knot_that_does_not_sum_to_one_is_refused` |
| M29 the textbook benchmark's entry charge drops `kappa_cal` | `test_the_static_benchmark_charges_the_spread_rate` |
| M30 the hindsight track's per-step charge drops `kappa_cal` | `test_the_hindsight_benchmark_charges_the_spread_rate` |
| M31 the hindsight diagnostic reports `grid()` as the universe | `test_the_hindsight_diagnostic_reports_the_ladder` |
| M32 the deck clamps a fixing past the strip onto the last leg | `test_the_deck_refuses_a_fixing_past_the_listed_strip` |

Two of those gates were BLIND when first written, and the matrix is what found them: M6 (the
water-fill's leg set) needed a fixture where the water-fill actually runs — the single-leg knot it
had leaves a deficit of exactly zero — and M15 (the verdict's rate) recomputed the charge two ways
and compared them to each other rather than to the number the accumulator wrote.

WHERE THE CORRIDOR ENTERS THE LADDER, since it is not where it enters the Cartesian grid. The
rungs are laid over `net_bounds(t)` intersected with the weighted live legs' own boxes and snapped
to whole contracts, so the corridor RE-SPACES the ladder and the per-row filter in `grid_at` is
provably a no-op. That is the point rather than an oversight: a rung a caller is offered is a rung
it can realize, and there is no filter left for a live mask to be forgotten by.

The configuration side goes through the JSON contract (`construct_hedge_runtime` on a minimal
simulate_only block, the pattern of `test_cost_model_frictions`); the solver side runs `_decide` /
`_fit_step` / `_verdict` / `_check_load_provenance` / `_policy_artifact` as UNBOUND functions
against a minimal stand-in solver, the harness idiom of `test_position_state`. CPU-only, no
simulation.
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from derivus.hedge_bundle import Bundle, _roll_rebate
from derivus.hedge_runtime import construct_hedge_runtime, per_contract_kappa
from derivus.hedge_solver import (DiffSolver, HedgeActionSpace, HindsightDpSolver,
                                  run_textbook_benchmark)

LEGS = ('M1', 'M2', 'M3')      # maturity-ordered synthetic hedge legs
CS = 50.0                      # contract size (platinum)
PRICE = 2000.0                 # flat mark, so every leg's kappa is the same number
CAP = 50.0                     # Total_Position_Abs_Limit — the Q_max p is measured in
SPREAD_BPS = 10.0              # outright half-spread ⇒ kappa 50.0/contract
CAL_BPS = 4.0                  # calendar half-spread ⇒ kappa 20.0/matched contract
K_OUT = 0.5 * SPREAD_BPS * 1.0e-4 * PRICE * CS        # 50.0, outright, one contract
K_CAL = 0.5 * CAL_BPS * 1.0e-4 * PRICE * CS           # 20.0, calendar, one matched contract


def _alloc(*knots):
    """`Allocation_Weights` rows from `(step, (w1, w2, w3))` knots — the long form the schema
    declares, a leg omitted from a knot being an explicit zero."""
    return [{'Step': step, 'Instrument': n, 'Weight': float(w)}
            for step, ws in knots for n, w in zip(LEGS, ws)]


def _runtime(evaluator=None, levels=51, lo=-50, hi=0, active=None, legs=LEGS, positions=None):
    """Minimal simulate_only runtime built through the JSON contract, with a futures book + one
    cash account and the given Evaluator overrides. `positions` seeds `Portfolio_State`, so the
    benchmark tracks measure their entry turnover from a standing book rather than from flat."""
    ev = {
        'Cash_Instrument': 'USD',
        'Position_Limits': {n: {'Min_Position': lo, 'Max_Position': hi} for n in legs},
        'Total_Position_Abs_Limit': CAP,
        'Bid_Offer_Spread_Bps': SPREAD_BPS,
        'Transaction_Cost_Per_Unit': 0.0,
    }
    ev.update(evaluator or {})
    solver = {'Object': 'DiffSolverV2', 'Training_Action_Grid_Levels_Per_Axis': levels}
    if active is not None:
        solver['Active_Hedge_Indices'] = active
    return construct_hedge_runtime({'Execution_Mode': 'simulate_only', 'Hedging_Problem': {
        'Evaluator': ev, 'Solver': solver,
        'Portfolio_State': {'Positions': dict(zip(legs, positions or ()))},
        'Tradable_Instruments': {
            'CommodityFutureDeal': {n: {'Currency': 'USD', 'Contract_Size': CS} for n in legs},
            'CashAccountDeal': {'USD': {'Currency': 'USD'}},
        }}})


def _aspace(runtime):
    return HedgeActionSpace(runtime, torch.device('cpu'))


def _solver(runtime, position_state=False, T_dec=3, n_steps=4, B=2):
    """The minimal stand-in `_decide` / `_fit_step` run on. `seen` is the probe continuation's
    record of every (W1, p) it was queried with, stashed on the object itself."""
    aspace = _aspace(runtime)
    s = types.SimpleNamespace(
        aspace=aspace, chunk=256, risk_kappa=0.0, churn_lambda=0.0,
        position_state=position_state,
        force_flat=runtime['accounting']['force_flat_at_end'],
        t_min=0, T_dec=T_dec, total_abs_limit=aspace.total_abs_limit,
        hedges=list(aspace.hedges), contract_size=aspace.contract_size,
        device=torch.device('cpu'),
        tradables_sim={r: torch.full((n_steps, B), PRICE) for r in aspace.hedges},
        seen=[],
    )
    s._wealth_step = types.MethodType(DiffSolver._wealth_step, s)
    s._project_leaf_grads = types.MethodType(DiffSolver._project_leaf_grads, s)
    s._unwind_kappa = types.MethodType(DiffSolver._unwind_kappa, s)
    s._calendar_kappa = types.MethodType(DiffSolver._calendar_kappa, s)
    s._check_action_universe = types.MethodType(DiffSolver._check_action_universe, s)
    s._check_calendar_spread = types.MethodType(DiffSolver._check_calendar_spread, s)
    s._reposition_charge = types.MethodType(DiffSolver._reposition_charge, s)
    s._continuation = lambda nets, market, W, t, p: (
        s.seen.append((W.detach().clone(), None if p is None else p.detach().clone())) or W)
    return s


def _decide(s, q_prev=None, t=0, kappa=None, dF=0.0, live=None, B=2, Bi=2, md=1):
    """Run the real `_decide` from zero wealth on a world whose one-step move is the flat `dF` per
    contract, live-masked exactly as `_inner_step` masks it."""
    moves = torch.full((B, Bi, len(s.hedges)), dF) * (1.0 if live is None else live)
    return DiffSolver._decide(
        s, None, torch.zeros(B, Bi, md), moves, torch.zeros(B, Bi),
        torch.zeros(B), t, q_prev=q_prev, kappa=kappa, live=live)


def _fit_step(s, t=0, B=2, Bi=2, md=1, q_prev_val=0.0):
    """Drive the real `_fit_step` on a zero-move world, so the label's target wealth IS minus the
    repositioning charge the chosen action paid."""
    n_h = len(s.hedges)
    live = torch.ones(n_h)
    zero_in = (torch.zeros(B, Bi, n_h), torch.zeros(B, Bi), torch.zeros(B, Bi, md),
               torch.zeros(B, md), live)
    # F_t1 at the OUTER mark, so the one-step move is 0 and the target wealth IS minus the charge.
    s.bundle = types.SimpleNamespace(inner_mc_grad=lambda _t: {
        'F_t1': {r: torch.full((B, Bi), PRICE) for r in s.hedges},
        'L_t1': torch.zeros(B, Bi), 'L_t': torch.zeros(B, Bi),
        'market_t1': torch.zeros(B, Bi, md), 'market_t': torch.zeros(B, md),
        'state_t_leaves': {}, 'state_t_leaf_widths': [],
    })
    label = {}
    s._fit_from_labels = lambda nets, W0_bank, market0, Y, gW, g_market, tt, q_star, p_bank: \
        label.update(Y=Y, q_star=q_star, p_bank=p_bank) or label
    s._decide = types.MethodType(DiffSolver._decide, s)
    DiffSolver._fit_step(s, None, {t: torch.zeros(B)}, t, zero_in,
                         {t: torch.full((B, n_h), q_prev_val)})
    return label


def _decided_and_realized(runtime, dq, prices=None):
    """The two numbers that must be one: what `turnover_cost` charges the argmax for `dq`, and
    what the env actually debits (full L1 leg by leg, less the calendar rebate credited back)."""
    aspace = _aspace(runtime)
    prices = prices or {n: torch.tensor([PRICE]) for n in LEGS}
    kappa = torch.stack([per_contract_kappa(runtime, prices[n], n) for n in LEGS]).reshape(-1)
    cal = torch.stack([per_contract_kappa(runtime, prices[n], n, None, True)
                       for n in LEGS]).reshape(-1)
    delta = torch.tensor([dq], dtype=torch.float32)
    decided = float(aspace.turnover_cost(delta, kappa, cal))
    realized = float((delta.abs() * kappa).sum()) - float(_roll_rebate(
        {n: torch.tensor([float(v)]) for n, v in zip(LEGS, dq)}, prices, runtime))
    return decided, realized


# --- (a) both switches absent: today's action universe and today's charge ---------------------
def test_absent_weights_leave_the_cartesian_grid_untouched():
    """No `Allocation_Weights` ⇒ `grid_at` returns the cached Cartesian grid ITSELF at every t and
    under any live mask — the same object, not merely an equal one, which is the identity the
    corridor gate already pins. Killed by building the ladder unconditionally."""
    a = _aspace(_runtime(levels=5))
    assert a.weights is None
    grid = a.grid()
    mesh = torch.stack([m.reshape(-1) for m in
                        torch.meshgrid(*a.axis_levels(), indexing='ij')], dim=-1)
    assert torch.equal(grid, mesh[mesh.abs().sum(-1) <= CAP + 1e-9])
    for t in (0, 1, 7, 99):
        assert a.grid_at(t) is grid
        assert a.grid_at(t, torch.ones(3)) is grid
    assert a.static_grid() is grid and a.universe_size() == grid.shape[0]


def test_absent_weights_leave_the_corridor_filter_untouched():
    """With a corridor and no weights, `grid_at(t)` is still the Cartesian grid filtered on
    Σ(q·live) — the pre-feature expression, unchanged."""
    a = _aspace(_runtime(levels=5, evaluator={'Total_Position_Schedule': [
        {'Step': 0, 'Min_Total': -50.0, 'Max_Total': 0.0},
        {'Step': 4, 'Min_Total': -20.0, 'Max_Total': -10.0}]}))
    live = torch.tensor([1.0, 1.0, 0.0])
    for t, (lo, hi) in ((0, (-50.0, 0.0)), (3, (-50.0, 0.0)), (4, (-20.0, -10.0))):
        grid = a.grid()
        tot = (grid * live).sum(-1)
        assert torch.equal(a.grid_at(t, live),
                           grid[(tot >= lo - 1e-9) & (tot <= hi + 1e-9)])


def test_absent_calendar_rate_leaves_the_l1_charge_untouched():
    """No `Calendar_Spread_Bps` ⇒ `calendar_kappa` is None, `turnover_cost` is the plain L1 sum,
    and the stepper's roll rebate stays switched off. The whole feature is one key."""
    rt = _runtime()
    a = _aspace(rt)
    assert a.calendar_bps is None
    assert rt['accounting']['roll_as_calendar_spread'] is False
    dq = torch.tensor([[-5.0, 3.0, 0.0], [2.0, -2.0, 1.0]])
    kappa = torch.tensor([1.0, 2.0, 3.0])
    assert torch.equal(a.turnover_cost(dq, kappa), (dq.abs() * kappa).sum(-1))
    s = _solver(rt)
    assert a.calendar_kappa(s.tradables_sim, 0) is None and s._calendar_kappa(0) is None


# --- (b) the ladder: one rung, one net, apportioned ------------------------------------------
def test_every_allocated_row_sums_to_its_rung():
    """The ladder's promise: rung k IS a net position. Every row's legs sum to exactly the rung it
    was built from, in whole contracts, across the full net range."""
    a = _aspace(_runtime(evaluator={'Allocation_Weights': _alloc((0, (0.34, 0.33, 0.33)))}))
    rows = a.allocation_grid(0)
    assert rows.shape == (51, 3)
    assert rows.dtype == torch.float32                 # float-typed, like the Cartesian grid
    assert torch.equal(rows.sum(-1), torch.linspace(-50.0, 0.0, 51))
    assert torch.equal(rows, rows.round())


def test_largest_remainder_beats_naive_rounding():
    """The apportionment case naive rounding loses: `-10 · (0.34, 0.33, 0.33)` is
    `(-3.4, -3.3, -3.3)`, which rounds per-leg to `-9` — a contract the ladder promised and did
    not deliver. Largest-remainder floors to `(-4, -4, -4)` and hands the two leftover units to
    the two largest remainders, landing on `(-4, -3, -3)`. Killed by `torch.round`."""
    a = _aspace(_runtime(evaluator={'Allocation_Weights': _alloc((0, (0.34, 0.33, 0.33)))}))
    rows = a.allocation_grid(0)
    row = rows[rows.sum(-1) == -10.0][0]
    naive = torch.tensor([-3.4, -3.3, -3.3]).round()
    assert float(naive.sum()) == -9.0, 'the fixture must be a case naive rounding fails'
    assert torch.equal(row, torch.tensor([-4.0, -3.0, -3.0]))


def test_the_weights_forward_fill_between_knots():
    """A knot holds until the next one: the composition at t=4 is the t=0 knot's and at t=6 is the
    t=5 knot's, and t=5 is where it moves. Killed by pinning either end of the schedule."""
    a = _aspace(_runtime(evaluator={'Allocation_Weights': _alloc(
        (0, (0.5, 0.5, 0.0)), (5, (0.0, 0.5, 0.5)))}))
    for t in (0, 1, 4):
        assert torch.equal(a._weights_at(t), torch.tensor([0.5, 0.5, 0.0]))
    for t in (5, 6, 99):
        assert torch.equal(a._weights_at(t), torch.tensor([0.0, 0.5, 0.5]))
    assert torch.equal(a.allocation_grid(4), a.allocation_grid(0))
    assert not torch.equal(a.allocation_grid(5), a.allocation_grid(4))


def test_a_zeroed_leg_gets_nothing():
    """A leg the knot zeroes carries no contract at any rung — which is how an expired leg leaves
    the ladder — and a leg the table simply omits is the same explicit zero."""
    a = _aspace(_runtime(evaluator={'Allocation_Weights': _alloc((0, (0.0, 0.7, 0.3)))}))
    rows = a.allocation_grid(0)
    assert float(rows[:, 0].abs().max()) == 0.0
    assert float(rows[:, 1:].abs().sum()) > 0.0
    omitted = _aspace(_runtime(evaluator={'Allocation_Weights': [
        {'Step': 0, 'Instrument': 'M2', 'Weight': 0.7},
        {'Step': 0, 'Instrument': 'M3', 'Weight': 0.3}]}))
    assert torch.equal(omitted.allocation_grid(0), rows)


def test_the_waterfill_cannot_re_arm_a_zeroed_leg():
    """The water-fill exists to honour the rung when a leg's box binds — and a leg the schedule
    zeroed has a WHOLE EMPTY BOX of headroom, so a water-fill over every active leg hands it the
    shortfall: measured, weights (1,0,0) against an M1 box of [-20,0] emitted [-20,-15,-15], 30
    contracts on legs the schedule said must carry none. The ladder's reach is what shrinks
    instead — 50 contracts of cover are simply not available on one leg that stops at 20.

    Killed by water-filling over `self.active` rather than the weighted legs."""
    rt = _runtime(evaluator={'Allocation_Weights': _alloc((0, (1.0, 0.0, 0.0)))})
    rt['accounting']['position_limits']['M1']['min_position'] = -20
    rows = _aspace(rt).allocation_grid(0)
    assert float(rows[:, 1:].abs().sum()) == 0.0, 'the zeroed legs must stay empty'
    assert (float(rows.sum(-1).min()), float(rows.sum(-1).max())) == (-20.0, 0.0)
    assert torch.equal(rows[0], torch.tensor([-20.0, 0.0, 0.0]))
    # ...and the case where the water-fill actually RUNS: a binding box on one weighted leg
    # leaves a real deficit, which must land wholly on the OTHER weighted leg. Above, the
    # ladder's own reach shrinks to the single leg's box and the deficit is identically zero,
    # so that fixture alone cannot see which leg set the water-fill was given.
    rt = _runtime(evaluator={'Allocation_Weights': _alloc((0, (0.6, 0.4, 0.0)))})
    rt['accounting']['position_limits']['M1']['min_position'] = -10
    rows = _aspace(rt).allocation_grid(0)
    assert float(rows[:, 2].abs().sum()) == 0.0, 'the deficit may not spill onto a zeroed leg'
    assert torch.equal(rows[rows.sum(-1).argmin()], torch.tensor([-10.0, -40.0, 0.0]))


def test_a_knot_that_does_not_sum_to_one_is_refused():
    """The JSON boundary validates rather than guesses: weights are non-negative, a knot sums to 1,
    names a real hedge leg, and names it once. A duplicate that happens to keep the sum at 1 would
    otherwise last-win and silently drop the author's first row."""
    with pytest.raises(ValueError, match='sums to'):
        _runtime(evaluator={'Allocation_Weights': _alloc((0, (0.5, 0.3, 0.1)))})
    with pytest.raises(ValueError, match='Weight must be'):
        _runtime(evaluator={'Allocation_Weights': _alloc((0, (1.4, -0.2, -0.2)))})
    with pytest.raises(ValueError, match='not a hedge leg'):
        _runtime(evaluator={'Allocation_Weights': [
            {'Step': 0, 'Instrument': 'NOPE', 'Weight': 1.0}]})
    with pytest.raises(ValueError, match='twice at Step'):
        _runtime(evaluator={'Allocation_Weights': [
            {'Step': 0, 'Instrument': 'M1', 'Weight': 0.5},
            {'Step': 0, 'Instrument': 'M1', 'Weight': 0.3},
            {'Step': 0, 'Instrument': 'M2', 'Weight': 0.7}]})


def test_a_ladder_coarser_than_one_contract_is_refused():
    """`Training_Action_Grid_Levels_Per_Axis` counts rungs on the TOTAL here, and nothing else
    compares it against the net range: at the shipping default of 9 the ladder over [-50, 0] has a
    7-contract quantum — the very grid the feature exists to replace, silently rebuilt. Refused
    loud, naming the number that would work. Killed by deleting the check."""
    for levels in (9, 11):
        with pytest.raises(ValueError, match='rungs on the TOTAL'):
            _aspace(_runtime(levels=levels, evaluator={
                'Allocation_Weights': _alloc((0, (0.5, 0.3, 0.2)))})).allocation_grid(0)
    nets = _aspace(_runtime(levels=51, evaluator={
        'Allocation_Weights': _alloc((0, (0.5, 0.3, 0.2)))})).allocation_grid(0).sum(-1)
    assert float(nets.diff().abs().max()) == 1.0


def test_duplicate_rungs_collapse():
    """A band narrower than `levels` contracts makes rungs that round together; scoring the same
    book 4 times over is waste on exactly the axis the feature was meant to make cheap."""
    a = _aspace(_runtime(levels=201, evaluator={'Allocation_Weights': _alloc(
        (0, (0.5, 0.3, 0.2)))}))
    rows = a.allocation_grid(0)
    assert rows.shape[0] == rows.unique(dim=0).shape[0] == 56


# --- (c) the ladder still obeys every constraint the Cartesian grid did -----------------------
def test_allocated_rows_respect_the_per_leg_boxes():
    """An uneven schedule against an uneven box: 60% of a -50 net is -30 contracts, and M1's box
    stops at -10. The rung must still be -50, so the excess water-fills onto the OTHER WEIGHTED
    legs, exactly as the corridor projection does. Killed by dropping the clamp + water-fill (the
    row becomes (-30, -15, -5) and M1 sits 20 contracts outside its own limit)."""
    rt = _runtime(evaluator={'Allocation_Weights': _alloc((0, (0.6, 0.3, 0.1)))})
    rt['accounting']['position_limits']['M1']['min_position'] = -10
    a = _aspace(rt)
    rows = a.allocation_grid(0)
    assert (rows >= a.q_lo - 1e-9).all() and (rows <= a.q_hi + 1e-9).all()
    assert torch.equal(rows.sum(-1), torch.linspace(-50.0, 0.0, 51))
    assert torch.equal(rows[0], torch.tensor([-10.0, -24.0, -16.0]))       # the deep rung


def test_a_fractional_box_still_bounds_the_apportioned_rows():
    """`_largest_remainder` floors toward −∞, so a leg clamped to a FRACTIONAL Min_Position could
    floor below its own box and not get the leftover unit back (45,362 of 200,000 random draws).
    The band and the clamp are snapped INWARD to whole contracts first, which is the honest fix:
    the apportionment deals in contracts, so the box it respects has to as well."""
    rt = _runtime(evaluator={'Allocation_Weights': _alloc((0, (0.358, 0.610, 0.032)))})
    rt['accounting']['position_limits']['M1']['min_position'] = -16.75
    a = _aspace(rt)
    rows = a.allocation_grid(0)
    assert (rows[:, 0] >= -16.75).all(), 'no row may sit below its own Min_Position'
    assert (rows >= a.q_lo).all() and (rows <= a.q_hi).all()
    assert torch.equal(rows.sum(-1), rows.sum(-1).round())


def test_allocated_rows_respect_the_total_cap():
    """Weights are non-negative, so a row's legs all carry the rung's sign and Σ|q| IS |Q| — the
    cap binds through `net_bounds`, and the row filter `grid()` applies is then a no-op rather
    than a skipped step. Both statements are pinned here."""
    a = _aspace(_runtime(evaluator={'Allocation_Weights': _alloc((0, (0.5, 0.3, 0.2)))}))
    rows = a.allocation_grid(0)
    assert float(rows.abs().sum(-1).max()) <= CAP + 1e-9
    assert a.net_bounds(0) == (-CAP, 0.0)               # -150 of per-leg room, capped at -50
    assert rows.shape[0] == 51                          # nothing dropped


def test_the_corridor_bounds_the_allocated_rungs_at_t():
    """`Total_Position_Schedule` reaches the ladder through `net_bounds(t)`, which is the interval
    `grid_at` can realize: at a step whose corridor is [-20, -10] the rungs are RE-SPACED over that
    band rather than a wide ladder being filtered down to a handful. Piecewise-constant in t, like
    every other reader of the schedule."""
    a = _aspace(_runtime(evaluator={
        'Allocation_Weights': _alloc((0, (0.5, 0.3, 0.2))),
        'Total_Position_Schedule': [{'Step': 0, 'Min_Total': -50.0, 'Max_Total': 0.0},
                                    {'Step': 4, 'Min_Total': -20.0, 'Max_Total': -10.0}]}))
    early, late = a.grid_at(0), a.grid_at(4)
    assert (float(early.sum(-1).min()), float(early.sum(-1).max())) == (-50.0, 0.0)
    assert (float(late.sum(-1).min()), float(late.sum(-1).max())) == (-20.0, -10.0)
    assert torch.equal(a.grid_at(3), early) and torch.equal(a.grid_at(9), late)


def test_the_ladder_is_built_on_the_live_legs():
    """Expiry has to reach the ladder BEFORE it is built, not as a filter afterwards. A knot's 20%
    on a dead leg is cover the book cannot realize: left in, every rung advertised a net it masked
    away from (measured: 255 contracts parked on the expired leg, and the deepest realizable short
    fell from -50 to -40 while the Cartesian grid still reached -50). Intersected first, the live
    legs carry the whole rung and the ladder realizes its full range.

    Killed by dropping `live` from `_weights_at`/`allocation_grid`."""
    a = _aspace(_runtime(evaluator={'Allocation_Weights': _alloc((0, (0.5, 0.3, 0.2)))}))
    live = torch.tensor([1.0, 1.0, 0.0])               # M3 expired, and it carries 20% of the rung
    rows = a.grid_at(7, live)
    assert float(rows[:, 2].abs().sum()) == 0.0, 'nothing may be parked on the dead leg'
    assert float((rows * live).sum(-1).min()) == -50.0, 'the full net range stays realizable'
    assert torch.equal(rows.sum(-1), (rows * live).sum(-1)), 'advertised IS realized'
    assert torch.equal(a._weights_at(7, live), torch.tensor([0.625, 0.375, 0.0]))
    assert not torch.equal(rows, a.grid_at(7))


def test_a_corridor_the_live_legs_can_satisfy_does_not_raise():
    """The other face of the same defect: with 60% of the knot on a dead leg, a corridor demanding
    a 45-to-50 short looked infeasible (the live legs only reached -20 of the advertised rung) and
    `grid_at` aborted the run — on a mandate the live legs can meet twice over."""
    a = _aspace(_runtime(evaluator={
        'Allocation_Weights': _alloc((0, (0.6, 0.3, 0.1))),
        'Total_Position_Schedule': [{'Step': 0, 'Min_Total': -50.0, 'Max_Total': -45.0}]}))
    rows = a.grid_at(0, torch.tensor([0.0, 1.0, 1.0]))
    assert rows.shape[0] > 0
    assert float(rows[:, 0].abs().max()) == 0.0
    assert (float(rows.sum(-1).min()), float(rows.sum(-1).max())) == (-50.0, -45.0)


def test_a_knot_with_no_live_weight_fails_loud():
    """...and when the knot really does put everything on legs that are gone, there is no ladder to
    build. Loud, naming the step, rather than a NaN row or an empty grid."""
    a = _aspace(_runtime(evaluator={'Allocation_Weights': _alloc((0, (1.0, 0.0, 0.0)))}))
    with pytest.raises(ValueError, match='no weight on a LIVE, ACTIVE'):
        a.grid_at(0, torch.tensor([0.0, 1.0, 1.0]))


def test_active_hedge_indices_compose_with_a_schedule():
    """`Active_Hedge_Indices` pins an axis to 0 in the Cartesian grid; under a schedule that does
    not know about it, the knot's weight on the pinned leg is zeroed and the rest renormalized.
    Killed by dropping the inactive-leg mask (the pinned leg takes 30% of every rung)."""
    a = _aspace(_runtime(evaluator={'Allocation_Weights': _alloc((0, (0.5, 0.3, 0.2)))},
                         active=[0, 2]))
    assert torch.equal(a._weights_at(0), torch.tensor([0.5, 0.0, 0.2]) / 0.7)
    rows = a.allocation_grid(0)
    assert float(rows[:, 1].abs().max()) == 0.0
    assert torch.equal(rows.sum(-1), torch.linspace(-50.0, 0.0, 51))


# --- (d) ONE charge: decided == realized, for every move shape --------------------------------
@pytest.mark.parametrize('dq,matched,residual', [
    ((5, -5, 0), 5, 0),           # one adjacent pair — the only shape the first cut got right
    ((-5, 0, 5), 0, 10),          # NON-adjacent: no listed spread between M1 and M3
    ((25, -10, -15), 10, 30),     # M1/M2 match 10, then M2 is spent; M3 is outright
    ((-5, -5, 10), 5, 10),
    ((-10, 5, 5), 5, 10),
    ((9, -4, -5), 4, 10),
    ((-8, 5, 0), 5, 3),
    ((5, -8, 3), 8, 0),           # a chained roll: M1/M2 then the M2 residual against M3
    ((0, 0, 0), 0, 0),
])
def test_decided_equals_realized_for_every_move_shape(dq, matched, residual):
    """THE one-spec gate, and the reason the rule moved into `hedge_runtime`. What the argmax is
    charged for a move and what the env debits for it must be the same number for EVERY shape, not
    just for a single-pair roll at a flat mark — which is the one case global netting and adjacent
    pairing happen to agree on, and the case the first cut pinned. Measured before the fix:
    (-5,0,+5) 100 decided vs 500 realized, a 5× gap.

    `matched`/`residual` restate the sweep independently of the code: matched contracts pay the
    calendar rate, everything left pays outright. Killed by netting globally in either seam."""
    rt = _runtime(evaluator={'Calendar_Spread_Bps': CAL_BPS})
    decided, realized = _decided_and_realized(rt, dq)
    assert decided == pytest.approx(realized, rel=1e-5)
    assert decided == pytest.approx(matched * K_CAL + residual * K_OUT, rel=1e-5)


def test_decided_equals_realized_under_a_per_instrument_spread():
    """The second half of the same rule: an UNMATCHED lot sits on one leg and crosses that leg's
    bid/offer, so it pays that leg's own kappa — not a blend of the legs the roll touched. Under
    the `Per_Instrument` spread the schema already supports, the blend charged 6 leftover M1 lots
    at 57.14 instead of 20.00 (decided 382.86 against a realized 160.00, +139%).

    Killed by restoring the `abs`-share-weighted kappa on either half."""
    rt = _runtime(evaluator={
        'Bid_Offer_Spread_Bps': {'Default_Bps': 10.0,
                                 'Per_Instrument': {'M1': 4.0, 'M2': 30.0, 'M3': 10.0}},
        'Calendar_Spread_Bps': 2.0})
    kappa = [float(per_contract_kappa(rt, torch.tensor(PRICE), n)) for n in LEGS]
    assert kappa == pytest.approx([20.0, 150.0, 50.0], rel=1e-5)
    for dq, want in (((-10, 4, 0), 4 * 10.0 + 6 * 20.0),        # 4 matched, 6 residual ON M1
                     ((0, -10, 5), 5 * 10.0 + 5 * 150.0),
                     ((0, 5, -10), 5 * 10.0 + 5 * 50.0),
                     ((-5, 0, 5), 5 * 20.0 + 5 * 50.0)):        # non-adjacent: both outright
        decided, realized = _decided_and_realized(rt, dq)
        assert decided == pytest.approx(realized, rel=1e-5)
        assert decided == pytest.approx(want, rel=1e-5)


def test_decided_equals_realized_under_contango_marks():
    """And with each leg on its own mark — the everyday case, no exotic spread spec — where the
    blended kappa left a persistent signed gap of +0.12…+0.15%."""
    rt = _runtime(evaluator={'Calendar_Spread_Bps': CAL_BPS})
    prices = {'M1': torch.tensor([900.0]), 'M2': torch.tensor([906.0]),
              'M3': torch.tensor([913.0])}
    for dq in ((-10, 4, 0), (5, -5, 0), (0, -8, 3), (-5, 0, 5)):
        decided, realized = _decided_and_realized(rt, dq, prices)
        assert decided == pytest.approx(realized, rel=1e-6)


def test_a_matched_roll_pays_both_clearing_fees():
    """A matched contract moves TWO futures and clears two of them, so it pays the flat
    `Transaction_Cost_Per_Unit` twice and one half-spread at the calendar rate on the average leg
    notional — which is what the pre-feature `_roll_rebate` charged, and what
    `per_contract_kappa(calendar=True)` has to reproduce. Every existing roll gate pins the fee to
    0.0, so only this one can see it. Killed by charging one fee."""
    for tc in (0.0, 3.0, 3.13):
        rt = _runtime(evaluator={'Calendar_Spread_Bps': 5.0, 'Transaction_Cost_Per_Unit': tc})
        spread = 0.5 * 5.0 * 1.0e-4 * PRICE * CS
        assert float(per_contract_kappa(rt, torch.tensor(PRICE), 'M1', None, True)) == \
            pytest.approx(2.0 * tc + spread)
        decided, realized = _decided_and_realized(rt, (-5, 5, 0))
        assert decided == pytest.approx(realized, rel=1e-5)
        assert decided == pytest.approx(5 * (2.0 * tc + spread), rel=1e-5)


def test_one_key_moves_the_solver_and_the_accounting():
    """MUTATE-THEN-VERIFY on the single key. Absent, the solver charges L1 and the stepper's roll
    rebate is not even armed. Present, BOTH move, to the same number, and both keep moving with
    the rate. Killed by the runtime's roll gate reading `Roll_As_Calendar_Spread` alone."""
    off = _runtime()
    assert off['accounting']['roll_as_calendar_spread'] is False
    dq = (-5, 5, 0)
    outright = 10 * K_OUT
    seen = []
    for bps in (CAL_BPS, 2 * CAL_BPS):
        rt = _runtime(evaluator={'Calendar_Spread_Bps': bps})
        assert rt['accounting']['roll_as_calendar_spread'] is True   # the key arms the stepper
        decided, realized = _decided_and_realized(rt, dq)
        assert decided == pytest.approx(realized, rel=1e-5)
        assert realized < outright
        seen.append(realized)
    assert seen[1] > seen[0], 'a wider spread costs more, on both sides'


def test_a_rate_that_cannot_mean_anything_is_refused():
    """A rate at or below zero makes matched volume free — or paid for, a money pump one typo
    away — and a rate beside an explicit `Roll_As_Calendar_Spread: 'No'` is a contradiction rather
    than a precedence question. Absence is how the feature is switched off."""
    for bad in (0.0, -4.0):
        with pytest.raises(ValueError, match='must be > 0'):
            _runtime(evaluator={'Calendar_Spread_Bps': bad})
    with pytest.raises(ValueError, match='beside'):
        _runtime(evaluator={'Calendar_Spread_Bps': CAL_BPS, 'Roll_As_Calendar_Spread': 'No'})


# --- (e) every seam reads the one rate --------------------------------------------------------
def test_the_ranking_pays_the_spread_rate():
    """`_decide` prices every candidate through `_reposition_charge`, so with a zero one-step move
    each candidate's continuation wealth IS minus its charge — the decomposed one. Killed by
    `_reposition_charge` ignoring `kappa_cal`."""
    rt = _runtime(levels=3, evaluator={'Calendar_Spread_Bps': CAL_BPS})
    s = _solver(rt)
    kappa = s.aspace.kappa(s.tradables_sim, 0)
    q_prev = torch.tensor([[-25.0, 0.0, 0.0], [-25.0, 0.0, 0.0]])
    _decide(s, q_prev=q_prev, kappa=kappa, dF=0.0)
    grid = s.aspace.grid_at(0)
    seen = s.seen[0][0].reshape(2, grid.shape[0], 2)[0, :, 0]
    expect = -s.aspace.turnover_cost(grid - q_prev[0], kappa, s._calendar_kappa(0))
    assert torch.allclose(seen, expect)
    assert not torch.allclose(seen, -s.aspace.turnover_cost(grid - q_prev[0], kappa))


def test_the_label_target_reads_the_same_spread_rate():
    """The seam gate: under `DiffV2_Position_State` the chosen action's charge is subtracted from
    the wealth entering the regressed TARGET, and it must be the identical number the ranking
    used. Killed by `_fit_step` charging the outright rate the ranking did not."""
    rt = _runtime(levels=3, evaluator={'Calendar_Spread_Bps': CAL_BPS})
    s = _solver(rt, position_state=True)
    label = _fit_step(s, q_prev_val=-8.0)
    q_prev = torch.full((2, 3), -8.0)
    kappa = s.aspace.kappa(s.tradables_sim, 0)
    expect = -s._reposition_charge(label['q_star'], q_prev, kappa, None, s._calendar_kappa(0))
    assert torch.allclose(label['Y'], expect)
    assert not torch.allclose(label['Y'], -s.aspace.turnover_cost(label['q_star'] - q_prev, kappa))


def _verdict_solver(runtime, T_dec=3, B=8, Bi=2, md=1, dF=None):
    """A stand-in `_verdict` runs on: real wealth law, identity continuation, flat PRICE marks so
    kappa is the configured spread. `dF` is the per-leg one-step move the argmax forecasts, big
    enough to decide the book against any friction — the outer marks stay flat, so the wealth
    track does not move and the turnover accumulator is what the gate reads."""
    s = _solver(runtime, position_state=False, T_dec=T_dec, n_steps=T_dec + 1, B=B)
    n_h = len(s.hedges)
    s.t_min, s.cost_aware = 0, True
    s.B_outer, s.n_hedge, s.active, s.n_active = B, n_h, list(range(n_h)), n_h
    s.q_lo, s.q_hi = s.aspace.q_lo, s.aspace.q_hi
    s.tradables_sim = {r: torch.full((T_dec + 1, B), PRICE) for r in s.hedges}
    s.liability_sim = torch.zeros(T_dec + 1, B)
    s._u = lambda W: W
    s._decide = types.MethodType(DiffSolver._decide, s)
    s._replication_hedge = types.MethodType(DiffSolver._replication_hedge, s)
    s._continuation = lambda nets, market, W, t, p: W
    moves = torch.zeros(n_h) if dF is None else torch.tensor(dF)
    inner = {t: (moves.expand(B, Bi, n_h).contiguous(), torch.zeros(B, Bi),
                 torch.zeros(B, Bi, md), torch.zeros(B, md), torch.ones(n_h))
             for t in range(T_dec)}
    return s, inner, list(range(T_dec))


def test_the_verdict_cost_prices_a_roll_like_the_argmax():
    """`turnover_cost_mean` / `*_net` are what the campaign reads, and they were still summing L1
    while the argmax that CHOSE the book, the fitted target and the stepper's realized debit all
    priced spreads — three seams, three numbers, biased against the roll-heavy greedy track.

    The opening book is short M2 and the greedy track moves to M1, so the entry is a genuine roll
    and the two prices differ. Recomputed from the REPORTED trajectory and compared against the
    REPORTED number — recomputing both ways and comparing them to each other, as this gate first
    did, passes whatever the accumulator actually wrote. Killed by dropping `kappa_cal` there."""
    rt = _runtime(levels=3, evaluator={'Calendar_Spread_Bps': CAL_BPS}, positions=(0, -25.0, 0))
    s, inner, ts = _verdict_solver(rt, dF=(-10.0, 0.0, 0.0))       # M1 falls: short it, roll off M2
    out = DiffSolver._verdict(s, None, inner, ts)
    a, sim, traj = s.aspace, s.tradables_sim, out['greedy_q_traj']
    moves = [(t, [c - p for c, p in zip(q, prev)])
             for t, (q, prev) in enumerate(zip(traj, [[0.0, -25.0, 0.0]] + traj[:-1]))]
    assert moves[0][1] == [-50.0, 25.0, 0.0], 'the fixture must open with a genuine roll'
    unwind = float(a.turnover_cost(torch.tensor([traj[-1]]), a.kappa(sim, s.T_dec)))
    assert out['greedy']['turnover_cost_mean'] == pytest.approx(
        _track_charge(a, sim, moves, True) + unwind, rel=1e-6)
    assert _track_charge(a, sim, moves, True) < _track_charge(a, sim, moves, False)
    assert out['argmax_charged'] is True


def _bundle(runtime, marks):
    """The stand-in `run_textbook_benchmark` / `HindsightDpSolver` consume: realized outer paths
    (the REAL `HedgeBundle.realized_paths`, so the two tracks read their marks exactly as they do
    in a run), no liability, no vol series. `marks` gives each leg its `(T+1,)` price path."""
    B = 2
    sim = {n: torch.tensor(m, dtype=torch.float32)[:, None].expand(-1, B).contiguous()
           for n, m in zip(LEGS, marks)}
    T = len(marks[0]) - 1
    b = types.SimpleNamespace(
        device=torch.device('cpu'), vol_sim=None, n_outer_steps=T + 1, last_live_mtm_index=T,
        liability_sim=torch.zeros(T + 1, B), tradables_sim=sim)
    b.realized_paths = types.MethodType(Bundle.realized_paths, b)
    return b


def _track_charge(aspace, sim, moves, calendar):
    """What a track's accumulator SHOULD report for a `(t, dq)` trajectory: the one rule at each
    step, plus the outright terminal unwind. `calendar=False` recomputes the plain L1 the six
    2-arg call sites were charging instead."""
    total = 0.0
    for t, dq in moves:
        cal = aspace.calendar_kappa(sim, t) if calendar else None
        total += float(aspace.turnover_cost(torch.tensor([dq]), aspace.kappa(sim, t), cal))
    return total


def test_the_static_benchmark_charges_the_spread_rate():
    """`run_textbook_benchmark`'s `turnover_cost_mean` / `v0_mean_net` are a campaign column, and
    it was still summing L1 while the argmax that chose the book priced spreads. Here the hold is
    a genuine roll off the opening book — M2 shorted, the standing M1 bought back — so the two
    prices differ. Killed by dropping `calendar_kappa` from the entry charge."""
    rt = _runtime(levels=3, evaluator={'Calendar_Spread_Bps': CAL_BPS}, positions=(-25.0, 0, 0))
    # M2 is the leg that falls, so the best constant short sits there; M1/M3 rise.
    bundle = _bundle(rt, ([PRICE, PRICE + 100.0], [PRICE, PRICE - 100.0], [PRICE, PRICE + 100.0]))
    out = run_textbook_benchmark(bundle, rt)
    a, sim = _aspace(rt), bundle.tradables_sim
    assert out['n_star'] == [0.0, -50.0, 0.0]
    entry = [(0, (25.0, -50.0, 0.0))]                  # from q0 = (-25, 0, 0): a matched roll
    unwind = float(a.turnover_cost(torch.tensor([[0.0, -50.0, 0.0]]), a.kappa(sim, 1)))
    assert out['turnover_cost_mean'] == pytest.approx(
        _track_charge(a, sim, entry, True) + unwind, rel=1e-6)
    assert out['turnover_cost_mean'] < _track_charge(a, sim, entry, False) + unwind


def test_the_hindsight_benchmark_charges_the_spread_rate():
    """The same seam on the clairvoyant track, whose bang-bang trajectory is ALL roll: perfect
    foresight moves the whole book from the leg that fell this step to the leg that falls next,
    which is exactly the move L1 over-charges. Killed by dropping `calendar_kappa` from the
    per-step accumulator."""
    rt = _runtime(levels=3, evaluator={'Calendar_Spread_Bps': CAL_BPS})
    bundle = _bundle(rt, ([PRICE, PRICE - 100.0, PRICE - 100.0],      # M1 falls on step 0
                          [PRICE, PRICE, PRICE - 100.0],              # M2 falls on step 1
                          [PRICE, PRICE, PRICE]))
    out = HindsightDpSolver(bundle, rt).solve()
    a, sim = _aspace(rt), bundle.tradables_sim
    moves = [(0, (-50.0, 0.0, 0.0)), (1, (50.0, -50.0, 0.0))]        # enter M1, then roll to M2
    unwind = float(a.turnover_cost(torch.tensor([[0.0, -50.0, 0.0]]), a.kappa(sim, 2)))
    assert out.diagnostics['turnover_cost_mean'] == pytest.approx(
        _track_charge(a, sim, moves, True) + unwind, rel=1e-6)
    assert out.diagnostics['turnover_cost_mean'] < _track_charge(a, sim, moves, False) + unwind


def test_the_calendar_kappa_is_the_one_switch_reader():
    """`Calendar_Spread_Bps` goes through `per_contract_kappa` (same notional, same vol scale, the
    flat fee twice), and `aspace.calendar_kappa` is the ONLY place the None-means-off switch is
    read — the solver seam, the benchmarks and the verdict all take their rate from it."""
    rt = _runtime(evaluator={'Calendar_Spread_Bps': CAL_BPS, 'Transaction_Cost_Per_Unit': 3.0})
    price = torch.tensor([PRICE])
    assert float(per_contract_kappa(rt, price, 'M1')) == pytest.approx(3.0 + K_OUT)
    assert float(per_contract_kappa(rt, price, 'M1', calendar=True)) == \
        pytest.approx(6.0 + K_CAL)
    s = _solver(rt)
    assert torch.allclose(s._calendar_kappa(0), torch.full((3,), 6.0 + K_CAL))
    assert torch.equal(s._calendar_kappa(0), s.aspace.calendar_kappa(s.tradables_sim, 0))
    assert _solver(_runtime()).aspace.calendar_kappa(s.tradables_sim, 0) is None


# --- (f) composition with everything downstream -----------------------------------------------
def test_an_allocated_row_carries_its_own_position_state():
    """`_decide` queries the successor at the book each CANDIDATE leaves standing,
    p' = Σ(a·live)/Q_max. An allocated row is a book like any other — and because the ladder is
    built on the live legs, its whole rung is live, so p' is the rung over the cap exactly."""
    rt = _runtime(evaluator={'Allocation_Weights': _alloc((0, (0.5, 0.3, 0.2)))})
    s = _solver(rt, position_state=True)
    live = torch.tensor([1.0, 1.0, 0.0])
    _decide(s, q_prev=torch.zeros(2, 3), kappa=None, dF=-1.0, live=live)
    rows = s.aspace.grid_at(0, live)
    p_seen = s.seen[0][1].reshape(2, rows.shape[0], 2)[0, :, 0]
    assert torch.equal(p_seen, (rows * live).sum(-1) / CAP)
    assert torch.equal(p_seen, torch.linspace(-50.0, 0.0, 51) / CAP)
    assert (p_seen >= -1.0).all() and (p_seen <= 1.0).all()


def test_the_rate_limit_band_measures_the_net_under_a_schedule():
    """`Max_Trade_Per_Step` is a per-leg band, and a knot recomposes EVERY leg: measured, 0 of 51
    rows at a knot step satisfied it, so `_decide`'s all-infeasible fallback handed the run the
    unrestricted argmax and the cap silently stopped existing. Under a schedule the policy's
    control is the net cover — the composition jump is the schedule's mandate — so the band
    measures |ΔΣq| and actually binds. Killed by restoring the per-leg test."""
    rt = _runtime(evaluator={
        'Allocation_Weights': _alloc((0, (0.6, 0.4, 0.0)), (5, (0.0, 0.5, 0.5))),
        'Max_Trade_Per_Step': 3.0})
    s = _solver(rt)
    q_prev = torch.tensor([[-24.0, -16.0, 0.0], [-24.0, -16.0, 0.0]])   # -40 net, old knot
    q, _ = _decide(s, q_prev=q_prev, kappa=None, dF=-1.0, t=5)
    assert float((q.sum(-1) - q_prev.sum(-1)).abs().max()) <= 3.0, 'the band must bind'
    assert float(q[0, 0].abs()) == 0.0, 'and the mandated recomposition still happens'
    # the same run with the band off walks straight to the corner, so the band is what bound it
    free = _solver(_runtime(evaluator={'Allocation_Weights': _alloc(
        (0, (0.6, 0.4, 0.0)), (5, (0.0, 0.5, 0.5)))}))
    q_free, _ = _decide(free, q_prev=q_prev, kappa=None, dF=-1.0, t=5)
    assert float(q_free.sum(-1)[0]) == -50.0


# --- (g) provenance: the two new keys are part of what a checkpoint IS -------------------------
def _artifact(s):
    s.runtime = {'objective': {'utility_scale': 1.0}, 'solver': {}}
    s.m_mean = s.m_std = torch.ones(1)
    s.w_mean = s.w_std = torch.tensor(1.0)
    s.a_bounds = [None]
    s.t_min, s.active = 0, [0, 1, 2]
    s._config_hash = lambda: 'h'
    s._frame_stamp = lambda: 'f'
    return DiffSolver._policy_artifact(s, [], 1, 8, 0.0, [0.0], 0.0)


def test_the_artifact_stamps_the_action_universe_and_the_rate():
    """`config_hash` cannot stand in for either: it hashes `runtime['solver']` and both keys live
    under `runtime['accounting']`. So the artifact has to stamp them, and its own checkpoint has to
    load back. Killed by dropping either stamp."""
    rt = _runtime(evaluator={'Allocation_Weights': _alloc((0, (0.5, 0.3, 0.2))),
                             'Calendar_Spread_Bps': CAL_BPS})
    s = _solver(rt)
    ck = _artifact(s)
    assert ck['allocation_weights'] == ((0, (0.5, 0.3, 0.2)),)
    assert ck['calendar_spread_bps'] == CAL_BPS
    assert DiffSolver._check_load_provenance(s, ck, '<ck>', 1) == ck['solver_version']
    plain = _artifact(_solver(_runtime()))
    assert plain['allocation_weights'] is None and plain['calendar_spread_bps'] is None


def test_a_foreign_ladder_is_refused_by_name():
    """A policy fitted on one ladder and rolled on a different one scored a different set of books
    — off the support it learned, and under position state off the p-support too. Refused, because
    the symptom otherwise is a silently worse roll rather than an error. A CARTESIAN-trained
    checkpoint is the widest universe, so restricting it to a ladder is valid and merely said."""
    trained = _artifact(_solver(_runtime(evaluator={
        'Allocation_Weights': _alloc((0, (0.5, 0.3, 0.2)))})))
    other = _solver(_runtime(evaluator={'Allocation_Weights': _alloc((0, (0.8, 0.1, 0.1)))}))
    with pytest.raises(ValueError, match='Allocation_Weights mismatch'):
        DiffSolver._check_load_provenance(other, trained, '<ck>', 1)
    with pytest.raises(ValueError, match='Allocation_Weights mismatch'):
        DiffSolver._check_load_provenance(_solver(_runtime()), trained, '<ck>', 1)
    # ...but Cartesian-trained -> ladder is a restriction to a learned subset
    cartesian = _artifact(_solver(_runtime()))
    assert DiffSolver._check_load_provenance(
        _solver(_runtime(evaluator={'Allocation_Weights': _alloc((0, (0.5, 0.3, 0.2)))})),
        cartesian, '<ck>', 1) == cartesian['solver_version']


def test_a_moved_rate_is_refused_under_position_state():
    """Under `DiffV2_Position_State` the reposition charge enters the regressed TARGET, so a
    different `Calendar_Spread_Bps` is a different value function — refused, exactly as a
    `DiffV2_Position_State` mismatch is. Position-free it shapes SELECTION only, which is the
    deliberate execution re-roll `Max_Trade_Per_Step` also allows: warned, not refused."""
    trained = _artifact(_solver(_runtime(evaluator={'Calendar_Spread_Bps': CAL_BPS}),
                                position_state=True))
    moved = _runtime(evaluator={'Calendar_Spread_Bps': 2 * CAL_BPS})
    with pytest.raises(ValueError, match='Calendar_Spread_Bps mismatch'):
        DiffSolver._check_load_provenance(_solver(moved, position_state=True), trained,
                                          '<ck>', 1)
    free = _artifact(_solver(_runtime(evaluator={'Calendar_Spread_Bps': CAL_BPS})))
    assert DiffSolver._check_load_provenance(_solver(moved), free, '<ck>', 1) == \
        free['solver_version']


# --- (h) the ladder is what gets searched, and what gets reported -----------------------------
def test_the_reported_universe_never_builds_the_meshgrid():
    """`action_grid_size` is the row count of the universe the argmax SEARCHES. Under a schedule
    that is the ladder — and the Cartesian product must not even be materialized to say so: at 4
    legs it is 6.8M rows, at 5 it is ~345M and OOMs a run whose ladder is 51 rows.
    Killed by reporting `grid()`."""
    for n in (3, 4, 5):
        legs = tuple(f'M{i + 1}' for i in range(n))
        a = _aspace(_runtime(legs=legs, evaluator={
            'Allocation_Weights': [{'Step': 0, 'Instrument': x, 'Weight': 1.0 / n}
                                   for x in legs]}))
        assert a.universe_size() == 51
        assert a._grid_cache is None, 'the meshgrid must never be built under a schedule'


def test_the_hindsight_diagnostic_reports_the_ladder():
    """The clairvoyant track reads the universe size at its OWN seam, so it needs its own gate:
    under a schedule that number is the ladder's, and the meshgrid must not be built to produce
    it. Killed by reporting `grid()` there."""
    rt = _runtime(evaluator={'Allocation_Weights': _alloc((0, (0.5, 0.3, 0.2)))})
    solver = HindsightDpSolver(_bundle(rt, ([PRICE, PRICE - 100.0], [PRICE] * 2, [PRICE] * 2)), rt)
    assert solver.solve().diagnostics['action_grid_size'] == 51
    assert solver.aspace._grid_cache is None


def test_the_static_benchmark_sees_every_composition_the_schedule_offers():
    """The constant-hold benchmark is the campaign's headline comparison. Pinned to `grid_at(0)` it
    would search only the ENTRY knot's ray while the DP recomposes at every later one, putting an
    action-space difference inside the DP-vs-textbook gap. `static_grid` offers the hold every
    composition the schedule ever reaches — it still cannot TRACK the schedule, which is the honest
    asymmetry, but it may choose which composition to sit in."""
    a = _aspace(_runtime(evaluator={'Allocation_Weights': _alloc(
        (0, (1.0, 0.0, 0.0)), (5, (0.0, 0.0, 1.0)))}))
    static = a.static_grid()
    assert static.shape[0] > a.grid_at(0).shape[0]
    assert float(static[:, 0].abs().max()) > 0.0 and float(static[:, 2].abs().max()) > 0.0
    assert torch.equal(_aspace(_runtime(levels=5)).static_grid(),
                       _aspace(_runtime(levels=5)).grid_at(0))     # Cartesian: unchanged


def test_the_deck_refuses_a_fixing_past_the_listed_strip():
    """The producer of the schedules above. `allocation_knots` promises each fixing 'the first
    listed expiry ON OR AFTER it', and clamped the ones with no such expiry onto the last (by then
    dead) leg instead — a single knot putting 100% of the weight on a contract that expires
    mid-horizon, which is a hard corridor abort at best. The strip is the caller's to widen."""
    import pandas as pd
    from experiments.plat_walk_forward_chain import allocation_knots
    mats = [pd.Timestamp('2024-02-15'), pd.Timestamp('2024-03-15'), pd.Timestamp('2024-04-15')]
    trade = pd.Timestamp('2024-01-15')
    with pytest.raises(ValueError, match='no live contract'):
        allocation_knots(trade, pd.bdate_range('2024-04-01', '2024-04-30'), mats, list(LEGS))
    knots = allocation_knots(trade, pd.bdate_range('2024-02-01', '2024-04-10'), mats, list(LEGS))
    assert {r['Instrument'] for r in knots} == set(LEGS)
