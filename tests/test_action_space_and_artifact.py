"""Gates for the solver-consistency fixes:

  1. ACTION UNIVERSE — one shared `HedgeActionSpace` drives DiffSolverV2, the textbook /
     hindsight benchmarks AND the stepper rollout, so an `Active_Hedge_Indices` restriction
     pins the inactive legs to exactly zero across every track (the old benchmarks used the
     generic all-hedges grid and could trade a leg the greedy run had frozen).
  2. INITIAL INVENTORY — first-step turnover is measured from the OPENING book `q0`
     (normalized `Portfolio_State` positions), not from flat, in the frictionless value
     tracks' net-of-cost diagnostics and in the stepper rollout.
  3. POLICY ARTIFACT — a solver run returns ONE `value_fn_artifacts` dict, the same one
     `DiffV2_Save_Value_Fn` persists.
  4. CORRIDOR — a policy trained inside a `Total_Position_Schedule` rolls inside its fence,
     carries the corridor it was trained in, and loads only into that one; a corridor-free
     policy loads into any.
  5. PASSIVE — the verdict's and the stepper's `passive` book is minus the swap's unfixed
     quantity, split by `Allocation_Weights`, at the costs every other policy pays.

Configs are built in code from the canonical fixture TEMPLATE (never edited) — small batch /
shallow sweep so the gates run fast. Every run, training and loaded evaluation alike, is a
document through `load_json` + `run_job`.
"""
import ast
import json as jsonlib
import logging
import os
import re

import numpy as np
import pandas as pd
import pytest
import torch

import derivus as rf
from derivus.hedge_bundle import per_contract_kappa
from derivus.stochasticprocess import FILLED_ROWS

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'fixtures', 'policy_test_simulate_only.json')

# Fixture hedge order (names['hedges']): PL_APR_2026, PL_JUL_2026, PL_OCT_2026.
_HEDGES = ('PL_APR_2026', 'PL_JUL_2026', 'PL_OCT_2026')

_LISTED_ARTIFACT_KEYS = {
    'state_dicts', 'm_mean', 'm_std', 'w_mean', 'w_std', 'utility_scale', 'a_bounds',
    'hedges', 'active_hedge_indices', 'total_position_schedule', 'T_dec', 't_min', 'hidden',
    'solver_version', 'config_hash',
}

#: Short early, long late on symmetric [-50, 50] limits: the sign flips at Step 107, inside the
#: T_Min=100 sweep window.
SIGN_CROSSING = [{'Step': 0, 'Min_Total': -50, 'Max_Total': -25},
                 {'Step': 107, 'Min_Total': 25, 'Max_Total': 50}]


def _cfg(*, batch=48, inner=8, seed=7, positions=None, active=None,
         benchmarks=False, save=None, load=None, schedule=None, symmetric=False):
    """A training document - a fit batch and a held-out one - or, with `load`, the frozen eval of
    that checkpoint, its stream of length one; `schedule` fences it, on [-50, 50] limits and no
    gross cap where `symmetric`."""
    cfg = jsonlib.load(open(FIXTURE))
    calc = cfg['Calc']['Calculation']
    calc['Execution_Mode'] = 'solve_hedge'
    calc['Batch_Size'], calc['Simulation_Batches'] = batch // 2, 1 if load else 2
    calc['Inner_Sub_Batch'] = inner
    calc['Inner_MC_Enabled'] = 'Yes'
    calc['Inner_Antithetic'] = 'Yes'
    calc['Random_Seed'] = seed
    hp = calc['Hedging_Problem']
    hp['Randomize_Initial_State'] = 'Yes'
    if positions is not None:
        hp.setdefault('Portfolio_State', {})['Positions'] = positions
    ev = hp['Evaluator']
    if symmetric:
        for lim in ev['Position_Limits'].values():
            lim['Min_Position'], lim['Max_Position'] = -50, 50
        ev['Total_Position_Abs_Limit'] = 0.0
    if schedule is not None:
        ev['Total_Position_Schedule'] = schedule
    solver = {
        'Object': 'DiffSolverV2',
        'Training_Action_Grid_Levels_Per_Axis': 5,
        'Training_Action_Chunk_Size': 64,
        'T_Min': 100,                       # shallow ~16-step sweep (fast)
        'DiffV2_Fit_Iters': 5,
    }
    if active is not None:
        solver['Active_Hedge_Indices'] = list(active)
    if benchmarks:
        solver['Run_Textbook_Benchmark'] = 'Yes'
        solver['Run_Hindsight_Diagnostic'] = 'Yes'
    if save is not None:
        solver['DiffV2_Save_Value_Fn'] = save
    if load is not None:
        solver['DiffV2_Load_Value_Fn'] = list(load) if isinstance(load, (list, tuple)) else [load]
    hp['Solver'] = solver
    return cfg


def _run(cfg):
    cx = rf.Context()
    cx.load_json((jsonlib.dumps(cfg), 'gate.json'))
    _, result = cx.run_job()
    return result


def _v0(result):
    return result.evaluation_summary['diagnostics']['V_0']


@pytest.fixture(scope='module')
def free(tmp_path_factory):
    """One corridor-free training run, saved: `(checkpoint path, result)`."""
    ckpt = str(tmp_path_factory.mktemp('vf') / 'value_fn.pt')
    return ckpt, _run(_cfg(seed=7, save=ckpt))


# --------------------------------------------------------------------------------------------
# (i) active-mask consistency — inactive legs hold EXACTLY zero on every track
# --------------------------------------------------------------------------------------------
def test_active_mask_zeroes_inactive_legs_on_all_tracks():
    """Killing mutation: `axis_levels` spanning every axis whatever `Active_Hedge_Indices` says."""
    active = [2]                              # only PL_OCT_2026 varies; legs 0,1 pinned to 0
    result = _run(_cfg(active=active, benchmarks=True))
    es = result.evaluation_summary
    comp = es['comparison']
    diag = es['diagnostics']

    textbook_n = comp['textbook']['n_star']
    hindsight_n = comp['HindsightDpSolver']['n_star']
    greedy_absq = diag['verdict']['greedy_mean_abs_q']
    assert len(textbook_n) == len(hindsight_n) == len(greedy_absq) == 3

    for i in (0, 1):                          # the two INACTIVE legs
        assert textbook_n[i] == 0.0, f'textbook traded inactive leg {i}: {textbook_n}'
        assert hindsight_n[i] == 0.0, f'hindsight traded inactive leg {i}: {hindsight_n}'
        assert greedy_absq[i] == 0.0, f'greedy traded inactive leg {i}: {greedy_absq}'

    # The artifact records which axes were active — the frozen policy carries its mask.
    assert result.policy_artifact['active_hedge_indices'] == active


def test_the_hindsight_bound_decides_where_the_solver_does():
    """The clairvoyant bound decides on the solver's decision rows, each book held to the next: on
    the fixture with `PL_OCT_2026` alone active, its first book is per path the grid's best position
    on the realised move from the first row to the next market row, -50 where the move falls and 0
    where it rises, -29.17 contracts on average (14 of 24 paths short, the move from Friday 10 April
    to Monday). 3 s.

    Killing mutation: the bound stepping every calendar row (its first book read on the move to the
    next calendar row, Saturday's, -33.33)."""
    result = _run(_cfg(active=[2], benchmarks=True))
    world = result.bundle
    fut = world.tradables_sim['PL_OCT_2026']
    move = fut[min(t for t in _decisions(world) if t > 0)] - fut[0]
    want = [0.0, 0.0, -50.0 * float((move < 0).double().mean())]
    assert result.evaluation_summary['comparison']['HindsightDpSolver']['n_star'] == pytest.approx(want, abs=1e-5)


def test_the_textbook_bound_holds_its_book_to_the_terminal():
    """The static-hold benchmark telescopes its constant book to the terminal T, the last live
    liability mark where `L_T` is read and the book's unwind is charged: on the fixture with
    `PL_OCT_2026` alone active under a CARA utility of gamma 1 at an explicit 100,000 scale, its
    `v0_mean` is the path mean of u(L_T + n* x 50 x (F_T - F_0)) at its own n*, within 1e-5
    relative (-2,386.49). 3 s.

    Killing mutation: the hold read to the clean-exit row after the terminal (`F[:, -1]`, Thursday
    6 August past Wednesday 5), the row the stepper and the hindsight bound were moved off: n* -25
    reads -2,744.35."""
    cfg = _cfg(active=[2], benchmarks=True)
    cfg['Calc']['Calculation']['Hedging_Problem']['Objective'].update(
        Object='AsymmetricUtility_CARA', CARA_Gamma=1.0, Utility_Scale_Explicit=100000.0)
    result = _run(cfg)
    world = result.bundle
    T = world.last_live_mtm_index
    F = world.tradables_sim['PL_OCT_2026'].double()
    L = world.liability_sim.double()
    textbook = result.evaluation_summary['comparison']['textbook']
    n = textbook['n_star'][2]
    want = float((-torch.expm1(-(L[T] + n * 50.0 * (F[T] - F[0])) / 100000.0)).mean())
    beyond = float((-torch.expm1(-(L[T] + n * 50.0 * (F[-1] - F[0])) / 100000.0)).mean())
    assert n != 0.0 and T == F.shape[0] - 2, (n, T, F.shape)
    assert abs(textbook['v0_mean'] / want - 1.0) < 1e-5, (textbook['v0_mean'], want, beyond)


# --------------------------------------------------------------------------------------------
# (ii) initial inventory — first-step turnover measured from the OPENING book q0, not flat
# --------------------------------------------------------------------------------------------
def test_first_step_turnover_measured_from_opening_book():
    """Killing mutations: the textbook's entry charged from flat, `n_star - 0` for `n_star - q0`;
    the verdict's wealth started at the liability's mark at its first decision (`L[t0]`), the
    no-hedge track's terminal wealth unmoved by the opening book's 9,164.21 to that decision."""
    positions = {'PL_APR_2026': -1, 'PL_JUL_2026': -4, 'PL_OCT_2026': 0}
    flat = _run(_cfg(seed=11, positions=None, benchmarks=True))
    book = _run(_cfg(seed=11, positions=positions, benchmarks=True))

    fb, bb = flat.evaluation_summary, book.evaluation_summary

    # The value function is POSITION-FREE: the opening book must NOT move any V_0 track...
    assert bb['comparison']['textbook']['v0_mean'] == fb['comparison']['textbook']['v0_mean']
    assert bb['comparison']['HindsightDpSolver']['v0_mean'] \
        == fb['comparison']['HindsightDpSolver']['v0_mean']
    # ...while the verdict holds it from the first row to its first decision, as the stepper roll
    # does, so the no-hedge track's terminal wealth moves by exactly the book's P&L there
    runtime, bundle = book.runtime, book.bundle
    hist, t0 = bundle.initial_time_index, bb['diagnostics']['root_t']
    carried = sum(positions[h] * float(runtime['tradables'][h]['contract_size'])
                  * float((bundle.tradables[h][hist + t0] - bundle.tradables[h][hist]).double().mean())
                  for h in _HEDGES)
    moved = (bb['diagnostics']['verdict']['nohedge']['wT_mean']
             - fb['diagnostics']['verdict']['nohedge']['wT_mean'])
    assert abs(moved - carried) < 1e-3 * (abs(carried) + 1.0), (moved, carried)

    # ...but the NET-OF-COST turnover DOES shift, because entry is now |q_target − q0|.
    tb_flat = fb['comparison']['textbook']['turnover_cost_mean']
    tb_book = bb['comparison']['textbook']['turnover_cost_mean']
    g_flat = fb['diagnostics']['verdict']['greedy']['turnover_cost_mean']
    g_book = bb['diagnostics']['verdict']['greedy']['turnover_cost_mean']
    assert tb_book != tb_flat, 'textbook entry turnover ignored the opening book'
    assert g_book != g_flat, '_verdict entry turnover ignored the opening book'

    # The frictionless argmax is position-independent, so both runs pick the SAME constant
    # hold n_star and the SAME kappa0 (identical bundle). The ONLY thing that changed is the
    # ENTRY term: |n_star − q0| (book) vs |n_star − 0| (flat). So the whole turnover
    # difference must equal Σ_i (|n_star_i − q0_i| − |n_star_i|) · kappa0_i — the crisp
    # signature of "first-step turnover measured from q0, not from flat".
    n_star = bb['comparison']['textbook']['n_star']
    assert fb['comparison']['textbook']['n_star'] == n_star     # position-free selection
    expected_diff = sum(
        (abs(n_star[i] - float(positions[h])) - abs(n_star[i]))
        * float(per_contract_kappa(runtime, bundle.tradables[h][hist:][0].mean(), h))
        for i, h in enumerate(_HEDGES))
    assert abs((tb_book - tb_flat) - expected_diff) < 1e-3 * (abs(expected_diff) + 1.0), \
        f'textbook turnover shift {tb_book - tb_flat} != Σ(|n*−q0|−|n*|)·kappa0 {expected_diff}'

    # The stepper OPENS from q0 (its opening state seeds Portfolio_State), so its realized
    # first-step trade is measured from the same book — the deployment convention agrees.
    stepper = book.create_stepper()
    opening = stepper.observe()['positions']
    for h in _HEDGES:
        assert float(opening[h][0]) == float(positions[h]), \
            f'stepper did not open from the book at {h}'
    while not stepper.is_decision_step:
        stepper.step(None)
    q_target = -10.0
    pre = float(stepper.observe()['positions']['PL_JUL_2026'][0])
    assert pre == float(positions['PL_JUL_2026'])          # still opening from the book at t0
    stepper.step({'PL_JUL_2026': q_target - pre})          # trade the verdict delta q_target − q0
    post = float(stepper.observe()['positions']['PL_JUL_2026'][0])
    assert round(post) == round(q_target)                  # reached the target
    # the realized first-step trade the stepper charged turnover on is q_target − q0 (the
    # verdict convention), NOT q_target − 0.
    assert round(post - pre) == round(q_target - float(positions['PL_JUL_2026']))


# --------------------------------------------------------------------------------------------
# (iii) artifact contract — the run returns the artifact, and the file is that dict
# --------------------------------------------------------------------------------------------
def test_the_returned_artifact_is_the_file_the_run_saved(free):
    """The returned artifact carries every listed key, and the checkpoint `DiffV2_Save_Value_Fn`
    wrote is that dict, every value equal, the fitted weights to the bit. A loaded eval reporting
    the file's own V_0 is `test_frozen_policy_is_frozen`'s.

    Killing mutation: a listed key dropped from `_policy_artifact` (`solver_version`)."""
    ckpt, train = free
    artifact = train.policy_artifact
    assert isinstance(artifact, dict)
    missing = _LISTED_ARTIFACT_KEYS - set(artifact)
    assert not missing, f'artifact missing listed keys: {missing}'
    assert artifact['solver_version'] and isinstance(artifact['config_hash'], str)

    def same(a, b):
        if torch.is_tensor(a):
            return torch.is_tensor(b) and torch.equal(a.cpu(), b.cpu())
        if isinstance(a, dict):
            return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
        if isinstance(a, (list, tuple)):
            return isinstance(b, (list, tuple)) and len(a) == len(b) and all(map(same, a, b))
        return a == b

    file_dict = torch.load(ckpt, map_location='cpu', weights_only=False)
    assert set(file_dict) == set(artifact)
    for k in artifact:
        assert same(file_dict[k], artifact[k]), f'file/artifact disagree on {k!r}'


def test_the_saved_checkpoint_carries_the_drift_null_the_verdict_measured(free):
    """The policy artifact is built after the training verdict, so the checkpoint carries the
    drift monitor's null sigma that verdict measured, and a frozen roll's tripwire is scaled by it.

    Killing mutation: the artifact built and saved before `_verdict`, so the stamp is None."""
    ckpt, train = free
    measured = train.evaluation_summary['diagnostics']['verdict']['drift_null_sigma']
    assert measured > 0.0, measured
    stamped = torch.load(ckpt, map_location='cpu', weights_only=False)['drift_null_sigma']
    assert stamped is not None and round(stamped, 4) == measured, (stamped, measured)


# --------------------------------------------------------------------------------------------
# (iv) corridor — trained inside a fence, rolled inside it, loaded only into it
# --------------------------------------------------------------------------------------------
def _corridor_at(sched, t):
    lo, hi = sched[0]['Min_Total'], sched[0]['Max_Total']
    for k in sched:
        if k['Step'] > t:
            break
        lo, hi = k['Min_Total'], k['Max_Total']
    return lo, hi


def test_a_corridor_trained_policy_rolls_in_its_fence_and_loads_only_into_it(tmp_path):
    """Trained inside a SIGN-CROSSING `Total_Position_Schedule`: the held-out greedy book sums
    inside the fence at every decision step, short early and long late; the artifact stamps that schedule;
    a frozen eval in the same corridor reports the trained V_0 to the bit, and one under another
    corridor or none is refused by name - the frozen value fn learned other reachable wealth.

    Killing mutations: the corridor filter skipped in `grid_at` (the greedy book breaches it); the
    stamp written as None; a mismatched corridor admitted as if trained corridor-free."""
    ckpt = str(tmp_path / 'value_fn_corr.pt')
    train = _run(_cfg(seed=7, save=ckpt, schedule=SIGN_CROSSING, symmetric=True))
    diag = train.evaluation_summary['diagnostics']
    t0 = int(diag['root_t'])
    saw_short = saw_long = False
    decisions = {r['t'] for r in diag['per_t']}
    for i, book in enumerate(diag['verdict']['greedy_q_traj']):
        if t0 + i not in decisions:
            continue                     # the book a decision holds over a row of no market time
        lo, hi = _corridor_at(SIGN_CROSSING, t0 + i)
        assert lo - 1e-3 <= sum(book) <= hi + 1e-3, (t0 + i, sum(book), lo, hi)
        saw_short, saw_long = saw_short or hi < 0, saw_long or lo > 0
    assert saw_short and saw_long, 'the sweep window did not cover both corridor signs'
    assert train.policy_artifact['total_position_schedule'] == (
        (0, -50.0, -25.0), (107, 25.0, 50.0))

    assert _v0(_run(_cfg(seed=7, load=ckpt, schedule=SIGN_CROSSING, symmetric=True))) == \
        _v0(train)
    for other in ([{'Step': 0, 'Min_Total': -40, 'Max_Total': -20}], None):
        with pytest.raises(ValueError, match='corridor mismatch'):
            _run(_cfg(seed=7, load=ckpt, schedule=other, symmetric=True))


def test_a_corridor_trained_policy_rolls_inside_its_own_corridor(tmp_path):
    """A fence inside the trained corridor on every decision step only restricts the roll to learned
    support, so the frozen policy rolls in it and its greedy book sums inside the narrower fence;
    a fence wider on either side is still refused by name.

    Killing mutation: the containment branch removed from the corridor check (refused as a mismatch)."""
    ckpt = str(tmp_path / 'value_fn_inside.pt')
    _run(_passive_cfg(save=ckpt))                       # trained in the flat [-40, 5] corridor
    inside = [{'Step': 0, 'Min_Total': -30, 'Max_Total': 0}]
    cfg = _passive_cfg(load=ckpt)
    cfg['Calc']['Calculation']['Hedging_Problem']['Evaluator']['Total_Position_Schedule'] = inside
    sv = _run(cfg).evaluation_summary['diagnostics']['stepper_verdict']
    assert all(-30 - 1e-3 <= sum(book) <= 0 + 1e-3 for book in sv['greedy_q_traj']), sv['greedy_q_traj'][:3]
    wider = [{'Step': 0, 'Min_Total': -45, 'Max_Total': 5}]
    cfg = _passive_cfg(load=ckpt)
    cfg['Calc']['Calculation']['Hedging_Problem']['Evaluator']['Total_Position_Schedule'] = wider
    with pytest.raises(ValueError, match='corridor mismatch'):
        _run(cfg)


def test_a_corridor_free_policy_rolls_in_any_corridor(free):
    """A policy trained corridor-FREE has the widest wealth support, so rolling it INSIDE a
    Total_Position_Schedule only restricts to a learned subset - allowed, and the eval reports
    the trained V_0.

    Killing mutation: the corridor-free arm refused like a mismatch."""
    ckpt, train = free
    assert train.policy_artifact['total_position_schedule'] is None
    fence = [{'Step': 0, 'Min_Total': -50, 'Max_Total': -20}]
    assert _v0(_run(_cfg(seed=7, load=ckpt, schedule=fence))) == _v0(train)


# --------------------------------------------------------------------------------------------
# (v) passive — minus the swap's unfixed quantity, through the same wealth step and costs
# --------------------------------------------------------------------------------------------
#: Sixteen May fixings of a swap bought on 2,400 oz and sold on 800: a net 100 oz a print.
FIXINGS = ('2026-05-01', '2026-05-04', '2026-05-05', '2026-05-06', '2026-05-07', '2026-05-08',
           '2026-05-11', '2026-05-12', '2026-05-13', '2026-05-14', '2026-05-15', '2026-05-18',
           '2026-05-19', '2026-05-20', '2026-05-21', '2026-05-22')
SWAP_LEGS = (('Buy', 2400.0, 2040.0), ('Sell', 800.0, 2060.0))
LEGS = ('PL_JUL_2026', 'PL_OCT_2026')
#: Half and half, then from Step 25 a quarter and three quarters.
KNOTS = ((0, (0.5, 0.5)), (25, (0.25, 0.75)))
PASSIVE_T_MIN = 15


def _decisions(world):
    """The sim-grid rows a solver decides on, read off the job's calendar: the first row and every
    market row before the terminal, the last live liability mark."""
    r0 = world.initial_time_index
    return {0} | {t for t in range(world.last_live_mtm_index)
                  if world.business_day.is_on_offset(world.scenario_dates[r0 + t])}


def _whole(q):
    """The book `q` in whole contracts carrying its rounded total: each leg floored, the units left
    over handed one each to the legs with the largest remainders, ties to the first."""
    base = q.floor()
    rest = (q - base).tolist()
    given = sorted(range(len(rest)), key=lambda i: -rest[i])[:round(float(q.sum() - base.sum()))]
    return base + torch.tensor([float(i in given) for i in range(len(rest))], device=q.device)


def _passive_cfg(save=None, load=None):
    """The swap hedged on the July and October legs in a flat corridor the passive book sits
    inside; with `load`, the frozen eval of that checkpoint with the stepper rollout on."""
    cfg = _cfg(batch=24, seed=5, save=save, load=load,
               schedule=[{'Step': 0, 'Min_Total': -40, 'Max_Total': 5}])
    hp = cfg['Calc']['Calculation']['Hedging_Problem']
    hp['Liabilities'] = {'CommodityAveragePriceSwapDeal': {
        f'PLAT_APS_{side.upper()}': {
            'Currency': 'USD', 'Commodity': 'PLATINUM_LME.LME_CME', 'Carry': 'PLATINUM_CARRY',
            'Discount_Rate': 'USD-SOFR', 'Buy_Sell': side, 'Units': units, 'Fixed_Price': strike,
            'Settlement_Date': {'.Timestamp': '2026-05-27'},
            'Sampling_Data': [[{'.Timestamp': d}, 0.0, 1.0] for d in FIXINGS]}
        for side, units, strike in SWAP_LEGS}}
    del hp['Tradable_Instruments']['CommodityFutureDeal']['PL_APR_2026']
    hp['Tradable_Instruments']['CashAccountDeal']['USD_CASH']['Investment_Horizon'] = {
        '.Timestamp': '2026-05-27'}
    for key in ('Positions', 'Settlement_Prices', 'Initial_Margin'):
        del hp['Portfolio_State'][key]['PL_APR_2026']
    ev = hp['Evaluator']
    ev['Position_Limits'] = {h: {'Min_Position': -20, 'Max_Position': 5} for h in LEGS}
    ev['Total_Position_Abs_Limit'] = 40
    ev['Allocation_Weights'] = [{'Step': s, 'Instrument': h, 'Weight': w}
                                for s, ws in KNOTS for h, w in zip(LEGS, ws)]
    hp['Solver'].update({'Training_Action_Grid_Levels_Per_Axis': 46, 'T_Min': PASSIVE_T_MIN})
    if load is not None:
        hp['Solver']['DiffV2_Stepper_Rollout'] = 'Yes'
    return cfg


def test_the_passive_book_is_the_swaps_unfixed_quantity_at_the_declared_cost(tmp_path):
    """The passive book is short the swap's net unfixed ounces, the bought leg less the sold,
    sixteen contracts a leg before the first fixing and two fewer at each print, split by the
    knot in force in whole contracts and inside the corridor, so no projection moves it. The
    verdict's terminal wealth is that book's P&L by the wealth law, the sum over decisions of
    position times contract size times the futures move to the next decision plus the liability's
    move, less the declared cost (half the 10 bp spread on the path-mean mark per contract traded,
    entry from the opening book and the unwind at the horizon), within 1e-10 relative; the frozen
    eval's stepper rollout is that book driven through `create_stepper()`, to the bit.

    Killing mutations, the verdict's net and the stepper's mean against the book's 120124.68 and
    -8480.32: the ramp read one fixing late (`fixing_day >= day` in `Bundle.unfixed_at`),
    114876.18 and -5603.66; the side dropped (`Buy_Sell` unread in `leg_descriptors`), the book
    clamped at -20 a leg, 110007.23 and 9730.19; the passive's turnover not charged in
    `_verdict`, 124265.91; the stepper roll's passive branch dropped, rolling the greedy argmax,
    142344.35."""
    ckpt = str(tmp_path / 'value_fn_passive.pt')
    cfg = _passive_cfg(save=ckpt)
    hp = cfg['Calc']['Calculation']['Hedging_Problem']
    sizes = [float(hp['Tradable_Instruments']['CommodityFutureDeal'][h]['Contract_Size'])
             for h in LEGS]
    net_units = sum((1.0 if side == 'Buy' else -1.0) * units for side, units, _ in SWAP_LEGS)

    def book(world, t):
        date = world.scenario_dates[world.initial_time_index + t]
        unfixed = net_units / len(FIXINGS) * sum(pd.Timestamp(d) > date for d in FIXINGS)
        weights = [ws for step, ws in KNOTS if step <= t][-1]
        return _whole(-unfixed * torch.tensor(weights, device=world.device)
                      / torch.tensor(sizes, device=world.device))

    train = _run(cfg)
    v, world = train.evaluation_summary['diagnostics']['verdict'], train.bundle
    F, L, horizon = world.tradables_sim, world.liability_sim, world.last_live_mtm_index
    decisions = _decisions(world)
    t0 = min(t for t in decisions if t >= PASSIVE_T_MIN)

    def kappa(t):
        return torch.stack([0.5 * hp['Evaluator']['Bid_Offer_Spread_Bps'] * 1e-4 * F[h][t].mean()
                            * size for h, size in zip(LEGS, sizes)])

    W, q_prev, books = L[t0].clone(), torch.zeros(len(LEGS), device=world.device), []
    cost = torch.zeros_like(W)
    steps = sorted(t for t in decisions if t >= t0) + [horizon]
    for t, t1 in zip(steps, steps[1:]):
        q = book(world, t)
        books.extend([q.tolist()] * (t1 - t))           # held over the rows to the next decision
        cost = cost + ((q - q_prev).abs() * kappa(t)).sum(-1)
        dF = torch.stack([F[h][t1] - F[h][t] for h in LEGS], dim=-1)
        W = W + (q * torch.tensor(sizes, device=world.device) * dF).sum(-1) + (L[t1] - L[t])
        q_prev = q
    cost = cost + (q_prev.abs() * kappa(horizon)).sum(-1)
    assert books[0] == [-16.0, -16.0] and books[-1] == [0.0, 0.0]
    assert v['passive_q_traj'] == books, 'the passive book is not the unfixed quantity'
    net = float((W - cost).mean())
    assert abs(v['passive']['wT_mean_net'] - net) <= 1e-10 * abs(net), \
        (v['passive']['wT_mean_net'], net)
    assert abs(v['passive']['turnover_cost_mean'] - float(cost.mean())) \
        <= 1e-10 * float(cost.mean())

    frozen = _run(_passive_cfg(load=ckpt))
    stepper, hist = frozen.create_stepper(), frozen.bundle.initial_time_index
    while not stepper.done:
        t = stepper.time_index - hist
        if stepper.is_decision_step and PASSIVE_T_MIN <= t < frozen.bundle.last_live_mtm_index:
            q, held = book(frozen.bundle, t), stepper.observe()['positions']
            last = stepper.step({h: q[i] - held[h] for i, h in enumerate(LEGS)})
        else:
            last = stepper.step(None)
    wT = (last['transition_pnl_excess'] + last['transition_liability_value']).to(torch.float64)
    assert frozen.evaluation_summary['diagnostics']['stepper_verdict']['passive']['wT_mean'] \
        == float(wT.mean())


@pytest.mark.parametrize('churn', [0.0, 50.0])
def test_the_position_state_verdict_charges_the_passive_book_inside_its_wealth(churn):
    """The swap document's passive track under `DiffV2_Position_State`, `DiffV2_Churn_Lambda` 0 and
    50: its terminal wealth `wT_mean` is the hand-rolled book's P&L less half the 10 bp spread on the
    path-mean mark per contract traded (entry from the opening book, the unwind at the horizon) and
    the churn term lambda * sum |dq|^2, within 1e-6 relative; `turnover_cost_mean` is that charge;
    no `wT_mean_net` is written. 5 s each.

    Killing mutations: the charge accounted beside the recursion (wT gross by the cost); the unwind
    dropped at the last decision (wT above by the unwind); the churn term left out of the wealth."""
    cfg = _passive_cfg()
    hp = cfg['Calc']['Calculation']['Hedging_Problem']
    hp['Solver']['DiffV2_Position_State'] = 'Yes'
    hp['Solver']['DiffV2_Churn_Lambda'] = churn
    sizes = [float(hp['Tradable_Instruments']['CommodityFutureDeal'][h]['Contract_Size']) for h in LEGS]
    net_units = sum((1.0 if side == 'Buy' else -1.0) * units for side, units, _ in SWAP_LEGS)
    train = _run(cfg)
    v, world = train.evaluation_summary['diagnostics']['verdict'], train.bundle
    F, L, horizon = world.tradables_sim, world.liability_sim, world.last_live_mtm_index
    decisions = _decisions(world)
    t0 = min(t for t in decisions if t >= PASSIVE_T_MIN)

    def book(t):
        date = world.scenario_dates[world.initial_time_index + t]
        unfixed = net_units / len(FIXINGS) * sum(pd.Timestamp(d) > date for d in FIXINGS)
        weights = [ws for step, ws in KNOTS if step <= t][-1]
        return _whole(-unfixed * torch.tensor(weights, device=world.device) / torch.tensor(sizes, device=world.device))

    def kappa(t):
        return torch.stack([0.5 * hp['Evaluator']['Bid_Offer_Spread_Bps'] * 1e-4 * F[h][t].mean() * size
                            for h, size in zip(LEGS, sizes)])

    W, q_prev = L[t0].clone(), torch.zeros(len(LEGS), device=world.device)
    cost = torch.zeros_like(W)
    steps = sorted(t for t in decisions if t >= t0) + [horizon]
    for t, t1 in zip(steps, steps[1:]):
        q = book(t)
        cost = cost + ((q - q_prev).abs() * kappa(t)).sum(-1) + churn * ((q - q_prev) ** 2).sum(-1)
        dF = torch.stack([F[h][t1] - F[h][t] for h in LEGS], dim=-1)
        W = W + (q * torch.tensor(sizes, device=world.device) * dF).sum(-1) + (L[t1] - L[t])
        q_prev = q
    cost = cost + (q_prev.abs() * kappa(horizon)).sum(-1)
    net = float((W - cost).mean())
    assert 'wT_mean_net' not in v['passive']
    assert abs(v['passive']['wT_mean'] - net) <= 1e-6 * abs(net), (v['passive']['wT_mean'], net, float(W.mean()))
    assert abs(v['passive']['turnover_cost_mean'] - float(cost.mean())) <= 1e-6 * float(cost.mean()), \
        (v['passive']['turnover_cost_mean'], float(cost.mean()))


@pytest.mark.parametrize('units, corridor, limits, weights', [
    (2425.0, (-40, -32.3), (-20, 5), KNOTS), (4800.0, (-60, 5), (-30, 5), KNOTS),
    (2400.0, (2, 5), (-20, 5), KNOTS), (2400.0, (-40, -35), (-50, 5), ((0, (1.0, 0.0)),))],
    ids=['fractional', 'cap', 'long_only', 'one_leg'])
def test_the_passive_book_stays_inside_a_fractional_corridor(units, corridor, limits, weights):
    """The swap's passive book at the edges of its band, on the verdict of its training document:
    on every row in whole contracts, each leg inside its box and nothing on a leg the
    `Allocation_Weights` knot weighs zero, its signed net inside the corridor and the cap of 40.
    Bought on 2,425 oz, a net 32.5 contracts unfixed before the first fixing, in a corridor whose
    upper edge is -32.3 the book sums to -33, the corridor's whole edge, where the unfixed quantity
    would round to -32; bought on 4,800 oz, 80 contracts, the cap binds before the corridor; a
    long-only corridor [2, 5] holds the book on its floor against the short the swap wants; the July
    leg weighted alone in a corridor [-40, -35] holds the whole deficit. 2 s each.

    Killing mutations: the net water-filled into the fractional corridor (`net_bounds` in
    `HedgeActionSpace.book`, not the whole-contract `band`) and apportioned to its rounded total,
    -32 on all 30 rows, outside the corridor; `book` without its water-fill, the clamped split
    apportioned as it stands, (-30, -30) past the cap, (-16, -16) below the long-only floor and
    (-32, 0) outside the one-leg corridor; the deficit water-filled onto every live active leg
    (`book` on `active_mask * live`), the October leg -2 contracts on the one-leg document."""
    cfg = _passive_cfg()
    hp = cfg['Calc']['Calculation']['Hedging_Problem']
    hp['Liabilities']['CommodityAveragePriceSwapDeal']['PLAT_APS_BUY']['Units'] = units
    ev = hp['Evaluator']
    ev['Total_Position_Schedule'] = [{'Step': 0, 'Min_Total': corridor[0], 'Max_Total': corridor[1]}]
    ev['Position_Limits'] = {h: {'Min_Position': limits[0], 'Max_Position': limits[1]} for h in LEGS}
    ev['Allocation_Weights'] = [{'Step': s, 'Instrument': h, 'Weight': w} for s, ws in weights for h, w in zip(LEGS, ws)]
    books = _run(cfg).evaluation_summary['diagnostics']['verdict']['passive_q_traj']
    weighed = [[w > 0.0 for w in ws] for _, ws in weights]
    bad = [q for q in books if any(x != round(x) or not limits[0] <= x <= limits[1] for x in q)
           or not corridor[0] <= sum(q) <= corridor[1] or abs(sum(q)) > ev['Total_Position_Abs_Limit']
           or (len(weighed) == 1 and any(x and not w for x, w in zip(q, weighed[0])))]
    assert len(books) == 30 and not bad, bad[:3]


def test_the_stepper_hands_the_greedy_book_to_the_passive_from_the_declared_day(tmp_path):
    """`DiffV2_Passive_From` hands the stepper roll's greedy policy to the passive book from the
    first decision day on or after the date: on the same decision days, the greedy book is the
    plain roll's before that day and the passive book from it, and a handover from the first
    decision day rolls the passive's terminal wealth to the bit. The swap world's passive book is
    not flat, so the handover cannot read as a flat book; its scenario carries a price history, so
    the recorded decision dates are on or after the base date only when they are read on the
    full grid.

    Killing mutations: the handover branch removed from `_rollout_on_stepper`, so the greedy roll
    ignores the field; the decision date indexed by the sim step alone, so the dates are recorded
    a history prefix early."""
    ckpt = str(tmp_path / 'value_fn_handover.pt')
    _run(_passive_cfg(save=ckpt))

    def roll(passive_from=None):
        cfg = _passive_cfg(load=ckpt)
        if passive_from is not None:
            cfg['Calc']['Calculation']['Hedging_Problem']['Solver'][
                'DiffV2_Passive_From'] = passive_from
        diag = _run(cfg).evaluation_summary['diagnostics']
        sv = diag['stepper_verdict']
        passive = [diag['verdict']['passive_q_traj'][t - diag['root_t']] for t in sv['greedy_q_t']]
        return sv, passive

    plain, passive = roll()
    dates = plain['greedy_q_dates']
    assert len(dates) >= 3, dates
    base = _passive_cfg()['Calc']['Calculation']['Base_Date']['.Timestamp']
    assert dates[0] >= base, (dates[0], base)
    k = len(dates) // 2
    mid, _ = roll(dates[k])
    first, _ = roll(dates[0])
    assert mid['greedy_q_dates'] == first['greedy_q_dates'] == dates
    assert mid['greedy_q_traj'][:k] == plain['greedy_q_traj'][:k]
    assert mid['greedy_q_traj'][k:] == passive[k:]
    assert plain['greedy_q_traj'][k:] != passive[k:], 'the greedy roll already held the passive'
    assert first['greedy']['wT_mean'] == first['passive']['wT_mean'], \
        (first['greedy']['wT_mean'], first['passive']['wT_mean'])


def test_the_first_decision_can_be_measured_from_the_passive_book(tmp_path):
    """`DiffV2_First_Decision_From` 'Passive' seats the stepper roll's first greedy decision on the
    passive book at that step, so the trade band holds the first book within a band of the passive
    rather than of the empty opening book; the accounting is untouched, every track still entering
    from the opening book at that day's price, so the passive track's wealth is the same either way.

    Killing mutation: the first decision's `q_prev` left at the opening book under 'Passive'."""
    ckpt = str(tmp_path / 'value_fn_first.pt')
    _run(_passive_cfg(save=ckpt))

    def roll(first_from):
        cfg = _passive_cfg(load=ckpt)
        hp = cfg['Calc']['Calculation']['Hedging_Problem']
        hp['Evaluator']['Max_Trade_Per_Step'] = 2.0
        hp['Solver']['DiffV2_First_Decision_From'] = first_from
        diag = _run(cfg).evaluation_summary['diagnostics']
        sv = diag['stepper_verdict']
        passive0 = diag['verdict']['passive_q_traj'][sv['greedy_q_t'][0] - diag['root_t']]
        return sv, passive0

    book, passive0 = roll('Opening_Book')
    seated, passive0_seated = roll('Passive')
    assert passive0 == passive0_seated and max(abs(q) for q in passive0) > 2.0, passive0
    assert all(abs(q) <= 2.0 + 1e-6 for q in book['greedy_q_traj'][0]), book['greedy_q_traj'][0]
    assert all(abs(q - p) <= 2.0 + 1e-6 for q, p in zip(seated['greedy_q_traj'][0], passive0)), \
        (seated['greedy_q_traj'][0], passive0)
    assert seated['greedy_q_dates'] == book['greedy_q_dates']
    assert seated['passive']['wT_mean'] == book['passive']['wT_mean']


def test_the_returns_state_fits_the_wealth_column_it_reads():
    """Under `DiffV2_Returns_State` the value state is dimensionless: price columns as log-returns
    against the t0 spot, the basis as a fraction of it and wealth as a fraction of the t0 book
    notional, in one spelling (`_w_norm`) on every path a net is fitted or read on. A training
    document under the scale-free LogWealth objective with the returns state on then fits its
    labels: the fitted residual tracks the bootstrap label from the first backward step, the
    sweep's `V_0` reads 0.10793102532625198 on the CPU (0.0859 on a card: the twelve-path,
    five-iteration fit is device-rounded) and the same document at twice every dollar trains
    to the same `V_0` within 1e-6, since the frame is dimensionless.

    Killing mutation: the fit loop and the ensemble path feeding wealth in dollars to the
    fraction frame (the three spellings this gate retired) - the net sees a column a million
    standard deviations wide, fits a residual an order of magnitude too small at the first step,
    the sweep reads V_0 0.01840450055897236 (0.0162 on a card) and the doubled document no
    longer trains alike."""
    cfg = _cfg(seed=7)
    cfg['Calc']['Calculation']['Deterministic_Kernels'] = 'Yes'     # the two fits repeat to the bit on a card
    hp = cfg['Calc']['Calculation']['Hedging_Problem']
    hp['Objective'] = {'Object': 'LogWealth'}
    hp['Solver']['DiffV2_Returns_State'] = 'Yes'
    v0 = _v0(_run(cfg))
    assert 0.05 < v0 < 0.3 and abs(_v0(_run(_doubled(cfg))) - v0) <= 1e-6 * v0, v0


def _logwealth(cfg):
    """`cfg` under the LogWealth objective. The fixture's twelve-path, five-iteration fit ruins its
    held-out batch under the growth reward (V_0 0.07 to 0.17 nats against a held-out greedy utility
    of -14 to -94), so the gates using it, which hold frames rather than convergence, declare the
    optimism bound past that."""
    hp = cfg['Calc']['Calculation']['Hedging_Problem']
    hp['Objective'] = {'Object': 'LogWealth'}
    hp['Solver']['DiffV2_Max_Value_Optimism'] = 1.0e3
    return cfg


def _doubled(cfg):
    """Every dollar of the document doubled, nothing else: the liability's volume, each future's
    contract size, the margins and the cash; prices, paths and contract counts unchanged."""
    cfg = jsonlib.loads(jsonlib.dumps(cfg))
    hp = cfg['Calc']['Calculation']['Hedging_Problem']
    for deal in hp['Liabilities']['FloatingEnergyDeal'].values():
        for item in deal['Payments']['Items']:
            item['Volume'] *= 2
    for fut in hp['Tradable_Instruments']['CommodityFutureDeal'].values():
        fut['Contract_Size'] *= 2
    ps = hp['Portfolio_State']
    for im in ps['Initial_Margin'].values():
        im['Amount'] *= 2
    for key in ('Cash_Balances', 'Margin_Balances'):
        for k in ps.get(key, {}):
            ps[key][k] *= 2
    return cfg


def test_the_capital_line_travels_with_the_returns_state_frame(tmp_path):
    """Under the returns state the value state is in fractions of the t0 notional, and so is the
    capital line the day's growth is read against: the checkpoint stamps its training notional and
    a roll scales the line by its own notional over it, so the same document at twice every
    dollar rolls the same book with exactly twice the wealth.

    Killing mutation: the displacement restored in the training world's dollars."""
    ckpt = str(tmp_path / 'value_fn_frame.pt')
    cfg = _logwealth(_cfg(seed=7, save=ckpt))
    cfg['Calc']['Calculation']['Hedging_Problem']['Solver']['DiffV2_Returns_State'] = 'Yes'
    _run(cfg)
    roll = _logwealth(_cfg(seed=7, load=ckpt))
    roll['Calc']['Calculation']['Hedging_Problem']['Solver']['DiffV2_Returns_State'] = 'Yes'
    base = _run(roll).evaluation_summary['diagnostics']['verdict']
    twice = _run(_doubled(roll)).evaluation_summary['diagnostics']['verdict']
    assert twice['greedy_q_traj'] == base['greedy_q_traj'], \
        [(a, b) for a, b in zip(base['greedy_q_traj'], twice['greedy_q_traj']) if a != b][:3]
    assert abs(twice['greedy']['wT_mean'] - 2.0 * base['greedy']['wT_mean']) <= 1e-6 * abs(base['greedy']['wT_mean']), \
        (base['greedy']['wT_mean'], twice['greedy']['wT_mean'])


def test_an_ensemble_scores_each_member_in_its_own_frame(tmp_path):
    """Two LogWealth checkpoints of one document trained under different seeds carry different
    capital lines (each seed's displacement is measured off its own bank), and a two-member roll
    reads the day's reward as each member's log-growth at its own line, averaged: the frozen
    ensemble rolls the same greedy book and the same V_0 whichever checkpoint is listed first, and
    its drift tripwire's null is the members' average, the same in either order.

    Killing mutation: the reward read at the FIRST member's capital line for both members, so
    the roll depends on the order of `DiffV2_Load_Value_Fn`."""
    ckpts = []
    for seed in (7, 11):
        path = str(tmp_path / f'value_fn_s{seed}.pt')
        _run(_logwealth(_cfg(seed=seed, save=path)))
        ckpts.append(path)
    reads = []
    for order in (ckpts, ckpts[::-1]):
        v = _run(_logwealth(_cfg(seed=7, load=order))).evaluation_summary['diagnostics']['verdict']
        reads.append((v['greedy_q_traj'], v['greedy']['wT_mean'], v['drift_null_sigma']))
    assert reads[0][0] == reads[1][0], [(a, b) for a, b in zip(reads[0][0], reads[1][0]) if a != b][:3]
    assert reads[0][1] == reads[1][1], (reads[0][1], reads[1][1])
    assert reads[0][2] == reads[1][2] > 0.0, (reads[0][2], reads[1][2])     # the tripwire's null, averaged


def test_the_drift_monitor_scores_a_decision_only_once_its_move_has_printed(tmp_path):
    """The stepper roll's drift CUSUM enters each decision's one-day residual at the NEXT decision,
    so a trip, the printed drift z and, under `DiffV2_Drift_Beta`, the fork's shift at a decision
    read only prices printed before it: the first decision scores nothing (z 0.0, no trip), every
    later decision trips under a hair trigger, the residual path is the fork's against the market
    whatever beta does to the book, and the first book under beta is the beta-off book.

    Killing mutation: today's order, the residual summed and the trip tested before the same
    decision is made (the first decision reads its own next move: z_path[0] non-zero, tripped[0]
    True, and the beta roll's first book shifted off the plain one)."""
    ckpt = str(tmp_path / 'value_fn_drift.pt')
    _run(_passive_cfg(save=ckpt))

    def roll(beta):
        cfg = _passive_cfg(load=ckpt)
        cfg['Calc']['Calculation']['Hedging_Problem']['Solver'].update(
            {'DiffV2_Drift_Threshold_Sigmas': 1e-6, 'DiffV2_Drift_Beta': beta})
        sv = _run(cfg).evaluation_summary['diagnostics']['stepper_verdict']
        return sv['drift_monitor'], sv['greedy_q_traj']
    (plain_m, plain_q), (beta_m, beta_q) = roll(0.0), roll(5.0)
    assert len(plain_m['z_path']) == len(plain_m['t']) >= 3
    assert plain_m['z_path'][0] == 0.0 and plain_m['tripped'][0] is False, (plain_m['z_path'][:2], plain_m['tripped'][:2])
    assert all(plain_m['tripped'][1:]) and all(beta_m['tripped'][1:]), plain_m['tripped']
    assert beta_m['z_path'] == plain_m['z_path'], (beta_m['z_path'][:3], plain_m['z_path'][:3])
    assert beta_q[0] == plain_q[0], (beta_q[0], plain_q[0])


#: A spot whose one market day of variance the fork and a replay share exactly: its GARCH(1,1)
#: variance is a recursion on the realised path.
GARCH_SPOT = {'Omega': 1.0e-5, 'Alpha': 0.05, 'Beta': 0.9, 'Nu': 8.0, 'H0': 2.0e-4}


def _tape(cfg, path, paths=2):
    """`paths` paths of `cfg`'s world simulated from its first sim row on seed 3, one
    `Observed_Scenario` tape each at `path` formatted with its index, every factor's rows of no market
    time on the job's calendar flagged filled; the tapes' paths."""
    cfg['Calc']['Calculation'].update(Execution_Mode='simulate_only', Batch_Size=paths, Random_Seed=3)
    world = _run(cfg).bundle
    r0 = world.initial_time_index
    filled = np.array([not world.business_day.is_on_offset(d) for d in world.scenario_dates[r0:]])
    names = [path.format(k) for k in range(paths)]
    for k, name in enumerate(names):
        np.savez(name, **{f'{n},{FILLED_ROWS}': filled for n in world.factors},
                 **{n: v[r0:].cpu().double().numpy()[..., k] for n, v in world.factors.items()})
    return names

#: A calendar of two consecutive weekday holidays (Tue 14, Wed 15 April), a four-day weekend (Fri 24
#: and Mon 27 April around it) and two Monday holidays.
HOLIDAYS = ('2026-04-14', '2026-04-15', '2026-04-24', '2026-04-27', '2026-05-04', '2026-05-25')


def _testland(tmp_path):
    """A calendar file whose `Testland` holds `HOLIDAYS` beside the weekends."""
    path = str(tmp_path / 'calendars.cal')
    with open(path, 'w') as f:
        f.write('<Calendars>\n  <Calendar Location="Testland" Weekends="Saturday and Sunday" Holidays="'
                + ', '.join(f'{d}|Holiday' for d in HOLIDAYS) + '"/>\n</Calendars>\n')
    return path


def _on(cfg, spot='GARCHSpotModel', block=GARCH_SPOT, calendar=None, **solver):
    """`cfg` with its spot `spot` on `block` over `SLOW_BASIS`, on the calendar file `calendar`'s
    Testland where one is given, its solver updated by `solver`."""
    market = cfg['Calc']['MergeMarketData']['ExplicitMarketData']
    market['Price Models'] = {f'{spot}.PLATINUM_LME': block, 'BasisLinkedSpotModel.PLATINUM_LME.LME_CME': SLOW_BASIS}
    market['Model Configuration'] = {'.ModelParams': {'modeldefaults': {'CommodityPrice': spot}, 'modelfilters': {}}}
    if calendar is not None:
        cfg['Calc']['CalendDataFile'] = calendar
        cfg['Calc']['Calculation']['Calendar'] = 'Testland'
    cfg['Calc']['Calculation']['Hedging_Problem'].get('Solver', {}).update(solver)
    return cfg


def test_consecutive_holidays_hold_every_factor_and_carry_one_market_step(tmp_path):
    """On a calendar with two consecutive weekday holidays and a holiday either side of a weekend, a
    simulated GARCH spot and its slow-mean basis hold to the bit on every holiday and weekend row,
    and the move from Thursday 23 to Tuesday 28 April (five calendar days, one market day) and from
    Monday 13 to Thursday 16 (three, one) each carry one market step's variance: against Tuesday
    21 to Wednesday 22 within 15% at 8,192 paths (1.006 and 0.982 for the spot, 1.053 and 1.024 for
    the basis); no decision row is a holiday. 1 s.

    Killing mutation: a weekday clock blind to the job's holidays (`utils.market_days` counting
    weekdays: the spot moving on Tuesday 14 April)."""
    cfg = _on(jsonlib.load(open(FIXTURE)), calendar=_testland(tmp_path))
    cfg['Calc']['Calculation'].update(Execution_Mode='simulate_only', Batch_Size=8192, Random_Seed=3)
    world = _run(cfg).bundle
    r0 = world.initial_time_index
    dates = pd.DatetimeIndex(world.scenario_dates[r0:])
    paths = {'spot': np.log(world.factors['CommodityPrice.PLATINUM_LME'][r0:].double().numpy()),
             'basis': world.factors['ObservedBasis.PLATINUM_LME.LME_CME'][r0:].double().numpy()}
    row = dates.get_loc
    held = [d for d in dates[dates <= '2026-05-31'] if not world.business_day.is_on_offset(d)]
    spans = {'hol_weekend': ('2026-04-23', '2026-04-28'), 'two_hol': ('2026-04-13', '2026-04-16'),
             'one': ('2026-04-21', '2026-04-22')}
    for name, x in paths.items():
        for d in held:
            assert np.array_equal(x[row(d)], x[row(d) - 1]), (name, d)
        var = {k: np.var(x[row(pd.Timestamp(b))] - x[row(pd.Timestamp(a))]) for k, (a, b) in spans.items()}
        for k in ('hol_weekend', 'two_hol'):
            assert abs(var[k] / var['one'] - 1.0) < 0.15, (name, k, var[k] / var['one'])
    decided = {str(world.scenario_dates[r0 + t].date()) for t in world.decision_steps}
    assert not decided & set(HOLIDAYS), decided & set(HOLIDAYS)


def test_the_utility_scale_counts_its_horizon_on_the_job_s_calendar(tmp_path):
    """Under the default `vol_scaled_notional` the utility scale is the leg volume times the spot
    times its annual vol times sqrt(tau), tau the business days of the job's calendar from the first
    simulated row to the last settlement over 252, the market clock the vol is annualised on. The
    simulate-only fixture on the Testland calendar of `HOLIDAYS` counts 76 market days there against
    82 on a calendar of weekends alone, the volume, the spot and its vol being one history's on
    either, so its scale reads sqrt(76 / 82) = 0.9627 of the other's, to 1e-12. 2 s.

    Killing mutation: tau read as the calendar rows from the first simulated row to the last
    settlement (`int(self.last_settlement_index) - H` in `Bundle._resolve_utility_scale`), 116 on
    either calendar, so the two scales agree (a ratio of 1, 3.9% off)."""
    calendar = str(tmp_path / 'calendars.cal')
    with open(calendar, 'w') as f:
        f.write('<Calendars>\n  <Calendar Location="Testland" Weekends="Saturday and Sunday" Holidays="'
                + ', '.join(f'{d}|Holiday' for d in HOLIDAYS) + '"/>\n'
                '  <Calendar Location="Weekdays" Weekends="Saturday and Sunday" Holidays=""/>\n</Calendars>\n')
    scale, days = {}, {}
    for name, holidays in (('Testland', list(HOLIDAYS)), ('Weekdays', [])):
        cfg = jsonlib.load(open(FIXTURE))
        cfg['Calc']['CalendDataFile'] = calendar
        cfg['Calc']['Calculation'].update(Calendar=name, Batch_Size=2, Random_Seed=3)
        world = _run(cfg).bundle
        after = [(world.scenario_dates[i] + pd.Timedelta(days=1)).date()
                 for i in (world.initial_time_index, world.last_settlement_index)]
        days[name] = int(np.busday_count(*after, holidays=holidays))
        scale[name] = world.utility_scale
    ratio = scale['Testland'] / scale['Weekdays'] / np.sqrt(days['Testland'] / days['Weekdays'])
    assert days == {'Testland': 76, 'Weekdays': 82} and abs(ratio - 1.0) < 1e-12, (days, scale)


def test_a_friday_decision_forks_to_monday_and_is_scored_against_monday(tmp_path):
    """A decision is a market day holding its book to the next, and its fork lands there with that
    interval's market time. On the fixture with a `GARCHSpotModel` spot over `SLOW_BASIS`, a policy
    trained from step 60 rolls frozen on a tape drawn from the world's own law, its weekends and
    holidays flagged filled: the drift monitor's standardised residual, each decision's fork forecast
    against the move to the next decision, reads an RMS of 1.200 over the 7 Friday decisions, each
    Friday to Monday, and 1.110 over the 32 others, both within 0.5 to 2. 9 s.

    Killing mutations: the fork stepping one calendar row, a Friday's landing on a Saturday with no
    market time (Friday RMS 0); the realised move read over one calendar row (0.074)."""
    path = _tape(_on(jsonlib.load(open(FIXTURE))), str(tmp_path / 'tape_{}.npz'))[0]
    ckpt = str(tmp_path / 'value_fn.pt')
    _run(_on(_cfg(seed=7, save=ckpt), T_Min=60))
    roll = _on(_cfg(seed=7, batch=8, load=ckpt), T_Min=60, DiffV2_Stepper_Rollout='Yes')
    roll['Calc']['Calculation']['Observed_Scenario'] = path
    roll['Calc']['Calculation']['Hedging_Problem']['Randomize_Initial_State'] = 'No'
    sv = _run(roll).evaluation_summary['diagnostics']['stepper_verdict']
    z = np.diff(sv['drift_monitor']['z_path'])
    friday = pd.DatetimeIndex(sv['greedy_q_dates'][:len(z)]).dayofweek == 4
    rms = [float(np.sqrt(np.mean(z[x] ** 2))) for x in (friday, ~friday)]
    assert friday.sum() >= 6 and all(0.5 < x < 2.0 for x in rms), (int(friday.sum()), rms)


def _paying(cfg, day):
    """`cfg` with the fixture's July average paying `day`, the cash account's horizon with it."""
    hp = cfg['Calc']['Calculation']['Hedging_Problem']
    hp['Liabilities']['FloatingEnergyDeal']['PLAT_JUL29']['Payments']['Items'][0]['Payment_Date'] = {'.Timestamp': day}
    hp['Tradable_Instruments']['CashAccountDeal']['USD_CASH']['Investment_Horizon'] = {'.Timestamp': day}
    return cfg


def _saturday_terminal(save=None, load=None, dump=None):
    """The fixture's July average paying Saturday 8 August, so its terminal, the last live mark, is a
    row of no market time: swept from Monday 3 August on 64 inner paths, the frozen roll on one path."""
    cfg = _paying(_cfg(batch=2 if load else 8, inner=64, save=save, load=load), '2026-08-08')
    hp = cfg['Calc']['Calculation']['Hedging_Problem']
    hp['Randomize_Initial_State'] = 'No'
    hp['Solver']['T_Min'] = 115
    if load is not None:
        hp['Solver'].update(DiffV2_Stepper_Rollout='Yes', DiffV2_Decision_Curve_Dump=dump)
    return cfg


def test_the_last_decision_s_fork_lands_on_a_terminal_of_no_market_time(tmp_path):
    """A decision's fork lands on its successor, and the last decision's is the terminal whatever its
    day: where the last live liability mark is a Saturday, the Friday fork lands there and its mean
    liability move is a day's drift. On the frozen roll's decision dump every decision's `E_dL` is
    within 1% of the liability's level, -7.01 to -6.68 on the four before the last and -6.58 on the
    last against 200,210. 5 s.

    Killing mutation: the fork's own landing, the next market row or the grid's last row, which past
    a Friday is the clean-exit row where the liability has settled (the last `E_dL` +55,586)."""
    ckpt, dump = str(tmp_path / 'vf.pt'), str(tmp_path / 'curve.csv')
    _run(_saturday_terminal(save=ckpt))
    world = _run(_saturday_terminal(load=ckpt, dump=dump)).bundle
    r0, horizon = world.initial_time_index, world.last_live_mtm_index
    assert world.scenario_dates[r0 + horizon].dayofweek == 5, world.scenario_dates[r0 + horizon]
    e_dl = pd.read_csv(dump)['E_dL']
    level = float(world.liability_sim[:horizon + 1].abs().mean())
    assert len(e_dl) >= 3 and (e_dl.abs() < 0.01 * level).all(), (e_dl.tolist(), level)


def _zero_rates(cfg):
    """`cfg` at zero SOFR and no opening cash, so neither the stepper's daily interest on cash and
    margin, which the verdict's wealth law leaves out, nor a margin's growth moves a P&L."""
    hp = cfg['Calc']['Calculation']['Hedging_Problem']
    hp['Portfolio_State']['Cash_Balances']['USD_CASH'] = 0.0
    sofr = cfg['Calc']['MergeMarketData']['ExplicitMarketData']['Price Factors']['InterestRate.USD-SOFR']
    sofr['Curve']['.Curve']['data'] = [[t, 0.0] for t, _ in sofr['Curve']['.Curve']['data']]
    return cfg


def _swap(**kw):
    """The swap document of the passive tests under the position state at zero rates."""
    cfg = _zero_rates(_passive_cfg(**kw))
    cfg['Calc']['Calculation']['Hedging_Problem']['Solver']['DiffV2_Position_State'] = 'Yes'
    return cfg


def _replayed_roll(tmp_path, doc, **declared):
    """`doc()` trained and rolled frozen on the first path of a tape of its own world with
    `Randomize_Initial_State` off, `declared` stated on the roll's `Evaluator`, or on its `Solver`
    for a `DiffV2_` key: the roll's diagnostics."""
    path = _tape(doc(), str(tmp_path / 'tape_{}.npz'))[0]
    ckpt = str(tmp_path / 'vf.pt')
    _run(doc(save=ckpt))
    roll = doc(load=ckpt)
    roll['Calc']['Calculation']['Observed_Scenario'] = path
    hp = roll['Calc']['Calculation']['Hedging_Problem']
    hp['Randomize_Initial_State'] = 'No'
    for key, value in declared.items():
        hp['Solver' if key.startswith('DiffV2_') else 'Evaluator'][key] = value
    return _run(roll).evaluation_summary['diagnostics']


def _swap_rolling(**kw):
    """The swap document with a matched roll priced at a 3 bp calendar spread."""
    cfg = _swap(**kw)
    cfg['Calc']['Calculation']['Hedging_Problem']['Evaluator']['Calendar_Spread_Bps'] = 3.0
    return cfg


@pytest.mark.parametrize('doc, declared', [
    (_swap, {}), (_swap, {'Max_Trade_Per_Step': 2.0}),
    (_swap, {'Max_Trade_Per_Step': 2.0, 'DiffV2_First_Decision_From': 'Passive'}), (_swap_rolling, {})],
    ids=['plain', 'max_trade', 'first_from_passive', 'calendar'])
def test_the_verdict_and_the_stepper_close_one_replayed_book_to_one_wealth(tmp_path, doc, declared):
    """On a replayed path of the swap world under the position state, at zero rates and no opening
    cash, so the stepper's daily interest on cash and margin, which the verdict's wealth law leaves
    out, is nil: the verdict and the stepper roll hold one greedy book at every decision and one
    passive book in whole contracts, and each closes its book at the terminal, the last live
    liability mark, at the declared 10 bp spread, so their terminal wealths agree within 5e-6
    relative (greedy 1.6e-7, passive 5.1e-7), the no-hedge track's to the bit; so too under a
    `Max_Trade_Per_Step` of 2 on the roll, with its first greedy decision measured from the passive
    book under `DiffV2_First_Decision_From` 'Passive', and with the matched half of a roll at a 3 bp
    `Calendar_Spread_Bps`, which the verdict's charge and the stepper's rebate both read. 3 s each.

    Killing mutations: the stepper holding the last book one row past the terminal to the
    clean-exit row and unwinding there (greedy +2,502, 1.8%); the passive book in fractional contracts
    the stepper rounds to whole ones (passive +52.5, 2.0e-3); the stepper roll's `_decide` without
    its band (`rate_limit=False`), its first book (3, 2) against the verdict's (1, 1), greedy 48%
    apart; the verdict's first decision measured from the opening book under 'Passive' (its first
    book (1, 1) against the stepper's (-15, -15), greedy 46% apart); the stepper's
    rebate unarmed, the verdict's charge pricing the roll the stepper pays outright (passive 2.0%)."""
    d = _replayed_roll(tmp_path, doc, **declared)
    v, sv = d['verdict'], d['stepper_verdict']
    assert [v['greedy_q_traj'][t - d['root_t']] for t in sv['greedy_q_t']] == sv['greedy_q_traj']
    assert v['nohedge']['wT_mean'] == sv['nohedge']['wT_mean'], (v['nohedge']['wT_mean'], sv['nohedge']['wT_mean'])
    for p in ('greedy', 'passive'):
        assert abs(sv[p]['wT_mean'] / v[p]['wT_mean'] - 1.0) < 5e-6, (p, v[p]['wT_mean'], sv[p]['wT_mean'])


@pytest.mark.parametrize('t_min', [0, 15], ids=['first_row', 'row_17'])
def test_the_verdict_and_the_stepper_carry_one_opening_book(tmp_path, t_min):
    """Every track holds the opening book from the first row to its first decision and trades from
    it there, in the verdict as in the stepper roll: on the swap replay opening short 5 July and 5
    October, swept from the first row or from step 15, its first decision Monday 27 April (row 17),
    the greedy, passive, textbook and no-hedge tracks close to one terminal wealth in both, within
    5e-6 relative (at most 4.7e-7). 4 s each.

    Killing mutations: the no-hedge track's close of the opening book left uncharged in the verdict
    (`_verdict` charging it from its own book), the stepper's no-hedge wealth 85,811.79 against the
    verdict's 86,331.25 from the first row (6.0e-3); the verdict's wealth started at the liability's
    mark at its first decision (`L[t0]`), every traded track 1,828.68 under the stepper's from row
    17, the opening book's P&L to it (greedy 1.3%, passive 6.7%, no hedge 2.1%)."""
    def doc(**kw):
        cfg = _swap(**kw)
        hp = cfg['Calc']['Calculation']['Hedging_Problem']
        hp['Portfolio_State']['Positions'].update({'PL_JUL_2026': -5, 'PL_OCT_2026': -5})
        hp['Solver']['T_Min'] = t_min
        return cfg
    d = _replayed_roll(tmp_path, doc)
    v, sv = d['verdict'], d['stepper_verdict']
    gaps = {p: sv[p]['wT_mean'] / v[p]['wT_mean'] - 1.0 for p in ('greedy', 'passive', 'textbook', 'nohedge')}
    assert all(abs(g) < 5e-6 for g in gaps.values()), gaps


def _cartesian(save=None, load=None):
    """The fixture's July average on its three legs under the position state at zero rates and no
    opening cash, the greedy on the Cartesian grid of 5 levels a leg over [-50, 0] in a corridor
    [-40, -30]; with `load`, the frozen roll on one path."""
    cfg = _zero_rates(_cfg(batch=2 if load else 12, seed=7, save=save, load=load,
                           schedule=[{'Step': 0, 'Min_Total': -40, 'Max_Total': -30}]))
    solver = cfg['Calc']['Calculation']['Hedging_Problem']['Solver']
    solver['DiffV2_Position_State'] = 'Yes'
    if load is not None:
        solver['DiffV2_Stepper_Rollout'] = 'Yes'
    return cfg


def test_the_verdict_and_the_stepper_hold_one_greedy_book_on_the_cartesian_grid(tmp_path):
    """The greedy argmax on the Cartesian grid, the default without `Allocation_Weights`, offers
    levels in whole contracts, `linspace(Min, Max, levels)` rounded: on a replayed path under the
    position state at zero rates and no opening cash, the verdict and the stepper roll hold one
    greedy book in whole contracts and agree within 5e-6 relative, as they do on the ladder and as
    the passive and no-hedge books do. 4 s.

    Killing mutation: the levels left fractional (-37.5 and -12.5 contracts on the fixture's
    [-50, 0] at 5 levels), which the verdict holds and the stepper rounds each trade of
    (`BundleStepper._structured_action`), so the two roll different books (greedy 6.1e-4 apart
    where the passive agrees to 1e-7)."""
    d = _replayed_roll(tmp_path, _cartesian)
    v, sv = d['verdict'], d['stepper_verdict']
    frac = sorted({x for r in v['greedy_q_traj'] for x in r if x != round(x)})
    gaps = {p: sv[p]['wT_mean'] / v[p]['wT_mean'] - 1.0 for p in ('greedy', 'passive', 'nohedge')}
    assert not frac and all(abs(g) < 5e-6 for g in gaps.values()), (frac, gaps)


def test_the_cartesian_grid_caps_the_signed_net_of_a_mixed_sign_book():
    """`Total_Position_Abs_Limit` caps the absolute signed book total, so the Cartesian grid keeps
    every row whose signed net is inside the cap whatever its gross: the July and October legs on
    boxes [-30, 30] at five levels, -30 to 30 by 15, under a cap of 20, offer the 13 books whose
    legs sum to -15, 0 or 15, a long 30 July against a short 30 October among them, of the 25 the
    levels make. 3 s.

    Killing mutation: the grid's rows dropped by their gross sum (`.abs()` before the sum in
    `HedgeActionSpace._within`), which leaves the 5 books of at most one leg's 15 contracts."""
    cfg = _cfg(batch=8, inner=4, active=[1, 2])
    ev = cfg['Calc']['Calculation']['Hedging_Problem']['Evaluator']
    for leg in LEGS:
        ev['Position_Limits'][leg] = {'Min_Position': -30, 'Max_Position': 30}
    ev['Total_Position_Abs_Limit'] = 20
    levels = np.linspace(-30, 30, 5)
    want = sum(abs(a + b) <= 20 for a in levels for b in levels)
    got = _run(cfg).evaluation_summary['diagnostics']['action_grid_size']
    assert want == 13 and got == want, (got, want)


def test_the_cap_holds_the_book_the_live_legs_carry():
    """`Total_Position_Abs_Limit` caps the signed book the live legs carry, so a row parked on an
    expired leg cannot carry the live ones past it: on the training document with the April leg's
    box long [0, 50] under the cap of 50, the April leg dead at every decision of the sweep from
    step 100, the verdict's greedy book on the live legs sums inside the cap on every path, its
    path mean -50 to -18.75. 3 s.

    Killing mutation: the cap held on a row's total over every leg (`rows.sum(-1)` in
    `HedgeActionSpace._within`), which admits (+50, -50, -50) at -50 and masks it to -100 on the
    live legs: a path-mean net of -62.5."""
    cfg = _cfg(batch=16, inner=8, seed=7)
    cfg['Calc']['Calculation']['Hedging_Problem']['Evaluator']['Position_Limits']['PL_APR_2026'] = {
        'Min_Position': 0, 'Max_Position': 50}
    nets = [sum(q) for q in _run(cfg).evaluation_summary['diagnostics']['verdict']['greedy_q_traj']]
    assert max(abs(x) for x in nets) <= 50 + 1e-9, nets


def test_the_verdict_and_the_stepper_hold_one_textbook_book(tmp_path):
    """The verdict and the stepper roll hold one textbook book on a simulated world, and with it one
    wealth: on the swap document's frozen eval at zero rates on 12 simulated paths, where the
    replication hedge is not zero as it is on a replayed path, and is one book on every path, so the
    verdict's path-mean spread is each path's, the textbook's terminal wealth agrees within 5e-6
    relative, as the passive's and the no-hedge's do. 4 s.

    Killing mutations: the verdict's textbook book left fractional, the replication hedge projected
    into the corridor (-10.62 and -10.26 contracts at the first decision; the stepper 1.43% below);
    the stepper roll's own spelling, scaled to the gross cap, projected into the corridor and
    rounded leg by leg (2.27% above)."""
    ckpt = str(tmp_path / 'vf.pt')
    _run(_swap(save=ckpt))
    roll = _swap(load=ckpt)
    roll['Calc']['Calculation']['Hedging_Problem']['Randomize_Initial_State'] = 'No'
    d = _run(roll).evaluation_summary['diagnostics']
    v, sv = d['verdict'], d['stepper_verdict']
    gaps = {p: sv[p]['wT_mean'] / v[p]['wT_mean'] - 1.0 for p in ('textbook', 'passive', 'nohedge')}
    assert all(abs(g) < 5e-6 for g in gaps.values()), gaps


@pytest.mark.parametrize('mode', ['futures', 'cash_account'])
@pytest.mark.parametrize('terminal', ['2026-08-05', '2026-08-08'])
def test_the_stepper_closes_a_book_held_at_every_decision_at_the_terminal(mode, terminal):
    """A client driving `create_stepper()` short 10 `PL_OCT_2026` at every row the stepper calls a
    decision, on the simulate-only fixture at zero rates paying Wednesday 5 August (a market row) or
    Saturday 8 August (none), ends flat under `Force_Flat_At_End`, and against the no-trade roll
    earns per path -10 x 50 x (F_T - F_0) less half the 10 bp spread on each path's own mark,
    entering at row 0 and unwinding at the terminal T, the last live liability mark: the P&L of a
    book held from row 0 and closed at T, within 1e-5 relative. 2 s each.

    Killing mutations: the stepper deciding on every row of market time, a Wednesday terminal among
    them, so the trade it takes there is held unclosed (final position -10, the Wednesday cases); the
    close one row late at the clean-exit row (`current == self._close_idx` closing at
    `_values_at(last)`) in `_futures_account_step` and in `_cash_account_step`, the hedge P&L
    picking up the Saturday-to-Sunday move (path 1 -305,396 against -334,473)."""
    cfg = _paying(_zero_rates(jsonlib.load(open(FIXTURE))), terminal)
    cfg['Calc']['Calculation'].update(Batch_Size=8, Random_Seed=3)
    hp = cfg['Calc']['Calculation']['Hedging_Problem']
    hp['Evaluator']['Accounting_Mode'] = mode
    result = _run(cfg)
    world = result.bundle
    T = world.last_live_mtm_index
    assert str(world.scenario_dates[world.initial_time_index + T].date()) == terminal

    def roll(book):
        stepper = result.create_stepper()
        while not stepper.done:
            if stepper.is_decision_step:
                held = stepper.observe()['positions']['PL_OCT_2026']
                last = stepper.step({'PL_OCT_2026': book - held})
            else:
                last = stepper.step(None)
        wealth = (last['transition_pnl_excess'] + last['transition_liability_value']).double()
        return wealth, stepper.observe()['positions']['PL_OCT_2026']

    flat, _ = roll(0.0)
    short, final = roll(-10.0)
    F = world.tradables_sim['PL_OCT_2026'].double()
    kappa = 0.5 * hp['Evaluator']['Bid_Offer_Spread_Bps'] * 1e-4 * 50.0
    want = -10.0 * 50.0 * (F[T] - F[0]) - 10.0 * kappa * (F[0] + F[T])
    assert (final == 0).all(), final.tolist()
    assert torch.allclose(short - flat, want, rtol=1e-5, atol=0.0), ((short - flat).tolist(), want.tolist())


def _client(mode='futures', force_flat='Yes', positions=None, evaluator=None):
    """The simulate-only fixture at zero rates paying Monday 10 August, so the row before its
    terminal is a Sunday the stepper does not decide on, on 4 paths: in accounting `mode`, under
    `Force_Flat_At_End` `force_flat`, opening on `positions`, its `Evaluator` updated by
    `evaluator`."""
    cfg = _paying(_zero_rates(jsonlib.load(open(FIXTURE))), '2026-08-10')
    cfg['Calc']['Calculation'].update(Batch_Size=4, Random_Seed=3)
    hp = cfg['Calc']['Calculation']['Hedging_Problem']
    hp['Evaluator'].update(Accounting_Mode=mode, Force_Flat_At_End=force_flat, **(evaluator or {}))
    hp['Portfolio_State']['Positions'].update(positions or {})
    return cfg


def _csv_against_the_stepper(tmp_path, stepper, last):
    """The diagnostic CSV `stepper` writes, by case and day, each case's last
    `hedge_portfolio_ex_funding` checked against the stepper's `pnl_excess` within 1e-5 relative."""
    stepper.write_diagnostic_csvs(str(tmp_path), 'client')
    pnl = last['transition_pnl_excess'].double()
    cases = {case: rows.set_index('day') for case, rows in pd.read_csv(tmp_path / 'client_paths.csv').groupby('case')}
    for case, rows in cases.items():
        want = float(pnl.mean() if case == 'mean' else pnl[int(rows['path_idx'].iloc[0])])
        got = rows['hedge_portfolio_ex_funding'].iloc[-1]
        assert abs(got / want - 1.0) < 1e-5, (case, got, want)
    return cases


#: A spread of 10 bp at a 30% vol, scaled by the step's vol; a calendar spread of 3 bp.
VOL_SPREAD = {'Bid_Offer_Spread_Bps': {'Default_Bps': 10.0, 'Vol_Scale': {'Ref_Vol': 0.3, 'Beta': 1.0}}}
CALENDAR = {'Calendar_Spread_Bps': 3.0}


@pytest.mark.parametrize('mode, force_flat, evaluator, book', [
    ('futures', 'Yes', {}, {}), ('cash_account', 'Yes', {}, {}), ('futures', 'No', {}, {}),
    ('futures', 'Yes', VOL_SPREAD, {}), ('cash_account', 'Yes', VOL_SPREAD, {}),
    ('futures', 'Yes', CALENDAR, {'PL_JUL_2026': 10.0})],
    ids=['futures', 'cash', 'held', 'vol_spread', 'cash_vol_spread', 'calendar'])
def test_the_diagnostic_csv_books_the_forced_close_on_the_terminal(tmp_path, mode, force_flat, evaluator, book):
    """A client short 10 `PL_OCT_2026` from the first decision through `create_stepper()`, on the
    simulate-only fixture at zero rates paying Monday 10 August, so the row before its terminal is a
    Sunday the stepper does not decide on, in either accounting mode: `write_diagnostic_csvs` books
    the forced close on the terminal, +10 at Monday's mark, the book flat from there and -10 the day
    before, or under `Force_Flat_At_End` 'No' the book short to the last row, and on each path it
    writes its last `hedge_portfolio_ex_funding` is the stepper's hedge P&L within 1e-5 relative;
    so too under a spread scaled by the step's vol in either accounting mode, the close paying it at
    the terminal's vol, and long 10 `PL_JUL_2026` beside the short under `Calendar_Spread_Bps`, the
    entry a matched roll and the forced close outright. 2 s each.

    Killing mutations: the CSV recording the trades of the decision rows alone, the close of the
    step into the terminal lost (trade 0 on the terminal, the book held short to the clean-exit
    row: the mean path's last 49,863.81 against 50,038.13); `_hold` after the forced close in either
    accounting step, the close booked on Sunday at Sunday's mark (49,984.73); the CSV's book left
    flat past the stepped rows (`pos[rows.stop:]` at zero), the book dropped on the clean-exit row
    under 'No'; the forced close charged at the vol of the row before it (`vol_t` handed to
    `_flatten_futures`, the stepper 2.3e-4 off its CSV on the best path; to `_flatten_cash`, 2.3e-4
    on the cash-account document's); the CSV crediting the matched-roll rebate on the forced close
    (3.7% above the stepper)."""
    result = _run(_client(mode, force_flat, evaluator=evaluator))
    stepper, r0 = result.create_stepper(), result.bundle.initial_time_index
    while not stepper.done:
        last = stepper.step({'PL_OCT_2026': -10.0, **book} if stepper.time_index == r0 else None)
    for case, rows in _csv_against_the_stepper(tmp_path, stepper, last).items():
        assert rows.loc['2026-08-09', 'PL_OCT_2026_position'] == -10.0
        closed = force_flat == 'Yes'
        assert rows.loc['2026-08-10', 'PL_OCT_2026_trade'] == (10.0 if closed else 0.0), (case, rows['PL_OCT_2026_trade'].tail(3))
        assert (rows.loc['2026-08-10':, 'PL_OCT_2026_position'] == (0.0 if closed else -10.0)).all()


def test_an_opening_book_s_pnl_runs_from_the_first_row_in_the_stepper_and_its_csv(tmp_path):
    """An opening book short 10 `PL_OCT_2026`, its declared settlement off the first row's mark,
    held untouched to the forced close on the client fixture: the stepper's `pnl_excess`, the wealth
    change since inception, is on every path the book's P&L from the first row, -10 x 50 x (F_T -
    F_0) less half the 10 bp spread on F_T a contract closed, within 1e-5 relative, and the CSV's
    last `hedge_portfolio_ex_funding` is that P&L on each path it writes. 2 s.

    Killing mutations: the inception baseline taken with the opening book settled at its declared
    price, the seed-to-first-row gain in the baseline and never in the value (`pnl_excess` below
    the P&L by 20,965 a path); the CSV's first row booking no trade cash for the opening book, its
    notional entering as P&L (the best path's last -1,024,063 against -9,652)."""
    result = _run(_client(positions={'PL_OCT_2026': -10}))
    stepper = result.create_stepper()
    while not stepper.done:
        last = stepper.step(None)
    world = result.bundle
    F, T = world.tradables_sim['PL_OCT_2026'].double(), world.last_live_mtm_index
    want = -10.0 * 50.0 * (F[T] - F[0]) - 10.0 * 0.5 * 10.0 * 1e-4 * 50.0 * F[T]
    got = last['transition_pnl_excess'].double()
    assert torch.allclose(got, want, rtol=1e-5, atol=0.0), ((got - want).tolist(), want.tolist())
    _csv_against_the_stepper(tmp_path, stepper, last)


def test_a_first_row_of_no_market_time_is_one_decision_for_the_verdict_and_the_stepper(tmp_path):
    """A job whose first row is a holiday of its calendar, Friday 10 April on a Testland calendar:
    the solver decides at row 0, so the stepper roll of the same frozen policy on the same replayed
    swap path, swept from `T_Min` 0, decides on the same rows, the root first among them, and closes
    its greedy, passive and no-hedge books to the verdict's terminal wealth within 5e-6 relative,
    as on a first row that is a market row. 5 s.

    Killing mutation: the stepper deciding on the rows of market time alone, which leave out a first
    row of none, so its roll first decides on Monday 13 (row 3) and holds the opening book over the
    move the verdict holds its first book over (greedy -8.1%, passive -67%)."""
    calendar = str(tmp_path / 'calendars.cal')
    with open(calendar, 'w') as f:
        f.write('<Calendars>\n  <Calendar Location="Testland" Weekends="Saturday and Sunday" '
                'Holidays="2026-04-10|Holiday"/>\n</Calendars>\n')

    def doc(**kw):
        cfg = _swap(**kw)
        cfg['Calc']['CalendDataFile'] = calendar
        cfg['Calc']['Calculation']['Calendar'] = 'Testland'
        cfg['Calc']['Calculation']['Hedging_Problem']['Solver']['T_Min'] = 0
        return cfg

    path = _tape(doc(), str(tmp_path / 'tape_{}.npz'))[0]
    ckpt = str(tmp_path / 'vf.pt')
    _run(doc(save=ckpt))
    roll = doc(load=ckpt)
    roll['Calc']['Calculation']['Observed_Scenario'] = path
    roll['Calc']['Calculation']['Hedging_Problem']['Randomize_Initial_State'] = 'No'
    d = _run(roll).evaluation_summary['diagnostics']
    v, sv = d['verdict'], d['stepper_verdict']
    gaps = {p: sv[p]['wT_mean'] / v[p]['wT_mean'] - 1.0 for p in ('greedy', 'passive', 'nohedge')}
    assert sv['greedy_q_t'][0] == d['root_t'], (sv['greedy_q_t'][:3], d['root_t'])
    assert all(abs(g) < 5e-6 for g in gaps.values()), gaps


# --------------------------------------------------------------------------------------------
# (vi) convergence - what a fit saves, what a roll loads, and the book column's label
# --------------------------------------------------------------------------------------------
def _swap_growth(seed, save=None, load=None):
    """The swap fixture under LogWealth with the position and returns states, trained on 512 paths a
    batch, 16 inner draws and 60 fit iterations over the sweep to the first decision; with `load`, the
    frozen roll of that checkpoint on the fixture's own batch."""
    cfg = _passive_cfg(save=save, load=load)
    calc = cfg['Calc']['Calculation']
    hp = calc['Hedging_Problem']
    hp['Objective'] = {'Object': 'LogWealth'}
    hp['Solver'].update({'DiffV2_Position_State': 'Yes', 'DiffV2_Returns_State': 'Yes',
                         'DiffV2_Fit_Iters': 60, 'T_Min': 0, 'DiffV2_Stepper_Rollout': 'No'})
    if load is None:
        calc['Random_Seed'], calc['Batch_Size'], calc['Inner_Sub_Batch'] = seed, 512, 16
    return cfg


@pytest.fixture(scope='module')
def growth(tmp_path_factory):
    """The growth document trained under seed 6, saved: `(checkpoint path, diagnostics)`."""
    ckpt = str(tmp_path_factory.mktemp('vf') / 'value_fn_s6.pt')
    return ckpt, _run(_swap_growth(6, save=ckpt)).evaluation_summary['diagnostics']


def test_a_fit_whose_value_outruns_its_held_out_utility_saves_nothing(growth, tmp_path):
    """`DiffV2_Max_Value_Optimism` bounds V_0 less the held-out greedy utility, net of the
    repositioning charge under the position state the swap fixture carries: the value a fit claims
    against what its own argmax earns on a world no fit step saw. Seed 5 claims V_0 +0.1937
    against a held-out greedy u of +0.2066 on the CPU (+0.1839 and +0.1990 on a card); the test
    declares the bound at -10, below any fit's gap, so on either device the save is refused naming
    the field, V_0 and the held-out utility, and no checkpoint is written. Seed 6 (V_0 +0.1639 on
    the CPU, +0.1755 on a card) saves, stamped with V_0 less its held-out greedy
    `u_mean`; a roll declaring a bound below the stamp refuses the member by name, and the same
    checkpoint without the stamp, as every checkpoint saved before it, loads at that bound. 20 s.

    Killing mutations: the save's check removed (seed 5 writes its checkpoint); the roll's check
    removed (the stamped member loads below its stamp). The verdict's wealth recursion gross of the
    charge is `test_a_label_past_the_horizon_is_what_its_greedy_policy_realises`'s."""
    ruined = str(tmp_path / 'value_fn_s5.pt')
    cfg = _swap_growth(5, save=ruined)
    cfg['Calc']['Calculation']['Hedging_Problem']['Solver']['DiffV2_Max_Value_Optimism'] = -10.0
    with pytest.raises(ValueError, match=r'DiffV2_Max_Value_Optimism.*V_0=[+-]\d\.\d+.*utility [+-]?\d'):
        _run(cfg)
    assert not os.path.exists(ruined)

    saved, diag = growth
    stamped = torch.load(saved, map_location='cpu', weights_only=False)
    assert stamped['value_optimism'] == diag['V_0'] - diag['verdict']['greedy']['u_mean'], \
        stamped['value_optimism']
    below = _swap_growth(6, load=saved)
    below['Calc']['Calculation']['Hedging_Problem']['Solver']['DiffV2_Max_Value_Optimism'] = \
        float(stamped['value_optimism']) - 1e-3
    with pytest.raises(ValueError, match='DiffV2_Max_Value_Optimism: .*value_fn_s6.pt was saved'):
        _run(below)
    del stamped['value_optimism']
    torch.save(stamped, str(tmp_path / 'value_fn_unstamped.pt'))
    below['Calc']['Calculation']['Hedging_Problem']['Solver']['DiffV2_Load_Value_Fn'] = [
        str(tmp_path / 'value_fn_unstamped.pt')]
    assert _run(below).evaluation_summary['diagnostics']['V_0'] == diag['V_0']


def test_an_ensemble_whose_members_disagree_beyond_the_declaration_is_refused(tmp_path):
    """`DiffV2_Max_Member_Disagreement` bounds the mean absolute difference, in contracts, between
    two ensemble members' swept net books. Two checkpoints of one document trained under seeds 7
    and 11 sweep books 4.73 contracts apart on average over their 12 swept steps on the CPU: a
    two-member roll loads at the default 8 and is refused at 0, naming the field, both
    members and the distance the two checkpoints' swept books give.

    Killing mutation: the check removed (the roll at 0 loads)."""
    ckpts = []
    for seed in (7, 11):
        ckpts.append(str(tmp_path / f'value_fn_s{seed}.pt'))
        _run(_cfg(seed=seed, save=ckpts[-1]))
    assert _run(_cfg(seed=7, load=ckpts)).evaluation_summary['diagnostics']['verdict']['greedy_q_traj']
    strict = _cfg(seed=7, load=ckpts)
    strict['Calc']['Calculation']['Hedging_Problem']['Solver']['DiffV2_Max_Member_Disagreement'] = 0.0
    books = [[sum(q) for q in torch.load(c, map_location='cpu', weights_only=False)['q_star_path']] for c in ckpts]
    apart = sum(abs(x - y) for x, y in zip(*books)) / min(map(len, books))
    with pytest.raises(ValueError, match=r'DiffV2_Max_Member_Disagreement: .*s7\.pt and .*s11\.pt sweep net books '
                       + re.escape(f'{apart:.4g}') + ' contracts apart on average'):
        _run(strict)


def test_the_book_column_is_fitted_to_the_repositioning_charge_s_slope(growth):
    """Under the position state the value net reads the standing book p, a control the bank samples
    whose slope no AAD label measures, so the value term alone leaves it free and the argmax
    harvests whatever slope it lands at. `DiffV2_Book_Gradient_Label` 'Yes' makes the book a leaf
    of the gradient fork, whose label is by the envelope theorem the repositioning charge's slope at
    the chosen action, and fits the net's dA/dp to it. On the swap fixture under seed 6 the field
    omitted and 'No' train the same V_0 and root book to the bit (0.1638568490743637, as before
    the field existed); under 'Yes' the fitted slope's mean over the bank departs from the label's
    by 8.3% of the label, summed over the 32 swept steps (bound 25%), and the root book moves. 30 s.

    Killing mutation: the slope term dropped from the fit's loss (the departure reads 123% and the
    'Yes' fit is the 'No' fit to the bit, root book and all)."""
    _, omitted = growth
    reads = {}
    for label in ('No', 'Yes'):
        cfg = _swap_growth(6)
        cfg['Calc']['Calculation']['Hedging_Problem']['Solver']['DiffV2_Book_Gradient_Label'] = label
        reads[label] = _run(cfg).evaluation_summary['diagnostics']
    assert (reads['No']['V_0'], reads['No']['n_star_0']) == (omitted['V_0'], omitted['n_star_0'])
    rows = reads['Yes']['per_t']
    departs = (sum(abs(r['book_slope'] - r['book_label']) for r in rows)
               / sum(abs(r['book_label']) for r in rows))
    assert len(rows) == 32 and departs < 0.25, (len(rows), departs)
    assert reads['Yes']['n_star_0'] != reads['No']['n_star_0'], reads['Yes']['n_star_0']


def _without_first_leg(cfg):
    """`cfg` with its first hedge leg removed: the future, its position, settlement price and
    margin, its position limits and its allocation rows, the whole net on the leg left."""
    gone, kept = LEGS
    hp = cfg['Calc']['Calculation']['Hedging_Problem']
    del hp['Tradable_Instruments']['CommodityFutureDeal'][gone]
    for key in ('Positions', 'Settlement_Prices', 'Initial_Margin'):
        del hp['Portfolio_State'][key][gone]
    del hp['Evaluator']['Position_Limits'][gone]
    hp['Evaluator']['Allocation_Weights'] = [{'Step': 0, 'Instrument': kept, 'Weight': 1.0}]
    hp['Solver']['Active_Hedge_Indices'] = [0]
    return cfg


def test_the_book_label_is_taken_along_the_ladder():
    """The label's slope in the standing book is the charge's along the ladder: a move in p lands
    on the live legs as the step's `Allocation_Weights` split them, intersected with the active
    legs, so a leg the ladder never moves carries no weight in the label whatever the bank books
    on it. The swap growth document under seed 6 with its first leg inactive and no bank noise
    trains to the bit as the same document with that leg removed: V_0 (0.10113716870546341 on
    the CPU), the root book (-9.044921875) and every swept step's `book_label` and `book_slope`;
    both documents declare `Deterministic_Kernels` so the identity holds to the bit on a card too.
    30 s.

    Killing mutation: the gradient summed over the live legs and divided by their count (the
    two-leg label reads 0.50 of the one-leg's, 50% apart summed over the 32 steps, and the fits
    part: V_0 0.0969 against 0.1011, the root book -9.047 against -9.045)."""
    cfg = _swap_growth(6)
    cfg['Calc']['Calculation']['Deterministic_Kernels'] = 'Yes'
    cfg['Calc']['Calculation']['Hedging_Problem']['Solver'].update({
        'DiffV2_Book_Gradient_Label': 'Yes', 'DiffV2_Bank_Noise_Frac': 0.0, 'Active_Hedge_Indices': [1]})
    two = _run(cfg).evaluation_summary['diagnostics']
    one = _run(_without_first_leg(cfg)).evaluation_summary['diagnostics']
    assert (two['V_0'], two['n_star_0']) == (one['V_0'], [0.0] + one['n_star_0']), (two['V_0'], one['V_0'])
    labels = [[(r['book_label'], r['book_slope']) for r in d['per_t']] for d in (two, one)]
    assert len(labels[0]) == 32 and labels[0] == labels[1]


def test_the_bank_is_composed_along_the_ladder():
    """Under `DiffV2_Bank_Along_Ladder` 'Yes' every training-bank book is a rung of the ladder on
    its live legs: the constructed net spans the ladder's band, its reserve is priced on the
    ladder's weights, and the exploration is drawn on the net, clamped into the band and held as a
    rung, so the bank explores only books the policy can hold. On the swap growth document (256
    paths, 8 inner draws, 20 fit iterations, the book label, a [-12, 10] corridor) A has both legs
    active, one knot weighting the July leg 0 and `Calendar_Spread_Bps` 3; B has only the October
    leg active, no calendar and October's box floored at -12, so B's band stops at -12 on its box
    where A's stops on the corridor, and both stop at 5 on October's box where A's `net_bounds`
    reaches 10. Under 'Yes' A trains to the bit as B: V_0 (0.05139961093664169 on the CPU), the
    root book ([0, -4.76953125]) and every swept step's label, slope and value; under 'No' they part
    (V_0 0.0495 against 0.0482), and A with the field omitted is A under 'No' to the bit. Under an
    even (0.5, 0.5) knot a rung moves both legs the same way, so the calendar flag meets no matched
    volume and the fits with and without it are one (V_0 0.06728600710630417).
    `Deterministic_Kernels` holds the identities on a card too.

    Killing mutations, A's V_0 against B's: the constructed policy's three water-fills left on the
    replication shape (0.0496 against 0.0514); the exploration drawn per active leg and projected
    into the corridor, as the default bank's is (0.0508 against 0.0482; the even knot's fits part,
    0.0662 against 0.0622); the noisy net left unclamped (0.0456 against 0.0470); the constructed
    net spanning `net_bounds` (0.0470 against 0.0514); the reserve priced on the replication split
    (0.0518 against 0.0514)."""
    def fit(ladder, weights=(0.0, 1.0), calendar=True, october_only=False):
        cfg = _swap_growth(6)
        calc = cfg['Calc']['Calculation']
        calc['Deterministic_Kernels'], calc['Batch_Size'], calc['Inner_Sub_Batch'] = 'Yes', 256, 8
        hp = calc['Hedging_Problem']
        hp['Solver'].update({'DiffV2_Fit_Iters': 20, 'DiffV2_Book_Gradient_Label': 'Yes'})
        hp['Evaluator'].update({
            'Allocation_Weights': [{'Step': 0, 'Instrument': h, 'Weight': w}
                                   for h, w in zip(LEGS, weights)],
            'Total_Position_Schedule': [{'Step': 0, 'Min_Total': -12, 'Max_Total': 10}]})
        if calendar:
            hp['Evaluator']['Calendar_Spread_Bps'] = 3.0
        if october_only:
            hp['Solver']['Active_Hedge_Indices'] = [1]
            hp['Evaluator']['Position_Limits'][LEGS[1]]['Min_Position'] = -12
        if ladder is not None:
            hp['Solver']['DiffV2_Bank_Along_Ladder'] = ladder
        diag = _run(cfg).evaluation_summary['diagnostics']
        return (diag['V_0'], diag['n_star_0'],
                [(r['book_label'], r['book_slope'], r['Y_mean']) for r in diag['per_t']])

    a = fit('Yes')
    assert a == fit('Yes', calendar=False, october_only=True), a[:2]
    a_no = fit('No')
    assert a_no[0] != fit('No', calendar=False, october_only=True)[0], a_no[:2]
    assert fit(None) == a_no, a_no[:2]
    even = fit('Yes', weights=(0.5, 0.5))
    assert even == fit('Yes', weights=(0.5, 0.5), calendar=False), even[:2]


def test_the_ladder_bank_reads_the_policy_s_live_legs_on_a_replayed_path(growth, tmp_path):
    """Under `DiffV2_Bank_Along_Ladder` 'Yes' the bank's live legs are the inner guard's, the mask
    the policy's ladder reads, so a contract the inner fork still prices is live on a day its mark
    did not move. The swap growth checkpoint rolls frozen on one replayed path, the carry and the
    rate at zero and day 20 a copy of day 19 (a forward-filled holiday), so neither leg's mark
    moves into step 20: the roll under 'Yes' reads the roll under 'No' to the bit over its 32 steps,
    the bank being inert on a loaded checkpoint (greedy wT_mean -13054.20 on the CPU, net of its
    charges, the book at step 20 [3, 2]). `Deterministic_Kernels` holds the identity on a card too. 5 s.

    Killing mutation: the bank's live legs read off the outer marks, a leg live while its mark
    moved into the step (refused at step 20: "Allocation_Weights knot in force at step 20 puts no
    weight on a LIVE, ACTIVE hedge leg")."""
    saved, _ = growth
    spot = 2055.0 + 15.0 * np.sin(0.7 * np.arange(64))
    spot[20] = spot[19]
    path = str(tmp_path / 'replay.npz')
    np.savez(path, **{'CommodityPrice.PLATINUM_LME': spot,
                      'ObservedBasis.PLATINUM_LME.LME_CME': np.full(64, -32.5375),
                      'ForwardRate.PLATINUM_CARRY': np.zeros((64, 2)),
                      'InterestRate.USD-SOFR': np.zeros((64, 31))})

    def roll(ladder):
        cfg = _swap_growth(6, load=saved)
        calc = cfg['Calc']['Calculation']
        calc['Batch_Size'], calc['Observed_Scenario'], calc['Deterministic_Kernels'] = 1, path, 'Yes'
        hp = calc['Hedging_Problem']
        hp['Randomize_Initial_State'] = 'No'
        hp['Solver']['DiffV2_Bank_Along_Ladder'] = ladder
        return _run(cfg).evaluation_summary['diagnostics']['verdict']
    yes = roll('Yes')
    assert yes == roll('No'), yes['greedy']


def test_a_no_table_document_whose_legs_all_die_still_labels():
    """Absent an `Allocation_Weights` table the label's ladder is even over the live active legs,
    and a swept step where none can carry a contract has no direction: the label is zero there,
    as the passive holds nothing. The swap growth document on the Cartesian grid, both legs
    expiring between the last fixing and settlement, trains under the label over 32 steps (V_0
    0.1799 on the CPU) with the dead step 46 labelled zero and step 0 not, and its passive splits
    the 32 unfixed contracts evenly, 16 a leg. 20 s.

    Killing mutations: the no-table branch of `_weights_at` removed (refused at step 46, naming a
    table the document never declared); its normalisation dropped (the passive reads [-20, -20],
    V_0 0.1840 against 0.1799 on the CPU)."""
    cfg = _swap_growth(6)
    hp = cfg['Calc']['Calculation']['Hedging_Problem']
    hp['Solver'].update({'DiffV2_Book_Gradient_Label': 'Yes', 'Active_Hedge_Indices': [0, 1],
                         'Training_Action_Grid_Levels_Per_Axis': 6})
    del hp['Evaluator']['Allocation_Weights']
    del hp['Evaluator']['Total_Position_Schedule']
    for leg in LEGS:
        hp['Tradable_Instruments']['CommodityFutureDeal'][leg]['Maturity_Date'] = {'.Timestamp': '2026-05-25'}
    diag = _run(cfg).evaluation_summary['diagnostics']
    labels = {r['t']: r['book_label'] for r in diag['per_t']}
    assert len(labels) == 32 and labels[46] == 0.0 and labels[0] != 0.0, (labels[46], labels[0])
    assert diag['verdict']['passive_q_traj'][0] == [-16.0, -16.0], diag['verdict']['passive_q_traj'][0]


def test_a_label_horizon_of_one_is_the_one_step_fit():
    """`DiffV2_Label_Horizon` 1 is the one-step bootstrap: the label test's document trains with
    the field omitted and stated 1 to the bit, V_0 (0.10113716870546341 on the CPU, as before the
    field existed), the root book and every swept step's diagnostics. 30 s.

    Killing mutation: the default the solver block is read with moved to 2 (the omitted document
    trains V_0 0.1145 against 0.1011)."""
    cfg = _swap_growth(6)
    cfg['Calc']['Calculation']['Deterministic_Kernels'] = 'Yes'
    cfg['Calc']['Calculation']['Hedging_Problem']['Solver'].update({
        'DiffV2_Book_Gradient_Label': 'Yes', 'DiffV2_Bank_Noise_Frac': 0.0, 'Active_Hedge_Indices': [1]})
    omitted = _run(cfg).evaluation_summary['diagnostics']
    cfg['Calc']['Calculation']['Hedging_Problem']['Solver']['DiffV2_Label_Horizon'] = 1
    one = _run(cfg).evaluation_summary['diagnostics']
    assert (one['V_0'], one['n_star_0'], one['per_t']) == (
        omitted['V_0'], omitted['n_star_0'], omitted['per_t']), (one['V_0'], omitted['V_0'])


def _horizon(objective, save=None, load=None):
    """The swap growth document on 512 paths and 8 inner draws under `Deterministic_Kernels`, its
    objective LogWealth at a `DiffV2_Risk_Aversion` of 0.5 or the fixture's terminal utility,
    trained with a label horizon of 100 steps; with `load`, the frozen roll of that checkpoint on a
    one-batch stream of the same seed and shape, which is the training batch."""
    cfg = _swap_growth(6, save=save, load=load)
    calc = cfg['Calc']['Calculation']
    calc.update({'Deterministic_Kernels': 'Yes', 'Random_Seed': 6, 'Batch_Size': 512, 'Inner_Sub_Batch': 8})
    hp = calc['Hedging_Problem']
    if objective == 'terminal':
        hp['Objective'] = jsonlib.load(open(FIXTURE))['Calc']['Calculation']['Hedging_Problem']['Objective']
    else:
        # 0.5 leaves the worst bank book 55% headroom where 1.0 leaves 9%, where some draws hold a
        # path near enough the line to swamp the root's one-step cross-fit
        hp['Solver']['DiffV2_Risk_Aversion'] = 0.5
    if load is None:
        hp['Solver']['DiffV2_Label_Horizon'] = 100
    return cfg


@pytest.mark.parametrize('objective', ['growth', 'terminal'])
def test_a_label_past_the_horizon_is_what_its_greedy_policy_realises(objective, tmp_path):
    """Under `DiffV2_Label_Horizon` k the value label gains, on the outer path, the k-step bootstrap
    along the fitted greedy policy less the one-step one, k clipped at the decision horizon. Past
    the horizon the root label is then the utility the greedy policy realises on the training
    paths, net of its repositioning charges, up to the root's one-step cross-fit, once the roll
    charges each step's wealth as the label does, so each decision reads the net wealth its value
    was fitted on: the swap growth document under seed 6 at a horizon of 100 steps, past its 32,
    trains a V_0 within 0.006 of the `u_mean` its checkpoint's frozen roll on the training batch
    realises (the roll's first greedy book the training root's). On the CPU 0.0981 against 0.0991
    under LogWealth and 0.1899 against 0.1906 under the fixture's terminal utility; seeds 1 to 8
    on a card read within 0.0031. 35 s each.

    Killing mutations, V_0 against the roll's utility: the roll's wealth recursion gross of the
    charge (its `u_mean` 0.1113 and 0.2059 against the net 0.0991 and 0.1906, 0.010 to 0.016 above
    V_0 on every seed 1 to 8); the correction's sign flipped (the recursion runs away, V_0 -1.5e6
    under both); the k-step end read at the bank's own wealth (terminal 0.0625 against 0.1592;
    LogWealth's end at the horizon is zero at any wealth); the rollout's charge dropped (0.1119
    against 0.1010, 0.2049 against 0.1910); the rewards after the first move dropped (growth
    0.0011 against 0.0757); k not clipped at the horizon (refused at the last step, which has no
    successor in the cache)."""
    ckpt = str(tmp_path / 'value_fn.pt')
    train = _run(_horizon(objective, save=ckpt)).evaluation_summary['diagnostics']
    roll = _run(_horizon(objective, load=ckpt)).evaluation_summary['diagnostics']['verdict']
    assert roll['greedy_q_traj'][0] == train['n_star_0'], (roll['greedy_q_traj'][0], train['n_star_0'])
    realised = roll['greedy']['u_mean']
    assert abs(train['V_0'] - realised) < 0.006, (train['V_0'], realised)


def _held_into_expiry(fee):
    """The passive document on its October leg alone under a linear utility at a declared scale of
    1e5, both legs expiring at the last decision step, the corridor pinning the book at -10
    contracts before it, the sweep the last two steps under a label horizon of 2, and a flat
    `Transaction_Cost_Per_Unit` of `fee` the only cost."""
    cfg = _passive_cfg()
    calc = cfg['Calc']['Calculation']
    calc.update({'Deterministic_Kernels': 'Yes', 'Batch_Size': 64})
    hp = calc['Hedging_Problem']
    hp['Objective'] = {'Object': 'AsymmetricUtility_Huber', 'Huber_Aversion': 0.0,
                       'Utility_Scale_Explicit': 1.0e5}
    hp['Solver'].update({'DiffV2_Position_State': 'Yes', 'T_Min': 42, 'Active_Hedge_Indices': [1],
                         'DiffV2_Bank_Noise_Frac': 0.0, 'Training_Action_Grid_Levels_Per_Axis': 26,
                         'DiffV2_Label_Horizon': 2})
    ev = hp['Evaluator']
    del ev['Allocation_Weights']
    ev.update({'Bid_Offer_Spread_Bps': 0.0, 'Transaction_Cost_Per_Unit': fee,
               'Total_Position_Schedule': [{'Step': 0, 'Min_Total': -10, 'Max_Total': -10},
                                           {'Step': 46, 'Min_Total': -20, 'Max_Total': 5}]})
    for leg in LEGS:
        hp['Tradable_Instruments']['CommodityFutureDeal'][leg]['Maturity_Date'] = {'.Timestamp': '2026-05-25'}
    return cfg


def test_the_label_charges_the_close_of_a_leg_that_dies_at_the_step():
    """The repositioning charge is on the move from the standing book, as the step that set it left
    it, to the new book on the live legs, so a book held into its leg's expiry pays the close at
    that step, as the roll pays it. With 10 contracts held into the last decision step at a fee of
    500 a contract, the last step's label sits below the fee-free document's by the close over the
    utility scale, 500 x 10 / 1e5 = 0.05, to 1e-6, and the step before's, whose two-step label
    realises that close on the outer path, by the same within 1e-3 (2.3e-4 on the CPU). 3 s.

    Killing mutations: the standing book masked by the step's live legs before the charge (both
    shifts 0); in the one-step label alone (the last step's 0); in the k-step correction alone (the
    step before's -2.3e-4). Not seen: the argmax's own spelling, the corridor and the expiry leaving
    each step one candidate."""
    rows = [{r['t']: r['Y_mean'] for r in _run(_held_into_expiry(fee)).evaluation_summary['diagnostics']['per_t']}
            for fee in (0.0, 500.0)]
    shift = {t: rows[1][t] - rows[0][t] for t in (42, 46)}
    assert abs(shift[46] + 0.05) < 1e-6 and abs(shift[42] + 0.05) < 1e-3, shift


#: A spot of one variance a market day, a GARCH(1,1) at alpha = beta = 0; and one whose long-run
#: variance that is, started at nine times it and reverting at beta 0.8 with alpha 0, so its variance
#: path is its forecast, E[h_{m+j}] = hbar + beta^j (h_m - hbar).
HBAR, REVERSION = 2.5e-5, 0.8
CONSTANT_VOL = {'Omega': HBAR, 'Alpha': 0.0, 'Beta': 0.0, 'Nu': 30.0, 'H0': HBAR}
HIGH_VOL = {'Omega': HBAR * (1.0 - REVERSION), 'Alpha': 0.0, 'Beta': REVERSION, 'Nu': 30.0,
            'H0': 9.0 * HBAR}
#: `CONSTANT_VOL` drifting at 50% a year, so the mean and the risk term pull a free book apart.
DRIFTING = dict(CONSTANT_VOL, Mu=0.5)
#: kappa 0.6 over one decision, and kappa 0.6 / sqrt(5) over five.
SCALED = ({'DiffV2_Risk_Kappa': 0.6}, {'DiffV2_Risk_Kappa': 0.6 / np.sqrt(5.0), 'DiffV2_Risk_Horizon': 5})
#: The default Huber shape on the day's increment.
RUNNING = {'Reference_Mode': 'Running_Wealth', 'Huber_Aversion': 2.5}


def _spot_alone(cfg, block, **objective):
    """`cfg` from its calibrated opening state with its spot a GARCH spot on `block` and every other
    factor constant, under a linear utility at a declared scale of 1e5 (its `Objective` updated by
    `objective`), swept from the first decision with no fit, so each continuation is the wealth over
    the scale plus a constant."""
    market = cfg['Calc']['MergeMarketData']['ExplicitMarketData']
    market['Price Models'] = {'GARCHSpotModel.PLATINUM_LME': block}
    market['Model Configuration'] = {'.ModelParams': {'modelfilters': {}, 'modeldefaults': {
        'CommodityPrice': 'GARCHSpotModel', 'ForwardRate': '', 'InterestRate': '', 'ObservedBasis': ''}}}
    hp = cfg['Calc']['Calculation']['Hedging_Problem']
    hp['Randomize_Initial_State'] = 'No'
    hp['Objective'] = {'Object': 'AsymmetricUtility_Huber', 'Huber_Aversion': 0.0,
                       'Utility_Scale_Explicit': 1.0e5, **objective}
    hp['Solver'].update({'T_Min': 0, 'DiffV2_Fit_Iters': 0})
    return cfg


def _risk_roll(ckpt, path, block, held=True, objective=None, **solver):
    """The frozen roll of `ckpt` on one path of `_spot_alone`'s world on `block` (under `objective`)
    at 256 inner draws, trading the October leg alone; where `held`, the liability cut to one ounce
    and the book pinned short 25 contracts, which trade past the terminal, so every such roll's
    wealth moves as that future does. The decision dump it writes at `path`, and its result."""
    cfg = _spot_alone(_cfg(batch=2, inner=256, load=ckpt), block, **(objective or {}))
    hp = cfg['Calc']['Calculation']['Hedging_Problem']
    if held:
        hp['Liabilities']['FloatingEnergyDeal']['PLAT_JUL29']['Payments']['Items'][0]['Volume'] = 1.0
        hp['Evaluator']['Total_Position_Schedule'] = [{'Step': 0, 'Min_Total': -25, 'Max_Total': -25}]
    hp['Solver'].update({'Active_Hedge_Indices': [2], 'DiffV2_Stepper_Rollout': 'Yes',
                         'DiffV2_Decision_Curve_Dump': path, **solver})
    result = _run(cfg)
    return pd.read_csv(path), result


def _free(cfg, **solver):
    """`_spot_alone`'s `cfg` on `DRIFTING`, trading the October leg alone against the whole
    liability, no optimism bound on its unfitted value."""
    cfg = _spot_alone(cfg, DRIFTING)
    cfg['Calc']['Calculation']['Hedging_Problem']['Solver'].update(
        {'Active_Hedge_Indices': [2], 'DiffV2_Max_Value_Optimism': 1.0e9, **solver})
    return cfg


def _semideviations(mean, risk):
    """Each decision's downside semideviation of the held book's continuation: its one rung's score
    in a held roll's dump at `DiffV2_Risk_Kappa` 0 less its score in one at 1."""
    return (mean['curve_pre'] - risk['curve_pre']).to_numpy()


@pytest.fixture(scope='module')
def held_risk(tmp_path_factory):
    """`(checkpoint, dir, rolls)`: the checkpoint `_spot_alone` saves on `CONSTANT_VOL` and its held
    rolls there, `(dump, result)` each, at `DiffV2_Risk_Kappa` 0 and at 1 over a risk horizon of one
    and of five decisions."""
    d = tmp_path_factory.mktemp('risk_horizon')
    ckpt = str(d / 'value_fn.pt')
    _run(_spot_alone(_cfg(batch=8, inner=8, save=ckpt), CONSTANT_VOL))
    rolls = {name: _risk_roll(ckpt, str(d / f'{name}.csv'), CONSTANT_VOL, **solver)
             for name, solver in (('mean', {}), ('one', {'DiffV2_Risk_Kappa': 1.0}),
                                  ('five', {'DiffV2_Risk_Kappa': 1.0, 'DiffV2_Risk_Horizon': 5}))}
    return ckpt, d, rolls


def test_the_risk_horizon_s_semideviation_grows_as_its_root_on_a_constant_variance(held_risk):
    """`DiffV2_Risk_Horizon` k scores a candidate by the mean of its next decision's continuation less
    kappa times the downside semideviation of the one k decisions on, the book held, capped at the
    terminal T, which on a spot of one variance a market day is sqrt(k) times the one-step one, so
    kappa / sqrt(k) chooses the books kappa does. On `CONSTANT_VOL`, every decision a market day and
    T one, the held book's ratio at k = 5 is sqrt(min(5, d)), d a decision's market days to T: it
    reads 0.996 of it over the 81 decisions (0.066 a decision), within 3%; and with the spot drifting
    at 50% a year, the free roll against the whole liability at kappa 0.6 holds the October book of
    kappa 0.6 / sqrt(5) at k = 5, in the stepper roll on 90% of its 81 decisions and in the verdict
    on 93% of its 117 rows, neither below 85%. 20 s with the fixture.

    Killing mutations: the window's inner paths drawn on a Sobol stream, which hands a path nearly
    one point on every row (2.19); the risk term read off the one-step draws (0.46, the roll's books
    agreeing on 28%); the landing one decision short (0.89); the verdict's `_decide` handed no risk
    fork (its books agreeing on 50%)."""
    ckpt, d, rolls = held_risk
    one = _semideviations(rolls['mean'][0], rolls['one'][0])
    five = _semideviations(rolls['mean'][0], rolls['five'][0])
    days = np.arange(len(one), 0, -1)
    ratio = five / one / np.sqrt(np.minimum(5, days))
    assert len(ratio) == 81 and abs(ratio.mean() - 1.0) < 0.03, (len(ratio), ratio.mean())
    diagnostics = [_risk_roll(ckpt, str(d / f'free_{k}.csv'), DRIFTING, held=False, **solver)[1]
                   .evaluation_summary['diagnostics'] for k, solver in enumerate(SCALED)]
    for track in ('stepper_verdict', 'verdict'):
        nets = [np.array([sum(q) for q in diag[track]['greedy_q_traj']]) for diag in diagnostics]
        assert np.mean(nets[0] == nets[1]) >= 0.85, (track, nets[0].tolist(), nets[1].tolist())


def test_a_variance_above_its_mean_holds_the_risk_horizon_below_its_root(held_risk, tmp_path):
    """The risk term's semideviation over k decisions follows the spot's variance forecast, so a
    variance reverting from above its mean prices a held book's risk below sqrt(k) times its next
    decision's, which the constant-variance rule cannot. On `HIGH_VOL` the held book's ratio at
    k = 10 is the forecast sqrt(sum_{j<min(10, d)} h_{m+j} / h_m), m a decision's market days from
    the first and d to T, at the first ten decisions 0.71 to 0.85 of sqrt(10): it reads 0.99 of the
    forecast there, 0.76 of sqrt(10), and 0.99 of it over the 81 (0.060 a decision), the first ten
    within 8% and below 0.85 sqrt(10), the 81 within 4%. 11 s with the fixture.

    Killing mutations: the k-decision semideviation taken as sqrt(k) times the one-step one, the
    constant-variance rule (1.30 of the forecast at the first ten); the window's inner paths drawn
    on a Sobol stream (3.18)."""
    ckpt = held_risk[0]
    mean, one, ten = [
        _risk_roll(ckpt, str(tmp_path / f'{name}.csv'), HIGH_VOL, **solver)[0]
        for name, solver in (('mean', {}), ('one', {'DiffV2_Risk_Kappa': 1.0}),
                             ('ten', {'DiffV2_Risk_Kappa': 1.0, 'DiffV2_Risk_Horizon': 10}))]
    ratio = _semideviations(mean, ten) / _semideviations(mean, one)
    n = len(ratio)
    h = HBAR + REVERSION ** np.arange(n) * (HIGH_VOL['H0'] - HBAR)
    forecast = np.sqrt([h[m:min(m + 10, n)].sum() / h[m] for m in range(n)])
    first = slice(0, 10)
    assert abs((ratio / forecast)[first].mean() - 1.0) < 0.08, (ratio / forecast)[first]
    assert (ratio / np.sqrt(10.0))[first].mean() < 0.85, (ratio / np.sqrt(10.0))[first]
    assert abs((ratio / forecast).mean() - 1.0) < 0.04, (ratio / forecast).mean()


def test_a_risk_horizon_past_the_terminal_lands_on_it(held_risk, tmp_path):
    """A decision's k-th successor past the terminal is the terminal T, the last live liability mark,
    where the continuation is the terminal utility. On `CONSTANT_VOL` the held roll at a risk horizon
    of 10,000 decisions dumps the roll at n, the decisions from the first, whose n-th successor is
    T, to the bit; its last decision, whose successor is T, scores on the one-step fork as a horizon
    of one does, to the bit; and each decision's semideviation is sqrt(d) times the one-step one, d
    its market days to T: 0.994 of it over the 81 (0.067 a decision), within 3%. 15 s with the
    fixture.

    Killing mutations: the landing capped at the last decision row (the last decision forks to its
    own row, a semideviation of 0, the ratio there NaN and 0.71 at the decision before); the landing
    one decision short (the roll at 10,000 no longer the roll at n); the window's inner paths drawn
    on a Sobol stream (6.8)."""
    ckpt, _, rolls = held_risk
    n = len(rolls['one'][1].bundle.decision_steps)
    far, at_n = [_risk_roll(ckpt, str(tmp_path / f'{k}.csv'), CONSTANT_VOL, DiffV2_Risk_Kappa=1.0,
                            DiffV2_Risk_Horizon=k)[0] for k in (10_000, n)]
    assert far.equals(at_n), far.compare(at_n)
    assert far['curve_pre'].iloc[-1] == rolls['one'][0]['curve_pre'].iloc[-1]
    ratio = _semideviations(rolls['mean'][0], far) / _semideviations(rolls['mean'][0], rolls['one'][0])
    ratio = ratio / np.sqrt(np.arange(n, 0, -1))
    assert abs(ratio.mean() - 1.0) < 0.03 and ratio[-1] == 1.0, (ratio.mean(), ratio[-1])


def _huber(x, aversion=2.5, knee=1.0):
    """The Huber utility of `x` in units of the scale, linear gains and a quadratic loss arm to
    `knee`, linear beyond it."""
    loss = np.maximum(-x, 0.0)
    return x - np.where(loss <= knee, aversion * loss ** 2,
                        aversion * knee ** 2 + 2.0 * aversion * knee * (loss - knee))


def test_a_running_reward_over_the_risk_horizon_sums_its_decisions(tmp_path):
    """Under `Running_Wealth` the risk horizon's score holds the reward of each decision's move,
    summed over the decisions the book is held across as the objective the DP fits sums it, not the
    reward of the whole move. On `CONSTANT_VOL` under the default Huber shape on the move (aversion
    2.5, its loss arm quadratic to a knee of the scale) the held book's moves are near-iid
    Student-t(30), their sd s in units of the scale read off each decision's one-step fork (0.12 to
    0.13), so its k = 5 semideviation is that of a sum of min(5, d) rewards: by Monte Carlo at each
    decision's s, 0.96 sqrt(min(5, d)) times the one-step one, and the roll's ratio reads 0.997 of
    it over the 81 decisions (0.080 a decision), within 4%. 15 s.

    Killing mutation: one reward over the whole move (`_landed` rewarding the landing's wealth
    against the decision's alone), 1.38 of it."""
    ckpt = str(tmp_path / 'value_fn.pt')
    _run(_spot_alone(_cfg(batch=8, inner=8, save=ckpt), CONSTANT_VOL, **RUNNING))
    mean, one, five = [
        _risk_roll(ckpt, str(tmp_path / f'{name}.csv'), CONSTANT_VOL, objective=RUNNING, **solver)[0]
        for name, solver in (('mean', {}), ('one', {'DiffV2_Risk_Kappa': 1.0}),
                             ('five', {'DiffV2_Risk_Kappa': 1.0, 'DiffV2_Risk_Horizon': 5}))]
    ratio = _semideviations(mean, five) / _semideviations(mean, one)
    # each decision's move, 25 contracts of 50 ounces short of the October leg and an ounce owed
    s = np.sqrt(625.0 * mean['var_dFnet'] - 50.0 * mean['cov_dFnet_dL'] + mean['sd_dL'] ** 2) / 1.0e5
    z = np.random.default_rng(0).standard_t(30.0, (2 ** 17, 5)) * np.sqrt(28.0 / 30.0)

    def semideviation(c):
        return np.sqrt((np.minimum(c - c.mean(), 0.0) ** 2).mean())

    days = np.minimum(5, np.arange(len(ratio), 0, -1))
    oracle = np.array([semideviation(_huber(x * z[:, :m]).sum(1)) / semideviation(_huber(x * z[:, 0]))
                       for x, m in zip(s.to_numpy(), days)])
    assert len(ratio) == 81 and abs((ratio / oracle).mean() - 1.0) < 0.04, (
        (ratio / oracle).mean(), (oracle / np.sqrt(days)).mean(), s.min(), s.max())


def test_a_risk_horizon_leaves_the_held_out_world_alone():
    """The risk horizon changes how a candidate is scored, not the world it is scored on: a training
    stream, a fit batch and a held-out one, at kappa 0.5 rolls the same held-out world at a horizon
    of one and of five, its outer paths and its one-step forks, so its no-hedge and textbook tracks
    and its in-simulation drift reading agree to the bit. 5 s.

    Killing mutations: the risk horizon's forks drawn on the run's streams (no
    `CMC_State.own_stream`): the no-hedge wT_mean -177,070 at k = 1 and +351,232 at k = 5; their
    generator their own and the quasi stream the run's: the one-step forks' drift reading -0.497 at
    k = 1 and -0.176 at k = 5."""
    tracks = []
    for k in (1, 5):
        cfg = _cfg(batch=16, inner=8, seed=7)
        cfg['Calc']['Calculation']['Hedging_Problem']['Solver'].update(
            {'DiffV2_Risk_Kappa': 0.5, 'DiffV2_Risk_Horizon': k, 'Run_Textbook_Benchmark': 'Yes'})
        verdict = _run(cfg).evaluation_summary['diagnostics']['verdict']
        tracks.append({name: verdict[name] for name in ('nohedge', 'textbook', 'validation_drift_z',
                                                        'drift_null_sigma')})
    assert tracks[0] == tracks[1], tracks


def test_the_sweep_s_labels_score_the_risk_horizon():
    """The label's argmax and the label horizon's greedy steps rank by the risk term over the risk
    horizon as the roll does: training with no fit on `DRIFTING` at `DiffV2_Label_Horizon` 2, kappa
    0.6 / sqrt(5) at k = 5 labels each swept step at kappa 0.6's book at k = 1, its mean over the fit
    batch's paths, the same on 81% of the 81 steps and 1.4 contracts apart on average, not below
    60% nor beyond 5; and its label, the next decision's greedy book held to the one after, 0.018
    apart on average, within 0.05. 15 s.

    Killing mutations: `_fit_step`'s `_decide` handed no risk fork (the books the same on 28%, 31.6
    contracts apart); `_horizon_correction`'s `_decide` handed none (the labels 0.095 apart)."""
    rows = [_run(_free(_cfg(batch=16, inner=64), DiffV2_Label_Horizon=2, **solver))
            .evaluation_summary['diagnostics']['per_t'] for solver in SCALED]
    books = [np.array([r['q_star_mean'][2] for r in per_t]) for per_t in rows]
    labels = [np.array([r['Y_mean'] for r in per_t]) for per_t in rows]
    gap = np.abs(books[0] - books[1])
    assert np.mean(gap < 1e-6) >= 0.6 and gap.mean() < 5.0, (books[0].tolist(), books[1].tolist())
    assert np.abs(labels[0] - labels[1]).mean() < 0.05, np.abs(labels[0] - labels[1]).mean()


def test_a_decision_is_a_row_of_market_time_on_a_coarse_grid():
    """A decision is a row of market time, one carrying a business day of the job's calendar since
    the row before (`utils.market_days`, the clock the laws step on), whatever the grid's spacing:
    on the fixture rolled every second day from Friday 10 April a Sunday row carries no market day
    and decides nothing, a Saturday row carries its Friday and decides. 1 s.

    Killing mutation: the decision rows read off the calendar's business days (`is_on_offset`),
    which leave out every Saturday row, so its Thursday's book is held over two market days."""
    cfg = jsonlib.load(open(FIXTURE))
    cfg['Calc']['Calculation'].update(Time_Grid='0d 2d(2d)', Batch_Size=2)
    world = _run(cfg).bundle
    dates = world.scenario_dates[world.initial_time_index:]
    market = [len(pd.date_range(dates[t - 1] + pd.Timedelta(days=1), dates[t], freq=world.business_day))
              for t in range(1, world.last_live_mtm_index)]
    assert world.decision_steps == [0] + [t for t, m in enumerate(market, 1) if m], world.decision_steps
    days = {dates[t].day_name() for t in world.decision_steps[1:]}
    assert 'Saturday' in days and 'Sunday' not in days, days


#: The fixture's basis reverting to a slow mean of lambda 0.95, and the GARCH(1,1) its innovation
#: variance can run on.
SLOW_BASIS = {'A': 0.05, 'Phi': 0.8, 'Nu': 5.0, 'Sigma': 4.0, 'Slow_Mean_Lambda': 0.95, 'Mu_0': -30.0}
OWN_GARCH = {'G_Omega': 0.8, 'G_Alpha': 0.1, 'G_Beta': 0.85, 'Sig2_0': 25.0}
LAST_STEP = 116


def _band_state(path, slow=True, garch=True, save=None, load=None):
    """`_cfg` on two paths a batch replaying the observed scenario `path` from the calibrated t0,
    swept at its last decision step alone, its basis `SLOW_BASIS`, off its slow mean unless `slow`
    and its own GARCH unless `garch`; with `load`, the frozen eval."""
    cfg = _cfg(batch=4, save=save, load=load)
    calc = cfg['Calc']['Calculation']
    calc['Observed_Scenario'] = path
    hp = calc['Hedging_Problem']
    hp['Randomize_Initial_State'] = 'No'
    hp['Solver']['T_Min'] = LAST_STEP
    basis = dict(SLOW_BASIS, **(OWN_GARCH if garch else {}), **({} if slow else {'Slow_Mean_Lambda': 0.0}))
    cfg['Calc']['MergeMarketData']['ExplicitMarketData']['Price Models'] = {
        'BasisLinkedSpotModel.PLATINUM_LME.LME_CME': basis}
    return cfg


def test_the_value_net_reads_the_band_deviation_the_world_declares(tmp_path, caplog):
    """The value net reads, before the basis level, its deviation from the slow level the next step
    reverts to where the basis has a slow mean and, under its own GARCH, its log variance. The
    fixture trained on one replayed path and swept at its last decision step, so the saved frame's
    market mean is that step's state to the bit: on a basis with a slow mean on its own GARCH the
    market dimension is 9 where the same document on a basis with neither reads 7, the frame is the
    plain one with two columns inserted before the level (-30.722), the first b - mu at the
    replay's published row of the step to the bit (-1.879) and the second log sigma^2 there within
    1e-6 (2.0838, the frame's float32); under `DiffV2_Returns_State` the deviation and the level
    are one fraction of the t0 spot and log sigma^2 is itself; and the plain checkpoint loaded into
    the slow mean's world off the GARCH, one column wider, is refused by the shape check naming 'md'
    (7 against 8). 10 s.

    Killing mutations: mu revealed in place of b - mu (-28.843); the published mean read a row
    early (-1.978) or late (the fork's two-row grid out of bounds); log sigma^2 dropped (8 against
    9); the level revealed first (-30.722 in the deviation's column); log sigma^2 taken for a
    fraction of the spot under the returns state (0.0010); the deviation revealed without a slow
    mean (the plain world 8)."""
    rng = np.random.default_rng(3)
    spot = 2055.0 + np.cumsum(rng.normal(0.0, 25.0, 160))
    basis = -30.0 + 6.0 * np.sin(0.3 * np.arange(160)) + rng.normal(0.0, 1.0, 160)
    path = str(tmp_path / 'replay.npz')
    np.savez(path, **{'CommodityPrice.PLATINUM_LME': spot, 'ObservedBasis.PLATINUM_LME.LME_CME': basis})
    hidden, shown = str(tmp_path / 'value_fn.pt'), str(tmp_path / 'value_fn_band.pt')
    plain = _run(_band_state(path, slow=False, garch=False, save=hidden)).evaluation_summary['diagnostics']
    with caplog.at_level(logging.DEBUG):
        band = _run(_band_state(path, save=shown)).evaluation_summary['diagnostics']
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith('BASIS_REPLAY'))
    replay = ast.literal_eval(line.split(' ', 2)[2])
    before, after = (torch.load(c, map_location='cpu', weights_only=False)['m_mean'].tolist() for c in (hidden, shown))
    level = np.float32(basis[LAST_STEP])
    col = before.index(float(level))
    deviation = float(level - np.float32(replay['mu'][LAST_STEP]))
    assert band['market_dim'] == plain['market_dim'] + 2, (band['market_dim'], plain['market_dim'])
    assert after[:col] + after[col + 2:] == before and after[col] == deviation, (after[col], deviation)
    assert abs(after[col + 1] - np.log(replay['sig2'][LAST_STEP])) < 1e-6, after[col + 1]
    returns = _band_state(path, save=str(tmp_path / 'value_fn_returns.pt'))
    returns['Calc']['Calculation']['Hedging_Problem']['Solver']['DiffV2_Returns_State'] = 'Yes'
    _run(returns)
    fraction = torch.load(str(tmp_path / 'value_fn_returns.pt'), map_location='cpu', weights_only=False)['m_mean']
    assert abs(fraction[col] * after[col + 2] / (after[col] * fraction[col + 2]) - 1.0) < 1e-5, fraction
    assert abs(fraction[col + 1] - after[col + 1]) < 1e-6, fraction
    with pytest.raises(ValueError, match=re.escape(
            f"checkpoint mismatch on 'md': {hidden} saved {plain['market_dim']} vs this run {plain['market_dim'] + 1}")):
        _run(_band_state(path, garch=False, load=hidden))


def test_a_garch_basis_without_a_slow_mean_reveals_its_log_variance_and_no_deviation(tmp_path, caplog):
    """A basis on its own GARCH with no slow mean shows the value net its log sigma^2 before its level
    and no deviation: one column more than the same document's basis with neither, inserted before
    the level, the replay's published log sigma^2 at the swept step within 1e-6, every other column
    the plain frame's to the bit. 2 s.

    Killing mutation: the deviation revealed whatever the mean (a b - mu column off a mean the law
    never publishes: the replay raises a KeyError on `basis_mu`)."""
    rng = np.random.default_rng(3)
    spot = 2055.0 + np.cumsum(rng.normal(0.0, 25.0, 160))
    basis = -30.0 + 6.0 * np.sin(0.3 * np.arange(160)) + rng.normal(0.0, 1.0, 160)
    path = str(tmp_path / 'replay.npz')
    np.savez(path, **{'CommodityPrice.PLATINUM_LME': spot, 'ObservedBasis.PLATINUM_LME.LME_CME': basis})
    plain_ck, garch_ck = str(tmp_path / 'plain.pt'), str(tmp_path / 'garch.pt')
    plain = _run(_band_state(path, slow=False, garch=False, save=plain_ck))
    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        garch = _run(_band_state(path, slow=False, garch=True, save=garch_ck))
    md = [r.evaluation_summary['diagnostics']['market_dim'] for r in (plain, garch)]
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith('BASIS_REPLAY'))
    replay = ast.literal_eval(line.split(' ', 2)[2])
    before, after = (torch.load(c, map_location='cpu', weights_only=False)['m_mean'].tolist() for c in (plain_ck, garch_ck))
    col = before.index(float(np.float32(basis[LAST_STEP])))
    assert md[1] == md[0] + 1 and 'mu' not in replay, (md, list(replay))
    assert after[:col] + after[col + 1:] == before, (after, before)
    assert abs(after[col] - np.log(replay['sig2'][LAST_STEP])) < 1e-6, (after[col], replay['sig2'][LAST_STEP])


#: A frictional LogWealth solve on the returns state, its basis a `Band_Mixture` revealing its band
#: state: a 2 $/oz dead band with a 0.2 pull around the slow mean, 10% stress days at four times the
#: quiet sd.
FRICTIONAL = {'DiffV2_Returns_State': 'Yes', 'DiffV2_Position_State': 'Yes', 'DiffV2_Book_Gradient_Label': 'Yes',
              'DiffV2_Bank_Along_Ladder': 'Yes', 'DiffV2_Risk_Kappa': 0.2}
BAND_BASIS = {'A': 0.01, 'Phi': 0.9, 'Nu': 0.0, 'Sigma': 1.5, 'Reversion_Model': 'Band_Mixture', 'Band_Kappa': 2.0,
              'Band_Beta': 0.2, 'Mix_Q_Stress': 0.1, 'Mix_Sigma_Quiet': 1.0, 'Mix_Sigma_Stress': 4.0,
              'Mix_Stay_Stress': 0.8, 'Mix_P0_Stress': 0.1, 'Slow_Mean_Lambda': 0.95, 'Mu_0': -30.0}


def _state_labels(caplog, returns=False, save=None, band=False):
    """`_cfg` swept over its last eight decision steps at 200 fit iterations, its basis `SLOW_BASIS`
    on `OWN_GARCH`, saving its value fn to `save`; under `band` the `FRICTIONAL` solve on 96 paths of
    seed 11 with `BAND_BASIS` over its last six, the July average running to its payment date so
    the last decision step still hedges a fixing: each fitted step's `(slope, label)` bank means per
    market column,
    from the debug line."""
    cfg = _cfg(seed=11, batch=96) if band else _cfg(seed=7, save=save)
    hp = cfg['Calc']['Calculation']['Hedging_Problem']
    solver = hp['Solver']
    solver['T_Min'], solver['DiffV2_Fit_Iters'] = LAST_STEP - (7 if band else 11), 200
    if returns:
        solver['DiffV2_Returns_State'] = 'Yes'
    basis = dict(SLOW_BASIS, **OWN_GARCH)
    if band:
        solver.update(FRICTIONAL)
        hp['Objective'] = {'Object': 'LogWealth', 'Utility_Scale_Mode': 'conditional_sim'}
        hp['Evaluator'].update(Max_Trade_Per_Step=10.0, Calendar_Spread_Bps=3.0)
        basis = BAND_BASIS
        pay = hp['Liabilities']['FloatingEnergyDeal']['PLAT_JUL29']['Payments']['Items'][0]
        pay['Period_End'] = pay['FX_Period_End'] = pay['Payment_Date']
    cfg['Calc']['MergeMarketData']['ExplicitMarketData']['Price Models'] = {
        'BasisLinkedSpotModel.PLATINUM_LME.LME_CME': basis}
    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        _run(cfg)
    rows = (r.getMessage().split(' ', 2) for r in caplog.records if r.getMessage().startswith('STATE_LABEL '))
    return {int(t): (np.array(v['slope']), np.array(v['label']))
            for t, v in ((t, ast.literal_eval(body)) for _, t, body in rows)}


def test_every_revealed_column_is_labelled_in_the_coordinates_the_net_reads(caplog, tmp_path):
    """The gradient fork is seeded from leaves at the basis's revealed deviation D = b - mu and log
    sigma^2, mu the level leaf less D, so each column's label is its slope at the others fixed: D's
    -dY/dmu, the level's dY/db + dY/dmu. On the fixture under the band (D the 7th market column, log
    sigma^2 the 8th, the level the 9th), over the eight swept decision steps, Friday's forking to
    Monday among them: the fitted slopes in D and log sigma^2 sum within 0.3% and 2.2% of their
    labels, and in the net's standardised coordinates every column's within 2.8% of the largest
    label (bound 15%: the twin loss meets a label in the bank mean of 24 states beside the value
    term, not exactly). At the last step, whose continuation is the closed-form utility, the level's
    label is 0.11% of D's (bound 1%): shifting
    the level and its slow mean together leaves the move's law unchanged; and under
    `DiffV2_Returns_State`, whose basis price columns are fractions of the 2,055 t0 spot, D's and the
    level's are 2,055 times the plain ones and log sigma^2's the plain one. 11 s.

    Killing mutations: the leaves not made (D's label 0); mu seeded off the outer level rather than
    the level leaf (the level's label at the last step D's, 16% off in standardised slope); the
    seeds detached from their leaves (D's label 0); the returns state's map dropped (ratio 1), or
    read on log sigma^2 too (its ratio 2,055); the
    wealth step differencing the fork's t+1 marks against the outer t marks, which carry no
    gradient (the level's label at the last step 5.0 times D's, a price column's standardised
    label -0.045 against a slope of 0.0033)."""
    ckpt = str(tmp_path / 'value_fn.pt')
    on, rs = _state_labels(caplog, save=ckpt), _state_labels(caplog, returns=True)
    slope_on, label_on = (sum(on[t][i] for t in on) for i in (0, 1))
    for col in (6, 7):
        assert label_on[col] > 0.0 and abs(slope_on[col] / label_on[col] - 1.0) < 0.15,             (col, slope_on[col], label_on[col])
    m_std = np.asarray(torch.load(ckpt, map_location='cpu', weights_only=False)['m_std'])
    z_err, z_label = np.abs(slope_on - label_on) * m_std, np.abs(label_on) * m_std
    assert z_err.max() < 0.15 * z_label.max(), (z_err, z_label)
    last = on[LAST_STEP][1]
    assert abs(last[8]) < 0.01 * last[6], last
    assert np.allclose(rs[LAST_STEP][1][6:9], np.array([2055.0, 1.0, 2055.0]) * last[6:9], rtol=1e-5, atol=0.0),         (rs[LAST_STEP][1], last)


def test_a_world_stating_the_folded_switches_off_runs_the_folded_law(caplog):
    """`Trading_Clock`, `Reveal_Band_State` and `DiffV2_State_Gradient_Label` are folded, so a stored
    world and solver stating each 'No' run the one law: the labelled-column document's state labels
    and market dimension are the same to the bit with the three keys stated off as without them. 2 s.

    Killing mutation: an off arm read again (the reveal's, its deviation column dropped where a block
    states `Reveal_Band_State` 'No': market dimension 8 against 9)."""
    def labels(off):
        cfg = _cfg(seed=7)
        hp = cfg['Calc']['Calculation']['Hedging_Problem']
        hp['Solver'].update(T_Min=LAST_STEP - 3, DiffV2_Fit_Iters=20)
        basis = dict(SLOW_BASIS, **OWN_GARCH)
        if off:
            hp['Solver']['DiffV2_State_Gradient_Label'] = 'No'
            basis.update(Trading_Clock='No', Reveal_Band_State='No')
        cfg['Calc']['MergeMarketData']['ExplicitMarketData']['Price Models'] = {
            'BasisLinkedSpotModel.PLATINUM_LME.LME_CME': basis}
        caplog.clear()
        with caplog.at_level(logging.DEBUG):
            md = _run(cfg).evaluation_summary['diagnostics']['market_dim']
        return md, [r.getMessage() for r in caplog.records if r.getMessage().startswith('STATE_LABEL ')]
    on, off = labels(False), labels(True)
    assert on[0] == off[0] and len(on[1]) >= 2 and on[1] == off[1], (on[0], off[0], len(on[1]))


def test_the_wealth_step_differences_the_fork_s_own_futures_marks(caplog):
    """The gradient fork's futures move is its t+1 mark
    less its own t mark, both moving with the leaves, so a column the marks read (the carry, the
    price, the basis level) is labelled with the move's slope rather than the t+1 mark's. On the
    `FRICTIONAL` solve with `BAND_BASIS` (D the 7th of 8 market columns, the level the 8th), at the
    last step, whose continuation is the closed-form utility and whose book hedges the last fixing,
    the level's label is 0.010% of D's label summed over the six swept steps (0.025% on the card,
    bound 1%): moving the level and its slow mean together leaves the move unchanged; and D's fitted
    slope sums within 1.1% of its label (2.2% on the card, bound 15%). Seeds 1 to 13 read at most
    0.065% and 5.2% on the CPU, 0.026% and 16% on the card (its seed 9 past the bound). 7 s.

    Killing mutation: the wealth step differencing the fork's t+1 marks against the outer t marks,
    which carry no gradient (the level's label at the last step 19% of D's summed label, 39% and
    48% on seeds 2 and 4; 35%, 36% and 31% on the card)."""
    run = _state_labels(caplog, band=True)
    last = run[LAST_STEP][1]
    slope, label = (sum(run[t][i] for t in run) for i in (0, 1))
    assert label[6] > 0.0 and abs(last[7]) < 0.01 * label[6], (last[7], label[6])
    assert abs(slope[6] / label[6] - 1.0) < 0.15, (slope[6], label[6])
