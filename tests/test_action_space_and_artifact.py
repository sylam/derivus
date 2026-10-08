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

Configs are built in code from the canonical fixture TEMPLATE (never edited) — small batch /
shallow sweep so the gates run fast. Every run, training and loaded evaluation alike, is a
document through `load_json` + `run_job`.
"""
import json as jsonlib
import os

import pytest
import torch

import derivus as rf
from derivus.hedge_runtime import per_contract_kappa

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
        solver['DiffV2_Load_Value_Fn'] = [load]
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


# --------------------------------------------------------------------------------------------
# (ii) initial inventory — first-step turnover measured from the OPENING book q0, not flat
# --------------------------------------------------------------------------------------------
def test_first_step_turnover_measured_from_opening_book():
    """Killing mutation: the textbook's entry charged from flat, `n_star - 0` for `n_star - q0`."""
    positions = {'PL_APR_2026': -1, 'PL_JUL_2026': -4, 'PL_OCT_2026': 0}
    flat = _run(_cfg(seed=11, positions=None, benchmarks=True))
    book = _run(_cfg(seed=11, positions=positions, benchmarks=True))

    fb, bb = flat.evaluation_summary, book.evaluation_summary

    # The value function is POSITION-FREE: the opening book must NOT move any V_0/u track.
    assert bb['comparison']['textbook']['v0_mean'] == fb['comparison']['textbook']['v0_mean']
    assert bb['comparison']['HindsightDpSolver']['v0_mean'] \
        == fb['comparison']['HindsightDpSolver']['v0_mean']
    for k in ('u_mean', 'wT_mean', 'wT_p5', 'wT_cvar5'):
        assert bb['diagnostics']['verdict']['greedy'][k] \
            == fb['diagnostics']['verdict']['greedy'][k], f'greedy {k} moved with the book'

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
    runtime, bundle = book.runtime, book.bundle
    hist = bundle.initial_time_index
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
    inside the fence at every step, short early and long late; the artifact stamps that schedule;
    a frozen eval in the same corridor reports the trained V_0 to the bit, and one under another
    corridor or none is refused by name - the frozen value fn learned other reachable wealth.

    Killing mutations: the corridor filter skipped in `grid_at` (the greedy book breaches it); the
    stamp written as None; a mismatched corridor admitted as if trained corridor-free."""
    ckpt = str(tmp_path / 'value_fn_corr.pt')
    train = _run(_cfg(seed=7, save=ckpt, schedule=SIGN_CROSSING, symmetric=True))
    diag = train.evaluation_summary['diagnostics']
    t0 = int(diag['root_t'])
    saw_short = saw_long = False
    for i, book in enumerate(diag['verdict']['greedy_q_traj']):
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


def test_a_corridor_free_policy_rolls_in_any_corridor(free):
    """A policy trained corridor-FREE has the widest wealth support, so rolling it INSIDE a
    Total_Position_Schedule only restricts to a learned subset - allowed, and the eval reports
    the trained V_0.

    Killing mutation: the corridor-free arm refused like a mismatch."""
    ckpt, train = free
    assert train.policy_artifact['total_position_schedule'] is None
    fence = [{'Step': 0, 'Min_Total': -50, 'Max_Total': -20}]
    assert _v0(_run(_cfg(seed=7, load=ckpt, schedule=fence))) == _v0(train)
