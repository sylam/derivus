"""The two INDEPENDENT answers to one measured pathology: with a fixed utility scale, terminal
wealth drifts to |x| of 2.6-6.1 by mid-horizon, so the objective is affine on ~89% of decision days
and the argmax is bang-bang on a curve that is a straight line.

`Objective.Reference_Mode='Running_Wealth'` rebases WHAT the utility is applied to — the DAY's
wealth increment, O(one day's move) on every day — so the recursion's reward is per-step:
Y = u(W1 - W_t) + E[C_{t+1}], with C_{T_dec} = 0 and (the ruling that follows from it) C_t = A_t
alone, because u(W) is the terminal reward evaluated early and there is no terminal reward of the
LEVEL left to anchor on.

`Objective.Utility_Scale_Mode='conditional_sim'` rebases the SCALE instead — a per-decision-step
knee c_t, measured once at the frame lock as the cross-sectional dispersion of the flat-book wealth
at t and floored at a fraction of its terminal entry, so x = (W - R)/c_t is a z-score on every day.

They are separate experiments and refuse each other by name: the increment already carries a
per-step scale of its own, and rescaling it per step again is a third objective nobody asked for.

THE FIXTURE, hand-computed, is the pathology in miniature (`_decide_world`): huber c = 1,
a = 2.5, delta = 1, a wealth of 20 (twenty knees out), a short-only ladder {-2,-1,0} and an inner
draw dF = (-0.4, +0.2) whose mean pays to be short.

    q        0        -1       -2
    Fixed    20.5     20.6     20.7      <- exactly affine; the corner wins because nothing bends
    Running   0.5      0.55     0.5      <- curved; the interior rung wins

THE SCHEDULE IS A FRAME, not provenance metadata. It is saved beside the nets and the
standardization stats, and a LOADED run consumes the saved one at every decision step rather than
re-measuring from the evaluation world — the per-step generalization of `BundleStepper`'s
`mirror_scale=False` contract, which already says the checkpoint's `c` is part of the value
function. An ensemble goes one further: each member's continuation is read at ITS OWN knees,
because a member's schedule rides its frame exactly as its z-stats do, and averaging two
disagreeing knees would evaluate at a scale nobody fitted.

KILL MATRIX - every mutant applied to the source, this module RUN, the death recorded, the mutant
reverted. 24 mutants, 24 deaths, none survived (M1 survived a first pass and its gate was rebuilt
around it: the fixture had left the LOCKED frame sitting on the runtime, so the lock passed
whatever `_bind` did — the later batch has to mirror its own frame on first, as a real one does).

M1-M11 are the build's own. M12-M24 are the ADVERSARIAL REVIEW's: thirteen findings, every one
reproduced, and the gate that would have caught each is named beside it. Two of them are the ones
worth reading twice — the mode was UNREACHABLE from the deck it was added to (the template's ruled
literal collided with the engine's own refusal), and neither `Reference_Mode` nor
`Utility_Scale_Floor_Frac` was validated anywhere, so a typo ran the default objective to a clean
finish and a zero floor ran to `V_0 = nan`. `values=[...]` on a declared field is a UI hint; the
engine reads nothing from it.

| mutant | died at |
| ------ | ------- |
| M1 the schedule recomputed per batch (`_bind` drops the re-assert) | `test_the_schedule_is_locked_at_the_warmup_batch` |
| M2 the wrap ignores `t` (c_0 / the scalar everywhere) | `test_the_scale_schedule_is_read_per_step`, `test_the_continuation_anchor_reads_the_step_knee` |
| M3 the floor dropped | `test_the_schedule_is_the_floored_cross_sectional_dispersion` |
| M4 the B2 label keeps the terminal-only u | `test_the_running_label_is_the_increment_plus_the_continuation`, `test_the_terminal_label_is_the_increment_with_its_unwind_inside` |
| M5 the B2 decide ranks terminal while the label ranks the increment | `test_the_running_ranking_orders_by_the_increment` |
| M6 the `reference_mode` stamp dropped from `_policy_artifact` | `test_the_artifact_stamps_both_dials` |
| M7 the `utility_scale_schedule` stamp dropped | `test_the_artifact_stamps_both_dials` |
| M8 the anchor kept under Running_Wealth (C_t = u(W) + A_t) | `test_the_running_ranking_orders_by_the_increment`, `test_the_running_label_is_the_increment_plus_the_continuation` |
| M9 the replay re-measures from the eval bundle (the restore drops the schedule) | `test_a_loaded_run_decides_on_the_SAVED_schedule_not_the_eval_worlds` |
| M10 the ensemble reads every member at member 0's schedule | `test_each_ensemble_member_reads_its_own_step_knee` |
| M11 the load check reads the MEASURED schedule instead of the declared mode | `test_the_artifact_stamps_both_dials`, `test_a_pre_feature_checkpoint_reads_as_fixed_and_unscheduled` |

| M12 the deck never clears the template's `Utility_Scale_Explicit` | `test_the_deck_clears_the_template_scale_under_conditional_sim` |
| M13 the `Reference_Mode` refusal deleted | `test_a_misspelled_reference_mode_is_refused` (×5) |
| M14 the floor-fraction refusal deleted | `test_a_non_positive_floor_fraction_is_refused` (×2) |
| M15 the non-positive-`c` guard deleted (back to `c is None`) | `test_a_non_positive_scale_fails_loud_where_c_is_resolved` |
| M16 the terminal early-return back above the ensemble branch | `test_each_ensemble_member_reads_its_own_step_knee` |
| M17 the dump keeps the wealth LEVEL under `Running_Wealth` | `test_the_dump_places_the_utility_where_the_ranking_operates` |
| M18 the dump reports the knee at `t`, not the `t+1` the rungs use | `test_the_dump_reports_the_knee_the_rungs_divide_by` |
| M19 the frame-lock dose log dropped | `test_the_frame_lock_logs_the_dose_the_schedule_imposes` |
| M20 the verdict indexes the increment by its departure step | `test_the_verdict_indexes_the_increment_by_the_step_it_arrives_at` |
| M21 the terminal unwind is its own utility term again | `test_the_verdicts_net_objective_charges_the_unwind_inside_the_last_step` |
| M22 the restored scalar comes from a stale scalar, not the terminal knee | `test_an_ensembles_scalar_is_the_mean_of_its_members_terminal_knees` |
| M23 the schedule key hashed into every frame stamp again | `test_a_schedule_free_frame_keeps_its_pre_feature_stamp` |
| M24 the proximity-prior refusal deleted | `test_the_proximity_prior_refuses_the_increment_objective` |

M11 is the one the deployment path turns on and is worth naming: a frozen roll is a single path,
so it measures no schedule of its own, and a check that compared MEASURED schedules would refuse
every roll of a scheduled policy — the one run the feature is ultimately for. The run side of that
check is the declared MODE; the values come from the checkpoint.

HARNESS. `_decide` / `_fit_step` / `_fit_from_labels` / `_verdict` / `_bind` /
`_check_load_provenance` / `_policy_artifact` run as UNBOUND functions against a minimal stand-in
solver - the `SimpleNamespace` pattern of `test_position_state` and `test_wealth_free_value`, where
the fake enumerates exactly what the seam touches so a new read is visible as a new attribute. The
residual net is a CONSTANT, so every continuation is `anchor + 0.5` and the arithmetic below is the
definition rather than a paraphrase of the code.
"""
import ast
import copy
import hashlib
import inspect
import json
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from derivus import calculation, hedge_bundle, hedge_runtime, hedge_solver
from derivus.hedge_bundle import (Bundle, _utility_local_curvature, _utility_scale,
                                  _utility_wrap_signed)
from derivus.hedge_solver import DiffSolver, HedgeActionSpace, _DiffV2Residual

REL = 1e-5         # float32: the transform is evaluated at the default dtype
CAP = 4.0          # Evaluator.Total_Position_Abs_Limit — the Q_max p is measured in
FEE = 0.25         # Transaction_Cost_Per_Unit (zero spread ⇒ kappa IS the fee)
CONST = 0.5        # the residual net's constant output, so C = anchor + CONST everywhere
W0 = 20.0          # twenty knees out: the drifted wealth the pathology is about
MOVES = (-0.4, 0.2)   # the inner draw; E[dF] < 0, so being short pays
MD = 1


def _objective(**over):
    """A huber objective at c = 1, so every dollar below IS a c-unit and the hand arithmetic in
    the docstring is readable. `reference_mode` and the schedule default OFF."""
    obj = {"object": "asymmetricutility_huber", "utility_scale": 1.0, "reference_wealth": 0.0,
           "huber_aversion": 2.5, "huber_delta": 1.0, "up_aversion": 0.0, "up_knee": 0.15,
           "reference_mode": "fixed", "utility_scale_schedule": None}
    obj.update(over)
    return obj


def _runtime(levels=3, lo=-2.0, hi=0.0, force_flat=False, fee=0.0, **objective):
    hedges = ["A"]
    return {
        "names": {"hedges": hedges},
        "tradables": {r: {"contract_size": 1.0} for r in hedges},
        "portfolio_state": {"positions": {}},
        "objective": _objective(**objective),
        "solver": {"training_action_grid_levels_per_axis": levels,
                   "training_action_chunk_size": 64, "active_hedge_indices": None},
        "accounting": {
            "position_limits": {r: {"min_position": lo, "max_position": hi} for r in hedges},
            "total_position_abs_limit": CAP,
            "total_position_schedule": None,
            "allocation_weights": None,
            "calendar_spread_bps": None,
            "max_trade_per_step": 0.0,
            "decision_deadband_sigma": 0.0,
            "force_flat_at_end": force_flat,
            "transaction_cost_per_unit": fee,
            "bid_offer_spread_bps": 0.0,
            "bid_offer_spread_spec": None,
        },
    }


class _Const(torch.nn.Module):
    """A residual that returns the same number for every row: the continuation is then
    `anchor + CONST`, so a gate reads the ANCHOR (which is what both dials move) and not a fit."""

    def forward(self, x):
        return torch.full((x.shape[0],), CONST)


def _solver(runtime, running_wealth=False, position_state=False, T_dec=3, n_steps=4, B=1):
    """The minimal stand-in every seam runs on."""
    aspace = HedgeActionSpace(runtime, torch.device("cpu"))
    hedges = runtime["names"]["hedges"]
    s = types.SimpleNamespace(
        runtime=runtime, aspace=aspace, chunk=64, risk_kappa=0.0, churn_lambda=0.0,
        position_state=position_state, wealth_free=False,
        running_wealth=running_wealth,
        utility_scale=float(runtime["objective"]["utility_scale"]),
        utility_scale_schedule=runtime["objective"].get("utility_scale_schedule"),
        # the declared MODE, which is what a load is checked against — a one-path roll measures
        # no schedule and still rolls a scheduled checkpoint
        scheduled_scale=runtime["objective"].get("utility_scale_mode") == "conditional_sim",
        force_flat=runtime["accounting"]["force_flat_at_end"],
        t_min=0, T_dec=T_dec, total_abs_limit=aspace.total_abs_limit,
        hedges=hedges, n_hedge=len(hedges), contract_size=aspace.contract_size,
        device=torch.device("cpu"), B_outer=B, cost_aware=False,
        active=aspace.active, n_active=aspace.n_active, q_lo=aspace.q_lo, q_hi=aspace.q_hi,
        # Flat marks: per_contract_kappa is then the Transaction_Cost_Per_Unit alone.
        tradables_sim={r: torch.zeros(n_steps, B) for r in hedges},
        liability_sim=torch.zeros(n_steps, B),
        m_mean=torch.zeros(MD), m_std=torch.ones(MD),
        w_mean=torch.tensor(0.0), w_std=torch.tensor(1.0),
        a_bounds=[None] * T_dec, _ensemble=None,
    )
    s.log_ratio = False                      # the diff-form arm; LogWealth has its own gates
    s.loaded = None
    s.w_floor = 1.0
    for name in ("_u", "_u_step", "_capital", "_member_anchor", "_wealth_step",
                 "_unwind_kappa", "_calendar_kappa",
                 "_reposition_charge", "_score_actions", "_decision_curve_row",
                 "_standardize", "_continuation", "_decide", "_project_leaf_grads", "_bind",
                 "_replication_hedge", "_check_action_universe", "_check_calendar_spread"):
        setattr(s, name, types.MethodType(getattr(DiffSolver, name), s))
    return s


def _decide_world(s, t=0, B=1, Bi=2, q_prev=None):
    """Run the real `_decide` from W0 on the two-draw world of the module docstring."""
    dF = torch.tensor(MOVES).reshape(1, Bi, 1).expand(B, Bi, 1)
    return DiffSolver._decide(
        s, [_Const()] * s.T_dec, torch.zeros(B, Bi, MD), dF, torch.zeros(B, Bi),
        torch.full((B,), W0), t, q_prev=q_prev)


def _u1(x, obj=None):
    """u of one scalar, straight from the definition — the reference the gates assert against."""
    return float(_utility_wrap_signed(torch.tensor([float(x)]),
                                      {"objective": obj or _objective()}))


def _fit_label(s, t=0, B=1, Bi=2, q_prev_val=0.0, live=None):
    """Drive the real `_fit_step` and hand back the label it regressed on. The grid is one row, so
    the argmax is FORCED and the only thing a dial can move is the target itself."""
    n_h = len(s.hedges)
    live = torch.ones(n_h) if live is None else live
    dF = torch.tensor(MOVES).reshape(1, Bi, 1).expand(B, Bi, n_h).contiguous()
    zero_in = (dF, torch.zeros(B, Bi), torch.zeros(B, Bi, MD), torch.zeros(B, MD), live)
    s.bundle = types.SimpleNamespace(inner_mc_grad=lambda _t: {
        "F_t1": {r: torch.tensor(MOVES).expand(B, Bi).contiguous() for r in s.hedges},
        "L_t1": torch.zeros(B, Bi), "L_t": torch.zeros(B, Bi),
        "market_t1": torch.zeros(B, Bi, MD), "market_t": torch.zeros(B, MD),
        "state_t_leaves": {}, "state_t_leaf_widths": [],
    })
    label = {}
    s._fit_from_labels = lambda nets, W0_bank, market0, Y, gW, g_market, tt, q_star, p_bank: \
        label.update(Y=Y, q_star=q_star) or label
    DiffSolver._fit_step(s, [_Const()] * s.T_dec, {t: torch.full((B,), W0)}, t, zero_in,
                         {t: torch.full((B, n_h), q_prev_val)})
    return label


# --- (a) both dials off ⇒ bit-identical labels, decisions, verdict ----------------------------
def _without_the_new_keys(runtime):
    """The same runtime as a job written BEFORE either dial existed: the objective simply has no
    `reference_mode` and no schedule key. Both are absent-means-off by construction, which is what
    makes the comparison below a bit-identity claim rather than a same-branch tautology."""
    rt = copy.deepcopy(runtime)
    rt["objective"].pop("reference_mode")
    rt["objective"].pop("utility_scale_schedule")
    return rt


def test_off_decides_and_labels_exactly_as_the_pre_change_solver():
    """(a) A runtime carrying the dials switched off and one that has never heard of them produce
    the same decision, the same value and the same label — `torch.equal`, not `allclose`."""
    on, off = _runtime(), _without_the_new_keys(_runtime())
    q_a, v_a = _decide_world(_solver(on))
    q_b, v_b = _decide_world(_solver(off))
    assert torch.equal(q_a, q_b) and torch.equal(v_a, v_b)
    assert torch.equal(_fit_label(_solver(on))["Y"], _fit_label(_solver(off))["Y"])
    # ...and the decision IS the pathology: the affine curve's corner (see the module docstring).
    assert float(q_a[0, 0]) == -2.0
    assert float(v_a[0]) == pytest.approx(20.2 + CONST, rel=REL)


def _fit_sha(runtime, md=MD, B=32):
    """sha1 of the parameters `_fit_from_labels` lands on, from one seed and one label set."""
    torch.manual_seed(0)
    net = _DiffV2Residual(md + 1, hidden=8)
    market0, W0_bank = torch.randn(B, md), torch.randn(B)
    Y, gW, g_market = torch.randn(B), torch.randn(B), torch.randn(B, md)
    s = _solver(runtime)
    s.use_adv, s.prox, s.fit_iters, s.fit_tol, s.lr = True, 0.0, 25, 0.0, 1e-2
    s.cfg, s._opts, s._bounds_frozen, s._breaches = {}, {}, False, []
    s.T_dec, s.a_bounds = 1, [None]
    DiffSolver._fit_from_labels(s, [net], W0_bank, market0, Y, gW, g_market, 0,
                                torch.zeros(B, 1), None)
    return hashlib.sha1(b"".join(p.detach().numpy().tobytes()
                                 for p in net.parameters())).hexdigest()


def test_off_fits_bit_identical_parameters():
    """(a) The fit too: the advantage decomposition, the twin loss and the trust region land on
    the SAME parameters whether or not the objective carries the (off) dials. A dial that leaked
    into the anchor — the one thing `Running_Wealth` removes — would move every parameter."""
    assert _fit_sha(_runtime()) == _fit_sha(_without_the_new_keys(_runtime()))
    # vacuity: the same harness under the dial ON must NOT match, or the gate proves nothing
    assert _fit_sha(_runtime()) != _fit_sha_running()


def _fit_sha_running():
    rt = _runtime(reference_mode="running_wealth")
    torch.manual_seed(0)
    net = _DiffV2Residual(MD + 1, hidden=8)
    market0, W0_bank = torch.randn(32, MD), torch.randn(32)
    Y, gW, g_market = torch.randn(32), torch.randn(32), torch.randn(32, MD)
    s = _solver(rt, running_wealth=True)
    s.use_adv, s.prox, s.fit_iters, s.fit_tol, s.lr = True, 0.0, 25, 0.0, 1e-2
    s.cfg, s._opts, s._bounds_frozen, s._breaches = {}, {}, False, []
    s.T_dec, s.a_bounds = 1, [None]
    DiffSolver._fit_from_labels(s, [net], W0_bank, market0, Y, gW, g_market, 0,
                                torch.zeros(32, 1), None)
    return hashlib.sha1(b"".join(p.detach().numpy().tobytes()
                                 for p in net.parameters())).hexdigest()


def _verdict(running):
    """A three-step verdict roll on a one-row ladder (q = -1 forced) with a +2/step outer move, so
    every wealth increment is -2 and the whole rollout is hand-computable."""
    rt = _runtime(levels=1, lo=-1.0, hi=-1.0, fee=0.5)
    s = _solver(rt, running_wealth=running, T_dec=3, n_steps=4, B=4)
    s.tradables_sim = {r: (2.0 * torch.arange(4, dtype=torch.float32))[:, None].expand(4, 4)
                       for r in s.hedges}
    s.liability_sim = torch.zeros(4, 4)
    s.B_outer = 4
    inner = {t: (torch.zeros(4, 2, 1), torch.zeros(4, 2), torch.zeros(4, 2, MD),
                 torch.zeros(4, MD), torch.ones(1)) for t in range(3)}
    return DiffSolver._verdict(s, [_Const()] * 3, inner, list(range(3)))


def test_the_verdict_reports_the_step_sum_only_under_running_wealth():
    """(a)+(b) The verdict's `u_mean` is the objective the DP maximised, so it follows the dial:
    a terminal u(W_T) with it off, the SUM of the rollout's per-step utilities with it on. The
    `*_net` sum charges each step's own turnover into that step's increment, which is what a
    per-step objective means by a cost, and `u_mean_is_step_sum` states which regime produced the
    number — no figure says it otherwise. `wT_mean` is raw dollars either way."""
    off, on = _verdict(False), _verdict(True)
    assert off["u_mean_is_step_sum"] is False and on["u_mean_is_step_sum"] is True
    assert off["greedy"]["wT_mean"] == on["greedy"]["wT_mean"] == pytest.approx(-6.0)
    assert off["greedy"]["u_mean"] == pytest.approx(_u1(-6.0), rel=REL)            # terminal: -33.5
    assert on["greedy"]["u_mean"] == pytest.approx(3.0 * _u1(-2.0), rel=REL)       # step sum: -28.5
    # the net column: the entry from flat costs 0.5, and it lands in the step that paid it
    assert on["greedy"]["u_mean_net"] == pytest.approx(_u1(-2.5) + 2.0 * _u1(-2.0), rel=REL)
    assert off["greedy"]["wT_mean_net"] == on["greedy"]["wT_mean_net"] == pytest.approx(-6.5)


# --- (b) Running_Wealth: the increment IS the reward, at the ranking and at the label ----------
def test_the_running_ranking_orders_by_the_increment():
    """(b) THE gate the dial exists for. At a wealth twenty knees out the terminal ranking is
    exactly affine in the position — 20.5, 20.6, 20.7 — so the deepest rung wins on a curve that
    never bends. Rebased on the day's increment the same world, the same draws and the same
    residual pick the INTERIOR rung, because u is curved where the increment lives.

    Killed by keeping the u(W) anchor under the dial (the affine term returns and the corner wins
    again), and by a `_decide` that ranks terminal wealth while the label ranks increments."""
    q_off, v_off = _decide_world(_solver(_runtime()))
    q_on, v_on = _decide_world(_solver(_runtime(reference_mode="running_wealth"),
                                       running_wealth=True))
    assert float(q_off[0, 0]) == -2.0, "the fixture must reproduce the bang-bang corner"
    assert float(q_on[0, 0]) == -1.0, "the increment ranking must bend to an interior rung"
    inc = {q: 0.5 * sum(_u1(q * d) for d in MOVES) for q in (-2.0, -1.0, 0.0)}
    assert float(v_on[0]) == pytest.approx(max(inc.values()) + CONST, rel=REL)
    assert float(v_on[0]) == pytest.approx(inc[-1.0] + CONST, rel=REL)
    # the anchor is gone, so the whole ranking is the increment's (plus one constant)
    assert inc[-1.0] > inc[-2.0] and inc[-1.0] > inc[0.0]


def test_the_running_label_is_the_increment_plus_the_continuation():
    """(b) The label the fit regresses on is `u(W1 - W_t) + E[C_{t+1}]` — the same expression
    `_decide` ranked with, which is the whole point of stating it once per seam. One grid row, so
    the argmax is forced and the label is the target alone.

    Killed by a label that keeps the terminal-only u: it would read `mean u(W1) + CONST`, which
    the second assertion pins as the OFF value."""
    rt = _runtime(levels=1, lo=-2.0, hi=-2.0)
    on = _fit_label(_solver(_runtime(levels=1, lo=-2.0, hi=-2.0,
                                     reference_mode="running_wealth"), running_wealth=True))
    off = _fit_label(_solver(rt))
    assert float(on["Y"][0]) == pytest.approx(
        0.5 * sum(_u1(-2.0 * d) for d in MOVES) + CONST, rel=REL)
    assert float(off["Y"][0]) == pytest.approx(
        0.5 * sum(_u1(W0 - 2.0 * d) for d in MOVES) + CONST, rel=REL)


def test_the_terminal_label_is_the_increment_with_its_unwind_inside():
    """(b) At the last decision C_{T_dec} = 0, so the label is exactly `u(dW_T)` — and the
    forced-flat unwind is INSIDE that increment, because the charge is subtracted from W1 before
    the increment is taken. Stated by turning `Force_Flat_At_End` off and watching the label move
    by exactly the liquidation.

    Killed by a terminal that still returns u(W) (the label would carry the level), and by an
    unwind charged outside the increment."""
    def label(force_flat):
        rt = _runtime(levels=1, lo=-2.0, hi=-2.0, force_flat=force_flat, fee=FEE,
                      reference_mode="running_wealth")
        return float(_fit_label(_solver(rt, running_wealth=True, position_state=True,
                                        T_dec=1))["Y"][0])

    #     charge = |dq|*fee + (unwind) |q|*fee  = 0.5 + 0.5   /   0.5
    flat = 0.5 * sum(_u1(-2.0 * d - 1.0) for d in MOVES)
    free = 0.5 * sum(_u1(-2.0 * d - 0.5) for d in MOVES)
    assert label(True) == pytest.approx(flat, rel=REL)
    assert label(False) == pytest.approx(free, rel=REL)
    assert flat < free, "the fixture must make the unwind visible"


def test_the_running_value_carries_no_terminal_anchor():
    """(b) The decomposition ruling, read directly: C_t = A_t alone under the dial, and the
    terminal continuation is identically 0 rather than u(W). Without it the recursion would carry
    a bounded function of a level that earns nothing all the way down."""
    s = _solver(_runtime(reference_mode="running_wealth"), running_wealth=True)
    nets = [_Const()] * s.T_dec
    W = torch.tensor([W0, -3.0])
    assert torch.equal(DiffSolver._continuation(s, nets, torch.zeros(2, MD), W, s.T_dec, None),
                       torch.zeros(2))
    assert torch.equal(DiffSolver._continuation(s, nets, torch.zeros(2, MD), W, 0, None),
                       torch.full((2,), CONST))
    off = _solver(_runtime())
    assert torch.equal(DiffSolver._continuation(off, nets, torch.zeros(2, MD), W, off.T_dec, None),
                       _utility_wrap_signed(W, off.runtime))


# --- (c) conditional_sim: a knee per decision step --------------------------------------------
#: A liability path whose cross-sectional dispersion RISES then TAPERS (the remaining exposure
#: falls as fixings print), and whose first row is common to every path — which is why the floor
#: exists at all.
LIAB = torch.tensor([[0.0, 0.0, 0.0, 0.0],
                     [1.0, -1.0, 2.0, -2.0],
                     [3.0, -3.0, 6.0, -6.0],
                     [2.0, -2.0, 4.0, -4.0],
                     [0.0, 0.0, 0.0, 0.0]])          # the post-settlement clean-exit row


def _bundle(liability=LIAB, floor=0.05):
    b = Bundle()
    b.liability_mtm = liability
    b.last_live_mtm_index = int(liability.shape[0]) - 2
    return b, {"utility_scale_floor_frac": floor}


def test_the_schedule_is_the_floored_cross_sectional_dispersion():
    """(c) One entry per step a continuation is ever read at — decisions 0..T_dec-1 plus the
    terminal mark — each the cross-sectional std of the flat-book wealth at that step, and the
    whole series floored at a fraction of its terminal entry.

    Killed by dropping the floor: every path shares L_0, so c_0 would be 0 and x = (W-R)/c_0
    unbounded on the first decision."""
    b, obj = _bundle()
    sched = b._conditional_sim_schedule(obj)
    raw = LIAB[:4].std(dim=-1)
    assert len(sched) == b.last_live_mtm_index + 1 == 4
    floor = 0.05 * float(raw[-1])
    assert sched == pytest.approx([max(float(v), floor) for v in raw])
    assert float(raw[0]) == 0.0 and sched[0] == pytest.approx(floor), "the floor must bind at t=0"
    assert sched[2] > sched[3], "non-increasing where the remaining exposure tapers"


def test_the_schedule_needs_a_cross_section_and_a_liability():
    """(c) One outer path has no cross-section to measure: that is the frozen ROLL, which takes
    its schedule from the checkpoint, so `None` is the honest answer rather than a NaN. No
    liability at all is a configuration error and fails loud."""
    assert _bundle(LIAB[:, :1])[0]._conditional_sim_schedule({"utility_scale_floor_frac": 0.05}) \
        is None
    b = Bundle()
    with pytest.raises(ValueError, match="conditional_sim"):
        b._conditional_sim_schedule({"utility_scale_floor_frac": 0.05})


def test_the_scalar_is_the_schedules_terminal_entry():
    """(c) The scale a TERMINAL read takes (the verdict's u(W_T), both benchmark tracks) is the
    schedule's own last entry, so the two agree at the terminal by construction instead of
    stepping across a discontinuity nobody declared."""
    b, _ = _bundle()
    rt = {"objective": {"object": "asymmetricutility_huber", "utility_scale_mode": "conditional_sim",
                        "utility_scale_floor_frac": 0.05}}
    c = b._resolve_utility_scale(rt)
    assert c == pytest.approx(float(LIAB[:4].std(dim=-1)[-1]))
    assert b.utility_scale_schedule[-1] == pytest.approx(c)
    assert _utility_scale(dict(rt["objective"], utility_scale=c,
                               utility_scale_schedule=b.utility_scale_schedule)) == pytest.approx(c)


def test_an_explicit_scale_beside_the_schedule_is_refused():
    """(d) The schedule measures the LEVEL of c as well as its shape, so a literal dollar value is
    a second source for one number. Refused by name rather than silently picking a winner."""
    b, _ = _bundle()
    with pytest.raises(ValueError, match="Utility_Scale_Explicit"):
        b._resolve_utility_scale({"objective": {
            "object": "asymmetricutility_huber", "utility_scale_mode": "conditional_sim",
            "utility_scale_floor_frac": 0.05, "utility_scale_explicit": 3.0e4}})


def _scheduled_runtime(schedule=(0.5, 2.0, 8.0, 8.0)):
    return _runtime(utility_scale=schedule[-1], utility_scale_schedule=schedule)


def test_the_scale_schedule_is_read_per_step():
    """(c) The wrap consumes c_t: the SAME dollar wealth is a different point on the utility at
    two steps whose knees differ, and the local risk aversion in force differs with it (the
    curvature probe — a knee eight times wider is an aversion eight times flatter). A caller that
    names no step reads the scalar, which is the terminal knee.

    Killed by a wrap that ignores `t`: all three readings would collapse onto one."""
    obj = _scheduled_runtime()["objective"]
    rt = {"objective": obj}
    W = torch.tensor([-0.4])
    u0, u2, term = (float(_utility_wrap_signed(W, rt, 0)), float(_utility_wrap_signed(W, rt, 2)),
                    float(_utility_wrap_signed(W, rt)))
    assert u0 != u2 and term == pytest.approx(float(_utility_wrap_signed(W, rt, 3)))
    # each is the flat-scale transform at ITS OWN c — the schedule is a scale, not a new shape
    for t, c in enumerate(obj["utility_scale_schedule"]):
        flat = {"objective": dict(obj, utility_scale=c, utility_scale_schedule=None)}
        assert float(_utility_wrap_signed(W, rt, t)) == pytest.approx(
            float(_utility_wrap_signed(W, flat)))
    ara0 = _utility_local_curvature(-0.1, rt, 0)[1]
    ara2 = _utility_local_curvature(-0.1, rt, 2)[1]
    assert ara0 > ara2 > 0.0, "a wider knee applies a flatter aversion at the same dollar loss"


def test_the_continuation_anchor_reads_the_step_knee():
    """(c) …and the DP consumes it where it decides: the anchor of C_t at the same wealth differs
    between two steps, so the ranking a decision bends by is the knee in force THAT day.

    Killed by a `_continuation` that hands the wrap no step (c_0 or the scalar everywhere)."""
    rt = _scheduled_runtime()
    s = _solver(rt, T_dec=3)
    nets, W = [_Const()] * 3, torch.tensor([-0.4])
    c0 = float(DiffSolver._continuation(s, nets, torch.zeros(1, MD), W, 0, None))
    c2 = float(DiffSolver._continuation(s, nets, torch.zeros(1, MD), W, 2, None))
    assert c0 != c2
    assert c0 == pytest.approx(float(_utility_wrap_signed(W, rt, 0)) + CONST, rel=REL)
    assert c2 == pytest.approx(float(_utility_wrap_signed(W, rt, 2)) + CONST, rel=REL)


# --- (d) provenance: both dials are part of what a checkpoint IS -------------------------------
def _artifact(s):
    s.m_mean = s.m_std = torch.ones(1)
    s.w_mean = s.w_std = torch.tensor(1.0)
    s.a_bounds = [None]
    s.active = [0]
    s._config_hash = lambda: "h"
    s._frame_stamp = lambda: "f"
    return DiffSolver._policy_artifact(s, [], 1, 8, 0.0, [0.0], 0.0)


def test_the_artifact_stamps_both_dials():
    """(d) Neither dial moves a single tensor SHAPE — a value fn trained on increments and one
    trained on terminal wealth have identical nets, and so do a scheduled and a flat knee — so a
    mismatched load has no symptom at all beyond a quietly wrong policy. Both are stamped, both
    round-trip, and both are refused by name in either direction.

    Killed by dropping either stamp: the round trip below fails on the run's own checkpoint."""
    for running in (False, True):
        s = _solver(_runtime(reference_mode="running_wealth" if running else "fixed"),
                    running_wealth=running)
        ck = _artifact(s)
        assert ck["reference_mode"] == ("running_wealth" if running else "fixed")
        assert DiffSolver._check_load_provenance(s, ck, "<ck>", 1) == ck["solver_version"]
        other = _solver(_runtime(), running_wealth=not running)
        with pytest.raises(ValueError, match="Reference_Mode"):
            DiffSolver._check_load_provenance(other, ck, "<ck>", 1)

    modes = {True: "conditional_sim", False: "vol_scaled_notional"}
    for scheduled in (False, True):
        s = _solver(_runtime(utility_scale_mode=modes[scheduled],
                             utility_scale_schedule=(1.0, 2.0, 3.0) if scheduled else None))
        ck = _artifact(s)
        assert bool(ck["utility_scale_schedule"]) is scheduled
        assert DiffSolver._check_load_provenance(s, ck, "<ck>", 1) == ck["solver_version"]
        other = _solver(_runtime(utility_scale_mode=modes[not scheduled]))
        with pytest.raises(ValueError, match="Utility_Scale_Mode"):
            DiffSolver._check_load_provenance(other, ck, "<ck>", 1)
    # THE DEPLOYMENT CASE: a frozen roll is one path, so it measures no schedule of its own and
    # takes the checkpoint's. The check is the declared MODE — reading the measured schedule here
    # would refuse every roll of a scheduled policy.
    roll = _solver(_runtime(utility_scale_mode="conditional_sim"))
    assert roll.utility_scale_schedule is None and roll.scheduled_scale is True
    assert DiffSolver._check_load_provenance(
        roll, _artifact(_solver(_runtime(utility_scale_mode="conditional_sim",
                                         utility_scale_schedule=(1.0, 2.0, 3.0)))),
        "<ck>", 1) is not None


def test_a_pre_feature_checkpoint_reads_as_fixed_and_unscheduled():
    """(d) A checkpoint written before either dial carries neither stamp: it is terminal-utility
    at a flat knee, so it loads under the defaults and is refused under either dial."""
    old = _artifact(_solver(_runtime()))
    del old["reference_mode"], old["utility_scale_schedule"]
    assert DiffSolver._check_load_provenance(_solver(_runtime()), old, "<ck>", 1) \
        == old["solver_version"]
    with pytest.raises(ValueError, match="Reference_Mode"):
        DiffSolver._check_load_provenance(
            _solver(_runtime(), running_wealth=True), old, "<ck>", 1)
    with pytest.raises(ValueError, match="Utility_Scale_Mode"):
        DiffSolver._check_load_provenance(
            _solver(_runtime(utility_scale_mode="conditional_sim")), old, "<ck>", 1)


def test_the_two_dials_refuse_each_other():
    """(d) Two independent answers to one pathology, and their composition is a third objective:
    the increment already carries a per-step scale of its own. Refused at construction, by name,
    before a training budget is spent on it."""
    rt = _runtime()
    rt["objective"].update(reference_mode="running_wealth", utility_scale_mode="conditional_sim")
    bundle = types.SimpleNamespace(device=torch.device("cpu"), vol_sim=None)
    with pytest.raises(ValueError, match="Reference_Mode"):
        DiffSolver(bundle, rt)
    # ...and each on its own constructs (the refusal is about the pair, not about either dial)
    for over in ({"reference_mode": "running_wealth"}, {"utility_scale_mode": "conditional_sim"}):
        solo = _runtime()
        solo["objective"].update(over)
        solo["solver"]["diffv2_load_value_fn"] = "ck.pt"      # a frozen roll: nothing to measure
        solo["accounting"]["total_position_abs_limit"] = CAP
        DiffSolver(types.SimpleNamespace(
            device=torch.device("cpu"), vol_sim=None, tradables_sim={}, n_outer_steps=4,
            liability_sim=torch.zeros(4, 2), last_live_mtm_index=2, utility_scale=1.0,
            utility_scale_schedule=None), solo)


def test_training_without_a_measured_schedule_fails_loud():
    """(d) `conditional_sim` with no schedule measured means a single-path batch — which can only
    be a frozen roll. Training there would silently fall back to a flat knee, so it raises and
    names the knob that fixes it."""
    rt = _runtime()
    rt["objective"]["utility_scale_mode"] = "conditional_sim"
    bundle = types.SimpleNamespace(
        device=torch.device("cpu"), vol_sim=None, tradables_sim={}, n_outer_steps=4,
        liability_sim=torch.zeros(4, 1), last_live_mtm_index=2, utility_scale=1.0,
        utility_scale_schedule=None)
    with pytest.raises(ValueError, match="Batch_Size"):
        DiffSolver(bundle, rt)


# --- (e) the frame lock: the WARMUP batch's schedule, and no later one ------------------------
def test_the_schedule_measures_the_worlds_dispersion_not_the_injected_scatter():
    """Under Randomize_Initial_State the paths START scattered (a bank-coverage device);
    the knee must measure the dispersion the world GENERATES from those starts, or the
    schedule sits ~flat at the terminal scale and a growth objective read against it is
    linear everywhere — the trained crash roll rode the conditional mean to max long on
    exactly this. With identical starts the two spellings coincide (every other gate).
    Kills a reverted-absolute-std mutant."""
    import types as _types
    import torch as _torch
    from derivus.hedge_bundle import Bundle
    g = _torch.Generator().manual_seed(3)
    B = 4096
    starts = 1000.0 * _torch.randn(B, generator=g)          # the injected scatter
    steps = _torch.randn(4, B, generator=g)                 # the world's own dispersion ~1
    L = starts.unsqueeze(0) + _torch.cat(
        [_torch.zeros(1, B), steps.cumsum(0)]).to(_torch.float32)
    s = _types.SimpleNamespace(liability_sim=L, liability_mtm=L,
                               last_live_mtm_index=4)
    sched = Bundle._conditional_sim_schedule(
        s, {"utility_scale_floor_frac": 0.05})
    assert sched is not None
    assert float(sched[1]) < 50.0, \
        f"day-1 knee must be the world's ~1-unit dispersion, not the ~1000 scatter; " \
        f"got {float(sched[1]):.1f}"
    assert 1.5 < float(sched[-1]) < 3.0, 'terminal ~ sqrt(4) of unit steps'


def test_the_schedule_is_locked_at_the_warmup_batch():
    """(e) Every streaming batch measures its own dispersion, and letting a later one reach the
    runtime would rescale every reward the recursion has already composed. `_bind` re-asserts the
    LOCKED frame — the scalar and the schedule together, because they are one frame.

    Killed by dropping the schedule's re-assert: batch 2's knees would silently take over.

    The later batch's frame is MIRRORED onto the runtime before the bind, because that is the
    sequence — `Bundle._resolve_frame` measures and mirrors as the batch is built, and only then
    does the solver bind to it. A gate that left the locked frame sitting on the runtime would
    pass whatever `_bind` did, which is exactly how a lock gate goes blind."""
    locked = (0.5, 2.0, 8.0, 8.0)
    s = _solver(_scheduled_runtime(locked), T_dec=3)
    s.utility_scale, s.utility_scale_schedule = 8.0, locked
    later, _ = _bundle()
    later.utility_scale, later.utility_scale_schedule = 999.0, (9.0, 9.0, 9.0, 9.0)
    later.mirror_utility_scale(s.runtime)                      # batch 2 builds and mirrors...
    assert s.runtime["objective"]["utility_scale_schedule"] == (9.0, 9.0, 9.0, 9.0)
    s.vol_sim = None
    s._bind(types.SimpleNamespace(
        vol_sim=None, tradables_sim={}, n_outer_steps=4, liability_sim=torch.zeros(4, 2),
        last_live_mtm_index=3, utility_scale=999.0,
        utility_scale_schedule=(9.0, 9.0, 9.0, 9.0)))          # ...and the bind takes it back
    assert s.runtime["objective"]["utility_scale_schedule"] == locked
    assert s.runtime["objective"]["utility_scale"] == 8.0


def test_the_bundle_mirrors_the_whole_frame():
    """(e) The mirror is the other half: a stepper that re-mirrors takes the bundle's scalar AND
    its schedule, and one that does not (a frozen roll, `mirror_scale=False`) keeps the
    checkpoint's. One frame, mirrored in one place."""
    b, _ = _bundle()
    b.utility_scale, b.utility_scale_schedule = 7.0, (1.0, 2.0)
    rt = {"objective": {"utility_scale": 99.0, "utility_scale_schedule": (9.0,)}}
    b.mirror_utility_scale(rt)
    assert rt["objective"]["utility_scale"] == 7.0
    assert rt["objective"]["utility_scale_schedule"] == (1.0, 2.0)


def test_a_loaded_run_decides_on_the_SAVED_schedule_not_the_eval_worlds():
    """(e) REPLAY. The schedule is part of the value function's FRAME, not provenance metadata:
    a frozen policy must be read at the knees it was fitted against, on every decision day, and
    never at the ones this world happened to measure. That is `BundleStepper`'s `mirror_scale
    =False` contract stated per step — and a frozen eval is precisely the run where the two
    disagree, because the eval world is often the stressed one.

    The fixture makes the disagreement loud: the eval bundle measured knees 100x the trained
    ones, so a continuation read at the wrong frame is off by two orders of magnitude.

    Killed by a replay that keeps the eval world's schedule (the restore is where the eval
    bundle's measurement is discarded)."""
    trained = (0.5, 2.0, 8.0, 8.0)
    world = tuple(100.0 * c for c in trained)
    s = _solver(_scheduled_runtime(world), T_dec=3)             # the EVAL world's own frame...
    assert s.runtime["objective"]["utility_scale_schedule"] == world
    ck = dict(_artifact(_solver(_scheduled_runtime(trained))), m_mean=torch.zeros(MD),
              m_std=torch.ones(MD), w_mean=torch.tensor(0.0), w_std=torch.tensor(1.0))
    s._restore_frame = types.MethodType(DiffSolver._restore_frame, s)
    s._restore_frame([ck])                                      # ...is discarded by the load
    assert s.utility_scale_schedule == trained
    assert s.runtime["objective"]["utility_scale_schedule"] == trained
    assert s.runtime["objective"]["utility_scale"] == trained[-1]
    W = torch.tensor([-0.4])
    for t in (0, 2):
        assert float(DiffSolver._continuation(s, [_Const()] * 3, torch.zeros(1, MD), W, t, None)) \
            == pytest.approx(float(_utility_wrap_signed(W, {"objective": dict(
                s.runtime["objective"], utility_scale_schedule=trained)}, t)) + CONST, rel=REL)
    # and a later streaming batch cannot take it back (the same lock, now over a loaded frame)
    s._bind(types.SimpleNamespace(
        vol_sim=None, tradables_sim={}, n_outer_steps=4, liability_sim=torch.zeros(4, 2),
        last_live_mtm_index=3, utility_scale=999.0, utility_scale_schedule=world))
    assert s.runtime["objective"]["utility_scale_schedule"] == trained


def test_each_ensemble_member_reads_its_own_step_knee():
    """(e) An ensemble argmax averages the members' CONTINUATIONS, and a member's continuation is
    read in its own frame — its z-stats, its trust region and, under `conditional_sim`, its own
    per-step knee. Schedules are therefore never averaged or otherwise reconciled: two members
    that disagree about c_t disagree about the utility, and the honest answer is the mean of the
    two readings, not a reading at a mean nobody fitted.

    The loop runs to T_dec INCLUSIVE, and that step is the one that matters most: the terminal
    continuation is anchor-only, so it IS the whole of the last decision's ranking. An earlier
    form of this gate stopped at T_dec-1 and let a terminal early-return — which sat above the
    ensemble branch and handed every member the first one's knee — land green.

    Killed by an ensemble that uses member 0's schedule for every member: the mean below would
    collapse onto the first member's anchor."""
    a_sched, b_sched = (0.5, 2.0, 8.0, 8.0), (4.0, 4.0, 4.0, 4.0)
    s = _solver(_scheduled_runtime(a_sched), T_dec=3)

    class _Zero(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0])

    frame = (torch.zeros(MD), torch.ones(MD), torch.tensor(0.0), torch.tensor(1.0), None)
    s._ensemble = [([_Zero()] * 3, *frame, a_sched), ([_Zero()] * 3, *frame, b_sched)]
    W = torch.tensor([-0.4])
    for t in (0, 1, 2, 3):                      # ...T_dec included: the anchor-only terminal
        each = [float(_utility_wrap_signed(W, {"objective": dict(
            s.runtime["objective"], utility_scale_schedule=sched)}, t))
            for sched in (a_sched, b_sched)]
        got = float(DiffSolver._continuation(s, None, torch.zeros(1, MD), W, t, None))
        assert got == pytest.approx(sum(each) / 2.0, rel=REL)
        if a_sched[t] != b_sched[t]:
            assert got != pytest.approx(each[0], rel=REL), 'member 0 governed both readings'


# === the adversarial review's findings, each with the gate that would have caught it ===========
#
# Every gate below exists because a shipped defect got past the ones above it.


def _deck_args(**over):
    """The deck's `args` as `apply_objective_flags` reads it — every flag off unless named."""
    flags = ("huber_aversion", "huber_delta", "utility_scale", "objective", "cara_gamma",
             "reference_wealth", "up_aversion", "up_knee", "reference_mode",
             "utility_scale_mode", "utility_scale_floor")
    return types.SimpleNamespace(**dict(dict.fromkeys(flags), **over))


#: The authored platinum job's Objective, as `--deal-template` carries it: a RULED literal scale.
TEMPLATE_OBJECTIVE = {"Object": "AsymmetricUtility_Huber", "Huber_Aversion": 6.0,
                      "Huber_Delta": 1.0, "Utility_Scale_Explicit": 30000.0}


def _scale_runtime(objective):
    """The deck's JSON Objective as the normalized runtime the bundle resolves against."""
    return {"objective": {
        "object": "asymmetricutility_huber",
        "utility_scale_mode": str(objective.get("Utility_Scale_Mode",
                                                "vol_scaled_notional")).lower(),
        "utility_scale_explicit": objective.get("Utility_Scale_Explicit"),
        "utility_scale_floor_frac": objective.get("Utility_Scale_Floor_Frac", 0.05)}}


def test_the_deck_clears_the_template_scale_under_conditional_sim():
    """FINDING 1 (HIGH): the mode was UNREACHABLE from the deck it was added to. Every authored
    job here carries a ruled `Utility_Scale_Explicit`, `build_deal_config` only ever SET that key,
    and the engine refuses the pair — so `--utility-scale-mode conditional_sim` died at the first
    bundle build, before a single fit. The flag's own help said "refuses --utility-scale", which
    reads as 'just don't pass it'; the template already had.

    The gate drives the deck's real flag application and then resolves the objective it produced
    through the real engine, which is the only way to see the two halves disagree. Killed by
    dropping the pop."""
    from experiments.plat_walk_forward_chain import apply_objective_flags

    armed = apply_objective_flags(dict(TEMPLATE_OBJECTIVE),
                                  _deck_args(utility_scale_mode="conditional_sim"))
    assert "Utility_Scale_Explicit" not in armed, "the retired literal must be cleared"
    bundle = _bundle()[0]
    assert bundle._resolve_utility_scale(_scale_runtime(armed)) > 0.0

    # ...and the un-cleared template is exactly what the engine refuses — the shipped defect.
    with pytest.raises(ValueError, match="Utility_Scale_Explicit"):
        bundle._resolve_utility_scale(_scale_runtime(
            dict(TEMPLATE_OBJECTIVE, Utility_Scale_Mode="conditional_sim")))

    # A literal the CALLER typed is left standing, so the contradiction is heard rather than
    # silently resolved in the deck's favour.
    typed = apply_objective_flags(dict(TEMPLATE_OBJECTIVE),
                                  _deck_args(utility_scale_mode="conditional_sim",
                                             utility_scale=12345.0))
    assert typed["Utility_Scale_Explicit"] == 12345.0
    with pytest.raises(ValueError, match="Utility_Scale_Explicit"):
        bundle._resolve_utility_scale(_scale_runtime(typed))
    # ...and every other mode leaves the template's ruled scale exactly where it was.
    kept = apply_objective_flags(dict(TEMPLATE_OBJECTIVE),
                                 _deck_args(utility_scale_mode="vol_scaled_notional"))
    assert kept["Utility_Scale_Explicit"] == 30000.0


def _hedge_json(**objective):
    """The smallest `Hedging_Problem` `construct_hedge_runtime` normalizes (simulate_only, so the
    solve-side validations stand aside and the objective's own are what fire)."""
    return {"Hedging_Problem": {
        "Tradable_Instruments": {"CommodityFutureDeal": {"F1": {"Currency": "USD"}}},
        "Evaluator": {},
        "Objective": dict({"Object": "AsymmetricUtility_Huber"}, **objective)}}


@pytest.mark.parametrize("spelling", ["RunningWealth", "running wealth", "Runnning_Wealth",
                                      "Running-Wealth", "true"])
def test_a_misspelled_reference_mode_is_refused(spelling):
    """FINDING 2 (HIGH): the dial was validated NOWHERE. `values=[...]` on the declared field is a
    UI hint — it picks a dropdown and nothing in the engine reads it — while the runtime key is
    consumed as `== 'running_wealth'` deep in the solver. So a near-miss meant `Fixed`, silently,
    end to end: the run completed, the checkpoint stamped `fixed`, and the provenance check agreed
    with itself. A campaign would have returned a clean null attributable to the dial rather than
    to the typo.

    Killed by dropping the refusal — every spelling below then normalizes to a quiet default."""
    with pytest.raises(ValueError, match="Reference_Mode"):
        hedge_runtime.construct_hedge_runtime(_hedge_json(Reference_Mode=spelling))


def test_the_two_reference_modes_and_the_default_are_accepted():
    """Vacuity check for the refusal above, and the case-fold contract: the boundary lowercases,
    so an authored 'Running_Wealth' and a lowercase one are one value."""
    for spelling, want in (("Fixed", "fixed"), ("Running_Wealth", "running_wealth"),
                           ("running_wealth", "running_wealth"), ("FIXED", "fixed")):
        rt = hedge_runtime.construct_hedge_runtime(_hedge_json(Reference_Mode=spelling))
        assert rt["objective"]["reference_mode"] == want
    assert hedge_runtime.construct_hedge_runtime(
        _hedge_json())["objective"]["reference_mode"] == "fixed"


@pytest.mark.parametrize("frac", [0.0, -0.05])
def test_a_non_positive_floor_fraction_is_refused(frac):
    """FINDING 3 (MED): every outer path shares L_0, so the raw dispersion at t=0 is exactly 0 —
    the floor is the ONLY thing standing between the schedule and a zero knee. At `--utility-scale
    -floor 0` the first step's c was 0, `x = (W−R)/c` was NaN, the fit spread the NaN through the
    nets and the run COMPLETED, reporting V_0 = nan."""
    with pytest.raises(ValueError, match="Utility_Scale_Floor_Frac"):
        hedge_runtime.construct_hedge_runtime(_hedge_json(Utility_Scale_Floor_Frac=frac))


def test_a_non_positive_scale_fails_loud_where_c_is_resolved():
    """FINDING 3, the other half: the guard tested `c is None` and let a zero through. It is now a
    contract at the ONE place c is resolved, so no future source of a zero (a floor, a literal, a
    degenerate measurement) can reach a reward as a NaN instead of an error.

    The reproduction, verbatim from the review: floor 0 ⇒ c_0 = 0 ⇒ u(anything, t=0) = nan."""
    b, _ = _bundle(floor=0.0)
    sched = b._conditional_sim_schedule({"utility_scale_floor_frac": 0.0})
    assert sched[0] == 0.0, "the fixture must reproduce the zero knee"
    rt = {"objective": dict(_objective(), utility_scale_schedule=sched, utility_scale=sched[-1])}
    with pytest.raises(ValueError, match="utility scale"):
        _utility_wrap_signed(torch.tensor([100.0]), rt, 0)
    with pytest.raises(ValueError, match="utility scale"):
        _utility_scale(rt["objective"], 0)
    assert float(_utility_wrap_signed(torch.tensor([100.0]), rt, 2)) != 0.0   # the rest is fine


def _curve_row(running, t=0):
    """One `_decision_curve_row` on the module's own decide-world, at B_outer = 1."""
    rt = _runtime(reference_mode="running_wealth" if running else "fixed")
    s = _solver(rt, running_wealth=running)
    Bi = len(MOVES)
    dF = torch.tensor(MOVES).reshape(1, Bi, 1)
    s._inner_step = lambda _t: (dF, torch.zeros(1, Bi), torch.zeros(1, Bi, MD),
                                torch.zeros(1, MD), torch.ones(1))
    return DiffSolver._decision_curve_row(
        s, [_Const()] * s.T_dec, t, torch.zeros(1, Bi, MD), dF, torch.zeros(1, Bi),
        torch.full((1,), W0), torch.zeros(1, 1), torch.zeros(1, 1), None, None)


def test_the_dump_places_the_utility_where_the_ranking_operates():
    """FINDING 5 (MED): the dump placed the utility at the wealth LEVEL while the objective in
    force was on the INCREMENT. At a level twenty knees out that reports `local_ARA` 0 and
    `in_knee` 0 on every row — "the ranking is affine here" — at exactly the decisions
    `Running_Wealth` has made curved, which is the one question the dump exists to answer.

    Under the dial the row is now placed on the increment (0 before the day happens, so the
    curvature is the LOSS wing's) and `u_is_increment` states the regime. Killed by reverting
    either half."""
    fixed, running = _curve_row(False), _curve_row(True)

    assert fixed["u_is_increment"] == 0 and running["u_is_increment"] == 1
    assert fixed["W"] == running["W"] == pytest.approx(W0)      # the level is a fact either way
    assert fixed["x0"] == pytest.approx(W0)                     # c = 1 in this fixture
    assert running["x0"] == 0.0
    # the review's exact finding: at the level, beyond the knee, the shape reports no aversion
    assert fixed["local_ARA"] == 0.0 and fixed["u_prime"] == pytest.approx(1.0)
    # ...while the increment sits in the loss-side knee, where the aversion is 2a/c
    assert running["local_ARA"] == pytest.approx(2.0 * 2.5, rel=1e-4)
    assert running["u_prime"] == pytest.approx(1.0, rel=1e-4)
    # P_band follows x0: the deepest rung's increments are (0.8, -0.4), so half the draws land in
    # the band, while the LEVEL's successors (20.8, 19.6) are nowhere near it
    assert fixed["P_band_deep"] == 0.0
    assert running["P_band_deep"] == pytest.approx(0.5)
    # and range_CE converts the curve's spread at the u' of the point the curve lives at
    curve = [float(v) for v in running["curve_charged"].split()]
    assert running["range_CE"] == pytest.approx(
        (max(curve) - min(curve)) / running["u_prime"], rel=1e-4)


def test_the_dump_reports_the_knee_the_rungs_divide_by():
    """FINDING 5, the `c` column: the rungs are continuations at `t+1`, so `t+1`'s knee is what
    `_utility_wrap_signed` divides them by — the comment claimed `c` was that number while the
    code read `t`. Inert under a flat scale, which is why it survived; visible the moment a
    schedule makes the two differ."""
    sched = (0.5, 2.0, 8.0, 8.0)
    s = _solver(_scheduled_runtime(sched), T_dec=3)
    Bi = len(MOVES)
    dF = torch.tensor(MOVES).reshape(1, Bi, 1)
    s._inner_step = lambda _t: (dF, torch.zeros(1, Bi), torch.zeros(1, Bi, MD),
                                torch.zeros(1, MD), torch.ones(1))
    for t in (0, 1):
        row = DiffSolver._decision_curve_row(
            s, [_Const()] * 3, t, torch.zeros(1, Bi, MD), dF, torch.zeros(1, Bi),
            torch.full((1,), W0), torch.zeros(1, 1), torch.zeros(1, 1), None, None)
        assert row["c"] == pytest.approx(sched[t + 1]), "the rungs divide by t+1's knee"
        assert row["x0"] == pytest.approx(W0 / sched[t + 1])


def test_the_frame_lock_logs_the_dose_the_schedule_imposes(caplog):
    """FINDING 6 (MED), the design ruling. The schedule owns the LEVEL of c as well as its shape,
    and the shape params are dimensionless in c — so arming the mode RE-DOSES the risk aversion by
    whatever factor the measured scale differs from the retired literal (~20x on the platinum
    book: $30k ruled against a measured terminal knee of O($5e5)). That re-levelling is INTENDED —
    the measured dispersion is the design scale and the literal was the diagnosed bug — but it
    must be visible, because a book that keeps its old `Huber_Aversion` has silently changed how
    averse it is, which is the recorded 'aversion inert' failure with the sign flipped.

    So the frame lock states the dose it imposes, in aversion per dollar. Killed by dropping the
    log line."""
    b, obj = _bundle()
    runtime = {"objective": dict(_objective(), utility_scale_mode="conditional_sim",
                                 utility_scale_floor_frac=0.05),
               "accounting": {"bid_offer_spread_spec": None, "im_funding_spread_bps": 0.0},
               "referenced_commodities": ()}
    with caplog.at_level("INFO"):
        b._resolve_frame(runtime, {})
    line = next(r.getMessage() for r in caplog.records
                if "SCHEDULE (conditional_sim)" in r.getMessage())
    assert "LEVEL" in line and "Huber_Aversion" in line
    sched = b.utility_scale_schedule
    # the dose IS the loss-wing in-knee aversion, 2a/c_t, at the first and last knee
    assert ("%.4g" % (2.0 * 2.5 / sched[0])) in line
    assert ("%.4g" % (2.0 * 2.5 / sched[-1])) in line


def test_the_verdict_indexes_the_increment_by_the_step_it_arrives_at():
    """FINDING 7 (LOW): the verdict indexed the running-wealth increment by the step it DEPARTS
    from while every other seam indexes the arrival (`_decide`, `_fit_step`, `_score_actions` all
    pass `t+1`). Numerically inert today only because the B2 x B3 refusal keeps `_u`'s step unused,
    which is the kind of load-bearing coincidence nobody remembers.

    The gate therefore holds the CONVENTION on a fake whose attributes are set past the
    constructor's refusal — a state the engine will not build, asserting the rule rather than a
    reachable configuration."""
    sched = (1.0, 2.0, 4.0, 8.0)
    rt = _runtime(levels=1, lo=-1.0, hi=-1.0, utility_scale=sched[-1],
                  utility_scale_schedule=sched)
    s = _solver(rt, running_wealth=True, T_dec=3, n_steps=4, B=4)
    s.tradables_sim = {r: (2.0 * torch.arange(4, dtype=torch.float32))[:, None].expand(4, 4)
                       for r in s.hedges}
    s.liability_sim = torch.zeros(4, 4)
    s.B_outer = 4
    inner = {t: (torch.zeros(4, 2, 1), torch.zeros(4, 2), torch.zeros(4, 2, MD),
                 torch.zeros(4, MD), torch.ones(1)) for t in range(3)}
    out = DiffSolver._verdict(s, [_Const()] * 3, inner, list(range(3)))
    arrival = sum(_u1(-2.0, dict(_objective(), utility_scale=sched[t + 1])) for t in range(3))
    departure = sum(_u1(-2.0, dict(_objective(), utility_scale=sched[t])) for t in range(3))
    assert out["greedy"]["u_mean"] == pytest.approx(arrival, rel=REL)
    assert arrival != pytest.approx(departure, rel=REL), 'the fixture must tell the two apart'


def test_the_verdicts_net_objective_charges_the_unwind_inside_the_last_step():
    """FINDING 9 (LOW): the verdict appended the terminal unwind as its OWN utility term while
    training folds it into the last increment (`_fit_step` subtracts it from W1 before taking
    u(W1 − W_t)). u is nonlinear, so the reported net objective was not the one optimised — in a
    column that exists precisely to be read net of cost."""
    rt = _runtime(levels=1, lo=-1.0, hi=-1.0, fee=0.5, force_flat=True,
                  reference_mode="running_wealth")
    s = _solver(rt, running_wealth=True, T_dec=3, n_steps=4, B=4)
    s.tradables_sim = {r: (2.0 * torch.arange(4, dtype=torch.float32))[:, None].expand(4, 4)
                       for r in s.hedges}
    s.liability_sim = torch.zeros(4, 4)
    s.B_outer = 4
    inner = {t: (torch.zeros(4, 2, 1), torch.zeros(4, 2), torch.zeros(4, 2, MD),
                 torch.zeros(4, MD), torch.ones(1)) for t in range(3)}
    out = DiffSolver._verdict(s, [_Const()] * 3, inner, list(range(3)))
    # q = -1 forced; entry from flat costs 0.5 at t=0, the unwind of |1| costs 0.5 at the terminal
    folded = _u1(-2.5) + _u1(-2.0) + _u1(-2.0 - 0.5)
    separate = _u1(-2.5) + 2.0 * _u1(-2.0) + _u1(-0.5)
    assert out["greedy"]["u_mean_net"] == pytest.approx(folded, rel=REL)
    assert folded != pytest.approx(separate, rel=REL), 'the fixture must tell the two apart'


def test_an_ensembles_scalar_is_the_mean_of_its_members_terminal_knees():
    """FINDING 8 (LOW): `_restore_frame` averaged the scalar over members but adopted member 0's
    schedule, so the documented invariant — the scalar IS the terminal read — held only by
    coincidence: the verdict's terminal stats divided by the mean while the terminal continuation
    divided by member 0's last knee. The scalar now comes from the members' terminal KNEES, which
    is the same statement their averaged terminal continuation makes.

    A saved artifact stamps the two EQUAL (the scalar it writes is the schedule's terminal entry),
    so the members below are deliberately inconsistent — a hand-built or pre-contract checkpoint —
    which is the only fixture in which 'which of the two the restore reads' is observable at all.
    That is the property worth pinning: the restore takes the knee, not a scalar beside it."""
    a, b = (0.5, 2.0, 8.0), (4.0, 4.0, 12.0)
    s = _solver(_scheduled_runtime(a), T_dec=3)
    s._restore_frame = types.MethodType(DiffSolver._restore_frame, s)
    frame = {"m_mean": torch.zeros(MD), "m_std": torch.ones(MD),
             "w_mean": torch.tensor(0.0), "w_std": torch.tensor(1.0), "utility_scale": 999.0}
    s._restore_frame([dict(frame, utility_scale_schedule=a),
                      dict(frame, utility_scale_schedule=b)])
    assert s.utility_scale == pytest.approx((a[-1] + b[-1]) / 2.0)
    assert s.utility_scale != pytest.approx(999.0), 'the stale scalar must not govern'
    assert s.utility_scale_schedule == a, 'the run-level schedule stays the primary member\'s'
    # ...and with no schedule in play the scalar is the members' own, exactly as before.
    s2 = _solver(_runtime(), T_dec=3)
    s2._restore_frame = types.MethodType(DiffSolver._restore_frame, s2)
    s2._restore_frame([dict(frame, utility_scale=10.0), dict(frame, utility_scale=20.0)])
    assert s2.utility_scale == pytest.approx(15.0) and s2.utility_scale_schedule is None


def test_a_schedule_free_frame_keeps_its_pre_feature_stamp():
    """FINDING 10 (LOW): the new key entered the hashed frame unconditionally, so every
    default-off run got a NEW frame stamp for a provably identical frame — and the stamp's own
    contract is that a different one means a different function. Omitted when absent."""
    s = _solver(_runtime(), T_dec=1)
    s.a_bounds = [(-1.0, 1.0)]
    s._frame_stamp = types.MethodType(DiffSolver._frame_stamp, s)
    pre_feature = hashlib.sha1(json.dumps({
        "utility_scale": round(float(s.runtime["objective"]["utility_scale"]), 6),
        "m_mean": [round(float(v), 6) for v in s.m_mean.tolist()],
        "m_std": [round(float(v), 6) for v in s.m_std.tolist()],
        "w_mean": round(float(s.w_mean), 6), "w_std": round(float(s.w_std), 6),
        "a_bounds": [-1.0, 1.0]}, sort_keys=True).encode()).hexdigest()
    assert s._frame_stamp() == pre_feature
    s.utility_scale_schedule = (1.0, 2.0)
    assert s._frame_stamp() != pre_feature, 'a scheduled frame IS a different frame'


def test_the_proximity_prior_refuses_the_increment_objective():
    """FINDING 12 (LOW): the successor-proximity prior penalizes the distance from A_t to its
    fitted successor, which is the right prior only while C_t ≈ C_{t+1}. Under a per-step reward
    the two differ by E[u(ΔW_t)] BY DEFINITION, so the regularizer pulls against exactly the
    offset the objective is made of. The warm START is untouched — an initial point is not a
    penalty."""
    rt = _runtime()
    rt["objective"]["reference_mode"] = "running_wealth"
    rt["solver"]["diffv2_temporal_proximity"] = 0.5
    bundle = types.SimpleNamespace(device=torch.device("cpu"), vol_sim=None)
    with pytest.raises(ValueError, match="DiffV2_Temporal_Proximity"):
        DiffSolver(bundle, rt)


# --- the declared defaults are the engine's, all the way down ---------------------------------
@pytest.mark.parametrize("json_key,runtime_key", [
    ("Reference_Mode", "reference_mode"),
    ("Utility_Scale_Mode", "utility_scale_mode"),
    ("Utility_Scale_Floor_Frac", "utility_scale_floor_frac")])
def test_one_default_per_new_objective_knob(json_key, runtime_key):
    """`test_hmc_declared_knobs` holds the declared `F` row against the JSON boundary's fallback;
    this is the third site — a DOWNSTREAM `.get(key, const)` on the normalized runtime, which the
    boundary walk cannot see because it reads a lowercased key off a dict nobody declared. Where
    one exists it must publish the same number; where none does (the read is a subscript, or a
    one-argument `.get` whose absence means off) there is nothing to disagree with."""
    objective = next(f for f in calculation.HedgeMonteCarlo.fields if f.key == 'Hedging_Problem')
    objective = next(f for f in objective.sub_fields if f.key == 'Objective')
    declared = next(f for f in objective.sub_fields if f.key == json_key).default
    for module in (hedge_bundle, hedge_solver):
        for node in ast.walk(ast.parse(inspect.getsource(module))):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == 'get' and len(node.args) == 2
                    and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value == runtime_key
                    and isinstance(node.args[1], ast.Constant)):
                assert node.args[1].value == str(declared).lower(), (
                    f"{module.__name__} falls back to {node.args[1].value!r} for {json_key}, "
                    f"declared {declared!r}")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-x", "-q"]))
