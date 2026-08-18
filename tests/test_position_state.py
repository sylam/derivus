"""`DiffV2_Position_State` — the FRICTIONAL Bellman: the signed net book fraction
p = Sum(q)/Q_max becomes a state coordinate of the fitted value, and the repositioning charge is
subtracted from the wealth that becomes the regressed TARGET rather than only from the wealth that
ranks the action.

The defect it exists for: a position-free, cost-free `C_t(market, W)` prices turnover as a one-day
toll the value function immediately forgets. A toll cannot produce hysteresis - the next step's
value is the same whatever book it inherits - so no no-trade region can form and the argmax
teleports across the grid. Charging inside the target makes the toll compound down the recursion,
and reading the successor at the book the CHOSEN action leaves standing is what carries it.

The other half is EXPIRY. `_decide` prices, states and returns only what still lives, and the p a
net is FITTED on is the book under the mask of the step that SET it — `live_{t-1}`, the only mask
the querying `_decide` at t-1 can hold. Both were measured defects, not hypotheses; see the two
gates that name them.

KILL MATRIX — every mutant applied to the source, the gate module run, the death recorded, the
mutant reverted. 24 mutants, 24 deaths, none survived (M12 and M24 survived a first pass and the
two gates below were rebuilt around them; an uneven pair of position boxes is what makes the
phantom row win a tie group and the phantom distance bind the band):

| mutant | died at |
| ------ | ------- |
| M1 the p column lands before W instead of after | `test_off_is_the_position_free_layout` |
| M2 `_decide` builds p regardless of the switch | `test_off_decides_cost_free_and_position_free` |
| M3 the target's charge deleted (charge left at the RANKING only) | `test_the_label_target_pays_the_charge` |
| M4 `_decide`'s p' taken from `q_prev`, not the candidate | `test_the_successor_is_queried_per_action` |
| M5 the label's p' taken from `q_prev`, not the chosen action | `test_the_label_successor_is_read_at_the_chosen_book` |
| M6 the stale-checkpoint refusal deleted | `test_a_mismatched_checkpoint_is_refused_by_name` (both directions), `test_a_pre_feature_checkpoint_reads_as_position_free`, `test_the_saved_artifact_stamps_the_position_state` |
| M7 the terminal unwind dropped | `test_the_terminal_unwind_is_charged_iff_forced_flat[True-True]` |
| M8 the unwind charged at every step, not only the terminal | `test_the_unwind_is_terminal_only` |
| M9 the missing-cap guard dropped | `test_a_missing_position_cap_fails_loud` |
| M10 `live_prev` ignored: p_bank read under THIS step's mask | `test_the_position_state_is_masked_by_the_step_that_set_the_book` |
| M11 `q_prev` unmasked in the charge | `test_a_dead_leg_is_neither_charged_nor_held_nor_stated`, `test_the_rate_limit_band_measures_tradeable_size_only`, `test_the_incumbent_is_queried_at_its_own_live_position` |
| M12 the returned action is the raw grid row | `test_a_dead_leg_is_neither_charged_nor_held_nor_stated` |
| M23 `acts` unmasked in the charge (the phantom priced again) | `test_a_dead_leg_is_neither_charged_nor_held_nor_stated` |
| M24 the rate-limit band measures the raw book | `test_the_rate_limit_band_measures_tradeable_size_only` |
| M13 `kappa_T` folded back inside the `q_prev` gate | `test_the_terminal_unwind_survives_a_missing_standing_book` |
| M14 the deadband incumbent pays no charge | `test_the_incumbent_pays_its_own_terminal_unwind` |
| M15 the incumbent evaluated position-free | `test_the_incumbent_is_queried_at_its_own_live_position` |
| M16 the incumbent p built from the RAW standing book | `test_the_incumbent_is_queried_at_its_own_live_position` |
| M17 the single-inner-draw guard dropped | `test_the_deadband_refuses_a_single_inner_draw` |
| M18 the bank keeps the narrow roll (no uniform net target) | `test_the_bank_covers_the_axis_the_argmax_queries` |
| M19 the ensemble concat drops the p column | `test_the_ensemble_branch_carries_the_position_column` |
| M20 the artifact forgets the `position_state` stamp | `test_the_saved_artifact_stamps_the_position_state` |
| M21 the verdict's cost omits the terminal unwind | `test_the_verdict_cost_includes_the_terminal_unwind` |
| M22 the charged-argmax regime is not stated | `test_the_verdict_cost_includes_the_terminal_unwind` |

SWITCH-OFF EQUIVALENCE, stated precisely. Every COMPUTED quantity is bit-identical to the
pre-merge solver: measured over 400 knob combinations (kappa / churn / max-trade band / deadband /
corridor / cap / risk-kappa / standing book present or absent), every returned action and value
and every `_standardize` row equal bit for bit. Three things move BY DESIGN and are not covered by
that claim: `config_hash` and the artifact key set (the normalized solver cfg gained a key, so the
same recipe hashes differently — nothing branches on it); the returned action is now the REALIZED
(live-masked) book rather than the raw grid row that masks to it; and on a dead leg with a standing
book the charge and the deadband incumbent no longer price the phantom, which is the whole point of
the expiry fix (112 further combinations, all of that one shape).

Every gate runs `_decide` / `_fit_step` / `_build_bank` / `_verdict` / `_check_load_provenance` /
`_policy_artifact` as UNBOUND functions against a minimal stand-in solver, the harness pattern of
`test_max_trade_per_step` and `test_churn_lambda`: the fake enumerates exactly what the seam
touches, so a new read is visible as a new attribute. The probe continuation stashes what it was
queried with ON THE SOLVER OBJECT - never in a map keyed by `id()`, which recycles.
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from derivus.hedge_solver import DiffSolver, HedgeActionSpace

KAPPA = 1.0        # Transaction_Cost_Per_Unit with a zero spread: one currency unit per contract
CAP = 4.0          # Evaluator.Total_Position_Abs_Limit — the Q_max that p is measured in


def _runtime(levels=5, lo=-4.0, hi=0.0, force_flat=False):
    hedges = ["A"]
    return {
        "names": {"hedges": hedges},
        "tradables": {r: {"contract_size": 1.0} for r in hedges},
        "portfolio_state": {"positions": {}},
        "solver": {"training_action_grid_levels_per_axis": levels,
                   "training_action_chunk_size": 64, "active_hedge_indices": None},
        "accounting": {
            "position_limits": {r: {"min_position": lo, "max_position": hi} for r in hedges},
            "total_position_abs_limit": CAP,
            "total_position_schedule": None,
            "max_trade_per_step": 0.0,
            "decision_deadband_sigma": 0.0,
            "force_flat_at_end": force_flat,
            "transaction_cost_per_unit": KAPPA,
            "bid_offer_spread_bps": 0.0,
            "bid_offer_spread_spec": None,
        },
    }


def _solver(runtime, position_state, T_dec=3, n_steps=4, B=2):
    """The minimal stand-in `_decide` / `_fit_step` run on. `seen` is the probe continuation's
    record of every (W1, p) it was queried with, stashed on the object itself."""
    aspace = HedgeActionSpace(runtime, torch.device("cpu"))
    hedges = runtime["names"]["hedges"]
    s = types.SimpleNamespace(
        aspace=aspace, chunk=64, risk_kappa=0.0, churn_lambda=0.0,
        position_state=position_state, wealth_free=False,
        # The two OBJECTIVE dials, off: the reward is terminal and the knee is one scalar.
        running_wealth=False, utility_scale_schedule=None, scheduled_scale=False,
        force_flat=runtime["accounting"]["force_flat_at_end"],
        T_dec=T_dec, total_abs_limit=aspace.total_abs_limit,
        hedges=hedges, contract_size=aspace.contract_size, device=torch.device("cpu"),
        # Flat marks: per_contract_kappa is then the Transaction_Cost_Per_Unit alone.
        tradables_sim={r: torch.zeros(n_steps, B) for r in hedges},
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


def _decide(s, q_prev, t=0, kappa=None, dF=0.0, live=None, B=2, Bi=2, md=1):
    """Run the real `_decide` from zero wealth on a world whose one-step move is the flat `dF` per
    contract, so the pre-charge continuation of action q is exactly `sum(q·live) * dF`. `dF` is
    live-masked here exactly as `_inner_step` masks it — a dead contract's one-step move is 0."""
    n_h = len(s.hedges)
    moves = torch.full((B, Bi, n_h), dF) * (1.0 if live is None else live)
    return DiffSolver._decide(
        s, None, torch.zeros(B, Bi, md), moves, torch.zeros(B, Bi),
        torch.zeros(B), t, q_prev=q_prev, kappa=kappa, live=live)


def _fit_step(s, t=0, B=2, Bi=2, md=1, q_prev_val=0.0, live=None, live_prev=None):
    """Drive the real `_fit_step` on a one-leg, one-action world: the grid is a single position, so
    the label argmax is FORCED and the only thing the switch can move is the target's wealth."""
    n_h = len(s.hedges)
    live = torch.ones(n_h) if live is None else live
    zero_in = (torch.zeros(B, Bi, n_h), torch.zeros(B, Bi), torch.zeros(B, Bi, md),
               torch.zeros(B, md), live)
    W_bank = {t: torch.zeros(B)}
    q_bank = {t: torch.full((B, n_h), q_prev_val)}
    s.bundle = types.SimpleNamespace(inner_mc_grad=lambda _t: {
        "F_t1": {r: torch.zeros(B, Bi) for r in s.hedges},
        "L_t1": torch.zeros(B, Bi), "L_t": torch.zeros(B, Bi),
        "market_t1": torch.zeros(B, Bi, md), "market_t": torch.zeros(B, md),
        "state_t_leaves": {}, "state_t_leaf_widths": [],
    })
    label = {}
    s._fit_from_labels = lambda nets, W0_bank, market0, Y, gW, g_market, tt, q_star, p_bank: \
        label.update(Y=Y, q_star=q_star, p_bank=p_bank) or label
    s._decide = types.MethodType(DiffSolver._decide, s)
    DiffSolver._fit_step(s, None, W_bank, t, zero_in, q_bank, live_prev=live_prev)
    return label


# --- (a) switch off: the pre-change layout, labels and decisions, unchanged -------------------
def test_off_is_the_position_free_layout():
    """`_standardize` with no position state is the two-block (market | W) concatenation, and the
    `p is None` contract is what the whole switch rests on at this level. Killed by appending p
    unconditionally (the net's input dim, and every checkpoint written under it, would move)."""
    s = types.SimpleNamespace(m_mean=torch.zeros(3), m_std=torch.ones(3),
                              w_mean=torch.tensor(0.0), w_std=torch.tensor(2.0),
                              wealth_free=False)
    market, W = torch.randn(5, 3), torch.randn(5)
    x = DiffSolver._standardize(s, market, W, None)
    assert x.shape == (5, 4)
    assert torch.equal(x, torch.cat([market, (W / 2.0).unsqueeze(-1)], dim=-1))
    on = DiffSolver._standardize(s, market, W, torch.full((5,), -0.5))
    assert on.shape == (5, 5) and torch.equal(on[:, :4], x)


def test_off_decides_cost_free_and_position_free():
    """The reference run, with everything the switch would arm made maximally visible: a standing
    book four contracts away, `Force_Flat_At_End` on and the successor TERMINAL. Off, none of it
    reaches the continuation — no p column, no reposition toll, no unwind — so the argmax is the
    plain E[C] winner at the short corner. Killed by any position-state term escaping its switch."""
    s = _solver(_runtime(force_flat=True), position_state=False, T_dec=1)
    q, _ = _decide(s, q_prev=torch.zeros(2, 1), dF=-1.0)
    assert torch.equal(q, torch.full((2, 1), -4.0))                    # 1/contract, nothing charged
    assert s.seen and all(p is None for _, p in s.seen)
    assert torch.equal(s.seen[0][0].reshape(2, 5, 2)[0, :, 0],
                       -s.aspace.grid().reshape(-1))                   # the bare value, no toll


def test_off_leaves_the_label_target_uncharged():
    """The label reference: with the switch off the target wealth carries no repositioning charge
    at all, even from a standing book four contracts away with a live kappa."""
    s = _solver(_runtime(levels=1, lo=-3.0, hi=-3.0), position_state=False)
    label = _fit_step(s, q_prev_val=1.0)
    assert torch.equal(label["q_star"], torch.full((2, 1), -3.0))
    assert label["p_bank"] is None
    assert torch.equal(label["Y"], torch.zeros(2))


# --- (b) switch on: the charge is inside the TARGET, not only the ranking ---------------------
def test_the_label_target_pays_the_charge():
    """One action in the grid, so the label argmax is FORCED to q* = -3 and the ranking charge is
    invisible: whatever separates the on-target from the off-target is the charge the TARGET pays.
    From a standing book of +1 that is |-3 - 1| * kappa = 4.

    This is the gate the whole feature turns on. Killed by deleting the target's charge line -
    the charge then lives at the ranking only, which is `DiffV2_Cost_Aware_Argmax`, a one-day toll
    the fitted value never sees."""
    rt = _runtime(levels=1, lo=-3.0, hi=-3.0)
    off = _fit_step(_solver(rt, position_state=False), q_prev_val=1.0)
    s = _solver(rt, position_state=True)
    on = _fit_step(s, q_prev_val=1.0)
    assert torch.equal(on["q_star"], off["q_star"])                       # same forced action
    assert torch.equal(off["Y"] - on["Y"], torch.full((2,), 4.0 * KAPPA))
    # and the bank's own standing book is the state the regression reads at t
    assert torch.equal(on["p_bank"], torch.full((2,), 1.0 / CAP))


def test_the_label_successor_is_read_at_the_chosen_book():
    """The target's continuation is queried at p' = Sum(q*)/Q_max — the book the label action
    leaves standing, not the book it inherited. Killed by p' taken from q_prev."""
    s = _solver(_runtime(levels=1, lo=-3.0, hi=-3.0), position_state=True)
    _fit_step(s, q_prev_val=1.0)
    _, p = s.seen[-1]                                        # the target query, after the argmax's
    assert torch.equal(p.unique(), torch.tensor([-3.0 / CAP]))


# --- (c) switch on: the successor is queried per CANDIDATE action -----------------------------
def test_the_successor_is_queried_per_action():
    """`_decide` must query C_{t+1} at a p' that varies across the candidate actions: that is the
    only channel through which the incoming book can make a REPOSITION cheaper than a jump. The
    probe records its inputs on the solver object; the recorded p' over the action axis must be
    exactly grid.sum(-1)/Q_max.

    Killed by p' computed from q_prev (constant across actions - the value function would then
    price every candidate at the same position state, which is the position-free defect wearing
    an extra input column)."""
    s = _solver(_runtime(levels=5), position_state=True)
    q_prev = torch.full((2, 1), -1.0)
    _decide(s, q_prev=q_prev, kappa=torch.tensor([KAPPA]), dF=-1.5)
    assert len(s.seen) == 1                                  # one chunk covers the 5-action grid
    W1, p = s.seen[0]
    p_per_action = p.reshape(2, 5, 2)[0, :, 0]
    assert torch.equal(p_per_action, s.aspace.grid().sum(-1) / CAP)
    assert p_per_action.unique().numel() == 5                # varies — not the constant q_prev/Q_max
    assert not torch.equal(p_per_action, torch.full((5,), -1.0 / CAP))


def test_the_ranking_charge_is_the_same_arithmetic_as_the_target():
    """The ranking seam, checked against the same independent arithmetic the label gate uses: the
    wealth each candidate is scored on is its cost-free continuation minus |q - q_prev| * kappa.
    Both seams call one expression, so this and `test_the_label_target_pays_the_charge` are two
    readings of the same number."""
    s = _solver(_runtime(levels=5), position_state=True)
    q_prev = torch.full((2, 1), -1.0)
    q, _ = _decide(s, q_prev=q_prev, kappa=torch.tensor([KAPPA]), dF=-1.5)
    grid = s.aspace.grid().reshape(-1)                                    # (5,) = -4 .. 0
    assert torch.equal(s.seen[0][0].reshape(2, 5, 2)[0, :, 0],
                       -1.5 * grid - (grid + 1.0).abs() * KAPPA)
    assert torch.equal(q, torch.full((2, 1), -4.0))          # 1.5/contract beats the 1/contract toll


# --- (d) checkpoint provenance ----------------------------------------------------------------
def _stub(position_state):
    rt = _runtime()
    s = types.SimpleNamespace(t_min=0, T_dec=3, hedges=list(rt["names"]["hedges"]),
                              position_state=position_state, wealth_free=False,
                              running_wealth=False, utility_scale_schedule=None,
                              scheduled_scale=False,
                              aspace=HedgeActionSpace(rt, torch.device("cpu")))
    s._check_action_universe = types.MethodType(DiffSolver._check_action_universe, s)
    s._check_calendar_spread = types.MethodType(DiffSolver._check_calendar_spread, s)
    return s


def _ck(**over):
    ck = {"t_min": 0, "T_dec": 3, "md": 1, "hedges": ["A"], "position_state": False,
          "solver_version": "x", "total_position_schedule": None}
    ck.update(over)
    return ck


@pytest.mark.parametrize("saved,run", [(True, False), (False, True)])
def test_a_mismatched_checkpoint_is_refused_by_name(saved, run):
    """A value fn trained with p and one trained without are different functions of different
    states, and p is an INPUT COLUMN — so the failure would otherwise surface as a
    `load_state_dict` size mismatch naming a Linear weight. The refusal must name the key that
    caused it, in both directions. Killed by dropping the stamp check."""
    with pytest.raises(ValueError, match="DiffV2_Position_State"):
        DiffSolver._check_load_provenance(_stub(run), _ck(position_state=saved), "<ck>", 1)


def test_a_matched_checkpoint_loads():
    """Both directions matched pass the guard — the refusal is about disagreement, not about the
    feature being on. Vacuity check for the pair above."""
    for flag in (False, True):
        assert DiffSolver._check_load_provenance(
            _stub(flag), _ck(position_state=flag), "<ck>", 1) == "x"


def test_a_pre_feature_checkpoint_reads_as_position_free():
    """A checkpoint written before the feature carries no stamp at all: it is position-FREE, so it
    loads under the default and is refused under the switch."""
    old = _ck()
    del old["position_state"]
    assert DiffSolver._check_load_provenance(_stub(False), old, "<ck>", 1) == "x"
    with pytest.raises(ValueError, match="DiffV2_Position_State"):
        DiffSolver._check_load_provenance(_stub(True), old, "<ck>", 1)


# --- (e) the terminal unwind ------------------------------------------------------------------
@pytest.mark.parametrize("force_flat,unwind", [(True, True), (False, False)])
def test_the_terminal_unwind_is_charged_iff_forced_flat(force_flat, unwind):
    """`Force_Flat_At_End` means whatever the LAST decision leaves standing is liquidated at the
    terminal kappa, so the frictional value has to price that unwind — without it the final step
    accumulates a position for free and the recursion ends in a discontinuity. Charged iff the
    accounting forces the flat.

    Killed by dropping the unwind: the forced-flat run's wealth would equal the free one's."""
    s = _solver(_runtime(force_flat=force_flat), position_state=True, T_dec=1)   # t+1 == T_dec
    _decide(s, q_prev=torch.zeros(2, 1), kappa=torch.tensor([KAPPA]), dF=-1.5)
    grid = s.aspace.grid().reshape(-1)                                           # (5,) = -4 .. 0
    tolls = 2.0 if unwind else 1.0            # the entry toll, and the liquidation of what it buys
    assert torch.equal(s.seen[0][0].reshape(2, 5, 2)[0, :, 0],
                       -1.5 * grid - tolls * grid.abs() * KAPPA)


def test_the_unwind_is_terminal_only():
    """An interior step pays the reposition toll and nothing else, however the end is mandated —
    the unwind belongs to the last decision, not to every one. Killed by a `t + 1 >= T_dec` test
    that always passes."""
    s = _solver(_runtime(force_flat=True), position_state=True, T_dec=3)         # t+1 < T_dec
    _decide(s, q_prev=torch.zeros(2, 1), kappa=torch.tensor([KAPPA]), dF=-1.5)
    grid = s.aspace.grid().reshape(-1)
    assert torch.equal(s.seen[0][0].reshape(2, 5, 2)[0, :, 0],
                       -1.5 * grid - grid.abs() * KAPPA)


def test_a_missing_position_cap_fails_loud():
    """p = Sum(q)/Q_max, and `Evaluator.Total_Position_Abs_Limit` is the only declared scale it can
    be measured in. An absent cap is a division by zero the whole state coordinate would silently
    inherit, so the switch refuses to start."""
    rt = _runtime()
    rt["accounting"]["total_position_abs_limit"] = 0.0
    rt["solver"]["diffv2_position_state"] = True
    bundle = types.SimpleNamespace(device=torch.device("cpu"), vol_sim=None)
    with pytest.raises(ValueError, match="Total_Position_Abs_Limit"):
        DiffSolver(bundle, {**rt, "solver": rt["solver"]})


# --- the ONE mask convention: p is registered under the mask that SET the book ----------------
def test_the_position_state_is_masked_by_the_step_that_set_the_book():
    """`nets[t]` is QUERIED from `_decide` at t-1, which can only ever hold `live_{t-1}`, so the
    fit has to meet it there: p_t = Sum(q · live_{t-1})/Q_max. Reading the bank's book under THIS
    step's mask instead writes `nets[t]` on one coordinate and reads it on another, and on a
    leg-rollover step the two differ by the full width of the axis.

    Measured on the simulate-only fixture: exactly one rollover step, where the gap was 0.3333 of
    a [-1,0] axis before and is 0.0000 after. Killed by `live_prev` ignored."""
    rt = _runtime(levels=1, lo=-1.0, hi=-1.0)
    rt["names"]["hedges"] = ["A", "B"]
    rt["tradables"]["B"] = {"contract_size": 1.0}
    rt["accounting"]["position_limits"]["B"] = {"min_position": -1.0, "max_position": -1.0}
    live_prev = torch.tensor([1.0, 1.0])            # both legs stood at t
    live = torch.tensor([1.0, 0.0])                 # leg B rolls off over [t, t+1]
    s = _solver(rt, position_state=True)
    label = _fit_step(s, q_prev_val=-1.0, live=live, live_prev=live_prev)
    assert torch.equal(label["p_bank"], torch.full((2,), -2.0 / CAP)), 'p must use live_prev'
    # ...and that is exactly what `_decide` at t-1 hands the successor for the same book.
    s2 = _solver(rt, position_state=True)
    _decide(s2, q_prev=torch.zeros(2, 2), kappa=torch.tensor([KAPPA, KAPPA]), live=live_prev)
    assert torch.equal(s2.seen[0][1].unique(), torch.tensor([-2.0 / CAP]))


def test_a_dead_leg_is_neither_charged_nor_held_nor_stated():
    """The measured deploy-path distortion: a standing book of |50| on a leg that has just died.
    Unmasked, moving the dead leg to 0 cost the full L1 toll while parking on it cost nothing, so
    the argmax parked there — and because the grid is cap-filtered, that parking spent the whole
    Total_Position_Abs_Limit on a position worth nothing and left the live leg flat.

    Masked, rows that differ only on dead legs are ONE book. Three consequences, one per mutant:
    the RANKING has no spread inside such a group (killed by `acts` unmasked in the charge), the
    whole decision is INVARIANT to what the standing book claims to hold on a dead leg (killed by
    `q_prev` unmasked — there it shifts every score by |q_prev_dead|·kappa and the returned value
    with it), and the ANSWER is the realizable book (killed by returning the raw grid row).

    The boxes are deliberately uneven — the live leg reaches -3, the dead one -4 — so the winning
    live book of -3 has a tie group of two rows whose FIRST member carries -1 on the dead leg.
    That is what makes the raw-grid-row mutant visible rather than accidentally correct."""
    rt = _runtime(levels=5, lo=-3.0, hi=0.0)
    rt["names"]["hedges"] = ["A", "B"]
    rt["tradables"]["B"] = {"contract_size": 1.0}
    rt["accounting"]["position_limits"]["B"] = {"min_position": -4.0, "max_position": 0.0}
    live = torch.tensor([1.0, 0.0])                       # leg B is dead
    kappa = torch.tensor([KAPPA, KAPPA])
    s = _solver(rt, position_state=True)
    q_prev = torch.tensor([[0.0, -4.0], [0.0, -4.0]])     # a phantom |4| standing on the dead leg
    q, v = _decide(s, q_prev=q_prev, kappa=kappa, dF=-1.5, live=live)

    grid = s.aspace.grid_at(0, live)
    a_live = grid * live
    assert (a_live[:, 1] == 0).all() and (grid[:, 1] != 0).any(), 'the fixture must offer phantoms'
    assert torch.equal(q, q * live), 'the answer must be the realizable book, not the grid row'
    assert torch.equal(q, torch.full((2, 1), -3.0).expand(2, 1).repeat(1, 2) * live), \
        'the live leg takes the corner; 1.5/contract beats the 1/contract toll'

    # the ranking cannot separate two rows that are one realizable book
    W1 = s.seen[0][0].reshape(2, grid.shape[0], 2)[0, :, 0]
    for book in a_live.unique(dim=0):
        same = (a_live == book).all(-1)
        assert float(W1[same].max() - W1[same].min()) == 0.0, f'ranking spread on {book.tolist()}'

    # ...and nothing the standing book claims on a DEAD leg can move the decision or its value
    s2 = _solver(rt, position_state=True)
    q2, v2 = _decide(s2, q_prev=torch.zeros(2, 2), kappa=kappa, dF=-1.5, live=live)
    assert torch.equal(q, q2) and torch.equal(v, v2), 'a phantom book changed the decision'


def test_the_rate_limit_band_measures_tradeable_size_only():
    """`Max_Trade_Per_Step` caps how much can be TRADED in a step, and a dead leg cannot be traded
    at all. Reading the band off the raw book makes the phantom |4| standing on an expired leg look
    like a trade the cap forbids, so every row that does not also carry that phantom is struck out
    and the live leg is frozen at whatever survives. Killed by the band reading `acts`/`q_prev`
    instead of `a_live`/`q_live`."""
    rt = _runtime(levels=5, lo=-3.0, hi=0.0)
    rt["names"]["hedges"] = ["A", "B"]
    rt["tradables"]["B"] = {"contract_size": 1.0}
    rt["accounting"]["position_limits"]["B"] = {"min_position": -4.0, "max_position": 0.0}
    # 0.8 reaches one 0.75 step on the live leg but nowhere near the |4| phantom, so a band that
    # measures the phantom admits only the row that keeps it — and the cap then pins the live leg.
    rt["accounting"]["max_trade_per_step"] = 0.8
    live = torch.tensor([1.0, 0.0])
    s = _solver(rt, position_state=True)
    q, _ = _decide(s, q_prev=torch.tensor([[0.0, -4.0], [0.0, -4.0]]),
                   kappa=torch.tensor([KAPPA, KAPPA]), dF=-1.5, live=live)
    assert torch.equal(q, torch.tensor([[-0.75, 0.0], [-0.75, 0.0]])), (
        'one grid step toward the corner is reachable; the dead leg is not a trade')


def test_the_terminal_unwind_survives_a_missing_standing_book():
    """The unwind is a property of the ACTION, not of the incoming book, so it must ride outside
    the `q_prev is not None` gate: a terminal step with nothing standing would otherwise liquidate
    for free. Killed by folding kappa_T back inside that gate."""
    s = _solver(_runtime(force_flat=True), position_state=True, T_dec=1)
    _decide(s, q_prev=None, kappa=None, dF=-1.5, live=None)
    grid = s.aspace.grid().reshape(-1)
    assert torch.equal(s.seen[0][0].reshape(2, 5, 2)[0, :, 0],
                       -1.5 * grid - grid.abs() * KAPPA)


# --- the deadband incumbent (merge spec D + G) ------------------------------------------------
def _deadband_solver(runtime, position_state, sigma=2.0, T_dec=3):
    rt = dict(runtime)
    rt["accounting"] = {**runtime["accounting"], "decision_deadband_sigma": sigma}
    return _solver(rt, position_state, T_dec=T_dec)


def test_the_incumbent_pays_its_own_terminal_unwind():
    """Merge hazard: every CANDIDATE pays the terminal unwind but the incoming incumbent paid
    nothing, so at the last decision the paired difference was biased against trading by the whole
    liquidation cost and the policy degenerated to 'hold whatever is standing'. Routing the
    incumbent through the same `_reposition_charge` with `acts = q_prev` makes `dq = 0` kill the
    L1 and churn terms while the unwind, a charge on the LEVEL, survives.

    Killed by dropping the incumbent's charge — its wealth would be the uncharged wealth step."""
    s = _deadband_solver(_runtime(force_flat=True), position_state=True, T_dec=1)
    q_prev = torch.full((2, 1), -3.0)
    _decide(s, q_prev=q_prev, kappa=torch.tensor([KAPPA]), dF=-1.5)
    W_inc, _ = s.seen[-1]                                   # the incumbent query, after the grid
    assert torch.equal(W_inc.unique(), torch.tensor([-1.5 * -3.0 - 3.0 * KAPPA]))
    # and with the flat NOT forced, the same incumbent pays nothing at all
    s2 = _deadband_solver(_runtime(force_flat=False), position_state=True, T_dec=1)
    _decide(s2, q_prev=q_prev, kappa=torch.tensor([KAPPA]), dF=-1.5)
    assert torch.equal(s2.seen[-1][0].unique(), torch.tensor([-1.5 * -3.0]))


def test_the_incumbent_is_queried_at_its_own_live_position():
    """The paired difference `best_f - C1f_inc` is only a difference of one function if both sides
    are read at their own position state. Passing `p=None` for the incumbent (the merge's most
    likely silent failure) subtracts a position-free reading from a position-aware one, and an
    UNMASKED incumbent claims budget the step's grid cannot.

    Killed by `p_inc=None` or by `p_inc` built from the raw `q_prev`."""
    rt = _runtime(levels=5, lo=-4.0, hi=0.0)
    rt["names"]["hedges"] = ["A", "B"]
    rt["tradables"]["B"] = {"contract_size": 1.0}
    rt["accounting"]["position_limits"]["B"] = {"min_position": -4.0, "max_position": 0.0}
    live = torch.tensor([1.0, 0.0])
    s = _deadband_solver(rt, position_state=True)
    q_prev = torch.tensor([[-2.0, -4.0], [-2.0, -4.0]])     # -4 of it on the DEAD leg
    _decide(s, q_prev=q_prev, kappa=torch.tensor([KAPPA, KAPPA]), dF=-1.5, live=live)
    _, p_inc = s.seen[-1]
    assert p_inc is not None, 'the incumbent must not be evaluated position-free'
    assert torch.equal(p_inc.unique(), torch.tensor([-2.0 / CAP])), 'live-masked, not raw'


@pytest.mark.parametrize("position_state", [False, True])
def test_the_deadband_at_zero_is_off_under_either_switch(position_state):
    """`Decision_Deadband_Sigma = 0` is bit-identically the plain argmax whether or not the
    position state is on — the two features compose, they do not interact."""
    rt = _runtime(levels=5)
    off = _solver(rt, position_state)
    on0 = _deadband_solver(rt, position_state, sigma=0.0)
    args = dict(q_prev=torch.full((2, 1), -4.0), kappa=torch.tensor([KAPPA]), dF=-1.5)
    q_a, v_a = _decide(off, **args)
    q_b, v_b = _decide(on0, **args)
    assert torch.equal(q_a, q_b) and torch.equal(v_a, v_b)


def test_the_deadband_refuses_a_single_inner_draw():
    """One inner draw makes the PAIRED std NaN, every comparison False and the book frozen for
    good. That fails loud rather than silently never trading again."""
    s = _deadband_solver(_runtime(), position_state=True)
    with pytest.raises(ValueError, match="Inner_Sub_Batch"):
        _decide(s, q_prev=torch.full((2, 1), -4.0), kappa=torch.tensor([KAPPA]), Bi=1)


# --- the bank must cover the axis the argmax queries ------------------------------------------
def _bank_solver(runtime, position_state, T_dec=4, B=64):
    s = _solver(runtime, position_state, T_dec=T_dec, n_steps=T_dec + 1, B=B)
    s.B_outer, s.n_hedge, s.active = B, len(s.hedges), s.aspace.active
    s.n_active = s.aspace.n_active
    s.q_lo, s.q_hi = s.aspace.q_lo, s.aspace.q_hi
    s.noise_frac = 0.15
    s.liability_sim = torch.zeros(T_dec + 1, B)
    s.tradables_sim = {r: torch.arange(T_dec + 1, dtype=torch.float32)[:, None] * (1.0 + i)
                       + torch.randn(T_dec + 1, B) for i, r in enumerate(s.hedges)}
    s._replication_hedge = types.MethodType(DiffSolver._replication_hedge, s)
    return s


def test_the_bank_covers_the_axis_the_argmax_queries():
    """p is an input the argmax sweeps over the whole of `net_bounds(t)`, while the bank roll only
    wanders `Bank_Noise_Frac` around the replication hedge. Measured on the platinum book, that
    left the deep-hedge ~44% of the axis unsampled and let p_bank breach the cap the grid enforces,
    so the hysteresis slope was extrapolation exactly where it decides. Under the switch each path
    draws a uniform net target across the feasible band and water-fills onto it.

    Killed by dropping the uniform draw: coverage collapses toward the replication hedge and the
    cap stops binding."""
    rt = _runtime(levels=5)
    rt["names"]["hedges"] = ["A", "B"]
    rt["tradables"]["B"] = {"contract_size": 1.0}
    rt["accounting"]["position_limits"]["B"] = {"min_position": -4.0, "max_position": 0.0}
    torch.manual_seed(0)
    s = _bank_solver(rt, position_state=True)
    _, q_list = DiffSolver._build_bank(s, torch.Generator().manual_seed(0))
    p = torch.cat([q.sum(-1) / CAP for q in q_list])
    lo, hi = s.aspace.net_bounds(0)
    assert float(p.min()) >= lo / CAP - 1e-6 and float(p.max()) <= hi / CAP + 1e-6, 'cap breached'
    assert -1.0 - 1e-6 <= float(p.min()) and float(p.max()) <= 1.0 + 1e-6
    edges = torch.linspace(lo / CAP, hi / CAP, 11)          # every decile of the band is sampled
    for k in range(10):
        assert ((p >= edges[k]) & (p <= edges[k + 1])).any(), f'decile {k} of the p axis unsampled'


def test_the_bank_is_untouched_with_the_switch_off():
    """Switch off, the bank draw is exactly the pre-change roll — the uniform net target is not a
    change to the position-free solver, and that roll does NOT respect the total-position cap,
    which is precisely the gap the switch closes."""
    rt = _runtime(levels=5)
    rt["accounting"]["position_limits"]["A"] = {"min_position": -80.0, "max_position": 0.0}
    torch.manual_seed(0)
    s = _bank_solver(rt, position_state=False)
    _, q_list = DiffSolver._build_bank(s, torch.Generator().manual_seed(3))
    assert float(torch.cat(q_list).abs().sum(-1).max()) > CAP, (
        'the position-free bank is NOT cap-projected; that is what the switch adds')


# --- the ensemble branch and the save stamp ---------------------------------------------------
def test_the_ensemble_branch_carries_the_position_column():
    """`_continuation`'s ensemble path builds the input in its OWN member frames, so it is a
    SECOND place the p column is assembled and the only one the argmax gates never reach. Killed
    by dropping p from the ensemble concat."""
    md, n = 3, 7
    seen = []

    class Rec(torch.nn.Module):
        def forward(self, x):
            seen.append(x.shape[-1])
            return torch.zeros(x.shape[0])

    s = types.SimpleNamespace(
        T_dec=2, a_bounds=[None, None], wealth_free=False, running_wealth=False,
        _ensemble=[([Rec(), Rec()], torch.zeros(md), torch.ones(md),
                    torch.tensor(0.0), torch.tensor(1.0), None, None)],
        _u=lambda W, t=None: torch.zeros_like(W))
    DiffSolver._continuation(s, None, torch.randn(n, md), torch.randn(n), 0, None)
    assert seen[-1] == md + 1, 'position-free ensemble input must stay (market | W)'
    DiffSolver._continuation(s, None, torch.randn(n, md), torch.randn(n), 0, torch.zeros(n))
    assert seen[-1] == md + 2, 'the ensemble must carry the p column too'


def test_the_saved_artifact_stamps_the_position_state():
    """The artifact must stamp every key `_check_load_provenance` refuses on — a stamp the save
    forgets is a run that refuses its own checkpoint on reload. Round-tripped here rather than
    asserted against a key list, so deleting the stamp fails the RELOAD. Killed by dropping
    `position_state` from `_policy_artifact`."""
    for flag in (False, True):
        s = _solver(_runtime(), flag)
        s.runtime = {"objective": {"utility_scale": 1.0}, "solver": {}}
        s.m_mean = s.m_std = torch.ones(1)
        s.w_mean = s.w_std = torch.tensor(1.0)
        s.a_bounds = [None]
        s.t_min, s.active = 0, [0]
        s._config_hash = lambda: "h"
        s._frame_stamp = lambda: "f"
        ck = DiffSolver._policy_artifact(s, [], 1, 8, 0.0, [0.0], 0.0)
        assert ck["position_state"] is flag
        # the round trip: this run's own checkpoint must load back under its own setting...
        assert DiffSolver._check_load_provenance(s, ck, "<ck>", 1) == ck["solver_version"]
        # ...and be refused by a run set the other way.
        with pytest.raises(ValueError, match="DiffV2_Position_State"):
            DiffSolver._check_load_provenance(_stub(not flag), ck, "<ck>", 1)


# --- the verdict's cost diagnostic must price the same execution as the benchmarks ------------
def _verdict_solver(runtime, T_dec=3, B=8, Bi=2, md=1):
    """A stand-in `_verdict` runs on: real wealth law, identity continuation, flat marks so kappa
    is the Transaction_Cost_Per_Unit alone."""
    s = _bank_solver(runtime, position_state=False, T_dec=T_dec, B=B)
    n_h = len(s.hedges)
    s.t_min, s.cost_aware = 0, False
    s.tradables_sim = {r: torch.zeros(T_dec + 1, B) for r in s.hedges}
    s.liability_sim = torch.zeros(T_dec + 1, B)
    s._u = lambda W, t=None: W
    s._decide = types.MethodType(DiffSolver._decide, s)
    s._continuation = lambda nets, market, W, t, p: W
    inner = {t: (torch.zeros(B, Bi, n_h), torch.zeros(B, Bi), torch.zeros(B, Bi, md),
                 torch.zeros(B, md), torch.ones(n_h)) for t in range(T_dec)}
    return s, inner, list(range(T_dec))


def test_the_verdict_cost_includes_the_terminal_unwind():
    """`run_textbook_benchmark` and `HindsightDpSolver` both charge their execution an unwind under
    `Force_Flat_At_End`; the verdict's greedy/textbook accumulator did not, so its
    `turnover_cost_mean` understated the friction the policy was optimized against (measured ~28%
    on the platinum book) and every `*_net` column read in greedy's favour against the benchmarks.

    Killed by dropping the unwind block: the two runs below would report the same cost."""
    rt = _runtime(levels=5)
    s_flat, inner, ts = _verdict_solver({**rt, "accounting": {**rt["accounting"],
                                                              "force_flat_at_end": True}})
    s_free, inner2, _ = _verdict_solver(rt)
    out_flat = DiffSolver._verdict(s_flat, None, inner, ts)
    out_free = DiffSolver._verdict(s_free, None, inner2, ts)
    extra = (out_flat["greedy"]["turnover_cost_mean"]
             - out_free["greedy"]["turnover_cost_mean"])
    final_book = abs(out_flat["greedy_q_traj"][-1][0])       # the book left standing at T_dec
    assert final_book > 0.0, 'the fake must end holding something for the gate to bite'
    assert extra == final_book * KAPPA, 'the unwind of the standing book, at the terminal kappa'
    # ...and the regime that chose the book is stated, not left to be inferred from the figures
    assert out_flat["argmax_charged"] is False
    s_flat.position_state = True
    assert DiffSolver._verdict(s_flat, None, inner, ts)["argmax_charged"] is True


def test_wealth_free_without_position_state_is_refused():
    """An action cannot move the market, so with the W column removed and no p the residual
    A(m') is one constant across every candidate and cancels from the argmax - the whole fit
    would be decision-irrelevant (measured: myopic argmax 20/20 seeds). The composition
    refuses to start rather than burning a training budget on nets no decision reads."""
    rt = _runtime()
    rt["solver"]["diffv2_wealth_free_value"] = True
    rt["solver"]["diffv2_position_state"] = False
    bundle = types.SimpleNamespace(device=torch.device("cpu"), vol_sim=None)
    with pytest.raises(ValueError, match="DiffV2_Position_State"):
        DiffSolver(bundle, rt)


def test_wealth_free_with_risk_kappa_is_refused():
    """The downside semideviation is nonlinear over inner draws, so the residual's dispersion
    leaks back into a ranking the switch exists to hand to the utility alone (measured: the
    A-free argmax diverges 19/20 seeds at kappa 0.5). Refused by name."""
    rt = _runtime()
    rt["solver"]["diffv2_wealth_free_value"] = True
    rt["solver"]["diffv2_position_state"] = True
    rt["solver"]["diffv2_risk_kappa"] = 0.5
    bundle = types.SimpleNamespace(device=torch.device("cpu"), vol_sim=None)
    with pytest.raises(ValueError, match="DiffV2_Risk_Kappa"):
        DiffSolver(bundle, rt)
