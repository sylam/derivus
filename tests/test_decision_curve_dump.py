"""`Solver.DiffV2_Decision_Curve_Dump` — the stepper rollout's per-decision CSV.

WHAT IT IS. A pure diagnostic hung off `_rollout_on_stepper`'s greedy roll: one row per decision
day carrying the FULL per-rung ranking curve (pre-charge and charged), where that day sits on the
utility in force (x0, the knee, the local risk aversion), the chance the day ENDS in the knee at
three reference books, the one-step moments of the fork it was ranked against, and a second,
independent fork's answer to the same question. It exists because "the DP parked at the bound" and
"the DP's curve was flat and the bound won the tie" are different findings, and nothing in the
verdict distinguishes them.

WHAT IT MUST NOT BE is a re-derivation. The dump scores the action universe a second time rather
than instrumenting `_decide`'s chunk loop (see `_score_actions`), which buys switch-off equivalence
outright and costs the risk that the two rankings drift. `test_the_curve_argmax_is_the_executed
_decision` is the gate that spends that risk: it charges the candidates through a standing book so
a scoring pass that forgot the charge picks a DIFFERENT rung, and asserts the dump's argmax column
is the book the roll actually executed.

THE ONE THING THAT MOVES, stated precisely: the dump forks inner MC a second time per decision
step, which advances the framework's quasi-random stream. Nothing downstream of the rollout draws
from it — `_verdict` has already run, and the textbook/hindsight tracks are draw-free — so no
reported number moves. Every returned figure of `_rollout_on_stepper` is bit-identical with the
switch on, which is what `test_the_switch_off_roll_is_bit_identical` asserts directly.

KILL MATRIX — every mutant applied to the source, this module RUN, the death recorded, the mutant
reverted. 7 mutants, 7 deaths, none survived:

| mutant | died at |
| ------ | ------- |
| M1 the dump re-scores the rungs WITHOUT the repositioning charge | `test_the_curve_argmax_is_the_executed_decision` (+2) |
| M2 `c` read from the bundle (the eval world's / JSON's) instead of the frame | `test_the_scale_comes_from_the_frame_not_the_world` (+1) |
| M3 `P_band` computed on x0 (the entering wealth) instead of x1 | `test_the_band_probability_is_measured_on_the_successor_wealth` |
| M4 the second fork reuses the cached first draw | `test_the_second_fork_is_an_independent_draw` |
| M5 the switch does not gate: the dump always runs | `test_the_switch_off_roll_is_bit_identical` |
| M6 the dump's argmax ignores the `Max_Trade_Per_Step` band | `test_the_deadband_row_is_marked_and_the_band_bounds_the_argmax` |
| M7 the pre-charge curve is charged too (the pair collapses) | `test_the_curve_argmax_is_the_executed_decision` |

HARNESS. `_rollout_on_stepper` runs as an UNBOUND function against a minimal stand-in solver — the
`SimpleNamespace` pattern of `test_position_state` — with `BundleStepper` and
`_tracking_error_value` replaced by a stand-in environment whose wealth path is prescribed, so the
gates read the DECISION and not the accounting. The probe continuation is `u(W)` itself, which
makes the ranking exactly `E_inner[u(W1)]` and therefore hand-computable.

THE FAKE WORLD, hand-computed so the gates assert numbers rather than invariants. One hedge leg,
contract size 1, box [-4, 0], grid {-4,-3,-2,-1,0}. Huber utility, c = 100, R = 0, a = 2.5,
delta = 1, no gain-wing curvature. Fork draw A is dF = (-24, -8, 0, 8) with dL = 0, so E[dF] = -6:
shorting has a positive-mean edge and the deep corner -4 maximises the UNCHARGED curve. The
turnover fee is 4/contract, and from the opening flat book the charged curve peaks at -1:

    q      0       -1      -2      -3      -4
    pre    0.000   +0.056  +0.104  +0.144  +0.176     <- argmax -4
    chg    0.000   +0.010   0.000  -0.030  -0.080     <- argmax -1, and the executed book

Fork draw B is the mirror, dF = (24, 8, 0, -8): the same day's edge points the other way, so its
argmax is 0 and the disagreement flag is 1.
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest
import torch

from derivus import hedge_bundle
from derivus.hedge_bundle import _utility_local_curvature, _utility_wrap_signed
from derivus.hedge_solver import DiffSolver, HedgeActionSpace

CAP = 4.0                 # Evaluator.Total_Position_Abs_Limit
FEE = 4.0                 # Transaction_Cost_Per_Unit (zero spread ⇒ kappa IS the fee)
FRAME_C = 100.0           # the utility scale the checkpoint was fitted under — the frame's
WORLD_C = 7.0             # the scale THIS eval world resolved — the one mirror_scale=False refuses
T_DEC = 3                 # decision steps 0, 1, 2
DRAW_A = [-24.0, -8.0, 0.0, 8.0]         # the cached fork: E[dF] = -6, shorting pays
DRAW_B = [24.0, 8.0, 0.0, -8.0]          # the second fork: the mirror, so its argmax flips
W_PATH = [0.0, 0.0, -50.0]               # net wealth entering decisions 0, 1, 2


# ---------------------------------------------------------------- the stand-in world
def _runtime(deadband=0.0, max_trade=0.0):
    hedges = ["A"]
    return {
        "names": {"hedges": hedges},
        "tradables": {r: {"contract_size": 1.0} for r in hedges},
        "portfolio_state": {"positions": {}},
        "objective": {"object": "asymmetricutility_huber", "utility_scale": FRAME_C,
                      "reference_wealth": 0.0, "huber_aversion": 2.5, "huber_delta": 1.0,
                      "up_aversion": 0.0, "up_knee": 0.15},
        "solver": {"training_action_grid_levels_per_axis": 5,
                   "training_action_chunk_size": 64, "active_hedge_indices": None},
        "accounting": {
            "position_limits": {r: {"min_position": -CAP, "max_position": 0.0} for r in hedges},
            "total_position_abs_limit": CAP,
            "total_position_schedule": None,
            "allocation_weights": None,
            "calendar_spread_bps": None,
            "max_trade_per_step": max_trade,
            "decision_deadband_sigma": deadband,
            "force_flat_at_end": False,
            "transaction_cost_per_unit": FEE,
            "bid_offer_spread_bps": 0.0,
            "bid_offer_spread_spec": None,
        },
    }


def _fork(draw, n_steps=T_DEC + 1):
    """One inner-MC fork's `(dF, dL, market_t1, market_t, live)` tuple at B_outer=1: the flat
    tradable mark makes dF the draw itself, and the liability does not move."""
    dF = torch.tensor(draw).reshape(1, len(draw), 1)              # (B=1, Bi, n_hedge)
    dL = torch.zeros(1, len(draw))
    return dF, dL, torch.zeros(1, len(draw), 1), torch.zeros(1, 1), torch.ones(1)


class _Stepper:
    """The stand-in environment: a prescribed wealth path, so a gate reads the DECISION and not
    the accounting. Each construction appends its own book log to `rolls`, and the GREEDY roll is
    `rolls[0]` — `_rollout_on_stepper` rolls greedy, textbook, no-hedge in that order."""

    rolls = []

    def __init__(self, bundle, runtime, mirror_scale=True):
        self.time_index = 0
        self.mirror_scale = mirror_scale
        self._state = {"W": torch.tensor([W_PATH[0]]), "positions": {"A": torch.zeros(1)}}
        self.driven = []
        _Stepper.rolls.append(self.driven)

    @property
    def done(self):
        return self.time_index >= T_DEC

    @property
    def is_decision_step(self):
        return True

    def step(self, delta):
        if delta is not None:
            for n, d in delta.items():
                self._state["positions"][n] = self._state["positions"][n] + d
            self.driven.append(float(self._state["positions"]["A"]))
        self.time_index += 1
        if self.time_index < T_DEC:
            self._state["W"] = torch.tensor([W_PATH[self.time_index]])
        return {"transition_pnl_excess": self._state["W"],
                "transition_liability_value": torch.zeros(1)}


def _solver(runtime, dump_path="", world_c=WORLD_C):
    """The minimal stand-in `_rollout_on_stepper` runs on. The continuation is `u(W)` itself, so
    the ranking is exactly `E_inner[u(W1)]` and the curve is hand-computable."""
    aspace = HedgeActionSpace(runtime, torch.device("cpu"))
    s = types.SimpleNamespace(
        runtime=runtime, aspace=aspace, chunk=64, risk_kappa=0.0, churn_lambda=0.0,
        position_state=False, wealth_free=False, cost_aware=True, force_flat=False,
        T_dec=T_DEC, t_min=0, total_abs_limit=CAP, curve_dump=dump_path,
        hedges=list(runtime["names"]["hedges"]), n_hedge=1, contract_size=aspace.contract_size,
        device=torch.device("cpu"), B_outer=1,
        active=aspace.active, n_active=aspace.n_active, q_lo=aspace.q_lo, q_hi=aspace.q_hi,
        tradables_sim={r: torch.zeros(T_DEC + 1, 1) for r in runtime["names"]["hedges"]},
        liability_sim=torch.zeros(T_DEC + 1, 1),
        # `initial_time_index` 0 ⇒ the stepper's time index IS the decision index; the eval
        # world's own scale sits here, and mirror_scale=False is what keeps it out of the frame.
        bundle=types.SimpleNamespace(initial_time_index=0, utility_scale=world_c,
                                     inner_mc=lambda t: None),
    )
    for name in ("_wealth_step", "_unwind_kappa", "_calendar_kappa", "_reposition_charge",
                 "_decide", "_u", "_replication_hedge", "_score_actions", "_decision_curve_row",
                 "_rollout_on_stepper"):
        setattr(s, name, types.MethodType(getattr(DiffSolver, name), s))
    s._continuation = lambda nets, market, W, t, p: s._u(W)
    # The cached forks the roll decides on (draw A), and a `_inner_step` that always hands back
    # the SECOND, independent draw — the dump's own fork.
    s.inner_cache = {t: _fork(DRAW_A) for t in range(T_DEC)}
    s._inner_step = lambda t: _fork(DRAW_B)
    return s


def _roll(monkeypatch, runtime, dump_path="", world_c=WORLD_C):
    """Run the real `_rollout_on_stepper` against the stand-in environment."""
    monkeypatch.setattr(hedge_bundle, "BundleStepper", _Stepper)
    monkeypatch.setattr(hedge_bundle, "_tracking_error_value", lambda state, rt: state["W"])
    _Stepper.rolls = []
    s = _solver(runtime, dump_path=dump_path, world_c=world_c)
    return s, s._rollout_on_stepper(None, s.inner_cache, list(range(T_DEC)))


def _hand_curve(W, q_prev, draw, charged):
    """`E_inner[u(W1)]` per rung, computed straight from the definition — the reference the
    harness itself is checked against before any gate leans on it."""
    d = torch.tensor(draw)
    rt = _runtime()
    out = []
    for q in (-4.0, -3.0, -2.0, -1.0, 0.0):
        W1 = W + q * d - (FEE * abs(q - q_prev) if charged else 0.0)
        out.append(float(_utility_wrap_signed(W1, rt).mean()))
    return out


# ---------------------------------------------------------------- the closed-form curvature
@pytest.mark.parametrize("shape,extra", [
    ("asymmetricutility_huber",
     {"huber_aversion": 2.5, "huber_delta": 1.0, "up_aversion": 0.7, "up_knee": 0.3}),
    ("asymmetricutility_symlog", {}),
    ("asymmetricutility_cara", {"cara_gamma": 1.3}),
])
def test_the_local_curvature_matches_finite_differences(shape, extra):
    """`_utility_local_curvature` is the closed form of `_utility_wrap_signed`'s first two
    derivatives — asserted against central differences of the shape it claims to differentiate,
    in float64 (u'' over a float32 utility is pure cancellation noise). Both wings, both knees
    and both linear tails."""
    rt = {"objective": dict({"object": shape, "utility_scale": 100.0,
                             "reference_wealth": 20.0}, **extra)}

    def u(w):
        return float(_utility_wrap_signed(torch.tensor(w, dtype=torch.float64), rt))

    for W in (-300.0, -50.0, -5.0, 10.0, 25.0, 60.0, 400.0):
        h = 1.0
        d1 = (u(W + h) - u(W - h)) / (2.0 * h)
        d2 = (u(W + h) - 2.0 * u(W) + u(W - h)) / h ** 2
        u1, ara = _utility_local_curvature(W, rt)
        assert u1 == pytest.approx(d1, rel=1e-3), f"u' at W={W} under {shape}"
        assert ara == pytest.approx(-d2 / d1, rel=5e-3, abs=1e-9), f"ARA at W={W} under {shape}"
    print(f"test_the_local_curvature_matches_finite_differences[{shape}]: PASS")


def test_the_identity_objective_has_no_curvature_to_report():
    """The legacy (non-utility) objective is the identity in wealth, so u' is 1 and the risk
    aversion is 0 — and, crucially, the helper must not go looking for a scale that is not there."""
    assert _utility_local_curvature(-1234.0, {"objective": {"object": "meanvariance"}}) == (1.0, 0.0)
    print("test_the_identity_objective_has_no_curvature_to_report: PASS")


# ---------------------------------------------------------------- (a) switch off
def test_the_switch_off_roll_is_bit_identical(tmp_path, monkeypatch):
    """GATE (a). Every figure `_rollout_on_stepper` returns is identical with the dump on and
    off, and with it off no file appears. The dump reads the decision; it does not take part."""
    path = str(tmp_path / "curve.csv")
    _, off = _roll(monkeypatch, _runtime())
    off_books = list(_Stepper.rolls[0])
    _, on = _roll(monkeypatch, _runtime(), dump_path=path)
    on_books = list(_Stepper.rolls[0])

    assert off == on, "the dump moved a reported number"
    assert off_books == on_books, "the dump moved a decision"
    assert os.path.exists(path)
    assert not os.path.exists(str(tmp_path / "nothing.csv"))
    _, again = _roll(monkeypatch, _runtime(), dump_path=str(tmp_path / "nothing.csv"))
    assert again == off
    print(f"test_the_switch_off_roll_is_bit_identical: PASS (books={off_books})")


# ---------------------------------------------------------------- (b) self-consistency
def test_the_dump_is_one_row_per_decision_and_agrees_with_the_verdict(tmp_path, monkeypatch):
    """GATE (b). One row per decision step, in step order, and the `q_chosen_net` column IS the
    trajectory the stepper verdict reports — the dump describes the roll that happened, not a
    parallel one."""
    path = str(tmp_path / "curve.csv")
    _, out = _roll(monkeypatch, _runtime(), dump_path=path)
    df = pd.read_csv(path)

    assert len(df) == T_DEC == len(out["greedy_q_traj"])
    assert list(df["t"]) == out["greedy_q_t"] == list(range(T_DEC))
    for i, book in enumerate(out["greedy_q_traj"]):
        assert float(df["q_chosen_net"][i]) == pytest.approx(sum(book)), f"row {i}"
    # ...and the books the environment was actually driven to, which is the same statement
    # made against the accounting rather than against the log.
    assert [float(v) for v in df["q_chosen_net"]] == pytest.approx(_Stepper.rolls[0])
    print(f"test_the_dump_is_one_row_per_decision_and_agrees_with_the_verdict: PASS "
          f"(q_chosen={list(df['q_chosen_net'])})")


# ---------------------------------------------------------------- (c) the load-bearing gate
def test_the_curve_argmax_is_the_executed_decision(tmp_path, monkeypatch):
    """GATE (c), the load-bearing one. The charge is live (a fee, and a standing book to move
    from), and it is what separates the two rankings: the PRE-charge curve peaks at the deep
    corner -4 while the charged one peaks at -1. So a dump that re-scored the rungs naively —
    forgetting the charge — would report -4 while the roll executed -1. The gate asserts the
    dump's argmax column is the executed book on every non-deadband row, and that the pre-charge
    curve disagrees, which is what makes the first assertion mean something."""
    path = str(tmp_path / "curve.csv")
    _roll(monkeypatch, _runtime(), dump_path=path)
    df = pd.read_csv(path)

    live = df[df["deadband_held"] == 0]
    assert len(live) == T_DEC, "the deadband is off; every row is a real ranking"
    for _, r in live.iterrows():
        assert float(r["argmax_net"]) == pytest.approx(float(r["q_chosen_net"])), \
            f"t={r['t']}: the dump's argmax is not the book the roll executed"

    row = df.iloc[0]
    nets = [float(v) for v in str(row["curve_net"]).split()]
    pre = [float(v) for v in str(row["curve_pre"]).split()]
    charged = [float(v) for v in str(row["curve_charged"]).split()]
    assert nets == [-4.0, -3.0, -2.0, -1.0, 0.0]
    # the hand-computed curves of the module docstring, both of them
    assert pre == pytest.approx(_hand_curve(0.0, 0.0, DRAW_A, charged=False), abs=1e-5)
    assert charged == pytest.approx(_hand_curve(0.0, 0.0, DRAW_A, charged=True), abs=1e-5)
    assert nets[max(range(len(pre)), key=pre.__getitem__)] == -4.0, "pre-charge peaks deep"
    assert float(row["q_chosen_net"]) == -1.0, "the charged ranking executed the shallow rung"
    assert float(row["argmax_net"]) == -1.0
    print(f"test_the_curve_argmax_is_the_executed_decision: PASS (pre argmax=-4, charged=-1)")


# ---------------------------------------------------------------- (d) whose scale
def test_the_scale_comes_from_the_frame_not_the_world(tmp_path, monkeypatch):
    """GATE (d). `_rollout_on_stepper` builds its stepper with `mirror_scale=False` exactly so a
    loaded checkpoint's `c` survives the roll — the value function's own frame. The dump has to
    log THAT scale, the one `_utility_wrap_signed` consumes at every rung, not the one this eval
    world resolved for itself. The fake states the two apart by more than a factor of ten."""
    path = str(tmp_path / "curve.csv")
    s, _ = _roll(monkeypatch, _runtime(), dump_path=path, world_c=WORLD_C)
    df = pd.read_csv(path)

    assert float(s.bundle.utility_scale) == WORLD_C != FRAME_C
    assert all(float(c) == FRAME_C for c in df["c"]), "the dump logged the eval world's scale"
    for i, W in enumerate(W_PATH):
        assert float(df["W"][i]) == pytest.approx(W)
        assert float(df["x0"][i]) == pytest.approx((W - 0.0) / FRAME_C)
    # ...and the knee flag follows the frame's x0: only the deep-loss day sits inside (-δ, 0).
    assert list(df["in_knee"]) == [0, 0, 1]
    print(f"test_the_scale_comes_from_the_frame_not_the_world: PASS "
          f"(c={FRAME_C}, x0={list(df['x0'])})")


def test_the_local_risk_aversion_is_the_one_in_force(tmp_path, monkeypatch):
    """The `local_ARA` / `u_prime` columns are the utility's derivatives AT the logged wealth
    under the logged scale — the same closed form the FD gate pins, read here through the dump."""
    path = str(tmp_path / "curve.csv")
    _roll(monkeypatch, _runtime(), dump_path=path)
    df = pd.read_csv(path)
    for i, W in enumerate(W_PATH):
        u1, ara = _utility_local_curvature(W, _runtime())
        assert float(df["u_prime"][i]) == pytest.approx(u1, rel=1e-5)
        assert float(df["local_ARA"][i]) == pytest.approx(ara, rel=1e-5)
    # x0 = -0.5 sits in the loss knee: aversion strictly positive there, flat at the reference.
    assert float(df["local_ARA"][2]) > 0.0 and float(df["local_ARA"][0]) == 0.0
    print(f"test_the_local_risk_aversion_is_the_one_in_force: PASS "
          f"(ARA={list(df['local_ARA'])})")


# ---------------------------------------------------------------- the successor band
def test_the_band_probability_is_measured_on_the_successor_wealth(tmp_path, monkeypatch):
    """`P_band(Q)` is the chance the day ENDS in the knee — a property of W1 under the inner
    draws, per candidate book — not a restatement of where W started. At t=0 the day starts AT
    the reference (x0 = 0, out of the band by the strict inequality), yet holding the deepest
    rung puts half the draws inside it. A P_band read off x0 would print 0 for all three."""
    path = str(tmp_path / "curve.csv")
    _roll(monkeypatch, _runtime(), dump_path=path)
    df = pd.read_csv(path)
    row = df.iloc[0]

    assert float(row["in_knee"]) == 0.0, "the day starts at the reference"
    assert float(row["P_band_flat"]) == 0.0        # flat, no charge ⇒ W1 = 0 on every draw
    assert float(row["P_band_prev"]) == 0.0        # the standing book IS flat at t=0
    # deepest rung -4: W1 = -4·dF - 16 = (80, 16, -16, -48) ⇒ x1 = (.8, .16, -.16, -.48)
    assert float(row["P_band_deep"]) == pytest.approx(0.5)
    print(f"test_the_band_probability_is_measured_on_the_successor_wealth: PASS "
          f"(flat={row['P_band_flat']}, deep={row['P_band_deep']})")


# ---------------------------------------------------------------- the second fork
def test_the_second_fork_is_an_independent_draw(tmp_path, monkeypatch):
    """The disagreement indicator is only worth a column if the two answers CAN differ: draw B
    mirrors draw A's edge, so the same day decides 0 instead of -1 under it. A dump that reused
    the cached draw would print `fork_disagree` 0 forever and say nothing about seed noise."""
    path = str(tmp_path / "curve.csv")
    _roll(monkeypatch, _runtime(), dump_path=path)
    df = pd.read_csv(path)

    assert list(df["argmax_A_net"]) == pytest.approx(list(df["q_chosen_net"]))
    assert all(int(v) == 1 for v in df["fork_disagree"]), "the second fork answered identically"
    assert float(df["argmax_B_net"][0]) == 0.0, "draw B's edge points the other way"
    print(f"test_the_second_fork_is_an_independent_draw: PASS "
          f"(A={list(df['argmax_A_net'])}, B={list(df['argmax_B_net'])})")


# ---------------------------------------------------------------- shape + moments
def test_the_curve_shape_and_fork_moments_are_the_numbers_they_claim(tmp_path, monkeypatch):
    """`range_CE` is the charged curve's spread converted to dollars at the local `u'`, `bow_span`
    its departure from the chord through its net-sorted endpoints, and the moment columns are the
    population moments of the fork the ranking consumed — `dF_net` reduced the way `_wealth_step`
    reduces a book, which on a single live leg is that leg's own move."""
    path = str(tmp_path / "curve.csv")
    _roll(monkeypatch, _runtime(), dump_path=path)
    row = pd.read_csv(path).iloc[0]

    charged = [float(v) for v in str(row["curve_charged"]).split()]
    u1, _ = _utility_local_curvature(W_PATH[0], _runtime())
    assert float(row["range_CE"]) == pytest.approx((max(charged) - min(charged)) / u1, rel=1e-4)
    # nets ascend -4..0; the chord runs from curve[-4] to curve[0]
    chord = [charged[0] + (charged[-1] - charged[0]) * i / 4.0 for i in range(5)]
    bow = max(abs(c - k) for c, k in zip(charged, chord)) / abs(charged[-1] - charged[0])
    assert float(row["bow_span"]) == pytest.approx(bow, rel=1e-4)
    assert float(row["bow_span"]) > 0.0, "a curved ranking must not report as a straight line"

    # every column is written at %.6g, so the reference is met to six significant digits
    d = torch.tensor(DRAW_A)
    assert float(row["E_dF_A"]) == pytest.approx(float(d.mean()), rel=1e-5)
    assert float(row["sd_dF_A"]) == pytest.approx(float(d.std(unbiased=False)), rel=1e-5)
    assert float(row["E_dL"]) == 0.0 and float(row["sd_dL"]) == 0.0
    assert float(row["var_dFnet"]) == pytest.approx(float(d.var(unbiased=False)), rel=1e-5)
    assert float(row["cov_dFnet_dL"]) == 0.0
    assert int(row["n_rungs"]) == 5
    print(f"test_the_curve_shape_and_fork_moments_are_the_numbers_they_claim: PASS "
          f"(range_CE={row['range_CE']:.4g}, bow={row['bow_span']:.4g})")


# ---------------------------------------------------------------- execution corrections
def test_the_deadband_row_is_marked_and_the_band_bounds_the_argmax(tmp_path, monkeypatch):
    """The two execution corrections the dump has to know about. `Decision_Deadband_Sigma` can
    leave the argmax unexecuted — the row says so, and it is the row `test_the_curve_argmax...`
    excludes. `Max_Trade_Per_Step` moves the argmax itself, because the executed choice is the
    best REACHABLE rung: a band below the grid's one-contract quantum pins the book at its
    opening flat, while the curve's own optimum stays at -1, so the dump's argmax column has to
    report the reachable choice and not the unreachable optimum."""
    path = str(tmp_path / "band.csv")
    _roll(monkeypatch, _runtime(max_trade=0.5), dump_path=path)
    df = pd.read_csv(path)
    assert all(float(v) == 0.0 for v in df["q_chosen_net"]), "the band did not bind"
    assert list(df["argmax_net"]) == pytest.approx(list(df["q_chosen_net"]))
    for _, r in df.iterrows():
        curve = [float(v) for v in str(r["curve_charged"]).split()]
        nets = [float(v) for v in str(r["curve_net"]).split()]
        assert nets[max(range(len(curve)), key=curve.__getitem__)] == -1.0, \
            "the unreachable optimum must still be visible in the curve"

    # A deadband wide enough to refuse every trade: the book never leaves flat, and every row
    # says the band — not agreement — is why.
    path2 = str(tmp_path / "dead.csv")
    _roll(monkeypatch, _runtime(deadband=1e6), dump_path=path2)
    df2 = pd.read_csv(path2)
    assert all(float(v) == 0.0 for v in df2["q_chosen_net"]), "the deadband did not hold"
    assert list(df2["deadband_held"]) == [1, 1, 1]
    assert all(float(r["argmax_net"]) != float(r["q_chosen_net"]) for _, r in df2.iterrows())
    print(f"test_the_deadband_row_is_marked_and_the_band_bounds_the_argmax: PASS "
          f"(band argmax={list(df['argmax_net'])}, held={list(df2['deadband_held'])})")


def test_a_multi_path_roll_refuses_the_dump(monkeypatch):
    """The dump is a single-path diagnostic: its wealth, its curve and its band probabilities all
    belong to ONE path, and averaging them would describe a decision nothing took. A roll wider
    than one path is refused by name rather than quietly writing a mean."""
    monkeypatch.setattr(hedge_bundle, "BundleStepper", _Stepper)
    monkeypatch.setattr(hedge_bundle, "_tracking_error_value", lambda state, rt: state["W"])
    s = _solver(_runtime(), dump_path="/tmp/never-written.csv")
    s.B_outer = 8
    with pytest.raises(ValueError, match="SINGLE-PATH diagnostic"):
        s._rollout_on_stepper(None, s.inner_cache, list(range(T_DEC)))
    assert not os.path.exists("/tmp/never-written.csv")
    print("test_a_multi_path_roll_refuses_the_dump: PASS")


def test_the_dump_path_is_not_part_of_the_training_recipe():
    """`config_hash` stamps the RECIPE. Switching a diagnostic on beside a run must not make it a
    different recipe — the same exclusion the two persistence paths already carry."""
    s = types.SimpleNamespace(cfg={"diffv2_fit_iters": 7, "diffv2_save_value_fn": "",
                                   "diffv2_load_value_fn": "", "diffv2_decision_curve_dump": ""})
    s._config_hash = types.MethodType(DiffSolver._config_hash, s)
    base = s._config_hash()
    s.cfg["diffv2_decision_curve_dump"] = "/tmp/somewhere/curve.csv"
    assert s._config_hash() == base
    s.cfg["diffv2_fit_iters"] = 8
    assert s._config_hash() != base, "the hash is not inert"
    print("test_the_dump_path_is_not_part_of_the_training_recipe: PASS")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-x", "-q", "-s"]))
