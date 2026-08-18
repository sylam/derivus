"""`DiffV2_Wealth_Free_Value` — the residual loses its WEALTH input column, so
C_t(market, W[, p]) = u(W) + A_t(market[, p]) and every dependence the continuation has on wealth
is the utility anchor's.

The defect it exists for: the fitted C_t ranks candidate actions partly through A_t's wealth slope.
The action set of one decision moves W1 across a span, and a residual free to carry a linear term
over that span adds a ranking contribution the anchor's curvature has to fight. Take the column
away and the ranking's shape in W is u's by construction rather than by fitting - the argmax then
bends because the utility bends, not because a regression happened to.

The two switches COMPOSE and each owns exactly one column: `DiffV2_Position_State` adds p, this one
removes W, and the layout is (market | [W] | [p]) under every combination. `_in_dim` is the single
source the nets are sized from and `_standardize` builds rows to, which is why the width gate below
holds one against the other rather than against a written-down number.

KILL MATRIX - every mutant APPLIED to the source, this module RUN, the death recorded, the mutant
reverted. 8 mutants, 8 deaths, none survived. Recorded from the run, never from the design: an
earlier form of the wealth-gradient gate poisoned the label with a NaN, and the mutant that keeps
the gradient term while materializing the absent input's grad as ZEROS walked straight through it -
a constant added to a loss has no path back to a parameter, so nothing moved. The gate below states
the mask as INVARIANCE instead (change the label, the fit must not move), which is the property the
mask actually asserts.

| mutant | died at |
| ------ | ------- |
| M1 the W column kept but ZEROED under the switch | `test_the_input_width_is_the_row_the_seam_builds`, `test_the_residual_reads_no_wealth_column`, `test_the_ranking_moves_only_as_the_utility_does`, `test_the_wealth_gradient_leaves_the_measured_columns`, `test_the_value_and_market_terms_still_supervise_the_wealth_free_fit` |
| M2 the `wealth_free` stamp dropped from `_policy_artifact` | `test_the_saved_artifact_stamps_the_wealth_free_value` |
| M3 the stale-checkpoint refusal deleted | `test_a_mismatched_checkpoint_is_refused_by_name` (both directions), `test_a_pre_feature_checkpoint_reads_as_wealth_bearing`, `test_the_saved_artifact_stamps_the_wealth_free_value` |
| M4 the twin loss keeps the W gradient column under the switch | `test_the_wealth_gradient_leaves_the_measured_columns`, `test_the_value_and_market_terms_still_supervise_the_wealth_free_fit` |
| M5 `_in_dim` computed from the stale layout (`md + 1 + position_state`) | `test_the_input_width_is_the_row_the_seam_builds`, `test_the_switch_removes_exactly_one_column_under_either_position_state` |
| M6 `_standardize` drops W unconditionally (the switch escapes) | `test_off_is_the_wealth_bearing_layout`, `test_off_decides_on_the_wealth_bearing_value`, `test_the_input_width_is_the_row_the_seam_builds`, `test_the_ranking_moves_only_as_the_utility_does`, `test_the_wealth_gradient_leaves_the_measured_columns` |
| M7 the ensemble concat keeps the W column under the switch | `test_the_ensemble_branch_drops_the_wealth_column` |
| M8 the FIT's inline row keeps W while `_standardize` drops it (train/deploy mismatch) | `test_the_wealth_gradient_leaves_the_measured_columns`, `test_the_value_and_market_terms_still_supervise_the_wealth_free_fit` |

Every gate runs `_in_dim` / `_standardize` / `_continuation` / `_decide` / `_fit_from_labels` /
`_check_load_provenance` / `_policy_artifact` as UNBOUND functions against a minimal stand-in
solver, the harness pattern of `test_position_state` and `test_churn_lambda`: the fake enumerates
exactly what the seam touches, so a new read is visible as a new attribute. The probe net stashes
the rows it was scored on ON ITSELF - never in a map keyed by `id()`, which recycles.
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from derivus.hedge_solver import DiffSolver, HedgeActionSpace, _DiffV2Residual

CAP = 4.0          # Evaluator.Total_Position_Abs_Limit — the Q_max p is measured in
MD = 2             # privileged market columns
MARKET = 0.5       # the constant market state every candidate's successor is read at
W_COEF = 10.0      # the probe residual's slope on the standardized wealth column


def _runtime(levels=2, lo=-4.0, hi=0.0):
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
            "force_flat_at_end": False,
            "transaction_cost_per_unit": 0.0,
            "bid_offer_spread_bps": 0.0,
            "bid_offer_spread_spec": None,
        },
    }


class _Probe(torch.nn.Module):
    """A residual net that RECORDS every standardized row it scores and returns the linear
    `x @ w` of it. Recording the row is how a gate reads the LAYOUT the seam built; a linear
    response is how it reads what that layout contributes to the ranking."""

    def __init__(self, w):
        super().__init__()
        self.w = w
        self.rows = []

    def forward(self, x):
        self.rows.append(x.detach().clone())
        return x @ self.w


def _weights(wealth_free):
    """The probe's weights for the layout this switch produces: the same market slopes either
    way, and the wealth slope only where a wealth column exists to carry it."""
    market = torch.arange(1.0, MD + 1.0)
    return market if wealth_free else torch.cat([market, torch.tensor([W_COEF])])


def _solver(runtime, wealth_free, position_state=False, T_dec=3, n_steps=4, B=2):
    """The minimal stand-in the seams run on. The frame is the identity z-frame, so a recorded
    row IS the raw state and the gates can read it without undoing a standardization."""
    aspace = HedgeActionSpace(runtime, torch.device("cpu"))
    hedges = runtime["names"]["hedges"]
    s = types.SimpleNamespace(
        aspace=aspace, chunk=64, risk_kappa=0.0, churn_lambda=0.0,
        position_state=position_state, wealth_free=wealth_free,
        force_flat=runtime["accounting"]["force_flat_at_end"],
        t_min=0, T_dec=T_dec, total_abs_limit=aspace.total_abs_limit,
        hedges=hedges, contract_size=aspace.contract_size, device=torch.device("cpu"),
        tradables_sim={r: torch.zeros(n_steps, B) for r in hedges},
        m_mean=torch.zeros(MD), m_std=torch.ones(MD),
        w_mean=torch.tensor(0.0), w_std=torch.tensor(1.0),
        a_bounds=[None] * T_dec, _ensemble=None,
        # CARA: bounded, strictly concave, strictly increasing — so E_inner[u] has an interior
        # maximiser over the action grid and a wealth slope in A can visibly outrank it.
        _u=lambda W: -torch.exp(-W),
    )
    s._wealth_step = types.MethodType(DiffSolver._wealth_step, s)
    s._unwind_kappa = types.MethodType(DiffSolver._unwind_kappa, s)
    s._calendar_kappa = types.MethodType(DiffSolver._calendar_kappa, s)
    s._reposition_charge = types.MethodType(DiffSolver._reposition_charge, s)
    s._check_action_universe = types.MethodType(DiffSolver._check_action_universe, s)
    s._check_calendar_spread = types.MethodType(DiffSolver._check_calendar_spread, s)
    s._in_dim = types.MethodType(DiffSolver._in_dim, s)
    s._standardize = types.MethodType(DiffSolver._standardize, s)
    s._continuation = types.MethodType(DiffSolver._continuation, s)
    return s


#: Per-inner-draw one-step move per contract. Deliberately NOT a martingale: a linear wealth slope
#: in A contributes W_COEF·E[dF]·q to the score, which is exactly the linear-in-Q term the switch
#: removes, and a zero-mean move would hide it behind the expectation.
MOVES = torch.tensor([1.0, -3.0])


def _decide(s, probe, B=2, t=0):
    """Run the real `_decide` from zero wealth on the two-draw world above, with the REAL
    `_continuation` so the layout `_standardize` builds is the layout that ranks."""
    Bi = MOVES.numel()
    dF = MOVES.reshape(1, Bi, 1).expand(B, Bi, 1)
    return DiffSolver._decide(
        s, [probe] * s.T_dec, torch.full((B, Bi, MD), MARKET), dF, torch.zeros(B, Bi),
        torch.zeros(B), t)


def _reference(s, probe):
    """The score `_decide` must produce, recomputed from the grid: E_inner[u(W1) + A], with A read
    off the probe's own recorded row so the arithmetic is the seam's, not a paraphrase of it."""
    grid = s.aspace.grid().reshape(-1)                                   # (c,)
    W1 = grid[:, None] * MOVES[None, :]                                  # (c, Bi)
    a = probe.rows[0].reshape(-1, W1.shape[0], W1.shape[1], probe.w.numel()) @ probe.w
    return (s._u(W1) + a[0]).mean(-1), grid


# --- (a) switch off: the wealth-bearing layout, rows and decisions, unchanged ------------------
def test_off_is_the_wealth_bearing_layout():
    """`_standardize` with the switch off is the pre-change concatenation — (market | W), and
    (market | W | p) under `DiffV2_Position_State`. Killed by dropping W unconditionally."""
    s = types.SimpleNamespace(m_mean=torch.zeros(3), m_std=torch.ones(3),
                              w_mean=torch.tensor(0.0), w_std=torch.tensor(2.0),
                              wealth_free=False)
    market, W, p = torch.randn(5, 3), torch.randn(5), torch.full((5,), -0.5)
    x = DiffSolver._standardize(s, market, W, None)
    assert x.shape == (5, 4)
    assert torch.equal(x, torch.cat([market, (W / 2.0).unsqueeze(-1)], dim=-1))
    on_p = DiffSolver._standardize(s, market, W, p)
    assert torch.equal(on_p, torch.cat([market, (W / 2.0).unsqueeze(-1), p.unsqueeze(-1)], dim=-1))


def test_off_decides_on_the_wealth_bearing_value():
    """The reference run: off, the residual reads the wealth column and the ranking carries its
    slope, so the argmax follows A rather than the anchor and takes the short corner. Every number
    is the independent arithmetic of E_inner[u(W1) + A]. Killed by any wealth-free term escaping
    its switch."""
    s = _solver(_runtime(), wealth_free=False)
    probe = _Probe(_weights(False))
    q, val = _decide(s, probe)
    ref, grid = _reference(s, probe)
    assert probe.rows[0].shape[-1] == MD + 1, 'the wealth column must still be there'
    assert torch.equal(q, torch.full((2, 1), float(grid[ref.argmax()])))
    assert torch.equal(val, ref.max().expand(2))
    assert float(q[0, 0]) == -4.0, 'the wealth slope outranks the anchor at the corner'


# --- (b) the input width is one number, and it is the row the seam builds ----------------------
@pytest.mark.parametrize("wealth_free", [False, True])
@pytest.mark.parametrize("position_state", [False, True])
def test_the_input_width_is_the_row_the_seam_builds(wealth_free, position_state):
    """`_in_dim` sizes every net and `_standardize` builds every row; a gate that checked either
    against a written-down number would pass while they disagreed with each other. So they are
    held against ONE ANOTHER, and the net built at that width has to consume that row.

    Killed by a stale `_in_dim` and by a column kept-but-zeroed in `_standardize` — the two edits
    that move exactly one of the pair."""
    s = _solver(_runtime(), wealth_free, position_state)
    market, W = torch.randn(5, MD), torch.randn(5)
    p = torch.full((5,), -0.5) if position_state else None
    x = s._standardize(market, W, p)
    assert x.shape[-1] == s._in_dim(MD)
    assert _DiffV2Residual(s._in_dim(MD), hidden=4)(x).shape == (5,)


def test_the_switch_removes_exactly_one_column_under_either_position_state():
    """The two switches compose and each owns one column: turning this one on shrinks the layout
    by exactly 1 whether or not p is present, and p SURVIVES it."""
    for position_state in (False, True):
        off = _solver(_runtime(), False, position_state)
        on = _solver(_runtime(), True, position_state)
        assert on._in_dim(MD) == off._in_dim(MD) - 1
    assert _solver(_runtime(), True, True)._in_dim(MD) == \
        _solver(_runtime(), True, False)._in_dim(MD) + 1, 'p must survive the switch'


def test_the_residual_reads_no_wealth_column():
    """The rows the net is actually scored on inside `_decide` are (market[, p]) alone — the
    positive statement behind the width arithmetic, read off the seam rather than off a shape.
    Killed by keeping the column and zeroing it: the width would still be MD + 1."""
    s = _solver(_runtime(), wealth_free=True)
    probe = _Probe(_weights(True))
    _decide(s, probe)
    rows = probe.rows[0]
    assert rows.shape[-1] == MD
    assert torch.equal(rows, torch.full_like(rows, MARKET))


# --- (c) on: the ranking bends only as the utility does ---------------------------------------
def test_the_ranking_moves_only_as_the_utility_does():
    """The gate the whole feature turns on. Two candidates whose W1 differs but whose successor
    market' (and p') is identical: with the wealth column gone the residual scores them on ONE row,
    so A is the same number for both and cannot separate them — the ranking is E_inner[u(W1)]'s,
    exactly.

    The same probe, the same world and the same anchor with the switch OFF pick the OTHER action,
    which is what makes this a measurement rather than a tautology: the wealth slope was worth more
    than the anchor's curvature over the span these two actions induce.

    Killed by dropping W unconditionally (the off run would agree), and by keeping the column
    zeroed (the rows would not collapse to one)."""
    s = _solver(_runtime(), wealth_free=True)
    probe = _Probe(_weights(True))
    q, val = _decide(s, probe)
    rows = probe.rows[0]
    assert rows.unique(dim=0).shape[0] == 1, 'A must see one row, so it ranks nothing'

    ref, grid = _reference(s, probe)
    u_only = s._u(grid[:, None] * MOVES[None, :]).mean(-1)
    assert torch.equal(ref.argsort(), u_only.argsort()), "the order must be the anchor's order"
    assert torch.equal(q, torch.full((2, 1), float(grid[u_only.argmax()])))
    assert torch.equal(val, ref.max().expand(2))
    assert float(q[0, 0]) == 0.0

    off = _solver(_runtime(), wealth_free=False)
    q_off, _ = _decide(off, _Probe(_weights(False)))
    assert float(q_off[0, 0]) == -4.0, 'the fixture must have a wealth slope worth outranking'


def test_the_ensemble_branch_drops_the_wealth_column():
    """`_continuation`'s ensemble path builds the input in its OWN member frames, so it is a
    SECOND place the layout is assembled and the only one `_decide`'s gates never reach. Killed
    by keeping the W column in the ensemble concat."""
    n = 7
    seen = []

    class Rec(torch.nn.Module):
        def forward(self, x):
            seen.append(x.shape[-1])
            return torch.zeros(x.shape[0])

    def _ens(wealth_free):
        return types.SimpleNamespace(
            T_dec=2, a_bounds=[None, None], wealth_free=wealth_free,
            _ensemble=[([Rec(), Rec()], torch.zeros(MD), torch.ones(MD),
                        torch.tensor(0.0), torch.tensor(1.0), None)],
            _u=lambda W: torch.zeros_like(W))

    for wealth_free, width in ((False, MD + 1), (True, MD)):
        s = _ens(wealth_free)
        DiffSolver._continuation(s, None, torch.randn(n, MD), torch.randn(n), 0, None)
        assert seen[-1] == width
        DiffSolver._continuation(s, None, torch.randn(n, MD), torch.randn(n), 0, torch.zeros(n))
        assert seen[-1] == width + 1, 'the ensemble must still carry the p column'


# --- (d) checkpoint provenance ----------------------------------------------------------------
def _stub(wealth_free, position_state=False):
    rt = _runtime()
    s = types.SimpleNamespace(t_min=0, T_dec=3, hedges=list(rt["names"]["hedges"]),
                              position_state=position_state, wealth_free=wealth_free,
                              aspace=HedgeActionSpace(rt, torch.device("cpu")))
    s._check_action_universe = types.MethodType(DiffSolver._check_action_universe, s)
    s._check_calendar_spread = types.MethodType(DiffSolver._check_calendar_spread, s)
    return s


def _ck(**over):
    ck = {"t_min": 0, "T_dec": 3, "md": 1, "hedges": ["A"], "position_state": False,
          "wealth_free": False, "solver_version": "x", "total_position_schedule": None}
    ck.update(over)
    return ck


@pytest.mark.parametrize("saved,run", [(True, False), (False, True)])
def test_a_mismatched_checkpoint_is_refused_by_name(saved, run):
    """A value fn fitted with the wealth column and one fitted without are different functions of
    different states, and the difference is an input COLUMN — so the failure would otherwise
    surface as a `load_state_dict` size mismatch naming a Linear weight. The refusal must name the
    key that caused it, in both directions. Killed by dropping the stamp check."""
    with pytest.raises(ValueError, match="DiffV2_Wealth_Free_Value"):
        DiffSolver._check_load_provenance(_stub(run), _ck(wealth_free=saved), "<ck>", 1)


def test_a_matched_checkpoint_loads():
    """Both directions matched pass the guard — the refusal is about disagreement, not about the
    feature being on. Vacuity check for the pair above."""
    for flag in (False, True):
        assert DiffSolver._check_load_provenance(
            _stub(flag), _ck(wealth_free=flag), "<ck>", 1) == "x"


def test_a_pre_feature_checkpoint_reads_as_wealth_bearing():
    """A checkpoint written before the feature carries no stamp at all: it is wealth-BEARING, so
    it loads under the default and is refused under the switch."""
    old = _ck()
    del old["wealth_free"]
    assert DiffSolver._check_load_provenance(_stub(False), old, "<ck>", 1) == "x"
    with pytest.raises(ValueError, match="DiffV2_Wealth_Free_Value"):
        DiffSolver._check_load_provenance(_stub(True), old, "<ck>", 1)


def test_the_saved_artifact_stamps_the_wealth_free_value():
    """The artifact must stamp every key `_check_load_provenance` refuses on — a stamp the save
    forgets is a run that refuses its own checkpoint on reload. Round-tripped rather than asserted
    against a key list, so deleting the stamp fails the RELOAD. Killed by dropping `wealth_free`
    from `_policy_artifact`."""
    for flag in (False, True):
        s = _solver(_runtime(), flag)
        s.runtime = {"objective": {"utility_scale": 1.0}, "solver": {}}
        s.active = [0]
        s._config_hash = lambda: "h"
        s._frame_stamp = lambda: "f"
        ck = DiffSolver._policy_artifact(s, [], 1, 8, 0.0, [0.0], 0.0)
        assert ck["wealth_free"] is flag
        assert DiffSolver._check_load_provenance(s, ck, "<ck>", 1) == ck["solver_version"]
        with pytest.raises(ValueError, match="DiffV2_Wealth_Free_Value"):
            DiffSolver._check_load_provenance(_stub(not flag), ck, "<ck>", 1)


# --- (e) the twin loss's measured columns are the net's input columns --------------------------
def _fit(wealth_free, w_shift=0.0, y_shift=0.0, m_shift=0.0, B=64, md=MD):
    """Run the real `_fit_from_labels` on one ANCHOR step (`t + 1 == T_dec`: no successor to
    inherit and no early stop, so the fit is the plain twin loss) and return the fitted net.

    Every input is drawn from one seed and the net is built at the width the switch declares, so
    a run with one label SHIFTED differs from the base run in that label alone — which is what
    makes the comparisons below statements about the mask rather than about noise."""
    torch.manual_seed(0)
    net = _DiffV2Residual(md + int(not wealth_free), hidden=8)
    market0, W0 = torch.randn(B, md), torch.randn(B)
    Y, gW, g_market = torch.randn(B), torch.randn(B), torch.randn(B, md)
    s = types.SimpleNamespace(
        wealth_free=wealth_free, position_state=False, use_adv=True, prox=0.0,
        fit_iters=25, fit_tol=0.0, lr=1e-2, cfg={}, _opts={}, T_dec=1,
        _bounds_frozen=False, a_bounds=[None], _breaches=[],
        m_mean=torch.zeros(md), m_std=torch.ones(md),
        w_mean=torch.tensor(0.0), w_std=torch.tensor(1.0),
        _u=lambda W: torch.tanh(W))
    s._standardize = types.MethodType(DiffSolver._standardize, s)
    out = DiffSolver._fit_from_labels(
        s, [net], W0, market0, Y + y_shift, gW + w_shift, g_market + m_shift, 0,
        torch.zeros(B, 1), None)
    return net, out


def _moved(a, b):
    """Did the two fits land on different parameters?"""
    return any(not torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))


def test_the_wealth_gradient_leaves_the_measured_columns():
    """A differential label on a channel the net does not consume has no parameter to reach:
    matching ∂A/∂wn against a measured wealth gradient while wn is absent from the input would
    push that slope into the market columns instead, which is the very slope the switch removes.
    So W leaves the input and the measured-column mask together — the mirror of p, which enters
    the input carrying no gradient label of its own.

    Stated as invariance, because that is what the mask MEANS: change the wealth-gradient label
    and nothing else, and the fitted parameters must come out bit-identical under the switch. Off,
    the same change moves them — which is the proof that the channel is genuinely measured there
    and that this gate is not passing vacuously.

    Killed by keeping the W gradient term under the switch."""
    assert not _moved(_fit(True)[0], _fit(True, w_shift=7.0)[0]), (
        'the wealth-gradient label reached the fit through a channel the net does not consume')
    assert _moved(_fit(False)[0], _fit(False, w_shift=7.0)[0]), (
        'the wealth gradient must be a measured column with the switch off')


def test_the_value_and_market_terms_still_supervise_the_wealth_free_fit():
    """Vacuity check for the invariance above: a net that fitted NOTHING would be invariant to
    every label. The two channels the switch leaves — the value labels and the market gradients —
    must both still move the parameters."""
    base, out = _fit(True)
    assert torch.isfinite(torch.tensor(out["val_loss"])) and out["A_absmean"] > 0.0
    assert _moved(base, _fit(True, y_shift=3.0)[0]), 'the value labels do not reach the fit'
    assert _moved(base, _fit(True, m_shift=3.0)[0]), 'the market gradients do not reach the fit'
