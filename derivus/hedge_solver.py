"""The hedge solver (`Execution_Mode` 'solve_hedge'): the action universe every track shares, the
value-function solvers, the benchmark tracks, the streaming driver and the result assembly, each
documented on the Hedging section's Solver page."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd
import torch

from . import utils
from .hedge_bundle import (
    _loss_side_curvature, _utility_local_curvature, _utility_scale, _utility_wrap_signed, _utility_x,
    MONEY, initial_q_from_runtime, kappa_pair, mirror_frame, terminal_stats, turnover_charge,
    wealth_step,
)

# Bumped whenever the fitted-value-function on-disk/artifact contract changes shape. Stamped
# into every artifact so a loader can tell which solver produced a checkpoint.
SOLVER_VERSION = "diffsolver/2026-08"


def _packed(values):
    """A whole per-rung curve as ONE `%.6g` CSV cell, space separated. Kept in a cell rather than
    spread over columns because a `Total_Position_Schedule` narrows the ladder per step, and a
    per-rung column layout would ragged the header the dump writes exactly once."""
    return " ".join("%.6g" % float(v) for v in values)


def _on_rows(fork, rows):
    """The risk term's fork `(dF, dL, market, live)` (`DiffSolver._ahead`) on the outer paths
    `rows`, or None."""
    if fork is None:
        return None
    dF, dL, market, live = fork
    return dF[:, rows], dL[:, rows], None if market is None else market[rows], live


def _tail_stats(runtime, wT, u_sum=None):
    """Terminal-P&L stats shared by the verdict and the stepper rollout: the objective mean —
    the terminal utility of `wT` under `runtime`, or the accumulated step-sum when the caller
    carries one — and the money tail (`terminal_stats`)."""
    stats = terminal_stats(wT)
    return {"u_mean": float((_utility_wrap_signed(wT, runtime) if u_sum is None else u_sum).mean()),
            "wT_mean": stats["mean"], "wT_p5": stats["p5"], "wT_cvar5": stats["cvar5"]}


@dataclass
class SolverResult:
    """High-level result of a hedge solver. The DP solvers (HindsightDpSolver, DiffSolver)
    produce per-(t, outer-path) grids of `actions` and `values`."""
    solver_name: str
    actions: Any
    values: Any
    value_fn_artifacts: Optional[Any] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def multiseed_summary(runs):
        """Aggregate a track's repeated solves into `v0_mean ± v0_std` (population std). Every
        solver writes its scalar V_0 to `diagnostics`. Multi-seed repeats re-use the cached outer
        paths but advance the inner-MC Sobol stream, so deterministic tracks (hindsight, textbook)
        report `std == 0`."""
        v0 = [float(r.diagnostics["V_0"]) for r in runs]
        mean = sum(v0) / len(v0)
        std = (sum((x - mean) ** 2 for x in v0) / len(v0)) ** 0.5 if len(v0) > 1 else 0.0
        return {"v0_mean": mean, "v0_std": std, "v0_seeds": v0,
                "n_star": runs[0].diagnostics["n_star_0"]}


class HedgeActionSpace:
    """The action universe and the turnover rates every track optimises over and is charged at,
    built once from `(runtime, device)`; documented on the Solver page."""

    documentation = ('Solver', [
        'The action universe, built once from the runtime and shared by every track (the solver,',
        'the benchmarks and the stepper roll), so no two can disagree about which books are',
        'reachable or what a trade costs.',
        '',
        '- `grid()` / `axis_levels()`: the Cartesian universe, each active leg at',
        '  `Training_Action_Grid_Levels_Per_Axis` levels of `linspace(Min_Position, Max_Position)`',
        '  rounded to whole contracts, each inactive leg pinned at 0.',
        '- `allocation_grid(...)`: under `Evaluator.Allocation_Mode` Exposure or Carry_Variance, one',
        '  ladder of rungs on the signed net cover, each split across the legs by the weights in force',
        '  (`rung`), so the search is over how much cover to carry, not over its composition.',
        '- `ladder(...)` / `book(...)`: the whole-contract band of the signed net on the legs a ladder',
        '  trades, and a book as the execution books it: clamped into those legs\' boxes, nothing on',
        '  any other, its net water-filled into the band and apportioned to whole contracts. The',
        '  passive and textbook books are booked by it.',
        '- `kappa(...)`: the per-leg turnover rates at a step (`hedge_bundle.kappa_pair`), which',
        '  `turnover_charge` prices a reposition at. A ranking prices a move at the cross-path mean',
        '  mark; every charge, the verdict\'s and the benchmarks\' as the stepper\'s and its CSV\'s,',
        '  at each path\'s own (`rows`, `per_contract_kappa` reading the mark\'s abs). **Held by**',
        '  `tests/test_action_space_and_artifact.py::test_every_track_charges_each_path_at_its_own_mark`.',
        '',
        'Every track holds its books to one rule (`admissible`): on the step\'s live legs each active',
        'leg inside its box, every other leg at zero, and the signed net inside',
        '`Total_Position_Abs_Limit` and the `Total_Position_Schedule` corridor (`bounds`). Each step\'s',
        'grid keeps the rows it admits (`grid_at`), whatever their gross, so a row parked on an expired',
        'leg cannot carry the live legs past the cap, nor an expired leg\'s box bound the live ones; the',
        'ladder, the passive and textbook books and the bank are built inside the same bounds; and',
        '`Decision_Deadband_Sigma` holds the standing book only while the step admits it, a split off',
        'the step\'s rungs included. **Held by**',
        '`tests/test_action_space_and_artifact.py::test_the_cartesian_grid_caps_the_signed_net_of_a_mixed_sign_book`,',
        '`::test_the_cap_holds_the_book_the_live_legs_carry`,',
        '`::test_a_dead_leg_s_box_bounds_nothing_on_the_legs_still_live` and',
        '`::test_the_deadband_holds_only_a_book_the_step_admits`.',
    ])

    def __init__(self, runtime, device, vol_sim=None):
        self.runtime = runtime
        self.device = device
        # sim-grid per-step annualized vol series (or None), threaded through `kappa` into
        # `per_contract_kappa`'s Vol_Scale
        self.vol_sim = vol_sim
        self.hedges = list(runtime["names"]["hedges"])
        self.n_hedge = len(self.hedges)
        acc = runtime["accounting"]
        limits = acc["position_limits"]
        self.q_lo = torch.tensor([float(limits[r]["min_position"]) for r in self.hedges], device=device)
        self.q_hi = torch.tensor([float(limits[r]["max_position"]) for r in self.hedges], device=device)
        # the boxes on the host too, which `bounds` sums at every decision without a sync
        self.boxes = list(zip(self.q_lo.tolist(), self.q_hi.tolist()))
        self.contract_size = torch.tensor(
            [float(runtime["tradables"][r]["contract_size"]) for r in self.hedges],
            device=device)
        self.total_abs_limit = float(acc["total_position_abs_limit"])
        self.schedule = acc["total_position_schedule"]
        # the band and the deadband are the execution argmax's: the label path passes
        # rate_limit=False, so the training argmax never arms them
        self.max_trade = acc["max_trade_per_step"]
        self.deadband_sigma = acc["decision_deadband_sigma"]
        self.weights = acc["allocation_weights"]
        self.mode = acc["allocation_mode"]
        self.calendar_bps = acc["calendar_spread_bps"]
        self._grid_cache = None
        self._ladder_cache = {}
        solver_cfg = runtime["solver"]
        self.levels = int(solver_cfg["training_action_grid_levels_per_axis"])
        active = solver_cfg["active_hedge_indices"]
        self.active = (list(range(self.n_hedge)) if active is None
                       else [int(i) for i in active])
        self.n_active = len(self.active)
        self.active_mask = torch.zeros(self.n_hedge, device=device)
        self.active_mask[self.active] = 1.0

    def axis_levels(self):
        """Per-hedge 1-D level values in whole contracts: `linspace(lo, hi, levels)` rounded on
        ACTIVE axes, a level repeated by the rounding kept once, a single 0 on inactive axes
        (pinned). The product of these is the action grid."""
        return [torch.linspace(float(self.q_lo[i]), float(self.q_hi[i]), self.levels,
                               device=self.device).round().unique()
                if i in self.active else torch.zeros(1, device=self.device)
                for i in range(self.n_hedge)]

    def grid(self):
        """Mask-aware target-position grid `(n_actions, n_hedge)`, the product of `axis_levels`,
        inactive axes pinned to 0. Cached (deterministic): `grid_at(t)` filters this base grid per
        decision step without rebuilding the meshgrid."""
        if self._grid_cache is None:
            mesh = torch.meshgrid(*self.axis_levels(), indexing="ij")
            self._grid_cache = torch.stack([m.reshape(-1) for m in mesh], dim=-1)
        return self._grid_cache

    @staticmethod
    def knot_at(knots, t):
        """The knot of a step-sorted schedule in force at decision step t: the rightmost with
        `Step <= t`, the first before it (piecewise-constant). The one lookup behind the
        `Total_Position_Schedule` corridor and the `Allocation_Weights` split."""
        return next((k for k in reversed(knots) if k[0] <= t), knots[0])

    def _weights_at(self, t, live=None):
        """The `Allocation_Weights` split in force at decision step t — the rightmost knot with
        `Step <= t` (piecewise-constant; clamps to the first knot for t before it) as an
        `(n_hedge,)` tensor, intersected with the legs that can actually carry a contract and
        renormalized over what is left: the INACTIVE legs (`Active_Hedge_Indices`) and, when the
        caller passes one, the EXPIRED legs (`live`). Absent a table the live active legs weigh
        evenly, and a step where none can carry a contract has no direction (every weight 0).

        The live intersection happens HERE, before the ladder is built, rather than as a filter
        afterwards: a knot's weight on a dead leg is cover the book cannot realize, so leaving it
        in makes every rung advertise a net it masks away from and makes a corridor the live legs
        can satisfy look infeasible."""
        w = (1.0,) * self.n_hedge if self.weights is None else self.knot_at(self.weights, t)[1]
        w = torch.tensor(w, device=self.device) * self.active_mask
        if live is not None:
            w = w * live
        if self.weights is None:
            # no table: even over the legs that can carry a contract, no direction where none can
            return w / max(1.0, float(w.sum()))
        if not float(w.sum()) > 0.0:
            raise ValueError(
                f"Allocation_Weights knot in force at step {t} puts no weight on a LIVE, ACTIVE "
                f"hedge leg — the ladder has nothing to allocate onto")
        return w / w.sum()

    @staticmethod
    def _largest_remainder(x, target):
        """Round `x` `(L, n_hedge)` to whole contracts so each row sums EXACTLY to `round(target)`
        `(L, 1)`: floor every leg, then hand the `round(target) − Σfloor` leftover units, one
        each, to the legs with the largest fractional remainders (Hare/largest-remainder
        apportionment, ties to the lower leg index).

        Naive per-leg rounding does not do this — three legs at `-3.4, -3.3, -3.3` round to `-9`
        for a row that must be `-10`. Flooring is toward −∞ so the leftover is non-negative and a
        short book is apportioned by the same expression as a long one; the leftover is also
        strictly smaller than the number of legs carrying a remainder, so a leg already on an
        integer box edge never receives a unit and no row leaves its box."""
        base = torch.floor(x)
        need = (target.round() - base.sum(-1, keepdim=True)).round()             # (L,1) whole units
        order = torch.argsort(x - base, dim=-1, descending=True, stable=True)
        rank = torch.empty_like(order)
        rank.scatter_(-1, order, torch.arange(x.shape[-1], device=x.device).expand_as(order))
        return base + (rank < need).to(x.dtype)

    def ladder(self, t, live=None):
        """The ladder at decision step t, `(weights, legs, box_lo, box_hi, lo, hi)`: the
        `_weights_at` split over `live`, the indices of the legs it weighs and their `bounds`, the
        net's band snapped INWARD to whole contracts, since a fractional band would emit rungs, or
        book a hedge, that round outside their own mandate."""
        weights = self._weights_at(t, live)
        usable = (weights > 0).to(self.q_lo.dtype)
        box_lo, box_hi, lo, hi = self.bounds(t, usable)
        return (weights, usable.nonzero().flatten().tolist(), box_lo, box_hi, *self.inward(lo, hi))

    @staticmethod
    def inward(lo, hi):
        """The net band `[lo, hi]` snapped INWARD to whole contracts, every integer net inside it."""
        return math.ceil(lo - 1e-9), math.floor(hi + 1e-9)

    def book(self, q, t, live):
        """`q` `(B, n_hedge)` as the execution books it at decision step t: each leg clamped into
        its `ladder` box, zero on a leg the ladder does not trade, the signed net water-filled onto
        the ladder's legs into its band, so inside the cap and the corridor in whole contracts, then
        apportioned to whole contracts carrying that net by `_largest_remainder`."""
        _, legs, box_lo, box_hi, lo, hi = self.ladder(t, live)
        q = self.waterfill(q.clamp(box_lo, box_hi), lo, hi, legs)
        return self._largest_remainder(q, q.sum(-1, keepdim=True))

    def rung(self, t, net, live=None):
        """The whole-contract book `(B, n_hedge)` carrying each signed net cover of `net` `(B, 1)`,
        inside the `ladder` band, along the ladder at decision step t: `Q·w` clamped into the
        weighted legs' boxes, water-filled back onto Q over those legs and rounded by
        `_largest_remainder`."""
        weights, legs, box_lo, box_hi, _, _ = self.ladder(t, live)
        return self._largest_remainder(
            self.waterfill((net * weights).clamp(box_lo, box_hi), net, net, legs), net)

    def allocation_grid(self, t, live=None):
        """The FACTORED action universe at decision step t: a Q-LADDER over the signed NET cover,
        each rung split across the legs by the `Allocation_Weights` knot in force. The solver then
        chooses one number — how much cover to carry — and a deterministic allocator decides where
        it sits.

        The Cartesian grid's quantum on the NET is the per-axis step times the number of legs, so a
        3-leg 9-level book moves the net in jumps larger than any realistic friction can smooth.
        `Training_Action_Grid_Levels_Per_Axis` counts rungs on the TOTAL here instead.

        Construction runs on the legs that can carry a contract — `_weights_at(t, live)` drops the
        inactive and expired ones and renormalizes, and every bound is summed over what survives:
          * the band is the `ladder` band: `bounds` over those legs (their boxes ∩ cap ∩
            corridor), snapped INWARD to whole contracts;
          * each rung is `rung(t, Q, live)`: `Q·w` clamped into the per-leg boxes and water-filled
            back onto Q over the SAME leg set, so nothing puts cover on a leg the schedule zeroed,
            then rounded by `_largest_remainder` to whole contracts summing to Q; duplicate rungs
            collapse.
        Cached per `(t, live)`. The corridor filter downstream in `grid_at` is then a no-op: the
        rungs a caller is offered are rungs it can realize."""
        key = (t, None if live is None else tuple(live.tolist()))
        if key not in self._ladder_cache:
            _, legs, _, _, lo, hi = self.ladder(t, live)
            if lo > hi:
                raise ValueError(
                    f"action ladder empty at step {t} — no whole-contract net is reachable on the "
                    f"weighted live legs {[self.hedges[i] for i in legs]} inside the band "
                    f"{(lo, hi)}")
            if hi - lo > self.levels - 1:
                raise ValueError(
                    f"Allocation_Weights ladder at step {t} spans {hi - lo} contracts over "
                    f"{self.levels} rungs — a quantum of {(hi - lo) / (self.levels - 1):.2f} "
                    f"contracts. Under a weight schedule "
                    f"Training_Action_Grid_Levels_Per_Axis counts rungs on the TOTAL, and a "
                    f"quantum above 1 rebuilds the coarse net grid the schedule exists to "
                    f"replace; raise it to at least {hi - lo + 1}.")
            net = torch.linspace(lo, hi, self.levels, device=self.device).unsqueeze(-1)  # (L,1)
            self._ladder_cache[key] = self.rung(t, net, live).unique(dim=0)
        return self._ladder_cache[key]

    def grid_at(self, t, live=None):
        """Action grid at decision step t: the rows of the base `grid()` — or of the factored
        `allocation_grid(t)` when `Allocation_Weights` is configured — that the step admits on the
        `live` legs (`admissible`). The single per-t filter site every track shares; empty after
        filtering ⇒ infeasible bounds.

        The rule holds the REALIZED book, the position that survives expiry masking: a row parked
        on an expired leg cannot carry the live legs past the cap, nor a corridor-satisfying short
        be masked to 0 and silently under-hedge."""
        rows = self.grid() if self.weights is None else self.allocation_grid(t, live)
        grid = rows[self.admissible(rows, t, live)]
        logging.debug('grid_at step %d admits %d of %d rows', t, grid.shape[0], rows.shape[0])
        if grid.shape[0] == 0:
            raise ValueError(
                f"action grid empty at step {t} — no row lies in the live legs' boxes with its "
                f"signed net in {self.bounds(t, live)[2:]}, Total_Position_Abs_Limit and the "
                f"Total_Position_Schedule corridor")
        return grid

    def bounds(self, t, live=None):
        """The bounds of a book at decision step t on the active legs the mask `live` marks (every
        active leg where None): `(box_lo, box_hi)` `(n_hedge,)`, each such leg's `Position_Limits`
        box and zero on every other leg, and `(lo, hi)`, the band of its signed net, those boxes'
        sum inside `Total_Position_Abs_Limit` and the `Total_Position_Schedule` corridor. The one
        rule every track's book is held to (`admissible`) and built inside (`ladder`, the bank), so
        it is also the support the position state p = Σq/Q_max is queried over."""
        alive = None if live is None else live.tolist()
        legs = [i for i in self.active if alive is None or alive[i] > 0]
        on = self.active_mask if live is None else self.active_mask * live
        lo = sum(self.boxes[i][0] for i in legs)
        hi = sum(self.boxes[i][1] for i in legs)
        if self.total_abs_limit > 0.0:
            lo, hi = max(lo, -self.total_abs_limit), min(hi, self.total_abs_limit)
        if self.schedule is not None:
            c_lo, c_hi = self.knot_at(self.schedule, t)[1:]
            lo, hi = max(lo, c_lo), min(hi, c_hi)
        logging.debug('bounds at step %d: live active legs %s, net band (%s, %s)', t, legs, lo, hi)
        return self.q_lo * on, self.q_hi * on, lo, hi

    def admissible(self, q, t, live=None):
        """`(K,)`: whether each book of `q` `(K, n_hedge)`, held on the legs the mask `live` marks
        (every leg where None), is one decision step t admits (`bounds`): each live active leg
        inside its box, every other leg at zero, and its signed net inside the cap and the
        corridor."""
        box_lo, box_hi, lo, hi = self.bounds(t, live)
        q = q if live is None else q * live
        net = q.sum(-1)
        return ((q >= box_lo) & (q <= box_hi)).all(-1) & (net >= lo - 1e-9) & (net <= hi + 1e-9)

    def in_band(self, dq):
        """Whether each move `dq` `(..., n_hedge)` lies inside `Max_Trade_Per_Step`: its net under
        `Allocation_Weights`, the control being the net cover so a composition jump at a knot is the
        schedule's mandate, each leg's move without one."""
        if self.weights is not None:
            return dq.sum(-1).abs() <= self.max_trade
        return (dq.abs() <= self.max_trade).all(-1)

    def waterfill(self, q, lo, hi, idx=None):
        """Shift the ACTIVE legs of `q` `(B, n_hedge)` so their signed total lands in `[lo, hi]`,
        each leg staying inside its own `[Min,Max]`: the deficit `d = clamp(Σq, lo, hi) − Σq` is
        distributed in proportion to each leg's remaining room in d's direction. `lo`/`hi` are
        scalars (the corridor) or `(B,1)` tensors (a per-path net target). Feasible whenever the
        headroom covers `|d|` — the same feasibility `grid_at` fails loud on.

        `idx` narrows which legs absorb the deficit (default: every active leg). The ladder passes
        the WEIGHTED live legs, because a leg the schedule zeroed has a full box of headroom and
        would otherwise be handed cover the schedule said it must not carry."""
        idx = self.active if idx is None else idx
        qa = q[:, idx]                                                   # (B, n_active)
        tot = qa.sum(-1, keepdim=True)                                   # (B,1) signed active total
        d = tot.clamp(lo, hi) - tot                                     # (B,1) deficit
        head = torch.where(d >= 0, self.q_hi[idx] - qa, qa - self.q_lo[idx]).clamp_min(0.0)
        out = q.clone()
        out[:, idx] = qa + d * head / head.sum(-1, keepdim=True).clamp_min(1e-9)
        return out

    def install_weights(self, weights):
        """One-shot install of a DERIVED `Allocation_Weights` schedule (Allocation_Mode
        'Carry_Variance'): the solver computes the table from the warmup sims — or restores a
        checkpoint's stamped table — and the ladder reads it from here on. Refused outside the
        derivation mode or over a standing table, so a declared universe can never be silently
        replaced."""
        if self.mode != "carry_variance":
            raise ValueError(
                "HedgeActionSpace.install_weights outside Allocation_Mode='Carry_Variance' — "
                "a declared universe is data, not something a solver may overwrite")
        if self.weights is not None:
            raise ValueError("HedgeActionSpace.install_weights: a weights table already stands")
        self.weights = self.weights_key(weights)

    def kappa(self, tradables_sim, t_index, rows=None, matched=True):
        """The per-hedge turnover rates `(κ, κ_cal)` at sim-grid `t_index` (`hedge_bundle.kappa_pair`):
        `(n_hedge,)` at each instrument's cross-path mean mark, the price a RANKING reads, or
        `(n, n_hedge)` at the own marks of the paths `rows` in `MONEY`, the price a CHARGE reads.
        The step's scalar vol drives a Vol_Scale spread; `matched=False` prices every lot outright."""
        vol = None if self.vol_sim is None else self.vol_sim[int(t_index)]
        marks = {r: tradables_sim[r][t_index] for r in self.hedges}
        return kappa_pair(self.runtime, {r: m.mean() if rows is None else m[rows].to(MONEY)
                                         for r, m in marks.items()}, vol, matched)

    def universe_size(self):
        """Row count of the action universe a diagnostic should report: the entry step's grid, the
        Cartesian one inside its bounds or the LADDER under `Allocation_Weights`. Never the meshgrid
        under a schedule — at 5 legs and 51 levels that is ~345M rows built and discarded for a
        51-row search."""
        return int(self.grid_at(0).shape[0])

    def static_grid(self):
        """The universe a CONSTANT hold is chosen from (`run_textbook_benchmark`). Cartesian: the
        entry-step grid. Under `Allocation_Weights`: every knot's ladder, deduped and held to the
        ENTRY corridor — every composition the schedule offers, any one of which a hold could have
        carried from day one.

        Not just the entry knot: the DP recomposes at every knot and a constant hold cannot, so
        pinning the benchmark to the entry ray would put an action-space difference inside the
        DP-vs-textbook gap. The hold still cannot TRACK the schedule, only choose which
        composition to sit in."""
        if self.weights is None:
            return self.grid_at(0)
        rows = torch.cat([self.allocation_grid(step) for step, _ in self.weights]).unique(dim=0)
        return rows[self.admissible(rows, 0)]

    @staticmethod
    def schedule_key(schedule):
        """Canonical, comparison-stable form of a `Total_Position_Schedule` (None or the sorted
        `(step, min, max)` knot tuple) — flattens tuple-vs-list and int/float drift so a
        checkpoint's stored corridor compares against the run's regardless of round-trip form."""
        return None if schedule is None else tuple(
            (int(step), float(lo), float(hi)) for step, lo, hi in schedule)

    @staticmethod
    def weights_key(weights):
        """Canonical, comparison-stable form of an `Allocation_Weights` schedule (None or the
        sorted `(step, weights)` knot tuple), rounded so a JSON round-trip of the normalized
        weights compares equal. The `schedule_key` of the action space's other schedule."""
        return None if weights is None else tuple(
            (int(step), tuple(round(float(w), 9) for w in ws)) for step, ws in weights)


def run_textbook_benchmark(bundle, runtime):
    """The comparison track's `textbook` (Solver page, Benchmarks and the verdict): the best
    constant book of `HedgeActionSpace.static_grid` by E[u], held frictionless to the terminal, its
    entry from the opening book and unwind there reported as a net-of-cost diagnostic."""
    T = bundle.last_live_mtm_index
    L_T, device = bundle.liability_sim[T], bundle.device
    acc = runtime["accounting"]
    aspace = HedgeActionSpace(runtime, device, bundle.vol_sim)
    tradables_sim = bundle.tradables_sim
    # entry-step corridor only: a constant hold cannot track a per-t one
    grid = aspace.static_grid()                                      # (n_actions, n_hedge)
    q0 = initial_q_from_runtime(runtime, bundle.batch_size, device)   # (B, n_hedge) opening book

    total_move = bundle.move(0, T, aspace.hedges, dim=0)               # (n_h, B) telescoped
    g_t = torch.einsum("ai,ib->ab", grid * aspace.contract_size, total_move)  # (n_actions, B)
    u = _utility_wrap_signed(L_T.unsqueeze(0) + g_t, runtime)         # FRICTIONLESS objective
    obj = u.mean(dim=-1)                                               # (n_actions,)
    best = int(obj.argmax())
    n_star = grid[best]                                                # (n_hedge,)
    # Net-of-cost diagnostic (shared kappa): entry |n_star − q0| + terminal unwind |0 − n_star|.
    paths = slice(None)                                                # each path's own mark
    cost = turnover_charge(n_star.unsqueeze(0) - q0,                   # (B,) entry from q0
                           *aspace.kappa(tradables_sim, 0, rows=paths))
    if acc["force_flat_at_end"]:
        # unwound at the PRE-SETTLEMENT row, since the last row is the post-settlement clean exit
        # where there is nothing left to liquidate. Outright: flat against the market, not leg
        # against leg.
        cost = cost + turnover_charge(n_star, *aspace.kappa(tradables_sim, T, paths, matched=False))
    u_net = _utility_wrap_signed(L_T + g_t[best] - cost, runtime)
    return {"v0_mean": float(obj[best]), "v0_std": 0.0,
            "v0_mean_net": float(u_net.mean()),
            "turnover_cost_mean": float(cost.mean()),
            "n_star": n_star.detach().cpu().tolist()}


class HindsightDpSolver:
    """The clairvoyant, frictionless upper-bound track; documented on the Solver page."""

    documentation = ('Solver', [
        'The clairvoyant upper bound, run under `Run_Hindsight_Diagnostic`. On each realized',
        'held-out path it holds the opening book to the first decision, then at every decision row',
        'from `T_Min` (`Bundle.decisions_from`), the rows the solver decides on, picks the grid book',
        'maximising its realized P&L to the next decision (`Bundle.successors`): perfect foresight and',
        'free repositioning, with no inner Monte Carlo and no fitted value. Frictionless, its per-step',
        'choices decouple, so the dynamic program is a per-step argmax and its mean value bounds any',
        'deployable policy; the turnover its path would pay is a net-of-cost diagnostic, never charged.',
        '**Held by**',
        '`tests/test_action_space_and_artifact.py::test_the_hindsight_bound_decides_where_the_solver_does`.',
    ])

    def __init__(self, bundle, runtime):
        self.bundle = bundle
        self.runtime = runtime
        self.aspace = HedgeActionSpace(runtime, bundle.device, bundle.vol_sim)

    def solve(self):
        bundle, runtime, aspace = self.bundle, self.runtime, self.aspace
        acc = runtime["accounting"]

        L_T, device = bundle.liability_sim[bundle.last_live_mtm_index], bundle.device
        b_outer = bundle.batch_size
        tradables_sim = bundle.tradables_sim
        cs = aspace.contract_size

        # frictionless clairvoyant: independent per-decision argmax over the realized move to the
        # decision's successor, within the per-t action universe. The net-of-cost trajectory is
        # accumulated for the diagnostic.
        q_prev = initial_q_from_runtime(runtime, b_outer, device)    # (B, n_h) opening book
        steps = bundle.decisions_from(runtime["solver"]["t_min"])
        logging.debug('hindsight bound decides on rows %s', steps)
        n_star_0 = None
        # the opening book held to the first decision
        G = torch.einsum("bi,ib->b", q_prev * cs, bundle.move(0, steps[0], aspace.hedges, dim=0))
        cost = torch.zeros(b_outer, device=device)
        paths = slice(None)                                            # each path's own mark
        for t in steps:
            grid = aspace.grid_at(t, bundle.marked(t, aspace.hedges))  # (n_actions_t, n_h) per-t
            dF = bundle.move(t, bundle.successors[t], aspace.hedges, dim=0)  # (n_h, B)
            step_pnl = torch.einsum("ai,ib->ab", grid * cs, dF)        # (n_actions_t, B)
            best_pnl, best_idx = step_pnl.max(dim=0)                   # (B,), (B,)
            q_now = grid[best_idx]                                     # (B, n_h)
            cost = cost + turnover_charge(q_now - q_prev, *aspace.kappa(tradables_sim, t, rows=paths))
            G = G + best_pnl
            q_prev = q_now
            if n_star_0 is None:
                n_star_0 = q_now
        if acc["force_flat_at_end"]:
            # pre-settlement row, as in `run_textbook_benchmark` and `DiffSolver._unwind_kappa`
            cost = cost + turnover_charge(q_prev, *aspace.kappa(
                tradables_sim, int(bundle.last_live_mtm_index), paths, matched=False))
        v0 = _utility_wrap_signed(L_T + G, runtime)                    # (B,) FRICTIONLESS
        v0_net = _utility_wrap_signed(L_T + G - cost, runtime)         # (B,) net-of-cost diagnostic

        return SolverResult(
            solver_name="HindsightDpSolver",
            actions=n_star_0.detach().cpu(),
            values=v0.detach().cpu(),
            diagnostics={
                "V_0": float(v0.mean()),
                "V_0_net": float(v0_net.mean()),
                "turnover_cost_mean": float(cost.mean()),
                "n_star_0": n_star_0.float().mean(dim=0).detach().cpu().tolist(),
                "v0_abs_max": float(v0.abs().max()),
                "action_grid_size": aspace.universe_size(),
            },
        )


class _DiffV2Residual(torch.nn.Module):
    """Zero-init residual head `A_t(market, q, W)`. The continuation is `C = u(W) + A`, so a zero
    final layer makes C start exactly at the bounded utility anchor. Under the successor chain only
    the TERMINAL net starts there; every earlier net starts at its fitted successor, and the
    trust-region clamp (`a_bounds`) is the brake covering the inherited starts."""

    def __init__(self, in_dim, hidden):
        super().__init__()
        self.body = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, 1),
        )
        for p in self.body[-1].parameters():
            torch.nn.init.zeros_(p)

    def forward(self, x):
        return self.body(x).squeeze(-1)


class DiffSolver:
    """The differential-ML value-function solver: a backward dynamic program fitted by the twin loss
    on inner Monte Carlo labels, its maximum an external grid search; documented on the Solver
    page."""

    documentation = ('Solver', [
        'The solver the Method section describes, its inner Monte Carlo forks priced through the',
        'ordinary deal pricers. `C_t = u(W) + A_t`: the bounded utility anchor (`DiffSolver._anchor`)',
        'plus a zero-initialised residual net, the only learned part, which keeps the backward',
        'recursion from running away; the training bank rolls the outer paths exploring books around',
        "each step's replication hedge, so each net is fitted where the policy lives.",
        '',
        '### Persistence',
        '',
        '`DiffV2_Save_Value_Fn` writes the fitted nets with their frame; `DiffV2_Load_Value_Fn`',
        'restores them and fits nothing, a frozen evaluation (`StreamingSolve`), a list loading an',
        'ensemble (the ensemble rule above). A checkpoint stamps what makes it a different function',
        'of a different state (`_provenance`), and a load differing on one is refused by name.',
        '',
        '!!! warning "The frame is locked at warmup"',
        '    The utility scale, the market and wealth standardisation and the per-step trust region',
        '    are measured on the first batch and frozen: refitted per batch, each `C_t` would be a',
        '    different function of different inputs and the recursion would compose mismatched',
        '    frames. Later batches report how often their fitted targets fall outside the frozen',
        '    region, and a loaded checkpoint evaluates in its own frame, measuring none of its own.',
        '    **Held by**',
        '    `tests/test_action_space_and_artifact.py::test_a_frozen_roll_measures_no_frame_of_its_own`;',
        '    **Not gated**: a later training batch resolving a scale of its own.',
    ])

    # Bank-policy provenance: the name stamped into checkpoints and compared on load.
    SOLVER_OBJECT = "DiffSolver"

    def __init__(self, bundle, runtime):
        """Build the solver against `bundle`: the shared action universe, the knobs, the bank RNG and
        the still-unlocked frame, then `_bind` to the bundle's sim views. The horizon `T_dec` is the
        bundle's close, the last live liability mark (`Bundle.last_live_mtm_index`)."""
        self.bundle = bundle
        self.runtime = runtime
        self.cfg = runtime["solver"]
        self.device = bundle.device
        self.aspace = HedgeActionSpace(runtime, self.device, bundle.vol_sim)
        self.chunk = int(self.cfg["training_action_chunk_size"])
        self.fit_iters = self.cfg["diffv2_fit_iters"]
        self.lr = self.cfg["diffv2_lr"]
        self.noise_frac = self.cfg["diffv2_bank_noise_frac"]
        self.t_min = self.cfg["t_min"]
        self.risk_kappa = self.cfg["diffv2_risk_kappa"]
        # with no penalty there is no horizon to hold the book over
        self.risk_horizon = self.cfg["diffv2_risk_horizon"] if self.risk_kappa > 0.0 else 1
        charge = self.cfg["diffv2_repositioning_charge"]
        self.argmax_charged = charge in ("argmax", "state")
        self.churn_lambda = self.cfg["diffv2_churn_lambda"]
        # the drift monitor's null: the unit z until a training verdict measures one or a load
        # stamps one, the value every reader takes
        self.drift_null_sigma = 1.0
        self.position_state = charge == "state"
        logging.debug("DiffV2_Repositioning_Charge %s: argmax charged %s, position state %s",
                      charge, self.argmax_charged, self.position_state)
        self.book_label = self.position_state and self.cfg["diffv2_book_gradient_label"]
        self.label_horizon = self.cfg["diffv2_label_horizon"]
        self.bank_ladder = self.cfg["diffv2_bank_along_ladder"]
        self.wealth_free = self.cfg["diffv2_wealth_free_value"]
        if self.wealth_free and not self.position_state:
            raise ValueError(
                "Solver.DiffV2_Wealth_Free_Value='Yes' requires "
                "DiffV2_Repositioning_Charge='State': an "
                "action cannot move the market, so without p the residual A(m') is one constant "
                "across every candidate and cancels from the argmax - the whole fit would be "
                "decision-irrelevant and the run a myopic one-step E[u(W1)] policy wearing a "
                "DP's verdict. Build the myopic baseline deliberately if it is wanted.")
        if self.wealth_free and self.risk_kappa > 0.0:
            raise ValueError(
                "Solver.DiffV2_Wealth_Free_Value='Yes' refuses DiffV2_Risk_Kappa > 0: the downside "
                "semideviation is nonlinear over inner draws, so the residual's dispersion leaks "
                "back into the ranking the switch exists to hand to the utility alone.")
        obj = runtime["objective"]
        self.running_wealth = obj["reference_mode"] == "running_wealth"
        self.log_ratio = obj["object"] == "logwealth"
        self.aversion = self.cfg["diffv2_risk_aversion"]
        # measured off the bank at warmup, or the checkpoint's (`_restore_frame`)
        self.displacement = self.capital = 0.0
        self.scheduled_scale = obj["utility_scale_mode"] == "conditional_sim"
        if self.running_wealth and self.scheduled_scale and not self.log_ratio:
            raise ValueError(
                "Objective.Reference_Mode='Running_Wealth' refuses "
                "Utility_Scale_Mode='conditional_sim': the increment already carries a per-step "
                "scale of its own (one day's move, not the whole horizon's), so a second per-step "
                "rescaling of it is a different objective again. Each is a separate experiment; "
                "run them one at a time. (LogWealth is the exception: the schedule is its "
                "CAPITAL line, not a rescaling.)")
        if self.running_wealth and self.cfg["diffv2_temporal_proximity"] > 0.0:
            raise ValueError(
                "Objective.Reference_Mode='Running_Wealth' refuses DiffV2_Temporal_Proximity > 0: "
                "the proximity prior penalises the distance from A_t to its fitted SUCCESSOR, "
                "which is the right prior only while C_t ≈ C_{t+1}. Under a per-step reward the "
                "two differ by E[u(ΔW_t)] BY DEFINITION, so the regularizer pulls against exactly "
                "the offset the objective is made of. The successor WARM START is untouched and "
                "still valid — it is an initial point, not a penalty.")
        self.force_flat = bool(runtime["accounting"]["force_flat_at_end"])
        self.curve_dump = self.cfg["diffv2_decision_curve_dump"]
        passive_from = self.cfg["diffv2_passive_from"]
        self.passive_from = pd.Timestamp(passive_from) if passive_from else None
        self.first_from_passive = self.cfg["diffv2_first_decision_from"] == "passive"
        if self.position_state and not self.aspace.total_abs_limit > 0.0:
            raise ValueError(
                "Solver.DiffV2_Repositioning_Charge='State' requires "
                "Evaluator.Total_Position_Abs_Limit > 0 "
                f"(it is {self.aspace.total_abs_limit}): the position state is p = Sum(q)/Q_max and the "
                "cap is the scale it is measured in")
        self.fit_tol = self.cfg["diffv2_fit_tol"]
        self.prox = self.cfg["diffv2_temporal_proximity"]
        # bank exploration RNG - deterministic, and PERSISTENT across batches so each batch
        # explores a fresh noise draw
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(0)
        # per-t Adam optimizers, created at the first fit of t and KEPT, so a streaming step
        # continues the same moments on the same net
        self._opts = {}
        # the locked frame. `utility_scale` is None until the first bind completes, which is what
        # marks that bind as the warmup one; `_bounds_frozen` turns `a_bounds` from
        # fitted-per-batch into frozen-and-reported.
        self.utility_scale = None
        self._bounds_frozen = False
        self._breaches = []
        self._bind(bundle)
        self.T_dec = int(bundle.last_live_mtm_index)
        if self.t_min >= self.T_dec:
            raise ValueError(
                f"Solver.T_Min={self.t_min} must be < decision horizon T_dec={self.T_dec}")
        self.steps, self.nxt = bundle.decision_steps, bundle.successors
        self.prev = dict(zip(self.steps[1:], self.steps))
        # the row each decision's risk term lands on: its risk_horizon-th successor, T_dec past it
        held = self.steps + [self.T_dec]
        self.landing = {t: held[min(i + self.risk_horizon, len(self.steps))]
                        for i, t in enumerate(self.steps)}
        self.utility_scale = float(bundle.utility_scale)      # locked here; re-asserted by _bind
        # ...and the per-step schedule with it: ONE frame, locked in one place. A streaming step
        # re-binds to a batch that measured its own, which must not reach the reward.
        self.utility_scale_schedule = bundle.utility_scale_schedule
        if (self.scheduled_scale and self.utility_scale_schedule is None
                and not self.cfg["diffv2_load_value_fn"]):
            raise ValueError(
                "Objective.Utility_Scale_Mode='conditional_sim' but this warmup batch measured no "
                "schedule: c_t is the cross-sectional dispersion of the flat-book wealth across "
                "OUTER PATHS, and a single-path batch has no cross-section. Raise Batch_Size (a "
                "one-path world can only be a frozen roll, which takes its schedule from the "
                "checkpoint).")

    def _bind(self, bundle):
        """Point the solver at a bundle: the history-stripped sim views the sweep indexes by `t`
        and the friction vol series the action space prices with. A frozen eval binds once at
        construction; a streaming step re-binds to each fresh batch, which is also where the
        LOCKED utility scale and its per-step schedule are re-asserted — every batch resolves its
        own frame at build time, and letting a later one reach the runtime would rescale the
        objective, and every fitted C_t with it, mid-training."""
        self.bundle = bundle
        self.aspace.vol_sim = bundle.vol_sim
        self.tradables_sim = bundle.tradables_sim
        self.liability_sim = bundle.liability_sim                     # (n_steps, B_outer)
        self.leg_volume = float(bundle.total_leg_volume)
        self.B_outer = int(self.liability_sim.shape[-1])
        if self.utility_scale is None:
            return                                                    # construction; nothing locked yet
        if int(bundle.last_live_mtm_index) != self.T_dec:
            raise ValueError(
                f"streaming batch has decision horizon T_dec={int(bundle.last_live_mtm_index)} "
                f"but the frame was locked at {self.T_dec} — every batch must share one time grid")
        logging.info("DiffSolver bound batch: B_outer=%d | frame utility_scale=%.6g "
                     "(this batch resolved %.6g)", self.B_outer, self.utility_scale,
                     float(bundle.utility_scale))
        mirror_frame(self.runtime, self.utility_scale, self.utility_scale_schedule)

    def _config_hash(self):
        """sha1 of the stable-JSON solver cfg — the stamp identifying this TRAINING RECIPE.
        The persistence paths are excluded so the same recipe hashes identically regardless of
        where its checkpoint lives — and the decision-curve dump path with them, because a
        diagnostic that is switched on beside a run must not make it a different recipe, and the
        two convergence bounds, which accept or refuse a fit without changing it."""
        stable = {k: v for k, v in self.cfg.items()
                  if k not in ("diffv2_save_value_fn", "diffv2_load_value_fn",
                               "diffv2_decision_curve_dump", "diffv2_max_value_optimism",
                               "diffv2_max_member_disagreement")}
        return hashlib.sha1(json.dumps(stable, sort_keys=True, default=str).encode()).hexdigest()

    # ---- utility anchor ------------------------------------------------------
    def _anchor(self, W, t, schedule=None):
        """The anchor u(W) of `C_t = u(W) + A_t` at decision step t, the one the fit subtracts and
        the read adds back: at the run's step knee, or under `schedule` at ONE ensemble member's
        own, since a member's `utility_scale_schedule` is part of its frame exactly as its z-stats
        are (see `_restore_frame`). Under a step-sum objective (Running_Wealth / LogWealth) the
        continuation carries NO terminal utility, so the anchor is exactly zero rather than a wrap
        call the growth objective refuses by name."""
        if self.running_wealth:
            return torch.zeros_like(W)
        if schedule is None:
            return _utility_wrap_signed(W, self.runtime, t)
        return _utility_wrap_signed(
            W, {"objective": dict(self.runtime["objective"],
                                  utility_scale_schedule=schedule)}, t)

    # ---- action grid (shared, mask-aware; inactive axes pinned to 0) ---------
    # ---- input standardization ----------------------------------------------
    def _in_dim(self, md):
        """Input width of a residual net: the `md` privileged market columns, plus W unless
        `DiffV2_Wealth_Free_Value` removes it, plus p under the position state. The single
        source `warmup` sizes every net from and `_standardize` builds every row to, so a switch
        can only move the two together."""
        return md + int(not self.wealth_free) + int(self.position_state)

    def _standardize(self, market, W, p):
        """Standardized state for the residual net: `(market | W)`, plus the signed net book
        fraction `p = Sum(q)/Q_max` as a trailing column under the position state. `p is None`
        IS the switch at this level.

        Market and wealth are z-scored against the locked bank frame; p is NOT, because it is
        already a fraction of the cap on both sides of the seam — `grid_at` keeps only the actions
        the step admits inside `Total_Position_Abs_Limit` (`HedgeActionSpace.admissible`) and
        `_build_bank` water-fills each sampled book onto `aspace.bounds(t)` — so p arrives in
        [-1, 1] over a support the bank covers.

        `DiffV2_Wealth_Free_Value` drops the W column instead of adding one. The two switches
        compose, each owning one column, so the layout is `(market | [W] | [p])` in that order
        under every combination."""
        cols = [(market - self.m_mean) / self.m_std]
        if not self.wealth_free:
            cols.append(self._w_norm(W).unsqueeze(-1))
        return torch.cat(cols if p is None else cols + [p.unsqueeze(-1)], dim=-1)

    def _w_norm(self, W, w_mean=None, w_std=None):
        """The wealth column as every net reads it: W in fractions of the t0 book notional under
        `DiffV2_Returns_State` (the frame `w_mean`/`w_std` was measured in), z-scored. One spelling
        for the fit, the query and an ensemble member, so no path feeds money to a fraction frame."""
        sn = getattr(self, "state_notional", None)
        w = W / sn if sn else W
        return (w - (self.w_mean if w_mean is None else w_mean)) / (self.w_std if w_std is None else w_std)

    def _continuation(self, nets, market, W, t, p, chunk=400_000):
        """`C_t = u(W) + A_t(market, W[, p])`; terminal `C_{T_dec} = u(W)`. Row-chunked net eval.
        Under `DiffV2_Wealth_Free_Value` the residual is `A_t(market[, p])` and W enters through the
        anchor alone; `_standardize` owns the layout either way.

        Under `Objective.Reference_Mode='Running_Wealth'` the anchor is IDENTICALLY ZERO and
        `C_t = A_t` alone, with terminal `C_{T_dec} ≡ 0`: once the reward is the day's increment
        there is no terminal utility of the level to anchor on. The per-step reward is added by the
        CALLER, the only place holding both W1 and the wealth it moved from.

        Ensemble mode (list-of-checkpoints load): `A` = mean over members, each evaluated in its
        OWN standardization frame — which under a `conditional_sim` schedule includes that member's
        own step knee.

        `A_t` is CLAMPED to its fitted-target trust region (one range-width of headroom): off
        support the zero-init MLP extrapolates freely, and the argmax then chases a phantom
        direction that poisons the t−1 bootstrap labels. Outside the clamp the gradient is zero,
        so the differential labels ignore those directions too."""
        base = self._anchor(W, t)
        ensemble = getattr(self, "_ensemble", None)
        if t >= self.T_dec and not ensemble:
            return base
        if ensemble:
            acc = torch.zeros_like(base)
            for m_nets, m_mean, m_std, w_mean, w_std, m_bounds, m_sched, _m_disp in ensemble:
                if m_sched is not None:
                    # the member's own step KNEE, as much a part of its frame as its z-stats.
                    # Written as a correction to the shared anchor so a schedule-free ensemble
                    # stays arithmetically untouched.
                    acc = acc + (self._anchor(W, t, m_sched) - base)
                if t >= self.T_dec:
                    # the TERMINAL has no residual, so this correction is the WHOLE of the
                    # member's continuation - which is what the last decision is ranked on
                    continue
                cols = [(market - m_mean) / m_std]
                if not self.wealth_free:
                    cols.append(self._w_norm(W, w_mean, w_std).unsqueeze(-1))
                x = torch.cat(cols + ([] if p is None else [p.unsqueeze(-1)]), dim=-1)
                b = m_bounds[t] if m_bounds is not None else None
                for i in range(0, x.shape[0], chunk):
                    a = m_nets[t](x[i:i + chunk])
                    acc[i:i + chunk] += a if b is None else torch.clamp(a, b[0], b[1])
            return base + acc / len(ensemble)
        x = self._standardize(market, W, p)
        b = self.a_bounds[t]
        if x.shape[0] <= chunk:
            a = nets[t](x)
            return base + (a if b is None else torch.clamp(a, b[0], b[1]))
        out = torch.empty_like(base)
        for i in range(0, x.shape[0], chunk):
            a = nets[t](x[i:i + chunk])
            out[i:i + chunk] = a if b is None else torch.clamp(a, b[0], b[1])
        return base + out

    # ---- friction: the one charge both the ranking and the label pay ---------
    def _unwind_kappa(self, t, rows=None):
        """The terminal liquidation kappa when decision `t`'s successor IS the terminal mark and
        `Force_Flat_At_End` forces the book flat there — else None. Under the position state
        the last decision has to price the unwind of whatever it leaves standing; without it the
        final step accumulates a position for free and the frictional recursion ends in a
        discontinuity. Read at `T_dec`, the pre-settlement row — the same row the benchmark
        tracks charge their own unwind at; `rows` as `HedgeActionSpace.kappa`."""
        return (self.aspace.kappa(self.tradables_sim, self.T_dec, rows, matched=False)[0]
                if self.position_state and self.force_flat and self.nxt[t] >= self.T_dec else None)

    def _ranking_kappa(self, t, execution=True):
        """The turnover rates `(κ, κ_cal)` a decision at t ranks its candidates with, at the
        cross-path mean mark, or None for no turnover charge: under `DiffV2_Repositioning_Charge`
        State always, the charge being part of the fitted value, and under Argmax in execution (the
        verdict and the stepper roll) alone."""
        charged = self.argmax_charged if execution else self.position_state
        return self.aspace.kappa(self.tradables_sim, t) if charged else None

    def _charge(self, t, acts, q_prev, kappa, rows=None):
        """The repositioning charge of the books `acts` (broadcastable `(..., n_hedge)`) from the
        standing `q_prev` at decision t, or None where none applies: the turnover toll at the rates
        `kappa` (`HedgeActionSpace.kappa`'s pair; None: none), the move's matched half at the
        calendar rate under `Calendar_Spread_Bps`, the quadratic churn term (`DiffV2_Churn_Lambda`), and the outright unwind
        of `acts` `_unwind_kappa` says t owes; with no standing book (`q_prev` None) the unwind
        alone. `rows` prices each path at its own mark, as `HedgeActionSpace.kappa`.

        ONE assembly, called from every seam: the wealth that RANKS a candidate in `_decide`, the
        wealth the INCUMBENT holds against it, the chosen action's regressed TARGET in `_fit_step`,
        the k-step correction, the verdict's charged wealth and the decision dump all price the
        same friction. That equality is the whole of the frictional Bellman. Callers pass `acts`
        live-masked and `q_prev` as the step that set it left it: a leg dying at the step pays its
        close, one dead before it no phantom toll."""
        kappa_T = self._unwind_kappa(t, rows)
        if kappa_T is None and (q_prev is None or (kappa is None and not self.churn_lambda > 0.0)):
            return None
        dq = acts - (acts if q_prev is None else q_prev)
        charge = self.churn_lambda * dq.pow(2).sum(-1)
        if kappa is not None:
            charge = charge + turnover_charge(dq, *kappa)
        if kappa_T is not None:
            charge = charge + turnover_charge(acts, kappa_T)
        return charge

    # ---- one-step wealth move ------------------------------------------------
    @property
    def w_floor(self):
        """The LogWealth ratio's floor, a hair above zero in the frame's money: 1e-4 of its utility
        scale. A launch at or under it is ruin, and the reward there is capped at zero."""
        return 1.0e-4 * self.utility_scale

    def _u_step(self, W1, W0, t, capital=None):
        """The per-step reward both step-sum objectives share. Running_Wealth: u((W1−W0)/c_t),
        the scaled increment. LogWealth: log(W1/W0) — the day's growth ratio, zero iff the step
        is perfectly hedged, scale-free (no c_t anywhere). Below `eps` the log continues
        linearly so a breached path keeps gradient while paying an unbounded-in-level penalty,
        and a step LAUNCHED at or under the floor can only score <= 0: ruin is absorbing, not a
        rebirth against a clamped denominator. `capital` reads the ratio at another member's
        capital line (the ensemble's `_reward`); None is this run's own."""
        if not self.log_ratio:
            return _utility_wrap_signed(W1 - W0, self.runtime, t)
        # the CAPITAL line: wealth in the ratio is capital + MTM, so "wealth 1" is one unit of
        # backing capital and ruin is losing it. Both legs sit on the SAME step's c, so the inner
        # fork is measured against its own launch.
        c = self.capital if capital is None else capital
        eps = 1.0e-3
        base = (c + W0).clamp_min(self.w_floor)
        r = (c + W1) / base
        u = torch.where(r > eps, r.clamp_min(eps).log(), math.log(eps) + (r - eps) / eps)
        # BOUNDED CONDITIONING: the log's economic content lives within a few nats, and beyond
        # ruin there are only magnitudes that wreck the fit. -20 nats is ruin by e^20, so the
        # cliff stays a cliff and the labels stay a distribution.
        u = u.clamp_min(-20.0)
        return torch.where((c + W0) > self.w_floor, u, u.clamp_max(0.0))

    def _reward(self, W1, W0, t):
        """The day's reward in the frame the value it is added to was fitted in: `_u_step`, or over
        an ensemble under LogWealth the mean of each member's growth at its OWN capital line, so a
        two-member roll ranks the same book whichever checkpoint is listed first."""
        ensemble = getattr(self, "_ensemble", None)
        if not ensemble or not self.log_ratio:
            return self._u_step(W1, W0, t)
        return sum(self._u_step(W1, W0, t, m[7]) for m in ensemble) / len(ensemble)


    def _set_displacement(self, W_bank):
        """The log's bias, MEASURED FROM THE FORWARD PASS: the worst book any simulated bank path
        touches, plus a 10% buffer. Wealth stays free to go negative, the shifted log is defined on
        everything training will see, and the -20-nat backstop guards only the rare inner-fork dip
        beyond the buffer.

        THE HEADROOM GATE: the capital line is `c = displacement/aversion`, so the worst path sits
        at headroom `(c + worst)/c = 1 - aversion/1.1`. Aversion rescales the very buffer the
        displacement provides — at 1.1 the buffer is gone and every worse outcome lands on the
        saturated tail — so a domain break refuses and thin headroom warns."""
        worst = self._bank_worst
        logging.debug("LogWealth displacement measured off the warmup bank: worst %.6g", worst)
        self.displacement = 1.1 * max(0.0, -worst)
        self.capital = c = self.displacement / self.aversion
        headroom = 1.0 if worst >= 0 else (c + worst) / c
        if headroom <= 0.0:
            raise ValueError(
                f"LogWealth domain break: DiffV2_Risk_Aversion={self.aversion:g} leaves the "
                f"worst bank book ({worst:.4g}) outside the shifted log's domain "
                f"(headroom {headroom:.1%}). The displacement buffers 10% at aversion 1.0 "
                f"and aversion divides it — use aversion < 1.1, or raise the displacement.")
        log = logging.warning if headroom < 0.15 else logging.info
        log("DiffSolver LogWealth displacement: worst bank book %.4g -> bias %.4g "
            "(aversion %.2f divides it; headroom %.1f%%)",
            worst, self.displacement, self.aversion, 100.0 * headroom)

    def _outer_move(self, t, rows=slice(None), to=None, money=False):
        """The OUTER-realized move from row t to the next decision row (`self.nxt`), or the row
        `to`: the hedges' `Bundle.move` `(..., n_hedge)` in hedge order and the liability's
        `L_to − L_t` `(...)`, differenced in `MONEY` where `money` (the verdict's wealth law). The
        one spelling behind the bank rolls, the labels' k-step side, the verdict, the stepper
        rollout's drift monitor, the textbook hedge and the constructed policy."""
        to = self.nxt[t] if to is None else to
        L = self.liability_sim
        dL = (L[to].to(MONEY) - L[t].to(MONEY) if money else L[to] - L[t])[rows]
        return self.bundle.move(t, to, self.aspace.hedges, rows), dL

    def _p(self, book, live=None):
        """The position state `p = Σq / Q_max` of `book` `(..., n_hedge)` on the legs `live` marks
        (every leg where None), measured in the `Total_Position_Abs_Limit` cap."""
        return (book if live is None else book * live).sum(-1) / self.aspace.total_abs_limit

    def _risk_adjusted(self, C1f, Ckf=None):
        """`E_inner[C]` over the one-step draws `C1f`, downside-adjusted when `DiffV2_Risk_Kappa` is
        armed: the mean less kappa times the downside semideviation over the inner draws of `Ckf`,
        the score at the decision's `DiffV2_Risk_Horizon` landing row, or of `C1f` where that row is
        the successor. The one scoring rule of `_decide`'s chunk loop and `_score_actions`."""
        C1 = C1f.mean(-1)
        if self.risk_kappa > 0.0:
            Ck, mk = (C1f, C1) if Ckf is None else (Ckf, Ckf.mean(-1))
            dev = (Ck - mk.unsqueeze(-1)).clamp(max=0.0)
            C1 = C1 - self.risk_kappa * (dev ** 2).mean(-1).sqrt()
        return C1

    def _landed(self, nets, W, book, fork, charge, s):
        """Each inner draw's score of holding `book` `(1|B, c, 1, n_hedge)` from the wealth `W`
        `(B,)` to row `s`, less `charge` `(B, c, 1)` (None for none). `fork`, `(dF, dL, market,
        live)`, carries the moves from t to `s`, or under `Running_Wealth` to each decision row up to
        it, `(m, B, Bi, n_hedge)` and `(m, B, Bi)` (a one-step fork's without the row axis), the
        market state at `s` and the leg mask the book fraction `p` the book leaves standing there is
        read under. The score is `C_s` at the wealth landed on, plus under `Running_Wealth` each
        row's reward, the expression the label regresses on. Returns the scores and the wealths
        landed on, each `(B, c, Bi)`."""
        dF, dL, market, live = fork
        path = wealth_step(W[:, None, None], book, self.aspace.contract_size,
                           dF.reshape((-1,) + dF.shape[-3:])[:, :, None],
                           dL.reshape((-1,) + dL.shape[-2:])[:, :, None])                 # (m,B,c,Bi)
        if charge is not None:
            path = path - charge
        W1 = path[-1]
        B, c, Bi = W1.shape
        # the terminal's continuation reads neither the market nor the book
        x = p = None
        if s < self.T_dec:
            x = market[:, None].expand(B, c, Bi, market.shape[-1]).reshape(-1, market.shape[-1])
            if self.position_state:
                p = self._p(book, live).expand(B, c, Bi).reshape(-1)
        C = self._continuation(nets, x, W1.reshape(-1), s, p).reshape(B, c, Bi)
        if self.running_wealth:
            # each move's reward off the CHARGED wealth, so a move pays its toll inside the step that
            # made it, all at s's knee: a running reward's is flat, `conditional_sim` refused beside it
            # and LogWealth's ratio reading none
            C = C + self._reward(path, torch.cat([W[None, :, None, None].expand(1, B, c, Bi),
                                                  path[:-1]]), s).sum(0)
        return C, W1

    # ---- inner-MC one-step quantities at outer t -----------------------------
    def _inner_step(self, t, landing=None):
        """Fork inner MC at t (resolve_structure / resolve_hedge_structure under the hood)
        and return the move tensors the bootstrap needs over decision t's interval, t+1 the row
        `landing`, by default the next decision row, or for a list `landing` its moves to each row
        on a leading axis and the market state at the last:
          dF   (B_outer, B_inner, n_hedge)  per-instrument futures move t→t+1
          dL   (B_outer, B_inner)           liability mark change t→t+1
          m1   (B_outer, B_inner, market_dim) market state at t+1
        plus the bank-state market at t (B_outer, market_dim).

        EXPIRED-CONTRACT GUARD (the returned `live` mask): the framework returns inner `F_t1 = 0`
        for a tradable that expired before the fork, while the OUTER `tradables_sim` FREEZES at the
        last traded price, so a naive `F_t1 − F_t` mints a spurious ~−F_t move a short would mine
        fake P&L from. A dead contract's move is forced to 0; `live_i` says the contract still
        trades at t (`Bundle.marked`)."""
        inner = self.bundle.inner_mc(t, self.nxt[t] if landing is None else landing)
        F_t = torch.stack([self.tradables_sim[ref][t] for ref in self.aspace.hedges], dim=-1)  # (B_outer, n_hedge)
        F_t1 = torch.stack([inner["F_t1"][ref] for ref in self.aspace.hedges], dim=-1)  # ([m,] B_outer, B_inner, n_hedge)
        live = self.bundle.marked(t, self.aspace.hedges)                                      # (n_hedge,)
        dF = (F_t1 - F_t.unsqueeze(1)) * live
        dL = inner["L_t1"] - inner["L_t"]                                                # (B_outer, B_inner)
        return dF, dL, inner["market_t1"], inner["market_t"], live

    def _ahead(self, t):
        """The risk term's fork at decision t to its landing row `s = self.landing[t]`, `(dF, dL,
        market, live)`: the moves from t to `s`, or under `Running_Wealth`, whose reward sums over
        them, to each decision row up to it, `(m, B_outer, B_inner, n_hedge)` and
        `(m, B_outer, B_inner)`, and at `s` the inner market state and the leg mask of the decision
        before it, the step that set the book standing there (`_fit_step`'s one-mask rule), both None
        at the terminal, which reads neither. None where `s` is t's successor, whose fork is the
        one-step one."""
        s = self.landing[t]
        if s == self.nxt[t]:
            return None
        dF, dL, market, _, _ = self._inner_step(t, self._held(t))
        if s >= self.T_dec:
            return dF, dL, None, None
        return dF, dL, market, self.inner_cache[self.prev[s]][4]

    def _held(self, t):
        """The decision rows decision t's risk fork reads its marks on: every one to its landing
        under `Running_Wealth`, whose reward sums over them, else the landing alone."""
        rows = [r for r in self.steps + [self.T_dec] if t < r <= self.landing[t]]
        return rows if self.running_wealth else rows[-1:]

    def _residual_z(self, t, dF, obs):
        """`(B,)` the drift residual at decision t on each path: the fork's mean move `dF`
        `(B, B_inner, n_hedge)` less the realised one `obs` `(B, n_hedge)`, over the fork's own
        dispersion, averaged over the legs that move over the step (`Bundle.moving`). The verdict's
        CUSUM and the roll's monitor score its path mean; the null is the per-path RMS."""
        m = self.bundle.moving(t, self.aspace.hedges)
        z = (((dF.mean(1) - obs) / dF.std(1).clamp_min(1e-9)) * m).sum(-1) / (float(m.sum()) or 1.0)
        logging.debug('residual z at step %d: path mean %s', t, z.mean())
        return z

    # ---- external argmax (Bellman max outside the fitted value) --------------
    def _decide(self, nets, market_t1, dF, dL, W, t, q_prev=None, kappa=None, *, live,
                rate_limit=True, ahead=None):
        """Pick the grid action maximising `E_inner[C_{t+1}]` per outer path. No grad. Under
        `Objective.Reference_Mode='Running_Wealth'` the score is `E_inner[u(W1 − W_t) + C_{t+1}]`
        instead — the same expression the LABEL regresses on in `_fit_step`, so the ranking and the
        labels cannot score different things.

        `ahead`, the risk term's fork on the same outer paths (`_ahead`), moves the semideviation
        `DiffV2_Risk_Kappa` penalises to the score at t's `DiffV2_Risk_Horizon` landing row
        (`self.landing`), the candidate held to it and charged as its one-step wealth is; None reads
        it off the one-step draws.

        The action universe is `aspace.grid_at(t, live)`: the books of the base grid, or of the
        FACTORED Q-ladder under `Allocation_Weights`, that the step admits on its live legs
        (`HedgeActionSpace.admissible`: each live active leg inside its box, every other leg at
        zero, the net inside the cap and the corridor), so the bounds hold the REALIZED book rather
        than a target the expiry mask then guts.

        `q_prev` carries the standing book and arms up to four decision-time corrections, each with
        its own switch: the cost-aware turnover charge (`kappa` present, priced net/paired when
        `Calendar_Spread_Bps` gives a calendar rate), the quadratic churn charge, the
        `Max_Trade_Per_Step` band (`rate_limit` True, so the label path keeps the band
        execution-only) and the `Decision_Deadband_Sigma` incumbent. The band takes the best action
        INSIDE the reachable set and falls back to the unrestricted best when nothing is reachable.
        Under `Allocation_Weights` it measures `|ΔΣq|` rather than per-leg `|Δq_i|`, because a
        composition jump at a knot is the schedule's mandate rather than a trade the argmax chose.
        Without the position state these shape SELECTION only and the fitted targets stay
        cost-free; with it, the same charge plus the terminal unwind enters the label's target and
        the successor is queried at `p' = Sum(a·live)/Q_max`, the book each candidate leaves.

        EXPIRY: every position this method holds, states and RETURNS is LIVE-MASKED — candidate,
        incumbent, band and answer alike — since a dead leg cannot be traded and earns nothing. The
        charge alone moves from `q_prev` as the step that set it left it, so a position on a leg
        dying at t pays its close, as the roll's accounting charges it.

        `Decision_Deadband_Sigma` makes the standing book the INCUMBENT candidate: the winner only
        trades when it beats holding by that many standard errors of the difference PAIRED across
        the common inner draws. Holding an inadmissible book is not on offer — a path whose live
        `q_prev` the step does not admit (`HedgeActionSpace.admissible`) moves regardless, though a
        split off the step's rungs it admits is held — and the incumbent runs through the SAME
        `_charge` as every rival, so the paired difference subtracts two readings of one
        function. The move test compares RISK-NEUTRAL means even when `DiffV2_Risk_Kappa` is on: a
        paired standard error is a statement about E[C], and a semideviation is not a mean."""
        grid = self.aspace.grid_at(t, live)
        limit = rate_limit and q_prev is not None and self.aspace.max_trade > 0.0
        dead = rate_limit and q_prev is not None and self.aspace.deadband_sigma > 0.0
        with torch.no_grad():
            B, Bi, _ = market_t1.shape
            if dead and Bi < 2:
                raise ValueError(
                    "Decision_Deadband_Sigma needs Inner_Sub_Batch >= 2: the hold test is a "
                    "PAIRED standard error over the inner draws, so one draw makes it NaN, every "
                    "comparison False, and the book would silently never trade again")
            rows = torch.arange(B, device=market_t1.device)
            # the standing book on the live legs, which the band and the incumbent hold
            q_live = None if q_prev is None else q_prev * live
            best_val = None
            best_q = None
            best_f = None
            band_val = None
            band_q = None
            band_f = None
            for s in range(0, grid.shape[0], self.chunk):
                acts = grid[s:s + self.chunk]                                            # (c, n_hedge)
                a_live = acts * live                                                      # (c,n_hedge)
                # the terminal unwind is a property of the ACTION, so a step with no standing
                # book must not liquidate for free
                charge = self._charge(t, a_live[None], None if q_prev is None else q_prev[:, None, :],
                                      kappa)
                charge = None if charge is None else charge[:, :, None]                   # (B,c,1) $
                C1f, _ = self._landed(nets, W, a_live[None, :, None, :], (dF, dL, market_t1, live),
                                      charge, self.nxt[t])
                Ckf = None if ahead is None else self._landed(
                    nets, W, a_live[None, :, None, :], ahead, charge, self.landing[t])[0]
                C1 = self._risk_adjusted(C1f, Ckf)                                        # E_inner[C] (B,c)
                if limit:
                    infeas = ~self.aspace.in_band(a_live[None, :, :] - q_live[:, None, :])  # (B,c)
                    bv, ba = C1.masked_fill(infeas, float('-inf')).max(dim=1)
                    bq = a_live[ba]                        # index 0 on an all-infeasible chunk;
                    if band_val is None:                   # discarded by the isfinite mask below
                        band_val, band_q, band_f = bv, bq, C1f[rows, ba]
                    else:
                        upd = bv > band_val
                        band_val = torch.where(upd, bv, band_val)
                        band_q = torch.where(upd.unsqueeze(-1), bq, band_q)
                        band_f = torch.where(upd.unsqueeze(-1), C1f[rows, ba], band_f)
                cval, carg = C1.max(dim=1)
                cact = a_live[carg]                                                      # (B,n_hedge)
                if best_val is None:
                    best_val, best_q, best_f = cval, cact, C1f[rows, carg]
                else:
                    upd = cval > best_val
                    best_val = torch.where(upd, cval, best_val)
                    best_q = torch.where(upd.unsqueeze(-1), cact, best_q)
                    best_f = torch.where(upd.unsqueeze(-1), C1f[rows, carg], best_f)
            if limit:
                ok = torch.isfinite(band_val)
                best_q = torch.where(ok.unsqueeze(-1), band_q, best_q)
                best_f = torch.where(ok.unsqueeze(-1), band_f, best_f)
            if dead:
                # the incumbent is a CANDIDATE priced by the same rules: dq = 0 on the live legs
                # kills the L1 and churn terms there, leaving a dying leg's close and the unwind
                charge = self._charge(t, q_live, q_prev, kappa)
                charge = None if charge is None else charge[:, None, None]
                C1f_inc = self._landed(nets, W, q_live[:, None, None, :], (dF, dL, market_t1, live),
                                       charge, self.nxt[t])[0][:, 0]                      # (B,Bi)
                diff = best_f - C1f_inc                                                   # (B,Bi) paired
                se = diff.std(-1) / math.sqrt(Bi)
                move = diff.mean(-1) > self.aspace.deadband_sigma * se
                # holding is only on offer while the step admits the LIVE book
                held = self.aspace.admissible(q_live, t, live)
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    logging.debug('deadband at step %d admits %d of %d standing books', t,
                                  int(held.sum()), held.shape[0])
                move = move | ~held
                best_q = torch.where(move.unsqueeze(-1), best_q, q_live)
            return best_q

    # ---- operating-region bank ----------------------------------------------
    def _replication_hedge(self, t):
        """Per-instrument diagonal min-var hedge from the OUTER moves to t's successor:
        q_i = −Cov(dL, dF_i)/(cs_i·Var(dF_i)) / n_active, both moments the outer paths' population,
        clamped to [lo,hi]. Inactive → 0."""
        moves, dL = self._outer_move(t)                                                  # (B_outer, n_h), (B_outer,)
        q = torch.zeros(self.aspace.n_hedge, device=self.device)
        for i in self.aspace.active:
            dF = moves[:, i]
            var = dF.var(unbiased=False)
            # skip degenerate OR non-finite instruments: an expired-contract row can carry a NaN
            # mark, and clamp(NaN) = NaN would poison the whole textbook book
            if not float(var) > 1e-12 or not torch.isfinite(dF).all():
                continue
            beta = ((dL - dL.mean()) * (dF - dF.mean())).mean() / var
            q[i] = (-beta / (self.aspace.contract_size[i] * max(self.aspace.n_active, 1)))
        logging.debug('replication hedge at step %d: %s contracts', t, q)
        return q.clamp(self.aspace.q_lo, self.aspace.q_hi)

    @staticmethod
    def _dust(rep):
        """Whether a replication book `(n_hedge,)` nets under half a contract: its liability fixed
        to within dust, a policy built on its net cover holds nothing."""
        return abs(float(rep.sum())) < 0.5

    def _textbook_hedge(self, t, live, batch):
        """The textbook book at decision step t `(batch, n_hedge)`: the replication hedge booked as
        the execution books it (`HedgeActionSpace.book`), as the stepper holds it."""
        q = self._replication_hedge(t)[None].expand(batch, self.aspace.n_hedge)
        return self.aspace.book(q, t, live)

    def _passive_hedge(self, t, live, batch):
        """The passive book at decision step t `(batch, n_hedge)`: minus the liability quantity
        still to fix after t (`Bundle.unfixed_at`), split across the live legs by `_weights_at`
        and booked as the execution books it (`HedgeActionSpace.book`), as the stepper holds it."""
        q = (-self.bundle.unfixed_at(self.bundle.initial_time_index + t)
             * self.aspace._weights_at(t, live) / self.aspace.contract_size)
        return self.aspace.book(q[None].expand(batch, self.aspace.n_hedge), t, live)

    def _first_book(self, t, live, batch):
        """The standing book `(batch, n_hedge)` the first greedy decision, at decision step t, is
        measured from by the trade band and the position state (`DiffV2_First_Decision_From`): the
        passive book there, or the opening book. Every track's accounting enters from the opening
        book either way."""
        return (self._passive_hedge(t, live, batch) if self.first_from_passive
                else initial_q_from_runtime(self.runtime, batch, self.device))

    def _build_bank(self, gen):
        """Operating-region bank — roll the OUTER paths forward (cheap; outer-realised
        moves only, no inner MC): hold q = clamp(q_rep_t + noise), its net water-filled into the
        `bounds(t)` band, each step, accumulate W = cum hedge P&L + L_t. Returns per-t lists W_t (B_outer,) and q_prev (B_outer,
        n_hedge). The bank-state `market_t` is read lazily from the SAME inner-MC fork the
        backward sweep makes at t (no extra inner-MC passes).

        Under the position state the bank also has to COVER THE p AXIS: p is an input the
        argmax queries over the whole of `aspace.bounds(t)` while the roll only wanders
        `Bank_Noise_Frac` around the replication hedge, which would leave the learned hysteresis
        slope extrapolating where it matters. Each path therefore draws a UNIFORM net target across
        the feasible band and water-fills onto it, so the per-leg shape stays the noisy replication
        hedge, the total spans what the argmax can ask for, and p_bank is inside [-1, 1]."""
        rng = (self.aspace.q_hi - self.aspace.q_lo)
        mask = self.aspace.active_mask

        def explore(t):
            lo_t, hi_t = self.aspace.bounds(t)[2:]                                        # its net band at t
            q_rep = self._replication_hedge(t)                                            # (n_hedge,)
            noise = self.noise_frac * rng * torch.randn(
                self.B_outer, self.aspace.n_hedge, generator=gen, device=self.device)
            q = (q_rep[None] + noise * mask).clamp(self.aspace.q_lo, self.aspace.q_hi)
            if self.position_state:
                # cover the p axis the argmax queries
                u = lo_t + (hi_t - lo_t) * torch.rand(
                    self.B_outer, 1, generator=gen, device=self.device)
                q = self.aspace.waterfill(q, u, u)
            # the bank's books inside the bounds every track's are held to, so the value fn trains
            # on reachable states only
            return self.aspace.waterfill(q, lo_t, hi_t)
        return self._roll_bank(explore)

    def _roll_bank(self, policy):
        """Roll the outer paths from the opening book under `policy(t)`, the book `(B, n_hedge)` held
        from decision row t to the next, a row of no market time holding the book and wealth of the
        decision before it. Returns per row `W_t` `(B,)` and the standing book `q_prev`
        `(B, n_hedge)` (under the position state, the sampled one), and under
        LogWealth sets `_bank_worst`, the worst wealth the bank reaches, its terminal included: in a
        crash month the terminal is often the minimum, and a bias measured only at decisions would
        leave the last fit step's states uncovered."""
        W = self.liability_sim[0].clone()                                                 # cum_pnl(0) + L_0
        q_prev = initial_q_from_runtime(self.runtime, self.B_outer, self.device)
        W_list, q_list = [], []
        for t in range(self.T_dec):
            if t not in self.steps:
                W_list.append(W_list[-1])
                q_list.append(q_list[-1])
                continue
            W_list.append(W.clone())
            q_list.append(q_prev.clone())
            q_prev = policy(t)
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                net = q_prev[:, self.aspace.active].sum(-1)
                logging.debug('BANK %d %r', t, (float(net.min()), float(net.max())))
            W = wealth_step(W, q_prev, self.aspace.contract_size, *self._outer_move(t))
        if self.log_ratio:
            self._bank_worst = float(torch.minimum(torch.stack(W_list).min(), W.min()))
        return W_list, q_list

    # ---- project per-process state-at-t leaf grads → market_t columns --------
    @staticmethod
    def _state_block(leaves, key, suffixes, rows, n):
        """A factor's state leaves, or their slopes, `(key, suffix)` stacked in column order as
        `(columns, n)` over the outer paths in `rows`; None where a suffix has none."""
        parts = [leaves.get((key, s)) for s in suffixes]
        if not parts or any(p is None for p in parts):
            return None
        return torch.cat([p[..., rows].reshape(-1, n) for p in parts])

    @classmethod
    def _leaf_columns(cls, leaves, widths, rows, n):
        """The market_t columns each factor's state-at-t `leaves` (or their slopes) fill: per factor
        of `widths` (`(key, width, state_suffixes)` in market-column order, the suffixes the
        process's DIFFERENTIABLE state coordinates) `(key, segments)`, each segment `(kind,
        columns, (n, w) block)` over the outer paths in `rows`. Layout: state columns first in
        declared order, the trailing price columns last; a factor with no state suffix maps 1:1
        raw to privileged or leaves the non-price state unmeasured, and a factor missing a state
        leaf is unmeasured whole (no segment). The one walk behind the label projection and its
        self-check."""
        col = 0
        for key, width, suffixes in widths:
            if width <= 0:                                      # privileged-empty factor (not in market_t)
                continue
            gr = leaves.get(key)
            kr = (gr[..., rows].numel() // n) if gr is not None else 0         # raw-leaf column count (price)
            raw = gr[..., rows].reshape(kr, n).transpose(0, 1) if kr else None   # the raw leaf's block
            price = slice(col + width - kr, col + width)
            segments = []
            if suffixes:
                gb = cls._state_block(leaves, key, suffixes, rows, n)
                if gb is not None and gb.shape[0] == width - kr:
                    segments.append(('state', slice(col, col + width - kr), gb.transpose(0, 1)))
                    if kr >= 1:
                        segments.append(('price', price, raw))
            elif kr == width:                                                 # 1:1 raw → privileged (curve)
                segments.append(('1:1', price, raw))
            elif kr >= 1:                                         # unmeasured state + price (raw last)
                segments.append(('price', price, raw))
            yield key, segments
            col += width

    def _project_leaf_grads(self, leaf_grads, widths, rows, n, md):
        """Map `∂Y/∂(state_t leaf)` for each simulated factor into the `(n, md)` gradient w.r.t.
        the privileged market_t columns the value net consumes (`_leaf_columns`), with the `(md,)`
        mask of the columns a leaf measured. An unmeasured column's gradient is 0, and the mask
        leaves it to the value term."""
        g = torch.zeros(n, md, device=self.device)
        measured = torch.zeros(md, dtype=torch.bool, device=self.device)
        for _, segments in self._leaf_columns(leaf_grads, widths, rows, n):
            for _, cols, block in segments:
                g[:, cols] = block
                measured[cols] = True
        return g, measured

    # ---- one backward step: bootstrap + advantage twin fit -------------------
    def _fit_step(self, nets, W_bank, t, inner, q_bank, rows=slice(None), live_prev=None):
        """One backward step at t: bootstrap the value target off the cached inner draws, then fit
        `nets[t]` to it through `_fit_from_labels`.

        The GRAD inner-MC fork returns AAD-live one-step `F_t1`/`L_t1`/`market_t1` plus the
        per-process state-at-t LEAVES, so the bootstrap value Y and its pathwise gradients w.r.t.
        W0 and the market state come from the SAME forward — the full Huge–Savine twin loss.
        `∂Y/∂market_t` is the differential constraint regularizing the market dimension, where a
        value-only fit overfits the few outer paths.

        Under `Objective.Reference_Mode='Running_Wealth'` the target gains the step's own reward,
        `Y = E_inner[u(W1 − W_t) + C_{t+1}]` with `C_{T_dec} ≡ 0`, so the last decision's label is
        exactly `u(ΔW_T)` — the same expression `_decide` ranks candidates by.

        Under the position state the step is FRICTIONAL: the bank's own standing book supplies
        the position state p at t, the selection is charged its repositioning cost, and the SAME
        charge is subtracted from the wealth entering the target, whose successor is read at the
        book the chosen action leaves standing. The toll then compounds down the recursion instead
        of being re-paid and forgotten each day, which is what distinguishes this from Argmax's
        execution-time toll.

        THE ONE MASK RULE for p, the coordinate `nets[t]` is both fitted and queried on:

            p_t = Sum(q_standing · live_{t-1}) / Q_max

        — the book masked by the step that SET it, not by the step about to trade it. `_decide` at
        t−1 can only hold its own `live`, so it hands the successor `Sum(a·live_{t-1})/Q_max` and
        the fit has to meet it there; masking with this step's `live` would read `nets[t]` on one
        coordinate and write it on another, disagreeing by the full width of the axis on exactly
        the step where a leg rolls off. `live_prev` carries that mask in. The charge moves from the
        same book, so a leg dying at t pays its close.

        Under `DiffV2_Label_Horizon` k > 1 the value label gains `_horizon_correction`, k clipped at
        the decision horizon; the gradient labels stay the one-step cross-fitted ones."""
        dF_ng, dL_ng, m1_ng, market0, live, ahead = inner                                # no-grad cache
        market0 = market0[rows]
        W0_bank = W_bank[t][rows]
        # the standing book masked by the step that set it: the position state and the book the
        # charge moves from. With churn_lambda 0 and no kappa the book is inert.
        q_prev = q_bank[t][rows] * (live if live_prev is None else live_prev)
        kappa_t = self._ranking_kappa(t, execution=False)
        # SELECT the action on the NO-GRAD inner draws; EVALUATE its value and pathwise gradients
        # on a fresh GRAD inner, so independent draws cross-fit away the winner's-curse max bias
        q_star = self._decide(
            nets, m1_ng[rows], dF_ng[rows], dL_ng[rows], W0_bank, t, live=live,
            q_prev=q_prev, kappa=kappa_t, rate_limit=False, ahead=_on_rows(ahead, rows))

        # GRAD inner-MC fork: AAD-live t -> t+1 quantities plus the state-at-t leaves
        ig = self.bundle.inner_mc_grad(t)
        leaves, widths = ig["state_t_leaves"], ig["state_t_leaf_widths"]
        if not getattr(self, "_proj_checked", False):
            self._proj_checked = True                  # one-time self-check of the label projection
            mt = ig["market_t"].detach()                       # detach: numeric self-check only
            errs = [f"{utils.check_tuple_name(key)}[" + (",".join(
                f"{kind}={float((mt[:, cols] - block.detach()).abs().max()):.1g}"
                for kind, cols, block in segments) or "unmeasured") + "]"
                for key, segments in self._leaf_columns(leaves, widths, slice(None), mt.shape[0])]
            logging.info("DiffSolver differential-label projection check (privileged market_t "
                         "cols vs state_t leaves; ≈0 ⇒ ∂Y/∂market_col == ∂Y/∂leaf): %s",
                         " ".join(errs))
        F_t1 = torch.stack([ig["F_t1"][r] for r in self.aspace.hedges], dim=-1) * live          # AAD-live
        # the fork's own t mark moves with the leaves as F_{t+1} does: the outer one would label a
        # column with the slope of F_{t+1} alone rather than of the move
        F_t = torch.stack([ig["F_t"][r] for r in self.aspace.hedges], dim=-1) * live
        dF_g = (F_t1 - F_t)[rows]
        dL_g = (ig["L_t1"] - ig["L_t"])[rows]
        m1_g = ig["market_t1"][rows]
        W0 = W0_bank.clone().requires_grad_(True)
        q = q_star[:, None, :]                                                           # (B,1,n_hedge)
        W1 = wealth_step(W0[:, None], q, self.aspace.contract_size, dF_g, dL_g)           # (B,Bi)
        B, Bi_e, md = m1_g.shape
        p_bank = p1 = None
        extra = []
        if self.position_state:
            p_bank = self._p(q_prev)                                      # state at t: live_{t-1}
            if self.book_label:
                # the standing book a leaf of the fork: by the envelope theorem the label's slope
                # in it is the charge's at the chosen action
                q_prev = q_prev.detach().requires_grad_(True)
                extra = [q_prev]
            # the charge the SELECTION above ranked with, now inside the regressed TARGET, and the
            # successor read at the book the chosen action leaves standing
            W1 = W1 - self._charge(t, q_star, q_prev, kappa_t)[:, None]
            p1 = self._p(q_star)[:, None].expand(B, Bi_e).reshape(-1)
        Y = self._continuation(
            nets, m1_g.reshape(-1, md), W1.reshape(-1), self.nxt[t], p1).reshape(B, Bi_e)  # (B,Bi)
        if self.running_wealth:
            # Y = u(W1 - W_t) + C_{t+1}, the same expression `_decide` ranked with. The increment
            # is level-free, so it contributes 0 to the wealth label and its MARKET slope joins
            # g_market - the reward is part of what the twin loss differentiates.
            Y = Y + self._u_step(W1, W0[:, None], self.nxt[t])
        Y = Y.mean(1)                                                                     # (B,)
        grads = torch.autograd.grad(Y.sum(), [W0] + list(leaves.values()) + extra, allow_unused=True)
        gW = grads[0].detach()
        # a leaf the label does not reach is measured, its slope exactly 0
        leaf_grads = {k: g.detach() if g is not None else torch.zeros_like(v)
                      for (k, v), g in zip(leaves.items(), grads[1:1 + len(leaves)])}
        g_market, measured = self._project_leaf_grads(leaf_grads, widths, rows, B, md)   # ∂Y/∂market_t (B,md)
        g_market = g_market * ig["market_t_label_scale"][rows]          # in the net's coordinates
        # indices, taken once: a boolean mask on the device syncs the host every Adam step
        measured = measured.nonzero().squeeze(1)
        # ∂Y/∂p along the ladder: a move in p lands on the live legs as the step's Allocation_Weights
        # split it (evenly where none is declared), so a leg the ladder never moves carries no weight
        g_p = ((grads[-1].detach() * self.aspace._weights_at(t, live)).sum(-1) * self.aspace.total_abs_limit
               if extra else None)
        Y = Y.detach()
        k = min(self.label_horizon, sum(s >= t for s in self.steps))
        if k > 1:
            Y = Y + self._horizon_correction(nets, W0_bank, q_star, q_prev, kappa_t, t, k, rows)
        return self._fit_from_labels(nets, W0_bank, market0, Y, gW, g_market, t, q_star, p_bank, g_p,
                                     measured)

    def _horizon_correction(self, nets, W0, q, q_prev, kappa, t, k, rows):
        """The `DiffV2_Label_Horizon` correction to step t's value label `(rows,)`, no grad: from the
        chosen book `q` on the outer paths in `rows`, the k-step bootstrap along the greedy policy
        less the one-step bootstrap at t+1. Each step moves the wealth by the outer move, less the
        repositioning charge from the book the step before left under the position state; the
        book at t+j is `_decide` on that step's cached no-grad draws; under `Running_Wealth` the
        rewards of the moves after t join the k-step side; both ends are `_continuation` at the
        outer market state, the policy's wealth and the book it leaves standing."""
        def move(s, W, q, q_prev, kappa):
            W1 = wealth_step(W, q, self.aspace.contract_size, *self._outer_move(s, rows))
            if not self.position_state:
                return W1
            return W1 - self._charge(s, q, q_prev, kappa)

        def bootstrap(s, W, q):
            market = self.inner_cache[s][3][rows] if s < self.T_dec else None
            p = self._p(q) if self.position_state else None
            return self._continuation(nets, market, W, s, p)

        with torch.no_grad():
            W = move(t, W0, q, q_prev, kappa)
            s = self.nxt[t]
            one = bootstrap(s, W, q)
            ret = torch.zeros_like(W)
            for _ in range(k - 1):
                dF, dL, m1, _, live, ahead = self.inner_cache[s]
                kappa = self._ranking_kappa(s, execution=False)
                q_prev = q
                q = self._decide(nets, m1[rows], dF[rows], dL[rows], W, s, q_prev=q_prev,
                                 kappa=kappa, live=live, rate_limit=False,
                                 ahead=_on_rows(ahead, rows))
                W1 = move(s, W, q, q_prev, kappa)
                if self.running_wealth:
                    ret = ret + self._u_step(W1, W, self.nxt[s])
                W, s = W1, self.nxt[s]
            return ret + bootstrap(s, W, q) - one

    def _fit_from_labels(self, nets, W0_bank, market0, Y, gW, g_market, t, q_star, p_bank, g_p,
                         measured):
        """Shared fit tail: advantage decomposition + standardized twin loss on the
        (value, wealth-grad, market-grad) labels of `_fit_step`.

        The twin loss matches gradients in STANDARDIZED space (`g_zn = std·g_raw`). Raw-space
        matching is mis-scaled — `∂A/∂W` ~ 1e-6 in money and `∂A/∂spot` ~ 1e-4, both inert
        against the O(0.1) value term — while standardized both are O(1) and the gradient match
        actually regularizes.

        Each term is normalized by its own label variance: with W ~ $1e6 and utility ~ O(1) the
        standardized W-gradient label is ~600x the value label, so an unnormalized sum would let it
        drown both the value fit and the market gradient, and `lam_g` would not balance
        value against gradient as intended.

        `p_bank` is a control the bank samples rather than a simulated coordinate, so the AAD fork
        measures no slope in it and it is supervised by the value term alone, unless
        `DiffV2_Book_Gradient_Label` hands in `g_p`, the label's slope in the standing book (the
        repositioning charge's), matched beside the others and normalized by its own variance.

        `measured` indexes the market columns the slope term matches, so a column no leaf measures
        is fitted by the value term alone.

        `DiffV2_Wealth_Free_Value` is the same discipline the other way: W is no longer an input,
        so it can no longer be a MEASURED COLUMN and its gradient term leaves the loss with it.
        Matching `∂A/∂wn` against a measured `g_zn_W` while wn is absent would ask the market
        columns to absorb the wealth slope, which is the slope the switch exists to remove."""
        # fit A = C - anchor(W0), the anchor `_continuation` adds back, and its wealth slope; a
        # step-sum objective's anchor is zero, so there is no slope to subtract
        a_val, a_gW = Y - self._anchor(W0_bank, t), gW
        if not self.running_wealth:
            Wb = W0_bank.clone().requires_grad_(True)
            a_gW = gW - torch.autograd.grad(self._anchor(Wb, t).sum(), Wb)[0].detach()
        logging.debug('step %d fits the residual over a mean anchor %s', t, (Y - a_val).mean())

        net = nets[t]
        # the chain: the terminal net anchors and trains the full budget, while every earlier net
        # inherits its fitted successor on its FIRST fit and only corrects it
        ref = nets[self.nxt[t]] if self.nxt[t] < self.T_dec else None
        if ref is not None and self._opts.get(t) is None:
            net.load_state_dict(ref.state_dict())
        prox_ref = (torch.cat([p.detach().reshape(-1) for p in ref.parameters()])
                    if ref is not None and self.prox > 0.0 else None)
        # twin loss in STANDARDIZED space: g_zn = std*g_raw, the wealth column's std carrying the
        # t0 notional the column is divided by under the returns state (`_w_norm`)
        lam_g = self.cfg["diffv2_lambda_grad"]
        g_zn_W = self.w_std * (getattr(self, "state_notional", None) or 1.0) * a_gW          # (B,)
        g_zn_m = self.m_std * g_market                                                   # (B,md); u indep of market
        g_zn_m = g_zn_m.index_select(1, measured)        # a column no leaf measures is not labelled
        # per-term label-variance normalization
        nrm_v = a_val.var() + 1e-8
        nrm_w = g_zn_W.var() + 1e-8
        # per-column lambda_j: each market column normalized by its own label variance, so one
        # fat-tailed column cannot deflate the differential constraint for the rest
        nrm_m = (g_zn_m.var(dim=0, keepdim=True) + 1e-8
                 if self.cfg["diffv2_per_column_grad_norm"]
                 else g_zn_m.var() + 1e-8)
        # one optimizer per t, so a streaming step resumes the same Adam moments on a warm-started
        # net instead of restarting them on every batch
        opt = self._opts.get(t)
        if opt is None:
            opt = self._opts[t] = torch.optim.Adam(
                net.parameters(), lr=self.lr,
                weight_decay=self.cfg["diffv2_weight_decay"])
        prev_loss, iters_run = None, self.fit_iters
        nrm_p = None if g_p is None else g_p.var() + 1e-8
        for it in range(self.fit_iters):
            mn = ((market0 - self.m_mean) / self.m_std).detach().requires_grad_(True)
            wn = self._w_norm(W0_bank).detach().requires_grad_(True)
            cols = [mn] if self.wealth_free else [mn, wn.unsqueeze(-1)]
            pn = p_bank if g_p is None else p_bank.detach().requires_grad_(True)
            a = net(torch.cat(cols
                              + ([] if p_bank is None else [pn.unsqueeze(-1)]), dim=-1))
            # the MEASURED COLUMNS of the twin loss are exactly the input channels the net
            # consumes, so `DiffV2_Wealth_Free_Value` takes W out of both at once
            das = torch.autograd.grad(a.sum(), ([mn] if self.wealth_free else [mn, wn])
                                      + ([] if g_p is None else [pn]), create_graph=True)
            da_m = das[0].index_select(1, measured)
            w_term = 0.0 if self.wealth_free else lam_g * ((das[1] - g_zn_W) ** 2).mean() / nrm_w
            loss = (((a - a_val) ** 2).mean() / nrm_v
                    + w_term
                    + lam_g * (((da_m - g_zn_m) ** 2) / nrm_m).mean())
            if g_p is not None:
                loss = loss + lam_g * ((das[-1] - g_p) ** 2).mean() / nrm_p
            if prox_ref is not None:
                flat = torch.cat([p.reshape(-1) for p in net.parameters()])
                loss = loss + self.prox * (flat - prox_ref).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            # block-wise early stop, non-anchor only: one host sync per 8-iter block, a 2-block
            # floor so a cold optimizer cannot plateau its way out, and a stall test on block ends
            if ref is not None and self.fit_tol > 0.0 and it % 8 == 7:
                cur = float(loss.detach())
                if it >= 15 and prev_loss is not None and abs(prev_loss - cur) <= \
                        self.fit_tol * max(abs(prev_loss), 1e-12):
                    iters_run = it + 1
                    break
                prev_loss = cur

        with torch.no_grad():
            a_fit = net(self._standardize(market0, W0_bank, p_bank))
            val_loss = float(((a_fit - a_val) ** 2).mean())
            # trust region for every future EVALUATION of this net: its fitted-target range plus
            # one range-width of headroom each side
            lo, hi = float(a_val.min()), float(a_val.max())
            if self._bounds_frozen:
                # streaming: the region is part of the LOCKED frame, since re-fitting it per batch
                # would re-open the phantom-extrapolation basin the clamp exists to close
                b0, b1 = self.a_bounds[t]
                self._breaches.append(
                    (t, float(((a_val < b0) | (a_val > b1)).to(torch.float32).mean()), lo, hi))
            else:
                pad = max(hi - lo, 1e-3)
                self.a_bounds[t] = (lo - pad, hi + pad)
        row = {
            "t": t, "val_loss": val_loss, "iters": iters_run,
            "Y_absmean": float(Y.abs().mean()),
            "A_absmean": float(a_fit.abs().mean()),
            "q_star_mean": q_star.mean(0).detach().cpu().tolist(),
            "Y_mean": float(Y.mean()),
        }
        if g_p is not None:
            # the fitted slope in the standing book against its label, over the bank's states
            pn = p_bank.detach().requires_grad_(True)
            (da_p,) = torch.autograd.grad(net(self._standardize(market0, W0_bank, pn)).sum(), pn)
            row.update(book_slope=float(da_p.mean()), book_label=float(g_p.mean()))
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            # each market column's fitted slope against its label over the bank's states, which
            # the state-label gate reads
            mx = market0.detach().requires_grad_(True)
            (da_x,) = torch.autograd.grad(net(self._standardize(mx, W0_bank, p_bank)).sum(), mx)
            logging.debug('STATE_LABEL %d %r', t, {'slope': da_x.mean(0).tolist(),
                                                   'label': g_market.mean(0).tolist()})
        return row

    # ---- greedy-rollout downside verdict -------------------------------------
    def _verdict(self, nets, inner_cache, sweep_ts, rows=slice(None)):
        """Roll the fitted argmax policy forward over [t_min, T_dec] on the OUTER paths in
        `rows` (wealth advanced by the outer-realised dF/dL), every track holding the opening book
        `q0` from the first row to t_min as the stepper roll does, and compare terminal-wealth
        downside against a textbook diagonal-min-var delta hedge, no hedge (flat from the first
        decision), and the passive hedge short the liability's unfixed quantity (`_passive_hedge`).
        The argmax uses the cached inner-MC E[C_{t+1}] estimate; the realised
        outcome uses the outer path. Pass held-out `rows` for an honest OUT-OF-SAMPLE verdict.

        Returns per-policy `{u_mean, wT_mean, wT_p5, wT_cvar5}`. The greedy policy should DOMINATE
        no-hedge on downside and stay competitive with textbook; a SPECULATING policy shows up as
        worse p5/CVaR than textbook with a wide wT spread.

        Under `Objective.Reference_Mode='Running_Wealth'` `u_mean` is the SUM of the rollout's
        per-step utilities, because that is the objective the DP maximised. It is therefore NOT
        comparable to a terminal-utility `u_mean`, nor to `run_textbook_benchmark` /
        `HindsightDpSolver`, which stay terminal-based. `wT_mean` and the tail columns are terminal
        wealth in money under either mode, under the position state net of the repositioning charge,
        its `DiffV2_Churn_Lambda` penalty included; `u_mean_is_step_sum` says which regime produced it.

        Each policy (greedy, textbook, passive and the no-hedge book's close) pays turnover cost on
        the same three-part rule the two benchmark tracks price their execution with: entry from
        the OPENING book `q0`, per-step repositioning, and the terminal unwind under
        `Force_Flat_At_End` (`turnover_cost_mean`); the first greedy decision is measured from
        `_first_book`, as the stepper roll's is. Without the position state the wealth
        recursion is cost-free and the cost is accounted in PARALLEL, so the `*_net` columns are
        diagnostics on that approximation. Under it each step's wealth is charged
        `_charge`, the charge the labels are fitted net of, churn term included: every
        greedy decision reads the net wealth its value was fitted on, the main columns are net, and
        there are no `*_net` columns.

        The greedy ARGMAX is charged under `DiffV2_Repositioning_Charge` Argmax and State: a value
        function fitted against a charged selection must be rolled under the same rule, or the
        policy reported is not the policy trained. `argmax_charged` says which regime produced
        these numbers."""
        L = self.liability_sim
        t0 = sweep_ts[0]
        n = L[t0][rows].shape[0]
        q0 = initial_q_from_runtime(self.runtime, n, self.device)
        # the opening book's P&L from the first row to the first decision, every track's, in MONEY
        W0 = wealth_step(L[t0][rows].to(MONEY), q0, self.aspace.contract_size,
                         self._outer_move(0, rows, t0)[0].to(MONEY), 0.0)
        W = {p: W0.clone() for p in ("greedy", "textbook", "nohedge", "passive")}
        # the turnover cost, entry measured from the opening book: paid inside the wealth under the
        # position state, a diagnostic beside it otherwise
        cost = {p: torch.zeros(n, dtype=MONEY, device=self.device) for p in W}
        q_traj = {p: [] for p in ("greedy", "textbook", "passive")}                 # mean q a decision
        q_prev = {p: q0.clone() for p in cost}
        # running-wealth: the rollout's value is the SUM of the per-step rewards, accumulated along
        # the roll, and the net sum charges each step's own turnover into that step's increment
        vdrift_z, vdrift_s2, vdrift_n = 0.0, 0.0, 0
        u_run = ({p: torch.zeros(n, dtype=MONEY, device=self.device) for p in W}
                 if self.running_wealth else None)
        u_net = ({p: torch.zeros(n, dtype=MONEY, device=self.device) for p in cost}
                 if self.running_wealth and not self.position_state else None)
        last_net_inc = {}          # the final step's net increment, which the unwind belongs to
        last_W_pre = {}            # ...and the wealth it launched from (the ratio needs the base)
        with torch.no_grad():
            for t in sweep_ts:
                dF_o, dL_o = self._outer_move(t, rows, money=True)                 # (n, n_hedge), (n,)
                dF, dL, m1, _, live, ahead = inner_cache[t]
                # in-sim validation CUSUM: the fork's expectation minus the outer realized move,
                # averaged over paths and summed over steps - a mean-zero tripwire within the
                # model, whose deployment twin is the stepper roll's market-drift monitor
                zstep = self._residual_z(t, dF[rows], dF_o)                         # (n,) per path
                vdrift_z += float(zstep.mean())
                vdrift_s2 += float(zstep.pow(2).mean())
                vdrift_n += 1
                logging.debug('verdict wealth at step %d: %.17g', t, float(W["greedy"].mean()))
                q_g = self._decide(
                    nets, m1[rows], dF[rows], dL[rows], W["greedy"].float(), t,
                    q_prev=self._first_book(t, live, n) if t == t0 else q_prev["greedy"],
                    kappa=self._ranking_kappa(t), live=live, ahead=_on_rows(ahead, rows))
                q_tb = self._textbook_hedge(t, live, n)
                q_pa = self._passive_hedge(t, live, n)
                z = torch.zeros(n, self.aspace.n_hedge, device=self.device)
                # each path's charge at its own mark, where the ranking reads the mean
                kappa_p = self.aspace.kappa(self.tradables_sim, t, rows=rows)
                logging.debug('verdict charge kappa at step %d: %s to %s over the paths', t,
                              kappa_p[0].amin(0).tolist(), kappa_p[0].amax(0).tolist())
                step_cost = {}
                for p, q_now in (("greedy", q_g), ("textbook", q_tb), ("nohedge", z),
                                 ("passive", q_pa)):
                    step_cost[p] = (self._charge(t, q_now, q_prev[p], kappa_p, rows)
                                    if self.position_state else
                                    turnover_charge(q_now - q_prev[p], *kappa_p))
                    cost[p] = cost[p] + step_cost[p]
                    q_prev[p] = q_now
                W_pre = dict(W)
                for p, q_now in (("greedy", q_g), ("textbook", q_tb), ("nohedge", z),
                                 ("passive", q_pa)):
                    W[p] = wealth_step(W[p], q_now, self.aspace.contract_size, dF_o.to(MONEY), dL_o)
                    if self.position_state:
                        # the charge inside the step, as the label pays it, so the next decision
                        # reads the wealth its value was fitted on
                        W[p] = W[p] - step_cost[p]
                if u_run is not None:
                    # indexed by the step the increment ARRIVES at, as every other seam does: the
                    # knee an increment is measured under belongs to the day it lands on
                    for p in W:
                        u_run[p] = u_run[p] + self._reward(W[p], W_pre[p], self.nxt[t])
                if u_net is not None:
                    for p in u_net:
                        last_net_inc[p] = W[p] - W_pre[p] - step_cost[p]
                        last_W_pre[p] = W_pre[p]
                        u_net[p] = u_net[p] + self._reward(
                            W_pre[p] + last_net_inc[p], W_pre[p], self.nxt[t])
                for p, q_now in (("greedy", q_g), ("textbook", q_tb), ("passive", q_pa)):
                    q_traj[p].append(q_now.mean(0).tolist())
            if self.force_flat and not self.position_state:
                # the unwind both benchmark tracks charge, so all three price one execution; under
                # the position state the last step's charge carried it
                kappa_T = self.aspace.kappa(self.tradables_sim, self.T_dec, rows, matched=False)[0]
                for p in cost:
                    unwind = turnover_charge(q_prev[p], kappa_T)
                    cost[p] = cost[p] + unwind
                    if u_net is not None:
                        # the liquidation belongs INSIDE the last increment, where training charged
                        # it; u is nonlinear, so appending u(-unwind) as its own term would report
                        # a different objective than the one optimised
                        u_net[p] = (u_net[p]
                                    - self._reward(last_W_pre[p] + last_net_inc[p],
                                                   last_W_pre[p], self.T_dec)
                                    + self._reward(last_W_pre[p] + last_net_inc[p] - unwind,
                                                   last_W_pre[p], self.T_dec))

        out = {p: _tail_stats(self.runtime, W[p], None if u_run is None else u_run[p]) for p in W}
        out["validation_drift_z"] = round(vdrift_z / max(1, vdrift_n) * (vdrift_n ** 0.5), 4)
        # THE NULL: the per-step std of the within-model standardized residual, so the tripwire
        # boundary at inference is k * null_sigma * sqrt(t). Stamped with the frame.
        if self.loaded is None:
            # measured on TRAINING validation only, where the fork and the outer paths share one
            # law. On a LOADED roll the same statistic reads model-vs-market drift, and stamping
            # that as the null would inflate the tripwire's threshold with the signal it detects.
            self.drift_null_sigma = max(1e-6, (vdrift_s2 / max(1, vdrift_n)) ** 0.5)
        out["drift_null_sigma"] = round(self.drift_null_sigma, 4)
        # which regime chose the greedy book, and whether `u_mean` is a terminal utility or a sum
        # of per-step ones - neither is readable off the figures
        out["argmax_charged"] = bool(self.argmax_charged)
        out["u_mean_is_step_sum"] = bool(self.running_wealth)
        for p in cost:
            out[p]["turnover_cost_mean"] = float(cost[p].mean())
            if not self.position_state:
                out[p].update({f"{k}_net": v for k, v in _tail_stats(
                    self.runtime, W[p] - cost[p], None if u_net is None else u_net[p]).items()})
        out.update(self._decision_record(sweep_ts, q_traj))
        # greedy position summary over the rows: each decision's book held to the next (is it hedging?)
        gq = torch.tensor(q_traj["greedy"]).repeat_interleave(
            torch.tensor([self.nxt[t] - t for t in sweep_ts]), dim=0)                  # (rows, n_hedge)
        out["greedy_mean_abs_q"] = gq.abs().mean(0).tolist()
        out["greedy_q_first"] = q_traj["greedy"][0]
        out["greedy_q_mid"] = gq[len(gq) // 2].tolist()
        return out

    def _decision_record(self, ts, books):
        """The per-decision record a roll reports: each policy's mean book (`books[p]`, one entry a
        decision) as `{p}_q_traj`, beside the decisions' rows `greedy_q_t` and dates
        `greedy_q_dates`, which consumers join on."""
        hist = self.bundle.initial_time_index
        out = {f"{p}_q_traj": b for p, b in books.items()}
        out["greedy_q_t"] = [int(t) for t in ts]
        out["greedy_q_dates"] = [str(self.bundle.scenario_dates[hist + int(t)].date()) for t in ts]
        return out

    # ---- decision-curve diagnostic (Solver.DiffV2_Decision_Curve_Dump) --------
    def _score_actions(self, nets, t, acts, market_t1, dF, dL, W, q_prev, kappa, live, ahead):
        """Score `acts` `(K, n_hedge)` the way `_decide`'s chunk loop ranks its grid rows —
        `E_inner[C_{t+1}]` before and after the repositioning charge, its risk term on the fork
        `ahead` — and hand back the CHARGED per-draw `W1` the ranking's continuation was actually
        read at. Returns `(pre, charged, W1)`, `(K,)`/`(K,)`/`(K, Bi)`: the single-path diagnostic
        drops the outer axis.

        A SECOND PASS rather than a probe wired into `_decide`, so a run with the dump on is
        bit-identical to one with it off and `_decide` keeps no knowledge of it — no parameter, no
        branch, no extra continuation in the hot loop. The two rankings could drift apart, which is
        what the gate on the argmax column measures. The pass is affordable because the dump is a
        `Batch_Size = 1` diagnostic."""
        a_live = acts * live
        with torch.no_grad():
            charge = self._charge(t, a_live[None], q_prev[:, None, :], kappa)
            charge = None if charge is None else charge[:, :, None]

            def rank(charge):
                C1f, W1 = self._landed(nets, W, a_live[None, :, None, :], (dF, dL, market_t1, live),
                                       charge, self.nxt[t])
                Ckf = None if ahead is None else self._landed(
                    nets, W, a_live[None, :, None, :], ahead, charge, self.landing[t])[0]
                return self._risk_adjusted(C1f, Ckf)[0], W1

            charged, W1c = rank(charge)
            return (charged if charge is None else rank(None)[0]), charged, W1c[0]

    def _decision_curve_row(self, nets, t, market_t1, dF, dL, W, q_prev, q_chosen, live, kappa,
                            ahead):
        """One CSV row of `Solver.DiffV2_Decision_Curve_Dump`: everything about decision `t` that
        says WHY the ranking chose the book it chose, and nothing that changes it.

        The columns, in four groups.

        *The decision.* `q_prev_net` / `q_chosen_net` are the realized signed covers `Σq·live`,
        `argmax_net` is the dump's own band-feasible charged argmax, and `deadband_held` is 1 only
        when `Decision_Deadband_Sigma` is armed, the executed book IS the incumbent, and the argmax
        was elsewhere — the band, not agreement, stopped the trade.

        *The utility in force*, the RANKING's rather than the ledger's. `W` is the stepper's net
        wealth entering the decision. `c` is the knee every rung is divided by: the scale off
        `runtime['objective']` at `t+1`, because the rungs are continuations there. It is the
        FRAME's, since `_rollout_on_stepper` builds its stepper with `mirror_scale=False` so a
        loaded checkpoint's scale and schedule survive the roll. `x0` places the day on that
        utility at the point the ranking operates — the wealth LEVEL under `Fixed`, the day's
        INCREMENT under `Running_Wealth`, which is zero before the day happens, so the curvature
        is read off the LOSS side of the reference. `in_knee`, `u_prime` and `local_ARA` follow it,
        and `u_is_increment` says which regime produced them.

        *What the ranking saw.* `curve_net` / `curve_pre` / `curve_charged` are the FULL per-rung
        curve, packed space-separated into three cells so a narrowed ladder does not ragged the
        header. `range_CE` converts the charged curve's spread into money at the local `u'`, and
        `bow_span` is the curve's departure from the chord through its net-sorted endpoints — a
        straight ranking (bow ≈ 0) is one whose argmax can only be a corner. `P_band_*` is the
        chance the day ENDS in the same knee band, per inner draw, at three reference books (flat,
        the standing book, the deepest rung), measured on the same quantity `x0` is.

        *What it was ranking over.* The one-step moments of the fork feeding it, plus a SECOND
        independent one-step fork at the same `t`, its risk term on the risk fork `ahead`, which
        the decision's own stream would draw again: `fork_disagree` says whether a rung won on
        signal or on inner-MC noise.

        Conventions. Moments are population (÷N) over the flattened outer × inner draws. `dF_net`
        is the tradable move a UNIT of net cover earns, reduced as `wealth_step` reduces a book —
        `Σ_i w_i·cs_i·dF_i`, with `w` the `Allocation_Weights` split in force or the uniform split
        over the live, active legs."""
        obj = self.runtime["objective"]
        # the knee the RUNGS divide by: `_score_actions` reads every continuation at t's successor,
        # so that is the step whose entry they consume. Under a flat scale the distinction is inert.
        c = _utility_scale(obj, self.nxt[t])
        R = float(obj["reference_wealth"])
        # the knee is huber's; shapes without one take a unit of c as the near-reference band
        delta = float(obj["huber_delta"]) if obj["object"] == "asymmetricutility_huber" else 1.0
        w0 = float(W[0])
        # WHERE the ranking operates: the wealth LEVEL under Fixed, the day's INCREMENT under
        # Running_Wealth. Reporting the level under an increment objective logs a curved ranking
        # as a flat one.
        w_rank = 0.0 if self.running_wealth else w0
        u1, ara = (_loss_side_curvature(self.runtime, self.nxt[t]) if self.running_wealth
                   else _utility_local_curvature(w_rank, self.runtime, self.nxt[t]))
        x0 = _utility_x(w_rank, obj, self.nxt[t])

        q_live = q_prev * live                                                  # (1,n_hedge)
        grid = self.aspace.grid_at(t, live)
        a_live = grid * live                                                      # (K,n_hedge)
        net = a_live.sum(-1)                                                        # (K,)
        pre, charged, _ = self._score_actions(
            nets, t, grid, market_t1, dF, dL, W, q_prev, kappa, live, ahead)
        # the rate-limit band the executed argmax searched inside, measured exactly as `_decide`
        # measures it - net under a weight schedule, per-leg without one - with its fallback
        pick = charged
        if self.aspace.max_trade > 0.0:
            feas = self.aspace.in_band(a_live - q_live)
            pick = charged.masked_fill(~feas, float("-inf")) if bool(feas.any()) else charged
        k_star = int(pick.argmax())
        deep = int(net.abs().argmax())

        # P_band at the three reference books, off the SAME charged W1 the ranking is built on.
        probes = torch.stack([torch.zeros_like(q_live[0]), q_live[0], a_live[deep]])
        _, _, W1p = self._score_actions(
            nets, t, probes, market_t1, dF, dL, W, q_prev, kappa, live, ahead)
        # ...on the same quantity `x0` is: the successor LEVEL, or the day's realized INCREMENT.
        x1 = _utility_x(W1p - (w0 if self.running_wealth else 0.0), obj, self.nxt[t])
        p_band = ((x1 > -delta) & (x1 < 0.0)).to(torch.float32).mean(-1)

        # The curve's shape: spread in money at the local u', and bow off the endpoint chord.
        order = torch.argsort(net, stable=True)
        cs, ns = charged[order], net[order]
        span, width = float(cs[-1] - cs[0]), float(ns[-1] - ns[0])
        chord = (cs[0] + (cs[-1] - cs[0]) * (ns - ns[0]) / width) if width else cs
        # a unit of net cover under the composition the schedule mandates
        w = self.aspace._weights_at(t, live)
        dfn = (dF * (w * self.aspace.contract_size)).reshape(-1, self.aspace.n_hedge).sum(-1)
        dl = dL.reshape(-1)

        # a SECOND one-step fork at the same t: the stream advances, so this is an independent draw
        # of the same conditional law, decided under identical rules
        dF2, dL2, m2, _, live2 = self._inner_step(t)
        q_b = self._decide(nets, m2, dF2, dL2, W, t, q_prev=q_prev, live=live2, kappa=kappa,
                           ahead=ahead)

        row = {
            "t": int(t),
            "q_prev_net": float(q_live.sum()),
            "q_chosen_net": float(q_chosen.sum()),
            "argmax_net": float(net[k_star]),
            "deadband_held": int(self.aspace.deadband_sigma > 0.0
                                 and bool(torch.equal(q_chosen[0], q_live[0]))
                                 and not bool(torch.equal(a_live[k_star], q_live[0]))),
            "W": w0, "c": c, "R": R, "x0": x0,
            # which quantity x0 / in_knee / u_prime / local_ARA / P_band_* are measured on
            "u_is_increment": int(self.running_wealth),
            "in_knee": int(-delta < x0 < 0.0),
            "u_prime": u1, "local_ARA": ara,
            "P_band_flat": float(p_band[0]),
            "P_band_prev": float(p_band[1]),
            "P_band_deep": float(p_band[2]),
            "range_CE": float(charged.max() - charged.min()) / u1,
            "bow_span": (float((cs - chord).abs().max()) / abs(span)) if span else float("nan"),
        }
        for j, name in enumerate(self.aspace.hedges):
            row[f"E_dF_{name}"] = float(dF[..., j].mean())
            row[f"sd_dF_{name}"] = float(dF[..., j].std(unbiased=False))
        row.update({
            "E_dL": float(dl.mean()), "sd_dL": float(dl.std(unbiased=False)),
            "cov_dFnet_dL": float(((dfn - dfn.mean()) * (dl - dl.mean())).mean()),
            "var_dFnet": float(dfn.var(unbiased=False)),
            "argmax_A_net": float(q_chosen.sum()),
            "argmax_B_net": float(q_b.sum()),
            "fork_disagree": int(not bool(torch.equal(q_chosen, q_b))),
            "n_rungs": int(net.shape[0]),
            "curve_net": _packed(net), "curve_pre": _packed(pre),
            "curve_charged": _packed(charged),
        })
        return row

    # ---- frozen-policy daily rollout on a realized path via the stepper -------
    def _rollout_on_stepper(self, nets, inner_cache, sweep_ts):
        """Deployment-faithful backtest: roll the frozen policy day-by-day along the bundle's
        observed path through `BundleStepper`, which owns the real futures accounting. Each day
        the book is chosen by `_decide` from the CAUSAL one-step fork forecast (`inner_cache[t]`)
        and the STEPPER'S OWN net wealth, never a verdict wealth recursion, so the decision cannot
        be contaminated by mis-accrued P&L.

        Returns `{greedy, textbook, nohedge, passive}` terminal-P&L stats in the verdict shape,
        with `u_mean_is_step_sum` stating whether `u_mean` is a terminal utility or the
        running-wealth sum. `Solver.DiffV2_Decision_Curve_Dump` hangs a per-decision CSV off the
        greedy roll; no branch below it depends on whether it is on. From the first decision day
        on or after `Solver.DiffV2_Passive_From` the greedy roll holds the passive book."""
        from .hedge_bundle import BundleStepper, _tracking_error_value
        hist = self.bundle.initial_time_index
        sweep_set = set(int(t) for t in sweep_ts)
        if self.curve_dump and self.B_outer != 1:
            raise ValueError(
                f"Solver.DiffV2_Decision_Curve_Dump is a SINGLE-PATH diagnostic but this roll "
                f"carries {self.B_outer} outer paths. Every column it writes — the wealth entering "
                f"the decision, the ranking curve over it, the band probabilities — belongs to one "
                f"path, and averaging them would describe a decision nothing took. Roll the dump "
                f"at Batch_Size 1.")

        q_log = {"greedy": [], "t": [], "w": []}
        dump_rows = [] if self.curve_dump else None

        def roll(policy):
            """Roll `policy` day-by-day through a fresh `BundleStepper`; returns
            `(terminal P&L, per-step utility sum or None)`.

            `mirror_scale=False`: this rollout is only ever reached with a checkpoint loaded, and
            that checkpoint's utility scale is the value function's own frame — the argmax below
            must read the same `c` the nets were fitted against. Re-mirroring would replace it
            with this world's `c` and decide under a different scale than `_verdict` did.

            Under `Reference_Mode='Running_Wealth'` the reported objective is the SUM of per-step
            rewards, accumulated here too, over the DECISION days the policy acted on plus the
            tail to the terminal."""
            stepper = BundleStepper(self.bundle, self.runtime, mirror_scale=False)
            last = None
            u_sum = w_prev = None
            drift = {"t": [], "cum_dF": [0.0] * self.aspace.n_hedge, "cum_dL": 0.0,
                     "cum_z": 0.0, "z_path": [], "tripped": []}
            pending = None      # the last decision's residual, scored once its move has printed
            # the null of the path mean the monitor scores: the per-path null over the paths' root
            null = self.drift_null_sigma / math.sqrt(self.bundle.batch_size)
            while not stepper.done:
                t = stepper.time_index - hist
                if t in sweep_set:
                    state = stepper._state
                    W = _tracking_error_value(state, self.runtime).to(self.device)     # (B,) net wealth
                    B = W.shape[0]
                    if self.running_wealth:
                        u_sum = (torch.zeros_like(W) if w_prev is None
                                 else u_sum + self._reward(W, w_prev, t))
                        w_prev = W
                    if policy == "nohedge":
                        q = torch.zeros(B, self.aspace.n_hedge, device=self.device)
                    elif policy == "textbook":
                        q = self._textbook_hedge(t, inner_cache[t][4], B)
                    elif policy == "passive":
                        q = self._passive_hedge(t, inner_cache[t][4], B)
                    else:
                        dF, dL, m1, _, live, ahead = inner_cache[t]
                        # MODEL-DRIFT CUSUM: each decision's one-day forecast minus what the market
                        # then did, standardized by the fork's own dispersion, entering at the NEXT
                        # decision so a trip reads only printed moves
                        if pending is not None:
                            drift["cum_dF"] = [a + b for a, b in zip(drift["cum_dF"], pending[0])]
                            drift["cum_dL"] += pending[1]
                            drift["cum_z"] += pending[2]
                        obs_dF, obs_dL = self._outer_move(t)                      # (B, n_h), (B,)
                        rF = (dF.mean(1) - obs_dF) * live                         # (B, n_h)
                        rL = dL.mean(1) - obs_dL                                  # (B,)
                        # the legs that move over the step only (`Bundle.moving`): a frozen mark's
                        # dust over the std clamp fabricates huge sigma readings
                        inf_m = self.bundle.moving(t, self.aspace.hedges)
                        pending = ([float(b) for b in rF.mean(0)], float(rL.mean()),
                                   float(self._residual_z(t, dF, obs_dF).sum() / B))
                        drift["t"].append(int(t))
                        drift["z_path"].append(round(drift["cum_z"], 4))
                        n_so_far = max(1, len(drift["t"]) - 1)         # residuals scored so far
                        boundary = (self.cfg["diffv2_drift_threshold_sigmas"]
                                    * null * (n_so_far ** 0.5))
                        tripped = abs(drift["cum_z"]) > boundary
                        drift["tripped"].append(bool(tripped))
                        beta = self.cfg["diffv2_drift_beta"]
                        if tripped and beta > 0.0:
                            # bias the forecast toward the REALIZED drift: the observed average
                            # residual per leg, scaled by beta, shifts the fork moves the ranking
                            # prices
                            shift = [-beta * c / n_so_far for c in drift["cum_dF"]]
                            logging.debug('drift shift at step %d: %s a contract on the moving legs',
                                          t, shift)
                            adj = torch.tensor(shift, device=self.device)
                            dF = dF + adj * inf_m
                            if ahead is not None:
                                # the risk fork's move to each of its rows by that drift over the
                                # decision intervals to it each leg moves over
                                moved = torch.stack([self.bundle.moving(d, self.aspace.hedges)
                                                     for d in self.steps
                                                     if t <= d < self.landing[t]]).cumsum(0)
                                ahead = (ahead[0] + (adj * moved[-len(ahead[0]):])[:, None, None],
                                         *ahead[1:])
                        kappa_t = self._ranking_kappa(t)
                        if t == sweep_ts[0]:
                            q_prev = self._first_book(t, live, B)
                        day = self.bundle.scenario_dates[hist + int(t)]    # the full grid, past the history
                        handed = self.passive_from is not None and day >= self.passive_from
                        if handed:
                            q = self._passive_hedge(t, live, B)
                        else:
                            logging.debug('stepper wealth at step %d: %.17g', t, float(W.mean()))
                            q = self._decide(nets, m1, dF, dL, W.float(), t, q_prev=q_prev,
                                             live=live, kappa=kappa_t, ahead=ahead)
                        q_log["greedy"].append(q.mean(0).detach().cpu().tolist())
                        q_log["t"].append(int(t))
                        q_log["w"].append(float(W.mean()))
                        if dump_rows is not None and not handed:
                            dump_rows.append(self._decision_curve_row(
                                nets, t, m1, dF, dL, W.float(), q_prev, q, live, kappa_t, ahead))
                    q_prev = q
                    cur = state["positions"]
                    delta = {n: q[:, j] - cur[n].to(dtype=q.dtype, device=q.device)
                             for j, n in enumerate(self.aspace.hedges)}
                    last = stepper.step(delta)
                else:
                    last = stepper.step(None)
            n_dec = max(1, len(drift["t"]) - 1)        # the last decision's residual is never scored
            drift["max_abs_z"] = max((abs(z) for z in drift["z_path"]), default=0.0)
            drift["n_tripped"] = int(sum(drift["tripped"]))
            drift["null_sigma"] = round(null, 4)
            drift["final_z_per_sqrt_n"] = drift["cum_z"] / (n_dec ** 0.5)
            wT = last["transition_net_pnl"]
            if u_sum is not None:
                u_sum = u_sum + self._reward(wT, w_prev, self.T_dec)  # tail
            return wT, u_sum, drift

        rolled = {p: roll(p) for p in ("greedy", "textbook", "nohedge", "passive")}
        out = {p: _tail_stats(self.runtime, wT, u_sum) for p, (wT, u_sum, _) in rolled.items()}
        # one drift monitor per rollout policy; the greedy one is THE inference guard
        out["drift_monitor"] = rolled["greedy"][2]
        out["u_mean_is_step_sum"] = bool(self.running_wealth)
        if dump_rows:
            pd.DataFrame(dump_rows).to_csv(self.curve_dump, index=False, float_format="%.6g")
            logging.info("DiffV2_Decision_Curve_Dump: %d decision rows -> %s",
                         len(dump_rows), self.curve_dump)
        out.update(self._decision_record(q_log["t"], {"greedy": q_log["greedy"]}))
        out["greedy_w_traj"] = q_log["w"]
        return out

    def _pad_steps(self, seq, target):
        """Clamp a per-step list to `target` entries by truncating or repeating the last one —
        identity when the lengths already agree (the provenance check enforces that unless
        `DiffV2_Load_Horizon_Pad` relaxed it)."""
        if seq is None:
            return None
        seq = list(seq)
        return seq[:target] if len(seq) >= target else seq + [seq[-1]] * (target - len(seq))

    # ---- driver: warmup (fit + frame lock) -> step (fresh batch) -> finish (verdict) ----
    def _provenance(self, md, ck):
        """The stamps that make a checkpoint a different function of a different state, `{key:
        (this run's value, what sets it, the reading of the checkpoint `ck` saved before the stamp,
        None where every checkpoint carries it)}`: `_policy_artifact` writes them and
        `_check_load_provenance` refuses a load that differs on one, by name."""
        return {
            "t_min": (self.t_min, "Solver.T_Min", None), "T_dec": (self.T_dec, "the horizon", None),
            "md": (md, "the market columns", None),
            "hedges": (list(self.aspace.hedges), "the hedge legs", None),
            "active_hedge_indices": (list(self.aspace.active), "Solver.Active_Hedge_Indices", None),
            "solver_object": (self.SOLVER_OBJECT, "Solver.Object", "DiffSolver"),
            # the charge the fitted value carries, 'argmax' fitting the cost-free one; a checkpoint
            # stamped before the key carries the `position_state` bool
            "repositioning_charge": ("state" if self.position_state else "no",
                                     "Solver.DiffV2_Repositioning_Charge",
                                     "state" if ck.get("position_state") else "no"),
            "returns_state": (self.cfg["diffv2_returns_state"], "Solver.DiffV2_Returns_State", False),
            "wealth_free": (self.wealth_free, "Solver.DiffV2_Wealth_Free_Value", False),
            "reference_mode": ("running_wealth" if self.running_wealth else "fixed",
                               "Objective.Reference_Mode", "fixed"),
        }

    def _check_load_provenance(self, ck, src, md):
        """Hold one `DiffV2_Load_Value_Fn` member against this run and return its solver version
        (the ensemble collects them). Everything a checkpoint stamps that makes it a DIFFERENT
        FUNCTION OF A DIFFERENT STATE is refused here by name; everything that merely deserves
        saying out loud is warned.

        Refused: every `_provenance` stamp (the shape contract, the legs the fit varied, the solver
        class whose bank shaped the value, `DiffV2_Repositioning_Charge` (State or not), `DiffV2_Returns_State` and
        `DiffV2_Wealth_Free_Value`, each owning input columns or their coordinates, and
        `Reference_Mode`, which moves no
        tensor shape, so a mismatch would have no symptom but a quietly wrong policy); whether a
        `conditional_sim` schedule is in force, the setting against the stamped schedule; a corridor
        mismatch; a stamped `value_optimism` above `DiffV2_Max_Value_Optimism` (a checkpoint saved
        before the stamp loads as before); and, in `_check_action_universe` /
        `_check_calendar_spread`, a narrowed `Allocation_Weights` universe or a
        `Calendar_Spread_Bps` that moved under position state.

        Warned: a stale `solver_version`, a missing corridor/universe/rate stamp under a live one,
        and a pre-stream frame. Corridor-FREE training spans the widest wealth support, so rolling
        it inside any corridor only restricts to a learned subset, and so does a corridor INSIDE the
        trained one; a policy trained inside a corridor is off-support under a wider, a different
        or an absent one."""
        for key, (want, name, unstamped) in self._provenance(md, ck).items():
            got = ck[key] if unstamped is None else ck.get(key, unstamped)
            if got == want:
                continue
            if key in ("t_min", "T_dec") and self.cfg["diffv2_load_horizon_pad"]:
                # horizon pad: per-step structures clamp to the saved range at load, and the
                # tail repeats the last fitted step
                logging.info("DiffV2 load horizon pad: %r saved %r vs this run %r", key, got, want)
                continue
            raise ValueError(
                f"DiffV2_Load_Value_Fn checkpoint mismatch on {key!r}: {src} saved {got!r} vs this "
                f"run {want!r} ({name}), a different function of a different state - retrain, or "
                f"match it.")
        # the run side of this one is the SETTING, never the schedule this world measured: a frozen
        # roll is a single path and measures none, and the checkpoint's governs it
        if bool(ck.get("utility_scale_schedule")) != self.scheduled_scale:
            raise ValueError(
                f"Objective.Utility_Scale_Mode mismatch: {src} was trained "
                f"{'WITH' if ck.get('utility_scale_schedule') else 'WITHOUT'} a conditional_sim "
                f"scale schedule and this run declares "
                f"{'conditional_sim' if self.scheduled_scale else 'a flat scale'}. The knee is "
                f"what the fitted residual's curvature is measured against, so a per-step "
                f"schedule and a flat scalar are different utilities of the same wealth — "
                f"retrain, or match Objective.Utility_Scale_Mode.")
        # the training ARCHITECTURE is part of what a version stamps, and an ensemble must not
        # average continuations trained under different ones
        ck_ver = ck.get("solver_version", "<pre-version checkpoint>")
        if ck_ver != SOLVER_VERSION:
            logging.warning(
                "DiffV2_Load_Value_Fn: %s trained under solver_version %r, this build "
                "is %r - the fitted architecture may differ", src, ck_ver, SOLVER_VERSION)
        want_sched = self.aspace.schedule_key(self.aspace.schedule)
        if "total_position_schedule" in ck:
            saved_sched = self.aspace.schedule_key(ck["total_position_schedule"])
            if saved_sched == want_sched:
                pass                                             # same corridor — exact match
            elif saved_sched is None:
                logging.info(
                    "DiffV2_Load_Value_Fn: %s trained corridor-free (widest wealth support); "
                    "rolling under a Total_Position_Schedule only restricts to a learned "
                    "subset — valid.", src)
            elif self._corridor_inside(saved_sched):
                logging.info(
                    "DiffV2_Load_Value_Fn: %s rolls inside the corridor it trained in: every "
                    "decision step's fence lies within the stamped one — valid.", src)
            else:
                raise ValueError(
                    f"DiffV2_Load_Value_Fn corridor mismatch: {src} was trained under "
                    f"Total_Position_Schedule {saved_sched} but this run rolls under "
                    f"{want_sched}. A policy trained inside a corridor is queried off its "
                    f"learned wealth support under a different (or absent) one — retrain, "
                    f"or match the Evaluator.Total_Position_Schedule.")
        elif want_sched is not None:
            logging.warning(
                "DiffV2_Load_Value_Fn: %s predates corridor provenance (no "
                "total_position_schedule stamp) but this run sets a Total_Position_Schedule "
                "— cannot verify the frozen policy was trained in it; roll validity "
                "unverified.", src)
        self._check_action_universe(ck, src)
        self._check_calendar_spread(ck, src)
        optimism, bound = ck.get("value_optimism"), self.cfg["diffv2_max_value_optimism"]
        if optimism is not None and not optimism <= bound:
            raise ValueError(
                f"DiffV2_Max_Value_Optimism: {src} was saved with V_0={ck['V_0']:+.6g} outrunning its "
                f"held-out greedy utility by {optimism:.6g}, above this run's {bound:g}. Its DP had "
                f"not converged, so its argmax ranks on a value no roll earns - redraw the member.")
        if "streaming" in ck and not ck["streaming"]:
            logging.warning(
                "DiffV2_Load_Value_Fn: %s carries a pre-stream frame, locked on a whole "
                "fixed simulation rather than a warmup batch — the policy evaluates in "
                "ITS frame, not this run's.", src)
        return ck_ver

    def _check_action_universe(self, ck, src):
        """`Allocation_Weights` provenance, the corridor's rule applied to the action universe a
        checkpoint's argmax was fitted against. A policy trained on the Cartesian grid saw the
        WIDEST set of books, so rolling it on a ladder only restricts it to a learned subset
        (valid, said out loud). A policy trained on one ladder and rolled on a different one — or
        on the full product — is queried on books it never scored, and under
        the position state on a p-support it never covered. Refused by name, because the
        symptom otherwise is a silently worse roll rather than an error."""
        want = self.aspace.weights_key(self.aspace.weights)
        saved = self.aspace.weights_key(ck["allocation_weights"]) if "allocation_weights" in ck \
            else None
        if "allocation_weights" not in ck:
            if want is not None:
                logging.warning(
                    "DiffV2_Load_Value_Fn: %s predates Allocation_Weights provenance but this "
                    "run sets a weight schedule — cannot verify which action universe the "
                    "frozen policy was fitted on; roll validity unverified.", src)
            return
        if saved == want:
            return
        if saved is None:
            logging.info(
                "DiffV2_Load_Value_Fn: %s was fitted on the full Cartesian action grid (the "
                "widest universe); rolling it on an Allocation_Weights ladder only restricts "
                "to a learned subset — valid.", src)
            return
        if self.cfg["diffv2_load_horizon_pad"]:
            # a ladder is derived from the run's own calendar, so a knot-for-knot match is
            # impossible under a horizon-padded load and the run's own ladder is searched
            logging.info("DiffV2 load horizon pad: Allocation_Weights differ from the saved "
                         "ladder; this run searches its own.")
            return
        raise ValueError(
            f"DiffV2_Load_Value_Fn Allocation_Weights mismatch: {src} was fitted on the ladder "
            f"{saved} but this run searches {want}. The argmax scored a different set of books, "
            f"so the frozen value is queried off the support it learned — retrain, or match the "
            f"Evaluator.Allocation_Weights.")

    def _check_calendar_spread(self, ck, src):
        """`Calendar_Spread_Bps` provenance. Under the position state the charge is a term of
        the regressed TARGET, so a different rate is a different value function and a mismatch is
        refused — the same reasoning that refuses a `DiffV2_Repositioning_Charge` mismatch itself.
        Position-free, the charge only shapes SELECTION (the fitted value never saw it), so a
        re-roll under a new rate is the deliberate execution-policy re-roll `Max_Trade_Per_Step`
        and `Decision_Deadband_Sigma` also allow — warned, not refused."""
        want = self.aspace.calendar_bps
        if "calendar_spread_bps" not in ck:
            if want is not None:
                logging.warning(
                    "DiffV2_Load_Value_Fn: %s predates Calendar_Spread_Bps provenance but this "
                    "run sets a rate; cannot verify the frozen policy priced spreads.", src)
            return
        if ck["calendar_spread_bps"] == want:
            return
        if self.position_state:
            raise ValueError(
                f"DiffV2_Load_Value_Fn Calendar_Spread_Bps mismatch under the position state: "
                f"{src} was trained at {ck['calendar_spread_bps']!r} and "
                f"this run rolls at {want!r}. Under position state the reposition charge enters "
                f"the regressed target, so the two are different value functions — retrain, or "
                f"match the Evaluator.Calendar_Spread_Bps.")
        logging.warning(
            "DiffV2_Load_Value_Fn: %s was trained at Calendar_Spread_Bps %r and this run rolls "
            "at %r. Position-free the charge shapes SELECTION only, so this is a deliberate "
            "execution re-roll — but the argmax is now ranking against a different friction "
            "than the one it was tuned on.", src, ck["calendar_spread_bps"], want)

    def _capital_line(self, ck):
        """A checkpoint's capital line, its displacement over its own `risk_aversion`, in this run's
        money: under the returns state the frame is in fractions of the t0 notional, so the line
        scales by this run's notional over the training one."""
        c = ck["displacement"]
        if self.state_notional:
            trained_on = ck.get("state_notional")
            if not trained_on:
                raise ValueError(
                    "a returns-state checkpoint carries no state_notional: its capital line cannot be "
                    "read in this run's frame; stamp it with its training world's t0 spot times the "
                    "liability volume, or retrain")
            c = float(c) * self.state_notional / float(trained_on)
        return c / float(ck["risk_aversion"])

    def _corridor_inside(self, saved):
        """True when this run's `Total_Position_Schedule` lies inside the stamped `saved` one on every
        decision step, so the roll only ever queries the fitted value on learned support."""
        mine = self.aspace.schedule
        if not mine or not saved:
            return False
        knot = HedgeActionSpace.knot_at
        return all(knot(saved, t)[1] - 1e-9 <= knot(mine, t)[1] and knot(mine, t)[2] <= knot(saved, t)[2] + 1e-9
                   for t in range(self.T_dec))

    def _restore_frame(self, members):
        """Adopt a loaded checkpoint's FRAME as this run's — the z-frame, the utility scale and
        its per-decision-step schedule, the LogWealth capital line at the checkpoint's own aversion
        — and mirror it onto the runtime, so every reward and every
        ranking reads the frame the nets were fitted against rather than the one this world
        resolved for itself.

        This is `BundleStepper`'s `mirror_scale=False` contract one level up and one step finer:
        the checkpoint's `c` is part of the value function, and a per-step SCHEDULE says the same
        on every decision day. The evaluation bundle measured its own at build time and mirrored
        it on; this is where that measurement is discarded. Nothing downstream re-measures — `_bind`
        re-asserts this frame on every later batch and the stepper rollout refuses to re-mirror.

        The scalar is AVERAGED over an ensemble, and under a schedule over the members' TERMINAL
        KNEES, since the scalar is by contract the terminal read. The SCHEDULES themselves are not
        averaged: each member's continuation is read in its own frame, so the run-level schedule is
        the primary member's and only the label and terminal reads consult it."""
        loaded = members[0]
        self.utility_scale_schedule = loaded.get("utility_scale_schedule")
        if self.cfg["diffv2_load_horizon_pad"]:
            self.utility_scale_schedule = self._pad_steps(self.utility_scale_schedule,
                                                          self.T_dec + 1)
        self.capital = self._capital_line(loaded)
        if float(loaded["risk_aversion"]) != self.aversion:
            logging.info("DiffSolver frozen roll reads the checkpoint's DiffV2_Risk_Aversion %g with its "
                         "capital line; this run's %g is not read", loaded["risk_aversion"], self.aversion)
        nulls = [float(ck["drift_null_sigma"]) for ck in members if ck.get("drift_null_sigma") is not None]
        if nulls:
            # one tripwire for the ensemble's forecast: the members' nulls averaged, whatever their order
            self.drift_null_sigma = sum(nulls) / len(nulls)
        # the scalar IS the terminal read, so under a schedule it comes from the members' terminal
        # KNEES rather than a scalar that only happens to equal them
        scales = [float(ck["utility_scale_schedule"][-1] if ck.get("utility_scale_schedule")
                        else ck["utility_scale"]) for ck in members]
        if max(scales) - min(scales) > 0.01 * max(scales):
            logging.warning(
                "DiffSolver ensemble utility_scale spread %.3g%% — members trained "
                "against different anchors; averaging is approximate",
                100.0 * (max(scales) - min(scales)) / max(scales))
        self.m_mean, self.m_std = loaded["m_mean"], loaded["m_std"]
        self.w_mean, self.w_std = loaded["w_mean"], loaded["w_std"]
        self.utility_scale = float(sum(scales) / len(scales))
        mirror_frame(self.runtime, self.utility_scale, self.utility_scale_schedule)

    def warmup(self, bundle=None):
        """Fit the value function on the first (or only) bundle and LOCK the frame: the
        standardization stats, the utility scale and the per-t trust region are computed here and
        frozen for every later batch. Re-fitting them per batch would mean batch 1's C_t and batch
        2's were different functions of different inputs, and the DP recursion would compose
        mismatched frames.

        Bank RNG is deterministic; multi-seed repeats (N solvers, each warmed up once) advance the
        framework's inner-MC Sobol stream, so V_0 spread reflects inner-MC noise.

        `DiffV2_Load_Value_Fn` makes this a FROZEN-POLICY EVAL instead: restore the fitted nets AND
        each function's frame, since the train-time standardization stats and utility scale are
        part of the value function and recomputing them from a possibly stressed eval world would
        change what the nets compute. Every eval path is unseen by the nets, so the verdict rolls
        over all paths. A LIST of checkpoints is an ensemble argmax: each member evaluated in its
        own frame, continuations averaged. `_check_load_provenance` holds every member against this
        run first."""
        if bundle is not None:
            self._bind(bundle)
        logging.info(
            "DiffSolver setup: n_hedge=%d active=%s T_dec=%d (the last-live mtm) "
            "B_outer=%d levels=%d fit_iters=%d lr=%.3g | "
            "contract_size=%s | q∈[%s, %s] total_abs_limit=%.3g",
            self.aspace.n_hedge, self.aspace.active, self.T_dec,
            self.B_outer, self.aspace.levels, self.fit_iters, self.lr, self.aspace.contract_size.tolist(),
            self.aspace.q_lo.tolist(), self.aspace.q_hi.tolist(), self.aspace.total_abs_limit)

        self.sweep_ts = sweep_ts = self.bundle.decisions_from(self.t_min)
        if not sweep_ts:
            raise ValueError(f"Solver.T_Min={self.t_min} leaves no market row before T_dec={self.T_dec}")
        inner_cache = self._fork_sweep()
        # the market columns the nets read, off the swept forks: every path, since the
        # out-of-sample world is the HELD-OUT BATCH
        M = torch.cat([inner_cache[t][3] for t in sweep_ts], 0)                       # (n_swept*B, md)
        self.md = md = M.shape[-1]
        # returns-state: the wealth column and its frame live in fractions of the t0 book notional,
        # so a stamped frame travels across price epochs like the market columns do
        self.state_notional = (float(self.runtime["objective"].get("state_notional") or 0.0)
                               if self.cfg["diffv2_returns_state"] else 0.0) or None

        hidden = self.cfg["diffv2_hidden"]
        load_cfg = self.cfg["diffv2_load_value_fn"]
        # a load member is either a checkpoint PATH or an already-materialised artifact DICT, so
        # the eval-from-artifact path treats an in-memory artifact exactly like a loaded file
        load_members = ([(p if isinstance(p, dict) else str(p)) for p in load_cfg]
                        if isinstance(load_cfg, (list, tuple))
                        else ([load_cfg] if isinstance(load_cfg, dict)
                              else ([str(load_cfg)] if load_cfg else [])))
        loaded = None
        if not load_members:
            # the frame, measured once on this training bank; a load reads the checkpoint's
            logging.debug("DiffSolver warmup measures the frame on its bank")
            W_bank, q_bank = self._build_bank(self.gen)
            if self.log_ratio:
                self._set_displacement(W_bank)
            self.m_mean, self.m_std = M.mean(0), M.std(0).clamp_min(1e-6)
            Wall = torch.stack([W_bank[t] for t in sweep_ts], 0).reshape(-1)
            if self.state_notional:
                Wall = Wall / self.state_notional
            self.w_mean, self.w_std = Wall.mean(), Wall.std().clamp_min(1e-6)
            logging.info(
                "DiffSolver bank: market_dim=%d | swept W∈[%.4g, %.4g] mean=%.4g std=%.4g | "
                "q_rep(t=0)=%s", md, float(Wall.min()), float(Wall.max()),
                float(self.w_mean), float(self.w_std),
                self._replication_hedge(0).detach().cpu().tolist())
        else:
            # frozen-policy eval: restore the fitted nets AND each member's own frame
            members, srcs = [], []
            versions = set()
            for member in load_members:
                # pre-contract checkpoints still load: only the frame and net keys below are read
                ck = member if isinstance(member, dict) else torch.load(member, map_location=self.device)
                src = "<in-memory artifact>" if isinstance(member, dict) else member
                srcs.append(src)
                versions.add(self._check_load_provenance(ck, src, md))
                drift = ((M.mean(0) - ck["m_mean"]).abs() / ck["m_std"]).max()
                logging.info(
                    "DiffSolver LOADED value fn from %s (train V_0=%+.6g) | eval-world "
                    "market drift vs train frame: max %.3g σ | utility_scale %.6g | frame %s",
                    src, ck["V_0"], float(drift), ck["utility_scale"], ck.get("frame_stamp"))
                members.append(ck)
            if len({bool(ck.get("streaming", False)) for ck in members}) > 1:
                raise ValueError(
                    "DiffV2_Load_Value_Fn ensemble mixes a streaming-locked frame with a "
                    "fixed-set one. The argmax averages the members' continuations, which is "
                    "only meaningful when each member standardizes the same state the same way — "
                    "load one provenance or the other, never both.")
            if len(versions) > 1:
                raise ValueError(
                    f"DiffV2_Load_Value_Fn: the ensemble mixes solver versions {sorted(versions)} "
                    "- members trained under different architectures cannot be averaged")
            # members whose swept books disagree beyond the declaration are DPs that have not
            # converged to one policy, and their mean ranks on the steepest member's slope
            books = [(src, [sum(q) for q in ck["q_star_path"]])
                     for src, ck in zip(srcs, members) if ck.get("q_star_path")]
            limit = self.cfg["diffv2_max_member_disagreement"]
            for i, (src_a, a) in enumerate(books):
                for src_b, b in books[i + 1:]:
                    apart = sum(abs(x - y) for x, y in zip(a, b)) / max(min(len(a), len(b)), 1)
                    if apart > limit:
                        raise ValueError(
                            f"DiffV2_Max_Member_Disagreement: {src_a} and {src_b} sweep net books "
                            f"{apart:.4g} contracts apart on average, above this run's {limit:g} "
                            f"- the ensemble has not converged to one policy.")
            loaded = members[0]
            self._restore_frame(members)
            hidden = int(loaded["hidden"])
        in_dim = self._in_dim(md)             # (market | [W] | [p]) — the row `_standardize` builds
        nets = [_DiffV2Residual(in_dim, hidden=hidden).to(self.device)
                for _ in range(self.T_dec)]
        # per-t trust region for A_t evaluation, set at fit time or restored from a checkpoint;
        # None = unclamped
        self.a_bounds = (self._pad_steps(loaded["a_bounds"], self.T_dec)
                         if loaded is not None and loaded.get("a_bounds")
                         else [None] * self.T_dec)
        logging.info("DiffSolver action grid: K=%d actions (levels=%d ^ active=%d)",
                     self.aspace.universe_size(), self.aspace.levels, self.aspace.n_active)

        self.nets = nets
        self.hidden = hidden
        self.loaded = loaded
        self.rows = []
        if loaded is not None:
            for net, sd in zip(nets, self._pad_steps(loaded["state_dicts"], self.T_dec)):
                net.load_state_dict(sd)
                net.eval()
            if len(members) > 1:
                # ensemble: per-member net stacks and frames; _continuation averages members
                self._ensemble = []
                for ck in members:
                    m_nets = [_DiffV2Residual(in_dim, hidden=int(ck["hidden"])).to(self.device)
                              for _ in range(self.T_dec)]
                    for net, sd in zip(m_nets, self._pad_steps(ck["state_dicts"], self.T_dec)):
                        net.load_state_dict(sd)
                        net.eval()
                    pad = self.cfg["diffv2_load_horizon_pad"]
                    self._ensemble.append(
                        (m_nets, ck["m_mean"], ck["m_std"], ck["w_mean"], ck["w_std"],
                         self._pad_steps(ck.get("a_bounds"), self.T_dec) if pad
                         else ck.get("a_bounds"),
                         self._pad_steps(ck.get("utility_scale_schedule"), self.T_dec + 1)
                         if pad else ck.get("utility_scale_schedule"),
                         self._capital_line(ck)))           # the member's own capital line
                logging.info("DiffSolver ENSEMBLE argmax over %d value fns", len(members))
            self.worst = float(loaded["max_abs_Y_boot"])
            self.root = {"t": sweep_ts[0], "Y_mean": float(loaded["V_0"]),
                         "q_star_mean": list(loaded["n_star_0"])}
        else:
            self._sweep(W_bank, q_bank)
        # the frame is now locked, and streaming freezes the trust region from here on: later
        # batches report their breach rate instead of re-fitting it
        self._bounds_frozen = True

    def _sweep(self, W_bank, q_bank):
        """One backward pass on the CURRENTLY BOUND bundle: fit C_t on each decision row from the last
        down to `T_Min` (`Bundle.decisions_from`) against the cached inner-MC forks. warmup runs it on
        batch 1, each streaming step again on fresh paths with the same nets and optimizers.

        `live_prev` is the leg mask of the step that SET the book standing at t — the one mask
        rule for the position state (`_fit_step`). The sweep runs over the decision rows, each a
        row of market time whose successor is the next one (`self.nxt`); the first swept step has no
        predecessor in the cache and `nets[t_min]` is fitted but never queried, so it falls back to
        its own."""
        rows = []
        for t in reversed(self.sweep_ts):
            r = self._fit_step(self.nets, W_bank, t, self.inner_cache[t], q_bank,
                               live_prev=self.inner_cache.get(self.prev.get(t), (None,) * 5)[4])
            rows.append(r)
            logging.info(
                "DiffSolver C[t=%d] fitted (%d iters): val_loss=%.4g |Y_boot|=%.4g |A|=%.4g "
                "Y_mean=%+.4g q*_mean=%s", r["t"], r["iters"], r["val_loss"], r["Y_absmean"],
                r["A_absmean"], r["Y_mean"],
                ["%.3f" % v for v in r["q_star_mean"]])
        self.rows = rows
        self.worst = max((r["Y_absmean"] for r in rows if math.isfinite(r["Y_absmean"])),
                         default=0.0)
        self.root = rows[-1] if rows else {"t": self.sweep_ts[0], "Y_mean": 0.0, "q_star_mean":
                                           [0.0] * self.aspace.n_hedge}

    def _fork_sweep(self):
        """One inner-MC fork per swept t on the bound world, reused for the argmax, the bootstrap,
        market_t and the bank's live legs, beside it the risk term's (`_ahead`); the last world's
        forks are freed before these are made, or both sets are live at once."""
        self.inner_cache = None
        self.inner_cache = {t: self._inner_step(t) for t in self.sweep_ts}
        # the risk term's forks read the leg mask at their landing off the one-step ones
        self.inner_cache = {t: fork + (self._ahead(t),) for t, fork in self.inner_cache.items()}
        return self.inner_cache

    def step(self, bundle):
        """Continue training on a FRESH batch: re-bind, rebuild the exploration bank and the
        inner-MC cache on the new paths, then sweep again with the same nets, the same optimizer
        moments and the frozen frame. Fresh paths per fit step are the point — overfitting to one
        simulated set stops being structurally possible.

        A LOADED checkpoint means evaluation, so there is nothing to continue: sweeping here would
        fine-tune the "frozen" policy on the evaluation world and report a verdict for a policy
        that is not the checkpoint and is never written anywhere."""
        if self.loaded is not None:
            logging.info("DiffSolver streaming step SKIPPED: value fn loaded — frozen-policy "
                         "eval, so this batch trains nothing")
            return
        self._bind(bundle)
        self._fork_sweep()
        W_bank, q_bank = self._build_bank(self.gen)
        self._breaches = []
        self._sweep(W_bank, q_bank)
        self._log_breaches()
        logging.info("DiffSolver streaming step complete: max|Y_boot|=%.4g V_0=%+.6g "
                     "n_star@t=%d=%s", self.worst, float(self.root["Y_mean"]),
                     self.root["t"], self.root["q_star_mean"])

    def _log_breaches(self):
        """Report how often this batch's fitted A-targets fell OUTSIDE the frozen trust region.
        A materially non-zero rate means the region locked at warmup no longer covers the states
        fresh batches produce — the signal for widening it from real targets."""
        if not self._breaches:
            return
        worst = max(self._breaches, key=lambda r: r[1])
        lo, hi = self.a_bounds[worst[0]]
        logging.info(
            "DiffSolver trust region (frozen at warmup) breach rate over %d fitted steps: "
            "mean=%.4f max=%.4f | worst t=%d frac=%.4f targets∈[%+.4g, %+.4g] "
            "bounds=[%+.4g, %+.4g]", len(self._breaches),
            sum(r[1] for r in self._breaches) / len(self._breaches), worst[1], worst[0],
            worst[1], worst[2], worst[3], lo, hi)

    def _policy_artifact(self, nets, md, hidden, V_0, n_star_0, worst, optimism):
        """The policy artifact: the fitted nets, the frame they read their inputs through, and the
        provenance every `DiffV2_Load_Value_Fn` guard compares against. Built ONCE per run (in
        `finish`) and returned AND `torch.save`d, so the file and the in-memory dict are the same
        object — the eval-from-artifact path is identical to loading the file.

        The stamps a load is refused on are written off `_provenance`, the table
        `_check_load_provenance` holds a load to, so neither can forget one the other reads."""
        return {
            "state_dicts": [net.state_dict() for net in nets],
            "m_mean": self.m_mean, "m_std": self.m_std,
            "w_mean": self.w_mean, "w_std": self.w_std,
            "utility_scale": float(self.runtime["objective"]["utility_scale"]),
            # the per-step knees this policy was fitted against (None = the single scalar), part of
            # the FRAME, so a load restores it rather than re-measuring
            "utility_scale_schedule": self.utility_scale_schedule,
            # the stamps a load is refused on (`_provenance`)
            **{key: want for key, (want, _, _) in self._provenance(md, {}).items()},
            "a_bounds": self.a_bounds,
            # corridor provenance: the Total_Position_Schedule this policy was trained inside
            # (None = unconstrained). A load under a DIFFERENT corridor fails loud.
            "total_position_schedule": self.aspace.schedule_key(self.aspace.schedule),
            # action-universe provenance: the Allocation_Weights this argmax searched (None = the
            # Cartesian grid). `config_hash` cannot stand in - it hashes runtime['solver'] while
            # this and the corridor live under runtime['accounting'].
            "allocation_weights": self.aspace.weights_key(self.aspace.weights),
            # friction provenance: under the position state the calendar rate is a term of the
            # charge entering the regressed TARGET, so another rate is a different value function
            "calendar_spread_bps": self.aspace.calendar_bps,
            "hidden": hidden,
            # the growth reward's shift, measured off the training bank's worst book - part of the
            # FRAME, since a roll under a different shift is a different utility
            "displacement": self.displacement,
            # the notional the returns-state frame divides wealth by; the roll scales the
            # displacement by its own notional over this one, so the reward stays in the frame
            "state_notional": getattr(self, "state_notional", None),
            "drift_null_sigma": self.drift_null_sigma,
            "risk_aversion": self.aversion,
            "phi_curves": getattr(self, "phi_curves", None),
            "solver_version": SOLVER_VERSION,
            "config_hash": self._config_hash(),
            # Frame provenance: WHICH path population locked the frame this policy reads its
            # inputs through, and a stamp over the frame itself (loads compare both).
            "frame_stamp": self._frame_stamp(),
            # Headline echoed so a loaded eval reads it back rather than recomputing.
            "V_0": V_0, "n_star_0": list(n_star_0), "max_abs_Y_boot": worst,
            # convergence stamps a roll is held to: V_0 less the held-out greedy utility, and the
            # swept mean book per decision step from t_min, which an ensemble's members must share
            "value_optimism": optimism,
            "q_star_path": [r["q_star_mean"] for r in reversed(self.rows)],
        }

    def _frame_stamp(self):
        """sha1 of the LOCKED frame — the utility scale AND its per-step schedule, the
        z-frame (market/wealth mean+std), the trust-region envelope and the streaming flag. Two
        checkpoints with the same stamp compute the same function of the same standardized
        state; a different stamp is a different frame,
        which is exactly what makes averaging them in one argmax (or reloading under a re-locked
        frame) invalid. Stamped into every artifact, logged on every load.

        A frame with no scale schedule hashes exactly as it did before schedules existed: the key
        is OMITTED rather than hashed as null, so a run that never armed the mode keeps its stamp
        and two checkpoints from either side of the feature still compare on what they compute."""
        env = [b for b in self.a_bounds if b is not None]
        sched = self.utility_scale_schedule
        frame = {
            "utility_scale": round(float(self.runtime["objective"]["utility_scale"]), 6),
            "m_mean": [round(float(v), 6) for v in self.m_mean.tolist()],
            "m_std": [round(float(v), 6) for v in self.m_std.tolist()],
            "w_mean": round(float(self.w_mean), 6), "w_std": round(float(self.w_std), 6),
            "a_bounds": ([round(min(b[0] for b in env), 6), round(max(b[1] for b in env), 6)]
                         if env else None),
        }
        if sched is not None:
            frame["utility_scale_schedule"] = [round(float(v), 6) for v in sched]
        return hashlib.sha1(json.dumps(frame, sort_keys=True).encode()).hexdigest()

    def finish(self, bundle):
        """Verdict, benchmarks-facing headline and the policy artifact. STREAMING passes the
        HELD-OUT batch — a world no fit step ever saw, so the whole batch is out-of-sample. A
        frozen eval is the stream of length one: warmup's batch is handed straight back here.

        A newly bound (held-out) world needs its OWN inner-MC forks: the argmax reads
        E_inner[C_{t+1}] at each swept t, and the cached ones belong to the last TRAINING batch. A
        stream of length 1 — a frozen eval, where warmup's batch IS the held-out one — is already
        bound to it, and re-forking would advance the Sobol stream and move the verdict.

        The POLICY ARTIFACT is built ONCE here: the fitted value function, its frame and the
        provenance stamps. It is the single source returned via `SolverResult` AND torch.saved to
        `DiffV2_Save_Value_Fn`, so the file and the in-memory dict are the same object and the
        eval-from-artifact path is identical to loading the file. A LOADED run produces none —
        nothing was fitted, and re-emitting the file's policy would claim this run as its
        provenance.

        The downside verdict — greedy policy against textbook delta hedge, no hedge and passive — is
        OUT-OF-SAMPLE by construction, since `finish` always runs on a world no fit step saw. With
        a checkpoint loaded and `DiffV2_Stepper_Rollout` set, `_rollout_on_stepper` adds the
        deployment-faithful backtest, which is the trustworthy P&L for a walk-forward because the
        simplified `_verdict` wealth recursion mis-accrues expiry."""
        if bundle is not self.bundle:
            self._bind(bundle)
            self._fork_sweep()
        nets, sweep_ts, inner_cache = self.nets, self.sweep_ts, self.inner_cache
        loaded, rows, worst, root = self.loaded, self.rows, self.worst, self.root
        md, hidden = self.md, self.hidden
        V_0 = float(root["Y_mean"])
        n_star_0 = root["q_star_mean"]
        bounded = math.isfinite(V_0) and worst < 1.0e4
        if loaded is None:
            logging.info(
                "DiffSolver sweep complete: t=%d→%d | max|Y_boot|=%.4g (%s) | "
                "V_0=%+.6g | n_star@t=%d=%s", self.T_dec - 1, self.t_min, worst,
                "BOUNDED" if bounded else "EXPLODED", V_0, root["t"], n_star_0)
        # downside verdict, OUT-OF-SAMPLE by construction: all rows of the held-out world
        verdict = self._verdict(nets, inner_cache, sweep_ts, rows=slice(None))
        # POLICY ARTIFACT, built ONCE here and after the verdict, whose drift null it stamps;
        # a LOADED run produces none
        artifact = None
        save_path = self.cfg["diffv2_save_value_fn"]
        if loaded is None:
            # the fitted value against what its own argmax earns held out: a bootstrap that
            # outruns it has ratcheted an unfitted slope into V_0, or its policy ruins. Under the
            # position state both are net of the repositioning charge.
            u_held = float(verdict["greedy"]["u_mean"])
            optimism = V_0 - u_held
            artifact = self._policy_artifact(nets, md, hidden, V_0, n_star_0, worst, optimism)
            if save_path:
                if not math.isfinite(V_0):
                    raise ValueError(
                        f"refusing to save non-finite value fn to {save_path}: V_0={V_0}")
                bound = self.cfg["diffv2_max_value_optimism"]
                if not optimism <= bound:
                    raise ValueError(
                        f"DiffV2_Max_Value_Optimism: refusing to save the value fn to {save_path}: "
                        f"V_0={V_0:+.6g} outruns the held-out greedy utility {u_held:+.6g} by "
                        f"{optimism:.6g}, above {bound:g}. The fit has not converged - redraw the "
                        f"seed or lengthen the stream.")
                tmp = save_path + ".tmp"
                torch.save(artifact, tmp)
                os.replace(tmp, save_path)                       # atomic on POSIX
                logging.info("DiffSolver SAVED value fn to %s (V_0=%+.6g)", save_path, V_0)

        # deployment-faithful backtest through BundleStepper: real futures accounting, decisions
        # off the stepper's own wealth
        stepper_verdict = None
        if loaded is not None and self.cfg["diffv2_stepper_rollout"]:
            stepper_verdict = self._rollout_on_stepper(nets, inner_cache, sweep_ts)
            sg, stb, snh, spa = (stepper_verdict[k]
                                 for k in ("greedy", "textbook", "nohedge", "passive"))
            logging.info(
                "DiffSolver STEPPER ROLLOUT (frozen policy, realized path, real accounting):\n"
                "  greedy   wT=%+.4e p5=%+.4e cvar5=%+.4e\n"
                "  textbook wT=%+.4e p5=%+.4e cvar5=%+.4e\n"
                "  nohedge  wT=%+.4e p5=%+.4e cvar5=%+.4e\n"
                "  passive  wT=%+.4e p5=%+.4e cvar5=%+.4e",
                sg["wT_mean"], sg["wT_p5"], sg["wT_cvar5"],
                stb["wT_mean"], stb["wT_p5"], stb["wT_cvar5"],
                snh["wT_mean"], snh["wT_p5"], snh["wT_cvar5"],
                spa["wT_mean"], spa["wT_p5"], spa["wT_cvar5"])

        g, tb, nh, pa = (verdict[k] for k in ("greedy", "textbook", "nohedge", "passive"))
        # PRIMARY metric = the optimization target E[u(W_T)], which already encodes downside
        # aversion through the concave utility; CVaR5 is a secondary tail diagnostic
        beats_nh = g["u_mean"] >= nh["u_mean"]
        beats_tb = g["u_mean"] >= tb["u_mean"]
        tail_vs_tb = g["wT_cvar5"] >= tb["wT_cvar5"] - abs(tb["wT_cvar5"]) * 0.05
        logging.info(
            "DiffSolver VERDICT (%s rollout t=%d→T over %d outer paths, from the opening book):\n"
            "  policy    u(W_T)mean    W_T mean       W_T p5         W_T CVaR5\n"
            "  greedy    %+.5f    %+.4e   %+.4e   %+.4e\n"
            "  textbook  %+.5f    %+.4e   %+.4e   %+.4e\n"
            "  nohedge   %+.5f    %+.4e   %+.4e   %+.4e\n"
            "  passive   %+.5f    %+.4e   %+.4e   %+.4e\n"
            "  → on the OBJECTIVE E[u(W_T)]: beats no-hedge=%s, beats textbook=%s | "
            "tail(CVaR5) competitive w/ textbook=%s",
            "OUT-OF-SAMPLE", self.t_min, self.B_outer,
            g["u_mean"], g["wT_mean"], g["wT_p5"], g["wT_cvar5"],
            tb["u_mean"], tb["wT_mean"], tb["wT_p5"], tb["wT_cvar5"],
            nh["u_mean"], nh["wT_mean"], nh["wT_p5"], nh["wT_cvar5"],
            pa["u_mean"], pa["wT_mean"], pa["wT_p5"], pa["wT_cvar5"],
            beats_nh, beats_tb, tail_vs_tb)
        logging.info(
            "DiffSolver greedy positions: mean|q| per instrument=%s | q@t0=%s | q@mid=%s "
            "(textbook q@t0=%s)", ["%.2f" % v for v in verdict["greedy_mean_abs_q"]],
            verdict.get("greedy_q_first"), verdict.get("greedy_q_mid"),
            (verdict["textbook_q_traj"] or [None])[0])

        return SolverResult(
            solver_name=type(self).__name__,
            actions=torch.tensor(n_star_0),
            values=V_0,
            value_fn_artifacts=artifact,              # the fitted policy (None in eval-from-load runs)
            diagnostics={
                "V_0": V_0,
                "n_star_0": n_star_0,
                "max_abs_Y_boot": worst,
                "bounded": bool(bounded),
                "root_t": int(root["t"]),
                "per_t": rows,
                "action_grid_size": self.aspace.universe_size(),
                "market_dim": md,
                "value_fn_path": save_path or None,       # where the artifact was persisted (if any)
                "verdict": verdict,                       # OUT-OF-SAMPLE (held-out paths)
                "stepper_verdict": stepper_verdict,       # frozen-policy realized-path rollout (real accounting)
                "verdict_is_oos": True,        # structural: finish only ever sees unfitted paths
                "verdict_beats_nohedge_on_utility": bool(beats_nh),
                "verdict_beats_textbook_on_utility": bool(beats_tb),
                "verdict_tail_competitive_vs_textbook": bool(tail_vs_tb),
            },
        )


# `DiffSolver` is the production deliverable; `HindsightDpSolver` is the upper-bound benchmark
# track and `run_textbook_benchmark` the lower-bound one.
class StreamingSolve:
    """The solve driver, one `Bundle` per simulation batch; documented on the Solver page."""

    documentation = ('Solver', [
        'A solve is a stream. `HedgeMonteCarlo.execute` builds one bundle per simulation batch and',
        'hands each over as it is built: `warmup` on the first, which constructs the solver, one per',
        '`Multi_Seed_Count` seed fed the same batches, and locks its frame (`DiffSolver`); `step` on',
        'every later batch but the last, fresh paths under the same nets, optimizer moments and',
        'frame; and `finish` on the last, never fitted, the held-out world the verdict and the',
        'benchmark tracks are measured on. A fork is as wide as one batch, and every fit step sees',
        'paths no earlier step did.',
        '',
        '!!! warning "Invariant: a frozen evaluation is the stream of length one"',
        '    Under `DiffV2_Load_Value_Fn` nothing is fitted, so the one batch is both warmup and',
        '    held-out world; `step` skips a loaded net, and `finish` does not re-fork its own',
        '    bundle, which would advance the Sobol stream and move the verdict. **Held by**',
        '    `tests/test_frozen_policy_is_frozen.py`.',
    ])

    def __init__(self, runtime):
        self.runtime = runtime
        self.cfg = runtime["solver"]
        self.solvers = []
        self.trained_batches = 0

    def warmup(self, bundle):
        """Batch 1: build the solver(s) and fit the first backward sweep. The frame (utility
        scale, z-frame, trust region) is locked here and frozen for every later batch."""
        n_seed = max(1, self.cfg["multi_seed_count"])
        cls = DiffSolverV2 if self.cfg["object"] == "diffsolverv2" else DiffSolver
        self.solvers = [cls(bundle, self.runtime) for _ in range(n_seed)]
        for solver in self.solvers:
            solver.warmup(bundle)
        # same rule as `step`: a loaded checkpoint fits nothing, so this batch was not a training one
        self.trained_batches = int(self.solvers[0].loaded is None)
        logging.info(
            "StreamingSolve WARMUP on batch 1: %d outer paths x %d seed(s) — frame LOCKED "
            "(utility_scale=%.6g, z-frame + trust region from this batch)",
            self.solvers[0].B_outer, len(self.solvers), self.solvers[0].utility_scale)

    def step(self, bundle):
        """A later batch: continue the same nets on fresh paths under the frozen frame. With a
        checkpoint loaded the solvers train nothing (a frozen-policy eval), so the batch is just
        consumed and `trained_batches` does not move."""
        self.trained_batches += self.solvers[0].loaded is None
        logging.info("StreamingSolve STEP on batch %d (%d outer paths)",
                     self.trained_batches, int(bundle.liability_sim.shape[-1]))
        for solver in self.solvers:
            solver.step(bundle)

    def finish(self, held_out):
        """The held-out batch: verdict + benchmarks + artifact, on paths no fit step saw."""
        logging.info(
            "StreamingSolve FINISH on the held-out batch (%d outer paths) after %d training "
            "batch(es) — this world was never fitted, so the whole batch is out-of-sample",
            int(held_out.liability_sim.shape[-1]), self.trained_batches)
        runs = [solver.finish(held_out) for solver in self.solvers]
        return assemble_hedge_result(runs, held_out, self.runtime)


def assemble_hedge_result(primary_runs, bundle, runtime):
    """Assemble the primary solver runs (one per seed) plus the benchmark tracks (hindsight upper
    bound / textbook lower bound, enabled by the `Run_*` flags) into the `comparison` table —
    V_0 mean ± std per track — the acceptance ladder, and the result dict
    `HedgeMonteCarlo.execute` unpacks. `bundle` is the HELD-OUT batch, which is exactly the world
    the verdict was rolled on, so the benchmark tracks measure the same paths.

    The ladder compares V_0 across tracks, and both benchmarks are TERMINAL-utility by
    construction. Under `Objective.Reference_Mode='Running_Wealth'` the solver's own V_0 is a sum
    of per-step utilities instead, so the rungs are not on one scale and their ORDER says nothing;
    the per-track table and the verdict's `wT_mean` stay comparable."""
    solver_cfg = runtime["solver"]
    obj = solver_cfg["object"]
    have_liability = bundle.liability_value is not None
    primary = primary_runs[0]
    comparison = {primary.solver_name: SolverResult.multiseed_summary(primary_runs)}

    # benchmark tracks, assembled alongside the DiffSolver-family deliverable
    if obj in ("diffsolver", "diffsolverv2"):
        for flag, label in (("run_hindsight_diagnostic", "hindsight"),
                             ("run_textbook_benchmark", "textbook")):
            if solver_cfg[flag] and not have_liability:
                logging.warning("solve_hedge: %s requested but bundle has no "
                                 "liability_value — track skipped", label)
        if solver_cfg["run_hindsight_diagnostic"] and have_liability:
            comparison["HindsightDpSolver"] = SolverResult.multiseed_summary(
                [HindsightDpSolver(bundle, runtime).solve()])     # deterministic — one run
        if solver_cfg["run_textbook_benchmark"] and have_liability:
            comparison["textbook"] = run_textbook_benchmark(bundle, runtime)

    # acceptance ordering - hindsight >= DiffSolver >= textbook over the tracks present, with a
    # tolerance for Monte-Carlo noise
    rungs = [(label, comparison[key]["v0_mean"])
             for key, label in (("HindsightDpSolver", "hindsight"),
                                (primary.solver_name, primary.solver_name),
                                ("textbook", "textbook")) if key in comparison]
    ladder = {"order": rungs,
              "holds": all(rungs[i][1] >= rungs[i + 1][1] - 1.0e-6 for i in range(len(rungs) - 1))}
    if len(rungs) > 1 and (runtime["objective"] or {}).get("reference_mode") == "running_wealth":
        logging.warning(
            "solve_hedge ladder under Objective.Reference_Mode='Running_Wealth': the solver's V_0 "
            "is a SUM of per-step utilities while the hindsight/textbook tracks are terminal — the "
            "rungs are not on one scale and their order is not an acceptance statement.")
    logging.info("solve_hedge tracks: %s | ladder holds=%s",
                 {k: round(v["v0_mean"], 4) for k, v in comparison.items()},
                 ladder["holds"])
    return {
        "evaluation_output": {
            "solver_name": primary.solver_name,
            "solver_result": primary,
            "actions": primary.actions,
            "values": primary.values,
            "diagnostics": primary.diagnostics,
            "comparison": comparison,
            "ladder": ladder,
        },
        "policy_artifact": primary.value_fn_artifacts,
    }


class DiffSolverV2(DiffSolver):
    """`DiffSolver` with its training bank rolled by a policy constructed from the simulated
    liability rather than noise around the replication hedge; documented on the Solver page."""

    SOLVER_OBJECT = "DiffSolverV2"

    documentation = ('Solver', [
        '`Solver.Object: DiffSolverV2` is `DiffSolver` with its training bank replaced: where',
        "`DiffSolver`'s bank explores noise around the per-step replication hedge, this one rolls a",
        'policy CONSTRUCTED from the simulated liability. The decision rule, the fitted value, the',
        'objective and the verdict are `DiffSolver`\'s.',
        '',
        '- **Forward pass.** At each decision row $t$ the curve $d_t(L)=P(L_T<V\\mid L_t=L)$ is',
        '  measured on the flat liability\'s outer paths, $V$ the floor `DiffV2_Book_Floor_Per_Unit`',
        "  times the liability's leg volume: up to 64 buckets in $L_t$ of at least 32 paths, made",
        '  non-increasing in $L$. The net position is',
        '  $q_{net}=hi-d_t(W_t)\\,(hi-lo)$ inside the feasible band, rounded away from zero to whole',
        '  contracts and water-filled onto the replication hedge\'s shape over the active legs, or',
        '  under `DiffV2_Bank_Along_Ladder` held as a rung of the ladder on its live legs.',
        '- **Clairvoyant reserve.** After the first step, each path moves its position by the',
        '  fewest whole contracts that keep the next day\'s book above the reserve its own remaining',
        '  path requires, a backward recursion on the realized moves. Half the bank paths roll this',
        '  floored policy and half the undefended delta, both with `DiffV2_Bank_Noise_Frac`',
        '  exploration around them.',
        '- **Backward pass.** The `DiffSolver` sweep, fitted on that bank.',
        '',
        'The hindsight shapes WHERE the nets are trained, not the deployed decision, which is the',
        "argmax over the fitted value on information at $t$; the verdict is out of sample as in",
        "`DiffSolver`. The step curves are part of the frame: measured at warmup, saved with the",
        'checkpoint, and a checkpoint written by the other solver class is refused by name. Under',
        "`Evaluator.Allocation_Mode: Carry_Variance` this solver derives the per-step leg weights",
        "from the outer paths, a long-only minimum of tracking variance less carry at the capital",
        "line's exchange rate.",
    ])

    def __init__(self, bundle, runtime):
        super().__init__(bundle, runtime)
        self.phi_curves = None

    # ---- the forward pass -------------------------------------------------
    @staticmethod
    def _phi_curve(x, y, buckets=64):
        """Bucketed conditional mean P(y=1|x), isotonic DECREASING in x (a higher book mark
        cannot raise the breach probability). Returns (bucket centers, probabilities) knots
        for linear interpolation; a degenerate x (t=0, every path at the same mark) is the
        flat unconditional mean."""
        if float(x.std()) < 1e-9:
            m = y.float().mean()
            return x.new_tensor([-1e30, 1e30]), torch.stack([m, m])
        o = torch.argsort(x)
        n = x.shape[0]
        k = max(2, min(int(buckets), n // 32))
        m = (n // k) * k                                 # tail paths fold into the last bucket
        bx = x[o][:m].reshape(k, -1).mean(-1)
        bp = y[o][:m].float().reshape(k, -1).mean(-1)
        bp = bp.flip(0).cummax(0).values.flip(0)
        return bx, bp

    @staticmethod
    def _phi_apply(curve, x):
        bx, bp = curve
        idx = torch.searchsorted(bx.contiguous(), x.contiguous()).clamp(1, bx.shape[0] - 1)
        x0, x1 = bx[idx - 1], bx[idx]
        w = ((x - x0) / (x1 - x0).clamp_min(1e-12)).clamp(0.0, 1.0)
        # the curve is float32 by construction, read in the books' own precision
        return torch.lerp(bp[idx - 1].to(w.dtype), bp[idx].to(w.dtype), w)

    def _constructed_policy(self, live=None):
        """One deterministic pass, measured once and constructed twice: the delta
        `d_t(L) = P(L_T < floor | L_t = L)` is measured on the FLAT liability, the only book that
        exists before a policy does, and the policy rolls once against those curves, read at its own
        book, so no fixed point is needed. Given the per-step mask `live` of the legs that still price, each
        step's net spans the `HedgeActionSpace.ladder` band, is held as its `rung` and its reserve
        is priced on the ladder's weights; absent `live`, the net spans the `bounds` band and is
        water-filled onto the replication hedge's shape, its reserve priced in that shape. Returns
        `(the floored q per decision row (B, n_hedge), the undefended delta's, curves per decision
        row, the floored terminal book)`, each decision holding its book to the next."""
        L, T = self.liability_sim, self.T_dec
        rep = {t: self._replication_hedge(t) for t in self.steps}
        moves = {t: self._outer_move(t) for t in self.steps}
        LT = L[T]
        logging.debug("constructed policy: measured once over %d decision rows", len(self.steps))

        def compose(t, net):
            if live is not None:
                return self.aspace.rung(t, net, live[t])
            return self.aspace._largest_remainder(self.aspace.waterfill(
                rep[t][None].expand(self.B_outer, self.aspace.n_hedge).clone(), net, net), net)

        # THE CLAIRVOYANT RESERVE, backward per path: req_t is the minimum book at t from which the
        # known remaining path can be played without dipping under the floor. Days a futures hedge
        # cannot defend propagate their need BACKWARD, so the forward pass pre-positions.
        # the floor, per unit of the liability's leg volume (`Bundle.total_leg_volume`)
        floor = self.cfg["diffv2_book_floor_per_unit"] * self.leg_volume
        m_t, bands, lo_is, hi_is = {}, {}, {}, {}
        for t in self.steps:
            bands[t] = lo_t, hi_t = (self.aspace.bounds(t)[2:] if live is None
                                     else self.aspace.ladder(t, live[t])[4:])
            lo_is[t], hi_is[t] = self.aspace.inward(lo_t, hi_t)
            if self._dust(rep[t]):
                m_t[t] = torch.zeros(self.B_outer, device=self.device)
            else:
                w = rep[t] / rep[t].sum() if live is None else self.aspace._weights_at(t, live[t])
                m_t[t] = torch.einsum('i,ib->b', w * self.aspace.contract_size, moves[t][0].T.contiguous())
        req = LT.new_full((self.B_outer,), floor)
        reqs = {}
        for t in reversed(self.steps):
            reqs[t] = req.clone()
            best = torch.maximum(hi_is[t] * m_t[t], lo_is[t] * m_t[t])
            req = torch.maximum(req - moves[t][1] - best,
                                torch.full_like(req, floor))
        # the strike is the FLOOR. The danger measured is terminal capital breach rather than a
        # high-water ratchet, so a rallying book with no breach risk reads d ~ 0 and the map parks
        # the seed at the TOP of the bound.
        curves = [self._phi_curve(L[t], LT < floor) for t in self.steps]

        def construct(floored):
            q, W = {}, L[0].clone()
            for t, cv in zip(self.steps, curves):
                if self._dust(rep[t]):
                    # the below-today comparison is float noise, so the constructed book is FLAT (the
                    # wealth step below still accrues the settling liability rows)
                    qt = torch.zeros(self.B_outer, self.aspace.n_hedge, device=self.device)
                else:
                    g = self._phi_apply(cv, W)
                    (lo_t, hi_t), lo_i, hi_i = bands[t], lo_is[t], hi_is[t]
                    net = hi_t - g * (hi_t - lo_t)
                    # WHOLE contracts, rounded AWAY from zero (up, in cover terms); the box is snapped
                    # inward so every integer net is realizable
                    net = torch.where(net < 0, net.floor(), net.ceil()).clamp(lo_i, hi_i)
                    # THE FLOOR BINDS, made enforceable by the seed's one-day hindsight: where the
                    # delta base would let tomorrow's book breach it, hold the smallest position that
                    # keeps it above. A day too small to defend keeps the delta base.
                    m, dL = m_t[t], moves[t][1]
                    # tomorrow's book must clear tomorrow's RESERVE, not just the floor. `reqs[t]` IS
                    # the requirement at t+1, since the backward loop clones before it updates.
                    target = reqs[t]
                    if t == self.steps[0] or not floored:
                        # t0 is the ONE decision deployment shares with the seed - a single real state
                        # and a single position - so it takes the uniform honest delta with NO
                        # clairvoyance; paths needing more catch up at t1
                        q[t] = compose(t, net.unsqueeze(-1))
                        W = wealth_step(W, q[t], self.aspace.contract_size, *moves[t])
                        continue
                    # the floor constraint q*m >= need is a HALF-LINE, and the repair is the delta base
                    # PROJECTED onto it: the least change, rounded toward the feasible side. An
                    # up-day's bound is a LIMIT on the short - rounding away from zero walks through.
                    defensible = m.abs() > 1e-9
                    bound = (target - W - dL) / torch.where(defensible, m, torch.ones_like(m))
                    net = torch.where(defensible & (m > 0),
                                      torch.maximum(net, bound.ceil()), net)
                    net = torch.where(defensible & (m < 0),
                                      torch.minimum(net, bound.floor()), net)
                    net = net.clamp(lo_i, hi_i).unsqueeze(-1)
                    qt = compose(t, net).clamp(self.aspace.q_lo, self.aspace.q_hi)
                    # EXECUTION-EXACT top-up: the bound above prices the hedge at the weighted basket
                    # move while the executed split deviates, so re-price the EXECUTED book and
                    # top the net up by the least whole contracts that clear the shortfall
                    exec_pnl = (qt * self.aspace.contract_size * moves[t][0]).sum(-1)
                    short = target - (W + dL + exec_pnl)
                    fix = defensible & (short > 0)
                    if bool(fix.any()):
                        # +1 because the exact amount can be re-split away again on the integer
                        # lattice; the floor's guarantee under whole-contract trading is therefore to
                        # within one contract's day move
                        extra = torch.where(fix, ((short / m.abs()).ceil() + 1.0) * torch.sign(m),
                                            torch.zeros_like(m))
                        net = (net.squeeze(-1) + extra).clamp(lo_i, hi_i).unsqueeze(-1)
                        qt = compose(t, net).clamp(self.aspace.q_lo, self.aspace.q_hi)
                q[t] = qt
                W = wealth_step(W, qt, self.aspace.contract_size, *moves[t])
            return q, W
        q_star, WT = construct(True)
        return q_star, construct(False)[0], curves, WT

    @staticmethod
    def _carry_variance_solve(mu, Sig, c, Q, scale):
        """argmin_w [Q²·wᵀΣw + 2Q·wᵀc] / (2·scale) − Q·wᵀμ  s.t. Σw = 1, w ≥ 0.

        The carry/tracking allocator for ONE step: Σ and c are the covariance of the legs'
        remaining-horizon P&L (per contract, $) with each other and with the liability's, μ
        their means, Q the signed reference net (short < 0), and the variance/mean exchange rate
        ½/scale at the frame scale `utility_scale / aversion`. Active-set over ≤ n legs: solve
        the equality-constrained stationarity, drop the most negative weight, repeat."""
        idx = list(range(int(mu.shape[0])))
        while True:
            S = Sig[idx][:, idx].double()
            S = S + 1.0e-8 * S.diagonal().mean().clamp_min(1e-12) * torch.eye(
                len(idx), dtype=S.dtype, device=S.device)
            Si = torch.linalg.inv(S)
            w = (scale / Q ** 2) * (Si @ (Q * mu[idx].double()
                                          - (Q / scale) * c[idx].double()))
            k = (scale / Q ** 2) * (Si @ torch.ones(len(idx), dtype=S.dtype, device=S.device))
            w = w - ((w.sum() - 1.0) / k.sum()) * k
            if bool((w >= -1e-9).all()) or len(idx) == 1:
                out = torch.zeros_like(mu, dtype=S.dtype)
                out[torch.tensor(idx, device=mu.device)] = w.clamp_min(0.0)
                return (out / out.sum().clamp_min(1e-12)).to(mu.dtype)
            idx.pop(int(w.argmin()))

    def _carry_variance_weights(self):
        """Derive the per-step `Allocation_Weights` table from the outer sims: per decision
        step, each leg's remaining-horizon P&L per contract (marks freeze at expiry, so a dead
        leg's variance is ~0 and it drops out of the active set on its own), the liability's
        remaining P&L, and the step's capital — solved by `_carry_variance_solve` at the
        replication net. Knots are emitted only where the split MOVES."""
        knots, last = [], None
        for t in range(self.T_dec):
            dF, RL = self._outer_move(t, to=self.T_dec)                  # the remaining horizon's
            R = (self.aspace.contract_size * dF).T.contiguous()                 # (n_h, B) $ per contract
            X = R - R.mean(-1, keepdim=True)
            n = R.shape[-1] - 1
            Sig = (X @ X.transpose(0, 1)) / n
            c = (X * (RL - RL.mean())[None]).sum(-1) / n
            alive = Sig.diagonal() > 1.0e-4
            rep = self._replication_hedge(t)
            Qb = float(rep.sum())
            if not bool(alive.any()) or self._dust(rep):
                w = last if last is not None else tuple(
                    1.0 / self.aspace.n_hedge for _ in range(self.aspace.n_hedge))
            else:
                mu = R.mean(-1)
                big = 1.0e12 * Sig.diagonal().max().clamp_min(1.0)
                Sig = Sig + torch.diag(torch.where(alive, torch.zeros_like(mu), big))
                # the frame scale, which exists at bind; the LogWealth capital line is
                # measured on the bank this ladder builds, so it cannot be read here
                sol = self._carry_variance_solve(
                    mu, Sig, c, Qb, float(self.utility_scale) / self.aversion)
                w = tuple(round(float(x), 9) for x in sol)
            if w != last:
                knots.append((t, w))
                last = w
        return tuple(knots)

    def _check_action_universe(self, ck, src):
        """Under Allocation_Mode='Carry_Variance' a LOAD restores the checkpoint's own derived
        table before the universe comparison — re-deriving from the eval world would refuse
        every load on Monte-Carlo dust. The first member installs; later members must match it
        (a mixed-universe ensemble stays refused by the parent's check)."""
        if (self.aspace.mode == "carry_variance" and self.aspace.weights is None
                and ck.get("allocation_weights")):
            self.aspace.install_weights(ck["allocation_weights"])
        return super()._check_action_universe(ck, src)

    def _build_bank(self, gen):
        """The operating region IS the constructed policy: the bank rolls the forward pass's
        own positions with `DiffV2_Bank_Noise_Frac` exploration around them. The parent's
        uniform p-axis coverage is deliberately absent — concentrating the fit on the
        constructed policy's neighbourhood is the point of the forward pass. Under
        `DiffV2_Bank_Along_Ladder` the exploration is drawn on the net, as `DiffV2_Bank_Noise_Frac`
        of the ladder band's width, clamped into the band, and every bank book is that net's
        `rung` on the legs live at the step (`Bundle.marked`)."""
        if (self.aspace.mode == "carry_variance" and self.aspace.weights is None
                and not self.cfg["diffv2_load_value_fn"]):
            sched = self._carry_variance_weights()
            self.aspace.install_weights(sched)
            logging.info(
                "DiffSolverV2 Carry_Variance allocator: %d knot(s); w(t=0)=%s w(mid)=%s",
                len(sched), sched[0][1] if sched else None,
                sched[len(sched) // 2][1] if sched else None)
        live = None
        if self.bank_ladder:
            live = {t: self.bundle.marked(t, self.aspace.hedges) for t in self.steps}
        # the floored seed keeps every training book above the floor, so a deep-under-water state
        # would never appear and the net would extrapolate there. Half the paths therefore roll the
        # clairvoyant seed and half the UNDEFENDED delta base.
        q_star, q_base, curves, WT = self._constructed_policy(live=live)
        half = self.B_outer // 2
        for t in self.steps:
            q_star[t] = torch.cat([q_star[t][:half], q_base[t][half:]], dim=0)
        self.phi_curves = [(bx.detach().cpu(), bp.detach().cpu()) for bx, bp in curves]
        phi0 = float(self._phi_apply(curves[0], self.liability_sim[0][:1])[0])
        logging.info(
            "DiffSolverV2 forward pass: aversion=%.2f d_0=%.3f q*_0 net=%.2f "
            "terminal book mean=%.4g frac<0=%.3f",
            self.aversion, phi0,
            float(q_star[0].sum(-1).mean()), float(WT.mean()), float((WT < 0).float().mean()))
        rng = (self.aspace.q_hi - self.aspace.q_lo)
        mask = self.aspace.active_mask

        def explore(t):
            if live is None:
                noise = self.noise_frac * rng * torch.randn(
                    self.B_outer, self.aspace.n_hedge, generator=gen, device=self.device)
                q = self.aspace.waterfill((q_star[t] + noise * mask).clamp(
                    self.aspace.q_lo, self.aspace.q_hi), *self.aspace.bounds(t)[2:])
            else:
                # the exploration rides the net and the book stays a rung, inside the band and so
                # inside the corridor
                _, legs, _, _, lo, hi = self.aspace.ladder(t, live[t])
                noise = self.noise_frac * (hi - lo) * torch.randn(
                    self.B_outer, 1, generator=gen, device=self.device)
                net = (q_star[t][:, legs].sum(-1, keepdim=True) + noise).clamp(lo, hi)
                q = self.aspace.rung(t, net, live[t])
            return q
        W_list, q_list = self._roll_bank(explore)
        if self.log_ratio:
            logging.info(
                "DiffSolverV2 bank floor (LogWealth domain): frac(W <= floor %.4g) = %.4f "
                "min W (incl terminal) = %.4g", self.w_floor,
                float((torch.stack(W_list) <= self.w_floor).float().mean()), self._bank_worst)
        return W_list, q_list
