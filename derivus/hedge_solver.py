"""Differential-ML dynamic-hedging solver (Execution_Mode='solve_hedge').

`DiffSolver` is the production solver: a backward-DP / differential-ML value function
fit by the Huge–Savine twin loss (value + AAD pathwise-gradient), consuming the simulated
scenario bundle and forking inner MC on demand via `Bundle.inner_mc` / `Bundle.inner_mc_grad`
(attached by `HedgeMonteCarlo.execute`). `HindsightDpSolver` (clairvoyant oracle, the
upper-bound track) and `run_textbook_benchmark` (averaging / min-var lower-bound track)
are kept as benchmarks. `StreamingSolve` is the driver — one bundle per simulation batch,
warmup/step/finish — and `assemble_hedge_result` builds the comparison table + acceptance
ladder.
"""

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
    _loss_side_curvature, _utility_local_curvature, _utility_scale, _utility_wrap_signed,
    wealth_step,
)
from .hedge_runtime import per_contract_kappa, initial_q_from_runtime, turnover_charge

# Bumped whenever the fitted-value-function on-disk/artifact contract changes shape. Stamped
# into every artifact so a loader can tell which solver produced a checkpoint.
SOLVER_VERSION = "diffsolver/2026-08"


def _packed(values):
    """A whole per-rung curve as ONE `%.6g` CSV cell, space separated. Kept in a cell rather than
    spread over columns because a `Total_Position_Schedule` narrows the ladder per step, and a
    per-rung column layout would ragged the header the dump writes exactly once."""
    return " ".join("%.6g" % float(v) for v in values)


@dataclass
class SolverResult:
    """High-level result of a hedge solver. The DP solvers (HindsightDpSolver, DiffSolver)
    produce per-(t, outer-path) grids of `actions` and `values`."""
    solver_name: str
    actions: Any
    values: Any
    terminal_pnl: Optional[Any] = None
    terminal_utility: Optional[Any] = None
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
                "n_star": runs[0].diagnostics.get("n_star_0")
                          or runs[0].diagnostics.get("n_star")}


class HedgeActionSpace:
    """The solver-owned ACTION UNIVERSE + friction access, built ONCE from (runtime, device)
    and shared by every track (DiffSolver, HindsightDpSolver, run_textbook_benchmark, and
    the stepper rollout) so they optimize over exactly the same positions and price the same
    frictions. One object replaces the old scattered `build_action_grid` (varied ALL hedges,
    letting the benchmarks trade axes an `Active_Hedge_Indices` run had pinned off) + the
    private `DiffSolver._action_grid` + the free `_per_contract_kappa`/`_axis_levels`.

    Surface:
      * `axis_levels()` / `grid()` — the MASK-AWARE target-position grid: the active hedge
        axes span `[min, max]` at `levels` points, inactive axes are pinned to a single 0,
        and rows over the total-position cap are dropped (identical to the old private grid).
      * `allocation_grid(t)` — the FACTORED alternative under `Allocation_Weights`: one ladder over
        the signed NET cover, each rung split across the legs by the schedule. `grid_at` dispatches
        on which universe is configured; everything downstream consumes the rows either way.
      * `kappa(tradables_sim, t)` — the per-hedge turnover kappa at sim-grid `t` (each
        instrument's mean mark), off the single `hedge_runtime.per_contract_kappa` rule.
      * `initial_q(batch, device)` — the opening book `q0` (see `initial_q_from_runtime`).
      * `turnover_cost(dq, kappa)` / `schedule_key(schedule)` — the friction charge on a
        reposition (L1, or net/paired under a calendar spread) and the comparison-stable corridor
        stamp, both owned here because the action universe owns kappa and the corridor."""

    documentation = ('Solver', [
        'The action universe, built once from the runtime and shared by every track — the solver,',
        'the benchmarks and the stepper rollout — so they all optimize over the same positions and',
        'price the same frictions.',
        '',
        '- `grid()` / `axis_levels()` — the target-position grid. Active hedge axes span',
        '  `[Min_Position, Max_Position]` at `Training_Action_Grid_Levels_Per_Axis` points,',
        '  inactive axes pin to a single 0, and rows breaching the total-position cap are dropped.',
        '- `allocation_grid(...)` — under `Evaluator.Allocation_Weights`, the FACTORED universe',
        '  instead: the levels count rungs on the signed NET cover and the schedule splits each',
        '  rung across the legs, so the search is over how much cover to carry, not over its',
        '  composition.',
        '- `kappa(...)` — the per-hedge turnover cost at a step, off the single',
        '  `per_contract_kappa` rule the environment and the diagnostic CSVs also use.',
        '- `initial_q(...)` — the opening book.',
        '- `turnover_cost(...)` — the charge on a reposition: L1, or (under',
        '  `Evaluator.Calendar_Spread_Bps`) an outright charge on the net change plus a cheaper',
        '  spread charge on the matched half.',
        '',
        'One object owning all of this is what makes a benchmark comparable to the solver: they',
        'cannot disagree about which positions are reachable or what a trade costs.',
    ])

    def __init__(self, runtime, device, vol_sim=None):
        self.runtime = runtime
        self.device = device
        # Sim-grid per-step annualized vol series (or None) for the state-dependent bid/offer
        # spread — threaded through `kappa` into `per_contract_kappa`'s Vol_Scale.
        self.vol_sim = vol_sim
        self.hedges = list(runtime["names"]["hedges"])
        self.n_hedge = len(self.hedges)
        acc = runtime["accounting"]
        limits = acc["position_limits"]
        self.q_lo = torch.tensor(
            [float(limits.get(r, {}).get("min_position", 0.0)) for r in self.hedges],
            device=device)
        self.q_hi = torch.tensor(
            [float(limits.get(r, {}).get("max_position", 0.0)) for r in self.hedges],
            device=device)
        self.contract_size = torch.tensor(
            [float(runtime["tradables"][r]["contract_size"]) for r in self.hedges],
            device=device)
        self.total_abs_limit = float(acc["total_position_abs_limit"])
        # Optional per-decision-step corridor on the SIGNED total position Σq_i (sorted
        # (step, min_total, max_total) knots or None). `grid_at(t)` filters the base grid to it.
        self.schedule = acc.get("total_position_schedule")
        # Optional per-leg rate limit on |Δq| per decision step (0 = off). EXECUTION policy
        # only: the label path passes rate_limit=False, so a standing book handed to the
        # training argmax (for the churn charge) never arms the band.
        self.max_trade = float(acc.get("max_trade_per_step", 0.0))
        # Optional no-trade band in standard errors of the PAIRED inner-draw difference
        # against the standing book (0 = off). Same execution-only path as `max_trade`.
        self.deadband_sigma = float(acc.get("decision_deadband_sigma", 0.0))
        # Optional per-decision-step split of the NET cover across the legs (sorted
        # (step, weights) knots or None). Present ⇒ `grid_at(t)` is the FACTORED Q-ladder.
        self.weights = acc.get("allocation_weights")
        self.mode = str(acc.get("allocation_mode", "exposure"))
        # Half-spread bps of the matched (calendar) leg of a reposition, or None ⇒ every
        # contract traded pays the outright rate.
        self.calendar_bps = acc.get("calendar_spread_bps")
        self._grid_cache = None
        self._ladder_cache = {}
        solver_cfg = runtime["solver"]
        self.levels = int(solver_cfg["training_action_grid_levels_per_axis"])
        active = solver_cfg.get("active_hedge_indices")
        self.active = (list(range(self.n_hedge)) if active is None
                       else [int(i) for i in active])
        self.n_active = len(self.active)
        self.active_mask = torch.zeros(self.n_hedge, device=device)
        self.active_mask[self.active] = 1.0

    def axis_levels(self):
        """Per-hedge 1-D level values: `linspace(lo, hi, levels)` on ACTIVE axes, a single 0
        on inactive axes (pinned). The product of these is the action grid."""
        return [torch.linspace(float(self.q_lo[i]), float(self.q_hi[i]), self.levels,
                               device=self.device)
                if i in self.active else torch.zeros(1, device=self.device)
                for i in range(self.n_hedge)]

    def grid(self):
        """Mask-aware target-position grid `(n_actions, n_hedge)`: inactive axes pinned to 0,
        rows over the total-position cap dropped. Cached (deterministic) — `grid_at(t)` reslices
        this base grid per decision step without rebuilding the meshgrid."""
        if self._grid_cache is None:
            mesh = torch.meshgrid(*self.axis_levels(), indexing="ij")
            grid = torch.stack([m.reshape(-1) for m in mesh], dim=-1)
            if self.total_abs_limit > 0.0:
                grid = grid[grid.abs().sum(-1) <= self.total_abs_limit + 1e-9]
            if grid.shape[0] == 0:
                raise ValueError(
                    "action grid empty — Position_Limits infeasible under Total_Position_Abs_Limit")
            self._grid_cache = grid
        return self._grid_cache

    def _corridor_at(self, t):
        """`(min_total, max_total)` at decision step t — the rightmost `Total_Position_Schedule`
        knot with `Step <= t` (piecewise-constant; clamps to the first knot for t before it)."""
        lo, hi = self.schedule[0][1], self.schedule[0][2]
        for step, mn, mx in self.schedule:
            if step > t:
                break
            lo, hi = mn, mx
        return lo, hi

    def _weights_at(self, t, live=None):
        """The `Allocation_Weights` split in force at decision step t — the rightmost knot with
        `Step <= t` (piecewise-constant; clamps to the first knot for t before it) as an
        `(n_hedge,)` tensor, intersected with the legs that can actually carry a contract and
        renormalized over what is left: the INACTIVE legs (`Active_Hedge_Indices`) and, when the
        caller passes one, the EXPIRED legs (`live`).

        The live intersection has to happen HERE, before the ladder is built, not as a filter
        afterwards: a knot's weight on a dead leg is cover the book cannot realize, so leaving it
        in makes every rung advertise a net it masks away from (measured: 255 contracts parked on
        an expired leg) and makes a corridor the live legs can satisfy look infeasible."""
        w = self.weights[0][1]
        for step, knot in self.weights:
            if step > t:
                break
            w = knot
        w = torch.tensor(w, device=self.device) * self.active_mask
        if live is not None:
            w = w * live
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
        when the row is supposed to be `-10` — and a ladder whose rows miss the net they advertise
        is not a ladder over the net. Flooring is toward −∞ and the leftover is therefore
        non-negative, so a short book is apportioned by the same expression as a long one; the
        leftover is also strictly smaller than the number of legs carrying a remainder, so a leg
        already sitting on an integer box edge never receives a unit and no row leaves its box."""
        base = torch.floor(x)
        need = (target.round() - base.sum(-1, keepdim=True)).round()             # (L,1) whole units
        order = torch.argsort(x - base, dim=-1, descending=True, stable=True)
        rank = torch.empty_like(order)
        rank.scatter_(-1, order, torch.arange(x.shape[-1], device=x.device).expand_as(order))
        return base + (rank < need).to(x.dtype)

    def allocation_grid(self, t, live=None):
        """The FACTORED action universe at decision step t: a Q-LADDER over the signed NET cover,
        each rung split across the legs by the `Allocation_Weights` knot in force. The solver then
        chooses one number — how much cover to carry — and a deterministic allocator decides where
        it sits.

        Why factor. The Cartesian grid's quantum on the NET is the per-axis step times the number
        of legs, so a 3-leg 9-level book moves the net in jumps far larger than any realistic
        friction can smooth: a frictional Bellman over it changes nothing, and the composition
        dimension it spends its rows on goes unused. `Training_Action_Grid_Levels_Per_Axis` counts
        rungs on the TOTAL here, and 51 of them cost a fraction of 9³ rows while quantizing the
        quantity the policy is actually choosing.

        Construction, all of it on the legs that can carry a contract — `_weights_at(t, live)`
        drops the inactive and the expired ones and renormalizes, and every bound below is summed
        over what survives:
          * the band is `net_bounds(t)` (cap ∩ corridor ∩ boxes) intersected with those legs' own
            boxes, then snapped INWARD to whole contracts — the apportionment deals in whole
            contracts, so a fractional band would emit rungs that round outside their own mandate;
          * `Q·w` is clamped into the per-leg boxes and water-filled back onto Q over the SAME leg
            set, so neither an uneven schedule nor the water-fill can put cover on a leg the
            schedule zeroed;
          * `_largest_remainder` rounds to whole contracts summing to Q, and duplicate rungs (a
            band narrower than `levels` contracts) collapse.
        Cached per (t, live). The corridor filter downstream in `grid_at` is then provably a
        no-op — which is the point: the rungs a caller is offered are rungs it can realize."""
        key = (t, None if live is None else tuple(live.tolist()))
        if key not in self._ladder_cache:
            weights = self._weights_at(t, live)
            usable = (weights > 0).to(self.q_lo.dtype)
            box_lo, box_hi = torch.ceil(self.q_lo) * usable, torch.floor(self.q_hi) * usable
            lo, hi = self.net_bounds(t)
            lo = math.ceil(max(lo, float(box_lo.sum())) - 1e-9)
            hi = math.floor(min(hi, float(box_hi.sum())) + 1e-9)
            if lo > hi:
                raise ValueError(
                    f"action ladder empty at step {t} — no whole-contract net is reachable on the "
                    f"weighted live legs {[self.hedges[i] for i in usable.nonzero().flatten().tolist()]} "
                    f"inside net_bounds {self.net_bounds(t)}")
            if hi - lo > self.levels - 1:
                raise ValueError(
                    f"Allocation_Weights ladder at step {t} spans {hi - lo} contracts over "
                    f"{self.levels} rungs — a quantum of {(hi - lo) / (self.levels - 1):.2f} "
                    f"contracts. Under a weight schedule "
                    f"Training_Action_Grid_Levels_Per_Axis counts rungs on the TOTAL, and a "
                    f"quantum above 1 rebuilds the coarse net grid the schedule exists to "
                    f"replace; raise it to at least {hi - lo + 1}.")
            net = torch.linspace(lo, hi, self.levels, device=self.device).unsqueeze(-1)  # (L,1)
            raw = (net * weights).clamp(box_lo, box_hi)
            rows = self._largest_remainder(
                self.waterfill(raw, net, net, usable.nonzero().flatten().tolist()), net)
            if self.total_abs_limit > 0.0:
                rows = rows[rows.abs().sum(-1) <= self.total_abs_limit + 1e-9]
            self._ladder_cache[key] = rows.unique(dim=0)
        return self._ladder_cache[key]

    def grid_at(self, t, live=None):
        """Action grid at decision step t: the base `grid()` — or the factored `allocation_grid(t)`
        when `Allocation_Weights` is configured — filtered to rows whose SIGNED total position lies
        in the `Total_Position_Schedule` corridor at t. No schedule → that grid unchanged
        (bit-identical to today). The single per-t filter site every track shares (DiffSolver
        argmax, hindsight, textbook). Empty after filtering ⇒ infeasible corridor at t, failed
        loud.

        The corridor bounds the REALIZED signed total — the position that survives expiry masking
        (the argmax callers apply `q = q * live` afterwards). Passing the step-t `live` leg mask
        filters on Σ(q_i·live_i), so a corridor-satisfying short can't be parked on an expired leg
        (a dF=0 wealth-neutral tie) only to be masked to 0 and silently under-hedge. Absent live ⇒
        Σq_i over all legs (entry step / static diagnostics, where nothing has expired)."""
        grid = self.grid() if self.weights is None else self.allocation_grid(t, live)
        if self.schedule is None:
            return grid
        lo, hi = self._corridor_at(t)
        tot = (grid * live).sum(-1) if live is not None else grid.sum(-1)
        grid_t = grid[(tot >= lo - 1e-9) & (tot <= hi + 1e-9)]
        if grid_t.shape[0] == 0:
            raise ValueError(
                f"action grid empty at step {t} — Total_Position_Schedule corridor "
                f"[{lo}, {hi}] infeasible on live legs (grid Σq·live spans "
                f"[{float(tot.min())}, {float(tot.max())}])")
        return grid_t

    def project_to_corridor(self, q, t):
        """Project the SIGNED total of the ACTIVE legs of `q` `(B, n_hedge)` into the
        `Total_Position_Schedule` corridor at decision step t, keeping every leg inside its own
        `[Min,Max]` via headroom water-filling: shift the active legs by the corridor deficit
        `d = clamp(Σq, lo, hi) − Σq`, distributed in proportion to each leg's remaining room in
        d's direction, so `Σ_active` lands exactly on the nearest corridor edge and no leg leaves
        its box. Feasible whenever the corridor is grid-feasible (headroom ≥ |d|, the same
        feasibility `grid_at` fails loud on). No schedule → returns `q` unchanged (bit-identical).

        Unlike `grid_at` (which FILTERS a discrete action universe to the corridor for the argmax),
        this CONTINUOUSLY nudges an exploration/benchmark position — the exploration bank rolls
        continuous q around the min-var hedge, so a filter has nothing to select from. Shared by
        the bank (so the fitted value trains on IN-corridor wealth states, not unreachable ones)
        and the verdict/stepper textbook (a fair in-corridor min-var comparison)."""
        if self.schedule is None:
            return q
        return self.waterfill(q, *self._corridor_at(t))

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

    def net_bounds(self, t):
        """The feasible band of the SIGNED active total at decision step t — the per-leg boxes
        intersected with `Total_Position_Abs_Limit` and with the `Total_Position_Schedule`
        corridor. Exactly the interval `grid_at(t)` can realize, so it is the support the position
        state p = Σq/Q_max is queried over and therefore the one the bank must explore."""
        lo = float(self.q_lo[self.active].sum())
        hi = float(self.q_hi[self.active].sum())
        if self.total_abs_limit > 0.0:
            lo, hi = max(lo, -self.total_abs_limit), min(hi, self.total_abs_limit)
        if self.schedule is not None:
            c_lo, c_hi = self._corridor_at(t)
            lo, hi = max(lo, c_lo), min(hi, c_hi)
        return lo, hi

    def kappa(self, tradables_sim, t_index, calendar=False):
        """Per-hedge kappa `(n_hedge,)` at sim-grid `t_index` — each instrument's mean mark
        through `hedge_runtime.per_contract_kappa` (the single turnover-cost rule). The step's
        scalar vol (when a Vol_Scale spread is configured) drives per_contract_kappa's Vol_Scale.
        `calendar=True` prices the MATCHED leg at `Calendar_Spread_Bps` instead of the outright
        half-spread — the rate `turnover_cost` charges matched volume at."""
        vol = None if self.vol_sim is None else self.vol_sim[int(t_index)]
        return torch.stack(
            [per_contract_kappa(self.runtime, tradables_sim[r][t_index].mean(), r, vol, calendar)
             for r in self.hedges])

    def calendar_kappa(self, tradables_sim, t_index):
        """The matched-leg kappa at `t_index`, or None when no `Calendar_Spread_Bps` is
        configured. `None` IS the switch, and this is the ONE place that reads it: every charge —
        the argmax's, the fitted target's, both benchmark tracks' and the verdict's — takes its
        `kappa_cal` from here, so they cannot end up pricing a roll three different ways."""
        return (self.kappa(tradables_sim, t_index, calendar=True)
                if self.calendar_bps is not None else None)

    def universe_size(self):
        """Row count of the action universe a diagnostic should report: the Cartesian grid, or the
        entry-step LADDER under `Allocation_Weights`. Never the meshgrid under a schedule — at 5
        legs and 51 levels that is ~345M rows built and discarded for a 51-row search."""
        return int((self.grid_at(0) if self.weights is not None else self.grid()).shape[0])

    def static_grid(self):
        """The universe a CONSTANT hold is chosen from (`run_textbook_benchmark`). Cartesian: the
        entry-step grid, as before. Under `Allocation_Weights`: every knot's ladder, deduped and
        held to the ENTRY corridor — every composition the schedule ever offers, any one of which
        a hold could have carried from day one.

        Not just the entry knot, which is what `grid_at(0)` would give: the DP recomposes at every
        knot and a constant hold cannot, so pinning the benchmark to the entry ray would put an
        action-space difference inside the DP-vs-textbook gap, which is the campaign's headline
        number. The hold still cannot TRACK the schedule — that is the honest asymmetry — but it
        may at least choose which composition to sit in."""
        if self.weights is None:
            return self.grid_at(0)
        rows = torch.cat([self.allocation_grid(step) for step, _ in self.weights]).unique(dim=0)
        if self.schedule is None:
            return rows
        lo, hi = self._corridor_at(0)
        return rows[(rows.sum(-1) >= lo - 1e-9) & (rows.sum(-1) <= hi + 1e-9)]

    def initial_q(self, batch, device):
        """The opening book `q0` `(batch, n_hedge)` (hedge legs only, `names['hedges']`
        order) from the normalized `Portfolio_State` positions."""
        return initial_q_from_runtime(self.runtime, batch, device)

    def turnover_cost(self, delta_contracts, kappa, kappa_cal=None):
        """The solver-side name for `hedge_runtime.turnover_charge`, the single reposition-cost
        rule the realized accounting reads too: the L1 sum `Σ_i |Δ_i|·κ_i`, or — with the matched
        -leg `kappa_cal` from `calendar_kappa` — a greedy sweep over consecutive maturities that
        charges matched volume one calendar crossing and every unmatched lot its own leg's kappa.
        Hard (no smoothing): the action search has no gradients on the action.

        This is what lets a policy roll. L1 charges a roll as two outright crossings, so a book
        that should move down the curve is held into expiry instead; and the SAME rule has to
        reach `hedge_bundle._roll_rebate` — it does, both call `turnover_charge` — or the policy
        is optimized against a friction the book never pays."""
        return turnover_charge(delta_contracts, kappa, kappa_cal)

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
    """Static-hedge reference: the single best CONSTANT position, held over the whole horizon
    with no rebalancing, evaluated FRICTIONLESS on the realized outer paths (the shared DP
    objective — greedy/hindsight are frictionless too). A valid lower bound for the dynamic
    DP: dynamic rebalancing can only add value. No inner MC, no V̂. Static hold telescopes
    the per-step P&L to `position · (F_T − F_0)`. Uses the shared `HedgeActionSpace` so it
    respects `Active_Hedge_Indices` (inactive axes pinned to 0) exactly like the greedy grid.

    The turnover a real execution of this constant hold would pay — entry from the OPENING
    book `q0` (not from flat) + terminal unwind at the PRE-SETTLEMENT row — is a shared-kappa
    net-of-cost DIAGNOSTIC (`turnover_cost_mean` / `v0_mean_net`), never charged against the V_0
    track. One unwind row across all three tracks, so their `*_net` columns compare.

    The corridor filter is the ENTRY-step one: the hold is chosen ONCE at entry, and a per-t
    corridor is a dynamic constraint a constant hold cannot track, so the entry-step corridor is
    the single well-defined filter (no schedule ⇒ the base grid, unchanged). Keeps textbook a
    within-entry-mandate static lower bound and shares the one filter site.

    Under `Allocation_Weights` the universe is `aspace.static_grid()` — every knot's composition,
    not just the entry knot's. The DP recomposes as the schedule moves and a constant hold cannot,
    so pinning the benchmark to the entry ray would put an action-space difference inside the
    DP-vs-textbook gap, which is the number the campaign reads. The hold still cannot TRACK the
    schedule; it may only choose which composition to sit in for the whole horizon."""
    F, L_T, t_outer = bundle.realized_paths(runtime)
    device = F.device
    acc = runtime["accounting"]
    aspace = HedgeActionSpace(runtime, device, bundle.vol_sim)
    tradables_sim = bundle.tradables_sim
    # Entry-step corridor only — a constant hold can't track a per-t one (see docstring).
    grid = aspace.static_grid()                                        # (n_actions, n_hedge)
    b_outer = F.shape[-1]
    q0 = aspace.initial_q(b_outer, device)                            # (B, n_hedge) opening book

    total_move = F[:, -1, :] - F[:, 0, :]                              # (n_h, B) telescoped
    g_t = torch.einsum("ai,ib->ab", grid * aspace.contract_size, total_move)  # (n_actions, B)
    u = _utility_wrap_signed(L_T.unsqueeze(0) + g_t, runtime)         # FRICTIONLESS objective
    obj = u.mean(dim=-1)                                               # (n_actions,)
    best = int(obj.argmax())
    n_star = grid[best]                                                # (n_hedge,)
    # Net-of-cost diagnostic (shared kappa): entry |n_star − q0| + terminal unwind |0 − n_star|.
    kappa0 = aspace.kappa(tradables_sim, 0)
    cost = aspace.turnover_cost(n_star.unsqueeze(0) - q0,              # (B,) entry from q0
                                kappa0, aspace.calendar_kappa(tradables_sim, 0))
    if acc["force_flat_at_end"]:
        # Unwound at the PRE-SETTLEMENT row `last_live_mtm_index`, the one the solver's own
        # terminal charge uses: the last row is the post-settlement clean exit, where the deal
        # has already paid and there is nothing left to liquidate at. Outright, like every
        # terminal unwind: it takes the book to flat against the market, not leg against leg.
        kappa_T = aspace.kappa(tradables_sim, int(bundle.last_live_mtm_index))
        cost = cost + aspace.turnover_cost(n_star, kappa_T)            # unwind to flat (scalar)
    u_net = _utility_wrap_signed(L_T + g_t[best] - cost, runtime)
    return {"v0_mean": float(obj[best]), "v0_std": 0.0,
            "v0_mean_net": float(u_net.mean()),
            "turnover_cost_mean": float(cost.mean()),
            "n_star": n_star.detach().cpu().tolist(),
            "terminal_utility": u[best].detach().cpu()}


class HindsightDpSolver:
    """Clairvoyant upper-bound diagnostic, FRICTIONLESS (the shared DP objective). For each
    realized outer path it picks, at EVERY step independently, the grid position maximizing
    that step's realized P&L `q·cs·(F_{t+1}−F_t)` — perfect foresight + free repositioning.
    No inner MC, no V̂: the realized path is its own one-sample future.

    `u_signed` is monotone and the liability terminal `L_T(b)` is path-fixed, so maximizing
    `u_signed(W_T)` ≡ maximizing the additive cash `G_T`; with no turnover cost the per-step
    choices decouple, so the max-plus DP collapses to a per-step argmax. `mean_b V_0(b)` is
    an upper bound on any deployable (non-clairvoyant) policy's value — the reference the DP
    is measured against. The turnover a real execution of this bang-bang trajectory would pay
    (entry from the OPENING book, per-step repositioning, terminal unwind at the PRE-SETTLEMENT
    row — the one row all three tracks unwind at, so their `*_net` columns compare) is a
    shared-kappa net-of-cost DIAGNOSTIC, never charged against the V_0 track. Uses the shared
    `HedgeActionSpace`, so it respects `Active_Hedge_Indices` exactly like the greedy grid."""

    documentation = ('Solver', [
        'A clairvoyant UPPER BOUND, enabled by `Run_Hindsight_Diagnostic`. For each realized path',
        "it picks, at every step independently, the grid position maximizing that step's realized",
        'P&L — perfect foresight and free repositioning. No inner MC and no fitted value: the',
        'realized path is its own one-sample future.',
        '',
        'It is a bound, not a policy. With no turnover cost the per-step choices decouple, so the',
        'dynamic program collapses to a per-step argmax, and the mean value is an upper bound on',
        'any deployable policy — the reference the solver is measured against. The turnover a real',
        'execution of that bang-bang trajectory would pay is reported as a net-of-cost diagnostic',
        'and never charged against the bound.',
        '',
        'It shares `HedgeActionSpace`, so it respects `Active_Hedge_Indices` exactly as the greedy',
        'grid does — a benchmark cannot trade an axis the solver had pinned off.',
    ])

    def __init__(self, bundle, runtime):
        self.bundle = bundle
        self.runtime = runtime
        self.aspace = HedgeActionSpace(runtime, bundle.device, bundle.vol_sim)

    def solve(self):
        bundle, runtime, aspace = self.bundle, self.runtime, self.aspace
        acc = runtime["accounting"]

        F, L_T, t_outer = bundle.realized_paths(runtime)               # F (n_h,t_outer,B)
        device = F.device
        b_outer = F.shape[-1]
        tradables_sim = bundle.tradables_sim
        cs = aspace.contract_size

        # Frictionless clairvoyant: independent per-step argmax over the realized move, within the
        # per-t action universe `grid_at(t)` (base grid, corridor-filtered when a schedule is set).
        # The net-of-cost trajectory (from the opening book q0) is accumulated for the diagnostic.
        q_prev = aspace.initial_q(b_outer, device)                     # (B, n_h) opening book
        n_star_0 = None
        G = torch.zeros(b_outer, device=device)
        cost = torch.zeros(b_outer, device=device)
        for t in range(t_outer - 1):
            grid = aspace.grid_at(t)                                   # (n_actions_t, n_h) per-t
            dF = F[:, t + 1, :] - F[:, t, :]                           # (n_h, B)
            step_pnl = torch.einsum("ai,ib->ab", grid * cs, dF)        # (n_actions_t, B)
            best_pnl, best_idx = step_pnl.max(dim=0)                   # (B,), (B,)
            q_now = grid[best_idx]                                     # (B, n_h)
            cost = cost + aspace.turnover_cost(q_now - q_prev, aspace.kappa(tradables_sim, t),
                                               aspace.calendar_kappa(tradables_sim, t))
            G = G + best_pnl
            q_prev = q_now
            if t == 0:
                n_star_0 = q_now
        if acc["force_flat_at_end"]:
            # Pre-settlement row, as in `run_textbook_benchmark` and `DiffSolver._unwind_kappa`.
            cost = cost + aspace.turnover_cost(
                q_prev, aspace.kappa(tradables_sim, int(bundle.last_live_mtm_index)))
        v0 = _utility_wrap_signed(L_T + G, runtime)                    # (B,) FRICTIONLESS
        v0_net = _utility_wrap_signed(L_T + G - cost, runtime)         # (B,) net-of-cost diagnostic

        return SolverResult(
            solver_name="HindsightDpSolver",
            actions=n_star_0.detach().cpu(),
            values=v0.detach().cpu(),
            terminal_pnl=(L_T + G).detach().cpu(),
            terminal_utility=v0.detach().cpu(),
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
    """Zero-init residual head A_t(market, q, W). The continuation is C = u(W) + A, so a
    zero final layer makes C start exactly at the bounded utility anchor (A ≡ 0) — the
    toy's run-away guard. Under the successor chain only the TERMINAL net starts there;
    every earlier net starts at its fitted successor, and the trust-region clamp
    (`a_bounds`) is the brake that covers the inherited starts."""

    def __init__(self, in_dim, hidden=128):
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
    """Clean-room differential-ML hedging solver — rebuilt from the toy (`diffml_hedge_huber.py`
    via `diffsolver_v2.py`, validated BOUNDED at T=119) and wired to the OFFICIAL derivus
    framework. All dynamics come from the bundle's inner-MC closures
    (`Bundle.inner_mc` / `Bundle.inner_mc_grad`), which fork the simulator and price via
    `resolve_structure` (tradeables) + `resolve_hedge_structure` (liability) — no analytic
    transition, no Jacobian reconstruction; the framework prices everything.

    Spirit carried over verbatim from the toy:
      * C_t(market, q, W) = u(W) + A_t(market, q, W).  u = the bounded utility anchor
        (`_utility_wrap_signed`, the symlog/Huber/CARA transform normalised by c); A is a
        zero-init residual net — the only learned part. The bounded anchor is what keeps the
        backward recursion from running away (the old solver's 1e8 bug).
      * External argmax — the Bellman max lives OUTSIDE the fitted value (a discrete grid
        search over target positions, not inside the net).
      * Advantage decomposition — fit A = C − u(W) (value AND the wealth-channel pathwise
        gradient: a Huge–Savine twin loss), so the unbounded residual can't drift off u.
      * Operating-region bank — roll the OUTER paths forward exploring q AROUND the per-t
        replication (diagonal min-var) hedge, so wealth stays in-band and A stays on-support.
      * Position-free value (toy-faithful, the default) — V(market, W) does NOT take the
        position as input; with no turnover cost the held position is a freely-reset control,
        so it enters the value only through next-step wealth W1 = W + Σ q_i·cs_i·dF_i + dL. The
        n_hedge instruments live in the ACTION grid + the wealth step (the net learns there are
        3 via the routing of W1), not as a state coordinate. The action grid spans all hedges; a
        single-future-of-three test pins inactive axes to 0 via `Active_Hedge_Indices`
        (e.g. [2] ⇒ [0,0,-50]…[0,0,0]).
      * FRICTIONAL Bellman (`DiffV2_Position_State`) — the move the toy's own caveat names: once
        turnover costs money the incoming position IS a state variable, so the net gains the
        signed net book fraction p = Σq/Q_max as an input column and the repositioning charge is
        subtracted from the wealth that becomes the regressed TARGET, not only from the wealth
        that ranks the action. Cost-free-target pricing charges turnover as a one-day toll the
        value function never remembers, and a no-trade region cannot form out of a toll; charging
        it inside the target makes it compound down the recursion, which is what a hysteresis
        band is made of.
      * RUNNING WEALTH (`Objective.Reference_Mode`) — the reward becomes the DAY's wealth
        increment: Y = u(W1 − W_t) + E[C_{t+1}], with C_{T_dec} ≡ 0 and C_t = A_t alone (the u(W)
        anchor is a terminal-utility construct and retires with it). Exists because terminal
        wealth drifts far outside the utility's knee by mid-horizon, which leaves the objective
        affine and the argmax bang-bang; an increment is O(one day's move) on every day. It is the
        one dial that changes what the DP is maximising, so it is stamped on the checkpoint even
        though it moves no tensor shape.
      * CONDITIONAL-SIM SCALE (`Objective.Utility_Scale_Mode`) — the same pathology attacked from
        the scale instead: a per-decision-step knee c_t measured off the warmup batch (the
        cross-sectional dispersion of the flat-book wealth at t, floored), so x = (W − R)/c_t is a
        z-score on every day. It owns the LEVEL of c as well as its shape (`Utility_Scale_Explicit`
        is refused beside it), and because the shape params are dimensionless in c that RE-DOSES
        the aversion: under this mode `Huber_Aversion` is the aversion dial and it is dosed against
        the MEASURED scale, with the frame lock logging the resulting aversion per dollar.
        Independent of the mode above and REFUSED beside it.
      * WEALTH-FREE residual (`DiffV2_Wealth_Free_Value`) — A_t loses its W input column, so
        C_t = u(W) + A_t(market[, p]) and the only W-dependence left in the ranking is u's own.
        A_t is fitted against the market state (and the position column, when both switches are
        on) and the two compose: p stays, W goes. The candidate actions of one decision move W1
        across a span the action set induces, and a residual free to carry a wealth slope over
        that span adds a term the anchor's curvature has to fight; dropping the column makes the
        ranking's shape in W exactly the utility's, by construction rather than by fitting.

    Wealth convention: net wealth W_t = cumulative hedge P&L + the
    marked liability L_t; W_{t+1} = W_t + Σ_i q_i·cs_i·(F_{t+1,i} − F_{t,i}) + (L_{t+1} − L_t);
    terminal utility u(W_{T_dec}) with W_{T_dec} = total hedge P&L + L_T.

    INCREMENT 1 (this build): value bootstrap + the WEALTH-channel pathwise-gradient twin
    loss. W is the solver's own autograd leaf, so ∂Y_boot/∂W is exact with pure torch (no
    framework AAD needed). INCREMENT 2 adds the market-state (spot/state) gradient via
    `inner_mc_grad`'s `state_t_leaves` (privileged-layout leaf projection; FD-checked by
    `test_diffml_spot_grad_fd`). Turnover cost is ignored here (the toy has none) — a
    documented next-increment slot.
    """

    documentation = ('Solver', [
        'The production solver: a backward dynamic program whose per-step value is fitted by the',
        "Huge-Savine twin loss (value plus the pathwise wealth gradient), consuming the bundle's",
        'inner-MC closures. No analytic transition and no Jacobian reconstruction — the framework',
        'prices every forked scenario through the ordinary deal pricers.',
        '',
        '`C_t(market, W) = u(W) + A_t(market, W)`: a bounded utility anchor plus a zero-init',
        'residual net. Only `A` is learned, and fitting the RESIDUAL is what stops the backward',
        'recursion running away. The Bellman maximum lives OUTSIDE the fitted value — a discrete',
        'grid search over target positions — so the net never has to represent an argmax.',
        '',
        '### Shape of a run',
        '',
        'A solve is a stream (see `StreamingSolve`): `warmup` fits the first batch and LOCKS the',
        'frame, `step` continues on fresh paths, `finish` reports on the held-out batch. Peak fork',
        'memory is a function of `Batch_Size x Inner_Sub_Batch` alone, however long the stream is.',
        '',
        '### Knobs',
        '',
        '| key | meaning |',
        '| --- | --- |',
        '| `T_Min` | backward-sweep depth; 0 sweeps to the first decision |',
        '| `Training_Action_Grid_Levels_Per_Axis` | action-grid resolution per hedge axis |',
        '| `DiffV2_Fit_Iters`, `DiffV2_LR`, `DiffV2_Hidden` | per-t residual-net optimizer |',
        '| `DiffV2_Lambda_Grad` | weight on the pathwise-gradient half of the twin loss |',
        '| `DiffV2_Per_Column_Grad_Norm` | normalize greeks per input column rather than pooled |',
        '| `DiffV2_Bank_Noise_Frac` | exploration around the per-t replication hedge |',
        '| `DiffV2_Risk_Kappa` | score actions by `mean(C) - kappa * downside-semidev(C)` |',
        '| `DiffV2_Cost_Aware_Argmax` | charge the L1 repositioning cost at the argmax |',
        '| `DiffV2_Position_State` | the frictional Bellman: `p` as state, the charge in the target |',
        '| `DiffV2_Wealth_Free_Value` | drop `W` from the residual: the ranking bends only as `u` does |',
        '| `DiffV2_Decision_Curve_Dump` | CSV of the stepper rollout\'s full per-rung ranking curve |',
        '| `Active_Hedge_Indices` | which hedge axes vary; the rest pin to 0 |',
        '| `Multi_Seed_Count` | independent solvers on the same batches |',
        '',
        'Two dials live under `Objective` rather than `Solver`, because they change what is being',
        'maximised rather than how it is fitted. `Reference_Mode` = `Running_Wealth` makes the',
        "reward the DAY's wealth increment (the value is then a SUM of per-step utilities, and the",
        'reported `u_mean` is not comparable to a terminal-utility one), and',
        '`Utility_Scale_Mode` = `conditional_sim` gives every decision step its own knee, measured',
        'off the warmup batch and frozen. Each attacks the same defect — a terminal wealth that',
        'has drifted outside the knee leaves the ranking affine — and they refuse to run together.',
        '',
        '!!! warning "`conditional_sim` re-doses the aversion"',
        '    The measured schedule sets the LEVEL of `c`, not only its shape, and there is no',
        '    `Utility_Scale_Explicit` under it. The shape parameters are dimensionless in units of',
        '    `c`, so arming the mode moves the risk aversion by whatever factor the measured scale',
        '    differs from the retired one: under it, `Huber_Aversion` is the aversion dial and it',
        '    is dosed AGAINST THE MEASURED SCALE. The frame lock logs the resulting absolute risk',
        '    aversion per dollar so the re-dosing is visible rather than inferred.',
        '',
        '### Persistence',
        '',
        '`DiffV2_Save_Value_Fn` writes the fitted nets with their standardization frame, utility',
        'scale and trust region. `DiffV2_Load_Value_Fn` restores them and fits NOTHING — a frozen',
        'evaluation. The two are separate runs, and setting both raises rather than silently',
        'discarding a retrained net. Load accepts a LIST for an ensemble argmax: each member',
        'evaluated in its own frame, the continuations averaged before the maximum, which reduces',
        "the winner's curse. Mixing frame provenances inside one ensemble is refused.",
        '',
        '!!! warning "The frame is locked at warmup"',
        '    The utility scale, the market/wealth standardization stats and the per-t trust region',
        '    are computed on the first batch and frozen. Re-fitting them per batch would make each',
        "    batch's `C_t` a different function of different inputs, and the recursion would compose",
        '    mismatched frames. Later batches report how often their fitted targets fall outside the',
        '    frozen region instead.',
    ])

    # Bank-policy provenance: the name stamped into checkpoints and compared on load.
    SOLVER_OBJECT = "DiffSolver"

    def __init__(self, bundle, runtime):
        """Build the solver against `bundle`: the shared action universe, the config knobs, the
        bank RNG and the still-unlocked frame, then `_bind` to the bundle's sim views.

        The decision horizon `T_dec` is the last LIVE liability mark, not the last row. The time
        grid appends one post-settlement clean-exit row (the deal pays out — the platinum
        average-rate forward marks its realised payoff at T-1, then 0 at the payment date T), so
        the meaningful terminal is the bundle's `last_live_mtm_index` (the structural
        pre-settlement `[-2]`). Telescoping wealth THROUGH the settlement drop cancels the
        liability's settlement risk (no-hedge W_T≡0). Decisions run 0..T_dec-1; the terminal
        continuation marks at T_dec."""
        self.bundle = bundle
        self.runtime = runtime
        self.cfg = runtime["solver"]
        self.device = bundle.device
        # The shared action universe (mask-aware grid + per-contract kappa + opening book) —
        # the SAME object the benchmark tracks and the stepper rollout consume, so every track
        # optimizes over identical positions and prices identical frictions.
        self.aspace = HedgeActionSpace(runtime, self.device, bundle.vol_sim)
        self.hedges = self.aspace.hedges
        self.n_hedge = self.aspace.n_hedge
        self.contract_size = self.aspace.contract_size                # (n_hedge,)
        self.q_lo = self.aspace.q_lo
        self.q_hi = self.aspace.q_hi
        self.total_abs_limit = self.aspace.total_abs_limit
        # Active hedge axes: which instruments the action grid varies (rest pinned to 0).
        self.active = self.aspace.active
        self.n_active = self.aspace.n_active
        self.levels = self.aspace.levels
        self.chunk = int(self.cfg["training_action_chunk_size"])
        self.fit_iters = int(self.cfg.get("diffv2_fit_iters", 150))
        self.lr = float(self.cfg.get("diffv2_lr", 2.0e-3))
        self.noise_frac = float(self.cfg.get("diffv2_bank_noise_frac", 0.15))
        self.t_min = int(self.cfg.get("t_min", 0))
        self.use_adv = bool(self.cfg.get("use_advantage_decomp", True))
        # Downside-aware action selection (toy: RISK_KAPPA). 0 = plain E[C] argmax (bit-identical).
        self.risk_kappa = float(self.cfg.get("diffv2_risk_kappa", 0.0))
        # Cost-aware EXECUTION: the verdict rollout charges κ·|q − q_prev| at the argmax
        # (hysteresis); training/selection stay cost-free. Default off = bit-identical.
        self.cost_aware = bool(self.cfg.get("diffv2_cost_aware_argmax", False))
        # Quadratic churn charge lambda*sum((q-q_prev)^2) at every argmax that knows q_prev.
        self.churn_lambda = float(self.cfg.get("diffv2_churn_lambda", 0.0))
        # FRICTIONAL BELLMAN: the net book fraction p = Sum(q)/Q_max becomes a state coordinate of
        # the fitted value AND the repositioning charge enters the regressed target. 'No' = the
        # position-free, cost-free value (bit-identical). Q_max is the Evaluator's total-position
        # cap: it is the only declared scale p can be measured in, so an absent cap fails loud
        # rather than dividing the state by zero.
        self.position_state = bool(self.cfg.get("diffv2_position_state", False))
        # WEALTH-FREE residual: the value net's W input column is REMOVED, so A_t = A_t(market[, p])
        # and C_t's whole dependence on wealth is the utility anchor's. 'No' = the wealth-bearing
        # residual (bit-identical). Composes with the position state: p stays, W goes.
        self.wealth_free = bool(self.cfg.get("diffv2_wealth_free_value", False))
        if self.wealth_free and not self.position_state:
            raise ValueError(
                "Solver.DiffV2_Wealth_Free_Value='Yes' requires DiffV2_Position_State='Yes': an "
                "action cannot move the market, so without p the residual A(m') is one constant "
                "across every candidate and cancels from the argmax - the whole fit would be "
                "decision-irrelevant and the run a myopic one-step E[u(W1)] policy wearing a "
                "DP's verdict. Build the myopic baseline deliberately if it is wanted.")
        if self.wealth_free and self.risk_kappa > 0.0:
            raise ValueError(
                "Solver.DiffV2_Wealth_Free_Value='Yes' refuses DiffV2_Risk_Kappa > 0: the downside "
                "semideviation is nonlinear over inner draws, so the residual's dispersion leaks "
                "back into the ranking the switch exists to hand to the utility alone.")
        # The two OBJECTIVE dials, read off the normalized runtime (absence IS the default, so no
        # second default is published here). RUNNING WEALTH rewards the day's wealth INCREMENT
        # instead of terminal wealth, which makes C_t a sum of per-step rewards and retires the
        # u(W) anchor; SCHEDULED SCALE gives each decision step its own knee.
        obj = runtime.get("objective") or {}
        self.running_wealth = obj.get("reference_mode") == "running_wealth"
        # LogWealth: the per-step reward is the growth ratio log(W1/W0) — step-sum structure
        # (running_wealth is forced by the runtime reader), scale-free, and its domain is the
        # wealth floor the forward-constructed bank maintains. `w_floor` (a tiny dollar floor
        # for the ratio's denominator) binds with the bundle.
        self.log_ratio = obj.get("object") == "logwealth"
        self.w_floor = 1.0
        # The DP's aversion (the seed's floor is clairvoyant and CANNOT be enforced by a
        # causal backward pass — this dial is its causal proxy): it divides the capital line
        # in the LogWealth reward, so gamma > 1 puts less capital at risk and the objective
        # punishes drawdowns harder. 1.0 neutral; inert under the other objectives.
        self.aversion = float(self.cfg.get("diffv2_risk_aversion", 1.0))
        self.displacement = 0.0                # measured off the bank at warmup, or restored
        self.loaded_displacement = None
        self.scheduled_scale = obj.get("utility_scale_mode") == "conditional_sim"
        if self.running_wealth and self.scheduled_scale and not self.log_ratio:
            raise ValueError(
                "Objective.Reference_Mode='Running_Wealth' refuses "
                "Utility_Scale_Mode='conditional_sim': the increment already carries a per-step "
                "scale of its own (one day's move, not the whole horizon's), so a second per-step "
                "rescaling of it is a different objective again. Each is a separate experiment; "
                "run them one at a time. (LogWealth is the exception: the schedule is its "
                "CAPITAL line, not a rescaling.)")
        if self.running_wealth and float(self.cfg.get("diffv2_temporal_proximity", 0.0)) > 0.0:
            raise ValueError(
                "Objective.Reference_Mode='Running_Wealth' refuses DiffV2_Temporal_Proximity > 0: "
                "the proximity prior penalises the distance from A_t to its fitted SUCCESSOR, "
                "which is the right prior only while C_t ≈ C_{t+1}. Under a per-step reward the "
                "two differ by E[u(ΔW_t)] BY DEFINITION, so the regularizer pulls against exactly "
                "the offset the objective is made of. The successor WARM START is untouched and "
                "still valid — it is an initial point, not a penalty.")
        self.force_flat = bool(runtime["accounting"]["force_flat_at_end"])
        # Pure diagnostic: where `_rollout_on_stepper` writes its per-decision ranking-curve CSV.
        # '' = off, and the roll is then bit-identical to a build without the switch.
        self.curve_dump = str(self.cfg.get("diffv2_decision_curve_dump", "") or "")
        if self.position_state and not self.total_abs_limit > 0.0:
            raise ValueError(
                "Solver.DiffV2_Position_State='Yes' requires Evaluator.Total_Position_Abs_Limit > 0 "
                f"(it is {self.total_abs_limit}): the position state is p = Sum(q)/Q_max and the "
                "cap is the scale it is measured in")
        # Early termination for non-anchor fits: BLOCK-wise relative loss plateau (blocks of
        # 8 iters, floor of 2 blocks), so cold-Adam stall cannot read as convergence and the
        # loss sync is paid once per block, not per iteration. The terminal anchor always
        # runs the full budget. 0 disables.
        self.fit_tol = float(self.cfg.get("diffv2_fit_tol", 1e-3))
        self.prox = float(self.cfg.get("diffv2_temporal_proximity", 0.0))
        # Bank exploration RNG — deterministic, and PERSISTENT across batches so each batch
        # explores a fresh noise draw.
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(0)
        # Per-t Adam optimizers, created at the first fit of t and KEPT: a streaming step
        # continues the same moments on the same net rather than restarting them each batch.
        self._opts = {}
        # The locked frame. `utility_scale` is None until the first bind completes (below), which
        # is what marks that bind as the warmup one; `_bounds_frozen` turns `a_bounds` from
        # fitted-per-batch into frozen-and-reported (streaming only).
        self.utility_scale = None
        self._bounds_frozen = False
        self._breaches = []
        self._bind(bundle)
        # Effective terminal = the last LIVE liability mark, not the last row (see docstring).
        self.T_dec = int(bundle.last_live_mtm_index)
        if self.t_min >= self.T_dec:
            raise ValueError(
                f"Solver.T_Min={self.t_min} must be < decision horizon T_dec={self.T_dec}")
        self.utility_scale = float(bundle.utility_scale)      # locked here; re-asserted by _bind
        # ...and the per-step schedule with it: ONE frame, locked in one place. A streaming step
        # re-binds to a batch that measured its own and must not let it reach the reward.
        self.utility_scale_schedule = bundle.utility_scale_schedule
        if (self.scheduled_scale and self.utility_scale_schedule is None
                and not self.cfg.get("diffv2_load_value_fn")):
            raise ValueError(
                "Objective.Utility_Scale_Mode='conditional_sim' but this warmup batch measured no "
                "schedule: c_t is the cross-sectional dispersion of the flat-book wealth across "
                "OUTER PATHS, and a single-path batch has no cross-section. Raise Batch_Size (a "
                "one-path world can only be a frozen roll, which takes its schedule from the "
                "checkpoint).")

    def _bind(self, bundle):
        """Point the solver at a bundle: the history-stripped sim views the sweep indexes by `t`
        and the friction vol series the action space prices with. A frozen eval binds once at
        construction; a streaming step re-binds to each fresh batch — which is also where the
        LOCKED utility scale AND its per-step schedule are re-asserted, because every batch
        resolves its own frame at build time and letting a later one reach the runtime would
        silently rescale the objective (and with it every fitted C_t) mid-training."""
        self.bundle = bundle
        self.aspace.vol_sim = bundle.vol_sim
        self.tradables_sim, self.n_steps = bundle.tradables_sim, bundle.n_outer_steps
        self.liability_sim = bundle.liability_sim                     # (n_steps, B_outer)
        self.leg_volume = float(getattr(bundle, "total_leg_volume", 0.0) or 0.0)
        self.B_outer = int(self.liability_sim.shape[-1])
        if self.log_ratio:
            # A hair above zero on the book's own dollar scale: ratios against a launch below
            # this are ruin, and the reward there is capped at zero (absorbing, not a rebirth).
            self.w_floor = 1.0e-4 * float(self.liability_sim[0].mean().abs() + 1.0)
        if self.utility_scale is None:
            return                                                    # construction; nothing locked yet
        if int(bundle.last_live_mtm_index) != self.T_dec:
            raise ValueError(
                f"streaming batch has decision horizon T_dec={int(bundle.last_live_mtm_index)} "
                f"but the frame was locked at {self.T_dec} — every batch must share one time grid")
        logging.info("DiffSolver bound batch: B_outer=%d | frame utility_scale=%.6g "
                     "(this batch resolved %.6g)", self.B_outer, self.utility_scale,
                     float(bundle.utility_scale))
        self.runtime["objective"]["utility_scale"] = self.utility_scale
        self.runtime["objective"]["utility_scale_schedule"] = self.utility_scale_schedule

    def _config_hash(self):
        """sha1 of the stable-JSON solver cfg — the stamp identifying this TRAINING RECIPE.
        The persistence paths are excluded so the same recipe hashes identically regardless of
        where its checkpoint lives — and the decision-curve dump path with them, because a
        diagnostic that is switched on beside a run must not make it a different recipe."""
        stable = {k: v for k, v in self.cfg.items()
                  if k not in ("diffv2_save_value_fn", "diffv2_load_value_fn",
                               "diffv2_decision_curve_dump")}
        return hashlib.sha1(json.dumps(stable, sort_keys=True, default=str).encode()).hexdigest()

    # ---- utility anchor ------------------------------------------------------
    def _u(self, W, t=None):
        """The framework's normalised utility applied to `W` at decision step `t` — the bounded
        anchor u(W) of C_t = u(W) + A_t, or (`Reference_Mode='Running_Wealth'`, where the callers
        hand it a wealth INCREMENT) the day's reward. `t` selects the knee under a locked
        `Utility_Scale_Mode='conditional_sim'` schedule and is omitted for a TERMINAL read — the
        verdict's u(W_T), which the benchmark tracks take at the same scale."""
        return _utility_wrap_signed(W, self.runtime, t)

    def _member_anchor(self, W, t, schedule):
        """u(W) at ONE ensemble member's own step knee. A member's `utility_scale_schedule` is
        part of its frame exactly as its z-stats are, so its continuation is evaluated at the
        scale it was fitted against — the schedules of an ensemble are never averaged or
        otherwise reconciled (see `_restore_frame`).

        Under a step-sum objective (Running_Wealth / LogWealth) the anchor concept is retired:
        the continuation carries NO terminal utility — the same statement the single-net path
        makes with its zero base — so the member anchor is exactly zero rather than a wrap
        call the growth objective refuses by name."""
        if self.running_wealth:
            return torch.zeros_like(W)
        return _utility_wrap_signed(
            W, {"objective": dict(self.runtime["objective"],
                                  utility_scale_schedule=schedule)}, t)

    # ---- action grid (shared, mask-aware; inactive axes pinned to 0) ---------
    @property
    def grid_size(self):
        """Rows in the action universe this run searches — reported in the sweep log and in
        `diagnostics['action_grid_size']`. A property rather than a construction-time int because
        under `Allocation_Weights` the Cartesian meshgrid must never be built at all (51 levels on
        5 legs is ~345M rows for a 51-row ladder), and its size would be the wrong number anyway."""
        return self.aspace.universe_size()

    # ---- input standardization ----------------------------------------------
    def _in_dim(self, md):
        """Input width of a residual net: the `md` privileged market columns, plus W unless
        `DiffV2_Wealth_Free_Value` removes it, plus p under `DiffV2_Position_State`. The single
        source `warmup` sizes every net from and `_standardize` builds every row to, so a switch
        can only move the two together."""
        return md + int(not self.wealth_free) + int(self.position_state)

    def _standardize(self, market, W, p):
        """Standardized state for the residual net: (market | W), plus the signed net book
        fraction p = Sum(q)/Q_max as a trailing column under `DiffV2_Position_State`. `p is None`
        IS the switch at this level — position-free (toy-faithful: with no turnover cost the held
        position is a freely-reset control, not a state) gives the bit-identical two-block layout.

        Market and wealth are z-scored against the locked bank frame; p is NOT, because it is
        already a fraction of the cap on both sides of the seam — `grid()` drops any action over
        `Total_Position_Abs_Limit`, and `_build_bank` water-fills each sampled book onto
        `aspace.net_bounds(t)`. So p arrives in [-1, 1] by construction over a support the bank
        actually covers, and a z-frame would only add a batch-dependent rescaling to a coordinate
        that already has a natural one.

        `DiffV2_Wealth_Free_Value` drops the W column instead of adding one: the residual then
        reads market (and p) alone, and every W-dependence of C_t = u(W) + A_t lives in the anchor.
        The two switches compose — each owns one column — so the layout is
        (market | [W] | [p]) with the blocks kept in that order under every combination."""
        cols = [(market - self.m_mean) / self.m_std]
        if not self.wealth_free:
            cols.append(((W - self.w_mean) / self.w_std).unsqueeze(-1))
        return torch.cat(cols if p is None else cols + [p.unsqueeze(-1)], dim=-1)

    def _continuation(self, nets, market, W, t, p, chunk=400_000):
        """C_t = u(W) + A_t(market, W[, p]); terminal C_{T_dec} = u(W). Row-chunked net eval.
        Under `DiffV2_Wealth_Free_Value` the residual is A_t(market[, p]) — W still enters through
        the anchor, which is the whole of C_t's wealth dependence there; the caller's contract is
        unchanged either way, because `_standardize` owns the layout.

        RULING — under `Objective.Reference_Mode='Running_Wealth'` the anchor is IDENTICALLY ZERO
        and C_t = A_t alone, with the terminal C_{T_dec} ≡ 0. u(W) anchors the recursion because
        it is the terminal reward evaluated early; once the reward is the day's INCREMENT there is
        no terminal utility of the level to anchor on, and keeping u(W) would add a bounded
        function of a state that earns nothing to every value in the sweep. The per-step reward
        itself is added by the CALLER, which is the only place that holds both W1 and the wealth
        it moved from (`_decide`, `_fit_step`, `_score_actions`).

        Ensemble mode (list-of-checkpoints load): A = mean over members, each evaluated in
        its OWN standardization frame — the frame is part of the function, and under a
        `conditional_sim` schedule that frame includes the member's own step KNEE, so its
        continuation is the anchor at ITS c_t plus its own residual.
        A_t is CLAMPED to its fitted-target trust region (one range-width of headroom):
        off-support the zero-init MLP extrapolates freely — measured printing −5 where its
        targets spanned ±0.5 — and the argmax then chases the phantom direction, poisoning
        the t−1 bootstrap labels (the dead-net and corner-over-leverage basins at large B).
        Outside the clamp the gradient is zero, so the differential labels ignore phantom
        directions too."""
        base = torch.zeros_like(W) if self.running_wealth else self._u(W, t)
        ensemble = getattr(self, "_ensemble", None)
        if t >= self.T_dec and not ensemble:
            return base
        if ensemble:
            acc = torch.zeros_like(base)
            for m_nets, m_mean, m_std, w_mean, w_std, m_bounds, m_sched in ensemble:
                if m_sched is not None:
                    # The member's own step KNEE, as much a part of its frame as its z-stats: its
                    # continuation is read at the c it was fitted against, never at a reconciled
                    # one. Written as a correction to the shared anchor so a schedule-free
                    # ensemble stays arithmetically untouched.
                    acc = acc + (self._member_anchor(W, t, m_sched) - base)
                if t >= self.T_dec:
                    # The TERMINAL has no residual to add — and this correction is then the WHOLE
                    # of the member's continuation, which is what the last decision is ranked on.
                    # Returning `base` above (as the single-member path may) would have handed
                    # every member the first one's knee at exactly the step that decides.
                    continue
                cols = [(market - m_mean) / m_std]
                if not self.wealth_free:
                    cols.append(((W - w_mean) / w_std).unsqueeze(-1))
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
    def _unwind_kappa(self, t):
        """The terminal liquidation kappa when decision `t`'s successor IS the terminal mark and
        `Force_Flat_At_End` forces the book flat there — else None. Under `DiffV2_Position_State`
        the last decision has to price the unwind of whatever it leaves standing; without it the
        final step accumulates a position for free and the frictional recursion ends in a
        discontinuity. Read at `T_dec`, the pre-settlement row — the same row the benchmark
        tracks charge their own unwind at."""
        return (self.aspace.kappa(self.tradables_sim, self.T_dec)
                if self.position_state and self.force_flat and t + 1 >= self.T_dec else None)

    def _calendar_kappa(self, t):
        """The MATCHED-leg kappa at decision step t (None with no `Evaluator.Calendar_Spread_Bps`)
        — the rate matched volume pays inside `_reposition_charge`. A one-line hop to
        `aspace.calendar_kappa`, which the benchmark and verdict tracks read too, so the ranking
        in `_decide`, the regressed target in `_fit_step` and every net-of-cost diagnostic cannot
        price a roll three different ways."""
        return self.aspace.calendar_kappa(self.tradables_sim, t)

    def _reposition_charge(self, acts, q_prev, kappa, kappa_T, kappa_cal=None):
        """The frictional charge on moving `q_prev` to `acts` (broadcastable `(..., n_hedge)`): the
        turnover toll at `kappa`, the quadratic churn term (`DiffV2_Churn_Lambda`), and the
        terminal unwind of `acts` at `kappa_T` when `_unwind_kappa` says one applies.

        ONE expression, called from every seam: the wealth that RANKS a candidate in `_decide`,
        the wealth the INCUMBENT holds against it, and the wealth that becomes the chosen action's
        regressed TARGET in `_fit_step` all price the same friction. That equality is the whole of
        the frictional Bellman — a charge applied only at the ranking is a one-day toll the value
        function forgets. Callers pass LIVE-MASKED positions on both sides; a dead leg then has
        `dq = 0` and carries no phantom toll (see `_decide`).

        `kappa_cal` (from `_calendar_kappa`) makes the toll SPREAD-AWARE: the move's matched half
        pays the calendar rate rather than two outright crossings (see `turnover_cost`). The
        terminal unwind stays outright — it takes the whole book to flat against the market, not
        one leg against another, and a same-signed book has no matched half to begin with."""
        dq = acts - q_prev
        charge = self.churn_lambda * dq.pow(2).sum(-1)
        if kappa is not None:
            charge = charge + self.aspace.turnover_cost(dq, kappa, kappa_cal)
        if kappa_T is not None:
            charge = charge + self.aspace.turnover_cost(acts, kappa_T)
        return charge

    # ---- one-step wealth move ------------------------------------------------
    def _u_step(self, W1, W0, t):
        """The per-step reward both step-sum objectives share. Running_Wealth: u((W1−W0)/c_t),
        the scaled increment. LogWealth: log(W1/W0) — the day's growth ratio, zero iff the step
        is perfectly hedged, scale-free (no c_t anywhere). Below `eps` the log continues
        linearly so a breached path keeps gradient while paying an unbounded-in-level penalty,
        and a step LAUNCHED at or under the floor can only score <= 0: ruin is absorbing, not a
        rebirth against a clamped denominator."""
        if not self.log_ratio:
            return self._u(W1 - W0, t)
        # The CAPITAL line: wealth in the ratio is capital + MTM, capital = the batch
        # dispersion at the step (the conditional_sim schedule when locked, else the scalar
        # scale) — "wealth 1" is one unit of backing capital, ruin is losing it. This is the
        # user's batch-std normalization: (c_t + W1)/(c_t + W0) with both legs on the SAME
        # step's c, so the inner fork is measured against its own launch.
        c = self.displacement / self.aversion
        eps = 1.0e-3
        base = (c + W0).clamp_min(self.w_floor)
        r = (c + W1) / base
        u = torch.where(r > eps, r.clamp_min(eps).log(), math.log(eps) + (r - eps) / eps)
        # BOUNDED CONDITIONING: log's economic content lives within a few nats; beyond ruin
        # there are no useful gradations, only magnitudes that wreck the fit (unbounded, the
        # penalty reached -4e8, the trust region spanned ±1e10, and the nets learned nothing
        # but the cliff — rolling the same garbage corner across retrains). -20 nats is ruin
        # by e^20: the cliff stays a cliff, the labels stay a distribution.
        u = u.clamp_min(-20.0)
        return torch.where((c + W0) > self.w_floor, u, u.clamp_max(0.0))

    def _capital(self, t):
        """The DISPLACEMENT of the growth reward — a shifted-lognormal bias term (user-ruled):
        one constant per run, the scalar utility scale, which is by contract the TERMINAL
        dispersion read (the horizon-sigma under conditional_sim, a declared literal under
        Utility_Scale_Explicit). Deliberately t-independent: a per-step capital line changes
        the objective's curvature day by day and its measured-std machinery is where the
        start-scatter bug lived; the shift just keeps the log's argument positive and the
        curvature stable, exactly as a displacement does for a vol model. `t` is accepted for
        the call sites' sake and ignored."""
        return float(self.utility_scale)

    def _wealth_step(self, W, q, dF, dL):
        """W_{t+1} = W + Σ_i q_i·cs_i·dF_i + dL — the frictionless analytic wealth law, owned by
        `hedge_bundle.wealth_step` (the single source `futures_account_step` discretizes; the
        solver's bank/verdict/inner-labels all funnel through here). q (...,n_hedge); dF
        (...,n_hedge); dL (...). Expiry is composed by callers via the `live` mask on dF."""
        return wealth_step(W, q, self.contract_size, dF, dL)

    # ---- inner-MC one-step quantities at outer t -----------------------------
    def _inner_step(self, t):
        """Fork inner MC at t (resolve_structure / resolve_hedge_structure under the hood)
        and return the one-step move tensors the bootstrap needs:
          dF   (B_outer, B_inner, n_hedge)  per-instrument futures move t→t+1
          dL   (B_outer, B_inner)           liability mark change t→t+1
          m1   (B_outer, B_inner, market_dim) market state at t+1
        plus the bank-state market at t (B_outer, market_dim).

        EXPIRED-CONTRACT GUARD (the returned `live` mask): the framework returns inner F_t1=0 for
        a tradable that has expired before the fork's t+1, while the OUTER `tradables_sim` FREEZES
        at the last traded price. So a naive F_t1−F_t mints a spurious ~−F_t "move" on a dead
        contract — shorting it would mine fake P&L, which is exactly what drove the
        corner-saturation and value inflation. A dead contract can't be traded ⇒ its one-step move
        is 0 (matching the outer's frozen convention); `live_i` = the contract still prices at
        t+1."""
        inner = self.bundle.inner_mc(t)
        F_t = torch.stack([self.tradables_sim[ref][t] for ref in self.hedges], dim=-1)   # (B_outer, n_hedge)
        F_t1 = torch.stack([inner["F_t1"][ref] for ref in self.hedges], dim=-1)          # (B_outer, B_inner, n_hedge)
        # EXPIRED-CONTRACT GUARD: a dead contract's one-step move is forced to 0 (see docstring).
        live = (F_t1.abs().amax(dim=(0, 1)) > 0).to(F_t1.dtype)                          # (n_hedge,)
        dF = (F_t1 - F_t.unsqueeze(1)) * live
        dL = inner["L_t1"] - inner["L_t"]                                                # (B_outer, B_inner)
        return dF, dL, inner["market_t1"], inner["market_t"], live

    # ---- external argmax (Bellman max outside the fitted value) --------------
    def _decide(self, nets, market_t1, dF, dL, W, t, q_prev=None, kappa=None, live=None,
                rate_limit=True):
        """Pick the grid action maximising E_inner[C_{t+1}] per outer path. No grad. Under
        `Objective.Reference_Mode='Running_Wealth'` the score is `E_inner[u(W1 − W_t) + C_{t+1}]`
        instead — the same expression the LABEL regresses on in `_fit_step`, which is the whole
        point of stating it once per seam rather than once per mode: a ranking that scored
        terminal wealth while the labels scored increments would be a train/deploy split.
        The action universe is `aspace.grid_at(t, live)` — the base grid, further filtered to the
        `Total_Position_Schedule` corridor at t when one is configured (else the base grid). The
        `live` leg mask (the callers zero `q*live` after expiry) enters the filter so the corridor
        bounds the REALIZED Σ(q_i·live_i), not a target the expiry mask then guts.
        Under `Allocation_Weights` that universe is the FACTORED Q-ladder instead — rungs on the
        signed net cover, split across the legs by the schedule, built on the LIVE legs so a rung
        is a book this step can actually hold. The charge, the deadband incumbent and the position
        state consume those rows unchanged. `Max_Trade_Per_Step` is the one correction that has to
        know: it measures |ΔΣq| rather than per-leg |Δq_i|, because the policy's control is the
        net cover and the composition jump at a knot is the schedule's mandate, not a trade the
        argmax chose (per-leg, every row at a knot step is infeasible and the band silently
        vanishes through the fallback below).

        `q_prev` carries the standing book and arms up to four decision-time corrections,
        each with its own further switch: the cost-aware turnover charge (`kappa` present, priced
        net/paired when `Calendar_Spread_Bps` gives `_calendar_kappa` a rate), the
        quadratic churn charge (`self.churn_lambda` > 0), the `Max_Trade_Per_Step` band
        (`rate_limit` True — the label path lowers it so the band stays execution-only) and the
        `Decision_Deadband_Sigma` incumbent. The band takes the best action INSIDE the reachable
        set and falls back to the unrestricted best when NOTHING is reachable — which only a
        corridor step can cause, now that a schedule's own recomposition is inside the |ΔΣq|
        measure rather than striking out every row and disarming the band silently. Without
        `DiffV2_Position_State` the fitted value targets stay cost-free and these shape
        SELECTION only.

        Under `DiffV2_Position_State` the same charge (plus the terminal unwind) also enters the
        label's target in `_fit_step`, and the successor is queried at the book each CANDIDATE
        leaves standing, p' = Sum(a·live)/Q_max — the realized book.

        EXPIRY, one convention end to end: every position this method prices, states and RETURNS
        is LIVE-MASKED — candidate, standing book, incumbent, rate-limit band and answer alike. A
        dead leg cannot be traded and earns nothing, so a move on it is a phantom, and pricing
        that phantom is not free: measured, an unmasked `q_prev` let the argmax park |50| on a
        dead leg purely to dodge the phantom's unwind, and because the action grid is cap-filtered
        that parking spent the whole `Total_Position_Abs_Limit` on nothing. Masked, the rows that
        differ only on dead legs collapse to one book, so the ranking compares realizable
        alternatives and the answer is the book the caller will actually hold.

        `Decision_Deadband_Sigma` (`aspace.deadband_sigma` > 0, also execution-only) then makes
        the standing book the INCUMBENT candidate: the winner only trades when it beats holding
        by that many standard errors of the difference PAIRED across the common inner draws, so
        a near-tie between two noisy estimates leaves the book alone instead of teleporting.
        Holding an infeasible book is not on offer — a path whose `q_prev` sits outside the
        corridor at t moves regardless. The incumbent runs through the SAME `_reposition_charge`
        as every rival — `dq = 0` kills the L1 and churn terms while the terminal unwind, a
        charge on the LEVEL, survives — and is queried at its OWN position state, so the paired
        difference subtracts two readings of one function rather than two functions.

        RULINGS. (i) The deadband is `rate_limit`-gated, so the label path fits against a
        no-deadband argmax while deployment holds — the same execution-only contract
        `Max_Trade_Per_Step` has. In the label path instead, it would make the target
        path-dependent on `q_prev` beyond the charge. (ii) The move test compares RISK-NEUTRAL
        means while `best_val`/`inc` carry the `DiffV2_Risk_Kappa` semideviation penalty: the
        paired standard error is an estimator statement about E[C], and a semideviation is not a
        mean it could be a standard error of. The band stays risk-neutral by construction."""
        grid = self.aspace.grid_at(t, live)
        limit = rate_limit and q_prev is not None and self.aspace.max_trade > 0.0
        dead = rate_limit and q_prev is not None and self.aspace.deadband_sigma > 0.0
        with torch.no_grad():
            B, Bi, md = market_t1.shape
            if dead and Bi < 2:
                raise ValueError(
                    "Decision_Deadband_Sigma needs Inner_Sub_Batch >= 2: the hold test is a "
                    "PAIRED standard error over the inner draws, so one draw makes it NaN, every "
                    "comparison False, and the book would silently never trade again")
            kappa_T = self._unwind_kappa(t)
            kappa_cal = self._calendar_kappa(t)
            rows = torch.arange(B, device=market_t1.device)
            # ONE expiry convention (see docstring): the standing book, like every candidate, is
            # only what still lives.
            q_live = q_prev if q_prev is None or live is None else q_prev * live
            best_val = None
            best_q = None
            best_f = None
            band_val = None
            band_q = None
            band_f = None
            for s in range(0, grid.shape[0], self.chunk):
                acts = grid[s:s + self.chunk]                                            # (c, n_hedge)
                c = acts.shape[0]
                q = acts[None, :, None, :]                                               # (1,c,1,n_hedge)
                a_live = acts if live is None else acts * live                            # (c,n_hedge)
                W1 = self._wealth_step(W[:, None, None], q, dF[:, None], dL[:, None])     # (B,c,Bi)
                # kappa_T rides OUTSIDE the q_prev gate: the terminal unwind is a property of the
                # ACTION, so a step with no standing book must not liquidate for free.
                if kappa_T is not None or (q_live is not None
                                           and (kappa is not None or self.churn_lambda > 0.0)):
                    prev = a_live[None] if q_live is None else q_live[:, None, :]
                    W1 = W1 - self._reposition_charge(                                    # (B,c) $
                        a_live[None, :, :], prev, kappa, kappa_T, kappa_cal)[:, :, None]
                # The successor is queried at the book the CANDIDATE leaves standing (see docstring).
                p1 = ((a_live.sum(-1) / self.total_abs_limit)[None, :, None]
                      .expand(B, c, Bi).reshape(-1) if self.position_state else None)
                C1f = self._continuation(
                    nets,
                    market_t1[:, None].expand(B, c, Bi, md).reshape(-1, md),
                    W1.reshape(-1), t + 1, p1).reshape(B, c, Bi)                          # (B,c,Bi)
                if self.running_wealth:
                    # The DAY's reward, added to a continuation that carries no anchor: the
                    # increment is read off the CHARGED W1, so a move pays its toll inside the
                    # step that made it rather than at a terminal the reward no longer reaches.
                    C1f = C1f + self._u_step(W1, W[:, None, None], t + 1)
                C1 = C1f.mean(-1)                                                         # E_inner[C] (B,c)
                if self.risk_kappa > 0.0:        # downside-aware: penalise per-action downside dispersion
                    dev = (C1f - C1.unsqueeze(-1)).clamp(max=0.0)                         # negatives only
                    C1 = C1 - self.risk_kappa * (dev ** 2).mean(-1).sqrt()
                if limit:
                    dq_c = a_live[None, :, :] - q_live[:, None, :]                        # (B,c,n_h)
                    # Under a weight schedule the policy's control is the NET cover, so the band
                    # measures |ΔΣq|: the composition jump at a knot is the SCHEDULE's, mandated
                    # and unavoidable, and a per-leg band would strike out every single row at
                    # exactly those steps and silently disarm itself through the fallback below.
                    infeas = ((dq_c.sum(-1).abs() > self.aspace.max_trade)
                              if self.aspace.weights is not None
                              else (dq_c.abs() > self.aspace.max_trade).any(-1))           # (B,c)
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
                best_val = torch.where(ok, band_val, best_val)
                best_f = torch.where(ok.unsqueeze(-1), band_f, best_f)
            if dead:
                # The incumbent is a CANDIDATE, priced by the same rules: hold q_live, so dq = 0
                # kills the L1 and churn terms and only the terminal unwind — a charge on the
                # level — survives. Holding therefore pays for its own liquidation exactly as
                # trading INTO the same book would, and the paired difference below is a
                # difference of two like things.
                W1 = self._wealth_step(W[:, None, None], q_live[:, None, None, :],
                                       dF[:, None], dL[:, None])                          # (B,1,Bi)
                if kappa_T is not None or kappa is not None or self.churn_lambda > 0.0:
                    W1 = W1 - self._reposition_charge(
                        q_live, q_live, kappa, kappa_T, kappa_cal)[:, None, None]
                p_inc = ((q_live.sum(-1) / self.total_abs_limit)[:, None, None]
                         .expand(B, 1, Bi).reshape(-1) if self.position_state else None)
                C1f_inc = self._continuation(
                    nets, market_t1.reshape(-1, md), W1.reshape(-1), t + 1,
                    p_inc).reshape(B, Bi)
                if self.running_wealth:
                    # The incumbent earns its own day's increment, priced exactly as every rival's.
                    C1f_inc = C1f_inc + self._u_step(W1, W[:, None, None], t + 1).reshape(B, Bi)
                inc = C1f_inc.mean(-1)
                if self.risk_kappa > 0.0:
                    dev = (C1f_inc - inc.unsqueeze(-1)).clamp(max=0.0)
                    inc = inc - self.risk_kappa * (dev ** 2).mean(-1).sqrt()
                diff = best_f - C1f_inc                                                   # (B,Bi) paired
                se = diff.std(-1) / math.sqrt(Bi)
                move = diff.mean(-1) > self.aspace.deadband_sigma * se
                # Holding is only on offer while the LIVE book is corridor-feasible at t.
                hold_ok = ((self.aspace.project_to_corridor(q_live, t) - q_live).abs()
                           <= 1e-9).all(-1)
                move = move | ~hold_ok
                best_q = torch.where(move.unsqueeze(-1), best_q, q_live)
                best_val = torch.where(move, best_val, inc)
            return best_q, best_val

    # ---- operating-region bank ----------------------------------------------
    def _replication_hedge(self, t):
        """Per-instrument diagonal min-var hedge from the OUTER one-step moves at t:
        q_i = −Cov(dL, dF_i)/(cs_i·Var(dF_i)) / n_active, clamped to [lo,hi]. Inactive → 0."""
        L = self.liability_sim
        dL = (L[t + 1] - L[t])                                                            # (B_outer,)
        q = torch.zeros(self.n_hedge, device=self.device)
        for i in self.active:
            ref = self.hedges[i]
            dF = self.tradables_sim[ref][t + 1] - self.tradables_sim[ref][t]              # (B_outer,)
            var = dF.var()
            # Skip degenerate OR non-finite instruments (an expired-contract row can carry a
            # NaN mark; without this guard clamp(NaN)=NaN poisons the whole textbook book).
            if not float(var) > 1e-12 or not torch.isfinite(dF).all():
                continue
            beta = ((dL - dL.mean()) * (dF - dF.mean())).mean() / var
            q[i] = (-beta / (self.contract_size[i] * max(self.n_active, 1)))
        return torch.minimum(torch.maximum(q, self.q_lo), self.q_hi)

    def _build_bank(self, gen):
        """Operating-region bank — roll the OUTER paths forward (cheap; outer-realised
        moves only, no inner MC): hold q = clamp(q_rep_t + noise) each step, accumulate
        W = cum hedge P&L + L_t. Returns per-t lists W_t (B_outer,) and q_prev (B_outer,
        n_hedge). The bank-state `market_t` is read lazily from the SAME inner-MC fork the
        backward sweep makes at t (no extra inner-MC passes).

        Under `DiffV2_Position_State` the bank also has to COVER THE p AXIS, because p is now an
        input the argmax queries over the whole of `aspace.net_bounds(t)` while the roll only
        wanders `Bank_Noise_Frac` around the replication hedge. Measured on the platinum book,
        that left the deep-hedge ~44% of the axis unsampled and let p_bank breach the cap the grid
        enforces, so the learned hysteresis slope was extrapolation exactly where it mattered.
        Each path therefore draws a UNIFORM net target across the feasible band and water-fills
        onto it: the per-leg shape stays the noisy replication hedge, the total spans what the
        argmax can ask for, and p_bank is inside [-1, 1] by construction rather than by hope."""
        L = self.liability_sim
        W = L[0].clone()                                                                 # (B_outer,) = cum_pnl(0)+L_0
        # Seed q_prev from the OPENING book. Position-free, this only labels q_list[0] (the bank
        # wealth W below never reads q_prev); under `DiffV2_Position_State` q_list[t] IS the
        # sampled position state at t, and `DiffV2_Bank_Noise_Frac` is what makes the p-slice
        # identifiable — the value fn only learns a no-trade region over books it has seen.
        q_prev = self.aspace.initial_q(self.B_outer, self.device)
        W_list, q_list = [], []
        rng = (self.q_hi - self.q_lo)
        mask = self.aspace.active_mask
        oob = []                                                                         # per-t bank corridor-breach diagnostic
        for t in range(self.T_dec):
            W_list.append(W.clone())
            q_list.append(q_prev.clone())
            q_rep = self._replication_hedge(t)                                            # (n_hedge,)
            noise = self.noise_frac * rng * torch.randn(
                self.B_outer, self.n_hedge, generator=gen, device=self.device)
            q = torch.minimum(torch.maximum(q_rep[None] + noise * mask, self.q_lo), self.q_hi)
            if self.position_state:
                # Cover the p axis the argmax queries: a uniform net target over the feasible
                # band, water-filled into the per-leg boxes (see docstring). Subsumes the
                # corridor projection below — net_bounds already intersects the corridor.
                lo_t, hi_t = self.aspace.net_bounds(t)
                u = lo_t + (hi_t - lo_t) * torch.rand(
                    self.B_outer, 1, generator=gen, device=self.device)
                q = self.aspace.waterfill(q, u, u)
            if self.aspace.schedule is not None:
                # Keep the exploration IN the corridor so the value fn trains on reachable wealth
                # states only (un-projected, ~50% of bank paths breach — the nets would fit
                # unreachable W). Diagnostic below then confirms ~0 residual breach.
                q = self.aspace.project_to_corridor(q, t)
                lo, hi = self.aspace._corridor_at(t)                                      # signed-total corridor at t
                tot = q[:, self.active].sum(-1)                                           # bank signed total (active legs)
                tol = 1e-3                     # float32 headroom-fill lands on the edge to ~1e-6
                oob.append((t, lo, hi, float((tot < lo - tol).float().mean()
                                             + (tot > hi + tol).float().mean()),
                            float(tot.min()), float(tot.max())))
            dF = torch.stack(
                [self.tradables_sim[ref][t + 1] - self.tradables_sim[ref][t] for ref in self.hedges],
                dim=-1)                                                                  # (B_outer, n_hedge)
            W = self._wealth_step(W, q, dF, L[t + 1] - L[t])
            q_prev = q
        if oob:
            worst = max(oob, key=lambda r: r[3])
            logging.info(
                "DiffSolver bank IN Total_Position_Schedule (post-projection): residual "
                "frac(Σq outside corridor) min=%.3f mean=%.3f max=%.3f | worst t=%d "
                "corridor=[%.4g, %.4g] bank Σq∈[%.4g, %.4g] frac_oob=%.3f (≈0 ⇒ projection clean)",
                min(r[3] for r in oob), sum(r[3] for r in oob) / len(oob),
                worst[3], worst[0], worst[1], worst[2], worst[4], worst[5], worst[3])
        if self.log_ratio:
            self._bank_worst = float(min(min(float(w.min()) for w in W_list), float(W.min())))
        return W_list, q_list

    # ---- project per-process state-at-t leaf grads → market_t columns --------
    def _project_leaf_grads(self, leaf_grads, widths, rows, n, md):
        """Map ∂Y/∂(state_t leaf) for each simulated factor into the `(n, md)` gradient w.r.t.
        the privileged market_t columns the value net consumes. `widths` is in market-column
        order (factor iteration order) as `(key, width, state_suffixes)`; the suffixes are the
        process's DIFFERENTIABLE state coordinates (`diff_state_leaves()` — model-agnostic, no
        regime/belief concept here). Layout convention: state columns first (from the state
        leaves, in declared order), the trailing price columns last (from the raw factor leaf).
        Factors with no state suffix map 1:1 raw→privileged (a curve) or leave the non-price
        state masked (e.g. a detached-variance coordinate). Unmeasured / unconnected leaves
        leave their columns at 0 (masked — only the value supervises them)."""
        g = torch.zeros(n, md, device=self.device)
        col = 0
        for key, width, suffixes in widths:
            if width <= 0:                                      # privileged-empty factor (not in market_t)
                continue
            gr = leaf_grads.get(key)
            kr = (gr[..., rows].numel() // n) if gr is not None else 0         # raw-leaf column count (price)
            if suffixes:
                gb = leaf_grads.get((key, suffixes[0]))                        # single state leaf today
                if gb is not None and gb[..., rows].numel() == (width - kr) * n:
                    nb = width - kr
                    g[:, col:col + nb] = gb[..., rows].reshape(nb, n).transpose(0, 1)
                    if gr is not None and kr >= 1:
                        g[:, col + nb:col + width] = gr[..., rows].reshape(kr, n).transpose(0, 1)
                # else: state leaf absent/mismatched → whole block masked (value-only)
            elif kr == width:                                                 # 1:1 raw → privileged (curve)
                g[:, col:col + width] = gr[..., rows].reshape(width, n).transpose(0, 1)
            elif gr is not None and kr >= 1:                                  # masked state + price (raw last)
                g[:, col + width - kr:col + width] = gr[..., rows].reshape(kr, n).transpose(0, 1)
            col += width
        return g

    # ---- one backward step: bootstrap + advantage twin fit -------------------
    def _fit_step(self, nets, W_bank, t, inner, q_bank, rows=slice(None), live_prev=None):
        """One backward step at t: bootstrap the value target off the cached inner draws, then fit
        `nets[t]` to it through `_fit_from_labels`.

        The GRAD inner-MC fork returns AAD-live one-step F_t1/L_t1/market_t1 plus the per-process
        state-at-t LEAVES, so the bootstrap value Y AND its pathwise gradients w.r.t. W0 (wealth)
        and the market state (spot/state) come from the SAME forward — the full Huge–Savine twin
        loss. ∂Y/∂market_t is the differential constraint that regularizes the market dimension,
        where a value-only / W-only fit overfits the few outer paths.

        Under `Objective.Reference_Mode='Running_Wealth'` the target gains the step's own reward:
        Y = E_inner[u(W1 − W_t) + C_{t+1}], with C_{T_dec} ≡ 0, so the LAST decision's label is
        exactly u(ΔW_T) — the forced-flat unwind already inside W1 under `DiffV2_Position_State`,
        because the charge is subtracted before the increment is taken. It is the same expression
        `_decide` ranks candidates by, by construction.

        Under `DiffV2_Position_State` the step is FRICTIONAL: the bank's own standing book supplies
        the position state p at t, the selection is charged its repositioning cost, and — the part
        that distinguishes this from `DiffV2_Cost_Aware_Argmax` — the SAME charge is subtracted from
        the wealth entering the target, whose successor is read at the book the chosen action leaves
        standing. So the toll compounds down the recursion instead of being re-paid and forgotten
        each day, which is the only way a no-trade region can appear in a fitted value.

        THE ONE MASK RULE for p, the coordinate `nets[t]` is both fitted and queried on:

            p_t = Sum(q_standing · live_{t-1}) / Q_max

        — the book masked by the step that SET it, not by the step that is about to trade it.
        `_decide` at t−1 can only ever hold its own `live`, so it hands the successor
        `Sum(a·live_{t-1})/Q_max`; the fit has to meet it there. Masking the bank's book with this
        step's `live` instead reads `nets[t]` on one coordinate and writes it on another, and the
        two disagree by the FULL width of the axis on exactly the step where a leg rolls off.
        `live_prev` carries that mask in (`_sweep` reads it off the cache; the first swept step has
        no predecessor and is fitted but never queried). The CHARGE keeps this step's `live` — what
        you own and what you may trade are different questions."""
        dF_ng, dL_ng, m1_ng, market0, live = inner                                       # no-grad cache
        market0 = market0[rows]
        W0_bank = W_bank[t][rows]
        # live-masked like every other caller (an expired leg's phantom book must not buy
        # position budget); with churn_lambda 0 and no kappa the book is inert - the band
        # stays execution-only via rate_limit
        q_prev = q_bank[t][rows] * live
        # ...but the position STATE is the book masked by the step that set it (see docstring).
        q_state = q_bank[t][rows] * (live if live_prev is None else live_prev)
        kappa_t = self.aspace.kappa(self.tradables_sim, t) if self.position_state else None
        # SELECT the action on the NO-GRAD inner draws; EVALUATE its value + pathwise gradients
        # on a fresh GRAD inner (independent draws → cross-fit, no winner's-curse max-bias).
        q_star, _ = self._decide(
            nets, m1_ng[rows], dF_ng[rows], dL_ng[rows], W0_bank, t, live=live,
            q_prev=q_prev, kappa=kappa_t, rate_limit=False)
        q_star = q_star * live          # expired contracts: dF=0 ⇒ wealth-neutral; report 0, not the tie

        # GRAD inner-MC fork: AAD-live t→t+1 quantities + state-at-t leaves (see docstring).
        ig = self.bundle.inner_mc_grad(t)
        leaves, widths = ig["state_t_leaves"], ig["state_t_leaf_widths"]
        if not getattr(self, "_proj_checked", False):
            self._proj_checked = True                  # one-time self-check of the label projection
            mt, col, errs = ig["market_t"].detach(), 0, []      # detach: numeric self-check only
            n = mt.shape[0]
            for key, width, suffixes in widths:
                if width <= 0:
                    continue
                pl = leaves.get(key)
                pl = pl.detach() if pl is not None else None
                kr = (pl.numel() // n) if pl is not None else 0
                bl = leaves.get((key, suffixes[0])) if suffixes else None
                bl = bl.detach() if bl is not None else None
                if bl is not None and bl.numel() == (width - kr) * n:          # state + price
                    nb = width - kr
                    be = float((mt[:, col:col + nb] - bl.reshape(nb, -1).transpose(0, 1)).abs().max())
                    pe = float((mt[:, col + nb:col + width] - pl.reshape(kr, -1).transpose(0, 1)).abs().max()) \
                        if pl is not None and kr >= 1 else -1.0
                    errs.append(f"{utils.check_tuple_name(key)}[state={be:.1g},price={pe:.1g}]")
                elif not suffixes and pl is not None and kr == width:          # 1:1 raw → privileged
                    e = float((mt[:, col:col + width] - pl.reshape(width, -1).transpose(0, 1)).abs().max())
                    errs.append(f"{utils.check_tuple_name(key)}[1:1={e:.1g}]")
                elif not suffixes and pl is not None and kr >= 1:              # masked state + price
                    pe = float((mt[:, col + width - kr:col + width] - pl.reshape(kr, -1).transpose(0, 1)).abs().max())
                    errs.append(f"{utils.check_tuple_name(key)}[maskedstate,price={pe:.1g}]")
                else:                                                          # leaf absent → masked
                    errs.append(f"{utils.check_tuple_name(key)}[unmeasured]")
                col += width
            logging.info("DiffSolver differential-label projection check (privileged market_t "
                         "cols vs state_t leaves; ≈0 ⇒ ∂Y/∂market_col == ∂Y/∂leaf): %s",
                         " ".join(errs))
        F_t = torch.stack([self.tradables_sim[r][t] for r in self.hedges], dim=-1)        # (B_outer,n_hedge)
        F_t1 = torch.stack([ig["F_t1"][r] for r in self.hedges], dim=-1) * live           # AAD-live
        dF_g = (F_t1 - F_t.unsqueeze(1))[rows]
        dL_g = (ig["L_t1"] - ig["L_t"])[rows]
        m1_g = ig["market_t1"][rows]
        W0 = W0_bank.clone().requires_grad_(True)
        q = q_star[:, None, :]                                                           # (B,1,n_hedge)
        W1 = self._wealth_step(W0[:, None], q, dF_g, dL_g)                                # (B,Bi)
        B, Bi_e, md = m1_g.shape
        p_bank = p1 = None
        if self.position_state:
            # The charge the SELECTION above ranked with, now inside the regressed TARGET, and the
            # successor read at the book the chosen action leaves standing (see docstring).
            W1 = W1 - self._reposition_charge(
                q_star, q_prev, kappa_t, self._unwind_kappa(t), self._calendar_kappa(t))[:, None]
            p_bank = q_state.sum(-1) / self.total_abs_limit               # state at t: live_{t-1}
            p1 = (q_star.sum(-1) / self.total_abs_limit)[:, None].expand(B, Bi_e).reshape(-1)
        Y = self._continuation(
            nets, m1_g.reshape(-1, md), W1.reshape(-1), t + 1, p1).reshape(B, Bi_e)       # (B,Bi)
        if self.running_wealth:
            # Y = u(W1 − W_t) + C_{t+1}, the same expression `_decide` ranked with. W0 is the
            # autograd leaf and the increment is level-free (W1 − W0 = Σq·cs·dF + dL − charge), so
            # it contributes exactly 0 to the wealth label and its MARKET slope joins g_market —
            # the reward is now part of what the twin loss differentiates.
            Y = Y + self._u_step(W1, W0[:, None], t + 1)
        Y = Y.mean(1)                                                                     # (B,)
        grads = torch.autograd.grad(Y.sum(), [W0] + list(leaves.values()), allow_unused=True)
        gW = grads[0].detach()
        leaf_grads = {k: (g.detach() if g is not None else None)
                      for k, g in zip(leaves.keys(), grads[1:])}
        g_market = self._project_leaf_grads(leaf_grads, widths, rows, B, md)             # ∂Y/∂market_t (B,md)
        Y = Y.detach()
        return self._fit_from_labels(nets, W0_bank, market0, Y, gW, g_market, t, q_star, p_bank)

    def _fit_from_labels(self, nets, W0_bank, market0, Y, gW, g_market, t, q_star, p_bank):
        """Shared fit tail: advantage decomposition + standardized twin loss on the
        (value, wealth-grad, market-grad) labels. Called by both the single-fork and the
        sub-sliced large-B label paths of `_fit_step`.

        The twin loss matches gradients in STANDARDIZED space (g_zn = std·g_raw). Raw-space
        matching is mis-scaled: ∂A/∂W ~ 1e-6 in dollars, ∂A/∂spot ~ 1e-4 — both inert against the
        O(0.1) value term. Standardized, ∂A/∂wn and ∂A/∂mn are O(1) and the gradient match
        actually regularizes — the principled regularizer of differential ML (NOT weight decay).

        Huge–Savine term BALANCING: with W~$1e6 and utility~O(1) the standardized W-gradient label
        is ~600× the value label, so an unnormalized sum lets the W-gradient drown the value fit
        AND the market gradient. Each term is normalized by its label variance so all are O(1) and
        `lam_g` balances value-vs-gradient as intended.

        `p_bank` (the position state, `DiffV2_Position_State`) is an input column with NO gradient
        label: the twin loss only ever had the channels the AAD fork measures (wealth, market
        state), and p is a control the bank samples rather than a simulated coordinate the label
        forward differentiates. It is supervised by the value term alone.

        `DiffV2_Wealth_Free_Value` is the same discipline read the other way: W is no longer an
        input, so it can no longer be a MEASURED COLUMN either and its gradient term leaves the
        loss with it. A differential label on a channel the net does not consume has no parameter
        to reach — matching ∂A/∂wn against a measured `g_zn_W` while wn is absent would be asking
        the market columns to absorb the wealth slope, which is precisely the slope the switch
        exists to remove. The wealth channel is then supervised nowhere, by design: the anchor
        carries it exactly."""
        # Advantage decomposition: fit A = C − u(W0); subtract the anchor's wealth slope. Under
        # `Reference_Mode='Running_Wealth'` C_t = A_t alone (`_continuation`'s ruling), so there is
        # no anchor to subtract and the labels ARE the residual's — the same statement `use_adv`
        # makes by hand.
        if self.use_adv and not self.running_wealth:
            Wb = W0_bank.clone().requires_grad_(True)
            (dB_dW,) = torch.autograd.grad(self._u(Wb, t).sum(), Wb)
            a_val = Y - self._u(W0_bank, t)
            a_gW = gW - dB_dW.detach()
        else:
            a_val, a_gW = Y, gW

        net = nets[t]
        # The chain: the terminal net anchors and trains the full budget; every earlier net
        # inherits its fitted successor on its FIRST fit and only corrects it.
        ref = nets[t + 1] if t + 1 < self.T_dec else None
        if ref is not None and self._opts.get(t) is None:
            net.load_state_dict(ref.state_dict())
        prox_ref = (torch.cat([p.detach().reshape(-1) for p in ref.parameters()])
                    if ref is not None and self.prox > 0.0 else None)
        # Twin loss in STANDARDIZED space: g_zn = std·g_raw (see docstring).
        lam_g = float(self.cfg.get("diffv2_lambda_grad", 1.0))
        g_zn_W = self.w_std * a_gW                                                       # (B,)
        g_zn_m = self.m_std * g_market                                                   # (B,md); u indep of market
        # Per-term label-variance normalization — the Huge–Savine balancing (see docstring).
        nrm_v = a_val.var() + 1e-8
        nrm_w = g_zn_W.var() + 1e-8
        # Per-column lambda_j (Huge-Savine official): each market column normalized by its
        # own label variance, so one fat-tailed column can't deflate the differential
        # constraint for the rest. 'No' = legacy pooled scalar.
        nrm_m = (g_zn_m.var(dim=0, keepdim=True) + 1e-8
                 if bool(self.cfg.get("diffv2_per_column_grad_norm", False))
                 else g_zn_m.var() + 1e-8)
        # One optimizer per t, created at its first fit and kept: a streaming step resumes the
        # same Adam moments on a warm-started net instead of restarting them on every batch.
        opt = self._opts.get(t)
        if opt is None:
            opt = self._opts[t] = torch.optim.Adam(
                net.parameters(), lr=self.lr,
                weight_decay=float(self.cfg.get("diffv2_weight_decay", 0.0)))
        prev_loss, iters_run = None, self.fit_iters
        for it in range(self.fit_iters):
            mn = ((market0 - self.m_mean) / self.m_std).detach().requires_grad_(True)
            wn = ((W0_bank - self.w_mean) / self.w_std).detach().requires_grad_(True)
            cols = [mn] if self.wealth_free else [mn, wn.unsqueeze(-1)]
            a = net(torch.cat(cols
                              + ([] if p_bank is None else [p_bank.unsqueeze(-1)]), dim=-1))
            # The MEASURED COLUMNS of the twin loss are exactly the input channels the net
            # consumes: `DiffV2_Wealth_Free_Value` takes W out of both at once (see docstring).
            das = torch.autograd.grad(a.sum(), [mn] if self.wealth_free else [mn, wn],
                                      create_graph=True)
            da_m = das[0]
            w_term = 0.0 if self.wealth_free else lam_g * ((das[1] - g_zn_W) ** 2).mean() / nrm_w
            loss = (((a - a_val) ** 2).mean() / nrm_v
                    + w_term
                    + lam_g * (((da_m - g_zn_m) ** 2) / nrm_m).mean())
            if prox_ref is not None:
                flat = torch.cat([p.reshape(-1) for p in net.parameters()])
                loss = loss + self.prox * (flat - prox_ref).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            # Block-wise early stop (non-anchor only): one host sync per 8-iter block, a
            # 2-block floor so a cold optimizer cannot plateau its way out, and the stall
            # test compares block ends - real progress, not per-step Adam jitter.
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
            # Trust region for all future EVALUATIONS of this net (argmax, bootstrap labels,
            # verdict): its fitted-target range plus one range-width of headroom each side.
            lo, hi = float(a_val.min()), float(a_val.max())
            if self._bounds_frozen:
                # Streaming: the region is part of the LOCKED frame — re-fitting it per batch
                # would re-open the phantom-extrapolation basin the clamp exists to close. A
                # later batch only REPORTS how often its own targets fall outside it.
                b0, b1 = self.a_bounds[t]
                self._breaches.append(
                    (t, float(((a_val < b0) | (a_val > b1)).to(torch.float32).mean()), lo, hi))
            else:
                pad = max(hi - lo, 1e-3)
                self.a_bounds[t] = (lo - pad, hi + pad)
        return {
            "t": t, "val_loss": val_loss, "iters": iters_run,
            "Y_absmean": float(Y.abs().mean()),
            "A_absmean": float(a_fit.abs().mean()),
            "q_star_mean": q_star.mean(0).detach().cpu().tolist(),
            "Y_mean": float(Y.mean()),
        }

    # ---- greedy-rollout downside verdict -------------------------------------
    def _verdict(self, nets, inner_cache, sweep_ts, rows=slice(None)):
        """Roll the fitted argmax policy forward over [t_min, T_dec] on the OUTER paths in
        `rows` (wealth advanced by the outer-realised dF/dL), starting FLAT at t_min, and
        compare terminal-wealth downside against a textbook diagonal-min-var delta hedge and
        no hedge. The argmax uses the cached inner-MC E[C_{t+1}] estimate; the realised
        outcome uses the outer path. Pass held-out `rows` for an honest OUT-OF-SAMPLE verdict.

        Returns per-policy {u_mean (the objective), wT_mean, wT_p5, wT_cvar5}. The verdict:
        the greedy policy should DOMINATE no-hedge on downside and be competitive with
        textbook — a SPECULATING policy (the old solver's failure) shows up as worse p5/CVaR
        than textbook and a wide wT spread.

        WHAT `u_mean` IS under `Objective.Reference_Mode='Running_Wealth'`: the SUM of the
        rollout's per-step utilities, because that is the objective the DP maximised. It is
        therefore NOT comparable to a terminal-utility `u_mean` — not across reference modes, and
        not against `run_textbook_benchmark` / `HindsightDpSolver`, which stay terminal-based by
        design (converting them would invent a rebalancing schedule neither one has). `wT_mean`
        and the tail columns are raw dollars under either mode and stay comparable throughout;
        `u_mean_is_step_sum` in the returned dict says which regime produced the number.

        Turnover cost is accounted in PARALLEL: Transaction_Cost_Per_Unit + half the
        Bid_Offer_Spread_Bps on |Δq| each rebalance, entry measured from the OPENING book q0 — not
        from flat, and the terminal unwind under `Force_Flat_At_End` — the same three-part rule
        `run_textbook_benchmark` and `HindsightDpSolver` price their own execution with, so the
        `*_net` columns of the three tracks are comparable. The wealth recursion itself stays
        cost-free, so these are DIAGNOSTICS quantifying that approximation per policy; the
        V_0/utility track stays the frictionless DP objective every track shares.

        The greedy ARGMAX is charged when either `DiffV2_Cost_Aware_Argmax` (execution hysteresis
        over a cost-free value) or `DiffV2_Position_State` is on: a frictional value function that
        was fitted against a charged selection must be rolled under the same rule, or the policy
        reported is not the policy trained. `argmax_charged` in the returned dict says which
        regime produced these numbers, because a charged argmax beside an uncharged wealth
        recursion is not readable from the figures alone."""
        L = self.liability_sim
        t0 = self.t_min
        n = L[t0][rows].shape[0]
        W = {p: L[t0][rows].clone() for p in ("greedy", "textbook", "nohedge")}
        q_traj = {"greedy": [], "textbook": []}                                          # mean |q| per step
        # Parallel turnover-cost DIAGNOSTIC; entry measured from the opening book (see docstring).
        cost = {p: torch.zeros(n, device=self.device) for p in ("greedy", "textbook")}
        q0 = self.aspace.initial_q(n, self.device)
        q_prev = {p: q0.clone() for p in ("greedy", "textbook")}
        # Running-wealth objective: the rollout's value is the SUM of the per-step rewards, so it
        # is accumulated along the roll rather than read off the terminal (see docstring). The net
        # sum charges each step's own turnover into that step's increment, which is what a
        # per-step objective means by a cost.
        u_run = ({p: torch.zeros(n, device=self.device) for p in W}
                 if self.running_wealth else None)
        u_net = ({p: torch.zeros(n, device=self.device) for p in cost}
                 if self.running_wealth else None)
        last_net_inc = {}          # the final step's net increment, which the unwind belongs to
        last_W_pre = {}            # ...and the wealth it launched from (the ratio needs the base)
        with torch.no_grad():
            for t in sweep_ts:
                dF_o = torch.stack(
                    [self.tradables_sim[r][t + 1][rows] - self.tradables_sim[r][t][rows]
                     for r in self.hedges], dim=-1)                                       # (n, n_hedge)
                dL_o = (L[t + 1] - L[t])[rows]
                dF, dL, m1, _, live = inner_cache[t]
                kappa_t = self.aspace.kappa(self.tradables_sim, t)
                q_g, _ = self._decide(
                    nets, m1[rows], dF[rows], dL[rows], W["greedy"], t,
                    q_prev=q_prev["greedy"],
                    kappa=kappa_t if self.cost_aware or self.position_state else None, live=live)
                q_g = q_g * live          # zero positions on expired contracts (wealth-neutral)
                # Textbook = diagonal min-var, PROJECTED into the corridor (no schedule ⇒ identity)
                # so the benchmark obeys the same mandate as greedy — a fair in-corridor comparison.
                q_tb = self.aspace.project_to_corridor(
                    self._replication_hedge(t)[None].expand(n, self.n_hedge), t)
                z = torch.zeros(n, self.n_hedge, device=self.device)
                kappa_cal_t = self.aspace.calendar_kappa(self.tradables_sim, t)
                step_cost = {}
                for p, q_now in (("greedy", q_g), ("textbook", q_tb)):
                    step_cost[p] = self.aspace.turnover_cost(q_now - q_prev[p], kappa_t,
                                                             kappa_cal_t)
                    cost[p] = cost[p] + step_cost[p]
                    q_prev[p] = q_now
                W_pre = dict(W)
                W["greedy"] = self._wealth_step(W["greedy"], q_g, dF_o, dL_o)
                W["textbook"] = self._wealth_step(W["textbook"], q_tb, dF_o, dL_o)
                W["nohedge"] = self._wealth_step(W["nohedge"], z, dF_o, dL_o)
                if u_run is not None:
                    # Indexed by the step the increment ARRIVES at, as every other seam does
                    # (`_decide`, `_fit_step`, `_score_actions` all pass t+1): the knee an
                    # increment is measured under belongs to the day it lands on.
                    for p in W:
                        u_run[p] = u_run[p] + self._u_step(W[p], W_pre[p], t + 1)
                    for p in u_net:
                        last_net_inc[p] = W[p] - W_pre[p] - step_cost[p]
                        last_W_pre[p] = W_pre[p]
                        u_net[p] = u_net[p] + self._u_step(
                            W_pre[p] + last_net_inc[p], W_pre[p], t + 1)
                q_traj["greedy"].append(q_g.mean(0).tolist())
                q_traj["textbook"].append(q_tb.mean(0).tolist())
            if self.force_flat:
                # The unwind both benchmark tracks charge, so all three price the same execution.
                kappa_T = self.aspace.kappa(self.tradables_sim, self.T_dec)
                for p in cost:
                    unwind = self.aspace.turnover_cost(q_prev[p], kappa_T)
                    cost[p] = cost[p] + unwind
                    if u_net is not None:
                        # The liquidation belongs INSIDE the last increment, because that is where
                        # training charged it: `_fit_step` subtracts the unwind from W1 before
                        # taking u(W1 − W_t), so the last label is u(ΔW_T − step − unwind). u is
                        # nonlinear, so appending u(−unwind) as its own term would report a
                        # different objective than the one optimised. Re-take the last step.
                        u_net[p] = (u_net[p]
                                    - self._u_step(last_W_pre[p] + last_net_inc[p],
                                                   last_W_pre[p], self.T_dec)
                                    + self._u_step(last_W_pre[p] + last_net_inc[p] - unwind,
                                                   last_W_pre[p], self.T_dec))

        def stats(wT, u_sum=None):
            p5 = torch.quantile(wT, 0.05)
            cvar5 = wT[wT <= p5].mean() if (wT <= p5).any() else p5
            return {"u_mean": float((self._u(wT) if u_sum is None else u_sum).mean()),
                    "wT_mean": float(wT.mean()),
                    "wT_p5": float(p5), "wT_cvar5": float(cvar5)}
        out = {p: stats(W[p], None if u_run is None else u_run[p]) for p in W}
        # Which regime chose the greedy book — a charged argmax over an uncharged wealth
        # recursion is not readable off the figures — and whether `u_mean` is a terminal utility
        # or a sum of per-step ones, which no figure states either.
        out["argmax_charged"] = bool(self.cost_aware or self.position_state)
        out["u_mean_is_step_sum"] = bool(self.running_wealth)
        for p in ("greedy", "textbook"):
            net_stats = stats(W[p] - cost[p], None if u_net is None else u_net[p])
            out[p]["turnover_cost_mean"] = float(cost[p].mean())
            out[p].update({f"{k}_net": v for k, v in net_stats.items()})
        # greedy position summary: mean over the rollout of |q| per instrument (is it hedging?)
        gq = torch.tensor(q_traj["greedy"])                                              # (n_steps, n_hedge)
        out["greedy_mean_abs_q"] = gq.abs().mean(0).tolist()
        out["greedy_q_traj"] = q_traj["greedy"]          # full per-t mean book (audit trail)
        out["textbook_q_traj"] = q_traj["textbook"]      # corridor-projected benchmark book (audit)
        out["greedy_q_first"] = q_traj["greedy"][0] if q_traj["greedy"] else None
        out["greedy_q_mid"] = (q_traj["greedy"][len(q_traj["greedy"]) // 2]
                               if q_traj["greedy"] else None)
        return out

    # ---- decision-curve diagnostic (Solver.DiffV2_Decision_Curve_Dump) --------
    def _score_actions(self, nets, t, acts, market_t1, dF, dL, W, q_live, kappa, live):
        """Score `acts` `(K, n_hedge)` the way `_decide`'s chunk loop ranks its grid rows —
        `E_inner[C_{t+1}]` before and after the repositioning charge — and hand back the CHARGED
        per-draw `W1` the ranking's continuation was actually read at. Returns `(pre, charged,
        W1)`, `(K,)`/`(K,)`/`(K, Bi)`: the single-path diagnostic drops the outer axis.

        WHY A SECOND PASS rather than a probe wired into `_decide`. The switch's contract is that
        a run with the dump on is bit-identical to one with it off, and the cheapest way to mean
        that is for `_decide` to keep no knowledge of it at all — no parameter, no branch, no
        extra continuation inside the hot loop that a future edit could leave armed. The cost of
        the choice is that the two rankings could drift apart, and that is exactly what the gate
        on the argmax column measures: it charges the candidates through `q_prev` so a scoring
        pass that forgot the charge would pick a different rung, and asserts the dump's argmax IS
        the executed book. The pass is affordable because the dump is a `Batch_Size = 1`
        diagnostic — one path times the rungs, against a training loop's thousands."""
        B, Bi, md = market_t1.shape
        K = acts.shape[0]
        kappa_T, kappa_cal = self._unwind_kappa(t), self._calendar_kappa(t)
        a_live = acts if live is None else acts * live
        with torch.no_grad():
            # `_decide` steps wealth with the RAW row and charges the LIVE-masked one; dF is
            # already zeroed on dead legs, so the two agree on W1 and differ only in the toll.
            W1 = self._wealth_step(W[:, None, None], acts[None, :, None, :],
                                   dF[:, None], dL[:, None])                         # (B,K,Bi)
            W1c = W1
            if kappa_T is not None or kappa is not None or self.churn_lambda > 0.0:
                W1c = W1 - self._reposition_charge(
                    a_live[None, :, :], q_live[:, None, :], kappa, kappa_T, kappa_cal)[:, :, None]
            p1 = ((a_live.sum(-1) / self.total_abs_limit)[None, :, None]
                  .expand(B, K, Bi).reshape(-1) if self.position_state else None)
            market = market_t1[:, None].expand(B, K, Bi, md).reshape(-1, md)

            def rank(Wx):
                C1f = self._continuation(nets, market, Wx.reshape(-1), t + 1, p1).reshape(B, K, Bi)
                if self.running_wealth:
                    C1f = C1f + self._u_step(Wx, W[:, None, None], t + 1)   # the day's own reward
                C1 = C1f.mean(-1)
                if self.risk_kappa > 0.0:
                    dev = (C1f - C1.unsqueeze(-1)).clamp(max=0.0)
                    C1 = C1 - self.risk_kappa * (dev ** 2).mean(-1).sqrt()
                return C1[0]

            charged = rank(W1c)
            return (charged if W1c is W1 else rank(W1)), charged, W1c[0]

    def _decision_curve_row(self, nets, t, market_t1, dF, dL, W, q_prev, q_chosen, live, kappa):
        """One CSV row of `Solver.DiffV2_Decision_Curve_Dump`: everything about decision `t` that
        says WHY the ranking chose the book it chose, and nothing that changes it.

        The columns, in four groups.

        *The decision.* `q_prev_net` / `q_chosen_net` are the realized signed covers Σq·live,
        `argmax_net` is the dump's own band-feasible charged argmax, and `deadband_held` is 1 only
        when `Decision_Deadband_Sigma` is armed, the executed book IS the incumbent, and the
        argmax was somewhere else — i.e. the band, not agreement, is what stopped the trade.

        *The utility in force*, and it is the RANKING's, not the ledger's. `W` is the stepper's net
        wealth entering the decision — a fact about the day, reported under either regime. `c` is
        the knee every rung is actually divided by: the scale off `runtime['objective']` at `t+1`,
        because the rungs are continuations at `t+1` and (under `Running_Wealth`) their reward is
        read there too. It is the FRAME's, because `_rollout_on_stepper` builds its stepper with
        `mirror_scale=False` precisely so a loaded checkpoint's scale — and, under
        `Utility_Scale_Mode='conditional_sim'`, its per-step schedule — survives the roll.

        `x0` places the day ON that utility, at the point the ranking operates: the wealth LEVEL
        under `Fixed`, and the day's INCREMENT under `Running_Wealth`, which is zero before the day
        happens — so the curvature there is read off the LOSS side of the reference
        (`_loss_side_curvature`), the wing an increment objective is dosed by. `in_knee`,
        `u_prime` and `local_ARA` follow it, and `u_is_increment` says which
        regime produced them — the way `_verdict` states `u_mean_is_step_sum`. That distinction is
        the whole reading: at a wealth drifted twenty knees out the level's `local_ARA` is 0 and
        `in_knee` is 0 on every row, which under `Running_Wealth` would report "the ranking is
        affine here" at exactly the decisions the dial has made curved.

        *What the ranking saw.* `curve_net` / `curve_pre` / `curve_charged` are the FULL per-rung
        curve, packed space-separated into three cells so a corridor that narrows the ladder does
        not ragged the header. `range_CE` converts the charged curve's spread into dollars at the
        local `u'` — the one at the ranking's operating point above, so the conversion is in the
        same units the curve is — and `bow_span` is the curve's departure from the chord through
        its net-sorted endpoints; a straight ranking (bow ≈ 0) is one whose argmax can only be a
        corner. `P_band_*` is the chance the day ENDS in the same knee band, per inner draw, at
        three reference books (flat, the standing book, the deepest rung), measured on the same
        quantity `x0` is: the successor LEVEL under `Fixed`, the day's realized INCREMENT under
        `Running_Wealth`.

        *What it was ranking over.* The one-step moments of the fork feeding it, and a SECOND
        independent fork at the same `t`: `fork_disagree` is the seed-disagreement indicator, the
        cheapest honest read on whether a rung won on signal or on inner-MC noise.

        Conventions. Moments are population (÷N) over the flattened outer × inner draws — at the
        single realized path that is exactly the inner ensemble. `dF_net` is the tradable move a
        UNIT of net cover earns, reduced the way `_wealth_step` reduces a book: Σ_i w_i·cs_i·dF_i,
        with `w` the `Allocation_Weights` split in force when one is configured (the schedule IS
        the composition of a rung) and otherwise the uniform split over the live, active legs (with
        no schedule, nothing mandates any other composition)."""
        obj = self.runtime["objective"]
        # The knee the RUNGS divide by — `_score_actions` reads every continuation (and, under
        # Running_Wealth, every reward) at t+1, so t+1 is the step whose entry they consume. Under
        # a flat scale this is the frame's single c and the distinction is inert.
        c = _utility_scale(obj, t + 1)
        R = float(obj.get("reference_wealth", 0.0))
        # The knee is huber's; the shapes without one take a unit of c as the near-reference band.
        delta = (float(obj.get("huber_delta", 1.0))
                 if obj.get("object") == "asymmetricutility_huber" else 1.0)
        w0 = float(W[0])
        # WHERE the ranking operates: the wealth LEVEL under Fixed, the day's INCREMENT (zero
        # before the day happens) under Running_Wealth. Reporting the level under the increment
        # objective is how a curved ranking gets logged as a flat one.
        w_rank = 0.0 if self.running_wealth else w0
        u1, ara = (_loss_side_curvature(self.runtime, t + 1) if self.running_wealth
                   else _utility_local_curvature(w_rank, self.runtime, t + 1))
        x0 = (w_rank - R) / c

        q_live = q_prev if live is None else q_prev * live                          # (1,n_hedge)
        grid = self.aspace.grid_at(t, live)
        a_live = grid if live is None else grid * live                              # (K,n_hedge)
        net = a_live.sum(-1)                                                        # (K,)
        pre, charged, _ = self._score_actions(
            nets, t, grid, market_t1, dF, dL, W, q_live, kappa, live)
        # The rate-limit band the executed argmax searched inside, measured exactly as `_decide`
        # measures it (net under a weight schedule, per-leg without one), with its fallback.
        pick = charged
        if self.aspace.max_trade > 0.0:
            dq = a_live - q_live
            feas = ((dq.sum(-1).abs() <= self.aspace.max_trade) if self.aspace.weights is not None
                    else (dq.abs() <= self.aspace.max_trade).all(-1))
            pick = charged.masked_fill(~feas, float("-inf")) if bool(feas.any()) else charged
        k_star = int(pick.argmax())
        deep = int(net.abs().argmax())

        # P_band at the three reference books, off the SAME charged W1 the ranking is built on.
        probes = torch.stack([torch.zeros_like(q_live[0]), q_live[0], a_live[deep]])
        _, _, W1p = self._score_actions(
            nets, t, probes, market_t1, dF, dL, W, q_live, kappa, live)
        # ...on the same quantity `x0` is: the successor LEVEL, or the day's realized INCREMENT.
        x1 = (W1p - (w0 if self.running_wealth else 0.0) - R) / c
        p_band = ((x1 > -delta) & (x1 < 0.0)).to(torch.float32).mean(-1)

        # The curve's shape: spread in dollars at the local u', and bow off the endpoint chord.
        order = torch.argsort(net, stable=True)
        cs, ns = charged[order], net[order]
        span, width = float(cs[-1] - cs[0]), float(ns[-1] - ns[0])
        chord = (cs[0] + (cs[-1] - cs[0]) * (ns - ns[0]) / width) if width else cs
        # A unit of net cover under the composition the schedule mandates (see docstring).
        w = (self.aspace._weights_at(t, live) if self.aspace.weights is not None
             else self.aspace.active_mask * (1.0 if live is None else live))
        w = w / w.sum()
        dfn = (dF * (w * self.contract_size)).reshape(-1, self.n_hedge).sum(-1)
        dl = dL.reshape(-1)

        # A SECOND fork at the same t — the stream advances, so this is an independent draw of
        # the same conditional law, decided under identical rules.
        dF2, dL2, m2, _, live2 = self._inner_step(t)
        q_b, _ = self._decide(nets, m2, dF2, dL2, W, t, q_prev=q_prev, live=live2, kappa=kappa)
        q_b = q_b * live2

        row = {
            "t": int(t),
            "q_prev_net": float(q_live.sum()),
            "q_chosen_net": float(q_chosen.sum()),
            "argmax_net": float(net[k_star]),
            "deadband_held": int(self.aspace.deadband_sigma > 0.0
                                 and bool(torch.equal(q_chosen[0], q_live[0]))
                                 and not bool(torch.equal(a_live[k_star], q_live[0]))),
            "W": w0, "c": c, "R": R, "x0": x0,
            # Which quantity x0 / in_knee / u_prime / local_ARA / P_band_* are measured on —
            # `_verdict`'s `u_mean_is_step_sum` for the dump.
            "u_is_increment": int(self.running_wealth),
            "in_knee": int(-delta < x0 < 0.0),
            "u_prime": u1, "local_ARA": ara,
            "P_band_flat": float(p_band[0]),
            "P_band_prev": float(p_band[1]),
            "P_band_deep": float(p_band[2]),
            "range_CE": float(charged.max() - charged.min()) / u1,
            "bow_span": (float((cs - chord).abs().max()) / abs(span)) if span else float("nan"),
        }
        for j, name in enumerate(self.hedges):
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
        """Deployment-faithful backtest: roll the frozen policy day-by-day along the
        bundle's (observed) path through `BundleStepper`, which owns the real futures
        accounting (variation margin, financing, per-instrument expiry, forced-flat).
        Each day the book is chosen by `_decide` from the CAUSAL one-step fork forecast
        (`inner_cache[t]`) and the STEPPER'S OWN net wealth — never a verdict wealth
        recursion, so the decision can't be contaminated by mis-accrued P&L. This is
        the JSON-contract interface for running the precomputed diff-ML nets daily.
        Returns {greedy, textbook, nohedge} terminal-P&L stats in the verdict shape, with
        `u_mean_is_step_sum` stating (as `_verdict` does) whether `u_mean` is a terminal utility
        or the running-wealth sum of the roll's per-step ones.

        `Solver.DiffV2_Decision_Curve_Dump` hangs a per-decision CSV off the greedy roll (see
        `_decision_curve_row`). It reads the same objects the decision read and writes a file;
        no branch below it depends on whether it is on."""
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

        q_log = {"greedy": [], "t": []}
        dump_rows = [] if self.curve_dump else None

        def roll(policy):
            """Roll `policy` day-by-day through a fresh `BundleStepper`; returns
            `(terminal P&L, per-step utility sum or None)`.

            `mirror_scale=False`: this rollout is only ever reached with a checkpoint loaded, and
            that checkpoint's utility scale is the value function's own frame — the argmax below
            must read the same `c` the nets were fitted against. Re-mirroring would replace it
            with this world's `c` and decide under a different scale than `_verdict` did.

            Under `Reference_Mode='Running_Wealth'` the reported objective is the SUM of per-step
            rewards, so it is accumulated here too. The partition is the DECISION days the policy
            acted on plus the tail to the terminal — the days this roll actually chooses on, which
            is the finest partition a stepper rollout observes."""
            stepper = BundleStepper(self.bundle, self.runtime, mirror_scale=False)
            # Seed the cost-aware decision q_prev from the OPENING book (the stepper's own
            # positions already open here too, so its realized first-step turnover is measured
            # from q0). Position-free, q0 only shifts first-step cost/P&L; under
            # `DiffV2_Position_State` it is the realized standing book the charge and p' ride.
            q_prev = self.aspace.initial_q(self.B_outer, self.device)
            last = None
            u_sum = w_prev = None
            while not stepper.done:
                t = stepper.time_index - hist
                if stepper.is_decision_step and t in sweep_set:
                    state = stepper._state
                    W = _tracking_error_value(state, self.runtime).to(self.device)     # (B,) net wealth
                    B = W.shape[0]
                    if self.running_wealth:
                        u_sum = (torch.zeros_like(W) if w_prev is None
                                 else u_sum + self._u_step(W, w_prev, t))
                        w_prev = W
                    if policy == "nohedge":
                        q = torch.zeros(B, self.n_hedge, device=self.device)
                    elif policy == "textbook":
                        qt = self._replication_hedge(t)     # (n_hedge,) per-instrument clamped
                        # Also honour the TOTAL position cap (replication clamps per-instrument
                        # only; unscaled it can hold n_hedge x the cap and blow up the stepper's
                        # margin accounting). Scale the book down proportionally if over-limit.
                        tot = float(qt.abs().sum())
                        if self.total_abs_limit > 0.0 and tot > self.total_abs_limit:
                            qt = qt * (self.total_abs_limit / tot)
                        # Obey the corridor mandate too (no schedule ⇒ identity), so the stepper
                        # textbook is the same in-corridor min-var benchmark the verdict rolls.
                        q = self.aspace.project_to_corridor(qt[None].expand(B, self.n_hedge), t)
                    else:
                        dF, dL, m1, _, live = inner_cache[t]
                        kappa_t = self.aspace.kappa(self.tradables_sim, t)
                        q, _ = self._decide(
                            nets, m1, dF, dL, W, t, q_prev=q_prev, live=live,
                            kappa=kappa_t if self.cost_aware or self.position_state else None)
                        q = q * live
                        q_log["greedy"].append(q.mean(0).detach().cpu().tolist())
                        q_log["t"].append(int(t))
                        if dump_rows is not None:
                            dump_rows.append(self._decision_curve_row(
                                nets, t, m1, dF, dL, W, q_prev, q, live,
                                kappa_t if self.cost_aware or self.position_state else None))
                    q_prev = q
                    cur = state["positions"]
                    delta = {n: q[:, j] - cur[n].to(dtype=q.dtype, device=q.device)
                             for j, n in enumerate(self.hedges)}
                    last = stepper.step(delta)
                else:
                    last = stepper.step(None)
            wT = (last["transition_pnl_excess"]
                  + last["transition_liability_value"]).to(torch.float64)
            if u_sum is not None:
                u_sum = u_sum + self._u_step(wT.to(torch.float32), w_prev, self.T_dec)  # tail
            return wT, u_sum

        def stats(wT, u_sum=None):
            p5 = torch.quantile(wT, 0.05)
            cvar5 = wT[wT <= p5].mean() if (wT <= p5).any() else p5
            return {"u_mean": float((self._u(wT.to(torch.float32)) if u_sum is None
                                     else u_sum).mean()),
                    "wT_mean": float(wT.mean()), "wT_p5": float(p5), "wT_cvar5": float(cvar5)}
        out = {p: stats(*roll(p)) for p in ("greedy", "textbook", "nohedge")}
        out["u_mean_is_step_sum"] = bool(self.running_wealth)
        if dump_rows:
            pd.DataFrame(dump_rows).to_csv(self.curve_dump, index=False, float_format="%.6g")
            logging.info("DiffV2_Decision_Curve_Dump: %d decision rows -> %s",
                         len(dump_rows), self.curve_dump)
        out["greedy_q_traj"] = q_log["greedy"]        # per-decision mean book (audit)
        out["greedy_q_t"] = q_log["t"]
        return out

    # ---- driver: warmup (fit + frame lock) -> step (fresh batch) -> finish (verdict) ----
    def _check_load_provenance(self, ck, src, md):
        """Hold one `DiffV2_Load_Value_Fn` member against this run and return its solver version
        (the ensemble collects them). Everything a checkpoint stamps that makes it a DIFFERENT
        FUNCTION OF A DIFFERENT STATE is refused here by name; everything that merely deserves
        saying out loud is warned.

        Refused: the shape contract (`t_min`/`T_dec`/`md`/`hedges`), `DiffV2_Position_State` and
        `DiffV2_Wealth_Free_Value` — each owns an input COLUMN, one added and one removed, so a
        mismatch would otherwise surface as a `load_state_dict` shape error naming a Linear weight
        rather than the key that caused it — the two OBJECTIVE dials (`Reference_Mode` and whether
        a `conditional_sim` scale schedule is in force, neither of which moves a single tensor
        shape, so a mismatch has no other symptom than a quietly wrong policy) — a corridor
        mismatch, and (in
        `_check_action_universe` / `_check_calendar_spread`) a
        narrowed `Allocation_Weights` universe or a `Calendar_Spread_Bps` that moved under
        position state. Warned: a stale `solver_version`, a missing corridor/universe/rate stamp
        under a live one, and a pre-stream frame. Corridor-FREE training spans the widest wealth
        support, so rolling it inside any corridor only restricts to a learned subset (valid);
        a policy trained INSIDE one is off-support under a different or absent corridor."""
        for key, want in (("t_min", self.t_min), ("T_dec", self.T_dec),
                          ("md", md), ("hedges", list(self.hedges))):
            if ck[key] != want:
                raise ValueError(
                    f"DiffV2_Load_Value_Fn checkpoint mismatch on {key!r}: "
                    f"{src} saved {ck[key]!r} vs this run {want!r}")
        ck_obj = ck.get("solver_object", "DiffSolver")
        if ck_obj != getattr(self, "SOLVER_OBJECT", "DiffSolver"):
            raise ValueError(
                f"Solver.Object mismatch: {src} was trained by {ck_obj} but this run's solver "
                f"is {getattr(self, 'SOLVER_OBJECT', 'DiffSolver')} — the bank policy that "
                f"shaped the fitted value is part of the function, not an execution detail")
        if bool(ck.get("position_state", False)) != self.position_state:
            raise ValueError(
                f"DiffV2_Position_State mismatch: {src} was trained with DiffV2_Position_State="
                f"{'Yes' if ck.get('position_state', False) else 'No'} but this run sets "
                f"{'Yes' if self.position_state else 'No'}. The net book fraction p = Sum(q)/Q_max "
                f"is an input column of the fitted value, so the two are different functions of "
                f"different states — retrain, or match Solver.DiffV2_Position_State.")
        want_mode = "running_wealth" if self.running_wealth else "fixed"
        if ck.get("reference_mode", "fixed") != want_mode:
            raise ValueError(
                f"Objective.Reference_Mode mismatch: {src} was trained with Reference_Mode="
                f"{ck.get('reference_mode', 'fixed')!r} but this run sets {want_mode!r}. One "
                f"value function scores TERMINAL wealth and the other a sum of daily increments, "
                f"and they have the same input WIDTH — no shape error would ever surface it — so "
                f"the two are refused here. Retrain, or match Objective.Reference_Mode.")
        # The run side of this one is the SETTING, never the schedule this world measured: a frozen
        # roll is a single path and measures none at all, and it is the checkpoint's that governs
        # it (`_restore_frame`). So the check is 'was this policy fitted under the mode this run
        # declares', which is exactly the consistency question a load can answer.
        if bool(ck.get("utility_scale_schedule")) != self.scheduled_scale:
            raise ValueError(
                f"Objective.Utility_Scale_Mode mismatch: {src} was trained "
                f"{'WITH' if ck.get('utility_scale_schedule') else 'WITHOUT'} a conditional_sim "
                f"scale schedule and this run declares "
                f"{'conditional_sim' if self.scheduled_scale else 'a flat scale'}. The knee is "
                f"what the fitted residual's curvature is measured against, so a per-step "
                f"schedule and a flat scalar are different utilities of the same wealth — "
                f"retrain, or match Objective.Utility_Scale_Mode.")
        if bool(ck.get("wealth_free", False)) != self.wealth_free:
            raise ValueError(
                f"DiffV2_Wealth_Free_Value mismatch: {src} was trained with "
                f"DiffV2_Wealth_Free_Value="
                f"{'Yes' if ck.get('wealth_free', False) else 'No'} but this run sets "
                f"{'Yes' if self.wealth_free else 'No'}. The switch REMOVES the wealth column from "
                f"the residual's inputs, so the two are different functions of different states — "
                f"retrain, or match Solver.DiffV2_Wealth_Free_Value.")
        # The training ARCHITECTURE is part of what a version stamps (the successor chain changed
        # it); an ensemble must not average continuations trained under different ones, and a lone
        # old checkpoint should say what it is.
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
        `DiffV2_Position_State` on a p-support it never covered. Refused by name, because the
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
        raise ValueError(
            f"DiffV2_Load_Value_Fn Allocation_Weights mismatch: {src} was fitted on the ladder "
            f"{saved} but this run searches {want}. The argmax scored a different set of books, "
            f"so the frozen value is queried off the support it learned — retrain, or match the "
            f"Evaluator.Allocation_Weights.")

    def _check_calendar_spread(self, ck, src):
        """`Calendar_Spread_Bps` provenance. Under `DiffV2_Position_State` the charge is a term of
        the regressed TARGET, so a different rate is a different value function and a mismatch is
        refused — the same reasoning that refuses a `DiffV2_Position_State` mismatch itself.
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
                f"DiffV2_Load_Value_Fn Calendar_Spread_Bps mismatch under "
                f"DiffV2_Position_State: {src} was trained at {ck['calendar_spread_bps']!r} and "
                f"this run rolls at {want!r}. Under position state the reposition charge enters "
                f"the regressed target, so the two are different value functions — retrain, or "
                f"match the Evaluator.Calendar_Spread_Bps.")
        logging.warning(
            "DiffV2_Load_Value_Fn: %s was trained at Calendar_Spread_Bps %r and this run rolls "
            "at %r. Position-free the charge shapes SELECTION only, so this is a deliberate "
            "execution re-roll — but the argmax is now ranking against a different friction "
            "than the one it was tuned on.", src, ck["calendar_spread_bps"], want)

    def _restore_frame(self, members):
        """Adopt a loaded checkpoint's FRAME as this run's — the z-frame, the utility scale and
        its per-decision-step schedule — and mirror it onto the runtime, so every reward and every
        ranking reads the frame the nets were fitted against rather than the one this world
        resolved for itself.

        This is `BundleStepper`'s `mirror_scale=False` contract one level up and one step finer.
        That contract says the checkpoint's `c` is part of the value function, so a frozen roll
        must decide under it; a per-step SCHEDULE says exactly the same thing on every decision
        day. The evaluation bundle measured its own at build time and mirrored it on — under a
        stressed world (which is what a frozen eval is FOR) those knees can be anywhere — and this
        is where that measurement is discarded. Nothing downstream re-measures: `_bind` re-asserts
        this frame on every later batch, and the stepper rollout refuses to re-mirror.

        The scalar is AVERAGED over an ensemble (a spread means the members disagree about the
        anchor, and is warned about) — and under a schedule it is averaged over the members'
        TERMINAL KNEES, because the scalar is by contract the terminal read. The SCHEDULES
        themselves are not averaged or reconciled: each member's continuation is read in its own
        frame — `_continuation`'s ensemble branch carries the member's schedule beside its z-stats,
        exactly as it carries its trust region — so the run-level schedule is the primary member's,
        and only the label and terminal reads (which a loaded run never fits) consult it."""
        loaded = members[0]
        self.utility_scale_schedule = loaded.get("utility_scale_schedule")
        self.loaded_displacement = loaded.get("displacement")
        if self.loaded_displacement is not None:
            self.displacement = float(self.loaded_displacement)
        # The scalar IS the terminal read (`hedge_bundle._utility_scale`), so under a schedule it
        # is taken from the members' terminal KNEES rather than from a scalar that only happens to
        # equal them: the ensemble's terminal stats and its members' terminal continuations then
        # remain one statement, the run-level scalar being the mean exactly as it is without a
        # schedule, while each member's own continuation is still read at its own (`_continuation`).
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
        self.runtime["objective"]["utility_scale"] = self.utility_scale
        self.runtime["objective"]["utility_scale_schedule"] = self.utility_scale_schedule

    def warmup(self, bundle=None):
        """Fit the value function on the first (or only) bundle and LOCK the frame: the
        standardization stats, the utility scale and the per-t trust region are computed here and
        frozen for every later batch. Re-fitting them per batch would mean batch 1's C_t and batch
        2's were different functions of different inputs, and the DP recursion would compose
        mismatched frames.

        Bank RNG is deterministic; multi-seed repeats (N solvers, each warmed up once) advance the
        framework's inner-MC Sobol stream, so V_0 spread reflects inner-MC noise.

        `DiffV2_Load_Value_Fn` makes this a FROZEN-POLICY EVAL instead: restore the fitted nets AND
        each function's frame — the train-time standardization stats and utility scale are part of
        the value function, and recomputing them from the (possibly stressed) eval world would
        silently change what the nets compute. Every eval path is unseen by the nets, so the
        verdict rolls over all paths. A LIST of checkpoints = ensemble argmax: each member
        evaluated in its own frame, continuations averaged (cross-fit winner's-curse reduction on
        top of antithetic).

        Every member is held against this run by `_check_load_provenance` first — shape contract,
        position state and corridor refused, stale version / missing corridor stamp / pre-stream
        frame warned."""
        if bundle is not None:
            self._bind(bundle)
        logging.info(
            "DiffSolver setup: n_hedge=%d active=%s T_dec=%d (of %d sim steps; last-live "
            "mtm=[-2]) B_outer=%d levels=%d fit_iters=%d lr=%.3g | "
            "contract_size=%s | q∈[%s, %s] total_abs_limit=%.3g",
            self.n_hedge, self.active, self.T_dec, self.n_steps,
            self.B_outer, self.levels, self.fit_iters, self.lr, self.contract_size.tolist(),
            self.q_lo.tolist(), self.q_hi.tolist(), self.total_abs_limit)

        W_bank, q_bank = self._build_bank(self.gen)
        if self.log_ratio and self.loaded_displacement is None and not self.displacement:
            # THE BIAS IS MEASURED FROM THE FORWARD PASS (user-ruled): the worst book any
            # simulated bank path touches, plus a 10% buffer — wealth stays free to go
            # negative, the shifted log is defined on everything training will see, and the
            # -20-nat backstop only guards the rare inner-fork dip beyond the buffer.
            worst = getattr(self, "_bank_worst", None)
            if worst is None:
                worst = min(float(W_bank[t].min()) for t in range(self.T_dec))
            self.displacement = 1.1 * max(0.0, -worst)
            logging.info("DiffSolver LogWealth displacement: worst bank book %.4g -> "
                         "bias %.4g (aversion %.2f divides it)",
                         worst, self.displacement, self.aversion)
        # Cache the framework inner-MC one-step quantities over the swept range — one
        # inner-MC fork per swept t, reused for the argmax, the bootstrap, AND market_t.
        self.sweep_ts = sweep_ts = list(range(self.t_min, self.T_dec))
        self.inner_cache = inner_cache = {t: self._inner_step(t) for t in sweep_ts}

        # Every path in this batch trains: the out-of-sample world is the HELD-OUT BATCH, an
        # independent draw rather than sibling rows of the same call.
        self.train = train = slice(None)

        # Standardization stats: market/wealth from the TRAIN swept states (no test peeking).
        M = torch.cat([inner_cache[t][3][train] for t in sweep_ts], 0)                # (n_swept*B, md)
        self.m_mean, self.m_std = M.mean(0), M.std(0).clamp_min(1e-6)
        Wall = torch.stack([W_bank[t][train] for t in sweep_ts], 0).reshape(-1)
        self.w_mean, self.w_std = Wall.mean(), Wall.std().clamp_min(1e-6)
        self.md = md = M.shape[-1]
        logging.info(
            "DiffSolver bank: market_dim=%d | swept W∈[%.4g, %.4g] mean=%.4g std=%.4g | "
            "q_rep(t=0)=%s", md, float(Wall.min()), float(Wall.max()),
            float(self.w_mean), float(self.w_std),
            self._replication_hedge(0).detach().cpu().tolist())

        hidden = int(self.cfg.get("diffv2_hidden", 128))
        load_cfg = self.cfg.get("diffv2_load_value_fn", "") or ""
        # A load member is either a checkpoint PATH (JSON contract) or an already-materialised
        # artifact DICT (the same dict solve() returns via value_fn_artifacts / torch.saves) —
        # the eval-from-artifact path treats an in-memory artifact exactly like a loaded file.
        load_members = ([(p if isinstance(p, dict) else str(p)) for p in load_cfg]
                        if isinstance(load_cfg, (list, tuple))
                        else ([load_cfg] if isinstance(load_cfg, dict)
                              else ([str(load_cfg)] if load_cfg else [])))
        loaded = None
        if load_members:
            # Frozen-policy eval: restore the fitted nets AND each member's own frame; a LIST of
            # checkpoints is an ensemble argmax (see docstring).
            members = []
            versions = set()
            for member in load_members:
                # Pre-contract checkpoints predate active_hedge_indices / solver_version /
                # config_hash — they still load: only the frame + net keys below are read.
                ck = member if isinstance(member, dict) else torch.load(member, map_location=self.device)
                src = "<in-memory artifact>" if isinstance(member, dict) else member
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
            loaded = members[0]
            self._restore_frame(members)
            hidden = int(loaded["hidden"])
        in_dim = self._in_dim(md)             # (market | [W] | [p]) — the row `_standardize` builds
        nets = [_DiffV2Residual(in_dim, hidden=hidden).to(self.device)
                for _ in range(self.T_dec)]
        # Per-t trust region for A_t evaluation (set at fit time / restored from checkpoint;
        # None = unclamped, e.g. pre-trust-region checkpoints).
        self.a_bounds = (list(loaded["a_bounds"]) if loaded is not None and loaded.get("a_bounds")
                         else [None] * self.T_dec)
        logging.info("DiffSolver action grid: K=%d actions (levels=%d ^ active=%d)",
                     self.grid_size, self.levels, self.n_active)

        self.nets = nets
        self.hidden = hidden
        self.loaded = loaded
        self.rows = []
        if loaded is not None:
            for net, sd in zip(nets, loaded["state_dicts"]):
                net.load_state_dict(sd)
                net.eval()
            if len(members) > 1:
                # Ensemble: per-member net stacks + frames; _continuation averages members.
                self._ensemble = []
                for ck in members:
                    m_nets = [_DiffV2Residual(in_dim, hidden=int(ck["hidden"])).to(self.device)
                              for _ in range(self.T_dec)]
                    for net, sd in zip(m_nets, ck["state_dicts"]):
                        net.load_state_dict(sd)
                        net.eval()
                    self._ensemble.append(
                        (m_nets, ck["m_mean"], ck["m_std"], ck["w_mean"], ck["w_std"],
                         ck.get("a_bounds"), ck.get("utility_scale_schedule")))
                logging.info("DiffSolver ENSEMBLE argmax over %d value fns", len(members))
            self.worst = float(loaded["max_abs_Y_boot"])
            self.root = {"t": self.t_min, "Y_mean": float(loaded["V_0"]),
                         "q_star_mean": list(loaded["n_star_0"])}
        else:
            self._sweep(W_bank, q_bank)
        # The frame is now locked. Streaming freezes the trust region from here on (later batches
        # report their breach rate instead of re-fitting it); a frozen eval never gets here
        # twice, so its regions stay exactly as this sweep fitted them.
        self._bounds_frozen = True

    def _sweep(self, W_bank, q_bank):
        """One backward pass over the swept range on the CURRENTLY BOUND bundle: fit C_t for
        t = T_dec-1 .. t_min against the cached inner-MC forks. warmup runs it on batch 1, each
        streaming step runs it again on fresh paths with the same nets and optimizers.

        `live_prev` is the leg mask of the step that SET the book standing at t — the one mask
        rule for the position state (`_fit_step`). The first swept step has no predecessor in the
        cache and `nets[t_min]` is fitted but never queried (every `_decide` reads `nets[t+1]`, and
        t_min−1 is outside the sweep), so it falls back to its own."""
        rows = []
        for t in reversed(self.sweep_ts):
            r = self._fit_step(self.nets, W_bank, t, self.inner_cache[t], q_bank,
                               rows=self.train,
                               live_prev=self.inner_cache.get(t - 1, (None,) * 5)[4])
            rows.append(r)
            logging.info(
                "DiffSolver C[t=%d] fitted (%d iters): val_loss=%.4g |Y_boot|=%.4g |A|=%.4g "
                "Y_mean=%+.4g q*_mean=%s", r["t"], r["iters"], r["val_loss"], r["Y_absmean"],
                r["A_absmean"], r["Y_mean"],
                ["%.3f" % v for v in r["q_star_mean"]])
        self.rows = rows
        self.worst = max((r["Y_absmean"] for r in rows if math.isfinite(r["Y_absmean"])),
                         default=0.0)
        self.root = rows[-1] if rows else {"t": self.t_min, "Y_mean": 0.0, "q_star_mean":
                                           [0.0] * self.n_hedge}

    def step(self, bundle):
        """Continue training on a FRESH batch: re-bind, rebuild the exploration bank and the
        inner-MC cache on the new paths, then sweep again with the same nets, the same optimizer
        moments and the frozen frame. Fresh paths per fit step are the point — overfitting to one
        simulated set stops being structurally possible.

        A LOADED checkpoint means evaluation, so there is nothing to continue: `warmup` skipped
        its sweep and the value function is the file's. Sweeping here would fine-tune the
        "frozen" policy on the evaluation world — batch by batch, with the verdict then reported
        for a policy that is not the checkpoint and is never written anywhere."""
        if self.loaded is not None:
            logging.info("DiffSolver streaming step SKIPPED: value fn loaded — frozen-policy "
                         "eval, so this batch trains nothing")
            return
        self._bind(bundle)
        W_bank, q_bank = self._build_bank(self.gen)
        if self.log_ratio and self.loaded_displacement is None and not self.displacement:
            # THE BIAS IS MEASURED FROM THE FORWARD PASS (user-ruled): the worst book any
            # simulated bank path touches, plus a 10% buffer — wealth stays free to go
            # negative, the shifted log is defined on everything training will see, and the
            # -20-nat backstop only guards the rare inner-fork dip beyond the buffer.
            worst = getattr(self, "_bank_worst", None)
            if worst is None:
                worst = min(float(W_bank[t].min()) for t in range(self.T_dec))
            self.displacement = 1.1 * max(0.0, -worst)
            logging.info("DiffSolver LogWealth displacement: worst bank book %.4g -> "
                         "bias %.4g (aversion %.2f divides it)",
                         worst, self.displacement, self.aversion)
        self.inner_cache = {t: self._inner_step(t) for t in self.sweep_ts}
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

    def _policy_artifact(self, nets, md, hidden, V_0, n_star_0, worst):
        """The policy artifact: the fitted nets, the frame they read their inputs through, and the
        provenance every `DiffV2_Load_Value_Fn` guard compares against. Built ONCE per run (in
        `finish`) and returned AND `torch.save`d, so the file and the in-memory dict are the same
        object — the eval-from-artifact path is identical to loading the file.

        Every key `_check_load_provenance` refuses on must be stamped HERE, which is why the two
        live next to each other: a stamp the artifact forgets is a run that refuses its own
        checkpoint (or worse, accepts a foreign one)."""
        return {
            "state_dicts": [net.state_dict() for net in nets],
            "m_mean": self.m_mean, "m_std": self.m_std,
            "w_mean": self.w_mean, "w_std": self.w_std,
            "utility_scale": float(self.runtime["objective"]["utility_scale"]),
            # The per-step knees this policy was fitted against (None = the single scalar). Part of
            # the FRAME, so a load restores it rather than re-measuring; the MODE is what a
            # mismatched load is refused on, because a scheduled and a flat knee are different
            # utilities of the same wealth.
            "utility_scale_schedule": self.utility_scale_schedule,
            # Objective provenance: 'running_wealth' rewards the day's increment and retires the
            # u(W) anchor, so the fitted nets are a different function of the same input WIDTH —
            # the one mismatch class no shape error would ever surface.
            "reference_mode": "running_wealth" if self.running_wealth else "fixed",
            "a_bounds": self.a_bounds,
            "hedges": list(self.hedges),
            "active_hedge_indices": list(self.active),
            # Corridor provenance: the Total_Position_Schedule this policy was trained inside
            # (None = unconstrained). A load under a DIFFERENT corridor fails loud.
            "total_position_schedule": self.aspace.schedule_key(self.aspace.schedule),
            # Action-universe provenance: the Allocation_Weights this policy's argmax searched
            # (None = the Cartesian grid, the widest universe). Same shape of check as the
            # corridor — and `config_hash` cannot stand in for either, because it hashes
            # runtime['solver'] while both of these live under runtime['accounting'].
            "allocation_weights": self.aspace.weights_key(self.aspace.weights),
            # Friction provenance: under DiffV2_Position_State the calendar rate is a term of the
            # charge that entered the regressed TARGET, so a load at a different rate is a
            # different value function, not a different execution policy.
            "calendar_spread_bps": self.aspace.calendar_bps,
            # Position-state provenance: p is an input column, so a load under the other
            # setting is refused BY NAME rather than as a net shape error. The wealth-free
            # switch is the same statement about the W column it REMOVES.
            "position_state": self.position_state,
            "wealth_free": self.wealth_free,
            "T_dec": self.T_dec, "t_min": self.t_min, "md": md, "hidden": hidden,
            # Bank-policy provenance: WHICH solver class shaped the wealth/position states the
            # nets were fitted on, plus DiffSolverV2's forward-pass frame (the per-step Phi
            # curves and dials) so a replay reads the solved construction instead of
            # re-measuring it on different paths.
            "solver_object": getattr(self, "SOLVER_OBJECT", "DiffSolver"),
            # The growth reward's shift, measured off the training bank's worst book — part
            # of the FRAME: a roll under a different shift is a different utility.
            "displacement": getattr(self, "displacement", 0.0),
            "risk_aversion": getattr(self, "aversion", None),
            "phi_curves": getattr(self, "phi_curves", None),
            "solver_version": SOLVER_VERSION,
            "config_hash": self._config_hash(),
            # Frame provenance: WHICH path population locked the frame this policy reads its
            # inputs through, and a stamp over the frame itself (loads compare both).
            "frame_stamp": self._frame_stamp(),
            # Headline echoed so a loaded eval reads it back rather than recomputing.
            "V_0": V_0, "n_star_0": list(n_star_0), "max_abs_Y_boot": worst,
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

        The POLICY ARTIFACT is built ONCE, here: the fitted value function + its frame + the
        provenance stamps. It is the single source returned via `SolverResult` (→
        `HedgeRuntimeExecutionResult.policy_artifact`) AND torch.saved to `DiffV2_Save_Value_Fn` —
        the file and the in-memory dict are byte-for-byte the same object, so the
        eval-from-artifact path (load member = this dict) is identical to loading the file. A
        LOADED run produces none: nothing was fitted, so the only policy in play is the file's and
        re-emitting it would claim this run as its provenance. The JSON boundary rejects a config
        that asks to save one anyway, so `save_path` is empty whenever `loaded` is set.

        The downside verdict — greedy policy vs textbook delta hedge vs no hedge — is OUT-OF-SAMPLE
        by construction: `finish` always runs on a world no fit step saw (the held-out batch, or,
        with a checkpoint loaded, every path, since frozen nets saw none of them). There is no
        in-sample counterpart to report a gap against; the training batches are already gone. With
        a checkpoint loaded and `DiffV2_Stepper_Rollout` set, `_rollout_on_stepper` adds the
        deployment-faithful backtest — the trustworthy P&L for a walk-forward, since the simplified
        `_verdict` wealth recursion mis-accrues expiry."""
        if bundle is not self.bundle:
            self._bind(bundle)
            # The held-out world needs its OWN forks (see docstring).
            self.inner_cache = {t: self._inner_step(t) for t in self.sweep_ts}
        nets, sweep_ts, inner_cache = self.nets, self.sweep_ts, self.inner_cache
        loaded, rows, worst, root = self.loaded, self.rows, self.worst, self.root
        train = self.train
        md, hidden = self.md, self.hidden
        V_0 = float(root["Y_mean"])
        n_star_0 = root["q_star_mean"]
        bounded = math.isfinite(V_0) and worst < 1.0e4
        if loaded is None:
            logging.info(
                "DiffSolver sweep complete: t=%d→%d | max|Y_boot|=%.4g (%s) | "
                "V_0=%+.6g | n_star@t=%d=%s", self.T_dec - 1, self.t_min, worst,
                "BOUNDED" if bounded else "EXPLODED", V_0, root["t"], n_star_0)
        # POLICY ARTIFACT, built ONCE here; a LOADED run produces none (see docstring).
        artifact = None
        save_path = str(self.cfg.get("diffv2_save_value_fn", "") or "")
        if loaded is None:
            artifact = self._policy_artifact(nets, md, hidden, V_0, n_star_0, worst)
            if save_path:
                if not math.isfinite(V_0):
                    raise ValueError(
                        f"refusing to save non-finite value fn to {save_path}: V_0={V_0}")
                tmp = save_path + ".tmp"
                torch.save(artifact, tmp)
                os.replace(tmp, save_path)                       # atomic on POSIX
                logging.info("DiffSolver SAVED value fn to %s (V_0=%+.6g)", save_path, V_0)

        # Downside verdict, OUT-OF-SAMPLE by construction: all rows, held-out world (see docstring).
        verdict = self._verdict(nets, inner_cache, sweep_ts, rows=slice(None))
        # Deployment-faithful backtest through BundleStepper — real futures accounting, decisions
        # off the stepper's own wealth (see docstring).
        stepper_verdict = None
        if loaded is not None and bool(self.cfg.get("diffv2_stepper_rollout", False)):
            stepper_verdict = self._rollout_on_stepper(nets, inner_cache, sweep_ts)
            sg, stb, snh = (stepper_verdict[k] for k in ("greedy", "textbook", "nohedge"))
            logging.info(
                "DiffSolver STEPPER ROLLOUT (frozen policy, realized path, real accounting):\n"
                "  greedy   wT=%+.4e p5=%+.4e cvar5=%+.4e\n"
                "  textbook wT=%+.4e p5=%+.4e cvar5=%+.4e\n"
                "  nohedge  wT=%+.4e p5=%+.4e cvar5=%+.4e",
                sg["wT_mean"], sg["wT_p5"], sg["wT_cvar5"],
                stb["wT_mean"], stb["wT_p5"], stb["wT_cvar5"],
                snh["wT_mean"], snh["wT_p5"], snh["wT_cvar5"])

        g, tb, nh = verdict["greedy"], verdict["textbook"], verdict["nohedge"]
        # PRIMARY metric = the optimization target E[u(W_T)] (already encodes downside aversion
        # via the concave utility). CVaR5 is a secondary tail diagnostic (noisy at small B).
        beats_nh = g["u_mean"] >= nh["u_mean"]
        beats_tb = g["u_mean"] >= tb["u_mean"]
        tail_vs_tb = g["wT_cvar5"] >= tb["wT_cvar5"] - abs(tb["wT_cvar5"]) * 0.05
        logging.info(
            "DiffSolver VERDICT (%s rollout t=%d→T over %d outer paths, start flat):\n"
            "  policy    u(W_T)mean    W_T mean       W_T p5         W_T CVaR5\n"
            "  greedy    %+.5f    %+.4e   %+.4e   %+.4e\n"
            "  textbook  %+.5f    %+.4e   %+.4e   %+.4e\n"
            "  nohedge   %+.5f    %+.4e   %+.4e   %+.4e\n"
            "  → on the OBJECTIVE E[u(W_T)]: beats no-hedge=%s, beats textbook=%s | "
            "tail(CVaR5) competitive w/ textbook=%s",
            "OUT-OF-SAMPLE", self.t_min, self.B_outer,
            g["u_mean"], g["wT_mean"], g["wT_p5"], g["wT_cvar5"],
            tb["u_mean"], tb["wT_mean"], tb["wT_p5"], tb["wT_cvar5"],
            nh["u_mean"], nh["wT_mean"], nh["wT_p5"], nh["wT_cvar5"],
            beats_nh, beats_tb, tail_vs_tb)
        logging.info(
            "DiffSolver greedy positions: mean|q| per instrument=%s | q@t0=%s | q@mid=%s "
            "(textbook q@t0=%s)", ["%.2f" % v for v in verdict["greedy_mean_abs_q"]],
            verdict.get("greedy_q_first"), verdict.get("greedy_q_mid"),
            self._replication_hedge(self.t_min).detach().cpu().tolist())

        return SolverResult(
            solver_name=type(self).__name__,
            actions=torch.tensor(n_star_0),
            values=V_0,
            value_fn_artifacts=artifact,              # the fitted policy (None in eval-from-load runs)
            diagnostics={
                "V_0": V_0,
                "n_star_0": n_star_0,
                "n_star": n_star_0,
                "max_abs_Y_boot": worst,
                "bounded": bool(bounded),
                "root_t": int(root["t"]),
                "per_t": rows,
                "action_grid_size": self.grid_size,
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


# The differential-ML solver `DiffSolver` is the production deliverable; the
# clairvoyant `HindsightDpSolver` is kept as the upper-bound (oracle) benchmark
# track. `run_textbook_benchmark` supplies the lower-bound (min-var / averaging)
# track.
class StreamingSolve:
    """The solve driver: one `Bundle` per simulation batch, handed over as it is built.

    `HedgeMonteCarlo.execute` builds a `Bundle` INSIDE its simulation loop and hands each one
    straight over: `warmup` on batch 1 (which constructs the solver(s) and locks the frame),
    `step` on every later batch (fresh paths, same nets / optimizer moments / frame), and
    `finish` on the final batch — never trained on, so it is the held-out world the verdict and
    the benchmark tracks are measured on.

    Why this shape: the inner-MC fork width is `Batch_Size` rather than the whole simulation, and
    every fit step sees paths no earlier step did, so overfitting to one simulated set is not
    structurally possible. A frozen-policy evaluation is the degenerate stream of length one —
    nothing to fit, so warmup's batch IS the held-out world.

    Multi-seed keeps one persistent solver per seed, all fed the same batches in the same order."""

    documentation = ('Solver', [
        'The solve driver: one bundle per simulation batch, handed over as the calc builds it.',
        '`warmup` on batch 1 (which constructs the solver(s) and locks the frame), `step` on every',
        'later batch (fresh paths, same nets, optimizer moments and frame), and `finish` on the',
        'final batch — never fitted, so it is the held-out world the verdict and the benchmark',
        'tracks are measured on.',
        '',
        'Why this shape: fork width follows `Batch_Size` rather than the whole simulation, and every',
        'fit step sees paths no earlier step did, so overfitting to one simulated set is not',
        'structurally possible.',
        '',
        '`Simulation_Batches` is the stream length. Trained paths are',
        '`(Simulation_Batches - 1) x Batch_Size`, and the minimum is 2. Measured on the platinum',
        'book across four shapes and three seeds, STREAM LENGTH is the lever and batch width is',
        'not: `4096 x 5` doubles the edge over textbook of `4096 x 2`, while `8192 x 5` buys a',
        'little more tail for twice the wall, twice the memory and a three-times worse seed spread.',
        '',
        '!!! warning "A frozen evaluation is the stream of length one"',
        '    `DiffV2_Load_Value_Fn` fits nothing, so `Simulation_Batches` must be exactly 1 — that',
        '    single batch is both the warmup bundle and the held-out world, since frozen nets saw',
        '    none of it. The contract refuses anything else, so there are no `step` batches to',
        '    sweep, and `step` refuses to sweep a loaded net regardless.',
    ])

    def __init__(self, runtime):
        self.runtime = runtime
        self.cfg = runtime["solver"]
        self.solvers = []
        self.trained_batches = 0

    def warmup(self, bundle):
        """Batch 1: build the solver(s) and fit the first backward sweep. The frame (utility
        scale, z-frame, trust region) is locked here and frozen for every later batch."""
        n_seed = max(1, int(self.cfg.get("multi_seed_count", 1)))
        cls = DiffSolverV2 if self.cfg.get("object") == "diffsolverv2" else DiffSolver
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
    of per-step utilities instead, so the rungs are not on one scale and their ORDER says nothing —
    said out loud rather than suppressed, because the per-track table is still the honest record
    and `wT_mean` in the verdict is still comparable."""
    solver_cfg = runtime["solver"]
    obj = solver_cfg["object"]
    have_liability = bundle.liability_mtm is not None
    primary = primary_runs[0]
    comparison = {primary.solver_name: SolverResult.multiseed_summary(primary_runs)}

    # Benchmark tracks — assembled alongside the DiffSolver-family deliverable.
    if obj in ("diffsolver", "diffsolverv2"):
        for flag, label in (("run_hindsight_diagnostic", "hindsight"),
                             ("run_textbook_benchmark", "textbook")):
            if solver_cfg.get(flag) and not have_liability:
                logging.warning("solve_hedge: %s requested but bundle has no "
                                 "liability_mtm — track skipped", label)
        if solver_cfg.get("run_hindsight_diagnostic") and have_liability:
            comparison["HindsightDpSolver"] = SolverResult.multiseed_summary(
                [HindsightDpSolver(bundle, runtime).solve()])     # deterministic — one run
        if solver_cfg.get("run_textbook_benchmark") and have_liability:
            comparison["textbook"] = run_textbook_benchmark(bundle, runtime)

    # Acceptance ordering — hindsight >= DiffSolver >= textbook over the tracks present,
    # with a tiny tolerance for Monte-Carlo noise.
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
        "policy": None,
        "evaluation_output": {
            "solver_name": primary.solver_name,
            "solver_result": primary,
            "actions": primary.actions,
            "values": primary.values,
            "diagnostics": primary.diagnostics,
            "comparison": comparison,
            "ladder": ladder,
        },
        "optimizer_diagnostics": {**primary.diagnostics,
                                  "comparison": comparison, "ladder": ladder},
        "policy_artifact": primary.value_fn_artifacts,
    }


class DiffSolverV2(DiffSolver):
    """Forward-backward DiffSolver: the bank is a CONSTRUCTED asymmetric-hedge policy rather
    than exploration noise around the replication hedge.

    FORWARD PASS — the policy is constructed from the simulated liability distribution. Per
    decision step, the MEASURED delta d_t(book) = P(terminal < max(TODAY'S book, 0) | book)
    is estimated cross-sectionally on the outer paths (bucketed conditional mean, isotonic
    decreasing in the book mark), ONCE — the forward pass is deterministic: the strike rides
    the book, so hedging does not change the measured object and the contracts held are known
    exactly after a single pass. A book sitting on banked gains is as defended as a book at
    par. The delta is never declared and takes NO dial — the seed is clairvoyant, so
    `DiffV2_Risk_Aversion` belongs to the backward DP (the causal proxy for this floor).
    The measured delta maps AFFINELY onto the net band, q_net = hi − d·(hi − lo), and the
    clairvoyant reserve then adds the LEAST extra cover the path needs to never fall under
    the $1/oz floor — by construction.

    BACKWARD PASS — the standard sweep, unchanged, fitted ON the constructed book: the bank's
    wealth and position states are the constructed policy's own roll (plus the usual
    `DiffV2_Bank_Noise_Frac` exploration so the argmax has local support), so every net trains
    where that policy lives and the one-step improvement happens there. The terminal anchor is
    the constructed policy's terminal distribution by construction, and each earlier net still
    inherits its fitted successor.

    The per-step Phi curves are FRAME: solved at warmup on the same batch that locks the
    z-frame, stamped into the checkpoint beside the nets, and a checkpoint written by a
    different solver class is refused by name (the bank policy shaped the fitted value — it is
    part of the function, not an execution detail)."""

    SOLVER_OBJECT = "DiffSolverV2"

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
        return bp[idx - 1] + w * (bp[idx] - bp[idx - 1])

    def _constructed_policy(self, floored=True):
        """ONE deterministic pass — measure, then construct. The moving-strike delta
        d_t(book) = P(terminal book < book at t | book at t) is measured on the FLAT
        liability (the only book that exists before a policy does), and the policy rolls
        once against those curves: by construction the strike rides the book, so hedging
        does not change the measured object and no fixed point is needed — the contracts
        held are known exactly after this single pass. Returns (q per step (B, n_hedge),
        curves per step, terminal book)."""
        L, T = self.liability_sim, self.T_dec
        rep = [self._replication_hedge(t) for t in range(T)]
        dF = [torch.stack([self.tradables_sim[r][t + 1] - self.tradables_sim[r][t]
                           for r in self.hedges], dim=-1) for t in range(T)]
        LT = L[T]
        # THE CLAIRVOYANT RESERVE (user-ruled: the smallest book anywhere on any path is
        # $1/oz — "we can look ahead and add more contracts than we need"). Backward, per
        # path: req_t is the minimum book at t from which the known remaining path can be
        # played without ever dipping under the floor, given each day's best earnable box
        # P&L on the KNOWN move. Days a futures hedge cannot defend (the basket barely
        # moves while the liability does) propagate their need BACKWARD, so the forward
        # pass pre-positions and the cushion arrives before the damage.
        floor = 1.0 * self.leg_volume
        m_t, lo_is, hi_is = [], [], []
        for t in range(T):
            rs = float(rep[t].abs().sum())
            lo_t, hi_t = self.aspace.net_bounds(t)
            lo_is.append(math.ceil(lo_t - 1e-9))
            hi_is.append(math.floor(hi_t + 1e-9))
            if rs < 0.5:
                m_t.append(torch.zeros(self.B_outer, device=self.device))
            else:
                w_rep = rep[t] / rep[t].sum()
                m_t.append(torch.einsum('i,ib->b', w_rep * self.contract_size,
                                        torch.stack([self.tradables_sim[r][t + 1]
                                                     - self.tradables_sim[r][t]
                                                     for r in self.hedges])))
        req = LT.new_full((self.B_outer,), floor)
        reqs = [None] * T
        for t in range(T - 1, -1, -1):
            reqs[t] = req.clone()
            best = torch.maximum(hi_is[t] * m_t[t], lo_is[t] * m_t[t])
            req = torch.maximum(req - (L[t + 1] - L[t]) - best,
                                torch.full_like(req, floor))
        q, W = [], L[0].clone()
        curves = []
        for t in range(T):
            # The strike is TODAY'S BOOK OR $1/oz, whichever is higher (user-ruled: "just
            # above 0" is a dollar over, not zero). The forward pass is hindsight-privileged
            # — it sees the simulated terminals and exists to seed the backward solver — so
            # the floor composite is allowed by construction: under water the question
            # becomes recovery to the floor (d saturates, full cover, and a recovery
            # ratchets the floor up behind it); above water today's book is the strike
            # (the high-water ratchet — gains stay defended, no release).
            cv = self._phi_curve(L[t], LT < L[t].clamp_min(1.0 * self.leg_volume))
            curves.append(cv)
            if float(rep[t].abs().sum()) < 0.5:
                # Less than half a contract of remaining replication delta: the liability is
                # fixed to within dust (the settling rows still carry discount pull-to-par, so
                # exact zero never arrives), and the below-today comparison is float noise. The
                # box's speculative allowance is not a mandate — the constructed book is FLAT
                # (the wealth step below still accrues the settling liability rows).
                qt = torch.zeros(self.B_outer, self.n_hedge, device=self.device)
            else:
                g = self._phi_apply(cv, W)
                lo_t, hi_t = self.aspace.net_bounds(t)
                lo_i, hi_i = math.ceil(lo_t - 1e-9), math.floor(hi_t + 1e-9)
                net = hi_t - g * (hi_t - lo_t)
                # WHOLE contracts, rounded AWAY from zero (up, in cover terms) — you cannot
                # trade 2.5 contracts, and the box is snapped inward so every integer net is
                # realizable.
                net = torch.where(net < 0, net.floor(), net.ceil()).clamp(lo_i, hi_i)
                # THE FLOOR BINDS (user-ruled: the smallest book is $1/oz, not a hope): the
                # seed's one-day hindsight makes it enforceable — it KNOWS tomorrow's move,
                # so on a day the delta base would let tomorrow's book breach the floor, hold
                # instead the smallest position that keeps it above, rounded away toward more
                # earning (the binding days land slightly OVER the floor — integer overshoot).
                # A day whose basket move is too small to defend (a basis-gap day: the
                # liability moves, the futures do not) keeps the delta base — best effort is
                # the honest limit of a futures hedge.
                m, dL = m_t[t], L[t + 1] - L[t]
                # tomorrow's book must clear tomorrow's RESERVE, not just the floor — the
                # reserve carries the look-ahead need of every later undefendable day.
                # reqs[t] IS the requirement at t+1: the backward loop clones before it
                # updates (an off-by-one here defends one day late and breaches exactly on
                # the defenseless day — the gate's business).
                target = reqs[t]
                if t == 0 or not floored:
                    # t0 is the ONE decision deployment shares with the seed: a single real
                    # state, a single position — the uniform honest delta, NO clairvoyance
                    # (user-ruled). The paths that need more catch up at t1, where the
                    # reserve already carries every later need.
                    q.append(self.aspace._largest_remainder(self.aspace.waterfill(
                        rep[t][None].expand(self.B_outer, self.n_hedge).clone(),
                        net.unsqueeze(-1), net.unsqueeze(-1)), net.unsqueeze(-1)))
                    W = self._wealth_step(W, q[-1], dF[t], L[t + 1] - L[t])
                    continue
                # The floor constraint q·m >= need is a HALF-LINE, and the repair is the
                # delta base PROJECTED onto it — the least change, rounded toward the
                # feasible side (an up-day's bound is a LIMIT on the short: rounding away
                # from zero there walks through the floor, which is how a 4-lot short into
                # a known +3σ rally breached a test fixture).
                defensible = m.abs() > 1e-9
                bound = (target - W - dL) / torch.where(defensible, m, torch.ones_like(m))
                net = torch.where(defensible & (m > 0),
                                  torch.maximum(net, bound.ceil()), net)
                net = torch.where(defensible & (m < 0),
                                  torch.minimum(net, bound.floor()), net)
                net = net.clamp(lo_i, hi_i).unsqueeze(-1)
                qt = self.aspace.waterfill(
                    rep[t][None].expand(self.B_outer, self.n_hedge).clone(), net, net)
                qt = self.aspace._largest_remainder(qt, net)
                qt = torch.minimum(torch.maximum(qt, self.q_lo), self.q_hi)
                # EXECUTION-EXACT top-up: the bound above prices the hedge at the
                # rep-weighted basket move, but the executed split deviates (box clamps,
                # integer apportionment), and on days the legs move DIFFERENTLY the realized
                # P&L misses the plan by enough to breach. Re-price the EXECUTED book and
                # top the net up by the least whole contracts that clear the shortfall.
                exec_pnl = (qt * self.contract_size * dF[t]).sum(-1)
                short = target - (W + dL + exec_pnl)
                fix = defensible & (short > 0)
                if bool(fix.any()):
                    # +1: the exact amount can be re-split away again (integer lattice);
                    # one more whole contract is then the LEAST that clears. The floor's
                    # guarantee under whole-contract trading is to within one contract's
                    # day move — granularity is not free.
                    extra = torch.where(fix, ((short / m.abs()).ceil() + 1.0) * torch.sign(m),
                                        torch.zeros_like(m))
                    net = (net.squeeze(-1) + extra).clamp(lo_i, hi_i).unsqueeze(-1)
                    qt = self.aspace.waterfill(
                        rep[t][None].expand(self.B_outer, self.n_hedge).clone(), net, net)
                    qt = self.aspace._largest_remainder(qt, net)
                    qt = torch.minimum(torch.maximum(qt, self.q_lo), self.q_hi)
            q.append(qt)
            W = self._wealth_step(W, qt, dF[t], L[t + 1] - L[t])
        return q, curves, W

    @staticmethod
    def _carry_variance_solve(mu, Sig, c, Q, C):
        """argmin_w [Q²·wᵀΣw + 2Q·wᵀc] / (2C) − Q·wᵀμ  s.t. Σw = 1, w ≥ 0.

        The growth-pinned carry/tracking allocator for ONE step: Σ and c are the covariance of
        the legs' remaining-horizon P&L (per contract, $) with each other and with the
        liability's, μ their means, Q the signed reference net (short < 0), C the capital at
        the step — which fixes the variance/mean exchange rate at the log objective's ½/C, so
        there is no free dial. Active-set over ≤ n legs: solve the equality-constrained
        stationarity, drop the most negative weight, repeat."""
        idx = list(range(int(mu.shape[0])))
        while True:
            S = Sig[idx][:, idx].double()
            S = S + 1.0e-8 * S.diagonal().mean().clamp_min(1e-12) * torch.eye(
                len(idx), dtype=S.dtype, device=S.device)
            Si = torch.linalg.inv(S)
            w = (C / Q ** 2) * (Si @ (Q * mu[idx].double() - (Q / C) * c[idx].double()))
            k = (C / Q ** 2) * (Si @ torch.ones(len(idx), dtype=S.dtype, device=S.device))
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
        L, T = self.liability_sim, self.T_dec
        RL_T = L[T]
        knots, last = [], None
        for t in range(T):
            R = torch.stack([self.contract_size[i]
                             * (self.tradables_sim[self.hedges[i]][T]
                                - self.tradables_sim[self.hedges[i]][t])
                             for i in range(self.n_hedge)])            # (n_h, B) $ per contract
            X = R - R.mean(-1, keepdim=True)
            n = R.shape[-1] - 1
            Sig = (X @ X.transpose(0, 1)) / n
            RL = RL_T - L[t]
            c = (X * (RL - RL.mean())[None]).sum(-1) / n
            alive = Sig.diagonal() > 1.0e-4
            Qb = float(self._replication_hedge(t).sum())
            if not bool(alive.any()) or abs(Qb) < 0.5:
                w = last if last is not None else tuple(
                    1.0 / self.n_hedge for _ in range(self.n_hedge))
            else:
                mu = R.mean(-1)
                big = 1.0e12 * Sig.diagonal().max().clamp_min(1.0)
                Sig = Sig + torch.diag(torch.where(alive, torch.zeros_like(mu), big))
                sol = self._carry_variance_solve(
                    mu, Sig, c, Qb, self._capital(t) / getattr(self, "aversion", 1.0))
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
        constructed policy's neighbourhood is the point of the forward pass."""
        if (self.aspace.mode == "carry_variance" and self.aspace.weights is None
                and not (self.cfg.get("diffv2_load_value_fn") or "")):
            sched = self._carry_variance_weights()
            self.aspace.install_weights(sched)
            logging.info(
                "DiffSolverV2 Carry_Variance allocator: %d knot(s); w(t=0)=%s w(mid)=%s",
                len(sched), sched[0][1] if sched else None,
                sched[len(sched) // 2][1] if sched else None)
        q_star, curves, WT = self._constructed_policy()
        # HALF THE CLASSROOM IS THE DROWNING: the floored seed keeps every training book
        # above the floor, so a deep-under-water state never appears — and the first trained
        # roll, queried exactly there in the crash month, extrapolated to MAX LONG and
        # pinned. The value function must see both the floor and what it prevents: half the
        # paths roll the clairvoyant seed, half the UNDEFENDED delta base, so the log
        # penalty is fitted where the bad states actually live.
        q_base, _, _ = self._constructed_policy(floored=False)
        half = self.B_outer // 2
        for t in range(self.T_dec):
            q_star[t] = torch.cat([q_star[t][:half], q_base[t][half:]], dim=0)
        self.phi_curves = [(bx.detach().cpu(), bp.detach().cpu()) for bx, bp in curves]
        phi0 = float(self._phi_apply(curves[0], self.liability_sim[0][:1])[0])
        logging.info(
            "DiffSolverV2 forward pass: aversion=%.2f d_0=%.3f q*_0 net=%.2f "
            "terminal book mean=%.4g frac<0=%.3f",
            self.aversion, phi0,
            float(q_star[0].sum(-1).mean()), float(WT.mean()), float((WT < 0).float().mean()))
        L = self.liability_sim
        W = L[0].clone()
        q_prev = self.aspace.initial_q(self.B_outer, self.device)
        W_list, q_list = [], []
        rng = (self.q_hi - self.q_lo)
        mask = self.aspace.active_mask
        for t in range(self.T_dec):
            W_list.append(W.clone())
            q_list.append(q_prev.clone())
            noise = self.noise_frac * rng * torch.randn(
                self.B_outer, self.n_hedge, generator=gen, device=self.device)
            q = torch.minimum(torch.maximum(q_star[t] + noise * mask, self.q_lo), self.q_hi)
            if self.aspace.schedule is not None:
                q = self.aspace.project_to_corridor(q, t)
            dF = torch.stack(
                [self.tradables_sim[ref][t + 1] - self.tradables_sim[ref][t] for ref in self.hedges],
                dim=-1)
            W = self._wealth_step(W, q, dF, L[t + 1] - L[t])
            q_prev = q
        if self.log_ratio:
            bw = torch.stack(W_list)
            # the WORST book includes the terminal wealth (W after the last step) — in a
            # crash month the terminal is often the global minimum, and a bias measured
            # only at decision times would leave the last fit step's states uncovered
            self._bank_worst = float(torch.minimum(bw.min(), W.min()))
            logging.info(
                "DiffSolverV2 bank floor (LogWealth domain): frac(W <= floor) = %.4f "
                "min W (incl terminal) = %.4g",
                float((bw <= self.w_floor).float().mean()), self._bank_worst)
        return W_list, q_list
