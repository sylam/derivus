"""The hedging stack's world side: the JSON boundary, the simulated bundle, the
environment, and the objective/utility stack.

`construct_hedge_runtime` is the ONE JSON → hedging-runtime boundary: it reads the
`Hedging_Problem` block and returns the normalized runtime dict every hedging consumer indexes
by key — canonical lowercased modes, the instrument / cash-account / hedge name sets,
per-instrument metadata, the accounting rules (position limits, turnover cost, spreads, margin
funding, corridor), the objective, the solver config and the portfolio state. Everything is
validated HERE and nowhere else — past this boundary the runtime is the contract. It also owns
the privileged-factor naming convention (what each stochastic process publishes as market
state) and `per_contract_kappa` / `turnover_charge`, the single turnover-cost rules the solver,
the environment and the diagnostic CSV writer all price frictions through.

`Bundle` is the simulated hedging world: `Bundle.from_batch` assembles the per-batch tensor
blocks the HedgeMonteCarlo simulator produces into one object (four construction stages —
tradables/time-axis, liability, factors, frame), and every downstream consumer reads it by
attribute. It owns the LOCKED frame (`utility_scale`, the per-step vol series), the sim-grid
views the solver indexes by `t`, and the inner-MC fork closures the calc attaches.

`BundleStepper` is the environment: the real futures/cash accounting (variation margin,
financing, each leg's turnover charge, IM funding, expiry flatten) advanced one day at a
time under any explicit policy. `run_hedge_execution` dispatches by Execution_Mode:
'solve_hedge' → the differential-ML solver (hedge_solver), 'simulate_only' → the no-trade
baseline. Module level keeps only the pure math both sides share: the wealth law, the utility
shapes, the friction debits, and the state-based portfolio reads.
"""

from __future__ import annotations

import logging
import math
import os
import time
from copy import deepcopy
from typing import Any, Dict, Mapping, Optional

import pandas as pd
import torch

from . import instruments, utils
from .schema import DealFields


# Rolling window (business days) for the realized-vol series feeding the symlog utility scale
# and the state-dependent bid/offer spread.
PRICE_ZSCORE_WINDOW = 20


# Terminal-utility SHAPES: each maps dollars→O(1) utility via the deal scale c (x = W/c) and
# lives in utility space (the DP / value-fn recursion). symlog is odd/symmetric, huber and cara
# downside-averse; any Object outside this tuple takes the identity (no-op) path. LogWealth is
# NOT one — it consumes no scale c — and a solve admits it beside these by name.
_UTILITY_OBJECTS = (
    "asymmetricutility_symlog", "asymmetricutility_huber", "asymmetricutility_cara")


def primary_spot(stoch_factors):
    """The key of the spot a hedge reads as its one underlying: the CommodityPrice factor exposing a
    revealed sufficient statistic (a non-empty `privileged_layout`), else the only CommodityPrice;
    refused by name where neither is unique."""
    spots = [k for k in stoch_factors if k.type == 'CommodityPrice']
    keys = [k for k in spots if stoch_factors[k].privileged_layout(stoch_factors[k].param)] or spots
    if len(keys) != 1:
        raise ValueError(
            f"the hedge reads one primary spot, the CommodityPrice exposing a revealed sufficient "
            f"statistic, else the only CommodityPrice; this world simulates "
            f"{[utils.check_tuple_name(k) for k in spots]}")
    return keys[0]


def _privileged_name(factor_name, attr_name, stoch_factors):
    """How a process's published state coordinate is keyed. Multi-commodity runs (more than one
    distinct primary factor name) prefix the attribute with `<factor>_` to disambiguate."""
    multi = len({f.name[0] for f in (stoch_factors or {})}) > 1
    return f'{factor_name.lower()}_{attr_name}' if multi else attr_name


def derive_privileged_layout(stoch_factors):
    """The `{name: dim}` schema, built by asking each live stoch-factor process what it emits via
    `type(process).privileged_layout(process.param)`."""
    layout = {}
    for factor, process in (stoch_factors or {}).items():
        for attr_name, dim in type(process).privileged_layout(process.param).items():
            layout[_privileged_name(factor.name[0], attr_name, stoch_factors)] = int(dim)
    return layout


def assemble_privileged_factors(privileged_factor_blocks, stoch_factors):
    """Per-batch privileged-factor tensors concatenated into one dict for the bundle. Input is
    keyed by `(factor_name, attr_name)`; output keys match `derive_privileged_layout`'s schema."""
    return {
        _privileged_name(factor_name, attr_name, stoch_factors): torch.cat(blocks, dim=1)
        for (factor_name, attr_name), blocks in privileged_factor_blocks.items()
    }


def privileged_block(privileged_factors, stoch_factors, attr_name):
    """`(factor, process, block)` for the first live factor that PUBLISHES `attr_name` in its
    privileged layout and has a matching assembled block, or `(None, None, None)`."""
    for factor, process in (stoch_factors or {}).items():
        if attr_name not in type(process).privileged_layout(process.param):
            continue
        block = privileged_factors.get(_privileged_name(factor.name[0], attr_name, stoch_factors))
        if block is not None:
            return factor, process, block
    return None, None, None


def _flatten_deals(config: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """`{DealType: {name: params}}` (the JSON object-map form) → `{name: {deal_type, params}}`."""
    return {str(name): {"deal_type": str(deal_type), "params": deepcopy(dict(params))}
            for deal_type, deals in config.items() for name, params in deals.items()}


def _instrument_metadata(entry, hedge):
    """Per-tradable metadata the engine reads: the currency cash routes by, the deal's parameters
    and, on a hedge leg, the `Contract_Size` its deal class declares."""
    params = entry["params"]
    # _flatten_deals already deep-copied `params` per entry
    meta = {"currency": params.get("Currency"), "params": params}
    if hedge:
        meta["contract_size"] = float(
            DealFields(params, getattr(instruments, entry["deal_type"]))["Contract_Size"])
    return meta


def _bid_offer_spread(evaluator_config: Mapping[str, Any]):
    """`Evaluator.Bid_Offer_Spread_Bps` normalized to `(scalar_bps, spec)`.

    The key is EITHER the scalar FULL quoted bid-offer spread in bps, applied to every instrument
    (each trade pays HALF of it — mid to touch), OR a spec dict for maturity- and vol-dependent
    spreads:

        {"Default_Bps": d,
         "Per_Instrument": {name: base_bps, ...},
         "Vol_Scale": {"Ref_Vol": r, "Beta": b}}

    The effective full spread for `name` at annualized vol σ_t is
    `base_bps[name] · (σ_t/Ref_Vol)**Beta`, where `base_bps[name]` falls back to `Default_Bps` and
    the vol factor is 1 when Vol_Scale is absent, Beta == 0, or σ_t is unknown. `spec` is None in
    the scalar case; `scalar_bps` is the `Default_Bps` in the dict case."""
    raw = evaluator_config["Bid_Offer_Spread_Bps"]
    if not isinstance(raw, Mapping):
        return float(raw), None
    default_bps = float(raw.get("Default_Bps", 0.0))
    vs = raw.get("Vol_Scale") or {}
    return default_bps, {
        "default_bps": default_bps,
        "per_instrument": {str(k): float(v) for k, v in (raw.get("Per_Instrument") or {}).items()},
        "vol_scale": ({"ref_vol": float(vs["Ref_Vol"]), "beta": float(vs.get("Beta", 0.0))}
                      if vs else None)}


def _position_schedule(evaluator_config: Mapping[str, Any]):
    """Optional per-decision-step corridor on the SIGNED total position Σq_i. A list of
    `{Step, Min_Total, Max_Total}` knots (piecewise-constant between knots): at sim-grid
    decision step t the signed book total must lie within [Min_Total, Max_Total] of the
    rightmost knot with `Step <= t`. Absent → None (no corridor). Returns a sorted tuple of
    `(step, min_total, max_total)` with strictly ascending, non-negative steps and
    Min_Total <= Max_Total per knot."""
    raw = evaluator_config.get("Total_Position_Schedule")
    if not raw:
        return None
    knots = sorted(
        (int(k["Step"]), float(k["Min_Total"]), float(k["Max_Total"])) for k in raw)
    if knots[0][0] < 0:
        raise ValueError(
            f"Total_Position_Schedule Step must be >= 0; got {knots[0][0]}")
    for (a, _, _), (b, _, _) in zip(knots, knots[1:]):
        if b <= a:
            raise ValueError(
                f"Total_Position_Schedule Steps must be strictly ascending; got {a} >= {b}")
    for step, lo, hi in knots:
        if lo > hi:
            raise ValueError(
                f"Total_Position_Schedule knot at Step {step}: Min_Total {lo} > Max_Total {hi}")
    return tuple(knots)


def _allocation_weights(evaluator_config: Mapping[str, Any], hedge_names):
    """Optional ALLOCATION SCHEDULE, which FACTORS the action space: the solver chooses one signed
    NET cover on a fine ladder and this splits it across the hedge legs, instead of searching the
    Cartesian product of per-leg levels. Absent → None (the Cartesian grid).

    Declared long — a list of `{Step, Instrument, Weight}` rows, the rows sharing a `Step` being
    one KNOT, piecewise-constant in the decision step like `Total_Position_Schedule`.

    Returns a sorted tuple of `(step, (w_0, ..., w_{n-1}))` in `names['hedges']` order, each knot
    NORMALIZED to sum 1. A leg no row names gets 0, which is how an expired leg leaves the ladder.
    Weights must be non-negative and already sum to ~1, so normalizing removes float dust rather
    than reinterpreting intent, and the ladder's `Σ_i round(Q·w_i) == Q` apportionment is exact."""
    raw = evaluator_config.get("Allocation_Weights")
    if not raw:
        return None
    by_step: Dict[int, Dict[str, float]] = {}
    for row in raw:
        step, name, weight = int(row["Step"]), str(row["Instrument"]), float(row["Weight"])
        if step < 0:
            raise ValueError(f"Allocation_Weights Step must be >= 0; got {step}")
        if weight < 0.0:
            raise ValueError(
                f"Allocation_Weights Weight must be >= 0; got {weight} for {name} at Step {step}")
        if name not in hedge_names:
            raise ValueError(
                f"Allocation_Weights names {name!r}, which is not a hedge leg {list(hedge_names)}")
        knot = by_step.setdefault(step, dict.fromkeys(hedge_names, 0.0))
        if knot[name] != 0.0:
            raise ValueError(
                f"Allocation_Weights names {name!r} twice at Step {step}: a last-wins overwrite "
                f"would silently drop the earlier weight")
        knot[name] = weight
    knots = []
    for step, weights in sorted(by_step.items()):
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Allocation_Weights knot at Step {step} sums to {total}, not 1")
        knots.append((step, tuple(weights[n] / total for n in hedge_names)))
    return tuple(knots)


def _spot_price_history(hedging_problem: Mapping[str, Any], lookback: int,
                        referenced_commodities: tuple) -> Dict[str, Dict[str, Any]]:
    """Realized spot history per commodity — the rolling-feature lookback the bundle prepends.
    OPTIONAL: absent it the utility scale falls back to the calibrated market data and the prefix
    no-ops, so an empty history is returned rather than demanding an entry per commodity. A
    PARTIAL history (some but not all referenced commodities) IS an error, as are ragged
    dates/prices, a series shorter than the lookback, non-ascending dates, and two commodities
    whose date axes disagree."""
    raw_history = hedging_problem["Portfolio_State"]["Spot_Price_History"]
    if not raw_history:
        return {}
    normalized: Dict[str, Dict[str, Any]] = {}
    for commodity, payload in raw_history.items():
        name = str(commodity)
        dates_raw = payload.get("Dates", ())
        prices_raw = payload.get("Prices", ())
        if len(dates_raw) != len(prices_raw):
            raise ValueError(
                f"Spot_Price_History['{name}']: Dates and Prices must have equal length "
                f"({len(dates_raw)} vs {len(prices_raw)})")
        if len(dates_raw) < lookback:
            raise ValueError(
                f"Spot_Price_History['{name}']: needs at least "
                f"History_Lookback_Business_Days={lookback} entries, got {len(dates_raw)}")
        dates = tuple(dates_raw)
        for i in range(1, len(dates)):
            if dates[i] <= dates[i - 1]:
                raise ValueError(
                    f"Spot_Price_History['{name}']: Dates must be strictly ascending; "
                    f"found {dates[i - 1]} >= {dates[i]} at index {i}")
        normalized[name] = {"dates": dates, "prices": tuple(float(p) for p in prices_raw)}
    missing = tuple(c for c in referenced_commodities if c not in normalized)
    if missing:
        raise ValueError(
            f"Spot_Price_History missing entries for referenced commodities: {missing}")
    names = list(normalized)
    for other in names[1:]:
        if normalized[other]["dates"] != normalized[names[0]]["dates"]:
            raise ValueError(
                f"Spot_Price_History['{other}'].Dates must match "
                f"Spot_Price_History['{names[0]}'].Dates exactly")
    return normalized


def _solver_config(solver_config: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """Normalize the `Solver` block (Execution_Mode='solve_hedge'). Accepts None for non-solve
    modes; requires `Object` — 'DiffSolver' | 'DiffSolverV2'.

    Beyond the per-t residual-net Adam iters / lr: `DiffV2_Bank_Noise_Frac` is bank q-exploration
    noise as a fraction of each instrument's [Min, Max] range, and `Active_Hedge_Indices` selects
    the hedge instruments whose action axis VARIES in the grid (others pinned to 0); None = all.

    `DiffV2_Risk_Kappa` — downside-aware action SELECTION: at the argmax, score each action by
    `mean(C) - kappa · downside-semidev(C)` over the inner-MC, de-risking only the bad-tail
    actions. 0 = off, bit-identical to the plain E[C] argmax.

    `DiffV2_Per_Column_Grad_Norm` — twin-loss differential normalization per INPUT COLUMN (a
    lambda_j vector). 'No' = the pooled scalar variance, where one fat-tailed column deflates the
    constraint for all columns.

    `DiffV2_Position_State` — the FRICTIONAL Bellman. The fitted value is position-free and
    cost-free by default, so a repositioning charge levied at the argmax is a one-day toll the
    value function never remembers and no no-trade region can form. Set, the signed net book
    fraction `p = Sum(q)/Total_Position_Abs_Limit` joins the net's input after the market columns
    (and after W, unless `DiffV2_Wealth_Free_Value` removes that column), the charge is subtracted
    from the wealth that becomes the regressed TARGET, the successor is read at the book the chosen
    action leaves standing, and a `Force_Flat_At_End` mandate makes the last decision pay for its
    own unwind. A cap of 0 fails loud — it is the scale p is measured in. A checkpoint carries the
    setting, and a mismatched load is refused by name.

    `DiffV2_Wealth_Free_Value` — removes the W column from the residual net, so
    `A_t = A_t(market[, p])` and the ranking's whole wealth dependence is the utility anchor
    u(W1)'s. Requires `DiffV2_Position_State` (without p the residual is one constant per state)
    and refuses `DiffV2_Risk_Kappa` > 0 (the semideviation leaks the residual's inner-draw
    dispersion back into the ranking). Stamped on checkpoints.

    `DiffV2_Save_Value_Fn` / `DiffV2_Load_Value_Fn` — value-function persistence: save the fitted
    nets (plus standardization stats and utility scale) after the backward sweep, or load them and
    SKIP training for a frozen-policy eval. Load accepts a LIST of checkpoint paths for an
    ENSEMBLE argmax: each member is evaluated in its own standardization frame and the
    continuations averaged before the argmax. Train and evaluate are SEPARATE runs - loading
    no-ops every fit step under streaming, and setting both keys at once raises."""
    if solver_config is None:
        return None
    if "Object" not in solver_config:
        raise ValueError("Hedging_Problem['Solver'] requires an 'Object' field")
    gamma = float(solver_config["DiffV2_Risk_Aversion"])
    if not 0.1 <= gamma <= 10.0:
        raise ValueError(f"DiffV2_Risk_Aversion must be in [0.1, 10.0] (the DP's aversion: it "
                         f"divides the LogWealth capital line — the clairvoyant seed's floor "
                         f"has no causal equivalent, this dial is its proxy); got {gamma}")
    return {
        # 'diffsolverv2' is its OWN object: the constructed put-delta bank is part of the fitted
        # function, not a spelling of DiffSolver
        "object": str(solver_config["Object"]).lower(),
        "multi_seed_count": int(solver_config["Multi_Seed_Count"]),
        # Backward-sweep depth: fit C_t for t in [t_outer-2 .. t_min]. 0 = full sweep to the
        # initial decision; t_min near t_outer-1 = a shallow (bounded) sweep.
        "t_min": int(solver_config["T_Min"]),
        # Greedy-decision action grid (levels per hedge axis) + batched-argmax chunk size.
        "training_action_grid_levels_per_axis":
            int(solver_config["Training_Action_Grid_Levels_Per_Axis"]),
        "training_action_chunk_size": int(solver_config["Training_Action_Chunk_Size"]),
        # DiffSolver knobs (see docstring): per-t residual-net Adam iters / lr, bank noise.
        "diffv2_fit_iters": int(solver_config["DiffV2_Fit_Iters"]),
        "diffv2_lr": float(solver_config["DiffV2_LR"]),
        "diffv2_bank_noise_frac": float(solver_config["DiffV2_Bank_Noise_Frac"]),
        # The DP's aversion: divides the capital line in the LogWealth reward. The forward pass
        # takes no dial.
        "diffv2_risk_aversion": gamma,
        # The drift tripwire: the tail multiple on the validation-measured null dispersion of the
        # cumsum, and the strength of the on-trip correction toward the realized drift (0 = report)
        "diffv2_drift_threshold_sigmas":
            float(solver_config["DiffV2_Drift_Threshold_Sigmas"]),
        "diffv2_drift_beta": float(solver_config["DiffV2_Drift_Beta"]),
        # the stepper roll's handover: the greedy policy holds the passive book from this date
        "diffv2_passive_from": str(solver_config["DiffV2_Passive_From"] or ""),
        "diffv2_first_decision_from":
            str(solver_config["DiffV2_First_Decision_From"]).lower(),
        # Residual-net regularization: the twin-loss pathwise-gradient match (diffv2_lambda_grad)
        # in STANDARDIZED space, plus optional weight decay for outer-path-starved problems.
        "diffv2_weight_decay": float(solver_config["DiffV2_Weight_Decay"]),
        "diffv2_hidden": int(solver_config["DiffV2_Hidden"]),
        "diffv2_lambda_grad": float(solver_config["DiffV2_Lambda_Grad"]),
        # Downside-aware SELECTION: mean(C) - kappa · semidev(C) at the argmax. 0 = off.
        "diffv2_risk_kappa": float(solver_config["DiffV2_Risk_Kappa"]),
        # ...the semideviation of the continuation this many decisions ahead, the book held
        "diffv2_risk_horizon": int(solver_config["DiffV2_Risk_Horizon"]),
        # Quadratic churn charge at the argmax and the label argmax; the fitted value targets
        # stay cost-free.
        "diffv2_churn_lambda": float(solver_config["DiffV2_Churn_Lambda"]),
        # Successor-inheritance extras: parameter proximity to the fitted neighbour, and the
        # early-termination tolerance an inherited fit stops at (the anchor runs the budget).
        "diffv2_temporal_proximity":
            float(solver_config["DiffV2_Temporal_Proximity"]),
        "diffv2_fit_tol": float(solver_config["DiffV2_Fit_Tol"]),
        # Cost-aware EXECUTION: the verdict rollout charges the L1 repositioning cost
        # (Transaction_Cost_Per_Unit + half Bid_Offer_Spread_Bps) at the argmax, trading
        # expected value against the cost of getting there. Training stays cost-free.
        "diffv2_cost_aware_argmax":
            solver_config["DiffV2_Cost_Aware_Argmax"] == "Yes",
        # FRICTIONAL Bellman: the net book fraction becomes a state coordinate and the charge
        # enters the fitted TARGET (see docstring). 'No' = the position-free value.
        "diffv2_position_state":
            solver_config["DiffV2_Position_State"] == "Yes",
        # The book column's own pathwise label under the position state: the charge's slope in the
        # standing book joins the twin loss. 'No' = the column fitted by the value term alone.
        "diffv2_book_gradient_label":
            solver_config["DiffV2_Book_Gradient_Label"] == "Yes",
        # The DiffSolverV2 bank's books as rungs of the ladder on its live legs, the noise on the
        # net. 'No' = the net water-filled onto the replication hedge's shape over the active legs.
        "diffv2_bank_along_ladder":
            solver_config["DiffV2_Bank_Along_Ladder"] == "Yes",
        # TD(k): the value label realises k steps along the greedy policy on the outer path before
        # it bootstraps. 1 = the one-step bootstrap.
        "diffv2_label_horizon": int(solver_config["DiffV2_Label_Horizon"]),
        # Load a checkpoint fitted on a different decision horizon: per-step structures clamp
        # to the saved range (the tail repeats the last fitted step). 'No' = exact-shape loads.
        "diffv2_load_horizon_pad":
            solver_config["DiffV2_Load_Horizon_Pad"] == "Yes",
        # Dimensionless state coordinates: price -> log-return vs calibrated t0 spot, basis ->
        # fraction of it, wealth -> fraction of t0 book notional. A different coordinate system
        # is a different function; checkpoints stamp it and refuse a mismatch.
        "diffv2_returns_state":
            solver_config["DiffV2_Returns_State"] == "Yes",
        # WEALTH-FREE residual: drop the W input column of the value net, so the ranking's
        # wealth dependence is the utility anchor's alone. 'No' = the wealth-bearing residual.
        "diffv2_wealth_free_value":
            solver_config["DiffV2_Wealth_Free_Value"] == "Yes",
        # Deployment-faithful backtest: with a frozen policy loaded, roll it day-by-day on the
        # observed path via BundleStepper (real futures accounting; decisions off the stepper's
        # own wealth). Exposes diagnostics['stepper_verdict']. 'No' = only the fast _verdict.
        "diffv2_stepper_rollout":
            solver_config["DiffV2_Stepper_Rollout"] == "Yes",
        # Pure DIAGNOSTIC: a CSV of the stepper rollout's per-step ranking curve and the local
        # shape of the utility it was ranked under. '' (or absent) = off, and nothing about the
        # roll changes when it is on.
        "diffv2_decision_curve_dump":
            str(solver_config["DiffV2_Decision_Curve_Dump"] or ""),
        # Per-input-column greek normalization in the twin loss. 'No' = legacy pooled variance.
        "diffv2_per_column_grad_norm":
            solver_config["DiffV2_Per_Column_Grad_Norm"] == "Yes",
        # Save the fitted nets, or load (a path, or a LIST for ensemble-argmax) and skip training.
        "diffv2_save_value_fn": str(solver_config["DiffV2_Save_Value_Fn"] or ""),
        "diffv2_load_value_fn":
            ([str(p) for p in solver_config["DiffV2_Load_Value_Fn"]]
             if isinstance(solver_config["DiffV2_Load_Value_Fn"], (list, tuple))
             else str(solver_config["DiffV2_Load_Value_Fn"] or "")),
        # Convergence declarations: the value's optimism over its held-out greedy utility a fit
        # saves and a roll loads, and how far an ensemble's members' swept books may disagree.
        "diffv2_max_value_optimism": float(solver_config["DiffV2_Max_Value_Optimism"]),
        "diffv2_max_member_disagreement":
            float(solver_config["DiffV2_Max_Member_Disagreement"]),
        "active_hedge_indices": (None if solver_config["Active_Hedge_Indices"] is None
                                 else list(solver_config["Active_Hedge_Indices"])),
        # the constructed policy's wealth floor, per unit of the liability's leg volume
        "diffv2_book_floor_per_unit": float(solver_config["DiffV2_Book_Floor_Per_Unit"]),
        # Benchmark tracks assembled alongside the DiffSolver deliverable (hindsight upper
        # bound / textbook lower bound).
        "run_hindsight_diagnostic": solver_config["Run_Hindsight_Diagnostic"] == "Yes",
        "run_textbook_benchmark": solver_config["Run_Textbook_Benchmark"] == "Yes",
    }


def _objective_config(objective_config):
    """Normalize the `Objective` block, None without one. Its dials are dispatched on downstream by
    EQUALITY (`== 'running_wealth'`, `== 'conditional_sim'`), so a near-miss spelling would quietly
    mean the default: `Reference_Mode` is validated here, `Utility_Scale_Mode` refuses its own.
    LogWealth's reward is the day's log ratio of capital + MTM, running-wealth shaped whatever
    `Reference_Mode` says."""
    if objective_config is None:
        return None
    reference_mode = str(objective_config["Reference_Mode"]).lower()
    if reference_mode not in ("fixed", "running_wealth"):
        raise ValueError(
            f"Unsupported Objective.Reference_Mode: "
            f"{objective_config['Reference_Mode']!r}. Supported: 'Fixed' (the utility "
            f"is applied to TERMINAL wealth) | 'Running_Wealth' (to the day's wealth increment). "
            f"The value is dispatched on by equality, so a near-miss would silently run 'Fixed'.")
    if str(objective_config["Object"]).lower() == "logwealth":
        reference_mode = "running_wealth"
    # every outer path shares L_0, so the raw dispersion at t=0 is exactly 0: a non-positive floor
    # makes c_0 = 0, x = (W-R)/c_0 infinite, and every label from step 0 a NaN
    floor_frac = float(objective_config["Utility_Scale_Floor_Frac"])
    if not floor_frac > 0.0:
        raise ValueError(
            f"Objective.Utility_Scale_Floor_Frac must be > 0; got {floor_frac}. It floors the "
            f"conditional_sim scale schedule at a fraction of its terminal entry, and the first "
            f"steps carry no dispersion at all, so a zero floor is a zero knee and a NaN reward.")
    if floor_frac > 1.0:
        logging.warning(
            "Objective.Utility_Scale_Floor_Frac=%.4g is above 1: the floor then exceeds the "
            "schedule's terminal entry, which flattens every knee onto one number and retires "
            "the per-step shape the mode exists for.", floor_frac)
    return {
        # canonical lowercased form: every dispatch site compares against the lowercase literal
        "object": str(objective_config["Object"]).lower(),
        # utility-transform scale, consumed by any utility Object; `utility_scale` is mirrored
        # in from the frame (hedge_bundle.mirror_frame)
        "utility_scale_mode": str(objective_config["Utility_Scale_Mode"]).lower(),
        "utility_scale_explicit":
            (None if objective_config.get("Utility_Scale_Explicit") is None
             else float(objective_config["Utility_Scale_Explicit"])),
        # Floor of the 'conditional_sim' schedule, as a fraction of its terminal entry —
        # inert under every other mode.
        "utility_scale_floor_frac": floor_frac,
        # WHAT the utility is applied to: terminal wealth ('fixed'), or the DAY's wealth
        # increment ('running_wealth', a per-step reward the DP sums down the recursion).
        "reference_mode": reference_mode,
        # The benchmark wealth the utility is measured against, in DOLLARS (not c-units) —
        # every shape subtracts it before the /c scaling.
        "reference_wealth": float(objective_config["Reference_Wealth"]),
        # Utility SHAPE params (dimensionless, in units of c); Symlog ignores all of them.
        "huber_aversion": float(objective_config["Huber_Aversion"]),
        "huber_delta": float(objective_config["Huber_Delta"]),
        "up_aversion": float(objective_config["Up_Aversion"]),
        "up_knee": float(objective_config["Up_Knee"]),
        "cara_gamma": float(objective_config["CARA_Gamma"]),
    }


def construct_hedge_runtime(
    config: Mapping[str, Any],
    stoch_factors: Optional[Mapping[Any, Any]] = None,
) -> Dict[str, Any]:
    """The JSON → runtime boundary: read `Hedging_Problem`, validate it, and return the runtime
    dict every consumer indexes directly. Nothing downstream re-validates.

    The objective's utility SHAPE params are DIMENSIONLESS, in units of the utility scale c
    (applied to x = (W − `Reference_Wealth`)/c; the reference is in DOLLARS and every shape reads
    it). Huber: quadratic small losses with curvature `huber_aversion`, a linear deep tail beyond
    the knee `huber_delta`, and the same form on the gain wing under `Up_Aversion`/`Up_Knee`
    (`Up_Aversion` 0 = exactly linear gains). CARA: u = (1−e^{−γx})/γ. Symlog ignores the shape
    params. See hedge_bundle._utility_wrap_signed for the exact forms.

    Two INDEPENDENT dials move what the shape is applied to and how it is scaled, and the solver
    refuses to arm them together. `Reference_Mode`: 'Fixed' measures TERMINAL wealth,
    'Running_Wealth' the day's wealth INCREMENT, which makes the DP's value a sum of per-step
    rewards. `Utility_Scale_Mode='conditional_sim'` replaces the single c with a per-decision-step
    schedule measured off the warmup batch and floored at `Utility_Scale_Floor_Frac` of its
    terminal entry; it owns the LEVEL of c as well as its shape, so `Utility_Scale_Explicit` is
    refused beside it and `Huber_Aversion` is dosed against the MEASURED scale.

    Both are dispatched on by equality downstream and are therefore VALIDATED at this boundary
    (`_objective_config`): a near-miss spelling would otherwise mean the default, silently.

    `im_funding_*` is a vol-linked initial-margin FUNDING charge on the post-trade book (realized
    accounting only). Per hedge leg i at step t the desk posts
    `IM_i = IM_Vol_Multiplier·(σ_t/IM_Ref_Vol)·F_i·|q_i^post|·cs_i` and pays
    `IM_Funding_Spread_Bps·1e-4·dt` to fund it over the calendar step, above the risk-free the
    margin ledger already earns. Spread default 0.0 ⇒ the term is exactly 0 and never executes."""
    hedging_problem = config["Hedging_Problem"]
    evaluator_config = hedging_problem["Evaluator"]
    objective_config = hedging_problem.get("Objective")
    solver_config = hedging_problem.get("Solver")
    execution_mode = config["Execution_Mode"]

    # --- instruments: the tradable universe splits into cash accounts and hedge legs ---
    tradables = _flatten_deals(hedging_problem["Tradable_Instruments"])
    cash_account_names = tuple(str(n) for n in evaluator_config["Cash_Instruments"])
    for account_name in cash_account_names:
        if account_name not in tradables:
            raise ValueError(
                f"Evaluator cash account '{account_name}' is not in Tradable_Instruments")
    hedge_names = tuple(n for n in tradables if n not in cash_account_names)
    if not hedge_names:
        raise ValueError("no hedge instruments: Tradable_Instruments has only cash accounts")

    normalized_tradables = {name: _instrument_metadata(entry, name in hedge_names)
                            for name, entry in tradables.items()}

    if execution_mode == "solve_hedge":
        if solver_config is None:
            raise ValueError("Execution_Mode 'solve_hedge' requires Hedging_Problem['Solver']")
        if str(solver_config.get("Object", "")).lower() not in ("diffsolver", "diffsolverv2"):
            raise ValueError(
                "Execution_Mode 'solve_hedge' requires Solver.Object='DiffSolver' or "
                f"'DiffSolverV2' (the forward-backward solver); got "
                f"{solver_config.get('Object')!r}. HindsightDpSolver remains available as the "
                "Run_Hindsight_Diagnostic track.")
        if int(config["Inner_Sub_Batch"]) < 2:
            raise ValueError("Execution_Mode 'solve_hedge' requires Inner_Sub_Batch >= 2")
        # A solve is a STREAM: Simulation_Batches - 1 fit batches, then a held-out batch no fit
        # step saw. A loaded checkpoint fits nothing, so it is a stream of one.
        n_batches = int(config["Simulation_Batches"])
        if solver_config["DiffV2_Load_Value_Fn"]:
            if n_batches != 1:
                raise ValueError(
                    "Execution_Mode 'solve_hedge' with DiffV2_Load_Value_Fn requires "
                    "Simulation_Batches == 1: a frozen policy fits nothing, so its one batch IS "
                    f"the held-out world; got {n_batches}.")
        elif n_batches < 2:
            raise ValueError(
                "Execution_Mode 'solve_hedge' requires Simulation_Batches >= 2 (fit batches, then "
                f"a held-out batch no fit step saw); got {n_batches}. Simulation_Batches is a path "
                "MULTIPLIER under 'simulate_only' and a STREAM LENGTH here, and derivus_batch "
                "divides it by the job count before this check.")
        if solver_config["DiffV2_Load_Value_Fn"] and solver_config["DiffV2_Save_Value_Fn"]:
            raise ValueError(
                "Solver.DiffV2_Save_Value_Fn is set alongside DiffV2_Load_Value_Fn: a loaded "
                "checkpoint is a frozen-policy EVALUATION and fits nothing, so there is no new "
                "value fn to write. Train (save) and evaluate (load) are separate runs.")
        # the DP lives in utility space: a scale-consuming shape, or LogWealth's per-step growth
        if str((objective_config or {}).get("Object", "")).lower() \
                not in _UTILITY_OBJECTS + ("logwealth",):
            raise ValueError(
                "Execution_Mode 'solve_hedge' requires a utility Objective.Object — one of "
                "'AsymmetricUtility_Symlog' | 'AsymmetricUtility_Huber' | 'AsymmetricUtility_CARA' | "
                "'LogWealth' (per-step growth). "
                "The DP recursion lives in utility space: an identity (legacy) objective leaves "
                "V-hat unbounded in dollars and the backward sweep blows up multiplicatively.")

    history_lookback = int(hedging_problem["History_Lookback_Business_Days"])
    if history_lookback < 0:
        raise ValueError("Hedging_Problem.History_Lookback_Business_Days must be non-negative")
    # Commodity names come from the live CommodityPrice factors the instruments created at
    # calc-dependency time — never re-parsed out of instrument JSON params.
    referenced_commodities = tuple(dict.fromkeys(
        utils.check_tuple_name(factor) for factor in (stoch_factors or {})
        if factor.type == 'CommodityPrice'))
    portfolio_state = hedging_problem["Portfolio_State"]
    scalar_spread_bps, spread_spec = _bid_offer_spread(evaluator_config)
    calendar_spread_bps = (float(evaluator_config["Calendar_Spread_Bps"])
                           if evaluator_config.get("Calendar_Spread_Bps") is not None else None)
    # a rate <= 0 makes matched volume free or PAID FOR, and `turnover_charge` then returns a
    # negative cost the argmax maximizes by churning; absence is the off switch
    if calendar_spread_bps is not None and calendar_spread_bps <= 0.0:
        raise ValueError(
            f"Evaluator.Calendar_Spread_Bps must be > 0; got {calendar_spread_bps}. Omit the "
            f"key to price every contract at the outright spread.")
    # static instrument -> cash_account routing by currency: the first cash account whose currency
    # matches wins, else the first account
    account_by_currency = {}
    for account_name in cash_account_names:
        account_by_currency.setdefault(
            normalized_tradables[account_name]["currency"], account_name)
    fallback_account = cash_account_names[0] if cash_account_names else None

    alloc_mode = str(evaluator_config["Allocation_Mode"]).lower()
    if alloc_mode not in ("exposure", "carry_variance"):
        raise ValueError(
            f"Unsupported Evaluator.Allocation_Mode: "
            f"{evaluator_config['Allocation_Mode']!r}. Supported: 'Exposure' (the declared "
            f"Allocation_Weights table) | 'Carry_Variance' (solver-derived). Dispatched by "
            f"equality, so a near-miss would silently run 'Exposure'.")
    if alloc_mode == "carry_variance":
        if evaluator_config.get("Allocation_Weights"):
            raise ValueError(
                "Evaluator.Allocation_Mode='Carry_Variance' DERIVES the per-step weights from "
                "the warmup sims — a declared Allocation_Weights table beside it is a "
                "contradiction the run refuses rather than arbitrates. Remove one of the two.")
        if solver_config is None:
            raise ValueError(
                "Evaluator.Allocation_Mode='Carry_Variance' needs a Solver to derive the "
                "weights (there are no sims to measure carry/tracking from in a non-solve "
                "mode). Declare Allocation_Weights instead.")
    return {
        "accounting_mode": str(evaluator_config["Accounting_Mode"]).lower(),
        "names": {
            "tradables": tuple(normalized_tradables),
            # the hedge legs ARE the action set; the solver builds its action grid over them
            "hedges": hedge_names,
            "cash_accounts": cash_account_names,
        },
        "referenced_commodities": referenced_commodities,
        "tradables": normalized_tradables,
        "objective": _objective_config(objective_config),
        "solver": _solver_config(solver_config),
        "history_lookback_business_days": history_lookback,
        "portfolio_state": {
            # whole contracts, as `Position_Limits` are read
            "positions": {str(n): int(v) for n, v in portfolio_state["Positions"].items()},
            "cash_balances": {str(n): float(v) for n, v in portfolio_state["Cash_Balances"].items()},
            "margin_balances": {str(n): float(v) for n, v in portfolio_state["Margin_Balances"].items()},
            "spot_price_history": _spot_price_history(
                hedging_problem, history_lookback, referenced_commodities),
        },
        "accounting": {
            "position_limits": {
                n: {"min_position": int(limit["Min_Position"]), "max_position": int(limit["Max_Position"])}
                for n, limit in ((n, evaluator_config["Position_Limits"][n]) for n in hedge_names)},
            "instrument_to_cash_account": {
                n: account_by_currency.get(meta["currency"], fallback_account)
                for n, meta in normalized_tradables.items()},
            "transaction_cost_per_unit":
                float(evaluator_config["Transaction_Cost_Per_Unit"]),
            # Scalar half-spread bps (fast-path + diagnostic display); `spec` is None for a scalar
            # Bid_Offer_Spread_Bps and a normalized per-instrument/vol-scale dict otherwise —
            # resolved in `per_contract_kappa`.
            "bid_offer_spread_bps": scalar_spread_bps,
            "bid_offer_spread_spec": spread_spec,
            # a rate prices the matched half of a rebalance across adjacent maturities at one
            # calendar crossing, in the solver's charge and the realized accounting alike
            "calendar_spread_bps": calendar_spread_bps,
            # Vol-linked IM funding charge on the post-trade book (realized accounting only).
            # Spread 0.0 ⇒ the term is exactly 0 and never executes.
            "im_funding_spread_bps": float(evaluator_config["IM_Funding_Spread_Bps"]),
            "im_vol_multiplier": float(evaluator_config["IM_Vol_Multiplier"]),
            "im_ref_vol": float(evaluator_config["IM_Ref_Vol"]),
            "force_flat_at_end": evaluator_config["Force_Flat_At_End"] == "Yes",
            "total_position_abs_limit":
                float(evaluator_config["Total_Position_Abs_Limit"]),
            "total_position_schedule": _position_schedule(evaluator_config),
            # Optional per-decision-step split of the NET cover across the legs. Present ⇒ the
            # action universe is the Q-ladder rather than the Cartesian product (see
            # HedgeActionSpace.allocation_grid); absent ⇒ today's grid, unchanged.
            "allocation_weights": _allocation_weights(evaluator_config, hedge_names),
            "allocation_mode": alloc_mode,
            # Per-leg |Δq| cap per decision step at the ARGMAX (0 = off). Execution policy
            # only — training labels never see it.
            "max_trade_per_step": float(evaluator_config["Max_Trade_Per_Step"]),
            # significance the argmax must beat the STANDING book by before it trades, in standard
            # errors of the paired inner-draw difference (0 = off). Execution policy only.
            "decision_deadband_sigma":
                float(evaluator_config["Decision_Deadband_Sigma"]),
        },
        "privileged_layout": derive_privileged_layout(stoch_factors),
    }


def per_contract_kappa(runtime, price, name, vol=None, calendar=False):
    """Per-contract turnover cost for tradable `name` at mark `price`: a flat
    `Transaction_Cost_Per_Unit` plus the half-spread charge on notional,
    `0.5 · spread_bps · 1e-4 · |price| · contract_size`, where `spread_bps` is the FULL quoted spread
    and the 0.5 is the mid-to-touch crossing. `price` is a scalar or tensor mark, read by its abs (a
    spread leg's mark goes negative, its notional does not). Single source for the solver's
    decision-time kappa, the env's realized debit and the diagnostic CSV writer.

    `spread_bps` is the scalar `Bid_Offer_Spread_Bps` unless a spread SPEC is configured, in which
    case it is the instrument's `Per_Instrument` base (falling back to `Default_Bps`) scaled by
    `(vol/Ref_Vol)**Beta` when the spec declares a `Vol_Scale` and an annualized `vol` is supplied.
    `vol=None`, a scalar spread, or no `Vol_Scale` ⇒ vol-independent.

    `calendar=True` prices one contract of a CALENDAR SPREAD quoted against this leg instead: the
    spread is `Evaluator.Calendar_Spread_Bps` on this leg's notional under the same vol scale, and
    the flat fee is charged TWICE because a spread contract moves and clears two futures. Paired as
    `0.5·(κcal_i + κcal_j)` by `turnover_charge`, the result is both clearing fees plus one
    half-spread on the average leg notional.

    The one scalar `Calendar_Spread_Bps` rate is a limitation (docs_src/limitations.md); the
    OUTRIGHT half of every charge reads each leg's own base."""
    acc = runtime["accounting"]
    contract_size = float(runtime["tradables"][name]["contract_size"])
    spec = acc["bid_offer_spread_spec"]
    scale = 1.0
    if spec is None:
        spread_bps = acc["bid_offer_spread_bps"]
    else:
        spread_bps = spec["per_instrument"].get(str(name), spec["default_bps"])
        vscale = spec["vol_scale"]
        if vscale is not None and vol is not None:
            scale = (vol / vscale["ref_vol"]) ** vscale["beta"]
    fee = acc["transaction_cost_per_unit"]
    if calendar:
        spread_bps, fee = acc["calendar_spread_bps"], 2.0 * fee
    return fee + 0.5 * spread_bps * scale * 1.0e-4 * abs(price) * contract_size


def turnover_charge(delta, kappa, kappa_cal=None, legs=False):
    """THE turnover-cost rule for a whole reposition, as `per_contract_kappa` is the rule for one
    contract. `delta` is `(..., n_hedge)` in `names['hedges']` (maturity) order; `kappa` and
    `kappa_cal` are per-leg and indexed `[..., i]`, so a `(n_hedge,)` decision kappa and a
    per-path `(B, n_hedge)` realized one both work. `legs=True` returns each leg's share
    `(..., n_hedge)`, a matched pair's calendar price half on each of its legs at that leg's own
    `kappa_cal`, so the shares sum to the charge and each leg pays its own.

    No `kappa_cal` ⇒ the L1 sum `Σ_i |Δ_i|·κ_i`, unchanged.

    With one, the move DECOMPOSES by a greedy sweep over CONSECUTIVE maturities: at each pair
    (i, i+1) the opposite-signed volume `x = min(|Δ_i|, |Δ_j|)` is matched off and pays
    `0.5·(κcal_i + κcal_j)` per contract — one calendar quote on that listed pair — and whatever
    survives the sweep pays its OWN leg's `κ_i`, because an unmatched lot physically sits on that
    leg and crosses that leg's bid/offer.

    ONE pairing rule, because the decision charge and the realized accounting have to be the same
    function. Global netting (`paired = (Σ|Δ| − |ΣΔ|)/2`) pairs the front against the back with no
    listed spread between them and under-charges a non-adjacent move by up to 5x; a `|Δ|`-share
    blend of the leg kappas over-charges leftover front-month lots by up to 2.4x."""
    if kappa_cal is None:
        charge = delta.abs() * kappa
    else:
        rem = list(delta.unbind(-1))
        paid = [torch.zeros_like(r) for r in rem]
        for i in range(len(rem) - 1):
            di, dj = rem[i], rem[i + 1]
            x = torch.minimum(di.abs(), dj.abs()) * ((di * dj) < 0)      # matched contracts
            rem[i], rem[i + 1] = di - di.sign() * x, dj - dj.sign() * x
            paid[i] = paid[i] + x * 0.5 * kappa_cal[..., i]
            paid[i + 1] = paid[i + 1] + x * 0.5 * kappa_cal[..., i + 1]
        charge = torch.stack(paid, -1) + torch.stack(rem, -1).abs() * kappa
    return charge if legs else charge.sum(dim=-1)


def kappa_pair(runtime, prices, vol=None, matched=True):
    """The per-leg rates `(κ, κ_cal)` `turnover_charge` prices a reposition at, each `(..., n_hedge)`
    in `names['hedges']` order, at the marks `prices` (keyed by hedge name) and annualized `vol`:
    the outright rate and the matched leg's calendar rate, None where no `Calendar_Spread_Bps` is
    configured (the one switch) or `matched` is False (every lot outright, as a forced close)."""
    hedges = runtime["names"]["hedges"]
    kappa = torch.stack([per_contract_kappa(runtime, prices[h], h, vol) for h in hedges], dim=-1)
    cal = (torch.stack([per_contract_kappa(runtime, prices[h], h, vol, True) for h in hedges], dim=-1)
           if matched and runtime["accounting"]["calendar_spread_bps"] is not None else None)
    return kappa, cal


def leg_charges(runtime, deltas, prices, vol=None, matched=True):
    """Each hedge leg's share `{name: (...)}` of the `turnover_charge` of a realized reposition
    `deltas` at marks `prices` (both keyed by hedge name) and annualized `vol`, at the rates
    `kappa_pair` reads: the charge the solver ranks the move at, split across the legs that pay it."""
    hedges = runtime["names"]["hedges"]
    charge = turnover_charge(torch.stack([deltas[h] for h in hedges], dim=-1),
                             *kappa_pair(runtime, prices, vol, matched), legs=True)
    return dict(zip(hedges, charge.unbind(-1)))


def initial_q_from_runtime(runtime, batch, device):
    """Per-hedge initial contract book `q0` `(batch, n_hedge)` from the normalized
    `Portfolio_State` positions, whole contracts, in `runtime['names']['hedges']` order - hedge
    legs only, cash accounts excluded. The one opening book every track holds: the stepper seeds
    its positions from it, and the solver's bank / verdict / benchmark tracks measure FIRST-step
    turnover from it rather than from flat.

    The value function is POSITION-FREE by default, so `q0` affects only first-step turnover
    diagnostics and the rolled P&L. Under `Solver.DiffV2_Position_State` it is also the standing
    position the first decision is charged from and the first position state the net reads."""
    positions = runtime["portfolio_state"]["positions"]
    hedges = runtime["names"]["hedges"]
    q0 = torch.tensor([float(positions.get(str(h), 0.0)) for h in hedges], device=device)
    return q0.unsqueeze(0).expand(batch, len(hedges)).contiguous()


def wealth_step(W, q, contract_size, dF, dL):
    """The ONE analytic hedged-wealth transition — frictionless telescoping core.

        W_{t+1} = W_t + Σ_i q_i·cs_i·dF_i + dL,   W_t = cumulative hedge P&L + marked liability L_t

    `q` (…,n_hedge) is the per-instrument position, `dF` (…,n_hedge) the per-instrument price
    move F_{t+1}−F_t, `dL` (…) the marked-liability change L_{t+1}−L_t. This is the law DiffSolver
    rolls and the one the twin loss DIFFERENTIATES: u(W_{t+1}) is taped back to a wealth leaf and
    the state-at-t market leaves, so it MUST stay a pure tensor op — no `.item()`/`.detach()`/
    `.cpu()`/`.to()`; callers own the grad context.

    `BundleStepper` is the DEPLOYMENT discretization of the same law: it books the same
    `Σ q·cs·dF` as per-instrument variation margin `pos·(price−settlement)·cs`, then layers on the
    extras this frictionless form omits — overnight financing, transaction cost, per-instrument
    settlement/expiry and the terminal forced-flat. The faithful walk-forward path is hedge_solver
    `_rollout_on_stepper`, which rolls the real stepper.
    """
    return W + (q * contract_size * dF).sum(dim=-1) + dL


def _is_utility_objective(runtime):
    """True iff the objective transforms wealth through a utility shape (symlog/huber/cara), so
    the DP and value function live in utility space and a scale `c` is required."""
    return (runtime.get("objective") or {}).get("object") in _UTILITY_OBJECTS


def _utility_scale(objective, t=None):
    """The dollar scale `c` in force at decision step `t` — ONE resolution rule, shared by the
    utility transform, its closed-form curvature and the decision-curve dump.

    `Objective.Utility_Scale_Mode='conditional_sim'` locks a per-step SCHEDULE at the frame lock
    (`Bundle.utility_scale_schedule`, mirrored onto the runtime objective beside the scalar): a
    caller that NAMES its step reads that step's knee, and one that does not — a TERMINAL read —
    reads the scalar, which under a schedule IS the schedule's terminal entry. An ensemble's
    run-level scalar is the MEAN of its members' terminal knees.

    A missing or NON-POSITIVE c is refused HERE rather than at each source that could produce one:
    `x = (W − R)/c` would otherwise be infinite or NaN, every label from that step with it, and
    the fit would spread the NaN through the nets and report a finished run."""
    schedule = objective.get("utility_scale_schedule")
    c = objective.get("utility_scale") if schedule is None or t is None else schedule[int(t)]
    if c is None:
        raise ValueError(
            "utility objective active but runtime['objective']['utility_scale'] is not set — "
            "Bundle.from_batch mirrors it from the resolved utility_scale; a hand-built runtime "
            "must set it explicitly.")
    c = float(c)
    if not c > 0.0:
        raise ValueError(
            f"utility scale c = {c} at step {t}: x = (W − R)/c is then not a number and neither "
            f"is any reward computed from it. Under Utility_Scale_Mode='conditional_sim' this is "
            f"the schedule's floor (Utility_Scale_Floor_Frac × its terminal entry) — the early "
            f"steps carry no dispersion of their own; otherwise it is Utility_Scale_Explicit.")
    return c


def mirror_frame(runtime, scale, schedule):
    """Cache a frame's utility scale `c` and its per-step `schedule` on `runtime['objective']`, where
    every reward, penalty and utility transform reads them without taking the bundle or the solver
    in its signature: a bundle's resolved frame, or the solver's locked one. A run with no objective
    holds none."""
    if runtime['objective'] is not None:
        runtime['objective']['utility_scale'] = float(scale)
        runtime['objective']['utility_scale_schedule'] = schedule


def _utility_x(W, objective, t=None):
    """Wealth `W` in the utility's coordinate `x = (W − R)/c`: against `Reference_Wealth` R, at
    the scale c in force at step t (`_utility_scale`). Every shape, its curvature and the
    decision dump read it."""
    return (W - float(objective["reference_wealth"])) / _utility_scale(objective, t)


def _utility_wrap_signed(x_dollars, runtime, t=None):
    """The terminal UTILITY u(W) applied to signed wealth — the single source of truth for
    the objective, the DP recursion, and the solver's value labels. Dispatches on the
    configured shape (all in normalised wealth x = (W − R) / c), identity for the legacy objective:

      symlog : sign(x)·log1p(|x|)                  — odd, tail-compressing (variance aversion)
      huber  : x − [a·loss² | a·δ²+2aδ(loss−δ)]    — quadratic small losses, linear deep tail
               − [a⁺·gain² | a⁺·κ²+2a⁺κ(gain−κ)]     (bounded scale, live grad); the same form
               (loss = max(−x,0), gain = max(x,0))   on the GAIN wing, off at a⁺ = 0
      cara   : (1 − exp(−γ·x)) / γ                  — bounded gains, exponentially-penalised loss

    `reference_wealth` R (DOLLARS, default 0) is the benchmark the utility is measured against and
    is subtracted BEFORE the /c scaling, which puts huber's knee at the operating point. All
    shapes read it.

    Huber's gain wing mirrors the loss wing: curvature `up_aversion` a⁺ (default 0 ⇒ exactly
    linear gains) out to the knee `up_knee` κ, linear beyond with the level-matching constant, so
    deep gains keep marginal utility 1 − 2a⁺κ > 0 and the objective is curved on both sides of R.

    Shape params (huber a/δ/a⁺/κ, cara γ) are DIMENSIONLESS in c-units. Differentiable in `w` —
    both huber knees are C¹, cara is smooth.

    `c` is `Bundle.utility_scale`, or, when a step `t` is named under a locked
    `Utility_Scale_Mode='conditional_sim'` schedule, that step's knee (`_utility_scale`). What u is
    applied TO is the CALLER's business: under `Objective.Reference_Mode='Running_Wealth'` the
    solver hands it the day's wealth INCREMENT, and this transform is unchanged."""
    if ((runtime.get("objective") or {}).get("object")) == "logwealth":
        # LogWealth has no terminal transform - its reward is the per-step ratio log(W1/W0), owned
        # by the solver's `_u_step`. Checked BEFORE `_is_utility_objective`, whose meaning is
        # "consumes a scale c": LogWealth does not, so it must not take the identity path either.
        raise ValueError(
            "Objective 'LogWealth' defines no terminal utility — the reward is the per-step "
            "growth ratio (DiffSolver._u_step). This caller must be step-sum-aware or disabled "
            "under LogWealth.")
    if not _is_utility_objective(runtime):
        return x_dollars
    obj = runtime["objective"]
    x = _utility_x(x_dollars, obj, t)
    shape = obj["object"]
    if shape == "asymmetricutility_symlog":
        return torch.sign(x) * torch.log1p(x.abs())
    if shape == "asymmetricutility_huber":
        a, d = float(obj["huber_aversion"]), float(obj["huber_delta"])
        au, k = float(obj["up_aversion"]), float(obj["up_knee"])
        loss = (-x).clamp(min=0.0)
        quad = a * loss * loss
        lin = a * d * d + 2.0 * a * d * (loss - d)
        gain = x.clamp(min=0.0)
        quad_up = au * gain * gain
        lin_up = au * k * k + 2.0 * au * k * (gain - k)
        return (x - torch.where(loss <= d, quad, lin)
                - torch.where(gain <= k, quad_up, lin_up))
    # asymmetricutility_cara
    g = float(obj["cara_gamma"])
    return -torch.expm1(-g * x) / g


def _utility_local_curvature(W, runtime, t=None):
    """`(u'(W), −u''(W)/u'(W))` in DOLLARS at one scalar wealth — the marginal utility and the
    local absolute risk aversion the configured shape is actually applying there. `t` names the
    decision step, so a locked per-step scale schedule reports the knee in force on THAT day.

    The closed forms in `_utility_wrap_signed`'s normalised x = (W − R)/c, chain-ruled back to
    dollars (u' = f'(x)/c and ARA = −f''(x)/(c·f'(x)), so the scale enters exactly once each):

      symlog : f' = 1/(1+|x|)                     −f''/f' = sign(x)/(1+|x|)  — risk-SEEKING below R
      huber  : f' = 1 + 2a·min(loss, δ)           −f''/f' = 2a/f'   inside the loss knee
               f' = 1 − 2a⁺·min(gain, κ)          −f''/f' = 2a⁺/f'  inside the gain knee
                                                  0 beyond either knee, where both wings go linear
      cara   : f' = exp(−γx)                      −f''/f' = γ, constant by construction
      identity (no utility objective): (1, 0) — there is no curvature to report.

    u'' jumps at the reference itself (the two wings carry different curvatures), so x < 0 reads
    the loss wing and x ≥ 0 the gain wing — the same side convention `_utility_wrap_signed`'s
    clamps take. Written out rather than differentiated: the caller is a diagnostic on a no-grad
    rollout and wants the two numbers at one wealth, not a tape."""
    if not _is_utility_objective(runtime):
        return 1.0, 0.0
    obj = runtime["objective"]
    c, x = _utility_scale(obj, t), _utility_x(float(W), obj, t)
    shape = obj["object"]
    if shape == "asymmetricutility_symlog":
        f1 = 1.0 / (1.0 + abs(x))
        return f1 / c, math.copysign(f1, x) / c
    if shape == "asymmetricutility_huber":
        # the loss wing steepens the slope and the gain wing flattens it; curvature is +2a either way
        a, knee, d, s = ((float(obj["huber_aversion"]), float(obj["huber_delta"]), -x, 1.0)
                         if x < 0.0 else
                         (float(obj["up_aversion"]), float(obj["up_knee"]), x, -1.0))
        f1 = 1.0 + s * 2.0 * a * min(d, knee)
        return f1 / c, (0.0 if d > knee else 2.0 * a / (f1 * c))
    g = float(obj["cara_gamma"])
    return math.exp(-g * x) / c, g / c


def _loss_side_curvature(runtime, t=None):
    """`(u'(R⁻), ARA(R⁻))` — the utility's slope and absolute risk aversion a hair BELOW the
    reference, which is the dose a hedging objective is actually run at.

    At EXACTLY R the closed form takes the gain wing by its side convention, and that wing's
    curvature is `Up_Aversion` — zero for every book that never armed it. Callers reporting a DOSE
    rather than a position (the frame-lock log, and the decision-curve dump under an increment
    objective, where the increment is ~0 before the day happens) want this one."""
    obj = runtime["objective"]
    reference = float(obj["reference_wealth"])
    return _utility_local_curvature(reference - 1.0e-6 * _utility_scale(obj, t), runtime, t)


def _rolling_realized_vol(spot, market, window=PRICE_ZSCORE_WINDOW):
    """Annualized trailing-window realized log-vol of a `(T, B)` mark path, `(T, B)`: the squared
    log moves of the last `window` rows over the market days they hold (`market` `(T,)`, a row of
    none moving nothing and counting nothing), at 252 a market day, the one clock tau and the
    decisions count. The one spelling behind the utility scale's sigma and the step vol."""
    log_ret = spot.clamp_min(1e-9).log().diff(dim=0)                                  # (T-1, B)
    ret_sq = torch.cat([torch.zeros_like(log_ret[:1]), log_ret * log_ret], dim=0)     # (T, B)
    cum = torch.cat([torch.zeros_like(ret_sq[:1]), ret_sq.cumsum(dim=0)], dim=0)      # (T+1, B)
    held = torch.as_tensor(market, dtype=spot.dtype, device=spot.device).cumsum(0)
    held = torch.cat([held.new_zeros(1), held])                                       # (T+1,)
    idx = torch.arange(ret_sq.shape[0] + 1, device=spot.device)
    lo = (idx - window).clamp_min(0)
    count = (held[idx[1:]] - held[lo[1:]]).clamp_min(1.0).unsqueeze(-1)
    return (252.0 * (cum[idx[1:]] - cum[lo[1:]]) / count).clamp_min(0.0).sqrt()   # (T, B)


def _history_key(commodity_field):
    """The `spot_price_history` key for a deal's raw `Commodity` field: the PRIMARY spot's full
    factor name, which is the key space `_spot_price_history` validates the dict against. The deal
    carries the bare name (`'X'`, or a composed `'X.Y'` whose history is its primary's), and
    keying the lookup by that raw field can never match."""
    if not commodity_field:
        return None
    return utils.check_tuple_name(
        utils.Factor('CommodityPrice', utils.check_rate_name(commodity_field)[:1]))


def _im_funding_charge(positions, prices, runtime, vol, dt):
    """Per-hedge-leg initial-margin funding debit `{name: (B,)}` on a step's POST-trade book:

        IM_i = IM_Vol_Multiplier · (σ_t/IM_Ref_Vol) · F_i · |q_i^post| · cs_i

    funded at `IM_i · IM_Funding_Spread_Bps · 1e-4 · dt` over the calendar step. That is the spread
    the desk pays ABOVE the risk-free the margin ledger already earns, so it is a pure debit.

    IM is posted on GROSS per-leg |q|, which over-margins calendar spreads: a −1/+1 roll posts two
    legs' IM rather than a netted spread margin. σ_t is the SAME per-step vol that drives the
    Vol_Scale bid/offer spread, so funding rises with vol. Only called when
    `im_funding_spread_bps` is truthy, at which point `vol` is a scalar and never None."""
    acc = runtime["accounting"]
    # the per-leg scalar; x (F_i·|q_i|·cs_i) gives funding_i
    factor = (acc["im_vol_multiplier"] * (vol / acc["im_ref_vol"])
              * acc["im_funding_spread_bps"] * 1.0e-4 * dt)
    return {n: factor * prices[n].abs() * positions[n].abs()
               * float(runtime["tradables"][n]["contract_size"])
            for n in runtime["names"]["hedges"]}


# wealth is money accumulated over hundreds of rows: float64, cast once where a net reads it
MONEY = torch.float64


def _portfolio_value(state, runtime):
    """Absolute total wealth: cash + margin + position value where applicable. `_pnl_excess` is the
    wealth change since inception, which is what the asymmetric utility needs.

    Cash and margin balances are raw dollars compounded daily (each step multiplies by
    `cash_tv(next)/cash_tv(curr)` ≈ 1 + SOFR·dt, then adds today's flows), so each dollar earns
    interest only from the day it landed.

    Two accounting modes. `cash_account`: position × price × contract_size plus the cash balances.
    Futures: cash is FROZEN at starting capital and margin carries every VM and trade-cost flow
    since inception, every row settled at its own mark."""
    hedges = runtime["names"]["hedges"]
    tradables = runtime["tradables"]
    positions, prices = state["positions"], state["tradable_values"]
    total = torch.zeros_like(state["done"], dtype=MONEY)
    if runtime["accounting_mode"] == "cash_account":
        for name in hedges:
            total = total + (positions[name].to(dtype=MONEY) * prices[name].to(dtype=MONEY)
                             * float(tradables[name]["contract_size"]))
        cash = None
        for balance in state["cash_accounts"].values():
            v = balance.to(dtype=MONEY)
            cash = v if cash is None else cash + v
        return total if cash is None else total + cash
    for accounts in (state["margin_accounts"], state["cash_accounts"]):
        balances = None
        for balance in accounts.values():
            v = balance.to(dtype=MONEY)
            balances = v if balances is None else balances + v
        if balances is not None:
            total = total + balances
    return total


def _pnl_excess(state, runtime):
    """Wealth change since inception: `portfolio_value - initial_portfolio_value`, which is what
    makes the utility's floor at zero meaningful. The initial baseline is snapshotted when the
    stepper builds its opening state and threaded through every transition."""
    pv = _portfolio_value(state, runtime)
    initial = state.get("initial_portfolio_value")
    if initial is None:
        return pv
    return pv - initial.to(device=pv.device, dtype=pv.dtype)


def _tracking_error_value(state, runtime):
    """The hedge-quality read — `pnl_excess + liability_value`, which an optimal hedge holds at ≈0
    at EVERY step."""
    pnl_excess = _pnl_excess(state, runtime).to(dtype=MONEY)
    return pnl_excess + state["liability_value"].to(dtype=MONEY, device=pnl_excess.device)


def terminal_stats(x):
    """The terminal-P&L summary every report reads off `x` `(B,)`, in `MONEY`: the mean, the
    population std, the min and max, the median, the 5th and 95th percentiles (linear
    interpolation) and the mean of the tail at or below the 5th."""
    x = x.to(dtype=MONEY)
    p5, median, p95 = torch.quantile(x, x.new_tensor([0.05, 0.5, 0.95]))
    return {"mean": float(x.mean()), "std": float(x.std(unbiased=False)), "min": float(x.min()),
            "max": float(x.max()), "median": float(median), "p5": float(p5), "p95": float(p95),
            "cvar5": float(x[x <= p5].mean())}


def _align_time_axis(tensor, steps):
    """Truncate/pad a time-major tensor to `steps` rows (padding repeats the last row) — the
    liability MTM grid is the reference axis every other series is aligned to."""
    current = int(tensor.shape[0])
    if current >= steps:
        return tensor[:steps]
    pad_shape = (steps - current,) + tuple(tensor.shape[1:])
    return torch.cat([tensor, tensor[-1:].expand(*pad_shape)], dim=0)


class Bundle:
    """The simulated hedging world: every time-indexed tensor the solver and the environment
    read, plus the frame locked at build time.

    Built ONCE per simulation by `from_batch` (four stages — `_resolve_tradables`,
    `_resolve_liability`, `_resolve_factors`, `_resolve_frame`), then read-only. Time tensors
    carry a `History_Lookback_Business_Days` prefix of realized rows in front of the simulated
    grid where a `Spot_Price_History` is declared (none where it is not), every series alike, so
    full-grid indexing is `initial_time_index + t`; the `*_sim` views strip that prefix for solver
    code that indexes by simulation-grid `t`.

    The frame — `utility_scale` (the objective's scale c, with its per-decision-step
    `utility_scale_schedule` under `Utility_Scale_Mode='conditional_sim'`, both mirrored onto the
    runtime objective) and `step_annual_vol` (the per-step vol driving the state-dependent spread
    and IM funding) — is resolved here and never recomputed: a per-rollout c silently rescales
    every reward. `inner_mc` / `inner_mc_grad` are attached by `HedgeMonteCarlo.execute` in
    solve_hedge mode; they fork the simulator at a decision step and price it to its landing row,
    or to each row of a `DiffV2_Risk_Horizon` list."""

    documentation = ('Bundle', [
        'Every time-indexed tensor the solver and the environment read, built ONCE per batch by',
        '`from_batch` and read-only afterwards. Consumers index it by attribute; nothing recomputes',
        'it.',
        '',
        '- **Tradables** — `tradables` / `tradables_sim`: the per-instrument mark series, harvested',
        '  from the priced deal tree via `DealStructure.tensor_marks`. An instrument that fails to',
        '  price is ABSENT rather than zero. A leg is live on the rows its deal\'s dates mark',
        '  (`marked`) and moves over a step it is still marked at the end of, the step carrying',
        '  market time (`moving`): the one liveness rule every track reads.',
        '- **Liability** — `liability_value` / `liability_sim`: its mark plus the cash it has paid',
        '  on earlier rows, and `unfixed_at`: the signed quantity its fixings still carry after a row.',
        '- **Factors** — the simulated factor paths, and `privileged_factors`: whatever each process',
        '  publishes as market state.',
        '- **Frame** — `utility_scale` (the objective scale `c`), its optional per-decision-step',
        '  schedule `utility_scale_schedule` (`Objective.Utility_Scale_Mode=\'conditional_sim\'`:',
        '  the cross-sectional dispersion of the flat-book wealth at each step, floored), and',
        '  `step_annual_vol` (the per-step vol driving state-dependent spreads and IM funding: a',
        '  process\'s revealed conditional vol, else the primary spot\'s trailing realised vol with',
        '  its `Spot_Price_History` in front, as the utility scale reads it).',
        '',
        'The frame is resolved at build time and never recomputed. A per-rollout `c` would silently',
        'rescale every reward, which is why a loaded checkpoint evaluates in ITS frame rather than',
        "the evaluation world's.",
        '',
        '!!! warning "The history prefix"',
        '    Time tensors carry `History_Lookback_Business_Days` realized rows in FRONT of the',
        '    simulated grid where a `Spot_Price_History` is declared, every series alike, so',
        '    full-grid indexing is `initial_time_index + t` while the `*_sim`',
        '    views strip the prefix. Indexing a full-grid tensor by simulation `t` silently reads',
        '    history.',
        '',
        'In `solve_hedge` mode `HedgeMonteCarlo.execute` attaches `inner_mc` / `inner_mc_grad`,',
        'which fork the simulator at a decision step and price it to its landing row, or to each',
        'row of a `DiffV2_Risk_Horizon` list, on demand.',
    ])

    def __init__(self, base_date=None, business_day=None, num_batches=1):
        self.base_date = base_date
        self.business_day = business_day
        self.num_batches = int(num_batches)
        # Time axis (history prefix + simulated grid).
        self.time_grid_days = None          # (T,) day offsets from base_date
        self.time_grid_days_cpu = []        # the same as python ints (sync-free step loop)
        self.scenario_dates = None          # DatetimeIndex over the full grid
        self.initial_time_index = 0         # history/sim boundary
        self.history_rows = 0               # History_Lookback_Business_Days prefix rows
        # Simulated series.
        self.tradables = {}                 # {instrument: (T, B)} marks
        self.last_marked = {}               # {instrument: the last sim-grid row its deal marks}
        self.factors = {}                   # {factor name: (T, …, B)} scenario paths
        self.privileged_factors = {}        # {name: (T, B, dim)} process-revealed state
        self.primary = None                 # the primary spot's factor name (`primary_spot`)
        self.liability_value = None         # (T, B) the liability's value, or None
        self.spot_price_history = {}        # {commodity: (H, B)} realized history prefix
        # Frame + liability descriptors.
        self.spot_realized_vol = {}         # {commodity: (T, B)} rolling realized vol
        self.step_annual_vol = None         # (T,) scalar vol per step, or None
        self.calibrated_utility_inputs = None
        self.utility_scale = None
        # per-step c_t under Utility_Scale_Mode='conditional_sim'
        self.utility_scale_schedule = None
        self.total_leg_volume = 0.0
        self.fixings = ()                   # ((day offset, signed quantity), ...) of the liability
        self.last_live_mtm_index = 0
        self.decision_steps, self.successors = [0], []    # the decision rows, each row's successor
        # inner-MC forks, attached by HedgeMonteCarlo.execute for Execution_Mode='solve_hedge'
        self.inner_mc = None
        self.inner_mc_grad = None

    # ---- construction --------------------------------------------------------
    @classmethod
    def from_batch(cls, base_date, business_day, time_grid_days, tradable_blocks,
                   factor_tensor_blocks, hedge_profile_blocks, num_batches, stoch_factors,
                   runtime, privileged_factor_blocks=None, total_leg_volume=None, fixings=()):
        """Assemble the per-batch tensor blocks the HedgeMonteCarlo simulator produced into one
        bundle: join them along the batch axis, then run the four resolve stages.

        `stoch_factors` is the simulator's factor map (factor_key → process); the processes are
        asked for their privileged surfaces, calibrated vols and revealed conditional vol."""
        self = cls(base_date, business_day, num_batches)
        self.primary = utils.check_tuple_name(primary_spot(stoch_factors))
        tradables = {n: torch.cat(blocks, dim=1) for n, blocks in tradable_blocks.items()}
        factors = {n: torch.cat(blocks, dim=-1) for n, blocks in factor_tensor_blocks.items()}
        mtm = (torch.cat(hedge_profile_blocks['mtm'], dim=1)
               if hedge_profile_blocks.get('mtm') else None)
        cashflows = {currency: torch.cat(blocks, dim=1) for currency, blocks
                     in (hedge_profile_blocks.get('realized_cashflows') or {}).items()}
        # The liability MTM sets the time axis (its native length is the mtm grid); everything
        # else pads/truncates to it. No liability ⇒ the shortest simulated series wins.
        steps = int(mtm.shape[0]) if mtm is not None else min(
            [int(time_grid_days.shape[0])]
            + [int(t.shape[0]) for t in tradables.values()]
            + [int(t.shape[0]) for t in factors.values()])
        self._resolve_tradables(tradables, time_grid_days, steps, runtime)
        self._resolve_liability(mtm, cashflows, steps, total_leg_volume, fixings)
        self._resolve_factors(factors, privileged_factor_blocks or {}, stoch_factors, steps)
        self._resolve_frame(runtime, stoch_factors)
        return self

    def _resolve_tradables(self, tradables, time_grid_days, steps, runtime):
        """Stage 1 — the time axis and the per-instrument mark series, both carrying the realized
        history prefix, and the prefix's length `history_rows` every later stage applies:
        `History_Lookback_Business_Days` rows of realized spot where a `Spot_Price_History` is
        declared, none where it is not, so a rolling-window feature at sim-day-0 already has its
        lookback; a tradable whose `Commodity` has a history and a spot factor that has one take
        that series, every other series broadcasts its first row. Also derives the CPU day mirror,
        the scenario dates and `initial_time_index`."""
        grid = _align_time_axis(time_grid_days, steps)
        # a deal's marks run to its last date and the alignment repeats that row after it
        self.last_marked = {n: min(int(t.shape[0]), steps) - 1 for n, t in tradables.items()}
        logging.debug('last marked sim-grid rows: %s', self.last_marked)
        tradables = {n: _align_time_axis(t, steps) for n, t in tradables.items()}
        history = runtime['portfolio_state']['spot_price_history']
        # the one prefix gate: the lookback where a realized history exists, none where it does not
        self.history_rows = H = int(runtime['history_lookback_business_days']) if history else 0
        logging.debug('history prefix: %d rows', H)
        base_ts = pd.Timestamp(self.base_date)
        if H > 0:
            batch_size = int(next(iter(tradables.values())).shape[1])
            for commodity, payload in history.items():
                late = [d for d in payload['dates'] if pd.Timestamp(d) >= base_ts]
                if late:
                    raise ValueError(
                        f"Spot_Price_History for {commodity!r} carries {len(late)} row(s) at/after "
                        f"the base date ({min(late)}): history must be STRICTLY before it. A "
                        f"base-date row duplicates sim day 0 in the bundle timeline and shifts "
                        f"every *_sim view one row - the solver then scores each decision "
                        f"against the step ENDING at its fork's anchor, a full day of lookahead")
            ref_dates = history[next(iter(history))]['dates'][-H:]
            prefix_days = torch.tensor([int((d - base_ts).days) for d in ref_dates],
                                       dtype=grid.dtype, device=grid.device)
            grid = torch.cat([prefix_days, grid], dim=0)
            self.spot_price_history = {
                commodity: torch.tensor(payload['prices'][-H:], dtype=torch.float32,
                                        device=grid.device).unsqueeze(1)
                                       .expand(-1, batch_size).contiguous()
                for commodity, payload in history.items()}
            tradables = {
                n: torch.cat([self._history_prefix(
                    t, self.spot_price_history.get(_history_key(
                        runtime['tradables'][n]['params'].get('Commodity')))), t], dim=0)
                for n, t in tradables.items()}
        self.time_grid_days = grid
        self.tradables = tradables
        self.time_grid_days_cpu = grid.detach().cpu().to(dtype=torch.int64).tolist()
        days = self.time_grid_days_cpu
        self.scenario_dates = pd.DatetimeIndex(
            [base_ts + pd.Timedelta(days=int(d)) for d in days])
        # where the history prefix ends and the simulation grid begins (history rows carry negative
        # day offsets); solvers strip this offset to index time tensors by sim-grid t
        self.initial_time_index = next(
            (i for i, d in enumerate(days) if int(d) >= 0), len(days))
        # the market days arriving on each row: the one clock decisions and realised vol count
        self.grid_market_days = utils.market_days(self.scenario_dates, self.business_day)
        logging.debug('market days a row: %s', self.grid_market_days.tolist())

    def _history_prefix(self, tensor, realized=None):
        """The `history_rows` prepended to a simulated series: the realized history broadcast
        across the batch when the series has one, else the series' own first row repeated."""
        if realized is not None:
            return realized.to(dtype=tensor.dtype)
        return tensor[:1].expand((self.history_rows,) + tuple(tensor.shape[1:])).contiguous()

    def _resolve_liability(self, mtm, cashflows, steps, total_leg_volume, fixings):
        """Stage 2 — the liability's value, its mark plus the cash it has paid on earlier rows, the
        leg volume the utility scale needs and the liability's fixings `unfixed_at` reads.
        `last_live_mtm_index` is the structural pre-settlement terminal: the grid appends one
        clean-exit row where the liability settles to zero, so the last LIVE mtm row is
        `steps - 2` — the single source for the DP depth and the L_T read."""
        if mtm is not None:
            mtm = _align_time_axis(mtm, steps)
            paid = sum((_align_time_axis(t, steps) for t in cashflows.values()), torch.zeros_like(mtm))
            # the mark carries a payment on its own row and drops it after, so the cash paid on
            # earlier rows joins it: the liability's value is continuous across a payment
            value = mtm + torch.cat([torch.zeros_like(paid[:1]), paid.cumsum(0)[:-1]])
            logging.debug('liability value: cash paid on rows %s', paid.any(-1).nonzero().flatten().tolist())
            self.liability_value = torch.cat([self._history_prefix(value), value], dim=0)
        self.last_live_mtm_index = T = int(steps - 2)
        # decisions on the rows of market time, the rows the market-clock laws step on, and on the
        # first row whatever its day, where the opening book is traded
        market = self.grid_market_days[self.initial_time_index:]
        self.decision_steps = [0] + [t for t in range(1, T) if market[t] > 0]
        self.market_time = market           # market time arriving on each sim-grid row
        self.successors = [next((s for s in self.decision_steps if s > t), T) for t in range(T)]
        if total_leg_volume:
            self.total_leg_volume = float(total_leg_volume)
        self.fixings = tuple(fixings)
        logging.info('liability scalars: total_leg_volume=%s close=%d',
                     self.total_leg_volume or None, T)

    def _resolve_factors(self, factors, privileged_factor_blocks, stoch_factors, steps):
        """Stage 3 — the simulated factor paths and the per-process privileged surfaces (the
        market state the value function consumes), both prefixed by `history_rows`, a spot's by its
        realized history."""
        self.factors = {n: torch.cat([self._history_prefix(t, self.spot_price_history.get(n)), t], dim=0)
                        for n, t in ((n, _align_time_axis(t, steps)) for n, t in factors.items())}
        self.privileged_factors = {
            n: torch.cat([self._history_prefix(t), t], dim=0)
            for n, t in assemble_privileged_factors(privileged_factor_blocks, stoch_factors).items()}

    def _resolve_frame(self, runtime, stoch_factors):
        """Stage 4 — the LOCKED frame: the realized-vol surface, the utility scale c and (under
        `Utility_Scale_Mode='conditional_sim'`) its per-decision-step schedule, both mirrored onto
        the runtime objective so every reward/penalty reads one frame, and the per-step vol series.
        Nothing here may be recomputed per rollout: a drifting c silently rescales the whole
        objective, and the vol series drives realized transaction cost."""
        self.spot_realized_vol = self._spot_realized_vol()
        # a run with no Objective reads no scale
        if runtime['objective'] is not None:
            if not self.spot_price_history:
                # No Spot_Price_History ⇒ source (spot, σ) from the CALIBRATED market data instead.
                self.calibrated_utility_inputs = self._calibrated_utility_inputs(stoch_factors)
            self.utility_scale = self._resolve_utility_scale(runtime)
            logging.info('utility_scale (symlog c) resolved to {0:.2f}'.format(self.utility_scale))
            mirror_frame(runtime, self.utility_scale, self.utility_scale_schedule)
            if self.utility_scale_schedule is not None:
                self._log_scale_dose(runtime)
        self.step_annual_vol = self._resolve_step_vol(runtime, stoch_factors)

    def _spot_realized_vol(self, window=PRICE_ZSCORE_WINDOW, min_periods=5):
        """Annualized rolling realized log-vol of each underlying spot, `{commodity: (T, B)}`. In
        mean-reverting regimes with σ scaled to hold the stationary std fixed, realized vol rises
        with kappa, which makes this a regime signal as well as the utility-scale σ source."""
        out = {}
        for commodity in (c for c in self.spot_price_history if c in self.factors):
            rv = _rolling_realized_vol(self.factors[commodity].float(), self.grid_market_days, window)
            rv[:min_periods] = 0.0
            out[commodity] = rv
        return out

    def _calibrated_utility_inputs(self, stoch_factors):
        """Utility-scale `(commodity, spot, σ)` sourced from CALIBRATED market data — the
        `Spot_Price_History`-absent fallback: the primary spot's sim-day-0 level and its process's
        calibrated annualized vol. None when the process reports none."""
        process = {utils.check_tuple_name(k): p for k, p in stoch_factors.items()}[self.primary]
        sigma = process.calibrated_annual_vol()
        if sigma is None or sigma <= 0.0:
            return None
        return (self.primary, float(self.factors[self.primary][0].median().item()), float(sigma))

    def _resolve_utility_scale(self, runtime):
        """The money scale `c` mapping wealth to utility (u(x; c) = sign(x)·log1p(|x|/c) for
        symlog). Single source of truth — every reward and penalty reads `Bundle.utility_scale`.

        Modes (`Objective.Utility_Scale_Mode`): `vol_scaled_notional` (default) gives
        c = total_leg_volume × initial_spot × σ_annual × √τ, τ the business days of the job's
        calendar from the first simulated row to the last settlement over 252, the market clock σ
        is annualised on; `conditional_sim` measures a per-decision-step SCHEDULE off this batch
        instead and takes its terminal entry as the scalar (`_conditional_sim_schedule`);
        `Objective.Utility_Scale_Explicit` overrides the
        formula with a literal value and is REFUSED beside the schedule mode.

        A degenerate input (no history and no calibrated vol, no volume, a non-positive σ) raises:
        no stand-in c is meaningful in every currency, and `_utility_scale` refuses a
        non-positive one.

        The history path reads spot and realized σ at the history/sim boundary H as a BATCH MEDIAN
        rather than slot [H, 0]: `full[H]` is the first sim step, and a process emitting a
        stochastic initial draw would otherwise make c path-dependent off path 0."""
        objective = runtime['objective']

        def _degenerate(reason):
            raise ValueError(
                f"utility scale: cannot compute a meaningful c — {reason}. Fix the bundle/config, "
                "or set Objective.Utility_Scale_Explicit to a literal value.")

        mode = objective['utility_scale_mode']
        if mode not in ('vol_scaled_notional', 'conditional_sim'):
            raise ValueError(
                f"Unsupported Objective.Utility_Scale_Mode: {mode!r}. "
                "Supported modes: 'vol_scaled_notional' | 'conditional_sim'. Set "
                "Utility_Scale_Explicit to override the formula with a literal dollar value.")
        explicit = objective['utility_scale_explicit']
        if mode == 'conditional_sim':
            if explicit is not None:
                raise ValueError(
                    "Objective.Utility_Scale_Explicit is set beside "
                    "Utility_Scale_Mode='conditional_sim': the schedule measures the LEVEL of c "
                    "as well as its shape (the scalar is its terminal entry), so a literal dollar "
                    "value is a second source for the same number and silently picking a winner "
                    "is how the two drift apart. Drop one of them.")
            self.utility_scale_schedule = self._conditional_sim_schedule(objective)
            if self.utility_scale_schedule is not None:
                return float(self.utility_scale_schedule[-1])
            # one outer path carries no cross-section to measure, so the formula below stands in
            # as the placeholder frame a frozen roll replaces from its checkpoint
        if explicit is not None:
            return explicit
        # tau measures the SIM horizon: the market time from the sim-grid origin through the close
        H = self.initial_time_index
        market = self.market_time[1:self.last_live_mtm_index + 1].sum()
        logging.debug('utility scale horizon: %s market days through the close row %d',
                      float(market), self.last_live_mtm_index)
        tau_years = max(float(market) / 252.0, 1.0 / 252.0)
        if not self.spot_price_history and self.calibrated_utility_inputs is None:
            _degenerate(
                "spot_price_history is empty and no calibrated underlying vol is available "
                "(no referenced CommodityPrice process reports calibrated_annual_vol)")
        if not self.total_leg_volume:
            _degenerate("total_leg_volume is zero")
        if self.spot_price_history:
            # history path: spot and realized sigma read at the history/sim boundary H
            commodity = self.primary
            # batch-median at H, never slot [H, 0]: keeps c path-independent under a stochastic
            # initial draw
            initial_spot = float(self.factors[commodity][H].float().median().item())
            sigma = float(self.spot_realized_vol[commodity][H].median().item())
        else:
            commodity, initial_spot, sigma = self.calibrated_utility_inputs
        if sigma <= 0.0:
            _degenerate(
                f"realized vol for {commodity!r} is non-positive (σ={sigma}) — "
                "likely early-calibration window or upstream calc bug")
        return self.total_leg_volume * initial_spot * sigma * (tau_years ** 0.5)

    def _log_scale_dose(self, runtime):
        """Log what a `conditional_sim` schedule DOES to the objective, not only what it measured.

        The schedule sets the LEVEL of c as well as its shape, and the utility's shape params are
        DIMENSIONLESS in units of c, so arming the mode re-doses the risk aversion by the same
        factor. This reports the absolute risk aversion per dollar the shape applies at the
        reference, under the first knee and the terminal one, so a stale `Huber_Aversion` is
        visible in the run log."""
        sched = self.utility_scale_schedule
        _, ara_0 = _loss_side_curvature(runtime, 0)
        _, ara_T = _loss_side_curvature(runtime, len(sched) - 1)
        logging.info(
            'utility_scale SCHEDULE (conditional_sim): %d steps, c_0=%.6g c_mid=%.6g c_T=%.6g. '
            'The schedule sets the objective LEVEL as well as its shape, and the shape params are '
            'dimensionless in units of c — so the DOSE moved with it: absolute risk aversion at '
            'the reference is %.4g per $ at t=0 and %.4g per $ at the terminal. Re-dose '
            'Huber_Aversion against THIS scale (Utility_Scale_Explicit does not exist under this '
            'mode).', len(sched), sched[0], sched[len(sched) // 2], sched[-1], ara_0, ara_T)

    def _conditional_sim_schedule(self, objective):
        """`Utility_Scale_Mode='conditional_sim'` — the per-decision-step utility scale `c_t`, and
        the whole of what that mode adds: the cross-sectional std ACROSS OUTER PATHS, at each step
        t, of the flat-book net wealth, floored at `Utility_Scale_Floor_Frac` of its terminal
        entry. `x = (W − R)/c_t` is then a z-score of the day's wealth against the dispersion the
        world itself carries that day, so the utility's knee sits where the mass is on EVERY
        decision rather than only at the terminal.

        WHICH OBJECT. `liability_sim[t]` is the wealth the DP itself carries under a flat book —
        `DiffSolver._build_bank` opens at `W = L[0]` and telescopes `W + Σ q·cs·dF + (L[t+1] −
        L[t])` — so its dispersion at t is the money the day's decision is about, in the reward's
        own units. `_tracking_error_value` is the same read one layer down, but it is
        POSITION-dependent and defined only on a rollout, so it cannot be frame-locked.

        THE FLOOR is a fraction of the TERMINAL entry. Every outer path shares `L_0`, so the
        dispersion of the first steps is ~0 and an unfloored `x = (W − R)/c_0` is unbounded. The
        terminal entry is the dispersion the deal ENDS on — for an average-rate swap the
        post-fixing residual, which can sit well below the mid-horizon peak — so this is a
        materially tighter floor than a fraction of the maximum.

        THE LEVEL is the schedule's too, not only the shape: there is no `Utility_Scale_Explicit`
        under this mode, and `_log_scale_dose` reports the resulting aversion per dollar.

        ONE ENTRY per step a continuation is ever read at: decisions 0..T_dec−1 plus the terminal
        mark T_dec (`last_live_mtm_index`). `None` when the batch carries a single outer path,
        which has no cross-section; the only 1-path world is the frozen roll, which evaluates in
        its checkpoint's frame regardless.

        Measured ONCE at the frame lock, frozen by the solver, stamped into the checkpoint and
        refused on mismatch: it is deterministic in the state and identical across every world the
        policy is rolled in."""
        if self.liability_value is None:
            raise ValueError(
                "Objective.Utility_Scale_Mode='conditional_sim' needs a marked liability: the "
                "schedule is the dispersion of the flat-book wealth path and this bundle has no "
                "liability_value to measure it on.")
        L = self.liability_sim[:self.last_live_mtm_index + 1].to(dtype=torch.float32)
        if int(L.shape[-1]) < 2:
            return None
        # measured FROM THE START, so the knee is the dispersion the WORLD generates by day t
        # rather than the start scatter Randomize_Initial_State injects at t0 - which would
        # otherwise flatten the schedule onto the terminal scale. Identical starts make them equal.
        c = (L - L[0]).std(dim=-1)
        terminal = float(c[-1])
        if not terminal > 0.0:
            raise ValueError(
                f"Objective.Utility_Scale_Mode='conditional_sim' measured a terminal scale of "
                f"{terminal}: the outer paths carry no dispersion at the last live liability mark, "
                f"so every knee would floor at zero and x = (W − R)/c would be unbounded.")
        floor = float(objective['utility_scale_floor_frac']) * terminal
        return tuple(float(v) for v in c.clamp_min(floor))

    def _resolve_step_vol(self, runtime, stoch_factors):
        """Per-step scalar annualized-vol series `(T,)` driving BOTH the state-dependent bid/offer
        half-spread (`per_contract_kappa` Vol_Scale) and the vol-linked IM funding charge. Built
        ONLY when a Vol_Scale spec OR IM funding is active — None otherwise, so `per_contract_kappa`
        ignores vol and no funding accrues. The source PREFERS a process-revealed conditional vol
        (GARCH publishes log h_t → σ_t = √(exp(log h_t)/dt_c)), falling back to the trailing
        realized-vol proxy off the primary spot, its realized history in front of the simulated
        path as the utility scale's σ reads it. Batch-reduced
        to one scalar per step, matching the solver's mean-mark kappa. The chosen source is logged
        once per build: a silent proxy fallback under a GARCH retrain would invalidate the whole
        vol coupling."""
        acc = runtime['accounting']
        spec = acc['bid_offer_spread_spec']
        vol_scale_active = bool(spec and spec['vol_scale'])
        im_funding_active = bool(acc['im_funding_spread_bps'])
        if not (vol_scale_active or im_funding_active):
            return None
        factor, proc, block = privileged_block(self.privileged_factors, stoch_factors, 'log_h')
        if block is not None:
            logging.info('step_annual_vol source: revealed %s log_h (Vol_Scale=%s IM_funding=%s)',
                         factor.name[0], vol_scale_active, im_funding_active)
            return proc.revealed_annual_vol(block[..., 0]).mean(dim=-1)                 # (T,)
        logging.info('step_annual_vol source: realized-vol proxy on %s (Vol_Scale=%s IM_funding=%s)',
                     self.primary, vol_scale_active, im_funding_active)
        # batch-reduced to one scalar per step, floored so the warm-up window never zeros the spread
        return _rolling_realized_vol(self.factors[self.primary].float(),
                                     self.grid_market_days).mean(dim=-1).clamp_min(1e-6)

    # ---- accessors -----------------------------------------------------------
    @property
    def device(self):
        return self.time_grid_days.device

    @property
    def batch_size(self):
        return int(next(iter(self.tradables.values())).shape[-1])

    @property
    def last_index(self):
        """Final grid row — the terminal the stepper stops at."""
        return max(int(self.time_grid_days.shape[0]) - 1, 0)

    @property
    def tradables_sim(self):
        """History-stripped mark series `{instrument: (t_outer, B)}`, indexed by sim-grid `t`
        exactly like the inner-MC forks."""
        return {k: v[self.initial_time_index:] for k, v in self.tradables.items()}

    @property
    def liability_sim(self):
        """History-stripped liability value `(t_outer, B)`."""
        return self.liability_value[self.initial_time_index:]

    @property
    def vol_sim(self):
        """Sim-grid slice of the per-step vol series (or None when no Vol_Scale/IM funding)."""
        v = self.step_annual_vol
        return None if v is None else v[self.initial_time_index:]

    def move(self, a, b, refs, rows=slice(None), dim=-1):
        """The marks' move `F_b − F_a` from sim-grid row `a` to row `b` of each tradable of `refs`
        over `rows`, stacked along `dim`: the one spelling of a held contract's realized P&L per
        unit of its contract size."""
        H = self.initial_time_index
        logging.debug('marks moved from row %d to row %d', a, b)
        return torch.stack([self.tradables[r][H + b][rows] - self.tradables[r][H + a][rows]
                            for r in refs], dim=dim)

    def decisions_from(self, t_min):
        """The decision rows at or after sim-grid row `t_min` (`Solver.T_Min`): where the solver
        sweeps and every bound decides, the opening book held before the first."""
        return [t for t in self.decision_steps if t >= t_min]

    def marked(self, t, refs):
        """`(len(refs),)` 1 or 0: whether each tradable of `refs` is still marked on sim-grid row
        `t`, its deal's dates pricing it there (`last_marked`): the one liveness rule, a leg
        trading at t where it is marked at t."""
        mask = [float(t <= self.last_marked[r]) for r in refs]
        logging.debug('marked on row %d: %s', t, mask)
        return torch.tensor(mask, device=self.device)

    def moving(self, t, refs):
        """`(len(refs),)` 1 or 0: whether each tradable of `refs` moves over decision row `t`'s one
        step, to its successor: still marked there (`marked`), the step carrying market time, since
        a market-clock law's prices do not move over a step of none."""
        landing = self.successors[t]
        return self.marked(landing, refs) * float(sum(self.market_time[t + 1:landing + 1]) > 0)

    def liability_at(self, time_index):
        """The liability's value `(B,)` at full-grid step `time_index` (zeros when there is none),
        in the accounting's `MONEY`."""
        if self.liability_value is None:
            return torch.zeros(self.batch_size, dtype=MONEY, device=self.device)
        return self.liability_value[time_index].to(dtype=MONEY, device=self.device)

    def unfixed_at(self, time_index):
        """Signed liability quantity still to fix after full-grid step `time_index`: the summed
        quantity of every liability fixing dated after that row."""
        day = self.time_grid_days_cpu[int(time_index)]
        return sum(quantity for fixing_day, quantity in self.fixings if fixing_day > day)

    def vol_at(self, time_index):
        """Scalar annualized vol at full-grid step `time_index`, or None when no vol series was
        built (`per_contract_kappa` then ignores vol)."""
        return None if self.step_annual_vol is None else self.step_annual_vol[int(time_index)]

    def calendar_dt(self, current, next_index):
        """CALENDAR-clock step-year fraction `(days[next]-days[cur])/365.25` — the sim dt
        convention (dt=1/365.25) over which posted initial margin is funded. Scalar python float
        off the CPU day mirror (no CUDA sync)."""
        days = self.time_grid_days_cpu
        return (days[int(next_index)] - days[int(current)]) / 365.25


class BundleStepper:
    """The hedging ENVIRONMENT: advance the bundle one day at a time under arbitrary actions.

    Owns the realized accounting the frictionless `wealth_step` law abstracts away — overnight
    financing on cash/margin, per-instrument variation margin and settlement, transaction cost
    (with the optional calendar-spread rebate and vol-linked IM funding), position rounding to
    integer contracts, and the forced flatten at the solver's terminal, the last live liability
    mark, the verdict's close. It trades only on the solver's decision rows
    (`Bundle.decision_steps`). Each step returns a state dict the caller inspects; the caller
    chooses the next action. Supports `copy.deepcopy(stepper)` to fork into
    counterfactual branches, and records its own trajectory for `write_diagnostic_csvs`.

    Vectorized over the bundle's full batch (B paths advance in lockstep). Action values may be
    scalars (broadcast) or per-path `(B,)` tensors. `runtime` is a PARAMETER of the replay rather
    than the bundle's: passing a variant with one accounting switch flipped is how the cost
    decomposition isolates each friction on an unchanged world. `mirror_scale=False` keeps the
    runtime's utility scale as the caller set it (see `__init__`)."""

    documentation = ('Bundle', [
        'The hedging ENVIRONMENT: advance the bundle one day at a time under arbitrary actions,',
        'with the realized accounting the frictionless wealth law abstracts away — overnight',
        'financing on cash and margin, per-instrument variation margin and settlement, turnover cost',
        'with the optional calendar-spread rebate and vol-linked IM funding, rounding to integer',
        'contracts, and the forced flatten at the last live liability mark, where the verdict closes.',
        'It trades only on the rows the solver decides on, the first row and every row of market',
        "time on the job's calendar before that mark (`Bundle.decision_steps`).",
        '',
        'Each `step` returns a state dict and the caller chooses the next action. Vectorized over',
        'the whole batch — every path advances in lockstep, and an action may be a scalar',
        '(broadcast) or one value per path. `copy.deepcopy` forks a counterfactual branch, and the',
        'recorded trajectory feeds `write_diagnostic_csvs`.',
        '',
        'The JSON that drives it lives under `Evaluator`: position limits,',
        '`Transaction_Cost_Per_Unit`, `Bid_Offer_Spread_Bps` (the FULL quoted spread — each trade',
        'pays half — or a spec with',
        '`Per_Instrument` bases and a `Vol_Scale` making the spread vol-dependent), margin funding,',
        'and the optional `Total_Position_Schedule` corridor on the signed book total.',
        '',
        '!!! note "`runtime` is a parameter of the replay"',
        '    Passing a variant — one accounting switch flipped — replays an UNCHANGED world under',
        '    different frictions, which is how the cost decomposition isolates each one.',
        '',
        '!!! warning "Whose `c` governs the roll"',
        '    `mirror_scale=True` (the default) re-mirrors the objective scale onto the runtime from',
        "    the bundle being stepped. A FROZEN-policy rollout passes `False`: the checkpoint's",
        "    scale is part of the value function's frame, and the argmax must read the same `c` the",
        '    nets were fitted against.',
    ])

    def __init__(self, bundle, runtime, mirror_scale=True):
        """Bind the environment to `bundle` + `runtime` and open the state at sim-day-0.

        The replay's rewards are marked against THIS bundle, so `mirror_scale=True` (the default)
        re-mirrors its scale onto the possibly-variant runtime. Under a frozen-policy run
        (`DiffV2_Load_Value_Fn`) the solver has already restored the CHECKPOINT's scale — the value
        function's own frame — and re-mirroring would overwrite it with the eval world's, leaving
        the policy rollout deciding under a different `c` than the verdict did. `mirror_scale=False`
        leaves the runtime's scale alone, which is what a frozen rollout wants."""
        self.bundle = bundle
        self.runtime = runtime
        self._accounting = runtime['accounting']
        self._account_of = self._accounting['instrument_to_cash_account']
        self._hedges = tuple(runtime['names']['hedges'])
        self._cash_names = tuple(runtime['names']['cash_accounts'])
        self._tradable_names = tuple(runtime['names']['tradables'])
        self._device = bundle.device
        self._batch_size = bundle.batch_size
        self._last_idx = bundle.last_index
        # the hedge closes at the terminal, the last live liability mark, as the verdict closes it
        self._close_idx = bundle.initial_time_index + bundle.last_live_mtm_index
        self._decision_set = {bundle.initial_time_index + t for t in bundle.decision_steps}
        # the default re-mirrors THIS bundle's scale; False keeps a checkpoint's frame
        if mirror_scale:
            mirror_frame(runtime, bundle.utility_scale, bundle.utility_scale_schedule)
        self._state = self._initial_state()
        # the opening book, then the book held over each row stepped: what write_diagnostic_csvs reads
        self._held = {n: [self._state['positions'][n]] for n in self._hedges}
        # each leg's turnover charges as booked, `(row, (B,))`: a trade's on its row, the close's on
        # the terminal
        self._charged = {n: [] for n in self._hedges}
        self._terminal_transition = None

    # ---- public surface ------------------------------------------------------
    @property
    def time_index(self):
        return int(self._state['time_index'])

    @property
    def is_decision_step(self):
        return self.time_index in self._decision_set

    @property
    def done(self):
        return self.time_index >= self._last_idx

    def observe(self):
        """Snapshot of the current pre-step state. Tensors are returned as-is (caller can
        `.cpu().numpy()` if needed); positions are post-last-step."""
        return {
            'time_index': self.time_index,
            'is_decision_step': self.is_decision_step,
            'done': self.done,
            'positions': dict(self._state['positions']),
            'tradable_values': dict(self._state['tradable_values']),
            'liability_value': self._state['liability_value'],
        }

    def step(self, action=None):
        """Advance one time step. `action` is `{instrument_name: scalar_or_(B,)_tensor}` applied
        as trade deltas (only meaningful at decision steps; ignored otherwise). Pass `None` for
        zero trades. Returns the post-step observation plus this transition's per-path
        pnl_excess + liability_value."""
        was_decision_step = self.is_decision_step
        structured = (self._structured_action(action)
                      if (action is not None and was_decision_step) else None)
        next_state = self._step_state(self._state, structured)
        transition = self._payoff(next_state)
        if self._batch_size == 1 and logging.getLogger().isEnabledFor(logging.DEBUG):
            # reconciliation organ for B=1 rolls, every step so the daily identities close exactly
            cur = self.time_index
            deltas = self._trade_deltas(structured)
            cost = leg_charges(self.runtime, deltas, self._state['tradable_values'], self.bundle.vol_at(cur))
            legs = {n: (float(self._state['tradable_values'][n][0]),
                        int(next_state['positions'][n][0]), int(deltas[n][0]),
                        float(cost[n][0]), self._account_of[n]) for n in self._hedges}
            logging.debug(
                'STEP %s t=%d dec=%d legs(mark,q,dq,cost,account)=%s vm=%.6f cost=%.6f margin=%.6f '
                'cash_tv=%s pnl_excess=%.6f liability=%.6f',
                self.bundle.scenario_dates[cur].date(), cur, int(was_decision_step), legs,
                sum(float(v[0]) for v in next_state.get('variation_margin', {}).values()),
                sum(float(c[0]) for c in cost.values()),
                sum(float(b[0]) for b in next_state['margin_accounts'].values()),
                {n: float(self._state['tradable_values'][n][0]) for n in self._cash_names
                 if n in self._state['tradable_values']},
                float(transition['pnl_excess'][0]), float(transition['liability_value'][0]))
        self._state = next_state
        self._terminal_transition = transition
        return {
            **self.observe(),
            'transition_pnl_excess': transition['pnl_excess'],
            'transition_net_pnl': transition['net_pnl'],
        }

    def evaluation_output(self, timing=None):
        """Terminal P&L summary in the `evaluation_summary` shape `HedgeMonteCarlo.execute`
        publishes. `simulate_only` rolls the no-trade policy, so the policy and the reference
        baseline are the same trajectory and the difference block is identically zero."""
        last = {k: self._terminal_transition[k].detach()
                for k in ('pnl_excess', 'liability_value', 'net_pnl')}
        net = terminal_stats(last['net_pnl'])
        metrics = {
            'average_net_pnl': net['mean'], 'median_net_pnl': net['median'], 'worst_net_pnl': net['min'],
            'average_hedge_pnl': float(last['pnl_excess'].mean()),
            'average_liability': float(last['liability_value'].mean()),
        }
        cpu = lambda values: {str(n): t.detach().to(dtype=torch.float32).cpu()
                              for n, t in values.items()}
        return {
            'metrics': metrics,
            'final_state': {
                'positions': cpu(self._state['positions']),
                'cash_accounts': cpu(self._state['cash_accounts']),
                'margin_accounts': cpu(self._state['margin_accounts']),
                **cpu(last),
            },
            'diagnostics': {'num_episodes': int(last['net_pnl'].shape[0]),
                            'num_batches': self.bundle.num_batches,
                            'trainer_type': 'simulate'},
            'timing': dict(timing or {}),
            'reference': {'no_trade': {'metrics': metrics},
                          'policy_minus_no_trade': {k: 0.0 for k in metrics}},
        }

    def write_diagnostic_csvs(self, output_dir: str, label: str = 'custom') -> None:
        """Write the per-day per-instrument breakdown + terminal P&L summary for the trajectory
        this stepper accumulated, driven by whatever policy the caller chose. Files:
          <output_dir>/<label>_paths.csv   — 5 representative paths (worst/p5/mean/p95/best)
          <output_dir>/<label>_summary.csv — terminal P&L stats

        Must be called after the rollout has reached `done`."""
        if self._terminal_transition is None:
            raise ValueError("Stepper has no recorded trajectory yet — call step() until done first.")
        rollout = self._rollout()
        fields = self._diag_fields(rollout)
        os.makedirs(output_dir, exist_ok=True)
        self._diag_write_paths(fields, label, os.path.join(output_dir, f'{label}_paths.csv'))
        net = terminal_stats(rollout['net_pnl'])
        total = terminal_stats(fields['total_ex_funding_discount'][-1])
        pd.DataFrame([{'policy': label, 'metric': m, 'net_pnl': net[m], 'total_ex_funding': total[m]}
                      for m in ('mean', 'std', 'min', 'p5', 'p95', 'max')]).to_csv(
            os.path.join(output_dir, f'{label}_summary.csv'), index=False, float_format='%.2f')

    # ---- state construction + transition -------------------------------------
    def _zeros_by_name(self, names):
        return {str(n): torch.zeros(self._batch_size, dtype=MONEY, device=self._device)
                for n in names}

    def _seed_by_name(self, seed_values, names):
        return {str(n): torch.full((self._batch_size,), float(seed_values.get(str(n), 0.0)),
                                   dtype=MONEY, device=self._device) for n in names}

    def _values_at(self, time_index):
        """Current mark `{instrument: (B,)}` at full-grid `time_index`, a hedge leg's held at the
        terminal past it: the objective ends at the last live liability mark, so a book left
        standing under `Force_Flat_At_End` 'No' is valued there and carried over no settlement row.
        In `MONEY`, so every flow off a mark is."""
        tradables = self.bundle.tradables
        return {n: tradables[n][min(time_index, self._close_idx) if n in self._hedges
                                else time_index].to(dtype=MONEY)
                for n in self._tradable_names if n in tradables}

    def _initial_state(self):
        """The opening state at sim-day-0 (bundle row `initial_time_index`) — the history prefix
        feeds features only, never the simulator, so JSON-supplied positions are today's overnight
        book rather than H days ago, marked at the first row's prices: the P&L since inception
        (`_pnl_excess`) is the book's from that row."""
        portfolio_state = self.runtime['portfolio_state']
        initial_time_index = self.bundle.initial_time_index
        state = {
            'done': torch.zeros(self._batch_size, dtype=torch.bool, device=self._device),
            'positions': dict(zip(self._hedges, initial_q_from_runtime(
                self.runtime, self._batch_size, self._device).unbind(-1))),
            'cash_accounts': self._seed_by_name(portfolio_state['cash_balances'], self._cash_names),
            'margin_accounts': self._seed_by_name(portfolio_state['margin_balances'], self._cash_names),
            'variation_margin': self._zeros_by_name(self._hedges),
        }
        state = self._refresh(state, initial_time_index)
        # the baseline `_pnl_excess` measures the change from
        state['initial_portfolio_value'] = _portfolio_value(state, self.runtime).detach().clone()
        return state

    def _refresh(self, state, time_index):
        """Cheap per-step refresh: re-read the price/liability views at `time_index` and carry
        the simulator state forward."""
        return {
            'time_index': int(time_index),
            'done': state['done'],
            'positions': state['positions'],
            'cash_accounts': state['cash_accounts'],
            'margin_accounts': state['margin_accounts'],
            'variation_margin': state['variation_margin'],
            'tradable_values': self._values_at(time_index),
            'liability_value': self.bundle.liability_at(time_index),
            'initial_portfolio_value': state.get('initial_portfolio_value'),
        }

    def _step_state(self, state, action):
        """Advance one day: compound the balances overnight, trade at the DECISION-TIME marks, its
        charge too (the next row's would let an unrelated overnight move distort the spread cost),
        then settle. `cash_account` mode moves each trade's consideration through cash and tracks
        no VM; futures mode freezes cash at starting capital and settles the post-trade book's
        price(t) → price(t+1) variation margin into margin, so the new delta accrues the move it
        traded into. A book adds whole contracts, `_structured_action` rounding the action; `done`
        reads the python-int time index, so a step costs no CUDA-CPU sync."""
        current = int(state['time_index'])
        if current >= self._last_idx:
            return state
        nxt = current + 1
        cash_mode = self.runtime['accounting_mode'] == 'cash_account'
        marks, next_values = state['tradable_values'], self._values_at(nxt)
        # shallow dict copies: tensors are replaced in-slot, never mutated in place
        next_positions = dict(state['positions'])
        next_cash = self._compound(state['cash_accounts'], marks, next_values)
        next_margin = (state['margin_accounts'] if cash_mode
                       else self._compound(state['margin_accounts'], marks, next_values))
        accounts = next_cash if cash_mode else next_margin
        variation_margin = self._zeros_by_name(self._hedges)
        deltas = self._trade_deltas(action)
        for n in self._hedges:
            # each contract is `contract_size` units of the underlying
            cs = float(self.runtime['tradables'][n]['contract_size'])
            next_positions[n] = next_positions[n] + deltas[n]
            if cash_mode:
                self._credit(accounts, self._account_of[n], -deltas[n] * marks[n] * cs)
            else:
                variation_margin[n] = next_positions[n] * (next_values[n] - marks[n]) * cs
                self._credit(accounts, self._account_of[n], variation_margin[n])
        self._apply_frictions(accounts, deltas, marks, next_positions, current, nxt,
                              self.bundle.vol_at(current))
        self._hold(next_positions)
        if self._accounting['force_flat_at_end'] and current == self._close_idx - 1:
            self._flatten(next_positions, accounts, next_values, self.bundle.vol_at(nxt), cash_mode)
        return self._refresh({
            'done': torch.full_like(state['done'], nxt >= self._last_idx, dtype=torch.bool),
            'positions': next_positions,
            'cash_accounts': next_cash,
            'margin_accounts': next_margin,
            'variation_margin': variation_margin,
            'initial_portfolio_value': state['initial_portfolio_value'],
        }, nxt)

    def _trade_deltas(self, action):
        """Per-hedge trade deltas `(B,)` from a structured action (zeros for no action)."""
        resolved = self._zeros_by_name(self._hedges)
        if action is None:
            return resolved
        for name, value in action['trade_deltas'].items():
            if name in resolved:
                tensor = torch.as_tensor(value, dtype=torch.float32, device=self._device)
                resolved[name] = tensor.repeat(self._batch_size) if tensor.ndim == 0 else tensor
        return resolved

    def _structured_action(self, action_dict):
        """Client action → the integer-contract trade deltas the env books. Position limits are
        enforced reward-side, not clipped here."""
        deltas = {}
        for name in self._hedges:
            v = action_dict.get(name, 0)
            deltas[name] = (v.to(device=self._device, dtype=torch.float32)
                            if isinstance(v, torch.Tensor)
                            else torch.full((self._batch_size,), float(v), dtype=torch.float32,
                                            device=self._device))
        ordered = torch.stack([deltas[n] for n in self._hedges],
                              dim=1).round().to(dtype=torch.int64)
        return {'trade_deltas': {n: ordered[:, i] for i, n in enumerate(self._hedges)}}

    def _apply_frictions(self, accounts, deltas, prices, next_positions, current, next_idx, vol_t):
        """The per-step friction flows both accounting modes share, debited from `accounts` (cash
        in cash mode, margin in futures mode), each leg's from its own account: its share of the
        trade's turnover charge, and the vol-linked IM funding on the post-trade book, so each
        flows through `_portfolio_value` → `_pnl_excess` → the utility."""
        for n, cost in leg_charges(self.runtime, deltas, prices, vol_t).items():
            self._credit(accounts, self._account_of[n], -cost)
            self._charged[n].append((current, cost))
        if self._accounting['im_funding_spread_bps']:
            dt = self.bundle.calendar_dt(current, next_idx)
            for n, funding in _im_funding_charge(next_positions, prices, self.runtime,
                                                 vol_t, dt).items():
                self._credit(accounts, self._account_of[n], -funding)

    def _hold(self, positions):
        # the book held over this step's row, before a close at its end books at the next row's
        # mark; a position tensor is rebound, never written in place, so its reference is the record
        for n in self._hedges:
            self._held[n].append(positions[n])

    def _credit(self, accounts, account_name, amount):
        if account_name is not None:
            accounts[account_name] = accounts[account_name] + amount

    def _compound(self, accounts, marks, next_marks):
        """Each balance grown overnight by its cash account's mark ratio `tv(next)/tv(cur)`: the
        cash tradable value is `Units / D(t) × fx_rep`, so the ratio is `D(cur)/D(next)`, about
        1 + SOFR·dt. A balance with no mark passes through by reference, safe because balances are
        always rebound to new tensors, never mutated in place."""
        return {n: (bal * (next_marks[n] / marks[n]) if n in marks and n in next_marks else bal)
                for n, bal in accounts.items()}

    def _flatten(self, positions, accounts, values, vol, cash_mode):
        """Close the hedge book at `values`, the terminal's marks, outright at `vol`: each leg pays
        its turnover charge and, in cash mode, its consideration, from its own account."""
        deltas = {n: -positions[n] for n in self._hedges}
        for n, cost in leg_charges(self.runtime, deltas, values, vol, matched=False).items():
            self._charged[n].append((self._close_idx, cost))
            cs = float(self.runtime['tradables'][n]['contract_size'])
            self._credit(accounts, self._account_of[n],
                         -(deltas[n] * values[n] * cs + cost) if cash_mode else -cost)
            positions[n] = positions[n] + deltas[n]

    def _payoff(self, state):
        """This transition's terminal read: `pnl_excess`, the wealth change since inception, and
        `net_pnl`, the net P&L `_tracking_error_value`, are only marked at the terminal step (a
        sync-free check on time_index — `done` is uniform across the batch since every scenario
        advances together); `liability_value` is the liability's value at the row, always carried."""
        zero = torch.zeros(self._batch_size, dtype=MONEY, device=self._device)
        terminal = int(state['time_index']) >= self._last_idx
        return {'pnl_excess': (_pnl_excess(state, self.runtime).to(device=self._device)
                               if terminal else zero),
                'net_pnl': (_tracking_error_value(state, self.runtime).to(device=self._device)
                            if terminal else zero),
                'liability_value': state['liability_value'].to(device=self._device)}

    # ---- diagnostic CSV assembly ---------------------------------------------
    def _rollout(self):
        """The recorded trajectory: per instrument the opening book and the book held over each row
        stepped, stacked."""
        return {
            'held': {n: torch.stack(self._held[n], dim=0).cpu() for n in self._hedges},
            'net_pnl': self._terminal_transition['net_pnl'].detach().cpu(),
        }

    def _diag_fields(self, rollout=None):
        """Per-instrument (T, B) per-day cashflow tensors plus portfolio totals, reconstructed on
        CPU from the recorded trajectory; the spot column is the primary spot's."""
        rollout = self._rollout() if rollout is None else rollout
        bundle, runtime = self.bundle, self.runtime
        spot = bundle.factors[bundle.primary].detach().cpu().float()
        T, B = spot.shape
        per_instr = {}
        # each mark held at the terminal past it, as `_values_at`, gathered by index: a copy, where
        # `.cpu().float()` of a CPU float32 world is the world's own tensor
        held_at = torch.arange(T).clamp(max=self._close_idx)
        # the liability's value, the stepper's, held at the terminal past it as the marks are
        liability = bundle.liability_value.detach().cpu().float()[held_at]
        logging.debug('CSV liability at the terminal row %d: %s', self._close_idx,
                      liability[self._close_idx].tolist())
        for name in self._hedges:
            cs = float(runtime['tradables'][name]['contract_size'])
            fut = bundle.tradables[name].detach().cpu().float()[held_at]
            # a row trades the change of the book it holds, at its own mark, so the forced close
            # sits on the terminal, where it is booked
            held = rollout['held'][name].float()
            rows = slice(bundle.initial_time_index, bundle.initial_time_index + held.shape[0] - 1)
            pos = torch.zeros((T, B)); trd = torch.zeros((T, B))
            pos[rows], trd[rows] = held[1:], held.diff(dim=0)
            pos[rows.stop:] = held[-1]
            trade_cash = -trd * fut * cs
            # the opening book enters at the first row's mark, where the stepper's P&L starts
            trade_cash[rows.start] -= held[0] * fut[rows.start] * cs
            # the leg's turnover charges as the stepper booked them, each on its row
            cost = torch.zeros((T, B))
            for row, charge in self._charged[name]:
                cost[row] += charge.detach().cpu().float()
            per_instr[name] = {
                'fut': fut, 'cs': cs, 'pos': pos, 'trd': trd,
                'trade_cash': trade_cash, 'position_mtm': pos * fut * cs,
                'cum_cash': trade_cash.cumsum(0), 'trade_cost': cost, 'cum_cost': cost.cumsum(0),
            }
        portfolio_pos_mtm = sum(p['position_mtm'] for p in per_instr.values())
        portfolio_cum_cash = sum(p['cum_cash'] for p in per_instr.values())
        portfolio_cum_cost = sum(p['cum_cost'] for p in per_instr.values())
        hp = portfolio_cum_cash + portfolio_pos_mtm - portfolio_cum_cost
        return {
            'spot': spot, 'spread_bps': float(runtime['accounting']['bid_offer_spread_bps']),
            'per_instr': per_instr, 'liability': liability,
            'portfolio_position_mtm': portfolio_pos_mtm,
            'portfolio_cum_trade_cash': portfolio_cum_cash,
            'portfolio_cum_trade_cost': portfolio_cum_cost,
            'hedge_portfolio_ex_funding': hp,
            'total_ex_funding_discount': liability + hp,
        }

    def _diag_write_paths(self, fields, label, csv_path):
        """Per-day per-instrument breakdown for 5 representative cases (worst/p5/mean/p95/best),
        selected by terminal `total_ex_funding_discount`."""
        total = fields['total_ex_funding_discount']
        T, B = total.shape
        sidx = torch.argsort(total[-1])
        cases = {'worst': int(sidx[0]), 'p5': int(sidx[round(0.05 * (B - 1))]),
                 'p95': int(sidx[round(0.95 * (B - 1))]), 'best': int(sidx[-1])}
        day_strs = [pd.Timestamp(d).strftime('%Y-%m-%d') for d in self.bundle.scenario_dates]
        rows = []

        def _row(t, case, path_idx, getter):
            row = {'policy': label, 'case': case, 'path_idx': int(path_idx), 'day': day_strs[t],
                   'spot': getter(fields['spot'][t]).item(),
                   'spread_bps': fields['spread_bps']}
            for name in self._hedges:
                p = fields['per_instr'][name]
                row[f'{name}_futures'] = getter(p['fut'][t]).item()
                row[f'{name}_contract_size'] = p['cs']
                row[f'{name}_position'] = float(getter(p['pos'][t]).item())
                row[f'{name}_trade'] = float(getter(p['trd'][t]).item())
                row[f'{name}_trade_cash'] = float(getter(p['trade_cash'][t]).item())
                row[f'{name}_trade_cost'] = float(getter(p['trade_cost'][t]).item())
                row[f'{name}_position_mtm'] = float(getter(p['position_mtm'][t]).item())
            row['portfolio_position_mtm'] = float(getter(fields['portfolio_position_mtm'][t]).item())
            row['portfolio_cum_trade_cash'] = float(getter(fields['portfolio_cum_trade_cash'][t]).item())
            row['portfolio_cum_trade_cost'] = float(getter(fields['portfolio_cum_trade_cost'][t]).item())
            row['hedge_portfolio_ex_funding'] = float(getter(fields['hedge_portfolio_ex_funding'][t]).item())
            row['liability_ex_post_settle_discount'] = float(getter(fields['liability'][t]).item())
            row['total_ex_funding_discount'] = float(getter(fields['total_ex_funding_discount'][t]).item())
            return row

        sim_start = self.bundle.initial_time_index
        for case_name, idx in cases.items():
            for t in range(sim_start, T):
                rows.append(_row(t, case_name, idx, lambda x, idx=idx: x[idx]))
        for t in range(sim_start, T):
            rows.append(_row(t, 'mean', -1, lambda x: x.mean()))
        pd.DataFrame(rows).to_csv(csv_path, index=False, float_format='%.6f')


class HedgeRuntimeExecutionResult:
    """High-level result for HedgeMonteCarlo's hedge-bundle handoff.

    Carries the `Bundle`, the normalized runtime, the evaluation summary and the solver artifact
    (`policy_artifact`, DiffSolver's saved value-function nets in JSON-serializable form), so
    downstream consumers work without touching framework internals. `create_stepper()` spawns a
    `BundleStepper` to drive the simulator day by day under any explicit policy.
    """
    def __init__(self, *, bundle=None, runtime=None, evaluation_summary=None, policy_artifact=None):
        self.bundle = bundle
        self.runtime = runtime
        self.evaluation_summary = evaluation_summary
        self.policy_artifact = policy_artifact

    def create_stepper(self) -> 'BundleStepper':
        """An interactive `BundleStepper` for the bundle, so client code drives the simulator one
        step at a time under arbitrary actions. Deep-copy the stepper to fork branches."""
        return BundleStepper(self.bundle, self.runtime)


def run_hedge_execution(bundle, runtime):
    """Roll the env forward with zero trades and report the terminal P&L summary — the unhedged
    baseline for `Execution_Mode='simulate_only'`. Callers drive an explicit policy on top via
    `HedgeRuntimeExecutionResult.create_stepper()`. `solve_hedge` does not come through here:
    `HedgeMonteCarlo.execute` drives `StreamingSolve` a batch at a time."""
    started = time.perf_counter()
    stepper = BundleStepper(bundle, runtime)
    while not stepper.done:
        stepper.step(None)
    return {'policy_artifact': None,
            'evaluation_output': stepper.evaluation_output(
                timing={'evaluation_time_seconds': float(time.perf_counter() - started)})}
