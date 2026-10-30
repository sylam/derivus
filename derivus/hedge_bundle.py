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
financing, transaction cost, roll rebate, IM funding, expiry flatten) advanced one day at a
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

import numpy as np
import pandas as pd
import torch

from . import utils


# Rolling window (business days) for the realized-vol series feeding the symlog utility scale
# and the state-dependent bid/offer spread.
PRICE_ZSCORE_WINDOW = 20


# Terminal-utility SHAPES: each maps dollars→O(1) utility via the deal scale c (x = W/c) and
# lives in utility space (the DP / value-fn recursion). symlog is odd/symmetric, huber and cara
# downside-averse; any Object outside this tuple takes the identity (no-op) path. LogWealth is
# NOT one — it consumes no scale c — and a solve admits it beside these by name.
_UTILITY_OBJECTS = (
    "asymmetricutility_symlog", "asymmetricutility_huber", "asymmetricutility_cara")


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


def _instrument_metadata(name, entry, *, hedge_names, cash_account_names, liability_expiry):
    """Per-tradable metadata: the expiry / last-trade dates the pricers and the expiry mask read,
    the routing flags, and the contract size. Dates fall back Expiry_Date → Maturity_Date →
    Investment_Horizon → the latest liability expiry."""
    params = entry["params"]
    investment_horizon = params.get("Investment_Horizon")
    maturity_date = params.get("Maturity_Date")
    fallback = investment_horizon if investment_horizon is not None else liability_expiry
    default = maturity_date if maturity_date is not None else fallback
    return {
        "name": str(name),
        "deal_type": entry["deal_type"],
        "is_hedge": name in hedge_names,
        "is_cash_account": name in cash_account_names,
        "currency": params.get("Currency"),
        "last_trade_date": params.get("Last_Trade_Date", default),
        "expiry_date": params.get("Expiry_Date", default),
        "first_notice_date": params.get("First_Notice_Date"),
        "auto_close_days_before_last_trade": int(params.get("Auto_Close_Days_Before_Last_Trade", 0)),
        "allow_new_positions_until_last_trade":
            params.get("Allow_New_Positions_Until_Last_Trade", "Yes") == "Yes",
        "allow_holding_past_last_trade": params.get("Allow_Holding_Past_Last_Trade", "No") == "Yes",
        "contract_size": float(params.get("Contract_Size", 1.0)),
        "params": params,                     # _flatten_deals already deep-copied per entry
    }


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
    raw = evaluator_config.get("Bid_Offer_Spread_Bps", 0.0)
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
    raw_history = (hedging_problem.get("Portfolio_State") or {}).get("Spot_Price_History") or {}
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
    modes; requires `Object` — 'DiffSolver' | 'DiffSolverV2' | 'HindsightDpSolver'.

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
    gamma = float(solver_config.get("DiffV2_Risk_Aversion", 1.0))
    if not 0.1 <= gamma <= 10.0:
        raise ValueError(f"DiffV2_Risk_Aversion must be in [0.1, 10.0] (the DP's aversion: it "
                         f"divides the LogWealth capital line — the clairvoyant seed's floor "
                         f"has no causal equivalent, this dial is its proxy); got {gamma}")
    return {
        # 'diffsolverv2' is its OWN object: the constructed put-delta bank is part of the fitted
        # function, not a spelling of DiffSolver
        "object": str(solver_config["Object"]).lower(),
        "multi_seed_count": int(solver_config.get("Multi_Seed_Count", 1)),
        # Backward-sweep depth: fit C_t for t in [t_outer-2 .. t_min]. 0 = full sweep to the
        # initial decision; t_min near t_outer-1 = a shallow (bounded) sweep.
        "t_min": int(solver_config.get("T_Min", 0)),
        # Greedy-decision action grid (levels per hedge axis) + batched-argmax chunk size.
        "training_action_grid_levels_per_axis":
            int(solver_config.get("Training_Action_Grid_Levels_Per_Axis", 11)),
        "training_action_chunk_size": int(solver_config.get("Training_Action_Chunk_Size", 64)),
        # Advantage decomposition: fit A = C - u(W) (NN residual over the bounded-utility anchor).
        "use_advantage_decomp": solver_config.get("Use_Advantage_Decomp", "Yes") == "Yes",
        # DiffSolver knobs (see docstring): per-t residual-net Adam iters / lr, bank noise.
        "diffv2_fit_iters": int(solver_config.get("DiffV2_Fit_Iters", 150)),
        "diffv2_lr": float(solver_config.get("DiffV2_LR", 2.0e-3)),
        "diffv2_bank_noise_frac": float(solver_config.get("DiffV2_Bank_Noise_Frac", 0.15)),
        # The DP's aversion: divides the capital line in the LogWealth reward. The forward pass
        # takes no dial.
        "diffv2_risk_aversion": gamma,
        # The drift tripwire: the tail multiple on the validation-measured null dispersion of the
        # cumsum, and the strength of the on-trip correction toward the realized drift (0 = report)
        "diffv2_drift_threshold_sigmas":
            float(solver_config.get("DiffV2_Drift_Threshold_Sigmas", 3.0)),
        "diffv2_drift_beta": float(solver_config.get("DiffV2_Drift_Beta", 0.0)),
        # the stepper roll's handover: the greedy policy holds the passive book from this date
        "diffv2_passive_from": str(solver_config.get("DiffV2_Passive_From", "") or ""),
        "diffv2_first_decision_from":
            str(solver_config.get("DiffV2_First_Decision_From", "Opening_Book")).lower(),
        # Residual-net regularization: the twin-loss pathwise-gradient match (diffv2_lambda_grad)
        # in STANDARDIZED space, plus optional weight decay for outer-path-starved problems.
        "diffv2_weight_decay": float(solver_config.get("DiffV2_Weight_Decay", 0.0)),
        "diffv2_hidden": int(solver_config.get("DiffV2_Hidden", 32)),
        "diffv2_lambda_grad": float(solver_config.get("DiffV2_Lambda_Grad", 1.0)),
        # Downside-aware SELECTION: mean(C) - kappa · semidev(C) at the argmax. 0 = off.
        "diffv2_risk_kappa": float(solver_config.get("DiffV2_Risk_Kappa", 0.0)),
        # Quadratic churn charge at the argmax and the label argmax; the fitted value targets
        # stay cost-free.
        "diffv2_churn_lambda": float(solver_config.get("DiffV2_Churn_Lambda", 0.0)),
        # Successor-inheritance extras: parameter proximity to the fitted neighbour, and the
        # early-termination tolerance an inherited fit stops at (the anchor runs the budget).
        "diffv2_temporal_proximity":
            float(solver_config.get("DiffV2_Temporal_Proximity", 0.0)),
        "diffv2_fit_tol": float(solver_config.get("DiffV2_Fit_Tol", 1e-3)),
        # Cost-aware EXECUTION: the verdict rollout charges the L1 repositioning cost
        # (Transaction_Cost_Per_Unit + half Bid_Offer_Spread_Bps) at the argmax, trading
        # expected value against the cost of getting there. Training stays cost-free.
        "diffv2_cost_aware_argmax":
            solver_config.get("DiffV2_Cost_Aware_Argmax", "No") == "Yes",
        # FRICTIONAL Bellman: the net book fraction becomes a state coordinate and the charge
        # enters the fitted TARGET (see docstring). 'No' = the position-free value.
        "diffv2_position_state":
            solver_config.get("DiffV2_Position_State", "No") == "Yes",
        # Load a checkpoint fitted on a different decision horizon: per-step structures clamp
        # to the saved range (the tail repeats the last fitted step). 'No' = exact-shape loads.
        "diffv2_load_horizon_pad":
            solver_config.get("DiffV2_Load_Horizon_Pad", "No") == "Yes",
        # Dimensionless state coordinates: price -> log-return vs calibrated t0 spot, basis ->
        # fraction of it, wealth -> fraction of t0 book notional. A different coordinate system
        # is a different function; checkpoints stamp it and refuse a mismatch.
        "diffv2_returns_state":
            solver_config.get("DiffV2_Returns_State", "No") == "Yes",
        # WEALTH-FREE residual: drop the W input column of the value net, so the ranking's
        # wealth dependence is the utility anchor's alone. 'No' = the wealth-bearing residual.
        "diffv2_wealth_free_value":
            solver_config.get("DiffV2_Wealth_Free_Value", "No") == "Yes",
        # Deployment-faithful backtest: with a frozen policy loaded, roll it day-by-day on the
        # observed path via BundleStepper (real futures accounting; decisions off the stepper's
        # own wealth). Exposes diagnostics['stepper_verdict']. 'No' = only the fast _verdict.
        "diffv2_stepper_rollout":
            solver_config.get("DiffV2_Stepper_Rollout", "No") == "Yes",
        # Pure DIAGNOSTIC: a CSV of the stepper rollout's per-step ranking curve and the local
        # shape of the utility it was ranked under. '' (or absent) = off, and nothing about the
        # roll changes when it is on.
        "diffv2_decision_curve_dump":
            str(solver_config.get("DiffV2_Decision_Curve_Dump", "") or ""),
        # Per-input-column greek normalization in the twin loss. 'No' = legacy pooled variance.
        "diffv2_per_column_grad_norm":
            solver_config.get("DiffV2_Per_Column_Grad_Norm", "Yes") == "Yes",
        # Save the fitted nets, or load (a path, or a LIST for ensemble-argmax) and skip training.
        "diffv2_save_value_fn": str(solver_config.get("DiffV2_Save_Value_Fn", "") or ""),
        "diffv2_load_value_fn":
            ([str(p) for p in solver_config["DiffV2_Load_Value_Fn"]]
             if isinstance(solver_config.get("DiffV2_Load_Value_Fn"), (list, tuple))
             else str(solver_config.get("DiffV2_Load_Value_Fn", "") or "")),
        # Convergence declarations: the value's optimism over its held-out greedy utility a fit
        # saves and a roll loads, and how far an ensemble's members' swept books may disagree.
        "diffv2_max_value_optimism": float(solver_config.get("DiffV2_Max_Value_Optimism", 1.0)),
        "diffv2_max_member_disagreement":
            float(solver_config.get("DiffV2_Max_Member_Disagreement", 8.0)),
        "active_hedge_indices":
            (list(solver_config["Active_Hedge_Indices"])
             if solver_config.get("Active_Hedge_Indices") is not None else None),
        # Benchmark tracks assembled alongside the DiffSolver deliverable (hindsight upper
        # bound / textbook lower bound).
        "run_hindsight_diagnostic": solver_config.get("Run_Hindsight_Diagnostic", "No") == "Yes",
        "run_textbook_benchmark": solver_config.get("Run_Textbook_Benchmark", "No") == "Yes",
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

    Both are dispatched on by equality downstream and are therefore VALIDATED here: a near-miss
    spelling would otherwise mean the default, silently.

    `im_funding_*` is a vol-linked initial-margin FUNDING charge on the post-trade book (realized
    accounting only). Per hedge leg i at step t the desk posts
    `IM_i = IM_Vol_Multiplier·(σ_t/IM_Ref_Vol)·F_i·|q_i^post|·cs_i` and pays
    `IM_Funding_Spread_Bps·1e-4·dt` to fund it over the calendar step, above the risk-free the
    margin ledger already earns. Spread default 0.0 ⇒ the term is exactly 0 and never executes."""
    config = config if "Hedging_Problem" in config else config["Calc"]["Calculation"]
    hedging_problem = config["Hedging_Problem"]
    evaluator_config = hedging_problem["Evaluator"]
    objective_config = hedging_problem.get("Objective")
    solver_config = hedging_problem.get("Solver")
    execution_mode = str(config.get("Execution_Mode", "simulate_only")).lower()
    if execution_mode not in ("solve_hedge", "simulate_only"):
        raise ValueError(
            f"Unknown Execution_Mode {config.get('Execution_Mode')!r}; supported: 'solve_hedge' | "
            "'simulate_only'.")

    # --- instruments: the tradable universe splits into cash accounts and hedge legs ---
    tradables = _flatten_deals(hedging_problem["Tradable_Instruments"])
    cash_account_names = tuple(str(n) for n in evaluator_config.get("Cash_Instruments") or ())
    for account_name in cash_account_names:
        if account_name not in tradables:
            raise ValueError(
                f"Evaluator cash account '{account_name}' is not in Tradable_Instruments")
    hedge_names = tuple(n for n in tradables if n not in cash_account_names)
    if not hedge_names:
        raise ValueError("no hedge instruments: Tradable_Instruments has only cash accounts")

    # --- liabilities: the book being hedged; its latest expiry dates the hedge instruments ---
    liabilities = {
        name: {"deal_type": entry["deal_type"],
               "expiry_date": entry["params"].get("Expiry_Date"), "params": entry["params"]}
        for name, entry in _flatten_deals(hedging_problem.get("Liabilities") or {}).items()}
    liability_expiry = None
    for liability in liabilities.values():
        expiry_date = liability["expiry_date"]
        if expiry_date is not None and (liability_expiry is None or expiry_date > liability_expiry):
            liability_expiry = expiry_date
    normalized_tradables = {
        name: _instrument_metadata(name, entry, hedge_names=hedge_names,
                                   cash_account_names=cash_account_names,
                                   liability_expiry=liability_expiry)
        for name, entry in tradables.items()}

    if execution_mode == "solve_hedge":
        if solver_config is None:
            raise ValueError("Execution_Mode 'solve_hedge' requires Hedging_Problem['Solver']")
        if str(config.get("Inner_MC_Enabled", "No")) != "Yes":
            raise ValueError("Execution_Mode 'solve_hedge' requires Inner_MC_Enabled='Yes'")
        min_inner = (2 if str(solver_config.get("Object", "")).lower()
                     in ("diffsolver", "diffsolverv2") else 128)
        if int(config.get("Inner_Sub_Batch", 0)) < min_inner:
            raise ValueError(
                "Execution_Mode 'solve_hedge' requires Inner_Sub_Batch >= "
                f"{min_inner} for Solver.Object={solver_config.get('Object')!r}")
        if str(solver_config.get("Object", "")).lower() not in ("diffsolver", "diffsolverv2"):
            raise ValueError(
                "Execution_Mode 'solve_hedge' requires Solver.Object='DiffSolver' or "
                f"'DiffSolverV2' (the forward-backward solver); got "
                f"{solver_config.get('Object')!r}. HindsightDpSolver remains available as the "
                "Run_Hindsight_Diagnostic track.")
        # A solve is a STREAM: Simulation_Batches - 1 fit batches, then a held-out batch no fit
        # step saw. A loaded checkpoint fits nothing, so it is a stream of one.
        n_batches = int(config.get("Simulation_Batches", 1))
        if solver_config.get("DiffV2_Load_Value_Fn"):
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
        if solver_config.get("DiffV2_Load_Value_Fn") and solver_config.get("DiffV2_Save_Value_Fn"):
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

    # Objective dials are validated here because they are dispatched on downstream by EQUALITY:
    # both are consumed as `== 'running_wealth'` / `== 'conditional_sim'` deep in the solver, so an
    # unrecognised spelling would quietly mean the DEFAULT. `Utility_Scale_Mode` refuses its own.
    reference_mode = str((objective_config or {}).get("Reference_Mode", "Fixed")).lower()
    if reference_mode not in ("fixed", "running_wealth"):
        raise ValueError(
            f"Unsupported Objective.Reference_Mode: "
            f"{(objective_config or {}).get('Reference_Mode')!r}. Supported: 'Fixed' (the utility "
            f"is applied to TERMINAL wealth) | 'Running_Wealth' (to the day's wealth increment). "
            f"The value is dispatched on by equality, so a near-miss would silently run 'Fixed'.")
    # LogWealth's reward is the day's log ratio of capital + MTM, so it IS running-wealth-shaped
    # and a terminal Reference_Mode beside it is a contradiction the run refuses
    if str((objective_config or {}).get("Object", "")).lower() == "logwealth":
        rm_declared = (objective_config or {}).get("Reference_Mode")
        if rm_declared is not None and str(rm_declared).lower() == "fixed":
            raise ValueError(
                "Objective 'LogWealth' is a per-step growth objective (reward = log(W1/W0)) — "
                "Reference_Mode='Fixed' (terminal utility) contradicts it. Omit Reference_Mode "
                "or set 'Running_Wealth'.")
        reference_mode = "running_wealth"
    # every outer path shares L_0, so the raw dispersion at t=0 is exactly 0: a non-positive floor
    # makes c_0 = 0, x = (W-R)/c_0 infinite, and every label from step 0 a NaN
    floor_frac = float((objective_config or {}).get("Utility_Scale_Floor_Frac", 0.05))
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

    history_lookback = int(hedging_problem.get("History_Lookback_Business_Days", 30))
    if history_lookback < 0:
        raise ValueError("Hedging_Problem.History_Lookback_Business_Days must be non-negative")
    # Commodity names come from the live CommodityPrice factors the instruments created at
    # calc-dependency time — never re-parsed out of instrument JSON params.
    referenced_commodities = tuple(dict.fromkeys(
        utils.check_tuple_name(factor) for factor in (stoch_factors or {})
        if factor.type == 'CommodityPrice'))
    portfolio_state = hedging_problem.get("Portfolio_State") or {}
    scalar_spread_bps, spread_spec = _bid_offer_spread(evaluator_config)
    calendar_spread_bps = (float(evaluator_config["Calendar_Spread_Bps"])
                           if evaluator_config.get("Calendar_Spread_Bps") is not None else None)
    if calendar_spread_bps is not None:
        # a rate <= 0 makes matched volume free or PAID FOR, and `turnover_charge` then returns a
        # negative cost the argmax maximizes by churning; absence is the off switch
        if calendar_spread_bps <= 0.0:
            raise ValueError(
                f"Evaluator.Calendar_Spread_Bps must be > 0; got {calendar_spread_bps}. Omit the "
                f"key to price every contract at the outright spread.")
        # ...and a rate beside an explicit refusal to price spreads is a contradiction: the rate
        # says a matched roll is cheap, the switch says charge it twice
        if evaluator_config.get("Roll_As_Calendar_Spread") == "No":
            raise ValueError(
                "Evaluator.Calendar_Spread_Bps is set beside Roll_As_Calendar_Spread='No': the "
                "rate prices a matched roll as one calendar crossing and the switch refuses to. "
                "Drop one of them.")
    # static instrument -> cash_account routing by currency: the first cash account whose currency
    # matches wins, else the first account
    account_by_currency = {}
    for account_name in cash_account_names:
        account_by_currency.setdefault(
            normalized_tradables[account_name]["currency"], account_name)
    fallback_account = cash_account_names[0] if cash_account_names else None

    alloc_mode = str(evaluator_config.get("Allocation_Mode", "Exposure")).lower()
    if alloc_mode not in ("exposure", "carry_variance"):
        raise ValueError(
            f"Unsupported Evaluator.Allocation_Mode: "
            f"{evaluator_config.get('Allocation_Mode')!r}. Supported: 'Exposure' (the declared "
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
        "execution_mode": execution_mode,
        "accounting_mode": str(evaluator_config.get("Accounting_Mode", "futures")).lower(),
        "names": {
            "tradables": tuple(normalized_tradables),
            # the hedge legs ARE the action set; the solver builds its action grid over them
            "hedges": hedge_names,
            "cash_accounts": cash_account_names,
        },
        "referenced_commodities": referenced_commodities,
        "tradables": normalized_tradables,
        "liabilities": liabilities,
        "objective": None if objective_config is None else {
            # canonical lowercased form: every dispatch site compares against the lowercase literal
            "object": str(objective_config["Object"]).lower(),
            # utility-transform scale, consumed by any utility Object; `utility_scale` is mirrored
            # in from the bundle's resolved c (hedge_bundle.Bundle.mirror_utility_scale)
            "utility_scale_mode":
                str(objective_config.get("Utility_Scale_Mode", "vol_scaled_notional")).lower(),
            "utility_scale_explicit":
                (None if objective_config.get("Utility_Scale_Explicit") is None
                 else float(objective_config["Utility_Scale_Explicit"])),
            # Floor of the 'conditional_sim' schedule, as a fraction of its terminal entry —
            # inert under every other mode. Both dials are validated above.
            "utility_scale_floor_frac": floor_frac,
            # WHAT the utility is applied to: terminal wealth ('fixed'), or the DAY's wealth
            # increment ('running_wealth', a per-step reward the DP sums down the recursion).
            "reference_mode": reference_mode,
            # The benchmark wealth the utility is measured against, in DOLLARS (not c-units) —
            # every shape subtracts it before the /c scaling.
            "reference_wealth": float(objective_config.get("Reference_Wealth", 0.0)),
            # Utility SHAPE params (dimensionless, in units of c); Symlog ignores all of them.
            "huber_aversion": float(objective_config.get("Huber_Aversion", 2.5)),
            "huber_delta": float(objective_config.get("Huber_Delta", 1.0)),
            "up_aversion": float(objective_config.get("Up_Aversion", 0.0)),
            "up_knee": float(objective_config.get("Up_Knee", 0.15)),
            "cara_gamma": float(objective_config.get("CARA_Gamma", 1.0)),
        },
        "solver": _solver_config(solver_config),
        "history_lookback_business_days": history_lookback,
        "portfolio_state": {
            "positions": {str(n): float(v)
                          for n, v in portfolio_state.get("Positions", {}).items()},
            "cash_balances": {str(n): float(v)
                              for n, v in portfolio_state.get("Cash_Balances", {}).items()},
            "settlement_prices": {str(n): float(v)
                                  for n, v in portfolio_state.get("Settlement_Prices", {}).items()},
            "margin_balances": {str(n): float(v)
                                for n, v in portfolio_state.get("Margin_Balances", {}).items()},
            "initial_margin": {
                str(n): {"method": str(spec["Method"]), "amount": float(spec["Amount"])}
                for n, spec in portfolio_state.get("Initial_Margin", {}).items()},
            "spot_price_history": _spot_price_history(
                hedging_problem, history_lookback, referenced_commodities),
        },
        "accounting": {
            "position_limits": {
                str(n): {"min_position": int(limit["Min_Position"]),
                         "max_position": int(limit["Max_Position"])}
                for n, limit in evaluator_config.get("Position_Limits", {}).items()},
            "cash_accounts": {n: {"currency": normalized_tradables[n]["currency"]}
                              for n in cash_account_names},
            "instrument_to_cash_account": {
                n: account_by_currency.get(meta["currency"], fallback_account)
                for n, meta in normalized_tradables.items()},
            "transaction_cost_per_unit":
                float(evaluator_config.get("Transaction_Cost_Per_Unit", 0.0)),
            # Scalar half-spread bps (fast-path + diagnostic display); `spec` is None for a scalar
            # Bid_Offer_Spread_Bps and a normalized per-instrument/vol-scale dict otherwise —
            # resolved in `per_contract_kappa`.
            "bid_offer_spread_bps": scalar_spread_bps,
            "bid_offer_spread_spec": spread_spec,
            # when a rebalance offsets dq across adjacent maturities, the matched quantity pays one
            # calendar half-cost instead of two outright half-spreads. A `Calendar_Spread_Bps` rate
            # arms it on its own, so the solver's charge and the realized accounting's agree.
            "roll_as_calendar_spread":
                (evaluator_config.get("Roll_As_Calendar_Spread", "No") == "Yes"
                 or calendar_spread_bps is not None),
            "calendar_spread_bps": calendar_spread_bps,
            # Vol-linked IM funding charge on the post-trade book (realized accounting only).
            # Spread 0.0 ⇒ the term is exactly 0 and never executes.
            "im_funding_spread_bps": float(evaluator_config.get("IM_Funding_Spread_Bps", 0.0)),
            "im_vol_multiplier": float(evaluator_config.get("IM_Vol_Multiplier", 0.0)),
            "im_ref_vol": float(evaluator_config.get("IM_Ref_Vol", 1.0)),
            "force_flat_at_end": evaluator_config.get("Force_Flat_At_End", "Yes") == "Yes",
            "total_position_abs_limit":
                float(evaluator_config.get("Total_Position_Abs_Limit", 0.0)),
            "total_position_schedule": _position_schedule(evaluator_config),
            # Optional per-decision-step split of the NET cover across the legs. Present ⇒ the
            # action universe is the Q-ladder rather than the Cartesian product (see
            # HedgeActionSpace.allocation_grid); absent ⇒ today's grid, unchanged.
            "allocation_weights": _allocation_weights(evaluator_config, hedge_names),
            "allocation_mode": alloc_mode,
            # Per-leg |Δq| cap per decision step at the ARGMAX (0 = off). Execution policy
            # only — training labels never see it.
            "max_trade_per_step": float(evaluator_config.get("Max_Trade_Per_Step", 0.0)),
            # significance the argmax must beat the STANDING book by before it trades, in standard
            # errors of the paired inner-draw difference (0 = off). Execution policy only.
            "decision_deadband_sigma":
                float(evaluator_config.get("Decision_Deadband_Sigma", 0.0)),
        },
        "privileged_layout": derive_privileged_layout(stoch_factors),
    }


def per_contract_kappa(runtime, price, name, vol=None, calendar=False):
    """Per-contract turnover cost for tradable `name` at mark `price`: a flat
    `Transaction_Cost_Per_Unit` plus the half-spread charge on notional,
    `0.5 · spread_bps · 1e-4 · price · contract_size`, where `spread_bps` is the FULL quoted spread
    and the 0.5 is the mid-to-touch crossing. `price` is a scalar or tensor mark. Single source for
    the solver's decision-time kappa, the env's realized debit and the diagnostic CSV writer.

    `spread_bps` is the scalar `Bid_Offer_Spread_Bps` unless a spread SPEC is configured, in which
    case it is the instrument's `Per_Instrument` base (falling back to `Default_Bps`) scaled by
    `(vol/Ref_Vol)**Beta` when the spec declares a `Vol_Scale` and an annualized `vol` is supplied.
    `vol=None`, a scalar spread, or no `Vol_Scale` ⇒ vol-independent.

    `calendar=True` prices one contract of a CALENDAR SPREAD quoted against this leg instead: the
    spread is `Evaluator.Calendar_Spread_Bps` on this leg's notional under the same vol scale, and
    the flat fee is charged TWICE because a spread contract moves and clears two futures. Paired as
    `0.5·(κcal_i + κcal_j)` by `turnover_charge`, the result is both clearing fees plus one
    half-spread on the average leg notional.

    LIMITATION: `Calendar_Spread_Bps` is one scalar rate, so the `Per_Instrument` base does not
    carry over to the calendar leg and a per-maturity spread ladder is inexpressible. The OUTRIGHT
    half of every charge still reads each leg's own base."""
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
    return fee + 0.5 * spread_bps * scale * 1.0e-4 * price * contract_size


def turnover_charge(delta, kappa, kappa_cal=None):
    """THE turnover-cost rule for a whole reposition, as `per_contract_kappa` is the rule for one
    contract. `delta` is `(..., n_hedge)` in `names['hedges']` (maturity) order; `kappa` and
    `kappa_cal` are per-leg and indexed `[..., i]`, so a `(n_hedge,)` decision kappa and a
    per-path `(B, n_hedge)` realized one both work.

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
        return (delta.abs() * kappa).sum(dim=-1)
    rem = [delta[..., i] for i in range(delta.shape[-1])]
    cost = torch.zeros_like(rem[0])
    for i in range(len(rem) - 1):
        di, dj = rem[i], rem[i + 1]
        x = torch.minimum(di.abs(), dj.abs()) * ((di * dj) < 0)          # matched contracts
        rem[i], rem[i + 1] = di - di.sign() * x, dj - dj.sign() * x
        cost = cost + x * 0.5 * (kappa_cal[..., i] + kappa_cal[..., i + 1])
    for i, residual in enumerate(rem):
        cost = cost + residual.abs() * kappa[..., i]
    return cost


def initial_q_from_runtime(runtime, batch, device):
    """Per-hedge initial contract book `q0` `(batch, n_hedge)` from the normalized
    `Portfolio_State` positions, in `runtime['names']['hedges']` order - hedge legs only, cash
    accounts excluded. The seed the stepper applies to its opening positions, exposed so the
    solver's bank / verdict / benchmark tracks measure FIRST-step turnover from the real opening
    book rather than from flat.

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
    x = (x_dollars - float(obj.get("reference_wealth", 0.0))) / _utility_scale(obj, t)
    shape = obj["object"]
    if shape == "asymmetricutility_symlog":
        return torch.sign(x) * torch.log1p(x.abs())
    if shape == "asymmetricutility_huber":
        a = float(obj.get("huber_aversion", 2.5))
        d = float(obj.get("huber_delta", 1.0))
        au = float(obj.get("up_aversion", 0.0))
        k = float(obj.get("up_knee", 0.15))
        loss = (-x).clamp(min=0.0)
        quad = a * loss * loss
        lin = a * d * d + 2.0 * a * d * (loss - d)
        gain = x.clamp(min=0.0)
        quad_up = au * gain * gain
        lin_up = au * k * k + 2.0 * au * k * (gain - k)
        return (x - torch.where(loss <= d, quad, lin)
                - torch.where(gain <= k, quad_up, lin_up))
    # asymmetricutility_cara
    g = float(obj.get("cara_gamma", 1.0))
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
    c = _utility_scale(obj, t)
    x = (float(W) - float(obj.get("reference_wealth", 0.0))) / c
    shape = obj["object"]
    if shape == "asymmetricutility_symlog":
        f1 = 1.0 / (1.0 + abs(x))
        return f1 / c, math.copysign(f1, x) / c
    if shape == "asymmetricutility_huber":
        # the loss wing steepens the slope and the gain wing flattens it; curvature is +2a either way
        a, knee, d, s = ((float(obj.get("huber_aversion", 2.5)), float(obj.get("huber_delta", 1.0)),
                          -x, 1.0) if x < 0.0 else
                         (float(obj.get("up_aversion", 0.0)), float(obj.get("up_knee", 0.15)),
                          x, -1.0))
        f1 = 1.0 + s * 2.0 * a * min(d, knee)
        return f1 / c, (0.0 if d > knee else 2.0 * a / (f1 * c))
    g = float(obj.get("cara_gamma", 1.0))
    return math.exp(-g * x) / c, g / c


def _loss_side_curvature(runtime, t=None):
    """`(u'(R⁻), ARA(R⁻))` — the utility's slope and absolute risk aversion a hair BELOW the
    reference, which is the dose a hedging objective is actually run at.

    At EXACTLY R the closed form takes the gain wing by its side convention, and that wing's
    curvature is `Up_Aversion` — zero for every book that never armed it. Callers reporting a DOSE
    rather than a position (the frame-lock log, and the decision-curve dump under an increment
    objective, where the increment is ~0 before the day happens) want this one."""
    obj = runtime["objective"] or {}
    reference = float(obj.get("reference_wealth", 0.0))
    return _utility_local_curvature(reference - 1.0e-6 * _utility_scale(obj, t), runtime, t)


def _rolling_realized_vol(spot, window=PRICE_ZSCORE_WINDOW):
    """Annualized trailing-window realized log-vol of a `(T, B)` mark path, `(T, B)` — the one
    spelling behind both the per-commodity vol surface and the per-step spread/IM-funding proxy."""
    log_ret = spot.clamp_min(1e-9).log().diff(dim=0)                                  # (T-1, B)
    ret_sq = torch.cat([torch.zeros_like(log_ret[:1]), log_ret * log_ret], dim=0)     # (T, B)
    cum = torch.cat([torch.zeros_like(ret_sq[:1]), ret_sq.cumsum(dim=0)], dim=0)      # (T+1, B)
    idx = torch.arange(ret_sq.shape[0] + 1, device=spot.device)
    lo = (idx - window).clamp_min(0)
    count = (idx - lo).to(dtype=torch.float32).clamp_min(1.0).unsqueeze(-1)
    return (252.0 * (cum[idx[1:]] - cum[lo[1:]]) / count[1:]).clamp_min(0.0).sqrt()   # (T, B)


def _history_key(commodity_field):
    """The `spot_price_history` key for a deal's raw `Commodity` field: the PRIMARY spot's full
    factor name, which is the key space `_spot_price_history` validates the dict against. The deal
    carries the bare name (`'PLATINUM_CME'`, or a composed `'PLATINUM_CME.LME_CME'` whose history
    is its primary's), and keying the lookup by that raw field can never match."""
    if not commodity_field:
        return None
    return utils.check_tuple_name(
        utils.Factor('CommodityPrice', utils.check_rate_name(commodity_field)[:1]))


def _roll_rebate(deltas, prices, runtime, vol=None):
    """Calendar-spread rebate for a step's REALIZED turnover, armed by
    `Evaluator.Roll_As_Calendar_Spread='Yes'` or by a `Calendar_Spread_Bps` rate on its own.
    A rebalance that REDUCES one contract month and INCREASES an adjacent month is a roll: the
    matched quantity should pay a single calendar-spread crossing, not two independent outright
    half-spreads. The env debits the full L1 turnover leg by leg; this is what it CREDITS back,
    `(B,)` per path.

    The matching and the pricing are `turnover_charge` — the same rule the solver's
    decision charge reads, so the realized cost of a reposition and the cost the argmax ranked it
    at are one function of one move. Absent a `Calendar_Spread_Bps` the matched leg falls back to
    half the sum of the two OUTRIGHT kappas, by handing that rule the outright kappa as the
    calendar one. `deltas`/`prices` key by hedge name; `vol` is the step's annualized vol."""
    hedges = list(runtime["names"]["hedges"])
    delta = torch.stack([deltas[h].to(torch.float32) for h in hedges], dim=-1)   # (B, n_hedge)
    kappa = torch.stack([per_contract_kappa(runtime, prices[h].abs(), h, vol)
                         for h in hedges], dim=-1)
    cal = (torch.stack([per_contract_kappa(runtime, prices[h].abs(), h, vol, True)
                        for h in hedges], dim=-1)
           if runtime["accounting"]["calendar_spread_bps"] is not None else kappa)
    return (delta.abs() * kappa).sum(-1) - turnover_charge(delta, kappa, cal)


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


def _portfolio_value(state, runtime):
    """Absolute total wealth: cash + margin + unrealized VM + position value where applicable.
    `_pnl_excess` is the wealth change since inception, which is what the asymmetric utility needs.

    Cash and margin balances are raw dollars compounded daily (each step multiplies by
    `cash_tv(next)/cash_tv(curr)` ≈ 1 + SOFR·dt, then adds today's flows), so each dollar earns
    interest only from the day it landed.

    Two accounting modes. `cash_account`: position × price × contract_size plus the cash balances.
    Futures: cash is FROZEN at starting capital and margin carries every VM and trade-cost flow
    since inception, so the unrealized `position × (price − last_settlement) × cs` term closes the
    gap between the most recent settlement and the current observable price — zero within an
    episode, non-zero at t=0 when the book opens with an overnight position."""
    hedges = runtime["names"]["hedges"]
    tradables = runtime["tradables"]
    positions, prices = state["positions"], state["tradable_values"]
    total = torch.zeros_like(state["done"], dtype=torch.float32)
    if runtime["accounting_mode"] == "cash_account":
        for name in hedges:
            total = total + (positions[name].to(dtype=torch.float32)
                             * prices[name].to(dtype=torch.float32)
                             * float(tradables[name]["contract_size"]))
        cash = None
        for balance in state["cash_accounts"].values():
            v = balance.to(dtype=torch.float32)
            cash = v if cash is None else cash + v
        return total if cash is None else total + cash
    # futures mode: margin carries the flows, the unrealized term closes the settlement/price gap
    for accounts in (state["margin_accounts"], state["cash_accounts"]):
        balances = None
        for balance in accounts.values():
            v = balance.to(dtype=torch.float32)
            balances = v if balances is None else balances + v
        if balances is not None:
            total = total + balances
    settlements = state["settlement_prices"]
    for name in hedges:
        if name not in settlements or name not in prices:
            continue
        total = total + (positions[name].to(dtype=torch.float32)
                         * (prices[name].to(dtype=torch.float32)
                            - settlements[name].to(dtype=torch.float32))
                         * float(tradables[name]["contract_size"]))
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
    """The hedge-quality read — `pnl_excess + liability_mtm + cumulative_liability_value`, which
    an optimal hedge holds at ≈0 at EVERY step.

    All three channels are required: `liability_mtm` alone drops by the cashflow amount on a
    payment date, injecting a ±cf shock unrelated to any action. The sum across the two liability
    channels is continuous across that boundary and matches the terminal invariant."""
    pnl_excess = _pnl_excess(state, runtime).to(dtype=torch.float32)
    liability_mtm = state["liability_mtm_value"].to(dtype=torch.float32, device=pnl_excess.device)
    cumulative = state["cumulative_liability_value"].to(dtype=torch.float32, device=pnl_excess.device)
    return pnl_excess + liability_mtm + cumulative


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
    grid, so full-grid indexing is `initial_time_index + t`; the `*_sim` views strip that prefix
    for solver code that indexes by simulation-grid `t`.

    The frame — `utility_scale` (the objective's scale c, with its per-decision-step
    `utility_scale_schedule` under `Utility_Scale_Mode='conditional_sim'`, both mirrored onto the
    runtime objective) and `step_annual_vol` (the per-step vol driving the state-dependent spread
    and IM funding) — is resolved here and never recomputed: a per-rollout c silently rescales
    every reward. `inner_mc` / `inner_mc_grad` are attached by `HedgeMonteCarlo.execute` in
    solve_hedge mode; they fork the simulator at a decision step and price the {t, t+1} window."""

    documentation = ('Bundle', [
        'Every time-indexed tensor the solver and the environment read, built ONCE per batch by',
        '`from_batch` and read-only afterwards. Consumers index it by attribute; nothing recomputes',
        'it.',
        '',
        '- **Tradables** — `tradables` / `tradables_sim`: the per-instrument mark series, harvested',
        '  from the priced deal tree via `DealStructure.tensor_marks`. An instrument that fails to',
        '  price is ABSENT rather than zero, and a fork reads absence as an expired contract.',
        '- **Liability** — `liability_mtm` / `liability_sim`, plus `realized_cashflows`, and',
        '  `unfixed_at`: the signed quantity its fixings still carry after a row.',
        '- **Factors** — the simulated factor paths, and `privileged_factors`: whatever each process',
        '  publishes as market state.',
        '- **Frame** — `utility_scale` (the objective scale `c`), its optional per-decision-step',
        '  schedule `utility_scale_schedule` (`Objective.Utility_Scale_Mode=\'conditional_sim\'`:',
        '  the cross-sectional dispersion of the flat-book wealth at each step, floored), and',
        '  `step_annual_vol` (the per-step vol driving state-dependent spreads and IM funding).',
        '',
        'The frame is resolved at build time and never recomputed. A per-rollout `c` would silently',
        'rescale every reward, which is why a loaded checkpoint evaluates in ITS frame rather than',
        "the evaluation world's.",
        '',
        '!!! warning "The history prefix"',
        '    Time tensors carry `History_Lookback_Business_Days` realized rows in FRONT of the',
        '    simulated grid, so full-grid indexing is `initial_time_index + t` while the `*_sim`',
        '    views strip the prefix. Indexing a full-grid tensor by simulation `t` silently reads',
        '    history.',
        '',
        'In `solve_hedge` mode `HedgeMonteCarlo.execute` attaches `inner_mc` / `inner_mc_grad`,',
        'which fork the simulator at a decision step and price the `{t, t+1}` window on demand.',
    ])

    def __init__(self, base_date=None, business_day=None, num_batches=1):
        self.base_date = base_date
        self.business_day = business_day
        self.num_batches = int(num_batches)
        # Time axis (history prefix + simulated grid).
        self.time_grid_days = None          # (T,) day offsets from base_date
        self.time_grid_days_cpu = []        # the same as python ints (sync-free step loop)
        self.scenario_dates = None          # DatetimeIndex over the full grid
        self.business_indices = ()          # decision steps: business days in the sim window
        self.initial_time_index = 0         # history/sim boundary
        self.history_rows = 0               # History_Lookback_Business_Days prefix rows
        # Simulated series.
        self.tradables = {}                 # {instrument: (T, B)} marks
        self.factors = {}                   # {factor name: (T, …, B)} scenario paths
        self.privileged_factors = {}        # {name: (T, B, dim)} process-revealed state
        self.liability_mtm = None           # (T, B) marked liability, or None
        self.realized_cashflows = {}        # {currency: (T, B)} paid cashflows
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
        self.last_settlement_index = None
        self.last_live_mtm_index = 0
        # inner-MC forks, attached by HedgeMonteCarlo.execute for Execution_Mode='solve_hedge'
        self.inner_mc = None
        self.inner_mc_grad = None

    # ---- construction --------------------------------------------------------
    @classmethod
    def from_batch(cls, base_date, business_day, time_grid_days, tradable_blocks,
                   factor_tensor_blocks, hedge_profile_blocks, num_batches, stoch_factors,
                   runtime, privileged_factor_blocks=None, total_leg_volume=None,
                   last_payment_day=None, fixings=()):
        """Assemble the per-batch tensor blocks the HedgeMonteCarlo simulator produced into one
        bundle: join them along the batch axis, then run the four resolve stages.

        `stoch_factors` is the simulator's factor map (factor_key → process); the processes are
        asked for their privileged surfaces, calibrated vols and revealed conditional vol."""
        self = cls(base_date, business_day, num_batches)
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
        self._resolve_liability(mtm, cashflows, steps, total_leg_volume, last_payment_day, fixings)
        self._resolve_factors(factors, privileged_factor_blocks or {}, stoch_factors, steps)
        self._resolve_frame(runtime, stoch_factors)
        return self

    def _resolve_tradables(self, tradables, time_grid_days, steps, runtime):
        """Stage 1 — the time axis and the per-instrument mark series, both carrying the realized
        history prefix. `History_Lookback_Business_Days` rows of realized spot are prepended so a
        rolling-window feature at sim-day-0 already has its lookback; a tradable whose `Commodity`
        has a history takes that series, every other series broadcasts its first row. Also derives
        the CPU day mirror, the scenario dates, `initial_time_index` and the decision indices."""
        grid = _align_time_axis(time_grid_days, steps)
        tradables = {n: _align_time_axis(t, steps) for n, t in tradables.items()}
        history = runtime['portfolio_state']['spot_price_history']
        self.history_rows = H = int(runtime['history_lookback_business_days'])
        base_ts = pd.Timestamp(self.base_date)
        if H > 0 and history:
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
        self.business_indices = tuple(
            i for i in range(max(len(self.scenario_dates) - 1, 0))
            if i >= self.initial_time_index and self.business_day.is_on_offset(self.scenario_dates[i]))

    def _history_prefix(self, tensor, realized=None):
        """The `history_rows` prepended to a simulated series: the realized history broadcast
        across the batch when the series has one, else the series' own first row repeated."""
        if realized is not None:
            return realized.to(dtype=tensor.dtype)
        return tensor[:1].expand((self.history_rows,) + tuple(tensor.shape[1:])).contiguous()

    def _resolve_liability(self, mtm, cashflows, steps, total_leg_volume, last_payment_day,
                           fixings):
        """Stage 2 — the marked liability, the realized cashflow ledger, the two schedule
        scalars the utility scale needs and the liability's fixings `unfixed_at` reads.
        `last_live_mtm_index` is the structural pre-settlement terminal: the grid appends one
        clean-exit row where the liability settles to zero, so the last LIVE mtm row is
        `steps - 2` — the single source for the DP depth and the L_T read."""
        prefixed = bool(self.spot_price_history)          # the stage-1 prefix gate
        if mtm is not None:
            mtm = _align_time_axis(mtm, steps)
            if prefixed:
                mtm = torch.cat([self._history_prefix(mtm), mtm], dim=0)
        self.liability_mtm = mtm
        self.realized_cashflows = {}
        for currency, tensor in cashflows.items():
            tensor = _align_time_axis(tensor, steps)
            if prefixed:
                # Nothing was paid before the sim starts — the prefix is zeros, not a repeat.
                tensor = torch.cat([torch.zeros((self.history_rows,) + tuple(tensor.shape[1:]),
                                                dtype=tensor.dtype, device=tensor.device),
                                    tensor], dim=0)
            self.realized_cashflows[currency] = tensor
        self.last_live_mtm_index = int(steps - 2)
        if total_leg_volume:
            self.total_leg_volume = float(total_leg_volume)
        self.fixings = tuple(fixings)
        if last_payment_day is not None:
            # Last (history-prefixed) grid step still strictly before the final payment.
            pending = [i for i, d in enumerate(self.time_grid_days_cpu) if d < last_payment_day]
            if pending:
                self.last_settlement_index = pending[-1]
        logging.info('liability scalars: total_leg_volume=%s last_settlement_index=%s',
                     self.total_leg_volume or None, self.last_settlement_index)

    def _resolve_factors(self, factors, privileged_factor_blocks, stoch_factors, steps):
        """Stage 3 — the simulated factor paths and the per-process privileged surfaces (the
        market state the value function consumes), both prefixed onto the history rows."""
        self.factors = {n: _align_time_axis(t, steps) for n, t in factors.items()}
        if self.spot_price_history:
            self.factors = {n: torch.cat([self._history_prefix(t), t], dim=0)
                            for n, t in self.factors.items()}
        privileged = assemble_privileged_factors(privileged_factor_blocks, stoch_factors)
        # the privileged prefix is gated on the lookback ALONE, so with a lookback but no
        # Spot_Price_History these rows are prepended while nothing else is, which offsets the
        # surface by `history_rows`
        if self.history_rows > 0 and privileged:
            privileged = {n: torch.cat([self._history_prefix(t), t], dim=0)
                          for n, t in privileged.items()}
        self.privileged_factors = privileged

    def _resolve_frame(self, runtime, stoch_factors):
        """Stage 4 — the LOCKED frame: the realized-vol surface, the utility scale c and (under
        `Utility_Scale_Mode='conditional_sim'`) its per-decision-step schedule, both mirrored onto
        the runtime objective so every reward/penalty reads one frame, and the per-step vol series.
        Nothing here may be recomputed per rollout: a drifting c silently rescales the whole
        objective, and the vol series drives realized transaction cost."""
        self.spot_realized_vol = self._spot_realized_vol()
        if not self.spot_price_history:
            # No Spot_Price_History ⇒ source (spot, σ) from the CALIBRATED market data instead.
            self.calibrated_utility_inputs = self._calibrated_utility_inputs(runtime, stoch_factors)
        self.utility_scale = self._resolve_utility_scale(runtime)
        logging.info('utility_scale (symlog c) resolved to {0:.2f}'.format(self.utility_scale))
        self.mirror_utility_scale(runtime)
        if self.utility_scale_schedule is not None:
            self._log_scale_dose(runtime)
        self.step_annual_vol = self._resolve_step_vol(runtime, stoch_factors)

    def mirror_utility_scale(self, runtime):
        """Cache the resolved `c` on `runtime['objective']` so every reward, penalty and utility
        transform reads one number without taking the bundle in its signature. Invariant:
        `runtime['objective']['utility_scale'] == bundle.utility_scale` for any rollout that
        computes rewards against this bundle.

        The per-step SCHEDULE travels with it, so a caller that keeps a checkpoint's scale
        (`BundleStepper(mirror_scale=False)`) keeps its knees too."""
        if runtime['objective'] is not None:
            runtime['objective']['utility_scale'] = float(self.utility_scale)
            runtime['objective']['utility_scale_schedule'] = self.utility_scale_schedule

    def _spot_timeline(self, commodity):
        """The (H+T_sim, B) spot tensor for `commodity`: JSON history rows 0..H-1 concatenated
        with the simulator's CommodityPrice factor from row H. `commodity` is the canonical
        factor name (`utils.check_tuple_name`), which both dicts key by."""
        hist = self.spot_price_history.get(commodity)
        sim = self.factors.get(commodity)
        if hist is None or sim is None:
            return None
        sim_full = sim.to(dtype=torch.float32)
        hist_t = hist.to(dtype=torch.float32, device=sim_full.device)
        return torch.cat([hist_t, sim_full[int(hist_t.shape[0]):]], dim=0)

    def _spot_realized_vol(self, window=PRICE_ZSCORE_WINDOW, min_periods=5):
        """Annualized rolling realized log-vol of each underlying spot, `{commodity: (T, B)}`. In
        mean-reverting regimes with σ scaled to hold the stationary std fixed, realized vol rises
        with kappa, which makes this a regime signal as well as the utility-scale σ source."""
        out = {}
        for commodity in self.spot_price_history:
            S = self._spot_timeline(commodity)
            if S is None:
                continue
            rv = _rolling_realized_vol(S, window)
            rv[:min_periods] = 0.0
            out[commodity] = rv
        return out

    def _calibrated_utility_inputs(self, runtime, stoch_factors):
        """Utility-scale `(commodity, spot, σ)` sourced from CALIBRATED market data — the
        `Spot_Price_History`-absent fallback. Spot is the sim-day-0 CommodityPrice level; σ is the
        process's calibrated annualized vol. With several CommodityPrice factors the
        sufficient-statistic-owning primary is preferred. None when none reports a calibrated
        vol."""
        referenced = set(runtime['referenced_commodities'])
        candidates = []
        for key, proc in (stoch_factors or {}).items():
            name = utils.check_tuple_name(key)
            if name not in referenced or name not in self.factors:
                continue
            sigma = proc.calibrated_annual_vol()
            if sigma is None or sigma <= 0.0:
                continue
            # spots exposing a revealed sufficient statistic (HMM belief, GARCH log-variance) sort
            # first: the martingale primary of a cross-market strip
            candidates.append((not bool(proc.privileged_layout(proc.param)), name, float(sigma)))
        if not candidates:
            return None
        candidates.sort()
        _, commodity, sigma = candidates[0]
        return (commodity, float(self.factors[commodity][0].median().item()), sigma)

    def _resolve_utility_scale(self, runtime):
        """The dollar scale `c` mapping dollars to utility (u(x; c) = sign(x)·log1p(|x|/c) for
        symlog). Single source of truth — every reward and penalty reads `Bundle.utility_scale`.

        Modes (`Objective.Utility_Scale_Mode`): `vol_scaled_notional` (default) gives
        c = total_leg_volume × initial_spot × σ_annual × √τ; `conditional_sim` measures a
        per-decision-step SCHEDULE off this batch instead and takes its terminal entry as the
        scalar (`_conditional_sim_schedule`); `Objective.Utility_Scale_Explicit` overrides the
        formula with a literal dollar value and is REFUSED beside the schedule mode.

        Under a utility objective every degenerate path RAISES: a floor-c symlog breaks tail
        compression (log1p($1M/$1k) ≈ 7 against log1p($100M/$1k) ≈ 11.5 — a 100x dollar gap
        becomes a 1.6x utility gap). The identity objective does not consume c and gets the $1k
        floor.

        The history path reads spot and realized σ at the history/sim boundary H as a BATCH MEDIAN
        rather than slot [H, 0]: `full[H]` is the first sim step, and a process emitting a
        stochastic initial draw would otherwise make c path-dependent off path 0."""
        objective = runtime['objective'] or {}
        needs_scale = _is_utility_objective(runtime)   # symlog / huber / cara all consume c

        def _degenerate(reason):
            if needs_scale:
                raise ValueError(
                    f"utility scale: cannot compute a meaningful c — {reason}. "
                    "A floor-c symlog silently compresses tails and defeats the reward shape. "
                    "Fix the bundle/config, or set Objective.Utility_Scale_Explicit to a "
                    "literal dollar value.")
            return 1.0e3

        mode = str(objective.get('utility_scale_mode', 'vol_scaled_notional')).lower()
        if mode not in ('vol_scaled_notional', 'conditional_sim'):
            raise ValueError(
                f"Unsupported Objective.Utility_Scale_Mode: {mode!r}. "
                "Supported modes: 'vol_scaled_notional' | 'conditional_sim'. Set "
                "Utility_Scale_Explicit to override the formula with a literal dollar value.")
        explicit = objective.get('utility_scale_explicit')
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
            # an EXPLICIT override is honored exactly, including below the $1k production floor,
            # which only guards the formula path
            c_explicit = float(explicit)
            if c_explicit < 1.0e3:
                logging.info(
                    'utility_scale Explicit override: c=%.4g (below the $1k '
                    'production floor — test mode; trust mode active)', c_explicit)
            return c_explicit
        # tau measures the SIM horizon, so anchor it at the sim-grid origin
        H = self.initial_time_index
        if self.last_settlement_index is None:
            return _degenerate("last_settlement_index missing from the bundle")
        tau_years = max(float(int(self.last_settlement_index) - H) / 252.0, 1.0 / 252.0)
        if not self.spot_price_history and self.calibrated_utility_inputs is None:
            return _degenerate(
                "spot_price_history is empty and no calibrated underlying vol is available "
                "(no referenced CommodityPrice process reports calibrated_annual_vol)")
        if not self.total_leg_volume:
            return _degenerate("total_leg_volume is zero")
        if self.spot_price_history:
            # history path: spot and realized sigma read at the history/sim boundary H
            commodity = next(iter(self.spot_price_history))
            full = self._spot_timeline(commodity)
            if full is None or H >= int(full.shape[0]):
                return _degenerate(
                    f"spot timeline for {commodity!r} has length "
                    f"{0 if full is None else int(full.shape[0])} ≤ history_lookback H={H}")
            # batch-median at H, never slot [H, 0]: keeps c path-independent under a stochastic
            # initial draw
            initial_spot = float(full[H].median().item())
            rv = self.spot_realized_vol.get(commodity)
            sigma = float(rv[H].median().item()) if rv is not None and H < int(rv.shape[0]) else 0.0
        else:
            commodity, initial_spot, sigma = self.calibrated_utility_inputs
        if sigma <= 0.0:
            return _degenerate(
                f"realized vol for {commodity!r} is non-positive (σ={sigma}) — "
                "likely early-calibration window or upstream calc bug")
        c = self.total_leg_volume * initial_spot * sigma * (tau_years ** 0.5)
        if c < 1.0e3:
            return _degenerate(
                f"formula produced c=${c:,.2f} < $1k floor "
                f"(volume={self.total_leg_volume}, spot={initial_spot:.2f}, σ={sigma:.4f}, "
                f"√τ={tau_years ** 0.5:.3f})")
        return c

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
        own units. `BundleStepper._tracking_error_value` is the same read one layer down, but it is
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
        if self.liability_mtm is None:
            raise ValueError(
                "Objective.Utility_Scale_Mode='conditional_sim' needs a marked liability: the "
                "schedule is the dispersion of the flat-book wealth path and this bundle has no "
                "liability_mtm to measure it on.")
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
        realized-vol proxy off the primary spot factor path. Batch-reduced to one scalar per step,
        matching the solver's mean-mark kappa. The chosen source is logged once per build: a silent
        proxy fallback under a GARCH retrain would invalidate the whole vol coupling."""
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
        commodity = next((c for c in runtime['referenced_commodities'] if c in self.factors), None)
        if commodity is None:
            return None
        logging.info('step_annual_vol source: realized-vol proxy on %s (Vol_Scale=%s IM_funding=%s)',
                     commodity, vol_scale_active, im_funding_active)
        # batch-reduced to one scalar per step, floored so the warm-up window never zeros the spread
        return _rolling_realized_vol(self.factors[commodity]).mean(dim=-1).clamp_min(1e-6)

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
        """History-stripped liability MTM `(t_outer, B)`."""
        return self.liability_mtm[self.initial_time_index:]

    @property
    def n_outer_steps(self):
        return int(self.liability_sim.shape[0])

    @property
    def vol_sim(self):
        """Sim-grid slice of the per-step vol series (or None when no Vol_Scale/IM funding)."""
        v = self.step_annual_vol
        return None if v is None else v[self.initial_time_index:]

    def realized_paths(self, runtime):
        """The realized outer-path data the no-inner-MC tracks (hindsight, textbook) consume:
        `F` `(n_hedge, t_outer, B)` hedge prices, `L_T` `(B,)` the liability terminal MTM at the
        pre-settlement `last_live_mtm_index`, and `t_outer`."""
        tradables_sim = self.tradables_sim
        F = torch.stack([tradables_sim[h] for h in runtime['names']['hedges']], dim=0)
        return F, self.liability_sim[self.last_live_mtm_index], self.n_outer_steps

    def liability_at(self, time_index):
        """Marked liability `(B,)` at full-grid step `time_index` (zeros when there is none)."""
        if self.liability_mtm is None:
            return torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)
        return self.liability_mtm[time_index].to(dtype=torch.float32, device=self.device)

    def cashflow_at(self, time_index):
        """Realized cashflow `(B,)` paid at full-grid step `time_index`, summed over currencies."""
        total = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)
        for tensor in self.realized_cashflows.values():
            total = total + tensor[time_index].to(dtype=torch.float32, device=self.device)
        return total

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
    integer contracts, and the terminal forced flatten. Each step returns a state dict the caller
    inspects; the caller chooses the next action. Supports `copy.deepcopy(stepper)` to fork into
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
        'contracts, and the terminal forced flatten.',
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
        self._decision_set = set(int(i) for i in bundle.business_indices)
        # the default re-mirrors THIS bundle's scale; False keeps a checkpoint's frame
        if mirror_scale:
            bundle.mirror_utility_scale(runtime)
        self._state = self._initial_state()
        # per-decision recording, always on so write_diagnostic_csvs has data to use
        self._times = []
        self._position_history = {n: [] for n in self._hedges}
        self._trade_history = {n: [] for n in self._hedges}
        self._price_history = {n: [] for n in self._hedges}
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
            'cumulative_liability_value': self._state['cumulative_liability_value'],
        }

    def step(self, action=None):
        """Advance one time step. `action` is `{instrument_name: scalar_or_(B,)_tensor}` applied
        as trade deltas (only meaningful at decision steps; ignored otherwise). Pass `None` for
        zero trades. Returns the post-step observation plus this transition's per-path
        pnl_excess + liability_value."""
        was_decision_step = self.is_decision_step
        if was_decision_step:
            # record the pre-step position and price for the diagnostic CSV
            self._times.append(self.time_index)
            for n in self._hedges:
                self._position_history[n].append(self._state['positions'][n].detach().cpu().clone())
                self._price_history[n].append(self._state['tradable_values'][n].detach().cpu().clone())
        structured = (self._structured_action(action)
                      if (action is not None and was_decision_step) else None)
        next_state = self._step_state(self._state, structured)
        transition = self._payoff(next_state)
        if was_decision_step:
            # realized trade = post-step position - pre-step position, so env clips and forces land
            for n in self._hedges:
                self._trade_history[n].append(
                    next_state['positions'][n].detach().cpu() - self._position_history[n][-1])
        if self._batch_size == 1 and logging.getLogger().isEnabledFor(logging.DEBUG):
            # reconciliation organ for B=1 rolls, every step so the daily identities close exactly
            cur = self.time_index
            deltas = self._trade_deltas(structured)
            vol_t = self.bundle.vol_at(cur)
            cost = sum(float(self._cost(deltas[n], self._state['tradable_values'][n], n,
                                        vol_t)[0]) for n in self._hedges)
            reb = (float(_roll_rebate(deltas, {h: self._state['tradable_values'][h]
                                               for h in self._hedges}, self.runtime, vol_t)[0])
                   if self._accounting['roll_as_calendar_spread'] else 0.0)
            legs = {n: (float(self._state['tradable_values'][n][0]),
                        int(next_state['positions'][n][0]),
                        int(deltas[n][0])) for n in self._hedges}
            logging.debug(
                'STEP %s t=%d dec=%d legs(mark,q,dq)=%s vm=%.6f cost=%.6f reb=%.6f margin=%.6f '
                'cash_tv=%s pnl_excess=%.6f liability=%.6f',
                self.bundle.scenario_dates[cur].date(), cur, int(was_decision_step), legs,
                sum(float(v[0]) for v in next_state.get('variation_margin', {}).values()),
                cost, reb, sum(float(b[0]) for b in next_state['margin_accounts'].values()),
                {n: float(self._state['tradable_values'][n][0]) for n in self._cash_names
                 if n in self._state['tradable_values']},
                float(transition['pnl_excess'][0]), float(transition['liability_value'][0]))
        self._state = next_state
        self._terminal_transition = transition
        return {
            **self.observe(),
            'transition_pnl_excess': transition['pnl_excess'],
            'transition_liability_value': transition['liability_value'],
        }

    def evaluation_output(self, timing=None):
        """Terminal P&L summary in the `evaluation_summary` shape `HedgeMonteCarlo.execute`
        publishes. `simulate_only` rolls the no-trade policy, so the policy and the reference
        baseline are the same trajectory and the difference block is identically zero."""
        hedge_pnl = self._terminal_transition['pnl_excess'].detach().to(dtype=torch.float32)
        liability = self._terminal_transition['liability_value'].detach().to(dtype=torch.float32)
        net_pnl = hedge_pnl + liability
        metrics = {
            'average_net_pnl': float(net_pnl.mean().item()),
            'median_net_pnl': float(torch.quantile(net_pnl.to(dtype=torch.float64), 0.5).item()),
            'worst_net_pnl': float(net_pnl.min().item()),
            'average_hedge_pnl': float(hedge_pnl.mean().item()),
            'average_liability': float(liability.mean().item()),
        }
        cpu = lambda values: {str(n): t.detach().to(dtype=torch.float32).cpu()
                              for n, t in values.items()}
        return {
            'metrics': metrics,
            'final_state': {
                'positions': cpu(self._state['positions']),
                'cash_accounts': cpu(self._state['cash_accounts']),
                'margin_accounts': cpu(self._state['margin_accounts']),
                'pnl_excess': hedge_pnl.cpu(),
                'liability_value': liability.cpu(),
                'net_pnl': net_pnl.cpu(),
            },
            'diagnostics': {'num_episodes': int(net_pnl.shape[0]),
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
        net = rollout['net_pnl'].numpy()
        total = fields['total_ex_funding_discount'][-1].numpy()
        pd.DataFrame([
            {'policy': label, 'metric': 'mean', 'net_pnl': float(net.mean()),
             'total_ex_funding': float(total.mean())},
            {'policy': label, 'metric': 'std', 'net_pnl': float(net.std()),
             'total_ex_funding': float(total.std())},
            {'policy': label, 'metric': 'min', 'net_pnl': float(net.min()),
             'total_ex_funding': float(total.min())},
            {'policy': label, 'metric': 'p5', 'net_pnl': float(np.percentile(net, 5)),
             'total_ex_funding': float(np.percentile(total, 5))},
            {'policy': label, 'metric': 'p95', 'net_pnl': float(np.percentile(net, 95)),
             'total_ex_funding': float(np.percentile(total, 95))},
            {'policy': label, 'metric': 'max', 'net_pnl': float(net.max()),
             'total_ex_funding': float(total.max())},
        ]).to_csv(os.path.join(output_dir, f'{label}_summary.csv'), index=False,
                  float_format='%.2f')

    # ---- state construction + transition -------------------------------------
    def _zeros_by_name(self, names):
        return {str(n): torch.zeros(self._batch_size, dtype=torch.float32, device=self._device)
                for n in names}

    def _seed_by_name(self, seed_values, names):
        return {str(n): torch.full((self._batch_size,), float(seed_values.get(str(n), 0.0)),
                                   dtype=torch.float32, device=self._device) for n in names}

    def _values_at(self, time_index):
        """Current mark `{instrument: (B,)}` at full-grid `time_index`."""
        tradables = self.bundle.tradables
        return {n: tradables[n][time_index].to(dtype=torch.float32)
                for n in self._tradable_names if n in tradables}

    def _initial_state(self):
        """The opening state at sim-day-0 (bundle row `initial_time_index`) — the history prefix
        feeds features only, never the simulator, so JSON-supplied positions are today's overnight
        book rather than H days ago.

        `settlement_prices` are re-seated to the sim-day-0 price once the inception baseline is
        snapshotted: the seed-to-sim-day-0 gap has just been absorbed into
        `initial_portfolio_value` through the unrealized-VM term, so subsequent VM is clean
        step-over-step P&L rather than carrying `(price_H − seed) × delta × cs`."""
        portfolio_state = self.runtime['portfolio_state']
        initial_time_index = self.bundle.initial_time_index
        fallback = self._values_at(initial_time_index)
        seeded_settlement = portfolio_state['settlement_prices']
        state = {
            'done': torch.zeros(self._batch_size, dtype=torch.bool, device=self._device),
            'positions': self._seed_by_name(portfolio_state['positions'], self._hedges),
            'cash_accounts': self._seed_by_name(portfolio_state['cash_balances'], self._cash_names),
            'margin_accounts': self._seed_by_name(portfolio_state['margin_balances'], self._cash_names),
            'realized_pnl': self._zeros_by_name(self._hedges),
            'variation_margin': self._zeros_by_name(self._hedges),
            'cumulative_pnl': self._zeros_by_name(self._hedges),
            'time_held': self._zeros_by_name(self._hedges),
            'cumulative_liability_value': torch.zeros(self._batch_size, dtype=torch.float32,
                                                      device=self._device),
            'settlement_prices': {
                name: torch.full((self._batch_size,), float(seeded_settlement[name]),
                                 dtype=torch.float32, device=self._device)
                if name in seeded_settlement else fallback[name].clone()
                for name in self._hedges if name in fallback},
        }
        state = self._refresh(state, initial_time_index)
        # snapshot the inception baseline so `_pnl_excess` returns the change
        state['initial_portfolio_value'] = _portfolio_value(state, self.runtime).detach().clone()
        # re-seat settlement to the sim-day-0 price; the seed gap now lives in the baseline
        for name, current_price in state['tradable_values'].items():
            if name in state['settlement_prices']:
                state['settlement_prices'][name] = current_price.detach().clone()
        return state

    def _refresh(self, state, time_index):
        """Cheap per-step refresh: re-read the price/liability views at `time_index` and carry
        the simulator state forward. `cumulative_liability_value` absorbs the step's realized
        cashflow so the liability channel stays continuous across payment dates."""
        refreshed = {
            'time_index': int(time_index),
            'done': state['done'],
            'positions': state['positions'],
            'cash_accounts': state['cash_accounts'],
            'margin_accounts': state['margin_accounts'],
            'realized_pnl': state['realized_pnl'],
            'variation_margin': state['variation_margin'],
            'cumulative_pnl': state['cumulative_pnl'],
            'time_held': state['time_held'],
            'settlement_prices': state['settlement_prices'],
            'tradable_values': self._values_at(time_index),
            'liability_mtm_value': self.bundle.liability_at(time_index),
            'realized_cashflow_value': self.bundle.cashflow_at(time_index),
            'initial_portfolio_value': state.get('initial_portfolio_value'),
        }
        refreshed['cumulative_liability_value'] = (
            state['cumulative_liability_value'].to(dtype=torch.float32)
            + refreshed['realized_cashflow_value'])
        return refreshed

    def _step_state(self, state, action):
        if self.runtime['accounting_mode'] == 'cash_account':
            return self._cash_account_step(state, action)
        return self._futures_account_step(state, action)

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

    def _cost(self, trade_delta, price, name, vol):
        # Realized debit on |Δq| contracts at the turnover-cost rule (per_contract_kappa).
        return trade_delta.abs() * per_contract_kappa(self.runtime, price.abs(), name, vol)

    def _apply_frictions(self, accounts, deltas, prices, next_positions, current, next_idx, vol_t):
        """The two per-step friction flows both accounting modes share, credited into `accounts`
        (cash in cash mode, margin in futures mode): the matched-roll rebate, and the vol-linked
        IM funding debit — per leg, routed by currency, so each flows through `_portfolio_value`
        → `_pnl_excess` → the utility."""
        acc = self._accounting
        if acc['roll_as_calendar_spread']:
            rebate = _roll_rebate(deltas, {h: prices[h] for h in self._hedges}, self.runtime, vol_t)
            self._credit(accounts, self._account_of[self._hedges[0]], rebate)
        if acc['im_funding_spread_bps']:
            dt = self.bundle.calendar_dt(current, next_idx)
            for n, funding in _im_funding_charge(next_positions, prices, self.runtime,
                                                 vol_t, dt).items():
                self._credit(accounts, self._account_of[n], -funding)

    def _credit(self, accounts, account_name, amount):
        if account_name is not None:
            accounts[account_name] = accounts[account_name] + amount

    def _growth_factors(self, current, next_idx):
        """Per-cash-account one-day growth factor for compounding balances overnight. The cash
        tradable value is `Units / D(t) × fx_rep`, so `tv(next)/tv(cur) = D(cur)/D(next)` is the
        growth factor (≈ 1 + SOFR·dt). Empty when next_idx == current (terminal pass-through)."""
        if current >= next_idx:
            return {}
        tv_curr = self._values_at(current)
        tv_next = self._values_at(next_idx)
        return {n: tv_next[n] / tv_curr[n] for n in self._cash_names
                if n in tv_curr and n in tv_next}

    def _compound(self, accounts, factors):
        """Multiply each balance by its growth factor (pass-through by reference when absent —
        safe because balances are always rebound to new tensors, never mutated in place)."""
        return {n: (bal * factors[n] if n in factors else bal) for n, bal in accounts.items()}

    def _step_time_held(self, time_held, next_positions):
        # Increment where the position is non-zero, reset where it returned to flat.
        return {name: torch.where(next_positions[name].abs() > 0, prev + 1.0, torch.zeros_like(prev))
                for name, prev in time_held.items()}

    def _step_cumulative_pnl(self, cumulative_pnl, variation_margin, next_positions):
        """Per-trade running P&L: accumulate VM while a position is open, reset to 0 when flat —
        the same lifetime-of-current-trade semantics as `_step_time_held`."""
        return {name: torch.where(next_positions[name].abs() > 0,
                                  prev + variation_margin[name], torch.zeros_like(prev))
                for name, prev in cumulative_pnl.items()}

    def _flatten_cash(self, positions, cash_accounts, terminal_values, vol):
        for n in self._hedges:
            delta = -positions[n]
            cost = self._cost(delta, terminal_values[n], n, vol)
            cs = float(self.runtime['tradables'][n]['contract_size'])
            account = self._account_of[n]
            if account is not None:
                cash_accounts[account] = (cash_accounts[account]
                                          - (delta * terminal_values[n] * cs + cost))
            positions[n] = positions[n] + delta

    def _flatten_futures(self, positions, margin_accounts, settlement_prices, terminal_values, vol):
        """Close hedge positions at `terminal_values` and capture any residual variation margin
        (`position × (terminal − settlement) × cs`) so no P&L leaks between the last settlement and
        the close. When the caller has already advanced settlement to terminal (the normal path)
        the residual is 0 and only the trade cost is debited. `vol` scales the forced-flat turnover
        exactly like every other decision-step debit."""
        for n in self._hedges:
            cs = float(self.runtime['tradables'][n]['contract_size'])
            residual_vm = positions[n] * (terminal_values[n] - settlement_prices[n]) * cs
            delta = -positions[n]
            cost = self._cost(delta, terminal_values[n], n, vol)
            account = self._account_of[n]
            self._credit(margin_accounts, account, residual_vm)
            self._credit(margin_accounts, account, -cost)
            positions[n] = positions[n] + delta

    def _cash_account_step(self, state, action):
        # `done` is a function of time_index against the last bundle index; checking the python int
        # avoids a CUDA-CPU sync on every step
        current = int(state['time_index'])
        last = self._last_idx
        if current >= last:
            return state
        next_idx = min(current + 1, last)
        acc = self._accounting
        # shallow dict copy: tensors are replaced in-slot, never mutated in place
        next_positions = dict(state['positions'])
        next_cash = self._compound(state['cash_accounts'], self._growth_factors(current, next_idx))
        deltas = self._trade_deltas(action)
        vol_t = self.bundle.vol_at(current)
        for n in self._hedges:
            delta = deltas[n]
            price = state['tradable_values'][n]
            cost = self._cost(delta, price, n, vol_t)
            cs = float(self.runtime['tradables'][n]['contract_size'])
            account = self._account_of[n]
            if account is not None:
                # each contract is `contract_size` units of the underlying
                next_cash[account] = next_cash[account] - (delta * price * cs + cost)
            next_positions[n] = (next_positions[n] + delta).round()
        self._apply_frictions(next_cash, deltas, state['tradable_values'], next_positions,
                              current, next_idx, vol_t)
        if acc['force_flat_at_end'] and current >= last - 1:
            self._flatten_cash(next_positions, next_cash, self._values_at(last), vol_t)
        # cash mode tracks no daily VM, but cumulative_pnl still resets at flat for the same
        # per-trade-lifetime semantics
        zero_vm = self._zeros_by_name(self._hedges)
        next_state = {
            'done': torch.full_like(state['done'], next_idx >= last, dtype=torch.bool),
            'positions': next_positions,
            'cash_accounts': next_cash,
            'margin_accounts': state['margin_accounts'],
            'realized_pnl': self._zeros_by_name(self._hedges),
            'variation_margin': zero_vm,
            'cumulative_pnl': self._step_cumulative_pnl(state['cumulative_pnl'], zero_vm,
                                                        next_positions),
            'time_held': self._step_time_held(state['time_held'], next_positions),
            'cumulative_liability_value': state['cumulative_liability_value'],
            'settlement_prices': state['settlement_prices'],
            'initial_portfolio_value': state['initial_portfolio_value'],
        }
        return self._refresh(next_state, next_idx)

    def _futures_account_step(self, state, action):
        """Advance one day under futures accounting: `cash_accounts` is FROZEN at starting capital
        and only margin tracks variation margin and trade cost.

        The trade is applied BEFORE VM is computed so the new delta participates in the
        price(t) → price(t+1) accrual: the agent transacted at the decision-time price and the
        whole post-trade position is then marked at price(t+1). Positions round to integer
        contracts so float drift cannot accumulate."""
        current = int(state['time_index'])
        last = self._last_idx
        if current >= last:
            return state
        settlement_idx = min(current + 1, last)
        acc = self._accounting
        next_positions = dict(state['positions'])
        growth = self._growth_factors(current, settlement_idx)
        next_cash = self._compound(state['cash_accounts'], growth)
        next_margin = self._compound(state['margin_accounts'], growth)
        next_settlement = dict(state['settlement_prices'])
        realized_pnl = self._zeros_by_name(self._hedges)
        variation_margin = self._zeros_by_name(self._hedges)
        deltas = self._trade_deltas(action)
        next_values = self._values_at(settlement_idx)
        # trade first, then mark: the new delta accrues price(t) -> price(t+1)
        for n in self._hedges:
            cs = float(self.runtime['tradables'][n]['contract_size'])
            next_positions[n] = (next_positions[n] + deltas[n]).round()
            vm = next_positions[n] * (next_values[n] - state['settlement_prices'][n]) * cs
            realized_pnl[n] = vm
            variation_margin[n] = vm
            next_settlement[n] = next_values[n].clone()
            self._credit(next_margin, self._account_of[n], vm)
        vol_t = self.bundle.vol_at(current)
        for n in self._hedges:
            # trade cost references the DECISION-TIME price the agent acted on; next_values would
            # let unrelated overnight mid moves distort the spread cost
            self._credit(next_margin, self._account_of[n],
                         -self._cost(deltas[n], state['tradable_values'][n], n, vol_t))
        self._apply_frictions(next_margin, deltas, state['tradable_values'], next_positions,
                              current, settlement_idx, vol_t)
        if acc['force_flat_at_end'] and current >= last - 1:
            self._flatten_futures(next_positions, next_margin, next_settlement, next_values, vol_t)
        next_state = {
            'done': torch.full_like(state['done'], settlement_idx >= last, dtype=torch.bool),
            'positions': next_positions,
            'cash_accounts': next_cash,
            'settlement_prices': next_settlement,
            'margin_accounts': next_margin,
            'realized_pnl': realized_pnl,
            'variation_margin': variation_margin,
            'cumulative_pnl': self._step_cumulative_pnl(state['cumulative_pnl'], variation_margin,
                                                        next_positions),
            'time_held': self._step_time_held(state['time_held'], next_positions),
            'cumulative_liability_value': state['cumulative_liability_value'],
            'initial_portfolio_value': state['initial_portfolio_value'],
        }
        return self._refresh(next_state, settlement_idx)

    def _payoff(self, state):
        """This transition's terminal read: `pnl_excess` is the wealth change since inception and
        is only marked at the terminal step (a sync-free check on time_index — `done` is uniform
        across the batch since every scenario advances together); `liability_value` is the
        cumulative realized liability, always carried."""
        pnl_excess = (_pnl_excess(state, self.runtime).to(device=self._device, dtype=torch.float32)
                      if int(state['time_index']) >= self._last_idx
                      else torch.zeros(self._batch_size, dtype=torch.float32, device=self._device))
        return {'pnl_excess': pnl_excess,
                'liability_value': state['cumulative_liability_value'].to(
                    device=self._device, dtype=torch.float32)}

    # ---- diagnostic CSV assembly ---------------------------------------------
    def _rollout(self):
        """The recorded trajectory, stacked per instrument over decision steps."""
        return {
            'times': self._times,
            'position': {n: torch.stack(self._position_history[n], dim=0) for n in self._hedges},
            'trade': {n: torch.stack(self._trade_history[n], dim=0) for n in self._hedges},
            'price': {n: torch.stack(self._price_history[n], dim=0) for n in self._hedges},
            'net_pnl': (self._terminal_transition['pnl_excess']
                        + self._terminal_transition['liability_value']).detach().cpu(),
        }

    def _diag_fields(self, rollout=None):
        """Per-instrument (T, B) per-day cashflow tensors plus portfolio totals, reconstructed on
        CPU from the recorded trajectory. Multi-commodity bundles raise — diagnostic CSVs for
        multi-commodity are out of scope."""
        rollout = self._rollout() if rollout is None else rollout
        bundle, runtime = self.bundle, self.runtime
        spot_keys = [k for k in bundle.factors if k in runtime['referenced_commodities']]
        if len(spot_keys) != 1:
            raise ValueError(
                f"Diagnostic CSV writer expects exactly one commodity-spot factor in the bundle; "
                f"got {spot_keys}. Multi-commodity diagnostic output is not yet implemented.")
        spot = bundle.factors[spot_keys[0]].detach().cpu().float()
        mtm_running = bundle.liability_mtm.detach().cpu().float()
        T, B = mtm_running.shape

        nonzero = (mtm_running != 0)
        last_nz = (nonzero * torch.arange(T).unsqueeze(1)).max(dim=0).values
        fill_mask = torch.arange(T).unsqueeze(1).expand(T, B) > last_nz.unsqueeze(0)
        realised = mtm_running.gather(0, last_nz.unsqueeze(0)).expand(T, B)
        mtm = torch.where(fill_mask, realised, mtm_running)

        # per-step vol `(T, 1)` for the vol-scaled kappa, index-aligned with the full-grid marks;
        # None when no vol series was built, and inert on the scalar-spread fast path
        vol_series = bundle.step_annual_vol
        diag_vol = None if vol_series is None else vol_series.detach().cpu().float().unsqueeze(-1)
        times = [int(t) for t in rollout['times']]
        per_instr = {}
        for name in self._hedges:
            cs = float(runtime['tradables'][name]['contract_size'])
            fut = bundle.tradables[name].detach().cpu().float()
            pos = torch.zeros((T, B)); trd = torch.zeros((T, B))
            cur = torch.zeros(B); j = 0
            rollout_pos = rollout['position'][name]
            rollout_trd = rollout['trade'][name]
            for t in range(T):
                if j < len(times) and t == times[j]:
                    trd[t] = rollout_trd[j].cpu().float()
                    cur = rollout_pos[j].cpu().float() + rollout_trd[j].cpu().float()
                    j += 1
                pos[t] = cur
            trade_cash = -trd * fut * cs
            trade_cost = trd.abs() * per_contract_kappa(runtime, fut, name, diag_vol)
            per_instr[name] = {
                'fut': fut, 'cs': cs, 'pos': pos, 'trd': trd,
                'trade_cash': trade_cash, 'trade_cost': trade_cost,
                'position_mtm': pos * fut * cs,
                'cum_cash': trade_cash.cumsum(0), 'cum_cost': trade_cost.cumsum(0),
            }
        portfolio_pos_mtm = sum(p['position_mtm'] for p in per_instr.values())
        portfolio_cum_cash = sum(p['cum_cash'] for p in per_instr.values())
        portfolio_cum_cost = sum(p['cum_cost'] for p in per_instr.values())
        # threaded with the SAME per-step vol as the cost above, so the matched-roll credit
        # reconstructs the realized vol-scaled rebate
        if runtime['accounting']['roll_as_calendar_spread']:
            rebate = torch.stack([
                _roll_rebate({h: per_instr[h]['trd'][t] for h in self._hedges},
                             {h: per_instr[h]['fut'][t] for h in self._hedges}, runtime,
                             None if diag_vol is None else diag_vol[t])
                for t in range(T)])
            portfolio_cum_cost = portfolio_cum_cost - rebate.cumsum(0)
        hp = portfolio_cum_cash + portfolio_pos_mtm - portfolio_cum_cost
        return {
            'spot': spot, 'spread_bps': float(runtime['accounting']['bid_offer_spread_bps']),
            'per_instr': per_instr, 'mtm': mtm,
            'portfolio_position_mtm': portfolio_pos_mtm,
            'portfolio_cum_trade_cash': portfolio_cum_cash,
            'portfolio_cum_trade_cost': portfolio_cum_cost,
            'hedge_portfolio_ex_funding': hp,
            'total_ex_funding_discount': mtm + hp,
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
            row['mtm_ex_post_settle_discount'] = float(getter(fields['mtm'][t]).item())
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
    def __init__(self, *, bundle=None, runtime=None, evaluation_summary=None,
                 optimizer_diagnostics=None, policy_artifact=None, metadata=None):
        self.bundle = bundle
        self.runtime = runtime
        self.evaluation_summary = evaluation_summary
        self.optimizer_diagnostics = optimizer_diagnostics
        self.policy_artifact = policy_artifact
        self.metadata = metadata or {}

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
    return {'policy_artifact': None, 'optimizer_diagnostics': None,
            'evaluation_output': stepper.evaluation_output(
                timing={'evaluation_time_seconds': float(time.perf_counter() - started)})}
