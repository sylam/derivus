"""The hedging stack's world side: the JSON boundary (`construct_hedge_runtime`), the simulated
bundle (`Bundle`), the environment (`BundleStepper`) and the pure math the solver and the
environment share (the wealth law, the utility shapes, the turnover charge, the portfolio reads),
documented on the Hedging section's Overview and Bundle pages."""

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


# rows in the trailing window of the realised vol the utility scale and the step vol read
PRICE_ZSCORE_WINDOW = 20


# the shapes that read a scale c; LogWealth reads none, and a solve admits it beside these by name
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


def _position_schedule(evaluator_config: Mapping[str, Any]):
    """Optional per-decision-step corridor on the SIGNED total position Σq_i. A list of
    `{Step, Min_Total, Max_Total}` knots (piecewise-constant between knots): at sim-grid
    decision step t the signed book total must lie within [Min_Total, Max_Total] of the
    rightmost knot with `Step <= t`. Absent → None (no corridor). Returns a sorted tuple of
    `(step, min_total, max_total)` with strictly ascending, non-negative steps and
    Min_Total <= Max_Total per knot."""
    raw = evaluator_config["Total_Position_Schedule"]
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
    """The declared ALLOCATION SCHEDULE of `Allocation_Mode` Exposure, which FACTORS the action
    space: the solver chooses one signed NET cover on a fine ladder and this splits it across the
    hedge legs, instead of searching the Cartesian product of per-leg levels. Absent → None.

    Declared long — a list of `{Step, Instrument, Weight}` rows, the rows sharing a `Step` being
    one KNOT, piecewise-constant in the decision step like `Total_Position_Schedule`.

    Returns a sorted tuple of `(step, (w_0, ..., w_{n-1}))` in `names['hedges']` order, each knot
    NORMALIZED to sum 1. A leg no row names gets 0, which is how an expired leg leaves the ladder.
    Weights must be non-negative and already sum to ~1, so normalizing removes float dust rather
    than reinterpreting intent, and the ladder's `Σ_i round(Q·w_i) == Q` apportionment is exact."""
    raw = evaluator_config["Allocation_Weights"]
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
        dates_raw, prices_raw = payload["Dates"], payload["Prices"]
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
    """Normalize the `Solver` block, None outside a solve: each declared key under its lowercased
    name, a Yes/No as a bool; what each means is its `F` description."""
    if solver_config is None:
        return None
    if "Object" not in solver_config:
        raise ValueError("Hedging_Problem['Solver'] requires an 'Object' field")
    gamma = float(solver_config["DiffV2_Risk_Aversion"])
    if not 0.1 <= gamma <= 10.0:
        raise ValueError(f"DiffV2_Risk_Aversion must be in [0.1, 10.0]; got {gamma}")
    return {
        "object": str(solver_config["Object"]).lower(),
        "multi_seed_count": int(solver_config["Multi_Seed_Count"]),
        "t_min": int(solver_config["T_Min"]),
        "training_action_grid_levels_per_axis":
            int(solver_config["Training_Action_Grid_Levels_Per_Axis"]),
        "training_action_chunk_size": int(solver_config["Training_Action_Chunk_Size"]),
        "diffv2_fit_iters": int(solver_config["DiffV2_Fit_Iters"]),
        "diffv2_lr": float(solver_config["DiffV2_LR"]),
        "diffv2_bank_noise_frac": float(solver_config["DiffV2_Bank_Noise_Frac"]),
        "diffv2_risk_aversion": gamma,
        "diffv2_drift_threshold_sigmas":
            float(solver_config["DiffV2_Drift_Threshold_Sigmas"]),
        "diffv2_drift_beta": float(solver_config["DiffV2_Drift_Beta"]),
        "diffv2_passive_from": str(solver_config["DiffV2_Passive_From"] or ""),
        "diffv2_first_decision_from":
            str(solver_config["DiffV2_First_Decision_From"]).lower(),
        "diffv2_weight_decay": float(solver_config["DiffV2_Weight_Decay"]),
        "diffv2_hidden": int(solver_config["DiffV2_Hidden"]),
        "diffv2_lambda_grad": float(solver_config["DiffV2_Lambda_Grad"]),
        "diffv2_risk_kappa": float(solver_config["DiffV2_Risk_Kappa"]),
        "diffv2_risk_horizon": int(solver_config["DiffV2_Risk_Horizon"]),
        "diffv2_churn_lambda": float(solver_config["DiffV2_Churn_Lambda"]),
        "diffv2_temporal_proximity":
            float(solver_config["DiffV2_Temporal_Proximity"]),
        "diffv2_fit_tol": float(solver_config["DiffV2_Fit_Tol"]),
        "diffv2_repositioning_charge": str(solver_config["DiffV2_Repositioning_Charge"]).lower(),
        "diffv2_book_gradient_label":
            solver_config["DiffV2_Book_Gradient_Label"] == "Yes",
        "diffv2_bank_along_ladder":
            solver_config["DiffV2_Bank_Along_Ladder"] == "Yes",
        "diffv2_label_horizon": int(solver_config["DiffV2_Label_Horizon"]),
        "diffv2_load_horizon_pad":
            solver_config["DiffV2_Load_Horizon_Pad"] == "Yes",
        "diffv2_returns_state":
            solver_config["DiffV2_Returns_State"] == "Yes",
        "diffv2_wealth_free_value":
            solver_config["DiffV2_Wealth_Free_Value"] == "Yes",
        "diffv2_stepper_rollout":
            solver_config["DiffV2_Stepper_Rollout"] == "Yes",
        "diffv2_decision_curve_dump":
            str(solver_config["DiffV2_Decision_Curve_Dump"] or ""),
        "diffv2_per_column_grad_norm":
            solver_config["DiffV2_Per_Column_Grad_Norm"] == "Yes",
        "diffv2_save_value_fn": str(solver_config["DiffV2_Save_Value_Fn"] or ""),
        "diffv2_load_value_fn":
            ([str(p) for p in solver_config["DiffV2_Load_Value_Fn"]]
             if isinstance(solver_config["DiffV2_Load_Value_Fn"], (list, tuple))
             else str(solver_config["DiffV2_Load_Value_Fn"] or "")),
        "diffv2_max_value_optimism": float(solver_config["DiffV2_Max_Value_Optimism"]),
        "diffv2_max_member_disagreement":
            float(solver_config["DiffV2_Max_Member_Disagreement"]),
        "active_hedge_indices": (None if solver_config["Active_Hedge_Indices"] is None
                                 else list(solver_config["Active_Hedge_Indices"])),
        "diffv2_book_floor_per_unit": float(solver_config["DiffV2_Book_Floor_Per_Unit"]),
        "run_hindsight_diagnostic": solver_config["Run_Hindsight_Diagnostic"] == "Yes",
        "run_textbook_benchmark": solver_config["Run_Textbook_Benchmark"] == "Yes",
    }


def _objective_config(objective_config):
    """Normalize the `Objective` block, None without one. Its dials are dispatched on downstream by
    EQUALITY (`== 'running_wealth'`, `== 'conditional_sim'`), so a near-miss spelling would quietly
    mean the default: `Reference_Mode` is validated here, `Utility_Scale_Mode` refuses its own."""
    if objective_config is None:
        return None
    logwealth = str(objective_config["Object"]).lower() == "logwealth"
    stated = objective_config["Reference_Mode"]
    reference_mode = str(stated or ("Running_Wealth" if logwealth else "Fixed")).lower()
    if reference_mode not in ("fixed", "running_wealth"):
        raise ValueError(
            f"Unsupported Objective.Reference_Mode: "
            f"{objective_config['Reference_Mode']!r}. Supported: 'Fixed' (the utility "
            f"is applied to TERMINAL wealth) | 'Running_Wealth' (to the day's wealth increment). "
            f"The value is dispatched on by equality, so a near-miss would silently run 'Fixed'.")
    if logwealth and reference_mode == "fixed":
        raise ValueError(
            "Objective.Reference_Mode 'Fixed' beside Object 'LogWealth', whose per-step growth is "
            "the day's increment: a contradiction the run refuses rather than arbitrates.")
    logging.debug("Objective.Reference_Mode %s%s", reference_mode,
                  " (the Object's own)" if stated is None else "")
    # every outer path shares L_0, so the raw dispersion at t=0 is exactly 0: a non-positive floor
    # makes c_0 = 0, x = (W-R)/c_0 infinite, and every label from step 0 a NaN
    floor_frac = float(objective_config["Utility_Scale_Floor_Frac"])
    if not floor_frac > 0.0:
        raise ValueError(f"Objective.Utility_Scale_Floor_Frac must be > 0; got {floor_frac}")
    if floor_frac > 1.0:
        logging.warning("Objective.Utility_Scale_Floor_Frac=%.4g is above 1: every knee of the "
                        "conditional_sim schedule is its floor", floor_frac)
    return {
        # canonical lowercased form: every dispatch site compares against the lowercase literal
        "object": str(objective_config["Object"]).lower(),
        # `utility_scale` and its schedule are mirrored in from the frame (`mirror_frame`)
        "utility_scale_mode": str(objective_config["Utility_Scale_Mode"]).lower(),
        "utility_scale_explicit":
            (None if objective_config["Utility_Scale_Explicit"] is None
             else float(objective_config["Utility_Scale_Explicit"])),
        "utility_scale_floor_frac": floor_frac,
        "reference_mode": reference_mode,
        "reference_wealth": float(objective_config["Reference_Wealth"]),
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
    """The JSON → runtime boundary: read `Hedging_Problem`, refuse what a run cannot read, and
    return the runtime dict every consumer indexes directly (Hedging Overview, One boundary)."""
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
                f"a held-out batch no fit step saw); got {n_batches}, which derivus_batch "
                "divides by the job count before this check.")
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
                "V-hat unbounded in money and the backward sweep blows up multiplicatively.")

    history_lookback = int(hedging_problem["History_Lookback_Business_Days"])
    if history_lookback < 0:
        raise ValueError("Hedging_Problem.History_Lookback_Business_Days must be non-negative")
    # Commodity names come from the live CommodityPrice factors the instruments created at
    # calc-dependency time — never re-parsed out of instrument JSON params.
    referenced_commodities = tuple(dict.fromkeys(
        utils.check_tuple_name(factor) for factor in (stoch_factors or {})
        if factor.type == 'CommodityPrice'))
    portfolio_state = hedging_problem["Portfolio_State"]
    vol_scale = evaluator_config.get("Bid_Offer_Spread_Vol_Scale")
    calendar_spread_bps = (None if evaluator_config["Calendar_Spread_Bps"] is None
                           else float(evaluator_config["Calendar_Spread_Bps"]))
    # a rate <= 0 makes matched volume free or PAID FOR, and `turnover_charge` then returns a
    # negative cost the argmax maximizes by churning; absence is the off switch
    if calendar_spread_bps is not None and calendar_spread_bps <= 0.0:
        raise ValueError(
            f"Evaluator.Calendar_Spread_Bps must be > 0; got {calendar_spread_bps}. Omit the "
            f"key to price every contract at the outright spread.")
    # static instrument -> cash_account routing by currency: the first cash account whose currency
    # matches wins, else the first account; none declared is one unmarked ledger, compounding nothing
    ledgers = cash_account_names or ("money",)
    logging.debug("ledgers: %s", ledgers)
    account_by_currency = {}
    for account_name in cash_account_names:
        account_by_currency.setdefault(
            normalized_tradables[account_name]["currency"], account_name)

    alloc_mode = str(evaluator_config["Allocation_Mode"]).lower()
    if alloc_mode not in ("cartesian", "exposure", "carry_variance"):
        raise ValueError(
            f"Unsupported Evaluator.Allocation_Mode: "
            f"{evaluator_config['Allocation_Mode']!r}. Supported: 'Cartesian' (the product of "
            f"per-leg levels) | 'Exposure' (the declared Allocation_Weights table) | "
            f"'Carry_Variance' (solver-derived). Dispatched by equality, so a near-miss would "
            f"silently run another universe.")
    if bool(evaluator_config["Allocation_Weights"]) != (alloc_mode == "exposure"):
        raise ValueError(
            f"Evaluator.Allocation_Mode={evaluator_config['Allocation_Mode']!r} "
            f"{'with' if evaluator_config['Allocation_Weights'] else 'without'} an "
            f"Allocation_Weights table: Exposure splits the net cover by the declared table and "
            f"requires one, and no other mode reads it - a contradiction the run refuses rather "
            f"than arbitrates.")
    if alloc_mode == "carry_variance" and solver_config is None:
        raise ValueError(
            "Evaluator.Allocation_Mode='Carry_Variance' needs a Solver to derive the "
            "weights (there are no sims to measure carry/tracking from in a non-solve "
            "mode). Declare Allocation_Weights under 'Exposure' instead.")
    logging.debug("Evaluator.Allocation_Mode %s: %s", alloc_mode, "the declared table"
                  if alloc_mode == "exposure" else "no table read")
    return {
        "accounting_mode": str(evaluator_config["Accounting_Mode"]).lower(),
        "names": {
            "tradables": tuple(normalized_tradables),
            # the hedge legs ARE the action set; the solver builds its action grid over them
            "hedges": hedge_names,
            "cash_accounts": ledgers,
        },
        "referenced_commodities": referenced_commodities,
        "tradables": normalized_tradables,
        "objective": _objective_config(objective_config),
        "solver": _solver_config(solver_config),
        "history_lookback_business_days": history_lookback,
        # the burn-in the calculation draws each batch's opening state by, and the inner fork's
        # belief switch a process reads (`reseed_inner_state`)
        "randomize_initial_state": hedging_problem["Randomize_Initial_State"] == "Yes",
        "inner_belief_filter": hedging_problem["Inner_Belief_Filter"] == "Yes",
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
                n: account_by_currency.get(meta["currency"], ledgers[0])
                for n, meta in normalized_tradables.items()},
            "transaction_cost_per_unit":
                float(evaluator_config["Transaction_Cost_Per_Unit"]),
            "bid_offer_spread_bps": float(evaluator_config["Bid_Offer_Spread_Bps"]),
            "bid_offer_spread_per_instrument": {
                str(n): float(v) for n, v in evaluator_config["Bid_Offer_Spread_Per_Instrument"].items()},
            "bid_offer_spread_vol_scale": (None if vol_scale is None else {
                "ref_vol": float(vol_scale["Ref_Vol"]), "beta": float(vol_scale["Beta"])}),
            "calendar_spread_bps": calendar_spread_bps,
            "im_funding_spread_bps": float(evaluator_config["IM_Funding_Spread_Bps"]),
            "im_vol_multiplier": float(evaluator_config["IM_Vol_Multiplier"]),
            "im_ref_vol": float(evaluator_config["IM_Ref_Vol"]),
            "force_flat_at_end": evaluator_config["Force_Flat_At_End"] == "Yes",
            "total_position_abs_limit":
                float(evaluator_config["Total_Position_Abs_Limit"]),
            "total_position_schedule": _position_schedule(evaluator_config),
            "allocation_weights": _allocation_weights(evaluator_config, hedge_names),
            "allocation_mode": alloc_mode,
            "max_trade_per_step": float(evaluator_config["Max_Trade_Per_Step"]),
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

    `spread_bps` is the leg's `Bid_Offer_Spread_Per_Instrument` entry, else `Bid_Offer_Spread_Bps`,
    scaled by `(vol/Ref_Vol)**Beta` under a `Bid_Offer_Spread_Vol_Scale` and an annualized `vol`;
    `vol=None` or no vol scale ⇒ vol-independent.

    `calendar=True` prices one contract of a CALENDAR SPREAD quoted against this leg instead: the
    spread is `Evaluator.Calendar_Spread_Bps` on this leg's notional under the same vol scale, and
    the flat fee is charged TWICE because a spread contract moves and clears two futures. Paired as
    `0.5·(κcal_i + κcal_j)` by `turnover_charge`, the result is both clearing fees plus one
    half-spread on the average leg notional.

    The one scalar `Calendar_Spread_Bps` rate is on Model Limitations; the OUTRIGHT half of every
    charge reads each leg's own base."""
    acc = runtime["accounting"]
    contract_size = float(runtime["tradables"][name]["contract_size"])
    spread_bps = acc["bid_offer_spread_per_instrument"].get(str(name), acc["bid_offer_spread_bps"])
    vscale = acc["bid_offer_spread_vol_scale"]
    scale = 1.0 if vscale is None or vol is None else (vol / vscale["ref_vol"]) ** vscale["beta"]
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
    function, and consecutive pairs alone, the maturities a calendar spread is listed between."""
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
    """The opening book `q0` `(batch, n_hedge)`: `Portfolio_State.Positions` in whole contracts, in
    hedge-leg order, zero on a leg it does not name. Every track holds it from the first row and is
    charged its first move from it; the first decision's trade band and position state measure from
    it too, or from the passive book under `DiffV2_First_Decision_From` 'Passive'."""
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

    `BundleStepper` books the same `Σ q·cs·dF` as each leg's variation margin with the realised
    accounting this form leaves out."""
    return W + (q * contract_size * dF).sum(dim=-1) + dL


def _is_utility_objective(runtime):
    """True iff the objective transforms wealth through a utility shape (symlog/huber/cara), so
    the DP and value function live in utility space and a scale `c` is required."""
    return (runtime.get("objective") or {}).get("object") in _UTILITY_OBJECTS


def _utility_scale(objective, t=None):
    """The money scale `c` in force at decision step `t` — ONE resolution rule, shared by the
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


def _utility_wrap_signed(w, runtime, t=None):
    """The utility u of the objective's shape, the forms the Hedging Overview tabulates, at
    x = (W − R)/c (`_utility_x`, the scale of step `t` under a schedule) of signed wealth `w`:
    the one transform the objective, the recursion and the labels read; identity without a utility
    objective, refused under LogWealth. Differentiable in `w`; what u is applied to is the
    caller's, the day's increment under Running_Wealth."""
    if ((runtime.get("objective") or {}).get("object")) == "logwealth":
        # LogWealth has no terminal transform - its reward is the per-step ratio log(W1/W0), owned
        # by the solver's `_u_step`. Checked BEFORE `_is_utility_objective`, whose meaning is
        # "consumes a scale c": LogWealth does not, so it must not take the identity path either.
        raise ValueError(
            "Objective 'LogWealth' defines no terminal utility — the reward is the per-step "
            "growth ratio (DiffSolver._u_step). This caller must be step-sum-aware or disabled "
            "under LogWealth.")
    if not _is_utility_objective(runtime):
        return w
    obj = runtime["objective"]
    x = _utility_x(w, obj, t)
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
    """`(u'(W), −u''(W)/u'(W))` in money at one scalar wealth — the marginal utility and the
    local absolute risk aversion the configured shape is actually applying there. `t` names the
    decision step, so a locked per-step scale schedule reports the knee in force on THAT day.

    The closed forms in `_utility_wrap_signed`'s normalised x = (W − R)/c, chain-ruled back to
    money (u' = f'(x)/c and ARA = −f''(x)/(c·f'(x)), so the scale enters exactly once each):

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

    IM is posted on each leg's gross |q| (Model Limitations). σ_t is the SAME per-step vol that drives the
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

    Cash and margin balances are raw money compounded daily (each step multiplies by
    `cash_tv(next)/cash_tv(curr)` ≈ 1 + r·dt at the cash curve's rate, then adds today's flows), so each unit earns
    interest only from the day it landed.

    Two accounting modes. `cash_account`: position × price × contract_size plus the cash balances.
    Futures: cash is FROZEN at starting capital and margin carries every VM and trade-cost flow
    since inception, every row settled at its own mark."""
    tradables = runtime["tradables"]
    positions, prices = state["positions"], state["tradable_values"]
    total = torch.zeros_like(state["done"], dtype=MONEY)
    cash = sum(balance.to(dtype=MONEY) for balance in state["cash_accounts"].values())
    if runtime["accounting_mode"] == "cash_account":
        for name in runtime["names"]["hedges"]:
            total = total + (positions[name].to(dtype=MONEY) * prices[name].to(dtype=MONEY)
                             * float(tradables[name]["contract_size"]))
        return total + cash
    return total + sum(balance.to(dtype=MONEY) for balance in state["margin_accounts"].values()) + cash


def _pnl_excess(state, runtime):
    """Wealth change since inception: `portfolio_value - initial_portfolio_value`, which is what
    makes the utility's floor at zero meaningful. The initial baseline is snapshotted when the
    stepper builds its opening state and threaded through every transition."""
    pv = _portfolio_value(state, runtime)
    return pv - state["initial_portfolio_value"].to(device=pv.device, dtype=pv.dtype)


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
    """The simulated hedging world: every time-indexed tensor the solver and the environment read,
    built once per batch by `from_batch` and read-only after; documented on the Bundle page."""

    documentation = ('Bundle', [
        'Every time-indexed tensor the solver and the environment read, built once per batch by',
        '`from_batch` in four stages (`_resolve_tradables`, `_resolve_liability`, `_resolve_factors`,',
        '`_resolve_frame`) and read-only after; consumers index it by attribute.',
        '',
        '- **Tradables** — `tradables` / `tradables_sim`: the per-instrument mark series, harvested',
        '  from the priced deal tree via `DealStructure.tensor_marks`. A leg is live on the rows its deal\'s dates mark',
        '  (`marked`) and moves over a step it is still marked at the end of, the step carrying',
        '  market time (`moving`): the one liveness rule every track reads.',
        '- **Liability** — `liability_value` / `liability_sim`: its mark plus the cash it has paid',
        '  on earlier rows, and `unfixed_at`: the signed quantity its fixings still carry after a row.',
        '- **Factors** — the simulated factor paths, and `privileged_factors`: whatever each process',
        '  publishes as market state.',
        '- **Frame** — `utility_scale`, the objective\'s scale `c` as this batch resolves it (the',
        '  solver locks warmup\'s), with its per-step `utility_scale_schedule` under `conditional_sim`',
        '  (Hedging Overview), and `step_annual_vol`, the per-step vol a vol-scaled spread and the',
        '  margin funding read: a process\'s revealed conditional vol, else the primary spot\'s',
        '  trailing realised vol with its `Spot_Price_History` in front.',
        '',
        'Under `solve_hedge` `HedgeMonteCarlo.execute` attaches `inner_mc` / `inner_mc_grad`, which',
        'fork the simulator at a decision row and price it to its landing row, or to each row of a',
        '`DiffV2_Risk_Horizon` list ([Calc Lifecycle](../developer/calc_lifecycle.md), the inner-MC',
        'fork).',
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
            logging.info('utility_scale c resolved to %.2f', self.utility_scale)
            mirror_frame(runtime, self.utility_scale, self.utility_scale_schedule)
            if self.utility_scale_schedule is not None:
                self._log_scale_dose(runtime)
        self.step_annual_vol = self._resolve_step_vol(runtime, stoch_factors)

    def _spot_realized_vol(self, window=PRICE_ZSCORE_WINDOW, min_periods=5):
        """Annualized rolling realized log-vol of each spot with a realized history,
        `{commodity: (T, B)}`, zero on its first `min_periods` rows: the utility scale's σ."""
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
        """The objective's scale `c` as this batch resolves it, by the rules of the Hedging
        Overview's scale table: the formula, `Utility_Scale_Explicit`, or the terminal entry of the
        `conditional_sim` schedule (`_conditional_sim_schedule`); a degenerate input raises.

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
                "Utility_Scale_Explicit to override the formula with a literal value in money.")
        explicit = objective['utility_scale_explicit']
        if mode == 'conditional_sim':
            if explicit is not None:
                raise ValueError(
                    "Objective.Utility_Scale_Explicit is set beside "
                    "Utility_Scale_Mode='conditional_sim', which measures c: drop one of them.")
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
        """Log the `conditional_sim` schedule and the absolute risk aversion per unit of money the
        shape applies a hair below the reference under its first knee and its terminal one, the
        dose the schedule's level sets (Hedging Overview, the scale table)."""
        sched = self.utility_scale_schedule
        _, ara_0 = _loss_side_curvature(runtime, 0)
        _, ara_T = _loss_side_curvature(runtime, len(sched) - 1)
        logging.info(
            'utility_scale schedule (conditional_sim): %d steps, c_0=%.6g c_mid=%.6g c_T=%.6g | '
            'absolute risk aversion at the reference %.4g per unit of money at t=0, %.4g at the '
            'terminal', len(sched), sched[0], sched[len(sched) // 2], sched[-1], ara_0, ara_T)

    def _conditional_sim_schedule(self, objective):
        """The `conditional_sim` schedule (Hedging Overview, the scale table): one entry per row to
        the close, `last_live_mtm_index`, the cross-path std of the flat-book wealth's move
        `L_t - L_0` floored at `Utility_Scale_Floor_Frac` of its terminal entry; None for a batch of
        one outer path, which has no cross-section."""
        if self.liability_value is None:
            raise ValueError(
                "Objective.Utility_Scale_Mode='conditional_sim' needs a marked liability: the "
                "schedule is the dispersion of the flat-book wealth path and this bundle has no "
                "liability_value to measure it on.")
        L = self.liability_sim[:self.last_live_mtm_index + 1].to(dtype=MONEY)
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
        vol_scale_active = acc['bid_offer_spread_vol_scale'] is not None
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
    """The hedging environment: the bundle advanced one row at a time under any policy, with the
    realised accounting the frictionless wealth law leaves out; documented on the Bundle page."""

    documentation = ('Bundle', [
        'The hedging environment: the bundle advanced one row at a time under any policy, with the',
        'realised accounting the frictionless wealth law (`wealth_step`) leaves out: overnight',
        "financing on cash and margin, each leg's variation margin and settlement, each leg's share of",
        'the turnover charge (`leg_charges`) and the vol-linked margin funding, whole contracts, and',
        'the forced flatten at the terminal, the last live liability mark, where the verdict closes.',
        'It trades only on the rows the solver decides on (`Bundle.decision_steps`), and reads its',
        'accounting off `Evaluator`; the Solver page says where its books and wealth meet the',
        "verdict's.",
        '',
        'Each `step` returns a state dict and the caller chooses the next action, vectorised over the',
        'batch, an action a scalar or one value per path. `copy.deepcopy` forks a counterfactual',
        'branch, and the recorded trajectory feeds `write_diagnostic_csvs`.',
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
        """Bind the environment to `bundle` and `runtime` and open the state at the first row,
        mirroring the bundle's scale onto the runtime unless `mirror_scale` is False (the Bundle
        page says whose `c` governs a roll)."""
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
            # the step's booked charges: its trade's, then on the step into the terminal the close's
            closed = self._charged[self._hedges[0]][-1][0] == self._close_idx != cur
            cost = {n: float(self._charged[n][-2 if closed else -1][1][0]) for n in self._hedges}
            legs = {n: (float(self._state['tradable_values'][n][0]),
                        int(next_state['positions'][n][0]), int(deltas[n][0]),
                        cost[n], self._account_of[n]) for n in self._hedges}
            logging.debug(
                'STEP %s t=%d dec=%d legs(mark,q,dq,cost,account)=%s vm=%.6f cost=%.6f close=%.6f '
                'margin=%.6f cash_tv=%s pnl_excess=%.6f liability=%.6f',
                self.bundle.scenario_dates[cur].date(), cur, int(was_decision_step), legs,
                sum(float(v[0]) for v in next_state['variation_margin'].values()), sum(cost.values()),
                sum(float(self._charged[n][-1][1][0]) for n in self._hedges) if closed else 0.0,
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
        publishes, of the policy this stepper was rolled under."""
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
            'diagnostics': {'num_batches': self.bundle.num_batches},
            'timing': dict(timing or {}),
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
            'initial_portfolio_value': None,
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
            'initial_portfolio_value': state['initial_portfolio_value'],
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
        """Client action → the trade deltas the env books, rounded to whole contracts and booked
        as given: the bounds are the policy's (`HedgeActionSpace`), not the environment's."""
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
        accounts[account_name] = accounts[account_name] + amount

    def _compound(self, accounts, marks, next_marks):
        """Each balance grown overnight by its cash account's mark ratio `tv(next)/tv(cur)`: the
        cash tradable value is `Units / D(t) × fx_rep`, so the ratio is `D(cur)/D(next)`, about
        1 + r·dt. A balance with no mark passes through by reference, safe because balances are
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

    def _diag_fields(self, rollout):
        """Per-instrument (T, B) per-day cashflow tensors plus portfolio totals, reconstructed on
        CPU from the recorded trajectory `rollout` (`_rollout`); the spot column is the primary
        spot's."""
        bundle, runtime = self.bundle, self.runtime
        spot = bundle.factors[bundle.primary].detach().cpu().to(MONEY)
        T, B = spot.shape
        per_instr = {}
        # each mark held at the terminal past it, as `_values_at`, gathered by index: a copy of the
        # world's tensor, never a view of it
        held_at = torch.arange(T).clamp(max=self._close_idx)
        # the liability's value, the stepper's, held at the terminal past it as the marks are
        liability = bundle.liability_value.detach().cpu().to(MONEY)[held_at]
        logging.debug('CSV liability at the terminal row %d: %s', self._close_idx,
                      liability[self._close_idx].tolist())
        for name in self._hedges:
            cs = float(runtime['tradables'][name]['contract_size'])
            fut = bundle.tradables[name].detach().cpu().to(MONEY)[held_at]
            # a row trades the change of the book it holds, at its own mark, so the forced close
            # sits on the terminal, where it is booked
            held = rollout['held'][name].to(MONEY)
            rows = slice(bundle.initial_time_index, bundle.initial_time_index + held.shape[0] - 1)
            pos, trd = torch.zeros((T, B), dtype=MONEY), torch.zeros((T, B), dtype=MONEY)
            pos[rows], trd[rows] = held[1:], held.diff(dim=0)
            pos[rows.stop:] = held[-1]
            trade_cash = -trd * fut * cs
            # the opening book enters at the first row's mark, where the stepper's P&L starts
            trade_cash[rows.start] -= held[0] * fut[rows.start] * cs
            # the leg's turnover charges as the stepper booked them, each on its row
            cost = torch.zeros((T, B), dtype=MONEY)
            for row, charge in self._charged[name]:
                cost[row] += charge.detach().cpu().to(MONEY)
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
            'spot': spot, 'per_instr': per_instr, 'liability': liability,
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
                   'spot': getter(fields['spot'][t]).item()}
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
