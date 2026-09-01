"""The ONE JSON → hedging-runtime boundary.

`construct_hedge_runtime` reads the `Hedging_Problem` JSON block and returns the normalized
runtime dict every hedging consumer indexes by key: canonical lowercased modes, the instrument /
cash-account / hedge name sets, per-instrument metadata, the accounting rules (position limits,
turnover cost, spreads, margin funding, corridor), the objective, the solver config and the
portfolio state. Everything is validated HERE and nowhere else — past this boundary the runtime is
the contract, so downstream code indexes it directly rather than re-checking it.

Also owns the privileged-factor naming convention (what each stochastic process publishes as
market state) and `per_contract_kappa`, the single turnover-cost rule the solver, the environment
and the diagnostic CSV writer all price frictions through.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Dict, Mapping, Optional

import torch

from . import utils


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
        "params": deepcopy(dict(params)),
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
        "active_hedge_indices":
            (list(solver_config["Active_Hedge_Indices"])
             if solver_config.get("Active_Hedge_Indices") is not None else None),
        # Benchmark tracks assembled alongside the DiffSolver deliverable (hindsight upper
        # bound / textbook lower bound).
        "run_hindsight_diagnostic": solver_config.get("Run_Hindsight_Diagnostic", "No") == "Yes",
        "run_textbook_benchmark": solver_config.get("Run_Textbook_Benchmark", "No") == "Yes",
    }


# Utility Objectives — the DP / value function lives in utility space and needs the scale c.
_UTILITY_OBJECTS = ("asymmetricutility_symlog", "asymmetricutility_huber",
                    "asymmetricutility_cara", "logwealth")


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
    if evaluator_config.get("Cash_Instruments") is not None:
        cash_account_names = tuple(str(n) for n in evaluator_config["Cash_Instruments"])
    elif evaluator_config.get("Cash_Accounts") is not None:
        cash_account_names = tuple(str(n) for n in evaluator_config["Cash_Accounts"])
    elif evaluator_config.get("Cash_Instrument") is not None:
        cash_account_names = (str(evaluator_config["Cash_Instrument"]),)
    else:
        cash_account_names = ()
    for account_name in cash_account_names:
        if account_name not in tradables:
            raise ValueError(
                f"Evaluator cash account '{account_name}' is not in Tradable_Instruments")
    hedge_names = tuple(n for n in tradables if n not in cash_account_names)
    if not hedge_names:
        raise ValueError("no hedge instruments: Tradable_Instruments has only cash accounts")

    # --- liabilities: the book being hedged; its latest expiry dates the hedge instruments ---
    liabilities = {}
    for name, entry in _flatten_deals(hedging_problem.get("Liabilities") or {}).items():
        params = entry["params"]
        liabilities[name] = {
            "reference": name, "object": entry["deal_type"], "deal_type": entry["deal_type"],
            "underlying": params.get("Underlying"), "currency": params.get("Currency"),
            "strike": float(params.get("Strike", params.get("Strike_Price", 0.0))),
            "quantity": float(params.get("Quantity", params.get("Units", 0.0))),
            "expiry_date": params.get("Expiry_Date"), "params": deepcopy(dict(params))}
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
        if str((objective_config or {}).get("Object", "")).lower() not in _UTILITY_OBJECTS:
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
            "hedges": hedge_names,
            "cash_accounts": cash_account_names,
            # The hedge legs ARE the action set; the solver builds its action grid over them.
            "action_instruments": hedge_names,
            "liabilities": tuple(liabilities),
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
        "policy": None,
        "optimizer": None,
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
