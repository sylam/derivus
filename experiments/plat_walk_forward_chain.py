"""Walk-forward backtest of the platinum hedge in the SESSION-CHAIN world - the AM-fix-anchored
sibling of `plat_walk_forward_sync.py`, which is itself `production_walk_forward.py`'s.

THE WORLD INVERTS, and that is the whole of what changes. The sync deck's primary is the
CME-implied spot with the LBMA fix carried as a basis off it; here the LBMA **AM fix is the
primary** and the CME curve hangs off it, so every routing is the other way round:

  * `CommodityPrice.LBMA_AM` (GARCHSpotModel) is the fixing itself - the liability's own
    reference, and NOT tradable.
  * `ObservedBasis.LBMA_AM.CME` (BasisLinkedSpotModel) is the CME session basis, and the futures
    reference the COMPOSED name `LBMA_AM.CME` = fix + basis. The hedge carries a basis the deal
    does not, which is the unhedgeable floor this world exists to measure.
  * `ObservedBasis.LBMA_AM.PM` (FixingBridgeModel) is the PM fixing bridged off the AM path, and
    `ObservedBasis.LBMA_AM.CME.PM` (ChainedBasisModel) closes the AM/PM session 2-cycle. Both are
    BRIDGES off a finished path - they carry no state of their own to restamp.
  * The liability is the AVG-fixing product: TWO `CommodityAveragePriceSwapDeal` legs, one on each
    session's fixing, equal volume, one strike. Its PM leg is where the bridge premium lands.
  * `Carry_Drift` is stamped back onto the recalibrated spot block - see `trade_world`.

WHAT IS DELIBERATELY THE SAME, so the decks read against each other: the trade calendar, the
margin, the volume, the seeds, the corridor, the causal bound, the realized-path roll, and the
solver - `production_solver`'s validated block, applied on top of whatever the template JSON
carries (the authored chain job declares `DiffSolverV2` with no knobs, i.e. the framework
defaults; the harness's own validated knobs win here, as they do in both sibling decks).

WHY A SIBLING RATHER THAN A `--world` MODE. The arithmetic is shared and IMPORTED, not copied -
the carry curve and its knots, the causal bound, the static-short benchmark, the state restamp,
the observed-path npz, the E[dF|state] guard and the calibration subprocess all live in the two
older decks and are called from here. What differs is the FACTOR SET and the LIABILITY, and a
mode flag would have to branch on it inside every one of those primitives, which is the
magic-string branch `conventions.md` refuses. Instead each primitive took a COLUMN-SET argument -
a composed commodity name is a sum of archive columns, which is the one true statement about the
difference - and this file is this world's spelling of them.

Usage (PYTHONPATH must carry the repo - derivus is not installed):
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python experiments/plat_walk_forward_chain.py \\
        --start 2024-01 --months 1 --batch 1024 --batches 2 --inner-sub-batch 32 --fit-iters 15

JSON-is-the-contract: import derivus, load_json, run_job. No internal imports, no monkey-patching.
"""
import argparse
import copy
import datetime
import glob
import json
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from production_solver import apply_config, run
from production_walk_forward import (_atomic_write_csv, _atomic_write_json, _ts, calibrate,
                                     delta_corridor_schedule)
from plat_walk_forward_sync import (DAYS_IN_YEAR, ZERO_CURVE, carry_curve, carry_knots,
                                    carry_state, forward, guard_e_df_state, listed_expiries,
                                    load_archive, observed_scenario_npz, pf_bound,
                                    resolved_calibration_config, restamp_state_seeds,
                                    sofr_curve, static_short_usd_oz)

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s %(message)s')

FIX_COL = 'CommodityPrice.LBMA_AM'              # the AM fix: primary, liability reference
B_PM = 'ObservedBasis.LBMA_AM.PM'               # PM fix - AM fix (FixingBridgeModel)
B_CME = 'ObservedBasis.LBMA_AM.CME'             # AM-session CME basis (BasisLinkedSpotModel)
B_CME_PM = 'ObservedBasis.LBMA_AM.PM.CME'       # PM-session CME basis (ChainedBasisModel,
                                                # PM-branch tail: composes the PM futures spot)
SCALAR_COLS = (FIX_COL, B_PM, B_CME, B_CME_PM)  # the four scalar factors; the carry is the fifth
HEDGE_COLS = (FIX_COL, B_PM, B_CME_PM)          # the futures' Commodity 'LBMA_AM.PM.CME'
                                                # = PM fix + PM-session basis (execution at the
                                                # day's LAST event - no intra-row lookahead)
CARRY = 'PLATINUM_CARRY'


def price_factors(row, trade_date, last_query, sofr_cols):
    """The complete `Price Factors` block off ONE archive row - the t0 restamp of every market
    LEVEL, so no trade sees a level it could not have observed.

    The two CME bases declare each other as `Chained_Basis`: that 2-cycle IS the session chain,
    and `ChainedBasisModel` reads the declaration and nothing else. The AM basis lags its link
    (`Chained_Lag` 1 - it steps off YESTERDAY's PM basis, the chain's day boundary), so the
    same-row link on the PM side is the one generation edge. The PM fixing basis declares
    nothing - `FixingBridgeModel` is the open-link family and takes its parent from the name."""
    p0, state = float(row[FIX_COL]), carry_state(row)
    return {
        'FxRate.USD': {'Domestic_Currency': '', 'Interest_Rate': 'USD-SOFR', 'Spot': 1.0},
        FIX_COL: {'Currency': 'USD', 'Interest_Rate': ZERO_CURVE, 'Forward_Rate': CARRY,
                  'Spot': p0, 'Property_Aliases': ''},
        B_CME: {'Spot': float(row[B_CME]), 'Chained_Basis': B_CME_PM.split('.', 1)[1],
                'Chained_Lag': 1},
        B_CME_PM: {'Spot': float(row[B_CME_PM]), 'Chained_Basis': B_CME.split('.', 1)[1]},
        B_PM: {'Spot': float(row[B_PM])},
        f'ForwardRate.{CARRY}': {
            'Currency': 'USD',
            'Curve': carry_curve(state, trade_date, carry_knots(trade_date, last_query))},
        'InterestRate.USD-SOFR': {'Day_Count': 'ACT_365', 'Currency': 'USD', 'Sub_Type': None,
                                  'Curve': {'.Curve': {'meta': [],
                                                       'data': sofr_curve(row, sofr_cols)}}},
        # The primary's repo. z is the TOTAL carry read off the futures curve, so adding a rate
        # under the same exponential would count financing twice; zero is the statement. THREE
        # knots, because rate curves interpolate with Hermite and Hermite needs three.
        f'InterestRate.{ZERO_CURVE}': {'Day_Count': 'ACT_365', 'Currency': 'USD', 'Sub_Type': None,
                                       'Curve': {'.Curve': {'meta': [], 'data': [
                                           [0.0, 0.0], [1.0, 0.0], [30.0, 0.0]]}}},
    }


def fair_strike(row, trade_date, fixings, bridge_premium):
    """The desk-fair AVG strike: the equal-weight mean over the fixing calendar of the AM index
    forward S exp(z(tau) tau), and the SAME forward times exp(premium) for the PM leg.

    The PM leg's law is `FixingBridgeModel`'s - log B = log P + W dlog P + premium - so its fixing
    is the AM path scaled by the premium in expectation, and the premium is the only term that
    survives averaging. The two legs carry equal volume, so the product's fair is their mean; no
    basis term appears at all, because neither leg references a CME basis (that asymmetry against
    the hedge is exactly what this world measures)."""
    p0, state = float(row[FIX_COL]), carry_state(row)
    days = np.array([(pd.Timestamp(f) - pd.Timestamp(trade_date)).days for f in fixings], float)
    am = float(forward(p0, state, days / DAYS_IN_YEAR).mean())
    return 0.5 * (am + am * float(np.exp(bridge_premium)))


def build_deal_config(template, arch, trade_date, calibrated_md, args, delta_corridor=None):
    """The trade-date job: the two-leg AVG liability struck at fair - margin, the listed CME strip
    as tradables, and every market level read off the archive row. Returns `(cfg, info)`.

    The row is FORWARD-FILLED: a session whose futures snapshot failed quality leaves that day's
    basis NaN by construction (`build_plat_archive_am`), and the trade date can land on one."""
    row = arch.loc[:trade_date].ffill().iloc[-1]
    cfg = copy.deepcopy(template)
    calc = cfg['Calc']['Calculation']
    merge = cfg['Calc']['MergeMarketData']
    hp = calc['Hedging_Problem']
    calc['Base_Date'] = _ts(trade_date)
    merge['MarketDataFile'] = calibrated_md
    # the Objective is the TEMPLATE's (the authored job carries the ruled scale); only the swept
    # axes are overridable
    if args.huber_aversion is not None:
        hp['Objective']['Huber_Aversion'] = float(args.huber_aversion)
    if args.utility_scale is not None:
        hp['Objective']['Utility_Scale_Explicit'] = float(args.utility_scale)

    p0, state = float(row[FIX_COL]), carry_state(row)
    premium = float(json.load(open(calibrated_md))['MarketData']['Price Models'][
        f'FixingBridgeModel.{B_PM.split(".", 1)[1]}']['Bridge_Premium'])

    # --- liability: the 3-month-out averaging month, both sessions, paid +5 days ---------------
    avg_start = (pd.Timestamp(trade_date) + pd.offsets.MonthBegin(3)).normalize()
    avg_end = (avg_start + pd.offsets.MonthEnd(0)).normalize()
    pay = avg_end + pd.Timedelta(days=5)
    fixings = pd.bdate_range(avg_start, avg_end)
    k_fair = fair_strike(row, trade_date, fixings, premium)
    legs = hp['Liabilities']['CommodityAveragePriceSwapDeal']
    for leg in legs.values():                      # the template owns each leg's Commodity/routing
        leg.update({'Units': args.volume / len(legs),
                    'Fixed_Price': round(k_fair - args.margin, 6), 'Settlement_Date': _ts(pay),
                    'Sampling_Data': [[_ts(f), 0.0, 1.0] for f in fixings]})

    # --- tradables: the listed CME ladder on the COMPOSED fix+basis reference -------------------
    futs = hp['Tradable_Instruments']['CommodityFutureDeal']
    mats = listed_expiries(args.expiries, trade_date, n=len(futs))
    positions, setts, margins, limits = {}, {}, {}, {}
    spot_cme = p0 + float(row[B_PM]) + float(row[B_CME_PM])   # PM-session synthetic CME spot
    for name, mat in zip(sorted(futs), mats):
        futs[name]['Maturity_Date'] = _ts(mat)
        positions[name] = 0
        setts[name] = round(float(forward(
            spot_cme, state, (mat - pd.Timestamp(trade_date)).days / DAYS_IN_YEAR)), 4)
        margins[name] = {'Method': 'per_contract',
                         'Amount': round(0.085 * setts[name] * futs[name]['Contract_Size'], 0)}
        limits[name] = dict(hp['Evaluator']['Position_Limits'][name])
    hp['Tradable_Instruments']['CashAccountDeal']['USD_CASH']['Investment_Horizon'] = _ts(pay)
    ps = hp['Portfolio_State']
    ps['Positions'], ps['Settlement_Prices'], ps['Initial_Margin'] = positions, setts, margins
    hp['Evaluator']['Position_Limits'] = limits
    if delta_corridor is not None:
        # with a long cap, the corridor's clamps widen into the asymmetric net range - the
        # ramp mandate and the long allowance compose instead of excluding each other
        bounds = ({'lo_floor': -60.0, 'hi_cap': float(args.long_cap)}
                  if args.long_cap is not None else {})
        hp['Evaluator']['Total_Position_Schedule'] = delta_corridor_schedule(
            trade_date, fixings, delta_corridor, **bounds)
    if args.max_trade is not None:
        hp['Evaluator']['Max_Trade_Per_Step'] = float(args.max_trade)
    if args.long_cap is not None:
        # the ruled asymmetric NET range [-60, +long_cap]: per-leg boxes open on the long
        # side, the abs limit widens to the short floor, and the net object is the flat
        # schedule - or the corridor above, already clamped into the same range
        for lim in limits.values():
            lim['Max_Position'] = int(args.long_cap)
        hp['Evaluator']['Total_Position_Abs_Limit'] = 60.0
        if delta_corridor is None:
            hp['Evaluator']['Total_Position_Schedule'] = [
                {'Step': 0, 'Min_Total': -60.0, 'Max_Total': float(args.long_cap)}]

    # No Spot_Price_History: its only live consumer is the utility-scale formula, which the
    # template's Utility_Scale_Explicit overrides; the policy's features are process-revealed
    # states restamped from the archive. Supplying it buys nothing and re-arms the
    # timeline-shift trap the bundle now refuses (history at/after the base date).
    ps.pop('Spot_Price_History', None)

    merge['ExplicitMarketData']['Price Factors'] = price_factors(
        row, trade_date, max(max(fixings), max(mats)), arch.columns)
    merge['ExplicitMarketData'].setdefault('Price Models', {})
    return cfg, {'k_fair': k_fair, 'mats': mats, 'pay': pay, 'fixings': fixings,
                 'premium': premium}


def trade_world(arch, trade_date, calibrated_md, out_path, carry_drift):
    """The per-trade world file: the calibrated block with its t0 STATE seeds re-filtered to the
    trade date, and the spot's `Carry_Drift` switch stamped back on.

    Carry_Drift is a RULING, not a fit, and `GARCHSpotCalibration` declares no passthrough for it
    the way its sibling `Convexity_Correction` does - so a recalibration silently DROPS the switch
    under which every futures leg is near-driftless in the training world instead of rolling down
    the curve with certainty. Restoring it in the world file the driver itself authors is the
    JSON-level fix; the framework fix is that passthrough field.

    Only the AM spot and the CME basis carry restampable state. The PM legs are bridges off a
    finished path (`FixingBridgeModel`, `ChainedBasisModel`) and are memoryless by their own law."""
    restamp_state_seeds(arch, trade_date, calibrated_md, out_path,
                        spot_col=FIX_COL, basis_col=B_CME)
    md = json.load(open(out_path))
    spot = md['MarketData']['Price Models'][f'GARCHSpotModel.{FIX_COL.split(".", 1)[1]}']
    spot['Carry_Drift'] = carry_drift
    _atomic_write_json(out_path, md)
    return out_path


def one_trade(template, arch, trade_date, calibrated_md, args, run_dir, tag):
    """Train the day-1 policy per seed in the calibrated chain world, then roll the frozen
    ensemble along the REALIZED archive path. Returns the recorded row."""
    calibrated_md = trade_world(arch, trade_date, calibrated_md,
                                os.path.join(run_dir, f'md_{tag}_trade.json'), args.carry_drift)
    cfg, info = build_deal_config(template, arch, trade_date, calibrated_md, args,
                                  delta_corridor=args.delta_corridor)
    # the observed window BEFORE training: a month the gap guard refuses must refuse in seconds,
    # not after a policy whose roll can never run
    obs_npz = os.path.abspath(os.path.join(run_dir, f'obs_{tag}.npz'))
    knots = carry_knots(trade_date, max(max(info['fixings']), max(info['mats'])))
    horizon = (info['pay'] - pd.Timestamp(trade_date)).days + 10
    observed_scenario_npz(arch, trade_date, obs_npz, horizon, knots, max_gap=args.max_gap,
                          cols=SCALAR_COLS)

    ckpts, train_us, v0s, market_dim = [], [], [], None
    for seed in args.seeds:
        ckpt = os.path.abspath(os.path.join(run_dir, f'value_fn_{tag}_s{seed}.pt'))
        ckpts.append(ckpt)
        if os.path.exists(ckpt):
            logging.info('=== TRAIN %s seed=%d SKIP (checkpoint exists) ===', tag, seed)
            train_us.append(None)
            v0s.append(None)
            continue
        train = apply_config(copy.deepcopy(cfg), batch=args.batch, seed=seed, save=ckpt)
        tcalc = train['Calc']['Calculation']
        tcalc['Simulation_Batches'] = args.batches
        tcalc['Inner_Sub_Batch'] = args.inner_sub_batch
        if args.fit_iters is not None:
            tcalc['Hedging_Problem']['Solver']['DiffV2_Fit_Iters'] = args.fit_iters
        if args.grid_levels is not None:
            tcalc['Hedging_Problem']['Solver']['Training_Action_Grid_Levels_Per_Axis'] = \
                args.grid_levels
        if args.churn_lambda is not None:
            tcalc['Hedging_Problem']['Solver']['DiffV2_Churn_Lambda'] = args.churn_lambda
        if args.solver is not None:
            tcalc['Hedging_Problem']['Solver']['Object'] = args.solver
        if args.temporal_proximity is not None:
            tcalc['Hedging_Problem']['Solver']['DiffV2_Temporal_Proximity'] = \
                args.temporal_proximity
        logging.info('=== TRAIN %s seed=%d (fair=%.2f, strike=%.2f, bridge premium=%+.6f) ===',
                     tag, seed, info['k_fair'], info['k_fair'] - args.margin, info['premium'])
        tdiag = run(train, f'train_{tag}_s{seed}')
        verdict = (tdiag.get('verdict') or {}).get('greedy') or {}
        train_us.append(None if verdict.get('u_mean') is None else round(verdict['u_mean'], 4))
        v0s.append(tdiag.get('V_0'))
        market_dim = tdiag.get('market_dim')

    roll = apply_config(copy.deepcopy(cfg), batch=1, seed=args.seeds[0], load=ckpts,
                        stepper_rollout=True, randomize_initial_state=False)
    roll['Calc']['Calculation']['Inner_Sub_Batch'] = args.roll_inner
    if args.grid_levels is not None:
        roll['Calc']['Calculation']['Hedging_Problem']['Solver'][
            'Training_Action_Grid_Levels_Per_Axis'] = args.grid_levels
    if args.churn_lambda is not None:
        roll['Calc']['Calculation']['Hedging_Problem']['Solver'][
            'DiffV2_Churn_Lambda'] = args.churn_lambda
    if args.solver is not None:
        roll['Calc']['Calculation']['Hedging_Problem']['Solver']['Object'] = args.solver
    if args.temporal_proximity is not None:
        roll['Calc']['Calculation']['Hedging_Problem']['Solver'][
            'DiffV2_Temporal_Proximity'] = args.temporal_proximity
    roll['Calc']['Calculation']['Observed_Scenario'] = obs_npz
    logging.info('=== ROLL %s (stepper, realized path, %d-seed ensemble, inner=%d) ===',
                 tag, len(ckpts), args.roll_inner)
    rdiag = run(roll, f'roll_{tag}')
    json.dump(rdiag, open(os.path.join(run_dir, f'diag_{tag}.json'), 'w'), indent=1, default=str)

    sv = rdiag.get('stepper_verdict') or {}
    greedy = (sv.get('greedy') or {}).get('wT_mean')
    nohedge = (sv.get('nohedge') or {}).get('wT_mean')
    q = np.array(sv.get('greedy_q_traj') or [[0.0]])
    bound = pf_bound(arch, trade_date, info['mats'], info['pay'], spot_cols=HEDGE_COLS)
    st_short = static_short_usd_oz(arch, trade_date, info['mats'], info['pay'], args.volume,
                                   spot_cols=HEDGE_COLS)
    return {
        'trade': tag, 'world': 'chain', 'margin': args.margin, 'corridor': args.delta_corridor,
        'fair': round(info['k_fair'], 2), 'strike': round(info['k_fair'] - args.margin, 2),
        'n_seeds': len(args.seeds), 'market_dim': market_dim,
        'train_u': train_us[0], 'train_u_seeds': train_us, 'V_0': v0s[0],
        'greedy_usd_oz': None if greedy is None else round(greedy / args.volume, 2),
        'nohedge_usd_oz': None if nohedge is None else round(nohedge / args.volume, 2),
        'static_short_usd_oz': (None if nohedge is None
                                else round(nohedge / args.volume + st_short, 2)),
        'pf_bound': round(bound, 2),
        'bound_pass': (None if (greedy is None or nohedge is None)
                       else bool(greedy / args.volume <= nohedge / args.volume + bound + 1e-6)),
        'churn': round(float(np.abs(np.diff(q, axis=0)).sum()), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--archive', default='data/plat_archive_am.csv',
                    help='Chain archive: AM fix, three bases, z(0.5), z(1.0) - already decomposed.')
    ap.add_argument('--rates', default='data/plat_archive.csv',
                    help='Rate archive the SOFR curve is read from (a second, EOD clock); this '
                         'world simulates no rate, so it discounts on a per-trade static curve.')
    ap.add_argument('--expiries', default='data/platinum_futures_contract_expiry.csv')
    ap.add_argument('--calibration-config', default='data/plat_world_am/calibration_config_chain.json')
    ap.add_argument('--marketdata', default='data/plat_world_am/MarketDataRF_am_chain.json',
                    help='Source MarketDataRF: its Model Configuration routes the five factors; '
                         'its Price Models are wiped and refit per trade date.')
    ap.add_argument('--deal-template', default='data/plat_world_am/job_aps_hedge_am.json',
                    help='The authored chain job - it owns the leg routing, the Objective and the '
                         'Evaluator; this driver restamps only what the trade date decides.')
    ap.add_argument('--start', default='2020-01', help='First trade month, YYYY-MM.')
    ap.add_argument('--months', type=int, default=12)
    ap.add_argument('--recal-months', type=int, default=1,
                    help='Recalibrate every N months. The chain block carries t0 STATE the fit '
                         'stamps (H0, Mu_0, the mixture start probability); the first two are '
                         'restamped per trade, the third only a fresh fit can move.')
    ap.add_argument('--garch-years', type=float, default=10.0,
                    help='Long window for the GARCH shape (window-study ruling)')
    ap.add_argument('--cal-years', type=float, default=3.0,
                    help='Rolling calibration window, years back from the trade date - the '
                         'basis/chain families\' ruled window. The carry runs its own 1y location '
                         'window inside `Location_Window`.')
    ap.add_argument('--margin', type=float, default=10.0, help='$/oz (strike = fair - margin).')
    ap.add_argument('--volume', type=float, default=2500.0, help='Total swap volume (oz), split '
                                                                 'equally across the session legs.')
    ap.add_argument('--batch', type=int, default=8192)
    ap.add_argument('--batches', type=int, default=2,
                    help='Training stream length: N-1 fit batches, then a held-out one.')
    ap.add_argument('--inner-sub-batch', type=int, default=128, help='Training inner-MC draws.')
    ap.add_argument('--roll-inner', type=int, default=256,
                    help='Inner_Sub_Batch for the realized-path ROLL only (Batch_Size=1, so a '
                         'large inner sub-batch is nearly free and de-noises the causal argmax).')
    ap.add_argument('--seeds', type=int, nargs='+', default=[7])
    ap.add_argument('--carry-drift', choices=['Yes', 'No'], default='Yes',
                    help="Stamped onto the recalibrated spot block - see `trade_world`.")
    ap.add_argument('--max-gap', type=int, default=5,
                    help='Largest hole (days) allowed in the realized window.')
    ap.add_argument('--fit-iters', type=int, default=None)
    ap.add_argument('--huber-aversion', type=float, default=None,
                    help='Override the template Objective\'s Huber_Aversion (trains a different '
                         'policy).')
    ap.add_argument('--utility-scale', type=float, default=None,
                    help='Override the template Objective\'s Utility_Scale_Explicit.')
    ap.add_argument('--max-trade', type=float, default=None,
                    help='Evaluator Max_Trade_Per_Step: per-leg |dq| cap per decision step '
                         'at the argmax (execution only; checkpoints re-rollable under it).')
    ap.add_argument('--long-cap', type=float, default=None,
                    help='Open the NET range to [-60, +long_cap] (flat Total_Position_Schedule '
                         '+ per-leg boxes). Changes the action space: a RETRAIN, never a '
                         're-roll. Mutually exclusive with --delta-corridor.')
    ap.add_argument('--solver', default=None,
                    help="Solver Object override, e.g. CoupledDiffSolver (the temporally-"
                         "coupled DiffSolverV2 sibling).")
    ap.add_argument('--temporal-proximity', type=float, default=None,
                    help='Solver DiffV2_Temporal_Proximity (CoupledDiffSolver only).')
    ap.add_argument('--churn-lambda', type=float, default=None,
                    help='Solver DiffV2_Churn_Lambda: quadratic repositioning charge '
                         '$/(contract^2) at the argmax and the training labels.')
    ap.add_argument('--grid-levels', type=int, default=None,
                    help='Override Solver Training_Action_Grid_Levels_Per_Axis (train AND '
                         'roll argmax). Changes the action space: a retrain.')
    ap.add_argument('--delta-corridor', type=float, default=None,
                    help='Causal delta-ramp corridor band on the SIGNED total position, applied '
                         'to BOTH train and roll.')
    ap.add_argument('--run-dir', default=None)
    args = ap.parse_args()

    arch = load_archive(args.archive, args.rates)
    template = json.load(open(args.deal_template))
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = args.run_dir or os.path.join(
        'artifacts', 'walk_forward_chain', f'{stamp}_{args.start}_{args.months}m')
    os.makedirs(run_dir, exist_ok=True)
    logging.info('run dir: %s  seeds: %s  corridor: %s  carry drift: %s',
                 run_dir, args.seeds, args.delta_corridor, args.carry_drift)

    # the chain archive IS the calibration archive - already decomposed, nothing to rebuild; only
    # its path needs resolving, because `calibrate` runs the calibrator from `experiments/`
    cal_cfg = resolved_calibration_config(args.calibration_config, args.archive,
                                          os.path.join(run_dir, 'calibration_config.json'))

    done = {}
    for f in sorted(glob.glob(os.path.join(run_dir, 'row_*.json'))):
        r = json.load(open(f))
        done[r['trade']] = r
    if done:
        logging.info('RESUME: %d completed trade(s) found in run dir; will skip them', len(done))

    rows, calibrated_md = [], None
    for m in range(args.months):
        trade_date = (pd.Timestamp(args.start + '-01') + pd.offsets.MonthBegin(m)
                      + pd.offsets.BDay(0)).normalize()
        tag = trade_date.strftime('%Y%m')
        try:
            if m % args.recal_months == 0:
                calibrated_md = os.path.abspath(os.path.join(run_dir, f'md_{tag}_chain.json'))
                cal_end = trade_date.strftime('%Y-%m-%d')
                cal_start = (trade_date - pd.DateOffset(years=args.cal_years)).strftime('%Y-%m-%d')
                # per-family windows (the window-study rulings): the basis/chain/fix/carry
                # families fit on the rolling short window; the GARCH shape wants the LONG
                # one (a 3y fit reads nu ~ 23, near-Gaussian tails), so the spot block is
                # spliced in from a long-window pass
                calibrate(os.path.abspath(args.marketdata), cal_cfg, cal_end, calibrated_md,
                          start=cal_start)
                md_long = calibrated_md.replace('.json', '_garchlong.json')
                cal_start_g = (trade_date - pd.DateOffset(years=args.garch_years)).strftime('%Y-%m-%d')
                calibrate(os.path.abspath(args.marketdata), cal_cfg, cal_end, md_long,
                          start=cal_start_g)
                _short = json.load(open(calibrated_md))
                _key = f'GARCHSpotModel.{FIX_COL.split(".", 1)[1]}'
                _short['MarketData']['Price Models'][_key] = json.load(
                    open(md_long))['MarketData']['Price Models'][_key]
                _atomic_write_json(calibrated_md, _short)
                mats_g = listed_expiries(args.expiries, trade_date)
                taus = [(e - trade_date).days / DAYS_IN_YEAR for e in mats_g]
                for leg, by_state in guard_e_df_state(
                        arch, cal_end, taus, mats=mats_g, hedge_cols=HEDGE_COLS,
                        state_cols={'b_cme': B_CME, 'b_pm': B_PM, 'b_cme_pm': B_CME_PM},
                        control_cols=(FIX_COL,), control='AM fix (control)').items():
                    logging.info('GUARD %s %-22s %s', cal_end, leg, '  '.join(
                        f'd~{k}: {v[0]:+.5f} (t {v[1]:+.2f})' for k, v in by_state.items()))
            if tag in done:
                logging.info('TRADE %s: SKIP (already completed, resuming)', tag)
                rec = done[tag]
            else:
                rec = one_trade(template, arch, trade_date, calibrated_md, args, run_dir, tag)
                logging.info('TRADE %s: greedy=%s nohedge=%s $/oz  bound=%s PASS=%s  churn=%s',
                             tag, rec['greedy_usd_oz'], rec['nohedge_usd_oz'], rec['pf_bound'],
                             rec['bound_pass'], rec['churn'])
                _atomic_write_json(os.path.join(run_dir, f'row_{tag}.json'), rec)
        except Exception as exc:
            logging.exception('TRADE %s FAILED', tag)
            rec = {'trade': tag, 'world': 'chain', 'margin': args.margin, 'fair': None,
                   'strike': None, 'n_seeds': len(args.seeds), 'market_dim': None, 'train_u': None,
                   'train_u_seeds': None, 'V_0': None, 'greedy_usd_oz': None,
                   'nohedge_usd_oz': None, 'static_short_usd_oz': None, 'pf_bound': None,
                   'bound_pass': None, 'churn': None, 'corridor': args.delta_corridor,
                   'error': str(exc)}
        rows.append(rec)
        _atomic_write_csv(os.path.join(run_dir, 'trades.csv'), rows)

    df = pd.DataFrame(rows)
    g = df['greedy_usd_oz']
    n_ok = int(g.notna().sum())
    print('\n===== SESSION-CHAIN WALK-FORWARD ($/oz) =====')
    print(df.to_string(index=False))
    # denominators are COMPLETED trades: g.mean() skips NaN, so counting failed rows in the
    # denominator reports a pass-rate over trades the mean never saw
    print(f"\ngreedy  mean {g.mean():+.2f}  min {g.min():+.2f}  max {g.max():+.2f}  "
          f"positives {int((g > 0).sum())}/{n_ok}  |  nohedge mean {df['nohedge_usd_oz'].mean():+.2f}"
          f"  |  bound-PASS {int(df['bound_pass'].fillna(False).sum())}/{n_ok}"
          f"  |  FAILED {len(df) - n_ok}  |  margin {args.margin:+.2f}"
          f"  |  greedy - margin {g.mean() - args.margin:+.2f}  |  seeds {args.seeds}")
    print('run dir:', run_dir)


if __name__ == '__main__':
    main()
