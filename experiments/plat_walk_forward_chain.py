"""Walk-forward backtest of the platinum hedge in the SESSION-CHAIN world - the AM-fix-anchored
sibling of `plat_walk_forward_sync.py`, which is itself `production_walk_forward.py`'s.

THE WORLD INVERTS, and that is the whole of what changes. The sync deck's primary is the
CME-implied spot with the LBMA fix carried as a basis off it; here the LBMA **AM fix is the
primary** and the CME curve hangs off it, so every routing is the other way round:

  * `CommodityPrice.LBMA_AM` (GARCHSpotModel) is the fixing itself - the liability's own
    reference, and NOT tradable.
  * `ObservedBasis.LBMA_AM.CME` (BasisLinkedSpotModel) is the AM-session CME basis, and
    `ObservedBasis.LBMA_AM.PM` (FixingBridgeModel) the PM fixing bridged off the AM path, with
    `ObservedBasis.LBMA_AM.PM.CME` (ChainedBasisModel) closing the AM/PM session 2-cycle. The PM
    two are BRIDGES off a finished path - they carry no state of their own to restamp - and enter
    the world only where a deal reaches the PM session (`price_factors`).
  * The futures reference a COMPOSED name, the fix plus its bases (`LBMA_AM.CME` at the AM print,
    `LBMA_AM.PM.CME` at the PM), so the hedge carries a basis the deal does not, which is the
    unhedgeable floor this world exists to measure.
  * The liability is the template's `CommodityAveragePriceSwapDeal` legs at equal volume and one
    strike: the AM fix alone, or the AVG product's two legs, one on each session's fixing, the PM
    leg where the bridge premium lands. The trade date and the fixings are London fixing days.
  * `Carry_Drift` is stamped back onto the recalibrated spot block - see `trade_world`.

WHAT IS DELIBERATELY THE SAME, so the decks read against each other: the margin, the volume,
the seeds, the corridor, the causal bound, the realized-path roll, and the solver -
`production_solver`'s validated block, applied on top of whatever the template JSON carries (the
authored chain job declares `DiffSolverV2` with no knobs, i.e. the framework defaults; the
harness's own validated knobs win here, as they do in both sibling decks).

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
from xml.etree import ElementTree

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
CARRY = 'PLATINUM_CARRY'
FIXING_CALENDAR = 'London'                      # the LBMA fixes on London business days
WEEK = ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday')


def composed_cols(name):
    """The archive columns a composed commodity name sums, its positional chain: `LBMA_AM.PM.CME` is
    the fix `CommodityPrice.LBMA_AM`, then `ObservedBasis.LBMA_AM.PM`, then
    `ObservedBasis.LBMA_AM.PM.CME`."""
    parts = name.split('.')
    return (f'CommodityPrice.{parts[0]}',) + tuple(
        f'ObservedBasis.{".".join(parts[:i + 1])}' for i in range(1, len(parts)))


def hedge_cols(template):
    """The columns the futures' one composed `Commodity` sums: the hedge's pricing spot, executed at
    the print its last basis is observed at."""
    (name,) = {f['Commodity'] for f in template['Calc']['Calculation']['Hedging_Problem'][
        'Tradable_Instruments']['CommodityFutureDeal'].values()}
    return composed_cols(name)


def replayed_cols(cfg):
    """The scalar columns the job simulates, so the roll replays each: the columns of every composed
    name its deals reference, closed under the `Chained_Basis` its price factors declare (a chain
    member entering the universe pulls every member, each with its own composed name)."""
    hp = cfg['Calc']['Calculation']['Hedging_Problem']
    factors = cfg['Calc']['MergeMarketData']['ExplicitMarketData']['Price Factors']
    todo = [d['Commodity'] for d in [*hp['Liabilities']['CommodityAveragePriceSwapDeal'].values(),
                                     *hp['Tradable_Instruments']['CommodityFutureDeal'].values()]]
    cols = []
    while todo:
        for c in composed_cols(todo.pop()):
            if c not in cols:
                cols.append(c)
                if factors[c].get('Chained_Basis'):
                    todo.append(factors[c]['Chained_Basis'])
    return cols


def fixing_day(template):
    """The London business day of the calendar file the template names, read as
    `Config.parse_calendar_file` reads it: `Holidays` a `date|name, ...` list, `Weekends` the days
    it names."""
    cal = next(e.attrib for e in ElementTree.parse(template['Calc']['CalendDataFile']).getroot().iter()
               if e.attrib.get('Location') == FIXING_CALENDAR)
    return pd.offsets.CustomBusinessDay(
        holidays=[h.split('|')[0] for h in cal['Holidays'].split(', ')] if cal['Holidays'] else [],
        weekmask=' '.join(d[:3] for d in WEEK if d not in cal['Weekends']))


def price_factors(row, trade_date, last_query, sofr_cols, chained):
    """The complete `Price Factors` block off ONE archive row - the t0 restamp of every market
    LEVEL, so no trade sees a level it could not have observed.

    Where the job reaches the PM-session basis (`chained`) the two CME bases declare each other
    as `Chained_Basis`: that 2-cycle IS the session chain, and `ChainedBasisModel` reads the
    declaration and nothing else. The AM basis lags its link (`Chained_Lag` 1 - it steps off
    YESTERDAY's PM basis, the chain's day boundary), so the same-row link on the PM side is the
    one generation edge. Otherwise the AM basis is the open link off the fix and nothing pulls
    the PM pair, whose prints come after the AM minute. The PM fixing basis declares nothing -
    `FixingBridgeModel` is the open-link family and takes its parent from the name."""
    p0, state = float(row[FIX_COL]), carry_state(row)
    return {
        'FxRate.USD': {'Domestic_Currency': '', 'Interest_Rate': 'USD-SOFR', 'Spot': 1.0},
        FIX_COL: {'Currency': 'USD', 'Interest_Rate': ZERO_CURVE, 'Forward_Rate': CARRY,
                  'Spot': p0, 'Property_Aliases': ''},
        B_CME: {'Spot': float(row[B_CME]), **({'Chained_Basis': B_CME_PM.split('.', 1)[1],
                                              'Chained_Lag': 1} if chained else {})},
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


def fair_strike(row, trade_date, fixings, premia):
    """The desk-fair strike of legs at equal volume: the mean of the legs' fairs, each the equal-weight
    mean over the fixing calendar of the AM index forward S exp(z(tau) tau) times exp(premium), with
    `premia` one bridge premium per leg (none on the AM fix itself).

    A PM leg's law is `FixingBridgeModel`'s - log B = log P + W dlog P + premium - so its fixing is
    the AM path scaled by the premium in expectation, and the premium is the only term that survives
    averaging. No basis term appears at all, because no leg references a CME basis (that asymmetry
    against the hedge is exactly what this world measures)."""
    p0, state = float(row[FIX_COL]), carry_state(row)
    days = np.array([(pd.Timestamp(f) - pd.Timestamp(trade_date)).days for f in fixings], float)
    am = float(forward(p0, state, days / DAYS_IN_YEAR).mean())
    return sum(am * float(np.exp(p)) for p in premia) / len(premia)


def allocation_knots(trade_date, fixings, mats, names):
    """The `Evaluator.Allocation_Weights` schedule as DATA: at each decision step (= calendar-day
    offset from the trade date, the same clock `delta_corridor_schedule` keys on), leg i's weight
    is its share of the fixings STILL UNFIXED, where a fixing is hedged by the first listed expiry
    ON OR AFTER it — the only contract still alive when that fixing prints.

    So the split is the remaining-exposure profile the listed strip actually carries, and every
    date in it is known at the trade date (causal, like the corridor). A leg whose expiry has
    passed covers no unfixed fixing and drops out of the ladder on its own; so does a front
    contract that expires before the averaging window opens at all, which is a statement about
    the quarterly ladder rather than a defect. Knots are emitted only where the weights MOVE, so
    the table is the handful of days the composition changes, not one row per day.

    A fixing with NO listed expiry after it RAISES: clamping it onto the last (by then dead) leg
    would hand the ladder a schedule whose weight sits on a contract that expires mid-horizon —
    a hard corridor abort at best, phantom cover at worst. The strip is the caller's to widen.

    TODO — the carry tilt. Multiplying leg i's share by `exp(z(tau_i)·tau_i)` before renormalizing
    would let the allocator prefer the cheaper point of the curve; the seam is the `share` vector
    below, and the trade-date curve is `carry_curve(carry_state(row), ...)`. v1 is pure exposure
    so the tilt is a measurable change of its own rather than a confound in the first read."""
    f = np.array([(pd.Timestamp(x) - pd.Timestamp(trade_date)).days for x in fixings])
    m = np.array([(pd.Timestamp(x) - pd.Timestamp(trade_date)).days for x in mats])
    leg = np.searchsorted(m, f)                             # first expiry on or after each fixing
    if leg.max() >= len(m):
        raise ValueError(
            f'allocation_knots: fixings run to {max(fixings).date()} but the listed strip ends '
            f'{max(mats).date()} — {int((leg >= len(m)).sum())} fixing(s) have no live contract '
            f'to be hedged in; widen the strip (more CommodityFutureDeal legs).')
    knots, last = [], None
    for step in range(int(f.max())):
        unfixed = f > step
        if not unfixed.any():
            break
        share = np.bincount(leg[unfixed], minlength=len(m)) / unfixed.sum()
        key = tuple(np.round(share, 9))
        if key != last:
            knots.extend({'Step': step, 'Instrument': n, 'Weight': float(w)}
                         for n, w in zip(names, share))
            last = key
    return knots


def apply_objective_flags(objective, args):
    """The CLI's `Objective` overrides, applied on top of the TEMPLATE's authored block (which
    carries the ruled shape and scale) — the swept axes, and nothing else. Mutates in place.

    `Utility_Scale_Mode='conditional_sim'` CLEARS the template's `Utility_Scale_Explicit`: the
    measured schedule sets the LEVEL of c as well as its shape, so the authored literal is exactly
    what the mode replaces, and the engine refuses the pair rather than picking a winner. Without
    the clear the flag is unusable — every authored job here carries a ruled
    `Utility_Scale_Explicit` and the run would die at the first bundle build.

    A `--utility-scale` typed BESIDE the mode is left standing, deliberately: the template's
    literal is an inherited default the mode supersedes, but a value the caller typed on the same
    command line is a contradiction they should hear about, and the engine says so by name.

    The dose moves with the clear, which is the part to carry into a campaign: `Huber_Aversion` is
    dimensionless in units of c, so re-levelling c re-doses the aversion. Under `conditional_sim`
    it is dosed against the MEASURED scale (the frame lock logs the resulting risk aversion per
    dollar), not against the retired literal."""
    if args.huber_aversion is not None:
        objective['Huber_Aversion'] = float(args.huber_aversion)
    if args.huber_delta is not None:
        objective['Huber_Delta'] = float(args.huber_delta)
    if args.utility_scale is not None:
        objective['Utility_Scale_Explicit'] = float(args.utility_scale)
    if args.objective is not None:
        objective['Object'] = args.objective
    if args.cara_gamma is not None:
        objective['CARA_Gamma'] = float(args.cara_gamma)
    if args.reference_wealth is not None:
        objective['Reference_Wealth'] = float(args.reference_wealth)
    if args.up_aversion is not None:
        objective['Up_Aversion'] = float(args.up_aversion)
    if args.up_knee is not None:
        objective['Up_Knee'] = float(args.up_knee)
    # The two objective dials ride the SHARED cfg, so train and roll carry them together: both
    # enter the labels, and the checkpoint refuses a load under the other setting.
    if args.reference_mode is not None:
        objective['Reference_Mode'] = args.reference_mode
    if args.utility_scale_mode is not None:
        objective['Utility_Scale_Mode'] = args.utility_scale_mode
        if args.utility_scale_mode == 'conditional_sim' and args.utility_scale is None:
            retired = objective.pop('Utility_Scale_Explicit', None)
            if retired is not None:
                logging.info(
                    'conditional_sim: retiring the template Utility_Scale_Explicit=%s — the '
                    'measured schedule sets the level as well as the shape, so Huber_Aversion is '
                    'now dosed against the MEASURED scale (see the frame-lock log)', retired)
    if args.utility_scale_floor is not None:
        objective['Utility_Scale_Floor_Frac'] = float(args.utility_scale_floor)
    return objective


def build_deal_config(template, arch, trade_date, calibrated_md, args, delta_corridor=None):
    """The trade-date job: the template's liability legs struck at fair - margin, the listed CME strip
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
    apply_objective_flags(hp['Objective'], args)

    state = carry_state(row)
    models = json.load(open(calibrated_md))['MarketData']['Price Models']

    # --- liability: the averaging month `--avg-lead-months` out (3: the desk's product; 0: the
    # trade month itself), its London fixing days, paid +5 days; `--avg-days` keeps only the
    # month's last N fixings, the short product that gives one independent tape a month ---------
    first = pd.Timestamp(trade_date).normalize().replace(day=1)
    avg_start = first + pd.offsets.MonthBegin(args.avg_lead_months) if args.avg_lead_months else first
    avg_end = (avg_start + pd.offsets.MonthEnd(0)).normalize()
    pay = avg_end + pd.Timedelta(days=5)
    fixings = pd.date_range(avg_start, avg_end, freq=fixing_day(template))
    if args.avg_days:
        fixings = fixings[-int(args.avg_days):]
    legs = hp['Liabilities']['CommodityAveragePriceSwapDeal']
    # a leg's fixing is the AM path through its bridges, each adding its premium (a KeyError names
    # a basis no FixingBridgeModel carries: such a leg has no fair here)
    premia = [sum(float(models[f'FixingBridgeModel.{c.split(".", 1)[1]}']['Bridge_Premium'])
                  for c in composed_cols(leg['Commodity'])[1:]) for leg in legs.values()]
    k_fair = fair_strike(row, trade_date, fixings, premia)
    for leg in legs.values():                      # the template owns each leg's Commodity/routing
        leg.update({'Units': args.volume / len(legs),
                    'Fixed_Price': round(k_fair - args.margin, 6), 'Settlement_Date': _ts(pay),
                    'Sampling_Data': [[_ts(f), 0.0, 1.0] for f in fixings]})

    # --- tradables: the listed CME ladder on the COMPOSED fix+basis reference -------------------
    futs = hp['Tradable_Instruments']['CommodityFutureDeal']
    mats = listed_expiries(args.expiries, trade_date, n=len(futs))
    positions, setts, margins, limits = {}, {}, {}, {}
    spot_cme = sum(float(row[c]) for c in hedge_cols(template))   # the synthetic CME spot
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
    if args.allocation:
        # FACTORED action space: the DP picks the net cover on a ladder and this schedule splits
        # it, so --grid-levels now counts rungs on the TOTAL rather than levels per leg.
        hp['Evaluator']['Allocation_Weights'] = allocation_knots(
            trade_date, fixings, mats, sorted(futs))
    if args.allocation_mode is not None:
        # 'carry_variance' DERIVES the split in the solver; declaring it beside --allocation's
        # exposure table is the contradiction the runtime refuses loud.
        hp['Evaluator']['Allocation_Mode'] = (
            'Carry_Variance' if args.allocation_mode == 'carry_variance' else 'Exposure')
    if args.calendar_spread_bps is not None:
        hp['Evaluator']['Calendar_Spread_Bps'] = float(args.calendar_spread_bps)
    if args.max_trade is not None:
        hp['Evaluator']['Max_Trade_Per_Step'] = float(args.max_trade)
    if args.deadband is not None:
        hp['Evaluator']['Decision_Deadband_Sigma'] = float(args.deadband)
    if args.long_cap is not None:
        # the ruled asymmetric NET range [-60, +long_cap]: per-leg boxes open on the long
        # side, the abs limit widens to the short floor, and the net object is the flat
        # schedule - or the corridor above, already clamped into the same range
        for lim in limits.values():
            lim['Max_Position'] = int(args.long_cap)
        hp['Evaluator']['Total_Position_Abs_Limit'] = 60.0
        if delta_corridor is None and not args.no_schedule:
            hp['Evaluator']['Total_Position_Schedule'] = [
                {'Step': 0, 'Min_Total': -60.0, 'Max_Total': float(args.long_cap)}]

    # No Spot_Price_History: its only live consumer is the utility-scale formula, which the
    # template's Utility_Scale_Explicit overrides; the policy's features are process-revealed
    # states restamped from the archive. Supplying it buys nothing and re-arms the
    # timeline-shift trap the bundle now refuses (history at/after the base date).
    ps.pop('Spot_Price_History', None)

    reached = {c for d in [*legs.values(), *futs.values()] for c in composed_cols(d['Commodity'])}
    merge['ExplicitMarketData']['Price Factors'] = price_factors(
        row, trade_date, max(max(fixings), max(mats)), arch.columns, chained=B_CME_PM in reached)
    merge['ExplicitMarketData'].setdefault('Price Models', {})
    return cfg, {'k_fair': k_fair, 'mats': mats, 'pay': pay, 'fixings': fixings,
                 'premia': premia}


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
    pm = md['MarketData']['Price Models']
    spot = pm[f'GARCHSpotModel.{FIX_COL.split(".", 1)[1]}']
    spot['Carry_Drift'] = carry_drift
    # Session-print conditioning needs no stamping: the t0 prints ARE state (the restamped
    # ObservedBasis Spots at the trade date), and the engine's `print_seed` protocol
    # conditions every start on the prints its state carries.
    _atomic_write_json(out_path, md)
    return out_path


def _apply_overrides(cfg, args):
    """Generic KEY=VALUE stamps onto the built Solver / Calculation blocks.

    `production_solver.apply_config` rebuilds `Solver` from its own defaults, so a dial that
    has no dedicated flag (DiffV2_Risk_Aversion, DiffV2_Hidden, DiffV2_LR, ...) is otherwise
    unreachable from a job JSON. Values parse as int, then float, else stay strings."""
    def coerce(v):
        for f in (int, float):
            try:
                return f(v)
            except ValueError:
                pass
        return v
    calc = cfg['Calc']['Calculation']
    for kv in (args.solver_set or []):
        k, v = kv.split('=', 1)
        calc['Hedging_Problem']['Solver'][k] = coerce(v)
    for kv in (args.calc_set or []):
        k, v = kv.split('=', 1)
        calc[k] = coerce(v)
    return cfg


def one_trade(template, arch, trade_date, calibrated_md, args, run_dir, tag):
    """Train the day-1 policy per seed in the calibrated chain world, then roll the frozen
    ensemble along the REALIZED archive path. Returns the recorded row."""
    chain_md = calibrated_md     # the month's calibration, before the trade world re-filters it
    calibrated_md = trade_world(arch, trade_date, calibrated_md,
                                os.path.join(run_dir, f'md_{tag}_trade.json'), args.carry_drift)
    cfg, info = build_deal_config(template, arch, trade_date, calibrated_md, args,
                                  delta_corridor=args.delta_corridor)
    if args.drift_overlay is not None:
        # Stamp Mu on the recalibrated spot block (composes with Carry_Drift); rides
        # ExplicitMarketData so train and roll see the same world.
        g = json.load(open(calibrated_md))['MarketData']['Price Models']['GARCHSpotModel.LBMA_AM']
        # The overlay IS the drift hypothesis: Mu replaces the carry drift rather than
        # riding on top of it.
        cfg['Calc']['MergeMarketData']['ExplicitMarketData']['Price Models'][
            'GARCHSpotModel.LBMA_AM'] = dict(g, Mu=args.drift_overlay, Carry_Drift='No')
    # the observed window BEFORE training: a month the gap guard refuses must refuse in seconds,
    # not after a policy whose roll can never run
    obs_npz = os.path.abspath(os.path.join(run_dir, f'obs_{tag}.npz'))
    knots = carry_knots(trade_date, max(max(info['fixings']), max(info['mats'])))
    horizon = (info['pay'] - pd.Timestamp(trade_date)).days + 10
    as_of = observed_scenario_npz(arch, trade_date, obs_npz, horizon, knots, max_gap=args.max_gap,
                                  cols=replayed_cols(cfg), to_date=args.to_date)
    passive_from = None
    if args.vol_switch is not None:
        # the live ledger's spelling: the fix's last ten log returns through the day against the
        # month's calibrated GARCH vol, on every archive day the roll observes
        am = arch[FIX_COL].astype(float)
        h0 = json.load(open(chain_md))['MarketData']['Price Models']['GARCHSpotModel.LBMA_AM']['H0']
        cal = float(np.sqrt(h0 * 252))
        for d in arch.index[(arch.index >= pd.Timestamp(trade_date)) & (arch.index <= as_of)]:
            rv10 = float(np.log(am.loc[:d]).diff().iloc[-10:].std() * np.sqrt(252))
            if rv10 > args.vol_switch * cal:
                passive_from = str(d.date())
                logging.info('VOL-SWITCH %s: trips %s (rv10 %.4f vs %g x cal %.4f)',
                             tag, passive_from, rv10, args.vol_switch, cal)
                break
        else:
            logging.info('VOL-SWITCH %s: no trip', tag)

    ckpts, train_us, v0s, market_dim = [], [], [], None
    u_step_sum = None                      # set by a TRAINED seed; a checkpoint-skip run has none
    for seed in args.seeds:
        ckpt = os.path.abspath(os.path.join(run_dir, f'value_fn_{tag}_s{seed}.pt'))
        ckpts.append(ckpt)
        if os.path.exists(ckpt):
            logging.info('=== TRAIN %s seed=%d SKIP (checkpoint exists) ===', tag, seed)
            train_us.append(None)
            v0s.append(None)
            continue
        train = _apply_overrides(
            apply_config(copy.deepcopy(cfg), batch=args.batch, seed=seed, save=ckpt), args)
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
        if args.position_state:
            tcalc['Hedging_Problem']['Solver']['DiffV2_Position_State'] = 'Yes'
        if args.returns_state:
            tcalc['Hedging_Problem']['Solver']['DiffV2_Returns_State'] = 'Yes'
        if args.wealth_free:
            tcalc['Hedging_Problem']['Solver']['DiffV2_Wealth_Free_Value'] = 'Yes'
        if args.bank_noise is not None:
            tcalc['Hedging_Problem']['Solver']['DiffV2_Bank_Noise_Frac'] = args.bank_noise
        if args.fit_tol is not None:
            tcalc['Hedging_Problem']['Solver']['DiffV2_Fit_Tol'] = args.fit_tol
        if args.action_chunk is not None:
            tcalc['Hedging_Problem']['Solver']['Training_Action_Chunk_Size'] = args.action_chunk
        if args.solver is not None:
            tcalc['Hedging_Problem']['Solver']['Object'] = args.solver
        if args.temporal_proximity is not None:
            tcalc['Hedging_Problem']['Solver']['DiffV2_Temporal_Proximity'] = \
                args.temporal_proximity
        logging.info('=== TRAIN %s seed=%d (fair=%.2f, strike=%.2f, leg bridge premia=%s) ===',
                     tag, seed, info['k_fair'], info['k_fair'] - args.margin, info['premia'])
        tdiag = run(train, f'train_{tag}_s{seed}')
        vd = tdiag.get('verdict') or {}
        verdict = vd.get('greedy') or {}
        # ...and WHICH objective that number is: under --reference-mode Running_Wealth it is a sum
        # of per-step utilities, not a terminal one, so a sweep that mixes the dial would
        # otherwise mix units in one column without saying so.
        u_step_sum = vd.get('u_mean_is_step_sum')
        train_us.append(None if verdict.get('u_mean') is None else round(verdict['u_mean'], 4))
        v0s.append(tdiag.get('V_0'))
        market_dim = tdiag.get('market_dim')

    roll = _apply_overrides(apply_config(
        copy.deepcopy(cfg), batch=1, seed=args.seeds[0], load=ckpts,
        stepper_rollout=True, randomize_initial_state=False), args)
    roll['Calc']['Calculation']['Inner_Sub_Batch'] = args.roll_inner
    if args.load_horizon_pad:
        roll['Calc']['Calculation']['Hedging_Problem']['Solver'][
            'DiffV2_Load_Horizon_Pad'] = 'Yes'
    if args.returns_state:
        roll['Calc']['Calculation']['Hedging_Problem']['Solver'][
            'DiffV2_Returns_State'] = 'Yes'
    if args.drift_beta is not None:
        roll['Calc']['Calculation']['Hedging_Problem']['Solver'][
            'DiffV2_Drift_Beta'] = args.drift_beta
    if passive_from is not None:
        roll['Calc']['Calculation']['Hedging_Problem']['Solver'][
            'DiffV2_Passive_From'] = passive_from
    if args.open_at == 'passive':
        roll['Calc']['Calculation']['Hedging_Problem']['Solver'][
            'DiffV2_First_Decision_From'] = 'Passive'
    if args.grid_levels is not None:
        roll['Calc']['Calculation']['Hedging_Problem']['Solver'][
            'Training_Action_Grid_Levels_Per_Axis'] = args.grid_levels
    if args.solver is not None:
        roll['Calc']['Calculation']['Hedging_Problem']['Solver']['Object'] = args.solver
    if args.position_state:
        # NOT train-only: p is an input column of the fitted value, so a roll under the other
        # setting is refused by the checkpoint's own stamp. --churn-lambda rides along, because
        # under position state it is a TERM of the charge that entered the regressed target, not
        # a training-only shaper: rolling without it would charge kappa|dq| against a value
        # fitted on kappa|dq| + lambda dq^2.
        roll['Calc']['Calculation']['Hedging_Problem']['Solver']['DiffV2_Position_State'] = 'Yes'
        if args.churn_lambda is not None:
            roll['Calc']['Calculation']['Hedging_Problem']['Solver'][
                'DiffV2_Churn_Lambda'] = args.churn_lambda
    if args.wealth_free:
        # Stamped on the roll for the same reason --position-state is: the switch decides which
        # columns the fitted net consumes, so a roll under the other setting loads a function of
        # a different state and the checkpoint's own stamp refuses it.
        roll['Calc']['Calculation']['Hedging_Problem']['Solver'][
            'DiffV2_Wealth_Free_Value'] = 'Yes'
    if args.temporal_proximity is not None:
        roll['Calc']['Calculation']['Hedging_Problem']['Solver'][
            'DiffV2_Temporal_Proximity'] = args.temporal_proximity
    if args.curve_dump:
        # ROLL only, and deliberately: the dump is a picture of a DEPLOYED decision — the wealth
        # the stepper actually carried in, the curve the frozen nets actually ranked — and the
        # training sweep has neither. It is also a single-path diagnostic, which the roll is and
        # the training batches are not.
        roll['Calc']['Calculation']['Hedging_Problem']['Solver'][
            'DiffV2_Decision_Curve_Dump'] = os.path.abspath(
                os.path.join(run_dir, f'curve_{tag}.csv'))
    roll['Calc']['Calculation']['Observed_Scenario'] = obs_npz
    logging.info('=== ROLL %s (stepper, realized path, %d-seed ensemble, inner=%d) ===',
                 tag, len(ckpts), args.roll_inner)
    rdiag = run(roll, f'roll_{tag}')
    json.dump(rdiag, open(os.path.join(run_dir, f'diag_{tag}.json'), 'w'), indent=1, default=str)

    sv = rdiag.get('stepper_verdict') or {}
    greedy = (sv.get('greedy') or {}).get('wT_mean')
    nohedge = (sv.get('nohedge') or {}).get('wT_mean')
    q = np.array(sv.get('greedy_q_traj') or [[0.0]])
    if args.to_date:
        # the live book: the last decision on or before the last archive day, beside the
        # passive's book at that step and the drift monitor's reading there
        dates = pd.to_datetime(sv.get('greedy_q_dates') or [])
        i = int(np.searchsorted(dates, as_of, side='right')) - 1
        dm = sv.get('drift_monitor') or {}
        to_date = {'as_of': str(as_of.date()), 'decision_date': str(dates[i].date()),
                   'book': [round(float(x), 2) for x in q[i]],
                   'passive': [round(float(x), 2) for x in
                               np.array((rdiag.get('verdict') or {}).get('passive_q_traj'))[sv['greedy_q_t'][i] - int(rdiag.get('root_t') or 0)]],
                   'drift_z': (dm.get('z_path') or [None])[min(i, len(dm.get('z_path') or []) - 1)],
                   'tripped': (dm.get('tripped') or [None])[min(i, len(dm.get('tripped') or []) - 1)]}
        logging.info('TO-DATE %s as of %s (decision %s): book %s contracts, passive %s, drift z %s tripped %s',
                     tag, to_date['as_of'], to_date['decision_date'], to_date['book'],
                     to_date['passive'], to_date['drift_z'], to_date['tripped'])
        bound = st_short = None
    else:
        to_date = None
        bound = pf_bound(arch, trade_date, info['mats'], info['pay'], spot_cols=hedge_cols(template))
        st_short = static_short_usd_oz(arch, trade_date, info['mats'], info['pay'], args.volume,
                                       spot_cols=hedge_cols(template))
    return {
        'to_date': to_date, 'passive_from': passive_from, 'vol_switch': args.vol_switch, 'open_at': args.open_at,
        'trade': tag, 'world': 'chain', 'margin': args.margin, 'corridor': args.delta_corridor,
        'fair': round(info['k_fair'], 2), 'strike': round(info['k_fair'] - args.margin, 2),
        'n_seeds': len(args.seeds), 'market_dim': market_dim,
        'train_u': train_us[0], 'train_u_seeds': train_us, 'V_0': v0s[0], 'V_0_seeds': v0s,
        'train_u_is_step_sum': u_step_sum,
        'greedy_usd_oz': None if greedy is None else round(greedy / args.volume, 2),
        'nohedge_usd_oz': None if nohedge is None else round(nohedge / args.volume, 2),
        'static_short_usd_oz': (None if (nohedge is None or st_short is None)
                                else round(nohedge / args.volume + st_short, 2)),
        'pf_bound': None if bound is None else round(bound, 2),
        'bound_pass': (None if (greedy is None or nohedge is None or bound is None)
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
    ap.add_argument('--drift-overlay', type=float, default=None,
                    help='Annualised log-drift stamped as GARCHSpotModel.Mu on top of the carry.')
    ap.add_argument('--drift-beta', type=float, default=None,
                    help='Roll-time Solver.DiffV2_Drift_Beta.')
    ap.add_argument('--no-schedule', action='store_true',
                    help='With --long-cap, leave the flat net schedule unwritten so the policy trains '
                         'corridor-free (the per-leg boxes and the gross limit still bound it) and rolls '
                         'in any corridor.')
    ap.add_argument('--open-at', choices=['passive', 'flat'], default='passive',
                    help="What the first decision is measured from: 'passive' (the swap hedged at "
                         "inception, the sleeve entering through the band) or 'flat' (the empty "
                         "opening book, every roll before 2026-10-29).")
    ap.add_argument('--vol-switch', type=float, default=None, metavar='K',
                    help='The greedy roll hands over to the passive from the first decision day '
                         'whose trailing 10-day realised vol of the fix exceeds K times the '
                         "month's calibrated vol; the trip date is stamped as "
                         'DiffV2_Passive_From and written in the row.')
    ap.add_argument('--returns-state', action='store_true',
                    help='Solver.DiffV2_Returns_State=Yes on train and roll.')
    ap.add_argument('--load-horizon-pad', action='store_true',
                    help='Roll-time DiffV2_Load_Horizon_Pad=Yes: accept checkpoints fitted on a '
                         'different decision horizon (steps clamp to the saved range).')
    ap.add_argument('--max-gap', type=int, default=5,
                    help='Largest hole (days) allowed in the realized window.')
    ap.add_argument('--to-date', action='store_true',
                    help='The live roll: a window past the archive end is padded flat and the '
                         'book on the last archive day is printed beside the passive book (the '
                         'row field to_date); the bound and the static short are not computed.')
    ap.add_argument('--fit-iters', type=int, default=None)
    ap.add_argument('--huber-aversion', type=float, default=None,
                    help='Override the template Objective\'s Huber_Aversion (trains a different '
                         'policy).')
    ap.add_argument('--utility-scale', type=float, default=None,
                    help='Override the template Objective\'s Utility_Scale_Explicit.')
    ap.add_argument('--objective', default=None,
                    help="Override the template Objective's Object (e.g. "
                         "AsymmetricUtility_CARA - trains a different policy).")
    ap.add_argument('--huber-delta', type=float, default=None,
                    help="Objective Huber_Delta (knee, c-units). Large (1e6) = quadratic "
                         "loss wing everywhere, linear gains - the semivariance shape.")
    ap.add_argument('--cara-gamma', type=float, default=None,
                    help="Objective CARA_Gamma (units of c; read by the CARA shape).")
    ap.add_argument('--reference-wealth', type=float, default=None,
                    help="Objective Reference_Wealth in DOLLARS: the benchmark the utility is "
                         "measured against, so 'loss' means underperforming it. Every shape.")
    ap.add_argument('--up-aversion', type=float, default=None,
                    help="Objective Up_Aversion (units of c): curvature of the huber GAIN wing. "
                         "0 = today's exactly-linear gains; >0 curves both sides of the "
                         "reference, deep gains keeping marginal utility 1 - 2*a+*knee.")
    ap.add_argument('--up-knee', type=float, default=None,
                    help="Objective Up_Knee (units of c): knee beyond which the gain wing goes "
                         "linear.")
    ap.add_argument('--reference-mode', choices=['Fixed', 'Running_Wealth'], default=None,
                    help="Objective Reference_Mode. 'Running_Wealth' rewards the DAY's wealth "
                         "increment instead of terminal wealth, so the DP's value is a sum of "
                         "per-step utilities and the reported u_mean is NOT comparable to a "
                         "terminal-utility one (wT is). Applied to TRAIN and ROLL - it decides "
                         "what the labels are, and the checkpoint stamps it. A retrain.")
    ap.add_argument('--utility-scale-mode', choices=['vol_scaled_notional', 'conditional_sim'],
                    default=None,
                    help='Objective Utility_Scale_Mode. conditional_sim measures a '
                         'per-decision-step knee c_t off the warmup batch (the cross-sectional '
                         'dispersion of the flat-book wealth at t, floored) instead of one '
                         'number, so x = (W-R)/c_t is a z-score on every day. It CLEARS the '
                         "template's Utility_Scale_Explicit (the schedule sets the level too, so "
                         'the two are refused together) and RE-DOSES the aversion with it: '
                         'Huber_Aversion is dimensionless in units of c, so re-level c and the '
                         'risk aversion moves - the frame lock logs the per-dollar figure. '
                         'Refuses --utility-scale typed beside it, and --reference-mode '
                         'Running_Wealth. Applied to TRAIN and ROLL; stamped on the checkpoint. '
                         'A retrain.')
    ap.add_argument('--utility-scale-floor', type=float, default=None,
                    help='Objective Utility_Scale_Floor_Frac: floor of the conditional_sim '
                         'schedule as a fraction of its terminal entry, so the early knees (where '
                         'the paths have barely diverged) never collapse below the unhedgeable '
                         'residual. Inert under every other scale mode.')
    ap.add_argument('--max-trade', type=float, default=None,
                    help='Evaluator Max_Trade_Per_Step: per-leg |dq| cap per decision step '
                         'at the argmax (execution only; checkpoints re-rollable under it).')
    ap.add_argument('--deadband', type=float, default=None,
                    help='Evaluator Decision_Deadband_Sigma: standard errors of the paired '
                         'inner-draw difference the argmax must beat the incumbent by to trade '
                         '(execution only; checkpoints re-rollable under it).')
    ap.add_argument('--long-cap', type=float, default=None,
                    help='Open the NET range to [-60, +long_cap] (flat Total_Position_Schedule '
                         '+ per-leg boxes). Changes the action space: a RETRAIN, never a '
                         're-roll. Mutually exclusive with --delta-corridor.')
    ap.add_argument('--solver', default=None,
                    help='Solver Object override (default: the template/BEST_SOLVER DiffSolver).')
    ap.add_argument('--temporal-proximity', type=float, default=None,
                    help='Solver DiffV2_Temporal_Proximity: successor-proximity weight.')
    ap.add_argument('--churn-lambda', type=float, default=None,
                    help='Solver DiffV2_Churn_Lambda. TRAIN only on its own: a checkpoint-reuse '
                         'run must not silently turn a training knob into an execution one. '
                         'WITH --position-state it is stamped on the roll too, because there it '
                         'is a term of the charge the fitted target already paid.')
    ap.add_argument('--position-state', action='store_true',
                    help='Solver DiffV2_Position_State: the frictional Bellman (net book '
                         'fraction as state, the charge inside the fitted target). Applied to '
                         'TRAIN and ROLL - the checkpoint stamps it and a mismatched load is '
                         'refused. Changes the value function: a retrain.')
    ap.add_argument('--wealth-free', action='store_true',
                    help='Solver DiffV2_Wealth_Free_Value: drop the wealth column from the '
                         'value net, so the residual reads market (and, with --position-state, '
                         'the book) alone and the ranking bends in wealth only as the utility '
                         'does. Applied to TRAIN and ROLL - the column is part of the fitted '
                         'function and the checkpoint stamps it. Changes the value function: '
                         'a retrain.')
    ap.add_argument('--bank-noise', type=float, default=None,
                    help='Solver DiffV2_Bank_Noise_Frac, TRAIN only: exploration around the '
                         'per-t replication hedge, as a fraction of each [Min,Max] range. It '
                         'is what makes the position-state slice identifiable.')
    ap.add_argument('--fit-tol', type=float, default=None,
                    help='Solver DiffV2_Fit_Tol override (0 pins the full fit budget).')
    ap.add_argument('--avg-lead-months', type=int, default=3,
                    help='The averaging month, this many months after the trade month (0 = the trade month).')
    ap.add_argument('--avg-days', type=int, default=0,
                    help='Only the last N fixings of the averaging month (0 = the whole month).')
    ap.add_argument('--action-chunk', type=int, default=None,
                    help='Solver Training_Action_Chunk_Size: grid rows scored per pass of the argmax. '
                         'Bit-identical at any value; bounds the peak memory of a decision.')
    ap.add_argument('--grid-levels', type=int, default=None,
                    help='Override Solver Training_Action_Grid_Levels_Per_Axis (train AND '
                         'roll argmax). Changes the action space: a retrain. WITH --allocation '
                         'the levels count rungs on the NET cover and must span it at one '
                         'contract a rung or the engine refuses: 51 over [-50, 0], but 71 with '
                         '--long-cap 10, and the template default of 9 is a 7-contract ladder.')
    ap.add_argument('--allocation', action='store_true',
                    help='Evaluator Allocation_Weights: FACTOR the action space. The deck '
                         'computes the remaining-fixings exposure share of each leg per decision '
                         'day (see allocation_knots) and the argmax then searches one ladder over '
                         'the net cover. Applied to TRAIN and ROLL - it is the action space, so a '
                         'retrain, never a re-roll. Needs --grid-levels (see above). Note the '
                         'static-hold benchmark is then chosen from the schedule\'s compositions '
                         'rather than the whole product, and Max_Trade_Per_Step becomes a band on '
                         'the NET rather than per leg.')
    ap.add_argument('--calendar-spread-bps', type=float, default=None,
                    help='Evaluator Calendar_Spread_Bps: half-spread bps the MATCHED half of a '
                         'move pays instead of two outright crossings. One key, both readers - '
                         'the argmax/fitted-target charge and the realized roll rebate.')
    ap.add_argument('--allocation-mode', default=None,
                    choices=['exposure', 'carry_variance'],
                    help='Evaluator.Allocation_Mode: carry_variance derives the leg split in the solver (use INSTEAD of --allocation).')
    ap.add_argument('--delta-corridor', type=float, default=None,
                    help='Causal delta-ramp corridor band on the SIGNED total position, applied '
                         'to BOTH train and roll.')
    ap.add_argument('--solver-set', action='append', metavar='KEY=VALUE',
                    help='Stamp an arbitrary Hedging_Problem.Solver field (repeatable).')
    ap.add_argument('--calc-set', action='append', metavar='KEY=VALUE',
                    help='Stamp an arbitrary Calculation field (repeatable).')
    ap.add_argument('--curve-dump', action='store_true',
                    help='Solver DiffV2_Decision_Curve_Dump on the ROLL: write '
                         '<run-dir>/curve_<trade>.csv with, per decision day, the FULL per-rung '
                         'ranking curve (pre-charge and charged), where the day sits on the '
                         'utility (x0, the knee, the local risk aversion), the band '
                         'probabilities, the fork moments and a second-fork disagreement flag. '
                         'Pure diagnostic: no decision and no reported number moves.')
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

    rows, calibrated_md, fixing = [], None, fixing_day(template)
    for m in range(args.months):
        # the first London fixing day of the month: the strike's AM print is a day the hedge trades
        trade_date = fixing.rollforward(pd.Timestamp(args.start + '-01') + pd.offsets.MonthBegin(m))
        tag = trade_date.strftime('%Y%m')
        try:
            if m % args.recal_months == 0:
                calibrated_md = os.path.abspath(os.path.join(run_dir, f'md_{tag}_chain.json'))
                cal_end = trade_date.strftime('%Y-%m-%d')
                cal_start = (trade_date - pd.DateOffset(years=args.cal_years)).strftime('%Y-%m-%d')
                # per-family windows (the window-study rulings): the basis/chain/fix/carry
                # families fit on the rolling short window; the GARCH shape wants the LONG
                # one (a 3y fit reads nu ~ 23, near-Gaussian tails), so the spot block is
                # spliced in from a long-window pass. An existing spliced file is the cache.
                if not os.path.exists(calibrated_md):
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
                        arch, cal_end, taus, mats=mats_g, hedge_cols=hedge_cols(template),
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
                   'train_u_seeds': None, 'train_u_is_step_sum': None, 'V_0': None,
                   'greedy_usd_oz': None,
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
