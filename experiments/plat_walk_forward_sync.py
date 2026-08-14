"""Walk-forward backtest of the platinum hedge in the SYNCHRONIZED world — the sibling of
`production_walk_forward.py`, run on the same window with the same trades and seeds so the two
decks can be read against each other.

WHAT CHANGES, and nothing else does. The solver, the evaluator, the objective, the corridor, the
accounting and the roll are `production_walk_forward`'s — imported from it rather than copied, so a
change there reaches both decks. Four things are new, and they are the increments E1-E4 shipped:

  * BASIS. `ObservedBasis.PLATINUM_CME.LBMA` is the INTRADAY-SYNCHRONIZED basis (b = LBMA fix -
    CME at the same instant), carrying both `BasisLinkedSpotModel` extensions: the slow observable
    mean the AR reverts to, and the basis's own GARCH innovation variance.
  * CARRY. `ForwardRate.PLATINUM_CARRY` is a continuous curve driven by `QuadraticCarryCurveModel`
    - two knots of the AVERAGE carry z(tau) = c + a*tau - instead of three listed-tenor VAR slots.
  * LIABILITY. `CommodityAveragePriceSwapDeal`: the average-price swap priced CLOSED FORM on the
    path, replacing the `FloatingEnergyDeal` leg the old deck marks through the Components curve.
  * THE CARRY IS TOTAL. Per the handover's section 17, z is extracted from the intraday futures
    curve alone and already embeds financing - so the primary's repo curve is a ZERO curve and
    F(t,T) = S(t)*exp(z(tau)*tau) with nothing added on top. The old deck's carry is NET of SOFR
    and its futures add the simulated SOFR back; both spell the same forward, from different data.

WHAT IS DELIBERATELY THE SAME, so the comparison is a comparison:

  * the trade calendar, the margin, the volume, the seeds, the batch/stream shape, the corridor;
  * the substituted realized path is (CME primary, published basis) in BOTH decks. The sync
    archive also carries the realized CARRY, and replaying it would be a strictly better backtest
    - it is left simulated here because the old deck cannot do it and a one-sided improvement is
    not a model comparison. That is the first follow-on.
  * SOFR. The old deck simulates it (`PCAInterestRateModel`); this world does NOT - the sync
    archive carries no rates, and the financing risk that matters is inside the TOTAL carry, which
    IS simulated. SOFR is read per trade date from `data/plat_archive.csv` and held static for
    discounting and cash financing. This is the one modelling difference that is not a deliberate
    increment; it is worth a fraction of a basis point of a three-month discount factor.

E[dF|state] INTEGRITY. The old deck's guard regresses the primary's next change on the basis.
This one runs it PER TRADABLE and against every state coordinate the world simulates - basis,
carry level, carry shape - because a hedge whose expected one-step return is non-zero conditional
on a state the policy can see is exactly the unexecutable reversion the basis saga was about.

Usage (PYTHONPATH must carry the repo - derivus is not installed):
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python experiments/plat_walk_forward_sync.py \\
        --start 2020-01 --months 1 --seeds 7 --batch 512 --batches 5 --fit-iters 40

    # the authored world + a runnable job JSON, no solve:
    PYTHONPATH=. python experiments/plat_walk_forward_sync.py --emit-world data/plat_world_2026

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
from production_walk_forward import (CONTRACT_SIZE, EXCEL, OBJECTIVE, ROOT, _atomic_write_csv,
                                     _atomic_write_json, _ts, calibrate,
                                     delta_corridor_schedule)

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s %(message)s')

DAYS_IN_YEAR = 365.25
CME_COL = 'CommodityPrice.PLATINUM_CME'                 # martingale primary P
BASIS_COL = 'ObservedBasis.PLATINUM_CME.LBMA'           # synchronized basis b = LBMA - CME
CARRY_COLS = ('ForwardRate.PLATINUM_CARRY,0.5', 'ForwardRate.PLATINUM_CARRY,1.0')
REF_TENORS = (0.5, 1.0)                                 # the sub-keys ARE the reference tenors
SOFR_PREFIX = 'InterestRate.USD-SOFR'
ZERO_CURVE = 'USD-ZERO'                                 # the primary's repo: the carry is total


# ---------------------------------------------------------------------------------------------
# the archive: synchronized series + the rate curve they were never extracted with
# ---------------------------------------------------------------------------------------------
def load_archive(sync_csv, rates_csv):
    """The synchronized archive (P, b, z(0.5), z(1.0)) with the EOD SOFR curve joined on date.

    The rate file is a SECOND observation clock and is deliberately NOT part of the curve
    extraction (handover section 17); it is joined here only because discounting and the causal
    bound need a rate, and because the shipped deck reads its SOFR from the same source."""
    arch = pd.read_csv(sync_csv, index_col=0, parse_dates=True)
    rates = pd.read_csv(rates_csv, index_col=0, parse_dates=True)
    sofr = [c for c in rates.columns if c.startswith(SOFR_PREFIX)]
    return arch.join(rates[sofr].reindex(arch.index).ffill(), how='left')


def carry_state(row):
    """`(L, D)` - the carry LEVEL and SHAPE at the two reference tenors, which is what the two
    archive columns' mean and difference ARE (`QuadraticCarryCurveModel`'s own rotation)."""
    z_a, z_b = (float(row[c]) for c in CARRY_COLS)
    return 0.5 * (z_a + z_b), z_b - z_a


def z_of(state, tau):
    """Average carry to maturity at `tau`, affine in tau: z = L + D (tau - tau_bar)/dtau."""
    L, D = state
    tau_a, tau_b = REF_TENORS
    return L + D * (np.asarray(tau) - 0.5 * (tau_a + tau_b)) / (tau_b - tau_a)


def forward(p0, state, tau):
    """F(t, t+tau) = S exp(z(tau) tau). The carry is TOTAL, so nothing is added for financing."""
    return p0 * np.exp(z_of(state, tau) * np.asarray(tau))


def sofr_curve(row, columns):
    """`[[tenor_years, rate], ...]` off one archive row, in the shape a `.Curve` carries."""
    return [[t, float(row[c])] for t, c in
            sorted((float(c.split(',')[1]), c) for c in columns if c.startswith(SOFR_PREFIX))]


def listed_expiries(expiry_csv, trade_date, n=3):
    """The next `n` LISTED contract expiries after the trade date - the real quarterly ladder,
    read rather than inferred (the handover's section 9: exact expiries beat inferred ones)."""
    exp = pd.read_csv(expiry_csv, parse_dates=['Expiry_Date'])
    live = exp.loc[exp['Expiry_Date'] > pd.Timestamp(trade_date), 'Expiry_Date'].sort_values()
    if len(live) < n:
        raise ValueError(f'{expiry_csv} lists {len(live)} expiries after {trade_date}, need {n}')
    return [pd.Timestamp(d) for d in live.iloc[:n]]


# ---------------------------------------------------------------------------------------------
# the guard: E[dF | state] on every tradable, against every simulated state coordinate
# ---------------------------------------------------------------------------------------------
def guard_e_df_state(arch, cal_end, taus):
    """`{leg: {state: (slope, t)}}` - the one-step change of each tradable's reconstructed forward
    regressed on each state coordinate the world simulates, on data up to `cal_end`.

    A tradable references the martingale PRIMARY, so every slope must be statistically zero: a
    non-zero one is reversion the policy can see and cannot execute, which is the failure the
    basis saga was. `dS~b` on the composed LBMA leg is the control - the catch-up lives there, by
    construction, and it is NOT tradable."""
    sub = arch.loc[:cal_end]
    state = pd.DataFrame({'b': sub[BASIS_COL],
                          'L': 0.5 * (sub[CARRY_COLS[0]] + sub[CARRY_COLS[1]]),
                          'D': sub[CARRY_COLS[1]] - sub[CARRY_COLS[0]]})
    legs = {f'F({tau:.2f}y)': forward(sub[CME_COL].to_numpy(),
                                      (state['L'].to_numpy(), state['D'].to_numpy()), tau)
            for tau in taus}
    legs['S_LBMA (control)'] = (sub[CME_COL] + sub[BASIS_COL]).to_numpy()
    out = {}
    for leg, series in legs.items():
        y = pd.Series(series, index=sub.index).diff().shift(-1)
        out[leg] = {}
        for name, x in state.items():
            d = pd.DataFrame({'y': y, 'x': x}).dropna()
            X = np.column_stack([np.ones(len(d)), d['x'].values])
            coef, *_ = np.linalg.lstsq(X, d['y'].values, rcond=None)
            resid = d['y'].values - X @ coef
            se = np.sqrt((resid ** 2).sum() / (len(d) - 2)
                         / ((d['x'] - d['x'].mean()) ** 2).sum())
            out[leg][name] = (float(coef[1]), float(coef[1] / se) if se > 0 else float('nan'))
    return out


# ---------------------------------------------------------------------------------------------
# the world and the job
# ---------------------------------------------------------------------------------------------
def price_factors(row, trade_date, last_query, sofr_cols):
    """The complete `Price Factors` block off ONE archive row.

    THE CARRY KNOTS ARE DATES and `CurveTenor` CLIPS a query to their bracket, so the rule
    `QuadraticCarryCurveModel` states is honoured here and nowhere else: the first knot AT the base
    date, the last at or after the longest query the book makes (the later of the last fixing and
    the last tradable expiry). Between them the linear interpolation in date IS the affine z, so
    two knots reproduce the whole curve exactly."""
    p0, state = float(row[CME_COL]), carry_state(row)
    knots = [pd.Timestamp(trade_date), pd.Timestamp(last_query) + pd.Timedelta(days=30)]
    return {
        'FxRate.USD': {'Domestic_Currency': '', 'Interest_Rate': 'USD-SOFR', 'Spot': 1.0},
        CME_COL: {'Currency': 'USD', 'Interest_Rate': ZERO_CURVE,
                  'Forward_Rate': 'PLATINUM_CARRY', 'Spot': p0, 'Property_Aliases': ''},
        BASIS_COL: {'Spot': float(row[BASIS_COL])},
        'ForwardRate.PLATINUM_CARRY': {'Currency': 'USD', 'Curve': {'.Curve': {'meta': [], 'data': [
            [float((k - EXCEL).days), float(z_of(state, (k - pd.Timestamp(trade_date)).days / DAYS_IN_YEAR))]
            for k in knots]}}},
        'InterestRate.USD-SOFR': {'Day_Count': 'ACT_365', 'Currency': 'USD', 'Sub_Type': None,
                                  'Curve': {'.Curve': {'meta': [], 'data': sofr_curve(row, sofr_cols)}}},
        # The primary's repo. z already carries financing, so adding a rate under the same
        # exponential would count it twice; zero is the statement, not a placeholder. THREE knots,
        # because the world interpolates rate curves with Hermite and Hermite needs three.
        f'InterestRate.{ZERO_CURVE}': {'Day_Count': 'ACT_365', 'Currency': 'USD', 'Sub_Type': None,
                                       'Curve': {'.Curve': {'meta': [], 'data': [
                                           [0.0, 0.0], [1.0, 0.0], [30.0, 0.0]]}}},
    }


def fair_strike(row, trade_date, fixings, basis_model):
    """The average-price swap's own closed form at inception, written from the deal's documented
    formula: `sum_j w_j (P exp(z tau_j) tau_j + mu + phi^n_j (b - mu))`, equal weights.

    `n_j` is SIMULATION STEPS, and the grid is `0d 1d(1d)` - one row per calendar day - so it is
    the day count. Reading it in years, or from the calibration clock, prices a different model
    from the one that gets simulated."""
    p0, b0, state = float(row[CME_COL]), float(row[BASIS_COL]), carry_state(row)
    mu = float(basis_model.get('Mu_0', 0.0)) if basis_model.get('Slow_Mean_Lambda') else 0.0
    phi = float(basis_model['Phi'])
    days = np.array([(pd.Timestamp(f) - pd.Timestamp(trade_date)).days for f in fixings], dtype=float)
    return float((forward(p0, state, days / DAYS_IN_YEAR) + mu + phi ** days * (b0 - mu)).mean())


def build_deal_config(template, arch, trade_date, calibrated_md, args, delta_corridor=None):
    """The trade-date job: the APS liability struck at fair - margin, the listed CME strip as
    tradables, and every market level read off the archive row. Returns `(cfg, info)`."""
    row = arch.loc[:trade_date].iloc[-1]
    cfg = copy.deepcopy(template)
    calc = cfg['Calc']['Calculation']
    merge = cfg['Calc']['MergeMarketData']
    hp = calc['Hedging_Problem']
    calc['Base_Date'] = _ts(trade_date)
    merge['MarketDataFile'] = calibrated_md
    hp['Objective'] = dict(OBJECTIVE)

    p0, state = float(row[CME_COL]), carry_state(row)
    basis_model = json.load(open(calibrated_md))['MarketData']['Price Models'][
        f'BasisLinkedSpotModel.{BASIS_COL.split(".", 1)[1]}']

    # --- liability: 3-month-out averaging month on the LBMA fixing, paid +5 days ---------------
    avg_start = (pd.Timestamp(trade_date) + pd.offsets.MonthBegin(3)).normalize()
    avg_end = (avg_start + pd.offsets.MonthEnd(0)).normalize()
    pay = avg_end + pd.Timedelta(days=5)
    fixings = pd.bdate_range(avg_start, avg_end)
    k_fair = fair_strike(row, trade_date, fixings, basis_model)
    hp['Liabilities'] = {'CommodityAveragePriceSwapDeal': {'PLAT_APS': {
        'Currency': 'USD', 'Commodity': f'{CME_COL.split(".")[1]}.{BASIS_COL.rsplit(".", 1)[1]}',
        'Carry': 'PLATINUM_CARRY', 'Discount_Rate': 'USD-SOFR', 'Buy_Sell': 'Buy',
        'Units': args.volume, 'Fixed_Price': round(k_fair - args.margin, 6),
        'Settlement_Date': _ts(pay),
        'Sampling_Data': [[_ts(f), 0.0, 1.0] for f in fixings]}}}

    # --- tradables: the listed CME ladder, each referencing the martingale primary -------------
    mats = listed_expiries(args.expiries, trade_date)
    futs, positions, setts, margins, limits = {}, {}, {}, {}, {}
    for i, mat in enumerate(mats, 1):
        name = f'PL_M{i}'
        futs[name] = {'Maturity_Date': _ts(mat), 'Currency': 'USD', 'Carry': 'PLATINUM_CARRY',
                      'Repo_Rate': ZERO_CURVE, 'Commodity': CME_COL.split('.')[1],
                      'Contract_Size': CONTRACT_SIZE}
        positions[name] = 0
        setts[name] = round(float(forward(p0, state, (mat - pd.Timestamp(trade_date)).days / DAYS_IN_YEAR)), 4)
        margins[name] = {'Method': 'per_contract', 'Amount': round(0.085 * setts[name] * CONTRACT_SIZE, 0)}
        limits[name] = {'Min_Position': -50, 'Max_Position': 0}
    hp['Tradable_Instruments']['CommodityFutureDeal'] = futs
    hp['Tradable_Instruments']['CashAccountDeal']['USD_CASH']['Investment_Horizon'] = _ts(pay)
    state_block = hp['Portfolio_State']
    state_block['Positions'], state_block['Settlement_Prices'] = positions, setts
    state_block['Initial_Margin'] = margins
    hp['Evaluator']['Position_Limits'] = limits
    if delta_corridor is not None:
        hp['Evaluator']['Total_Position_Schedule'] = delta_corridor_schedule(
            trade_date, fixings, delta_corridor)

    # --- realized primary spot history, STRICTLY before the trade date -------------------------
    hist = arch.loc[arch.index < trade_date].iloc[-35:]
    state_block['Spot_Price_History'] = {CME_COL: {
        'Dates': [_ts(d) for d in hist.index], 'Prices': [float(x) for x in hist[CME_COL]]}}

    merge['ExplicitMarketData']['Price Factors'] = price_factors(
        row, trade_date, max(max(fixings), max(mats)), arch.columns)
    merge['ExplicitMarketData'].setdefault('Price Models', {})
    merge['ExplicitMarketData'].pop('Valuation Configuration', None)   # the APS reads no forward curve
    return cfg, {'k_fair': k_fair, 'mats': mats, 'pay': pay, 'fixings': fixings}


def resolved_calibration_config(source, archive, out_path):
    """The calibration config with an ABSOLUTE archive path, written to `out_path`.

    `calibrate` runs the calibrator as a subprocess from `experiments/`, so a config carrying a
    repo-relative archive name would resolve it against the wrong directory."""
    cfg = json.load(open(source))
    cfg['CalibrationConfig']['MarketDataArchiveFile']['name'] = os.path.abspath(archive)
    out_path = os.path.abspath(out_path)
    _atomic_write_json(out_path, cfg)
    return out_path


def observed_scenario_npz(arch, base_date, path, max_gap=5):
    """Dense daily realized (CME primary, published basis) from the base date, forward-filled.
    The framework composes the LBMA fixing S = P + b itself. The realized CARRY is available in
    this archive and deliberately not substituted - see the module docstring.

    TWO gap guards, and the second one is the whole reason this function has a parameter. Running
    past the archive END fabricates flat prices, which the old driver already refuses; a hole in
    the MIDDLE fabricates exactly the same flat prices and nothing refused it. The synchronized
    archive is built from intraday ticks that pass a quality filter, so it HAS holes - 78 gaps
    over four days, the largest 29 - and the biggest one covers the whole first half of April
    2020, which is the averaging month of the 2020-01 trade the smoke gate anchors on. Measured
    there: the realized average comes out $26.02/oz below the EOD archive's, i.e. the roll marks a
    crash that the forward fill flattened. `--max-gap 999` to run anyway and own the number."""
    base = pd.Timestamp(base_date)
    dates = pd.DatetimeIndex([base + pd.Timedelta(days=i) for i in range(220)])
    if dates.max() > arch.index[-1]:
        raise ValueError(
            f'Observed window {base.date()}+220d ends {dates.max().date()} past archive end '
            f'{arch.index[-1].date()}: the realized roll would run on fabricated flat prices')
    window = pd.Series(arch.index[(arch.index >= base) & (arch.index <= dates.max())])
    gap = int(max(window.diff().dt.days.max(), (window.iloc[0] - base).days))
    if gap > max_gap:
        raise ValueError(
            f'Observed window {base.date()}+220d has a {gap}-day hole in the archive (max_gap='
            f'{max_gap}): the roll would replay forward-filled flat prices through it')
    rows = arch.reindex(arch.index.union(dates)).ffill().loc[dates]
    np.savez(path, **{CME_COL: rows[CME_COL].to_numpy(), BASIS_COL: rows[BASIS_COL].to_numpy()})


def pf_bound(arch, trade_date, mats, pay):
    """Portfolio causal bound, sum_t max_leg max(0, -dF_obs), on the reconstructed observed CME
    strip ($/oz): a hedge can only lose versus no-hedge by the worst adverse forward move it could
    be caught in, summed over days."""
    bdays = pd.bdate_range(trade_date, pay)
    if bdays.max() > arch.index[-1]:
        raise ValueError(
            f'Bound window ends {bdays.max().date()} past archive end {arch.index[-1].date()}')
    sub = arch.reindex(arch.index.union(bdays)).ffill().loc[bdays]
    state = (0.5 * (sub[CARRY_COLS[0]] + sub[CARRY_COLS[1]]).to_numpy(),
             (sub[CARRY_COLS[1]] - sub[CARRY_COLS[0]]).to_numpy())
    legs = []
    for mat in mats:
        tau = np.array([(mat - d).days for d in bdays]) / DAYS_IN_YEAR
        legs.append(np.where(tau > 0, forward(sub[CME_COL].to_numpy(), state, np.clip(tau, 0, None)), np.nan))
    dF = np.diff(np.array(legs), axis=1)
    worst = np.nanmax(np.where(np.isnan(dF), -np.inf, np.maximum(0.0, -dF)), axis=0)
    return float(np.where(np.isfinite(worst), worst, 0.0).sum())


# ---------------------------------------------------------------------------------------------
# one trade
# ---------------------------------------------------------------------------------------------
def one_trade(template, arch, trade_date, calibrated_md, args, run_dir, tag):
    """Train the day-1 policy per seed, then roll the frozen ensemble on the realized path.
    Returns one row per corridor band rolled (one row when there is no sweep).

    THE CORRIDOR IS A ROLL-TIME SCHEDULE. `--corridor-sweep` trains ONCE, corridor-free, and rolls
    the SAME checkpoint under each band: the shipped operating point is a corridor-free policy with
    the schedule applied at roll time, and holding the policy fixed is also what makes three bands
    a comparison of the corridor rather than of three training runs.

    The old deck's stale-calibration retry has no counterpart here BY CONSTRUCTION: a VAR carry
    slot could expire between the calibration and trade dates (tau -> 0), and a continuous curve
    has no slots to expire."""
    bands = args.corridor_sweep or [args.delta_corridor]
    cfg, info = build_deal_config(template, arch, trade_date, calibrated_md, args,
                                  delta_corridor=None if args.corridor_sweep else args.delta_corridor)
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
        if args.fit_iters is not None:
            train['Calc']['Calculation']['Hedging_Problem']['Solver']['DiffV2_Fit_Iters'] = args.fit_iters
        train['Calc']['Calculation']['Simulation_Batches'] = args.batches
        logging.info('=== TRAIN %s seed=%d (fair=%.2f, strike=%.2f) ===',
                     tag, seed, info['k_fair'], info['k_fair'] - args.margin)
        tdiag = run(train, f'train_{tag}_s{seed}')
        verdict = (tdiag.get('verdict') or {}).get('greedy') or {}
        train_us.append(None if verdict.get('u_mean') is None else round(verdict['u_mean'], 4))
        v0s.append(tdiag.get('V_0'))
        market_dim = tdiag.get('market_dim')

    obs_npz = os.path.abspath(os.path.join(run_dir, f'obs_{tag}.npz'))
    observed_scenario_npz(arch, trade_date, obs_npz, max_gap=args.max_gap)
    bound = pf_bound(arch, trade_date, info['mats'], info['pay'])
    rows = []
    for band in bands:
        band_cfg = cfg if band == (None if args.corridor_sweep else args.delta_corridor) else \
            build_deal_config(template, arch, trade_date, calibrated_md, args,
                              delta_corridor=band)[0]
        roll = apply_config(copy.deepcopy(band_cfg), batch=1, seed=args.seeds[0], load=ckpts,
                            stepper_rollout=True, randomize_initial_state=False)
        roll['Calc']['Calculation']['Inner_Sub_Batch'] = args.roll_inner
        roll['Calc']['Calculation']['Observed_Scenario'] = obs_npz
        label = f'{tag}' if band is None else f'{tag}_c{band}'
        logging.info('=== ROLL %s (stepper, realized path, %d-seed ensemble, inner=%d, '
                     'corridor=%s) ===', tag, len(ckpts), args.roll_inner, band)
        rdiag = run(roll, f'roll_{label}')
        json.dump(rdiag, open(os.path.join(run_dir, f'diag_{label}.json'), 'w'), indent=1, default=str)

        sv = rdiag.get('stepper_verdict') or {}
        greedy = (sv.get('greedy') or {}).get('wT_mean')
        nohedge = (sv.get('nohedge') or {}).get('wT_mean')
        q = np.array(sv.get('greedy_q_traj') or [[0.0]])
        rows.append({
            'trade': tag, 'world': 'sync', 'corridor': band,
            'fair': round(info['k_fair'], 2), 'strike': round(info['k_fair'] - args.margin, 2),
            'n_seeds': len(args.seeds), 'market_dim': market_dim,
            'train_u': train_us[0], 'train_u_seeds': train_us, 'V_0': v0s[0],
            'greedy_usd_oz': None if greedy is None else round(greedy / args.volume, 2),
            'nohedge_usd_oz': None if nohedge is None else round(nohedge / args.volume, 2),
            'pf_bound': round(bound, 2),
            'bound_pass': (None if (greedy is None or nohedge is None)
                           else bool(greedy / args.volume <= nohedge / args.volume + bound + 1e-6)),
            'churn': round(float(np.abs(np.diff(q, axis=0)).sum()), 1),
        })
    return rows


# ---------------------------------------------------------------------------------------------
# the authored world (data only)
# ---------------------------------------------------------------------------------------------
def emit_world(arch, template, out_dir, args):
    """The self-contained world at the archive's LAST row, plus a runnable job on it.

    The market data is the calibrated one this script's own `calibrate` step writes, with the
    Price Factors of that row folded in; the job is the walk-forward's own trade shape frozen at
    that date. Both are DATA - regenerable from this function, the archive and the calibration
    config - which is why they live under `data/` and not in the repo."""
    os.makedirs(out_dir, exist_ok=True)
    trade_date = arch.index[-1]
    md_path = os.path.abspath(os.path.join(out_dir, 'MarketDataRF_platinum_sync.json'))
    calibrate(os.path.abspath(args.marketdata),
              resolved_calibration_config(args.calibration_config, args.archive,
                                          os.path.join(out_dir, 'calibration_config_run.json')),
              trade_date.strftime('%Y-%m-%d'), md_path)
    cfg, info = build_deal_config(template, arch, trade_date, md_path, args,
                                  delta_corridor=args.delta_corridor)
    # fold the row's Price Factors into the calibrated file: the world becomes self-contained
    md = json.load(open(md_path))
    md['MarketData']['Price Factors'] = cfg['Calc']['MergeMarketData']['ExplicitMarketData']['Price Factors']
    md['MarketData']['System Parameters']['Base_Date'] = _ts(trade_date)
    _atomic_write_json(md_path, md)
    # the job reads the world by NAME and carries no market data of its own
    cfg['Calc']['MergeMarketData']['ExplicitMarketData'].pop('Price Factors')
    cfg['Calc']['MergeMarketData']['MarketDataFile'] = './' + os.path.relpath(
        md_path, os.path.dirname(ROOT))
    cfg['Calc']['Calculation']['Execution_Mode'] = 'simulate_only'
    job = os.path.join(out_dir, 'job_aps_hedge_sync.json')
    _atomic_write_json(job, cfg)
    print(f'world  {md_path}')
    print(f'job    {job}  (fair {info["k_fair"]:.2f}, strike {info["k_fair"] - args.margin:.2f}, '
          f'settle {info["pay"].date()}, {len(info["fixings"])} fixings, '
          f'expiries {[str(m.date()) for m in info["mats"]]})')


# ---------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--archive', default='data/plat_archive_sync.csv',
                    help='Synchronized archive: P, b, z(0.5), z(1.0).')
    ap.add_argument('--rates', default='data/plat_archive.csv',
                    help='Rate archive the SOFR curve is read from (a second, EOD clock).')
    ap.add_argument('--expiries', default='data/platinum_futures_contract_expiry.csv')
    ap.add_argument('--calibration-config', default='data/plat_world_2026/calibration_config_sync.json')
    ap.add_argument('--marketdata', default='data/plat_world_2026/MarketDataRF_sync_source.json')
    ap.add_argument('--deal-template', default='tests/fixtures/policy_test_simulate_only.json')
    ap.add_argument('--start', default='2020-01', help='First trade month, YYYY-MM.')
    ap.add_argument('--months', type=int, default=12)
    ap.add_argument('--recal-months', type=int, default=3)
    ap.add_argument('--margin', type=float, default=8.0)
    ap.add_argument('--volume', type=float, default=2500.0)
    ap.add_argument('--batch', type=int, default=8192)
    ap.add_argument('--batches', type=int, default=5)
    ap.add_argument('--seeds', type=int, nargs='+', default=[7])
    ap.add_argument('--roll-inner', type=int, default=256)
    ap.add_argument('--max-gap', type=int, default=5,
                    help='Largest hole (days) allowed in the realized window - see '
                         'observed_scenario_npz. 999 runs on forward-filled prices.')
    ap.add_argument('--fit-iters', type=int, default=None)
    ap.add_argument('--delta-corridor', type=float, default=None,
                    help='Causal delta-ramp corridor band on the SIGNED total position, applied '
                         'to BOTH train and roll (the old deck\'s behaviour).')
    ap.add_argument('--corridor-sweep', type=float, nargs='+', default=None,
                    help='Bands to ROLL the same corridor-free checkpoint under, one row each - '
                         'the shipped operating point (corridor-free policy, roll-time schedule). '
                         'Overrides --delta-corridor.')
    ap.add_argument('--run-dir', default=None)
    ap.add_argument('--emit-world', default=None,
                    help='Write the authored world + job at the archive end and exit (no solve).')
    args = ap.parse_args()

    arch = load_archive(args.archive, args.rates)
    template = json.load(open(args.deal_template))
    if args.emit_world:
        emit_world(arch, template, args.emit_world, args)
        return

    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = args.run_dir or os.path.join(
        'artifacts', 'walk_forward_sync', f'{stamp}_{args.start}_{args.months}m')
    os.makedirs(run_dir, exist_ok=True)
    logging.info('run dir: %s  seeds: %s  corridor: %s', run_dir, args.seeds, args.delta_corridor)

    # the calibration reads ITS archive from the calibration config; write the joined one so the
    # rate curve and the synchronized series are one file and the fit is reproducible
    arch_csv = os.path.abspath(os.path.join(run_dir, 'archive_sync.csv'))
    arch.to_csv(arch_csv)
    cal_cfg = resolved_calibration_config(args.calibration_config, arch_csv,
                                          os.path.join(run_dir, 'calibration_config.json'))

    done = {}
    for f in sorted(glob.glob(os.path.join(run_dir, 'row_*.json'))):
        r = json.load(open(f))
        done[r[0]['trade']] = r
    if done:
        logging.info('RESUME: %d completed trade(s) found in run dir; will skip them', len(done))

    rows, calibrated_md = [], None
    for m in range(args.months):
        trade_date = (pd.Timestamp(args.start + '-01') + pd.offsets.MonthBegin(m)
                      + pd.offsets.BDay(0)).normalize()
        tag = trade_date.strftime('%Y%m')
        try:
            if m % args.recal_months == 0:
                calibrated_md = os.path.abspath(os.path.join(run_dir, f'md_{tag}_sync.json'))
                cal_end = trade_date.strftime('%Y-%m-%d')
                calibrate(os.path.abspath(args.marketdata), cal_cfg, cal_end, calibrated_md)
                taus = [(e - trade_date).days / DAYS_IN_YEAR
                        for e in listed_expiries(args.expiries, trade_date)]
                for leg, by_state in guard_e_df_state(arch, cal_end, taus).items():
                    logging.info('GUARD %s %-18s %s', cal_end, leg, '  '.join(
                        f'd~{k}: {v[0]:+.5f} (t {v[1]:+.2f})' for k, v in by_state.items()))
            if tag in done:
                logging.info('TRADE %s: SKIP (already completed, resuming)', tag)
                rec = done[tag]
            else:
                rec = one_trade(template, arch, trade_date, calibrated_md, args, run_dir, tag)
                for r in rec:
                    logging.info('TRADE %s corridor=%s: greedy=%s nohedge=%s $/oz  bound=%s '
                                 'PASS=%s  churn=%s', tag, r['corridor'], r['greedy_usd_oz'],
                                 r['nohedge_usd_oz'], r['pf_bound'], r['bound_pass'], r['churn'])
                _atomic_write_json(os.path.join(run_dir, f'row_{tag}.json'), rec)
        except Exception as exc:
            logging.exception('TRADE %s FAILED', tag)
            rec = [{'trade': tag, 'world': 'sync', 'fair': None, 'strike': None,
                    'n_seeds': len(args.seeds), 'market_dim': None, 'train_u': None,
                    'train_u_seeds': None, 'V_0': None, 'greedy_usd_oz': None,
                    'nohedge_usd_oz': None, 'pf_bound': None, 'bound_pass': None, 'churn': None,
                    'corridor': args.delta_corridor, 'error': str(exc)}]
        rows.extend(rec)
        _atomic_write_csv(os.path.join(run_dir, 'trades.csv'), rows)

    df = pd.DataFrame(rows)
    g = df['greedy_usd_oz']
    print('\n===== SYNCHRONIZED WALK-FORWARD ($/oz) =====')
    print(df.to_string(index=False))
    print(f"\ngreedy  mean {g.mean():+.2f}  min {g.min():+.2f}  max {g.max():+.2f}  "
          f"positives {int((g > 0).sum())}/{len(df)}  |  nohedge mean {df['nohedge_usd_oz'].mean():+.2f}"
          f"  |  bound-PASS {int(df['bound_pass'].sum())}/{len(df)}  |  margin {args.margin:+.2f}"
          f"  |  seeds {args.seeds}  |  corridor {args.delta_corridor}")
    print('run dir:', run_dir)


if __name__ == '__main__':
    main()
