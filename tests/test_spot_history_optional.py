"""Spot_Price_History is OPTIONAL in the hedging JSON contract.

With no `Portfolio_State.Spot_Price_History`, the solver must still train sane:
  * `referenced_commodities` is derived from the deal's live CommodityPrice factors (never
    from history keys), so the declared-underlying set is unchanged;
  * the utility scale falls back to CALIBRATED market data — spot from the CommodityPrice
    factor, σ from the underlying MarkovHMMSpotModel's stationary regime-weighted vol
    (`calibrated_annual_vol`) — instead of the realized-vol read off a history window;
  * the history prefix no-ops (`initial_time_index == 0`), value bounded, artifact present.

JSON-is-the-contract: load_json + run_job, history removed in code (the fixture template is
never edited). Companion to test_utility_scale_unit.py (the fail-loud unit coverage)."""
import json as jsonlib
import math
import os

import pytest

import numpy as np

import derivus as rf

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'fixtures', 'policy_test_simulate_only.json')


def _cfg_without_history():
    cfg = jsonlib.load(open(FIXTURE))
    calc = cfg['Calc']['Calculation']
    calc['Execution_Mode'] = 'solve_hedge'
    calc['Batch_Size'], calc['Simulation_Batches'] = 24, 2
    calc['Inner_Sub_Batch'] = 8
    calc['Inner_MC_Enabled'] = 'Yes'
    calc['Random_Seed'] = 1234
    hp = calc['Hedging_Problem']
    hp['Randomize_Initial_State'] = 'Yes'
    hp['Portfolio_State'].pop('Spot_Price_History', None)   # <-- the whole point
    hp['Solver'] = {
        'Object': 'DiffSolverV2',
        'Training_Action_Grid_Levels_Per_Axis': 5,
        'Training_Action_Chunk_Size': 64,
        'T_Min': 100,
        'DiffV2_Fit_Iters': 5,
    }
    return cfg


def test_a_history_row_at_the_base_date_is_refused():
    """History must be STRICTLY before the base date. A base-date row duplicates sim day 0 in
    the bundle timeline and shifts every *_sim view one row back — the solver then scores each
    decision against the step ENDING at its fork's anchor, a full day of lookahead per
    decision. Measured before the guard: the in-sim greedy verdict read +$982/oz expected on a
    fair three-month book (fork E[dF] correlated +0.89 with the realized outer step at β≈1),
    and the fork's L_t equalled liability_sim[t+1] BITWISE. Killed by removing the refusal —
    the run then completes with a confidently wrong verdict and nothing but this gate sees it."""
    cfg = jsonlib.load(open(FIXTURE))
    calc = cfg['Calc']['Calculation']
    calc['Execution_Mode'] = 'solve_hedge'
    calc['Batch_Size'], calc['Simulation_Batches'] = 24, 2
    calc['Inner_Sub_Batch'] = 8
    calc['Inner_MC_Enabled'] = 'Yes'
    hp = calc['Hedging_Problem']
    hp['Solver'] = {'Object': 'DiffSolverV2', 'Training_Action_Grid_Levels_Per_Axis': 3,
                    'Training_Action_Chunk_Size': 64, 'T_Min': 100, 'DiffV2_Fit_Iters': 2}
    sph = hp['Portfolio_State']['Spot_Price_History']
    fac = next(iter(sph))
    base = calc['Base_Date']['.Timestamp'] if isinstance(calc['Base_Date'], dict) \
        else calc['Base_Date']
    sph[fac]['Dates'].append({'.Timestamp': base})
    sph[fac]['Prices'].append(sph[fac]['Prices'][-1])
    cx = rf.Context(path_transform={}, file_transform={})
    cx.load_json((jsonlib.dumps(cfg), 'history_at_base.json'))
    with pytest.raises(Exception, match='STRICTLY before'):
        cx.run_job()


def test_the_realized_history_reaches_the_tradable_prefix():
    """The companion POSITIVE case: with `Spot_Price_History` present, a commodity tradable's
    history prefix must BE the realized series, not the flat first-row broadcast. The lookup
    resolves the deal's raw `Commodity` field (`'PLATINUM_LME.LME_CME'`, a composed name) to the
    primary spot's full factor name - the key space `_spot_price_history` validates the dict
    against. Keyed by the raw field it can never match: `.get()` -> None -> `tensor[:1].expand`
    -> thirty rows of zero variation ahead of every rolling feature, silently, for every config
    that passes validation - which is why this asserts variation AND the values."""
    cfg = jsonlib.load(open(FIXTURE))
    calc = cfg['Calc']['Calculation']
    calc['Execution_Mode'] = 'solve_hedge'
    calc['Batch_Size'], calc['Simulation_Batches'] = 24, 2
    calc['Inner_Sub_Batch'] = 8
    calc['Inner_MC_Enabled'] = 'Yes'
    calc['Random_Seed'] = 1234
    hp = calc['Hedging_Problem']
    hp['Solver'] = {'Object': 'DiffSolverV2', 'Training_Action_Grid_Levels_Per_Axis': 5,
                    'Training_Action_Chunk_Size': 64, 'T_Min': 100, 'DiffV2_Fit_Iters': 5}
    cx = rf.Context()
    cx.load_json((jsonlib.dumps(cfg), 'spot_history_prefix.json'))
    _, result = cx.run_job()
    bundle, runtime = result.bundle, result.runtime

    H = bundle.history_rows
    assert H > 0 and bundle.initial_time_index == H, (H, bundle.initial_time_index)
    declared = {k: [float(p) for p in v['Prices'][-H:]]
                for k, v in hp['Portfolio_State']['Spot_Price_History'].items()}
    checked = 0
    for n, meta in runtime['tradables'].items():
        raw = (meta.get('params') or {}).get('Commodity')
        key = 'CommodityPrice.' + raw.split('.')[0] if raw else None
        if key not in declared:
            continue
        got = bundle.tradables[n][:H, 0].detach().cpu().numpy()
        assert np.ptp(got) > 0.0, f'{n}: flat prefix - the realized branch never engaged'
        assert np.allclose(got, declared[key], rtol=1e-5), (n, got[:3], declared[key][:3])
        checked += 1
    assert checked > 0, 'no tradable reached the realized-prefix assertion'


def test_the_cash_account_prices_and_its_mark_accrues():
    """`CashAccountDeal` is a PRICED tradable - `Units / D(t)` - and its mark ratio is the ONLY
    financing path: `_growth_factors` reads consecutive marks, and variation margin routes off the
    runtime config, so a skipped cash deal loses interest on cash AND margin while everything else
    stays plausible. The template's block lacked `Units`: `KeyError` -> the canonical deal guard
    -> no tensor mark -> `_growth_factors` {} -> every balance passed through flat, in every run
    of this fixture, with a repeated CRITICAL as the only evidence. Worth $0.54/oz at 2026 rates.
    Any non-zero `Units` restores it - only the mark RATIO is consumed."""
    cfg = jsonlib.load(open(FIXTURE))
    calc = cfg['Calc']['Calculation']
    calc['Execution_Mode'] = 'solve_hedge'
    calc['Batch_Size'], calc['Simulation_Batches'] = 24, 2
    calc['Inner_Sub_Batch'] = 8
    calc['Inner_MC_Enabled'] = 'Yes'
    calc['Random_Seed'] = 1234
    calc['Hedging_Problem']['Solver'] = {
        'Object': 'DiffSolverV2', 'Training_Action_Grid_Levels_Per_Axis': 5,
        'Training_Action_Chunk_Size': 64, 'T_Min': 100, 'DiffV2_Fit_Iters': 5}
    cx = rf.Context()
    cx.load_json((jsonlib.dumps(cfg), 'cash_account_prices.json'))
    _, result = cx.run_job()
    bundle = result.bundle
    assert 'USD_CASH' in bundle.tradables, 'the cash account produced no mark - financing is off'
    cash = bundle.tradables['USD_CASH'][bundle.initial_time_index:, 0].detach().cpu().numpy()
    d = np.diff(cash)
    # strictly accruing while live, END-PADDED flat past the Investment_Horizon by the grid align
    assert np.all(d >= 0.0) and d.max() > 0.0, 'the cash mark does not accrue'
    assert cash[-1] / cash[0] > 1.001, f'no visible funding growth: {cash[0]} -> {cash[-1]}'


def test_spot_price_history_optional_trains_via_calibrated_scale():
    # the package under test must be this checkout, not another copy earlier on sys.path
    assert os.path.dirname(os.path.dirname(os.path.abspath(__file__))) in rf.__file__, rf.__file__
    cx = rf.Context()
    cx.load_json((jsonlib.dumps(_cfg_without_history()), 'spot_history_optional.json'))
    _, result = cx.run_job()
    bundle, runtime = result.bundle, result.runtime
    diag = (result.evaluation_summary or {}).get('diagnostics') or {}

    # Underlying set derived from the deal (live CommodityPrice factors), not history keys.
    assert runtime['referenced_commodities'] == ('CommodityPrice.PLATINUM_LME',), \
        runtime['referenced_commodities']

    # History is genuinely absent: no history tensor, prefix no-ops.
    assert not bundle.spot_price_history
    assert bundle.initial_time_index == 0

    # Utility scale came from the calibrated fallback (spot + stationary σ), not the floor.
    calib = bundle.calibrated_utility_inputs
    assert calib is not None, 'calibrated fallback not assembled'
    commodity, spot, sigma = calib
    assert commodity == 'CommodityPrice.PLATINUM_LME'
    assert spot > 0.0 and sigma > 0.0
    c = bundle.utility_scale
    assert c > 1.0e3, f'utility_scale collapsed to the floor: {c}'
    # c = total_leg_volume · spot · σ · √τ — reproduce it from the parts.
    tau = max(float(bundle.last_settlement_index - bundle.initial_time_index) / 252.0,
              1.0 / 252.0)
    expected_c = bundle.total_leg_volume * spot * sigma * (tau ** 0.5)
    assert math.isclose(c, expected_c, rel_tol=1e-6), (c, expected_c)

    # σ is the stationary regime-weighted vol of the calibrated 3-state HMM (Log_Price=True).
    P = np.array([[0.9903413013204756, 0.009412628327557179, 0.0002460703519671834],
                  [0.007917212174787837, 0.9783132181049096, 0.013769569720302594],
                  [0.0011542393670975122, 0.32217292523078384, 0.6766728354021186]])
    sig = np.array([0.15171719016534066, 0.24138966217119134, 0.6216348894195312])
    evals, evecs = np.linalg.eig(P.T)
    pi = np.real(evecs[:, int(np.argmin(np.abs(evals - 1.0)))]); pi = pi / pi.sum()
    expected_sigma = float(np.sqrt(float((pi * sig * sig).sum())))
    assert math.isclose(sigma, expected_sigma, rel_tol=1e-9), (sigma, expected_sigma)

    # Trains sane: bounded value, artifact present.
    assert diag.get('bounded') is True
    assert math.isfinite(float(diag['V_0'])) and abs(float(diag['V_0'])) < 50.0
    assert result.policy_artifact is not None
