"""Model interchangeability: ONE calc config, TWO spot-model worlds.

The base config (deal + tradables + calc + solver knobs) is IDENTICAL across both runs; the ONLY
difference is `MergeMarketData.MarketDataFile` pointing at a different `Price Models` block:

  * HMM world   — tests/fixtures/data/MarketDataRF_platinum_calibrated_cme.json (MarkovHMMSpotModel primary)
  * GARCH world — tests/fixtures/data/MarketDataRF_platinum_garch.json           (GARCHSpotModel primary)

No calc or config change is needed to swap: the calc speaks only the model-agnostic
StochasticProcess protocol, and every model-specific buffer key and recursion lives inside the
process class.

Covers:
  1. simulate_only + a full stepper replay to done, both worlds (exercises generate + the observed
     replay reseed path with zero config change).
  2. The deep-state privileged layout tracks the swapped model (regime one-hot/belief → log-variance).
  3. A few-iter solve_hedge, both worlds: bounded, and the V̂ market width resizes with the model
     (HMM market_dim 9 = belief(3)+price(1)+shared(5); GARCH 7 = log_h(1)+price(1)+shared(5)).
"""
import json
import os
from types import SimpleNamespace

import numpy as np
import torch

import derivus as rf
from derivus.stochasticprocess import LogOUSpotModel

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHIPPING = os.path.join(REPO, 'tests', 'fixtures', 'platinum_hedge_shipping.json')
HMM_MKT = os.path.join(REPO, 'tests', 'fixtures', 'data', 'MarketDataRF_platinum_calibrated_cme.json')
GARCH_MKT = os.path.join(REPO, 'tests', 'fixtures', 'data', 'MarketDataRF_platinum_garch.json')


def _cfg(market_file, mode):
    """The shipping config with ONLY the market-data file swapped (+ tiny CPU sizing)."""
    cfg = json.load(open(SHIPPING))
    cfg['Calc']['MergeMarketData']['MarketDataFile'] = market_file
    calc = cfg['Calc']['Calculation']
    calc['Execution_Mode'] = mode
    calc['Batch_Size'] = 12
    calc['Simulation_Batches'] = 2
    calc['Random_Seed'] = 1
    if mode == 'simulate_only':
        calc['Hedging_Problem'].pop('Solver', None)
    else:
        calc['Inner_Sub_Batch'] = 4
        calc['Inner_MC_Enabled'] = 'Yes'
        calc['Hedging_Problem']['Randomize_Initial_State'] = 'No'
        sol = calc['Hedging_Problem']['Solver']
        sol['DiffV2_Fit_Iters'] = 1
        sol['Training_Action_Grid_Levels_Per_Axis'] = 3
        sol['DiffV2_Hidden'] = 8
    return cfg


def _run(market_file, mode):
    cx = rf.Context()
    cx.load_json((json.dumps(_cfg(market_file, mode), default=str), 'interchange.json'))
    _, res = cx.run_job()
    return res


def test_simulate_only_and_stepper_both_worlds():
    layouts = {}
    for name, mkt in (('HMM', HMM_MKT), ('GARCH', GARCH_MKT)):
        res = _run(mkt, 'simulate_only')
        # A full stepper replay to done — exercises the observed-path reseed with zero config change.
        stepper = res.create_stepper()
        steps = 0
        while not stepper.done:
            last = stepper.step(None)
            steps += 1
        assert steps > 0, f'{name}: stepper did not advance'
        pnl = (last['transition_pnl_excess'] + last['transition_liability_value'])
        assert torch.isfinite(pnl).all(), f'{name}: non-finite stepper P&L'
        layouts[name] = res.runtime['privileged_layout']

    # The deep-state privileged layout tracks the swapped model, with zero calc change.
    assert layouts['HMM'] == {'platinum_cme_regime_onehot': 3, 'platinum_cme_regime_belief': 3}, layouts['HMM']
    assert layouts['GARCH'] == {'platinum_cme_log_h': 1}, layouts['GARCH']


def test_every_spot_model_reveals_its_state_in_the_job_s_dtype(tmp_path):
    """The HMM's regime one-hot and belief, the GARCH's log-variance and the log-OU's deviation,
    reversion and vol are the walk's own coordinates, so a float64 job publishes every one of them
    in float64 rather than rounding them to float32. The log-OU world is the GARCH one with the
    platinum spot walked by a log-OU, its utility scale stated since a log-OU reports no vol.

    KILLING MUTATION: any of the five casts back to `torch.float32`.
    """
    market = json.load(open(GARCH_MKT))
    models = market['MarketData']['Price Models']
    models.pop('GARCHSpotModel.PLATINUM_CME')
    models['LogOUSpotModel.PLATINUM_CME'] = {'Kappa': 2.0, 'Theta': 0.0, 'Sigma': 0.25}
    market['MarketData']['Model Configuration']['.ModelParams']['modeldefaults'][
        'CommodityPrice'] = 'LogOUSpotModel'
    log_ou = tmp_path / 'MarketDataRF_platinum_logou.json'
    log_ou.write_text(json.dumps(market))
    revealed = {}
    for mkt in (HMM_MKT, GARCH_MKT, str(log_ou)):
        cfg = _cfg(mkt, 'simulate_only')
        cfg['Calc']['Calculation']['Hedging_Problem']['Objective']['Utility_Scale_Explicit'] = 1e6
        cx = rf.Context()
        cx.load_json((json.dumps(cfg, default=str), 'dtype.json'))
        _, out = rf.run_hedgemontecarlo(cx.current_cfg, prec=torch.float64)
        revealed.update(out.bundle.privileged_factors)
    assert {name: str(value.dtype) for name, value in revealed.items()} == dict.fromkeys(
        ('platinum_cme_regime_onehot', 'platinum_cme_regime_belief', 'platinum_cme_log_h',
         'platinum_cme_log_deviation', 'platinum_cme_kappa', 'platinum_cme_sigma'), 'torch.float64')


def test_an_anchored_log_ou_holds_its_level_to_an_ulp_in_float32():
    """A float32 log-OU anchored at its spot with no noise, stepped daily - S0 60 at Kappa 2 for five
    years, S0 100 at Kappa 0.5 for ten: every row reads its level within one float32 ulp (measured
    6.4e-8 and 7.6e-8), the walk being centred on Theta.

    Killed by: the pull `-expm1(-Kappa dt)` beside the rounded decay, 2.05e-5 and 4.95e-5 off; the
    pull `1 - e^{-Kappa dt}`, 5.1e-7 and 5.3e-7 off."""
    for kappa, spot, rows in ((2.0, 60.0, 1826), (0.5, 100.0, 3653)):
        process = LogOUSpotModel(None, {'Kappa': kappa, 'Theta': 0.0, 'Sigma': 0.0})
        years = np.arange(rows) / 365.0
        process.precalculate(None, SimpleNamespace(scen_time_grid=years, time_grid_years=years),
                             torch.tensor([spot]), None, 0)
        path = process.generate(SimpleNamespace(t_random_numbers=torch.zeros(1, rows, 4)))
        assert path.dtype == torch.float32 and float(
            (path.double() / spot - 1.0).abs().max()) <= 2.0 ** -23, (kappa, spot)


def test_solve_reveal_width_resizes_with_model():
    dims = {}
    for name, mkt in (('HMM', HMM_MKT), ('GARCH', GARCH_MKT)):
        diag = (_run(mkt, 'solve_hedge').evaluation_summary or {}).get('diagnostics') or {}
        assert diag.get('bounded') is True, f'{name}: solve not bounded'
        dims[name] = diag.get('market_dim')
    # V̂ market width resizes with the model: HMM belief(3)+price(1)+shared(3)=7; GARCH log_h(1)+price(1)+3=5.
    # (shared dropped 4 -> 3 when the retired VAR carry model (3 revealed coordinates) gave way
    # to QuadraticCarryCurveModel's 2-knot curve; earlier it dropped 5 -> 4 when the CME_FLAT
    # identity basis was removed — a composed name needs no +0 factor.)
    assert dims['HMM'] == 7, dims['HMM']
    assert dims['GARCH'] == 5, dims['GARCH']
    assert dims['HMM'] - dims['GARCH'] == 2, dims
