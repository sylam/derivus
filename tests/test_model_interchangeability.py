"""Model interchangeability: ONE calc config, TWO spot-model worlds.

The base config (deal + tradables + calc + solver knobs) is IDENTICAL across both runs; the ONLY
difference is `MergeMarketData.MarketDataFile` pointing at a different `Price Models` block:

  * HMM world   — tests/fixtures/data/MarketDataRF_platinum_calibrated_cme.json (MarkovHMMSpotModel primary)
  * GARCH world — tests/fixtures/data/MarketDataRF_platinum_garch.json           (GARCHSpotModel primary)

No calc or config change is needed to swap: the calc speaks only the model-agnostic
StochasticProcess protocol, and every model-specific buffer key and recursion lives inside the
process class.

Covers:
  1. The deep-state privileged layout tracks the swapped model (regime one-hot/belief →
     log-variance → the log-OU's deviation, reversion and vol), every coordinate in the job's dtype.
  2. A few-iter solve_hedge, both worlds: bounded, and the V̂ market width resizes with the model
     (HMM market_dim 7 = belief(3)+price(1)+shared(3); GARCH 5 = log_h(1)+price(1)+shared(3)).
"""
import json
import os

import torch

import derivus as rf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHIPPING = os.path.join(REPO, 'tests', 'fixtures', 'platinum_hedge_shipping.json')
HMM_MKT = os.path.join(REPO, 'tests', 'fixtures', 'data', 'MarketDataRF_platinum_calibrated_cme.json')
GARCH_MKT = os.path.join(REPO, 'tests', 'fixtures', 'data', 'MarketDataRF_platinum_garch.json')


def _cfg(market_file, mode):
    """The shipping config with ONLY the market-data file swapped (+ tiny sizing)."""
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
        sol['T_Min'] = 100                  # a short sweep: the width is set before the first fit
    return cfg


def _run(market_file, mode):
    cx = rf.Context()
    cx.load_json((json.dumps(_cfg(market_file, mode), default=str), 'interchange.json'))
    _, res = cx.run_job()
    return res


def test_every_spot_model_reveals_its_state_in_the_job_s_dtype(tmp_path):
    """The deep-state layout tracks the swapped model with zero calc change - the HMM's regime
    one-hot and belief, the GARCH's log-variance, the log-OU's deviation, reversion and vol - and
    each is the walk's own coordinate, so a float64 job publishes every one of them in float64
    rather than rounding them to float32. The log-OU world is the GARCH one with the platinum spot
    walked by a log-OU, its utility scale stated since a log-OU reports no vol.

    Killing mutations: any of the five casts back to `torch.float32`; the GARCH layout declaring a
    width its walk does not publish.
    """
    market = json.load(open(GARCH_MKT))
    models = market['MarketData']['Price Models']
    models.pop('GARCHSpotModel.PLATINUM_CME')
    models['LogOUSpotModel.PLATINUM_CME'] = {'Kappa': 2.0, 'Theta': 0.0, 'Sigma': 0.25}
    market['MarketData']['Model Configuration']['.ModelParams']['modeldefaults'][
        'CommodityPrice'] = 'LogOUSpotModel'
    log_ou = tmp_path / 'MarketDataRF_platinum_logou.json'
    log_ou.write_text(json.dumps(market))
    revealed, layouts = {}, {}
    for name, mkt in (('HMM', HMM_MKT), ('GARCH', GARCH_MKT), ('LOGOU', str(log_ou))):
        cfg = _cfg(mkt, 'simulate_only')
        cfg['Calc']['Calculation']['Hedging_Problem']['Objective']['Utility_Scale_Explicit'] = 1e6
        cx = rf.Context()
        cx.load_json((json.dumps(cfg, default=str), 'dtype.json'))
        _, out = rf.run_hedgemontecarlo(cx.current_cfg, prec=torch.float64)
        revealed.update(out.bundle.privileged_factors)
        layouts[name] = out.runtime['privileged_layout']
        assert layouts[name] == {key: value.shape[-1] for key, value
                                 in out.bundle.privileged_factors.items()}, name
    assert layouts == {
        'HMM': {'platinum_cme_regime_onehot': 3, 'platinum_cme_regime_belief': 3},
        'GARCH': {'platinum_cme_log_h': 1},
        'LOGOU': {'platinum_cme_log_deviation': 1, 'platinum_cme_kappa': 1,
                  'platinum_cme_sigma': 1}}, layouts
    assert {name: str(value.dtype) for name, value in revealed.items()} == dict.fromkeys(
        ('platinum_cme_regime_onehot', 'platinum_cme_regime_belief', 'platinum_cme_log_h',
         'platinum_cme_log_deviation', 'platinum_cme_kappa', 'platinum_cme_sigma'), 'torch.float64')


def test_solve_reveal_width_resizes_with_model():
    """The V̂ market width is each world's revealed segments: HMM belief(3)+price(1)+shared(3)=7,
    GARCH log_h(1)+price(1)+shared(3)=5. Swept from step 100 at one fit iteration - the width is
    fixed before the first fit, so depth buys nothing here (30 s at T_Min 0, 7.5 s at 100).

    Killing mutation: the GARCH reveal dropping its log-variance segment, which reads 4."""
    dims = {}
    for name, mkt in (('HMM', HMM_MKT), ('GARCH', GARCH_MKT)):
        diag = _run(mkt, 'solve_hedge').evaluation_summary['diagnostics']
        assert diag['bounded'] is True, f'{name}: solve not bounded'
        dims[name] = diag['market_dim']
    assert dims == {'HMM': 7, 'GARCH': 5}, dims
