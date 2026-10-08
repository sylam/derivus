"""FXPartialTimeBarrierOption end to end, through the JSON contract and nothing else.

A partial-time (window) barrier: the barrier is live only on [0, Limit] (Barrier_At_Start Yes)
or [Limit, Expiry] (No), priced by the Heynen-Kat closed forms through `BivN` - Genz's bivariate
normal, 2e-16 in double, fully vectorised and differentiable everywhere.

THE ORACLE is an independent numpy Monte Carlo: weekly GBM steps inside the window with the
Brownian-bridge crossing probability per step and one exact step across the rest of the life. The
bridge's crossing law given both ends is exact for GBM at any step, so its monitoring is continuous
and matches the document's `Barrier_Monitoring_Frequency: 0M` (no discrete-monitoring shift); the
step size reaches only the knock-out rebate's hit-time discount, at most `r * 7/365 / 2` of the
leg. Nothing of the engine's is reused - the deal's own put-call/up-down transformations are
exactly what the oracle must not share.

The in-out parity KI = BS - KO is NOT a gate here: the pricer DEFINES knock-in that way, so the
identity is tautological. The oracle carries both directions independently instead.

MEASURED on the daily oracle, all eight direction/window configurations: worst 0.47%, five of the
eight under 0.2%, inside a 2% gate that carries the oracle's sampling error and nothing else - the
CDF's share of it is 2e-16.

`Cash_Rebate` was worth EXACTLY NOTHING before this: `getpartialbarrierpayoff` carried no rebate
term, so the rebate moved the mark by 0.0000 in all eight configurations where the oracle puts it
at 14 to 34 on a notional of 1000. Both directions - the knock-in through the expiry pad, the
knock-out through its missing pre-hit expectation - and the isolated leg is now gated to 0.56%.

THE REBATE LEGS ARE THE LIVE SIDE OF A LIVE WINDOW and nothing else, and the two other branches are
reachable through the document: a spot already on the barrier's far side, and a
`Barrier_Limit_Date` in the PAST. Both are certainties with no model in them, gated as equalities;
unguarded they read 83.76-107.00 where 0 or 50 is due and -30.31 to -52.47 where 0 or 48.04 is.

THE GRID THE CMC GATES RUN ON IS [0, Limit_Date, Expiry_Date] - three rows, because `'0d 12m(1m)'`
parses to {0d, 12m} and the deal contributes the limit. Not the deal's doing: `calc_deal_grid`
unions `base_time_grid` into every deal's own grid, so a finer `Time_grid` is honoured row by row.
The window is therefore monitored over TWO intervals here, which is exact for GBM given the bridge.

NOT GATED: that a knock-out's rebate settles on the reporting grid on the date of its hit while an
untouched knock-in's falls due at expiry alone - the settlement dates `FXPartialTimeBarrierOption`
declares were read off the deal object, not a document.
"""
import functools
import json
import math
import os
import sys

# reference-derivus shadow-import guard (MEMORY): pin the package under test to THIS repo.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

import derivus
from derivus import utils
from derivus.config import CustomJsonEncoder

BASE = pd.Timestamp('2024-06-28')
X0, R_USD, R_EUR, SIGMA = 1.25, 0.04, 0.02, 0.15
NOTIONAL = 1000.0
EXPIRY_D, LIMIT_D = 365, 182
T = EXPIRY_D / 365.0
B_CARRY = R_USD - R_EUR

FACTORS = {
    'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Spot': 1.0},
    'FxRate.EUR': {'Domestic_Currency': None, 'Interest_Rate': 'EUR', 'Spot': X0},
    'InterestRate.USD': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                         'Curve': utils.Curve([], [[0.0, R_USD], [5.0, R_USD]])},
    'InterestRate.EUR': {'Currency': 'EUR', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                         'Curve': utils.Curve([], [[0.0, R_EUR], [5.0, R_EUR]])},
    'FXVol.EUR.USD': {'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness',
                      'Surface': utils.Curve([], [[m, t, SIGMA] for m in (0.8, 1.0, 1.2)
                                                  for t in (0.02, 2.0)])}}


def _deal(barrier_type, barrier, strike=1.25, at_start='No', option_type='Call',
          limit_days=LIMIT_D, expiry_days=EXPIRY_D, rebate=0.0):
    return {'Object': 'FXPartialTimeBarrierOption', 'Reference': 'PB', 'Currency': 'USD',
            'Underlying_Currency': 'EUR', 'Payoff_Currency': 'USD', 'Discount_Rate': 'USD',
            'FX_Volatility': 'EUR.USD', 'Buy_Sell': 'Buy', 'Option_Type': option_type,
            'Strike_Price': strike, 'Barrier_Price': barrier, 'Barrier_Type': barrier_type,
            'Barrier_At_Start': at_start,
            'Barrier_Limit_Date': BASE + pd.DateOffset(days=limit_days),
            'Expiry_Date': BASE + pd.DateOffset(days=expiry_days),
            'Barrier_Monitoring_Frequency': pd.DateOffset(days=0), 'Cash_Rebate': rebate,
            'Underlying_Amount': NOTIONAL}


def _job(deal, calc=None):
    return {'Calc': {
        'Calculation': dict({'Object': 'BaseValuation', 'Base_Date': BASE, 'Currency': 'USD',
                             'MCMC_Simulations': 1, 'Random_Seed': 1}, **(calc or {})),
        'Deals': {'Reference': 'pb',
                  'Deals': {'Children': [{'Instrument': {'.Deal': deal}}]}},
        'MergeMarketData': {'MarketDataFile': '', 'ExplicitMarketData': {
            'System Parameters': {'Base_Currency': 'USD', 'Base_Date': BASE},
            'Valuation Configuration': {},
            'Price Factors': FACTORS}}}}


def _run(job):
    cx = derivus.Context()
    cx.load_json((json.dumps(job, cls=CustomJsonEncoder), 'pb'))
    _, out = cx.run_job()
    return out


def _mtm(out, ref='PB'):
    rows = out['Results']['mtm']
    return float(rows[rows['Reference'] == ref]['Value'].iloc[0])


# --------------------------------------------------------------------------------------------
# the oracle: bridge-corrected daily Monte Carlo, window-aware, engine-free
# --------------------------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _oracle_legs(barrier_type, barrier, strike=1.25, at_start='No', option_type='Call',
                 paths=1 << 17, seed=7, step_days=7):
    """The deal's two legs at t0: the option, and the rebate PER UNIT of `Cash_Rebate`. The rebate
    follows the deal's own conventions - a knock-OUT pays at the hit, an untouched knock-IN at
    expiry - so the hit-time discount accumulates per step rather than applying once.
    """
    lo, hi = (0, LIMIT_D) if at_start == 'Yes' else (LIMIT_D, EXPIRY_D)
    days = np.unique(np.r_[0, np.arange(lo, hi, step_days), hi, EXPIRY_D])
    dts = np.diff(days) / 365.0
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((len(dts), paths))
    z = np.concatenate([z, -z], axis=1)                    # antithetic
    log_s = np.log(X0) + np.cumsum(((B_CARRY - 0.5 * SIGMA ** 2) * dts)[:, None] +
                                   SIGMA * np.sqrt(dts)[:, None] * z, axis=0)
    s = np.exp(np.vstack([np.full((1, z.shape[1]), np.log(X0)), log_s]))
    up = 'Up' in barrier_type
    surv = np.ones(z.shape[1])
    hit_pv = np.zeros(z.shape[1])
    for i, dt in enumerate(dts):
        if not lo <= days[i] < hi:
            continue
        s0, s1 = s[i], s[i + 1]
        if up:
            hit = (s0 >= barrier) | (s1 >= barrier)
            bridge = np.exp(-2.0 * np.log(barrier / np.minimum(s0, barrier)) *
                            np.log(barrier / np.minimum(s1, barrier)) / (SIGMA ** 2 * dt))
        else:
            hit = (s0 <= barrier) | (s1 <= barrier)
            bridge = np.exp(-2.0 * np.log(np.maximum(s0, barrier) / barrier) *
                            np.log(np.maximum(s1, barrier) / barrier) / (SIGMA ** 2 * dt))
        p_no_cross = np.where(hit, 0.0, 1.0 - bridge)
        hit_pv = hit_pv + surv * (1.0 - p_no_cross) * math.exp(-R_USD * days[i + 1] / 365.0)
        surv = surv * p_no_cross
    cp = 1.0 if option_type == 'Call' else -1.0
    payoff = np.maximum(cp * (s[-1] - strike), 0.0)
    is_out = 'Out' in barrier_type
    weight = surv if is_out else 1.0 - surv
    return (NOTIONAL * math.exp(-R_USD * T) * float((payoff * weight).mean()),
            float(hit_pv.mean()) if is_out else math.exp(-R_USD * T) * float(surv.mean()))


#: the plain European the resolved states are measured against, engine to engine
VANILLA = {'Object': 'FXOptionDeal', 'Reference': 'VAN', 'Currency': 'USD',
           'Underlying_Currency': 'EUR', 'Payoff_Currency': 'USD', 'Discount_Rate': 'USD',
           'FX_Volatility': 'EUR.USD', 'Buy_Sell': 'Buy', 'Option_Type': 'Call',
           'Strike_Price': 1.25, 'Underlying_Amount': NOTIONAL,
           'Expiry_Date': BASE + pd.DateOffset(days=EXPIRY_D),
           'Option_Style': 'European', 'Settlement_Style': 'Cash'}

CASES = [(bt, at, bar) for at in ('Yes', 'No')
         for bt, bar in (('Up_And_Out', 1.40), ('Up_And_In', 1.40),
                         ('Down_And_Out', 1.12), ('Down_And_In', 1.12))]


REBATE = 50.0
# both directions, both windows, and a PUT of each direction: the rebate leg is valued on the
# deal's own coordinates, BEFORE the put-call reflection the option leg takes, so a put is the
# case that says the two are not accidentally sharing a variable
REBATE_CASES = CASES + [('Up_And_Out', 'Yes', 1.40), ('Down_And_In', 'No', 1.12)]
REBATE_TYPES = ['Call'] * len(CASES) + ['Put', 'Put']


@pytest.mark.parametrize('barrier_type,at_start,barrier,option_type',
                         [c + (t,) for c, t in zip(REBATE_CASES, REBATE_TYPES)],
                         ids=['%s-%s-%s' % (bt, at, t)
                              for (bt, at, _), t in zip(REBATE_CASES, REBATE_TYPES)])
def test_the_closed_form_and_its_rebate_price_to_the_independent_oracle(
        barrier_type, at_start, barrier, option_type):
    """Every direction and both windows, the option leg and the ISOLATED rebate leg - `v(R) - v(0)`
    against the oracle's own rebate leg, so the option leg's error cannot pay for a rebate error.
    The rebate legs were worth EXACTLY NOTHING once (`getpartialbarrierpayoff` carried no rebate
    term), where the oracle puts them at 14 to 34 on a notional of 1000; on the daily oracle the
    worst rebate reading over the eight Call configurations was 0.56%. A start-window leg reads no
    bivariate at all and an end-window one reads `BivN` twice. The in-out parity KI = BS - KO is NOT
    a gate here: the pricer DEFINES knock-in that way, so the oracle carries both directions.

    Killing mutation: the closed form handed the other window (`Barrier_At_Start` inverted).
    """
    deal = dict(barrier_type=barrier_type, at_start=at_start, option_type=option_type)
    v0 = _mtm(_run(_job(_deal(barrier=barrier, rebate=0.0, **deal))))
    vr = _mtm(_run(_job(_deal(barrier=barrier, rebate=REBATE, **deal))))
    opt, reb = _oracle_legs(barrier_type, barrier, at_start=at_start, option_type=option_type)
    scale = max(abs(opt), 0.02 * NOTIONAL)
    assert abs(v0 - opt) / scale < 2e-2, ('option leg', v0, opt)
    assert abs(vr - v0 - REBATE * reb) / (REBATE * reb) < 2e-2, ('rebate leg', vr - v0, REBATE * reb)


#: `(barrier_type, barrier, limit_days, state)`. A START window resolves two ways with no model
#: left in it: TOUCHED - a live window whose spot already sits on the barrier's far side - and
#: CLOSED, a limit date in the past, which base valuation carries no touch through.
RESOLVED = [
    ('Up_And_Out', 1.12, LIMIT_D, 'dead'), ('Down_And_Out', 1.40, LIMIT_D, 'dead'),
    ('Up_And_In', 1.12, LIMIT_D, 'vanilla'), ('Down_And_In', 1.40, LIMIT_D, 'vanilla'),
    ('Up_And_Out', 1.12, -30, 'vanilla'), ('Up_And_Out', 1.40, -30, 'vanilla'),
    ('Down_And_Out', 1.12, -30, 'vanilla'), ('Down_And_Out', 1.40, -30, 'vanilla'),
    ('Up_And_In', 1.12, -30, 'nothing'), ('Up_And_In', 1.40, -30, 'nothing'),
    ('Down_And_In', 1.12, -30, 'nothing'), ('Down_And_In', 1.40, -30, 'nothing')]


def test_a_resolved_start_window_is_the_state_it_resolved_to():
    """Both legs of a start window that has RESOLVED, on the twelve documents that reach it, as
    equalities with no model in them: TOUCHED is a certainty off today's spot - the knock-out is
    dead and its rebate is due NOW, the knock-in is the vanilla with no rebate leg - while CLOSED is
    the history the deal carries, which is none, so the knock-out is the vanilla and the knock-in
    worth nothing but its expiry rebate `Cash_Rebate * exp(-rT)`. The rebate legs unguarded, on a
    rebate of 50: the touched knock-out paid 83.76/92.30 where 50 is due, the touched knock-in
    -30.31/-38.18 where nothing is, the seasoned knock-out 97.97/107.00 against nothing, and the
    seasoned knock-in -44.11/-52.47 against 48.04.

    THE EIGHT-FLOAT IDENTITY IS ABOUT DETECTION, NOT ARITHMETIC. `pv = where(closed, bs, pv)` hands
    back the literal `bs` and the knock-in parity subtracts it from itself, so the one float all
    eight land on (85.2998492970, 0x40555330bb1b22a7) can only break if a document is read into the
    wrong state. WHICH documents land there is the content: a touched knock-in and a closed
    knock-out, in both directions and on BOTH sides of the level. The other four are exact literals:
    nothing, or the rebate at its own timing.

    Unresolved, the same twelve read: 85.98 and 404.02 for the touched knock-in, -0.68 and -318.72
    for the touched knock-out where 0 and 50 are due, 91.44 and 456.40 for the closed knock-in where
    nothing is, and 84.63 / **-6.141370** / 84.63 / -371.10 for the closed knock-out - the -6.14
    being a seasoned `Up_And_Out` sitting below its own barrier, which is the reading this defect
    was carried under.

    THE TIE TO A REAL VANILLA STAYS A TOLERANCE. The deal's own Black reads 85.2998 against the
    engine's `FXOptionDeal` at 85.4172, 0.137% apart on inputs that agree - the same gap
    `test_an_unreachable_barrier_is_the_vanilla_and_a_certain_one_is_nothing` carries, and not this
    gate's subject.

    Killing mutation: a limit date already past no longer closing the window.
    """
    v_van = _mtm(_run(_job(VANILLA)), 'VAN')
    vanillas = set()
    for barrier_type, barrier, limit_days, state in RESOLVED:
        kwargs = dict(barrier_type=barrier_type, barrier=barrier, at_start='Yes',
                      limit_days=limit_days)
        v0 = _mtm(_run(_job(_deal(rebate=0.0, **kwargs))))
        vr = _mtm(_run(_job(_deal(rebate=REBATE, **kwargs))))
        where = (barrier_type, barrier, limit_days, v0, vr)
        if state == 'vanilla':
            assert vr == v0, ('a resolved window carries no rebate leg', where)
            vanillas.add(v0)
        elif state == 'dead':
            assert v0 == 0.0, ('a touched knock-out is worth nothing', where)
            assert vr == REBATE, ('the rebate is paid AT the hit, which is now', where)
        else:
            assert v0 == 0.0, ('a closed knock-in never knocked in', where)
            assert abs(vr - REBATE * math.exp(-R_USD * T)) < 1e-9 * REBATE, where
    assert len(vanillas) == 1, ('the vanilla is one number, not %d' % len(vanillas), vanillas)
    assert abs(vanillas.pop() - v_van) / v_van < 2e-3, (vanillas, v_van)


#: the eight CLOSED documents of `RESOLVED`, both directions and both sides of the level
CLOSED_BOTH_SIDES = [(bt, bar) for bt in ('Up_And_Out', 'Down_And_Out', 'Up_And_In', 'Down_And_In')
                     for bar in (1.12, 1.40)]


def test_a_closed_start_window_reads_the_same_state_its_own_scenario_walk_does():
    """The mark reads the resolution off a closed form and the scenario walk off its touch
    accumulator. They are the same deal on the same date and must name the same state.

    ROW 0 IS A POINT TEST, and `in_window` took it for every `Barrier_At_Start` document whether or
    not the window was still open - so a deal whose window closed 30 days ago tested a barrier
    nobody was watching, exactly the defect `test_a_touch_outside_the_window_does_not_knock` names
    one row earlier. Rows 1..n never had it, so the walk contradicted itself as well as the mark:
    the seasoned `Up_And_Out` 1.12 marked 85.2998 and its own row 0 read **0.0000**, while its
    knock-in marked 0 and read **85.2999**. The four documents INSIDE the level always agreed and
    must not move.

    The row is a certainty, so it is read per scenario: every path shares the base date's spot, and
    a resolved knock-in is exactly `0.0` on all of them. The knock-out's tie to the mark is 1.4e-8,
    the CMC's float32 against the mark's float64.

    Killing mutation: row 0 point-tested for every start-window document, closed or not.
    """
    for barrier_type, barrier in CLOSED_BOTH_SIDES:
        deal = _deal(barrier_type, barrier, at_start='Yes', limit_days=-30)
        base = _mtm(_run(_job(deal)))
        job = _job(deal, calc={
            'Object': 'CreditMonteCarlo', 'Time_grid': '0d 12m(1m)', 'Batch_Size': 256,
            'Simulation_Batches': 1, 'MCMC_Simulations': 1, 'Deflation_Interest_Rate': 'USD'})
        md = job['Calc']['MergeMarketData']['ExplicitMarketData']
        md['Price Models'] = {'GBMAssetPriceModel.EUR': {'Vol': SIGMA, 'Drift': R_USD - R_EUR}}
        md['Model Configuration'] = {'.ModelParams': {
            'modeldefaults': {'FxRate': 'GBMAssetPriceModel'}, 'modelfilters': {}}}
        row0 = np.asarray(_run(job)['Results']['mtm'], float)[0]
        where = (barrier_type, barrier, base, row0.min(), row0.max())
        assert row0.min() == row0.max(), ('a resolved row is one number per scenario', where)
        if 'In' in barrier_type:
            assert base == 0.0 and row0.max() == 0.0, ('a closed knock-in never knocked in', where)
        else:
            assert abs(row0.max() - base) < 1e-6 * base, ('the walk and the mark disagree', where)


def test_an_unreachable_barrier_is_the_vanilla_and_a_certain_one_is_nothing():
    """The two degenerate anchors: a KO whose barrier no path reaches is the plain European
    (engine-vs-engine against an FXOptionDeal document), and its KI is worth nothing.

    Killing mutation: the knock-in's parity taken with the wrong sign (`bs + pv`).
    """
    v_van = _mtm(_run(_job(VANILLA)), 'VAN')
    for at_start in ('Yes', 'No'):
        ko = _mtm(_run(_job(_deal('Up_And_Out', 5.0, at_start=at_start))))
        ki = _mtm(_run(_job(_deal('Up_And_In', 5.0, at_start=at_start))))
        assert abs(ko - v_van) / v_van < 2e-3, (at_start, ko, v_van)
        assert abs(ki) < 2e-3 * v_van, (at_start, ki)


@pytest.mark.parametrize('barrier_type,at_start,barrier,rebate', [
    ('Up_And_Out', 'Yes', 1.40, 0.0), ('Up_And_Out', 'No', 1.40, 0.0),
    ('Down_And_In', 'No', 1.20, 0.0), ('Down_And_In', 'Yes', 1.12, REBATE)],
    ids=['KO-start', 'KO-end', 'KI-end', 'KI-start-rebate'])
def test_the_deal_ages_as_a_martingale(barrier_type, at_start, barrier, rebate):
    """Under a risk-neutral simulation the DEFLATED profile mean sits on the t0 mark at every row -
    across the window edge, through the knock transitions the bridge probability accumulates, and
    onto the expiry settlement. Any aging defect is decay or drift in what must be flat:

      a window monitored outside itself - a start window closing at its limit leaves the rest of
        the life in which a crossing has no effect, and monitoring it read the expiry row 87% low;
      the limit left unclamped past the limit date, which priced every later row NaN;
      an untouched knock-in's rebate carried only through the expiry pad - a t0 mark of 5.62
        against an expiry row of 39.9, where the profile holds 39.86 across every row.

    Killing mutation: the window mask dropped, every interval monitored.
    """
    t0 = _mtm(_run(_job(_deal(barrier_type, barrier, at_start=at_start, rebate=rebate))))
    job = _job(_deal(barrier_type, barrier, at_start=at_start, rebate=rebate), calc={
        'Object': 'CreditMonteCarlo', 'Time_grid': '0d 12m(1m)', 'Batch_Size': 8192,
        'Simulation_Batches': 2, 'MCMC_Simulations': 1, 'Deflation_Interest_Rate': 'USD'})
    md = job['Calc']['MergeMarketData']['ExplicitMarketData']
    md['Price Models'] = {'GBMAssetPriceModel.EUR': {'Vol': SIGMA, 'Drift': R_USD - R_EUR}}
    md['Model Configuration'] = {'.ModelParams': {
        'modeldefaults': {'FxRate': 'GBMAssetPriceModel'}, 'modelfilters': {}}}
    profile = _run(job)['Results']['mtm']
    days = np.array([(d - BASE).days for d in profile.index], dtype=float)
    deflated = np.asarray(profile, float).mean(axis=1) * np.exp(-R_USD * days / 365.0)
    assert np.abs(deflated - t0).max() / max(abs(t0), 0.02 * NOTIONAL) < 3e-2, (
        t0, deflated, list(days))


# --------------------------------------------------------------------------------------------
# the window-touch decision, and which of its two forms is a LATCH
# --------------------------------------------------------------------------------------------
# `get_fx_barrier_underlying` publishes a bridge variance rate only while the deal's QUOTE leg is
# static. Report off a THIRD currency and USD is simulated too, the rate is absent, every interval
# variance is zero, and `barrier_touched` collapses to a 0/1 endpoint test - the latch form.
def _cva_job(deal, spot=X0, gradient=False, bridge=True, hessian=False,
             batch=8192, batches=4, bandwidth=0.01, window_touch=None):
    factors = {k: dict(v) for k, v in FACTORS.items()}
    factors['FxRate.EUR'] = dict(factors['FxRate.EUR'], Spot=spot)
    factors['SurvivalProb.CPTY'] = {
        'Recovery_Rate': 0.4, 'Curve': utils.Curve([], [[0.0, 0.0], [10.0, 0.4]])}
    models = {'GBMAssetPriceModel.EUR': {'Vol': SIGMA, 'Drift': 0.0}}
    base = 'USD'
    if not bridge:
        base = 'GBP'
        factors['FxRate.GBP'] = {'Domestic_Currency': None, 'Interest_Rate': 'GBP', 'Spot': 1.0}
        factors['FxRate.USD'] = dict(factors['FxRate.USD'], Domestic_Currency='GBP', Spot=1.0)
        factors['FxRate.EUR'] = dict(factors['FxRate.EUR'], Domestic_Currency='GBP')
        factors['InterestRate.GBP'] = {
            'Currency': 'GBP', 'Day_Count': 'ACT_365', 'Sub_Type': None,
            'Curve': utils.Curve([], [[0.0, 0.03], [5.0, 0.03]])}
        models['GBMAssetPriceModel.USD'] = {'Vol': 0.10, 'Drift': 0.0}
    return {'Calc': {
        'Calculation': {
            'Object': 'CreditMonteCarlo', 'Base_Date': BASE, 'Currency': base,
            'Time_grid': '0d 12m(1m)', 'Batch_Size': batch, 'Simulation_Batches': batches,
            'MCMC_Simulations': 1, 'Random_Seed': 1, 'Deflation_Interest_Rate': base,
            'Gradient_Variables': 'Factors', 'Boundary_AAD_Bandwidth': bandwidth,
            **({'Boundary_AAD_Window_Touch': window_touch} if window_touch else {}),
            'Credit_Valuation_Adjustment': {
                'Calculate': 'Yes', 'Counterparty': 'CPTY', 'Deflate_Stochastically': 'No',
                'Stochastic_Hazard_Rates': 'No', 'Gradient': 'Yes' if gradient else 'No',
                'Hessian': 'Yes' if hessian else 'No'}},
        'Deals': {'Reference': 'pb',
                  'Deals': {'Children': [{'Instrument': {'.Deal': deal}}]}},
        'MergeMarketData': {'MarketDataFile': '', 'ExplicitMarketData': {
            'System Parameters': {'Base_Currency': base, 'Base_Date': BASE},
            'Valuation Configuration': {}, 'Price Factors': factors, 'Price Models': models,
            'Model Configuration': {'.ModelParams': {
                'modeldefaults': {'FxRate': 'GBMAssetPriceModel'}, 'modelfilters': {}}}}}}}


def _cva(**kwargs):
    """`(cva, dCVA/d(EUR spot) or None)`. The bumped runs change one factor value and nothing else,
    so common random numbers arrive through the contract rather than a patch."""
    out = _run(_cva_job(**kwargs))
    cva = float(out['Results']['cva'])
    if not kwargs.get('gradient'):
        return cva, None
    g = out['Results']['grad_cva']['Gradient']
    rows = [i for i in g.index if 'FxRate' in str(i[0]) and 'EUR' in str(i[0])]
    return cva, (float(g.loc[rows[0]]) if rows else 0.0)


# a window that CLOSES mid-profile, on a barrier the spot reaches: the latch fires at the limit
# date, which is a reporting row of its own
LATCH_DEAL = _deal('Up_And_Out', 1.32, at_start='Yes', limit_days=182)


def test_the_window_touch_decision_registers_only_where_it_is_a_latch_and_by_default():
    """ONE DECISION, ONE ESTIMATOR, and which estimator depends on whether the bridge is there.
    With an interval variance `barrier_touched` returns a Brownian-bridge PROBABILITY continuous in
    the spot, so ordinary AAD already carries the flux (dCVA/dspot 0.44% off a CRN ladder flat to
    1.02%, nothing registered); without one it returns a 0/1 indicator with zero derivative almost
    everywhere - a latch, the only form needing a `LatchedBoundarySet`.

    The seam that says which happened, nothing patched: a CVA Hessian is refused BY NAME on a book
    that registered a boundary correction, strictly before the exposure kink term is built. Both
    branches refuse - a knocked-out path is worth exactly zero, so an ATOM sits at the kink either
    way - but for different reasons, which makes the MESSAGE the reading: the bridge's `ATOM of
    exposure`, the endpoints' `registered a boundary correction`.

    `Boundary_AAD_Window_Touch` defaults to Yes, on the measured SIGN - on a grid carrying six live
    decisions every CRN reading over five seeds and both path counts is negative where the
    unregistered delta is positive - and `'No'` is the unregistered estimator one value away: the
    endpoint branch then reads the bridge's message, and its gradient is BIT-IDENTICAL to the same
    run with the correction suppressed through the bandwidth.

    Killing mutation: `Boundary_AAD_Window_Touch` declared with default 'No'.
    """
    for bridge, says in ((True, 'ATOM of exposure'), (False, 'registered a boundary correction')):
        with pytest.raises(utils.SecondOrderRefused) as refusal:
            _run(_cva_job(LATCH_DEAL, gradient=True, bridge=bridge, hessian=True, batch=512,
                          batches=1))
        assert says in str(refusal.value), (bridge, str(refusal.value))
    with pytest.raises(utils.SecondOrderRefused) as refusal:
        _run(_cva_job(LATCH_DEAL, gradient=True, bridge=False, hessian=True, batch=512, batches=1,
                      window_touch='No'))
    assert 'ATOM of exposure' in str(refusal.value), str(refusal.value)

    kw = dict(deal=LATCH_DEAL, gradient=True, bridge=False, batch=1024, batches=1)
    off = _cva(window_touch='No', **kw)[1]
    suppressed = _cva(bandwidth=1e-12, **kw)[1]
    assert off == suppressed, (off, suppressed)


REBATED_LATCH_DEAL = _deal('Up_And_Out', 1.32, at_start='Yes', limit_days=182, rebate=REBATE)


def test_asking_for_the_partial_barrier_sensitivities_does_not_move_the_exposure():
    """BIT-identical, not approximately: the correction is `gap - gap.detach()`, worth exactly zero
    forward, so any drift means the registration path perturbed the valuation. On the endpoint
    branch, the one that registers at all, and on a REBATED knock-out, whose registration declares
    one `cash_events` entry per decision - the rebate that decision pays if it fires first - on a
    path no other fixture takes.

    THE TERM'S MAGNITUDE ON THIS GRID is not established: the deal prices on THREE rows, so the
    window carries two decisions and one is live, and over five seeds dCVA/dspot reads a registered
    median -0.666 at 32768 paths against an unregistered +3.292 while the CRN oracle spreads 327% of
    its own median. A ROW A MONTH is where it resolves: -1.915 registered against +1.318 at 32768
    paths, every CRN reading negative and the pooled oracle 0.5% from the registered delta.

    Killing mutation: the branch the registration settles at expiry scaled into the reported mark in
    place - an aliased write.
    """
    off, _ = _cva(deal=REBATED_LATCH_DEAL, bridge=False, batch=1024, batches=1)
    on, grad = _cva(deal=REBATED_LATCH_DEAL, gradient=True, bridge=False, batch=1024, batches=1)
    assert off == on, 'the exposure moved when sensitivities were requested: %r -> %r' % (off, on)
    assert grad is not None and abs(grad) > 0.0, 'no EUR spot gradient was reported at all'
