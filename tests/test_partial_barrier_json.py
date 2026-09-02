"""FXPartialTimeBarrierOption end to end, through the JSON contract and nothing else.

A partial-time (window) barrier: the barrier is live only on [0, Limit] (Barrier_At_Start Yes)
or [Limit, Expiry] (No), priced by the Heynen-Kat closed forms through `ApproxBivN` - the
Tsay-Ke bivariate-normal approximation, ~1e-4 absolute, chosen because it is fully vectorised
and differentiable everywhere.

THE ORACLE is an independent numpy Monte Carlo: daily GBM steps inside the window with the
Brownian-bridge crossing probability per step, so its monitoring is continuous up to the
flat-parameter discretisation, matching the document's `Barrier_Monitoring_Frequency: 0M`
(no discrete-monitoring barrier shift). Nothing of the engine's is reused - the deal's own
put-call/up-down transformations are exactly what the oracle must not share.

The in-out parity KI = BS - KO is NOT a gate here: the pricer DEFINES knock-in that way, so the
identity is tautological. The oracle carries both directions independently instead.

MEASURED, all eight direction/window configurations against the oracle: worst 0.43% (Up_And_Out
start-window), most under 0.2% - inside a 2% gate that carries the oracle's own daily-bridge
error and ApproxBivN's ~7e-4 worst-case CDF error. MUTATIONS: the window mask dropped reads the
expiry row at 9.51 against 75.83 (-87%, mid-life crossings knocking a two-day window); the limit
clamp dropped brings the NaN back and kills the finiteness gate.

`Cash_Rebate` IS NOW GATED, and it was worth exactly nothing before: `getpartialbarrierpayoff`
carried no rebate term, so at base valuation the rebate moved the mark by 0.0000 in every one of
the eight configurations while the oracle puts it at 14 to 34 on a notional of 1000 at a rebate of
50. Both directions were affected - the knock-in through the roadmap's expiry pad, the knock-out
through its missing pre-hit expectation. MEASURED on the isolated leg, worst 0.56%.

THE REBATE LEGS ARE THE LIVE SIDE OF A LIVE WINDOW and nothing else, and the two other branches
are reachable through the document rather than only in principle: a spot already on the barrier's
far side (the deal is knocked, and base valuation carries NO touch mask - `need_spot_at_expiry` is
zero, so the closed form answers alone) and a `Barrier_Limit_Date` in the PAST (a seasoned deal,
where `limit` reaches the payoff negative). Both are certainties with no model in them and are
gated as equalities; unguarded they read 83.76-107.00 where 0 or 50 is due and -30.31 to -52.47
where 0 or 48.04 is.

THE GRID THESE CMC GATES RUN ON IS THE DEAL'S OWN, and it is [0, Limit_Date, Expiry_Date] - three
rows, whatever `Time_grid` asks for. `FXPartialTimeBarrierOption` declares `path_dependent` but
does not override `add_grid_dates`, so it contributes only its two reval dates and, as the only
deal in the portfolio, its dates ARE the reporting grid. The window is therefore monitored over
two intervals, which is exact for GBM given the bridge and is why the oracle agrees; it also means
a profile assertion here reads three rows, and the middle one is the limit date.
"""
import io
import json
import logging
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
T, T1 = EXPIRY_D / 365.0, LIMIT_D / 365.0
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
        'Deals': {'Tag_Titles': '', 'Reference': 'pb',
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
def _oracle_legs(barrier_type, barrier, strike=1.25, at_start='No', option_type='Call',
                 t1=T1, t=T, paths=1 << 17, seed=7):
    """The deal's two legs at t0: the option, and the rebate PER UNIT of `Cash_Rebate`.

    The rebate follows the deal's own conventions - a knock-OUT pays at the hit, an untouched
    knock-IN at expiry - so the hit-time discount accumulates per step rather than applying once.
    """
    rng = np.random.default_rng(seed)
    steps = int(round(t * 365))
    dt = t / steps
    z = rng.standard_normal((steps, paths))
    z = np.concatenate([z, -z], axis=1)                    # antithetic
    log_s = np.log(X0) + np.cumsum((B_CARRY - 0.5 * SIGMA ** 2) * dt +
                                   SIGMA * math.sqrt(dt) * z, axis=0)
    s = np.exp(np.vstack([np.full((1, z.shape[1]), np.log(X0)), log_s]))
    up = 'Up' in barrier_type
    lo = int(round((0.0 if at_start == 'Yes' else t1) * 365))
    hi = int(round((t1 if at_start == 'Yes' else t) * 365))
    surv = np.ones(z.shape[1])
    hit_pv = np.zeros(z.shape[1])
    for i in range(lo, hi):
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
        hit_pv = hit_pv + surv * (1.0 - p_no_cross) * math.exp(-R_USD * (i + 1) * dt)
        surv = surv * p_no_cross
    cp = 1.0 if option_type == 'Call' else -1.0
    payoff = np.maximum(cp * (s[-1] - strike), 0.0)
    is_out = 'Out' in barrier_type
    weight = surv if is_out else 1.0 - surv
    return (NOTIONAL * math.exp(-R_USD * t) * float((payoff * weight).mean()),
            float(hit_pv.mean()) if is_out else math.exp(-R_USD * t) * float(surv.mean()))


def _oracle(barrier_type, barrier, **kwargs):
    """The option leg alone, which is what a rebate-free deal is worth."""
    return _oracle_legs(barrier_type, barrier, **kwargs)[0]


CASES = [(bt, at, bar) for at in ('Yes', 'No')
         for bt, bar in (('Up_And_Out', 1.40), ('Up_And_In', 1.40),
                         ('Down_And_Out', 1.12), ('Down_And_In', 1.12))]


@pytest.mark.parametrize('barrier_type,at_start,barrier', CASES,
                         ids=['%s-%s' % (bt, at) for bt, at, _ in CASES])
def test_the_closed_form_prices_to_the_independent_oracle(barrier_type, at_start, barrier):
    """Every direction and both windows against the bridge-corrected Monte Carlo. The tolerance
    carries the oracle's own error (daily bridge under flat parameters) plus ApproxBivN's ~1e-4
    CDF error scaled by notional - measured, with the worst case pinned in the module docstring."""
    v = _mtm(_run(_job(_deal(barrier_type, barrier, at_start=at_start))))
    ref = _oracle(barrier_type, barrier, at_start=at_start)
    scale = max(abs(ref), 0.02 * NOTIONAL)
    assert abs(v - ref) / scale < 2e-2, (barrier_type, at_start, v, ref)


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
def test_the_rebate_prices_to_the_independent_oracle(barrier_type, at_start, barrier, option_type):
    """`Cash_Rebate` was worth EXACTLY NOTHING: `getpartialbarrierpayoff` carried no rebate term at
    all, so at base valuation the rebate moved the mark by 0.0000 in all eight configurations while
    it is worth 14-34 on a notional of 1000 at a rebate of 50. The knock-in's only appearance was
    the expiry pad, which is why the roadmap named it - but the knock-out's pre-hit expectation was
    missing by the same omission, and both are gated here.

    The isolated leg is what this measures, not the mark: `v(R) - v(0)` against the oracle's own
    rebate leg, so the option leg's error cannot pay for a rebate error. MEASURED at R=50 over the
    eight Call configurations: worst 0.56% (Up_And_Out, start window), the rest 0.15-0.43%, and the
    SIGN is the oracle's own daily-bridge bias every time - a knock-out's hit rebate reads high
    against an oracle that misses crossings, an untouched knock-in's reads low.

    The tolerance also carries `ApproxBivN`'s ~7e-4 CDF error, which enters the end-window legs
    twice and is worth ~0.07 at this rebate."""
    deal = dict(barrier_type=barrier_type, at_start=at_start, option_type=option_type)
    v0 = _mtm(_run(_job(_deal(barrier=barrier, rebate=0.0, **deal))))
    vr = _mtm(_run(_job(_deal(barrier=barrier, rebate=REBATE, **deal))))
    opt, reb = _oracle_legs(barrier_type, barrier, at_start=at_start, option_type=option_type)
    assert abs(vr - v0 - REBATE * reb) / (REBATE * reb) < 2e-2, (
        'the rebate leg is %r where the oracle has %r' % (vr - v0, REBATE * reb))
    scale = max(abs(opt + REBATE * reb), 0.02 * NOTIONAL)
    assert abs(vr - (opt + REBATE * reb)) / scale < 2e-2, (barrier_type, at_start, vr)


#: (barrier_type, barrier, limit_days, the rebate leg's EXACT value). A start window is closed
#: once its limit date has passed, and touched once the spot is on the barrier's far side - two
#: shapes whose rebate legs are certainties with no model in them.
CERTAIN = [
    ('Up_And_Out', 1.12, LIMIT_D, REBATE), ('Up_And_In', 1.12, LIMIT_D, 0.0),
    ('Down_And_Out', 1.40, LIMIT_D, REBATE), ('Down_And_In', 1.40, LIMIT_D, 0.0),
    ('Up_And_Out', 1.12, -30, 0.0), ('Up_And_In', 1.12, -30, REBATE * math.exp(-R_USD * T)),
    ('Up_And_Out', 1.40, -30, 0.0), ('Up_And_In', 1.40, -30, REBATE * math.exp(-R_USD * T)),
    ('Down_And_Out', 1.12, -30, 0.0), ('Down_And_In', 1.12, -30, REBATE * math.exp(-R_USD * T))]


@pytest.mark.parametrize('barrier_type,barrier,limit_days,leg', CERTAIN,
                         ids=['%s-%s' % (bt, 'touched' if d > 0 else 'seasoned')
                              for bt, _, d, _ in CERTAIN])
def test_a_touched_or_a_closed_window_pays_its_rebate_exactly_once_or_not_at_all(
        barrier_type, barrier, limit_days, leg):
    """`partial_window_rebate`'s start-window forms are the LIVE side of a LIVE window and nothing
    else, and both other branches are reachable through the document.

    A spot already on the barrier's far side has touched at time zero: the knock-out's rebate is
    paid NOW and is worth exactly `Cash_Rebate`, the knock-in's can never be collected and is worth
    exactly nothing. A `Barrier_Limit_Date` in the PAST closes the window: nothing can be hit any
    more, so the knock-out's is nothing and the untouched knock-in's is a certain payment at expiry
    worth exactly `Cash_Rebate * exp(-rT)`. Neither is a Monte Carlo statement - these are
    equalities, gated to the bit-ish, and base valuation carries no touch mask at all
    (`need_spot_at_expiry` is zero), so the closed form answers alone.

    MEASURED against the unguarded forms, on a rebate of 50: the touched knock-out paid
    83.76/92.30 (up/down) where 50 is due, and the touched knock-in -30.31/-38.18 where nothing is;
    the seasoned knock-out paid 97.97/107.00 against nothing due, and the seasoned knock-in
    -44.11/-52.47 against 48.04. A survival probability read -1.05 and a discounted one-touch
    +2.14 at their worst.

    NOT this gate's subject: the OPTION leg of a start window is the same live-side-only reading
    (a seasoned Up_And_Out below its own barrier marks -6.141370 where the vanilla is 85.23), and
    that is unchanged at HEAD - a separate, older defect with its own row.
    """
    kwargs = dict(barrier_type=barrier_type, barrier=barrier, at_start='Yes',
                  limit_days=limit_days)
    v0 = _mtm(_run(_job(_deal(rebate=0.0, **kwargs))))
    vr = _mtm(_run(_job(_deal(rebate=REBATE, **kwargs))))
    assert abs((vr - v0) - leg) < 1e-9 * REBATE, (barrier_type, limit_days, vr - v0, leg)


def test_an_untouched_knock_in_ages_carrying_its_discounted_rebate():
    """The roadmap's own statement of the defect: the knock-IN's rebate entered ONLY through the
    expiry pad, so every pre-expiry row of an untouched deal marked it at zero and the profile
    stepped up to it at the last row.

    Aging is the test that can see it. A knock-in settles nothing before expiry, so under a
    risk-neutral simulation its DEFLATED profile mean must sit on the t0 mark at every row - and a
    rebate that appears only at expiry is exactly a profile that does not. MEASURED with the term
    absent: the t0 mark is 5.62 against an expiry row of 39.9, a 7x step; with it, the deflated
    profile holds the t0 mark of 39.86 across every row."""
    deal = _deal('Down_And_In', 1.12, at_start='Yes', rebate=REBATE)
    t0 = _mtm(_run(_job(deal)))
    job = _job(deal, calc={
        'Object': 'CreditMonteCarlo', 'Time_grid': '0d 12m(1m)', 'Batch_Size': 8192,
        'Simulation_Batches': 2, 'MCMC_Simulations': 1, 'Deflation_Interest_Rate': 'USD'})
    md = job['Calc']['MergeMarketData']['ExplicitMarketData']
    md['Price Models'] = {'GBMAssetPriceModel.EUR': {'Vol': SIGMA, 'Drift': R_USD - R_EUR}}
    md['Model Configuration'] = {'.ModelParams': {
        'modeldefaults': {'FxRate': 'GBMAssetPriceModel'}, 'modelfilters': {}}}
    profile = _run(job)['Results']['mtm']
    days = np.array([(d - BASE).days for d in profile.index], dtype=float)
    deflated = np.asarray(profile, float).mean(axis=1) * np.exp(-R_USD * days / 365.0)
    assert np.abs(deflated - t0).max() / abs(t0) < 3e-2, (t0, deflated, list(days))


def test_an_unreachable_barrier_is_the_vanilla_and_a_certain_one_is_nothing():
    """The two degenerate anchors: a KO whose barrier no path reaches is the plain European
    (engine-vs-engine against an FXOptionDeal document), and its KI is worth nothing."""
    vanilla = {'Object': 'FXOptionDeal', 'Reference': 'VAN', 'Currency': 'USD',
               'Underlying_Currency': 'EUR', 'Payoff_Currency': 'USD', 'Discount_Rate': 'USD',
               'FX_Volatility': 'EUR.USD', 'Buy_Sell': 'Buy', 'Option_Type': 'Call',
               'Strike_Price': 1.25, 'Underlying_Amount': NOTIONAL,
               'Expiry_Date': BASE + pd.DateOffset(days=EXPIRY_D),
               'Option_Style': 'European', 'Settlement_Style': 'Cash'}
    v_van = _mtm(_run(_job(vanilla)), 'VAN')
    for at_start in ('Yes', 'No'):
        ko = _mtm(_run(_job(_deal('Up_And_Out', 5.0, at_start=at_start))))
        ki = _mtm(_run(_job(_deal('Up_And_In', 5.0, at_start=at_start))))
        assert abs(ko - v_van) / v_van < 2e-3, (at_start, ko, v_van)
        assert abs(ki) < 2e-3 * v_van, (at_start, ki)


def test_rows_past_the_limit_date_are_finite_and_vanilla():
    """The NaN the roadmap suspected: past Barrier_Limit_Date `limit` is negative and reaches
    `sqrt` unclamped, so every CMC row between the limit and expiry priced NaN. Past the limit a
    start-window deal IS a vanilla on its surviving paths - the profile must be finite
    everywhere and the post-limit rows must track the vanilla document's profile."""
    def cmc(deal):
        job = _job(deal, calc={
            'Object': 'CreditMonteCarlo', 'Time_grid': '0d 12m(1m)', 'Batch_Size': 1024,
            'Simulation_Batches': 2, 'MCMC_Simulations': 1, 'Deflation_Interest_Rate': 'USD'})
        md = job['Calc']['MergeMarketData']['ExplicitMarketData']
        md['Price Models'] = {'GBMAssetPriceModel.EUR': {'Vol': SIGMA, 'Drift': 0.0}}
        md['Model Configuration'] = {'.ModelParams': {
            'modeldefaults': {'FxRate': 'GBMAssetPriceModel'}, 'modelfilters': {}}}
        return _run(job)['Results']['mtm']
    profile = cmc(_deal('Up_And_Out', 1.40, at_start='Yes'))
    values = np.asarray(profile, dtype=float)
    assert np.isfinite(values).all(), 'NaN rows past the barrier window'


def test_a_touch_outside_the_window_does_not_knock():
    """The window is the deal: a start-window that closes after two days leaves ten months in
    which the spot can cross the barrier with no contractual effect - the deal is a vanilla on
    virtually every path. The engine monitored EVERY interval, so mid-life crossings knocked a
    barrier that no longer existed and the profile collapsed on exactly those paths. Post-fix
    the partial document's profile tracks the vanilla document's row for row."""
    def profile_of(deal):
        job = _job(deal, calc={
            'Object': 'CreditMonteCarlo', 'Time_grid': '0d 12m(1m)', 'Batch_Size': 2048,
            'Simulation_Batches': 2, 'MCMC_Simulations': 1, 'Deflation_Interest_Rate': 'USD'})
        md = job['Calc']['MergeMarketData']['ExplicitMarketData']
        md['Price Models'] = {'GBMAssetPriceModel.EUR': {'Vol': SIGMA, 'Drift': 0.0}}
        md['Model Configuration'] = {'.ModelParams': {
            'modeldefaults': {'FxRate': 'GBMAssetPriceModel'}, 'modelfilters': {}}}
        return _run(job)['Results']['mtm']
    partial = profile_of(_deal('Up_And_Out', 1.35, at_start='Yes', limit_days=2))
    vanilla = profile_of({'Object': 'FXOptionDeal', 'Reference': 'PB', 'Currency': 'USD',
                          'Underlying_Currency': 'EUR', 'Payoff_Currency': 'USD',
                          'Discount_Rate': 'USD', 'FX_Volatility': 'EUR.USD', 'Buy_Sell': 'Buy',
                          'Option_Type': 'Call', 'Strike_Price': 1.25,
                          'Underlying_Amount': NOTIONAL,
                          'Expiry_Date': BASE + pd.DateOffset(days=EXPIRY_D),
                          'Option_Style': 'European', 'Settlement_Style': 'Cash'})
    assert np.isfinite(np.asarray(partial, float)).all()
    # the discriminator is the EXPIRY row: pre-fix, mid-life crossings (P ~ 55% at this barrier
    # by 12m) knocked a window that closed after two days, collapsing realised payoffs; post-fix
    # the row matches the vanilla document's within Monte Carlo noise (the two documents' grids
    # draw different paths, so the comparison is of means, not paths)
    a = float(np.asarray(partial.iloc[-1], float).mean())
    v = float(np.asarray(vanilla.iloc[-1], float).mean())
    assert abs(a - v) / v < 8e-2, (a, v)


@pytest.mark.parametrize('barrier_type,at_start,barrier', [
    ('Up_And_Out', 'Yes', 1.40), ('Up_And_Out', 'No', 1.40), ('Down_And_In', 'No', 1.20)],
    ids=['KO-start', 'KO-end', 'KI-end'])
def test_the_deal_ages_as_a_martingale(barrier_type, at_start, barrier):
    """The aging statement: under a risk-neutral simulation (model drift = the carry) the
    DEFLATED profile mean must sit on the t0 mark at every row - across the window edge, through
    the knock transitions the bridge probability accumulates, and onto the expiry settlement. Any
    aging defect (a window monitored outside itself, a state dropped at the limit date, a
    mis-bridged knock) shows up as decay or drift in what must be flat."""
    t0 = _mtm(_run(_job(_deal(barrier_type, barrier, at_start=at_start))))
    job = _job(_deal(barrier_type, barrier, at_start=at_start), calc={
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
# static. Report the book off a THIRD currency and USD is simulated too, the rate is absent, every
# interval variance is zero, and `barrier_touched` collapses from a Brownian-bridge probability to
# a 0/1 endpoint test - the one form of this decision that is a latch.
def _cva_job(deal, spot=X0, gradient=False, bridge=True, hessian=False,
             batch=8192, batches=4, bandwidth=0.01, window_touch='Yes'):
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
            'Boundary_AAD_Window_Touch': window_touch,
            'Credit_Valuation_Adjustment': {
                'Calculate': 'Yes', 'Counterparty': 'CPTY', 'Deflate_Stochastically': 'No',
                'Stochastic_Hazard_Rates': 'No', 'Gradient': 'Yes' if gradient else 'No',
                'Hessian': 'Yes' if hessian else 'No'}},
        'Deals': {'Tag_Titles': '', 'Reference': 'pb',
                  'Deals': {'Children': [{'Instrument': {'.Deal': deal}}]}},
        'MergeMarketData': {'MarketDataFile': '', 'ExplicitMarketData': {
            'System Parameters': {'Base_Currency': base, 'Base_Date': BASE},
            'Valuation Configuration': {}, 'Price Factors': factors, 'Price Models': models,
            'Model Configuration': {'.ModelParams': {
                'modeldefaults': {'FxRate': 'GBMAssetPriceModel'}, 'modelfilters': {}}}}}}}


def _cva(**kwargs):
    """(cva, dCVA/d(EUR spot) or None). The bumped runs change one factor value and nothing else,
    so common random numbers arrive through the contract rather than through a patch."""
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


@pytest.mark.parametrize('bridge', [True, False], ids=['bridge', 'endpoints'])
def test_the_window_touch_decision_registers_only_where_it_is_a_latch(bridge):
    """One decision, ONE estimator - and which estimator depends on whether the bridge is there.

    `barrier_touched` has two branches. With an interval variance it returns a Brownian-bridge
    PROBABILITY that is continuous in the spot: an endpoint beyond the barrier makes the bridge
    term exactly one, so the two agree at the crossing and ordinary AAD already carries the flux.
    Without one it returns `(prev + beyond).clip(max=1)`, a 0/1 indicator with zero derivative
    almost everywhere - a latch, and the only form that needs a `LatchedBoundarySet`. Registering
    on the bridge branch as well would count the same decision twice.

    The seam that says which happened, with nothing patched: a CVA Hessian is refused BY NAME on a
    book that registered a boundary correction, and that refusal is raised strictly before the
    exposure kink term is built. Both branches refuse here - a knocked-out path is worth exactly
    zero, so this deal puts an ATOM at the exposure kink either way - but they refuse for different
    reasons and say so, which makes the message the exact reading. The 'register on both' and
    'register on neither' mutants each swap one branch's message for the other's.

    MEASURED on the bridge branch, which is why that half is a design statement and not a defect:
    dCVA/dspot reads -2.742147 against a CRN ladder flat to 1.02% whose best rung is -2.730047,
    0.44% apart, with no registration at all."""
    job = _cva_job(LATCH_DEAL, gradient=True, bridge=bridge, hessian=True, batch=512, batches=1)
    with pytest.raises(utils.SecondOrderRefused) as refusal:
        _run(job)
    message = str(refusal.value)
    if bridge:
        assert 'ATOM of exposure' in message, (
            'the bridge branch registered a boundary correction, so the same decision is being '
            'estimated twice - once by the bridge probability and once by the kernel: ' + message)
    else:
        assert 'registered a boundary correction' in message, (
            'the endpoint branch is a 0/1 latch and registered nothing: ' + message)


def test_the_window_touch_registration_is_opt_in_and_off_by_default():
    """It decides the SIGN of the reported delta on a magnitude nobody has established, so it does
    not ship live: `Boundary_AAD_Window_Touch` defaults to No and the pricer registers nothing.

    Two readings, both exact. The Hessian refusal on the endpoint branch says `ATOM of exposure`
    with the switch off - the same message the bridge branch gives, i.e. nothing was registered -
    and the reported gradient there is BIT-IDENTICAL to the same run with the correction suppressed
    through `Boundary_AAD_Bandwidth`, which is what "registers nothing" means as a number.

    WHY IT IS NOT ON. At 16384 paths the CRN ladder over h = 1e-4..1e-2 reads -0.741, -1.027,
    -0.909, -0.683, -0.785, -0.522, -0.585: a spread of 68% of its own median, against a corrected
    -1.8134911 (77% out) and a suppressed +0.7865276. Every rung there is negative and only the
    corrected delta shares their sign - but at 8192 paths the ladder is not even sign-unanimous
    (+1.246, -1.214, -1.087), so the sign is not a gateable statement either.
    """
    job = _cva_job(LATCH_DEAL, gradient=True, bridge=False, hessian=True, batch=512, batches=1,
                   window_touch='No')
    with pytest.raises(utils.SecondOrderRefused) as refusal:
        _run(job)
    assert 'ATOM of exposure' in str(refusal.value), (
        'the default registered a boundary correction: ' + str(refusal.value))

    kw = dict(deal=LATCH_DEAL, gradient=True, bridge=False, batch=1024, batches=1)
    off = _cva(window_touch='No', **kw)[1]
    suppressed = _cva(bandwidth=1e-12, **kw)[1]
    assert off == suppressed, (off, suppressed)


def test_asking_for_the_partial_barrier_sensitivities_does_not_move_the_exposure():
    """BIT-identical, not approximately: the correction is `gap - gap.detach()`, worth exactly zero
    in the forward pass, so any drift means the registration path perturbed the valuation instead
    of observing it. Run on the endpoint branch, the one that registers at all.

    THE MAGNITUDE OF THE TERM IT GATES, and the declared limit on it. On this fixture at 32768
    paths the reported dCVA/dspot reads -2.2467692 with the registration and +0.5207422 without -
    the sign itself is what the flux decides. Its own bandwidth ladder is FLAT: -2.27999, -2.24677,
    -2.12172, -2.02104 over 0.005 to 0.05, a 12% spread over a factor of ten, and the reading moves
    2.4% between 4096 and 32768 paths.

    WHAT IS NOT ESTABLISHED IS THE MAGNITUDE, and no gate here pretends otherwise. The CRN oracle
    does not converge on this fixture: over h = 1e-4 to 1e-2 at 32768 paths it reads -1.141,
    -0.911, -0.803, -1.466, -1.034, -0.774, -0.661 - a spread of 88% of its own median, scattering
    rather than refining, which is what differencing across a genuine discontinuity looks like.
    Against its closest rung the corrected delta is 35% out and the suppressed one 227%. This deal
    prices on THREE rows (see the module docstring), so the whole window carries at most two
    decisions and exactly one of them is live - there is no fixture of this instrument with more,
    which is why the ladder cannot be tightened by raising paths alone. WHICH IS WHY THE
    REGISTRATION IS OPT-IN: this gate asks for it by name through `Boundary_AAD_Window_Touch`, and
    the shipped default registers nothing (the gate above)."""
    off, _ = _cva(deal=LATCH_DEAL, bridge=False, batch=1024, batches=1)
    on, grad = _cva(deal=LATCH_DEAL, gradient=True, bridge=False, batch=1024, batches=1)
    assert off == on, 'the exposure moved when sensitivities were requested: %r -> %r' % (off, on)
    assert grad is not None and abs(grad) > 0.0, 'no EUR spot gradient was reported at all'
