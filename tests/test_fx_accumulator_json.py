"""FXAccumulatorOptionDeal end to end, through the JSON contract and nothing else.

THE FORM: a test is a job JSON run through the public surface (`Context.load_json` + `run_job`),
whose expected answer is decided BEFORE the run from a closed form this engine has no part in. No
`Config` is built by hand, no internal constructor imported, nothing patched.

The deal is authored to have a closed form: with the barrier out of reach an accumulator is a STRIP
OF EUROPEANS - long `Underlying_Amount` of the ITM side, short `LeverageNotional` of the OTM one at
each fixing, each paid at its own settlement date. So the value AND the FX delta are Black, and the
greek is checked as its own statement rather than for being finite.

The USD curve is STEEP with a knot at every tenor the deal reads, because a flat curve is a
degeneracy: the interval carry strip and the raw zero-rate gather agree wherever the curve is flat,
so a pricer differencing neither would pass. The reference reads the same curve with `numpy.interp`,
which the knots make exact. Every constant is re-derived at import off the loaded document, so the
assertions follow `tests/fixtures/fx_accumulator_job.json` rather than a transcription of it.

TOLERANCES ARE MEASURED. The seed-to-seed spread of the reported value across five seeds is 0.210
at 16,384 inner paths, 0.148 at 65,536 and 0.059 at 262,144, so the gate runs at 262,144 and allows
1%, about five times the measured sd. The first draft ran at 16,384 and allowed 0.5% - INSIDE one
standard error - and passed on CPU by luck while failing on GPU, which is what produced the table.
"""
import json
import math
import os
import sys

# reference-derivus shadow-import guard (MEMORY): pin the package under test to THIS repo.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import derivus as rf

TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'fixtures', 'fx_accumulator_job.json')


def _template():
    with open(TEMPLATE) as f:
        return json.load(f)


def _deal_of(job):
    return job['Calc']['Deals']['Deals']['Children'][0]['Instrument']['.Deal']


# every constant the reference needs is READ OUT OF THE DOCUMENT, so edit the template and the
# closed form follows it
_T = _template()
_D = _deal_of(_T)
_PF = _T['Calc']['MergeMarketData']['ExplicitMarketData']['Price Factors']

BASE = _T['Calc']['Calculation']['Base_Date']['.Timestamp']
SIMS = _T['Calc']['Calculation']['MCMC_Simulations']
SPOT = _PF['FxRate.EUR']['Spot']
STRIKE = _D['Strike_Price']
N1, N2 = _D['Underlying_Amount'], _D['LeverageNotional']
USD_CURVE = _PF['InterestRate.USD']['Curve']['.Curve']['data']
Q_EUR = _PF['InterestRate.EUR']['Curve']['.Curve']['data'][0][1]
SIGMA = _PF['FXVol.EUR.USD']['Surface']['.Curve']['data'][0][2]
DAYS = 365.0
TOL = 1e-2


def _offset(stamp):
    import datetime
    d = datetime.date.fromisoformat(stamp)
    return (d - datetime.date.fromisoformat(BASE)).days


FIXINGS = [(_offset(r[0]['.Timestamp']), _offset(r[1]['.Timestamp']))
           for r in _D['Accumulator_ExpiryDates']]


def _r_usd(t):
    import numpy as _np
    ts, vs = zip(*USD_CURVE)
    return float(_np.interp(t, ts, vs))


# --------------------------------------------------------------------------------------------
# the expectation, decided before the run - textbook Black, no engine code
# --------------------------------------------------------------------------------------------
def _ndtr(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _expected(spot=SPOT):
    """Value and dV/dspot of the strip of Europeans the deal degenerates to."""
    value = delta = 0.0
    for fix, settle in FIXINGS:
        t, ts = fix / DAYS, settle / DAYS
        fwd_factor = math.exp((_r_usd(t) - Q_EUR) * t)
        F = spot * fwd_factor
        sd = SIGMA * math.sqrt(t)
        D = math.exp(-_r_usd(ts) * ts)
        d1 = (math.log(F / STRIKE) + 0.5 * sd * sd) / sd
        d2 = d1 - sd
        call = F * _ndtr(d1) - STRIKE * _ndtr(d2)
        put = STRIKE * _ndtr(-d2) - F * _ndtr(-d1)
        value += D * (N1 * call - N2 * put)
        # dC/dF = N(d1), dP/dF = -N(-d1); dF/dspot is the forward factor
        delta += D * fwd_factor * (N1 * _ndtr(d1) - N2 * (-_ndtr(-d1)))
    return value, delta


EXPECTED_VALUE, EXPECTED_DELTA = _expected()


# --------------------------------------------------------------------------------------------
# the job document
# --------------------------------------------------------------------------------------------
def _job(greeks='No', **deal_overrides):
    """The canonical document, varied: switch coverage is overrides on ONE template rather than a
    second hand-built document that can quietly stop resembling it."""
    job = _template()
    job['Calc']['Calculation']['Greeks'] = greeks
    _deal_of(job).update(deal_overrides)
    return job


def _day(offset):
    import datetime
    return (datetime.date(2024, 6, 28) + datetime.timedelta(days=offset)).isoformat()


def _run(job, tmp_path, name='acc'):
    """JSON in, results out. `load_json` + `run_job` is the whole public surface."""
    path = os.path.join(str(tmp_path), f'{name}.json')
    with open(path, 'w') as f:
        json.dump(job, f, default=str)
    cx = rf.Context()
    cx.load_json(path)
    _, out = cx.run_job()
    return out


def _mtm(out, ref='ACC1'):
    rows = out['Results']['mtm']
    rows = rows[rows['Reference'] == ref]
    return float(rows['Value'].iloc[0])


# --------------------------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------------------------
def test_the_job_document_prices_the_expected_value(tmp_path):
    """The whole framework, from the document to the number, against Black decided beforehand."""
    v = _mtm(_run(_job(), tmp_path))
    assert abs(v - EXPECTED_VALUE) / abs(EXPECTED_VALUE) < TOL, (v, EXPECTED_VALUE)


def test_the_job_document_reports_the_expected_fx_delta(tmp_path):
    """The greek is a statement of its own and not a finiteness check: a strip of Europeans has a
    closed-form delta, so `dV/d(EUR.USD spot)` is known before the run. A value that is right with
    a delta that is wrong is the failure mode a price-only gate cannot see.
    """
    out = _run(_job(greeks='First'), tmp_path)
    assert abs(_mtm(out) - EXPECTED_VALUE) / abs(EXPECTED_VALUE) < TOL
    frame = out['Results']['Greeks_First']
    column = [c for c in frame.columns if c != 'Value'][0]
    index, = [i for i in frame.index if str(i[0]) == 'FxRate.EUR']
    delta = float(frame.loc[index, column])
    assert abs(delta - EXPECTED_DELTA) / abs(EXPECTED_DELTA) < 2e-2, (delta, EXPECTED_DELTA)


def test_the_declared_switches_flip_the_sign_they_should(tmp_path):
    """`Buy_Sell` and `Option_Type` are declared switches, exercised through the document: selling
    mirrors exactly, and a put on this strip is the ITM and OTM legs swapped."""
    buy = _mtm(_run(_job(), tmp_path, 'buy'))
    sell = _mtm(_run(_job(Buy_Sell='Sell'), tmp_path, 'sell'))
    assert abs(buy + sell) <= 1e-9 * abs(buy), (buy, sell)
    put = _mtm(_run(_job(Option_Type='Put', Barrier_Type='Down_And_Out',
                         Barrier_Price=0.01), tmp_path, 'put'))
    value = 0.0
    for fix, settle in FIXINGS:
        t, ts = fix / DAYS, settle / DAYS
        F = SPOT * math.exp((_r_usd(t) - Q_EUR) * t)
        sd = SIGMA * math.sqrt(t)
        d1 = (math.log(F / STRIKE) + 0.5 * sd * sd) / sd
        d2 = d1 - sd
        c = F * _ndtr(d1) - STRIKE * _ndtr(d2)
        p = STRIKE * _ndtr(-d2) - F * _ndtr(-d1)
        value += math.exp(-_r_usd(ts) * ts) * (N1 * p - N2 * c)
    assert abs(put - value) / abs(value) < 5e-3, (put, value)


def test_a_declared_dead_deal_is_worth_nothing(tmp_path):
    """`Barrier_Hit` is a declared switch and its whole meaning is that the deal has ended."""
    assert _mtm(_run(_job(Barrier_Hit='Yes'), tmp_path, 'dead')) == 0.0


# --------------------------------------------------------------------------------------------
# the knock-out itself, against a law the engine has no part in
# --------------------------------------------------------------------------------------------
def _brute_force_ko(barrier, barrier_up, n_paths=400_000, seed=7):
    """`(value, standard error)` for exact-law GBM at the fixing dates with the indicator
    knock-out - the product the analytic survival truncation must reproduce."""
    import numpy as np
    rng = np.random.default_rng(seed)
    times = np.array([f / DAYS for f, _ in FIXINGS])
    settles = np.array([s / DAYS for _, s in FIXINGS])
    dt = np.diff(np.concatenate([[0.0], times]))
    z = rng.standard_normal((n_paths // 2, len(times)))
    z = np.concatenate([z, -z], axis=0)
    # the carry over each INTERVAL is a difference of cumulative integrals and not one tenor's rate
    # times the interval - on this steep curve the two differ by whole percent
    cum = np.array([(_r_usd(t) - Q_EUR) * t for t in times])
    drift = np.diff(np.concatenate([[0.0], cum])) - 0.5 * SIGMA ** 2 * dt
    s = np.exp(np.log(SPOT) + np.cumsum(drift + SIGMA * np.sqrt(dt) * z, axis=1))
    survive = (s < barrier) if barrier_up else (s > barrier)
    alive_in = np.concatenate([np.ones((len(s), 1), dtype=bool),
                               np.cumprod(survive, axis=1)[:, :-1].astype(bool)], axis=1)
    pv = ((alive_in & survive) * (N1 * np.maximum(s - STRIKE, 0.0) -
                                  N2 * np.maximum(STRIKE - s, 0.0)) *
          np.exp(-np.array([_r_usd(x) for x in settles]) * settles)).sum(axis=1)
    return float(pv.mean()), float(pv.std() / math.sqrt(len(pv)))


import pytest


@pytest.mark.parametrize('barrier,up,btype', [(1.20, True, 'Up_And_Out'),
                                              (1.02, False, 'Down_And_Out')])
def test_the_knock_out_matches_an_exact_law_simulation(barrier, up, btype, tmp_path):
    """Both declared `Barrier_Type` values against an independent indicator simulation: the pricer
    integrates survival analytically and draws from the truncated law, the reference simulates and
    applies the indicator - the same product by two routes sharing no code.
    """
    v = _mtm(_run(_job(Barrier_Type=btype, Barrier_Price=barrier), tmp_path, f'ko{barrier}'))
    ref, se = _brute_force_ko(barrier, up)
    assert abs(v - ref) < max(4.0 * se, 5e-3 * abs(ref)), (v, ref, se)


def test_an_observed_fixing_is_its_exact_payoff(tmp_path):
    """A document whose fixings are all in the past, settling later: no simulation is left and
    the value is arithmetic, so this asserts to 1e-9 rather than to a Monte Carlo tolerance."""
    obs = [1.15, 1.08]
    job = _job()
    job['Calc']['Calculation']['Base_Date'] = {'.Timestamp': _day(200)}
    job['Calc']['MergeMarketData']['ExplicitMarketData'][
        'System Parameters']['Base_Date'] = {'.Timestamp': _day(200)}
    deal = job['Calc']['Deals']['Deals']['Children'][0]['Instrument']['.Deal']
    deal['Accumulator_ExpiryDates'] = [
        [{'.Timestamp': _day(f)}, {'.Timestamp': _day(300 + i)}, v]
        for i, ((f, _), v) in enumerate(zip(FIXINGS, obs))]
    v = _mtm(_run(job, tmp_path, 'observed'))
    ref = sum(math.exp(-_r_usd((300 + i - 200) / DAYS) * (300 + i - 200) / DAYS) *
              (N1 * max(x - STRIKE, 0.0) - N2 * max(STRIKE - x, 0.0))
              for i, x in enumerate(obs))
    assert abs(v - ref) <= 1e-9 * max(1.0, abs(ref)), (v, ref)


def test_a_settled_fixing_beyond_the_barrier_kills_the_deal(tmp_path):
    """Knock-out state carried in from before the base date is DATA: a settled fixing whose
    recorded value breached ends the deal even when `Barrier_Hit` was not set."""
    job = _job(Barrier_Price=1.14)
    deal = job['Calc']['Deals']['Deals']['Children'][0]['Instrument']['.Deal']
    deal['Accumulator_ExpiryDates'] = [
        [{'.Timestamp': _day(-21)}, {'.Timestamp': _day(-19)}, 1.15]] + \
        deal['Accumulator_ExpiryDates']
    assert _mtm(_run(job, tmp_path, 'breached')) == 0.0


def test_a_blank_barrier_is_refused_rather_than_priced_at_zero(tmp_path):
    """`Barrier_Price` is `default=REQUIRED`, and a document omitting it anyway must refuse: left at
    0.0 the deal survives NO fixing and prices to a silent scalar zero, a whole trade vanishing
    from a book with no error. BOTH directions, because at 0.0 a `Down_And_Out` is merely
    unbarriered and prices its full strip, so a one-direction fixture scores the hole harmless.
    """
    for btype in ('Up_And_Out', 'Down_And_Out'):
        job = _job(Barrier_Type=btype, Barrier_Price=0.0)
        out = _run(job, tmp_path, f'blank{btype}')
        rows = out['Results']['mtm']
        rows = rows[rows['Reference'] == 'ACC1']
        assert rows.empty or float(rows['Value'].iloc[0]) == 0.0, btype


def test_a_fixing_dated_today_follows_the_price_factor(tmp_path):
    """A fixing dated ON the base date reads the SIMULATED spot and not the value the trade
    recorded, which is what makes a sensitivity correct: bumping `FxRate.EUR` has to move that
    fixing's payoff and cannot if the fixing is a constant.

    A VALUE assertion cannot tell the two designs apart - the declared print and the current spot
    are the same number today - so the discriminating statement is the DELTA. One fixing today,
    struck in the money: the deal is worth `N1 * (S - K)` discounted and its spot delta is `N1 * D`
    exactly, where a constant fixing would report ZERO.
    """
    strike, settle_day = 1.05, 3                      # ITM: the OTM leg contributes nothing
    job = _job(greeks='First', Strike_Price=strike, Accumulator_ExpiryDates=[
        [{'.Timestamp': BASE}, {'.Timestamp': _day(settle_day)}, 0.0]])
    out = _run(job, tmp_path, 'today')
    D = math.exp(-_r_usd(settle_day / DAYS) * settle_day / DAYS)

    value = _mtm(out)
    assert abs(value - N1 * (SPOT - strike) * D) < 1e-6, (value, N1 * (SPOT - strike) * D)

    frame = out['Results']['Greeks_First']
    column = [c for c in frame.columns if c != 'Value'][0]
    index, = [i for i in frame.index if str(i[0]) == 'FxRate.EUR']
    delta = float(frame.loc[index, column])
    assert abs(delta - N1 * D) < 1e-3 * N1 * D, (
        delta, N1 * D, 'the today-fixing does not follow the price factor, so a bump cannot '
                       'reach it and the reported delta is short that whole fixing')


def test_the_deal_tags_reach_the_reported_row(tmp_path):
    """`Tags` carries the book-keeping and `Tag_Titles` names the columns it lands in. Nothing in
    pricing reads either, which is the reason to assert them: a field no pricer touches is one
    whose round trip nothing else can notice going wrong, and a desk sorts its blotter by it.
    """
    out = _run(_job(), tmp_path, 'tags')
    rows = out['Results']['mtm']
    rows = rows[rows['Reference'] == 'ACC1']
    assert 'Portfolio' in rows.columns and 'Trader' in rows.columns, list(rows.columns)
    assert rows['Portfolio'].iloc[0] == 'FX-EXOTICS', rows['Portfolio'].iloc[0]
    assert rows['Trader'].iloc[0] == 'jdoe', rows['Trader'].iloc[0]


def _long_lag_job(greeks='No', spot=None, sims=1 << 16):
    """The dead-branch document: settlements lag their fixings by MORE than the fixing spacing, so a
    knocked deal carries several fixed-but-unsettled payoffs - the pending head - through every
    reporting row between the knock and the last landing. A live barrier makes the latch fire on
    real paths, which routes those rows through the registration's dead branch."""
    job = _job(greeks=greeks, Barrier_Price=1.18, Accumulator_ExpiryDates=[
        [{'.Timestamp': _day(30 + 20 * i)}, {'.Timestamp': _day(30 + 20 * i + 45)}, 0.0]
        for i in range(6)])
    job['Calc']['Calculation']['MCMC_Simulations'] = sims
    if spot is not None:
        job['Calc']['MergeMarketData']['ExplicitMarketData'][
            'Price Factors']['FxRate.EUR']['Spot'] = spot
    return job


def _cva_long_lag(gradient='No', spot=None):
    """The long-lag document as a credit Monte Carlo with CVA on: outer rows observe fixings, the
    latch registration carries real dead cells, and the boundary correction scores the knock jumps -
    the only place the dead branch is ever read."""
    job = _long_lag_job(spot=spot, sims=256)
    job['Calc']['Calculation'] = {
        'Object': 'CreditMonteCarlo', 'Base_Date': {'.Timestamp': BASE}, 'Currency': 'USD',
        'Time_grid': '0d 6m(10d)', 'Batch_Size': 1024, 'Simulation_Batches': 4,
        'Random_Seed': 1, 'MCMC_Simulations': 256, 'Deflation_Interest_Rate': 'USD',
        'Credit_Valuation_Adjustment': {
            'Calculate': 'Yes', 'Counterparty': 'CPTY', 'Deflate_Stochastically': 'No',
            'Stochastic_Hazard_Rates': 'No', 'Gradient': gradient}}
    market = job['Calc']['MergeMarketData']['ExplicitMarketData']
    market['Price Models'] = {'GBMAssetPriceModel.EUR': {'Vol': SIGMA, 'Drift': 0.0}}
    market['Model Configuration'] = {'.ModelParams': {
        'modeldefaults': {'FxRate': 'GBMAssetPriceModel'}, 'modelfilters': {}}}
    market['Price Factors']['SurvivalProb.CPTY'] = {
        'Recovery_Rate': 0.4,
        'Curve': {'.Curve': {'meta': [], 'data': [[0.0, 0.0], [10.0, 0.4]]}}}
    return job


def test_a_knocked_deal_still_carries_its_pending_settlements(tmp_path):
    """The dead branch is NOT zero: a knocked deal keeps the payoffs of fixings it survived whose
    settlements have not landed - the pending head - through every reporting row between the knock
    and the last landing, and only a gradient run reads those cells.

    The DISCRIMINATOR is the `ACCUMULATOR_LATCH` organ: the registration reconstructed at the
    booked flags against the engine's reported rows. With the head carried the residual is float
    roundoff while the head itself is material. The CVA delta against its CRN ladder rides along as
    an economic sanity bound rather than the gate.
    """
    import io as _io
    import logging as _logging
    buf, root = _io.StringIO(), _logging.getLogger()
    handler, old = _logging.StreamHandler(buf), root.level
    root.addHandler(handler)
    root.setLevel(_logging.DEBUG)
    try:
        out = _run(_cva_long_lag(gradient='Yes'), tmp_path, 'aad')
    finally:
        root.removeHandler(handler)
        root.setLevel(old)
    organs = [ln for ln in buf.getvalue().splitlines() if 'ACCUMULATOR_LATCH' in ln]
    assert organs, 'the latch reconstruction logged nothing at DEBUG'
    parse = lambda ln, key: float(ln.split(key + '=')[1].split()[0])
    recon = max(parse(ln, 'recon_max') for ln in organs)
    head = max(parse(ln, 'head_max') for ln in organs)
    scale = max(parse(ln, 'scale') for ln in organs)
    # MEASURED: head_max 444 on a 1600 profile scale (28%) and the residual 7.6e-8 relative, which
    # is float32 roundoff, so the bound carries 13x. MUTATION: the dead branch zeroed reads 156,
    # nearly 100,000x over the bound.
    assert head > 1e-3 * scale, 'the document must carry a material pending head'
    assert recon < 1e-6 * scale, (recon, head, scale)

    g = out['Results']['grad_cva']['Gradient']
    aad = float(g.loc[[i for i in g.index if 'FxRate.EUR' in str(i[0])][0]])
    cva = float(out['Results']['cva'])
    crn = []
    for h in (0.002, 0.004):
        up = float(_run(_cva_long_lag(spot=SPOT + h), tmp_path, f'u{h}')['Results']['cva'])
        dn = float(_run(_cva_long_lag(spot=SPOT - h), tmp_path, f'd{h}')['Results']['cva'])
        crn.append((up - dn) / (2.0 * h))
    best = min(crn, key=lambda c: abs(aad - c))
    assert abs(aad - best) / abs(best) < 5e-2, (aad, crn, cva)


# --------------------------------------------------------------------------------------------
# THE RECIPROCAL AXIS: one law per pair, and which side of it the deal is on.
#
# `FxRate.<ccy>` is that currency priced in the BASE, so on a USD-base book only `.EUR` is a rate
# the engine can simulate and only `.EUR` is a block the calibration can write. A deal whose
# `Underlying_Currency` IS the base pays on the RECIPROCAL of the fitted axis and settles in the
# other currency - a change of NUMERAIRE as well as of axis - and the fitted law carries to it
# exactly at the cost of one parameter (`utils.hn_reciprocal_gamma`).
#
# The deal below is the template's mirror: USD notional, EUR settlement, the same schedule and
# barrier out of reach, so it is still a strip of Europeans - on `1/S`.
# --------------------------------------------------------------------------------------------
SPY = 252.0

#: The daily variance of a 10% flat vol and the DEGENERATE Heston-Nandi holding it: no ARCH, no
#: leverage, no GARCH memory, so `h` is that number for all t and the strip has a Black closed form
#: on whichever axis the law is carried to - the only fixture that can PIN the AXIS exactly. At
#: `Alpha: 0.0` the leverage term drops out of `hn_variance_step`, so `hn_reciprocal_gamma` has no
#: effect on it and the carry is pinned elsewhere.
SIGMA_HN = 0.10
H_DAILY = SIGMA_HN ** 2 / DAYS
DEGENERATE = {'Property_Aliases': None, 'Omega': H_DAILY, 'Alpha': 0.0, 'Beta': 0.0,
              'Gamma_Star': 0.0, 'H0': H_DAILY}

#: A calibrated-looking fit carrying the LEVERAGE the degenerate one cannot: what the change of
#: numeraire is worth is read off this one.
LEVERED = {'Property_Aliases': None, 'Omega': 1e-12, 'Alpha': 2.0e-6, 'Beta': 0.45,
           'Gamma_Star': -474.34, 'H0': 7.8e-5}

#: The deal's own axis - EUR per USD, the reciprocal of the `FxRate.EUR` the engine simulates.
RECIPROCAL_SPOT = 1.0 / SPOT
RECIPROCAL_STRIKE = round(RECIPROCAL_SPOT, 4)


def _reciprocal_job(factor=DEGENERATE, model='HestonNandi', sims=None, **calc):
    """The template mirrored onto the base currency, under a declared spot model."""
    job = _template()
    market = job['Calc']['MergeMarketData']['ExplicitMarketData']
    _deal_of(job).update({
        'Currency': 'EUR', 'Underlying_Currency': 'USD', 'Discount_Rate': 'EUR',
        'Strike_Price': RECIPROCAL_STRIKE, 'Barrier_Type': 'Up_And_Out', 'Barrier_Price': 5.0})
    market['Price Factors']['{}ModelParameters.EUR'.format(model)] = dict(factor)
    market['Valuation Configuration'] = {'FXAccumulatorOptionDeal': {'SpotModel': model}}
    if sims:
        job['Calc']['Calculation']['MCMC_Simulations'] = sims
    job['Calc']['Calculation'].update(calc)
    return job


def _reciprocal_expected():
    """Black on the DEAL's own axis: the forward at its own carry (`r_EUR - r_USD`) and the variance
    the daily recursion accumulates. No reciprocal appears in it, which is the claim."""
    value, prev_t, days = 0.0, 0.0, 0
    for fix, settle in FIXINGS:
        t, ts = fix / DAYS, settle / DAYS
        days += max(int(round((t - prev_t) * SPY)), 1)
        prev_t = t
        F = RECIPROCAL_SPOT * math.exp((Q_EUR - _r_usd(t)) * t)
        sd = math.sqrt(days * H_DAILY)
        d1 = (math.log(F / RECIPROCAL_STRIKE) + 0.5 * sd * sd) / sd
        d2 = d1 - sd
        call = F * _ndtr(d1) - RECIPROCAL_STRIKE * _ndtr(d2)
        put = RECIPROCAL_STRIKE * _ndtr(-d2) - F * _ndtr(-d1)
        value += math.exp(-Q_EUR * ts) * (N1 * call - N2 * put)
    # reported in the job's USD reporting currency, off the EUR the deal settles in
    return value * SPOT


def test_a_base_currency_underlying_prices_on_its_own_axis_under_the_fitted_law(tmp_path):
    """The reciprocal arm's AXIS against a closed form the engine has no part in. The deal pays
    EUR-per-USD while the only law the market carries is fitted on `FxRate.EUR`; it names the
    pair's non-base token, and the strip is Black on the DEAL's own axis at its own carry
    `r_EUR - r_USD`, with no convexity term and no reciprocal in the reference.

    WHAT IT PINS: the token rule and the deal-axis carry. Keyed the pre-rule way the deal is
    SKIPPED and there is no row to read. WHAT IT CANNOT PIN: the parameter carry, because at
    `Alpha: 0.0` the leverage term is identically zero and `hn_reciprocal_gamma` has no effect -
    measured, the carry mutated to the identity still passes here. The SIGN is read by
    `test_one_trade_authored_from_either_axis_is_one_number_under_the_fitted_law` and the whole map
    in closed form by `test_hn_component.py`. The 2e-3 bound is the 3.3e-4 seed spread with room.
    """
    reference = _reciprocal_expected()
    value = _mtm(_run(_reciprocal_job(sims=1 << 14), tmp_path, 'recip'))
    assert abs(value / reference - 1.0) < 2e-3, (value, reference)


#: The two axes of one accumulator, the equivalent notional converting at the STRIKE (`N` USD is
#: `N * K_A` EUR). Both barriers are out of reach, so both are strips of Europeans and the identity
#: is the change of numeraire and nothing else.
DIRECT_STRIKE = 1.10
MIRROR_STRIKE = 1.0 / DIRECT_STRIKE


def _mirrored_job(orientation, sims):
    """One trade, authored from either side of the pair, on one calibrated market."""
    job = _template()
    market = job['Calc']['MergeMarketData']['ExplicitMarketData']
    market['Price Factors']['HestonNandiModelParameters.EUR'] = dict(LEVERED)
    market['Valuation Configuration'] = {'FXAccumulatorOptionDeal': {'SpotModel': 'HestonNandi'}}
    if orientation == 'direct':
        # the fitted axis itself: EUR underlying, USD settlement, and a market call on the pair is
        # an engine PUT here with its knock-out direction flipped to match
        _deal_of(job).update({
            'Currency': 'USD', 'Underlying_Currency': 'EUR', 'Discount_Rate': 'USD',
            'Option_Type': 'Put', 'Strike_Price': DIRECT_STRIKE,
            'Underlying_Amount': N1 * MIRROR_STRIKE, 'LeverageNotional': N2 * MIRROR_STRIKE,
            'Barrier_Type': 'Down_And_Out', 'Barrier_Price': 0.2})
    else:
        _deal_of(job).update({
            'Currency': 'EUR', 'Underlying_Currency': 'USD', 'Discount_Rate': 'EUR',
            'Option_Type': 'Call', 'Strike_Price': MIRROR_STRIKE,
            'Underlying_Amount': N1, 'LeverageNotional': N2,
            'Barrier_Type': 'Up_And_Out', 'Barrier_Price': 5.0})
    job['Calc']['Calculation']['MCMC_Simulations'] = sims
    return job


def test_one_trade_authored_from_either_axis_is_one_number_under_the_fitted_law(tmp_path):
    """THE CONSISTENCY GATE, and the one that pins the DIRECTION of the carry. An accumulator on
    1,000 dollars settled in euro and the euro accumulator it IS - notional converted at the
    strike, sense inverted, knock-out flipped - are one trade, and one law must price them at one
    number. The direct side rides the fit as written, the reciprocal side rides it CARRIED.

    Carried, the two agree to 1.2e-3/2.3e-3 over 4,096 to 262,144 paths, which is the spread of two
    INDEPENDENT runs and does not shrink. With the carry removed they disagree by 2.7% to 4.1% -
    twenty times the 5e-3 bound - and it GROWS with the path count, being bias rather than noise.

    WHAT IT DOES NOT RESOLVE: the map is `1 - gamma*` and the UNIT SHIFT is worth only 2.7e-4 to
    1.5e-3 here, under the floor. This reads the SIGN; the whole map is pinned in closed form at
    1.4e-12 in `test_hn_component.py`.
    """
    import derivus.utils as utils

    assert utils.hn_reciprocal_gamma(utils.hn_reciprocal_gamma(-474.34)) == -474.34
    assert utils.hn_reciprocal_gamma(-474.34) > 0.0, 'the leverage skew mirrors with the axis'

    sims = 1 << 16
    mirror = _mtm(_run(_mirrored_job('reciprocal', sims), tmp_path, 'mirror'))
    direct = _mtm(_run(_mirrored_job('direct', sims), tmp_path, 'direct'))
    assert abs(mirror / direct - 1.0) < 5e-3, (mirror, direct)


def test_a_family_that_cannot_be_carried_to_the_reciprocal_axis_refuses_by_name(tmp_path):
    """The component family does not transport, so a deal that would have to be REFUSES rather than
    pricing off a law nobody fitted: the change of numeraire puts a state-dependent term in the
    long-run intercept, so the carried law is not a component parameter set and no `L_Curve`
    describes it. The refusal is FATAL, because a compile guard's canonical answer is to log and
    skip and a skipped deal marks at nothing on a job that reports success.
    """
    component = {'Property_Aliases': None, 'Alpha': 3.5681e-06, 'Beta': 0.8138,
                 'Gamma_1': -64.992, 'Rho': 0.99, 'Phi': 1.9820e-06, 'Gamma_2': -64.992,
                 'H0': 7.295e-05,
                 'L_Curve': {'.Curve': {'meta': [], 'data': [[0.0, 7.295e-05], [0.5, 9.647e-05]]}}}
    try:
        _run(_reciprocal_job(component, model='HestonNandiComponent', sims=1 << 10),
             tmp_path, 'component')
        raise AssertionError('the component family priced on an axis it cannot be carried to')
    except Exception as refusal:
        message = str(refusal)
    assert 'HestonNandiComponent' in message and 'ACC1' in message, message
    assert 'base currency' in message and 'long-run intercept' in message, message
    assert "SpotModel: 'HestonNandi'" in message, 'a refusal without a remedy'

    # and the same document under the family that DOES carry prices a number
    assert math.isfinite(_mtm(_run(_reciprocal_job(sims=1 << 10), tmp_path, 'plain')))


def test_a_cross_pair_keeps_the_underlyings_own_read(tmp_path):
    """Neither leg the base is OUT of the ruling's scope - both tokens of a cross are simulated
    factors, so the composed spot's law is nothing the pair's calibration describes. What the
    engine must still do is read `Underlying_Currency` byte for byte: a cross keyed off the
    settlement leg would price one pair's deal off another pair's fit and never say so.
    """
    def cross(block):
        job = _template()
        market = job['Calc']['MergeMarketData']['ExplicitMarketData']
        factors = market['Price Factors']
        factors['FxRate.GBP'] = dict(factors['FxRate.EUR'], Interest_Rate='GBP', Spot=1.27)
        factors['InterestRate.GBP'] = dict(factors['InterestRate.EUR'], Currency='GBP')
        factors['FXVol.EUR.GBP'] = dict(factors['FXVol.EUR.USD'])
        factors[block] = dict(DEGENERATE)
        market['Valuation Configuration'] = {
            'FXAccumulatorOptionDeal': {'SpotModel': 'HestonNandi'}}
        _deal_of(job).update({'Currency': 'GBP', 'Discount_Rate': 'GBP',
                              'FX_Volatility': 'EUR.GBP', 'Barrier_Price': 50.0})
        job['Calc']['Calculation']['MCMC_Simulations'] = 1 << 10
        return job

    priced = _run(cross('HestonNandiModelParameters.EUR'), tmp_path, 'cross_eur')
    assert math.isfinite(_mtm(priced)), 'a cross must still read its own underlying'

    skipped = _run(cross('HestonNandiModelParameters.GBP'), tmp_path, 'cross_gbp')
    rows = skipped['Results']['mtm']
    assert not len(rows[rows['Reference'] == 'ACC1']), (
        "the cross read the settlement leg - a deal priced off another pair's fit")
