"""QEDI_CustomAutoCallSwap end to end, through the JSON contract and nothing else.

Same form as the accumulator and TARF files: a job document run through `Context.load_json` +
`run_job`, an answer decided BEFORE the run from a closed form, and the DEBUG line the pricer
emits about what it decided.

THE CLOSED FORM. An autocall with a SINGLE coupon date is a cash-or-nothing DIGITAL: it pays the
coupon exactly when the spot finishes at or above the autocall threshold, so

    PV = Units * coupon * D(T) * N(d2),   d2 = (ln(S/K) + (r - q - sigma^2/2) T) / (sigma sqrt T)

with `K = threshold * strike`. That is exact under the GBM the document declares, so the gate is a
value assertion rather than a sanity check - the same reason the OSS pricer gates use a
one-coupon autocall to pin the spot-model read.

Pre-registered from the document below (spot 100, strike 100, r 4%, q 1%, sigma 25%, 1y coupon at
threshold 1.00, coupon 0.08, Units 10):

    single-coupon digital   PV = 5.5972   (N(d2) = 0.5822)

and the two degenerate limits that bracket it: a threshold no path reaches pays nothing, and a
threshold every path clears pays the coupon with certainty.
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

import derivus as rf
from derivus import utils
from derivus.config import CustomJsonEncoder

TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'fixtures', 'autocall_job.json')


def _template():
    with open(TEMPLATE) as f:
        return json.load(f)


def _deal_of(job):
    return job['Calc']['Deals']['Deals']['Children'][0]['Instrument']['.Deal']


_T = _template()
_D = _deal_of(_T)
_PF = _T['Calc']['MergeMarketData']['ExplicitMarketData']['Price Factors']

BASE = _T['Calc']['Calculation']['Base_Date']['.Timestamp']
SPOT = _PF['EquityPrice.EQ']['Spot']
STRIKE = _D['Strike_Price']
UNITS = _D['Units']
COUPON = _D['Autocall_Coupons'][0][1]
R_USD = _PF['InterestRate.USD']['Curve']['.Curve']['data'][0][1]
Q_EQ = _PF['DividendRate.EQ']['Curve']['.Curve']['data'][0][1]
SIGMA = _PF['VolatilityGrid.EQ']['Surface']['.Curve']['data'][0][2]
DAYS = 365.0


def _offset(stamp):
    import datetime
    return (datetime.date.fromisoformat(stamp) - datetime.date.fromisoformat(BASE)).days


HORIZON = _offset(_D['Autocall_Coupons'][0][0]['.Timestamp'])


def _ndtr(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _digital(threshold):
    """PV of the one-coupon autocall: a cash-or-nothing digital struck at threshold * strike."""
    t = HORIZON / DAYS
    k = threshold * STRIKE
    sd = SIGMA * math.sqrt(t)
    d2 = (math.log(SPOT / k) + (R_USD - Q_EQ - 0.5 * SIGMA ** 2) * t) / sd
    return UNITS * COUPON * math.exp(-R_USD * t) * _ndtr(d2)


EXPECTED_ATM = _digital(1.00)


def _job(threshold=1.00, greeks='No', **deal_overrides):
    """The canonical document, varied. The autocall threshold is the switch this file spans."""
    job = _template()
    job['Calc']['Calculation']['Greeks'] = greeks
    deal = _deal_of(job)
    for row in deal['Autocall_Thresholds']:
        row[1] = threshold
    deal.update(deal_overrides)
    return job


def _run(job, tmp_path, name='ac', debug=False):
    path = os.path.join(str(tmp_path), f'{name}.json')
    with open(path, 'w') as f:
        json.dump(job, f, default=str)
    buf, root = io.StringIO(), logging.getLogger()
    handler = logging.StreamHandler(buf)
    old = root.level
    if debug:
        root.addHandler(handler)
        root.setLevel(logging.DEBUG)
    try:
        cx = rf.Context()
        cx.load_json(path)
        _, out = cx.run_job()
    finally:
        if debug:
            root.removeHandler(handler)
            root.setLevel(old)
    return out, buf.getvalue()


def _mtm(out, ref='AC1'):
    rows = out['Results']['mtm']
    rows = rows[rows['Reference'] == ref]
    return float(rows['Value'].iloc[0])


def test_a_single_coupon_autocall_is_a_digital_between_its_two_limits(tmp_path):
    """One coupon date makes the payoff cash-or-nothing, and Black prices it with no Monte Carlo
    error of its own. The two limits fix the SCALE, so a pricer returning a constant cannot satisfy
    all three: no path clears a 10x threshold, so the deal is worth ~0, and a threshold at a
    hundredth of spot is cleared almost surely, so it is the discounted coupon. The DEBUG line says
    what the pricer decided - one coupon date, one fixing per coupon (the OSS branch), one block -
    which the value alone cannot.

    Killing mutation: the coupon's cash paid undiscounted (`D[j]` dropped).
    """
    out, log = _run(_job(), tmp_path, debug=True)
    v = _mtm(out)
    assert abs(v - EXPECTED_ATM) / EXPECTED_ATM < 1e-2, (v, EXPECTED_ATM)
    organ = [ln for ln in log.splitlines() if 'AUTOCALL ' in ln and 'coupons=' in ln][-1]
    assert 'coupons=1' in organ and 'fullpath=0' in organ and 'blocks=1' in organ, organ

    far, _ = _run(_job(threshold=10.0), tmp_path, 'far')
    assert abs(_mtm(far)) < 1e-6 * UNITS * COUPON, _mtm(far)
    near, _ = _run(_job(threshold=0.01), tmp_path, 'near')
    certain = UNITS * COUPON * math.exp(-R_USD * HORIZON / DAYS)
    assert abs(_mtm(near) - certain) / certain < 1e-3, (_mtm(near), certain)


def test_a_barrier_dated_on_the_fixing_is_the_barrier_dated_on_the_coupon(tmp_path):
    """A barrier date is a fixing: observed on its coupon's window, settled on the coupon date.

    Two coupons fixed three days early with a 70% put. The barrier dated on the FIXINGS prices
    bit-identically to the barrier dated on the COUPONS (-0.0542905931889), both on the
    one-step-survival arm, and the put is live: the no-barrier document reads +0.2804032290490.
    Killing mutation: a barrier date read by its date alone, with no window behind it - the
    fixing-dated document falls to the full-path arm (`fullpath=1`) and reads +0.0185267.
    """
    coupons, fixings = ['2024-12-27', '2025-06-27'], ['2024-12-24', '2025-06-24']

    def doc(dates, barrier=0.7):
        return _job(Expiry_Date={'.Timestamp': coupons[-1]},
                    Price_Fixing=[[{'.Timestamp': x}, 0.0] for x in fixings],
                    Autocall_Coupons=[[{'.Timestamp': x}, 0.04] for x in coupons],
                    Autocall_Thresholds=[[{'.Timestamp': x}, 1.0] for x in coupons],
                    Barrier=barrier, Barrier_Dates=[{'.Timestamp': x} for x in dates])

    on_coupons, log_c = _run(doc(coupons), tmp_path, 'bar_c', debug=True)
    on_fixings, log_f = _run(doc(fixings), tmp_path, 'bar_f', debug=True)
    none, _ = _run(doc([], barrier=0.0), tmp_path, 'bar_0')
    assert _mtm(on_fixings) == _mtm(on_coupons), (_mtm(on_fixings), _mtm(on_coupons))
    for log in (log_c, log_f):
        organ = [ln for ln in log.splitlines() if 'AUTOCALL ' in ln and 'coupons=2' in ln][-1]
        assert 'fullpath=0' in organ and 'barrier=70' in organ, organ
    assert _mtm(none) != _mtm(on_coupons)


def test_a_coupon_observed_on_or_before_the_coupon_before_it_refuses_by_name(tmp_path):
    """`Coupon_Observations` pairing the second coupon with a fixing on or before the first coupon
    would read, on the wholly observed arm, the level the first coupon already decided on - the
    pricer carries one observed fixing forward per coupon. Such a table refuses by name beside the
    crossing check - a fixing on the first coupon's own day as one before it - and the same table
    naming a fixing after the first coupon prices.

    Killing mutation: the check strict (`observed < previous`), the fixing on the coupon's day
    priced.
    """
    coupons = ['2024-12-27', '2025-06-27']
    fixings = ['2024-12-20', '2024-12-24', '2024-12-27', '2025-06-24']

    def doc(observed):
        return _job(Expiry_Date={'.Timestamp': coupons[-1]},
                    Price_Fixing=[[{'.Timestamp': x}, 0.0] for x in fixings],
                    Autocall_Coupons=[[{'.Timestamp': x}, 0.04] for x in coupons],
                    Autocall_Thresholds=[[{'.Timestamp': x}, 1.0] for x in coupons],
                    Coupon_Observations=[[{'.Timestamp': c}, {'.Timestamp': f}]
                                         for c, f in zip(coupons, observed)])

    for stale in ('2024-12-24', '2024-12-27'):
        with pytest.raises(utils.UnpriceableSchedule, match=(
                r'Coupon_Observations dates the coupon of 2025-06-27 on a fixing of {}, on or '
                r'before the coupon of 2024-12-27'.format(stale))):
            _run(doc(['2024-12-20', stale]), tmp_path, 'stale')
    assert math.isfinite(_mtm(_run(doc(['2024-12-24', '2025-06-24']), tmp_path, 'fresh')[0]))


# --------------------------------------------------------------------------------------------
# compo: the same digital on the CONVERTED spot
# --------------------------------------------------------------------------------------------
FX_EUR = 1.25       # USD per EUR: the USD->EUR cross the compo scales by is 1/FX_EUR
R_EUR = 0.02
FX_SIGMA = 0.15
CORR = 0.35         # authored on the SORTED pair EUR.USD; the USD->EUR deal reads -CORR


def _compo_job(threshold=1.00, corr=CORR):
    """The template as a COMPO deal: EUR payoff on the USD asset, monitored on S*X.

    The strike is a PAYOFF-currency quantity, so it is authored at the same moneyness in EUR:
    `STRIKE / FX_EUR` against a compo spot of `SPOT / FX_EUR`.
    """
    job = _template()
    deal = _deal_of(job)
    deal['Payoff_Currency'] = 'EUR'
    deal['Payoff_Type'] = 'Compo'
    deal['Discount_Rate'] = 'EUR'
    deal['Strike_Price'] = STRIKE / FX_EUR
    for row in deal['Autocall_Thresholds']:
        row[1] = threshold
    pf = job['Calc']['MergeMarketData']['ExplicitMarketData']['Price Factors']
    pf['InterestRate.EUR'] = {
        'Currency': 'EUR', 'Day_Count': 'ACT_365', 'Sub_Type': None,
        'Curve': {'.Curve': {'meta': [], 'data': [[0.0, R_EUR], [5.0, R_EUR]]}}}
    pf['FxRate.EUR'] = {
        'Domestic_Currency': None, 'Interest_Rate': 'EUR', 'Priority': 1, 'Spot': FX_EUR}
    pf['FXVol.EUR.USD'] = {
        'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness',
        'Surface': {'.Curve': {'meta': [], 'data': [
            [m, t, FX_SIGMA] for m in (0.8, 1.0, 1.2) for t in (0.02, 2.0)]}}}
    pf['Correlation.EquityPrice.EQ/FxRate.EUR.USD'] = {'Value': corr}
    return job


def _compo_digital(threshold, corr):
    """Closed form for the one-coupon compo autocall: a cash-or-nothing digital in EUR on
    C = S*X, reported in USD.

        C0 = SPOT / FX_EUR                 (the USD->EUR cross is 1/FX_EUR)
        b  = R_EUR - Q_EQ                  ((r_usd - q) + (r_eur - r_usd))
        sigma_c^2 = SIGMA^2 + 2*rho*SIGMA*FX_SIGMA + FX_SIGMA^2,  rho = -corr
        PV = Units * coupon * exp(-R_EUR*T) * N(d2) * FX_EUR      (EUR mark, USD report)

    `rho = -corr` is the sorted-pair convention: the correlation is authored on EUR.USD and this
    deal's cross runs USD->EUR, so `check_fx_name` flips the sign - the oracle carrying the flip
    is what makes this gate test the convention and not just the arithmetic.
    """
    t = HORIZON / DAYS
    c0 = SPOT / FX_EUR
    k = threshold * STRIKE / FX_EUR
    rho = -corr
    sigma_c = math.sqrt(SIGMA ** 2 + 2.0 * rho * SIGMA * FX_SIGMA + FX_SIGMA ** 2)
    b = R_EUR - Q_EQ
    d2 = (math.log(c0 / k) + (b - 0.5 * sigma_c ** 2) * t) / (sigma_c * math.sqrt(t))
    return UNITS * COUPON * math.exp(-R_EUR * t) * _ndtr(d2) * FX_EUR


def test_a_compo_autocall_is_a_digital_on_the_converted_spot(tmp_path):
    """The compo value oracle, exact under the flat document: the pricer simulates the PRODUCT S*X
    and a single coupon makes the payoff a digital Black prices with no Monte Carlo error. Compo OSS
    deals had never priced before this (`calc_vol_adjustment` returned a python-float b_adj that
    `torch.unsqueeze` refused, skipping the deal).

    MEASURED: 0.45887454 against the closed form's 0.45887454, 4.8e-16 relative; the flipped
    correlation lands on ITS closed form (0.43677514) exactly, so the sorted-pair sign flip is
    measured rather than assumed.

    Killing mutation: the composite variance's cross term taken with the wrong sign.
    """
    out, _ = _run(_compo_job(), tmp_path, 'compo')
    expected = _compo_digital(1.00, CORR)
    assert abs(_mtm(out) - expected) / expected < 1e-9, (_mtm(out), expected)
    flipped, _ = _run(_compo_job(corr=-CORR), tmp_path, 'compo_flip')
    mirrored = _compo_digital(1.00, -CORR)
    assert abs(_mtm(flipped) - mirrored) / mirrored < 1e-9, (_mtm(flipped), mirrored)
    assert abs(expected - mirrored) / expected > 0.01, 'the two arms must separate'


# --------------------------------------------------------------------------------------------
# the ledger, and the mark it has to agree with
# --------------------------------------------------------------------------------------------
def _cmc_job(threshold, units=10.0, buy_sell='Buy', coupon_days=(91, 182, 273)):
    """The template as a credit Monte Carlo on a simulated GBM spot, with several coupon dates.

    A SINGLE coupon date is blind to the defect this file exists to pin - the settle fired once
    per coupon on a settling row, so with one coupon it looked exactly right. The reconciliation
    also needs the deal to survive at least one date to be worth anything.
    """
    job = _template()
    deal = _deal_of(job)
    deal['Units'] = units
    deal['Buy_Sell'] = buy_sell
    dates = [{'.Timestamp': _stamp(d)} for d in coupon_days]
    deal['Autocall_Coupons'] = [[d, COUPON] for d in dates]
    deal['Autocall_Thresholds'] = [[d, threshold] for d in dates]
    deal['Price_Fixing'] = [[d, 0.0] for d in dates]
    deal['Expiry_Date'] = dates[-1]
    job['Calc']['Calculation'] = {
        'Object': 'CreditMonteCarlo', 'Base_Date': {'.Timestamp': BASE}, 'Currency': 'USD',
        'Time_grid': '0d 12m(1m)', 'Batch_Size': 256, 'Simulation_Batches': 1,
        'Random_Seed': 1, 'MCMC_Simulations': 1 << 12, 'Deflation_Interest_Rate': 'USD',
        'Generate_Cashflows': 'Yes'}
    market = job['Calc']['MergeMarketData']['ExplicitMarketData']
    market['Price Models'] = {'GBMAssetPriceModel.EQ': {'Vol': SIGMA, 'Drift': 0.0}}
    market['Model Configuration'] = {'.ModelParams': {
        'modeldefaults': {'EquityPrice': 'GBMAssetPriceModel'}, 'modelfilters': {}}}
    return job


def _stamp(offset):
    import datetime
    return (datetime.date.fromisoformat(BASE) + datetime.timedelta(days=offset)).isoformat()


def _cmc(job, tmp_path, name):
    path = os.path.join(str(tmp_path), f'{name}.json')
    with open(path, 'w') as f:
        json.dump(job, f, cls=CustomJsonEncoder)
    cx = rf.Context()
    cx.load_json(path)
    _, out = cx.run_job()
    # under a credit Monte Carlo `mtm` is the exposure PROFILE (dates x scenarios), not the
    # per-deal frame a base valuation returns
    return out['Results']['cashflows']['USD'], out['Results']['mtm']


FLOAT = 0.0125


def _swap(job, barrier=70.0, lag=0, margin=50.0):
    """The swap version: a floating payment on each coupon date and a put barrier observed on the
    last. The first period has started, so its row states the amount it fixed, `FLOAT`; every
    later row states `-FLOAT`, which nothing reads - a later period is forecast, at `margin` basis
    points over the curve. With a `lag`, each coupon fixes that many days early, and a floating
    payment also falls mid-period and between each fixing and its coupon."""
    deal = _deal_of(job)
    coupons = [pd.Timestamp(row[0]['.Timestamp']) for row in deal['Autocall_Coupons']]
    fixings = [d - pd.Timedelta(days=lag) for d in coupons]
    floats = coupons
    if lag:
        starts = [pd.Timestamp(BASE)] + coupons[:-1]
        floats = sorted(coupons + [d - pd.Timedelta(days=lag // 2) for d in coupons] +
                        [a + pd.Timedelta(days=(b - a).days // 2) for a, b in zip(starts, coupons)])
    deal.update({
        'Object': 'QEDI_CustomAutoCallSwap_V2', 'Barrier': barrier, 'Barrier_Dates': fixings[-1:],
        'Price_Fixing': [[d, 0.0] for d in fixings], 'Forecast_Rate': 'USD',
        'Floating_Margin': utils.Basis(margin), 'Reset_Frequency': pd.DateOffset(months=3),
        'Autocall_Floating': [[d, FLOAT if k == 0 else -FLOAT] for k, d in enumerate(floats)]})
    return job


def _forecast(days):
    """A forecast period's payment per unit: the flat curve's simple forward plus the 50bp margin."""
    return math.expm1(R_USD * days / DAYS) + 0.005 * days / DAYS


def test_the_swap_leg_pays_what_it_declares_and_settles_on_its_own_rows(tmp_path):
    """The swap version's floating leg is the one it declares: the started period pays the amount
    its row states, `FLOAT`, and each later one the forecast off `Forecast_Rate` plus the margin,
    whatever its row states - here `-FLOAT`. With a threshold no path reaches every path survives,
    so each coupon date books `-Units` times its period's payment exactly, and the expiry row books
    that beside the put: there, where everything the deal owes is paid, the ledger IS the mark,
    scenario by scenario; and with no put the base row is the leg alone, each payment discounted
    from its date. The run is the calculation's float32, so the readings are held to 1e-6.

    Killing mutations: the row's stated value paid and a row stated at or below zero skipped, the
    later rows reading 0 against -0.11269; every row paying the first period's amount, the base row
    reading -0.36761 against -0.34359.
    """
    ledger, mtm = _cmc(_swap(_cmc_job(threshold=100.0)), tmp_path, 'swap')
    dates = [pd.Timestamp(_stamp(d)) for d in (91, 182, 273)]
    owed = [FLOAT, _forecast(91), _forecast(91)]
    for date, amount in zip(dates[:-1], owed):
        cash = np.asarray(ledger.loc[date].values, dtype=float)
        assert np.all(np.abs(cash + UNITS * amount) < 1e-6), (date, cash[:4], -UNITS * amount)
    paid, marked = (np.asarray(frame.loc[dates[-1]].values, dtype=float) for frame in (ledger, mtm))
    assert np.all(np.abs(paid - marked) <= 1e-6 * np.maximum(np.abs(marked), 1.0)), (paid[:4], marked[:4])
    assert paid.mean() < -UNITS * owed[-1], 'the put pays on some path'
    # with no put, the base row is the leg alone, each period's payment discounted from its date
    bare = np.asarray(_cmc(_swap(_cmc_job(threshold=100.0), barrier=0.0), tmp_path, 'bare')[1].iloc[0])
    leg = -UNITS * sum(a * math.exp(-R_USD * d / DAYS) for a, d in zip(owed, (91, 182, 273)))
    assert np.all(np.abs(bare - leg) < 1e-6 * UNITS), (bare[:4], leg)


def test_a_row_reads_only_what_has_printed_by_its_own_date(tmp_path):
    """Two coupons fixed well before their dates - day 50 for day 91, day 172 for day 182 - at a
    threshold of the strike, under a credit Monte Carlo reporting monthly. Day 61 falls between the
    first fixing and its coupon: a scenario there is called, worth the coupon discounted to day 91,
    or walks the second coupon from ITS OWN spot, worth that coupon's digital to day 172 paid on day
    182. Day 122 falls after the first coupon and before the second fixing: dead, or that digital on
    its own spot. Scenario by scenario, to 1e-5.

    Killing mutations: the walk opening on the first fixing's print, day 61 missing by up to 0.123;
    the fixing no block boundary, day 122 reading the second fixing's print - 0 or the whole coupon
    - and missing by up to 0.738.
    """
    job = _cmc_job(threshold=1.0, coupon_days=(91, 182))
    _deal_of(job)['Price_Fixing'] = [[{'.Timestamp': _stamp(d)}, 0.0] for d in (50, 172)]
    job['Calc']['Calculation'].update(Time_grid='0d 1m(1m)', Calc_Scenarios='All')
    out = _cva(job, tmp_path, 'printed')
    spot = out['Results']['scenarios']['EquityPrice.EQ'].loc[0.0]

    def second(day):
        s = np.asarray(spot[pd.Timestamp(_stamp(day))].values, dtype=float)
        t = (172 - day) / DAYS
        d2 = (np.log(s / STRIKE) + (R_USD - Q_EQ - 0.5 * SIGMA ** 2) * t) / (SIGMA * math.sqrt(t))
        return UNITS * COUPON * math.exp(-R_USD * (182 - day) / DAYS) * np.vectorize(_ndtr)(d2)

    for day, called in ((61, UNITS * COUPON * math.exp(-R_USD * 30 / DAYS)), (122, 0.0)):
        marked = np.asarray(out['Results']['mtm'].loc[pd.Timestamp(_stamp(day))].values, dtype=float)
        miss = np.minimum(np.abs(marked - called), np.abs(marked - second(day)))
        assert miss.max() < 1e-5, (day, miss.max(), marked[:4], second(day)[:4])


def test_an_autocalled_path_pays_its_coupon_once_and_is_worth_nothing_after(tmp_path):
    """A path pays EXACTLY `Units * coupon`, exactly ONCE - the payment, scaled, then the latch. The
    document autocalls at its first coupon with certainty, so the first date books the whole
    payment, later dates book nothing (0.8 / 0 / 0) and the profile is worth nothing after it
    (0.79206 / 0.8 / 0 / 0). And the t0 mark is that first payment discounted back: the mark and the
    ledger are the same cashflow seen from two places, which neither a value gate nor a cashflow
    gate can say alone.

    Four defects, all invisible to a SINGLE-coupon document: the settle in the coupon loop under a
    ROW-level `tau` (0.24 where 0.08 pays); `P`, the accumulated VALUE, booked rather than the
    payment; `nominal` scaling the mark and not the ledger; and `terminationDate` stamped inside
    `sim_spot` and never returned, so every later block re-paid the deal (0.8 / 0.8 / 0.8).

    Killing mutation: the latch a block returns discarded, so the next block re-prices and re-pays.
    """
    ledger, mtm = _cmc(_cmc_job(threshold=0.01), tmp_path, 'once')
    cash = np.asarray(ledger.values, dtype=float).mean(axis=1)
    profile = np.asarray(mtm.values, dtype=float).mean(axis=1)
    assert len(cash) > 1, 'a single-coupon document cannot see the defects this gate pins'
    assert abs(cash[0] - UNITS * COUPON) < 1e-6, (cash[0], UNITS * COUPON)
    assert np.all(np.abs(cash[1:]) < 1e-9), (cash, 'an autocalled path pays once')
    assert np.all(np.abs(profile[2:]) < 1e-9), (profile, 'and is worth nothing from then on')
    t = (ledger.index[0] - pd.Timestamp(BASE)).days / DAYS
    discounted = float(cash[0]) * math.exp(-R_USD * t)
    assert abs(profile[0] - discounted) / abs(discounted) < 5e-3, (profile[0], discounted, t)


def test_an_autocall_on_a_static_equity_is_skipped_by_name_under_a_credit_monte_carlo(tmp_path):
    """The autocall walks a simulated equity, so a credit Monte Carlo holding its equity static
    cannot value it: beside a cash flow the run values, it is counted under `Deals Skipped`, under
    `No` as under `Yes`, the switch being the compile's - where it was marked at nothing and
    counted nowhere.

    Killing mutation: the static equity's refusal dropped, the deal marked rather than skipped.
    """
    job = _cmc_job(threshold=0.01)
    market = job['Calc']['MergeMarketData']['ExplicitMarketData']
    market['Price Models'] = {'HullWhite1FactorInterestRateModel.USD': {
        'Alpha': 0.05, 'Lambda': 0.0, 'Quanto_FX_Correlation': 0.0,
        'Quanto_FX_Volatility': {'.Curve': {'meta': [], 'data': [[0.0, 0.0]]}},
        'Sigma': {'.Curve': {'meta': [], 'data': [[0.0, 0.01]]}}}}
    market['Model Configuration'] = {'.ModelParams': {'modeldefaults': {
        'InterestRate': 'HullWhite1FactorInterestRateModel'}, 'modelfilters': {}}}
    job['Calc']['Deals']['Deals']['Children'].append({'Instrument': {'.Deal': {
        'Object': 'FixedCashflowDeal', 'Reference': 'CF', 'Currency': 'USD',
        'Discount_Rate': 'USD', 'Amount': 100.0, 'Payment_Date': {'.Timestamp': _stamp(180)}}}})

    def run(name):
        path = os.path.join(str(tmp_path), f'{name}.json')
        with open(path, 'w') as f:
            json.dump(job, f, default=str)
        cx = rf.Context()
        cx.load_json(path)
        return cx.run_job()[1]

    assert run('static')['Stats'].get('Deals Skipped') == 1
    market['System Parameters']['Exclude_Deals_With_Missing_Market_Data'] = 'No'
    assert run('static_kept')['Stats'].get('Deals Skipped') == 1


def _cva_job(threshold=1.02, spot=None, gradient='No'):
    """The CMC document with a counterparty and the CVA block on - the sensitivity run.

    The threshold sits 2% out of the money so the trigger is LIVE: scenarios cross it at every
    coupon date, which is what makes the CVA's spot delta carry boundary flux in both halves of
    the decision's reach - the fired/survived fork on the decision row, and the carried latch
    killing every later row. A threshold nothing reaches is the control (`test_..._control`).
    """
    job = _cmc_job(threshold=threshold)
    calc = job['Calc']['Calculation']
    calc['Batch_Size'] = 1024
    calc['Simulation_Batches'] = 4
    calc['MCMC_Simulations'] = 256
    calc['Credit_Valuation_Adjustment'] = {
        'Calculate': 'Yes', 'Counterparty': 'CPTY', 'Deflate_Stochastically': 'No',
        'Stochastic_Hazard_Rates': 'No', 'Gradient': gradient}
    market = job['Calc']['MergeMarketData']['ExplicitMarketData']
    # the counterparty curve, in the wire form CustomJsonEncoder writes for a utils.Curve
    market['Price Factors']['SurvivalProb.CPTY'] = {
        'Recovery_Rate': 0.4,
        'Curve': {'.Curve': {'meta': [], 'data': [[0.0, 0.0], [10.0, 0.4]]}}}
    if spot is not None:
        market['Price Factors']['EquityPrice.EQ']['Spot'] = spot
    return job


def _cva(job, tmp_path, name):
    path = os.path.join(str(tmp_path), f'{name}.json')
    with open(path, 'w') as f:
        json.dump(job, f, cls=CustomJsonEncoder)
    cx = rf.Context()
    cx.load_json(path)
    _, out = cx.run_job()
    return out


def _cva_ladder(tmp_path, threshold, rungs=(0.3, 0.5, 1.0), wrap=lambda j: j):
    """AAD spot delta of the CVA, and a central-difference ladder of the SAME document.

    Common random numbers arrive through the contract: `Random_Seed` is in the document and the
    bumped runs change nothing but the `EquityPrice.EQ` `Spot` value, so each rung differences
    two runs drawing identical paths. No internals, nothing patched.
    """
    out = _cva(wrap(_cva_job(threshold=threshold, gradient='Yes')), tmp_path, 'aad')
    g = out['Results']['grad_cva']['Gradient']
    eq_rows = [i for i in g.index if 'EquityPrice' in str(i[0])]
    aad = float(g.loc[eq_rows[0]]) if eq_rows else 0.0
    crn = []
    for h in rungs:
        up = float(_cva(wrap(_cva_job(threshold=threshold, spot=SPOT + h)),
                        tmp_path, f'up{h}')['Results']['cva'])
        dn = float(_cva(wrap(_cva_job(threshold=threshold, spot=SPOT - h)),
                        tmp_path, f'dn{h}')['Results']['cva'])
        crn.append((up - dn) / (2.0 * h))
    return aad, crn, float(out['Results']['cva'])


def test_the_cva_spot_delta_matches_the_same_document_bumped(tmp_path):
    """`grad_cva`'s equity entry against a CRN central-difference ladder of the same job document,
    live trigger.

    MEASURED at 1024 x 4 batches, 256 inner, rungs 0.3/0.5/1.0 on spot 100:

        cva 0.005450535   AAD +1.3809097e-04   CRN 1.3581/1.4697/1.4362e-04
        disagreement 1.68% at the best rung, ladder flatness 8.22%

    so the 5% tolerance carries a 3x margin inside a ladder whose own spread is wider. The decision
    registers ONCE, and each half of its reach suppressed alone kills this gate:

        latch neutralised (triggered := untriggered): AAD +2.5547825e-04, +73.83%
        own-row override suppressed: AAD +3.7148406e-05, -72.65%

    Opposite signs, each ~43x the corrected residual - which is why the WHOLE registration
    suppressed is a weak mutant here (+5.15%): the two halves nearly cancel on this fixture.

    Killing mutation: the own-row fired/survived override suppressed.
    """
    aad, crn, cva = _cva_ladder(tmp_path, threshold=1.02)
    best = min(crn, key=lambda c: abs(aad - c))
    flat = (max(crn) - min(crn)) / abs(best)
    assert abs(aad - best) / abs(best) < 0.05, (aad, crn, cva, flat)


def _collateralised(job):
    """Wrap the deal in a zero-threshold NettingCollateralSet, in the wire form the decoder reads."""
    csa = {'.CreditSupportList': [[0.0, 0.0]]}
    netting = {
        'Object': 'NettingCollateralSet', 'Reference': 'NS1', 'Netted': 'True',
        'Collateralized': 'True', 'Agreement_Currency': 'USD', 'Funding_Rate': 'USD',
        'Balance_Currency': 'USD', 'Liquidation_Period': 10.0, 'Settlement_Period': 0.0,
        'Credit_Support_Amounts': {
            'Received_Threshold': csa, 'Posted_Threshold': csa, 'Independent_Amount': csa,
            'Minimum_Received': csa, 'Minimum_Posted': csa}}
    deals = job['Calc']['Deals']['Deals']
    deals['Children'] = [{'Instrument': {'.Deal': netting}, 'Children': deals['Children']}]
    return job


def _lagged_swap(job):
    """The swap version under a zero-threshold CSA at 4096 paths: each coupon fixed six days early,
    a put at 90, and a funding spread of 400bp, wide enough that every floating row carries weight."""
    job['Calc']['Calculation']['Batch_Size'] = 4096
    return _collateralised(_swap(job, barrier=90.0, lag=6, margin=400.0))


def test_a_collateralised_cva_delta_carries_every_row_a_decision_settles(tmp_path):
    """Under a CSA each decision's counterfactual runs the gross->net chain and the settled-cash
    ledger, so it replays every row's WHOLE cash - the leg and the coupon if the row's decision
    fires, the leg and the put if not, a float row between decisions under the last one before it -
    and forks every row observing it, a lagged coupon's float row before its own block included.
    An observed put is a decision of its own the same way. The lagged swap version, `_lagged_swap`.

    MEASURED at 4096 x 4 batches, rungs 0.3/0.5/1.0 on spot 100:

        cva 0.001358907   AAD -6.3203508e-05   CRN -6.19671/-6.04400/-6.02352e-05
        disagreement +2.00% at the best rung, ladder flatness 2.79%

    Killing mutations, at the best rung: the leg left out of a decision row's cash +11.15%, the put
    +21.23%, a float row between decisions undeclared +6.67%, a lagged decision's forks in an
    earlier block dropped +44.50%, every row undeclared +26.24%; the put's decision dropped +17.03%,
    its cash left where it was booked +7.15%, its forks on its own row alone +11.16%.
    """
    aad, crn, cva = _cva_ladder(tmp_path, threshold=1.02, wrap=_lagged_swap)
    best = min(crn, key=lambda c: abs(aad - c))
    assert abs(aad - best) / abs(best) < 0.05, (aad, crn, cva)


def test_a_dead_trigger_contributes_no_spurious_delta(tmp_path):
    """The control this deal shape can have. A threshold no path reaches makes the deal WORTHLESS -
    a coupon-only autocall is nothing but its trigger, so there is no live-but-fluxless
    configuration to control against. What a saturated trigger CAN gate is silence: the
    registrations still run under `Gradient: Yes` and must contribute neither value nor a spurious
    delta - cva, AAD and every CRN rung exactly zero.

    Killing mutation: the boundary kernel admitting every scenario, however far from its trigger.
    """
    aad, crn, cva = _cva_ladder(tmp_path, threshold=10.0, rungs=(0.3, 0.5))
    assert cva == 0.0 and aad == 0.0 and all(c == 0.0 for c in crn), (aad, crn, cva)


def test_the_ledger_mirrors_and_scales_with_the_deal(tmp_path):
    """`Units` and `Buy_Sell` reach the ledger exactly as they reach the mark - they did not.

    Killing mutation: the settled coupon booked without `nominal`.
    """
    one, _ = _cmc(_cmc_job(0.01, units=1.0), tmp_path, 'u1')
    ten, _ = _cmc(_cmc_job(0.01, units=10.0), tmp_path, 'u10')
    sold, _ = _cmc(_cmc_job(0.01, units=10.0, buy_sell='Sell'), tmp_path, 'sold')
    total = lambda led: float(np.asarray(led.values, dtype=float).sum() / led.shape[1])
    assert abs(total(ten) - 10.0 * total(one)) < 1e-6, (total(ten), total(one))
    assert abs(total(sold) + total(ten)) < 1e-6, (total(sold), total(ten))
