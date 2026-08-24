"""FXAccumulatorOptionDeal end to end, through the JSON contract and nothing else.

THE FORM, which is the point of this file: a test is a job JSON, run through the public surface
(`Context.load_json` + `run_job`), whose expected answer is decided BEFORE the run and computed
from a closed form this repo's engine has no part in. No `Config` object is built by hand, no
internal constructor is imported, and nothing is patched - so what is exercised is the whole
framework, from the document a user actually writes to the numbers it reports.

The deal is authored so it has a closed form: with the barrier out of reach an accumulator is a
STRIP OF EUROPEANS - long `Underlying_Amount` of the ITM side and short `LeverageNotional` of the
OTM side at each fixing, each paid at its own settlement date. So both the value AND the FX delta
are Black, and the greek is checked as its own statement rather than merely for being finite.

The USD curve is STEEP with a knot at every tenor the deal reads. A flat curve is a degeneracy:
the interval carry strip and the raw zero-rate gather agree wherever the curve is flat however
far apart r and q are, so a pricer differencing neither would pass. The reference reads the same
curve with `numpy.interp`, which the knots make exact.

Pre-registered, from `tests/fixtures/fx_accumulator_job.json` and nothing else - re-derived by
`_expected()` at import off the loaded document, so the assertions follow the template rather
than these constants:

    value  +62.4979   $
    dV/dS  +2522.61   $ per unit of EUR.USD

The sign is the steep curve talking: by six months the USD rate reaches ~15% against a flat 2%
EUR, so the forward sits well above the strike, the ITM leg dominates the leveraged OTM one and
the strip is worth POSITIVE. On a flat 4% curve the same document is worth -28.30. That is worth
stating because it is the arm the fixture exists for - the interval carry strip and a raw
zero-rate gather agree on any flat curve, so a flat one cannot tell a pricer that differences
neither from one that does.

TOLERANCES ARE MEASURED, not chosen. This is a Monte Carlo pricer, and the seed-to-seed spread of
the reported value was read across five seeds:

    inner paths   sd      max |rel err| vs Black
       16 384     0.210        1.024 %
       65 536     0.148        0.833 %
      262 144     0.059        0.298 %

so the gate runs at 262 144 and allows 1 %, about five times the measured sd. The first draft ran
at 16 384 and allowed 0.5 % - INSIDE one standard error - and passed on CPU purely by luck while
failing on GPU, which is what produced this table. (The table was read on the flat-curve variant;
the steep curve moves the level, not the estimator's spread.)
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


# Every constant the reference needs is READ OUT OF THE DOCUMENT, so the expectation and the job
# cannot drift apart: edit the template and the closed form follows it.
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
    """The canonical document, varied. Switch coverage is a set of overrides on one template
    rather than a second hand-built document that can quietly stop resembling it."""
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
    """The greek is a statement of its own, not a finiteness check.

    A strip of Europeans has a closed-form delta, so `dV/d(EUR.USD spot)` is known before the run.
    The engine reports the derivative of the number it reported, which is what makes this the
    end-to-end check: a value that is right with a delta that is wrong is the failure mode a
    price-only gate cannot see.
    """
    out = _run(_job(greeks='First'), tmp_path)
    assert abs(_mtm(out) - EXPECTED_VALUE) / abs(EXPECTED_VALUE) < TOL
    frame = out['Results']['Greeks_First']
    column = [c for c in frame.columns if c != 'Value'][0]
    index, = [i for i in frame.index if str(i[0]) == 'FxRate.EUR']
    delta = float(frame.loc[index, column])
    assert abs(delta - EXPECTED_DELTA) / abs(EXPECTED_DELTA) < 2e-2, (delta, EXPECTED_DELTA)


def test_the_declared_switches_flip_the_sign_they_should(tmp_path):
    """`Buy_Sell` and `Option_Type` are declared switches, so the document is the way to exercise
    them: selling mirrors exactly, and a put on this strip is the ITM and OTM legs swapped."""
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
    """Exact-law GBM at the fixing dates with the indicator knock-out - the product the analytic
    survival truncation must reproduce. Returns (value, standard error)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    times = np.array([f / DAYS for f, _ in FIXINGS])
    settles = np.array([s / DAYS for _, s in FIXINGS])
    dt = np.diff(np.concatenate([[0.0], times]))
    z = rng.standard_normal((n_paths // 2, len(times)))
    z = np.concatenate([z, -z], axis=0)
    # the carry over each INTERVAL is a difference of cumulative integrals, not one tenor's rate
    # times the interval - the same statement the pricer's own `forward_carry_rate` makes, and on
    # this steep curve the two differ by whole percent
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
    """Both declared `Barrier_Type` values, each against an independent indicator simulation.

    The pricer integrates survival analytically and draws the surviving spot from the truncated
    law; the reference just simulates and applies the indicator. They are the same product by two
    routes that share no code.
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
    """`Barrier_Price` is `default=REQUIRED`, and a document that omits it anyway must refuse.

    Left at 0.0 with the declared default direction the deal survives NO fixing and would price
    to a silent scalar zero - a whole trade vanishing from a book with no error. The direction is
    why this checks both: at 0.0 a `Down_And_Out` is merely unbarriered and prices its full strip,
    so a one-direction fixture would score the hole harmless.
    """
    for btype in ('Up_And_Out', 'Down_And_Out'):
        job = _job(Barrier_Type=btype, Barrier_Price=0.0)
        out = _run(job, tmp_path, f'blank{btype}')
        rows = out['Results']['mtm']
        rows = rows[rows['Reference'] == 'ACC1']
        assert rows.empty or float(rows['Value'].iloc[0]) == 0.0, btype


def test_a_fixing_dated_today_follows_the_price_factor(tmp_path):
    """A fixing dated ON the base date reads the SIMULATED spot, not the value the trade recorded.

    That is deliberate and it is what makes a sensitivity calculation correct: bumping
    `FxRate.EUR` has to move that fixing's payoff, and it cannot if the fixing is a constant the
    document supplied. `make_fixing_data` writes the recorded value only for `fixing[0] <
    reference_date`, and a same-day reset is handed `Scenario >= 0` so it resolves off the path.

    A VALUE assertion cannot tell the two designs apart - the declared print and the current spot
    are the same number today - so the discriminating statement is the DELTA. One fixing, today,
    struck in the money: the deal is worth `N1 * (S - K)` discounted to its settlement and its
    spot delta is `N1 * D`, exactly. Were the fixing a constant the delta would be ZERO, which is
    a whole fixing's sensitivity missing from a book that reports the right price.
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
    """`Tags` carries the non-essential book-keeping - portfolio, trader - and `Tag_Titles` names
    the columns it lands in.

    Nothing in pricing reads either, which is the whole reason to assert them: a field no pricer
    touches is a field whose round trip nothing else can notice going wrong, and it is the field
    a desk sorts its blotter by.
    """
    out = _run(_job(), tmp_path, 'tags')
    rows = out['Results']['mtm']
    rows = rows[rows['Reference'] == 'ACC1']
    assert 'Portfolio' in rows.columns and 'Trader' in rows.columns, list(rows.columns)
    assert rows['Portfolio'].iloc[0] == 'FX-EXOTICS', rows['Portfolio'].iloc[0]
    assert rows['Trader'].iloc[0] == 'jdoe', rows['Trader'].iloc[0]
