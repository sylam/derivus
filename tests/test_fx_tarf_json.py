"""FXTARFOptionDeal end to end, through the JSON contract and nothing else.

Same form as `test_fx_accumulator_json.py`: a job document, run through `Context.load_json` +
`run_job`, whose answer is decided BEFORE the run from a closed form the engine has no part in,
plus the DEBUG line the pricer emits about what it decided.

The degenerate limit that gives a closed form: a TARF whose target can never be reached never
knocks out, so it is a STRIP OF EUROPEANS - long `Underlying_Amount` of the ITM side and short
`LeverageNotional` of the OTM side at each fixing, paid at that fixing's settlement. Its value and
its FX delta are Black.

Pre-registered from `tests/fixtures/fx_tarf_job.json` and nothing else - re-derived by
`_expected()` at import off the loaded document, so they follow the template:

    value  +62.4979   $
    dV/dS  +2522.61   $ per unit of EUR.USD

Positive because the curve is steep: by six months USD reaches ~15% against a flat 2% EUR, so the
forward is well above the strike and the ITM leg dominates the leveraged OTM one. A FLAT curve is
the degeneracy this fixture exists to avoid - the interval carry strip and a raw zero-rate gather
agree wherever the curve is flat, however far apart r and q are.

and the log line the pricer must emit for that document:

    TARF T1 fixings=2 resolved=0 target=1e+09 accrued=0 barrier=0 blocks=1
"""
import io
import json
import logging
import math
import os
import sys

# reference-derivus shadow-import guard (MEMORY): pin the package under test to THIS repo.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import derivus as rf

TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'fixtures', 'fx_tarf_job.json')


def _template():
    with open(TEMPLATE) as f:
        return json.load(f)


def _deal_of(job):
    return job['Calc']['Deals']['Deals']['Children'][0]['Instrument']['.Deal']


# every constant read OUT of the document, so the closed form follows the template
_T = _template()
_D = _deal_of(_T)
_PF = _T['Calc']['MergeMarketData']['ExplicitMarketData']['Price Factors']

BASE = _T['Calc']['Calculation']['Base_Date']['.Timestamp']
SIMS = _T['Calc']['Calculation']['MCMC_Simulations']
SPOT = _PF['FxRate.EUR']['Spot']
STRIKE = _D['Strike_Price']
N1, N2 = _D['Underlying_Amount'], _D['LeverageNotional']
UNREACHABLE = _D['TargetLevel']
USD_CURVE = _PF['InterestRate.USD']['Curve']['.Curve']['data']
Q_EUR = _PF['InterestRate.EUR']['Curve']['.Curve']['data'][0][1]
SIGMA = _PF['FXVol.EUR.USD']['Surface']['.Curve']['data'][0][2]
DAYS = 365.0
TOL = 1e-2


def _offset(stamp):
    import datetime
    return (datetime.date.fromisoformat(stamp) - datetime.date.fromisoformat(BASE)).days


FIXINGS = [(_offset(r[0]['.Timestamp']), _offset(r[1]['.Timestamp']))
           for r in _D['TARF_ExpiryDates']]


def _r_usd(t):
    import numpy as _np
    ts, vs = zip(*USD_CURVE)
    return float(_np.interp(t, ts, vs))


def _ndtr(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _expected():
    """Value and dV/dspot of the strip of Europeans an unreachable target degenerates to."""
    value = delta = 0.0
    for fix, settle in FIXINGS:
        t, ts = fix / DAYS, settle / DAYS
        fwd = math.exp((_r_usd(t) - Q_EUR) * t)
        F, sd, D = SPOT * fwd, SIGMA * math.sqrt(fix / DAYS), math.exp(-_r_usd(ts) * ts)
        d1 = (math.log(F / STRIKE) + 0.5 * sd * sd) / sd
        d2 = d1 - sd
        value += D * (N1 * (F * _ndtr(d1) - STRIKE * _ndtr(d2)) -
                      N2 * (STRIKE * _ndtr(-d2) - F * _ndtr(-d1)))
        delta += D * fwd * (N1 * _ndtr(d1) + N2 * _ndtr(-d1))
    return value, delta


EXPECTED_VALUE, EXPECTED_DELTA = _expected()


def _job(greeks='No', **deal_overrides):
    """The canonical document, varied - the switches are overrides on one template."""
    job = _template()
    job['Calc']['Calculation']['Greeks'] = greeks
    _deal_of(job).update(deal_overrides)
    return job


def _run(job, tmp_path, name='tarf', debug=False):
    """JSON in, results out - and the DEBUG log the run emitted, which is the other half of what
    a test asserts: the value says the arithmetic is right, the log says the pricer decided what
    it was supposed to decide."""
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


def _mtm(out, ref='T1'):
    rows = out['Results']['mtm']
    rows = rows[rows['Reference'] == ref]
    return float(rows['Value'].iloc[0])


def test_an_unreachable_target_is_a_strip_of_europeans(tmp_path):
    out, _ = _run(_job(), tmp_path)
    v = _mtm(out)
    assert abs(v - EXPECTED_VALUE) / abs(EXPECTED_VALUE) < TOL, (v, EXPECTED_VALUE)


def test_the_tarf_reports_the_expected_fx_delta(tmp_path):
    """The greek as its own statement: a value that is right with a delta that is wrong is the
    failure a price-only gate cannot see."""
    out, _ = _run(_job(greeks='First'), tmp_path)
    frame = out['Results']['Greeks_First']
    column = [c for c in frame.columns if c != 'Value'][0]
    index, = [i for i in frame.index if str(i[0]) == 'FxRate.EUR']
    delta = float(frame.loc[index, column])
    assert abs(delta - EXPECTED_DELTA) / abs(EXPECTED_DELTA) < 2e-2, (delta, EXPECTED_DELTA)


def test_the_pricer_logs_what_it_decided(tmp_path):
    """The pre-registered log line, diffed against what the run emits.

    A value can be right while the pricer classified its fixings wrongly - counted one as already
    observed, or split the book into the wrong blocks - and on this fixture those mistakes are
    invisible in the number. The log is where they are not.
    """
    _, log = _run(_job(), tmp_path, 'tarflog', debug=True)
    lines = [ln for ln in log.splitlines() if 'TARF ' in ln and 'fixings=' in ln]
    assert lines, 'the TARF logged nothing at DEBUG'
    organ = lines[-1]
    assert 'fixings=2' in organ, organ
    assert 'resolved=0' in organ, organ          # nothing is observed on this document
    assert 'blocks=1' in organ, organ            # one date, so one block


def test_buy_sell_mirrors_exactly(tmp_path):
    buy, _ = _run(_job(), tmp_path, 'buy')
    sell, _ = _run(_job(Buy_Sell='Sell'), tmp_path, 'sell')
    assert abs(_mtm(buy) + _mtm(sell)) <= 1e-9 * abs(_mtm(buy))


def test_a_reachable_target_is_worth_less_than_an_unreachable_one(tmp_path):
    """The target is what makes a TARF a TARF: knocking out early can only remove cashflows the
    holder was accruing, so a small target must move the value toward zero from the strip."""
    strip, _ = _run(_job(), tmp_path, 'strip')
    knocked, _ = _run(_job(TargetLevel=0.02), tmp_path, 'knocked')
    assert abs(_mtm(knocked)) < abs(_mtm(strip)), (_mtm(knocked), _mtm(strip))
