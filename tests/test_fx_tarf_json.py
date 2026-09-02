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


# --------------------------------------------------------------------------------------------
# THE REMAINING TARGET IS CLAMPED AT ZERO - and the branch where that clamp is load-bearing
#
# `intr` clamps at the BLOCK's remaining target, and what makes that the PATH's own is the one-step
# survival truncation capping the drawn spot at `B_pnl = K + (R/N)*cp`. Two branches have no such
# cap: an OBSERVED fixing is data and is never truncated, and under Heston-Nandi a cap many daily
# sigmas away saturates the draw. Both walk R below zero, which is a knocked-out weight paying a
# NEGATIVE remaining target. The observed-fixing shape is reachable under plain GBM, which is why it
# is gated here: a settlement lag spanning two fixing periods puts two of them in one block.
# --------------------------------------------------------------------------------------------
#: Two fixings behind the base date settling ahead of it, plus the live one. Both sit inside the
#: one-month window `calc_dependencies` keeps a pre-base fixing by, or the compile drops them.
LAGGED_SCHEDULE = [[{'.Timestamp': '2024-06-10'}, {'.Timestamp': '2024-07-01'}, 1.3],
                   [{'.Timestamp': '2024-06-20'}, {'.Timestamp': '2024-07-05'}, 1.3],
                   [{'.Timestamp': '2024-09-27'}, {'.Timestamp': '2024-09-29'}, 0.0]]

#: Small enough that the first observed fixing alone exhausts it, so the second one's accrual has
#: nothing left to take and the clamp is what decides the answer. 0.5 against two 0.2 accruals.
LAGGED_TARGET = 0.5


def _lagged_job(smooth=False, **overrides):
    job = _job(TARF_ExpiryDates=LAGGED_SCHEDULE, TargetLevel=LAGGED_TARGET,
               Expiry_Date={'.Timestamp': '2024-09-29'})
    job['Calc']['Calculation']['MCMC_Simulations'] = 1 << 14
    if smooth:
        job['Calc']['Calculation']['Branch_And_Weight'] = 'Yes'
    _deal_of(job).update(overrides)
    return job


def test_two_observed_fixings_in_one_settlement_lag_cannot_drive_the_target_negative(tmp_path):
    """The clamp on the remaining target, on the one branch that has no truncation to do it.

    Two fixings are already FIXED and not yet SETTLED, so both walk in-loop with `p = 1` and no
    survival cap. `intr` clamps at the BLOCK's remaining target (0.1 after the declared prefix), so
    each accrues the whole of it: the second takes `R` from 0 to -100 unless the decrement is
    clamped, and a negative `R` is a `B_pnl` BELOW the strike and a knocked-out weight that PAYS a
    negative number.

    THE ORACLE. Clamped, `R` is zero at the live fixing, so `B_pnl` is the strike and the survivors
    are the paths that fell from 1.3 to below 1.10 in three months - `Phi(z_max)`, 1.5e-4 here. What
    is left is the two banked accruals at their own settlements.

    MEASURED: 199.9566 clamped against 102.0116 unclamped, the 97.9 between them being
    `(1 - p) * R * D` at `R = -100`. GBM, with no Heston-Nandi anywhere.

    BOTH ESTIMATORS, because both reach it. The smooth arm decrements without the mask and the clamp
    - deliberately, since both would sever the kink term's curvature from its trigger - but an
    OBSERVED fixing builds no kink term, so there the clamp costs nothing and the two arms agree to
    the bit. Unclamped, the smooth arm returned the crisp arm's unclamped number exactly.

    WHAT THIS DOES NOT BLESS: that each of two observed fixings accrues the whole block remaining
    target is a separate, pre-existing question about the clamp's own bound, as is the redemption
    that does not fire on an observed fixing. This gate pins the decrement, not the bound.
    """
    accrued = LAGGED_TARGET - 2.0 * (LAGGED_SCHEDULE[0][2] - STRIKE)   # 0.1 left of the target
    banked = sum(math.exp(-_r_usd(_offset(row[1]['.Timestamp']) / DAYS) *
                          _offset(row[1]['.Timestamp']) / DAYS) * accrued * N1
                 for row in LAGGED_SCHEDULE[:2])

    # the live fixing walks on from the last OBSERVED level, capped at the strike by R == 0
    t = _offset(LAGGED_SCHEDULE[2][0]['.Timestamp']) / DAYS
    ts = _offset(LAGGED_SCHEDULE[2][1]['.Timestamp']) / DAYS
    fwd = LAGGED_SCHEDULE[0][2] * math.exp((_r_usd(t) - Q_EUR) * t)
    sd = SIGMA * math.sqrt(t)
    survival = _ndtr((math.log(STRIKE / fwd) + 0.5 * sd * sd) / sd)
    bound = math.exp(-_r_usd(ts) * ts) * N2 * STRIKE * survival

    value = _mtm(_run(_lagged_job(), tmp_path, 'lagged')[0])
    assert survival < 1e-3, 'the fixture stopped isolating the banked legs (%.3e)' % survival
    assert abs(value - banked) < max(bound, 1e-2), (value, banked, bound)

    smooth = _mtm(_run(_lagged_job(smooth=True), tmp_path, 'lagged_smooth')[0])
    assert smooth == value, ('the smooth arm decremented past zero', smooth, value)
