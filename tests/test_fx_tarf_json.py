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
# THE OBSERVED FIXING REDEEMS - and the remaining target it is held under
#
# `intr` clamps at the remaining target, and what makes the BLOCK's clamp the PATH's own is the
# one-step survival truncation capping the drawn spot at `B_pnl = K + (R/N)*cp`. An OBSERVED fixing
# is data and is never truncated, so it clamps at `R` itself: the crossing fixing pays exactly the
# remainder and the deal REDEEMS there, the alive weight zeroing as it does on the simulated side.
# Reachable under plain GBM, which is why it is gated here: a settlement lag spanning two fixing
# periods puts two observed fixings in one block.
# --------------------------------------------------------------------------------------------
#: Two fixings behind the base date settling ahead of it, plus the live one. Both sit inside the
#: one-month window `calc_dependencies` keeps a pre-base fixing by, or the compile drops them.
LAGGED_SCHEDULE = [[{'.Timestamp': '2024-06-10'}, {'.Timestamp': '2024-07-01'}, 1.3],
                   [{'.Timestamp': '2024-06-20'}, {'.Timestamp': '2024-07-05'}, 1.3],
                   [{'.Timestamp': '2024-09-27'}, {'.Timestamp': '2024-09-29'}, 0.0]]

#: Large enough that neither OBSERVED fixing exhausts it - 0.5 against two 0.2 accruals - so both
#: bank what they are worth and the live fixing behind them is capped by what they left.
LAGGED_TARGET = 0.5

#: Exactly what they leave. A deal written to it CROSSES at the first observed fixing.
REDEEM_TARGET = 0.1


def _lagged_job(smooth=False, **overrides):
    job = _job(TARF_ExpiryDates=LAGGED_SCHEDULE, TargetLevel=LAGGED_TARGET,
               Expiry_Date={'.Timestamp': '2024-09-29'})
    job['Calc']['Calculation']['MCMC_Simulations'] = 1 << 14
    if smooth:
        job['Calc']['Calculation']['Branch_And_Weight'] = 'Yes'
    _deal_of(job).update(overrides)
    return job


def _leg(row, accrual):
    """One schedule row's payment at its own settlement, discounted to the base date."""
    ts = _offset(row[1]['.Timestamp']) / DAYS
    return math.exp(-_r_usd(ts) * ts) * accrual * N1


def _black(fwd, k, sd, call=True):
    return ((fwd * _ndtr((math.log(fwd / k) + 0.5 * sd * sd) / sd) -
             k * _ndtr((math.log(fwd / k) - 0.5 * sd * sd) / sd)) if call else
            (k * _ndtr(-(math.log(fwd / k) - 0.5 * sd * sd) / sd) -
             fwd * _ndtr(-(math.log(fwd / k) + 0.5 * sd * sd) / sd)))


def _live_leg(row, level, remaining):
    """The one SIMULATED fixing, walking on from the last observed level under a remaining target
    of `remaining`.

    However the one-step survival splits it, that fixing pays `min(relu(S - K), remaining)` on
    `N1` - the knocked weight banks the remainder, the surviving one its own intrinsic - so the ITM
    leg is the CALL SPREAD `C(K) - C(K + remaining)`. The OTM leg is unclamped and lies wholly
    inside the surviving set, so it is the plain leveraged put.
    """
    t, ts = (_offset(row[0]['.Timestamp']) / DAYS, _offset(row[1]['.Timestamp']) / DAYS)
    fwd, sd = level * math.exp((_r_usd(t) - Q_EUR) * t), SIGMA * math.sqrt(t)
    return math.exp(-_r_usd(ts) * ts) * (
        N1 * (_black(fwd, STRIKE, sd) - _black(fwd, STRIKE + remaining, sd)) -
        N2 * _black(fwd, STRIKE, sd, call=False))


def test_two_observed_fixings_in_one_settlement_lag_bank_their_own_accruals(tmp_path):
    """The pot a block opens on, on the one branch where a fixing is data rather than a draw.

    Two fixings are already FIXED and not yet SETTLED, so both walk in-loop with `p = 1` and no
    survival cap - and the opening accrual must not have netted them, or the loop pays what the pot
    has already taken. It nets the SETTLED fixings only, of which there are none here: `R` opens at
    the whole 0.5, each observed fixing banks its own 0.2 at its own settlement, and the live fixing
    behind them is capped by the 0.1 they left.

    THE ORACLE is those two banked legs plus a closed form for the live one - a call spread against
    a leveraged put, no target arithmetic left in it. It agrees at 3.9e-6, which is the inner Sobol
    draw and not a model difference; the gate is set 25x wider.

    MEASURED: 497.1924 against an oracle of 497.1905. With every declared reset in the pot, settled
    or not, the same document read 99.98989 - the first fixing banking 0.1, the remainder that
    netting had already left, rather than the 0.2 it is worth.

    BOTH ESTIMATORS, because both reach it: an OBSERVED fixing builds no kink term, so the smooth
    arm's decrement is the crisp one's and the two agree to the bit.
    """
    accrual = LAGGED_SCHEDULE[0][2] - STRIKE
    oracle = (_leg(LAGGED_SCHEDULE[0], accrual) + _leg(LAGGED_SCHEDULE[1], accrual) +
              _live_leg(LAGGED_SCHEDULE[2], LAGGED_SCHEDULE[1][2],
                        LAGGED_TARGET - 2.0 * accrual))
    value = _mtm(_run(_lagged_job(), tmp_path, 'lagged')[0])
    assert abs(value - oracle) < 1e-4 * oracle, (value, oracle)

    smooth = _mtm(_run(_lagged_job(smooth=True), tmp_path, 'lagged_smooth')[0])
    assert smooth == value, ('the smooth arm decremented differently', smooth, value)


def test_a_redeemed_deal_pays_nothing_after_the_crossing_fixing(tmp_path):
    """Redemption zeroes the alive weight, so nothing a later fixing declares can reach the mark.

    Written to `REDEEM_TARGET`, the first observed fixing is worth more than the whole target: it
    CROSSES, paying exactly the remainder, and the second observed fixing and the live one fall
    behind a zero weight.

    The second observed fixing's SETTLEMENT is pushed out a month - its discount factor, and nothing
    else the block reads. A deal still paying it moves by that factor on 0.1 of target; a redeemed
    one cannot see it at all, and the two runs must agree bit for bit.

    Its LEVEL is deliberately left alone: it is the level the crossing is measured against, so
    moving it moves the shape this gate is about rather than the discounting.
    """
    later = [LAGGED_SCHEDULE[0],
             [LAGGED_SCHEDULE[1][0], {'.Timestamp': '2024-08-05'}, LAGGED_SCHEDULE[1][2]],
             LAGGED_SCHEDULE[2]]
    base = _mtm(_run(_lagged_job(TargetLevel=REDEEM_TARGET), tmp_path, 'redeem')[0])
    moved = _mtm(_run(_lagged_job(TargetLevel=REDEEM_TARGET, TARF_ExpiryDates=later),
                      tmp_path, 'redeem_moved')[0])
    assert moved == base, ('the deal paid after it redeemed', moved, base)
    assert abs(base - _leg(LAGGED_SCHEDULE[0], REDEEM_TARGET)) < 1e-9 * base, base


def test_the_second_observed_fixing_in_a_block_reads_its_own_level(tmp_path):
    """The strip's j-th observed fixing, and not its first one twice.

    `past_fixings` was indexed without `j`, so every observed fixing of a block read the SAME
    resolved sample. The gate above cannot see it - both of its fixings are declared at 1.3 - and
    neither can any block holding one observed fixing, which is every fixture that reaches a
    reporting row between a fixing and its settlement.

    So: two observed fixings at DIFFERENT levels, an UNREACHABLE target so the per-fixing remainder
    never binds, and nothing live behind them. The mark is then both intrinsics at their own
    settlements and no model at all - an equality. Reading the first level twice reads 0.2 where
    0.15 is due on the second leg.

    The live fixing is dropped on purpose: it walks on from the last OBSERVED level, so it moves
    with the very thing this gate varies and would have to be subtracted rather than gated.
    """
    schedule = [LAGGED_SCHEDULE[0],
                [LAGGED_SCHEDULE[1][0], LAGGED_SCHEDULE[1][1], LAGGED_SCHEDULE[1][2] - 0.05]]
    oracle = sum(math.exp(-_r_usd(t) * t) * (row[2] - STRIKE) * N1
                 for row, t in ((r, _offset(r[1]['.Timestamp']) / DAYS) for r in schedule))

    value = _mtm(_run(_lagged_job(TargetLevel=UNREACHABLE, TARF_ExpiryDates=schedule,
                                  Expiry_Date=LAGGED_SCHEDULE[1][1]), tmp_path, 'own_level')[0])
    assert abs(value - oracle) < 1e-9 * oracle, (
        'the observed fixings are not reading their own levels', value, oracle)
