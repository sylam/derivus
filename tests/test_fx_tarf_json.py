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


def _expected(cp=1.0):
    """Value and dV/dspot of the strip of Europeans an unreachable target degenerates to, at the
    deal's own side `cp`: the ITM leg on `Underlying_Amount` against the leveraged OTM one."""
    value = delta = 0.0
    for fix, settle in FIXINGS:
        t, ts = fix / DAYS, settle / DAYS
        fwd = math.exp((_r_usd(t) - Q_EUR) * t)
        F, sd, D = SPOT * fwd, SIGMA * math.sqrt(fix / DAYS), math.exp(-_r_usd(ts) * ts)
        d1 = (math.log(F / STRIKE) + 0.5 * sd * sd) / sd
        d2 = d1 - sd
        value += cp * D * (N1 * (F * _ndtr(cp * d1) - STRIKE * _ndtr(cp * d2)) -
                           N2 * (STRIKE * _ndtr(-cp * d2) - F * _ndtr(-cp * d1)))
        delta += cp * D * fwd * (N1 * _ndtr(cp * d1) + N2 * _ndtr(-cp * d1))
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


#: What the fixture as it stands marks - the control that the cap's mask is inert wherever the cap
#: is live, the CALL side being the one where it never goes negative.
BANKED_CALL = float.fromhex('0x1.f36e678d65e72p+5')


def test_a_put_target_above_its_strike_is_the_uncapped_strip(tmp_path):
    """A fill no single fixing can reach is not a level the pricer may take a log of.

    One-step survival standardises the PnL cap `B = K + (R/N)*cp`, which for a PUT is the strike
    LESS the remaining target per unit - negative at every fixing whose target is above the strike,
    and `log(B/S)` then takes the whole deal to not-a-number. A put accrues at most the strike at
    one fixing, so fifty and five times it are both unreachable; on main both mark NaN, as does a
    USDZAR put TARF at every target above its strike (positive at 0.5, negative at 5, NaN at
    50 and above).

    A cap at or below zero cannot be crossed, so survival is one and the deal is the UNCAPPED
    STRIP: the two targets agree to the BIT, the only quantity differing between them being the
    remaining target the mask takes out of the arithmetic, and both land on the closed-form strip -
    `Underlying_Amount` puts against the leveraged calls - at 0.066%.

    MUTATION: drop the mask and the first assertion reads NaN. Substitute the OTHER infinity and
    the step knocks out with certainty, paying the remaining target - ten times apart on the two
    arms, which the bit-equality catches. The banked call is the third arm: a live cap is masked by
    nothing and must not move.
    """
    fifty = _mtm(_run(_job(Option_Type='Put', TargetLevel=50.0 * STRIKE), tmp_path, 'put50')[0])
    five = _mtm(_run(_job(Option_Type='Put', TargetLevel=5.0 * STRIKE), tmp_path, 'put5')[0])
    strip, _ = _expected(-1.0)

    assert math.isfinite(fifty), 'the put marked not-a-number above its strike'
    assert fifty == five, ('the masked cap leaked the remaining target', fifty, five)
    assert abs(fifty - strip) / abs(strip) < TOL, (fifty, strip)
    assert _mtm(_run(_job(), tmp_path, 'banked')[0]) == BANKED_CALL, 'a live cap moved'


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
    job['Calc']['Calculation']['Branch_And_Weight'] = 'Yes' if smooth else 'No'
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


# --------------------------------------------------------------------------------------------
# A SEASONED TARF PRICES WHAT IS LEFT OF IT
#
# A row whose SETTLEMENT date is behind the base date has paid: its accrual opens the pot the
# strip walks from, and both its dates leave the grid so every remaining fixing is paired with
# its own settlement again. The oracle is the SUBSTITUTED deal - the same document with that row
# deleted and `TargetLevel` reduced by its accrual - which walks the same fixings and draws the
# same numbers, so the equality is to the BIT once the reduced target is the same double.
#
# The surface is SKEWED and the spot is the level the last observed fixing printed at: the strip
# walks on from an observed level, so an oracle written at another spot reads the smile elsewhere.
# --------------------------------------------------------------------------------------------
#: (Option_Type, spot, the settled fixing, the observed-but-unsettled one), each accruing on its
#: own side of the strike - 0.05 settled and 0.04 observed, either way round.
SEASONED_WORLDS = [('Call', 1.14, 1.15, 1.14), ('Put', 1.06, 1.05, 1.06)]
SEASONED_SKEW = [[0.8, 0.02, 0.13], [0.8, 2.0, 0.125], [1.0, 0.02, 0.10], [1.0, 2.0, 0.105],
                 [1.2, 0.02, 0.09], [1.2, 2.0, 0.095]]
#: settles BEFORE the base date, its fixing close enough behind it that the compile used to keep
#: the fixing while dropping the settlement - which is what left the strip short a discount factor
SETTLED_DATES = [{'.Timestamp': '2024-06-10'}, {'.Timestamp': '2024-06-12'}]
#: fixed before the base date, settling after it
OBSERVED_DATES = [{'.Timestamp': '2024-06-27'}, {'.Timestamp': '2024-07-01'}]
SEASONED_TARGET = 0.12
LIVE_ROWS = _D['TARF_ExpiryDates']
SEASONED_SIMS = 1 << 16
#: a LIVE NIG kit for the pair, so the seasoned identity is read on the walk as well as on GBM
LV_FACTOR = {
    'Kappa_L': 0.5, 'Sigma_L': 1.0, 'Rho_L': -0.4, 'Kappa_S': 6.0, 'C_Min': 0.12,
    'Cap_A': 4.605170185988092, 'Residual_Law': 'NIG', 'On_Guard': '', 'Skew_Gradient': '',
    'Stickiness_Band': 0.0,
    'Xi_Curve': {'.Curve': {'meta': [], 'data': [[0.0, 0.01], [5.0, 0.01]]}},
    'Rho_S': {'.Curve': {'meta': [], 'data': [[0.0, -0.6]]}},
    'Sigma_S': {'.Curve': {'meta': [], 'data': [[0.0, 2.4]]}},
    'Alpha': {'.Curve': {'meta': [], 'data': [[0.0, 8.0]]}},
    'Beta': {'.Curve': {'meta': [], 'data': [[0.0, -2.0]]}}}


def _accrual(option_type, level):
    """One fixing's accrual toward the target - the pricer's own expression, to the bit."""
    return max((level - STRIKE) if option_type == 'Call' else (STRIKE - level), 0.0)


def _seasoned(rows, target, spot, model='None', smooth=False, sims=SEASONED_SIMS, **overrides):
    """The template reseated on `rows`: a skewed surface and a spot on the last observed level."""
    job = _job(TARF_ExpiryDates=rows, TargetLevel=target, Expiry_Date=rows[-1][1], **overrides)
    job['Calc']['Calculation']['MCMC_Simulations'] = sims
    job['Calc']['Calculation']['Branch_And_Weight'] = 'Yes' if smooth else 'No'
    market = job['Calc']['MergeMarketData']['ExplicitMarketData']
    market['Price Factors']['FxRate.EUR']['Spot'] = spot
    market['Price Factors']['FXVol.EUR.USD']['Surface']['.Curve']['data'] = SEASONED_SKEW
    if model != 'None':
        market['Price Factors']['LogVar2FJModelParameters.EUR'] = LV_FACTOR
        market['Valuation Configuration'] = {'FXTARFOptionDeal': {
            'SpotModel': model, 'Steps_Per_Year': 252.0, 'Internal_Step_Days': 1}}
    return job


@pytest.mark.parametrize('option_type,spot,settled,observed', SEASONED_WORLDS)
@pytest.mark.parametrize('buy_sell', ['Buy', 'Sell'])
def test_a_settled_fixing_opens_the_pot_rather_than_vanishing(
        tmp_path, option_type, spot, settled, observed, buy_sell):
    """A fixing that has already paid is not a fixing the deal has stopped having.

    Both its dates were cut at the base date and its accrual went nowhere, so the deal priced
    against its WHOLE original target - bit-identical to deleting the row. Measured here: a bought
    call read 80.7373 where the deal is worth 46.4464, 73.8% over; a bought put -71.6956 against
    -80.0158, 10.4% under. Which way it errs is the payoff's, not the defect's.

    Equal to the BIT, not approximately: the substituted deal walks the same two fixings and draws
    the same Sobol numbers, and the only quantity that differs is the remaining target - the same
    double on both sides, since the oracle reduces it by the pricer's own accrual expression.
    """
    side = dict(Option_Type=option_type, Buy_Sell=buy_sell)
    rows = [SETTLED_DATES + [settled]] + LIVE_ROWS
    accrual = _accrual(option_type, settled)
    seasoned = _mtm(_run(_seasoned(rows, SEASONED_TARGET, spot, **side), tmp_path, 'seasoned')[0])
    substituted = _mtm(_run(_seasoned(LIVE_ROWS, SEASONED_TARGET - accrual, spot, **side),
                            tmp_path, 'substituted')[0])
    full = _mtm(_run(_seasoned(LIVE_ROWS, SEASONED_TARGET, spot, **side), tmp_path, 'full')[0])
    assert seasoned == substituted, (
        'the settled fixing did not open the pot', seasoned, substituted)
    assert seasoned != full, (
        'the deal prices against its whole original target, so the settled fixing reached nothing')


@pytest.mark.parametrize('option_type,spot,settled,observed', SEASONED_WORLDS)
@pytest.mark.parametrize('buy_sell', ['Buy', 'Sell'])
def test_a_fixing_observed_but_not_settled_banks_its_own_settlement(
        tmp_path, option_type, spot, settled, observed, buy_sell):
    """A row fixed before the base date and settling after it is still the STRIP's.

    Beside a settled row it was also the crash: the settled fixing survived the window the compile
    kept a pre-base fixing by while its settlement did not, so the strip carried four fixings
    against three discount factors and ran off the end of them. The deal was skipped and the book
    marked NOT A NUMBER - which is what this document read on main, on every one of these arms.

    Two oracles. Deleting the SETTLED row and reducing the target by its accrual leaves the same
    three-fixing strip, so that equality is to the bit. Deleting the observed row as well pays its
    settlement in cash instead, and that one is the estimator's own: the shorter strip draws one
    fewer Sobol dimension, which is 1.1e-4 of the mark for the call and 1.9e-4 for the put at the
    262,144 inner paths used here, and 4.5e-3 at 65,536.
    """
    side = dict(Option_Type=option_type, Buy_Sell=buy_sell, sims=1 << 18)
    sign = 1.0 if buy_sell == 'Buy' else -1.0
    settled_accrual, observed_accrual = (_accrual(option_type, settled),
                                         _accrual(option_type, observed))
    rows = [SETTLED_DATES + [settled], OBSERVED_DATES + [observed]] + LIVE_ROWS
    seasoned = _mtm(_run(_seasoned(rows, SEASONED_TARGET, spot, **side), tmp_path, 'both')[0])
    assert math.isfinite(seasoned), 'the seasoned deal marked not-a-number'
    kept = _mtm(_run(_seasoned(rows[1:], SEASONED_TARGET - settled_accrual, spot, **side),
                     tmp_path, 'kept')[0])
    assert seasoned == kept, ('the settled row did not fold into the pot', seasoned, kept)

    banked = sign * _leg(OBSERVED_DATES, observed_accrual)
    live = _mtm(_run(_seasoned(LIVE_ROWS, SEASONED_TARGET - settled_accrual - observed_accrual,
                               spot, **side), tmp_path, 'live')[0])
    assert abs(seasoned - live - banked) < 5e-4 * abs(seasoned), (seasoned, live + banked)


def test_settled_fixings_that_reach_the_target_redeem_the_deal(tmp_path):
    """There is nothing left to price: the pot the settled rows leave IS the target, so the deal
    is worth zero, simulates nothing and settles nothing.

    Said by name in the log, because a row marking flat is otherwise indistinguishable from one
    nobody priced. On main the same document read 10.1089 - a deal still carrying its whole target.
    """
    option_type, spot, settled, _observed = SEASONED_WORLDS[0]
    rows = [SETTLED_DATES + [settled]] + LIVE_ROWS
    out, log = _run(_seasoned(rows, _accrual(option_type, settled) / 2.0, spot,
                              Option_Type=option_type), tmp_path, 'redeemed', debug=True)
    assert _mtm(out) == 0.0, _mtm(out)
    assert any('redeemed before the base date' in line for line in log.splitlines()), log


def test_a_fixing_whose_date_has_passed_must_carry_the_rate_it_fixed_at(tmp_path):
    """A blank row behind the base date records nothing, and reading it as a zero rate accrues
    nothing where the deal accrued - the same mis-mark as dropping the row. Refused by name at
    the compile, where the remedy is the observation."""
    rows = [SETTLED_DATES + [0.0]] + LIVE_ROWS
    with pytest.raises(rf.utils.UnpriceableSchedule):
        _run(_seasoned(rows, SEASONED_TARGET, 1.14), tmp_path, 'blank')


@pytest.mark.parametrize('model', ['None', 'LogVar2FJ'])
@pytest.mark.parametrize('smooth', [False, True])
def test_the_seasoned_pot_is_one_number_on_every_arm(tmp_path, model, smooth):
    """The crisp estimator and the default, GBM and the LogVar2FJ kit: four estimators, one
    spelling of what the settled fixings left, and the substituted identity is to the bit on all
    of them. Measured: 46.5710 on GBM and 40.3114 on the walk, each unmoved by the switch."""
    option_type, spot, settled, observed = SEASONED_WORLDS[0]
    arm = dict(Option_Type=option_type, model=model, smooth=smooth)
    rows = [SETTLED_DATES + [settled], OBSERVED_DATES + [observed]] + LIVE_ROWS
    seasoned = _mtm(_run(_seasoned(rows, SEASONED_TARGET, spot, **arm), tmp_path, 'arm')[0])
    substituted = _mtm(_run(_seasoned(rows[1:], SEASONED_TARGET - _accrual(option_type, settled),
                                      spot, **arm), tmp_path, 'arm_sub')[0])
    assert math.isfinite(seasoned) and seasoned == substituted, (model, smooth, seasoned,
                                                                 substituted)
