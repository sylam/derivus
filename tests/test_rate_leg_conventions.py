"""A floating rate leg's three declared conventions - `Reset_Type: Advance`, `Payment_Timing:
Discounted` and `Index_Publication_Calendars` - and a deposit's principal under each timing, each
against a hand derivation, through the JSON contract.

The world is the one the conventions were measured on: USD-PROJ's zero rates rising from 2% to 6%
over five years and USD 25bp under it, both ACT/365 and LINEAR IN TENOR, so the engine's own
interpolation reproduces them exactly and every forward below is read off the authored line by
hand; a two-year quarterly swap of a million from 16 December 2026 paying 4% ACT/360 against the
index, on ACT/360 legs:

    r(T) = a + b T,   P(T) = exp(-r(T) T)                 T in ACT/365 years from the base date
    F(s, e) tau(s, e) = P(s) / P(e) - 1                    tau ACT/360, the index's and the leg's

A flat curve reads a reset taken a period early as the same rate, and one curve discounts a period
at its own rate, which is why both curves slope and sit apart.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import pytest
import torch

import derivus
import test_declared_defaults as book
import trial_rates
from derivus import spine, utils
from derivus.config import CustomJsonEncoder
from rates_world import BASE, deposit, par_swap

#: `(a, b)` of each zero curve, the `Price Factors` written from them below.
PROJECTION, DISCOUNT = (0.02, 0.008), (0.0175, 0.008)
FACTORS = {'InterestRate.' + name: {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                                    'Curve': utils.Curve([], [(t, a + b * t) for t in (0, 0.5, 1, 2, 3, 4, 5)])}
           for name, (a, b) in (('USD-PROJ', PROJECTION), ('USD', DISCOUNT))}

START, END, QUARTER, N, RATE = pd.Timestamp('2026-12-16'), pd.Timestamp('2028-12-16'), \
    pd.DateOffset(months=3), 1e6, 0.04
DATES = [END - QUARTER * k for k in range(9)][::-1]
SWAP = dict(par_swap('SWAP', 'USD', 'USD-PROJ', 'USD', 2, 100 * RATE, fixed_frequency=3, float_frequency=3),
            Effective_Date=START, Maturity_Date=END)
#: Johannesburg's weekday holidays from 16 December 2026 for a year, in the engine's calendar file.
HOLIDAYS = {pd.Timestamp(day) for day in (
    '2026-12-16', '2026-12-25', '2027-01-01', '2027-03-22', '2027-03-26', '2027-03-29', '2027-04-27',
    '2027-06-16', '2027-08-09', '2027-09-24')}


def P(day, curve=PROJECTION):
    years = (day - BASE).days / 365.0
    return float(np.exp(-(curve[0] + curve[1] * years) * years))


def D(day):
    return P(day, DISCOUNT)


def tau(start, end):
    return (end - start).days / 360.0


def growth(start, end):
    """`F tau` over [start, end] off the projection curve."""
    return P(start) / P(end) - 1.0


def fixed_leg(dates=DATES):
    """The 4% leg paid, each coupon at its period's end."""
    return -sum(N * RATE * tau(s, e) * D(e) for s, e in zip(dates[:-1], dates[1:]))


def marks(deals, factors=FACTORS):
    """`{reference: mark}` off one base valuation of `deals`, every one of them priced."""
    hexes, stats = book.marks(deals, factors=factors)
    assert 'Deals Skipped' not in stats, stats
    return {reference: float.fromhex(value) for reference, value in hexes.items()}


def bp(amount):
    """`amount` in basis points of the notional."""
    return 1e4 * amount / N


def test_an_advance_reset_fixes_on_the_reset_its_period_before_takes():
    """`Reset_Type` places each period's single reset: `Standard` at its start, `Arrears` at its end
    and `Advance` at its predecessor's start, the first period a quarter before its own, each over
    the window of its period's length from there and fixing two weekdays before it, which a forward
    base valuation does not read. By hand, with p the reset day and L the period's days,

        V = fixed_leg + sum N (P(p) / P(p + L) - 1) D(e)

    each to 1e-12. On this world `Advance` marks -5,065.11 against `Standard`'s 2,668.65, -77.3bp
    of notional, the size the convention was measured at while it read as `Standard`.

    Killing mutation: `Advance` read as `Standard`, +7,733.76 against the hand.
    """
    before = {DATES[0]: DATES[0] - QUARTER, **dict(zip(DATES[1:], DATES[:-1]))}
    resets = {'Standard': lambda s, e: s, 'Arrears': lambda s, e: e, 'Advance': lambda s, e: before[s]}
    values = marks([dict(SWAP, Reference=kind, Reset_Type=kind, Index_Offset=2) for kind in resets])
    for kind, reset in resets.items():
        hand = fixed_leg() + sum(N * growth(reset(s, e), reset(s, e) + (e - s)) * D(e)
                                 for s, e in zip(DATES[:-1], DATES[1:]))
        assert abs(values[kind] / hand - 1.0) < 1e-12, (kind, values[kind], hand)
    assert round(bp(values['Advance'] - values['Standard']), 1) == -77.3, values


def discounted_leg(dates=DATES, curve=DISCOUNT):
    """Both legs paid at each period's start, each amount discounted over the period at its own
    rate: `F tau / (1 + F tau) = 1 - P(e) / P(s)` received and `r tau / (1 + r tau)` paid."""
    return sum(N * ((1.0 - P(e) / P(s)) - RATE * tau(s, e) / (1.0 + RATE * tau(s, e))) * P(s, curve)
               for s, e in zip(dates[:-1], dates[1:]))


def test_a_discounted_period_pays_at_its_start_what_its_own_rate_discounts():
    """`Payment_Timing: Discounted` pays each period at its start, its amount discounted over the
    period at the period's own rate - the fixing on the floating leg and 4% on the fixed one - so
    the swap with both legs `Discounted` is `discounted_leg()` to 1e-12, 2,639.78. Its floating leg
    alone reads 0.50bp of notional under the leg paid at its end (2,618.44 against 2,668.65 with
    the fixed leg paid at its end), the 25bp basis between the curve a period is discounted on and
    the rate it fixes; on one curve the two are the same to 1e-12. The diary announces each payment
    on its period's start, the fixed leg's amount the discounted coupon `-N r tau / (1 + r tau)`
    and the floating leg's undetermined.

    Killing mutation: `Discounted` read as `End`, +28.87 against the hand and every diary row on
    its period's end.
    """
    both = dict(SWAP, Reference='BOTH', Pay_Timing='Discounted', Receive_Timing='Discounted')
    floating = dict(SWAP, Reference='FLOATING', Receive_Timing='Discounted')
    single = [dict(deal, Reference=deal['Reference'] + '_ONE', Discount_Rate='USD-PROJ') for deal in (SWAP, floating)]
    values = marks([SWAP, both, floating] + single)
    assert abs(values['BOTH'] / discounted_leg() - 1.0) < 1e-12, (values['BOTH'], discounted_leg())
    assert round(bp(values['SWAP'] - values['FLOATING']), 2) == 0.50, values
    assert abs(values['FLOATING_ONE'] / values['SWAP_ONE'] - 1.0) < 1e-12, values

    job = book.book([both])
    job['Calc']['MergeMarketData']['ExplicitMarketData']['Price Factors'].update(FACTORS)
    context = derivus.Context()
    context.load_json((json.dumps(job, cls=CustomJsonEncoder), 'discounted'))
    due = {}
    for row in spine.diary(context):
        if row['kind'] == 'payment':
            due.setdefault(row['leg'], {})[pd.Timestamp(row['due_date'])] = row['amount']
    assert due['FloatCashflows'] == {s: None for s in DATES[:-1]}, due['FloatCashflows']
    assert list(due['FixedCashflows']) == DATES[:-1], due['FixedCashflows']
    for (s, e), amount in zip(zip(DATES[:-1], DATES[1:]), due['FixedCashflows'].values()):
        assert amount == pytest.approx(-N * RATE * tau(s, e) / (1.0 + RATE * tau(s, e)), rel=1e-12)


def ledger(deal):
    """`(the deal's mean profile by row, its cash booked by day)` off a zero-vol Hull-White credit
    Monte Carlo in float64, where every path is the authored curves' own evolution."""
    _, out = book.simulated([deal], ['USD', 'USD-PROJ'], factors=FACTORS, sigma=0.0, prec=torch.float64,
                            Generate_Cashflows='Yes')
    cash = out['Results']['cashflows']['USD']
    return out['Results']['mtm'].mean(axis=1), {day: float(cash.loc[day].mean()) for day in cash.index
                                                if np.abs(cash.loc[day].values).max() > 0.0}


def test_a_discounted_period_books_its_cash_and_leaves_the_book_at_its_start():
    """Under the credit Monte Carlo a `Discounted` period is cash on its start and nothing after
    it, as an FRA's: the swap with both legs `Discounted` books on each period's start, and on no
    other day, `N (1 - P(e) / P(s)) - N r tau / (1 + r tau)`, to 1e-12, and is worth nothing from
    the day after its last period starts, 16 September 2028. With the floating leg alone
    `Discounted` each day books the floating period starting on it and the fixed one ending on it,
    and the swap is a position until its last fixed coupon.

    Killing mutations: `Discounted` read as `End`, the cash booked on each period's end; a leg
    priced on the rows after its last payment, the swap skipped on an empty reduction and nothing
    valued.
    """
    profile, cash = ledger(dict(SWAP, Pay_Timing='Discounted', Receive_Timing='Discounted'))
    hand = {s: N * ((1.0 - P(e) / P(s)) - RATE * tau(s, e) / (1.0 + RATE * tau(s, e)))
            for s, e in zip(DATES[:-1], DATES[1:])}
    assert list(cash) == list(hand), list(cash)
    for day, amount in hand.items():
        assert abs(cash[day] - amount) < 1e-12 * N, (day, cash[day], amount)
    assert (profile[profile.index > DATES[-2]] == 0.0).all() and (profile[profile.index <= DATES[-2]] != 0.0).all()

    profile, cash = ledger(dict(SWAP, Receive_Timing='Discounted'))
    hand = {day: (N * (1.0 - P(DATES[k + 1]) / P(day)) if k < 8 else 0.0) -
                 (N * RATE * tau(DATES[k - 1], day) if k else 0.0) for k, day in enumerate(DATES)}
    assert list(cash) == DATES, list(cash)
    for day, amount in hand.items():
        assert abs(cash[day] - amount) < 1e-12 * N, (day, cash[day], amount)
    assert (profile[profile.index < END] != 0.0).all()


def test_a_discounted_period_whose_rate_fixes_after_its_start_is_refused_by_name():
    """A `Discounted` period pays at its start, so a reset fixing after that leaves nothing known to
    pay there: an `Arrears` swap leg, a leg averaging a daily index through each period and a
    `Pre_Aggregation` cap capping each day's fixing are refused by name.

    Killing mutation: the refusal dropped, the two swaps priced off rates their payments precede
    and the cap failing on a shape that names nothing.
    """
    cap = {'Object': 'CapDeal', 'Reference': 'CAPLETS', 'Currency': 'USD', 'Discount_Rate': 'USD',
           'Forecast_Rate': 'USD-PROJ', 'Forecast_Rate_Volatility': 'USD-PROJ', 'Buy_Sell': 'Buy',
           'Effective_Date': START, 'Maturity_Date': END, 'Cap_Rate': 4.0, 'Principal': N,
           'Payment_Timing': 'Discounted', 'Averaging_Method': 'Pre_Aggregation',
           'Reset_Frequency': pd.DateOffset(days=1)}
    for deal in (dict(SWAP, Reference='ARREARS', Receive_Timing='Discounted', Reset_Type='Arrears'),
                 dict(SWAP, Reference='DAILY', Receive_Timing='Discounted', Index_Tenor=pd.DateOffset(days=1)),
                 cap):
        with pytest.raises(utils.UnpriceableSchedule, match='Payment_Timing Discounted') as refused:
            book.marks([deal], factors=FACTORS)
        assert deal['Reference'] in str(refused.value), str(refused.value)


#: A year's deposit of a million from 16 December 2026 in quarters, accruing ACT/360 at the
#: forecast or pinned at 4.2%, 250,000 of it amortised in its third quarter: balances `BALANCES`.
DEPOSIT = dict(deposit('DEPOSIT', 'USD', 'USD', 12, 4.2), Effective_Date=START, Maturity_Date=DATES[4],
               Payment_Frequency=QUARTER, Interest_Frequency=QUARTER,
               Amortisation=utils.DateList({pd.Timestamp('2027-08-01'): 250_000.0}))
BALANCES, PINNED = [N, N, N, 0.75 * N], 0.042


def deposited(timing, pinned):
    """`(day, amount)` the deposit pays: the million lent on its effective date, each quarter's
    interest when its timing says - `Discounted` dividing it by `1 + r tau` - and each quarter's end
    repaying the step its successor's balance is smaller by, the last quarter's the balance left."""
    flows = [(DATES[0], -N)]
    for k, (s, e) in enumerate(zip(DATES[:4], DATES[1:5])):
        rate = PINNED * tau(s, e) if pinned else growth(s, e)
        flows += [(e if timing == 'End' else s, BALANCES[k] * (rate / (1.0 + rate) if timing == 'Discounted' else rate)),
                  (e, BALANCES[k] - (BALANCES[k + 1] if k < 3 else 0.0))]
    return flows


def test_a_deposit_repays_its_principal_as_its_periods_end_whenever_it_pays_interest():
    """The deposit pinned and forecast under each `Payment_Timing`: its interest paid when the timing
    says, its principal repaid as its quarters end whatever the timing, so by hand
    `sum amount D(day)` over `deposited(...)`, each to 1e-12 - the forecast one under `Begin` and
    `Discounted` included, whose amount lent shares its effective date with its first coupon.

    Killing mutation: the principal repaid on its coupon's row, the pinned deposit under `Begin`
    +8,814.13 against the hand.
    """
    schedules = {'PINNED': {'Interest_Rate_Schedule': utils.DateList({s: 100 * PINNED for s in DATES[:4]})},
                 'FORECAST': {'Interest_Rate': 'USD-PROJ', 'Interest_Rate_Schedule': None}}
    values = marks([dict(DEPOSIT, Reference=kind + timing, Payment_Timing=timing, **terms)
                    for kind, terms in schedules.items() for timing in ('End', 'Begin', 'Discounted')])
    for kind in schedules:
        for timing in ('End', 'Begin', 'Discounted'):
            hand = sum(amount * D(day) for day, amount in deposited(timing, kind == 'PINNED'))
            assert abs(values[kind + timing] / hand - 1.0) < 1e-12, (kind, timing, values[kind + timing], hand)


def test_a_principal_stated_on_its_own_prices_as_the_coupon_carrying_it():
    """A floating list repaying its million as an item of its own on the last coupon's date - a
    bullet sharing its day with a coupon - marks to the bit what the list carrying it on that coupon
    does, the bullet sorted before the coupon or after it; and a deposit forecast under `Begin`
    books, under the credit Monte Carlo, its first quarter's interest less the amount lent on its
    effective date, each later quarter's interest with the step its predecessor repays, and its last
    balance alone on its maturity, the rows after its last coupon holding that bullet and nothing
    else.

    Killing mutations: a bullet held out of its day's count row by row, both lists sharing a day
    skipped; a day's fixed amount read off its last row, the list sorting its bullet first
    -852,050.41 against the one carrying it.
    """
    items = book.float_items()
    last = items[-1]

    def principal(day):
        return dict(last, Notional=0.0, Resets=[], Fixed_Amount=N, Accrual_Year_Fraction=0.0,
                    Accrual_Start_Date=day, Accrual_End_Date=day)

    lists = {'RIDING': items[:-1] + [dict(last, Fixed_Amount=N)],
             'AFTER': items + [principal(last['Payment_Date'])],
             'BEFORE': items + [principal(last['Accrual_Start_Date'])]}
    hexes, stats = book.marks([trial_rates.float_list(name, 'Buy', rows) for name, rows in lists.items()])
    assert 'Deals Skipped' not in stats and hexes['RIDING'] == hexes['AFTER'] == hexes['BEFORE'], hexes

    _, cash = ledger(dict(DEPOSIT, Payment_Timing='Begin', Interest_Rate='USD-PROJ', Interest_Rate_Schedule=None))
    hand = {}
    for day, amount in deposited('Begin', False):
        hand[day] = hand.get(day, 0.0) + amount
    assert list(cash) == list(hand), list(cash)
    for day, amount in hand.items():
        assert abs(cash[day] - amount) < 1e-12 * N, (day, cash[day], amount)


def test_a_leg_with_nothing_left_to_pay_is_worth_nothing():
    """The swap of a million paying 4% against the index for two years to 16 October 2026, valued
    on 3 August 2026 inside its last quarter with a leg paying at its periods' starts paid up: the
    floating leg under `Begin` or `Discounted`, or the fixed leg under `Begin` beside a floating leg
    whose last reset fixed at 5%. Each marks its live leg alone, `-N r tau D(e)` or `N 5% tau D(e)`
    over that quarter, to 1e-12; both legs paid up mark zero; none is skipped. Under the credit
    Monte Carlo each is its live leg on the base date to 1e-12, books that leg's coupon on the
    maturity and nothing else, and is worth nothing after it.

    Killing mutations: the empty schedule refused - the paid leg's periods read past its last date,
    no empty cashflow schedule built, or a leg owing nothing priced - all four swaps skipped.
    """
    end = pd.Timestamp('2026-10-16')
    start = end - QUARTER
    swap = dict(SWAP, Effective_Date=end - pd.DateOffset(years=2), Maturity_Date=end,
                Known_Rates=utils.DateList({start: 5.0}))
    live = {'FLOATING_BEGIN': ({'Receive_Timing': 'Begin'}, -N * RATE * tau(start, end)),
            'FLOATING_DISCOUNTED': ({'Receive_Timing': 'Discounted'}, -N * RATE * tau(start, end)),
            'FIXED_BEGIN': ({'Pay_Timing': 'Begin'}, N * 0.05 * tau(start, end))}
    values = marks([dict(swap, Reference=name, **timings) for name, (timings, _) in live.items()] +
                   [dict(swap, Reference='BOTH_BEGIN', Pay_Timing='Begin', Receive_Timing='Begin')])
    assert values['BOTH_BEGIN'] == 0.0, values
    for name, (timings, coupon) in live.items():
        assert abs(values[name] / (coupon * D(end)) - 1.0) < 1e-12, (name, values[name], coupon * D(end))
        profile, cash = ledger(dict(swap, Reference=name, **timings))
        assert abs(profile.iloc[0] / (coupon * D(end)) - 1.0) < 1e-12, (name, profile.iloc[0])
        assert list(cash) == [end] and abs(cash[end] - coupon) < 1e-12 * N, (name, cash)
        assert (profile[profile.index > end] == 0.0).all() and (profile[profile.index <= end] != 0.0).all()


def published(day):
    return day.weekday() < 5 and day not in HOLIDAYS


def daily(compounded, calendar):
    """The swap over a year on a daily index, its resets AVERAGED or `compounded`, each day read off
    the rate `calendar` publishes on or before it, every day where None: by hand, the days reading
    one rate are one reset over their joined windows, weighted by their count."""
    dates = DATES[:5]
    value = fixed_leg(dates)
    for s, e in zip(dates[:-1], dates[1:]):
        days = list(pd.date_range(s, e, freq='D', inclusive='left'))
        runs = {}
        for day in days:
            fixing = day
            while calendar and not published(fixing):
                fixing -= pd.Timedelta(days=1)
            runs.setdefault(fixing, []).append(day)
        windows = [(run[0], run[-1] + pd.Timedelta(days=1), len(run)) for run in runs.values()]
        coupon = np.prod([1.0 + growth(a, b) for a, b, _ in windows]) - 1.0 if compounded else sum(
            growth(a, b) / tau(a, b) * count / len(days) for a, b, count in windows) * tau(s, e)
        value += N * coupon * D(e)
    return value


def test_a_daily_index_fixes_on_the_days_its_calendar_publishes():
    """`Index_Publication_Calendars: Johannesburg` on a year's swap against a daily index: a day the
    calendar does not publish on reads the rate published before it, so a weekend reads Friday's
    rate over Friday to Monday and 16 December 2026, a holiday starting the first period, the 15th's
    rate over its own day; the averaged leg is `daily(False, True)` to 1e-12, the first quarter's
    average 2.75104% against 2.75094% on every calendar day, and the leg resetting every day reads
    0.015bp of notional under it. Compounded, the days tile the period whatever the calendar, so
    both marks are `daily(True, ...)` and agree to 1e-12. Blank is every calendar day, as before.

    Killing mutation: the calendar unread, the averaged leg -1.50 against the hand.
    """
    leg = dict(SWAP, Maturity_Date=DATES[4], Index_Tenor=pd.DateOffset(days=1))
    values = marks([dict(leg, Reference=method + calendar, Compounding_Method=method,
                         Index_Publication_Calendars=calendar or None)
                    for method in ('None', 'OIS') for calendar in ('', book.CALENDAR)])
    for method, compounded in (('None', False), ('OIS', True)):
        for calendar in ('', book.CALENDAR):
            hand = daily(compounded, calendar)
            assert abs(values[method + calendar] / hand - 1.0) < 1e-12, (method, calendar, hand)
    assert abs(values['OIS'] / values['OIS' + book.CALENDAR] - 1.0) < 1e-12
    assert round(bp(values['None'] - values['None' + book.CALENDAR]), 3) == -0.015, values
