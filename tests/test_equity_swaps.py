"""An equity swap leg is the swaplet list its schedule spells: one swaplet per period of its
coupon schedule, priced through the list's own seam (`Deal.equity_swaplets`), so each leg below is
held to the bit against the list a hand authors from its conventions."""
import numpy as np
import pandas as pd

import test_declared_defaults as book
import test_interest_rate_prices as rates
from derivus import utils

B, QUARTER = book.WORLD_BASE, pd.DateOffset(months=3)
#: the rate legs' sloped curves, the equity's repo read off USD-PROJ 25bp over its USD discount, a
#: dividend yield rising from 1% and a surface skewed in strike and sloped in time
WORLD = dict(rates.SLOPED, **{
    'EquityPrice.EQ': dict(book.world()['EquityPrice.EQ'], Interest_Rate='USD-PROJ'),
    'DividendRate.EQ': {'Floor': None, 'Currency': 'USD',
                        'Curve': utils.Curve([], [(t, 0.01 + 0.002 * t) for t in (0, 1, 2, 5)])},
    'EquityPriceVol.EQ': {'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness', 'Surface': utils.Curve(
        [], [[m, t, 0.25 - 0.1 * (m - 1.0) - 0.01 * t] for m in (0.6, 1.0, 1.4) for t in (0.25, 2.0, 5.0)])}})


def leg(reference, amount_type, effective, maturity, **conventions):
    """A leg of a million, 10,000 shares at the spot of 100 or a principal, as `amount_type` says."""
    return dict({'Object': 'EquitySwapLeg', 'Reference': reference, 'Currency': 'USD', 'Discount_Rate': 'USD',
                 'Equity': 'EQ', 'Equity_Currency': 'USD', 'Buy_Sell': 'Buy', 'Effective_Date': effective,
                 'Maturity_Date': maturity, 'Principal_Fixed_Variable': amount_type, 'Units': 1e4,
                 'Principal': 1e6}, **conventions)


def listed(reference, amount_type, fixings, paid, known=None, dividends=None):
    """The swaplet list a hand authors: a total-return swaplet per period of `fixings` paying on
    `paid`, `known` the `{fixing: price}` a fixing by the base date reads at an fx of one and
    `dividends` the `{start: dividends stated between it and the base date}`."""
    known, dividends = known or {}, dividends or {}
    return {'Object': 'EquitySwapletListDeal', 'Reference': reference, 'Currency': 'USD', 'Discount_Rate': 'USD',
            'Equity': 'EQ', 'Equity_Currency': 'USD', 'Buy_Sell': 'Buy',
            'Amount_Type': 'Shares' if amount_type == 'Variable' else 'Principal', 'Cashflows': {'Items': [
                {'Start_Date': s, 'End_Date': e, 'Payment_Date': pay, 'Amount': 1e4 if amount_type == 'Variable'
                 else 1e6, 'Start_Multiplier': 1.0, 'End_Multiplier': 1.0, 'Dividend_Multiplier': 1.0,
                 'Known_Start_Price': known.get(s, 0.0), 'Known_End_Price': known.get(e, 0.0),
                 'Known_Start_FX_Rate': 1.0 if s in known else 0.0, 'Known_End_FX_Rate': 1.0 if e in known else 0.0,
                 'Known_Dividend_Sum': dividends.get(s, 0.0)}
                for s, e, pay in zip(fixings[:-1], fixings[1:], paid)]}}


def test_a_leg_is_the_swaplet_list_its_periods_spell():
    """A two-year quarterly leg of a million from three months before the base date, paying two
    Johannesburg business days after each period ends, on shares and on principal: its first
    period started at the 97 stated for it, ends on the base date at the spot and carries the 1.5
    dividend paid inside it; the second starts at the spot. The base valuation mark and the
    credit Monte Carlo exposure are the eight swaplets the hand writes, to the bit.

    KILLING MUTATIONS, the leg's mark less the hand's on shares and on principal, then the largest
    gap in exposure: the periods ignored, one swaplet from the effective date to the maturity as
    the leg priced before, -2,287.11 and +288.25 (22.9 and 2.9 bp of the million), 1.60m and
    1.65m; the dividend sum taken from the effective date for every period rather than from its
    own start, +15,007.68 on both, 30,199.13 and 56,224.09; the fixing on the base date read as
    the latest price stated before it, -147.86 and +18.81, 29,908.92 and 41,324.41.
    """
    effective = B - QUARTER
    fixings = [effective + QUARTER * k for k in range(9)]
    paid = [pd.Timestamp(day) for day in ('2026-08-05', '2026-11-05', '2027-02-05', '2027-05-05',
                                          '2027-08-05', '2027-11-05', '2028-02-07', '2028-05-05')]
    terms = dict(Payment_Calendars=book.CALENDAR, Payment_Offset=2,
                 Equity_Known_Prices=utils.DateEqualList([[effective, 97.0, 1.0]]),
                 Known_Dividends=utils.DateEqualList([[effective + pd.DateOffset(months=1), 1.5]]))
    pairs = [(leg(amount_type, amount_type, effective, fixings[-1], **terms),
              listed('HAND_' + amount_type, amount_type, fixings, paid, {effective: 97.0, B: 100.0},
                     {effective: 1.5})) for amount_type in ('Variable', 'Principal')]
    marks, stats = book.marks([deal for pair in pairs for deal in pair], factors=WORLD)
    assert 'Deals Skipped' not in stats, stats
    moved = {}
    for deal, hand in pairs:
        exposure, listing = (book.simulated(
            [document], ('USD', 'USD-PROJ'), WORLD,
            models={'EquityPrice.EQ': ('GBMAssetPriceModel', {'Vol': 0.2, 'Drift': 0.0})})[1]['Results']['mtm']
            for document in (deal, hand))
        if marks[deal['Reference']] != marks[hand['Reference']] or not np.array_equal(exposure.values, listing.values):
            moved[deal['Reference']] = (
                float.fromhex(marks[deal['Reference']]) - float.fromhex(marks[hand['Reference']]),
                (exposure - listing).abs().max().max())
    assert not moved, moved


def test_each_convention_moves_the_leg_to_the_list_it_spells():
    """Two-year legs on 10,000 shares starting after the base date, each convention stated against
    the list it means, to the bit: a five-month frequency rolled `Forward`, the leg's default,
    leaves a four-month last period and `Backward` a four-month first; a `First_Coupon_Date` two
    months in rolls the quarters forward from it to a one-month last period; `Following` on the
    Johannesburg accrual calendar rolls the five weekend month-ends of a schedule from 31 October
    into the next month where `Modified_Following` rolls them back; and a `Reset_Offset` of two on
    the Johannesburg reset calendar fixes each period two business days before its accrual dates,
    the four after a holiday a day earlier than weekdays would, paying on its accrual end.

    KILLING MUTATIONS, the leg's mark less the hand's: `Roll_Direction` unread as the seam's
    backward roll, -19.86 forward, or as forward, +19.86 backward; the first coupon date unread,
    -14.77; the adjustment method unread, the seam's Modified Following, +194.56; the reset offset
    unread, +332.14; its calendar unread, weekdays, +3.96.
    """
    start, five = pd.Timestamp('2026-09-01'), pd.DateOffset(months=5)
    end = start + pd.DateOffset(years=2)
    first = start + pd.DateOffset(months=2)
    month_end = pd.Timestamp('2026-10-31')
    following = [pd.Timestamp(day) for day in (
        '2026-11-02', '2027-02-01', '2027-04-30', '2027-08-02', '2027-11-01', '2028-01-31', '2028-05-02',
        '2028-07-31', '2028-10-31')]
    lagged = pd.Timestamp('2026-09-17')
    accrued = [lagged + QUARTER * k for k in range(9)]
    fixed = [pd.Timestamp(day) for day in (
        '2026-09-15', '2026-12-14', '2027-03-15', '2027-06-14', '2027-09-15', '2027-12-14', '2028-03-15',
        '2028-06-14', '2028-09-14')]
    forward = [start + five * k for k in range(5)] + [end]
    backward = [start] + [end - five * k for k in range(4, -1, -1)]
    stub = [start] + [first + QUARTER * k for k in range(8)] + [end]
    cases = {
        'FORWARD': ((start, end, {'Payment_Frequency': five}), forward, forward[1:]),
        'BACKWARD': ((start, end, {'Payment_Frequency': five, 'Roll_Direction': 'Backward'}), backward,
                     backward[1:]),
        'FIRST': ((start, end, {'First_Coupon_Date': first}), stub, stub[1:]),
        'FOLLOWING': ((month_end, month_end + pd.DateOffset(years=2),
                       {'Accrual_Calendars': book.CALENDAR, 'Rate_Adjustment_Method': 'Following'}),
                      following, following[1:]),
        'RESET': ((lagged, lagged + pd.DateOffset(years=2),
                   {'Reset_Offset': 2, 'Reset_Calendars': book.CALENDAR}), fixed, accrued[1:])}
    documents = []
    for reference, ((effective, maturity, conventions), fixings, paid) in cases.items():
        documents += [leg(reference, 'Variable', effective, maturity, **conventions),
                      listed('HAND_' + reference, 'Variable', fixings, paid)]
    marks, stats = book.marks(documents, factors=WORLD)
    assert 'Deals Skipped' not in stats, stats
    moved = {reference: float.fromhex(marks[reference]) - float.fromhex(marks['HAND_' + reference])
             for reference in cases if marks[reference] != marks['HAND_' + reference]}
    assert not moved, moved
