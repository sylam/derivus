"""`FRADeal.Payment_Timing`: three timings, two dates each, and the ledger that tells them apart.

An FRA states two dates the field name does not distinguish. Its PV runs from one of them and its
CASH settles on the other, and the three declared timings are the three combinations a desk trades:

| `Payment_Timing` | PV from | reval ends | cash books | amount booked |
| --- | --- | --- | --- | --- |
| `End` (the default) | Maturity | Maturity | Maturity | the full realized amount |
| `Discounted` | Maturity | Effective | Effective | that amount discounted over the period |
| `Begin` | Effective | Effective | Effective | the full amount, undiscounted |

`End` and `Discounted` carry the SAME PV - both discount the full amount from maturity - and differ
only in where the deal dies and what the ledger receives. So a base valuation cannot separate them
at all, and only the CREDIT MONTE CARLO gates below can: `cash_settle` is a no-op under base
valuation (`shared.t_Cashflows is None`), so the ledger does not exist there, and the reval end is
only visible as the rows a profile occupies.

THE CURVE SUITE IS BLIND TO ALL OF THIS, and the reason is what makes these fixtures the shape they
are. `tests/test_interest_rate_prices.py` prices its FRAs at PAR - the bootstrap generates the
quote that makes each benchmark worth exactly zero - and a zero flow is zero whatever date it is
discounted from. Every fixture here is OFF PAR on a genuinely SLOPED, genuinely MULTI-curve world,
with a fixing lag (`Reset_Date` two days before `Effective_Date`) so the reset row spans the true
FRA period rather than starting at its own fixing.

THE REFERENCE IS HAND-DERIVED, not the engine asked twice:

    r(T)   = a + b * T                            the authored zero curve, T in ACT/365 years
    D(T)   = exp(-r(T) * T)
    F      = (D_proj(T_s) / D_proj(T_e) - 1) / tau       tau = ACT/360 over the FRA period
    payoff = position * N * (F - K) * tau                position = +1 Borrower, -1 Lender
    PV     = payoff * D_disc(T_pay)                      T_pay = T_e, except Begin's T_s

The zero curves are LINEAR IN TENOR on purpose, and it buys the Monte Carlo gates their oracle. A
Hull-White curve at time t is the arbitrage-free forward curve `[r(t+s)(t+s) - r(t)t] / s`, which
for a linear `r` is `a + b(2t+s)` - linear in `s`, so the engine's own linear interpolation
reproduces it exactly at every tenor, and the forward over `[T_s, T_e]` seen at ANY row comes back
as `r(T_e)T_e - r(T_s)T_s`, the number above. At zero vol that makes the whole simulated profile
and every booked amount hand-derivable to the last bit. A curved zero curve does not: reconstructed
on a knot grid and interpolated back, its forward is a different number at every row, and on a
plausible six-knot curve that gap measures 2.7% - which would stand between the engine and the
reference and force a tolerance loose enough to admit the defect these gates exist to catch.

KILL READINGS, taken on these fixtures against the pre-fix build (`discount_date = Maturity_Date if
timing == 'Discounted' else Effective_Date`, which handed `End` the effective date):

  * `End` base valuation discounted from `T_s` instead of `T_e`: **21777.712514** against the
    hand-derived **21500.597058**, 1.2889% high, on every one of the four moneyness/direction
    combinations - against a gate that holds to 1e-12. `Discounted` and `Begin` are untouched, so
    the default timing was the one that was wrong.
  * `End` under credit Monte Carlo did not misprice - it did not PRICE. The payment day was the
    effective date while `reset`'s `pay_date` kept the deal alive to maturity, so
    `get_cashflow_start_index` returned index 1 on every row past effective and
    `pv_float_cashflow_list` indexed a one-cashflow schedule out of bounds. The deal is caught and
    SKIPPED (`CRITICAL FRA1 Deal FRA1 skipped - index 1 is out of bounds for axis 0 with size 1`),
    the profile comes back as a scalar zero, and the run reports success with the trade missing
    from it - which is precisely the failure mode a price gate cannot see, and why the exposure
    and ledger gates are owed rather than optional.

Seven of the eleven gates below are red against that build; the four that stay green are the
`Discounted` and `Begin` readings, which the pre-fix branch already had right.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import pytest
import torch

import derivus
from derivus import run_baseval, utils
from derivus.config import Config
from derivus.instruments import construct_instrument

import rates_world as rw

DTYPE = torch.float64
BASE = rw.BASE

#: Six and nine months out, so the trade is a 6x9 - forward starting, and both of its dates land on
#: rows of the monthly reporting grid rather than between them.
START_MONTHS, END_MONTHS = 6, 9
EFFECTIVE = BASE + pd.DateOffset(months=START_MONTHS)
MATURITY = BASE + pd.DateOffset(months=END_MONTHS)
#: The fixing lag. `Reset_Date != Effective_Date` is what exercises the reset row's rate window: the
#: reset FIXES here and the rate it fixes spans [Effective, Maturity], which is the pairing the row
#: has to carry.
FIXING_LAG = 2

#: The FRA's dates in the curves' own ACT/365 years.
T_S = (EFFECTIVE - BASE).days / 365.0
T_E = (MATURITY - BASE).days / 365.0
#: The FRA's own accrual, ACT/360 - a different convention from the curves' ACT/365 on purpose, so a
#: year fraction read off the wrong axis could not cancel.
TAU = (MATURITY - EFFECTIVE).days / 360.0

#: `(intercept, slope)` of each zero curve. Both slope steeply upward and the projection curve
#: carries a basis over the discount curve, so this is a real multi-curve world: a flat curve would
#: leave `D(T_s)` and `D(T_e)` differing only by the period, and a single curve would let a
#: discounting error hide inside the forward.
DISCOUNT = (0.0400, 0.0100)
PROJECTION = (0.0450, 0.0115)
#: The knot grid. The front knot is a day rather than zero: `HullWhite1FactorInterestRateModel`
#: divides its assembled curve by the factor's own tenors, so a zero knot is a division by zero and
#: every simulated row comes back NaN.
KNOTS = [1.0 / 365.0, 0.25, T_S, T_E, 2.0, 5.0]

CURRENCY, DISCOUNT_CURVE, PROJECTION_CURVE = 'ZAR', 'ZAR-DISC', 'ZAR-PROJ'
PRINCIPAL = 1e7

#: Off par on both sides. The projected forward is 5.901126% on this world, so a Borrower is 90bp
#: in the money at 5.00 and 110bp out of it at 7.00: realized amounts of +22277.85 and -27166.60 on
#: a notional of ten million, signed and large. A par-held FRA carries zero instead, and a zero
#: flow is zero whatever date it is discounted from - which is the degeneracy this file exists to
#: stay out of.
LOW_STRIKE, HIGH_STRIKE = 5.00, 7.00

#: (side, strike) per timing for the Monte Carlo gates. Across the three, both directions and both
#: signs of moneyness appear; each base valuation gate carries all four combinations at once.
CMC_CASES = {'End': ('Borrower', LOW_STRIKE),
             'Discounted': ('Lender', LOW_STRIKE),
             'Begin': ('Borrower', HIGH_STRIKE)}

#: A three-year par swap, alongside the FRA where the gate needs a grid that outlives it. It shares
#: both curves and settles in the same currency, so it is an ordinary book-mate rather than a prop.
COMPANION = 'SWAP3Y'


def zero_rate(curve, tenor):
    """`a + b * T`, the authored zero curve. The `Price Factors` block below is written FROM this,
    so the curve the engine reads and the curve the reference derives from are one statement."""
    return curve[0] + curve[1] * tenor


def discount_factor(curve, tenor):
    return float(np.exp(-zero_rate(curve, tenor) * tenor))


#: The projected simple forward over [Effective, Maturity], off the projection curve.
FORWARD = (discount_factor(PROJECTION, T_S) / discount_factor(PROJECTION, T_E) - 1.0) / TAU
#: What discounting the realized amount over the FRA period costs - the `Discounted` timing's whole
#: difference from `Begin`, and (inverted) `Begin`'s from `End`.
PERIOD_DISCOUNT = discount_factor(DISCOUNT, T_E) / discount_factor(DISCOUNT, T_S)


def hand_derived_payoff(side, strike):
    """The realized amount the FRA pays at maturity: `position * N * (F - K) * tau`."""
    position = 1.0 if side == 'Borrower' else -1.0
    return position * PRINCIPAL * (FORWARD - strike / 100.0) * TAU


def hand_derived_pv(timing, side, strike):
    """The PV at the base date, derived here rather than read back off the engine."""
    return hand_derived_payoff(side, strike) * discount_factor(
        DISCOUNT, T_S if timing == 'Begin' else T_E)


def hand_derived_settlement(timing, side, strike):
    """What the ledger receives, and on what date."""
    amount = hand_derived_payoff(side, strike)
    if timing == 'Discounted':
        return EFFECTIVE, amount * PERIOD_DISCOUNT
    return (MATURITY, amount) if timing == 'End' else (EFFECTIVE, amount)


def last_live_date(timing):
    """Where the deal stops being a position - `reset`'s `pay_date`."""
    return MATURITY if timing == 'End' else EFFECTIVE


def fra(reference, timing, side, strike, fixing_lag=FIXING_LAG):
    """One off-par FRA on the authored world, through `rates_world`'s own builder."""
    return rw.fra(reference, CURRENCY, PROJECTION_CURVE, DISCOUNT_CURVE,
                  START_MONTHS, END_MONTHS, strike, timing=timing, side=side,
                  principal=PRINCIPAL, fixing_lag=fixing_lag)


def config(deals, sigma=None):
    """A `Config` holding the authored world and `deals`, each directly under the root.

    `sigma` gives both curves a Hull-White factor. Zero is not "no model": the simulation runs, the
    resets fix off it and the ledger is booked from it - every path is just the arbitrage-free
    evolution of the authored curve, which is what makes the readings exact. A credit Monte Carlo
    needs at least one stochastic factor to run at all (`reset` reshapes a zero-factor draw and
    raises), so a static world is not the alternative.
    """
    c = Config(base_currency=CURRENCY)
    c.params['System Parameters']['Base_Date'] = BASE
    c.params['Price Factors'] = rw.market(
        CURRENCY,
        {DISCOUNT_CURVE: (KNOTS, [zero_rate(DISCOUNT, t) for t in KNOTS]),
         PROJECTION_CURVE: (KNOTS, [zero_rate(PROJECTION, t) for t in KNOTS])},
        DISCOUNT_CURVE, day_count='ACT_365')
    if sigma is not None:
        for curve in (DISCOUNT_CURVE, PROJECTION_CURVE):
            c.params['Price Models']['HullWhite1FactorInterestRateModel.' + curve] = {
                'Alpha': 0.05, 'Lambda': 0.0, 'Quanto_FX_Correlation': 0.0,
                'Quanto_FX_Volatility': utils.Curve([], [[0.0, 0.0], [5.0, 0.0]]),
                'Sigma': utils.Curve([], [[0.0, sigma], [5.0, sigma]])}
        c.params['Model Configuration'].append(
            'InterestRate', (), 'HullWhite1FactorInterestRateModel')
    c.deals = {'Attributes': {'Reference': 'test', 'Tag_Titles': ''},
               'Deals': {'Children': [{'Instrument': construct_instrument(d, {})} for d in deals]},
               'Calculation': {'Base_Date': BASE, 'Currency': CURRENCY}}
    return c


def priced(deals):
    """`{Reference: PV}` off one base valuation."""
    _, out = run_baseval(config(deals), prec=DTYPE)
    rows = out['Results']['mtm']
    return rows[rows['Parent'] == 'root'].set_index('Reference')['Value'].astype(float)


def cmc(deals, sigma=0.0, batch=64):
    """One credit Monte Carlo over `deals`, ledger on, monthly reporting rows.

    `0d 1m(1m)` off a base date on the 3rd puts the effective and maturity dates ON rows rather
    than between them, and fills the six months in front of them so the profile is a profile. How
    far the grid runs is the BOOK's horizon rather than this string's: with the FRA alone it stops
    on the deal's own settlement date, which is the reval end being read.
    """
    return derivus.run_cmc(config(deals, sigma=sigma), prec=DTYPE, overrides={
        'Run_Date': BASE.strftime('%Y-%m-%d'), 'Time_grid': '0d 1m(1m)', 'Batch_Size': batch,
        'Simulation_Batches': 1, 'Random_Seed': 1, 'Currency': CURRENCY, 'Tenor_Offset': 0.0,
        'MCMC_Simulations': 1, 'Deflation_Interest_Rate': DISCOUNT_CURVE,
        'Generate_Cashflows': 'Yes'})


def deal_profile(calc, reference):
    """One deal's OWN (rows, paths) profile, off the structure the run left behind.

    `pricing.interpolate` stashes it on the deal's `Calc_res` before the tail pad, so its row count
    IS the number of MTM rows the deal is alive on - which is the reval end the timing decides,
    read without netting it against anything else in the book.
    """
    for dependency in calc.netting_sets.dependencies:
        if dependency.Instrument.field.get('Reference') == reference:
            return np.concatenate(dependency.Calc_res['Value'], axis=-1)
    raise AssertionError('{} is not a deal of this run'.format(reference))


def settled_rows(out):
    """`{date: (paths,)}` for every ledger row carrying anything at all."""
    ledger = out['Results']['cashflows'][CURRENCY]
    return {date: ledger.loc[date].values for date in ledger.index
            if np.abs(ledger.loc[date].values).max() > 0.0}


# --------------------------------------------------------------- the three base valuation gates

@pytest.mark.parametrize('timing', ['End', 'Discounted', 'Begin'])
def test_the_pv_discounts_from_the_date_the_timing_names(timing):
    """One base valuation per timing, against the hand-derived discount off the authored curve.

    Four FRAs in each run - Borrower and Lender, in and out of the money - so a sign convention
    cannot pass by cancelling and a discounting error cannot pass by being small on one side.

    `End` and `Discounted` COINCIDE here, and that is the ruling rather than an accident: both book
    the realized amount at maturity and discount it from there, so the only things separating them
    are where the cash settles and how long the deal stays alive, neither of which a base valuation
    can see. `Begin` is the one that moves, by `D(T_e)/D(T_s)`.
    """
    cases = [('BORROW_LOW', 'Borrower', LOW_STRIKE), ('BORROW_HIGH', 'Borrower', HIGH_STRIKE),
             ('LEND_LOW', 'Lender', LOW_STRIKE), ('LEND_HIGH', 'Lender', HIGH_STRIKE)]
    values = priced([fra(ref, timing, side, strike) for ref, side, strike in cases])

    for ref, side, strike in cases:
        expected = hand_derived_pv(timing, side, strike)
        assert abs(values[ref] / expected - 1.0) < 1e-12, (
            '{} {} at K={}: engine {:.6f} against the hand-derived {:.6f}'.format(
                timing, side, strike, values[ref], expected))
    assert values['BORROW_LOW'] > 0.0 > values['BORROW_HIGH'], (
        'the two strikes must straddle the forward, or the gate reads one sign twice')
    assert values['LEND_LOW'] == -values['BORROW_LOW'], 'a Lender is the mirror of a Borrower'


def test_end_and_discounted_price_alike_and_begin_is_the_one_that_moves():
    """The identity the table above states, gated so it cannot drift silently.

    It is also the placebo check on the gate above: three timings agreeing to 1e-12 against a
    reference that never looked at the timing would prove nothing, so the ratio that SEPARATES
    `Begin` is asserted against the discount factors it is made of, and that ratio is required to
    be a real number rather than a rounding one.
    """
    values = {timing: float(priced([fra('F1', timing, 'Borrower', LOW_STRIKE)])['F1'])
              for timing in ('End', 'Discounted', 'Begin')}

    assert values['End'] == values['Discounted'], (
        'End and Discounted must PV identically - both discount the full amount from maturity')
    assert abs(values['End'] / values['Begin'] - PERIOD_DISCOUNT) < 1e-12, (
        'Begin values from the effective date, so it sits exactly D(T_e)/D(T_s) = {:.8f} away: '
        '{}'.format(PERIOD_DISCOUNT, values))
    assert abs(PERIOD_DISCOUNT - 1.0) > 1e-2, (
        'the period discount is {:.6f} - too close to one for this fixture to separate '
        'anything'.format(PERIOD_DISCOUNT))


def test_the_reset_reads_the_fra_period_and_not_its_own_fixing_window():
    """The fixing lag, which is what the reset row's own start and end dates are for.

    A reset fixing before the effective date reads the rate over [Effective, Maturity] - the period
    the FRA accrues - not [Reset, Maturity]. Gated by pricing the same trade at two very different
    lags and requiring the PV to be IDENTICAL: nothing else about the deal moves with the lag, so a
    reset row carrying its own fixing date as the rate START would make these two numbers differ.

    The pre-636f9a8 row was `[Reset, Reset, Maturity]`, which is exactly that. It widens the rate
    window without touching the accrual, so the projected forward is read over 134 days and divided
    by a 89-day year fraction: on this curve 8.7024% against 5.9011%, and the trade prices at
    88338.21 against 21500.60 - four times its value. The second assertion below is that
    counterfactual, derived from the same formula the reference is, so the gate states how far
    wrong the alternative is rather than only that the two lags agree.
    """
    values = priced([fra('LAG_0', 'End', 'Borrower', LOW_STRIKE, fixing_lag=0),
                     fra('LAG_45', 'End', 'Borrower', LOW_STRIKE, fixing_lag=45)])
    assert values['LAG_0'] == values['LAG_45'], (
        'the fixing lag moved the PV by {:.6g} - the reset is reading its rate over the fixing '
        'window rather than over the FRA period'.format(values['LAG_45'] - values['LAG_0']))
    assert abs(values['LAG_0'] - hand_derived_pv('End', 'Borrower', LOW_STRIKE)) < 1e-6

    lagged = T_S - 45.0 / 365.0
    wrong_forward = (np.exp(zero_rate(PROJECTION, T_E) * T_E -
                            zero_rate(PROJECTION, lagged) * lagged) - 1.0) / TAU
    wrong_pv = (PRINCIPAL * (wrong_forward - LOW_STRIKE / 100.0) * TAU *
                discount_factor(DISCOUNT, T_E))
    assert wrong_pv / values['LAG_45'] > 3.0, (
        'reading the rate from the fixing date would price this at {:.2f} against {:.2f}, which is '
        'not far enough apart for the equality above to be worth asserting'.format(
            wrong_pv, values['LAG_45']))


# --------------------------------------------------- the three credit Monte Carlo gates

@pytest.mark.parametrize('timing', ['End', 'Discounted', 'Begin'])
def test_the_exposure_ends_and_the_cash_books_where_the_timing_says(timing):
    """One credit Monte Carlo per timing: the REVAL END and the LEDGER together.

    Neither half is enough on its own. The profile says where the deal stops being a position -
    `End` runs to maturity, `Discounted` and `Begin` stop at effective - and the ledger says what
    was paid and when. `End` and `Discounted` agree at every row they share and at the base date,
    so nothing but these two readings separates them.

    Both are compared against the hand derivation to float precision, which the zero-vol Hull-White
    world is what buys (see the module docstring). The three expected amounts are three different
    numbers: `End` and `Begin` book the full realized amount, `Discounted` books it multiplied by
    the period discount, and `End` books three months later than the other two.
    """
    side, strike = CMC_CASES[timing]
    calc, out = cmc([fra('FRA1', timing, side, strike)])

    profile = out['Results']['mtm']
    dates = list(profile.index)
    own = deal_profile(calc, 'FRA1')
    assert dates[own.shape[0] - 1] == last_live_date(timing), (
        '{}: the deal is alive to {:%Y-%m-%d}, not to {:%Y-%m-%d}'.format(
            timing, dates[own.shape[0] - 1], last_live_date(timing)))
    assert dates[-1] == last_live_date(timing), (
        '{}: the reporting grid outlives the deal, so the row count above is not the reading '
        'it claims to be'.format(timing))
    assert np.abs(own).min() > 0.0, '{}: the profile is dead on a row it should be alive on'
    # every path is the same arbitrage-free evolution, so a spread here would mean the zero-vol
    # world is not the deterministic one the reference assumes
    assert np.abs(own - own[:, :1]).max() == 0.0, (
        '{}: the zero-vol world produced a spread across paths'.format(timing))

    expected_date, expected_amount = hand_derived_settlement(timing, side, strike)
    settled = settled_rows(out)
    assert list(settled) == [expected_date], (
        '{}: the ledger settled on {} rather than only {:%Y-%m-%d}'.format(
            timing, [str(d.date()) for d in settled], expected_date))
    booked = settled[expected_date]
    assert abs(float(booked.mean()) / expected_amount - 1.0) < 1e-12, (
        '{}: the ledger booked {:.6f} where the hand derivation says {:.6f}'.format(
            timing, float(booked.mean()), expected_amount))
    assert np.array_equal(booked, own[-1]), (
        '{}: the ledger is not the deal\'s own last row - `pv_fra_leg` books `local_pv[-1]`, so '
        'these are the same number by construction'.format(timing))


def test_the_deal_contributes_nothing_after_its_own_date_on_a_grid_that_outlives_it():
    """The other half of the reval end: a book-mate that outlives the FRA, so the grid does too.

    The gate above reads the deal's last live row off a grid that ENDS there, which is true and is
    half the statement - the horizon is the book's own last settlement. Put a three-year swap
    beside it and the grid runs two years past every timing. The FRA still stops on its own date -
    `End` at maturity, the other two at effective - and every reported row beyond it is the swap's
    value and nothing else, which is what "dead" means where a portfolio keeps reporting.
    """
    dead = {}
    for timing in ('End', 'Discounted', 'Begin'):
        side, strike = CMC_CASES[timing]
        calc, out = cmc([fra('FRA1', timing, side, strike),
                         rw.par_swap(COMPANION, CURRENCY, PROJECTION_CURVE, DISCOUNT_CURVE, 3,
                                     6.0, day_count='ACT_365')])
        netted = out['Results']['mtm'].values
        dates = list(out['Results']['mtm'].index)
        own = deal_profile(calc, 'FRA1')
        companion = deal_profile(calc, COMPANION)
        assert dates[-1] > MATURITY, 'the companion did not extend the grid past the FRA'
        assert own.shape[0] < len(dates), (
            '{}: the FRA occupies every row of a grid that outlives it'.format(timing))

        after = own.shape[0]
        assert np.abs(netted[after:] - companion[after:len(dates)]).max() == 0.0, (
            '{}: the book still carries the FRA after {:%Y-%m-%d}'.format(
                timing, dates[after - 1]))
        assert np.abs(companion[after:len(dates)]).min() > 0.0, (
            'the companion is worth nothing on those rows, so the comparison above is 0 == 0')
        dead[timing] = dates[after - 1]

    assert dead == {'End': MATURITY, 'Discounted': EFFECTIVE, 'Begin': EFFECTIVE}, dead


def test_the_three_timings_book_three_different_things_on_the_same_trade():
    """The separation, on ONE trade priced three ways - the reading a single timing cannot give.

    `End` and `Discounted` agree on the base-date PV and disagree on the ledger; `Begin` and
    `Discounted` agree on the ledger DATE and disagree on the amount; `End` and `Begin` agree on
    the amount and disagree on the date. Asserting all three pairs together is what says these are
    three branches rather than two spellings of one.
    """
    booked = {}
    for timing in ('End', 'Discounted', 'Begin'):
        _, out = cmc([fra('FRA1', timing, 'Borrower', LOW_STRIKE)])
        (date, amount), = settled_rows(out).items()
        booked[timing] = (date, float(np.mean(amount)))

    assert booked['End'][0] == MATURITY, booked
    assert booked['Discounted'][0] == booked['Begin'][0] == EFFECTIVE, booked
    assert booked['End'][1] == booked['Begin'][1], (
        'End and Begin book the SAME undiscounted amount on different dates: {}'.format(booked))
    assert abs(booked['Discounted'][1] / booked['Begin'][1] - PERIOD_DISCOUNT) < 1e-12, (
        'Discounted books the period-discounted amount, {:.8f} of Begin\'s: {}'.format(
            PERIOD_DISCOUNT, booked))


def test_the_ledger_survives_a_live_simulation():
    """The same three statements under a real Hull-White vol, where the profile is a distribution.

    The zero-vol gates above are exact because every path is the same; this one is what says they
    are not exact because the pricer went down a degenerate branch. The date, the reval end and the
    identity `booked == the deal's own last row` are all still EXACT path for path. The AMOUNT is
    the one thing that cannot be, and it is compared against its own STANDARD ERROR rather than a
    tolerance chosen by hand - the payoff's spread is about a fifth of its level here, so a fixed
    percentage would be either vacuous or a reseed away from failing.

    The residual is NOISE and not bias, which is what makes a standard-error bound the right shape.
    Across seeds 1/2/3 the three timings sit at 1.15/2.32/2.32, 1.59/0.55/0.55 and 0.99/0.28/0.28
    standard errors, and raising seed 1's count to 16384 moves it to 1.38/0.96/0.96 while the
    relative gap falls from 0.41%/0.83%/0.68% to 0.25%/0.17%/0.14% - sqrt(n), as it should. A
    systematic term would have held its relative size and grown in sigmas instead.
    """
    for timing in ('End', 'Discounted', 'Begin'):
        side, strike = CMC_CASES[timing]
        calc, out = cmc([fra('FRA1', timing, side, strike)], sigma=0.003, batch=4096)
        own = deal_profile(calc, 'FRA1')
        dates = list(out['Results']['mtm'].index)

        assert np.abs(own - own[:, :1]).max() > 0.0, (
            '{}: a live vol produced no spread at all - this is the zero-vol run again'.format(
                timing))
        assert dates[own.shape[0] - 1] == last_live_date(timing), timing

        expected_date, expected_amount = hand_derived_settlement(timing, side, strike)
        settled = settled_rows(out)
        assert list(settled) == [expected_date], (
            '{}: settled on {}'.format(timing, [str(d.date()) for d in settled]))
        booked = settled[expected_date]
        assert np.array_equal(booked, own[-1]), (
            '{}: the ledger is not the deal\'s own last row'.format(timing))

        error = float(booked.std(ddof=1)) / np.sqrt(booked.size)
        assert abs(float(booked.mean()) - expected_amount) < 4.0 * error, (
            '{}: booked {:.2f} against the hand-derived {:.2f}, which is more than four standard '
            'errors ({:.2f})'.format(timing, float(booked.mean()), expected_amount, error))
