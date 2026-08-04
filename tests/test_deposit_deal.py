"""`DepositDeal` was declared in the schema for years with no class behind it, so
`construct_instrument` logged an error and returned `{}` - the deal vanished from the portfolio
rather than raising.

It is wanted for curve bootstrapping, and that use imposes a requirement an ordinary valuation test
would never check: a deposit whose rate is fully pinned by **Interest_Rate_Schedule** must not
depend on the forecast curve at all, because a bootstrapper is solving for that very curve. So the
gates below gate the DEPENDENCY as well as the number, and each mutates the pinning to prove the
assertion can fail - a schedule that pins everything makes the forecast path unreachable, which is
exactly the shape that turns a gate into a placebo.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
import torch

import derivus
from derivus import utils
from derivus.config import Config
from derivus.instruments import construct_instrument

BASE = pd.Timestamp('2024-06-28')
DTYPE = torch.float64
RATE = 0.02          # flat continuously-compounded curve
COUPON = 3.0         # percent, as the schedule quotes it
AMOUNT = 1_000_000.0
STARTS = [BASE + pd.DateOffset(months=m) for m in (0, 3, 6, 9)]


def _deal(schedule_dates):
    """`schedule_dates` pins those accrual starts at COUPON; anything unpinned forecasts."""
    return {
        'Object': 'DepositDeal', 'Reference': 'DEPO1', 'Currency': 'USD',
        'Discount_Rate': 'USD', 'Interest_Rate': 'USD',
        'Effective_Date': BASE, 'Maturity_Date': BASE + pd.DateOffset(months=12),
        'Payment_Frequency': pd.DateOffset(months=3),
        'Interest_Frequency': pd.DateOffset(months=3),
        'Accrual_Day_Count': 'ACT_365', 'Amount': AMOUNT, 'Amortisation': None,
        'Compounding': 'No', 'Payment_Timing': 'End', 'Payment_Offset': 0,
        'Accrual_Calendars': None, 'Payment_Calendars': None,
        'First_Coupon_Date': None, 'Penultimate_Coupon_Date': None,
        'Rate_Currency': '', 'FX_Reset_Offset': 0, 'Known_FX_Rates': None,
        'Interest_Rate_Schedule': utils.DateList({d: COUPON for d in schedule_dates}),
    }


def _cfg(schedule_dates):
    c = Config()
    c.params['System Parameters']['Base_Currency'] = 'USD'
    c.params['System Parameters']['Base_Date'] = BASE
    c.params['Price Factors'] = {
        'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Priority': 1, 'Spot': 1.0},
        'InterestRate.USD': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                             'Curve': utils.Curve([], [[0.0, RATE], [5.0, RATE]])},
        'DiscountRate.USD': {'Interest_Rate': 'USD'},
    }
    c.params['Price Models'] = {}
    c.params['Valuation Configuration'] = {}
    c.deals = {'Attributes': {'Reference': 'test', 'Tag_Titles': ''},
               'Deals': {'Children': [{'Instrument': construct_instrument(_deal(schedule_dates), {})}]},
               'Calculation': {'Base_Date': BASE, 'Currency': 'USD'}}
    return c


def _price(schedule_dates):
    """Base valuation, not CMC: with everything pinned there are no stochastic factors at all, and
    the degenerate single-date lifecycle is the reference this reconciles against."""
    _, out = derivus.run_baseval(_cfg(schedule_dates), prec=DTYPE,
                                  overrides={'MCMC_Simulations': 1, 'Random_Seed': 1})
    rows = out['Results']['mtm']
    return float(rows[rows['Reference'] == 'DEPO1']['Value'].iloc[0])


MATURITY = BASE + pd.DateOffset(months=12)


def _coupon_pvs():
    """Per-period discounted coupon on the flat curve. ACT_365 over the actual day counts, so the
    quarters are unequal and a 0.25 shortcut would not reconcile."""
    ends = STARTS[1:] + [MATURITY]
    return [AMOUNT * ((end - start).days / 365.0) * (COUPON / 100.0)
            * np.exp(-RATE * ((end - BASE).days / 365.0))
            for start, end in zip(STARTS, ends)]


def _hand_value():
    """Coupons plus redemption LESS the principal placed at the effective date. Both legs count:
    that is what makes a depo a bootstrapping instrument, since PV goes to zero exactly when the
    quoted rate is consistent with the curve. Omitting the outflow overstates this by ~1,000,000."""
    t_final = (MATURITY - BASE).days / 365.0
    return sum(_coupon_pvs()) + AMOUNT * np.exp(-RATE * t_final) - AMOUNT


def test_a_fully_pinned_deposit_matches_a_hand_discounted_cashflow():
    """The value gate. Every accrual start is pinned, so this is a pure fixed leg and the answer is
    arithmetic - no forecasting, nothing model-dependent to hide an error in."""
    priced = _price(STARTS)
    expected = _hand_value()
    assert priced == pytest.approx(expected, rel=2e-3), (
        f'priced {priced:,.2f} against a hand-discounted {expected:,.2f}')


def test_a_pinned_deposit_takes_no_dependency_on_the_forecast_curve():
    """The bootstrapping requirement. Discovery reads `factor_fields` off the instance after
    `reset`, so a fully-pinned depo must not list the forecast curve - otherwise a bootstrapper
    solving that curve would find the instrument depending on its own answer."""
    pinned = construct_instrument(_deal(STARTS), {})
    pinned.reset(None)
    assert 'Interest_Rate' not in pinned.factor_fields, (
        f'a fully-pinned deposit still pulls the forecast curve: {pinned.factor_fields}')

    # MUTATE: unpin one accrual start and the dependency must come back, or the assertion above
    # is passing for a reason unrelated to pinning
    partial = construct_instrument(_deal(STARTS[:-1]), {})
    partial.reset(None)
    assert 'Interest_Rate' in partial.factor_fields, (
        'an unpinned period must forecast, so the curve dependency has to return')


def test_a_forecasting_deposit_prices_to_par():
    """A deposit that forecasts every period is a par floater - principal out, forecast coupons,
    principal back, all off one curve - so at inception it is worth exactly zero. That is a real
    invariant rather than a fitted constant, and it is the cleanest evidence the float branch
    forecasts and discounts consistently.

    It also pins the all-or-nothing coverage rule. The schedule here is forward-dated and covers
    three of the four periods, but `generate_float_cashflows` pins only resets before the reference
    date, so nothing is pinned and the deal forecasts throughout. A per-period reading would have
    predicted three 3% coupons and one forecast; the correct answer is par."""
    assert _price(STARTS[:-1]) == pytest.approx(0.0, abs=1e-6), (
        'a fully-forecasting deposit must be worth par at inception')

    # MUTATE: complete the schedule and the fixed branch takes over at an above-market 3%, so the
    # value has to leave par - without this the assertion above would pass on a dead pricer
    assert _price(STARTS) > 1_000.0, (
        'completing the schedule must switch to the fixed branch and move the value off par')
