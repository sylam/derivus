"""Where does the spread sit when a coupon's sub-periods compound?

`pv_float_cashflow_list` folds several cashflow ROWS that share one payment date into a single
coupon, and the three declared conventions differ only in where the margin enters that fold.
Writing `int` for `(rate + margin) * accrual`, `mrg` for `margin * accrual` and `N` for the
nominal:

    Include_Margin   total + int * (total + N)
    Flat             total + int * N + total * (int - mrg)
    Exclude_Margin   total + (int - mrg) * (total + N) + mrg * N

`Exclude_Margin` was byte-identical to `Flat` and is now written as the spread-exclusive
convention reads. THAT CHANGES NO NUMBER, and the reason is the finding: expand the last two and
each is `total * (1 + rate * accrual) + int * N`. They are one function written two ways, at every
margin and not only at zero, so no test can tell them apart - which is why the discriminating gate
below is against `Include_Margin` rather than against `Flat`.

The forecast curve is flat and continuously compounded, so every quantity in the fold is a closed
form: a reset spanning `d` days is `expm1(R * d / 365)` and a payment `d` days out discounts at
`exp(-R * d / 365)`. Nothing here is fitted or copied.
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
from derivus import utils
from derivus.config import Config
from derivus.instruments import construct_instrument

BASE = pd.Timestamp('2020-01-01')
DTYPE = torch.float64
RATE = 0.03           # flat continuously-compounded zero curve, ACT/365
NOTIONAL = 1_000_000.0
MARGIN = 75.0         # basis points, and the whole point - at zero margin all three folds agree

#: One annual coupon built from three quarterly sub-periods, all paying on the coupon's date. That
#: is the shape the fold exists for: `cash_counts` is 3 on one payment date, and each row carries
#: exactly ONE reset so nothing is reduced before the fold sees it.
SUB_PERIODS = [(0, 3), (3, 6), (6, 9)]
PAY_DATE = BASE + pd.DateOffset(months=9)


def _items(margin):
    out = []
    for start_m, end_m in SUB_PERIODS:
        start, end = BASE + pd.DateOffset(months=start_m), BASE + pd.DateOffset(months=end_m)
        accrual = (end - start).days / 365.0
        out.append({'Payment_Date': PAY_DATE, 'Notional': NOTIONAL, 'Fixed_Amount': 0.0,
                    'Accrual_Start_Date': start, 'Accrual_End_Date': end,
                    'Accrual_Day_Count': 'ACT_365', 'Accrual_Year_Fraction': accrual,
                    'Margin': utils.Basis(margin),
                    'Resets': [[start, start, end, accrual, pd.DateOffset(months=3),
                                'ACT_365', '0D', 0.0, 'No', utils.Percent(0.0)]]})
    return out


def _price(method, margin):
    deal = {'Object': 'CFFloatingInterestListDeal', 'Reference': 'FLT1', 'Currency': 'USD',
            'Discount_Rate': 'USD', 'Forecast_Rate': 'USD', 'Buy_Sell': 'Buy',
            'Forecast_Rate_Cap_Volatility': '', 'Discount_Rate_Cap_Volatility': '',
            'Forecast_Rate_Swaption_Volatility': '', 'Discount_Rate_Swaption_Volatility': '',
            'Cashflows': {'Compounding_Method': method, 'Averaging_Method': 'Average_Interest',
                          'Properties': [], 'Items': _items(margin)}}
    config = Config()
    config.params['System Parameters']['Base_Currency'] = 'USD'
    config.params['System Parameters']['Base_Date'] = BASE
    config.params['Price Factors'] = {
        'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Priority': 1, 'Spot': 1.0},
        'InterestRate.USD': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                             'Curve': utils.Curve([], [[0.0, RATE], [5.0, RATE]])}}
    config.params['Price Models'] = {}
    config.params['Valuation Configuration'] = {}
    config.deals = {'Attributes': {'Reference': 'test', 'Tag_Titles': ''},
                    'Deals': {'Children': [{'Instrument': construct_instrument(deal, {})}]},
                    'Calculation': {'Base_Date': BASE, 'Currency': 'USD'}}
    _, out = derivus.run_baseval(config, prec=DTYPE,
                                 overrides={'MCMC_Simulations': 1, 'Random_Seed': 1})
    rows = out['Results']['mtm']
    return float(rows[rows['Reference'] == 'FLT1']['Value'].iloc[0])


def closed_form(method, margin):
    """The coupon by hand: fold the three sub-periods, then discount the one payment.

    A sub-period spanning `d` days forecasts `expm1(R * d / 365)` off the flat curve, which the
    pricer divides by the accrual and the fold multiplies back - so `int` is that number plus
    `margin * accrual`, with no rate ever formed. `mrg` is `margin * accrual`.
    """
    total, simple = 0.0, 0.0
    for start_m, end_m in SUB_PERIODS:
        start, end = BASE + pd.DateOffset(months=start_m), BASE + pd.DateOffset(months=end_m)
        accrual = (end - start).days / 365.0
        mrg = margin * 1e-4 * accrual
        interest = np.expm1(RATE * (end - start).days / 365.0) + mrg
        if method == 'Include_Margin':
            total = total + interest * (total + NOTIONAL)
        elif method == 'Flat':
            total = total + interest * NOTIONAL + total * (interest - mrg)
        else:
            # two accumulators, per the spec: only the rate part J compounds; each period's margin
            # is simple on the nominal and must never enter the compounding pot
            simple = simple + mrg * NOTIONAL
            total = total + (interest - mrg) * (total + NOTIONAL)
    if method == 'Exclude_Margin':
        total = total + simple
    return total * np.exp(-RATE * (PAY_DATE - BASE).days / 365.0)


def test_exclude_margin_folds_the_way_the_convention_reads():
    """The value gate: a three-sub-period coupon with a 75bp spread against a hand-computed fold.

    The margin has to be non-zero or this gate is blind - every convention collapses to the same
    number at zero spread, which is exactly why the branch could be wrong for years.
    """
    assert _price('Exclude_Margin', MARGIN) == pytest.approx(
        closed_form('Exclude_Margin', MARGIN), rel=1e-11)


def test_the_closed_form_rejects_the_neighbouring_convention():
    """MUTATE the oracle. `Include_Margin` is the same fold with the spread left inside it, and it
    is the only one of the three that is genuinely a different function - so the gate above has to
    reject its closed form, or it is asserting a coincidence of two conventions."""
    priced = _price('Exclude_Margin', MARGIN)
    wrong = closed_form('Include_Margin', MARGIN)
    assert abs(priced - wrong) / abs(priced) > 1e-3, (
        'the two conventions are indistinguishable on this fixture, so the value gate proves '
        'nothing: {} against {}'.format(priced, wrong))


@pytest.mark.parametrize('method', ['Include_Margin', 'Flat', 'Exclude_Margin'])
def test_every_convention_collapses_to_one_at_zero_spread(method):
    """With no margin there is nothing to place, so all three folds are `total*(1+int) + int*N` and
    a zero-margin book cannot move whatever this branch does. That is what makes the restatement
    safe, and it is the half of the identity worth holding in place."""
    assert _price(method, 0.0) == pytest.approx(closed_form('Include_Margin', 0.0), rel=1e-11)


def test_flat_and_exclude_margin_differ_by_the_compounded_margin():
    """The discriminating gate the spec makes writable.

    Per the convention spec: Flat lets period i's FULL interest (margin included) enter the
    compounding pot, Exclude keeps each period's margin simple - so on a multi-sub-period coupon
    with a positive spread, Flat - Exclude = sum_i m_i.alpha_i.N.(prod_{j>i}(1+J_j) - 1) > 0.
    Before the two-accumulator fold this difference was unrepresentable: both branches were the
    same function, and the original one-line restatement collapsed back to Flat because a margin
    lump inside `total` earns (1+J) in every later step."""
    flat, exclude = _price('Flat', MARGIN), _price('Exclude_Margin', MARGIN)

    expected_gap = 0.0
    js = []
    for start_m, end_m in SUB_PERIODS:
        start, end = BASE + pd.DateOffset(months=start_m), BASE + pd.DateOffset(months=end_m)
        accrual = (end - start).days / 365.0
        js.append((np.expm1(RATE * (end - start).days / 365.0), MARGIN * 1e-4 * accrual))
    for i, (_, mrg) in enumerate(js):
        compounding = np.prod([1.0 + j for j, _ in js[i + 1:]])
        expected_gap += mrg * NOTIONAL * (compounding - 1.0)
    expected_gap *= np.exp(-RATE * (PAY_DATE - BASE).days / 365.0)

    assert flat > exclude, 'Flat must exceed Exclude at positive margin'
    assert flat - exclude == pytest.approx(expected_gap, rel=1e-10)
