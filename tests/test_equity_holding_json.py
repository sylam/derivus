"""A share held to its `Investment_Horizon`, through the JSON contract.

A holding is paid what its share pays as it falls, to its horizon: the dividends of each interval
between two of its rows, on the later row, realised as an equity swap realises them; on the horizon
it is closed out at its value; it is worth nothing after; on a row it marks its shares and the
dividends it is paid on it, `(+/- N S + D) X`.

The world is the declared-defaults book's: the share EQ at 100 in USD, its repo the USD curve flat
at 4% and its dividend yield flat at 1%, both ACT/365 and unmodelled, so a share's dividends over
`(t0, t1]` are `N S(t0) exp(r tau) (1 - exp(-q tau))`, tau the interval in years; the share under
a GBM of 20% vol, float64, 256 paths, seed 1, on a quarterly grid, ten shares held a year.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import pytest
import torch

import test_declared_defaults as book
from derivus import utils

B = book.WORLD_BASE
HORIZON, N, R, Q = B + pd.DateOffset(years=1), 10.0, 0.04, 0.01
SHARE = {'Object': 'EquityDeal', 'Reference': 'SHARE', 'Currency': 'USD', 'Equity': 'EQ',
         'Buy_Sell': 'Buy', 'Units': N, 'Investment_Horizon': HORIZON}
GBM = {'EquityPrice.EQ': ('GBMAssetPriceModel', {'Vol': 0.2, 'Drift': 0.0})}


def dividend_curve(q):
    return {'DividendRate.EQ': {'Floor': None, 'Currency': 'USD', 'Curve': utils.Curve([], [[0.0, q], [5.0, q]])}}


def simulated(share, q=Q):
    """`(report dates, the share's rows by paths, its scenarios by date, {day: cash by paths})` off one
    float64 credit Monte Carlo of `share` alone."""
    _, out = book.simulated([share], [], factors=dividend_curve(q), models=GBM, prec=torch.float64,
                            Generate_Cashflows='Yes', Calc_Scenarios='All')
    assert 'Deals Skipped' not in out['Stats'], out['Stats']
    rows = np.concatenate(next(out['Netting'].deals()).Calc_res['Value'], axis=-1)
    cash = out['Results']['cashflows']['USD']
    return (list(out['Results']['mtm'].index), rows, out['Results']['scenarios']['EquityPrice.EQ'].loc[0.0],
            {day: cash.loc[day].values for day in cash.index})


def test_a_share_held_marks_its_shares_and_what_they_pay_that_day():
    """Ten shares held a year mark 1,000 on the base valuation, exactly `N S`, and under the Monte
    Carlo every row to the horizon is ten times the share's own scenario on that date and the
    dividends the holding is paid on it, to the bit, so the last row is what the horizon settles;
    none is reported after it. Paying no dividend, every row is `N S`.

    Killing mutations: the mark discounted on the dividend yield, its rows after the base date then
    under `N S`; the day's dividends left out of the row, the horizon then settling more than the
    holding is worth on it.
    """
    marks, stats = book.marks([SHARE])
    assert float.fromhex(marks['SHARE']) == N * 100.0 and 'Deals Skipped' not in stats, (marks, stats)
    for q in (Q, 0.0):
        dates, rows, spot, cash = simulated(SHARE, q)
        held = [day for day in dates if day <= HORIZON]
        assert held[-1] == HORIZON and rows.shape[0] == len(held), (rows.shape, held)
        shares = N * np.stack([spot[day].values for day in held])
        paid = np.stack([cash[day] for day in held[:-1]])
        assert np.array_equal(rows[:-1], shares[:-1] + paid) and np.array_equal(rows[-1], cash[HORIZON]), q
        assert bool(q) == bool(paid.any())


@pytest.mark.parametrize('q', [Q, 0.0])
def test_a_share_held_is_paid_its_dividends_as_they_fall_and_closed_out_on_its_horizon(q):
    """Every row after the base date settles: on each the dividends since the row before,
    `N S(t0) exp(r tau) (1 - exp(-q tau))` off that row's own price, by hand to 1e-12, and on the
    horizon those of its last quarter with the holding closed out at `N S(H)`; nothing on the base
    date or after the horizon. At a 1% yield the first quarter pays 2.5429 on every path, off the
    base date's price; at none the holding is paid nothing before its horizon and on it exactly its
    row.

    Killing mutations: the dividends not booked; the close-out not booked.
    """
    dates, rows, spot, cash = simulated(SHARE, q)
    held = [day for day in dates if day <= HORIZON]
    shares = N * np.stack([spot[day].values for day in held])
    assert list(cash) == held and not cash[B].any(), list(cash)
    for k, day in enumerate(held[1:], start=1):
        tau = (day - held[k - 1]).days / 365.0
        hand = shares[k - 1] * np.exp(R * tau) * -np.expm1(-q * tau) + (shares[k] if day == HORIZON else 0.0)
        assert np.allclose(cash[day], hand, rtol=1e-12, atol=0.0), (day, cash[day][:3], hand[:3])
    if q:
        assert np.all(cash[held[1]] == cash[held[1]][0]) and round(cash[held[1]][0], 4) == 2.5429, cash[held[1]]
    else:
        assert not any(cash[day].any() for day in held[:-1]) and np.array_equal(cash[HORIZON], rows[-1])


def test_a_share_sold_is_paid_the_other_way():
    """Ten shares held short on the same paths mark and are paid, row by row and day by day, minus
    what the long holding is, to the bit.

    Killing mutation: the dividends read off the units unsigned, the short holding then paid them.
    """
    _, long_rows, _, long_cash = simulated(SHARE)
    _, short_rows, _, short_cash = simulated(dict(SHARE, Buy_Sell='Sell'))
    assert np.array_equal(short_rows, -long_rows) and list(short_cash) == list(long_cash)
    assert all(np.array_equal(short_cash[day], -long_cash[day]) for day in long_cash)
