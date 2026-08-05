"""The inflation index reference: which index observations a cashflow reads, and how weighted.

This was four hand-written closures - IndexReference{2M,3M,Interpolated3M,Interpolated4M} - selected
by building a STRING from two schema fields and dispatching on it through `locals()`. Two
consequences. Any (rule, lag) pair outside the four raised, including `Interpolated1M` and
`Interpolated2M` which `fields.py` offers as valid values and the shipped `Index_Reference` default
`('Interpolated', 1)`, which no lookup entry could satisfy. And the rule and the lag - genuinely
independent - were welded into one enumerated token.

They are two shapes parameterised by lag, so this pins both against the originals and checks the
lags that used to raise. Nothing else in the suite prices an inflation deal, so this is the only
thing standing between that generalisation and a silent regression.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest

from derivus.utils import index_reference_samples

# mid-month, so the interpolation weight is neither 0 nor 1 and a swapped pair would show
PRICING = pd.Timestamp('2024-06-18')
MONTH = lambda y, m: pd.Timestamp(year=y, month=m, day=1)
W = (PRICING - MONTH(2024, 6)).days / 30.0          # June has 30 days


@pytest.mark.parametrize('months_lag,expected', [
    (2, MONTH(2024, 4)),      # was IndexReference2M
    (3, MONTH(2024, 3)),      # was IndexReference3M
])
def test_a_plain_reference_reads_one_month_start(months_lag, expected):
    assert index_reference_samples(PRICING, months_lag, False) == [(expected, 1.0)]


@pytest.mark.parametrize('months_lag,first,second', [
    (3, MONTH(2024, 3), MONTH(2024, 4)),    # was IndexReferenceInterpolated3M
    (4, MONTH(2024, 2), MONTH(2024, 3)),    # was IndexReferenceInterpolated4M
])
def test_an_interpolated_reference_straddles_two(months_lag, first, second):
    """The weight is how far into its own month the pricing date sits, and it goes on the NEARER
    sample - getting that backwards is the failure a month-start fixture cannot see."""
    got = index_reference_samples(PRICING, months_lag, True)
    assert [d for d, _ in got] == [first, second]
    assert got[0][1] == pytest.approx(1.0 - W)
    assert got[1][1] == pytest.approx(W)
    assert sum(w for _, w in got) == pytest.approx(1.0)


@pytest.mark.parametrize('months_lag', [1, 2, 5, 12])
def test_the_lags_that_used_to_raise_now_resolve(months_lag):
    """`fields.py` offers Interpolated1M and Interpolated2M and defaults to a 1M lag, none of which
    the four-entry lookup could dispatch. Lag is now just an integer."""
    got = index_reference_samples(PRICING, months_lag, True)
    assert len(got) == 2
    assert got[0][0] == (PRICING - pd.DateOffset(months=months_lag)).to_period('M').to_timestamp('D')
    assert sum(w for _, w in got) == pytest.approx(1.0)


def test_a_month_start_pricing_date_puts_all_weight_on_the_far_sample():
    """w == 0 at a month start, so an interpolated reference degenerates to the plain one at the
    same lag - the boundary that says the weight is attached to the right end."""
    got = index_reference_samples(MONTH(2024, 6), 3, True)
    assert got[0] == (MONTH(2024, 3), 1.0)
    assert got[1][1] == 0.0
