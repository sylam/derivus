"""The credit family through the JSON contract, priced off `trial_credit`'s names in the
declared-defaults world."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest

import test_declared_defaults as book
import trial_credit
from derivus import utils

B = book.WORLD_BASE


def test_an_upfront_is_paid_by_the_protection_buyer_on_its_day():
    """The trial default swap without its amortisation: ten million protected on ISSUER_A for five
    years at a 100bp running coupon, quarterly ACT/365 with the maturity day counted, hazard 2% and
    recovery 40% on the flat 4% USD curve, and a 2% upfront three days out. By hand, over each
    quarter (s, e] and the upfront's day u,

        V = sum [ 0.6 N (D(s) + D(e)) / 2 (S(s) - S(e)) - c a N S(e) D(e) ] - 2% N D(u)

    the upfront paid by the protection buyer whatever the name does, so discounted and never
    survival-weighted: -110,464.68 bought and its negative sold. Stated as a bare 2.0 it is 2%, as
    the bare coupon 1.0 is 1%. Stated without a date it is paid on the effective date; dated behind
    the base date it has been paid, and the mark is the plain swap's to the bit. A credit Monte
    Carlo books it as the buyer's cash on its day.

    KILLING MUTATION: the upfront term dropped from `pv_credit_cashflows`: the bought swap reads
    89,469.57 against -110,464.68.
    """
    def discount(day, rate=0.04):
        return math.exp(-rate * (day - B).days / 365.0)

    dates = [B + pd.DateOffset(months=3 * k) for k in range(21)]
    plain = sum(0.6 * 1e7 * (discount(s) + discount(e)) / 2 *
                (discount(s, 0.02) - discount(e, 0.02))
                - 0.01 * ((e - s).days + (e == dates[-1])) / 365.0 * 1e7 * discount(e, 0.02) *
                discount(e) for s, e in zip(dates, dates[1:]))
    day = B + pd.Timedelta(days=3)
    cds = {key: value for key, value in trial_credit.DEALS[0].items() if key != 'Amortisation'}
    upfront = dict(cds, Reference='UPFRONT', Upfront=utils.Percent(2.0), Upfront_Date=day)
    hexes, _ = book.marks([cds, upfront, dict(upfront, Reference='SOLD', Buy_Sell='Sell'),
                           dict(upfront, Reference='BARE', Upfront=2.0),
                           dict(upfront, Reference='UNDATED', Upfront_Date=None),
                           dict(upfront, Reference='PAID', Upfront_Date=B - pd.Timedelta(days=3))],
                          trial_credit.FACTORS)
    valued = {reference: float.fromhex(value) for reference, value in hexes.items()}

    assert valued['CDS'] == pytest.approx(plain, rel=1e-12)
    assert valued['UPFRONT'] == pytest.approx(plain - 2e5 * discount(day), rel=1e-12)
    assert valued['SOLD'] == -valued['UPFRONT']
    assert valued['BARE'] == valued['UPFRONT']
    assert valued['UNDATED'] == pytest.approx(plain - 2e5, rel=1e-12)
    assert valued['PAID'] == valued['CDS']

    _, out = book.simulated([upfront], ('USD',), trial_credit.FACTORS, Generate_Cashflows='Yes')
    assert float(out['Results']['cashflows']['USD'].loc[day].iloc[0]) == pytest.approx(
        -2e5, rel=1e-6)
