"""The commodity one-touch and no-touch, through the JSON contract and nothing else.

The two types are the equity touches on a reference price's fixing: the forward curve read at the
delivery each date maps to is the monitored price, and its carry to the expiry is the log-ratio of
the expiry's forward to it. So a commodity touch on a reference whose forward curve IS an equity's
forward prices as that equity touch does - the identity the first test holds every variant to.
"""
import logging
import sys

import numpy as np
import pytest

sys.path.insert(0, 'tests')
import test_declared_defaults as book  # noqa: E402
import trial_commodity  # noqa: E402
from derivus import utils  # noqa: E402

B, E = book.WORLD_BASE, book.WORLD_EXPIRY
BASE, EXPIRY = ((d - utils.excel_offset).days for d in (B, E))
#: the world's equity: spot 100 on the flat 4% USD curve less a flat 1% dividend
SPOT, CARRY = 100.0, 0.04 - 0.01
YEARS = (E - B).days / 365.0
IDENTITY = utils.Curve([], [[40000, 40000], [60000, 60000]])


def reference(name, fixing_curve=IDENTITY):
    """A reference whose forward curve is the world equity's own forward, read at the delivery
    `fixing_curve` maps each date to, with the equity's vol surface."""
    return {
        'ReferencePrice.' + name: {'Fixing_Curve': fixing_curve, 'ForwardPrice': name},
        'ForwardPrice.' + name: {'Currency': 'USD', 'Curve': utils.Curve(
            [], [[BASE, SPOT], [EXPIRY, SPOT * np.exp(CARRY * YEARS)]])},
        'CommodityPriceVol.' + name: book.world()['EquityPriceVol.EQ']}


def touch(kind, ref, barrier_type, barrier, timing=None, **extra):
    deal = {'Object': kind, 'Reference': ref, 'Currency': 'USD', 'Discount_Rate': 'USD',
            'Buy_Sell': 'Buy', 'Expiry_Date': E, 'Cash_Payoff': 10_000.0,
            'Barrier_Type': barrier_type, 'Barrier_Price': barrier, **extra}
    return dict(deal, Payment_Timing=timing) if timing else deal


VARIANTS = [('OneTouch', 'Down', 90.0, 'Touch'), ('OneTouch', 'Down', 90.0, 'Expiry'),
            ('OneTouch', 'Up', 110.0, 'Touch'), ('OneTouch', 'Up', 110.0, 'Expiry'),
            ('NoTouch', 'Down', 90.0, None), ('NoTouch', 'Up', 110.0, None)]


def test_a_commodity_touch_prices_as_the_equity_touch_its_forward_maps_to():
    """Six variants - one-touch down and up paid on the touch and at expiry, no-touch down and up -
    each booked on the world's equity and on a spot-like reference whose forward curve is that
    equity's forward (spot 100, carry 3%, the same surface): the two marks agree to 1e-12 of the
    payout, every mark non-zero.

    KILLING MUTATION: the carry left at zero, the closed form's forward the watched fixing itself -
    the down one-touch paid on the touch 4,996.10 against the equity's 4,332.39.
    """
    deals = []
    for kind, barrier_type, barrier, timing in VARIANTS:
        tag = '{}_{}_{}'.format(kind, barrier_type, timing or '')
        deals.append(touch('Equity' + kind + 'Option', 'EQ_' + tag, barrier_type, barrier, timing,
                           Equity='EQ', Equity_Volatility='EQ'))
        deals.append(touch('Commodity' + kind + 'Option', 'CM_' + tag, barrier_type, barrier, timing,
                           Reference_Type='EQC', Reference_Volatility='EQC'))
    marks, _ = book.marks(deals, reference('EQC'))
    for kind, barrier_type, barrier, timing in VARIANTS:
        tag = '{}_{}_{}'.format(kind, barrier_type, timing or '')
        equity, commodity = (float.fromhex(marks[side + tag]) for side in ('EQ_', 'CM_'))
        assert equity != 0.0
        assert commodity == pytest.approx(equity, abs=1e-12 * 10_000.0), tag


#: one 0.3 correlation spelled the way each family names it: the equity's on the sorted pair, read
#: with the flip the deal's USD -> EUR direction gives it, the reference's local to payoff, unsigned
CORR = 0.3
CROSS = {'Correlation.EquityPrice.EQ/FxRate.EUR.USD': {'Value': CORR},
         'Correlation.FxRate.USD.EUR/ReferencePrice.EQC.USD': {'Value': -CORR}}
#: a quanto's barrier is a USD level on the forward, a compo's a EUR level on S*X (80 today)
PAID_IN_EUR = [('Quanto', 'Down', 90.0), ('Quanto', 'Up', 110.0), ('Compo', 'Down', 72.0), ('Compo', 'Up', 88.0)]


def test_a_quanto_or_compo_commodity_touch_prices_as_the_equity_touch_it_maps_to():
    """Paid in EUR on a USD reference: each quanto and compo one-touch, down and up, on the touch and
    at expiry, booked on the world's equity and on the reference whose forward is that equity's, the
    one correlation stated in each family's own spelling - the two agree to 1e-12 of the payout,
    every mark non-zero.

    KILLING MUTATION: the reference's correlation read with the opposite sign - the compo up-touch
    at 88 paid on the touch reads 8,350.41 against the equity's 7,213.00.
    """
    deals = []
    for ptype, barrier_type, barrier in PAID_IN_EUR:
        for timing in ('Touch', 'Expiry'):
            tag = '{}_{}_{}'.format(ptype, barrier_type, timing)
            deals.append(touch('EquityOneTouchOption', 'EQ_' + tag, barrier_type, barrier, timing,
                               Equity='EQ', Equity_Volatility='EQ', Payoff_Currency='EUR',
                               Discount_Rate='EUR', Payoff_Type=ptype))
            deals.append(touch('CommodityOneTouchOption', 'CM_' + tag, barrier_type, barrier, timing,
                               Reference_Type='EQC', Reference_Volatility='EQC', Currency='EUR',
                               Discount_Rate='EUR', Payoff_Type=ptype))
    marks, _ = book.marks(deals, {**reference('EQC'), **CROSS})
    for deal in deals[::2]:
        tag = deal['Reference'][3:]
        equity, commodity = (float.fromhex(marks[side + tag]) for side in ('EQ_', 'CM_'))
        assert equity != 0.0
        assert commodity == pytest.approx(equity, abs=1e-12 * 10_000.0), tag


def test_the_touch_paid_at_expiry_and_the_no_touch_sum_to_the_discounted_payout():
    """On the trial's own fuel reference, whose forward curve slopes from 80 to 84: the one-touch
    paid at expiry and the no-touch on the same terms are complements, together the payout
    discounted on the 4% USD curve to 1e-9 of it.

    KILLING MUTATION: `no_touch` dropped from the no-touch, which then prices the one-touch: the two
    sum to 17,365.84 against the 9,607.89 payout (8,682.92 + 924.98 as built).
    """
    terms = dict(trial_commodity.TOUCH, Barrier_Price=83.0)
    deals = [dict(terms, Object='CommodityOneTouchOption', Reference='OT', Payment_Timing='Expiry'),
             dict(terms, Object='CommodityNoTouchOption', Reference='NT')]
    marks, _ = book.marks(deals, trial_commodity.FACTORS)
    one, none = (float.fromhex(marks[r]) for r in ('OT', 'NT'))
    assert 0.0 < one < 10_000.0 and 0.0 < none < 10_000.0
    assert one + none == pytest.approx(10_000.0 * np.exp(-0.04 * YEARS), rel=1e-9)


def test_a_reference_rolling_to_another_contract_before_the_expiry_is_refused_by_name(caplog):
    """A reference stepping from one delivery to the next a month in watches a fixing that jumps at
    the roll, which the closed form cannot price: the deal is skipped with its name and the roll
    in the log, and the touch on a reference that stays on one contract prices beside it.

    KILLING MUTATION: the roll check dropped, the deal priced off one forward it never watched.
    """
    roll = BASE + 30
    rolling = utils.Curve([], [[40000, EXPIRY], [roll, EXPIRY], [roll + 1, EXPIRY + 31], [60000, EXPIRY + 31]])
    one_contract = utils.Curve([], [[40000, EXPIRY], [60000, EXPIRY]])
    factors = {**reference('ROLL', rolling), **reference('HOLD', one_contract)}
    deals = [touch('CommodityOneTouchOption', 'ROLLS', 'Up', 110.0, 'Touch',
                   Reference_Type='ROLL', Reference_Volatility='ROLL'),
             touch('CommodityOneTouchOption', 'HOLDS', 'Up', 110.0, 'Touch',
                   Reference_Type='HOLD', Reference_Volatility='HOLD')]
    with caplog.at_level(logging.ERROR):
        marks, _ = book.marks(deals, factors)
    assert 'ROLLS' not in marks and float.fromhex(marks['HOLDS']) != 0.0
    assert any('ROLLS' in r.getMessage() and 'rolls from one delivery to another' in r.getMessage()
               for r in caplog.records)


def test_under_a_credit_monte_carlo_the_touch_profile_is_finite_and_ends_at_its_expiry(caplog):
    """The trial's two touches in a credit Monte Carlo with the discount curve simulated and the
    forward curve static: neither is skipped, every row is finite, and the profile is positive
    before the expiry and zero after it.

    KILLING MUTATION: a static forward curve's single column left as it is, which the touch pricer
    cannot stack against the simulated discount's 256 scenarios - the one-touch is skipped.
    """
    deals = [d for d in trial_commodity.DEALS if 'Touch' in d['Object']]
    with caplog.at_level(logging.ERROR):
        _, out = book.simulated(deals, ['USD'], trial_commodity.FACTORS)
    assert not [r for r in caplog.records if 'skipped' in r.getMessage()]
    mtm, profile = out['Results']['mtm'], out['Results']['exposure_profile']
    assert mtm.shape == (len(profile), 256) and np.isfinite(mtm.to_numpy()).all()
    before, after = profile.index < E, profile.index > E
    assert before.any() and after.any()
    assert (profile.loc[before, 'EE'] > 0.0).all() and (profile.loc[after].abs().to_numpy() == 0.0).all()
