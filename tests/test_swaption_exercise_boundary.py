"""A physically settled swaption exercises on ONE number, and that number is what it registers.

`SwaptionDeal.post_process` splits its rows at expiry - `mn_option` are the rows up to and
including it, `mn_swap` the rows after - and takes the exercise decision on the LAST PRE-EXPIRY
row, `delta * mn_option[-1]`. The rows after expiry then report the delivered swap or nothing:

    exercised = (delta * mn_option[-1]) >= 0
    mtm[t > expiry] = buysell * Ut_swap * exercised

The boundary registration was handed `Ut_swap[0]` - the FIRST POST-EXPIRY row - as its gap. The two
agree only where moneyness did not move between those rows, and the flux a boundary registers is
precisely the paths closest to zero: the ones a day of rate moves flips. So the correction repaired
the derivative of a decision the pricer never took.

The fix passes the DECISION as the gap, making `fired` equal `gap >= 0` by construction - and both
halves are READABLE OFF A PRICED DOCUMENT:

  * at the expiry row the option has no life left, so its value is the INTRINSIC
    `pvbp * max(delta * (s - K), 0)`, strictly positive exactly where `delta * mn_option[-1] > 0`.
    The payer's expiry row and the same two legs priced as a standalone swap agree to the last bit
    (733808.55014745 against 733808.55014746), which says the expiry row IS the gap with `pvbp`
    divided out.
  * on the first post-expiry row the swaption reports `buysell * Ut_swap[0]` if it exercised and
    exactly zero if it did not.

THE FIXTURE. A payer and a receiver on the SAME underlying - 1Y into 3Y, semi-annual fixed against
quarterly floating, ZAR 100m - struck at 9.50 against a 9.4932 forward, so about half the paths
exercise. The pair makes both directions of the flip visible.

The rows either side of expiry are one day apart structurally: `SwaptionDeal.finalize_dates` holds
the swap from the day after the expiry. One day of Hull-White at 1% absolute vol is what has to move
a path across, so the fixture pays for flips in PATHS: at 4096, seeds 1/2/3 give 23/32/25 payer
flips and 36/31/33 receiver flips against 2076/2077/2091 exercises.

KILL READINGS, two earlier builds failing different gates. With the DECISION also at `Ut_swap[0]`,
the first post-expiry row is `relu(Ut_swap[0])` for a bought payer, so the flip count is
structurally zero and the consistency gate fails on 59 of 4096 paths for each swaption - the SAME
59. With the decision right and only the GAP from `Ut_swap[0]`, nothing reported moves - profile,
exposure and CVA bit-identical - and only `gap_disagrees` sees it: 10 of 512 at seed 1, 59 of 4096.

THE VOL THE OPTION READS, gated last on base valuations of job documents against Black and
Bachelier by hand: an amortising swap at its duration-equivalent tenor, one starting after the
expiry at its mid-curve vol.
"""
import json
import logging
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import pytest
import torch

import derivus
from derivus import utils
from derivus.config import Config, CustomJsonEncoder
from derivus.instruments import construct_instrument

import rates_world as rw
import test_declared_defaults as book

DTYPE = torch.float64
BASE = rw.BASE
CURRENCY, DISCOUNT_CURVE, PROJECTION_CURVE = 'ZAR', 'ZAR-DISC', 'ZAR-PROJ'
COUNTERPARTY = 'CPTY'

#: One year into three. The option expiry IS the swap effective date - a forward-starting
#: underlying is explicitly out of scope for this pricer's vol read.
EXPIRY = BASE + pd.DateOffset(years=1)
SWAP_END = EXPIRY + pd.DateOffset(years=3)
DAY_COUNT = 'ACT_365'
NOTIONAL = 1e8

#: A hiking ZAR-shaped world, sloped and multi-curve: the 3M projection curve carries a basis over
#: the discount curve that widens with maturity. The front knot is a day rather than zero, because
#: `HullWhite1FactorInterestRateModel` divides its assembled curve by the factor's own tenors.
KNOTS = [1.0 / 365.0, 0.25, 1.0, 2.0, 3.0, 5.0, 10.0]
DISCOUNT_ZEROS = [0.0700, 0.0715, 0.0760, 0.0800, 0.0830, 0.0860, 0.0875]
PROJECTION_ZEROS = [0.0745, 0.0762, 0.0810, 0.0852, 0.0884, 0.0916, 0.0932]

#: Six basis points out of the money. The forward par rate of this underlying is 9.4932%, priced
#: off these curves by the engine's own cashflow pricers; the strike only has to be NEAR it for the
#: exercise decision to be live, which the partition gate below is what measures.
STRIKE = 9.50

#: One percent ABSOLUTE short-rate vol - high for rates, and deliberately so: the flip window is a
#: single day, so the fixture buys its boundary crossings with vol and paths.
SIGMA = 0.010
VOL = 0.20
PATHS = 4096

NETTING = {
    'Object': 'NettingCollateralSet', 'Reference': 'NS1', 'Netted': 'True',
    'Agreement_Currency': CURRENCY, 'Funding_Rate': DISCOUNT_CURVE,
    'Balance_Currency': CURRENCY, 'Liquidation_Period': 0.0, 'Settlement_Period': 0.0,
    'Collateralized': 'False',
    'Credit_Support_Amounts': {
        'Received_Threshold': utils.CreditSupportList([[0.0, 0.0]]),
        'Posted_Threshold': utils.CreditSupportList([[0.0, 0.0]]),
        'Independent_Amount': utils.CreditSupportList([[0.0, 0.0]]),
        'Minimum_Received': utils.CreditSupportList([[0.0, 0.0]]),
        'Minimum_Posted': utils.CreditSupportList([[0.0, 0.0]])}}


def schedule(frequency_months):
    """Coupon dates from the swap effective date to its maturity."""
    dates = [EXPIRY]
    while dates[-1] < SWAP_END:
        dates.append(EXPIRY + pd.DateOffset(months=frequency_months * len(dates)))
    return dates


def fixed_leg(reference, buy_sell):
    """The semi-annual fixed leg, struck at `STRIKE`."""
    dates = schedule(6)
    items = [{'Payment_Date': end, 'Notional': NOTIONAL, 'Rate': utils.Percent(STRIKE),
              'Accrual_Start_Date': start, 'Accrual_End_Date': end,
              'Accrual_Day_Count': DAY_COUNT,
              'Accrual_Year_Fraction': utils.DayCount.accrual(
                  start, (end - start).days, utils.DayCount.code(DAY_COUNT)),
              'Fixed_Amount': 0.0, 'Discounted': 'No',
              'FX_Reset_Date': None, 'Known_FX_Rate': 0.0}
             for start, end in zip(dates[:-1], dates[1:])]
    return rw._cashflow_leg('CFFixedInterestListDeal', reference, CURRENCY, DISCOUNT_CURVE,
                            buy_sell, {'Compounding': 'No', 'Items': items},
                            Calendars=None, Rate_Currency='')


def floating_leg(reference, buy_sell):
    """The quarterly floating leg - one reset per coupon, spanning its own accrual period."""
    dates = schedule(3)
    items = []
    for start, end in zip(dates[:-1], dates[1:]):
        accrual = utils.DayCount.accrual(
            start, (end - start).days, utils.DayCount.code(DAY_COUNT))
        items.append({
            'Payment_Date': end, 'Notional': NOTIONAL,
            'Accrual_Start_Date': start, 'Accrual_End_Date': end,
            'Accrual_Day_Count': DAY_COUNT, 'Accrual_Year_Fraction': accrual,
            'Resets': [[start, start, end, accrual, pd.DateOffset(days=1), DAY_COUNT, '0D', 0.0,
                        'No', utils.Percent(0.0)]],
            'Margin': utils.Basis(0.0), 'Fixed_Amount': 0.0,
            'FX_Reset_Date': None, 'Known_FX_Rate': 0.0})
    return rw._cashflow_leg('CFFloatingInterestListDeal', reference, CURRENCY, DISCOUNT_CURVE,
                            buy_sell,
                            {'Compounding_Method': 'None', 'Averaging_Method': 'Average_Rate',
                             'Properties': [], 'Items': items},
                            Forecast_Rate=PROJECTION_CURVE, Rate_Adjustment_Method='None',
                            Rate_Sticky_Month_End='Yes', Rate_Offset=0, Rate_Calendars=None,
                            Accrual_Calendars=None, Forecast_Rate_Cap_Volatility='',
                            Forecast_Rate_Swaption_Volatility='',
                            Discount_Rate_Cap_Volatility='',
                            Discount_Rate_Swaption_Volatility='')


def swaption(reference, payer_receiver):
    """One physically settled swaption. Its legs are its CHILDREN - `post_process` prices those and
    ignores the schedules the deal compiles for itself - and both are bought with positive
    notionals, so `pvbp`, `vfixed` and `vfloat` are all positive and `delta` carries the
    direction."""
    return {
        'Object': 'SwaptionDeal', 'Reference': reference, 'Currency': CURRENCY,
        'Discount_Rate': DISCOUNT_CURVE, 'Forecast_Rate': PROJECTION_CURVE,
        'Forecast_Rate_Volatility': PROJECTION_CURVE,
        'Buy_Sell': 'Buy', 'Payer_Receiver': payer_receiver, 'Settlement_Style': 'Physical',
        'Option_Expiry_Date': EXPIRY, 'Swap_Effective_Date': EXPIRY,
        'Swap_Maturity_Date': SWAP_END, 'Settlement_Date': EXPIRY,
        'Principal': NOTIONAL, 'Swap_Rate': STRIKE, 'Floating_Margin': 0.0,
        'Reset_Type': 'Standard', 'Index_Day_Count': DAY_COUNT,
        'Index_Tenor': pd.DateOffset(months=3), 'Index_Offset': 0, 'Index_Calendars': None,
        'Index_Publication_Calendars': None, 'Rate_Schedule': None, 'Margin_Schedule': None,
        'Pay_Frequency': pd.DateOffset(months=6), 'Pay_Day_Count': DAY_COUNT,
        'Pay_Timing': 'End', 'Pay_Payment_Offset': 0, 'Pay_Calendars': None,
        'Pay_Payment_Calendars': None, 'Pay_First_Coupon_Date': None,
        'Pay_Penultimate_Coupon_Date': None, 'Pay_Amortisation': None,
        'Receive_Frequency': pd.DateOffset(months=3), 'Receive_Day_Count': DAY_COUNT,
        'Receive_Timing': 'End', 'Receive_Payment_Offset': 0, 'Receive_Calendars': None,
        'Receive_Payment_Calendars': None, 'Receive_First_Coupon_Date': None,
        'Receive_Penultimate_Coupon_Date': None, 'Receive_Amortisation': None}


def config():
    """The world and the book: a payer and a receiver on one underlying, under one netting set."""
    c = Config(base_currency=CURRENCY)
    c.params['System Parameters']['Base_Date'] = BASE
    c.params['Price Factors'] = rw.market(
        CURRENCY, {DISCOUNT_CURVE: (KNOTS, DISCOUNT_ZEROS),
                   PROJECTION_CURVE: (KNOTS, PROJECTION_ZEROS)},
        DISCOUNT_CURVE, day_count=DAY_COUNT)
    surface = [(m, e, t, VOL) for m in (-0.02, 0.0, 0.02)
               for e in (0.25, 1.0, 2.0, 5.0) for t in (1.0, 3.0, 5.0)]
    # the swaption reads InterestYieldVol (its underlying is longer than a year); the caplet space
    # is authored beside it because `SwaptionDeal.factor_fields` declares both and a missing one is
    # a warning on every run
    c.params['Price Factors']['InterestYieldVol.' + PROJECTION_CURVE] = {
        'Property_Aliases': None, 'Distribution_Type': 'Lognormal', 'Shift': utils.Percent(0.0),
        'Surface': utils.Curve([], surface)}
    c.params['Price Factors']['InterestRateVol.' + PROJECTION_CURVE] = {
        'Property_Aliases': None, 'Surface': utils.Curve([], surface)}
    c.params['Price Factors']['SurvivalProb.' + COUNTERPARTY] = {
        'Recovery_Rate': 0.4, 'Curve': utils.Curve([], [[0.0, 0.0], [10.0, 0.4]])}
    for curve in (DISCOUNT_CURVE, PROJECTION_CURVE):
        c.params['Price Models']['HullWhite1FactorInterestRateModel.' + curve] = {
            'Alpha': 0.05, 'Lambda': 0.0, 'Quanto_FX_Correlation': 0.0,
            'Quanto_FX_Volatility': utils.Curve([], [[0.0, 0.0], [10.0, 0.0]]),
            'Sigma': utils.Curve([], [[0.0, SIGMA], [10.0, SIGMA]])}
    c.params['Model Configuration'].append(
        'InterestRate', (), 'HullWhite1FactorInterestRateModel')

    def node(reference, payer_receiver):
        return {'Instrument': construct_instrument(swaption(reference, payer_receiver), {}),
                'Children': [{'Instrument': construct_instrument(leg, {})} for leg in
                             (fixed_leg(reference + '_FIX', 'Buy'),
                              floating_leg(reference + '_FLT', 'Buy'))]}

    c.deals = {'Attributes': {'Reference': 'test'},
               'Deals': {'Children': [
                   {'Instrument': construct_instrument(NETTING, {}),
                    'Children': [node('PAYER', 'Payer'), node('RECEIVER', 'Receiver')]}]},
               'Calculation': {'Base_Date': BASE, 'Currency': CURRENCY}}
    return c


def run(seed=1, paths=PATHS, bandwidth=None):
    """One credit Monte Carlo with the CVA gradient on, which is what arms the registration.

    `boundary_aad` has no JSON switch of its own - wanting sensitivities IS the switch - so asking
    for `Credit_Valuation_Adjustment.Gradient` is what makes the swaption register at all.
    `DealLevel` is what keeps each deal's own profile, which is the whole reading below: a netting
    set's mtm is a SUM, and the payer and the receiver are in it together.
    """
    overrides = {
        'Run_Date': BASE.strftime('%Y-%m-%d'), 'Time_grid': '0d 3m(3m)', 'Batch_Size': paths,
        'Simulation_Batches': 1, 'Random_Seed': seed, 'Currency': CURRENCY, 'Tenor_Offset': 0.0,
        'MCMC_Simulations': 1, 'Deflation_Interest_Rate': DISCOUNT_CURVE,
        'Generate_Cashflows': 'No', 'Gradient_Variables': 'Factors', 'DealLevel': True,
        'Credit_Valuation_Adjustment': {
            'Calculate': 'Yes', 'Counterparty': COUNTERPARTY, 'Deflate_Stochastically': 'No',
            'Stochastic_Hazard_Rates': 'No', 'Gradient': 'Yes'}}
    if bandwidth is not None:
        overrides['Boundary_AAD_Bandwidth'] = bandwidth
    return derivus.run_cmc(config(), prec=DTYPE, overrides=overrides)


def deal_profiles(calc):
    """`{Reference: (rows, paths)}` for every deal and structure the run stored one for."""
    found = {}

    def walk(struct):
        for sub in struct.sub_structures:
            result = sub.obj.Calc_res
            if result and 'Value' in result:
                found[sub.obj.Instrument.field.get('Reference')] = np.concatenate(
                    result['Value'], axis=-1)
            walk(sub)
        for dependency in struct.dependencies:
            if dependency.Calc_res and 'Value' in dependency.Calc_res:
                found[dependency.Instrument.field.get('Reference')] = np.concatenate(
                    dependency.Calc_res['Value'], axis=-1)

    walk(calc.netting_sets)
    return found


def rows_either_side(dates):
    """`(last pre-expiry row, first post-expiry row)`. The split the pricer takes is
    `tenor >= 0`, so the pre-expiry row is the expiry date itself."""
    pre = max(index for index, date in enumerate(dates) if date <= EXPIRY)
    assert dates[pre] == EXPIRY, 'the expiry is not an MTM row: {}'.format(dates[pre])
    assert pre + 1 < len(dates), 'nothing is reported after expiry, so no swap row exists'
    return pre, pre + 1


def read(seed=1, paths=PATHS, **kwargs):
    """`(dates, pre, post, {Reference: profile})` off one run."""
    calc, out = run(seed=seed, paths=paths, **kwargs)
    dates = list(out['Results']['mtm'].index)
    pre, post = rows_either_side(dates)
    return dates, pre, post, deal_profiles(calc), out


# ------------------------------------------------------------------- the decision IS the gap

@pytest.mark.parametrize('reference', ['PAYER', 'RECEIVER'])
def test_the_exercise_matches_the_moneyness_of_the_row_it_was_decided_on(reference):
    """`exercised == (gap >= 0)`, path for path, read off the priced document.

    The gap is `delta * mn_option[-1]` and the expiry row reports `pvbp` times its RELU, so
    `profile[expiry] > 0` is the gap's sign with `pvbp > 0` divided out. `exercised` is the first
    post-expiry row being anything other than the zero the unexercised branch reports.

    Against a DECISION taken at the post-expiry row the two disagree on every path whose moneyness
    crossed zero between the rows - the population the correction exists to weight.

    Killing mutation: the decision read off the first post-expiry row.
    """
    _, pre, post, profiles, _ = read()
    profile = profiles[reference]
    in_the_money_at_expiry = profile[pre] > 0.0
    exercised = profile[post] != 0.0

    disagreed = np.nonzero(in_the_money_at_expiry != exercised)[0]
    assert disagreed.size == 0, (
        '{}: {} of {} paths exercised against the moneyness of the row the decision was taken on '
        '- first offenders {}'.format(reference, disagreed.size, profile.shape[1],
                                      disagreed[:8].tolist()))
    # a gate that fires on nothing would pass here too
    assert 0 < int(exercised.sum()) < exercised.size, (
        '{}: {} of {} paths exercised - the strike is not near enough the money for the decision '
        'to be live'.format(reference, int(exercised.sum()), exercised.size))


def test_a_path_whose_moneyness_flips_after_expiry_still_delivers_the_swap():
    """THE FLIP - the population every gate here is about, counted. A path in the money at expiry
    exercises, and if the rate moves against it overnight it holds a swap worth LESS THAN NOTHING on
    the next reporting row. With the decision at `Ut_swap[0] >= 0` that row is `relu(Ut_swap[0])`
    and can never be negative, so the flip count is structurally zero.

    Both directions are read, which is what the receiver is in the book for: a receiver exercises
    exactly where a payer does not, so a single deal leaves half the boundary unmeasured.

    Killing mutation: the decision read off the first post-expiry row.
    """
    _, pre, post, profiles, _ = read()
    flips = {}
    for reference in ('PAYER', 'RECEIVER'):
        profile = profiles[reference]
        flipped = (profile[pre] > 0.0) & (profile[post] < 0.0)
        flips[reference] = int(flipped.sum())
        assert flips[reference] > 0, (
            '{}: no path flipped between {} and the next row, so this gate asserts nothing. Raise '
            'the path count or the vol.'.format(reference, EXPIRY.date()))
        # the delivered swap, not a rounding artefact
        assert np.abs(profile[post][flipped]).min() > 1.0, (
            '{}: the flipped paths carry no material swap value'.format(reference))

    assert set(np.nonzero((profiles['PAYER'][pre] > 0.0) &
                          (profiles['PAYER'][post] < 0.0))[0]).isdisjoint(
        np.nonzero((profiles['RECEIVER'][pre] > 0.0) &
                   (profiles['RECEIVER'][post] < 0.0))[0]), (
        'the same path flipped on both swaptions, which cannot happen - they exercise on opposite '
        'sides of one boundary')


def test_the_payer_and_the_receiver_partition_the_paths():
    """The complement identity, which says the two decisions are one boundary read twice.

    `delta` is the only difference between them, so their gaps are exact negatives and every path
    exercises precisely one of the pair. It is also the non-vacuity check the consistency gate
    needs from outside itself: a build that exercised everything, or nothing, would satisfy
    `exercised == (gap >= 0)` and fail here.

    Killing mutation: the decision taken without the payer/receiver sign.
    """
    _, _, post, profiles, _ = read()
    payer = profiles['PAYER'][post] != 0.0
    receiver = profiles['RECEIVER'][post] != 0.0
    assert not (payer & receiver).any(), 'a path exercised both the payer and the receiver'
    assert (payer | receiver).all(), 'a path exercised neither'
    assert 0.3 < payer.mean() < 0.7, (
        'the payer exercises on {:.1%} of paths - this strike is not near the money'.format(
            payer.mean()))


# ------------------------------------------------ the gap the registration was actually handed

def test_the_registered_gap_is_the_decision_it_was_taken_on(caplog):
    """THE FIX ITSELF, and the only gate that can see it. The profile is IDENTICAL whichever number
    the registration was handed - the gap reaches only the backward pass, worth exactly zero forward
    - so the only honest way to see it without patching the library is to make the pricer SAY it.

    `register_exercise_boundary` logs `gap_disagrees` at DEBUG: scenarios where `gap >= 0` and
    `fired` differ. Zero is the contract; against a gap taken at the first post-expiry row it reads
    10 of 512 at seed 1 (59 of 4096, which is the payer's 23 down-flips plus the receiver's 36).

    512 paths rather than 4096 because this gate runs the whole engine at DEBUG.

    Killing mutation: the registration handed the first post-expiry row as its gap.
    """
    with caplog.at_level(logging.DEBUG, logger=''):
        read(paths=512)

    lines = [message for message in caplog.messages if message.startswith('SWAPTION BOUNDARY')]
    assert len(lines) == 2, (
        'expected one registration per swaption, got {}: {}'.format(len(lines), lines))
    for line in lines:
        reference, fired, total, disagrees = re.match(
            r'SWAPTION BOUNDARY (\S+) fired=(\d+)/(\d+) gap_disagrees=(\d+)', line).groups()
        assert int(disagrees) == 0, (
            '{}: the registered gap disagrees with the recorded decision on {} of {} scenarios - '
            'the flux is being weighted at a boundary the pricer never decided on'.format(
                reference, disagrees, total))
        assert 0 < int(fired) < int(total), (
            '{}: {} of {} fired, so the registration carries no boundary at all'.format(
                reference, fired, total))


# ------------------------------------------------------------- the registration is load bearing

def test_the_registration_moves_the_cva_gradient_and_not_the_cva():
    """The correction is worth EXACTLY ZERO forward and a great deal backward.

    `Boundary_AAD_Bandwidth` is the one JSON knob that reaches it: at 1e-12 the kernel underflows on
    every scenario and `boundary_weights` lands on its empty-kernel branch, contributing an exact
    zero derivative. Nothing is patched; the registration still happens and is still scored.

    MEASURED at 4096 paths, seed 1: the CVA is bit-identical at both bandwidths
    (152340.3657561677) and the gradient is not - the difference has an L2 norm 2.69x the
    suppressed gradient's, the largest bucket moving by 2294109.46 and the projection curve's 1Y
    point by a factor of 14.6.

    A full CRN ladder against a bumped adjoint is NOT taken here: this gate says the registration is
    live and correctly sourced. Whether the corrected number is right is `tests/crn_ladder.py`'s
    question.

    Killing mutation: the exercise boundary never registered.
    """
    _, _, _, _, corrected = read(bandwidth=0.01)
    _, _, _, _, suppressed = read(bandwidth=1e-12)

    assert float(corrected['Results']['cva']) == float(suppressed['Results']['cva']), (
        'the correction moved the reported CVA - it is worth exactly zero in the forward pass by '
        'construction, so this is a different number being compared')

    a = corrected['Results']['grad_cva']['Gradient']
    b = suppressed['Results']['grad_cva']['Gradient']
    assert list(a.index) == list(b.index), 'the two runs reported different gradient supports'
    delta = (a - b).values.astype(np.float64)
    scale = np.linalg.norm(b.values.astype(np.float64))
    assert scale > 0.0, 'the suppressed gradient is empty, so there is nothing to compare against'
    assert np.linalg.norm(delta) / scale > 0.1, (
        'suppressing the boundary correction moved the CVA gradient by only {:.3%} of its own '
        'norm - the registration is decorative'.format(np.linalg.norm(delta) / scale))


# ------------------------------------------------------------------------ the vol the option reads

#: The declared-defaults base date under curves sloped from 1% to 6%, the projection 20bp over the
#: discount, and swaption surfaces linear in tenor, flat in moneyness and expiry.
SLOPE_KNOTS = [0.25, 1.0, 3.0, 5.0, 10.0]
SLOPE_DISCOUNT = [0.010, 0.015, 0.030, 0.045, 0.060]
SLOPE_PROJECTION = [0.012, 0.018, 0.033, 0.048, 0.063]
VOL_TENORS = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
MID_CURVE = 'Correlation.InterestYieldVol.USD-PROJ/InterestYieldVol.USD-PROJ'


def years(day):
    return (day - book.WORLD_BASE).days / 365.0


def growth(day, zeros):
    """A curve's zero rate times its ACT_365 years to `day`, linear in rate between its knots."""
    return float(np.interp(years(day), SLOPE_KNOTS, zeros)) * years(day)


def coupons(start, end, months):
    """Coupon dates back from `end` every `months` months to `start`, as the deal's terms roll them."""
    dates = [end]
    while dates[-1] > start:
        dates.append(end - pd.DateOffset(months=months * len(dates)))
    return dates[::-1]


def sloped(distribution, vol, shift=0.0, tenors=VOL_TENORS, **factors):
    """The sloped world under the surface `vol(tenor)`, `factors` beside it."""
    world = rw.market('USD', {'USD': (SLOPE_KNOTS, SLOPE_DISCOUNT),
                              'USD-PROJ': (SLOPE_KNOTS, SLOPE_PROJECTION)}, 'USD')
    world['InterestYieldVol.USD-PROJ'] = {
        'Property_Aliases': None, 'Distribution_Type': distribution,
        'Shift': utils.Percent(100.0 * shift), 'Surface': utils.Curve([], [
            (m, e, t, vol(t)) for m in (-0.02, 0.0, 0.02) for e in (0.25, 1.0, 2.0, 5.0)
            for t in tenors])}
    return dict(world, **factors)


def payer(reference, expiry, effective, maturity, rate, months, cut=0.0):
    """A bought payer on its legs, fixed every `months` months and floating every quarter, the
    principal stepping down by `cut` at each fixed coupon, as its terms and its legs both say."""
    fixed_dates, float_dates = coupons(effective, maturity, months), coupons(effective, maturity, 3)
    principal = lambda day: 1e6 - cut * sum(start <= day for start in fixed_dates[1:-1])
    rows = lambda dates: [(a, b, principal(a), (b - a).days / 365.0) for a, b in zip(dates[:-1], dates[1:])]
    fixed = [{'Payment_Date': b, 'Notional': n, 'Rate': utils.Percent(rate), 'Accrual_Start_Date': a,
              'Accrual_End_Date': b, 'Accrual_Day_Count': 'ACT_365', 'Accrual_Year_Fraction': year,
              'Fixed_Amount': 0.0, 'Discounted': 'No', 'FX_Reset_Date': None, 'Known_FX_Rate': 0.0}
             for a, b, n, year in rows(fixed_dates)]
    floating = [{'Payment_Date': b, 'Notional': n, 'Accrual_Start_Date': a, 'Accrual_End_Date': b,
                 'Accrual_Day_Count': 'ACT_365', 'Accrual_Year_Fraction': year,
                 'Resets': [[a, a, b, year, pd.DateOffset(days=1), 'ACT_365', '0D', 0.0, 'No',
                             utils.Percent(0.0)]],
                 'Margin': utils.Basis(0.0), 'Fixed_Amount': 0.0, 'FX_Reset_Date': None,
                 'Known_FX_Rate': 0.0} for a, b, n, year in rows(float_dates)]
    amortisation = utils.DateList({day: cut for day in fixed_dates[1:-1]}) if cut else None
    return dict(swaption(reference, 'Payer'), Currency='USD', Discount_Rate='USD',
                Forecast_Rate='USD-PROJ', Forecast_Rate_Volatility='USD-PROJ',
                Option_Expiry_Date=expiry, Settlement_Date=expiry, Swap_Effective_Date=effective,
                Swap_Maturity_Date=maturity, Principal=1e6, Swap_Rate=rate,
                Pay_Frequency=pd.DateOffset(months=months), Pay_Amortisation=amortisation,
                Receive_Amortisation=amortisation, Children=[
                    rw._cashflow_leg('CFFixedInterestListDeal', reference + '_FIXED', 'USD', 'USD', 'Sell',
                                     {'Compounding': 'No', 'Items': fixed}, Calendars=None, Rate_Currency=''),
                    rw._cashflow_leg('CFFloatingInterestListDeal', reference + '_FLOAT', 'USD', 'USD', 'Buy',
                                     {'Compounding_Method': 'None', 'Averaging_Method': 'Average_Rate',
                                      'Properties': [], 'Items': floating}, Forecast_Rate='USD-PROJ')])


def priced(deal):
    """`(annuity, forward swap rate)` of a swaption's legs off the sloped curves, by hand."""
    fixed, floating = (leg['Cashflows']['Items'] for leg in deal['Children'])
    discount = lambda day: math.exp(-growth(day, SLOPE_DISCOUNT))
    annuity = sum(row['Notional'] * row['Accrual_Year_Fraction'] * discount(row['Payment_Date']) for row in fixed)
    return annuity, sum(row['Notional'] * math.expm1(
        growth(row['Accrual_End_Date'], SLOPE_PROJECTION) - growth(row['Accrual_Start_Date'], SLOPE_PROJECTION))
        * discount(row['Payment_Date']) for row in floating) / annuity


def option(distribution, forward, strike, vol, shift=0.0):
    """A payer's undiscounted value to the declared-defaults expiry: Black, displaced by `shift`, or
    Bachelier."""
    deviation = vol * math.sqrt(years(book.WORLD_EXPIRY))
    cdf = lambda x: 0.5 * math.erfc(-x / math.sqrt(2.0))
    if distribution == 'Normal':
        d = (forward - strike) / deviation
        return (forward - strike) * cdf(d) + deviation * math.exp(-0.5 * d * d) / math.sqrt(2.0 * math.pi)
    d = math.log((forward + shift) / (strike + shift)) / deviation + 0.5 * deviation
    return (forward + shift) * cdf(d) - (strike + shift) * cdf(d - deviation)


def bullet_to(deal, end):
    """The swap a swaption's legs hold at their first principal to `end`."""
    first = deal['Children'][0]['Cashflows']['Items'][0]['Notional']
    return dict(deal, Children=[dict(leg, Cashflows=dict(leg['Cashflows'], Items=[
        dict(row, Notional=first) for row in leg['Cashflows']['Items'] if row['Accrual_End_Date'] <= end]))
        for leg in deal['Children']])


def duration_tenor(deal):
    """A swaption's fixed leg's duration-equivalent tenor by hand, and the bullets' forward there:
    on the axis its stated tenor spans, the point its Macaulay duration takes among those of the
    bullets it holds at constant principal to each accrual end, and their par rates interpolated
    with the same weights."""
    rows = deal['Children'][0]['Cashflows']['Items']
    days = np.array([(row['Accrual_End_Date'] - deal['Swap_Effective_Date']).days for row in rows], dtype=float)
    axis = (deal['Swap_Maturity_Date'] - deal['Swap_Effective_Date']).days / 365.0 * (days / days[-1])
    coupon = np.array([row['Rate'].amount * row['Accrual_Year_Fraction'] for row in rows])
    discount = np.array([math.exp(-growth(row['Accrual_End_Date'], SLOPE_DISCOUNT)) for row in rows])
    principal = np.array([row['Notional'] for row in rows] + [0.0]) / rows[0]['Notional']
    bullet = lambda i: (coupon[:i + 1] + (np.arange(i + 1) == i)) * discount[:i + 1]
    knots = [(axis[:i + 1] * bullet(i)).sum() / bullet(i).sum() for i in range(len(rows))]
    leg = (principal[:-1] * coupon + principal[:-1] - principal[1:]) * discount
    duration = (axis * leg).sum() / leg.sum()
    forwards = [priced(bullet_to(deal, row['Accrual_End_Date']))[1] for row in rows]
    k = int(np.searchsorted(knots, duration))
    w = (duration - knots[k - 1]) / (knots[k] - knots[k - 1])
    return (1.0 - w) * axis[k - 1] + w * axis[k], (1.0 - w) * forwards[k - 1] + w * forwards[k]


@pytest.mark.parametrize('distribution, level, slope, shift', [
    ('Lognormal', 0.30, -0.02, 0.01), ('Normal', 0.012, -0.0004, 0.0)])
def test_an_amortising_swap_reads_its_vol_at_its_duration_equivalent_tenor(distribution, level, slope, shift):
    """A 1Y into 5Y payer, quarterly, amortising linearly to a tenth, under the sloped curves and a
    surface `level + slope x tenor` (Lognormal displaced 1%, and Normal), marks Black or Bachelier
    at the vol of its duration-equivalent tenor, 2.660 of its stated 5.005, by hand off its legs
    and the curve, to 1e-10: 39,025.65 Lognormal, that vol being the co-starting bullets' and so
    scaled by their displaced forward there, 4.53% interpolated as the tenor is, over the swap's,
    5.00%, and 38,644.28 Normal, unscaled. The same swaption not amortising, and one into 4Y9M,
    priced beside it, each mark to the bit what they mark on the surface held flat at their stated
    tenor's vol: a swap that does not amortise reads today's tenor and a scale of one exactly, which
    the surface sees, nodes a nanoyear either side of each stated tenor reading double, so that an
    ulp of tenor moves the vol by 9e-7 of itself.

    Killing mutations, Lognormal then Normal: the scale dropped, 39,499.67; the shift on one side of
    the ratio only, 40,236.77 on the bullets' and 38,217.12 on the swap's; the scale applied on the
    Normal surface, 38,281.95; the tenor unadjusted, 38,183.98 and 38,307.83; the expiry read of
    one tenor's blend cached for every tenor at that expiry (`VolSurface._interp`), the 5Y bullet
    reading the amortising swap's tenor, 97,158.56 and 96,214.32 where it marks 96,234.66 and
    96,030.50; the tenor axis spelled `stated * days / days`, an ulp off over the 4Y9M bullet's
    1,735 days, 89,935.10558 and 89,675.07538 where it marks 89,935.10284 and 89,675.07378.
    """
    expiry = book.WORLD_EXPIRY
    amortising = payer('AMORT', expiry, expiry, expiry + pd.DateOffset(years=5), 3.5, 3, cut=0.9e6 / 19)
    bullets = [payer(reference, expiry, expiry, expiry + term, 3.5, 3) for reference, term in (
        ('BULLET', pd.DateOffset(years=5)), ('BULLET_4Y9M', pd.DateOffset(years=4, months=9)))]
    stated = [(deal['Swap_Maturity_Date'] - expiry).days / 365.0 for deal in bullets]
    off = [tenor + side * 1e-9 for tenor in stated for side in (-1.0, 1.0)]
    line = lambda tenor: level + slope * tenor
    marked, _ = book.marks([amortising] + bullets, factors=sloped(
        distribution, lambda tenor: line(tenor) * (1.0 + (tenor in off)), shift,
        tenors=sorted(VOL_TENORS + stated + off)))
    annuity, forward = priced(amortising)
    tenor, at_tenor = duration_tenor(amortising)
    scale = (at_tenor + shift) / (forward + shift) if distribution == 'Lognormal' else 1.0
    by_hand = annuity * option(distribution, forward, 0.035, line(tenor) * scale, shift)
    assert float.fromhex(marked['AMORT']) == pytest.approx(by_hand, rel=1e-10)
    for deal, tenor in zip(bullets, stated):
        flat, _ = book.marks([deal], factors=sloped(distribution, lambda _: line(tenor), shift))
        assert marked[deal['Reference']] == flat[deal['Reference']], deal['Reference']


def mid_curve(distribution, deal, gap, sigma_1, sigma_2, rho, shift):
    """The vol of a forward-start swap's rate s = omega s_2 - (omega - 1) s_1 by hand, s_1 the gap's
    from the expiry to its start and s_2 the co-starting swap's to its maturity, omega that swap's
    annuity over the deal's: the variance of the difference, or under a Lognormal surface its
    displaced second moment matched."""
    annuity, rate = priced(deal)
    gap_annuity, gap_rate = priced(gap)
    omega = (annuity + gap_annuity) / annuity
    if distribution == 'Normal':
        return math.sqrt(omega ** 2 * sigma_2 ** 2 - 2.0 * omega * (omega - 1.0) * rho * sigma_1 * sigma_2
                         + (omega - 1.0) ** 2 * sigma_1 ** 2)
    t = years(book.WORLD_EXPIRY)
    s, s_1 = rate + shift, gap_rate + shift
    s_2 = (rate * annuity + gap_rate * gap_annuity) / (annuity + gap_annuity) + shift
    second = ((omega * s_2) ** 2 * math.exp(sigma_2 ** 2 * t) + ((omega - 1.0) * s_1) ** 2 * math.exp(
        sigma_1 ** 2 * t) - 2.0 * omega * (omega - 1.0) * s_1 * s_2 * math.exp(rho * sigma_1 * sigma_2 * t))
    return math.sqrt(math.log(second / s ** 2) / t)


@pytest.mark.parametrize('distribution, level, slope, shift', [
    ('Lognormal', 0.30, -0.02, 0.01), ('Normal', 0.012, -0.0004, 0.0)])
def test_a_forward_start_swap_reads_the_mid_curve_vol_of_two_co_starting_swaps(distribution, level, slope, shift):
    """A 1Y option on a 5Y swap effective a year after the expiry, semi-annual fixed at 4.5%, under
    the sloped curves and a surface `level + slope x tenor` (Lognormal displaced 1%, and Normal),
    with the mid-curve correlation stated 0.8: it marks Black or Bachelier at the vol its rate has
    as `omega` swaps from the expiry to its maturity less `omega - 1` swaps from the expiry to its
    start, the surface read at 6.005 and 1.003 from the expiry, the gap's annuity and swap rate by
    hand off its own terms' coupons, to 1e-10: 85,538.16 Lognormal and 85,168.42 Normal, omega
    1.236. With the correlation absent it reads 1, and on the surface held flat in tenor the option
    marks the plain read, Black or Bachelier at that one vol, to 1e-12.

    Killing mutations: the gap ignored, today's read at the tenor from its start to its maturity,
    86,137.27 and 85,202.32; the correlation not discovered (`config.conditional_fields`), reading
    1, 85,371.98 and 85,066.96; an absent correlation reading the quanto's 0, the flat surface's
    option at a vol of omega and omega - 1 in quadrature, 93,228.94 and 87,315.28 where the plain
    read marks 90,670.37 and 85,753.59.
    """
    expiry = book.WORLD_EXPIRY
    effective, maturity = expiry + pd.DateOffset(years=1), expiry + pd.DateOffset(years=6)
    deal = payer('FWD', expiry, effective, maturity, 4.5, 6)
    gap = payer('GAP', expiry, expiry, effective, 4.5, 6)
    annuity, rate = priced(deal)
    marked, _ = book.marks([deal], factors=sloped(
        distribution, lambda tenor: level + slope * tenor, shift, **{MID_CURVE: {'Value': 0.8}}))
    sigma_1, sigma_2 = (level + slope * (day - expiry).days / 365.0 for day in (effective, maturity))
    vol = mid_curve(distribution, deal, gap, sigma_1, sigma_2, 0.8, shift)
    assert float.fromhex(marked['FWD']) == pytest.approx(
        annuity * option(distribution, rate, 0.045, vol, shift), rel=1e-10)
    flat, _ = book.marks([deal], factors=sloped(distribution, lambda _: level, shift))
    assert float.fromhex(flat['FWD']) == pytest.approx(
        annuity * option(distribution, rate, 0.045, level, shift), rel=1e-12)


def test_the_mid_curve_correlation_reports_its_bump():
    """Under `Greeks: First` a base valuation bumps the correlation the forward-start swaption
    reads, as it does a quanto's: its one `Correlation_Bump` row is the deal marked at 0.825 less
    marked at 0.775 over 0.05, -576.50 under the Normal surface, to 1e-9.

    Killing mutation: the swaption's own block unscanned, its legs being the structure's leaves -
    no row.
    """
    expiry = book.WORLD_EXPIRY
    deal = payer('FWD', expiry, expiry + pd.DateOffset(years=1), expiry + pd.DateOffset(years=6), 4.5, 6)
    world = lambda rho: sloped('Normal', lambda tenor: 0.012 - 0.0004 * tenor, **{MID_CURVE: {'Value': rho}})
    job = book.book([deal])
    job['Calc']['Calculation']['Greeks'] = 'First'
    job['Calc']['MergeMarketData']['ExplicitMarketData']['Price Factors'].update(world(0.8))
    context = derivus.Context()
    context.load_json((json.dumps(job, cls=CustomJsonEncoder), 'bump'))
    bumped = context.run_job()[1]['Results']['Correlation_Bump']
    up, down = (float.fromhex(book.marks([deal], factors=world(rho))[0]['FWD']) for rho in (0.825, 0.775))
    assert len(bumped) == 1 and bumped['Bump'].iloc[0] == 0.025
    assert bumped['Gradient'].iloc[0] == pytest.approx((up - down) / 0.05, rel=1e-9)
