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

The rows either side of expiry are one day apart structurally: `SwaptionDeal.finalize_dates` puts
the expiry into each child's settlement dates and offsets them by a day. One day of Hull-White at 1%
absolute vol is what has to move a path across, so the fixture pays for flips in PATHS: at 4096,
seeds 1/2/3 give 33/30/31 payer flips and 28/24/30 receiver flips against 2099/2063/2078 exercises.

KILL READINGS, two earlier builds failing different gates. With the DECISION also at `Ut_swap[0]`,
the first post-expiry row is `relu(Ut_swap[0])` for a bought payer, so the flip count is
structurally zero and the consistency gate fails on 61 of 4096 paths for each swaption - the SAME
61. With the decision right and only the GAP from `Ut_swap[0]`, nothing reported moves - profile,
exposure and CVA bit-identical - and only `gap_disagrees` sees it: 9 of 512 at seed 1, 61 of 4096.
"""
import logging
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
from derivus.config import Config
from derivus.instruments import construct_instrument

import rates_world as rw

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
              'Accrual_Year_Fraction': utils.get_day_count_accrual(
                  start, (end - start).days, utils.get_day_count(DAY_COUNT)),
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
        accrual = utils.get_day_count_accrual(
            start, (end - start).days, utils.get_day_count(DAY_COUNT))
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

    c.deals = {'Attributes': {'Reference': 'test', 'Tag_Titles': ''},
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
    9 of 512 at seed 1 (61 of 4096, which is the payer's 33 down-flips plus the receiver's 28).

    512 paths rather than 4096 because this gate runs the whole engine at DEBUG.
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
    (151901.98057820834) and the gradient is not - the difference has an L2 norm 1.03x the
    suppressed gradient's, the largest bucket moving by 846334.74 and the projection curve's 1Y
    point by a factor of 5.7.

    A full CRN ladder against a bumped adjoint is NOT taken here: this gate says the registration is
    live and correctly sourced. Whether the corrected number is right is `tests/crn_ladder.py`'s
    question.
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
