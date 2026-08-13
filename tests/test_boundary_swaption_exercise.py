"""A physically-settled swaption's exercise decision, and whether its derivative reaches the tape.

`SwaptionDeal.post_process` freezes exercise on the first post-expiry row:

    Ut_mask = Ut_swap * (Ut_swap[0] >= 0)

a bool taken on SIMULATED state and broadcast over every later row. The value is right - physical
settlement really is path dependent, the holder either owns the swap for the rest of its life or
owns nothing - so the indicator must NOT be smoothed. What ordinary AAD drops is the flux: as a
factor moves, scenarios cross the exercise boundary, and an indicator has zero derivative almost
everywhere.

This matters more than the barrier sites it follows: Settlement_Style DEFAULTS to 'Physical'
(fields.py), so the frozen branch is what a book gets by omission, and NO test in this suite
exercised SwaptionDeal at all before this file.

Same two kinds of test as test_boundary_pricer_events.py. SAFETY: asking for sensitivities must not
move a reported number, bit-for-bit. ACCEPTANCE: AAD against a common-random-numbers bump ladder,
reporting agreement and flatness separately - a ladder that scatters with the bump size is
differencing across the jump rather than converging on a derivative.

Both netting shapes are measured, and the collateralised one is the larger defect rather than the
harder one - unlike the barrier's, whose collateralised ladder is still a strict xfail in
test_boundary_pricer_events. Uncorrected the reported delta is 19.96-21.46% low uncollateralised
and 156.9-167.4% low collateralised, over four seeds on the grid below.

WHAT THE ORACLE COSTS, because this file was red for five tests on exactly that. A CRN central
difference of a CVA is a difference of two Monte Carlo means divided by a bump of 1e-4, so its
variance goes like sigma^2/h^2 and the SMALL rungs are the noisy end, not the accurate one. At 1024
paths the 5e-5 rung read 9.7% away from a ladder that was otherwise flat to 1.2%, and `flatness`
did what it is for and refused to quote any of it - a NOISE FLOOR presenting as a convergence
failure. It is noise: over rungs 2e-5 to 2e-3 the flatness falls 11.22% -> 5.97% -> 8.07% -> 3.74%
-> 2.44% at 1k, 2k, 4k, 8k and 16k paths, and the outlying rung is the smallest one at every count.
So the fix is paths and a rung window that starts above the floor, never a wider tolerance: at
32768 paths over rungs 1e-4 to 2e-3 the AAD lands within 1.56% of the ladder on four seeds
uncollateralised and 2.13% collateralised, against tolerances of 3% and 4%.

MUTATIONS, all three run against the file as it stands:
    suppress `pricing.boundary_correction` entirely      -> 7 killed, the 6 structural and safety
                                                            tests correctly untouched
    strike the always-exercises control AT the money,    -> killed at 20.93%, so the control is a
        correction suppressed                               control and not a vacuous pass
    scale every requested bandwidth by 100               -> 4 of 5 killed, see that test's docstring
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
from crn_ladder import Ladder, ladder

BASE = pd.Timestamp('2024-06-28')
DTYPE = torch.float64

CURVE = 0.03            # flat continuously-compounded zero curve
SWAP_RATE = 3.05        # % - near the money, so the exercise boundary is populated
VOL = 0.20
PRINCIPAL = 10000000.0
EXPIRY = BASE + pd.DateOffset(years=1)
MATURITY = BASE + pd.DateOffset(years=6)
GRID = '0d 6y(6m)'
# 32768 paths and a rung window that starts above the noise floor - see the module docstring. Both
# backends read the same ladder: CUDA 2.213e6 at 0.70% flatness, CPU 2.221e6 at 1.02%, 0.36% apart
# against tolerances of 3%, so there is nothing left for the reduction order to spend.
BATCH, BATCHES = 4096, 8
RUNGS = (1e-4, 2e-4, 5e-4, 1e-3, 2e-3)


def _legs(rate):
    """Semi-annual fixed and floating cashflow items for the underlying swap.

    Both legs share payment and accrual dates, which makes K(t) the constant fixed rate and the
    swaption's effective strike exactly `rate` - one less moving part between the fixture and the
    thing being measured."""
    dates = [EXPIRY + pd.DateOffset(months=6 * i)
             for i in range(1 + round((MATURITY - EXPIRY).days / 182.625))]
    fixed, float_ = [], []
    for start, end in zip(dates[:-1], dates[1:]):
        accrual = (end - start).days / 365.0
        fixed.append({'Payment_Date': end, 'Accrual_Start_Date': start, 'Accrual_End_Date': end,
                      'Accrual_Year_Fraction': accrual, 'Notional': PRINCIPAL,
                      'Rate': utils.Percent(rate), 'Fixed_Amount': 0.0})
        float_.append({'Payment_Date': end, 'Accrual_Start_Date': start, 'Accrual_End_Date': end,
                       'Accrual_Year_Fraction': accrual, 'Notional': PRINCIPAL,
                       'Fixed_Amount': 0.0, 'Margin': utils.Basis(0.0),
                       'Resets': [[start, start, end, accrual, pd.DateOffset(months=6),
                                   'ACT_365', '0D', 0.0, 'No', utils.Percent(0.0)]]})
    return fixed, float_


def _swaption(rate=SWAP_RATE):
    """The parent swaption plus the two children post_process actually prices.

    post_process reads `child_map['CFFixedInterestListDeal']` and
    `child_map['CFFloatingInterestListDeal']` - the parent's own FixedCashflows/FloatCashflows are
    built by calc_dependencies and then never read - so the children are not decoration."""
    fixed, float_ = _legs(rate)
    parent = {
        'Object': 'SwaptionDeal', 'Reference': 'SWPT1', 'Currency': 'EUR',
        'Discount_Rate': 'EUR', 'Forecast_Rate': 'EUR', 'Forecast_Rate_Volatility': 'EUR',
        'Buy_Sell': 'Buy', 'Payer_Receiver': 'Payer', 'Settlement_Style': 'Physical',
        'Option_Expiry_Date': EXPIRY, 'Settlement_Date': EXPIRY,
        'Swap_Effective_Date': EXPIRY, 'Swap_Maturity_Date': MATURITY,
        'Swap_Rate': rate, 'Principal': PRINCIPAL,
        'Pay_Frequency': pd.DateOffset(months=6), 'Receive_Frequency': pd.DateOffset(months=6),
        'Index_Tenor': pd.DateOffset(months=6), 'Index_Day_Count': 'ACT_365',
        'Pay_Day_Count': 'ACT_365', 'Receive_Day_Count': 'ACT_365', 'Floating_Margin': 0.0,
        'Pay_Amortisation': None, 'Receive_Amortisation': None,
    }
    children = [
        {'Instrument': construct_instrument({
            'Object': 'CFFixedInterestListDeal', 'Reference': 'SWPT1_FIX', 'Currency': 'EUR',
            'Discount_Rate': 'EUR', 'Buy_Sell': 'Buy',
            'Cashflows': {'Compounding': 'No', 'Items': fixed}}, {})},
        {'Instrument': construct_instrument({
            'Object': 'CFFloatingInterestListDeal', 'Reference': 'SWPT1_FLT', 'Currency': 'EUR',
            'Discount_Rate': 'EUR', 'Forecast_Rate': 'EUR', 'Buy_Sell': 'Buy',
            'Cashflows': {'Compounding_Method': 'None', 'Averaging_Method': 'None',
                          'Properties': [], 'Items': float_}}, {})}]
    return {'Instrument': construct_instrument(parent, {}), 'Children': children}


def _cfg(curve=CURVE, rate=SWAP_RATE, collateralised=False):
    c = Config()
    c.params['System Parameters']['Base_Currency'] = 'USD'
    c.params['System Parameters']['Base_Date'] = BASE
    # The deal is in EUR because find_models refuses to simulate the BASE currency's curve: with
    # everything in USD the interest rate is static, nothing crosses the exercise boundary and the
    # fixture would measure a frozen indicator that never moves.
    c.params['Price Factors'] = {
        'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Priority': 1, 'Spot': 1.0},
        # Spot is deliberately NOT 1: the branch values are scored against a netting MTM in
        # REPORTING currency, so an fx factor left off them is a real error that a same-currency
        # fixture cannot see (test_the_registered_branches_reproduce_the_reported_value pins it)
        'FxRate.EUR': {'Domestic_Currency': 'USD', 'Interest_Rate': 'EUR', 'Priority': 1, 'Spot': 1.1},
        'InterestRate.USD': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                             'Curve': utils.Curve([], [[0.0, 0.03], [10.0, 0.03]])},
        'InterestRate.EUR': {'Currency': 'EUR', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                             # knots start at 0.25, not 0: HullWhite1Factor divides the simulated
                             # curve by its own tenor, so a zero-tenor knot is 0/0 and NaNs the
                             # whole netting set
                             'Curve': utils.Curve([], [[t, curve] for t in (0.25, 1.0, 3.0, 5.0)])},
        'InterestYieldVol.EUR': {
            'Property_Aliases': None,
            'Surface': utils.Curve([], [[m, e, t, VOL] for m in (-0.01, 0.0, 0.01)
                                        for e in (0.5, 1.0, 2.0) for t in (1.0, 2.0, 5.0)])},
        'SurvivalProb.CPTY': {'Recovery_Rate': 0.4,
                              'Curve': utils.Curve([], [[0.0, 0.0], [10.0, 0.4]])},
    }
    c.params['Price Models'] = {
        'HullWhite1FactorInterestRateModel.EUR': {
            'Lambda': 0.0, 'Alpha': 0.05, 'Quanto_FX_Correlation': 0.0,
            'Quanto_FX_Volatility': None, 'Sigma': utils.Curve([], [[0.0, 0.008]])}}
    c.params['Model Configuration'].append('InterestRate', (), 'HullWhite1FactorInterestRateModel')
    node = _swaption(rate)
    if collateralised:
        # A collateralised set is a DIFFERENT route to the same correction: the deal's delta
        # reaches the net through Vte AND through the balance the scan derives from it, so it goes
        # in as a gross via the chain the netting set publishes rather than being added on.
        netting = {
            'Object': 'NettingCollateralSet', 'Reference': 'NS1', 'Netted': 'True',
            'Collateralized': 'True', 'Agreement_Currency': 'USD', 'Funding_Rate': 'USD',
            'Balance_Currency': 'USD', 'Liquidation_Period': 10.0, 'Settlement_Period': 0.0,
            'Credit_Support_Amounts': {
                'Received_Threshold': utils.CreditSupportList([[0.0, 0.0]]),
                'Posted_Threshold': utils.CreditSupportList([[0.0, 0.0]]),
                'Independent_Amount': utils.CreditSupportList([[0.0, 0.0]]),
                'Minimum_Received': utils.CreditSupportList([[0.0, 0.0]]),
                'Minimum_Posted': utils.CreditSupportList([[0.0, 0.0]])}}
        node = {'Instrument': construct_instrument(netting, {}), 'Children': [node]}
    c.deals = {'Attributes': {'Reference': 'test', 'Tag_Titles': ''},
               'Deals': {'Children': [node]},
               'Calculation': {'Base_Date': BASE, 'Currency': 'USD'}}
    return c


def _run(curve=CURVE, gradient=False, batch=1024, batches=1, rate=SWAP_RATE, bandwidth=None,
         seed=1, collateralised=False):
    """One CMC run -> (netting mtm, cva, d cva / d(parallel curve shift) or None).

    The curve is the factor to bump: it moves the forward swap rate, so scenarios cross the
    exercise boundary as it shifts. AAD reports one gradient per curve knot, so the parallel-shift
    derivative the CRN ladder measures is their SUM."""
    c = _cfg(curve, rate, collateralised)
    overrides = {
        'Run_Date': BASE.strftime('%Y-%m-%d'), 'Time_grid': GRID, 'Batch_Size': batch,
        'Simulation_Batches': batches, 'Random_Seed': seed, 'Currency': 'USD', 'Tenor_Offset': 0.0,
        'Deflation_Interest_Rate': 'USD', 'Gradient_Variables': 'Factors',
        'Credit_Valuation_Adjustment': {
            'Calculate': 'Yes', 'Counterparty': 'CPTY', 'Deflate_Stochastically': 'No',
            'Stochastic_Hazard_Rates': 'No', 'Gradient': 'Yes' if gradient else 'No'}}
    if bandwidth is not None:
        overrides['Boundary_AAD_Bandwidth'] = bandwidth
    _, out = derivus.run_cmc(c, prec=DTYPE, overrides=overrides)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    grad = None
    if gradient:
        g = out['Results']['grad_cva']['Gradient']
        grad = float(sum(g.loc[i] for i in g.index if str(i[0]).startswith('InterestRate.EUR')))
    return out['Results']['mtm'].values, float(out['Results']['cva']), grad


@pytest.fixture(scope='module')
def crn_curve():
    """The uncollateralised CRN central differences, priced once.

    The bump-and-reprice side knows nothing about the AAD it will be scored against, and the
    acceptance test and every bandwidth case difference the SAME prices at the same seed - so this
    is one ladder read six times, not six ladders. Ten 32768-path valuations at 0.79s each;
    recomputing it per case would add ~40s and buy another sample of the same estimator, which is
    what the four-seed calibration in the docstrings is for and a rerun of one seed is not."""
    return ladder(price=lambda x: _run(curve=x, batch=BATCH, batches=BATCHES)[1], aad=1.0,
                  base=CURVE, rungs=RUNGS, absolute=True).crn


# ------------------------------------------------------------------ the fixture reaches the site

def test_the_fixture_actually_exercises_the_frozen_branch():
    """Before measuring anything: confirm this deal reaches the code under test.

    Three things have to hold or the fixture is testing nothing. The deal must take the PHYSICAL
    branch (Cash_Settled False); the post-expiry block must be NON-EMPTY, since `if
    Ut_swap.shape[0]` skips the frozen mask entirely when it is not; and with greeks requested the
    decision must actually have been recorded. The fourth condition - that the decision is
    genuinely uncertain - is the test below."""
    seen = {}
    import derivus.instruments as instruments
    original = instruments.SwaptionDeal.post_process

    def spy(self, accum, shared, time_grid, deal_data, child_dependencies):
        result = original(self, accum, shared, time_grid, deal_data, child_dependencies)
        tenor = deal_data.Factor_dep['Discount'][0][utils.FACTOR_INDEX_Daycount](
            deal_data.Factor_dep['Expiry'] -
            time_grid.time_grid[deal_data.Time_dep.deal_time_grid][:, utils.TIME_GRID_MTM])
        seen['cash_settled'] = deal_data.Factor_dep['Cash_Settled']
        seen['post_expiry_rows'] = int((tenor < 0.0).sum())
        seen['boundary_sets'] = len(getattr(shared, 'boundary_sets', []))
        return result

    instruments.SwaptionDeal.post_process = spy
    try:
        _run(gradient=True, batch=256)
    finally:
        instruments.SwaptionDeal.post_process = original

    assert seen['cash_settled'] is False, 'the fixture is not on the physical branch'
    assert seen['post_expiry_rows'] > 0, (
        'no reporting row falls after option expiry, so the frozen mask is never applied')
    assert seen['boundary_sets'] == 1, (
        f'post_process registered {seen["boundary_sets"]} boundary sets with greeks requested - '
        f'the exercise decision is not reaching the correction at all')


def test_the_exercise_decision_is_genuinely_uncertain():
    """A boundary correction can only recover what crosses. If every scenario exercises (or none
    does) the indicator is constant, its flux is zero and the fixture would measure nothing while
    still looking like a swaption test."""
    fired = {}
    import derivus.pricing as pricing
    original = pricing.interpolate

    def spy(mtm, shared, time_grid, deal_data, interpolate_grid=True):
        if deal_data.Instrument.field.get('Reference') == 'SWPT1':
            tenor = deal_data.Factor_dep['Discount'][0][utils.FACTOR_INDEX_Daycount](
                deal_data.Factor_dep['Expiry'] -
                time_grid.time_grid[deal_data.Time_dep.deal_time_grid][:, utils.TIME_GRID_MTM])
            post = mtm[int((tenor >= 0.0).sum()):]
            fired['share'] = float((post[0] != 0.0).to(torch.float64).mean())
        return original(mtm, shared, time_grid, deal_data, interpolate_grid)

    pricing.interpolate = spy
    try:
        _run(batch=1024)
    finally:
        pricing.interpolate = original

    assert 0.1 < fired['share'] < 0.9, (
        f'{fired["share"]:.1%} of scenarios exercise - the boundary is not populated, so this '
        f'fixture cannot see the defect')


def test_the_registered_branches_reproduce_the_reported_value():
    """The two branches, selected by the recorded flag, must be the deal's reported profile EXACTLY.

    One comparison pins three things that are individually easy to get wrong and individually
    silent: UNITS (the counterfactual is scored against a netting MTM in reporting currency, so
    `to_mtm` carries fx_rep), GRID (it goes through the same interpolate-and-pad the reported value
    did, so a row cannot slip), and SIGN (which branch is `triggered`).

    The selection happens on the PRICER grid and `to_mtm` runs on the result, which is the order
    `branch_deltas` uses - so this reads the registration exactly as the correction will.

    Silent because a boundary correction is worth exactly zero in the forward pass - none of these
    would move a reported number, only a gradient, and only by a factor that looks like ordinary
    Monte Carlo error. torch.equal, not allclose: every one of these is an exact identity."""
    import derivus.pricing as pricing
    original = pricing.interpolate
    seen = {}

    def spy(mtm, shared, time_grid, deal_data, interpolate_grid=True):
        result = original(mtm, shared, time_grid, deal_data, interpolate_grid)
        if deal_data.Instrument.field.get('Reference') == 'SWPT1':
            seen['reported'] = result
            seen['sets'] = list(shared.boundary_sets)
        return result

    pricing.interpolate = spy
    try:
        _run(gradient=True, batch=256)
    finally:
        pricing.interpolate = original

    bset, = seen['sets']
    reported = seen['reported'].detach()
    selected = bset.to_mtm(torch.where(bset.fired[0], bset.triggered, bset.untriggered))
    assert torch.equal(selected, reported), (
        'the registered branches do not reconstruct the reported deal value - the counterfactual '
        f'is being scored in the wrong units, on the wrong grid, or with the branches swapped; '
        f'max |d| {float((selected - reported).abs().max()):.6g}')


# ---------------------------------------------------------------- safety, must pass now and after

@pytest.mark.parametrize('collateralised', [False, True],
                         ids=['uncollateralised', 'collateralised'])
def test_asking_for_sensitivities_does_not_move_the_swaption_exposure(collateralised):
    """BIT-identical, not approximately. The correction is `gap - gap.detach()`, worth exactly zero
    forward, so this holds by construction - but the registration code that feeds it does not, and
    runs only when greeks are wanted, which is exactly when nobody is checking the value.

    Both netting shapes, because they are different assembly routes: uncollateralised adds the
    deal's delta to the reported MTM, collateralised pushes it through the gross-to-net chain the
    netting set publishes. Both are measured against an oracle further down."""
    kw = dict(batch=512, collateralised=collateralised)
    mtm_off, cva_off, _ = _run(**kw)
    mtm_on, cva_on, grad = _run(gradient=True, **kw)
    assert np.array_equal(mtm_off, mtm_on), 'exposure moved when sensitivities were requested'
    assert cva_off == cva_on, f'cva moved: {cva_off!r} -> {cva_on!r}'
    assert grad is not None and abs(grad) > 0.0, 'no interest rate gradient was reported at all'


def test_the_frozen_exercise_is_what_the_residual_is():
    """Attribution, so the fix is aimed at the right thing - and the fixture's noise floor, so the
    acceptance tolerance below means something.

    Struck at 1% against a 3% curve, every scenario exercises: the indicator is CONSTANT, there is
    almost no flux across the boundary, and the same machinery already agrees. On the acceptance
    grid, four seeds: 0.040-0.215% from the ladder at 0.045-0.308% flatness, with the correction
    itself worth 0.428-0.716% of the reported delta - against 19.96-21.46% at the money. That ratio
    is what makes the at-the-money reading signal rather than Monte Carlo error, and it is also
    this fixture's noise floor: the acceptance tolerance below is an order of magnitude above it.

    Run on the SAME paths as the acceptance test, because a noise floor measured on a different
    grid calibrates nothing. It stops at the 5e-4 rung: the ladder is already inside 0.31% there,
    so the two wide rungs would only be measuring this fixture's curvature. Any claimed fix has to
    close the at-the-money reading without disturbing this one."""
    kw = dict(batch=BATCH, batches=BATCHES, rate=1.0)
    aad = _run(gradient=True, **kw)[2]
    r = ladder(price=lambda x: _run(curve=x, **kw)[1], aad=aad, base=CURVE,
               rungs=RUNGS[:3], absolute=True)
    assert r.agrees(tol=0.005), f'a swaption that always exercises should already agree\n{r}'


# ------------------------------------------------------------------------------------- acceptance

def test_physical_exercise_gradient_matches_bump_and_reprice(crn_curve):
    """The frozen exercise indicator in SwaptionDeal.post_process.

    A physically settled swaption really is worth the swap or nothing from expiry on, so the jump
    is genuine product economics and must not be smoothed - the flux of scenarios across the
    exercise boundary is what has to reach the tape.

    Uncorrected this reads 19.96-21.46% LOW across four seeds, always the same sign, against a
    ladder that is already flat (0.63-2.10%) - so this defect does NOT announce itself by
    scattering the way the barrier's did, and the always-exercises control above is what separates
    it from noise. Corrected: 0.14-1.56% on the same four seeds. 3% sits between the two, 1.9x the
    worst corrected reading and 6.7x below the best uncorrected one.

    The rungs start at 1e-4 because the oracle is variance-limited below that (5e-5 read 9.7% away
    at 1024 paths) and stop at 2e-3 because the CVA's curvature in a parallel shift takes over
    above it; the reading is quoted where the ladder is flat, which is the whole point of measuring
    flatness separately."""
    aad = _run(gradient=True, batch=BATCH, batches=BATCHES)[2]
    assert abs(aad) > 1e-6, 'a swaption CVA must have an interest rate delta'
    r = Ladder(aad, CURVE, RUNGS, crn_curve)
    assert r.agrees(tol=0.03), f'{r}'


def test_collateralised_physical_exercise_gradient_matches_bump_and_reprice():
    """The same defect with collateral in the way, which is the harder half: a gross-mtm delta
    reaches the net through Vte AND through the balance the collateral scan derives from it, so a
    correction that only handles the additive path passes the test above and fails this one.

    It is also the LARGER half. Collateral removes most of the smooth exposure and leaves the
    boundary term as a much bigger share of what is left: uncorrected 156.9-167.4% low across four
    seeds against 19.96-21.46% uncollateralised. Corrected 0.16-2.13% on the same seeds, at
    1.02-5.33% flatness - the residual CVA here is ~19x smaller, so the ladder is correspondingly
    noisier and this is the one place the path count has to double again to hold still: at 32768 it
    read 3.79-6.38% flat and 0.79-3.79% from the AAD, at 16384 one seed reached 9.13% flat, which
    is the verdict about to refuse to quote. 4% is twice the worst corrected reading and a factor
    of thirty-nine below the best uncorrected one.

    It does not share `crn_curve`: this is a different portfolio at a different path count, so
    there is no ladder here to reuse."""
    kw = dict(batch=2 * BATCH, batches=BATCHES, collateralised=True)
    aad = _run(gradient=True, **kw)[2]
    r = ladder(price=lambda x: _run(curve=x, **kw)[1], aad=aad, base=CURVE,
               rungs=RUNGS, absolute=True)
    assert r.agrees(tol=0.04), f'{r}'


@pytest.mark.parametrize('bandwidth', [0.01, 0.02, 0.05, 0.1, 0.2])
def test_the_correction_holds_still_across_the_usable_bandwidth(bandwidth, crn_curve):
    """No single bandwidth can be argued for on its own, so the estimate has to hold still over a
    range of them - that is what the local-linear weights buy and the only acceptance criterion
    worth having for the kernel itself.

    Over four seeds it holds across the twenty-fold span asserted here: 0.01 reads 0.14-1.56% from
    the ladder, 0.02 0.16-0.38%, 0.05 0.00-0.22%, 0.10 0.26-0.84%, 0.20 0.59-1.60%. The flattest
    part is 0.02-0.05 and the ends are where it costs something: BELOW the asserted range the
    kernel runs out of points and the estimator turns variance-limited - 0.005 reads 0.06-2.63% and
    0.002 0.29-4.23%, and both shrink with paths, so that is noise and not the local-linear
    residual. The default of 0.01 is the bottom of the asserted range for that reason.

    This supersedes an earlier reading of +1.5%/+3.5%/+7.3% at 0.05/0.10/0.20, called estimator
    BIAS on the strength of not shrinking when the paths were quadrupled. It was measured at 1024
    paths against a ladder whose own flatness was 10.85%; at 32768 the same three bandwidths read
    0.22%, 0.84% and 1.60% at worst. The bias diagnosis was the oracle's noise.

    A gate that passes at every bandwidth is worth asking whether the knob is connected. Scaling
    every requested bandwidth by 100 kills four of the five cases and walks the disagreement
    monotonically toward the suppressed value: 1.0 -> 0.09% (the one survivor), 2.0 -> 5.48%,
    5.0 -> 13.08%, 10.0 -> 16.56%, 20.0 -> 18.45%, against 20.43% with no correction at all. So the
    knob is live, the plateau really does reach 1.0 on seed 1 (0.5 reads 1.61%), and what a
    too-wide kernel does is wash the density out until the correction stops arriving rather than
    diverge. The asserted range stops at 0.2 because that is where four seeds were measured."""
    aad = _run(gradient=True, bandwidth=bandwidth, batch=BATCH, batches=BATCHES)[2]
    r = Ladder(aad, CURVE, RUNGS, crn_curve)
    assert r.agrees(tol=0.03), f'bandwidth {bandwidth}\n{r}'
