"""The boundary correction on PRICER events, end to end: the CVA and FVA gradients of documents whose
deals take a decision on simulated state.

A trigger OBSERVED at a reporting row has a real value jump and a flux across the trigger that
ordinary AAD drops; `pricing.stochastic_boundary_correction` carries it, worth exactly zero forward.
The gates are AAD against a common-random-numbers bump ladder (`crn_ladder`), and the scoping
identities a portfolio of more than one netting set makes observable.

NOT GATED HERE, each read by rebinding an engine function and gone with that: that a registration's
branches reconstruct the deal's reported profile through its own grid map and currency (a padded
tail mis-maps a row another deal inserts; a branch left in the payment currency is out by the
cross); that the map holds neither `shared` (a reference cycle that kept 19.6 GB resident) nor the
fx cross's graph; that a set's gross-to-net chain reproduces its own net at a zero delta and at the
ledger as booked (`Vte` re-derived under `Exclude_Paid_Today` rebased every correction 46x), and
reads a declared ledger row in its declared currency (0.80 per unit in EUR); that a run of held
balance registers its MTA transfer once; that each latched pricer declares every settlement in its
reach; and that the correction is per batch, `report()` averaging it once. The collateralised
ladder below runs with `Exclude_Paid_Today` off, where the zero-delta identity holds trivially.

THE GAP'S SIGN IS GATED ELSEWHERE: on an up-and-in call the correction is 1.6% of the gradient, so
the sign mutant reads 2.86% against a 1% gate at 16384 paths; `test_boundary_scoping_dominance.py`
holds both directions where the correction is the sensitivity. NOR IS A TWO-SET PORTFOLIO WITH A
LIVE EXPOSURE: the term is 2.4% of its gradient, so it meets its ladder within 5% with the
correction suppressed or scored on the set alike.
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
from derivus.instruments import construct_instrument
from crn_ladder import ladder
import test_barrier_bridge as bb

MONTHLY = [bb.BASE + pd.Timedelta(days=d) for d in range(30, 366, 30)]
DISCRETE_BARRIER = dict(bb.BARRIER_DEAL, Barrier_Dates=MONTHLY)

# WHERE THE ORACLE CONVERGES, which is a property of the PAYOFF and not of the backend. A discretely
# monitored barrier is observed on a scenario row of its own, so the crossing indicator is a step in
# the spot: differencing across it does not refine as h shrinks, it changes how many paths sit on
# the wrong side of the jump. Measured, five rungs each: the uncollateralised barrier reads 6.8%
# flatness over 2e-4..5e-3 against 1.9% over 2e-3..2e-2, the collateralised one 13.8% against 1.7%,
# fva 11.2% against 3.2%. So every gate whose decision is LIVE reads the large window.
LIVE_RUNGS = (2e-3, 3e-3, 5e-3, 7e-3, 1e-2)

QUARTERLY = [bb.BASE + pd.Timedelta(days=d) for d in (91, 182, 273, 365)]


def _autocall(threshold, barrier=0.0):
    """A quarterly autocall. Every coupon fixing lands on a reporting row - a deal's own dates are
    folded into the grid - and an ALIGNED fixing is the hard case: it is decided by the scenario's
    own spot before any inner draw, so the pricer takes the indicator branch, where a FUTURE fixing
    is a survival probability through `norm_cdf`. `Barrier` is a ratio of the strike."""
    return {
        'Object': 'QEDI_CustomAutoCallSwap', 'Reference': 'AC1', 'Currency': 'USD',
        'Payoff_Currency': 'USD', 'Equity': 'EQ', 'Dividends': 'EQ', 'Discount_Rate': 'USD',
        'Equity_Volatility': 'EQ', 'Buy_Sell': 'Buy', 'Option_Type': 'Call',
        'Strike_Price': 100.0, 'Expiry_Date': QUARTERLY[-1], 'Units': 1.0,
        'Settlement_Style': 'Cash', 'Option_On_Forward': 'No', 'Option_Style': 'European',
        'Barrier': barrier, 'Payoff_Type': None,
        'Price_Fixing': [[d, 0.0] for d in QUARTERLY],
        'Autocall_Coupons': [[d, 0.05] for d in QUARTERLY],
        'Autocall_Thresholds': [[d, threshold] for d in QUARTERLY],
        'Barrier_Dates': [d for d in QUARTERLY] if barrier else [],
        'Autocall_Floating': []}


AUTOCALL = _autocall(1.02)


NETTING = {
    'Object': 'NettingCollateralSet', 'Netted': 'True', 'Agreement_Currency': 'USD',
    'Funding_Rate': 'USD', 'Balance_Currency': 'USD', 'Liquidation_Period': 10.0,
    'Settlement_Period': 0.0,
    'Credit_Support_Amounts': {
        'Received_Threshold': utils.CreditSupportList([[0.0, 0.0]]),
        'Posted_Threshold': utils.CreditSupportList([[0.0, 0.0]]),
        'Independent_Amount': utils.CreditSupportList([[0.0, 0.0]]),
        'Minimum_Received': utils.CreditSupportList([[0.0, 0.0]]),
        'Minimum_Posted': utils.CreditSupportList([[0.0, 0.0]])}}


def _run(deal, spot=bb.SPOT, gradient=False, batch=512, mcmc=128, collateralised=False,
         batches=1, children=None, bandwidth=None, seed=1):
    """One CMC run returning (netting mtm, cva, equity-spot gradient or None).

    `bandwidth` overrides the declared `Boundary_AAD_Bandwidth`; None leaves the document alone
    and every reading in this file is taken at the declared 0.01. `seed` is the document's, so a
    second draw is a second document rather than a second code path."""
    c = bb._cfg()
    c.params['Price Factors']['EquityPrice.EQ']['Spot'] = spot
    c.params['Price Factors']['SurvivalProb.CPTY'] = {
        'Recovery_Rate': 0.4, 'Curve': utils.Curve([], [[0.0, 0.0], [10.0, 0.4]])}
    kids = [{'Instrument': construct_instrument(deal, {})}]
    if children is not None:
        c.deals['Deals']['Children'] = children(c)
    elif collateralised:
        c.deals['Deals']['Children'] = [
            {'Instrument': construct_instrument(dict(NETTING, Reference='NS1', Collateralized='True'),
                                                {}),
             'Children': kids}]
    else:
        c.deals['Deals']['Children'] = kids
    overrides = {
        'Run_Date': bb.BASE.strftime('%Y-%m-%d'), 'Time_grid': '0d 3m(3m)', 'Batch_Size': batch,
        'Simulation_Batches': batches, 'Random_Seed': seed, 'Currency': 'USD',
        'Tenor_Offset': 0.0,
        'MCMC_Simulations': mcmc, 'Deflation_Interest_Rate': 'USD', 'Generate_Cashflows': 'Yes',
        'Gradient_Variables': 'Factors',
        'Credit_Valuation_Adjustment': {
            'Calculate': 'Yes', 'Counterparty': 'CPTY', 'Deflate_Stochastically': 'No',
            'Stochastic_Hazard_Rates': 'No', 'Gradient': 'Yes' if gradient else 'No'}}
    if bandwidth is not None:
        overrides['Boundary_AAD_Bandwidth'] = bandwidth
    _, out = derivus.run_cmc(c, prec=bb.DTYPE, overrides=overrides)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()   # the OSS forks an inner MC per path; runs here are sequential
    grad = None
    if gradient:
        g = out['Results']['grad_cva']['Gradient']
        # `gradients_as_df` reports only the non-zero entries, so an absent factor is a zero
        # sensitivity - which is a reading in its own right, not a missing one
        rows = [i for i in g.index if 'EquityPrice' in str(i[0])]
        grad = float(g.loc[rows[0]]) if rows else 0.0
    return out['Results']['mtm'].values, float(out['Results']['cva']), grad


# ---------------------------------------------------------------- safety

@pytest.mark.parametrize('collateralised', [False, True], ids=['uncollateralised', 'collateralised'])
def test_asking_for_sensitivities_does_not_move_the_autocall_exposure(collateralised):
    """BIT-identical, not approximately: a boundary correction is worth exactly zero forward, so any
    drift means the registration path perturbed the valuation rather than observing it. The
    autocall records both branches of its coupon trigger from ONE forward pass, the untaken one on a
    second accumulator rather than a second simulation - which is what makes this checkable, a
    re-run having consumed the random stream. Both netting shapes, being different code paths.

    Killing mutation: the counterfactual's accumulator added to in place where it aliases the
    reported one.
    """
    mtm_off, cva_off, _ = _run(AUTOCALL, collateralised=collateralised)
    mtm_on, cva_on, grad = _run(AUTOCALL, gradient=True, collateralised=collateralised)
    assert np.array_equal(mtm_off, mtm_on), 'exposure moved when sensitivities were requested'
    assert cva_off == cva_on, f'cva moved: {cva_off!r} -> {cva_on!r}'
    assert grad is not None and abs(grad) > 0.0, 'no equity gradient was reported at all'


# ---------------------------------------------------------------- the barrier latch

def test_discrete_barrier_latch_gradient_matches_bump_and_reprice():
    """The already-hit latch in `pv_discrete_barrier_option`: a discretely monitored knock-out is
    worth nothing once it crosses, so the flux of paths across the barrier has to reach the tape.

    1.28% apart at 4096 paths on a ladder flat to 5.64%; the correction is 24% of the reported
    gradient. Four batches rather than one because a single batch puts BOTH estimators below their
    own noise: a 1.9% wander between 1024 and 16384 paths, the same size as the residual gated.

    Killing mutation: the barrier's latch never registered - 36.22%, a 7x margin.
    """
    kw = dict(batch=1024, mcmc=256, batches=4)
    aad = _run(DISCRETE_BARRIER, gradient=True, **kw)[2]
    r = ladder(price=lambda s: _run(DISCRETE_BARRIER, spot=s, **kw)[1], aad=aad, base=bb.SPOT,
               rungs=LIVE_RUNGS)
    assert r.agrees(tol=0.05), f'{r}'


COLLATERALISED_RUNGS = (2e-3, 5e-3, 1e-2)


def test_collateralised_barrier_latch_gradient_matches_bump_and_reprice():
    """The same defect with collateral in the way, which is the harder half: a gross-mtm delta
    reaches the net through Vte AND through the balance the collateral scan produces, so a fix
    handling only the additive path passes the gate above and fails this one - which is what sent
    the gross-to-net chain into `post_process`. A collateralised set puts its margin-call schedule
    on the mtm grid - 86 mtm rows against the barrier's own 51, 81 interpolated - so this is where
    the branch profile was worst mis-mapped.

    At 4096 paths on five rungs: 3.23% apart, flat to 3.50%; seeds 2 and 3 read 5.57% and 2.56%,
    16384 paths 1.08% / 0.61% / 1.06% - 6.71% before the settled ledger was declared - against a
    bandwidth envelope of +1.09 / +5.45 / +1.34 / +1.24% at 0.01 / 0.005 / 0.02 / 0.05. On the three
    rungs run here: 3.54% flat to 3.67%, 4.83% at seed 2; at 2048 paths seed 2 read 12.81%, which is
    what keeps the 4096.

    Killing mutation: the barrier's latch never registered - 52.24% at 4096 paths.
    """
    kw = dict(batch=512, mcmc=128, collateralised=True, batches=8)
    aad = _run(DISCRETE_BARRIER, gradient=True, **kw)[2]
    r = ladder(price=lambda s: _run(DISCRETE_BARRIER, spot=s, **kw)[1],
               aad=aad, base=bb.SPOT, rungs=COLLATERALISED_RUNGS)
    assert r.agrees(tol=0.08), f'{r}'


def _fva(spot, gradient, batch=1024, mcmc=192, batches=4, benefit='FUND'):
    """`(fva, its equity-spot gradient or None)`. A funding SPREAD is what makes it non-zero: with
    cost, benefit and risk-free curves equal the adjustment is identically zero."""
    c = bb._cfg()
    c.params['Price Factors']['EquityPrice.EQ']['Spot'] = spot
    c.params['Price Factors']['SurvivalProb.CPTY'] = {
        'Recovery_Rate': 0.4, 'Curve': utils.Curve([], [[0.0, 0.0], [10.0, 0.4]])}
    for name in ('FUND', benefit):
        c.params['Price Factors']['InterestRate.' + name] = {
            'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
            'Curve': utils.Curve([], [[0.0, 0.02], [10.0, 0.02]])}
    c.deals['Deals']['Children'] = [{'Instrument': construct_instrument(DISCRETE_BARRIER, {})}]
    _, out = derivus.run_cmc(c, prec=bb.DTYPE, overrides={
        'Run_Date': bb.BASE.strftime('%Y-%m-%d'), 'Time_grid': '0d 3m(3m)', 'Batch_Size': batch,
        'Simulation_Batches': batches, 'Random_Seed': 1, 'Currency': 'USD', 'Tenor_Offset': 0.0,
        'MCMC_Simulations': mcmc, 'Deflation_Interest_Rate': 'USD', 'Gradient_Variables': 'Factors',
        'Funding_Valuation_Adjustment': {
            'Calculate': 'Yes', 'Funding_Cost_Interest_Curve': 'FUND',
            'Funding_Benefit_Interest_Curve': benefit, 'Risk_Free_Curve': 'USD',
            'Counterparty': 'CPTY', 'Gradient': 'Yes' if gradient else 'No'}})
    fva = float(out['Results']['fva'])
    if not gradient:
        return fva, None
    g = out['Results']['grad_fva']['Gradient']
    return fva, float(g.loc[[i for i in g.index if 'EquityPrice' in str(i[0])][0]])


FVA_RUNGS = (2e-3, 5e-3, 1e-2)


def test_fva_gradient_carries_the_boundary_term_too():
    """FVA reads the same exposure as CVA and drops the same boundary terms - and it is the path
    that matters in production, the shipped batch job DELETING the CVA section, so a correction
    assembled only over there could never fire for it.

    0.89% apart at 16384 paths on five rungs flat to 2.18%, two further seeds 3.61% and 3.19%; at
    the 4096 paths and three rungs run here, 1.37% flat to 5.81%. Four batches, because one puts
    both estimators below their own noise.

    Killing mutation: the barrier's latch never registered - 30.1%, a 6x margin, the correction
    being 24% of the reported gradient.
    """
    fva, aad = _fva(bb.SPOT, gradient=True)
    assert fva > 0.0, 'no funding spread - the adjustment is identically zero'
    r = ladder(price=lambda s: _fva(s, False)[0], aad=aad, base=bb.SPOT, rungs=FVA_RUNGS)
    assert r.agrees(tol=0.05), f'the fva gradient is missing its boundary term\n{r}'


def test_a_funding_benefit_curve_no_deal_reads_is_a_dependency_of_the_run():
    """The dependency walk registers the benefit curve as it does the cost curve: named apart from
    it and carrying the same values, it gives the same FVA to the bit.

    Killing mutation: the walk registering the cost curve alone - `Cannot find InterestRate.BEN`.
    """
    assert _fva(bb.SPOT, False, batches=1, benefit='BEN') == _fva(bb.SPOT, False, batches=1)


# ---------------------------------------------------------------- scoping: two netting sets

MTA_LIVE = 2.0          # agreement currency, against a barrier worth ~5: transfers get suppressed

# Negative in EVERY scenario at EVERY reporting date by construction: a sold forward struck at
# ~zero is -(S - K) per unit, and 100 of them swamp anything one barrier can be worth - and it
# outlives the collateralised set, which reports its closeout ten days past the barrier's expiry.
DOMINATING_SHORT = {
    'Object': 'EquityForwardDeal', 'Reference': 'SHORT1', 'Currency': 'USD', 'Equity': 'EQ',
    'Discount_Rate': 'USD', 'Payoff_Currency': 'USD', 'Buy_Sell': 'Sell', 'Units': 100.0,
    'Forward_Price': 1.0, 'Maturity_Date': bb.BASE + pd.Timedelta(days=395)}


def _collateralised_barrier(c):
    """The barrier inside a collateralised set whose MTA BINDS - so the balance holds instead of
    resetting, and the transfer decision is a discontinuity of its own alongside the barrier's."""
    return [{'Instrument': construct_instrument(dict(
        NETTING, Reference='NS_COL', Collateralized='True', Credit_Support_Amounts=dict(
            NETTING['Credit_Support_Amounts'],
            Minimum_Received=utils.CreditSupportList([[0.0, MTA_LIVE]]),
            Minimum_Posted=utils.CreditSupportList([[0.0, MTA_LIVE]]))), {}),
        'Children': [{'Instrument': construct_instrument(DISCRETE_BARRIER, {})}]}]


def _beside(deal):
    """The collateralised barrier and a SECOND, uncollateralised set holding `deal`."""
    return lambda c: _collateralised_barrier(c) + [
        {'Instrument': construct_instrument(
            dict(NETTING, Reference='NS_' + deal['Reference'], Collateralized='False'), {}),
         'Children': [{'Instrument': construct_instrument(deal, {})}]}]


def test_a_netting_set_is_scored_on_the_portfolio_not_on_itself():
    """The objective is applied to `resolve_structure`'s root sum over every netting set, so a
    counterfactual scored on one SET's net is the wrong quantity - and with one set the two
    coincide, which is every other fixture here. TWO sets: the collateralised barrier, whose MTA
    binds, beside a dominating short.

    The portfolio is out of the money in every scenario, so its CVA and every sensitivity of it are
    EXACTLY zero; scored on the collateralised set alone both decisions land on a positive exposure
    and report a delta to a CVA that does not exist (+3.83e-04 by the MTA route, +2.42e-04 by the
    pricer route). The same set ALONE, same registrations and draws, has a CVA and a gradient - so
    the zero is the scoping and not a dead fixture.

    Killing mutation: the counterfactual scored on the set's own net (the objective applied to the
    set's level).
    """
    _, cva, grad = _run(DISCRETE_BARRIER, gradient=True, batch=256, mcmc=64,
                        children=_beside(DOMINATING_SHORT))
    assert cva == 0.0, f'a portfolio worth nothing to the counterparty has cva {cva!r}'
    assert grad == 0.0, (
        f'dCVA/dSpot is {grad!r} on a portfolio whose CVA is identically zero: a boundary '
        f'counterfactual is being scored on one netting set instead of on the portfolio')
    _, solo_cva, solo_grad = _run(DISCRETE_BARRIER, gradient=True, batch=256, mcmc=64,
                                  children=_collateralised_barrier)
    assert solo_cva > 0.0 and abs(solo_grad) > 0.0, (solo_cva, solo_grad)
