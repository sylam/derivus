"""Acceptance criteria for extending the boundary correction from collateral to PRICER events.

Written BEFORE the change, so "done" is defined by measurement. Four sites - the discrete barrier's
already-hit latch, the autocall coupon digital, the autocall put barrier, the TARF knock-in - are
one defect: a trigger OBSERVED at a reporting row, whose value jump is real and whose flux across
the trigger is missing from the tape.

SAFETY gates: asking for sensitivities must not move a reported number. The correction is
`gap - gap.detach()`, worth exactly zero forward, so this holds by construction - but the
registration code feeding it does not, and runs only when greeks are wanted.

ACCEPTANCE gates: AAD against a common-random-numbers bump ladder.
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
from conftest import needs_hn_fused
import test_barrier_bridge as bb

MONTHLY = [bb.BASE + pd.Timedelta(days=d) for d in range(30, 366, 30)]
DISCRETE_BARRIER = dict(bb.BARRIER_DEAL, Barrier_Dates=MONTHLY)

# WHERE THE ORACLE CONVERGES, which is a property of the PAYOFF and not of the backend. A discretely
# monitored barrier is observed on a scenario row of its own, so the crossing indicator is a step in
# the spot: differencing across it does not refine as h shrinks, it changes how many paths sit on
# the wrong side of the jump. Measured, five rungs each: the uncollateralised barrier reads 6.8%
# flatness over 2e-4..5e-3 against 1.9% over 2e-3..2e-2, the collateralised one 13.8% against 1.7%,
# fva 11.2% against 3.2%. So every gate whose decision is LIVE reads the large window.
#
# The two ATTRIBUTION gates keep the small one for the opposite reason: their trigger is
# unreachable, the payoff is smooth, and the large bumps measure curvature instead - 0.37% at
# 5e-4..2e-3 against 3.45% at 2e-3..1e-2 on an AAD that is provably already right.
LIVE_RUNGS = (2e-3, 3e-3, 5e-3, 7e-3, 1e-2)
SMOOTH_RUNGS = (5e-4, 1e-3, 2e-3)

# There is no blanket CUDA skip: what failed on CPU was the small-bump window at 1024 paths, not
# precision. Two gates below carry their own NARROW preconditions instead, each measured.

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
# Both indicators saturated: a threshold at 5x spot is reached by no path in a year at 25% vol and
# a put barrier at 2x the strike by every one. The branches are still taken and the registration
# still runs, but no scenario sits near either boundary, so ordinary AAD is already the derivative.
AUTOCALL_NO_TRIGGER = _autocall(5.0, barrier=2.0)


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

# A deal contributing an mtm date NOBODY else has. With the barrier alone the deal grid IS the mtm
# grid and `Deal.calculate`'s interpolation is the identity, which is the state in which a branch
# padded at the tail happens to be right. Day 137 makes `gather_interp_matrix` insert a row in the
# MIDDLE, which is the parameter the defect lives in.
INTERPOLATING_DEAL = {
    'Object': 'EquityForwardDeal', 'Reference': 'FWD1', 'Currency': 'USD', 'Equity': 'EQ',
    'Discount_Rate': 'USD', 'Payoff_Currency': 'USD', 'Buy_Sell': 'Buy', 'Units': 1.0,
    'Forward_Price': 100.0, 'Maturity_Date': bb.BASE + pd.Timedelta(days=137)}


def _foreign_report_currency(c, ccy='EUR', vol=0.12):
    """Report in `ccy` while every deal still pays USD, so `fx_rep` is a simulated (T, B) cross
    rather than `shared.one`. Nothing else about the portfolio changes.

    THE MODEL IS WHAT MAKES IT SIMULATED. `Base_Currency` stays USD, so USD is the numeraire and
    never carries a process; the cross is stochastic only because `FxRate.EUR` does. Declared the
    other way round the factor is skipped for lacking a price model and the cross reads as a
    constant 1/1.25 at every row - which is what this fixture did while its own docstring claimed a
    (T, B) cross, and it left every row index the branches are gathered by dead.

    `vol` at zero keeps the (T, B) shape and every row gather while pinning the rows to one value,
    which is what an identity between two DIFFERENT rows needs."""
    c.params['Price Factors']['FxRate.' + ccy] = {
        'Domestic_Currency': None, 'Interest_Rate': ccy, 'Priority': 1, 'Spot': 1.25}
    c.params['Price Factors']['InterestRate.' + ccy] = {
        'Currency': ccy, 'Day_Count': 'ACT_365', 'Sub_Type': None,
        'Curve': utils.Curve([], [[0.0, 0.0], [5.0, 0.0]])}
    c.params['Price Models']['GBMAssetPriceModel.' + ccy] = {'Vol': vol, 'Drift': 0.0}
    c.params['Model Configuration'].append('FxRate', (), 'GBMAssetPriceModel')
    return ccy


def _run(deal, spot=bb.SPOT, gradient=False, batch=512, mcmc=128, collateralised=False,
         batches=1, exclude_paid_today=False, extra_deals=(), report_currency='USD',
         children=None, bandwidth=None, report_vol=0.12, seed=1):
    """One CMC run returning (netting mtm, cva, equity-spot gradient or None).

    `bandwidth` overrides the declared `Boundary_AAD_Bandwidth`; None leaves the document alone
    and every reading in this file is taken at the declared 0.01. `seed` is the document's, so a
    second draw is a second document rather than a second code path."""
    c = bb._cfg()
    c.params['Price Factors']['EquityPrice.EQ']['Spot'] = spot
    c.params['Price Factors']['SurvivalProb.CPTY'] = {
        'Recovery_Rate': 0.4, 'Curve': utils.Curve([], [[0.0, 0.0], [10.0, 0.4]])}
    if report_currency != 'USD':
        _foreign_report_currency(c, report_currency, vol=report_vol)
    kids = [{'Instrument': construct_instrument(d, {})} for d in (deal,) + tuple(extra_deals)]
    if children is not None:
        c.deals['Deals']['Children'] = children(c)
    elif collateralised:
        # Exclude_Paid_Today lives in the VALUATION CONFIGURATION - NettingCollateralSet reads it
        # from valuation_options, so putting it on the deal dict is silently ignored
        c.deals['Deals']['Children'] = [
            {'Instrument': construct_instrument(
                dict(NETTING, Reference='NS1', Collateralized='True'),
                {'NettingCollateralSet': {'Exclude_Paid_Today': exclude_paid_today}}),
             'Children': kids}]
    else:
        c.deals['Deals']['Children'] = kids
    overrides = {
        'Run_Date': bb.BASE.strftime('%Y-%m-%d'), 'Time_grid': '0d 3m(3m)', 'Batch_Size': batch,
        'Simulation_Batches': batches, 'Random_Seed': seed, 'Currency': report_currency,
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


def _spy_registration(reference, **run_kw):
    """Run, and return the deal's OWN reported profile alongside the sets registered for it.
    `pricing.interpolate` is the last thing `Deal.calculate` does, so its return value IS what the
    deal contributes - already gathered onto the MTM grid and padded - which is the only honest
    thing to compare a branch against once a second deal is on the grid."""
    import derivus.pricing as pricing
    original = pricing.interpolate
    seen = {}

    def spy(mtm, shared, time_grid, deal_data, interpolate_grid=True):
        result = original(mtm, shared, time_grid, deal_data, interpolate_grid)
        if deal_data.Instrument.field.get('Reference') == reference:
            seen['reported'] = result.detach()
            seen['sets'] = [x for x in shared.boundary_sets if isinstance(x, utils.BoundarySet)]
            seen['mtm_dates'] = time_grid.mtm_time_grid
            seen['deal_dates'] = time_grid.time_grid[
                deal_data.Time_dep.deal_time_grid][:, utils.TIME_GRID_MTM]
        return result

    pricing.interpolate = spy
    try:
        _run(gradient=True, **run_kw)
    finally:
        pricing.interpolate = original
    return seen


def _latched_reported(bset):
    """The value the registration says was reported - the latch state after every recorded decision,
    selecting between the branches through the deal's own map. Verbatim what
    `LatchedBoundarySet.branch_deltas` computes as its baseline."""
    prefix = [torch.zeros_like(bset.fired[0])]
    for flag in bset.fired:
        prefix.append(prefix[-1] | flag)
    return bset.to_mtm(torch.where(
        torch.stack(prefix)[bset.obs_before], bset.triggered, bset.untriggered))


# ------------------------------------------------- the branch has to land where the value landed

@pytest.mark.parametrize('report_currency', ['USD', 'EUR'])
@pytest.mark.parametrize('interpolated', [False, True])
def test_the_registered_barrier_branches_reproduce_the_reported_value(interpolated,
                                                                      report_currency):
    """The branches selected by the recorded flags must be the deal's reported profile EXACTLY -
    `torch.equal`, both being identities. Two defects, both invisible forward.

    GRID: `Deal.calculate` puts the pricer's `deal_time_grid` profile on the MTM grid with
    `gather_interp_matrix`, which INSERTS rows in the middle wherever another deal contributes an
    mtm date inside this deal's life. Padding the tail instead lands deal row i on mtm row i, which
    is the same row only while no such date exists - hence `interpolated` and a second deal.

    UNITS: `fx_rep` is `shared.one` only when payoff and reporting currencies match; otherwise it
    is a simulated (T, B) cross, so a branch registered without it is a delta in the wrong currency
    and leaves the fx factor's own flux off the tape - out by that cross at every row, which starts
    at the declared 1.25 USD/EUR spot and moves with the simulation from there."""
    seen = _spy_registration(
        'BARR1', deal=DISCRETE_BARRIER, batch=256, mcmc=64, report_currency=report_currency,
        extra_deals=(INTERPOLATING_DEAL,) if interpolated else ())
    bset, = seen['sets']
    selected = _latched_reported(bset)
    assert torch.equal(selected, seen['reported']), (
        'the registered branches do not reconstruct the reported deal value - the counterfactual '
        'is being scored on the wrong grid or in the wrong currency; max |d| '
        f'{float((selected - seen["reported"]).abs().max()):.6g} against a reported |mean| of '
        f'{float(seen["reported"].abs().mean()):.6g}')


# The autocall row-delta placement test that stood here was SCRAPPED: its currency statement rides
# `test_autocall_json.py::test_the_cva_spot_delta_matches_in_a_foreign_reporting_currency`, and its
# interpolated-row date-split statement for the ROW shape is not re-expressed - a deliberate loss.


def test_a_registration_does_not_hold_the_calculation_state():
    """A boundary set is held until the batch's backward pass, so what its grid map closes over is
    a memory contract no reported number can show.

    Closing over `shared` makes a cycle (shared -> boundary_sets -> the closure -> shared) that
    refcounting cannot break, so the whole calculation waits on the cyclic collector. Measured on
    the collateralised barrier at batch 1024: 19.6 GB resident after ONE run against 32 MiB, and
    the next run OOMed. Gated at the cause, because the suite only saw it as some other test
    failing in whichever file ran last."""
    seen = _spy_registration('BARR1', deal=DISCRETE_BARRIER, batch=256, mcmc=64,
                             collateralised=True)
    bset, = seen['sets']
    held = [cell.cell_contents for cell in (bset.to_mtm.__closure__ or ())]
    assert held, 'the grid map closes over nothing at all - this is not reading the real map'
    assert not any(isinstance(x, utils.Calculation_State) for x in held), (
        'the grid map closes over the calculation state, which is a reference CYCLE through '
        'shared.boundary_sets - the whole calculation survives the run')
    for name in ('untriggered', 'triggered'):
        assert not getattr(bset, name).requires_grad, f'{name} carries a graph'


def test_the_grid_map_detaches_the_fx_cross_it_captures():
    """A unit test because no fixture here can reach it: `fx_rep` is a SIMULATED cross only when the
    currencies differ AND the fx factor is stochastic, and everywhere here it resolves to a static
    rate carrying no graph - the mutant that keeps the graph survived the EUR-reported fixture.

    Live, it would pin the deal's whole tape for as long as the set exists, and `.detach()` on the
    way OUT of the map is too late for what it captured."""
    import derivus.pricing as pricing
    cross = torch.ones(3, 2, dtype=bb.DTYPE, requires_grad=True)
    to_mtm = pricing.deal_to_mtm_grid(None, None, cross)
    captured = [c.cell_contents for c in to_mtm.__closure__ if torch.is_tensor(c.cell_contents)]
    assert captured, 'the map captured no tensor at all, so the fx cross went somewhere else'
    assert not any(t.requires_grad for t in captured), (
        'the grid map captured the fx cross with its graph attached, pinning the deal tape for '
        'the life of the registration')


def test_the_netting_set_a_registration_sits_under_is_the_one_that_scores_it():
    """`boundary_sets` accumulates from every deal in every netting set, so a single slot on
    `shared` cannot say which set a registration belonged to: an UNCOLLATERALISED set's barrier was
    pushed through a collateralised set's gross-to-net chain, and with two collateralised sets the
    last to run spoke for both. Both need more than one netting set, which is why no single-set
    fixture could see either."""
    def portfolio(c):
        def netting(ref, collateralised, barrier_ref):
            return {'Instrument': construct_instrument(
                dict(NETTING, Reference=ref, Collateralized=collateralised), {}),
                'Children': [{'Instrument': construct_instrument(
                    dict(DISCRETE_BARRIER, Reference=barrier_ref), {})}]}
        return [netting('NS_UNCOL', 'False', 'B_UNCOL'),
                netting('NS_COL_A', 'True', 'B_COL_A'),
                netting('NS_COL_B', 'True', 'B_COL_B')]

    import derivus.pricing as C
    seen = {}
    original = C.boundary_correction

    def probe(shared, objective, reported_mtm, bandwidth):
        seen['chains'] = [x.net_from_gross for x in shared.boundary_sets
                          if isinstance(x, utils.BoundarySet)]
        return original(shared, objective, reported_mtm, bandwidth)

    C.boundary_correction = probe
    try:
        _run(DISCRETE_BARRIER, gradient=True, batch=256, mcmc=64, children=portfolio)
    finally:
        C.boundary_correction = original

    chains = seen.get('chains')
    assert chains is not None and len(chains) == 3, f'expected three registrations, got {chains}'
    uncollateralised, col_a, col_b = chains
    assert uncollateralised is None, (
        'a barrier in an UNCOLLATERALISED netting set was handed a gross-to-net chain, so its '
        'delta is scored through a collateral scan that never touched it')
    assert col_a is not None and col_b is not None, 'a collateralised set published no chain'
    assert col_a is not col_b, (
        'both collateralised sets are scored through the SAME chain - the last one to run is '
        'speaking for the other one as well')


MTA_LIVE = 2.0          # agreement currency, against a barrier worth ~5: transfers get suppressed

# Negative in EVERY scenario at EVERY reporting date by construction: a sold forward struck at
# ~zero is -(S - K) per unit, and 100 of them swamp anything one barrier can be worth.
DOMINATING_SHORT = {
    'Object': 'EquityForwardDeal', 'Reference': 'SHORT1', 'Currency': 'USD', 'Equity': 'EQ',
    'Discount_Rate': 'USD', 'Payoff_Currency': 'USD', 'Buy_Sell': 'Sell', 'Units': 100.0,
    'Forward_Price': 1.0, 'Maturity_Date': bb.BASE + pd.Timedelta(days=365)}


def _collateralised_barrier(c):
    """The barrier inside a collateralised set whose MTA BINDS - so the balance holds instead of
    resetting, and the transfer decision is a discontinuity of its own alongside the barrier's."""
    return [{'Instrument': construct_instrument(dict(
        NETTING, Reference='NS_COL', Collateralized='True', Credit_Support_Amounts=dict(
            NETTING['Credit_Support_Amounts'],
            Minimum_Received=utils.CreditSupportList([[0.0, MTA_LIVE]]),
            Minimum_Posted=utils.CreditSupportList([[0.0, MTA_LIVE]]))), {}),
        'Children': [{'Instrument': construct_instrument(DISCRETE_BARRIER, {})}]}]


def _with_a_second_netting_set(c):
    """The same set plus a SECOND one holding the dominating short. Nothing else moves: the short
    matures on the barrier's own expiry and needs no factor the barrier does not use, so the grid,
    the draws and the first set's numbers are what they were."""
    return _collateralised_barrier(c) + [
        {'Instrument': construct_instrument(
            dict(NETTING, Reference='NS_SHORT', Collateralized='False'), {}),
         'Children': [{'Instrument': construct_instrument(DOMINATING_SHORT, {})}]}]


def test_a_netting_set_is_scored_on_the_portfolio_not_on_itself():
    """Both mis-scoping routes at once, on the only portfolio shape that can see either: TWO netting
    sets. The objective is applied to `resolve_structure`'s root sum over every set, so a
    counterfactual scored on one SET's net is the wrong quantity - and with one set the two
    coincide, which is every other fixture here.

    The portfolio is out of the money in every scenario, so its CVA and every sensitivity of it are
    EXACTLY zero; scored on the collateralised set alone both decisions land on a positive exposure
    and report a delta to a CVA that does not exist (+3.83e-04 by the MTA route, +2.42e-04 by the
    pricer route). No tolerance is needed and none is used. The one-set companion run varies the
    parameter: same registrations, same draws, and a legitimately non-zero gradient."""
    import derivus.pricing as C
    seen = {}
    original = C.boundary_correction

    def probe(shared, objective, reported_mtm, bandwidth):
        mta = [x for x in shared.boundary_sets if isinstance(x, utils.MTABoundarySet)]
        seen['events'] = sum(len(x.events) for x in mta)
        seen['gaps'] = sum(len(x.gaps) for x in shared.boundary_sets
                           if isinstance(x, utils.BoundarySet))
        # both gaps non-positive is a call where the MTA suppressed the transfer
        by_call = {}
        for event in mta[0].events:
            by_call.setdefault(event.call_index, []).append(event.gap <= 0)
        seen['suppressed'] = any(bool((a & b).any()) for a, b in by_call.values())
        seen['set_max'] = max(float(x.replay(x.balance).max()) for x in mta)
        seen['portfolio_max'] = float(reported_mtm.detach().max())
        return original(shared, objective, reported_mtm, bandwidth)

    C.boundary_correction = probe
    try:
        _, cva, grad = _run(DISCRETE_BARRIER, gradient=True, batch=256, mcmc=64,
                            children=_with_a_second_netting_set)
    finally:
        C.boundary_correction = original

    assert seen.get('events') and seen['gaps'], (
        f'only one route registered anything ({seen}) - this fixture gates half of what it claims')
    assert seen['suppressed'], 'the MTA never suppressed a transfer, so it is not a live decision'
    assert seen['portfolio_max'] <= 0.0, (
        f'the portfolio is in the money somewhere (max {seen["portfolio_max"]:.6g}), so a non-zero '
        f'gradient below would be legitimate and this gate reads nothing')
    assert seen['set_max'] > 0.0, (
        'the collateralised set is out of the money on its own too, so scoring it in isolation '
        'would give the same answer and the defect has no room to show')
    assert cva == 0.0, f'a portfolio worth nothing to the counterparty has cva {cva!r}'
    assert grad == 0.0, (
        f'dCVA/dSpot is {grad!r} on a portfolio whose CVA is identically zero: a boundary '
        f'counterfactual is being scored on one netting set instead of on the portfolio')

    _, solo_cva, solo_grad = _run(DISCRETE_BARRIER, gradient=True, batch=256, mcmc=64,
                                  children=_collateralised_barrier)
    assert solo_cva > 0.0 and abs(solo_grad) > 0.0, (
        f'the same set ALONE reports cva {solo_cva!r} and gradient {solo_grad!r} - with nothing '
        f'for the second set to hide, the fixture must be live, or the gate above is vacuous')


# ---------------------------------------------------------------- safety, must pass now and after

@pytest.mark.parametrize('collateralised', [False, True], ids=['uncollateralised', 'collateralised'])
def test_asking_for_sensitivities_does_not_move_the_barrier_exposure(collateralised):
    """BIT-identical, not approximately: a boundary correction is worth exactly zero forward, so any
    drift means the registration path perturbed the valuation rather than observing it. Both
    netting shapes, being different code paths - the collateralised one runs the gross/net split in
    `post_process`, the uncollateralised one interpolates the accumulated deal mtm."""
    mtm_off, cva_off, _ = _run(DISCRETE_BARRIER, collateralised=collateralised)
    mtm_on, cva_on, grad = _run(DISCRETE_BARRIER, gradient=True, collateralised=collateralised)
    assert np.array_equal(mtm_off, mtm_on), 'exposure moved when sensitivities were requested'
    assert cva_off == cva_on, f'cva moved: {cva_off!r} -> {cva_on!r}'
    assert grad is not None and abs(grad) > 0.0, 'no equity gradient was reported at all'


@pytest.mark.parametrize('collateralised', [False, True], ids=['uncollateralised', 'collateralised'])
def test_asking_for_sensitivities_does_not_move_the_autocall_exposure(collateralised):
    """The autocall records both branches of its coupon trigger from ONE forward pass, the untaken
    one on a second accumulator rather than a second simulation - which is what makes this
    checkable, a re-run having consumed the random stream. BIT-identical."""
    mtm_off, cva_off, _ = _run(AUTOCALL, collateralised=collateralised)
    mtm_on, cva_on, grad = _run(AUTOCALL, gradient=True, collateralised=collateralised)
    assert np.array_equal(mtm_off, mtm_on), 'exposure moved when sensitivities were requested'
    assert cva_off == cva_on, f'cva moved: {cva_off!r} -> {cva_on!r}'
    assert grad is not None and abs(grad) > 0.0, 'no equity gradient was reported at all'


def test_the_autocall_trigger_is_what_the_residual_is():
    """ATTRIBUTION, and the control for the gates that do move. With no scenario near either
    indicator the registration still runs and costs nothing: 0.37% apart at 3.82% flatness, and
    deleting the correction repeats the AAD to every digit.

    `SMOOTH_RUNGS`, the one place in the file where that is right: with the trigger unreachable the
    payoff is smooth in spot, so the large bumps measure curvature (3.45%) on the same AAD."""
    kw = dict(batch=1024, mcmc=256, batches=16)
    aad = _run(AUTOCALL_NO_TRIGGER, gradient=True, **kw)[2]
    r = ladder(price=lambda s: _run(AUTOCALL_NO_TRIGGER, spot=s, **kw)[1], aad=aad, base=bb.SPOT,
               rungs=SMOOTH_RUNGS)
    assert r.agrees(tol=0.02), f'an unreachable trigger should already agree\n{r}'


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="the 5% CRN gate is calibrated under CUDA reduction order - CPU float32 "
                           "reads 14.05% on the same fixture, a platform reading, not a defect")
def test_autocall_coupon_digital_gradient_matches_bump_and_reprice():
    """The aligned coupon digital in `pv_MC_AutoCallSwap`. An autocall observed on its coupon date
    really has redeemed, so the jump is product economics and must not be smoothed away: what
    reaches the tape is the flux across the threshold in BOTH halves - the own-row fired/survived
    override and the carried knock-out latch killing every later row.

    0.06% apart at 65536 paths on rungs 5e-3..1e-2 flat to 5.0%. 64 batches and the top of
    `LIVE_RUNGS` for one reason: the carried latch shrank this gradient tenfold against an oracle
    whose ABSOLUTE noise did not move, so at 16 batches both estimators sit in their own noise
    (8.45% apart, rungs scattering to 31%) and the 2e-3/3e-3 rungs price out of resolution.

    Mutations, each half suppressed alone: the latch neutralised reads 84% off, and the own-row
    suppression FLIPS the sign. An aligned coupon digital is nothing BUT flux."""
    kw = dict(batch=1024, mcmc=256, batches=64)
    aad = _run(AUTOCALL, gradient=True, **kw)[2]
    r = ladder(price=lambda s: _run(AUTOCALL, spot=s, **kw)[1], aad=aad, base=bb.SPOT,
               rungs=LIVE_RUNGS[2:])
    assert r.agrees(tol=0.05), f'{r}'


# The collateralised autocall gradient gate that stood here moved to the JSON contract:
# `test_autocall_json.py::test_a_collateralised_cva_delta_carries_the_settled_coupon`
# (strict xfail, current measurements in its docstring). The missing channel is the
# counterfactual's CASH: firing pays a coupon a collateralised exposure reads through
# C_ts_te, and gross_to_net takes only an mtm delta.


def test_the_barrier_latch_is_what_the_residual_is():
    """ATTRIBUTION. With the barrier UNREACHABLE the latch never fires and the same machinery agrees
    with bump-and-reprice - 0.00% apart on a ladder flat to 0.00%, which is what a smooth payoff
    differenced under common random numbers looks like. Deleting the correction repeats the AAD to
    every digit: the registration is live and worth exactly nothing."""
    far = dict(bb.BARRIER_DEAL, Barrier_Price=1e-6,
               Barrier_Dates=list(MONTHLY))
    kw = dict(batch=1024, mcmc=256)
    aad = _run(far, gradient=True, **kw)[2]
    r = ladder(price=lambda s: _run(far, spot=s, **kw)[1], aad=aad, base=bb.SPOT,
               rungs=SMOOTH_RUNGS)
    assert r.agrees(tol=0.02), f'an unreachable barrier should already agree\n{r}'


# ------------------------------------------------------- acceptance, xfail until the change lands

def test_discrete_barrier_latch_gradient_matches_bump_and_reprice():
    """The already-hit latch in `pv_discrete_barrier_option`: a discretely monitored knock-out is
    worth nothing once it crosses, so the flux of paths across the barrier has to reach the tape.

    1.28% apart at 4096 paths on a ladder flat to 5.64%. MUTATION - the correction deleted - reads
    36.22%, a 7x margin, the correction being 24% of the reported gradient. Four batches rather
    than one because a single batch puts BOTH estimators below their own noise: a 1.9% wander
    between 1024 and 16384 paths, the same size as the residual being gated."""
    kw = dict(batch=1024, mcmc=256, batches=4)
    aad = _run(DISCRETE_BARRIER, gradient=True, **kw)[2]
    r = ladder(price=lambda s: _run(DISCRETE_BARRIER, spot=s, **kw)[1], aad=aad, base=bb.SPOT,
               rungs=LIVE_RUNGS)
    assert r.agrees(tol=0.05), f'{r}'


def test_collateralised_barrier_latch_gradient_matches_bump_and_reprice():
    """The same defect with collateral in the way, which is the harder half: a gross-mtm delta
    reaches the net through Vte AND through the balance the collateral scan produces, so a fix
    handling only the additive path passes the gate above and fails this one - which is what sent
    the gross-to-net chain into `post_process`.

    A collateralised set puts its margin-call schedule on the mtm grid - 86 mtm rows against the
    barrier's own 51, 81 interpolated - so this is where the branch profile was worst mis-mapped
    and the uncollateralised gate (17 rows, no interpolation) could not see it.

    1.08% apart at 16384 paths on a ladder flat to 1.73%; two further seeds read 0.61% and 1.06%.
    It was 6.71% before the settled ledger was declared, which is what most of that residual was.
    MUTATION - the correction deleted - reads 45.27%, 42x clear, the correction being 31.9% of the
    reported gradient. The 8% tolerance is the correction estimator's own bandwidth envelope at
    this path count, re-taken against one oracle: +1.09 / +5.45 / +1.34 / +1.24% at bandwidths
    0.01 / 0.005 / 0.02 / 0.05, so the narrowest kernel is what sizes it and not the reading."""
    kw = dict(batch=512, mcmc=128, collateralised=True, batches=32)
    aad = _run(DISCRETE_BARRIER, gradient=True, **kw)[2]
    r = ladder(price=lambda s: _run(DISCRETE_BARRIER, spot=s, **kw)[1],
               aad=aad, base=bb.SPOT, rungs=LIVE_RUNGS)
    assert r.agrees(tol=0.08), f'{r}'


def _fva(spot, gradient, batch=1024, mcmc=192, batches=16):
    """FVA and its equity-spot gradient. A funding SPREAD is what makes it non-zero: with cost,
    benefit and risk-free curves equal the adjustment is identically zero."""
    c = bb._cfg()
    c.params['Price Factors']['EquityPrice.EQ']['Spot'] = spot
    c.params['Price Factors']['SurvivalProb.CPTY'] = {
        'Recovery_Rate': 0.4, 'Curve': utils.Curve([], [[0.0, 0.0], [10.0, 0.4]])}
    c.params['Price Factors']['InterestRate.FUND'] = {
        'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
        'Curve': utils.Curve([], [[0.0, 0.02], [10.0, 0.02]])}
    c.deals['Deals']['Children'] = [{'Instrument': construct_instrument(DISCRETE_BARRIER, {})}]
    _, out = derivus.run_cmc(c, prec=bb.DTYPE, overrides={
        'Run_Date': bb.BASE.strftime('%Y-%m-%d'), 'Time_grid': '0d 3m(3m)', 'Batch_Size': batch,
        'Simulation_Batches': batches, 'Random_Seed': 1, 'Currency': 'USD', 'Tenor_Offset': 0.0,
        'MCMC_Simulations': mcmc, 'Deflation_Interest_Rate': 'USD', 'Gradient_Variables': 'Factors',
        'Funding_Valuation_Adjustment': {
            'Calculate': 'Yes', 'Funding_Cost_Interest_Curve': 'FUND',
            'Funding_Benefit_Interest_Curve': 'FUND', 'Risk_Free_Curve': 'USD',
            'Counterparty': 'CPTY', 'Gradient': 'Yes' if gradient else 'No'}})
    if not gradient:
        return float(out['Results']['fva'])
    g = out['Results']['grad_fva']['Gradient']
    return float(g.loc[[i for i in g.index if 'EquityPrice' in str(i[0])][0]])


def test_fva_gradient_carries_the_boundary_term_too():
    """FVA reads the same exposure as CVA and drops the same boundary terms - and it is the path
    that matters in production, the shipped batch job DELETING the CVA section, so a correction
    assembled only over there could never fire for it.

    0.89% apart at 16384 paths on a ladder flat to 2.18%; two further seeds read 3.61% and 3.19%.
    MUTATION - the correction deleted - reads 30.1%, a 6x margin, the correction being 24% of the
    reported gradient. 16 batches because one puts both estimators below their own noise."""
    assert _fva(bb.SPOT, False) > 0.0, 'no funding spread - the adjustment is identically zero'
    aad = _fva(bb.SPOT, gradient=True)
    r = ladder(price=lambda s: _fva(s, False), aad=aad, base=bb.SPOT, rungs=LIVE_RUNGS)
    assert r.agrees(tol=0.05), f'the fva gradient is missing its boundary term\n{r}'


def test_the_correction_generalises_to_the_other_barrier_direction():
    """The gap must be signed so gap > 0 means CROSSED, and that sign flips with the barrier
    direction - a DOWN barrier is crossed from above, an UP one from below. Backwards it still
    converges and just pulls the wrong way, so the second direction is measured.

    Up-and-IN is the variant with material exposure. The mirror images are deliberately NOT gated:
    both carry a CVA delta near -0.0003 where the CRN oracle stops converging (flatness 28% and
    82%), so a gate there would pin Monte Carlo noise.

    0.02% apart at 16384 paths on a ladder flat to 0.08%. THE MUTATION HERE IS THE SIGN, not the
    term's existence: an up-and-in call is nearly continuous at its own barrier, so the correction
    is 1.6% of the reported gradient against the knock-out's 24%, and suppression reads only 2.35%.
    The tolerance is 1% and not 2% because at 2% that mutant SURVIVED; negating the gap is killed
    at 2.86%. A 1.6% term cannot be gated to better than that on this payoff."""
    H = 110.0
    deal = dict(bb.BARRIER_DEAL, Barrier_Type='Up_And_In', Barrier_Price=H,
                Barrier_Dates=[bb.BASE + pd.Timedelta(days=d) for d in range(30, 366, 30)])
    kw = dict(batch=1024, mcmc=256, batches=16)
    aad = _run(deal, gradient=True, **kw)[2]
    r = ladder(price=lambda s: _run(deal, spot=s, **kw)[1], aad=aad, base=bb.SPOT, rungs=LIVE_RUNGS)
    assert r.agrees(tol=0.01), f'up-barrier gap sign or counterfactual is wrong\n{r}'


@needs_hn_fused
def test_the_correction_covers_heston_nandi_barriers():
    """`instruments.py` refuses the CONTINUOUS barrier variant for `SpotModel='HestonNandi'`, so
    every HN barrier routes through the discrete pricer and its already-hit latch. The registration
    sits in the shared pricer, but HN takes a different branch through `sim_spot_oss` and its
    `hit_value` for a knock-out is zeros rather than a closed form.

    1.18% apart on a ladder flat to 3.49%, on a CVA delta of 1.47 the oracle resolves cleanly.
    MUTATION - the correction deleted - reads 6.19%, a 3x margin, the correction being 4.7% of the
    reported gradient (an order under the GBM barrier's 24%, the HN counterfactual being the
    model-free zeros branch).

    Every reading is on the repaired `pricing.boundary_weights` guard: it carried a refusal that
    could never fire - a Cauchy-Schwarz ratio bounded by 1, tested against 1e-30 - and one HN
    decision solved a local-linear fit on two points 0.021 apart, returning weights +50.4/-49.5."""
    import test_hn_barrier_cmc as hb

    def run(spot, gradient):
        c = hb._cfg(True)
        c.params['Price Factors']['EquityPrice.EQ']['Spot'] = spot
        c.params['Price Factors']['SurvivalProb.CPTY'] = {
            'Recovery_Rate': 0.4, 'Curve': utils.Curve([], [[0.0, 0.0], [10.0, 0.4]])}
        _, out = derivus.run_cmc(c, prec=hb.DTYPE, overrides={
            'Run_Date': hb.BASE.strftime('%Y-%m-%d'), 'Time_grid': '0d 3m(3m)', 'Batch_Size': 512,
            'Simulation_Batches': 1, 'Random_Seed': 1, 'Currency': 'USD', 'Tenor_Offset': 0.0,
            'MCMC_Simulations': 256, 'Deflation_Interest_Rate': 'USD',
            'Gradient_Variables': 'Factors',
            'Credit_Valuation_Adjustment': {
                'Calculate': 'Yes', 'Counterparty': 'CPTY', 'Deflate_Stochastically': 'No',
                'Stochastic_Hazard_Rates': 'No', 'Gradient': 'Yes' if gradient else 'No'}})
        if not gradient:
            return float(out['Results']['cva'])
        g = out['Results']['grad_cva']['Gradient']
        return float(g.loc[[i for i in g.index if 'EquityPrice' in str(i[0])][0]])

    aad = run(100.0, True)
    r = ladder(price=lambda s: run(s, False), aad=aad, base=100.0, rungs=LIVE_RUNGS)
    assert r.agrees(tol=0.02), f'the HN barrier path is not carrying the boundary term\n{r}'


def _cumulative_gradients(**kw):
    """(reported equity-spot gradient, its cumulative `.grad` after each simulation batch).
    `SensitivitiesEstimator.__init__` IS the backward pass, so wrapping the class puts a probe
    exactly between batches - the only place the per-batch decomposition of an accumulating buffer
    is visible. `report_grad` copies off the live buffer, so snapshots do not alias."""
    import derivus.calculation as C
    original, seen = C.SensitivitiesEstimator, []

    class Spy(original):
        def __init__(self, value, params, create_graph=False):
            super().__init__(value, params, create_graph)
            g = self.report_grad()
            seen.append(next(v for k, v in g.items() if 'EquityPrice' in str(k)).item())

    C.SensitivitiesEstimator = Spy
    try:
        reported = _run(DISCRETE_BARRIER, gradient=True, **kw)[2]
    finally:
        C.SensitivitiesEstimator = original
    return reported, seen


def test_the_correction_scales_correctly_across_simulation_batches():
    """`boundary_sets` is cleared per batch while `.grad` ACCUMULATES across them and `report()`
    divides by `Simulation_Batches`, so a correction added once but averaged over N - or
    accumulated N times without averaging - makes the reported gradient scale with the batch count.

    EXACT, not a CRN ladder, which cannot see this at all: at 1024 paths the AAD reads +0.0265527
    against +0.0161766 at 16384, a 64% wander on an oracle scattering 12.8% across rungs. Three
    assertions, each killing a different mutant.

    PREFIX: batch 0 of an N-batch run draws what the 1-batch run draws (one seeded stream), so the
    per-batch increments at N=1, 2 and 4 are prefixes of one another. A correction applied only on
    `final_run` puts batch 0's N=2 increment at the uncorrected value.
    THE DIVISION: `reported == cumulative / N` exactly.
    THE SUBJECT IS LIVE: with the correction deleted, batch 0's contribution is two thirds of that
    batch's whole gradient and the same number at N=1, 2 and 4. Without this the first two hold
    with the correction suppressed - they are properties of the accumulator, not of the term.
    (Batch 0 rather than a mean: batch 2 contributes -1.3e-05, no path near the barrier.)"""
    import derivus.pricing as pricing
    kw = dict(batch=512, mcmc=192)
    live = {n: _cumulative_gradients(batches=n, **kw) for n in (1, 2, 4)}
    original = pricing.boundary_correction
    pricing.boundary_correction = lambda *a: None
    try:
        dead = {n: _cumulative_gradients(batches=n, **kw) for n in (1, 2, 4)}
    finally:
        pricing.boundary_correction = original

    for n, (reported, cumulative) in live.items():
        assert len(cumulative) == n, f'{n} batches ran {len(cumulative)} backward passes'
        assert reported == cumulative[-1] / n, (
            f'the reported gradient is {reported!r} where the gradient accumulated over {n} '
            f'batches is {cumulative[-1]!r}, i.e. {cumulative[-1] / n!r} once averaged: a term '
            f'added per batch is being reported scaled by the batch count')
    for i in range(4):
        batch_i = [np.diff([0.0] + c)[i] for _, c in live.values() if len(c) > i]
        assert len(set(batch_i)) == 1, (
            f'batch {i} contributed {batch_i} at batch counts {[n for n in live if n > i]} - the '
            f'same paths gave a different gradient, so something is per-RUN and not per-batch')
    contribution = [np.diff([0.0] + live[n][1])[0] - np.diff([0.0] + dead[n][1])[0] for n in live]
    assert len(set(contribution)) == 1 and abs(contribution[0]) > 0.0, (
        f'the boundary correction contributed {contribution} to the first batch at 1, 2 and 4 '
        f'batches - it must be the same non-zero number, or this gate is reading the accumulator '
        f'rather than the correction')


@pytest.mark.parametrize('exclude_paid_today', [False, True], ids=['plain', 'exclude_paid_today'])
def test_a_zero_gross_delta_reproduces_the_reported_net(exclude_paid_today):
    """`gross_to_net` pushes a gross-mtm delta through At -> required balance -> bands -> scan ->
    the netting arithmetic. Feed it ZERO and it must reproduce the set's own reported net EXACTLY,
    or every correction is measured against a rebased baseline and is the wrong size while still
    converging and still looking bandwidth-stable.

    It did not: `Vte` was re-derived as `g_Vt[Te]` rather than taken from the reported `b_Vte`, and
    under `Exclude_Paid_Today` the two carry DIFFERENT cashflow adjustments - measured max|diff|
    120.51 against a reported |mean| of 2.61, a 46x rebasing. Both settings are gated because with
    the option off the two forms coincide exactly and no value gate can see it either way.
    `Exclude_Paid_Today` is read from the VALUATION CONFIGURATION; on the netting dict it is
    silently ignored and this test is vacuous."""
    import derivus.pricing as C
    seen = {}
    original = C.boundary_correction

    def probe(shared, objective, reported_mtm, bandwidth):
        bset = next((x for x in shared.boundary_sets if isinstance(x, utils.BoundarySet)), None)
        if bset is not None and bset.net_from_gross is not None:
            with torch.no_grad():
                # a zero delta on the MTM grid, which is the grid the chain consumes and which only
                # `to_mtm` knows how to reach from the pricer's own rows
                seen['diff'] = float(
                    (bset.net_from_gross(bset.to_mtm(torch.zeros_like(bset.untriggered)))
                     - reported_mtm).abs().max())
        return original(shared, objective, reported_mtm, bandwidth)

    C.boundary_correction = probe
    try:
        _run(dict(DISCRETE_BARRIER, Cash_Rebate=5.0), gradient=True, collateralised=True,
             batch=256, mcmc=64, exclude_paid_today=exclude_paid_today)
    finally:
        C.boundary_correction = original

    assert 'diff' in seen, 'the collateral chain never ran - the fixture is not exercising it'
    assert seen['diff'] == 0.0, (
        f'a zero gross delta moved the net by {seen["diff"]:.4e}: the counterfactual is rebased, '
        f'so every correction against it is mis-sized')


def _spy_boundary(look, **run_kw):
    """Run with `look(shared, reported_mtm)` called at the moment the correction is assembled -
    the one place a registration, its chain and the reported portfolio all exist at once."""
    import derivus.pricing as C
    seen = {}
    original = C.boundary_correction

    def probe(shared, objective, reported_mtm, bandwidth):
        seen.update(look(shared, reported_mtm) or {})
        return original(shared, objective, reported_mtm, bandwidth)

    C.boundary_correction = probe
    try:
        _run(gradient=True, **run_kw)
    finally:
        C.boundary_correction = original
    return seen


REBATED_BARRIER = dict(DISCRETE_BARRIER, Cash_Rebate=5.0)


def test_a_run_of_held_balance_registers_its_transfer_decision_once():
    """A transfer the balance cannot make twice is ONE decision, however many calls publish it.

    Over a run of margin calls the balance is held across, `previous` is constant, so every call in
    the run yields the same counterfactual and the same jump - and the estimator was weighting one
    decision's flux by the length of the run: 81 calls inside one run on this fixture, every one of
    them live. `mark_binding_calls` keeps the call whose gap is largest, ties to the first.

    PER SIDE, which is the whole of why receive and post are not pooled: with the minimum transfer
    amount at zero the two boundaries coincide and their contributions cancel exactly, and pooling
    would keep one of the pair and invent an MTA term on a book that has no MTA.

    A COUNTING identity with no tolerance, per scenario. The second assertion is what stops it
    being vacuous: a fixture whose every call transfers has runs of one and passes by construction.
    """
    def look(shared, reported_mtm):
        mta = [b for b in shared.boundary_sets if isinstance(b, utils.MTABoundarySet)]
        assert mta, 'no MTA registration at all - this fixture gates nothing'
        worst, longest = 0, 0
        for bset in mta:
            for side in ('receive', 'post'):
                events = [e for e in bset.events if e.side == side]
                previous = torch.stack([e.previous_balance for e in events])
                # an engine declaring no mask registers every call, which IS the mutant reading -
                # taken through this gate's own assertion rather than an AttributeError
                live = torch.stack([torch.ones_like(previous[0], dtype=torch.bool)
                                    if getattr(e, 'live', None) is None else e.live
                                    for e in events])
                new_run = torch.ones_like(previous, dtype=torch.bool)
                new_run[1:] = previous[1:] != previous[:-1]
                run_id = torch.cumsum(new_run.to(torch.int64), dim=0)
                for r in range(1, int(run_id.max()) + 1):
                    in_run = run_id == r
                    worst = max(worst, int((live & in_run).sum(dim=0).max()))
                    longest = max(longest, int(in_run.sum(dim=0).max()))
        return {'live_per_run': worst, 'longest_run': longest}

    seen = _spy_boundary(look, deal=DISCRETE_BARRIER, batch=256, mcmc=64,
                         children=_collateralised_barrier)
    assert seen['longest_run'] > 1, (
        f'no run of held balance is longer than one call ({seen["longest_run"]}), so every call '
        f'transfers and this gate cannot see a repeat')
    assert seen['live_per_run'] == 1, (
        f'a run of calls over which the balance is HELD registers {seen["live_per_run"]} live '
        f'events for one transfer decision - the same counterfactual counted once per call')


def test_a_latched_registration_declares_every_settlement_in_its_reach():
    """A registration that does not name its deal's settlements is scored against the REALISED
    ledger: `net_from_gross` folds the cash that was actually paid into both branches while the
    collateral scan follows the counterfactual, and the two disagree from the settlement onward.

    So the declaration is a counting identity - every row the deal books cash at, from the first
    decision's reach onward, is named by `settles` or by `cash_events`. No tolerance. The rebated
    barrier is the sharp fixture: rows 12 20 30 36 44 54 60 68 78 84 92 94, eleven crossings each
    paying on their own date plus an expiry that is the twelfth crossing's date as well - and the
    registration that shipped declared NONE of them, first decision reaching row 7.

    ITS SCOPE IS THE BARRIER FAMILY, which is what this fixture registers. `settles` states an
    identity - the row pays out everything the deal is still worth - and a pricer settling a STREAM
    of per-fixing cashflows cannot declare through it, so the accumulator, the TARF and the
    extendable are undeclared and this gate is not run on them.

    One deal per netting set, so `shared.t_Cashflows` at this point is that deal's own ledger.
    """
    def look(shared, reported_mtm):
        booked = {int(r) for rows in (shared.t_Cashflows or {}).values() for r, v in rows.items()
                  if float(v.detach().abs().max()) > 0.0}
        rows = []
        for bset in shared.boundary_sets:
            if not isinstance(bset, utils.LatchedBoundarySet):
                continue
            first = None
            for _, on, off in bset.branch_deltas():
                live = torch.nonzero((on - off).abs().amax(dim=1) > 0)
                first = int(live.min()) if live.numel() else None
                break
            # `getattr`, so an engine that declares neither field dies on the assertion below with
            # the rows it left undeclared rather than on a missing attribute
            declared = {int(t) for t, _, _ in (getattr(bset, 'cash_events', None) or [])}
            declared |= {int(t) for t, _ in (getattr(bset, 'settles', None) or [])}
            rows.append((bset.deal, first,
                         {r for r in booked if first is not None and r >= first}, declared))
        return {'rows': rows, 'booked': booked}

    seen = _spy_boundary(look, deal=REBATED_BARRIER, collateralised=True, batch=256, mcmc=64)
    assert len(seen.get('booked', ())) > 1, (
        'the deal books at most one row, so a registration naming only its expiry would pass')
    assert seen['rows'], 'nothing latched registered at all - the fixture is not exercising this'
    for deal, first, reach, declared in seen['rows']:
        assert reach <= declared, (
            f'{deal} books cash at rows {sorted(reach - declared)} inside the reach of a decision '
            f'first landing on row {first}, and declares {sorted(declared) or "nothing"} - those '
            f'payments stay at their realised amount in every counterfactual')


@pytest.mark.parametrize('exclude_paid_today', [False, True], ids=['plain', 'exclude_paid_today'])
def test_a_ledger_row_declared_at_what_was_booked_moves_the_net_by_nothing(exclude_paid_today):
    """The sibling of the zero-gross-delta gate, on the other channel: declaring the world that
    HAPPENED must reproduce the set's own net EXACTLY.

    `cash_to_C` relu-splits received from paid before differencing, and the split of a difference
    is not the difference of the splits - so a row whose branch amount equals its booked one is the
    arithmetic's own fixed point and anything but bit-identical says the ledger channel rebases
    every correction that rides it. Both `Exclude_Paid_Today` settings, the option moving the
    window edges AND adding a Vte term of its own.
    """
    def look(shared, reported_mtm):
        for bset in shared.boundary_sets:
            if not isinstance(bset, utils.LatchedBoundarySet) or bset.net_from_gross is None:
                continue
            settles = getattr(bset, 'settles', None)
            if not settles:
                continue
            with torch.no_grad():
                zero = bset.to_mtm(torch.zeros_like(bset.untriggered))
                base = bset.net_from_gross(zero)
                same = bset.net_from_gross(zero, [(t, amount, amount) for t, amount in settles])
                return {'diff': float((same - base).abs().max()),
                        'scale': float(base.abs().max())}
        return None

    seen = _spy_boundary(look, deal=REBATED_BARRIER, collateralised=True, batch=256, mcmc=64,
                         exclude_paid_today=exclude_paid_today)
    assert 'diff' in seen, 'no registration declared a settlement under a chain - nothing gated'
    assert seen['diff'] == 0.0, (
        f'declaring the ledger as it was booked moved the net by {seen["diff"]:.4e} against a '
        f'level of {seen["scale"]:.4g}: the counterfactual is rebased through its cash')


@pytest.mark.parametrize('report_currency', ['USD', 'EUR'])
def test_a_declared_ledger_row_is_read_in_the_currency_it_was_declared_in(report_currency):
    """A registration declares its payments in the REPORTING currency; `Cf_Rec`/`Cf_Pay` cumulate
    BASE-currency amounts, each converted at its own row.

    So the identity is a unit: one reporting-currency unit of extra cash at a settlement row moves
    the reported net by one. Unscaled it moved it by 0.80 in EUR - exactly 1/1.25, the declared
    USD/EUR spot - and by 1.00 in USD, which is why no USD-reporting fixture could see it and why
    the zero-ledger gate above cannot either: at the booked amount both splits are zero whatever
    the scale is.

    THE CROSS IS SIMULATED AND FLAT, which is what makes the unit exact. Cash settled at row t is
    carried as BASE money and reported at row j's own cross, so the chain owes `fx(t) / fx(j)` and
    the unit is that ratio - MEASURED at 12% vol, 1.0183 / 1.0310 / 1.0322 / 1.0419 across the four
    report rows this payment reaches, per-path spread to 0.019. That ratio is the correct answer
    and not a statement about currency, so the gate pins the vol at zero: the cross keeps its
    (T, B) shape and every row gather runs, while the two rows carry one value.
    """
    def look(shared, reported_mtm):
        for bset in shared.boundary_sets:
            if not isinstance(bset, utils.LatchedBoundarySet) or bset.net_from_gross is None:
                continue
            settles = getattr(bset, 'settles', None)
            if not settles:
                continue
            with torch.no_grad():
                zero = bset.to_mtm(torch.zeros_like(bset.untriggered))
                t, amount = settles[-1]
                moved = bset.net_from_gross(zero, [(t, amount + torch.ones_like(amount), amount)])
                return {'unit': float((moved - bset.net_from_gross(zero)).abs().max())}
        return None

    seen = _spy_boundary(look, deal=REBATED_BARRIER, collateralised=True, batch=256, mcmc=64,
                         report_currency=report_currency, report_vol=0.0)
    assert 'unit' in seen, 'no registration declared a settlement under a chain - nothing gated'
    assert abs(seen['unit'] - 1.0) < 1e-9, (
        f'one reporting-currency unit of declared cash moved the net by {seen["unit"]:.10g}: the '
        f'ledger channel is being read in the wrong currency')


LONG_ADDING_EXPOSURE = dict(DOMINATING_SHORT, Reference='LONG1', Buy_Sell='Buy', Units=50.0,
                            Forward_Price=50.0)


def test_two_netting_sets_with_a_live_exposure_agree_with_bump_and_reprice():
    """The companion to the zero-CVA gate above, which catches mis-scoping in one DIRECTION only:
    mutating `portfolio_delta` to add the set's LEVEL rather than its CHANGE leaves the portfolio
    still <= 0, the relu still zero, and the mutant survives. So the second set here ADDS exposure,
    putting the portfolio comfortably in the money where a wrong delta must move a real number.

    Sizing it the other way does not work: a collateralised set with a binding MTA has near-zero
    exposure by design, so shrinking the short until the CVA is merely positive parks the portfolio
    ON the relu boundary, where the AAD changed SIGN across path counts while the CRN ladder stayed
    flat to 0.87%. Flatness measures the ORACLE's spread only."""
    kw = dict(batch=1024, mcmc=128, children=lambda c: _collateralised_barrier(c) + [
        {'Instrument': construct_instrument(
            dict(NETTING, Reference='NS_LONG', Collateralized='False'), {}),
         'Children': [{'Instrument': construct_instrument(LONG_ADDING_EXPOSURE, {})}]}])
    cva = _run(DISCRETE_BARRIER, **kw)[1]
    assert cva > 0.5, f'the portfolio must be COMFORTABLY in the money; cva={cva:.4f}'
    aad = _run(DISCRETE_BARRIER, gradient=True, **kw)[2]
    r = ladder(price=lambda s: _run(DISCRETE_BARRIER, spot=s, **kw)[1], aad=aad, base=bb.SPOT,
               rungs=LIVE_RUNGS)
    assert r.agrees(tol=0.05), f'two netting sets, live exposure - scoping is wrong\n{r}'

    # HONEST LIMIT: this and the zero-CVA gate measure the END-TO-END gradient, where the boundary
    # term is 2.4% of it - deleting the correction moves the reading only 2.20% -> 0.23%, so even
    # the SUPPRESSION mutant survives here. A gate that isolates the scoping needs a portfolio
    # where the correction DOMINATES the smooth sensitivity, and that one is authored rather than
    # found: `test_boundary_scoping_dominance.py`, where the term is 2.96x the smooth sensitivity
    # and the same mutant reads 381.77% off its own oracle.
