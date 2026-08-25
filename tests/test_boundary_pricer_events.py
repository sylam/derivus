"""Acceptance criteria for extending the boundary correction from collateral to PRICER events.

Written BEFORE the change, so "done" is defined by measurement rather than by the change looking
plausible. Four sites - the discrete barrier's already-hit latch, the autocall coupon digital, the
autocall put barrier, the TARF knock-in - are all the same defect: a trigger OBSERVED at a
reporting row, whose value jump is real and whose flux across the trigger is missing from the tape.

Two kinds of test here, and the distinction matters.

SAFETY (must pass now and after): asking for sensitivities must not move a reported number. The
correction is `gap - gap.detach()`, worth exactly zero in the forward pass, so this holds by
construction - which is precisely the sort of claim worth pinning, because the registration code
that feeds it does NOT hold by construction and runs only when greeks are wanted.

ACCEPTANCE (xfail now, must pass after): AAD against a common-random-numbers bump ladder. Marked
strict, so they turn the suite RED the moment they start passing and the marker has to come off -
an xfail left lying around after the fix would quietly stop being a gate.
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

# WHERE THE ORACLE CONVERGES, which is a property of the PAYOFF and not of the backend.
#
# A discretely monitored barrier is now observed on a scenario row of its own - `Dynamic_Scenario_
# Dates` defaults to 'Yes' - so the crossing indicator is a step in the spot rather than a convex
# combination of two neighbouring rows. Differencing across it does not refine as h shrinks: it
# changes how many paths sit on the wrong side of the jump. Measured on the declared default, five
# rungs each, at the path counts these gates now run: the uncollateralised barrier reads 6.8%
# flatness over 2e-4..5e-3 and 1.9% over 2e-3..2e-2; the collateralised one 13.8% against 1.7%; fva
# 11.2% against 3.2%. Same runs, same seeds - only the window moves. So every gate whose decision is
# LIVE reads the large window.
#
# The two ATTRIBUTION gates are the exception and for the opposite reason: their trigger is
# unreachable, the payoff is smooth, and the large bumps start measuring curvature instead - the
# unreachable autocall reads 0.37% at 5e-4..2e-3 and 3.45% at 2e-3..1e-2 against an AAD that is
# provably already right (its correction is 0.00% of the gradient). They keep the small window.
LIVE_RUNGS = (2e-3, 3e-3, 5e-3, 7e-3, 1e-2)
SMOOTH_RUNGS = (5e-4, 1e-3, 2e-3)

# Nothing here is CUDA-only any more. The skip this replaces blamed "float32 differencing off CUDA",
# but `bb.DTYPE` is float64 and what actually failed on CPU was the small-bump window above at 1024
# paths. On the windows and path counts below, the three gates that carried the marker pass on CPU
# too: 2.00% / 3.95% / 0.01% disagreement, against readings of 34.92% / 46.39% / 34.62% with the
# correction suppressed.

QUARTERLY = [bb.BASE + pd.Timedelta(days=d) for d in (91, 182, 273, 365)]


def _autocall(threshold, barrier=0.0):
    """A quarterly autocall. Every coupon fixing lands on a reporting row - which is not a choice:
    a deal's own dates are folded into the grid - and an ALIGNED fixing is exactly the hard case.
    It is decided by the scenario's own spot, before any inner draw has advanced it, so the pricer
    takes the indicator branch; every FUTURE fixing is a survival probability through norm_cdf and
    was differentiable all along. `Barrier` is a ratio of the strike."""
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
# Both of this deal's indicators saturated: a threshold at 5x spot is reached by no path in a year
# at 25% vol, and a put barrier at 2x the strike is breached by every one. The branches are still
# taken - the registration still runs - but no scenario sits near either boundary, so there is no
# flux to recover and ordinary AAD is already the derivative of the reported value.
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

# A deal that contributes an mtm date NOBODY else has. The barrier's own reval dates are the 3m
# reporting grid plus its monitoring dates, so with it alone the deal grid IS the mtm grid and the
# interpolation `Deal.calculate` performs is the identity - the state in which a branch registered
# on the deal grid and padded at the tail happens to be right. Day 137 is the parameter the defect
# lives in: it makes `gather_interp_matrix` insert a row in the MIDDLE.
INTERPOLATING_DEAL = {
    'Object': 'EquityForwardDeal', 'Reference': 'FWD1', 'Currency': 'USD', 'Equity': 'EQ',
    'Discount_Rate': 'USD', 'Payoff_Currency': 'USD', 'Buy_Sell': 'Buy', 'Units': 1.0,
    'Forward_Price': 100.0, 'Maturity_Date': bb.BASE + pd.Timedelta(days=137)}


def _foreign_report_currency(c, ccy='EUR'):
    """Report in `ccy` while every deal still pays USD, so `fx_rep` is a simulated (T, B) cross
    rather than `shared.one`. Nothing else about the portfolio changes."""
    c.params['Price Factors']['FxRate.' + ccy] = {
        'Domestic_Currency': None, 'Interest_Rate': ccy, 'Priority': 1, 'Spot': 1.25}
    c.params['Price Factors']['FxRate.USD']['Domestic_Currency'] = ccy
    c.params['Price Factors']['InterestRate.' + ccy] = {
        'Currency': ccy, 'Day_Count': 'ACT_365', 'Sub_Type': None,
        'Curve': utils.Curve([], [[0.0, 0.0], [5.0, 0.0]])}
    c.params['Price Models']['GBMAssetPriceModel.USD'] = {'Vol': 0.12, 'Drift': 0.0}
    c.params['Model Configuration'].append('FxRate', (), 'GBMAssetPriceModel')
    return ccy


def _run(deal, spot=bb.SPOT, gradient=False, batch=512, mcmc=128, collateralised=False,
         batches=1, exclude_paid_today=False, extra_deals=(), report_currency='USD',
         children=None):
    """One CMC run returning (netting mtm, cva, equity-spot gradient or None)."""
    c = bb._cfg()
    c.params['Price Factors']['EquityPrice.EQ']['Spot'] = spot
    c.params['Price Factors']['SurvivalProb.CPTY'] = {
        'Recovery_Rate': 0.4, 'Curve': utils.Curve([], [[0.0, 0.0], [10.0, 0.4]])}
    if report_currency != 'USD':
        _foreign_report_currency(c, report_currency)
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
    _, out = derivus.run_cmc(c, prec=bb.DTYPE, overrides={
        'Run_Date': bb.BASE.strftime('%Y-%m-%d'), 'Time_grid': '0d 3m(3m)', 'Batch_Size': batch,
        'Simulation_Batches': batches, 'Random_Seed': 1, 'Currency': report_currency,
        'Tenor_Offset': 0.0,
        'MCMC_Simulations': mcmc, 'Deflation_Interest_Rate': 'USD', 'Generate_Cashflows': 'Yes',
        'Gradient_Variables': 'Factors',
        'Credit_Valuation_Adjustment': {
            'Calculate': 'Yes', 'Counterparty': 'CPTY', 'Deflate_Stochastically': 'No',
            'Stochastic_Hazard_Rates': 'No', 'Gradient': 'Yes' if gradient else 'No'}})
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
    deal contributes to the netting MTM - already gathered onto the MTM grid and already padded.
    That makes it the only honest thing to compare a branch against: the netting mtm is a SUM, so
    once a second deal is on the grid it stops being this deal's value at all."""
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
    """The value the registration says was reported: the latch state after every recorded decision,
    selecting between the branches, through the deal's own map. This is verbatim what
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
    """The branches, selected by the recorded flags, must be the deal's reported profile EXACTLY.

    Both defects this pins are invisible in a forward pass - a boundary correction is worth zero
    there - so only a gradient moves, and only by a factor that reads as Monte Carlo error.

    GRID. The pricer builds its profile over `deal_time_grid`; `Deal.calculate` puts it on the MTM
    grid with `gather_interp_matrix`, which INSERTS rows in the middle wherever another deal
    contributes an mtm date inside this deal's life. Padding the tail instead lands deal row i on
    mtm row i, which is the same row only while no such date exists - hence the `interpolated`
    parameter, and hence a fixture with a second deal in it. Measured on this one: mtm grid
    [0 30 60 90 92 120 137 150 180 183 210 240 270 273 300 330 360 365], the barrier's own rows
    everything but 137, so every branch value from day 150 on sat one row early and the expiry row
    was left as the zero pad.

    UNITS. `fx_rep` is `shared.one` only when the payoff and reporting currencies match; otherwise
    it is a simulated (T, B) cross, so a branch registered without it is a delta in the wrong
    currency AND leaves the fx factor's own flux off the tape. Measured: the branch was exactly
    0.8x the reported value, the USD/EUR spot.

    torch.equal, not allclose: both are exact identities."""
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


# The autocall row-delta placement test that stood here was SCRAPPED, not converted: it captured
# the registration through a `_spy_registration` monkey-patch of `pricing.interpolate`, which the
# test architecture forbids, and it broke on arity the day the autocall correctly registered two
# sets. Its currency statement now rides `test_autocall_json.py::
# test_the_cva_spot_delta_matches_in_a_foreign_reporting_currency` through the contract; its
# interpolated-row date-split statement for the ROW shape is NOT re-expressed - the composite
# gradient gates run with deal dates on mtm rows - and that loss is deliberate and recorded.


def test_a_registration_does_not_hold_the_calculation_state():
    """A boundary set outlives the pricing call - it is held until the batch's backward pass - so
    what its grid map closes over is a memory contract, and no reported number can show you when
    it is wrong.

    Closing over `shared` makes a cycle: shared -> boundary_sets -> the closure -> shared.
    Refcounting cannot break it, so the calculation state and everything reachable from it
    survives the run and waits on the cyclic collector. MEASURED on the collateralised barrier at
    batch 1024: 19.6 GB still resident after ONE run where the same run had held 32 MiB before,
    and the next run OOMed. The suite only ever saw it as some other test failing, in whichever
    file happened to run last, which is why it is gated at the cause and not at a byte count.

    `interp_to_mtm_grid` never used `shared` either, so the fix removed a parameter rather than
    working around one."""
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
    """The other half, and a unit test because no fixture in this file can reach it: `fx_rep` is a
    SIMULATED (T, B) cross only when the payoff and reporting currencies differ AND the fx factor
    is stochastic; everywhere here it resolves to a static rate, which carries no graph, so an
    integration gate cannot tell a detached capture from a live one. It could not - the mutant
    that keeps the graph survived the EUR-reported fixture.

    Live, it would pin the deal's whole tape for as long as the set exists. Branch values are
    coefficients: the rule that they stay detached is a memory contract as much as a correctness
    one, and `.detach()` on the way OUT of the map is too late for the thing it captured."""
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
    `shared` cannot say which set a given registration belonged to. It was one slot: an
    UNCOLLATERALISED set's barrier was pushed through a collateralised set's gross-to-net chain,
    and with two collateralised sets the last one to run spoke for both.

    Both failure modes need a portfolio with more than one netting set in it, which is why no
    single-set fixture could see either. Measured on this one: before, one chain object served all
    three registrations; after, the uncollateralised set's barrier carries None and the two
    collateralised sets carry two DIFFERENT chains."""
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

# Negative in EVERY scenario at EVERY reporting date, by construction: a sold forward struck at
# ~zero is -(S - K) per unit, and 100 of them swamp anything one barrier can be worth, a call being
# bounded by the spot itself. So the PORTFOLIO is out of the money wherever the collateralised set
# on its own is not.
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
    """The same set, plus a SECOND one holding the dominating short. Nothing else moves: the short
    matures on the barrier's own expiry and needs no factor the barrier does not already use, so
    the grid, the draws and the first set's own numbers are what they were."""
    return _collateralised_barrier(c) + [
        {'Instrument': construct_instrument(
            dict(NETTING, Reference='NS_SHORT', Collateralized='False'), {}),
         'Children': [{'Instrument': construct_instrument(DOMINATING_SHORT, {})}]}]


def test_a_netting_set_is_scored_on_the_portfolio_not_on_itself():
    """Both routes at once, on the only portfolio shape that can see either: TWO netting sets.

    The objective is applied by Credit_Monte_Carlo to `resolve_structure`'s root sum over every
    netting set. A counterfactual scored on one SET's own net is therefore the wrong quantity, and
    with exactly one set in the portfolio the two coincide - which is every other fixture here, and
    why the suite could not see it. The MTA route scored `objective(replay(...))`, the set's own
    level; the collateral chain returned the same thing until 01b038c.

    Here the second set makes them disagree as far as they can. The portfolio is out of the money
    in every scenario, so its CVA is EXACTLY zero and so is every sensitivity of it - there is no
    exposure for a barrier crossing or a margin call to move. Scored on the collateralised set
    alone, both decisions land on a positive exposure and report a delta to a CVA that does not
    exist. Measured on this fixture, against a true zero: the MTA route mis-scoped reads
    dCVA/dSpot +3.83e-04, the pricer route mis-scoped (the pre-01b038c chain scoring) +2.42e-04,
    the two together +6.24e-04, and with the boundary term suppressed entirely 0.0 - so this reads
    the correction and nothing else. No tolerance is needed and none is used.

    The one-set companion run is the parameter this varies: the SAME registrations, the same draws,
    the second set removed, and a gradient that is legitimately non-zero."""
    import derivus.pricing as C
    seen = {}
    original = C.boundary_correction

    def probe(shared, objective, reported_mtm, bandwidth):
        mta = [x for x in shared.boundary_sets if isinstance(x, utils.MTABoundarySet)]
        seen['events'] = sum(len(x.events) for x in mta)
        seen['gaps'] = sum(len(x.gaps) for x in shared.boundary_sets
                           if isinstance(x, utils.BoundarySet))
        # a call where BOTH gaps are non-positive is one where the MTA suppressed the transfer
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
    """BIT-identical, not approximately - a boundary correction is worth exactly zero forward, so
    any drift means the registration path perturbed the valuation rather than observed it.

    Both netting shapes, because they are different code paths: the collateralised branch runs the
    gross/net split in post_process, the uncollateralised one returns an interpolation of the
    accumulated deal mtm with no split at all."""
    mtm_off, cva_off, _ = _run(DISCRETE_BARRIER, collateralised=collateralised)
    mtm_on, cva_on, grad = _run(DISCRETE_BARRIER, gradient=True, collateralised=collateralised)
    assert np.array_equal(mtm_off, mtm_on), 'exposure moved when sensitivities were requested'
    assert cva_off == cva_on, f'cva moved: {cva_off!r} -> {cva_on!r}'
    assert grad is not None and abs(grad) > 0.0, 'no equity gradient was reported at all'


@pytest.mark.parametrize('collateralised', [False, True], ids=['uncollateralised', 'collateralised'])
def test_asking_for_sensitivities_does_not_move_the_autocall_exposure(collateralised):
    """The autocall records both branches of its coupon trigger from ONE forward pass, and the
    branch it did not take is carried on a second accumulator rather than a second simulation.
    That is what makes this checkable: a re-run would have consumed the random stream the reported
    value was built from, and the exposure would drift. BIT-identical, not approximately."""
    mtm_off, cva_off, _ = _run(AUTOCALL, collateralised=collateralised)
    mtm_on, cva_on, grad = _run(AUTOCALL, gradient=True, collateralised=collateralised)
    assert np.array_equal(mtm_off, mtm_on), 'exposure moved when sensitivities were requested'
    assert cva_off == cva_on, f'cva moved: {cva_off!r} -> {cva_on!r}'
    assert grad is not None and abs(grad) > 0.0, 'no equity gradient was reported at all'


def test_the_autocall_trigger_is_what_the_residual_is():
    """Attribution, so the fix is aimed at the right thing. With no scenario anywhere near either
    of this deal's indicators the registration still runs and still costs nothing, and the
    uncorrected gradient already agrees with bump-and-reprice.

    MEASURED: AAD +0.0002982402, CRN best +0.00029712759, 0.37% apart at 3.82% flatness. Deleting
    the correction changes NOTHING - the AAD repeats to every digit - and that is the reading, not
    a placebo: the claim is precisely that this fixture's registration contributes zero. It is the
    control for the gates that do move.

    SMALL_RUNGS, not LIVE_RUNGS, and this is the one place in the file where that is right: with
    the trigger unreachable the payoff is smooth in spot, so the large bumps measure curvature
    instead - 3.45% at 2e-3..1e-2 against 0.37% here, on the same AAD."""
    kw = dict(batch=1024, mcmc=256, batches=16)
    aad = _run(AUTOCALL_NO_TRIGGER, gradient=True, **kw)[2]
    r = ladder(price=lambda s: _run(AUTOCALL_NO_TRIGGER, spot=s, **kw)[1], aad=aad, base=bb.SPOT,
               rungs=SMOOTH_RUNGS)
    assert r.agrees(tol=0.02), f'an unreachable trigger should already agree\n{r}'


def test_autocall_coupon_digital_gradient_matches_bump_and_reprice():
    """The aligned coupon digital in pv_MC_AutoCallSwap. An autocall observed on its coupon date
    really has redeemed, so the jump is product economics and must NOT be smoothed away - what has
    to reach the tape is the flux of scenarios across the threshold, in BOTH halves of its reach:
    the own-row fired/survived override and the carried knock-out latch
    killing every later row (LatchedBoundarySet).

    MEASURED, 65536 paths: AAD +2.29385e-06 against a CRN best of +2.2925346e-06, 0.06% apart on
    rungs 5e-3/7e-3/1e-2 flat to 5.0%. The path count is 64 batches and the window is the top of
    LIVE_RUNGS, both for the same measured reason: the carried latch made the deal DIE when it
    autocalls, which shrank this gradient tenfold (it was +2.27e-05 while the deal kept paying)
    against an oracle whose ABSOLUTE noise did not move - so at 16 batches both estimators sit in
    their own noise (8.45% apart, rungs scattering to 31%), and a rung's relative noise scales as
    1/h, which prices the 2e-3 and 3e-3 rungs out of the oracle's resolution (12.84% off at 3e-3
    beside 0.06% at 7e-3, at 64 batches). Same reasoning that moved every live gate to the large
    window in the first place, one octave further.

    MUTATION, each half of the one-per-decision registration suppressed alone, same 64 batches
    (the oracle does not move, so both die on the AGREEMENT clause at any window):
      latch neutralised:   +1.5035267e-05, 84% disagreement - the tape keeps paying rows the
                           latch killed, reproducing the pre-carry reading (+1.5043e-05)
      own-row suppressed:  -1.7815224e-06 - the sign FLIPS without the decision row's flux
    An aligned coupon digital is nothing BUT flux."""
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
    """Attribution, so the fix is aimed at the right thing. With the barrier UNREACHABLE the latch
    never fires and the same machinery agrees with bump-and-reprice; with it live, it does not.
    Any claimed fix has to close the second reading without disturbing the first.

    MEASURED: AAD +0.013276255 against a CRN best of +0.013276433 - 0.00% apart on a ladder flat to
    0.00%, which is what a smooth payoff differenced under common random numbers looks like when
    nothing discontinuous is in the way. Deleting the correction repeats the AAD to every digit,
    same as the autocall control above: the registration is live and worth exactly nothing."""
    far = dict(bb.BARRIER_DEAL, Barrier_Price=1e-6,
               Barrier_Dates=list(MONTHLY))
    kw = dict(batch=1024, mcmc=256)
    aad = _run(far, gradient=True, **kw)[2]
    r = ladder(price=lambda s: _run(far, spot=s, **kw)[1], aad=aad, base=bb.SPOT,
               rungs=SMOOTH_RUNGS)
    assert r.agrees(tol=0.02), f'an unreachable barrier should already agree\n{r}'


# ------------------------------------------------------- acceptance, xfail until the change lands

def test_discrete_barrier_latch_gradient_matches_bump_and_reprice():
    """The already-hit latch in pv_discrete_barrier_option. A discretely monitored knock-out really
    is worth nothing once it crosses, so the jump is genuine product economics and must NOT be
    smoothed away - the flux of paths across the barrier is what has to reach the tape.

    MEASURED on the grid this now runs, 4096 paths: AAD +0.016022041 against a CRN best of
    +0.016227701, 1.28% apart on a ladder flat to 5.64%.

    MUTATION - the correction deleted (`pricing.boundary_correction` returns None): the same ladder
    reads 36.22% from it. KILLED with a 7x margin, so this gate is measuring its own subject and not
    the smooth part of the sensitivity. The correction is 24% of the reported gradient.

    The path count is four batches rather than one because a single batch puts BOTH estimators below
    their own noise: at 1024 paths the AAD read +0.015927 and at 16384 +0.015625, a 1.9% wander that
    is the same size as the residual being gated."""
    kw = dict(batch=1024, mcmc=256, batches=4)
    aad = _run(DISCRETE_BARRIER, gradient=True, **kw)[2]
    r = ladder(price=lambda s: _run(DISCRETE_BARRIER, spot=s, **kw)[1], aad=aad, base=bb.SPOT,
               rungs=LIVE_RUNGS)
    assert r.agrees(tol=0.05), f'{r}'


def test_collateralised_barrier_latch_gradient_matches_bump_and_reprice():
    """The same defect with collateral in the way, which is the harder half: a gross-mtm delta
    reaches the net through Vte AND through the balance the collateral scan produces, so a fix
    that only handles the additive path will pass the test above and fail this one - which is
    exactly what happened, and what sent the gross-to-net chain into post_process.

    A COLLATERALISED netting set puts its own margin-call schedule on the mtm grid - measured here,
    86 mtm rows against the barrier's own 51, with 81 interpolated - so this is the fixture in the
    repo where the branch profile was WORST mis-mapped, and the uncollateralised one above (17 rows,
    17 deal rows, no interpolation) could not see it.

    MEASURED, 16384 paths: AAD +0.001502237 against a CRN best of +0.00160297, 6.71% apart on a
    ladder flat to 1.70%. Two further seeds read 3.37% and 4.80% at flatness 2.07% and 1.13%, so
    5% +/- 2% is the residual and not a lucky rung.

    MUTATION - the correction deleted: 47.4% from the same ladder. KILLED, 5.9x clear of the
    tolerance. The correction is 28% of the reported gradient here.

    THE TOLERANCE IS 8% AND THAT IS A RE-BASELINE, not a widening to fit. At 65536 paths the
    disagreement is 4.55% on a ladder flat to 4.72% and still falling monotonically in h, so the
    residual is real but small; at the gate's own path count the correction estimator's bandwidth
    scatter is the same size - reading -2.41% at bandwidth 0.005, -6.11% at 0.01 and 0.02, -6.25% at
    0.05, all against one oracle. 8% is that envelope. The old 5% was measured at 2.25% on the
    INTERPOLATED grid (`Dynamic_Scenario_Dates='No'`), where the barrier observation was a convex
    combination of two scenario rows and the oracle was differencing a smoothed payoff; that reading
    reproduces exactly if the pin is put back, which is how it was told apart from a regression."""
    kw = dict(batch=512, mcmc=128, collateralised=True, batches=32)
    aad = _run(DISCRETE_BARRIER, gradient=True, **kw)[2]
    r = ladder(price=lambda s: _run(DISCRETE_BARRIER, spot=s, **kw)[1],
               aad=aad, base=bb.SPOT, rungs=LIVE_RUNGS)
    assert r.agrees(tol=0.08), f'{r}'


def _fva(spot, gradient, batch=1024, mcmc=192, batches=16):
    """FVA and its equity-spot gradient. A funding SPREAD is what makes it non-zero - with the cost,
    benefit and risk-free curves all equal the adjustment is identically zero and measures nothing."""
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
    """FVA reads the same exposure as CVA, so it drops the same boundary terms - and it is the path
    that matters in production, because the shipped batch job DELETES the CVA section, so a
    correction assembled only over there could never fire for it.

    MEASURED, 16384 paths: AAD +0.01328747 against a CRN best of +0.01316879, 0.89% apart on a
    ladder flat to 2.18%. Two further seeds read 3.61% and 3.19% at flatness 3.59% and 3.18%.

    MUTATION - the correction deleted: AAD +0.010118628, which the same ladder reads 30.1% from.
    KILLED with a 6x margin; the correction is 24% of the reported gradient.

    THE PATH COUNT IS THE RE-BASELINE. One simulation batch is 1024 paths and both estimators are
    below their own noise there: the same gate read 3.30% at 1024, 0.88% at 2048 and 1.03% at 16384
    against a CRN median that moved by less than either step. The old docstring's 0.65% at 1.27%
    flatness was taken with `Dynamic_Scenario_Dates='No'`, i.e. against a smoothed barrier
    observation; nothing in the engine moved."""
    assert _fva(bb.SPOT, False) > 0.0, 'no funding spread - the adjustment is identically zero'
    aad = _fva(bb.SPOT, gradient=True)
    r = ladder(price=lambda s: _fva(s, False), aad=aad, base=bb.SPOT, rungs=LIVE_RUNGS)
    assert r.agrees(tol=0.05), f'the fva gradient is missing its boundary term\n{r}'


def test_the_correction_generalises_to_the_other_barrier_direction():
    """The gap must be signed so gap > 0 means CROSSED, and that sign flips with the barrier
    direction - a DOWN barrier is crossed from above, an UP barrier from below. Getting it backwards
    still converges, it just pulls the wrong way, so a second direction has to be measured rather
    than reasoned about.

    Up-and-IN is the variant with material exposure: it knocks in as the call goes into the money.
    Its mirror images are deliberately NOT gated - an up-and-OUT call is knocked out exactly when it
    becomes valuable, and a down-and-in call knocks in deep out of the money, so both carry a CVA
    delta near -0.0003 where the CRN oracle itself stops converging (measured flatness 28% and 82%).
    A gate there would be pinning Monte Carlo noise.

    MEASURED, 16384 paths: AAD +0.013331229 against a CRN best of +0.013328579, 0.02% apart on a
    ladder flat to 0.08%.

    THE MUTATION HERE IS THE SIGN, NOT THE SUBJECT'S EXISTENCE, and the distinction is the whole
    reason this deal is in the file. An up-and-in CALL is nearly continuous at its own barrier - at
    the crossing it becomes the vanilla it was already worth most of - so the correction is only
    1.6% of the reported gradient, against 24% for the knock-out above. Deleting it moves the
    reading to 2.35%, which is a kill only because the tolerance is 1%: at the 2% this carried
    before, the suppression mutant SURVIVED and this gate was a placebo for the term's existence.
    Negating the gap (`stochastic_boundary_correction(-gap, ...)`, i.e. the correction pulling the
    wrong way, which is the defect the docstring is about) is killed at 2.86% on the shipped 1024
    paths and by the same 2x margin here. Tightened rather than widened, and both mutants recorded
    because a 1.6% term cannot be gated to better than that on this payoff."""
    H = 110.0
    deal = dict(bb.BARRIER_DEAL, Barrier_Type='Up_And_In', Barrier_Price=H,
                Barrier_Dates=[bb.BASE + pd.Timedelta(days=d) for d in range(30, 366, 30)])
    kw = dict(batch=1024, mcmc=256, batches=16)
    aad = _run(deal, gradient=True, **kw)[2]
    r = ladder(price=lambda s: _run(deal, spot=s, **kw)[1], aad=aad, base=bb.SPOT, rungs=LIVE_RUNGS)
    assert r.agrees(tol=0.01), f'up-barrier gap sign or counterfactual is wrong\n{r}'


def test_the_correction_covers_heston_nandi_barriers():
    """instruments.py refuses the CONTINUOUS barrier variant for SpotModel='HestonNandi', so every
    HN barrier deal routes through the discrete pricer and its already-hit latch - the audit put
    that at 100% of them. The registration sits in the shared pricer, which should cover HN, but
    'should' is not a measurement: HN takes a different branch through sim_spot_oss, and its
    hit_value for a knock-out is zeros rather than a closed form.

    MEASURED: AAD +1.4698473 against a CRN best of +1.4871248, 1.18% apart on a ladder flat to
    3.49%, on a CVA delta of 1.47 - large enough that the oracle resolves it cleanly, unlike the
    mirror-image barrier variants.

    MUTATION - the correction deleted: +1.4004674, read 6.19% from the same ladder. KILLED with a 3x
    margin; the correction is 4.7% of the reported gradient, an order smaller than the GBM barrier's
    24% because a knocked-out HN barrier's counterfactual is the model-free zeros branch.

    THIS GATE WAS FAILING AT 72.8% AND IT WAS NOT ITS OWN FAULT. `pricing.boundary_weights` carried
    a refusal that could never fire - a Cauchy-Schwarz ratio bounded by 1, tested against 1e-30 -
    and one HN decision was solving a local-linear fit on two points 0.021 apart, returning weights
    of +50.4 and -49.5. Refusing on ||weights||_1 fixed it there and nowhere else; every reading
    above is on the repaired guard."""
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

    `SensitivitiesEstimator` is constructed once per batch and its `__init__` IS the backward pass,
    so wrapping the class puts a probe exactly between batches - the only place from which the
    per-batch decomposition of an accumulating buffer is visible at all. `report_grad` copies off
    the live buffer, so the snapshots do not alias each other."""
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
    """`boundary_sets` is cleared per batch while `.grad` ACCUMULATES across them, and report()
    divides by Simulation_Batches. A correction added once but averaged over N - or accumulated N
    times without averaging - makes the reported gradient scale with the batch count.

    THIS IS AN EXACT GATE AND IT USED TO BE A CRN LADDER, which cannot see the defect at all. Two
    simulation batches of 512 paths is 1024 paths, and at 1024 paths the AAD itself reads
    +0.0265527 where 16384 paths give +0.0161766 - a 64% wander, on an oracle whose own readings
    scatter 12.8% across rungs. The ladder was failing at 29.16% and measuring nothing but its own
    variance. Every quantity below is bit-exact instead: no bump, no tolerance, no path count.

    Three assertions, and each kills a different mutant.

    PREFIX. Batch 0 of an N-batch run draws exactly what the 1-batch run draws - the seed is set
    once and `reset()` continues one stream - so the per-batch increments of the N=1, 2 and 4 runs
    are prefixes of one another, MEASURED +0.03610596959 for batch 0 in all three and
    +0.01699952959 for batch 1 in both runs that have one. A correction applied only on `final_run`
    puts batch 0's increment of an N=2 run at the uncorrected value, which is a different number.

    THE DIVISION. `reported == cumulative / N` exactly, which is calculation.py's
    `v / self.params['Simulation_Batches']` and nothing else.

    THE SUBJECT IS LIVE. The same runs with `pricing.boundary_correction` deleted give batch 0's
    correction contribution as +0.02399922704, two thirds of that batch's whole gradient, and it too
    is the same number at N=1, 2 and 4. Without this assertion the first two hold with the
    correction suppressed - they are properties of the accumulator, not of the term.

    (Batch 2's contribution is -1.3e-05, essentially nothing: a batch with no path near the barrier
    at a reporting row has no flux to recover. That is why the assertion is on batch 0, whose
    contribution is material, rather than on a mean over batches.)"""
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
    """The invariant the collateral counterfactual rests on, which nothing asserted.

    `gross_to_net` pushes a gross-mtm delta through At -> required balance -> bands -> scan -> the
    netting arithmetic. Feed it ZERO and it must reproduce the set's own reported net EXACTLY - if
    it does not, every correction is measured against a rebased baseline and is the wrong size
    while still converging and still looking bandwidth-stable. (What the SET then does with that
    level is a separate question, and the answer is `portfolio_delta`: it subtracts this same
    zero-delta baseline and hands the assembler the difference, which goes on the reported
    PORTFOLIO - see test_a_netting_set_is_scored_on_the_portfolio_not_on_itself.)

    It did not. `Vte` was re-derived as `g_Vt[Te]` rather than taken from the reported `b_Vte`, and
    under Exclude_Paid_Today the two carry DIFFERENT cashflow adjustments - a local-grid one and a
    Te-grid one. Measured with the mutation restored: max|diff| 120.51 against a reported |mean| of
    2.61, a 46x rebasing. Taking b_Vte makes the invariant true by construction.

    Both settings are gated because the defect is INVISIBLE at the default: with the option off the
    two forms coincide exactly, and no value gate can see it either way - the reported mtm is
    unchanged, which is what let it hide. Note Exclude_Paid_Today is read from the VALUATION
    CONFIGURATION, not the deal's fields; setting it on the netting dict is silently ignored and
    makes this test vacuous."""
    import derivus.pricing as C
    seen = {}
    original = C.boundary_correction

    def probe(shared, objective, reported_mtm, bandwidth):
        bset = next((x for x in shared.boundary_sets if isinstance(x, utils.BoundarySet)), None)
        if bset is not None and bset.net_from_gross is not None:
            with torch.no_grad():
                # a zero delta on the MTM grid - which is the grid the chain consumes, and which
                # only `to_mtm` knows how to reach from the pricer's own rows
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


LONG_ADDING_EXPOSURE = dict(DOMINATING_SHORT, Reference='LONG1', Buy_Sell='Buy', Units=50.0,
                            Forward_Price=50.0)


def test_two_netting_sets_with_a_live_exposure_agree_with_bump_and_reprice():
    """The companion to the zero-CVA gate above, and the reason it needs one.

    That gate builds a portfolio out of the money in EVERY scenario, so its CVA is exactly zero and
    any sensitivity must be too - elegant, no tolerance, and it catches the mis-scoping it was
    written for. But only that DIRECTION: mutating `portfolio_delta` to add the set's LEVEL rather
    than its CHANGE leaves the portfolio still <= 0, the relu still returns zero, and the mutant
    SURVIVES the whole suite. Verified - it did.

    So the second set here ADDS exposure instead of cancelling it, putting the portfolio
    comfortably in the money where a wrong delta has to move a real number.

    Sizing it the other way does not work, and the failure is worth recording. A collateralised set
    with a BINDING MTA has near-zero exposure by design - that is what collateral is - so shrinking
    the offsetting short until the CVA is merely positive parks the portfolio ON the relu boundary,
    where the gradient is a difference of cancelling terms and NEITHER estimator resolves it: the
    AAD read +1.44e-05, +6.97e-06 and -5.77e-05 across path counts, changing SIGN, while the CRN
    ladder stayed flat enough (0.87%) to look trustworthy. Flatness measures the ORACLE's spread
    only; it says nothing about the AAD's own noise, which is why a flat ladder beside an unstable
    AAD is not evidence of a residual."""
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

    # HONEST LIMIT, stated so nobody mistakes this for a mutation-level gate on the scoping. Both
    # this and the zero-CVA gate above measure the END-TO-END gradient, and the boundary term is a
    # small fraction of it - MEASURED, AAD +1.2132461 against a CRN best of +1.1865093, 2.20% apart
    # at 0.04% flatness, and deleting the correction entirely moves that to 0.23%, so even the
    # SUPPRESSION mutant survives here, never mind a mis-scoped one. The correction is 2.4% of the
    # reported gradient - so scoring it at the set rather than the portfolio moves the total by well
    # under this tolerance and the mutant SURVIVES both.
    # The scoping itself is verified by measuring THAT TERM directly against a CRN ladder on a
    # two-set portfolio (mis-scoped -15.3%, fixed +1.3%), which is what 0a6ee69's message records.
    # A gate that isolates it needs a portfolio where the correction DOMINATES the smooth
    # sensitivity, which is not a portfolio anyone runs.
