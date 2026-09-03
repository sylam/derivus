"""The portfolio where the boundary correction IS the sensitivity, and the mutation gate it buys.

`test_boundary_pricer_events.py` gates the correction end to end, but on both its fixtures the
boundary term is a small fraction of the reported gradient - 2.4% on the live-exposure one - so
suppressing the term leaves every gate there green. Those are gates on the TOTAL; this is a gate on
the TERM, and it needs a portfolio nobody runs.

THE DEAL IS A DIGITAL. A discretely monitored down-and-out BINARY struck at ~zero is worth
`Cash_Payoff` times the probability it never crossed, so its spot sensitivity is almost entirely the
flux of paths across the barrier; a knock-out CALL carries a vanilla's intrinsic delta the
correction has to compete with. MEASURED at 1024 paths, correction over smooth term: knock-out call
0.56 / 0.83 / 2.07 against 3.04 for the H=95 monthly digital. That reading is the FIXTURE CHOICE and
predates the settled ledger; the digital's own ratio has since risen to 3.80 at 16384 paths, which
is the direction the choice needed, and the knock-out-call arm has no fixture here to re-take it on.

THE SETS ARE COLLATERALISED AND PARKED ON THE RELU KINK, which buys two things: the collateral
tracks the gross so what smooth delta survives is crushed further, and each set's net crosses zero
constantly so scoring a counterfactual on one SET rather than the PORTFOLIO is a different number.
Two sets, because two published gross-to-net chains is the only shape in which a registration can be
scored through the wrong one. No minimum transfer amount, because that registers a SECOND decision
and the subject here is the pricer event.

THE SEAM IS THE DECLARED FIELD, and nothing here patches a library object. At
`SUPPRESSED_BANDWIDTH` the kernel underflows on every gap, `boundary_weights`'s local-linear fit is
unsolvable and the correction is an exact zero - VERIFIED off-gate as BIT-IDENTICAL to the same run
with `pricing.boundary_correction` deleted.

THE LIFT, which now ships as a lane of its own. A DELTA-FREE cash cushion lifts the reported
portfolio clear of the relu and multiplies the correction's share - and used to make the reported
gradient WRONG: CRN disagreement 10.1 / 56.8 / 84.3 / 93.4 / 93.4% on the knock-out call at cushion
0 / 10 / 30 / 100 / 300, saturating exactly where the relu stops binding. Most of that was the
LEDGER: a lifted portfolio prices its exposure off cash the counterfactual never flipped, and with
the settlement undeclared the digital lane below reports -0.00010726372 where its own oracle wants
+0.00034761669. What is left of the lift is inside the estimator's own residual at this path count.
The first gate still asserts the portfolio straddles zero, because the two lanes measure different
things.

WHAT THIS FILE DOES NOT GATE. The mutant is `Boundary_AAD_Bandwidth`, which suppresses the WHOLE
correction, so what dies by 366.61% is the correction's existence rather than its SCOPING. A
mis-scoping mutant has no public seam - every registration in `derivus/` names its `BoundarySet`
class directly, with none of the `cls.apply` indirection the recompute node offers - so reaching one
needs a module rebind, which this lane does not do. The fixture is BUILT so such a mutant would move
a dominant term, but that claim is asserted nowhere and that half of the row stays open.

THE 10% TOLERANCE IS NOT AN ACCURACY CLAIM: the live AAD sits 2.74% from its own CRN ladder (0.48%
at seed 2), which is the estimator's residual at this path count. What does the work is the 366.61%
kill, thirty-six times clear of it.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from crn_ladder import Ladder, ladder
import test_barrier_bridge as bb
from derivus.instruments import construct_instrument
from test_boundary_pricer_events import LIVE_RUNGS, NETTING, _run

#: The kernel selects nothing at this bandwidth, so the correction is an exact zero - the
#: suppression mutant, taken through the declared field rather than through a patched engine.
SUPPRESSED_BANDWIDTH = 1e-12

#: The measured residual across two seeds - 2.74% and 0.48% - with room for the ladder's own 2.8%
#: flatness, not a widening. The second gate's docstring quotes the five rungs it is taken from.
TOLERANCE = 0.10
#: The suppression mutant reads 3.67x / 3.65x its own gradient off the same oracle; this is the
#: floor, and it is 20x the tolerance above.
KILL_MARGIN = 2.0

#: 16384 paths. The batch is a MEMORY constraint as much as a statistical one: two collateralised
#: sets each put a whole margin schedule on the mtm grid, and 512 paths against 128 inner sims OOMs
#: a 24GB device on this fixture.
PATHS = dict(batch=512, mcmc=64, batches=32)

MONTHLY = [bb.BASE + pd.Timedelta(days=d) for d in range(30, 366, 30)]


def _digital(reference, barrier):
    """A discretely monitored down-and-out binary. The strike sits at ~zero so the terminal digital
    is certain and the deal is worth `Cash_Payoff` times the probability it never crossed - which
    makes its spot delta the barrier flux and almost nothing else."""
    return {'Object': 'EquityBarrierBinaryOption', 'Reference': reference, 'Currency': 'USD',
            'Payoff_Currency': 'USD', 'Equity': 'EQ', 'Dividends': 'EQ', 'Discount_Rate': 'USD',
            'Equity_Volatility': 'EQ', 'Buy_Sell': 'Buy', 'Option_Type': 'Call',
            'Strike_Price': 1e-6, 'Expiry_Date': bb.BASE + pd.Timedelta(days=365), 'Units': 1.0,
            'Barrier_Type': 'Down_And_Out', 'Barrier_Price': barrier, 'Cash_Payoff': 10.0,
            'Barrier_Dates': list(MONTHLY), 'Settlement_Date': ''}


DIGITAL_A = _digital('DIG_A', 95.0)
DIGITAL_B = _digital('DIG_B', 90.0)


def _collateralised(reference, deal):
    """One collateralised netting set holding one digital. Every credit-support amount stays at the
    base fixture's zero, so the only decision registered beneath it is the barrier's latch."""
    return {'Instrument': construct_instrument(
        dict(NETTING, Reference=reference, Collateralized='True'), {}),
        'Children': [{'Instrument': construct_instrument(deal, {})}]}


def _two_collateralised_sets(c):
    """Two sets at different barriers, so the two corrections add rather than cancel and neither
    set's chain can stand in for the other's without moving the answer."""
    return [_collateralised('NS_A', DIGITAL_A), _collateralised('NS_B', DIGITAL_B)]


def _gradient(bandwidth=None):
    """The reported equity-spot CVA delta, live or with the correction suppressed."""
    return _run(DIGITAL_A, gradient=True, children=_two_collateralised_sets,
                bandwidth=bandwidth, **PATHS)[2]


def test_the_correction_dominates_the_smooth_cva_delta():
    """The reading this file exists to make: the boundary term is 3.80x the smooth sensitivity.

    MEASURED at 16384 paths: reported delta +0.00096053299, suppressed +0.00020020475, so the term
    is +0.00076033 - nearly four times the whole pathwise sensitivity and 79% of what gets reported
    (3.63x at seed 2). On the neighbouring file's fixtures it is 2.4%, which is why the mutant
    survives there. The suppressed half is BIT-IDENTICAL to c938d6e, so all of the move the settled
    ledger bought - reported +0.00078302273 there - is in the correction.

    Two guards. THE PORTFOLIO MUST STRADDLE ZERO: lifting it clear of the relu also makes the
    correction dominate, and used to make the reported delta wrong by 84-96%, so the relu binding is
    asserted rather than assumed (the portfolio spans -18.0 to +14.4). THE DOMINANCE ITSELF, floored
    at 1.5 against a measured 3.80 - under 1.0 the gate below measures the smooth part."""
    mtm, cva, live = _run(DIGITAL_A, gradient=True, children=_two_collateralised_sets, **PATHS)
    assert mtm.min() < 0.0 < mtm.max(), (
        f'the portfolio no longer straddles zero (it spans {mtm.min():+.6g} to {mtm.max():+.6g}) - '
        f'the CVA relu has stopped binding, and the reported delta on such a portfolio is measured '
        f'to be 84-96% from bump-and-reprice')
    assert cva > 0.0, f'the portfolio has no exposure to be sensitive to; cva {cva!r}'

    smooth = _gradient(bandwidth=SUPPRESSED_BANDWIDTH)
    dominance = abs(live - smooth) / abs(smooth)
    assert dominance >= 1.5, (
        f'the boundary correction is {dominance:.2f}x the smooth sensitivity (reported '
        f'{live:+.8g}, suppressed {smooth:+.8g}) - this fixture has stopped being '
        f'correction-dominated and the gate below is measuring the wrong thing')


def test_the_suppressed_correction_dies_against_bump_and_reprice():
    """AAD against a CRN bump ladder, with the suppression mutant read off the SAME oracle.

    MEASURED, 16384 paths, seed 1: AAD +0.00096053299 against a CRN best of +0.00093416981, 2.74%
    apart on a ladder flat to 2.83% (rungs 3.61 / 2.74 / 2.37 / 4.76 / 5.09% - a residual, not
    scatter). Seed 2 reads 0.48% at 2.71% flatness. It was 21.54% before the settled ledger was
    declared, which is what most of that residual was.

    THE 10% TOLERANCE IS THAT RESIDUAL PLUS THE SEED SPREAD PLUS THE LADDER'S OWN FLATNESS, not a
    widening to fit: it is the estimator's accuracy at the declared bandwidth and this path count.
    The estimator has no measured bandwidth plateau and its documented operating point is 32768
    paths, which no gate here runs.

    MUTATION - `Boundary_AAD_Bandwidth` at SUPPRESSED_BANDWIDTH: the same oracle reads 366.61%
    (365.07% at seed 2). KILLED, 36x clear. On the neighbouring file's two-set fixture the identical
    mutant moves the CRN disagreement from 2.20% to 0.23% and SURVIVES."""
    live = _gradient()
    smooth = _gradient(bandwidth=SUPPRESSED_BANDWIDTH)
    r = ladder(price=lambda s: _run(DIGITAL_A, spot=s, children=_two_collateralised_sets,
                                    **PATHS)[1],
               aad=live, base=bb.SPOT, rungs=LIVE_RUNGS)
    assert r.agrees(tol=TOLERANCE), f'the correction-dominated CVA delta is wrong\n{r}'

    # the same oracle readings against the suppressed gradient - the mutant costs one run rather
    # than a second ladder, and the kill is asserted here rather than remembered in a docstring
    mutant = Ladder(smooth, bb.SPOT, r.rungs, r.crn)
    killed = abs(mutant.best - smooth) / abs(smooth)
    assert not mutant.agrees(tol=TOLERANCE) and killed >= KILL_MARGIN, (
        f'suppressing the boundary correction through Boundary_AAD_Bandwidth left the gate green '
        f'at {killed:.1%} - the fixture is not correction-dominated and this file gates nothing\n'
        f'{mutant}')


# ------------------------------------------------------------------------------------- the lift

#: A delta-free cash cushion: buy the forward struck at zero, sell the one struck at CUSHION, in
#: its own uncollateralised set. Worth CUSHION*DF every scenario with zero equity delta.
CUSHION = 300.0

#: The lifted lane's own residual: 10.79% at 16384 paths on a ladder flat to 6.34%, and it FALLS
#: with paths (12.03% at 4096, 9.28% at 8192). Not the un-lifted lane's 30% - the relu is not
#: binding here, so the objective is locally linear and the estimator has an easier job.
LIFTED_TOLERANCE = 0.20


def _forward(reference, side, price):
    return {'Object': 'EquityForwardDeal', 'Reference': reference, 'Currency': 'USD',
            'Equity': 'EQ', 'Discount_Rate': 'USD', 'Payoff_Currency': 'USD', 'Buy_Sell': side,
            'Units': 1.0, 'Forward_Price': price,
            'Maturity_Date': bb.BASE + pd.Timedelta(days=365)}


def _lifted(c):
    """The same two collateralised digitals, plus the cushion that lifts the portfolio off the
    relu. The cushion carries no equity delta, so everything the gradient reads is still the
    digitals'."""
    return _two_collateralised_sets(c) + [{'Instrument': construct_instrument(
        dict(NETTING, Reference='NS_CUSH', Collateralized='False'), {}),
        'Children': [{'Instrument': construct_instrument(_forward('CUSH_L', 'Buy', 0.0), {})},
                     {'Instrument': construct_instrument(_forward('CUSH_S', 'Sell', CUSHION), {})}]}]


def _lifted_gradient(bandwidth=None):
    return _run(DIGITAL_A, gradient=True, children=_lifted, bandwidth=bandwidth, **PATHS)[2]


def test_the_lifted_portfolio_reports_a_delta_its_own_oracle_agrees_with():
    """The lane the lift used to fail. Off the relu the objective is locally linear, every
    counterfactual is scored at full weight, and what a collateralised set is worth turns on the
    cash it has already paid - so a counterfactual whose SETTLEMENT does not follow its branch
    prices the wrong exposure and the error is unbounded rather than small.

    MEASURED at 16384 paths, cushion 300: AAD +0.00031376127 against a CRN best of +0.00034761669,
    10.79% on a ladder flat to 6.34%, and falling with paths (12.03% at 4096, 9.28% at 8192).

    TWO MUTANTS, DYING ON DIFFERENT ASSERTIONS, which is worth stating because only one of them
    reaches the ladder. The SETTLEMENT undeclared - the state this file shipped in, reproduced
    OFF-GATE by dropping `settles` at the registration, there being no document switch for it -
    collapses the correction to the size of the smooth term and dies on the DOMINANCE GUARD, 1.0380x
    against a floor of 3.0; off the same oracle it reads -0.00010726372, sign-flipped and 424.08%
    out where this lane reads 10.79%, but the gate never gets there. `Boundary_AAD_Bandwidth` at
    SUPPRESSED_BANDWIDTH is the one taken through the declared field: it reads 760.48% and dies on
    the ladder. The un-lifted lane above cannot see the settlement mutant at all - with the relu
    binding, the exposure is crushed to near zero exactly where the ledger error lives.

    Two guards, and the first of them is the discriminator above. THE CUSHION MUST LIFT - if the
    portfolio still straddles zero this is the gate above with extra deals. THE CORRECTION MUST
    DOMINATE, floored at 3.0 against a measured 6.96x.
    """
    mtm, cva, live = _run(DIGITAL_A, gradient=True, children=_lifted, **PATHS)
    assert mtm.min() >= 0.0, (
        f'the cushion did not lift the portfolio clear of the relu (it spans {mtm.min():+.6g} to '
        f'{mtm.max():+.6g}) - this lane is then the one above and gates nothing new')
    assert cva > 0.0, f'the lifted portfolio has no exposure to be sensitive to; cva {cva!r}'

    smooth = _lifted_gradient(bandwidth=SUPPRESSED_BANDWIDTH)
    dominance = abs(live - smooth) / abs(smooth)
    assert dominance >= 3.0, (
        f'the boundary correction is {dominance:.4f}x the smooth sensitivity (reported '
        f'{live:+.8g}, suppressed {smooth:+.8g}) - lifted, it is measured at 6.96x, and a '
        f'registration that does not declare its settlement reads 1.0380x here')

    r = ladder(price=lambda s: _run(DIGITAL_A, spot=s, children=_lifted, **PATHS)[1],
               aad=live, base=bb.SPOT, rungs=LIVE_RUNGS)
    assert r.agrees(tol=LIFTED_TOLERANCE), f'the lifted CVA delta is wrong\n{r}'

    mutant = Ladder(smooth, bb.SPOT, r.rungs, r.crn)
    killed = abs(mutant.best - smooth) / abs(smooth)
    assert not mutant.agrees(tol=LIFTED_TOLERANCE) and killed >= KILL_MARGIN, (
        f'suppressing the correction left the lifted gate green at {killed:.1%}\n{mutant}')
