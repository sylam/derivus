"""The portfolio where the boundary correction IS the sensitivity, and the mutation gate it buys.

`test_boundary_pricer_events.py` gates the correction end to end, but on both its fixtures the
boundary term is a small fraction of the reported gradient - 2.4% on the live-exposure one - so
suppressing the term leaves every gate there green. Those are gates on the TOTAL; this is a gate on
the TERM, and it needs a portfolio nobody runs.

THE DEAL IS A DIGITAL. A discretely monitored down-and-out BINARY struck at ~zero is worth
`Cash_Payoff` times the probability it never crossed, so its spot sensitivity is almost entirely the
flux of paths across the barrier; a knock-out CALL carries a vanilla's intrinsic delta the
correction has to compete with. MEASURED at 1024 paths, correction over smooth term: knock-out call
0.56 / 0.83 / 2.07 against 3.04 for the H=95 monthly digital.

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

THE LIFT, and why it is not used. A DELTA-FREE cash cushion lifts the reported portfolio clear of
the relu and multiplies the correction's share to 113% - and makes the reported gradient WRONG.
MEASURED on the knock-out call at 4096 paths, cushion 0/10/30/100/300: CRN disagreement 10.1% /
56.8% / 84.3% / 93.4% / 93.4%, saturating exactly where the relu stops binding. That is an engine
defect, reported rather than gated here, and it is why the first gate asserts the portfolio still
straddles zero.

WHAT THIS FILE DOES NOT GATE. The mutant is `Boundary_AAD_Bandwidth`, which suppresses the WHOLE
correction, so what dies by 381.77% is the correction's existence rather than its SCOPING. A
mis-scoping mutant has no public seam - every registration in `derivus/` names its `BoundarySet`
class directly, with none of the `cls.apply` indirection the recompute node offers - so reaching one
needs a module rebind, which this lane does not do. The fixture is BUILT so such a mutant would move
a dominant term, but that claim is asserted nowhere and that half of the row stays open.

THE 30% TOLERANCE IS NOT AN ACCURACY CLAIM: the live AAD sits 21.54% from its own CRN ladder (23.82%
at seed 2), which is the estimator's residual at this path count. What does the work is the 381.77%
kill, twelve times clear of it.
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

#: The measured residual across two seeds, not a widening - see the second gate's docstring, which
#: quotes the five rungs it is taken from and the path counts it falls with.
TOLERANCE = 0.30
#: The suppression mutant reads 3.82x / 3.75x its own gradient off the same oracle; this is the
#: floor, and it is 12x the tolerance above.
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
    """The reading this file exists to make: the boundary term is 2.96x the smooth sensitivity.

    MEASURED at 16384 paths: reported delta +0.00077085, suppressed +0.00019447, so the term is
    +0.00057638 - three times the whole pathwise sensitivity and 75% of what gets reported (2.84x at
    seed 2). On the neighbouring file's fixtures it is 2.4%, which is why the mutant survives there.

    Two guards. THE PORTFOLIO MUST STRADDLE ZERO: lifting it clear of the relu also makes the
    correction dominate, and makes the reported delta wrong by 84-96%, so the relu binding is
    asserted rather than assumed (the portfolio spans -18.0 to +14.3). THE DOMINANCE ITSELF, floored
    at 1.5 against a measured 2.96 - under 1.0 the gate below measures the smooth part."""
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

    MEASURED, 16384 paths, seed 1: AAD +0.00077085 against a CRN best of +0.00093689, 21.54% apart
    on a ladder flat to 2.83% (rungs 20.4 / 21.5 / 21.8 / 18.9 / 18.4% - a residual, not scatter).
    Seed 2 reads 23.82% at 2.99% flatness.

    THE 30% TOLERANCE IS THAT RESIDUAL PLUS THE SEED SPREAD, not a widening to fit: it is the
    estimator's accuracy at the declared bandwidth and this path count, and it falls with paths (the
    one-set companion reads 19.27% at 4096 and 15.36% at 16384). The estimator has no measured
    bandwidth plateau and its documented operating point is 32768 paths, which no gate here runs.

    MUTATION - `Boundary_AAD_Bandwidth` at SUPPRESSED_BANDWIDTH: the same oracle reads 381.77%
    (374.96% at seed 2). KILLED, 12x clear. On the neighbouring file's two-set fixture the identical
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
