"""The portfolio where the boundary correction IS the sensitivity, and the mutation gate it buys.

`test_boundary_pricer_events.py` gates the correction end to end, and its own closing comment
states the limit: on both two-netting-set fixtures the boundary term is a small fraction of the
reported gradient - 2.4% on the live-exposure one - so suppressing the term entirely leaves every
gate there green. Those are gates on the TOTAL. This one is a gate on the TERM, and it needs a
portfolio nobody runs.

THE DEAL IS A DIGITAL, and that is the whole of the trick. A discretely monitored down-and-out
BINARY struck at ~zero is worth `Cash_Payoff` times the probability it never crossed, so its spot
sensitivity is almost entirely the flux of paths across the barrier; a knock-out CALL carries a
vanilla's intrinsic delta that the correction then has to compete with. MEASURED at 1024 paths in
one collateralised set, correction over smooth term: knock-out call 0.56 (H=90, monthly), 0.83
(H=95), 2.07 (H=98, fortnightly), against 3.04 for the H=95 MONTHLY digital - the same term over a
tenth of the smooth sensitivity, at a third of the fortnightly call's runtime.

THE SETS ARE COLLATERALISED AND PARKED ON THE RELU KINK, which is the row's own suggestion and buys
two things at once: the collateral tracks the gross, so what smooth delta survives the digital is
crushed further, and each set's own net crosses zero constantly, so scoring a counterfactual on one
SET rather than on the PORTFOLIO is a different number. Two sets, because two published
gross-to-net chains is the only shape in which a registration can be scored through the wrong one.

NO MINIMUM TRANSFER AMOUNT. `_collateralised_barrier` in the neighbouring file binds one on purpose
- it wants the transfer decision live - but that registers a SECOND decision whose own term does not
survive this fixture (see THE LIFT), and the subject here is the pricer event.

THE SEAM IS THE DECLARED FIELD, and nothing here patches a library object. `Boundary_AAD_Bandwidth`
is declared on the calculation with a default of 0.01; at `SUPPRESSED_BANDWIDTH` the kernel
underflows on every gap, so `boundary_weights`'s local-linear fit is unsolvable, its weights are
exactly zero and the correction is an exact zero rather than a small one. VERIFIED off-gate on this
fixture: the reported gradient there is +0.00019686216001920737, BIT-IDENTICAL to the same run with
`pricing.boundary_correction` deleted and identical again at a bandwidth of 1e-14, against a live
+0.0010291805076251752. The suppression is total, and it is not a bandwidth reading.

THE LIFT, and why it is not used - the obvious route, and a trap. A DELTA-FREE cash cushion (a
forward bought at strike zero and one sold at K, whose sum is exactly K in every scenario and
carries no equity delta) lifts the reported portfolio clear of the relu, makes the objective locally
linear and stops the counterfactual jumps being truncated, which multiplies the correction's share
to 113%. It also makes the reported gradient WRONG. MEASURED on the knock-out call at 4096 paths,
cushion scanned 0/10/30/100/300: CRN disagreement 10.1% / 56.8% / 84.3% / 93.4% / 93.4%, saturating
exactly where the relu stops binding (`mtm.min()` reaches 0); with a binding MTA in the set it reads
95.8%, the MTA registration carrying +0.01655 of a +0.01710 correction against a CRN total of
+0.00064. That is an engine defect, reported rather than gated here - this lane owns no engine file
- and it is why the first gate below asserts the portfolio still straddles zero.

WHAT THIS FILE DOES NOT GATE, said plainly because the row it answers is called "boundary SCOPING
is not mutation-gated". The mutant here is `Boundary_AAD_Bandwidth`, which suppresses the WHOLE
correction; what dies by 381.77% is therefore the correction's existence, not its scoping. A
mis-scoping mutant - `portfolio_delta` returning the SET level, or a counterfactual scored against a
zero portfolio - has no public seam: every registration in `derivus/` names its `BoundarySet` class
directly (`pricing.py` 2330/3149/3635/4164/4170/4888, `instruments.py` 1417/3265), with none of the
`cls.apply` indirection the recompute node offers, so reaching one needs a module rebind and this
lane does not patch. The fixture is BUILT so such a mutant would move a dominant term - two
collateralised sets, two published gross-to-net chains, each net crossing zero - but that claim is
untested and is asserted nowhere. The correction is mutation-gated; its scoping is not, and that
half of the row stays open.

AND THE 30% TOLERANCE IS NOT AN ACCURACY CLAIM. The live AAD sits 21.54% from its own CRN ladder
(23.82% at seed 2) and the gate accepts that: it is the estimator's residual at this path count,
not agreement. What does the work is the 381.77% kill, twelve times clear of the tolerance.
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

    MEASURED at 16384 paths: the reported delta is +0.00077085 and the same run with the correction
    suppressed reads +0.00019447, so the term is +0.00057638 - THREE TIMES the whole pathwise
    sensitivity and 75% of the number that gets reported (2.84x at seed 2). On the neighbouring
    file's two-set fixtures it is 2.4%, which is why the suppression mutant survives there and dies
    here.

    Two guards, because a fixture can stop being its own subject in two different ways.

    THE PORTFOLIO MUST STRADDLE ZERO. Lifting it clear of the relu is the other way to make the
    correction dominate - the jumps stop being truncated and the share goes to 113% - and it is
    measured to make the reported delta wrong by 84-96% (module docstring). A fixture that drifts
    into that regime would be gating a defect as if it were the answer, so the relu binding is
    asserted rather than assumed. Here the portfolio spans -18.0 to +14.3.

    THE DOMINANCE ITSELF, floored at 1.5 against a measured 2.96. Under 1.0 the gate below is
    measuring the smooth part of the delta and its mutant starts surviving."""
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
    on a ladder flat to 2.83%. Flat is the word - the five rungs read 20.4 / 21.5 / 21.8 / 18.9 /
    18.4%, which is a residual and not scatter. Seed 2 reads 23.82% at 2.99% flatness (AAD
    +0.00076225, CRN +0.00094385), so the residual is the estimator's and not one seed's luck.

    THE TOLERANCE IS 30% AND IT IS THAT RESIDUAL PLUS THE SEED SPREAD, NOT A WIDENING TO FIT. It is
    the correction estimator's own accuracy at the declared bandwidth and this path count, and it
    falls with paths: the one-collateralised-set companion (the same digital alone, 128 inner sims)
    reads 19.27% at 4096 paths and 15.36% at 16384, at 2.28x dominance with a 278.57% mutant. The
    roadmap's open row on `stochastic_boundary_correction` says why - the estimator has no measured
    bandwidth plateau and its documented operating point is 32768 paths, which no gate here runs.

    MUTATION - `Boundary_AAD_Bandwidth` at SUPPRESSED_BANDWIDTH, where the correction is an exact
    zero: the same oracle reads 381.77% from it (374.96% at seed 2). KILLED, 12x clear of the
    tolerance. That is the reading the roadmap row asked for: on the neighbouring file's
    live-exposure two-set fixture the identical mutant moves the CRN disagreement from 2.20% to
    0.23% and SURVIVES."""
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
