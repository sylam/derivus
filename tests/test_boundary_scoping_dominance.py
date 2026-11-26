"""The portfolio where the boundary correction IS the sensitivity.

`test_boundary_pricer_events.py` gates the correction end to end, but on its fixtures the boundary
term is a small fraction of the reported gradient - 2.4% on the live-exposure one - so a defect in
the term moves the total by little. This portfolio is authored so the term dominates.

THE DEAL IS A DIGITAL. A discretely monitored knock-out BINARY struck at ~zero is worth
`Cash_Payoff` times the probability it never crossed, so its spot sensitivity is almost entirely the
flux of paths across the barrier; a knock-out CALL carries a vanilla's intrinsic delta the
correction has to compete with.

BOTH BARRIER DIRECTIONS, mirrored about the spot. The digital's survival is monotone in the spot, so
the CVA delta's SIGN is known before anything is run: positive below a down barrier, negative under
an up one. The correction carries it at three to four times the smooth term, so a gap signed the
wrong way for its direction reports the wrong one.

TWO COLLATERALISED SETS, at different barriers, so the two corrections add rather than cancel and
neither set's gross-to-net chain can stand in for the other's. No minimum transfer amount, because
that registers a SECOND decision and the subject here is the pricer event.

LIFTED OFF THE RELU. A DELTA-FREE cash cushion in its own uncollateralised set lifts the reported
portfolio clear of the CVA's kink. Off the relu every counterfactual is scored at full weight and
what a collateralised set is worth turns on the cash it has already paid - so a counterfactual
whose SETTLEMENT does not follow its branch prices the wrong exposure. With the settlement
undeclared the lifted portfolio reads +1.96e-4 down where its own CRN oracle wants +5.29e-4.
Parked ON the relu instead (the un-lifted portfolio spans -16.6 to +14.4), the same two sets read
the correction at 3.65x the smooth term and the AAD 0.04% from its CRN ladder at 8192 paths (0.04%
at seed 2), where suppressing the correction reads 365.20% - and the undeclared settlement is
invisible there, the exposure crushed to near zero exactly where the ledger error lives.

THE SEAM IS THE DECLARED FIELD, and nothing here patches a library object. At
`SUPPRESSED_BANDWIDTH` the kernel underflows on every gap, `boundary_weights`' local-linear fit is
unsolvable and the correction is an exact zero - verified bit-identical to the same run with
`pricing.boundary_correction` deleted.

NOT GATED: a mis-SCOPED correction (scored through the wrong set's chain). Every registration names
its `BoundarySet` class directly, so such a mutant needs a module rebind; the fixture is built so it
would move a dominant term, but that is asserted nowhere.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import pytest

import test_barrier_bridge as bb
from derivus.instruments import construct_instrument
from test_boundary_pricer_events import NETTING, _digital, _run

#: The kernel selects nothing at this bandwidth, so the correction is an exact zero.
SUPPRESSED_BANDWIDTH = 1e-12

#: 512 paths, one batch. Over seeds 1-3 the correction reads 2.4 to 5.8 times the smooth term on
#: the CPU and 2.7 to 4.5 on the card: the smooth term is steady near 1.2e-4 and the correction is
#: the part that moves. At 4096 paths it reads 3.2 to 4.1 on either.
PATHS = dict(batch=512, mcmc=64, batches=1)

#: Each direction's two barriers, mirrored about the spot, and the sign its CVA delta must carry.
DIRECTIONS = {'Down_And_Out': ((95.0, 90.0), 1.0), 'Up_And_Out': ((105.0, 110.0), -1.0)}

#: A delta-free cash cushion: buy the forward struck at zero, sell the one struck at CUSHION, in
#: its own uncollateralised set. Worth CUSHION*DF every scenario with zero equity delta, and
#: outliving the collateralised sets, which report their closeout ten days past the digitals.
CUSHION = 300.0


def _forward(reference, side, price):
    return {'Object': 'EquityForwardDeal', 'Reference': reference, 'Currency': 'USD',
            'Equity': 'EQ', 'Discount_Rate': 'USD', 'Payoff_Currency': 'USD', 'Buy_Sell': side,
            'Units': 1.0, 'Forward_Price': price,
            'Maturity_Date': bb.BASE + pd.Timedelta(days=395)}


def _set(reference, collateralised, *deals):
    return {'Instrument': construct_instrument(
        dict(NETTING, Reference=reference, Collateralized=collateralised), {}),
        'Children': [{'Instrument': construct_instrument(deal, {})} for deal in deals]}


def _lifted(first, second):
    """Two collateralised sets, one digital each, and the cushion that lifts the portfolio off the
    relu; the cushion carries no equity delta, so everything the gradient reads is the digitals'."""
    return lambda c: [_set('NS_A', 'True', first), _set('NS_B', 'True', second),
                      _set('NS_CUSH', 'False', _forward('CUSH_L', 'Buy', 0.0),
                           _forward('CUSH_S', 'Sell', CUSHION))]


@pytest.mark.parametrize('barrier_type', list(DIRECTIONS))
def test_the_correction_signs_and_dominates_the_cva_delta_of_a_lifted_portfolio(barrier_type):
    """THE CUSHION MUST LIFT the portfolio clear of zero, the reported delta MUST CARRY the sign the
    digitals' monotone survival gives it, and THE CORRECTION MUST DOMINATE the smooth sensitivity
    read off the same document with the correction suppressed through `Boundary_AAD_Bandwidth`:
    it is at least twice it.

    At the 512 paths run here, seed 1: down +6.25e-4 against a smooth +1.22e-4, up -7.79e-4 against
    -1.14e-4. Measured at 8192 paths, cushion 300: down 3.47x, the AAD +0.000535653 against a CRN
    best of +0.000528546, 1.33% apart on a ladder flat to 2.41%; up 4.07x, 4.43% apart on a
    ladder flat to 6.04%.

    Killing mutations: the discrete barrier's gap signed against its direction - down reads
    -3.80e-4, up +5.50e-4; an undeclared `settles` collapses the correction to 0.60x the smooth
    term down and 1.03x up.
    """
    (h_first, h_second), sign = DIRECTIONS[barrier_type]
    first = _digital('DIG_A', h_first, barrier_type)
    children = _lifted(first, _digital('DIG_B', h_second, barrier_type))
    mtm, cva, live = _run(first, gradient=True, children=children, **PATHS)
    assert mtm.min() >= 0.0, (
        f'the cushion did not lift the portfolio clear of the relu (it spans {mtm.min():+.6g} to '
        f'{mtm.max():+.6g})')
    assert cva > 0.0, f'the lifted portfolio has no exposure to be sensitive to; cva {cva!r}'
    assert sign * live > 0.0, (
        f'{barrier_type}: the CVA delta reads {live:+.6g} where the monotone survival gives it '
        f'the other sign - a gap signed against its direction reads this, as does a branch whose '
        f'settlement does not follow it')
    smooth = _run(first, gradient=True, children=children, bandwidth=SUPPRESSED_BANDWIDTH,
                  **PATHS)[2]
    dominance = abs(live - smooth) / abs(smooth)
    assert dominance >= 2.0, (
        f'the boundary correction is {dominance:.4f}x the smooth sensitivity (reported '
        f'{live:+.8g}, suppressed {smooth:+.8g})')
