"""The portfolio where the boundary correction IS the sensitivity.

`test_boundary_pricer_events.py` gates the correction end to end, but on its fixtures the boundary
term is a small fraction of the reported gradient - 2.4% on the live-exposure one - so a defect in
the term moves the total by little. This portfolio is authored so the term dominates.

THE DEAL IS A DIGITAL. A discretely monitored knock-out BINARY struck at ~zero is worth
`Cash_Payoff` times the probability it never crossed, so its spot sensitivity is almost entirely the
flux of paths across the barrier; a knock-out CALL carries a vanilla's intrinsic delta the
correction has to compete with (correction over smooth term 0.56 / 0.83 / 2.07 against 3.04 for
the H=95 monthly digital, at 1024 paths).

BOTH BARRIER DIRECTIONS, mirrored about the spot. The digital's survival is monotone in the spot, so
the CVA delta's SIGN is known before anything is run: positive below a down barrier, negative under
an up one. The smooth term alone carries the OPPOSITE sign in both, so the sign is the correction's,
and a gap signed the wrong way for its direction reports the wrong one.

TWO COLLATERALISED SETS, at different barriers, so the two corrections add rather than cancel and
neither set's gross-to-net chain can stand in for the other's. No minimum transfer amount, because
that registers a SECOND decision and the subject here is the pricer event.

LIFTED OFF THE RELU. A DELTA-FREE cash cushion in its own uncollateralised set lifts the reported
portfolio clear of the CVA's kink. Off the relu every counterfactual is scored at full weight and
what a collateralised set is worth turns on the cash it has already paid - so a counterfactual
whose SETTLEMENT does not follow its branch prices the wrong exposure. With the settlement
undeclared the lifted portfolio reported -0.00010726372 where its own CRN oracle wants +0.00034761669.
Parked ON the relu instead (the un-lifted portfolio spans -16.6 to +14.4), the same two sets read
the correction at 3.71x the smooth term and the AAD 1.47% from its CRN ladder at 8192 paths (1.14%
at seed 2), where suppressing the correction reads 363.78% - and the undeclared settlement is
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
from test_boundary_pricer_events import NETTING, _run

#: The kernel selects nothing at this bandwidth, so the correction is an exact zero.
SUPPRESSED_BANDWIDTH = 1e-12

#: 512 paths, one batch: the smallest count at which both directions hold the dominance clear of
#: its floor over seeds 1-3 (down 10.17 / 9.87 / 4.98x, up 9.31 / 7.60 / 7.14x); at 256 the up
#: digital reads 4.00x at seed 2. At 1024 paths: 8.78x and 8.64x; at 8192 the down one 7.38x.
PATHS = dict(batch=512, mcmc=64, batches=1)

MONTHLY = [bb.BASE + pd.Timedelta(days=d) for d in range(30, 366, 30)]

#: Each direction's two barriers, mirrored about the spot, and the sign its CVA delta must carry.
DIRECTIONS = {'Down_And_Out': ((95.0, 90.0), 1.0), 'Up_And_Out': ((105.0, 110.0), -1.0)}

#: A delta-free cash cushion: buy the forward struck at zero, sell the one struck at CUSHION, in
#: its own uncollateralised set. Worth CUSHION*DF every scenario with zero equity delta.
CUSHION = 300.0


def _digital(reference, barrier, barrier_type):
    """A discretely monitored knock-out binary struck at ~zero: worth `Cash_Payoff` times the
    probability it never crossed, so its spot delta is the barrier flux and almost nothing else."""
    return {'Object': 'EquityBarrierBinaryOption', 'Reference': reference, 'Currency': 'USD',
            'Payoff_Currency': 'USD', 'Equity': 'EQ', 'Dividends': 'EQ', 'Discount_Rate': 'USD',
            'Equity_Volatility': 'EQ', 'Buy_Sell': 'Buy', 'Option_Type': 'Call',
            'Strike_Price': 1e-6, 'Expiry_Date': bb.BASE + pd.Timedelta(days=365), 'Units': 1.0,
            'Barrier_Type': barrier_type, 'Barrier_Price': barrier, 'Cash_Payoff': 10.0,
            'Barrier_Dates': list(MONTHLY), 'Settlement_Date': ''}


def _forward(reference, side, price):
    return {'Object': 'EquityForwardDeal', 'Reference': reference, 'Currency': 'USD',
            'Equity': 'EQ', 'Discount_Rate': 'USD', 'Payoff_Currency': 'USD', 'Buy_Sell': side,
            'Units': 1.0, 'Forward_Price': price,
            'Maturity_Date': bb.BASE + pd.Timedelta(days=365)}


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
    read off the same document with the correction suppressed through `Boundary_AAD_Bandwidth`,
    floored at 3.0.

    At the 512 paths run here, seed 1: down +4.65e-4 against a smooth -5.07e-5, up -3.41e-4 against
    +4.10e-5. Measured at 8192 paths, down, cushion 300: 7.38x (6.62x at seed 2); the AAD
    +0.00032924550 against a CRN best of +0.00035978764, 9.28% on a ladder flat to 3.59%, the
    estimator's residual here (14.80% at seed 2, 10.79% at 16384).

    Killing mutation: the discrete barrier's gap signed against its direction - up reads +4.23e-4
    (+4.90 / +4.92e-4 at seeds 2 and 3), down -5.66e-4; an undeclared `settles` flips both too.
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
    assert dominance >= 3.0, (
        f'the boundary correction is {dominance:.4f}x the smooth sensitivity (reported '
        f'{live:+.8g}, suppressed {smooth:+.8g})')
