"""`GBMAssetPriceTSModelParameters`: a vol surface's ATM column as the instantaneous vol curve
`GBMAssetPriceTSModelImplied` integrates, built by a job document's bootstrap and read back as the
process reads it - piecewise linear between knots, flat outside, its square integrated exactly."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

import derivus
from derivus import utils
from derivus.config import CustomJsonEncoder


#: A Skew surface's smile off its ATM vol, and the level it is referenced to.
SKEW = {'ATM_Ref': 100.0, 's': -0.15, 'L': 0.30, 'R': 0.20, 'C': -0.35, 'D': 0.35, 'lam': 0.5,
        'rho': 0.5}


def surface(expiries, atm, surface_type='Explicit', axis=lambda m: m):
    """A `surface_type` surface whose ATM column is `atm` beneath a smile 20% up at moneyness 0.8
    and 1.2, each node placed on the type's own axis by `axis`."""
    return {'Surface_Type': surface_type, 'Moneyness_Rule': 'Sticky_Moneyness',
            'Surface': utils.Curve([], [[axis(m), t, v * (1.0 + abs(m - 1.0))]
                                        for t, v in zip(expiries, atm) for m in (0.8, 1.0, 1.2)])}


def skew(expiries, atm):
    """A Skew surface whose ATM vol is `atm`, sloping and curving off it as `SKEW` says."""
    curve = lambda values: utils.Curve([], [list(row) for row in zip(expiries, values)])
    return dict({name: curve([x] * len(atm)) for name, x in SKEW.items()}, Surface_Type='Skew',
                Moneyness_Rule='Sticky_Moneyness', ATM_Vol=curve(atm))


def built(expiries, atm, vol=surface):
    """The `Vol` knots a bootstrap writes off the equity surface `vol` makes of the ATM column
    `atm`."""
    base = {'.Timestamp': '2026-01-15'}
    document = {'Calc': {
        'Calculation': {'Object': 'BaseValuation', 'Base_Date': base, 'Currency': 'USD'},
        'Deals': {'Reference': 'gbm', 'Deals': {'Children': []}},
        'MergeMarketData': {'MarketDataFile': '', 'ExplicitMarketData': {
            'System Parameters': {'Base_Currency': 'USD', 'Base_Date': base},
            'Price Factors': {'EquityPriceVol.EQ': vol(expiries, atm)},
            'Market Prices': {'GBMAssetPriceTSModelPrices.EQ': {
                'instrument': {'Asset_Price_Volatility': 'EQ', 'Quote_Sensitivity': 'Yes'}}},
            'Bootstrapper Configuration': {
                'GBMAssetPriceTSModelParameters': {'Prices': 'GBMAssetPriceTSModel'}}}}}}
    context = derivus.Context()
    context.load_json((json.dumps(document, cls=CustomJsonEncoder), 'gbm'))
    context.bootstrap()
    return context.current_cfg.params['Price Factors']['GBMAssetPriceTSModelParameters.EQ']['Vol'].array


def implied(knots, t):
    """The vol the process implies to `t`: `sqrt(V(t)/t)`, `V` the exact integral of the square."""
    grid = np.union1d([0.0, t], knots[knots[:, 0] < t, 0])
    s = np.interp(grid, knots[:, 0], knots[:, 1])
    return np.sqrt(np.sum(np.diff(grid) / 3.0 * (s[:-1] ** 2 + s[:-1] * s[1:] + s[1:] ** 2)) / t)


@pytest.mark.parametrize('atm', [[0.10, 0.30, 0.25, 0.26], [0.10, 0.20, 0.25, 0.30],
                                 [0.30, 0.25, 0.22, 0.20]], ids=['humped', 'rising', 'falling'])
def test_the_curve_reprices_every_atm_vol_it_was_built_from(atm):
    """Every column whose total variance rises - the humped one included, which a curve knotted
    only at the expiries cannot fit, its 1y vol forcing more variance onto the 2y than the 2y
    quote holds - is matched at every expiry.

    Killing mutations: the column's average vol written where the process reads an instantaneous
    one (the rising column's 1y misses by 5.9 vol points); the curve knotted at the expiries alone
    (the humped column's 2y is repaired to 30.4%)."""
    expiries = [0.25, 1.0, 2.0, 5.0]
    knots = built(expiries, atm)
    assert [implied(knots, t) for t in expiries] == pytest.approx(atm, abs=1e-14)


def test_a_falling_total_variance_is_repaired_to_a_zero_vol():
    """30%, 20%, 10% at 6m, 1y, 2y: total variance falls, so each later step is raised to the least
    variance it can reach - its vol zero after a day's ramp - and the curve misses those quotes.

    Killing mutation: the repair dropped - the walk writes negative vols."""
    knots = built([0.5, 1.0, 2.0], [0.30, 0.20, 0.10])
    assert knots[1:, 1].tolist() == [0.0] * 4
    assert implied(knots, 2.0) == pytest.approx(0.30 * np.sqrt((0.5 + 1.0 / 365.0 / 3.0) / 2.0), rel=1e-12)


@pytest.mark.parametrize('vol', [
    lambda e, a: surface(e, a, 'Malz', np.log),
    lambda e, a: surface(e, a, 'Relative_Forward', lambda m: 1.0 / m - 1.0),
    skew], ids=['Malz', 'Relative_Forward', 'Skew'])
def test_a_surface_is_read_at_the_atm_coordinate_of_its_type(vol):
    """A hand-authored surface and its Explicit twin, one ATM column under a smile, integrate one
    vol curve, each read where a strike at the forward sits on its own axis: a Malz and a
    Relative_Forward surface node for node at log(F/K) 0 and (K - F)/F 0, a Skew one at its
    log-moneyness 0, its ATM vol, on its own expiries.

    Killing mutations: the column read at moneyness 1 whatever the type, a wing of each smile, the
    Malz and Relative_Forward curves 20% high (6.6 vol points) and the Skew one 29% (3.0 points);
    the Skew surface read off the placeholder grid its factor carries, one knot at 1%."""
    expiries, atm = [0.25, 1.0, 2.0, 5.0], [0.10, 0.20, 0.25, 0.30]
    np.testing.assert_allclose(built(expiries, atm, vol), built(expiries, atm), rtol=0.0,
                               atol=1e-12)
