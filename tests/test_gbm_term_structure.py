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


#: Each currency's flat rate on the quanto book.
RATES = {'USD': 0.04, 'EUR': 0.02, 'ZAR': 0.08}


def quanto(vol, currency):
    """A USD CVA on a year's forward on an equity in `currency`, the equity and `FxRate.<currency>`
    both bootstrapped onto GBMAssetPriceTSModelImplied, their correlation 0.5 named as a quanto deal
    names it: `Correlation.EquityPrice.<vol>/FxRate.<sorted pair>`."""
    base, pair = {'.Timestamp': '2026-01-15'}, '.'.join(sorted([currency, 'USD']))
    flat = lambda rate: utils.Curve([], [[0.0, rate], [10.0, rate]])
    expiries = [0.25, 1.0, 2.0, 5.0]
    factors = dict({'FxRate.' + ccy: {'Domestic_Currency': 'USD', 'Interest_Rate': ccy,
                                      'Spot': 1.0 if ccy == 'USD' else 0.5} for ccy in RATES}, **{
        'InterestRate.' + ccy: {'Currency': ccy, 'Day_Count': 'ACT_365', 'Sub_Type': None,
                                'Curve': flat(rate)} for ccy, rate in RATES.items()}, **{
        'EquityPrice.EQ': {'Spot': 100.0, 'Currency': currency, 'Interest_Rate': currency,
                           'Issuer': '', 'Respect_Default': 'No', 'Jump_Level': 0.0},
        'DividendRate.EQ': {'Currency': currency, 'Floor': None, 'Curve': flat(0.0)},
        'EquityPriceVol.' + vol: surface(expiries, [0.30] * 4),
        'FXVol.' + pair: surface(expiries, [0.20] * 4),
        'Correlation.EquityPrice.{}/FxRate.{}'.format(vol, pair): {'Value': 0.5},
        'SurvivalProb.CPTY': {'Recovery_Rate': 0.4,
                              'Curve': utils.Curve([], [[0.0, 0.0], [10.0, 0.4]])}})
    forward = {'Object': 'EquityForwardDeal', 'Reference': 'FWD', 'Equity': 'EQ',
               'Currency': currency, 'Discount_Rate': currency, 'Payoff_Currency': currency,
               'Buy_Sell': 'Buy', 'Units': 1000.0, 'Forward_Price': 90.0,
               'Maturity_Date': {'.Timestamp': '2027-01-15'}}
    return json.dumps({'Calc': {
        'Calculation': {
            'Object': 'CreditMonteCarlo', 'Base_Date': base, 'Currency': 'USD',
            'Time_grid': '1d 3m(3m) 12m', 'Batch_Size': 4096, 'Simulation_Batches': 1,
            'Random_Seed': 1, 'MCMC_Simulations': 1, 'Deflation_Interest_Rate': 'USD',
            'Credit_Valuation_Adjustment': {
                'Calculate': 'Yes', 'Counterparty': 'CPTY', 'Deflate_Stochastically': 'No',
                'Stochastic_Hazard_Rates': 'No', 'Gradient': 'No'}},
        'Deals': {'Reference': 'quanto', 'Deals': {'Children': [{
            'Instrument': {'.Deal': {'Object': 'NettingCollateralSet', 'Reference': 'NS1',
                                     'Netted': 'True', 'Collateralized': 'False'}},
            'Children': [{'Instrument': {'.Deal': forward}}]}]}},
        'MergeMarketData': {'MarketDataFile': '', 'ExplicitMarketData': {
            'System Parameters': {'Base_Currency': 'USD', 'Base_Date': base},
            'Price Factors': factors,
            'Market Prices': {
                'GBMAssetPriceTSModelPrices.EQ': {'instrument': {'Asset_Price_Volatility': vol}},
                'GBMAssetPriceTSModelPrices.' + currency: {
                    'instrument': {'Asset_Price_Volatility': pair}}},
            'Bootstrapper Configuration': {
                'GBMAssetPriceTSModelParameters': {'Prices': 'GBMAssetPriceTSModel'}},
            'Model Configuration': {'.ModelParams': {'modeldefaults': {
                'EquityPrice': 'GBMAssetPriceTSModelImplied',
                'FxRate': 'GBMAssetPriceTSModelImplied'}, 'modelfilters': {}}}}}}},
        cls=CustomJsonEncoder), 'quanto'


def test_the_quanto_correlation_is_read_as_a_quanto_deal_reads_it():
    """A surface named `EQ` or `EQ.EUR` is one market, one CVA to the bit; a rand equity reads its
    pair's correlation with the sign a quanto deal reads it with, -0.5.

    Killing mutations: the currency taken off the surface name's last token (`EQ` misses its
    correlation and the run refuses); the sorted pair read unsigned (+0.5)."""
    cva = []
    for vol in ('EQ', 'EQ.EUR'):
        context = derivus.Context().load_json(quanto(vol, 'EUR'))
        context.bootstrap()
        assert context.current_cfg.params['Price Factors'][
            'GBMAssetPriceTSModelParameters.EQ']['Quanto_FX_Correlation'] == 0.5
        cva.append(context.run_job()[1]['Results']['cva'])
    assert cva[0] == cva[1]

    rand = derivus.Context().load_json(quanto('EQ', 'ZAR'))
    rand.bootstrap()
    assert rand.current_cfg.params['Price Factors'][
        'GBMAssetPriceTSModelParameters.EQ']['Quanto_FX_Correlation'] == -0.5
