"""`Chained_Basis` — the declared session pair: each partner's ObservedBasis block names the
other, and discovery pulls the partner into the factor universe whenever either side enters,
under every calculation. The declared `Chained_Lag` states where each link binds: a same-row
link (lag 0) enters the graph as an edge — the link simulates first — and a lagged link is
the chain's day boundary and orders nothing, which is what keeps the loop out of the sort.

Each gate is a credit Monte Carlo document through `Context.load_json`, read at its own factor
discovery - the walk `run_job` takes before any model is asked for.
"""
import json

import pandas as pd
import pytest

import derivus as rf
from derivus import utils

BASE = pd.Timestamp('2026-01-15')


def _world(entry_name, chained=True, partner_of_cme='LBMA_AM.PM.CME', cross_chain=False,
           open_chain=False, no_lag=False):
    """One future on `entry_name`. `chained` pairs the two CME bases; `cross_chain` instead pairs
    LBMA_AM.PM with LBMA_AM.CME.PM - different BRANCHES of the name tree, so only the declaration
    can pull one from the other; `open_chain` leaves the back-pointer off (must refuse). The lags
    mirror production - the AM basis lags its link, so the PM side's same-row link is the one
    generation edge; `no_lag` omits the day boundary (must refuse)."""
    cme = {'Spot': -7.35}
    cme_pm = {'Spot': -10.65}
    pm_diff = {'Spot': -12.4}
    if chained and not cross_chain:
        cme['Chained_Basis'] = partner_of_cme
        if not no_lag:
            cme['Chained_Lag'] = 1
        if partner_of_cme == 'LBMA_AM.PM.CME' and not open_chain:
            cme_pm['Chained_Basis'] = 'LBMA_AM.CME'
    if cross_chain:
        pm_diff['Chained_Basis'] = 'LBMA_AM.CME'
        pm_diff['Chained_Lag'] = 1
        cme['Chained_Basis'] = 'LBMA_AM.PM'
    return {'Calc': {
        'Calculation': {
            'Object': 'CreditMonteCarlo', 'Base_Date': {'.Timestamp': '2026-01-15'},
            'Currency': 'USD', 'Batch_Size': 64, 'Simulation_Batches': 1, 'Random_Seed': 1,
            'Deflation_Interest_Rate': 'USD-SOFR', 'Time_Grid': '0d 1m(1m)'},
        'Deals': {'Reference': 'chained', 'Deals': {'Children': [{
            'Instrument': {'.Deal': {'Object': 'NettingCollateralSet', 'Reference': 'NS',
                                     'Netted': 'True', 'Collateralized': 'False'}},
            'Children': [{'Instrument': {'.Deal': {
                'Object': 'CommodityFutureDeal', 'Reference': 'FUT', 'Commodity': entry_name,
                'Currency': 'USD', 'Repo_Rate': 'USD-SOFR', 'Carry': 'PLATINUM_CARRY',
                'Maturity_Date': {'.Timestamp': '2026-10-15'}}}}]}]}},
        'MergeMarketData': {'ExplicitMarketData': {
            'System Parameters': {'Base_Currency': 'USD',
                                  'Base_Date': {'.Timestamp': '2026-01-15'}},
            'Model Configuration': {'.ModelParams': {'modeldefaults': {}, 'modelfilters': {}}},
            'Price Factors': {
                'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD-SOFR',
                               'Spot': 1.0},
                'InterestRate.USD-SOFR': {
                    'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                    'Curve': {'.Curve': {'meta': [], 'data': [[0.0, 0.04], [5.0, 0.03]]}}},
                'CommodityPrice.LBMA_AM': {
                    'Spot': 1638.9, 'Currency': 'USD', 'Interest_Rate': 'USD-SOFR',
                    'Forward_Rate': 'PLATINUM_CARRY'},
                'ObservedBasis.LBMA_AM.PM': pm_diff,
                'ObservedBasis.LBMA_AM.CME': cme,
                'ObservedBasis.LBMA_AM.PM.CME': cme_pm,
                'ForwardRate.PLATINUM_CARRY': {'Currency': 'USD', 'Curve': {'.Curve': {
                    'meta': [], 'data': [[46213.0, 0.031], [46395.0, 0.033]]}}}},
            'Price Models': {},
            'Correlations': {}}}}}


def _discover(cfg):
    cx = rf.Context(path_transform={}, file_transform={})
    cx.load_json((json.dumps(cfg), 'chained_basis.json'))
    c = cx.current_cfg
    return c.discover_factors(c.deals['Calculation'], BASE, '0d 1m(1m)')[0]


CME = utils.Factor('ObservedBasis', ('LBMA_AM', 'CME'))
PM_CME = utils.Factor('ObservedBasis', ('LBMA_AM', 'PM', 'CME'))


def test_the_declaration_pulls_the_partner_and_the_omitted_field_pulls_nothing():
    """Either side pulls the other, after its positional parent and with a horizon of its own -
    across BRANCHES of the name tree too, where neither is the other's prefix and only the
    declaration can pull, in both directions. The same book without the field is the same
    universe less the partner and its own positional chain.

    Killing mutation: the `Chained_Basis` read dropped - the partner never enters.
    """
    dependent = _discover(_world('LBMA_AM.CME'))
    assert PM_CME in dependent                           # not positionally required by the entry
    order = list(dependent)
    assert order.index(CME) < order.index(PM_CME)        # depth orders the source (1) before
                                                         # the bridge (2); no cycle in the sort
    assert dependent[PM_CME] is not None and dependent[PM_CME] >= BASE

    without = _discover(_world('LBMA_AM.CME', chained=False))
    assert set(dependent) - set(without) == {
        PM_CME, utils.Factor('ObservedBasis', ('LBMA_AM', 'PM'))}

    pm_diff = utils.Factor('ObservedBasis', ('LBMA_AM', 'PM'))
    assert CME in _discover(_world('LBMA_AM.PM', cross_chain=True))
    assert pm_diff in _discover(_world('LBMA_AM.CME', cross_chain=True))


@pytest.mark.parametrize('world,match', [
    (dict(partner_of_cme='OTHER_ROOT.CME'), 'same primary'),
    (dict(partner_of_cme='LBMA_AM.CME'), 'different factor'),
    (dict(open_chain=True), 'does not close'),
    (dict(no_lag=True), 'lags nowhere')])
def test_a_chain_that_cannot_close_refuses_by_name(world, match):
    """A foreign primary or a self-reference, an open link (the linked-parent family, not a chain)
    and a chain that lags nowhere (a same-instant loop the sort would refuse namelessly) each
    refuse naming the chain.

    Killing mutation: each refusal softened to ending the walk.
    """
    entry = 'LBMA_AM.PM.CME' if world.get('no_lag') else 'LBMA_AM.CME'
    with pytest.raises(Exception, match=match):
        _discover(_world(entry, **world))


def test_the_same_row_entry_orders_its_link_first():
    """The production book enters from the same-row side only, so its link is pulled last and
    positional depth cannot order it - the sort emits whole chains within a pass in insertion
    order. The lag-0 link must therefore be a graph edge: without it the engine emits the link
    LAST and every walk-forward trade dies in ChainedBasisModel.generate.

    Killing mutation: the lag-0 edge dropped - the link sorts after its bridge."""
    order = list(_discover(_world('LBMA_AM.PM.CME')))
    assert CME in order                                # the pull itself, from the same-row side
    assert order.index(CME) < order.index(PM_CME)
