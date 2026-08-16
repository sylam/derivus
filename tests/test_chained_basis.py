"""`Chained_Basis` — the declared session pair: each partner's ObservedBasis block names the
other, and discovery pulls the partner into the factor universe whenever either side enters,
under every calculation. Inclusion only: ordering still comes from the positional name chain,
so the sort sees no cycle even though the declaration spells one.

Gates and their killing mutations:

1. EITHER SIDE PULLS THE OTHER — a book referencing only the composed AM name discovers the PM
   partner (and the reverse), positioned after its positional parent. Killed by the field read
   dropped, or the pull wired one-directional.
2. THE OMITTED FIELD CHANGES NOTHING — a world without Chained_Basis discovers the identical
   universe, key for key. Killed by an unconditional partner add or a truthiness slip.
3. THE DECLARATION IS VALIDATED LOUD — a partner on a foreign primary, or a self-reference,
   raises naming both factors. Killed by the check softened to a skip.
4. THE MUTUAL POINTERS TERMINATE — the pair's blocks point at each other; discovery must not
   recurse forever (gates 1-2 hang rather than fail if this breaks, so the guard IS the gate).
5. THE PARTNER INHERITS A HORIZON — the pulled factor carries a max date (its parent's), so
   construction does not die on a dateless factor.
"""
import json

import pandas as pd
import pytest

import derivus as rf
from derivus import utils

BASE = pd.Timestamp('2026-01-15')


def _world(entry_name, chained=True, partner_of_cme='LBMA_AM.CME.PM', cross_chain=False,
           open_chain=False):
    """One future on `entry_name`. `chained` pairs the two CME bases; `cross_chain` instead
    pairs LBMA_AM.PM with LBMA_AM.CME.PM — two different BRANCHES of the name tree, so neither
    is the other's positional prefix and only the declaration can pull one from the other;
    `open_chain` leaves the back-pointer off, which must refuse (a chain closes)."""
    cme = {'Spot': -7.35}
    cme_pm = {'Spot': -10.65}
    pm_diff = {'Spot': -12.4}
    if chained and not cross_chain:
        cme['Chained_Basis'] = partner_of_cme
        if partner_of_cme == 'LBMA_AM.CME.PM' and not open_chain:
            cme_pm['Chained_Basis'] = 'LBMA_AM.CME'
    if cross_chain:
        pm_diff['Chained_Basis'] = 'LBMA_AM.CME.PM'
        cme_pm['Chained_Basis'] = 'LBMA_AM.PM'
    return {'Calc': {
        'Calculation': {
            'Object': 'CreditMonteCarlo', 'Base_Date': {'.Timestamp': '2026-01-15'},
            'Currency': 'USD', 'Batch_Size': 64, 'Simulation_Batches': 1, 'Random_Seed': 1,
            'Deflation_Interest_Rate': 'USD-SOFR', 'Time_Grid': '0d 1m(1m)'},
        'Deals': {'Tag_Titles': '', 'Reference': 'chained', 'Deals': {'Children': [{
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
                'ObservedBasis.LBMA_AM.CME.PM': cme_pm,
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
CME_PM = utils.Factor('ObservedBasis', ('LBMA_AM', 'CME', 'PM'))


def test_the_declaration_pulls_the_partner():
    dependent = _discover(_world('LBMA_AM.CME'))
    assert CME_PM in dependent                           # not positionally required by the entry
    order = list(dependent)
    assert order.index(CME) < order.index(CME_PM)        # positional order, no cycle in the sort


def test_an_alias_pulls_its_source_and_orders_after_it():
    """`Alias_Of` is the second inclusion declaration: a book naming only the PM-session
    composed name must discover the alias's SOURCE — with the source's own chain closure
    following (the CME pair enters whole) — and the alias must sort AFTER its source, because
    unlike the chained loop this declaration IS an ordering edge (a source never reads its
    alias, so the edge is acyclic). Killed by dropping the pull (the source never enters) or
    by inclusion-without-edge (insertion order could put the alias first)."""
    cfg = _world('LBMA_AM.PM.CME')
    pf = cfg['Calc']['MergeMarketData']['ExplicitMarketData']['Price Factors']
    pf['ObservedBasis.LBMA_AM.PM.CME'] = {'Spot': -10.65, 'Alias_Of': 'LBMA_AM.CME.PM'}
    dependent = _discover(cfg)
    alias = utils.Factor('ObservedBasis', ('LBMA_AM', 'PM', 'CME'))
    assert CME_PM in dependent and CME in dependent      # source + its closed chain, whole
    order = list(dependent)
    assert order.index(CME_PM) < order.index(alias)      # the alias generates after its source

    pf['ObservedBasis.LBMA_AM.PM.CME']['Alias_Of'] = 'LBMA_AM.PM.CME'
    with pytest.raises(Exception, match='names itself'):
        _discover(cfg)


def test_the_pull_crosses_branches_both_ways():
    """LBMA_AM.PM ↔ LBMA_AM.CME.PM sit on different branches of the name tree — neither is the
    other's prefix, so only the declaration can pull one from the other. The reverse direction
    on the nested CME pair would be a PLACEBO (the prefix enters positionally regardless)."""
    pm_diff = utils.Factor('ObservedBasis', ('LBMA_AM', 'PM'))
    fwd = _discover(_world('LBMA_AM.PM', cross_chain=True))
    assert CME_PM in fwd and CME in fwd                  # partner + its positional intermediate
    rev = _discover(_world('LBMA_AM.CME.PM', cross_chain=True))
    assert pm_diff in rev


def test_the_omitted_field_changes_nothing():
    with_field = set(_discover(_world('LBMA_AM.CME')))
    without = _discover(_world('LBMA_AM.CME', chained=False))
    assert CME_PM not in without
    assert with_field - set(without) == {CME_PM}


@pytest.mark.parametrize('bad,match', [
    ('OTHER_ROOT.CME', 'same primary'),
    ('LBMA_AM.CME', 'different factor')])
def test_the_declaration_is_validated_loud(bad, match):
    with pytest.raises(Exception, match=match):
        _discover(_world('LBMA_AM.CME', partner_of_cme=bad))


def test_a_chain_must_close():
    """The word is CHAINED: the declarations walk back to their start. An open link is the
    linked-parent family (BasisLinkedSpotModel) and must refuse here, naming the break."""
    with pytest.raises(Exception, match='does not close'):
        _discover(_world('LBMA_AM.CME', open_chain=True))


def test_the_partner_inherits_a_horizon():
    dependent = _discover(_world('LBMA_AM.CME'))
    assert dependent[CME_PM] is not None
    assert dependent[CME_PM] >= BASE
