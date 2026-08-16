"""`Chained_Basis` — the declared session pair: each partner's ObservedBasis block names the
other, and discovery pulls the partner into the factor universe whenever either side enters,
under every calculation. A member whose model routes to ChainedBasisModel is a BRIDGE off its
declared link's finished path, and that read enters the graph as an edge (bridge <- link);
the sort sees no cycle because a chain's source member contributes no edge.

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
6. THE BRIDGE ENTRY ORDERS ITS SOURCE FIRST — the production book enters from the bridge side
   only, and the pulled source is inserted last; positional depth cannot order it because the
   sort emits whole chains within a pass in insertion order. Killed by the edge dropped — the
   pre-fix engine emitted the source last and every walk-forward trade died at generate.
7. A CHAIN OF BRIDGES REFUSES — every member routing to ChainedBasisModel leaves no member
   generating a path of its own; the edges hand the sort the declaration's cycle and its own
   refusal is the guard. Killed by the edge dropped (the cycle dissolves and the misconfig
   simulates).
"""
import json

import pandas as pd
import pytest

import derivus as rf
from derivus import utils

BASE = pd.Timestamp('2026-01-15')


def _world(entry_name, chained=True, partner_of_cme='LBMA_AM.PM.CME', cross_chain=False,
           open_chain=False, all_bridges=False):
    """One future on `entry_name`. `chained` pairs the two CME bases; `cross_chain` instead
    pairs LBMA_AM.PM with LBMA_AM.CME.PM — two different BRANCHES of the name tree, so neither
    is the other's positional prefix and only the declaration can pull one from the other;
    `open_chain` leaves the back-pointer off, which must refuse (a chain closes). The model
    routing mirrors production: the PM-session basis is the ChainedBasisModel bridge, so its
    declared link is a generation edge; `all_bridges` routes BOTH members to the bridge model,
    which must refuse (a chain needs a source)."""
    cme = {'Spot': -7.35}
    cme_pm = {'Spot': -10.65}
    pm_diff = {'Spot': -12.4}
    if chained and not cross_chain:
        cme['Chained_Basis'] = partner_of_cme
        if partner_of_cme == 'LBMA_AM.PM.CME' and not open_chain:
            cme_pm['Chained_Basis'] = 'LBMA_AM.CME'
    if cross_chain:
        pm_diff['Chained_Basis'] = 'LBMA_AM.CME'
        cme['Chained_Basis'] = 'LBMA_AM.PM'
    filters = {'ObservedBasis': [[['ID', 'LBMA_AM.PM.CME'], 'ChainedBasisModel']]}
    if all_bridges:
        filters['ObservedBasis'].append([['ID', 'LBMA_AM.CME'], 'ChainedBasisModel'])
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
            'Model Configuration': {'.ModelParams': {'modeldefaults': {},
                                                     'modelfilters': filters}},
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


def test_the_declaration_pulls_the_partner():
    dependent = _discover(_world('LBMA_AM.CME'))
    assert PM_CME in dependent                           # not positionally required by the entry
    order = list(dependent)
    assert order.index(CME) < order.index(PM_CME)        # depth orders the source (1) before
                                                         # the bridge (2); no cycle in the sort


def test_the_pull_crosses_branches_both_ways():
    """LBMA_AM.PM ↔ LBMA_AM.CME sit on different branches of the name tree — neither is the
    other's prefix, so only the declaration can pull one from the other, in BOTH directions.
    (The production pair CME ↔ PM.CME is itself cross-branch under the ruled naming, so the
    declaration is load-bearing both ways there too — gate 1 covers it.)"""
    pm_diff = utils.Factor('ObservedBasis', ('LBMA_AM', 'PM'))
    fwd = _discover(_world('LBMA_AM.PM', cross_chain=True))
    assert CME in fwd                                    # declared pull, no prefix relation
    rev = _discover(_world('LBMA_AM.CME', cross_chain=True))
    assert pm_diff in rev


def test_the_omitted_field_changes_nothing():
    with_field = set(_discover(_world('LBMA_AM.CME')))
    without = _discover(_world('LBMA_AM.CME', chained=False))
    assert PM_CME not in without
    # the pulled partner enters WITH its own positional chain (the PM-branch prefix)
    assert with_field - set(without) == {
        PM_CME, utils.Factor('ObservedBasis', ('LBMA_AM', 'PM'))}


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
    assert dependent[PM_CME] is not None
    assert dependent[PM_CME] >= BASE


def test_the_bridge_entry_orders_its_source_first():
    """The production book enters from the BRIDGE side only (PM-session tradables), so the
    source is pulled by the declaration and inserted last. Positional depth cannot order it —
    the sort emits whole chains within a pass in insertion order — so the bridge's read of its
    link's finished path must be a graph edge. Killed by the edge dropped: the pre-fix engine
    emitted the source LAST and every walk-forward trade died in ChainedBasisModel.generate."""
    order = list(_discover(_world('LBMA_AM.PM.CME')))
    assert CME in order                                  # the pull itself, from the bridge side
    assert order.index(CME) < order.index(PM_CME)


def test_a_chain_of_bridges_refuses():
    """Every member routing to ChainedBasisModel leaves no member generating a path of its own,
    and the declared edges hand the sort the declaration's cycle — its own refusal is the
    guard, so this gate pins that the misconfig refuses rather than generating garbage."""
    with pytest.raises(RuntimeError, match='cyclic'):
        _discover(_world('LBMA_AM.PM.CME', all_bridges=True))
