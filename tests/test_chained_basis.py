"""`Chained_Basis` — the declared session pair: each partner's ObservedBasis block names the
other, and discovery pulls the partner into the factor universe whenever either side enters,
under every calculation. The declared `Chained_Lag` states where each link binds: a same-row
link (lag 0) enters the graph as an edge — the link simulates first — and a lagged link is
the chain's day boundary and orders nothing, which is what keeps the loop out of the sort.

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
6. THE SAME-ROW ENTRY ORDERS ITS LINK FIRST — the production book enters from the same-row
   side only, and its pulled link is inserted last; positional depth cannot order it because
   the sort emits whole chains within a pass in insertion order. Killed by the lag-0 edge
   dropped — the pre-fix engine emitted the link last and every walk-forward trade died at
   generate — or by the lag declared on the wrong member.
7. A CHAIN THAT LAGS NOWHERE REFUSES — every link same-row is a same-instant loop with no
   member generating a path of its own. Killed by the refusal softened to a skip (the edges
   then hand the sort a nameless cycle).
"""
import json

import pandas as pd
import pytest

import derivus as rf
from derivus import utils

BASE = pd.Timestamp('2026-01-15')


def _world(entry_name, chained=True, partner_of_cme='LBMA_AM.PM.CME', cross_chain=False,
           open_chain=False, no_lag=False):
    """One future on `entry_name`. `chained` pairs the two CME bases; `cross_chain` instead
    pairs LBMA_AM.PM with LBMA_AM.CME.PM — two different BRANCHES of the name tree, so neither
    is the other's positional prefix and only the declaration can pull one from the other;
    `open_chain` leaves the back-pointer off, which must refuse (a chain closes). The lags
    mirror production: the AM basis lags its link (the day boundary), so the PM side's
    same-row link is the one generation edge; `no_lag` omits the day boundary, which must
    refuse (a same-instant loop)."""
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


def test_the_same_row_entry_orders_its_link_first():
    """The production book enters from the same-row side only (PM-session tradables), so its
    link is pulled by the declaration and inserted last. Positional depth cannot order it —
    the sort emits whole chains within a pass in insertion order — so the same-row (lag-0)
    link must be a graph edge. Killed by the edge dropped, or by the day-boundary lag declared
    on the wrong member: the pre-fix engine emitted the link LAST and every walk-forward trade
    died in ChainedBasisModel.generate."""
    order = list(_discover(_world('LBMA_AM.PM.CME')))
    assert CME in order                                # the pull itself, from the same-row side
    assert order.index(CME) < order.index(PM_CME)


def test_a_chain_that_lags_nowhere_refuses():
    """Every link same-row is a same-instant loop — no member generates a path of its own for
    the others to ride. Refuse loud, naming the chain, before the sort refuses its cycle
    namelessly."""
    with pytest.raises(Exception, match='lags nowhere'):
        _discover(_world('LBMA_AM.PM.CME', no_lag=True))
