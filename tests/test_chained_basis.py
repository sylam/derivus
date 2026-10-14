"""`Chained_Basis` — the declared session pair: each partner's ObservedBasis block names the
other, and discovery pulls the partner into the factor universe whenever either side enters,
under every calculation. The declared `Chained_Lag` states where each link binds: a same-row
link (lag 0) enters the graph as an edge — the link simulates first — and a lagged link is
the chain's day boundary and orders nothing, which is what keeps the loop out of the sort. The
positional chain every link rides is one registry, `utils.NESTED`, which discovery and the resolver
both read.

Each chained-basis gate is a credit Monte Carlo document through `Context.load_json`, read at its
own factor discovery - the walk `run_job` takes before any model is asked for; the registry's is a
one-deal base valuation per head, described and priced, and a composed FX rate and equity are each
priced beside their primary.
"""
import json
from types import SimpleNamespace

import pandas as pd
import pytest

import derivus as rf
import test_position_scaling as trials
import trial_rates
from derivus import instruments, utils

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


def test_a_composed_fx_link_orders_its_chain_before_the_factor_that_reads_it():
    """A commodity priced in a composed FX rate - `Currency: ZAR.ONSHORE`, the ZAR spot plus an
    ObservedBasis tail - reaches that rate as a LINK, not as a deal's own factor. The head keeps the
    curve its block names, the commodity follows the chain's last link, and the full two-part name,
    being virtual, is no factor at all.

    Killing mutation: the composed-link branch dropped - the head re-registers with no dependants
    and the commodity's only edge is to the dropped full name, so it sorts BEFORE the FX rate and
    the basis it converts through."""
    world = _world('LBMA_AM.CME', chained=False)
    factors = world['Calc']['MergeMarketData']['ExplicitMarketData']['Price Factors']
    factors['CommodityPrice.LBMA_AM']['Currency'] = 'ZAR.ONSHORE'
    factors['FxRate.ZAR'] = {'Domestic_Currency': 'USD', 'Interest_Rate': 'ZAR-JIBAR',
                             'Spot': 0.055}
    factors['InterestRate.ZAR-JIBAR'] = {
        'Currency': 'ZAR', 'Day_Count': 'ACT_365', 'Sub_Type': None,
        'Curve': {'.Curve': {'meta': [], 'data': [[0.0, 0.07], [5.0, 0.075]]}}}
    factors['ObservedBasis.ZAR.ONSHORE'] = {'Spot': 0.0001}
    order = list(_discover(world))
    curve, head = utils.Factor('InterestRate', ('ZAR-JIBAR',)), utils.Factor('FxRate', ('ZAR',))
    tail = utils.Factor('ObservedBasis', ('ZAR', 'ONSHORE'))
    commodity = utils.Factor('CommodityPrice', ('LBMA_AM',))
    assert utils.Factor('FxRate', ('ZAR', 'ONSHORE')) not in order
    assert order.index(curve) < order.index(head) < order.index(tail) < order.index(commodity)


def test_the_order_factors_draw_in_is_the_graph_s_not_the_book_s():
    """Two futures on two spots of one depth, booked in both orders, discover the same factor
    order: the random rows each simulated factor reads are assigned in that order, so a book's
    draws cannot depend on how its deals happen to be listed.

    Killing mutation: the sort visiting each pass in insertion order - the two books then order
    the spots as each lists them."""
    def two(first, second):
        world = _world('LBMA_AM.CME', chained=False)
        factors = world['Calc']['MergeMarketData']['ExplicitMarketData']['Price Factors']
        factors['CommodityPrice.ALUMINIUM'] = dict(factors['CommodityPrice.LBMA_AM'], Spot=2400.0)
        children = world['Calc']['Deals']['Deals']['Children'][0]['Children']
        future = children[0]['Instrument']['.Deal']
        children[:] = [{'Instrument': {'.Deal': dict(future, Reference=name, Commodity=name)}}
                       for name in (first, second)]
        return list(_discover(world))
    assert two('LBMA_AM', 'ALUMINIUM') == two('ALUMINIUM', 'LBMA_AM')


def test_the_same_row_entry_orders_its_link_first():
    """The production book enters from the same-row side only, so its link is pulled last and
    positional depth cannot order it - the sort emits whole chains within a pass in insertion
    order. The lag-0 link must therefore be a graph edge: without it the engine emits the link
    LAST and every walk-forward trade dies in ChainedBasisModel.generate.

    Killing mutation: the lag-0 edge dropped - the link sorts after its bridge."""
    order = list(_discover(_world('LBMA_AM.PM.CME')))
    assert CME in order                                # the pull itself, from the same-row side
    assert order.index(CME) < order.index(PM_CME)


#: Per head the chain registry names: a trial deal, its field naming that head, the composed name,
#: the tail type its second period is and the tail's block - a curve's spread, a spot's basis. A
#: spot's deal reads what rides its primary, the FX option the pair's repo curve and the equity
#: forward the dividend curve.
COMPOSED = {
    'InterestRate': ('CF', 'Discount_Rate', 'USD.FUNDING', 'InterestRate', {
        'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
        'Curve': utils.Curve([], [[0.0, 0.002], [5.0, 0.002]])}),
    'InflationRate': ('LINKER', 'Index', 'INFL.SPREAD', 'InflationRate', dict(
        trial_rates.FACTORS['InflationRate.INFL'], Curve=utils.Curve([], [[0.0, 0.003], [10.0, 0.003]]))),
    'FxRate': ('FXO', 'Underlying_Currency', 'EUR.ONSHORE', 'ObservedBasis', {'Spot': 0.01}),
    'EquityPrice': ('EQF', 'Equity', 'EQ.ADR', 'ObservedBasis', {'Spot': 0.5}),
    'CommodityPrice': ('METAL_FUT', 'Commodity', 'METAL.LBMA', 'ObservedBasis', {'Spot': 2.0})}


@pytest.mark.parametrize('head', sorted(utils.NESTED))
def test_discovery_and_the_resolver_read_one_chain_registry(head):
    """For every head `utils.NESTED` names, one trial deal naming it composed: the universe holds the
    head on the primary and the tail on the whole name - the full name a factor only where it is the
    tail - and a type the field declares beside a spot, the equity's dividend curve, on the primary
    and never on the whole name; and the deal prices, its compile finding each link of its chain and
    what rides the primary in that universe by name. A head added to the registry with no row here
    fails by name.

    Killing mutations: `calc_factor_code_chain` reading a tail of its own for one head - `InterestRate`
    for `EquityPrice` - names `InterestRate.EQ.ADR`, which discovery never registers, and the equity
    deal is skipped; discovery registering a type declared beside a spot on the whole name again
    names `DividendRate.EQ.ADR` and not the `DividendRate.EQ` the forward reads, which is skipped;
    `get_fx_zero_rate_factor` looking the pair up on the whole name finds no `FxRate.EUR.ONSHORE`
    and the FX option is skipped.
    """
    reference, field, name, tail, block = COMPOSED[head]
    deal, factors = next((deal, family.FACTORS) for family in trials.FAMILIES.values()
                         for deal in family.DEALS if deal['Reference'] == reference)
    job = trials.document(SimpleNamespace(DEALS=[dict(deal, **{field: name})], CONFIGURATION={},
                                          FACTORS=dict(factors, **{tail + '.' + name: block})))
    described = rf.Context().load_json((json.dumps(job), 'chain')).describe()['factors']
    universe, primary = set(described['resolved']), name.split('.')[0]
    beside = [kind for kind in getattr(instruments, deal['Object']).factor_fields[field] if kind != head]
    assert {head + '.' + primary, tail + '.' + name} | {k + '.' + primary for k in beside} <= universe, described
    assert not {k + '.' + name for k in beside} & universe.union(described['missing']), described
    assert (head + '.' + name in universe) == (tail == head), described
    assert reference in trials.marks(job), 'the deal did not resolve its chain'


def test_a_composed_equity_reads_the_dividend_curve_it_states_whole():
    """An equity option on the composed `EQ.ADR` stating `Dividends: EQ.ADR` reads that curve,
    whole, and its twin stating none reads the primary's `DividendRate.EQ`: at 4% against the
    primary's 1% the two mark apart, and the stated one marks to the bit what the twin marks once
    the primary's curve is the 4% one. Alone in a world holding no `DividendRate.EQ`, the stated
    option's universe holds its own curve and neither resolves nor names missing the primary's, and
    its booking verdict names no factor.

    Killing mutations: `get_dividend_rate_factor` truncating the name it is handed to its first
    period - the stated option reads `DividendRate.EQ` and marks its twin's value; the option's
    fallback naming the whole equity - the twin reads `DividendRate.EQ.ADR`, which discovery never
    registers, and is skipped; discovery registering the curve riding the equity whatever the deal
    states - `DividendRate.EQ` named missing.
    """
    _, _, name, tail, block = COMPOSED['EquityPrice']
    deal = next(deal for deal in trials.CORE.DEALS if deal['Reference'] == 'EQO')
    curve = {'Floor': None, 'Currency': 'USD', 'Curve': utils.Curve([], [[0.0, 0.04], [5.0, 0.04]])}

    def document(dividends, *deals):
        return trials.document(SimpleNamespace(
            DEALS=list(deals), CONFIGURATION={}, FACTORS={tail + '.' + name: block, dividends: curve}))

    stated = dict(deal, Reference='STATED', Equity=name, Dividends=name)
    primary = dict(deal, Reference='PRIMARY', Equity=name)
    apart = trials.marks(document('DividendRate.' + name, stated, primary))
    alike = trials.marks(document('DividendRate.EQ', primary))
    assert apart['STATED'] == alike['PRIMARY'] != apart['PRIMARY'], (apart, alike)

    job = document('DividendRate.' + name, stated)
    del job['Calc']['MergeMarketData']['ExplicitMarketData']['Price Factors']['DividendRate.EQ']
    context = rf.Context().load_json((json.dumps(job), 'stated'))
    described = context.describe()['factors']
    assert 'DividendRate.' + name in described['resolved'], described
    assert 'DividendRate.EQ' not in set(described['resolved']) | set(described['missing']), described
    assert not context.validate()['factors'], context.validate()


@pytest.mark.parametrize('head,reference,field', [
    ('FxRate', 'FWD', 'Buy_Currency'), ('FxRate', 'FXO', 'Underlying_Currency'),
    ('EquityPrice', 'EQF', 'Equity')])
def test_a_composed_name_beside_its_primary_marks_the_same_in_either_order(head, reference, field):
    """A trial forward or FX option and its twin on the registry's composed name, booked in one
    document, each mark the same whichever is booked first, and the two differ: the FX cross, the FX
    forward and the equity forward are cached on every element of every chain they read, so the
    composed name never reads its primary's entry. The basis is non-zero, else the twins agree and
    no key can be told apart.

    Killing mutations: `calc_fx_cross` keying each chain on its first element - the FX forward
    booked second marks the first's value, the plain one 11.9% below the composed; `calc_fx_forward`
    keying each spot on its first element - the FX option booked second marks the first's, the plain
    one 6.4% below the composed; `calc_eq_forward` keying the equity on its first element - the
    equity forward booked second marks the first's, the plain one 9.3% below the composed.
    """
    _, _, name, tail, block = COMPOSED[head]
    deal, factors = next((deal, family.FACTORS) for family in trials.FAMILIES.values()
                         for deal in family.DEALS if deal['Reference'] == reference)
    world = dict(factors, **{tail + '.' + name: block})

    def marks(*deals):
        return trials.marks(trials.document(SimpleNamespace(
            DEALS=list(deals), CONFIGURATION={}, FACTORS=world)))

    plain, composed = dict(deal, Reference='PLAIN'), dict(deal, Reference='COMPOSED', **{field: name})
    together = marks(plain, composed)
    assert together == marks(composed, plain) and together['PLAIN'] != together['COMPOSED'], together
