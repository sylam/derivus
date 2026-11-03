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
priced beside their primary. The band basis's fit is a calibration document through
`Config.calibrate_factors`, and its draw on the innovation's own GARCH variance a pair of credit
Monte Carlo documents read side by side; its replay is a simulate-only hedging document on an observed
scenario, its recursions read off the replay's `BASIS_REPLAY` debug line, and a GARCH spot's off
`GARCH_REPLAY`.
"""
import ast
import io
import json
import logging
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import derivus as rf
import test_position_scaling as trials
import trial_rates
from derivus import instruments, utils
from derivus.config import Config, CustomJsonEncoder, ModelParams
from derivus.stochasticprocess import FILLED_ROWS

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
    the primary's curve is the 4% one.

    Killing mutations: `get_dividend_rate_factor` truncating the name it is handed to its first
    period - the stated option reads `DividendRate.EQ` and marks its twin's value; the option's
    fallback naming the whole equity - the twin reads `DividendRate.EQ.ADR`, which discovery never
    registers, and is skipped.
    """
    _, _, name, tail, block = COMPOSED['EquityPrice']
    deal = next(deal for deal in trials.CORE.DEALS if deal['Reference'] == 'EQO')
    curve = {'Floor': None, 'Currency': 'USD', 'Curve': utils.Curve([], [[0.0, 0.04], [5.0, 0.04]])}

    def marks(dividends, *deals):
        return trials.marks(trials.document(SimpleNamespace(
            DEALS=list(deals), CONFIGURATION={}, FACTORS={tail + '.' + name: block, dividends: curve})))

    stated = dict(deal, Reference='STATED', Equity=name, Dividends=name)
    primary = dict(deal, Reference='PRIMARY', Equity=name)
    apart, alike = marks('DividendRate.' + name, stated, primary), marks('DividendRate.EQ', primary)
    assert apart['STATED'] == alike['PRIMARY'] != apart['PRIMARY'], (apart, alike)


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


#: A `Band_Mixture` basis: a 2 $/oz dead band with a 0.2 pull around the slow mean, the stress state
#: 10% of days at four times the quiet sd and kept with probability 0.8.
BAND = {
    'A': 0.01, 'Phi': 0.9, 'Nu': 0.0, 'Sigma': 1.5, 'Reversion_Model': 'Band_Mixture',
    'Band_Kappa': 2.0, 'Band_Beta': 0.2, 'Mix_Q_Stress': 0.1, 'Mix_Sigma_Quiet': 1.0,
    'Mix_Sigma_Stress': 4.0, 'Mix_Stay_Stress': 0.8, 'Mix_P0_Stress': 0.1,
    'Slow_Mean_Lambda': 0.95, 'Mu_0': -7.0, 'Calibration_DT_Years': 1.0 / 252.0}

#: Mixture sds of unit stationary variance: 20% stress at sqrt(3), 80% quiet at sqrt(1/2).
UNIT_MIX = {'Mix_Q_Stress': 0.2, 'Mix_Sigma_Quiet': 0.5 ** 0.5, 'Mix_Sigma_Stress': 3.0 ** 0.5}

#: The daily month the GARCH documents simulate: a future maturing a week after it.
MONTH = {'maturity': '2026-02-20', 'Time_Grid': '0d 1d(1d)', 'Deterministic_Kernels': 'Yes'}


def _band_mtm(basis, vol=0.3, maturity='2026-10-15', flat_curves=False, deal=None, **calculation):
    """The credit Monte Carlo mtm of the one future on `LBMA_AM.CME` maturing at `maturity`, or of
    the one `deal` in its place, its basis the `BasisLinkedSpotModel` block `basis` and its primary
    a GBM at `vol`, with the calculation's fields `calculation` stated over `_world`'s;
    `flat_curves` puts the rate and the carry at zero, so the future's mtm is the spot plus the
    basis."""
    world = _world('LBMA_AM.CME', chained=False)
    world['Calc']['Calculation'].update(calculation)
    future = world['Calc']['Deals']['Deals']['Children'][0]['Children'][0]['Instrument']['.Deal']
    future['Maturity_Date'] = {'.Timestamp': maturity}
    if deal is not None:
        world['Calc']['Deals']['Deals']['Children'][0]['Children'][0]['Instrument']['.Deal'] = deal
    market = world['Calc']['MergeMarketData']['ExplicitMarketData']
    if flat_curves:
        factors = market['Price Factors']
        factors['InterestRate.USD-SOFR']['Curve']['.Curve']['data'] = [[0.0, 0.0], [5.0, 0.0]]
        factors['ForwardRate.PLATINUM_CARRY']['Curve']['.Curve']['data'] = [[46213.0, 0.0], [46395.0, 0.0]]
    market['Model Configuration'] = {'.ModelParams': {
        'modeldefaults': {'CommodityPrice': 'GBMAssetPriceModel', 'ObservedBasis': 'BasisLinkedSpotModel'},
        'modelfilters': {}}}
    market['Price Models'] = {
        'GBMAssetPriceModel.LBMA_AM': {'Vol': vol, 'Drift': 0.0},
        'BasisLinkedSpotModel.LBMA_AM.CME': basis}
    cx = rf.Context(path_transform={}, file_transform={})
    cx.load_json((json.dumps(world), 'band_basis.json'))
    return rf.run_cmc(cx.current_cfg, prec=torch.float64)[1]['Results']['mtm']


def _band_basis(spot, eta, span=63, a=0.05, kappa=4.0, beta=0.3):
    """The basis the band fit's own law draws from -5 along `spot` with the innovations `eta`:
    d_t - d_{t-1} = A dS_t - pull(d_{t-1}) + eta_t, d_t the deviation from the span's slow mean."""
    n, lam = len(spot), 1.0 - 2.0 / (span + 1.0)
    basis, ewm = np.full(n, -5.0), np.full(n, -5.0)
    for j in range(2, n):
        d = basis[j - 1] - ewm[j - 2]
        basis[j] = (ewm[j - 1] + d + a * (spot[j] - spot[j - 1])
                    - beta * np.sign(d) * max(abs(d) - kappa, 0.0) + eta[j])
        ewm[j] = lam * ewm[j - 1] + (1.0 - lam) * basis[j]
    return basis


def _fit_band(tmp_path, dates, spot, basis, garch=None, **tuning):
    """The `BasisLinkedSpotModel` block a calibration document fits under `Band_Mixture` with the
    63-day slow mean and `tuning`, through `Config.calibrate_factors`, on the archive of `spot`
    and `basis` dated `dates`; given `garch`, the `GARCHSpotModel` block the spot is fitted to in
    place of a GBM under the calibration fields `garch`."""
    spot_model, spot_fit = (('GBMAssetPriceModel', {'Method': 'GBMAssetPriceCalibration'}) if garch is None
                            else ('GARCHSpotModel', dict(garch, Method='GARCHSpotCalibration')))
    archive = pd.DataFrame({'CommodityPrice.PLAT': spot, 'ObservedBasis.PLAT.CME': basis}, index=dates)
    archive.index.name = 'Date'
    archive.to_csv(tmp_path / 'archive.csv', sep='\t')
    retrieval = {'ID': '', 'Use_Pre_Computed_Statistics': 'Yes', 'Data_Retrieval_Parameters': {
        'Frequency': '1d', 'Calendar': '', 'Business_Days_In_Year': 252}}
    document = {
        'MarketData': {
            'System Parameters': {'Base_Currency': 'USD', 'Base_Date': dates[-1]},
            'Model Configuration': ModelParams(({'CommodityPrice': spot_model,
                                                'ObservedBasis': 'BasisLinkedSpotModel'}, {})),
            'Price Factors': {}, 'Price Models': {}, 'Correlations': {},
            'Price Factor Interpolation': ModelParams(({}, {})),
            'Bootstrapper Configuration': {}, 'Market Prices': {}},
        'Version': ['JSONVersion', '22.05.30'],
        'CalibrationConfig': {
            'MarketDataArchiveFile': {'name': str(tmp_path / 'archive.csv'), 'skiprows': 0,
                                      'index_column': 'Date'},
            'Calibrations': {
                spot_model: dict(retrieval, **spot_fit),
                'BasisLinkedSpotModel': dict(retrieval, Method='BasisLinkedSpotCalibration',
                                             Slow_Mean_Span=63, Reversion_Model='Band_Mixture',
                                             **tuning)}}}
    with io.open(tmp_path / 'calibration.json', 'w', encoding='utf-8') as handle:
        handle.write(json.dumps(document, indent=1, cls=CustomJsonEncoder))
    config = Config()
    config.parse_json(str(tmp_path / 'calibration.json'))
    factors = config.fetch_all_calibration_factors()
    config.calibrate_factors(dates[0], dates[-1], dict(factors['present'], **factors['absent']))
    return config.params['Price Models']['BasisLinkedSpotModel.PLAT.CME' if garch is None else 'GARCHSpotModel.PLAT']


def test_the_band_basis_prices_under_torch_s_distribution_validation():
    """A `Band_Mixture` basis carries `Nu` 0, the calibrator's own convention, the mixture being its
    innovation law: the chi-squared the t-scale is drawn from is never read on the band branch, and
    torch validates a chi-squared's degrees of freedom whenever `__debug__` is on, as it is here and
    in every run not started with -O. The band document prices: every row of the credit Monte
    Carlo's mtm is finite and the base-date row is one value across the paths.

    Killing mutation: `validate_args=False` dropped from the basis draw - torch refuses the
    degrees of freedom 0 before the first step.
    """
    mtm = _band_mtm(BAND)
    assert mtm.shape == (11, 64) and np.isfinite(mtm.values).all(), mtm.shape
    assert np.ptp(mtm.iloc[0].values) == 0.0, mtm.iloc[0].values[:4]


def test_a_band_basis_on_a_constant_garch_variance_is_the_band_with_its_sds_scaled():
    """`G_Omega` > 0 puts the band mixture's draw on the innovation's own GARCH(1,1),
    eta_t = sigma_t m_t Z_t: with `G_Alpha` = `G_Beta` = 0 and `G_Omega` = `Sig2_0` = c^2 the
    variance is c^2 on every step, so the band document with unit-variance mixture sds simulates
    to the bit what the same document without `G_*` keys simulates with both sds times c, over a
    daily month of 256 paths, the spot at 30% vol, the band's pull and the state chain live. c is
    2, so c^2 and its root are exact and the two products round alike; the draws are consumed
    identically, the recursion drawing nothing.

    Killing mutations: the constructor's `self.garch = False` under the band restored - the GARCH
    document simulates its unit sds, its mtm up to 9.98 from the scaled one; the draw scaled by
    `s2` in place of its root - every innovation twice the scaled document's, the mtm up to 19.53
    apart.
    """
    c = 2.0
    garch = _band_mtm(dict(BAND, **UNIT_MIX, G_Omega=c * c, G_Alpha=0.0, G_Beta=0.0, Sig2_0=c * c),
                      Batch_Size=256, **MONTH)
    scaled = _band_mtm(dict(BAND, **{k: v * (c if k.startswith('Mix_Sigma') else 1.0)
                                     for k, v in UNIT_MIX.items()}), Batch_Size=256, **MONTH)
    assert garch.shape == (38, 256) and np.array_equal(garch.values, scaled.values)


def test_the_garch_seed_sets_the_band_basis_dispersion_over_its_first_month():
    """Two band documents on one GARCH(1,1), `G_Alpha` 0.05 and `G_Beta` 0.9 with `G_Omega` at
    0.05 c^2 so the stationary variance is c^2, one seeded at `Sig2_0` = c^2 and one at 4c^2, the
    spot at no vol and the band's pull off, so a month's basis move is the sum of its innovations;
    the stress state is drawn afresh each day (`Mix_Stay_Stress` = `Mix_P0_Stress` =
    `Mix_Q_Stress`), so each innovation's expected variance is f times the recursion's on the
    trading clock, E[sigma^2_k] = c^2 (1 + 3 p^k) after the seed, p = 1 - f (1 - `G_Alpha` -
    `G_Beta`) a calendar row of f = 252 / 365.25 calibration steps. Over the 31 daily steps to the
    one-month row the seeded document's mtm dispersion is the other's times sqrt(mean(1 + 3 p^k)) =
    1.6913 within 2% on 4,096 paths (measured 1.6863).

    Killing mutations: the band's draw left off the GARCH variance - both documents at the unit
    mixture, the ratio 1.0; the seed ignored under the band, both started at the stationary
    variance - 1.0; the recursion never advanced, each document at its seed - 2.0; `G_Beta` dropped
    from the recursion - 1.4716 (-7.7%); the recursion fed the relative draw m_t Z_t in place of
    eta_t - 1.4935 (-6.3%).
    """
    c, alpha, beta, n = 2.0, 0.05, 0.9, 31
    law = dict(BAND, **UNIT_MIX, Band_Beta=0.0, Mix_Stay_Stress=0.2, Mix_P0_Stress=0.2,
               G_Omega=(1.0 - alpha - beta) * c * c, G_Alpha=alpha, G_Beta=beta)
    calm, seeded = (_band_mtm(dict(law, Sig2_0=sig2_0), vol=0.0, Batch_Size=4096, **MONTH).loc['2026-02-15']
                    for sig2_0 in (c * c, 4.0 * c * c))
    p = 1.0 - 252.0 / 365.25 * (1.0 - alpha - beta)
    predicted = np.sqrt(np.mean(1.0 + 3.0 * p ** np.arange(n)))
    assert abs(seeded.std() / calm.std() / predicted - 1.0) < 0.02, (seeded.std() / calm.std(), predicted)


def test_the_band_fit_leaves_a_jump_across_a_calendar_gap_out_of_the_innovation(tmp_path):
    """A calibration document on a synthetic `Band_Mixture` archive: three years of business days
    drawn from the fit's own law (A 0.05, kappa 4, beta 0.3, the 63-day slow mean, a two-state
    innovation 80% at sd 2 and 20% at sd 6, the law's sd 3.2249), one week missing three days so
    a Wednesday is followed by the next Tuesday, and a 60 $/oz jump across that 6-day gap. The fit
    reads only rows at most 4 calendar days apart, so the jump is not an innovation: the fitted
    Sigma is the law's within 5% (measured 3.2303) and the stress sd is the law's 6 within 5%
    (5.9152). 3 s.

    Killing mutation: the gap read as `(DatetimeIndex(index[2:]) - DatetimeIndex(index[1:-1])).days`
    on the archive's Excel-day index, every gap 0 days, so the jump enters the fit: Sigma 3.8839
    (+20%) and the stress sd 10.2898, the stress weight halved (0.098 against 0.203).
    """
    n, p_quiet, s_quiet, s_stress = 756, 0.8, 2.0, 6.0
    rng = np.random.default_rng(1)
    days = pd.bdate_range('2021-01-04', periods=n + 3)
    dates = days[:n // 2].append(days[n // 2 + 3:])
    spot = 1000.0 + np.cumsum(np.concatenate([[0.0], rng.normal(0.0, 15.0, n - 1)]))
    eta = np.where(rng.random(n) < p_quiet, s_quiet, s_stress) * rng.standard_normal(n)
    eta[n // 2] = 60.0                                  # the row after the gap
    fitted = _fit_band(tmp_path, dates, spot, _band_basis(spot, eta))
    law = np.sqrt(p_quiet * s_quiet ** 2 + (1.0 - p_quiet) * s_stress ** 2)
    assert abs(fitted['Sigma'] / law - 1.0) < 0.05, (fitted['Sigma'], law)
    assert abs(fitted['Mix_Sigma_Stress'] / s_stress - 1.0) < 0.05, fitted['Mix_Sigma_Stress']


#: The `Band_Mixture` block the calibration fits under `GARCH_Innovation` 'No' on the archive of
#: the test below.
BAND_FIT_WITHOUT_GARCH = {
    'A': 0.04241021394456613, 'Phi': 0.8174279678972644, 'Nu': 0.0, 'Sigma': 3.8213804346664833,
    'Reversion_Model': 'Band_Mixture', 'Band_Kappa': 4.093865983658383,
    'Band_Beta': 0.32257990121444935, 'Mix_Q_Stress': 0.11168874340078161,
    'Mix_Sigma_Quiet': 2.5231836261197125, 'Mix_Sigma_Stress': 8.954691898496261,
    'Mix_Stay_Stress': 0.10344827586206896, 'Mix_P0_Stress': 0.04033416302268044,
    'Slow_Mean_Lambda': 0.96875, 'Mu_0': 10.822116754114283,
    'Calibration_DT_Years': 0.003968253968253968}


def test_the_band_fit_under_garch_simulates_the_window_s_variance_from_its_trailing_level(tmp_path):
    """A calibration document on a synthetic `Band_Mixture` archive of four years whose innovation
    is a GARCH(1,1) (omega 0.45, alpha 0.1, beta 0.85, persistence 0.95) on a two-state mixture of
    unit variance, stress 5% of days at variance 8, fitted under `GARCH_Innovation` 'Yes' with
    `Max_Persistence` 0.9 below the law's; the mixture variance v of the standardised innovation
    is 1.0785. The stamp is simulated by a credit Monte Carlo document of 16,384 paths over a
    daily year on curves flat at zero, the spot at no vol and the band's pull off, so each day's
    move is its innovation, the block's `Calibration_DT_Years` a calendar day so a row is one
    calibration step, and the stress chain started at its stationary probability: (a) the first
    day's variance is the archive's last 21 innovations' within 8% (1.031, the paths' standard
    error about 2.5%); (b) the daily variance past day 100 is the archive's innovation variance
    within 4% (1.016, the stress state's own persistence adding about 1%); (c) `G_Alpha` +
    `G_Beta` is the cap 0.9 to 1e-12. The location fit is the one 'No' makes, and under 'No' the
    block is `BAND_FIT_WITHOUT_GARCH` to the bit. 2 s.

    Killing mutations: `G_Omega` the fit's own intercept - (b) 0.255; `Sig2_0` the fit's h at the
    last observation - (a) 1.824; `Sig2_0` the window's whole variance - (a) 3.463; `G_Alpha`
    v alpha - (b) 1.091, (c) 0.9067; the sds left at the mixture variance v - (a) 1.112, (b) 1.177;
    v folded into `G_Omega` and `Sig2_0` beside h's seed - (a) 1.967, (b) 0.275; the switch unread
    under the band - no `G_Alpha`; the GARCH branch taken under 'No' - the frozen block parts.
    """
    n, omega, alpha, beta = 1008, 0.45, 0.1, 0.85
    rng = np.random.default_rng(3)
    dates = pd.bdate_range('2021-01-04', periods=n)
    spot = 1000.0 + np.cumsum(np.concatenate([[0.0], rng.normal(0.0, 15.0, n - 1)]))
    z = np.sqrt(np.where(rng.random(n) < 0.05, 8.0, 0.6 / 0.95)) * rng.standard_normal(n)
    eta, h = np.empty(n), np.empty(n + 1)
    h[0] = omega / (1.0 - alpha - beta)
    for j in range(n):
        eta[j] = np.sqrt(h[j]) * z[j]
        h[j + 1] = omega + alpha * eta[j] ** 2 + beta * h[j]
    basis = _band_basis(spot, eta)

    fitted = _fit_band(tmp_path, dates, spot, basis, GARCH_Innovation='Yes', Max_Persistence=0.9)
    assert abs(fitted['G_Alpha'] + fitted['G_Beta'] - 0.9) < 1e-12, fitted
    stationary = dict(fitted, Band_Beta=0.0, Mix_P0_Stress=fitted['Mix_Q_Stress'], Calibration_DT_Years=1.0 / 365.25)
    mtm = _band_mtm(
        stationary, vol=0.0, maturity='2027-01-29', flat_curves=True, Batch_Size=16384,
        Time_Grid='0d 1d(1d)', Deterministic_Kernels='Yes')
    daily = np.diff(mtm.loc[:'2027-01-29'].values, axis=0).var(axis=1)
    assert abs(daily[0] / eta[-21:].var() - 1.0) < 0.08, (daily[0], eta[-21:].var())
    assert abs(daily[100:].mean() / eta[2:].var() - 1.0) < 0.04, (daily[100:].mean(), eta[2:].var())
    location = ('A', 'Phi', 'Sigma', 'Band_Kappa', 'Band_Beta', 'Mu_0')
    assert {k: fitted[k] for k in location} == {k: BAND_FIT_WITHOUT_GARCH[k] for k in location}
    assert _fit_band(tmp_path, dates, spot, basis, GARCH_Innovation='No') == BAND_FIT_WITHOUT_GARCH


def test_a_scale_window_moves_the_band_fit_s_three_sds_by_the_trailing_level_alone(tmp_path):
    """A calibration document on a synthetic `Band_Mixture` archive of two years drawn from the
    fit's own law (a two-state innovation 85% at sd 2 and 15% at sd 7), its last day, the
    calibration date, a 40 $/oz move, fitted with and without `Basis_Scale_Window` 20: `Sigma`,
    `Mix_Sigma_Quiet` and `Mix_Sigma_Stress` are the plain fit's times the realised sd (ddof 1)
    of the last 20 basis changes before the calibration date over the fitted law's stationary sd
    sqrt(q s_stress^2 + (1 - q) s_quiet^2), q = `Mix_Q_Stress`, to 1e-12 (the ratio 0.7992), and
    every other key is the plain fit's to the bit. The same archive without five business days before
    its last week reads the 20 changes across at most 4 calendar days, as the fit does (0.8643). 0.2 s.

    Killing mutations: the change into the calibration date counted - the ratio 2.7005; ddof 0 -
    0.7790; the law's sd with q on the quiet sd - 0.2964; `Sigma` left unscaled - 3.4785 against
    2.7801; the move across the seven-day hole counted - 0.8809.
    """
    n, window = 504, 20
    rng = np.random.default_rng(5)
    dates = pd.bdate_range('2023-01-02', periods=n)
    spot = 1000.0 + np.cumsum(np.concatenate([[0.0], rng.normal(0.0, 15.0, n - 1)]))
    eta = np.where(rng.random(n) < 0.85, 2.0, 7.0) * rng.standard_normal(n)
    eta[-1] = 40.0
    basis = _band_basis(spot, eta)
    for keep in (np.arange(n), np.r_[0:n - 12, n - 7:n]):
        plain = _fit_band(tmp_path, dates[keep], spot[keep], basis[keep])
        scaled = _fit_band(tmp_path, dates[keep], spot[keep], basis[keep], Basis_Scale_Window=window)
        q, sds = plain['Mix_Q_Stress'], ('Sigma', 'Mix_Sigma_Quiet', 'Mix_Sigma_Stress')
        moves = np.diff(basis[keep][:-1])[np.diff(dates[keep][:-1].values) <= np.timedelta64(4, 'D')]
        ratio = moves[-window:].std(ddof=1) / np.sqrt(
            q * plain['Mix_Sigma_Stress'] ** 2 + (1.0 - q) * plain['Mix_Sigma_Quiet'] ** 2)
        assert all(abs(scaled[k] / (plain[k] * ratio) - 1.0) < 1e-12 for k in sds), (ratio, scaled, plain)
        assert {k: v for k, v in scaled.items() if k not in sds} == {k: v for k, v in plain.items() if k not in sds}


HEDGE_FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures',
                             'policy_test_simulate_only.json')


def _holiday_tape():
    """119 calendar rows from Friday 2026-04-10 over 199 business days, London's 4 May bank holiday not
    among them: `(the business days' calendar row, the calendar's flag of the rows the London fixing
    calendar filled, the holiday's row)`."""
    days = pd.bdate_range('2026-04-10', periods=200).drop(pd.Timestamp('2026-05-04'))
    calendar = pd.date_range('2026-04-10', periods=200)
    return (np.searchsorted(days, calendar, side='right') - 1, ~calendar.isin(days),
            calendar.get_loc(pd.Timestamp('2026-05-04')))


def _replayed(tmp_path, caplog, basis_block, spot, basis, spot_block=None, filled=None):
    """The recursions `{'mu', 'sig2', 'regime'}` per row on the first path that the simulate-only
    hedging fixture, rolling daily from Friday 2026-04-10, publishes replaying the observed `spot`
    and `basis`, the rows `filled` flagged forward-filled, its basis the `BasisLinkedSpotModel` block
    `basis_block`; given `spot_block`, the spot is that `GARCHSpotModel` block and the recursion its
    `{'log_h', 'h'}`, the terminal h."""
    observed = str(tmp_path / 'observed.npz')
    np.savez(observed, **{'CommodityPrice.PLATINUM_LME': spot, 'ObservedBasis.PLATINUM_LME.LME_CME': basis},
             **({} if filled is None else {FILLED_ROWS: filled}))
    cfg = json.load(open(HEDGE_FIXTURE))
    cfg['Calc']['Calculation'].update({'Batch_Size': 2, 'Observed_Scenario': observed})
    market = cfg['Calc']['MergeMarketData']['ExplicitMarketData']
    market['Price Models'] = {'BasisLinkedSpotModel.PLATINUM_LME.LME_CME': basis_block}
    if spot_block is not None:
        market['Price Models']['GARCHSpotModel.PLATINUM_LME'] = spot_block
        market['Model Configuration'] = {'.ModelParams': {
            'modeldefaults': {'CommodityPrice': 'GARCHSpotModel'}, 'modelfilters': {}}}
    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        rf.Context().load_json((json.dumps(cfg), 'replay.json')).run_job()
    tag = 'BASIS_REPLAY' if spot_block is None else 'GARCH_REPLAY'
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith(tag))
    return ast.literal_eval(line.split(' ', 2)[2])


@pytest.mark.parametrize('garch', ['No', 'Yes'])
def test_a_replay_holds_every_basis_recursion_across_a_forward_filled_row(tmp_path, caplog, garch):
    """A calibration document on an archive drawn as the scale-window test's, under
    `GARCH_Innovation` 'No' and 'Yes'. The simulate-only hedging fixture replays its block on 119
    calendar rows, the weekends and London's
    4 May bank holiday flagged forward-filled, the holiday printing a spot and a basis of its own (a
    CME print), and on the same business days with the filled rows removed: on every business row
    the published mu, sig2 and regime equal, to the bit, the compressed replay's at that row's count
    of business days (84 rows), and every filled row holds the row before it. A market row is one
    calibration step whatever the clock says: the compressed replay's mu is the slow mean's own
    recursion mu' = lambda mu + (1 - lambda) b on every row within 1e-4 (1.1e-8, the job's
    float32). On the compressed path, which flags nothing, the basis and the spot both flat on its
    sixth business day and the spot flat on its ninth each move mu. 2 s.

    Killing mutations: the flag unread (mu 2.13 apart on 78 of the 84 rows, sig2 1.10 under GARCH,
    the regime on 18 and 19); the regime drawn at the calendar row rather than the business count
    (19 and 20 rows); the clock's per-row f kept on a market row (mu 3.61 off the recursion);
    the old rule, a row on which both prices repeat filled (mu held on the sixth, 1.6523 twice); the
    parent's move and the basis read from the row before rather than the last market row (sig2
    0.040 under GARCH on 40 rows).
    """
    n = 504
    rng = np.random.default_rng(5)
    dates = pd.bdate_range('2023-01-02', periods=n)
    history = 1000.0 + np.cumsum(np.concatenate([[0.0], rng.normal(0.0, 15.0, n - 1)]))
    archive = _band_basis(history, np.where(rng.random(n) < 0.85, 2.0, 7.0) * rng.standard_normal(n))
    fitted = _fit_band(tmp_path, dates, history, archive, GARCH_Innovation=garch)

    row, filled, holiday = _holiday_tape()
    spot = 2055.0 + np.cumsum(np.concatenate([[0.0], rng.normal(0.0, 30.0, 198)]))
    basis = fitted['Mu_0'] + np.cumsum(np.concatenate([[0.0], rng.normal(0.0, 3.0, 198)]))
    basis[6], spot[6], spot[9] = basis[5], spot[5], spot[8]
    spot_cal, basis_cal = spot[row], basis[row]
    spot_cal[holiday], basis_cal[holiday] = spot_cal[holiday] + 10.0, basis_cal[holiday] + 1.0
    calendar = _replayed(tmp_path, caplog, fitted, spot_cal, basis_cal, filled=filled)
    business = _replayed(tmp_path, caplog, fitted, spot, basis)
    market = [t for t in range(len(calendar['mu'])) if t == 0 or row[t] != row[t - 1]]
    assert len(market) == 84 and set(calendar) == {'mu', 'regime'} | ({'sig2'} if garch == 'Yes' else set())
    assert {k: [v[t] for t in market] for k, v in calendar.items()} == \
        {k: [v[row[t]] for t in market] for k, v in business.items()}
    assert all(v[t] == v[t - 1] for v in calendar.values() for t in range(1, len(v)) if t not in market)
    mu = [fitted['Mu_0']]
    for b in basis[1:len(business['mu'])].astype(np.float32):
        mu.append(fitted['Slow_Mean_Lambda'] * mu[-1] + (1.0 - fitted['Slow_Mean_Lambda']) * b)
    assert np.abs(np.array(business['mu']) - mu).max() < 1e-4, np.abs(np.array(business['mu']) - mu).max()
    assert business['mu'][6] != business['mu'][5] and business['mu'][9] != business['mu'][8], business['mu'][4:10]


#: A flat-Sigma Linear basis, which a GARCH spot without regimes can carry.
FLAT_BASIS = {'A': 0.0, 'Phi': 0.4, 'Nu': 5.0, 'Sigma': 8.0, 'Calibration_DT_Years': 1.0 / 252.0}


def test_a_spot_replay_holds_its_variance_across_a_forward_filled_holiday(tmp_path, caplog):
    """A calibration document fits `GARCHSpotModel` on 504 days of a GARCH(1,1) spot beside a band
    basis. The simulate-only hedging fixture, its spot that block and its basis a flat Linear,
    replays it on 119 calendar rows, the weekends and London's 4 May bank holiday flagged
    forward-filled, the holiday printing a spot of its own (a CME print), with a flat fix on the
    fourth business day (a Wednesday), and on the same business days with the filled rows removed:
    on all 84 business rows the published log h equals, to the bit, the compressed replay's at that
    row's count of business days, and every filled row holds the row before it. A market row is one
    calibration step and a filled row's move and drift are the next market row's: under a `Mu` of
    0.5 the calendar replay's log h is the business-day filter written here, h' = Omega + Alpha r^2 +
    Beta h with r the log return less the drift over its calendar days plus h / 2, within 5e-5 on
    every market row (9.1e-6 on the CPU, the job's float32) and its terminal h to 5e-5 (6.7e-7). 5 s.

    Killing mutations: the flag unread (83 of the 84 rows apart, log h 0.51 off on 5 May); the old
    rule, a weekend row repeating the spot filled, so the holiday steps (68 rows apart, 0.25 off on
    5 May); the trading clock's per-row fraction kept on a market row (0.48 off the filter, the
    compressed replay alike); the holiday's move dropped (68 rows apart, 0.19 off on 5 May); a
    filled row's drift dropped (0.079 off the filter)."""
    n = 504
    rng = np.random.default_rng(11)
    returns, h = np.zeros(n), 2.0e-4
    for j in range(1, n):
        returns[j] = h ** 0.5 * rng.standard_normal()
        h = 1.0e-5 + 0.1 * returns[j] ** 2 + 0.85 * h
    dates = pd.bdate_range('2023-01-02', periods=n)
    history = 1000.0 * np.exp(np.cumsum(returns))
    archive = _band_basis(history, 2.0 * rng.standard_normal(n))
    fitted = _fit_band(tmp_path, dates, history, archive, garch={})

    row, filled, holiday = _holiday_tape()
    days = pd.bdate_range('2026-04-10', periods=200).drop(pd.Timestamp('2026-05-04'))
    spot = 2055.0 * np.exp(np.cumsum(np.concatenate([[0.0], rng.normal(0.0, 0.02, 198)])))
    basis = -30.0 + np.cumsum(rng.normal(0.0, 3.0, 199))
    spot[3] = spot[2]
    spot_cal = spot[row]
    spot_cal[holiday] *= 1.01
    calendar = _replayed(tmp_path, caplog, FLAT_BASIS, spot_cal, basis[row], spot_block=fitted, filled=filled)
    business = _replayed(tmp_path, caplog, FLAT_BASIS, spot, basis, spot_block=fitted)
    log_h = calendar['log_h']
    market = [t for t in range(len(log_h)) if t == 0 or row[t] != row[t - 1]]
    apart = sum(log_h[t] != business['log_h'][row[t]] for t in market)
    held = all(log_h[t] == log_h[t - 1] for t in range(1, len(log_h)) if t not in market)

    mu = 0.5
    drifted = _replayed(tmp_path, caplog, FLAT_BASIS, spot_cal, basis[row], spot_block=dict(fitted, Mu=mu),
                        filled=filled)
    tape = np.log(spot.astype(np.float32).astype(np.float64))
    years = np.diff(days.values).astype('timedelta64[D]').astype(float) / 365.25
    filtered = [fitted['H0']]
    for k in range(1, row[len(log_h) - 1] + 1):
        r = tape[k] - tape[k - 1] - mu * years[k - 1] + 0.5 * filtered[-1]
        filtered.append(fitted['Omega'] + fitted['Alpha'] * r * r + fitted['Beta'] * filtered[-1])
    worst = max(abs(drifted['log_h'][t] - np.log(filtered[row[t]])) for t in market)
    assert len(market) == 84 and apart == 0 and held, (len(market), apart, held)
    assert worst < 5e-5 and abs(drifted['h'] / filtered[-1] - 1.0) < 5e-5, (worst, drifted['h'], filtered[-1])


def test_a_calendar_week_on_the_trading_clock_carries_the_business_days_its_rows_span():
    """The trading clock on a calendar-daily grid, a row f = (1/365.25) / (1/252) = 0.6899 calibration
    steps: the credit Monte Carlo of the one future on flat curves, the spot at no vol, so the mtm's
    move over the first calendar week (rows 0 to 7) is the basis's, on 65,536 paths.
    (a) The band's pull off and the stress chain at its stationary 20% (`UNIT_MIX`, `Mix_Stay_Stress`
    0.5): the week's variance is 7f = 4.8296 business days of the law's unit daily variance within
    3% (measured 4.8641), and its kurtosis 3 (1 + sum_ij rho^|i-j| / 49) within 4% with the chain's
    eigenvalue rho = 1 - f (1 - stay + enter) = 0.5688 (4.1918, measured 4.1918).
    (b) A dead band of width 0 with pull 0.3 around a slow mean of lambda 0.8 seeded at the basis,
    the innovation Gaussian on a GARCH(1,1) (alpha 0.1, beta 0.85, stationary variance 1) seeded at
    4: the week's variance is the closed form of the linear (b, mu) recursion and the GARCH mean
    variance, each row f steps, within 3%: 8.4115 (measured 8.4535). 1 s.

    Killing mutations: the innovation left unscaled - (a) 7.05, (b) 13.83;
    scaled by f rather than its root - (a) 3.36; the pull left at a whole step - (b) 6.51; the slow
    mean's step left whole - (b) 9.42; the GARCH recursion in its calibrated spelling - (b) 7.18;
    the chain's switching left whole - (a) kurtosis 3.86.
    """
    walk = dict(BAND, **UNIT_MIX, Band_Beta=0.0, Mix_Stay_Stress=0.5, Mix_P0_Stress=0.2)
    pull = dict(BAND, Band_Kappa=0.0, Band_Beta=0.3, Slow_Mean_Lambda=0.8, Mu_0=-7.35, Mix_Q_Stress=0.5,
                Mix_Sigma_Quiet=1.0, Mix_Sigma_Stress=1.0, Mix_Stay_Stress=0.5, Mix_P0_Stress=0.5,
                G_Omega=0.05, G_Alpha=0.1, G_Beta=0.85, Sig2_0=4.0)

    def week(law):
        mtm = _band_mtm(law, vol=0.0, maturity='2026-01-23', flat_curves=True,
                        Batch_Size=65536, Time_Grid='0d 1d(1d)', Deterministic_Kernels='Yes')
        move = (mtm.loc['2026-01-22'] - mtm.loc['2026-01-15']).values
        return move - move.mean()

    def location(f):
        cov, h = np.zeros((2, 2)), 4.0
        for _ in range(7):
            m, step = 1.0 - 0.3 * f, 0.2 * f                  # the pull's and the slow mean's
            weights = np.array([[m, 1.0 - m], [step * m, 1.0 - step * m]])
            cov = weights @ cov @ weights.T + np.outer([1.0, step], [1.0, step]) * f * h
            h += f * (0.05 - 0.05 * h)
        return cov[0, 0]

    f = 252.0 / 365.25
    move, rho = week(walk), 1.0 - f * 0.625
    kurtosis = 3.0 * (1.0 + sum(rho ** abs(i - j) for i in range(7) for j in range(7)) / 49.0)
    assert abs(move.var() / (7.0 * f) - 1.0) < 0.03, (move.var(), 7.0 * f)
    assert abs((move ** 4).mean() / move.var() ** 2 / kurtosis - 1.0) < 0.04, kurtosis
    assert abs(week(pull).var() / location(f) - 1.0) < 0.03, location(f)


def test_the_average_price_swap_projects_the_basis_at_the_rate_the_clock_simulates():
    """An average-price swap on the composed `LBMA_AM.CME` with one fixing five daily rows out, its
    basis a Linear `BasisLinkedSpotModel` (Phi 0.817, a slow mean of lambda 0.96875 seeded 5 above
    the basis at -7.35, a flat Sigma of 0.5), on flat curves and a spot at no vol, over 16,384 paths:
    the base-date mark prices the deviation's share pi + (1 - pi)(lambda phi)^5 left at the fixing,
    pi = (1 - lambda) phi / ((1 - lambda) phi + 1 - phi), at the pair the law simulates a row with,
    1 - f(1 - Phi) and 1 - f(1 - lambda) at f = 0.6899 (0.527195), to 1e-9, the paths' mean realised
    share at the fixing within 0.005 (0.528442, a standard error of 0.0012). 1 s.

    Killing mutation: `basis_decay` reading the declared pair (0.394985 projected against the paths'
    0.528442)."""
    law = {'A': 0.0, 'Phi': 0.817, 'Nu': 5.0, 'Sigma': 0.5, 'Slow_Mean_Lambda': 0.96875, 'Mu_0': -2.35,
           'Calibration_DT_Years': 1.0 / 252.0}
    swap = {'Object': 'CommodityAveragePriceSwapDeal', 'Reference': 'APS', 'Currency': 'USD',
            'Commodity': 'LBMA_AM.CME', 'Carry': 'PLATINUM_CARRY', 'Discount_Rate': 'USD-SOFR', 'Buy_Sell': 'Buy',
            'Units': 1.0, 'Fixed_Price': 1630.0, 'Settlement_Date': {'.Timestamp': '2026-01-22'},
            'Sampling_Data': [[{'.Timestamp': '2026-01-20'}, 0.0, 1.0]]}

    def share(mark):                                    # the spot 1638.9, the deviation -5 off -2.35
        return (mark + 1630.0 - 1638.9 + 2.35) / -5.0

    f = 252.0 / 365.25
    phi, lam = 1.0 - f * (1.0 - 0.817), 1.0 - f * (1.0 - 0.96875)
    pi = (1.0 - lam) * phi / ((1.0 - lam) * phi + 1.0 - phi)
    mtm = _band_mtm(law, vol=0.0, flat_curves=True, deal=swap, Batch_Size=16384, Time_Grid='0d 1d(1d)',
                    Deterministic_Kernels='Yes')
    projected, realised = share(mtm.iloc[0].values), share(mtm.loc['2026-01-20'].values)
    assert np.abs(projected - (pi + (1.0 - pi) * (lam * phi) ** 5)).max() < 1e-9, projected[0]
    assert abs(realised.mean() - projected[0]) < 0.005, (realised.mean(), projected[0])


def test_a_coarse_row_on_the_trading_clock_walks_sub_steps_of_at_most_one_calibration_step():
    """The trading clock on a monthly grid, the first row's 31 calendar days f = 21.388 calibration
    steps walked as 21 whole steps and 0.388 of one: the credit Monte Carlo of the one future on flat
    curves, so the mtm's move over the month is the spot's plus the basis's.
    (a) The band's pull off and the stress chain at its stationary 20% (the week test's), the spot at
    no vol, 65,536 paths: the month's variance is f within 3%, the stress share held, on the monthly
    grid (21.309) as on the daily one (21.242).
    (b) The week test's pull, slow mean and GARCH: the walk's closed form within 3%, 15.937 (measured
    15.785); the daily grid's 31 rows of 0.69 steps are another discretisation of the law, 14.781
    (14.697).
    (c) A Linear law without innovation, the spot at 30% vol, 4,096 paths: the basis's month move is
    A w times the spot's, its twin at A = 0 the regressor, w = 0.33918 the closed form of the spot's
    move shared among the sub-steps by length, each share then pulled toward the slow mean, to 1e-9.
    (d) The average-price swap test's law on a weekly grid whose rows include the fixing and the
    settlement, steps of 7, 7 and 5 days to the fixing, 16,384 paths: the base-date mark projects the
    composition of the three steps' pairs over their sub-steps, 0.164246, to 1e-9, the paths'
    realised share within 0.005 (0.166791, a standard error of 0.0014). 2 s.

    Killing mutations: each row one step of f - (a) 53.02, the chain's probabilities outside [0, 1];
    (b) NaN, the run refusing its value; (c) 1.0; (d) 0.022854; the sub-steps reusing the row's draw
    - (a) 395.0; the spot's move on the first sub-step - (c) 0.1571, on the last 1.0; the projection
    at the pair of the grid's last step - (d) 0.673525.
    """
    f, grid = 31 * 252.0 / 365.25, {'vol': 0.0, 'flat_curves': True, 'Deterministic_Kernels': 'Yes'}
    walk = dict(BAND, **UNIT_MIX, Band_Beta=0.0, Mix_Stay_Stress=0.5, Mix_P0_Stress=0.2)
    pull = dict(BAND, Band_Kappa=0.0, Band_Beta=0.3, Slow_Mean_Lambda=0.8, Mu_0=-7.35, Mix_Q_Stress=0.5,
                Mix_Sigma_Quiet=1.0, Mix_Sigma_Stress=1.0, Mix_Stay_Stress=0.5, Mix_P0_Stress=0.5,
                G_Omega=0.05, G_Alpha=0.1, G_Beta=0.85, Sig2_0=4.0)

    def substeps(x):
        return [1.0] * int(x) + [x - int(x)]

    def month(law, time_grid, **calculation):
        mtm = _band_mtm(law, maturity='2026-02-20', Time_Grid=time_grid, **dict(grid, **calculation))
        return (mtm.loc['2026-02-15'] - mtm.loc['2026-01-15']).values

    def location(steps, h=4.0):
        cov = np.zeros((2, 2))
        for r in steps:
            m, step = 1.0 - 0.3 * r, 0.2 * r                  # the pull's and the slow mean's
            weights = np.array([[m, 1.0 - m], [step * m, 1.0 - step * m]])
            cov = weights @ cov @ weights.T + np.outer([1.0, step], [1.0, step]) * r * h
            h += r * (0.05 - 0.05 * h)
        return cov[0, 0]

    def carried(steps, phi=0.817, lam=0.96875):
        """The (b, mu) map the Linear law's sub-steps compose to, each the pull and then the slow
        mean's step."""
        total = np.eye(2)
        for r in steps:
            total = np.array([[1.0, 0.0], [r * (1.0 - lam), 1.0 - r * (1.0 - lam)]]) @ np.array(
                [[1.0 - r * (1.0 - phi), r * (1.0 - phi)], [0.0, 1.0]]) @ total
        return total

    for time_grid in ('0d 1m(1m)', '0d 1d(1d)'):
        assert abs(month(walk, time_grid, Batch_Size=65536).var() / f - 1.0) < 0.03, time_grid
    assert abs(month(pull, '0d 1m(1m)', Batch_Size=65536).var() / location(substeps(f)) - 1.0) < 0.03

    linear = {'Phi': 0.817, 'Nu': 5.0, 'Sigma': 0.0, 'Slow_Mean_Lambda': 0.96875, 'Mu_0': -2.35,
              'Calibration_DT_Years': 1.0 / 252.0}
    still, moved = (month(dict(linear, A=a), '0d 1m(1m)', Batch_Size=4096, vol=0.3) for a in (0.0, 0.5))
    spot, basis = still - still.mean(), moved - still - (moved - still).mean()
    steps = substeps(f)
    w = sum(r / f * (carried(steps[j + 1:]) @ [1.0, r * 0.03125])[0] for j, r in enumerate(steps))
    assert abs((basis * spot).sum() / (spot * spot).sum() / (0.5 * w) - 1.0) < 1e-9, w

    swap = {'Object': 'CommodityAveragePriceSwapDeal', 'Reference': 'APS', 'Currency': 'USD',
            'Commodity': 'LBMA_AM.CME', 'Carry': 'PLATINUM_CARRY', 'Discount_Rate': 'USD-SOFR', 'Buy_Sell': 'Buy',
            'Units': 1.0, 'Fixed_Price': 1630.0, 'Settlement_Date': {'.Timestamp': '2026-02-05'},
            'Sampling_Data': [[{'.Timestamp': '2026-02-03'}, 0.0, 1.0]]}
    mtm = _band_mtm(dict(linear, A=0.0, Sigma=0.5), deal=swap, Batch_Size=16384, Time_Grid='0d 1w(1w)',
                    Dynamic_Scenario_Dates='Yes', **grid)
    assert list(mtm.index.strftime('%m-%d')) == ['01-15', '01-22', '01-29', '02-03', '02-05', '02-06']
    projected, realised = ((mtm.loc[day].values + 1630.0 - 1638.9 + 2.35) / -5.0
                           for day in ('2026-01-15', '2026-02-03'))
    kept = carried([r for days in (7, 7, 5) for r in substeps(days * 252.0 / 365.25)])[0, 0]
    assert np.abs(projected - kept).max() < 1e-9, (projected[0], kept)
    assert abs(realised.mean() - kept) < 0.005, (realised.mean(), kept)
