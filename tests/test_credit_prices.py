"""Does the credit family recover the survival curve its CDS quotes came from?

The curve module's gate for a survival curve. A hazard rising over four segments and a sloped USD
discount curve are a known world, and the standard default swaps are spelt here BY HAND - bought
from the base date to the 20th of the quarter month each tenor reaches, quarterly ACT/360 with the
accrued paid on default - and priced off it by base valuation to MAKE their quotes: a par spread
from two marks at coupons of zero and a basis point, an upfront from two at upfronts of zero and a
percent at the standard coupon, the value being affine in either. The bootstrap is then required to
recover the curve, so a family authoring another contract misses it rather than repricing its own.

WHAT IS HELD:

  the round trip    every quote type, and a block mixing them, back to the curve to 1e-10
  the standard model the credit Monte Carlo's `CDS_Tenors` reprices the solved curve's quotes
  the knobs         read off the block and its family's entry
  the refusals      a contract maturing at the base date; a quote type the family does not take
  the order         a block discounting on a curve another family solves is solved after it
  the quote side    dtheta/dq against re-bootstrapped differences, and a default swap's dV/dq
                    beside its factor greek, the two one chain rule
  the emitter       a scripted terminal authors the blocks the family solves, a re-tick is a
                    values patch, and a seed entry without its conventions refuses by name

NO MONKEYPATCHING: every block is a JSON document bootstrapped and priced through `Context`, and
the terminal is a `BloombergSession` subclass whose event walk yields canned rows.
"""
import ast
import functools
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
import torch

import derivus
import rates_world
from derivus import bootstrappers, schema, utils
from derivus.config import CustomJsonEncoder, ModelParams
from derivus_bloomberg import cds, discover
from derivus_bloomberg.errors import BloombergConfigurationError
from derivus_bloomberg.session import BloombergSession

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

B = rates_world.BASE                      # a Monday
NAME = 'ISSUER'
BLOCK, CURVE = 'SurvivalProbPrices.' + NAME, 'SurvivalProb.' + NAME
FACTOR = utils.Factor('SurvivalProb', (NAME,))

#: The 20th of the quarter month each tenor reaches from the base date: what the family's standard
#: contract matures on, spelt by hand.
MATURITY = dict({'6M': pd.Timestamp('2027-03-20')},
                **{'{}Y'.format(years): pd.Timestamp('{}-09-20'.format(B.year + years))
                   for years in (1, 2, 3, 4, 5, 7, 10)})

#: The world: four contracts, a hazard rising over the four segments between their maturities, and
#: the cumulative hazard at the knots - the maturities on ACT/365 - the curve the quotes come from.
TENORS = ('1Y', '3Y', '5Y', '7Y')
KNOTS = np.array([(MATURITY[tenor] - B).days / 365.0 for tenor in TENORS])
HAZARDS = np.array([0.012, 0.018, 0.026, 0.031])
TRUE = np.cumsum(np.diff(KNOTS, prepend=0.0) * HAZARDS)

#: A USD curve sloped from 4.2% down to 3.8% and back to 4%, on whole-day knots so the curve family
#: can quote it on its own (`bootstrappers.knot_prices`).
FACTORS = rates_world.market('USD', {'USD': ((1.0, 2.0, 3.0, 5.0, 10.0),
                                             (0.042, 0.039, 0.0385, 0.038, 0.040))}, 'USD')


def survival(cumulative=TRUE, recovery=0.4):
    """`SurvivalProb.ISSUER` holding `cumulative` at the knots beside its anchor at tenor zero."""
    return {'Recovery_Rate': recovery, 'Curve': utils.Curve(
        [], [[0.0, 0.0]] + [[knot, value] for knot, value in zip(KNOTS, cumulative)])}


def contract(tenor, coupon, upfront=0.0, day_count='ACT_360', reference=None, **terms):
    """The standard default swap at `tenor`, spelt by hand: `coupon` in basis points and `upfront`
    in percent of a million bought, quarterly on `day_count` with the accrued paid on default."""
    return dict({
        'Object': 'DealDefaultSwap', 'Reference': reference or tenor, 'Currency': 'USD',
        'Discount_Rate': 'USD', 'Name': NAME, 'Buy_Sell': 'Buy', 'Principal': 1e6,
        'Effective_Date': B, 'Maturity_Date': MATURITY[tenor],
        'Pay_Frequency': pd.DateOffset(months=3), 'Accrual_Day_Count': day_count,
        'Accrue_Fee': 'Yes', 'Pay_Rate': utils.Basis(coupon), 'Upfront': utils.Percent(upfront)},
        **terms)


#: The calculation a document runs unless it states another.
VALUATION = {'Object': 'BaseValuation', 'MCMC_Simulations': 1, 'Random_Seed': 1}


def document(deals=(), factors=None, market_prices=None, entries=None, calculation=VALUATION):
    """A job running `calculation` over `deals` on the world's curve, `factors` beside it,
    carrying `market_prices` and the credit family's `Bootstrapper Configuration` entry or
    `entries`."""
    return {'Calc': {
        'Calculation': dict({'Base_Date': B, 'Currency': 'USD'}, **calculation),
        'Deals': {'Reference': 'credit', 'Deals': {'Children': [
            {'Instrument': {'.Deal': deal}} for deal in deals]}},
        'MergeMarketData': {'MarketDataFile': '', 'ExplicitMarketData': {
            'System Parameters': {'Base_Currency': 'USD', 'Base_Date': B},
            'Price Factors': dict(FACTORS, **(factors or {})),
            'Market Prices': market_prices or {},
            'Bootstrapper Configuration': entries or {'SurvivalProb': {'Prices': 'SurvivalProb'}}}}}}


def loaded(job):
    context = derivus.Context()
    context.load_json((json.dumps(job, cls=CustomJsonEncoder), 'credit'))
    return context


def marks(deals, factors=None, **calculation):
    """`({reference: mark}, results)` for `deals` by base valuation."""
    results = loaded(document(deals, factors, calculation=dict(VALUATION, **calculation))
                     ).run_job()[1]['Results']
    return dict(zip(results['mtm']['Reference'], results['mtm']['Value'])), results


@functools.lru_cache(maxsize=None)
def quotes(quote_type, day_count='ACT_360', recovery=0.4, cumulative=tuple(TRUE)):
    """Each tenor's `quote_type` quote off the curve: the root of the marks at quote units zero and
    one, the value being affine in the quote."""
    upfront = quote_type == 'Upfront'
    legs = [contract(tenor, 100.0 if upfront else unit, unit if upfront else 0.0, day_count,
                     '{} {}'.format(tenor, unit)) for tenor in TENORS for unit in (0, 1)]
    valued, _ = marks(legs, {CURVE: survival(np.array(cumulative), recovery)})
    return tuple(valued[tenor + ' 0'] / (valued[tenor + ' 0'] - valued[tenor + ' 1'])
                 for tenor in TENORS)


def block(quote_types, **fields):
    """The `SurvivalProbPrices.ISSUER` block quoting the world's four contracts, row by row in
    `quote_types` (one type for every row where it is a string), `fields` over the block."""
    types = [quote_types] * len(TENORS) if isinstance(quote_types, str) else quote_types
    priced = (fields.get('Accrual_Day_Count', 'ACT_360'), fields.get('Recovery_Rate', 0.4))
    return {'instrument': dict({'Currency': 'USD', 'Discount_Rate': 'USD', 'Points': [
        {'Use': 'Yes', 'Descriptor': '{} {}'.format(NAME, tenor), 'Tenor': tenor,
         'DealType': 'DealDefaultSwap', 'Quote_Type': kind,
         'Quoted_Market_Value': quotes(kind, *priced)[index], 'Deal': {}}
        for index, (tenor, kind) in enumerate(zip(TENORS, types))]}, **fields)}


def bootstrapped(market_prices, deals=(), factors=None, entries=None, **calculation):
    """A context over the document, bootstrapped."""
    context = loaded(document(deals, factors, market_prices, entries,
                              dict(VALUATION, **calculation)))
    context.bootstrap()
    return context


def curve(context):
    return context.current_cfg.params['Price Factors'][CURVE]['Curve'].array


@pytest.mark.parametrize('quote_types', ['Par_Spread', 'Upfront',
                                         ('Par_Spread', 'Upfront', 'Par_Spread', 'Upfront')])
def test_the_bootstrap_recovers_the_survival_curve_its_quotes_came_from(quote_types):
    """THE ROUND TRIP: the curve back at its knots to 1e-10 relative, measured within 1.1e-14 off
    par spreads, 5.3e-15 off upfronts and 9.3e-15 off a block mixing the two, in four Newton
    iterations. The knots are the maturities by hand on ACT/365 beside the anchor at zero, and the
    seed - each row's running spread as a flat hazard, an upfront's percent over its years added
    to the coupon - sits 4.5% off the answer off par spreads and 8.4% off upfronts, so a solver
    returning it fails.

    A par spread block and an upfront block of the same four contracts write one curve to 1.1e-14
    and not to the bit: two residuals with one root, Newton lands on it through two roundings.

    KILLING MUTATIONS, by the relative miss: the solver returning its seed, 4.5e-2 and 8.4e-2; the
    maturity the tenor from the base date rather than the standard date it reaches, the 1Y knot
    at 1.0 against 1.1315; the accrued on default left unpaid, 2.9e-3 and 2.2e-3; an upfront
    priced at no coupon, 1.40; the anchor solved for with the knots, a 4 by 5 Jacobian refusing.
    """
    kinds = [quote_types] * len(TENORS) if isinstance(quote_types, str) else quote_types
    seed = np.array([
        (quotes(kind)[index] / 1e4 if kind == 'Par_Spread' else 1e-2 + quotes(kind)[index] / 100.0 /
         knot) * knot / 0.6 for index, (kind, knot) in enumerate(zip(kinds, KNOTS))])
    assert np.abs(seed / TRUE - 1.0).max() > 1e-2, 'the seed is already the answer'

    solved = curve(bootstrapped({BLOCK: block(quote_types)}))
    assert solved.dtype == np.float64
    assert np.array_equal(solved[:, 0], np.concatenate([[0.0], KNOTS])), solved[:, 0]
    assert solved[0, 1] == 0.0, 'the anchor moved'
    error = np.abs(solved[1:, 1] / TRUE - 1.0).max()
    assert error < 1e-10, 'recovered to {:.3g}, not 1e-10'.format(error)

    if quote_types == 'Upfront':
        spreads = curve(bootstrapped({BLOCK: block('Par_Spread')}))
        assert np.abs(solved[1:, 1] / spreads[1:, 1] - 1.0).max() < 1e-13


def standard_model(factor):
    """`{tenor in years: par spread}` the ISDA standard model reads off `factor` - the credit Monte
    Carlo's `CDS_Tenors`, its counterparty the name, a CVA gradient asked for beside a cashflow."""
    cashflow = {'Object': 'FixedCashflowDeal', 'Reference': 'CF', 'Currency': 'USD',
                'Discount_Rate': 'USD', 'Amount': 1e6, 'Payment_Date': B + pd.DateOffset(years=1)}
    job = document([], {CURVE: factor}, calculation={
        'Object': 'CreditMonteCarlo', 'Time_Grid': '0d 3m(3m)', 'Batch_Size': 16,
        'Random_Seed': 1, 'Deflation_Interest_Rate': 'USD', 'Credit_Valuation_Adjustment': {
            'Calculate': 'Yes', 'Counterparty': NAME, 'Gradient': 'Yes',
            'Deflate_Stochastically': 'No', 'Stochastic_Hazard_Rates': 'No',
            'CDS_Tenors': [float(tenor[:-1]) for tenor in TENORS]}})
    job['Calc']['Deals']['Deals']['Children'] = [{'Instrument': {'.Deal': {
        'Object': 'NettingCollateralSet', 'Reference': 'NS', 'Netted': 'True',
        'Collateralized': 'False'}}, 'Children': [{'Instrument': {'.Deal': cashflow}}]}]
    market = job['Calc']['MergeMarketData']['ExplicitMarketData']
    market['Model Configuration'] = {'.ModelParams': {
        'modeldefaults': {'InterestRate': 'HullWhite1FactorInterestRateModel'}, 'modelfilters': {}}}
    market['Price Models'] = {'HullWhite1FactorInterestRateModel.USD': {
        'Alpha': 0.05, 'Lambda': 0.0, 'Quanto_FX_Correlation': 0.0,
        'Quanto_FX_Volatility': utils.Curve([], [[0.0, 0.0], [10.0, 0.0]]),
        'Sigma': utils.Curve([], [[0.0, 0.01], [10.0, 0.01]])}}
    frame = loaded(job).run_job()[1]['Results']['CS01']
    return dict(zip(frame.columns.get_level_values('Tenor'),
                    frame.columns.get_level_values('Par CDS')))


def test_the_standard_model_reprices_the_solved_curves_quotes_to_its_conventions():
    """AGAINST THE ISDA STANDARD MODEL (`utils.calc_cds_rates`), which integrates protection and
    the accrued on default exactly, accrues on the curve's own ACT/365 year and subtracts the
    coupon accrued since the last standard date: on a block accruing ACT/365 at a 25% recovery,
    the par spreads it reads off the solved curve sit 3.0e-3, 1.0e-3, 6.0e-4 and 4.1e-4 over the
    quotes at 1Y, 3Y, 5Y and 7Y, held under 5e-3. The maturity day the default swap counts in its
    last accrual and the model does not is the most of it: without it the gap is 6.0e-4, 2.2e-4,
    1.4e-4 and 1.0e-4, the first coupon accrued from the base date against the accrued subtracted
    the most of that, and the midpoint form of the protection leg ~1e-5 a period.

    KILLING MUTATIONS: the curve written at the declared 40% recovery rather than the block's 25%
    it was priced at, every par spread 19.8% to 20.0% under its quote; the block's day count
    unread and the contracts accruing ACT/360, every par spread 1.4% to 1.7% over.
    """
    fields = {'Accrual_Day_Count': 'ACT_365', 'Recovery_Rate': 0.25}
    solved = bootstrapped({BLOCK: block('Par_Spread', **fields)})
    factor = solved.current_cfg.params['Price Factors'][CURVE]
    assert factor['Recovery_Rate'] == 0.25
    par = standard_model(factor)
    gap = np.array([par[float(tenor[:-1])] * 1e4 for tenor in TENORS]) / np.array(
        quotes('Par_Spread', 'ACT_365', 0.25)) - 1.0
    assert ((gap > 0.0) & (gap < 5e-3)).all(), gap


@pytest.mark.parametrize('knob,value,where', [('N_Iter', 1, 'block'),
                                              ('Damping_Halvings', -1, 'block'),
                                              ('N_Iter', 1, 'entry')])
def test_the_solver_knobs_are_read_off_the_block_and_its_entry(knob, value, where):
    """The curve family's knobs, declared on this family as the same fields: one Newton iteration
    cannot converge four knots from a flat-hazard seed, and `-1` halvings forbids even the full
    step. A knob stated on the family's `Bootstrapper Configuration` entry reaches every block
    that does not state its own.

    Killing mutation: the iteration cap read off a constant rather than the block - one iteration
    allowed and the solve converging.
    """
    entries = {'SurvivalProb': {'Prices': 'SurvivalProb', knob: value}} if where == 'entry' else None
    with pytest.raises(Exception, match='Curve bootstrap'):
        bootstrapped({BLOCK: block('Par_Spread', **({knob: value} if where == 'block' else {}))},
                     entries=entries)


def test_a_quote_maturing_at_the_base_date_refuses_by_name():
    """A row whose own `Deal` matures its contract ON the base date - a seasoned quarter's swap -
    wants its knot at tenor zero, where the anchor is and no quote identifies anything, so the
    family refuses naming the block, the row, its last cashflow, the base date and the remedies.
    With that row's `Use` off the block solves and recovers its curve, which says the refusal
    reads the tenor.

    Killing mutation: the refusal dropped - the solve dies on `linalg.solve`'s "5 by 4 matrices",
    naming no quote.
    """
    market = {BLOCK: block('Par_Spread')}
    market[BLOCK]['instrument']['Points'].insert(0, {
        'Use': 'Yes', 'Descriptor': 'ISSUER seasoned', 'Tenor': '3M', 'DealType': 'DealDefaultSwap',
        'Quote_Type': 'Par_Spread', 'Quoted_Market_Value': 60.0,
        'Deal': {'Effective_Date': B - pd.DateOffset(months=3), 'Maturity_Date': B}})
    with pytest.raises(ValueError) as refusal:
        bootstrapped(market)
    message = str(refusal.value)
    assert message.startswith(BLOCK), message
    assert 'ISSUER seasoned (last cashflow {:%Y-%m-%d})'.format(B) in message, message
    assert 'base date {:%Y-%m-%d}'.format(B) in message, message
    assert 'tenor zero' in message and 'Use No' in message, message

    market[BLOCK]['instrument']['Points'][0]['Use'] = 'No'
    assert np.abs(curve(bootstrapped(market))[1:, 1] / TRUE - 1.0).max() < 1e-10


def test_a_quote_type_the_family_does_not_take_is_refused_by_name_and_left_out(caplog):
    """`takes` is the curve family's: a `Quote_Type` the default swap declares no landing for is
    logged by name at ERROR - on the service the channel that refuses the write - and left out,
    so a conventional spread is never solved as a par spread unsaid. The 7Y row typed so drops
    the 7Y knot and the three before it are the curve's to 1e-10, a knot reading no later quote.

    Killing mutation: `takes` admitting any type - the default swap has no field for it and the
    solve dies on `KeyError: 'Conventional_Spread'`, naming no row.
    """
    market = {BLOCK: block('Par_Spread')}
    market[BLOCK]['instrument']['Points'][-1]['Quote_Type'] = 'Conventional_Spread'
    with caplog.at_level(logging.ERROR):
        solved = curve(bootstrapped(market))
    assert ('{} quote ISSUER 7Y - Quote_Type Conventional_Spread is no default swap quote; the '
            'family takes Par_Spread or Upfront'.format(BLOCK)) in caplog.text, caplog.text
    assert np.array_equal(solved[:, 0], np.concatenate([[0.0], KNOTS[:-1]]))
    assert np.abs(solved[1:, 1] / TRUE[:-1] - 1.0).max() < 1e-10


def test_a_block_discounting_on_a_curve_another_family_solves_is_solved_after_it():
    """The book carries a STALE USD curve, 1% flat, and an `InterestRatePrices.USD` block quoting
    the world's curve on its own knots (`bootstrappers.knot_prices`); the entries list the credit
    family first. `bootstrap_order` runs the curve family first because this one `reads` what it
    writes, so the survival curve is solved on the curve its quotes were priced on and is the
    world's to 1e-10, the USD curve recovered to 1e-10 beside it.

    Killing mutation: `reads` emptied - the file's order stands, the survival curve is solved on
    the stale curve and misses by 1.4e-2.
    """
    name, curve_block = bootstrappers.knot_prices('USD', FACTORS, ModelParams(), B, {})
    stale = {'InterestRate.USD': dict(FACTORS['InterestRate.USD'],
                                      Curve=utils.Curve([], [[1.0, 0.01], [10.0, 0.01]]))}
    entries = {'SurvivalProb': {'Prices': 'SurvivalProb'}, 'InterestRate': {'Prices': 'InterestRate'}}
    context = bootstrapped({BLOCK: block('Par_Spread'), name: curve_block}, factors=stale,
                           entries=entries)
    solved = context.current_cfg.params['Price Factors']
    assert np.abs(solved['InterestRate.USD']['Curve'].array[:, 1] -
                  FACTORS['InterestRate.USD']['Curve'].array[:, 1]).max() < 1e-10
    assert np.abs(curve(context)[1:, 1] / TRUE - 1.0).max() < 1e-10


def jacobian(config):
    """`dtheta/dq` the implicit function theorem gives, a row per knot, off the connected curve."""
    theta = config.calibrated_factors[FACTOR]
    leaves = config.quote_leaves[BLOCK][1]
    return np.array([torch.autograd.grad(theta[i], leaves, retain_graph=True, allow_unused=True)[0]
                     .cpu().numpy() for i in range(theta.numel())])


def moved(quote_type, index, step, **fields):
    """The curve re-bootstrapped with quote `index` moved by `step`."""
    market = {BLOCK: block(quote_type, **fields)}
    market[BLOCK]['instrument']['Points'][index]['Quoted_Market_Value'] += step
    return curve(bootstrapped(market))[:, 1]


@pytest.mark.parametrize('quote_type,step', [('Par_Spread', 1e-3), ('Upfront', 1e-5)])
def test_the_quote_derivative_is_the_re_bootstrapped_difference(quote_type, step):
    """`Quote_Sensitivity` leaves the solved curve connected to its quotes through
    `utils.CalibrationSolve`, and `dtheta/dq` is the implicit function theorem: against central
    differences of full re-bootstraps a thousandth of a basis point or a hundred-thousandth of a
    percent either side, within 1e-8 of the largest entry, measured within 3.8e-11 and 1.7e-10.
    The anchor's row is zero, and the switch writes the curve it writes off, bit for bit.

    KILLING MUTATIONS: the bumped compile authored a hundred units over the quote rather than one,
    every entry a hundred times the difference; the theorem's sign dropped, every entry negated.
    """
    context = bootstrapped({BLOCK: block(quote_type, Quote_Sensitivity='Yes')})
    assert np.array_equal(curve(context), curve(bootstrapped({BLOCK: block(quote_type)})))
    derivative = jacobian(context.current_cfg)
    assert not derivative[0].any(), 'the anchor moves with a quote'

    reference = np.array([(moved(quote_type, j, step) - moved(quote_type, j, -step)) / (2 * step)
                          for j in range(len(TENORS))]).T
    error = np.abs(derivative - reference).max() / np.abs(reference).max()
    assert error < 1e-8, 'dtheta/dq is not the re-bootstrapped difference: {:.3g}'.format(error)


def test_a_default_swap_on_the_name_reports_its_quote_delta_beside_its_factor_greek():
    """A default swap on the bootstrapped name - protection sold on five million to 3 February
    2031, a date no quote matures on, at a 250bp coupon - under `Greeks: First`: its `dV/dq` lands
    on the block's quote leaf in the pass that reports its factor greek `dV/dtheta`, and the two
    are one chain rule on the 5Y quote, `dV/dq = sum_i dV/dtheta_i dtheta_i/dq` with `dtheta/dq`
    re-bootstrapped, and the difference of the deal re-valued on curves re-solved a thousandth of
    a basis point either side, both within 1e-8 (-1,606.76 a basis point, measured within 3.5e-11
    and 4.3e-11).

    KILLING MUTATIONS: the connected curve harvested with its anchor last rather than first, the
    5Y quote delta -847.64; the implicit function theorem's sign dropped, +1,606.76.
    """
    deal = dict(contract('5Y', 250.0, reference='SOLD'), Buy_Sell='Sell', Principal=5e6,
                Maturity_Date=pd.Timestamp('2031-02-03'))
    context = bootstrapped({BLOCK: block('Par_Spread', Quote_Sensitivity='Yes')}, [deal],
                           Greeks='First')
    _, out = context.run_job()
    # a knot the deal reads nothing past is reported nowhere, its greek zero
    greeks = out['Results']['Greeks_First'].xs(CURVE, level='Rate')['root']
    factor_greek = greeks.droplevel(['Tenor2', 'Tenor3']).reindex(
        np.concatenate([[0.0], KNOTS]), fill_value=0.0).to_numpy()
    quote_delta = context.current_cfg.quote_leaves[BLOCK][1].grad.cpu().numpy()

    index, step = TENORS.index('5Y'), 1e-3
    dtheta = (moved('Par_Spread', index, step) - moved('Par_Spread', index, -step)) / (2 * step)
    chained = factor_greek @ dtheta
    assert abs(quote_delta[index] / chained - 1.0) < 1e-8, (quote_delta[index], chained)

    revalued = []
    for sign in (1.0, -1.0):
        market = {BLOCK: block('Par_Spread')}
        market[BLOCK]['instrument']['Points'][index]['Quoted_Market_Value'] += sign * step
        factors = {CURVE: bootstrapped(market).current_cfg.params['Price Factors'][CURVE]}
        revalued.append(marks([deal], factors)[0]['SOLD'])
    difference = (revalued[0] - revalued[1]) / (2 * step)
    assert abs(quote_delta[index] / difference - 1.0) < 1e-8, (quote_delta[index], difference)


# =============================================================================================
# the emitter, off a scripted terminal
# =============================================================================================

AS_OF = B.date()
YESTERDAY = (B - pd.Timedelta(days=3)).date().isoformat()   # the Friday before


def packaged_credit():
    """The shipped seed's credit section: two names and an index, each with its conventions."""
    with open(os.path.join(ROOT, 'derivus_bloomberg', 'seed.json'), encoding='utf-8') as handle:
        return {'credit': json.load(handle)['credit']}


#: What the terminal prints, invented: GS's spread rising with tenor and its price beside it on
#: the five middle tenors, the index's spread alone, and its 10Y quoted against another coupon.
SPREADS = {'6M': 22.0, '1Y': 30.0, '2Y': 41.0, '3Y': 52.0, '4Y': 63.0, '5Y': 74.0, '7Y': 92.0,
           '10Y': 108.0}
PRICES = {'1Y': 100.7, '2Y': 101.2, '3Y': 101.4, '4Y': 101.3, '5Y': 101.1}


class Walked(BloombergSession):
    """A session whose event walk is canned rows, so the emitter is driven through the package's
    real reader with no socket and no patching."""

    def __init__(self, rows):
        super().__init__()
        self._api = self._session = self._service = object()
        self.rows = rows

    def _walk(self, securities, fields):
        for security in securities:
            if security in self.rows:
                yield security, None, {key: value for key, value in self.rows[security].items()
                                       if key in fields}


def terminal(seed, spreads=SPREADS, prices=PRICES):
    """`(verified map, session)`: the map discovery builds off canned NAME answers carrying each
    entry's `expect`, and the terminal answering every contract its spread, two-way and print
    date, GS its price where `prices` has one and the index its 10Y against a 500bp coupon."""
    candidates = list(discover.candidates_from_seed(seed))
    report = {candidate.security: {'ok': True, 'error': None, 'fields': {
        'NAME': '{} {}'.format(candidate.expect[0], candidate.path[-1]), 'PX_LAST': 1.0,
        'LAST_UPDATE_DT': YESTERDAY}} for candidate in candidates}
    mapped = discover.build_map(seed, discover.verify(candidates, report, AS_OF),
                                AS_OF.isoformat())
    rows = {}
    for candidate in candidates:
        name, tenor = candidate.path[1:]
        spread = spreads[tenor] + 20.0 * (name == 'CDX-IG')
        rows[candidate.security] = dict(
            {'PX_LAST': spread, 'PX_BID': spread - 0.5, 'PX_ASK': spread + 0.5,
             'LAST_UPDATE_DT': YESTERDAY,
             'SW_SPREAD': 500.0 if (name, tenor) == ('CDX-IG', '10Y') else 100.0},
            **({'CDS_QUOTED_PRICE': prices[tenor]} if name == 'GS' and tenor in prices else {}))
    return mapped, Walked(rows)


def emitted(seed, name, **answers):
    """`(Market Prices name, block, strip)` the emitter authors off the scripted terminal, the
    block in the wire form a job carries."""
    mapped, session = terminal(seed, **answers)
    strip = cds.fetch_cds_strip(session, mapped, seed, name, AS_OF)
    market_price, authored = cds.cds_block(strip)
    return market_price, json.loads(json.dumps(authored)), strip


def test_the_terminal_authors_the_blocks_the_family_solves():
    """A single name and an index off one scripted terminal. GS's five middle tenors answer
    `CDS_QUOTED_PRICE` and are `Upfront` rows at a hundred less the price, its 6M, 7Y and 10Y
    answer the spread alone and are `Par_Spread` rows with the two-way; the index answers no price
    and its 10Y, quoted against a 500bp coupon, is refused `off-coupon` by name. The seed's
    conventions are emitted beside the rows, and the family solves both blocks: one knot per row
    at its standard maturity, and every row's contract spelt by hand at its own quote marking
    within a millionth of the principal of par on the solved curve.

    Killing mutation: the price read as the upfront, every GS upfront row a hundred points out
    and the solve off the market.
    """
    seed = packaged_credit()
    name, gs, _ = emitted(seed, 'GS')
    rows = gs['instrument']['Points']
    assert name == 'SurvivalProbPrices.GS'
    assert [(row['Tenor'], row['Quote_Type']) for row in rows] == [
        (tenor, 'Upfront' if tenor in PRICES else 'Par_Spread') for tenor in SPREADS]
    assert [row['Quoted_Market_Value'] for row in rows] == [
        100.0 - PRICES[tenor] if tenor in PRICES else SPREADS[tenor] for tenor in SPREADS]
    assert all(('Quoted_Bid' in row) == (row['Quote_Type'] == 'Par_Spread') for row in rows)
    assert all(row['Deal'] == {} and row['Timestamp'] == {'.Timestamp': YESTERDAY} for row in rows)
    conventions = seed['credit']['GS']['conventions']
    assert {key: value for key, value in gs['instrument'].items() if key != 'Points'} == {
        'Currency': conventions['currency'], 'Discount_Rate': '',
        'Recovery_Rate': conventions['recovery_rate'], 'Standard_Coupon': conventions['coupon'],
        'Pay_Frequency': {'.DateOffset': conventions['pay_frequency']},
        'Accrual_Day_Count': conventions['accrual_day_count']}

    index_name, index, strip = emitted(seed, 'CDX-IG')
    assert [row['Tenor'] for row in index['instrument']['Points']] == ['3Y', '5Y']
    assert {row['Quote_Type'] for row in index['instrument']['Points']} == {'Par_Spread'}
    assert strip.rejected == {'CDX IG CDSI GEN 10Y Corp': 'off-coupon'}

    context = loaded(document(market_prices=json.loads(json.dumps(
        {name: gs, index_name: index}, cls=CustomJsonEncoder))))
    context.bootstrap()
    factors = context.current_cfg.params['Price Factors']
    for market_price, authored in ((name, gs), (index_name, index)):
        written = factors['SurvivalProb.' + market_price.split('.', 1)[1]]
        points = authored['instrument']['Points']
        assert np.array_equal(written['Curve'].array[:, 0], [0.0] + [
            (MATURITY[row['Tenor']] - B).days / 365.0 for row in points])
        legs = [dict(contract(row['Tenor'], 100.0 if row['Quote_Type'] == 'Upfront' else
                              row['Quoted_Market_Value'], row['Quoted_Market_Value'] if
                              row['Quote_Type'] == 'Upfront' else 0.0, reference=row['Tenor']),
                     Name=market_price.split('.', 1)[1]) for row in points]
        valued, _ = marks(legs, {key: value for key, value in factors.items()
                                 if key.startswith('SurvivalProb.')})
        assert max(abs(value) for value in valued.values()) < 1.0, (market_price, valued)


def test_a_re_tick_is_a_values_patch_and_a_row_changing_its_type_a_new_plan():
    """The rows name no date and their `Deal` no number, so the same contracts printing other
    spreads and prices the next day are a VALUES change: `schema.update_market_quote` updates the
    installed block in place. A tenor whose price stops answering is a row of another type, which
    the guard refuses as a moved plan by name.

    Killing mutation: the quote authored into the row's `Deal` - every re-tick a new plan.
    """
    seed = packaged_credit()
    name, first, _ = emitted(seed, 'GS')
    job = {'Calc': {'MergeMarketData': {'ExplicitMarketData': {'Market Prices': {}}}}}
    assert schema.update_market_quote(job, name, first) == 'installed'

    _, ticked, _ = emitted(seed, 'GS', spreads={tenor: value + 3.0 for tenor, value in
                                                SPREADS.items()},
                           prices={tenor: value - 0.2 for tenor, value in PRICES.items()})
    assert ticked != first
    assert schema.update_market_quote(job, name, ticked) == 'updated'

    _, retyped, _ = emitted(seed, 'GS', prices={'1Y': 100.7})
    with pytest.raises(ValueError, match='structure differs'):
        schema.update_market_quote(job, name, retyped)


def test_a_seed_entry_without_its_conventions_refuses_by_name():
    """A CDS quote is not a contract until something says what it is quoted against, so a seed
    entry with no `conventions` refuses naming the entry, and one declaring half of them refuses
    naming every field it lacks at once.

    Killing mutation: a missing coupon defaulted rather than refused.
    """
    seed = packaged_credit()
    del seed['credit']['GS']['conventions']
    with pytest.raises(BloombergConfigurationError, match='GS carries no `conventions` block'):
        emitted(seed, 'GS')
    seed = packaged_credit()
    for field in ('coupon', 'recovery_rate'):
        del seed['credit']['CDX-IG']['conventions'][field]
    with pytest.raises(BloombergConfigurationError, match='CDX-IG declares no coupon, recovery_rate'):
        emitted(seed, 'CDX-IG')


def test_the_emitter_imports_the_standard_library_and_its_package_alone():
    """`cds` is reached on a workstation with no engine, so its imports, read off its source, are
    the standard library's and its package's.

    Killing mutation: the emitter importing the engine for `utils.cds_dates`.
    """
    with open(os.path.join(ROOT, 'derivus_bloomberg', 'cds.py'), encoding='utf-8') as handle:
        tree = ast.parse(handle.read())
    names = {alias.name.split('.')[0] for node in ast.walk(tree) if isinstance(node, ast.Import)
             for alias in node.names} | {
        (node.module or '').split('.')[0] for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and not node.level}
    assert names == {'collections', 'dataclasses'}, names
