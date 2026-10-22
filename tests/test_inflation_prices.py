"""Does the inflation family recover the curve its zero-coupon swap breakevens came from?

The curve module's gate for an inflation curve, on the inflation list module's index: printed
monthly to June 2026 at 0.2% a month. A growth curve rising over six knots and a sloped USD
discount curve are a known world, and the swaps are spelt here BY HAND - an inflation list row
paying a million grown by the index ratio from the base date's reference to the maturity's,
beside a fixed cashflow of the million - and priced off it by base valuation to MAKE their quotes:
the breakeven is the T-th root of the two marks' ratio, both legs paying on one day. The bootstrap
is then required to recover the curve, so a family authoring another contract misses it rather
than repricing its own.

WHAT IS HELD:

  the round trip    both reference rules back to the curve to 1e-10, the closed form exact where
                    the base reference is the last print and stated where it is not
  the linker        the list module's linker marks on the solved curve as on the curve it came from
  the order         a block discounting on a curve another family solves is solved after it
  the narrowing     a curve move re-solves the inflation curve solved on it and no other
  the quote side    dtheta/dq against re-bootstrapped differences, a linker's dV/dq beside its
                    factor greek, the two one chain rule, and its swap quote deltas against both
                    curves refitted
  the refusals      a quote referencing only printed months; an index the market does not carry
  the emitter       a scripted terminal authors the block and the index the family solves, and a
                    re-tick is a values patch

NO MONKEYPATCHING: every block is a JSON document bootstrapped and priced through `Context`, and
the terminal is a `BloombergSession` subclass whose event walks yield canned rows.
"""
import copy
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
import trial_rates
from derivus import bootstrappers, schema, utils
from derivus.config import CustomJsonEncoder, ModelParams
from derivus_bloomberg import discover
from derivus_bloomberg.errors import BloombergConfigurationError, IncompleteStrip, InvalidQuote
from derivus_bloomberg.inflation import fetch_inflation_strip, inflation_block
from derivus_bloomberg.session import BloombergSession

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

B = rates_world.BASE                      # 2026-08-03
NAME = 'INFL'
BLOCK, CURVE = 'InflationRatePrices.' + NAME, 'InflationRate.' + NAME
FACTOR = utils.Factor('InflationRate', (NAME,))
LAST = trial_rates.FACTORS['PriceIndex.INFL']['Last_Period_Start']      # June 2026's print

#: The world: six swaps, and the growth rates at their knots - the June each maturity's
#: reference reads last, three months back interpolated or two back single, on ACT/365 from the
#: last print - the curve the quotes come from.
TENORS = (1, 2, 3, 5, 7, 10)
KNOTS = np.array([(pd.Timestamp('{}-06-01'.format(B.year + years)) - LAST).days / 365.0
                  for years in TENORS])
TRUE = np.array([0.021, 0.0235, 0.0245, 0.026, 0.0255, 0.027])

#: The rules a reference reads by, `(Reference_Type, Months_Lag)`: the block's declared one, and
#: one whose base reference on this base date is the last print itself.
INTERPOLATED, SINGLE = ('Interpolated', 3), ('Single', 2)

#: A USD curve sloped from 4.2% down to 3.8% and back to 4%, on whole-day knots so the curve family
#: can quote it on its own (`bootstrappers.knot_prices`), beside the list module's index.
FACTORS = dict(rates_world.market('USD', {'USD': ((1.0, 2.0, 3.0, 5.0, 10.0),
                                                  (0.042, 0.039, 0.0385, 0.038, 0.040))}, 'USD'),
               **trial_rates.FACTORS)

#: The calculation a document runs unless it states another.
VALUATION = {'Object': 'BaseValuation', 'MCMC_Simulations': 1}

#: Both families' `Bootstrapper Configuration` entries, the inflation family's listed first.
FAMILIES = {'InflationRate': {'Prices': 'InflationRate'}, 'InterestRate': {'Prices': 'InterestRate'}}

#: The list module's seasoned linker, its base print stated and its finals forecast.
LINKER = next(deal for deal in trial_rates.DEALS if deal['Reference'] == 'LINKER')


def inflation(rates=TRUE):
    """`InflationRate.INFL` holding `rates` at the knots, on the list module's index."""
    return dict(trial_rates.FACTORS['InflationRate.INFL'], Curve=utils.Curve(
        [], [[knot, rate] for knot, rate in zip(KNOTS, rates)]))


def legs(maturity, rule, reference):
    """The swap to `maturity`, spelt by hand: the inflation row paying a million grown from the
    base date's reference to the maturity's under `rule`, and the fixed million paid beside it."""
    kind, lag = rule
    return [{'Object': 'YieldInflationCashflowListDeal', 'Reference': reference + ' index',
             'Currency': 'USD', 'Discount_Rate': 'USD', 'Buy_Sell': 'Buy', 'Index': NAME,
             'Index_Reference': {'Months_Lag': lag, 'Reference_Type': kind},
             'Cashflows': {'Items': [{
                 'Payment_Date': maturity, 'Notional': 1e6, 'Base_Reference_Date': B,
                 'Base_Reference_Value': 0.0, 'Final_Reference_Date': maturity,
                 'Final_Reference_Value': 0.0, 'Accrual_Start_Date': B, 'Accrual_End_Date': maturity,
                 'Accrual_Year_Fraction': 1.0, 'Yield': utils.Percent(100.0),
                 'Margin': utils.Basis(0.0), 'Rate_Multiplier': 1.0, 'Is_Coupon': 'No'}]}},
            {'Object': 'FixedCashflowDeal', 'Reference': reference + ' fixed', 'Currency': 'USD',
             'Discount_Rate': 'USD', 'Amount': 1e6, 'Payment_Date': maturity}]


def document(deals=(), factors=None, market_prices=None, entries=None, calculation=VALUATION):
    """A job running `calculation` over `deals` on the world, `factors` over it - one stated None
    left out - carrying `market_prices` and the inflation family's `Bootstrapper Configuration`
    entry or `entries`."""
    return {'Calc': {
        'Calculation': dict({'Base_Date': B, 'Currency': 'USD'}, **calculation),
        'Deals': {'Reference': 'inflation', 'Deals': {'Children': [
            {'Instrument': {'.Deal': deal}} for deal in deals]}},
        'MergeMarketData': {'MarketDataFile': '', 'ExplicitMarketData': {
            'System Parameters': {'Base_Currency': 'USD', 'Base_Date': B},
            'Price Factors': {key: value for key, value in dict(FACTORS, **(factors or {})).items()
                              if value is not None},
            'Market Prices': market_prices or {},
            'Bootstrapper Configuration': entries or {'InflationRate': {'Prices': 'InflationRate'}}}}}}


def loaded(job):
    context = derivus.Context()
    context.load_json((json.dumps(job, cls=CustomJsonEncoder), 'inflation'))
    return context


def marks(deals, factors=None, **calculation):
    """`({reference: mark}, results)` for `deals` by base valuation."""
    results = loaded(document(deals, factors, calculation=dict(VALUATION, **calculation))
                     ).run_job()[1]['Results']
    return dict(zip(results['mtm']['Reference'], results['mtm']['Value'])), results


@functools.lru_cache(maxsize=None)
def quotes(rule=INTERPOLATED, rates=tuple(TRUE)):
    """Each tenor's breakeven off the curve, in percent: the T-th root of the index leg's mark over
    the fixed million's, the two paying on one day."""
    valued, _ = marks([leg for years in TENORS for leg in legs(
        B + pd.DateOffset(years=years), rule, str(years))], {CURVE: inflation(np.array(rates))})
    return tuple(100.0 * ((valued['{} index'.format(years)] / valued['{} fixed'.format(years)]) ** (
        1.0 / years) - 1.0) for years in TENORS)


def block(rule=INTERPOLATED, **fields):
    """The `InflationRatePrices.INFL` block quoting the world's six swaps under `rule`, `fields`
    over it."""
    return {'instrument': dict({
        'Currency': 'USD', 'Discount_Rate': 'USD', 'Price_Index': NAME,
        'Reference_Type': rule[0], 'Months_Lag': rule[1], 'Points': [
            {'Use': 'Yes', 'Descriptor': '{} {}Y'.format(NAME, years), 'Tenor': '{}Y'.format(years),
             'DealType': 'StructuredDeal', 'Quote_Type': 'Breakeven', 'Quoted_Market_Value': quote,
             'Deal': {}} for years, quote in zip(TENORS, quotes(rule))]}, **fields)}


def bootstrapped(market_prices, deals=(), factors=None, entries=None, **calculation):
    """A context over the document, bootstrapped."""
    context = loaded(document(deals, factors, market_prices, entries,
                              dict(VALUATION, **calculation)))
    context.bootstrap()
    return context


def curve(context):
    return context.current_cfg.params['Price Factors'][CURVE]['Curve'].array


@pytest.mark.parametrize('rule', [INTERPOLATED, SINGLE])
def test_the_bootstrap_recovers_the_inflation_curve_its_breakevens_came_from(rule):
    """THE ROUND TRIP: the curve back at its knots to 1e-10 relative, measured within 7.8e-15 off
    the interpolated three-month rule and 7.3e-15 off the single two-month one, in three Newton
    iterations each. The knots are the Junes by hand on ACT/365 from the last print, and the seed -
    each breakeven as a continuous rate - sits 9.6e-3 and 1.4e-3 off the answer, so a solver
    returning it fails.

    THE CLOSED FORM, `r(tau) = T log(1 + b) / tau` at a knot `tau` years after the last print, is
    the root where the base reference IS the last print and the final one reads one month, as under
    the single two-month rule here: measured within 8.0e-17. Under the interpolated rule the base
    reference is May's and June's prints, 0.187% under June's, and a final reference reads May off
    the curve below its knot, so the closed form misses the solve by up to 2.0e-4, at 1Y, held
    under 2.5e-4.

    KILLING MUTATIONS, by the relative miss: the solver returning its seed, 9.6e-3 and 1.4e-3; a
    knot at the first month a reference reads rather than its last, the interpolated 1Y knot at
    0.9151 against 1.0 and the curve 8.7e-3 off on the true knots; the block's rule unread, every
    swap referencing three months back interpolated, the single rule's curve 1.0e-2 off; the fixed
    leg paid at the base date, undiscounted, 2.17 and 2.00.
    """
    kind, years = rule[0], np.array(TENORS)
    seed = np.log1p(np.array(quotes(rule)) / 100.0)
    assert np.abs(seed / TRUE - 1.0).max() > 1e-3, 'the seed is already the answer'

    solved = curve(bootstrapped({BLOCK: block(rule)}))
    assert solved.dtype == np.float64
    assert np.array_equal(solved[:, 0], KNOTS), solved[:, 0]
    error = np.abs(solved[:, 1] / TRUE - 1.0).max()
    assert error < 1e-10, 'recovered to {:.3g}, not 1e-10'.format(error)

    closed = years * np.log1p(np.array(quotes(rule)) / 100.0) / KNOTS
    gap = np.abs(closed - solved[:, 1]).max()
    assert gap < (1e-15 if kind == 'Single' else 2.5e-4), gap


def test_a_curve_quoted_off_itself_solves_to_itself_in_no_iteration(caplog):
    """A CURVE QUOTED OFF ITSELF IS ITS OWN ROOT. The world's inflation curve standing in `Price
    Factors` beside its breakevens bootstraps back to itself to the bit in no Newton iteration,
    the curve written on the solve's knots tested before any step as the curve family tests its
    own; from the breakevens as continuous rates the same quotes solve to within 7.8e-15 of it in
    three.

    Killing mutations: the standing curve never tested - the curve 7.8e-15 off in three
    iterations; a standing curve passing the test given its step - 4.9e-15 off in one.
    """
    with caplog.at_level(logging.INFO):
        solved = curve(bootstrapped({BLOCK: block()}, factors={CURVE: inflation()}))
    assert [value.hex() for value in solved.ravel()] == [
        value.hex() for value in inflation()['Curve'].array.ravel()]
    assert rates_world.newton_steps(caplog.text) == [0], caplog.text


def test_the_linker_on_the_solved_curve_marks_as_on_the_curve_it_came_from():
    """The list module's linker, its coupons referencing November 2026 to June 2028, marked on the
    solved curve and on the curve its quotes came from: the knots coincide, so the two marks agree
    to one ulp, 39,829.272018665855 against ...85, not to the bit - the solved rates sit within
    1.6e-16 of the authored ones, the solve's own rounding - held to 1e-15.

    Killing mutation: the curve written alone, without the block's index and day count - the
    round trip still reads its curve, and the linker refuses, `KeyError: 'Day_Count'`.
    """
    on_true, _ = marks([LINKER], {CURVE: inflation()})
    solved = bootstrapped({BLOCK: block()}).current_cfg.params['Price Factors'][CURVE]
    on_solved, _ = marks([LINKER], {CURVE: solved})
    assert abs(on_solved['LINKER'] / on_true['LINKER'] - 1.0) < 1e-15, (on_solved, on_true)


def test_a_block_discounting_on_a_curve_another_family_solves_is_solved_after_it():
    """The book carries NO USD curve, and an `InterestRatePrices.USD` block quotes the world's on
    its own knots (`bootstrappers.knot_prices`); the entries list the inflation family first.
    `bootstrap_order` runs the curve family first because this one `reads` what it writes, so the
    swaps price on the curve it solved and the inflation curve is the world's to 1e-10, the USD
    curve recovered to 1e-10 beside it. A stale curve would not tell the orders apart: both legs
    of a swap pay on one day, so the inflation curve does not move with the discount curve.

    Killing mutation: `reads` emptied - the file's order stands and the inflation family prices
    its swaps on a discount curve nobody has written, `KeyError: 'InterestRate.USD'`.
    """
    name, curve_block = bootstrappers.knot_prices('USD', FACTORS, ModelParams(), B, {})
    context = bootstrapped({BLOCK: block(), name: curve_block}, factors={'InterestRate.USD': None},
                           entries=FAMILIES)
    solved = context.current_cfg.params['Price Factors']
    assert np.abs(solved['InterestRate.USD']['Curve'].array[:, 1] -
                  FACTORS['InterestRate.USD']['Curve'].array[:, 1]).max() < 1e-10
    assert np.abs(curve(context)[:, 1] / TRUE - 1.0).max() < 1e-10


def test_a_curve_move_re_solves_the_inflation_curve_solved_on_it_and_no_other():
    """The book carries the inflation curve its block's quotes solve, those quotes since moved a
    basis point and unsolved, beside two curve blocks quoting their own curves: USD, which the
    swaps discount on, and EUR, which they do not. A run narrowed to the EUR block leaves the
    inflation curve exactly as it stands; one narrowed to the USD block re-solves it on its moved
    quotes, to the bit the curve the whole document bootstraps (`bootstrappers.block_readers`
    naming the block's `Discount_Rate`, else its currency's curve, as the credit family's).

    Killing mutation: the inflation block read at its family's level, by every curve block - the
    EUR run re-solves it too, every knot moving by 9.8e-5 to 1.06e-4.
    """
    eur = rates_world.market('EUR', {'EUR': ((1.0, 2.0, 5.0), (0.021, 0.024, 0.026))}, 'EUR')
    curves = dict([bootstrappers.knot_prices('USD', FACTORS, ModelParams(), B, {}),
                   bootstrappers.knot_prices('EUR', eur, ModelParams(), B, {})])
    moved_block = block()
    for point in moved_block['instrument']['Points']:
        point['Quoted_Market_Value'] += 0.01
    standing = bootstrapped({BLOCK: block()}).current_cfg.params['Price Factors'][CURVE]
    market = dict(curves, **{BLOCK: moved_block})
    job = document(factors=dict(eur, **{CURVE: standing}), market_prices=market, entries=FAMILIES)

    whole = loaded(job)
    whole.bootstrap()
    for moved_curve, expected in (('InterestRatePrices.EUR', standing['Curve'].array),
                                  ('InterestRatePrices.USD', curve(whole))):
        narrowed = loaded(job)
        narrowed.bootstrap(only={moved_curve})
        assert np.array_equal(curve(narrowed), expected), (moved_curve, curve(narrowed) - expected)
    assert not np.array_equal(curve(whole), standing['Curve'].array), 'the moved quotes moved nothing'


def jacobian(config):
    """`dtheta/dq` the implicit function theorem gives, a row per knot, off the connected curve."""
    theta = config.calibrated_factors[FACTOR]
    leaves = config.quote_leaves[BLOCK][1]
    return np.array([torch.autograd.grad(theta[i], leaves, retain_graph=True)[0].cpu().numpy()
                     for i in range(theta.numel())])


def moved(index, step, rule=INTERPOLATED):
    """The curve re-bootstrapped with quote `index` moved by `step`."""
    market = {BLOCK: block(rule)}
    market[BLOCK]['instrument']['Points'][index]['Quoted_Market_Value'] += step
    return curve(bootstrapped(market))[:, 1]


@pytest.mark.parametrize('rule', [INTERPOLATED, SINGLE])
def test_the_quote_derivative_is_the_re_bootstrapped_difference(rule):
    """`Quote_Sensitivity` leaves the solved curve connected to its quotes through
    `utils.CalibrationSolve`, and `dtheta/dq` is the implicit function theorem: against central
    differences of full re-bootstraps a hundredth of a basis point either side, within 1e-8 of the
    largest entry, measured within 4.0e-11 and 6.3e-11. Lower triangular - a knot reads no later
    quote - diagonal under the single rule, where each knot is its own quote's closed form, and the
    switch writes the curve it writes off, bit for bit.

    KILLING MUTATIONS: the bumped compile authored a hundred units over the quote rather than one,
    every entry a hundred times the difference; the theorem's sign dropped, every entry negated.
    """
    context = bootstrapped({BLOCK: block(rule, Quote_Sensitivity='Yes')})
    assert np.array_equal(curve(context), curve(bootstrapped({BLOCK: block(rule)})))
    derivative = jacobian(context.current_cfg)
    assert not np.triu(derivative, 1).any(), 'a knot moves with a later quote'

    step = 1e-4
    reference = np.array([(moved(j, step, rule) - moved(j, -step, rule)) / (2 * step)
                          for j in range(len(TENORS))]).T
    error = np.abs(derivative - reference).max() / np.abs(reference).max()
    assert error < 1e-8, 'dtheta/dq is not the re-bootstrapped difference: {:.3g}'.format(error)


def test_a_linker_reports_its_quote_delta_beside_its_factor_greek():
    """The list module's linker on the bootstrapped curve under `Greeks: First`: its `dV/dq` lands
    on the block's quote leaf in the pass that reports its factor greek `dV/dtheta`, and the two
    are one chain rule on the 2Y quote, `dV/dq = sum_i dV/dtheta_i dtheta_i/dq` with `dtheta/dq`
    re-bootstrapped, and the difference of the linker re-valued on curves re-solved a hundredth of
    a basis point either side, both within 1e-8 (258.50 a percent, measured within 2.4e-11 and
    2.0e-11). Its references read nothing past June 2028, so the quotes past 2Y carry nothing and
    the 1Y carries 224.15.

    KILLING MUTATIONS: the implicit function theorem's sign dropped, -258.50; the bumped compile a
    hundred units over, 25,850.04; the calibrated curve harvested detached, no quote delta at all.
    """
    context = bootstrapped({BLOCK: block(Quote_Sensitivity='Yes')}, [LINKER], Greeks='First')
    _, out = context.run_job()
    greeks = out['Results']['Greeks_First'].xs(CURVE, level='Rate')['root']
    factor_greek = greeks.droplevel(['Tenor2', 'Tenor3']).reindex(KNOTS, fill_value=0.0).to_numpy()
    quote_delta = context.current_cfg.quote_leaves[BLOCK][1].grad.cpu().numpy()
    assert not quote_delta[2:].any(), quote_delta

    index, step = TENORS.index(2), 1e-4
    dtheta = (moved(index, step) - moved(index, -step)) / (2 * step)
    chained = factor_greek @ dtheta
    assert abs(quote_delta[index] / chained - 1.0) < 1e-8, (quote_delta[index], chained)

    revalued = []
    for sign in (1.0, -1.0):
        market = {BLOCK: block()}
        market[BLOCK]['instrument']['Points'][index]['Quoted_Market_Value'] += sign * step
        factors = {CURVE: bootstrapped(market).current_cfg.params['Price Factors'][CURVE]}
        revalued.append(marks([LINKER], factors)[0]['LINKER'])
    difference = (revalued[0] - revalued[1]) / (2 * step)
    assert abs(quote_delta[index] / difference - 1.0) < 1e-8, (quote_delta[index], difference)


def test_a_linkers_delta_to_a_swap_quote_is_both_curves_refitted():
    """The linker with its discount curve bootstrapped too, from a block quoting the world's USD
    curve on its own knots, both blocks under `Quote_Sensitivity`: the inflation residual reads the
    connected curve and `utils.CalibrationSolve` hands its cotangent upstream, as the credit
    family's does. Each swap quote's `dV/dq` is the linker revalued on both curves re-bootstrapped
    a thousandth of a percent either side, within 1e-6 of the largest: -215.81, -274.98 and -0.556
    at 1Y to 3Y and nothing at 5Y and 10Y, measured within 2.9e-11.

    The inflation curve's term is ZERO: both legs of a swap pay on one day, so the discount factor
    multiplies the residual and `dF/dD` is the residual itself, nil at the root - a zero-coupon
    swap prices the index ratio whatever it discounts on. Measured: the inflation curve re-solved
    on a swap quote moved a thousandth of a percent moves by 1.1e-16, its `dtheta/dq` to the swap
    quotes is 2.5e-18, so the deltas are the linker's own discounting and the inflation block's
    switch moves none of them by more than 5.7e-14, held to 1e-9 of the largest.

    KILLING MUTATIONS: the fixed leg paid a day after the index leg, the inflation curve then
    moving with the discount curve - the switch moves the 2Y delta by 0.75; with the connected
    curve not handed beside it, the deltas -215.79, -274.95 and -0.555 against the refit's
    -215.77, -275.20 and -1.31. The connected curve not handed alone leaves this green: its term
    is zero.
    """
    name, curve_block = bootstrappers.knot_prices('USD', FACTORS, ModelParams(), B, {})

    def market(sensitivity, index=None, step=0.0):
        curve_quotes = copy.deepcopy(curve_block)
        curve_quotes['instrument']['Quote_Sensitivity'] = 'Yes'
        if index is not None:
            curve_quotes['instrument']['Points'][index]['Quoted_Market_Value'] += step
        return {BLOCK: block(Quote_Sensitivity=sensitivity), name: curve_quotes}

    deltas = {}
    for sensitivity in ('Yes', 'No'):
        context = bootstrapped(market(sensitivity), [LINKER], entries=FAMILIES, Greeks='First')
        context.run_job()
        deltas[sensitivity] = context.current_cfg.quote_leaves[name][1].grad.cpu().numpy()
    assert np.abs(deltas['Yes'] - deltas['No']).max() < 1e-9 * np.abs(deltas['No']).max()

    step, difference = 1e-3, []
    for index in range(len(curve_block['instrument']['Points'])):
        revalued = []
        for sign in (1.0, -1.0):
            solved = bootstrapped(market('No', index, sign * step), entries=FAMILIES
                                  ).current_cfg.params['Price Factors']
            revalued.append(marks([LINKER], {key: solved[key] for key in (
                CURVE, 'InterestRate.USD')})[0]['LINKER'])
        difference.append((revalued[0] - revalued[1]) / (2 * step))
    error = np.abs(deltas['Yes'] - difference).max() / np.abs(difference).max()
    assert error < 1e-6, (deltas['Yes'], difference)


def test_a_quote_referencing_only_printed_months_is_refused_by_name():
    """A row whose own `Deal` carries its legs, a swap maturing a week after the base date: its
    final reference reads May's and June's prints, a stated level, so it reads nothing of the curve
    and its knot identifies nothing, and the family refuses naming the block, the row, the month it
    reads, the last print and the remedy. With that row's `Use` off the block solves and recovers
    its curve, which says the refusal reads the reference.

    Killing mutation: the refusal dropped - the solve dies on `linalg.solve`'s singular matrix,
    naming no quote.
    """
    market = {BLOCK: block()}
    market[BLOCK]['instrument']['Points'].insert(0, {
        'Use': 'Yes', 'Descriptor': 'INFL 1W', 'Tenor': '1W', 'DealType': 'StructuredDeal',
        'Quote_Type': 'Breakeven', 'Quoted_Market_Value': 2.0,
        'Deal': {'Children': legs(B + pd.Timedelta(days=7), INTERPOLATED, '1W')}})
    with pytest.raises(ValueError) as refusal:
        bootstrapped(market)
    message = str(refusal.value)
    assert message.startswith(BLOCK), message
    assert 'INFL 1W (reads 2026-06)' in message and 'printed by 2026-06-01' in message, message
    assert 'Use No' in message, message

    market[BLOCK]['instrument']['Points'][0]['Use'] = 'No'
    assert np.abs(curve(bootstrapped(market))[:, 1] / TRUE - 1.0).max() < 1e-10


def test_a_price_index_the_market_does_not_carry_is_refused_by_name():
    """A block naming an index the market carries no `PriceIndex` for is refused naming the block
    and the index, before a swap is authored: the knots stand on its last print.

    Killing mutation: the refusal dropped - `TypeError: 'NoneType' object is not subscriptable`,
    naming neither.
    """
    with pytest.raises(ValueError, match='{}: Price_Index CPI names no PriceIndex the market '
                                         'carries'.format(BLOCK)):
        bootstrapped({BLOCK: block(Price_Index='CPI')})


# =============================================================================================
# the emitter, off a scripted terminal
# =============================================================================================

AS_OF = B.date()
FRIDAY = (B - pd.Timedelta(days=3)).date().isoformat()   # the Friday before

#: When the terminal says June's print came out, and when July's comes out: the world's own next
#: publication, twelve days after the base date.
RELEASED, NEXT_RELEASE = '2026-07-15', '2026-08-15'


def packaged_inflation():
    """The shipped seed's inflation section: three indices, each with its swaps and conventions."""
    with open(os.path.join(ROOT, 'derivus_bloomberg', 'seed.json'), encoding='utf-8') as handle:
        return {'inflation': json.load(handle)['inflation']}


def printed():
    """What the swaps print, in percent: the world's breakevens to 10Y, invented beyond it, and
    nothing at 20Y."""
    return dict(zip(['{}Y'.format(years) for years in TENORS], quotes()),
                **{'15Y': 2.75, '30Y': 2.8})


class Walked(BloombergSession):
    """A session whose two event walks are canned rows, so the emitter is driven through the
    package's real readers with no socket and no patching."""

    def __init__(self, rows, history):
        super().__init__()
        self._api = self._session = self._service = object()
        self.rows, self.history = rows, history

    def _walk(self, securities, fields):
        for security in securities:
            if security in self.rows:
                yield security, None, {key: value for key, value in self.rows[security].items()
                                       if key in fields}

    def _walk_history(self, security, fields, start, end):
        for day, level in self.history[security]:
            if start.isoformat() <= day <= end.isoformat():
                yield day, None, {'PX_LAST': level}


def terminal(seed, breakevens=None, stamps={}, next_release=NEXT_RELEASE, unprinted=()):
    """`(verified map, session)`: the map discovery builds off canned NAME answers carrying each
    entry's `expect`, each index last updated on June's release; the terminal answering each swap
    in `breakevens` its print, a two-way and the Friday's date or the one `stamps` gives it, and
    each index its next release and the world's monthly prints but those dated `unprinted`, each
    at its month's end as the terminal dates an economic series."""
    candidates = list(discover.candidates_from_seed(seed))
    report = {candidate.security: {'ok': True, 'error': None, 'fields': {
        'NAME': '{} {}'.format(candidate.expect[0], candidate.path[-1]), 'PX_LAST': 1.0,
        'LAST_UPDATE_DT': RELEASED if candidate.path[2] == 'index' else FRIDAY}}
        for candidate in candidates}
    mapped = discover.build_map(seed, discover.verify(candidates, report, AS_OF),
                                AS_OF.isoformat())
    prints = trial_rates.FACTORS['PriceIndex.INFL']['Index'].array
    month_ends = [(utils.excel_offset + pd.Timedelta(days=day) + pd.offsets.MonthEnd(0)).date(
        ).isoformat() for day in prints[:, 0]]
    history = [(day, level) for day, level in zip(month_ends, prints[:, 1]) if day not in unprinted]
    rows, histories = {}, {}
    for candidate in candidates:
        quote = (printed() if breakevens is None else breakevens).get(candidate.path[-1])
        if candidate.path[2] == 'index':
            rows[candidate.security], histories[candidate.security] = {
                'ECO_RELEASE_DT': next_release}, history
        elif quote is not None:
            rows[candidate.security] = {
                'PX_LAST': quote, 'PX_BID': quote - 0.02, 'PX_ASK': quote + 0.02,
                'LAST_UPDATE_DT': stamps.get(candidate.security, FRIDAY)}
    return mapped, Walked(rows, histories)


def emitted(seed, name, as_of=AS_OF, **answers):
    """`(block name, block, factor name, factor, strip)` the emitter authors off the scripted
    terminal on `as_of`, the block and the factor in the wire form a job carries."""
    mapped, session = terminal(seed, **answers)
    strip, index = fetch_inflation_strip(session, mapped, seed, name, as_of)
    (block_name, authored), (factor_name, factor) = inflation_block(strip, index)
    return (block_name, json.loads(json.dumps(authored)), factor_name,
            json.loads(json.dumps(factor)), strip)


def test_the_terminal_authors_the_block_and_the_index_the_family_solves():
    """Three indices off one scripted terminal. USCPI's swaps print the world's breakevens to 10Y,
    two invented beyond it and nothing at 20Y, which is held out with `Use` No by name and carries
    no number, its verdict on the ledger; the seed's conventions ride beside the rows. The index
    is its prints from the first month the base date's reference reads to the last: the world's
    own May and June rows, in excel days from the month's start, under three months back
    interpolated, June alone under UKRPI's two months back single, June the last period and the
    terminal's next release the next publication. A 30Y printing 280, a decimal-shifted 2.8, is
    held out `off-market`. A swap printed eleven days before the base date, an index whose next
    release is the base date and a history missing May each refuse the trip by name, as does a
    seed entry declaring no `months_lag`.

    The family solves the block on the module's world reading the emitted index alone: the
    world's curve back at its six knots within 7.8e-15, held to 1e-10, the 15Y and 30Y knots
    beyond them, every used row's swap at par within 1.3e-9 on its million, held to 1e-6, and the
    list module's linker on the solved curve marking as on the world's, 39,829.27201866585 against
    ...855, one ulp, held to 1e-15.

    KILLING MUTATIONS: the breakeven read in decimal, every row a hundredth of its print and the
    solved 1Y rate -1.8e-3 against 2.1e-2, the curve 109% off and the linker 2.7% under; the
    prints authored newest first, which the factor's reading alone sees, the engine's curve sorting
    them; the excel day counted from the ordinal origin, the curve 9.7e-2 off and the linker 3.0e-4
    under; the conventions dropped from the block, the family refusing `KeyError: 'Price_Index'`;
    the index's publication allowance dropped at discovery, the index ledgered dead nineteen days
    after June's release and the trip refused for want of it; the freshness rule dropped, the late
    swap held out by the screen and the trip standing; the next-release rule dropped, the due
    print unnoticed; the month check dropped, June authored alone, the curve 9.7e-2 off and the
    linker 0.17% over; the band in basis points, the credit screen's, the shifted 30Y believed; a
    missing lag defaulted rather than refused.
    """
    seed = packaged_inflation()
    name, authored, index_name, index, strip = emitted(seed, 'USCPI')
    rows = authored['instrument']['Points']
    tenors = seed['inflation']['USCPI']['swaps']['tenors']
    assert (name, index_name) == ('InflationRatePrices.USCPI', 'PriceIndex.USCPI')
    assert [(row['Tenor'], row['Use'], row.get('Quoted_Market_Value')) for row in rows] == [
        (tenor, 'No' if tenor == '20Y' else 'Yes', printed().get(tenor)) for tenor in tenors]
    assert strip.rejected == {'USSWIT20 Curncy': 'invalid'}
    assert all(row['Deal'] == {} and row['DealType'] == 'StructuredDeal' for row in rows)
    assert all(row.get('Timestamp') == ({'.Timestamp': FRIDAY} if row['Use'] == 'Yes' else None)
               for row in rows)
    assert {key: value for key, value in authored['instrument'].items() if key != 'Points'} == {
        'Currency': 'USD', 'Discount_Rate': '', 'Day_Count': 'ACT_365', 'Price_Index': 'USCPI',
        'Months_Lag': 3, 'Reference_Type': 'Interpolated'}
    assert index == {
        'Index': {'.Curve': {'meta': [], 'data': trial_rates.FACTORS['PriceIndex.INFL'][
            'Index'].array[-2:].tolist()}},
        'Last_Period_Start': {'.Timestamp': '2026-06-01'},
        'Next_Publication_Date': {'.Timestamp': NEXT_RELEASE}, 'Publication_Period': 'Monthly',
        'Currency': 'USD'}
    for other, currency, lag, kind, months in (('UKRPI', 'GBP', 2, 'Single', 1),
                                               ('EUHICPX', 'EUR', 3, 'Single', 2)):
        _, block_of, _, index_of, _ = emitted(seed, other)
        assert [block_of['instrument'][key] for key in (
            'Currency', 'Months_Lag', 'Reference_Type', 'Price_Index')] == [
            currency, lag, kind, other]
        assert len(index_of['Index']['.Curve']['data']) == months, (other, index_of)
    shifted = emitted(seed, 'USCPI', breakevens=dict(printed(), **{'30Y': 280.0}))
    assert shifted[1]['instrument']['Points'][-1]['Use'] == 'No'
    assert shifted[4].rejected == {'USSWIT20 Curncy': 'invalid', 'USSWIT30 Curncy': 'off-market'}
    with pytest.raises(InvalidQuote, match='USSWIT5 Curncy last printed 2026-07-23'):
        emitted(seed, 'USCPI', stamps={'USSWIT5 Curncy': '2026-07-23'})
    with pytest.raises(InvalidQuote, match='CPURNSA Index next releases on 2026-08-03'):
        emitted(seed, 'USCPI', next_release=AS_OF.isoformat())
    with pytest.raises(IncompleteStrip, match='CPURNSA Index answered 2026-06 105.965'):
        emitted(seed, 'USCPI', unprinted=('2026-05-31',))
    unstated = packaged_inflation()
    del unstated['inflation']['UKRPI']['conventions']['months_lag']
    with pytest.raises(BloombergConfigurationError, match='UKRPI declares no months_lag'):
        emitted(unstated, 'UKRPI')

    solved_name, emitted_index = 'InflationRate.USCPI', {index_name: index}
    solved = bootstrapped({name: authored}, factors=emitted_index).current_cfg.params[
        'Price Factors'][solved_name]
    assert np.array_equal(solved['Curve'].array[:len(TENORS), 0], KNOTS)
    error = np.abs(solved['Curve'].array[:len(TENORS), 1] / TRUE - 1.0).max()
    assert error < 1e-10, 'recovered to {:.3g}, not 1e-10'.format(error)

    factors = dict(emitted_index, **{solved_name: solved})
    used = [(row['Tenor'], row['Quoted_Market_Value']) for row in rows if row['Use'] == 'Yes']
    valued, _ = marks([dict(leg, Index='USCPI') if 'Index' in leg else leg for tenor, _ in used
                       for leg in legs(B + utils.parse_period(tenor), INTERPOLATED, tenor)],
                      factors)
    par = [valued[tenor + ' index'] - valued[tenor + ' fixed'] * (1.0 + quote / 100.0) ** int(
        tenor[:-1]) for tenor, quote in used]
    assert max(abs(value) for value in par) < 1e-6, par

    linker = dict(LINKER, Index='USCPI')
    on_world, _ = marks([LINKER], {CURVE: inflation()})
    on_emitted, _ = marks([linker], factors)
    assert abs(on_emitted['LINKER'] / on_world['LINKER'] - 1.0) < 1e-15, (on_emitted, on_world)


def test_a_re_tick_is_a_values_patch_and_a_moved_tenor_a_new_plan():
    """The rows name no date and their `Deal` no number, so the same swaps printing other
    breakevens on a later day are a VALUES change: `schema.update_market_quote` updates the
    installed block in place, the held-out 20Y standing as it stood. A swap that stops answering
    is a knot fewer, `Use` being structure, which the guard refuses as a moved plan by name.

    Killing mutation: the breakeven authored into the row's `Deal` - every re-tick a new plan.
    """
    seed = packaged_inflation()
    name, first = emitted(seed, 'USCPI')[:2]
    job = {'Calc': {'MergeMarketData': {'ExplicitMarketData': {'Market Prices': {}}}}}
    assert schema.update_market_quote(job, name, first) == 'installed'

    tuesday = AS_OF + pd.Timedelta(days=1)
    _, ticked = emitted(seed, 'USCPI', tuesday, breakevens={
        tenor: quote + 0.05 for tenor, quote in printed().items()}, stamps={
        security: AS_OF.isoformat() for security in ('USSWIT1 Curncy', 'USSWIT10 Curncy')})[:2]
    assert ticked != first
    assert schema.update_market_quote(job, name, ticked) == 'updated'

    dark = {tenor: quote for tenor, quote in printed().items() if tenor != '7Y'}
    _, moved = emitted(seed, 'USCPI', breakevens=dark)[:2]
    with pytest.raises(ValueError, match='structure differs'):
        schema.update_market_quote(job, name, moved)
