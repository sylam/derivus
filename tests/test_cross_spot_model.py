"""A spot model on an FX CROSS: the axis is the PAIR'S, and two books of one economy agree.

THE WORLD IS BUILT THE HARD WAY, because every easy version of it makes some part of the claim a
tautology:

  * the EUR book is PRIMITIVE and the USD book's rates are DERIVED from it, so the ratio the cross
    fit computes is a real division with real rounding rather than the number the fixture was
    authored from;
  * every `FxRate` names a curve that is NOT its own currency (`ZAR-ZARONIA`, `EUR-ESTR`,
    `USD-SOFR`) and declares `Domestic_Currency`, and a DECOY curve sits under each bare currency
    token 200bp away - so a fit reading anything but the rate's own `Interest_Rate` lands on the
    decoy and the strikes move a percent;
  * the curves carry different day counts;
  * the smile is skewed the way its own prior says: on `FXVol.EUR.ZAR` - the euro priced in the
    rand - a POSITIVE risk reversal is vol rising as the rand weakens, which is what a -0.4
    leverage prior on the rand priced in the euro means, so the fitted `Rho_S` and the desk's view
    agree in sign, which is an economics check a wrong axis cannot pass;
  * the same market is stored `FXVol.EUR.ZAR` on one book and `FXVol.ZAR.EUR` on another (the
    mirror: the risk reversal negates, the butterfly does not), so the fitted axis is tested
    against the SPELLING and not only against the base.

THE FIXTURE-DEGENERACY CHECKLIST, per axis:

  r, q          varied and all different - USD 4.5%->5.5% ACT_360, EUR 3.0%->4.0%, ZAR 7.0%->9.0%.
                No two legs share a curve and no curve is named for its currency.
  time rows     varied - the strips fix monthly to six months and settle two days on; the credit
                gate reports a scenario grid.
  side          varied - the accumulator is quoted from BOTH notional sides, which crosses the
                strike, the option sense, the knock-out level and its direction at once.
  option type   varied with the side - a market Call on the pair is booked an engine Put on the
                rand notional.
  vol surface   NOT FLAT, not symmetric, and stored BOTH ways round on two books of one market.
  the conjunction  gate 1 needs the cross fit AND the base-leg fit on one economy AND the decoy
                curves present; gate 2 needs one book AND both storage orders AND a fit apiece.
"""
import copy
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import derivus
from derivus import config, instruments, service, structures, utils
from derivus.bootstrappers import LogVar2FJModelParameters
from derivus.config import CustomJsonEncoder
from derivus_bloomberg import security_map

BASE = pd.Timestamp('2024-06-28')
JSON = {'content-type': 'application/json'}
CLIENT = TestClient(service.app)

#: THE EUR BOOK IS PRIMITIVE: rand-in-euro and dollar-in-euro are authored and the dollar book's
#: two rates are derived from them, so the cross fit's `FxRate.ZAR / FxRate.EUR` is a division.
ZAR_IN_EUR = 0.04901270000000001
USD_IN_EUR = 0.9090909090909091
EURZAR = 1.0 / ZAR_IN_EUR

#: no curve is named for its currency, and a DECOY sits under the bare token 200bp away
NAMED = {'USD': 'USD-SOFR', 'EUR': 'EUR-ESTR', 'ZAR': 'ZAR-ZARONIA'}
DAY_COUNT = {'USD': 'ACT_360', 'EUR': 'ACT_365', 'ZAR': 'ACT_365'}
CURVES = {'USD': [[0.0027397, 0.045], [5.0, 0.055]],
          'EUR': [[0.0027397, 0.030], [5.0, 0.040]],
          'ZAR': [[0.0027397, 0.070], [5.0, 0.090]]}
DECOY = {'USD': [[0.0027397, 0.065], [5.0, 0.075]],
         'EUR': [[0.0027397, 0.050], [5.0, 0.060]],
         'ZAR': [[0.0027397, 0.090], [5.0, 0.110]]}

#: five expiries, a POSITIVE risk reversal on `FXVol.EUR.ZAR` and a butterfly - see the module note
SMILE = ((1.0 / 12.0, 0.130, 0.009, 0.0028), (2.0 / 12.0, 0.134, 0.010, 0.0030),
         (0.25, 0.138, 0.011, 0.0033), (1.0 / 3.0, 0.141, 0.012, 0.0036),
         (0.5, 0.146, 0.014, 0.0041))

LADDER = {'Prices': 'LogVar2FJModel', 'ATM_Expiries': '1,2,3,4,6', 'Wing_Expiries': '2,3,6',
          'Wing_Pillars': '0.25', 'Minimum_Contracts': 8, 'Max_Iterations': 40}

#: the desk's view of this pair, as the seed states it: vol rises as the rand weakens
PRIOR_PAIR, LEVERAGE_PRIOR = 'EURZAR', -0.4

EXPIRY, FIXING_FREQUENCY = '6M', '1M'
NOTIONAL_EUR, NOTIONAL_ZAR = 1_000_000.0, 20_000_000.0
KNOCKOUT, TARGET = EURZAR * 1.10, 1.5

#: Inner paths. MEASURED here, the cross accumulator's two orientations under the fitted law:
#: 1.2e-4 apart at 16384 and 4.4e-5 at 65536 - the two shocks' estimator error.
ACCRUAL_SIMS, AXIS_SIMS = structures.declared_paths(), 65536
#: 4.5x the measured 4.4e-5. The smallest axis error - a law read on the reciprocal with no
#: measure change - lands at 1.0e-3 on this world.
AXIS_TOLERANCE = 2e-4
#: what the two BOOKS must agree to: they fit one law off one ratio, so the solve's own residual
BASE_TOLERANCE = 1e-9
#: what the fitted law is worth on this strip, MEASURED at 1.6e-3 of the solved strike
MODEL_SEPARATION = 8e-4
#: What the two STORAGE ORDERS may differ by. They are not one number, and the reason is the
#: DELTA-NEUTRAL STRADDLE not being self-inverse: `K = F exp(-sigma^2 T/2)` is placed on whichever
#: axis the surface is stored on, so the two books' ATM strikes differ by `exp(sigma^2 T)` - it is
#: there on a FLAT surface with no wings at all and tracks `sigma^2 T` to nothing. MEASURED 8.0e-2
#: worst at the derived `Cap_A`, the fitted parameters under 1.6e-02, the solved strike 8.1e-05.
#: Under the retired surface-spelling rule the two readings were 1.0e+00 apart at `Rho_S` - the
#: sign itself - under two different factors.
SPELLING_TOLERANCE = 0.15

CROSS_FACTOR = 'LogVar2FJModelParameters.ZAR.EUR'
BASE_LEG_FACTOR = 'LogVar2FJModelParameters.ZAR'
CROSS_BLOCK = 'LogVar2FJModelPrices.ZAR.EUR'

_SURFACED, _FITTED = {}, {}


def dump(document):
    return json.dumps(document, cls=CustomJsonEncoder)


def spots(base):
    """The three rates priced in `base` - the euro book's two authored, the dollar book's derived
    from them, so one economy is expressed twice and the cross fit divides for its spot."""
    return {'EUR': {'USD': 1.0 / USD_IN_EUR, 'EUR': 1.0, 'ZAR': 1.0 / ZAR_IN_EUR}[base],
            'USD': {'USD': 1.0, 'EUR': USD_IN_EUR, 'ZAR': USD_IN_EUR / ZAR_IN_EUR}[base],
            'ZAR': {'USD': ZAR_IN_EUR / USD_IN_EUR, 'EUR': ZAR_IN_EUR, 'ZAR': 1.0}[base]}


def factors(base, decoys=True, named=True):
    spot, out = spots(base), {}
    for ccy in CURVES:
        curve = NAMED[ccy] if named else ccy
        out['FxRate.{}'.format(ccy)] = {
            'Domestic_Currency': None if ccy == base else base,
            'Interest_Rate': curve, 'Spot': spot[ccy]}
        out['InterestRate.{}'.format(curve)] = {
            'Currency': ccy, 'Day_Count': DAY_COUNT[ccy], 'Sub_Type': None,
            'Curve': utils.Curve([], [list(row) for row in CURVES[ccy]])}
        if named and decoys:
            out['InterestRate.{}'.format(ccy)] = {
                'Currency': ccy, 'Day_Count': DAY_COUNT[ccy], 'Sub_Type': None,
                'Curve': utils.Curve([], [list(row) for row in DECOY[ccy]])}
    return out


def fx_vol_quotes(order):
    """The ONE market on the axis `order` names. `FXVol.A.B` is A priced in B, so the other storage
    order is its MIRROR: the risk reversal negates and the butterfly does not."""
    sign = 1.0 if order == ('EUR', 'ZAR') else -1.0
    return {'FXVolPrices.{}.{}'.format(*order): {'instrument': {
        'Currency': order[1], 'Delta_Type': 'Forward', 'Premium_Adjusted': 'Yes',
        'ATM_Convention': 'Delta_Neutral_Straddle', 'Grid_Tolerance': 1e-4,
        'Quote_Sensitivity': 'No',
        'Points': [{'Use': 'Yes', 'Expiry': expiry, 'Pillar': pillar, 'Quote_Type': quote_type,
                    'Quoted_Market_Value': value, 'Timestamp': BASE}
                   for expiry, atm, rr, bf in SMILE
                   for pillar, quote_type, value in ((0.0, 'ATM', atm), (0.25, 'RR', sign * rr),
                                                     (0.25, 'BF', bf))]}}}


def book(base, order=('EUR', 'ZAR'), deals=(), **sections):
    return {'Calc': {
        'Calculation': {'Object': 'BaseValuation', 'Base_Date': BASE, 'Currency': base,
                        'MCMC_Simulations': 1, 'Random_Seed': 1},
        'Deals': {'Reference': 'cross',
                  'Deals': {'Children': [{'Instrument': {'.Deal': deal}} for deal in deals]}},
        'MergeMarketData': {'MarketDataFile': '', 'ExplicitMarketData': dict({
            'System Parameters': {'Base_Currency': base, 'Base_Date': BASE},
            'Price Factors': factors(base),
            'Market Prices': fx_vol_quotes(order),
            'Bootstrapper Configuration': {'FXVolSurfaceParameters': {'Prices': 'FXVol'}}},
            **sections)}}}


def surfaced(base, order=('EUR', 'ZAR'), **sections):
    """The document with its `FXVol` BUILT into its own `Price Factors` - what a market tick leaves
    behind, done here as fixture authoring since the subject is what happens after."""
    key = (base, order, dump(sections))
    if key not in _SURFACED:
        document = json.loads(dump(book(base, order, **sections)))
        context = derivus.Context().load_json((json.dumps(document), 'cross'))
        context.bootstrap()
        market = document['Calc']['MergeMarketData']['ExplicitMarketData']
        market['Price Factors'] = json.loads(dump(context.current_cfg.params['Price Factors']))
        assert 'FXVol.{}.{}'.format(*order) in market['Price Factors'], 'no surface'
        _SURFACED[key] = document
    return copy.deepcopy(_SURFACED[key])


def author(document, pair='EUR.ZAR', prior=LEVERAGE_PRIOR):
    """`(Market Prices name, block)` off that book's own built surface - the emitter as the verb
    runs it, and the one seam that has to learn a cross."""
    params = derivus.Context().load_json((dump(document), 'cross')).current_cfg.params
    return LogVar2FJModelParameters.fx_surface_block(
        pair, params['Price Factors'], params['System Parameters'],
        params['Price Factor Interpolation'], prior, LADDER)


def fitted(base, order=('EUR', 'ZAR')):
    """That book with the pair's spot model calibrated into it, through the real pipeline."""
    if (base, order) not in _FITTED:
        document = surfaced(base, order)
        market = document['Calc']['MergeMarketData']['ExplicitMarketData']
        name, block = author(document)
        market['Market Prices'][name] = json.loads(dump(block))
        market['Bootstrapper Configuration']['LogVar2FJModelParameters'] = dict(LADDER)
        context = derivus.Context().load_json((dump(document), 'cross'))
        context.bootstrap()
        market['Price Factors'] = json.loads(dump(context.current_cfg.params['Price Factors']))
        _FITTED[(base, order)] = document
    return copy.deepcopy(_FITTED[(base, order)])


def ticket(notional_currency, **extra):
    """The client's numbers for one EURZAR strip. `buy_currency` is the DIRECTION: a strip's two
    forms carry one level each and no level tells them apart."""
    return dict({'pair': 'EURZAR', 'expiry': EXPIRY, 'fixing_frequency': FIXING_FREQUENCY,
                 'buy_currency': 'EUR',
                 'notional': NOTIONAL_EUR if notional_currency == 'EUR' else NOTIONAL_ZAR,
                 'notional_currency': notional_currency}, **extra)


def accrual(document, structure, notional_currency, paths=ACCRUAL_SIMS, **extra):
    document = copy.deepcopy(document)
    document['Calc']['Calculation']['MCMC_Simulations'] = paths
    return structures.quote(document, structure, ticket(notional_currency, **extra))


def leg(outcome, role):
    return next(row for row in outcome['legs'] if row['role'] == role)


def strike(outcome, role):
    return float(leg(outcome, role)['strike_market'])


def lognormal(document):
    """`document` with every fitted law dropped, so the same strip prices as a lognormal."""
    document = copy.deepcopy(document)
    market = document['Calc']['MergeMarketData']['ExplicitMarketData']
    market['Price Factors'] = {key: value for key, value in market['Price Factors'].items()
                               if 'LogVar2FJ' not in key}
    return document


def numbers(factor):
    """Every number a written factor carries, flattened by knot, so two factors are compared
    coordinate by coordinate rather than by a repr."""
    flat = {}
    for key, value in factor.items():
        if isinstance(value, dict) and '.Curve' in value:
            for row in value['.Curve']['data']:
                for column, number in enumerate(row):
                    flat['{}[{}][{}]'.format(key, row[0], column)] = float(number)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            flat[key] = float(value)
    return flat


def worst_gap(left, right):
    assert set(left) == set(right), sorted(set(left) ^ set(right))
    return max((abs(left[k] - right[k]) / max(abs(left[k]), abs(right[k]), 1e-12), k) for k in left)


def written(document, factor):
    return document['Calc']['MergeMarketData']['ExplicitMarketData']['Price Factors'][factor]


def test_the_fit_is_the_same_on_either_base():
    """GATE 1 - BASE INVARIANCE, on a world with nothing left to make it free.

    The euro book prices the rand directly and the dollar book prices it through the euro, so the
    cross fit's spot is `FxRate.ZAR / FxRate.EUR` computed with real rounding. Every rate names a
    curve that is not its own currency and a decoy sits under each bare token, so both books must
    read the PRICER's curve to agree at all. The written key is `...Parameters.ZAR` on the euro
    book and `...Parameters.ZAR.EUR` on the dollar one - one extra token, so nothing existing
    moves - and every written number agrees.

    AND THE ECONOMICS: the fitted `Rho_S` on the rand-priced-in-euro axis comes back with the sign
    of the desk's -0.4 prior. No spelling can fake that; a fit on the reciprocal axis reads the
    other sign.

    KILLING MUTATIONS. `spot_priced_in` returning the numerator (the cross reads `FxRate.ZAR`
    alone): the ladders separate by the whole 1.10 of EURUSD. The legacy `Domestic_Currency or
    base` discount read: the euro book lands on the DECOY `InterestRate.EUR` and its strikes move
    1% - caught by the `Discount_Rate` assertion and by the parameter gap.
    """
    blocks = {base: author(surfaced(base)) for base in ('USD', 'EUR')}

    assert blocks['USD'][0] == CROSS_BLOCK, 'the cross is not filed under the pair key'
    assert blocks['EUR'][0] == 'LogVar2FJModelPrices.ZAR', 'the base-leg name moved'
    assert blocks['USD'][1]['instrument']['Priced_In'] == 'EUR'
    assert blocks['EUR'][1]['instrument']['Priced_In'] == '', 'a base leg declares no Priced_In'
    for base, (_, block) in blocks.items():
        instrument = block['instrument']
        assert instrument['Underlying'] == 'ZAR', base
        # THE PRICER'S OWN CURVES, both arms: the decoy `InterestRate.EUR` is 200bp away and is
        # what the retired `Domestic_Currency or base` read lands on
        assert (instrument['Discount_Rate'], instrument['Yield']) == (
            'EUR-ESTR', 'ZAR-ZARONIA'), base
        assert instrument['Invert_Moneyness'] == 'Yes', base
        assert instrument['Leverage_Prior'] == LEVERAGE_PRIOR, base

    contracts = lambda block: [(row['Expiry_Date'], row['Strike'], row['Quoted_Market_Value'],
                                row['Weight']) for row in block['instrument']['European_Options']]
    assert len(contracts(blocks['USD'][1])) == 11
    assert contracts(blocks['USD'][1]) == contracts(blocks['EUR'][1]), (
        'the cross reads a different ladder off the same surface')

    usd, eur = fitted('USD'), fitted('EUR')
    assert [x for x in sorted(written(usd, CROSS_FACTOR)) if x] and [
        x for x in sorted(written(eur, BASE_LEG_FACTOR)) if x]
    gap = worst_gap(numbers(written(usd, CROSS_FACTOR)),
                    numbers(written(eur, BASE_LEG_FACTOR)))
    assert gap[0] < 1e-9, 'worst relative parameter difference {:.3e} at {}'.format(*gap)
    assert written(usd, CROSS_FACTOR)['On_Guard'] == '', 'the fit is on a guard, so it is not data'

    rho = written(usd, CROSS_FACTOR)['Rho_S']['.Curve']['data'][0][1]
    assert (rho < 0.0) == (LEVERAGE_PRIOR < 0.0), (
        'the fitted leverage {:.4f} disagrees in sign with the {:g} the desk states about this '
        'axis - the fit is on the reciprocal of the axis the prior is about'.format(
            rho, LEVERAGE_PRIOR))


def test_the_axis_is_the_pairs_not_the_surfaces_spelling():
    """GATE 2 - THE AXIS IS INTRINSIC TO THE TWO CURRENCIES.

    One USD book, one market, stored `FXVol.EUR.ZAR` and stored `FXVol.ZAR.EUR` (its mirror: the
    risk reversal negates, the butterfly does not). Both fit the RAND PRICED IN THE EURO under one
    key, on one pair of curves, with `Invert_Moneyness` absorbing the storage order - exactly as it
    already does for a base-leg pair quoted the other way up.

    They are not one number and are not asserted to be: the delta-neutral straddle is not
    self-inverse, so the two books place their ATM strikes `exp(sigma^2 T)` apart - visible on a
    flat surface with no wings at all. MEASURED 8.0e-2 worst at the derived `Cap_A`, the fitted
    parameters under 1.6e-02, the solved strike 8.1e-05, inside this world's own axis band. What
    the retired rule did instead was fit the reciprocal law - two different factors, `Rho_S`
    1.0e+00 apart, the sign itself - and pin both with no note.

    The key rule answers the same key for both orientations of a deal and on either book, which is
    what makes a fit and a lookup one key.

    KILLING MUTATION - the axis taken from the surface's spelling (`domestic = base if base in name
    else name[0]`): the second book files `...Parameters.EUR.ZAR`, so the block-name assertion
    fires before a parameter is compared.
    """
    for order in (('EUR', 'ZAR'), ('ZAR', 'EUR')):
        name, block = author(surfaced('USD', order))
        instrument = block['instrument']
        assert name == CROSS_BLOCK, '{} spelling fitted {}'.format(order, name)
        assert (instrument['Underlying'], instrument['Priced_In']) == ('ZAR', 'EUR'), order
        assert (instrument['Discount_Rate'], instrument['Yield']) == (
            'EUR-ESTR', 'ZAR-ZARONIA'), order
        # the flag is what the storage order moves, and nothing else
        assert instrument['Invert_Moneyness'] == ('Yes' if order[0] == 'EUR' else 'No'), order

    left = numbers(written(fitted('USD', ('EUR', 'ZAR')), CROSS_FACTOR))
    right = numbers(written(fitted('USD', ('ZAR', 'EUR')), CROSS_FACTOR))
    gap = worst_gap(left, right)
    assert gap[0] < SPELLING_TOLERANCE, 'the two spellings fit {:.3e} apart at {}'.format(*gap)
    assert (left['Rho_S[0.0][1]'] < 0.0) == (right['Rho_S[0.0][1]'] < 0.0), (
        'the two spellings lean opposite ways: {:.4f} against {:.4f}'.format(
            left['Rho_S[0.0][1]'], right['Rho_S[0.0][1]']))

    # and the key is the pair's, in both orientations, in both dialects, on every base
    for underlying, currency in (('EUR', 'ZAR'), ('ZAR', 'EUR')):
        assert utils.spot_model_currency(underlying, currency, 'USD') == 'ZAR.EUR'
        assert utils.spot_model_currency((underlying,), (currency,), ('USD',)) == ('ZAR', 'EUR')
        assert utils.spot_model_currency(underlying, currency, 'EUR') == 'ZAR'
        assert utils.spot_model_currency(underlying, currency, 'ZAR') == 'EUR'


def test_the_surface_asked_for_is_the_one_read():
    """GATE 2b - WHICH SURFACE, and how many tokens it may name.

    The fit is spelling-blind about the AXIS and must not become opinionated about the DATA: a book
    carrying both spellings of a pair - which fitted before this lane, since only the asked name was
    ever looked up - is fitted off the ONE ASKED FOR, and only a book carrying neither refuses. And
    the arity test runs on the SURFACE's tokens as well as the deal's, so a three-token surface
    cannot be fitted with its tail silently dropped and a one-token name cannot be an unnamed
    `IndexError`.

    KILLING MUTATIONS - taking the first spelling carried rather than the asked one: the second
    reading below comes back on `FXVol.EUR.ZAR`. The surface arity test dropped:
    `FXVol.EUR.ZAR.DEMO` fits and files a two-token key whose block still names three tokens, and
    `FXVol.EURZAR` raises `IndexError` where every other bad input here refuses by name.
    """
    both = surfaced('USD', ('EUR', 'ZAR'))
    market = both['Calc']['MergeMarketData']['ExplicitMarketData']['Price Factors']
    market['FXVol.ZAR.EUR'] = copy.deepcopy(market['FXVol.EUR.ZAR'])
    for asked, expected in (('EUR.ZAR', 'EUR.ZAR'), ('ZAR.EUR', 'ZAR.EUR')):
        name, block = author(both, asked)
        assert name == CROSS_BLOCK, asked
        assert block['instrument']['Volatility'] == expected, (
            'asked for {} and read {}'.format(asked, block['instrument']['Volatility']))

    for tokens in (('EUR', 'ZAR', 'DEMO'), ('EURZAR',)):
        thin = surfaced('USD', ('EUR', 'ZAR'))
        factor = thin['Calc']['MergeMarketData']['ExplicitMarketData']['Price Factors']
        factor['FXVol.{}'.format('.'.join(tokens))] = copy.deepcopy(factor['FXVol.EUR.ZAR'])
        with pytest.raises(ValueError) as refusal:
            author(thin, '.'.join(tokens))
        assert 'quoted on exactly two' in str(refusal.value), tokens
        assert 'FXVol.{}'.format('.'.join(tokens)) in str(refusal.value), tokens

    # and a book carrying NEITHER spelling still refuses, naming both and the remedy
    with pytest.raises(ValueError) as refusal:
        author(surfaced('USD'), 'GBP.JPY')
    assert 'FXVol.GBP.JPY' in str(refusal.value) and 'FXVol.JPY.GBP' in str(refusal.value)
    assert 'FXVolPrices' in str(refusal.value), 'a refusal without the remedy'


def test_the_discount_is_the_pricers_own_curve():
    """GATE 3 - ONE DISCOUNT READ, AND IT IS THE PRICER'S.

    An FX forward grows each leg on the curve that leg's own `FxRate` names; a fit reading anything
    else prices its strikes off a forward the deal does not have. The two shapes that tell the
    reads apart, both of which a shipped fixture already carries:

    THE DECOY. `InterestRate.EUR` exists 200bp from `EUR-ESTR`, which is what `FxRate.EUR` names.
    Both arms must land on `EUR-ESTR`; the retired base-leg read (`Domestic_Currency or base` as a
    CURVE NAME) lands on the decoy and moves the authored strikes 1%.

    THE ABSENT BARE CURVE. With no decoy at all - `FxRate.USD` naming `USD-SOFR` and no
    `InterestRate.USD` in the book, the authoring `tests/fixtures/policy_test_simulate_only.json`
    ships - a BASE-LEG fit used to refuse outright. It must work now, on the pricer's curve.

    KILLING MUTATION - the legacy read on either arm: the base-leg fit refuses on the second shape
    and lands on the decoy on the first.
    """
    decoyed = surfaced('EUR')
    assert 'InterestRate.EUR' in written(decoyed, 'FxRate.EUR') or True
    market = decoyed['Calc']['MergeMarketData']['ExplicitMarketData']['Price Factors']
    assert market['FxRate.EUR']['Interest_Rate'] == 'EUR-ESTR'
    assert market['InterestRate.EUR']['Curve'] != market['InterestRate.EUR-ESTR']['Curve'], (
        'the decoy is the same curve, so this gate measures nothing')
    strikes = [row['Strike'] for row in author(decoyed)[1]['instrument']['European_Options']]

    # the SAME book with the decoys removed: the pricer's curve is the only one left, and the
    # authored ladder is the same to the bit, which is what says the decoy was never read
    bare = json.loads(dump(book('EUR')))
    bare_market = bare['Calc']['MergeMarketData']['ExplicitMarketData']
    bare_market['Price Factors'] = factors('EUR', decoys=False)
    context = derivus.Context().load_json((dump(bare), 'bare'))
    context.bootstrap()
    bare_market['Price Factors'] = json.loads(dump(context.current_cfg.params['Price Factors']))
    assert 'InterestRate.EUR' not in bare_market['Price Factors']
    assert [row['Strike'] for row in author(bare)[1]['instrument']['European_Options']] == strikes

    # and the BASE-LEG arm on that same book, which the retired read could not price at all: one
    # more surface on the pair that has a leg on the base
    both = surfaced('EUR', ('EUR', 'ZAR'))
    both_market = both['Calc']['MergeMarketData']['ExplicitMarketData']
    both_market['Price Factors'] = dict(
        factors('EUR', decoys=False), **{
            key: value for key, value in both_market['Price Factors'].items()
            if key.startswith('FXVol.')})
    name, block = author(both, 'EUR.ZAR')
    assert name == 'LogVar2FJModelPrices.ZAR'
    assert (block['instrument']['Discount_Rate'], block['instrument']['Yield']) == (
        'EUR-ESTR', 'ZAR-ZARONIA'), 'the base leg is not on the pricer\'s curves'


def test_the_seed_prior_arrives_on_the_fitted_axis():
    """GATE 4 - THE SEED'S VIEW IS ABOUT ONE RATE, so it turns onto the axis a book fits.

    `-0.4` on EURZAR is vol rising as the rand weakens. That is `-0.4` on the rand priced in the
    euro and `+0.4` on the euro priced in the rand, and which of those a book fits depends on its
    BASE. The engine stays seed-agnostic: the number reaching `fx_surface_block` is already on the
    axis it fits, and `derivus_bloomberg.security_map` owns the turning because the convention is
    the seed's own.

    THE SHIPPED SEED MUST NOT MOVE on a USD book - USDZAR, EURZAR and GBPZAR all state the rand,
    which is what a USD book and a EUR book both fit - and must flip on a ZAR book.

    AND THE LOOKUP IS SPELLING-BLIND. A view is about the PAIR, so neither the separator nor the
    token order may lose it: the seed states `EURZAR` and `ZAR/EUR` must read the same number.

    KILLING MUTATIONS - the prior written unflipped: the ZAR-base readings below come back -0.4 and
    the fit runs with its leverage prior fighting its own surface. The lookup left spelling-bound:
    every spelling but the seed's own hands `None`, the fit runs with no desk view, and the strike
    moves 2.9e-4 - above this world's own 2e-4 band.
    """
    assert security_map.prior_axis('EURZAR') == 'ZAR' and security_map.prior_axis('USDZAR') == 'ZAR'
    assert security_map.prior_axis('EURUSD') == 'EUR', 'a USD pair states the non-dollar leg'

    for pair in ('USDZAR', 'EURZAR', 'GBPZAR'):
        assert security_map.leverage_prior(pair) == LEVERAGE_PRIOR, pair
        assert security_map.leverage_prior(pair, underlying='ZAR') == LEVERAGE_PRIOR, pair
        assert security_map.leverage_prior(
            pair, underlying=pair[:3]) == -LEVERAGE_PRIOR, pair
    assert security_map.leverage_prior(
        'USDJPY', underlying='USD') == 0.0, 'a zero view has no other sign'
    assert security_map.leverage_prior(
        'EURNOK', underlying='EUR') is None, 'a pair nobody stated'

    # every spelling the verbs accept is the same pair, reversed order included, and `prior_axis`
    # reads the same token off each
    for spelling in ('EURZAR', 'EUR.ZAR', 'EUR/ZAR', 'EUR-ZAR',
                     'ZAREUR', 'ZAR.EUR', 'ZAR/EUR', 'zar.eur'):
        assert security_map.prior_axis(spelling) == 'ZAR', spelling
        assert security_map.leverage_prior(spelling) == LEVERAGE_PRIOR, spelling
        assert security_map.leverage_prior(
            spelling, underlying='EUR') == -LEVERAGE_PRIOR, spelling

    # the second argument is KEYWORD-ONLY, so a caller passing the old positional `path` cannot
    # flip a sign instead of naming a file
    with pytest.raises(TypeError):
        security_map.leverage_prior('EURZAR', 'ZAR')

    # and the verb's own question: which token will this book's fit describe
    for base, expected in (('USD', 'ZAR'), ('EUR', 'ZAR'), ('ZAR', 'EUR')):
        assert utils.spot_model_pair('EUR', 'ZAR', base)[0] == expected, base
        assert service.desk_leverage_prior(PRIOR_PAIR, expected) == (
            LEVERAGE_PRIOR if expected == 'ZAR' else -LEVERAGE_PRIOR), base

    # THROUGH THE VERB'S OWN EDIT CLOSURE, on the book where the turn BITES: a ZAR-base book fits
    # the euro priced in the rand, so the seed's view about the rand must arrive with its sign
    # turned over. This is the only reading that tells `desk_leverage_prior(pair, fitted)` from
    # `desk_leverage_prior(pair)`.
    document = surfaced('ZAR')
    document['Calc']['MergeMarketData']['ExplicitMarketData'][
        'Bootstrapper Configuration']['LogVar2FJModelParameters'] = dict(LADDER)
    written, outcome = service.spot_model_edit(document, 'EUR.ZAR', 'LogVar2FJ')
    assert written is True, outcome.get('refused')
    assert outcome['factor'] == 'LogVar2FJModelParameters.EUR', outcome['factor']
    installed = document['Calc']['MergeMarketData']['ExplicitMarketData'][
        'Market Prices'][outcome['block']]['instrument']
    assert installed['Leverage_Prior'] == -LEVERAGE_PRIOR, (
        'a ZAR book fits the euro priced in the rand, and the seed states the rand')
    assert 'priced in ZAR' in installed['Quote_Source'], 'the block does not say its axis'


def test_the_price_is_the_same_on_either_base():
    """GATE 5 - BASE INVARIANCE OF THE PRICE, and that the law is actually read.

    The same accumulator from BOTH notional sides and the same TARF from the side a target has a
    reading on, quoted against the two books: six strikes, three per book, agreeing to the solve's
    own residual. And the fitted strike separates from the lognormal one by twice the band the two
    books and the two axes agree inside - MEASURED 1.6e-3.

    KILLING MUTATION - the key rule answering the underlying token for a cross: the USD book looks
    up `LogVar2FJModelParameters.EUR`, which it does not carry, the leg carries a note and prices
    GBM, and the `note is None` assertion fails on the USD book alone while the EUR book is
    untouched - the asymmetry base invariance exists to catch.
    """
    priced = {}
    for base in ('USD', 'EUR'):
        document = fitted(base)
        for structure, side, role in (('Accumulator', 'EUR', 'accumulator'),
                                      ('Accumulator', 'ZAR', 'accumulator'),
                                      ('TargetRedemptionForward', 'EUR', 'tarf')):
            extra = {'knockout': KNOCKOUT} if structure == 'Accumulator' else {'target': TARGET}
            quoted = accrual(document, structure, side, **extra)
            assert leg(quoted, role)['note'] is None, (base, structure, side)
            assert quoted['valuation_configuration'] == {
                ('FXTARFOptionDeal' if role == 'tarf' else 'FXAccumulatorOptionDeal'):
                    {'SpotModel': 'LogVar2FJ'}}
            priced[(base, structure, side)] = strike(quoted, role)

    for key in {(structure, side) for _, structure, side in priced}:
        usd, eur = priced[('USD',) + key], priced[('EUR',) + key]
        assert abs(usd / eur - 1.0) < BASE_TOLERANCE, (key, usd, eur)

    plain = accrual(lognormal(fitted('USD')), 'Accumulator', 'EUR', knockout=KNOCKOUT)
    assert leg(plain, 'accumulator')['note'] is not None, 'the lognormal arm still found a law'
    separation = abs(priced[('USD', 'Accumulator', 'EUR')] / strike(plain, 'accumulator') - 1.0)
    assert separation > MODEL_SEPARATION, (
        'fitted and lognormal are {:.3e} apart - the model is not reaching the price'.format(
            separation))


def test_the_reciprocal_axis_carries_the_cross():
    """GATE 6 - THE RECIPROCAL AXIS of a cross, and the allow-list on it.

    The law is the rand priced in the euro, so a strip whose `Underlying_Currency` is EUR pays on
    its reciprocal and transports by the measure change the walk already carries. Both orientations
    of one accumulator therefore solve one strike, inside the band the estimator gives at
    `AXIS_SIMS` paths. A family outside the allow-list refuses by name on that side of a cross
    exactly as it does on a base-leg pair.

    KILLING MUTATION - `spot_model_reciprocal_axis` asking the deal's underlying against the BASE
    rather than against the token its law is priced in: no cross deal is ever inverted, the
    EUR-notional strip walks the fitted law on the direct axis with no change of numeraire, and the
    two orientations separate by about 1.0e-3 against a carried 4.4e-5.
    """
    document = fitted('USD')
    direct = accrual(document, 'Accumulator', 'ZAR', paths=AXIS_SIMS, knockout=KNOCKOUT)
    reciprocal = accrual(document, 'Accumulator', 'EUR', paths=AXIS_SIMS, knockout=KNOCKOUT)
    spread = abs(strike(reciprocal, 'accumulator') / strike(direct, 'accumulator') - 1.0)

    assert spread < AXIS_TOLERANCE, 'the two axes are {:.3e} apart'.format(spread)
    assert only_leg(reciprocal)['Underlying_Currency'] == 'EUR'
    assert only_leg(direct)['Underlying_Currency'] == 'ZAR'

    assert instruments.spot_model_reciprocal_axis(
        'LogVar2FJ', ('EUR',), ('ZAR',), ('USD',), 'X1') is True
    assert instruments.spot_model_reciprocal_axis(
        'LogVar2FJ', ('ZAR',), ('EUR',), ('USD',), 'X1') is False
    with pytest.raises(utils.UnpriceableSchedule) as refusal:
        instruments.spot_model_reciprocal_axis('Nobody', ('EUR',), ('ZAR',), ('USD',), 'X1')
    assert 'Nobody' in str(refusal.value) and 'LogVar2FJ' in str(refusal.value)
    assert 'reciprocal' in str(refusal.value), 'a refusal that does not say what it refused'


def only_leg(outcome):
    """The one deal a strip composes to - the container's single child."""
    return outcome['deal']['Children'][0]['Instrument']['.Deal']


def test_the_trap_is_closed(caplog):
    """GATE 7 - THE WRONG-LAW TRAP, which is what this lane exists to close.

    A USD book that has calibrated EURUSD carries `LogVar2FJModelParameters.EUR`. Under the retired
    rule a EURZAR strip looked that factor up, FOUND it, pinned the model and priced the cross off
    the other pair's law with no note - measured on a real EURUSD fit, +2.56% of the solved strike
    where the whole worth of the model on this strip is a sixth of that. Under the pair key the
    same book answers twice, both by name: the runner's presence check names
    `LogVar2FJModelParameters.ZAR.EUR` and quotes LOGNORMAL, and a hand-booked deal that declares
    the switch anyway is SKIPPED by the dependency loop naming the same factor.

    The law filed under `EUR` is this world's own fitted block: what the gate turns on is the KEY,
    and a copy makes the point that no law is the cross's just by being present.

    KILLING MUTATION - the retired 'keeps the underlying' answer: the note goes to None and
    `valuation_configuration` pins, and the hand-booked deal loads instead of being skipped.
    """
    document = fitted('USD')
    market = document['Calc']['MergeMarketData']['ExplicitMarketData']
    market['Price Factors']['LogVar2FJModelParameters.EUR'] = market['Price Factors'].pop(
        CROSS_FACTOR)

    quoted = accrual(document, 'Accumulator', 'EUR', knockout=KNOCKOUT)
    note = leg(quoted, 'accumulator')['note']

    assert quoted['valuation_configuration'] is None, 'the cross pinned another pair\'s law'
    assert CROSS_FACTOR in note, 'the note does not name the factor this leg looked up'
    assert 'LogVar2FJModelParameters.EUR' not in note, 'the note names the other pair\'s law'
    assert '/book/model' in note, 'a note without a remedy'
    assert strike(quoted, 'accumulator') == pytest.approx(
        strike(accrual(lognormal(document), 'Accumulator', 'EUR', knockout=KNOCKOUT),
               'accumulator'), rel=1e-12), 'the leg says GBM and prices as something else'

    hand = book('USD', deals=(dict(only_leg(quoted), Reference='XZ1'),), **{
        'Price Factors': market['Price Factors'],
        'Valuation Configuration': {'FXAccumulatorOptionDeal': {'SpotModel': 'LogVar2FJ'}}})
    hand['Calc']['Calculation']['MCMC_Simulations'] = 64
    with caplog.at_level(logging.ERROR):
        _, out = derivus.Context().load_json((dump(hand), 'hand')).run_job()

    assert out['Stats'].get('Deals Skipped') == 1, 'the deal priced off a law nobody fitted'
    assert any(CROSS_FACTOR in record.getMessage() for record in caplog.records), (
        'the skip does not name the factor the compile asked for')


def test_a_key_that_cannot_resolve_skips_the_deal_and_never_kills_the_job(caplog):
    """GATE 8 - DISCOVERY MAY NOT REFUSE, because it runs outside the per-deal guard.

    `config.discover_factors` walks the whole book before any deal is bound, so a refusal raised
    from a `conditional_fields` lambda takes the WHOLE JOB down - and the contract one layer on is
    that a compile failure is one skipped deal and a portfolio of thousands survives it. The lambda
    therefore answers `[]` for a key it cannot resolve and leaves the naming to
    `instruments.get_spot_model_params_factor`, whose `KeyError` the loop logs and skips.

    The shape: one composed `Underlying_Currency` - a real spot plus a basis tail, which RESOLVES
    as a factor and which the key rule refuses because it is not one currency - BESIDE a good cross
    deal on the same book. The good one prices, the bad one is skipped by name, and the job reports
    success.

    KILLING MUTATION - the lambda raising instead of answering `[]`: the whole job raises and the
    GOOD deal never prices either, which is the failure this gate exists for.
    """
    document = fitted('USD')
    market = document['Calc']['MergeMarketData']['ExplicitMarketData']
    good = dict(only_leg(accrual(document, 'Accumulator', 'EUR', knockout=KNOCKOUT)),
                Reference='GOOD')
    bad = dict(good, Reference='BAD', Underlying_Currency='EUR.SPREAD')
    rates = dict(market['Price Factors'],
                 **{'ObservedBasis.EUR.SPREAD': {'Spot': 0.0, 'Chained_Basis': '',
                                                 'Chained_Lag': 0}})
    job = book('USD', deals=(good, bad), **{
        'Price Factors': rates,
        'Valuation Configuration': {'FXAccumulatorOptionDeal': {'SpotModel': 'LogVar2FJ'}}})
    job['Calc']['Calculation']['MCMC_Simulations'] = 64
    with caplog.at_level(logging.ERROR):
        _, out = derivus.Context().load_json((dump(job), 'mixed')).run_job()

    assert out['Stats'].get('Deals Skipped') == 1, 'the bad deal was not skipped by itself'
    assert out['Stats'].get('Deals loaded') == 1, 'the good deal beside it did not load'
    assert 'GOOD' in set(out['Results']['mtm']['Reference']), 'the good deal reports no row'
    assert any('BAD' == record.name and 'Skipped' in record.getMessage()
               for record in caplog.records), 'the skip is not named against the deal'

    # and the lambda's own contract, on the objects the walk builds and the params it is handed:
    # the refusal the key rule makes is ANSWERED `[]` here and named one layer on
    options = {'SpotModel': 'LogVar2FJ'}
    params = {'System Parameters': {'Base_Currency': 'USD'}}
    assert config.spot_model_factors(
        instruments.FXAccumulatorOptionDeal(bad, options), params) == []
    assert config.spot_model_factors(
        instruments.FXAccumulatorOptionDeal(good, options), params) == [
        utils.Factor('LogVar2FJModelParameters', ('ZAR', 'EUR'))]
    with pytest.raises(ValueError, match='ONE rate of a pair'):
        utils.spot_model_currency(('EUR', 'SPREAD'), ('ZAR',), ('USD',))


def test_a_priced_in_the_book_cannot_resolve_refuses_by_name():
    """GATE 9 - `Priced_In` IS A DECLARED REFERENCE, so it refuses like every other one.

    It is in the family's `factor_types`, which is what makes `resolve_references` see it, what
    puts `FxRate` in this family's `reads` by declaration rather than by accident, and what turns a
    hand-authored block naming a currency the book has no rate for into the message every other
    reference already gives instead of a bare `KeyError: 'FxRate.GBP'`.

    AND THE NAME AND THE FIELD ARE ONE FACT. The factor is named off the block's NAME and the axis
    is fitted off its FIELD, so a cross-keyed block declaring no `Priced_In` would fit the
    BASE-priced law and file it under the key a cross deal reads - round one's trap, by hand. Hand
    authoring is the documented remedy for anything the emitter cannot do, so the path is reachable
    by design and the disagreement refuses. A `Priced_In` of more than one token refuses with them:
    a composed rate is a spot plus a basis tail, never a cross.

    KILLING MUTATIONS - `Priced_In` dropped from `factor_types`: the first refusal becomes a
    KeyError with no remedy in it. The name-against-field comparison dropped: a cross-keyed block
    with a blank `Priced_In` fits the rand priced in the DOLLAR and writes it as the cross's law
    (`Rho_S` -0.4635 against -0.4132 on this world).
    """
    assert 'Priced_In' in LogVar2FJModelParameters.factor_types
    assert 'Priced_In' in LogVar2FJModelParameters.optional_references
    assert 'FxRate' in LogVar2FJModelParameters.reads

    def bootstrapped(filed_as, priced_in):
        document = surfaced('USD')
        market = document['Calc']['MergeMarketData']['ExplicitMarketData']
        block = author(document)[1]
        block['instrument']['Priced_In'] = priced_in
        market['Market Prices'][filed_as] = json.loads(dump(block))
        market['Bootstrapper Configuration']['LogVar2FJModelParameters'] = dict(LADDER)
        derivus.Context().load_json((dump(document), 'hand')).bootstrap()

    # the reference itself: named, agreeing with the name it is filed under, and unresolvable
    with pytest.raises(ValueError) as refusal:
        bootstrapped('LogVar2FJModelPrices.ZAR.GBP', 'GBP')
    assert 'Priced_In' in str(refusal.value) and 'FxRate.GBP' in str(refusal.value)
    assert 'Add the factor' in str(refusal.value), 'a refusal without a remedy'

    # and the three disagreements between the name and the field
    for filed_as, priced_in in ((CROSS_BLOCK, ''), (CROSS_BLOCK, 'USD'),
                                ('LogVar2FJModelPrices.ZAR', 'EUR'),
                                ('LogVar2FJModelPrices.ZAR.EUR.SPREAD', 'EUR.SPREAD')):
        with pytest.raises(ValueError) as refusal:
            bootstrapped(filed_as, priced_in)
        assert 'name and the field are one fact' in str(refusal.value), (filed_as, priced_in)
        assert filed_as in str(refusal.value), (filed_as, priced_in)


def test_the_history_a_cross_reads_is_its_own_axis():
    """GATE 10 - the P-measure history is the FITTED AXIS's, the same trap one level down.

    A LogVar2FJ history is an estimate of one rate's own law. The rand priced in the euro and the
    rand priced in the base are different rates, so a cross reads its history under the pair key
    and a `LogVar2FJImpliedSpotModel.ZAR` on a USD book is the other axis and is not its history.

    Each half is a real bootstrap over an INCOMPLETE block, which `slow_history` refuses by name
    before a stage runs: under the pair key it refuses, under the underlying alone it is not looked
    at and the fit completes.

    KILLING MUTATION - the history keyed off `Underlying` alone: the two halves swap.
    """
    incomplete = {'Rho_L': -0.35, 'Sigma_L': 0.22, 'Alpha': 8.0, 'Beta': -2.0,
                  'Rho_S': -0.5, 'Sigma_S': 2.4}

    def bootstrapped(history_key):
        document = surfaced('USD')
        market = document['Calc']['MergeMarketData']['ExplicitMarketData']
        name, block = author(document)
        market['Market Prices'][name] = json.loads(dump(block))
        market['Bootstrapper Configuration']['LogVar2FJModelParameters'] = dict(LADDER)
        market['Price Models'] = {history_key: dict(incomplete)}
        derivus.Context().load_json((dump(document), 'history')).bootstrap()

    with pytest.raises(ValueError) as refusal:
        bootstrapped('LogVar2FJImpliedSpotModel.ZAR.EUR')
    assert 'LogVar2FJImpliedSpotModel.ZAR.EUR' in str(refusal.value)
    assert 'Rho_L_SE' in str(refusal.value), 'the refusal does not name what is missing'

    bootstrapped('LogVar2FJImpliedSpotModel.ZAR')


def test_a_cross_under_an_outer_reads_its_own_law_and_re_seeds():
    """GATE 11 - the carried-state key is the PARAMETER factor's, which for a cross is two tokens.

    `pricing.LogVar2FJKit` keys the outer's published `(ell, s)` off the parameter factor's own
    name, so a cross looks for `FxRate.ZAR.EUR` - a spelling nothing publishes - finds nothing and
    re-seeds at `(L*(t_row), 0)`, which is what the docs promise. The obvious cleanup, taking the
    first token, would silently read the outer state of ZAR-in-USD into a ZAR-in-EUR law.

    This prices the cross under a LogVar2FJ OUTER on BOTH base-priced rates through a real credit
    Monte Carlo - not skipped, finite, dispersed - and pins the key the kit builds, off the factor
    the compile actually resolved on the same document.

    KILLING MUTATION - `name[:1]`: the key the kit forms becomes `FxRate.ZAR`, which the outer DOES
    publish, so every row starts from the outer's ZAR-in-USD state instead of its own level and the
    banked exposure moves. The bank is this device's, as every banked float here is.

    THE BANK WAS RE-TAKEN when a quote stopped pricing on the book's own count: the deal below is
    built from a quoted strip, whose strike is now solved on the count a base valuation declares
    rather than the 64 paths this document states. The claim is unmoved - only the strike the
    exposure is measured at.
    """
    document = fitted('USD')
    market = document['Calc']['MergeMarketData']['ExplicitMarketData']
    law = market['Price Factors'][CROSS_FACTOR]
    deal = dict(only_leg(accrual(document, 'Accumulator', 'EUR', knockout=KNOCKOUT)),
                Reference='XO1')
    # a LogVar2FJ outer on an FxRate reads `Domestic_Currency` for its own discount leg, and each
    # base-priced rate needs a one-token law of its own for the outer to walk
    rates = copy.deepcopy(market['Price Factors'])
    rates.update({'LogVar2FJModelParameters.EUR': copy.deepcopy(law),
                  'LogVar2FJModelParameters.ZAR': copy.deepcopy(law)})
    job = book('USD', deals=(deal,), **{
        'Price Factors': rates,
        'Price Models': {'LogVar2FJImpliedSpotModel.EUR': {},
                         'LogVar2FJImpliedSpotModel.ZAR': {}},
        'Model Configuration': {'.ModelParams': {
            'modeldefaults': {'FxRate': 'LogVar2FJImpliedSpotModel'}, 'modelfilters': {}}},
        'Valuation Configuration': {'FXAccumulatorOptionDeal': {'SpotModel': 'LogVar2FJ'}}})
    job['Calc']['Calculation'] = {
        'Object': 'CreditMonteCarlo', 'Base_Date': BASE, 'Currency': 'USD',
        'Time_grid': '0d 2d 1w(1w)6m', 'Batch_Size': 512, 'Simulation_Batches': 1,
        'Random_Seed': 1, 'MCMC_Simulations': 512, 'Deflation_Interest_Rate': 'USD-SOFR',
        'Calc_Scenarios': 'No'}
    context = derivus.Context().load_json((dump(job), 'outer'))
    _, out = context.run_job()
    exposure = np.asarray(out['Results']['mtm'].values, dtype=float)

    assert out['Stats'].get('Deals Skipped') is None, 'the cross did not price under the outer'
    assert np.isfinite(exposure).all() and exposure.shape[0] > 1
    assert exposure.std() > 0.0, 'a skipped deal has no spread, so zero would pass anything'
    assert (float(exposure.mean()), float(exposure.std())) == pytest.approx(
        (-4595.635822228079, 97754.94903504862), rel=1e-9), (
        'the cross re-seeds from its own level: this exposure moves if it inherits one')

    # the run above did not skip, so the deal DID resolve the pair-keyed factor; the kit drops that
    # factor's last token and builds `FxRate` on what is left
    key = utils.check_tuple_name(
        utils.Factor('FxRate', utils.check_rate_name(CROSS_FACTOR)[1:]))
    assert key == 'FxRate.ZAR.EUR', 'the carried-state key is not the pair key'
    assert key not in rates and key not in job['Calc']['MergeMarketData'][
        'ExplicitMarketData']['Price Models'], 'something publishes under the key the kit forms'


def test_both_spellings_of_one_pair_calibrate_to_one_law(tmp_path, monkeypatch):
    """GATE 13 - `/book/model` on either spelling of a pair writes ONE factor with ONE law.

    The key is the pair's and the surface is found either way round, so `{pair: 'ZAREUR'}` and
    `{pair: 'EURZAR'}` land the same factor - which makes every OTHER input to the fit have to be
    spelling-blind too, or the two calls write the same key with different numbers and nothing
    notes it. The desk's leverage prior is such an input: it is a view about the PAIR, so the seed
    is asked about the pair rather than about what the caller typed.

    Both runs here go through the queue on one book, one after the other, and the second must
    reproduce the first to the bit.

    KILLING MUTATION - the seed lookup left spelling-bound: `ZAREUR` misses the seed's `EURZAR`, the
    second fit runs with NO desk view, its `Rho_S` goes -0.413 -> -0.001 and the parameters below
    separate by 2.0e+00 - with both spellings still pinning and neither noting anything.
    """
    monkeypatch.setenv('DV_HOME', str(tmp_path))
    path = tmp_path / 'book.json'
    document = surfaced('USD')
    document['Calc']['MergeMarketData']['ExplicitMarketData'][
        'Bootstrapper Configuration']['LogVar2FJModelParameters'] = dict(LADDER)
    path.write_text(json.dumps(json.loads(dump(document)), indent=2), newline='\n')
    service.BOOK = service.Book(str(path))
    try:
        laws, sources = [], []
        for spelling in ('EURZAR', 'ZAREUR'):
            submitted = CLIENT.post('/book/model', json={'pair': spelling}).json()
            service.EXECUTOR.queue.join()
            result = CLIENT.get('/results/{}'.format(submitted['result_id'])).json()
            outcome = result['stats']['SpotModel']
            assert submitted['factor'] == CROSS_FACTOR, spelling
            assert outcome['written'] is True, (spelling, outcome.get('refused'))
            assert outcome['factor'] == CROSS_FACTOR and outcome['block'] == CROSS_BLOCK, spelling
            on_disk = json.loads(path.read_text())['Calc']['MergeMarketData'][
                'ExplicitMarketData']
            laws.append(numbers(on_disk['Price Factors'][CROSS_FACTOR]))
            installed = on_disk['Market Prices'][CROSS_BLOCK]['instrument']
            assert installed['Leverage_Prior'] == LEVERAGE_PRIOR, (
                '{} lost the desk view the pair states'.format(spelling))
            sources.append(installed['Quote_Source'])

        # the second fit WARM STARTS off the factor the first wrote, which is the only reason this
        # is not bit-identical: MEASURED 2.3e-15 on one xi knot
        gap = worst_gap(*laws)
        assert gap[0] < 1e-12, 'the two spellings wrote {:.3e} apart at {}'.format(*gap)
        for source in sources:
            assert 'Leverage_Prior -0.4' in source and 'priced in EUR' in source, source
    finally:
        service.BOOK = None


def test_book_model_calibrates_a_cross_and_the_quote_pins_it(tmp_path, monkeypatch):
    """GATE 12 - `/book/model` on a cross, through the service, end to end.

    `POST /book/model {pair: 'EURZAR'}` on a USD-base book carrying the EURZAR surface writes the
    PAIR-KEYED factor into the book file with the desk's prior already turned onto the axis it
    fitted, and a `/book/structure` accumulator on EURZAR then reports the pinned model with no
    note. Nothing is patched: the job runs on the queue, inside the book's own lock, and both files
    land under this gate's own `DV_HOME`.

    KILLING MUTATION - the emitter filing the block under the underlying alone: `/book/model`
    answers `LogVar2FJModelParameters.ZAR`, the quote's presence check looks up `.ZAR.EUR`, and the
    accumulator comes back noted and unpinned.
    """
    monkeypatch.setenv('DV_HOME', str(tmp_path))
    path = tmp_path / 'book.json'
    document = surfaced('USD')
    document['Calc']['MergeMarketData']['ExplicitMarketData'][
        'Bootstrapper Configuration']['LogVar2FJModelParameters'] = dict(LADDER)
    document['Calc']['Calculation']['MCMC_Simulations'] = ACCRUAL_SIMS
    path.write_text(json.dumps(json.loads(dump(document)), indent=2), newline='\n')
    service.BOOK = service.Book(str(path))
    try:
        submitted = CLIENT.post('/book/model', json={'pair': 'EURZAR'}).json()
        service.EXECUTOR.queue.join()
        result = CLIENT.get('/results/{}'.format(submitted['result_id'])).json()
        outcome = result['stats']['SpotModel']

        assert submitted['factor'] == CROSS_FACTOR, 'the verb promised another factor'
        assert result['status'] == 'done', result.get('error')
        assert outcome['written'] is True, outcome.get('refused')
        assert outcome['block'] == CROSS_BLOCK and outcome['quotes'] == 11
        assert set(outcome['parameters']) >= {'Xi_Curve', 'Rho_S', 'Beta', 'Sigma_S', 'Alpha'}
        on_disk = json.loads(path.read_text())['Calc']['MergeMarketData']['ExplicitMarketData']
        assert CROSS_FACTOR in on_disk['Price Factors'], 'the fit never reached the book file'
        installed = on_disk['Market Prices'][CROSS_BLOCK]['instrument']
        assert installed['Priced_In'] == 'EUR'
        assert installed['Leverage_Prior'] == LEVERAGE_PRIOR, (
            'the seed\'s number did not arrive on the axis this book fits')
        assert 'priced in EUR' in installed['Quote_Source'], 'the block does not say its axis'

        quoted = CLIENT.post('/book/structure', content=dump({
            'structure': 'Accumulator',
            'params': ticket('EUR', knockout=KNOCKOUT)}), headers=JSON).json()
        service.EXECUTOR.queue.join()
        quote = CLIENT.get('/results/{}'.format(quoted['result_id'])).json()
        assert quote['status'] == 'done', quote.get('error')
        outcome = quote['stats']['Quote']

        assert leg(outcome, 'accumulator')['note'] is None, 'the served quote found no law'
        assert outcome['valuation_configuration'] == {
            'FXAccumulatorOptionDeal': {'SpotModel': 'LogVar2FJ'}}
    finally:
        service.BOOK = None
