"""A spot model holds the forward, and a quote is priced on paths that can read it.

THE WORLD is `logvar2fj_world.json`'s shapes carrying a factor AS VIOLENT AS A FITTED RAND ONE -
Alpha 58, Beta -19, Sigma_S 1.9, Rho_S -0.42, a NEGATIVE Cap_A - because every gentle set of
parameters makes both claims below tautologies: a residual near Gaussian is compensated by a term
that is nearly the variance, and a law with a per-path spread of a few tenths of a percent reads
the forward off any number of paths at all.

  the law    E[S_T]/F_T is one on BOTH axes - the fitted axis and the reciprocal a deal on the
             other notional pays on - through the walk's own blocks and the residual on their
             clocks, which is the arithmetic `pricing.LogVar2FJKit` hands the pricers.
  the quote  the two notional sides of ONE zero-cost forward strip solve one strike. A leverage-1
             accumulator whose knock-out no path reaches IS a strip of forwards, so its solved
             strike is the discount-weighted average forward from either side.

THE FIXTURE-DEGENERACY CHECKLIST:

  r, q          varied and different - USD 4.5%->5.5%, EUR 3.0%->4.0%, both sloping, so an
                interval's carry is a difference of cumulative integrals rather than a zero rate.
  time rows     varied - the strip fixes monthly for a year and settles two days on.
  side          varied - the strip is quoted from BOTH notional sides, which crosses the strike,
                the option sense and the knock-out direction at once.
  parameters    violent, and the cap negative, so the corner is a real branch in the walk.
  the conjunction  the quote claim needs the violent law AND a book stating a count no simulated
                leg can be read at AND both notional sides.
"""
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from derivus import service, structures, utils
from derivus.config import CustomJsonEncoder

HERE = os.path.dirname(os.path.abspath(__file__))
WORLD = os.path.join(HERE, 'fixtures', 'data', 'logvar2fj_world.json')
BASE = '2024-06-28'
FACTOR = 'LogVar2FJModelParameters.EUR'
PAIR, EXPIRY, FIXING_FREQUENCY = 'EURUSD', '1Y', '1M'
NOTIONAL = {'EUR': 1_000_000.0, 'USD': 1_100_000.0}

#: A FITTED RAND LAW's own shape - see the module note. `c = 1 - Rho_S^2 - Rho_L^2` is 0.79 and the
#: conditioning share 0.90, so the factor loads on its own asserts rather than a relaxed bound.
VIOLENT = {'Kappa_L': 0.5, 'Sigma_L': 0.5, 'Rho_L': -0.2, 'Kappa_S': 6.0,
           'Cap_A': -0.1266, 'C_Min': 0.12, 'Steps_Per_Year': 252.0, 'Residual_Law': 'NIG',
           'On_Guard': '', 'Skew_Gradient': '', 'Stickiness_Band': 0.0,
           'Xi_Curve': {'.Curve': {'meta': [], 'data': [
               [0.0, 0.0097216], [0.0821918, 0.0113690], [0.2493151, 0.0150484]]}},
           'Rho_S': {'.Curve': {'meta': [], 'data': [[0.0, -0.4155255]]}},
           'Beta': {'.Curve': {'meta': [], 'data': [[0.0, -18.5795935]]}},
           'Sigma_S': {'.Curve': {'meta': [], 'data': [[0.0, 1.9417594]]}},
           'Alpha': {'.Curve': {'meta': [], 'data': [[0.0, 58.4727]]}}}

#: The walk is read at the count a base valuation DECLARES and again at four times it, each inside
#: its own band. MEASURED over six seeds at these parameters: the per-path spread of
#: `exp(M + var/2)` is 2.0e-2 on the fitted axis and 2.4e-2 on the reciprocal, so the standard error
#: is 1.6e-4 / 1.8e-4 at the declared count and 7.9e-5 / 9.3e-5 at four times it, and the worst of
#: the six is 3.2e-4 and 1.6e-4. Each band is three of its own worst; a law whose compensator is
#: wrong by a step misses by percent.
DECLARED_BAND, WALK_PATHS, WALK_BAND = 1e-3, 1 << 16, 5e-4

#: What the two notional sides may differ by. MEASURED on this world at the paths a quote is priced
#: on; the same strip read off the book's own count lands 7.4e-2 apart, which is what this gate is.
AXIS_BAND = 1e-3


def curve(rows):
    return {'.Curve': {'meta': [], 'data': rows}}


def document(paths):
    """The world with the violent law on the euro, a strip's book, and `paths` inner paths.

    The market data travels EXPLICITLY, not behind `MarketDataFile`: the runner reads the book to
    find the pair's surface and never loads a file.
    """
    with open(WORLD) as handle:
        world = json.load(handle)['MarketData']
    market = {section: copy.deepcopy(world[section]) for section in (
        'System Parameters', 'Model Configuration', 'Price Factor Interpolation', 'Price Factors')}
    market['Price Factors'] = {name: value for name, value in market['Price Factors'].items()
                               if name.split('.')[0] in ('FxRate', 'InterestRate', 'FXVol')}
    market['Price Factors'][FACTOR] = copy.deepcopy(VIOLENT)
    market['Valuation Configuration'] = {'FXAccumulatorOptionDeal': {'SpotModel': 'LogVar2FJ'}}
    return {'Calc': {
        'Calculation': {'Object': 'BaseValuation', 'Base_Date': {'.Timestamp': BASE},
                        'Currency': 'USD', 'MCMC_Simulations': paths, 'Random_Seed': 1},
        'MergeMarketData': {'MarketDataFile': '', 'ExplicitMarketData': market},
        'Deals': {'Reference': 'test', 'Deals': {'Children': []}}}}


def params(device, invert=False):
    """The factor's leaves as the walk takes them, and the two the residual reads."""
    read = lambda name: VIOLENT[name]['.Curve']['data'][0][1]
    wide = lambda x: torch.tensor(float(x), dtype=torch.float64, device=device)
    return (dict({x: wide(VIOLENT[x]) for x in ('Kappa_L', 'Sigma_L', 'Rho_L', 'Kappa_S')},
                 Cap_A=VIOLENT['Cap_A'], Rho_S=wide(read('Rho_S')), Sigma_S=wide(read('Sigma_S'))),
            wide(read('Alpha')), wide(read('Beta')))


def walked(invert, horizon, blocks, paths, seed, device):
    """`(E[S_T]/F_T, its standard error)` off the model's own walk at a zero carry.

    One block per fixing interval, the residual on that block's clock, and the forward every
    lognormal primitive in `pricing` prices off - `exp(M + Sigma^2/2)` - averaged over the paths.
    """
    lever, alpha, beta = params(device)
    steps = int(round(horizon / blocks * float(VIOLENT['Steps_Per_Year'])))
    delta = torch.full((steps * blocks,), horizon / blocks / steps,
                       dtype=torch.float64, device=device)
    at = torch.cat([delta.new_zeros(1), delta.cumsum(0)])
    xi = utils.TermStructure(
        [row[0] for row in VIOLENT['Xi_Curve']['.Curve']['data']],
        torch.tensor([row[1] for row in VIOLENT['Xi_Curve']['.Curve']['data']],
                     dtype=torch.float64, device=device))
    level = torch.log(xi.at(at)) - 0.5 * utils.LogVar2FJ.state_variance(lever, delta)
    generator = torch.Generator(device=device).manual_seed(seed)
    draw = lambda *shape: torch.randn(shape, dtype=torch.float64, device=device,
                                      generator=generator)
    state = (delta.new_zeros(2 * paths) + level[0], delta.new_zeros(2 * paths))
    mirror = lambda x: torch.cat([x, -x])
    total = delta.new_zeros(2 * paths)
    for block in range(blocks):
        a, b = block * steps, (block + 1) * steps
        M, clock, *state = utils.LogVar2FJ.walk(
            lever, level[a:b + 1], delta[a:b], mirror(draw(paths, steps)),
            mirror(draw(paths, steps)), state, invert)
        budget = utils.LogVar2FJ.nig_budget(clock, alpha, beta)
        gamma = torch.sqrt(alpha * alpha - (beta + 1.0) ** 2) if invert else budget[2]
        u = torch.rand(paths, dtype=torch.float64, device=device, generator=generator)
        G = utils.LogVar2FJ.ig_quantile(torch.cat([u, 1.0 - u]), budget[0] / gamma,
                                        budget[0] * budget[0])
        M = M + budget[1] + beta * G
        total = total + (-(M + G) if invert else M) + 0.5 * G
    pairs = 0.5 * (torch.exp(total[:paths]) + torch.exp(total[paths:]))
    return float(pairs.mean()), float(pairs.std() / np.sqrt(paths))


def quoted_strip(paths, side):
    """The forward strip quoted with `side` as the notional currency, off a book stating `paths`."""
    return structures.quote(document(paths), 'Accumulator', {
        'pair': PAIR, 'expiry': EXPIRY, 'fixing_frequency': FIXING_FREQUENCY, 'leverage': 1.0,
        'notional': NOTIONAL[side], 'notional_currency': side, 'buy_currency': PAIR[:3],
        'knockout': 10.0})


def test_the_walked_law_holds_the_forward_on_both_axes():
    """E[S_T]/F_T is one at violent parameters, on the fitted axis and on the reciprocal.

    The two axes are ONE law read under two numeraires: the shocks tilt inside the walk and the
    residual's mixer is Esscher-tilted beside it, and each measure compensates its own return. A
    compensator formed on a clock the innovation does not spend - the uncapped variance against a
    capped draw, a per-block budget against a per-step clock, a native mixer under the tilt - is a
    percent of forward a year out, which is thirty of these bands.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for paths, band in ((structures.declared_paths(), DECLARED_BAND), (WALK_PATHS, WALK_BAND)):
        for invert in (False, True):
            ratio, error = walked(invert, 1.0, 12, paths, 1, device)
            assert error < band / 4.0, 'the reading is not sharp enough to claim anything'
            assert abs(ratio - 1.0) < band, '{} axis reads {:.6f} at {} paths'.format(
                'reciprocal' if invert else 'fitted', ratio, paths)


def test_a_quoted_forward_strip_solves_one_strike_from_either_side():
    """One zero-cost strip, two notional sides, one strike - off a book stating ONE inner path.

    A leverage-1 accumulator whose knock-out no path reaches is twelve forwards, so its solved
    strike is the average forward and cannot depend on which currency the notional is stated in.
    What it CAN depend on is how many paths the average was taken over: a strip walking a fitted law
    carries 2.8e-2 of spread per path where the same strip on the surface's own lognormal carries
    a thirtieth of that, so a count a cashflow's arithmetic is right for answers a draw.

    The book here STATES one path - what a desk book written before the declaration carried it
    holds - because a quote off a book already on disk still has to be a price, and the outcome
    says which count it was priced on.

    KILLING MUTATION - `structures.alone` and `risk_document` taking the book's own count rather
    than `structures.priced_job`'s floor: the two sides land 7.4e-2 apart, moving the solved strike
    by four percent of itself in each direction.
    """
    assert 'MCMC_Simulations' not in service.blank_book()['Calc']['Calculation'], (
        'a minted book states no count, so the declaration is the one number')
    assert 'MCMC_Simulations' not in service.JOB_SKELETON['Calc']['Calculation']

    quoted, notes = {}, {}
    for side in ('EUR', 'USD'):
        outcome = quoted_strip(1, side)
        quoted[side] = float(outcome['legs'][0]['strike_market'])
        notes[side] = outcome['notes']
    spread = abs(quoted['EUR'] / quoted['USD'] - 1.0)

    assert spread < AXIS_BAND, 'the two notional sides are {:.3e} apart: {}'.format(spread, quoted)
    # the thin book is TOLD, on the quote itself, which count it was priced on
    assert all('MCMC_Simulations is 1' in note[0] for note in notes.values()), notes
    assert quoted_strip(structures.declared_paths(), 'EUR')['notes'] is None


def test_a_quote_declares_the_paths_a_simulated_leg_needs():
    """The job a leg prices on carries the count the DECLARATION states, whatever the book says,
    and a book asking for more keeps its own."""
    declared = structures.declared_paths()

    assert declared == 1 << 14, 'the declaration moved - re-measure the bands above'
    assert structures.priced_job({'MCMC_Simulations': 1})['MCMC_Simulations'] == declared
    assert structures.priced_job({})['MCMC_Simulations'] == declared
    assert structures.priced_job(
        {'MCMC_Simulations': 1 << 20})['MCMC_Simulations'] == 1 << 20
    assert structures.priced_job({'Object': 'CreditMonteCarlo'})['Object'] == 'BaseValuation'
    assert json.dumps(document(1), cls=CustomJsonEncoder)
    # the greeks run a two-way is charged off is floored the same way, not just the pricing job
    thin = document(1)
    thin['Calc']['MergeMarketData']['ExplicitMarketData']['Market Prices'] = {
        'FXVolPrices.EUR.USD': {'instrument': {'Points': []}}}
    greeks = structures.risk_document(thin, [], 'EUR.USD')['Calc']['Calculation']
    assert greeks['MCMC_Simulations'] == declared and greeks['Greeks'] == 'First'
