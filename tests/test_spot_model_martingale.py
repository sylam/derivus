"""A spot model holds the forward, and a quote is priced on paths that can read it.

THE WORLD is `logvar2fj_world.json`'s shapes carrying a factor AS VIOLENT AS A FITTED RAND ONE -
Alpha 58, Beta -19, Sigma_S 1.9, Rho_S -0.42, a NEGATIVE Cap_A - because every gentle set of
parameters makes the claim below a tautology: a law with a per-path spread of a few tenths of a
percent reads the forward off any number of paths at all. The two notional sides of ONE zero-cost
forward strip solve one strike: a leverage-1 accumulator whose knock-out no path reaches IS a strip
of forwards, so its solved strike is the discount-weighted average forward from either side, the
fitted axis and the reciprocal a deal on the other notional pays on. The walk's own E[S_T]/F_T on
both axes measured 3.2e-4 and 1.6e-4 at worst over six seeds at the declared count.

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

from derivus import service, structures

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


def quoted_strip(paths, side):
    """The forward strip quoted with `side` as the notional currency, off a book stating `paths`."""
    return structures.quote(document(paths), 'Accumulator', {
        'pair': PAIR, 'expiry': EXPIRY, 'fixing_frequency': FIXING_FREQUENCY, 'leverage': 1.0,
        'notional': NOTIONAL[side], 'notional_currency': side, 'buy_currency': PAIR[:3],
        'knockout': 10.0})


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
