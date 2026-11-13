"""A parametric equity surface under `Sticky_Strike`, read at every fixing of a monitoring strip.

`calc_moneyness` answers a `Skew`/`SVI` surface under `Sticky_Strike` with the BARE STRIKE - the
division by `ATM_Ref` happens inside the surface, per expiry - so that answer carries neither a
fixing axis nor a batch, while every other rule's answer carries both. `forward_vol_strip` reads
its moneyness once per fixing and is the site that knows the strip's shape, so it is where the
coordinate is broadcast onto that shape. Unbroadcast, a discrete barrier on such a surface could
not be priced at all: at one reporting row the per-fixing index ran off the end, and at more than
one the read came back with a row per reporting date and was reshaped to a single number.

THE ORACLE IS THE SAME SMILE IN THE OTHER CONVENTION. `Sticky_Strike` reads the surface at
`log(K / ATM_Ref(T))` and `Sticky_Moneyness` at `log(K / F(T))`, so a surface whose `ATM_Ref` IS
its forward curve is the same coordinate written twice - and the two documents must then price to
the LAST BIT, not to a tolerance. Below the knots are the fixing tenors themselves and the spot
dates their forwards carry between, so the forward-valued `ATM_Ref` and every carry are read at
their own knots and no interpolation stands between them.

A SECOND ORACLE removes the smile instead of the difference: with no slope and no wings every
coordinate reads `ATM_Vol`, so the two rules agree whatever the reference is.

MEASURED on the document below (notional 1, strike 100, barrier 85, monthly monitoring, one year,
ATM 25 vol with a -0.15 slope). Declared `Sticky_Strike` against a flat `ATM_Ref` of 100 the
barrier marks 10.089815, and against `Sticky_Moneyness` 10.180800 - **-0.89%**, which is the whole
of the difference between reading a 25.00 vol at every fixing and reading the vol down the smile's
slope, the strike sitting under the forward by the 2% carry between spot dates.

THE SAME STRIP ON A SLOPED WORLD, its repo curve rising against a falling dividend one, reads the
surface's TERM STRUCTURE and the curves' FORWARD, each against a reference written here off the
stated curves and surface alone: a down-and-out against a quadrature of its monitoring, its
knock-in and knock-out against Black at the curves' forward, and the two conventions above against
each other where `ATM_Ref` is that forward, over a book of every pricer reading the surface. An
implied vol and a zero rate are both cumulative, so a wrong allocation over the fixings telescopes
to the right total: a European, a flat surface and a flat curve are blind to it, and only the
monitoring sees it.

RATE UNITS last, by dimension: the book with its dates twice as far, in a world of half the rates
and half the variance, prices and walks as it does at home, so a rate, a carry or a variance read
without the year fraction it accrues over shows at any tenor, a year included.
"""
import json
import os
import sys

# reference-derivus shadow-import guard (MEMORY): pin the package under test to THIS repo.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtr

import derivus
from derivus import utils
from derivus.config import CustomJsonEncoder, ModelParams
from test_option_settlement_json import lag

BASE = pd.Timestamp('2024-06-28')
SPOT, R_USD, Q_EQ = 100.0, 0.04, 0.02
X0, R_EUR, FX_VOL, RHO = 1.25, 0.02, 0.15, 0.35
STRIKE, UNITS, BARRIER, EXPIRY_D = 100.0, 1.0, 85.0, 365
MONITOR_D = list(range(30, 361, 30))

#: the knots ARE the fixing tenors and the spot dates their forwards carry between, so a
#: forward-valued ATM_Ref and every carry are read at their own knots
TENORS = sorted({0.02, 1.0, 2.0} | {d / 365.0 for d in MONITOR_D} | {
    (d + lag(BASE + pd.Timedelta(days=d), 3)) / 365.0 for d in [0] + MONITOR_D + [EXPIRY_D]})
SKEW = {'ATM_Vol': 0.25, 's': -0.15, 'L': 0.30, 'R': 0.20,
        'C': -0.35, 'D': 0.35, 'lam': 0.5, 'rho': 0.5}
#: Each world's repo and dividend zero rates by tenor: flat, and sloped against each other.
WORLDS = {'flat': (lambda t: R_USD, lambda t: Q_EQ),
          'sloped': (lambda t: 0.01 + 0.03 * t, lambda t: 0.04 - 0.02 * t)}


def curve(value):
    return utils.Curve([], [[t, value] for t in TENORS])


def log_forward(world, days):
    """`log(F / S)` of `world`'s curves to a fixing `days` on: the carry `(r(t) - q(t)) t` between
    the spot dates three business days on from the base date and from the fixing."""
    repo, dividend = WORLDS[world]
    zero = lambda d: (repo(d / 365.0) - dividend(d / 365.0)) * d / 365.0
    return zero(days + lag(day(round(days)), 3)) - zero(lag(BASE, 3))


def forward_ref(world):
    """The forward to each tenor knot of `world`'s curves, `S exp(log_forward)`."""
    return utils.Curve([], [[t, SPOT * float(np.exp(log_forward(world, t * 365.0)))] for t in TENORS])


def factors(rule, atm_ref, smile=True, world='flat'):
    repo, dividend = WORLDS[world]
    vol = {'Surface_Type': 'Skew', 'Moneyness_Rule': rule, 'Currency': 'USD', 'ATM_Ref': atm_ref}
    vol.update({k: curve(v if smile or k not in ('s', 'L', 'R') else 0.0)
                for k, v in SKEW.items()})
    return {
        'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Spot': 1.0},
        'InterestRate.USD': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                             'Curve': utils.Curve([], [[t, repo(t)] for t in TENORS])},
        'EquityPrice.EQ': {'Spot': SPOT, 'Currency': 'USD', 'Interest_Rate': 'USD',
                           'Issuer': '', 'Respect_Default': 'No', 'Jump_Level': 0.0},
        'DividendRate.EQ': {'Currency': 'USD', 'Floor': None,
                            'Curve': utils.Curve([], [[t, dividend(t)] for t in TENORS])},
        'EquityPriceVol.EQ': vol,
        'FxRate.EUR': {'Domestic_Currency': None, 'Interest_Rate': 'EUR', 'Spot': X0},
        'InterestRate.EUR': {'Currency': 'EUR', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                             'Curve': curve(R_EUR)},
        'FXVol.EUR.USD': fx_grid(FX_VOL),
        'Correlation.EquityPrice.EQ/FxRate.EUR.USD': {'Value': RHO}}


def explicit(sigma):
    """An `Explicit` surface reading `sigma(m, t)` at its moneyness and tenor knots."""
    return {'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness', 'Surface': utils.Curve(
        [], [[m, t, float(sigma(m, t))] for m in (0.6, 1.0, 1.4) for t in TENORS])}


def day(offset):
    return BASE + pd.DateOffset(days=offset)


DEAL = {'Object': 'EquityBarrierOption', 'Reference': 'BR', 'Currency': 'USD',
        'Payoff_Currency': 'USD', 'Equity': 'EQ', 'Dividends': 'EQ', 'Discount_Rate': 'USD',
        'Equity_Volatility': 'EQ', 'Buy_Sell': 'Buy', 'Option_Type': 'Call',
        'Strike_Price': STRIKE, 'Units': UNITS, 'Cash_Rebate': 0.0,
        'Expiry_Date': day(EXPIRY_D), 'Barrier_Type': 'Down_And_Out', 'Barrier_Price': BARRIER,
        'Barrier_Monitoring_Frequency': pd.DateOffset(days=0),
        'Barrier_Dates': [[day(d), ''] for d in MONITOR_D]}

BV = {'Object': 'BaseValuation', 'Base_Date': BASE, 'Currency': 'USD',
      'MCMC_Simulations': 8192, 'Quasi_Seed': 1}

#: the equity and the fx rate are simulated, so a CreditMonteCarlo reports a profile with many rows
CMC = {'Object': 'CreditMonteCarlo', 'Base_Date': BASE, 'Currency': 'USD',
       'Time_grid': '0d 12m(1m)', 'Batch_Size': 64, 'Simulation_Batches': 1,
       'Random_Seed': 1, 'MCMC_Simulations': 2048, 'Deflation_Interest_Rate': 'USD',
       'Generate_Cashflows': 'No'}


def models(stretch=1):
    """GBM for the equity and the fx rate, each variance divided by `stretch`."""
    return {'Model Configuration': ModelParams((
                {'EquityPrice': 'GBMAssetPriceModel', 'FxRate': 'GBMAssetPriceModel'}, {})),
            'Price Models': {'GBMAssetPriceModel.EQ': {'Drift': 0.0, 'Vol': 0.25 / np.sqrt(stretch)},
                             'GBMAssetPriceModel.EUR': {'Drift': 0.0, 'Vol': FX_VOL / np.sqrt(stretch)}}}


def run(price_factors, calc=BV, deals=(DEAL,), stretch=1):
    market = {'System Parameters': {'Base_Currency': 'USD', 'Base_Date': BASE},
              'Valuation Configuration': {}, 'Price Factors': price_factors}
    if calc is not BV:
        market.update(models(stretch))
    job = {'Calc': {
        'Calculation': dict(calc),
        'Deals': {'Reference': 'sticky',
                  'Deals': {'Children': [{'Instrument': {'.Deal': deal}} for deal in deals]}},
        'MergeMarketData': {'MarketDataFile': '', 'ExplicitMarketData': market}}}
    cx = derivus.Context()
    cx.load_json((json.dumps(job, cls=CustomJsonEncoder), 'sticky'))
    _, out = cx.run_job()
    return out['Results']['mtm']


def marks(price_factors, deals=(DEAL,)):
    """`{reference: mark}` of `deals` priced in one document. A deal that left no row was
    skipped, and says so."""
    rows = run(price_factors, deals=deals)
    own = {deal['Reference']: rows[rows['Reference'] == deal['Reference']]['Value'] for deal in deals}
    assert all(len(v) == 1 for v in own.values()), 'a deal left no row in Results: it was skipped'
    return {reference: float(v.iloc[0]) for reference, v in own.items()}


def book(stretch=1):
    """The monthly knock-out and, to 300 days, a tenor other than a year, every other pricer of the
    strip and the closed forms that reads a surface: the knock-in, the continuous knock-out, the
    one-touch, the Asian, a composite knock-in and an fx partial-time knock-in, each date `stretch`
    times as far from the base date."""
    at = lambda d: day(stretch * d)
    short = dict(DEAL, Expiry_Date=at(300))
    equity = {k: DEAL[k] for k in ('Currency', 'Payoff_Currency', 'Equity', 'Dividends',
                                   'Discount_Rate', 'Equity_Volatility', 'Buy_Sell')}
    return (
        dict(DEAL, Expiry_Date=at(EXPIRY_D), Barrier_Dates=[[at(d), ''] for d in MONITOR_D]),
        dict(short, Reference='KI', Barrier_Type='Down_And_In',
             Barrier_Dates=[[at(d), ''] for d in MONITOR_D[:10]]),
        dict(short, Reference='CONT', Barrier_Dates=[]),
        dict(equity, Object='EquityOneTouchOption', Reference='TOUCH', Cash_Payoff=100.0,
             Barrier_Type='Down', Barrier_Price=BARRIER, Payment_Timing='Expiry', Expiry_Date=at(300),
             Barrier_Dates=[], Barrier_Monitoring_Frequency=pd.DateOffset(days=0)),
        dict(equity, Object='EquityDiscreteExplicitAsianOption', Reference='ASIAN', Option_Type='Call',
             Strike_Price=STRIKE, Units=UNITS, Expiry_Date=at(300), Is_Digital='No',
             Sampling_Data=[[at(d), 0.0, 1.0] for d in (90, 180, 270, 300)]),
        dict(short, Reference='COMPO', Payoff_Currency='EUR', Discount_Rate='EUR', Payoff_Type='Compo',
             Strike_Price=80.0, Barrier_Price=70.0, Barrier_Type='Down_And_In',
             Barrier_Dates=[[at(d), ''] for d in (90, 180, 270, 300)]),
        {'Object': 'FXPartialTimeBarrierOption', 'Reference': 'PARTIAL', 'Currency': 'USD',
         'Underlying_Currency': 'EUR', 'Payoff_Currency': 'USD', 'Discount_Rate': 'USD',
         'FX_Volatility': 'EUR.USD', 'Buy_Sell': 'Buy', 'Option_Type': 'Call', 'Strike_Price': X0,
         'Barrier_Price': 1.12, 'Barrier_Type': 'Down_And_In', 'Barrier_At_Start': 'Yes',
         'Barrier_Limit_Date': at(150), 'Expiry_Date': at(300), 'Underlying_Amount': 1000.0,
         'Barrier_Monitoring_Frequency': pd.DateOffset(days=0)})


def test_a_skew_surface_under_sticky_strike_prices_a_monitoring_strip():
    """The row this file exists for: the barrier prices, and prices to its measured mark,
    re-measured when a base valuation's inner paths became the canonical block's, 10.088311 to
    10.093277, and when the strip's forward carried between spot dates, 10.089815.

    Killing mutation: the strip's moneyness left unbroadcast onto its fixing axis.
    """
    assert marks(factors('Sticky_Strike', curve(SPOT)))['BR'] == pytest.approx(10.089815, rel=1e-6)


@pytest.mark.parametrize('world', ['flat', 'sloped'])
def test_the_two_conventions_are_one_coordinate_where_the_reference_is_the_forward(world):
    """`log(K / ATM_Ref)` and `log(K / F)` are the same number when `ATM_Ref` is `F`, so the two
    documents read the same vol at every fixing and every expiry and mark the book alike within
    1e-12, the reference being built here and the forward on the device.
    This is what places the coordinate on the surface: a read at the wrong tenor, or at a forward
    other than the curves' (the spot, the interval carry's, one missing its year fraction), breaks
    this identity while still producing a number - on the sloped world the knock-out marks
    10.141810 where the reference at the spot reads 10.051586, -0.89%.

    THE CONTROL: with no slope and no wings every coordinate reads `ATM_Vol`, so the reference
    cannot matter and all three spellings agree - the difference between that and the identity
    above is what sizes the convention, for every deal reading the equity surface as its own.

    Killing mutations: `Sticky_Moneyness` read against the spot where it wants the forward; the
    barrier's strip forwarded on the interval carry (`forward_carry_rate`'s) where it wants the zero
    carry, red on the sloped world alone; `forward_vol_strip`'s forward without its year fractions;
    the discrete barrier's expiry read, the continuous barrier's and the one-touch's each forwarded
    without the year fraction to the expiry.
    """
    def priced(rule, atm_ref, smile=True):
        return marks(factors(rule, atm_ref, smile, world), book())
    ref = forward_ref(world)
    assert priced('Sticky_Strike', ref) == pytest.approx(
        priced('Sticky_Moneyness', curve(SPOT)), rel=1e-12)
    flat = priced('Sticky_Strike', curve(SPOT), smile=False)
    assert flat == priced('Sticky_Moneyness', curve(SPOT), smile=False) == priced(
        'Sticky_Strike', ref, smile=False)
    smiled = priced('Sticky_Moneyness', curve(SPOT))
    assert all(flat[r] != smiled[r] for r in ('BR', 'KI', 'CONT', 'TOUCH', 'ASIAN')), (
        'the smile must be worth something')


def stretched(k):
    """The flat world `k` times as slow: every rate and variance divided by `k`, each surface a
    smile proportional to its moneyness and constant in tenor."""
    price_factors = factors('Sticky_Moneyness', curve(SPOT))
    for name, rate in (('InterestRate.USD', R_USD), ('DividendRate.EQ', Q_EQ),
                       ('InterestRate.EUR', R_EUR)):
        price_factors[name]['Curve'] = curve(rate / k)
    for name, vol in (('EquityPriceVol.EQ', SKEW['ATM_Vol']), ('FXVol.EUR.USD', FX_VOL)):
        price_factors[name] = explicit(lambda m, t, vol=vol: vol * m / np.sqrt(k))
    return price_factors


def unlagged(stretch):
    """`book(stretch)`, every deal stating `Settlement_Days` 0: a spot date is business days on, a
    lag that does not stretch."""
    return tuple(dict(deal, Settlement_Days=0) for deal in book(stretch))


def test_the_book_twice_as_far_in_a_world_half_as_fast_prices_the_same():
    """RATE UNITS. A price reads a rate, a carry or a variance only times the year fraction it
    accrues over, so the book with every date twice as far from the base date, in the world whose
    rates and variances are halved, prices the same, to 1e-13: 10.093277 the knock-out, 54.756450
    the one-touch, 3.523722 the partial-time knock-in. A quantity reaching `exp` or a closed form
    without its year fraction reads the slow world's half. The book carries no spot lag, which is
    business days on and does not stretch.

    Killing mutations, each a year fraction dropped where a carry meets it: in the Reiner-Rubinstein
    barrier (the continuous knock-out -5.9%), the one-touch's drift (+0.01%), the partial-time
    closed form's forward (+540%), the Asian's sample forwards (-7.4%), the composite strip's fx
    forwards (-22.7%), `zero_carry_rate`'s division (every deal, -2.9% to +35%).
    """
    assert marks(stretched(2), unlagged(2)) == pytest.approx(marks(stretched(1), unlagged(1)), rel=1e-13)


def test_the_book_twice_as_far_in_a_world_half_as_fast_walks_the_same_profile():
    """The same statement of a credit Monte Carlo's profile, its grid and its models stretched with
    the book (GBM equity and fx rate, each variance halved), which reaches the branches only a row
    after the base date reads: 14 rows of 64 paths agree to 2.7e-6, the float32 the simulation runs
    in, where 1e-5 is asked. The book carries no spot lag here either.

    Killing mutation: the partial-time knock-in's knocked-in vanilla forwarded without the year
    fraction to its expiry (10.6% on its worst entry).
    """
    def walked(k):
        return run(stretched(k), dict(CMC, Time_grid='0d {0}d({0}d)'.format(30 * k)), unlagged(k), k)
    np.testing.assert_allclose(walked(2).values, walked(1).values, rtol=1e-5)


#: Two surfaces flat in moneyness that agree at the one-year expiry: flat, and rising to it from 10%.
SURFACES = {'flat': lambda t: 0.25 + 0.0 * t, 'rising': lambda t: 0.10 + 0.15 * np.minimum(t, 1.0)}
#: The down-and-out at 95, which the term structure moves, its knock-in and the European.
KNOCK_OUT = dict(DEAL, Reference='KO', Barrier_Price=95.0)
KNOCK_IN = dict(KNOCK_OUT, Reference='KI', Barrier_Type='Down_And_In')
EUROPEAN = {k: v for k, v in DEAL.items() if not k.startswith(('Barrier', 'Cash'))} | {
    'Object': 'EquityOptionDeal', 'Reference': 'EU'}


def call(forward, sd):
    """Black's undiscounted call at `STRIKE`."""
    d1 = np.log(forward / STRIKE) / sd + 0.5 * sd
    return forward * ndtr(d1) - STRIKE * ndtr(d1 - sd)


def quadrature(sigma, n=1 << 14):
    """The knock-out off the sloped world's curves and `sigma` alone: the log-spot law stepped from
    fixing to fixing in Fourier space, each interval's carry and variance the difference of the
    cumulative ones the curves and the surface state, each carry between spot dates, the mass below
    the barrier dropped at each fixing and the last interval's call in closed form, discounted on the
    repo curve."""
    repo = WORLDS['sloped'][0]
    t = np.array(MONITOR_D + [EXPIRY_D]) / 365.0
    carry = np.diff([log_forward('sloped', d) for d in MONITOR_D + [EXPIRY_D]], prepend=0.0)
    var = np.diff(sigma(t) ** 2 * t, prepend=0.0)
    h = 20.0 * np.sqrt(var.sum()) / n
    x = np.log(SPOT) + h * (np.arange(n) - n // 2)
    omega = 2.0 * np.pi * np.fft.rfftfreq(n, h)
    mass = (x == np.log(SPOT)).astype(float)
    for c, v in zip(carry[:-1], var[:-1]):
        law = np.exp(-1j * omega * (c - 0.5 * v) - 0.5 * omega ** 2 * v)
        mass = np.fft.irfft(np.fft.rfft(mass) * law, n) * (x >= np.log(KNOCK_OUT['Barrier_Price']))
    return np.exp(-repo(1.0)) * np.sum(mass * call(np.exp(x + carry[-1]), np.sqrt(var[-1])))


@pytest.fixture(scope='module')
def monitored():
    """`{surface: {reference: mark}}`: the knock-out, the knock-in and the European on each surface
    of the sloped world, one document a surface."""
    return {name: marks(dict(factors('Sticky_Moneyness', curve(SPOT), world='sloped'),
                             **{'EquityPriceVol.EQ': explicit(lambda m, t, sigma=sigma: sigma(t))}),
                        (KNOCK_OUT, KNOCK_IN, EUROPEAN))
            for name, sigma in SURFACES.items()}


def test_two_surfaces_agreeing_at_expiry_monitor_apart_each_on_its_quadrature(monitored):
    """The vol term structure reaches the monitoring. The flat surface and the rising one agree at
    expiry, so the European reads them to the bit, while the monthly down-and-out steps each
    interval's own forward variance and carry, the cumulative ones differenced (`forward_vol_rate`,
    `forward_carry_rate`), and lands on the quadrature of the curves and surface it reads: 7.024910
    against 7.030409 (-0.08%) on the flat surface and 6.526941 against 6.520982 (+0.09%) on the
    rising one, the two quadratures 7.2% apart. 3e-3 is the inner Sobol block's slack. 2 s.

    Killing mutations: every interval read at the expiry vol (the shipped defect: the rising surface
    then prices as the flat, +7.9%); the interval carry undifferenced (the zero rate to each fixing
    times the interval, -10.9%); the step's carry without its year fraction (`carry * dt`, +98%);
    the zero carry without its year fraction (`zero_carry_rate`'s division, +1.3%).
    """
    assert monitored['flat']['EU'] == monitored['rising']['EU']
    for name, sigma in SURFACES.items():
        assert monitored[name]['KO'] == pytest.approx(quadrature(sigma), rel=3e-3), name
    assert quadrature(SURFACES['rising']) < 0.95 * quadrature(SURFACES['flat']), (
        'the surfaces must monitor apart')


def test_the_knock_in_and_out_sum_to_the_vanilla_at_the_curves_forward(monitored):
    """The knock-in is priced by parity, the vanilla less the knock-out, so the two sum to the
    vanilla the barrier's payoff is valued at: Black at the forward `total_log_forward` integrates
    off the interval strip, at the surface's expiry vol, discounted. On the sloped world it is
    Black at `S exp((r - q) t)` between the spot dates off the stated curves on either surface,
    10.722376, to the bit where 1e-12 is asked.

    Killing mutations: `total_log_forward` summing the interval carry without its year fractions
    (35.300588); the interval carry undifferenced, whose strip telescopes to the curves' total only
    on a flat curve (9.401694).
    """
    repo = WORLDS['sloped'][0]
    for name, sigma in SURFACES.items():
        vanilla = np.exp(-repo(1.0)) * call(SPOT * np.exp(log_forward('sloped', EXPIRY_D)), sigma(1.0))
        assert monitored[name]['KO'] + monitored[name]['KI'] == pytest.approx(
            vanilla, rel=1e-12), name


#: A `Skew` FX surface, `Sticky_Strike` about 1.25 with a 15% ATM and a 0.5 slope, no wings.
FX_SKEW = {'Surface_Type': 'Skew', 'Moneyness_Rule': 'Sticky_Strike', 'ATM_Ref': curve(1.25),
           'ATM_Vol': curve(0.15), 's': curve(0.5), 'L': curve(0.0), 'R': curve(0.0),
           'C': curve(-1.0), 'D': curve(1.0), 'lam': curve(0.0), 'rho': curve(0.0)}


def fx_grid(vol):
    """`FXVol.EUR.USD` as an explicit grid flat in moneyness and tenor at `vol`."""
    return {'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness', 'Surface': utils.Curve(
        [], [[m, t, vol] for m in (0.6, 1.0, 1.4) for t in (0.02, 2.0)])}


def test_a_skew_fx_surface_mints_its_parameters_and_prices_at_its_own_read():
    """FX vol surfaces share the equity ones' getter, so a `Skew` FX surface mints its parameter
    sub-factors where it named the bare surface and every FX deal on it skipped. An FX option reads
    it at its strike - `0.15 + 0.5 log(1.30 / 1.25)` under `Sticky_Strike` about an `ATM_Ref` of
    1.25 - and prices to the bit as on an explicit grid flat at that vol; an accumulator prices.

    Killing mutation: the vol getter answering the bare surface name for a parametric surface.
    """
    import test_declared_defaults as book
    deals = [book.fx_leg('FXOptionDeal', kind, Expiry_Date=book.WORLD_EXPIRY, Strike_Price=1.30,
                         Option_Type=kind) for kind in ('Call', 'Put')]
    on_skew, stats = book.marks(deals + [next(d for d in book.BOOK if d['Reference'] == 'ACC')],
                                {'FXVol.EUR.USD': FX_SKEW})
    assert 'Deals Skipped' not in stats and np.isfinite(float.fromhex(on_skew['ACC']))
    on_grid, _ = book.marks(deals, {'FXVol.EUR.USD': fx_grid(0.15 + 0.5 * np.log(1.30 / 1.25))})
    assert all(on_skew[kind] == on_grid[kind] for kind in ('Call', 'Put')), (on_skew, on_grid)


def test_the_fx_averages_and_the_extendable_forward_read_the_smile_where_the_fx_option_does():
    """The double Asian reads its at-the-money vol at the spot and the extendable forward each
    transition vol at its extension strike, both through `calc_moneyness`. On the Skew above the
    double Asian reads 15% at the 1.25 spot and the forward `0.15 + 0.5 log(1.28 / 1.25)` at every
    fixing; on a Malz smile, 10% at the money and 30% from |x| = 0.5, the double Asian reads
    `0.1 + 0.4 log(F(t) / S)` at each sample's own tenor `t`, its forward the 2% carry between spot
    dates - each within an ulp of the deal on a grid flat in moneyness at that vol, tenor by tenor.

    Killing mutation: the double Asian's read back to its own ratio, which lands on the Malz
    wing's 30%.
    """
    import test_declared_defaults as book
    import trial_fx
    deals = [deal for deal in trial_fx.DEALS if deal['Reference'] in ('FXDASN', 'FXEXT')]
    malz = {'Surface_Type': 'Malz', 'Moneyness_Rule': 'Sticky_Moneyness', 'Surface': utils.Curve(
        [], [[x, t, vol] for x, vol in ((-0.5, 0.3), (0.0, 0.1), (0.5, 0.3)) for t in (0.02, 2.0)])}
    base = book.WORLD_BASE
    samples = [row[0] for deal in deals if deal['Reference'] == 'FXDASN'
               for key in ('Sampling_Data_1', 'Sampling_Data_2') for row in deal[key]]
    on_samples = {'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness', 'Surface': utils.Curve(
        [], [[m, (d - base).days / 365.0, 0.1 + 0.4 * (book.R_USD - book.R_EUR) * (
            (d - base).days + lag(d, 2) - lag(base, 2)) / 365.0] for m in (0.6, 1.0, 1.4) for d in samples])}

    def mark(surface, reference):
        return float.fromhex(book.marks(deals, {'FXVol.EUR.USD': surface})[0][reference])

    for surface, reference, grid in ((FX_SKEW, 'FXDASN', fx_grid(0.15)),
                                     (FX_SKEW, 'FXEXT', fx_grid(0.15 + 0.5 * np.log(1.28 / 1.25))),
                                     (malz, 'FXDASN', on_samples)):
        assert mark(surface, reference) == pytest.approx(mark(grid, reference), rel=1e-15), reference


def test_the_strip_is_read_the_same_way_at_every_reporting_row():
    """A profile asks for the strip once per reporting date, where a valuation asks once. The
    second shape is the one that came back with a row per date and was reshaped to one number.

    Killing mutation: the strip's moneyness left unbroadcast onto its fixing axis.
    """
    profile = run(factors('Sticky_Strike', curve(SPOT)), CMC)
    assert len(profile) > 1 and np.isfinite(profile.values).all()
    assert float(np.mean(profile.iloc[0].values)) == pytest.approx(10.088311, rel=5e-3)
