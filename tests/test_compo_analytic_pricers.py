"""A composite-currency barrier, one-touch and discrete Asian price on the COMPOSITE, through the
JSON contract and nothing else.

The payoff of a compo is written on S*X - the local asset at the cross into the payoff currency -
so the priced and MONITORED process is that product: spot at the cross, forward at the outright fx
forward, carry at the local carry plus the fx forward's own, vol at the composition of the two
legs. The local smile is read at the strike TRANSLATED by the fx forward, `K / F_X(T)` being the
local strike a payoff-currency strike on S*X is worth, so the moneyness is a local moneyness.

THE ORACLES ARE INDEPENDENT. In-out parity reads the barrier's two closed forms against
`pv_european_option`, a different pricer that already rebuilt its own composite forward; the
one-touch reads the textbook `erfc` pair written out here; the Asian reads the two-moment match
recomputed in python on the composite forward, vol and carry. The limit is the engine against
itself: an fx leg identically one with no vol and no correlation must reprice the Standard deal.

WHY THE PARITY IS SPLIT IN TWO. `EquityOptionDeal` carries its equity forward to expiry plus the
two-business-day settlement lag (`option_date_info`) while the three analytic pricers carry theirs
to expiry, so the two forwards differ by the carry over that lag whenever the local carry is live -
a convention difference that predates the composite and is nothing to do with it. The exact reading
is therefore taken where the local carry is flat and the lag is the identity; the live-carry
conjunction is read at the lag it explains.

MEASURED. In-out parity against the composite European, flat local carry: worst 2.0e-15 relative
over the eight arms, against 221% on main. The whole conjunction: worst 1.4e-3, the settlement lag,
against 145%. The one-touch already beyond the composite spot: exact, against a 47.6% miss. A live
one-touch against the textbook: worst 1.6e-15, against 126%. The Asian against its own moment
match: equal to the bit, against 333%. The unit-fx limit: bit-equal on all four arms, on both
trees. The monitored path: 0.5435 mean touch probability against 0.0000.

DEGENERACY CHECKLIST: r = 4% (USD) against 2% (EUR) against q = 1%, none zero and none equal; both
barrier directions and both option types on every parity; a skewed equity surface beside the flat
one; fx vol 15% and an implied correlation of -35% (authored +35% on the sorted pair, the deals
running USD -> EUR); and one run holding the whole conjunction at once.
"""
import json
import math
import os
import sys

# reference-derivus shadow-import guard (MEMORY): pin the package under test to THIS repo.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

import derivus
from derivus import utils
from derivus.config import CustomJsonEncoder

BASE = pd.Timestamp('2024-06-28')
EXPIRY = BASE + pd.DateOffset(years=1)
T = (EXPIRY - BASE).days / 365.0
SPOT, R_USD, R_EUR, Q_EQ = 100.0, 0.04, 0.02, 0.01
VOL_ATM, SKEW = 0.25, 0.5           # sigma(m) = VOL_ATM + SKEW * (m - 1), m = spot / strike
FX_SPOT, FX_SIGMA, CORR = 1.25, 0.15, 0.35
RHO = -CORR                         # authored on the sorted pair, the deals run USD -> EUR
UNITS, CASH = 1000.0, 1000.0
STRIKE, UP, DOWN = 80.0, 92.0, 70.0                 # payoff-currency levels on S*X

#: the outright fx forward to expiry, EUR per USD, and the composite spot it scales
FX_FWD = math.exp((R_EUR - R_USD) * T) / FX_SPOT
CMP_SPOT = SPOT / FX_SPOT

RATES = {
    'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Spot': 1.0},
    'FxRate.EUR': {'Domestic_Currency': None, 'Interest_Rate': 'EUR', 'Spot': FX_SPOT},
    'InterestRate.USD': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                         'Curve': utils.Curve([], [[0.0, R_USD], [5.0, R_USD]])},
    'InterestRate.EUR': {'Currency': 'EUR', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                         'Curve': utils.Curve([], [[0.0, R_EUR], [5.0, R_EUR]])}}


def _surface(atm, slope):
    """Three COLLINEAR moneyness nodes, constant in time - every interpolation is the same line."""
    return {'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness',
            'Surface': utils.Curve([], [[m, t, atm + slope * (m - 1.0)]
                                        for m in (0.8, 1.0, 1.2) for t in (0.02, 2.0)])}


def _factors(skew=SKEW, q=Q_EQ, eq_vol=VOL_ATM, fx_vol=FX_SIGMA, corr=CORR):
    return dict(RATES, **{
        'EquityPrice.EQ': {'Spot': SPOT, 'Currency': 'USD', 'Interest_Rate': 'USD', 'Issuer': '',
                           'Respect_Default': 'No', 'Jump_Level': 0.0},
        'DividendRate.EQ': {'Currency': 'USD', 'Curve': utils.Curve([], [[0.0, q], [5.0, q]])},
        'VolatilityGrid.EQ': _surface(eq_vol, skew),
        'FXVol.EUR.USD': _surface(fx_vol, 0.0),
        'Correlation.EquityPrice.EQ/FxRate.EUR.USD': {'Value': corr}})


def _one_factors():
    """The limit world: a second numeraire-priced currency whose cross with USD is identically one,
    with no fx vol and no correlation authored. The compo arithmetic must reduce to the Standard
    deal's on it."""
    return dict(_factors(), **{
        'FxRate.USDX': {'Domestic_Currency': None, 'Interest_Rate': 'USDX', 'Spot': 1.0},
        'InterestRate.USDX': {'Currency': 'USDX', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                              'Curve': utils.Curve([], [[0.0, R_USD], [5.0, R_USD]])},
        'FXVol.USD.USDX': _surface(0.0, 0.0)})


# --------------------------------------------------------------------------------------------
# the deals, one spelling each, Standard or Compo by the payoff currency they settle in
# --------------------------------------------------------------------------------------------
def _ccy(payoff):
    return ({'Payoff_Currency': 'USD', 'Discount_Rate': 'USD'} if payoff is None else
            {'Payoff_Currency': payoff, 'Discount_Rate': payoff, 'Payoff_Type': 'Compo'})


def _barrier(ref, barrier_type, barrier, option_type='Call', payoff='EUR', strike=STRIKE):
    return dict({
        'Object': 'EquityBarrierOption', 'Reference': ref, 'Currency': 'USD', 'Equity': 'EQ',
        'Dividends': 'EQ', 'Equity_Volatility': 'EQ', 'Buy_Sell': 'Buy',
        'Option_Type': option_type, 'Strike_Price': strike, 'Units': UNITS,
        'Expiry_Date': EXPIRY, 'Barrier_Type': barrier_type, 'Barrier_Price': barrier,
        'Cash_Rebate': 0.0, 'Barrier_Dates': [],
        'Barrier_Monitoring_Frequency': pd.DateOffset(days=0)}, **_ccy(payoff))


def _vanilla(ref, option_type='Call', payoff='EUR', strike=STRIKE):
    return dict({
        'Object': 'EquityOptionDeal', 'Reference': ref, 'Currency': 'USD', 'Equity': 'EQ',
        'Dividends': 'EQ', 'Equity_Volatility': 'EQ', 'Buy_Sell': 'Buy',
        'Option_Type': option_type, 'Strike_Price': strike, 'Units': UNITS,
        'Expiry_Date': EXPIRY}, **_ccy(payoff))


def _one_touch(ref, direction, barrier, payoff='EUR'):
    return dict({
        'Object': 'EquityOneTouchOption', 'Reference': ref, 'Currency': 'USD', 'Equity': 'EQ',
        'Equity_Volatility': 'EQ', 'Buy_Sell': 'Buy', 'Cash_Payoff': CASH,
        'Barrier_Type_One': direction, 'Barrier_Price': barrier, 'Payment_Timing': 'Expiry',
        'Expiry_Date': EXPIRY, 'Barrier_Dates': [],
        'Barrier_Monitoring_Frequency': pd.DateOffset(days=0)}, **_ccy(payoff))


SAMPLES = [BASE + pd.DateOffset(months=m) for m in (3, 6, 9, 12)]


def _asian(ref, option_type='Call', payoff='EUR', strike=STRIKE):
    return dict({
        'Object': 'EquityDiscreteExplicitAsianOption', 'Reference': ref, 'Currency': 'USD',
        'Equity': 'EQ', 'Dividends': 'EQ', 'Equity_Volatility': 'EQ', 'Buy_Sell': 'Buy',
        'Option_Type': option_type, 'Strike_Price': strike, 'Units': UNITS,
        'Expiry_Date': EXPIRY, 'Is_Digital': 'No',
        'Sampling_Data': [[d, 0.0, 1.0] for d in SAMPLES]}, **_ccy(payoff))


# --------------------------------------------------------------------------------------------
# the harness
# --------------------------------------------------------------------------------------------
def _job(deals, factors, calc=None, models=None, currency='EUR'):
    # reported in EUR, the compo deals' own payoff currency, so a reported value IS its price
    market = {'System Parameters': {'Base_Currency': 'USD', 'Base_Date': BASE},
              'Valuation Configuration': {}, 'Price Factors': factors}
    if models:
        market['Price Models'] = models
        market['Model Configuration'] = {'.ModelParams': {
            'modeldefaults': {'EquityPrice': 'GBMAssetPriceModel', 'FxRate': 'GBMAssetPriceModel'},
            'modelfilters': {}}}
    return {'Calc': {
        'Calculation': dict({'Object': 'BaseValuation', 'Base_Date': BASE,
                             'Currency': currency, 'MCMC_Simulations': 1,
                             'Random_Seed': 1}, **(calc or {})),
        'Deals': {'Tag_Titles': '', 'Reference': 'compo',
                  'Deals': {'Children': [{'Instrument': {'.Deal': d}} for d in deals]}},
        'MergeMarketData': {'MarketDataFile': '', 'ExplicitMarketData': market}}}


def _run(job):
    cx = derivus.Context()
    cx.load_json((json.dumps(job, cls=CustomJsonEncoder), 'compo'))
    _, out = cx.run_job()
    return out


def _mtm(out, ref):
    rows = out['Results']['mtm']
    return float(rows[rows['Reference'] == ref]['Value'].iloc[0])


# --------------------------------------------------------------------------------------------
# the oracles
# --------------------------------------------------------------------------------------------
def _local_sigma(level, skew=SKEW, atm=VOL_ATM):
    """The local surface at the TRANSLATED level: the moneyness is the local spot over the local
    strike `K / F_X(T)`, which is the local spot times the fx forward over K."""
    return atm + skew * (SPOT * FX_FWD / level - 1.0)


def _compo_sigma(level, **kwargs):
    s = _local_sigma(level, **kwargs)
    return math.sqrt(s * s + 2.0 * RHO * s * FX_SIGMA + FX_SIGMA * FX_SIGMA)


def _one_touch_closed(barrier, up, spot, carry, sigma, rate):
    """The textbook one-touch paying at EXPIRY, written from `erfc` and sharing nothing with the
    engine's flattened algebra."""
    mu = carry / sigma - 0.5 * sigma
    x = math.log(barrier / spot) / sigma
    # BARRIER_UP is -1 in the engine's enumeration, and that is the sign this erfc pair carries
    root, scale = math.sqrt(T), 0.7071067811865476 * (-1.0 if up else 1.0)
    return CASH * math.exp(-rate * T) * 0.5 * (
        math.erfc(scale * (mu * root - x / root)) +
        math.exp(2.0 * mu * x) * math.erfc(scale * (-mu * root - x / root)))


def _moment_matched(weights, taus, carry, sigma, spot):
    """`pv_discrete_asian_option`'s two-moment match, recomputed here on the composite."""
    ft = np.array([w * math.exp(carry * t) for w, t in zip(weights, taus)])
    m1 = ft.sum()
    product = ft * np.exp(np.array(taus) * sigma * sigma)
    running = np.concatenate([[0.0], np.cumsum(product[:-1])])
    m2 = float((ft * (product + 2.0 * running)).sum())
    return m1 * spot, math.sqrt(math.log(m2) - 2.0 * math.log(m1))


def _black(forward, strike, sd, call):
    d1 = math.log(forward / strike) / sd + 0.5 * sd
    phi = 1.0 if call else -1.0
    ndtr = lambda z: 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return phi * (forward * ndtr(phi * d1) - strike * ndtr(phi * (d1 - sd)))


# --------------------------------------------------------------------------------------------
# 1. in-out parity against the composite European
# --------------------------------------------------------------------------------------------
PARITY = [(ud, barrier, option_type, skew)
          for ud, barrier in (('Up', UP), ('Down', DOWN))
          for option_type in ('Call', 'Put')
          for skew in (0.0, SKEW)]


@pytest.mark.parametrize('ud,barrier,option_type,skew', PARITY,
                         ids=['%s-%s-%s' % (u, o, 'skew' if s else 'flat')
                              for u, _, o, s in PARITY])
def test_a_composite_knock_in_plus_its_knock_out_is_the_composite_european(
        ud, barrier, option_type, skew):
    """The two closed forms of `pv_barrier_option` must add up to `pv_european_option`'s composite
    price, on a flat surface and on a skewed one. Both pricers rebuild the composite themselves and
    neither shares a line with the other; on main the barrier keeps the LOCAL spot and forward and
    the sum misses by the whole fx translation.

    The local carry is flat here so the vanilla's two-business-day forward settlement is the
    identity and the reading is float64 round-off; measured worst 2.0e-15 relative."""
    factors = _factors(skew=skew, q=R_USD)
    deals = [_barrier('KI', '%s_And_In' % ud, barrier, option_type),
             _barrier('KO', '%s_And_Out' % ud, barrier, option_type),
             _vanilla('EUR_VANILLA', option_type)]
    out = _run(_job(deals, factors))
    ki, ko, eur = _mtm(out, 'KI'), _mtm(out, 'KO'), _mtm(out, 'EUR_VANILLA')
    assert abs(ki + ko - eur) / abs(eur) < 1e-12, (ki, ko, eur)


def test_the_whole_conjunction_parities_at_the_vanilla_s_settlement_lag():
    """One run holding every live quantity at once: a non-zero local carry against two non-zero and
    different rates, the skewed surface, a live fx vol and a live correlation, both barrier
    directions and both option types. The only residual is the vanilla's own forward settlement
    lag: four calendar days at a 3% carry move the forward by 3.3e-4, and the vanilla's forward
    elasticity of about 4.6 carries that to the price. Measured worst 1.4e-3 relative."""
    factors = _factors()
    deals = []
    for ud, barrier in (('Up', UP), ('Down', DOWN)):
        for option_type in ('Call', 'Put'):
            tag = '%s%s' % (ud[0], option_type[0])
            deals += [_barrier('KI' + tag, '%s_And_In' % ud, barrier, option_type),
                      _barrier('KO' + tag, '%s_And_Out' % ud, barrier, option_type)]
    deals.append(_vanilla('VC', 'Call'))
    deals.append(_vanilla('VP', 'Put'))
    out = _run(_job(deals, factors))
    for ud, _ in (('Up', UP), ('Down', DOWN)):
        for option_type in ('Call', 'Put'):
            tag = '%s%s' % (ud[0], option_type[0])
            got = _mtm(out, 'KI' + tag) + _mtm(out, 'KO' + tag)
            ref = _mtm(out, 'V' + option_type[0])
            assert abs(got - ref) / abs(ref) < 2e-3, (tag, got, ref)


# --------------------------------------------------------------------------------------------
# 2. the one-touch
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize('direction,barrier', [('Down', CMP_SPOT * 1.001), ('Up', CMP_SPOT * 0.999)],
                         ids=['down', 'up'])
def test_a_one_touch_reads_its_crossing_on_the_composite_spot(direction, barrier):
    """A barrier placed just BEYOND the composite spot and far short of the local one has already
    been touched, so the deal is a certain claim on its cash payoff and worth exactly the discounted
    nominal - no vol, no smile, no closed form in it. On main the crossing test reads the local spot
    of 100: the Down arm never touches a barrier near 80 and misses by 47.6%, and the Up arm sits
    beyond the same barrier for the wrong reason.

    There is no no-touch deal to complete the partition with; this is the same statement read at the
    end where the touch probability is one."""
    out = _run(_job([_one_touch('OT', direction, barrier)], _factors()))
    ref = CASH * math.exp(-R_EUR * T)
    assert abs(_mtm(out, 'OT') - ref) / ref < 1e-12, (_mtm(out, 'OT'), ref)


@pytest.mark.parametrize('direction,barrier,up', [('Up', UP, True), ('Down', DOWN, False)],
                         ids=['up', 'down'])
def test_a_live_one_touch_is_the_textbook_closed_form_on_the_composite(direction, barrier, up):
    """A live barrier against the textbook `erfc` pair evaluated on the composite spot, the
    composite carry and the composite vol read at the TRANSLATED barrier - the whole composite
    spelling in one number. Measured worst 1.6e-15 relative."""
    out = _run(_job([_one_touch('OT', direction, barrier)], _factors()))
    ref = _one_touch_closed(barrier, up, CMP_SPOT, R_EUR - Q_EQ, _compo_sigma(barrier), R_EUR)
    assert abs(_mtm(out, 'OT') - ref) / ref < 1e-9, (_mtm(out, 'OT'), ref)


# --------------------------------------------------------------------------------------------
# 3. the limit: an fx leg identically one is the Standard deal
# --------------------------------------------------------------------------------------------
#: the limit world settles in USD, where the deal's own scale is the local spot of 100
STRIKE_L, UP_L, DOWN_L = 100.0, 115.0, 85.0

LIMIT = [('barrier-up', _barrier('C', 'Up_And_Out', UP_L, strike=STRIKE_L)),
         ('barrier-down', _barrier('C', 'Down_And_In', DOWN_L, 'Put', strike=STRIKE_L)),
         ('one-touch', _one_touch('C', 'Up', UP_L)),
         ('asian', _asian('C', strike=STRIKE_L))]


@pytest.mark.parametrize('name,deal', LIMIT, ids=[n for n, _ in LIMIT])
def test_a_composite_on_a_unit_fx_leg_prices_the_standard_deal(name, deal):
    """All three pricers: with the payoff currency's cross identically one, its curve the local
    one's and neither fx vol nor correlation authored, every composite quantity is a multiplication
    by one or an addition of zero and the deal must reprice its Standard twin. The composite vol
    passes through `sqrt(v*v + 0 + 0)`, the one step that is not the identity by construction;
    measured equal to the bit on every arm."""
    compo = dict(deal, Reference='CMP', Payoff_Currency='USDX', Discount_Rate='USDX',
                 Payoff_Type='Compo')
    plain = dict(deal, Reference='STD', Payoff_Currency='USD', Discount_Rate='USD')
    plain.pop('Payoff_Type', None)
    out = _run(_job([compo, plain], _one_factors()))
    assert _mtm(out, 'CMP') == _mtm(out, 'STD'), (name, _mtm(out, 'CMP'), _mtm(out, 'STD'))


# --------------------------------------------------------------------------------------------
# 4. the discrete Asian
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize('option_type', ['Call', 'Put'])
def test_the_composite_asian_moment_matches_on_the_composite(option_type):
    """The future half of the average against the same two-moment match recomputed here, on the
    composite spot, the composite carry per sample and the composite vol - every sample's forward
    is `S*X`'s. Measured equal to the bit on both option types."""
    factors = _factors()
    out = _run(_job([_asian('AS', option_type)], factors))
    taus = [(d - BASE).days / 365.0 for d in SAMPLES]
    forward, sd = _moment_matched([0.25] * 4, taus, R_EUR - Q_EQ, _compo_sigma(STRIKE), CMP_SPOT)
    ref = UNITS * _black(forward, STRIKE, sd, option_type == 'Call') * math.exp(-R_EUR * T)
    assert abs(_mtm(out, 'AS') - ref) / abs(ref) < 1e-9, (_mtm(out, 'AS'), ref)


# --------------------------------------------------------------------------------------------
# 5. the monitored path: the crossing is decided on the composite
# --------------------------------------------------------------------------------------------
CMC = {'Object': 'CreditMonteCarlo', 'Time_grid': '0d 12m(1m)', 'Batch_Size': 2048,
       'Simulation_Batches': 1, 'Deflation_Interest_Rate': 'EUR'}
MC_BARRIER = 65.0                   # below the composite spot of 80, far below the local 100


def test_the_monitored_path_crosses_on_the_composite_not_the_local_spot():
    """A credit Monte Carlo whose equity walks at 1% and whose fx walks at 40%: the LOCAL spot of
    100 cannot reach a barrier of 65 on any path, and the composite spot of 80 reaches it on most.
    A one-touch paying at expiry reports `nominal * touched` on its last row, so the mean of that
    row IS the mean touch probability the monitored path decided.

    Measured 0.5435 on this tree against 0.0000 when the crossing reads the local spot, at 2048
    paths and seed 1."""
    factors = _factors(skew=0.0, eq_vol=0.01, fx_vol=0.40)
    models = {'GBMAssetPriceModel.EQ': {'Vol': 0.01, 'Drift': 0.0},
              'GBMAssetPriceModel.EUR': {'Vol': 0.40, 'Drift': 0.0}}
    out = _run(_job([_one_touch('OT', 'Down', MC_BARRIER)], factors, calc=CMC, models=models))
    profile = out['Results']['mtm']
    touched = float(np.asarray(profile.iloc[-1].values, dtype=float).mean()) / CASH
    assert touched > 0.2, touched
    local = _one_touch_closed(MC_BARRIER, False, SPOT, R_USD - Q_EQ, 0.01, R_EUR)
    assert local / CASH < 1e-6, local
