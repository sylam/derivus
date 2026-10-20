"""Relative_Digital_Spread end to end, through the JSON contract and nothing else.

A digital is the strike derivative of the CALL, so it carries a `-vega * dVol/dK` term the
single-vol closed form cannot see. With `Relative_Digital_Spread` set, the engine prices it as a
call/put spread of half-width `eps * Strike`, each leg reading the surface at its OWN strike.
`CFFloatingInterestListDeal` takes the same replication under `Digital_Spread`, where the
half-width is ABSOLUTE in rate - the convention a zero or negative strike requires - with the
`call_or_put` factor orienting the floorlet's put spread. `EquityBarrierBinaryOption` takes it as
the spread of the barrier calls at the two strikes, each walking its own strike's strip and
integrating its terminal step, its parity and already-hit legs the European spread.

THE ORACLES ARE EXACT, not O(eps^2): every skewed surface here is authored on COLLINEAR moneyness
nodes, so any interpolation reproduces the same line and the expected value is the two-leg Black
difference computed below. Each statement's other half is its two degenerate anchors - the option
absent is the single-vol closed form, and the spread deal equals the two-vanilla portfolio it
replicates, priced as ordinary EuropeanOption deals by the same document.
"""
import io
import json
import logging
import math
import os
import sys

# reference-derivus shadow-import guard (MEMORY): pin the package under test to THIS repo.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch

import derivus
from derivus import utils
from derivus.config import CustomJsonEncoder

BASE = pd.Timestamp('2024-06-28')
EXPIRY = BASE + pd.DateOffset(years=1)
T = (EXPIRY - BASE).days / 365.0
# the equity forward carries to expiry + 2 BUSINESS days (`option_date_info`'s
# Forward_Settlement); vol tenor and discounting stay at expiry
T_FWD = ((EXPIRY + pd.tseries.offsets.BDay(2)) - BASE).days / 365.0
SPOT, STRIKE, CASH = 100.0, 100.0, 1000.0
R_USD, R_EUR, Q_EQ = 0.04, 0.02, 0.01
VOL_ATM, SKEW = 0.25, 0.5           # sigma(m) = VOL_ATM + SKEW * (m - 1), m = spot / strike
FX_SPOT, FX_ATM, FX_SKEW = 1.25, 0.10, 0.3
FX_SIGMA, CORR = 0.15, 0.35         # the compo document's flat fx vol and EUR.USD correlation
EPS = 0.01

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


def _equity_factors(skew=SKEW):
    return dict(RATES, **{
        'EquityPrice.EQ': {'Spot': SPOT, 'Currency': 'USD', 'Interest_Rate': 'USD', 'Issuer': '',
                           'Respect_Default': 'No', 'Jump_Level': 0.0},
        'DividendRate.EQ': {'Currency': 'USD', 'Curve': utils.Curve([], [[0.0, Q_EQ], [5.0, Q_EQ]])},
        'EquityPriceVol.EQ': _surface(VOL_ATM, skew)})


BINARY = {'Object': 'EquityBinaryOption', 'Reference': 'BIN', 'Currency': 'USD',
          'Payoff_Currency': 'USD', 'Equity': 'EQ', 'Dividends': 'EQ', 'Discount_Rate': 'USD',
          'Equity_Volatility': 'EQ', 'Buy_Sell': 'Buy', 'Option_Type': 'Call',
          'Strike_Price': STRIKE, 'Payoff': CASH, 'Expiry_Date': EXPIRY,
          'Settlement_Date': EXPIRY}


def _vanilla(ref, strike, buy_sell, units):
    return {'Object': 'EquityOptionDeal', 'Reference': ref, 'Currency': 'USD',
            'Payoff_Currency': 'USD', 'Equity': 'EQ', 'Dividends': 'EQ', 'Discount_Rate': 'USD',
            'Equity_Volatility': 'EQ', 'Buy_Sell': buy_sell, 'Option_Type': 'Call',
            'Strike_Price': strike, 'Units': units, 'Expiry_Date': EXPIRY}


def _job(deals, factors, valuation=None, **calculation):
    return {'Calc': {
        'Calculation': dict({'Object': 'BaseValuation', 'Base_Date': BASE, 'Currency': 'USD',
                             'MCMC_Simulations': 1}, **calculation),
        'Deals': {'Reference': 'digital',
                  'Deals': {'Children': [{'Instrument': {'.Deal': d}} for d in deals]}},
        'MergeMarketData': {'MarketDataFile': '', 'ExplicitMarketData': {
            'System Parameters': {'Base_Currency': 'USD', 'Base_Date': BASE},
            'Valuation Configuration': valuation or {},
            'Price Factors': factors}}}}


def _run(job, debug=False):
    buf, root = io.StringIO(), logging.getLogger()
    handler, old = logging.StreamHandler(buf), logging.getLogger().level
    if debug:
        root.addHandler(handler)
        root.setLevel(logging.DEBUG)
    try:
        cx = derivus.Context()
        cx.load_json((json.dumps(job, cls=CustomJsonEncoder), 'digital'))
        _, out = cx.run_job()
    finally:
        if debug:
            root.removeHandler(handler)
            root.setLevel(old)
    return out, buf.getvalue()


def _mtm(out, ref):
    rows = out['Results']['mtm']
    return float(rows[rows['Reference'] == ref]['Value'].iloc[0])


def _ndtr(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _black(fwd, strike, vol, cap=math.inf):
    """Undiscounted Black call, the exact expression the engine's legs evaluate - paid only up to
    `cap` where one is given, the expectation over `(strike, cap]` a barrier at expiry leaves."""
    sd = vol * math.sqrt(T)
    d1 = math.log(fwd / strike) / sd + 0.5 * sd
    c1 = math.log(fwd / cap) / sd + 0.5 * sd if cap < math.inf else -math.inf
    return fwd * (_ndtr(d1) - _ndtr(c1)) - strike * (_ndtr(d1 - sd) - _ndtr(c1 - sd))


def _eq_sigma(strike):
    return VOL_ATM + SKEW * (SPOT / strike - 1.0)


EQ_FWD = SPOT * math.exp((R_USD - Q_EQ) * T_FWD)
SPREAD_ON = {'EquityBinaryOption': {'Relative_Digital_Spread': EPS}}


def _digital_closed(fwd, strike, vol, rate):
    sd = vol * math.sqrt(T)
    d2 = math.log(fwd / strike) / sd - 0.5 * sd
    return CASH * math.exp(-rate * T) * _ndtr(d2)


def _spread_closed(fwd, strike, sigma_of, rate, cap=math.inf):
    lo, hi = strike * (1.0 - EPS), strike * (1.0 + EPS)
    return CASH * math.exp(-rate * T) * (
        _black(fwd, lo, sigma_of(lo), cap) - _black(fwd, hi, sigma_of(hi), cap)) / (
        2.0 * EPS * strike)


def test_the_default_is_the_single_vol_closed_form():
    """The regression anchor: no Valuation Configuration entry, and the digital is the closed form
    at the deal's own moneyness - the option defaulting to 0.0 IS the old pricer.

    Killing mutation: the closed form's cash and asset payoffs swapped.
    """
    out, _ = _run(_job([BINARY], _equity_factors()))
    expected = _digital_closed(EQ_FWD, STRIKE, _eq_sigma(STRIKE), R_USD)
    assert abs(_mtm(out, 'BIN') - expected) / expected < 1e-9, (_mtm(out, 'BIN'), expected)


def test_a_spread_digital_is_the_two_vanillas_it_replicates():
    """The replication identity, interpolation-agnostic: the SAME document prices the binary with
    the spread on and the two EuropeanOption vanillas the spread is made of - long the low strike,
    short the high, `Payoff / (2 eps K)` units each. Both sides query the same surface at the
    same strikes, so they must agree to float precision, whatever the smile.

    Killing mutation: the cash spread left undivided by the strike.
    """
    units = CASH / (2.0 * EPS * STRIKE)
    deals = [BINARY,
             _vanilla('LO', STRIKE * (1.0 - EPS), 'Buy', units),
             _vanilla('HI', STRIKE * (1.0 + EPS), 'Sell', units)]
    out, _ = _run(_job(deals, _equity_factors()))
    out_spread, _ = _run(_job(deals, _equity_factors(), valuation=SPREAD_ON))
    portfolio = _mtm(out_spread, 'LO') + _mtm(out_spread, 'HI')
    assert abs(_mtm(out_spread, 'BIN') - portfolio) / abs(portfolio) < 1e-9, (
        _mtm(out_spread, 'BIN'), portfolio)
    # and the option OFF in the same portfolio document reproduces the closed form, so the
    # configuration switch is what separates the two prices
    expected_off = _digital_closed(EQ_FWD, STRIKE, _eq_sigma(STRIKE), R_USD)
    assert abs(_mtm(out, 'BIN') - expected_off) / expected_off < 1e-9


def test_the_spread_reads_each_leg_at_its_own_vol():
    """The exact statement, with the smile's sign: on an equity skew (vol FALLING in strike) the
    digital call is worth MORE than the single-vol closed form by the `-vega * dVol/dK` the spread
    picks up. The DEBUG organ pins what the pricer read - one line per leg, low strike at the
    HIGHER vol.

    MEASURED 670.395 against the closed form's 478.856; both legs read at the CENTER moneyness
    price 478.882, -28.6% against the oracle.

    Killing mutation: both legs read at the deal's own moneyness.
    """
    out, log = _run(_job([BINARY], _equity_factors(), valuation=SPREAD_ON), debug=True)
    expected = _spread_closed(EQ_FWD, STRIKE, _eq_sigma, R_USD)
    assert abs(_mtm(out, 'BIN') - expected) / expected < 1e-9, (_mtm(out, 'BIN'), expected)
    closed = _digital_closed(EQ_FWD, STRIKE, _eq_sigma(STRIKE), R_USD)
    assert _mtm(out, 'BIN') > closed * 1.05, 'the skew term must move the price, upward'
    organs = [ln for ln in log.splitlines() if 'DIGITAL_SPREAD' in ln]
    assert len(organs) == 2, organs
    vols = [float(ln.split('vol=')[1]) for ln in organs]
    legs = [ln.split('leg=')[1][0] for ln in organs]
    assert legs == ['-', '+'] and vols[0] > vols[1], organs


def test_an_asset_or_nothing_digital_is_the_vanilla_plus_the_strike_times_the_cash_digital():
    """`Payoff_Style: Asset` pays `Payoff` units of the equity in the money - `(S - K)+` plus `K`
    cash digitals. Closed form that is `F N(w d1)` at the deal's own vol; under the spread, the
    vanilla at its strike plus the strike times the two-leg cash spread, each leg at its own vol -
    exact on the collinear skew, for the call and the put. Every other test here states no style,
    so a cash payoff is what omitting it means.

    Killing mutation: the closed form's cash and asset payoffs swapped.
    """
    discount = CASH * math.exp(-R_USD * T)
    lo, hi = STRIKE * (1.0 - EPS), STRIKE * (1.0 + EPS)
    sd = _eq_sigma(STRIKE) * math.sqrt(T)
    d1 = math.log(EQ_FWD / STRIKE) / sd + 0.5 * sd
    for kind, w in (('Call', 1.0), ('Put', -1.0)):
        def vanilla(strike):
            return _black(EQ_FWD, strike, _eq_sigma(strike)) - (w < 0) * (EQ_FWD - strike)

        spread = w * vanilla(STRIKE) + w * (vanilla(lo) - vanilla(hi)) / (2.0 * EPS)
        for valuation, expected in ((None, discount * EQ_FWD * _ndtr(w * d1)),
                                    (SPREAD_ON, discount * spread)):
            out, _ = _run(_job([dict(BINARY, Payoff_Style='Asset', Option_Type=kind)],
                               _equity_factors(), valuation=valuation))
            assert abs(_mtm(out, 'BIN') - expected) / expected < 1e-9, (
                kind, valuation, _mtm(out, 'BIN'), expected)


def test_an_fx_binary_spread_reads_the_fx_smile():
    """The FX twin through its own moneyness convention (forward / strike, `use_forward`): the
    same exact two-leg oracle on the collinear FX smile.

    Killing mutation: both legs read at the deal's own moneyness.
    """
    deal = {'Object': 'FXBinaryOption', 'Reference': 'FXB', 'Currency': 'USD',
            'Underlying_Currency': 'EUR', 'Discount_Rate': 'USD', 'FX_Volatility': 'EUR.USD',
            'Buy_Sell': 'Buy', 'Option_Type': 'Call', 'Strike_Price': FX_SPOT,
            'Payoff': CASH, 'Expiry_Date': EXPIRY}
    factors = dict(RATES, **{'FXVol.EUR.USD': _surface(FX_ATM, FX_SKEW)})
    fwd = FX_SPOT * math.exp((R_USD - R_EUR) * T)
    sigma_of = lambda k: FX_ATM + FX_SKEW * (fwd / k - 1.0)
    out, _ = _run(_job([deal], factors,
                       valuation={'FXBinaryOption': {'Relative_Digital_Spread': EPS}}))
    expected = _spread_closed(fwd, FX_SPOT, sigma_of, R_USD)
    assert abs(_mtm(out, 'FXB') - expected) / expected < 1e-9, (_mtm(out, 'FXB'), expected)


def test_a_commodity_digital_spread_reads_the_commodity_smile():
    """The commodity twin: a digital on a reference price, its forward read off a flat ForwardPrice
    curve at the delivery date the reference maps the expiry to, through the same moneyness
    convention as the FX one (forward / strike, `use_forward`) on the collinear commodity smile -
    the exact two-leg oracle under the spread, the closed form with the option absent.

    Killing mutation: both legs read at the deal's own moneyness - 347.72 against 518.11.
    """
    oil, strike = 90.0, 95.0
    deal = {'Object': 'CommodityDigitalOption', 'Reference': 'CDG', 'Currency': 'USD',
            'Discount_Rate': 'USD', 'Reference_Type': 'OIL', 'Reference_Volatility': 'OIL',
            'Buy_Sell': 'Buy', 'Option_Type': 'Call', 'Strike_Price': strike, 'Payoff': CASH,
            'Expiry_Date': EXPIRY}
    factors = dict(RATES, **{
        'ReferencePrice.OIL': {'ForwardPrice': 'OIL',
                               'Fixing_Curve': utils.Curve([], [[40000.0, 40000.0], [60000.0, 60000.0]])},
        'ForwardPrice.OIL': {'Currency': 'USD', 'Curve': utils.Curve([], [[40000.0, oil], [60000.0, oil]])},
        'CommodityPriceVol.OIL': _surface(VOL_ATM, SKEW)})
    sigma_of = lambda k: VOL_ATM + SKEW * (oil / k - 1.0)
    out, _ = _run(_job([deal], factors,
                       valuation={'CommodityDigitalOption': {'Relative_Digital_Spread': EPS}}))
    expected = _spread_closed(oil, strike, sigma_of, R_USD)
    assert abs(_mtm(out, 'CDG') - expected) / expected < 1e-9, (_mtm(out, 'CDG'), expected)
    closed, _ = _run(_job([deal], factors))
    anchor = _digital_closed(oil, strike, sigma_of(strike), R_USD)
    assert abs(_mtm(closed, 'CDG') - anchor) / anchor < 1e-9, (_mtm(closed, 'CDG'), anchor)


def test_a_commodity_digital_paid_in_another_currency_is_a_compo_or_a_quanto():
    """A digital on a USD reference price paid in EUR, through the option seam every equity option
    wires its payoff type on. Under `Payoff_Type` Compo the underlying is S*X on the EUR per USD
    forward, the strike a EUR level the local smile is read at translated, and the vol composed
    with the pair's; under Quanto the local forward is carried by `-rho sigma_S sigma_X` and the
    EUR payoff is paid at one. The correlation is read under the name the energy family declares,
    `Correlation.FxRate.USD.EUR/ReferencePrice.OIL.USD`, oriented local to payoff, so +0.35 is
    +0.35, and the deal's reference registers it beside the pair's vol: each document validates
    with no factor missing. The commodity smile is the collinear skew and the
    FX surface flat, so every read is exact: the compo's local vol at the translated strike's
    moneyness, which is the composed forward over the EUR strike, the quanto's at the local
    forward's with the ATM vol in its carry, and under the spread each leg composed at its own.

    Killing mutations: the correlation read with the equity's sorted-pair sign - the compo reads
    514.30 against 504.08; the strike untranslated under the compo, the local smile read at a EUR
    level's moneyness - 486.92 against 504.08; the correlation dropped from the forward's
    conditional row, the compo priced uncorrelated - 509.67 against 504.08; a second type declared
    on `Reference_Volatility` beside the surface, a block no document authors, named missing.
    """
    oil = 90.0
    factors = dict(RATES, **{
        'ReferencePrice.OIL': {'ForwardPrice': 'OIL',
                               'Fixing_Curve': utils.Curve([], [[40000.0, 40000.0], [60000.0, 60000.0]])},
        'ForwardPrice.OIL': {'Currency': 'USD', 'Curve': utils.Curve([], [[40000.0, oil], [60000.0, oil]])},
        'CommodityPriceVol.OIL': _surface(VOL_ATM, SKEW),
        'FXVol.EUR.USD': _surface(FX_SIGMA, 0.0),
        'Correlation.FxRate.USD.EUR/ReferencePrice.OIL.USD': {'Value': CORR}})
    fx_fwd = math.exp((R_EUR - R_USD) * T) / FX_SPOT
    strike_eur, strike_usd = 72.0, 95.0
    deal = {'Object': 'CommodityDigitalOption', 'Reference': 'CDG', 'Currency': 'EUR',
            'Discount_Rate': 'EUR', 'Reference_Type': 'OIL', 'Reference_Volatility': 'OIL',
            'Buy_Sell': 'Buy', 'Option_Type': 'Call', 'Payoff': CASH, 'Expiry_Date': EXPIRY}

    def local_vol(moneyness):
        return VOL_ATM + SKEW * (moneyness - 1.0)

    def composed(strike):
        sigma = local_vol(oil * fx_fwd / strike)
        return math.sqrt(sigma ** 2 + 2.0 * CORR * sigma * FX_SIGMA + FX_SIGMA ** 2)

    compo = _digital_closed(oil * fx_fwd, strike_eur, composed(strike_eur), R_EUR) * FX_SPOT
    quanto = _digital_closed(oil * math.exp(-CORR * VOL_ATM * FX_SIGMA * T), strike_usd,
                             local_vol(oil / strike_usd), R_EUR) * FX_SPOT
    spread = _spread_closed(oil * fx_fwd, strike_eur, composed, R_EUR) * FX_SPOT
    for terms, valuation, expected in (
            ({'Strike_Price': strike_eur, 'Payoff_Type': 'Compo'}, None, compo),
            ({'Strike_Price': strike_usd, 'Payoff_Type': 'Quanto'}, None, quanto),
            ({'Strike_Price': strike_eur, 'Payoff_Type': 'Compo'},
             {'CommodityDigitalOption': {'Relative_Digital_Spread': EPS}}, spread)):
        job = _job([dict(deal, **terms)], factors, valuation=valuation)
        cx = derivus.Context()
        cx.load_json((json.dumps(job, cls=CustomJsonEncoder), 'digital'))
        assert cx.validate()['factors'] == [], (terms, cx.validate()['factors'])
        out, _ = _run(job)
        assert abs(_mtm(out, 'CDG') - expected) / expected < 1e-9, (terms, _mtm(out, 'CDG'), expected)


def test_a_compo_binary_spread_composes_each_leg():
    """A compo digital under the spread: the underlying is S*X, the strike a payoff-currency
    quantity, and each leg's vol is the COMPO composition of its own strike's read. Flat surfaces
    make the oracle exact and keep the compo conventions - the sorted-pair correlation sign flip
    among them - the only thing the gate can fail on.

    Killing mutation: the cash spread left undivided by the strike.
    """
    strike_eur = STRIKE / FX_SPOT
    deal = dict(BINARY, Payoff_Currency='EUR', Payoff_Type='Compo', Discount_Rate='EUR',
                Strike_Price=strike_eur)
    factors = dict(_equity_factors(skew=0.0), **{
        'FXVol.EUR.USD': _surface(FX_SIGMA, 0.0),
        'Correlation.EquityPrice.EQ/FxRate.EUR.USD': {'Value': CORR}})
    rho = -CORR                     # authored on the sorted pair, the deal runs USD -> EUR
    sigma_c = math.sqrt(VOL_ATM ** 2 + 2.0 * rho * VOL_ATM * FX_SIGMA + FX_SIGMA ** 2)
    # equity leg to T_FWD, the fx forward to expiry - the engine's two tenors
    fwd_c = (SPOT * math.exp((R_USD - Q_EQ) * T_FWD) / FX_SPOT) * math.exp((R_EUR - R_USD) * T)
    out, _ = _run(_job([deal], factors, valuation=SPREAD_ON))
    expected = _spread_closed(fwd_c, strike_eur, lambda k: sigma_c, R_EUR) * FX_SPOT
    assert abs(_mtm(out, 'BIN') - expected) / expected < 1e-9, (_mtm(out, 'BIN'), expected)
    closed, _ = _run(_job([deal], factors))
    anchor = _digital_closed(fwd_c, strike_eur, sigma_c, R_EUR) * FX_SPOT
    assert abs(_mtm(closed, 'BIN') - anchor) / anchor < 1e-9, (_mtm(closed, 'BIN'), anchor)


# --------------------------------------------------------------------------------------------
# the barrier binary: the same spread, each leg a barrier call on its own strike's strip
# --------------------------------------------------------------------------------------------
BARRIER = {'Object': 'EquityBarrierBinaryOption', 'Reference': 'BB', 'Currency': 'USD',
           'Payoff_Currency': 'USD', 'Equity': 'EQ', 'Dividends': 'EQ', 'Discount_Rate': 'USD',
           'Equity_Volatility': 'EQ', 'Buy_Sell': 'Buy', 'Option_Type': 'Call',
           'Strike_Price': STRIKE, 'Cash_Payoff': CASH, 'Expiry_Date': EXPIRY,
           'Settlement_Date': EXPIRY, 'Barrier_Type': 'Up_And_Out', 'Barrier_Price': 10.0 * SPOT,
           'Barrier_Dates': [EXPIRY]}
BARRIER_ON = {'EquityBarrierBinaryOption': {'Relative_Digital_Spread': EPS}}
# the barrier's strip carries to the expiry, where the plain binary's forward carries two business
# days past it
FWD_T = SPOT * math.exp((R_USD - Q_EQ) * T)
CASH_T = CASH * math.exp(-R_USD * T)


def test_a_barrier_binary_no_path_reaches_is_the_spread_the_plain_binary_takes():
    """An up-and-out barrier binary observed once, on its expiry, at ten times the spot: each
    leg's one step is integrated from the row's own spot and nothing is sampled, so under the
    spread it is the European spread the plain binary prices, each leg at its own strike's vol -
    the call and a sold put, against the oracle the plain binary is held to, each on its own
    forward: the barrier's strip carries to the expiry, the plain binary's two business days past
    it, 0.057% apart. Off, it is the single-vol closed form.

    MEASURED 670.0161877023206 against the oracle's ...274 and -290.7732514499992 against ...958,
    1.2e-14; the plain binaries 670.395 and -290.394 within 2.3e-14; off, 2.4e-16.

    Killing mutations: the legs read at the centre moneyness, 478.50 (-28.6%); the spread left
    undivided by the strike, 67,001.6; the legs walking the strike's one strip, the European legs
    their own, 478.50.
    """
    put = dict(BARRIER, Reference='BBP', Option_Type='Put', Buy_Sell='Sell')
    plain_put = dict(BINARY, Reference='BINP', Option_Type='Put', Buy_Sell='Sell')
    out, _ = _run(_job([BARRIER, put, BINARY, plain_put], _equity_factors(),
                       valuation=dict(SPREAD_ON, **BARRIER_ON)))
    call = _spread_closed(FWD_T, STRIKE, _eq_sigma, R_USD)
    plain = _spread_closed(EQ_FWD, STRIKE, _eq_sigma, R_USD)
    for ref, expected, tol in (('BB', call, 1e-12), ('BBP', call - CASH_T, 1e-12),
                               ('BIN', plain, 1e-9), ('BINP', plain - CASH_T, 1e-9)):
        assert abs(_mtm(out, ref) - expected) / abs(expected) < tol, (ref, _mtm(out, ref), expected)
    off, _ = _run(_job([BARRIER, put], _equity_factors()))
    closed = _digital_closed(FWD_T, STRIKE, _eq_sigma(STRIKE), R_USD)
    for ref, expected in (('BB', closed), ('BBP', closed - CASH_T)):
        assert abs(_mtm(off, ref) - expected) / abs(expected) < 1e-12, (
            ref, _mtm(off, ref), expected)


def test_a_knock_in_digital_no_path_reaches_is_worth_nothing_under_the_spread():
    """The same deal up-and-in: the in-out-parity vanilla is the European spread the knock-out's
    terminal step is, so with the option on the knock-in is worth nothing, as it is off, the call
    and the put.

    MEASURED 3.4e-12 and 0.0.

    Killing mutation: the parity vanilla left at one vol - the put reads 191.56, the single-vol
    digital less the spread, the skew term; the call's difference is negative and the clamped
    inner mean reads it as nothing, so the put is the half that kills.
    """
    deals = [dict(BARRIER, Reference=kind, Option_Type=kind, Barrier_Type='Up_And_In')
             for kind in ('Call', 'Put')]
    out, _ = _run(_job(deals, _equity_factors(), valuation=BARRIER_ON))
    for kind in ('Call', 'Put'):
        assert abs(_mtm(out, kind)) < 1e-9 * CASH, (kind, _mtm(out, kind))


def test_a_monthly_barrier_binary_no_path_reaches_is_the_european_spread_within_its_noise():
    """Observed monthly, the barrier still out of reach, at 16,384 inner paths: each leg walks the
    twelve fixings on its own strike's strip, which telescopes to that strike's vol at expiry, so
    the knock-out is the European spread and the knock-in nothing, each to the inner Monte Carlo's
    error - a smile read on the last interval alone would leave the walk at the centre's vol and
    the knock-in worth the skew term.

    MEASURED the knock-out call 669.642 and put 291.070 against 670.016 and 290.773, the knock-in
    call 0.374 and put nothing, 3.7e-4 of the payoff at most; at quasi seeds 2 and 3 the knock-out
    call reads 669.926 and 670.146, the knock-ins 0.117 at most. Held to 2e-3 of the payoff.

    Killing mutation: the legs read on the last interval alone, the walk on the strike's own strip -
    the knock-out call 494.85 and the knock-in call 175.16.
    """
    monthly = [BASE + pd.DateOffset(months=m) for m in range(1, 13)]
    deals = [dict(BARRIER, Reference=kind + barrier_type, Option_Type=kind,
                  Barrier_Type=barrier_type, Barrier_Dates=monthly)
             for kind in ('Call', 'Put') for barrier_type in ('Up_And_Out', 'Up_And_In')]
    out, _ = _run(_job(deals, _equity_factors(), valuation=BARRIER_ON, MCMC_Simulations=1 << 14))
    call = _spread_closed(FWD_T, STRIKE, _eq_sigma, R_USD)
    for ref, expected in (('CallUp_And_Out', call), ('PutUp_And_Out', CASH_T - call),
                          ('CallUp_And_In', 0.0), ('PutUp_And_In', 0.0)):
        assert abs(_mtm(out, ref) - expected) < 2e-3 * CASH, (ref, _mtm(out, ref), expected)


def test_a_reachable_barrier_truncates_each_leg_under_its_own_vol():
    """The barrier at 1.15 times the spot, observed on the expiry: each leg is the expectation of
    its call over `(K', H]`, the step's lognormal law at that leg's own vol, and the digital their
    spread - the oracle below, a difference of two truncated Blacks.

    MEASURED 176.11229177810569 against the oracle's ...1424, 4.9e-14.

    Killing mutations: the survival dropped from the legs, 670.016, the unreachable barrier's
    number; both legs truncated at the barrier's standard score under their mean law, 206.87
    (+17.5%).
    """
    cap = 1.15 * SPOT
    out, _ = _run(_job([dict(BARRIER, Barrier_Price=cap)], _equity_factors(), valuation=BARRIER_ON))
    expected = _spread_closed(FWD_T, STRIKE, _eq_sigma, R_USD, cap=cap)
    assert abs(_mtm(out, 'BB') - expected) / expected < 1e-10, (_mtm(out, 'BB'), expected)


def _greeks(out, factor):
    frame = out['Results']['Greeks_First']
    return frame.loc[[i for i in frame.index if str(i[0]).startswith(factor)]].values


def test_a_monitored_barrier_binary_takes_the_skew_and_recomputes_to_the_bit():
    """A down-and-out call digital observed monthly with its barrier at 85, at 16,384 inner
    paths: the spread moves it up by the skew term, a call digital on a surface whose vol falls in
    strike. Its delta and vega are finite, and under `Recompute_Inner_MC` the value and every
    first-order greek are the run without it to the bit - the legs' strips enter the node as its
    arguments, and nothing drawn is stored.

    MEASURED off 429.568, on 575.289 (+33.9%), the equity delta 17.46 and the vega rows between
    -593.5 and +356.9.

    Killing mutations: the walk's strip read from the enclosing scope rather than handed to the
    node - under the recompute the equity delta reads 22.16 and the vega rows nothing; the legs
    walking the strike's one strip, on 429.54 against off.
    """
    deal = dict(BARRIER, Barrier_Type='Down_And_Out', Barrier_Price=85.0,
                Barrier_Dates=[BASE + pd.DateOffset(months=m) for m in range(1, 13)])
    calc = {'MCMC_Simulations': 1 << 14, 'Greeks': 'First'}
    off, _ = _run(_job([deal], _equity_factors(), **calc))
    on, _ = _run(_job([deal], _equity_factors(), valuation=BARRIER_ON, **calc))
    node, _ = _run(_job([deal], _equity_factors(), valuation=BARRIER_ON, Recompute_Inner_MC='Yes',
                        **calc))
    assert _mtm(on, 'BB') > _mtm(off, 'BB') * 1.05, (_mtm(on, 'BB'), _mtm(off, 'BB'))
    for factor in ('EquityPrice.EQ', 'EquityPriceVol.EQ'):
        greeks = _greeks(on, factor)
        assert greeks.size and np.isfinite(greeks).all() and np.abs(greeks).max() > 0, (
            factor, greeks)
    assert _mtm(node, 'BB') == _mtm(on, 'BB')
    assert np.array_equal(node['Results']['Greeks_First'].values,
                          on['Results']['Greeks_First'].values), 'the node moved a greek'


BARRIER_DAY = BASE + pd.DateOffset(months=1)
HIT_LEVEL = 105.0


def _hit_spread(spot, at):
    """The plain binary's spread on one path at `at`, on the hit leg's forward: the European
    spread from the path's spot to expiry, each leg at the surface's vol at its own moneyness,
    flat past the outer nodes; on the expiry row each leg's intrinsic."""
    tau = (EXPIRY - at).days / 365.0
    lo, hi = STRIKE * (1.0 - EPS), STRIKE * (1.0 + EPS)
    if tau <= 0.0:
        return CASH * (max(spot - lo, 0.0) - max(spot - hi, 0.0)) / (2.0 * EPS * STRIKE)
    fwd = spot * math.exp((R_USD - Q_EQ) * tau)

    def leg(k):
        vol = VOL_ATM + SKEW * (min(max(spot / k, 0.8), 1.2) - 1.0)
        sd = vol * math.sqrt(tau)
        d1 = math.log(fwd / k) / sd + 0.5 * sd
        return fwd * _ndtr(d1) - k * _ndtr(d1 - sd)
    return CASH * math.exp(-R_USD * tau) * (leg(lo) - leg(hi)) / (2.0 * EPS * STRIKE)


def _cmc_mtm(deals, valuation=None):
    doc = _job(deals, _equity_factors(), valuation=valuation, Object='CreditMonteCarlo',
               Time_grid='0d 1m(1m)', Batch_Size=256, Simulation_Batches=2, Random_Seed=1,
               Deflation_Interest_Rate='USD', MCMC_Simulations=64)
    market = doc['Calc']['MergeMarketData']['ExplicitMarketData']
    market['Price Models'] = {'GBMAssetPriceModel.EQ': {'Vol': VOL_ATM, 'Drift': 0.0}}
    market['Model Configuration'] = {'.ModelParams': {
        'modeldefaults': {'EquityPrice': 'GBMAssetPriceModel'}, 'modelfilters': {}}}
    cx = derivus.Context()
    cx.load_json((json.dumps(doc, cls=CustomJsonEncoder), 'digital_cmc'))
    return derivus.run_cmc(cx.current_cfg, prec=torch.float64)[1]['Results']['mtm']


def test_an_already_knocked_in_digital_is_the_plain_binarys_spread_on_its_own_path():
    """A credit Monte Carlo on a GBM equity at the surface's ATM vol: an up-and-in call digital
    observed once, a month out, at 105, which some paths cross. On every row from that date, a
    crossed path is worth the plain binary's spread off its own spot - the already-hit leg after
    the date, the parity vanilla on it - and an uncrossed one nothing, the expiry row each leg's
    intrinsic. The spot comes off a share position on the same paths.

    MEASURED 108 of 512 paths crossed, twelve rows from the date, the crossed paths within 2.8e-14
    of the payoff and the uncrossed within 2.7e-14.

    Killing mutations: the already-hit leg left at one vol, a crossed path 0.414 of the payoff
    off; a zero-length terminal step left at the indicator, an uncrossed path above the strike on
    the expiry row worth the whole payoff.
    """
    deal = dict(BARRIER, Barrier_Type='Up_And_In', Barrier_Price=HIT_LEVEL,
                Barrier_Dates=[BARRIER_DAY])
    share = {'Object': 'EquityDeal', 'Reference': 'SHR', 'Currency': 'USD', 'Equity': 'EQ',
             'Buy_Sell': 'Buy', 'Units': 1.0, 'Investment_Horizon': EXPIRY}
    mtm = _cmc_mtm([deal], valuation=BARRIER_ON)
    spot = _cmc_mtm([share])
    assert (spot.iloc[0].values == SPOT).all() and spot.shape == mtm.shape, spot.shape
    crossed = spot.loc[BARRIER_DAY].values > HIT_LEVEL
    assert 0 < crossed.sum() < crossed.size, crossed.sum()
    rows = mtm[mtm.index >= BARRIER_DAY]
    assert len(rows) > 2, 'rows after the date are the already-hit leg'
    worst = 0.0
    for at, row in rows.iterrows():
        for value, s, hit in zip(row.values, spot.loc[at].values, crossed):
            hand = _hit_spread(s, at) if hit else 0.0
            worst = max(worst, abs(value - hand) / CASH)
    assert worst < 1e-10, worst


def test_a_compo_barrier_binary_composes_each_leg_as_the_plain_one_does():
    """The compo twin on the barrier binary, no path reaching its barrier: the underlying is S*X,
    strike and barrier payoff-currency levels on it, and each leg reads the local smile at its own
    translated strike, `K'/F_X(T)`, composed with the pair's vol - the oracle the compo plain
    binary beside it meets on its own forward.

    MEASURED 767.315311875994 against the oracle's ...0102 and the plain binary 767.843 on its
    own, each 2.1e-14.

    Killing mutation: the legs' vols left uncomposed, 813.02 (+6.0%).
    """
    strike_eur = STRIKE / FX_SPOT
    terms = dict(Payoff_Currency='EUR', Payoff_Type='Compo', Discount_Rate='EUR',
                 Strike_Price=strike_eur)
    factors = dict(_equity_factors(), **{
        'FXVol.EUR.USD': _surface(FX_SIGMA, 0.0),
        'Correlation.EquityPrice.EQ/FxRate.EUR.USD': {'Value': CORR}})
    rho = -CORR                     # authored on the sorted pair, the deal runs USD -> EUR
    fx_fwd = math.exp((R_EUR - R_USD) * T) / FX_SPOT

    def composed(k):
        sigma = VOL_ATM + SKEW * (SPOT * fx_fwd / k - 1.0)
        return math.sqrt(sigma ** 2 + 2.0 * rho * sigma * FX_SIGMA + FX_SIGMA ** 2)

    out, _ = _run(_job([dict(BARRIER, Barrier_Price=10.0 * strike_eur, **terms),
                        dict(BINARY, **terms)], factors, valuation=dict(SPREAD_ON, **BARRIER_ON)))
    for ref, fwd, tol in (('BB', FWD_T * fx_fwd, 1e-12),
                          ('BIN', SPOT * math.exp((R_USD - Q_EQ) * T_FWD) * fx_fwd, 1e-9)):
        expected = _spread_closed(fwd, strike_eur, composed, R_EUR) * FX_SPOT
        assert abs(_mtm(out, ref) - expected) / expected < tol, (ref, _mtm(out, ref), expected)


# --------------------------------------------------------------------------------------------
# digital caplets / floorlets: the same spread through the cap surface
# --------------------------------------------------------------------------------------------
CAP_START = BASE + pd.DateOffset(years=1)
CAP_END = CAP_START + pd.DateOffset(months=6)
TAU = (CAP_END - CAP_START).days / 365.0
T_RESET = (CAP_START - BASE).days / 365.0
FWD_RATE = math.expm1(R_USD * (CAP_END - CAP_START).days / 365.0) / TAU
NOTIONAL, PAYOFF_RATE, CAP_STRIKE = 1e6, 0.05, 4.0      # strike in percent, as authored
RATE_EPS = 0.001                                        # ABSOLUTE half-width: 10bp of rate
CAP_ATM, CAP_SLOPE = 0.30, -0.05    # sigma(m) = CAP_ATM + CAP_SLOPE * m, m = 100 * (K - F)
CAP_SPREAD_ON = {'CFFloatingInterestListDeal': {'Digital_Spread': RATE_EPS}}


def _cap_surface(slope):
    """Collinear moneyness nodes again - the cap space reads m = -100 * (forward - strike)."""
    return {'Surface': utils.Curve([], [[m, e, t, CAP_ATM + slope * m]
                                        for m in (-2.0, 0.0, 2.0) for e in (0.02, 3.0)
                                        for t in (0.25, 1.0)])}


def _float_item(start, end, tau, payment, known=0.0):
    return {'Payment_Date': payment, 'Notional': NOTIONAL,
            'Accrual_Start_Date': start, 'Accrual_End_Date': end,
            'Accrual_Day_Count': 'ACT_365', 'Accrual_Year_Fraction': tau,
            'Resets': [[start, start, end, tau, pd.DateOffset(days=1), 'ACT_365', '0D',
                        0.0, 'No', utils.Percent(known)]],
            'Margin': utils.Basis(0.0), 'Fixed_Amount': 0.0,
            'FX_Reset_Date': None, 'Known_FX_Rate': 0.0}


def _cap_deal(ref, is_cap, items=None, averaging='Average_Interest', compounding='None',
              digital=True):
    strike = utils.Percent(CAP_STRIKE)
    props = {'Cap_Multiplier': 1.0 if is_cap else 0.0, 'Cap_Strike': strike,
             'Floor_Multiplier': 0.0 if is_cap else 1.0, 'Floor_Strike': strike}
    if digital:
        props['Digital_Payoff_Rate'] = utils.Percent(100.0 * PAYOFF_RATE)
    return {'Object': 'CFFloatingInterestListDeal', 'Reference': ref, 'Currency': 'USD',
            'Discount_Rate': 'USD', 'Buy_Sell': 'Buy', 'Description': '',
            'Settlement_Date': None, 'Settlement_Amount': 0.0, 'Settlement_Style': 'Physical',
            'Settlement_Amount_Is_Clean': 'Yes', 'Is_Defaultable': 'No', 'Repo_Rate': '',
            'Recovery_Rate': '', 'Survival_Probability': '', 'Investment_Horizon': None,
            'Issuer': '', 'Settlement_Rate': '', 'Forecast_Rate': 'USD',
            'Forecast_Rate_Cap_Volatility': 'CAPVOL', 'Forecast_Rate_Swaption_Volatility': '',
            'Discount_Rate_Cap_Volatility': '', 'Discount_Rate_Swaption_Volatility': '',
            'Rate_Adjustment_Method': 'None', 'Rate_Sticky_Month_End': 'Yes', 'Rate_Offset': 0,
            'Rate_Calendars': None, 'Accrual_Calendars': None,
            'Cashflows': {
                'Compounding_Method': compounding, 'Averaging_Method': averaging,
                'Properties': [props],
                'Items': items or [_float_item(CAP_START, CAP_END, TAU, CAP_END)]}}


def _cap_factors(slope):
    return dict(RATES, **{'InterestRateVol.CAPVOL': _cap_surface(slope)})


def _rate_sigma(strike, slope):
    return CAP_ATM + slope * 100.0 * (strike - FWD_RATE)


def _black_rate(strike, vol, cp):
    sd = vol * math.sqrt(T_RESET)
    d1 = math.log(FWD_RATE / strike) / sd + 0.5 * sd
    return cp * (FWD_RATE * _ndtr(cp * d1) - strike * _ndtr(cp * (d1 - sd)))


CAP_SCALE = NOTIONAL * TAU * math.exp(-R_USD * (CAP_END - BASE).days / 365.0)


def _digital_rate_closed(cp, slope):
    k = 0.01 * CAP_STRIKE
    vol = _rate_sigma(k, slope)
    sd = vol * math.sqrt(T_RESET)
    d2 = math.log(FWD_RATE / k) / sd - 0.5 * sd
    return CAP_SCALE * PAYOFF_RATE * _ndtr(cp * d2)


def _spread_rate(cp, slope):
    k = 0.01 * CAP_STRIKE
    lo, hi = k - RATE_EPS, k + RATE_EPS
    return CAP_SCALE * PAYOFF_RATE * cp * (
        _black_rate(lo, _rate_sigma(lo, slope), cp) -
        _black_rate(hi, _rate_sigma(hi, slope), cp)) / (2.0 * RATE_EPS)


def test_a_digital_caplet_and_floorlet_default_to_the_closed_form():
    """The rates anchors, flat surface, no valuation configuration: `N(d2)` and `N(-d2)` of the
    SIMPLE forward the reset compiles to - which pins the whole convention chain (expm1 forward,
    accrual year fraction, payment-date discounting, the surface's tenor axis) before any spread
    statement is made on top of it.

    Killing mutation: the caplet priced with no digital payoff (`cash_payoff=0`).
    """
    out, _ = _run(_job([_cap_deal('CAP', True), _cap_deal('FLR', False)], _cap_factors(0.0)))
    for ref, cp in (('CAP', 1.0), ('FLR', -1.0)):
        expected = _digital_rate_closed(cp, 0.0)
        assert abs(_mtm(out, ref) - expected) / expected < 1e-7, (ref, _mtm(out, ref), expected)


def test_a_digital_floorlet_spread_is_its_put_spread():
    """Replication with the payoff's own sign, flat surface: a digital floorlet is a PUT spread,
    `(P(K+eps) - P(K-eps)) / 2 eps`, and the caplet its call spread - `Digital_Spread` is an
    ABSOLUTE half-width in rate, the only convention that survives a zero or negative rates
    strike. Flat vols make the oracle pure replication arithmetic, so the gate can only fail on
    the sign or the width.

    Killing mutation: the `call_or_put` factor removed - the floorlet reads -12894.06 against
    +12894.06.
    """
    out, _ = _run(_job([_cap_deal('CAP', True), _cap_deal('FLR', False)], _cap_factors(0.0),
                       valuation=CAP_SPREAD_ON))
    for ref, cp in (('CAP', 1.0), ('FLR', -1.0)):
        expected = _spread_rate(cp, 0.0)
        assert abs(_mtm(out, ref) - expected) / expected < 1e-7, (ref, _mtm(out, ref), expected)


def test_a_caplet_spread_reads_each_leg_at_its_own_cap_vol():
    """The smile statement on the cap surface: collinear nodes in the space's own moneyness
    (m = 100 (K - F)), vol falling in strike, so the digital caplet is worth more than its
    closed form and the exact two-leg oracle says by how much.

    Killing mutation: both legs' cap vols read at the centre strike.
    """
    out, _ = _run(_job([_cap_deal('CAP', True)], _cap_factors(CAP_SLOPE),
                       valuation=CAP_SPREAD_ON))
    expected = _spread_rate(1.0, CAP_SLOPE)
    assert abs(_mtm(out, 'CAP') - expected) / expected < 1e-7, (_mtm(out, 'CAP'), expected)
    assert _mtm(out, 'CAP') > _digital_rate_closed(1.0, CAP_SLOPE)


# --------------------------------------------------------------------------------------------
# aggregated legs: Averaging_Method picks the cap convention around the SAME payoff
# --------------------------------------------------------------------------------------------
def _agg_items(start, n=4, step=7, known=0.0):
    """One item per fixing sharing the period's payment date - the OIS authoring shape
    `compress_no_compounding` merges into a single cashflow owning every reset at weight 1."""
    end = start + pd.DateOffset(days=n * step)
    return [_float_item(start + pd.DateOffset(days=i * step),
                        start + pd.DateOffset(days=(i + 1) * step),
                        step / 365.0, end, known=known) for i in range(n)]


def _bs_rate(fwd, strike, vol, t, cp):
    sd = vol * math.sqrt(t)
    d1 = math.log(fwd / strike) / sd + 0.5 * sd
    return cp * (fwd * _ndtr(cp * d1) - strike * _ndtr(cp * (d1 - sd)))


AGG_N, AGG_STEP = 4, 7
AGG_TAU = AGG_STEP / 365.0
AGG_FWD = math.expm1(R_USD * AGG_STEP / 365.0) / AGG_TAU
AGG_PAY_DAYS = (CAP_START - BASE).days + AGG_N * AGG_STEP
AGG_SCALE = NOTIONAL * math.exp(-R_USD * AGG_PAY_DAYS / 365.0)
K_RATE = 0.01 * CAP_STRIKE


def test_an_aggregated_cap_prices_per_convention():
    """The interaction the two pricers exist for: on the SAME aggregated leg,
    `Pre_Aggregation` caps every reset before compounding - each an optionlet at its own
    end-day expiry, accrual-weighted and summed - while `Post_Aggregation` compounds first and
    prices ONE option on the period rate, vol read at the period END with the averaging-decay
    Black time `t_end - (2/3)(t_end - t_start)`. Flat surface, both oracles exact, and the two
    conventions must separate - capping a sum is not summing the caps.

    Killing mutation: `Pre_Aggregation` compounded like the period pricer (no daily branch), which
    kills the range-accrual gate below too.
    """
    items = _agg_items(CAP_START)
    out, _ = _run(_job(
        [_cap_deal('PRE', True, items=items, averaging='Pre_Aggregation',
                   compounding='OIS', digital=False),
         _cap_deal('POST', True, items=items, averaging='Post_Aggregation',
                   compounding='OIS', digital=False)], _cap_factors(0.0)))
    start_days = (CAP_START - BASE).days
    pre = AGG_SCALE * sum(
        AGG_TAU * _bs_rate(AGG_FWD, K_RATE, CAP_ATM,
                           (start_days + AGG_STEP * (i + 1)) / 365.0, 1.0)
        for i in range(AGG_N))
    tau_a = AGG_N * AGG_STEP / 365.0
    fwd_a = math.expm1(R_USD * AGG_N * AGG_STEP / 365.0) / tau_a
    t_eff = (AGG_PAY_DAYS - (2.0 / 3.0) * AGG_N * AGG_STEP) / 365.0
    post = AGG_SCALE * tau_a * _bs_rate(fwd_a, K_RATE, CAP_ATM, t_eff, 1.0)
    assert abs(_mtm(out, 'PRE') - pre) / pre < 1e-7, (_mtm(out, 'PRE'), pre)
    assert abs(_mtm(out, 'POST') - post) / post < 1e-7, (_mtm(out, 'POST'), post)
    assert abs(pre - post) / post > 0.005, 'the conventions must separate'


def test_a_pre_aggregation_digital_strip_is_a_range_accrual():
    """The payoff helper is SHARED, so the daily pricer composes with the digital spread for
    free: every reset a digital caplet paying `Digital_Payoff_Rate` on its own accrual - a range
    accrual leg - each priced as its absolute call spread.

    Killing mutation: `Pre_Aggregation` compounded like the period pricer (no daily branch).
    """
    out, _ = _run(_job([_cap_deal('RA', True, items=_agg_items(CAP_START),
                                  averaging='Pre_Aggregation', compounding='OIS')],
                       _cap_factors(0.0), valuation=CAP_SPREAD_ON))
    start_days = (CAP_START - BASE).days
    expected = AGG_SCALE * sum(
        AGG_TAU * PAYOFF_RATE * (
            _bs_rate(AGG_FWD, K_RATE - RATE_EPS, CAP_ATM, t_i, 1.0) -
            _bs_rate(AGG_FWD, K_RATE + RATE_EPS, CAP_ATM, t_i, 1.0)) / (2.0 * RATE_EPS)
        for i in range(AGG_N)
        for t_i in [(start_days + AGG_STEP * (i + 1)) / 365.0])
    assert abs(_mtm(out, 'RA') - expected) / expected < 1e-7, (_mtm(out, 'RA'), expected)


def test_an_in_period_post_aggregation_cap_integrates_the_remaining_variance():
    """A period already RUNNING at the valuation date: two resets fixed (known rates), two still
    forward. The naive decay rule `t_end - (2/3)(t_end - t_start)` goes NEGATIVE here and the
    Black guard silently prices intrinsic; the exact in-period integral
    `t_end^3 / (3 (t_end - t_start)^2)` is what the average's remaining variance actually is.

    Killing mutation: the plain decay rule inside the period - 4.116 against the oracle's 22.868.
    """
    known = 4.0                 # ATM: the remaining-variance term IS the price here
    items = _agg_items(BASE - pd.DateOffset(days=14), known=known)
    out, _ = _run(_job([_cap_deal('MID', True, items=items, averaging='Post_Aggregation',
                                  compounding='OIS', digital=False)], _cap_factors(0.0)))
    tau_a = AGG_N * AGG_STEP / 365.0
    fwd_a = math.expm1(2.0 * math.log1p(0.01 * known * AGG_TAU) +
                       2.0 * R_USD * AGG_STEP / 365.0) / tau_a
    t_eff = (14.0 ** 3 / (3.0 * 28.0 ** 2)) / 365.0
    expected = NOTIONAL * math.exp(-R_USD * 14.0 / 365.0) * tau_a * _bs_rate(
        fwd_a, K_RATE, CAP_ATM, t_eff, 1.0)
    assert abs(_mtm(out, 'MID') - expected) / expected < 1e-7, (_mtm(out, 'MID'), expected)
