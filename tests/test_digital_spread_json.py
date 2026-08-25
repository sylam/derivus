"""Relative_Digital_Spread end to end, through the JSON contract and nothing else.

A digital priced off one vol at its own strike misses the smile: the true price is the strike
derivative of the CALL, so it carries a `-vega * dVol/dK` term the single-vol closed form cannot
see. With the **Relative_Digital_Spread** Valuation Configuration option set, the engine prices
the digital as a call/put spread of half-width `eps * Strike`, each leg reading the surface at its
OWN strike - the smile term arrives as the finite difference the spread takes.

The same replication prices digital caplets and floorlets through `CFFloatingInterestListDeal`,
where the option is **Digital_Spread** and the half-width is ABSOLUTE in rate - the convention a
zero or negative rates strike requires - with the `call_or_put` factor orienting the floorlet's
put spread.

THE ORACLES ARE EXACT, not O(eps^2) approximations: every skewed surface here is authored on
COLLINEAR moneyness nodes, so any interpolation reproduces the same line and the expected value is
the two-leg Black difference computed below with the same vols the engine must read. The other
half of each statement is the two degenerate anchors - the option absent is the single-vol closed
form (regression), and the spread deal equals the two-vanilla portfolio the spread replicates,
priced as ordinary EuropeanOption deals by the same document.
"""
import io
import json
import logging
import math
import os
import sys

# reference-derivus shadow-import guard (MEMORY): pin the package under test to THIS repo.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

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
        'VolatilityGrid.EQ': _surface(VOL_ATM, skew)})


BINARY = {'Object': 'EquityBinaryOption', 'Reference': 'BIN', 'Currency': 'USD',
          'Payoff_Currency': 'USD', 'Equity': 'EQ', 'Dividends': 'EQ', 'Discount_Rate': 'USD',
          'Equity_Volatility': 'EQ', 'Buy_Sell': 'Buy', 'Option_Type': 'Call',
          'Strike_Price': STRIKE, 'Cash_Payoff': CASH, 'Expiry_Date': EXPIRY,
          'Settlement_Date': EXPIRY}


def _vanilla(ref, strike, buy_sell, units):
    return {'Object': 'EquityOptionDeal', 'Reference': ref, 'Currency': 'USD',
            'Payoff_Currency': 'USD', 'Equity': 'EQ', 'Dividends': 'EQ', 'Discount_Rate': 'USD',
            'Equity_Volatility': 'EQ', 'Buy_Sell': buy_sell, 'Option_Type': 'Call',
            'Strike_Price': strike, 'Units': units, 'Expiry_Date': EXPIRY}


def _job(deals, factors, valuation=None):
    return {'Calc': {
        'Calculation': {'Object': 'BaseValuation', 'Base_Date': BASE, 'Currency': 'USD',
                        'MCMC_Simulations': 1, 'Random_Seed': 1},
        'Deals': {'Tag_Titles': '', 'Reference': 'digital',
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


def _black(fwd, strike, vol):
    """Undiscounted Black call, the exact expression the engine's legs evaluate."""
    sd = vol * math.sqrt(T)
    d1 = math.log(fwd / strike) / sd + 0.5 * sd
    return fwd * _ndtr(d1) - strike * _ndtr(d1 - sd)


def _eq_sigma(strike):
    return VOL_ATM + SKEW * (SPOT / strike - 1.0)


EQ_FWD = SPOT * math.exp((R_USD - Q_EQ) * T_FWD)
SPREAD_ON = {'EquityBinaryOption': {'Relative_Digital_Spread': EPS}}


def _digital_closed(fwd, strike, vol, rate):
    sd = vol * math.sqrt(T)
    d2 = math.log(fwd / strike) / sd - 0.5 * sd
    return CASH * math.exp(-rate * T) * _ndtr(d2)


def _spread_closed(fwd, strike, sigma_of, rate):
    lo, hi = strike * (1.0 - EPS), strike * (1.0 + EPS)
    return CASH * math.exp(-rate * T) * (
        _black(fwd, lo, sigma_of(lo)) - _black(fwd, hi, sigma_of(hi))) / (2.0 * EPS * strike)


def test_the_default_is_the_single_vol_closed_form():
    """The regression anchor: no Valuation Configuration entry, and the digital is the closed form
    at the deal's own moneyness - the option defaulting to 0.0 IS the old pricer."""
    out, _ = _run(_job([BINARY], _equity_factors()))
    expected = _digital_closed(EQ_FWD, STRIKE, _eq_sigma(STRIKE), R_USD)
    assert abs(_mtm(out, 'BIN') - expected) / expected < 1e-9, (_mtm(out, 'BIN'), expected)


def test_a_spread_digital_is_the_two_vanillas_it_replicates():
    """The replication identity, interpolation-agnostic: the SAME document prices the binary with
    the spread on and the two EuropeanOption vanillas the spread is made of - long the low strike,
    short the high, `Cash_Payoff / (2 eps K)` units each. Both sides query the same surface at the
    same strikes, so they must agree to float precision, whatever the smile."""
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
    picks up. The oracle is the two-leg Black difference at the line's own vols - exact because
    the surface nodes are collinear.

    MEASURED: 670.395 against the closed form's 478.856 - the smile term is +40% of this
    document's digital, which is what makes the anchors decisive. MUTATION: both legs read at the
    CENTER moneyness prices 478.882 (-28.6% against the oracle, the closed form plus the spread's
    own curvature) - this gate, the replication identity and the FX twin all die on it, and the
    flat-surface compo gate correctly survives.

    The DEBUG organ pins WHAT the pricer read: one line per leg, and the low strike must read the
    HIGHER vol - the smile pickup is the subject, so the log states it directly.
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


def test_an_fx_binary_spread_reads_the_fx_smile():
    """The FX twin through its own moneyness convention (forward / strike, `use_forward`): the
    same exact two-leg oracle on the collinear FX smile."""
    deal = {'Object': 'FXBinaryOption', 'Reference': 'FXB', 'Currency': 'USD',
            'Underlying_Currency': 'EUR', 'Discount_Rate': 'USD', 'FX_Volatility': 'EUR.USD',
            'Buy_Sell': 'Buy', 'Option_Type': 'Call', 'Strike_Price': FX_SPOT,
            'Cash_Payoff': CASH, 'Expiry_Date': EXPIRY}
    factors = dict(RATES, **{'FXVol.EUR.USD': _surface(FX_ATM, FX_SKEW)})
    fwd = FX_SPOT * math.exp((R_USD - R_EUR) * T)
    sigma_of = lambda k: FX_ATM + FX_SKEW * (fwd / k - 1.0)
    out, _ = _run(_job([deal], factors,
                       valuation={'FXBinaryOption': {'Relative_Digital_Spread': EPS}}))
    expected = _spread_closed(fwd, FX_SPOT, sigma_of, R_USD)
    assert abs(_mtm(out, 'FXB') - expected) / expected < 1e-9, (_mtm(out, 'FXB'), expected)


def test_a_compo_binary_spread_composes_each_leg():
    """A compo digital under the spread: the underlying is S*X, the strike a payoff-currency
    quantity, and each leg's vol is the COMPO composition of its own strike's read. Flat surfaces
    make the oracle exact and keep the compo conventions - the sorted-pair correlation sign flip
    among them - the only thing the gate can fail on."""
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
    statement is made on top of it."""
    out, _ = _run(_job([_cap_deal('CAP', True), _cap_deal('FLR', False)], _cap_factors(0.0)))
    for ref, cp in (('CAP', 1.0), ('FLR', -1.0)):
        expected = _digital_rate_closed(cp, 0.0)
        assert abs(_mtm(out, ref) - expected) / expected < 1e-7, (ref, _mtm(out, ref), expected)


def test_a_digital_floorlet_spread_is_its_put_spread():
    """Replication with the payoff's own sign, flat surface: a digital floorlet is a PUT spread,
    `(P(K+eps) - P(K-eps)) / 2 eps`, and the caplet its call spread - `Digital_Spread` is an
    ABSOLUTE half-width in rate, the only convention that survives a zero or negative rates
    strike. Flat vols make the oracle pure replication arithmetic, so the two things this gate
    can fail on are the sign and the width - which are exactly the two defects the migration
    introduced here: the ported branch dropped the `call_or_put` factor and turned the width
    relative, and no gate existed to see either.

    MUTATION: the `call_or_put` factor removed reads the floorlet at -12894.06 against +12894.06
    - the exact sign flip every digital floorlet priced with before this gate."""
    out, _ = _run(_job([_cap_deal('CAP', True), _cap_deal('FLR', False)], _cap_factors(0.0),
                       valuation=CAP_SPREAD_ON))
    for ref, cp in (('CAP', 1.0), ('FLR', -1.0)):
        expected = _spread_rate(cp, 0.0)
        assert abs(_mtm(out, ref) - expected) / expected < 1e-7, (ref, _mtm(out, ref), expected)


def test_a_caplet_spread_reads_each_leg_at_its_own_cap_vol():
    """The smile statement on the cap surface: collinear nodes in the space's own moneyness
    (m = 100 (K - F)), vol falling in strike, so the digital caplet is worth more than its
    closed form and the exact two-leg oracle says by how much."""
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

    MUTATION: the pre-port routing restored (compound unconditionally, no daily branch) kills
    this gate and the range-accrual one together - Pre_Aggregation silently priced as one
    option on the compound, which is the defect the routing exists to prevent."""
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
    accrual leg - each priced as its absolute call spread."""
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

    MUTATION: the plain rule reads 4.116 against the oracle's 22.868 (-82%) - the negative
    tenor trips the Black guard and the ATM cap collapses to intrinsic."""
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
