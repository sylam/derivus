"""A vol surface's moneyness axis means what its type declares: a Malz `Delta_Surface` names its
pillars in the delta the surface declares, and a `Relative_Forward` axis is (K - F)/F as a fraction.

`Premium_Adjusted` says which: `Yes` the premium-adjusted forward delta (K/F)N(d2), whose
delta-neutral straddle sits at K = F exp(-sigma^2 T/2), `No` the plain forward delta N(d1), whose
straddle is a true 0.5 at K = F exp(sigma^2 T/2). An `FXVol` declares the first by default, a pair's
premium being paid in the foreign currency, and a `CommodityPriceVol` the second.

THE ORACLE IS AN IDENTITY: under either convention the strike a pillar's delta names reprices at
that pillar's vol. The smile is quoted at two expiries and every deal expires on the first, so no
expiry interpolation stands between; 5e-4 is the log-moneyness grid's, refined to 1e-4 vol at its
midpoints. The smile is not flat - a flat one hides any placement - and the 25 delta call's two
strikes sit 0.039 apart in log(F/K), where the other convention reads 1.2 and 0.8 vol points off.
DEGENERATE BECAUSE the identity is in the placement alone: the commodity's forward is its curve's
100, the pair's two rates are equal so its forward is its spot, and the rate only discounts.
"""
import json
import os
import sys

# reference-derivus shadow-import guard (MEMORY): pin the package under test to THIS repo.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri

import derivus
from derivus import utils
from derivus.config import CustomJsonEncoder

BASE = pd.Timestamp('2024-06-28')
EXPIRY = BASE + pd.DateOffset(days=182)
T, RATE, FORWARD, SPOT, SIGMA = 182 / 365, 0.03, 100.0, 1.25, 0.40
#: (delta, vol): the 10 and 25 delta calls, the ATM label and the 25 and 10 delta puts
SMILE = [(0.1, 0.46), (0.25, SIGMA), (0.5, 0.35), (-0.25, 0.33), (-0.1, 0.36)]
#: the strike's log(F/K) where N(d1) = 0.25 at the pillar's vol, and where (K/F)N(d2) = 0.25
PLAIN_X = ndtri(0.25) * SIGMA * np.sqrt(T) - 0.5 * SIGMA * SIGMA * T
ADJUSTED_X = brentq(lambda x: np.exp(-x) * ndtr(
    (x - 0.5 * SIGMA * SIGMA * T) / (SIGMA * np.sqrt(T))) - 0.25, -1.5, 0.5 * 0.35 ** 2 * T)
#: between the 25 delta call's 0.40 and the ATM's 0.35 a plain wing is linear in N(d1), so the
#: strike where N(d1) = 0.375 at 0.375 reads 0.375 - if the ATM node sits at a true 0.5
MID_SIGMA = 0.375
MID_X = ndtri(0.375) * MID_SIGMA * np.sqrt(T) - 0.5 * MID_SIGMA * MID_SIGMA * T


def flat(rate):
    return utils.Curve([], [[0.0, rate], [5.0, rate]])


def malz(**stated):
    deltas = [[d, t, v] for t in (T, 2 * T) for d, v in SMILE]
    return dict({'Surface_Type': 'Malz', 'Moneyness_Rule': 'Sticky_Moneyness',
                 'Delta_Surface': utils.Curve([], deltas)}, **stated)


def commodity(surface, strike):
    """`(Price Factors, deal)`: a one-fixing energy option, Black on the WIDGET forward."""
    factors = {
        'ReferencePrice.WIDGET': {
            'ForwardPrice': 'WIDGET', 'Fixing_Curve': utils.Curve([], [[4e4, 4e4], [6e4, 6e4]])},
        'ForwardPrice.WIDGET': {
            'Currency': 'USD', 'Curve': utils.Curve([], [[4e4, FORWARD], [6e4, FORWARD]])},
        'ForwardPriceSample.WIDGET': {
            'Offset': 0, 'Holiday_Calendar': '', 'Sampling_Convention': 'ForwardPriceSampleDaily'},
        'CommodityPriceVol.WIDGET': surface}
    deal = {
        'Object': 'EnergySingleOption', 'Reference': 'OPT', 'Currency': 'USD',
        'Discount_Rate': 'USD', 'Sampling_Type': 'WIDGET', 'Reference_Type': 'WIDGET',
        'Reference_Volatility': 'WIDGET', 'Buy_Sell': 'Buy', 'Option_Type': 'Call',
        'Settlement_Date': EXPIRY, 'Period_Start': EXPIRY, 'Period_End': EXPIRY,
        'FX_Period_Start': EXPIRY, 'FX_Period_End': EXPIRY, 'Strike': strike,
        'Realized_Average': 0.0, 'FX_Realized_Average': 0.0, 'Volume': 1.0}
    return factors, deal


def fx(surface, strike):
    """`(Price Factors, deal)`: a EUR call on a USD book, the two rates equal."""
    factors = {
        'FxRate.EUR': {'Domestic_Currency': None, 'Interest_Rate': 'EUR', 'Spot': SPOT},
        'InterestRate.EUR': {
            'Currency': 'EUR', 'Day_Count': 'ACT_365', 'Sub_Type': None, 'Curve': flat(RATE)},
        'FXVol.EUR.USD': surface}
    deal = {
        'Object': 'FXOptionDeal', 'Reference': 'OPT', 'Currency': 'USD',
        'Underlying_Currency': 'EUR', 'Discount_Rate': 'USD', 'FX_Volatility': 'EUR.USD',
        'Buy_Sell': 'Buy', 'Option_Type': 'Call', 'Expiry_Date': EXPIRY, 'Strike_Price': strike,
        'Underlying_Amount': 1.0}
    return factors, deal


def value(world, surface, strike):
    """The deal's base valuation through the JSON contract."""
    factors, deal = world(surface, strike)
    factors.update({
        'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Spot': 1.0},
        'InterestRate.USD': {
            'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None, 'Curve': flat(RATE)}})
    job = {'Calc': {
        'Calculation': {'Object': 'BaseValuation', 'Base_Date': BASE, 'Currency': 'USD'},
        'Deals': {'Reference': 'delta', 'Deals': {'Children': [{'Instrument': {'.Deal': deal}}]}},
        'MergeMarketData': {'MarketDataFile': '', 'ExplicitMarketData': {
            'System Parameters': {'Base_Currency': 'USD', 'Base_Date': BASE},
            'Valuation Configuration': {}, 'Price Factors': factors}}}}
    context = derivus.Context()
    context.load_json((json.dumps(job, cls=CustomJsonEncoder), 'delta'))
    _, output = context.run_job()
    rows = output['Results']['mtm']
    row = rows[rows['Reference'] == 'OPT']['Value']
    assert len(row) == 1, 'the deal left no row in Results: it was skipped, not priced'
    return float(row.iloc[0])


def implied_vol(world, surface, forward, x):
    """The Black vol the deal struck at log(F/K) = `x` prices at."""
    strike = forward * np.exp(-x)
    priced = value(world, surface, strike)

    def black(sigma):
        d1 = (x + 0.5 * sigma * sigma * T) / (sigma * np.sqrt(T))
        return np.exp(-RATE * T) * (forward * ndtr(d1) - strike * ndtr(d1 - sigma * np.sqrt(T)))
    return brentq(lambda sigma: black(sigma) - priced, 1e-3, 2.0, xtol=1e-12)


def test_a_commodity_delta_surface_is_placed_by_a_plain_delta_unless_it_states_otherwise():
    """A `CommodityPriceVol` stating no `Premium_Adjusted` reads its pillars as plain forward
    deltas: struck where N(d1) = 0.25 at 0.40, the option reprices at the 25 delta call's 0.40.

    Killing mutations: the commodity default flipped to `Yes`, or the solve ignoring the field -
    the premium-adjusted placement reads 0.4121 at that strike.
    """
    assert implied_vol(commodity, malz(), FORWARD, PLAIN_X) == pytest.approx(SIGMA, abs=5e-4)


def test_a_commodity_delta_surface_stating_a_premium_adjusted_delta_is_placed_by_it():
    """The same surface stating `Premium_Adjusted: Yes`, struck where (K/F)N(d2) = 0.25 at 0.40 on
    the call side, reprices at 0.40: the stated value is read, not the class's.

    Killing mutation: `Factor2D.premium_adjusted` answering the class's declaration whatever the
    block states - the plain placement reads 0.3919 at that strike.
    """
    surface = malz(Premium_Adjusted='Yes')
    assert implied_vol(commodity, surface, FORWARD, ADJUSTED_X) == pytest.approx(SIGMA, abs=5e-4)


def test_a_plain_atm_node_sits_at_a_true_half_delta():
    """Between the 25 delta call and the ATM pillar a plain wing is linear in N(d1), so the strike
    where N(d1) = 0.375 at 0.375 reprices at 0.375 - which holds only with the ATM node at a true
    0.5; at a pillar's own strike the clamp would hide a misplaced one.

    Killing mutation: the plain ATM node placed at the premium-adjusted 0.5 exp(-sigma^2 T/2) -
    the read there moves to 0.3736.
    """
    assert implied_vol(commodity, malz(), FORWARD, MID_X) == pytest.approx(MID_SIGMA, abs=5e-4)


RELATIVE_FORWARD = {'Surface_Type': 'Relative_Forward', 'Surface': utils.Curve([], [
    [m, t, v] for t in (T, 2 * T) for m, v in ((-0.4, 0.5), (0.0, 0.35), (0.4, 0.5))])}


def test_a_relative_forward_axis_is_a_fraction():
    """A `Relative_Forward` surface is read at (K - F)/F as a fraction: struck at -0.2, the option
    reprices at the smile's 0.425 there, halfway between the ATM and the -0.4 node.

    Killing mutation: the read at (K - F)/F in percent, which lands on the clamped wing's 0.5.
    """
    x = -np.log(0.8)  # (K - F)/F = -0.2
    assert implied_vol(commodity, RELATIVE_FORWARD, FORWARD, x) == pytest.approx(0.425, abs=1e-6)


def seasoned(surface, strike):
    """`(Price Factors, deal)`: the energy option averaging daily from two weeks before the base
    date to two weeks after it, its realised average the forward's 100."""
    factors, deal = commodity(surface, strike)
    deal.update({'Period_Start': BASE - pd.DateOffset(days=14), 'Period_End': BASE + pd.DateOffset(days=14),
                 'FX_Period_Start': BASE - pd.DateOffset(days=14), 'FX_Period_End': BASE + pd.DateOffset(days=14),
                 'Settlement_Date': BASE + pd.DateOffset(days=16), 'Realized_Average': FORWARD})
    return factors, deal


def test_a_seasoned_asian_reads_its_smile_at_the_remaining_averages_strike():
    """Half-way through its averaging, with the realised average at the forward and the strike at
    the forward, the remaining average's own strike is the forward: the option on the smile prices
    as the option on a surface flat at the smile's ATM vol, to the bit of the two-moment arithmetic.

    Killing mutation: the smile read at the strike net of the realised average against the full
    forward, -0.5 here, which lands on the clamped wing's 0.5 against the ATM 0.35.
    """
    flat = {'Surface_Type': 'Relative_Forward', 'Surface': utils.Curve([], [
        [m, t, 0.35] for t in (T, 2 * T) for m in (-0.4, 0.0, 0.4)])}
    on_smile, on_flat = (value(seasoned, surface, FORWARD) for surface in (RELATIVE_FORWARD, flat))
    assert on_smile == pytest.approx(on_flat, rel=1e-12)


def test_an_fx_delta_surface_keeps_the_premium_adjusted_delta():
    """An `FXVol` stating no `Premium_Adjusted` reads its pillars as premium-adjusted forward
    deltas: an FX option struck where (K/F)N(d2) = 0.25 at 0.40 reprices at 0.40.

    Killing mutation: the FX default flipped to `No` - the plain placement reads 0.3919 there.
    """
    assert implied_vol(fx, malz(), SPOT, ADJUSTED_X) == pytest.approx(SIGMA, abs=5e-4)
