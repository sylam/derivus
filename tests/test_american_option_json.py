"""`pv_american_option` end to end, through the JSON contract and nothing else.

The census names this pricer `never-called`: every equity option fixture in the repo is European. It
is the Bjerksund-Stensland trigger approximation floored at Black - a SUB-OPTIMAL exercise policy,
so its value is a LOWER bound on the American price, and the gates are bounds rather than a
symmetric tolerance.

THE ORACLE is a Cox-Ross-Rubinstein binomial in numpy at the same carry `b` the document implies,
averaged over `n` and `n+1` steps (CRR's leading error oscillates with the strike node's parity).
Nothing of the engine's is reused, and its convergence is gated: n=3000 against n=8000 agrees to
better than 1e-4 relative.

`b` IS THE DOCUMENT'S OWN and is not `r - q`: `EquityOptionDeal` reads its carry off the equity
forward at `Forward_Settlement`, two business days past expiry, so `b = (r - q) * T_fwd / T`. The
European control reproduces Black at that carry to 6e-5 and at `r - q` only to 1e-3.

MEASURED, eighteen fixtures over three worlds, both option types, strikes 80/100/120 on a spot of
100 at two years:

  * dominance holds everywhere - Black <= engine <= converged binomial - with a tightest margin of
    -9.6e-7 relative, inside the binomial's OWN convergence error, so the upper bound carries that
    slack and the convergence gate is what measures it;
  * worst relative gap to the converged binomial 2.07%, where the early-exercise premium is 13.4%
    of the value - the gate is 3%;
  * the premium the approximation CAPTURES is 82.3% to 99.9% of the oracle's on the nine fixtures
    where it is material - the gate is 75%, and a mutant returning Black reads 0%.

THE TWO DEGENERATE ANCHORS are exact: at `q = 0` the American call IS the European, bit-identical
(the `(b >= r) * Black` arm), and a call past its own exercise trigger marks exactly its intrinsic
(the `(S >= I) * (S - K)` arm, with the `first_knockout` settlement beside it).

DEGENERACY CHECKLIST: every fixture runs a non-zero `r != q`, both option types, and both sides of
the early-exercise decision - the deeply American leg of each world and its near-European twin.
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
EXPIRY = BASE + pd.DateOffset(years=2)
T = (EXPIRY - BASE).days / 365.0
# the equity forward carries to expiry + 2 BUSINESS days (`option_date_info`'s Forward_Settlement);
# vol tenor and discounting stay at expiry, so the implied carry is stretched by T_FWD / T
T_FWD = ((EXPIRY + pd.tseries.offsets.BDay(2)) - BASE).days / 365.0
SPOT, UNITS = 100.0, 10.0

#: (rate, dividend yield, vol). The first is American on the CALL leg, the second on the PUT leg,
#: the third is the first at a lower vol - where the approximation is furthest from the binomial
#: and where a deep call sits past its own trigger.
WORLDS = [(0.05, 0.10, 0.30), (0.08, 0.02, 0.30), (0.05, 0.10, 0.20)]
STRIKES = [80.0, 100.0, 120.0]
CASES = [(w, opt, k) for w in WORLDS for opt in ('Call', 'Put') for k in STRIKES]


def _factors(r, q, vol):
    return {
        'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Spot': 1.0},
        'InterestRate.USD': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                             'Curve': utils.Curve([], [[0.0, r], [10.0, r]])},
        'EquityPrice.EQ': {'Spot': SPOT, 'Currency': 'USD', 'Interest_Rate': 'USD', 'Issuer': '',
                           'Respect_Default': 'No', 'Jump_Level': 0.0},
        'DividendRate.EQ': {'Currency': 'USD', 'Curve': utils.Curve([], [[0.0, q], [10.0, q]])},
        'VolatilityGrid.EQ': {'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness',
                              'Surface': utils.Curve([], [[m, t, vol] for m in (0.5, 1.0, 1.5)
                                                          for t in (0.02, 5.0)])}}


def _deal(strike, option_type, style, ref):
    return {'Object': 'EquityOptionDeal', 'Reference': ref, 'Currency': 'USD',
            'Payoff_Currency': 'USD', 'Equity': 'EQ', 'Dividends': 'EQ', 'Discount_Rate': 'USD',
            'Equity_Volatility': 'EQ', 'Buy_Sell': 'Buy', 'Option_Type': option_type,
            'Strike_Price': strike, 'Units': UNITS, 'Expiry_Date': EXPIRY,
            'Option_Style': style, 'Settlement_Style': 'Cash'}


def _run(deals, factors):
    job = {'Calc': {
        'Calculation': {'Object': 'BaseValuation', 'Base_Date': BASE, 'Currency': 'USD',
                        'MCMC_Simulations': 1, 'Random_Seed': 1},
        'Deals': {'Tag_Titles': '', 'Reference': 'american',
                  'Deals': {'Children': [{'Instrument': {'.Deal': d}} for d in deals]}},
        'MergeMarketData': {'MarketDataFile': '', 'ExplicitMarketData': {
            'System Parameters': {'Base_Currency': 'USD', 'Base_Date': BASE},
            'Valuation Configuration': {}, 'Price Factors': factors}}}}
    cx = derivus.Context()
    cx.load_json((json.dumps(job, cls=CustomJsonEncoder), 'american'))
    _, out = cx.run_job()
    rows = out['Results']['mtm']
    return {ref: float(rows[rows['Reference'] == ref]['Value'].iloc[0])
            for ref in rows['Reference'].unique()}


def _both(strike, option_type, world):
    """`(american, european)` from ONE document, so the two marks share a market by construction."""
    got = _run([_deal(strike, option_type, 'American', 'AM'),
                _deal(strike, option_type, 'European', 'EU')], _factors(*world))
    return got['AM'], got['EU']


# --------------------------------------------------------------------------------------------
# the oracle: a CRR binomial at the document's own carry, engine-free
# --------------------------------------------------------------------------------------------
def _crr(strike, cp, r, b, vol, steps, american):
    dt = T / steps
    u = math.exp(vol * math.sqrt(dt))
    p = (math.exp(b * dt) - 1.0 / u) / (u - 1.0 / u)
    disc = math.exp(-r * dt)
    node = np.arange(steps + 1)
    v = np.maximum(cp * (SPOT * u ** (2.0 * node - steps) - strike), 0.0)
    for i in range(steps - 1, -1, -1):
        v = disc * (p * v[1:] + (1 - p) * v[:-1])
        if american:
            v = np.maximum(v, cp * (SPOT * u ** (2.0 * np.arange(i + 1) - i) - strike))
    return float(v[0])


def _oracle(strike, option_type, world, steps=3000, american=True):
    """`n` and `n+1` averaged - CRR's leading error oscillates with the strike's node parity."""
    r, q, vol = world
    cp = 1.0 if option_type == 'Call' else -1.0
    b = (r - q) * T_FWD / T
    return UNITS * 0.5 * (_crr(strike, cp, r, b, vol, steps, american) +
                          _crr(strike, cp, r, b, vol, steps + 1, american))


@pytest.mark.parametrize('world,option_type,strike', CASES,
                         ids=['r%g-q%g-v%g-%s-%g' % (w[0], w[1], w[2], o, k)
                              for w, o, k in CASES])
def test_the_american_price_is_bounded_by_black_and_the_converged_binomial(
        world, option_type, strike):
    """The bound, not a tolerance: the approximation prices a sub-optimal exercise policy, so it
    can never beat the binomial, and it is floored at Black, so it can never be worth less than
    waiting. Both sides are asserted, and the gap to the binomial carries the measured 2.07%."""
    american, european = _both(strike, option_type, world)
    oracle = _oracle(strike, option_type, world)
    assert american >= european - 1e-9, (american, european)
    assert american <= oracle * (1.0 + 1e-4), (american, oracle)
    assert abs(american - oracle) / oracle < 3e-2, (american, oracle, world)


@pytest.mark.parametrize('world,option_type', [((0.05, 0.10, 0.30), 'Call'),
                                               ((0.08, 0.02, 0.30), 'Put'),
                                               ((0.05, 0.10, 0.20), 'Call')],
                         ids=['call-hi-div', 'put-hi-rate', 'call-hi-div-low-vol'])
def test_the_early_exercise_premium_is_material_and_the_approximation_captures_it(
        world, option_type):
    """A fixture whose American premium is noise cannot tell this pricer from `pv_european_option`,
    which is the state every equity fixture in the repo was in. Each leg here carries a premium
    worth at least a tenth of the mark, and the engine must find at least three quarters of it."""
    for strike in STRIKES:
        american, european = _both(strike, option_type, world)
        oracle = _oracle(strike, option_type, world)
        oracle_european = _oracle(strike, option_type, world, american=False)
        premium = oracle - oracle_european
        assert premium / oracle > 0.10, (strike, premium, oracle)
        assert (american - european) / premium > 0.75, (strike, american - european, premium)


def test_the_carry_is_read_off_the_forward_settlement_not_the_expiry():
    """The convention the oracle has to share, measured rather than assumed: the European control
    reproduces the binomial at `b = (r - q) * T_fwd / T` an order and a half better than at the
    naive `r - q`, on the deal shape whose two-business-day forward settlement causes it."""
    world, strike = (0.05, 0.10, 0.30), 100.0
    _, european = _both(strike, 'Call', world)
    r, q, vol = world
    at_forward = UNITS * 0.5 * (_crr(strike, 1.0, r, (r - q) * T_FWD / T, vol, 3000, False) +
                                _crr(strike, 1.0, r, (r - q) * T_FWD / T, vol, 3001, False))
    at_expiry = UNITS * 0.5 * (_crr(strike, 1.0, r, r - q, vol, 3000, False) +
                               _crr(strike, 1.0, r, r - q, vol, 3001, False))
    assert abs(european - at_forward) / at_forward < 1e-4, (european, at_forward)
    assert abs(european - at_expiry) / at_expiry > 3e-4, (european, at_expiry)


def test_the_oracle_is_converged():
    """The tolerance above is only worth what the binomial is worth: 3000 steps against 8000, on
    the fixture where the approximation is furthest away, and on the deepest premium."""
    for world, option_type, strike in (((0.05, 0.10, 0.20), 'Call', 120.0),
                                       ((0.08, 0.02, 0.30), 'Put', 120.0)):
        coarse = _oracle(strike, option_type, world)
        fine = _oracle(strike, option_type, world, steps=8000)
        assert abs(coarse - fine) / fine < 1e-4, (world, option_type, strike, coarse, fine)


def test_a_zero_dividend_call_is_bit_identically_its_european():
    """The `b >= r` arm. With no dividend the carry IS the rate, early exercise is never optimal
    on a call, and the pricer must take Black unchanged - the same float, not a close one."""
    world = (0.05, 0.0, 0.30)
    american, european = _both(100.0, 'Call', world)
    assert american == european, (american, european)
    oracle = _oracle(100.0, 'Call', world)
    assert abs(american - oracle) / oracle < 1e-4, (american, oracle)


def test_a_call_past_its_exercise_trigger_marks_exactly_its_intrinsic():
    """The `(S >= I) * (S - K)` arm and the `first_knockout` settlement beside it. A deep call in
    the high-dividend world sits past its own trigger today, so its mark is the intrinsic to the
    last bit - the binomial puts the true value 0.027% above it, which is the approximation's own
    floor and is what the dominance gate above admits."""
    american, _ = _both(80.0, 'Call', (0.05, 0.10, 0.20))
    assert american == UNITS * (SPOT - 80.0), american
    oracle = _oracle(80.0, 'Call', (0.05, 0.10, 0.20))
    assert american < oracle and (oracle - american) / oracle < 1e-3, (american, oracle)
