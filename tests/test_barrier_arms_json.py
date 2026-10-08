"""All eight closed-form payoff arms of `getbarrierpayoff`, through the JSON contract and nothing
else.

The selector inside `getbarrierpayoff.barrier_option` is `(direction, eta, phi, strike vs H)` -
four arms under knock-IN, four under knock-OUT, each reached by TWO spellings of the same geometry
(a Call with an Up barrier, a Put with a Down one). This file prices all sixteen spellings at rebate
0 and at a live rebate, and pins them as MUST_COVER in the census.

THE ORACLE IS THE TEXTBOOK, longhand: `_reiner_rubinstein` builds the six terms A-F from
`math.erfc` and selects the arm from the same enumeration the pricer states. It shares nothing
with the engine, whose arms are a sympy-flattened `erfc` algebra in which a sign slip is invisible
by inspection.

MEASURED. Against the textbook, all sixteen at both rebates: worst 1.4e-14 relative. The textbook
itself was held against a model-free bridge-corrected daily Monte Carlo (worst 1.3e-2 at rebate 0,
the Down-and-In Call, the smallest mark in the table) and IN + OUT = the vanilla to 7.2e-15, arm by
arm. Each fixture priced through the three arms it must NOT take reads 0.209 at the nearest and
181.0 at the farthest on a notional of 1000, eleven of the forty-eight NEGATIVE, so the gate's 1e-11
is many orders inside any mis-selection. An Up-and-Out Call struck ABOVE its barrier is worthless on
survival - 0.000000 with no rebate, 18.470170 with one - the only fixture reaching the knock-out's
`cash_settle` of the rebate.

DEGENERACY CHECKLIST: r = 4% against q = 2%, so the carry is 2% and neither rate is zero; both
option types and both barrier directions on EVERY arm; both knock directions; both rebates.
"""
import json
import math
import os
import sys

# reference-derivus shadow-import guard (MEMORY): pin the package under test to THIS repo.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest

import derivus
from derivus import utils
from derivus.config import CustomJsonEncoder

BASE = pd.Timestamp('2024-06-28')
X0, R_USD, R_EUR, SIGMA = 1.25, 0.04, 0.02, 0.15
NOTIONAL, REBATE, EXPIRY_D = 1000.0, 40.0, 365
T = EXPIRY_D / 365.0
B_CARRY = R_USD - R_EUR

FACTORS = {
    'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Spot': 1.0},
    'FxRate.EUR': {'Domestic_Currency': None, 'Interest_Rate': 'EUR', 'Spot': X0},
    'InterestRate.USD': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                         'Curve': utils.Curve([], [[0.0, R_USD], [5.0, R_USD]])},
    'InterestRate.EUR': {'Currency': 'EUR', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                         'Curve': utils.Curve([], [[0.0, R_EUR], [5.0, R_EUR]])},
    'FXVol.EUR.USD': {'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness',
                      'Surface': utils.Curve([], [[m, t, SIGMA] for m in (0.6, 1.0, 1.4)
                                                  for t in (0.02, 2.0)])}}

#: `(arm, Up|Down, barrier, strike, option type)`. Two rows per arm, the two spellings the
#: selector treats as one case, chosen so every arm carries a Call AND a Put and an Up AND a Down.
#: Arm 4 is the one every pre-existing fixture took; it is here as the control.
ARMS = [
    (1, 'Up', 1.40, 1.45, 'Call'), (1, 'Down', 1.12, 1.05, 'Put'),
    (2, 'Up', 1.40, 1.20, 'Call'), (2, 'Down', 1.12, 1.30, 'Put'),
    (3, 'Up', 1.40, 1.45, 'Put'), (3, 'Down', 1.12, 1.05, 'Call'),
    (4, 'Down', 1.12, 1.25, 'Call'), (4, 'Up', 1.40, 1.30, 'Put'),
]
CASES = [(arm, f'{ud}_And_{io}', h, k, opt, rebate)
         for arm, ud, h, k, opt in ARMS for io in ('Out', 'In') for rebate in (0.0, REBATE)]


def _deal(barrier_type, barrier, strike, option_type, rebate):
    return {'Object': 'FXBarrierOption', 'Reference': 'BR', 'Currency': 'USD',
            'Underlying_Currency': 'EUR', 'Payoff_Currency': 'USD', 'Discount_Rate': 'USD',
            'FX_Volatility': 'EUR.USD', 'Buy_Sell': 'Buy', 'Option_Type': option_type,
            'Strike_Price': strike, 'Barrier_Price': barrier, 'Barrier_Type': barrier_type,
            'Barrier_Monitoring_Frequency': pd.DateOffset(days=0), 'Cash_Rebate': rebate,
            'Underlying_Amount': NOTIONAL,
            'Expiry_Date': BASE + pd.DateOffset(days=EXPIRY_D)}


def _run(deal):
    job = {'Calc': {
        'Calculation': {'Object': 'BaseValuation', 'Base_Date': BASE, 'Currency': 'USD',
                        'MCMC_Simulations': 1, 'Random_Seed': 1},
        'Deals': {'Reference': 'arms',
                  'Deals': {'Children': [{'Instrument': {'.Deal': deal}}]}},
        'MergeMarketData': {'MarketDataFile': '', 'ExplicitMarketData': {
            'System Parameters': {'Base_Currency': 'USD', 'Base_Date': BASE},
            'Valuation Configuration': {}, 'Price Factors': FACTORS}}}}
    cx = derivus.Context()
    cx.load_json((json.dumps(job, cls=CustomJsonEncoder), 'arms'))
    _, out = cx.run_job()
    rows = out['Results']['mtm']
    return float(rows[rows['Reference'] == 'BR']['Value'].iloc[0])


# --------------------------------------------------------------------------------------------
# oracle one: the textbook terms, longhand
# --------------------------------------------------------------------------------------------
def _n(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _reiner_rubinstein(barrier_type, barrier, strike, option_type, rebate):
    """The six terms A-F and the eight-way selector, straight from the closed forms - no engine."""
    up, knock_out = 'Up' in barrier_type, 'Out' in barrier_type
    eta = -1.0 if up else 1.0
    phi = 1.0 if option_type == 'Call' else -1.0
    reb = rebate / NOTIONAL
    v = SIGMA * math.sqrt(T)
    mu = (B_CARRY - 0.5 * SIGMA ** 2) / SIGMA ** 2
    lam = math.sqrt(mu * mu + 2.0 * R_USD / SIGMA ** 2)
    x1 = math.log(X0 / strike) / v + (1.0 + mu) * v
    x2 = math.log(X0 / barrier) / v + (1.0 + mu) * v
    y1 = math.log(barrier * barrier / (X0 * strike)) / v + (1.0 + mu) * v
    y2 = math.log(barrier / X0) / v + (1.0 + mu) * v
    z = math.log(barrier / X0) / v + lam * v
    hs, carry, disc = barrier / X0, math.exp((B_CARRY - R_USD) * T), math.exp(-R_USD * T)

    def term(d, eps):
        return phi * X0 * carry * _n(eps * d) - phi * strike * disc * _n(eps * (d - v))

    def reflected(d, eps):
        return (phi * X0 * carry * hs ** (2 * (mu + 1)) * _n(eps * d) -
                phi * strike * disc * hs ** (2 * mu) * _n(eps * (d - v)))

    a, b_, c, d_ = term(x1, phi), term(x2, phi), reflected(y1, eta), reflected(y2, eta)
    e = reb * disc * (_n(eta * (x2 - v)) - hs ** (2 * mu) * _n(eta * (y2 - v)))
    f = reb * (hs ** (mu + lam) * _n(eta * z) +
               hs ** (mu - lam) * _n(eta * (z - 2 * lam * v)))
    above = strike > barrier
    first = (phi > 0 and up and above) or (phi < 0 and not up and not above)
    second = (phi > 0 and up and not above) or (phi < 0 and not up and above)
    third = (phi < 0 and up and above) or (phi > 0 and not up and not above)
    if knock_out:
        pick = f if first else (a - b_ + c - d_ + f if second else
                                (b_ - d_ + f if third else a - c + f))
    else:
        pick = a + e if first else (b_ - c + d_ + e if second else
                                    (a - b_ + d_ + e if third else c + e))
    return NOTIONAL * pick


IDS = ['arm%d-%s-K%g-%s-reb%g' % (a, bt, k, o, r) for a, bt, _, k, o, r in CASES]


@pytest.mark.parametrize('arm,barrier_type,barrier,strike,option_type,rebate', CASES, ids=IDS)
def test_every_payoff_arm_is_the_textbook_closed_form(
        arm, barrier_type, barrier, strike, option_type, rebate):
    """Sixteen spellings at two rebates against the longhand terms. The tolerance is float64
    round-off, not a modelling allowance: the two algebras are the same function or they are not,
    and the worst reading over the whole table is 1.4e-14.

    Killing mutation: the knock-out selector's down-put spelling of arm 1 inverted
    (`strike <= H` -> `strike > H`).
    """
    got = _run(_deal(barrier_type, barrier, strike, option_type, rebate))
    ref = _reiner_rubinstein(barrier_type, barrier, strike, option_type, rebate)
    assert abs(got - ref) <= 1e-9 + 1e-11 * abs(ref), (arm, barrier_type, got, ref)
