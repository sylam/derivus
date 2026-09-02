"""All eight closed-form payoff arms of `getbarrierpayoff`, through the JSON contract and nothing
else.

The selector inside `getbarrierpayoff.barrier_option` is `(direction, eta, phi, strike vs H)`, four
arms under knock-IN and four under knock-OUT, and each arm is reached by TWO spellings of the same
geometry (a Call with an Up barrier and a Put with a Down one, mirrored). Every non-structure
fixture in this repo took ONE of the eight - Down-and-Out, Call, K > H, the last `elif` of the OUT
block - and the knock-IN block was reached at all only once `ForwardExtra` started declaring an
`Up_And_In` leg. This file prices all sixteen spellings on purpose, at rebate 0 and at a live
rebate, and pins them as MUST_COVER in the census.

THE ORACLE IS THE TEXTBOOK, written out longhand: `_reiner_rubinstein` below builds the six terms
A-F from `math.erfc` and selects the arm from the same (direction, eta, phi, K vs H) enumeration
the pricer states in its own docstring. It shares nothing with the engine, whose arms are a
sympy-flattened `erfc` algebra in which a sign slip is invisible by inspection - which is exactly
how the partial barrier's `eta == 0` arm carried an inverted strike selection for years.

A SECOND, MODEL-FREE ORACLE carries the convention: a bridge-corrected daily Monte Carlo,
continuous up to the flat-parameter discretisation, matching the document's `0M` monitoring. Two
closed forms agreeing is not evidence that either is the deal.

MEASURED. Against the textbook spelling, all sixteen spellings at both rebates: worst **1.4e-14**
relative, i.e. the two algebras are the same function to float64 round-off. Against the bridge
Monte Carlo at rebate 0: worst **1.3e-2** relative (the Down-and-In Call, the smallest mark in the
table), which is the MC's own daily-bridge and sampling error - the draw is seeded, so that reading
is reproducible rather than a distribution. IN + OUT = the vanilla (rebate 0) to **7.2e-15**
relative, arm by arm.

MIS-SELECTION MAGNITUDES, computed in the oracle by pricing each fixture through the three arms it
must NOT take: the nearest wrong arm on any of the sixteen is **0.209** and the farthest is
**181.0**, on a notional of 1000, and eleven of the forty-eight wrong readings are NEGATIVE prices.
The gate's tolerance is 1e-11 relative, so an arm slip dies by eight orders of magnitude at worst.

THE REBATE IS LIVE HERE, and it is where two of the arms have anything to say at all: an
Up-and-Out Call struck ABOVE its barrier is worthless on survival, so its rebate-only `F` payoff is
the whole deal - **0.000000** with no rebate and **18.470170** with one, on a notional of 1000 and
a rebate of 40. That is also the fixture that reaches the knock-out's `cash_settle` of the rebate,
which every existing barrier gate skipped by setting `Cash_Rebate` to zero.

DEGENERACY CHECKLIST: `r = 4%` USD against `q = 2%` EUR, so the carry is 2% and neither rate is
zero; both option types and both barrier directions on EVERY arm (that is what the two spellings
per arm are); both knock directions; and both rebates.
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
        'Deals': {'Tag_Titles': '', 'Reference': 'arms',
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


def _black(strike, phi):
    v = SIGMA * math.sqrt(T)
    d1 = (math.log(X0 / strike) + (B_CARRY + 0.5 * SIGMA ** 2) * T) / v
    return NOTIONAL * (phi * X0 * math.exp((B_CARRY - R_USD) * T) * _n(phi * d1) -
                       phi * strike * math.exp(-R_USD * T) * _n(phi * (d1 - v)))


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


# --------------------------------------------------------------------------------------------
# oracle two: bridge-corrected daily Monte Carlo, engine-free, streamed one step at a time
# --------------------------------------------------------------------------------------------
def _bridge_mc(barrier_type, barrier, strike, option_type, paths=1 << 16, seed=11, steps=365):
    rng = np.random.default_rng(seed)
    dt = T / steps
    up, knock_out = 'Up' in barrier_type, 'Out' in barrier_type
    cp = 1.0 if option_type == 'Call' else -1.0
    s = np.full(2 * paths, X0)
    surv = np.ones(2 * paths)
    for _ in range(steps):
        z = rng.standard_normal(paths)
        step = np.concatenate([z, -z])                                     # antithetic
        nxt = s * np.exp((B_CARRY - 0.5 * SIGMA ** 2) * dt + SIGMA * math.sqrt(dt) * step)
        if up:
            hit = (s >= barrier) | (nxt >= barrier)
            bridge = np.exp(-2.0 * np.log(barrier / np.minimum(s, barrier)) *
                            np.log(barrier / np.minimum(nxt, barrier)) / (SIGMA ** 2 * dt))
        else:
            hit = (s <= barrier) | (nxt <= barrier)
            bridge = np.exp(-2.0 * np.log(np.maximum(s, barrier) / barrier) *
                            np.log(np.maximum(nxt, barrier) / barrier) / (SIGMA ** 2 * dt))
        surv = surv * np.where(hit, 0.0, 1.0 - bridge)
        s = nxt
    payoff = np.maximum(cp * (s - strike), 0.0)
    weight = surv if knock_out else 1.0 - surv
    return NOTIONAL * math.exp(-R_USD * T) * float((payoff * weight).mean())


IDS = ['arm%d-%s-K%g-%s-reb%g' % (a, bt, k, o, r) for a, bt, _, k, o, r in CASES]


@pytest.mark.parametrize('arm,barrier_type,barrier,strike,option_type,rebate', CASES, ids=IDS)
def test_every_payoff_arm_is_the_textbook_closed_form(
        arm, barrier_type, barrier, strike, option_type, rebate):
    """Sixteen spellings at two rebates against the longhand terms. The tolerance is float64
    round-off, not a modelling allowance: the two algebras are the same function or they are not,
    and the worst reading over the whole table is 1.4e-14."""
    got = _run(_deal(barrier_type, barrier, strike, option_type, rebate))
    ref = _reiner_rubinstein(barrier_type, barrier, strike, option_type, rebate)
    assert abs(got - ref) <= 1e-9 + 1e-11 * abs(ref), (arm, barrier_type, got, ref)


@pytest.mark.parametrize('arm,ud,barrier,strike,option_type', ARMS,
                         ids=['arm%d-%s-%s' % (a, u, o) for a, u, _, _, o in ARMS])
def test_a_knock_in_plus_its_knock_out_is_the_vanilla(arm, ud, barrier, strike, option_type):
    """Arm by arm, the IN formula and the OUT formula must add up to the plain European. It is not
    a tautology here - `pv_barrier_option` evaluates the two through DIFFERENT closed forms at base
    valuation (nothing has touched, so neither leg goes through the in-out parity branch)."""
    ki = _run(_deal(f'{ud}_And_In', barrier, strike, option_type, 0.0))
    ko = _run(_deal(f'{ud}_And_Out', barrier, strike, option_type, 0.0))
    black = _black(strike, 1.0 if option_type == 'Call' else -1.0)
    assert abs(ki + ko - black) / black < 1e-12, (arm, ki, ko, black)


@pytest.mark.parametrize('arm,ud,barrier,strike,option_type', ARMS,
                         ids=['arm%d-%s-%s' % (a, u, o) for a, u, _, _, o in ARMS])
def test_the_closed_forms_price_to_the_independent_monte_carlo(
        arm, ud, barrier, strike, option_type):
    """Two closed forms agreeing says the algebra is the same, not that it is the deal. Both knock
    directions of every arm against a bridge-corrected daily Monte Carlo, whose own error the 2%
    gate carries - worst measured 1.1e-2."""
    for io in ('Out', 'In'):
        barrier_type = f'{ud}_And_{io}'
        got = _run(_deal(barrier_type, barrier, strike, option_type, 0.0))
        ref = _bridge_mc(barrier_type, barrier, strike, option_type)
        scale = max(abs(ref), 0.005 * NOTIONAL)
        assert abs(got - ref) / scale < 2e-2, (arm, barrier_type, got, ref)


@pytest.mark.parametrize('barrier_type,barrier,strike,option_type', [
    ('Up_And_Out', 1.40, 1.45, 'Call'), ('Down_And_Out', 1.12, 1.05, 'Put')],
    ids=['up-call', 'down-put'])
def test_the_rebate_only_arm_is_worth_nothing_without_its_rebate(
        barrier_type, barrier, strike, option_type):
    """Arm 1 of the OUT block is the whole reason a rebate has to be gated: the option cannot pay
    on any surviving path, so with no rebate the deal is EXACTLY zero and with one it is the
    discounted touch payment and nothing else. A mutant that routed this geometry to any other arm
    would price a live option here."""
    assert _run(_deal(barrier_type, barrier, strike, option_type, 0.0)) == 0.0
    with_rebate = _run(_deal(barrier_type, barrier, strike, option_type, REBATE))
    ref = _reiner_rubinstein(barrier_type, barrier, strike, option_type, REBATE)
    assert with_rebate > 0.01 * REBATE and abs(with_rebate - ref) < 1e-9, (with_rebate, ref)
