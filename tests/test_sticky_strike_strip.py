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
the LAST BIT, not to a tolerance. Below the knots are the fixing tenors themselves, so the
forward-valued `ATM_Ref` is read at its own knot and no interpolation stands between them.

A SECOND ORACLE removes the smile instead of the difference: with no slope and no wings every
coordinate reads `ATM_Vol`, so the two rules agree whatever the reference is.

MEASURED on the document below (notional 1, strike 100, barrier 85, monthly monitoring, one year,
ATM 25 vol with a -0.15 slope). Declared `Sticky_Strike` against a flat `ATM_Ref` of 100 the
barrier marks 10.088311, and against `Sticky_Moneyness` 10.178774 - **-0.89%**, which is the whole
of the difference between reading a 25.00 vol at every fixing and reading 25.02 through 25.31 vol
down the smile's slope, the strike sitting under the forward by the 2% carry.
"""
import json
import os
import sys

# reference-derivus shadow-import guard (MEMORY): pin the package under test to THIS repo.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

import derivus
from derivus import utils
from derivus.config import CustomJsonEncoder, ModelParams

BASE = pd.Timestamp('2024-06-28')
SPOT, R_USD, Q_EQ = 100.0, 0.04, 0.02
STRIKE, UNITS, BARRIER, EXPIRY_D = 100.0, 1.0, 85.0, 365
MONITOR_D = list(range(30, 361, 30))

#: the knots ARE the fixing tenors, so a forward-valued ATM_Ref is read at its own knot
TENORS = [0.02] + [d / 365.0 for d in MONITOR_D] + [1.0, 2.0]
SKEW = {'ATM_Vol': 0.25, 's': -0.15, 'L': 0.30, 'R': 0.20,
        'C': -0.35, 'D': 0.35, 'lam': 0.5, 'rho': 0.5}


def curve(value):
    return utils.Curve([], [[t, value] for t in TENORS])


FORWARD_REF = utils.Curve([], [[t, SPOT * float(np.exp((R_USD - Q_EQ) * t))] for t in TENORS])


def factors(rule, atm_ref, smile=True):
    vol = {'Surface_Type': 'Skew', 'Moneyness_Rule': rule, 'Currency': 'USD', 'ATM_Ref': atm_ref}
    vol.update({k: curve(v if smile or k not in ('s', 'L', 'R') else 0.0)
                for k, v in SKEW.items()})
    return {
        'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Spot': 1.0},
        'InterestRate.USD': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                             'Curve': utils.Curve([], [[0.0, R_USD], [5.0, R_USD]])},
        'EquityPrice.EQ': {'Spot': SPOT, 'Currency': 'USD', 'Interest_Rate': 'USD',
                           'Issuer': '', 'Respect_Default': 'No', 'Jump_Level': 0.0},
        'DividendRate.EQ': {'Currency': 'USD', 'Floor': None,
                            'Curve': utils.Curve([], [[0.0, Q_EQ], [5.0, Q_EQ]])},
        'EquityPriceVol.EQ': vol}


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
      'MCMC_Simulations': 8192, 'Random_Seed': 1}

#: the equity alone is simulated, so a CreditMonteCarlo reports a profile with many rows
CMC = {'Object': 'CreditMonteCarlo', 'Base_Date': BASE, 'Currency': 'USD',
       'Time_grid': '0d 12m(1m)', 'Batch_Size': 64, 'Simulation_Batches': 1,
       'Random_Seed': 1, 'MCMC_Simulations': 2048, 'Deflation_Interest_Rate': 'USD',
       'Generate_Cashflows': 'No'}
MODELS = {'Model Configuration': ModelParams(({'EquityPrice': 'GBMAssetPriceModel'}, {})),
          'Price Models': {'GBMAssetPriceModel.EQ': {'Drift': 0.0, 'Vol': 0.25}}}


def run(price_factors, calc=BV):
    market = {'System Parameters': {'Base_Currency': 'USD', 'Base_Date': BASE},
              'Valuation Configuration': {}, 'Price Factors': price_factors}
    if calc is CMC:
        market.update(MODELS)
    job = {'Calc': {
        'Calculation': dict(calc),
        'Deals': {'Tag_Titles': '', 'Reference': 'sticky',
                  'Deals': {'Children': [{'Instrument': {'.Deal': DEAL}}]}},
        'MergeMarketData': {'MarketDataFile': '', 'ExplicitMarketData': market}}}
    cx = derivus.Context()
    cx.load_json((json.dumps(job, cls=CustomJsonEncoder), 'sticky'))
    _, out = cx.run_job()
    return out['Results']['mtm']


def mtm(price_factors):
    """The deal's own row. A deal that left no row was skipped, and says so."""
    rows = run(price_factors)
    own = rows[rows['Reference'] == 'BR']['Value']
    assert len(own) == 1, 'the deal left no row in Results: it was skipped, not priced'
    return float(own.iloc[0])


def test_a_skew_surface_under_sticky_strike_prices_a_monitoring_strip():
    """The row this file exists for: the barrier prices, and prices to its measured mark."""
    assert mtm(factors('Sticky_Strike', curve(SPOT))) == pytest.approx(10.088311, rel=1e-6)


def test_the_two_conventions_are_one_coordinate_where_the_reference_is_the_forward():
    """`log(K / ATM_Ref)` and `log(K / F)` are the same number when `ATM_Ref` is `F`, so the two
    documents read the same vol at every fixing and mark to the BIT. This is what places the
    broadcast coordinate on the surface: a strip read at the wrong tenor, or at the spot where it
    wants the forward, breaks this identity while still producing a number.
    """
    assert mtm(factors('Sticky_Strike', FORWARD_REF)) == mtm(
        factors('Sticky_Moneyness', curve(SPOT)))


def test_without_a_smile_the_rule_cannot_be_read_differently():
    """No slope and no wings: every coordinate reads `ATM_Vol`, so the reference cannot matter.
    It is the difference between THIS and the identity above that sizes the convention.
    """
    flat = mtm(factors('Sticky_Strike', curve(SPOT), smile=False))
    assert flat == mtm(factors('Sticky_Moneyness', curve(SPOT), smile=False))
    assert flat == mtm(factors('Sticky_Strike', FORWARD_REF, smile=False))
    assert flat != mtm(factors('Sticky_Moneyness', curve(SPOT))), 'the smile must be worth something'


def test_the_strip_is_read_the_same_way_at_every_reporting_row():
    """A profile asks for the strip once per reporting date, where a valuation asks once. The
    second shape is the one that came back with a row per date and was reshaped to one number.
    """
    profile = run(factors('Sticky_Strike', curve(SPOT)), CMC)
    assert len(profile) > 1 and np.isfinite(profile.values).all()
    assert float(np.mean(profile.iloc[0].values)) == pytest.approx(10.088311, rel=5e-3)
