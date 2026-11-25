"""An option paid after it expires, through the JSON contract and nothing else.

`Settlement_Date` is the day the payoff is paid, the expiry where blank. Between the two the
underlying is what the expiry fixed: on a simulated grid the path's own level on the expiry row,
and on a base date past the expiry the print `Price_Fixing` states - which an expired deal must
state, since the spot of today is not the level it settled on. A swaption settled in cash is
paid the same way; settled physically it becomes the swap its legs make and books its cash.

THE ORACLES: under a credit Monte Carlo every row from the expiry on is the settlement row's payoff
discounted to it, scenario by scenario, and the ledger books that payoff on the settlement row and
nothing on the expiry's; under a base valuation past the expiry the mark is the print's intrinsic
discounted to the settlement, exact. A barrier, a touch and an Asian are read besides against a twin
settling on the expiry, on the same paths.
"""
import functools
import json
import math
import os
import sys

# reference-derivus shadow-import guard (MEMORY): pin the package under test to THIS repo.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
import torch

import derivus
import test_declared_defaults as book
import trial_rates
from derivus import bootstrappers, utils
from derivus.config import CustomJsonEncoder

BASE = pd.Timestamp('2024-06-28')
EXPIRY = pd.Timestamp('2025-06-27')                  # a Friday
SETTLE = EXPIRY + pd.offsets.BDay(2)                 # the Tuesday after
SPOT, STRIKE, UNITS, CASH, PRINT = 100.0, 100.0, 10.0, 1000.0, 112.0
R_USD, Q_EQ, SIGMA = 0.04, 0.01, 0.25
FX_SPOT, R_EUR, FX_SIGMA = 1.25, 0.02, 0.15

FACTORS = {
    'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Spot': 1.0},
    'FxRate.EUR': {'Domestic_Currency': None, 'Interest_Rate': 'EUR', 'Spot': FX_SPOT},
    'InterestRate.USD': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                         'Curve': utils.Curve([], [[0.0, R_USD], [5.0, R_USD]])},
    'InterestRate.EUR': {'Currency': 'EUR', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                         'Curve': utils.Curve([], [[0.0, R_EUR], [5.0, R_EUR]])},
    'FXVol.EUR.USD': {'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness',
                      'Surface': utils.Curve([], [[m, t, FX_SIGMA] for m in (0.6, 1.0, 1.4)
                                                  for t in (0.02, 2.0)])},
    'EquityPrice.EQ': {'Spot': SPOT, 'Currency': 'USD', 'Interest_Rate': 'USD', 'Issuer': '',
                       'Respect_Default': 'No', 'Jump_Level': 0.0},
    'DividendRate.EQ': {'Currency': 'USD', 'Curve': utils.Curve([], [[0.0, Q_EQ], [5.0, Q_EQ]])},
    'EquityPriceVol.EQ': {'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness',
                          'Surface': utils.Curve([], [[m, t, SIGMA] for m in (0.6, 1.0, 1.4)
                                                      for t in (0.02, 2.0)])}}

VANILLA = {'Object': 'EquityOptionDeal', 'Reference': 'EQO', 'Currency': 'USD',
           'Payoff_Currency': 'USD', 'Equity': 'EQ', 'Dividends': 'EQ', 'Discount_Rate': 'USD',
           'Equity_Volatility': 'EQ', 'Buy_Sell': 'Buy', 'Option_Type': 'Call',
           'Strike_Price': STRIKE, 'Units': UNITS, 'Expiry_Date': EXPIRY, 'Settlement_Date': SETTLE}
BINARY = dict({k: v for k, v in VANILLA.items() if k != 'Units'},
              Object='EquityBinaryOption', Reference='EQB', Payoff=CASH)
FX = {'Object': 'FXOptionDeal', 'Reference': 'FXO', 'Currency': 'USD', 'Underlying_Currency': 'EUR',
      'Discount_Rate': 'USD', 'FX_Volatility': 'EUR.USD', 'Buy_Sell': 'Buy', 'Option_Type': 'Call',
      'Strike_Price': FX_SPOT, 'Underlying_Amount': UNITS, 'Expiry_Date': EXPIRY, 'Delivery_Date': EXPIRY}
CMC = {'Object': 'CreditMonteCarlo', 'Time_grid': '0d 12m(1m)', 'Batch_Size': 256, 'Random_Seed': 1,
       'Simulation_Batches': 1, 'Deflation_Interest_Rate': 'USD', 'Generate_Cashflows': 'Yes'}


def node(deal):
    """A deal-tree node, a netting set's `Children` beneath it."""
    out = {'Instrument': {'.Deal': {k: v for k, v in deal.items() if k != 'Children'}}}
    return dict(out, Children=[node(child) for child in deal['Children']]) if 'Children' in deal else out


def world(deals, base=BASE, calc=None, factors=None, models=None):
    """A context over one job on `deals`: a base valuation at `base`, or `calc` on a GBM spot and
    the factor types `models` names simulated as it says, `factors` beside the world's."""
    market = {'System Parameters': {'Base_Currency': 'USD', 'Base_Date': base},
              'Valuation Configuration': {}, 'Price Factors': dict(FACTORS, **(factors or {}))}
    if calc:
        market['Price Models'] = {'GBMAssetPriceModel.EQ': {'Vol': SIGMA, 'Drift': 0.0}}
        market['Model Configuration'] = {'.ModelParams': {
            'modeldefaults': dict({'EquityPrice': 'GBMAssetPriceModel'}, **(models or {})), 'modelfilters': {}}}
    job = {'Calc': {
        'Calculation': dict({'Object': 'BaseValuation', 'Base_Date': base, 'Currency': 'USD',
                             'MCMC_Simulations': 1}, **(calc or {})),
        'CalendDataFile': book.CALENDARS,
        'Deals': {'Reference': 'settle', 'Deals': {'Children': [node(d) for d in deals]}},
        'MergeMarketData': {'MarketDataFile': '', 'ExplicitMarketData': market}}}
    return derivus.Context().load_json((json.dumps(job, cls=CustomJsonEncoder), 'settle'))


def run(deals, base=BASE, calc=None, factors=None, models=None):
    """The results of the job `world` holds."""
    return world(deals, base, calc, factors, models).run_job()[1]['Results']


def marks(deals, base=BASE, factors=None):
    frame = run(deals, base, factors=factors)['mtm']
    return dict(zip(frame['Reference'], frame['Value'].astype(float)))


def fitted(base, factors, expiry, underlying, priced_in=''):
    """The forward the option-quote fit grows a quote to `expiry` at, on `underlying` priced in
    `priced_in` or the base, in the world `run` prices in."""
    config = world([], base, factors=factors).current_cfg
    params, family = config.params, bootstrappers.construct_bootstrapper('LogVar2FJModelParameters', {})
    instrument = dict(family.param, Underlying=underlying, Priced_In=priced_in, Discount_Rate=priced_in or 'USD',
                      Yield=underlying, Volatility='{}.{}'.format(underlying, priced_in or 'USD'),
                      Quote_Type='Implied_Volatility', European_Options=[{
                          'Expiry_Date': expiry, 'Strike': 0.0, 'Option_Type': 'Call', 'Units': 1.0, 'Weight': 1.0,
                          'Quoted_Market_Value': FX_SIGMA}])
    resolved, spot, convention = family.resolve_block(
        '.'.join(['LogVar2FJModelPrices', underlying] + [priced_in] * bool(priced_in)), instrument,
        params['Price Factors'], params['Price Factor Interpolation'], params['System Parameters'], config.holidays)
    return next(family.prepare_quotes(params['System Parameters'], instrument, resolved, spot, convention))[4]


def lagged(**days):
    """The world's share `EQ` or rate `EUR` stating `Settlement_Days`, keyed by its name."""
    names = {'EQ': 'EquityPrice.EQ', 'EUR': 'FxRate.EUR'}
    return {names[key]: dict(FACTORS[names[key]], Settlement_Days=value) for key, value in days.items()}


def df(days):
    return math.exp(-R_USD * days / 365.0)


def ndtr(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def lag(date, days, holidays=()):
    """The calendar days from `date` to its spot date, `days` business days on past `holidays`."""
    return ((date + pd.offsets.CustomBusinessDay(days, holidays=list(holidays))) - date).days


def swaplet(reference, start, end, paid, **stated):
    """A one-period equity swaplet list on shares, paying the level its end fixes and no other."""
    return dict({
        'Object': 'EquitySwapletListDeal', 'Reference': reference, 'Currency': 'USD', 'Discount_Rate': 'USD',
        'Equity': 'EQ', 'Equity_Currency': 'USD', 'Buy_Sell': 'Buy', 'Amount_Type': 'Shares',
        'Cashflows': {'Items': [{
            'Start_Date': start, 'End_Date': end, 'Payment_Date': paid, 'Amount': UNITS, 'Start_Multiplier': 0.0,
            'End_Multiplier': 1.0, 'Dividend_Multiplier': 0.0, 'Known_Start_Price': SPOT}]}}, **stated)


def test_past_the_expiry_every_row_is_the_settlement_payoff_discounted_and_the_ledger_books_it_there():
    """Under a credit Monte Carlo on a simulated spot the grid runs to the settlement, with the
    monthly date between it and the expiry. On every row from the expiry on the mark is the
    settlement row's payoff discounted to it, scenario by scenario - the level is the expiry row's
    own and not that row's spot - and the ledger books that payoff on the settlement row alone.
    The run is float32, so the readings are held to 1e-5.

    Killing mutations: the level not held past the expiry (`held_past_expiry` handing back its
    input), every later row marking its own spot's intrinsic; the settlement not a reval date,
    which leaves no row to book on.
    """
    results = run([VANILLA], calc=CMC)
    mtm, ledger = results['mtm'], results['cashflows']['USD']
    assert np.isfinite(np.asarray(mtm.values, dtype=float)).all()
    assert list(ledger.index) == [SETTLE], ledger.index
    paid, settled = (np.asarray(frame.loc[SETTLE].values, dtype=float) for frame in (ledger, mtm))
    assert np.allclose(paid, settled, rtol=1e-5, atol=1e-6)
    assert settled.mean() > 0.0 and (settled == 0.0).any(), 'in and out of the money across the paths'
    later = [day for day in mtm.index if day >= EXPIRY]
    assert len(later) == 3, later                    # the expiry, the monthly date after it, the settlement
    for day in later:
        marked = np.asarray(mtm.loc[day].values, dtype=float)
        assert np.allclose(marked, settled * df((SETTLE - day).days), rtol=1e-5, atol=1e-6), day


def test_expired_before_the_base_date_the_payoff_is_on_the_print_and_a_missing_one_refuses():
    """On a base date between the expiry and the settlement the vanilla marks its print's intrinsic
    discounted to the settlement, exact; the cash digital its payoff discounted, the asset digital
    its payoff times the print; and a document stating no print - the table absent, or its row
    blank as the record leaves one it has no close for - refuses by name, naming the deal, the
    table and the day.

    Killing mutations: the print unread, today's spot pricing the call at nothing; a blank row read
    as a print.
    """
    base = EXPIRY + pd.offsets.BDay(1)
    fixed = [[EXPIRY, PRINT]]
    marked = marks([dict(VANILLA, Price_Fixing=fixed),
                    dict(VANILLA, Reference='PUT', Option_Type='Put', Price_Fixing=fixed),
                    dict(BINARY, Price_Fixing=fixed),
                    dict(BINARY, Reference='ASSET', Payoff_Style='Asset', Price_Fixing=fixed)], base)
    d = df((SETTLE - base).days)
    assert marked['EQO'] == pytest.approx(UNITS * (PRINT - STRIKE) * d, rel=1e-9)
    assert marked['PUT'] == 0.0
    assert marked['EQB'] == pytest.approx(CASH * d, rel=1e-9)
    assert marked['ASSET'] == pytest.approx(CASH * PRINT * d, rel=1e-9)
    for unstated in (VANILLA, dict(VANILLA, Price_Fixing=[[EXPIRY, '']])):
        with pytest.raises(Exception) as refusal:
            marks([unstated], base)
        message = str(refusal.value)
        assert all(word in message for word in ('EQO', 'Price_Fixing', '2025-06-27')), message


def test_an_fx_option_expired_before_the_base_date_marks_on_its_print_and_a_missing_one_refuses():
    """The FX option and binary declare the `Price_Fixing` table the equity ones do, filled by the
    record's prints of the pair: on a base date between the expiry and the delivery the vanilla
    marks its print's intrinsic discounted to the delivery, exact, the cash binary its payoff
    discounted, and one stating no print refuses by name, naming the deal, the table and the day.

    Killing mutations: the table undeclared, under which every expired FX option refuses; the
    print unread, today's spot pricing the call at nothing.
    """
    base, print_ = EXPIRY + pd.offsets.BDay(1), 1.40
    fixed = [[EXPIRY, print_]]
    binary = {'Object': 'FXBinaryOption', 'Reference': 'FXB', 'Currency': 'USD',
              'Underlying_Currency': 'EUR', 'Discount_Rate': 'USD', 'FX_Volatility': 'EUR.USD',
              'Buy_Sell': 'Buy', 'Option_Type': 'Call', 'Strike_Price': FX_SPOT, 'Payoff': CASH,
              'Expiry_Date': EXPIRY, 'Delivery_Date': SETTLE}
    marked = marks([dict(FX, Delivery_Date=SETTLE, Price_Fixing=fixed),
                    dict(FX, Reference='PUT', Option_Type='Put', Delivery_Date=SETTLE,
                         Price_Fixing=fixed),
                    dict(binary, Price_Fixing=fixed)], base)
    d = df((SETTLE - base).days)
    assert marked['FXO'] == pytest.approx(UNITS * (print_ - FX_SPOT) * d, rel=1e-9)
    assert marked['PUT'] == 0.0
    assert marked['FXB'] == pytest.approx(CASH * d, rel=1e-9)
    for unstated in (dict(FX, Delivery_Date=SETTLE),
                     dict(FX, Delivery_Date=SETTLE, Price_Fixing=[[EXPIRY, '']])):
        with pytest.raises(Exception) as refusal:
            marks([unstated], base)
        message = str(refusal.value)
        assert all(word in message for word in ('FXO', 'Price_Fixing', '2025-06-27')), message


def test_on_the_expiry_the_engine_fixes_the_level_and_before_it_the_lag_only_moves_the_discount():
    """A base date ON the expiry prices off the engine's own spot - the forward between spot dates
    is the spot itself at zero tenor, intrinsic - and reads no print, as a fixing dated today
    follows the factor; before the expiry the lagged option is Black on the forward between the
    spot dates, as far apart here as the two Fridays, discounted to the settlement with the vol
    tenor at the expiry, exact.

    Killing mutation: the forward carried from the valuation date, 100.041 on its own expiry.
    """
    carry = (SETTLE - EXPIRY).days
    marked = marks([dict(VANILLA, Strike_Price=90.0),
                    dict(VANILLA, Reference='PRINTED', Strike_Price=90.0, Price_Fixing=[[EXPIRY, PRINT]])],
                   EXPIRY)
    intrinsic = UNITS * (SPOT - 90.0) * df(carry)
    assert marked['EQO'] == pytest.approx(intrinsic, rel=1e-9)
    assert marked['PRINTED'] == marked['EQO']
    t, t_settle = (EXPIRY - BASE).days / 365.0, (SETTLE - BASE).days / 365.0
    forward, sd = SPOT * math.exp((R_USD - Q_EQ) * t), SIGMA * math.sqrt(t)
    d1 = math.log(forward / STRIKE) / sd + 0.5 * sd
    black = forward * ndtr(d1) - STRIKE * ndtr(d1 - sd)
    assert marks([VANILLA])['EQO'] == pytest.approx(UNITS * black * math.exp(-R_USD * t_settle), rel=1e-9)


def test_an_fx_option_pays_on_the_delivery_date_its_family_declares():
    """The FX option settles on `Delivery_Date`, the field its family declares it settles on, stated
    on its expiry or ten days on: Black on the forward between spot dates, as far apart here as the
    two Fridays, discounted to the delivery, exact. A `Settlement_Date` written on it declares nothing
    and is not read.

    Killing mutation: the settlement read off a `Settlement_Date` the type does not declare.
    """
    delivery = EXPIRY + pd.DateOffset(days=10)
    marked = marks([FX, dict(FX, Reference='UNDECLARED', Settlement_Date=delivery),
                    dict(FX, Reference='DELIVERED', Delivery_Date=delivery)])
    t = (EXPIRY - BASE).days / 365.0
    forward, sd = FX_SPOT * math.exp((R_USD - R_EUR) * t), FX_SIGMA * math.sqrt(t)
    d1 = math.log(forward / FX_SPOT) / sd + 0.5 * sd
    black = UNITS * (forward * ndtr(d1) - FX_SPOT * ndtr(d1 - sd))
    assert marked['FXO'] == pytest.approx(black * df((EXPIRY - BASE).days), rel=1e-9)
    assert marked['UNDECLARED'] == marked['FXO']
    assert marked['DELIVERED'] == pytest.approx(black * df((delivery - BASE).days), rel=1e-9)


#: A bought up-and-in and up-and-out call on the world's pair, 10% above its spot, each also with a
#: unit rebate
BARRIER = dict(FX, Object='FXBarrierOption', Barrier_Price=FX_SPOT * 1.10, Cash_Rebate=0.0)
KNOCKS = {'KI': dict(BARRIER, Reference='KI', Barrier_Type='Up_And_In'),
          'KO': dict(BARRIER, Reference='KO', Barrier_Type='Up_And_Out'),
          'KI_REBATE': dict(BARRIER, Reference='KI_REBATE', Barrier_Type='Up_And_In', Cash_Rebate=1.0),
          'KO_REBATE': dict(BARRIER, Reference='KO_REBATE', Barrier_Type='Up_And_Out', Cash_Rebate=1.0)}


def test_a_barrier_pays_on_the_delivery_date_its_family_declares():
    """`Delivery_Date` stated on all three, a knock-in plus a knock-out is the vanilla, and each
    barrier its twin delivered on the expiry discounted from the expiry to the delivery, a touch's
    rebate aside.

    Killing mutation: the barrier discounting to the expiry, the pair 4.4e-4 above the vanilla.
    """
    delivered = {reference: dict(deal, Reference=reference + '_D', Delivery_Date=SETTLE)
                 for reference, deal in KNOCKS.items()}
    marked = marks([dict(FX, Delivery_Date=SETTLE), *KNOCKS.values(), *delivered.values()])
    assert marked['KI_D'] > 0.0 and marked['KO_D'] > 0.0, 'the barrier is never reached'
    assert marked['KI_D'] + marked['KO_D'] == pytest.approx(marked['FXO'], rel=1e-9)
    for reference in ('KI', 'KO', 'KI_REBATE'):
        assert marked[reference + '_D'] == pytest.approx(
            marked[reference] * df((SETTLE - EXPIRY).days), rel=1e-12), reference
    touch = marked['KO_REBATE'] - marked['KO']
    assert touch > 0.0 and marked['KO_REBATE_D'] == pytest.approx(marked['KO_D'] + touch, rel=1e-12)


#: A start-window barrier to six months on, a one-touch at 1.40 paid at expiry or at the touch, the
#: no-touch, and a quarterly-averaged Asian, each on the world's pair and expiry
PARTIAL = dict(BARRIER, Object='FXPartialTimeBarrierOption', Payoff_Currency='USD', Barrier_At_Start='Yes',
               Barrier_Limit_Date=BASE + pd.DateOffset(months=6))
TOUCH = {'Currency': 'USD', 'Payoff_Currency': 'USD', 'Discount_Rate': 'USD', 'Buy_Sell': 'Buy',
         'Underlying_Currency': 'EUR', 'FX_Volatility': 'EUR.USD', 'Cash_Payoff': CASH,
         'Expiry_Date': EXPIRY, 'Delivery_Date': EXPIRY, 'Barrier_Type': 'Up', 'Barrier_Price': 1.40}
PAID = {'PKO': dict(PARTIAL, Reference='PKO', Barrier_Type='Up_And_Out'),
        'PKI_REBATE': dict(PARTIAL, Reference='PKI_REBATE', Barrier_Type='Up_And_In', Cash_Rebate=1.0),
        'PKO_REBATE': dict(PARTIAL, Reference='PKO_REBATE', Barrier_Type='Up_And_Out', Cash_Rebate=1.0),
        'OT': dict(TOUCH, Object='FXOneTouchOption', Reference='OT', Payment_Timing='Expiry'),
        'OT_TOUCH': dict(TOUCH, Object='FXOneTouchOption', Reference='OT_TOUCH', Payment_Timing='Touch'),
        'NT': dict(TOUCH, Object='FXNoTouchOption', Reference='NT'),
        'ASIAN': dict(FX, Object='FXDiscreteExplicitAsianOption', Reference='ASIAN', Sampling_Data=[
            [BASE + pd.DateOffset(months=m), 0.0, 1.0] for m in (3, 6, 9)] + [[EXPIRY, 0.0, 1.0]])}


def test_an_fx_option_paid_at_expiry_pays_on_the_delivery_date_its_family_declares():
    """`Delivery_Date` stated, each partial barrier, one-touch and no-touch paid at expiry and Asian
    marks its twin delivered on the expiry discounted from the expiry to the delivery; a touch's
    payment is not.

    Killing mutation: `carried_to_settlement` answering none, every one at its twin.
    """
    marked = marks([*PAID.values(), *(dict(deal, Reference=name + '_D', Delivery_Date=SETTLE)
                                      for name, deal in PAID.items())])
    for name in ('PKO', 'PKI_REBATE', 'OT', 'NT', 'ASIAN'):
        assert marked[name + '_D'] == pytest.approx(marked[name] * df((SETTLE - EXPIRY).days), rel=1e-12), name
    assert marked['OT_TOUCH_D'] == marked['OT_TOUCH'] > marked['OT'] > 0.0
    touch = marked['PKO_REBATE'] - marked['PKO']
    assert touch > 0.0 and marked['PKO_REBATE_D'] == pytest.approx(marked['PKO_D'] + touch, rel=1e-12)


#: Quarterly fixings each paid three days on, a year out, and a strip on them
ROWS = [[BASE + pd.DateOffset(months=m), BASE + pd.DateOffset(months=m, days=3), 0.0] for m in (3, 6, 9, 12)]
STRIP = dict({k: v for k, v in FX.items() if k not in ('Expiry_Date', 'Delivery_Date')},
             LeverageNotional=2.0 * UNITS, Strike_Price=FX_SPOT * 0.98)
#: An FX option of each type paying on `Delivery_Date`, delivered on its expiry
DELIVERING = {
    'FXO': FX, 'FXE': dict(FX, Object='FXEuropeanOption', Reference='FXE'),
    'FXB': dict({k: v for k, v in FX.items() if k != 'Underlying_Amount'}, Object='FXBinaryOption',
                Reference='FXB', Payoff=CASH),
    'KI': KNOCKS['KI'], 'PKO': PAID['PKO'], 'OT': PAID['OT'], 'NT': PAID['NT'], 'ASIAN': PAID['ASIAN'],
    'DASIAN': dict(FX, Object='FXDiscreteExplicitDoubleAsianOption', Reference='DASIAN', Strike_Price=0.01,
                   Sampling_Data_1=[[BASE + pd.DateOffset(months=m), 0.0, 1.0] for m in (9, 10, 11, 12)],
                   Sampling_Data_2=[[BASE + pd.DateOffset(months=m), 0.0, 1.0] for m in (1, 2, 3, 4)])}
#: Each strip settling on its own schedule, which declares `Delivery_Date` and never reads it
STRIPS = {
    'TARF': dict(STRIP, Object='FXTARFOptionDeal', Reference='TARF', TargetLevel=0.1, TARF_ExpiryDates=ROWS,
                 Expiry_Date=ROWS[-1][1]),
    'ACCU': dict(STRIP, Object='FXAccumulatorOptionDeal', Reference='ACCU', Barrier_Type='Up_And_Out',
                 Barrier_Price=FX_SPOT * 1.10, Accumulator_ExpiryDates=ROWS),
    'EXT': dict({k: v for k, v in STRIP.items() if k != 'LeverageNotional'}, Object='FXExtendableForwardDeal',
                Reference='EXT', Extension_Strike=FX_SPOT * 1.02, Extension_Date=ROWS[1][0], Extension_Style='Strip',
                Extendable_ExpiryDates=[[fixing, paid, None, None] for fixing, paid, _ in ROWS]),
    'RANGE': {'Object': 'FXRangeAccrualDeal', 'Reference': 'RANGE', 'Currency': 'USD', 'Underlying_Currency': 'EUR',
              'Discount_Rate': 'USD', 'FX_Volatility': 'EUR.USD', 'Buy_Sell': 'Buy', 'Lower_Barrier': 1.15,
              'Upper_Barrier': 1.35, 'Payoff': CASH, 'Settlement_Date': ROWS[-1][1],
              'Observation_Dates': [[fixing, None] for fixing, *_ in ROWS]}}


def blank(deal):
    """`deal` leaving `Delivery_Date` out."""
    return {k: v for k, v in deal.items() if k != 'Delivery_Date'}


@pytest.mark.parametrize('name', sorted(DELIVERING) + sorted(STRIPS))
def test_validate_names_a_blank_delivery_date_on_each_fx_option_paid_on_it(name):
    """`Delivery_Date` is the day the payoff is paid, a term the author states: `validate` names it,
    left out or written blank, on each FX option type paying on it, and says nothing of it on a strip
    settling on its own schedule, which declares it and never reads it.

    Killing mutation: the declaration a convention again (`default=''`, `convention=True`), every
    blank passed over.
    """
    for unstated in (blank(dict(DELIVERING, **STRIPS)[name]), dict(dict(DELIVERING, **STRIPS)[name], Delivery_Date='')):
        said = world([unstated]).validate()['deals'].get(name, [])
        assert [m for m in said if 'Delivery_Date' in m] == ['Delivery_Date is required'] * (name in DELIVERING), said


def test_a_blank_delivery_date_prices_nothing_and_a_strip_never_reads_it(caplog):
    """A run prices no FX option leaving `Delivery_Date` blank: the day its payoff is paid is
    required, so its compile skips each, counted once and logged in the sentence naming the option
    and the field, the book priced without them - under a base valuation and a credit Monte Carlo of
    the equity vanilla beside them alike. Each strip, whose blank is its convention, marks to the
    bit with it blank as stated on any day.

    Killing mutations: the run judging every deal's unstated fields (`Config.unstated()` whole in
    `Calculation.parameters`), which refuses the book; the compile's own judgement dropped, each
    option skipped on the `ValueError` of `max` over nothing.
    """
    skipped = ['{0} {1} {1}: Delivery_Date is required - Skipped'.format(deal['Object'], name)
               for name, deal in DELIVERING.items()]
    strips = [*STRIPS.values(), *(dict(deal, Reference=name + '_D', Delivery_Date=SETTLE) for name, deal in STRIPS.items())]
    for calc, deals in (({'MCMC_Simulations': 256}, strips), (CMC, [VANILLA])):
        caplog.clear()
        out = world([blank(deal) for deal in DELIVERING.values()] + deals, calc=calc).run_job()[1]
        logged = [record.getMessage() for record in caplog.records]
        assert out['Stats']['Deals Skipped'] == len(DELIVERING) and all(line in logged for line in skipped), logged
        if deals is strips:
            marked = dict(zip(out['Results']['mtm']['Reference'], out['Results']['mtm']['Value'].astype(float)))
            assert not set(DELIVERING).intersection(marked), marked
            assert all(marked[name] != 0.0 and marked[name].hex() == marked[name + '_D'].hex() for name in STRIPS), marked


def held(reference, deals):
    """`deals` in an uncollateralised netting set of their own."""
    return {'Object': 'NettingCollateralSet', 'Reference': reference, 'Netted': 'True',
            'Collateralized': 'False', 'Children': deals}


def by_set(sets, curves, **simulated):
    """One float64 credit Monte Carlo of the declared-defaults world over `sets`, each a netting set
    beneath the one `book.simulated` holds them in, its cash generated: `(dates, {set: its USD
    ledger}, {reference: profile})`."""
    calc, out = book.simulated(sets, curves, prec=torch.float64, Generate_Cashflows='Yes', **simulated)
    sets = calc.netting_sets.sub_structures[0].sub_structures
    priced = [x.obj for s in sets for x in s.sub_structures] + [d for s in sets for d in s.dependencies]
    ledgers = {s.obj.Instrument.field['Reference']: pd.concat(s.obj.Calc_res['cashflows']['USD'], axis=1)
               for s in sets}
    profiles = {x.Instrument.field['Reference']: np.concatenate(x.Calc_res['Value'], axis=-1)
                for x in priced if (x.Calc_res or {}).get('Value')}
    return list(out['Results']['mtm'].index), ledgers, profiles


def valued_on(job, base):
    """`{reference: mark}` of the job document `job` valued on `base`, its market's date moved too."""
    job = json.loads(json.dumps(job, cls=CustomJsonEncoder))
    day = {'.Timestamp': base.strftime('%Y-%m-%d')}
    job['Calc']['Calculation']['Base_Date'] = day
    job['Calc']['MergeMarketData']['ExplicitMarketData']['System Parameters']['Base_Date'] = day
    context = derivus.Context()
    context.load_json((json.dumps(job), 'valued'))
    frame = context.run_job()[1]['Results']['mtm']
    return dict(zip(frame['Reference'], frame['Value'].astype(float)))


#: The declared-defaults world's FX options on its expiry, each delivered forty days after it, past
#: the next monthly row: up barriers at 1.375 and a 50 rebate, start-window ones to six months on
#: and an end-window one from there, a one-touch at 1.35 paid at expiry and one paid at the touch,
#: the no-touch, and a quarterly Asian
WORLD_EXPIRY = book.WORLD_EXPIRY
DELIVERED = WORLD_EXPIRY + pd.DateOffset(days=40)
UP = {'Expiry_Date': WORLD_EXPIRY, 'Barrier_Price': 1.375}
WINDOW = dict(UP, Barrier_At_Start='Yes', Barrier_Limit_Date=book.WORLD_BASE + pd.DateOffset(months=6))
TOUCHES = {'Currency': 'USD', 'Underlying_Currency': 'EUR', 'Discount_Rate': 'USD', 'FX_Volatility': 'EUR.USD',
           'Buy_Sell': 'Buy', 'Expiry_Date': WORLD_EXPIRY, 'Delivery_Date': WORLD_EXPIRY, 'Cash_Payoff': 10_000.0,
           'Barrier_Type': 'Up', 'Barrier_Price': 1.35}
SETTLING = {
    'KO': book.fx_leg('FXBarrierOption', 'KO', Barrier_Type='Up_And_Out', **UP),
    'KI': book.fx_leg('FXBarrierOption', 'KI', Barrier_Type='Up_And_In', **UP),
    'KOR': book.fx_leg('FXBarrierOption', 'KOR', Barrier_Type='Up_And_Out', Cash_Rebate=50.0, **UP),
    'PKO': book.fx_leg('FXPartialTimeBarrierOption', 'PKO', Barrier_Type='Up_And_Out', **WINDOW),
    'PKOE': book.fx_leg('FXPartialTimeBarrierOption', 'PKOE', Barrier_Type='Up_And_Out',
                        **dict(WINDOW, Barrier_At_Start='No')),
    'PKOR': book.fx_leg('FXPartialTimeBarrierOption', 'PKOR', Barrier_Type='Up_And_Out', Cash_Rebate=50.0,
                        **WINDOW),
    'PKIR': book.fx_leg('FXPartialTimeBarrierOption', 'PKIR', Barrier_Type='Up_And_In', Cash_Rebate=50.0,
                        **WINDOW),
    'OT': dict(TOUCHES, Object='FXOneTouchOption', Reference='OT'),
    'OTT': dict(TOUCHES, Object='FXOneTouchOption', Reference='OTT', Payment_Timing='Touch'),
    'NT': dict(TOUCHES, Object='FXNoTouchOption', Reference='NT'),
    'ASN': book.fx_leg('FXDiscreteExplicitAsianOption', 'ASN', Expiry_Date=WORLD_EXPIRY, Sampling_Data=[
        [book.WORLD_BASE + pd.DateOffset(months=m), 0.0, 1.0] for m in (3, 6, 9, 12)])}
#: Each settles where `SETTLING` does, on its expiry, and forty days after it
TWINS = {'': {}, '_D': {'Delivery_Date': DELIVERED}}
#: EURUSD a GBM at the surface's 15%
EURUSD = {'FxRate.EUR': ('GBMAssetPriceModel', {'Vol': book.SIGMA, 'Drift': 0.0})}


def units():
    """A unit paid on the world's expiry and one on the delivery, each in a set of its own."""
    return [held(name, [{'Object': 'FixedCashflowDeal', 'Reference': name, 'Currency': 'USD', 'Discount_Rate': 'USD',
                         'Amount': 1.0, 'Payment_Date': day}])
            for name, day in (('AT_EXPIRY', WORLD_EXPIRY), ('AT_DELIVERY', DELIVERED))]


@functools.lru_cache(maxsize=None)
def delivered():
    """One float64 credit Monte Carlo on a monthly grid, `EURUSD` and a Hull-White on USD, of each of
    `SETTLING` in each of `TWINS` beside the two `units`, every one in a netting set of its own."""
    return by_set([held(name + tag, [dict(deal, Reference=name + tag, **stated)])
                   for name, deal in SETTLING.items() for tag, stated in TWINS.items()] + units(),
                  ('USD',), models=EURUSD, Time_Grid='0d 1m(1m)')


def delivered_identities(dates, ledgers, profiles, names, expiry, paid_on, exact, rtol=1e-13):
    """Each of `names` paid on `paid_on`, against its twin delivered on `expiry`, on one run of
    a monthly grid, read on the rows both price - the months, the expiry and `paid_on`, a row only
    another deal adds reading each interpolated: every row to the expiry the twin's times that row's
    forward discount between the two days, a unit on each read path by path; every row after it the
    payoff booked on `paid_on` discounted to that row, the unit paid there; nothing booked on the
    expiry; and the payoff booked whole, path by path the twin's on its expiry - to the bit where
    `exact` names it."""
    months = {dates[0] + pd.DateOffset(months=n) for n in range(len(dates))} | {expiry, paid_on}
    last, priced = dates.index(expiry), [row for row, day in enumerate(dates) if day in months]
    before, after = [row for row in priced if row <= last], [row for row in priced if expiry < dates[row] <= paid_on]
    assert len(before) > 3 and len(after) > 1 and after[-1] == dates.index(paid_on), [dates[row] for row in after]
    unit = profiles['AT_DELIVERY']
    forward = unit[before] / profiles['AT_EXPIRY'][before]
    for name in names:
        blank, late = profiles[name], profiles[name + '_D']
        assert np.allclose(late[before], blank[before] * forward, rtol=rtol, atol=0.0), name
        paid = np.asarray(ledgers[name + '_D'].loc[paid_on].values, dtype=float)
        owed = np.asarray(ledgers[name].loc[expiry].values, dtype=float)
        assert paid.any() and paid.min() < paid.max(), (name, 'a payoff varying across the paths')
        assert np.array_equal(paid, owed) if name in exact else np.allclose(paid, owed, rtol=1e-12, atol=0.0), (
            name, np.abs(paid - owed).max())
        assert not np.asarray(ledgers[name + '_D'].loc[expiry].values, dtype=float).any(), name
        for row in after:
            assert late[row].any() and np.allclose(late[row], paid * unit[row], rtol=1e-13, atol=0.0), (
                name, dates[row])


def test_delivered_after_its_expiry_an_fx_option_holds_what_the_expiry_fixed_and_books_it_on_the_delivery():
    """Delivered forty days after its expiry, under a float64 credit Monte Carlo with a Hull-White on
    USD, each barrier, start-window and end-window barrier, one-touch and no-touch paid at expiry and the Asian,
    read on the monthly rows, the expiry and the delivery: to the expiry every row is its twin's
    delivered on the expiry times that row's own forward discount from the expiry to the delivery, read off a unit
    paid on each, to 1e-13 - the closed form carries the same two curve reads in another order; the
    monthly row after the expiry and the delivery's are not zero and hold the payoff the delivery
    books, discounted to them as the unit paid there is; nothing is booked on the expiry; and the
    delivery books the payoff whole, path by path that twin's on its expiry - to the bit but for
    the Asian, within 1e-12, whose expiry row reads its last sample as a forward at no tenor where
    the rows after it read it fixed.

    Killing mutation: the cash left on the expiry row, each pricer booking its expiry row's mark there
    (`pv_barrier_option`, `pv_partial_barrier_option`, `pv_one_touch_option`,
    `pv_discrete_asian_option`); or the end-window barrier still watched on the rows past its expiry.
    """
    dates, ledgers, profiles = delivered()
    delivered_identities(dates, ledgers, profiles, ('KO', 'KI', 'PKO', 'PKIR', 'OT', 'NT', 'ASN'), WORLD_EXPIRY,
                         DELIVERED, exact=('KO', 'KI', 'PKO', 'PKIR', 'OT', 'NT'))
    # the end-window barrier's closed form is ill-conditioned on the day its window opens, 1.6e-11 there
    delivered_identities(dates, ledgers, profiles, ('PKOE',), WORLD_EXPIRY, DELIVERED, ('PKOE',), rtol=1e-10)


def test_a_knock_out_s_rebate_and_a_one_touch_paid_at_the_touch_pay_when_they_pay():
    """Delivered forty days after the expiry, a knock-out and a start-window knock-out paying a 50
    rebate at the touch book every rebate before the expiry to the bit as their twins delivered on the
    expiry do, the knock-out the one its last interval touches on the expiry itself, on some paths
    and not others, and the delivery the rest: the expiry's and the delivery's cash sum to that
    twin's on the expiry to 1e-13. A one-touch paid at the touch reads no delivery, its rows and cash
    that twin's to the bit.

    Killing mutations: the knock-out's rebate on the expiry row left unbooked (`pv_barrier_option`
    booking it only before the expiry), short by that interval's touch on those paths; the one-touch
    paid at the touch registering its delivery (`TouchLifecycle.reset`), its rows running on to it.
    """
    dates, ledgers, profiles = delivered()
    for name in ('KOR', 'PKOR'):
        blank, late = ledgers[name], ledgers[name + '_D']
        before = [day for day in blank.index if day < WORLD_EXPIRY]
        assert np.array_equal(late.loc[before].values, blank.loc[before].values), name
        assert np.asarray(late.loc[before].values, dtype=float).any(), name
        on_expiry, paid = (np.asarray(late.loc[day].values, dtype=float) for day in (WORLD_EXPIRY, DELIVERED))
        owed = np.asarray(blank.loc[WORLD_EXPIRY].values, dtype=float)
        assert np.allclose(on_expiry + paid, owed, rtol=1e-13, atol=1e-13) and paid.any(), name
    touched = np.asarray(ledgers['KOR_D'].loc[WORLD_EXPIRY].values, dtype=float)
    assert (touched > 0.0).any() and (touched == 0.0).any(), 'touched in the last interval on some paths'
    assert np.array_equal(profiles['OTT_D'], profiles['OTT'])
    assert np.array_equal(ledgers['OTT_D'].values, ledgers['OTT'].values)
    assert np.asarray(ledgers['OTT'].values, dtype=float).any()


def test_between_its_expiry_and_its_delivery_an_fx_option_marks_what_the_expiry_fixed():
    """On a base date between the expiry and the delivery, the knock-out and the start-window
    knock-out mark the print `Price_Fixing` states, 1.30 inside the barrier, its intrinsic discounted
    to the delivery, the Asian its printed samples' average's, to 1e-12. No touch before the base
    date is known - nothing states one, as a seasoned barrier states none - so the knock-in marks
    the rebate it pays untouched, the one-touch paid at expiry nothing and the no-touch its payoff,
    each discounted, neither touch reading a print. A barrier stating no print refuses by name,
    naming the deal, the table and the day.

    Killing mutation: the print not read, each barrier's level today's spot at the expiry's zero
    tenor (`held_level` in `pv_barrier_option`, `held_past_expiry` in `pv_partial_barrier_option`),
    both knock-outs at nothing.
    """
    base, printed = EXPIRY + pd.offsets.BDay(1), 1.30
    fixed, d = [[EXPIRY, printed]], df((SETTLE - base).days)
    samples = [[day, 1.28 + 0.02 * n, weight] for n, (day, _, weight) in enumerate(PAID['ASIAN']['Sampling_Data'])]
    barriers = [dict(KNOCKS['KO'], Delivery_Date=SETTLE), dict(KNOCKS['KI_REBATE'], Delivery_Date=SETTLE),
                dict(PAID['PKO'], Delivery_Date=SETTLE)]
    marked = marks([dict(deal, Price_Fixing=fixed) for deal in barriers] + [
        dict(PAID[name], Delivery_Date=SETTLE) for name in ('OT', 'NT')] + [
        dict(PAID['ASIAN'], Delivery_Date=SETTLE, Sampling_Data=samples)], base)
    intrinsic = UNITS * (printed - FX_SPOT) * d
    assert marked['KO'] == pytest.approx(intrinsic, rel=1e-12)
    assert marked['PKO'] == pytest.approx(intrinsic, rel=1e-12)
    assert marked['KI_REBATE'] == pytest.approx(1.0 * d, rel=1e-12)
    assert marked['OT'] == 0.0 and marked['NT'] == pytest.approx(CASH * d, rel=1e-12)
    assert marked['ASIAN'] == pytest.approx(UNITS * (1.31 - FX_SPOT) * d, rel=1e-12)
    for unstated in barriers:
        with pytest.raises(Exception) as refusal:
            marks([unstated], base)
        message = str(refusal.value)
        assert all(word in message for word in (unstated['Reference'], 'Price_Fixing', '2025-06-27')), message


def test_a_forward_carries_between_spot_dates():
    """An equity or FX option's forward is the spot carried from the valuation date's spot date to
    the expiry's, each the underlying factor's `Settlement_Days` business days on: three where the
    share states none, two where the pair's underlying rate states none, and the number the factor
    states, a T+1 pair's rate stating one. Read off deals deep in the money, whose mark is the
    forward less the strike, discounted: from a Friday to a Wednesday three days on are five
    calendar days from each and two are four and two, so three carry the tenor and two carry two
    days fewer; to a Thursday one day on is three and one, two days fewer, where two carry the
    tenor. Under a credit Monte Carlo each row starts at its own spot date, three, four or five
    calendar days on over two daily weeks: against the share stating no lag, path by path at
    float32. An equity swaplet list reads the share's lag on the share's calendar, never on its own
    `Accrual_Calendars`: paying the level a Monday fixes, its start fixed or ahead, it carries two
    days fewer than the tenor at the share's three, one fewer where the Tuesday is a holiday of the
    share's calendar, and the tenor where the share states none; no lag is no roll, a fixing on the
    holiday carrying to it; and on a principal with both ends ahead it pays the ratio of two forwards, each
    to its own spot date. The equity forward deal and an FX start-window knock-out no path reaches,
    the vanilla, read the same forward. Paid at a Saturday expiry a year on, an FX one-touch and
    no-touch and an equity no-touch meet the textbook one-touch at the carry between spot dates; a
    compo swaplet list paid on a Monday its composite forward, its fx conversion on the share's
    lag, reported at EUR's price today; and a swap autocall with one coupon on that Monday and a
    floating leg paying nothing the cash digital.

    Killing mutations: the factor's stated lag unread, its class's read in its place, the share
    stating two 1.6e-4 high; the carry started at the valuation date, the equity forward 4.1e-4
    high; a default of two on the share, 1.6e-4 low; every row given the base date's lag, the
    Monday's row 1.6e-4 low; the list's lag unread, its forward 1.6e-4 high; the share's calendar
    unread, 8.2e-5 low; the list's `Accrual_Calendars` counted in its place, 8.2e-5 high; a lag of zero days rolled where `Deal.spot_lag` returns None, the daily grid's
    Saturday row red; a period's start carried to its day, the ratio 4.1e-4 high. A `Spot_Lag`
    unread: the forward deal's, at the share's two 1.6e-4 high; the partial barrier's, 1.1e-4 high;
    the FX touch's, the one-touch 1.8e-4 high and the no-touch 3.0e-4 low; the equity touch's, the
    no-touch 1.8e-4 high; the autocall's, the swap 5.3e-4 high. The compo list's fx carry handed
    days already shifted, 2.7e-4 low.
    """
    midweek = EXPIRY - pd.DateOffset(days=2)
    equity, pair = (SPOT, R_USD - Q_EQ, 3, 'EQ'), (FX_SPOT, R_USD - R_EUR, 2, 'EUR')

    def options(expiry):
        terms = {'Expiry_Date': expiry, 'Strike_Price': 0.01}
        return [
            (dict(VANILLA, Settlement_Date=expiry, **terms), UNITS, 0.01) + equity,
            (dict(BINARY, Settlement_Date=expiry, Payoff_Style='Asset', **terms), CASH, 0.0) + equity,
            (dict(FX, Delivery_Date=expiry, **terms), UNITS, 0.01) + pair,
            (dict({k: v for k, v in FX.items() if k != 'Underlying_Amount'}, Object='FXBinaryOption',
                  Reference='FXB', Payoff=CASH, Payoff_Style='Asset', Delivery_Date=expiry, **terms), CASH, 0.0) + pair,
            ({'Object': 'EquityForwardDeal', 'Reference': 'EQF', 'Currency': 'USD', 'Equity': 'EQ',
              'Discount_Rate': 'USD', 'Buy_Sell': 'Buy', 'Forward_Price': 0.01, 'Units': UNITS,
              'Maturity_Date': expiry}, UNITS, 0.01) + equity,
            # a start-window knock-out at a barrier no path reaches is the vanilla
            (dict(FX, Object='FXPartialTimeBarrierOption', Reference='FXP', Payoff_Currency='USD',
                  Barrier_Type='Down_And_Out', Barrier_Price=0.001, Barrier_At_Start='Yes',
                  Barrier_Limit_Date=BASE + pd.DateOffset(days=90), Delivery_Date=expiry, **terms), UNITS, 0.01) + pair]

    # a Thursday parts one day from two, which from a Friday to a Wednesday carry alike
    for expiry, stated in ((midweek, {}), (midweek, {'EQ': 2, 'EUR': 3}),
                           (midweek + pd.DateOffset(days=1), {'EQ': 1, 'EUR': 1})):
        tenor, booked = (expiry - BASE).days, options(expiry)
        marked = marks([deal for deal, *_ in booked], factors=lagged(**stated))
        for deal, scale, strike, spot, carry, declared, name in booked:
            days = stated.get(name, declared)
            between = tenor + lag(expiry, days) - lag(BASE, days)
            assert marked[deal['Reference']] / (scale * df(tenor)) + strike == pytest.approx(
                spot * math.exp(carry * between / 365.0), rel=1e-12), (deal['Reference'], days)
    deep = options(midweek)[0][0]
    spot_dated, flat = (run([deep], calc=dict(CMC, Time_grid='0d 1d(1d) 2w'), factors=factors)['mtm']
                        for factors in ({}, lagged(EQ=0)))
    assert {lag(day, 3) for day in spot_dated.index} == {3, 4, 5}
    for day in spot_dated.index:
        # the strike is a ten-thousandth of the spot, so the marks' ratio is the forwards'
        ratio = np.asarray(spot_dated.loc[day].values, dtype=float) / np.asarray(flat.loc[day].values, dtype=float)
        stretch = math.exp((R_USD - Q_EQ) * (lag(midweek, 3) - lag(day, 3)) / 365.0)
        assert np.allclose(ratio, stretch, rtol=2e-6, atol=0.0), day
    holiday = pd.Timestamp('2024-09-24')                 # a Tuesday
    monday, paid = holiday - pd.DateOffset(days=1), holiday + pd.DateOffset(days=3)
    named = dict(FACTORS['EquityPrice.EQ'], Holiday_Calendar=book.CALENDAR)
    tenor, ahead = (monday - BASE).days, BASE + pd.DateOffset(days=7)
    for start in (BASE - pd.DateOffset(days=30), ahead):
        marked = marks([
            swaplet('WEEK', start, monday, paid, Accrual_Calendars=book.CALENDAR),
            swaplet('RATIO', ahead, monday, paid, Amount_Type='Principal')])
        marked.update(marks([swaplet('NAMED', start, monday, paid)], factors={'EquityPrice.EQ': named}))
        marked.update(marks([swaplet('NONE', start, monday, paid), swaplet('HELD', start, holiday, paid)],
                            factors={'EquityPrice.EQ': dict(named, Settlement_Days=0)}))
        for ref, level, between in (
                ('NONE', SPOT, tenor), ('WEEK', SPOT, tenor + lag(monday, 3) - lag(BASE, 3)),
                ('NAMED', SPOT, tenor + lag(monday, 3, [holiday]) - lag(BASE, 3)), ('HELD', SPOT, tenor + 1),
                ('RATIO', 1.0, tenor - 7 + lag(monday, 3) - lag(ahead, 3))):
            assert marked[ref] / (UNITS * df((paid - BASE).days)) == pytest.approx(
                level * math.exp((R_USD - Q_EQ) * between / 365.0), rel=1e-12), (ref, start)
    # imported here, as that module imports this one
    from test_compo_analytic_pricers import _one_touch_closed
    year, monday = BASE + pd.DateOffset(years=1), EXPIRY - pd.DateOffset(days=4)
    years = lambda date, days: ((date - BASE).days + lag(date, days) - lag(BASE, days)) / 365.0
    touch = {'Currency': 'USD', 'Payoff_Currency': 'USD', 'Discount_Rate': 'USD', 'Buy_Sell': 'Buy',
             'Cash_Payoff': CASH, 'Expiry_Date': year, 'Barrier_Monitoring_Frequency': pd.DateOffset(days=0)}
    fx_touch = dict(touch, Underlying_Currency='EUR', FX_Volatility='EUR.USD', Barrier_Type='Up', Barrier_Price=1.35,
                    Delivery_Date=year)
    marked = marks([
        dict(fx_touch, Object='FXOneTouchOption', Reference='FXOT'),
        dict(fx_touch, Object='FXNoTouchOption', Reference='FXNT'),
        dict(touch, Object='EquityNoTouchOption', Reference='EQNT', Equity='EQ', Equity_Volatility='EQ',
             Barrier_Type='Down', Barrier_Price=80.0, Barrier_Dates=[]),
        swaplet('CMP', BASE + pd.DateOffset(days=7), monday, monday, Currency='EUR', Discount_Rate='EUR',
                Payoff_Type='Compo', Equity_Volatility='EQ')])
    up = _one_touch_closed(1.35, True, FX_SPOT, (R_USD - R_EUR) * years(year, 2), FX_SIGMA, R_USD)
    down = _one_touch_closed(80.0, False, SPOT, (R_USD - Q_EQ) * years(year, 3), SIGMA, R_USD)
    composite = UNITS * SPOT * math.exp((R_EUR - Q_EQ) * years(monday, 3) - R_EUR * (monday - BASE).days / 365.0) * (
        today() / FX_SPOT)
    for ref, expected in (('FXOT', up), ('FXNT', CASH * df(365) - up), ('EQNT', CASH * df(365) - down),
                          ('CMP', composite)):
        assert marked[ref] == pytest.approx(expected, rel=1e-12), (ref, marked[ref], expected)
    autocall = dict({k: v for k, v in VANILLA.items() if k != 'Settlement_Date'}, Reference='V2',
                    Object='QEDI_CustomAutoCallSwap_V2', Expiry_Date=monday, Barrier=0.0, Barrier_Dates=[],
                    Price_Fixing=[[monday, 0.0]], Autocall_Coupons=[[monday, 0.05]],
                    Autocall_Thresholds=[[monday, 1.0]], Forecast_Rate='USD',
                    Reset_Frequency=pd.DateOffset(years=1), Autocall_Floating=[[monday, 0.0]])
    t = (monday - BASE).days / 365.0
    d2 = (math.log(SPOT / STRIKE) + (R_USD - Q_EQ) * years(monday, 3) - 0.5 * SIGMA ** 2 * t) / (SIGMA * math.sqrt(t))
    swap = run([autocall], calc={'MCMC_Simulations': 1})['mtm']
    assert float(swap[swap['Reference'] == 'V2']['Value'].iloc[0]) == pytest.approx(
        UNITS * 0.05 * df((monday - BASE).days) * ndtr(d2), rel=1e-12)


#: The pound beside the world's two rates, for a cross on the euro
GBP_SPOT, R_GBP = 1.27, 0.03
GBP = {'FxRate.GBP': {'Domestic_Currency': None, 'Interest_Rate': 'GBP', 'Spot': GBP_SPOT},
       'InterestRate.GBP': {'Currency': 'GBP', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                            'Curve': utils.Curve([], [[0.0, R_GBP], [5.0, R_GBP]])},
       'FXVol.EUR.GBP': FACTORS['FXVol.EUR.USD']}


def test_a_pair_counts_its_spot_date_as_the_market_does():
    """A pair's spot date is read off both its rates, each naming the calendar its own currency
    settles on: the most `Settlement_Days` its currencies but USD state, counted on the business
    days of each, rolled forward onto a business day of both and of USD. Read off FX options deep
    in the money, the forward carried from the base date's spot date to that of the expiry, a
    quiet Wednesday: EUR on TARGET against USD on New York counts past Independence Day 2024, a
    Thursday, so a Wednesday's spot date is the Friday, two days on, and a Tuesday's two TARGET
    days end on the holiday and roll to the Friday; EUR in GBP on London counts past London's 26
    August, a Friday's spot date five days on, and rolls off New York's 2 September, a Thursday's
    five days on; EUR stating one day and GBP two, the cross counts two. The option-quote fit grows
    its forward between the same spot dates, on the pair priced in the base and on the cross. USD's
    own days are never read: an option on USD priced in EUR marks the same, to the bit, whatever USD
    states.

    Killing mutations: USD's calendar counted before the spot date, the Wednesday's forward 1.6e-4
    low; no roll onto the calendars the pair settles on, the Tuesday's 5.5e-5 high, as a pair
    settling off New York's; a cross counted on its underlying's calendar alone, the cross over
    26 August 2.7e-5 high; its underlying's days in place of the most, the cross 5.5e-5 high;
    USD's days read, the inverse option's forward 1.1e-4 low. The fit counting Monday to Friday,
    the Tuesday's 5.5e-5 high; USD's rate left out of the fit's, the cross's Tuesday 2.7e-5 high.
    """
    expiry, inverse = pd.Timestamp('2024-12-11'), pd.Timestamp('2024-10-17')
    calendars = {'FxRate.EUR': 'TARGET (Euro)', 'FxRate.USD': 'New York', 'FxRate.GBP': 'London'}
    named = {name: dict(dict(FACTORS, **GBP)[name], Holiday_Calendar=calendar) for name, calendar in calendars.items()}
    deep = dict(FX, Strike_Price=0.01, Expiry_Date=expiry, Delivery_Date=expiry)
    cross = dict(deep, Reference='CROSS', Currency='GBP', Discount_Rate='GBP', FX_Volatility='EUR.GBP')
    unit = {'Object': 'FixedCashflowDeal', 'Reference': 'UNIT', 'Currency': 'GBP', 'Discount_Rate': 'GBP',
            'Amount': 1.0, 'Payment_Date': expiry}

    def factors(stated):
        return dict(GBP, **{name: dict(block, **stated.get(name, {})) for name, block in named.items()})

    # the base date, the days stated and each pair's lag from the base date and from the expiry
    for base, stated, usd, gbp in (
            ('2024-07-03', {}, (2, 2), (2, 2)), ('2024-07-02', {}, (3, 2), (3, 2)),
            ('2024-08-23', {}, (4, 2), (5, 2)), ('2024-08-29', {}, (5, 2), (5, 2)),
            ('2024-10-17', {'FxRate.EUR': {'Settlement_Days': 1}}, (1, 1), (4, 2))):
        base = pd.Timestamp(base)
        tenor, marked = (expiry - base).days, marks([deep, cross, unit], base, factors(stated))
        # the cross's mark is in GBP times GBP's price today, which the unit paid in GBP divides out
        for forward, spot, carry, (start, end) in (
                (marked['FXO'] / (UNITS * df(tenor)) + 0.01, FX_SPOT, R_USD - R_EUR, usd),
                (fitted(base, factors(stated), expiry, 'EUR'), FX_SPOT, R_USD - R_EUR, usd),
                (marked['CROSS'] / (UNITS * marked['UNIT']) + 0.01, FX_SPOT / GBP_SPOT, R_GBP - R_EUR, gbp),
                (fitted(base, factors(stated), expiry, 'EUR', 'GBP'), FX_SPOT / GBP_SPOT, R_GBP - R_EUR, gbp)):
            assert forward == pytest.approx(spot * math.exp(carry * (tenor + end - start) / 365.0), rel=1e-12), (
                base, spot)
    on_usd = dict(deep, Reference='USD', Underlying_Currency='USD', Currency='EUR', Discount_Rate='EUR')
    readings = {(eur, usd): marks([on_usd], inverse, factors({'FxRate.EUR': {'Settlement_Days': eur},
                                                              'FxRate.USD': {'Settlement_Days': usd}}))['USD']
                for eur, usd in ((1, 1), (1, 3), (2, 1))}
    assert readings[1, 1] == readings[1, 3] != readings[2, 1], readings


#: The forwards' delivery, a Friday half a year on, and the EUR they buy
DELIVERY, NOTIONAL = pd.Timestamp('2024-12-27'), 1e6


def today():
    """EUR's price in USD on the base date: its quote, the price for delivery on its spot date,
    carried back on the world's curves."""
    return FX_SPOT * math.exp(-(R_USD - R_EUR) * lag(BASE, 2) / 365.0)


def fx_forward(reference, strike, **terms):
    """`NOTIONAL` EUR bought for `strike` USD each, delivered on `DELIVERY`."""
    return dict({'Object': 'FXForwardDeal', 'Reference': reference, 'Buy_Currency': 'EUR', 'Buy_Amount': NOTIONAL,
                 'Buy_Discount_Rate': 'EUR', 'Sell_Currency': 'USD', 'Sell_Amount': NOTIONAL * strike,
                 'Sell_Discount_Rate': 'USD', 'Settlement_Date': DELIVERY}, **terms)


def test_an_fx_forward_struck_at_the_outright_from_its_spot_date_is_worth_nothing():
    """A deliverable EUR forward against USD values each currency at its price today, its quote
    carried back from its spot date on the USD curve less the EUR one: struck at the outright its
    curves carry from the base date's spot date, two business days on, to the delivery, it is worth
    nothing to the rounding of its notional; struck at the outright carried from the base date
    itself it loses the carry over the four calendar days from a Friday to its Tuesday spot date,
    exact. Stating `Settlement_Days` 0 the quote is today's price, and that outright is worth
    nothing. Naming a `Holiday_Calendar` whose Tuesday is a holiday, a Friday's spot date is the
    Wednesday, and the outright from it is worth nothing.

    Killing mutations: the cross converting at the quotes (`calc_fx_cross` defaulting to `quoted`),
    the spot-dated forward 271.22 where it is worth nothing; the factor's `Holiday_Calendar` unread,
    the holiday's 68.11.
    """
    tau = (DELIVERY - BASE).days / 365.0
    spot_dated = FX_SPOT * math.exp((R_USD - R_EUR) * (tau - lag(BASE, 2) / 365.0))
    rowed = FX_SPOT * math.exp((R_USD - R_EUR) * tau)
    marked = marks([fx_forward('SPOT', spot_dated), fx_forward('ROW', rowed)])
    assert lag(BASE, 2) == 4 and abs(marked['SPOT']) < 1e-12 * NOTIONAL * FX_SPOT, marked
    assert marked['ROW'] == pytest.approx(NOTIONAL * (spot_dated - rowed) * df((DELIVERY - BASE).days), rel=1e-9)
    assert abs(marks([fx_forward('ROW', rowed)], factors=lagged(EUR=0))['ROW']) < 1e-12 * NOTIONAL * FX_SPOT
    friday, tuesday = pd.Timestamp('2024-09-20'), pd.Timestamp('2024-09-24')
    later = FX_SPOT * math.exp((R_USD - R_EUR) * ((DELIVERY - friday).days - lag(friday, 2, [tuesday])) / 365.0)
    named = {'FxRate.EUR': dict(FACTORS['FxRate.EUR'], Holiday_Calendar=book.CALENDAR)}
    assert lag(friday, 2, [tuesday]) == lag(friday, 2) + 1 == 5
    assert abs(marks([fx_forward('LATER', later)], friday, named)['LATER']) < 1e-12 * NOTIONAL * FX_SPOT


def test_a_non_deliverable_forward_is_worth_its_deliverable_twin():
    """Settled in USD on the delivery date, a EUR non-deliverable forward pays the outright agreed
    for delivery that day, EUR's price today carried on the curves from the base date, less its
    strike: struck at the spot, its value is the deliverable forward's on the same terms, to 1e-12.

    Killing mutation: the forward read as a fixing (no `delivery`), the quote carried from the base
    date rather than its price today, 12,280.72 where it is worth 12,009.49.
    """
    twin = fx_forward('NDF', FX_SPOT, Object='FXNonDeliverableForward', Settlement_Currency='USD',
                      Discount_Rate='USD')
    marked = marks([fx_forward('FWD', FX_SPOT), {k: v for k, v in twin.items() if not k.endswith('_Discount_Rate')}])
    assert marked['FWD'] > 1e4 and marked['NDF'] == pytest.approx(marked['FWD'], rel=1e-12), marked


def test_a_value_reported_in_a_foreign_currency_is_its_value_over_that_currency_s_price_today():
    """Reported in EUR, a USD cashflow's value is its USD one over EUR's price today, the quote
    carried back from its spot date four calendar days on; reported in USD, a EUR cashflow's is its
    EUR one times that price, exact against the curves both ways.

    Killing mutation: the report conversion reading the quote, the USD cashflow in EUR 2.2e-4 low.
    """
    cashflows = [{'Object': 'FixedCashflowDeal', 'Reference': currency, 'Currency': currency, 'Discount_Rate': currency,
                  'Amount': NOTIONAL, 'Payment_Date': DELIVERY} for currency in ('USD', 'EUR')]
    usd, eur = ({reference: float(value) for reference, value in zip(frame['Reference'], frame['Value'])}
                for frame in (run(cashflows, calc={'Currency': currency})['mtm'] for currency in ('USD', 'EUR')))
    local = {currency: NOTIONAL * math.exp(-rate * (DELIVERY - BASE).days / 365.0)
             for currency, rate in (('USD', R_USD), ('EUR', R_EUR))}
    assert usd['USD'] == pytest.approx(local['USD'], rel=1e-12) and eur['EUR'] == pytest.approx(local['EUR'], rel=1e-12)
    assert eur['USD'] == pytest.approx(local['USD'] / today(), rel=1e-12), (eur['USD'], local['USD'] / FX_SPOT)
    assert usd['EUR'] == pytest.approx(local['EUR'] * today(), rel=1e-12), (usd['EUR'], local['EUR'] * FX_SPOT)


@pytest.mark.parametrize('currency, reported', [('EUR', 'USD'), ('USD', 'EUR')])
def test_a_rate_stating_no_curve_converts_on_the_curve_named_as_its_currency(currency, reported):
    """Alone in its job, a cashflow reported in a currency whose `FxRate` states no `Interest_Rate`
    converts at EUR's price today: no deal reaches the reporting currency's curve, and the job
    holds it as the curve named as the currency.

    Killing mutation: discovery linking only a stated `Interest_Rate`, the value converted at the
    quote 2.2e-4 off.
    """
    cashflow = {'Object': 'FixedCashflowDeal', 'Reference': 'CF', 'Currency': currency, 'Discount_Rate': currency,
                'Amount': NOTIONAL, 'Payment_Date': DELIVERY}
    rate = {'FxRate.' + reported: {'Spot': FACTORS['FxRate.' + reported]['Spot']}}
    frame = run([cashflow], calc={'Currency': reported}, factors=rate)['mtm']
    local = NOTIONAL * math.exp(-{'USD': R_USD, 'EUR': R_EUR}[currency] * (DELIVERY - BASE).days / 365.0)
    worth = local * today() if reported == 'USD' else local / today()
    assert dict(zip(frame['Reference'], frame['Value']))['CF'] == pytest.approx(worth, rel=1e-12)


@pytest.mark.parametrize('model', ['GBMAssetPriceTSModelImplied', 'LogVar2FJImpliedSpotModel'])
def test_under_simulation_a_foreign_cashflow_deflates_to_its_value_today_on_every_row(model):
    """Under a credit Monte Carlo on flat curves, EUR simulated at a variance of 1e-12 - the implied
    GBM, and the LogVar2FJ outer at its GBM limit - a EUR cashflow's USD value deflated on the USD
    curve is its value on the base date on every row of a daily grid over a weekend, path by path
    at float32: the quote drifts from each row's spot date to the next's, two, three or four
    calendar days on, and converts at its price today.

    Killing mutation, in either model: the drift read row to row, the Saturday three calendar days
    from its spot date 5.5e-5 high against the base date's four, a Monday's two 1.1e-4.

    A rate is priced in the base whatever `Domestic_Currency` it states: EUR stating EUR has the
    exposure of EUR stating none, to the bit. Killing mutation: the stated currency read again, in
    the quote's carry or in either model's drift, which then carries EUR on its own curve, at nothing.
    """
    # the GBM limit factor is that module's own, and only this gate reads it
    from test_logvar2fj_json import GBM_LIMIT, _curve
    implied = {
        'GBMAssetPriceTSModelImplied': {'GBMAssetPriceTSModelParameters.EUR': {
            'Quanto_FX_Volatility': None, 'Quanto_FX_Correlation': 0.0, 'Vol': _curve([[0.0, 1e-6], [5.0, 1e-6]])}},
        'LogVar2FJImpliedSpotModel': {'LogVar2FJModelParameters.EUR': dict(
            GBM_LIMIT, Xi_Curve=_curve([[0.0, 1e-12], [5.0, 1e-12]]))}}
    cashflow = {'Object': 'FixedCashflowDeal', 'Reference': 'EUR', 'Currency': 'EUR', 'Discount_Rate': 'EUR',
                'Amount': NOTIONAL, 'Payment_Date': BASE + pd.DateOffset(months=1)}

    def exposure(domestic):
        return run([cashflow], calc=dict(CMC, Time_grid='0d 1d(1d) 2w'), models={'FxRate': model}, factors=dict(
            implied[model], **{'FxRate.EUR': dict(FACTORS['FxRate.EUR'], Domestic_Currency=domestic)}))['mtm']

    mtm = exposure(None)
    assert mtm.equals(exposure('EUR'))
    assert {lag(day, 2) for day in mtm.index} == {2, 3, 4}
    worth = NOTIONAL * math.exp(-R_EUR * (cashflow['Payment_Date'] - BASE).days / 365.0) * today()
    for day in mtm.index:
        deflated = np.asarray(mtm.loc[day].values, dtype=float) * df((day - BASE).days)
        assert np.allclose(deflated, worth, rtol=2e-6, atol=0.0), (day, deflated.mean() / worth - 1.0)


#: The declared-defaults swaption, settled physically on its legs, and its cash twin paid 40 days
#: after the expiry
SWPT = next(deal for deal in book.BOOK if deal['Reference'] == 'SWPT')
SWPT_PAID = book.WORLD_EXPIRY + pd.DateOffset(days=40)
CASH_SWPT = dict(SWPT, Reference='CSWPT', Settlement_Style='Cash', Settlement_Date=SWPT_PAID,
                 Children=[dict(leg, Reference='C' + leg['Reference']) for leg in SWPT['Children']])
#: The rates trial's amortising swaption settled in cash on the par yield annuity, paid with the twin
AMORT = next(deal for deal in trial_rates.DEALS if deal['Reference'] == 'SWPT_AMORT')
YIELD_SWPT = dict(AMORT, Reference='YSWPT', Settlement_Style='Cash', Settlement_Date=SWPT_PAID,
                  Cash_Settlement_Method='Par_Yield_Curve',
                  Children=[dict(leg, Reference='Y' + leg['Reference']) for leg in AMORT['Children']])


def cash_annuity(rate, items):
    """A fixed leg's coupons discounted at `rate`, compounded period by period, by hand."""
    tau = np.array([row['Accrual_Year_Fraction'] for row in items])
    coupons = np.array([row['Notional'] for row in items]) * tau
    rate = np.asarray(rate, dtype=float)[..., None]
    return (coupons * np.exp(-np.cumsum(np.log1p(tau * rate), axis=-1))).sum(axis=-1)


@functools.lru_cache(maxsize=None)
def swaptions():
    """One float64 credit Monte Carlo of the declared-defaults world, a Hull-White on both curves:
    the swaption, its cash twin, the swap it delivers beside a unit paid on the twin's day, and the
    amortising par yield swaption beside the swap it becomes, each in a netting set of its own -
    `(dates, {set: ledger}, {reference: profile})`."""
    swap = [dict(leg, Reference='SWAP' + leg['Reference'][4:]) for leg in SWPT['Children']]
    amortised = [dict(leg, Reference='AMORTISED' + leg['Reference'][10:]) for leg in AMORT['Children']]
    unit = {'Object': 'FixedCashflowDeal', 'Reference': 'UNIT', 'Currency': 'USD',
            'Discount_Rate': 'USD', 'Amount': 1.0, 'Payment_Date': SWPT_PAID}
    return by_set([held('PHYSICAL', [SWPT]), held('CASH', [CASH_SWPT]), held('SWAP', swap + [unit]),
                   held('YIELD', [YIELD_SWPT]), held('AMORTISED', amortised)], ('USD', 'USD-PROJ'))


def test_a_physical_swaption_settles_the_swap_it_becomes_and_nothing_where_it_lapses():
    """Exercised, the swaption IS the swap its legs make: on the paths in the money at the expiry -
    the delivered swap's own mark positive there - its ledger is that swap's ledger, the same two
    legs held alone beside it in the run, on every payment date the swap has, to the bit, and on
    the others it books nothing. Nothing settles on the base date or the expiry: the ledger's rows
    are the swap's payment dates and the cash twin's.

    Killing mutations: the legs' cash unbooked, an empty ledger where the swap's first coupon pays
    11,166.27 on the first path; booked on every path, the lapsed paying the swap's coupons; the
    base date or the expiry registered as a day it settles, a 2026-08-03 or 2027-08-03 row.
    """
    dates, ledgers, profiles = swaptions()
    exercised = (profiles['SWAP_FIXED'] + profiles['SWAP_FLOAT'])[dates.index(book.WORLD_EXPIRY)] > 0.0
    assert 0 < exercised.sum() < exercised.size, 'in and out of the money across the paths'
    pays = sorted({row['Payment_Date'] for leg in SWPT['Children'] for row in leg['Cashflows']['Items']})
    assert list(ledgers['PHYSICAL'].index) == sorted(pays + [SWPT_PAID]), ledgers['PHYSICAL'].index
    for day in pays:
        delivered, booked = (np.asarray(ledgers[name].loc[day].values, dtype=float) for name in ('SWAP', 'PHYSICAL'))
        assert np.abs(delivered).min() > 0.0, day
        assert np.array_equal(booked, delivered * exercised), day


def test_a_cash_swaption_pays_the_expiry_value_on_its_settlement_date_and_holds_it_between():
    """Settled in cash, the swaption pays `max(U(t0), 0)` - the delivered swap's own mark on the
    expiry row, floored - on its settlement date, path by path, and nothing on any other day; from
    the expiry on it marks that amount discounted to the settlement, the unit paid that day beside
    it on the same path, which is the level the expiry fixed and not a later day's swap.

    Killing mutations: the amount not held past the expiry, the row after it marking that day's
    swap; the settlement not booked, an empty ledger.
    """
    dates, ledgers, profiles = swaptions()
    expiry = dates.index(book.WORLD_EXPIRY)
    paid = np.asarray(ledgers['CASH'].loc[SWPT_PAID].values, dtype=float)
    swap = profiles['SWAP_FIXED'][expiry] + profiles['SWAP_FLOAT'][expiry]
    assert np.allclose(paid, np.maximum(swap, 0.0), rtol=1e-12, atol=1e-8)
    assert (paid > 0.0).any() and (paid == 0.0).any(), 'in and out of the money across the paths'
    assert not np.asarray(ledgers['CASH'].drop(SWPT_PAID).values, dtype=float).any()
    later = [row for row, day in enumerate(dates) if book.WORLD_EXPIRY <= day <= SWPT_PAID]
    assert len(later) == 3, later                    # the expiry, the day after it, the settlement
    for row in later:
        assert np.allclose(profiles['CSWPT'][row], paid * profiles['UNIT'][row], rtol=1e-12, atol=1e-8), row


def test_a_cash_swaption_is_discounted_on_to_its_settlement_and_either_style_expired_refuses():
    """Paid 40 days after its expiry, the cash twin marks what it marks settling on the expiry -
    where the blank date reads it - times the forward discount over the 40 days, `exp(-4% 40/365)`
    on the flat USD curve, to 1e-12. On a base date past the expiry the document refuses naming the
    deal, the day and what it needs: in cash the amount the expiry fixed, stated nowhere; settled
    physically the holder's election, the swap it delivered booked or nothing if it lapsed.

    Killing mutations: the forward discount dropped, 10,361.17 where it marks 10,315.85; either
    refusal removed, the swaption skipped and the job run without it.
    """
    marked, _ = book.marks([CASH_SWPT, dict(CASH_SWPT, Reference='ON_EXPIRY', Settlement_Date=None)])
    forward = math.exp(-0.04 * 40 / 365.0)
    assert float.fromhex(marked['CSWPT']) == pytest.approx(float.fromhex(marked['ON_EXPIRY']) * forward, rel=1e-12)
    for deal, needs in ((CASH_SWPT, 'stated nowhere'), (SWPT, 'nothing if it lapsed')):
        expired, base = book.book([deal]), book.WORLD_EXPIRY + pd.DateOffset(days=10)
        expired['Calc']['Calculation']['Base_Date'] = base
        expired['Calc']['MergeMarketData']['ExplicitMarketData']['System Parameters']['Base_Date'] = base
        context = derivus.Context()
        context.load_json((json.dumps(expired, cls=CustomJsonEncoder), 'expired'))
        with pytest.raises(Exception) as refusal:
            context.run_job()
        message = str(refusal.value)
        assert all(word in message for word in (deal['Reference'], 'Option_Expiry_Date', '2027-08-03', needs)), message


def test_a_par_yield_swaption_is_black_on_its_coupons_discounted_at_the_swap_rate():
    """Settled in cash on `Par_Yield_Curve`, the swaption is struck on its fixed coupons discounted
    at the swap rate s, `A_c(s) = sum N tau prod (1 + tau s)^-1`, and marks the market formula
    `D(0, T) A_c(s) Black(s, k, sigma, t0)`: the declared-defaults swaption struck at its forward,
    payer and receiver, paid 40 days after the expiry, against that by hand off the world's flat
    4% discount and 4.5% projection curves and its flat 20% surface, to 1e-12. The projection
    above the discount is what parts the annuities: on one flat curve they agree to 1e-6.

    Killing mutation: the swap's own annuity restored, 9,707.85 for the payer where it marks
    9,625.17.
    """
    years = lambda day: (day - book.WORLD_BASE).days / 365.0
    discount = lambda day: math.exp(-book.DISCOUNT[0] * years(day))
    fixed, floating = (leg['Cashflows']['Items'] for leg in SWPT['Children'])
    forward = sum(row['Notional'] * math.expm1(book.PROJECTION[0] * row['Accrual_Year_Fraction'])
                  * discount(row['Payment_Date']) for row in floating) / sum(
        row['Notional'] * row['Accrual_Year_Fraction'] * discount(row['Payment_Date']) for row in fixed)
    struck = book.fixed_items(100.0 * forward)
    payer = dict(CASH_SWPT, Reference='PAYER', Cash_Settlement_Method='Par_Yield_Curve',
                 Swap_Rate=100.0 * forward, Children=[
                     dict(SWPT['Children'][0], Reference='PAYER_FIXED',
                          Cashflows=dict(SWPT['Children'][0]['Cashflows'], Items=struck)),
                     dict(SWPT['Children'][1], Reference='PAYER_FLOAT')])
    receiver = dict(payer, Reference='RECEIVER', Payer_Receiver='Receiver', Children=[
        dict(leg, Reference='RECEIVER' + leg['Reference'][5:],
             Buy_Sell={'Buy': 'Sell', 'Sell': 'Buy'}[leg['Buy_Sell']]) for leg in payer['Children']])
    marked, _ = book.marks([payer, receiver])
    black = forward * math.erf(0.20 * math.sqrt(years(book.WORLD_EXPIRY)) / math.sqrt(8.0))
    by_hand = discount(SWPT_PAID) * cash_annuity(forward, struck) * black
    for reference in ('PAYER', 'RECEIVER'):
        assert float.fromhex(marked[reference]) == pytest.approx(by_hand, rel=1e-12), reference


def test_an_amortising_par_yield_swaption_pays_the_cash_annuity_of_the_swap_it_becomes():
    """Settled in cash on `Par_Yield_Curve`, the amortising swaption pays `A_c(S) max(S - k, 0)` on
    its settlement date, path by path, and nothing on any other day. S is the swap rate of the swap
    it becomes on the expiry row - its legs held alone beside it, the floating leg's mark over the
    fixed leg's annuity - and A_c the fixed leg's own coupons, stepping down with its principal,
    discounted at S.

    Killing mutation: the swap's own annuity restored, paying `max(U(t0), 0)`, 22,145.52 on the
    first path that exercises where it owes 21,810.59.
    """
    dates, ledgers, profiles = swaptions()
    expiry, strike = dates.index(book.WORLD_EXPIRY), YIELD_SWPT['Swap_Rate'] / 100.0
    annuity = -profiles['AMORTISED_FIXED'][expiry] / strike
    rate = profiles['AMORTISED_FLOAT'][expiry] / annuity
    paid = np.asarray(ledgers['YIELD'].loc[SWPT_PAID].values, dtype=float)
    owed = cash_annuity(rate, YIELD_SWPT['Children'][0]['Cashflows']['Items']) * np.maximum(rate - strike, 0.0)
    assert np.allclose(paid, owed, rtol=1e-12, atol=1e-8)
    assert (paid > 0.0).any() and (paid == 0.0).any(), 'in and out of the money across the paths'
    assert not np.asarray(ledgers['YIELD'].drop(SWPT_PAID).values, dtype=float).any()


#: Four equity types paying in the payoff currency at the expiry, none stating it
BLANK_PAYOFF = [
    dict({k: v for k, v in VANILLA.items() if k not in ('Payoff_Currency', 'Settlement_Date')},
         Reference='AMER', Option_Type='Put', Strike_Price=120.0, Option_Style='American'),
    {'Object': 'EquityDiscreteExplicitAsianOption', 'Reference': 'ASN', 'Currency': 'USD',
     'Equity': 'EQ', 'Dividends': 'EQ', 'Discount_Rate': 'USD', 'Equity_Volatility': 'EQ',
     'Buy_Sell': 'Buy', 'Option_Type': 'Call', 'Strike_Price': 95.0, 'Units': UNITS,
     'Expiry_Date': EXPIRY,
     'Sampling_Data': [[BASE + pd.DateOffset(months=m), 0.0, 1.0] for m in (3, 6, 9)] + [[EXPIRY, 0.0, 1.0]]},
    {'Object': 'EquityBarrierBinaryOption', 'Reference': 'KOBIN', 'Currency': 'USD',
     'Equity': 'EQ', 'Dividends': 'EQ', 'Discount_Rate': 'USD', 'Equity_Volatility': 'EQ',
     'Buy_Sell': 'Buy', 'Option_Type': 'Call', 'Strike_Price': STRIKE, 'Cash_Payoff': CASH,
     'Barrier_Type': 'Up_And_Out', 'Barrier_Price': 130.0, 'Expiry_Date': EXPIRY,
     'Barrier_Dates': [[BASE + pd.DateOffset(months=m)] for m in range(1, 12)] + [[EXPIRY]]},
    {'Object': 'QEDI_CustomAutoCallSwap', 'Reference': 'AC', 'Currency': 'USD', 'Equity': 'EQ',
     'Dividends': 'EQ', 'Discount_Rate': 'USD', 'Equity_Volatility': 'EQ', 'Buy_Sell': 'Buy',
     'Option_Type': 'Call', 'Strike_Price': STRIKE, 'Expiry_Date': EXPIRY, 'Units': UNITS,
     'Price_Fixing': [[EXPIRY, 0.0]], 'Autocall_Coupons': [[EXPIRY, 0.08]],
     'Autocall_Thresholds': [[EXPIRY, 1.0]]}]

#: A zero-threshold CSA closing out over ten days
CSA = {'Object': 'NettingCollateralSet', 'Reference': 'CSA', 'Netted': 'True', 'Collateralized': 'True',
       'Agreement_Currency': 'USD', 'Balance_Currency': 'USD', 'Liquidation_Period': 10,
       'Settlement_Period': 0, 'Credit_Support_Amounts': dict.fromkeys(
           ('Received_Threshold', 'Posted_Threshold', 'Independent_Amount', 'Minimum_Received',
            'Minimum_Posted'), {'.CreditSupportList': [[0.0, 0.0]]})}


@pytest.mark.card
def test_a_blank_payoff_currency_is_the_deal_s_own_currency():
    """`Payoff_Currency` declares its blank the deal's own `Currency`, so a document omitting it is
    the one stating USD: under a credit Monte Carlo the American put, the Asian, the barrier binary
    and the autocall mark and book bit for bit alike either way, 570.74 a path on the expiry, and
    a collateralised set holding the American alone prices as it does.

    Killing mutations, each a reader taking the raw field again: the barrier binary registering its
    dates under it, the ledger's thirteen rows down to two; the Asian booking under it, the expiry
    471.89; the autocall naming its factor off it, skipped, the book's t0 mark 756.153 against
    756.536; the American keying its grid dates under it, a '' ledger beside USD - and, held under
    the CSA, the whole set refused ('Cannot find FxRate.').
    """
    stated = [dict(deal, Payoff_Currency='USD') for deal in BLANK_PAYOFF]
    held = tuple([dict(CSA, Children=deals[:1])] for deals in (BLANK_PAYOFF, stated))
    for pair in ((BLANK_PAYOFF, stated), held):
        blank, usd = (run(deals, calc=CMC) for deals in pair)
        assert np.array_equal(blank['mtm'].values, usd['mtm'].values), pair[0][0]['Reference']
        assert set(blank['cashflows']) == set(usd['cashflows']) == {'USD'}, set(blank['cashflows'])
        paid, owed = blank['cashflows']['USD'], usd['cashflows']['USD']
        assert paid.index.equals(owed.index) and np.array_equal(paid.values, owed.values), (
            pair[0][0]['Reference'], paid.loc[EXPIRY].mean(), owed.loc[EXPIRY].mean())
        if len(pair[0]) > 1:
            assert paid.loc[EXPIRY].mean() == pytest.approx(570.74260938, rel=1e-6)
