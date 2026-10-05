"""A declared default is a convention or a placeholder, and the declaration says which.

`Deal.__init__` took the authored block verbatim, so a `fields` declaration's `default=` was
schema-only: a pricer reading an unauthored field BY NAME raised `KeyError` inside
`calc_dependencies`, the deal was logged and SKIPPED, and the job succeeded with the deal priced at
nothing. The seam is `schema.DealFields`, which `Deal.__init__` wraps the authored block in: a read
by name falls through to `schema.deal_defaults`, and nothing else does.

WHICH DEFAULTS COMPLETE IS ON THE DECLARATION. `convention=True` says the declared value IS what
omission means - `Pay_Timing: End`, a null calendar, a blank `Discount_Rate` - and those complete.
Every other default is a PLACEHOLDER: what a blank panel shows and nobody means by leaving it out.
Answering `FXBarrierOption.Strike_Price` 0.0 turns a schema-invalid block into a plausible wrong
number - 741.53 against the 78.93 the author meant - so a placeholder keeps its `KeyError`, and
`schema.validate_instrument` refuses it BY NAME at booking before a pricer ever reaches it.
Unflagged is a placeholder, so a field added later refuses rather than folds.

WHAT A DEFAULT DOES NOT DO IS ENTER THE PROGRAM. `get`, `in`, iteration, `len` and the JSON round
trip see exactly the authored keys, so `plan_hash`, the factor universe and a saved book are
byte-identical across every job document in the tree, hashes pinned below. That split is the
design: `field[key]` is the read that used to raise and the declaration answers it, while
`get(key, fallback)` is the READER's own statement of what an omitted field means. Seven equity
types branch on `'Payoff_Type' in self.field`, and a read of it must not make that True.

A DEFAULT IS CONVERTED, not copied. A declaration is authoring metadata - a blank Table is the
string `'null'`, a Period `'3M'`, a rate a whole number of percent - so `schema.engine_default`
converts each the way the loader converts that field's wire form. Uncoerced, `'0M'` reaches
`base_date + self.field[...]` as a str and the repair swaps one skip for another.

DEGENERACY CHECKLIST for the barrier gate: r = 4% against q = 2% (non-zero, r != q), both knock
directions and both option types, and the repaired reading compared against the furnished one
rather than against zero.
"""
import copy
import glob
import json
import math
import os
import pickle
import sys

# reference-derivus shadow-import guard (MEMORY): pin the package under test to THIS repo.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import pytest
import torch

import derivus
import rates_world
from derivus import instruments, schema, utils
from derivus.config import Config, CustomJsonEncoder
from derivus.instruments import construct_instrument
from derivus.schema import REQUIRED

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES = os.path.join(ROOT, 'tests', 'fixtures')

BASE = pd.Timestamp('2024-06-28')
X0, R_USD, R_EUR, SIGMA = 1.25, 0.04, 0.02, 0.15
NOTIONAL, EXPIRY_D = 1000.0, 365

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

#: The two fields `pv_barrier_option` asks for by name, in the WIRE form a document carries them.
#: Both are conventions, so an author omitting them is an author saying nothing.
FURNISHED = {'Barrier_Monitoring_Frequency': {'.DateOffset': '0M'}, 'Cash_Rebate': 0.0}

#: The twelve-type book below, on `rates_world`'s own date and curves plus a EUR leg and a surface.
WORLD_BASE = rates_world.BASE
WORLD_EXPIRY = WORLD_BASE + pd.DateOffset(years=1)
KNOTS, DISCOUNT, PROJECTION = [0.25, 1.0, 3.0, 5.0], [0.040] * 4, [0.045] * 4

#: A REAL holiday calendar, because a convention that names one is only tested against one: the
#: engine's own `.cal` file, and a location whose holidays move a rolled date off the weekday rule.
CALENDARS = os.path.join(FIXTURES, 'data', 'calendars.cal')
CALENDAR = 'Johannesburg'


def world():
    """`Price Factors` for the book: a USD discount and projection curve, a EUR leg crossed onto
    USD, one FX surface, an equity, and the two rate-vol spaces a cap and a swaption read."""
    factors = rates_world.market(
        'USD', {'USD': (KNOTS, DISCOUNT), 'USD-PROJ': (KNOTS, PROJECTION)}, 'USD')
    factors['FxRate.EUR'] = {'Domestic_Currency': None, 'Interest_Rate': 'EUR', 'Spot': X0}
    factors['InterestRate.EUR'] = {'Currency': 'EUR', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                                   'Curve': utils.Curve([], [[0.0, R_EUR], [5.0, R_EUR]])}
    factors['FXVol.EUR.USD'] = FACTORS['FXVol.EUR.USD']
    factors['EquityPrice.EQ'] = {'Issuer': None, 'Respect_Default': 'No', 'Jump_Level': 0.0,
                                 'Spot': 100.0, 'Interest_Rate': 'USD', 'Currency': 'USD'}
    factors['DividendRate.EQ'] = {'Floor': None, 'Currency': 'USD',
                                  'Curve': utils.Curve([], [[0.0, 0.01], [5.0, 0.01]])}
    factors['EquityPriceVol.EQ'] = {
        'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness',
        'Surface': utils.Curve([], [[m, t, SIGMA] for m in (0.6, 1.0, 1.4)
                                    for t in (0.02, 2.0, 5.0)])}
    surface = [(m, e, t, 0.20) for m in (-0.02, 0.0, 0.02)
               for e in (0.25, 1.0, 2.0, 5.0) for t in (1.0, 3.0, 5.0)]
    factors['InterestYieldVol.USD-PROJ'] = {
        'Property_Aliases': None, 'Distribution_Type': 'Lognormal', 'Shift': utils.Percent(0.0),
        'Surface': utils.Curve([], surface)}
    factors['InterestRateVol.USD-PROJ'] = {'Property_Aliases': None,
                                           'Surface': utils.Curve([], surface)}
    return factors

#: `(Barrier_Type, barrier, strike, option type)` - both directions and both option types, each
#: barrier struck on the live side of its own axis.
BARRIERS = [('Down_And_Out', 1.12, 1.25, 'Call'), ('Down_And_In', 1.12, 1.25, 'Call'),
            ('Up_And_Out', 1.40, 1.30, 'Put'), ('Up_And_In', 1.40, 1.30, 'Put')]

#: The plan and the factor universe of every job document under `tests/fixtures`, as they read
#: BEFORE declared defaults reached a deal. `(plan_hash, resolved factors, missing factors)` - a
#: default entering the program moves the first and a default minting a factor moves the rest.
#: The hash is taken over `Correlations` in the nested form the file carries, which is the only
#: spelling of that section: a document declaring none hashes the same either way.
PINNED = {
    # re-pinned 2026-09-25, every document carrying a deal tree: `Tag_Titles` left the deals'
    # attributes and `Tags` every deal, keys no program read - the factor universes are unmoved and
    # the three documents priced here mark bit for bit
    'autocall_job.json': (
        'dc88939c63f341fb641977e4995ac4cead587aa12c52e4753db531882c74b8c8', 5, 0),
    # re-pinned 2026-09-22: the only plan a declared default has ever moved, and it moved by a
    # constructor no longer WRITING one. `NettingCollateralSet.__init__` used to `setdefault`
    # `Settlement_Period`, `Liquidation_Period` and `Opening_Balance` into the authored block,
    # which the plan hashes; the declaration says all three, so the block is four keys instead of
    # seven and the program is the same one. Measured both ways: 65,584 reported floats, 0
    # mismatches, and the factor universe unmoved at (6, 0)
    'commodity_aps_world.json': (
        '038c69eef5b5c933cc0ef41e5dd2194351385e02834117ea91442fba736766cf', 6, 0),
    # re-pinned 2026-09-03: the DOCUMENT changed, not the reading of it. `Barrier_Hit` retired -
    # the knock-out is a fold over the schedule - so the block lost a field and the plan is a
    # different program. The factor universe is untouched (5, 0) and the mark is bit-identical at
    # 62.428908447906807, which is the half of this pin that says nothing else moved
    'fx_accumulator_job.json': (
        '4f022be39f1d4d49ebcf3e01a5901e9fcd4f63ec93e9d5d6b9bf782907d50e76', 5, 0),
    'fx_tarf_job.json': (
        '6e145cf5df2e9d2d79a0e21a9a13faf54115363ed1406d4d786f73eeb9ecd72d', 5, 0),
    # re-pinned 2026-09-16: both market files carried a null Base_Date, which the loader fills
    # with the wall clock, so these two plans moved with the calendar. Each file now carries
    # its job's own base date; the runs are unmoved, the date being the calculation's to set
    'platinum_hedge_shipping.json': (
        'aeb9c875021626cc15de29d15bc705d8d77a5b11a1b87b961332ba162a44b262', 2, 0),
    'policy_test_simulate_only.json': (
        '695ab01aa2aeebf526e64d007d47d63d2ffee852f0b162bd05fb72147f603f13', 2, 0),
}


def coupons(months, years=3):
    """Coupon dates from the option expiry out `years`, every `months` months."""
    dates = [WORLD_EXPIRY]
    while dates[-1] < WORLD_EXPIRY + pd.DateOffset(years=years):
        dates.append(WORLD_EXPIRY + pd.DateOffset(months=months * len(dates)))
    return dates


def fixed_items(rate, months=6, notional=1e6):
    """A fixed cashflow list, one item per coupon."""
    rows = []
    for start, end in zip(coupons(months)[:-1], coupons(months)[1:]):
        rows.append({
            'Payment_Date': end, 'Notional': notional, 'Rate': utils.Percent(rate),
            'Accrual_Start_Date': start, 'Accrual_End_Date': end, 'Accrual_Day_Count': 'ACT_365',
            'Accrual_Year_Fraction': utils.DayCount.accrual(
                start, (end - start).days, utils.DayCount.code('ACT_365')),
            'Fixed_Amount': 0.0, 'Discounted': 'No', 'FX_Reset_Date': None, 'Known_FX_Rate': 0.0})
    return rows


def float_items(months=3, notional=1e6):
    """A floating cashflow list, one reset spanning each coupon."""
    rows = []
    for start, end in zip(coupons(months)[:-1], coupons(months)[1:]):
        accrual = utils.DayCount.accrual(
            start, (end - start).days, utils.DayCount.code('ACT_365'))
        rows.append({
            'Payment_Date': end, 'Notional': notional, 'Accrual_Start_Date': start,
            'Accrual_End_Date': end, 'Accrual_Day_Count': 'ACT_365',
            'Accrual_Year_Fraction': accrual,
            'Resets': [[start, start, end, accrual, pd.DateOffset(days=1), 'ACT_365', '0D', 0.0,
                        'No', utils.Percent(0.0)]],
            'Margin': utils.Basis(0.0), 'Fixed_Amount': 0.0, 'FX_Reset_Date': None,
            'Known_FX_Rate': 0.0})
    return rows


def fx_leg(object_type, reference, **extra):
    """The block every FX option here shares - a EUR call on a USD book off one surface."""
    return dict({
        'Object': object_type, 'Reference': reference, 'Currency': 'USD',
        'Underlying_Currency': 'EUR', 'Discount_Rate': 'USD', 'FX_Volatility': 'EUR.USD',
        'Buy_Sell': 'Buy', 'Option_Type': 'Call', 'Strike_Price': 1.25,
        'Underlying_Amount': 1000.0}, **extra)


#: The ten types gate 2 prices both ways, each AUTHORED IN FULL: `stripped` takes the conventions
#: back out and `furnished` writes them back in, and the two documents have to agree to the bit.
BOOK = [
    rates_world.par_swap('SWAP', 'USD', 'USD-PROJ', 'USD', 3, 4.5),
    rates_world.deposit('DEPO', 'USD', 'USD', 6, 4.2),
    rates_world.fra('FRA', 'USD', 'USD-PROJ', 'USD', 3, 6, 4.3),
    {'Object': 'SwaptionDeal', 'Reference': 'SWPT', 'Currency': 'USD', 'Discount_Rate': 'USD',
     'Forecast_Rate': 'USD-PROJ', 'Forecast_Rate_Volatility': 'USD-PROJ', 'Buy_Sell': 'Buy',
     'Payer_Receiver': 'Payer', 'Settlement_Style': 'Physical',
     'Option_Expiry_Date': WORLD_EXPIRY, 'Swap_Effective_Date': WORLD_EXPIRY,
     'Swap_Maturity_Date': WORLD_EXPIRY + pd.DateOffset(years=3), 'Settlement_Date': WORLD_EXPIRY,
     'Principal': 1e6, 'Swap_Rate': 4.5, 'Index_Tenor': pd.DateOffset(months=3),
     'Pay_Frequency': pd.DateOffset(months=6), 'Receive_Frequency': pd.DateOffset(months=3),
     'Children': [
         rates_world._cashflow_leg(
             'CFFixedInterestListDeal', 'SWPT_FIXED', 'USD', 'USD', 'Sell',
             {'Compounding': 'No', 'Items': fixed_items(4.5)}, Calendars=None, Rate_Currency=''),
         rates_world._cashflow_leg(
             'CFFloatingInterestListDeal', 'SWPT_FLOAT', 'USD', 'USD', 'Buy',
             {'Compounding_Method': 'None', 'Averaging_Method': 'Average_Rate', 'Properties': [],
              'Items': float_items()},
             Forecast_Rate='USD-PROJ', Rate_Adjustment_Method='None', Rate_Sticky_Month_End='Yes',
             Rate_Offset=0, Rate_Calendars=None, Accrual_Calendars=None,
             Forecast_Rate_Cap_Volatility='', Forecast_Rate_Swaption_Volatility='',
             Discount_Rate_Cap_Volatility='', Discount_Rate_Swaption_Volatility='')]},
    {'Object': 'CapDeal', 'Reference': 'CAP', 'Currency': 'USD', 'Discount_Rate': 'USD',
     'Forecast_Rate': 'USD-PROJ', 'Forecast_Rate_Volatility': 'USD-PROJ', 'Buy_Sell': 'Buy',
     'Effective_Date': WORLD_BASE, 'Maturity_Date': WORLD_BASE + pd.DateOffset(years=2),
     'Cap_Rate': 4.5, 'Principal': 1e6},
    fx_leg('FXOptionDeal', 'FXO', Expiry_Date=WORLD_EXPIRY),
    fx_leg('FXTARFOptionDeal', 'TARF', Expiry_Date=WORLD_EXPIRY, LeverageNotional=2000.0,
           TargetLevel=1e9, InvertedTarget=False,
           TARF_ExpiryDates=[[WORLD_EXPIRY - pd.DateOffset(months=3), WORLD_EXPIRY, 0.0],
                             [WORLD_EXPIRY, WORLD_EXPIRY + pd.DateOffset(days=2), 0.0]]),
    fx_leg('FXAccumulatorOptionDeal', 'ACC', LeverageNotional=2000.0, Barrier_Type='Up_And_Out',
           Barrier_Price=5.0,
           Accumulator_ExpiryDates=[[WORLD_EXPIRY - pd.DateOffset(months=3), WORLD_EXPIRY, 0.0],
                                    [WORLD_EXPIRY, WORLD_EXPIRY + pd.DateOffset(days=2), 0.0]]),
    fx_leg('FXBarrierOption', 'BARR', Expiry_Date=WORLD_EXPIRY, Barrier_Type='Down_And_Out',
           Barrier_Price=1.12),
    {'Object': 'NettingCollateralSet', 'Reference': 'NS', 'Netted': 'True',
     'Collateralized': 'False',
     'Children': [rates_world._cashflow_leg(
         'CFFixedInterestListDeal', 'NS_FIXED', 'USD', 'USD', 'Buy',
         {'Compounding': 'No', 'Items': fixed_items(4.5)}, Calendars=None, Rate_Currency='')]},
    # a deposit paying on a REAL calendar - the conventions that name one are only tested against
    # one, and this rolls its coupons on Johannesburg's own holidays
    dict(rates_world.deposit('DEPC', 'USD', 'USD', 12, 4.2),
         Accrual_Calendars=CALENDAR, Payment_Calendars=CALENDAR, Payment_Offset=2,
         Reference='DEPC'),
]

#: An equity swap leg accruing on a real calendar, three business days out - what `reset` reads
#: `Payment_Calendars` against, and the leg its dividend table is spelled three ways on.
EQUITY_LEG = {'Object': 'EquitySwapLeg', 'Reference': 'EQL', 'Currency': 'USD',
              'Discount_Rate': 'USD', 'Equity': 'EQ', 'Equity_Volatility': 'EQ',
              'Equity_Currency': 'USD', 'Buy_Sell': 'Buy', 'Effective_Date': WORLD_BASE,
              'Maturity_Date': WORLD_EXPIRY + pd.DateOffset(days=1), 'Units': 100.0,
              'Principal': 1e6, 'Accrual_Calendars': CALENDAR, 'Payment_Offset': 3}


def deal_classes():
    """Every constructible deal type, by name."""
    return sorted(
        (name, cls) for name, cls in vars(instruments).items()
        if isinstance(cls, type) and issubclass(cls, instruments.Deal) and getattr(cls, 'fields', None))


def declared(cls):
    """`{key: F}` for every field `cls` declares a default for, inherited declarations included."""
    return {f.key: f for group in (getattr(cls, 'fields', []) or []) for f in group.fields
            if f.default is not REQUIRED and f.default is not None}


def barrier_deal(barrier_type, barrier, strike, option_type, **extra):
    block = {'Object': 'FXBarrierOption', 'Reference': 'BR', 'Currency': 'USD',
             'Underlying_Currency': 'EUR', 'Payoff_Currency': 'USD', 'Discount_Rate': 'USD',
             'FX_Volatility': 'EUR.USD', 'Buy_Sell': 'Buy', 'Option_Type': option_type,
             'Strike_Price': strike, 'Barrier_Price': barrier, 'Barrier_Type': barrier_type,
             'Underlying_Amount': NOTIONAL,
             'Expiry_Date': {'.Timestamp': (BASE + pd.DateOffset(days=EXPIRY_D)).strftime('%Y-%m-%d')}}
    block.update(extra)
    return block


def price(block):
    """One deal through the JSON contract; its own row of `Results['mtm']`, not the book total."""
    job = {'Calc': {
        'Calculation': {'Object': 'BaseValuation', 'Base_Date': BASE, 'Currency': 'USD',
                        'MCMC_Simulations': 1, 'Random_Seed': 1},
        'Deals': {'Reference': 'defaults',
                  'Deals': {'Children': [{'Instrument': {'.Deal': block}}]}},
        'MergeMarketData': {'MarketDataFile': '', 'ExplicitMarketData': {
            'System Parameters': {'Base_Currency': 'USD', 'Base_Date': BASE},
            'Valuation Configuration': {}, 'Price Factors': FACTORS}}}}
    cx = derivus.Context()
    cx.load_json((json.dumps(job, cls=CustomJsonEncoder), 'defaults'))
    _, out = cx.run_job()
    rows = out['Results']['mtm']
    own = rows[rows['Reference'] == 'BR']['Value']
    # a skipped deal leaves NO row of its own - the failure mode this gate exists for
    assert len(own) == 1, 'the deal left no row in Results: it was skipped, not priced'
    return float(own.iloc[0])


# --------------------------------------------------------------------------------------------
# the repair
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize('barrier_type,barrier,strike,option_type', BARRIERS)
def test_a_barrier_omitting_the_two_declared_fields_prices_instead_of_skipping(
        barrier_type, barrier, strike, option_type):
    """`pv_barrier_option`'s own block, both ways: the declaration is what the author would have
    written, so the two readings agree to the bit and neither is zero."""
    furnished = price(barrier_deal(barrier_type, barrier, strike, option_type, **FURNISHED))
    omitted = price(barrier_deal(barrier_type, barrier, strike, option_type))
    assert omitted == furnished, (barrier_type, omitted, furnished)
    assert abs(furnished) > 1.0, 'the fixture must have something to lose'


def test_the_repaired_deal_reads_the_engine_form_of_both_defaults():
    """Uncoerced, `'0M'` is a str and `base_date + str` is the next skip. The monitoring frequency
    arrives as a `DateOffset` of zero months and the rebate as the declared zero."""
    deal = construct_instrument(dict(barrier_deal(*BARRIERS[0]), Expiry_Date=BASE), {})
    frequency = deal.field['Barrier_Monitoring_Frequency']
    assert (BASE + frequency - BASE).days == 0
    assert isinstance(frequency, pd.DateOffset)
    assert deal.field['Cash_Rebate'] == 0


# --------------------------------------------------------------------------------------------
# a default answers a read; it does not enter the program
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize('name,cls', deal_classes())
def test_a_default_answers_a_read_and_never_enters_the_program(name, cls):
    """The seam itself, over every deal type: a CONVENTION answers `field[key]`, a PLACEHOLDER
    still raises, and the dict holds only what was authored - which is what `plan_hash`,
    `get_fieldname` and the JSON round trip read. Read off `DealFields` rather than a constructed
    deal: 21 classes cannot be built from `Object` alone, by the same KeyError a placeholder
    deliberately keeps.
    """
    block = {'Object': name}
    field = schema.DealFields(dict(block), cls)

    assert set(field) == set(block), 'a default entered the program'
    assert json.loads(json.dumps(field, cls=CustomJsonEncoder)).keys() == field.keys()

    answered = 0
    for key, declaration in declared(cls).items():
        if key in block:
            continue
        assert key not in field, '{}: a default answered a membership test'.format(key)
        assert field.get(key) is None, '{}: a default displaced a get() fallback'.format(key)
        if declaration.convention:
            field[key]                                      # answers rather than raising
            answered += 1
        else:
            with pytest.raises(KeyError):
                field[key]

    assert set(field) == set(block), 'a read wrote the default into the block'
    assert answered == sum(1 for f in declared(cls).values() if f.convention)

    with pytest.raises(KeyError):
        field['A_Field_No_Declaration_Names']


def test_a_read_of_a_convention_does_not_make_it_present():
    """THE GUARANTEE, on the branch that would pay for it. Seven equity types decide the quanto
    wiring on `'Payoff_Type' in self.field`, and `Payoff_Type` is a convention - so a completion
    that entered the block would compile a plain option as a quanto one. An option omitting it
    compiles with the branch OFF and the key still absent after the read."""
    block = {'Object': 'EquityOptionDeal', 'Reference': 'EQ', 'Currency': 'USD',
             'Payoff_Currency': 'EUR', 'Equity': 'EQ', 'Equity_Volatility': 'EQ',
             'Strike_Price': 100.0, 'Units': 1.0, 'Buy_Sell': 'Buy', 'Option_Type': 'Call',
             'Expiry_Date': BASE}
    deal = construct_instrument(dict(block), {})
    assert deal.field['Payoff_Type'] == 'Standard'
    assert 'Payoff_Type' not in deal.field and set(deal.field) == set(block)

    index = {}
    deal.check_option_data(
        {'Payoff_Currency': ('EUR',), 'Currency': ('USD',), 'Equity_Volatility': ('EQ',)},
        index, {}, {}, {}, {})
    assert index == {'Check_Payoff_Type': False}, index


def test_the_deal_constructor_is_where_the_seam_is_wired():
    """`Deal.__init__` wraps the authored block, so the completion travels with the deal rather
    than being applied at one reader."""
    deal = construct_instrument(dict(barrier_deal(*BARRIERS[0])), {})
    assert isinstance(deal.field, schema.DealFields)
    assert set(deal.field) == set(barrier_deal(*BARRIERS[0]))
    assert deal.field['Cash_Rebate'] == 0 and 'Cash_Rebate' not in deal.field


@pytest.mark.parametrize('name,cls', deal_classes())
def test_a_completed_default_is_the_deal_s_own(name, cls):
    """A completion is deep-copied per deal: `DateList.consume` mutates, and two deals of one type
    sharing a declaration's object would consume each other's fixings."""
    one, two = (schema.DealFields({'Object': name}, cls) for _ in range(2))
    store = schema.deal_defaults(cls)
    assert store, '{} declares no convention at all'.format(name)
    for key in store:
        assert one[key] is one[key], '{}: a read must be stable'.format(key)
        assert reading(one[key]) == reading(store[key]), \
            '{}: the completion is not the declaration'.format(key)
        if not isinstance(store[key], (int, float, str, bytes, bool)):
            # an immutable scalar may legitimately be interned; a container may not be shared
            assert one[key] is not store[key], '{}: the class store was handed out'.format(key)
            assert one[key] is not two[key], key


def test_no_declared_default_can_mint_a_price_factor():
    """Discovery reads the raw block through `get_fieldname`, which drops a blank - so a default
    landing on a factor-naming field has to BE blank, or a deal would name a curve nobody loaded."""
    minting = []
    for name, cls in deal_classes():
        defaults = schema.deal_defaults(cls)
        for key in getattr(cls, 'factor_fields', {}) or {}:
            head = key[0] if isinstance(key, tuple) else key
            value = defaults.get(head)
            if isinstance(key, tuple):
                # a nested field is walked one level at a time and each level is dropped on falsy
                value = (value or {}).get(key[1]) if len(key) > 1 else value
            if value:
                minting.append('{}.{}={!r}'.format(name, key, value))
    assert not minting, minting


# --------------------------------------------------------------------------------------------
# the engine form of a declared default is the loader's own
# --------------------------------------------------------------------------------------------
def test_a_period_default_parses_through_the_grammar_the_loader_uses():
    """Every Period a deal declares, against `Config.parse_period` - the one spelling of that
    parse, and the form `{'.DateOffset': ...}` decodes to."""
    config = Config()
    seen = 0
    for _, cls in deal_classes():
        for key, field in declared(cls).items():
            if field.obj != 'Period':
                continue
            seen += 1
            engine = schema.engine_default(field)
            assert engine.kwds == config.parse_period(field.default).kwds, (cls.__name__, key)
    assert seen >= 20, 'the Period declarations went somewhere'


def test_a_table_default_is_the_empty_container_its_tag_names():
    """`'null'` is what a widget writes for an empty table; the engine reads a `utils` container or
    a list, and never the four characters."""
    kinds = {'DateList': utils.DateList, 'DateEqualList': utils.DateEqualList,
             'CreditSupportList': utils.CreditSupportList, 'DateValueList': list, None: list}
    seen = 0
    for _, cls in deal_classes():
        for key, field in declared(cls).items():
            if field.type != 'Table':
                continue
            seen += 1
            value = schema.engine_default(field)
            assert isinstance(value, kinds[field.tag]), (cls.__name__, key, field.tag)
            assert not value, (cls.__name__, key)
    assert seen >= 30, 'the Table declarations went somewhere'


def test_a_blank_table_reads_the_same_however_it_is_spelled():
    """A table left out completes to its tag's empty container, which an author also writes as
    `{".DateEqualList": []}` and a widget as `null`. An empty container is falsy, so a reader's
    `if self.field[<table>]` takes the blank arm on all three: the equity swap leg's dividend read
    prices the three spellings to the bit, and a leg stating a dividend compiles beside them.

    KILLING MUTATIONS: `__len__` removed from the containers, so an empty one is truthy; the
    `sum_range` call back to two of its three arguments, which skips the leg stating a dividend.
    """
    for blank in (utils.DateList({}), utils.DateEqualList([]), utils.CreditSupportList([])):
        assert not blank and len(blank) == 0
    assert utils.DateList({WORLD_BASE: 1.0}) and utils.CreditSupportList([[0, 1.0]])

    leg = dict(EQUITY_LEG, Reference='OMITTED', Payoff_Currency='USD',
               Equity_Known_Prices=utils.DateEqualList([[WORLD_BASE, 100.0, 1.0]]))
    legs = [leg, dict(leg, Reference='NULL', Known_Dividends=None),
            dict(leg, Reference='EMPTY', Known_Dividends=utils.DateEqualList([])),
            dict(leg, Reference='STATED', Known_Dividends=utils.DateEqualList(
                [[WORLD_BASE - pd.DateOffset(days=10), 1.5]]))]
    valued, stats = marks(legs)
    assert 'Deals Skipped' not in stats, stats
    assert valued['OMITTED'] == valued['NULL'] == valued['EMPTY'] == valued['STATED']
    assert math.isfinite(float.fromhex(valued['NULL'])) and float.fromhex(valued['NULL']) != 0.0


def test_a_one_day_realised_dividend_holds_its_digits_in_float32():
    """`utils.calc_realized_dividends` over one day in float32 on flat 4% repo and 3% dividend
    curves, S0 e^{r/365} (1 - e^{-q/365}), against `math.expm1` in float64: under 1e-6.

    Killed by: the old spelling `1 - exp(-q t)`, 7.8e-5 off."""
    class Static:
        riskneutral, t_Buffer = True, {}
        t_Static_Buffer = [torch.tensor([r, r], dtype=torch.float32) for r in (0.04, 0.03)]

    flat = utils.CurveTenor(np.array([0.0, 10.0]))
    repo, dividend = ([(False, i, None, flat, lambda days: days / 365.0)] for i in (0, 1))
    reset = np.array([[0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    got = utils.calc_realized_dividends(
        torch.tensor(100.0, dtype=torch.float32), repo, dividend, [reset], Static())
    assert float(got) == pytest.approx(
        100.0 * math.exp(0.04 / 365.0) * -math.expm1(-0.03 / 365.0), rel=1e-6)


def test_an_equity_swap_leg_reads_its_payoff_currency_start_price_and_known_dividends():
    """The trial's seasoned leg, ten thousand shares struck at 97, USD at 4% for discount and repo:

    - a blank or omitted `Payoff_Currency` is the leg's own currency, to the bit of the stated one;
    - started on the base date with no `Equity_Known_Prices` it fixes at the spot and is worth
      Units S0 (1 - e^{-rT}), 39,210.56; started a month back with none, it refuses by name;
    - a dividend of 1.5 between its start and the base date, which the model's own dividends do not
      reach, joins the realised dividends and is carried at the repo rate to the end: on a repo of
      4.5% against the 4% it discounts at it adds Units x 1.5 e^{0.005 T}, 15,075.19; and an ended
      leg awaiting payment adds it at the payment's discount factor;
    - an ended leg awaiting payment is its realised move Units (S1 - S0) D(pay), 19,991.23: the
      model's dividends reach no interval already ended.

    Killing mutations: the payoff currency read raw; the FX slot left None; the refusal dropped;
    either dividend site dropped; the dividend carried at the discount rate, 75.19 short; the ended
    interval read reversed, 53.12 short.
    """
    import trial_equity_swaps
    leg = trial_equity_swaps.equity_leg('EQL', 'Variable')
    end = WORLD_BASE - pd.DateOffset(days=2)
    ended = dict(leg, Reference='ENDED', Maturity_Date=end, Payment_Offset=5,
                 Equity_Known_Prices=utils.DateEqualList([[trial_equity_swaps.SEASONED, 97.0, 1.0],
                                                          [end, 99.0, 1.0]]))
    dividend = utils.DateEqualList([[trial_equity_swaps.SEASONED + pd.DateOffset(days=10), 1.5]])
    unpriced = dict(leg, Reference='UNPRICED', Equity_Known_Prices=None)
    legs = [leg, dict(leg, Reference='BLANK', Payoff_Currency=''),
            {k: v for k, v in dict(leg, Reference='OMITTED').items() if k != 'Payoff_Currency'},
            dict(leg, Reference='AT_BASE', Effective_Date=WORLD_BASE, Equity_Known_Prices=None), ended,
            dict(ended, Reference='ENDED_DIVIDEND', Known_Dividends=dividend)]
    valued, stats = marks(legs)
    value = {reference: float.fromhex(mark) for reference, mark in valued.items()}
    assert 'Deals Skipped' not in stats and valued['EQL'] == valued['BLANK'] == valued['OMITTED']
    assert value['AT_BASE'] == pytest.approx(1e4 * 100.0 * -math.expm1(-0.04), rel=1e-14)
    repo, _ = marks([leg, dict(leg, Reference='DIVIDEND', Known_Dividends=dividend)], {
        'EquityPrice.EQ': dict(world()['EquityPrice.EQ'], Interest_Rate='USD-PROJ')})
    carried = (leg['Maturity_Date'] - WORLD_BASE).days / 365.0
    assert float.fromhex(repo['DIVIDEND']) - float.fromhex(repo['EQL']) == pytest.approx(
        1e4 * 1.5 * math.exp(0.005 * carried), rel=1e-12)
    paid = (end + pd.offsets.BDay(5) - WORLD_BASE).days / 365.0
    assert value['ENDED_DIVIDEND'] - value['ENDED'] == pytest.approx(1e4 * 1.5 * math.exp(-0.04 * paid),
                                                                    rel=1e-12)
    assert value['ENDED'] == pytest.approx(1e4 * (99.0 - 97.0) * math.exp(-0.04 * paid), rel=1e-12)
    assert marks([leg, unpriced])[1].get('Deals Skipped') == 1
    with pytest.raises(utils.UnpriceableSchedule, match=(
            r"EquitySwapLeg UNPRICED \('Equity_Known_Prices states no price on or before the "
            r"Effective_Date \d{4}-\d\d-\d\d, before the base date',\)")):
        marks([leg, unpriced], system={'Exclude_Deals_With_Missing_Market_Data': 'No'})


def test_an_omitted_floating_margin_completes_to_the_basis_its_reader_takes():
    """The V2 autocall swap reads `Floating_Margin` as `.amount`, so it is declared a `Basis`: left
    out it completes to `Basis(0)` and prices to the bit as `{".Basis": 0}`, where a float 0.0
    skipped the deal, and the store publishes the Basis wire form. A stated 25bp moves the forecast
    leg by the hand margin `-Units 25e-4 sum_k a_k D(t_k)`, on the trial's terms with every trigger
    out of reach and the barrier, out of reach too, observed after the last coupon - the arm that
    pays that leg - and by the same amount with two price fixings a quarter, each payment
    discounted at its own date.

    KILLING MUTATIONS: the declaration back to a bare Float, which skips the omitted margin's deal;
    a payment written at its count among the payments, the k-th read at the k-th fixing's date.
    """
    import trial_equity_swaps as trial
    quarters = trial.QUARTERS
    v2 = {key: value for key, value in trial.DEALS[-1].items() if key != 'Floating_Margin'}
    v2.update(Reference='OMITTED', Autocall_Thresholds=[[date, 1000.0] for date in quarters],
              Barrier=1e-6, Barrier_Dates=[quarters[-1] + pd.Timedelta(days=1)])
    twice = dict(v2, Price_Fixing=[
        [day, 0.0] for quarter in quarters for day in (quarter - pd.DateOffset(months=1), quarter)])
    valued, stats = marks([v2] + [
        dict(deal, Reference=name + suffix, Floating_Margin=utils.Basis(bp))
        for deal, suffix in ((v2, ''), (twice, '_TWICE'))
        for name, bp in (('ZERO', 0.0), ('MARGIN', 25.0))])
    assert 'Deals Skipped' not in stats and valued['OMITTED'] == valued['ZERO'], stats

    starts = [quarters[0] - pd.DateOffset(months=3)] + quarters[:-1]
    hand = -1e6 * 25e-4 * sum((end - start).days / 365.0 * math.exp(
        -0.04 * (end - WORLD_BASE).days / 365.0) for start, end in zip(starts, quarters))
    for suffix in ('', '_TWICE'):
        moved = float.fromhex(valued['MARGIN' + suffix]) - float.fromhex(valued['ZERO' + suffix])
        assert moved == pytest.approx(hand, rel=1e-12), (suffix, moved, hand)
    assert schema.mapping['Instrument']['sections']['QEDI_CustomSwap.Fields'][
        'Floating_Margin']['obj'] == 'Basis'


def test_a_commodity_forward_omitting_its_forward_date_is_read_at_the_spot():
    """`Forward_Date` is a convention: left out, the forward is read at the spot and carried to its
    maturity, so on a zero-carry commodity the trial's forward validates and marks `Units S D(T)`,
    50 x 1,000 x e^-0.04; stated at the maturity it marks `Units F(T) D(T)` off the curve's 1,025.

    KILLING MUTATION: the convention flag dropped, which refuses the omitted date at booking.
    """
    import trial_commodity as trial
    factors = dict(trial.FACTORS, **{
        'CommodityPrice.METAL': dict(trial.FACTORS['CommodityPrice.METAL'], Interest_Rate='ZERO'),
        'InterestRate.ZERO': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                              'Curve': utils.Curve([], [[0.0, 0.0], [5.0, 0.0]])}})
    spot = dict(trial.DEALS[0], Reference='SPOT')
    assert schema.validate_instrument(construct_instrument(dict(spot), {})) == []
    valued, _ = marks([spot, dict(spot, Reference='DATED', Forward_Date=WORLD_EXPIRY)], factors)
    discount = math.exp(-0.04 * (WORLD_EXPIRY - WORLD_BASE).days / 365.0)
    assert float.fromhex(valued['SPOT']) == pytest.approx(50.0 * 1000.0 * discount, rel=1e-14)
    assert float.fromhex(valued['DATED']) == pytest.approx(50.0 * 1025.0 * discount, rel=1e-14)


def test_an_equity_binary_settles_at_its_expiry_where_it_states_no_settlement_date():
    """`Settlement_Date` on an equity binary is a convention, the expiry its declared settlement
    falls back to: omitted, blank and stated at the expiry, the trial binary validates and marks
    one number, bit for bit.

    KILLING MUTATIONS: the convention flag dropped, which names the omitted date at booking; the
    date read as `get('Settlement_Date', expiry)`, under which a blank one meets `'' - Timestamp`
    and the deal is skipped.
    """
    import trial_equity as trial
    binary = next(deal for deal in trial.DEALS if deal['Reference'] == 'EQBIN')
    assert 'Settlement_Date' not in binary
    assert schema.validate_instrument(construct_instrument(dict(binary), {})) == []
    valued, stats = marks([dict(binary, Reference='OMITTED'),
                           dict(binary, Reference='BLANK', Settlement_Date=''),
                           dict(binary, Reference='AT_EXPIRY', Settlement_Date=binary['Expiry_Date'])])
    assert 'Deals Skipped' not in stats, stats
    assert valued['OMITTED'] == valued['BLANK'] == valued['AT_EXPIRY'], valued
    assert math.isfinite(float.fromhex(valued['OMITTED'])) and float.fromhex(valued['OMITTED']) != 0.0


def test_a_rate_default_carries_its_unit():
    """A `Percent`/`Basis` declaration is a whole number of percent or of basis points, and the
    engine reads `.amount`."""
    for _, cls in deal_classes():
        for key, field in declared(cls).items():
            if field.obj not in ('Percent', 'Basis'):
                continue
            value = schema.engine_default(field)
            assert isinstance(value, utils.Percent if field.obj == 'Percent' else utils.Basis)
            assert float(value) == field.default / value.divisor


# --------------------------------------------------------------------------------------------
# nothing else moved
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize('filename', sorted(PINNED))
def test_a_job_document_keeps_its_plan_hash_and_its_factor_universe(filename):
    """The HARD acceptance, per document: the program and the want-list are what they were."""
    context = derivus.Context()
    context.load_json(os.path.join(FIXTURES, filename))
    universe = context.current_cfg.factor_universe()
    plan, resolved, missing = PINNED[filename]
    assert context.plan_hash() == plan
    assert (len(universe['resolved']), len(universe['missing'])) == (resolved, missing)


def test_every_deal_in_every_job_document_holds_exactly_its_authored_block():
    """The invariant behind the pinned hashes, over every document in the tree that loads: the
    constructed deal's field dict IS the `.Deal` block the file carries. No constructor writes into
    it any more - `NettingCollateralSet` used to `setdefault` three keys the declaration now says.
    """
    paths = sorted(glob.glob(os.path.join(FIXTURES, '*.json')) +
                   glob.glob(os.path.join(ROOT, 'data', '*', 'job_*.json')))
    checked = 0
    for path in paths:
        with open(path, 'rt', encoding='utf-8') as handle:
            raw = handle.read()
        if '".Deal"' not in raw:
            continue
        context = derivus.Context()
        context.load_json(path)
        authored = []
        stack = [json.loads(raw)]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                if isinstance(node.get('.Deal'), dict):
                    authored.append(node['.Deal'])
                stack.extend(node.values())
            elif isinstance(node, list):
                stack.extend(node)
        built = list(context.current_cfg.walk_deals())
        assert len(built) == len(authored), path
        wrote = [set(block) for block in authored]
        for deal in built:
            assert set(deal.field) in wrote, '{}: {} carries keys no author wrote - {}'.format(
                path, deal.field.get('Reference'), sorted(set(deal.field).difference(*wrote)))
            checked += 1
    assert checked, 'the sweep found no deals to check'


#: The fields of the SAME `FXBarrierOption` block whose declared default is a blank panel's value
#: and not the engine's meaning. Dropping one is a schema-INVALID document, and the value beside
#: each is what completing it would have priced.
NOT_COMPLETABLE = [('Strike_Price', 741.5344181072), ('Barrier_Type', 6.3673278714),
                   ('Buy_Sell', 78.9325214257), ('Option_Type', 78.9325214257)]


@pytest.mark.parametrize('key,silent', NOT_COMPLETABLE, ids=[k for k, _ in NOT_COMPLETABLE])
def test_an_economic_field_is_refused_by_name_and_does_not_price(key, silent):
    """The seam's own hazard, gated at the booking rather than at the skip. `FXBarrierOption` marks
    none of its economic fields REQUIRED, so a blanket completion prices a strikeless deal at
    741.53 and flips a Down_And_Out to its In at 6.37, against 78.93 for the deal the author meant.
    The block is now REFUSED BY NAME before anything prices it, and a pricing run still leaves no
    row: a placeholder keeps its `KeyError` and the loader's skip is the last line, not the first.
    """
    block = barrier_deal('Down_And_Out', 1.12, 1.25, 'Call', **FURNISHED)
    del block[key]
    deal = construct_instrument(dict(block), {})
    assert '{} is not stated'.format(key) in schema.validate_instrument(deal)

    try:
        priced = price(block)
    except AssertionError:
        return                                              # skipped: no row of its own, as before
    assert priced != priced, (
        '{} was completed: the deal priced {!r} where the author meant {!r}'.format(
            key, priced, silent))


def test_a_blank_date_default_keeps_its_named_refusal():
    """`Expiry_Date` declares `''`, and a blank Date is not an absent one: completed, it reaches a
    date comparison as a `str` and the deal dies four layers down on
    `'<' not supported between instances of 'Timestamp' and 'str'` instead of naming the field."""
    block = barrier_deal('Down_And_Out', 1.12, 1.25, 'Call', **FURNISHED)
    del block['Expiry_Date']
    with pytest.raises(KeyError) as refusal:
        price(block)
    assert 'Expiry_Date' in str(refusal.value)


def test_the_declaration_is_the_only_source_a_deal_default_comes_from():
    """`declared_defaults` completes a CALCULATION's params and `deal_defaults` a deal's read, off
    the same `default=`. Neither reads the other's shape, and neither completes a REQUIRED field or
    a placeholder."""
    conventions = schema.deal_defaults(instruments.FXBarrierOption)
    assert conventions.keys() <= {
        f.key for group in instruments.FXBarrierOption.fields for f in group.fields}
    for group in instruments.FXBarrierOption.fields:
        for field in group.fields:
            assert (field.key in conventions) is bool(field.convention), field.key
            if field.default is REQUIRED:
                assert not field.convention and field.key not in conventions


def test_a_deal_survives_the_round_trips_a_job_puts_it_through():
    """A book crosses a process boundary by pickle and a solve step by `deepcopy`; both have to
    bring the completion with them, or a forked worker prices the skip again."""
    deal = construct_instrument(dict(barrier_deal(*BARRIERS[0])), {})
    authored = dict(deal.field)
    for clone in (copy.deepcopy(deal), pickle.loads(pickle.dumps(deal))):
        assert dict(clone.field) == authored
        assert clone.field['Cash_Rebate'] == 0
        assert 'Cash_Rebate' not in clone.field


# --------------------------------------------------------------------------------------------
# a sparse document and a full one are ONE document
# --------------------------------------------------------------------------------------------
def reading(value):
    """The value a reader sees - a utils container by its rows, a scaled rate by its amount. Two
    completions of one declaration are equal by this and not by identity."""
    return getattr(value, 'data', getattr(value, 'amount', value))


def furnished(block):
    """`block` with every CONVENTION its type declares written out at the declared value, legs
    included - the FULL document a sparse one has to price identically to."""
    cls = getattr(instruments, block['Object'])
    full = dict(schema.deal_defaults(cls), **{k: v for k, v in block.items() if k != 'Children'})
    if 'Children' in block:
        full['Children'] = [furnished(child) for child in block['Children']]
    return full


def stripped(block):
    """`block` with every convention it states AT the declared value removed - what an author who
    wrote only the terms would have posted. A stated value that DIFFERS stands, as does every
    placeholder whatever it carries."""
    declared = schema.deal_defaults(getattr(instruments, block['Object']))
    out = {}
    for key, value in block.items():
        if key == 'Children':
            out[key] = [stripped(child) for child in value]
        elif key not in declared or reading(value) != reading(declared[key]):
            out[key] = value
    return out


def book(deals):
    """A base-valuation job document over `deals`, in the wire form `Context.load_json` reads."""
    def node(deal):
        block = {key: value for key, value in deal.items() if key != 'Children'}
        out = {'Instrument': {'.Deal': block}}
        if deal.get('Children'):
            out['Children'] = [node(child) for child in deal['Children']]
        return out

    return {'Calc': {
        'Calculation': {'Object': 'BaseValuation', 'Base_Date': WORLD_BASE, 'Currency': 'USD',
                        'MCMC_Simulations': 1, 'Random_Seed': 1},
        'CalendDataFile': CALENDARS,
        'Deals': {'Reference': 'defaults',
                  'Deals': {'Children': [node(deal) for deal in deals]}},
        'MergeMarketData': {'MarketDataFile': '', 'ExplicitMarketData': {
            'System Parameters': {'Base_Currency': 'USD', 'Base_Date': WORLD_BASE},
            'Valuation Configuration': {}, 'Price Factors': world()}}}}


def marks(deals, factors=None, system=None):
    """`{reference: mark as hex}` for one document, `factors` beside the world's and `system`
    beside its System Parameters, and the statistics the run reported."""
    job = book(deals)
    job['Calc']['MergeMarketData']['ExplicitMarketData']['Price Factors'].update(factors or {})
    job['Calc']['MergeMarketData']['ExplicitMarketData']['System Parameters'].update(system or {})
    context = derivus.Context()
    context.load_json((json.dumps(job, cls=CustomJsonEncoder), 'defaults'))
    _, answer = context.run_job()
    frame = answer['Results']['mtm']
    return ({str(reference): float(value).hex() for reference, value
             in zip(frame['Reference'], frame['Value'])}, answer['Stats'])


def simulated(deals, curves, factors=None, sigma=0.01, prec=None, system=None, models=None,
              **calculation):
    """A credit Monte Carlo over `deals` held under one netting set, a Hull-White of volatility
    `sigma` on each of `curves` and `models` `{factor: (model, block)}` beside them, `system` beside
    the System Parameters, through the JSON contract and in the torch precision `prec` where one is
    given: `(calculation, output)`."""
    job = json.loads(json.dumps(book(deals), cls=CustomJsonEncoder))
    calc = job['Calc']
    calc['Calculation'] = dict(
        Object='CreditMonteCarlo', Base_Date=calc['Calculation']['Base_Date'], Currency='USD',
        Time_Grid='0d 3m(3m)', Batch_Size=256, Random_Seed=1, Deflation_Interest_Rate='USD',
        DealLevel=True, **calculation)
    calc['Deals']['Deals']['Children'] = [{'Instrument': {'.Deal': {
        'Object': 'NettingCollateralSet', 'Reference': 'NS', 'Netted': 'True',
        'Collateralized': 'False'}}, 'Children': calc['Deals']['Deals']['Children']}]
    market = calc['MergeMarketData']['ExplicitMarketData']
    market['System Parameters'].update(system or {})
    market['Price Factors'].update(json.loads(json.dumps(factors or {}, cls=CustomJsonEncoder)))
    market['Model Configuration'] = {'.ModelParams': {
        'modeldefaults': {'InterestRate': 'HullWhite1FactorInterestRateModel'}, 'modelfilters': {}}}
    market['Price Models'] = {'HullWhite1FactorInterestRateModel.' + curve: {
        'Alpha': 0.05, 'Lambda': 0.0, 'Quanto_FX_Correlation': 0.0,
        'Quanto_FX_Volatility': {'.Curve': {'meta': [], 'data': [[0.0, 0.0], [10.0, 0.0]]}},
        'Sigma': {'.Curve': {'meta': [], 'data': [[0.0, sigma], [10.0, sigma]]}}}
        for curve in curves}
    for factor, (model, block) in (models or {}).items():
        kind, name = factor.split('.', 1)
        market['Model Configuration']['.ModelParams']['modelfilters'].setdefault(kind, []).append(
            [['ID', name], model])
        market['Price Models'][model + '.' + name] = json.loads(json.dumps(block, cls=CustomJsonEncoder))
    context = derivus.Context()
    context.load_json((json.dumps(job), 'simulated'))
    return context.run_job() if prec is None else derivus.run_cmc(context.current_cfg, prec=prec)


def test_a_document_stating_only_its_terms_prices_the_full_one_to_the_bit():
    """TEN TYPES, both ways. Every deal is authored in full, `stripped` takes out the conventions
    it states at the declared value and `furnished` writes every one of them back, and the two
    documents price bit-identically - a swap, a deposit, a FRA, a swaption over its two legs, a
    cap, an FX option, a TARF, an accumulator, a barrier and a netting set over a cashflow list.

    Both documents are checked to have priced: a completion that failed leaves NO row, which is the
    failure this whole seam exists to end, so a missing reference fails before the hexes are read.
    The swaption's own legs mark NaN by construction - they are priced by its `post_process`
    rather than on their own rows - so what those two hold here is the COMPILE, which is where a
    missing field has always shown up.
    """
    def dropped(deal):
        return len(furnished(deal)) - len(stripped(deal)) + sum(
            dropped(child) for child in deal.get('Children', ()))

    lean, full = [stripped(deal) for deal in BOOK], [furnished(deal) for deal in BOOK]
    assert sum(dropped(deal) for deal in BOOK) == 165

    lean_marks, lean_stats = marks(lean)
    full_marks, full_stats = marks(full)
    assert lean_stats.get('Deals Skipped', 0) == full_stats.get('Deals Skipped', 0) == 0
    assert set(lean_marks) == set(full_marks)
    assert {deal['Reference'] for deal in BOOK} <= set(lean_marks)
    assert lean_marks == full_marks, [key for key in lean_marks
                                      if lean_marks[key] != full_marks[key]]
    assert all(lean_marks[deal['Reference']] != float(0).hex() for deal in BOOK
               if deal['Object'] != 'NettingCollateralSet')


def test_a_cap_stated_by_its_terms_prices_its_caplets():
    """The book's cap - a million capped at 4.5% for two years, quarterly on USD-PROJ - and its
    floor twin, stated by their terms alone. By hand, each optionlet Black on its simple ACT/365
    forward off the flat 4.5% curve, 20% lognormal to its reset, paid on the USD discount:

        V = sum_k 1e6 a_k D(e_k) Black(F_k, 4.5%, 0.20 sqrt(s_k))

    6,016.17 capped and 5,528.87 floored. Under a credit Monte Carlo, in a netting set beside a
    swap that outlives them, each keeps its own dates and opens on that value.

    KILLING MUTATION: `finalize_dates` handing a cap stated by its terms its siblings' dates, as it
    hands a cap on its optionlet list its children's: the Monte Carlo prices it past its last
    caplet and skips it.
    """
    cap = next(deal for deal in BOOK if deal['Reference'] == 'CAP')
    floor = dict({key: value for key, value in cap.items() if key != 'Cap_Rate'},
                 Object='FloorDeal', Reference='FLOOR', Floor_Rate=cap['Cap_Rate'])
    dates = [WORLD_BASE + pd.DateOffset(months=3 * k) for k in range(9)]

    def black(sign):
        total = 0.0
        for start, end in zip(dates, dates[1:]):
            accrual = (end - start).days / 365.0
            forward, deviation = math.expm1(0.045 * accrual) / accrual, 0.20 * math.sqrt(
                (start - WORLD_BASE).days / 365.0)
            value = max(sign * (forward - 0.045), 0.0)
            if deviation:
                d1 = math.log(forward / 0.045) / deviation + deviation / 2
                value = sign * (
                    forward * 0.5 * math.erfc(-sign * d1 / math.sqrt(2.0))
                    - 0.045 * 0.5 * math.erfc(-sign * (d1 - deviation) / math.sqrt(2.0)))
            total += 1e6 * accrual * math.exp(-0.04 * (end - WORLD_BASE).days / 365.0) * value
        return total

    valued, _ = marks([cap, floor])
    assert float.fromhex(valued['CAP']) == pytest.approx(black(1.0), rel=1e-12)
    assert float.fromhex(valued['FLOOR']) == pytest.approx(black(-1.0), rel=1e-12)

    calc, _ = simulated([BOOK[0], cap, floor], ('USD', 'USD-PROJ'), Generate_Cashflows='No')
    opened = {deal.Instrument.field['Reference']: float(deal.Calc_res['Value'][0][0].mean())
              for deal in calc.netting_sets.deals() if (deal.Calc_res or {}).get('Value')}
    for reference in ('CAP', 'FLOOR'):
        assert reference in opened, 'the Monte Carlo skipped {}'.format(reference)
        assert opened[reference] == pytest.approx(float.fromhex(valued[reference]), rel=1e-5)


def test_a_convention_whose_fallback_is_another_field_still_means_its_declared_value():
    """THE ONE READER SHAPE A FLAG CANNOT FIX BY ITSELF. `dict.get` never reaches `__missing__`, so
    `self.field.get('Payment_Calendars', self.field['Accrual_Calendars'])` made an OMITTED key mean
    the accrual calendar and a STATED `''` mean Monday-to-Friday - two documents where the store
    now publishes one, `{"value": "", "convention": true}`.

    Read through the declaration instead (`self.field['Payment_Calendars'] or
    self.field['Accrual_Calendars']`) and the two are one document, which is what this measures on
    the engine's own Johannesburg calendar three business days out.

    KILLING MUTATION: the `.get` fallback back. Omitted reads 2027-08-09 and a stated `''` reads
    2027-08-05 - four business days apart on a reval date, and no message anywhere.
    """
    calendars = Config()
    calendars.parse_calendar_file(CALENDARS)
    assert CALENDAR in calendars.holidays

    def paydates(**extra):
        deal = construct_instrument(dict(EQUITY_LEG, **extra), {})
        deal.reset(calendars.holidays)
        return sorted(deal.get_reval_dates())

    omitted, blank = paydates(), paydates(Payment_Calendars='')
    assert omitted == blank, (omitted, blank)
    # and a STATED calendar is that calendar - the declaration is a fallback, not a floor
    assert paydates(Payment_Calendars=CALENDAR) == omitted
    assert paydates(Accrual_Calendars='', Payment_Calendars='') != omitted


def test_a_calendar_file_is_parsed_once_and_again_when_it_changes(tmp_path):
    """A calendar file is parsed once per process: a second config loading it holds the first one's
    holidays, the same objects, and a third loading an edited file reads the edit - the service is
    long-running. The edit is a Johannesburg holiday on a Tuesday, a business day off it.

    Killed by: the parse kept by path alone - the third config reads the unedited calendar."""
    path = tmp_path / 'calendars.cal'
    with open(CALENDARS, encoding='utf-8') as handle:
        text = handle.read()
    path.write_text(text, encoding='utf-8')
    first, second = Config(), Config()
    first.parse_calendar_file(str(path))
    second.parse_calendar_file(str(path))
    assert second.holidays is first.holidays and CALENDAR in first.holidays
    anchor = 'Location="{}" Weekends="Saturday and Sunday" Holidays="'.format(CALENDAR)
    path.write_text(text.replace(anchor, anchor + '2031-06-17|Edited Day, '), encoding='utf-8')
    third = Config()
    third.parse_calendar_file(str(path))
    roll = pd.Timestamp('2031-06-16') + third.holidays[CALENDAR]['businessday']
    assert '2031-06-17' in third.holidays[CALENDAR]['holidays'] and roll == pd.Timestamp('2031-06-18')
    assert pd.Timestamp('2031-06-16') + first.holidays[CALENDAR]['businessday'] == pd.Timestamp(
        '2031-06-17'), 'the edit reached the configs loaded before it'


def test_a_location_stating_no_holidays_has_none(tmp_path):
    """A location whose `Holidays` is empty rolls on its weekends alone: between two locations that
    close on a Tuesday, the one stating nothing is open that day - as the shipped file's LBMA is,
    which states none after Santiago's 598.

    Killed by: the holidays read only where stated, the location before it lending its own - the
    Tuesday rolls to Wednesday and LBMA reads Santiago's."""
    path = tmp_path / 'three.cal'
    path.write_text('<Calendars>{}</Calendars>'.format(''.join(
        '<Calendar Location="{}" Weekends="Saturday and Sunday" Holidays="{}"/>'.format(name, days)
        for name, days in (('First', '2031-06-17|Closed'), ('Silent', ''), ('Last', '2031-06-17|Closed')))),
        encoding='utf-8')
    three, shipped = Config(), Config()
    three.parse_calendar_file(str(path))
    shipped.parse_calendar_file(CALENDARS)
    monday = pd.Timestamp('2031-06-16')
    assert [monday + three.holidays[name]['businessday'] for name in ('First', 'Silent', 'Last')] == [
        pd.Timestamp('2031-06-18'), pd.Timestamp('2031-06-17'), pd.Timestamp('2031-06-18')]
    assert three.holidays['Silent']['holidays'] == {} == shipped.holidays['LBMA']['holidays']
    assert shipped.holidays['Santiago']['holidays']


def wire_default(field):
    """One declared default in the WIRE form an author writes it, spelled here rather than read off
    `engine_default` - the whole point being that the two are different paths to one value.

    A Period is `{'.DateOffset': '3M'}`, a rate `{'.Percent': 0}`, a blank Table its own empty
    container; everything else is the literal a panel shows. A widget also writes a blank Table as
    JSON `null`, which the loader reads as `None` - a different value a reader takes alike.
    """
    if field.type == 'Table' and field.default == 'null':
        tag = {'DateList': '.DateList', 'DateEqualList': '.DateEqualList',
               'CreditSupportList': '.CreditSupportList'}.get(field.tag)
        return {tag: []} if tag else []
    if field.obj in ('Period', 'Percent', 'Basis'):
        return {'.DateOffset' if field.obj == 'Period' else '.' + field.obj: field.default}
    return copy.deepcopy(field.default)


#: Every type's sparse and full block, decoded ONCE through the engine's own JSON reader - a file
#: write per type would be 48 of them.
_DECODED = {}


def decoded_blocks():
    """`{type: (sparse, full)}` - the two documents as the LOADER hands them to a deal.

    The full block states every convention in `wire_default`'s spelling, which is what makes this
    gate say something: the left-hand side of the comparison comes from `engine_default` through
    `DealFields`, the right from the wire form through `Config.read_json`.
    """
    if not _DECODED:
        blocks = {}
        for name, cls in deal_classes():
            fields = schema.declared_fields(cls)
            conventions = schema.deal_defaults(cls)
            lean = {key: wire_default(field) if field.default is not REQUIRED else 'X'
                    for key, field in fields.items() if key not in conventions}
            lean['Object'] = name
            blocks[name] = (lean, dict(
                {key: wire_default(fields[key]) for key in conventions}, **lean))
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_defaults_probe.json')
        document = book([block for pair in blocks.values() for block in pair])
        try:
            with open(path, 'wt', encoding='utf-8', newline='\n') as handle:
                json.dump(json.loads(json.dumps(document, cls=CustomJsonEncoder)), handle)
            read = Config().read_json(path)
        finally:
            if os.path.isfile(path):
                os.remove(path)
        walked = [node['Instrument'].payload for node in
                  read['Calc']['Deals']['Deals']['Children']]
        for i, name in enumerate(blocks):
            _DECODED[name] = (walked[2 * i], walked[2 * i + 1])
    return _DECODED


@pytest.mark.parametrize('name,cls', deal_classes())
def test_completion_is_identity_for_every_declared_type(name, cls):
    """PER TYPE, without a market: the sparse block states what must be stated, the full one states
    every convention TOO, in the wire form an author writes it, and both go through the engine's
    own JSON reader. The two then answer every declared key the SAME - the sparse one by
    completing from the declaration, the full one by carrying what the loader decoded. What differs
    is exactly the conventions: `in`, `len` and the JSON round trip see the authored keys and
    nothing else.

    KILLING MUTATION: any spelling in `engine_default` that is not the loader's own - the builder's
    `uncoerced-period` is one, and a Percent or a Table would read the same way.
    """
    lean, full = decoded_blocks()[name]
    fields = schema.declared_fields(cls)
    conventions = schema.deal_defaults(cls)

    sparse, whole = schema.DealFields(lean, cls), schema.DealFields(full, cls)
    assert set(whole) - set(sparse) == set(conventions)
    assert len(whole) - len(sparse) == len(conventions)
    for key in fields:
        assert reading(sparse[key]) == reading(whole[key]), key
    assert set(sparse) == set(lean), 'a read entered the sparse block'
    round_trip = json.loads(json.dumps(sparse, cls=CustomJsonEncoder))
    assert set(round_trip) == set(lean)


# --------------------------------------------------------------------------------------------
# the store says which is which
# --------------------------------------------------------------------------------------------
def test_the_store_publishes_the_flag_and_the_must_state_list():
    """What a host and a panel both ask: `required` is everything that MUST BE STATED - REQUIRED
    and every placeholder - and `convention` marks the rest, one or the other on every declared
    field of every type. A placeholder keeps its declared `value`, which is what a blank panel
    shows; a REQUIRED field has none to show."""
    store = schema.mapping['Instrument']
    for deal_type, sections in store['types'].items():
        fields = {}
        for section in sections:
            fields.update(store['sections'][section])
        for key, meta in fields.items():
            assert meta.get('required', False) != meta.get('convention', False), (deal_type, key)
    swap = {}
    for section in store['types']['SwapInterestDeal']:
        swap.update(store['sections'][section])
    assert sorted(key for key, meta in swap.items() if meta.get('required')) == [
        'Currency', 'Effective_Date', 'Maturity_Date', 'Object', 'Pay_Rate_Type', 'Principal',
        'Swap_Rate']
    assert swap['Swap_Rate']['value'] == 0.0 and swap['Object']['value'] == ''
    assert swap['Pay_Timing']['convention'] is True and swap['Pay_Timing']['value'] == 'End'
