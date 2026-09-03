"""Barrier state as a fold over fixings, end to end through the JSON contract and nothing else.

A discretely-monitored barrier deal carries its own history in `Barrier_Dates`: each row is
`[date, Observed]`, the close that date fixed at. The engine folds those rows at COMPILE. Where a
crossing leaves no decisions the deal is not priced through a dead branch - it compiles as the deal
it became, built as a document of that type.

WHY NOT A BRANCH IN THE PRICER. `pv_discrete_barrier_option` already carries one for the CVA outer
path (`torch.where(row_barrier_hit, hit_value, oss_result)`), and its `hit_value` leg is a SECOND
spelling of the same European the in-out-parity leg values. The two spellings shipped once
disagreeing and marked every already-hit row at +1432% of its value. A t0 crossing is a scalar
fact, not a per-scenario one, so it needs no branch at all - and the substitute reads the vanilla's
own pricer, which cannot disagree with itself.

THE ORACLES ARE DOCUMENTS. A knocked-in barrier's oracle is the plain `EquityOptionDeal` written
out longhand, and the gate is EQUALITY TO THE BIT rather than a tolerance: two documents that price
the same instrument through the same closed form have nothing to be near about. A knocked-out
barrier's oracle is arithmetic - the rebate is a certain cashflow on a date the fold names, so a
crossing dated ON the base date marks at exactly the rebate.

MEASURED, what the fold is worth on the document below (vanilla 10.664675, rebate 40, notional 1).
The knocked-in call reads 18.656964 unfolded against 10.664675 folded - **+74.9%** - and the
knocked-in Up-and-In reads 31.250790, **+193.0%**, the surplus being the never-knocked-in rebate leg
the deal stopped owning the day it knocked in. The knock-outs are worth exactly their 40.00 where
the unfolded document marks 30.980621 and 18.229263 of option value it no longer owns.

DEGENERACY CHECKLIST: both barrier directions and both knock directions, each priced crossed AND
uncrossed; a close exactly ON the level and one a tick the surviving side of it; a crossing dated ON
the base date and one strictly before it; the rebate live and dead; both deal types; r = 4% against
q = 2%, so the carry is 2% and neither rate is zero.

UNDER A CREDIT MONTE CARLO the same identities hold on the whole profile, and the unaffected
documents are asserted bit-identical there too. The grid is built from the ORIGINAL deal's dates
before the fold runs, so a folded document shares the vanilla's grid only where its monitoring dates
are all behind the base date - which is the shape those gates use.

OUT OF SCOPE HERE, and named: the CONTINUOUS barrier's history. `utils.bars_touched` is the
predicate a daily `(low, high)` series is folded with and it is gated below on authored series, but
where the bars come from - hydrating `(index, date, source)` facts from the log or the market data -
is spine increment 4's.
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
from derivus.config import CustomJsonEncoder

BASE = pd.Timestamp('2024-06-28')
SPOT, R_USD, Q_EQ, SIGMA = 100.0, 0.04, 0.02, 0.25
STRIKE, UNITS, REBATE = 100.0, 1.0, 40.0
EXPIRY_D = 365
BARRIER_UP, BARRIER_DOWN = 115.0, 90.0

FACTORS = {
    'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Spot': 1.0},
    'InterestRate.USD': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                         'Curve': utils.Curve([], [[0.0, R_USD], [5.0, R_USD]])},
    'EquityPrice.EQ': {'Spot': SPOT, 'Currency': 'USD', 'Interest_Rate': 'USD',
                       'Issuer': '', 'Respect_Default': 'No', 'Jump_Level': 0.0},
    'DividendRate.EQ': {'Currency': 'USD', 'Floor': None,
                        'Curve': utils.Curve([], [[0.0, Q_EQ], [5.0, Q_EQ]])},
    'VolatilityGrid.EQ': {'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness',
                          'Surface': utils.Curve([], [[m, t, SIGMA] for m in (0.6, 1.0, 1.4)
                                                      for t in (0.02, 2.0)])}}


def day(offset):
    return BASE + pd.DateOffset(days=offset)


#: monthly monitoring from three months BEFORE the base date to just short of expiry, so the
#: schedule straddles today: three resolved rows and eleven still to come
PAST_DAYS = [-90, -60, -30]
FUTURE_DAYS = list(range(30, 331, 30))


def rows(observed_by_day=None):
    """The monitoring table: every past row carries a close, every future row is blank."""
    observed_by_day = observed_by_day or {}
    return ([[day(d), observed_by_day.get(d, SPOT)] for d in PAST_DAYS] +
            [[day(d), ''] for d in FUTURE_DAYS])


def barrier(barrier_type, monitoring, rebate=REBATE, reference='BR'):
    up = 'Up' in barrier_type
    return {'Object': 'EquityBarrierOption', 'Reference': reference, 'Currency': 'USD',
            'Payoff_Currency': 'USD', 'Equity': 'EQ', 'Dividends': 'EQ', 'Discount_Rate': 'USD',
            'Equity_Volatility': 'EQ', 'Buy_Sell': 'Buy', 'Option_Type': 'Call',
            'Strike_Price': STRIKE, 'Units': UNITS, 'Cash_Rebate': rebate,
            'Expiry_Date': day(EXPIRY_D), 'Barrier_Type': barrier_type,
            'Barrier_Price': BARRIER_UP if up else BARRIER_DOWN,
            'Barrier_Monitoring_Frequency': pd.DateOffset(days=0),
            'Barrier_Dates': monitoring}


def vanilla(reference='BR'):
    """The document a knocked-in call BECOMES, written out longhand."""
    return {'Object': 'EquityOptionDeal', 'Reference': reference, 'Currency': 'USD',
            'Payoff_Currency': 'USD', 'Equity': 'EQ', 'Dividends': 'EQ', 'Discount_Rate': 'USD',
            'Equity_Volatility': 'EQ', 'Buy_Sell': 'Buy', 'Option_Type': 'Call',
            'Strike_Price': STRIKE, 'Units': UNITS, 'Expiry_Date': day(EXPIRY_D)}


def job(deal, calc=None):
    return {'Calc': {
        'Calculation': dict({'Object': 'BaseValuation', 'Base_Date': BASE, 'Currency': 'USD',
                             'MCMC_Simulations': 1024, 'Random_Seed': 1}, **(calc or {})),
        'Deals': {'Tag_Titles': '', 'Reference': 'fold',
                  'Deals': {'Children': [{'Instrument': {'.Deal': d}}
                                         for d in (deal if isinstance(deal, list) else [deal])]}},
        'MergeMarketData': {'MarketDataFile': '', 'ExplicitMarketData': {
            'System Parameters': {'Base_Currency': 'USD', 'Base_Date': BASE},
            'Valuation Configuration': {}, 'Price Factors': FACTORS}}}}


def run(deal, calc=None):
    cx = derivus.Context()
    cx.load_json((json.dumps(job(deal, calc), cls=CustomJsonEncoder), 'fold'))
    _, out = cx.run_job()
    return out


def mtm(deal, reference='BR', calc=None):
    """One deal's own row. A deal that left no row was skipped or expired, and says so."""
    rows_out = run(deal, calc)['Results']['mtm']
    own = rows_out[rows_out['Reference'] == reference]['Value']
    assert len(own) == 1, 'the deal left no row in Results: it was skipped, not priced'
    return float(own.iloc[0])


def no_row(deal, reference='BR', calc=None):
    rows_out = run(deal, calc)['Results']['mtm']
    return rows_out[rows_out['Reference'] == reference].empty


# --------------------------------------------------------------------------------------------
# the Observed column: what a blank means on each side of the base date
# --------------------------------------------------------------------------------------------
def test_a_blank_observed_before_the_base_date_refuses_by_name():
    """The absence of a fixing is not the absence of a hit. A monitoring date already past with no
    close recorded is a schedule the engine has no verdict for, and walking past it silently prices
    the deal ALIVE on no evidence - which is what the one-column table did.

    FATAL rather than skipped: a refusal swallowed into `Deals Skipped` marks the trade at nothing
    on a job that reports success, which is the same silence in a different costume.
    """
    for barrier_type in ('Down_And_In', 'Down_And_Out', 'Up_And_In', 'Up_And_Out'):
        # the one-column form every document written before the column carried
        blank = [day(d) for d in PAST_DAYS] + [day(d) for d in FUTURE_DAYS]
        with pytest.raises(Exception) as refusal:
            run(barrier(barrier_type, blank))
        message = str(refusal.value)
        assert 'Observed' in message and 'blank' in message, message
        assert '2024-03-30' in message, message                 # the first offending row, by date
        assert 'Barrier_Dates' in message, message              # the remedy names where to write it


def test_a_blank_observed_after_the_base_date_is_nothing():
    """A future row has nothing to observe yet, so the column's presence must not move a price.
    Priced to the BIT against the same schedule with no closes at all - the only rows that differ
    are ones neither reading resolves.
    """
    future_only = [[day(d), ''] for d in FUTURE_DAYS]
    bare = [day(d) for d in FUTURE_DAYS]
    for barrier_type in ('Down_And_In', 'Down_And_Out', 'Up_And_In', 'Up_And_Out'):
        with_column = mtm(barrier(barrier_type, future_only))
        without = mtm(barrier(barrier_type, bare))
        assert with_column == without, (barrier_type, with_column, without)
        assert abs(with_column) > 1.0, 'the fixture must have something to lose'


def test_the_authored_block_is_untouched_by_the_column():
    """`plan_hash` and the factor universe read what the AUTHOR wrote, and `DealFields` holds
    exactly that. A document that does not use `Observed` therefore hashes to the same bytes it
    did before the column existed, which is the declared-defaults discipline restated.
    """
    from derivus.instruments import construct_instrument
    block = barrier('Down_And_Out', [day(d) for d in FUTURE_DAYS])
    deal = construct_instrument(dict(block), {})
    assert set(deal.field) == set(block), 'a declaration entered the authored block'
    round_trip = json.loads(json.dumps(deal.field, cls=CustomJsonEncoder))
    assert round_trip.keys() == deal.field.keys()


# --------------------------------------------------------------------------------------------
# the fold: what the deal became
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize('barrier_type,crossing', [('Down_And_In', 85.0), ('Up_And_In', 120.0)])
def test_a_knocked_in_barrier_prices_as_the_vanilla_s_own_document(barrier_type, crossing):
    """A knock-in that has knocked in IS the vanilla, and prices through the vanilla's own pricer -
    so the two documents agree to the last bit rather than to a tolerance.

    The unfolded reading is quoted beside it: it is the number this gate exists to have stopped.
    """
    knocked = mtm(barrier(barrier_type, rows({-60: crossing})))
    plain = mtm(vanilla())
    assert knocked == plain, (barrier_type, knocked, plain)

    unfolded = mtm(barrier(barrier_type, rows()))          # nothing crossed - still a barrier
    assert abs(unfolded - plain) > 0.5 * abs(plain), (
        'the fold must be worth something: unfolded {:.4f} vs vanilla {:.4f}'.format(
            unfolded, plain))


@pytest.mark.parametrize('barrier_type,crossing', [('Down_And_Out', 85.0), ('Up_And_Out', 120.0)])
def test_a_knocked_out_barrier_pays_its_rebate_on_the_crossing_date(barrier_type, crossing):
    """A knock-out that has knocked out owes its rebate and nothing else. Dated ON the base date the
    cashflow is certain and undiscounted, so the mark is the rebate EXACTLY - arithmetic, not a
    tolerance - and it flips sign with `Buy_Sell` because a sold knock-out PAYS the rebate.
    """
    today = [[BASE, crossing]] + [[day(d), ''] for d in FUTURE_DAYS]
    assert mtm(barrier(barrier_type, today)) == REBATE

    sold = dict(barrier(barrier_type, today), Buy_Sell='Sell')
    assert mtm(sold) == -REBATE

    no_rebate = barrier(barrier_type, today, rebate=0.0)
    assert mtm(no_rebate) == 0.0


def test_the_rebate_is_absolute_cash_and_the_vanilla_carries_the_units():
    """`Cash_Rebate` is the deal's cash, not per unit (`pv_discrete_barrier_option` divides it by the
    size to price it per unit), so a folded knock-out marks the rebate whatever `Units` says, while
    a folded knock-in carries `Units` into the vanilla it became. At Units = 1 the two conventions
    are indistinguishable, which is why every reading above needed this one beside it.
    """
    units = 3.0
    today = [[BASE, 85.0]] + [[day(d), ''] for d in FUTURE_DAYS]
    assert mtm(dict(barrier('Down_And_Out', today), Units=units)) == REBATE
    knocked = mtm(dict(barrier('Down_And_In', rows({-60: 85.0})), Units=units))
    plain = mtm(dict(vanilla(), Units=units))
    assert knocked == plain and knocked != mtm(vanilla()), (knocked, plain)


@pytest.mark.parametrize('barrier_type,crossing', [('Down_And_Out', 85.0), ('Up_And_Out', 120.0)])
def test_a_knock_out_whose_rebate_already_settled_leaves_no_mark(barrier_type, crossing):
    """A crossing STRICTLY before the base date paid its rebate then. What remains is a deal with
    no cashflow left, which is what expiry means - and the unfolded document, which would still be
    marking option value it does not own, is quoted against it.
    """
    assert no_row(barrier(barrier_type, rows({-60: crossing})))
    alive = mtm(barrier(barrier_type, rows()))
    assert abs(alive) > 1.0, ('the unfolded document marks {:.4f} of an option that '
                              'knocked out three months ago'.format(alive))


@pytest.mark.parametrize('barrier_type', ['Down_And_In', 'Down_And_Out', 'Up_And_In', 'Up_And_Out'])
def test_an_observed_close_that_does_not_cross_changes_nothing(barrier_type):
    """The other side of every corner above. A resolved row that did NOT cross leaves the deal the
    deal it was, and prices to the bit of the same document with those rows deleted - the pricer's
    first monitored date is already past them either way.
    """
    with_history = mtm(barrier(barrier_type, rows()))
    without = mtm(barrier(barrier_type, [[day(d), ''] for d in FUTURE_DAYS]))
    assert with_history == without, (barrier_type, with_history, without)


@pytest.mark.parametrize('barrier_type,level', [('Down_And_In', BARRIER_DOWN),
                                                ('Up_And_In', BARRIER_UP)])
def test_a_close_exactly_ON_the_level_has_crossed(barrier_type, level):
    """The crossing corner. Touching IS crossing on both sides, so a close landing exactly on the
    barrier resolves the deal - the same weak inequality the pricer's own survival test uses
    (`s < H` for Up, `s > H` for Down: equality does NOT survive).
    """
    assert mtm(barrier(barrier_type, rows({-60: level}))) == mtm(vanilla())
    # and a close one tick the SURVIVING side of it does not
    inside = level * (1.0 - 1e-9) if 'Up' in barrier_type else level * (1.0 + 1e-9)
    assert mtm(barrier(barrier_type, rows({-60: inside}))) == mtm(barrier(barrier_type, rows()))


def test_a_folded_state_registers_no_boundary_correction():
    """A t0 crossing is DATA, so it registers nothing and the sensitivity architecture is untouched
    by the fold - which is the whole reason a scalar fact must not become a pricer branch.

    Read through the document rather than by inspecting `shared`: `Greeks: 'All'` refuses on a deal
    that registered a boundary correction, so the unfolded barrier refuses BY NAME and the folded
    one answers - with the vanilla's own second-order block, to the bit.
    """
    with pytest.raises(Exception) as refusal:
        mtm(barrier('Down_And_In', rows()), calc={'Greeks': 'All'})
    assert 'boundary correction' in str(refusal.value), str(refusal.value)

    folded = mtm(barrier('Down_And_In', rows({-60: 85.0})), calc={'Greeks': 'All'})
    assert folded == mtm(vanilla(), calc={'Greeks': 'All'})


def test_the_digital_barrier_folds_to_its_own_vanilla():
    """`EquityBarrierBinaryOption` takes the same seam and lands on `EquityBinaryOption`, which is
    a different pricer from the one the non-digital lands on - so the substitute is chosen by the
    deal, not by the fold.
    """
    cash = 250.0
    digital = {'Object': 'EquityBarrierBinaryOption', 'Reference': 'BR', 'Currency': 'USD',
               'Payoff_Currency': 'USD', 'Equity': 'EQ', 'Dividends': 'EQ',
               'Discount_Rate': 'USD', 'Equity_Volatility': 'EQ', 'Buy_Sell': 'Buy',
               'Option_Type': 'Call', 'Strike_Price': STRIKE, 'Cash_Payoff': cash,
               'Expiry_Date': day(EXPIRY_D), 'Settlement_Date': day(EXPIRY_D),
               'Barrier_Type': 'Down_And_In', 'Barrier_Price': BARRIER_DOWN,
               'Barrier_Dates': rows({-60: 85.0})}
    plain = {'Object': 'EquityBinaryOption', 'Reference': 'BR', 'Currency': 'USD',
             'Payoff_Currency': 'USD', 'Equity': 'EQ', 'Dividends': 'EQ',
             'Discount_Rate': 'USD', 'Equity_Volatility': 'EQ', 'Buy_Sell': 'Buy',
             'Option_Type': 'Call', 'Strike_Price': STRIKE, 'Cash_Payoff': cash,
             'Expiry_Date': day(EXPIRY_D), 'Settlement_Date': day(EXPIRY_D)}
    assert mtm(digital) == mtm(plain)


def test_the_continuous_barrier_is_left_alone():
    """No `Barrier_Dates` is a CONTINUOUSLY monitored deal, whose history is a fold over daily bars
    rather than over closes. The bar source is spine increment 4's, so the fold must not touch this
    document - not even to refuse it.
    """
    continuous = barrier('Down_And_Out', [])
    assert abs(mtm(continuous)) > 1.0


# --------------------------------------------------------------------------------------------
# the same fold under a credit Monte Carlo: the substitute walks its own document's profile
# --------------------------------------------------------------------------------------------
CMC_CALC = {'Object': 'CreditMonteCarlo', 'Time_grid': '0d 12m(1m)', 'Batch_Size': 256,
            'Simulation_Batches': 1, 'Deflation_Interest_Rate': 'USD'}


def cmc_profile(deal):
    """The deal's exposure profile (dates x scenarios) under a credit Monte Carlo on a GBM spot."""
    doc = job(deal, CMC_CALC)
    market = doc['Calc']['MergeMarketData']['ExplicitMarketData']
    market['Price Models'] = {'GBMAssetPriceModel.EQ': {'Vol': SIGMA, 'Drift': 0.0}}
    market['Model Configuration'] = {'.ModelParams': {
        'modeldefaults': {'EquityPrice': 'GBMAssetPriceModel'}, 'modelfilters': {}}}
    cx = derivus.Context()
    cx.load_json((json.dumps(doc, cls=CustomJsonEncoder), 'fold'))
    _, out = cx.run_job()
    return out['Results']['mtm']


def same_profile(a, b):
    return a.shape == b.shape and np.array_equal(a.values, b.values)


def rebate_cashflow(amount, payment_date, reference='BR'):
    """The document a knocked-out barrier BECOMES, written out longhand."""
    return {'Object': 'FixedCashflowDeal', 'Reference': reference, 'Currency': 'USD',
            'Discount_Rate': 'USD', 'Amount': amount, 'Payment_Date': payment_date}


@pytest.mark.parametrize('barrier_type,crossing', [('Down_And_In', 85.0), ('Up_And_In', 120.0)])
def test_a_knocked_in_barrier_walks_the_vanilla_s_own_exposure_profile(barrier_type, crossing):
    """The base-valuation identity on the grid a credit Monte Carlo walks: with every monitoring
    date behind the base date the folded document and the vanilla's own share one grid, one path
    set and one pricer, so the profile is the vanilla's to the bit."""
    past_only = rows({-60: crossing})[:len(PAST_DAYS)]
    folded = cmc_profile(barrier(barrier_type, past_only))
    plain = cmc_profile(vanilla())
    assert same_profile(folded, plain), (folded.shape, plain.shape)
    assert np.isfinite(folded.values).all() and folded.values.std() > 0, 'a profile, not a constant'


@pytest.mark.parametrize('barrier_type,crossing', [('Down_And_Out', 85.0), ('Up_And_Out', 120.0)])
def test_a_knocked_out_barrier_walks_its_rebate_s_own_exposure_profile(barrier_type, crossing):
    """A knock-out crossed ON the base date is the `FixedCashflowDeal` paying its rebate today,
    written out longhand and run under the same credit Monte Carlo - to the bit.

    Each sits beside the same vanilla. Alone, a book whose only deal folded to a static cashflow
    while the factor the original discovered is still simulated does not frame under a credit
    Monte Carlo (a (1, 1) root against a (T, B) grid) - a roadmap row, not this gate's.
    """
    today = [[day(d), SPOT] for d in PAST_DAYS] + [[BASE, crossing]]
    folded = cmc_profile([barrier(barrier_type, today), vanilla('V')])
    plain = cmc_profile([rebate_cashflow(REBATE, BASE), vanilla('V')])
    assert same_profile(folded, plain), (folded.shape, plain.shape)
    assert not same_profile(folded, cmc_profile(vanilla('V'))), 'the rebate must be in the profile'


@pytest.mark.parametrize('barrier_type', ['Down_And_In', 'Down_And_Out', 'Up_And_In', 'Up_And_Out'])
def test_an_unaffected_document_is_bit_identical_under_a_credit_monte_carlo(barrier_type):
    """The other half of the rule: a document the fold does not touch must not move. A blank
    column on future rows, and past rows that did not cross, both walk the profile the bare
    one-column schedule walks, bit for bit."""
    bare = cmc_profile(barrier(barrier_type, [day(d) for d in FUTURE_DAYS]))
    with_column = cmc_profile(barrier(barrier_type, [[day(d), ''] for d in FUTURE_DAYS]))
    with_history = cmc_profile(barrier(barrier_type, rows()))
    assert same_profile(bare, with_column), (bare.shape, with_column.shape)
    assert same_profile(bare, with_history), (bare.shape, with_history.shape)
    assert bare.values.std() > 0, 'the fixture must have something to lose'


# --------------------------------------------------------------------------------------------
# the bar predicate, on authored series - the continuous fold's other half
# --------------------------------------------------------------------------------------------
def test_a_bar_series_says_whether_the_level_was_touched():
    """A daily `(low, high)` bar BRACKETS every intraday print of its day, so the verdict is exact
    without the prints. Touching IS crossing: the inequalities are weak, so an exact-touch day is a
    hit and a gap that opens through the level is one whether or not it ever printed there.
    """
    quiet = [(98.0, 102.0), (99.0, 101.0), (97.5, 100.5)]
    assert not utils.bars_touched(quiet, 90.0, barrier_up=False)
    assert not utils.bars_touched(quiet, 115.0, barrier_up=True)

    # a crossing HIGH is an up touch and says nothing about a down level
    crossing_high = quiet + [(101.0, 116.0)]
    assert utils.bars_touched(crossing_high, 115.0, barrier_up=True)
    assert not utils.bars_touched(crossing_high, 90.0, barrier_up=False)

    # a crossing LOW is a down touch and says nothing about an up level
    crossing_low = quiet + [(89.0, 99.0)]
    assert utils.bars_touched(crossing_low, 90.0, barrier_up=False)
    assert not utils.bars_touched(crossing_low, 115.0, barrier_up=True)

    # EXACT TOUCH, both directions: the level is reached and not passed
    assert utils.bars_touched([(95.0, 115.0)], 115.0, barrier_up=True)
    assert utils.bars_touched([(90.0, 105.0)], 90.0, barrier_up=False)

    # a GAP DAY that opens beyond the level: the whole bar is on the far side
    assert utils.bars_touched(quiet + [(118.0, 124.0)], 115.0, barrier_up=True)
    assert utils.bars_touched(quiet + [(80.0, 86.0)], 90.0, barrier_up=False)
    # ... and the same gap is not a touch of the level it jumped AWAY from
    assert not utils.bars_touched(quiet + [(118.0, 124.0)], 90.0, barrier_up=False)

    assert not utils.bars_touched([], 90.0, barrier_up=False)


def test_an_inverted_bar_brackets_nothing_and_refuses():
    """A low above its high is not a range, and a predicate that answered it would be reading a
    fact that does not exist."""
    with pytest.raises(utils.UnpriceableSchedule) as refusal:
        utils.bars_touched([(101.0, 99.0)], 90.0, barrier_up=False)
    assert 'brackets nothing' in str(refusal.value)
