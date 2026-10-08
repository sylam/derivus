"""A deal the run cannot value: skipped and counted, or a refusal that is FATAL.

`Deal.calculate`'s canonical guard turns a pricing failure into a zero mark, counted once under
`Deals Skipped` - load-bearing for base valuation and credit Monte Carlo, where a portfolio of
thousands must not die on one unpriceable deal - and `Exclude_Deals_With_Missing_Market_Data`
decides what a compile that cannot read a deal does with it: `Yes`, the default, drops it, `No` keeps
it marked at zero. A structure the run could not value is counted once under `Structs Skipped`.

`utils.is_fatal_pricing_error` names what no guard may swallow: running out of memory, a schedule
the calculation never bound, and `utils.UnpriceableSchedule` - the DOCUMENT is wrong, and a named
refusal swallowed into a zero mark has said nothing at all. A hedge book is stricter still: a leg
that fails to compile kills the run, because a skipped tradable shrinks the solver's menu and a
skipped liability shrinks the target it is hedging.

NOT GATED HERE: a failure INSIDE an inner-MC fork - an out-of-memory in a tradable's pricing reaching
the caller rather than reading as `F_t1 = 0`, a skipped tradable leaving a loud hole, the fork's
published row blocks answering as a joined grid would and refusing a late write - each was read by
rebinding an engine function, and no document can inject the failure.
"""
import copy
import json as jsonlib
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import pytest

import derivus as rf
import test_declared_defaults as book
from derivus import utils
from derivus.config import CustomJsonEncoder

# the one-cashflow job every service gate is built on, reused rather than re-authored: the
# degenerate leg below needs a real USD/ZAR market and a real `BaseValuation`, and that document
# already is one
from test_service import BASE as SERVICE_BASE, dump as service_dump, job as service_job

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'fixtures', 'policy_test_simulate_only.json')


def _cfg(t_min):
    cfg = jsonlib.load(open(FIXTURE))
    calc = cfg['Calc']['Calculation']
    calc.update({'Execution_Mode': 'solve_hedge', 'Batch_Size': 8, 'Simulation_Batches': 2,
                 'Inner_Sub_Batch': 4,
                 'Inner_MC_Enabled': 'Yes', 'Random_Seed': 1234})
    calc['Hedging_Problem']['Randomize_Initial_State'] = 'Yes'
    calc['Hedging_Problem']['Solver'] = {
        'Object': 'DiffSolverV2', 'Training_Action_Grid_Levels_Per_Axis': 3,
        'Training_Action_Chunk_Size': 64, 'T_Min': t_min, 'DiffV2_Fit_Iters': 2}
    return cfg


def _skipping(cfg):
    """`cfg` stating `Exclude_Deals_With_Missing_Market_Data: Yes` over its market file's `No`, so
    a deal that cannot be read or priced reaches the hedge's own checks."""
    cfg = copy.deepcopy(cfg)
    cfg['Calc']['MergeMarketData']['ExplicitMarketData']['System Parameters'][
        'Exclude_Deals_With_Missing_Market_Data'] = 'Yes'
    return cfg


def _run(cfg, name):
    """One JSON-only hedge run."""
    cx = rf.Context()
    cx.load_json((jsonlib.dumps(cfg), f'{name}.json'))
    cx.run_job()


def test_a_hedge_book_leg_that_fails_to_compile_kills_the_run():
    """The COMPILE-level arm. `add_deal_to_structure`'s skip-and-continue is the reporting book's
    contract, but on a hedge book a skipped tradable shrinks the solver's menu and a skipped
    liability halves the target, so the solve reports a confident answer to a different problem.
    Measured before the guard: an APS leg whose basis law could not state its projection dropped n*
    from -44.8 to -22.1 with nothing but an ERROR log. Both roles raise, naming the leg, under a
    document dropping such a deal and under the fixture's own `No`, which keeps it at zero - a leg
    kept at zero is a leg that did not compile.

    Killing mutation: the hedge book's compile check dropped, the leg skipped and the solve run.
    """
    for role, block, patch in (
            ('liability', 'Liabilities', {'Currency': 'XXX'}),
            ('tradable', 'Tradable_Instruments', {'Sampling_Type': 'NOPE', 'Currency': 'XXX'})):
        cfg = _cfg(t_min=113)
        hp = cfg['Calc']['Calculation']['Hedging_Problem']
        src = hp[block].setdefault('FloatingEnergyDeal', {})
        leg = copy.deepcopy(cfg['Calc']['Calculation']['Hedging_Problem']['Liabilities']
                            ['FloatingEnergyDeal']['PLAT_JUL29'])
        leg.update(patch)
        src['BROKEN_LEG'] = leg
        for document in (copy.deepcopy(cfg), _skipping(cfg)):
            with pytest.raises(Exception, match=f'{role} legs failed to compile.*BROKEN_LEG'):
                _run(document, f'broken_{role}')


# --------------------------------------------------------------------------------------------
# A schedule the engine will not guess at: refused by name, and the refusal is fatal
# --------------------------------------------------------------------------------------------
def _degenerate_float_leg(rate_end_is_start):
    """A real `CFFloatingInterestListDeal`: one quarterly coupon, one reset.

    `rate_end_is_start` is the whole fixture. The reset's rate window is either the coupon's own
    three months (healthy) or a single instant (degenerate) - and nothing else about the two deals
    differs, so what the gate below measures is the window and not the deal.
    """
    start = SERVICE_BASE + pd.DateOffset(months=3)
    end = start + pd.DateOffset(months=3)
    return {'Object': 'CFFloatingInterestListDeal', 'Reference': 'FLT-DEGENERATE',
            'Currency': 'USD', 'Discount_Rate': 'USD', 'Forecast_Rate': 'USD', 'Buy_Sell': 'Buy',
            'Cashflows': {'Compounding_Method': 'None', 'Averaging_Method': 'Average_Rate',
                          'Properties': [], 'Items': [{
                              'Payment_Date': end, 'Accrual_Start_Date': start,
                              'Accrual_End_Date': end, 'Accrual_Year_Fraction': 0.25,
                              'Notional': 1_000_000.0, 'Margin': utils.Basis(0.0),
                              'Fixed_Amount': 0.0,
                              'Resets': [[start, start, start if rate_end_is_start else end, 0.25,
                                          'ACT_365', 0.0, utils.Percent(0.0)]]}]}}


def _priced(deal, name):
    """The deal's own row out of a real `BaseValuation` run - JSON in, `run_job` out."""
    context = rf.Context()
    context.load_json((service_dump(service_job(deals=(deal,))), name))
    _, result = context.run_job()
    table = result['Results']['mtm']
    return result['Stats'], dict(zip(table['Reference'], table['Value']))


def test_a_degenerate_reset_window_refuses_by_name_and_the_run_fails_loud():
    """`TensorCashFlows.float` read `cashflow['Rate_Tenor']` whenever a reset's rate window had zero
    length - a key no `Row` declares and nothing writes - so the one document that reached it died
    `KeyError: 'Rate_Tenor'`, which `add_deal_to_structure` caught.

    MEASURED on this pair of documents with the old read put back:

        ERROR:FLT-DEGENERATE:CFFloatingInterestListDeal ('Rate_Tenor',) - Skipped
        Stats: {'Deals Skipped': 1}     mtm: root 0.0     job: SUCCEEDED

    - a deal gone from the report, a book netting to nothing, and an exit code saying everything was
    fine. The healthy twin prices 4948.879641 on the same market data.

    NOW: `utils.UnpriceableSchedule`, naming the deal, the fixing, the cashflow and the instant the
    window collapsed to, with the remedy, and FATAL. The tenor is NOT derived: a rate window is not
    the accrual window and the schedule states no tenor.

    THREE THINGS IT MUST NOT BE, each asserted: not a `KeyError`, not a skipped deal, not a zero
    mark.

    Killing mutation: `is_fatal_pricing_error` no longer naming `UnpriceableSchedule`, so the
    compile guard swallows the refusal into a skipped deal.
    """
    healthy_stats, healthy = _priced(_degenerate_float_leg(False), 'healthy_float_leg')
    assert healthy_stats.get('Deals loaded') == 1 and 'Deals Skipped' not in healthy_stats
    assert healthy['FLT-DEGENERATE'] == pytest.approx(4948.879641, rel=1e-9), healthy
    assert healthy['FLT-DEGENERATE'] != 0.0, 'the healthy twin is worth nothing - nothing is gated'

    with pytest.raises(utils.UnpriceableSchedule) as refused:
        _priced(_degenerate_float_leg(True), 'degenerate_float_leg')

    message = str(refused.value)
    assert 'FLT-DEGENERATE' in message, 'the refusal does not name the deal'
    assert 'rate window that starts and ends on' in message
    assert str((SERVICE_BASE + pd.DateOffset(months=3)).date()) in message, (
        'the refusal does not carry the degenerate window\'s own date')
    assert 'Author the reset\'s rate end date after its rate start' in message, (
        'the refusal states no remedy')
    assert not isinstance(refused.value, KeyError)


# --------------------------------------------------------------------------------------------
# The document's own switch: a deal the compile cannot read is dropped, or kept at zero
# --------------------------------------------------------------------------------------------
GOOD = book.fx_leg('FXOptionDeal', 'FXO', Expiry_Date=book.WORLD_EXPIRY)
NO_STRIKE = {key: value for key, value in book.fx_leg(
    'FXOptionDeal', 'FXO_NO_STRIKE', Expiry_Date=book.WORLD_EXPIRY).items() if key != 'Strike_Price'}
NO_SURFACE = book.fx_leg('FXOptionDeal', 'FXO_NO_SURFACE', Expiry_Date=book.WORLD_EXPIRY,
                         FX_Volatility='EUR.JPY')


def _switched(switch, deals):
    """The declared-defaults book over `deals` with `Exclude_Deals_With_Missing_Market_Data` stated
    `switch`, or left out at None: its Stats and `{reference: mark}`."""
    job = book.book(deals)
    if switch is not None:
        job['Calc']['MergeMarketData']['ExplicitMarketData']['System Parameters'][
            'Exclude_Deals_With_Missing_Market_Data'] = switch
    context = rf.Context()
    context.load_json((jsonlib.dumps(job, cls=CustomJsonEncoder), 'switched'))
    _, result = context.run_job()
    table = result['Results']['mtm']
    return result['Stats'], dict(zip(table['Reference'], table['Value']))


def test_a_document_saying_no_keeps_every_deal_its_compile_could_not_read_at_zero():
    """An FX option beside one missing its strike and one on a surface the market lacks. Under
    `Yes`, and with the switch left out, the two are dropped and counted and the option marks as it
    marks alone; under `No` the run completes with both kept and marked at zero, the one held
    inside a structure included, the option marking exactly as before.

    The switch not handed down a structure drops the nested one.

    Killing mutation: the switch read by nothing, so the `No` run drops the two and reports no row
    for them.
    """
    alone = _switched(None, [GOOD])[1]['FXO']
    for switch in (None, 'Yes'):
        stats, marks = _switched(switch, [GOOD, NO_STRIKE, NO_SURFACE])
        assert (stats.get('Deals loaded'), stats.get('Deals Skipped')) == (1, 2), stats
        assert marks['FXO'] == alone != 0.0 and 'FXO_NO_STRIKE' not in marks
    held = {'Object': 'StructuredDeal', 'Reference': 'HELD', 'Currency': 'USD',
            'Children': [NO_SURFACE]}
    for deals in ([GOOD, NO_STRIKE, NO_SURFACE], [GOOD, NO_STRIKE, held]):
        stats, marks = _switched('No', deals)
        assert (stats.get('Deals loaded'), stats.get('Deals Skipped')) == (1, 2), stats
        assert marks['FXO'] == alone and marks['FXO_NO_STRIKE'] == marks['FXO_NO_SURFACE'] == 0.0, marks


def test_a_deal_its_pricer_could_not_value_marks_nothing_under_either_switch():
    """A swaption stated by its terms alone compiles and fails in its pricer, its legs being what
    value it. Beside a priced option it is skipped under `Yes` and under `No` alike - the switch is
    the compile's - its row carrying no value, counted under `Deals Skipped` once though a credit
    Monte Carlo prices it batch by batch, settled in cash or physically; the option alone counts
    nothing.

    The physical one's dates reading legs it has none of stops the credit Monte Carlo unnamed.

    Killing mutation: the pricer's skip counted per batch, twice over two.
    """
    terms = {key: value for key, value in next(
        deal for deal in book.BOOK if deal['Reference'] == 'SWPT').items() if key != 'Children'}
    for switch in ('Yes', 'No'):
        stats, marks = _switched(switch, [GOOD, terms])
        assert (stats.get('Deals loaded'), stats.get('Deals Skipped')) == (2, 1), stats
        assert math.isnan(marks['SWPT']) and marks['FXO'] != 0.0, marks
    assert 'Deals Skipped' not in _switched('Yes', [GOOD])[0]
    for style in ('Cash', 'Physical'):
        _, simulated = book.simulated([GOOD, dict(terms, Settlement_Style=style)], ('USD',),
                                      Simulation_Batches=2)
        assert simulated['Stats'].get('Deals Skipped') == 1, (style, simulated['Stats'])


def _structure_guards(container):
    """`container` beside the option, valued and simulated under each switch: the option's mark
    and both runs' `Structs Skipped`, per switch."""
    out = {}
    for switch in ('Yes', 'No'):
        stats, marks = _switched(switch, [GOOD, container])
        _, simulated = book.simulated([GOOD, container], ('USD',), Simulation_Batches=2, system={
            'Exclude_Deals_With_Missing_Market_Data': switch})
        out[switch] = (marks['FXO'], (stats.get('Structs Skipped'),
                                      simulated['Stats'].get('Structs Skipped')))
    return out


#: a netting set whose one deal states a NaN amount, so its value holds NaN
NAN_SET = {'Object': 'NettingCollateralSet', 'Reference': 'NAN_SET', 'Netted': 'True',
           'Collateralized': 'False', 'Children': [dict(GOOD, Reference='FXO_NAN', Underlying_Amount=math.nan)]}


@pytest.mark.parametrize('container', [
    {'Object': 'StructuredDeal', 'Reference': 'HELD_JPY', 'Currency': 'JPY',
     'Children': [dict(GOOD, Reference='FXO_HELD')]},
    dict(next(deal for deal in book.BOOK if deal['Reference'] == 'SWPT'), Children=[
        leg for leg in next(deal for deal in book.BOOK if deal['Reference'] == 'SWPT')['Children']
        if leg['Object'] == 'CFFixedInterestListDeal']),
    NAN_SET,
], ids=['compile', 'post_process', 'nan'])
def test_a_structure_the_run_could_not_value_is_counted_once_under_either_switch(container):
    """The three structure guards: a structure on a currency the market lacks fails its compile, a
    swaption holding its fixed leg alone fails the `post_process` pricing it, and a netting set
    whose value holds NaN is dropped. Under `Yes` and under `No` each is counted once under
    `Structs Skipped`, valued and over two simulated batches, the option marking as it marks alone.

    The `post_process` guard or the NaN drop refusing the run fails here too.

    Killing mutation: the structure's skip counted per batch.
    """
    alone = _switched(None, [GOOD])[1]['FXO']
    for mark, skipped in _structure_guards(container).values():
        assert mark == alone != 0.0
        assert skipped == (1, 1), skipped


def test_a_set_skipped_under_yes_is_left_out_of_what_the_run_reports():
    """A netting set dropped for NaN under `Yes` is skipped and counted, never summed: a base
    valuation's root, which sums its sets, is the priced set's mark beside it, bit for bit, where it
    read NaN; and a credit Monte Carlo in which nothing at all was valued refuses by name, where it
    died framing a profile of nothing.

    The report framing an empty book fails the second half.

    Killing mutation: the root summing a skipped set's value.
    """
    held = {'Object': 'NettingCollateralSet', 'Reference': 'HELD', 'Netted': 'True',
            'Collateralized': 'False', 'Children': [GOOD]}
    stats, marks = _switched('Yes', [held, NAN_SET])
    assert stats.get('Structs Skipped') == 1, stats
    assert marks['root'] == _switched('Yes', [held])[1]['root'] == marks['HELD'], marks
    assert math.isfinite(marks['root']) and marks['root'] != 0.0
    with pytest.raises(utils.UnpriceableSchedule, match=r'Nothing in the book was valued \(1 Structs '
                                                        r'Skipped\), so the run has nothing to report'):
        book.simulated([NAN_SET], ('USD',))
