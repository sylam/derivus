"""A pricing failure inside an inner-MC fork must be LOUD, not a zero.

`Deal.calculate`'s canonical guard turns any exception into `0.0 * shared.one`, which is right for
"this deal cannot price on this grid" and wrong for everything else: no `Calc_res['tensor']` is
written, so the deal vanishes from `DealStructure.tensor_marks()`, so the fork reports
`F_t1 = 0` for it, so the solver's expired-contract mask retires it from the hedge set. The run
finishes and reports a verdict for a hedge book it silently shrank.

One class is therefore distinguished and re-raised (`utils.is_fatal_pricing_error`): running out
of memory — the failure mode the single-pass fork documents as its contract. Everything else keeps
the skip, which is asserted here too, because base valuation and credit Monte Carlo depend on it.

`utils.UnpriceableSchedule` joins that class from the other direction. The first two members say the
FRAMEWORK is wrong; this one says the DOCUMENT is, and it is fatal because a named refusal swallowed
into a zero mark has said nothing at all. It is read by four guards - the two in `Deal`, and both
compile guards in `DealStructure`, where an authored schedule is first touched.

The end-to-end gates rebuild the deal shapes a fork's curve reads used to be predicted wrong for,
each asserting the ANSWER rather than the absence of a crash: the lazily built run must equal the
one whose Hermite coefficients cover the whole block.
"""
import copy
import json as jsonlib
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import pytest
import torch

import derivus as rf
from derivus import instruments, utils
from derivus.calculation import HedgeMonteCarlo

# the one-cashflow job every service gate is built on, reused rather than re-authored: the
# degenerate leg below needs a real USD/ZAR market and a real `BaseValuation`, and that document
# already is one
from test_service import BASE as SERVICE_BASE, dump as service_dump, job as service_job

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'fixtures', 'policy_test_simulate_only.json')
TS = lambda s: {'.Timestamp': s}


# --------------------------------------------------------------------------------------------
# The distinguished classes, at the guard itself
# --------------------------------------------------------------------------------------------
class _Boom(instruments.Deal):
    """A deal whose pricing raises whatever it was constructed with."""

    def __init__(self, exc):
        super().__init__({'Reference': 'BOOM'}, {})
        self.exc = exc

    def generate(self, shared, time_grid, deal_data):
        raise self.exc


class _Shared:
    one = torch.zeros(1, 1)
    keep_tensor = False
    gamma = False


def _calculate(exc):
    deal = _Boom(exc)
    return deal.calculate(_Shared(), None, utils.DealDataType(
        Instrument=deal, Factor_dep={}, Time_dep=None, Calc_res=None))


@pytest.mark.parametrize('exc', [
    torch.cuda.OutOfMemoryError('CUDA out of memory. Tried to allocate 2.67 GiB'),
    RuntimeError('CUDA out of memory. Tried to allocate 2.67 GiB'),
    MemoryError('host allocation failed'),
    utils.ScheduleLifecycleError('TensorCashFlows(4 rows) was never bound to a calculation'),
])
def test_a_framework_fault_is_re_raised(exc):
    """The single-pass fork's documented contract is that a config too wide for the card raises
    CUDA OOM naming the fork. That contract only held for the liability while a tradable's OOM
    became `F_t1 = 0` — a silently smaller hedge set, or a fake one-step move when only the
    heavier grad fork failed. A schedule the calculation never bound says the same thing about the
    framework, and a fork that swallowed it would price the deal out of the book instead."""
    with pytest.raises((torch.cuda.OutOfMemoryError, RuntimeError, MemoryError,
                        utils.ScheduleLifecycleError),
                       match='out of memory|allocation failed|never bound'):
        _calculate(exc)


@pytest.mark.parametrize('exc', [
    ValueError('this deal cannot price on this grid'),
    IndexError('index 7 is out of bounds'),
    KeyError('Discount'),
])
def test_an_ordinary_pricing_failure_still_skips(exc):
    """The canonical skip is load-bearing for base valuation / credit Monte Carlo — a portfolio of
    thousands must not die on one unpriceable deal. Only the two distinguished classes re-raise;
    note a plain IndexError still skips, so the distinction is the type, not the base class."""
    assert torch.equal(_calculate(exc), torch.zeros(1, 1))


def test_build_features_follows_the_same_rule():
    """`build_features` has its own copy of the guard, and a leg swallowed there drops out of the
    liability accumulation, where a second leg's mark broadcasts over the (1,1) gap and the fork's
    shape check cannot see it."""
    deal = _Boom(MemoryError('host allocation failed'))
    data = utils.DealDataType(Instrument=deal, Factor_dep={}, Time_dep=None, Calc_res=None)
    with pytest.raises(MemoryError):
        deal.build_features(_Shared(), None, data)
    deal.exc = ValueError('unpriceable')
    assert torch.equal(deal.build_features(_Shared(), None, data)['mtm'], torch.zeros(1, 1))


# --------------------------------------------------------------------------------------------
# End to end: the deal shapes the derived bound was wrong for
# --------------------------------------------------------------------------------------------
def _energy_leg(name, start, end, pay):
    return {name: {
        'Currency': 'USD', 'Sampling_Type': 'USD', 'FX_Sampling_Type': 'USD',
        'Discount_Rate': 'USD-SOFR', 'Commodity': 'PLATINUM_LME', 'Reference_Type': 'PLATINUM',
        'Payer_Receiver': 'Receiver',
        'Payments': {'Items': [{
            'Payment_Date': TS(pay), 'Period_Start': TS(start), 'Period_End': TS(end),
            'Volume': 2500.0, 'Fixed_Basis': -2045.0, 'Price_Multiplier': 1.0,
            'FX_Period_Start': TS(start), 'FX_Period_End': TS(end),
            'Realized_Average': 0.0, 'FX_Realized_Average': 0.0}]}}}


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


def _run(cfg, name, forks=None):
    """One JSON-only run. `forks` collects every inner-MC fork's outputs when a caller wants to
    inspect what the solver was handed."""
    original_fork = HedgeMonteCarlo._run_inner_mc_at_t
    if forks is not None:
        def record(self, t, *a, **kw):
            r = original_fork(self, t, *a, **kw)
            forks.append({k: float(v.detach().abs().max())
                          for k, v in (r.get('F_t1') or {}).items()})
            return r
        HedgeMonteCarlo._run_inner_mc_at_t = record
    try:
        cx = rf.Context()
        cx.load_json((jsonlib.dumps(cfg), f'{name}.json'))
        _, result = cx.run_job()
        return ((result.evaluation_summary or {}).get('diagnostics') or {}).get('V_0')
    finally:
        HedgeMonteCarlo._run_inner_mc_at_t = original_fork


def test_a_hedge_book_leg_that_fails_to_compile_kills_the_run():
    """The COMPILE-level arm. `add_deal_to_structure`'s skip-and-continue is the reporting book's
    contract, but on a hedge book a skipped tradable shrinks the solver's menu and a skipped
    liability halves the target, so the solve reports a confident answer to a different problem.
    Measured before the guard: an APS leg whose basis law could not state its projection dropped n*
    from -44.8 to -22.1 with nothing but an ERROR log. Both roles raise, naming the leg."""
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
        with pytest.raises(Exception, match=f'{role} legs failed to compile.*BROKEN_LEG'):
            _run(copy.deepcopy(cfg), f'broken_{role}')


def test_an_averaging_tradable_is_priced_not_retired():
    """An averaging swap hedging an averaging offtake — the obvious hedge for this book — reads
    further back than the liability does. The window was derived from the liability alone, so the
    swap's gather fell below it, its mark was swallowed, and it read downstream as an expired
    contract: the policy trained against a hedge set silently reduced from two instruments to one,
    with a full verdict returned."""
    cfg = _cfg(t_min=113)
    cfg['Calc']['Calculation']['Hedging_Problem']['Tradable_Instruments']['FloatingEnergyDeal'] = \
        _energy_leg('PL_AVG_SWAP', '2026-04-15', '2026-07-31', '2026-08-05')
    forks = []
    built = _run(copy.deepcopy(cfg), 'avg_tradable', forks=forks)
    assert forks, 'no inner-MC forks ran'
    assert all(f['PL_AVG_SWAP'] > 0.0 for f in forks), \
        'the averaging tradable is zero in some fork — it was retired from the hedge set'
    assert built == _run(cfg, 'avg_tradable_again'), \
        'the lazily built answer differs from the full-block one'


def test_a_settlement_lagged_liability_solves():
    """Deferred settlement (pay well after the averaging period ends) keeps the cashflow unpaid, so
    `sim_resets` keeps re-reading its fixings. A bound measured as the within-period reset span is
    short by exactly that lag, and the liability could not be solved at all."""
    cfg = _cfg(t_min=150)
    hp = cfg['Calc']['Calculation']['Hedging_Problem']
    hp['Liabilities']['FloatingEnergyDeal']['PLAT_JUL29']['Payments']['Items'][0][
        'Payment_Date'] = TS('2026-10-30')
    hp['Tradable_Instruments']['CashAccountDeal']['USD_CASH'][
        'Investment_Horizon'] = TS('2026-10-30')
    assert _run(copy.deepcopy(cfg), 'lagged') == _run(cfg, 'lagged_again')


def test_a_bullet_sampled_leg_solves():
    """`ForwardPriceSampleBullet` writes ONE fixing per period, at the period END. The `count > 1`
    filter made such a leg declare that it reads no history while its fixing still sat a
    settlement lag below the cutoff."""
    cfg = _cfg(t_min=113)
    hp = cfg['Calc']['Calculation']['Hedging_Problem']
    hp['Liabilities']['FloatingEnergyDeal']['PLAT_JUL29']['Payments']['Items'][0][
        'Payment_Date'] = TS('2026-08-20')
    hp['Tradable_Instruments']['CashAccountDeal']['USD_CASH'][
        'Investment_Horizon'] = TS('2026-08-20')
    cfg['Calc']['MergeMarketData']['ExplicitMarketData']['Price Factors'][
        'ForwardPriceSample.USD']['Sampling_Convention'] = 'ForwardPriceSampleBullet'
    assert _run(copy.deepcopy(cfg), 'bullet') == _run(cfg, 'bullet_again')


def test_a_two_leg_liability_book_solves():
    """Two averaging legs reading DIFFERENT depths of history. A single bound over the book had to
    be the min of both, and the second leg's own reads had to already be inside it; the coefficients
    now follow whichever leg gathers first and widen for the other."""
    cfg = _cfg(t_min=113)
    hp = cfg['Calc']['Calculation']['Hedging_Problem']
    hp['Liabilities']['FloatingEnergyDeal'].update(
        _energy_leg('PLAT_AUG29', '2026-05-01', '2026-08-31', '2026-09-04'))
    hp['Tradable_Instruments']['CashAccountDeal']['USD_CASH'][
        'Investment_Horizon'] = TS('2026-09-04')
    assert _run(copy.deepcopy(cfg), 'two_leg') == _run(cfg, 'two_leg_again')


def test_a_failed_tradable_inside_a_fork_stops_the_run():
    """The OOM the chunk-loop deletion designated as the expected failure mode, injected into a
    hedge instrument. It must reach the caller instead of becoming `F_t1 = 0` — and because the
    `live` mask comes from the cheaper no-grad fork, a grad-only failure would leave the leg live
    while its one-step move read as -F_t of fake P&L in the training labels."""
    original = instruments.CommodityFutureDeal.generate
    original_fork = HedgeMonteCarlo._run_inner_mc_at_t
    in_fork = {'v': False}

    def fork(self, t, *a, **kw):
        in_fork['v'] = True
        try:
            return original_fork(self, t, *a, **kw)
        finally:
            in_fork['v'] = False

    def generate(self, shared, time_grid, deal_data):
        if in_fork['v'] and self.field.get('Reference') == 'PL_OCT_2026':
            raise torch.cuda.OutOfMemoryError('CUDA out of memory. Tried to allocate 2.67 GiB')
        return original(self, shared, time_grid, deal_data)

    instruments.CommodityFutureDeal.generate = generate
    HedgeMonteCarlo._run_inner_mc_at_t = fork
    try:
        with pytest.raises(torch.cuda.OutOfMemoryError, match='out of memory'):
            _run(_cfg(t_min=115), 'oom_tradable')
    finally:
        instruments.CommodityFutureDeal.generate = original
        HedgeMonteCarlo._run_inner_mc_at_t = original_fork


def test_a_skipped_tradable_leaves_a_loud_hole_in_the_fork():
    """The tradable half of the fork's degenerate-pricing guard. A non-distinguished failure still
    skips (correctly), but inside a fork the missing mark is indistinguishable from an expired
    contract — so the fork checks that every tradable still live in its dependency list produced
    one, mirroring the liability's shape check."""
    original = instruments.CommodityFutureDeal.generate
    original_fork = HedgeMonteCarlo._run_inner_mc_at_t
    in_fork = {'v': False}

    def fork(self, t, *a, **kw):
        in_fork['v'] = True
        try:
            return original_fork(self, t, *a, **kw)
        finally:
            in_fork['v'] = False

    def generate(self, shared, time_grid, deal_data):
        if in_fork['v'] and self.field.get('Reference') == 'PL_OCT_2026':
            raise ValueError('some ordinary pricing failure')
        return original(self, shared, time_grid, deal_data)

    instruments.CommodityFutureDeal.generate = generate
    HedgeMonteCarlo._run_inner_mc_at_t = fork
    try:
        with pytest.raises(RuntimeError, match="tradable pricing failed for \\['PL_OCT_2026'\\]"):
            _run(_cfg(t_min=115), 'skipped_tradable')
    finally:
        instruments.CommodityFutureDeal.generate = original
        HedgeMonteCarlo._run_inner_mc_at_t = original_fork


# --------------------------------------------------------------------------------------------
# The fork publishes a SEQUENCE of row blocks, not a joined grid
# --------------------------------------------------------------------------------------------
_SOURCE, _BUILD = utils.ScenarioSource, utils.build_interpolation


def _at_the_read_boundary(hook):
    """Run with `hook(source)` applied to every block sequence the pricer is handed. The factory
    is the boundary now: a leaf never sees a source, which is the separation under test."""
    def wrapped(value, *args, **kwargs):
        return _BUILD(hook(value) if isinstance(value, _SOURCE) else value, *args, **kwargs)
    return wrapped


def _run_with(hook, cfg, name):
    utils.build_interpolation = _at_the_read_boundary(hook)
    try:
        return _run(cfg, name)
    finally:
        utils.build_interpolation = _BUILD


def _join(source):
    """The pre-feature grid: ONE tensor carrying every row at the flat width, which is what the
    fork used to stuff with a `cat` of a past expanded across the inner draws. Built HERE rather
    than asked of the source, which no longer offers it — the join is the thing being replaced."""
    width = source.shape[-1]
    return torch.cat(
        [b.tensor if b.batch_index is None else b.tensor.index_select(-1, b.batch_index)
         for b in source.blocks], dim=0) if width else None


def test_a_fork_reading_deep_history_answers_as_if_the_grid_were_joined():
    """The case the whole change turns on. A one-step fork windows the DEAL to {t, t+1}, but the
    liability's past fixings build their own grid off the reset schedule and gather densely over
    0..t — so the realized past has to be there, it just does not have to be there B_inner times.
    Solving the deepest configuration both ways must give the same value to the last bit."""
    cfg = _cfg(t_min=113)
    hp = cfg['Calc']['Calculation']['Hedging_Problem']
    hp['Liabilities']['FloatingEnergyDeal'].update(
        _energy_leg('PLAT_AUG29', '2026-05-01', '2026-08-31', '2026-09-04'))
    hp['Tradable_Instruments']['CashAccountDeal']['USD_CASH'][
        'Investment_Horizon'] = TS('2026-09-04')
    blocked = _run(copy.deepcopy(cfg), 'deep_history_blocks')
    assert blocked is not None
    assert blocked == _run_with(_join, cfg, 'deep_history_joined')


def test_the_published_source_is_write_once():
    """A fork builds its block sequence AFTER every process's `generate` has published, and
    nothing writes into `t_Scenario_Buffer` afterwards. The published value carries only what
    `make_curve_tensor` does to a buffer value, so a late write fails loud instead of silently
    materializing the grid the sequence exists to avoid."""
    seen = []
    _run_with(lambda source: seen.append(source) or source, _cfg(t_min=113), 'write_once')
    assert seen, 'no fork published a block sequence'
    deep = [s for s in seen if len(s.blocks) > 1]
    assert deep, 'every fork published one block — no fork reached past its cutoff'
    source = deep[0]
    past, inner = source.blocks
    assert source.shape == (past.n_rows + inner.n_rows,) + tuple(inner.tensor.shape[1:])
    assert past.first_row == 0 and inner.first_row == past.n_rows
    # the past's batch map is the fork's own flatten, so it must land every logical column on the
    # outer path that produced it
    fan = inner.tensor.shape[-1] // past.tensor.shape[-1]
    assert inner.batch_index is None, 'the forked block is already at the logical width'
    assert torch.equal(past.batch_index,
                       torch.arange(inner.tensor.shape[-1],
                                    device=past.batch_index.device) // fan)
    for write in (lambda: source.__iadd__(0.0001), lambda: source.copy_(inner),
                  lambda: source.__setitem__(0, 0.0)):
        with pytest.raises((TypeError, AttributeError)):
            write()


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
    """`make_float_cashflows` read `cashflow['Rate_Tenor']` whenever a reset's rate window had zero
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


def test_the_named_refusal_is_fatal_at_the_compile_guard_too():
    """`is_fatal_pricing_error` is read by FOUR guards, and the two compile ones are why the gate
    above sees a raise at all: an authored schedule is touched in `calc_dependencies`, which
    `add_deal_to_structure` wraps in the skip-and-continue.

    So the predicate is asserted at both ends: it admits the new class and still admits the two it
    always did, and an ordinary pricing failure is still swallowed.
    """
    assert utils.is_fatal_pricing_error(utils.UnpriceableSchedule('a zero-length rate window'))
    assert utils.is_fatal_pricing_error(utils.ScheduleLifecycleError('never bound'))
    assert utils.is_fatal_pricing_error(MemoryError('host allocation failed'))
    assert not utils.is_fatal_pricing_error(KeyError('Rate_Tenor')), (
        'the bare KeyError this row replaced must still take the canonical skip - it is what an '
        'unrelated missing field looks like')

    # and both compile guards read it, off the source rather than off a second fixture: a guard
    # that stopped consulting the predicate would swallow the refusal again with nothing red
    import inspect

    from derivus.calculation import DealStructure
    for guard in (DealStructure.add_deal_to_structure, DealStructure.add_structure_to_structure):
        body = inspect.getsource(guard)
        assert 'is_fatal_pricing_error' in body and 'raise' in body, guard.__name__
