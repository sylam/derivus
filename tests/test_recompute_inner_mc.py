"""`Recompute_Inner_MC`: a TARF's inner Monte Carlo re-simulated in backward() instead of taped.

An MC-priced deal builds one autograd graph per pricing and the terminal backward holds every one at
once, while the simulation that built them is cheap. `pricing.InnerMCRecompute` trades the tape for
a second forward: the node's forward runs under `no_grad`, its backward re-runs the SAME callable
under `enable_grad` and contracts the cotangent through one graph that dies immediately.

NOTHING DRAWN IS STORED. The pricer's inner rows are the canonical inner block's, a function of
their shape and `Quasi_Seed` at every outer batch size, so the replay reads them again with nothing
to rewind: base valuation's one scenario and the 512-scenario exposure read the same block. A base
valuation draws nothing else, so a trade's mark is its own whatever is booked beside it, and a
second `Quasi_Seed` moves it by its Monte Carlo error.

THE GRADIENT GATE IS `array_equal` AND NOT A TOLERANCE. The replay is the same kernels on the same
inputs in the same order, and a tolerance would let a desynchronised stream through as "close
enough". Measured bit-identical on both paths, base valuation (7 factors) and CVA (14), while the
smallest stream mutation moves the gradient by 1.2e-04 relative. Also GRID-INVARIANT: re-taken on
`0d 1m(1m)`, `0d 2m(2m)` and `0d 3m(3m)`, with `Dynamic_Scenario_Dates` off and on.

WHERE THE REPLAY RUNS IS WHERE ITS DEFECTS ARE VISIBLE. `backward()` runs once per pricing block and
only when a gradient is asked for - 1 forward under base valuation, 6 under exposure - so the
by-product law (a settled cashflow is an output the caller performs once, never a side effect of
`simulate`) is gated at `Gradient: 'Yes'`, where a replay that books settles 36 extra cashflows and
moves the reported frame by 6.0e+06 while cva, profile AND the whole CVA gradient stay
bit-identical.

WHAT THE NODE CANNOT DO is gated beside the equity adopters: a second derivative through it is
severed, so `backward` refuses `create_graph` naming the switch. On this fixture base valuation
refuses first, the TARF registering a boundary correction under the crisp estimator.
"""
import gc
import logging
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import pytest
import torch

import derivus
from derivus import calculation, run_baseval, utils
from derivus.instruments import construct_instrument
import test_barrier_bridge as bb
import test_boundary_pricer_events as bp
import test_boundary_tarf_events as tarf

DTYPE = torch.float64
# the fixture's own world, unchanged - `test_boundary_tarf_events` owns the deals and the market,
# and both decisions inside the pricer are already reachable there
KNOCK_IN, KNOCK_IN_CMC, PIN_CMC = tarf.KNOCK_IN, tarf.KNOCK_IN_CMC, tarf.PIN_CMC


def baseval(deal, greeks=False, sims=1 << 12, recompute='No'):
    """(price, the WHOLE first-order gradient vector). One scenario, so the pricer's inner Monte
    Carlo, the canonical block's rows, is the entire simulation."""
    overrides = {'MCMC_Simulations': sims, 'Quasi_Seed': 1, 'Recompute_Inner_MC': recompute,
                 'Greeks': 'First' if greeks else 'No'}
    _, out = run_baseval(tarf._cfg(deal, tarf.SPOT), overrides=overrides)
    rows = out['Results']['mtm']
    price = float(rows[rows['Reference'] == 'TARF1']['Value'].iloc[0])
    if not greeks:
        return price, None
    frame = out['Results']['Greeks_First']
    # 'Value' is the factor LEVEL (display_val=True); the other column is the gradient
    column, = [x for x in frame.columns if x != 'Value']
    return price, frame[column].values.astype(np.float64)


def cmc(deal, gradient=False, recompute='No', batches=1, batch=512, mcmc=128,
        deterministic='No'):
    """(cva, mtm profile, the WHOLE CVA gradient vector, cashflows). 512 scenarios, so the
    boundary correction has a population to fit a kernel to."""
    overrides = {
        'Run_Date': tarf.BASE.strftime('%Y-%m-%d'), 'Time_grid': '0d 2m(2m)', 'Batch_Size': batch,
        'Simulation_Batches': batches, 'Random_Seed': 1, 'Currency': 'USD', 'Tenor_Offset': 0.0,
        'MCMC_Simulations': mcmc, 'Deflation_Interest_Rate': 'USD', 'Generate_Cashflows': 'Yes',
        'Gradient_Variables': 'Factors', 'Recompute_Inner_MC': recompute,
        'Deterministic_Kernels': deterministic,
        'Credit_Valuation_Adjustment': {
            'Calculate': 'Yes', 'Counterparty': 'CPTY', 'Deflate_Stochastically': 'No',
            'Stochastic_Hazard_Rates': 'No', 'Gradient': 'Yes' if gradient else 'No'}}
    _, out = derivus.run_cmc(
        tarf._cfg(deal, tarf.SPOT, counterparty=True, simulate_fx=True),
        prec=DTYPE, overrides=overrides)
    grad = out['Results']['grad_cva']['Gradient'].values.astype(np.float64) if gradient else None
    # .values per currency FRAME - iterating a DataFrame yields its column labels, and a gate
    # built on those cannot fail (verified against a zeroed-cashflow run)
    cashflows = [frames.values for frames in out['Results'].get('cashflows', {}).values()]
    return float(out['Results']['cva']), out['Results']['mtm'].values, grad, cashflows


# ------------------------------------------------- the sibling switch, same calculation block

@pytest.mark.skipif(not torch.cuda.is_available(), reason='the atomics are the GPU backward')
def test_deterministic_kernels_is_read_every_run_and_set_both_ways():
    """`Deterministic_Kernels` is PROCESS-GLOBAL state the calculation sets from the document: a
    `Yes` run leaves the flag on, and a `No` run LEAVES IT OFF, or the next job in the process
    inherits a pin it never asked for. `warn_only` is the field's own spelling: an operation torch
    cannot pin warns and runs unpinned rather than refusing the valuation.

    Killing mutation: the switch set one way only, a `No` run leaving the process as it found it.
    """
    cmc(KNOCK_IN_CMC, deterministic='Yes')
    assert torch.are_deterministic_algorithms_enabled(), (
        'the calculation never read Deterministic_Kernels: Yes out of the document')
    cmc(KNOCK_IN_CMC, deterministic='No')
    assert not torch.are_deterministic_algorithms_enabled(), (
        'a No run left the process pinned - the switch is being set one way only')


# ---------------------------------------------------------------- the value and gradient must not move

def test_the_base_price_and_gradient_are_bit_identical_with_the_node_on():
    """The price and the WHOLE vector, boundary correction included (`Greeks: First` is what turns
    `boundary_aad` on), on a fixture small enough that the full tape fits - which is the only place
    the two paths can be compared at all. BIT-identical, not approximately: a recompute that drew
    different numbers would still converge to the same price at 4096 paths and be wrong in every
    digit that matters.

    Killing mutation: the pseudo-random draw restored below 16 outer paths with no position to
    replay it from - the replay draws the generator's next numbers and the spot gradient reads
    3,802,184.70 against the taped 3,796,451.71.
    """
    price_off, grad_off = baseval(KNOCK_IN, greeks=True)
    price_on, grad_on = baseval(KNOCK_IN, greeks=True, recompute='Yes')
    assert grad_off is not None and np.abs(grad_off).max() > 0.0, 'no gradient was reported'
    assert price_off == price_on, (price_off, price_on)
    assert np.array_equal(grad_off, grad_on), (
        'the recomputed gradient is not the taped one:\n{}\n{}'.format(grad_off, grad_on))


@pytest.mark.parametrize('deal,label', [(KNOCK_IN_CMC, 'knock-in'), (PIN_CMC, 'pin')])
def test_the_exposure_its_cashflows_and_the_cva_gradient_are_bit_identical_with_the_node_on(
        deal, label):
    """The whole profile, the settled cashflows and the whole CVA gradient, with sensitivities ON:
    the simulation is called TWICE under the node, so a cashflow accrued inside it rather than
    returned would settle twice, and only with a gradient asked for does the replay run.
    `CreditMonteCarlo` harvests `t_Cashflows` AFTER the batch's backward, so a booking made in the
    replay is still in the frame when it is read. Both boundary registrations are live - the
    latched redemption, whose gaps are built OUTSIDE the node, and the knock-in.

    Killing mutation: the replay settling the block's cashflows a second time.
    """
    cva_off, mtm_off, grad_off, cash_off = cmc(deal, gradient=True)
    cva_on, mtm_on, grad_on, cash_on = cmc(deal, gradient=True, recompute='Yes')
    assert np.array_equal(mtm_off, mtm_on), f'{label}: exposure moved with the node on'
    assert cva_off == cva_on, f'{label}: cva moved: {cva_off!r} -> {cva_on!r}'
    assert cash_off and all(np.array_equal(a, b) for a, b in zip(cash_off, cash_on)), (
        f'{label}: a settled cashflow moved - the simulation ran its side effects twice')
    assert np.abs(grad_off).max() > 0.0, f'{label}: no gradient was reported'
    assert np.array_equal(grad_off, grad_on), (
        '{}: the recomputed CVA gradient is not the taped one:\n{}\n{}'.format(
            label, grad_off, grad_on))


# ---------------------------------------------------------------- (f) the strip is streamed

#: The replay's peak bound for `streamed` under `STREAM_CAP_MIB`, at 16 x 4096, float32.
STREAM_PEAK_MIB = 265.0
#: A per-process cap above the floor that the TARF's strips walked plain do not fit.
STREAM_CAP_MIB = 300.0


def replayed(config, grid, batch, inner, cap=None, recompute='Yes'):
    """(the run's results, its peak in MiB above its floor) of `config` reported on `grid`, its CVA
    gradient replayed by `Recompute_Inner_MC` (taped with `recompute` 'No'), float32 on the device,
    under a per-process memory fraction leaving `cap` MiB above the floor where one is given."""
    base = config.params['System Parameters']['Base_Date']
    params = {
        'Base_Date': base, 'Run_Date': base.strftime('%Y-%m-%d'), 'Currency': 'USD',
        'Time_grid': grid, 'Batch_Size': batch, 'Simulation_Batches': 1, 'Random_Seed': 1,
        'Tenor_Offset': 0.0, 'MCMC_Simulations': inner, 'Deflation_Interest_Rate': 'USD',
        'Gradient_Variables': 'Factors', 'Recompute_Inner_MC': recompute,
        'Credit_Valuation_Adjustment': {
            'Calculate': 'Yes', 'Counterparty': 'CPTY', 'Deflate_Stochastically': 'No',
            'Stochastic_Hazard_Rates': 'No', 'Gradient': 'Yes'}}
    # the cycle a boundary registration makes outlives refcounting, so collect before the floor
    gc.collect()
    torch.cuda.empty_cache()
    floor = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    if cap:
        torch.cuda.set_per_process_memory_fraction(
            (torch.cuda.memory_reserved() + cap * 2.0 ** 20) / torch.cuda.mem_get_info()[1])
    try:
        out = calculation.CreditMonteCarlo(
            config, device=utils.calculation_device(None, 0), prec=torch.float32).execute(params)
    finally:
        torch.cuda.set_per_process_memory_fraction(1.0)
    return out['Results'], (torch.cuda.max_memory_allocated() - floor) / 2.0 ** 20


def logged(messages, pattern):
    """The integers of every logged line `pattern` matches, a tuple a line."""
    return [tuple(int(x) for x in m.groups())
            for m in map(re.compile(pattern).match, messages) if m]


def streamed(cap=None, batch=16, inner=4096):
    """(cva, CVA gradient, the run's peak in MiB above its floor) of a 24-fixing knock-in TARF
    reported monthly, under `cap` where one is given."""
    deal = tarf._tarf(tarf.UNREACHABLE_TARGET, [30 * (i + 1) for i in range(24)],
                      barrier=tarf.BARRIER, buy_sell='Sell')
    results, peak = replayed(tarf._cfg(deal, tarf.SPOT, counterparty=True, simulate_fx=True),
                             '0d 1m(1m)', batch, inner, cap)
    return float(results['cva']), results['grad_cva']['Gradient'].values.astype(np.float64), peak


@pytest.mark.skipif(not torch.cuda.is_available(), reason='reads the device allocator')
def test_the_streamed_strip_holds_its_peak_and_the_unstreamed_gradient(caplog):
    """Uncapped, every strip of the replay walks plain; under a per-process memory fraction leaving
    `STREAM_CAP_MIB` above the floor, which those strips do not fit, a strip is checkpointed from
    the stretch its projected tape passes half of what is free, and the run peaks under
    `STREAM_PEAK_MIB` and reproduces the uncapped run's cva and gradient bit for bit.

    Readings: uncapped 922 MiB as a process's first run (cuBLAS's 32 MiB workspace among it), all
    48 strips plain and each two-row block one piece, a row-step's replay measured at 19.4 MB, 466
    over the probed row's 24 fixings; capped 224 MiB, 33 strips checkpointed and every block cut to
    one row, the row-step 18.3 MB, its tape's, the probe's own replay having checkpointed.

    Killing mutations: the projection ignored, every strip walked plain - out of memory under the
    cap; every strip checkpointed past its first stretch - 44 of the uncapped run's 48 strips.
    """
    walks = r'OSS_STREAM \S+ tape -?\d+ bytes, budget \S+: (\d+) of (\d+) fixings plain'
    with caplog.at_level(logging.DEBUG):
        whole_cva, whole_gradient, whole_peak = streamed()
        whole = logged(caplog.messages, walks)
        caplog.clear()
        cva, gradient, peak = streamed(STREAM_CAP_MIB)
        capped = logged(caplog.messages, walks)
    assert whole and all(w == n for w, n in whole) and any(w < n for w, n in capped), (
        'uncapped {} of {} strips plain, capped {} of {}'.format(
            sum(w == n for w, n in whole), len(whole), sum(w == n for w, n in capped), len(capped)))
    assert cva == whole_cva and np.array_equal(gradient, whole_gradient), (
        'streaming moved the CVA or its gradient:\n{}\n{}'.format(gradient, whole_gradient))
    assert peak < STREAM_PEAK_MIB < whole_peak, (
        'peak {:.1f} MiB under the cap, {:.1f} without, bound {}'.format(
            peak, whole_peak, STREAM_PEAK_MIB))


#: A per-process cap above the floor that `dense`'s daily block replayed whole does not fit.
DENSE_CAP_MIB = 300.0


def dense(grid, cap=None, recompute='Yes'):
    """(the marks on each reported date, the run's peak in MiB above its floor) of a twelve-coupon
    monthly autocall with a put at 70, reported on `grid` under `cap` where one is given, its CVA
    gradient replayed or, with `recompute` 'No', taped. Its
    equity's GBM has no vol, so the spot is the base spot on every date and a row's mark is a
    function of its date alone; the sloped curves and smile give each row its own discounts, carry
    and vols; 64 scenarios take the canonical inner block."""
    monthly = [bb.BASE + pd.Timedelta(days=30 * (i + 1)) for i in range(12)]
    deal = dict(bp.AUTOCALL, Expiry_Date=monthly[-1], Price_Fixing=[[d, 0.0] for d in monthly],
                Autocall_Coupons=[[d, 0.05] for d in monthly],
                Autocall_Thresholds=[[d, 1.02] for d in monthly], Barrier=70.0,
                Barrier_Dates=monthly[-1:])
    config = bb._cfg()
    config.params['Price Models']['GBMAssetPriceModel.EQ']['Vol'] = 0.0
    factors = config.params['Price Factors']
    factors['InterestRate.USD']['Curve'] = utils.Curve([], [[0.0, 0.02], [2.0, 0.05]])
    factors['DividendRate.EQ']['Curve'] = utils.Curve([], [[0.0, 0.01], [2.0, 0.03]])
    factors['EquityPriceVol.EQ']['Surface'] = utils.Curve([], [
        [m, t, 0.25 - 0.2 * (m - 1.0) + 0.02 * t] for m in (0.8, 1.0, 1.2) for t in (0.02, 2.0)])
    factors['SurvivalProb.CPTY'] = {
        'Recovery_Rate': 0.4, 'Curve': utils.Curve([], [[0.0, 0.0], [10.0, 0.4]])}
    config.deals['Deals']['Children'] = [{'Instrument': construct_instrument(deal, {})}]
    results, peak = replayed(config, grid, 64, 4096, cap, recompute)
    return results['mtm'], peak


@pytest.mark.skipif(not torch.cuda.is_available(), reason='reads the device allocator')
def test_a_daily_grid_marks_the_weekly_dates_within_its_cap(caplog):
    """A pricer prices the rows sharing a fixing interval as one block: a weekly grid puts four or
    five of `dense`'s rows in an interval, a daily one thirty. Under a per-process memory fraction
    leaving `DENSE_CAP_MIB` above the floor, which the daily block replayed whole does not fit, the
    pricer measures one row's replay and prices each block in the pieces half of what is free
    replays. A row's strip reads only its own row and draws what it would in the whole block, so
    the daily grid marks every weekly date to the bit, peaking under the cap.

    Readings: weekly uncapped 1,855 MiB, 1,887 as a process's first run, every block whole and
    every strip plain; daily under the cap 251 MiB, a row-step measured at 24.1 MB, 12 to a row,
    and every block of 30 or 31 rows cut to one, 182 of its 362 strips checkpointed. Uncapped, the
    daily grid replays its blocks whole and its strips plain at 9,544 MiB.

    Killing mutations: the measured row ignored, every block replayed whole - out of memory under
    the cap; the vol strip handed to every piece whole - a weekly date moves by 7.3e-05 on marks
    spread 5.1e-03.
    """
    weekly, _ = dense('0d 1w(1w)')
    with caplog.at_level(logging.DEBUG):
        daily, peak = dense('0d 1d(1d)', DENSE_CAP_MIB)
    pieces = logged(caplog.messages, r'OSS_ROWS \S+ (\d+) bytes a row-step x \d+ fixings, '
                                     r'budget \S+: (\d+) rows a piece of (\d+)')
    shared = daily.loc[weekly.index].values
    assert weekly.values.std() > 0.0 and np.array_equal(weekly.values, shared), (
        'the daily grid moved a weekly date by {:.6g}'.format(
            float(np.abs(weekly.values - shared).max())))
    assert pieces and all(row > 0 for row, _, _ in pieces) and any(
        k < n for _, k, n in pieces), 'no block was cut to its measured row: {}'.format(pieces)
    assert peak < DENSE_CAP_MIB, 'peak {:.1f} MiB daily, cap {}'.format(peak, DENSE_CAP_MIB)


#: The bound on `dense`'s weekly run with its CVA gradient taped, at 64 x 4096, float32.
TAPED_PEAK_MIB = 550.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='reads the device allocator')
def test_a_taped_strip_that_fits_is_one_checkpoint(caplog):
    """A taped run holds its checkpoints' edges to its batch's end, so a strip whose projected
    tape, its fixings times the row-step the probe measured, fits half of what is free is one
    checkpoint: `dense`'s weekly run with its CVA gradient taped peaks under `TAPED_PEAK_MIB`, every
    strip but the probe's walked as one checkpoint.

    Readings: 458 MiB, 490 as a process's first run, the row-step 27.6 MB taped; the chunk of four
    fixings it replaces 610.

    Killing mutation: the strip in `ceil(sqrt(n))` stretches whatever it projects - 718 MiB, 750 as
    a first run, a strip of twelve fixings in stretches of four.
    """
    with caplog.at_level(logging.DEBUG):
        _, peak = dense('0d 1w(1w)', recompute='No')
    walks = logged(caplog.messages, r'OSS_STREAM \S+ tape -?\d+ bytes, budget \S+: (\d+) of '
                                    r'(\d+) fixings plain, (\d+) a checkpoint')
    held = [(n, c) for w, n, c in walks if w < n]
    assert held and all(c == n for n, c in held) and peak < TAPED_PEAK_MIB, (
        'peak {:.1f} MiB, bound {}; {} of {} strips one checkpoint'.format(
            peak, TAPED_PEAK_MIB, sum(c == n for n, c in held), len(held)))


# ---------------------------------------------------------------- (g) a trade's draws are its own

def accumulator():
    """An accumulator fixing on the TARF's first three bimonthly dates, which asks the inner draws
    for other strip lengths than the TARF's."""
    dates = [tarf.BASE + pd.Timedelta(days=d) for d in tarf.BIMONTHLY[:3]]
    return {'Instrument': construct_instrument({
        'Object': 'FXAccumulatorOptionDeal', 'Reference': 'ACC1', 'Currency': 'USD',
        'Underlying_Currency': 'AUD', 'Discount_Rate': 'USD', 'FX_Volatility': 'AUD.USD',
        'Buy_Sell': 'Sell', 'Option_Type': 'Call', 'Strike_Price': tarf.STRIKE,
        'Underlying_Amount': tarf.N1, 'LeverageNotional': tarf.N2,
        'Barrier_Type': 'Up_And_Out', 'Barrier_Price': 0.75,
        'Accumulator_ExpiryDates': [[d, d, None] for d in dates]}, {})}


def knock_in_tensor(neighbour):
    """The knock-in TARF's own exposure tensor at 64 x 128, booked alone or after the
    accumulator."""
    config = tarf._cfg(KNOCK_IN_CMC, tarf.SPOT, counterparty=True, simulate_fx=True)
    if neighbour:
        config.deals['Deals']['Children'].insert(0, accumulator())
    calc, _ = derivus.run_cmc(config, prec=DTYPE, overrides={
        'Run_Date': tarf.BASE.strftime('%Y-%m-%d'), 'Time_grid': '0d 2m(2m)', 'Batch_Size': 64,
        'Simulation_Batches': 1, 'Random_Seed': 1, 'Currency': 'USD', 'MCMC_Simulations': 128,
        'Deflation_Interest_Rate': 'USD', 'DealLevel': True, 'Keep_Tensor': 'Yes'})
    deal, = [x for x in calc.netting_sets.dependencies
             if x.Instrument.field['Reference'] == 'TARF1']
    return deal.Calc_res['tensor'].detach().cpu().numpy()


def test_a_trade_is_valued_on_its_own_draws_whatever_is_booked_before_it():
    """The TARF's own exposure is bit for bit the same booked alone and after an accumulator: a
    trade's inner draws are the canonical block's first rows, whatever else the job asked for.

    Killing mutation: the block keyed on the request, each new strip length drawn at the stream's
    running position as the memoized quasi stream drew it - 7.3e+03 on a largest entry of 9.1e+05.
    """
    alone, beside = knock_in_tensor(False), knock_in_tensor(True)
    assert alone.std() > 0.0 and np.array_equal(alone, beside), (
        'a neighbour moved the TARF by {:.6g}'.format(float(np.abs(alone - beside).max())))


def booked(order, **calculation):
    """The knock-in TARF's base valuation mark in a book of `order` - `T` the TARF, `A` the
    accumulator - at `calculation`'s dials."""
    config = tarf._cfg(KNOCK_IN, tarf.SPOT)
    held = {'T': config.deals['Deals']['Children'][0], 'A': accumulator()}
    config.deals['Deals']['Children'] = [held[name] for name in order]
    rows = run_baseval(config, overrides=calculation)[1]['Results']['mtm']
    return float(rows[rows['Reference'] == 'TARF1']['Value'].iloc[0])


def test_a_base_valuation_values_a_trade_on_its_own_draws_in_any_book():
    """A base valuation draws a Monte Carlo trade's inner paths off `Quasi_Seed` alone, so the
    TARF marks the same alone, after the accumulator and before it, to the bit, at -8,262.65.

    Killing mutation: the pseudo-random draw restored below 16 outer paths, seeded once a valuation
    - alone the TARF reads -8,378.22, after the accumulator -7,851.62.
    """
    alone = booked('T', MCMC_Simulations=1 << 12)
    assert alone == booked('AT', MCMC_Simulations=1 << 12) == booked('TA', MCMC_Simulations=1 << 12)


#: The knock-in TARF's inner Monte Carlo error at the declared 16,384 paths: the standard
#: deviation of its base valuation mark over `Quasi_Seed` 1 to 32.
TARF_INNER_ERROR = 3.68


def test_a_second_quasi_seed_moves_the_mark_by_its_monte_carlo_error():
    """`Quasi_Seed` scrambles the inner block, so a second seed is an independent randomisation of
    the same points: at the declared path count the TARF reads -8,268.63 at seed 1 and -8,274.89
    at seed 2, 6.27 apart, where its marks over seeds 1 to 32 average -8,271.45 and spread
    `TARF_INNER_ERROR`. The difference is asserted non-zero and inside four standard deviations of
    the difference of two seeds.

    Killing mutations: the scramble seeded by the constant whatever the calculation states - the
    two marks equal at -8,268.29; the pseudo-random draw restored below 16 outer paths - the two
    seeds read -8,227.49 and -8,334.62, 107.12 apart.
    """
    one, two = booked('T', Quasi_Seed=1), booked('T', Quasi_Seed=2)
    assert 0.0 < abs(one - two) < 4.0 * 2.0 ** 0.5 * TARF_INNER_ERROR, (one, two)
