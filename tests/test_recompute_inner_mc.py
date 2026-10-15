"""`Recompute_Inner_MC`: a TARF's inner Monte Carlo re-simulated in backward() instead of taped.

An MC-priced deal builds one autograd graph per pricing and the terminal backward holds every one at
once, while the simulation that built them is cheap. `pricing.InnerMCRecompute` trades the tape for
a second forward: the node's forward runs under `no_grad`, its backward re-runs the SAME callable
under `enable_grad` and contracts the cotangent through one graph that dies immediately.

THE POSITION IS THE STORAGE. What is saved is where the regular generator stood
(`rng_position`), never what it drew, and its state is restored for the replay; the Sobol rows
are the canonical inner block's, a function of their shape, with no position to save. `pv_MC_Tarf`
takes Sobol above 16 scenarios, so base valuation's one scenario reads `torch.rand` - the generator
one draw ahead moves its gradient by 5.73e+03 - and the 512-scenario exposure reads the block, its
replay bit-identical with the generator and the quasi counter both a draw ahead.

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
import os
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
    Carlo is the entire simulation and `quasi_rng` is not reached - this is the `torch.rand` half
    of the stream contract."""
    overrides = {'MCMC_Simulations': sims, 'Random_Seed': 1, 'Recompute_Inner_MC': recompute,
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
    """(cva, mtm profile, the WHOLE CVA gradient vector, cashflows). 512 scenarios, so the pricer
    takes the Sobol branch - the inner block's half of the stream contract - and the boundary
    correction has a population to fit a kernel to."""
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

    Killing mutation: the replay one draw ahead on the regular generator.
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

#: The replay's peak bound for `streamed` at 16 x 4096, float32.
STREAM_PEAK_MIB = 265.0


class Unstreamed(calculation.CreditMonteCarlo):
    """A credit Monte Carlo whose pricers walk every fixing strip unstreamed."""

    def _init_shared_mem(self, *args, **kwargs):
        shared = super()._init_shared_mem(*args, **kwargs)
        shared.oss_chunk = 0
        return shared


def replayed(calc, config, grid, batch, inner):
    """(the run's results, its peak in MiB above its floor) of `config` reported on `grid`, its CVA
    gradient replayed by `Recompute_Inner_MC`, float32 on the device."""
    base = config.params['System Parameters']['Base_Date']
    params = {
        'Base_Date': base, 'Run_Date': base.strftime('%Y-%m-%d'), 'Currency': 'USD',
        'Time_grid': grid, 'Batch_Size': batch, 'Simulation_Batches': 1, 'Random_Seed': 1,
        'Tenor_Offset': 0.0, 'MCMC_Simulations': inner, 'Deflation_Interest_Rate': 'USD',
        'Gradient_Variables': 'Factors', 'Recompute_Inner_MC': 'Yes',
        'Credit_Valuation_Adjustment': {
            'Calculate': 'Yes', 'Counterparty': 'CPTY', 'Deflate_Stochastically': 'No',
            'Stochastic_Hazard_Rates': 'No', 'Gradient': 'Yes'}}
    # the cycle a boundary registration makes outlives refcounting, so collect before the floor
    gc.collect()
    torch.cuda.empty_cache()
    floor = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = calc(config, device=utils.calculation_device(None, 0), prec=torch.float32).execute(params)
    return out['Results'], (torch.cuda.max_memory_allocated() - floor) / 2.0 ** 20


def streamed(calc=calculation.CreditMonteCarlo, batch=16, inner=4096):
    """(cva, CVA gradient, the run's peak in MiB above its floor) of a 24-fixing knock-in TARF
    reported monthly."""
    deal = tarf._tarf(tarf.UNREACHABLE_TARGET, [30 * (i + 1) for i in range(24)],
                      barrier=tarf.BARRIER, buy_sell='Sell')
    results, peak = replayed(calc, tarf._cfg(deal, tarf.SPOT, counterparty=True, simulate_fx=True),
                             '0d 1m(1m)', batch, inner)
    return float(results['cva']), results['grad_cva']['Gradient'].values.astype(np.float64), peak


@pytest.mark.skipif(not torch.cuda.is_available(), reason='reads the device allocator')
def test_the_streamed_strip_holds_its_peak_and_the_unstreamed_gradient():
    """The replay's strip checkpointed `oss_chunk` fixings at a time peaks under
    `STREAM_PEAK_MIB` and reproduces the unstreamed loop's cva and gradient bit for bit.

    Readings: streamed 197 MiB, unstreamed 909; a process's first run carries cuBLAS's 32 MiB
    workspace too. The alive branch back on the tape (the stream's `no_grad` blocks removed) reads
    388 MiB.

    Killing mutation: the strip walked as one piece, never checkpointed - 941 MiB.
    """
    cva, gradient, peak = streamed()
    whole_cva, whole_gradient, whole_peak = streamed(Unstreamed)
    assert cva == whole_cva and np.array_equal(gradient, whole_gradient), (
        'streaming moved the CVA or its gradient:\n{}\n{}'.format(gradient, whole_gradient))
    assert peak < STREAM_PEAK_MIB < whole_peak, (
        'peak {:.1f} MiB streamed, {:.1f} unstreamed, bound {}'.format(
            peak, whole_peak, STREAM_PEAK_MIB))


#: The bound on `dense`'s daily replay peak over its weekly one, at 64 x 4096, float32.
DENSE_PEAK_FACTOR = 1.75


def dense(grid):
    """(the marks on each reported date, the run's peak in MiB above its floor) of a twelve-coupon
    monthly autocall with a put at 70, reported on `grid`. Its equity's GBM has no vol, so the spot
    is the base spot on every date and a row's mark is a function of its date alone; the sloped
    curves and smile give each row its own discounts, carry and vols; 64 scenarios take the
    canonical inner block."""
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
    results, peak = replayed(calculation.CreditMonteCarlo, config, grid, 64, 4096)
    return results['mtm'], peak


@pytest.mark.skipif(not torch.cuda.is_available(), reason='reads the device allocator')
def test_a_daily_grid_marks_the_weekly_dates_within_the_weekly_peak():
    """A pricer prices the rows sharing a fixing interval as one block: a weekly grid puts four or
    five of `dense`'s rows in an interval, a daily one thirty, past `oss_rows` (12), so the daily
    block is priced twelve rows at a time. A row's strip reads only its own row and draws what it
    would in the whole block, so the daily grid marks every weekly date to the bit, and its replay
    peaks within `DENSE_PEAK_FACTOR` of the weekly one's.

    Readings: weekly 216 MiB, 248 as a process's first run (cuBLAS's workspace), daily 294.

    Killing mutations: the block priced whole, every row at once - daily 529 MiB, 2.45 times the
    weekly; the vol strip handed to every piece whole - a weekly date moves by 6.4e-05 on marks
    spread 5e-03.
    """
    weekly, weekly_peak = dense('0d 1w(1w)')
    daily, daily_peak = dense('0d 1d(1d)')
    shared = daily.loc[weekly.index].values
    assert weekly.values.std() > 0.0 and np.array_equal(weekly.values, shared), (
        'the daily grid moved a weekly date by {:.6g}'.format(
            float(np.abs(weekly.values - shared).max())))
    assert daily_peak < DENSE_PEAK_FACTOR * weekly_peak, (
        'peak {:.1f} MiB daily, {:.1f} weekly, bound {}x'.format(
            daily_peak, weekly_peak, DENSE_PEAK_FACTOR))


# ---------------------------------------------------------------- (g) a trade's draws are its own

def knock_in_tensor(neighbour):
    """The knock-in TARF's own exposure tensor at 64 x 128, booked alone or after an accumulator
    fixing on its first three dates, which asks the inner draws for other strip lengths first."""
    config = tarf._cfg(KNOCK_IN_CMC, tarf.SPOT, counterparty=True, simulate_fx=True)
    if neighbour:
        dates = [tarf.BASE + pd.Timedelta(days=d) for d in tarf.BIMONTHLY[:3]]
        config.deals['Deals']['Children'].insert(0, {'Instrument': construct_instrument({
            'Object': 'FXAccumulatorOptionDeal', 'Reference': 'ACC1', 'Currency': 'USD',
            'Underlying_Currency': 'AUD', 'Discount_Rate': 'USD', 'FX_Volatility': 'AUD.USD',
            'Buy_Sell': 'Sell', 'Option_Type': 'Call', 'Strike_Price': tarf.STRIKE,
            'Underlying_Amount': tarf.N1, 'LeverageNotional': tarf.N2,
            'Barrier_Type': 'Up_And_Out', 'Barrier_Price': 0.75,
            'Accumulator_ExpiryDates': [[d, d, None] for d in dates]}, {})})
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
