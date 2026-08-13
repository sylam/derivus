"""`Recompute_Inner_MC`: a TARF's inner Monte Carlo re-simulated in backward() instead of taped.

An MC-priced deal builds one autograd graph per pricing and the terminal backward holds every one
of them at once - every fixing of every reporting row of every deal - while the simulation that
built them is cheap. `pricing.InnerMCRecompute` trades the tape for a second forward: the node's
forward runs under `no_grad`, its backward re-runs the SAME callable under `enable_grad` and
contracts the cotangent through that one graph, which dies immediately.

THE COUNTER IS THE STORAGE. What is saved is where each random stream stood
(`utils.rng_position`), never what it produced. Sobol draws are memoized per
(dimension, sample_size, batch), so rewinding the counter hands the replay the very same tensor;
the regular generator - `torch.rand` at small batches, and every Heston-Nandi unmonitored sub-step
- has no memo, so its own state is saved and restored. Both streams are exercised here, and which
one each fixture reads is ATTRIBUTED rather than asserted: `pv_MC_Tarf` takes Sobol above 16
scenarios, so base valuation's one scenario reads `torch.rand` and the 512-scenario exposure reads
Sobol, and desynchronising ONE of them moves that fixture's gradient and the other fixture's by
exactly zero (`test_each_fixture_reads_the_stream_it_claims`, measured 2.80e+05 and 1.17e+03).

WHY THE GRADIENT GATE IS `array_equal` AND NOT A TOLERANCE. The replay is the same kernels on the
same inputs in the same order, so nothing about it is approximate, and a tolerance would let a
desynchronised stream through as "close enough" - which is exactly what a wrong replay looks like
at 8192 paths. Measured: the whole gradient vector is bit-identical on both paths, base valuation
(7 factors) and CVA (14), so no tolerance is owed and none is offered - and the smallest of the
mutations below moves the gradient by 1.5e-04 relative, so a tolerance loose enough to be called
one would pass it. The identity is also GRID-INVARIANT, which is what says it is exactness and not
a coincidence of one grid: re-taken on `0d 1m(1m)`, `0d 2m(2m)` and `0d 3m(3m)` (18, 12 and 10
reporting rows), and with `Dynamic_Scenario_Dates` pinned off as well as on the declared default,
every one of cva, profile, cashflows and gradient is equal to the last bit on both deals.

THE BOUNDARY DECISIONS RIDE THE NODE'S OUTPUTS. `stochastic_boundary_correction` is
`gap - gap.detach()` times a detached coefficient, and the untaped forward has no graph to give
`gap`. So the gap's VALUE is computed under `no_grad` for the registration to report and the NODE
is what connects it: the correction's coefficient arrives as that output's cotangent and the
graph-carrying half is built inside the recompute. `NoBoundaryInjection` below drops exactly that
cotangent and the gradient moves, which is what says the correction is live rather than assumed.

The mutations are the point of the file. Bit-identity gates pass trivially against a node that
quietly reuses the forward's own graph, or one that never rewinds anything - so the counter is
desynchronised by one draw, the boundary cotangent is dropped, and the replay is fed stale inputs,
and each must break the gradient it is supposed to break, on BOTH streams (measured, as a fraction
of the gradient's own largest entry: 7.5e-02 / 3.5e-01 / 1.5e-04 on base valuation and
1.4e-02 / 2.8e-01 / 8.4e-05 on the CVA). That division of labour is itself measured: a
`Recompute_Inner_MC: 'Yes'` that silently kept taping - the switch dead, the node never engaged -
leaves every bit-identity gate here green and fails ten, all of them scored on a mutation.

WHERE THE REPLAY RUNS IS WHERE ITS DEFECTS ARE VISIBLE, and only there. `backward()` runs once per
pricing block and only when a gradient is asked for - measured on this fixture, the node's forward
runs 1 time under base valuation and 6 under exposure, and its backward runs the same number of
times WITH sensitivities on and zero times with them off. So the by-product law (a settled cashflow
is an output the caller performs once, never a side effect of `simulate`) is gated at
`Gradient: 'Yes'`, where a replay that books settles 36 extra cashflows and the reported frame
moves by 6.0e+06 while the cva, the profile AND the whole CVA gradient stay bit-identical - the
cashflow assertion is the only detector there is, and at `Gradient: 'No'` it detects nothing
because nothing replayed.

WHAT THE NODE CANNOT DO is gated too. Detaching the saved inputs is what stops the replay walking
back into the outer graph, and it is also why a SECOND derivative through it is severed and comes
back partly zero - so `backward` refuses `create_graph` naming the switch, rather than reporting a
Hessian that looks like a Hessian.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pytest
import torch

import derivus
from derivus import pricing, run_baseval, utils
import test_boundary_tarf_events as tarf

DTYPE = torch.float64
#: The node under test, bound at import. The mutations below SUBCLASS it and the gate monkeypatches
#: the pricer's global, so a mutant calling `pricing.InnerMCRecompute` would call itself.
RECOMPUTE = pricing.InnerMCRecompute
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


def cmc(deal, gradient=False, recompute='No', batches=1, batch=512, mcmc=128):
    """(cva, mtm profile, the WHOLE CVA gradient vector, cashflows). 512 scenarios, so the pricer
    takes the Sobol branch - the memoized half of the stream contract - and the boundary
    correction has a population to fit a kernel to."""
    overrides = {
        'Run_Date': tarf.BASE.strftime('%Y-%m-%d'), 'Time_grid': '0d 2m(2m)', 'Batch_Size': batch,
        'Simulation_Batches': batches, 'Random_Seed': 1, 'Currency': 'USD', 'Tenor_Offset': 0.0,
        'MCMC_Simulations': mcmc, 'Deflation_Interest_Rate': 'USD', 'Generate_Cashflows': 'Yes',
        'Gradient_Variables': 'Factors', 'Recompute_Inner_MC': recompute,
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


def base_hessian(recompute, sims=1 << 10):
    """The reported second-order block. `Greeks: 'All'` is what sets `Base_Reval_State.gamma`, so
    `SensitivitiesEstimator` runs with `create_graph=True` and the node is asked to be
    differentiated twice.

    On THIS fixture it never gets that far, which is the point of the gate below: the TARF
    registers a boundary correction and base valuation refuses a second derivative over one
    before any node is reached (`tests/test_base_valuation_gamma.py` owns that refusal). The
    autocall in `test_recompute_equity_pricers` registers none at base valuation and is where the
    node's OWN refusal is measured."""
    _, out = run_baseval(tarf._cfg(KNOCK_IN, tarf.SPOT), overrides={
        'MCMC_Simulations': sims, 'Random_Seed': 1, 'Greeks': 'All',
        'Recompute_Inner_MC': recompute})
    return out['Results']['Greeks_Second'].values.astype(np.float64)


# ---------------------------------------------------------------- the switch is declared

def test_the_switch_is_declared_where_the_engine_reads_it():
    """A framework feature ships behind a JSON switch or it does not ship. The home is the
    CALCULATION block and not the deal, because which pricings a run can afford to tape is a
    property of the valuation engine and the machine it is on, not of the trade."""
    from derivus.schema import mapping
    for calc_type in ('BaseValuation', 'CreditMonteCarlo'):
        declared = mapping['Calculation']['types'][calc_type]
        assert 'Recompute_Inner_MC' in declared, f'{calc_type} cannot author the switch'
        assert declared['Recompute_Inner_MC']['value'] == 'No', 'the default is not off'
        assert declared['Recompute_Inner_MC']['values'] == ['Yes', 'No']


def test_a_state_that_was_never_told_runs_the_taped_path():
    """The attribute is declared on `Calculation_State` rather than by the calculations that set
    it, so a pricer reads it with no fallback and a state built by anything else - a fork, a
    solver's inner MC - is on the taped path rather than on an AttributeError."""
    state = utils.Calculation_State(
        {}, torch.ones([1, 1], dtype=DTYPE), 8, None, 'Constant', 1, False)
    assert state.recompute_inner_mc is False


# ---------------------------------------------------------------- (a) the value must not move

@pytest.mark.parametrize('greeks', [False, True])
def test_the_base_price_is_bit_identical_with_the_node_on(greeks):
    """BIT-identical, not approximately: a recompute that drew different numbers would still
    converge to the same price at 4096 paths and be wrong in every digit that matters."""
    off, _ = baseval(KNOCK_IN, greeks=greeks)
    on, _ = baseval(KNOCK_IN, greeks=greeks, recompute='Yes')
    assert off == on, f'price moved with the node on: {off!r} -> {on!r}'


@pytest.mark.parametrize('gradient', [False, True])
@pytest.mark.parametrize('deal,label', [(KNOCK_IN_CMC, 'knock-in'), (PIN_CMC, 'pin')])
def test_the_exposure_and_its_cashflows_are_bit_identical_with_the_node_on(deal, label, gradient):
    """The whole profile and the settled cashflows, because the by-products are where an untaped
    forward goes wrong quietly: the simulation is called TWICE under the node, and a cashflow
    accrued inside it rather than returned would be settled twice.

    BOTH sensitivity settings, because the second call only happens under one of them. With them
    off the node's backward never runs (measured: 6 forwards, 0 backwards), so this is the untaped
    forward's by-product plumbing alone; with them on the replay fires once per block (6) and the
    double-settle is reachable - `test_a_cashflow_settled_inside_the_replay_moves_only_the_frame`
    is the mutation that proves the cashflow assertion sees it, and that nothing else in this file
    does. The reporting order is what makes that true: `Credit_Monte_Carlo` harvests
    `t_Cashflows` AFTER the batch's backward (calculation.py:1727), so a booking made in the replay
    is still in the frame when it is read."""
    cva_off, mtm_off, _, cash_off = cmc(deal, gradient=gradient)
    cva_on, mtm_on, _, cash_on = cmc(deal, gradient=gradient, recompute='Yes')
    assert np.array_equal(mtm_off, mtm_on), f'{label}: exposure moved with the node on'
    assert cva_off == cva_on, f'{label}: cva moved: {cva_off!r} -> {cva_on!r}'
    assert cash_off and all(np.array_equal(a, b) for a, b in zip(cash_off, cash_on)), (
        f'{label}: a settled cashflow moved - the simulation ran its side effects twice')


# ---------------------------------------------------------------- (b) the gradient must not move

def test_the_base_gradient_is_bit_identical_with_the_node_on():
    """The WHOLE vector, boundary correction included (`Greeks: First` is what turns
    `boundary_aad` on), on a fixture small enough that the full tape fits - which is the only
    place the two paths can be compared at all."""
    price_off, grad_off = baseval(KNOCK_IN, greeks=True)
    price_on, grad_on = baseval(KNOCK_IN, greeks=True, recompute='Yes')
    assert grad_off is not None and np.abs(grad_off).max() > 0.0, 'no gradient was reported'
    assert price_off == price_on
    assert np.array_equal(grad_off, grad_on), (
        'the recomputed gradient is not the taped one:\n{}\n{}'.format(grad_off, grad_on))


@pytest.mark.parametrize('deal,label', [(KNOCK_IN_CMC, 'knock-in'), (PIN_CMC, 'pin')])
def test_the_cva_gradient_is_bit_identical_with_the_node_on(deal, label):
    """The same statement under exposure, where the Sobol stream is the one being rewound and both
    boundary registrations are live - the latched redemption, whose gaps are built OUTSIDE the node
    and keep their own graph, and the knock-in, whose gaps are node OUTPUTS."""
    _, _, grad_off, _ = cmc(deal, gradient=True)
    _, _, grad_on, _ = cmc(deal, gradient=True, recompute='Yes')
    assert np.abs(grad_off).max() > 0.0, f'{label}: no gradient was reported'
    assert np.array_equal(grad_off, grad_on), (
        '{}: the recomputed CVA gradient is not the taped one:\n{}\n{}'.format(
            label, grad_off, grad_on))


def test_the_second_derivative_is_refused_rather_than_reported_wrong():
    """A second derivative over this fixture is REFUSED, whichever of the two reasons gets there
    first - and on the TARF there are two.

    THE NODE'S REASON, which is what this file is about and what it was measured on. The replay is
    rooted at detached copies of the saved inputs - that is what stops `autograd.grad`
    walking back into the outer graph and double-counting the first derivative - and a second
    derivative taken through a detached leaf is severed from the graph the outer pass holds. The
    failure is silent: it comes back with the entries that needed that path set to ZERO, which is a
    Hessian that looks like a Hessian. Measured before the refusal went in, `Greeks: 'All'` on this
    fixture: three leading entries of -1.74e6 / -4.03e5 / -1.67e4 taped, all three zero recomputed,
    and a fourth that merely disagreed.

    So the gate is the RAISE, and it names the switch - a caller who wanted gamma has to be told to
    turn it off rather than handed a plausible zero.

    ON THIS FIXTURE THE OUTER REFUSAL COMES FIRST, and both spellings are asserted because that
    ordering is itself a statement. The TARF registers a boundary correction, and a second
    derivative taken through THAT is wrong for a different reason - the correction's coefficient
    is detached, so differentiating it twice drops the density-derivative term - so base valuation
    refuses over the deal before the node is ever asked. Nothing is lost: the node's own refusal
    is measured in `test_recompute_equity_pricers`, on the autocall, which registers nothing at
    base valuation and does reach it.
    """
    with pytest.raises(Exception, match="Greeks: 'All' is refused"):
        base_hessian('No')
    with pytest.raises(Exception, match="Greeks: 'All' is refused"):
        base_hessian('Yes')


# ---------------------------------------------------------------- (e) the mutations

class AheadByOne(dict):
    """The quasi-random counter one draw ahead, INCLUDING for a shape nobody has drawn yet.

    `quasi_rng` reads its counter with `setdefault(key, 0)`, so a mutation that only increments the
    keys already present moves nothing: the counter is per (dimension, sample_size) and a pricing
    block is usually the first reader of its own shape. That was measured - the first version of
    this mutant passed against the node it was written to break. Overriding the read is what makes
    the desynchronisation apply to the draw actually taken.
    """

    def setdefault(self, key, default):
        return super().setdefault(key, default + 1)


def desynced_node(quasi, regular):
    """The node replaying ONE DRAW off the position it saved, on the streams named.

    Parameterised rather than written three times because the halves are the interesting objects:
    which stream a run reads is a function of its batch size - the memoized Sobol counter above 16
    scenarios (`pv_MC_Tarf`) and the regular generator below it - so a mutation of BOTH kills every
    fixture without saying which stream that fixture actually read, and the file would keep passing
    if one of them silently stopped being covered. The halves attribute it; the pair is the
    mutation the gradient gate is scored on.

    The forward pass is untouched, so every reported number is identical and only the derivative is
    wrong. That is the failure a recompute has and a tape does not, and it is the whole reason the
    position is saved; one draw is the smallest desynchronisation there is.
    """

    class Desynced(RECOMPUTE):
        @staticmethod
        def backward(ctx, *cotangents):
            simulate = ctx.simulate

            def replay(*theta):
                if quasi:
                    ctx.shared.t_quasi_rng_batch = AheadByOne(ctx.shared.t_quasi_rng_batch)
                if regular:
                    torch.rand(1, dtype=ctx.shared.one.dtype, device=ctx.shared.one.device)
                return simulate(*theta)

            ctx.simulate = replay
            return RECOMPUTE.backward(ctx, *cotangents)

    Desynced.__name__ = 'Desynced' + ''.join(
        [n for n, on in (('Sobol', quasi), ('Generator', regular)) if on])
    return Desynced


DesyncedStreams = desynced_node(True, True)
SOBOL_AHEAD = desynced_node(True, False)
GENERATOR_AHEAD = desynced_node(False, True)


class BookingReplay(RECOMPUTE):
    """The replay with a SIDE EFFECT: it settles a cashflow, which is what a pricer that accrued
    inside `simulate` instead of returning it would do on the second call.

    The one mutation in this file that leaves the whole gradient alone - it is the by-product law
    being broken rather than the derivative, and the cashflow frame is where it shows.
    """

    @staticmethod
    def backward(ctx, *cotangents):
        simulate = ctx.simulate

        def booking(*theta):
            outputs = simulate(*theta)
            for currency, by_time in ctx.shared.t_Cashflows.items():
                for time_index in by_time:
                    pricing.cash_settle(ctx.shared, currency, time_index, ctx.shared.one.new_full(
                        [ctx.shared.simulation_batch], 1e6))
            return outputs

        ctx.simulate = booking
        return RECOMPUTE.backward(ctx, *cotangents)


class NoBoundaryInjection(RECOMPUTE):
    """The node with every cotangent but the first dropped - the boundary half of the backward.

    Under the node a knock-in gap is an OUTPUT, and the correction assembled at the objective
    reaches the simulation as that output's cotangent. Dropping it is the recompute-shaped way to
    lose the boundary term: the price, the exposure and the ordinary AAD gradient all survive it.
    """

    @staticmethod
    def backward(ctx, *cotangents):
        return RECOMPUTE.backward(ctx, cotangents[0], *[None] * (len(cotangents) - 1))


class StaleInputs(RECOMPUTE):
    """The node replaying off inputs that are not the ones it was given - the spot strip moved by
    a basis point. A recompute is only a recompute if it re-runs at the SAME theta; this is the
    failure where it re-runs at a neighbouring one, which converges to something plausible."""

    @staticmethod
    def backward(ctx, *cotangents):
        simulate = ctx.simulate
        ctx.simulate = lambda spot, *rest: simulate(spot * 1.0001, *rest)
        return RECOMPUTE.backward(ctx, *cotangents)


def base_gradient(recompute):
    """(price, gradient) off the one-scenario run - the `torch.rand` stream."""
    return baseval(KNOCK_IN, greeks=True, recompute=recompute)


def cva_gradient(recompute):
    """(cva, gradient) off the 512-scenario run - the Sobol stream."""
    cva, _, gradient, _ = cmc(KNOCK_IN_CMC, gradient=True, recompute=recompute)
    return cva, gradient


@pytest.mark.parametrize('mutant', [DesyncedStreams, NoBoundaryInjection, StaleInputs])
@pytest.mark.parametrize('run,stream', [(base_gradient, 'torch.rand'), (cva_gradient, 'Sobol')])
def test_a_mutated_node_fails_the_gradient_gate(mutant, run, stream, monkeypatch):
    """Every mutation must break the gradient the unmutated node reproduces bit for bit, on BOTH
    streams, and the unmutated reading is taken in the same run so the gate cannot pass by
    measuring nothing.

    Scored on the gradient alone: all three leave the forward pass untouched, so the reported value
    agrees in every digit under each of them - which is the point, and the reason a price gate over
    this subsystem is worth nothing. As a fraction of the gradient's largest entry (3.73e+06 base,
    8.22e+04 CVA) the kills are 7.5e-02 / 3.5e-01 / 1.5e-04 and 1.4e-02 / 2.8e-01 / 8.4e-05, in the
    order they are parametrized.
    """
    value_off, grad_off = run('No')
    monkeypatch.setattr(pricing, 'InnerMCRecompute', mutant)
    value_on, grad_on = run('Yes')
    assert value_off == value_on, 'the mutation moved the VALUE - it is not a backward-only defect'
    assert not np.array_equal(grad_off, grad_on), (
        '{} on the {} stream reproduced the taped gradient exactly, so the gate it is meant to '
        'fail measures nothing:\n{}'.format(mutant.__name__, stream, grad_off))


@pytest.mark.parametrize('run,stream,kills,spectates', [
    (base_gradient, 'torch.rand', GENERATOR_AHEAD, SOBOL_AHEAD),
    (cva_gradient, 'Sobol', SOBOL_AHEAD, GENERATOR_AHEAD)], ids=['torch.rand', 'Sobol'])
def test_each_fixture_reads_the_stream_it_claims(run, stream, kills, spectates, monkeypatch):
    """One HALF of the desynchronisation at a time, so the file states which stream each fixture
    covers instead of assuming it.

    The mutation of both streams above kills everything and would go on killing everything if a
    fixture quietly stopped reading one of them - move `pv_MC_Tarf`'s Sobol threshold past 512, or
    the exposure's batch under 16, and the memoized counter is no longer exercised anywhere while
    every test still passes. Here the OTHER half must be a NO-OP, bit for bit, which is a claim
    only the fixture that reads exactly one stream can make.

    Measured: the regular generator one draw ahead moves the base gradient by 2.80e+05 and the CVA
    gradient by exactly 0.0; the Sobol counter one batch ahead moves the CVA gradient by 1.17e+03
    and the base gradient by exactly 0.0.
    """
    _, grad_off = run('No')
    monkeypatch.setattr(pricing, 'InnerMCRecompute', kills)
    _, grad_desynced = run('Yes')
    monkeypatch.setattr(pricing, 'InnerMCRecompute', spectates)
    _, grad_other = run('Yes')
    assert not np.array_equal(grad_off, grad_desynced), (
        f'{stream} is not the stream the {run.__name__} fixture reads: desynchronising it '
        f'changed nothing')
    assert np.array_equal(grad_off, grad_other), (
        f'the {run.__name__} fixture also reads the other stream, so {stream} is not what the '
        f'combined mutation kills it on:\n{grad_off}\n{grad_other}')


def test_a_cashflow_settled_inside_the_replay_moves_only_the_frame(monkeypatch):
    """The by-product law, gated: a `simulate` that BOOKS instead of returning is caught by the
    cashflow assertion and by nothing else in this file.

    Measured on the knock-in at `Gradient: 'Yes'`, 36 extra settlements over the 6 blocks: the cva,
    the exposure profile and the whole 14-entry CVA gradient are bit-identical, and the reported
    frame moves by 6.0e+06. That asymmetry is the reason the exposure gate carries a cashflow
    comparison at all, and the reason it is taken with sensitivities ON - with them off the replay
    never runs, the booking never happens, and every one of the four channels agrees.
    """
    cva_off, mtm_off, grad_off, cash_off = cmc(KNOCK_IN_CMC, gradient=True, recompute='Yes')
    monkeypatch.setattr(pricing, 'InnerMCRecompute', BookingReplay)
    cva_on, mtm_on, grad_on, cash_on = cmc(KNOCK_IN_CMC, gradient=True, recompute='Yes')
    assert cva_off == cva_on and np.array_equal(mtm_off, mtm_on), (
        'the booking reached the reported exposure - this mutation is meant to be a by-product one')
    assert np.array_equal(grad_off, grad_on), 'the booking reached the CVA gradient'
    assert not all(np.array_equal(a, b) for a, b in zip(cash_off, cash_on)), (
        'a cashflow settled inside the replay did not move the reported frame, so the cashflow '
        'half of the exposure gate measures nothing')


def test_the_boundary_correction_is_what_the_injection_carries(monkeypatch):
    """Names the size of what `NoBoundaryInjection` drops, so the mutation above is not merely
    "different" - the gap cotangent IS the knock-in's boundary correction, and dropping it leaves
    the uncorrected AAD gradient that `test_boundary_tarf_events` measures as short.

    Measured: 1.32e+06 on a gradient whose largest entry is 3.73e+06, and all 7 entries move. The
    floor is one percent of that gradient's own scale rather than zero, because a correction that
    arrived at 1e-300 would satisfy "moved" and mean the cotangent had been all but lost; 35x is
    the margin it currently clears by.
    """
    _, corrected = baseval(KNOCK_IN, greeks=True, recompute='Yes')
    monkeypatch.setattr(pricing, 'InnerMCRecompute', NoBoundaryInjection)
    _, uncorrected = baseval(KNOCK_IN, greeks=True, recompute='Yes')
    moved = np.abs(corrected - uncorrected)
    assert moved.max() > 0.01 * np.abs(corrected).max(), (
        'the boundary correction reaches all but nothing through the node: {:.6g} on {:.6g}'.format(
            moved.max(), np.abs(corrected).max()))
    print('\nboundary correction through the node: max |delta| = {:.6g} on a gradient of '
          '{:.6g}'.format(moved.max(), np.abs(corrected).max()))
