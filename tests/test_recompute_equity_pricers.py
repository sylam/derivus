"""`Recompute_Inner_MC` on the two equity MC pricers, against the shape `pv_MC_Tarf` proved.

`tests/test_recompute_inner_mc.py` states the contract on the TARF; this file makes the same
statements about `pv_MC_AutoCallSwap` and `pv_discrete_barrier_option`, which reach the node through
`pricing.InnerMCRecompute.run`. One file because the two share a world, a market and every
bit-identity statement.

THE GAP IS A NODE OUTPUT WHEN THE SIMULATION IS WHAT DECIDED IT. The autocall's coupon trigger is
read off `Sj` inside the fixing loop, so its gap is an OUTPUT and the correction's coefficient
arrives as its cotangent: dropping every cotangent but the marks' is bit-identical to removing the
correction outright, max |delta| 5.512191e-06 on a gradient of 2.147838e-03. The barrier's latch is
decided on an OUTER scenario spot, so its registration stays outside and the same drop is a no-op,
while removing the correction moves it by 2.194252e-03 on 7.259363e-01.

THE SETTLE-OUTSIDE RULE has two observable halves. A cashflow settled inside the callable books
twice, and the credit Monte Carlo takes the batch's `backward()`, where the replay runs, before it
saves the batch's cashflows, so the report carries both. And the forward's booking carries no
GRAPH, so a COLLATERALISED exposure reading that ledger through `C_ts_te` loses 8.7% of the CVA
gradient.

UNGATED: `past_fixings` and the floating leg, reached only by `QEDI_CustomAutoCallSwap_V2`. Both
are hoisted into theta on the same rule as everything else and carried on trust.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pytest

import derivus
from derivus import run_baseval, utils
from derivus.instruments import construct_instrument
import test_barrier_bridge as bb
import test_boundary_pricer_events as bp

DTYPE = bb.DTYPE
#: The two adopters, in the fixtures that already gate their boundary registrations -
#: `test_boundary_pricer_events` owns the deals and `test_barrier_bridge` owns the market.
DEALS = {'autocall': bp.AUTOCALL, 'barrier': bp.DISCRETE_BARRIER}
PRICERS = list(DEALS)


def _cfg(pricer, spot=bb.SPOT, counterparty=False, collateralised=False):
    c = bb._cfg()
    c.params['Price Factors']['EquityPrice.EQ']['Spot'] = spot
    if counterparty:
        c.params['Price Factors']['SurvivalProb.CPTY'] = {
            'Recovery_Rate': 0.4, 'Curve': utils.Curve([], [[0.0, 0.0], [10.0, 0.4]])}
    kids = [{'Instrument': construct_instrument(DEALS[pricer], {})}]
    c.deals['Deals']['Children'] = [{'Instrument': construct_instrument(
        dict(bp.NETTING, Reference='NS1', Collateralized='True'), {}), 'Children': kids}
    ] if collateralised else kids
    return c


def _ulps(a, b):
    """Distance in representable float64 steps - the only non-arbitrary way to say "the last bit".

    A tolerance would be a number somebody chose; this is a count of the floats between two
    readings, so a gate written on it cannot be widened until it passes. Used for exactly one
    statement below, where the node legitimately sums two cotangent paths in a different order from
    the taped engine, and it is quoted beside the defect it has to separate (1 step against 8.7%).
    """
    def ordinal(x):
        bits = x.view(np.int64)
        return np.where(np.signbit(x), -(bits & np.int64(0x7FFFFFFFFFFFFFFF)), bits)

    return np.abs(ordinal(np.ascontiguousarray(a)) - ordinal(np.ascontiguousarray(b)))


def cmc(pricer, gradient=False, recompute='No', batch=256, mcmc=64, collateralised=False):
    """(cva, mtm profile, the WHOLE CVA gradient vector, cashflows). 256 scenarios, so the pricer
    takes the Sobol branch - the inner block's half of the stream contract - and the boundary
    registration has a population to fit a kernel to."""
    overrides = {
        'Run_Date': bb.BASE.strftime('%Y-%m-%d'), 'Time_grid': '0d 3m(3m)', 'Batch_Size': batch,
        'Simulation_Batches': 1, 'Random_Seed': 1, 'Currency': 'USD', 'Tenor_Offset': 0.0,
        'MCMC_Simulations': mcmc, 'Deflation_Interest_Rate': 'USD', 'Generate_Cashflows': 'Yes',
        'Gradient_Variables': 'Factors', 'Recompute_Inner_MC': recompute,
        'Credit_Valuation_Adjustment': {
            'Calculate': 'Yes', 'Counterparty': 'CPTY', 'Deflate_Stochastically': 'No',
            'Stochastic_Hazard_Rates': 'No', 'Gradient': 'Yes' if gradient else 'No'}}
    _, out = derivus.run_cmc(_cfg(pricer, counterparty=True, collateralised=collateralised),
                             prec=DTYPE, overrides=overrides)
    grad = out['Results']['grad_cva']['Gradient'].values.astype(np.float64) if gradient else None
    # .values per currency FRAME - iterating a DataFrame yields its column labels, and a gate
    # built on those cannot fail (verified against a zeroed-cashflow run)
    cashflows = [frames.values for frames in out['Results'].get('cashflows', {}).values()]
    return float(out['Results']['cva']), out['Results']['mtm'].values, grad, cashflows


def baseval(pricer, greeks=False, sims=1 << 12, recompute='No'):
    """(price, the WHOLE first-order gradient vector). One scenario, so `quasi_rng` is not reached
    and this is the `torch.rand` half of the stream contract."""
    _, out = run_baseval(_cfg(pricer), overrides={
        'MCMC_Simulations': sims, 'Random_Seed': 1, 'Recompute_Inner_MC': recompute,
        'Greeks': 'First' if greeks else 'No'})
    rows = out['Results']['mtm']
    price = float(rows[rows['Reference'] == DEALS[pricer]['Reference']]['Value'].iloc[0])
    if not greeks:
        return price, None
    frame = out['Results']['Greeks_First']
    # 'Value' is the factor LEVEL (display_val=True); the other column is the gradient
    column, = [x for x in frame.columns if x != 'Value']
    return price, frame[column].values.astype(np.float64)


def base_hessian(pricer, recompute, sims=1 << 10, estimator='Yes'):
    """The reported second-order block. `Greeks: 'All'` is what sets `Base_Reval_State.gamma`, so
    `SensitivitiesEstimator` runs with `create_graph=True` and the node is asked to be
    differentiated twice. `estimator` is `Branch_And_Weight`: its default registers nothing to
    refuse, so a gate about the registration declares the crisp one."""
    _, out = run_baseval(_cfg(pricer), overrides={
        'MCMC_Simulations': sims, 'Random_Seed': 1, 'Greeks': 'All',
        'Recompute_Inner_MC': recompute, 'Branch_And_Weight': estimator})
    return out['Results']['Greeks_Second'].values.astype(np.float64)


# ---------------------------------------------------------------- the collateralised gradient

@pytest.mark.parametrize('pricer', PRICERS)
def test_a_collateralised_gradient_is_what_the_settle_outside_rule_is_gated_on(pricer):
    """The one place a cashflow's GRAPH is observable, and therefore the only real gate on the rule
    that a simulation returns its settlements rather than performing them.

    A `cash_settle` called inside the callable runs under the node's `no_grad` forward, so the
    cashflow reaches `t_Cashflows` carrying NOTHING. Uncollateralised the defect is invisible;
    collateralised the exposure reads it through `C_ts_te` and the CVA gradient moves - max |delta|
    7.585440e-05 on a gradient of 8.678265e-04, 8.7%.

    TWO STEPS, NOT ZERO, and only for the autocall - 12 paired repeats per pricer at a fixed seed:

      autocall  off vs off  0 steps on 11 of 11 - the TAPED path reproduces itself exactly.
                on  vs on   2 steps on 10 of 11, on entries 7 and 8 (`EquityPriceVol.EQ` 1.2).
                off vs on   0 on 7 readings, 2 on 5. Never 1, never above 2.
      barrier   0 steps on every one of the 35 readings, all three comparisons.

    So the step IS the node's replayed backward disagreeing with itself. A defect at 8.7% and a
    reduction at 2 ulps are not confusable.

    Killing mutation: the replay fed theta a basis point off (`spot * 1.0001`).
    """
    cva_off, mtm_off, grad_off, cash_off = cmc(pricer, gradient=True, collateralised=True)
    cva_on, mtm_on, grad_on, cash_on = cmc(pricer, gradient=True, recompute='Yes',
                                           collateralised=True)
    assert np.array_equal(mtm_off, mtm_on) and cva_off == cva_on, (
        f'{pricer}: the collateralised exposure moved with the node on')
    assert all(np.array_equal(a, b) for a, b in zip(cash_off, cash_on))
    assert np.abs(grad_off).max() > 0.0, f'{pricer}: no gradient was reported'
    steps = _ulps(grad_off, grad_on)
    # <= 2 is the measured quantum, not a tolerance: the autocall's two states are exactly two
    # steps apart and nothing reads between them
    assert steps.max() <= 2, (
        '{}: the collateralised gradient moved by {} float64 steps (max |d| {:.6g} on {:.6g}) - '
        'that is not summation order, it is a channel the node lost:\n{}\n{}'.format(
            pricer, int(steps.max()), float(np.abs(grad_off - grad_on).max()),
            float(np.abs(grad_off).max()), grad_off, grad_on))


# ---------------------------------------------------------------- the value and gradient must not move

@pytest.mark.parametrize('pricer', PRICERS)
def test_the_base_price_and_gradient_are_bit_identical_with_the_node_on(pricer):
    """The price and the WHOLE vector (`Greeks: First` is what turns `boundary_aad` on), on a fixture
    small enough that the full tape fits - which is the only place the two paths can be compared.
    One scenario, so `quasi_rng` is not reached: this is the `torch.rand` half of the stream
    contract, the generator one draw ahead moving it.

    Killing mutation: the replay one draw ahead on the regular generator.
    """
    price_off, grad_off = baseval(pricer, greeks=True)
    price_on, grad_on = baseval(pricer, greeks=True, recompute='Yes')
    assert price_off != 0.0, f'{pricer}: the fixture prices at zero and gates nothing'
    assert np.abs(grad_off).max() > 0.0, f'{pricer}: no gradient was reported'
    assert price_off == price_on, (price_off, price_on)
    assert np.array_equal(grad_off, grad_on), (
        '{}: the recomputed gradient is not the taped one:\n{}\n{}'.format(
            pricer, grad_off, grad_on))


@pytest.mark.parametrize('pricer', PRICERS)
def test_the_exposure_its_cashflows_and_the_cva_gradient_are_bit_identical_with_the_node_on(pricer):
    """The whole profile, the settled cashflows and the whole CVA gradient, under exposure, where
    the Sobol stream is the one being rewound and each pricer's boundary registration is live - the
    autocall's through the node's own outputs, the barrier's alongside it on the outer graph.

    Dropping every cotangent but the marks' fails the autocall's half - its coupon-trigger
    correction rides the gap output - and leaves the barrier's alone.

    Killing mutation: the replay fed theta a basis point off (`spot * 1.0001`).
    """
    cva_off, mtm_off, grad_off, cash_off = cmc(pricer, gradient=True)
    cva_on, mtm_on, grad_on, cash_on = cmc(pricer, gradient=True, recompute='Yes')
    assert np.array_equal(mtm_off, mtm_on), f'{pricer}: exposure moved with the node on'
    assert cva_off == cva_on, f'{pricer}: cva moved: {cva_off!r} -> {cva_on!r}'
    assert any(np.abs(x).max() > 0.0 for x in cash_off), (
        f'{pricer}: every settled cashflow is zero - this gate is reading nothing')
    assert all(np.array_equal(a, b) for a, b in zip(cash_off, cash_on)), (
        f'{pricer}: a settled cashflow moved with the node on')
    assert np.abs(grad_off).max() > 0.0, f'{pricer}: no gradient was reported'
    assert np.array_equal(grad_off, grad_on), (
        '{}: the recomputed CVA gradient is not the taped one:\n{}\n{}'.format(
            pricer, grad_off, grad_on))


def test_the_second_derivative_is_refused_rather_than_reported_wrong():
    """The node is FIRST ORDER whichever pricer holds it, and one switch governs them all, so the
    refusal has to reach every adopter rather than the one it was measured on. The taped reading
    beside it is what says the refused path is a path anyone would take.

    The AUTOCALL is where that can be said, and it is the only fixture in this repo that can say
    it. Base valuation gives it one reporting row, so no coupon is OBSERVED and it registers no
    boundary correction - which leaves the node as the only thing standing between `Greeks: 'All'`
    and a second derivative. Every other adopter registers one and is refused a step earlier (see
    `test_base_valuation_gamma`), which is a different finding, not this one.

    Killing mutation: the node's refusal of `create_graph` dropped, the severed derivative reported.
    """
    taped = base_hessian('autocall', 'No')
    assert np.abs(taped).max() > 0.0, 'no second derivative even with the node off'
    with pytest.raises(Exception, match='create_graph is not supported'):
        base_hessian('autocall', 'Yes')
