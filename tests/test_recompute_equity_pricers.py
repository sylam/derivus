"""`Recompute_Inner_MC` on the two equity MC pricers, against the shape `pv_MC_Tarf` proved.

`tests/test_recompute_inner_mc.py` is the template and states the contract; this file is the same
statements about `pv_MC_AutoCallSwap` and `pv_discrete_barrier_option`, which now reach the node
through `pricing.InnerMCRecompute.run` - the one line and the whole switch. One file rather than
two because the two pricers share a world, a market and every bit-identity statement; where they
genuinely DIFFER they get their own gate, and that difference is the finding below.

THE GAP IS A NODE OUTPUT WHEN THE SIMULATION IS WHAT DECIDED IT, which is the rule the three ports
settle - not symmetry.

  AUTOCALL   the coupon trigger is read off `Sj` inside the fixing loop, selected by loop state, so
             the untaped forward has no graph to give it: the gap is an OUTPUT and the correction's
             coefficient arrives as its cotangent. Measured below - dropping that cotangent is
             BIT-IDENTICAL to suppressing the correction outright.
  BARRIER    the latch is decided on `spot_block[-1]`, an OUTER scenario spot at an observation
             date. Its graph is the scenario generation's, which this node does not untape, so the
             whole registration stays outside and needs nothing from the replay. Measured - the
             injection mutant is a NO-OP here, and suppressing the correction still moves the
             gradient, so the correction is live and simply does not ride the node.

A CASHFLOW GATE ON THE REPORTED CASHFLOWS IS A PLACEBO, which was measured rather than reasoned
about. The autocall SETTLED inside its simulation before this change, so a replay would have
settled twice - but the replay runs in `backward()` and `save_cashflows` runs inside
`resolve_structure`, in the forward pass, so the second settlement lands after the snapshot and
every reported cashflow is bit-identical with the defect in place. What IS observable is the
cashflow's GRAPH: settled inside, it is booked under the node's `no_grad` forward and reaches
`t_Cashflows` carrying nothing, so a COLLATERALISED exposure reading that ledger through `C_ts_te`
loses the channel - 8.7% of the CVA gradient on this fixture. That is the gate, and it is the one
the TARF's own suite does not have.

`pv_MC_AutoCallSwap`'s SECOND BRANCH - the AVERAGING full-path sim, with its own
`torch.randn`/`quasi_rng` draw - is gated in section (f), on a fixture authored around three
defects the branch has at HEAD rather than in spite of them (see `_averaging_autocall`), which is
why no earlier attempt in this repo reached the code at all.

HONEST LIMITS, so nobody reads more coverage into this than there is. What stays UNGATED is
`past_fixings` - the averaging branch is handed an empty one - and the floating leg, reached only
by `QEDI_CustomAutoCallSwap_V2`, which does not price at HEAD (a floating-leg tensor-shape
mismatch, identically before this change). Both are hoisted into theta on the same rule as
everything else and carried on trust.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import pytest

import derivus
from derivus import pricing, run_baseval, utils
from derivus.instruments import construct_instrument
import test_barrier_bridge as bb
import test_boundary_pricer_events as bp
import test_hn_oss_pricers as hn
import test_recompute_inner_mc as rc

DTYPE = bb.DTYPE
#: The two adopters, in the fixtures that already gate their boundary registrations -
#: `test_boundary_pricer_events` owns the deals and `test_barrier_bridge` owns the market.
DEALS = {'autocall': bp.AUTOCALL, 'barrier': bp.DISCRETE_BARRIER}
PRICERS = list(DEALS)


def _averaging_autocall():
    """`bp.AUTOCALL` with TWO `Price_Fixing` dates per `Autocall_Coupon` date, which is the whole of
    what selects `pv_MC_AutoCallSwap`'s other branch: `no_averaging` (instruments.py:3889) is
    `len(pf_dates) == len(ac_dates)` over the fixings inside the window, so 9 against 4 is False.

    THREE OF THESE DATES ARE LOAD-BEARING, each around a defect the branch has at HEAD - which is
    why every autocall fixture in this repo is a one-fixing-per-coupon one.

      A FIXING ON THE COUPON DATE. `sim_autocall` divides its running sum by `averageCounter` and
      zeroes the counter at each coupon, so a block whose window OPENS on a coupon with no fixing
      before it inside that window divides by zero. The coupon's own fixing is inside every window
      that starts there, which makes the branch reachable wherever the reporting grid opens a block.
      A TRAILING FIXING, a week after the LAST coupon. `sample_ts` stays NUMPY on a block whose
      every remaining fixing is observed today (pricing.py:2587) and this branch calls
      `times.unsqueeze` - so the last reporting row, whose only remaining sample is itself, dies on
      "'numpy.ndarray' object has no attribute 'unsqueeze'". A fixing is not a reval date, so one
      after the last coupon adds no reporting row; it just keeps that block's window two samples
      long.
      COUPONS TWO DAYS OFF THE 3m REPORTING GRID. Rows sharing a start index are one block and a
      settlement is `tau == 0` - the row that IS the sample - so coupons ON the grid make every
      block one row long, and there `settle_rows.append(i)` and `.append(0)` are the same function.
      Two days late puts a reporting row immediately before each coupon, in its block, and the
      settlement on block-local row 1. Measured: 5 blocks, four of them `(2, [1])`.

    Returns `(the coupon dates, the deal)`; the dates are what the settlement gate reads.
    """
    coupons = [bb.BASE + pd.DateOffset(months=m) + pd.Timedelta(days=2) for m in (3, 6, 9, 12)]
    fixings = sorted([d - pd.Timedelta(days=7) for d in coupons] + coupons +
                     [coupons[-1] + pd.Timedelta(days=7)])
    return coupons, dict(bp.AUTOCALL, Expiry_Date=coupons[-1],
                         Price_Fixing=[[d, 0.0] for d in fixings],
                         Autocall_Coupons=[[d, 0.05] for d in coupons],
                         Autocall_Thresholds=[[d, 1.02] for d in coupons])


#: A DEAL, not a third port - registered after `PRICERS` is taken so that it reaches the shared
#: `_cfg`/`baseval`/`cmc` harness without joining the parametrizations above.
AVERAGING_COUPONS, DEALS['averaging'] = _averaging_autocall()


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
    takes the Sobol branch - the memoized half of the stream contract - and the boundary
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


def base_hessian(pricer, recompute, sims=1 << 10):
    """The reported second-order block. `Greeks: 'All'` is what sets `Base_Reval_State.gamma`, so
    `SensitivitiesEstimator` runs with `create_graph=True` and the node is asked to be
    differentiated twice."""
    _, out = run_baseval(_cfg(pricer), overrides={
        'MCMC_Simulations': sims, 'Random_Seed': 1, 'Greeks': 'All',
        'Recompute_Inner_MC': recompute})
    return out['Results']['Greeks_Second'].values.astype(np.float64)


#: The same two pricers under Heston-Nandi, base valuation. These are what gate the THETA HOIST:
#: the five GARCH scalars come off `t_Static_Buffer` and were read from the enclosing scope, and a
#: closure read is differentiated as a CONSTANT under the node - silently, and only for the factor
#: that was hoisted. Nothing in the GBM fixtures above can see it.
HN_DEALS = {
    'autocall': lambda: hn._autocall_cfg([30], [1.05], [0.05], 30, hn_params=hn.STRONG),
    'barrier': lambda: hn._barrier_cfg(
        'Down_And_Out', 90.0, list(range(1, 31)), 30, hn_params=hn.STRONG)}


def hn_baseval(pricer, recompute, sims=1 << 12):
    """(price, whole gradient vector, the Heston-Nandi entries of it)."""
    config, reference = HN_DEALS[pricer]()
    _, out = run_baseval(config, overrides={
        'MCMC_Simulations': sims, 'Random_Seed': 1, 'Greeks': 'First',
        'Recompute_Inner_MC': recompute})
    rows = out['Results']['mtm']
    price = float(rows[rows['Reference'] == reference]['Value'].iloc[0])
    frame = out['Results']['Greeks_First']
    column, = [x for x in frame.columns if x != 'Value']
    gradient = frame[column]
    hn_rows = [i for i in gradient.index if 'HestonNandiModelParameters' in str(i[0])]
    return (price, gradient.values.astype(np.float64),
            gradient.loc[hn_rows].values.astype(np.float64))


def suppressed_correction(run, monkeypatch):
    """`run()` with the boundary correction assembled at the objective removed entirely - the
    difference is what the correction is worth, whatever route it takes to the tape."""
    monkeypatch.setattr(pricing, 'boundary_correction', lambda *args, **kwargs: None)
    return run()


# ---------------------------------------------------------------- (a) the value must not move

@pytest.mark.parametrize('pricer', PRICERS)
@pytest.mark.parametrize('greeks', [False, True])
def test_the_base_price_is_bit_identical_with_the_node_on(pricer, greeks):
    """BIT-identical, not approximately: a recompute that drew different numbers would still
    converge to the same price at 4096 inner paths and be wrong in every digit that matters."""
    off, _ = baseval(pricer, greeks=greeks)
    on, _ = baseval(pricer, greeks=greeks, recompute='Yes')
    assert off != 0.0, f'{pricer}: the fixture prices at zero and gates nothing'
    assert off == on, f'{pricer}: price moved with the node on: {off!r} -> {on!r}'


@pytest.mark.parametrize('pricer', PRICERS)
def test_the_exposure_and_its_cashflows_are_bit_identical_with_the_node_on(pricer):
    """The whole profile and the settled cashflows, on both switch settings.

    WHAT THIS GATE DOES NOT SEE, stated because the obvious reading of it is wrong. A cashflow
    settled INSIDE the simulation would be settled a second time by the replay - but the replay
    runs in `backward()`, and `save_cashflows` is called by `resolve_structure`, inside the forward
    pass. Measured: with `cash_settle` put back inside `sim_spot`, every reported cashflow here is
    bit-identical, gradient on or off. So this reads the forward pass and nothing else, and the
    settle-outside rule is gated one test down, where it is actually observable."""
    cva_off, mtm_off, _, cash_off = cmc(pricer)
    cva_on, mtm_on, _, cash_on = cmc(pricer, recompute='Yes')
    assert np.array_equal(mtm_off, mtm_on), f'{pricer}: exposure moved with the node on'
    assert cva_off == cva_on, f'{pricer}: cva moved: {cva_off!r} -> {cva_on!r}'
    assert any(np.abs(x).max() > 0.0 for x in cash_off), (
        f'{pricer}: every settled cashflow is zero - this gate is reading nothing')
    assert all(np.array_equal(a, b) for a, b in zip(cash_off, cash_on)), (
        f'{pricer}: a settled cashflow moved with the node on')


@pytest.mark.parametrize('pricer', PRICERS)
def test_a_collateralised_gradient_is_what_the_settle_outside_rule_is_gated_on(pricer):
    """The one place a cashflow's GRAPH is observable, and therefore the only real gate on the rule
    that a simulation returns its settlements rather than performing them.

    A `cash_settle` called inside the callable runs under the node's `no_grad` forward, so the
    cashflow reaches `t_Cashflows` carrying NOTHING. Uncollateralised, nobody differentiates that
    ledger and the defect is invisible in every number the run reports. Collateralised, the
    exposure reads it through `C_ts_te` and the CVA gradient moves - measured on this fixture with
    the autocall's settlement put back inside its loop: max |delta| 7.585440e-05 on a gradient of
    8.678265e-04, 8.7%.

    ONE STEP, NOT ZERO, and only for the autocall - and the step is the TAPED side's, not the
    node's. Node-ON reproduces the pre-refactor (HEAD) gradient bit for bit; it is the refactored
    OFF path that moved 1 ulp under collateral, because settlement now flows through a stacked
    output indexed after the loop instead of being consumed inside it, which reorders the
    collateral chain's backward accumulation. Measured: 1 ulp on 2 of 13 entries, adversarially
    attributed by a three-way HEAD/taped/node comparison. The barrier returns marks and nothing
    else and is exactly bit-identical here (0 ulp on 14). A defect at 8.7% and a rounding at 1 ulp
    are not confusable, which is what makes this a gate rather than a widened tolerance."""
    cva_off, mtm_off, grad_off, cash_off = cmc(pricer, gradient=True, collateralised=True)
    cva_on, mtm_on, grad_on, cash_on = cmc(pricer, gradient=True, recompute='Yes',
                                           collateralised=True)
    assert np.array_equal(mtm_off, mtm_on) and cva_off == cva_on, (
        f'{pricer}: the collateralised exposure moved with the node on')
    assert all(np.array_equal(a, b) for a, b in zip(cash_off, cash_on))
    assert np.abs(grad_off).max() > 0.0, f'{pricer}: no gradient was reported'
    steps = _ulps(grad_off, grad_on)
    assert steps.max() <= 1, (
        '{}: the collateralised gradient moved by {} float64 steps (max |d| {:.6g} on {:.6g}) - '
        'that is not summation order, it is a channel the node lost:\n{}\n{}'.format(
            pricer, int(steps.max()), float(np.abs(grad_off - grad_on).max()),
            float(np.abs(grad_off).max()), grad_off, grad_on))


# ---------------------------------------------------------------- (b) the gradient must not move

@pytest.mark.parametrize('pricer', PRICERS)
def test_the_base_gradient_is_bit_identical_with_the_node_on(pricer):
    """The WHOLE vector (`Greeks: First` is what turns `boundary_aad` on), on a fixture small
    enough that the full tape fits - which is the only place the two paths can be compared."""
    price_off, grad_off = baseval(pricer, greeks=True)
    price_on, grad_on = baseval(pricer, greeks=True, recompute='Yes')
    assert np.abs(grad_off).max() > 0.0, f'{pricer}: no gradient was reported'
    assert price_off == price_on
    assert np.array_equal(grad_off, grad_on), (
        '{}: the recomputed gradient is not the taped one:\n{}\n{}'.format(
            pricer, grad_off, grad_on))


@pytest.mark.parametrize('pricer', PRICERS)
def test_the_cva_gradient_is_bit_identical_with_the_node_on(pricer):
    """The same statement under exposure, where the Sobol stream is the one being rewound and each
    pricer's boundary registration is live - the autocall's through the node's own outputs, the
    barrier's alongside it on the outer graph."""
    _, _, grad_off, _ = cmc(pricer, gradient=True)
    _, _, grad_on, _ = cmc(pricer, gradient=True, recompute='Yes')
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
    `test_a_registered_boundary_correction_is_refused_first`), which is a different finding, not
    this one.
    """
    taped = base_hessian('autocall', 'No')
    assert np.abs(taped).max() > 0.0, 'no second derivative even with the node off'
    with pytest.raises(Exception, match='create_graph is not supported'):
        base_hessian('autocall', 'Yes')


def test_a_registered_boundary_correction_is_refused_first():
    """The barrier's half of the statement above, and why the gate could not stay parametrized.

    `pv_discrete_barrier_option` registers its latch on the OUTER scenario spot, at base valuation
    as much as under exposure, and a second derivative taken through that correction silently
    drops the density-derivative term. So the refusal that fires is the outer one, whichever way
    `Recompute_Inner_MC` is set - the node is never asked. It names the deal.
    """
    for recompute in ('No', 'Yes'):
        with pytest.raises(Exception, match=r"Greeks: 'All' is refused.*BARR1"):
            base_hessian('barrier', recompute)


@pytest.mark.parametrize('pricer', PRICERS)
def test_the_heston_nandi_theta_survives_the_node(pricer):
    """The gate on the hoist itself, and the only one that can see it.

    A tensor `simulate` reads out of a CLOSURE is differentiated as a constant under the node -
    autograd only returns a gradient for what was passed to `apply` - and it fails silently, in one
    factor, on a fixture that has that factor. Both pricers read their five GARCH scalars off
    `t_Static_Buffer` in the enclosing scope; both now pass them in. Verified by reverting the
    barrier's hoist in the source and re-running: this gate turns red, and it is the only one here
    with a GARCH factor to turn red on. The non-zero assertion is what stops a GARCH-free fixture
    making the bit-identity vacuous."""
    price_off, grad_off, hn_off = hn_baseval(pricer, 'No')
    price_on, grad_on, hn_on = hn_baseval(pricer, 'Yes')
    assert len(hn_off) and np.abs(hn_off).max() > 0.0, (
        f'{pricer}: no Heston-Nandi sensitivity is reported at all, so a hoist that was never '
        f'made would pass this gate')
    assert price_off == price_on, f'{pricer}: HN price moved: {price_off!r} -> {price_on!r}'
    assert np.array_equal(grad_off, grad_on), (
        '{}: the recomputed HN gradient is not the taped one - a theta the simulation reads from '
        'its closure is being differentiated as a constant:\n{}\n{}'.format(
            pricer, grad_off, grad_on))


# ---------------------------------------------------------------- (e) the mutations

def base_gradient(pricer, recompute):
    """(price, gradient) off the one-scenario run - the `torch.rand` stream."""
    return baseval(pricer, greeks=True, recompute=recompute)


def cva_gradient(pricer, recompute):
    """(cva, gradient) off the 256-scenario run - the Sobol stream."""
    cva, _, gradient, _ = cmc(pricer, gradient=True, recompute=recompute)
    return cva, gradient


@pytest.mark.parametrize('pricer', PRICERS)
@pytest.mark.parametrize('mutant', [rc.DesyncedStreams, rc.StaleInputs])
@pytest.mark.parametrize('run,stream', [(base_gradient, 'torch.rand'), (cva_gradient, 'Sobol')])
def test_a_mutated_node_fails_the_gradient_gate(pricer, mutant, run, stream, monkeypatch):
    """Bit-identity passes trivially against a node that quietly reuses the forward's own graph or
    never rewinds anything, so the counter is desynchronised by one draw and the replay is fed
    inputs a basis point off, and each must break the gradient it is supposed to break, on BOTH
    streams and BOTH pricers. The mutations are imported rather than restated - one definition of
    each defect, whichever adopter is under it.

    `StaleInputs` perturbs theta[0], which is why every adopter puts its spot strip there.

    Scored on the gradient alone: both leave the forward pass untouched, so the reported value
    agrees in every digit under each - which is the point, and the reason a price gate over this
    subsystem is worth nothing."""
    value_off, grad_off = run(pricer, 'No')
    monkeypatch.setattr(pricing, 'InnerMCRecompute', mutant)
    value_on, grad_on = run(pricer, 'Yes')
    assert value_off == value_on, 'the mutation moved the VALUE - it is not a backward-only defect'
    assert not np.array_equal(grad_off, grad_on), (
        '{} on {} over the {} stream reproduced the taped gradient exactly, so the gate it is '
        'meant to fail measures nothing:\n{}'.format(mutant.__name__, pricer, stream, grad_off))


# ------------------------------------------- where the gap lives, which is where the two differ

def test_the_autocall_correction_is_exactly_what_the_gap_cotangent_carries(monkeypatch):
    """The autocall's trigger gap is a node OUTPUT, so the correction reaches the simulation as
    that output's cotangent and NOTHING ELSE carries it. Both halves are gated, and the second is
    what makes the mutation attributable rather than merely different: dropping every cotangent but
    the marks' is BIT-IDENTICAL to removing the correction at the objective. Measured on this
    fixture, max |delta| 5.512191e-06 on a gradient of 2.147838e-03 - 0.26%.

    That equality also says the settled-cashflow cotangent contributes nothing here, which is true
    of an UNCOLLATERALISED set: cash reaches an exposure through `C_ts_te`, and there is no
    collateral chain in this portfolio to read it."""
    _, _, corrected, _ = cmc('autocall', gradient=True, recompute='Yes')
    with monkeypatch.context() as patch:
        patch.setattr(pricing, 'InnerMCRecompute', rc.NoBoundaryInjection)
        _, _, dropped, _ = cmc('autocall', gradient=True, recompute='Yes')
    with monkeypatch.context() as patch:
        _, _, suppressed, _ = suppressed_correction(
            lambda: cmc('autocall', gradient=True, recompute='Yes'), patch)
    moved = np.abs(corrected - dropped)
    assert moved.max() > 0.0, 'the boundary correction reaches nothing through the node'
    assert np.array_equal(dropped, suppressed), (
        'dropping the gap cotangent is not the same as removing the correction, so the injection '
        'mutant moves something else as well and attributes nothing:\nmax |d| {:.6g}'.format(
            float(np.abs(dropped - suppressed).max())))
    print('\nautocall correction through the node: max |delta| = {:.6g} on a gradient of '
          '{:.6g}'.format(moved.max(), np.abs(corrected).max()))


def test_the_barrier_correction_is_live_and_rides_no_cotangent(monkeypatch):
    """The barrier's latch is decided on the OUTER scenario spot, so its registration keeps its own
    graph and the node has nothing to do with it. Asserting that costs two readings and they have
    to disagree with each other, or the gate is vacuous in one direction or the other:

      the injection mutant is a NO-OP - dropping every cotangent but the marks' reproduces the
      corrected gradient BIT for BIT, which is what says no boundary term rides this node;
      suppressing the correction at the objective MOVES it - measured max |delta| 2.194252e-03 on a
      gradient of 7.259363e-01, 0.30% - which is what says there is a live correction to have
      missed.

    Written this way round on purpose. A gate that only checked the no-op would pass just as well
    against a barrier whose registration had silently stopped firing."""
    _, _, corrected, _ = cmc('barrier', gradient=True, recompute='Yes')
    with monkeypatch.context() as patch:
        patch.setattr(pricing, 'InnerMCRecompute', rc.NoBoundaryInjection)
        _, _, dropped, _ = cmc('barrier', gradient=True, recompute='Yes')
    with monkeypatch.context() as patch:
        _, _, suppressed, _ = suppressed_correction(
            lambda: cmc('barrier', gradient=True, recompute='Yes'), patch)
    assert np.array_equal(corrected, dropped), (
        'a cotangent of the barrier node carries part of the boundary correction, so the latch is '
        'no longer registered from outer state alone; max |d| {:.6g}'.format(
            float(np.abs(corrected - dropped).max())))
    moved = np.abs(corrected - suppressed)
    assert moved.max() > 0.0, (
        'the barrier latch contributes nothing to this gradient, so the no-op above is vacuous - '
        'the registration is not firing on this fixture')
    print('\nbarrier correction beside the node: max |delta| = {:.6g} on a gradient of '
          '{:.6g}'.format(moved.max(), np.abs(corrected).max()))


# ---------------------------------- (f) the AVERAGING branch, which nothing above reaches at all

def averaging_run(monkeypatch, **kwargs):
    """`cmc('averaging', ...)` plus what the pricer DID, which is what makes the branch gates below
    non-vacuous rather than a second spelling of the fixture: the `no_averaging` flag it was
    dispatched with, each block's `(row count, settle_rows)`, and the mtm-grid DAY every settlement
    was booked on.

    Three spies and ONE run. `cash_settle` and `pv_MC_AutoCallSwap` are looked up on the `pricing`
    module at every call, so `monkeypatch` reaches both; the node is subclassed the way the mutants
    above are and calls the BASE `run`, so what it observes is what would have happened.
    """
    flags, blocks, booked, grids = [], [], [], []
    pricer, settle, node = pricing.pv_MC_AutoCallSwap, pricing.cash_settle, pricing.InnerMCRecompute

    class SettleRowSpy(node):
        @classmethod
        def run(cls, shared, simulate, *theta):
            outputs = node.run(shared, simulate, *theta)
            blocks.append((int(outputs[0].shape[0]), list(outputs[2])))
            return outputs

    def spy_pricer(shared, time_grid, deal_data, *args):
        flags.append(deal_data.Factor_dep['no_averaging'])
        grids.append(time_grid.mtm_time_grid)
        return pricer(shared, time_grid, deal_data, *args)

    monkeypatch.setattr(pricing, 'pv_MC_AutoCallSwap', spy_pricer)
    monkeypatch.setattr(pricing, 'InnerMCRecompute', SettleRowSpy)
    monkeypatch.setattr(pricing, 'cash_settle', lambda shared, currency, index, value: (
        booked.append(index), settle(shared, currency, index, value))[1])
    result = cmc('averaging', **kwargs)
    return result, flags, blocks, [int(grids[0][i]) for i in booked]


def test_the_averaging_branch_is_reached_and_settles_off_its_returned_rows(monkeypatch):
    """THE GATE ON THE GATES. Every statement below is a bit-identity, and a bit-identity holds
    trivially over code neither side runs - so this says the fixture reaches the branch, that the
    branch's settlement is PLACED by the `settle_rows` the simulation returns, and that the
    placement is observable at all.

    The third is the one a fixture silently loses. `t_Cashflows` is pre-allocated per DECLARED
    payment date (`reset_cashflows` over the currency map) and `cash_settle` drops anything booked
    elsewhere, so mis-placing a settlement does not move a number - it deletes one. And the
    placement is only observable where a settling row is not its block's first, which is what the
    coupons two days off the reporting grid buy: measured 5 blocks, four of them `(2, [1])`.

    Mutation-verified against `settle_rows.append(i) -> .append(0)`, the edit's own by-product
    index: blocks read `(2, [0])`, the days become the reporting grid's 92/183/273/365 instead of
    the coupons' 94/185/275/367, and every reported cashflow collapses to zero - `cash_settle` was
    handed an index `reset_cashflows` never allocated. Each statement here is false under it (the
    first is what pytest stops on) and so is the non-zero clause of the exposure gate below: 3 of
    this file's 37 turn red. Every bit-identity COMPARISON stays green, because a mutation in the
    source moves both switch settings alike - which is why this gate is written as an absolute
    statement about the fixture and not as a comparison."""
    (_, _, _, cash), flags, blocks, days = averaging_run(monkeypatch)
    assert flags and not any(flags), (
        'the averaging fixture priced through the no_averaging branch, so section (f) is a slower '
        'copy of section (a): {}'.format(flags))
    settling = [b for b in blocks if b[1]]
    assert settling and all(b == (2, [1]) for b in settling), (
        'a settlement is on its block-local row 0, where `settle_rows` is indistinguishable from a '
        'constant and this file cannot see the difference: {}'.format(blocks))
    assert days == [(d - bb.BASE).days for d in AVERAGING_COUPONS], (
        'a coupon settled on a day that is not its own: {} against {}'.format(
            days, [(d - bb.BASE).days for d in AVERAGING_COUPONS]))
    assert all(np.abs(x).max() > 0.0 for x in cash), (
        'every settled cashflow is zero - `cash_settle` dropped them, which is what a mis-placed '
        'settlement looks like from the outside')


def test_the_averaging_base_price_is_bit_identical_with_the_node_on():
    """The `torch.rand` half of the stream contract on this branch, whose draw is a full-path
    `torch.randn`/`quasi_rng` block rather than the one-step-survival branch's uniforms."""
    off, _ = baseval('averaging')
    on, _ = baseval('averaging', recompute='Yes')
    assert off != 0.0, 'the averaging fixture prices at zero and gates nothing'
    assert off == on, f'the averaging price moved with the node on: {off!r} -> {on!r}'


@pytest.mark.parametrize('collateralised', [False, True])
def test_the_averaging_exposure_and_cashflows_are_bit_identical_with_the_node_on(collateralised):
    """Value, exposure and every settled cashflow, uncollateralised and collateralised - the
    branch's settlement is now a returned output indexed after the loop rather than a `cash_settle`
    performed inside it, and this is what says the two spellings agree."""
    cva_off, mtm_off, _, cash_off = cmc('averaging', collateralised=collateralised)
    cva_on, mtm_on, _, cash_on = cmc('averaging', recompute='Yes', collateralised=collateralised)
    assert np.array_equal(mtm_off, mtm_on), 'the averaging exposure moved with the node on'
    assert cva_off == cva_on, f'the averaging cva moved: {cva_off!r} -> {cva_on!r}'
    assert any(np.abs(x).max() > 0.0 for x in cash_off), 'no cashflow settled - this reads nothing'
    assert all(np.array_equal(a, b) for a, b in zip(cash_off, cash_on)), (
        'a settled cashflow moved with the node on')


@pytest.mark.parametrize('run,stream', [(base_gradient, 'torch.rand'), (cva_gradient, 'Sobol')])
def test_the_averaging_gradient_is_bit_identical_with_the_node_on(run, stream):
    """The WHOLE vector, on both streams. This branch has no boundary registration of its own - the
    trigger is smoothed by `smooth_heaviside_up` inside `sim_autocall`, so there is nothing to
    register and `event_rows` comes back empty - which makes the ordinary AAD path the entire
    statement here.

    Mutation-verified against the branch's OWN edited line, the row loop reading the closure's
    `spot_block` instead of the `spot_prices` theta: a no-op with the switch off (the same object is
    passed in) and a stale read under it. Off `torch.rand` it is exactly the defect the hoist exists
    to stop, and it is SILENT - the `EquityPrice.EQ` entry, 1.245414e-03, simply vanishes from the
    reported vector, 7 entries down to 6, because autograd has no gradient to give for a theta the
    replay never read. Off Sobol it is loud instead: the closure holds the LAST block's strip, so
    the replay's output stops matching the forward's shape and autograd raises. Every graph-carrying
    theta in this branch is block-shaped, which is what makes a stale closure read here hard to keep
    quiet - and the base stream is where it manages it."""
    value_off, grad_off = run('averaging', 'No')
    value_on, grad_on = run('averaging', 'Yes')
    assert np.abs(grad_off).max() > 0.0, f'{stream}: no gradient was reported'
    assert value_off == value_on
    assert np.array_equal(grad_off, grad_on), (
        'the recomputed averaging gradient over the {} stream is not the taped one:\n{}\n{}'.format(
            stream, grad_off, grad_on))


def test_the_collateralised_averaging_gradient_is_the_taped_one_to_the_last_bit():
    """ONE STEP again, and this time NEITHER SIDE OF THE REFACTOR is what spent it.

    Attributed three ways - HEAD, refactored-taped, node - on the same discipline as the autocall's
    collateralised gate above. HEAD (`cash_settle` performed inside the loop) and the refactored OFF
    path agree bit for bit and each reproduces itself: this branch stacks the very tensor it marked
    (`pv`, one node with two consumers) instead of a second `.mean` of the quantity it marked, which
    is what the one-step-survival branch does and what earns the 1 ulp there. The node agrees with
    both, but not reproducibly - measured over 6 repeats, 1 float64 step on entry 13
    (`EquityPriceVol.EQ` 1.2/2.0, the smallest of the fourteen) on 2 of them and 0 on the other 4,
    while the taped side moved 0 on all 6, uncollateralised readings moved 0 on all 6, and HEAD
    moved 0. The step is therefore the replayed backward disagreeing with ITSELF under collateral -
    nondeterminism in a reduction - and the defect it has to stay separable from is 8.7%. One step
    because that is the measurement; the gates above are `array_equal` because there it is zero."""
    cva_off, mtm_off, grad_off, cash_off = cmc('averaging', gradient=True, collateralised=True)
    cva_on, mtm_on, grad_on, cash_on = cmc('averaging', gradient=True, recompute='Yes',
                                           collateralised=True)
    assert np.array_equal(mtm_off, mtm_on) and cva_off == cva_on, (
        'the collateralised averaging exposure moved with the node on')
    assert all(np.array_equal(a, b) for a, b in zip(cash_off, cash_on))
    assert np.abs(grad_off).max() > 0.0, 'no gradient was reported'
    steps = _ulps(grad_off, grad_on)
    assert steps.max() <= 1, (
        'the collateralised averaging gradient moved by {} float64 steps (max |d| {:.6g} on '
        '{:.6g}) - that is not a reduction disagreeing with itself, it is a channel the node '
        'lost:\n{}\n{}'.format(int(steps.max()), float(np.abs(grad_off - grad_on).max()),
                               float(np.abs(grad_off).max()), grad_off, grad_on))


@pytest.mark.parametrize('mutant', [rc.DesyncedStreams, rc.StaleInputs])
@pytest.mark.parametrize('run,stream', [(base_gradient, 'torch.rand'), (cva_gradient, 'Sobol')])
def test_a_mutated_node_fails_the_averaging_gradient_gate(mutant, run, stream, monkeypatch):
    """The same two defects as section (e), against the branch that draws a whole path rather than
    a survival uniform - a bit-identity gate is worth what it can fail, and this branch's draw and
    theta surface are its own. `StaleInputs` moves theta[0], which here is the spot strip the
    averaged path is grown from."""
    value_off, grad_off = run('averaging', 'No')
    monkeypatch.setattr(pricing, 'InnerMCRecompute', mutant)
    value_on, grad_on = run('averaging', 'Yes')
    assert value_off == value_on, 'the mutation moved the VALUE - it is not a backward-only defect'
    assert not np.array_equal(grad_off, grad_on), (
        '{} over the {} stream reproduced the taped averaging gradient exactly, so the gate it is '
        'meant to fail measures nothing:\n{}'.format(mutant.__name__, stream, grad_off))
