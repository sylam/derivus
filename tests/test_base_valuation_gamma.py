"""`Greeks: 'All'` - the second-order block, its oracles, and the two things it refuses.

THE BLOCK WAS UNREACHABLE, WHICH IS WHY IT HAD NO COVERAGE: `__init_shared_mem` tests
`params['Greeks'] == 'All'` and the field declared `values=['First', 'No']`, so no panel or
schema-authored job could ask for it. `test_schema_emission.py` now holds the menu against the
engine; this file is the block itself.

WHAT THE ORACLES ARE - closed forms, identities and convergent differences, none of them derivus
asked twice:

  GAMMA      Black-Scholes d2V/dS2 on a vanilla European call. At r = q = 0 the engine's forward IS
             the spot and every discount factor is one, so the reading is asserted in ULPS. Carry
             is then switched on and the difference is a CONVENTION: the equity forward runs to
             expiry plus a two-business-day settlement lag while the discounting runs to expiry.
  VANNA      the cross-column identity, which the diagonal cannot see. The surface is interpolated
             in MONEYNESS, so a spot move moves the interpolation WEIGHTS: the 1.0 nodes carry
             vanna less what the weight shift takes and the 1.2 nodes carry exactly that shift.
             A Hessian dropping the weight derivative still looks plausible on the diagonal.
  ZERO       an FX forward is LINEAR in the spot, so d2V/dFX2 is exactly zero. The mutation that
             matters is the placebo: the same matrix has curve convexity of 1e7 in it.
  CONVEXITY  a swap's d2V/dr2 against a CENTRAL DIFFERENCE of the reported AAD delta. Base
             valuation is deterministic, so the ladder is pure truncation error: 3.0e-6 -> 3.0e-8
             -> 2.9e-10 over h = 1e-3..1e-5, h^2 to three digits.
  LADDER     the same difference on the two MONTE CARLO fixtures, which is the only correctness
             statement where no closed form is. It is owed because SYMMETRY IS NOT ONE:
             `report_hessian` mirrors an upper triangle, so `H == H.T` holds of whatever the AAD
             put there. The HN rung is also the eager-vs-fused equality gate.

TWO REFUSALS, both because the alternative is a number that looks right.

  A BOUNDARY CORRECTION is `(gap - gap.detach())` times a DETACHED coefficient, so differentiating
  it twice silently drops the density-derivative term the correction exists to supply while
  keeping the smooth part. The refusal names the deals and points at bumping the ADJOINT under
  common random numbers; it is `utils.SecondOrderRefused` so a caller can fall back to `'First'`.
  `Recompute_Inner_MC` is refused for its own reason and gated in `test_recompute_equity_pricers`.

AND ONE REPAIR: `utils.hn_log_substep` is `torch.compile`d and AOTAutograd's compiled backward has
no double backward, so every Heston-Nandi valuation with an unmonitored sub-step died on
`Greeks: 'All'`. The eager spelling is kept beside the fused one and `shared.gamma` picks; the
LADDER is what says the two spellings are one function. Both those gates reset `torch._dynamo`
afterwards, which is load-bearing rather than tidy - see the seam gate's docstring.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.stats import norm

from derivus import run_baseval, utils
from derivus.config import Config
from derivus.instruments import construct_instrument
from derivus.schema import mapping
import rates_world as rw
import test_recompute_equity_pricers as re_
from conftest import needs_hn_fused

DTYPE = torch.float64
#: The device `run_baseval` hands a job, and therefore the one the seam gate must walk: its oracle
#: is the compiled backward's OWN refusal, and inductor needs a backend for the tensors' device
#: before it can raise anything at all.
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BASE = pd.Timestamp('2024-06-28')
SPOT, STRIKE, VOL = 100.0, 100.0, 0.25
#: One year to the day on ACT_365 curves, so T is exactly 1.0 and the closed form needs no day
#: count of its own.
EXPIRY = BASE + pd.Timedelta(days=365)
#: 1.0 is where an ATM option sits; 1.2 is the node the interpolation weight moves TOWARDS as the
#: spot rises, which is what the vanna identity reads.
MONEYNESS = (0.8, 1.0, 1.2)

EQ_OPTION = {
    'Object': 'EquityOptionDeal', 'Reference': 'EQOPT', 'Currency': 'USD',
    'Payoff_Currency': 'USD', 'Equity': 'EQ', 'Dividends': 'EQ', 'Discount_Rate': 'USD',
    'Equity_Volatility': 'EQ', 'Buy_Sell': 'Buy', 'Option_Type': 'Call',
    'Strike_Price': STRIKE, 'Expiry_Date': EXPIRY, 'Units': 1.0, 'Settlement_Style': 'Cash',
    'Option_On_Forward': 'No', 'Option_Style': 'European', 'Payoff_Type': 'Standard'}

FX_FORWARD = {
    'Object': 'FXForwardDeal', 'Reference': 'FWD1', 'Buy_Currency': 'EUR', 'Sell_Currency': 'USD',
    'Buy_Amount': 10_000_000.0, 'Sell_Amount': 11_000_000.0, 'Buy_Discount_Rate': 'EUR',
    'Sell_Discount_Rate': 'USD', 'Settlement_Date': BASE + pd.Timedelta(days=730),
    'Discount_Rate': 'USD'}

#: Five knots so the Hessian has an off-diagonal to be wrong on, and a 7Y swap so the 5Y and 10Y
#: buckets both carry weight.
SWAP_KNOTS = [0.25, 1.0, 3.0, 5.0, 10.0]
SWAP_LEVELS = np.array([0.030, 0.032, 0.035, 0.037, 0.040])


def equity_cfg(r=0.0, q=0.0, spot=SPOT, vol=VOL):
    """The vanilla call, alone, in a flat world. `r = q = 0` is the exact case: the engine's equity
    forward is the spot and every discount factor one, so nothing but the payoff formula separates
    it from the closed form."""
    c = Config()
    c.params['System Parameters']['Base_Currency'] = 'USD'
    c.params['System Parameters']['Base_Date'] = BASE
    c.params['Price Factors'] = {
        'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Priority': 1,
                       'Spot': 1.0},
        'InterestRate.USD': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                             'Curve': utils.Curve([], [[0.0, r], [5.0, r]])},
        'EquityPrice.EQ': {'Spot': spot, 'Currency': 'USD', 'Interest_Rate': 'USD', 'Issuer': '',
                           'Respect_Default': 'No', 'Jump_Level': 0.0},
        'DividendRate.EQ': {'Currency': 'USD', 'Floor': None,
                            'Curve': utils.Curve([], [[0.0, q], [5.0, q]])},
        'EquityPriceVol.EQ': {
            'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness',
            'Surface': utils.Curve([], [[m, t, vol] for m in MONEYNESS for t in (0.02, 2.0)])}}
    c.deals = {'Attributes': {'Reference': 'test', 'Tag_Titles': ''},
               'Deals': {'Children': [{'Instrument': construct_instrument(EQ_OPTION, {})}]},
               'Calculation': {'Base_Date': BASE, 'Currency': 'USD'}}
    return c


def fx_forward_cfg():
    """The forward, alone. Two currencies, two curves, and a spot the value is LINEAR in."""
    c = Config()
    c.params['System Parameters']['Base_Currency'] = 'USD'
    c.params['System Parameters']['Base_Date'] = BASE
    c.params['Price Factors'] = {
        'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Priority': 1,
                       'Spot': 1.0},
        'FxRate.EUR': {'Domestic_Currency': 'USD', 'Interest_Rate': 'EUR', 'Priority': 2,
                       'Spot': 1.1},
        'InterestRate.USD': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                             'Curve': utils.Curve([], [[0.0, 0.03], [5.0, 0.035]])},
        'InterestRate.EUR': {'Currency': 'EUR', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                             'Curve': utils.Curve([], [[0.0, 0.02], [5.0, 0.025]])}}
    c.deals = {'Attributes': {'Reference': 'test', 'Tag_Titles': ''},
               'Deals': {'Children': [{'Instrument': construct_instrument(FX_FORWARD, {})}]},
               'Calculation': {'Base_Date': BASE, 'Currency': 'USD'}}
    return c


def swap_cfg(levels=SWAP_LEVELS):
    """A 7Y par-style swap off one curve, whose knots the ladder bumps."""
    c = Config(base_currency='USD')
    c.params['System Parameters']['Base_Date'] = rw.BASE
    c.params['Price Factors'] = rw.market(
        'USD', {'USD': (SWAP_KNOTS, list(levels))}, 'USD', day_count='ACT_365')
    c.deals = {'Attributes': {'Reference': 'test', 'Tag_Titles': ''},
               'Deals': {'Children': [{'Instrument': construct_instrument(
                   rw.par_swap('SW1', 'USD', 'USD', 'USD', 7, 3.5, day_count='ACT_365'), {})}]},
               'Calculation': {'Base_Date': rw.BASE, 'Currency': 'USD'}}
    return c


def gbm_autocall_cfg(spot=None):
    """`test_recompute_equity_pricers`' autocall in its own world, at a bumped spot when asked."""
    config = re_._cfg('autocall')
    if spot is not None:
        config.params['Price Factors']['EquityPrice.EQ']['Spot'] = spot
    return config


def hn_autocall_cfg(spot=None):
    """The Heston-Nandi autocall of `test_hn_oss_pricers` - one coupon 30 days out, so 29
    unmonitored sub-steps are walked to reach it - at a bumped spot when asked."""
    import test_hn_oss_pricers as hn
    config, _ = hn._autocall_cfg([30], [1.05], [0.05], 30, hn_params=hn.STRONG)
    if spot is not None:
        config.params['Price Factors']['EquityPrice.EQ']['Spot'] = spot
    return config


#: (the fixture, its inner paths, the h-ladder with what each rung is allowed). Both are autocalls
#: at BASE valuation, which is what lets them past the refusal below - one reporting row means no
#: coupon is OBSERVED - and they are the only Monte Carlo fixtures here reporting a second-order
#: block at all. 4096 is `test_recompute_equity_pricers`' OWN count, so this leaves the compiler no
#: new shape. The last rung's tolerance is an ULP ENVELOPE and not a truncation bound.
MONTE_CARLO_LADDER = {
    'gbm': (gbm_autocall_cfg, 1 << 12, [(1e-2, 1e-6), (1e-3, 1e-8), (1e-4, 2e-10)]),
    'heston-nandi': (hn_autocall_cfg, 1 << 12, [(1e-2, 2e-3), (1e-3, 2e-5), (1e-4, 2e-7)])}


def valued(config, greeks='All', simulations=1):
    """(portfolio value, first-order frame, second-order frame); the second is None below `'All'`.
    The first-order frame carries the factor LEVEL in `Value` and the gradient in the column named
    after the reporting reference, which is why that column is picked by elimination. `simulations`
    is 1 for every analytic fixture and is what the Monte Carlo ladder raises.
    """
    _, out = run_baseval(config, prec=DTYPE, overrides={
        'Greeks': greeks, 'Random_Seed': 1, 'MCMC_Simulations': simulations})
    rows = out['Results']['mtm']
    value = float(rows[rows['Parent'] == 'root']['Value'].sum())
    frame = out['Results'].get('Greeks_First')
    return value, frame, out['Results'].get('Greeks_Second')


def first_order(frame):
    """The gradient column of a `Greeks_First` frame, as a numpy vector in reported order."""
    column, = [c for c in frame.columns if c != 'Value']
    return frame[column].values.astype(np.float64)


def reported_delta(config, simulations):
    """The reported dV/dS of a `Greeks: 'First'` run - the quantity the ladder differences."""
    return float(valued(config, 'First', simulations)[1].loc[EQUITY].iloc[-1])


def entry(second, row, column=None):
    """One (row, column) of the second-order frame, addressed by the FIRST-order frame's row labels:
    the two share an index, and the columns carry the reporting reference as an outer level."""
    column = row if column is None else column
    return float(second.loc[row][('root',) + tuple(column)])


def black_scholes_call(S, K, T, r, q, sigma):
    """(price, delta, gamma, vanna) for a European call with continuous carry, hand-authored from
    the textbook forms. `vanna` is d2V/dS dsigma, which the cross-column identity reads."""
    root = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / root
    d2 = d1 - root
    return (S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2),
            np.exp(-q * T) * norm.cdf(d1),
            np.exp(-q * T) * norm.pdf(d1) / (S * root),
            -np.exp(-q * T) * norm.pdf(d1) * d2 / sigma)


EQUITY = ('EquityPrice.EQ', 0.0, 0.0, 0.0)
FX = ('FxRate.EUR', 0.0, 0.0, 0.0)


# ---------------------------------------------------------------- the block is reachable at all

def test_the_second_order_block_is_reachable_from_the_schema():
    """The declaration, which is the whole of what was missing: `'All'` on the menu, `'No'` still
    the default - a second derivative on by omission would multiply every job's cost."""
    declared = mapping['Calculation']['types']['BaseValuation']['Greeks']
    assert declared['values'] == ['All', 'First', 'No'], (
        'the second-order block is not on the menu the engine acts on')
    assert declared['value'] == 'No', 'second derivatives have become the default'


def test_the_second_order_block_is_reported_under_a_stable_key_beside_the_first():
    """The shape, so a consumer can be written against it: `Greeks_Second` appears iff `'All'` was
    asked for, beside `Greeks_First`, square, symmetric, labelled on both axes by
    (Rate, Tenor, Tenor2, Tenor3) with the reporting reference on top of the columns.

    ITS ROWS ARE NOT THE FIRST-ORDER BLOCK'S. Both frames report the SUPPORT of what they hold, and
    a factor can have no first derivative and a real second one: the moneyness-1.2 vol nodes carry
    zero vega because the interpolation gives them no weight at the money, while d2V/dS dsigma
    there is 0.98 because a spot move is what gives them weight. Joining on the index drops those.
    """
    for greeks in ('No', 'First'):
        _, _, absent = valued(equity_cfg(), greeks=greeks)
        assert absent is None, f"Greeks: '{greeks}' reported a second-order block"

    _, first, second = valued(equity_cfg())
    assert second.shape[0] == second.shape[1], 'the reported Hessian is not square'
    assert np.array_equal(second.values, second.values.T), (
        'the reported Hessian is not symmetric - the upper-triangle assembly is wrong')
    assert [c[1:] for c in second.columns] == list(second.index), (
        'the columns are not the rows with the reporting reference on top')
    assert all(len(row) == first.index.nlevels for row in second.index), (
        'the second-order rows are not (Rate, Tenor, Tenor2, Tenor3)')

    second_only = sorted(set(second.index) - set(first.index))
    assert second_only == [('EquityPriceVol.EQ', 1.2, 0.02, 0.0),
                           ('EquityPriceVol.EQ', 1.2, 2.0, 0.0)], (
        f'the second-order-only rows are not the ones the surface explains: {second_only}')
    for row in second_only:
        assert row not in first.index and abs(entry(second, EQUITY, row)) > 0.9


# ---------------------------------------------------------------- the closed forms

def test_black_scholes_gamma_is_exact_in_the_zero_carry_world():
    """d2V/dS2 against the closed form where the engine and the formula are the SAME arithmetic, so
    the gate counts representable steps rather than choosing a tolerance. Four and not equality:
    this fixture lands on 0 steps for all three readings, while the same oracle at
    S=87.5, K=95, vol=0.31, T=2y lands on 4, 2 and 1. Price and delta are asserted beside gamma,
    because a gamma matching off a value that did not would mean the two agreed by accident.
    """
    value, first, second = valued(equity_cfg(r=0.0, q=0.0))
    price, delta, gamma, _ = black_scholes_call(SPOT, STRIKE, 1.0, 0.0, 0.0, VOL)

    for label, engine, closed in (('price', value, price),
                                  ('delta', float(first.loc[EQUITY].iloc[-1]), delta),
                                  ('gamma', entry(second, EQUITY), gamma)):
        steps = int(re_._ulps(np.array([engine]), np.array([closed]))[0])
        assert steps <= 4, (
            f'{label} {engine!r} is {steps} float64 steps off the closed form {closed!r}')


def test_black_scholes_gamma_survives_carry_at_a_measured_tolerance():
    """The same reading at r = 3%, q = 1%, where what separates engine and formula is a CONVENTION
    and not a numerical error. Not curve interpolation - the curve is FLAT:
    `instruments.option_date_info` runs the equity forward to `Forward_Settlement`, expiry plus a
    two-business-day settlement lag, while the option discounts to expiry. Put-call parity puts the
    discount factor at exp(-rT) to the last bit and the carry over T + 3/365; the offset is a DATE,
    so it moves with the expiry and no day count fits it.

    So the difference scales with the CARRY and the bound is chosen against that rather than this
    one point: gamma reads 2.9e-5 here against 1.2e-4 at (5%, 0%) and 1.4e-4 at (0%, 2%), so 3e-4
    clears its neighbours where the 1e-4 it replaces fitted this point with no headroom.
    """
    value, first, second = valued(equity_cfg(r=0.03, q=0.01))
    price, delta, gamma, _ = black_scholes_call(SPOT, STRIKE, 1.0, 0.03, 0.01, VOL)

    assert abs(value / price - 1.0) < 1e-3, 'the price moved off the closed form'
    assert abs(float(first.loc[EQUITY].iloc[-1]) / delta - 1.0) < 1e-3, 'the delta moved'
    assert abs(entry(second, EQUITY) / gamma - 1.0) < 3e-4, (
        'gamma {} is further from the closed form {} than the measured 2.9e-5'.format(
            entry(second, EQUITY), gamma))


def test_the_spot_vol_cross_terms_sum_to_the_closed_form_vanna():
    """The oracle the diagonal cannot supply, and the one saying the surface's own derivative is in
    the Hessian. The vol is interpolated in MONEYNESS, so a spot move both moves the option along a
    fixed vol (vanna, landing on the node the weight sits on) and moves the interpolation WEIGHT
    (landing on the node it moves towards). The surface is flat, so the second changes no VALUE at
    all - but it is a real second derivative, and the two halves are large and opposite: -1.7812
    and +1.9792, summing to the closed-form vanna 0.19795. A Hessian that dropped or double-counted
    the weight derivative fails here while every diagonal entry stays right.
    """
    _, first, second = valued(equity_cfg(r=0.0, q=0.0))
    _, _, _, vanna = black_scholes_call(SPOT, STRIKE, 1.0, 0.0, 0.0, VOL)

    vol_rows = [r for r in second.index if r[0] == 'EquityPriceVol.EQ']
    total = sum(entry(second, EQUITY, row) for row in vol_rows)
    assert abs(total / vanna - 1.0) < 1e-6, (
        f'the spot-vol column sums to {total}, not the closed-form vanna {vanna}')

    at_atm = sum(entry(second, EQUITY, r) for r in vol_rows if r[1] == 1.0)
    assert at_atm < 0.0 and total - at_atm > 0.0, (
        'the two halves are not opposite - the interpolation weight derivative is missing: '
        f'{at_atm} and {total - at_atm}')
    assert abs(at_atm) > 5.0 * abs(vanna), (
        'the halves are not large against their sum, so this gate is not testing cancellation: '
        f'{at_atm} against {vanna} (measured 9.0x)')


def test_an_fx_forward_has_exactly_zero_gamma():
    """A linear product's second derivative in its own underlying is EXACTLY zero - the forward is
    `Buy * FX * D_buy - Sell * D_sell` with the spot appearing once. The placebo it has to survive
    is the empty matrix, so the same reading must carry real curve convexity: an engine reporting
    nothing at all would pass the zero and fail the second half.
    """
    _, _, second = valued(fx_forward_cfg())
    assert entry(second, FX) == 0.0, (
        f'an FX forward reported gamma {entry(second, FX)!r} in its own spot')

    curve = [r for r in second.index if r[0] == 'InterestRate.EUR']
    convexity = max(abs(entry(second, r)) for r in curve)
    assert convexity > 1e6, (
        f'nothing else was reported either ({convexity}) - the zero above is vacuous')
    assert abs(entry(second, FX, curve[0])) > 1e6, (
        'the spot ROW is empty too, so the zero is a dropped row rather than a linear product')


@pytest.mark.parametrize('h,tolerance', [(1e-3, 4e-6), (1e-4, 4e-8), (1e-5, 1e-9)])
def test_the_swap_curve_convexity_is_the_derivative_of_the_reported_delta(h, tolerance):
    """d2V/dr2 against a CENTRAL DIFFERENCE of the AAD delta, on a ladder that has to converge. No
    common random numbers are needed - base valuation is one date and one scenario, and a linear
    rates deal draws nothing - so what is left is pure truncation error: 3.0e-6, 3.0e-8, 2.9e-10,
    h^2 to three digits. h = 1e-6 is deliberately absent, reading 2.2e-10 off the ladder and into
    the difference's own cancellation.

    The WHOLE matrix is compared: the off-diagonal is where a mis-assembled upper triangle shows.
    """
    _, _, second = valued(swap_cfg())
    columns = []
    for j in range(len(SWAP_KNOTS)):
        up, down = SWAP_LEVELS.copy(), SWAP_LEVELS.copy()
        up[j] += h
        down[j] -= h
        columns.append((first_order(valued(swap_cfg(up), greeks='First')[1]) -
                        first_order(valued(swap_cfg(down), greeks='First')[1])) / (2.0 * h))

    finite_difference = np.array(columns).T
    scale = np.abs(second.values).max()
    relative = np.abs(finite_difference - second.values).max() / scale
    assert scale > 1e6, 'no convexity was reported - the ladder has nothing to converge to'
    assert relative < tolerance, (
        f'h={h:g}: the reported convexity is {relative:.3e} off the differenced delta')


# ---------------------------------------------------------------- what it refuses

def test_a_deal_that_registered_a_boundary_correction_is_refused_by_name():
    """The refusal, and that it says WHICH deals. A second derivative through
    `(gap - gap.detach())` times a detached coefficient keeps the smooth part and silently loses
    the density-derivative term, so what comes back is a plausible gamma with a term missing. The
    message names the deals - a portfolio's author cannot otherwise tell which to take out - and
    points at bumping the adjoint under common random numbers.
    """
    with pytest.raises(Exception) as raised:
        valued(re_._cfg('barrier'))
    message = str(raised.value)
    assert 'BARR1' in message, f'the refusal does not name the deal: {message}'
    assert 'density-derivative' in message and 'common random numbers' in message, (
        f'the refusal does not say why, or what to do instead: {message}')

    # the same portfolio is fine at first order, so the refusal is about the SECOND derivative and
    # not about the deal being unpriceable
    _, first, second = valued(re_._cfg('barrier'), greeks='First')
    assert second is None and np.abs(first_order(first)).max() > 0.0


def test_hedge_monte_carlo_refuses_the_second_order_block():
    """`Greeks` is not a HedgeMonteCarlo parameter, so `'All'` there is a job asking for something it
    will not get - and silently ignoring it is the failure worth raising over: the run succeeds,
    the report has no second-order block, and nothing says the request was dropped.

    ASKED THE WAY A JOB ASKS IT: the key rides in the file's `Calculation` block, which
    `run_hedgemontecarlo` folds into its parameters, so the finding is that the STORE can carry
    `'All'` to a calculation that cannot honour it. Set in memory; the template file is untouched.
    """
    import derivus as dv
    context = dv.Context()
    context.load_json(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures',
                                   'policy_test_simulate_only.json'))
    context.current_cfg.deals['Calculation']['Greeks'] = 'All'
    with pytest.raises(Exception, match=r"Greeks: 'All' is not a HedgeMonteCarlo parameter"):
        context.run_job()


# ---------------------------------------------------------------- the repair the audit found

@needs_hn_fused
def test_the_heston_nandi_sub_step_is_twice_differentiable():
    """`utils.hn_log_substep` is `torch.compile`d and AOTAutograd's compiled backward RAISES on
    double backward, so every HN valuation whose fixings are more than a day apart died on
    `Greeks: 'All'` - invisible because nothing could ask.

    Gated at the SEAM rather than through a pricer, `shared.gamma` being what picks the eager
    spelling. Both directions are asserted: the fused build must still be what an ordinary run
    takes, or the fix is a silent 5.9x on every Heston-Nandi job.

    IT WALKS THE DEVICE A JOB WALKS. With no device on the tensors the fused half compiled for the
    CPU inductor backend whatever the box was, and on a CUDA box with no host C++ compiler that
    backend cannot build - so what came back was `InductorError: cl is not found` rather than the
    refusal being gated, and the gate could not tell one raise from the other. `needs_hn_fused` is
    this gate's own precondition: it asks for exactly the backend the tensors need before the
    compiled backward exists to refuse.

    `torch._dynamo.reset()` afterwards is NOT tidiness. The compiler cache is a process global keyed
    on traced shapes, and this file feeds it a shape no pricer uses, a compile that FAILS, and both
    spellings at a third shape. Without the resets `test_recompute_equity_pricers`' HN price moves
    in its LAST BIT between the taped and recomputed passes, because the two no longer get the same
    generated kernel - an order effect only the whole suite sees.
    """
    class Shared:
        gamma = True
        simulation_batch = 4
        one = torch.ones([1, 1], dtype=DTYPE, device=DEVICE)

    def second_derivative(gamma):
        shared = Shared()
        shared.gamma = gamma
        omega = torch.tensor(1e-6, dtype=DTYPE, device=DEVICE, requires_grad=True)
        params = (omega, torch.tensor(0.1, dtype=DTYPE, device=DEVICE),
                  torch.tensor(0.8, dtype=DTYPE, device=DEVICE),
                  torch.tensor(2.0, dtype=DTYPE, device=DEVICE))
        spot = torch.full([4, 2], 100.0, dtype=DTYPE, device=DEVICE)
        h = torch.full([4, 2], 4e-5, dtype=DTYPE, device=DEVICE)
        torch.manual_seed(1)
        walked, _ = utils.hn_unmonitored_substeps(
            spot, h, torch.zeros([4, 2], dtype=DTYPE, device=DEVICE), 3, params, shared, 2,
            antithetic=False)
        grad, = torch.autograd.grad(walked.sum(), omega, create_graph=True)
        return float(torch.autograd.grad(grad, omega)[0])

    try:
        assert second_derivative(gamma=True) != 0.0, 'the eager sub-step reported no curvature'
        with pytest.raises(RuntimeError, match='double backward'):
            second_derivative(gamma=False)
    finally:
        torch._dynamo.reset()


def test_a_heston_nandi_valuation_reports_a_second_order_block():
    """The same repair end to end on the fixture that exposed it - an HN autocall whose single
    coupon is 30 days out, so 29 unmonitored sub-steps are walked to reach it. It registers no
    boundary correction at base valuation, so it gets the block. THAT IS SHAPE, NOT CORRECTNESS:
    symmetry is what `report_hessian` does to whatever it assembled and "non-trivial" is a floor.
    The ladder below measures the numbers."""
    _, first, second = valued(hn_autocall_cfg())
    assert np.abs(second.values).max() > 0.0, 'the HN second-order block came back empty'
    assert np.array_equal(second.values, second.values.T)
    assert {r[0] for r in second.index} & {'EquityPrice.EQ'}, (
        'the spot is not in the reported block: {}'.format(sorted({r[0] for r in second.index})))


# the HN rung differences a `Greeks: 'First'` delta, and THAT walks the compiled sub-step (only the
# gamma takes the eager spelling), so it alone carries the fused precondition rather than dying
# three layers downstream on a collapsed frame
@pytest.mark.parametrize('fixture', [pytest.param(f, marks=needs_hn_fused) if f == 'heston-nandi'
                                     else f for f in sorted(MONTE_CARLO_LADDER)])
def test_a_monte_carlo_gamma_is_the_derivative_of_its_own_reported_delta(fixture):
    """d2V/dS2 against a CENTRAL DIFFERENCE of the reported AAD delta on the two Monte Carlo
    fixtures - the correctness statement the block has nowhere a closed form reaches.

    IT IS OWED BECAUSE SYMMETRY IS NOT A CHECK: `report_hessian` mirrors an upper triangle, so
    `H == H.T` is true of whatever the AAD put there, and that beside "not empty" was every gate
    the Heston-Nandi block had.

    Common random numbers leave the ladder as pure truncation error - one seed, one scenario, and a
    bumped spot rescaling the paths the same draws generate. On CUDA float64:

        gbm            gamma -2.91303e-05,  2.12e-07 -> 2.12e-09 -> 7.61e-11
        heston-nandi   gamma  2.69093e-06,  3.62e-04 -> 3.62e-06 -> 3.44e-08

    h^2 to two digits over the first two rungs. h = 1e-5 turns back UP on the GBM fixture, which is
    the difference's own cancellation. THE LAST RUNG IS AN ULP ENVELOPE: at h = 1e-4 one ulp of
    either delta moves the quotient by 3.7e-11 relative, so the CUDA reading is 2.05 ulps, the CPU
    one 0.60, and the 2e-10 tolerance is 5.4 - two orders inside the rung above, so the ladder
    still has to FALL. A wrong gamma is caught at h = 1e-3.

    Every rung is scored against the gamma from its OWN run and never the number written here: the
    draw stream is per-device, so a CPU box prices different paths and reads 1.3% away.

    THE HN RUNG IS ALSO THE EAGER-AGAINST-FUSED EQUALITY GATE - `shared.gamma` walks the eager
    sub-step for the second derivative while the differenced delta walks the compiled one - which
    is why it carries `needs_hn_fused` and the GBM one does not: without a backend the deal is
    skipped CRITICAL and the mark collapses to a scalar zero three layers downstream.

    BOTH FIXTURES RUN AT `test_recompute_equity_pricers`' OWN 4096 PATHS, which is a requirement:
    a count nothing else uses caches a shape nothing else has, after which that file's last-bit
    gates move. The reset afterwards is the same precaution taken twice.
    """
    build, simulations, ladder = MONTE_CARLO_LADDER[fixture]
    try:
        spot = build().params['Price Factors']['EquityPrice.EQ']['Spot']
        _, _, second = valued(build(), simulations=simulations)
        gamma = entry(second, EQUITY)
        assert abs(gamma) > 1e-6, f'{fixture} reports no spot curvature to converge to: {gamma!r}'

        previous = None
        for h, tolerance in ladder:
            up = reported_delta(build(spot + h), simulations)
            down = reported_delta(build(spot - h), simulations)
            relative = abs((up - down) / (2.0 * h) / gamma - 1.0)
            assert relative < tolerance, (
                f'{fixture} h={h:g}: the reported gamma {gamma!r} is {relative:.3e} off the '
                f'differenced delta {(up - down) / (2.0 * h)!r}')
            assert previous is None or relative < previous / 20.0, (
                f'{fixture} h={h:g}: {relative:.3e} against {previous:.3e} is not h^2 convergence')
            previous = relative
    finally:
        torch._dynamo.reset()
