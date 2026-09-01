"""`Greeks: 'All'` - the second-order block, its oracles, and the two things it refuses.

THE BLOCK WAS UNREACHABLE, WHICH IS WHY IT HAD NO COVERAGE. `Base_Revaluation.__init_shared_mem`
tests `params['Greeks'] == 'All'` to set `Base_Reval_State.gamma`, and that is the ONLY place the
engine looks for the string; the field declared `values=['First', 'No']`, so no panel, validator or
schema-authored job could ask for it. Hand-written JSON always could, which is what makes it a
silent gap rather than a broken run - and what left an entire reported block with nothing measuring
it. `tests/test_schema_emission.py::test_every_value_the_engine_tests_for_is_one_the_menu_offers`
is the gate that will not let a menu drift from the engine again; this file is the block itself.

WHAT THE ORACLES ARE. Closed forms, identities and convergent differences, hand-authored here, and
none of them is derivus asked twice:

  GAMMA      Black-Scholes d2V/dS2 on a vanilla European call. In a zero-rate zero-dividend world
             the engine's forward IS the spot and every discount factor is one, so the two are
             the same arithmetic and the reading is asserted in ULPS rather than to a tolerance.
             Carry is then switched on and the difference is a CONVENTION, measured and named:
             the equity forward runs to expiry plus a two-business-day settlement lag
             (`instruments.option_date_info`), the discounting runs to expiry, so exp(-rT) is the
             formula's to the ulp and the carry is (r - q) over a slightly longer year.
  VANNA      the cross-column identity, which is the one an oracle on the diagonal cannot see. The
             surface is interpolated in MONEYNESS, so moving the spot moves the interpolation
             WEIGHTS: the 1.0 nodes carry vanna less what the weight shift takes away and the 1.2
             nodes carry exactly that shift, and the two sum back to the closed-form vanna. A
             Hessian that dropped the weight derivative would still look plausible on the diagonal.
  ZERO       an FX forward is LINEAR in the spot, so d2V/dFX2 is exactly zero - not small. The
             mutation that matters is the placebo one: the same matrix has curve convexity of 1e7
             in it, so the gate cannot be passing because nothing was reported.
  CONVEXITY  a swap's d2V/dr2 against a CENTRAL DIFFERENCE of the reported AAD delta. Base
             valuation is deterministic - one date, one scenario, no Monte Carlo in a linear rates
             deal - so no common random numbers are needed and the ladder is pure truncation
             error: 3.0e-6 -> 3.0e-8 -> 2.9e-10 over h = 1e-3, 1e-4, 1e-5, which is h^2 to three
             digits and lands on the FD noise floor at 1e-6.
  LADDER     the same central difference on the two MONTE CARLO fixtures, which is the only
             correctness statement available where no closed form is. It is owed because SYMMETRY
             IS NOT ONE: `report_hessian` mirrors an upper triangle, so `H == H.T` holds of
             whatever the AAD put there, and that assertion plus "not empty" was the whole of what
             measured the Heston-Nandi block. The HN rung is also the eager-vs-fused equality
             gate - the gamma is walked by the eager sub-step and the delta it is differenced from
             by the compiled one.

TWO REFUSALS, both because the alternative is a number that looks right.

  A BOUNDARY CORRECTION. A deal that decides on simulated state has its FIRST derivative repaired
  by `pricing.boundary_correction`, which is `(gap - gap.detach())` times a DETACHED coefficient.
  Differentiate it twice and the coefficient cannot move, so the density-derivative term - the one
  the correction exists to supply - is silently absent from the second derivative while the smooth
  part is present. The refusal names the deals; the honest route it points at is bumping the
  ADJOINT under common random numbers. It is `utils.SecondOrderRefused` and not a bare `Exception`
  so a caller can FALL BACK - only the second-order block is refused, so re-running the same job
  at `'First'` keeps the mark and the first-order one (`derivus_batch.CollateralBaseVal`).
  `Recompute_Inner_MC`. Already refused for its own reason (`pricing.InnerMCRecompute`), and gated
  in `test_recompute_equity_pricers` on the one adopter that reaches it.

AND ONE REPAIR THE AUDIT FOUND. `utils.hn_log_substep` is `torch.compile`d, and AOTAutograd's
compiled backward raises `does not currently support double backward` - so every Heston-Nandi
valuation with an unmonitored sub-step (any fixing more than a day apart) died on `Greeks: 'All'`
rather than reporting one. The eager spelling is kept beside the fused one and `shared.gamma`
picks; the gate below is on the seam, so it holds whatever pricer is walking it, and the LADDER
above is what says the two spellings are the same function. Both of those reset `torch._dynamo`
after themselves, and that line is load-bearing rather than tidy - see the seam gate's docstring.
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
#: The device `run_baseval` hands a job - `cuda:0` when there is one, `cpu` when there is not - and
#: therefore the device the seam gate has to walk. It is not a preference: the oracle that gate
#: reads is the compiled backward's OWN refusal, and inductor has to reach a backend for the device
#: the tensors are on before it can raise anything at all.
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BASE = pd.Timestamp('2024-06-28')
SPOT, STRIKE, VOL = 100.0, 100.0, 0.25
#: One year to the day, and the curves are ACT_365, so T is exactly 1.0 and the closed form below
#: needs no day-count of its own.
EXPIRY = BASE + pd.Timedelta(days=365)
#: The moneyness axis of the vol surface. 1.0 is where an ATM option sits and 1.2 is the node the
#: interpolation weight moves TOWARDS as the spot rises - which is what the vanna identity reads.
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

#: The rates world the convexity ladder runs in - five knots so the Hessian has an off-diagonal to
#: be wrong on, and a 7Y swap so the 5Y and 10Y buckets both carry weight.
SWAP_KNOTS = [0.25, 1.0, 3.0, 5.0, 10.0]
SWAP_LEVELS = np.array([0.030, 0.032, 0.035, 0.037, 0.040])


def equity_cfg(r=0.0, q=0.0, spot=SPOT, vol=VOL):
    """The vanilla call, alone, in a flat world. `r = q = 0` is the exact case: the engine's equity
    forward is then the spot itself and every discount factor is one, so nothing but the payoff
    formula separates it from the closed form."""
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
    """The Heston-Nandi autocall of `test_hn_oss_pricers` - one coupon 30 days out, so the pricer
    walks 29 unmonitored sub-steps to reach it - at a bumped spot when asked."""
    import test_hn_oss_pricers as hn
    config, _ = hn._autocall_cfg([30], [1.05], [0.05], 30, hn_params=hn.STRONG)
    if spot is not None:
        config.params['Price Factors']['EquityPrice.EQ']['Spot'] = spot
    return config


#: (the fixture, its inner paths, and the h-ladder with what each rung is allowed). Both are
#: autocalls at BASE valuation, which is what lets them past the refusal below: one reporting row
#: means no coupon is OBSERVED, so neither registers a boundary correction, and they are the only
#: Monte Carlo fixtures in this repo that report a second-order block at all.
#:
#: 4096 IS `test_recompute_equity_pricers`' OWN COUNT, deliberately - see the ladder's docstring.
#: This gate walks the batch shapes that file already walks, so it leaves the compiler nothing new.
#:
#: THE LAST RUNG'S TOLERANCE IS AN ULP ENVELOPE, not a truncation bound, and the gbm one is stated
#: that way after being measured - see the ladder's docstring for both devices' readings.
MONTE_CARLO_LADDER = {
    'gbm': (gbm_autocall_cfg, 1 << 12, [(1e-2, 1e-6), (1e-3, 1e-8), (1e-4, 2e-10)]),
    'heston-nandi': (hn_autocall_cfg, 1 << 12, [(1e-2, 2e-3), (1e-3, 2e-5), (1e-4, 2e-7)])}


def valued(config, greeks='All', simulations=1):
    """(portfolio value, first-order frame, second-order frame). The second is None below `'All'`.

    The first-order frame carries the factor LEVEL in `Value` and the gradient in the column named
    after the reporting reference, which is why the gradient column is picked by elimination.

    `simulations` is 1 for every analytic fixture here - none of them draws - and is what the Monte
    Carlo ladder raises, on the one seed both sides of a bump have to share.
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
    """One (row, column) of the second-order frame, addressed by the row labels of the FIRST-order
    frame - the two share an index by construction, and the columns carry the reporting reference
    as one extra outer level."""
    column = row if column is None else column
    return float(second.loc[row][('root',) + tuple(column)])


def black_scholes_call(S, K, T, r, q, sigma):
    """(price, delta, gamma, vanna) for a European call with continuous carry. Hand-authored from
    the textbook forms; `vanna` is d2V/dS dsigma, which the cross-column identity reads."""
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
    """The declaration, which is the whole of what was missing. `'All'` has to be on the menu for
    the block to exist as far as any schema-driven author is concerned, and `'No'` has to remain
    the default - turning a second derivative on by omission would multiply every job's cost."""
    declared = mapping['Calculation']['types']['BaseValuation']['Greeks']
    assert declared['values'] == ['All', 'First', 'No'], (
        'the second-order block is not on the menu the engine acts on')
    assert declared['value'] == 'No', 'second derivatives have become the default'


def test_the_second_order_block_is_reported_under_a_stable_key_beside_the_first():
    """The shape, stated once so a consumer can be written against it: `Greeks_Second` appears iff
    `'All'` was asked for, always beside `Greeks_First`, square and symmetric, and labelled on both
    axes by (Rate, Tenor, Tenor2, Tenor3) - the columns carrying the reporting reference on top.

    ITS ROWS ARE NOT THE FIRST-ORDER BLOCK'S ROWS, and that is a fact about the instrument rather
    than an inconsistency. Both frames report the SUPPORT of what they hold - all-zero rows are
    dropped - and a factor can have no first derivative and a real second one: the moneyness-1.2
    vol nodes below carry zero vega, because the surface interpolation gives them zero weight at
    the money, while d2V/dS dsigma there is 0.98 because a spot move is exactly what gives them
    weight. Joining the two frames on their index would silently drop those.
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
    """d2V/dS2 against the closed form where the engine and the formula are the SAME arithmetic.

    r = q = 0 makes the equity forward the spot and every discount factor one, so nothing but the
    ORDER the float64s were summed in is left between the two, and the gate is a count of the
    representable steps between them rather than a tolerance somebody chose. Four, not equality:
    this fixture happens to land on 0 steps for all three readings, and the audit's re-derivation
    of the same oracle at S=87.5, K=95, vol=0.31, T=2y lands on 4, 2 and 1 - so `==` here is a
    property of these numbers and not a statement about the block. Price and delta are asserted
    beside gamma because a gamma that matched off a value that did not would mean the two agreed
    by accident.
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
    """The same reading with r = 3% and q = 1%, where the engine is no longer doing the formula's
    arithmetic - and what separates them is a CONVENTION, not a numerical error.

    NAMED, because "curve interpolation" is not it: the curve is FLAT, so there is nothing to
    interpolate. `instruments.option_date_info` runs the equity forward to `Forward_Settlement` -
    expiry plus a two-business-day settlement lag - while the option discounts to expiry. Measured
    by put-call parity on this world: the engine's discount factor IS exp(-rT) to the last bit
    (r_engine - r = 5e-16), and its carry is (r - q) over T + 3/365, three calendar days being
    what two business days past a Saturday expiry comes to. The offset is a date, so it moves with
    the expiry - 2 days at 730d, 4 days at 182d - which is why no day-count fits it.

    So the difference scales with the CARRY, and the bound is chosen against that rather than
    against this one point: price 8.8e-4, delta 6.1e-4, gamma 2.9e-5 relative here, but the same
    convention reads 1.2e-4 at (5%, 0%) and 1.4e-4 at (0%, 2%). 3e-4 is 10x this reading and
    clear of its neighbours; the 1e-4 it replaces fitted this point with no headroom at all.
    """
    value, first, second = valued(equity_cfg(r=0.03, q=0.01))
    price, delta, gamma, _ = black_scholes_call(SPOT, STRIKE, 1.0, 0.03, 0.01, VOL)

    assert abs(value / price - 1.0) < 1e-3, 'the price moved off the closed form'
    assert abs(float(first.loc[EQUITY].iloc[-1]) / delta - 1.0) < 1e-3, 'the delta moved'
    assert abs(entry(second, EQUITY) / gamma - 1.0) < 3e-4, (
        'gamma {} is further from the closed form {} than the measured 2.9e-5'.format(
            entry(second, EQUITY), gamma))


def test_the_spot_vol_cross_terms_sum_to_the_closed_form_vanna():
    """The oracle the diagonal cannot supply, and the one that says the surface's own derivative is
    in the Hessian.

    The vol is interpolated in MONEYNESS, so a spot move does two things at once: it moves the
    option along a fixed vol (vanna, which lands on the node the weight sits on) and it moves the
    interpolation WEIGHT (which lands on the node the weight moves towards). The surface is flat,
    so the second has no effect on the VALUE at all - but it is a real second derivative, and the
    two halves are individually large and opposite. Their sum is the closed-form vanna, so a
    Hessian that dropped the weight derivative - or double counted it - fails here while every
    diagonal entry stays right.

    Measured on this fixture: the 1.0 nodes carry -1.7812 and the 1.2 nodes +1.9792, which is
    vanna = 0.19795 to five digits. The 0.8 nodes carry nothing, and that is not an omission - a
    rising spot moves weight from 1.0 towards 1.2 and leaves 0.8 at zero either way.
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
    """A linear product's second derivative in its own underlying is ZERO, and exactly zero - the
    forward is `Buy * FX * D_buy - Sell * D_sell`, with the spot appearing once.

    The placebo this has to survive is the empty matrix, so the same reading is required to have
    real curve convexity in it: an engine that reported nothing at all, or dropped every row, would
    pass the zero and fail the second half.
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
    """d2V/dr2 against a CENTRAL DIFFERENCE of the AAD delta, on a ladder that has to converge.

    No common random numbers are needed and none are used: base valuation is one date and one
    scenario, and a linear rates deal draws nothing, so the two bumped runs differ only by the
    bump. What is left is pure truncation error, and the tolerances above are the measured
    readings - 3.0e-6, 3.0e-8, 2.9e-10 - which is h^2 to three digits. h = 1e-6 is deliberately
    NOT here: it reads 2.2e-10, off the ladder and into the difference's own cancellation, and a
    gate that included it would be asserting the noise floor.

    The whole matrix is compared, not the diagonal: the off-diagonal is where a mis-assembled
    upper triangle would show, and the FD knows nothing about how the engine assembled it.
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
    """The refusal, and that it says WHICH deals.

    The correction that makes these deals' first derivative right is `(gap - gap.detach())` times
    a detached coefficient. A second derivative through it keeps the smooth part and silently
    loses the density-derivative term - the very thing the correction supplies - so what would come
    back is a plausible gamma with a term missing rather than a failure. The message has to name
    the deals because a portfolio's author cannot otherwise tell which ones to take out, and it
    points at bumping the adjoint under common random numbers, which is what does work here.
    """
    with pytest.raises(Exception) as raised:
        valued(re_._cfg('barrier'))
    message = str(raised.value)
    assert 'BARR1' in message, f'the refusal does not name the deal: {message}'
    assert 'density-derivative' in message and 'common random numbers' in message, (
        f'the refusal does not say why, or what to do instead: {message}')

    # and the same portfolio is fine at first order, so the refusal is about the SECOND derivative
    # and not about the deal being unpriceable
    _, first, second = valued(re_._cfg('barrier'), greeks='First')
    assert second is None and np.abs(first_order(first)).max() > 0.0


def test_hedge_monte_carlo_refuses_the_second_order_block():
    """`Greeks` is not a HedgeMonteCarlo parameter - nothing in that calculation reads the key -
    so `'All'` there is a job asking for something it will not get. Silently ignoring it is the
    failure mode worth raising over: the run succeeds, the report has no second-order block, and
    nothing says the request was dropped.

    ASKED THE WAY A JOB ASKS IT, which is the whole of what this gate adds over the raise itself:
    the key rides in the file's `Calculation` block, `run_hedgemontecarlo` folds that block into
    the parameters it executes with, and the refusal is what comes back from `run_job`. A
    hand-constructed calculation object would only prove the `if` exists - the finding is that the
    STORE can carry `'All'` to a calculation which cannot honour it. The fixture is a TEMPLATE, so
    the field is set in memory and the file is not touched.
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
    """`utils.hn_log_substep` is `torch.compile`d and AOTAutograd's compiled backward has no double
    backward - it RAISES - so every HN valuation whose fixings are more than a day apart died on
    `Greeks: 'All'`. Since the OSS pricers were written, and invisible because nothing could ask.

    Gated at the SEAM rather than through a pricer: `shared.gamma` is what picks the eager
    spelling, so this holds for whichever pricer walks the sub-step next. Both directions are
    asserted - the fused build must still be what an ordinary run takes, or the fix is a silent
    5.9x on every Heston-Nandi job.

    IT WALKS THE DEVICE A JOB WALKS, and that is what it was missing. The tensors carried no device
    at all, so the fused half compiled for the CPU inductor backend whatever the box was - and on a
    CUDA box with no host C++ compiler (this one: triton, no MSVC) that backend cannot build, so
    what came back was `InductorError: InvalidCxxCompiler: Compiler: cl is not found` and not the
    refusal being gated. The gate read a raise and could not tell which one. MEASURED, both
    spellings at both devices, float64, this box:

        cpu    eager d2/domega2  7,381,689.19    fused  InductorError: cl is not found
        cuda   eager d2/domega2 58,006,153.76    fused  RuntimeError: ... double backward

    The two eager numbers differ because `torch.manual_seed(1)` seeds a different stream per device
    and the gate asserts only that the curvature is THERE, which both rows say. `run_baseval` takes
    `cuda:0` when there is one, so the CUDA row is what a Heston-Nandi job actually walks, and
    `DEVICE` is that same choice spelled once.

    `needs_hn_fused` is now this gate's OWN precondition rather than a spare one: it asks for triton
    under CUDA and a host C++ compiler under CPU, which is exactly the backend the tensors below
    need before the compiled backward exists to refuse. A compiler-less CPU box skips by name
    instead of failing on an oracle it cannot produce.

    `torch._dynamo.reset()` afterwards is not tidiness, and it took two attempts to state why
    honestly. The compiler cache is a process global keyed on traced shapes, and this file feeds
    it a shape no pricer uses, a compile that FAILS, and - in the ladder above - both spellings of
    the sub-step at a third shape. Leave the resets out and `test_recompute_equity_pricers`' HN
    price moves in its LAST BIT between the taped and recomputed passes (0.17244252124564025 vs
    ...402), because the two passes no longer get the same generated kernel.

    IT IS AN ORDER EFFECT AND ONLY THE WHOLE SUITE SEES IT, which is what the audit's failure to
    reproduce it turned out to mean: this file and `test_recompute_equity_pricers` run back to
    back are green either way, and the move needs the compiler state the intervening files leave.
    That is why the reading is worth keeping and worth distrusting in equal measure - and why
    both gates here reset rather than one. Nothing about derivus; but a gate that silently
    retunes the compiler for every test after it is a gate that breaks other gates.
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
    """The same repair end to end, on the fixture that exposed it - a Heston-Nandi autocall whose
    single coupon is 30 days out, so the pricer walks 29 unmonitored sub-steps to reach it. It
    registers no boundary correction at base valuation (one reporting row, no coupon observed), so
    the second-order block is what it gets, and it is symmetric and non-trivial.

    THAT IS SHAPE, NOT CORRECTNESS, and neither half of it can fail on a number: symmetry is what
    `report_hessian` does to whatever it assembled, and "non-trivial" is a floor. What measures
    the numbers is the ladder below."""
    _, first, second = valued(hn_autocall_cfg())
    assert np.abs(second.values).max() > 0.0, 'the HN second-order block came back empty'
    assert np.array_equal(second.values, second.values.T)
    assert {r[0] for r in second.index} & {'EquityPrice.EQ'}, (
        'the spot is not in the reported block: {}'.format(sorted({r[0] for r in second.index})))


# the Heston-Nandi rung differences a `Greeks: 'First'` delta, and THAT walks the compiled sub-step
# (only the gamma takes the eager spelling) - so this rung, alone in the file, has the fused
# precondition and states it rather than dying three layers downstream on a collapsed frame
@pytest.mark.parametrize('fixture', [pytest.param(f, marks=needs_hn_fused) if f == 'heston-nandi'
                                     else f for f in sorted(MONTE_CARLO_LADDER)])
def test_a_monte_carlo_gamma_is_the_derivative_of_its_own_reported_delta(fixture):
    """d2V/dS2 against a CENTRAL DIFFERENCE of the reported AAD delta, on the two Monte Carlo
    fixtures - the correctness statement the block has nowhere a closed form reaches.

    IT IS OWED BECAUSE SYMMETRY IS NOT A CHECK. `report_hessian` assembles an upper triangle and
    mirrors it, `hessian + np.triu(hessian, k=1).T`, so `H == H.T` is true of whatever the AAD put
    there - a random upper triangle through those two lines comes out symmetric. Every gate the
    Heston-Nandi block had was that assertion beside "not empty", and neither can fail on a wrong
    number.

    Common random numbers are what leaves the ladder as pure truncation error: one seed, one
    scenario, and a bumped spot rescales the paths the same draws generate. Measured over three
    decades of h, ON CUDA float64, which is the device `run_baseval` takes when there is one -

        gbm            gamma -2.91303e-05,  2.12e-07 -> 2.12e-09 -> 7.61e-11
        heston-nandi   gamma  2.69093e-06,  3.62e-04 -> 3.62e-06 -> 3.44e-08

    which is h^2 to two digits over the first two rungs and not Monte Carlo noise: the gamma and
    the deltas come off the SAME paths, so what is being measured is whether the second derivative
    is the first one's derivative, whatever the paths priced. h = 1e-5 is left off for the reason
    the swap ladder leaves it off: it turns back UP on the GBM fixture (4.1e-10 here, 6.6e-10 on
    CPU), which is the difference's own cancellation rather than anything about the estimator. The
    Heston-Nandi spot curvature is genuinely SMALL at 4096 paths - 2.7e-06, and it changes sign
    against a longer run - which costs this gate nothing and is why the floor below is a placebo
    guard rather than a claim.

    THE GBM FIXTURE'S LAST RUNG IS AN ULP ENVELOPE AND ITS OLD 1e-11 WAS BELOW ONE ULP. At
    h = 1e-4 the two reported deltas are 1.176960350e-03 and 1.176966176e-03 and their difference
    is 5.826e-09, so ONE ulp of either delta (2.17e-19, both sit in the 2^-10 binade) moves the
    quotient by 3.7e-11 RELATIVE. That is the arithmetic's own resolution at this rung, and no
    tolerance below it is measuring the estimator. Readings taken here:

        cuda float64   7.614e-11   =  2.05 ulps      (this box, RTX 3090)
        cpu  float64   2.238e-11   =  0.60 ulps      (CUDA_VISIBLE_DEVICES=-1, same fixture)

    and the recorded 1.70e-12 that 1e-11 was cut to fit is 0.046 ulps - two roundings cancelling,
    not a bound. 2e-10 is 5.4 ulps: it clears the CUDA reading by 2.6x and the CPU one by 8.9x, and
    it is two orders inside the rung above's own 1e-8, so the ladder still has to FALL to pass. A
    gamma wrong by anything a mutation would produce is caught at h = 1e-3, where the reading is
    2.12e-09 against a tolerance of 1e-8 and the arithmetic still resolves four decades.

    THE GAMMA ITSELF IS A CUDA READING, and a CPU box does not reproduce it: the draw stream is
    per-device, so the same fixture prices different paths and reads gamma -2.87533e-05 there, 1.3%
    away. Both are correct answers to different sample sets - which is why every rung is scored
    against the gamma from its OWN run and never against the number written here. The Heston-Nandi
    rungs are already an ulp envelope by the same arithmetic (its h = 1e-4 difference resolves to
    6.4e-09 per ulp, the reading is 3.44e-08 = 5.3 ulps against a 2e-07 = 31-ulp tolerance) and are
    left alone.

    THE HESTON-NANDI RUNG IS ALSO THE EAGER-AGAINST-FUSED EQUALITY GATE, which is why it is here
    rather than a second GBM fixture. `shared.gamma` walks the eager `hn_log_substep` for the
    second derivative while `Greeks: 'First'` - the delta being differenced - walks the compiled
    one, so a fused kernel that was not the eager function shows up here as a ladder that will
    not converge, and nowhere else in this repo.

    WHICH IS ALSO WHY THAT RUNG CARRIES `needs_hn_fused` AND THE GBM ONE DOES NOT. The delta side
    needs the compiled sub-step to BUILD, and on a box with no backend for its device it does not:
    the deal is skipped CRITICAL, the mark collapses to a scalar zero, and what this gate reported
    was `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` out
    of `run_baseval` - three layers downstream of the thing that was actually missing. Measured on
    this box with the GPU hidden (`CUDA_VISIBLE_DEVICES=-1`, no MSVC): 1 failed that way, 13 passed,
    against 15 passed on CUDA. Stating the precondition turns that into a skip by name, which is
    what `conftest` exists for.

    BOTH FIXTURES RUN AT `test_recompute_equity_pricers`' OWN 4096 PATHS, and that is a
    requirement rather than a coincidence. Run this at a count nothing else uses and the compiler
    caches a shape nothing else has, after which that file's LAST-BIT gates move - measured: its
    HN price gate at 1<<14 here, its collateralised-gradient gate (2 ulps, 8.5e-22 absolute) at
    1<<13. Neither is a derivus defect and neither reproduces outside a full-suite run; both are
    the hazard the seam gate below documents, and matching the shapes is what removes it at
    source. The reset afterwards is the same precaution taken twice.
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
