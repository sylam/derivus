"""Is the HW2F swaption residual differentiable in its QUOTES, and does saying so cost nothing?

This is stage A of the same contract [Quote Sensitivities](../docs_src/developer/quote_sensitivities.md)
put around the curve bootstrap: the residual closure audited and its quote side closed, with the
forward solve untouched. There is no implicit-function wrapper here yet - the solve still leaves the
tape at the scipy boundary - so what is gated is exactly the two halves that wrapper will need.

    the forward   theta* is BIT-IDENTICAL with the quote side on and off, `np.array_equal` on the
                  solved parameter vector rather than a tolerance. The splice is `base + (carried -
                  detach(carried))` and is worth exactly zero, so this is structural.
    the quote     one backward pass off the residual reports d(residual)/d(vol), and it is the
                  derivative of the number the solve actually minimises - checked against a central
                  finite difference on the same frozen paths.

Plus the property both of those rest on: common random numbers are FROZEN for the life of a solve
AND fixed across solves, because the Sobol engine is seeded once with a constant. An optimizer that
re-drew its sample would be differencing the noise.

Nothing here is copied from a vendor file. The zero curve, the swaption surface and the four
benchmark swaptions are invented and only have to be plausibly shaped - the gates are identities
between two runs of the same world, not assertions about a level.

Run: ``pytest tests/test_swaption_quote_graph.py -q``
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
import scipy.optimize
import torch

from derivus import bootstrappers, riskfactors, utils
from derivus.config import ModelParams

BASE = pd.Timestamp('2026-08-03')
CCY, CURVE, VOL = 'ZAR', 'ZAR-JIBAR-3M', 'ZAR_SWAPTION'
BLOCK = 'HullWhite2FactorModelPrices.' + CURVE
#: A humped ZAR-shaped zero curve and a swaption cube that rises with expiry, falls with the
#: underlying tenor and smiles in moneyness. `ATM` builds a bicubic spline, so both surface axes
#: need four points.
ZERO = ((1.0, 0.0800), (2.0, 0.0835), (3.0, 0.0880), (5.0, 0.0915), (7.0, 0.0950), (10.0, 0.0905))
EXPIRIES, UNDERLYING, MONEYNESS = (0.25, 1.0, 2.0, 5.0), (1.0, 2.0, 5.0, 10.0), (-0.01, 0.0, 0.01)
#: (start, tenor, quoted vol in percent) - the last row quotes 0, which is what sends the
#: bootstrapper to the surface's own ATM read instead of the row
BENCHMARKS = ((1, 1, 20.5), (1, 5, 19.0), (2, 5, 21.0), (5, 5, 0.0))


def price_factors():
    quads = [[m, e, t, 0.20 + 0.01 * np.log1p(e) - 0.005 * np.log1p(t) + 2.0 * m * m]
             for t in UNDERLYING for e in EXPIRIES for m in MONEYNESS]
    return {
        'FxRate.{}'.format(CCY): {
            'Domestic_Currency': None, 'Interest_Rate': CURVE, 'Priority': 1, 'Spot': 1.0},
        'InterestRate.{}'.format(CURVE): {
            'Currency': CCY, 'Day_Count': 'ACT_365', 'Sub_Type': None, 'Curve': utils.Curve([], list(ZERO))},
        'InterestYieldVol.{}'.format(VOL): {
            'Property_Aliases': None, 'Currency': CCY, 'Distribution_Type': 'Lognormal',
            'Shift': utils.Percent(0), 'Surface': utils.Curve([], quads)}}


def definitions(bumps=()):
    """The `Instrument_Definitions` table, with `bumps` a `{row: shift in percent}` on the quote."""
    return [{'Start': pd.DateOffset(years=start), 'Tenor': pd.DateOffset(years=tenor),
             'Floating_Frequency': pd.DateOffset(months=3),
             'Fixed_Frequency': pd.DateOffset(months=3),
             'Floating_Day_Count': 'ACT_365', 'Fixed_Day_Count': 'ACT_365',
             'Market_Volatility': utils.Percent(vol + dict(bumps).get(row, 0.0)), 'Weight': 1.0}
            for row, (start, tenor, vol) in enumerate(BENCHMARKS)]


def premium_file(prices):
    """The ATM premium CSV `set_premiums` reads, in the shape `get_premium` indexes it by."""
    return pd.DataFrame([
        {'Currency': CCY, 'Strike': 'ATM', 'Shift': '0%', 'StrikeValue': 8.0,
         'Expiry': '{}Y'.format(start), 'UnderlyingTenor': '{}Y'.format(tenor),
         'Payer': price * 10000.0}
        for (start, tenor, _), price in zip(BENCHMARKS, prices)])


def closure(connect, bumps=(), premiums=None, delta=0.0, dtype=torch.float32):
    """The calibration's residual closure, built the way `bootstrap` builds it.

    `dtype` is `construct_bootstrapper`'s own, and float32 is what a job gets - stated here because
    it is the precision the residual and every model swaption price are computed in.
    """
    factors, interp = price_factors(), ModelParams()
    block = {'Swaption_Volatility': VOL, 'Generate_Instruments': 'No',
             'Quote_Sensitivity': 'Yes' if connect else 'No',
             'Instrument_Definitions': definitions(bumps)}
    boot = bootstrappers.construct_bootstrapper('HullWhite2FactorModelParameters', {}, dtype)
    rate = utils.check_rate_name(BLOCK)
    ir_factor = utils.Factor('InterestRate', rate[1:])
    surface = riskfactors.construct_factor(utils.Factor('InterestYieldVol', (VOL,)), factors, interp)
    surface.delta = delta
    ir_curve = riskfactors.construct_factor(ir_factor, factors, interp)
    surface.set_premiums(premiums, ir_curve.get_currency())
    implied_obj, process, vol_tenors = boot.implied_process(CCY, factors, {}, ir_curve, rate)
    mtm_dates = set([BASE + x['Start'] for x in block['Instrument_Definitions']])
    time_grid = utils.TimeGrid(mtm_dates, mtm_dates, mtm_dates)
    time_grid.set_base_date(BASE, delta=(10, vol_tenors * utils.DAYS_IN_YEAR))
    return boot.calc_loss({'instrument': block, 'Children': []}, BASE, time_grid, process,
                          implied_obj, ir_factor, surface)


def residual(loss_fn, implied_var, swaps):
    errors = loss_fn(implied_var)[1]
    return torch.stack([errors[name] for name in swaps])


def at(implied_var, theta):
    """Stand the closure's parameters at `theta`, the way the two scipy adapters do."""
    tensors = list(implied_var.values())
    for tensor, value in zip(tensors, np.split(theta, np.cumsum([x.numel() for x in tensors[:-1]]))):
        tensor.data = torch.from_numpy(value).to(tensor.device)
    return implied_var


SOLVED = {}


def solved(connect):
    """The closure with its parameters left standing at theta*, cached - the solve is the slow part.

    `least_squares` is one of the two optimizers `calc_loss` hands `bootstrap`, and the one whose
    path is a pure function of the residual and its Jacobian; basin hopping adds a numpy RNG on top
    and answers the same question more slowly.
    """
    if connect not in SOLVED:
        loss_fn, optimizers, implied_var, swaps, _ = closure(connect)
        name, x0, lsq_fn, jacobian, bounds = [o for o in optimizers if o[0] == 'leastsq'][0]
        theta = scipy.optimize.least_squares(lsq_fn, x0=x0, jac=jacobian, bounds=bounds)['x']
        # scipy's last evaluation need not be the accepted one, so put the vars back on theta*
        lsq_fn(theta)
        SOLVED[connect] = (loss_fn, implied_var, swaps, theta)
    return SOLVED[connect]


def test_the_solve_is_bit_identical_with_the_quote_side_on():
    """The forward pass cannot move, and `np.array_equal` is the only honest way to say so.

    Both halves are checked because they fail differently: the residual is what scipy minimises and
    theta* is where it stops, and a splice that perturbed the JACOBIAN alone would leave the first
    identical and the second not - which is exactly what attaching the model price to the carried
    half did when this was built.
    """
    off_fn, off_var, off_swaps, off_theta = solved(False)
    on_fn, on_var, on_swaps, on_theta = solved(True)
    assert np.array_equal(np.array([off_swaps[k].price for k in off_swaps]),
                          np.array([on_swaps[k].price for k in on_swaps]))
    assert np.array_equal(residual(off_fn, off_var, off_swaps).detach().cpu().numpy(),
                          residual(on_fn, on_var, on_swaps).detach().cpu().numpy())
    assert np.array_equal(off_theta, on_theta)


def test_the_quote_side_is_absent_unless_the_block_asks_for_it():
    """A quote leaf where no job asked for one is a graph nobody can see holding memory."""
    _, _, off_swaps, _ = solved(False)
    _, _, on_swaps, _ = solved(True)
    assert all(swap.quote is None and swap.premium is None for swap in off_swaps.values())
    assert all(swap.quote.dtype == torch.float64 and swap.quote.requires_grad
               for swap in on_swaps.values())


def test_the_residual_differentiates_in_its_quotes():
    """One backward pass off the residual at theta*, and every quote gets a finite non-zero number.

    Zero is the failure this whole workstream is about: a severed quote does not raise and does not
    move a value, it silently reports no sensitivity. The fourth benchmark quotes 0 on its row and
    therefore reads the surface's ATM spline, so this also says the leaf is placed on that read
    rather than only on the hand-authored column.
    """
    loss_fn, implied_var, swaps, _ = solved(True)
    grad = torch.autograd.grad(residual(loss_fn, implied_var, swaps).sum(),
                               [swap.quote for swap in swaps.values()])
    values = np.array([float(g) for g in grad])
    assert np.isfinite(values).all()
    assert (np.abs(values) > 0.0).all()


def test_the_quote_gradient_is_the_derivative_of_the_number_the_solve_minimises():
    """The one gate that can see a plausible wrong answer, and the frozen paths make it cheap.

    The Sobol engine is seeded with a constant, so a closure rebuilt at a bumped quote prices the
    SAME sample: the central difference below differences the quote and nothing else, and no
    variance reduction has to be argued for.

    Run in float64 at the float32 solve's theta*, because the identity is a mathematical one and
    the PRECISION of the residual is the separate finding. In float32 the difference is a
    cancellation of two numbers near a minimum and agreement stalls at about 5e-3 however small the
    bump gets - it is the residual's own resolution, not the derivative's. Float64 converges as
    h-squared, three decades over three bump sizes: 1e-2, 1e-4, 1e-6 - measured, and the reason the
    tolerance below can be tight.

    The surface-quoted benchmark is not bumped. Its `Market_Volatility` column is what SELECTS
    between the row and the surface's ATM read, so moving it off zero changes where the quote comes
    from instead of moving the quote; a bump on the surface itself is a different derivative, and
    that map is severed by `RectBivariateSpline` on purpose.
    """
    _, _, _, theta = solved(True)
    loss_fn, _, implied_var, swaps, _ = closure(True, dtype=torch.float64)
    grad = torch.autograd.grad(residual(loss_fn, at(implied_var, theta), swaps).sum(),
                               [swap.quote for swap in swaps.values()])
    h = 0.0005
    for row, quoted in enumerate(vol for _, _, vol in BENCHMARKS):
        if not quoted:
            continue
        moved = []
        for shift in (h, -h):
            fn, _, var, bumped, _ = closure(True, bumps=((row, shift),), dtype=torch.float64)
            moved.append(float(residual(fn, at(var, theta), bumped).sum().detach()))
        # the bump is authored in percent and the leaf carries the decimal vol
        finite = (moved[0] - moved[1]) / (2.0 * h / 100.0)
        assert abs(float(grad[row]) - finite) < 1e-4 * abs(finite), (row, float(grad[row]), finite)


def test_the_residual_survives_an_optimizer_that_does_not_retain_its_graph():
    """Basin hopping calls `total_loss.backward()` with no `retain_graph`, so it frees the graph
    behind every evaluation. A quote-side subgraph compiled ONCE with the benchmark set is freed
    with the first call and every call after it raises - a whole optimizer dying rather than a
    number coming out wrong, and the least-squares branch would not notice because its own Jacobian
    retains. `premium` is a map rebuilt per evaluation for this reason, and two turns say so."""
    loss_fn, implied_var, swaps, _ = solved(True)
    for _ in range(2):
        residual(loss_fn, implied_var, swaps).sum().backward()


def test_common_random_numbers_are_frozen_for_the_life_of_a_solve():
    """Twice through the same closure, and once through a closure built from scratch.

    `reset` clears the pricing memo tables at every evaluation but re-draws the Sobol sample only
    when there is not one, and the engine is seeded with a constant - so the first equality says the
    optimizer differences its parameters rather than its sample, and the second says two solves of
    the same block are comparable at all.
    """
    loss_fn, implied_var, swaps, theta = solved(True)
    first = residual(loss_fn, implied_var, swaps).detach().cpu().numpy()
    assert np.array_equal(first, residual(loss_fn, implied_var, swaps).detach().cpu().numpy())
    fresh_fn, _, fresh_var, fresh_swaps, _ = closure(True)
    assert np.array_equal(first, residual(
        fresh_fn, at(fresh_var, theta), fresh_swaps).detach().cpu().numpy())


def test_the_tensor_black_is_the_numpy_black():
    """The differentiable premium is a twin of the market premium, not a second opinion of it.

    `utils.black_european_option` already existed - it is what the cap/floor and swaption pricers
    value an option with - so the quote side reuses it rather than adding a second Black beside the
    numpy one in `utils.black_european_option_price`. If the two ever disagreed, the derivative
    reported would belong to a number nobody priced.
    """
    _, _, swaps, _ = solved(True)
    for swap in swaps.values():
        assert float(swap.premium(swap.quote).detach()) == pytest.approx(swap.price, rel=1e-12)


def test_a_premium_quote_carries_itself():
    """A block quoting PREMIUMS needs no Black at all - the splice is the identity, and the market
    number is the leaf. Same closure, same switch, one branch further up."""
    prices = [swap.price for swap in solved(True)[2].values()]
    loss_fn, _, implied_var, swaps, _ = closure(True, premiums=premium_file(prices))
    assert all(swap.premium(swap.quote) is swap.quote for swap in swaps.values())
    grad = torch.autograd.grad(residual(loss_fn, implied_var, swaps).sum(),
                               [swap.quote for swap in swaps.values()])
    assert all(float(g) != 0.0 and np.isfinite(float(g)) for g in grad)


def test_a_premium_restruck_by_volatility_delta_declines_the_quote_side():
    """`Volatility_Delta` re-strikes a premium through a brentq implied-vol solve, and a numerical
    root find carries no derivative. Reporting zero there would be the exact failure this gate
    exists to prevent, so the block says so instead."""
    prices = [swap.price for swap in solved(True)[2].values()]
    with pytest.raises(Exception, match='Quote_Sensitivity'):
        closure(True, premiums=premium_file(prices), delta=0.01)
