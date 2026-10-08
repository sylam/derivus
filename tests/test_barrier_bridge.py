"""A barrier's survival between grid dates is a probability, not an endpoint check.

`pv_barrier_option` prices the remaining life with Reiner-Rubinstein, which assumes CONTINUOUS
monitoring. The historical path state asked only whether the spot sat beyond the barrier ON a grid
date, so every path that crossed and came back counted as still alive - a state inconsistent with
the formula applied to it, and worse the coarser the grid.

The gate needs no external reference: at r = q = 0 with the simulation drift to match, the option
value is a MARTINGALE, so E[MTM_t] equals the t=0 value at every t on every grid, and the t=0 row
is the pure closed form because no history has accumulated. Endpoint-only survival fails it by
+12.8% at 3m and +26.8% at 9m on the quarterly grid, moving WITH the grid.

A MARTINGALE STATISTIC IS AN EXPECTATION AND COSTS PATHS, so every gate reading one states the
seed distribution it was measured over: a tolerance set against a single seed is a coin toss
dressed as a threshold. Where a deterministic LEDGER says the same thing it is used instead, and
where the expectation is the only route the paths were raised until the noise sat clear of the
defect. No tolerance here was widened to make it green.
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import pytest
import torch

import derivus
from derivus import utils
from derivus.config import Config
from derivus.instruments import construct_instrument
from crn_ladder import ladder
import trial_equity
import trial_fx
from test_position_scaling import document, marks

BASE = pd.Timestamp('2024-06-28')
DTYPE = torch.float64
VOL = 0.25
SPOT = 100.0


def _cfg():
    """Down-and-out call, continuously monitored, in a zero-rate zero-dividend world so the value is
    a martingale under the simulation measure. GBM, whose lognormal interval law is what publishes
    a bridge variance rate at all."""
    field = {
        'Object': 'EquityBarrierOption', 'Reference': 'BARR1', 'Currency': 'USD',
        'Payoff_Currency': 'USD', 'Equity': 'EQ', 'Dividends': 'EQ', 'Discount_Rate': 'USD',
        'Equity_Volatility': 'EQ', 'Buy_Sell': 'Buy', 'Option_Type': 'Call',
        'Strike_Price': 100.0, 'Expiry_Date': BASE + pd.Timedelta(days=365), 'Units': 1.0,
        'Barrier_Type': 'Down_And_Out', 'Barrier_Price': 90.0, 'Cash_Rebate': 0.0,
        'Barrier_Dates': [], 'Barrier_Monitoring_Frequency': pd.DateOffset(days=0),
    }
    c = Config()
    c.params['System Parameters']['Base_Currency'] = 'USD'
    c.params['System Parameters']['Base_Date'] = BASE
    c.params['Price Factors'] = {
        'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Priority': 1, 'Spot': 1.0},
        'InterestRate.USD': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                             'Curve': utils.Curve([], [[0.0, 0.0], [5.0, 0.0]])},
        'EquityPrice.EQ': {'Spot': SPOT, 'Currency': 'USD', 'Interest_Rate': 'USD',
                           'Issuer': '', 'Respect_Default': 'No', 'Jump_Level': 0.0},
        'DividendRate.EQ': {'Currency': 'USD', 'Floor': None,
                            'Curve': utils.Curve([], [[0.0, 0.0], [5.0, 0.0]])},
        'VolatilityGrid.EQ': {'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness',
                              'Surface': utils.Curve([], [[m, t, VOL] for m in (0.8, 1.0, 1.2)
                                                          for t in (0.02, 2.0)])},
    }
    # drift 0 with r = q = 0 makes the SIMULATED spot a martingale, which is what lets the option
    # value be one: the pricing measure and the simulation measure have to be the same
    c.params['Price Models'] = {'GBMAssetPriceModel.EQ': {'Vol': VOL, 'Drift': 0.0}}
    c.params['Model Configuration'].append('EquityPrice', (), 'GBMAssetPriceModel')
    c.deals = {'Attributes': {'Reference': 'test'},
               'Deals': {'Children': [{'Instrument': construct_instrument(field, {})}]},
               'Calculation': {'Base_Date': BASE, 'Currency': 'USD'}}
    return c


ONE_TOUCH = {
    'Object': 'EquityOneTouchOption', 'Reference': 'OT1', 'Currency': 'USD',
    'Payoff_Currency': 'USD', 'Equity': 'EQ', 'Discount_Rate': 'USD', 'Equity_Volatility': 'EQ',
    'Buy_Sell': 'Buy', 'Cash_Payoff': 100.0, 'Payoff_Type': 'Cash', 'Barrier_Price': 90.0,
    'Barrier_Type': 'Down', 'Expiry_Date': BASE + pd.Timedelta(days=365),
    'Barrier_Monitoring_Frequency': pd.DateOffset(days=0),
}


def _profile(grid, seed=1, batch=8192, deal=None, mcmc=None):
    params = {'Run_Date': BASE.strftime('%Y-%m-%d'), 'Time_grid': grid, 'Batch_Size': batch,
              'Simulation_Batches': 1, 'Random_Seed': seed, 'Currency': 'USD',
              'Tenor_Offset': 0.0, 'Deflation_Interest_Rate': 'USD',
              **({'MCMC_Simulations': mcmc} if mcmc else {})}
    c = _cfg()
    if deal is not None:
        c.deals['Deals']['Children'] = [{'Instrument': construct_instrument(deal, {})}]
    _, out = derivus.run_cmc(c, prec=DTYPE, overrides=params)
    return out['Results']['mtm']


def _cva(spot, deal, gradient, batch=4096, mcmc=None):
    """CVA, its AAD spot gradient when asked, or the whole gradient frame at `'frame'`. A
    counterparty is what gives the barrier a sensitivity worth measuring: the exposure profile is
    where the touch state accumulates, which base valuation - one deal-time row, no interval, no
    history - structurally cannot show."""
    c = _cfg()
    c.params['Price Factors']['EquityPrice.EQ']['Spot'] = spot
    c.params['Price Factors']['SurvivalProb.CPTY'] = {
        'Recovery_Rate': 0.4, 'Curve': utils.Curve([], [[0.0, 0.0], [10.0, 0.4]])}
    c.deals['Deals']['Children'] = [{'Instrument': construct_instrument(deal, {})}]
    _, out = derivus.run_cmc(c, prec=DTYPE, overrides={
        'Run_Date': BASE.strftime('%Y-%m-%d'), 'Time_grid': '0d 3m(3m)', 'Batch_Size': batch,
        'Simulation_Batches': 1, 'Random_Seed': 1, 'Currency': 'USD', 'Tenor_Offset': 0.0,
        'Deflation_Interest_Rate': 'USD', 'Gradient_Variables': 'Factors',
        **({'MCMC_Simulations': mcmc} if mcmc else {}),
        'Credit_Valuation_Adjustment': {
            'Calculate': 'Yes', 'Counterparty': 'CPTY', 'Deflate_Stochastically': 'No',
            'Stochastic_Hazard_Rates': 'No', 'Gradient': 'Yes' if gradient else 'No'}})
    if not gradient:
        return float(out['Results']['cva'])
    g = out['Results']['grad_cva']['Gradient']
    if gradient == 'frame':
        return g
    rows = [i for i in g.index if 'EquityPrice' in str(i[0])]
    return float(g.loc[rows[0]]) if rows else 0.0             # a factor with no .grad is dropped


def _analytic_touch_probability():
    """P(the minimum of the GBM breaches the barrier before expiry), by reflection - independent of
    derivus, which is the point."""
    import math
    mu, sig, b = -0.5 * VOL ** 2, VOL, math.log(90.0 / SPOT)
    phi = lambda x: 0.5 * math.erfc(-x / math.sqrt(2.0))
    return phi((b - mu) / sig) + math.exp(2.0 * mu * b / sig ** 2) * phi((b + mu) / sig)


@pytest.mark.parametrize('grid,freq_days', [('0d 3m(3m)', 0), ('0d 1m(1m)', 30)],
                         ids=['quarterly-continuous', 'monthly-monthly'])
def test_the_barrier_value_is_a_martingale_on_any_grid_and_any_monitoring(grid, freq_days):
    """With r = 0 the value is a martingale, so every date on every grid reports the t=0 price -
    at any monitoring frequency too: a discretely monitored barrier is priced by a CONTINUOUS
    closed form against a barrier shifted away from the live region (Broadie-Glasserman-Kou), and
    the bridge has to monitor that same shifted barrier.

    Endpoint-only survival fails by +12.8% at 3m and +26.8% at 9m on the quarterly grid, a
    DIFFERENT amount on each grid (10.96-28.60% across quarterly, monthly and weekly); handing the
    bridge the RAW barrier while the formula prices the shifted one decayed monthly monitoring
    -11.58%. THE PATHS WERE RAISED, THE TOLERANCE WAS NOT: at 8192 the statistic read 6.19-7.08%
    over seeds 1-20; at 65536 the worst over seeds 1-20 is 2.28% on the grids and 1.64% across
    frequencies, so 4% and 5% sit clear of the noise. Inception RISES with coarser monitoring (8.485
    monthly against 7.176 continuous), fewer observations meaning fewer chances to knock out.

    Killing mutation: endpoint-only survival (the bridge's crossing probability dropped).
    """
    deal = dict(BARRIER_DEAL, Barrier_Monitoring_Frequency=pd.DateOffset(days=freq_days))
    v = _profile(grid, deal=deal, batch=65536).values.mean(axis=1)
    assert v[0] > 0.0, 'a bought down-and-out call should be worth something at inception'
    drift = np.abs(v - v[0]).max() / v[0]
    assert drift < (0.04 if freq_days == 0 else 0.05), (
        f'{grid}: profile drifts {drift:.1%} from inception {v[0]:.4f} - survival is not being '
        f'carried as a probability of the barrier the closed form prices\n{np.round(v, 3)}')


def test_a_one_touch_holds_its_payout_when_paid_at_expiry_and_settles_when_paid_on_touch():
    """A one-touch paying at EXPIRY owes the nominal on every touched path, so between the touch and
    expiry such a path holds a CERTAIN claim worth its discounted value - carried as zero once, the
    deal jumping to the nominal on the last date. At r=0 the value is a martingale equal to the
    nominal times the reflection formula's touch probability. Paid ON touch the cash settles and the
    path stops carrying it, so that profile DECAYS; both timings agree at inception, r=0 leaving
    nothing to discount between them.

    Killing mutation: endpoint-only survival (the bridge's crossing probability dropped).
    """
    at_expiry = _profile('0d 3m(3m)', deal=dict(ONE_TOUCH, Payment_Timing='Expiry'))
    v = at_expiry.values.mean(axis=1)
    expected = 100.0 * _analytic_touch_probability()
    assert v[0] == pytest.approx(expected, rel=2e-3), (v[0], expected)
    assert np.abs(v - v[0]).max() / v[0] < 0.03, f'paid at expiry is not being held: {np.round(v, 2)}'
    on_touch = _profile('0d 3m(3m)', deal=dict(ONE_TOUCH, Payment_Timing='Touch')).values.mean(axis=1)
    assert on_touch[0] == pytest.approx(v[0], rel=2e-3), (on_touch[0], v[0])
    assert on_touch[-1] < 0.25 * on_touch[0], f'paid-on-touch should run off: {np.round(on_touch, 2)}'


#: The same terms paid at expiry where the barrier was NEVER touched.
NO_TOUCH = dict(ONE_TOUCH, Object='EquityNoTouchOption', Reference='NT1')


def test_a_no_touch_is_the_one_touch_paid_at_expiry_s_complement():
    """A NO-TOUCH PAYS WHERE THE BARRIER WAS NEVER TOUCHED, at expiry. Beside the one-touch paid at
    expiry on the same terms it holds the payout on every path and every date, touched and
    untouched alike, and the two settle it between them at expiry; alone it is worth the payout
    less the reflection formula's touch probability; and in a world with rates the pair, on an
    equity and on an exchange rate, is worth the cashflow paying the payout at expiry.

    Killing mutation: the no-touch's untouched leg left undiscounted (`1 - payoff` for
    `exp(-r tau) - payoff`).
    """
    c = _cfg()
    c.deals['Deals']['Children'] = [{'Instrument': construct_instrument(deal, {})} for deal in (
        dict(ONE_TOUCH, Payment_Timing='Expiry'), NO_TOUCH)]
    _, out = derivus.run_cmc(c, prec=DTYPE, overrides={
        'Run_Date': BASE.strftime('%Y-%m-%d'), 'Time_grid': '0d 3m(3m)', 'Batch_Size': 512,
        'Simulation_Batches': 1, 'Random_Seed': 1, 'Currency': 'USD', 'Tenor_Offset': 0.0,
        'Generate_Cashflows': 'Yes', 'Deflation_Interest_Rate': 'USD'})
    assert np.abs(out['Results']['mtm'].values - 100.0).max() < 1e-9, 'the pair holds the payout'
    settled = out['Results']['cashflows']['USD'].loc[BASE + pd.Timedelta(days=365)]
    assert np.abs(settled.values - 100.0).max() < 1e-9, 'and settles it between them at expiry'

    alone = _profile('0d 3m(3m)', deal=NO_TOUCH).values.mean(axis=1)
    assert alone[0] == pytest.approx(100.0 * (1.0 - _analytic_touch_probability()), rel=2e-3)

    for family, pair in ((trial_equity, ('EQOT', 'EQNT')), (trial_fx, ('FXOT', 'FXNT'))):
        deals = [deal for deal in family.DEALS if deal['Reference'] in pair] + [
            {'Object': 'FixedCashflowDeal', 'Reference': 'PAYOUT', 'Currency': 'USD',
             'Discount_Rate': 'USD', 'Amount': 10_000.0, 'Payment_Date': trial_fx.E}]
        marked = {name: float.fromhex(mark) for name, mark in marks(document(SimpleNamespace(
            DEALS=deals, FACTORS=family.FACTORS, CONFIGURATION=family.CONFIGURATION))).items()}
        assert marked[pair[0]] + marked[pair[1]] == pytest.approx(marked['PAYOUT'], rel=1e-9), (
            pair, marked)


BARRIER_DEAL = {
    'Object': 'EquityBarrierOption', 'Reference': 'BARR1', 'Currency': 'USD',
    'Payoff_Currency': 'USD', 'Equity': 'EQ', 'Dividends': 'EQ', 'Discount_Rate': 'USD',
    'Equity_Volatility': 'EQ', 'Buy_Sell': 'Buy', 'Option_Type': 'Call', 'Strike_Price': 100.0,
    'Expiry_Date': BASE + pd.Timedelta(days=365), 'Units': 1.0, 'Barrier_Type': 'Down_And_Out',
    'Barrier_Price': 90.0, 'Cash_Rebate': 0.0, 'Barrier_Dates': [],
    'Barrier_Monitoring_Frequency': pd.DateOffset(days=0),
}


@pytest.mark.parametrize('deal,label', [
    (BARRIER_DEAL, 'barrier'),
    (dict(ONE_TOUCH, Payment_Timing='Expiry'), 'one_touch'),
    (NO_TOUCH, 'no_touch')])
def test_aad_delta_matches_bump_and_reprice(deal, label):
    """The gradient has to be the derivative of the value actually reported, so under common random
    numbers a central difference estimates the same derivative without touching the tape.

    An INDICATOR has zero derivative almost everywhere, so the knock-out channel contributed
    nothing and AAD reported the wrong number while looking well-behaved: 9-19% off for the barrier
    and 31-44% for the one-touch, and - the discriminating signal - the ladder SCATTERED instead of
    converging, shrinking the bump changing how many paths sit on the far side of the jump.
    Carrying survival as a probability gives 0.00% at 0.00% flatness.

    Killing mutation: endpoint-only survival (the bridge's crossing probability dropped).
    """
    aad = _cva(SPOT, deal, gradient=True)
    assert abs(aad) > 1e-6, 'a barrier with a live knock-out should have a spot delta'
    r = ladder(price=lambda s: _cva(s, deal, False), aad=aad, base=SPOT,
               rungs=(2e-4, 5e-4, 1e-3, 2e-3))
    assert r.agrees(tol=0.02), (
        f'{label}: a channel through which spot moves the value is not being differentiated\n{r}')


MONTHLY_BARRIER = [BASE + pd.Timedelta(days=d) for d in range(30, 366, 30)]


def _cashflow_run(deal_overrides, batch, mcmc, grid='0d 1m(1m)'):
    """One discrete-barrier run with its cash ledger, as FRAMES rather than a total: which row a
    settlement lands on is itself under test, and a sum over rows cannot see it."""
    c = _cfg()
    c.deals['Deals']['Children'] = [{'Instrument': construct_instrument(
        dict(BARRIER_DEAL, **deal_overrides), {})}]
    _, out = derivus.run_cmc(c, prec=DTYPE, overrides={
        'Run_Date': BASE.strftime('%Y-%m-%d'), 'Time_grid': grid, 'Batch_Size': batch,
        'Simulation_Batches': 1, 'Random_Seed': 1, 'Currency': 'USD', 'Tenor_Offset': 0.0,
        'MCMC_Simulations': mcmc, 'Generate_Cashflows': 'Yes', 'Deflation_Interest_Rate': 'USD'})
    return out['Results']['mtm'], out['Results']['cashflows']


def _totals(deal_overrides, batch=512, mcmc=128):
    """Inception price and every currency's settled cash, for the gates that need magnitudes."""
    mtm, cf = _cashflow_run(deal_overrides, batch, mcmc)
    return (mtm.values.mean(axis=1)[0], sum(float(np.nansum(v.values)) for v in cf.values()))


@pytest.mark.parametrize('grid', ['0d 1m(1m)', '0d 2d 1w(1w) 3m(1m)'])
def test_discrete_barrier_is_observed_only_on_its_own_dates(grid):
    """A DISCRETELY monitored barrier is observed on the dates its terms name and nowhere else.
    `pv_discrete_barrier_option` latched the crossing with a cumsum over every MTM row of each
    block, so it monitored 37 reporting rows against 12 barrier dates - knocking scenarios out on
    dates the deal never observes, monitoring expiry, and missing the first barrier date entirely.

    THE STATEMENT IS DETERMINISTIC IN THE PRICER'S OWN LEDGER, which is why this is not a martingale
    check: that statistic had a 1.46% seed sd against a 1% tolerance and failed 23 of 50 seeds. At
    `Cash_Rebate=1.0` and `Units=1.0` the knock-out rebate is one unit of ABSOLUTE cash, so
    `Generate_Cashflows` writes a bare 0/1 indicator carrying the same `barrier_hit` latch the
    price is built from. Four exact facts, no expectation anywhere:

      the settled rows ARE the authored barrier dates,
      every entry is EXACTLY 0.0 or 1.0 and a path settles at most once,
      a knocked-out path is worth EXACTLY 0.0 on every later row and strictly positive on every row
        up to and including its knock-out,
      and it is owed EXACTLY 0.0 by the terminal settle.

    The last two stop this being a cash-date check: they read `row_barrier_hit`, the mask that
    prices the block, and tie it to the cash. Both grids assert bit-identically.

    Each KILLED on both grids once: the historical cumsum form, the settle moved one row earlier,
    the PRICE mask alone cumsummed with the ledger untouched (which proves the mtm half
    load-bearing), and the OSS skipping its strip's first observation. Monitoring expiry SURVIVES,
    at the latch because the last block has no later block to inform.

    Killing mutation: `newly_hit` dropped, so a crossed path re-settles its rebate.
    """
    mtm, cf = _cashflow_run(dict(Barrier_Price=95.0, Cash_Rebate=1.0, Units=1.0,
                                 Barrier_Dates=MONTHLY_BARRIER), 2048, 128, grid)
    assert list(cf) == ['USD'], f'one currency, so the USD frame IS the ledger: {list(cf)}'
    dates, value = list(mtm.index), mtm.values
    assert list(cf['USD'].index) == dates, 'cash and mtm must be reported on one grid'
    assert len(dates) > len(MONTHLY_BARRIER) + 1, (
        'the grid must be FINER than the barrier schedule or nothing is being tested')

    # the terminal row settles the whole surviving mtm, so the rebate ledger is everything above it
    cash = cf['USD'].values[:-1]
    settled = {dates[i] for i in range(len(cash)) if cash[i].any()}
    assert settled == set(MONTHLY_BARRIER), (
        f'knock-out cash settles on {sorted(str(d.date()) for d in settled ^ set(MONTHLY_BARRIER))} '
        f'against the deal\'s own {len(MONTHLY_BARRIER)} barrier dates')
    assert np.array_equal(np.unique(cash), [0.0, 1.0]), (
        f'a unit of absolute cash per knock-out is a bare indicator, got {np.unique(cash)}')

    settle_row = np.where(cash > 0.0, np.arange(len(cash))[:, np.newaxis], -1).max(axis=0)
    assert cash.sum(axis=0).max() == 1.0, 'a path knocks out once'
    row_of = np.arange(len(dates))[:, np.newaxis]
    dead, alive = row_of > settle_row, (row_of <= settle_row) & (settle_row >= 0)
    assert 0.1 < (settle_row >= 0).mean() < 0.9, (
        f'{(settle_row >= 0).sum()} of {cash.shape[1]} paths knocked out - fixture is vacuous')
    assert np.count_nonzero(value[dead & (settle_row >= 0)]) == 0, (
        'a knocked-out path is worth exactly zero on every later row')
    assert (value[alive] > 0.0).all(), (
        f'{(value[alive] <= 0.0).sum()} rows are worth nothing at or before their own knock-out - '
        f'the price mask is resolving scenarios on rows the deal never observes')
    assert not cf['USD'].values[-1][settle_row >= 0].any(), (
        'the terminal settle owes a knocked-out path nothing')


def test_discrete_monitoring_prices_to_an_independent_simulation():
    """What the martingale statistic cannot say: the twelve observations are priced RIGHT rather
    than consistently with themselves. Inception carries no history, so this is the OSS's analytic
    treatment of the whole strip against a simulation derivus did not produce - 10.5m paths with
    the terminal stub integrated in closed form, V_0 = 8.4787 +- 0.0051, against a reading of
    8.469040 (-0.114%), deterministic because the inner OSS draws the canonical Sobol block.

    3e-3 is 2.6x the gap, and the slack is the inner QMC count's rather than the reference's: at
    inception every outer path reads the same inner rows, so the reading is ONE inner sample's -
    +1.34% at 256 inner paths, +0.64% at 1024, -0.12% at 4096, -0.05% at 65536 - and the batch
    is the Sobol arm's smallest.

    The OSS monitoring expiry SURVIVES, the barrier at 90 being below the strike at 100, so a path
    the extra observation knocks out already pays zero and only a rebate would reveal it.

    Killing mutation: the OSS skipping each strip's first observation, +1.36%.
    """
    v0 = _profile('0d 3m(3m)', batch=32, mcmc=8192,
                  deal=dict(BARRIER_DEAL, Barrier_Dates=MONTHLY_BARRIER)).values.mean(axis=1)[0]
    assert v0 == pytest.approx(8.4787, rel=3e-3), (
        f'inception {v0:.6f} against an independent 10.5m-path 8.4787 +- 0.0051 '
        f'({(v0 - 8.4787) / 8.4787:+.3%}) - the discrete strip is not being priced as authored')


def test_a_strip_longer_than_a_block_chunk_prices_to_an_independent_simulation():
    """The same down-and-out observed every three days - 121 observations, so its strip runs past
    the inner block's 64-row chunk - against an exact GBM walk derivus did not produce: 294m
    antithetic paths in float64, V_0 = 7.6649 +- 0.0008, against a reading of 7.650688 (-0.185%).

    5e-3 is the inner QMC count's slack, not the reference's: one inner sample at inception reads
    -0.185% at 8192 paths, -0.17% at 32768 and -0.11% at 131072, the pseudo-random arm +0.03%.

    Killing mutation: every chunk of the inner block read off the first 64 dimensions, so row
    64 + k repeats row k - -14.2% when it was read `c * sims` points further along one engine.
    """
    every_third_day = [BASE + pd.Timedelta(days=d) for d in range(3, 365, 3)]
    v0 = _profile('0d 3m(3m)', batch=32, mcmc=8192,
                  deal=dict(BARRIER_DEAL, Barrier_Dates=every_third_day)).values.mean(axis=1)[0]
    assert v0 == pytest.approx(7.6649, rel=5e-3), (
        f'inception {v0:.6f} against an independent 294m-path 7.6649 +- 0.0008 '
        f'({(v0 - 7.6649) / 7.6649:+.3%}) - a strip past one chunk is walked on the wrong law')


def _digital(H, btype='Down_And_Out'):
    return {'Object': 'EquityBarrierBinaryOption', 'Reference': 'DIG1', 'Currency': 'USD',
            'Payoff_Currency': 'USD', 'Equity': 'EQ', 'Dividends': 'EQ', 'Discount_Rate': 'USD',
            'Equity_Volatility': 'EQ', 'Buy_Sell': 'Buy', 'Option_Type': 'Call',
            'Strike_Price': 100.0, 'Expiry_Date': BASE + pd.Timedelta(days=365),
            'Cash_Payoff': 100.0, 'Barrier_Type': btype, 'Barrier_Price': H,
            'Settlement_Date': BASE + pd.Timedelta(days=365),
            'Barrier_Dates': [BASE + pd.Timedelta(days=d) for d in range(30, 366, 30)]}


def test_digital_terminal_step_is_integrated_not_sampled():
    """A digital's payoff was an indicator on the DRAWN terminal spot, whose derivative is zero
    almost everywhere, so the density term that is most of a digital's delta and vega never reached
    the tape. The barrier is out of reach so the outer latch never fires and this isolates the
    terminal step: AAD reported EXACTLY zero, with the equity, vol and dividend factors absent from
    the report rather than showing zero rows - a MISSING number, `report_grad` dropping a factor
    whose `.grad` is None. Now 0.00% at 0.01% flatness, and the equity and vol rows reported. The
    same deal WITH a live barrier still disagrees (33.7%) - the outer latch's flux, which is the
    boundary correction's job rather than this terminal step's.

    Killing mutation: the terminal step sampled - the digital's indicator on the drawn spot.
    """
    deal = _digital(1e-6)
    # the OSS forks an inner Monte Carlo per outer path, so the outer batch stays small
    kw = dict(batch=1024, mcmc=256)
    aad = _cva(SPOT, deal, gradient=True, **kw)
    assert abs(aad) > 1e-6, 'a digital must have a spot delta'
    r = ladder(price=lambda s: _cva(s, deal, False, **kw), aad=aad, base=SPOT,
               rungs=(5e-4, 1e-3, 2e-3, 5e-3))
    assert r.agrees(tol=0.02), f'digital terminal step is not being integrated\n{r}'
    # the surface is authored under the PRE-TAG name and the gradient comes back tagged:
    # `resolve_factor_key` accepts the old spelling on read, and the gradient is filed under the
    # type the resolver asked for
    factors = {str(i[0]).split('.')[0] for i in _cva(SPOT, deal, 'frame', **kw).index}
    for needed in ('EquityPrice', 'EquityPriceVol'):
        assert needed in factors, f'{needed} missing from the greeks report; got {sorted(factors)}'


def test_a_knock_out_rebate_is_settled_absolute_cash_and_mirrors_with_the_side():
    """Three defects in one field. The knock-out rebate was PRICED - `sim_spot_oss` accrues it into
    the knocking row's mtm - but never settled, the settled cash bit-identical to the same deal with
    no rebate. It was scaled wrongly: `pv_barrier_option` reads `Cash_Rebate` as ABSOLUTE cash while
    everything `sim_spot_oss` returns is scaled by nominal, so one field meant Units times more cash
    under discrete monitoring than continuous. And `nominal` ALREADY carries `Buy_Sell` here, so
    dividing the rebate by it cancelled the direction - a seller who must PAY on knock-out booking
    a receipt. Buy and Sell are exact mirror images, in price and in settled cash.

    Killing mutation: the rebate read per unit, scaling with `Units`.
    """
    kw = dict(Barrier_Price=95.0, Barrier_Dates=MONTHLY_BARRIER)
    buy = np.subtract(_totals(dict(kw, Cash_Rebate=5.0)), _totals(dict(kw, Cash_Rebate=0.0)))
    assert buy[0] > 0.0 and buy[1] > 0.0, 'a bought knock-out prices and settles its rebate'
    sell = np.subtract(_totals(dict(kw, Buy_Sell='Sell', Cash_Rebate=5.0)),
                       _totals(dict(kw, Buy_Sell='Sell', Cash_Rebate=0.0)))
    assert sell[0] == pytest.approx(-buy[0], rel=1e-9), (buy, sell)
    assert sell[1] == pytest.approx(-buy[1], rel=1e-9), (buy, sell)
    units = np.subtract(_totals(dict(kw, Units=2.0, Cash_Rebate=5.0)),
                        _totals(dict(kw, Units=2.0, Cash_Rebate=0.0)))
    assert units[0] == pytest.approx(buy[0], rel=1e-9), (
        f'a cash rebate must not scale with Units: {buy[0]:.4f} at 1, {units[0]:.4f} at 2')


def test_a_barrier_date_on_expiry_settles_its_rebate_once():
    """The per-observation settle fires on every barrier date and the single settle after the loop
    pays the whole terminal row, which already contains that rebate. A deal whose last barrier date
    IS expiry therefore paid twice - and `instruments.py` unions `Expiry_Date` into the observation
    dates, so that is the common case. `pv_barrier_option` guards the same double count with
    `expiry[index] > 0.0`. The strike is out of reach, so the rebate is the only cash in the run.

    Killing mutation: the terminal row's rebate settled by the loop as well as after it.
    """
    expiry = BASE + pd.Timedelta(days=365)
    at_expiry = _totals({'Barrier_Price': 95.0, 'Cash_Rebate': 5.0, 'Strike_Price': 1e6,
                          'Barrier_Dates': [expiry]})[1]
    earlier = _totals({'Barrier_Price': 95.0, 'Cash_Rebate': 5.0, 'Strike_Price': 1e6,
                        'Barrier_Dates': [BASE + pd.Timedelta(days=330)]})[1]
    assert at_expiry < 1.5 * earlier, (
        f'rebate settled twice: {at_expiry:.2f} against {earlier:.2f} for a barrier date 35 days '
        f'earlier - a single count differs only by the extra knock-out probability')


def test_an_unknown_payment_timing_is_refused():
    """`Payment_Timing` has two values and the pricer's closed-form chain has two branches, no else.
    A third used to price as whatever the last branch assignment left behind; it refuses at
    CONSTRUCTION now, before `reset` and `add_grid_dates` read the field.

    Killing mutation: the refusal dropped, the third value priced.
    """
    with pytest.raises(ValueError, match='Payment_Timing'):
        _profile('0d 3m(3m)', deal=dict(ONE_TOUCH, Payment_Timing='AtMaturity'))


# --------------------------------------------------------------------------------------------
# THE ZERO-LENGTH STEP
#
# A reporting row that IS an observation date opens the OSS strip with `dt = 0`. That step used to
# be SIMULATED at the variance floor - a 1% sigma kick with an Ito correction - so the survival it
# decided was a `Phi` around the level rather than the level itself. `instruments.py` unions the
# barrier dates into the reval dates, so on a discrete barrier EVERY reporting row past inception
# is one of these.
# --------------------------------------------------------------------------------------------
ZERO_STEP_REBATE = 7.0


def _one_date_rebate_run(batch=512, mcmc=64):
    """A knock-out whose only observation is expiry, with a strike out of reach: the expiry row is
    the rebate and nothing else."""
    c = _cfg()
    c.deals['Deals']['Children'] = [{'Instrument': construct_instrument(dict(
        BARRIER_DEAL, Barrier_Price=100.0, Strike_Price=1000.0, Units=1.0,
        Cash_Rebate=ZERO_STEP_REBATE,
        Barrier_Dates=[BASE + pd.Timedelta(days=365)]), {})}]
    _, out = derivus.run_cmc(c, prec=DTYPE, overrides={
        'Run_Date': BASE.strftime('%Y-%m-%d'), 'Time_grid': '0d 1y(3m)', 'Batch_Size': batch,
        'Simulation_Batches': 1, 'Random_Seed': 1, 'Currency': 'USD', 'Tenor_Offset': 0.0,
        'MCMC_Simulations': mcmc, 'Deflation_Interest_Rate': 'USD'})
    return np.asarray(out['Results']['mtm'], dtype=float)


def test_a_rebate_read_at_an_observation_date_row_is_exact():
    """The zero-length step resolves by comparing the row's own spot to the level, so the mark at
    that row is a TWO-POINT distribution: a knocked scenario is worth exactly `Cash_Rebate`,
    undiscounted because the fixing is the row, and a surviving one exactly nothing. There is no
    third value for a sampled indicator to put between them.

    Simulated at the floor it read 127 distinct values across 512 scenarios, smeared upward from
    1.1e-14, and left 41.8% of them marking the rebate exactly where the level knocks 54.7%.

    The strike is out of reach and the only observation is expiry, so the rebate is the whole mark
    and nothing has to be believed about the option leg.

    Killing mutation: the zero-length step simulated at the variance floor - the exact branch and
    the zero-length mask both dropped, since either alone resolves the step exactly.
    """
    last = _one_date_rebate_run()[-1]
    values = np.unique(last)
    assert set(values.tolist()) == {0.0, ZERO_STEP_REBATE}, (
        f'{len(values)} distinct marks where the level allows two: {values[:8]}')
    assert 0.1 < float((last == ZERO_STEP_REBATE).mean()) < 0.9, (
        'the fixture knocked out every scenario or none - it gates nothing')


def test_a_row_that_is_not_an_observation_date_is_untouched():
    """The other half of the same statement: nothing moves where there is no zero-length step.

    Inception has every observation ahead of it, so its first interval is positive and the exact
    branch is never taken. Pinned as a CMC row rather than a base valuation, whose reduction is not
    bit-stable run to run on this device (8.402328989083189 against ...82585 over two processes).

    Every row AFTER it is an observation date, and those moved by -0.13% to +0.46% over
    four discrete-barrier profiles (knock-out, knock-out with rebate, knock-in, and one whose dates
    sit off the reporting clock), the knock-in moving most because its parity leg reads the
    survival twice.

    Killing mutation: every monitored step resolved as if it were zero-length.
    """
    monthly = [BASE + pd.DateOffset(months=k) for k in range(1, 12)]
    profile = _profile('0d 1y(1m)', deal=dict(BARRIER_DEAL, Barrier_Price=90.0,
                                              Barrier_Dates=monthly), batch=1024, mcmc=128)
    rows = np.asarray(profile, dtype=float)
    assert float(rows[0].mean()) == 8.444798793160231, float(rows[0].mean())
    assert not np.array_equal(rows[1], rows[0]), 'the profile is flat - nothing is being compared'
