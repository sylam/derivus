"""A barrier's survival between grid dates is a probability, not an endpoint check.

`pv_barrier_option` prices the remaining life with Reiner-Rubinstein, which assumes CONTINUOUS
monitoring. The historical path state asked only whether the spot sat beyond the barrier ON a grid
date, so every path that crossed and came back counted as still alive - a state inconsistent with
the formula being applied to it, and one that got worse the coarser the grid.

The gate needs no external reference. Set r = q = 0 and the simulation drift to match, and the
option value is a MARTINGALE: E[MTM_t] must equal the t=0 value at every t, on every grid. The t=0
row is the pure closed form because no history has accumulated yet, so the test compares the
pricer against itself and still pins the thing that was wrong. Endpoint-only survival fails it by
+12.8% at 3m and +26.8% at 9m on the quarterly grid, and the error moves with the grid, which is
what `test_bridge_is_grid_independent` reads directly.

A MARTINGALE STATISTIC IS AN EXPECTATION AND COSTS PATHS. Every gate here that reads one states
the seed distribution it was measured over, because a tolerance set against a single seed is a
coin toss dressed as a threshold - `test_discrete_barrier_is_observed_only_on_its_own_dates` was
exactly that, and it is gone: its statistic had a 1.46% seed sd against a 1% tolerance and failed
23 of 50 seeds. Where a deterministic ledger says the same thing it is used INSTEAD of the
expectation, and where the expectation is the only route the paths were raised until the noise
sat clear of the defect. No tolerance in this file was widened to make it green.
"""
import os
import sys

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

BASE = pd.Timestamp('2024-06-28')
DTYPE = torch.float64
VOL = 0.25
SPOT = 100.0


def _cfg():
    """Down-and-out call, continuously monitored, in a zero-rate zero-dividend world so that the
    value is a martingale under the simulation measure. The equity is driven by GBM, whose
    lognormal interval law is what publishes a bridge variance rate at all."""
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
    # Drift 0 with r = q = 0 makes the SIMULATED spot a martingale, which is what lets the option
    # value be one too - the pricing measure and the simulation measure have to be the same one.
    c.params['Price Models'] = {'GBMAssetPriceModel.EQ': {'Vol': VOL, 'Drift': 0.0}}
    c.params['Model Configuration'].append('EquityPrice', (), 'GBMAssetPriceModel')
    c.deals = {'Attributes': {'Reference': 'test', 'Tag_Titles': ''},
               'Deals': {'Children': [{'Instrument': construct_instrument(field, {})}]},
               'Calculation': {'Base_Date': BASE, 'Currency': 'USD'}}
    return c


ONE_TOUCH = {
    'Object': 'EquityOneTouchOption', 'Reference': 'OT1', 'Currency': 'USD',
    'Payoff_Currency': 'USD', 'Equity': 'EQ', 'Discount_Rate': 'USD', 'Equity_Volatility': 'EQ',
    'Buy_Sell': 'Buy', 'Cash_Payoff': 100.0, 'Payoff_Type': 'Cash', 'Barrier_Price': 90.0,
    'Barrier_Type_One': 'Down', 'Expiry_Date': BASE + pd.Timedelta(days=365),
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
    """CVA, and its AAD gradient when asked for. A counterparty is what gives the barrier a
    sensitivity worth measuring: the exposure profile is where the touch state accumulates, which
    base valuation - one deal-time row, no interval, no history - structurally cannot show."""
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
    return float(g.loc[[i for i in g.index if 'EquityPrice' in str(i[0])][0]])


def _analytic_touch_probability():
    """P(the minimum of the GBM breaches the barrier before expiry), by reflection. Independent of
    derivus - the whole point is that the pricer is checked against something it did not produce."""
    import math
    mu, sig, b = -0.5 * VOL ** 2, VOL, math.log(90.0 / SPOT)
    phi = lambda x: 0.5 * math.erfc(-x / math.sqrt(2.0))
    return phi((b - mu) / sig) + math.exp(2.0 * mu * b / sig ** 2) * phi((b + mu) / sig)


def test_variance_rate_reproduces_the_processes_own_variance():
    """The rate is only exact if it is the SIMULATION variance. A process discretises the scenario
    grid into per-step vols; a rate against elapsed time must sum back to the same total, or the
    bridge is being handed a vol that means something else - the pricing implied vol for the
    option's remaining life is exactly such a quantity, carrying the same units."""
    from derivus.stochasticprocess import GBMAssetPriceModel
    import types

    # UNEVEN scenario dates, and they must be the SCENARIO set - a single date leaves dt all zero,
    # which makes both sides of the comparison zero and the assertion true for any rate at all.
    dates = {BASE + pd.Timedelta(days=d) for d in (0, 30, 90, 365)}
    grid = utils.TimeGrid(dates, dates, dates)
    grid.set_base_date(BASE)
    p = GBMAssetPriceModel(factor=types.SimpleNamespace(param={}), param={'Vol': VOL, 'Drift': 0.0})
    p.precalculate(BASE, grid, torch.tensor([SPOT], dtype=DTYPE), None, 0)

    stepwise = float((p.vol * p.vol).sum())
    elapsed = grid.time_grid_years[-1]
    assert stepwise > 0.0 and elapsed > 0.0, 'degenerate grid - the comparison below is vacuous'
    assert p.bridge_variance_rate * elapsed == pytest.approx(stepwise, rel=1e-12), (
        'rate x elapsed must equal the variance the process actually simulates')


@pytest.mark.parametrize('grid,label', [('0d 3m(3m)', 'quarterly'),
                                        ('0d 1m(1m)', 'monthly'),
                                        ('0d 1w(1w)', 'weekly')])
def test_bridge_is_grid_independent(grid, label):
    """With r = 0 the value is a martingale, so every date on every grid must report the t=0
    price. The endpoint-only state fails this by +12.8% at 3m and +26.8% at 9m on the quarterly
    grid, and by a DIFFERENT amount on each grid - a barrier's exposure profile should not depend
    on how often the engine happens to look at it.

    THE PATHS WERE RAISED, THE TOLERANCE WAS NOT. At the 8192 this ran at, the same statistic read
    max 6.19% / 7.08% / 6.31% (quarterly / monthly / weekly) over seeds 1-20 and tripped its own 4%
    on 1, 2 and 3 of the first 12 - it was pinned to seed 1, not proven. At 65536 the readings are
    mean 0.72% / 0.84% / 0.98%, max 1.15% / 1.64% / 2.28% over the same 20 seeds, so 4% is 1.8x
    above the worst noise and the defect sits 2.7x-7.2x above 4%: endpoint-only survival (the
    Brownian-bridge variance forced falsy in `utils.barrier_touched`) reads 28.60% / 18.87% /
    10.96% at seed 1. The cost is ~1.1s a grid.

    MUTATIONS: endpoint-only survival => KILLED on all three grids. Note the defect SHRINKS as the
    grid refines - the endpoint test misses less when it looks more often - so the weekly grid is
    the binding rung and is what the batch was sized on."""
    mtm = _profile(grid, batch=65536)
    t0 = mtm.values[0].mean()
    assert t0 > 0.0, 'a bought down-and-out call should be worth something at inception'
    drift = np.abs(mtm.values.mean(axis=1) - t0) / t0
    assert drift.max() < 0.04, (
        f'{label}: exposure profile drifts {drift.max():.1%} from the t=0 value {t0:.4f} '
        f'at row {drift.argmax()} of {len(drift)} - survival is not being carried as a probability')


def test_one_touch_paid_at_expiry_holds_its_value_until_then():
    """A one-touch paying at EXPIRY owes the nominal on every path that has touched, so between
    the touch and expiry such a path holds a CERTAIN claim worth its discounted value. That value
    was carried as zero: a touched deal reported nothing for the rest of its life and then jumped
    to the nominal on the last date. With r=0 the correct value is a martingale, and equals the
    nominal times the touch probability at t=0."""
    mtm = _profile('0d 3m(3m)', deal=dict(ONE_TOUCH, Payment_Timing='Expiry'))
    v = mtm.values.mean(axis=1)
    expected = 100.0 * _analytic_touch_probability()
    assert v[0] == pytest.approx(expected, rel=2e-3), (
        f'inception value {v[0]:.3f} should be the analytic {expected:.3f}')
    assert np.abs(v - v[0]).max() / v[0] < 0.03, (
        f'value paid at expiry is not being held: profile {np.round(v, 2)}')


def test_one_touch_paid_on_touch_settles_and_leaves():
    """The counterpart, and the reason the fix above is confined to Expiry timing: paid ON touch,
    the cash settles and the path stops carrying it, so this profile SHOULD decay. Both timings
    must still agree at inception, since with r=0 there is nothing to discount between them."""
    on_touch = _profile('0d 3m(3m)', deal=dict(ONE_TOUCH, Payment_Timing='Touch'))
    at_expiry = _profile('0d 3m(3m)', deal=dict(ONE_TOUCH, Payment_Timing='Expiry'))
    v = on_touch.values.mean(axis=1)
    assert v[0] == pytest.approx(at_expiry.values[0].mean(), rel=2e-3), (
        'with r=0 the two payment timings are worth the same at inception')
    assert v[-1] < 0.25 * v[0], f'paid-on-touch value should run off, got {np.round(v, 2)}'


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
    (dict(ONE_TOUCH, Payment_Timing='Expiry'), 'one_touch')])
def test_aad_delta_matches_bump_and_reprice(deal, label):
    """The gradient has to be the derivative of the value actually reported. Under common random
    numbers - same seed, so the same normals, the draws depending on seed and factor ordering
    rather than on the spot - a central difference estimates the same derivative without touching
    the tape, so agreement is evidence and not a restatement.

    An INDICATOR has zero derivative almost everywhere, so the knock-out channel contributed
    nothing and AAD reported the wrong number while looking perfectly well-behaved: measured 9-19%
    off for the barrier and 31-44% off for the one-touch, and - the discriminating signal - the
    ladder SCATTERED instead of converging, because shrinking the bump changes how many paths sit
    on the far side of the jump rather than refining a limit. Carrying survival as a probability
    gives 0.00% disagreement at 0.00% flatness."""
    aad = _cva(SPOT, deal, gradient=True)
    assert abs(aad) > 1e-6, 'a barrier with a live knock-out should have a spot delta'
    r = ladder(price=lambda s: _cva(s, deal, False), aad=aad, base=SPOT,
               rungs=(2e-4, 5e-4, 1e-3, 2e-3))
    assert r.agrees(tol=0.02), (
        f'{label}: a channel through which spot moves the value is not being differentiated\n{r}')


MONTHLY_BARRIER = [BASE + pd.Timedelta(days=d) for d in range(30, 366, 30)]


def _cashflow_run(deal_overrides, batch, mcmc, grid='0d 1m(1m)'):
    """One discrete-barrier run with its cash ledger, returned as the FRAMES rather than a total:
    which row a settlement lands on is itself under test, and a sum over rows cannot see it."""
    c = _cfg()
    c.deals['Deals']['Children'] = [{'Instrument': construct_instrument(
        dict(BARRIER_DEAL, **deal_overrides), {})}]
    _, out = derivus.run_cmc(c, prec=DTYPE, overrides={
        'Run_Date': BASE.strftime('%Y-%m-%d'), 'Time_grid': grid, 'Batch_Size': batch,
        'Simulation_Batches': 1, 'Random_Seed': 1, 'Currency': 'USD', 'Tenor_Offset': 0.0,
        'MCMC_Simulations': mcmc, 'Generate_Cashflows': 'Yes', 'Deflation_Interest_Rate': 'USD'})
    return out['Results']['mtm'], out['Results']['cashflows']


def _totals(deal_overrides, batch=512, mcmc=128):
    """Inception price and every currency's settled cash, for the gates that only need magnitudes."""
    mtm, cf = _cashflow_run(deal_overrides, batch, mcmc)
    return (mtm.values.mean(axis=1)[0], sum(float(np.nansum(v.values)) for v in cf.values()))


def _rebate_run(rebate, units):
    return _totals(dict(Barrier_Price=95.0, Cash_Rebate=rebate, Units=units,
                        Barrier_Dates=MONTHLY_BARRIER), batch=2048, mcmc=256)


@pytest.mark.parametrize('grid', ['0d 1m(1m)', '0d 2d 1w(1w) 3m(1m)'])
def test_discrete_barrier_is_observed_only_on_its_own_dates(grid):
    """A DISCRETELY monitored barrier is observed on the dates its terms name, and nowhere else.

    pv_discrete_barrier_option latched the crossing with a cumsum over every MTM row of each block,
    so on this fixture it monitored 37 reporting rows against 12 barrier dates - knocking scenarios
    out on dates the deal never observes, monitoring expiry although BarrierDates flags it -1, and
    (because the guard tested samples passed BEFORE the block) missing the first barrier date
    entirely while lagging one block thereafter.

    THIS GATE USED TO BE A MARTINGALE CHECK AND WAS NEVER A GATE. It read the profile mean against
    inception at 4096 paths against a 1% tolerance; measured over 50 seeds that statistic has mean
    +0.35%, sd 1.46%, range -3.78% to +2.98%, and |stat| > 1% on 23 of the 50. Seed 1 reads -1.40%,
    so the file was red for a reason that had nothing to do with the pricer. Averaging seeds cannot
    rescue it cheaply either: 8 seeds only reach sd 0.34% and cost 8 runs, against a defect of the
    same order.

    THE SAME STATEMENT IS DETERMINISTIC IN THE PRICER'S OWN LEDGER. With `Cash_Rebate=1.0` and
    `Units=1.0` the knock-out rebate is one unit of ABSOLUTE cash per path, so `Generate_Cashflows`
    writes a bare 0/1 indicator of "this path knocked out here", carrying the same `barrier_hit`
    latch the price is built from. Four exact facts, no expectation anywhere:

      the settled rows ARE the authored barrier dates - the anchor is the deal's own date list,
      every entry is EXACTLY 0.0 or 1.0 and a path settles at most once,
      a knocked-out path is worth EXACTLY 0.0 on every later row (`hit_value` is zeros for a
        knock-out) and strictly positive on every row up to and including its knock-out,
      and it is owed EXACTLY 0.0 by the terminal settle.

    The last two are what stop this being a cash-date check: they read `row_barrier_hit`, the mask
    that prices the block, and tie it to the cash. Both grids assert bit-identically, which is the
    grid-independence the martingale statistic was groping at - 24 rows against 12 barrier dates on
    one, 37 against 12 on the other. Nothing here is an expectation, so nothing here is seed-luck:
    measured, seeds 1-10 pass on both grids, where the statistic it replaces failed 23 of 50.

    MUTATIONS (in-process source rewrite of the pricer; the ledger is read at 2048 paths), each
    verdict below is this test failing on both grids:
      the historical cumsum-every-mtm-row form, guard on the observations passed BEFORE the block
        => KILLED on the DATE assertion: 2024-07-28, the first barrier date, never settles. With
        the dates assertion lifted it goes on to fail the value assertion too (1657 cells). Its
        profile statistic reads -3.54%, inside one seed sd of the -2.79% the old prose quoted -
        which is the measure of what that tolerance was worth.
      the settle moved one row earlier => KILLED on dates on both grids, and it also leaves all
        1504 knocked-out paths carrying value after their own settle
      `newly_hit` dropped, so a crossed path re-settles on every later date => KILLED, 12 per path
      the PRICE mask alone cumsummed over the block, cash ledger untouched => KILLED on the VALUE
        assertion (1678 cells monthly, 5007 fine) - this is the mutant that proves the mtm half is
        load-bearing rather than decorative, and its profile statistic moves only -1.40% -> -1.97%,
        0.4 of one seed sd, so the old gate could not have seen it at any tolerance
      the OSS skipping the first observation of its strip => KILLED (18 cells / 6 cells)
      CONTROL, the latch's `|` commuted => SURVIVED, nothing moves
      LIMITS OF THE GATE, both SURVIVED and both for reasons worth knowing: monitoring expiry at
        the latch (`BarrierDates[i] != 0`) is INERT in the current shape - the last block's
        `barrier_hit` has no later block to inform and its settle is excluded as the terminal row,
        so that third of the original defect is unobservable here and needed the cumsum form to
        bite. Monitoring expiry INSIDE the OSS (`isBarrierDate_block[j] != 0`) is invisible by
        construction: this gate reads the outer latch's date structure and the exact zero/positive
        pattern it produces, and an inner-fixing flag moves magnitudes, not either of those.
        Magnitude is `test_discrete_monitoring_prices_to_an_independent_simulation`'s job - which
        survives this one too, for its own fixture reason, recorded there."""
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
    """What the martingale statistic could never say: the twelve observations are priced RIGHT, not
    merely priced consistently with themselves. Inception carries no history - every scenario is
    still alive - so this is the OSS's analytic treatment of the whole barrier strip, against a
    simulation derivus did not produce.

    REFERENCE: 10.5m paths of the same discrete down-and-out (S=K=100, H=90, sigma=25%, r=q=0,
    twelve monthly observations on ACT_365, which is the clock the OSS's own fixings use), with the
    five-day terminal stub past the last observation integrated in closed form rather than sampled:
    V_0 = 8.4787 +- 0.0051. derivus reads 8.475681, i.e. -0.036%, and it is deterministic in
    Random_Seed (identical to 6dp on seeds 1-5) because the inner OSS draws a Sobol sequence and
    the t=0 scenario spot is the spot.

    3e-3 is 8x the measured gap, and the slack is not the reference's - it is the inner QMC count's:
    the same configuration reads +0.80% at MCMC_Simulations=128 and -0.10% at 512, so the batch and
    the sim count are pinned and a sub-0.3% pricing error would not be seen here.

    MUTATIONS: the OSS skipping the first observation of the strip => KILLED (+1.41%, 4.7x the
    tolerance). LIMIT: the OSS monitoring expiry too (`isBarrierDate_block[j] != 0`) => SURVIVED,
    for a fixture reason worth knowing - the barrier at 90 is BELOW the strike at 100, so a path
    the extra observation would knock out already pays zero, and only a rebate would reveal it."""
    v0 = _profile('0d 3m(3m)', batch=1024, mcmc=256,
                  deal=dict(BARRIER_DEAL, Barrier_Dates=MONTHLY_BARRIER)).values.mean(axis=1)[0]
    assert v0 == pytest.approx(8.4787, rel=3e-3), (
        f'inception {v0:.6f} against an independent 10.5m-path 8.4787 +- 0.0051 '
        f'({(v0 - 8.4787) / 8.4787:+.3%}) - the discrete strip is not being priced as authored')


def test_discrete_barrier_rebate_is_paid_and_is_absolute_cash():
    """Two defects in one field. The knock-out rebate was PRICED - sim_spot_oss accrues it, and it
    is in the mtm of the row that knocks out - but never settled: hit_value is zero from the next
    row on and the single cash_settle fires on the last row only, so the total settled cash was
    bit-identical to the same deal with no rebate at all.

    And it was scaled wrongly. pv_barrier_option reads Cash_Rebate as ABSOLUTE cash - it hands the
    closed form cash_rebate/nominal and multiplies back - while everything sim_spot_oss returns is
    scaled by nominal, so the same field on the same deal class meant Units times more cash under
    discrete monitoring than under continuous. Doubling Units must not double a cash rebate."""
    p0, c0 = _rebate_run(0.0, 1.0)
    p5, c5 = _rebate_run(5.0, 1.0)
    assert c5 - c0 > 0.0, 'the rebate is priced into the mtm but never settled'

    p0x, _ = _rebate_run(0.0, 2.0)
    p5x, _ = _rebate_run(5.0, 2.0)
    assert (p5x - p0x) == pytest.approx(p5 - p0, rel=1e-9), (
        f'a cash rebate must not scale with Units: adds {p5 - p0:.4f} at Units=1 but '
        f'{p5x - p0x:.4f} at Units=2')


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
    almost everywhere - so the density term that is most of a digital's delta and vega never
    reached the tape at all.

    The barrier is put out of reach so the outer already-hit latch never fires and this isolates
    the terminal step. Before the fix AAD reported EXACTLY zero here, and the equity, vol and
    dividend factors were absent from the greeks report rather than showing zero rows - a silent
    total loss of sensitivity. Now: 0.00% against bump-and-reprice at 0.01% flatness.

    NOTE the same deal WITH a live barrier still disagrees (33.7% at 9.96% flatness). That residual
    is the outer barrier_hit latch in pv_discrete_barrier_option, a genuine jump in the value
    function needing the boundary-flux machinery, not this terminal step - which is exactly why
    this gate uses an unreachable barrier."""
    deal = _digital(1e-6)
    # the OSS forks an inner Monte Carlo per outer path, so the outer batch stays small here
    kw = dict(batch=1024, mcmc=256)
    aad = _cva(SPOT, deal, gradient=True, **kw)
    assert abs(aad) > 1e-6, 'a digital must have a spot delta'
    r = ladder(price=lambda s: _cva(s, deal, False, **kw), aad=aad, base=SPOT,
               rungs=(5e-4, 1e-3, 2e-3, 5e-3))
    assert r.agrees(tol=0.02), f'digital terminal step is not being integrated\n{r}'


def test_digital_reports_its_equity_and_vol_factors():
    """The failure mode was not a wrong number but a MISSING one: with a zero gradient the factor's
    .grad is None and pricing.report_grad drops it, so the risk report simply had no equity row."""
    c = _cfg()
    c.params['Price Factors']['SurvivalProb.CPTY'] = {
        'Recovery_Rate': 0.4, 'Curve': utils.Curve([], [[0.0, 0.0], [10.0, 0.4]])}
    c.deals['Deals']['Children'] = [{'Instrument': construct_instrument(_digital(1e-6), {})}]
    _, out = derivus.run_cmc(c, prec=DTYPE, overrides={
        'Run_Date': BASE.strftime('%Y-%m-%d'), 'Time_grid': '0d 3m(3m)', 'Batch_Size': 256,
        'Simulation_Batches': 1, 'Random_Seed': 1, 'Currency': 'USD', 'Tenor_Offset': 0.0,
        'MCMC_Simulations': 128, 'Deflation_Interest_Rate': 'USD', 'Gradient_Variables': 'Factors',
        'Credit_Valuation_Adjustment': {
            'Calculate': 'Yes', 'Counterparty': 'CPTY', 'Deflate_Stochastically': 'No',
            'Stochastic_Hazard_Rates': 'No', 'Gradient': 'Yes'}})
    factors = {str(i[0]).split('.')[0] for i in out['Results']['grad_cva']['Gradient'].index}
    # the surface above is authored under the PRE-TAG name and the gradient comes back tagged:
    # `resolve_factor_key` accepts the old spelling on read, and the type the resolver asked for is
    # what the factor-keyed gradient is filed under. Both halves of the leniency, in one assertion.
    for needed in ('EquityPrice', 'EquityPriceVol'):
        assert needed in factors, f'{needed} missing from the greeks report; got {sorted(factors)}'


def test_a_sold_knock_out_pays_its_rebate_rather_than_receiving_it():
    """`nominal` in the discrete pricer ALREADY carries Buy_Sell, unlike pv_barrier_option where
    buy_or_sell is a separate factor. Dividing the rebate by it therefore cancelled the direction,
    and every rebate leg came back as +cash_rebate whichever way the deal was done - a seller who
    must PAY on knock-out booked it as a receipt, in the reported price and in the settled cash.

    Buy and Sell must be exact mirror images. The original rebate gate only ever ran Buy, which is
    why this survived it."""
    kw = dict(Barrier_Price=95.0,
              Barrier_Dates=[BASE + pd.Timedelta(days=d) for d in range(30, 361, 30)])
    buy = np.subtract(_totals(dict(kw, Buy_Sell='Buy', Cash_Rebate=5.0)),
                      _totals(dict(kw, Buy_Sell='Buy', Cash_Rebate=0.0)))
    sell = np.subtract(_totals(dict(kw, Buy_Sell='Sell', Cash_Rebate=5.0)),
                       _totals(dict(kw, Buy_Sell='Sell', Cash_Rebate=0.0)))
    assert buy[0] > 0.0 and buy[1] > 0.0, 'a bought knock-out receives its rebate'
    assert sell[0] == pytest.approx(-buy[0], rel=1e-9), (
        f'rebate does not flip with direction: buy {buy[0]:+.4f} vs sell {sell[0]:+.4f}')
    assert sell[1] == pytest.approx(-buy[1], rel=1e-9), (
        f'settled rebate cash does not flip: buy {buy[1]:+.2f} vs sell {sell[1]:+.2f}')


def test_a_barrier_date_on_expiry_settles_its_rebate_once():
    """The per-observation settle fires on every barrier date, and the single settle after the loop
    pays the whole terminal row - which already contains that rebate, because sim_spot_oss accrued
    it. A deal whose last barrier date IS expiry therefore paid twice, and instruments.py unions
    Expiry_Date into the observation dates, so that is the common case rather than a corner.

    pv_barrier_option guards the identical double count with `expiry[index] > 0.0`. Strike is put
    out of reach here so the rebate is the only cash in the run."""
    expiry = BASE + pd.Timedelta(days=365)
    at_expiry = _totals({'Barrier_Price': 95.0, 'Cash_Rebate': 5.0, 'Strike_Price': 1e6,
                          'Barrier_Dates': [expiry]})[1]
    earlier = _totals({'Barrier_Price': 95.0, 'Cash_Rebate': 5.0, 'Strike_Price': 1e6,
                        'Barrier_Dates': [BASE + pd.Timedelta(days=330)]})[1]
    assert at_expiry < 1.5 * earlier, (
        f'rebate settled twice: {at_expiry:.2f} against {earlier:.2f} for a barrier date 35 days '
        f'earlier - a single count differs only by the extra knock-out probability')


@pytest.mark.parametrize('freq_days,label', [(0, 'continuous'), (30, 'monthly'), (7, 'weekly')])
def test_the_bridge_honours_the_monitoring_frequency(freq_days, label):
    """A discretely monitored barrier is priced by a CONTINUOUS closed form against a barrier
    shifted away from the live region (Broadie-Glasserman-Kou). The bridge was handed the RAW
    barrier while the formula three lines later priced the shifted one, so the path state monitored
    continuously a barrier the product observes monthly, and the two disagreed about the same deal.

    With r = q = 0 the value is a martingale at ANY monitoring frequency. Measured before the fix:
    monthly monitoring decayed -11.58% over the profile; continuous was unaffected, which is why
    the original gate - written at the default 0d - could not see it.

    Same correction as `test_bridge_is_grid_independent` and for the same reason: at 8192 this
    statistic read max 7.08% / 6.22% / 6.55% (continuous / monthly / weekly) over seeds 1-12 and
    tripped its own 5% on 2, 1 and 2 of them. At 65536, over seeds 1-20:

        continuous  mean 0.84% max 1.64%   monthly  mean 0.74% max 1.45%   weekly  mean 0.78% max 1.51%

    so 5% is 3.0x above the worst noise, and endpoint-only survival reads 18.87% / 9.31% / 13.41%
    at seed 1 - 1.9x above the tolerance at the binding monthly rung.

    Inception rises with coarser monitoring (8.485 monthly against 7.176 continuous) because fewer
    observations means fewer chances to knock out - a second reason a frequency-blind gate is weak.

    MUTATIONS: endpoint-only survival => KILLED at all three frequencies."""
    deal = dict(BARRIER_DEAL, Barrier_Monitoring_Frequency=pd.DateOffset(days=freq_days))
    v = _profile('0d 1m(1m)', deal=deal, batch=65536).values.mean(axis=1)
    drift = np.abs(v - v[0]).max() / v[0]
    assert drift < 0.05, (
        f'{label}: profile drifts {drift:.1%} from inception {v[0]:.4f} - the bridge and the '
        f'closed form are testing different barriers\n{np.round(v, 3)}')


def test_an_unknown_payment_timing_is_refused():
    """`Payment_Timing` has exactly two values and the pricer's closed-form chain has exactly two
    branches - no else. A third value used to price as whatever the last branch assignment left
    behind; it now refuses at CONSTRUCTION, before `reset` and `add_grid_dates` read the field."""
    with pytest.raises(ValueError, match='Payment_Timing'):
        construct_instrument(dict(ONE_TOUCH, Payment_Timing='AtMaturity'), {})
