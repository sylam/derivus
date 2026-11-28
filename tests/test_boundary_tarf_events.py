"""The two decisions inside the TARF one-step-survival pricer that jump, and reach the tape.

`pv_MC_Tarf` runs an INNER Monte Carlo of `2 * MCMC_Simulations` paths per reporting row, and two
things inside it are decided on simulated state and carry a real value jump:

  KNOCK-IN   the OTM leg pays only where the fixing crosses `barrier`. Not smoothable - the client
             either owes the leveraged leg or does not - and the jump is `N_otm * |K - barrier|`,
             which on this fixture is 60,000 on a deal worth 8,271.
  TARGET PIN `q = remaining_target == 0` zeroes the survival weight, so the deal is worth nothing
             from there on. `calc_accum_value` ends in `.clamp(max=targetValue)`, so this exact
             float equality fires on a POSITIVE-MEASURE set - measured below at 27.7%-61.3% of
             outer paths - and it is a redemption, not a rounding artifact.

A REACHABILITY TRAP: `LeverageNotional` (N_otm) defaults to 0, and BOTH sites are dead there.
  `cf_otm = relu(-intr) * N_otm * barrier_hit` multiplies the knock-in by zero, and the target pin
  becomes CONTINUOUS: as `remaining_target -> 0` the KO term, the clamped intrinsic and every
  surviving cashflow go to zero with it, so zeroing the weight costs nothing. Measured at the same
  61.3% firing rate with `LeverageNotional=0`: the uncorrected AAD agrees with bump-and-reprice to
  0.00%-1.14%.

A knock-in on a fixing the inner walk simulates is integrated against the interval's law, value and
derivative (`lognormal_fired_gain`), under either estimator; one on an OBSERVED fixing, the
scenario's own print and so every inner path's, registers a boundary correction.

THE TARGET PIN IS GATED ON FOUR FIXINGS, where its ladder resolves. On six, its registration -
the fired flags, the branches reconstructing the reported profile, the pending head of a row that
redeems inside its own strip - was read by spying on the pricer, and the TARF emits no
reconstruction organ as the accumulator and the extendable forward do. Uncorrected the pin read 27%
short with neither estimator nor oracle resolving better than ~10%, so no CRN ladder stands in for
it there.

WHY THE LADDERS START AT 3e-4. Differencing across a jump does not converge as h shrinks - it
changes how many paths sit on the wrong side. Below ~1e-4 the CRN readings scatter over 1/h
(-69k, -1.59e6, -525k, -222k, -152k as h goes 1e-6 -> 1e-4). The rungs kept are where the oracle
plateaus.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch

import derivus
from derivus import run_baseval, utils
from derivus.config import Config
from derivus.instruments import construct_instrument
from crn_ladder import ladder

BASE = pd.Timestamp('2024-06-28')
DTYPE = torch.float64
SPOT = STRIKE = 0.65
SIGMA = 0.12
N1 = 1_000_000.0
N2 = 2_000_000.0          # LeverageNotional -> the OTM leg. Zero kills BOTH sites; see the header.
BARRIER = 0.62            # < K, so the knock-in bites where the OTM leg pays
MONTHLY = [30, 60, 90]
BIMONTHLY = [60 * (i + 1) for i in range(6)]
UNREACHABLE_TARGET = 1e9


def _price_factors(spot):
    return {
        'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Priority': 1, 'Spot': 1.0},
        'FxRate.AUD': {'Domestic_Currency': 'USD', 'Interest_Rate': 'AUD', 'Priority': 1,
                       'Spot': spot},
        'InterestRate.USD': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                             'Curve': utils.Curve([], [[0.0, 0.0], [5.0, 0.0]])},
        'InterestRate.AUD': {'Currency': 'AUD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                             'Curve': utils.Curve([], [[0.0, 0.0], [5.0, 0.0]])},
        'FXVol.AUD.USD': {'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness',
                          'Surface': utils.Curve([], [[m, t, SIGMA] for m in (0.5, 1.0, 1.5)
                                                      for t in (0.02, 2.0)])},
    }


def _tarf(target, fix_days, barrier=None, leverage=N2, buy_sell='Buy'):
    fix_dates = [BASE + pd.Timedelta(days=d) for d in fix_days]
    deal = {
        'Object': 'FXTARFOptionDeal', 'Reference': 'TARF1', 'Currency': 'USD',
        'Underlying_Currency': 'AUD', 'Discount_Rate': 'USD', 'FX_Volatility': 'AUD.USD',
        'Buy_Sell': buy_sell, 'Expiry_Date': fix_dates[-1], 'Underlying_Amount': N1,
        'Option_Type': 'Call', 'Strike_Price': STRIKE, 'Settlement_Style': 'Physical',
        'Option_Style': 'European', 'InvertedTarget': False, 'LeverageNotional': leverage,
        'TargetLevel': target,
        'TARF_ExpiryDates': [[d, d, None] for d in fix_dates]}
    if barrier is not None:
        deal['Barrier'] = barrier
    return deal


def _cfg(deal, spot, counterparty=False, simulate_fx=False):
    c = Config()
    c.params['System Parameters']['Base_Currency'] = 'USD'
    c.params['System Parameters']['Base_Date'] = BASE
    c.params['Price Factors'] = _price_factors(spot)
    c.params['Price Models'] = {}
    c.params['Valuation Configuration'] = {}
    if counterparty:
        c.params['Price Factors']['SurvivalProb.CPTY'] = {
            'Recovery_Rate': 0.4, 'Curve': utils.Curve([], [[0.0, 0.0], [10.0, 0.4]])}
    if simulate_fx:
        c.params['Price Models'] = {'GBMAssetPriceModel.AUD': {'Vol': SIGMA, 'Drift': 0.0}}
        c.params['Model Configuration'].append('FxRate', (), 'GBMAssetPriceModel')
    c.deals = {'Attributes': {'Reference': 'test'},
               'Deals': {'Children': [{'Instrument': construct_instrument(deal, {})}]},
               'Calculation': {'Base_Date': BASE, 'Currency': 'USD'}}
    return c


def _baseval(deal, spot=SPOT, greeks=False, sims=1 << 16):
    """(price, d(price)/d(FxRate.AUD spot)). One date, one scenario - and still a full inner MC
    underneath, which is where the knock-in is decided."""
    # the CRISP estimator declared: this whole module is about the boundary correction, and
    # the default swaps the estimator rather than correcting it, registering nothing
    overrides = {'MCMC_Simulations': sims, 'Quasi_Seed': 1, 'Branch_And_Weight': 'No',
                 'Greeks': 'First' if greeks else 'No'}
    _, out = run_baseval(_cfg(deal, spot), overrides=overrides)
    rows = out['Results']['mtm']
    price = float(rows[rows['Reference'] == 'TARF1']['Value'].iloc[0])
    grad = None
    if greeks:
        frame = out['Results']['Greeks_First']
        # two columns: 'Value' is the FACTOR LEVEL (display_val=True), the other is the gradient
        column = [x for x in frame.columns if x != 'Value'][0]
        index, = [i for i in frame.index if str(i[0]) == 'FxRate.AUD']
        grad = float(frame.loc[index, column])
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return price, grad


def _cmc(deal, spot=SPOT, gradient=False, batches=4, batch=512, mcmc=128, grid='0d 2m(2m)'):
    """(cva, mtm profile, d(cva)/d(FxRate.AUD spot))."""
    overrides = {
        'Run_Date': BASE.strftime('%Y-%m-%d'), 'Time_grid': grid, 'Batch_Size': batch,
        'Simulation_Batches': batches, 'Random_Seed': 1, 'Currency': 'USD', 'Tenor_Offset': 0.0,
        'MCMC_Simulations': mcmc, 'Deflation_Interest_Rate': 'USD', 'Generate_Cashflows': 'Yes',
        'Gradient_Variables': 'Factors',
        'Credit_Valuation_Adjustment': {
            'Calculate': 'Yes', 'Counterparty': 'CPTY', 'Deflate_Stochastically': 'No',
            'Stochastic_Hazard_Rates': 'No', 'Gradient': 'Yes' if gradient else 'No'}}
    _, out = derivus.run_cmc(
        _cfg(deal, spot, counterparty=True, simulate_fx=True), prec=DTYPE, overrides=overrides)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    grad = None
    if gradient:
        g = out['Results']['grad_cva']['Gradient']
        # absent rather than zero when the book has no exposure to the factor at all - which is
        # the case for the zero-leverage control, whose registration is still what is being read
        index = [i for i in g.index if str(i[0]) == 'FxRate.AUD']
        grad = float(g.loc[index[0]]) if index else None
    return float(out['Results']['cva']), out['Results']['mtm'].values, grad


KNOCK_IN = _tarf(UNREACHABLE_TARGET, MONTHLY, barrier=BARRIER)
# Sell, or the exposure of this book is ~0 on every path and the CVA gradient is noise
KNOCK_IN_CMC = _tarf(UNREACHABLE_TARGET, BIMONTHLY, barrier=BARRIER, buy_sell='Sell')
PIN_CMC = _tarf(0.02, BIMONTHLY, buy_sell='Sell')
# nearly spent, so at the declared bandwidth the target sits inside a kernel width of the strike;
# four fixings, so a path redeemed at one still decides the next
NEARLY_SPENT_CMC = _tarf(0.0005, BIMONTHLY[:4], buy_sell='Sell')
# two fixings, each observed on the quarterly row it settles on, a target most scenarios keep short of
OBSERVED_KNOCK_IN_CMC = _tarf(0.02, [183, 273], barrier=BARRIER, buy_sell='Sell')


# ---------------------------------------------------------------- safety: the value must not move

def test_asking_for_sensitivities_does_not_move_the_tarf_price_or_exposure():
    """BIT-identical, not approximately. A boundary correction is `gap - gap.detach()`, worth
    exactly zero forward, so this holds by construction - but the registration that feeds it does
    not: the target-pin counterfactual carries a SECOND survival weight through the same loop, and
    a second accumulator that consumed one random number would move the reported price. The base
    valuation's one scenario reads the canonical inner block; under exposure the pin has rows to
    latch across and the counterfactual weight runs the whole block loop beside the reported one.

    Killing mutation: the counterfactual's accumulator added to in place where it aliases the
    reported one.
    """
    off, _ = _baseval(KNOCK_IN, sims=1 << 14)
    on, grad = _baseval(KNOCK_IN, greeks=True, sims=1 << 14)
    assert off == on, f'price moved when sensitivities were requested: {off!r} -> {on!r}'
    assert grad is not None and abs(grad) > 0.0, 'no FX gradient was reported at all'

    cva_off, mtm_off, _ = _cmc(PIN_CMC, batches=1)
    cva_on, mtm_on, grad = _cmc(PIN_CMC, gradient=True, batches=1)
    assert np.array_equal(mtm_off, mtm_on), 'exposure moved when sensitivities were requested'
    assert cva_off == cva_on, f'cva moved: {cva_off!r} -> {cva_on!r}'
    assert grad is not None and abs(grad) > 0.0, 'no FX gradient was reported at all'


# ---------------------------------------------------------------- B1: the OTM-leg knock-in

def test_the_knock_in_gradient_matches_bump_and_reprice_under_exposure():
    """The acceptance gate for the knock-in, and the one whose oracle resolves cleanly. Under CMC a
    deal's own dates are folded into the mtm grid, so EVERY fixing date is also a reporting row: on
    it the first fixing is an observed one, whose knock-in registers, and the rest are integrated.

    Measured on this fixture on the CPU: AAD -86,709.21 against an oracle of -86,462.43 at 0.38%
    flatness, 0.28%.

    Killing mutation: the integrated knock-in leg off the tape, its value kept - the AAD reads
    -27,835.7, 68% short.
    """
    _, _, aad = _cmc(KNOCK_IN_CMC, gradient=True)
    r = ladder(price=lambda s: _cmc(KNOCK_IN_CMC, spot=s)[0], aad=aad, base=SPOT,
               rungs=(3e-4, 1e-3, 3e-3, 1e-2))
    assert r.agrees(tol=0.02), f'the knock-in flux is not reaching the tape\n{r}'


def test_the_knock_in_gradient_matches_bump_and_reprice_at_base_valuation():
    """Base valuation, where the pricer's inner MC is the ONLY simulation there is - one date, one
    scenario, 2x65536 inner paths - and every fixing is live, so the knock-in is the integral, value
    and derivative. Measured on the CPU: AAD 3,794,734 against a CRN ladder reading 3,794,968 at
    0.44% flatness, 0.01%.

    Killing mutation: the integrated knock-in leg off the tape, its value kept - the AAD reads
    1,528,370, 60% short.
    """
    _, aad = _baseval(KNOCK_IN, greeks=True)
    r = ladder(price=lambda s: _baseval(KNOCK_IN, spot=s)[0], aad=aad, base=SPOT,
               rungs=(3e-4, 1e-3, 3e-3, 1e-2))
    assert r.agrees(tol=0.02), f'the knock-in flux is not reaching the tape\n{r}'


def test_an_observed_knock_in_is_one_counterfactual_scored_across_its_jump():
    """A fixing an exposure row has observed knocks in on the scenario's own print, every inner
    path alike, so the decision moves the rows observing it by its whole jump: it registers as one
    counterfactual a fixing, those rows forked and the row it settles on declared, and the CVA is
    differenced across the jump, whose kink a derivative times the jump does not see.

    MEASURED at 16,384 paths x 4 batches, rungs 3e-3/7e-3/1.5e-2: AAD -31,244.4 against a ladder
    reading -31,396.9 at 1.15% flatness, 0.49%; seeds 2 and 3 read 0.65% and 0.89%.

    Killing mutations: the decision scored by the CVA's derivative times its jump, 1.82% (seeds 2
    and 3, 1.90% and 2.37%); its forks dropped, so it moves no row, 11.33%.
    """
    kw = dict(batches=4, batch=16384, mcmc=64, grid='0d 3m(3m)')
    _, _, aad = _cmc(OBSERVED_KNOCK_IN_CMC, gradient=True, **kw)
    r = ladder(price=lambda s: _cmc(OBSERVED_KNOCK_IN_CMC, spot=s, **kw)[0], aad=aad, base=SPOT,
               rungs=(3e-3, 7e-3, 1.5e-2))
    assert r.agrees(tol=0.012), f'an observed knock-in is misread across its jump\n{r}'


# ---------------------------------------------------------------- B2: the redemption's own margin

def test_a_nearly_spent_redemption_gradient_matches_bump_and_reprice_under_exposure():
    """The redemption's gap is its signed margin, the unclamped accrual before the fixing plus the
    fixing's SIGNED intrinsic less the target, and its flag is that margin's sign. On the clamped
    accrual, the scenarios a fixing leaves out of the money hold one gap and those that redeemed
    another, two point masses with no slope: the redeemed side's flux is lost and the kernel's
    width collapses onto them. And a scenario that redeemed holds the target, so every later
    fixing reads as fired whatever it printed and an earlier decision's counterfactual redeems
    there too.

    At 65,536 paths the CVA delta is 0.28% from a ladder flat to 0.34% on the CPU and 1.63% from
    one flat to 0.73% on the card. Four batches, because at one the flag's mutant sits 4.57% off on
    the card against 2.16%.

    Killing mutation: the flag read off the clamped accrual - 9.12% on the CPU, 6.60% on the
    card. The gap built on the clamped accrual is a noisy miss, not a clean kill: 13.50% on the
    CPU and 0.57% on the card at this seed, and between those over three seeds on each.
    """
    kw = dict(batches=4, batch=16384, mcmc=64)
    _, _, aad = _cmc(NEARLY_SPENT_CMC, gradient=True, **kw)
    r = ladder(price=lambda s: _cmc(NEARLY_SPENT_CMC, spot=s, **kw)[0], aad=aad, base=SPOT,
               rungs=(3e-3, 7e-3, 1.5e-2))
    assert r.agrees(tol=0.04), f'the redemption flux is misread beside a nearly spent target\n{r}'
