"""The two decisions inside the TARF one-step-survival pricer that jump, and reach the tape.

`pv_MC_Tarf` runs an INNER Monte Carlo of `2 * MCMC_Simulations` paths per reporting row, and two
things inside it are decided on simulated state and carry a real value jump:

  KNOCK-IN   `barrier_hit = (barrier - Sj) * callOrPut >= 0` switches the OTM leg on for one INNER
             path. Not smoothable - the client either owes the leveraged leg or does not - and the
             jump is `N_otm * |K - barrier|`, which on this fixture is 60,000 on a deal worth 8,262.
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

Under GBM the crisp estimator takes the knock-in by the conditional-p mixture (the decision's
probability one conditioning step back, spliced so the value is the indicator's and the derivative
the integral's) wherever a fixing interval is one simulated step; the kernel registration is left
for an interval that straddles a scenario node.

THE TARGET PIN HAS NO DOCUMENT-LEVEL GATE HERE. Its registration - the fired flags, the branches
reconstructing the reported profile, the pending head of a row that redeems inside its own strip -
was read by spying on the pricer, and the TARF emits no reconstruction organ as the accumulator and
the extendable forward do. Uncorrected the pin read 27% short with neither estimator nor oracle
resolving better than ~10%, so no CRN ladder can stand in for it.

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
        'VolatilityGrid.AUD.USD': {'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness',
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
    overrides = {'MCMC_Simulations': sims, 'Random_Seed': 1, 'Branch_And_Weight': 'No',
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


def _cmc(deal, spot=SPOT, gradient=False, batches=4, batch=512, mcmc=128):
    """(cva, mtm profile, d(cva)/d(FxRate.AUD spot))."""
    overrides = {
        'Run_Date': BASE.strftime('%Y-%m-%d'), 'Time_grid': '0d 2m(2m)', 'Batch_Size': batch,
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


# ---------------------------------------------------------------- safety: the value must not move

def test_asking_for_sensitivities_does_not_move_the_tarf_price_or_exposure():
    """BIT-identical, not approximately. A boundary correction is `gap - gap.detach()`, worth
    exactly zero forward, so this holds by construction - but the registration that feeds it does
    not: the target-pin counterfactual carries a SECOND survival weight through the same loop, and
    a second accumulator that consumed one random number would move the reported price. The base
    valuation's one scenario reads `torch.rand`; under exposure the pin has rows to latch across
    and the counterfactual weight runs the whole block loop beside the reported one.

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
    """The acceptance gate for the knock-in, and the one whose oracle resolves cleanly.

    Measured on this fixture, identical CRN readings before and after: AAD -59,971.64 against an
    oracle of -83,856.45 at 0.90% flatness, i.e. 39.83% SHORT; corrected, -83,663.21, i.e. 0.23%.
    Registering only the fixings this row has yet to observe (`dt > 0`) left 5.50% behind: under
    CMC a deal's own dates are folded into the mtm grid, so EVERY fixing date is also a reporting
    row, and on that row the first fixing is a past reset whose knock-in was going unregistered.

    Killing mutation: the conditional-p splice contributing nothing, so the knock-in's flux is
    missing.
    """
    _, _, aad = _cmc(KNOCK_IN_CMC, gradient=True)
    r = ladder(price=lambda s: _cmc(KNOCK_IN_CMC, spot=s)[0], aad=aad, base=SPOT,
               rungs=(3e-4, 1e-3, 3e-3, 1e-2))
    assert r.agrees(tol=0.02), f'the knock-in flux is not reaching the tape\n{r}'


def test_the_knock_in_gradient_matches_bump_and_reprice_at_base_valuation():
    """Base valuation, where the pricer's inner MC is the ONLY simulation there is - one date, one
    scenario, 2x65536 inner paths - and the route that had no boundary term at all: a deal priced
    by Monte Carlo reported a gradient with the flux missing. Measured: AAD 2,421,013 against a CRN
    plateau of ~3.766e6, 35.7% short; corrected, 3,768,262, i.e. 0.07%. The same TARF with no
    `Barrier` agrees to 0.00% at 0.01% flatness either way, so the 35.7% is the barrier. Across a
    40x range of kernel bandwidths, 0.005 to 0.2, the corrected delta spreads 0.62%.

    Killing mutation: the conditional-p splice contributing nothing, so the knock-in's flux is
    missing.
    """
    _, aad = _baseval(KNOCK_IN, greeks=True)
    r = ladder(price=lambda s: _baseval(KNOCK_IN, spot=s)[0], aad=aad, base=SPOT,
               rungs=(3e-4, 1e-3, 3e-3, 1e-2))
    assert r.agrees(tol=0.02), f'the knock-in flux is not reaching the tape\n{r}'
