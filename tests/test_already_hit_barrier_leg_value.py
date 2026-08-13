"""THE VALUE GATE for the discrete barrier's already-hit knock-in leg.

This leg carried a 106.7% forward-convention error and a 15.8% model mix into production with a
green suite behind it. No AAD-vs-CRN gate could ever have caught either: a bump ladder
differentiates whatever the leg computes, so a wrong value with a consistent derivative reads as a
pass. The only thing that catches it is the VALUE, against a reference recomputed from the deal's
own dates, on a fixture where this leg IS the entire answer.

WHY THE SUITE WAS BLIND - three independent degeneracies, any one of which being absent would have
exposed the defect, so this fixture removes all three and the test asserts it has:

  * BASE VALUATION. One deal-time row, so the hit mask is all-False at row 0 and the leg is never
    evaluated. Here: an exposure grid whose report rows sit AFTER the first barrier observation.
  * ONE REMAINING OBSERVATION. With a single interval left, the per-interval daily sub-step count
    the HN leg sums coincides with a single rounding of the whole tenor, and the two candidate
    discretisations are indistinguishable. Here: 26 observations, and `_daily_steps` differs from
    `round(tau * 252)` by 8 steps at the first asserted row - checked, not assumed.
  * r = q = 0. Every other HN fixture in this repo sets both to zero, which kills the carry
    channel outright: the forward is the spot, the HN leg's `exp(r_step * n_total)` growth factor
    is 1, and the missing `dt` in `sum(drifts)` has nothing to multiply. Here r = 5%, q = 1%, both
    non-zero and different - r != q keeps the forward alive, r != 0 keeps the discounting alive.

ANTI-PLACEBO, MEASURED. The r = q = 0 fixture is not merely weaker, it is GREEN-BUT-BLIND to a
third of the matrix. Re-run with r = q = 0 and nothing else changed, the same six mutants read:

    mutation                                        r=5% q=1%     r = q = 0
    (a) forward reverted to (drifts + 0.5*var).sum()  KILLED       KILLED (+1.75e-01, the
                                                                   0.5*sigma^2*tau half survives)
    (b) dt kept, spurious +0.5*var restored           KILLED       KILLED (+1.75e-01, identical
                                                                   to (a) - the halves collapse)
    (c) HN branch reverted to Black                   KILLED       KILLED
    (d) n_total <- round(rem_exp * 252)               KILLED       KILLED
    (e) HN growth factor exp(r_step*n_total) dropped  KILLED       SURVIVED (-1.5e-11)  <-- blind
    (f) terminal discount factor dropped              KILLED       SURVIVED (+1.0e-14)  <-- blind

(e) and (f) are the whole reason the fixture is not "simplified": with a zero carry `r_step` is
zero so the growth factor is identically 1, and with a zero rate the discount factor is identically
1, so two of the six mutants become no-ops and the gate passes while measuring nothing about
either. (a) and (b) stop being distinguishable at the same time. Do not set r = q, and do not set
either to zero.

THE REFERENCE is rebuilt here from the deal's own date list - never by importing the pricer's
expression. The GBM arm is `D(t,T) * Black(S*exp((r-q)*tau), K, sigma*sqrt(tau))` with the normal
CDF taken straight off `torch.erfc`, so `utils.black_european_option` is not in the loop. The HN
arm goes through `tests/hn_reference.py`'s parameter plumbing into `utils.hn_call`, the model's
closed form - the same reference the parity gate at `tests/test_hn_oss_pricers.py` uses; what this
test recomputes independently is the ASSEMBLY around it, which is where every defect lived: the
per-row step count, the per-step carry, the forward-growth factor, the discount, and the units.

TOLERANCE 1e-10 relative. Closed form against closed form on an `all_hit` block: the OSS is
skipped entirely, the scenario spot is a zero-vol zero-drift GBM so every path carries exactly the
initial spot, and there is no Monte Carlo error anywhere to hide behind. Measured worst-row
agreement is 7.8e-15 on the GBM arm (both sides evaluate the same algebra) and 1.4e-11 on the HN
arm, where the Fourier inversion amplifies the last-bit difference between the pricer's
`sum(drifts*dt)/n` and this file's `(r-q)*tau/n`. That is 7x of headroom against a matrix whose
smallest kill is 1.3e-2, i.e. nine orders clear. The parity gate next door runs 1e-6 because it
compares against a sampled recursion; this one has no such excuse.

REPRODUCING THE MATRIX. Every mutant below was applied by rebinding `pricing.total_log_forward`,
`utils.hn_call` or `utils.calc_discount_rate` from a scratch runner - no edit to the tree - which
is also why (c) and (d) can be stated exactly: a fake `hn_call` can look the row's tenor up by its
unique step count and price Black at the implied surface, or re-enter the real one at the other
discretisation.
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
import derivus.pricing as pricing
from derivus import utils
from derivus.config import Config
from derivus.instruments import construct_instrument
import hn_reference as hnref

BASE = pd.Timestamp('2024-06-28')
DTYPE = torch.float64
SPOT, STRIKE, UNITS = 100.0, 100.0, 100.0
R, Q, SIG = 0.05, 0.01, 0.25            # both non-zero and different - see the module docstring
SPY = 252.0                             # HN_Steps_Per_Year, the pricer's declared default
EXPIRY_DAY = 365
# fortnightly observations: 25 barrier dates, and instruments.py unions Expiry_Date in, so the
# remaining strip at every asserted row is many intervals long and each one rounds on its own
BARRIER_DAYS = [14 * k for k in range(1, 26)]
OBS_DAYS = np.array(BARRIER_DAYS + [EXPIRY_DAY], dtype=float)

_SP = hnref.hn_params_from_targets(
    ann_vol=0.30, persistence=0.94, gamma=350.0, leverage_share=0.7, steps_per_year=SPY)
H0 = 1.6 * float(utils.hn_stationary_var(
    _SP['omega'], _SP['alpha'], _SP['beta'], _SP['gamma_star']))
HN = {'Omega': float(_SP['omega']), 'Alpha': float(_SP['alpha']), 'Beta': float(_SP['beta']),
      'Gamma_Star': float(_SP['gamma_star']), 'H0': H0}


def _cfg(hn, barrier):
    """An Up_And_In call on a DETERMINISTIC spot: zero-vol zero-drift GBM holds every path at
    exactly `SPOT`, which is what turns the expectation into a number the test can assert to the
    last bit. The model under test is the law of the REMAINING horizon, not the scenario diffusion.

    `barrier` selects the two fixtures. Below spot (50) every scenario knocks in at the day-14
    observation, so every later row is `all_hit`, the OSS is skipped and the block's mark IS this
    leg. Above spot (200) nothing ever crosses, which is the boundary fixture: the leg is built
    anyway (`if some_hit or boundary_aad`) and shows up only as the LatchedBoundarySet's triggered
    branch, where a price-only assertion cannot see it.
    """
    field = {
        'Object': 'EquityBarrierOption', 'Reference': 'BARR1', 'Currency': 'USD',
        'Payoff_Currency': 'USD', 'Equity': 'EQ', 'Dividends': 'EQ', 'Discount_Rate': 'USD',
        'Equity_Volatility': 'EQ', 'Buy_Sell': 'Buy', 'Option_Type': 'Call',
        'Strike_Price': STRIKE, 'Expiry_Date': BASE + pd.Timedelta(days=EXPIRY_DAY),
        'Units': UNITS, 'Barrier_Type': 'Up_And_In', 'Barrier_Price': barrier, 'Cash_Rebate': 0.0,
        'Barrier_Dates': [BASE + pd.Timedelta(days=d) for d in BARRIER_DAYS],
        'Barrier_Monitoring_Frequency': pd.DateOffset(days=1),
    }
    val = {'EquityBarrierOption': {'SpotModel': 'HestonNandi'}} if hn else {}
    c = Config()
    c.params['System Parameters']['Base_Currency'] = 'USD'
    c.params['System Parameters']['Base_Date'] = BASE
    c.params['Price Factors'] = {
        'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Priority': 1, 'Spot': 1.0},
        'InterestRate.USD': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                             'Curve': utils.Curve([], [[0.0, R], [5.0, R]])},
        'EquityPrice.EQ': {'Spot': SPOT, 'Currency': 'USD', 'Interest_Rate': 'USD',
                           'Issuer': '', 'Respect_Default': 'No', 'Jump_Level': 0.0},
        'DividendRate.EQ': {'Currency': 'USD', 'Floor': None,
                            'Curve': utils.Curve([], [[0.01, Q], [5.0, Q]])},
        'VolatilityGrid.EQ': {'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness',
                              'Surface': utils.Curve([], [[m, t, SIG] for m in (0.8, 1.0, 1.2)
                                                          for t in (0.02, 2.0)])},
        'HestonNandiModelParameters.EQ': dict(HN, Property_Aliases=None),
    }
    c.params['Price Models'] = {'GBMAssetPriceModel.EQ': {'Vol': 0.0, 'Drift': 0.0}}
    c.params['Model Configuration'].append('EquityPrice', (), 'GBMAssetPriceModel')
    c.params['Valuation Configuration'] = val
    c.deals = {'Attributes': {'Reference': 'test', 'Tag_Titles': ''},
               'Deals': {'Children': [{'Instrument': construct_instrument(field, val)}]},
               'Calculation': {'Base_Date': BASE, 'Currency': 'USD'}}
    return c


# ------------------------------------------------------------------- the independent reference

def _daily_steps(day):
    """The remaining strip's daily sub-step count, summed PER INTERVAL as the parity leg does.

    Rebuilt from the deal's date list: the observations at or after `day` (searchsorted 'left' is
    what `TensorResets.get_start_index` does), the year fractions between them, and each one's own
    `max(round(dt*252), 1)`. It is NOT `round(tau*252)` - the roundings do not commute with the
    sum, and on this fixture they differ by 8 steps at day 28, which is what makes the two
    candidate discretisations distinguishable at all.
    """
    fixings = OBS_DAYS[int(np.searchsorted(OBS_DAYS, day)):] / 365.0
    dts = np.hstack([fixings[0] - day / 365.0, np.diff(fixings)])
    return int(sum(max(int(round(dt * SPY)), 1) for dt in dts))


def _ref_gbm(day):
    """D(t,T) * Black(S*exp((r-q)*tau), K, sigma*sqrt(tau)) * units, normal CDF off erfc."""
    tau = (EXPIRY_DAY - day) / 365.0
    fwd, vol = SPOT * np.exp((R - Q) * tau), SIG * np.sqrt(tau)
    d1 = torch.tensor((np.log(fwd / STRIKE) + 0.5 * vol * vol) / vol, dtype=DTYPE)
    n1, n2 = (0.5 * torch.erfc(-d / np.sqrt(2.0)) for d in (d1, d1 - vol))
    return UNITS * (fwd * float(n1) - STRIKE * float(n2)) * np.exp(-R * tau)


def _ref_hn(day):
    """D(t,T) * exp(b*tau) * hn_call(S, K, n, H0, ..., r_step) * units.

    `n` is `_daily_steps`, `r_step` the per-step carry b*tau/n, and the growth factor puts the
    forward-measure value on the forward - the same separation `black_european_option` makes. The
    variance recursion is seeded at the H0 FACTOR, not at the stationary variance.
    """
    tau = (EXPIRY_DAY - day) / 365.0
    n = _daily_steps(day)
    r_step = (R - Q) * tau / n
    p = hnref.as_tensors(dict(omega=_SP['omega'], alpha=_SP['alpha'], beta=_SP['beta'],
                              gamma_star=_SP['gamma_star'], r=r_step))
    call = float(utils.hn_call(SPOT, STRIKE, n, H0, **p))
    return UNITS * call * np.exp(r_step * n) * np.exp(-R * tau)


REF = {False: _ref_gbm, True: _ref_hn}
OVERRIDES = {'Run_Date': BASE.strftime('%Y-%m-%d'), 'Time_grid': '0d 12m(3m)', 'Batch_Size': 8,
             'Simulation_Batches': 1, 'Random_Seed': 1, 'Currency': 'USD',
             'MCMC_Simulations': 64, 'Tenor_Offset': 0.0, 'Deflation_Interest_Rate': 'USD'}


def _assert_fixture_is_not_degenerate(days):
    """Each of these is a degeneracy the suite already had; assert them so a later simplification
    fails here rather than quietly turning the gate into a placebo."""
    assert R != Q and R != 0.0 and Q != 0.0, 'zero or equal carry legs kill the forward channel'
    assert len(days) > 1, 'one MTM row is base valuation, where this leg is never evaluated'
    first = days[0]
    assert len(OBS_DAYS) - np.searchsorted(OBS_DAYS, first) > 1, 'one remaining observation date'
    assert _daily_steps(first) != round((EXPIRY_DAY - first) / 365.0 * SPY), (
        'per-interval and whole-tenor step counts coincide - the discretisation is untested')
    assert abs(_ref_gbm(first) / _ref_hn(first) - 1.0) > 0.1, (
        'the two models agree on this fixture - it cannot tell which one priced the leg')


def _hit_rows(index):
    """Report rows strictly after the first observation (so `all_hit`) and strictly before expiry.

    Expiry is excluded on purpose: `rem_exp` is zero there, both arms floor - the HN leg at one
    daily step, the Black leg at `rem_exp.clamp(min=1e-4)` - and the row is an artifact of the
    floor rather than a statement about the leg."""
    days = [(pd.Timestamp(t) - BASE).days for t in index]
    return [d for d in days if BARRIER_DAYS[0] < d < EXPIRY_DAY]


@pytest.mark.parametrize('hn', [False, True], ids=['gbm', 'heston_nandi'])
def test_the_already_hit_leg_is_the_closed_form_it_claims_to_be(hn):
    """Every all_hit row of a knocked-in barrier, against the reference, at 1e-10.

    `all_hit` is what makes this an identity rather than an estimate: the OSS is skipped entirely
    (`theo_cashflow = hit_value`), so the reported mark IS this leg with no Monte Carlo between
    them. The flat-row assertion is the proof that the scenario spot is deterministic - a
    dispersed row would mean the reference is an average and 1e-10 would be pinning noise.

    MUTATION KILL MATRIX. `reported` and `rel` are the day-28 row; `worst` is the largest
    deviation over the 24 asserted rows, which is what the gate actually reads. See the module
    docstring for the r = q = 0 rerun, where (e) and (f) go green.

        mutation                                  arm   reported day28    rel      worst   verdict
        (correct)                                 gbm      1120.6186   +7.8e-15  +7.8e-15  pass
        (correct)                                  hn      1332.3194   +3.9e-12  -1.4e-11  pass
        (a) forward <- (drifts + 0.5*var).sum()   gbm     17167.4465   +1.43e+01 +1.43e+01 KILLED
        (b) dt kept, spurious +0.5*var restored   gbm      1303.5780   +1.63e-01 +1.63e-01 KILLED
        (c) HN branch reverted to Black            hn      1120.6186   -1.59e-01 -3.13e-01 KILLED
        (d) n_total <- round(rem_exp * 252)        hn      1314.3623   -1.35e-02 -4.03e-02 KILLED
        (e) growth factor exp(r_step*n) dropped    hn      1284.0123   -3.63e-02 -3.63e-02 KILLED
        (f) terminal discount factor dropped      gbm      1173.5640   +4.72e-02 +4.72e-02 KILLED

    (a) is not a hypothetical: it is what the tree shipped. The leg reported 17167 against a true
    1121 - the option was worth 15x its value on every all_hit row - and the suite was green.
    """
    _, out = derivus.run_cmc(_cfg(hn, 0.5 * SPOT), prec=DTYPE, overrides=OVERRIDES)
    mtm = out['Results']['mtm']
    days = _hit_rows(mtm.index)
    _assert_fixture_is_not_degenerate(days)

    for day in days:
        row = mtm.loc[BASE + pd.Timedelta(days=day)].values
        assert row.std() == 0.0, 'day %d: deterministic spot must give a flat row' % day
        ref = REF[hn](day)
        assert row[0] == pytest.approx(ref, rel=1e-10), (
            'day %d: already-hit leg %.10f vs the closed form it declares %.10f (rel %.3e)'
            % (day, row[0], ref, row[0] / ref - 1.0))


def _registered_branch(hn):
    """Run with sensitivities on and spy the barrier's registration.

    `pricing.interpolate` is the last thing `Deal.calculate` does, so the spy sees `shared` with
    every set already appended. Wanting greeks IS the `boundary_aad` switch (calculation.py), and
    a CVA gradient is how a CMC run asks for them."""
    c = _cfg(hn, 2.0 * SPOT)
    c.params['Price Factors']['SurvivalProb.CPTY'] = {
        'Recovery_Rate': 0.4, 'Curve': utils.Curve([], [[0.0, 0.0], [10.0, 0.4]])}
    seen, original = {}, pricing.interpolate

    def spy(mtm, shared, time_grid, deal_data, interpolate_grid=True):
        result = original(mtm, shared, time_grid, deal_data, interpolate_grid)
        seen['sets'] = [x for x in shared.boundary_sets if isinstance(x, utils.BoundarySet)]
        seen['days'] = time_grid.time_grid[
            deal_data.Time_dep.deal_time_grid][:, utils.TIME_GRID_MTM]
        return result

    pricing.interpolate = spy
    try:
        derivus.run_cmc(c, prec=DTYPE, overrides=dict(
            OVERRIDES, MCMC_Simulations=32, Gradient_Variables='Factors',
            Credit_Valuation_Adjustment={
                'Calculate': 'Yes', 'Counterparty': 'CPTY', 'Deflate_Stochastically': 'No',
                'Stochastic_Hazard_Rates': 'No', 'Gradient': 'Yes'}))
    finally:
        pricing.interpolate = original
    return seen


@pytest.mark.parametrize('hn', [False, True], ids=['gbm', 'heston_nandi'])
def test_the_boundary_counterfactual_is_the_same_closed_form(hn):
    """With NOTHING hit and sensitivities on, the triggered branch must be that same reference.

    The leg has a second consumer a price assertion cannot reach. `if some_hit or boundary_aad`
    builds it on every block whether or not a scenario has crossed, and under greeks it becomes
    the LatchedBoundarySet's `triggered` half - the value the correction says the deal WOULD have
    had. It is worth exactly zero in the forward pass, so a wrong leg there moves no reported
    number and shows up only in a derivative, which is precisely the shape of defect this repo
    keeps shipping. Here the barrier sits at twice a deterministic spot, so `fired` is empty on
    every observation and the branch is pure counterfactual.

    Same reference function, same 1e-10, and the whole matrix turns this red as well - measured
    worst deviation over the 26 non-terminal deal rows: (a) +1.49e+01, (b) +1.71e-01, (c)
    -3.13e-01, (d) -4.03e-02, (e) -3.92e-02, (f) +5.13e-02, against +7.8e-15 / -1.4e-11 correct.
    Its rows 0 and 14 are the ones the price gate cannot reach at all: they are priced by the OSS,
    so only the counterfactual exposes the leg there, and (a) reads +1492% on row 14.
    """
    seen = _registered_branch(hn)
    bset, = seen['sets']
    assert not any(bool(f.any()) for f in bset.fired), (
        'a scenario crossed a barrier at twice the spot - the branch is no longer counterfactual')
    assert bset.triggered.shape[0] == len(seen['days'])

    for i, day in enumerate(seen['days']):
        if day >= EXPIRY_DAY:
            continue
        ref, got = REF[hn](int(day)), float(bset.triggered[i][0])
        assert got == pytest.approx(ref, rel=1e-10), (
            'day %d: triggered branch %.10f vs the closed form %.10f (rel %.3e)'
            % (day, got, ref, got / ref - 1.0))
