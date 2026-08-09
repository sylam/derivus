"""Closed-form gates for the `CreditNthToDefault` premium leg - `pv_credit_step_down_cashflows`,
which had no coverage at all.

WHAT THE PRICER COMPUTES, read off the code rather than the deal's documentation: per payment
period it integrates E[rate] dt by trapezoid over hazard samples spaced 30 days, and discounts
each period's expected accrual at the PAY date. `expected_rate_gaussian_copula` supplies E[rate],
and with `Max_Defaults` = n that is the STEP-DOWN expectation E[c(1 - k/n) 1_{k<n}] over the
default count k - NOT c P(k < n); the two coincide only at n = 1. Names default under a
one-factor Gaussian copula integrated by 11-node Gauss-Hermite over the common factor, on
per-name cumulative hazards scaled by the index proxy g.

CONVENTIONS THESE GATES REST ON, each verified against the code:

  * g == 1 EXACTLY at the base date - `surv` and `surv_base` gather the same curve at the same
    points, so the quotient is bitwise one whatever the CDS_Index curve says. Pinned below.
  * `add_maturity_accrual` is a NO-OP for this pricer: it edits CASHFLOW_INDEX_Year_Frac, and the
    pricer reads only CASHFLOW_INDEX_Pay_Day. dt comes from the DISCOUNT curve's day count, never
    from the deal's `Accrual_Day_Count`, so both are ACT_365 here to keep them consistent.
  * `cash_settle` on the first payment cannot move the t0 number: `shared.t_Cashflows` is None
    under base valuation and the call is a no-op.
  * `Principal` and `Buy_Sell` never reach the pricer - the schedule is built with nominal 1.0 and
    rate 1.0, and the value is the coupon integral alone. The fixture carries Principal 1e6 and
    Buy_Sell 'Buy' while the closed forms are per unit notional and unsigned, so the value gates
    are themselves the evidence that neither field is read.
  * the accrual window opens at t + 1 DAY, not at the period start. At t0 that drops one day of
    accrual (2.8e-3 of this leg); the closed forms integrate from day 1 to match. The strict xfail
    at the bottom states the rule the engine should follow and measures what it costs.

CLOSED FORMS, flat hazards S_j(t) = exp(-h_j t) and flat discounting DF(T) = exp(-r T):

    n = 1, any number of names:  E[rate] = c prod_j S_j  - one effective hazard sum_j h_j
    n = 2, two names:            E[rate] = c (P0 + P1/2) = c (S_1 + S_2)/2, the cross term
                                 cancelling exactly, so the answer is the survival AVERAGE and in
                                 exact arithmetic carries no correlation at all
    PV = c sum_i DF(T_i) sum_k w_k [exp(-h_k tau_{i-1}) - exp(-h_k tau_i)] / h_k,  tau_0 = 1/365

TOLERANCES are the trapezoid bias and nothing else: the engine reproduces a discretised replica of
its own trapezoid to 3.6e-16, and each measured error matches the h^2 dt^2 / 12 bound at
dt = 30/365 (dt^2/12 = 5.63e-4) - 2.23e-7 measured against 2.25e-7 predicted for h = 0.02 (pinned
3e-7), 2.73e-6 against 2.76e-6 for the h = 0.07 product (pinned 4e-6), 8.03e-7 against 8.17e-7 for
the (0.02, 0.05) average (pinned 1e-6).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
import torch

import derivus
from derivus import utils
from derivus.config import Config
from derivus.instruments import construct_instrument

BASE = pd.Timestamp('2024-06-28')
DTYPE = torch.float64
RATE = 0.03            # flat continuously compounded discount curve
COUPON = 0.01          # Pay_Rate, which the schema quotes in percent
INDEX_HAZARD = 0.03


def _curve(h, top=10.0):
    """A SurvivalProb curve is negative log survival, so a flat hazard is H(T) = h T. Tenors clip
    to the knot range, hence the first knot at T+1 rather than at 0."""
    return utils.Curve([], [[1.0 / 365.0, h / 365.0], [top, h * top]])


def _cfg(hazards, max_defaults, rho=0.0, index_hazard=INDEX_HAZARD, effective=0, maturity=12):
    names = ['NAME{}'.format(i + 1) for i in range(len(hazards))]
    c = Config()
    c.params['System Parameters']['Base_Currency'] = 'USD'
    c.params['System Parameters']['Base_Date'] = BASE
    c.params['Price Factors'] = {
        'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Priority': 1, 'Spot': 1.0},
        'InterestRate.USD': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                             'Curve': utils.Curve([], [[1.0 / 365.0, RATE], [10.0, RATE]])},
        'SurvivalProb.IDX': {'Recovery_Rate': 0.4, 'Curve': _curve(index_hazard)}}
    c.params['Price Factors'].update({
        'SurvivalProb.' + name: {'Recovery_Rate': 0.4, 'Curve': _curve(h)}
        for name, h in zip(names, hazards)})
    c.params['Price Models'] = {}
    c.params['Valuation Configuration'] = {}
    c.deals = {
        'Attributes': {'Reference': 'test', 'Tag_Titles': ''},
        'Deals': {'Children': [{'Instrument': construct_instrument({
            'Object': 'CreditNthToDefault', 'Reference': 'NTD1', 'Currency': 'USD',
            'Discount_Rate': 'USD', 'CDS_Index': 'IDX', 'Names': names,
            'Effective_Date': BASE + pd.DateOffset(months=effective),
            'Maturity_Date': BASE + pd.DateOffset(months=maturity),
            'Pay_Frequency': pd.DateOffset(months=3), 'Pay_Rate': 100.0 * COUPON,
            'Max_Defaults': max_defaults, 'Defaults_So_Far': 0, 'Correlation': rho,
            'Buy_Sell': 'Buy', 'Principal': 1e6, 'Accrual_Day_Count': 'ACT_365',
            'Amortisation': None, 'Calendars': None}, {})}]},
        'Calculation': {'Base_Date': BASE, 'Currency': 'USD'}}
    return c


def _price(**kwargs):
    """Base valuation: one deal-time row, so g collapses and the trapezoid plus the Gauss-Hermite
    quadrature are the only approximations left. Row 0 of the frame is the root aggregate, so the
    deal is read by Reference."""
    _, out = derivus.run_baseval(_cfg(**kwargs), prec=DTYPE,
                                 overrides={'MCMC_Simulations': 1, 'Random_Seed': 1})
    rows = out['Results']['mtm']
    return float(rows[rows['Reference'] == 'NTD1']['Value'].iloc[0])


def _pays(first=3, last=12):
    """Pay days as day offsets, derived from the schedule rule rather than from the deal."""
    return np.array([(BASE + pd.DateOffset(months=m) - BASE).days
                     for m in range(first, last + 1, 3)])


def _pv(terms, pays, start=1):
    """c sum_i DF(T_i) sum_k w_k int over period i of exp(-h_k t) dt, ACT_365 throughout, the
    first period opening `start` days out. `terms` is the (weight, hazard) decomposition of
    E[rate]/c."""
    tau = np.r_[start, pays] / 365.0
    return COUPON * sum(
        np.exp(-RATE * pays[i] / 365.0) * sum(
            w * (np.exp(-h * tau[i]) - np.exp(-h * tau[i + 1])) / h for w, h in terms)
        for i in range(pays.size))


def test_a_single_name_leg_matches_the_exact_survival_integral():
    """One name at Max_Defaults=1: k < 1 means survival, E[rate] = c S(t), and the period integral
    is elementary. Everything between the engine and this number is the 30-day trapezoid."""
    pays = _pays()
    priced = _price(hazards=[0.02], max_defaults=1)
    assert priced == pytest.approx(_pv([(1.0, 0.02)], pays), rel=3e-7), (
        f'priced {priced:.12g} against the exact survival integral {_pv([(1.0, 0.02)], pays):.12g}')

    # MUTATE: drop the last period. A gate that cannot tell four periods from three is a placebo
    assert priced != pytest.approx(_pv([(1.0, 0.02)], pays[:-1]), rel=3e-7), (
        'a three-period closed form matched - the gate is not reading the schedule')


def test_one_name_collapses_the_copula_whatever_the_correlation():
    """With a single name E_z[q(z)] = p identically, so the correlation cannot reach the price.
    What survives is 11-node Gauss-Hermite error, and it grows with rho: 3.9e-13 at 0.2, 4.7e-5 at
    0.8, 1.7e-3 at 0.95 - quadrature, not model."""
    flat = _price(hazards=[0.02], max_defaults=1, rho=0.0)
    assert _price(hazards=[0.02], max_defaults=1, rho=0.8) == pytest.approx(flat, rel=1e-4), (
        'one name must price the same at any correlation')

    # MUTATE: rho is inert above only because one name collapses it. Three names at Max_Defaults=2
    # is a basket where rho is live, and there it has to move the price
    basket = _price(hazards=[0.02, 0.05, 0.10], max_defaults=2, rho=0.0)
    assert _price(hazards=[0.02, 0.05, 0.10], max_defaults=2, rho=0.8) != pytest.approx(
        basket, rel=1e-3), 'Correlation reaches nothing - the collapse gate is vacuous'


def test_the_step_down_expectation_follows_max_defaults():
    """Two names at rho = 0, where the quadrature is exact in z and only the trapezoid is left.
    Max_Defaults=2 pays c(P0 + P1/2) = c(S1 + S2)/2, the survival AVERAGE; Max_Defaults=1 pays
    c S1 S2, the product at hazard h1 + h2. Each closed form is the other's mutation - they sit
    1.7e-2 apart, four orders of magnitude outside either tolerance."""
    pays, h1, h2 = _pays(), 0.02, 0.05
    average, product = [(0.5, h1), (0.5, h2)], [(1.0, h1 + h2)]

    step_down = _price(hazards=[h1, h2], max_defaults=2)
    assert step_down == pytest.approx(_pv(average, pays), rel=1e-6), (
        f'priced {step_down:.12g} against the survival average {_pv(average, pays):.12g}')
    assert step_down != pytest.approx(_pv(product, pays), rel=1e-6), (
        'Max_Defaults=2 priced as a first-to-default product')

    first = _price(hazards=[h1, h2], max_defaults=1)
    assert first == pytest.approx(_pv(product, pays), rel=4e-6), (
        f'priced {first:.12g} against the survival product {_pv(product, pays):.12g}')
    assert first != pytest.approx(_pv(average, pays), rel=4e-6), (
        'Max_Defaults=1 priced as a step-down average')


def test_the_index_proxy_is_the_identity_at_the_base_date():
    """g scales every name's hazard by H_index(samples) over H_index of the SAME samples at t0.
    One deal-time row makes those the same gather, so g is bitwise one and the CDS_Index curve
    cannot reach the price - exact equality, not a tolerance."""
    flat = _price(hazards=[0.02], max_defaults=1)
    assert _price(hazards=[0.02], max_defaults=1, index_hazard=0.5) == flat, (
        'the index curve moved a base valuation, so g is not the identity')

    # MUTATE: a hazard move of that size on the NAME has to be seen, or the equality above is
    # reporting a dead harness rather than a collapsed proxy
    assert _price(hazards=[0.5], max_defaults=1) != pytest.approx(flat, rel=1e-3), (
        'a 25x name hazard changed nothing - this harness cannot see a curve')


@pytest.mark.xfail(strict=True, reason='DEFECT: the accrual window opens at t+1 rather than at '
                                       'the period start. pv_credit_step_down_cashflows builds '
                                       'samples_points from time_block[0]+1 and never reads '
                                       'CASHFLOW_INDEX_Start_Day, so a forward-starting deal '
                                       'accrues from tomorrow instead of its effective date - '
                                       '51% over on this one')
def test_the_accrual_window_opens_at_the_effective_date():
    """Effective in 6m, quarterly to 18m: the first coupon covers 6m to 9m. The engine integrates
    from tomorrow to 9m instead. It is the same defect the gates above absorb as a one-day drop,
    at a size nothing can absorb."""
    pays = _pays(9, 18)
    effective = (BASE + pd.DateOffset(months=6) - BASE).days
    priced = _price(hazards=[0.02], max_defaults=1, effective=6, maturity=18)
    assert priced == pytest.approx(_pv([(1.0, 0.02)], pays, start=effective), rel=3e-7), (
        f'priced {priced:.12g} against an effective-date accrual '
        f'{_pv([(1.0, 0.02)], pays, start=effective):.12g}')
