"""Closed-form gates for the `CreditNthToDefault` premium leg - `pv_credit_step_down_cashflows`,
which had no coverage at all and, when it got some, five defects.

WHAT THE PRICER COMPUTES, read off the code rather than the deal's documentation: per payment
period it integrates E[rate] dt by trapezoid over hazard samples spaced 30 days, weights that by
the period's NOMINAL and discounts it at the PAY date. `expected_rate_gaussian_copula` supplies
E[rate], and with `Max_Defaults` = n and `Defaults_So_Far` = k0 that is the STEP-DOWN expectation
E[c(1 - (k0+k)/n) 1_{k0+k<n}] over the default count k - NOT c P(k < n); the two coincide only at
n = 1, k0 = 0. Names default under a one-factor Gaussian copula integrated by Gauss-Hermite over
the common factor on per-name cumulative hazards scaled by the index proxy g.

CONVENTIONS THESE GATES REST ON, each verified against the code:

  * g == 1 EXACTLY at the base date - `surv` and `surv_base` gather the same curve at the same
    points, so the quotient is bitwise one whatever the CDS_Index curve says. Pinned below, at a
    level AND at a shape, because a flat index curve cancels its own level in any ratio. The one
    thing that quotient cannot survive is a CDS_Index curve anchored at exactly T = 0: the window
    now opens on the valuation date, both sides evaluate H_index(0) = 0 there and the price is
    NaN. See `_curve` - every survival curve here is anchored a hair above zero, as a bootstrapped
    one is.
  * the accrual window opens at the first unpaid period's own `Start_Day`. `get_cashflows` emits
    (start_i, pay_i) = (reset_i, reset_{i+1}), so periods are contiguous and that one start plus
    the pay days are all the boundaries - pinned by the contiguity gate, which is the assumption
    the pricer's sample grid rests on.
  * two day counts doing two jobs: dt is measured in the deal's `Accrual_Day_Count`, discounting
    and the hazard lookups in each curve's own (the survival factor hardcodes ACT_365). An ACT_360
    deal on an ACT_365 curve is therefore 365/360 of the same deal written ACT_365, and that ratio
    is a gate rather than a coincidence.
  * `add_maturity_accrual` is still a NO-OP here: it edits CASHFLOW_INDEX_Year_Frac and the pricer
    reads Start_Day, Pay_Day and Nominal. The coupon c likewise comes from `factor_dep['Coupon']`
    and not from the schedule's rate column, which the sibling convention fills anyway.
  * `cash_settle` on the first payment cannot move the t0 number: `shared.t_Cashflows` is None
    under base valuation and the call is a no-op.
  * `Principal`, `Buy_Sell` and `Amortisation` all reach the price through the ONE nominal column,
    the way `pv_credit_cashflows` reads it and the way `DealDefaultSwap` signs it (a Sell builds
    the schedule at negative principal). So the price is exactly linear in Principal and a Sell is
    the exact negation - equalities, not tolerances.

CLOSED FORMS, flat hazards S_j(t) = exp(-h_j t) and flat discounting DF(T) = exp(-r T):

    n = 1, k0 = 0, any number of names:  E[rate] = c prod_j S_j - one effective hazard sum_j h_j
    n = 2, k0 = 0, two names:            E[rate] = c (P0 + P1/2) = c (S_1 + S_2)/2, the cross term
                                         cancelling exactly, so the answer is the survival AVERAGE
                                         and in exact arithmetic carries no correlation at all
    n = 2, k0 = 1, two names:            E[rate] = c P0/2 = c S_1 S_2 / 2 - one step left
    PV = c sum_i N_i DF(T_i) sum_k w_k [exp(-h_k tau_{i-1}) - exp(-h_k tau_i)] / h_k, tau_0 = the
    first unpaid period's start day and N_i that period's nominal

TOLERANCES are the trapezoid bias and nothing else: each measured error matches the h^2 dt^2 / 12
bound at dt = 30/365 (dt^2/12 = 5.63e-4) - 2.2e-7 for h = 0.02 (pinned 3e-7), 2.7e-6 for the
h = 0.07 product (pinned 4e-6), 8.0e-7 for the (0.02, 0.05) average (pinned 1e-6).

QUADRATURE. A single name makes the exact price independent of rho, so any dependence left is
Gauss-Hermite error on a factor integral whose integrand approaches a step as rho -> 1. Measured
worst case over hazards 0.005 to 0.5 at rho = 0.95: 5.3e-3 at 11 nodes, 1.2e-3 at 21, 3.7e-4 at
31, 3.1e-5 at 51, 2.3e-6 at 71, 7.5e-7 at 81, 5.3e-8 at 101. `Quadrature_Points` defaults to the
smallest of those holding below 1e-6, and the gate below is that criterion.
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
PRINCIPAL = 1e6


def _curve(h, top=10.0):
    """A SurvivalProb curve is negative log survival, so a flat hazard is H(T) = h T, and two
    knots on that line reproduce it exactly between them.

    The first knot sits a HAIR above zero rather than at zero, and the two curves care for two
    different reasons now that the accrual window opens on the valuation date itself. On a NAME
    curve the cost is clipping: tenors clip to the knot range, so a first knot at 1/365 returns
    H = h/365 at T = 0 and costs 2.1e-6 of the price - an order of magnitude over the trapezoid
    bias these gates measure. On the INDEX curve the cost is a 0/0: the proxy is
    H_index(s)/H_index(s at t0), NaN when the first sample sits on a 0.0 knot - which is the
    no-knot-at-tenor-zero curve contract asserting itself. 1e-6 years leaves H(0) = 2e-8 h,
    8e-10 of the leg."""
    return utils.Curve([], [[1e-6, h * 1e-6], [top, h * top]])


def _cfg(hazards, max_defaults, rho=0.0, index_hazard=INDEX_HAZARD, index_curve=None, effective=0,
         maturity=12, **over):
    """`over` overwrites deal fields by JSON name, which is how every gate below varies one."""
    names = ['NAME{}'.format(i + 1) for i in range(len(hazards))]
    deal = {
        'Object': 'CreditNthToDefault', 'Reference': 'NTD1', 'Currency': 'USD',
        'Discount_Rate': 'USD', 'CDS_Index': 'IDX', 'Names': names,
        'Effective_Date': BASE + pd.DateOffset(months=effective),
        'Maturity_Date': BASE + pd.DateOffset(months=maturity),
        'Pay_Frequency': pd.DateOffset(months=3), 'Pay_Rate': 100.0 * COUPON,
        'Max_Defaults': max_defaults, 'Defaults_So_Far': 0, 'Correlation': rho,
        'Buy_Sell': 'Buy', 'Principal': PRINCIPAL, 'Accrual_Day_Count': 'ACT_365',
        'Amortisation': None, 'Calendars': None}
    deal.update(over)

    c = Config()
    c.params['System Parameters']['Base_Currency'] = 'USD'
    c.params['System Parameters']['Base_Date'] = BASE
    c.params['Price Factors'] = {
        'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Priority': 1, 'Spot': 1.0},
        'InterestRate.USD': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                             'Curve': utils.Curve([], [[1.0 / 365.0, RATE], [10.0, RATE]])},
        'SurvivalProb.IDX': {'Recovery_Rate': 0.4, 'Curve': index_curve or _curve(index_hazard)}}
    c.params['Price Factors'].update({
        'SurvivalProb.' + name: {'Recovery_Rate': 0.4, 'Curve': _curve(h)}
        for name, h in zip(names, hazards)})
    c.params['Price Models'] = {}
    c.params['Valuation Configuration'] = {}
    c.deals = {
        'Attributes': {'Reference': 'test', 'Tag_Titles': ''},
        'Deals': {'Children': [{'Instrument': construct_instrument(deal, {})}]},
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


def _pv(terms, pays, start=0, nominal=PRINCIPAL, basis=365.0):
    """-c sum_i N_i DF(T_i) int over period i of E[rate]/c dt, the first period opening on day
    `start`. `terms` is the (weight, hazard) decomposition of E[rate]/c and `nominal` the
    per-period notional, one entry per payment or a scalar for a flat profile. NEGATIVE for a
    Buy: the buyer PAYS the coupon, pv_credit_cashflows' premium convention.

    Hazards and discounting are ACT_365 - the survival factor hardcodes it and the fixture's
    interest curve declares it - while the ACCRUAL measure is the deal's Accrual_Day_Count, which
    on these conventions only rescales dt by 365/`basis`."""
    tau = np.r_[start, pays] / 365.0
    nom = np.broadcast_to(nominal, pays.shape)
    return -COUPON * (365.0 / basis) * sum(
        nom[i] * np.exp(-RATE * pays[i] / 365.0) * sum(
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


def test_the_periods_the_sample_grid_assumes_are_the_periods_the_schedule_builds():
    """The pricer's accrual window is the first unpaid period's Start_Day and then the PAY days,
    which covers every period only if start_i == pay_{i-1}. `get_cashflows` builds it that way -
    (start, end, pay) = (reset_i, reset_{i+1}, reset_{i+1}) - and this pins that, because a gap or
    an overlap there would silently drop or double-count accrual with no value gate to catch it."""
    deal = _cfg(hazards=[0.02], max_defaults=1).deals['Deals']['Children'][0]['Instrument']
    deal.reset({})
    schedule = utils.generate_fixed_cashflows(
        BASE, deal.resetdates, PRINCIPAL, None, utils.DAYCOUNT_ACT365, COUPON).schedule

    starts = schedule[:, utils.CASHFLOW_INDEX_Start_Day]
    pays = schedule[:, utils.CASHFLOW_INDEX_Pay_Day]
    assert np.array_equal(starts[1:], pays[:-1]), (
        f'periods are not contiguous: starts {starts} against pays {pays}')
    assert starts[0] == 0 and np.array_equal(pays, _pays()), (
        f'the schedule is not the one the closed forms assume: {starts[0]}, {pays}')

    # MUTATE: the same test against End_Day + 1, the off-by-one an inclusive-end convention would
    # produce, has to fail - otherwise the equality above is comparing nothing
    assert not np.array_equal(starts[1:], schedule[:-1, utils.CASHFLOW_INDEX_End_Day] + 1), (
        'an inclusive-end schedule would also pass - this gate cannot see a one-day gap')


def test_one_name_collapses_the_copula_whatever_the_correlation():
    """With a single name E_z[q(z)] = p identically, so the correlation cannot reach the price and
    what is left is pure Gauss-Hermite error. It grows with rho, and at the declared
    `Quadrature_Points` it stays under 1e-6 out to rho = 0.95 - the criterion that picked 81."""
    flat = _price(hazards=[0.02], max_defaults=1, rho=0.0)
    for rho in (0.5, 0.8, 0.95):
        assert _price(hazards=[0.02], max_defaults=1, rho=rho) == pytest.approx(flat, rel=1e-6), (
            f'one name must price the same at any correlation - rho {rho} moved it')

    # MUTATE: rho is inert above only because one name collapses it. Three names at Max_Defaults=2
    # is a basket where rho is live, and there it has to move the price
    basket = _price(hazards=[0.02, 0.05, 0.10], max_defaults=2, rho=0.0)
    assert _price(hazards=[0.02, 0.05, 0.10], max_defaults=2, rho=0.8) != pytest.approx(
        basket, rel=1e-3), 'Correlation reaches nothing - the collapse gate is vacuous'


def test_the_quadrature_node_count_is_read_from_the_deal():
    """`Quadrature_Points` is a declared field, so a deliberately tiny count has to degrade the
    invariance above by orders of magnitude - and at rho = 0 the integrand is constant in z, so
    the SAME tiny count cannot move the price at all. The pair is the read: one leg alone would
    also pass if the field were ignored and the price merely wrong."""
    exact = _price(hazards=[0.02], max_defaults=1, rho=0.0)
    assert _price(hazards=[0.02], max_defaults=1, rho=0.0, Quadrature_Points=3) == pytest.approx(
        exact, rel=1e-12), 'at rho = 0 the node count cannot matter, and it did'

    fine = _price(hazards=[0.02], max_defaults=1, rho=0.95)
    coarse = _price(hazards=[0.02], max_defaults=1, rho=0.95, Quadrature_Points=3)
    assert abs(coarse / fine - 1.0) > 1e-4, (
        f'3 nodes priced within {abs(coarse / fine - 1.0):.3g} of the declared node count - the '
        f'field is not read')


def test_the_step_down_expectation_follows_max_defaults():
    """Two names at rho = 0, where the quadrature is exact in z and only the trapezoid is left.
    Max_Defaults=2 pays c(P0 + P1/2) = c(S1 + S2)/2, the survival AVERAGE; Max_Defaults=1 pays
    c S1 S2, the product at hazard h1 + h2. Each closed form is the other's mutation - they sit
    1.7e-2 apart in relative terms, four orders of magnitude outside either tolerance."""
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


def test_defaults_so_far_is_the_initial_step():
    """k0 = 1 of n = 2 leaves one step: only the no-further-default state pays, and it pays half
    the coupon, so E[rate] = c P(k=0)/2 = c S1 S2 / 2 at rho = 0. k0 = n pays nothing at all, which
    is the indicator 1_{k0+k<n} rather than a rate that merely went small."""
    pays, h1, h2 = _pays(), 0.02, 0.05
    stepped = _price(hazards=[h1, h2], max_defaults=2, Defaults_So_Far=1)
    assert stepped == pytest.approx(_pv([(0.5, h1 + h2)], pays), rel=4e-6), (
        f'priced {stepped:.12g} against half the survival product '
        f'{_pv([(0.5, h1 + h2)], pays):.12g}')

    # MUTATE: the k0 = 0 closed form is the survival AVERAGE, and it must not match
    assert stepped != pytest.approx(_pv([(0.5, h1), (0.5, h2)], pays), rel=4e-6), (
        'Defaults_So_Far was ignored - this is the k0 = 0 answer')
    assert _price(hazards=[h1, h2], max_defaults=2, Defaults_So_Far=2) == 0.0, (
        'a basket already through its last step still paid a coupon')


def test_principal_scales_the_price_and_buy_sell_signs_it():
    """The nominal column carries both, the way `DealDefaultSwap` builds it - signed principal
    into `generate_fixed_cashflows` - so doubling Principal doubles every period's contribution
    exactly and a Sell negates it exactly. Equalities, because IEEE scaling by 2 and by -1 is."""
    buy = _price(hazards=[0.02], max_defaults=1)
    assert _price(hazards=[0.02], max_defaults=1, Principal=2 * PRINCIPAL) == 2 * buy, (
        'the price is not linear in Principal')
    assert _price(hazards=[0.02], max_defaults=1, Buy_Sell='Sell') == -buy, (
        'a Sell did not negate the premium leg')

    # MUTATE: a gate that would pass on a price of zero is a placebo, and zero is exactly what an
    # unread Principal used to leave here after the coupon integral
    assert buy == pytest.approx(_pv([(1.0, 0.02)], _pays()), rel=3e-7), (
        f'priced {buy:.12g} against the unit closed form at Principal {PRINCIPAL:g}')


def test_amortisation_steps_the_nominal_down():
    """Two quarterly periods with half the principal repaid on the first pay date. The engine
    weights each period by its OWN nominal, so the closed form is the same integral with
    (P, P/2) - a single average nominal, or the opening one, gives a different number."""
    pays, amort_date = _pays(3, 6), BASE + pd.DateOffset(months=3)
    nominal = np.array([PRINCIPAL, 0.5 * PRINCIPAL])
    priced = _price(hazards=[0.02], max_defaults=1, maturity=6,
                    Amortisation=utils.DateList({amort_date: 0.5 * PRINCIPAL}))
    assert priced == pytest.approx(_pv([(1.0, 0.02)], pays, nominal=nominal), rel=3e-7), (
        f'priced {priced:.12g} against the amortising closed form '
        f'{_pv([(1.0, 0.02)], pays, nominal=nominal):.12g}')

    # MUTATE: a flat profile at the opening nominal is what an unread amortisation gives, and the
    # mean nominal is what a single average would give. Neither may match
    assert priced != pytest.approx(_pv([(1.0, 0.02)], pays), rel=1e-4), (
        'the amortisation never reached the price')
    assert priced != pytest.approx(
        _pv([(1.0, 0.02)], pays, nominal=nominal.mean()), rel=1e-4), (
        'the periods share one averaged nominal')


def test_the_accrual_measure_is_the_deals_own_day_count():
    """dt is measured in `Accrual_Day_Count` while the hazard and discount lookups keep the
    curves' ACT_365. On flat conventions that makes an ACT_360 deal exactly 365/360 of the ACT_365
    one - the ratio is what says the two day counts are doing different jobs rather than one
    silently standing in for the other."""
    pays = _pays()
    priced = _price(hazards=[0.02], max_defaults=1, Accrual_Day_Count='ACT_360')
    assert priced == pytest.approx(_pv([(1.0, 0.02)], pays, basis=360.0), rel=3e-7), (
        f'priced {priced:.12g} against the /360 closed form '
        f'{_pv([(1.0, 0.02)], pays, basis=360.0):.12g}')

    # MUTATE: the /365 closed form is 1.4% away, and it is what the discount curve's day count
    # would have given - the defect this gate exists for
    assert priced != pytest.approx(_pv([(1.0, 0.02)], pays), rel=1e-3), (
        'the accrual read the discount curve day count')


def test_the_index_proxy_is_the_identity_at_the_base_date():
    """g scales every name's hazard by H_index(samples) over H_index of the SAME samples at t0.
    One deal-time row makes those the same gather, so g is bitwise one and the CDS_Index curve
    cannot reach the price - exact equality, not a tolerance."""
    flat = _price(hazards=[0.02], max_defaults=1)
    assert _price(hazards=[0.02], max_defaults=1, index_hazard=0.5) == flat, (
        'the index curve moved a base valuation, so g is not the identity')

    # a FLAT index curve cancels its own level in any ratio of two gathers, so the line above sees
    # "g is a ratio", not "g is one". A curve with a shape does see the difference
    humped = utils.Curve([], [[1e-6, 1e-7], [0.5, 0.10], [10.0, 0.15]])
    assert _price(hazards=[0.02], max_defaults=1, index_curve=humped) == flat, (
        'a shaped index curve moved a base valuation, so g varies across the samples')

    # MUTATE: a hazard move of that size on the NAME has to be seen, or the equalities above are
    # reporting a dead harness rather than a collapsed proxy
    assert _price(hazards=[0.5], max_defaults=1) != pytest.approx(flat, rel=1e-3), (
        'a 25x name hazard changed nothing - this harness cannot see a curve')


def test_the_accrual_window_opens_at_the_effective_date():
    """Effective in 6m, quarterly to 18m: the first coupon covers 6m to 9m and the engine now
    opens the window there. It used to open at t + 1 day and accrue 51% too much - the same defect
    the spot gates absorbed as a one-day drop, at a size nothing can absorb."""
    pays = _pays(9, 18)
    effective = (BASE + pd.DateOffset(months=6) - BASE).days
    priced = _price(hazards=[0.02], max_defaults=1, effective=6, maturity=18)
    assert priced == pytest.approx(_pv([(1.0, 0.02)], pays, start=effective), rel=3e-7), (
        f'priced {priced:.12g} against an effective-date accrual '
        f'{_pv([(1.0, 0.02)], pays, start=effective):.12g}')

    # MUTATE: the old rule - accrue from tomorrow - is 51% larger, so this gate is the defect's
    # own measurement and cannot pass under either reading by accident
    assert priced != pytest.approx(_pv([(1.0, 0.02)], pays, start=1), rel=1e-2), (
        'a window opening at t + 1 day also matched')


def test_every_declared_day_count_prices_finite():
    """The 30/360 walkers reconstruct dates by accumulating increments from a reference date,
    which the pricer's sampled accrual grid does not carry - handed one they raise, the
    log_exception guard swallows it, and the deal silently marks NaN. So the declaration excludes
    them: a value the engine cannot honor is the same defect as a field nothing reads. This gate
    holds the declaration to what actually prices, so putting a value back means making it work."""
    from derivus import instruments as inst
    declared = next(
        f.values for g in inst.CreditNthToDefault.fields for f in getattr(g, 'fields', [])
        if getattr(f, 'key', None) == 'Accrual_Day_Count')
    assert declared == ['ACT_365', 'ACT_360', 'ACT_365_ISDA', 'ACT_ACT_ICMA'], (
        f'the declared list changed to {declared} - extend this gate with the new value')

    act365 = _price(hazards=[0.02], max_defaults=1)
    for value in declared:
        priced = _price(hazards=[0.02], max_defaults=1, Accrual_Day_Count=value)
        assert np.isfinite(priced) and priced != 0.0, f'{value} did not price: {priced}'
        # ACT_365_ISDA and ACT_ACT_ICMA share the /365 measure; ACT_360 is the /360 rescale
        assert priced == pytest.approx(
            act365 * (365.0 / 360.0 if value == 'ACT_360' else 1.0), rel=1e-12), (
            f'{value} moved the price off its measure: {priced:.12g}')
