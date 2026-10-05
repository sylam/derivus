"""A floating energy leg is the sum of its cashflows, whatever currency its forward curve is in, and an
energy option is Black on its sample average.

`pv_energy_cashflows` prices a leg one block of rows at a time - a block per set of cashflows still
unpaid - and converts every forecast fixing into the payoff currency at the forward FX seen from its
own row. Rows paying on one day are summed onto it before that day's discount. `pv_energy_option`
moment-matches the average, each sample's variance accrued from the row's own time to the sample's
day - from the base date, on a base valuation's one row.
"""

import json
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

import derivus
import test_declared_defaults as book
import trial_commodity
from derivus import utils
from derivus.config import CustomJsonEncoder
from derivus.stochasticprocess import CSForwardPriceModel
from test_position_scaling import document, marks

BASE = pd.Timestamp('2026-01-15')
#: (period start, period end, payment) of two delivery months
FEB = ('2026-02-02', '2026-02-27', '2026-03-05')
MAR = ('2026-03-02', '2026-03-31', '2026-04-06')
#: a third month, and the first paid on the second's day
APR = ('2026-04-01', '2026-04-30', '2026-05-06')
FEB_PAID_WITH_MAR = FEB[:2] + MAR[2:]
EXCEL = pd.Timestamp('1899-12-30')


def rate(currency, rates):
    return {'Currency': currency, 'Day_Count': 'ACT_365', 'Sub_Type': None,
            'Curve': utils.Curve([], [[0.0, rates[0]], [5.0, rates[1]]])}


#: a EUR forward curve under a Clewlow-Strickland model, paying USD at a static EURUSD
MARKET = {
    'System Parameters': {'Base_Currency': 'USD', 'Base_Date': BASE},
    'Model Configuration': {'.ModelParams': {
        'modeldefaults': {'ForwardPrice': 'CSForwardPriceModel'}, 'modelfilters': {}}},
    'Price Models': {'CSForwardPriceModel.FUEL': {'Alpha': 0.35, 'Drift': 0.0, 'Sigma': 0.2}},
    'Price Factors': {
        'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Spot': 1.0},
        'FxRate.EUR': {'Domestic_Currency': 'USD', 'Interest_Rate': 'EUR', 'Spot': 1.1},
        'InterestRate.USD': rate('USD', (0.041, 0.033)),
        'InterestRate.EUR': rate('EUR', (0.020, 0.025)),
        'ReferencePrice.FUEL': {'Fixing_Curve': utils.Curve([], [[45900, 45900], [46600, 46600]]),
                                'ForwardPrice': 'FUEL'},
        'ForwardPrice.FUEL': {'Currency': 'EUR',
                              'Curve': utils.Curve([], [[45900, 1450.0], [46600, 1550.0]])},
        'ForwardPriceSample.DAILY': {'Offset': 0, 'Holiday_Calendar': '',
                                     'Sampling_Convention': 'ForwardPriceSampleDaily'}}}


def leg(reference, *months):
    return {'Instrument': {'.Deal': {
        'Object': 'FloatingEnergyDeal', 'Reference': reference, 'Currency': 'USD',
        'Discount_Rate': 'USD', 'Reference_Type': 'FUEL', 'Sampling_Type': 'DAILY',
        'FX_Sampling_Type': '', 'Payer_Receiver': 'Receiver', 'Payments': {'Items': [
            {'Period_Start': pd.Timestamp(start), 'Period_End': pd.Timestamp(end),
             'Payment_Date': pd.Timestamp(paid), 'Volume': 2500.0, 'Realized_Average': 0.0,
             'FX_Realized_Average': 0.0} for start, end, paid in months]}}}}


def results(*legs):
    """The results of a daily credit Monte Carlo over `legs`, held under one netting set."""
    job = {'Calc': {
        'Calculation': {'Object': 'CreditMonteCarlo', 'Base_Date': BASE, 'Currency': 'USD',
                        'Batch_Size': 16, 'Simulation_Batches': 1, 'Random_Seed': 1,
                        'Deflation_Interest_Rate': 'USD', 'Time_Grid': '0d 1d(1d)'},
        'Deals': {'Reference': 'fx', 'Deals': {'Children': [{
            'Instrument': {'.Deal': {'Object': 'NettingCollateralSet', 'Reference': 'NS',
                                     'Netted': 'True', 'Collateralized': 'False'}},
            'Children': list(legs)}]}},
        'MergeMarketData': {'ExplicitMarketData': MARKET}}}
    context = derivus.Context()
    context.load_json((json.dumps(job, cls=CustomJsonEncoder), 'energy_leg_fx'))
    return context.run_job()[1]['Results']


def relative_gap(joined, split):
    """The largest gap on any row, relative to that row's largest value."""
    return ((joined - split).abs().max(axis=1) / split.abs().max(axis=1).clip(lower=1.0)).max()


def test_a_two_cashflow_leg_on_a_foreign_curve_prices_as_its_two_legs():
    """A leg on a EUR forward curve paying USD, run as a daily credit Monte Carlo, marks on every
    row and path as its two cashflows held as two legs - on the same paths, since both documents
    simulate the same factors on the same grid. The rows after the first payment are the leg's
    second block, where only the second cashflow is left to price.

    Killing mutation: the forward FX read off the whole grid (`discounts.time_grid`) rather than
    the block's own rows, which converts the second cashflow's fixings at the first block's dates
    and misses on every row after the first payment that still has a fixing to come.
    """
    joined = results(leg('FEB_MAR', FEB, MAR))['mtm']
    split = results(leg('FEB', FEB), leg('MAR', MAR))['mtm']
    assert relative_gap(joined, split) <= 1e-6, relative_gap(joined, split)


def fuel_forward(day):
    """`trial_commodity`'s fuel forward at `day`, linear in the Excel serial between its knots."""
    knots = trial_commodity.FACTORS['ForwardPrice.FUEL']['Curve'].array
    return np.interp((pd.DatetimeIndex(day) - EXCEL).days, knots[:, 0], knots[:, 1])


def trial(reference, **terms):
    """`trial_commodity`'s deal `reference` with `terms` over it, as its one-deal wire document."""
    deal = dict(next(d for d in trial_commodity.DEALS if d['Reference'] == reference), **terms)
    return document(SimpleNamespace(DEALS=[deal], FACTORS=trial_commodity.FACTORS,
                                    CONFIGURATION={}))


def discount(rate, day):
    """A flat continuously compounded ACT/365 discount factor from the trial base date."""
    return math.exp(-rate * (day - trial_commodity.B).days / 365.0)


def test_an_energy_option_accrues_each_sample_s_variance_from_the_base_date():
    """`trial_commodity`'s FUEL_OPT - a call struck at 82 on the daily average of the fuel forward
    over one delivery month, 30% flat - marks as Black on that average's two moments under a base
    valuation, whose one row is the base date: each sample's variance accrued over its own days from
    it and each converted at its own fx forward, paid in USD on the fuel's USD curve, and paid in
    EUR struck at 65, the compo branch, with the 15% EURUSD vol uncorrelated.

    Reference, by hand: `G_i = F_i X_i`, `F_i` the forward at sample i's serial, `X_i` one in USD
    and `exp((r_EUR - r_USD) t_i) / 1.25` EUR per USD, `t_i = days_i / 365`; `M1 = mean G`,
    `M2 = sum_ij G_i G_j exp(s2 min(t_i, t_j)) / n^2`, `s2` the fuel variance plus the fx one;
    `Volume DF(settle) Black(M1, K, sqrt(log M2 / M1^2))`, crossed to USD at 1.25.

    Killing mutations: a sample's variance time read off `Start_Day`, its Excel serial - 72,799.24
    against 5,532.62 - and the compo's fx forward dated there, 127 years out.
    """
    deal = next(d for d in trial_commodity.DEALS if d['Reference'] == 'FUEL_OPT')
    samples = pd.bdate_range(deal['Period_Start'], deal['Period_End'])
    t = (samples - trial_commodity.B).days.to_numpy() / 365.0
    forwards = fuel_forward(samples)
    fuel_vol = trial_commodity.FACTORS['CommodityPriceVol.FUEL']['Surface'].array[0, 2]
    r_usd = book.DISCOUNT[0]

    def black_on_the_average(g, variance, strike):
        m1 = g.mean()
        m2 = (np.outer(g, g) * np.exp(variance * np.minimum.outer(t, t))).sum() / g.size ** 2
        total = math.sqrt(math.log(m2 / m1 ** 2))
        d1 = (math.log(m1 / strike) + 0.5 * total ** 2) / total
        return deal['Volume'] * (m1 * norm.cdf(d1) - strike * norm.cdf(d1 - total))

    usd = discount(r_usd, deal['Settlement_Date']) * black_on_the_average(
        forwards, fuel_vol ** 2, deal['Strike'])
    eur = book.X0 * discount(book.R_EUR, deal['Settlement_Date']) * black_on_the_average(
        forwards * np.exp((book.R_EUR - r_usd) * t) / book.X0, fuel_vol ** 2 + book.SIGMA ** 2,
        65.0)
    for job, reference in ((trial('FUEL_OPT'), usd),
                           (trial('FUEL_OPT', Currency='EUR', Discount_Rate='EUR', Strike=65.0), eur)):
        mark = float.fromhex(marks(job)['FUEL_OPT'])
        assert abs(mark - reference) <= 1e-12 * reference, (mark, reference)


def test_an_energy_leg_paying_two_periods_on_one_day_beside_a_third_is_its_hand_sum():
    """`trial_commodity`'s floating fuel leg over three delivery months, the first two paid on the
    second's day and the third on its own - beside its controls, every period on its own day (the
    trial as written) and all three on one day - marks as the sum of its periods.

    Reference, by hand: `sum Volume (Price_Multiplier mean F + Fixed_Basis) DF(pay)`, the mean over
    the period's weekdays of the forward at each one's serial, USD at 4%.

    Killing mutation: the payments left one column per period against the discounts' one per pay
    day, which is the 3-against-2 shape the shared day marked NaN on.
    """
    fuel = next(d for d in trial_commodity.DEALS if d['Reference'] == 'FUEL_FLOAT')

    def item(n, paid):
        return dict(trial_commodity.period(n), Payment_Date=paid, Volume=5000.0 * n,
                    Fixed_Basis=2.5, Price_Multiplier=1.0, Realized_Average=0.0,
                    FX_Realized_Average=0.0)

    paid = trial_commodity.paid
    for reference, items in (('FUEL_SHARED_DAY', [item(2, paid(3)), item(3, paid(3)), item(4, paid(4))]),
                             ('FUEL_FLOAT', fuel['Payments']['Items']),
                             ('FUEL_ONE_DAY', [item(n, paid(4)) for n in (2, 3, 4)])):
        hand = sum(row['Volume'] * discount(book.DISCOUNT[0], row['Payment_Date']) * (
            row['Price_Multiplier'] * fuel_forward(pd.bdate_range(
                row['Period_Start'], row['Period_End'])).mean() + row['Fixed_Basis'])
            for row in items)
        job = trial('FUEL_FLOAT', Reference=reference, Payments={'Items': items})
        mark = float.fromhex(marks(job)[reference])
        assert abs(mark - hand) <= 1e-12 * hand, (reference, mark, hand)


def test_a_day_paying_two_periods_books_both_under_a_credit_monte_carlo():
    """The leg paying February's period on March's day beside April's, run as a daily credit Monte
    Carlo, marks on every row and path, and books on each pay day the cash, of its three periods
    held as three legs on the same paths.

    The reference is the engine's own run of the three one-period legs, each paying on a day of its
    own - a consistency check between two readings of one leg; the base valuation beside this carries
    the hand sum.

    Killing mutation: the day's cash booked off its first row alone, which leaves March's payment
    out of the cash on the shared day.
    """
    joined = results(leg('SHARED', FEB_PAID_WITH_MAR, MAR, APR))
    split = results(leg('FEB', FEB_PAID_WITH_MAR), leg('MAR', MAR), leg('APR', APR))
    assert relative_gap(joined['mtm'], split['mtm']) <= 1e-6
    cash, split_cash = joined['cashflows']['USD'], split['cashflows']['USD']
    paid = [pd.Timestamp(MAR[2]), pd.Timestamp(APR[2])]
    assert (cash.loc[paid].abs().min(axis=1) > 0).all(), cash.loc[paid]
    assert relative_gap(cash, split_cash) <= 1e-6, relative_gap(cash, split_cash)


def test_an_implied_forward_variance_holds_its_digits_over_a_day_in_float32():
    """`CSForwardPriceModel` on an implied Alpha 5%, Sigma 30% in float32: the first day's vol of
    a contract five years out is s e^{-a tau} sqrt((1 - e^{-2a dt}) / 2a), against `math.expm1` in
    float64 under 1e-6.

    Killed by: the old spelling `1 - exp(-2a t)`, 3.0e-5 off."""
    expiry = float((BASE - EXCEL).days + 1826)
    process = CSForwardPriceModel(SimpleNamespace(get_tenor=lambda: np.array([expiry])), {})
    process.precalculate(BASE, SimpleNamespace(scen_time_grid=np.arange(3.0)), torch.ones(1), None,
                         0, implied_tensor={'Alpha': torch.tensor(0.05), 'Sigma': torch.tensor(0.3)})
    dt, tau = 1.0 / utils.DayCount.DAYS_IN_YEAR, 1825.0 / utils.DayCount.DAYS_IN_YEAR
    want = 0.3 * math.exp(-0.05 * tau) * math.sqrt(-math.expm1(-0.1 * dt) / 0.1)
    assert process.vol.dtype == torch.float32 and abs(float(process.vol[1, 0, 0]) / want - 1.0) < 1e-6
