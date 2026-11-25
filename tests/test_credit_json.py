"""The credit family through the JSON contract, priced off `trial_credit`'s names in the
declared-defaults world."""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
import torch

import derivus
import rates_world
import test_declared_defaults as book
import trial_credit
from derivus import spine, utils
from derivus.config import CustomJsonEncoder

B = book.WORLD_BASE

#: The trial default swap without its amortisation: ten million protected on ISSUER_A, five years.
CDS ={key: value for key, value in trial_credit.DEALS[0].items() if key != 'Amortisation'}


def discount(day, rate=0.04):
    return math.exp(-rate * (day - B).days / 365.0)


def by_hand(recovery=0.4, nominal=lambda end: 1e7, survival=lambda day: discount(day, 0.02)):
    """The trial swap bought, over each quarter (s, e] at a 100bp running coupon, quarterly ACT/365
    with the maturity day counted, hazard 2% on the flat 4% USD curve:

        V = sum N(e) [ (1 - R) (D(s) + D(e)) / 2 (S(s) - S(e)) - c a S(e) D(e) ]
    """
    dates = [B + pd.DateOffset(months=3 * k) for k in range(21)]
    return sum(nominal(e) * (
        (1.0 - recovery) * (discount(s) + discount(e)) / 2 * (survival(s) - survival(e))
        - 0.01 * ((e - s).days + (e == dates[-1])) / 365.0 * survival(e) * discount(e))
        for s, e in zip(dates, dates[1:]))


def test_an_upfront_is_paid_by_the_protection_buyer_on_its_day():
    """The trial default swap without its amortisation and a 2% upfront three days out: by hand,
    `by_hand() - 2% N D(u)`, the upfront paid by the protection buyer whatever the name does, so
    discounted and never survival-weighted: -110,464.68 bought and its negative sold. Stated as a
    bare 2.0 it is 2%, as the bare coupon 1.0 is 1%. Stated without a date it is paid on the
    effective date; dated behind the base date it has been paid, and the mark is the plain swap's
    to the bit. A credit Monte Carlo books it as the buyer's cash on its day.

    KILLING MUTATION: the upfront term dropped from `pv_credit_cashflows`: the bought swap reads
    89,469.57 against -110,464.68.
    """
    plain = by_hand()
    day = B + pd.Timedelta(days=3)
    upfront = dict(CDS, Reference='UPFRONT', Upfront=utils.Percent(2.0), Upfront_Date=day)
    hexes, _ = book.marks([CDS, upfront, dict(upfront, Reference='SOLD', Buy_Sell='Sell'),
                           dict(upfront, Reference='BARE', Upfront=2.0),
                           dict(upfront, Reference='UNDATED', Upfront_Date=None),
                           dict(upfront, Reference='PAID', Upfront_Date=B - pd.Timedelta(days=3))],
                          trial_credit.FACTORS)
    valued = {reference: float.fromhex(value) for reference, value in hexes.items()}

    assert valued['CDS'] == pytest.approx(plain, rel=1e-12)
    assert valued['UPFRONT'] == pytest.approx(plain - 2e5 * discount(day), rel=1e-12)
    assert valued['SOLD'] == -valued['UPFRONT']
    assert valued['BARE'] == valued['UPFRONT']
    assert valued['UNDATED'] == pytest.approx(plain - 2e5, rel=1e-12)
    assert valued['PAID'] == valued['CDS']

    _, out = book.simulated([upfront], ('USD',), trial_credit.FACTORS, Generate_Cashflows='Yes')
    assert float(out['Results']['cashflows']['USD'].loc[day].iloc[0]) == pytest.approx(
        -2e5, rel=1e-6)


def test_a_survival_curve_ending_before_the_maturity_hazards_on_at_its_last_rate():
    """ISSUER_A written at a 1% hazard to one year and 4% to two: past its last knot a cumulative
    hazard goes on at its last rate, H = max(1% t, 4% t - 3%), and the five-year swap marks
    `by_hand()` off it, 420,785.05, where it read no default after its second year.

    KILLING MUTATIONS: the curve read flat past its last knot, -156,936.29; carried on at its
    average hazard H_N t / T_N, 212,000.51.
    """
    two = dict(trial_credit.FACTORS, **{'SurvivalProb.ISSUER_A': dict(
        trial_credit.FACTORS['SurvivalProb.ISSUER_A'],
        Curve=utils.Curve([], [[0.0, 0.0], [1.0, 0.01], [2.0, 0.05]]))})
    hexes, _ = book.marks([CDS], two)
    assert float.fromhex(hexes['CDS']) == pytest.approx(by_hand(
        survival=lambda day: min(discount(day, 0.01), discount(day) * math.exp(0.03))), rel=1e-12)


def loaded(base, curve=None):
    """A context loaded with a credit Monte Carlo valued on `base` (a date string) beside its CVA
    gradient: a million paid in a year under one netting set against ISSUER_A - a flat 2% hazard, or
    `curve` (years, negative log survival) - on the flat 4% USD curve, CDS tenors 1y, 3y and 5y."""
    factors = dict(trial_credit.FACTORS)
    if curve is not None:
        factors['SurvivalProb.ISSUER_A'] = dict(factors['SurvivalProb.ISSUER_A'],
                                                Curve=utils.Curve([], curve))
    cashflow = {'Object': 'FixedCashflowDeal', 'Reference': 'CF', 'Currency': 'USD',
                'Discount_Rate': 'USD', 'Amount': 1e6, 'Payment_Date': B + pd.DateOffset(years=1)}
    job = basket([cashflow], factors)
    calc = job['Calc']
    calc['Calculation'] = {
        'Object': 'CreditMonteCarlo', 'Base_Date': {'.Timestamp': base}, 'Currency': 'USD',
        'Time_Grid': '0d 3m(3m)', 'Batch_Size': 16, 'Random_Seed': 1,
        'Deflation_Interest_Rate': 'USD', 'Credit_Valuation_Adjustment': {
            'Calculate': 'Yes', 'Counterparty': 'ISSUER_A', 'Gradient': 'Yes',
            'Deflate_Stochastically': 'No', 'Stochastic_Hazard_Rates': 'No',
            'CDS_Tenors': [1.0, 3.0, 5.0]}}
    calc['Deals']['Deals']['Children'] = [{'Instrument': {'.Deal': {
        'Object': 'NettingCollateralSet', 'Reference': 'NS', 'Netted': 'True',
        'Collateralized': 'False'}}, 'Children': calc['Deals']['Deals']['Children']}]
    market = calc['MergeMarketData']['ExplicitMarketData']
    market['System Parameters']['Base_Date'] = {'.Timestamp': base}
    market['Model Configuration'] = {'.ModelParams': {
        'modeldefaults': {'InterestRate': 'HullWhite1FactorInterestRateModel'}, 'modelfilters': {}}}
    market['Price Models'] = {'HullWhite1FactorInterestRateModel.USD': {
        'Alpha': 0.05, 'Lambda': 0.0, 'Quanto_FX_Correlation': 0.0,
        'Quanto_FX_Volatility': {'.Curve': {'meta': [], 'data': [[0.0, 0.0], [10.0, 0.0]]}},
        'Sigma': {'.Curve': {'meta': [], 'data': [[0.0, 0.01], [10.0, 0.01]]}}}}
    context = derivus.Context()
    context.load_json((json.dumps(job), 'cs01'))
    return context


def cs01(base, curve=None):
    """`Results['CS01']` of the job `loaded` builds."""
    return loaded(base, curve).run_job()[1]['Results']['CS01']


def test_a_run_leaves_the_context_it_ran_on_as_it_was_loaded():
    """`CDS_Tenors` re-knots the counterparty's survival curve to the CDS dates on the run's own
    copy of the factor, so the context a job was loaded on keeps the curve it loaded, and a second
    run on it reads the same CS01 to the bit.

    Killing mutation: the re-knot written onto the context's factor, whose curve then carries the
    CDS knots.
    """
    context = loaded('2026-03-20')
    survival = context.current_cfg.params['Price Factors']['SurvivalProb.ISSUER_A']['Curve']
    knots = survival.array.copy()
    first = context.run_job()[1]['Results']['CS01']
    assert np.array_equal(survival.array, knots), 'the run re-knotted the context\'s own curve'
    assert context.run_job()[1]['Results']['CS01'].equals(first)


def par_rates(frame):
    """`{tenor: par CDS rate}` off a CS01 frame's column labels."""
    return dict(zip(frame.columns.get_level_values('Tenor'),
                    frame.columns.get_level_values('Par CDS')))


def test_a_par_cds_rate_holds_across_the_quarter_and_a_tenor_bump_holds_the_other_tenors():
    """The CS01 a credit Monte Carlo reports beside its CVA gradient: ISSUER_A's par CDS rates under
    the ISDA standard model at 1y, 3y and 5y, a flat 2% hazard on the flat 4% USD curve. The 5y rate
    is (1 - R)h to the coupon effect, 1.2%, and the same on every day of a quarter - 20 March, the
    20th of April and of May, 19 June - the accrued since the last standard date SUBTRACTED and the
    first coupon the NEXT standard date. Each tenor's row is the log-survival curve a basis point on
    that tenor alone implies: valued on it, that tenor's rate is a basis point up and every other
    tenor's where it was, the later segments re-solved.

    Killing mutations: the accrued added, the 5y rate falling from 1.206% to 1.081% across the
    quarter; the first standard date a quarter late on or after the 20th of a non-roll month (20
    April and 20 May to 20 September), a 3.8e-6 spread; the bump carried on its own segment alone,
    the later segments not re-solved, the 3y rate taking 0.39bp of the 1y basis point against 1e-13.
    """
    quarter = {day: cs01(day) for day in ('2026-03-20', '2026-04-20', '2026-05-20', '2026-06-19')}
    five = {day: par_rates(frame)[5.0] for day, frame in quarter.items()}
    assert max(five.values()) - min(five.values()) < 2e-6, five
    assert abs(five['2026-03-20'] - 0.6 * 0.02) < 1e-4, five

    frame = quarter['2026-04-20']
    par, knots = par_rates(frame), frame.index.to_numpy()
    for (bumped, _), shift in frame.items():
        moved = par_rates(cs01('2026-04-20', np.column_stack(
            [knots, 0.02 * knots + shift.to_numpy()]).tolist()))
        for tenor in par:
            assert abs(moved[tenor] - par[tenor] - 1e-4 * (tenor == bumped)) < 1e-13, (
                bumped, tenor, moved[tenor] - par[tenor])


def test_a_premium_is_announced_with_the_sign_its_cash_moves():
    """The default swap's and the basket's premium schedules carry the protection buyer's minus,
    so the diary announces every bought premium paid and every sold one received - the sign the
    engine's own cash books - while the trial swap, amortising 4m at two years, still marks its
    hand formula and each side its mirror to the bit.

    KILLING MUTATION: the buyer's minus left in the pricers as well, which flips both marks.
    """
    deals = [dict(deal, Reference=deal['Reference'] + side, Buy_Sell=side)
             for deal in trial_credit.DEALS[:2] for side in ('Buy', 'Sell')]
    job = book.book(deals)
    job['Calc']['MergeMarketData']['ExplicitMarketData']['Price Factors'].update(
        trial_credit.FACTORS)
    context = derivus.Context()
    context.load_json((json.dumps(job, cls=CustomJsonEncoder), 'signs'))
    signs = {}
    for row in spine.diary(context):
        if row['kind'] == 'payment':
            signs.setdefault(row['reference'], set()).add(math.copysign(1.0, row['amount']))
    assert signs == {'CDSBuy': {-1.0}, 'CDSSell': {1.0}, 'NTDBuy': {-1.0}, 'NTDSell': {1.0}}

    valued, _ = book.marks(deals, trial_credit.FACTORS)
    amortised = by_hand(nominal=lambda end: 1e7 if end <= B + pd.DateOffset(years=2) else 6e6)
    assert float.fromhex(valued['CDSBuy']) == pytest.approx(amortised, rel=1e-12)
    assert float.fromhex(valued['NTDBuy']) < 0.0, 'a premium leg is worth nothing but a payment'
    for name in ('CDS', 'NTD'):
        assert float.fromhex(valued[name + 'Sell']) == -float.fromhex(valued[name + 'Buy'])


def basket(deals=(trial_credit.DEALS[1],), factors=None, **calculation):
    """The wire document of `deals` beside the trial's names, `calculation` over its block."""
    job = book.book(list(deals))
    job['Calc']['MergeMarketData']['ExplicitMarketData']['Price Factors'].update(
        factors or trial_credit.FACTORS)
    job['Calc']['Calculation'].update(calculation)
    return json.loads(json.dumps(job, cls=CustomJsonEncoder))


def simulate(job, models, rate_model=None, grid='0d 3m(3m)', sigma=0.01):
    """`job` as a float64 credit Monte Carlo on `grid` over one netting set, a hazard model per name
    in `models` and a Hull-White of volatility `sigma` on `rate_model`'s curve:
    `(calculation, {reference: profile})`."""
    calc = job['Calc']
    calc['Calculation'] = dict(
        Object='CreditMonteCarlo', Base_Date=calc['Calculation']['Base_Date'], Currency='USD',
        Time_Grid=grid, Batch_Size=256, Random_Seed=1, Deflation_Interest_Rate='USD',
        DealLevel=True, Generate_Cashflows='No')
    calc['Deals']['Deals']['Children'] = [{'Instrument': {'.Deal': {
        'Object': 'NettingCollateralSet', 'Reference': 'NS', 'Netted': 'True',
        'Collateralized': 'False'}}, 'Children': calc['Deals']['Deals']['Children']}]
    market = calc['MergeMarketData']['ExplicitMarketData']
    market['Model Configuration'] = {'.ModelParams': {'modelfilters': {}, 'modeldefaults': dict(
        [('SurvivalProb', 'HWHazardRateModel')] * bool(models) +
        [('InterestRate', 'HullWhite1FactorInterestRateModel')] * bool(rate_model))}}
    market['Price Models'] = {'HWHazardRateModel.' + name: {
        'Alpha': 0.1, 'Lambda': 0.0, 'Sigma': 0.01} for name in models}
    if rate_model:
        market['Price Models']['HullWhite1FactorInterestRateModel.' + rate_model] = {
            'Alpha': 0.05, 'Lambda': 0.0, 'Quanto_FX_Correlation': 0.0,
            'Quanto_FX_Volatility': {'.Curve': {'meta': [], 'data': [[0.0, 0.0], [10.0, 0.0]]}},
            'Sigma': {'.Curve': {'meta': [], 'data': [[0.0, sigma], [10.0, sigma]]}}}
    context = derivus.Context()
    context.load_json((json.dumps(job), 'basket'))
    calc, _ = derivus.run_cmc(context.current_cfg, prec=torch.float64)
    return calc, {deal.Instrument.field['Reference']: np.asarray(deal.Calc_res['Value'][0])
                  for deal in calc.netting_sets.deals() if (deal.Calc_res or {}).get('Value')}


def test_a_one_name_basket_at_zero_correlation_pays_while_every_name_survives():
    """At zero correlation the copula's names are independent, so a first-to-default coupon earns
    `c prod_j S_j(t)` at every horizon - the Poisson-binomial at no default - and the premium leg is
    that rate summed by the deal's own trapezoid on 30-day samples, `-sum_i N_i D(p_i)
    sum (E(a) + E(b)) / 2 (b - a)`, with `E(t) = 2.5% exp(-6.5% t)` on the trial's three names.

    KILLING MUTATION: the step-down rate read at one default, `(1 - (k0 + k)) / n` shifted by one.
    """
    one = dict(trial_credit.DEALS[1], Reference='ONE', Max_Defaults=1, Correlation=0.0)
    valued = float.fromhex(book.marks([one], trial_credit.FACTORS)[0]['ONE'])
    pays = [(B + pd.DateOffset(months=3 * k) - B).days for k in range(13)]
    rate = lambda day: 0.025 * math.exp(-0.065 * day / 365.0)
    hand = 0.0
    for start, end in zip(pays, pays[1:]):
        samples = sorted(set(range(start, end, 30)) | {end})
        accrued = sum((rate(a) + rate(b)) / 2 * (b - a) / 365.0
                      for a, b in zip(samples, samples[1:]))
        nominal = 5e6 if B + pd.Timedelta(days=end) <= B + pd.DateOffset(months=18) else 4e6
        hand -= nominal * accrued * math.exp(-0.04 * end / 365.0)
    assert valued == pytest.approx(hand, rel=1e-10)


def test_a_basket_s_first_order_greeks_are_its_central_difference():
    """The step-down recurrence builds a new tensor per name, so the graph keeps every factor its
    products saved and `Greeks: First` runs: the trial basket's sensitivity to ISSUER_A's 10y
    cumulative hazard is the central difference of two base valuations 1e-5 either side of it.

    KILLING MUTATION: the copula's odds detached from the graph, which drops every name's hazard
    from the gradient.
    """
    context = derivus.Context()
    context.load_json((json.dumps(basket(Greeks='First')), 'greeks'))
    first = context.run_job()[1]['Results']['Greeks_First']
    first = first.xs('SurvivalProb.ISSUER_A', level='Rate')
    delta = float(first[first.index.get_level_values('Tenor') == 10.0]['root'].iloc[0])

    def bumped(h):
        factors = dict(trial_credit.FACTORS, **{'SurvivalProb.ISSUER_A': {
            'Recovery_Rate': 0.4, 'Curve': utils.Curve([], [[0.0, 0.0], [10.0, 0.2 + h]])}})
        return float.fromhex(book.marks([trial_credit.DEALS[1]], factors)[0]['NTD'])

    assert delta == pytest.approx((bumped(1e-5) - bumped(-1e-5)) / 2e-5, rel=1e-6)


def test_a_basket_on_simulated_hazards_prices_every_row():
    """Under a credit Monte Carlo whose hazards are SIMULATED each name's curve is split with its
    block like the index's, so the trial basket prices a dispersed profile opening on its base
    valuation, where it met a three-row block with all 25 rows and was skipped; on static curves
    its mark is unmoved at the number the trial reads, to the bit.

    KILLING MUTATION: the names read over every row rather than their block's.
    """
    mark = book.marks([trial_credit.DEALS[1]], trial_credit.FACTORS)[0]['NTD']
    assert mark == '-0x1.28cf4f83a85a8p+18'
    _, profiles = simulate(basket(), ('ISSUER_A', 'ISSUER_B', 'ISSUER_C', 'ISSUER_INDEX'))
    assert 'NTD' in profiles, 'the basket was skipped'
    assert profiles['NTD'][0] == pytest.approx(float.fromhex(mark), rel=1e-12)
    assert np.isfinite(profiles['NTD']).all() and profiles['NTD'][3].std() > 0.0


def test_a_basket_on_static_curves_rolls_to_its_base_valuation():
    """On flat static curves the index has not moved, so the names' conditional scale is one on
    every row - each row's own horizons read off today's index, where every row read the block's
    first row's and scaled by `(s - t) / s` - and the credit Monte Carlo's profile is the base
    valuation rolled to each row's date. The one model the run needs walks USD-PROJ, which a FRA
    beside the basket forecasts on and the basket never reads.

    KILLING MUTATION: the base index read at the block's first row's horizons for every row.
    """
    fra = rates_world.fra('FRA', 'USD', 'USD-PROJ', 'USD', 3, 6, 4.3)
    calc, profiles = simulate(basket([trial_credit.DEALS[1], fra]), (), rate_model='USD-PROJ')
    deal = next(d for d in calc.netting_sets.deals() if d.Instrument.field['Reference'] == 'NTD')
    days = calc.time_grid.mtm_time_grid[deal.Time_dep.deal_time_grid]
    for row in (1, 2, 4, 12):
        rolled = basket(Base_Date=B + pd.Timedelta(days=int(days[row])))
        market = rolled['Calc']['MergeMarketData']['ExplicitMarketData']
        market['System Parameters']['Base_Date'] = rolled['Calc']['Calculation']['Base_Date']
        context = derivus.Context()
        context.load_json((json.dumps(rolled), 'rolled'))
        frame = context.run_job()[1]['Results']['mtm']
        value = float(frame[frame['Reference'] == 'NTD']['Value'].iloc[0])
        assert profiles['NTD'][row].mean() == pytest.approx(value, rel=1e-12), (row, days[row])


def test_a_digital_default_swap_recovers_its_stated_share():
    """`Is_Digital: Yes` pays the protection at `Digital_Recovery` in place of the name's own 40%:
    at a 25% digital recovery the swap marks `by_hand(0.25)`, a bare 25.0 reading as the percent it
    is declared; `Is_Digital: No` beside a stated digital recovery is the plain swap to the bit.

    KILLING MUTATION: the pricer reading the curve's recovery whatever the compile carries.
    """
    digital = dict(CDS, Reference='DIGITAL', Is_Digital='Yes', Digital_Recovery=utils.Percent(25.0))
    hexes, _ = book.marks([CDS, digital, dict(digital, Reference='BARE', Digital_Recovery=25.0),
                           dict(digital, Reference='NOT', Is_Digital='No')], trial_credit.FACTORS)
    assert float.fromhex(hexes['DIGITAL']) == pytest.approx(by_hand(0.25), rel=1e-12)
    assert hexes['BARE'] == hexes['DIGITAL'] and hexes['NOT'] == hexes['CDS']


def test_the_cash_account_accrues_each_interval_at_the_rate_its_start_reads():
    """The trial's million in a cash account for a year, on a USD curve whose zero rate rises
    linearly from 1% to 5% over two years and a Hull-White at zero volatility: a deterministic
    world whose simulated curves are its forwards, exact between their knots, so an account
    accruing each interval at the rate its start reads is `1e6 exp(R(t) t)` on every row, on a
    weekly, a monthly and a half-yearly grid alike - 1,030,454.53 at the horizon - to 1e-10.

    KILLING MUTATION: each interval accrued at the rate its END reads, the reading this replaced -
    the horizon at 1,031,243.47 weekly, 1,033,897.88 monthly and 1,051,272.52 half-yearly.
    """
    line = {'InterestRate.USD': {
        'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
        'Curve': utils.Curve([], [[0.0, 0.01], [1.0, 0.03], [2.0, 0.05]])
        }}
    for grid in ('0d 1w(1w)', '0d 1m(1m)', '0d 6m(6m)'):
        calc, profiles = simulate(basket([trial_credit.DEALS[2]], line), (), 'USD', grid, sigma=0.0)
        deal = next(d for d in calc.netting_sets.deals() if d.Instrument.field['Reference'] == 'CASH')
        years = calc.time_grid.mtm_time_grid[deal.Time_dep.deal_time_grid] / 365.0
        accrued = 1e6 * np.exp((0.01 + 0.02 * years) * years)[:, None]
        assert years[-1] == 1.0, (grid, years)
        assert np.allclose(profiles['CASH'], accrued, rtol=1e-10, atol=0.0), (grid, profiles['CASH'][-1, 0])


def test_a_hazard_simulated_at_zero_reversion_reads_its_limit():
    """ISSUER_A's CVA on a hazard simulated at a stated Alpha of 0 is the one at 1e-8 to float32;
    killing mutation: the loadings read as (1 - exp(-a t)) / a, 0/0 there and the CVA NaN."""
    cashflow = {'Object': 'FixedCashflowDeal', 'Reference': 'CF', 'Currency': 'USD',
                'Discount_Rate': 'USD', 'Amount': 1e6, 'Payment_Date': B + pd.DateOffset(years=1)}
    cva = {}
    for alpha in (0.0, 1e-8):
        job = basket([cashflow])
        calc = job['Calc']
        calc['Calculation'] = {
            'Object': 'CreditMonteCarlo', 'Base_Date': calc['Calculation']['Base_Date'], 'Currency': 'USD',
            'Time_Grid': '0d 3m(3m)', 'Batch_Size': 64, 'Random_Seed': 1, 'Deflation_Interest_Rate': 'USD',
            'Credit_Valuation_Adjustment': {'Calculate': 'Yes', 'Counterparty': 'ISSUER_A',
                                            'Deflate_Stochastically': 'No', 'Stochastic_Hazard_Rates': 'Yes'}}
        calc['Deals']['Deals']['Children'] = [{'Instrument': {'.Deal': {
            'Object': 'NettingCollateralSet', 'Reference': 'NS', 'Netted': 'True',
            'Collateralized': 'False'}}, 'Children': calc['Deals']['Deals']['Children']}]
        market = calc['MergeMarketData']['ExplicitMarketData']
        market['Model Configuration'] = {'.ModelParams': {
            'modeldefaults': {'SurvivalProb': 'HWHazardRateModel'}, 'modelfilters': {}}}
        market['Price Models'] = {'HWHazardRateModel.ISSUER_A': {'Alpha': alpha, 'Lambda': 0.0, 'Sigma': 0.01}}
        context = derivus.Context()
        context.load_json((json.dumps(job), 'hazard'))
        cva[alpha] = float(context.run_job()[1]['Results']['cva'])
    assert cva[0.0] == pytest.approx(cva[1e-8], rel=1e-6) and cva[1e-8] > 0.0, cva


@pytest.mark.parametrize('block, switch, curve', [
    ('Funding_Valuation_Adjustment', 'Deflate_Stochastically', 'USD'),
    ('Credit_Valuation_Adjustment', 'Deflate_Stochastically', 'USD'),
    ('Credit_Valuation_Adjustment', 'Stochastic_Hazard_Rates', 'ISSUER_A')])
def test_a_stochastic_reading_of_a_curve_no_model_moves_is_refused_by_name(block, switch, curve):
    """Only EUR's FX rate carries a model, so the USD deflation curve and ISSUER_A's survival are
    static, and a switch reading one along the path is refused naming the switch and the curve.

    KILLING MUTATION: the `needs_model` call deleted from that switch's branch - a tensor broadcast
    error for either deflation, a clean run for the hazard.
    """
    cashflow = {'Object': 'FixedCashflowDeal', 'Reference': 'CF', 'Currency': 'EUR',
                'Discount_Rate': 'EUR', 'Amount': 1e6, 'Payment_Date': B + pd.DateOffset(years=1)}
    stated = {'Credit_Valuation_Adjustment': {
                  'Calculate': 'Yes', 'Counterparty': 'ISSUER_A', 'Deflate_Stochastically': 'No',
                  'Stochastic_Hazard_Rates': 'No'},
              'Funding_Valuation_Adjustment': {
                  'Calculate': 'Yes', 'Risk_Free_Curve': 'USD', 'Funding_Cost_Interest_Curve': 'USD',
                  'Funding_Benefit_Interest_Curve': 'USD'}}[block]
    with pytest.raises(ValueError, match='{}.{} is Yes but {} has no model'.format(block, switch, curve)):
        book.simulated([cashflow], (), trial_credit.FACTORS,
                       models={'FxRate.EUR': ('GBMAssetPriceModel', {'Vol': 0.1, 'Drift': 0.0})},
                       **{block: dict(stated, **{switch: 'Yes'})})


#: A rand cashflow on a USD-based book: every curve flat, the rand held at its spot by a GBM of no
#: volatility stating no spot lag, the counterparty's hazard flat at 2%; beside the dollar curve a
#: dollar funding curve, and beside the rand's own a rand swap curve and a rand funding curve.
X0, R_USD, R_ZAR, R_FUND, HAZARD = 0.055, 0.03, 0.07, 0.05, 0.02
R_SWAP, R_ZAR_FUND = 0.08, 0.10


def deflated(report, sigma=0.0, amount=1e6, dollar=None, judged=False, **calculation):
    """`Results` of a float64 credit Monte Carlo, `calculation` over its block, of R`amount` paid in
    a year, in one uncollateralised set against CPTY, reported in `report`, its CVA and its FVA
    funding at FUND over the dollar curve both on, the rand a GBM of volatility `sigma` and the
    dollar curve a Hull-White of volatility `dollar` where one is given; or, `judged`, what
    `validate` says of its calculation."""
    def flat(currency, rate):
        return {'Currency': currency, 'Day_Count': 'ACT_365', 'Sub_Type': None,
                'Curve': utils.Curve([], [[0.0, rate], [5.0, rate]])}

    hull_white = {} if dollar is None else {'HullWhite1FactorInterestRateModel.USD': {
        'Alpha': 0.05, 'Lambda': 0.0, 'Quanto_FX_Correlation': 0.0,
        'Quanto_FX_Volatility': utils.Curve([], [[0.0, 0.0], [10.0, 0.0]]),
        'Sigma': utils.Curve([], [[0.0, dollar], [10.0, dollar]])}}
    adjustment = {'Calculate': 'Yes', 'Counterparty': 'CPTY', 'Deflate_Stochastically': 'No'}
    cashflow = {'Object': 'FixedCashflowDeal', 'Reference': 'CF', 'Currency': 'ZAR',
                'Discount_Rate': 'ZAR', 'Amount': amount, 'Payment_Date': B + pd.DateOffset(years=1)}
    job = {'Calc': {
        'Calculation': dict({
            'Object': 'CreditMonteCarlo', 'Base_Date': B, 'Currency': report,
            'Time_Grid': '0d 3m(3m)', 'Batch_Size': 64, 'Random_Seed': 1,
            'Credit_Valuation_Adjustment': dict(adjustment, Stochastic_Hazard_Rates='No'),
            'Funding_Valuation_Adjustment': dict(
                adjustment, Risk_Free_Curve='USD', Funding_Cost_Interest_Curve='FUND',
                Funding_Benefit_Interest_Curve='FUND')}, **calculation),
        'Deals': {'Reference': 'deflated', 'Deals': {'Children': [{'Instrument': {'.Deal': {
            'Object': 'NettingCollateralSet', 'Reference': 'NS', 'Netted': 'True',
            'Collateralized': 'False', 'Credit_Support_Amounts': {'Counterparty': 'CPTY'}}},
            'Children': [{'Instrument': {'.Deal': cashflow}}]}]}},
        'MergeMarketData': {'MarketDataFile': '', 'ExplicitMarketData': {
            'System Parameters': {'Base_Currency': 'USD', 'Base_Date': B},
            'Price Factors': {
                'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Spot': 1.0},
                'FxRate.ZAR': {'Domestic_Currency': None, 'Interest_Rate': 'ZAR', 'Spot': X0,
                               'Settlement_Days': 0},
                'InterestRate.USD': flat('USD', R_USD), 'InterestRate.ZAR': flat('ZAR', R_ZAR),
                'InterestRate.FUND': flat('USD', R_FUND),
                'InterestRate.ZAR-SWAP': flat('ZAR', R_SWAP),
                'InterestRate.ZAR-FUND': flat('ZAR', R_ZAR_FUND),
                'SurvivalProb.CPTY': {'Recovery_Rate': 0.4, 'Curve': utils.Curve(
                    [], [[0.0, 0.0], [10.0, 10 * HAZARD]])}},
            'Price Models': dict({'GBMAssetPriceModel.ZAR': {'Vol': sigma, 'Drift': 0.0}}, **hull_white),
            'Model Configuration': {'.ModelParams': {
                'modeldefaults': {'FxRate': 'GBMAssetPriceModel'}, 'modelfilters': {'InterestRate': [
                    [['ID', 'USD'], 'HullWhite1FactorInterestRateModel']]} if hull_white else {}}}}}}}
    context = derivus.Context()
    context.load_json((json.dumps(job, cls=CustomJsonEncoder), 'deflated'))
    if judged:
        return context.validate()['calculation']
    return derivus.run_cmc(context.current_cfg, prec=torch.float64)[1]['Results']


def received(cost=R_FUND, reference=R_USD):
    """`(CVA, FVA)` of `deflated`'s cashflow received, by hand, its FVA funding at the flat rate
    `cost` over `reference`. On the report rows t_i, days 0, 92, 184, 273, 365 and 366 over 365,
    its value is V_i = 1e6 exp(-7% (1 - t_i)), nil on the last row; the rand is X0 on every row, so
    converted to dollars, deflated on the dollar curve and converted back at today's rate the
    exposure is P_i = V_i exp(-3% t_i); S_i = exp(-2% t_i), and d_i = t_i - t_(i-1).

        CVA = 0.6 sum_i (P_(i-1) + P_i) / 2 (S_(i-1) - S_i)
        FVA = sum_i (exp(cost d_i) - exp(reference d_i)) (P_(i-1) S_(i-1) + P_i S_i) / 2
    """
    rows = [(B + pd.DateOffset(months=3 * k) - B).days / 365.0 for k in range(5)] + [366 / 365.0]
    value = [1e6 * math.exp(-R_ZAR * (1.0 - t)) if t <= 1.0 else 0.0 for t in rows]
    exposure = [v * math.exp(-R_USD * t) for v, t in zip(value, rows)]
    survival = [math.exp(-HAZARD * t) for t in rows]
    cva = 0.6 * sum((exposure[i - 1] + exposure[i]) / 2 * (survival[i - 1] - survival[i])
                    for i in range(1, 6))
    fva = sum((math.exp(cost * (t - s)) - math.exp(reference * (t - s)))
              * (exposure[i - 1] * survival[i - 1] + exposure[i] * survival[i]) / 2
              for i, (s, t) in enumerate(zip(rows, rows[1:]), 1))
    return cva, fva


def test_a_book_reported_in_rand_deflates_its_exposure_on_the_dollar_curve():
    """The rand cashflow reported in ZAR on its USD book, by hand (`received`): a CVA of 11,317.14
    and an FVA of 19,051.06 funding at 5% over the dollar curve, the FVA's cube carrying each row's
    survival. Both to 1e-10. Killing mutation: the deflation read off the report currency's own
    curve, the rand at 7%, under which the CVA reads 11,092.60.
    """
    cva, fva = received()
    results = deflated('ZAR')
    assert float(results['cva']) == pytest.approx(cva, rel=1e-10)
    assert float(results['fva']) == pytest.approx(fva, rel=1e-10)


def test_a_book_funded_in_rand_takes_its_spreads_over_the_risk_free_curve_it_states():
    """The rand cashflow received on its USD book reported in ZAR, funding at ZAR-FUND, 10%, over
    the risk-free curve it states, ZAR-SWAP at 8%, a rand curve no deal reads: by hand (`received`)
    an FVA of 19,290.42 to 1e-10, the spread over the stated curve and the exposure deflated on
    the dollar curve.

    Killing mutations: the spread's reference read off the base currency's own curve again, under
    which the FVA reads 67,097.07, 10% over the dollar curve's 3%; the factor walk not adding the
    risk-free curve, under which the run finds no ZAR-SWAP.
    """
    results = deflated('ZAR', Funding_Valuation_Adjustment={
        'Calculate': 'Yes', 'Counterparty': 'CPTY', 'Risk_Free_Curve': 'ZAR-SWAP',
        'Funding_Cost_Interest_Curve': 'ZAR-FUND', 'Funding_Benefit_Interest_Curve': 'ZAR-FUND'})
    assert float(results['fva']) == pytest.approx(received(R_ZAR_FUND, R_SWAP)[1], rel=1e-10)


@pytest.mark.parametrize('benefit', ['ZAR-SWAP', ''])
@pytest.mark.parametrize('stochastically', ['No', 'Yes'])
def test_a_benefit_funded_at_the_risk_free_curve_earns_nothing(stochastically, benefit):
    """The rand cashflow paid, its benefit funded at the risk-free curve it states, ZAR-SWAP, named
    or left blank, the dollar curve a Hull-White: every interval's benefit spread is nothing, so
    the FVA is exactly 0.0, deflated on today's dollar curve or along the simulated one.

    Killing mutations: the spread's reference read off the base currency's own curve again, under
    which each reads the benefit over the dollar curve; a blank benefit curve read as the base
    currency's own, which the blank arms read.
    """
    results = deflated('ZAR', amount=-1e6, dollar=0.01, Funding_Valuation_Adjustment={
        'Calculate': 'Yes', 'Counterparty': 'CPTY', 'Deflate_Stochastically': stochastically,
        'Risk_Free_Curve': 'ZAR-SWAP', 'Funding_Cost_Interest_Curve': 'ZAR-FUND',
        'Funding_Benefit_Interest_Curve': benefit})
    assert float(results['fva']) == 0.0


def test_a_funding_curve_stated_beside_no_risk_free_curve_is_refused_by_name():
    """An FVA block funding at ZAR-FUND and stating no risk-free curve takes a spread over nothing:
    `validate` names it and the run refuses with the same sentence. A block stating no funding
    curve, or not calculating, says nothing. Killing mutation: the judgement dropped from
    `CreditMonteCarlo.refusals`, under which `validate` says nothing and the run prices.
    """
    said = 'Funding_Valuation_Adjustment states Funding_Cost_Interest_Curve, so Risk_Free_Curve is required'
    funding = {'Calculate': 'Yes', 'Counterparty': 'CPTY', 'Funding_Cost_Interest_Curve': 'ZAR-FUND'}
    assert deflated('ZAR', judged=True, Funding_Valuation_Adjustment=funding) == [said]
    with pytest.raises(ValueError) as refused:
        deflated('ZAR', Funding_Valuation_Adjustment=funding)
    assert str(refused.value) == 'CreditMonteCarlo: ' + said
    for silent in (dict(funding, Funding_Cost_Interest_Curve=''), dict(funding, Calculate='No')):
        assert deflated('ZAR', judged=True, Funding_Valuation_Adjustment=silent) == []


def test_an_adjustment_reported_in_rand_is_the_dollar_one_at_today_s_rate():
    """The rand at 15% volatility, the book reported in ZAR and in USD, its base currency: either
    way a path's exposure is deflated in dollars, so the rand CVA and FVA times today's rate are
    the dollar ones. Held at 1e-15 relative: a path's dollar exposure differs from the rand one
    times today's rate by the rounding of one product and one quotient, which the mean and the sum
    over rows do not grow; measured 2.2e-16 on each. Killing mutation: the deflation read off the
    report currency's own curve, the rand run deflated at 7% against the dollar run's 3%.
    """
    rand, dollar = deflated('ZAR', sigma=0.15), deflated('USD', sigma=0.15)
    for key in ('cva', 'fva'):
        assert float(rand[key]) * X0 == pytest.approx(float(dollar[key]), rel=1e-15, abs=0.0), key
