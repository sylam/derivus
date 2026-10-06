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
from derivus import riskfactors, spine, utils
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


def cds_world(hazard=0.02, rate=0.03, curve=None):
    """A flat hazard (or `curve`, a negative log survival) and a flat zero curve as the engine's own
    factor objects, with the descriptors `utils.calc_cds_rates` reads them through."""
    factors = {'S': riskfactors.SurvivalProb({'Recovery_Rate': 0.4, 'Curve': curve or utils.Curve(
        [], [[0.0, 0.0], [10.0, 10.0 * hazard]])}), 'D': riskfactors.InterestRate({
            'Currency': 'USD', 'Day_Count': 'ACT_365', 'Curve': utils.Curve(
                [], [[0.0, rate], [10.0, rate]])})}
    days = lambda d: d / 365.0
    return [None, 'S', None, None, days], [None, 'D', None, None, days], factors


def test_a_par_cds_rate_holds_across_the_quarter_and_a_tenor_bump_holds_the_other_tenors():
    """The ISDA standard model on a flat 2% hazard: the 5y par rate is (1 - R)h to the coupon
    effect and the same on every day of the quarter, the accrued since the last standard date
    SUBTRACTED and the first coupon the next standard date; and the curve one tenor's bump implies
    moves that tenor's par rate by the bump and no other's, the later segments re-solved.
    MUTATIONS: the accrued added (`v_fee = -tau[0]`) reads the rate 1.08%-1.30% across the
    quarter; the first date a quarter late on the 20th of a non-roll month (20 April to
    20 September) is caught by name; the shift dropped after its segment leaks 1-3.5% of the
    bump into the later tenors against 1e-9 here."""
    rates = {}
    for day in ('2026-03-20', '2026-04-20', '2026-05-20', '2026-06-19'):
        survival, discount, factors = cds_world()
        rates[day] = utils.calc_cds_rates(
            0.4, survival, discount, pd.Timestamp(day), [5.0], factors, bump=0)[5.0]
    assert max(rates.values()) - min(rates.values()) < 2e-6, rates
    assert abs(rates['2026-03-20'] - 0.6 * 0.02) < 1e-4, rates
    assert utils.cds_dates(pd.Timestamp('2026-04-20'), 3)[0] == pd.Timestamp('2026-06-20')
    assert utils.cds_dates(pd.Timestamp('2026-06-20'), 3)[0] == pd.Timestamp('2026-09-20')

    survival, discount, factors = cds_world()
    base, tenors = pd.Timestamp('2026-06-10'), [1.0, 3.0, 5.0]
    par, knots, shifted = utils.calc_cds_rates(0.4, survival, discount, base, tenors, factors)
    for bumped, curve in zip(tenors, shifted[1:]):
        survival, discount, factors = cds_world(
            curve=utils.Curve([], np.column_stack([knots, curve]).tolist()))
        moved = utils.calc_cds_rates(0.4, survival, discount, base, tenors, factors, bump=0)
        for tenor in tenors:
            assert abs(moved[tenor] - par[tenor] - 1e-4 * (tenor == bumped)) < 1e-9 * 1e-4, (
                bumped, tenor, moved[tenor] - par[tenor])


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
    for row in spine.diary(context, {deal['Reference']: deal['Reference'] for deal in deals}):
        if row['kind'] == 'payment':
            signs.setdefault(row['instrument'], set()).add(math.copysign(1.0, row['amount']))
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


def simulate(job, models, rate_model=None):
    """`job` as a float64 credit Monte Carlo over one netting set, a hazard model per name in
    `models` and a Hull-White on `rate_model`'s curve: `(calculation, {reference: profile})`."""
    calc = job['Calc']
    calc['Calculation'] = dict(
        Object='CreditMonteCarlo', Base_Date=calc['Calculation']['Base_Date'], Currency='USD',
        Time_Grid='0d 3m(3m)', Batch_Size=256, Random_Seed=1, Deflation_Interest_Rate='USD',
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
            'Sigma': {'.Curve': {'meta': [], 'data': [[0.0, 0.01], [10.0, 0.01]]}}}
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

    KILLING MUTATION: the recurrence writing its slices in place, which refuses the backward pass.
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
    block like the index's, so the trial basket prices a finite, dispersed profile where it met a
    three-row block with all 25 rows and was skipped; on static curves its mark is unmoved at the
    number the trial reads, to the bit.

    KILLING MUTATION: the names read over every row rather than their block's.
    """
    assert book.marks([trial_credit.DEALS[1]], trial_credit.FACTORS)[0]['NTD'] == \
        '-0x1.28cf4f83a85a8p+18'
    _, profiles = simulate(basket(), ('ISSUER_A', 'ISSUER_B', 'ISSUER_C', 'ISSUER_INDEX'))
    assert 'NTD' in profiles, 'the basket was skipped'
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
