"""A price swap through the JSON contract: its children's value less a fixed strike, settled once.

A `PriceSwapDeal` holds the deals it swaps as its `Children`. To its `Maturity_Date` it marks their
value less `Strike_Amount`, the strike a fixed amount of its `Currency` and not discounted; the
children settle nothing beneath it, their coupons belonging to another deal; on the maturity it
settles once, in its `Currency`, their value less what they would pay that day, less the strike;
after it, nothing.

The world is the declared-defaults book's: USD discounting flat at 4% and USD-PROJ at 4.5%, both
ACT/365, a share EQ at 100 and EUR at 1.25 dollars on a flat 2% curve; under the credit Monte Carlo
each curve a deal reads runs a Hull-White of 1%, float64, 256 paths, seed 1, a quarterly grid. The
bond is 100 for three years at 3.3% paid on stated half years, so every coupon is 1.65 and the
principal rides the last: it marks 97.917. The floating list is 100 quarterly from a month out
against USD-PROJ, the principal with its last coupon. The strike is 98 and the swap matures a year
out, on a coupon date, or nine months out, between two and on a grid date.

Under the Monte Carlo the swap and its children booked plain are priced in two runs of one book
holding both, each with the other at zero units, so both price on one grid and one set of paths
and each run's cash is its own deals'.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import pytest
import torch

import derivus
import test_declared_defaults as book
import test_position_scaling as trials
import trial_rates
from derivus import spine, utils
from derivus.config import CustomJsonEncoder

B = book.WORLD_BASE
N, STRIKE = 100.0, 98.0
RATE = utils.Percent(3.3)
COUPON = RATE.amount * 0.5 * N


def months(count):
    return B + pd.DateOffset(months=count)


def D(day, rate=0.04):
    """The discount factor to `day` off a flat ACT/365 curve at `rate`."""
    return float(np.exp(-rate * (day - B).days / 365.0))


#: three years of 3.3% on 100, paid semi-annually on stated half years, the principal with the last
BOND = trial_rates.fixed_list('BOND', 'Buy', [
    {'Payment_Date': months(6 * k), 'Notional': N, 'Rate': RATE, 'Accrual_Start_Date': months(6 * k - 6),
     'Accrual_End_Date': months(6 * k), 'Accrual_Day_Count': 'ACT_365', 'Accrual_Year_Fraction': 0.5,
     'Fixed_Amount': N if k == 6 else 0.0, 'Discounted': 'No', 'FX_Reset_Date': None, 'Known_FX_Rate': 0.0}
    for k in range(1, 7)])

#: today's value of what the bond pays from a year out, the coupon of that day with it: 96.3001
LATER = sum((row['Fixed_Amount'] + COUPON) * D(row['Payment_Date']) for row in BOND['Cashflows']['Items'][1:])


def floating():
    """Two years of quarterly periods on 100 from a month out, each resetting at its start."""
    starts = [months(1 + 3 * k) for k in range(9)]
    rows = []
    for start, end in zip(starts[:-1], starts[1:]):
        accrual = utils.DayCount.accrual(start, (end - start).days, utils.DayCount.code('ACT_365'))
        rows.append({'Payment_Date': end, 'Notional': N, 'Accrual_Start_Date': start, 'Accrual_End_Date': end,
                     'Accrual_Day_Count': 'ACT_365', 'Accrual_Year_Fraction': accrual,
                     'Resets': [[start, start, end, accrual, pd.DateOffset(days=1), 'ACT_365', '0D', 0.0, 'No',
                                 utils.Percent(0.0)]],
                     'Margin': utils.Basis(0.0), 'Fixed_Amount': N if end == starts[-1] else 0.0,
                     'FX_Reset_Date': None, 'Known_FX_Rate': 0.0})
    return rows


FLOAT = trial_rates.float_list('FLOAT', 'Buy', floating())
#: one share held to a horizon well after every maturity here, which its type requires stated
SHARES = {'Object': 'EquityDeal', 'Reference': 'SHARES', 'Currency': 'USD', 'Equity': 'EQ',
          'Buy_Sell': 'Buy', 'Units': 1.0, 'Investment_Horizon': months(30)}


def swap(children, reference='SWAP', maturity=months(12), **terms):
    """A price swap over `children`, each renamed under the swap, struck at 98."""
    return dict({'Object': 'PriceSwapDeal', 'Reference': reference, 'Currency': 'USD', 'Buy_Sell': 'Buy',
                 'Strike_Amount': STRIKE, 'Maturity_Date': maturity, 'Children': [
                     dict(child, Reference='{}_{}'.format(reference, child['Reference'])) for child in children]},
                **terms)


def near(paths, level):
    """Whether the average over `paths` is `level` within four of its standard errors."""
    return abs(paths.mean() - level) < 4.0 * paths.std() / np.sqrt(paths.size)


def marks(deals):
    """`{reference: mark}` off one base valuation, nothing skipped."""
    hexes, stats = book.marks(deals)
    assert 'Deals Skipped' not in stats and 'Structs Skipped' not in stats, stats
    return {reference: float.fromhex(value) for reference, value in hexes.items()}


def simulated(deals, priced, curves=('USD',), currency='USD', factors=None):
    """`(report dates, {reference: rows by paths}, {day: cash in currency by paths})` off one float64
    credit Monte Carlo of `deals`, each one whose reference is not among `priced` held at zero units."""
    book_deals = [deal if deal['Reference'] in priced else trials.held(deal, 0.0) for deal in deals]
    _, out = book.simulated(book_deals, list(curves), factors=factors, prec=torch.float64, Generate_Cashflows='Yes')
    assert 'Deals Skipped' not in out['Stats'] and 'Structs Skipped' not in out['Stats'], out['Stats']
    nodes = [structure.obj for structure in out['Netting'].structures()] + list(out['Netting'].deals())
    rows = {node.Instrument.field['Reference']: np.concatenate(node.Calc_res['Value'], axis=-1)
            for node in nodes if (node.Calc_res or {}).get('Value')}
    cash = out['Results']['cashflows'][currency]
    return list(out['Results']['mtm'].index), rows, {day: cash.loc[day].values for day in cash.index}


def test_a_price_swap_marks_its_children_less_the_strike():
    """A base valuation of the swap over the bond, worth 97.917, struck at 98 for a year: it marks the
    bond less the strike, -0.0828, to the bit; sold, the negative of that; over the bond and the
    floating list, their sum less the strike.

    Killing mutation: the `Buy_Sell` sign dropped, the sold swap marking -0.0828 as the bought does.
    """
    values = marks([BOND, FLOAT, swap([BOND]), swap([BOND], 'SOLD', Buy_Sell='Sell'), swap([BOND, FLOAT], 'BOTH')])
    assert values['SWAP'] == values['BOND'] - STRIKE and round(values['SWAP'], 4) == -0.0828, values
    assert values['SOLD'] == -(values['BOND'] - STRIKE), values
    assert values['BOTH'] == values['BOND'] + values['FLOAT'] - STRIKE, values


#: `(children, the curves simulated, {where the maturity falls: the maturity})`
CASES = {'bond': ([BOND], ('USD',), {'between': months(9), 'payment': months(12)}),
         'floating and shares': ([FLOAT, SHARES], ('USD', 'USD-PROJ'), {'between': months(9)})}


@pytest.mark.parametrize('kind,on', [(kind, on) for kind in sorted(CASES) for on in CASES[kind][2]])
def test_under_simulation_a_price_swap_is_its_children_less_the_strike_and_settles_once(kind, on):
    """The credit Monte Carlo of a swap and of its children booked plain, on one grid and one set of
    paths: every row of the swap to its maturity is the children's less the strike, to the bit, and
    none is reported after it; it books nothing on any day before the maturity, where the children
    book payments; on the maturity it books their row less their own cash that day less the
    strike, to the bit, and nothing after. A year out, on a coupon date, the bond's swap books on
    average the forward value of what the bond pays after that day less the strike, 0.58, within the
    Monte Carlo error. The second swap holds a floating list and a share held past its maturity, on
    a row of the share's own, where it is paid a quarter's dividends.

    Killing mutation: the children's cash booked into the run's, the swap paying the bond's coupon
    before its maturity.
    """
    children, curves, maturities = CASES[kind]
    maturity = maturities[on]
    deals = [swap(children, maturity=maturity)] + children
    dates, swapped, cash_swap = simulated(deals, {'SWAP'}, curves)
    _, plain, cash_plain = simulated(deals, {child['Reference'] for child in children}, curves)
    rows = sum(day <= maturity for day in dates)
    held = sum(plain[child['Reference']][:rows] for child in children)
    assert dates[rows - 1] == maturity and swapped['SWAP'].shape[0] == rows, (rows, swapped['SWAP'].shape)
    assert np.array_equal(swapped['SWAP'], held - STRIKE)

    before = [day for day in cash_plain if day < maturity and cash_plain[day].any()]
    assert before and not any(cash_swap[day].any() for day in cash_swap if day < maturity), before
    assert np.array_equal(cash_swap[maturity], held[rows - 1] - cash_plain[maturity] - STRIKE)
    assert cash_plain[maturity].any() == (kind != 'bond' or on == 'payment')
    assert not any(cash_swap[day].any() for day in cash_swap if day > maturity)
    if kind == 'bond' and on == 'payment':
        assert np.all(cash_plain[maturity] == COUPON)
        assert near(cash_swap[maturity], LATER / D(maturity) - COUPON - STRIKE), cash_swap[maturity].mean()


def test_a_swap_over_a_share_settles_its_price_less_the_strike_and_none_of_its_dividends():
    """A share at 100 on every path, held past a swap maturing seven months out, 28 days after a row
    of the share's: beneath the swap the share is worth on the maturity its price and the dividends
    of those days, 0.0769 by hand at a 1% yield carried at 4%, to 2e-4 on every path of the
    simulated curve, and the swap books its price less the strike, 2, to 1e-12, and nothing on
    any other day.

    Killing mutation: the swap settling its children's row with their cash of that day left in it,
    the swap then paying the share's dividends.
    """
    maturity = months(7)
    _, swapped, cash = simulated([swap([SHARES], maturity=maturity), SHARES], {'SWAP'})
    tau = (maturity - months(6)).days / 365.0
    dividends = 100.0 * np.exp(0.04 * tau) * -np.expm1(-0.01 * tau)
    assert round(dividends, 4) == 0.0769
    assert np.abs(swapped['SWAP'][-1] - (100.0 + dividends - STRIKE)).max() < 2e-4, swapped['SWAP'][-1][:3]
    assert np.allclose(cash[maturity], 100.0 - STRIKE, rtol=1e-12, atol=0.0), cash[maturity][:3]
    assert not any(cash[day].any() for day in cash if day != maturity)


def test_a_swap_in_a_currency_other_than_the_report_s_marks_and_settles_in_it():
    """The bond in euros, discounted on the EUR curve under its Hull-White, swapped against 98 euros
    a year out on a dollar report, EUR quoted at 1.25 for delivery that day so a euro is 1.25
    dollars on every row: every row to the maturity is the bond's less 122.5 dollars, to the bit,
    and on the maturity it books in euros the bond's row there in euros less its coupon that day
    less 98, to 1e-12.

    Killing mutation: the settlement left in the reporting currency, the dollar row less the coupon
    in dollars less 98.
    """
    bond, maturity = dict(BOND, Currency='EUR', Discount_Rate='EUR'), months(12)
    deals, spot = [swap([bond], Currency='EUR'), bond], {'FxRate.EUR': dict(book.world()['FxRate.EUR'], Settlement_Days=0)}
    dates, swapped, cash_swap = simulated(deals, {'SWAP'}, ('EUR',), 'EUR', spot)
    _, plain, cash_plain = simulated(deals, {'BOND'}, ('EUR',), 'EUR', spot)
    rows = sum(day <= maturity for day in dates)
    assert np.array_equal(swapped['SWAP'], plain['BOND'][:rows] - STRIKE * book.X0)
    assert np.all(cash_plain[maturity] == COUPON)
    left = plain['BOND'][rows - 1] / book.X0 - cash_plain[maturity] - STRIKE
    assert np.abs(cash_swap[maturity] - left).max() < 1e-12 and abs(left.mean()) < 10.0, left.mean()


def test_a_swap_leaves_out_what_a_child_pays_on_its_maturity_in_another_currency():
    """The bond in euros swapped against 98 dollars a year out on a dollar report, a euro 1.25
    dollars on every row: on the maturity the swap books in dollars the bond's row there less the
    euro coupon the bond pays that day, at 1.25, less 98, to 1e-12.

    Killing mutation: the maturity not read as a cash day of each currency the children pay in,
    the euro coupon then left in the settlement.
    """
    bond, maturity = dict(BOND, Currency='EUR', Discount_Rate='EUR'), months(12)
    deals, spot = [swap([bond]), bond], {'FxRate.EUR': dict(book.world()['FxRate.EUR'], Settlement_Days=0)}
    dates, _, cash_swap = simulated(deals, {'SWAP'}, ('EUR',), 'USD', spot)
    _, plain, cash_plain = simulated(deals, {'BOND'}, ('EUR',), 'EUR', spot)
    rows = sum(day <= maturity for day in dates)
    left = plain['BOND'][rows - 1] - cash_plain[maturity] * book.X0 - STRIKE
    assert np.all(cash_plain[maturity] == COUPON) and np.abs(cash_swap[maturity] - left).max() < 1e-12


def test_a_swap_framed_by_no_netting_set_takes_no_later_payment_of_a_child_as_paid_on_its_maturity():
    """A swap at the root of its book, struck at nothing nine months out over an FX forward buying
    100 euros for 120 dollars eighteen months out, under the credit Monte Carlo: its dates run a day
    past the maturity, as a netting set's would, so the forward's settlement, booked on the last row
    of a grid ending before it, is not read as paid on the maturity, and the swap books on it the
    forward's row there, to the bit: on average the 125 dollars of the euros on EUR's 2% less the 120
    on USD's 4% over the nine months between, 6.70, within the Monte Carlo error.

    Killing mutation: the day after the maturity left off the swap's dates, the swap then booking
    the forward's row less the five dollars its later settlement nets to, 1.70.
    """
    maturity = months(9)
    forward = {'Object': 'FXForwardDeal', 'Reference': 'FWD', 'Buy_Currency': 'EUR', 'Sell_Currency': 'USD',
               'Buy_Amount': 100.0, 'Sell_Amount': 120.0, 'Settlement_Date': months(18),
               'Buy_Discount_Rate': 'EUR', 'Sell_Discount_Rate': 'USD'}
    job = json.loads(json.dumps(book.book([swap([forward], maturity=maturity, Strike_Amount=0.0)]),
                                cls=CustomJsonEncoder))
    job['Calc']['Calculation'] = {
        'Object': 'CreditMonteCarlo', 'Base_Date': job['Calc']['Calculation']['Base_Date'], 'Currency': 'USD',
        'Time_Grid': '0d 3m(3m)', 'Batch_Size': 256, 'Random_Seed': 1, 'Deflation_Interest_Rate': 'USD',
        'DealLevel': True, 'Generate_Cashflows': 'Yes'}
    market = job['Calc']['MergeMarketData']['ExplicitMarketData']
    market['Model Configuration'] = {'.ModelParams': {
        'modeldefaults': {'InterestRate': 'HullWhite1FactorInterestRateModel'}, 'modelfilters': {}}}
    market['Price Models'] = {'HullWhite1FactorInterestRateModel.USD': {
        'Alpha': 0.05, 'Lambda': 0.0, 'Quanto_FX_Correlation': 0.0,
        'Quanto_FX_Volatility': {'.Curve': {'meta': [], 'data': [[0.0, 0.0], [10.0, 0.0]]}},
        'Sigma': {'.Curve': {'meta': [], 'data': [[0.0, 0.01], [10.0, 0.01]]}}}}
    context = derivus.Context()
    context.load_json((json.dumps(job), 'unframed'))
    _, out = derivus.run_cmc(context.current_cfg, prec=torch.float64)
    row = np.concatenate(next(out['Netting'].structures()).obj.Calc_res['Value'], axis=-1)[-1]
    cash = out['Results']['cashflows']['USD'].loc[maturity].values
    tau = (months(18) - maturity).days / 365.0
    assert np.array_equal(cash, row) and near(cash, 100.0 * book.X0 * np.exp(-0.02 * tau) - 120.0 * np.exp(-0.04 * tau))


def test_a_child_s_payment_days_are_no_cash_days_of_a_run_holding_the_swap_alone():
    """A netting set holding one price swap over the bond, a year out, under the credit Monte Carlo:
    the run's ledger holds the maturity alone, the bond's coupon six months out no cash day of it,
    and the report ends the day after the maturity.

    Killing mutation: the children's settlement days left registered beneath the swap, the coupon's
    day then a cash day of the run.
    """
    _, out = book.simulated([swap([BOND])], ['USD'], prec=torch.float64, Generate_Cashflows='Yes')
    assert list(out['Results']['cashflows']['USD'].index) == [months(12)]
    assert out['Results']['mtm'].index[-1] == months(12) + pd.DateOffset(days=1)


def test_a_position_scales_the_strike_and_the_children_together():
    """Held at half a unit the swap marks half and held two short it marks minus twice, to the bit:
    the strike and the children's amounts sized at the position's magnitude, the swap's side flipped
    and its children's kept, they being what it is written on.

    Killing mutation: `PriceSwapDeal.option_on_children` dropped, the short swap's children flipping
    with it so that it marks the bond plus the strike.
    """
    priced = swap([BOND])
    half, short = trials.held(priced, 0.5), trials.held(priced, -2.0)
    assert (short['Buy_Sell'], short['Strike_Amount'], short['Children'][0]['Buy_Sell']) == ('Sell', 2 * STRIKE, 'Buy')
    values = marks([priced, dict(half, Reference='HALF'), dict(short, Reference='SHORT')])
    assert values['HALF'] == values['SWAP'] / 2 and values['SHORT'] == -2 * values['SWAP'], values


def test_a_structure_beneath_a_swap_settles_nothing_and_the_swap_settles_once():
    """A swap a year out, on a coupon date of the bond, over a structure of the bond and the floating
    list, beside the same structure booked plain, under the credit Monte Carlo: every row of the swap
    to its maturity is the structure's less the strike, to the bit; it books nothing before the
    maturity, where the structure's legs book a payment on each; on the maturity it books the
    structure's row less the legs' own cash that day less the strike, to the bit; and nothing after.

    Killing mutation: the structures beneath the swap resolved before their cash is held apart, the
    legs' payments then booked before the maturity, the first four months out.
    """
    pair, maturity = {'Object': 'StructuredDeal', 'Reference': 'PAIR', 'Currency': 'USD', 'Children': [BOND, FLOAT]}, months(12)
    deals = [swap([pair], maturity=maturity), pair]
    dates, swapped, cash_swap = simulated(deals, {'SWAP'}, ('USD', 'USD-PROJ'))
    _, plain, cash_plain = simulated(deals, {'PAIR'}, ('USD', 'USD-PROJ'))
    rows = sum(day <= maturity for day in dates)
    assert np.array_equal(swapped['SWAP'], plain['PAIR'][:rows] - STRIKE)
    before = [day for day in cash_plain if day < maturity]
    assert before and all(cash_plain[day].any() and not cash_swap[day].any() for day in before), before
    assert cash_plain[maturity].any() and np.array_equal(
        cash_swap[maturity], plain['PAIR'][rows - 1] - cash_plain[maturity] - STRIKE)
    assert not any(cash_swap[day].any() for day in cash_swap if day > maturity)


def test_the_diary_announces_one_settlement_on_the_maturity_and_of_the_children_their_fixings_to_it():
    """The diary of the swap over the bond and the floating list nine months out: the swap announces
    one payment, on its maturity in dollars at an amount the market decides, null on the
    `Settlement` leg, and its expiry there; the bond under it announces nothing; the floating list
    under it announces only its fixings to the maturity, the plain list's own, and the plain lists
    beside it their payments.

    Killing mutation: `PriceSwapDeal.settles_children` dropped, the diary then reading the swap as a
    frame, announcing its children's coupons and no settlement.
    """
    maturity = months(9)
    job = book.book([swap([BOND, FLOAT], maturity=maturity), BOND, FLOAT])
    context = derivus.Context()
    context.load_json((json.dumps(job, cls=CustomJsonEncoder), 'diary'))
    rows = {}
    for row in spine.diary(context):
        rows.setdefault(row['reference'], []).append(row)
    day = maturity.date().isoformat()
    assert [(row['kind'], row['leg'], row['due_date'], row['currency'], row['amount'], row['determined'])
            for row in rows['SWAP']] == [('expiry', 'Expiry', day, None, None, False),
                                         ('payment', 'Settlement', day, 'USD', None, False)], rows['SWAP']
    assert 'SWAP_BOND' not in rows and len([row for row in rows['BOND'] if row['kind'] == 'payment']) == 6

    def fixings(reference):
        return [(row['leg'], row['due_date'], row['index']) for row in rows[reference] if row['kind'] == 'fixing']
    assert all(row['kind'] == 'fixing' for row in rows['SWAP_FLOAT'])
    assert fixings('SWAP_FLOAT') == [fixing for fixing in fixings('FLOAT') if fixing[1] <= day]
    assert len(fixings('SWAP_FLOAT')) == 3 and len(fixings('FLOAT')) == 8
