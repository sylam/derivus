"""A cashflow list held to an `Investment_Horizon`, through the JSON contract.

A holding is paid what the list pays to its horizon as it falls and, on the horizon, is closed out
at its value then, the cashflows after it; before the horizon it marks as the list held to its last
payment, and after it it is worth nothing.

The world is the declared-defaults book's: USD discounting flat at 4% and USD-PROJ at 4.5%, both
ACT/365, each curve a list reads under a Hull-White of 1% in the credit Monte Carlo, float64, 256
paths, seed 1, a quarterly grid. The bond is 100 for three years at 3.3% paid on stated half
years, so every coupon is 1.65 and the principal rides the last: it marks 97.917. The floating list is 100 quarterly from a
month out against USD-PROJ, the principal with its last coupon, and the linker the trial module's
seasoned one on its own index. Each list is held to a horizon between two payments on a grid date,
to one on a payment date, and to its last payment and after it.

Under the Monte Carlo the holding and its list are priced in two runs of one book holding both,
each with the other at zero units, so both price on one grid and one set of paths and each run's
cash is its own deal's.
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
from derivus import schema, spine, utils
from derivus.config import CustomJsonEncoder
from derivus.instruments import construct_instrument

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
BOND = [{'Payment_Date': months(6 * k), 'Notional': N, 'Rate': RATE, 'Accrual_Start_Date': months(6 * k - 6),
         'Accrual_End_Date': months(6 * k), 'Accrual_Day_Count': 'ACT_365', 'Accrual_Year_Fraction': 0.5,
         'Fixed_Amount': N if k == 6 else 0.0, 'Discounted': 'No', 'FX_Reset_Date': None,
         'Known_FX_Rate': 0.0} for k in range(1, 7)]

#: today's value of what the bond pays from a year out, the coupon of that day with it: 96.3001
LATER = sum((row['Fixed_Amount'] + COUPON) * D(row['Payment_Date']) for row in BOND[1:])


def near(paths, level):
    """Whether the average over `paths` is `level` within four of its standard errors."""
    return abs(paths.mean() - level) < 4.0 * paths.std() / np.sqrt(paths.size)


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


LINKER = next(deal for deal in trial_rates.DEALS if deal['Reference'] == 'LINKER')

#: `(list, the world it prices in, the curves simulated, the calculation's extra terms, horizons)`:
#: between two payments on a grid date, on a payment date, on the last payment and after it
LISTS = {
    'fixed': (trial_rates.fixed_list('LIST', 'Buy', BOND), {}, ('USD',), {},
              (months(9), months(12), months(36), months(40))),
    'floating': (trial_rates.float_list('LIST', 'Buy', floating()), {}, ('USD', 'USD-PROJ'), {},
                 (months(9), months(7), months(25), months(30))),
    'inflation': (dict(LINKER, Reference='LIST'), trial_rates.FACTORS, ('USD',), {'NoModel': 'RiskNeutral'},
                  (months(15), months(12), months(24), months(30)))}


def held(deal, horizon, reference='HOLD'):
    return dict(deal, Reference=reference, Investment_Horizon=horizon)


def marks(deals, factors=None):
    """`{reference: mark}` off one base valuation, nothing skipped."""
    hexes, stats = book.marks(deals, factors=factors)
    assert 'Deals Skipped' not in stats, stats
    return {reference: float.fromhex(value) for reference, value in hexes.items()}


def simulated(deals, priced, factors=None, curves=('USD',), **calculation):
    """`(report dates, {reference: rows by paths}, {day: cash by paths})` off one float64 credit Monte
    Carlo of `deals`, each one but `priced` held at zero units."""
    book_deals = [deal if deal['Reference'] == priced else trials.held(deal, 0.0) for deal in deals]
    _, out = book.simulated(book_deals, list(curves), factors=factors, prec=torch.float64,
                            Generate_Cashflows='Yes', **calculation)
    assert 'Deals Skipped' not in out['Stats'], out['Stats']
    nodes = [structure.obj for structure in out['Netting'].structures()] + list(out['Netting'].deals())
    rows = {node.Instrument.field['Reference']: np.concatenate(node.Calc_res['Value'], axis=-1)
            for node in nodes if (node.Calc_res or {}).get('Value')}
    cash = out['Results']['cashflows']['USD']
    return list(out['Results']['mtm'].index), rows, {day: cash.loc[day].values for day in cash.index}


@pytest.mark.parametrize('kind', sorted(LISTS))
def test_a_holding_marks_as_the_list_until_its_horizon(kind):
    """A base valuation of each list beside the same list held to a horizon between two payments,
    to one on a payment date, to its last payment and after it: every holding marks as the list to
    the bit, the bond 97.917, the floating list 100.625 and the linker 39,974.69.

    Killing mutation: the cashflows after the horizon dropped from the value, the bond held nine
    months then marking 1.617 and held a year 3.202 for 97.917.
    """
    deal, factors, _, _, horizons = LISTS[kind]
    values = marks([deal] + [held(deal, day, 'HOLD{}'.format(k)) for k, day in enumerate(horizons)], factors)
    assert values['LIST'] and all(values['HOLD{}'.format(k)] == values['LIST'] for k in range(len(horizons))), values


@pytest.mark.parametrize('kind,on', [(kind, on) for kind in sorted(LISTS) for on in ('between', 'payment')])
def test_under_simulation_a_holding_is_the_list_to_its_horizon_and_is_closed_out_there(kind, on):
    """The credit Monte Carlo of a holding and of its list, on one grid and one set of paths: every
    row of the holding up to its horizon is the list's to the bit and none is reported after it;
    its cash on each day before the horizon is the list's to the bit, a coupon on every one; on the
    horizon it books the list's value there, to the bit since it books the very row it reports, so
    what it is paid beyond a coupon falling that day is the list's value excluding it; and it books
    nothing after. On the payment date a year out the bond's holding books on average the forward
    value of all it is still owed, 100.23, within the Monte Carlo error.

    Killing mutations: the close-out not booked, nothing then booked on the horizon; a coupon
    before the horizon dropped.
    """
    deal, factors, curves, calculation, horizons = LISTS[kind]
    horizon = horizons[0] if on == 'between' else horizons[1]
    deals = [held(deal, horizon), deal]
    dates, holding, cash_held = simulated(deals, 'HOLD', factors, curves, **calculation)
    _, listed, cash_listed = simulated(deals, 'LIST', factors, curves, **calculation)
    rows = sum(day <= horizon for day in dates)
    assert dates[rows - 1] == horizon and holding['HOLD'].shape[0] == rows, (rows, holding['HOLD'].shape)
    assert np.array_equal(holding['HOLD'], listed['LIST'][:rows])

    before = [day for day in cash_listed if day < horizon]
    assert before and all(cash_listed[day].any() for day in before), before
    assert all(np.array_equal(cash_held[day], cash_listed[day]) for day in before)
    assert np.array_equal(cash_held[horizon], listed['LIST'][rows - 1])
    assert (on == 'payment') == bool(cash_listed.get(horizon, np.zeros(1)).any())
    assert not any(cash_held[day].any() for day in cash_held if day > horizon)
    if kind == 'fixed' and on == 'payment':
        assert np.all(cash_listed[horizon] == COUPON) and near(cash_held[horizon], LATER / D(horizon))


def test_a_holding_is_not_the_forward_by_the_coupons_of_its_life_the_carry_and_the_strike():
    """The bond held a year against the same list bought forward that day at 98, carried on
    USD-PROJ's 4.5% repo and settled on the bond's own curve: the holding marks over the forward by
    the coupon of its first half year, 1.6171, which the forward does not own; the carry of what
    the forward buys, `U (1 - D / D_repo)` on its value U = 96.3001 with the coupon on the day,
    -0.4827; and the strike the forward pays, `98 D`, 94.1574: 95.2917 in all, by hand to 1e-12,
    the holding 97.9172 and the forward 2.6254.

    Killing mutation: the holding read as the forward, its coupons before the horizon dropped.
    """
    bond, horizon = LISTS['fixed'][0], months(12)
    forward = dict(bond, Reference='FORWARD', Settlement_Date=horizon, Settlement_Amount=STRIKE,
                   Repo_Rate='USD-PROJ')
    values = marks([held(bond, horizon), forward])
    life, strike = COUPON * D(months(6)), STRIKE * D(horizon)
    carry = LATER * (1.0 - D(horizon) / D(horizon, 0.045))
    hand = life + carry + strike
    assert abs((values['HOLD'] - values['FORWARD']) / hand - 1.0) < 1e-12, (values, life, carry, strike)
    assert [round(x, 4) for x in (life, LATER, carry, strike, hand, values['HOLD'], values['FORWARD'])] == [
        1.6171, 96.3001, -0.4827, 94.1574, 95.2917, 97.9172, 2.6254]


@pytest.mark.parametrize('kind', sorted(LISTS))
def test_a_list_stating_both_its_ends_is_refused_by_name_before_a_run_and_by_it(kind):
    """A list stating a `Settlement_Date` beside its `Investment_Horizon` is refused at booking by
    name - the fixed and the linker in their own words, the floating list as it refuses any
    settlement date - and the run refuses it naming the deal, rather than pricing either.

    Killing mutation: `instruments.forward_and_holding` saying nothing, the fixed list then
    booking clean and pricing as its forward.
    """
    deal, factors, _, _, horizons = LISTS[kind]
    both = dict(held(deal, horizons[1]), Settlement_Date=horizons[0])
    said = schema.validate_instrument(construct_instrument(dict(both), {}))
    words = ('Settlement_Date is stated, and a floating cashflow list settles no forward' if kind == 'floating' else
             'Settlement_Date and Investment_Horizon are both stated, and a forward bought on the one is not '
             'a holding ending on the other')
    assert words in said and schema.validate_instrument(construct_instrument(dict(deal), {})) == [], said
    with pytest.raises(utils.UnpriceableSchedule, match='HOLD: Settlement_Date and Investment_Horizon are both'):
        book.marks([both], factors=factors)


@pytest.mark.parametrize('kind', sorted(LISTS))
def test_a_horizon_behind_the_base_date_leaves_the_holding_out_as_expired(kind):
    """A holding whose horizon is a month behind the base date has ended: the base valuation leaves
    it out as an expired deal, counted nowhere and with no row, and prices its list beside it.

    Killing mutation: the dates not cut at the horizon, the holding marking as its list.
    """
    deal, factors, _, _, _ = LISTS[kind]
    hexes, stats = book.marks([deal, held(deal, months(-1))], factors=factors)
    assert set(hexes) == {'root', 'LIST'} and 'Deals Skipped' not in stats, (hexes, stats)


def test_the_diary_of_a_holding_pays_to_its_horizon_and_settles_there_and_nothing_after():
    """The diary of the bond held a year pays its two coupons of 1.65, the second on the horizon,
    and is closed out there at an amount the market decides, null on the `Settlement` leg; the
    floating list held nine months fixes and pays to its horizon and is closed out there; neither
    announces a payment or a fixing after it. The plain bond announces all six coupons, and held
    to its last payment or past it, it announces the same six and no settlement.

    Killing mutation: the diary reading a holding as its list, every coupon announced and no
    settlement; or a horizon with nothing after it still read as one, a seventh payment announced.
    """
    bond, float_list = LISTS['fixed'][0], LISTS['floating'][0]
    job = book.book([bond, held(bond, months(12)), held(float_list, months(9), 'FLOATING'),
                     held(bond, months(36), 'WHOLE'), held(bond, months(40), 'PAST')])
    context = derivus.Context()
    context.load_json((json.dumps(job, cls=CustomJsonEncoder), 'holding'))
    rows = {}
    for row in spine.diary(context):
        rows.setdefault(row['reference'], []).append(row)

    def payments(reference):
        return [(row['due_date'], row['leg'], row['amount']) for row in rows[reference] if row['kind'] == 'payment']

    day = months(12).date().isoformat()
    assert payments('HOLD') == [(months(6).date().isoformat(), 'Cashflows', COUPON), (day, 'Cashflows', COUPON),
                                (day, 'Settlement', None)], payments('HOLD')
    assert payments('WHOLE') == payments('PAST') == payments('LIST') and len(payments('LIST')) == 6
    end = months(9).date().isoformat()
    assert payments('FLOATING')[-1] == (end, 'Settlement', None) and len(payments('FLOATING')) == 3
    assert all(row['due_date'] <= end for row in rows['FLOATING']) and any(
        row['kind'] == 'fixing' and row['due_date'] > months(6).date().isoformat() for row in rows['FLOATING'])
    assert max(row['due_date'] for row in rows['HOLD']) == day
