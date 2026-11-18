"""The commodity barrier option, through the JSON contract and nothing else.

The type is the equity barrier's continuous arm on a reference price's fixing, watched as the
commodity touches watch it: the forward curve read at the delivery each date maps to is the
monitored price, and its carry to the expiry is the log-ratio of the expiry's forward to it. So a
commodity barrier on a reference whose forward curve IS an equity's forward prices as that equity
barrier does - the identity the first test holds every variant to.
"""
import functools
import logging
import math
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, 'tests')
import test_commodity_touch_json as touch  # noqa: E402
import test_declared_defaults as book  # noqa: E402
import test_option_settlement_json as settled  # noqa: E402
import trial_commodity  # noqa: E402
from derivus import utils  # noqa: E402

E = book.WORLD_EXPIRY
#: 100 units struck at the world equity's spot of 100, the notional every tolerance is a share of
UNITS, STRIKE = 100.0, 100.0
NOTIONAL = UNITS * touch.SPOT
LEVEL = {'Down': 90.0, 'Up': 110.0}
#: every barrier type, call and put, with no rebate and with one
VARIANTS = [(direction + '_And_' + knock, option_type, rebate) for direction in ('Down', 'Up')
            for knock in ('In', 'Out') for option_type in ('Call', 'Put') for rebate in (0.0, 500.0)]


def barrier(kind, reference, barrier_type, level, option_type='Call', **extra):
    """A bought barrier option on the world's expiry, struck at the spot."""
    return dict({'Object': kind, 'Reference': reference, 'Currency': 'USD', 'Discount_Rate': 'USD',
                 'Buy_Sell': 'Buy', 'Expiry_Date': E, 'Units': UNITS, 'Strike_Price': STRIKE,
                 'Option_Type': option_type, 'Barrier_Type': barrier_type, 'Barrier_Price': level}, **extra)


def both(tag, barrier_type, level, option_type='Call', equity=None, commodity=None, **extra):
    """The barrier booked on the world's equity as `EQ_<tag>` and on the reference `EQC`, whose
    forward is that equity's, as `CM_<tag>`."""
    return [barrier('EquityBarrierOption', 'EQ_' + tag, barrier_type, level, option_type,
                    Equity='EQ', Equity_Volatility='EQ', **(equity or {}), **extra),
            barrier('CommodityBarrierOption', 'CM_' + tag, barrier_type, level, option_type,
                    Reference_Type='EQC', Reference_Volatility='EQC', **(commodity or {}), **extra)]


def identical(deals, factors):
    """Each equity barrier among `deals` non-zero and its commodity twin equal to it to 1e-12 of
    the notional."""
    marks, _ = book.marks(deals, {**touch.reference('EQC'), **factors})
    for deal in deals[::2]:
        tag = deal['Reference'][3:]
        equity, commodity = (float.fromhex(marks[side + tag]) for side in ('EQ_', 'CM_'))
        assert equity != 0.0, tag
        assert commodity == pytest.approx(equity, abs=1e-12 * NOTIONAL), tag
    return marks


def test_a_commodity_barrier_prices_as_the_equity_barrier_its_forward_maps_to():
    """Sixteen variants - down and up, in and out, call and put, with no rebate and a 500 one -
    each booked on the world's equity and on a spot-like reference whose forward curve is that
    equity's forward (spot 100, carry 3%, the same surface): the two marks agree to 1e-12 of the
    10,000 notional, every mark non-zero. Measured worst: 9.1e-13 absolute.

    KILLING MUTATION: the carry left at zero, the closed form's forward the watched fixing itself -
    the down-and-in call reads 52.18 against the equity's 61.25.
    """
    identical([deal for barrier_type, option_type, rebate in VARIANTS
               for deal in both('{}_{}_{}'.format(barrier_type, option_type, int(rebate)), barrier_type,
                                LEVEL[barrier_type[:barrier_type.index('_')]], option_type,
                                Cash_Rebate=rebate)], {})


#: the trial's fuel surface with a skew: 30% at the money, 5 points lower per 10% of moneyness above
SKEW = {'CommodityPriceVol.FUEL': {
    'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness',
    'Surface': utils.Curve([], [[m, t, 0.30 - 0.5 * (m - 1.0)] for m in (0.8, 0.9, 1.0, 1.1, 1.2)
                                for t in (0.1, 1.0, 3.0)])}}


def test_with_no_rebate_the_knock_in_and_out_sum_to_the_energy_option_on_the_expiry_fixing():
    """On the trial's fuel reference, its forward sloping from 80 to 84, under a skewed surface: a
    knock-in and a knock-out struck at 82, down at 76 and up at 90, call and put, sum to the
    `EnergySingleOption` sampling the expiry's fixing alone on the same terms, to 1e-12 of it, each
    leg non-zero. The closed form's knock-out weights are the vanilla's less the knock-in's, so the
    two sum to its vanilla term, Black on the expiry's forward and discount at the vol read at the
    forward's moneyness, which is the energy option's Black on its one sample: what is left is
    rounding, 2.7e-15 of it measured.

    KILLING MUTATION: the barrier's vol read at the watched fixing's moneyness (`use_forwards`
    dropped) - the down call pair sums to 10,881.95 against the option's 10,117.21.
    """
    deals, pairs = [], [(direction, option_type) for direction in ('Down', 'Up') for option_type in ('Call', 'Put')]
    for direction, option_type in pairs:
        deals += [dict(trial_commodity.BARRIER, Reference=direction + option_type + knock, Cash_Rebate=0.0,
                       Barrier_Type='{}_And_{}'.format(direction, knock), Option_Type=option_type,
                       Barrier_Price={'Down': 76.0, 'Up': 90.0}[direction]) for knock in ('In', 'Out')]
    deals += [{'Object': 'EnergySingleOption', 'Reference': option_type, 'Currency': 'USD',
               'Discount_Rate': 'USD', 'Sampling_Type': 'FUEL', 'Reference_Type': 'FUEL',
               'Reference_Volatility': 'FUEL', 'Buy_Sell': 'Buy', 'Option_Type': option_type,
               'Strike': 82.0, 'Volume': 1000.0, 'Period_Start': E, 'Period_End': E, 'FX_Period_Start': E,
               'FX_Period_End': E, 'Settlement_Date': E, 'Realized_Average': 0.0, 'FX_Realized_Average': 0.0}
              for option_type in ('Call', 'Put')]
    marks, _ = book.marks(deals, {**trial_commodity.FACTORS, **SKEW})
    for direction, option_type in pairs:
        knocked_in, knocked_out = (float.fromhex(marks[direction + option_type + knock]) for knock in ('In', 'Out'))
        assert knocked_in > 0.0 and knocked_out > 0.0, (direction, option_type)
        assert knocked_in + knocked_out == pytest.approx(
            float.fromhex(marks[option_type]), rel=1e-12), (direction, option_type)


#: a quanto's levels are USD levels on the forward, a compo's EUR levels on S*X (80 today)
PAID_IN_EUR = [('Quanto', 'Down', 90.0, 100.0), ('Quanto', 'Up', 110.0, 100.0),
               ('Compo', 'Down', 72.0, 80.0), ('Compo', 'Up', 88.0, 80.0)]


def test_a_quanto_or_compo_commodity_barrier_prices_as_the_equity_barrier_it_maps_to():
    """Paid in EUR on a USD reference: each quanto and compo barrier, down and up, in and out, call
    and put, booked on the world's equity and on the reference whose forward is that equity's, the
    one correlation stated in each family's own spelling (the touch's) - the two agree to 1e-12 of
    the notional, every mark non-zero.

    KILLING MUTATION: the reference's correlation read with the opposite sign - the quanto
    down-and-in call reads 75.53 against the equity's 80.63.
    """
    identical([deal for ptype, direction, level, strike in PAID_IN_EUR for knock in ('In', 'Out')
               for option_type in ('Call', 'Put')
               for deal in both('{}_{}_{}_{}'.format(ptype, direction, knock, option_type),
                                '{}_And_{}'.format(direction, knock), level, option_type,
                                equity={'Payoff_Currency': 'EUR'}, commodity={'Currency': 'EUR'},
                                Discount_Rate='EUR', Payoff_Type=ptype, Strike_Price=strike)],
              touch.CROSS)


def test_a_monitoring_frequency_shifts_the_barrier_away_as_the_equity_barrier_s_does():
    """Monitored monthly, each barrier is shifted away from the live region, so every knock-out is
    worth more than its continuous twin and every knock-in less, down and up, call and put - and
    each equals the equity barrier at the same frequency to 1e-12 of the notional.

    KILLING MUTATION: the commodity's `Barrier_Monitoring` compiled as zero - the monthly
    down-and-in call reads its continuous 61.25 against the equity's 27.26.
    """
    monthly = identical([deal for barrier_type, option_type, _ in VARIANTS[::2]
                         for deal in both('{}_{}'.format(barrier_type, option_type), barrier_type,
                                          LEVEL[barrier_type[:barrier_type.index('_')]], option_type,
                                          Barrier_Monitoring_Frequency=pd.DateOffset(months=1))], {})
    continuous, _ = book.marks([barrier('CommodityBarrierOption', barrier_type + option_type, barrier_type,
                                        LEVEL[barrier_type[:barrier_type.index('_')]], option_type,
                                        Reference_Type='EQC', Reference_Volatility='EQC')
                                for barrier_type, option_type, _ in VARIANTS[::2]], touch.reference('EQC'))
    for barrier_type, option_type, _ in VARIANTS[::2]:
        shifted, unshifted = (float.fromhex(m) for m in (
            monthly['CM_{}_{}'.format(barrier_type, option_type)], continuous[barrier_type + option_type]))
        assert (shifted > unshifted) == barrier_type.endswith('Out'), (barrier_type, option_type, shifted, unshifted)


def df(days):
    """The world's flat 4% USD discount over `days`."""
    return math.exp(-0.04 * days / 365.0)


def test_a_barrier_settled_after_its_expiry_marks_its_blank_twin_discounted_and_books_where_the_fx_barrier_does():
    """`Settlement_Date` ten days after the expiry: each up barrier, in and out, without and with a
    500 rebate, marks its blank twin times the ten days' discount to 1e-12 - but for the knock-out's
    rebate, paid at the touch and not carried. Under a credit Monte Carlo a knock-out settled so
    books its payoff, non-zero on every path, on the days the FX knock-out delivering ten days after
    the same expiry books on.

    KILLING MUTATION: `Settlement` left out of the compile, the payoff priced as paid at the
    expiry - the knock-in reads its blank 715.39 against 714.60.
    """
    settled = E + pd.DateOffset(days=10)
    knocks = {'{}_{}'.format(knock, int(rebate)): ('Up_And_' + knock, rebate)
              for knock in ('In', 'Out') for rebate in (0.0, 500.0)}
    marks, _ = book.marks([barrier('CommodityBarrierOption', name + suffix, barrier_type, LEVEL['Up'],
                                   Reference_Type='EQC', Reference_Volatility='EQC', Cash_Rebate=rebate, **stated)
                           for name, (barrier_type, rebate) in knocks.items()
                           for suffix, stated in (('', {}), ('_D', {'Settlement_Date': settled}))],
                          touch.reference('EQC'))
    marked = {name: float.fromhex(value) for name, value in marks.items()}
    for name in ('In_0', 'Out_0', 'In_500'):
        assert marked[name + '_D'] == pytest.approx(marked[name] * df(10), rel=1e-12), name
    rebate = marked['Out_500'] - marked['Out_0']
    assert rebate > 0.0 and marked['Out_500_D'] == pytest.approx(marked['Out_0_D'] + rebate, rel=1e-12)

    def booked(deal, factors):
        _, out = book.simulated([deal], ['USD'], factors)
        return out['Results']['cashflows']['USD']

    # both forwards static and short of the level, so each knock-out pays its intrinsic on every path
    commodity = booked(barrier('CommodityBarrierOption', 'CM', 'Up_And_Out', LEVEL['Up'], Reference_Type='EQC',
                               Reference_Volatility='EQC', Settlement_Date=settled), touch.reference('EQC'))
    fx = booked(book.fx_leg('FXBarrierOption', 'FX', Expiry_Date=E, Delivery_Date=settled, Strike_Price=1.20,
                            Barrier_Type='Up_And_Out', Barrier_Price=1.40), {})
    assert list(commodity.index) == list(fx.index) and (commodity.to_numpy() != 0.0).all(), (commodity, fx)


def test_a_reference_rolling_to_another_contract_before_the_expiry_is_refused_by_name(caplog):
    """A reference stepping from one delivery to the next a month in watches a fixing that jumps at
    the roll, which the closed form cannot price: the barrier is skipped with its name and the roll
    in the log, in the touch's words, and the barrier on a reference that stays on one contract
    prices beside it.

    KILLING MUTATION: the roll check dropped, the barrier priced off one forward it never watched.
    """
    roll = touch.BASE + 30
    rolling = utils.Curve([], [[40000, touch.EXPIRY], [roll, touch.EXPIRY], [roll + 1, touch.EXPIRY + 31],
                               [60000, touch.EXPIRY + 31]])
    one_contract = utils.Curve([], [[40000, touch.EXPIRY], [60000, touch.EXPIRY]])
    factors = {**touch.reference('ROLL', rolling), **touch.reference('HOLD', one_contract)}
    deals = [barrier('CommodityBarrierOption', name + 'S', 'Up_And_Out', LEVEL['Up'], Reference_Type=name,
                     Reference_Volatility=name) for name in ('ROLL', 'HOLD')]
    with caplog.at_level(logging.ERROR):
        marks, _ = book.marks(deals, factors)
    assert 'ROLLS' not in marks and float.fromhex(marks['HOLDS']) != 0.0
    assert any('ROLLS' in r.getMessage() and 'rolls from one delivery to another' in r.getMessage()
               for r in caplog.records)


def test_under_a_credit_monte_carlo_the_knock_out_pays_its_rebate_on_the_grid_and_ends_at_its_expiry(caplog):
    """The trial's knock-out, up at 90 on the fuel reference with a 2,000 rebate, in a credit Monte
    Carlo with the discount curve simulated and the forward curve under a Clewlow-Strickland model:
    it is not skipped, every row is finite, the rebate is booked on grid dates before the expiry
    on some paths and not on others, the last booking is the expiry's, and the profile is positive
    before the expiry and zero after it.

    KILLING MUTATION: `pays_on_grid` answering False - the ledger books the expiry alone, where
    3.9% and 13.3% of the CPU's paths book the rebate on the two quarters before it as built.
    """
    deals = [d for d in trial_commodity.DEALS if d['Object'] == 'CommodityBarrierOption']
    with caplog.at_level(logging.ERROR):
        _, out = book.simulated(deals, ['USD'], trial_commodity.FACTORS, models={'ForwardPrice.FUEL': (
            'CSForwardPriceModel', {'Alpha': 0.35, 'Drift': 0.0, 'Sigma': 0.2})})
    assert not [r for r in caplog.records if 'skipped' in r.getMessage()]
    mtm, profile = out['Results']['mtm'], out['Results']['exposure_profile']
    assert mtm.shape == (len(profile), 256) and np.isfinite(mtm.to_numpy()).all()
    ledger = out['Results']['cashflows']['USD']
    paid = ledger.loc[ledger.index < E].to_numpy()
    assert (paid != 0.0).any() and (paid == 0.0).any() and max(ledger.index) == E, ledger.index
    before, after = profile.index < E, profile.index > E
    assert before.any() and after.any()
    assert (profile.loc[before, 'EE'] > 0.0).all() and (profile.loc[after].abs().to_numpy() == 0.0).all()


#: The trial's knock-out with its 2,000 rebate, without one, and its knock-in, each settled where
#: `settled.TWINS` says, `Settlement_Date` being the field the type settles on
FUEL = {'KOR': trial_commodity.BARRIER, 'KO': dict(trial_commodity.BARRIER, Cash_Rebate=0.0),
        'KI': dict(trial_commodity.BARRIER, Barrier_Type='Up_And_In', Cash_Rebate=0.0)}
TWINS = {tag: {'Settlement_Date': day['Delivery_Date']} if day else {} for tag, day in settled.TWINS.items()}


@functools.lru_cache(maxsize=None)
def delivered():
    """`FUEL` in each of `TWINS`, each in a netting set of its own beside a unit paid on the expiry and
    one forty days after it, in one float64 credit Monte Carlo on a monthly grid, the fuel forward a
    Clewlow-Strickland and a Hull-White on USD."""
    return settled.by_set([settled.held(name + tag, [dict(deal, Reference=name + tag, **stated)])
                           for name, deal in FUEL.items() for tag, stated in TWINS.items()] + settled.units(),
                          ['USD'], factors=trial_commodity.FACTORS, Time_Grid='0d 1m(1m)', models={'ForwardPrice.FUEL': (
                              'CSForwardPriceModel', {'Alpha': 0.35, 'Drift': 0.0, 'Sigma': 0.2})})


def test_stated_on_its_expiry_a_commodity_barrier_simulates_and_books_as_its_blank_twin():
    """Under a float64 credit Monte Carlo the trial's knock-out, with and without its rebate, and the
    knock-in each stated settling on its expiry simulate every row and book every cash flow, the
    ledger's days among them, to the bit as with `Settlement_Date` blank, and mark so on the base
    date.

    Killing mutation: a blank `Settlement_Date` read two days after the expiry (in
    `schema.settlement_days`), the blank twins' rows and cash moving off the stated ones.
    """
    _, ledgers, profiles = delivered()
    for name in FUEL:
        assert np.array_equal(profiles[name + '_E'], profiles[name]), name
        blank, stated = ledgers[name], ledgers[name + '_E']
        assert stated.index.equals(blank.index) and np.array_equal(stated.values, blank.values), name
    marks, _ = book.marks([dict(deal, Reference=name + tag, **stated) for name, deal in FUEL.items()
                           for tag, stated in list(TWINS.items())[:2]], trial_commodity.FACTORS)
    assert all(marks[name + '_E'] == marks[name] != (0.0).hex() for name in FUEL), marks


def test_delivered_after_its_expiry_a_commodity_barrier_holds_what_the_expiry_fixed_and_books_it_then():
    """Settled forty days after its expiry under that run, the rebate-free knock-out and the
    knock-in read on the monthly rows, the expiry and the settlement as every FX option delivered
    there does (`settled.delivered_identities`): to the expiry each row its blank twin's times that
    row's own forward discount to the settlement, to 1e-13; after it, the monthly row and the
    settlement's not zero and the payoff discounted to them; nothing booked on the expiry; and the
    settlement booking, path by path, to the bit what the blank twin books on its expiry. The
    knock-out's rebate is paid at the touch: every one before the expiry booked to the bit as its
    blank twin books it, and the expiry's and the settlement's cash summing to that twin's on its
    expiry to 1e-13.

    Killing mutation: the cash left on the expiry row (`pv_barrier_option` booking its expiry row's
    mark there).
    """
    dates, ledgers, profiles = delivered()
    settled.delivered_identities(dates, ledgers, profiles, ('KO', 'KI'), E, settled.DELIVERED, exact=('KO', 'KI'))
    blank, late = ledgers['KOR'], ledgers['KOR_D']
    before = [day for day in blank.index if day < E]
    assert np.array_equal(late.loc[before].values, blank.loc[before].values)
    assert np.asarray(late.loc[before].values, dtype=float).any()
    on_expiry, paid = (np.asarray(late.loc[day].values, dtype=float) for day in (E, settled.DELIVERED))
    assert np.allclose(on_expiry + paid, np.asarray(blank.loc[E].values, dtype=float), rtol=1e-13, atol=1e-13)


def test_between_its_expiry_and_its_settlement_a_commodity_barrier_marks_its_print_and_refuses_without_it():
    """On a base date the day after the expiry, settled forty days after it: the trial's knock-out
    without its rebate marks the expiry's fixing `Price_Fixing` states, 85 under its 90 barrier,
    `1000 (85 - 82)` discounted from the settlement, to 1e-12, no touch before the base date known;
    stating no print it refuses by name, naming the deal, the table and the day; and a `Compo` one,
    whose S*X no print of the reference fixes, refuses by name with its print stated.

    Killing mutation: the print not read (`held_level` in `pv_barrier_option` passed over), the
    knock-out at today's fixing's intrinsic.
    """
    base, paid_on = E + pd.offsets.BDay(1), settled.DELIVERED

    def job(deal, factors):
        document = book.book([dict(deal, Settlement_Date=paid_on)])
        document['Calc']['MergeMarketData']['ExplicitMarketData']['Price Factors'].update(factors)
        return document

    fixed = [[E, 85.0]]
    marked = settled.valued_on(job(dict(FUEL['KO'], Price_Fixing=fixed), trial_commodity.FACTORS), base)
    assert marked['FUEL_KO'] == pytest.approx(1000.0 * 3.0 * df((paid_on - base).days), rel=1e-12)
    compo = barrier('CommodityBarrierOption', 'COMPO', 'Up_And_Out', 88.0, Reference_Type='EQC',
                    Reference_Volatility='EQC', Currency='EUR', Discount_Rate='EUR', Payoff_Type='Compo',
                    Strike_Price=80.0, Price_Fixing=fixed)
    for deal, factors, words in ((FUEL['KO'], trial_commodity.FACTORS, ('FUEL_KO', 'Price_Fixing', '2027-08-03')),
                                 (compo, {**touch.reference('EQC'), **touch.CROSS}, ('COMPO', 'a compo expired'))):
        with pytest.raises(Exception) as refusal:
            settled.valued_on(job(deal, factors), base)
        assert all(word in str(refusal.value) for word in words), str(refusal.value)
