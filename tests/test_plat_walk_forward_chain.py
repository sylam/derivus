"""The walk-forward chain driver's `--regime-ramp`: the corridor it writes into one trade's job, built from a synthetic
archive through the driver's own command line and `build_deal_config`."""
import importlib
import os

import numpy as np
import pandas as pd

TRADE = pd.Timestamp('2024-03-01')
LEGS = ('PL_M1', 'PL_M2', 'PL_M3')
FIX, BASIS = 'CommodityPrice.LBMA_AM', 'ObservedBasis.LBMA_AM.CME'


def world(tmp_path):
    """The template, its calendar (London, weekends only), an empty calibrated model block and the listed expiries."""
    cal = tmp_path / 'calendars.cal'
    cal.write_text('<Calendars><Calendar Location="London" Weekends="Saturday and Sunday" Holidays=""/></Calendars>')
    md = tmp_path / 'md.json'
    md.write_text('{"MarketData": {"Price Models": {}}}')
    expiries = tmp_path / 'expiries.csv'
    expiries.write_text('Expiry_Date\n2024-04-26\n2024-07-29\n2024-10-29\n2025-01-28\n')
    template = {'Calc': {'CalendDataFile': str(cal), 'MergeMarketData': {'ExplicitMarketData': {}}, 'Calculation': {
        'Calendar': 'London', 'Hedging_Problem': {
            'Objective': {}, 'Portfolio_State': {},
            'Liabilities': {'CommodityAveragePriceSwapDeal': {'AM': {'Commodity': 'LBMA_AM', 'Buy_Sell': 'Buy'}}},
            'Tradable_Instruments': {
                'CommodityFutureDeal': {n: {'Commodity': 'LBMA_AM.CME', 'Contract_Size': 50} for n in LEGS},
                'CashAccountDeal': {'USD_CASH': {}}},
            'Evaluator': {'Position_Limits': {n: {'Min_Position': -60, 'Max_Position': 0} for n in LEGS}}}}}}
    return template, str(md), ['--expiries', str(expiries), '--long-cap', '10']


def archive(hot=1.0):
    """Weekdays 2014-06 to 2024-08: the fix a walk of $10 daily changes, the AM.CME basis noise whose sd cycles 1, 2, 4
    a month, then 50 from 2024-01-15 (a ratio over every earlier row), 2 from 2024-03-18 (the middle tercile) and
    0.01 from May (under every earlier row once its 21 rows are May's); `hot` scales the basis from 2024-01-15."""
    days = pd.bdate_range('2014-06-02', '2024-08-30')
    rng = np.random.default_rng(7)
    sd = np.select([days >= '2024-05-01', days >= '2024-03-18', days >= '2024-01-15'], [0.01, 2.0, 50.0],
                   np.array([1.0, 2.0, 4.0])[days.month % 3]) * np.where(days >= '2024-01-15', hot, 1.0)
    return pd.DataFrame({FIX: 1000.0 + np.cumsum(rng.normal(0.0, 10.0, len(days))),
                         BASIS: sd * rng.normal(0.0, 1.0, len(days)), 'ObservedBasis.LBMA_AM.PM': 0.0,
                         'ObservedBasis.LBMA_AM.PM.CME': 0.0, 'ForwardRate.PLATINUM_CARRY,0.5': 0.01,
                         'ForwardRate.PLATINUM_CARRY,1.0': 0.012}, index=days)


def regime(arch, days):
    """r at each of `days` as the pre-registration states it: the share of the ratios of the rows from 2015 to the
    day before the trade at or below the ratio through the row before the day, 0 to a third, 1 from two thirds."""
    ratio = arch[BASIS].ffill().rolling(21).std() / arch[FIX].diff().rolling(21).std()
    window = ratio[(ratio.index >= '2015-01-01') & (ratio.index < TRADE)].dropna()
    return np.array([min(max(3.0 * (window <= ratio[ratio.index < d].iloc[-1]).mean() - 1.0, 0.0), 1.0)
                     for d in days])


def test_the_regime_ramp_is_the_record_at_one_and_the_passive_at_zero_off_the_day_before(tmp_path, monkeypatch):
    """One trade's job (2024-03-01, its June fixings, the record's corridor [-60, +10] by `--long-cap 10`) with and
    without `--regime-ramp`. Where r = 1 at a decision the knot in force there is the record's corridor, and a trade
    seeing r = 1 at every decision writes the record's job itself; where r = 0 both sides are the passive net, the
    liability's quantity still to fix after the day over the contract size, whole and short, read off the job's own
    `Sampling_Data`; and the knot at a decision moves with a basis spike on the archive row before it and not with
    one on its own row, which reaches the next decision's.

    Killing mutation: the ratio read through the decision day itself (`regime_ramp`'s `searchsorted(decisions)` with
    `side='right'`): 2024-04-15, r = 1 off the row before, reads (-58.44, 0.66), and the spike on 2024-06-12's own
    row opens its knot from the passive's (-30, -30) to (-60, 10), each red alone. Also red: the passive counted with
    the decision day's fixing still unfixed (`fixings > day` as `>=`), the r = 0 knots from 2024-06-03 a fixing
    short (-50 where the passive is -48)."""
    # the harness caps a visible card's memory at import, so the driver is imported with the cards hidden
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    monkeypatch.syspath_prepend(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'experiments'))
    chain = importlib.import_module('plat_walk_forward_chain')
    template, md, flags = world(tmp_path)
    ramp, record = chain.parse_args(flags + ['--regime-ramp']), chain.parse_args(flags)
    arch = archive()

    def knots(a, args=ramp):
        cfg, _ = chain.build_deal_config(template, a, TRADE, md, args)
        return cfg, cfg['Calc']['Calculation']['Hedging_Problem']['Evaluator']['Total_Position_Schedule']

    def at(schedule, day):
        k = [k for k in schedule if k['Step'] <= (day - TRADE).days][-1]
        return k['Min_Total'], k['Max_Total']

    cfg, schedule = knots(arch)
    (leg,) = cfg['Calc']['Calculation']['Hedging_Problem']['Liabilities']['CommodityAveragePriceSwapDeal'].values()
    fixings = [pd.Timestamp(f['.Timestamp']) for f, _, _ in leg['Sampling_Data']]
    days = pd.bdate_range(TRADE, leg['Settlement_Date']['.Timestamp'])
    r = regime(arch, days)
    passive = [round(-leg['Units'] * sum(f > d for f in fixings) / len(fixings) / 50) for d in days]
    assert (r == 1).sum() > 20 and (r == 0).sum() > 20 and ((r > 0) & (r < 1)).sum() > 5
    assert all(at(schedule, d) == (-60.0, 10.0) for d in days[r == 1])
    assert all(at(schedule, d) == (p, p) for d, p, ri in zip(days, passive, r) if ri == 0)
    assert len({p for p, ri in zip(passive, r) if ri == 0}) > 10            # the r = 0 knots follow the fixings

    hot = archive(hot=1000.0)
    assert (regime(hot, days) == 1).all()
    assert knots(hot) == knots(hot, record)

    day = pd.Timestamp('2024-06-12')
    assert regime(arch, [day, day + pd.offsets.BDay()]).tolist() == [0.0, 0.0]
    on_day, before = arch.copy(), arch.copy()
    on_day.loc[day, BASIS] = before.loc[day - pd.offsets.BDay(), BASIS] = 1000.0
    spiked = knots(on_day)[1]
    assert at(spiked, day) == at(schedule, day) == (passive[days.get_loc(day)],) * 2
    assert at(spiked, day + pd.offsets.BDay()) == (-60.0, 10.0)
    assert at(knots(before)[1], day) == (-60.0, 10.0)
