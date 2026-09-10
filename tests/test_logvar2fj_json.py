"""LogVar2FJ through the JSON contract: the limits, the calibration's rulings and the hex set.

Lane T. Every gate is a real document over `tests/fixtures/data/logvar2fj_world.json` - a
synthetic index `INDEX_A` quoted in EUR on a USD book, a hand-authored skewed five-expiry ladder,
and a GBM sibling to correlate against. Nothing is mocked and nothing is patched; a fit IS run,
at `Paths` 2048 and a small `Max_Iterations`, and what is asserted is what a ruling fixed rather
than theta*.

THE FIXTURE-DEGENERACY CHECKLIST for the world, per axis:

  r, q          varied - EUR 3%->4%, USD 4.5%->5.5%, dividend 1.5%. Both curves slope, so an
                interval carry cannot be read off a raw zero rate.
  time rows     varied - four autocall fixings, and the credit grid `1d 3m(3m) 2y` reports
                thirteen scenario dates, twelve of them past the first.
  side          varied - `Buy` and `Sell` mirror in the GBM limit.
  option type   the autocall carries a `Put` at 70% of strike under its coupon ladder, and every
                fixing is a barrier date, so the downside leg is live (see gate 1).
  coupon        varied and non-zero - 2/4/6/8% on the four fixings.
  vol surface   TWO surfaces on purpose: the PRICING surface is flat at the GBM limit's own 20%,
                because gate 1 IS the flat limit and cannot be stated on any other; every FITTED
                gate reads the ladder, which carries 30 vol points of skew per unit of
                log-moneyness and a vol point of term slope a year.
  correlation   varied and non-zero - the quanto reads a marked equity/FX -0.35 and the credit
                gate a declared +0.30, each against its own zero control.
  netting sets  one, uncollateralised: the CVA objective is the root sum and a second set adds
                rows to the same tensor.
  the conjunction  gate 1 needs the LogVar2FJ arm AND the GBM arm on ONE document differing only
                in the `SpotModel` switch; gate 9 needs a LogVar2FJ OUTER process AND a declared
                correlation AND a scenario report, which is one run.
"""
import copy
import io
import json
import logging
import os
import re
import sys

# reference-derivus shadow-import guard (MEMORY): pin the package under test to THIS repo.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

import derivus as rf
from derivus.config import CustomJsonEncoder

HERE = os.path.dirname(os.path.abspath(__file__))
WORLD = os.path.join(HERE, 'fixtures', 'data', 'logvar2fj_world.json')
HEX = os.path.join(HERE, 'fixtures', 'logvar2fj_hex.json')
PRIORS_OFF_FILE = os.path.join(HERE, 'fixtures', 'logvar2fj_priors_off.json')

with open(WORLD) as _handle:
    _W = json.load(_handle)['MarketData']

BASE = _W['System Parameters']['Base_Date']['.Timestamp']
FACTOR = 'LogVar2FJModelParameters.INDEX_A'
BLOCK = 'LogVar2FJModelPrices.INDEX_A'
#: the world's PRICING surface is flat at this vol, and the GBM limit's xi is its variance
XI = _W['Price Factors']['EquityPriceVol.INDEX_A.EUR']['Surface']['.Curve']['data'][0][2] ** 2
#: the four fixings and their coupons - the deal every base-valuation gate prices
FIXINGS = ['2024-12-27', '2025-06-27', '2025-12-26', '2026-06-26']
COUPONS = [0.02, 0.04, 0.06, 0.08]


def _autocall(**overrides):
    """The synthetic autocall: four quarterly-ish fixings, a coupon ladder and a 70% terminal put."""
    deal = {'Object': 'QEDI_CustomAutoCallSwap', 'Reference': 'AC1', 'Tags': '', 'MtM': '',
            'Currency': 'EUR', 'Payoff_Currency': 'EUR', 'Equity': 'INDEX_A',
            'Dividends': 'INDEX_A', 'Discount_Rate': 'EUR', 'Equity_Volatility': 'INDEX_A.EUR',
            'Buy_Sell': 'Buy', 'Option_Type': 'Put', 'Strike_Price': 100.0,
            'Expiry_Date': {'.Timestamp': FIXINGS[-1]}, 'Units': 10.0,
            'Settlement_Style': 'Cash', 'Option_On_Forward': 'No', 'Option_Style': 'European',
            'Barrier': 0.7, 'Payoff_Type': 'Standard',
            'Price_Fixing': [[{'.Timestamp': d}, 0.0] for d in FIXINGS],
            'Autocall_Coupons': [[{'.Timestamp': d}, c] for d, c in zip(FIXINGS, COUPONS)],
            'Autocall_Thresholds': [[{'.Timestamp': d}, 1.0] for d in FIXINGS],
            'Barrier_Dates': [{'.Timestamp': d} for d in FIXINGS], 'Autocall_Floating': []}
    deal.update(overrides)
    return deal


def _job(calc, deals=(), **sections):
    """THE ONE DOCUMENT SHAPE: the world file, one calculation, one book, and whatever sections a
    gate overrides on top of the file (`Price Factors` merges per factor key)."""
    return {'Calc': {
        'Calculation': calc,
        'MergeMarketData': {'MarketDataFile': WORLD, 'ExplicitMarketData': dict(sections)},
        'Deals': {'Reference': 'test', 'Tag_Titles': '',
                  'Deals': {'Children': list(deals)}}}}


def _base(greeks='No', paths=8192, **extra):
    return dict({'Object': 'BaseValuation', 'Base_Date': {'.Timestamp': BASE}, 'Currency': 'USD',
                 'Greeks': greeks, 'MCMC_Simulations': paths, 'Random_Seed': 1}, **extra)


def _curve(rows):
    return {'.Curve': {'meta': [], 'data': rows}}


#: THE GBM LIMIT factor - every lever at zero, a Gaussian residual and a flat xi at the pricing
#: surface's own variance. Authored here and not in the world file because a calibrated factor is a
#: RESULT: one sitting in the market data warm starts every fit that runs over it, and a Gaussian
#: one refuses to seed an NIG block by name.
GBM_LIMIT = {'Kappa_L': 0.5, 'Sigma_L': 0.0, 'Rho_L': 0.0, 'Kappa_S': 6.0,
             'Cap_A': 4.605170185988092, 'Cap_Beta': 0.25, 'C_Min': 0.12,
             'Residual_Law': 'Gaussian', 'On_Guard': '', 'Skew_Gradient': '',
             'Stickiness_Band': 0.0, 'Xi_Curve': _curve([[0.0, XI], [5.0, XI]]),
             'Rho_S': _curve([[0.0, 0.0]]), 'Sigma_S': _curve([[0.0, 0.0]]),
             'Alpha': _curve([[0.0, 1.0]]), 'Beta': _curve([[0.0, 0.0]])}

#: A LIVE NIG factor: |Beta| < Alpha, |Beta+1| < Alpha, the conditioning share 0.94 over the 0.4
#: bound and c = 0.48 over C_Min. The limit's Gaussian residual leaves Alpha and Beta UNREAD, so
#: they carry no derivative and no reserve can be composed off them.
LIVE_NIG = {'Residual_Law': 'NIG', 'Sigma_L': 1.0, 'Rho_L': -0.4,
            'Rho_S': _curve([[0.0, -0.6]]), 'Sigma_S': _curve([[0.0, 2.4]]),
            'Alpha': _curve([[0.0, 8.0]]), 'Beta': _curve([[0.0, -2.0]])}


def _priced(spot_model='LogVar2FJ', factor=None, **sections):
    """The sections a PRICED document carries: the `SpotModel` switch the deal reads and the
    LogVar2FJ factor it reads it off, varied."""
    return dict({
        'Valuation Configuration': {'QEDI_CustomAutoCallSwap': {
            'SpotModel': spot_model, 'Steps_Per_Year': 252.0, 'Internal_Step_Days': 1}},
        'Price Factors': dict({FACTOR: dict(GBM_LIMIT, **(factor or {}))},
                              **sections.pop('Price Factors', {}))}, **sections)


def _dumps(job):
    """The document on the wire - the engine's own encoder, so a written factor's Curves round
    trip back through the contract rather than through a repr."""
    return json.dumps(job, cls=CustomJsonEncoder)


def _run(job, name='lv'):
    cx = rf.Context()
    cx.load_json((_dumps(job), name + '.json'))
    return cx.run_job()[1]


def _mtm(out, reference='AC1'):
    rows = out['Results']['mtm']
    return float(rows[rows['Reference'] == reference]['Value'].iloc[0])


# ------------------------------------------------------------------------------------------
# 1  THE GBM LIMIT
# ------------------------------------------------------------------------------------------
def test_the_gbm_limit_reproduces_the_gbm_arm():
    """Every lever at zero with a Gaussian residual and a flat xi IS geometric Brownian motion.

    HOLDS 1e-12 relative (lane 1 measured 8.2e-16; this document reads 1.3e-16 on the Buy leg).
    The two runs differ in ONE character of the document - `SpotModel` - so the deal, the draws
    and the discounting are the same program, and the walk's own arithmetic is the whole of what
    is being compared.

    BOTH SIDES, because a mirrored sign is the cheapest thing a walk can get wrong and a Buy-only
    gate cannot see it.

    THE DEAL IS NOT A COUPON STRIP. Buy reads -0.5396: the coupon ladder alone is +0.2327 (measured
    at `Barrier` 0, where the pricer reads the identical number) and the barrier put -0.7723, so
    the SIGN is the put leg's and a fixture whose downside was dead could not produce it. It was
    dead: with `Barrier_Dates` empty the whole deal is its coupons, +0.2327, and every autocall
    gate here would have run on a payoff with no barrier in it.
    """
    for side, sign in (('Buy', -1.0), ('Sell', +1.0)):
        deals = [{'Instrument': {'.Deal': _autocall(Buy_Sell=side)}}]
        lv = _mtm(_run(_job(_base(), deals, **_priced('LogVar2FJ')), 'lv_limit'))
        gbm = _mtm(_run(_job(_base(), deals, **_priced('None')), 'gbm_limit'))
        assert abs(lv - gbm) <= 1e-12 * abs(gbm), (side, lv, gbm)
        assert sign * gbm > 0.5, ('the put leg is not in the number', side, gbm)


# ------------------------------------------------------------------------------------------
# 7  THE QUANTO ARM
# ------------------------------------------------------------------------------------------
def test_the_quanto_loading_is_off_at_zero_correlation():
    """A quanto drift is `-rho_q sigma_FX sqrt(V_k delta_k)` per day, so at `rho_q = 0` the walk is
    the un-quantoed one BIT FOR BIT - `utils.lv_walk` adds a tensor of exact zeros.

    HOLDS 0 ULP (-0x1.f68b7d9f0e5d5p-2) against the SINGLE-CURRENCY TWIN - the same schedule paid
    in the same USD off the same EUR discount curve, which is the one document differing from the
    quanto in nothing but the measure change. Lane Q. The marked -0.35 moves the same deal to
    -0.4269, so the loading is live on this world and the gate is not reading a drift of zero
    twice.
    """
    quanto = _autocall(Payoff_Type='Quanto', Payoff_Currency='USD')
    twin = _autocall(Currency='USD', Payoff_Currency='USD')
    zero = _priced(**{'Price Factors': {
        'Correlation.EquityPrice.INDEX_A.EUR/FxRate.EUR.USD': {'Value': 0.0}}})
    a = _mtm(_run(_job(_base(), [{'Instrument': {'.Deal': quanto}}], **zero), 'quanto_zero'))
    b = _mtm(_run(_job(_base(), [{'Instrument': {'.Deal': twin}}], **_priced()), 'quanto_twin'))
    marked = _mtm(_run(_job(_base(), [{'Instrument': {'.Deal': quanto}}], **_priced()),
                       'quanto_rho'))
    assert a.hex() == b.hex(), (a, b)
    assert abs(marked - a) > 0.01 * abs(a), ('the quanto loading is inert here', marked, a)


def test_the_quanto_gbm_limit_reproduces_the_gbm_quanto_deal():
    """The quanto document at the GBM limit against the GBM arm's own quanto carry.

    The walking arm reads the fx surface's ATM FORWARD strip on its own daily grid where the GBM
    arm reads ONE expiry ATM and calls it the whole deal's; on the FLAT fx surface this world
    carries the two are the same number. HOLDS 1e-12 relative (lane Q measured 7.6e-16).
    """
    deals = [{'Instrument': {'.Deal': _autocall(Payoff_Type='Quanto', Payoff_Currency='USD')}}]
    lv = _mtm(_run(_job(_base(), deals, **_priced('LogVar2FJ')), 'quanto_lv'))
    gbm = _mtm(_run(_job(_base(), deals, **_priced('None')), 'quanto_gbm'))
    assert abs(lv - gbm) <= 1e-12 * abs(gbm), (lv, gbm)


# ------------------------------------------------------------------------------------------
# 8  THE RESERVE IS COMPOSED AS DOCUMENTED
# ------------------------------------------------------------------------------------------
#: a reserve line a fit could have written, and a band wide enough that the reserve is not noise
SKEW_GRADIENT = '-0.0632095348,8.2032296200'
STICKINESS_BAND = 0.5


def test_the_skew_reserve_is_the_minimum_norm_contraction():
    """`Skew_Reserve` on the mtm frame is `|dPV/dDelta_skew| x Stickiness_Band`, and the parameter
    move behind a vol point of `Delta_skew` is the MINIMUM-NORM one.

    Composed here by hand off the factor's own `Skew_Gradient` and the reported first-order greeks
    - `|g . row| / (row . row) x band` - so the gate states the arithmetic rather than calling the
    engine's own helper. HOLDS to 1e-9 relative.
    """
    out = _run(_job(_base(greeks='First'), [{'Instrument': {'.Deal': _autocall()}}],
                    **_priced(factor=dict(LIVE_NIG, Skew_Gradient=SKEW_GRADIENT,
                                          Stickiness_Band=STICKINESS_BAND))), 'reserve')
    frame = out['Results']['Greeks_First']
    column = [c for c in frame.columns if c != 'Value'][0]
    lever = [float(frame.loc[[i for i in frame.index
                              if str(i[0]) == '%s.%s' % (FACTOR, name)][-1], column])
             for name in ('Beta', 'Rho_S')]
    row = [float(x) for x in SKEW_GRADIENT.split(',')]
    hand = abs(np.dot(lever, row) / np.dot(row, row)) * STICKINESS_BAND
    reported = float(out['Results']['mtm']['Skew_Reserve'].iloc[0])
    assert hand > 0.0, 'the reserve is being read where the deal has no skew sensitivity'
    assert abs(reported - hand) <= 1e-9 * hand, (reported, hand)


def test_a_factor_with_no_reserve_line_reports_none():
    """A blank `Skew_Gradient` states no reserve, and the mtm frame carries no column for one.

    The same LIVE NIG factor as the gate above with its line blanked, so the difference between
    the two runs is the LINE and not the model.
    """
    out = _run(_job(_base(greeks='First'), [{'Instrument': {'.Deal': _autocall()}}],
                    **_priced(factor=LIVE_NIG)), 'no_reserve')
    assert 'Skew_Reserve' not in out['Results']['mtm'].columns


# ------------------------------------------------------------------------------------------
# THE CALIBRATION HALF - one `Config.bootstrap` per gate, at `Paths` 2048 on the world's own
# five-expiry ladder. A fit IS run; what is asserted is what a ruling fixed, never theta*.
# ------------------------------------------------------------------------------------------
def _ladder(flat=False, rungs=None, **declared):
    """The world's ladder with its instrument varied, optionally re-quoted FLAT and optionally cut
    to its first `rungs` expiries - a fit costs one inner pillar pass per ATM expiry per iterate
    over a walk as long as the last one, so a gate that is about a BRANCH takes the short ladder."""
    block = copy.deepcopy(_W['Market Prices'][BLOCK])
    quotes = block['instrument']['European_Options']
    if rungs:
        keep = sorted({row['Expiry_Date']['.Timestamp'] for row in quotes})[:rungs]
        quotes = [row for row in quotes if row['Expiry_Date']['.Timestamp'] in keep]
        block['instrument']['European_Options'] = quotes
    if flat:
        for row in quotes:
            row['Quoted_Market_Value'] = XI ** 0.5
    block['instrument'].update(declared)
    return {'Market Prices': {BLOCK: block}}


def _fit(sections, name='fit'):
    """`(written factor, the INFO report)` - the report is half of what a calibration gate reads:
    a number can be right while the fit decided something else."""
    buf, root = io.StringIO(), logging.getLogger()
    handler, level = logging.StreamHandler(buf), root.level
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    try:
        cx = rf.Context()
        cx.load_json((_dumps(_job(_base(), (), **sections)), name + '.json'))
        cx.bootstrap()
    finally:
        root.removeHandler(handler)
        root.setLevel(level)
    return cx.current_cfg.params['Price Factors'].get(FACTOR), buf.getvalue()


def _refused(sections, name):
    """The refusal a document earns, however it is delivered: a fit that raises out of
    `bootstrap` answers its message, and a FAMILY that cannot be constructed at all is caught and
    logged by `Config.bootstrap`, which leaves the factor unwritten. Either way the name is said."""
    try:
        factor, report = _fit(sections, name)
    except ValueError as failure:
        return str(failure)
    assert factor is None, 'the document was fitted, not refused'
    return report


def _report_floats(report, marker, where=None):
    """Every float after `marker` on the last report line carrying `where or marker`."""
    line = [ln for ln in report.splitlines() if (where or marker) in ln][-1]
    return [float(x) for x in re.findall(r'[-+]?\d*\.?\d+(?:e[-+]?\d+)?',
                                         line.split(marker, 1)[1])]


@pytest.fixture(scope='module')
def fitted():
    """ONE fit of the world's skewed ladder, shared by every gate that reads a clean one: 2048
    paths, `Max_Iterations` 20, 87 s on an idle RTX 3090 and 213 s with three lanes on it."""
    return _fit(_ladder(), 'ladder')


# ------------------------------------------------------------------------------------------
# 3  THE CALIBRATION'S CONTRACT
# ------------------------------------------------------------------------------------------
def test_every_atm_pillar_reprices_exactly(fitted):
    """The xi strip is re-bootstrapped at every outer iterate, so theta* reprices the ATM term
    structure EXACTLY and is judged on the smile alone.

    HOLDS 1e-10 relative premium per pillar; this ladder's five read 8.0e-15, 8.6e-15, 1.2e-12,
    1.8e-11 and 4.0e-11 - the Newton solve's own tolerance, not a fit residual.
    """
    misses = _report_floats(fitted[1], 'ATM misses')
    assert len(misses) == 5, misses
    assert max(abs(x) for x in misses) <= 1e-10, misses


def test_the_wing_rmse_is_inside_its_declared_bound(fitted):
    """The smile the fit is JUDGED on: 15 quotes carrying 30 vol points of skew per unit of
    log-moneyness, fitted to 0.789 vol points RMSE unweighted.

    The bound is 1.0 vol points, 27% over the reading, because what this gate defends is that the
    ladder is FITTED at all - the same document read at its own seed is 3.247 vol points, which
    the report prints beside the fit as the slow pair's unfitted row.
    """
    rmse = _report_floats(fitted[1], 'RMSE', 'vol points unweighted')[0]
    assert rmse <= 1.0, rmse


def test_the_written_factor_carries_every_declared_field(fitted):
    """What a fit WRITES is the contract the pricer reads: the structural scalars, the five curves
    on their own knots, the reserve line and the guard flag - every one present, or a document
    priced off this factor reads a schema default nothing measured.
    """
    factor = fitted[0]
    for name in ('Kappa_L', 'Sigma_L', 'Rho_L', 'Kappa_S', 'Cap_A', 'Cap_Beta', 'C_Min',
                 'Residual_Law', 'On_Guard', 'Skew_Gradient', 'Stickiness_Band'):
        assert name in factor, name
    for name in ('Xi_Curve', 'Rho_S', 'Beta', 'Sigma_S', 'Alpha'):
        assert factor[name].array.shape[1] == 2, name
    assert factor['Xi_Curve'].array.shape[0] == 5, 'one xi segment per ATM expiry'
    assert (factor['Xi_Curve'].array[:, 1] > 0.0).all(), 'xi is a variance'
    assert len(str(factor['Skew_Gradient']).split(',')) == 2, factor['Skew_Gradient']
    assert float(factor['Stickiness_Band']) > 0.0


def test_the_identification_line_is_reported(fitted):
    """Each prior row's column norm over ONE quote row's, per coordinate and per stage - the line
    that says whether a fitted number is the data's or the row's.

    Three tables on this ladder - the residual pair, the leverage pair and the joint polish - each
    naming its coordinates, its singular values and its prior ratios.
    """
    lines = [ln for ln in fitted[1].splitlines() if 'identification,' in ln]
    assert len(lines) == 3, lines
    assert all('singular values' in ln and 'column norms' in ln for ln in lines)
    assert sum('multiple of ONE quote row' in ln for ln in fitted[1].splitlines()) == 3


# ------------------------------------------------------------------------------------------
# 5  THE ON-GUARD FLAG
# ------------------------------------------------------------------------------------------
def test_a_clean_fit_writes_no_guard(fitted):
    """A fit no box is holding says so with a blank `On_Guard`, so nothing priced off it inherits
    a warning it did not earn. This ladder fits clean."""
    assert str(fitted[0]['On_Guard']) == '', fitted[0]['On_Guard']


def test_a_fit_forced_onto_a_box_names_it_and_the_mark_reports_it():
    """A parameter a BOX stopped is not a fitted one, so `On_Guard` names the box on the factor and
    `Base_Revaluation` reports it in `Stats` under the same key.

    Forced with `Sigma_S_Bounds` 0.5,0.6 - a vol-of-vol window this ladder cannot fit inside - on
    the two-rung ladder at `Paths` 512 and `Max_Iterations` 3: enough to land ON the edge, not
    enough to be a calibration. The clean five-rung fit above writes the flag blank.
    """
    factor, _ = _fit(_ladder(rungs=2, Sigma_S_Bounds='0.5,0.6', Paths=512, Max_Iterations=3,
                             Stationary_Spread='Floor'), 'guarded')
    flag = str(factor['On_Guard'])
    assert flag.startswith('ON GUARD:') and 'Sigma_S' in flag, flag

    out = _run(_job(_base(), [{'Instrument': {'.Deal': _autocall()}}],
                    **_priced(factor=factor)), 'guarded_mark')
    assert out['Stats']['On_Guard'][FACTOR] == flag, out['Stats'].get('On_Guard')


# ------------------------------------------------------------------------------------------
# 4  THE PRIORS' RULINGS AS REFUSALS
#
# Each row is a declaration this family no longer reads, or reads and cannot use, and the name it
# must refuse under. No fit runs: every one of these raises out of `LVFit.__init__` or the family's
# own construction, before a quote is priced.
# ------------------------------------------------------------------------------------------
def _class_table(equity):
    """One class prior per underlying this family fits - a missing class refuses on its own, which
    is not the refusal a row here is about."""
    return ';'.join(['EquityPrice:%s' % equity] +
                    ['%s:0.0' % name for name in ('FxRate', 'CommodityPrice', 'FuturesPrice')])


REFUSALS = [
    ({'Beta_Prior_Defaults': _class_table(-22)}, 'Residual_Skew_Share_Defaults'),
    ({'Beta_Prior_Sd': 10.0}, 'Residual_Skew_Share_Sd'),
    ({'Stickiness_Prior': 0.3}, 'RESERVE LINE'),
    ({'Vanilla_Guard': 'Yes'}, 'no longer declares'),
    ({'Forward_Smile_Source': 'Prior'}, 'WITHDRAWN'),
    ({'Residual_Skew_Share_Defaults': _class_table(-0.9)}, 'conditioning bound'),
    ({'Sigma_S_Reference': 0.0}, 'Sigma_S_Reference'),
    ({'Leverage_Prior': -0.7, 'Leverage_Prior_SE': 0.0}, 'Leverage_Prior_SE'),
]


@pytest.mark.parametrize('declared,names', REFUSALS)
def test_a_withdrawn_declaration_refuses_by_name(declared, names):
    """A retired or unusable field REFUSES, naming itself and what replaced it - never carried past
    `declared_defaults` and read as a default nobody wrote.

    The eight rows are lane P2's and lane L2's rulings: the two beta priors that moved onto the
    SHARE beta/alpha, the two halves of the withdrawn forward view, `Forward_Smile_Source: Prior`
    itself, a skew share outside the conditioning bound the factor asserts at load, a
    `Sigma_S_Reference` that cannot scale the product row, and a standard error that divides a
    prior row by nothing.
    """
    assert names in _refused(_ladder(**declared), 'refuse')


def test_two_class_tables_disagreeing_in_sign_refuse():
    """`Leverage_Product_Defaults` is the row and `Leverage_Prior_Defaults` signs the seed, so a
    class whose two disagree would seed a fit AGAINST its own prior. Refused at construction, in
    the `Bootstrapper Configuration` block that declares them, before a quote is read.
    """
    assert 'disagree in SIGN' in _refused(
        dict(_ladder(), **{'Bootstrapper Configuration': {'LogVar2FJModelParameters': {
            'Prices': 'LogVar2FJModel', 'Leverage_Prior_Defaults': _class_table(-0.7),
            'Leverage_Product_Defaults': _class_table(1.9)}}}), 'sign')


# ------------------------------------------------------------------------------------------
# 2  THE NIG LIMIT, and 4's Refuse/Floor pair - one flat ladder fitted both ways
# ------------------------------------------------------------------------------------------
#: The flat ladder is small on purpose - the ATM pillar is solved EXACTLY at every iterate, so what
#: gate 2 reads does not depend on the outer search. `Log_Vol_Sd_Band` is DECLARED narrow because
#: this fit's stationary sd is 0.508 and the shipped 0.4-0.9 band would not bite: what the guard
#: gate is about is the Refuse/Floor branch, not where two iterations happen to land.
FLAT = dict(rungs=2, Paths=1024, Max_Iterations=2, Log_Vol_Sd_Band='0.70,0.90')


@pytest.fixture(scope='module')
def flat_fit():
    """The world's ladder re-quoted FLAT at 20%, fitted under `Stationary_Spread: Floor`."""
    return _fit(_ladder(flat=True, Stationary_Spread='Floor', **FLAT), 'flat')


def test_a_flat_surface_is_reproduced_by_the_residual(flat_fit):
    """The NIG limit: a residual on a flat 20% ladder has to give the ATM back.

    The ATM pillar is solved in PREMIUM, so its miss converts at the quote's own `P/vega`, which
    for an at-the-forward option IS sigma - a relative premium miss of `e` is `e x 0.20` vol
    points. This ladder's ATM rungs come back at 1e-15 to 1e-11 relative, which is 1e-12 vol
    points.

    HELD AT 1e-5 VOL POINTS AND NOT THE 1e-4 LANE 1 MEASURED, because `Atm_Miss_Max` (1e-4
    RELATIVE, so 2e-5 vol points) already refuses the fit above that: a 1e-4 gate here could not
    fail without the engine having refused first, which is a gate that cannot go red.
    """
    misses = _report_floats(flat_fit[1], 'ATM misses')
    assert len(misses) == FLAT['rungs'], misses
    assert max(abs(x) for x in misses) * XI ** 0.5 <= 1e-5, misses


def test_the_stationary_guard_floors_a_flat_surface_and_says_so(flat_fit):
    """A stationary log-vol sd outside the band VIX options imply is a surface wanting vol dynamics
    that market does not price. `Floor` TAKES the fit and names the bucket in the report."""
    assert 'FLOORED' in flat_fit[1] and 'stationary log-vol sd' in flat_fit[1]


def test_the_stationary_guard_refuses_the_same_surface_by_name():
    """The same document at `Stationary_Spread: Refuse` - the declared default - refuses instead,
    naming the bucket, its sd and the pair that produced it. Nothing is scaled either way: the
    guard is a READING of theta*."""
    message = _refused(_ladder(flat=True, Stationary_Spread='Refuse', **FLAT), 'flat_refuse')
    assert 'stationary log-vol sd' in message, message[:400]
    assert 'Set Stationary_Spread to Floor' in message, message[:400]


# ------------------------------------------------------------------------------------------
# 6  `Model_Priors: Off` IS THE VANILLA-ONLY OBJECTIVE
# ------------------------------------------------------------------------------------------
PRIORS_OFF = dict(rungs=2, Model_Priors='Off', Paths=2048, Max_Iterations=2,
                  Stationary_Spread='Floor')


def _floats_of(factor):
    """Every float a written factor holds, keyed by its own path - the scalars and each curve's
    knots and values, in hex, which is the only comparison that catches a one-ulp move."""
    out = {}
    for name, value in sorted(factor.items()):
        array = getattr(value, 'array', None)
        if array is not None:
            out.update({'%s[%d][%d]' % (name, i, j): float(x).hex()
                        for i, row in enumerate(array.tolist())
                        for j, x in enumerate(row)})
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            out[name] = float(value).hex()
    return out


def test_a_priors_off_fit_is_bit_identical_to_its_banked_reading():
    """`Model_Priors: Off` is the objective and the box the vanilla-only fit had - no leverage row,
    no shape floor, no class prior - and it is the switch every LogVar2FJ landing carries its
    bit-identity gate under.

    The banked reading is `tests/fixtures/logvar2fj_priors_off.json`: 20 floats of the written
    factor at `Paths` 2048 under `Pseudo`, hex for hex. A landing that moves this without moving
    the switch has changed the vanilla objective, whatever else it says it changed.

    BANKED ON AN RTX 3090: the DRAW is the seed's and not the silicon's, but the reductions are the
    device's, so a CPU-only box re-banks rather than reading a defect.
    """
    factor, _ = _fit(_ladder(**PRIORS_OFF), 'priors_off')
    with open(PRIORS_OFF_FILE) as handle:
        banked = json.load(handle)
    read = _floats_of(factor)
    assert read == banked, {k: (v, banked.get(k)) for k, v in read.items()
                            if banked.get(k) != v}


# ------------------------------------------------------------------------------------------
# 1 (CVA) and 9  THE CREDIT GATES
# ------------------------------------------------------------------------------------------
#: 1d start, then out past the deal: `GBMAssetPriceTSModelImplied` builds its per-step vol as
#: sqrt(V(t_k) - V(t_{k-1})), so a t0 row makes the first step zero-length.
GRID = '1d 3m(3m) 2y'
OUTER_GBM = {'EquityPrice': 'GBMAssetPriceTSModelImplied', 'FxRate': 'GBMAssetPriceModel'}
OUTER_LV = {'EquityPrice': 'LogVar2FJImpliedSpotModel', 'FxRate': 'GBMAssetPriceModel'}


def _credit(paths=4096, scenarios='No'):
    return {'Object': 'CreditMonteCarlo', 'Base_Date': {'.Timestamp': BASE}, 'Currency': 'USD',
            'Time_grid': GRID, 'Batch_Size': paths, 'Simulation_Batches': 1, 'Random_Seed': 1,
            'MCMC_Simulations': 1, 'Deflation_Interest_Rate': 'USD', 'Calc_Scenarios': scenarios,
            'Credit_Valuation_Adjustment': {
                'Calculate': 'Yes', 'Counterparty': 'CPTY', 'Deflate_Stochastically': 'No',
                'Stochastic_Hazard_Rates': 'No', 'Gradient': 'No', 'Hessian': 'No'}}


def _netted(deal):
    """The autocall under an uncollateralised netting set: a bare deal reports only its OWN reval
    dates, while a set reports every mtm date it spans, which is what gives the profile its rows."""
    return [{'Instrument': {'.Deal': {
        'Object': 'NettingCollateralSet', 'Reference': 'NS1', 'Netted': 'True',
        'Collateralized': 'False'}},
        'Children': [{'Instrument': {'.Deal': deal}}]}]


def _outer(models, spot_model='LogVar2FJ', factor=None):
    """A CREDIT document's sections: the scenario generator each factor type is simulated by, the
    implied GBM factor that generator reads, and the priced deal's own switch and factor."""
    sections = _priced(spot_model, factor)
    sections['Model Configuration'] = {'.ModelParams': {
        'modeldefaults': models, 'modelfilters': {}}}
    sections['Price Factors']['GBMAssetPriceTSModelParameters.INDEX_A'] = {
        'Quanto_FX_Volatility': None, 'Quanto_FX_Correlation': 0.0,
        'Vol': _curve([[0.0027397, XI ** 0.5], [5.0, XI ** 0.5]])}
    return sections


def test_the_gbm_limit_holds_through_the_cva():
    """The limit again, on the CVA rather than the mark: the SAME scenario walk under the GBM
    term-structure outer, the pricer switched from GBM to the LogVar2FJ kit at its own limit.

    HOLDS one float32 ulp on `cva`, which is what the exposure is accumulated in. The book runs on
    this configuration - the GBM outer with the LogVar2FJ pricer - so the gate is on the pair the
    book uses and not on a research one.

    ITS RESOLUTION, measured: `cva` 0.0045964 and its float32 ulp 4.7e-10, which catches a
    residual-clock error of 1e-4 relative and NOT one of 1e-6 - where the mark half above catches
    1e-6 at 1.6e-6 relative. A harder CVA gate wants the exposure profile row by row.
    """
    both = [float(_run(_job(_credit(), _netted(_autocall()), **_outer(OUTER_GBM, spot_model)),
                       'cva_' + spot_model)['Results']['cva'])
            for spot_model in ('LogVar2FJ', 'None')]
    assert abs(both[0]) > 1e-6, ('the CVA is zero, so the gate compares nothing', both)
    assert abs(both[0] - both[1]) <= np.spacing(np.float32(both[1])), both


def test_a_logvar2fj_equity_reads_its_correlation_under_the_innovation_key():
    """A correlation is a correlation OF AN INNOVATION, so the LogVar2FJ outer answers the same
    `LognormalDiffusionProcess` row the implied GBM does - not a key named after its own process.

    The world declares the pair TWICE: +0.30 under `LognormalDiffusionProcess`, which every
    process walking a Gaussian answers, and -0.90 under `LogVar2FJSpotProcess`, which nothing
    answers. The realised scenario correlation is POSITIVE, so the row that was read is the
    innovation's; the invented key is inert, which is the reading a document keyed on it gets -
    a silent zero.

    The realised share is DILUTED: the return also carries the leverage and the mixer's own
    spread, so a declared rho realises `rho D_i D_j`. `get_cholesky_decomp` states `D` at INFO -
    0.4525 on this grid, the GBM sibling's being 1 - and the gate holds the realised correlation
    to that product within three standard errors. MEASURED: 0.1379 against 0.3 x 0.4525 = 0.1358
    over 49,152 intervals, 0.5 se; the invented key would put it at -0.41.
    """
    buf, root = io.StringIO(), logging.getLogger()
    handler, level = logging.StreamHandler(buf), root.level
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    try:
        out = _run(_job(_credit(scenarios='All'), _netted(_autocall()),
                        **_outer(OUTER_LV, factor=LIVE_NIG)), 'corr')
    finally:
        root.removeHandler(handler)
        root.setLevel(level)

    line = [ln for ln in buf.getvalue().splitlines()
            if 'realises' in ln and 'declared correlation' in ln]
    assert line, 'the dilution was not reported at INFO'
    dilution = float(re.search(r'realises ([\d.]+) of', line[-1]).group(1))
    assert 0.0 < dilution < 1.0, dilution

    scenarios = out['Results']['scenarios']
    equity = np.log(scenarios['EquityPrice.INDEX_A'].values)
    rate = np.log(scenarios['FxRate.EUR'].values)
    a, b = np.diff(equity, axis=1).ravel(), np.diff(rate, axis=1).ravel()
    realised = float(np.corrcoef(a, b)[0, 1])
    assert realised > 0.0, ('the invented key was read', realised)
    assert abs(realised - 0.3 * dilution) <= 3.0 / np.sqrt(a.size), (realised, dilution)


# ------------------------------------------------------------------------------------------
# 10  THE HEX SET
#
# The documents a LogVar2FJ landing must not move: the repo's own GBM Monte Carlo fixtures, valued
# and with their first-order frames, every reported float pinned in hex. The standing pack script
# runs the same list; this makes it red under `pytest` rather than only under a pack.
# ------------------------------------------------------------------------------------------
HEX_DOCUMENTS = [
    ('autocall_gbm', 'autocall_job.json', None),
    ('autocall_gbm_first', 'autocall_job.json', {'Greeks': 'First'}),
    ('tarf_gbm', 'fx_tarf_job.json', None),
    ('tarf_gbm_first', 'fx_tarf_job.json', {'Greeks': 'First'}),
    ('accumulator_gbm', 'fx_accumulator_job.json', None),
    ('accumulator_gbm_first', 'fx_accumulator_job.json', {'Greeks': 'First'}),
]
# NOT `hw2f_four_quote_job.json`, which the pack's list carries: its Hull-White fit costs 170 s on
# this card against 0.1-0.7 s for every row above, and the solver it shares with this family is
# already exercised by the fits here. It belongs in a campaign run.


def _document_hex(name, overrides=None):
    """Every float a banked document reports, keyed by its own frame, row and column."""
    cx = rf.Context()
    cx.load_json(os.path.join(HERE, 'fixtures', name))
    out = cx.run_job(overrides=overrides)[1]['Results']
    return {'%s|%s|%s' % (frame_name, index, column): float(value).hex()
            for frame_name, frame in out.items() if hasattr(frame, 'columns')
            for index, row in zip(frame.index, frame.values)
            for column, value in zip(frame.columns, row)
            if isinstance(value, (int, float, np.floating))
            and not isinstance(value, bool) and np.isfinite(value)}


@pytest.mark.parametrize('name,document,overrides', HEX_DOCUMENTS)
def test_a_banked_document_is_unmoved(name, document, overrides):
    """The repo's own GBM Monte Carlo documents, float for float against `logvar2fj_hex.json`.

    A LogVar2FJ landing shares `oss_uniforms`, `sim_spot_oss`, the three OSS pricers and
    `LeastSquaresSolve` with every one of these, so a change that reaches them moves a number HERE
    and nowhere in the LogVar2FJ gates above. The three `Greeks: First` rows are the reason the
    valuations alone are not enough: a wrong allocation telescopes to the right total, and the
    sensitivity frame is where it does not.

    BANKED ON AN RTX 3090, which is where a base valuation runs whenever CUDA is visible. The hex
    is the DEVICE's as much as the tree's; a CPU-only box re-banks rather than reading a defect.
    """
    with open(HEX) as handle:
        banked = json.load(handle)[name]
    read = _document_hex(document, overrides)
    assert read and read == banked, {k: (v, banked.get(k)) for k, v in read.items()
                                     if banked.get(k) != v}
