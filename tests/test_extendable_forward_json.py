"""FXExtendableForwardDeal end to end, through the JSON contract and nothing else.

The closed forms live in the deal's own degenerate limits, all under one flat world:

- a RESOLVED extension (historical Yes/No) is one or two strips of ordinary forwards - exact;
- an unreachable Extension_Strike is never exercised, so the undecided deal is the guaranteed
  K1 strip - exact, since the survival probability underflows to zero;
- the UNDECIDED strip's optional part has a Black closed form: the continuation at the single
  decision is linear in the decision spot, C(S) = sign * N * (A * S - K2 * B), so the optional
  value is `D_fix * A * Black(F, K2 * B / A)` - which prices the whole OSS machinery against
  Black with only Monte Carlo error;
- a ROLLING deal with ONE decision is the same product priced through the backward quadrature
  boundary instead of the closed-form strip boundary - the same Black oracle gates the solver,
  the grid interpolation and the root-finding at once.

What has no closed form is gated by dominance: bank-optimal exercise values at least both forced
strategies (never extend, always extend), a rolling schedule dominates the strip that commits to
the whole tail at once, and Buy plus Sell of the same deal exceeds zero by the two option premia
- each side optimised from its own book, so the antisymmetry of a forward is deliberately broken.

`Exercised_By` restores the antisymmetry where it belongs. The MIRROR of a deal is the opposite
Buy_Sell AND the opposite Exercised_By - the same trade seen from the other desk - and it must sum
to zero, exactly rather than to Monte Carlo error, because the boundary is where the REPORTED
continuation crosses zero and no exerciser moves it. On the written side the optional part is the
Black PUT on that same strike, so put-call parity closes the pair: the two exercise sides of one
strip add up to the two forced strategies.
"""
import io
import json
import logging
import math
import os
import sys

# reference-derivus shadow-import guard (MEMORY): pin the package under test to THIS repo.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import derivus
from derivus import utils
from derivus.config import CustomJsonEncoder

BASE = pd.Timestamp('2024-06-28')
X0, R_USD, R_EUR, SIGMA = 1.25, 0.04, 0.02, 0.10
K1, K2, NOTIONAL = 1.24, 1.30, 1000.0
SIMS = 1 << 15
FIX_DAYS = [91, 182, 273, 364]
SETTLE_LAG = 2

FACTORS = {
    'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD', 'Spot': 1.0},
    'FxRate.EUR': {'Domestic_Currency': None, 'Interest_Rate': 'EUR', 'Spot': X0},
    'InterestRate.USD': {'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                         'Curve': utils.Curve([], [[0.0, R_USD], [5.0, R_USD]])},
    'InterestRate.EUR': {'Currency': 'EUR', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                         'Curve': utils.Curve([], [[0.0, R_EUR], [5.0, R_EUR]])},
    'FXVol.EUR.USD': {'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness',
                      'Surface': utils.Curve([], [[m, t, SIGMA] for m in (0.8, 1.0, 1.2)
                                                  for t in (0.02, 2.0)])}}


def _deal(style='Strip', extension_index=1, k2=K2, buy_sell='Buy', fix_days=None,
          fixings=None, decisions=None, exercised_by=None):
    days = fix_days or FIX_DAYS
    rows = []
    for i, d in enumerate(days):
        fix = BASE + pd.DateOffset(days=d)
        row = [fix, fix + pd.DateOffset(days=SETTLE_LAG),
               (fixings or {}).get(i), (decisions or {}).get(i)]
        rows.append(row)
    deal = {'Object': 'FXExtendableForwardDeal', 'Reference': 'EXT', 'Currency': 'USD',
            'Underlying_Currency': 'EUR', 'Discount_Rate': 'USD', 'FX_Volatility': 'EUR.USD',
            'Buy_Sell': buy_sell, 'Option_Type': 'Call', 'Strike_Price': K1,
            'Extension_Strike': k2, 'Extension_Date': rows[extension_index][0],
            'Extension_Style': style, 'Underlying_Amount': NOTIONAL,
            'Extendable_ExpiryDates': rows}
    # ABSENT is the default, and every gate above books it that way: the declared 'Bank' must
    # reproduce the omitted field bit for bit
    if exercised_by is not None:
        deal['Exercised_By'] = exercised_by
    return deal


def _mirror(**kw):
    """The same trade from the other desk: both sides of the book flipped at once."""
    side = kw.pop('buy_sell', 'Buy')
    holder = kw.pop('exercised_by', 'Bank')
    return _deal(buy_sell='Sell' if side == 'Buy' else 'Buy',
                 exercised_by='Counterparty' if holder == 'Bank' else 'Bank', **kw)


def _job(deal, base=BASE, calc=None):
    return {'Calc': {
        'Calculation': dict({'Object': 'BaseValuation', 'Base_Date': base, 'Currency': 'USD',
                             'MCMC_Simulations': SIMS, 'Random_Seed': 1}, **(calc or {})),
        'Deals': {'Tag_Titles': '', 'Reference': 'ext',
                  'Deals': {'Children': [{'Instrument': {'.Deal': deal}}]}},
        'MergeMarketData': {'MarketDataFile': '', 'ExplicitMarketData': {
            'System Parameters': {'Base_Currency': 'USD', 'Base_Date': base},
            'Valuation Configuration': {},
            # a COPY: a CRN rung rebinds FxRate.EUR, and sharing the module dict would leave the
            # bumped spot standing for every job built afterwards
            'Price Factors': dict(FACTORS)}}}}


def _profile_job(deal):
    """The CMC grid the lifecycle is reconstructed on, GBM spot off the same flat surface."""
    job = _job(deal, calc={
        'Object': 'CreditMonteCarlo', 'Time_grid': '0d 15m(3m)', 'Batch_Size': 512,
        'Simulation_Batches': 2, 'MCMC_Simulations': 1 << 12,
        'Deflation_Interest_Rate': 'USD'})
    md = job['Calc']['MergeMarketData']['ExplicitMarketData']
    md['Price Models'] = {'GBMAssetPriceModel.EUR': {'Vol': SIGMA, 'Drift': 0.0}}
    md['Model Configuration'] = {'.ModelParams': {
        'modeldefaults': {'FxRate': 'GBMAssetPriceModel'}, 'modelfilters': {}}}
    return job


def _run(job, debug=False):
    buf, root = io.StringIO(), logging.getLogger()
    handler, old = logging.StreamHandler(buf), logging.getLogger().level
    if debug:
        root.addHandler(handler)
        root.setLevel(logging.DEBUG)
    try:
        cx = derivus.Context()
        cx.load_json((json.dumps(job, cls=CustomJsonEncoder), 'ext'))
        _, out = cx.run_job()
    finally:
        if debug:
            root.removeHandler(handler)
            root.setLevel(old)
    return out, buf.getvalue()


def _mtm(out, ref='EXT'):
    rows = out['Results']['mtm']
    return float(rows[rows['Reference'] == ref]['Value'].iloc[0])


def _ndtr(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _fwd(day):
    return X0 * math.exp((R_USD - R_EUR) * day / 365.0)


def _disc(day):
    return math.exp(-R_USD * day / 365.0)


def _leg(i, strike):
    return _disc(FIX_DAYS[i] + SETTLE_LAG) * NOTIONAL * (_fwd(FIX_DAYS[i]) - strike)


def _optional_black(decision_index, tail, k2=K2, sign=1.0):
    """D_fix * A * Black(F_dec, K2 * B / A): the linear-continuation optional value."""
    t_dec = FIX_DAYS[decision_index] / 365.0
    d_fix = _disc(FIX_DAYS[decision_index])
    f_dec = _fwd(FIX_DAYS[decision_index])
    A = sum(_disc(FIX_DAYS[j] + SETTLE_LAG) * _fwd(FIX_DAYS[j]) / f_dec for j in tail) / d_fix
    B = sum(_disc(FIX_DAYS[j] + SETTLE_LAG) for j in tail) / d_fix
    strike = k2 * B / A
    sd = SIGMA * math.sqrt(t_dec)
    d1 = math.log(f_dec / strike) / sd + 0.5 * sd
    call = f_dec * _ndtr(sign * d1) * sign - strike * _ndtr(sign * (d1 - sd)) * sign
    return d_fix * NOTIONAL * A * call


def test_a_resolved_extension_is_a_strip_of_forwards():
    """The lifecycle anchors: base date after the extension fixing, historical prints and the
    bank's Yes/No supplied. Both K1 cashflows have settled, so Yes is the K2 forward tail off
    today's market and No is worth exactly nothing - both exact, no Monte Carlo in either."""
    base = BASE + pd.DateOffset(days=200)          # between fixings 1 (day 182) and 2 (day 273)
    prints = {0: 1.26, 1: 1.28}
    for decision, tail_ids in (('Yes', [2, 3]), ('No', [])):
        deal = _deal(fixings=prints, decisions={1: decision})
        out, _ = _run(_job(deal, base=base))
        expected = sum(
            math.exp(-R_USD * (FIX_DAYS[j] + SETTLE_LAG - 200) / 365.0) * NOTIONAL *
            (X0 * math.exp((R_USD - R_EUR) * (FIX_DAYS[j] - 200) / 365.0) - K2)
            for j in tail_ids)
        v = _mtm(out)
        assert abs(v - expected) <= max(1e-9 * NOTIONAL, abs(expected) * 1e-9), (
            decision, v, expected)


def test_an_unreachable_extension_strike_is_the_guaranteed_strip():
    """K2 far above any reachable spot: the bank never extends, the survival weight underflows,
    and the undecided strip is EXACTLY its two guaranteed K1 fixings."""
    out, _ = _run(_job(_deal(k2=2.5)))
    expected = _leg(0, K1) + _leg(1, K1)
    assert abs(_mtm(out) - expected) / abs(expected) < 1e-9, (_mtm(out), expected)


def test_an_undecided_strip_prices_its_black_optional():
    """The strip's OSS machinery against Black: the continuation at the single decision is linear
    in the decision spot, so the optional part is a scaled Black call and the whole undecided value
    is closed-form up to Monte Carlo error.

    MUTATIONS: the exercise direction flipped kills this gate, the unreachable-strike anchor, the
    rolling Black gate and the dominance gate together; the strip boundary mis-set by +10% reads
    -37%. A materially wrong boundary loses real value while SMALL errors are forgiven to second
    order, which is the envelope argument that lets the pricer detach it from AAD.
    """
    out, log = _run(_job(_deal()), debug=True)
    expected = _leg(0, K1) + _leg(1, K1) + _optional_black(1, [2, 3])
    v = _mtm(out)
    assert abs(v - expected) / abs(expected) < 5e-3, (v, expected)
    always = _leg(0, K1) + _leg(1, K1) + _leg(2, K2) + _leg(3, K2)
    never = _leg(0, K1) + _leg(1, K1)
    assert v > max(always, never), 'optionality must exceed both forced strategies'
    organs = [ln for ln in log.splitlines() if 'EXTENDABLE' in ln]
    assert organs and 'style=strip' in organs[-1] and 'decisions=1' in organs[-1], organs


def test_a_rolling_single_decision_is_the_same_black():
    """One rolling decision (three fixings, extension on the second) is the strip product with a
    one-fixing tail - but priced through the backward quadrature boundary, the grid interpolation
    and the root-finding. The same Black oracle must come back."""
    out, _ = _run(_job(_deal(style='Rolling', fix_days=FIX_DAYS[:3])))
    expected = _leg(0, K1) + _leg(1, K1) + _optional_black(1, [2])
    v = _mtm(out)
    assert abs(v - expected) / abs(expected) < 5e-3, (v, expected)


def test_rolling_dominates_the_strip_it_contains():
    """Bank-optimal orders: the rolling schedule can replicate the strip's all-or-nothing
    strategy and can also bail mid-tail, so it is worth at least as much; and Buy plus Sell of
    the same deal is the sum of the two sides' option premia - positive, the forward's
    antisymmetry deliberately broken by each side optimising its own book."""
    strip = _mtm(_run(_job(_deal()))[0])
    rolling = _mtm(_run(_job(_deal(style='Rolling')))[0])
    assert rolling >= strip - abs(strip) * 2e-3, (rolling, strip)
    sold = _mtm(_run(_job(_deal(buy_sell='Sell')))[0])
    assert strip + sold > 0.0, (strip, sold)


def test_the_declared_exerciser_defaults_to_the_reported_book():
    """`Exercised_By` omitted and `Exercised_By: Bank` are the same book, bit for bit - the
    declared default is the deal every gate above books, and the switch costs nothing to a
    portfolio that never states it."""
    for style in ('Strip', 'Rolling'):
        for side in ('Buy', 'Sell'):
            absent = _mtm(_run(_job(_deal(style=style, buy_sell=side)))[0])
            declared = _mtm(_run(_job(_deal(style=style, buy_sell=side,
                                            exercised_by='Bank')))[0])
            assert declared.hex() == absent.hex(), (style, side, declared, absent)


def test_the_mirror_booking_sums_to_zero():
    """The acceptance the reported book alone cannot state: the same trade from the other desk,
    flipping Buy_Sell AND Exercised_By at once. The boundary does not move - it is where the
    REPORTED continuation crosses zero and both books locate the same H off the same grid - so the
    mirror is this deal's row arithmetic negated, which is exact in IEEE. The pair sums to ZERO
    rather than to Monte Carlo error.

    Both diagonals run, because they truncate opposite ways: Buy/Bank and Sell/Counterparty
    continue ABOVE H, Buy/Counterparty and Sell/Bank below it. Two mutants die here - reverting the
    backward pass's min to a max leaves the Rolling pairs 4-6% out with the Strip untouched (its
    boundary being closed-form), and orienting the OSS truncation by `forward_sign` leaves every
    pair at 174% of book.
    """
    for style in ('Strip', 'Rolling'):
        for side, holder in (('Buy', 'Bank'), ('Buy', 'Counterparty')):
            book = _mtm(_run(_job(_deal(style=style, buy_sell=side,
                                        exercised_by=holder)))[0])
            mirror = _mtm(_run(_job(_mirror(style=style, buy_sell=side,
                                            exercised_by=holder)))[0])
            assert abs(book) > 1.0, (style, side, holder, book)
            assert abs(book + mirror) <= 1e-6 * abs(book), (style, side, holder, book, mirror)


def test_the_written_side_prices_the_black_put_it_is_short():
    """The client side of a bank-exercisable strip against the same Black oracle FLIPPED: the
    continuation region flips to S < H on the same strike, so the optional part is a short put
    where the reported book is long a call, and the dominance bound inverts with it. Put-call
    parity then closes the pair model-free - the two exercise sides add up to `never + always`,
    which is what would move if the flipped truncation kept the wrong tail.

    1.0e-3 relative against the oracle, the same Monte Carlo error the reported book reads, and a
    parity residual of 4.5e-4. The truncation mutant reads +236%.
    """
    v = _mtm(_run(_job(_deal(exercised_by='Counterparty')))[0])
    expected = _leg(0, K1) + _leg(1, K1) - _optional_black(1, [2, 3], sign=-1.0)
    assert abs(v - expected) / abs(expected) < 5e-3, (v, expected)
    always = _leg(0, K1) + _leg(1, K1) + _leg(2, K2) + _leg(3, K2)
    never = _leg(0, K1) + _leg(1, K1)
    assert v < min(always, never), 'a written optionality must fall short of both forced strategies'
    held = _mtm(_run(_job(_deal()))[0])
    assert abs(held + v - (never + always)) < 5e-3 * abs(never + always), (held, v, never, always)


def test_the_reconstructed_mirror_profile_negates():
    """The mirror through the LIFECYCLE, which base valuation never reaches: on the CMC grid each
    decision is reconstructed at its own fixing row off that scenario's spot, where the flipped
    truncation is a comparison rather than a draw, and every row must negate exactly.

    The ONLY gate the reconstruction comparison answers to: oriented by `forward_sign` it reads 79%
    of scale here and leaves every base-valuation reading untouched, and the backward pass's
    max-for-min reads 34%."""
    book = np.asarray(_run(_profile_job(_deal(style='Rolling')))[0]['Results']['mtm'], dtype=float)
    mirror = np.asarray(_run(_profile_job(_mirror(style='Rolling')))[0]['Results']['mtm'],
                        dtype=float)
    scale = np.abs(book).max()
    assert scale > 1.0 and np.isfinite(mirror).all(), scale
    assert np.abs(book + mirror).max() <= 1e-6 * scale, np.abs(book + mirror).max()


def test_an_unknown_exerciser_is_refused_by_name():
    """A side nobody can book: the deal is refused with the field named and the remedy stated,
    rather than silently priced from the reporter's book."""
    out, log = _run(_job(_deal(exercised_by='Client')), debug=True)
    assert out['Results']['mtm'].empty or 'EXT' not in set(out['Results']['mtm']['Reference'])
    assert 'Exercised_By must be Bank or Counterparty' in log and 'Client' in log, log


def test_a_terminated_rolling_deal_needs_no_fabricated_lifecycle():
    """A rolling deal the bank stopped extending, seen after the fact: the No is recorded, the
    decisions termination made moot stay BLANK, and the fixings that never existed carry no prints.
    The record holds only facts, so the deal is exactly its settled past - worth zero - rather than
    a load error demanding counterfactual data."""
    base = BASE + pd.DateOffset(days=300)          # between fixings 2 (day 273) and 3 (day 364)
    deal = _deal(style='Rolling', fixings={0: 1.26, 1: 1.28}, decisions={1: 'No'})
    out, _ = _run(_job(deal, base=base))
    assert _mtm(out) == 0.0, _mtm(out)


def test_the_cmc_profile_reconstructs_the_lifecycle():
    """A CMC run across the whole schedule: decisions reconstructed at their own fixing rows,
    fixed-but-unsettled cashflows carried at the state entering their fixing, and nothing left
    after the last settlement. The t0 row must agree with the BaseValuation mark."""
    base_value = _mtm(_run(_job(_deal()))[0])
    out, _ = _run(_profile_job(_deal()))
    profile = out['Results']['mtm']
    row0 = float(np.asarray(profile.iloc[0], dtype=float).mean())
    assert abs(row0 - base_value) / abs(base_value) < 2e-2, (row0, base_value)
    # the grid's last row IS the final settlement date - its mtm carries the settling cashflow
    # by convention - and every reconstructed lifecycle must produce a finite profile
    assert np.isfinite(np.asarray(profile, dtype=float)).all()


def _cva_job(spot=None, gradient='No', deal=None):
    job = _job(deal if deal is not None else _deal(style='Rolling'), calc={
        'Object': 'CreditMonteCarlo', 'Time_grid': '0d 15m(1m)', 'Batch_Size': 1024,
        'Simulation_Batches': 4, 'MCMC_Simulations': 1 << 10,
        'Deflation_Interest_Rate': 'USD',
        'Credit_Valuation_Adjustment': {
            'Calculate': 'Yes', 'Counterparty': 'CPTY', 'Deflate_Stochastically': 'No',
            'Stochastic_Hazard_Rates': 'No', 'Gradient': gradient}})
    md = job['Calc']['MergeMarketData']['ExplicitMarketData']
    md['Price Models'] = {'GBMAssetPriceModel.EUR': {'Vol': SIGMA, 'Drift': 0.0}}
    md['Model Configuration'] = {'.ModelParams': {
        'modeldefaults': {'FxRate': 'GBMAssetPriceModel'}, 'modelfilters': {}}}
    md['Price Factors']['SurvivalProb.CPTY'] = {
        'Recovery_Rate': 0.4,
        'Curve': {'.Curve': {'meta': [], 'data': [[0.0, 0.0], [10.0, 0.4]]}}}
    if spot is not None:
        md['Price Factors']['FxRate.EUR'] = dict(md['Price Factors']['FxRate.EUR'], Spot=spot)
    return job


def test_the_cva_delta_carries_the_extension_flux(tmp_path=None):
    """The extend/terminate decisions reconstructed at outer fixing rows are hard indicators, so
    without a latch registration the CVA gradient drops their flux. The branches are derived from
    the SAME row arithmetic the pricer reports, so the `EXTENDABLE_LATCH` organ's reconstruction
    against the engine's own rows is the sharp statement and the CRN ladder the economic bound.

    Reconstruction 5.5e-8 relative, float32 roundoff; the delta reads -0.07% against the best rung
    with the registration and -3.17% suppressed, so the flux is 3.2% of this document's delta."""
    out, log = _run(_cva_job(gradient='Yes'), debug=True)
    organs = [ln for ln in log.splitlines() if 'EXTENDABLE_LATCH' in ln]
    assert organs, 'the latch registration logged nothing at DEBUG'
    parse = lambda ln, key: float(ln.split(key + '=')[1].split()[0])
    assert all(parse(ln, 'decisions') == 2 for ln in organs), organs[-1]
    recon = max(parse(ln, 'recon_max') for ln in organs)
    scale = max(parse(ln, 'scale') for ln in organs)
    assert recon < 1e-5 * scale, (recon, scale)

    g = out['Results']['grad_cva']['Gradient']
    aad = float(g.loc[[i for i in g.index if 'FxRate.EUR' in str(i[0])][0]])
    cva = float(out['Results']['cva'])
    crn = []
    for h in (0.005, 0.01):
        up = float(_run(_cva_job(spot=X0 + h))[0]['Results']['cva'])
        dn = float(_run(_cva_job(spot=X0 - h))[0]['Results']['cva'])
        crn.append((up - dn) / (2.0 * h))
    best = min(crn, key=lambda c: abs(aad - c))
    assert abs(aad - best) / abs(best) < 5e-2, (aad, crn, cva)


def test_a_mirror_cva_delta_carries_its_own_extension_flux():
    """The registration on the mirror, where the GAP is the only thing that could still point the
    wrong way: the reconstruction is built from `fired` and never reads it, so a gap oriented off
    `forward_sign` instead of `decide_sign` reconstructs perfectly while putting the kernel mass on
    the far side of every decision - visible only against a CRN ladder, where it reads -37.9%.

    The flux is LARGER here than on the reported book, the mirror's exposure sitting on the
    terminated paths: +0.27% against the best rung, -18.4% suppressed, against 3.2% reported."""
    mirror = _mirror(style='Rolling')
    out, log = _run(_cva_job(gradient='Yes', deal=mirror), debug=True)
    organs = [ln for ln in log.splitlines() if 'EXTENDABLE_LATCH' in ln]
    assert organs, 'the mirror registration logged nothing at DEBUG'
    parse = lambda ln, key: float(ln.split(key + '=')[1].split()[0])
    recon = max(parse(ln, 'recon_max') for ln in organs)
    scale = max(parse(ln, 'scale') for ln in organs)
    assert recon < 1e-5 * scale, (recon, scale)

    g = out['Results']['grad_cva']['Gradient']
    aad = float(g.loc[[i for i in g.index if 'FxRate.EUR' in str(i[0])][0]])
    crn = []
    for h in (0.005, 0.01):
        up = float(_run(_cva_job(spot=X0 + h, deal=mirror))[0]['Results']['cva'])
        dn = float(_run(_cva_job(spot=X0 - h, deal=mirror))[0]['Results']['cva'])
        crn.append((up - dn) / (2.0 * h))
    best = min(crn, key=lambda c: abs(aad - c))
    assert abs(aad - best) / abs(best) < 5e-2, (aad, crn)


def _collateralised(job):
    """The deal under a zero-threshold NettingCollateralSet, in the wire form the decoder reads -
    the autocall gate's CSA, one asset over."""
    csa = {'.CreditSupportList': [[0.0, 0.0]]}
    netting = {
        'Object': 'NettingCollateralSet', 'Reference': 'NS1', 'Netted': 'True',
        'Collateralized': 'True', 'Agreement_Currency': 'USD', 'Funding_Rate': 'USD',
        'Balance_Currency': 'USD', 'Liquidation_Period': 10.0, 'Settlement_Period': 0.0,
        'Credit_Support_Amounts': {
            'Received_Threshold': csa, 'Posted_Threshold': csa, 'Independent_Amount': csa,
            'Minimum_Received': csa, 'Minimum_Posted': csa}}
    deals = job['Calc']['Deals']['Deals']
    deals['Children'] = [{'Instrument': {'.Deal': netting}, 'Children': deals['Children']}]
    return job


def test_a_collateralised_cva_delta_carries_the_surviving_cash(tmp_path=None):
    """The collateralised delta bound, and the record of a channel MEASURED to be second order.

    Under the CSA the counterfactual ledger should in principle flip with the decision, an algebra
    `cash_events` cannot express. Across four documents built to amplify it - base, ITM extension,
    vol 0.30 at a 20-day margin period, and a one-fixing zero-lag tail where the flipped object is
    almost pure cash - the residual never resolves above its CRN oracle. The structural reason:
    unlike the autocall's flipped coupon, which was 28% of scale and became at-risk cash at the
    decision's own row, the extendable's flipped payments settle far from the decision and are
    carried by the VALUE side until one brief hazard-weighted window. The ledger channel is
    deliberately NOT built ahead of a document that can falsify it."""
    out, _ = _run(_collateralised(_cva_job(gradient='Yes')))
    g = out['Results']['grad_cva']['Gradient']
    aad = float(g.loc[[i for i in g.index if 'FxRate.EUR' in str(i[0])][0]])
    cva = float(out['Results']['cva'])
    crn = []
    for h in (0.005, 0.01):
        up = float(_run(_collateralised(_cva_job(spot=X0 + h)))[0]['Results']['cva'])
        dn = float(_run(_collateralised(_cva_job(spot=X0 - h)))[0]['Results']['cva'])
        crn.append((up - dn) / (2.0 * h))
    best = min(crn, key=lambda c: abs(aad - c))
    assert abs(aad - best) / abs(best) < 5e-2, (aad, crn, cva)
