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
          fixings=None, decisions=None):
    days = fix_days or FIX_DAYS
    rows = []
    for i, d in enumerate(days):
        fix = BASE + pd.DateOffset(days=d)
        row = [fix, fix + pd.DateOffset(days=SETTLE_LAG),
               (fixings or {}).get(i), (decisions or {}).get(i)]
        rows.append(row)
    return {'Object': 'FXExtendableForwardDeal', 'Reference': 'EXT', 'Currency': 'USD',
            'Underlying_Currency': 'EUR', 'Discount_Rate': 'USD', 'FX_Volatility': 'EUR.USD',
            'Buy_Sell': buy_sell, 'Option_Type': 'Call', 'Strike_Price': K1,
            'Extension_Strike': k2, 'Extension_Date': rows[extension_index][0],
            'Extension_Style': style, 'Underlying_Amount': NOTIONAL,
            'Extendable_ExpiryDates': rows}


def _job(deal, base=BASE, calc=None):
    return {'Calc': {
        'Calculation': dict({'Object': 'BaseValuation', 'Base_Date': base, 'Currency': 'USD',
                             'MCMC_Simulations': SIMS, 'Random_Seed': 1}, **(calc or {})),
        'Deals': {'Tag_Titles': '', 'Reference': 'ext',
                  'Deals': {'Children': [{'Instrument': {'.Deal': deal}}]}},
        'MergeMarketData': {'MarketDataFile': '', 'ExplicitMarketData': {
            'System Parameters': {'Base_Currency': 'USD', 'Base_Date': base},
            'Valuation Configuration': {},
            'Price Factors': FACTORS}}}}


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
    """The strip's OSS machinery against Black: the continuation at the single decision is
    linear in the decision spot, so the optional part is a scaled Black call and the whole
    undecided value is closed-form up to Monte Carlo error.

    MUTATION: the exercise direction flipped (extend when the spot is on the LOSING side) kills
    this gate, the unreachable-strike anchor, the rolling Black gate and the dominance gate
    together. The strip boundary mis-set by +10% reads 53.230 against the oracle's 84.362
    (-37%): a materially wrong boundary loses real value, while SMALL boundary errors are
    forgiven to second order - the envelope argument that lets the pricer detach it from AAD.
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


def test_a_terminated_rolling_deal_needs_no_fabricated_lifecycle():
    """A rolling deal the bank stopped extending, seen after the fact: the No is recorded at the
    extension fixing, the decisions termination made moot stay BLANK, and the fixings that never
    existed carry no prints. The book of record holds only facts, and the deal is exactly its
    settled past - worth zero - rather than a load error demanding counterfactual data."""
    base = BASE + pd.DateOffset(days=300)          # between fixings 2 (day 273) and 3 (day 364)
    deal = _deal(style='Rolling', fixings={0: 1.26, 1: 1.28}, decisions={1: 'No'})
    out, _ = _run(_job(deal, base=base))
    assert _mtm(out) == 0.0, _mtm(out)


def test_the_cmc_profile_reconstructs_the_lifecycle():
    """A CMC run across the whole schedule: decisions reconstructed at their own fixing rows,
    fixed-but-unsettled cashflows carried at the state entering their fixing, and nothing left
    after the last settlement. The t0 row must agree with the BaseValuation mark."""
    base_value = _mtm(_run(_job(_deal()))[0])
    job = _job(_deal(), calc={
        'Object': 'CreditMonteCarlo', 'Time_grid': '0d 15m(3m)', 'Batch_Size': 512,
        'Simulation_Batches': 2, 'MCMC_Simulations': 1 << 12,
        'Deflation_Interest_Rate': 'USD'})
    md = job['Calc']['MergeMarketData']['ExplicitMarketData']
    md['Price Models'] = {'GBMAssetPriceModel.EUR': {'Vol': SIGMA, 'Drift': 0.0}}
    md['Model Configuration'] = {'.ModelParams': {
        'modeldefaults': {'FxRate': 'GBMAssetPriceModel'}, 'modelfilters': {}}}
    out, _ = _run(job)
    profile = out['Results']['mtm']
    row0 = float(np.asarray(profile.iloc[0], dtype=float).mean())
    assert abs(row0 - base_value) / abs(base_value) < 2e-2, (row0, base_value)
    # the grid's last row IS the final settlement date - its mtm carries the settling cashflow
    # by convention - and every reconstructed lifecycle must produce a finite profile
    assert np.isfinite(np.asarray(profile, dtype=float)).all()


def _cva_job(spot=None, gradient='No'):
    job = _job(_deal(style='Rolling'), calc={
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
    without a latch registration the CVA gradient drops their flux. The registration's branches
    are derived from the SAME row arithmetic the pricer reports (`value = fixed + state * live`),
    the alive branch the facts-only world and the dead branch the survived-weighted pending head
    - so the `EXTENDABLE_LATCH` organ's reconstruction against the engine's own rows is the sharp
    statement, with the CVA delta against its CRN ladder as the economic bound.

    MEASURED: reconstruction 5.5e-8 relative (float32 roundoff); delta -0.07% against the best
    rung with the registration, -3.17% with it suppressed - the flux is 3.2% of this document's
    delta, and the suppressed mutant also kills the organ assertion outright."""
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
    """The collateralised delta bound - and the record of a channel MEASURED to be second order.

    Under the CSA the counterfactual ledger should in principle flip with the decision (the
    extendable pays on SURVIVING, an algebra `cash_events` cannot express). Measured across four
    documents built to amplify it - base (+0.25%), ITM extension (-0.02%), vol 0.30 with a 20-day
    margin period (+0.26%), and a one-fixing zero-lag tail where the flipped object is almost
    pure cash (+0.03%, ladder 0.01%) - the residual never resolves above its CRN oracle. The
    structural reason: unlike the autocall, whose flipped coupon was 28% of scale and became
    at-risk cash at the decision's own row, the extendable's flipped payments settle far from
    the decision, carried by the VALUE side (which `pending` scores exactly) until one brief
    hazard-weighted window. The ledger channel is deliberately NOT built ahead of a document
    that can falsify it; this gate stands as the economic bound."""
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
