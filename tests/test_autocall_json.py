"""QEDI_CustomAutoCallSwap end to end, through the JSON contract and nothing else.

Same form as the accumulator and TARF files: a job document run through `Context.load_json` +
`run_job`, an answer decided BEFORE the run from a closed form, and the DEBUG line the pricer
emits about what it decided.

THE CLOSED FORM. An autocall with a SINGLE coupon date is a cash-or-nothing DIGITAL: it pays the
coupon exactly when the spot finishes at or above the autocall threshold, so

    PV = Units * coupon * D(T) * N(d2),   d2 = (ln(S/K) + (r - q - sigma^2/2) T) / (sigma sqrt T)

with `K = threshold * strike`. That is exact under the GBM the document declares, so the gate is a
value assertion rather than a sanity check - the same reason `test_hn_oss_pricers.py` uses a
one-coupon autocall to pin the Heston-Nandi read.

Pre-registered from the document below (spot 100, strike 100, r 4%, q 1%, sigma 25%, 1y coupon at
threshold 1.00, coupon 0.08, Units 10):

    single-coupon digital   PV = 5.5972   (N(d2) = 0.5822)

and the two degenerate limits that bracket it: a threshold no path reaches pays nothing, and a
threshold every path clears pays the coupon with certainty.
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

import derivus as rf

TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'fixtures', 'autocall_job.json')


def _template():
    with open(TEMPLATE) as f:
        return json.load(f)


def _deal_of(job):
    return job['Calc']['Deals']['Deals']['Children'][0]['Instrument']['.Deal']


_T = _template()
_D = _deal_of(_T)
_PF = _T['Calc']['MergeMarketData']['ExplicitMarketData']['Price Factors']

BASE = _T['Calc']['Calculation']['Base_Date']['.Timestamp']
SPOT = _PF['EquityPrice.EQ']['Spot']
STRIKE = _D['Strike_Price']
UNITS = _D['Units']
COUPON = _D['Autocall_Coupons'][0][1]
R_USD = _PF['InterestRate.USD']['Curve']['.Curve']['data'][0][1]
Q_EQ = _PF['DividendRate.EQ']['Curve']['.Curve']['data'][0][1]
SIGMA = _PF['VolatilityGrid.EQ']['Surface']['.Curve']['data'][0][2]
DAYS = 365.0


def _offset(stamp):
    import datetime
    return (datetime.date.fromisoformat(stamp) - datetime.date.fromisoformat(BASE)).days


HORIZON = _offset(_D['Autocall_Coupons'][0][0]['.Timestamp'])


def _ndtr(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _digital(threshold):
    """PV of the one-coupon autocall: a cash-or-nothing digital struck at threshold * strike."""
    t = HORIZON / DAYS
    k = threshold * STRIKE
    sd = SIGMA * math.sqrt(t)
    d2 = (math.log(SPOT / k) + (R_USD - Q_EQ - 0.5 * SIGMA ** 2) * t) / sd
    return UNITS * COUPON * math.exp(-R_USD * t) * _ndtr(d2)


EXPECTED_ATM = _digital(1.00)


def _job(threshold=1.00, greeks='No', **deal_overrides):
    """The canonical document, varied. The autocall threshold is the switch this file spans."""
    job = _template()
    job['Calc']['Calculation']['Greeks'] = greeks
    deal = _deal_of(job)
    for row in deal['Autocall_Thresholds']:
        row[1] = threshold
    deal.update(deal_overrides)
    return job


def _run(job, tmp_path, name='ac', debug=False):
    path = os.path.join(str(tmp_path), f'{name}.json')
    with open(path, 'w') as f:
        json.dump(job, f, default=str)
    buf, root = io.StringIO(), logging.getLogger()
    handler = logging.StreamHandler(buf)
    old = root.level
    if debug:
        root.addHandler(handler)
        root.setLevel(logging.DEBUG)
    try:
        cx = rf.Context()
        cx.load_json(path)
        _, out = cx.run_job()
    finally:
        if debug:
            root.removeHandler(handler)
            root.setLevel(old)
    return out, buf.getvalue()


def _mtm(out, ref='AC1'):
    rows = out['Results']['mtm']
    rows = rows[rows['Reference'] == ref]
    return float(rows['Value'].iloc[0])


def test_a_single_coupon_autocall_is_a_digital(tmp_path):
    """The exact statement: one coupon date makes the payoff cash-or-nothing, and Black prices it
    with no Monte Carlo error of its own."""
    out, _ = _run(_job(), tmp_path)
    v = _mtm(out)
    assert abs(v - EXPECTED_ATM) / EXPECTED_ATM < 1e-2, (v, EXPECTED_ATM)


def test_an_unreachable_threshold_pays_nothing(tmp_path):
    """No path clears a 10x threshold, so the coupon is never paid and the deal is worth ~0.

    This is the anti-placebo half of the digital gate above: it fixes the SCALE, so a pricer
    returning a constant could not satisfy both.
    """
    out, _ = _run(_job(threshold=10.0), tmp_path, 'far')
    assert abs(_mtm(out)) < 1e-6 * UNITS * COUPON, _mtm(out)


def test_a_threshold_every_path_clears_pays_the_coupon_with_certainty(tmp_path):
    """The other bracket: a threshold at a hundredth of spot is cleared almost surely, so the PV
    is the discounted coupon and nothing else."""
    out, _ = _run(_job(threshold=0.01), tmp_path, 'near')
    certain = UNITS * COUPON * math.exp(-R_USD * HORIZON / DAYS)
    assert abs(_mtm(out) - certain) / certain < 1e-3, (_mtm(out), certain)


def test_the_pricer_logs_what_it_decided(tmp_path):
    """The pre-registered log line. The value alone cannot say how many coupon dates the pricer
    thought it had, nor whether it took the averaging branch - both of which change the product."""
    _, log = _run(_job(), tmp_path, 'aclog', debug=True)
    lines = [ln for ln in log.splitlines() if 'AUTOCALL ' in ln and 'coupons=' in ln]
    assert lines, 'the autocall logged nothing at DEBUG'
    organ = lines[-1]
    assert 'coupons=1' in organ, organ
    assert 'averaging=0' in organ, organ        # one fixing per coupon: the OSS branch
    assert 'blocks=1' in organ, organ


# --------------------------------------------------------------------------------------------
# compo: the same digital on the CONVERTED spot
# --------------------------------------------------------------------------------------------
FX_EUR = 1.25       # USD per EUR: the USD->EUR cross the compo scales by is 1/FX_EUR
R_EUR = 0.02
FX_SIGMA = 0.15
CORR = 0.35         # authored on the SORTED pair EUR.USD; the USD->EUR deal reads -CORR


def _compo_job(threshold=1.00, corr=CORR):
    """The template as a COMPO deal: EUR payoff on the USD asset, monitored on S*X.

    The strike is a PAYOFF-currency quantity, so it is authored at the same moneyness in EUR:
    `STRIKE / FX_EUR` against a compo spot of `SPOT / FX_EUR`.
    """
    job = _template()
    deal = _deal_of(job)
    deal['Payoff_Currency'] = 'EUR'
    deal['Payoff_Type'] = 'Compo'
    deal['Discount_Rate'] = 'EUR'
    deal['Strike_Price'] = STRIKE / FX_EUR
    for row in deal['Autocall_Thresholds']:
        row[1] = threshold
    pf = job['Calc']['MergeMarketData']['ExplicitMarketData']['Price Factors']
    pf['InterestRate.EUR'] = {
        'Currency': 'EUR', 'Day_Count': 'ACT_365', 'Sub_Type': None,
        'Curve': {'.Curve': {'meta': [], 'data': [[0.0, R_EUR], [5.0, R_EUR]]}}}
    pf['FxRate.EUR'] = {
        'Domestic_Currency': None, 'Interest_Rate': 'EUR', 'Priority': 1, 'Spot': FX_EUR}
    pf['FXVol.EUR.USD'] = {
        'Surface_Type': 'Explicit', 'Moneyness_Rule': 'Sticky_Moneyness',
        'Surface': {'.Curve': {'meta': [], 'data': [
            [m, t, FX_SIGMA] for m in (0.8, 1.0, 1.2) for t in (0.02, 2.0)]}}}
    pf['Correlation.EquityPrice.EQ/FxRate.EUR.USD'] = {'Value': corr}
    return job


def _compo_digital(threshold, corr):
    """Closed form for the one-coupon compo autocall: a cash-or-nothing digital in EUR on
    C = S*X, reported in USD.

        C0 = SPOT / FX_EUR                 (the USD->EUR cross is 1/FX_EUR)
        b  = R_EUR - Q_EQ                  ((r_usd - q) + (r_eur - r_usd))
        sigma_c^2 = SIGMA^2 + 2*rho*SIGMA*FX_SIGMA + FX_SIGMA^2,  rho = -corr
        PV = Units * coupon * exp(-R_EUR*T) * N(d2) * FX_EUR      (EUR mark, USD report)

    `rho = -corr` is the sorted-pair convention: the correlation is authored on EUR.USD and this
    deal's cross runs USD->EUR, so `check_fx_name` flips the sign - the oracle carrying the flip
    is what makes this gate test the convention and not just the arithmetic.
    """
    t = HORIZON / DAYS
    c0 = SPOT / FX_EUR
    k = threshold * STRIKE / FX_EUR
    rho = -corr
    sigma_c = math.sqrt(SIGMA ** 2 + 2.0 * rho * SIGMA * FX_SIGMA + FX_SIGMA ** 2)
    b = R_EUR - Q_EQ
    d2 = (math.log(c0 / k) + (b - 0.5 * sigma_c ** 2) * t) / (sigma_c * math.sqrt(t))
    return UNITS * COUPON * math.exp(-R_EUR * t) * _ndtr(d2) * FX_EUR


def test_a_compo_autocall_is_a_digital_on_the_converted_spot(tmp_path):
    """The compo value oracle, exact under the flat document: the pricer simulates the PRODUCT
    S*X (spot scaled by the cross, fx carry per fixing, interval vol composed with the fx read),
    and a single coupon makes the payoff a digital Black prices with no Monte Carlo error - the
    survival probability at the t0 row is analytic. Compo OSS deals had NEVER priced before this
    (`calc_vol_adjustment` returned a python-float b_adj that `torch.unsqueeze` refused, skipping
    the deal), so this is the first value any compo autocall has ever been held to.

    MEASURED: 0.45887454 against the closed form's 0.45887454, 4.8e-16 relative; the flipped
    correlation lands on ITS closed form (0.43677514) exactly - both arms, so the sorted-pair
    sign flip is measured to decide something rather than assumed."""
    out, _ = _run(_compo_job(), tmp_path, 'compo')
    expected = _compo_digital(1.00, CORR)
    assert abs(_mtm(out) - expected) / expected < 1e-9, (_mtm(out), expected)
    flipped, _ = _run(_compo_job(corr=-CORR), tmp_path, 'compo_flip')
    mirrored = _compo_digital(1.00, -CORR)
    assert abs(_mtm(flipped) - mirrored) / mirrored < 1e-9, (_mtm(flipped), mirrored)
    assert abs(expected - mirrored) / expected > 0.01, 'the two arms must separate'


# --------------------------------------------------------------------------------------------
# the ledger, and the mark it has to agree with
# --------------------------------------------------------------------------------------------
def _cmc_job(threshold, units=10.0, buy_sell='Buy', coupon_days=(91, 182, 273)):
    """The template as a credit Monte Carlo on a simulated GBM spot, with several coupon dates.

    A SINGLE coupon date is blind to the defect this file exists to pin - the settle fired once
    per coupon on a settling row, so with one coupon it looked exactly right. The reconciliation
    also needs the deal to survive at least one date to be worth anything.
    """
    job = _template()
    deal = _deal_of(job)
    deal['Units'] = units
    deal['Buy_Sell'] = buy_sell
    dates = [{'.Timestamp': _stamp(d)} for d in coupon_days]
    deal['Autocall_Coupons'] = [[d, COUPON] for d in dates]
    deal['Autocall_Thresholds'] = [[d, threshold] for d in dates]
    deal['Price_Fixing'] = [[d, 0.0] for d in dates]
    deal['Expiry_Date'] = dates[-1]
    job['Calc']['Calculation'] = {
        'Object': 'CreditMonteCarlo', 'Base_Date': {'.Timestamp': BASE}, 'Currency': 'USD',
        'Time_grid': '0d 12m(1m)', 'Batch_Size': 256, 'Simulation_Batches': 1,
        'Random_Seed': 1, 'MCMC_Simulations': 1 << 12, 'Deflation_Interest_Rate': 'USD',
        'Generate_Cashflows': 'Yes'}
    market = job['Calc']['MergeMarketData']['ExplicitMarketData']
    market['Price Models'] = {'GBMAssetPriceModel.EQ': {'Vol': SIGMA, 'Drift': 0.0}}
    market['Model Configuration'] = {'.ModelParams': {
        'modeldefaults': {'EquityPrice': 'GBMAssetPriceModel'}, 'modelfilters': {}}}
    return job


def _stamp(offset):
    import datetime
    return (datetime.date.fromisoformat(BASE) + datetime.timedelta(days=offset)).isoformat()


def _cmc(job, tmp_path, name):
    path = os.path.join(str(tmp_path), f'{name}.json')
    with open(path, 'w') as f:
        json.dump(job, f, default=str)
    cx = rf.Context()
    cx.load_json(path)
    _, out = cx.run_job()
    # under a credit Monte Carlo `mtm` is the exposure PROFILE (dates x scenarios), not the
    # per-deal frame a base valuation returns
    return out['Results']['cashflows']['USD'], out['Results']['mtm']


def test_each_booked_date_carries_the_coupon_that_pays(tmp_path):
    """A path pays EXACTLY `Units * coupon`, exactly ONCE - the payment, scaled, then the latch.

    The document autocalls at its first coupon with certainty, so the first date books the whole
    payment and the later dates book nothing. Four defects had to be fixed for this, and a
    SINGLE-coupon document is blind to all four, the loop firing once so the wrong quantity and
    the right one coincide:

      * the settle sat in the coupon loop under a ROW-level `tau`, so it fired once per coupon
        and `cash_settle` accumulated - 0.24 where 0.08 pays;
      * it booked `P`, the accumulated VALUE, rather than the payment;
      * `nominal` scaled the mark and not the ledger, so `Units` and `Buy_Sell` never arrived;
      * `terminationDate` was stamped inside `sim_spot` and never returned, so every later block
        re-priced and RE-PAID the deal as though it had never fired - 0.8/0.8/0.8 where
        0.8/0/0 pays.
    """
    ledger, _ = _cmc(_cmc_job(threshold=0.01), tmp_path, 'amount')
    per_date = np.asarray(ledger.values, dtype=float).mean(axis=1)
    assert len(per_date) > 1, 'a single-coupon document cannot see the defects this gate pins'
    assert abs(per_date[0] - UNITS * COUPON) < 1e-6, (per_date[0], UNITS * COUPON)
    assert np.all(np.abs(per_date[1:]) < 1e-9), (per_date, 'an autocalled path pays once')


def test_an_autocalled_path_pays_once_and_is_worth_nothing_after(tmp_path):
    """A path that has autocalled pays once and is worth nothing after.

    The document autocalls at its FIRST coupon with certainty, so the ledger reads `0.8 / 0 / 0`
    and the profile `0.79206 / 0.8 / 0 / 0` - the t0 mark is that single payment discounted, the
    decision row carries the payment being made, and every later row is zero.

    This was a strict xfail: `terminationDate` was maintained correctly inside `sim_spot` - per
    scenario, off the observed indicator - and never returned, so the outer loop rebuilt it from
    -1 for every block and the deal kept paying (0.8 / 0.8 / 0.8). The latch is now a by-product
    of the simulation, carried into the next block's theta, and each decision registers ONE
    counterfactual carrying its whole reach - the latch over every later row plus an own-row
    fired/survived override; see `pv_MC_AutoCallSwap`'s docstring and
    `test_boundary_pricer_events.py` for the gradient side.
    """
    ledger, mtm = _cmc(_cmc_job(threshold=0.01), tmp_path, 'zero_tail')
    cash = np.asarray(ledger.values, dtype=float).mean(axis=1)
    profile = np.asarray(mtm.values, dtype=float).mean(axis=1)
    assert np.all(np.abs(cash[1:]) < 1e-9), (
        cash, 'the deal autocalled at its first coupon, so nothing pays after it')
    assert np.all(np.abs(profile[2:]) < 1e-9), (
        profile, 'a path that has autocalled is worth nothing from then on')


def _cva_job(threshold=1.02, spot=None, gradient='No', report='USD'):
    """The CMC document with a counterparty and the CVA block on - the sensitivity run.

    The threshold sits 2% out of the money so the trigger is LIVE: scenarios cross it at every
    coupon date, which is what makes the CVA's spot delta carry boundary flux in both halves of
    the decision's reach - the fired/survived fork on the decision row, and the carried latch
    killing every later row. A threshold nothing reaches is the control (`test_..._control`).
    """
    job = _cmc_job(threshold=threshold)
    calc = job['Calc']['Calculation']
    calc['Batch_Size'] = 1024
    calc['Simulation_Batches'] = 4
    calc['MCMC_Simulations'] = 256
    calc['Credit_Valuation_Adjustment'] = {
        'Calculate': 'Yes', 'Counterparty': 'CPTY', 'Deflate_Stochastically': 'No',
        'Stochastic_Hazard_Rates': 'No', 'Gradient': gradient}
    market = job['Calc']['MergeMarketData']['ExplicitMarketData']
    # the counterparty curve, in the wire form CustomJsonEncoder writes for a utils.Curve
    market['Price Factors']['SurvivalProb.CPTY'] = {
        'Recovery_Rate': 0.4,
        'Curve': {'.Curve': {'meta': [], 'data': [[0.0, 0.0], [10.0, 0.4]]}}}
    if spot is not None:
        market['Price Factors']['EquityPrice.EQ']['Spot'] = spot
    if report != 'USD':
        # report in a currency the deal does not pay, so every boundary branch crosses an fx
        # conversion on its way to the mtm grid. The cross is STATIC - no FX model is declared -
        # which exercises the conversion, not a simulated cross.
        calc['Currency'] = report
        market['Price Factors']['InterestRate.' + report] = {
            'Currency': report, 'Day_Count': 'ACT_365', 'Sub_Type': None,
            'Curve': {'.Curve': {'meta': [], 'data': [[0.0, 0.02], [5.0, 0.02]]}}}
        market['Price Factors']['FxRate.' + report] = {
            'Domestic_Currency': None, 'Interest_Rate': report, 'Priority': 1, 'Spot': 1.25}
    return job


def _cva(job, tmp_path, name):
    path = os.path.join(str(tmp_path), f'{name}.json')
    with open(path, 'w') as f:
        json.dump(job, f, default=str)
    cx = rf.Context()
    cx.load_json(path)
    _, out = cx.run_job()
    return out


def _cva_ladder(tmp_path, threshold, rungs=(0.3, 0.5, 1.0), report='USD', collateral=False):
    """AAD spot delta of the CVA, and a central-difference ladder of the SAME document.

    Common random numbers arrive through the contract: `Random_Seed` is in the document and the
    bumped runs change nothing but the `EquityPrice.EQ` `Spot` value, so each rung differences
    two runs drawing identical paths. No internals, nothing patched.
    """
    wrap = _collateralised if collateral else (lambda j: j)
    out = _cva(wrap(_cva_job(threshold=threshold, gradient='Yes', report=report)), tmp_path, 'aad')
    g = out['Results']['grad_cva']['Gradient']
    eq_rows = [i for i in g.index if 'EquityPrice' in str(i[0])]
    aad = float(g.loc[eq_rows[0]]) if eq_rows else 0.0
    crn = []
    for h in rungs:
        up = float(_cva(wrap(_cva_job(threshold=threshold, spot=SPOT + h, report=report)),
                        tmp_path, f'up{h}')['Results']['cva'])
        dn = float(_cva(wrap(_cva_job(threshold=threshold, spot=SPOT - h, report=report)),
                        tmp_path, f'dn{h}')['Results']['cva'])
        crn.append((up - dn) / (2.0 * h))
    return aad, crn, float(out['Results']['cva'])


def test_the_cva_spot_delta_matches_the_same_document_bumped(tmp_path):
    """The sensitivity gate, wholly inside the contract: `grad_cva`'s equity entry against a CRN
    central-difference ladder of the same job document, live trigger.

    MEASURED at 1024 x 4 batches, 256 inner, rungs 0.3/0.5/1.0 on spot 100:

        cva 0.005450535   AAD +1.3809097e-04   CRN 1.3581/1.4697/1.4362e-04
        disagreement 1.68% at the best rung, ladder flatness 8.22%

    so the 5% tolerance carries a 3x margin inside a ladder whose own spread is wider than it.
    The decision registers ONCE (`LatchedBoundarySet` with an own-row fired/survived override),
    and each half of its reach suppressed alone kills this gate, measured on this document:

        latch neutralised (triggered := untriggered): AAD +2.5547825e-04, +73.83% - the tape
                                  keeps paying rows the latch killed
        own-row override suppressed: AAD +3.7148406e-05, -72.65% - the decision row's digital
                                  flux gone

    Opposite signs, each ~43x the corrected residual - which is also why the WHOLE registration
    suppressed is a weak mutant here (+5.15%): the two flux halves nearly cancel on this fixture,
    so the per-half mutants are the ones that measure the subject.
    """
    aad, crn, cva = _cva_ladder(tmp_path, threshold=1.02)
    best = min(crn, key=lambda c: abs(aad - c))
    flat = (max(crn) - min(crn)) / abs(best)
    assert abs(aad - best) / abs(best) < 0.05, (aad, crn, cva, flat)


def _collateralised(job):
    """Wrap the deal in a zero-threshold NettingCollateralSet, in the wire form the decoder reads."""
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


def test_a_collateralised_cva_delta_carries_the_settled_coupon(tmp_path):
    """The collateralised twin of the sensitivity gate: the same document under a zero-threshold
    CSA, so each decision's counterfactual runs the gross->net chain and the settled-cash ledger
    instead of reaching the report grid additively.

    What this gate measures is the decision's LEDGER REACH. A trigger forced ON kills every later
    coupon's settled cash; forced OFF it pays at the path's first later firing - so the
    counterfactual must flip every payment row it touches, not only its own. Scoring the own
    payment alone leaves the later coupons' booked cash in the margin period's exposure windows,
    measured at +6.5% per added later decision on a two-coupon cut of this document (dumped
    engine branch rows against a closed-form chain replica; the excess sits exactly on the later
    coupon's C_ts_te window rows).

    MEASURED at 1024 x 4 batches, rungs 0.3/0.5/1.0 on spot 100:

        cva 0.0017986481   AAD +5.0705166e-05   CRN 4.9345/5.0634/5.15172e-05
        disagreement 0.14% at the best rung, ladder flatness 4.29%

    MUTATION: `latched_cash` truncated to the decision's own row reads +7.73% against the same
    ladder - the reach rows are what this gate kills over.
    """
    aad, crn, cva = _cva_ladder(tmp_path, threshold=1.02, collateral=True)
    best = min(crn, key=lambda c: abs(aad - c))
    assert abs(aad - best) / abs(best) < 0.05, (aad, crn, cva)


def test_the_cva_spot_delta_matches_in_a_foreign_reporting_currency(tmp_path):
    """The live gate reported in EUR, so every boundary branch crosses an fx conversion on its way
    to the mtm grid exactly as the reported value does - a branch the conversion skipped would
    land 1.25x off on a correction that is most of this number, far outside the tolerance.

    MEASURED: cva 0.0043604281 (the USD gate's 0.005450535 / 1.25 to six digits), AAD
    +1.1047279e-04 against CRN 1.0864/1.1757/1.1489e-04 - disagreement 1.68%, the USD ladder
    through the static cross, coherently."""
    aad, crn, cva = _cva_ladder(tmp_path, threshold=1.02, report='EUR')
    best = min(crn, key=lambda c: abs(aad - c))
    assert abs(aad - best) / abs(best) < 0.05, (aad, crn, cva)


def test_a_dead_trigger_contributes_no_spurious_delta(tmp_path):
    """The control this deal shape can actually have. A threshold no path reaches makes the deal
    WORTHLESS - a coupon-only autocall is nothing but its trigger, so there is no live-but-
    fluxless configuration to control against (the fixture-degeneracy rule cuts both ways: state
    the degeneracy, do not dress it as coverage). What a saturated trigger CAN gate is silence:
    the registrations still run under `Gradient: Yes`, and they must contribute neither value nor
    a spurious delta - cva, AAD and every CRN rung exactly zero."""
    aad, crn, cva = _cva_ladder(tmp_path, threshold=10.0, rungs=(0.3, 0.5))
    assert cva == 0.0 and aad == 0.0 and all(c == 0.0 for c in crn), (aad, crn, cva)


def test_the_ledger_mirrors_and_scales_with_the_deal(tmp_path):
    """`Units` and `Buy_Sell` reach the ledger exactly as they reach the mark - they did not."""
    one, _ = _cmc(_cmc_job(0.01, units=1.0), tmp_path, 'u1')
    ten, _ = _cmc(_cmc_job(0.01, units=10.0), tmp_path, 'u10')
    sold, _ = _cmc(_cmc_job(0.01, units=10.0, buy_sell='Sell'), tmp_path, 'sold')
    total = lambda led: float(np.asarray(led.values, dtype=float).sum() / led.shape[1])
    assert abs(total(ten) - 10.0 * total(one)) < 1e-6, (total(ten), total(one))
    assert abs(total(sold) + total(ten)) < 1e-6, (total(sold), total(ten))


def test_the_mark_is_the_first_settled_coupon_discounted(tmp_path):
    """The reconciliation: a deal that autocalls at its first coupon with certainty is worth that
    payment discounted back, so the t0 mark and the first ledger entry are the same cashflow seen
    from two places.

    Holding the two reports against each other through the discount curve is a statement neither
    a value gate nor a cashflow gate can make alone: a ledger correct in isolation but out of step
    with the mark still fails here.
    """
    ledger, mtm = _cmc(_cmc_job(threshold=0.01), tmp_path, 'recon')
    per_date = np.asarray(ledger.values, dtype=float).mean(axis=1)
    t = (ledger.index[0] - pd.Timestamp(BASE)).days / DAYS
    discounted = float(per_date[0]) * math.exp(-R_USD * t)
    t0 = float(np.asarray(mtm.values[0], dtype=float).mean())
    assert abs(t0 - discounted) / abs(discounted) < 5e-3, (t0, discounted, t)
