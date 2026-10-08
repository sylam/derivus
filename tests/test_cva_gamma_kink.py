"""The exposure-gamma kink term (½Ku²) on the CVA-Hessian route, end to end.

What second-order AAD drops at `relu(V)` is `delta(V)·V_θV_θᵀ`. `pricing.exposure_kink_term` puts it
back as a term worth an exact zero forward and a bit-identical zero at first order, and
`Credit_Monte_Carlo.execute` rides it through the CVA's own trapezoid under `Hessian: 'Yes'`.

THE FIXTURE IS THE MUTATION DETECTOR, and that is the whole reason it is an `EquityForwardDeal`. A
LINEAR payoff's pathwise gamma is IDENTICALLY ZERO - `V_t = S_t·e^{-q(T-t)} - K·e^{-r(T-t)}` has
`∂²V/∂S₀² = 0` exactly, so the only second derivative the spot-spot entry can carry is the density
term, and an engine that dropped it reports `0.0` against a CRN ladder that does not. Measured on
this document at 65536 paths, with the hook at `calculation.py`'s CVA Gradient block suppressed and
restored:

    gamma   0.0          ->  +4.2418932e-04   ladder +4.2608660e-04, 0.45%, flatness 3.09%
    vanna  +4.9641660e-03 -> -1.2843500e-02   ladder -1.3044104e-02, 1.56%, flatness 5.18%

and `cva` 0.2474386841 with `grad_cva`'s spot entry 0.017783891 on BOTH runs, to every digit either
prints - the admission equality, taken independently of the gate that asserts it.

The vanna reading is why a diagonal-only gate is not enough: the pathwise cross entry is a
plausible-looking number of the WRONG SIGN at 39% of the magnitude. (An independent JAX prototype
under GBM reads the same shape at its own geometry - the diagonal dies, the cross survives looking
healthy.)

SEED STABILITY over five seeds at 65536 paths: gamma spread 0.41%, vanna 0.69%; at 16384 they are
2.60% and 6.00%, which is what sizes the path count.

The atom refusal (`exposure_kink_term` refusing a row whose density climbs as `1/h` across the
bandwidth ladder) is reached by no document here: the collateralised net it was written for is
refused one step earlier, as the decision-product gate below reads.

THE FIXTURE-DEGENERACY CHECKLIST, per axis:

  r, q          varied - r = 4%, q = 1%. Equal rates would kill the forward's carry and put the
                crossing point at the strike.
  time rows     varied - five reporting rows, and the exposure CROSSES ZERO on rows 1-4 (23.3% /
                32.7% / 37.7% / 41.4% of paths negative). A book whose V never crosses has no kink,
                which is the deep-ITM control below.
  vol surface   varied - the exposure reads `GBMAssetPriceTSModelParameters.EQ`'s term structure,
                0.20 / 0.28 / 0.32 piecewise linear, so the vanna entry is a sum over knots a flat
                curve would collapse. (`VolatilityGrid.EQ` is degenerate because a linear forward
                never reads it.)
  side          varied - Buy on the live document, Sell on its mirror in the atom gate.
  netting sets  one: the CVA objective is the ROOT sum, so a second uncollateralised set adds rows
                to the same tensor and reaches nothing. The set is there because a bare deal reports
                only its own reval dates.
  direction /   degenerate because a forward has no barrier and no option type - which is why the
  option type   decision-product gate adds an autocall, and it REFUSES rather than reporting.

TWO GRID CHOICES ARE LOAD-BEARING.

  The grid STARTS AT 1d. `GBMAssetPriceTSModelImplied` builds its per-step vol as
  `sqrt(V(t_k) - V(t_{k-1}))`, so a t0 row makes the first step zero-length and `sqrt` at exactly
  zero has an infinite derivative: value and first order come back finite, the DOUBLE backward
  returns NaN on every `GBMAssetPriceTSModelParameters.EQ.Vol` entry. A property of the diffusion,
  not of this term.

  The grid CLIPS at 12m, which pins the document to the readings above. Left open it runs one row
  PAST maturity, where the exposure is identically zero across paths - and the run comes back whole
  (0.11% and 0.08% off the clipped readings, for the extra trapezoid pair). `exposure_kink_term`
  writes that row's kernel to zero, which is right rather than a rescue: the row's `V_θ` is zero
  too, so `K·V_θV_θᵀ` is zero whatever `K` is. An earlier build REFUSED that row as an atom.
"""
import datetime
import json
import logging
import os
import sys

# reference-derivus shadow-import guard (MEMORY): pin the package under test to THIS repo.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pytest
import torch

import derivus as rf
from derivus import utils
from crn_ladder import ladder

TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'fixtures', 'autocall_job.json')


def _template():
    with open(TEMPLATE) as f:
        return json.load(f)


# every constant read OUT of the market fixture, so the document follows the template
_T = _template()
_PF = _T['Calc']['MergeMarketData']['ExplicitMarketData']['Price Factors']

BASE = _T['Calc']['Calculation']['Base_Date']['.Timestamp']
SPOT = _PF['EquityPrice.EQ']['Spot']
R_USD = _PF['InterestRate.USD']['Curve']['.Curve']['data'][0][1]
Q_EQ = _PF['DividendRate.EQ']['Curve']['.Curve']['data'][0][1]
# the deal is struck 5% BELOW spot, which is what keeps the t0 row well away from the kink while
# leaving 23%-41% of paths across it further out: an at-the-forward strike would pin row 0 at zero
FORWARD_PRICE = 95.0
DEEP_ITM_PRICE = 10.0
# the simulated vol TERM STRUCTURE - the curve the exposure diffuses on, and the one bumped for vanna
VOL = [0.20, 0.28, 0.32]
VOL_TENOR = [0.0, 1.0, 3.0]
# 0.41% gamma spread across five seeds here; 2.60% at 16384, which is not enough for the ladders
PATHS = 1 << 16
GRID = '1d 3m(3m) 12m'


def _stamp(days):
    return (datetime.date.fromisoformat(BASE) + datetime.timedelta(days=days)).isoformat()


def _forward(reference, price, buy_sell='Buy'):
    """A LINEAR payoff: `Units·(F(t,T) - K)·D(t,T)`, whose pathwise gamma is identically zero."""
    return {'Object': 'EquityForwardDeal', 'Reference': reference, 'Equity': 'EQ',
            'Currency': 'USD', 'Discount_Rate': 'USD', 'Payoff_Currency': 'USD',
            'Buy_Sell': buy_sell, 'Units': 1.0, 'Forward_Price': price,
            'Maturity_Date': {'.Timestamp': _stamp(365)}}


def _autocall():
    """The template's own autocall - a DECISION product, priced by the same market."""
    deal = _template()['Calc']['Deals']['Deals']['Children'][0]['Instrument']['.Deal']
    deal['Reference'] = 'AC1'
    return deal


def _job(children=None, hessian='No', gradient='Yes', paths=PATHS, seed=1, batches=1):
    """The market fixture as a credit Monte Carlo with a counterparty and the CVA block on.

    The deals hang under an uncollateralised `NettingCollateralSet` because a bare deal reports only
    its OWN reval dates - one row at maturity - while a netting set reports every mtm date it spans,
    which is what gives the profile the rows the kink term is estimated per.
    """
    job = _template()
    job['Calc']['Deals']['Deals']['Children'] = [{
        'Instrument': {'.Deal': {
            'Object': 'NettingCollateralSet', 'Reference': 'NS1', 'Netted': 'True',
            'Collateralized': 'False'}},
        'Children': [{'Instrument': {'.Deal': deal}}
                     for deal in (children or [_forward('FWD1', FORWARD_PRICE)])]}]
    job['Calc']['Calculation'] = {
        'Object': 'CreditMonteCarlo', 'Base_Date': {'.Timestamp': BASE}, 'Currency': 'USD',
        'Time_grid': GRID, 'Batch_Size': paths, 'Simulation_Batches': batches, 'Random_Seed': seed,
        'MCMC_Simulations': 1, 'Deflation_Interest_Rate': 'USD', 'Gradient_Variables': 'All',
        'Credit_Valuation_Adjustment': {
            'Calculate': 'Yes', 'Counterparty': 'CPTY', 'Deflate_Stochastically': 'No',
            'Stochastic_Hazard_Rates': 'No', 'Gradient': gradient, 'Hessian': hessian}}
    market = job['Calc']['MergeMarketData']['ExplicitMarketData']
    market['Price Factors']['SurvivalProb.CPTY'] = {
        'Recovery_Rate': 0.4,
        'Curve': {'.Curve': {'meta': [], 'data': [[0.0, 0.0], [10.0, 0.4]]}}}
    # the simulated vol lives on the IMPLIED factor, which is what makes it an AAD leaf: a
    # GBMAssetPriceModel's Vol is a Price Models float and carries no gradient at all, so there
    # would be no vanna column to gate
    market['Price Factors']['GBMAssetPriceTSModelParameters.EQ'] = {
        'Quanto_FX_Volatility': None, 'Quanto_FX_Correlation': 0.0,
        'Vol': {'.Curve': {'meta': [], 'data': [[t, v] for t, v in zip(VOL_TENOR, VOL)]}}}
    market['Model Configuration'] = {'.ModelParams': {
        'modeldefaults': {'EquityPrice': 'GBMAssetPriceTSModelImplied'}, 'modelfilters': {}}}
    return job


def _run(job, tmp_path, name, patch=None):
    """JSON in, results out. A bump is a `patch_market` VALUES patch applied to the loaded
    document, so every rung runs the identical program under the identical seed - common random
    numbers arrive through the contract, with nothing reached into."""
    path = os.path.join(str(tmp_path), '{}.json'.format(name))
    with open(path, 'w') as f:
        json.dump(job, f, default=str)
    cx = rf.Context()
    cx.load_json(path)
    if patch:
        cx.patch_market(patch)
    _, out = cx.run_job()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out['Results']


def _delta(results):
    """`grad_cva`'s spot entry - the first-order number the Hessian's spot row differentiates."""
    grad = results['grad_cva']['Gradient']
    return float(grad.loc[[i for i in grad.index if i[0] == 'EquityPrice.EQ'][0]])


def _second_order(results):
    """(spot-spot, spot-vol) off `grad_cva_hessian`.

    The vanna is SUMMED over the vol curve's knots because the ladder that checks it shifts the
    whole curve: `sum_k d²CVA/dS dσ_k` IS the derivative of the spot delta under a parallel shift,
    the interpolation being linear in the knot values.
    """
    frame = results['grad_cva_hessian']
    row = [i for i in frame.index if i[0] == 'EquityPrice.EQ'][0]
    gamma = float(frame.loc[row, [c for c in frame.columns if c[1] == 'EquityPrice.EQ'][0]])
    vanna = sum(float(frame.loc[row, c]) for c in frame.columns
                if c[1].startswith('GBMAssetPriceTSModelParameters'))
    return gamma, vanna


def _kink_widths(messages):
    """The Silverman width `pricing.kink_kernel` sized per reporting row, off its DEBUG line - one
    run's rows in order, a multi-batch run's batch after batch."""
    return [float(m.split('eps=')[1].split()[0])
            for m in messages if m.startswith('KINK exposure ')]


def _spot_patch(spot):
    return {'EquityPrice.EQ': {'Spot': spot}}


def _vol_patch(level):
    """A PARALLEL shift of the simulated vol curve, expressed as its first knot's new level."""
    return {'GBMAssetPriceTSModelParameters.EQ': {'Vol': [v + level - VOL[0] for v in VOL]}}


# ---------------------------------------------------------------- admission

def test_asking_for_the_hessian_moves_nothing_the_run_already_reported(tmp_path):
    """ONE ORDER STRICTER than the boundary correction's admission, and that is the point.

    The correction is worth an exact zero forward, so it is gated on value alone. This term is worth
    an exact zero forward AND its gradient is `K·u·V_θ` with `u = V - V.detach()` an exact IEEE
    zero, so first order accumulates `+0.0` bit-for-bit and `grad_cva` must come back `array_equal`
    as well - not merely close. Only `grad_cva_hessian` differs, by existing.

    Killing mutation: `exposure_kink_term` taking `u = V`, a term with a first derivative.
    """
    off = _run(_job(hessian='No'), tmp_path, 'admit_off')
    on = _run(_job(hessian='Yes'), tmp_path, 'admit_on')

    assert off['cva'] == on['cva'], (off['cva'], on['cva'])
    assert np.array_equal(off['mtm'].values, on['mtm'].values), 'the exposure profile moved'
    assert off['grad_cva'].index.equals(on['grad_cva'].index), 'the gradient index moved'
    assert np.array_equal(off['grad_cva'].values, on['grad_cva'].values), (
        'grad_cva moved - the kink term contributed something at FIRST order, which it cannot do '
        'unless u is not an exact zero')
    assert 'grad_cva_hessian' not in off, 'a Hessian was reported without being asked for'
    assert 'grad_cva_hessian' in on, 'no Hessian was reported'


# ---------------------------------------------------------------- the ladders

def test_the_gamma_and_vanna_entries_land_on_crn_ladders_of_the_reported_delta(tmp_path):
    """THE STRUCTURAL KILL. A linear payoff has `∂²V/∂S₀² = 0` on every path, so an engine
    differentiating the frozen-decision graph twice reports EXACTLY 0.0 for gamma. The CRN ladder
    of the same document's `grad_cva` spot entry reads 4.1578 / 4.2416 / 4.2895 / 4.2609 / 4.2587
    e-04 across a 25x range of bumps, flat to 3.09%; with the term AAD reads +4.2418932e-04, 0.45%.

    THE CROSS ENTRY, which a diagonal-only gate cannot see: the pathwise spot-vol entry is
    +4.9641660e-03 - the WRONG SIGN, at 39% of the size of a ladder of -1.3044104e-02, flat to
    5.18%. The corrected entry lands inside it (1.56%) and TWICE it does not (96.9%), so the
    doubling is pinned without a second engine run.

    The ladders are of the GRADIENT, not of the value - a second-order gate that differenced the CVA
    twice would be measuring its own cancellation.

    Killing mutation: the kink term's hook at the CVA objective dropped - gamma reads exactly 0.0.
    """
    gamma, vanna = _second_order(_run(_job(hessian='Yes'), tmp_path, 'second_aad'))
    assert gamma != 0.0, (
        'the spot-spot entry is exactly zero, which is what a pathwise-only Hessian reports on a '
        'linear payoff - the kink term did not reach the objective')

    rung = ladder(price=lambda s: _delta(
        _run(_job(), tmp_path, 'gamma_bump', patch=_spot_patch(s))),
        aad=gamma, base=SPOT, rungs=(2e-3, 5e-3, 1e-2, 2e-2, 5e-2))
    assert rung.agrees(tol=0.05), 'the exposure gamma is not the derivative of the reported delta\n{}'.format(rung)

    rung = ladder(price=lambda x: _delta(
        _run(_job(), tmp_path, 'vanna_bump', patch=_vol_patch(x))),
        aad=vanna, base=VOL[0], rungs=(2e-3, 5e-3, 1e-2, 2e-2, 5e-2), absolute=True)
    assert rung.agrees(tol=0.10), 'the exposure vanna is not the derivative of the reported delta\n{}'.format(rung)
    doubled = abs(2.0 * vanna - rung.best) / abs(rung.best)
    assert doubled > 0.5, (
        'twice the corrected vanna is {:.1%} from the ladder, so this document cannot tell a '
        'corrected entry from a doubled one and pins nothing'.format(doubled))


def test_a_book_that_never_crosses_zero_has_no_kink_to_correct(tmp_path):
    """The control: the term must be INERT where there is no boundary, or the ladders above could
    pass on a term that manufactures curvature wherever it is switched on.

    Struck at 10 against a spot of 100 the forward is in the money on every path of every row
    (minimum exposure 27.32), so `relu` is the identity, the CVA is LINEAR in spot and its true
    gamma is exactly zero - every rung of a CRN ladder of its delta reads exactly 0.0. The kernel
    underflows and the entry reads 4.12e-29, against the live document's 4.24e-04.

    Killing mutation: the kernel centred on the row's mean rather than on the kink.
    """
    itm = [_forward('FWD1', DEEP_ITM_PRICE)]
    results = _run(_job(children=itm, hessian='Yes'), tmp_path, 'itm_aad')
    assert np.asarray(results['mtm'].values).min() > 0.0, (
        'this control is meant to have no crossing mass at all; it has some, so it controls nothing')
    gamma, _ = _second_order(results)
    assert abs(gamma) < 1e-12, (
        'the term manufactured a gamma of {:.6g} on a book with no kink'.format(gamma))


def test_a_counterparty_curve_ending_at_a_year_hazards_on_past_it(tmp_path):
    """A survival curve is a cumulative hazard, so past its last knot its last hazard goes on: a
    counterparty written at a 1% hazard to six months and 4% to a year is the same counterparty
    written on at 4% to fifty years, and a five-year forward's CVA reads one number off both - as
    declared, under a 91-day `Tenor_Offset`, which reads the factor past its last knot, and with
    `CDS_Tenors` adding knots to five years - where the one-year curve froze survival after it.

    Killing mutation: the survival factor read flat past its last knot.
    """
    def cva(curve, offset=0, cds_tenors=None):
        forward = dict(_forward('FWD5', FORWARD_PRICE), Maturity_Date={'.Timestamp': _stamp(5 * 365)})
        job = _job(children=[forward], gradient='No', paths=1 << 12)
        job['Calc']['Calculation'].update(Time_grid='1d 3m(3m)', Tenor_Offset=offset)
        if cds_tenors:
            job['Calc']['Calculation']['Credit_Valuation_Adjustment']['CDS_Tenors'] = cds_tenors
        job['Calc']['MergeMarketData']['ExplicitMarketData']['Price Factors']['SurvivalProb.CPTY'][
            'Curve'] = {'.Curve': {'meta': [], 'data': curve}}
        return float(_run(job, tmp_path, 'hazard')['cva'])

    for case in ({}, {'offset': 91}, {'cds_tenors': [1, 3, 5]}):
        year = cva([[0.0, 0.0], [0.5, 0.005], [1.0, 0.025]], **case)
        fifty = cva([[0.0, 0.0], [0.5, 0.005], [50.0, 1.985]], **case)
        assert year == pytest.approx(fifty, rel=1e-12) and fifty > 0.0, (case, year, fifty)


# ---------------------------------------------------------------- the two-sided atom logic

def test_a_netted_mirror_contributes_nothing_rather_than_refusing(tmp_path):
    """A deal against its exact mirror nets to an identical zero on every path of every row - and
    that is NOT a case for refusing.

    The term is self-limiting there: `K·V_θV_θᵀ` needs a `V_θ`, and the mirror's two deltas cancel
    exactly, so the contribution is zero whatever `K` is. The build that refused this document was
    refusing over a row it would have got right.

    So the assertion is that the mirror ADMITS - the mutant-killer for re-introducing a
    spread-and-mass classifier: `cva` 0.0 on an identically zero book (itself the check that the
    mirror is a mirror), an empty `grad_cva`, and a (0, 0) Hessian. No NaN, no refusal.

    Killing mutation: a row of zero spread no longer written to zero (`eps <= floor` -> `<`).
    """
    mirror = [_forward('FWD1', FORWARD_PRICE), _forward('FWD2', FORWARD_PRICE, buy_sell='Sell')]
    results = _run(_job(children=mirror, hessian='Yes'), tmp_path, 'atom')

    assert results['cva'] == 0.0, (
        'the mirror does not net to zero, so this gate says nothing about a pinned row')
    assert np.abs(np.asarray(results['mtm'].values)).max() == 0.0, 'the mirror leaves exposure'
    hessian = results['grad_cva_hessian']
    assert hessian.shape == (0, 0), (
        'a book with no differentiable exposure reported a {} Hessian'.format(hessian.shape))
    assert not np.isnan(np.asarray(hessian.values, dtype=float)).any(), (
        'the pinned row reached the kernel at a zero bandwidth and came back NaN')


# ---------------------------------------------------------------- decision products

def test_a_decision_product_refuses_the_hessian_and_keeps_its_gradient(tmp_path):
    """`Base_Revaluation`'s posture, adopted one calculation over. An autocall registers a
    `BoundarySet`, which is what makes its FIRST derivative right: `(gap - gap.detach())` times a
    DETACHED coefficient. Differentiate twice and the coefficient cannot move, so what comes back is
    the smooth part with the density-derivative flux block silently absent. Refused by name, naming
    the deal; first order is UNCHANGED, so the refusal is a fall-back rather than a loss.

    A collateralised set registers an `MTABoundarySet` whatever it holds, so the linear forward
    alone under a zero-threshold CSA is refused by the same refusal naming NS1 - upstream of the
    kink term, which is why the atom refusal's remedy cannot name a margin period. First order
    there: `cva` 0.0631180 at 1024 paths.

    Killing mutation: the CVA Hessian's refusal over registered boundary corrections dropped.
    """
    book = [_forward('FWD1', FORWARD_PRICE), _autocall()]
    with pytest.raises(utils.SecondOrderRefused) as refusal:
        _run(_job(children=book, hessian='Yes', paths=1024), tmp_path, 'decision')
    message = str(refusal.value)
    assert 'AC1' in message, 'the refusal must name the registering deal: ' + message
    assert 'Second-order flux at a JUMP' in message, message

    survives = _run(_job(children=book, hessian='No', paths=1024), tmp_path, 'decision_first')
    assert survives['cva'] > 0.0, 'the same book must still price at first order'
    assert abs(_delta(survives)) > 0.0, 'grad_cva stopped reporting a spot delta'

    job = _job(hessian='Yes', paths=1024)
    job['Calc']['Deals']['Deals']['Children'][0]['Instrument']['.Deal'].update({
        'Collateralized': 'True', 'Agreement_Currency': 'USD', 'Balance_Currency': 'USD',
        'Liquidation_Period': 0, 'Settlement_Period': 0, 'Opening_Balance': 0.0,
        'Credit_Support_Amounts': {
            'Bank': 'CPTY', 'Counterparty': 'CPTY',
            'Independent_Amount': {'.CreditSupportList': [[1, 0.0]]},
            'Received_Threshold': {'.CreditSupportList': [[1, 0.0]]},
            'Posted_Threshold': {'.CreditSupportList': [[1, 0.0]]},
            'Minimum_Received': {'.CreditSupportList': [[1, 0.0]]},
            'Minimum_Posted': {'.CreditSupportList': [[1, 0.0]]}}})
    with pytest.raises(utils.SecondOrderRefused) as refusal:
        _run(job, tmp_path, 'collateral')
    message = str(refusal.value)
    assert 'boundary correction: NS1' in message and 'exposure_kink_term' not in message, message
    job['Calc']['Calculation']['Credit_Valuation_Adjustment']['Hessian'] = 'No'
    assert _run(job, tmp_path, 'collateral_first')['cva'] > 0.0


# ---------------------------------------------------------------- the bandwidth's sample

def test_the_kernel_width_is_the_runs_path_count_and_not_one_batchs(tmp_path, caplog):
    """THE SILVERMAN WIDTH BELONGS TO THE RUN, NOT TO A BATCH.

    Every batch re-estimates these same five reporting rows and their gradients are ACCUMULATED, so
    the density's effective sample is `Simulation_Batches x Batch_Size`. Sizing from `Batch_Size`
    alone is too wide by `Simulation_Batches ** 0.2` - 14.87% at two batches - and the term's
    O(eps^2) bias rides it: this document's 65536 paths split two ways read the vanna entry 2.99%
    off its own CRN ladder where one batch reads 1.61%, and four ways 5.24% against 3.64% here.

    THE WIDTH IS THE ASSERTION, because it is what the mutation moves. Off `kink_kernel`'s own
    DEBUG line the two-batch run's rows sit 0.26 / 0.75 / 0.19 / 0.79 / 1.95% from the one-batch
    run's, which is the per-batch spread estimated off half the paths; reading the batch's own
    count again multiplies every one of them by 1.1487. The corrected entries FOLLOW rather than
    gate - gamma 0.08% and vanna 0.30%, against 0.27% and 1.14% uncorrected - because a 2% move is
    inside this document's own seed spread either way.

    Killing mutation: `exposure_kink_term` sizing its width off the batch's own paths.
    """
    with caplog.at_level(logging.DEBUG):
        one = _run(_job(hessian='Yes'), tmp_path, 'width_one')
        widths_one = _kink_widths(caplog.messages)
        caplog.clear()
        two = _run(_job(hessian='Yes', paths=PATHS // 2, batches=2), tmp_path, 'width_two')
        widths_two = _kink_widths(caplog.messages)

    assert widths_one and len(widths_two) == 2 * len(widths_one), (
        'the two-batch run did not estimate the same rows twice: {} widths against {}'.format(
            len(widths_two), len(widths_one)))
    off = max(abs(w / widths_one[i % len(widths_one)] - 1.0) for i, w in enumerate(widths_two))
    assert off <= 0.03, (
        'a two-batch run of the same total paths did not use the one-batch width: worst row '
        '{:.2%} against a tolerance of 3%, and sizing from the batch alone would read {:.2%} '
        'wider than that\n  one batch {}\n  two batches {}'.format(
            off, 2 ** 0.2 - 1.0, widths_one, widths_two))

    for name, a, b in zip(('gamma', 'vanna'), _second_order(one), _second_order(two)):
        assert abs(b / a - 1.0) <= 0.02, (
            'the corrected {} moved {:.2%} when the same 65536 paths were run in two batches, '
            'against a tolerance of 2%: {:.8g} -> {:.8g}'.format(name, b / a - 1.0, a, b))


