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
plausible-looking number of the WRONG SIGN at 39% of the magnitude, and nothing about it looks like
a failure. (The owner's external table - JAX prototype, ATM forward-style exposure under GBM,
θ = (S₀, σ), 400k paths - reads the same shape at its own geometry: pathwise gamma 0.0000 and vanna
0.3966, double the truth, becoming 2.6415 and 0.1992 against Black 2.6521 / 0.1989. The RATIO is
the fixture's, not the term's; what reproduces is that the diagonal dies and the cross survives
looking healthy.)

SEED STABILITY, measured over five seeds at this document's 65536 paths: gamma 4.2419 / 4.2354 /
4.2262 / 4.2351 / 4.2246 e-04, a 0.41% spread; vanna -1.28435 / -1.28475 / -1.27759 / -1.28310 /
-1.27591 e-02, 0.69%. At 16384 the same spreads are 2.60% and 6.00%, which is what sizes the path
count here. The two-seed gate below carries a 2% tolerance, ~5x the measured spread.

THE FIXTURE-DEGENERACY CHECKLIST, per axis:

  r, q          varied - r = 4%, q = 1%, different and both non-zero. Equal rates would kill the
                forward's carry and put the crossing point at the strike.
  time rows     varied - five reporting rows, four of them after the first, and the exposure CROSSES
                ZERO on rows 1-4 (23.3% / 32.7% / 37.7% / 41.4% of paths negative). A book whose V
                never crosses has no kink at all, which is the deep-ITM control below.
  vol surface   varied - the vol the exposure actually reads is `GBMAssetPriceTSModelParameters.EQ`'s
                term structure, 0.20 / 0.28 / 0.32 piecewise linear, so the diffusion sees a real
                forward-variance strip and the vanna entry is a sum over knots that a flat curve
                would collapse. (The pricing surface `VolatilityGrid.EQ` is degenerate BECAUSE a
                linear forward never reads it - it has no optionality to imply.)
  side          varied - Buy on the live document, Sell on its mirror in the atom gate.
  netting sets  one, and stated: the CVA objective is the ROOT sum, so a second uncollateralised set
                adds rows to the same tensor and reaches nothing this one does not. The set is there
                at all because a bare deal reports only its own reval dates and this fixture needs a
                grid of them.
  direction /   degenerate BECAUSE a forward has no barrier and no option type. The decision-product
  option type   gate adds an autocall for exactly that reason, and it REFUSES rather than reporting.
  the conjunction   the ladder gates need a linear payoff AND crossing mass AND a differentiable
                vol AND second order in one run; that conjunction is the live document below.

TWO GRID CHOICES ARE LOAD-BEARING AND NEITHER IS COSMETIC.

  The grid STARTS AT 1d, not 0d. `GBMAssetPriceTSModelImplied` builds its per-step vol as
  `sqrt(V(t_k) - V(t_{k-1}))`, and a t0 row makes the first step zero-length: `sqrt` at exactly zero
  has an infinite derivative, and while the value and the first-order gradient come back finite the
  DOUBLE backward returns NaN on every `GBMAssetPriceTSModelParameters.EQ.Vol` entry - measured, the
  whole vol row and its cross with spot. That is a property of the diffusion, not of this term, and
  it is why no reporting row here has a collapsed spread.

  The grid CLIPS at 12m (`'1d 3m(3m) 12m'` does not end in a bracket, so `update_time_grid` takes the
  clipping branch), which pins the document to the readings above rather than rescuing it. Left open
  the grid runs one row PAST the deal's maturity, where every deal has matured and the exposure is
  identically zero across paths - and the run comes back whole: `cva` 0.2478635758, gamma
  +4.2465373e-04, vanna -1.2853217e-02, ZERO NaN entries, 0.11% and 0.08% off the clipped readings
  for the extra trapezoid pair, which is real. `exposure_kink_term` writes that row's kernel to zero
  and that is the mathematically right answer, not a rescue: the row's `V_θ` is zero as well, so
  `K·V_θV_θᵀ` is zero whatever `K` is. An earlier build REFUSED that row as an atom - a false
  positive on a row whose Hessian contribution is identically zero, blocking the whole second-order
  block over a reporting grid left open. The ladder that replaced that test does not fire there.
"""
import datetime
import json
import os
import sys

# reference-derivus shadow-import guard (MEMORY): pin the package under test to THIS repo.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pytest
import torch

import derivus as rf
import derivus.pricing as pricing
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
SEED_TOL = 0.02


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


def _job(children=None, hessian='No', gradient='Yes', paths=PATHS, seed=1):
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
        'Time_grid': GRID, 'Batch_Size': paths, 'Simulation_Batches': 1, 'Random_Seed': seed,
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

    A term that could not make that guarantee would be changing the reported book to report a greek,
    which is the one thing the CVA block does not do.
    """
    off = _run(_job(hessian='No'), tmp_path, 'admit_off')
    on = _run(_job(hessian='Yes'), tmp_path, 'admit_on')

    assert off['cva'] == on['cva'], (
        'the reported CVA moved when the Hessian was asked for: {!r} -> {!r}'.format(
            off['cva'], on['cva']))
    assert np.array_equal(off['mtm'].values, on['mtm'].values), 'the exposure profile moved'
    assert off['grad_cva'].index.equals(on['grad_cva'].index), 'the gradient index moved'
    assert np.array_equal(off['grad_cva'].values, on['grad_cva'].values), (
        'grad_cva moved - the kink term contributed something at FIRST order, which it cannot do '
        'unless u is not an exact zero')
    assert 'grad_cva_hessian' not in off, 'a Hessian was reported without being asked for'
    assert 'grad_cva_hessian' in on, 'no Hessian was reported'


# ---------------------------------------------------------------- the ladders

def test_the_gamma_entry_lands_on_a_crn_ladder_of_the_reported_delta(tmp_path):
    """THE STRUCTURAL KILL. A linear payoff has `∂²V/∂S₀² = 0` on every path, so an engine
    differentiating the frozen-decision graph twice reports EXACTLY 0.0 here - measured, on this
    document, before the term. The CRN ladder of the same document's `grad_cva` spot entry reads
    4.1578 / 4.2416 / 4.2895 / 4.2609 / 4.2587 e-04 across a 25x range of bumps, flat to 3.09%, so
    the pathwise answer is 100% wrong and no tolerance can rescue it.

    With the term: AAD +4.2418932e-04 against that ladder, 0.45% at the flattest rung.

    The ladder is of the GRADIENT, not of the value - a second-order gate that differenced the CVA
    twice would be measuring its own cancellation.
    """
    results = _run(_job(hessian='Yes'), tmp_path, 'gamma_aad')
    gamma, _ = _second_order(results)
    assert gamma != 0.0, (
        'the spot-spot entry is exactly zero, which is what a pathwise-only Hessian reports on a '
        'linear payoff - the kink term did not reach the objective')

    rung = ladder(price=lambda s: _delta(
        _run(_job(), tmp_path, 'gamma_bump', patch=_spot_patch(s))),
        aad=gamma, base=SPOT, rungs=(2e-3, 5e-3, 1e-2, 2e-2, 5e-2))
    assert rung.agrees(tol=0.05), 'the exposure gamma is not the derivative of the reported delta\n{}'.format(rung)


def test_the_vanna_entry_lands_on_its_own_ladder_and_pins_the_doubling(tmp_path):
    """THE CROSS ENTRY, which a diagonal-only gate cannot see.

    The pathwise spot-vol entry is not zero - it is +4.9641660e-03, a plausible-looking number that
    is wrong in SIGN and at 39% of the size against a ladder of -1.3044104e-02 (readings -1.29268 /
    -1.30441 / -1.30530 / -1.34338 / -1.36028 e-02, flat to 5.18%). That is the failure mode the
    owner's external table records as "double the truth" and it is the reason this file gates the
    off-diagonal at all.

    The doubling is pinned WITHOUT a second engine run: one run cannot report both the corrected and
    the uncorrected entry, so the statement is made against the ladder from both sides - the
    corrected entry lands inside it, and TWICE the corrected entry does not land anywhere near it.
    Measured: 1.56% and 96.9%, a 62x separation.
    """
    results = _run(_job(hessian='Yes'), tmp_path, 'vanna_aad')
    _, vanna = _second_order(results)

    rung = ladder(price=lambda x: _delta(
        _run(_job(), tmp_path, 'vanna_bump', patch=_vol_patch(x))),
        aad=vanna, base=VOL[0], rungs=(2e-3, 5e-3, 1e-2, 2e-2, 5e-2), absolute=True)
    assert rung.agrees(tol=0.10), 'the exposure vanna is not the derivative of the reported delta\n{}'.format(rung)
    doubled = abs(2.0 * vanna - rung.best) / abs(rung.best)
    assert doubled > 0.5, (
        'twice the corrected vanna is {:.1%} from the ladder, so this document cannot tell a '
        'corrected entry from a doubled one and pins nothing'.format(doubled))


def test_a_book_that_never_crosses_zero_has_no_kink_to_correct(tmp_path):
    """The control, and the other half of the fixture-degeneracy statement: the term must be INERT
    where there is no boundary.

    Struck at 10 against a spot of 100 the forward is in the money on every path of every row
    (measured minimum exposure 27.32), so `relu` is the identity, the CVA is LINEAR in spot and its
    true gamma is exactly zero. The kernel underflows and the entry reads 4.12e-29 - 1e-26 of the
    live document's 4.24e-04 - against a CRN ladder of the delta whose every rung is exactly 0.0,
    the delta being constant.

    Without this the ladder gate above could be passing on a term that manufactures curvature
    wherever it is switched on.
    """
    itm = [_forward('FWD1', DEEP_ITM_PRICE)]
    results = _run(_job(children=itm, hessian='Yes'), tmp_path, 'itm_aad')
    assert np.asarray(results['mtm'].values).min() > 0.0, (
        'this control is meant to have no crossing mass at all; it has some, so it controls nothing')
    gamma, _ = _second_order(results)

    crn = [_delta(_run(_job(children=itm), tmp_path, 'itm_bump', patch=_spot_patch(SPOT + h)))
           - _delta(_run(_job(children=itm), tmp_path, 'itm_bump', patch=_spot_patch(SPOT - h)))
           for h in (0.5, 1.0, 5.0)]
    assert crn == [0.0, 0.0, 0.0], (
        'the delta of a book with no crossing mass is not constant in spot: {}'.format(crn))
    live, _ = _second_order(_run(_job(hessian='Yes'), tmp_path, 'live_aad'))
    assert abs(gamma) < 1e-6 * abs(live), (
        'the term manufactured a gamma of {:.6g} on a book with no kink, against {:.6g} where '
        'there is one'.format(gamma, live))


# ---------------------------------------------------------------- the two-sided atom logic

def test_a_netted_mirror_contributes_nothing_rather_than_refusing(tmp_path):
    """A deal against its exact mirror in one netting set nets to an identical zero on every path of
    every row - and that row is NOT a case for refusing, which is the correction this gate carries.

    The term is self-limiting there. `K·V_θV_θᵀ` needs a `V_θ`, and an exposure that is functionally
    zero in every factor has none: the mirror's two deltas cancel exactly, so the contribution is
    zero whatever `K` is and whatever the density at the kink does. The build that refused this
    document was refusing over a row it would have got right, and the same test blocked the whole
    second-order block on any reporting grid left open past a book's last maturity.

    So the assertion is the opposite one, and it is the mutant-killer for re-introducing a
    spread-and-mass classifier: the mirror ADMITS. `cva` 0.0 on an identically zero book (which is
    itself the check that the mirror really is a mirror), an empty `grad_cva`, and a (0, 0) Hessian
    - empty because the objective is identically zero as a function of every factor, not because
    anything failed. No NaN, no refusal.
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


def test_a_row_whose_density_climbs_as_its_bandwidth_narrows_refuses_by_name():
    """THE ATOM, diagnosed the way the roadmap says to diagnose one: on the bandwidth LADDER.

    A point mass at the kink and a narrow density at the kink have the same spread, the same
    near-zero mass and the same Silverman width - every scalar a single-width estimator can read.
    What separates them is what `f_V(0)` DOES as the width is varied: a density plateaus, a mass of
    weight p reads `p/(h·√2π)` and climbs as `1/h`. Measured here at 65536 paths, p of the paths at
    exactly zero and the rest at 1.0: the climb across the ladder's factor of 8 is 8.000 at EVERY p
    from 0.999 down to 0.0001, against 1.003 / 1.031 / 1.027 / 1.045 on the live document's own
    reporting rows above. A 7.7x separation, and the threshold sits at 2.0.

    Taken on the function because no CMC document in this build reaches it. The case the refusal is
    FOR - a collateralised net pinned inside its threshold, at the kink with a real `V_θ` - is
    refused one step earlier by the decision-product refusal, which the collateral gate below
    measures; and the remedy this message names is therefore not "go and build one".

    The previous classifier is what this kills. It fired on `collapsed AND mass-at-zero`, a
    conjunction no p in (0, 1) could satisfy: `collapsed` holds only for `√(p/(1−p)) ≤ 0.0867`, i.e.
    p ≤ 0.0075, while the mass test wanted p ≥ 0.01. Every row below rode straight through it while
    reporting a density that grows without bound as the path count rises.
    """
    n = PATHS
    row = torch.full((1, n), 1.0, dtype=torch.float64)
    row[0, :int(0.5 * n)] = 0.0
    with pytest.raises(utils.SecondOrderRefused) as refusal:
        pricing.exposure_kink_term(row)
    message = str(refusal.value)
    assert 'ATOM' in message and 'exposure_kink_term' in message, message
    assert 'row(s) [0]' in message, (
        'the refusal must name the rows it refuses on, or nobody can clip a grid off it: ' + message)
    assert "Hessian: 'No'" in message and 'clip the reporting grid' in message, (
        'a refusal names the remedies that WORK: ' + message)
    assert 'refused one step earlier' in message, (
        'the message must not send a caller off to price a collateralised set, which this build '
        'refuses for a different reason one step earlier: ' + message)

    # the separation itself, so the threshold is gated and not merely configured
    def climb(rows):
        Vbar = torch.tensor(rows, dtype=torch.float64)
        eps = 1.06 * Vbar.std(dim=1, keepdim=True) * Vbar.shape[1] ** -0.2
        rungs = [pricing._kink_density_at_zero(Vbar, c * eps, 1)
                 for c in (pricing.KINK_ATOM_LADDER[0], pricing.KINK_ATOM_LADDER[-1])]
        return float(rungs[1] / rungs[0])

    for p in (0.999, 0.5, 0.0001):
        pinned = torch.full((1, n), 1.0, dtype=torch.float64)
        pinned[0, :int(p * n)] = 0.0
        assert climb(pinned.tolist()) > 7.9, (
            'an atom of weight {} does not read as 1/bandwidth, so the ladder is not measuring '
            'what this refusal is asserted on'.format(p))
    healthy = torch.randn(1, n, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    assert climb(healthy.tolist()) < 1.1, (
        'a plain normal draw climbs across the ladder, so the threshold would refuse live rows')
    assert pricing.KINK_ATOM_LADDER_DIVERGENCE == 2.0, (
        'the threshold moved off the value the 7.9-versus-1.1 separation above was measured for')


def test_a_collapsed_row_away_from_the_kink_is_ignored_rather_than_refused():
    """The OTHER side of the same test, taken directly on the function because no reporting row of a
    live CMC document can reach it (see this file's header: a t0 row is what has no spread, and this
    diffusion's double backward is NaN there).

    A row with no spread is not an atom unless its mass is AT zero. A book marked at a constant 7.7
    across paths - one scenario, or a portfolio of pure cash - has a bandwidth of exactly zero and
    nothing within it of the kink: the kernel is ZERO there, not 0/0, and the row must contribute an
    exact 0.0 to the Hessian rather than a NaN or a refusal.

    Both readings are taken as second derivatives of the term itself, because the term's VALUE is an
    exact zero on every row and says nothing about which branch was taken.
    """
    def curvature(rows):
        theta = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        term = pricing.exposure_kink_term(theta * torch.tensor(rows, dtype=torch.float64)).sum()
        first, = torch.autograd.grad(term, theta, create_graph=True)
        second, = torch.autograd.grad(first, theta)
        return float(term.detach()), float(first), float(second)

    spread = np.linspace(-4.0, 4.0, 64).tolist()
    value, first, second = curvature([[7.7] * 64])
    assert value == 0.0 and first == 0.0, (value, first)
    assert second == 0.0, (
        'a row with no spread 7.7 away from the kink contributed {:.6g} of curvature - the kernel '
        'was evaluated at a zero bandwidth instead of being written to zero'.format(second))
    assert curvature([spread])[2] > 0.0, (
        'a row that does cross the kink contributed no curvature, so the reading above is vacuous')

    # one sample has no spread either, and the same answer follows from the same test - AT the kink
    # as well as away from it, because a row with no bandwidth has no ladder to diverge on and its
    # V_theta is zero anyway. Neither reading may be a NaN or a refusal
    assert curvature([[7.7]])[2] == 0.0
    assert curvature([[0.0]])[2] == 0.0, (
        'a single sample sitting exactly at the kink refused or returned a NaN; it has no '
        'bandwidth, so there is nothing there to estimate and nothing to refuse')
    assert curvature([[0.0] * 64])[2] == 0.0, (
        'a row pinned at the kink on every path refused; its V_theta is zero, so its contribution '
        'is zero whatever the density does')


def test_the_kernel_argument_is_detached_so_K_prime_never_reaches_the_tape():
    """The confinement claim, and the one claim here no engine gate can reach: with K's argument
    attached the second derivative is BIT-IDENTICAL (31.752557989663714 on this probe either way),
    so the ladders, the admission and the seed gate all pass on a term whose kernel is on the tape.

    What changes is the ORDER the graph stops at. `0.5*K(Vbar)*u**2` differentiated twice is
    `K(Vbar)*V_theta**2` - a detached coefficient times a constant, so the second derivative is a
    LEAF and autograd refuses a third. Attach the argument and it carries a graph whose third
    derivative is `3*K'(Vbar)` - the density DERIVATIVE, the noisier estimator a JUMP needs, built
    and retained on every reporting row. The reason the mutant is invisible at second order is
    structural rather than lucky: with `u = V - V.detach()` an exact zero, every `K'` term in the
    double backward carries a factor of u or u squared.
    """
    rows = torch.linspace(-4.0, 4.0, 512, dtype=torch.float64).reshape(1, -1)
    theta = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    term = pricing.exposure_kink_term(theta * rows).sum()
    first, = torch.autograd.grad(term, theta, create_graph=True)
    second, = torch.autograd.grad(first, theta, create_graph=True)
    assert float(second.detach()) > 0.0, 'the probe found no curvature, so it tests nothing'
    assert not second.requires_grad, (
        "the exposure gamma carries a graph into third order, so K's argument was not detached "
        "and K' is on the tape: grad_fn {}".format(second.grad_fn))


# ---------------------------------------------------------------- decision products

def test_a_decision_product_refuses_the_hessian_and_keeps_its_gradient(tmp_path):
    """`Base_Revaluation`'s posture, adopted one calculation over.

    An autocall added to the book registers a `BoundarySet`, and that registration is what makes its
    FIRST derivative right: `(gap - gap.detach())` times a DETACHED coefficient. Differentiate it a
    second time and the coefficient cannot move, so what comes back is the smooth part of the second
    derivative with the density-derivative flux block silently absent - a cross-gamma that looks
    like a cross-gamma. Refused by name, naming the deal.

    First-order behaviour is UNCHANGED, which is the half that makes the refusal a fall-back rather
    than a loss: the same book at `Hessian: 'No'` prices, reports a CVA and reports `grad_cva`.
    """
    book = [_forward('FWD1', FORWARD_PRICE), _autocall()]
    with pytest.raises(utils.SecondOrderRefused) as refusal:
        _run(_job(children=book, hessian='Yes', paths=1024), tmp_path, 'decision')
    message = str(refusal.value)
    assert 'AC1' in message, 'the refusal must name the registering deal: ' + message
    assert 'Second-order flux at a JUMP' in message, (
        'the refusal must say where the design that will answer it lives: ' + message)

    survives = _run(_job(children=book, hessian='No', paths=1024), tmp_path, 'decision_first')
    assert survives['cva'] > 0.0, 'the same book must still price at first order'
    assert abs(_delta(survives)) > 0.0, 'grad_cva stopped reporting a spot delta'


def test_a_collateralised_set_is_refused_one_step_before_the_kink_term_sees_it(tmp_path):
    """WHY THE ATOM REFUSAL'S REMEDY DOES NOT NAME A MARGIN PERIOD, measured rather than asserted.

    The collateralised net matched inside its threshold - pinned at the kink with a real `V_θ` - is
    the case the atom refusal exists for, and it is the one case `exposure_kink_term` never sees. A
    `NettingCollateralSet` with `Collateralized: 'True'` registers an `MTABoundarySet` whatever it
    holds: the set below holds ONLY the linear forward, no barrier, no autocall, nothing that is a
    decision product in the deal sense, and `Hessian: 'Yes'` is still refused by the decision-product
    refusal naming NS1, upstream of the CVA objective's kink hook.

    So a refusal telling a caller to "price the set with a margin period or a threshold" would be an
    instruction to build a book this build refuses for a different reason, which is what the message
    used to say. The gate that keeps it honest is this one, plus the wording assertion above.

    First order is unaffected, as everywhere else in this posture: `cva` 0.0631180 at 1024 paths.
    """
    job = _job(hessian='Yes', paths=1024)
    job['Calc']['Deals']['Deals']['Children'][0]['Instrument']['.Deal'].update({
        'Collateralized': 'True', 'Agreement_Currency': 'USD', 'Balance_Currency': 'USD',
        'Liquidation_Period': 0, 'Settlement_Period': 0, 'Opening_Balance': 0.0,
        'Credit_Support_Amounts': {
            'Bank': 'CPTY', 'Counterparty': 'CPTY', 'Independent_Amount_Reference': 'None',
            'Independent_Amount': {'.CreditSupportList': [[1, 0.0]]},
            'Received_Threshold': {'.CreditSupportList': [[1, 0.0]]},
            'Posted_Threshold': {'.CreditSupportList': [[1, 0.0]]},
            'Minimum_Received': {'.CreditSupportList': [[1, 0.0]]},
            'Minimum_Posted': {'.CreditSupportList': [[1, 0.0]]}}})

    with pytest.raises(utils.SecondOrderRefused) as refusal:
        _run(job, tmp_path, 'collateral')
    message = str(refusal.value)
    assert 'boundary correction: NS1' in message, (
        'the collateralised set was refused, but not by the decision-product refusal - so the atom '
        "refusal's remedy may be reachable after all and its wording must be re-derived: " + message)
    assert 'exposure_kink_term' not in message, (
        'the kink term saw a collateralised set, which the atom refusal states it cannot: ' + message)

    job['Calc']['Calculation']['Credit_Valuation_Adjustment']['Hessian'] = 'No'
    survives = _run(job, tmp_path, 'collateral_first')
    assert survives['cva'] > 0.0, 'the same collateralised book must still price at first order'


# ---------------------------------------------------------------- noise

def test_two_seeds_agree_on_the_gamma_entry(tmp_path):
    """A kernel estimate is only worth quoting if it does not track the draws. Measured over five
    seeds at this document's 65536 paths: 4.2419 / 4.2354 / 4.2262 / 4.2351 / 4.2246 e-04, a 0.41%
    spread, against 2.60% at 16384 - which is what chose the path count. The tolerance here is 2%,
    ~5x the measured spread and still far inside the ladder's own resolution.
    """
    first, _ = _second_order(_run(_job(hessian='Yes', seed=1), tmp_path, 'seed1'))
    second, _ = _second_order(_run(_job(hessian='Yes', seed=2), tmp_path, 'seed2'))
    spread = abs(first - second) / abs(0.5 * (first + second))
    assert spread < SEED_TOL, (
        'the gamma entry moves {:.2%} between seeds, which is estimator noise and not an '
        'estimate: {:.6g} against {:.6g}'.format(spread, first, second))
