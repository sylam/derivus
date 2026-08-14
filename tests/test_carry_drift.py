"""`GARCHSpotModel.Carry_Drift` — the switch that puts the commodity's own cost of carry into the
simulated spot's log-drift, so a futures leg stops rolling down the curve with certainty.

OFF (the default) the per-step log-drift is `Mu·dt`. ON it is `Mu·dt + z(0)·dt`, where `z(0)` is
the FRONT of the factor's declared `Forward_Rate` carry curve, read per (step, path) out of the
`(carry_key, 'z0')` series `QuadraticCarryCurveModel.generate` publishes. Every gate here is a
whole `CreditMonteCarlo` job through `load_json` + `run_job` — JSON is the contract, and the world
is built inline (`commodity_aps_world.json`'s shape: `GARCHSpotModel.PLATINUM_CME` +
`QuadraticCarryCurveModel.PLATINUM_CARRY` + the basis block) so nothing here reads `data/`.

WHAT EACH GATE HOLDS

  1. OFF IS OFF, BITWISE. The key absent and the key at `'No'` price to the same bits; the same
     world at `'Yes'` does not. A default that leaked, or a truthiness read of the text field,
     shows up here and nowhere else.
  2. ON IS THE DRIFT IT DECLARES. On a carry curve held still (`Sigma ~ 0`, `Mu_L/Mu_D` at the
     initial state) the mean log return is `z(0)·T` within 3 MC standard errors, and the PAIRED
     difference against the same seed's OFF run is `z(0)·T` to 1e-5 relative — the drift is
     deterministic given the carry path, so switching it on is a pure shift of every path.
  3. THE FUTURE GOES DRIFTLESS. The world-before-solver statement: with the carry in the spot the
     mean futures mark is flat, without it the mark rolls down at `−z(0)·F_0` per year.
  4. FAIL LOUD. `Carry_Drift='Yes'` on a spot with no `Forward_Rate` link, or whose carry factor
     has no process to publish `z0`, raises naming the factor rather than pricing driftlessly.

THE TWO CONVENTIONS ARE DIFFERENT GATES, deliberately. `Convexity_Correction` is orthogonal to
this switch and each gate takes the spelling in which its statement is exact: gate 2 runs
`Convexity_Correction='No'`, where the declared log-drift IS the whole drift and the log-mean
measures it directly (under `'Yes'` the log-drift also carries `−½Var(r_t)`, ~1.5% a year here,
which would swamp a 3-s.e. gate on a 2-3% carry); gate 3 runs `'Yes'`, where the PRICE is the
martingale and the LEVEL statement about the futures mark is the exact one.

ANTI-PLACEBO — the fixture property each gate needs, and what goes blind without it.

| property | value | what goes blind without it |
|---|---|---|
| front carry | +3%, −2%, +2% | at `z(0) = 0` the switch is a no-op and every gate passes on a dead feature; one sign only cannot see a sign folded into the drift |
| a SLOPED carry case | knots 3.50%/6.50%, front 2.00% | with a FLAT curve `D = 0`, so `z(0) = L` whatever `z0_coeff` is — a mutant reading a knot, the level, or the wrong shape coefficient survives every flat case. The sloped case puts the front BELOW both knots |
| carry frozen, not zero | `Sigma_L/Sigma_D = 1e-12`, `Mu` at the initial state | a moving carry makes `z(0)·T` a random variable and the paired identity stops being exact; a carry with no process at all is gate 4's negative, not gate 2's world |
| `Antithetic` | `Yes` | the sampling error in the MEAN log return is an odd moment: unpaired, this seed sits 3.0 s.e. from its own target and the 3-s.e. gate is a coin flip; paired, 0.4 s.e. |
| flat carry in gate 3 | slope 0 | the future is driftless only when the curve is flat: `d log F = (z(0) − z(τ) − τ z'(τ))dt = −2aτ dt`, so a curved carry leaves a residual the gate would have to model |
| `Mu = 0` | GARCH's own drift off | a non-zero `Mu` and the gate measures `Mu + z(0)` and cannot say which moved |
| daily grid, 1y / 9m, 4096 paths | `'0d 1d(1d)'` | one step per year and the per-step drift is untested; too few paths and the 3-s.e. band is wider than the signal |
| repo curve at zero | `USD-REPO = 0` | the futures forward is `S·exp(z·τ + ∫r)`; a live repo adds a second deterministic roll and gate 3 stops being a statement about the carry |

MUTATION MATRIX — every one RUN, by monkeypatching a parameterised copy of `_carry_drift` /
`__init__` / `precalculate` onto the two classes and scoring the mutant on this whole file.
Control: 7 passed, 1 xfailed, 0 failing. Zero survivors.

| mutant | killed by | count |
|---|---|---|
| `Carry_Drift` read for truthiness (absent False, `'No'` True) | the OFF-identity gate, all three carry levels, the futures gate | 5 |
| the switch dead — `_carry_drift` always returns `Mu·dt` | everything, including the pinned defect (which stops reproducing) | 7 |
| the curve publishes the LEVEL `L` instead of the front `L + z0_coeff·D` | **the sloped carry ONLY** | 1 |
| `z0` added without `dt` | six, the OFF-identity gate included | 6 |
| the `z0` row shifted one step (the fork-index convention broken) | the same six | 6 |
| both fail-louds softened to a zero drift | everything but the OFF-identity gate | 6 |

The one-gate kill is the entry worth reading twice, and it is why the third parametrisation
exists: on a FLAT carry `D = 0`, so `L`, `z(0)`, `z(0.5)` and `z(1)` are all the same number and a
process publishing any of them passes both flat gates. Only a curve with a slope can tell the front
from the level — one fixture property, one gate, one mutant.

A KNOWN DEFECT IS PINNED HERE, as a strict xfail rather than as a note — see
`test_the_carry_must_be_simulated_as_far_as_the_spot`.
"""
import json

import numpy as np
import pandas as pd
import pytest

import derivus as rf
from derivus import utils

BASE = pd.Timestamp('2026-01-15')
EXCEL0 = float((BASE - utils.excel_offset).days)
KNOT_DAYS = (183, 548)                      #: dated carry knots, bracketing every contract priced
DT_C = 1.0 / 252.0
OMEGA, ALPHA, BETA = 9.17774951127383e-07, 0.027616302198889837, 0.9690723589689383
SPOT = 1661.4839212399793


def _world(front=0.03, slope=0.0, switch=None, convexity='No', months=12,
           commodity='PLATINUM_CME', forward_rate=True, carry_model=True, seed=1234):
    """`commodity_aps_world.json`'s world, inline, with a `CommodityFutureDeal` in place of the
    average-price swap and a carry curve held still at a declared front rate and slope.

    The two knots are `z(tau) = front + slope*tau` at their own aged tenors, and `Mu_L`/`Mu_D` are
    that same line rotated onto `Reference_Tenors` — so the AR(1) mean is the initial state and the
    curve neither reverts nor diffuses. `H0` is the long-run variance, so the GARCH is stationary
    and the 3-s.e. bands below are the calibrated 26.4% vol rather than a decaying transient.
    """
    tau = np.array(KNOT_DAYS) / utils.DAYS_IN_YEAR
    knots = front + slope * tau
    garch = {'Omega': OMEGA, 'Alpha': ALPHA, 'Beta': BETA, 'Nu': 7.566473643035478, 'Mu': 0.0,
             'H0': OMEGA / (1.0 - ALPHA - BETA), 'Calibration_DT_Years': DT_C,
             'Convexity_Correction': convexity}
    if switch is not None:
        garch['Carry_Drift'] = switch
    spot_factor = {'Spot': SPOT, 'Currency': 'USD', 'Interest_Rate': 'USD-REPO',
                   'Forward_Rate': 'PLATINUM_CARRY'}
    if not forward_rate:
        spot_factor.pop('Forward_Rate')
    defaults = {'CommodityPrice': 'GARCHSpotModel', 'ObservedBasis': 'BasisLinkedSpotModel',
                'ForwardRate': 'QuadraticCarryCurveModel'}
    models = {
        'GARCHSpotModel.PLATINUM_CME': garch,
        'BasisLinkedSpotModel.PLATINUM_CME.LBMA': {
            'A': -0.02270837039894683, 'Phi': 0.6505398690713546, 'Nu': 4.9433208837818565,
            'Sigma': 3.7630499724163147, 'Calibration_DT_Years': DT_C,
            'Slow_Mean_Lambda': 0.96875, 'Mu_0': 6.25},
        'QuadraticCarryCurveModel.PLATINUM_CARRY': {
            'Phi_L': 0.9962113396325056, 'Mu_L': front + 0.75 * slope, 'Sigma_L': 1.0e-12,
            'Phi_D': 0.946836023305215, 'Mu_D': 0.5 * slope, 'Sigma_D': 1.0e-12,
            'Gamma': 0.0, 'Nu': 3.0, 'Reference_Tenors': [0.5, 1.0],
            'Calibration_DT_Years': DT_C}}
    if not carry_model:
        # the carry factor stays in the world as a STATIC curve: nothing publishes (key, 'z0')
        defaults.pop('ForwardRate')
        models.pop('QuadraticCarryCurveModel.PLATINUM_CARRY')
    maturity = (BASE + pd.DateOffset(months=months)).strftime('%Y-%m-%d')
    return {'Calc': {
        'Calculation': {
            'Object': 'CreditMonteCarlo', 'Base_Date': {'.Timestamp': BASE.strftime('%Y-%m-%d')},
            'Currency': 'USD', 'Batch_Size': 4096, 'Simulation_Batches': 1, 'Random_Seed': seed,
            'Antithetic': 'Yes', 'Deflation_Interest_Rate': 'USD-SOFR',
            'Generate_Cashflows': 'No', 'Calc_Scenarios': 'All', 'Time_Grid': '0d 1d(1d)'},
        'Deals': {'Tag_Titles': '', 'Reference': 'carry', 'Deals': {'Children': [{
            'Instrument': {'.Deal': {'Object': 'NettingCollateralSet', 'Reference': 'NS',
                                     'Netted': 'True', 'Collateralized': 'False'}},
            'Children': [{'Instrument': {'.Deal': {
                'Object': 'CommodityFutureDeal', 'Reference': 'FUT', 'Commodity': commodity,
                'Currency': 'USD', 'Repo_Rate': 'USD-REPO', 'Carry': 'PLATINUM_CARRY',
                'Maturity_Date': {'.Timestamp': maturity}}}}]}]}},
        'MergeMarketData': {'ExplicitMarketData': {
            'System Parameters': {'Base_Currency': 'USD',
                                  'Base_Date': {'.Timestamp': BASE.strftime('%Y-%m-%d')}},
            'Model Configuration': {'.ModelParams': {'modeldefaults': defaults, 'modelfilters': {}}},
            'Price Factors': {
                'FxRate.USD': {'Domestic_Currency': None, 'Interest_Rate': 'USD-SOFR', 'Spot': 1.0},
                'InterestRate.USD-SOFR': {
                    'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                    'Curve': {'.Curve': {'meta': [], 'data': [[0.0, 0.041], [5.0, 0.033]]}}},
                'InterestRate.USD-REPO': {
                    'Currency': 'USD', 'Day_Count': 'ACT_365', 'Sub_Type': None,
                    'Curve': {'.Curve': {'meta': [], 'data': [[0.0, 0.0], [5.0, 0.0]]}}},
                'CommodityPrice.PLATINUM_CME': spot_factor,
                'ObservedBasis.PLATINUM_CME.LBMA': {'Spot': 9.5646587191794},
                'ForwardRate.PLATINUM_CARRY': {'Currency': 'USD', 'Curve': {'.Curve': {
                    'meta': [], 'data': [[EXCEL0 + KNOT_DAYS[0], float(knots[0])],
                                         [EXCEL0 + KNOT_DAYS[1], float(knots[1])]]}}}},
            'Price Models': models,
            'Correlations': {
                'GARCHSpotProcess.PLATINUM_CME': {
                    'QuadraticCarryCurveProcess.PLATINUM_CARRY.L': -0.3167,
                    'QuadraticCarryCurveProcess.PLATINUM_CARRY.D': 0.0157,
                    'BasisLinkedSpotProcess.PLATINUM_CME.LBMA': -0.1524},
                'QuadraticCarryCurveProcess.PLATINUM_CARRY.L': {
                    'QuadraticCarryCurveProcess.PLATINUM_CARRY.D': -0.2245,
                    'BasisLinkedSpotProcess.PLATINUM_CME.LBMA': 0.2278},
                'QuadraticCarryCurveProcess.PLATINUM_CARRY.D': {
                    'BasisLinkedSpotProcess.PLATINUM_CME.LBMA': -0.271}}}}}}


def _run(cfg):
    """The contract: `load_json` on the job text, then whatever calculation the JSON names."""
    cx = rf.Context(path_transform={}, file_transform={})
    cx.load_json((json.dumps(cfg), 'carry_drift.json'))
    return cx.run_job()[1]['Results']


def _log_return(results):
    """`(log(S_T/S_0) per path, T in years)` off the reported scenario frame."""
    frame = results['scenarios']['CommodityPrice.PLATINUM_CME'].xs(0.0, level='tenor')
    path = frame.values.T
    return np.log(path[-1] / path[0]), (frame.columns[-1] - frame.columns[0]).days / utils.DAYS_IN_YEAR


def _front_rate(cfg):
    """The declared front carry `z(0)`, rebuilt from the two KNOTS in the JSON by linear
    extrapolation to zero tenor — the affine statement, not the model's L/D rotation, so
    agreement is two routes meeting."""
    data = np.array(cfg['Calc']['MergeMarketData']['ExplicitMarketData']['Price Factors'][
                        'ForwardRate.PLATINUM_CARRY']['Curve']['.Curve']['data'])
    tau = (data[:, 0] - EXCEL0) / utils.DAYS_IN_YEAR
    return data[0, 1] - (data[1, 1] - data[0, 1]) * tau[0] / (tau[1] - tau[0])


def _se(sample):
    return sample.std(ddof=1) / np.sqrt(sample.size)


# ---------------------------------------------------------------------------
# 1. off is off, bitwise
# ---------------------------------------------------------------------------

def test_the_absent_switch_and_the_declared_no_are_the_same_run():
    """The key omitted and the key at `'No'` are the same world to the bit, on the same seed.

    KILLED BY: reading the text field for truthiness (`bool(param.get('Carry_Drift'))` — absent is
    False, `'No'` is True), or a declared default of `'Yes'`. The `'Yes'` arm is the anti-placebo:
    without it this gate passes on a switch that is wired to nothing.
    """
    absent = _run(_world(switch=None))['mtm']
    off = _run(_world(switch='No'))['mtm']
    on = _run(_world(switch='Yes'))['mtm']
    assert np.array_equal(absent.values, off.values), 'the default is not OFF'
    assert absent.std(axis=1).max() > 0.0, 'the marks do not disperse — nothing is being compared'
    assert not np.array_equal(absent.values, on.values), 'Carry_Drift=Yes changed nothing'


# ---------------------------------------------------------------------------
# 2. on is the drift it declares
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('front,slope', [(0.03, 0.0), (-0.02, 0.0), (0.02, 0.03)])
def test_the_carry_drift_is_the_front_of_the_curve(front, slope):
    """`E[log(S_T/S_0)] = z(0)·T` with the switch on and `0` with it off, each within 3 MC standard
    errors; and the PAIRED difference of the two runs is `z(0)·T` to 1e-5 relative — the drift is
    deterministic given the (frozen) carry path, so it shifts every path by the same amount and the
    difference of the means carries no MC error at all, only float32 rounding on the spot path
    (measured: 1.0e-6 to 2.0e-6 relative across the three curves).

    Both signs of the carry run, and the third case is SLOPED — knots at 3.50% and 6.50% with a
    front of 2.00%, so the target is not any number on the curve.

    KILLED BY: dropping the `z0` term; reading a knot (or the level `L`, or `z` at a reference
    tenor) instead of the front — the sloped case only; the wrong sign on the drift — either flat
    case; `z0` added without `dt` (or with the calibration step in place of the grid step) — the
    paired arm, by a factor of 252/365.25; a `z0` row shifted by one step — the paired arm, since
    row 0 is the anchor and the last row would fall off the end.
    """
    cfg_on, cfg_off = _world(front, slope, switch='Yes'), _world(front, slope, switch='No')
    on, years = _log_return(_run(cfg_on))
    off, _ = _log_return(_run(cfg_off))
    target = _front_rate(cfg_on) * years

    assert abs(on.mean() - target) < 3.0 * _se(on), (on.mean(), target, _se(on))
    assert abs(off.mean()) < 3.0 * _se(off), (off.mean(), _se(off))
    assert abs((on.mean() - off.mean()) / target - 1.0) < 1.0e-5, (on.mean() - off.mean(), target)


# ---------------------------------------------------------------------------
# 3. the future goes driftless
# ---------------------------------------------------------------------------

def test_the_futures_mark_rolls_down_without_the_switch_and_is_flat_with_it():
    """The world-before-solver statement. `F(t,T) = S_t exp(z (T−t))` on a flat carry and a zero
    repo, so with the spot drifting at `z(0)` the two halves cancel and the mean mark is flat,
    and without it the mark rolls down at `−z·F_0` a year with certainty. `Convexity_Correction`
    is `'Yes'` here: the statement is about the LEVEL, and that is the setting in which the price
    (not the log price) is the martingale.

    Measured on this world (9m contract, `F_0 = 1699.16`, 3 s.e. = 24.7/yr): ON drifts +1.2/yr,
    6.3 s.e. from the roll-down; OFF drifts −49.3/yr, 6.1 s.e. from flat. Each arm is asserted
    against BOTH targets, so neither can pass by landing in the middle.

    KILLED BY: anything gate 2 kills, plus a `z0` read that is not per-path (a scalar front rate
    would still flatten the MEAN here but not the paired arm above), and — the reason this gate
    exists beside gate 2 — a drift applied to the log path AFTER the forward is built, which
    leaves the mark rolling down while the spot statistics look right.
    """
    maturity = pd.Timestamp('2026-10-15')                       # BASE + 9 months
    marks = {}
    for switch in ('Yes', 'No'):
        mtm = _run(_world(front=0.03, switch=switch, convexity='Yes', months=9))['mtm'].loc[:maturity]
        years = (mtm.index[-1] - mtm.index[0]).days / utils.DAYS_IN_YEAR
        marks[switch] = ((mtm.iloc[-1].values.mean() - mtm.iloc[0, 0]) / years,
                         _se(mtm.iloc[-1].values) / years, mtm.iloc[0, 0])

    for switch, (drift, se, forward0) in marks.items():
        roll = -0.03 * forward0
        flat, rolling = abs(drift) < 3.0 * se, abs(drift - roll) < 3.0 * se
        assert (flat, rolling) == (switch == 'Yes', switch == 'No'), (switch, drift, roll, se)


# ---------------------------------------------------------------------------
# 4. fail loud
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('missing,message', [
    ({'forward_rate': False}, 'declares no Forward_Rate'),
    ({'carry_model': False}, "published no \\(key, 'z0'\\) series")])
def test_a_carry_drift_spot_without_a_carry_raises(missing, message):
    """`Carry_Drift='Yes'` with nothing to read the carry from is the phantom the switch exists to
    remove — a world whose spot silently does not drift while the futures still roll down — so both
    holes raise, naming the factor. `forward_rate=False` drops the link the spot factor declares;
    `carry_model=False` leaves the carry factor in the world as a STATIC curve, which resolves but
    publishes no `z0`.

    KILLED BY: either raise softened to a warning-and-continue, or a `.get(...)` default that
    substitutes a zero drift; and by a message that names the carry alone, since the factor that
    has to be corrected is the SPOT that declared the switch.
    """
    with pytest.raises(Exception, match='CommodityPrice.PLATINUM_CME'):
        _run(_world(switch='Yes', months=6, **missing))
    with pytest.raises(Exception, match=message):
        _run(_world(switch='Yes', months=6, **missing))


def test_the_carry_must_be_simulated_as_far_as_the_spot():
    """A DEFECT, pinned. Every stochastic factor gets its own `ScenarioTimeGrid`, cut one row past
    its own last dependent date, so the spot's step schedule and the carry's published `z0` need
    not have the same number of rows — and `_carry_drift` adds them elementwise. Pricing the
    COMPOSED spot name (`PLATINUM_CME.LBMA`, the platinum book's own `Commodity`) puts the
    `CommodityPrice` cutoff one row past the `ForwardRate`'s and the run dies inside torch:

        RuntimeError: The size of tensor a (182) must match the size of tensor b (183) at
        non-singleton dimension 0

    Not a tolerance and not a fixture accident: 6, 9 and 12 month contracts all fail, always by
    exactly one row, and the same world on the plain primary name runs. Whatever the fix, this
    file's `xfail` must come off with it (`strict`, so a fix that lands turns this red).
    """
    _run(_world(switch='Yes', months=6, commodity='PLATINUM_CME.LBMA'))
