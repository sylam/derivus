"""`QuadraticCarryCurveCalibration.Location_Window`: the LEVEL's location is refitted on the
trailing rows while everything else keeps the full sample.

The carry level's mean is the fast-moving component - a stale mean forecasts worse than a random
walk - while its scale, its tail and the whole shape equation want the long sample. The field cuts
the fit in exactly one place: after the joint full-window (L, D) AR(1)-t fit, `Phi_L` and `Mu_L` are
replaced by an AR(1)-t on `L[-w:]`, and NOTHING else moves.

What this file gates:

  * OFF IS OFF, bit-identically. Absent, `0`, and `>= len(data)` are the same fit to the last ulp -
    the branch is `0 < w < len(L)` and both ends of that condition are exercised.
  * THE SPLIT DOES WHAT IT SAYS. On a level with a regime shift in its mean, `w = 252` puts `Mu_L`
    on the TRAILING regime (3.82% against a 4.0% truth) while the single-window fit lands at 1.90%
    - and `Sigma_L`, `Sigma_D`, `Nu`, `Gamma`, `Phi_D`, `Mu_D` come back EXACTLY equal, `==` and
    not `approx`, because a location refit that moved a scale would be a different estimator.
  * THE RESIDUALS STAY FULL-WINDOW. `delta` feeds the framework's correlation consolidation, so a
    trailing refit that also re-published its own residuals would silently reconsolidate the
    correlation off 252 rows. The returned frame is bit-identical to the single-window fit's.
  * THE BOUNDARY. `w = len(data) - 1` is the last value that still fires the branch, and it must
    return to the full fit (loosely - it is a solo AR(1) with its own nu, not the joint fit).

ANTI-PLACEBO - the fixture property each gate needs, and what goes blind without it.

| property | value | what goes blind without it |
|---|---|---|
| a regime shift in the level's mean | 1.5% -> 4.0% | at ONE mean both windows estimate the same number and gate 2 cannot tell "refitted on the tail" from "not refitted at all" |
| `Phi_L` = 0.95, deliberately NOT the archive's near-unit root | 252 rows is ~19 half-lives | at phi ~ 0.996 the AR mean is barely identified in a 252-row window (mu = intercept/(1-phi) with 1-phi ~ 0.004) and the trailing `Mu_L` is noise - the [3%, 5%] band would be a coin flip |
| the shift is 300 rows from the end, the window 252 | the window sits ENTIRELY inside the new regime | a window straddling the break returns a blend, and the gate would need a tolerance where it can have a band |
| n = 2300, N_OLD >> WINDOW | full-window `Mu_L` = 1.90% | on a short sample the two windows nearly coincide and "the single window lands materially lower" is vacuous |
| `Gamma` = -0.30, `Sigma_D` <> `Sigma_L` | the shape equation is live and distinct | a dead or degenerate D equation cannot show that only the LEVEL's location moved |
| `Nu` = 5, fitted (5.095) strictly inside [3, 50] | not on a bound | at the `Nu_Min` floor both fits pin to 3.0 and the `Nu` equality is satisfied by the bound rather than by the code path |

MUTATION MATRIX - every one RUN, by monkeypatching a mutated `calibrate` onto the class and scoring
it on this whole file. Control: 4 passed, 0 failing.

| mutant | killed by | count |
|---|---|---|
| the upper guard dropped, `if 0 < w` (so `w >= len(L)` refits on all of `L`, solo) | off-identity | 1 |
| the refit dropped entirely (the field read and ignored) | off-identity, the split | 2 |
| `Sigma_L` taken from the trailing fit as well | the split, the boundary | 2 |
| `Nu` taken from the trailing fit as well | the split, the boundary | 2 |
| the window taken from the HEAD, `L[:w]` | the split | 1 |
| the residuals follow the window (`res_L` from the trailing fit, `res_D` truncated to it) | delta, the boundary | 2 |

The head-window mutant is the one the BOUNDARY gate cannot see - at `w = len(L) - 1` it drops the
last row instead of the first, which is the same fit to three places. That is the point of having
the split gate and not only a boundary gate on the same branch.
"""
import functools

import numpy as np
import pandas as pd
import pytest

from derivus.stochasticprocess import QuadraticCarryCurveCalibration

#: The world the archive is generated from. `Phi_L` is well short of the platinum archive's 0.9962
#: on purpose - see the anti-placebo table.
TRUE = {'Phi_L': 0.95, 'Sigma_L': 0.0010, 'Phi_D': 0.90, 'Mu_D': -0.0005,
        'Sigma_D': 0.0020, 'Gamma': -0.30, 'Nu': 5.0}

#: The regime shift in the LEVEL's mean, and the trailing window that has to find the second one.
MU_OLD, MU_NEW = 0.015, 0.040
N_OLD, N_NEW, WINDOW = 2000, 300, 252
N = N_OLD + N_NEW


@functools.lru_cache(maxsize=None)
def _archive(seed=7):
    """The two average-carry columns the calibration reads, off an (L, D) world whose LEVEL mean
    steps from 1.5% to 4.0% with `N_NEW` rows to go - the shape equation carrying a live Gamma on
    the same step's dL, as the model does."""
    rng = np.random.default_rng(seed)
    nu = TRUE['Nu']
    eps = rng.standard_normal((N, 2)) * np.sqrt((nu - 2.0) / rng.chisquare(nu, N))[:, None]
    mu = np.r_[np.full(N_OLD, MU_OLD), np.full(N_NEW, MU_NEW)]
    L, D = np.empty(N), np.empty(N)
    L[0], D[0] = MU_OLD, TRUE['Mu_D']
    for i in range(1, N):
        L[i] = mu[i] + TRUE['Phi_L'] * (L[i - 1] - mu[i]) + TRUE['Sigma_L'] * eps[i, 0]
        D[i] = (TRUE['Mu_D'] + TRUE['Phi_D'] * (D[i - 1] - TRUE['Mu_D'])
                + TRUE['Gamma'] * (L[i] - L[i - 1]) + TRUE['Sigma_D'] * eps[i, 1])
    return pd.DataFrame({'ForwardRate.PLATINUM_CARRY,0.5': L - 0.5 * D,
                         'ForwardRate.PLATINUM_CARRY,1.0': L + 0.5 * D},
                        index=pd.bdate_range('2015-01-01', periods=N))


@functools.lru_cache(maxsize=None)
def _fit(window):
    """`window is None` is the param dict with no `Location_Window` key at all."""
    param = {} if window is None else {'Location_Window': window}
    return QuadraticCarryCurveCalibration(model=None, param=param).calibrate(_archive(), 0.0)


def _identical(got, want, what, keys=None):
    """Every value compared with `==` (`np.array_equal` for the sequence ones) - no tolerance, so
    "unchanged" means the same bits and not the same number to five places."""
    assert list(got) == list(want), f'{what}: key set moved, {list(got)} vs {list(want)}'
    for k in keys or list(want):
        same = (np.array_equal(got[k], want[k]) if isinstance(want[k], (list, np.ndarray))
                else got[k] == want[k])
        assert same, f'{what}: {k} moved, {got[k]!r} vs {want[k]!r}'


def test_the_window_is_off_outside_the_strict_bracket():
    """Both ends of `0 < w < len(L)`. No key, `0`, `len(data)` and well past it all return the
    single-window fit BIT-identically - the field is inert unless it names a genuine sub-window.

    Killed by: dropping the upper guard (`if 0 < loc_window`), which refits `L[-w:] == L` as a solo
    AR(1) with its own nu and moves `Phi_L`/`Mu_L` off the joint fit."""
    base = _fit(None)
    for w in (0, N, 5 * N):
        _identical(_fit(w).param, base.param, f'Location_Window={w} is not the legacy fit')
        assert _fit(w).delta.equals(base.delta), f'Location_Window={w} moved the residuals'
    # ...and the field is not inert everywhere, or the above is a tautology
    assert _fit(WINDOW).param['Mu_L'] != base.param['Mu_L']


def test_the_trailing_window_moves_the_level_location_and_nothing_else():
    """The split, on a level whose mean stepped 1.5% -> 4.0% with 300 rows to go.

    `Mu_L` has to land on the TRAILING regime while the single-window fit - which sees 2000 rows of
    the old one - lands materially lower; and every parameter the field does not name has to come
    back EXACTLY equal, `==` rather than `approx`.

    Killed by: dropping the refit (Mu_L stays at 1.90%); taking the window from the HEAD `L[:w]`
    (Mu_L lands on the OLD regime); overriding `Sigma_L` or `Nu` from the trailing fit as well
    (the exact-equality half, which no tolerance-based gate would see)."""
    base, cut = _fit(None), _fit(WINDOW)
    assert base.param['Mu_L'] < 0.03, (
        f"the single-window fit already found the new regime ({base.param['Mu_L']:.4f}) - "
        f'the fixture cannot show a split')
    assert 0.03 < cut.param['Mu_L'] < 0.05, (
        f"the trailing fit put the level mean at {cut.param['Mu_L']:.4f}, not on the {MU_NEW:.1%} "
        f'regime its window covers')
    # the location is the PAIR: phi moves too, off the shift-inflated 0.994 back onto the truth
    assert cut.param['Phi_L'] == pytest.approx(TRUE['Phi_L'], abs=0.03), cut.param['Phi_L']
    assert base.param['Phi_L'] > 0.99, (
        'the regime shift did not inflate the full-window persistence - Phi_L has nothing to move')

    _identical(cut.param, base.param, 'the location refit moved a parameter it does not own',
               keys=['Sigma_L', 'Phi_D', 'Mu_D', 'Sigma_D', 'Gamma', 'Nu',
                     'Reference_Tenors', 'Calibration_DT_Years'])
    # and the fitted nu is strictly inside its bounds, so that equality is the code path, not a bound
    assert 3.0 < base.param['Nu'] < 50.0, base.param['Nu']


def test_the_delta_residuals_stay_full_window():
    """`delta` is what the framework's correlation consolidation sees, so it must remain the
    full-window innovation pair whatever the location was fitted on - a 252-row residual frame
    would reconsolidate rho off a sliver of the sample and silently shorten every other factor's
    overlap with this one.

    Killed by: republishing `res_L` from the trailing fit (which is 251 rows against 2299, so this
    dies on the index before it dies on the values)."""
    base, cut = _fit(None), _fit(WINDOW)
    assert cut.delta.shape == (N - 1, 2) and list(cut.delta.columns) == list(base.delta.columns)
    assert cut.delta.index.equals(base.delta.index), 'the residual index moved'
    assert np.array_equal(cut.delta.values, base.delta.values), (
        f'the residuals moved by {np.abs(cut.delta.values - base.delta.values).max():.3e}')
    assert np.array_equal(cut.correlation, base.correlation)


def test_a_window_one_row_short_of_the_sample_is_the_full_fit():
    """The branch boundary from the inside: `len(data) - 1` is the largest window that still fires
    the refit, and dropping one row of 2300 cannot move the location anywhere. Loose on purpose -
    this is a solo AR(1)-t with its own nu against a joint fit sharing one, so it is the same
    number by a different likelihood rather than the same computation.

    Killed by: any override that reaches past the location - `Sigma_L`, `Nu` or the residuals taken
    from the trailing fit - which shows up HERE as well as in the split gate, because a window of
    2299 rows still refits a solo nu and a solo scale. Not killed by the head-window mutant: at this
    width `L[:w]` and `L[-w:]` differ by one row of 2300, which is what makes the split gate rather
    than this one the gate on the direction."""
    base, edge = _fit(None), _fit(N - 1)
    assert edge.param['Mu_L'] == pytest.approx(base.param['Mu_L'], rel=0.02)
    assert edge.param['Phi_L'] == pytest.approx(base.param['Phi_L'], rel=0.005)
    _identical(edge.param, base.param, 'the near-full window moved a parameter it does not own',
               keys=['Sigma_L', 'Phi_D', 'Mu_D', 'Sigma_D', 'Gamma', 'Nu'])
    assert edge.delta.equals(base.delta)
