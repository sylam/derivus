"""`BasisLinkedSpotCalibration.Max_Persistence` — the cap on the basis GARCH's alpha + beta.

An escalating-variance window (the 2026 squeeze on a 3y fit) drives the joint MLE past a unit
root; simulated, an explosive basis GARCH random-walks the basis to hundreds of $/oz (measured
$32.8/day innovations against a $6.6 calibrated state before the cap). The projection mirrors
GARCHSpotCalibration's: beta is scaled down so the sum lands on the cap, everything else keeps
the fitted values.

Killing mutations, each run against this file: dropping the projection (gate 1 dies on the sum),
capping alpha instead of scaling beta (gate 1's alpha equality dies), applying the cap on the
non-binding fixture too (gate 2 dies bitwise).
"""
import numpy as np
import pandas as pd
import pytest

from derivus import stochasticprocess as sp


def _frame(seed, escalate):
    """A basis + parent-spot frame whose basis innovation variance optionally escalates 5x
    across the sample - the shape that pushes the GARCH MLE past a unit root."""
    rng = np.random.default_rng(seed)
    n = 750
    scale = np.linspace(1.0, 5.0, n) if escalate else np.full(n, 1.5)
    b = np.zeros(n)
    for t in range(1, n):
        b[t] = 0.7 * b[t - 1] + scale[t] * rng.standard_t(5)
    spot = 1500 * np.exp(np.cumsum(0.012 * rng.standard_normal(n)))
    idx = pd.RangeIndex(n)
    return pd.DataFrame({'ObservedBasis.X.Y': b, 'CommodityPrice.X': spot}, index=idx)


def _calibrate(frame, **param):
    cal = sp.BasisLinkedSpotCalibration(
        'BasisLinkedSpotModel', dict({'Slow_Mean_Span': 63, 'GARCH_Innovation': 'Yes'}, **param))
    return cal.calibrate(frame, 0.0).param


def test_the_cap_binds_on_an_escalating_window_by_scaling_beta():
    uncapped = _calibrate(_frame(11, escalate=True), Max_Persistence=10.0)
    assert uncapped['G_Alpha'] + uncapped['G_Beta'] > 0.999, 'fixture no longer escalates past the cap'
    capped = _calibrate(_frame(11, escalate=True))
    assert capped['G_Alpha'] + capped['G_Beta'] == pytest.approx(0.999, abs=1e-12)
    assert capped['G_Alpha'] == uncapped['G_Alpha']          # beta is the scaled one
    for k in ('A', 'Phi', 'Nu', 'Sigma', 'G_Omega', 'Sig2_0'):
        assert capped[k] == uncapped[k], k


def test_the_cap_is_inert_when_the_fit_is_stationary():
    calm = _frame(7, escalate=False)
    assert _calibrate(calm) == _calibrate(calm, Max_Persistence=10.0)
