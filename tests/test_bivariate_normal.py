"""How wrong is the Tsay-Ke bivariate normal, measured against the exact one we keep for the purpose?

`utils.ApproxBivN` is the error-function approximation of the standard bivariate normal integral
(Tsay & Ke); it is what the partial-barrier pricer calls, four times per leg, in
`pricing.py:488-489`. `utils.BivN` is the exact scipy `multivariate_normal.cdf`, vectorised - too
slow for the pricing path, and called by nothing else in the repo. This module is why it is kept:
it is the ORACLE that pins the approximation's error, and it earns its place here or nowhere.

The comment on `ApproxBivN` claims "accurate to around 4 decimal places". MEASURED, on a 65x65
grid of (P, Q) over [-4, 4], the claim holds only at high correlation and is off by most of a
decimal place as rho -> 0:

    |rho|      0.95        0.70        0.40        0.15
    measured   1.085e-04   3.093e-04   5.164e-04   7.970e-04     <- max abs error vs BivN
    pinned     1.25e-04    3.6e-04     6.0e-04     9.2e-04       <- TOL below, ~15% headroom

Headroom is deliberately thin because the fitted constants c1, c2 sit at an optimum, so the error
responds to them only at SECOND order: perturbing c1 by 0.1% leaves every tolerance above intact
(it even improves two of them), 0.5% trips the |rho| <= 0.4 rows, 1% trips all four.

The error is even in rho (the +rho and -rho columns agree to five digits) and keeps growing as
|rho| falls, to a floor of 1.086e-03 below |rho| ~ 1e-3.

That floor is the seam. At rho EXACTLY 0 we get a = -rho / sqrt(1 - rho^2) = 0, all four cases are
false, and the function returns its default `norm_cdf(P) * norm_cdf(Q)` - which is not an
approximation at all but the exact answer, so the error there is 0 to machine precision. Take rho
off zero by 1e-6 and the case-3/4 branch engages, whose a -> 0 limit is `norm_cdf(Q) * Phi_tilde(P)`
with `Phi_tilde(P) = 1 - exp(P * (sqrt(2) * c1 + P * c2) / 2) / 2`, the UNIVARIATE Tsay-Ke CDF. So
`ApproxBivN` steps by up to 1.086e-03 across rho = 0, and that step is exactly
sup_P |Phi_tilde(P) - Phi(P)| = 1.102e-03. It is a real discontinuity, but it is not a spike: it is
the same univariate error the whole low-|rho| region already carries, and the two non-zero branches
join across zero to 3.1e-07 at rho = +-1e-06. Both facts are gated below - a seam bug that broke
the dispatch would separate those two branches long before the coarse accuracy grid noticed.

Two further properties are pinned as MEASURED, not as identities, because the approximation does
not satisfy either exactly: it is not symmetric in (P, Q) (up to 1.206e-03 at |rho| = 0.15, worse
than its own accuracy), and it violates the Frechet bounds max(0, Phi(P) + Phi(Q) - 1) <= C <=
min(Phi(P), Phi(Q)) by up to 7.7e-04. Non-negativity, in contrast, held everywhere probed.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch

from derivus import utils

DTYPE = torch.float64
GRID = np.linspace(-4.0, 4.0, 65)
#: max abs error vs the scipy oracle, keyed by |rho|: measured on this grid, plus ~15% headroom
TOL = {0.95: 1.25e-4, 0.70: 3.6e-4, 0.40: 6.0e-4, 0.15: 9.2e-4}
#: max |ApproxBivN(P, Q, rho) - ApproxBivN(Q, P, rho)| - a wart, pinned so it cannot grow
ASYM = {0.95: 1.3e-4, 0.70: 3.7e-4, 0.40: 8.1e-4, 0.15: 1.35e-3}
RHOS = [s * r for r in TOL for s in (1.0, -1.0)]
SEAM_STEP = 1.2e-3      # measured 1.086e-03: the univariate Tsay-Ke error, see the module docstring

_P, _Q = (x.ravel() for x in np.meshgrid(GRID, GRID, indexing='ij'))
P = torch.tensor(_P, dtype=DTYPE)
Q = torch.tensor(_Q, dtype=DTYPE)


def approx(rho, p=P, q=Q):
    """ApproxBivN with rho broadcast to the shape of p, as numpy."""
    return utils.ApproxBivN(p, q, torch.full_like(p, rho)).numpy()


def exact(rho):
    """The oracle: scipy's exact bivariate normal CDF on the same grid."""
    return np.asarray(utils.BivN(_P, _Q, rho), dtype=float)


@pytest.mark.parametrize('rho', RHOS)
def test_accuracy_against_scipy_oracle(rho):
    """The headline gate: pinned per-|rho| tolerances, tightest where the approximation is best."""
    assert np.abs(approx(rho) - exact(rho)).max() < TOL[abs(rho)]


def test_rho_zero_returns_the_exact_gaussian_product():
    """At rho = 0 every case is false and the default branch is the exact answer, bit for bit."""
    default = (utils.norm_cdf(P) * utils.norm_cdf(Q)).numpy()
    assert np.abs(approx(0.0) - default).max() == 0.0
    assert np.abs(approx(0.0) - exact(0.0)).max() < 1e-12


def test_rho_zero_seam_is_a_bounded_step_not_a_spike():
    """The case dispatch flips either side of rho = 0; the branches must still meet there."""
    eps = 1e-6
    plus, minus, zero = approx(eps), approx(-eps), approx(0.0)
    assert np.abs(plus - minus).max() < 1e-6                     # the two branches join across zero
    assert np.abs(plus - zero).max() < SEAM_STEP                 # ... but step off the exact default
    assert np.abs(minus - zero).max() < SEAM_STEP
    for r in (eps, -eps):
        assert np.abs(approx(r) - exact(r)).max() < SEAM_STEP    # the step IS the whole error there


@pytest.mark.parametrize('rho', RHOS)
def test_symmetry_holds_only_approximately(rho):
    """C(P, Q) = C(Q, P) for the true integral; the approximation misses it by more than its error."""
    assert np.abs(approx(rho) - approx(rho, Q, P)).max() < ASYM[abs(rho)]


@pytest.mark.parametrize('rho', RHOS)
def test_probability_and_frechet_bounds(rho):
    """A CDF value: non-negative (exactly), and inside the Frechet bounds up to the pinned slack."""
    a, cp, cq = approx(rho), utils.norm_cdf(P).numpy(), utils.norm_cdf(Q).numpy()
    assert a.min() >= 0.0
    assert (a - np.minimum(cp, cq)).max() < 1e-3
    assert (np.maximum(0.0, cp + cq - 1.0) - a).max() < 1e-3


def test_call_site_shapes_and_broadcasting():
    """pricing.py passes 0-dim rho against shaped P, Q; the in-place case writes must survive that."""
    p, q = P[:64].reshape(8, 8), Q[:64].reshape(8, 8)
    shaped = utils.ApproxBivN(p, q, torch.tensor(0.4, dtype=DTYPE))
    assert shaped.shape == (8, 8) and shaped.dtype == DTYPE
    assert (shaped.reshape(-1) - torch.tensor(approx(0.4, P[:64], Q[:64]), dtype=DTYPE)).abs().max() == 0.0
