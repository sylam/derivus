"""`utils.BivN`, the lower-tail bivariate normal, against a thirty-digit oracle.

THE ORACLE is mpmath's `quad` of int_-inf^P phi(x) Phi((Q - rho x)/s) dx at 30 digits, s being
sqrt(1 - rho^2), subdivided where the inner Phi turns over - scipy's own bivariate cdf is Genz's
integration at a 1e-5 tolerance and is not an independent reading of it. The grid crosses the 0.925
branch in both signs; the pin is the algorithm's accuracy claim, not a regression of its output.

THE GRADIENT is closed form and needs no oracle: dPhi2/dP = phi(P) Phi((Q - rho P)/s) and
dPhi2/drho is the bivariate density itself (Plackett), which is what a caller differentiating a
barrier through the correlation is entitled to at |rho| = 0.9999.
"""
import os
import sys

import torch

# reference-derivus shadow-import guard (MEMORY): pin the package under test to THIS repo.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from derivus import utils

RHO = (0.0, 0.6, -0.6, 0.93, -0.93, 0.9999, -0.9999)

#: P(X <= P, Y <= Q) at each of RHO, keyed by (P, Q)
ORACLE = {
    (-2.0, -2.0): (0.0005175685036595643, 0.005499975977136193, 3.143618040753213e-07,
                   0.01483313051748325, 7.881363321063941e-29, 0.022445528154435478, 0.0),
    (-2.0, 0.0): (0.011375065974089604, 0.021804363898214447, 0.0009457680499647598,
                  0.022750131229603166, 7.185760405586272e-10, 0.02275013194817921, 0.0),
    (-2.0, 1.5): (0.02123025930422412, 0.022746158621496704, 0.012244428471443277,
                  0.02275013194817921, 0.0013249746718981212, 0.02275013194817921, 1.5384649593569192e-278),
    (0.0, -2.0): (0.011375065974089604, 0.021804363898214447, 0.0009457680499647598,
                  0.022750131229603166, 7.185760405586272e-10, 0.02275013194817921, 0.0),
    (0.0, 0.0): (0.25, 0.35241638234956674, 0.1475836176504333,
                 0.44009670829099257, 0.05990329170900741, 0.4977491904525954, 0.002250809547404597),
    (0.0, 1.5): (0.46659639936557096, 0.4946257261265028, 0.4385670726046391,
                 0.49999922075561215, 0.43319357797552976, 0.5, 0.4331927987311419),
    (1.5, -2.0): (0.02123025930422412, 0.022746158621496704, 0.012244428471443277,
                  0.02275013194817921, 0.0013249746718981212, 0.02275013194817921, 1.3800171084647451e-278),
    (1.5, 0.0): (0.46659639936557096, 0.4946257261265028, 0.4385670726046391,
                 0.49999922075561215, 0.43319357797552976, 0.5, 0.4331927987311419),
    (1.5, 1.5): (0.8708487996036616, 0.8891798860691565, 0.8664263080068073,
                 0.9140038827008531, 0.8663855974622838, 0.9324620815594984, 0.8663855974622838)}


def _grid(dtype):
    """The banked grid as three broadcastable tensors and the oracle beside them."""
    pts = [(p, q, r) for p, q in ORACLE for r in RHO]
    return ([torch.tensor([x[i] for x in pts], dtype=dtype) for i in range(3)],
            torch.tensor([v for row in ORACLE.values() for v in row], dtype=torch.float64))


def test_the_lower_tail_is_the_thirty_digit_oracle():
    """Genz's algorithm is a double-precision one: the double reads the oracle to 1e-13 (measured
    2e-16, the rounding of the answer itself) and the single to its own last bits."""
    args, want = _grid(torch.float64)
    worst = float((utils.BivN(*args).double() - want).abs().max())
    assert worst < 1e-13, 'double reads the oracle %.3e away' % worst
    args, want = _grid(torch.float32)
    worst = float((utils.BivN(*args).double() - want).abs().max())
    assert worst < 1e-6, 'single reads the oracle %.3e away' % worst


def test_the_gradient_stands_up_where_the_correlation_is_extreme():
    """The three partials against their closed forms at |rho| = 0.9999, where the tail expansion
    runs and 1 - rho^2 is 2e-4. |rho| = 1 - a Barrier_Limit_Date ON the expiry date - is asked for
    finiteness only: the floor on 1 - rho^2 leaves the value there good to 1e-9, not to 1e-13."""
    p, q, r = (torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in (
        [-3.0, 0.0, 3.0, -3.0, 0.0, 3.0], [3.0, 1.0, 3.0, 3.0, 1.0, 3.0],
        [0.9999, 0.9999, 0.9999, -0.9999, -0.9999, -0.9999]))
    got = torch.autograd.grad(utils.BivN(p, q, r).sum(), (p, q, r))
    s = torch.sqrt(1.0 - r * r)
    want = (utils.norm_pdf(p) * utils.norm_cdf((q - r * p) / s),
            utils.norm_pdf(q) * utils.norm_cdf((p - r * q) / s),
            torch.exp(-(p * p - 2.0 * r * p * q + q * q) / (2.0 * s * s)) / (6.283185307179586 * s))
    for name, x, y in zip(('dP', 'dQ', 'drho'), got, want):
        assert torch.allclose(x, y, rtol=1e-10, atol=1e-12), (name, x, y)
    for corner in (1.0, -1.0):
        r = torch.full_like(p, corner).requires_grad_(True)
        g = torch.autograd.grad(utils.BivN(p, q, r).sum(), (p, q, r))
        assert all(bool(torch.isfinite(x).all()) for x in g), corner


def test_the_backward_is_the_closed_form_and_differentiates_again():
    """The three partials are taken in closed form rather than off the quadrature's tape: over the
    banked grid, against their explicit spelling, and a second derivative through them - which the
    Hessian path asks for when it differentiates a barrier's gradient."""
    p, q, r = (x.requires_grad_(True) for x in _grid(torch.float64)[0])
    got = torch.autograd.grad(utils.BivN(p, q, r).sum(), (p, q, r), create_graph=True)
    s = torch.sqrt(1.0 - r * r)
    want = (utils.norm_pdf(p) * utils.norm_cdf((q - r * p) / s),
            utils.norm_pdf(q) * utils.norm_cdf((p - r * q) / s),
            torch.exp(-(p * p - 2.0 * r * p * q + q * q) / (2.0 * s * s)) / (6.283185307179586 * s))
    for name, x, y in zip(('dP', 'dQ', 'drho'), got, want):
        assert torch.allclose(x, y, rtol=1e-10, atol=1e-14), (name, x, y)
    second = torch.autograd.grad(sum(x.sum() for x in got), (p, q, r))
    assert all(bool(torch.isfinite(x).all()) for x in second), second
