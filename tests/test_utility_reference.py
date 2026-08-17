"""`_utility_wrap_signed`'s two orthogonal dials: the `Reference_Wealth` the utility is measured
against, and the gain-side curvature of the huber shape.

Both default to EXACTLY today's arithmetic, so the first gate here is bit-identity — not
`allclose`, `torch.equal` — against `_pre_change_utility` below, a copy of the transform as it
stood at 9170870 that was verified bit-exact against the unmodified tree before the edit landed.
A reference that changes a trained policy's labels by a rounding step is a silent recalibration of
every checkpoint on disk, so "off" has to mean the same floats, not nearly the same floats.

The three fallbacks are the other half. `Reference_Wealth`/`Up_Aversion`/`Up_Knee` are each spelled
THREE times - the `F` row in `calculation`, the `.get` in `hedge_runtime`, and the `.get` on the
normalized runtime inside `_utility_wrap_signed` itself. `test_hmc_declared_knobs` holds the first
two together; the third is downstream of the JSON boundary and no gate reached it, which is the
same two-published-defaults class one level further in.

KILL MATRIX, mutate-then-verify, every row RUN. `a` = the two bit-identity gates, `b` = the
reference shift, `c` = the gain wing, `d` = C¹, `e` = the two default gates; `K` =
test_hmc_declared_knobs, `S` = test_symlog_unit.

  mutant                                                          gates that FAIL
  reference on huber only (symlog/cara keep x_dollars/c)          b
  gain penalty charged on `loss` instead of `gain`                c, d
  gain linear arm drops the level constant (au*k*k)               d
  reference subtracted AFTER /c  (x = x_dollars/c - R)            b, d
  bundle reads `upside_knee` - a key no F row names               c, d, e
  bundle `up_knee` fallback 0.15 -> 0.20                          e
  bundle `reference_wealth` fallback 0.0 -> 1.0                   a, e, S
  bundle `up_aversion` fallback 0.0 -> 0.5                        a, b, e, S
  declared `Up_Knee` default 0.15 -> 0.2                          e, K
  declared `Up_Aversion` F row removed                            e, K

Two rows read the other way and are the reason for the split. `gain penalty on loss` survives (a)
because a+ = 0 zeroes the penalty whichever wing it lands on, and `up_knee 0.20` survives (c)
because that gate names the knee explicitly. Bit-identity gates the OFF switch; only the fallback
gates see a moved default.
"""
import ast
import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from derivus import calculation, hedge_bundle, hedge_runtime
from derivus.hedge_bundle import _utility_wrap_signed
from test_symlog_unit import make_util_runtime

C = 5.0e5
#: Shape params deliberately off their defaults, so a gate that silently read a default would fail.
HUBER = {"huber_aversion": 6.0, "huber_delta": 1.0}
CARA = {"cara_gamma": 1.5}
SHAPES = {"symlog": {}, "huber": HUBER, "cara": CARA}


@pytest.fixture(autouse=True)
def _float64():
    """Bit-identity and finite differences both want the wide type; restore it so the float32
    default does not leak into whatever module the runner reaches next."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


def _pre_change_utility(x_dollars, obj):
    """The transform as of 9170870, before the reference and the gain wing - the bit-identity
    baseline. Verified `torch.equal` against the unmodified `_utility_wrap_signed` over the grid
    below for all three shapes prior to the edit."""
    c = obj["utility_scale"]
    x = x_dollars / float(c)
    shape = obj["object"]
    if shape == "asymmetricutility_symlog":
        return torch.sign(x) * torch.log1p(x.abs())
    if shape == "asymmetricutility_huber":
        a = float(obj.get("huber_aversion", 2.5))
        d = float(obj.get("huber_delta", 1.0))
        loss = (-x).clamp(min=0.0)
        quad = a * loss * loss
        lin = a * d * d + 2.0 * a * d * (loss - d)
        return x - torch.where(loss <= d, quad, lin)
    g = float(obj.get("cara_gamma", 1.0))
    return (1.0 - torch.exp(-g * x)) / g


def wealth_grid(c=C, half_width=50.0, n=1001):
    """Signed dollar wealth over x = W/c in [-half_width, +half_width], with both huber knees and
    the origin landed EXACTLY (a linspace straddles a knee, it does not sit on one)."""
    x = torch.cat([torch.linspace(-half_width, half_width, n),
                   torch.tensor([-1.0, -0.15, 0.0, 0.15, 1.0])])
    return x * c


def test_defaults_are_bit_identical():
    """(a) With all three dials at their defaults every shape returns the SAME floats as the
    pre-change transform - `torch.equal`, not `allclose`. Subtracting +0.0 and a gain penalty that
    evaluates to +0.0 are IEEE no-ops, which is why "off" can be stated this strongly."""
    W = wealth_grid()
    for shape, params in SHAPES.items():
        rt = make_util_runtime(shape, C, **params)
        got = _utility_wrap_signed(W, rt)
        assert torch.equal(got, _pre_change_utility(W, rt["objective"])), shape
    print("test_defaults_are_bit_identical: PASS  (symlog / huber / cara, %d points)" % W.numel())


def test_defaults_leave_the_gradient_bit_identical():
    """(a) The AAD path too: the labels the solver fits are gradients, so a dial that is off must
    add no d/dW either. The zero-curvature gain arm contributes exactly 0, not ~0."""
    W = wealth_grid(n=257)
    for shape, params in SHAPES.items():
        rt = make_util_runtime(shape, C, **params)
        w = W.clone().requires_grad_(True)
        (got,) = torch.autograd.grad(_utility_wrap_signed(w, rt).sum(), w)
        w0 = W.clone().requires_grad_(True)
        (ref,) = torch.autograd.grad(_pre_change_utility(w0, rt["objective"]).sum(), w0)
        assert torch.equal(got, ref), shape
    print("test_defaults_leave_the_gradient_bit_identical: PASS  (symlog / huber / cara)")


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_reference_wealth_shifts_every_shape(shape):
    """(b) `Reference_Wealth` R is a pure translation of the argument, for EVERY shape:
    u_new(W; R) == u_old(W - R), bit for bit. A reference wired into the huber branch alone leaves
    symlog and cara on the old argument and fails here; one applied after the /c scaling shifts by
    R c-units instead of R dollars and fails by a factor of c."""
    W, R = wealth_grid(), 1.25e6
    rt = make_util_runtime(shape, C, reference_wealth=R, **SHAPES[shape])
    got = _utility_wrap_signed(W, rt)
    ref = _pre_change_utility(W - R, make_util_runtime(shape, C, **SHAPES[shape])["objective"])
    assert torch.equal(got, ref), shape
    print("test_reference_wealth_shifts_every_shape[%s]: PASS  (R = $%.3g)" % (shape, R))


def test_up_aversion_curves_the_gain_wing():
    """(c) With `Up_Aversion` > 0 the gain arm is strictly concave inside the knee, goes linear
    beyond it with marginal utility 1 - 2*a+*k (still positive - the upside is KEPT, only taxed),
    and the loss arm is untouched to the last bit."""
    au, k, h = 0.8, 0.5, 1.0e-3
    on = make_util_runtime("huber", C, up_aversion=au, up_knee=k, **HUBER)
    off = make_util_runtime("huber", C, **HUBER)

    # strictly concave inside the knee: second difference of a - a+ x^2 arm is -2 a+ h^2 < 0
    x = torch.linspace(0.05 * k, 0.95 * k, 64)
    u = [_utility_wrap_signed((x + s * h) * C, on) for s in (-1, 0, 1)]
    second = u[2] - 2.0 * u[0 + 1] + u[0]
    assert (second < 0).all(), "gain arm is not concave inside the knee"
    assert torch.allclose(second, torch.full_like(second, -2.0 * au * h * h), atol=1e-12)

    # deep gains: FD marginal utility per c-unit is exactly the linear arm's slope
    deep = torch.linspace(4.0 * k, 40.0 * k, 32) * C
    fd = (_utility_wrap_signed(deep + h * C, on) - _utility_wrap_signed(deep - h * C, on)) / (2 * h)
    assert torch.allclose(fd, torch.full_like(fd, 1.0 - 2.0 * au * k), atol=1e-9), fd[:3]
    assert float(1.0 - 2.0 * au * k) > 0.0, "the test's own dials would invert the gain arm"

    # the loss arm never sees the gain penalty
    loss_side = wealth_grid()[wealth_grid() <= 0.0]
    assert torch.equal(_utility_wrap_signed(loss_side, on), _utility_wrap_signed(loss_side, off))
    print("test_up_aversion_curves_the_gain_wing: PASS  (deep-gain slope %.4f)" % (1 - 2 * au * k))


def test_the_shape_is_c1_at_the_reference_and_both_knees():
    """(d) The shape is C¹ at all three breakpoints - the reference itself and the two knees -
    stated as a LIMIT, because a fixed tolerance cannot tell the three failure modes apart. The
    one-sided slope gap of a piecewise-quadratic C¹ function is O(h) and shrinks with the step; a
    kink leaves it CONSTANT, and a linear arm missing its level-matching constant leaves a value
    jump J whose gap DIVERGES as J/h. So: the gap must fall a hundredfold when h does, and the
    central difference must converge to the analytic slope from both sides."""
    a, d, au, k, R = 6.0, 1.0, 0.8, 0.5, 3.0e5
    rt = make_util_runtime("huber", C, huber_aversion=a, huber_delta=d,
                           up_aversion=au, up_knee=k, reference_wealth=R)
    for name, x0, slope in (("reference", 0.0, 1.0),
                            ("loss knee", -d, 1.0 + 2.0 * a * d),
                            ("gain knee", k, 1.0 - 2.0 * au * k)):
        W0 = R + x0 * C
        gaps = []
        for h in (1.0e-3, 1.0e-5):                                          # c-units
            u = {s: float(_utility_wrap_signed(torch.tensor([W0 + s * h * C]), rt))
                 for s in (-1.0, 0.0, 1.0)}
            gaps.append(abs((u[1.0] - u[0.0]) / h - (u[0.0] - u[-1.0]) / h))
            central = (u[1.0] - u[-1.0]) / (2.0 * h)
            assert abs(central - slope) < 20.0 * h, (
                "%s: slope %.9f vs %.9f at h=%g" % (name, central, slope, h))
        assert gaps[1] < 0.02 * gaps[0], (
            "%s: one-sided slope gap %.3e -> %.3e over a 100x smaller step" % (name, *gaps))
    print("test_the_shape_is_c1_at_the_reference_and_both_knees: PASS  (reference / both knees)")


def _bundle_fallbacks():
    """`{runtime key: default}` for every `obj.get(key, const)` inside `_utility_wrap_signed` -
    the third place a default is published, downstream of the JSON boundary the declared-knobs
    gate walks."""
    tree = ast.parse(inspect.getsource(hedge_bundle._utility_wrap_signed).lstrip())
    return {node.args[0].value: node.args[1].value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get" and len(node.args) == 2
            and isinstance(node.args[0], ast.Constant) and isinstance(node.args[1], ast.Constant)}


@pytest.mark.parametrize("json_key,runtime_key", [
    ("Reference_Wealth", "reference_wealth"), ("Huber_Aversion", "huber_aversion"),
    ("Huber_Delta", "huber_delta"), ("Up_Aversion", "up_aversion"), ("Up_Knee", "up_knee"),
    ("CARA_Gamma", "cara_gamma")])
def test_one_default_per_objective_knob(json_key, runtime_key):
    """(e) The declared `F` row, the boundary's `.get` fallback and the bundle's own `.get`
    fallback are ONE number. `test_hmc_declared_knobs` pins the first pair; the bundle reads the
    normalized runtime, so its fallback is invisible to that walk and only fires for a hand-built
    runtime - which is exactly what every unit gate and every ad-hoc script constructs."""
    objective = next(f for f in calculation.HedgeMonteCarlo.fields if f.key == 'Hedging_Problem')
    objective = next(f for f in objective.sub_fields if f.key == 'Objective')
    declared = next(f for f in objective.sub_fields if f.key == json_key).default
    source = inspect.getsource(hedge_runtime.construct_hedge_runtime)
    boundary = next(node.args[1].value for node in ast.walk(ast.parse(source.lstrip()))
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "get" and len(node.args) == 2
                    and isinstance(node.args[0], ast.Constant) and node.args[0].value == json_key)
    bundle = _bundle_fallbacks()[runtime_key]
    assert declared == boundary == bundle, (
        "%s publishes %r (declared) / %r (boundary) / %r (bundle)"
        % (json_key, declared, boundary, bundle))
    print("test_one_default_per_objective_knob[%s]: PASS  (%r)" % (json_key, declared))


def test_the_bundle_reads_exactly_the_declared_shape_knobs():
    """(e), the converse: a knob `_utility_wrap_signed` honours but no `F` row names is a shape
    no schema-authored job can reach, which is how the gain wing would have shipped invisible."""
    assert set(_bundle_fallbacks()) == {
        "reference_wealth", "huber_aversion", "huber_delta", "up_aversion", "up_knee",
        "cara_gamma"}, sorted(_bundle_fallbacks())
    print("test_the_bundle_reads_exactly_the_declared_shape_knobs: PASS")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
