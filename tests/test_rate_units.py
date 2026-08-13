"""A RATE IS NOT A RATE TIMES TIME, and no expression may confuse the two.

``multiply_by_time`` is a UNITS SWITCH on the curve gathers. ``True`` integrates - the gather
returns ``r * T`` and is ready to exponentiate. ``False`` returns the raw curve value, and what that
MEANS depends on the factor:

  * rate curves (``InterestRate``, ``DividendRate``, and the repo/carry blocks behind
    ``calc_eq_drift`` / ``calc_fx_drift``) hold an annualised rate. The CONSUMER owes the ``dt``.
  * ``SurvivalProb`` holds ``-log`` survival - already integrated (``riskfactors.SurvivalProb``
    documents its Curve as "Negative log survival probability"), so ``False`` is the only correct
    call and multiplying by time would be the defect.
  * ``ForwardPrice`` holds price levels. Same: ``False`` is mandatory, time is meaningless.

So the switch cannot be checked in the abstract - two thirds of its ``False`` call sites owe no
``dt`` at all. Only the first class can be wrong in the way that shipped, which is why this gate
covers the CARRY VOCABULARY and nothing else.

WHAT SHIPPED. ``pv_discrete_barrier_option``'s already-knocked-in leg valued its vanilla with

    total_log_fwd = (drifts + 0.5 * var_per_step).sum(dim=1)
    fwd_to_expiry = spot_block * torch.exp(total_log_fwd)

where ``drifts`` came from ``calc_eq_drift(..., multiply_by_time=False)``: annualised rates summed
with no ``dt``, plus a half-variance whose cancelling subtraction lived on the sibling branch it was
copied from. On a fixture with ``r != q`` and a long observation strip the leg reported +1432% of the
option's value on every ``all_hit`` row, and the whole suite was green.

WHAT THIS GATE IS. A taint pass, per function, over ``derivus/``: a name that holds an annualised
carry rate may not reach ``exp`` without being multiplied by a year fraction first. Taint is seeded
where a carry-vocabulary name is bound from a ``multiply_by_time=False`` rate gather, and wherever a
carry-vocabulary name is bound at all - a PARAMETER (the strip crosses into ``sim_spot_oss`` /
``sim_spot_tarf`` / ``sim_spot`` / ``forward_vols`` inside a ``theta`` tuple, and no AST can follow
it there) or a ``for`` target (the per-row strip inside the fixing loops). The name is the handoff,
so the gate holds the name to its meaning. Taint is discharged by a product with a time-like name,
or by ``total_log_forward``, the one sanctioned consumer.

That makes it a NAMING CONTRACT WITH TEETH, not type inference, and the honest cost is two
vocabularies (``CARRY_VOCAB``, ``TIME_NAMES``) that a new spelling has to be added to. Both are
below; the failure message says so. Measured: a legitimate ``carry * step_years`` fires until
``step_years`` is declared. That is the price of the check, and it is a two-word edit.

MUTATION MATRIX (applied to the current tree's source, scanned, discarded):

    mutation                                                            verdict
    (1)  total_log_forward inlined back to a bare `.sum()`              KILLED (the shipped defect)
    (2)  sim_spot_oss builds carry_int without dt                       KILLED
    (3)  forward_vols drops fixing_t                                    KILLED
    (4)  CommodityFutureDeal's forward drops T_t_years                  KILLED
    (5)  pv_MC_Tarf's fwd_drift drops dt                                KILLED
    (6)  pv_MC_Tarf's HN b_step drops dt                                KILLED
    (7)  pv_MC_AutoCallSwap's drift drops dt                            KILLED
    (8)  CONTROL: the product commuted, `dt * carry`                    no fire (correct)
    (9)  CONTROL: a legitimate `carry * step_years`                     KILLED - the vocabulary cost
    (10) the sloped-carry strip: `carry * dt` vs differencing           SURVIVED - see below
    (11) total_log_forward's OWN body drops `times`                     SURVIVED - see below
    (12) forward_carry_rate returns its `carry_rate` undifferenced      SURVIVED - see below

WHAT IT CANNOT SEE, stated so nobody trusts it further than it goes:

  * VALUE errors that keep the units. (10) and (12) are the same shape - the interval carry built as
    ``b(T_j) * dt_j`` rather than ``b(T_j)T_j - b(T_j-1)T_j-1``, correct only on a flat curve. Both
    spellings are a rate times a time, so no units check can separate them. That defect WAS live in
    ``sim_spot_oss`` and worth -20.10% of a never-knocking barrier on a sloped curve; it is now
    ``pricing.forward_carry_rate``, and it is pinned by a CONSISTENCY assertion in
    ``test_sibling_forward_agreement.py``, which is the right instrument for it. This gate is blind
    to it either way, and that is why the two live side by side.
  * ``total_log_forward`` itself: it is the discharger, so the gate stops at its call. Its body is
    held by ``test_total_log_forward_is_rank_polymorphic``'s einsum assertion instead.
  * The mirror error - integrating twice. No site in the tree binds a carry-vocabulary name from an
    INTEGRATING gather, so a symmetric rule would have nothing to bite on and would be a placebo.
  * Anything crossing a call boundary under a name outside the vocabulary.
"""
import ast
import os

PKG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'derivus')

RATE_GATHERS = {'calc_eq_drift', 'calc_fx_drift', 'gather_weighted_curve'}
CARRY_VOCAB = {'carry', 'carry_rate', 'carry_rates', 'drifts', 'fwd_drifts'}
TIME_NAMES = {'dt', 'times', 'sample_ts', 'delta_t', 'full_t', 'fixing_t', 'tau', 'expiry',
              'expiry_years', 'T_t_years', 'tenor_in_days', 'year_frac', 'rem_exp', 'cum_t'}
FACTORY_ATTRS = {'new', 'new_tensor', 'new_zeros', 'new_ones', 'new_empty', 'new_full',
                 'zeros_like', 'ones_like', 'empty_like', 'full_like',
                 'shape', 'dtype', 'device', 'size', 'numel', 'dim'}
DISCHARGERS = {'total_log_forward'}
SINKS = {'exp', 'expm1'}


def fname(call):
    f = call.func
    return f.attr if isinstance(f, ast.Attribute) else getattr(f, 'id', None)


def is_time(node):
    """A year fraction by name - the only thing that may discharge an annualised rate."""
    return bool(TIME_NAMES & ({n.id for n in ast.walk(node) if isinstance(n, ast.Name)} |
                              {a.attr for a in ast.walk(node) if isinstance(a, ast.Attribute)}))


def carries(node, tainted):
    """Does this expression still hold an annualised rate that nobody has multiplied by a time?"""
    if isinstance(node, ast.Name):
        return node.id in tainted
    if isinstance(node, ast.Attribute):
        return node.attr not in FACTORY_ATTRS and carries(node.value, tainted)
    if isinstance(node, ast.Subscript):
        return carries(node.value, tainted)
    if isinstance(node, ast.BinOp):
        l, r = carries(node.left, tainted), carries(node.right, tainted)
        if isinstance(node.op, ast.Mult) and (
                (l and is_time(node.right)) or (r and is_time(node.left))):
            return False
        return l or r
    if isinstance(node, ast.Call) and fname(node) in DISCHARGERS | FACTORY_ATTRS:
        return False
    return any(carries(c, tainted) for c in ast.iter_child_nodes(node))


def binds_rate(value):
    return any(fname(c) in RATE_GATHERS and any(
        k.arg == 'multiply_by_time' and k.value.value is False for k in c.keywords)
        for c in ast.walk(value) if isinstance(c, ast.Call))


def targets_of(stmt):
    """Plain names bound by this statement, unpacking nested tuples; `a[i] = ...` binds nothing,
    so a subscripted write can neither taint nor clear the name it writes through."""
    out, stack = set(), ([stmt.target] if isinstance(stmt, (ast.AugAssign, ast.For))
                         else list(stmt.targets))
    while stack:
        t = stack.pop()
        stack.extend(t.elts) if isinstance(t, (ast.Tuple, ast.List)) else None
        if isinstance(t, ast.Name):
            out.add(t.id)
    return out


def own(fn):
    """This function's own nodes; a nested def is scanned in its own right, under its own params."""
    out, stack = [], list(fn.body)
    while stack:
        n = stack.pop()
        out.append(n)
        if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            stack.extend(ast.iter_child_nodes(n))
    return out


def scan_src(src, name):
    """Every `exp` of an undischarged annualised rate, as `file:line in function`."""
    bad = []
    for fn in [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef)]:
        nodes = own(fn)
        # the vocabulary declares units wherever the name is BOUND - parameter (the strip crosses
        # into the sim functions inside a theta tuple) or `for` target (the per-row strip)
        tainted = ({a.arg for a in fn.args.args + fn.args.kwonlyargs} | set().union(
            *[targets_of(n) for n in nodes if isinstance(n, ast.For)] or [set()])) & CARRY_VOCAB
        for stmt in sorted([n for n in nodes if isinstance(n, (ast.Assign, ast.AugAssign))],
                           key=lambda n: n.lineno):
            names = targets_of(stmt)
            hot = carries(stmt.value, tainted) or (names & CARRY_VOCAB and binds_rate(stmt.value))
            tainted = (tainted | names) if hot else (tainted - names)
        for call in [n for n in nodes if isinstance(n, ast.Call) and fname(n) in SINKS]:
            if any(carries(a, tainted) for a in call.args):
                bad.append('%s:%d in %s' % (name, call.lineno, fn.name))
    return sorted(bad)


def scan_package():
    return sum([scan_src(open(os.path.join(PKG, f)).read(), f)
                for f in sorted(os.listdir(PKG)) if f.endswith('.py')], [])


SHIPPED = '''
def pv_discrete_barrier_option(shared, deal_data, spot, b, tau):
    drifts = utils.calc_eq_drift(zero, divi, fixings, t_block, shared, multiply_by_time=False)
    sample_ts = drifts.new(np.hstack([fixing_block[:, 0, np.newaxis], np.diff(fixing_block, axis=1)]))
    var_per_step = (vols * vols).unsqueeze(1) * sample_ts.unsqueeze(2)
    total_log_fwd = (drifts + 0.5 * var_per_step).sum(dim=1)
    fwd_to_expiry = spot_block * torch.exp(total_log_fwd)
    return fwd_to_expiry
'''
FIXED = SHIPPED.replace(
    '''    var_per_step = (vols * vols).unsqueeze(1) * sample_ts.unsqueeze(2)
    total_log_fwd = (drifts + 0.5 * var_per_step).sum(dim=1)''',
    '''    total_log_fwd = total_log_forward(drifts, sample_ts)''')


def test_the_gate_kills_the_defect_it_was_written_for():
    """The scanner on the shipped expression and on its replacement. A gate that cannot be shown to
    fire is a placebo, and this one carries its own witness so it stays honest for free - no fixture,
    no market data, no import of the pricer. Both strings are the real code, lifted from a87e3b5^
    and a87e3b5."""
    assert scan_src(SHIPPED, 'shipped.py') == ['shipped.py:7 in pv_discrete_barrier_option']
    assert scan_src(FIXED, 'fixed.py') == []


def test_no_annualised_rate_reaches_exp_without_its_time():
    """The gate proper, over every module in `derivus/`.

    A hit is one of two things. Either a carry rate really is being exponentiated with no `dt` - the
    defect, fix the expression or route it through `total_log_forward`. Or the multiplication IS
    there under a year-fraction name this file has never seen, in which case add the name to
    `TIME_NAMES` above. Nothing else makes it go away, which is the point."""
    offenders = scan_package()
    assert not offenders, (
        'annualised rate exponentiated with no year fraction: %s\n'
        'multiply by the fixing interval (or call total_log_forward); if the time IS there under a '
        'new name, add it to TIME_NAMES in %s' % (offenders, os.path.basename(__file__)))
