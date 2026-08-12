"""`HedgeMonteCarlo.Hedging_Problem` declares the knobs the hedge stack reads, held to the reads
in both directions.

The sub-containers were bare `F(..., 'Container', default={})` while `construct_hedge_runtime`
read forty knobs out of them with `.get(key, fallback)`. That is the mismatch class the
market-price and calculation stores are already gated for, one level down: a knob read with a
fallback publishes TWO defaults, an author who fills a panel in gets the DECLARED one and a job
that omits the key gets the ENGINE'S, and nothing raises. Here it was worse than a mismatch -
there was no declared side at all, so no panel, no validator and no schema-authored job could
name `DiffV2_Fit_Iters` or `Force_Flat_At_End` in the first place.

The blocks are reached by ALIAS, not by name: `evaluator_config = hedging_problem["Evaluator"]`
and then `evaluator_config.get(...)` forty lines away, or the same dict arriving as the parameter
of `_solver_config`. So the walk resolves an expression to a declared container path and follows
assignments to a fixpoint, and the declared CONTAINERS are what say where a chain stops - which
ties the gate to the declaration rather than to a second list of block names.

What stays undeclared is `EXEMPT`, and every entry is one shape: a map keyed by a name the AUTHOR
invents (an instrument, a commodity), or a list of them. `Table` declares fixed columns and
`Container` fixed named children, so the vocabulary cannot state either - the same reason
`Tradable_Instruments`, `Liabilities` and `Portfolio_State` are pinned shapeless in
`test_schema_emission`. Per-INSTRUMENT keys (`Expiry_Date`, `Contract_Size`,
`Allow_Holding_Past_Last_Trade`, `Quantity`, `Strike`, ...) need no entry at all: they are read off
a per-entry dict INSIDE one of those maps, so no chain reaches them and inventing a shape for them
is what this gate exists to prevent.
"""
import ast
import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from derivus import calculation, hedge_bundle, hedge_runtime, hedge_solver, stochasticprocess, schema

#: The modules a `Hedging_Problem` key can be read in. `hedge_bundle` / `hedge_solver` index the
#: NORMALIZED runtime (lowercased keys) and are walked to pin that: `hedge_runtime` is the one
#: JSON boundary, and a raw JSON read appearing downstream of it is the defect.
WALKED = (hedge_runtime, calculation, stochasticprocess, hedge_bundle, hedge_solver)
DOWNSTREAM = (hedge_bundle, hedge_solver)

#: `HedgeMonteCarlo.execute` does `self._inner_state_opts = hedging_problem` and forwards it to
#: every process's `reseed_inner_state(..., opts, ...)` OPAQUELY - the calc never reads a model
#: switch out of it. That hop is through an attribute and a call argument, so no AST walk recovers
#: it; it is declared here instead, which is also where it is reviewable.
BRIDGED = {'opts': ''}

#: Read against a declared container and NOT declared, because the value is a map keyed by a name
#: the author invents, or a list of such names. Pinned in both directions: declaring one fails
#: here, and so does a new undeclared read.
EXEMPT = {
    ('Portfolio_State', 'Positions'),            # {instrument: contracts}
    ('Portfolio_State', 'Cash_Balances'),        # {account: amount}
    ('Portfolio_State', 'Settlement_Prices'),    # {instrument: price}
    ('Portfolio_State', 'Margin_Balances'),      # {account: amount}
    ('Portfolio_State', 'Initial_Margin'),       # {instrument: {Method, Amount}}
    ('Portfolio_State', 'Spot_Price_History'),   # {commodity: {Dates, Prices}}
    ('Evaluator', 'Position_Limits'),            # {instrument: {Min_Position, Max_Position}}
    ('Evaluator', 'Cash_Instruments'),           # [instrument, ...] - and its two older spellings
    ('Evaluator', 'Cash_Accounts'),
    ('Evaluator', 'Cash_Instrument'),
    ('Solver', 'Active_Hedge_Indices'),          # [hedge index, ...]
}


def declared():
    """`{(container_path, key): F}` for every field declared under `Hedging_Problem`, and the set
    of container paths a read chain may pass through (`''` being the block itself)."""
    block = next(f for f in calculation.HedgeMonteCarlo.fields if f.key == 'Hedging_Problem')
    fields, containers = {}, {''}

    def walk(path, field):
        for child in field.sub_fields or ():
            fields[(path, child.key)] = child
            if child.type == 'Container':
                containers.add(f'{path}.{child.key}'.lstrip('.'))
                walk(f'{path}.{child.key}'.lstrip('.'), child)
    walk('', block)
    return fields, containers


def _unwrap(node):
    """`x or {}` reads as `x` - the boundary spells an absent block that way twice."""
    while isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
        node = node.values[0]
    return node


#: A subscript or a one-argument `.get` publishes no default to disagree with.
NO_FALLBACK = object()


def _read(node):
    """`(receiver, key, fallback)` for `d['K']` and `d.get('K', ...)`."""
    node = _unwrap(node)
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) \
            and isinstance(node.slice.value, str):
        return node.value, node.slice.value, NO_FALLBACK
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
            and node.func.attr == 'get' and node.args and isinstance(node.args[0], ast.Constant) \
            and isinstance(node.args[0].value, str):
        fallback = (node.args[1].value if len(node.args) == 2
                    and isinstance(node.args[1], ast.Constant) else NO_FALLBACK)
        return node.func.value, node.args[0].value, fallback
    return None, None, None


def resolve(node, aliases, containers):
    """The declared container path this expression IS, or None. `Hedging_Problem` is the root
    whatever it is read off (`config[...]` in the boundary, `params.get(...)` in the calc)."""
    node = _unwrap(node)
    if isinstance(node, ast.Name):
        return aliases.get(node.id)
    receiver, key, _ = _read(node)
    if key == 'Hedging_Problem':
        return ''
    if receiver is None:
        return None
    parent = resolve(receiver, aliases, containers)
    path = None if parent is None else f'{parent}.{key}'.lstrip('.')
    return path if path in containers else None


def block_reads(containers):
    """`{(container_path, key): {fallback, ...}}` over every walked module.

    Aliases are resolved to a FIXPOINT and keyed by name alone: the boundary binds
    `solver_config` / `evaluator_config` / `hedging_problem` once each and then passes them as
    same-named parameters of the three normalizing helpers, so the name IS the binding."""
    reads = {}
    for module in WALKED:
        tree = ast.parse(inspect.getsource(module))
        aliases = dict(BRIDGED)
        while True:
            found = dict(aliases)
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                        and isinstance(node.targets[0], ast.Name):
                    path = resolve(node.value, found, containers)
                    if path is not None:
                        found[node.targets[0].id] = path
            if found == aliases:
                break
            aliases = found
        for node in ast.walk(tree):
            receiver, key, fallback = _read(node)
            if receiver is None:
                continue
            path = resolve(receiver, aliases, containers)
            if path is not None:
                reads.setdefault((path, key), set()).add(fallback)
    return reads


DECLARED, CONTAINERS = declared()
READS = block_reads(CONTAINERS)


def test_the_gate_is_not_vacuous():
    """Every assertion below is vacuous over an empty walk - an alias rule that stops resolving
    would turn this file green rather than red."""
    assert len(CONTAINERS) > 1, 'no declared container under Hedging_Problem'
    assert len(READS) > 30, f'the alias walk found only {len(READS)} reads - it stopped resolving'


def test_hedge_runtime_is_the_only_json_boundary():
    """`construct_hedge_runtime` normalizes the block once and everything downstream indexes the
    result, which is why the runtime keys are lowercased. A raw `Hedging_Problem` key read in the
    bundle or the solver is a second boundary, and a second place to keep a default in step."""
    for module in DOWNSTREAM:
        tree = ast.parse(inspect.getsource(module))
        leaked = {key for node in ast.walk(tree)
                  for receiver, key, _ in [_read(node)]
                  if receiver is not None
                  and resolve(receiver, dict(BRIDGED), CONTAINERS) is not None}
        assert not leaked, f'{module.__name__} reads raw JSON keys {sorted(leaked)}'


@pytest.mark.parametrize('path,key', sorted(DECLARED))
def test_a_declared_knob_is_read(path, key):
    """A descriptor no read reaches is a panel field that writes a key nobody honours - the
    `Base_Time_Grid` defect, which is what put a hedging job on the hardcoded default grid."""
    assert (path, key) in READS, (
        f'Hedging_Problem.{path}.{key} is declared and the hedge stack never reads it'.replace(
            '..', '.'))


@pytest.mark.parametrize('path,key', sorted(DECLARED))
def test_a_declared_default_is_the_default_the_engine_falls_back_to(path, key):
    """The two published defaults have to be one.

    `Container` and `default=REQUIRED` are skipped: a container's value is its children, and a
    REQUIRED knob is one the boundary VALIDATES rather than defaults - it raises on a missing
    `Solver.Object` and subscripts `Objective.Object`, so the `.get(key, '')` reads beside those
    are inside the validation, not a default anyone gets.

    A key with no two-argument read publishes no fallback either, and the engine's one-argument
    `.get` returns None - so None is the declared default that means the same thing."""
    field = DECLARED[(path, key)]
    if field.type == 'Container' or field.default is schema.REQUIRED:
        return
    fallbacks = READS.get((path, key), set()) - {NO_FALLBACK}
    assert len(fallbacks) < 2, (
        f'Hedging_Problem.{path}.{key} is read with {len(fallbacks)} different fallbacks: '
        f'{sorted(map(repr, fallbacks))}')
    engine = fallbacks.pop() if fallbacks else None
    assert field.default == engine, (
        f'Hedging_Problem.{path}.{key} declares {field.default!r} and the engine falls back to '
        f'{engine!r}')


def test_every_knob_the_engine_reads_is_declared():
    """The converse, and the reason the sub-containers were opened up: forty knobs the hedge
    stack honours that no schema-authored job could name."""
    undeclared = sorted(read for read in READS if read not in DECLARED and read not in EXEMPT)
    assert not undeclared, (
        f'Hedging_Problem keys the engine reads that no schema declares: {undeclared}')


def test_the_exempt_shapes_are_exactly_these():
    """Pinned in both directions, as `SHAPELESS` is: declaring one of these fails here, and so
    does an entry the engine stopped reading."""
    stale = sorted(read for read in EXEMPT if read not in READS)
    declared_anyway = sorted(read for read in EXEMPT if read in DECLARED)
    assert not stale, f'exempt keys nothing reads: {stale}'
    assert not declared_anyway, f'exempt keys that are now declared: {declared_anyway}'
