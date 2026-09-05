"""REACH MAP: which consumers a symbol reaches, and which document reads it.

    python gates/reach.py LeastSquaresSolve         # consumers + documents + holes
    python gates/reach.py --from QEDI_CustomAutoCallSwap_V2   # the inverse
    python gates/reach.py --dirty [--since REV] [--repo DIR]  # changed SYMBOLS + a minimal doc set
    python gates/reach.py --build-map [--repo DIR]  # rebuild the document map (engine runs)

Symbol-and-document-granular sibling of `gates/impacted.py`, which is file-and-test-granular.
They are NOT composed: that one answers "which test files", this one "which consumer classes and
which JSON documents", and neither reduces to the other. Run both.

VERB 1 - STATIC REACH (seconds, always fresh, no engine import). An AST call graph over
`derivus/*.py`; nodes are module-level functions, classes, methods and nested functions, keyed
`module.Qualname`. An edge is EXACT where the receiver's type is written down and a GUESS where it
is not:
  * a NAME -> its import binding if the module imports it, else the same module's symbol of that
    name, else (a guess) every symbol so named. The import table is the sharpest lever there is:
    `pricing` binds `F` to `torch.nn.functional` while `schema` DEFINES an `F`;
  * `mod.attr` for a derivus module, and `self.` / `super()` / `Class.attr` -> resolved up the
    MRO, and where the base declares no body, down to the subclasses that do. Not found in the
    hierarchy means a DATA attribute and no edge at all;
  * an unnamed receiver CALLING a protocol verb (a name defined only inside one consumer family -
    `.calculate` is a Deal's by construction) or a name with exactly one definition in the package;
  * registries read as DATA: the five `construct_*` reach WHOLE what their store selects
    (`calc_type`, `market_factor_type`, `factor_types`, `model_type`, `Deal` subclasses), and so
    does every module-level `UPPER = {key: symbol}` dict, found by shape rather than by name
    (`OSS_SPOT_MODEL_KITS`, `QUOTE_WRITERS`);
  * a class -> its bases and its `__init__` (plus an autograd `Function`'s `forward`/`backward`,
    reached from C). NAMING a class constructs it; it does not run all of it. An `isinstance(x,
    Config)` that reached `Config.bootstrap` put every deal on every bootstrapper;
  * anything else `x.attr` -> a GUESS to every symbol so named. Guesses are NOT chained: chained,
    45 of 50 deals reach `lv_walk` through `.blocks` on an unrelated object. `--loose` chains them
    and the count is printed on every query, so a duck-typed-only route is never silently absent.
CONSUMERS are deals, pricers, processes, factors, bootstrapper families, calculations, and the
model families deals declare in `spot_models` (whose members are the kit from
`pricing.OSS_SPOT_MODEL_KITS`, the bootstrapper whose `market_factor_type` is `<family>ModelPrices`,
and the deals declaring it).

WHAT THE GRAPH CANNOT SEE. String-keyed dispatch outside the registries it finds by shape; a
callable passed as a VALUE (`partial(f, ...)`, a hook stored on an object); `getattr(o, name)`; an
attribute READ that is never called; and virtual dispatch out of an INHERITED body - `Deal.
calculate`'s `self.generate` stops at `Deal.generate`, so a calculation does not reach a pricer
that way and the DEAL does, through its own override. It sees text, not behaviour: a symbol
reached only on a branch no data takes still reads as reached. That is what verb 2 is for.

VERB 2 - DOCUMENT MAP (an engine run per document, a campaign-boundary artefact). Every JSON JOB
document (a `Calc` block carrying `Calculation` and `Deals`) under the document directories is run
through `derivus.Context` under a `sys.monitoring` PY_START tracer - the census's idiom, every
callback DISABLEs itself, so a traced run costs its untraced clock. Recorded per document: the
`derivus/` functions executed, the calculation type, the wall clock, the models and `Market Prices`
families exercised. Documents run ONE AT A TIME (the RNG and the device are shared state), each
capped at `CAP` seconds; over the cap is recorded SKIPPED, never run to completion.

One document per SHAPE - `(calc object, deal objects, process types, market-price families,
Valuation Configuration, market-data file, calculation flags)`, cheapest representative by
`Batch_Size * Simulation_Batches`. A parameter sweep is 121 documents of one shape and executes one
set of functions; running all of them buys nothing and costs hours. The shape count and the sweep
sizes are in the map.

VERB 3 - QUERY joins the two: static consumers, the documents that executed the symbol (fastest
first, so the cheapest read is the one you run), and the HOLES - the consumers no document reaches,
which is what a change there cannot be read by and also what nothing tests.
"""
import argparse
import ast
import glob
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# artifacts live in the MAIN checkout, whatever worktree the tool runs from
MAIN = os.path.dirname(subprocess.run(
    ['git', '-C', ROOT, 'rev-parse', '--path-format=absolute', '--git-common-dir'],
    capture_output=True, text=True).stdout.strip())
MAP = os.path.join(MAIN, 'artifacts', 'reach', 'document_map.json')
CAP = 180
DOC_DIRS = ('tests/fixtures', 'artifacts')             # walked recursively, in the MAIN checkout
SIZE_KEYS = ('Base_Date', 'Random_Seed', 'Batch_Size', 'Simulation_Batches', 'MCMC_Simulations')


# ======================================================================================
# verb 1: the static graph
# ======================================================================================

def _recv(node):
    """How an attribute's receiver names its type: 'self', 'super', a Name, or None."""
    if isinstance(node, ast.Name):
        return 'self' if node.id == 'self' else node.id
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
            and node.func.id == 'super':
        return 'super'
    return None


def _scope_refs(fn):
    """`(names, attrs)` in a function's OWN scope; an attr is `(receiver, name, is a call)`."""
    names, attrs, bound, called = set(), set(), set(), set()
    a = fn.args
    bound.update(x.arg for x in a.posonlyargs + a.args + a.kwonlyargs)
    bound.update(x.arg for x in (a.vararg, a.kwarg) if x)
    stack = list(fn.body)
    while stack:
        n = stack.pop()
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(n.name)                          # a node of its own
            continue
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            called.add(id(n.func))
        elif isinstance(n, ast.Name):
            (bound if isinstance(n.ctx, (ast.Store, ast.Del)) else names).add(n.id)
        elif isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Load):
            attrs.add((_recv(n.value), n.attr, id(n) in called))   # `self.x = y` is not a call
        stack.extend(ast.iter_child_nodes(n))
    return names - bound, attrs


def _base_names(cls):
    return [b.id if isinstance(b, ast.Name) else b.attr
            for b in cls.bases if isinstance(b, (ast.Name, ast.Attribute))]


def _declared(cls, field):
    """A class-level literal assignment, the registries read as data."""
    for st in cls.body:
        if isinstance(st, ast.Assign) and any(getattr(t, 'id', None) == field for t in st.targets):
            try:
                return ast.literal_eval(st.value)
            except (ValueError, TypeError, SyntaxError):
                return None
    return None


class Graph:
    """The symbol call graph and the consumer sets, built from `derivus/*.py` alone."""

    def __init__(self, repo):
        self.repo, self.node, self.mods = repo, {}, {}
        for path in sorted(glob.glob(os.path.join(repo, 'derivus', '*.py'))):
            mod = os.path.basename(path)[:-3]
            self.mods[mod] = ast.parse(open(path, encoding='utf-8').read(), path)
        for mod, tree in self.mods.items():
            self._collect(tree, mod, mod + '.', None)
        self.imports = {m: self._imports(t) for m, t in self.mods.items()}
        self.short, self.member = {}, {}
        for k in self.node:
            head, name = k.rsplit('.', 1)
            self.short.setdefault(name, []).append(k)
            self.member.setdefault(head, []).append(k)
        self.sub = {}
        for k, v in self.node.items():
            for b in (self._all_bases(k) if v['kind'] == 'class' else ()):
                self.sub.setdefault(b, set()).add(k)
        self.table, self.protocol = self._tables(), self._protocols()
        self.edge, self.guess = {}, {}
        for k, v in self.node.items():
            self.edge[k], self.guess[k] = self._edges(k, v)
        self._registries()
        self.rev, self.rev_guess = {}, {}
        for rev, src in ((self.rev, self.edge), (self.rev_guess, self.guess)):
            for s, dsts in src.items():
                for d in dsts:
                    rev.setdefault(d, set()).add(s)

    def _imports(self, tree):
        """`{bound name: qualified key, or None where it is not derivus}` for one module.

        Exact, and the single biggest precision lever: `pricing` binds `F` to
        `torch.nn.functional` while `schema` DEFINES an `F`, and guessing by short name alone wires
        every pricer to the schema layer.
        """
        out = {}
        for st in tree.body:
            if isinstance(st, ast.Import):
                out.update({(a.asname or a.name).split('.')[0]: None for a in st.names})
            elif isinstance(st, ast.ImportFrom):
                mod = st.module or ''
                local = st.level and (mod or None)     # `from . import x` / `from .schema import F`
                out.update({(a.asname or a.name):
                            '{}.{}'.format(local, a.name) if local else
                            (a.name if st.level and a.name in self.mods else None)
                            for a in st.names})
        return out

    def _collect(self, parent, mod, prefix, cls):
        for st in parent.body:
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                key = prefix + st.name
                is_cls = isinstance(st, ast.ClassDef)
                self.node[key] = {'ast': st, 'mod': mod, 'cls': cls, 'line': st.lineno,
                                  'kind': 'class' if is_cls else 'func'}
                self._collect(st, mod, key + '.', key if is_cls else cls)

    def _mro(self, cls, attr, own=True):
        """The attribute up a class's bases; `own=False` starts at the bases (`super()`)."""
        seen, stack = set(), [cls] if own else list(self._bases(cls))
        while stack:
            c = stack.pop()
            if c is None or c in seen:
                continue
            seen.add(c)
            if c + '.' + attr in self.node:
                return c + '.' + attr
            stack.extend(self._bases(c))
        return None

    def _bases(self, cls):
        return [k for b in _base_names(self.node[cls]['ast'])
                for k in self.short.get(b, ()) if self.node[k]['kind'] == 'class']

    def _classes_named(self, name):
        return [k for k in self.short.get(name or '', ()) if self.node[k]['kind'] == 'class']

    def roots(self, key):
        """A consumer's entry points: itself, and for a class every method it can run."""
        if self.node[key]['kind'] != 'class':
            return {key}
        seen, stack, out = set(), [key], {key}
        while stack:
            c = stack.pop()
            if c in seen:
                continue
            seen.add(c)
            out.update(self.member.get(c, ()))
            stack.extend(self._bases(c))
        return out

    def _edges(self, key, v):
        # NAMING a class is constructing it, not running all of it: `isinstance(x, Config)` does
        # not reach `Config.bootstrap`. The rest of a class is reached through `self.` from its own
        # methods, and by an attribute edge from whoever calls one. `torch.autograd.Function` is
        # the exception - `apply` is C code and reaches `forward`/`backward` invisibly.
        if v['kind'] == 'class':
            auto = 'Function' in {b.split('.')[-1] for b in _base_names(v['ast'])}
            return set(self._bases(key)) | {
                k for k in (self._mro(key, m) for m in ('__init__', '__new__') +
                            (('forward', 'backward', 'setup_context') if auto else ())) if k}, set()
        names, attrs = _scope_refs(v['ast'])
        imports, out, maybe = self.imports[v['mod']], set(), set()
        for nm in names:
            same = v['mod'] + '.' + nm
            if nm in self.table:
                out.update(self.table[nm])
            elif nm in imports:
                if imports[nm] in self.node:
                    out.add(imports[nm])
            elif same in self.node:
                out.add(same)
            else:
                maybe.update(self.short.get(nm, ()))
        for recv, at, call in attrs:
            qual = '{}.{}'.format(recv, at)
            if recv in self.mods and qual in self.node:
                out.add(qual)                          # `utils.lv_walk` - exact
                continue
            if recv in imports and imports[recv] is None:
                continue                               # `np.where` - not derivus
            named = ([v['cls']] if v['cls'] else []) if recv in ('self', 'super') \
                else self._classes_named(recv)
            if named:
                # a receiver whose class IS named resolves inside that hierarchy or not at all: up
                # the bases, and only where the base declares no body, down to the subclasses that
                # do. Chasing every override of an implemented base method would make each deal
                # reach every other deal's pricer. Nothing found is a data attribute.
                hits = {self._mro(c, at, recv != 'super') for c in named if c} - {None}
                out.update(hits or {d + '.' + at for c in named
                                    for d in self.sub.get(c, ()) if d + '.' + at in self.node})
            elif call and (at in self.protocol or len(self.short.get(at, ())) == 1):
                # a CALL of a declared protocol verb, or of a name with one definition in the
                # package: the receiver's type is not written down but it is not in doubt
                out.update(self.short[at])
            elif not at.startswith('__'):
                maybe.update(self.short.get(at, ()))   # unknown receiver - every symbol so named
        return out - {key}, maybe - out - {key}

    def classes(self, mod, field=None, base=None):
        """The classes of a module, optionally those declaring `field` or subclassing `base`."""
        out = []
        for k, v in self.node.items():
            if v['mod'] != mod or v['kind'] != 'class' or v['cls']:
                continue
            if field and _declared(v['ast'], field) is None:
                continue
            if base and base not in self.ancestors(k):
                continue
            out.append(k)
        return out

    def ancestors(self, key):
        seen, stack = set(), [key]
        while stack:
            c = stack.pop()
            for b in _base_names(self.node[c]['ast']):
                for k in self.short.get(b, ()):
                    if self.node[k]['kind'] == 'class' and k not in seen:
                        seen.add(k)
                        stack.append(k)
        return {k.rsplit('.', 1)[1] for k in seen}

    def _tables(self):
        """Module-level `UPPER = {key: symbol}` - a registry read as DATA, per the house rule.

        `OSS_SPOT_MODEL_KITS` and `QUOTE_WRITERS` are found this way rather than named, so the next
        registry is picked up the day it is written. A member is reached WHOLE (a kit is used
        through duck-typed verbs), which is the over-approximation the registry licenses.
        """
        out = {}
        for tree in self.mods.values():
            for st in tree.body:
                if not (isinstance(st, ast.Assign) and isinstance(st.value, ast.Dict)
                        and getattr(st.targets[0], 'id', '').isupper()):
                    continue
                hits = {k for v in st.value.values if isinstance(v, ast.Name)
                        for k in self.short.get(v.id, ())}
                if hits:
                    out[st.targets[0].id] = set().union(*(self.roots(k) for k in hits))
        return out

    def _protocols(self):
        """Attribute names defined ONLY inside one consumer family - the declared protocols.

        `x.calculate` on an unnamed receiver is a Deal by construction and the edge is exact;
        `x.blocks` could be anyone's and stays a guess. This is what keeps the deal walk and the
        process protocol in the reach without letting every duck-typed read wander.
        """
        fams = [set(self.classes('instruments', base='Deal')), set(self.processes()),
                set(self.classes('riskfactors')),
                set(self.classes('bootstrappers', 'market_factor_type')),
                set(self.classes('calculation', 'calc_type'))]
        fams = [f | {b for m in f for b in self._all_bases(m)} for f in fams]
        return {name for name, keys in self.short.items()
                if all(self.node[k]['cls'] for k in keys)
                and any({self.node[k]['cls'] for k in keys} <= f for f in fams)}

    def _all_bases(self, cls):
        seen, stack = set(), self._bases(cls)
        while stack:
            c = stack.pop()
            if c not in seen:
                seen.add(c)
                stack.extend(self._bases(c))
        return seen

    def _registries(self):
        """`globals()` dispatch: each `construct_*` edges to WHAT its store selects, whole."""
        for fn, targets in (
                ('calculation.construct_calculation', self.classes('calculation', 'calc_type')),
                ('bootstrappers.construct_bootstrapper',
                 self.classes('bootstrappers', 'market_factor_type')),
                ('instruments.construct_instrument', self.classes('instruments', base='Deal')),
                ('riskfactors.construct_factor', self.classes('riskfactors')),
                ('stochasticprocess.construct_process', self.processes()),
                ('stochasticprocess.construct_calibration_config',
                 self.classes('stochasticprocess', 'model_type'))):
            if fn in self.edge:
                self.edge[fn].update(*(self.roots(k) for k in targets))

    def processes(self):
        return [k for k in self.classes('stochasticprocess')
                if _declared(self.node[k]['ast'], 'factor_types') is not None
                or k + '.generate' in self.node]

    def kits(self):
        """`OSS_SPOT_MODEL_KITS` read as data: model family -> kit class key."""
        for st in self.mods['pricing'].body:
            if isinstance(st, ast.Assign) and getattr(st.targets[0], 'id', '') == \
                    'OSS_SPOT_MODEL_KITS':
                return {ast.literal_eval(k): 'pricing.' + v.id
                        for k, v in zip(st.value.keys, st.value.values)}
        return {}

    def consumers(self):
        """`{category: {name: {root keys}}}` - the classes and pricers a change is read by."""
        if getattr(self, '_con', None):
            return self._con
        deals = self.classes('instruments', base='Deal')
        boots = self.classes('bootstrappers', 'market_factor_type')
        groups = {'deals': {k.split('.')[-1]: [k] for k in deals},
                  'pricers': {k.split('.')[-1]: [k] for k, v in self.node.items()
                              if v['mod'] == 'pricing' and v['kind'] == 'func' and not v['cls']
                              and k.split('.')[-1].startswith(('pv_', 'sim_'))},
                  'processes': {k.split('.')[-1]: [k] for k in self.processes()},
                  'factors': {k.split('.')[-1]: [k] for k in self.classes('riskfactors')},
                  'bootstrappers': {_declared(self.node[k]['ast'], 'market_factor_type'): [k]
                                    for k in boots},
                  'calculations': {_declared(self.node[k]['ast'], 'calc_type'): [k]
                                   for k in self.classes('calculation', 'calc_type')}}
        kits, families = self.kits(), {}
        for d in deals:
            for m in _declared(self.node[d]['ast'], 'spot_models') or ():
                if m != 'None':
                    families.setdefault(m, set()).add(d)
        groups['models'] = {m: sorted(members | ({kits[m]} if m in kits else set()) |
                                      {k for k in boots
                                       if _declared(self.node[k]['ast'], 'market_factor_type') ==
                                       m + 'ModelPrices'})
                            for m, members in families.items()}
        # `roots` is what the consumer can RUN, inherited bodies included, and answers the static
        # reach. `own` is what it declares itself, and answers "did a document exercise THIS one" -
        # every deal inherits `Deal.calculate`, which every document runs, so roots would report
        # all fifty deals as covered by any document at all.
        self._con = {cat: {n: {'roots': set().union(*(self.roots(k) for k in ks)),
                               'own': {x for k in ks for x in [k] + self.member.get(k, [])}}
                           for n, ks in items.items()} for cat, items in groups.items()}
        return self._con

    def reaches(self, symbol, loose=False):
        """Every node that can reach `symbol` - one reverse BFS."""
        seen = set(self.resolve(symbol))
        stack = list(seen)
        while stack:
            n = stack.pop()
            for p in self.rev.get(n, set()) | (self.rev_guess.get(n, set()) if loose else set()):
                if p not in seen:
                    seen.add(p)
                    stack.append(p)
        return seen

    def reached_by(self, key, loose=False):
        """Every node `key` can reach - one forward BFS."""
        seen, stack = {key}, [key]
        while stack:
            n = stack.pop()
            for d in self.edge.get(n, set()) | (self.guess.get(n, set()) if loose else set()):
                if d not in seen:
                    seen.add(d)
                    stack.append(d)
        return seen

    def resolve(self, symbol):
        """A query name -> node keys: `module.Class.method`, `Class.method` or a bare name."""
        if symbol in self.node:
            return [symbol]
        hits = [k for k in self.node if k == symbol or k.endswith('.' + symbol)]
        return hits or []


# ======================================================================================
# verb 2: the document map
# ======================================================================================

def _objects(node, out):
    if isinstance(node, dict):
        if isinstance(node.get('Object'), str):
            out.add(node['Object'])
        for v in node.values():
            _objects(v, out)
    elif isinstance(node, list):
        for v in node:
            _objects(v, out)


def documents(main):
    """One cheapest representative per SHAPE of every job document under `DOC_DIRS`."""
    groups = {}
    for d in DOC_DIRS:
        for path in sorted(glob.glob(os.path.join(main, d, '**', '*.json'), recursive=True)):
            fixture = os.path.dirname(path) == os.path.join(main, 'tests', 'fixtures')
            try:                                       # a job document is a `Calc` with a mode
                calc = json.load(open(path, encoding='utf-8'))['Calc']
                cal, deals = calc['Calculation'], calc.get('Deals', {})
                assert cal['Object']
            except Exception:
                continue
            md = calc.get('MergeMarketData', {})
            em = md.get('ExplicitMarketData', {}) or {}
            objs = set()
            _objects(deals, objs)
            vc = em.get('Valuation Configuration', {})
            # a fixture is never deduped away: it is a gate's own input, and the question "does a
            # TEST reach this" is answered by that set alone
            shape = json.dumps([path if fixture else '', cal.get('Object'),
                                sorted(objs),
                                sorted({k.split('.')[0] for k in em.get('Price Models', {})}),
                                sorted({k.split('.')[0] for k in em.get('Market Prices', {})}),
                                vc, md.get('MarketDataFile') or '',
                                {k: v for k, v in cal.items() if k not in SIZE_KEYS}],
                               sort_keys=True, default=str)
            cost = (cal.get('Batch_Size') or 1) * (cal.get('Simulation_Batches') or 1)
            rec = {'path': os.path.relpath(path, main).replace(os.sep, '/'),
                   'calc': cal.get('Object'), 'cost': cost,
                   'models': sorted({v['SpotModel'] for k, v in vc.items()
                                     if k in objs and v.get('SpotModel', 'None') != 'None'}),
                   'families': sorted({k.split('.')[0] for k in em.get('Market Prices', {})}),
                   'deals': sorted(objs)}
            groups.setdefault(shape, []).append(rec)
    return [min(v, key=lambda r: (r['cost'], r['path'])) | {'sweep': len(v)}
            for v in groups.values()]


def trace_one(part, doc, repo):
    """Run ONE document under a PY_START tracer and write `{functions, seconds}` to `part`."""
    sys.path.insert(0, repo)
    import derivus as rf
    want = os.path.normcase(os.path.join(repo, 'derivus')) + os.sep
    # the census's guard: an editable install of another checkout would trace nothing and read as
    # a document that executes no derivus function at all - a silent zero, not an error
    assert os.path.normcase(os.path.abspath(rf.__file__)).startswith(want), rf.__file__
    mon, seen = sys.monitoring, set()
    tool = next(i for i in range(6) if mon.get_tool(i) is None)
    mon.use_tool_id(tool, 'derivus-reach')

    def on_start(code, _offset):
        f = os.path.normcase(code.co_filename)
        if not f.startswith(want):
            return mon.DISABLE
        seen.add(os.path.basename(f)[:-3] + '.' + code.co_qualname.replace('.<locals>', ''))
        return mon.DISABLE                             # one hit is the whole question

    mon.register_callback(tool, mon.events.PY_START, on_start)
    mon.set_events(tool, mon.events.PY_START)
    t0, err = time.time(), None
    try:
        cx = rf.Context()
        cx.load_json(doc)
        # a document carrying `Market Prices` is a CALIBRATION job - quotes become factors before
        # anything prices, and its book is usually empty. Without this the whole bootstrapper
        # surface reads as reached by nothing.
        if cx.current_cfg.params.get('Market Prices'):
            cx.bootstrap()
        cx.run_job()
    except Exception as e:
        err = repr(e)[:300]
    finally:
        mon.set_events(tool, 0)
        mon.free_tool_id(tool)
        json.dump({'functions': sorted(seen), 'seconds': round(time.time() - t0, 2),
                   'error': err}, open(part, 'w'))


def build_map(repo):
    """Run every document representative, one at a time, and write the map.

    RESUMES: a document already recorded at this engine is kept, so an interrupted run costs one
    document and a new representative costs only itself.
    """
    docs = documents(MAIN)
    head = subprocess.run(['git', '-C', repo, 'rev-parse', 'HEAD'],
                          capture_output=True, text=True).stdout.strip()
    os.makedirs(os.path.dirname(MAP), exist_ok=True)
    part = os.path.join(os.path.dirname(MAP), 'part.json')
    prior = load_map(repo)
    out = [r for r in prior['documents'] if prior.get('engine') == head
           and r['path'] in {d['path'] for d in docs}] if os.path.exists(MAP) else []
    docs = [d for d in docs if d['path'] not in {r['path'] for r in out}]
    for i, rec in enumerate(sorted(docs, key=lambda r: r['cost']), 1):
        doc = os.path.join(MAIN, rec['path'])
        print('[{}/{}] {}'.format(i, len(docs), rec['path']), flush=True)
        t0 = time.time()
        try:
            subprocess.run([sys.executable, os.path.abspath(__file__), '--trace', part, doc,
                            '--repo', repo], cwd=repo, timeout=CAP, check=False,
                           capture_output=True)
            rec |= json.load(open(part))
        except subprocess.TimeoutExpired:
            rec |= {'functions': [], 'seconds': round(time.time() - t0, 1),
                    'error': 'SKIPPED: over the {} s cap'.format(CAP)}
        except Exception as e:
            rec |= {'functions': [], 'seconds': round(time.time() - t0, 1), 'error': repr(e)[:200]}
        print('    {:7.1f}s  {:5d} functions  {}'.format(
            rec['seconds'], len(rec['functions']), rec.get('error') or ''), flush=True)
        out.append(rec)
        json.dump({'engine': head, 'cap': CAP, 'documents': out}, open(MAP, 'w'), indent=1)
    os.path.exists(part) and os.remove(part)
    print('wrote {}: {} documents at {}'.format(MAP, len(out), head[:7]))


def load_map(repo):
    if not os.path.exists(MAP):
        return {'engine': None, 'documents': []}
    m = json.load(open(MAP))
    head, edits = (subprocess.run(['git', '-C', repo] + a, capture_output=True, text=True).stdout
                   for a in (['rev-parse', 'HEAD'], ['status', '--porcelain', 'derivus']))
    if m.get('engine') != head.strip():
        print('# STALE: map built at {}, repo at {}'.format(
            (m.get('engine') or '?')[:7], head[:7]), file=sys.stderr)
    if edits.strip():
        print('# the map was NOT run against these working-tree edits', file=sys.stderr)
    return m


# ======================================================================================
# verb 3: the queries
# ======================================================================================

def hit(rec, keys):
    """Did this document execute any of `keys` (a class counts if any of its methods ran)?"""
    fns = set(rec['functions'])
    return any(k in fns or any(f.startswith(k + '.') for f in fns) for k in keys)


def query(g, symbol, docs, loose=False):
    keys = g.resolve(symbol)
    if not keys:                                       # a deleted symbol, or a typo
        return {'symbol': symbol, 'keys': [], 'consumers': {}, 'callers': [], 'duck': 0,
                'documents': [], 'holes': {}, 'error': 'not in derivus/ at this revision'}
    back = g.reaches(symbol, loose)
    wide = g.reaches(symbol, True)
    con = {cat: sorted(n for n, v in items.items() if v['roots'] & back)
           for cat, items in g.consumers().items()}
    runs = sorted((r for r in docs if hit(r, keys)), key=lambda r: r['seconds'])
    seen = {c for r in runs for items in g.consumers().values()
            for c, v in items.items() if hit(r, v['own'])}
    return {'symbol': symbol, 'keys': keys, 'consumers': con,
            'callers': sorted({c for k in keys for c in g.rev.get(k, ())} - set(keys)),
            'duck': sum(len([n for n, v in items.items() if v['roots'] & wide]) - len(con[cat])
                        for cat, items in g.consumers().items()),
            'documents': [{'path': r['path'], 'seconds': r['seconds'], 'calc': r['calc']}
                          for r in runs],
            'holes': {cat: sorted(set(v) - seen) for cat, v in con.items() if set(v) - seen}}


def show(q):
    if q.get('error'):
        return print('{}: {}'.format(q['symbol'], q['error']))
    print('{}  ({})'.format(q['symbol'], ', '.join(q['keys'][:4])))
    print('  {:<14s} {}'.format('callers:', ', '.join(q['callers'][:8]) or 'none'))
    for cat, v in q['consumers'].items():
        print('  {:<14s} {}'.format(cat + ':', ', '.join(v) if v else 'none'))
    print('  {:<14s} {} more only through a duck-typed call (--loose)'.format('unnamed:',
                                                                              q['duck']))
    print('  documents:     {}'.format(len(q['documents'])))
    for r in q['documents'][:12]:
        print('      {:7.1f}s  {:<17s} {}'.format(r['seconds'], r['calc'], r['path']))
    if len(q['documents']) > 12:
        print('      ... {} more'.format(len(q['documents']) - 12))
    for cat, v in q['holes'].items():
        print('  HOLE {:<9s} no document reaches: {}'.format(cat, ', '.join(v)))


# ======================================================================================
# the dirty reading: changed SYMBOLS, not changed files
# ======================================================================================

def _symbols(src, path):
    """`{key: source text}` for every function and class in one file."""
    out, mod = {}, os.path.basename(path)[:-3]

    def walk(parent, prefix):
        for st in parent.body:
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                out[prefix + st.name] = ast.get_source_segment(src, st)
                walk(st, prefix + st.name + '.')
    try:
        walk(ast.parse(src), mod + '.')
    except SyntaxError:
        pass
    return out


def _sh(repo, *args):
    return subprocess.run(['git', '-C', repo] + list(args), capture_output=True, text=True).stdout


def changed_symbols(repo, since=None):
    """The changed function/method/class SYMBOLS of a diff, by AST on both sides."""
    files = _sh(repo, 'diff', '--name-only', since or 'HEAD').split()
    if not since:
        files += [ln[3:] for ln in _sh(repo, 'status', '--porcelain').splitlines()
                  if ln.startswith('??')]
    out = []
    for f in files:
        if not (f.startswith('derivus/') and f.endswith('.py')):
            continue
        old = _sh(repo, 'show', '{}:{}'.format(since or 'HEAD', f))
        try:
            new = open(os.path.join(repo, f), encoding='utf-8').read()
        except OSError:
            new = ''
        a, b = _symbols(old, f), _symbols(new, f)
        for k in sorted(set(a) | set(b)):
            if a.get(k) != b.get(k) and (k not in a or k not in b or
                                         not any(k2.startswith(k + '.') and a.get(k2) != b.get(k2)
                                                 for k2 in set(a) | set(b))):
                out.append(k)
    return out


def dirty(g, docs, since, repo, loose=False):
    syms = changed_symbols(repo, since)
    rows = [query(g, s, docs, loose) for s in syms]
    need = {r['symbol']: {d['path'] for d in r['documents']} for r in rows}
    picked, left = [], {s for s, d in need.items() if d}
    while left:
        # greedy set cover; ties to a document that RAN, then to the faster one. A document that
        # raised still executed what it executed, but it is not a check anyone can read
        best = max(docs, key=lambda r: (sum(r['path'] in need[s] for s in left),
                                        not r.get('error'), -r['seconds']))
        take = {s for s in left if best['path'] in need[s]}
        if not take:
            break
        picked.append({'path': best['path'], 'seconds': best['seconds'], 'covers': sorted(take)})
        left -= take
    return {'changed': syms, 'reach': rows, 'cover': picked,
            'unreached': sorted(s for s, d in need.items() if not d)}


def show_dirty(d):
    print('{} changed symbols\n\nminimal document set ({} documents, {:.1f}s):'.format(
        len(d['changed']), len(d['cover']), sum(c['seconds'] for c in d['cover'])))
    for c in d['cover']:
        print('  {:7.1f}s  {}\n            covers {}'.format(
            c['seconds'], c['path'], ', '.join(c['covers'])))
    groups = {}
    for r in d['reach']:
        sig = '; '.join('{}: {}'.format(k, ', '.join(v))
                        for k, v in r['consumers'].items() if v) or \
            'no consumer ({} duck-typed)'.format(r['duck'])
        groups.setdefault(sig, []).append(r['symbol'])
    print('\nconsumers, by group:')
    for sig, syms in sorted(groups.items(), key=lambda x: -len(x[1])):
        print('  {}\n      {}'.format(', '.join(syms[:6]) +
                                      (' +{}'.format(len(syms) - 6) if len(syms) > 6 else ''), sig))
    if d['unreached']:
        print('\nNO DOCUMENT REACHES ({}): {}'.format(
            len(d['unreached']), ', '.join(d['unreached'][:20])))
    print('\n# file-granular test selection is its sibling: gates/impacted.py --dirty')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('symbol', nargs='?')
    ap.add_argument('--from', dest='frm', help='a consumer class: what it reaches')
    ap.add_argument('--dirty', action='store_true')
    ap.add_argument('--since')
    ap.add_argument('--repo', default=ROOT, help='the checkout to read and diff (this one)')
    ap.add_argument('--build-map', action='store_true')
    ap.add_argument('--trace', metavar='PART', help='trace ONE document - how --build-map recurses')
    ap.add_argument('--json', action='store_true')
    ap.add_argument('--loose', action='store_true', help='chain duck-typed calls too')
    args = ap.parse_args()
    repo = os.path.abspath(args.repo)

    if args.trace:
        return trace_one(args.trace, args.symbol, repo)
    if args.build_map:
        return build_map(repo)

    t0 = time.time()
    g = Graph(repo)
    print('# graph: {} symbols, {} edges (+{} duck-typed), {:.2f}s'.format(
        len(g.node), sum(len(v) for v in g.edge.values()),
        sum(len(v) for v in g.guess.values()), time.time() - t0), file=sys.stderr)
    docs = load_map(repo)['documents']

    if args.frm:
        keys = [r for key in g.resolve(args.frm) for r in g.roots(key)]
        out = sorted({k for key in keys for k in g.reached_by(key, args.loose)})
        if args.json:
            return print(json.dumps({'from': args.frm, 'reaches': out}, indent=1))
        by_mod = {}
        for k in out:
            by_mod.setdefault(k.split('.')[0], []).append(k)
        print('{} reaches {} symbols'.format(args.frm, len(out)))
        for m, v in sorted(by_mod.items()):
            print('  {:<20s} {}'.format(m, len(v)))
        return
    if args.dirty or args.since:
        d = dirty(g, docs, args.since, repo, args.loose)
        return print(json.dumps(d, indent=1)) if args.json else show_dirty(d)
    if not args.symbol:
        return print('a symbol, --from, --dirty or --build-map is required', file=sys.stderr)
    q = query(g, args.symbol, docs, args.loose)
    return print(json.dumps(q, indent=1)) if args.json else show(q)


if __name__ == '__main__':
    sys.exit(main())
