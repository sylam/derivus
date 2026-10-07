"""Impact-based test selection: run the tests a change reaches, line by line.

    python gates/impacted.py --dirty               # the tests the working tree's change reaches
    python gates/impacted.py --since BASE          # ... the diff BASE..HEAD
    python gates/impacted.py --commit SHA          # ... one commit's own diff
        --files  round up to whole test files     --time  with the map's recorded seconds
        --run    run the selection
    python gates/impacted.py --measure DIR         # the campaign-boundary pass, then the map

THE MAP (`gates/test_impact_map.json`, local) is one instrumented pass: every test file in its own
process under coverage with per-test contexts, homes on temp folders, GPU 0, and a static context
naming the file so a test module's import-time lines keep their file. It records, at its commit,
which tests executed each line of derivus/, derivus_spine/, derivus_mcp/, derivus_bloomberg/ and
the gates/ modules tests import, and each test's instrumented seconds.

SELECTION reads a diff as changed statements, each version aligned to the map's commit line by line:
  - a changed or deleted statement selects the tests that executed it;
  - a new statement selects the tests that reach its place - those that executed the unchanged
    statement before it in its block, else those that entered the block;
  - a changed `def` or `class` header selects every test that entered the function or class; a new
    top-level def or class selects nothing, its callers' lines do;
  - the header or docstring of a def a decorator may register at import - any but `BENIGN` - selects
    the module's tests, since a registry is read without entering what it holds;
  - a module- or class-level binding (a constant, an import) selects every test reaching a line
    that names it, closed over the bindings those lines make; any other module-level statement
    selects the module's tests;
  - any other docstring, a comment or a blank line selects nothing.
The map's lines are WHAT EXECUTED: a test whose result a changed line cannot reach is not selected,
and a selected test need not move - a last-bit respelling reaches every test that runs the line.
A test file selects its changed test functions, or itself and the test modules importing it where
the change is outside one; a fixture JSON the test files that name it, closed over fixtures naming
fixtures. A selected test brings the tests of its file that share a module-scoped fixture or a
cached helper with it. It FAILS OPEN on `ALWAYS_ALL` and on a path the map has never seen - a new
module, a data file a package reads - and says which; docs, experiments, notebooks, web, data,
artifacts, markdown and scripts no test imports select nothing.

WHAT THIS DOES NOT REPLACE: the full suite at a campaign boundary, for three reasons. A statistical
gate can be execution-order sensitive (the global torch RNG stream moves with the selected set); the
map is a snapshot, blind to a dependency a test gained after it; and a value one test leaves in an
engine-level cache (`structures.RISK_CACHE`, the service's book caches, `quad_nodes`) is charged to
that test alone, so a later test of the same file reading it comes only through a shared fixture or
helper - `--files` rounds that residue away.
"""
import argparse
import ast
import csv
import difflib
import functools
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import namedtuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAP_PATH = os.path.join(ROOT, 'gates', 'test_impact_map.json')
MEASURED = ('derivus', 'derivus_spine', 'derivus_mcp', 'derivus_bloomberg', 'gates')
ALWAYS_ALL = {'derivus/__init__.py', 'tests/conftest.py'}
NOTHING = ('docs_src/', 'experiments/', 'notebooks/', 'web/', 'data/', 'artifacts/', '.github/')
#: Files that drive the service's book verbs and provision their own spine home - with one set
#: in the environment their book verbs answer as a recorded desk, and refuse.
NO_SPINE_HOME = {'test_diary.py', 'test_service.py', 'test_mcp.py', 'test_cross_spot_model.py',
                 'test_market_prices_partition.py'}
KEYWORD = re.compile(r'\s*(else|finally)\s*:')
#: Decorators that read nothing off what they wrap at import; any other one may register it - its
#: signature, its docstring - somewhere a test reads without entering it.
BENIGN = {'staticmethod', 'classmethod', 'property', 'setter', 'getter', 'deleter', 'wraps',
          'cached_property', 'lru_cache', 'cache', 'abstractmethod', 'no_grad', 'enable_grad'}

#: A statement as coverage sees it: its header lines, `def`/`class`/`stmt`/`doc`, a def's or class's
#: qualname, the innermost enclosing def or class (`''` at module level) and whether that is a
#: `def`, `class` or the `module`, the unit holding its block and the block's name, the unit before
#: it there, the first unit of each of its blocks, the names a module- or class-level statement
#: binds (None for any other statement), whether a decorator may register it, and its last line.
Unit = namedtuple('Unit', 'lines kind name scope level parent block prev first binds decorated end')


def git(*args):
    return subprocess.run(('git',) + args, cwd=ROOT, capture_output=True, text=True,
                          encoding='utf-8', errors='replace').stdout


@functools.lru_cache(maxsize=None)
def show(rev, path):
    """`path`'s text at `rev`, the working tree's where `rev` is None; None where it is absent."""
    if rev is None:
        full = os.path.join(ROOT, path)
        return open(full, encoding='utf-8', errors='replace').read() if os.path.isfile(full) \
            else None
    got = subprocess.run(['git', 'show', f'{rev}:{path}'], cwd=ROOT, capture_output=True)
    return got.stdout.decode('utf-8', 'replace') if got.returncode == 0 else None


def opcodes(a, b):
    return difflib.SequenceMatcher(None, (a or '').splitlines(), (b or '').splitlines(),
                                   autojunk=False).get_opcodes()


def align(a, b):
    """`{line of a: line of b}` over the lines the two texts share."""
    if a.splitlines() == b.splitlines():
        return {i: i for i in range(1, len(a.splitlines()) + 1)}
    return {i1 + k + 1: j1 + k + 1 for tag, i1, i2, j1, j2 in opcodes(a, b) if tag == 'equal'
            for k in range(i2 - i1)}


def changed_lines(a, b):
    """The lines of `a` a diff to `b` removes and the lines of `b` it adds."""
    old, new = set(), set()
    for tag, i1, i2, j1, j2 in opcodes(a, b):
        if tag != 'equal':
            old.update(range(i1 + 1, i2 + 1))
            new.update(range(j1 + 1, j2 + 1))
    return old, new


def binds(node):
    """The names a statement binds and does nothing else with, else None."""
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        names = [a.asname or a.name.split('.')[0] for a in node.names]
        return None if '*' in names else names
    targets = node.targets if isinstance(node, ast.Assign) else \
        [node.target] if isinstance(node, (ast.AnnAssign, ast.AugAssign)) else None
    if targets is None:
        return None
    names = []
    for t in targets:
        for n in (t.elts if isinstance(t, ast.Tuple) else [t]):
            if not isinstance(n, ast.Name):
                return None
            names.append(n.id)
    return names


def parse_units(text):
    """`(units, {line: [unit index]})` for a source text, every statement a unit."""
    src, units, at = text.splitlines(), [], {}

    def add(lines, kind, name, scope, level, parent, block, prev, binding, decorated, end):
        units.append(Unit(tuple(lines), kind, name, scope, level, parent, block, prev, {}, binding,
                          decorated, end))
        for ln in lines:
            at.setdefault(ln, []).append(len(units) - 1)
        return len(units) - 1

    def keywords(s):
        """The `else:` / `finally:` lines opening a compound statement's later blocks."""
        out = []
        for block in (getattr(s, 'orelse', None), getattr(s, 'finalbody', None)):
            if block and not src[block[0].lineno - 1].lstrip().startswith('elif'):
                ln = block[0].lineno - 1
                while ln > s.lineno and not KEYWORD.match(src[ln - 1]):
                    ln -= 1
                out += [ln] if ln > s.lineno else []
        return out

    def walk(stmts, parent, block, scope, qual, level):
        prev = first = None
        for i, s in enumerate(stmts):
            body = s.cases if isinstance(s, ast.Match) else getattr(s, 'body', None)
            start = body[0].pattern.lineno if isinstance(s, ast.Match) else \
                body[0].lineno if body else None
            head = s.end_lineno if body is None else max(s.lineno, start - 1)
            if isinstance(s, ast.Expr) and i == 0 and block == 'body' and \
                    isinstance(s.value, ast.Constant) and isinstance(s.value.value, str) and \
                    (parent is None or units[parent].kind in ('def', 'class')):
                idx = add(range(s.lineno, s.end_lineno + 1), 'doc', None, qual, level, parent,
                          block, prev, None, False, s.end_lineno)
            elif isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f'{qual}.{s.name}' if qual else s.name
                kind = 'class' if isinstance(s, ast.ClassDef) else 'def'
                lead = min([s.lineno] + [d.lineno for d in s.decorator_list])
                registers = any(re.sub(r'\(.*', '', ast.unparse(d), flags=re.S).split('.')[-1]
                                not in BENIGN for d in s.decorator_list)
                idx = add(range(lead, head + 1), kind, name, qual, level, parent, block, prev,
                          None, registers, s.end_lineno)
                units[idx].first['body'] = walk(s.body, idx, 'body', name, name, kind)
            else:
                idx = add(list(range(s.lineno, head + 1)) + keywords(s), 'stmt', None, qual, level,
                          parent, block, prev, binds(s) if level != 'def' else None, False,
                          s.end_lineno)
                for field in ('body', 'orelse', 'finalbody'):
                    stmts_ = getattr(s, field, None)
                    if isinstance(stmts_, list) and stmts_ and isinstance(stmts_[0], ast.stmt):
                        units[idx].first[field] = walk(stmts_, idx, field, qual, qual, level)
                for k, h in enumerate(getattr(s, 'handlers', []) + getattr(s, 'cases', [])):
                    lead = h.lineno if isinstance(h, ast.ExceptHandler) else h.pattern.lineno
                    hid = add(range(lead, max(lead, h.body[0].lineno - 1) + 1), 'stmt', None, qual,
                              level, idx, f'handlers[{k}]', None, None, False, h.body[-1].end_lineno)
                    units[idx].first[f'handlers[{k}]'] = hid
                    units[hid].first['body'] = walk(h.body, hid, 'body', qual, qual, level)
            first = idx if first is None else first
            prev = idx
        return first

    walk(ast.parse(text).body, None, 'body', '', '', 'module')
    return units, at


class Version:
    """One file at one revision, parsed and aligned to the map's copy of it."""

    def __init__(self, text, map_text):
        self.units, self.at = parse_units(text)
        self.to_map = align(text, map_text) if map_text is not None else {}

    def anchored(self, i):
        """The map's lines for unit `i`'s header where every one is unchanged, else None."""
        got = [self.to_map.get(ln) for ln in self.units[i].lines]
        return got if got and all(got) else None


class Selector:
    """The tests a diff from `base` to `target` (None: the working tree) reaches, as node ids,
    with the reason each path selected what it did."""

    def __init__(self, imap, target):
        self.map, self.target, self.reasons, self.fail = imap, target, [], None
        self.sets = [frozenset(s) for s in imap['sets']]
        self.cache, self.versions, self.named, self.extra = {}, {}, {}, set()
        self.moved, self.imports = set(), (set(), set())
        self.known = {n: k for k, n in enumerate(imap['nodes'])}
        self.by_test = {}
        for n in imap['nodes']:
            self.by_test.setdefault(n.split('[')[0], []).append(n)

    # ---- the map's side ---------------------------------------------------------------------
    def ctx(self, path, lines):
        table, out = self.map['files'].get(path, {}), set()
        for ln in lines:
            if str(ln) in table:
                out |= self.sets[table[str(ln)]]
        return out

    def mapped(self, path):
        """The map's copy of `path`: its text, units and line index."""
        if path not in self.cache:
            text = show(self.map['commit'], path)
            self.cache[path] = (text, *(parse_units(text) if text is not None else ([], {})))
        return self.cache[path]

    @functools.lru_cache(maxsize=None)
    def entered(self, path, name):
        """Every test that entered def or class `name` in the map, None where it has none: the
        body's lines, never the header's, which run at import."""
        _, units, _ = self.mapped(path)
        for u in units:
            if u.name == name and u.kind == 'def':
                return frozenset(self.ctx(path, range(u.lines[-1] + 1, u.end + 1)))
            if u.name == name:
                return frozenset().union(*[self.entered(path, v.name) for v in units
                                           if v.kind in ('def', 'class') and v.scope == name])
        return None

    def version(self, path):
        """The target's copy of `path`, parsed and aligned to the map's."""
        if path not in self.versions:
            text = show(self.target, path)
            self.versions[path] = None if text is None else Version(text, self.mapped(path)[0])
        return self.versions[path]

    def module(self, path):
        return set().union(*map(self.sets.__getitem__, self.map['files'].get(path, {}).values()))

    # ---- the diff's side ----------------------------------------------------------------------
    def registered(self, path, ver, i):
        """The module's tests where unit `i` is the header or docstring of a def or class a
        decorator may have registered at import, else None."""
        u = ver.units[i]
        holder = u if u.kind in ('def', 'class') else \
            ver.units[u.parent] if u.kind == 'doc' and u.parent is not None else None
        return self.module(path) if holder is not None and holder.decorated and \
            holder.level != 'def' else None

    def old(self, path, ver, i):
        """The tests that executed unit `i` of the base version, now changed or gone."""
        u = ver.units[i]
        if self.registered(path, ver, i) is not None:
            return self.registered(path, ver, i)
        if u.kind == 'doc':
            return set()
        name = u.name if u.kind in ('def', 'class') else u.scope
        if u.level != 'def' and u.kind == 'stmt':
            return self.bound(path, u, self.imports[0])
        lines = ver.anchored(i) if u.kind == 'stmt' else None
        if lines:
            return self.ctx(path, lines)
        if self.entered(path, name) is None:     # gone by the map's commit as well: what reached
            return self.names([name.split('.')[-1]])     # it is what still names it
        return self.header(path, name, 'def' if u.kind == 'stmt' else u.kind)

    def header(self, path, name, kind):
        """The tests a changed header of def or class `name` reaches; a class with no method a
        test entered is reached through its module."""
        got = self.entered(path, name)
        return self.module(path) if got is None or (kind == 'class' and not got) else got

    def new(self, path, ver, i):
        """The tests that reach unit `i` of the target version, new or changed."""
        u = ver.units[i]
        if self.registered(path, ver, i) is not None:
            return self.registered(path, ver, i)
        if u.kind == 'doc':
            return set()
        if u.kind in ('def', 'class'):
            if self.entered(path, u.name) is not None:
                return self.header(path, u.name, u.kind)
            if u.level == 'def':
                return self.place(path, ver, i)
            if u.level == 'class':
                return self.header(path, u.scope, 'class')
            return set()
        if u.level != 'def':
            return self.bound(path, u, self.imports[1])
        lines = ver.anchored(i)
        return self.ctx(path, lines) if lines else self.place(path, ver, i)

    def place(self, path, ver, i):
        """The tests that reach unit `i`'s place: the unchanged statement before it in its block,
        else the block's entry - its first statement in the map, or the unit holding it."""
        p = ver.units[i].prev
        while p is not None:
            lines = ver.anchored(p) if ver.units[p].kind != 'doc' else None
            if lines:
                return self.ctx(path, lines)
            p = ver.units[p].prev
        parent, block = ver.units[i].parent, ver.units[i].block
        lines = ver.anchored(parent) if ver.units[parent].kind == 'stmt' else None
        if not lines:
            return self.new(path, ver, parent)
        _, munits, mat = self.mapped(path)
        for m in mat.get(lines[0], []):
            if munits[m].kind == 'stmt' and munits[m].lines[0] == lines[0]:
                first = munits[m].first.get(block)
                return self.ctx(path, munits[first].lines if first is not None else lines)
        return self.ctx(path, lines)

    def bound(self, path, u, imports):
        """A module- or class-level statement's tests: those reaching a name it binds - of an
        import, only the names whose source moved - or the module's where it binds nothing alone."""
        if not u.binds:
            return self.module(path)
        names = [n for n in u.binds if u.lines[0] not in imports or n in self.moved]
        return self.names(names, None if u.level == 'class' else path) if names else set()

    def names(self, names, home=None):
        """Every test reaching a line that READS one of `names` - a class's attribute anywhere, a
        module's name in `home` itself, as `module.name`, or wherever it is imported by name -
        closed over the module- and class-level bindings those lines make, and every test function
        that reads one."""
        key = (frozenset(names), home)
        if key in self.named:
            return self.named[key]
        out, seen, todo = set(), set(), [(n, home) for n in names]
        while todo:
            batch = [n for n in todo if n not in seen]
            seen.update(batch)
            todo = []
            if not batch:
                break
            tests = glob.glob(os.path.join(ROOT, 'tests', '*.py'))
            for path in list(self.map['files']) + sorted('tests/' + os.path.basename(t)
                                                         for t in tests):
                text = show(self.target, path)
                lines = text and reads(batch, path, text)
                if not lines:
                    continue
                if path.startswith('tests/'):
                    out |= self.index(self.test_lines(path, text, lines))
                    continue
                ver = self.version(path)
                for ln in lines:
                    for i in ver.at.get(ln, []):
                        u = ver.units[i]
                        if u.kind in ('def', 'class'):
                            out |= self.registered(path, ver, i) or \
                                self.entered(path, u.name) or set()
                        elif u.level != 'def':
                            if u.binds:
                                todo += [(b, None if u.level == 'class' else path) for b in u.binds]
                            elif u.kind == 'stmt':
                                out |= self.module(path)
                        elif ver.anchored(i):
                            out |= self.ctx(path, ver.anchored(i))
        self.named[key] = out
        return out

    def index(self, nodes):
        self.extra |= {n for n in nodes if n not in self.known}
        return {self.known[n] for n in nodes if n in self.known}

    # ---- test files -------------------------------------------------------------------------
    def test_lines(self, rel, text, lines):
        """Node ids for `lines` of a test module: the test functions holding them, or the module
        and the test modules importing it where a line of code is outside one."""
        tests, src = test_functions(text), text.splitlines()
        code = {ln for ln in lines if ln <= len(src) and src[ln - 1].strip() and
                not src[ln - 1].strip().startswith('#')}
        if not os.path.basename(rel).startswith('test_') or code - set(tests):
            return {'tests/' + f for f in importers(rel)}
        return {n for prefix in {f'{rel}::{tests[ln]}' for ln in code}
                for n in self.by_test.get(prefix, [prefix])}

    # ---- the diff ---------------------------------------------------------------------------
    def select(self, base, paths):
        nodes = set()
        for path in paths:
            if path in ALWAYS_ALL:
                self.fail = f'{path}: whole-suite file by construction'
                return None
            if path.startswith(NOTHING) or path.endswith(('.md', '.txt', '.rst')) or \
                    path in ('LICENSE', '.gitignore', '.readthedocs.yaml', 'mkdocs.yml'):
                continue
            a, b = show(base, path), show(self.target, path)
            if path.startswith('tests/fixtures/'):
                hits = fixture_map().get(path, [])
                nodes |= set(hits)
                self.reasons.append(f'{path}: {len(hits)} test files name it')
                continue
            if path.startswith('tests/') and path.endswith('.py'):
                old, new = changed_lines(a, b)
                got = (self.test_lines(path, a, old) if a else set()) | \
                    (self.test_lines(path, b, new) if b else set())
                nodes |= got
                self.reasons.append(f'{path}: {len(got)} tests or test files')
                continue
            if path.endswith('.py') and path.split('/')[0] not in MEASURED + ('tests',):
                stem = os.path.basename(path)[:-3]
                if not imported_anywhere(stem):
                    self.reasons.append(f'{path}: no test or package imports it')
                    continue
            if path not in self.map['files']:
                if path in self.map['known'] or (path.startswith('gates/') and a is None and
                                                  not imported_anywhere(path.split('/')[1][:-3])):
                    self.reasons.append(f'{path}: no test executes it')
                    continue
                self.fail = f'{path}: a path the map has never seen'
                return None
            mtext, got = self.mapped(path)[0], set()
            ia, ib = import_pairs(a), import_pairs(b)
            self.imports = (set(ia), set(ib))
            self.moved = {n for n, _ in set().union(set(), *ia.values()) ^
                          set().union(set(), *ib.values())}
            old, new = changed_lines(a, b)
            if a is not None:
                va = Version(a, mtext)
                for i in sorted({i for ln in old for i in va.at.get(ln, [])}):
                    got |= self.old(path, va, i)
            if b is not None:
                vb = Version(b, mtext)
                for i in sorted({i for ln in new for i in vb.at.get(ln, [])}):
                    got |= self.new(path, vb, i)
            got = {self.map['nodes'][k] for k in got}
            nodes |= got
            self.reasons.append(f'{path}: {len(got)} tests')
        nodes |= self.extra
        return self.alive(share(nodes, self.map['nodes'], self.target))

    def alive(self, nodes):
        """`nodes` less the tests and files the target no longer has."""
        out, texts = set(), {}
        for n in nodes:
            path = n.split('::')[0]
            if path not in texts:
                text = show(self.target, path)
                texts[path] = None if text is None else set(test_functions(text).values())
            if texts[path] is not None and ('::' not in n or
                                             n.split('::', 1)[1].split('[')[0] in texts[path]):
                out.add(n)
        return out


@functools.lru_cache(maxsize=None)
def read_names(text):
    """`{line: {read}}` for the code of `text` - every name a line reads, as the syntax tree holds
    it and never a comment, a docstring or a string: a bare name `('', name)`, an attribute of
    anything `('.', attr)`, and `module.name` `(module, name)`."""
    out = {}
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            out.setdefault(node.lineno, set()).add(('', node.id))
        elif isinstance(node, ast.Attribute):
            out.setdefault(node.lineno, set()).add(('.', node.attr))
            if isinstance(node.value, ast.Name):
                out[node.lineno].add((node.value.id, node.attr))
    return out


def reads(batch, path, text):
    """The lines of `path` that read one of `batch`'s `(name, home)`: an attribute anywhere for a
    class's (`home` None), the bare name in its home or in a file importing it by name,
    `module.name` anywhere else."""
    wanted = set()
    for name, home in batch:
        stem = home and os.path.basename(home)[:-3]
        wanted.add(('.', name) if home is None else ('', name)
                   if home == path or name in imported(text, stem) else (stem, name))
    return {ln for ln, got in read_names(text).items() if got & wanted}


@functools.lru_cache(maxsize=None)
def imported(text, stem):
    """The names `text` imports by name from a module called `stem`, wherever it imports them."""
    return {a.asname or a.name for node in ast.walk(ast.parse(text))
            if isinstance(node, ast.ImportFrom) and (node.module or '').split('.')[-1] == stem
            for a in node.names}


def import_pairs(text):
    """`{line: {(name bound, what it names)}}` for the module- and class-level imports of `text`."""
    out = {}
    for node in ast.parse(text or '').body:
        for s in node.body if isinstance(node, ast.ClassDef) else [node]:
            if isinstance(s, (ast.Import, ast.ImportFrom)):
                out[s.lineno] = {(a.asname or a.name.split('.')[0],
                                  '{}:{}'.format(getattr(s, 'module', None), a.name))
                                 for a in s.names}
    return out


def imported_anywhere(stem):
    """Whether a module named `stem` is imported by a test or a measured package."""
    pat = re.compile(r'^\s*(from\s+\S*\b%s\b\S*\s+import|import\s+.*\b%s\b)' % (stem, stem), re.M)
    for root in MEASURED + ('tests',):
        for f in glob.glob(os.path.join(ROOT, root, '**', '*.py'), recursive=True):
            if pat.search(open(f, encoding='utf-8', errors='replace').read()):
                return True
    return False


def test_functions(text):
    """`{line: test name}` over the lines of each test function, its decorators included, a test
    method as `Class::name`."""
    out = {}
    for node in ast.parse(text).body:
        defs = [(node, '')] if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else \
            [(n, node.name + '::') for n in node.body if isinstance(n, ast.FunctionDef)] \
            if isinstance(node, ast.ClassDef) and node.name.startswith('Test') else []
        for d, cls in defs:
            if d.name.startswith('test'):
                for ln in range(min([d.lineno] + [x.lineno for x in d.decorator_list]),
                                d.end_lineno + 1):
                    out[ln] = f'{cls}{d.name}'
    return out


@functools.lru_cache(maxsize=None)
def importers(rel):
    """The test modules among `rel` and the modules importing it, transitively."""
    bodies = {os.path.basename(p): show(None, 'tests/' + os.path.basename(p))
              for p in glob.glob(os.path.join(ROOT, 'tests', '*.py'))}
    hit, names, grew = {os.path.basename(rel)}, {os.path.basename(rel)[:-3]}, True
    while grew:
        grew = False
        pat = re.compile(r'^\s*(from\s+(tests\.)?(%s)\s+import|import\s+(tests\.)?(%s)\b)'
                         % ('|'.join(names), '|'.join(names)), re.M)
        for f, body in bodies.items():
            if f not in hit and pat.search(body):
                hit.add(f)
                names.add(f[:-3])
                grew = True
    return sorted(f for f in hit if f.startswith('test_'))


def share(nodes, known, target):
    """`nodes` with the tests of their files that share a module-scoped fixture or a cached helper
    with one of them: what those read was computed in the other's context."""
    out, by_file = set(nodes), {}
    for n in nodes:
        if '::' in n:
            by_file.setdefault(n.split('::')[0], set()).add(n.split('::', 1)[1].split('[')[0])
    for path, picked in by_file.items():
        if path in nodes:
            continue
        reach = shared_state(show(target, path))
        want = set().union(*(reach.get(t, set()) for t in picked))
        if want:
            out |= {n for n in known if n.startswith(path + '::') and
                    reach.get(n.split('::', 1)[1].split('[')[0], set()) & want}
    return out


def shared_state(text):
    """`{test name: {shared things it reaches}}` for a test module: fixtures scoped wider than a
    function, functions under a cache decorator, and module-level names a function writes into,
    followed through fixture parameters and the module's own calls."""
    if text is None:
        return {}
    tree, defs, shared = ast.parse(text), {}, set()
    module_names = {t.id for n in tree.body if isinstance(n, ast.Assign)
                    for t in n.targets if isinstance(t, ast.Name)}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        defs.setdefault(node.name, node)
        if any(re.search(r"\b(lru_cache|cache)\b|scope\s*=\s*['\"](module|class|session|package)",
                         ast.unparse(d)) for d in node.decorator_list):
            shared.add(node.name)
        for sub in ast.walk(node):
            if isinstance(sub, ast.Global):
                shared.update(sub.names)
            written = sub.value if isinstance(sub, ast.Subscript) and isinstance(
                sub.ctx, ast.Store) else sub.func.value if isinstance(sub, ast.Call) and isinstance(
                    sub.func, ast.Attribute) and sub.func.attr in ('setdefault', 'update') else None
            if isinstance(written, ast.Name) and written.id in module_names:
                shared.add(written.id)
    if not shared:
        return {}

    def reaches(name, seen):
        if name in seen or name not in defs:
            return set()
        seen.add(name)
        node = defs[name]
        used = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} | \
            {a.arg for a in node.args.args}
        return ({name} | used) & shared | set().union(*(reaches(u, seen) for u in used))
    return {name: reaches(name, set()) for name in defs if name.startswith('test')}


def fixture_map():
    """`{tests/fixtures path: [test files]}`: the fixture literals in each test file, closed over
    the fixture-to-fixture references inside the JSONs."""
    names, refs, out = {}, {}, {}
    for dirpath, _, files in os.walk(os.path.join(ROOT, 'tests', 'fixtures')):
        for f in files:
            names[f] = os.path.relpath(os.path.join(dirpath, f), ROOT).replace('\\', '/')
    for rel in names.values():
        body = open(os.path.join(ROOT, rel), errors='ignore').read()
        refs[rel] = {other for base, other in names.items() if other != rel and base in body}
    for f in sorted(glob.glob(os.path.join(ROOT, 'tests', 'test_*.py'))):
        body = open(f, encoding='utf-8', errors='ignore').read()
        for base, rel in names.items():
            if base in body:
                out.setdefault(rel, set()).add('tests/' + os.path.basename(f))
    grew = True
    while grew:
        grew = False
        for fx, named in refs.items():
            for r in named:
                extra = out.get(fx, set()) - out.get(r, set())
                if extra:
                    out.setdefault(r, set()).update(extra)
                    grew = True
    return {k: sorted(v) for k, v in out.items()}


# ---- the map ----------------------------------------------------------------------------------
def build_map(data_dir):
    """The map from a pass's coverage and JUnit files, keyed to the commit checked out here."""
    import coverage
    data = coverage.CoverageData(os.path.join(data_dir, 'pass.coverage'))
    data.read()
    nodes, index, sets, set_index, files = [], {}, [], {}, {}

    def node(ctx):
        name = f'tests/{ctx}.py' if '|' not in ctx else ctx.split('|', 1)[1].rsplit('|', 1)[0]
        return index.setdefault(name, len(index))
    for path in sorted(data.measured_files()):
        parts = path.replace('\\', '/').split('/')
        rel = '/'.join(parts[max(k for k, p in enumerate(parts) if p in MEASURED):])
        table = {}
        for ln, ctxs in sorted((data.contexts_by_lineno(path) or {}).items()):
            key = tuple(sorted({node(c) for c in ctxs if c}))
            if key:
                table[str(ln)] = set_index.setdefault(key, len(set_index))
        if table:
            files[rel] = table
    nodes = sorted(index, key=index.get)
    sets = [list(k) for k in sorted(set_index, key=set_index.get)]
    seconds = {}
    for xml in glob.glob(os.path.join(data_dir, '*.xml')):
        for case in ET.parse(xml).iter('testcase'):
            parts = case.get('classname').split('.')
            k = next(i for i, p in enumerate(parts) if p.startswith('test_'))
            nid = '/'.join(parts[:k + 1]) + '.py::' + '::'.join(parts[k + 1:] + [case.get('name')])
            seconds[nid] = float(case.get('time') or 0.0)
    commit = git('rev-parse', 'HEAD').strip()
    out = {'commit': commit, 'built': time.strftime('%Y-%m-%d %H:%M'), 'nodes': nodes,
           'seconds': [round(seconds.get(n, 0.0), 3) for n in nodes], 'sets': sets,
           'files': files,
           'known': [p for p in git('ls-files', *MEASURED).split('\n') if p.endswith('.py')]}
    with open(MAP_PATH, 'w') as fh:
        json.dump(out, fh, separators=(',', ':'))
    print(f'wrote {MAP_PATH} at {commit[:7]}: {len(files)} modules, {len(nodes)} tests, '
          f'{len(sets)} distinct context sets')


def measure(data_dir):
    """The instrumented pass, one process per test file, resumable from what is on disk."""
    os.makedirs(data_dir, exist_ok=True)
    times = os.path.join(data_dir, 'pass_times.csv')
    done = {r['file'] for r in csv.DictReader(open(times))} if os.path.exists(times) else set()
    if not os.path.exists(times):
        with open(times, 'w', newline='') as fh:
            csv.writer(fh).writerow(['file', 'seconds', 'returncode', 'started'])
    for path in sorted(glob.glob(os.path.join(ROOT, 'tests', 'test_*.py'))):
        f = os.path.basename(path)
        if f in done:
            continue
        home = tempfile.mkdtemp(prefix='dvpass_')
        rc_path = os.path.join(home, 'cov.rc')
        with open(rc_path, 'w') as fh:
            fh.write('[run]\ncontext = ' + f[:-3] + '\n')
        env = test_env(f, home)
        env['COVERAGE_FILE'] = os.path.join(data_dir, 'pass.coverage')
        cmd = [sys.executable, '-m', 'pytest', 'tests/' + f, '-q', '-p', 'no:cacheprovider',
               '--durations=0', '--cov-config=' + rc_path, '--cov-context=test', '--cov-append',
               '--cov-report=', '--junitxml=' + os.path.join(data_dir, f[:-3] + '.xml')]
        started, t0 = time.strftime('%Y-%m-%d %H:%M:%S'), time.perf_counter()
        with open(os.path.join(data_dir, f[:-3] + '.log'), 'w', encoding='utf-8') as log:
            rc = subprocess.run(cmd + [f'--cov={m}' for m in MEASURED], cwd=ROOT, env=env,
                                stdout=log, stderr=subprocess.STDOUT).returncode
        with open(times, 'a', newline='') as fh:
            csv.writer(fh).writerow([f, f'{time.perf_counter() - t0:.1f}', rc, started])
    build_map(data_dir)


def test_env(f, home):
    """GPU 0, homes on temp folders, no live Bloomberg; `DV_SPINE_HOME` unset for the files that
    provision their own."""
    env = {k: v for k, v in os.environ.items()
           if k not in ('DERIVUS_LIVE_BLOOMBERG', 'DV_SPINE_HOME', 'COVERAGE_PROCESS_START')}
    env.update(CUDA_VISIBLE_DEVICES='0', DV_HOME=os.path.join(home, 'dv_home'),
               PYTHONIOENCODING='utf-8')
    if f not in NO_SPINE_HOME:
        env['DV_SPINE_HOME'] = os.path.join(home, 'dv_spine')
    return env


def run(selected):
    """One pytest per home policy, the selection handed over in an argument file."""
    groups, rc = {}, 0
    for n in sorted(selected):
        groups.setdefault(os.path.basename(n.split('::')[0]), []).append(n)
    for no_spine in (False, True):
        ids = [n for f, ns in groups.items() if (f in NO_SPINE_HOME) == no_spine for n in ns]
        if ids:
            home = tempfile.mkdtemp(prefix='dvsel_')
            with open(os.path.join(home, 'selected.txt'), 'w', encoding='utf-8') as fh:
                fh.write('\n'.join(ids) + '\n')
            rc |= subprocess.run([sys.executable, '-m', 'pytest', '@' + fh.name, '-q', '-p',
                                  'no:cacheprovider'], cwd=ROOT,
                                 env=test_env('test_mcp.py' if no_spine else '', home)).returncode
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--since')
    ap.add_argument('--commit')
    ap.add_argument('--dirty', action='store_true')
    ap.add_argument('--files', action='store_true')
    ap.add_argument('--time', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--measure', metavar='DIR')
    ap.add_argument('--build-map', metavar='DIR')
    args = ap.parse_args()
    if args.measure or args.build_map:
        (measure if args.measure else build_map)(args.measure or args.build_map)
        return 0
    if args.commit:
        base, target = args.commit + '^', args.commit
    elif args.since or args.dirty:
        base, target = args.since or 'HEAD', None if args.dirty else 'HEAD'
    else:
        ap.error('one of --dirty, --since, --commit, --measure, --build-map')
    paths = set(git('diff', '--name-only', base, *([target] if target else [])).split('\n'))
    if target is None:                  # new files where code lives; the root's notes are not code
        paths |= {p for p in git('ls-files', '--others', '--exclude-standard').split('\n')
                  if p.split('/')[0] in MEASURED + ('tests',)}
    paths = sorted(p for p in paths if p)
    picked = None
    if not os.path.exists(MAP_PATH):
        print('# no map recorded - FAIL OPEN: full suite (record one with --measure)',
              file=sys.stderr)
    else:
        imap = json.load(open(MAP_PATH))
        sel = Selector(imap, target)
        picked = sel.select(base, paths)
        for r in sel.reasons:
            print('#', r, file=sys.stderr)
        if picked is None:
            print(f'# FAIL OPEN: full suite - {sel.fail}', file=sys.stderr)
    picked = {'tests'} if picked is None else \
        {n.split('::')[0] for n in picked} if args.files else picked
    if not picked:
        print('# no test reaches this change', file=sys.stderr)
        return 0
    if args.time and os.path.exists(MAP_PATH):
        total = sum(s for n, s in zip(imap['nodes'], imap['seconds'])
                    if picked == {'tests'} or n in picked or n.split('::')[0] in picked)
        print(f'# {len(picked)} selected over {len({n.split("::")[0] for n in picked})} files, '
              f'{total:.0f} s as instrumented', file=sys.stderr)
    print('\n'.join(sorted(picked)))
    return run(picked) if args.run else 0


if __name__ == '__main__':
    sys.exit(main())
