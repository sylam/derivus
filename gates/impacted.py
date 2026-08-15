"""Impact-based test selection: run the tests a change can actually reach.

    python gates/impacted.py --dirty              # tests impacted by uncommitted changes
    python gates/impacted.py --since HEAD~3       # ... by a git range
    python gates/impacted.py --dirty --run        # select and execute in one step
    python gates/impacted.py --build-map          # rebuild the map from a coverage DB

THE MAP has two halves, because tests depend on two kinds of file:

  1. EXECUTION coverage (python half): which test FILES executed lines in which derivus/
     modules. Import graphs are useless here — everything imports everything through
     `derivus/__init__` — so the map comes from a coverage DB with per-test contexts,
     recorded as a byproduct of a campaign-boundary certification run:

         python -m pytest tests -q --cov=derivus --cov-context=test
         python gates/impacted.py --build-map

  2. FIXTURE reads (data half): tests load tests/fixtures/*.json at runtime, and fixtures
     chain (a job names a MarketDataFile). Static scan: string literals in tests/*.py plus
     the fixture-to-fixture references inside the JSONs, closed transitively. Rebuilt on
     every --build-map and cheap enough to refresh on every selection.

SELECTION is file-granular and FAILS OPEN: a changed file the map has never seen (a new
module, a conftest edit, derivus/__init__, utils.py by default) selects the whole suite and
says why. docs_src/, gates/, experiments/, notebooks and data/ select nothing (data/ is
regenerable and no test may read it — the repo rule).

WHAT THIS DOES NOT REPLACE: the full suite at campaign boundaries. Two measured reasons,
recorded here so the shortcut is not mistaken for the certification: (a) at least one
statistical gate is execution-ORDER-sensitive (the global torch RNG stream position moves
with the selected set — a full run in 2026-08 failed a fork gate that no targeted subset
reproduced); (b) the map is a snapshot — it goes stale the moment a test gains a dependency,
and only the next instrumented boundary run refreshes it.
"""
import argparse
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAP_PATH = os.path.join(ROOT, 'gates', 'test_impact_map.json')
# modules so widely executed that a change to them is a whole-suite event by construction
ALWAYS_ALL = {'derivus/__init__.py', 'derivus/utils.py', 'derivus/calculation.py',
              'tests/conftest.py'}


def _sh(*args):
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True).stdout


def build_fixture_map():
    """tests/fixtures path literals in each test file, closed over fixture-to-fixture
    references (a job JSON naming another fixture by path or bare filename)."""
    fixture_refs = {}                                     # fixture -> {fixtures it names}
    fdir = os.path.join(ROOT, 'tests', 'fixtures')
    names = {}
    for dirpath, _, files in os.walk(fdir):
        for f in files:
            if f.endswith('.json'):
                rel = os.path.relpath(os.path.join(dirpath, f), ROOT)
                names[f] = rel
    for rel in list(names.values()):
        try:
            body = open(os.path.join(ROOT, rel), errors='ignore').read()
        except OSError:
            continue
        fixture_refs[rel] = {other for base, other in names.items()
                             if other != rel and base in body}
    test_map = {}                                         # fixture -> {test files}
    tdir = os.path.join(ROOT, 'tests')
    for f in sorted(os.listdir(tdir)):
        if not (f.startswith('test_') and f.endswith('.py')):
            continue
        body = open(os.path.join(tdir, f), errors='ignore').read()
        for base, rel in names.items():
            if base in body:
                test_map.setdefault(rel, set()).add('tests/' + f)
    # close: a test reading fixture A also depends on every fixture A names, transitively
    changed = True
    while changed:
        changed = False
        for fx, refs in fixture_refs.items():
            for r in refs:
                extra = test_map.get(fx, set()) - test_map.get(r, set())
                if extra:
                    test_map.setdefault(r, set()).update(extra)
                    changed = True
    return {k: sorted(v) for k, v in sorted(test_map.items())}


def build_map(coverage_db='.coverage'):
    try:
        import coverage
    except ImportError:
        sys.exit('coverage is not importable - pip install coverage, or run the instrumented '
                 'suite via pytest-cov first')
    db = os.path.join(ROOT, coverage_db)
    if not os.path.exists(db):
        sys.exit(f'{coverage_db} not found - record one with: '
                 f'python -m pytest tests -q --cov=derivus --cov-context=test')
    cov = coverage.Coverage(data_file=db)
    cov.load()
    data = cov.get_data()
    py_map = {}
    for path in data.measured_files():
        rel = os.path.relpath(path, ROOT)
        if not rel.startswith('derivus') or not rel.endswith('.py'):
            continue
        tests = set()
        for _, contexts in (data.contexts_by_lineno(path) or {}).items():
            for ctx in contexts:
                m = re.match(r'(tests/[^:]+\.py)', ctx.replace('.', '/', 1) + '.py') \
                    if not ctx.startswith('tests/') else re.match(r'(tests/[^:]+\.py)', ctx)
                name = ctx.split('::', 1)[0]
                if name.startswith('tests/'):
                    tests.add(name)
        if tests:
            py_map[rel] = sorted(tests)
    out = {'python': py_map, 'fixtures': build_fixture_map()}
    json.dump(out, open(MAP_PATH, 'w'), indent=1)
    n = sum(len(v) for v in py_map.values())
    print(f'wrote {MAP_PATH}: {len(py_map)} modules, {len(out["fixtures"])} fixtures, '
          f'{n} python edges')


def changed_files(since=None, dirty=False):
    if dirty:
        out = _sh('git', 'status', '--porcelain')
        return [line[3:].split(' -> ')[-1] for line in out.splitlines() if line.strip()]
    return [f for f in _sh('git', 'diff', '--name-only', since, 'HEAD').splitlines() if f]


def select(files):
    have_map = os.path.exists(MAP_PATH)
    impact_map = json.load(open(MAP_PATH)) if have_map else {'python': {}, 'fixtures': {}}
    # the fixture half is cheap - always fresh
    impact_map['fixtures'] = build_fixture_map()
    picked, reasons = set(), []
    for f in files:
        if f.startswith(('docs_src/', 'gates/', 'experiments/', 'notebooks/', 'data/',
                         'artifacts/')) or f in ('mkdocs.yml',):
            continue
        if f in ALWAYS_ALL:
            return None, [f'{f}: whole-suite module by construction']
        if f.startswith('tests/') and f.endswith('.py'):
            picked.add(f)
            continue
        if f.startswith('tests/fixtures/'):
            hits = impact_map['fixtures'].get(f)
            if hits is None:
                return None, [f'{f}: fixture unknown to the map - fail open']
            picked.update(hits)
            reasons.append(f'{f}: {len(hits)} test files via fixture map')
            continue
        if f.startswith('derivus/') and f.endswith('.py'):
            if not have_map:
                return None, [f'{f}: no coverage map recorded yet - fail open '
                              f'(build one with --build-map after an instrumented run)']
            hits = impact_map['python'].get(f)
            if hits is None:
                return None, [f'{f}: module unknown to the coverage map - fail open']
            picked.update(hits)
            reasons.append(f'{f}: {len(hits)} test files via coverage map')
            continue
        return None, [f'{f}: unclassified path - fail open']
    return sorted(picked), reasons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--since')
    ap.add_argument('--dirty', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--build-map', action='store_true')
    ap.add_argument('--coverage-db', default='.coverage')
    args = ap.parse_args()

    if args.build_map:
        build_map(args.coverage_db)
        return
    if not (args.since or args.dirty):
        ap.error('one of --since/--dirty (or --build-map) is required')
    files = changed_files(since=args.since, dirty=args.dirty)
    picked, reasons = select(files)
    for r in reasons:
        print('#', r, file=sys.stderr)
    if picked is None:
        print('# FAIL OPEN: full suite', file=sys.stderr)
        picked = ['tests']
    elif not picked:
        print('# no impacted tests', file=sys.stderr)
        return
    print(' '.join(picked))
    if args.run:
        os.execvp('python', ['python', '-m', 'pytest', '-q'] + picked)


if __name__ == '__main__':
    main()
