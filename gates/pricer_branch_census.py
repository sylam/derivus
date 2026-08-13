"""BRANCH-EXECUTION CENSUS: which branch of which pricer does no test in the suite RUN?

The measuring half of ``tests/test_pricer_branch_ledger.py`` - read that file first, it carries the
defect this exists for and the shape of the ledger. This half runs the whole suite under
``coverage --branch`` with the tracer restricted to ``derivus/pricing.py``, maps every arc the run
never took onto the enclosing pricer, and asserts the result against the committed ``UNREACHED``
ledger in BOTH directions:

  * an arc the ledger does not name  -> a NEW unexecuted branch has appeared;
  * a ledger entry the run now takes -> a fixture reached it, delete the line (no stale alibis).

The third failure - a ledger entry the AST cannot resolve - is checked here too, but it is the
suite's job (it needs no coverage and costs milliseconds) and it is why the ledger lives in tests/.

WHY THE WHOLE SUITE. Any test may be the one that takes an arc, so a subset can only OVER-report,
and an over-reported census is a work-list with invented work in it. The run is therefore long
(~50 minutes on this repo), the fact it produces changes only when a pricer or a fixture changes,
and re-analysis of an existing data file is free - which is the whole reason this is a gate and not
a test. It traces one test FILE at a time and resumes from what is already on disk, so an
interrupted run is not a lost one; the data lands in a fixed directory under the system temp, never
in the repo.

PERCENTAGES ARE NOT THE OUTPUT. ``pricing.py`` was at high line coverage throughout the years the
barrier leg was 1432% wrong; what was missing was one ARC. The output is the list of arcs.

    CUDA_VISIBLE_DEVICES=0 python gates/pricer_branch_census.py            # measure, then assert
    CUDA_VISIBLE_DEVICES=0 python gates/pricer_branch_census.py --data F   # re-analyse F, no run
    CUDA_VISIBLE_DEVICES=0 python gates/pricer_branch_census.py --emit     # print a fresh ledger
"""
import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tests'))

import coverage

from test_pricer_branch_ledger import (FAMILY, MUST_COVER, PRICING, UNREACHED,
                                       anchors, branch_sites, family_scopes)


def measure(data_file):
    """`{(qualname, text, direction)}` for every family arc the run never took.

    A scope NO test calls is reported once, as `never-called`, and its interior arcs are dropped:
    coverage lists every arc of a never-executed region, so a dead closure would otherwise arrive
    as dozens of findings that are all one fact. Its `def` line is excluded from the liveness test
    because that line executes when the closure is CREATED, which says nothing about it being
    called - the exact distinction the barrier leg died on."""
    cov = coverage.Coverage(data_file=data_file)
    cov.load()
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, 'cov.json')
        cov.json_report(morfs=[PRICING], outfile=out)
        entry = next(v for k, v in json.load(open(out))['files'].items()
                     if k.endswith('derivus/pricing.py'))
    executed = set(entry['executed_lines'])

    dead = [(s, e, n) for s, e, n in family_scopes() if not any(s < ln <= e for ln in executed)]
    outermost = [d for d in dead if not any(s < d[0] and d[1] <= e for s, e, _ in dead)]
    found = {(n, f'def {n.split(".")[-1]}', 'never-called') for _, _, n in outermost}

    sites = branch_sites()
    for line, dest in entry.get('missing_branches', []):
        if line not in sites or any(s < line <= e for s, e, _ in dead):
            continue
        name, text, body, other, tag = sites[line]
        found.add((name, text, ('body' if dest in body else other) + tag))
    return found


def run_suite(data_file):
    """Trace the suite ONE TEST FILE AT A TIME into `<data_file>.d/`, then combine.

    One traced `pytest tests/` would be simpler and it is what this did first - but coverage writes
    its data at process exit, so the run that was killed at 96% produced nothing at all. Per file,
    an interruption costs one file, a re-run resumes from what is already on disk, and the per-file
    data sets answer "which test reaches this arc", which no combined run can.

    The tree is hashed before and after: arcs are line numbers resolved against the source AFTER the
    run, so an edit landing mid-run reattributes every arc below it to the wrong branch."""
    before = hashlib.md5(open(PRICING, 'rb').read()).hexdigest()
    parts = data_file + '.d'
    os.makedirs(parts, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(ROOT, 'tests', 'test_*.py'))):
        name = os.path.basename(path)[:-3]
        part = os.path.join(parts, f'.coverage.{name}')
        if os.path.exists(part):
            continue
        subprocess.run([sys.executable, '-m', 'coverage', 'run', '--branch',
                        '--include=*/derivus/pricing.py', '-m', 'pytest', f'tests/{name}.py', '-q'],
                       cwd=ROOT, env=dict(os.environ, COVERAGE_FILE=part), check=False)
    assert hashlib.md5(open(PRICING, 'rb').read()).hexdigest() == before, \
        'pricing.py changed during the run - the arcs no longer line up, re-run'
    cov = coverage.Coverage(data_file=data_file)
    cov.combine(data_paths=sorted(glob.glob(os.path.join(parts, '.coverage.*'))), keep=True)
    cov.save()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', help='an existing coverage data file; skips the suite run')
    ap.add_argument('--emit', action='store_true', help='print a fresh UNREACHED ledger and stop')
    args = ap.parse_args()

    # stable so an interrupted run resumes, and OUTSIDE the repo because coverage data is
    # regenerable and has no business being committable
    data_file = args.data or os.path.join(tempfile.gettempdir(), 'derivus_census', '.coverage')
    if not args.data:
        os.makedirs(os.path.dirname(data_file), exist_ok=True)
        run_suite(data_file)
    found = measure(data_file)

    if args.emit:
        print('UNREACHED = {')
        for key in sorted(found):
            print(f'    {key!r}:\n        "",')
        print('}')
        return 0

    print(f'\n{len(found)} unexecuted branch arcs across {len(FAMILY)} pricers; '
          f'the ledger names {len(UNREACHED)}\n')
    for key in sorted(found):
        print(f'  {key[0]}\n      {key[2]:<12s} {key[1][:96]}'
              f'\n      -> {UNREACHED.get(key, "*** NOT ON THE LEDGER ***")}')

    failures = (
        ('unexecuted branches the ledger does not name', sorted(k for k in found
                                                                if k not in UNREACHED)),
        ('ledger entries a test now reaches - delete them', sorted(k for k in UNREACHED
                                                                   if k not in found)),
        ('ledger entries the AST cannot resolve - re-anchor them', sorted(k for k in UNREACHED
                                                                          if k not in anchors())),
        ('MUST_COVER branches that stopped being executed', sorted(MUST_COVER & found)),
    )
    for label, rows in failures:
        if rows:
            print(f'\nFAIL: {label}:')
            for key in rows:
                print(f'  {key}')
    return 1 if any(rows for _, rows in failures) else 0


if __name__ == '__main__':
    sys.exit(main())
