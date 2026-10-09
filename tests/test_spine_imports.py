########################################################################
# Copyright (C)  Shuaib Osman (vretiel@gmail.com)
# This file is part of Derivus.
#
# Derivus is free for noncommercial use under the terms of the PolyForm
# Noncommercial License 1.0.0. You should have received a copy of the license
# along with Derivus. If not, see
# <https://polyformproject.org/licenses/noncommercial/1.0.0>.
#
# Derivus is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
########################################################################
"""The spine's import surface, its packaging, and the verbs `DV_Spine` answers to.

The book of record depends on stdlib plus `cryptography` - no engine, no torch, no network. That
is a PROPERTY OF THE SOURCE, so the first gate reads it rather than the loaded modules; the second
proves it again by watching what lands in a fresh interpreter's `sys.modules`. Both cover every
`derivus_spine/*.py` by glob, so a module added tomorrow is gated the day it appears.

The packaging gate reads `setup.py` as text and AST and installs nothing: the spine ships as a
sibling package, `cryptography` is an EXTRA rather than a base dependency, the console script
exists. The CLI gates drive `DV_Spine` as a subprocess on real homes.
"""
import ast
import glob
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPINE = os.path.join(ROOT, 'derivus_spine')
SETUP = os.path.join(ROOT, 'setup.py')

#: `sys.stdlib_module_names` IS this question's answer from 3.10 on. Below it, NAME the modules
#: the spine may reach for rather than wave the question through - a gate that silently degrades
#: to "anything at all" is not a gate.
STDLIB_FALLBACK = frozenset(
    ('__future__ argparse base64 binascii collections contextlib datetime errno functools getpass '
     'glob hashlib hmac io itertools json logging math os pathlib random re secrets shutil stat '
     'string struct sys tempfile time typing uuid warnings').split())
STDLIB = frozenset(getattr(sys, 'stdlib_module_names', None) or STDLIB_FALLBACK)

#: The spine's own name is not a dependency (a package importing itself is structure), and
#: `cryptography` is the single declared exception the design allows.
ALLOWED = STDLIB | {'cryptography', 'derivus_spine'}
FORBIDDEN = {'derivus', 'torch', 'numpy', 'pandas', 'scipy', 'requests', 'duckdb'}


def spine_sources():
    """Every module of the package as it stands right now, at any depth - the glob is the
    point: a subpackage cannot smuggle an import past a gate that descends into it."""
    return sorted(glob.glob(os.path.join(SPINE, '**', '*.py'), recursive=True))


def imported_names(source):
    """The top-level names a file imports, however deep in it the import sits. Relative imports
    are skipped rather than allowed: they resolve inside the package by construction and carry no
    module name to judge."""
    names = set()
    with open(source, encoding='utf-8') as handle:
        tree = ast.parse(handle.read(), filename=source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split('.')[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and not node.level:
            names.add((node.module or '').split('.')[0])
    return names


def test_one_module_under_derivus_imports_the_spine_and_it_is_the_seam():
    """THE ONE-IMPORTER LAW, as a machine-checked fact rather than a promise. `derivus/spine.py` is
    the whole seam: no other module under `derivus/` may name `derivus_spine`, so an engine tree
    without the extra keeps working and the record's vocabulary has one door.

    Read off the SOURCE, at any depth, so an import that never executes still counts - and a
    lazy import inside a function counts exactly as much as one at the top of the file.

    Killing mutation: a second module under `derivus/` importing the spine.
    """
    importers = set()
    for source in sorted(glob.glob(os.path.join(ROOT, 'derivus', '**', '*.py'), recursive=True)):
        with open(source, encoding='utf-8') as handle:
            tree = ast.parse(handle.read(), filename=source)
        for node in ast.walk(tree):
            names = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                     else [node.module or ''] if isinstance(node, ast.ImportFrom) and not node.level
                     else [])
            if any(name.split('.')[0] == 'derivus_spine' for name in names):
                importers.add(os.path.relpath(source, ROOT).replace(os.sep, '/'))

    assert importers == {'derivus/spine.py'}, importers


def test_the_spine_imports_nothing_but_the_standard_library_and_cryptography():
    """The dependency budget, read off the source of every module the package has. `cryptography`
    is in - bodies are sealed and checkpoints signed from genesis; the engine, torch and the HTTP
    client are out.

    Killing mutation: a spine module importing `numpy`.
    """
    sources = spine_sources()
    assert sources, 'derivus_spine holds no modules at all'
    assert os.path.join(SPINE, 'cli.py') in sources

    for source in sources:
        imported = imported_names(source)
        assert imported <= ALLOWED, (os.path.basename(source), sorted(imported - ALLOWED))
        assert imported.isdisjoint(FORBIDDEN), (os.path.basename(source),
                                                sorted(imported & FORBIDDEN))
    # non-vacuous: a package that imported nothing would pass the two assertions above
    assert set().union(*(imported_names(source) for source in sources))


def test_importing_the_spine_lands_neither_the_engine_nor_torch():
    """The source gate's answer proved a second way, because the first trusts the parser: a FRESH
    interpreter imports the package and reports what arrived. Run out of the repo root so the tree
    under test is this checkout.

    EVERY MODULE BY GLOB: `__init__.py` keeps its surface to the truth layer, so importing the
    package alone would leave the CLI, custody, identity and the verbs unloaded and unwitnessed.

    Killing mutation: a spine module importing `numpy`.
    """
    modules = sorted('derivus_spine.{}'.format(os.path.basename(source)[:-3])
                     for source in spine_sources()
                     if os.path.basename(source) != '__init__.py')
    assert modules, 'derivus_spine holds no modules to import'
    code = ('import json, sys; import derivus_spine; import {}; '
            'print(json.dumps(sorted({{name.split(".")[0] for name in sys.modules}})))'.format(
                ', '.join(modules)))
    done = subprocess.run([sys.executable, '-c', code], cwd=ROOT, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, universal_newlines=True)

    assert done.returncode == 0, done.stderr
    landed = set(json.loads(done.stdout))
    assert 'derivus_spine' in landed, 'the package did not import'
    assert landed.isdisjoint(FORBIDDEN), sorted(landed & FORBIDDEN)


def setup_call():
    """`setup.py` read, never run: importing it executes setuptools, and what is under test is the
    declaration. Module-level names are carried along because the file composes its extras out of
    them, so resolving one means following the Name to its assignment."""
    with open(SETUP, encoding='utf-8') as handle:
        tree = ast.parse(handle.read(), filename=SETUP)
    env = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    env[target.id] = node.value
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == 'setup'):
            return {kw.arg: kw.value for kw in node.keywords}, env
    raise AssertionError('setup.py declares no setup() call')


def literal(node, env):
    """Resolve a declaration node the way the file means it - a list, a name standing for one, or
    the sum of several (which is how `desk` is spelled)."""
    if isinstance(node, ast.Name):
        return literal(env[node.id], env)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return literal(node.left, env) + literal(node.right, env)
    return ast.literal_eval(node)


def test_the_spine_ships_as_a_sibling_package_with_cryptography_as_an_extra():
    """Three declarations: the spine is in the wheel beside its siblings; `cryptography` is an
    EXTRA, so an engine install grows no crypto dependency and `desk` keeps its edge (they compose
    as `derivus[desk,enterprise]`); and the console script exists.

    Killing mutation: the enterprise extra declaring another pin.
    """
    kwargs, env = setup_call()

    packages = kwargs['packages']
    include = literal({kw.arg: kw.value for kw in packages.keywords}['include'], env)
    assert 'derivus_spine' in include and 'derivus_spine.*' in include

    extras_node = kwargs['extras_require']
    extras = dict(zip([ast.literal_eval(key) for key in extras_node.keys], extras_node.values))
    assert literal(extras['enterprise'], env) == ['cryptography>=42']
    # the edge is unchanged: a desk install pulls no crypto, and the base install pulls none either
    assert not any('cryptography' in item for item in literal(extras['desk'], env))
    assert not any('cryptography' in item for item in literal(kwargs['install_requires'], env))

    scripts = literal(kwargs['entry_points'], env)['console_scripts']
    assert 'DV_Spine = derivus_spine.cli:main' in scripts


def spine(*argv, **kwargs):
    """`DV_Spine` as a subprocess, which is the only honest way to gate an exit code."""
    return subprocess.run([sys.executable, '-m', 'derivus_spine.cli'] + list(argv), cwd=ROOT,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
                          env=kwargs.get('env'))


def test_the_cli_mints_verifies_checkpoints_and_reports(tmp_path):
    """The runbook end to end on a real home: mint, verify entitled, verify as an unentitled
    replica, sign the head, read where it stands. Every answer is JSON on stdout; the head moving
    across the checkpoint is what says the verb did something.

    Killing mutation: the checkpoint verb reporting without appending.
    """
    home = str(tmp_path / 'spine')

    minted = spine('init', '--home', home)
    assert minted.returncode == 0, minted.stderr
    assert json.loads(minted.stdout)

    entitled = spine('verify', '--home', home)
    assert entitled.returncode == 0, entitled.stderr
    report = json.loads(entitled.stdout)
    assert report['head_lsn'] >= 4 and len(report['head_hash']) == 64

    unentitled = spine('verify', '--chain-only', '--home', home)
    assert unentitled.returncode == 0, unentitled.stderr
    assert json.loads(unentitled.stdout)['mode'] != report['mode'], 'both modes read the same'

    signed = spine('checkpoint', '--home', home)
    assert signed.returncode == 0, signed.stderr

    where = spine('status', '--home', home)
    assert where.returncode == 0, where.stderr
    standing = json.loads(where.stdout)
    assert standing['head_lsn'] > report['head_lsn'], 'the checkpoint did not extend the log'
    assert standing['home'] == os.path.abspath(home)
    assert standing['bodies_readable'] is True

    # the home is answered by the environment too, and to the same place - `DV_HOME` one level over
    named = dict(os.environ, DV_SPINE_HOME=home)
    assert json.loads(spine('status', env=named).stdout) == standing


def test_the_cli_declares_a_policy_from_a_file_and_reports_what_is_in_force(tmp_path):
    """The policy-file editor this deployment has, in its CLI form. A document goes on the record
    from a JSON file and comes back with the blob it was stored under and the LSN it stands at;
    a reserved name nobody declared reports as nulls, so silence is never mistaken for absence.

    A document no parser reads NEVER LANDS - the refusal is the library's own sentence with exit 1
    - and what was in force before it stays in force, which is the half a round trip exists for.
    The capabilities document `grant` declares reads back the same way, and a home that lost it
    still lists every other policy, naming the one it cannot read.

    Killing mutations: the capabilities name refused by the reading verb, which leaves a head
    re-seating its desk with no way to read the document in force; one unreadable policy failing
    the whole listing, which hides every policy that does read.
    """
    home = str(tmp_path / 'spine')
    assert spine('init', '--home', home).returncode == 0
    path = tmp_path / 'tiers.json'
    path.write_text(json.dumps({
        'tiers': [{'name': 'auto', 'max_notional': {'amount': 5000000.0, 'currency': 'USD'}},
                  {'name': 'desk', 'four_eyes': True}],
        'designations': {'settlement_export': 'official'}}), encoding='utf-8')

    declared = spine('declare', 'tiers', str(path), '--actor', 'subject-deployment', '--home', home)
    assert declared.returncode == 0, declared.stderr
    answer = json.loads(declared.stdout)
    assert answer['policy'] == 'tiers' and len(answer['blob']) == 64

    standing = json.loads(spine('policy', 'tiers', '--home', home).stdout)['tiers']
    assert (standing['blob'], standing['lsn']) == (answer['blob'], answer['lsn'])
    assert [tier['name'] for tier in standing['document']['tiers']] == ['auto', 'desk']

    # THE BLOB AND ITS POSITION COME OFF ONE WALK. `policy_declared` is open-bodied, so a
    # declaration under a reserved name carrying no blob is legal and is a fact about the name -
    # and a readout that joined a second fold to this one would print the standing blob at ITS LSN
    from derivus_spine import SpineLog
    log = SpineLog(home)
    try:
        inline = log.append('policy_declared', {'policy': 'tiers', 'note': 'by hand'},
                            actor='subject-deployment')['lsn']
    finally:
        log.close()
    joined = json.loads(spine('policy', 'tiers', '--home', home).stdout)['tiers']
    assert inline > answer['lsn']
    assert (joined['blob'], joined['lsn']) == (answer['blob'], answer['lsn']), \
        'the readout borrowed the position of a declaration that carried no blob'

    every = json.loads(spine('policy', '--home', home).stdout)
    assert sorted(every) == ['capabilities', 'firmness', 'fixings', 'tiers', 'tolerance']
    assert every['tolerance'] == every['capabilities'] == {'blob': None, 'document': None,
                                                           'lsn': None}

    broken = tmp_path / 'broken.json'
    broken.write_text(json.dumps({'tiers': [{'name': 'auto', 'firm': True}]}), encoding='utf-8')
    refused = spine('declare', 'tiers', str(broken), '--actor', 'subject-deployment', '--home', home)
    assert refused.returncode == 1 and 'firmness' in refused.stderr
    assert 'Traceback' not in refused.stderr
    assert json.loads(spine('policy', 'tiers', '--home', home).stdout)['tiers']['blob'] \
        == answer['blob'], 'a refused declaration moved what is in force'

    # the capabilities document is `grant`'s own file, and this verb reads it back as stored
    document = {'grants': [{'subject': 'subject-deployment', 'verb': 'admin', 'book': '*'}],
                'read': []}
    path.write_text(json.dumps(document), encoding='utf-8')
    granted = json.loads(spine('grant', '--file', str(path), '--actor', 'subject-deployment',
                               '--home', home).stdout)
    assert json.loads(spine('policy', 'capabilities', '--home', home).stdout)['capabilities'] == {
        'blob': granted['blob'], 'document': document, 'lsn': granted['lsn']}

    # a copy lacking that document lists every other policy and says which one it cannot read
    os.remove(os.path.join(home, 'blobs', granted['blob'][:2], granted['blob'][2:4],
                           granted['blob']))
    listed = spine('policy', '--home', home)
    assert listed.returncode == 0, listed.stderr
    every = json.loads(listed.stdout)
    assert every['tiers']['blob'] == answer['blob'] and every['capabilities']['blob'] is None
    assert granted['blob'] in every['capabilities']['unreadable']
    assert spine('policy', 'capabilities', '--home', home).returncode == 1


def test_verifying_a_home_that_is_not_there_refuses_by_name(tmp_path):
    """A refusal reaches the terminal as a SENTENCE and exit 1 - naming the thing and the remedy -
    never as a traceback. Nothing is minted on the way out: a verify that provisioned would be a
    second source of truth.

    Killing mutation: the CLI catching nothing, which reaches the terminal as a traceback.
    """
    missing = str(tmp_path / 'nothing-here')

    refused = spine('verify', '--home', missing)

    assert refused.returncode == 1
    assert refused.stderr.strip() and 'Traceback' not in refused.stderr
    assert 'home' in refused.stderr.lower()
    assert not os.path.exists(missing), 'a refusal wrote a home'
