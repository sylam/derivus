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
"""The spine's import surface, its packaging, and the four verbs `DV_Spine` answers to.

The book of record depends on stdlib plus `cryptography` - no engine, no torch, no network. That
is a PROPERTY OF THE SOURCE, so the first gate reads it rather than the loaded modules; the second
proves it again by watching what lands in a fresh interpreter's `sys.modules`. Both cover every
`derivus_spine/*.py` by glob, so a module added tomorrow is gated the day it appears.

The packaging gate reads `setup.py` as text and AST and installs nothing: the spine ships as a
sibling package, `cryptography` is an EXTRA rather than a base dependency, the console script
exists.

The CLI smoke gates state their precondition rather than dying downstream of it - the CLI is a
mouth over the core modules, and the skip heals the moment `derivus_spine/log.py` exists.
"""
import ast
import glob
import json
import os
import subprocess
import sys

import pytest

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
#: `cryptography` is the single declared exception the brief allows.
ALLOWED = STDLIB | {'cryptography', 'derivus_spine'}
FORBIDDEN = {'derivus', 'torch', 'numpy', 'pandas', 'scipy', 'requests', 'duckdb'}


def spine_sources():
    """Every module of the package as it stands right now - the glob is the point."""
    return sorted(glob.glob(os.path.join(SPINE, '*.py')))


def spine_has(name):
    return os.path.isfile(os.path.join(SPINE, name))


needs_spine_package = pytest.mark.skipif(
    not spine_has('__init__.py'),
    reason='derivus_spine/__init__.py: the package surface lands with the core modules, and this '
           'gate imports the package rather than reading it')
needs_spine_core = pytest.mark.skipif(
    not spine_has('log.py'),
    reason='derivus_spine/log.py: the CLI is a mouth over the writer, so its smoke gates wait for '
           'the core modules and go green the moment they land')


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


def test_the_spine_imports_nothing_but_the_standard_library_and_cryptography():
    """The dependency budget, read off the source of every module the package has. `cryptography`
    is in - bodies are sealed and checkpoints signed from genesis; the engine, torch and the HTTP
    client are out."""
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


@needs_spine_package
def test_importing_the_spine_lands_neither_the_engine_nor_torch():
    """The source gate's answer proved a second way, because the first trusts the parser: a FRESH
    interpreter imports the package and reports what arrived. Run out of the repo root so the tree
    under test is this checkout.

    EVERY MODULE BY GLOB: `__init__.py` keeps its surface to the truth layer, so importing the
    package alone would leave the CLI, custody, identity and the verbs unloaded and unwitnessed.
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
    as `derivus[desk,enterprise]`); and the console script exists."""
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


def test_the_cli_declares_four_verbs_and_the_home_flag():
    """Gated off the source, before the core exists to drive it. Four verbs and no fifth - what
    the spine grows later arrives as an event or a fold, not a new mouth."""
    with open(os.path.join(SPINE, 'cli.py'), encoding='utf-8') as handle:
        tree = ast.parse(handle.read(), filename='cli.py')

    verbs, flags = set(), set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr in ('add_parser', 'add_argument') and node.args:
            named = node.args[0]
            if isinstance(named, ast.Constant) and isinstance(named.value, str):
                (verbs if node.func.attr == 'add_parser' else flags).add(named.value)

    assert verbs == {'init', 'verify', 'checkpoint', 'status'}
    assert {'--home', '--chain-only', '--actor'} <= flags


def spine(*argv, **kwargs):
    """`DV_Spine` as a subprocess, which is the only honest way to gate an exit code."""
    return subprocess.run([sys.executable, '-m', 'derivus_spine.cli'] + list(argv), cwd=ROOT,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
                          env=kwargs.get('env'))


@needs_spine_core
def test_the_cli_mints_verifies_checkpoints_and_reports(tmp_path):
    """The runbook end to end on a real home: mint, verify entitled, verify as an unentitled
    replica, sign the head, read where it stands. Every answer is JSON on stdout; the head moving
    across the checkpoint is what says the verb did something."""
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


@needs_spine_core
def test_verifying_a_home_that_is_not_there_refuses_by_name(tmp_path):
    """A refusal reaches the terminal as a SENTENCE and exit 1 - naming the thing and the remedy -
    never as a traceback. Nothing is minted on the way out: a verify that provisioned would be a
    second source of truth."""
    missing = str(tmp_path / 'nothing-here')

    refused = spine('verify', '--home', missing)

    assert refused.returncode == 1
    assert refused.stderr.strip() and 'Traceback' not in refused.stderr
    assert 'home' in refused.stderr.lower()
    assert not os.path.exists(missing), 'a refusal wrote a home'
