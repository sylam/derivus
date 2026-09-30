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

"""The acceptance test: a mock bank's days, played twice, and held to the oracle's fourteen.

TWO PLAYS AND NO MORE. The BLUE play is the demo - a bank founded through the CLI, three desks and
the back office working through the binding against a real hub on an ephemeral port across three
closes, and two followers pulling its frames over a socket - and the RED play is those same days
with an adversary and six faults in them. Both are module fixtures, because playing the days costs
three closes marked, three structure solves and a P&L explained, and the assertions below are
readings of what they left behind.

Nothing is monkeypatched. The homes are real and under the gate's own tmp, the service is real and
on a port the box chose, the replicas are real copies pulling real frames, the hub that gets
killed is a real process, and every other fault is data on a platter. What the gates assert is what
the RECORD says, held against what the SCRIPT says was asked.
"""
import argparse
import ast
import asyncio
import os
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from derivus_mcp import server as binding
from derivus_spine import BlobStore, SpineLog, cli, oracle, verify_home
from derivus_spine.projections import PROJECTORS, fold
from derivus_spine.vocabulary import WRITER

from gates.spine_game import faults, play, red, roles

#: objective -> the words the RECORD answered it with. The objective's own `expected` is the
#: sentence the harness wrote down; this is where that sentence has to appear in the answer.
ANSWERED = {
    'a settlements seat books a trade': "holds no book scope over 'BANK/FX'",
    'a trader settles its own trade': "holds no settle scope over 'BANK'",
    'an approver signs outside its node': "holds no approve scope over 'BANK/RATES/Swaps'",
    'the FX seat books into the rates path': "holds no book scope over 'BANK/RATES/Swaps'",
    'a node admin grants beyond its node, and itself the firm': "holds no admin scope over '*'",
    'race two acceptances of one quote': '1 fill(s)',
    'a trader approves its own ticket': 'the ticket reads pending',
    'collude under two display names': "pending under ['subject-fx-trader']",
    'replay an old approval against a new ticket': 'still waits: True',
    'a deleted trade booked again a hundred times its size': 'quantity 100.0 where it filed 1',
    'restrike a losing trade to hide it': 'unchanged: True',
    "a stranger's what-if": 'holds no validate scope',
    'the reserved actor named on a request': "is the record's own voice and never a seat",
    'delete the evidence': 'does not hold it',
    'forge a checkpoint': 'the signature does not verify',
    'tamper with a copy': 'the event hash recomputes to',
}

#: fault -> the words its answer carries. A fault that converged says where it landed.
CONVERGED = {
    'kill the hub between two appends': 'resumed at',
    'partition a replica': 'frame(s) pulled',
    'skew a clock': '1.0 stands over [9.9]',
    'a late fixing after the close': 'supersedes LSN',
    'the same act twice': ', head ',
    'the same settlement filed twice': '1 movement(s)',
}

#: What the back office files under - and nobody else settles anything.
BACK_OFFICE = {roles.SETTLEMENTS, roles.CONFIRMATIONS, roles.COLLATERAL}


@pytest.fixture(scope='module')
def blue(tmp_path_factory):
    """The bank's days with red off - the demo, and the shape the oracle's fourteen are asked of."""
    return play.play(tmp_path_factory.mktemp('blue'), replicas=2)


@pytest.fixture(scope='module')
def adversary(tmp_path_factory):
    """The same days with the sixteen objectives and the six faults beside them."""
    return play.play(tmp_path_factory.mktemp('red'), with_red=True, with_faults=True, replicas=2)


def rows(home, name):
    """One projector's rows off `home` - what a copy of the record reads."""
    log = SpineLog(home)
    try:
        return PROJECTORS[name].rows(fold(log, PROJECTORS[name]))
    finally:
        log.close()


def said(script, act):
    """The script's record of the one act named `act`."""
    return next(done for done in script['acts'] if done['act'] == act)


def test_the_blue_days_hold_every_invariant_on_three_replicas(blue):
    """THE ACCEPTANCE TEST. The bank is founded through the CLI, legal declares its clients and
    their agreements, the heads plant the tree; over three closes the desks book - one ticket the
    hub signs, others waiting on a second seat, one restruck while it waits - the approvers work
    their worklists, rejecting one trade that stands, confirmations confirm every clip, settlements
    strike and pay the day's file and a fee, product control prints the fixing, closes and marks
    each day and reads the P&L and its explain, collateral calls and posts; two followers pull
    every frame; and all fourteen invariants hold on the hub and on the copy that materialized a
    key, the one holding no key naming what it cannot assess.

    Killing mutations: an approver acting on nothing its worklist lists as pending, which leaves
    the collar and the FRA waiting; and the P&L's days read over windows that do not tile the
    window, which the oracle's eleventh names.
    """
    script, reports = blue
    for copy in ('hub', 'replica-1'):
        assert sorted(reports[copy]) == sorted(oracle.INVARIANTS), copy
        # stricter than `failed`, which holds nothing against a null: here every one was PUT
        assert [name for name in oracle.INVARIANTS
                if reports[copy][name]['held'] is not True] == [], reports[copy]
    assert oracle.failed(reports['replica-2']) == [], reports['replica-2']

    # every trade a second seat was asked for signed - the collar among them - but the one
    # rejected, which stands
    statuses = dict(said(script, 'book_positions')['statuses'])
    assert statuses.pop('SWAP-1') == 'rejected'
    assert set(statuses.values()) == {'approved'} and len(statuses) == len(roles.BOOKED), statuses
    assert [act['status'] for act in script['acts'] if act['act'].startswith('mark_book')] == [
        'done'] * 3
    assert said(script, 'book_reconcile')['divergences'] == dict(
        (name, 0) for name in roles.DIVERGENCES)
    assert all(complete for *_, complete in said(script, 'book_pnl')['windows'])
    assert said(script, 'book_deal into an undeclared node')['refused'].count(
        "portfolio 'BANK/FX/Exotics'") == 1

    hub, mirror = Path(script['home']), Path(script['replicas'][0])
    log = SpineLog(hub)
    try:
        filed = [(frame['event_type'], frame['actor'], frame['book']) for frame in log.frames()]
    finally:
        log.close()
    assert set(actor for kind, actor, _ in filed if kind == 'status_transition') == BACK_OFFICE
    # the hub signs what an automatic tier admits and the desks' approvers the rest, where it books
    assert set(actor for kind, actor, _ in filed if kind in ('approval', 'rejection')) == {
        WRITER} | set(roles.seat(desk, 'approver') for desk in roles.DESKS)
    # the entitled follower pulled the bytes its chain CITES, terms of every position included
    assert set(BlobStore(mirror).walk()) == set(BlobStore(hub).walk())
    assert verify_home(mirror)['head_lsn'] == verify_home(hub)['head_lsn']


def test_the_bank_is_founded_and_played_with_the_shipped_verbs_alone(blue):
    """THE BANK IS STOOD UP BY WHAT A DEPLOYMENT ALREADY HAS. Every verb the founding typed is a
    subcommand `DV_Spine` ships, and the heads' declarations stand in the record as scoped ones;
    every binding call a player makes is a tool the MCP server publishes; the transport is read
    for a run's raw result and nothing else; and the hub's writer is reached for a print alone,
    the one fact no verb takes.

    Killing mutation: a player posting a booking over the raw transport, which no host can ask.
    """
    script, _ = blue
    shipped = next(action.choices for action in cli.build_parser()._actions
                   if isinstance(action, argparse._SubParsersAction))
    typed = [act['cli'][0] for act in script['founded']]
    assert set(typed) <= set(shipped) and {'init', 'enroll', 'grant', 'rewrap', 'declare',
                                           'portfolio', 'name'} <= set(typed), typed
    assert [act['seat'] for act in script['founded'] if act['cli'][0] == 'grant'] == [
        roles.FOUNDER] + [roles.seat(desk, 'head') for desk in roles.DESKS]

    tools = set(tool.name for tool in asyncio.run(binding.MCP.list_tools()))
    for module in (roles, red, faults):
        with open(module.__file__, encoding='utf-8') as handle:
            tree = ast.parse(handle.read(), filename=module.__file__)
        called = set(node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
                     and isinstance(node.value, ast.Name) and node.value.id == 'binding')
        assert called - {'configure', 'service'} <= tools, (module.__name__, called - tools)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                shown = [ast.unparse(argument) for argument in node.args[:2]]
                if node.func.attr == 'hub':
                    assert shown[0] == "'GET'" and shown[1].startswith("'/results/"), shown
                if node.func.attr == 'file':
                    assert shown[1] == "'fixing_observed'", shown


def test_every_red_objective_fails_and_is_readable_in_the_record(adversary):
    """SIXTEEN OBJECTIVES, SIXTEEN ANSWERS. Every attempt is denied, refused or simply carried,
    and the record reads the same afterwards as it would have without the adversary in it.

    The ABSENCES are the other half: the stranger and the writer's borrowed name wrote nothing, no
    execution was filled twice, and the denials the record holds are exactly the six the script
    asked for - the node admin's two declarations one denial between them. A refusal that mints
    nothing leaves exactly that - nothing - which is why the script has to say it was asked.

    Killing mutations: an objective whose `answer` is the harness's own `expected` echoed back,
    which would pass a table of words while the record said something else; and the denials fold
    read for presence rather than for its whole contents, which would let a second refusal in
    unnoticed; and an old approval sent again read as a new fact, which is a replay that lands.
    """
    script, reports = adversary
    objectives = dict((row['objective'], row) for row in script['objectives'])

    assert sorted(objectives) == sorted(ANSWERED), sorted(objectives)
    for name, words in ANSWERED.items():
        assert words in str(objectives[name]['answer']), (name, objectives[name]['answer'])
        assert objectives[name]['answer'] != objectives[name]['expected'], name
        assert 'NOT REFUSED' not in str(objectives[name]['answer']), name

    log = SpineLog(script['home'])
    try:
        assert [frame['lsn'] for frame in log.frames()
                if frame['actor'] == roles.STRANGER] == [], 'the stranger reached the platter'
        filled = [log.open_body(frame)['execution_reference'] for frame in log.frames()
                  if frame['event_type'] == 'fill']
    finally:
        log.close()
    assert len(filled) == len(set(filled)), 'one execution reference, two clips'
    assert sorted((row['subject'], row['verb'], row['book']) for row in rows(
        script['home'], 'denials')) == sorted(set(
            (act['denied']['subject'], act['denied']['verb'], act['denied']['book'])
            for act in script['acts'] if act.get('denied'))), rows(script['home'], 'denials')
    assert len(rows(script['home'], 'denials')) == 6
    replayed = said(script, "approve_ticket (the collar's approval, again)")
    assert replayed['recorded'] == replayed['stood'], 'an old approval sent again landed anew'
    for copy in ('hub', 'replica-1'):
        assert oracle.failed(reports[copy]) == [], reports[copy]


def test_every_scripted_fault_converges(adversary):
    """SIX FAULTS, AND THE DAYS CARRY ON. A hub killed mid-append keeps every append it
    acknowledged, is served again on the tail it left, re-derives whole and takes the next frame;
    a partitioned replica asks once and stands where the hub does; a backdated print
    does not win by arriving last and the print it lost to keeps it on the row; a late fixing is a
    SECOND close naming the one it stands over, marked again; and one act said twice - a mark, a
    settlement under its reference - is one fact at one LSN, the money moved once.

    Killing mutations: the killed hub never having acknowledged an append, so "the chain
    re-derives" is a statement about an empty range - which is why the act records where the record
    STOOD before it started and the acknowledgements are held to be past it and still there; the
    restated close COALESCING onto the first, which
    reads as a supersession only until the LSNs are compared; and the settlement repeated under a
    second reference, which moves the money twice.
    """
    script, reports = adversary
    struck = dict((row['fault'], row) for row in script['faults'])

    assert sorted(struck) == sorted(CONVERGED), sorted(struck)
    for name, words in CONVERGED.items():
        assert words in str(struck[name]['answer']), (name, struck[name]['answer'])

    killed = said(script, 'the hub was killed mid-append and served again')
    assert killed['stood'] < killed['acked'][0], 'the killed hub never wrote, so nothing tore'
    assert killed['acked'][-1] <= killed['restarted_at'], 'an acknowledged append was lost'
    assert killed['resumed_at'] == killed['restarted_at'] + 1 == killed['verified']
    assert killed['truncated'], 'the hub served again never truncated the torn tail'
    closes = rows(script['home'], 'markets')['closes']
    assert len(closes) == 1 and closes[0]['lsn'] > closes[0]['supersedes_lsn'], closes
    for act in ('declare_market official (twice)', 'file_status payment (twice)'):
        first, again = said(script, act)['recorded']
        assert first == again, (act, 'the second act minted a second fact')
    assert said(script, 'file_status payment (twice)')['movements'] == 1
    assert oracle.failed(reports['hub']) == [], reports['hub']


def test_the_oracle_names_what_a_copy_cannot_assess(blue, tmp_path):
    """A COPY HOLDING NO KEY still answers four of the fourteen, and says by name which ten it
    cannot.

    The chain, the positions, the tags and the types are the envelope's, so a crypto-shredded copy
    re-derives them over ciphertext; everything else lives inside a body. The oracle says so as a
    SENTENCE and never as a pass - a question nobody could put is not a question that held.

    THE OTHER POSTURE IS THE DAY'S OWN FOLLOWER WITH ITS BLOBS LEFT BEHIND - what `DV_Spine follow`
    without `--blobs` leaves once a key is materialized. It holds every frame and not the documents
    those frames were judged under, the terms the calls were read over nor the jobs the marks
    attested, so the invariants reading them ask for the blobs first and tell that copy to follow
    again - and the ONE it does fail is the true one: it cannot resolve a citation it holds.

    Killing mutations: the unassessable ten answering `held: true`, which is a green report about a
    record nobody read; and the blob check dropped from the judging walk, which accuses the
    deployment's own seats of forging the documents they declared.
    """
    keyless = blue[1]['replica-2']

    blobless = shutil.copytree(blue[0]['replicas'][0], str(tmp_path / 'blobless'))
    for blob in Path(blobless).glob('blobs/*/*/*'):
        blob.unlink()
    pulled = oracle.report(blobless, script=blue[0], against=blue[0]['home'])
    for name in ('nothing_outside_its_seat', 'an_amended_plan_is_a_new_approval',
                 'nothing_outside_its_scope', 'every_close_marked', 'the_call_is_the_formula'):
        assert pulled[name]['held'] is None, pulled[name]
        assert '--blobs' in pulled[name]['evidence'][0], pulled[name]
    assert oracle.failed(pulled) == ['no_delete'], pulled
    assert 'never given one' in pulled['no_delete']['evidence'][0]

    assert oracle.failed(keyless) == [], keyless
    assert sorted(name for name in keyless if keyless[name]['held'] is None) == sorted((
        'an_amended_plan_is_a_new_approval', 'closes_superseded_never_edited',
        'every_numbers_replay_tuple', 'every_refusal_is_a_denial', 'the_diary_equals_the_filings',
        'nothing_outside_its_scope', 'the_pnl_is_additive', 'cash_reconciles',
        'every_close_marked', 'the_call_is_the_formula'))
    for name, found in keyless.items():
        if found['held'] is None:
            assert found['evidence'][0].startswith('not assessed:'), name
    assert keyless['nothing_outside_its_seat']['evidence'][0].endswith('by envelope alone')
    assert keyless['copies_agree']['evidence'][-1] == '1 projector(s) compared: activity', \
        'a copy holding no key opened a body to compare a fold'


def test_the_game_is_played_through_the_binding_and_nowhere_else():
    """WHAT MAKES THIS AN ACCEPTANCE TEST is that the seats reach the record the way a model does.

    Read off the players' own source: every act is a call on the BINDING, on the hub's own
    transport, on the CLI an operator types, or - for the one fact no verb files - on the hub's
    writer through the engine's single seam, which `roles` is the only module here to reach. A
    player that imported the engine itself would be answering something no desk can ask.

    Killing mutation: `derivus` reached from `red` or `faults`, which would put an adversary and a
    fault on a path no host has.
    """
    def imports(source):
        with open(source, encoding='utf-8') as handle:
            tree = ast.parse(handle.read(), filename=source)
        return set(alias.name.split('.')[0] for node in ast.walk(tree)
                   if isinstance(node, ast.Import) for alias in node.names) | set(
            (node.module or '').split('.')[0] for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and not node.level)

    players = dict((module.__name__, imports(module.__file__))
                   for module in (roles, red, faults))
    for name, imported in players.items():
        assert imported.isdisjoint({'torch', 'numpy', 'pandas', 'fastapi', 'uvicorn'}), name
        assert 'derivus_mcp' in imported, '{} reaches no binding at all'.format(name)
    assert [name for name, imported in sorted(players.items()) if 'derivus' in imported] == [
        roles.__name__], 'the engine seam is reached from more than the one place'
