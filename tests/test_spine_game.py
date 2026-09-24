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

"""The acceptance test: a day on the desk, played twice, and held to the oracle's nine.

TWO PLAYS AND NO MORE. The BLUE day is the demo - seven seats working through the binding against a
real hub on an ephemeral port, with two followers pulling its frames over a socket - and the RED day
is that same day with an adversary and five faults in it. Both are module fixtures, because playing
a day costs a Monte Carlo and four structure solves and the assertions below are readings of what it
left behind.

Nothing is monkeypatched. The homes are real and under the gate's own tmp, the service is real and
on a port the box chose, the replicas are real copies pulling real frames, the writer that gets
killed is a real process, and every other fault is data on a platter. What the gates assert is what
the RECORD says, held against what the SCRIPT says was asked.
"""
import os
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from derivus_spine import BlobStore, SpineLog, oracle, verify_home
from derivus_spine.projections import PROJECTORS, fold

from gates.spine_game import faults, play, red, roles

#: objective -> the words the RECORD answered it with. The objective's own `expected` is the
#: sentence the harness wrote down; this is where that sentence has to appear in the answer.
ANSWERED = {
    'approve your own ticket': 'the booker and the approver are one seat',
    'collude under two display names': roles.TRADER,
    'race two acceptances of one quote': '1 fill(s)',
    'replay an old approval against an amended ticket': 'no verdict is filed against this ticket',
    'backdate an amendment to hide a loss': 'unchanged: True',
    'delete the evidence': 'the manifest under',
    'forge a checkpoint': 'the signature does not verify',
    'tamper with a copy': 'the event hash recomputes to',
    "a stranger's what-if": 'holds no validate scope',
}

#: fault -> the words its answer carries. A fault that converged says where it landed.
CONVERGED = {
    'kill the hub between two appends': 'resumed at',
    'partition a replica': 'frame(s) pulled',
    'skew a clock': '1.0 stands over [9.9]',
    'a late fixing after the close': 'supersedes LSN',
    'the same act twice': ', head ',
}


@pytest.fixture(scope='module')
def blue(tmp_path_factory):
    """The default day with red off - the demo, and the shape the oracle's nine are asked of."""
    return play.play(tmp_path_factory.mktemp('blue'), replicas=2)


@pytest.fixture(scope='module')
def adversary(tmp_path_factory):
    """The same day with the nine objectives and the five faults beside it."""
    return play.play(tmp_path_factory.mktemp('red'), with_red=True, with_faults=True, replicas=2)


def rows(home, name):
    """One projector's rows off `home` - what a copy of the record reads."""
    log = SpineLog(home)
    try:
        return PROJECTORS[name].rows(fold(log, PROJECTORS[name]))
    finally:
        log.close()


def test_the_blue_day_holds_every_invariant_on_three_replicas(blue):
    """THE ACCEPTANCE TEST. A desk marks, strikes a settlement file, quotes, accepts, signs, books,
    confirms, attests the close's numbers and declares it; two followers pull every frame; and all
    nine invariants hold on the hub and on the copy that materialized a key.

    The seventh is assessed HERE and nowhere else: the diary is a compile of the book rather than a
    fold of the record, so the keys come out of the hub's own process and the question is put with
    them in hand. The third and the sixth are assessed because a SCRIPT says what was asked.

    Killing mutations: the oracle answering `held` for an invariant it never put (the nulls below
    are named, so a report of nine nulls is red); the day playing without booking anything, so that
    every fold is empty and agrees trivially; and the book file starting with a deal nobody booked,
    which leaves the reconcile reading tripped from act one and saying nothing thereafter.
    """
    script, reports = blue
    for copy in ('hub', 'replica-1'):
        assert sorted(reports[copy]) == sorted(oracle.INVARIANTS), copy
        # stricter than `failed`, which holds nothing against a null: here every one was PUT
        assert [name for name in oracle.INVARIANTS
                if reports[copy][name]['held'] is not True] == [], reports[copy]

    assert [act['act'] for act in script['acts']].count('book_quote (retry)') == 1
    # the day's own control against a SECOND WRITER, which `red` names as the answer to one: the
    # file and the record disagree about nothing, every deal in it having been booked through the hub
    assert next(act for act in script['acts'] if act['act'] == 'book_reconcile')[
        'divergences'] == dict((name, 0) for name in roles.DIVERGENCES)
    assert len(rows(script['home'], 'positions')) >= 1, 'the day booked nothing to agree about'
    assert len(rows(script['home'], 'attestations')) == 1, 'the close cited no attested numbers'
    assert [close['market'] for close in rows(script['home'], 'markets')['closes']] == ['official']

    # the entitled follower pulled the bytes its chain CITES, terms of every position included
    hub, mirror = Path(script['home']), Path(script['replicas'][0])
    held = set(BlobStore(mirror).walk())
    assert held == set(BlobStore(hub).walk()), 'the follower holds frames whose terms it lacks'
    assert verify_home(mirror)['head_lsn'] == verify_home(hub)['head_lsn'], held

    log = SpineLog(hub)
    try:
        # ONE SCOPE FOR AN APPROVAL: the second seat holds `approve` over the desk's own book and
        # signs by hand, which is the grant the tier's automatic signature would have used
        assert [frame['book'] for frame in log.frames()
                if frame['event_type'] == 'approval'] == [roles.DESK_BOOK]
        assert set(frame['actor'] for frame in log.frames()
                   if frame['event_type'] == 'status_transition') == {
            roles.SETTLEMENTS, roles.CONFIRMATIONS}, 'the back office filed under another name'
    finally:
        log.close()


def test_every_red_objective_fails_and_is_readable_in_the_record(adversary):
    """NINE OBJECTIVES, NINE ANSWERS. Every attempt is refused, denied or simply carried, and the
    record reads the same afterwards as it would have without the adversary in it.

    The three ABSENCES are the other half: the stranger wrote nothing, no quote was filled twice,
    and the one denial the record holds is the one the script asked for. A refusal that mints
    nothing leaves exactly that - nothing - which is why the script has to say it was asked.

    Killing mutations: an objective whose `answer` is the harness's own `expected` echoed back,
    which would pass a table of words while the record said something else; and the denials fold
    read for presence rather than for its whole contents, which would let a second refusal in
    unnoticed.
    """
    script, reports = adversary
    objectives = dict((row['objective'], row) for row in script['objectives'])

    assert sorted(objectives) == sorted(ANSWERED), sorted(objectives)
    for name, said in ANSWERED.items():
        assert said in str(objectives[name]['answer']), (name, objectives[name]['answer'])
        assert objectives[name]['answer'] != objectives[name]['expected'], name
    for mask in red.MASKS:
        assert mask not in str(objectives['collude under two display names']['answer'])

    log = SpineLog(script['home'])
    try:
        assert [frame['lsn'] for frame in log.frames()
                if frame['actor'] == roles.STRANGER] == [], 'the stranger reached the platter'
        filled = [log.open_body(frame)['execution_reference'] for frame in log.frames()
                  if frame['event_type'] == 'fill']
    finally:
        log.close()
    assert len(filled) == len(set(filled)), 'one execution reference, two clips'
    assert [(row['subject'], row['verb']) for row in rows(script['home'], 'denials')] == [
        (roles.STRANGER, 'validate')]
    assert oracle.failed(reports['hub']) == [], reports['hub']

    # the adversary's day is where a trade is RESTRUCK, so it is where an amendment's two addresses
    # are asked for: the follower holds the terms on both sides of every one of them
    assert set(BlobStore(Path(script['replicas'][0])).walk()) == set(
        BlobStore(Path(script['home'])).walk()), 'an amendment left its terms on the hub'


def test_every_scripted_fault_converges(adversary):
    """FIVE FAULTS, AND THE DAY CARRIES ON. A killed writer's home re-derives whole and takes the
    next frame; a partitioned replica asks once and stands where the hub does; a backdated print
    does not win by arriving last and the print it lost to keeps it on the row; a late fixing is a
    SECOND close that names the one it stands over; and one act said twice is one fact at one LSN.

    Killing mutations: the killed writer never having written, so "the chain re-derives" is a
    statement about an empty range - which is why the act records where the record STOOD before it
    started and the head is held to be past it, the head alone being true either way; and the
    restated close COALESCING onto the first, which reads as a supersession only until the LSNs are
    compared.
    """
    script, reports = adversary
    struck = dict((row['fault'], row) for row in script['faults'])

    assert sorted(struck) == sorted(CONVERGED), sorted(struck)
    for name, said in CONVERGED.items():
        assert said in str(struck[name]['answer']), (name, struck[name]['answer'])

    killed = next(act for act in script['acts'] if act['act'].startswith('the writer was killed'))
    assert killed['killed_at'] > killed['stood'], 'the killed writer never wrote, so nothing tore'
    assert killed['resumed_at'] == killed['killed_at'] + 1
    closes = rows(script['home'], 'markets')['closes']
    assert len(closes) == 1 and closes[0]['supersedes_lsn'] is not None, closes
    assert closes[0]['lsn'] > closes[0]['supersedes_lsn'], 'the restated close coalesced'
    twice = next(act for act in script['acts'] if act['act'].endswith('(twice)'))
    assert twice['recorded'][0] == twice['recorded'][1], 'the second mark minted a second fact'
    assert oracle.failed(reports['hub']) == [], reports['hub']


def test_the_oracle_names_what_a_copy_cannot_assess(blue, tmp_path):
    """A COPY HOLDING NO KEY still answers four of the nine, and says by name which five it cannot.

    The chain, the positions, the tags and the types are the envelope's, so a crypto-shredded copy
    re-derives them over ciphertext; everything else lives inside a body. The oracle says so as a
    SENTENCE and never as a pass - a question nobody could put is not a question that held.

    THE OTHER POSTURE IS THE DAY'S OWN FOLLOWER WITH ITS BLOBS LEFT BEHIND - what `DV_Spine follow`
    without `--blobs` leaves once a key is materialized. It holds every frame and not the document
    those frames were judged under, and the capabilities fold FAILS CLOSED on a blob that is gone,
    so re-running the writer against it would call every frame after the declaration a forgery. Two
    invariants ask for the blob first and tell that copy to follow again, and the ONE it does fail
    is the true one - it cannot resolve a citation it holds, which is the state a shredded hub is
    in too, and no reading of those bytes tells "never given" from "taken".

    Killing mutations: the unassessable five answering `held: true`, which is a green report about a
    record nobody read; and the blob check dropped from `nothing_outside_its_seat`, which accuses
    the deployment's own seat of forging the document it declared.
    """
    keyless = blue[1]['replica-2']

    blobless = shutil.copytree(blue[0]['replicas'][0], str(tmp_path / 'blobless'))
    for blob in Path(blobless).glob('blobs/*/*/*'):
        blob.unlink()
    pulled = oracle.report(blobless, script=blue[0], against=blue[0]['home'])
    for name in ('nothing_outside_its_seat', 'an_amended_plan_is_a_new_approval'):
        assert pulled[name]['held'] is None, pulled[name]
        assert '--blobs' in pulled[name]['evidence'][0], pulled[name]
    # and the one it DOES fail is the true one: this copy cannot resolve a citation it holds, which
    # is the same state of the same store a shredded hub would be in
    assert oracle.failed(pulled) == ['no_delete'], pulled
    assert 'never given one' in pulled['no_delete']['evidence'][0]

    assert oracle.failed(keyless) == [], keyless
    assert sorted(name for name in keyless if keyless[name]['held'] is None) == sorted((
        'an_amended_plan_is_a_new_approval', 'closes_superseded_never_edited',
        'every_numbers_replay_tuple', 'every_refusal_is_a_denial', 'the_diary_equals_the_filings'))
    for name, found in keyless.items():
        if found['held'] is None:
            assert found['evidence'][0].startswith('not assessed:'), name
    assert keyless['nothing_outside_its_seat']['evidence'][0].endswith('by envelope alone')
    assert keyless['copies_agree']['evidence'][-1] == '1 projector(s) compared: activity', \
        'a copy holding no key opened a body to compare a fold'


def test_the_game_is_played_through_the_binding_and_nowhere_else():
    """WHAT MAKES THIS AN ACCEPTANCE TEST is that the seats reach the record the way a model does.

    Read off the players' own source: every act is a call on the BINDING, on the hub's own
    transport, or - for the one type the vocabulary carries and no verb files - on the hub's writer
    through the engine's single seam, which `roles` is the only module here to reach. A player that
    imported the engine itself would be answering something no desk can ask.

    Killing mutation: `derivus` reached from `red` or `faults`, which would put an adversary and a
    fault on a path no host has.
    """
    import ast

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
