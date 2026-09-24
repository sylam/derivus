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

"""The oracle's nine, each made to FAIL, and each named where it did.

A gate that only ever sees a green report is a gate that cannot tell an invariant from a constant,
so every one of these doctors a home until the question it asks has the wrong answer and then reads
the report back. The doctoring is DATA every time: a second home that is not a copy, a frame put on
a platter by hand, a policy declared and a seat that is not in it, a fill booked against a ticket
nobody signed, a close body no validating writer would have taken, a script asking for a run the
record never attested, a settlement against a key the book has no row for, a line removed, and one
fact written twice under one tag.

The frames put on a platter by hand go through `SpineLog.accept`, which is what a replica writes
with: it asks the chain and never the vocabulary, never scope and never the tag, so it is exactly
the door a copy of a newer hub - or somebody with a file handle - comes through. That is the door
the oracle stands behind.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from derivus_spine import ChainBroken, SpineLog, oracle, policy, verify_home
from derivus_spine.verbs import STANDING, approve, book, file_quote
from derivus_spine.vocabulary import cited_blobs

from gates.spine_game.red import forged

from test_spine import ACTOR, BOOK, INSTRUMENT, fill, seeded, segments, synthetic_book
from test_spine_doorbell import STRANGER, granted
from test_spine_imports import spine

#: A plan hash and a ticket the gates below sign, book against and collide on. Sixty-four hex, as
#: every address in the record is.
TICKET = 'a' * 64
PLAN = 'b' * 64
#: The workflow that makes an unsigned fill a failure: with no tiers policy in force the desk has
#: declared no second pair of eyes, and an unsigned booking is what it asked for.
TIERS = {'tiers': [{'name': 'desk', 'four_eyes': True}]}


def copied(home, out, name):
    """A file copy of `home` a forged frame can be accepted onto - what a replica looks like on
    disk, keys included, so the copy can open its own bodies."""
    import shutil

    shutil.copytree(str(home), str(out / name))
    return out / name


def planted(home, out, name, event_type, body, actor=ACTOR, book_name=None, tag=None):
    """A copy of `home` carrying one frame this writer would not have written, and its path."""
    where = copied(home, out, name)
    log = SpineLog(where)
    try:
        log.accept(forged(log, event_type, body, actor, book=book_name, tag=tag))
    finally:
        log.close()
    return where


def quoted(tmp_path, name, ticket=TICKET, approver=None, second=None):
    """A home carrying a workflow, a quote at `ticket`, whatever verdict `approver` filed, and a
    fill booked against that quote. `second` files a second quote at the same ticket.

    The spine's own verbs and no engine anywhere near them - what a booking puts on the record is
    plain data, so the shape invariant four reads is buildable without a pricer.
    """
    home = seeded(tmp_path, name, clips=())
    log = SpineLog(home)
    try:
        policy.declare(log, ACTOR, policy.TIERS_POLICY, TIERS)
        values = b'{"EURUSD":1.0851}'
        file_quote(log, ACTOR, 'Q-1', 'ZeroCostCollar', PLAN, values, {'floor': 1.07}, 4100.0,
                   ticket=ticket, book=BOOK)
        if second is not None:
            file_quote(log, ACTOR, second, 'Accumulator', PLAN, values, {'strike': 1.05}, 900.0,
                       ticket=ticket, book=BOOK)
        if approver is not None:
            approve(log, approver, ticket, book=BOOK)
        book(log, ACTOR, b'{"Reference":"CF1"}', -1.0, 'LEI-549300', 'CSA-0007', 'Q-1', book=BOOK)
    finally:
        log.close()
    return home


def script(*acts):
    """The day's script as the oracle takes it: a document with an `acts` list and nothing else."""
    return {'acts': list(acts)}


def test_the_report_answers_nine_and_names_what_it_could_not_assess(tmp_path):
    """THE GREEN READING, so the reds below are differences rather than the only thing seen. The
    design's synthetic book holds every fact type this vocabulary has, including a close restated
    by a second close, and every invariant the record alone can answer holds over it.

    The four a home with no script and no second copy cannot put are NULLS carrying a sentence, not
    passes: a question nobody asked is not a question that held.

    Killing mutation: `unasked` answering `held: true`, which turns a report about a record nobody
    compared into a clean bill.
    """
    home = synthetic_book(tmp_path)[:2][0]
    answers = oracle.report(home, diary_keys=[INSTRUMENT])

    assert sorted(answers) == sorted(oracle.INVARIANTS)
    assert oracle.failed(answers) == [], answers
    assert sorted(name for name, found in answers.items() if found['held'] is None) == sorted(
        ('copies_agree', 'every_numbers_replay_tuple', 'every_refusal_is_a_denial'))
    assert '2 close(s) over 1 market(s)' in answers['closes_superseded_never_edited']['evidence'][0]
    assert answers['duplicates_coalesce']['evidence'] == ['21 tag(s) over 21 frame(s)']


def test_two_homes_that_are_not_copies_of_one_history_are_named(tmp_path):
    """COPIES AGREE. Two homes minted separately are two histories, whatever they hold: the
    comparison is taken at the SHALLOWER head, so a copy that is a PREFIX of the hub agrees - a
    replica behind its hub has not pulled yet rather than disagreed - and two mints do not.

    Killing mutation: the comparison taken at the deeper of the two heads, which asks the copy for
    a position it does not hold and calls every follower mid-pull a divergence.
    """
    answers = oracle.report(seeded(tmp_path, 'one'), against=seeded(tmp_path, 'two'))

    assert oracle.failed(answers) == ['copies_agree'], answers
    assert 'copies of different histories' in answers['copies_agree']['evidence'][0]

    behind = copied(seeded(tmp_path, 'three'), tmp_path, 'four')
    agreed = oracle.report(tmp_path / 'three', against=behind)
    assert agreed['copies_agree']['held'] is True, agreed['copies_agree']

    log = SpineLog(tmp_path / 'three')
    try:
        log.append('fill', fill('EXEC-AHEAD'), actor=ACTOR, book=BOOK)
    finally:
        log.close()
    ahead = oracle.report(tmp_path / 'three', against=behind)
    assert ahead['copies_agree']['held'] is True, ahead['copies_agree']
    assert 'at LSN {}'.format(SpineLog(behind).head()[0]) in ahead['copies_agree']['evidence'][0]


def test_a_frame_under_a_seat_the_document_never_scoped_is_named(tmp_path):
    """NOTHING OUTSIDE ITS SEAT. A replica's `accept` does not authorize - scope is the hub's
    question, answered where the fact was made - so a frame under a stranger chains perfectly well
    and the oracle re-runs the decision the writer would have made.

    Killing mutation: the capability state folded at the head rather than at the frame's own
    position, which judges every append under whatever document is in force today.
    """
    home = seeded(tmp_path, 'scoped', clips=())
    log = SpineLog(home)
    try:
        # booked BEFORE any document, so it is a legal append that the document in force today
        # would refuse - which is what says the state is read at the frame and not at the head
        log.append('fill', fill('EXEC-EARLY'), actor='subject-desk-two', book=BOOK)
        granted(log, ACTOR)
    finally:
        log.close()
    assert oracle.report(home)['nothing_outside_its_seat']['held'] is True, \
        'a fact filed before the document was judged under it'

    answers = oracle.report(planted(home, tmp_path, 'stranger', 'fill', fill('EXEC-FORGED'),
                                    actor=STRANGER, book_name=BOOK))
    assert oracle.failed(answers) == ['nothing_outside_its_seat'], answers
    said = answers['nothing_outside_its_seat']['evidence'][0]
    assert STRANGER in said and 'no book scope' in said and 'the writer would have refused' in said


def test_a_refusal_the_script_asked_for_and_a_denial_it_did_not_are_both_named(tmp_path):
    """EVERY REFUSAL IS A DENIAL, and every denial a refusal somebody asked for. The two directions
    are one set equality, because a denial nobody asked for is a seat this script never mentions
    reaching this record.

    Killing mutation: the asked denials checked for presence alone, which lets a refusal the day
    never made sit in the record unremarked.
    """
    from derivus_spine import CapabilityDenied

    home = seeded(tmp_path, 'refused', clips=())
    log = SpineLog(home)
    try:
        granted(log, ACTOR)
        with pytest.raises(CapabilityDenied):
            log.append('fill', fill('EXEC-STRANGER'), actor=STRANGER, book=BOOK)
    finally:
        log.close()

    denial = {'subject': STRANGER, 'verb': 'book', 'book': BOOK, 'attempted_type': 'fill'}
    assert oracle.report(home, script=script({'denied': denial}))[
        'every_refusal_is_a_denial']['held'] is True

    surplus = oracle.report(home, script=script())['every_refusal_is_a_denial']
    assert surplus['held'] is False and 'never asked for' in surplus['evidence'][0]
    missing = oracle.report(home, script=script(
        {'denied': dict(denial, verb='mark')}))['every_refusal_is_a_denial']
    assert missing['held'] is False and 'the record holds none' in missing['evidence'][0]


def test_a_fill_booked_at_a_ticket_nobody_else_signed_is_named(tmp_path):
    """AN AMENDED PLAN IS A NEW APPROVAL, read as `quotes` x `decisions`. Three ways it breaks and
    one way it holds: no verdict at the ticket, the booker's own signature on it, two quotes sharing
    one ticket, and a second seat signing.

    Killing mutation: the standing approval read off the FIRST verdict rather than the latest by
    LSN, which lets a rejection filed after an approval book anyway.
    """
    unsigned = oracle.report(quoted(tmp_path, 'unsigned'))
    assert oracle.failed(unsigned) == ['an_amended_plan_is_a_new_approval'], unsigned
    assert 'no verdict on that plan' in unsigned['an_amended_plan_is_a_new_approval']['evidence'][0]

    own = oracle.report(quoted(tmp_path, 'own', approver=ACTOR))
    assert oracle.failed(own) == ['an_amended_plan_is_a_new_approval'], own
    assert 'signed its own ticket' in own['an_amended_plan_is_a_new_approval']['evidence'][0]

    shared = oracle.report(quoted(tmp_path, 'shared', approver='subject-desk-two', second='Q-2'))
    assert 'share the ticket' in shared['an_amended_plan_is_a_new_approval']['evidence'][0]

    signed = oracle.report(quoted(tmp_path, 'signed', approver='subject-desk-two'))
    assert oracle.failed(signed) == [], signed


def test_a_close_no_validating_writer_wrote_is_named(tmp_path):
    """CLOSES SUPERSEDED, NEVER EDITED. Supersession is computed by the fold, so what a platter can
    carry that the fold cannot is a close body the closed vocabulary would have refused - which is
    what a copy of a newer hub, or a hand on a file, puts there.

    Killing mutation: the markets fold taken before the bodies are read, which raises out of the
    projector instead of reporting the line.
    """
    home = synthetic_book(tmp_path)[:2][0]
    where = planted(home, tmp_path, 'edited', 'official_close_declared',
                    {'values_hash': 'not an address'})

    answers = oracle.report(where)
    assert oracle.failed(answers) == ['closes_superseded_never_edited'], answers
    assert answers['closes_superseded_never_edited']['evidence'][0].endswith(
        "no writer that validates wrote this line")


def test_a_standing_run_without_its_attestation_is_named(tmp_path):
    """EVERY NUMBER'S REPLAY TUPLE. The attestations are exactly the standing runs the script asked
    for: a standing ask with no row is a number nothing can replay, and a row no standing ask names
    is a lane that mints nothing having minted.

    Killing mutation: the asks counted rather than matched, which passes any script with as many
    standing runs in it as the record has attestations.
    """
    home, log = synthetic_book(tmp_path)[:2]
    try:
        cited = dict(values_hash=log.store.put(b'{"EURUSD":1.09}'),
                     job=log.store.put(b'{"Calc":{}}'), result=log.store.put(b'{"mtm":1.0}'))
        tuple_ = dict(plan_hash=PLAN, engine_version='2.0', seed=1,
                      values_hash=cited['values_hash'])
        log.append('run_completed', dict(tuple_, lane=STANDING, **cited), actor=ACTOR,
                   blob_refs=tuple(cited.values()))
    finally:
        log.close()

    assert oracle.report(home, script=script(
        {'lane': STANDING, 'replay': tuple_}))['every_numbers_replay_tuple']['held'] is True

    missing = oracle.report(home, script=script(
        {'lane': STANDING, 'replay': dict(tuple_, seed=99)}))['every_numbers_replay_tuple']
    assert missing['held'] is False
    assert 'the record holds no attestation' in missing['evidence'][0]
    assert 'no standing run asked for' in missing['evidence'][1], 'the other direction went unread'

    quiet = oracle.report(home, script=script({'lane': 'curiosity', 'replay': tuple_}))
    assert 'a lane that mints nothing minted' in quiet['every_numbers_replay_tuple']['evidence'][0]


def test_a_settlement_against_a_key_the_diary_has_no_row_for_is_named(tmp_path):
    """THE DIARY EQUALS THE FILINGS - the one invariant a replica cannot put, because the diary is a
    COMPILE of the book and not a fold of the record. The keys arrive as data from a caller that
    holds an engine, and a transition against a key no row carries is a status moving nothing.

    Killing mutation: a transition against a name rather than an address counted as a settlement,
    which reads every lifecycle row as a payment.
    """
    home = synthetic_book(tmp_path)[:2][0]

    assert oracle.report(home)['the_diary_equals_the_filings']['held'] is None
    orphaned = oracle.report(home, diary_keys=[])['the_diary_equals_the_filings']
    assert orphaned['held'] is False and INSTRUMENT in orphaned['evidence'][0]
    assert oracle.report(home, diary_keys=[INSTRUMENT])[
        'the_diary_equals_the_filings']['evidence'] == ['1 settlement(s) filed against 1 diary '
                                                        'key(s)']


def test_a_blob_taken_off_a_copy_is_named_and_a_line_taken_off_it_is_refused(tmp_path):
    """NO DELETE, both ways a record can suffer one. The store has NO VERB for forgetting, so what
    takes a blob is a file system - and the frame that cited it is still on the platter naming an
    address nothing answers for, which is the deletion that actually happens. Every citing frame is
    re-read and every address asked of the store, so a copy a blob was taken off is named by it.

    A LINE that left is met earlier and harder: the sequence is dense from genesis, so the home
    refuses when it is OPENED and there is no report to write. The hub the copies were taken from
    is untouched either way.

    Killing mutations: the closure arm dropped, which leaves every arm of this invariant a constant
    - the blob walk compares the store with itself microseconds apart on a home nobody is writing,
    and the verbs that do not exist never will; and the removed line healed as a torn tail, which
    is the one repair this package has and is reserved for an unterminated FINAL line.
    """
    hub = seeded(tmp_path, 'whole')
    log = SpineLog(hub)
    try:
        taken = next(digest for frame in log.frames() if frame['event_type'] == 'fill'
                     for _, digest in cited_blobs('fill', log.open_body(frame)))
    finally:
        log.close()
    shredded = copied(hub, tmp_path, 'shredded')
    (shredded / 'blobs' / taken[:2] / taken[2:4] / taken).unlink()

    answers = oracle.report(shredded)
    assert oracle.failed(answers) == ['no_delete'], answers
    said = answers['no_delete']['evidence'][0]
    assert taken in said and 'instrument' in said and 'logged retention event' in said
    assert oracle.report(hub)['no_delete']['held'] is True, 'the hub was touched'

    # a copy holding no key cannot open a body to find a citation, so it keeps the other three arms
    (shredded / 'keys' / 'class_firm.key').unlink()
    assert oracle.report(shredded)['no_delete']['evidence'][0].endswith('citations not assessed')

    cut = copied(hub, tmp_path, 'cut')
    segment = segments(cut)[0]
    lines = [raw for raw in segment.read_bytes().split(b'\n') if raw.strip()]
    segment.write_bytes(b'\n'.join(lines[:1] + lines[2:]) + b'\n')
    with pytest.raises(ChainBroken, match='dense'):
        oracle.report(cut)


def test_one_fact_written_twice_under_one_tag_is_named(tmp_path):
    """DUPLICATES COALESCE. The writer folds a repeat onto the position it already has; a replica
    takes the order the hub gave it and coalesces nothing, so a tag on two lines is one act the
    record counted twice.

    Read off the ENVELOPE, so the copy below answers it holding no key at all - which is asserted
    here by deleting the class key and asking again.

    Killing mutation: the tags counted rather than located, which cannot say which two positions
    share one.
    """
    home = seeded(tmp_path, 'once')
    log = SpineLog(home)
    try:
        repeated = next(frame for frame in log.frames()
                        if frame['event_type'] == 'fill')['idempotency_tag']
    finally:
        log.close()

    where = planted(home, tmp_path, 'twice', 'fill', fill('EXEC-1'), book_name=BOOK,
                    tag=repeated)
    answers = oracle.report(where)
    assert oracle.failed(answers) == ['duplicates_coalesce'], answers
    assert 'one fact, written twice' in answers['duplicates_coalesce']['evidence'][0]

    (where / 'keys' / 'class_firm.key').unlink()
    sealed = oracle.report(where)
    assert oracle.failed(sealed) == ['duplicates_coalesce'], sealed
    assert verify_home(where, entitled=False)['mode'] == 'chain-only'


def test_the_cli_answers_the_nine_and_exits_on_one_that_did_not_hold(tmp_path):
    """The mouth: `DV_Spine oracle` prints the report and exits 1 where an invariant did not hold,
    which is what makes it usable from a cron and from a gate that is not this suite.

    A subprocess, which is the only honest way to gate an exit code.
    """
    home = str(synthetic_book(tmp_path)[:2][0])
    asked = spine('oracle', '--home', home)
    assert asked.returncode == 0, asked.stderr
    assert sorted(json.loads(asked.stdout)) == sorted(oracle.INVARIANTS)

    against = str(seeded(tmp_path, 'elsewhere'))
    diverged = spine('oracle', '--home', home, '--against', against)
    assert diverged.returncode == 1, diverged.stdout
    assert json.loads(diverged.stdout)['copies_agree']['held'] is False

    where = tmp_path / 'script.json'
    where.write_text(json.dumps({'acts': [{'denied': {
        'subject': STRANGER, 'verb': 'book', 'book': BOOK, 'attempted_type': 'fill'}}]}),
        newline='\n')
    scripted = spine('oracle', '--home', home, '--script', str(where))
    assert scripted.returncode == 1
    assert 'the record holds none' in json.dumps(
        json.loads(scripted.stdout)['every_refusal_is_a_denial'])


def test_the_oracle_holds_the_package_to_its_one_dependency():
    """The oracle is a module of the record, so it reaches stdlib and `cryptography` and nothing
    else - the import gate's glob already covers it, and this says out loud that it MUST.

    Killing mutation: the oracle importing the engine for the diary invariant, which is exactly the
    temptation the `diary_keys` argument exists to remove.
    """
    import ast

    with open(oracle.__file__, encoding='utf-8') as handle:
        tree = ast.parse(handle.read(), filename=oracle.__file__)
    absolute = set((node.module or '').split('.')[0] for node in ast.walk(tree)
                   if isinstance(node, ast.ImportFrom) and not node.level) | set(
        alias.name.split('.')[0] for node in ast.walk(tree)
        if isinstance(node, ast.Import) for alias in node.names)

    assert absolute == set(), 'the oracle reaches outside the package: {}'.format(absolute)
    assert 'diary_keys' in oracle.report.__code__.co_varnames
