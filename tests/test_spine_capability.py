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

"""Who may write, proved on real homes - and every refusal read back off the log that recorded it.

Two halves. A home nobody has told about capabilities enforces NOTHING: a fresh `init_home` takes a
fill from any actor - the honest reading of a record never told who may do what, and why increment
1's gates are still green beside this file. And a single declaration turns the writer on for every
verb-bearing type at once, with each refusal LANDING AS A FACT: a denial nobody can replay is a
decision that happened outside the record.

The sweep is closed over the vocabulary rather than listed by hand: `VERB_BEARING` is computed from
`EVENT_VERB` and the fixture table is asserted to cover exactly it, so a type added tomorrow with a
verb and no case here turns this file red the day it appears.

Faults are the house's kind - doctored data on disk, never a patched function. A non-canonical
capabilities document is put into the store directly, behind the verb that would have canonicalised
it, which is how a hand-edited policy arrives.

The strand-and-recover walk is the break-glass gate in full, every step an ordinary appended fact: a
document with zero admin grants LANDS (its declarer had admin under the document it replaces), every
later declaration is refused and recorded, a stranger's break-glass use lands and grants nothing,
the genesis seat's use restores admin, a new document lands, and the history verifies green.
"""
import ast
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from derivus_spine import (
    CapabilityDenied, CollisionRefusal, MalformedEvent, MissingBlobRefusal, SpineLog,
    UnknownEventType, canonical_bytes, init_home, verify_home, write_checkpoint)
from derivus_spine import capability, cli, genesis
from derivus_spine.capability import (
    ANY_BOOK, CAPABILITIES_POLICY, UNREADABLE, canonical_document, evaluate, read_subjects,
    state_at, verb_for)
from derivus_spine.vocabulary import (
    ADMIN, APPROVE, BOOK, CUSTODY_TYPES, EVENT_TYPES, EVENT_VERB, FACT_TYPES, MARK,
    PROVENANCE_TYPES, RECOVERY, VERBS, WRITER, WRITER_TYPES, BLOB_FIELDS, classify, validate)

MINT = 'subject-deployment'
DESK = 'subject-desk-one'
MARKER = 'subject-marks'
GOVERNOR = 'subject-governance'
STRANGER = 'subject-nobody'
BOOK_ONE = 'FX-VANILLA'
BOOK_TWO = 'FX-EXOTIC'
HASH_A = 'a' * 64
HASH_B = 'b' * 64
WHEN = '2026-08-29T09:15:00.000000Z'

#: Every type whose append demands one of the six document verbs. Computed, not typed out: the
#: sweep below asserts its own fixture covers exactly this set.
VERB_BEARING = frozenset(name for name, verb in EVENT_VERB.items() if verb in VERBS)


# --------------------------------------------------------------------------------------------
# Documents, homes, and reading the record back.

def document(grants=(), read=()):
    """A capabilities document out of `(subject, verb, book)` and `(subject, class)` rows."""
    return {'grants': [{'subject': s, 'verb': v, 'book': b} for s, v, b in grants],
            'read': [{'subject': s, 'class': c} for s, c in read]}


def declare(log, actor, doc):
    """Declare `doc` the way the writer's own law requires: canonical bytes in the store first,
    then the ordinary `policy_declared` naming the hash."""
    blob = log.store.put(canonical_document(doc))
    return log.append('policy_declared', {'policy': CAPABILITIES_POLICY, 'blob': blob},
                      actor=actor, blob_refs=(blob,))


def minted(tmp_path, name='home'):
    """A home mid-genesis, handed back with the writer open on it."""
    home = tmp_path / name
    init_home(home, MINT)
    return home, SpineLog(home)


def fill(reference, book=BOOK_ONE):
    return {'instrument': HASH_A, 'quantity': 1000000.0, 'netting_set': 'CSA-0007',
            'counterparty': 'LEI-5493001KJTIIGC8Y1R12', 'execution_reference': reference}


def denials(log):
    """Every `capability_denied` fact on the log, as `(lsn, actor, body)`.

    Read off the platter and decrypted like any other event, because that is the assertion: the
    refusal is a chained fact a fold can reach, not a line in a log file.
    """
    return [(frame['lsn'], frame['actor'], log.open_body(frame)) for frame in log.frames()
            if frame['event_type'] == 'capability_denied']


def attempts(log):
    """One well-formed append per verb-bearing type: `(type, body, book)`.

    Blobs are put first for the types whose bodies name them, so the same table drives the denial
    sweep (where closure is never reached) and the granted appends (where it is).
    """
    surface = log.store.put(b'{"surface":"the vol cube of 2026-08-26"}')
    values = log.store.put(b'{"EURUSD":1.0851}')
    retention = log.store.put(b'{"tape":"90 days, then the logged reduction"}')
    seat = log.store.put(b'a seat public key, 32 bytes in the real thing')
    wrap = log.store.put(b'{"algorithm":"x25519-hkdf-sha256-aesgcm-v1"}')
    # increment 3's three, which carry blobs of their own: the job a plan recompiles from, the
    # result, the values vector, and the tolerance policy a pin was held to
    job = log.store.put(b'{"Calc":{"Calculation":{"Object":"BaseValuation"}}}')
    produced = log.store.put(b'{"mtm":1234.5}')
    tolerance = log.store.put(b'{"tolerances":{"mtm":1e-09}}')
    return (
        ('fill', fill('EXEC-1'), BOOK_ONE),
        ('amendment', {'instrument': HASH_A, 'amended_to': HASH_B}, BOOK_ONE),
        ('election', {'instrument': HASH_A, 'choice': 'exercise'}, BOOK_ONE),
        ('status_transition', {'subject': HASH_A, 'status': 'confirmed'}, BOOK_ONE),
        ('snapshot_registered', {'blob': surface}, BOOK_ONE),
        ('approval', {'plan_hash': HASH_B}, BOOK_ONE),
        ('rejection', {'plan_hash': HASH_B, 'reason': 'the booker and the approver are one seat'},
         BOOK_ONE),
        ('determination', {'subject': HASH_A, 'ruling': 'the barrier was touched at 14:02'},
         BOOK_ONE),
        ('market_declared', {'name': 'official', 'values_hash': values}, None),
        ('official_close_declared', {'market': 'official', 'values_hash': values}, None),
        ('fixing_observed', {'index': 'EURUSD-ECB', 'date': '2026-08-26', 'source': 'ECB',
                             'value': 1.0851}, None),
        ('policy_declared', {'policy': 'firmness', 'pillar_age_seconds': 30}, None),
        ('retention_declared', {'blob_class': 'tape', 'policy_blob': retention}, None),
        ('rehash_declared', {'algorithm': 'sha256'}, None),
        ('checkpoint', {'lsn': 1, 'event_hash': HASH_A, 'signature': 'de' * 32}, None),
        ('seat_enrolled', {'subject': DESK, 'algorithm': 'x25519', 'public_key': seat}, None),
        ('key_wrapped', {'class': 'firm', 'subject': DESK, 'wrap': wrap}, None),
        # book: an attestation and a quote are things a desk puts on the record
        ('run_completed', {'plan_hash': HASH_A, 'values_hash': values, 'engine_version': '0.1.0',
                           'seed': 1, 'lane': 'standing', 'job': job, 'result': produced},
         BOOK_ONE),
        ('quote_filed', {'quote_id': 'Q-1', 'structure': 'ZeroCostCollar', 'plan_hash': HASH_A,
                         'values_hash': values, 'solved': {'floor': 17.25}, 'edge': 4200.0},
         BOOK_ONE),
        # approve: a tuple the hub did not witness acquiring standing is a second pair of eyes
        ('result_pinned', {'plan_hash': HASH_A, 'values_hash': values, 'engine_version': '0.1.0',
                           'seed': None, 'job': job, 'result': produced,
                           'tolerance_policy': tolerance}, BOOK_ONE),
    )


# --------------------------------------------------------------------------------------------
# Enforcement is off until it is declared.

def test_a_fresh_home_enforces_nothing_and_writes_exactly_as_increment_one_did(tmp_path):
    """A home never told who may do what has no document to consult, so `evaluate` is never called
    and a stranger's fill lands - in the increment-1 FRAME: twelve fields, the firm class the
    classifier derives, and a chain that verifies entitled. The one case that has to hold for the
    gates beside this file to mean anything after the writer learned to refuse.
    """
    home, log = minted(tmp_path)

    landed = log.append('fill', fill('EXEC-1'), actor=STRANGER, book=BOOK_ONE, effective_time=WHEN)
    assert landed['lsn'] == 5 and landed['coalesced'] is False
    assert landed['entitlement_class'] == 'firm'

    frame = log.frame_at(5)
    assert sorted(frame) == ['actor', 'body', 'book', 'effective_time', 'entitlement_class',
                             'event_hash', 'event_type', 'event_version', 'idempotency_tag',
                             'lsn', 'prev_hash', 'record_time'], 'the frame grew a field'
    assert frame['actor'] == STRANGER and frame['event_version'] == 1

    # Nothing was consulted because there is nothing to consult, and the genesis rows are still read
    # for the break-glass they exist for.
    doc, seats = state_at(log)
    assert doc is None
    assert seats == {'admin': (MINT,), 'break_glass': MINT, 'recovered': ()}
    assert evaluate(doc, seats, STRANGER, ADMIN, None) is True, 'no document, no refusal'
    assert not denials(log), 'a home with no policy refused nothing'

    log.close()
    assert verify_home(home) == {'mode': 'entitled', 'events': 5, 'checkpoints_verified': 1,
                                 'head_lsn': 5, 'head_hash': landed['event_hash']}


def test_the_break_glass_handle_is_gated_from_event_one_and_not_by_declaration(tmp_path):
    """The one authorization that does not wait for a document. The recovery handle is not IN a
    document - a recovery a declaration could revoke is not a recovery - so there is no declaration
    whose absence could mean "anyone may". Left ungated it is an actor-free write channel into the
    book of record, `reason` being free text in a sealed body.

    So the seat genesis named may use it here and now, nobody else may, and the reach is a fact
    either way.
    """
    home, log = minted(tmp_path)
    assert state_at(log)[0] is None, 'this home has never been told who may do what'

    with pytest.raises(CapabilityDenied) as refusal:
        log.append('break_glass_used', {'reason': 'trying it on'}, actor=STRANGER)
    assert MINT in str(refusal.value) and RECOVERY in str(refusal.value)
    assert denials(log)[-1][2] == {'subject': STRANGER, 'verb': RECOVERY, 'book': ANY_BOOK,
                                  'attempted_type': 'break_glass_used'}

    # the seat itself, on the same undeclared home, and a stranger's fill still lands beside it
    assert log.append('break_glass_used', {'reason': 'the hub was rebuilt'},
                      actor=MINT)['coalesced'] is False
    assert log.append('fill', fill('EXEC-1'), actor=STRANGER,
                      book=BOOK_ONE)['coalesced'] is False, 'the six verbs wait for a declaration'

    log.close()
    assert verify_home(home)['events'] == SpineLog(home).head()[0]


def test_the_frame_is_still_the_twelve_fields_increment_one_froze(tmp_path):
    """A sibling of the check above, said as arithmetic: the nine sealed-against envelope fields,
    the body, the chain hash and the position - and no thirteenth. A capability layer that had
    leaked one field into the envelope would be caught here rather than three increments later by a
    verifier that cannot open its own bodies. The class in every frame is the classifier's answer,
    which is the seam consumed rather than a constant stamped in beside it."""
    home, log = minted(tmp_path)
    log.close()
    for frame in SpineLog(home).frames():
        assert len(frame) == 12 and frame['entitlement_class'] == classify(
            frame['event_type'], frame['book'])


# --------------------------------------------------------------------------------------------
# The brief's capability-denial gate: every verb, refused and recorded.

def test_every_verb_bearing_type_refuses_an_unscoped_actor_and_records_the_refusal(tmp_path):
    """Every verb, with the unscoped actor refused and the refusal itself logged. One declaration
    turns the writer on for all of them - authorization is a document and a function, not a check
    bolted onto each verb - and each refusal is asserted twice: the raised `CapabilityDenied` naming
    subject, verb, book scope, attempted type and the denial's LSN, and the `capability_denied` fact
    at that LSN. The document here grants nothing, so the sweep is total.
    """
    home, log = minted(tmp_path)
    table = attempts(log)
    assert set(name for name, _, _ in table) == VERB_BEARING, \
        'the fixture and the verb map have parted company'
    declare(log, MINT, document())

    expected = []
    for event_type, body, book in table:
        verb = verb_for(event_type)
        scope = book if book is not None else ANY_BOOK
        head = log.head()[0]
        with pytest.raises(CapabilityDenied) as refusal:
            log.append(event_type, body, actor=STRANGER, book=book)
        said = str(refusal.value)
        for named in (STRANGER, verb, scope, event_type):
            assert named in said, (event_type, named, said)
        assert 'LSN {}'.format(head + 1) in said, (event_type, said)
        # the refusal moved the head by exactly one event, and that event is the denial
        assert log.head()[0] == head + 1
        expected.append((head + 1, WRITER,
                         {'subject': STRANGER, 'verb': verb, 'book': scope,
                          'attempted_type': event_type}))

    assert denials(log) == expected, 'a refusal went unrecorded or was recorded wrong'
    # the denials are ordinary events - firm class, no book, the writer's own name on them
    for lsn, _, _ in expected:
        frame = log.frame_at(lsn)
        assert frame['book'] is None and frame['entitlement_class'] == 'firm'
        assert frame['effective_time'] is None, 'a denial is true when it was recorded'

    log.close()
    assert verify_home(home)['events'] == expected[-1][0], \
        'a log full of refusals is still a log that verifies'


def test_a_repeated_refusal_is_one_fact_because_it_is_one_fact(tmp_path):
    """Idempotency reaches the denials too. "This subject was refused this verb over this book for
    this type" is one fact however many times it is attempted - the tempo of the attempts is
    serving-layer telemetry under its own retention, which the brief keeps out of the record."""
    home, log = minted(tmp_path)
    declare(log, MINT, document())

    for _ in range(3):
        with pytest.raises(CapabilityDenied):
            log.append('fill', fill('EXEC-1'), actor=STRANGER, book=BOOK_ONE)

    assert len(denials(log)) == 1
    log.close()
    assert verify_home(home)['events'] == log.head()[0]


def test_the_writers_own_denial_is_not_a_type_a_submitter_may_speak(tmp_path):
    """`capability_denied` is the writer's voice. The public append refuses it BY NAME whether or
    not a document is in force, because the alternative is a submitter writing the record's account
    of its own refusal - and it refuses without appending anything, since a denial about an attempt
    that never entered the vocabulary is a fact about nothing."""
    home, log = minted(tmp_path)
    body = {'subject': DESK, 'verb': BOOK, 'book': BOOK_ONE, 'attempted_type': 'fill'}
    head = log.head()

    with pytest.raises(CapabilityDenied) as refusal:
        log.append('capability_denied', body, actor=MINT)
    assert 'capability_denied' in str(refusal.value) and WRITER in str(refusal.value)
    assert log.head() == head, 'the reserved refusal wrote nothing'

    # and still refused once enforcement is on, under an actor holding admin over everything
    declare(log, MINT, document(grants=((MINT, ADMIN, ANY_BOOK),)))
    with pytest.raises(CapabilityDenied):
        log.append('capability_denied', body, actor=MINT)
    assert not denials(log), 'the writer minted a denial about a type nobody may submit'

    log.close()
    assert verify_home(home)['events'] == log.head()[0]


# --------------------------------------------------------------------------------------------
# Scope: the grant is the grant that was given.

def test_a_granted_actor_appends_and_a_book_scope_reaches_exactly_its_book(tmp_path):
    """Book matching, on the three cases that are actually different from each other.

    A named grant reaches its own book and no other. `*` reaches every book AND the firm-level facts
    that carry no book at all. And a named grant does NOT reach a book-less event - policy,
    checkpoints and official market declarations are acts of the deployment, and a desk's scope over
    its own book is not a licence to govern the firm. The checkpoint is the same rule met from the
    other side: `checkpoint` maps to admin, so signing the head under enforcement needs an admin.
    """
    home, log = minted(tmp_path)
    declare(log, MINT, document(
        grants=((DESK, BOOK, BOOK_ONE), (MARKER, MARK, BOOK_ONE), (GOVERNOR, ADMIN, ANY_BOOK),
                (GOVERNOR, MARK, ANY_BOOK))))

    assert log.append('fill', fill('EXEC-1'), actor=DESK, book=BOOK_ONE)['coalesced'] is False
    with pytest.raises(CapabilityDenied) as refusal:
        log.append('fill', fill('EXEC-2', BOOK_TWO), actor=DESK, book=BOOK_TWO)
    assert BOOK_TWO in str(refusal.value)

    # the same verb the desk holds, on a fact that carries no book: only `*` reaches it
    values = log.store.put(b'{"EURUSD":1.0851}')
    with pytest.raises(CapabilityDenied) as refusal:
        log.append('market_declared', {'name': 'official', 'values_hash': values},
                   actor=MARKER, book=None)
    assert ANY_BOOK in str(refusal.value), 'the denial must name the scope that was needed'
    assert log.append('determination', {'subject': HASH_A, 'ruling': 'touched at 14:02'},
                      actor=MARKER, book=BOOK_ONE)['coalesced'] is False
    assert log.append('market_declared', {'name': 'official', 'values_hash': values},
                      actor=GOVERNOR, book=None)['coalesced'] is False

    # admin over `*` signs the head; the desk cannot
    with pytest.raises(CapabilityDenied):
        write_checkpoint(log, actor=DESK)
    assert write_checkpoint(log, actor=GOVERNOR)['event_type'] == 'checkpoint'

    log.close()
    assert verify_home(home)['events'] == log.head()[0]


def test_the_evaluator_is_one_pure_function_over_a_document_and_a_fold():
    """No home, no log, no store, no clock. Everything `evaluate` needs was folded out of the record
    before it was called, which is what lets a replica reach the hub's verdict locally - the brief's
    "a check both sides evaluate locally because both hold the same log and the same policy fold."
    """
    doc = document(grants=((DESK, BOOK, BOOK_ONE), (GOVERNOR, ADMIN, ANY_BOOK)))
    seats = {'admin': (MINT,), 'break_glass': MINT, 'recovered': ()}

    assert evaluate(doc, seats, DESK, BOOK, BOOK_ONE) is True
    assert evaluate(doc, seats, DESK, BOOK, BOOK_TWO) is False
    assert evaluate(doc, seats, DESK, BOOK, None) is False
    assert evaluate(doc, seats, DESK, APPROVE, BOOK_ONE) is False
    assert evaluate(doc, seats, GOVERNOR, ADMIN, None) is True
    assert evaluate(doc, seats, GOVERNOR, ADMIN, BOOK_TWO) is True
    # genesis admin does NOT survive a document: a declaration can strand the last admin, and that
    # is the property break-glass exists to answer rather than a defect to paper over
    assert evaluate(doc, seats, MINT, ADMIN, None) is False
    assert evaluate(None, seats, STRANGER, ADMIN, None) is True, 'no document, no enforcement'
    # the two authorizations outside every document: the writer's own is unconditional (a denial
    # that could be denied is a regress), the recovery is the GENESIS SEAT's and nobody else's -
    # it must survive a document that stranded every admin and still be a gated write path
    assert evaluate(doc, seats, WRITER, WRITER, None) is True
    assert evaluate(doc, seats, MINT, RECOVERY, None) is True
    assert evaluate(doc, seats, STRANGER, RECOVERY, None) is False
    assert evaluate(None, seats, STRANGER, RECOVERY, None) is False, 'the seat, doc or no doc'
    # and a recovered admin outranks the document it was recovered against
    assert evaluate(doc, dict(seats, recovered=(MINT,)), MINT, ADMIN, None) is True

    # a document IN FORCE that cannot be read grants nothing at all, and leaves exactly the two
    # authorizations above to rescue the home with
    assert evaluate(UNREADABLE, seats, DESK, BOOK, BOOK_ONE) is False
    assert evaluate(UNREADABLE, seats, GOVERNOR, ADMIN, None) is False
    assert evaluate(UNREADABLE, seats, MINT, RECOVERY, None) is True
    assert evaluate(UNREADABLE, dict(seats, recovered=(MINT,)), MINT, ADMIN, None) is True
    assert read_subjects(UNREADABLE) == ()

    assert read_subjects(doc) == ()
    assert read_subjects(document(read=((DESK, 'firm'), (STRANGER, 'desk-two')))) == (DESK,)
    assert read_subjects(None) == ()


# --------------------------------------------------------------------------------------------
# As-of: authorization replays like everything else.

def test_authorization_answers_as_of_the_lsn_it_is_asked_about(tmp_path):
    """"Could X book in March" is a fold, not a memory.

    A subject granted at one position and stripped by a replacement at another answers True across
    the span between them and False from the replacement on - read by passing the LSN, exactly as
    the checkpoint-key ladder reads the key in force at a checkpoint's own position. A declaration
    applies to the appends AFTER it, so the position it landed at is the first at which it answers.
    """
    home, log = minted(tmp_path)
    granted = declare(log, MINT, document(
        grants=((DESK, BOOK, BOOK_ONE), (MINT, ADMIN, ANY_BOOK))))['lsn']
    log.append('fill', fill('EXEC-1'), actor=DESK, book=BOOK_ONE)
    stripped = declare(log, MINT, document(grants=((MINT, ADMIN, ANY_BOOK),)))['lsn']

    assert state_at(log, granted - 1)[0] is None, 'before the first declaration there is no policy'
    for position in range(granted, stripped):
        doc, seats = state_at(log, position)
        assert evaluate(doc, seats, DESK, BOOK, BOOK_ONE) is True, position
    for position in (stripped, log.head()[0], None):
        doc, seats = state_at(log, position)
        assert evaluate(doc, seats, DESK, BOOK, BOOK_ONE) is False, position
        assert evaluate(doc, seats, MINT, ADMIN, None) is True, position

    # and the writer's own cached fold is the same fold, freshly reopened off the platter
    log.close()
    assert state_at(SpineLog(home)) == state_at(log)
    assert verify_home(home)['events'] == log.head()[0]


# --------------------------------------------------------------------------------------------
# Strand and recover - the brief's break-glass gate, walked.

def test_a_declaration_can_strand_the_last_admin_and_break_glass_is_the_way_back(tmp_path):
    """The walk, every step an ordinary appended fact. A replacement with zero admin grants LANDS
    (its declarer held admin under the document it replaces). From there every later declaration is
    refused and recorded. A stranger reaching for the break-glass handle is refused and the REACH is
    recorded. The seat genesis named restores admin, a new document lands and clears the recovery,
    and the whole history verifies with the denials inside it.

    Two mutants nothing else here reaches. `capability.apply_event`'s
    `if genesis['break_glass'] is None:` -> `if True:`: the seat is read ONCE from the mint's own
    grant, not from the latest thing calling itself one - without that guard, naming a new seat and
    then stranding every admin is a permanently unwritable home whose recovery somebody else holds.
    The same line for `genesis['admin']`: the LSN-1 grants are read once and scope-filtered.
    """
    home, log = minted(tmp_path)
    declare(log, MINT, document(grants=((MINT, ADMIN, ANY_BOOK), (DESK, BOOK, BOOK_ONE))))

    # Read once, while MINT still holds admin and can therefore declare anything at all. Neither
    # genesis row is a thing a later declaration may re-open, and a home where they were would have
    # no recovery left after the stranding two lines down.
    seats = state_at(log)[1]
    assert seats['admin'] == (MINT,), 'the LSN-1 admin grants, read once and scope-filtered'
    assert seats['break_glass'] == MINT
    log.append('policy_declared',
               {'policy': 'break_glass', 'grant': {'subject': STRANGER}}, actor=MINT)
    log.append('policy_declared',
               {'policy': 'genesis', 'grants': [{'subject': STRANGER, 'scope': ADMIN}]}, actor=MINT)
    seats = state_at(log)[1]
    assert seats['break_glass'] == MINT, \
        'the break-glass seat is the mint\'s, not the latest thing calling itself a grant'
    assert seats['admin'] == (MINT,), 'and neither is the admin roster'

    stranding = declare(log, MINT, document())['lsn']

    doc, seats = state_at(log)
    assert doc == document() and seats['break_glass'] == MINT and seats['recovered'] == ()
    assert evaluate(doc, seats, MINT, ADMIN, None) is False, 'the last admin is stranded'

    for actor in (MINT, DESK, STRANGER):
        with pytest.raises(CapabilityDenied) as refusal:
            declare(log, actor, document(grants=((actor, ADMIN, ANY_BOOK),)))
        assert 'break-glass' in str(refusal.value), refusal.value
    assert [body['attempted_type'] for _, _, body in denials(log)] == ['policy_declared'] * 3

    # A stranger reaches for the handle. The reach is recorded; the fact is not theirs to write.
    with pytest.raises(CapabilityDenied) as refusal:
        log.append('break_glass_used', {'reason': 'trying it on'}, actor=STRANGER)
    assert MINT in str(refusal.value), 'the refusal names the seat that may'
    assert denials(log)[-1][2] == {'subject': STRANGER, 'verb': RECOVERY, 'book': ANY_BOOK,
                                  'attempted_type': 'break_glass_used'}
    assert state_at(log)[1]['recovered'] == (), 'break-glass is the genesis seat, not the type'
    assert not [frame for frame in log.frames() if frame['event_type'] == 'break_glass_used'], \
        'an unscoped actor wrote free text into a sealed body of the record'

    used = log.append('break_glass_used',
                      {'reason': 'the declaration at LSN {} left no admin'.format(stranding)},
                      actor=MINT)
    assert used['lsn'] == log.head()[0], 'the recovery is in the chain'
    doc, seats = state_at(log)
    assert seats['recovered'] == (MINT,)
    assert evaluate(doc, seats, MINT, ADMIN, None) is True
    assert evaluate(doc, seats, STRANGER, ADMIN, None) is False

    restored = declare(log, MINT, document(
        grants=((GOVERNOR, ADMIN, ANY_BOOK), (DESK, BOOK, BOOK_ONE))))
    assert restored['coalesced'] is False
    # the desk is writing again, under a document a recovered admin declared
    assert log.append('fill', fill('EXEC-1'), actor=DESK, book=BOOK_ONE,
                      effective_time=WHEN)['coalesced'] is False

    # and the recovery is SPENT. A grant that lives outside every document and outlives every
    # document is the one admin no policy could revoke, which this module calls a finding; the
    # declaration that ended the emergency is what ends it, and the mint is governed again.
    doc, seats = state_at(log)
    assert seats['recovered'] == (), 'a break-glass admin survived the document that restored order'
    assert evaluate(doc, seats, MINT, ADMIN, None) is False
    assert evaluate(doc, seats, GOVERNOR, ADMIN, None) is True
    head = log.head()
    with pytest.raises(CapabilityDenied):
        declare(log, MINT, document(grants=((MINT, ADMIN, ANY_BOOK),)))
    assert log.head() == head, 'one subject refused one verb for one type is one fact'
    log.close()

    # every step of the walk is a chained fact, denials included
    assert verify_home(home)['events'] == SpineLog(home).head()[0]
    assert [body['subject'] for _, _, body in denials(SpineLog(home))] == \
        [MINT, DESK, STRANGER, STRANGER], \
        'three seats reached for the policy, one more reached for the handle, four are recorded'


# --------------------------------------------------------------------------------------------
# The document itself: what will not land, and why.

def test_a_capabilities_document_that_cannot_be_evaluated_does_not_land(tmp_path):
    """A document the evaluator cannot read authorizes nothing, so it is refused where the operator
    can still fix it rather than discovered by a home that can no longer write.

    The last case is the house's fault injection: canonical bytes put into the store BEHIND the verb
    that would have canonicalised them, which is exactly how a hand-edited policy file arrives. One
    policy must be one blob, or the record holds two histories of one decision.
    """
    home, log = minted(tmp_path)
    head = log.head()

    def refused(raw):
        blob = log.store.put(raw)
        with pytest.raises(CapabilityDenied) as refusal:
            log.append('policy_declared', {'policy': CAPABILITIES_POLICY, 'blob': blob},
                       actor=MINT, blob_refs=(blob,))
        assert log.head() == head, 'a refused declaration wrote a line'
        return str(refusal.value)

    assert 'not JSON' in refused(b'grants: everyone')
    assert 'grants' in refused(canonical_bytes({'read': []}))
    assert 'beyond grants, read' in refused(canonical_bytes(
        dict(document(), revokes=[])))
    assert 'six scopes' in refused(canonical_bytes(document(grants=((DESK, 'launch', ANY_BOOK),))))
    assert 'row' in refused(canonical_bytes({'grants': [{'subject': DESK}], 'read': []}))
    assert 'names nothing' in refused(canonical_bytes(document(grants=((DESK, BOOK, ''),))))
    # valid JSON, valid shape, NON-canonical bytes: one policy, two spellings, refused
    assert 'canonical' in refused(
        json.dumps(document(grants=((DESK, BOOK, BOOK_ONE),)), indent=2).encode('utf-8'))

    # and the body must name a blob at all, then a blob the store already holds
    with pytest.raises(CapabilityDenied) as refusal:
        log.append('policy_declared', {'policy': CAPABILITIES_POLICY, 'blob': 'inline'}, actor=MINT)
    assert '64-hex' in str(refusal.value)
    with pytest.raises(MissingBlobRefusal):
        log.append('policy_declared', {'policy': CAPABILITIES_POLICY, 'blob': HASH_A}, actor=MINT)

    assert log.head() == head
    log.close()
    assert verify_home(home)['events'] == head[0]


def test_a_doctored_policy_blob_folds_to_unreadable_and_break_glass_walks_out_of_it(tmp_path):
    """A policy blob rewritten under its own name no longer hashes to it, so the store refuses on
    the way out and the fold never sees the substituted document.

    What it sees is `UNREADABLE`. The fold runs inside the writer's authorization hook, so a home
    with a doctored blob would otherwise meet that store refusal on EVERY append and be permanently
    unwritable while verifying green - the state the genesis break-glass grant exists to recover
    from. So: every verb refused by name, then the walk out - the genesis seat's use, a replacement
    document, the desk writing again.
    """
    home, log = minted(tmp_path)
    blob = log.store.put(canonical_document(document(
        grants=((MINT, ADMIN, ANY_BOOK), (DESK, BOOK, BOOK_ONE)), read=((DESK, 'firm'),))))
    declare(log, MINT, document(
        grants=((MINT, ADMIN, ANY_BOOK), (DESK, BOOK, BOOK_ONE)), read=((DESK, 'firm'),)))
    log.append('fill', fill('EXEC-1'), actor=DESK, book=BOOK_ONE)
    log.close()

    path = home / 'blobs' / blob[:2] / blob[2:4] / blob
    path.write_bytes(canonical_document(document(grants=((STRANGER, ADMIN, ANY_BOOK),))))

    log = SpineLog(home)
    with pytest.raises(CollisionRefusal) as refusal:
        log.store.get(blob)
    assert blob in str(refusal.value), 'the store handed back bytes that are not what was filed'

    doc, seats = state_at(log)
    assert doc is UNREADABLE, 'the substituted document reached the fold'
    assert evaluate(doc, seats, STRANGER, ADMIN, None) is False, 'the substitution granted nothing'
    assert evaluate(doc, seats, DESK, BOOK, BOOK_ONE) is False, 'and the real one grants nothing now'
    assert read_subjects(doc) == (), 'a class key is not handed out on a document nobody can read'
    assert seats['break_glass'] == MINT and seats['admin'] == (MINT,)

    with pytest.raises(CapabilityDenied) as refusal:
        log.append('fill', fill('EXEC-2'), actor=DESK, book=BOOK_ONE)
    assert 'will not READ' in str(refusal.value) and blob not in str(refusal.value)
    assert denials(log)[-1][2]['subject'] == DESK

    # The way out, and every step of it an ordinary append: the seat genesis named, then a document
    # that reads. Not the same document - the store will not hand back an address it no longer
    # answers for, so recovery declares new bytes rather than re-filing the doctored ones.
    log.append('break_glass_used', {'reason': 'the capabilities blob will not read'}, actor=MINT)
    assert state_at(log)[1]['recovered'] == (MINT,)
    declare(log, MINT, document(grants=((MINT, ADMIN, ANY_BOOK), (DESK, BOOK, BOOK_ONE),
                                        (GOVERNOR, MARK, ANY_BOOK))))
    doc, seats = state_at(log)
    assert doc is not UNREADABLE and evaluate(doc, seats, DESK, BOOK, BOOK_ONE) is True
    assert log.append('fill', fill('EXEC-2'), actor=DESK, book=BOOK_ONE)['coalesced'] is False

    log.close()
    assert verify_home(home)['events'] == SpineLog(home).head()[0]


def test_a_fold_answers_off_the_platter_rather_than_off_the_index_it_opened_with(tmp_path):
    """A handle held across somebody else's append answers today's question. Reading never claims
    the home - deliberately, so a replica is never locked out - which makes an open `SpineLog`
    outliving an append the ORDINARY case. A fold walking an index derived when the handle opened
    would answer authorization out of a replaced policy: a revocation that does not reach the
    custody path, and "could X do Y" wrong AT THE HEAD.

    So the assertion is equality with a handle opened after the fact, on all three of the fold's
    answers - the document, the enumeration custody wraps keys on, and the evaluator's verdict.
    """
    home, log = minted(tmp_path)
    declare(log, MINT, document(grants=((MINT, ADMIN, ANY_BOOK), (DESK, BOOK, BOOK_ONE)),
                                read=((DESK, 'firm'), (MARKER, 'firm'))))
    log.close()

    stale = SpineLog(home)
    doc, seats = state_at(stale)
    assert evaluate(doc, seats, DESK, BOOK, BOOK_ONE) is True
    assert read_subjects(doc) == (DESK, MARKER)

    # somebody else declares a replacement that strips the desk of both, through their own handle
    writer = SpineLog(home)
    stripped = declare(writer, MINT, document(grants=((MINT, ADMIN, ANY_BOOK),),
                                              read=((MARKER, 'firm'),)))['lsn']
    writer.close()

    fresh = SpineLog(home)
    assert state_at(stale) == state_at(fresh), 'the held handle folded a log that had moved'
    doc, seats = state_at(stale)
    assert evaluate(doc, seats, DESK, BOOK, BOOK_ONE) is False, 'a revocation the fold did not see'
    assert read_subjects(doc) == (MARKER,), 'a revoked subject is still entitled to the class key'

    # and the as-of question still answers as-of: the desk COULD book, before the replacement landed
    before, seats_before = state_at(stale, stripped - 1)
    assert evaluate(before, seats_before, DESK, BOOK, BOOK_ONE) is True
    assert state_at(stale, stripped - 1) == state_at(fresh, stripped - 1)


# --------------------------------------------------------------------------------------------
# The vocabulary's own closure.

def test_the_verb_map_is_closed_over_the_closed_vocabulary():
    """Every type has a verb and every verb-map entry has a type. A type without a verb would be a
    write nobody could be scoped for and nobody could be refused for - a hole in enforcement shaped
    exactly like the thing enforcement is for."""
    assert set(EVENT_TYPES) == (set(FACT_TYPES) | set(CUSTODY_TYPES) | set(PROVENANCE_TYPES)
                                | set(WRITER_TYPES))
    assert set(EVENT_VERB) == set(EVENT_TYPES)
    assert set(EVENT_VERB.values()) <= set(VERBS) | {RECOVERY, WRITER}
    assert EVENT_VERB['break_glass_used'] == RECOVERY
    assert EVENT_VERB['capability_denied'] == WRITER
    assert VERB_BEARING == set(EVENT_TYPES) - {'break_glass_used', 'capability_denied'}
    # fails closed: a type nobody taught the map demands governance
    assert verb_for('a_type_from_the_future') == ADMIN


def test_the_classifier_is_the_seam_and_it_answers_firm_in_phase_one():
    """Class is DERIVED, not assigned - one function whose inputs a later declaration changes,
    rather than ten thousand per-object ACLs. Phase 1 is one trading unit, so it answers `firm` for
    everything, and the mechanism ships dormant exactly as the brief's posture says."""
    for event_type in EVENT_TYPES:
        assert classify(event_type, None) == 'firm'
        assert classify(event_type, BOOK_ONE) == 'firm'


def test_the_envelope_asks_the_classifier_rather_than_stamping_a_constant():
    """The seam read off the SOURCE, because in phase 1 no behaviour can tell the two apart: with
    one class, a writer that hardcoded `firm` and one that derived it produce identical frames, and
    the difference surfaces the day desk two arrives. So this asserts the call."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'derivus_spine', 'log.py')
    with open(path, encoding='utf-8') as handle:
        tree = ast.parse(handle.read(), filename='log.py')

    stamped = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if isinstance(key, ast.Constant) and key.value == 'entitlement_class':
                stamped.append(value)

    assert len(stamped) == 1, 'the envelope is built in exactly one place'
    assert isinstance(stamped[0], ast.Call) and getattr(stamped[0].func, 'id', None) == 'classify', \
        'the envelope stamps a constant where it should ask the classifier'
    assert len(stamped[0].args) == 2, 'the classifier is asked about the type AND the book'


def test_the_two_genesis_policy_names_this_fold_reads_are_the_two_genesis_writes():
    """`capability` spells the genesis policy names itself because `genesis` imports the writer and
    the writer imports `capability` - so the two spellings are pinned here instead of by an import
    that would be a cycle."""
    assert capability.GENESIS_POLICY == genesis.GENESIS_POLICY
    assert capability.BREAK_GLASS_POLICY == genesis.BREAK_GLASS_POLICY
    assert capability.CAPABILITIES_POLICY not in (
        genesis.GENESIS_POLICY, genesis.BREAK_GLASS_POLICY, genesis.VERIFYING_KEY_POLICY)


def test_the_three_new_types_are_shaped_and_their_blob_fields_are_closed(tmp_path):
    """The custody vocabulary validates like everything else - every field present and of its kind,
    nothing surplus - and the two fields that name BYTES join `BLOB_FIELDS`, so durability ordering
    binds them: no enrollment appends before the key it publishes is on the platter."""
    validate('seat_enrolled', {'subject': DESK, 'algorithm': 'x25519', 'public_key': HASH_A})
    validate('key_wrapped', {'class': 'firm', 'subject': DESK, 'wrap': HASH_A})
    validate('capability_denied', {'subject': DESK, 'verb': BOOK, 'book': ANY_BOOK,
                                   'attempted_type': 'fill'})
    with pytest.raises(MalformedEvent):
        validate('seat_enrolled', {'subject': DESK, 'algorithm': 'x25519', 'public_key': 'short'})
    with pytest.raises(MalformedEvent):
        validate('key_wrapped', {'class': 'firm', 'subject': DESK, 'wrap': HASH_A, 'note': 'why'})
    with pytest.raises(UnknownEventType):
        validate('key_unwrapped', {})
    assert BLOB_FIELDS['seat_enrolled'] == ('public_key',)
    assert BLOB_FIELDS['key_wrapped'] == ('wrap',)

    home, log = minted(tmp_path)
    with pytest.raises(MissingBlobRefusal) as refusal:
        log.append('seat_enrolled', {'subject': DESK, 'algorithm': 'x25519', 'public_key': HASH_B},
                   actor=MINT)
    assert HASH_B in str(refusal.value)
    log.close()


# --------------------------------------------------------------------------------------------
# The mouth.

NINE = ('init', 'verify', 'checkpoint', 'status', 'enroll', 'grant', 'rewrap', 'name', 'whoami')
APPENDING = ('enroll', 'grant', 'rewrap')


def test_the_cli_seats_the_identity_verbs_beside_the_home_verbs(capsys):
    """Nine verbs, each reachable and each taking the home flag - the four increment-1 verbs plus
    the five identity ones, which are the policy-file editor the non-goals allow and nothing more.
    Every verb that APPENDS takes `--actor`, because an event without an authenticated pseudonymous
    actor is the one thing this workstream will not write."""
    for verb in NINE:
        with pytest.raises(SystemExit) as left:
            cli.main([verb, '--help'])
        assert left.value.code == 0, verb
        said = capsys.readouterr().out
        assert '--home' in said, verb
        assert ('--actor' in said) is (verb in APPENDING or verb == 'init'), verb

    with pytest.raises(SystemExit) as left:
        cli.main(['transmute', '--home', 'x'])
    assert left.value.code == 2, 'an unknown verb is a command-line error, not a refusal'


def test_the_grant_verb_canonicalises_the_operators_file_and_declares_it(tmp_path, capsys):
    """The editor end to end on a real home: a JSON file spelled however the operator spelled it
    becomes ONE canonical blob and one `policy_declared`, and enforcement is live on the next
    append. A second declaration of the same policy from differently-spelled JSON coalesces onto
    the first, which is what "one policy, one blob" buys."""
    home = tmp_path / 'home'
    init_home(home, MINT)
    path = tmp_path / 'policy.json'
    path.write_text(json.dumps(document(grants=((MINT, ADMIN, ANY_BOOK), (DESK, BOOK, BOOK_ONE))),
                               indent=4, sort_keys=False), encoding='utf-8')

    assert cli.main(['grant', '--home', str(home), '--file', str(path), '--actor', MINT]) == 0
    declared = json.loads(capsys.readouterr().out)
    assert declared['policy'] == CAPABILITIES_POLICY and declared['coalesced'] is False

    log = SpineLog(home)
    assert log.store.get(declared['blob']) == canonical_document(
        document(grants=((MINT, ADMIN, ANY_BOOK), (DESK, BOOK, BOOK_ONE))))
    with pytest.raises(CapabilityDenied):
        log.append('fill', fill('EXEC-1'), actor=STRANGER, book=BOOK_ONE)
    assert log.append('fill', fill('EXEC-1'), actor=DESK, book=BOOK_ONE)['coalesced'] is False
    log.close()

    # the same policy, re-spelled: one blob, and the declaration is the same fact
    path.write_text(json.dumps(document(grants=((MINT, ADMIN, ANY_BOOK), (DESK, BOOK, BOOK_ONE))),
                               separators=(',', ':')), encoding='utf-8')
    assert cli.main(['grant', '--home', str(home), '--file', str(path), '--actor', MINT]) == 0
    again = json.loads(capsys.readouterr().out)
    assert again['blob'] == declared['blob'] and again['coalesced'] is True
    assert verify_home(home)['events'] == SpineLog(home).head()[0]


def test_the_grant_verb_refuses_a_file_it_cannot_read_and_says_which(tmp_path, capsys):
    """A refusal reaches the terminal as the library's own sentence and exit 1, naming the PATH -
    which is the thing the operator can go and fix - and writes nothing on the way out."""
    home = tmp_path / 'home'
    init_home(home, MINT)
    head = SpineLog(home).head()

    missing = str(tmp_path / 'nowhere.json')
    assert cli.main(['grant', '--home', str(home), '--file', missing, '--actor', MINT]) == 1
    assert missing in capsys.readouterr().err

    broken = tmp_path / 'broken.json'
    broken.write_text('{"grants": [', encoding='utf-8')
    assert cli.main(['grant', '--home', str(home), '--file', str(broken), '--actor', MINT]) == 1
    assert 'not JSON' in capsys.readouterr().err

    assert SpineLog(home).head() == head, 'a refused grant moved the head'
