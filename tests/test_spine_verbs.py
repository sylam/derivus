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

"""Increment 3's verb layer, its two policies and its firmness check - the spine half, engine-free.

Nothing here imports `derivus`, which is the seam: the booking verbs, the attestation lanes, the
tolerance comparison and the firmness check are plain functions over plain data, exercised on real
homes in temp directories. `pin_result` takes its executor as an ARGUMENT and the executors below
are ordinary functions in this file, which is what lets the re-execution path run without a GPU.

  * CONSEQUENCE PURITY reaches the verb. The writer refuses `knocked_out` as out of vocabulary but
    would accept a `fill` through the lifecycle arm, so the arm refuses that itself.
  * IDEMPOTENCY through `book`, both ways: one clip retried coalesces, two identical clips differing
    ONLY by execution reference are two facts.
  * MARKET DECLARATION: `official` without `mark` scope is refused and the refusal is a chained fact.
  * THE THREE NEW VERBS each meet an unscoped actor, and each denial lands as a fact.
  * THE LANES mint exactly once: an unknown lane refuses by name, and telemetry and curiosity are
    refused by `complete_run` rather than dropped.
  * PIN_RESULT on all four paths: a cache hit, a bit-identical re-execution, one inside a declared
    tolerance, and the three refusals - wrong version, bytes that will not reproduce, and a claim
    contradicting an attestation this hub made itself.
  * THE TOLERANCE POLICY is the only epsilon in the package, so an unnamed class is refused and a
    home with no policy pins nothing.
  * FIRMNESS: two answers that refuse - the book having moved, the board already stale against the
    one window a desk declares - and one that REPORTS, the market having moved since the quote.
"""
import hashlib
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from derivus_spine import (
    CapabilityDenied, MalformedEvent, MissingBlobRefusal, QuoteNotFirm, ReplayRefused, SpineLog,
    UnknownEventType, init_home, verify_home)
from derivus_spine import firmness, policy, projections, verbs
from derivus_spine.capability import CAPABILITIES_POLICY, canonical_document

MINT = 'subject-deployment'
DESK = 'subject-desk-one'
STRANGER = 'subject-nobody'
BOOK = 'FX-VANILLA'
CLIENT = 'CSA-0007'
COUNTERPARTY = 'LEI-5493001KJTIIGC8Y1R12'
WHEN = '2026-08-29T09:15:00.000000Z'

#: A tiny instrument and a tiny values vector - bytes, because that is what the record addresses.
INSTRUMENT = b'{"Currency":"USD","Object":"FXOptionDeal","Strike_Price":18.5}'
AMENDED = b'{"Currency":"USD","Object":"FXOptionDeal","Strike_Price":18.75}'
VALUES = b'{"FxRate.ZAR":{"Spot":18.5}}'
MOVED_VALUES = b'{"FxRate.ZAR":{"Spot":18.6}}'
JOB = b'{"Calc":{"Calculation":{"Object":"BaseValuation"}}}'
RESULT = b'{"mtm":1234.5}'

#: The same result one ulp out, and one that is a different answer altogether.
NEAR_RESULT = b'{"mtm":1234.500000001}'
FAR_RESULT = b'{"mtm":1240.0}'

TOLERANCE = {'tolerances': {'mtm': 1e-6}}


def address(data):
    """What the store will file `data` under - the gate computing the address itself rather than
    asking the store what it thinks."""
    return hashlib.sha256(data).hexdigest()


def minted(tmp_path, name='home'):
    """A home mid-genesis, handed back with the writer open on it."""
    home = tmp_path / name
    init_home(home, MINT)
    return home, SpineLog(home)


def document(grants=(), read=()):
    return {'grants': [{'subject': s, 'verb': v, 'book': b} for s, v, b in grants],
            'read': [{'subject': s, 'class': c} for s, c in read]}


def declare(log, actor, doc):
    """Declare a capabilities document the way the writer's own law requires."""
    blob = log.store.put(canonical_document(doc))
    return log.append('policy_declared', {'policy': CAPABILITIES_POLICY, 'blob': blob},
                      actor=actor, blob_refs=(blob,))


def denials(log):
    """Every `capability_denied` fact on the log, read off the platter and decrypted like any
    other event - which is the assertion: a refusal is a chained fact, not a line in a log file."""
    return [(frame['lsn'], log.open_body(frame)) for frame in log.frames()
            if frame['event_type'] == 'capability_denied']


def claim(values=VALUES, engine_version='0.1.0', seed=1, plan_hash=None):
    """A replay tuple over the values vector a gate is about to hand in. The values hash IS the
    address of those bytes, which is the composition the verb checks rather than believes."""
    return {'plan_hash': plan_hash or address(JOB + b'plan'), 'values_hash': address(values),
            'engine_version': engine_version, 'seed': seed}


def executor_of(result, version='0.1.0', seen=None):
    """An executor that answers `result` - a real function, which is what the injection seam takes.

    `seen` collects the calls, so a gate can assert that the CACHE HIT path never ran anything: an
    attestation the hub already holds must not cost a second execution, and the only honest way to
    say so is to count.
    """
    def execute(job, values, engine_version):
        if seen is not None:
            seen.append((job, values, engine_version))
        return version, result
    return execute


def with_tolerance(log, actor=MINT, tolerances=TOLERANCE):
    """Declare the tolerance policy a pin is held to, and answer its blob."""
    return policy.declare(log, actor, policy.TOLERANCE_POLICY, tolerances)['blob']


# --------------------------------------------------------------------------------------------
# Consequence purity, through the verb rather than only through the vocabulary.

def test_the_lifecycle_verb_refuses_anything_consequence_shaped_with_the_closure_stated(tmp_path):
    """The facts law met on the verb's own arm. A knock is refused by the WRITER (`knocked_out` is
    not in the vocabulary); a `fill` through the lifecycle arm is not - it is a perfectly good type,
    and a lifecycle verb that quietly booked a trade is a hole the closed vocabulary cannot see. So
    the arm names the three facts it files, states the closure, and its refusal names the VERB.
    """
    home, log = minted(tmp_path)
    head = log.head()

    for shaped in ('knocked_out', 'expired', 'accrued', 'barrier_touched'):
        with pytest.raises(UnknownEventType) as refusal:
            verbs.apply_lifecycle(log, MINT, shaped, {'instrument': 'a' * 64, 'choice': 'x'},
                                  book=BOOK)
        said = str(refusal.value)
        assert 'apply_lifecycle' in said and shaped in said, said
        assert 'CONSEQUENCE' in said and 'projection' in said, said
        assert 'election, fixing_observed, determination' in said, said

    # the sharper half: a type the writer knows perfectly well, refused because this is not the
    # verb that files it
    with pytest.raises(UnknownEventType) as refusal:
        verbs.apply_lifecycle(log, MINT, 'fill', {
            'instrument': 'a' * 64, 'quantity': 1.0, 'counterparty': COUNTERPARTY,
            'netting_set': CLIENT, 'execution_reference': 'EXEC-1'}, book=BOOK)
    assert 'fill' in str(refusal.value)
    assert log.head() == head, 'a refused lifecycle wrote something'

    # and the three that ARE lifecycle facts land
    landed = [
        verbs.apply_lifecycle(log, MINT, 'election', {'instrument': 'a' * 64, 'choice': 'exercise'},
                              book=BOOK, effective_time=WHEN),
        verbs.apply_lifecycle(log, MINT, 'fixing_observed', {
            'index': 'EURUSD-ECB', 'date': '2026-08-26', 'source': 'ECB', 'value': 1.0851}),
        verbs.apply_lifecycle(log, MINT, 'determination', {
            'subject': 'a' * 64, 'ruling': 'the barrier was touched at 14:02'}, book=BOOK)]
    assert [one['lsn'] for one in landed] == [5, 6, 7]
    log.close()
    assert verify_home(home)['events'] == 7


def test_a_settlement_is_filed_under_its_own_key_and_read_back_under_it(tmp_path):
    """THE BACK OFFICE'S VERB. A state a party MOVED - a payment made, a confirmation matched - is
    a fact rather than a consequence, so `transition` files it where `apply_lifecycle` refuses it
    and names this verb instead.

    The subject is an ADDRESS, which the vocabulary is wider than on purpose: a state filed against
    something nobody can resolve is a state nobody can read back, so the verb is the narrower one.
    What the record does NOT ask is whether a book announces a row under that key - it holds what
    it was told and a fold says what answers for it - so a key nothing carries lands.

    Killing mutations: the subject left to the validator's TEXT, which files a settlement against a
    deal's reference; and `transition` reached through `apply_lifecycle`, whose refusal now names
    it.
    """
    home, log = minted(tmp_path)
    key, orphan = address(b'CF1/fixed/payment/2026-06-28'), address(b'a row nobody carries')

    landed = verbs.transition(log, DESK, key, 'settled', book=BOOK, effective_time=WHEN)
    assert landed['lsn'] == 5 and landed['coalesced'] is False
    assert verbs.transition(log, DESK, key, 'settled', book=BOOK,
                            effective_time=WHEN)['coalesced'] is True, 'saying it twice is two'
    assert verbs.transition(log, DESK, orphan, 'settled', book=BOOK)['lsn'] == 6

    for said, status in ((key, ''), (key, None), ('CF1', 'settled'), ('a' * 63, 'settled')):
        with pytest.raises(MalformedEvent):
            verbs.transition(log, DESK, said, status, book=BOOK)
    with pytest.raises(UnknownEventType) as sent:
        verbs.apply_lifecycle(log, DESK, 'status_transition', {'subject': key, 'status': 'settled'},
                              book=BOOK)
    assert '`transition` files it' in str(sent.value), str(sent.value)

    standing = projections.fold(log, projections.PROJECTORS['lifecycle'])['transitions']
    assert sorted(standing) == sorted((key, orphan))
    assert standing[key]['status'] == 'settled' and standing[key]['lsn'] == 5

    declare(log, MINT, document(grants=((DESK, 'mark', BOOK),)))
    with pytest.raises(CapabilityDenied) as refusal:
        verbs.transition(log, DESK, key, 'unsettled', book=BOOK)
    assert 'book' in str(refusal.value) and DESK in str(refusal.value)
    assert denials(log)[-1][1] == {'subject': DESK, 'verb': 'book', 'book': BOOK,
                                   'attempted_type': 'status_transition'}
    log.close()


# --------------------------------------------------------------------------------------------
# Booking: idempotency in both directions, and what a fill will not do without.

def test_a_retried_booking_coalesces_and_two_identical_clips_both_land(tmp_path):
    """Idempotency through `book`, both directions at once. ONE CLIP RETRIED is one fact: the same
    semantic tuple meets its own tag and coalesces. TWO IDENTICAL CLIPS are two facts, differing
    only in the execution reference the venue gave them - which is why that reference is required
    rather than defaulted.
    """
    home, log = minted(tmp_path)

    first = verbs.book(log, DESK, INSTRUMENT, 1_000_000.0, COUNTERPARTY, CLIENT, 'EXEC-1',
                       book=BOOK, effective_time=WHEN)
    assert first['lsn'] == 5 and first['coalesced'] is False
    assert first['instrument'] == address(INSTRUMENT)

    retried = verbs.book(log, DESK, INSTRUMENT, 1_000_000.0, COUNTERPARTY, CLIENT, 'EXEC-1',
                         book=BOOK, effective_time=WHEN)
    assert retried['coalesced'] is True and retried['lsn'] == first['lsn']
    assert log.head()[0] == 5, 'a retry moved the head'

    second = verbs.book(log, DESK, INSTRUMENT, 1_000_000.0, COUNTERPARTY, CLIENT, 'EXEC-2',
                        book=BOOK, effective_time=WHEN)
    assert second['coalesced'] is False and second['lsn'] == 6
    # one instrument, two events - booking the same strike twice finds the same row
    assert log.open_body(log.frame_at(5))['instrument'] == \
        log.open_body(log.frame_at(6))['instrument']

    log.close()
    assert verify_home(home) == {'mode': 'entitled', 'events': 6, 'checkpoints_verified': 1,
                                'head_lsn': 6, 'head_hash': log.head()[1]}


def test_a_fill_refuses_by_name_without_the_three_things_a_fill_carries(tmp_path):
    """A signed quantity, a counterparty, a netting set and an execution reference - each missing
    one refuses BY NAME rather than defaulting, because every default here would be the record
    inventing a term of the trade."""
    home, log = minted(tmp_path)
    good = dict(quantity=1.0, counterparty=COUNTERPARTY, netting_set=CLIENT,
                execution_reference='EXEC-1')

    for field, broken in (('quantity', None), ('quantity', 'a lot'), ('counterparty', ''),
                          ('netting_set', ''), ('execution_reference', ''),
                          ('execution_reference', None)):
        with pytest.raises(MalformedEvent) as refusal:
            verbs.book(log, DESK, INSTRUMENT, book=BOOK, **dict(good, **{field: broken}))
        assert field in str(refusal.value), (field, str(refusal.value))
    assert log.head()[0] == 4, 'a refused booking wrote something'
    log.close()


def test_an_amendment_is_a_new_instrument_hash_and_never_a_changed_one(tmp_path):
    """Economics are never edited. The amendment registers both spellings and links them, and terms
    that did not move are refused - an amendment of nothing is an operational fact wearing a
    booking's clothes, and the refusal says where it belongs instead."""
    home, log = minted(tmp_path)
    verbs.book(log, DESK, INSTRUMENT, 1.0, COUNTERPARTY, CLIENT, 'EXEC-1', book=BOOK)

    amended = verbs.amend(log, DESK, INSTRUMENT, AMENDED, book=BOOK, effective_time=WHEN)
    assert amended['instrument'] == address(INSTRUMENT)
    assert amended['amended_to'] == address(AMENDED)
    assert log.store.has(address(AMENDED)), 'the amended terms were not registered'

    with pytest.raises(MalformedEvent) as refusal:
        verbs.amend(log, DESK, INSTRUMENT, INSTRUMENT, book=BOOK)
    assert 'status_transition' in str(refusal.value)
    log.close()
    assert verify_home(home)['events'] == 6


# --------------------------------------------------------------------------------------------
# Market declaration, and the scope that moves a name.

def test_declaring_official_without_mark_scope_is_refused_and_the_refusal_is_logged(tmp_path):
    """Officialness is a property of the NAME rather than of the data, and the name moves only by a
    declaration from a `mark`-scoped actor. The refusal is the writer's and lands as a
    `capability_denied` fact, so "who tried to move the official close in March" is a fold. A
    private scratch market is the same call under a different name and needs the same scope.
    """
    home, log = minted(tmp_path)
    declare(log, MINT, document(grants=((DESK, 'mark', BOOK), (MINT, 'mark', '*'),
                                        (MINT, 'admin', '*'))))
    head = log.head()[0]

    with pytest.raises(CapabilityDenied) as refusal:
        verbs.declare_market(log, STRANGER, 'official', VALUES)
    said = str(refusal.value)
    assert STRANGER in said and 'mark' in said and 'market_declared' in said, said
    assert denials(log)[-1] == (head + 1, {'subject': STRANGER, 'verb': 'mark', 'book': '*',
                                           'attempted_type': 'market_declared'})

    # a book-scoped mark does not reach a FIRM-level act either: the official close is the
    # deployment's, and a desk's scope over its own book is not a licence to govern it
    with pytest.raises(CapabilityDenied):
        verbs.declare_market(log, DESK, 'official', VALUES)

    marked = verbs.declare_market(log, MINT, 'official', VALUES)
    assert marked['values_hash'] == address(VALUES)
    assert log.open_body(log.frame_at(marked['lsn'])) == {
        'name': 'official', 'values_hash': address(VALUES)}
    log.close()
    assert verify_home(home)['events'] == log.head()[0]


# --------------------------------------------------------------------------------------------
# Capability denial, per verb this increment added.

def test_every_verb_refuses_an_unscoped_actor_and_records_the_refusal(tmp_path):
    """The capability-denial gate on the VERBS - a different claim from the type-level sweep in
    `test_spine_capability.py`, since a verb could reach the writer under an actor the verb chose
    rather than the one the caller named. One document granting nothing turns every arm off at once,
    which is the design: authorization is a document and a pure function. The verb each type demands
    is spelled here rather than read off the vocabulary, so this file disagrees with that table if
    the table moves.
    """
    home, log = minted(tmp_path)
    tolerance = with_tolerance(log)
    declare(log, MINT, document())
    ran = []

    attempts = (
        ('fill', 'book', lambda: verbs.book(log, STRANGER, INSTRUMENT, 1.0, COUNTERPARTY, CLIENT,
                                            'E-1', book=BOOK)),
        ('amendment', 'book', lambda: verbs.amend(log, STRANGER, INSTRUMENT, AMENDED, book=BOOK)),
        ('election', 'book', lambda: verbs.apply_lifecycle(
            log, STRANGER, 'election', {'instrument': 'a' * 64, 'choice': 'exercise'}, book=BOOK)),
        ('market_declared', 'mark',
         lambda: verbs.declare_market(log, STRANGER, 'private/' + STRANGER, VALUES)),
        ('official_close_declared', 'mark',
         lambda: verbs.declare_close(log, STRANGER, 'official', VALUES)),
        ('approval', 'approve', lambda: verbs.approve(log, STRANGER, 'a' * 64, book=BOOK)),
        ('rejection', 'approve', lambda: verbs.reject(log, STRANGER, 'a' * 64, 'the terms moved',
                                                      book=BOOK)),
        ('quote_filed', 'book', lambda: verbs.file_quote(
            log, STRANGER, 'Q-1', 'ZeroCostCollar', 'a' * 64, VALUES, {'floor': 17.25}, 4200.0,
            book=BOOK)),
        ('run_completed', 'book', lambda: verbs.complete_run(
            log, STRANGER, verbs.STANDING, claim(), JOB, VALUES, RESULT, book=BOOK)),
        ('result_pinned', 'approve', lambda: verbs.pin_result(
            log, STRANGER, claim(), JOB, VALUES, RESULT, executor_of(RESULT, seen=ran),
            book=BOOK)),
    )

    expected = []
    for event_type, verb, attempt in attempts:
        head = log.head()[0]
        with pytest.raises(CapabilityDenied) as refusal:
            attempt()
        said = str(refusal.value)
        for named in (STRANGER, verb, event_type):
            assert named in said, (event_type, named, said)
        assert log.head()[0] == head + 1, event_type
        # the two firm-level acts carry no book, so the denial's scope is the wildcard
        scope = '*' if event_type in ('market_declared', 'official_close_declared') else BOOK
        expected.append((head + 1, {'subject': STRANGER, 'verb': verb, 'book': scope,
                                    'attempted_type': event_type}))

    assert denials(log) == expected, 'a refusal went unrecorded or was recorded wrong'
    assert tolerance and len(ran) == 1, \
        'the pin re-executed before the writer refused it - a declared boundary, gated so a change ' \
        'to it is a change to this line'
    log.close()
    assert verify_home(home)['events'] == log.head()[0]


def test_the_writer_records_a_refusal_it_did_not_itself_make_and_a_repeat_is_one_fact(tmp_path):
    """`SpineLog.refuse` is the denial verb in public, for the caller that turns a seat away BEFORE
    the append - queue admission, which refuses WORK and so never reaches `_authorize` at all.

    The row is the one the hook lands, in the writer's own name and under the reserved type no
    submitter may speak, so a denial minted by the queue and a denial minted by the writer read the
    same off the log; a repeated refusal coalesces onto the LSN it already has, the tuple carrying
    no clock of the writer's own.

    Killing mutations: the caller appending `capability_denied` itself, which the writer refuses by
    name because the type is its own voice; and `refuse` stamping the row with a clock, after which
    a stranger turned away twice a second is a fact per attempt.
    """
    home, log = minted(tmp_path)
    head = log.head()[0]

    denied = log.refuse(STRANGER, 'validate', BOOK, 'curiosity')
    assert denied['lsn'] == head + 1 and denied['actor'] == 'writer'
    assert log.open_body(log.frame_at(denied['lsn'])) == {
        'subject': STRANGER, 'verb': 'validate', 'book': BOOK, 'attempted_type': 'curiosity'}

    again = log.refuse(STRANGER, 'validate', BOOK, 'curiosity')
    assert (again['lsn'], again['coalesced']) == (denied['lsn'], True)
    assert log.head()[0] == denied['lsn'], 'a repeated refusal is a second fact'
    # a firm-level job asks for the wildcard, which is the scope the body records
    assert log.open_body(log.frame_at(log.refuse(STRANGER, 'book', None, 'run_completed')[
        'lsn']))['book'] == '*'

    with pytest.raises(CapabilityDenied) as forged:
        log.append('capability_denied', {'subject': STRANGER, 'verb': 'validate', 'book': BOOK,
                                         'attempted_type': 'curiosity'}, actor=DESK)
    assert 'writer\'s own voice' in str(forged.value)
    log.close()
    assert verify_home(home)['events'] == log.head()[0]


def test_a_private_market_is_declared_by_the_seat_its_name_names(tmp_path):
    """SELF-DECLARED MEANS SELF-DECLARED. A `private/<subject>/<name>` market is one seat's own
    board, so the subject in the name is the seat that declares it - otherwise any `mark`-scoped
    actor could mint a market inside another seat's namespace and own it, while the seat the name
    names was refused their own prefix, and ownership would read one way off the name and another
    off the fold.

    EVERY VERB THAT MOVES WHAT A NAME STANDS ON ASKS IT, the close included: a close is one way of
    declaring a market, the `markets` fold resolves a name across both, and one landing inside
    another subject's namespace would be a board its owner never declared and is the only reader
    of - exactly the second answer this rule exists to prevent.

    Killing mutations: the check dropped from `declare_market`, after which `subject-deployment`
    owns `private/subject-desk-one/screen` and the desk that name points at cannot read it; and the
    check absent from `declare_close`, which is the same fact wearing a close's clothes.
    """
    home, log = minted(tmp_path)
    taken = ('private/{}/screen'.format(STRANGER), 'private/screen', 'private/')

    mine = verbs.declare_market(log, DESK, 'private/{}/screen'.format(DESK), VALUES)
    assert log.open_body(log.frame_at(mine['lsn']))['name'] == 'private/{}/screen'.format(DESK)
    closed = verbs.declare_close(log, DESK, 'private/{}/screen'.format(DESK), VALUES)
    assert log.open_body(log.frame_at(closed['lsn']))['market'] == 'private/{}/screen'.format(DESK)

    for verb in ('declare_market', 'declare_close'):
        for named in taken:
            with pytest.raises(MalformedEvent) as refusal:
                getattr(verbs, verb)(log, DESK, named, VALUES)
            said = str(refusal.value)
            assert verb in said and DESK in said, (verb, named)
            assert 'one seat\'s own board' in said, (verb, named)
    assert log.head()[0] == closed['lsn'], 'a refused declaration wrote something'

    # the rule reaches only the private prefix: a firm name is whatever the firm declares it
    assert verbs.declare_market(log, DESK, 'dealer', VALUES)['lsn'] == closed['lsn'] + 1
    log.close()
    assert verify_home(home)['events'] == log.head()[0]


# --------------------------------------------------------------------------------------------
# The decisions, and the close.

def test_an_approval_retried_by_one_seat_is_one_fact_and_a_rejection_carries_its_grounds(tmp_path):
    """A decision is a fact over a plan HASH, so an amended plan is a new hash needing a new
    signature. One seat signing one plan twice is ONE fact - the semantic tuple carries no clock of
    the writer's own, so the retry meets its own tag - while a second seat signing the same plan is
    a second fact, because who signed is part of what was said.

    A rejection's reason is required and has no default: a verdict is never withdrawn, so grounds
    nobody can read are grounds nothing can be filed against later.
    """
    home, log = minted(tmp_path)
    plan, other = 'a' * 64, 'b' * 64

    signed = verbs.approve(log, DESK, plan, book=BOOK)
    assert signed['coalesced'] is False
    assert log.open_body(log.frame_at(signed['lsn'])) == {'plan_hash': plan}

    again = verbs.approve(log, DESK, plan, book=BOOK)
    assert again['coalesced'] is True and again['lsn'] == signed['lsn']
    assert verbs.approve(log, MINT, plan, book=BOOK)['lsn'] == signed['lsn'] + 1

    amended = verbs.approve(log, DESK, other, book=BOOK)
    assert amended['lsn'] > signed['lsn'], 'an amended plan reached the signature of the old one'

    refused = verbs.reject(log, MINT, plan, 'the booker and the approver are one seat', book=BOOK)
    assert log.open_body(log.frame_at(refused['lsn'])) == {
        'plan_hash': plan, 'reason': 'the booker and the approver are one seat'}

    for broken in (('not-a-hash', 'why'), (plan, ''), (plan, None)):
        with pytest.raises(MalformedEvent) as refusal:
            verbs.reject(log, MINT, broken[0], broken[1], book=BOOK)
        assert ('plan_hash' if broken[0] != plan else 'reason') in str(refusal.value)
    with pytest.raises(MalformedEvent):
        verbs.approve(log, DESK, 'not-a-hash', book=BOOK)

    log.close()
    assert verify_home(home)['events'] == log.head()[0]


def test_a_second_close_supersedes_the_first_and_a_read_before_it_is_unmoved(tmp_path):
    """A close is declared by a verb and superseded by a SECOND close rather than corrected in
    place, so the row names the LSN it stands over and the day as it was marked first is still
    readable at the position it was marked at.

    Killing mutation: the second close overwriting the values hash of the first, after which a fold
    taken as at the first close answers the restated vector and no read of the record can say the
    day was marked twice.
    """
    home, log = minted(tmp_path)

    first = verbs.declare_close(log, MINT, 'official', VALUES)
    assert first['values_hash'] == address(VALUES)
    assert log.open_body(log.frame_at(first['lsn'])) == {
        'market': 'official', 'values_hash': address(VALUES)}

    second = verbs.declare_close(log, MINT, 'official', MOVED_VALUES)
    assert second['values_hash'] == address(MOVED_VALUES) and second['lsn'] > first['lsn']

    markets = projections.PROJECTORS['markets']
    standing = markets.rows(projections.fold(log, markets))['closes'][0]
    assert (standing['values_hash'], standing['supersedes_lsn']) == (address(MOVED_VALUES),
                                                                    first['lsn'])
    was = markets.rows(projections.fold(log, markets, lsn=first['lsn']))['closes'][0]
    assert (was['values_hash'], was['supersedes_lsn']) == (address(VALUES), None)

    with pytest.raises(MalformedEvent) as refusal:
        verbs.declare_close(log, MINT, '', VALUES)
    assert 'declare_close' in str(refusal.value) and 'market' in str(refusal.value)
    log.close()
    assert verify_home(home)['events'] == log.head()[0]


# --------------------------------------------------------------------------------------------
# The lanes: a run is recorded IFF its output will be cited by a fact.

def test_the_lanes_are_three_and_exactly_one_of_them_mints(tmp_path):
    """The rule in one sentence, gated as three answers to it.

    An unknown lane is refused BY NAME rather than read as a default, because a lane is a decision
    written down and a caller who has not made it must not have one made for them. Telemetry and
    curiosity mint nothing, and `complete_run` refuses them out loud rather than dropping them -
    silence would leave a caller believing the record holds something it does not.
    """
    home, log = minted(tmp_path)
    assert verbs.LANES == ('telemetry', 'curiosity', 'standing')

    for unknown in ('standing ', 'STANDING', 'exploration', None, ''):
        with pytest.raises(MalformedEvent) as refusal:
            verbs.check_lane(unknown)
        assert 'telemetry, curiosity, standing' in str(refusal.value)

    assert verbs.mints(verbs.STANDING) is True
    assert verbs.mints(verbs.TELEMETRY) is False and verbs.mints(verbs.CURIOSITY) is False

    for silent in (verbs.TELEMETRY, verbs.CURIOSITY):
        with pytest.raises(MalformedEvent) as refusal:
            verbs.complete_run(log, MINT, silent, claim(), JOB, VALUES, RESULT, book=BOOK)
        assert silent in str(refusal.value) and 'cited by a fact' in str(refusal.value)
    assert log.head()[0] == 4, 'a lane that mints nothing minted something'

    attested = verbs.complete_run(log, MINT, verbs.STANDING, claim(), JOB, VALUES, RESULT,
                                  book=BOOK)
    body = log.open_body(log.frame_at(attested['lsn']))
    assert body['lane'] == verbs.STANDING
    assert body == dict(claim(), lane=verbs.STANDING, job=address(JOB), result=address(RESULT))
    # every object the claim is checkable from is on the platter, and the closure says so
    for blob in (address(JOB), address(RESULT), address(VALUES)):
        assert log.store.has(blob)
    log.close()
    assert verify_home(home)['events'] == 5


def test_an_attestation_checks_the_values_vector_it_is_handed_rather_than_believing_it(tmp_path):
    """The engine's values hash IS the SHA-256 of that vector's canonical bytes, so the citation
    and the store address are one number. A caller handing in a vector that is not the one the run
    read is refused by name, because attesting it would put a tuple in the record pointing at bytes
    that never produced it - and the record never trusts what it can re-derive."""
    home, log = minted(tmp_path)

    with pytest.raises(MalformedEvent) as refusal:
        verbs.complete_run(log, MINT, verbs.STANDING, claim(VALUES), JOB, MOVED_VALUES, RESULT,
                           book=BOOK)
    said = str(refusal.value)
    assert address(VALUES) in said and address(MOVED_VALUES) in said, said
    assert log.head()[0] == 4
    log.close()


# --------------------------------------------------------------------------------------------
# Promotion: re-execute, cache-hit, or refuse by name.

def test_a_result_pinned_that_fails_re_execution_is_refused_by_name(tmp_path):
    """A `result_pinned` that fails re-execution within tolerance is refused by name. Three ways to
    fail, each its own refusal because the remedies differ: bytes that are a different answer name
    the departure and the policy; a version that is not the recorded one names both, since a replay
    claim is a claim AT a version; and a result class the policy does not name refuses too, because
    the spine admits no tolerance of its own.

    Nothing is appended on any of them.
    """
    home, log = minted(tmp_path)
    with_tolerance(log)
    head = log.head()[0]

    with pytest.raises(ReplayRefused) as refusal:
        verbs.pin_result(log, MINT, claim(), JOB, VALUES, RESULT, executor_of(FAR_RESULT),
                         book=BOOK)
    said = str(refusal.value)
    assert 'mtm' in said and '1234.5' in said and '1240' in said, said
    assert 'tolerance' in said and 'Nothing is pinned' in said, said

    with pytest.raises(ReplayRefused) as refusal:
        verbs.pin_result(log, MINT, claim(engine_version='0.1.0'), JOB, VALUES, RESULT,
                         executor_of(RESULT, version='0.2.0'), book=BOOK)
    assert "'0.1.0'" in str(refusal.value) and "'0.2.0'" in str(refusal.value)

    # a class the policy does not name: the epsilon lives only where a deployment declared one
    other = b'{"cva":12.0}'
    with pytest.raises(ReplayRefused) as refusal:
        verbs.pin_result(log, MINT, claim(), JOB, VALUES, other,
                         executor_of(b'{"cva":12.5}'), book=BOOK)
    assert 'cva' in str(refusal.value) and 'no epsilon' in str(refusal.value)

    assert log.head()[0] == head, 'a refused pin wrote something'
    log.close()
    assert verify_home(home)['events'] == head


def test_a_result_pinned_matching_a_known_tuple_resolves_as_a_cache_hit(tmp_path):
    """A `result_pinned` matching a known tuple resolves as a cache hit against the original
    attestation. This hub witnessed the run and said so as `run_completed`, so NOTHING IS EXECUTED -
    asserted by counting the executor's calls rather than trusting the report. The pin still lands
    (a promotion is a fact about a claim being accepted) and still names the tolerance policy in
    force.

    A claim naming a DIFFERENT result under the same four coordinates is refused: one replay tuple
    cannot have two results.
    """
    home, log = minted(tmp_path)
    tolerance = with_tolerance(log)
    attested = verbs.complete_run(log, MINT, verbs.STANDING, claim(), JOB, VALUES, RESULT,
                                  book=BOOK)
    ran = []

    pinned = verbs.pin_result(log, MINT, claim(), JOB, VALUES, RESULT,
                              executor_of(RESULT, seen=ran), book=BOOK)
    assert pinned['resolution'] == 'cache hit'
    assert ran == [], 'a cache hit re-executed the run it already had'
    assert pinned['tolerance_policy'] == tolerance
    assert log.open_body(log.frame_at(pinned['lsn'])) == dict(
        claim(), job=address(JOB), result=address(RESULT), tolerance_policy=tolerance)

    # the attestation it hit is the one the hub wrote, at the LSN it wrote it
    found = verbs.attestation(log, claim())
    assert found['lsn'] == attested['lsn'] and found['result'] == address(RESULT)

    # and a second identical pin is one fact, by the ordinary tag rule
    again = verbs.pin_result(log, MINT, claim(), JOB, VALUES, RESULT, executor_of(RESULT),
                             book=BOOK)
    assert again['coalesced'] is True and again['lsn'] == pinned['lsn']

    # one tuple, two results: a claim about another history, refused rather than promoted
    with pytest.raises(ReplayRefused) as refusal:
        verbs.pin_result(log, MINT, claim(), JOB, VALUES, FAR_RESULT, executor_of(FAR_RESULT),
                         book=BOOK)
    said = str(refusal.value)
    assert 'already attested' in said and str(attested['lsn']) in said, said

    log.close()
    assert verify_home(home)['events'] == log.head()[0]


def test_a_pin_re_executes_bit_identically_or_inside_the_declared_epsilon(tmp_path):
    """The two accepting paths, kept apart. BIT-EQUALITY IS THE FAST PATH - no JSON parsed, no
    epsilon consulted. The tolerance path exists because the engine's float boundary is where drift
    lives. A tuple this hub never attested is not a cache hit, so both of these DO execute, which is
    the difference between this gate and the one above.
    """
    home, log = minted(tmp_path)
    with_tolerance(log)
    ran = []

    exact = verbs.pin_result(log, MINT, claim(), JOB, VALUES, RESULT,
                             executor_of(RESULT, seen=ran), book=BOOK)
    assert exact['resolution'] == 'reproduced' and len(ran) == 1
    # the executor was handed the job, the values and the version the claim names
    assert ran[0] == (JOB, VALUES, '0.1.0')

    close = verbs.pin_result(log, MINT, claim(seed=2), JOB, VALUES, RESULT,
                             executor_of(NEAR_RESULT, seen=ran), book=BOOK)
    assert close['resolution'] == 'reproduced' and len(ran) == 2
    assert log.open_body(log.frame_at(close['lsn']))['seed'] == 2

    log.close()
    assert verify_home(home)['events'] == log.head()[0]


def test_a_home_that_declared_no_tolerance_policy_pins_nothing_at_all(tmp_path):
    """The epsilon lives only where a deployment declared one, so a home that has never said what
    "reproduces" means cannot attest anybody's claim - not even one it could re-execute perfectly,
    and not even one it already holds as an attestation. The refusal names the document to declare
    and the verb that declares it."""
    home, log = minted(tmp_path)
    verbs.complete_run(log, MINT, verbs.STANDING, claim(), JOB, VALUES, RESULT, book=BOOK)
    head = log.head()[0]

    with pytest.raises(ReplayRefused) as refusal:
        verbs.pin_result(log, MINT, claim(), JOB, VALUES, RESULT, executor_of(RESULT), book=BOOK)
    said = str(refusal.value)
    assert 'tolerance' in said and 'policy.declare' in said, said
    assert log.head()[0] == head, 'a pin with no standard behind it wrote something'

    with_tolerance(log)
    assert verbs.pin_result(log, MINT, claim(), JOB, VALUES, RESULT, executor_of(RESULT),
                            book=BOOK)['resolution'] == 'cache hit'
    log.close()


def test_a_pin_never_bootstraps_itself_off_another_pin(tmp_path):
    """A cache hit means the hub WITNESSED the run, so the fold reads `run_completed` and never
    `result_pinned`. Reading pins would let one unverified promotion become the evidence for the
    next, which is a claim bootstrapping itself into a fact - so a second pin of a tuple this hub
    never ran re-executes, every time, and the counter says so."""
    home, log = minted(tmp_path)
    with_tolerance(log)
    ran = []

    verbs.pin_result(log, MINT, claim(), JOB, VALUES, RESULT, executor_of(RESULT, seen=ran),
                     book=BOOK)
    assert verbs.attestation(log, claim()) is None, 'a pin was read as an attestation'
    # the same tuple, a different pinner: still not a cache hit
    verbs.pin_result(log, DESK, claim(), JOB, VALUES, RESULT, executor_of(RESULT, seen=ran),
                     book=BOOK)
    assert len(ran) == 2
    log.close()


# --------------------------------------------------------------------------------------------
# The tolerance policy: the one epsilon in the package, and it is declared.

def test_the_tolerance_comparison_is_structural_first_and_numeric_second():
    """Two results whose SHAPES differ are two answers rather than two readings of one number, and
    no epsilon admits that: a class one side does not carry, a row count that moved, a label that
    changed. Only numbers inside a named class get the epsilon."""
    tolerances = {'mtm': 1e-6, 'cashflows': 0.5}

    assert policy.compare({'mtm': 1.0}, {'mtm': 1.0 + 1e-9}, tolerances) == []
    assert policy.compare({'mtm': 1.0}, {'mtm': 1.0}, {}) == [], 'equality needs no tolerance'

    moved = policy.compare({'mtm': 1.0}, {'mtm': 2.0}, tolerances)
    assert len(moved) == 1 and 'mtm' in moved[0] and '1e-06' in moved[0]

    assert 'produced no' in policy.compare({'mtm': 1.0}, {}, tolerances)[0]
    assert 'does not carry' in policy.compare({}, {'mtm': 1.0}, tolerances)[0]

    rows = policy.compare({'cashflows': {'ZAR': [1.0, 2.0]}},
                          {'cashflows': {'ZAR': [1.0, 2.0, 3.0]}}, tolerances)
    assert len(rows) == 1 and 'row count' in rows[0]

    labels = policy.compare({'cashflows': {'ZAR': ['pay', 1.0]}},
                            {'cashflows': {'ZAR': ['receive', 1.0]}}, tolerances)
    assert len(labels) == 1 and 'no epsilon that makes them one answer' in labels[0]

    keys = policy.compare({'cashflows': {'ZAR': 1.0}}, {'cashflows': {'USD': 1.0}}, tolerances)
    assert len(keys) == 2 and all('present on one side only' in said for said in keys)

    # nested numbers ride their CLASS's epsilon, whatever depth they sit at
    assert policy.compare({'cashflows': {'ZAR': [[1.0, 2.0]]}},
                          {'cashflows': {'ZAR': [[1.2, 2.0]]}}, tolerances) == []
    deep = policy.compare({'cashflows': {'ZAR': [[1.0]]}}, {'cashflows': {'ZAR': [[1.9]]}},
                          tolerances)
    assert len(deep) == 1 and 'cashflows.ZAR[0][0]' in deep[0]


def test_a_policy_document_is_closed_at_the_field_level_and_refuses_where_it_is_declared(tmp_path):
    """A policy the record cannot read is a standard nobody can be held to, so both documents are
    checked at the moment they are DECLARED - while the operator still has the file open - rather
    than at the moment a verb tries to compare against one."""
    home, log = minted(tmp_path)

    for broken in ({'tolerances': {'mtm': -1e-9}}, {'tolerances': {'mtm': 'tight'}},
                   {'tolerances': []}, {'tolerances': {}, 'note': 'why'}, ['tolerances']):
        with pytest.raises(MalformedEvent):
            policy.declare(log, MINT, policy.TOLERANCE_POLICY, broken)

    for broken in ({'pillar_seconds': -1}, {'pillar_seconds': 'ten minutes'},
                   {'pillar_seconds': 30, 'surprise': 1}):
        with pytest.raises(MalformedEvent):
            policy.declare(log, MINT, policy.FIRMNESS_POLICY, broken)

    with pytest.raises(MalformedEvent) as refusal:
        policy.declare(log, MINT, 'liquidity', {'anything': 1})
    assert 'firmness, fixings, tiers, tolerance' in str(refusal.value)

    assert log.head()[0] == 4, 'a refused declaration wrote something'
    # an empty tolerance document is a policy that tolerates nothing, and it is legal - silence is
    # never mistaken for absence
    assert policy.declare(log, MINT, policy.TOLERANCE_POLICY, {'tolerances': {}})['lsn'] == 5
    log.close()


def test_the_policy_in_force_is_a_fold_and_absence_is_the_answer(tmp_path):
    """The last declaration at or before a position is the one that answers, which makes "what
    standard was this claim held to in March" a fold like every other question. A home that
    declared no FIRMNESS policy refuses nothing on staleness, because it has not said how stale is
    too stale - while a home that declared no TOLERANCE policy pins nothing, because attesting
    somebody else's numbers is a claim and quoting off your own board is not.

    The fold answers WHERE as well as what, off the same walk, so a reader showing an operator what
    is in force and where it came from cannot show one declaration's blob at another's position.
    """
    home, log = minted(tmp_path)
    assert policy.in_force(log, policy.TOLERANCE_POLICY) == (None, None, None)
    assert policy.firmness_in_force(log) == {}

    first = policy.declare(log, MINT, policy.FIRMNESS_POLICY, {'pillar_seconds': 5})
    assert policy.firmness_in_force(log) == {'pillar_seconds': 5.0}
    second = policy.declare(log, MINT, policy.FIRMNESS_POLICY, {'pillar_seconds': 90})
    assert policy.firmness_in_force(log) == {'pillar_seconds': 90.0}
    # as of the earlier position the earlier document is what governed
    assert policy.firmness_in_force(log, second['lsn'] - 1)['pillar_seconds'] == 5.0
    assert first['lsn'] < second['lsn']

    # ONE WALK, ONE ANSWER: the position comes off the frame this fold chose, so an open-bodied
    # declaration under the same name - legal, and what `canonical_policy`'s own refusal points at
    # - is stepped over rather than lending its LSN to somebody else's blob
    log.append('policy_declared', {'policy': policy.FIRMNESS_POLICY, 'note': 'by hand'},
               actor=MINT)
    assert policy.in_force(log, policy.FIRMNESS_POLICY)[2] == second['lsn']
    assert policy.in_force(log, policy.FIRMNESS_POLICY)[0] == second['blob']
    assert policy.firmness_in_force(log)['pillar_seconds'] == 90.0

    # a declared document whose blob stopped answering for it refuses out loud rather than
    # folding to a sentinel: this fold is read by a VERB, so nothing is bricked by refusing
    blob = policy.in_force(log, policy.FIRMNESS_POLICY)[0]
    os.remove(str(home / 'blobs' / blob[:2] / blob[2:4] / blob))
    with pytest.raises(MalformedEvent) as refusal:
        policy.firmness_in_force(log)
    assert blob in str(refusal.value)
    log.close()


# --------------------------------------------------------------------------------------------
# Firmness: two answers that refuse, one that reports.

PINNED = {'plan_hash': 'a' * 64, 'values_hash': 'b' * 64}
WINDOW = {'pillar_seconds': 900.0}


def test_a_quote_may_be_booked_when_the_plan_stands_and_the_board_was_fresh():
    """The accepting case, and it reports everything it read - both hashes, the age and the window -
    so a desk shown a verdict never has to re-derive the comparison to believe it."""
    verdict = firmness.assess(PINNED, dict(PINNED), 1.0, WINDOW)

    assert verdict['firm'] is True and verdict['refusals'] == []
    assert verdict['plan'] == {'pinned': 'a' * 64, 'current': 'a' * 64, 'moved': False}
    assert verdict['pillar'] == {'age': 1.0, 'window': 900.0, 'firm': True}
    assert verdict['market'] == {'pinned': 'b' * 64, 'current': 'b' * 64, 'moved': False}
    assert firmness.check(PINNED, dict(PINNED), 1.0, WINDOW, quote_id='Q-1')['firm'] is True


def test_a_moved_market_is_reported_and_never_refused():
    """THE RULING: between a quote and the client's word the board may move materially, and the
    booking follows the spine as usual. The move is NEWS - the values struck on, the ones standing
    and that they differ - and the desk's own `firm_seconds` is the promise that bounds it."""
    ticked = dict(PINNED, values_hash='d' * 64)
    verdict = firmness.assess(PINNED, ticked, 1.0, WINDOW)

    assert verdict['firm'] is True and verdict['refusals'] == []
    assert verdict['market'] == {'pinned': 'b' * 64, 'current': 'd' * 64, 'moved': True}
    assert firmness.check(PINNED, ticked, 1.0, WINDOW, quote_id='Q-2')['market']['moved'] is True


def test_a_moved_book_refuses_on_the_plan_and_says_so():
    """The book moved since the solve: the marginal charge was priced against a portfolio this
    trade would no longer join. The board is untouched, which is the whole point of answering them
    apart - a desk told "stale" learns nothing, a desk told "the book moved" re-solves."""
    moved = dict(PINNED, plan_hash='c' * 64)
    verdict = firmness.assess(PINNED, moved, 1.0, WINDOW)

    assert verdict['firm'] is False and verdict['pillar']['firm'] is True
    assert verdict['plan']['moved'] is True and verdict['market']['moved'] is False
    said = verdict['refusals'][0]
    assert 'the book moved under it' in said and 'a' * 64 in said and 'c' * 64 in said

    with pytest.raises(QuoteNotFirm) as refusal:
        firmness.check(PINNED, moved, 1.0, WINDOW, quote_id='Q-7')
    assert 'Q-7' in str(refusal.value) and 'the book moved under it' in str(refusal.value)
    assert 'oldest stamped pillar' not in str(refusal.value), 'the two answers were conflated'


def test_a_board_already_stale_when_the_price_was_given_refuses_by_name():
    """The pillar window is about the board the quote was STRUCK on: a price given off a surface
    nobody had refreshed for an hour is not one a client may hold the desk to, and the remedy is
    the tick rather than a re-quote against a market that never moved."""
    verdict = firmness.assess(PINNED, dict(PINNED), 3600.0, WINDOW)

    assert verdict['firm'] is False and verdict['plan']['moved'] is False
    assert verdict['pillar'] == {'age': 3600.0, 'window': 900.0, 'firm': False}
    said = verdict['refusals'][0]
    assert '3600.0s old at its oldest stamped pillar' in said and '900s window' in said

    # and a home that declared NO window refuses nothing, reporting the age it measured
    silent = firmness.assess(PINNED, dict(PINNED), 3600.0, {})
    assert silent['firm'] is True and silent['pillar'] == {'age': 3600.0, 'window': None,
                                                           'firm': True}


def test_both_refusals_fire_together_and_the_refusal_names_both():
    """A quote with two things wrong with it has two remedies, so BOTH are named - reporting the
    first would send a salesperson back to re-quote into the second."""
    with pytest.raises(QuoteNotFirm) as refusal:
        firmness.check(PINNED, {'plan_hash': 'c' * 64, 'values_hash': 'd' * 64}, 1e6, WINDOW,
                       quote_id='Q-9')
    said = str(refusal.value)
    assert 'the book moved under it' in said and 'oldest stamped pillar' in said
    assert ' AND ' in said


def test_an_age_that_cannot_be_established_is_not_an_age_inside_the_window():
    """A book no row of which is stamped, met here in the general case. A clock that ran backwards
    reads as unknown too, because a board stamped after the quote that read it is the one reading
    that would let an arbitrarily stale one through. With no window declared, none of it refuses."""
    for age in (None, -5.0, 'a while'):
        verdict = firmness.assess(PINNED, dict(PINNED), age, WINDOW)
        assert verdict['firm'] is False and verdict['plan']['moved'] is False
        assert 'carries no stamped pillar' in verdict['refusals'][0]
        assert verdict['pillar']['age'] is None
        assert firmness.assess(PINNED, dict(PINNED), age, {})['firm'] is True


def test_the_firmness_check_refuses_what_it_cannot_compare_rather_than_answering_anyway():
    """A missing pin is a quote that never pinned rather than a quote that went stale, and a window
    that will not read is one no booking could be measured against. Both refuse by name, so a
    verdict of "not firm" always means what it says."""
    with pytest.raises(MalformedEvent) as refusal:
        firmness.assess({'plan_hash': 'a' * 64}, dict(PINNED), 1.0, WINDOW)
    assert 'values_hash' in str(refusal.value) and 'never pinned' in str(refusal.value)

    with pytest.raises(MalformedEvent):
        firmness.assess(PINNED, dict(PINNED), 1.0, {'pillar_seconds': 'fifteen minutes'})
    with pytest.raises(MalformedEvent):
        firmness.assess(PINNED, dict(PINNED), 1.0, 900.0)
    with pytest.raises(MalformedEvent):
        firmness.assess('a' * 64, dict(PINNED), 1.0, WINDOW)


def test_the_two_retired_windows_are_refused_at_the_declaration_by_name(tmp_path):
    """A desk carrying the old document must be TOLD where its question went rather than declaring
    a window nothing enforces: `values_seconds` asked whether the board had moved, which is now
    reported, and `plan_seconds` aged a book the plan hash already catches."""
    home, log = minted(tmp_path)

    for retired in ({'values_seconds': 30}, {'plan_seconds': 600},
                    {'pillar_seconds': 900, 'plan_seconds': 600}):
        with pytest.raises(MalformedEvent) as refusal:
            policy.declare(log, MINT, policy.FIRMNESS_POLICY, retired)
        assert 'pillar_seconds' in str(refusal.value)
        assert sorted(retired)[0] in str(refusal.value) or 'plan_seconds' in str(refusal.value)

    assert policy.declare(log, MINT, policy.FIRMNESS_POLICY, {})['lsn'] == 5
    assert policy.firmness_in_force(log) == {}, 'an empty document declared a window'
    policy.declare(log, MINT, policy.FIRMNESS_POLICY, {'pillar_seconds': 900})
    assert policy.firmness_in_force(log) == {'pillar_seconds': 900.0}
    log.close()


# --------------------------------------------------------------------------------------------
# The quote, filed.

def test_a_quote_pins_two_hashes_and_carries_its_erasable_request(tmp_path):
    """A filed quote is the ATOMICITY gate's own subject: exactly one values hash and exactly one
    book plan, in one body, so a priced ticket can never reference two of either.

    The relayed client request is optional and lives INSIDE the sealed body, which is the whole of
    its erasure mechanism - destroy the class key and what was asked is unreadable forever while
    the chain over it still verifies. The gate asserts both halves: the field is there when a
    salesperson relayed something and absent when nobody did, and the crypto-shredded home's chain
    still stands with the utterance gone.
    """
    home, log = minted(tmp_path)

    filed = verbs.file_quote(log, DESK, 'Q-1', 'ZeroCostCollar', 'a' * 64, VALUES,
                             {'floor': 17.25}, 4200.0,
                             request='the client wants three months of downside at zero cost',
                             book=BOOK, effective_time=WHEN)
    body = log.open_body(log.frame_at(filed['lsn']))
    assert body['plan_hash'] == 'a' * 64 and body['values_hash'] == address(VALUES)
    assert body['solved'] == {'floor': 17.25} and body['edge'] == 4200.0
    assert body['request'].startswith('the client wants')
    assert sorted(body) == ['edge', 'plan_hash', 'quote_id', 'request', 'solved', 'structure',
                            'values_hash'], 'a quote body grew a field'

    quiet = verbs.file_quote(log, DESK, 'Q-2', 'Straddle', 'a' * 64, VALUES, {}, 0.0, book=BOOK)
    assert 'request' not in log.open_body(log.frame_at(quiet['lsn']))
    assert 'ticket' not in log.open_body(log.frame_at(quiet['lsn']))

    log.close()
    assert verify_home(home)['events'] == 6

    # the shred: the class key goes, the utterance goes with it, and the chain still verifies
    os.remove(str(home / 'keys' / 'class_firm.key'))
    assert verify_home(home, entitled=False)['events'] == 6
    shredded = SpineLog(home)
    with pytest.raises(Exception):
        shredded.open_body(shredded.frame_at(filed['lsn']))
    shredded.close()


def test_a_quote_carries_the_ticket_its_approval_would_sign_where_one_was_computed(tmp_path):
    """The optional third hash. `plan_hash` is the BOOK's plan, which is the right question for
    firmness - has the book this would land against moved - and the wrong one for identity, since
    two quotes struck against an unmoved book pin the same one. `ticket` is the plan the book WOULD
    have with this quote's mirror spliced in, so an approval over it reaches this quote and no
    other, and an amended mirror is a new hash by construction.

    Optional because a v1 body carries none and still validates. Checked where it is given: a
    ticket that is not a content address is not a ticket.
    """
    home, log = minted(tmp_path)
    spliced, other = 'c' * 64, 'd' * 64

    filed = verbs.file_quote(log, DESK, 'Q-1', 'ZeroCostCollar', 'a' * 64, VALUES,
                             {'floor': 17.25}, 4200.0, ticket=spliced, book=BOOK)
    body = log.open_body(log.frame_at(filed['lsn']))
    assert body['ticket'] == spliced and body['plan_hash'] == 'a' * 64
    assert sorted(body) == ['edge', 'plan_hash', 'quote_id', 'solved', 'structure', 'ticket',
                            'values_hash']

    # the same book plan, a different mirror: one pinned hash, two tickets
    second = verbs.file_quote(log, DESK, 'Q-2', 'ZeroCostCollar', 'a' * 64, VALUES,
                              {'floor': 17.5}, 4300.0, ticket=other, book=BOOK)
    assert log.open_body(log.frame_at(second['lsn']))['plan_hash'] == body['plan_hash']
    assert log.open_body(log.frame_at(second['lsn']))['ticket'] != body['ticket']

    with pytest.raises(MalformedEvent) as refusal:
        verbs.file_quote(log, DESK, 'Q-3', 'Straddle', 'a' * 64, VALUES, {}, 0.0,
                         ticket='not-a-hash', book=BOOK)
    assert 'ticket' in str(refusal.value)
    assert log.head()[0] == second['lsn'], 'a refused quote wrote something'
    log.close()
    assert verify_home(home)['events'] == log.head()[0]


def test_a_quote_refuses_a_pin_that_is_not_a_pin_and_a_coordinate_that_is_not_a_number(tmp_path):
    """The solved coordinates are an object of name to finite number, and the plan hash is a
    content address. Neither is coerced: a coordinate that will not read is a term of the trade
    nobody can act on, and the record does not guess at one."""
    home, log = minted(tmp_path)

    with pytest.raises(MalformedEvent) as refusal:
        verbs.file_quote(log, DESK, 'Q-1', 'Straddle', 'not-a-hash', VALUES, {}, 0.0, book=BOOK)
    assert 'plan_hash' in str(refusal.value)

    for broken in ({'floor': 'seventeen'}, {'': 17.0}, {'floor': None}, [17.0], 17.0):
        with pytest.raises(MalformedEvent):
            verbs.file_quote(log, DESK, 'Q-1', 'Straddle', 'a' * 64, VALUES, broken, 0.0,
                             book=BOOK)
    with pytest.raises(MalformedEvent):
        verbs.file_quote(log, DESK, 'Q-1', 'Straddle', 'a' * 64, VALUES, {}, 'wide', book=BOOK)
    assert log.head()[0] == 4
    log.close()


# --------------------------------------------------------------------------------------------
# Durability ordering, and the claim's own shape.

def test_no_attestation_appends_before_the_objects_it_cites_are_on_the_platter(tmp_path):
    """Durability ordering is law and the verbs obey it by construction - every blob is fsynced by
    the verb itself before the event citing it appends. The other half is the writer's, and it is
    what a hand-built body meets: an attestation naming a job nobody stored does not append."""
    home, log = minted(tmp_path)
    absent = 'f' * 64

    with pytest.raises(MissingBlobRefusal) as refusal:
        log.append('run_completed', dict(claim(), lane=verbs.STANDING, job=absent,
                                         result=address(RESULT)), actor=MINT, book=BOOK)
    assert absent in str(refusal.value)
    assert log.head()[0] == 4

    # through the verb, the same body lands: the bytes go first
    assert verbs.complete_run(log, MINT, verbs.STANDING, claim(), JOB, VALUES, RESULT,
                              book=BOOK)['lsn'] == 5
    log.close()
    assert verify_home(home)['events'] == 5


def test_a_replay_claim_is_four_coordinates_and_no_fifth(tmp_path):
    """The tuple names the numbers, so a claim with a coordinate missing names no result at all and
    one with a fifth is not the tuple a result was filed under. A null seed is legal and a
    substituted one is not: a job that declared no `Random_Seed` was hashed with a null there, and
    a zero would record a tuple no result was ever filed under."""
    home, log = minted(tmp_path)

    for broken in ({'plan_hash': 'a' * 64}, dict(claim(), device='cuda'), 'a tuple',
                   dict(claim(), engine_version=''), dict(claim(), seed=1.5),
                   dict(claim(), plan_hash='short')):
        with pytest.raises(MalformedEvent):
            verbs.complete_run(log, MINT, verbs.STANDING, broken, JOB, VALUES, RESULT, book=BOOK)

    seedless = verbs.complete_run(log, MINT, verbs.STANDING, claim(seed=None), JOB, VALUES,
                                  RESULT, book=BOOK)
    assert log.open_body(log.frame_at(seedless['lsn']))['seed'] is None
    log.close()
    assert verify_home(home)['events'] == 5


def test_the_injected_executor_is_checked_rather_than_trusted(tmp_path):
    """A callable somebody handed in is not trusted to answer the contract. Every departure from it
    is a named refusal, because the alternative is a TypeError raised from inside an attestation
    path with a traceback where a refusal should be."""
    home, log = minted(tmp_path)
    with_tolerance(log)

    for broken in (None, 'not callable', lambda job, values, version: RESULT,
                   lambda job, values, version: ('0.1.0',),
                   lambda job, values, version: ('0.1.0', {'mtm': 1234.5})):
        with pytest.raises(ReplayRefused):
            verbs.pin_result(log, MINT, claim(), JOB, VALUES, RESULT, broken, book=BOOK)

    # and a result blob that is not JSON refuses as "nothing to compare" rather than as a departure
    with pytest.raises(ReplayRefused) as refusal:
        verbs.pin_result(log, MINT, claim(), JOB, VALUES, b'not json at all',
                         executor_of(b'also not json'), book=BOOK)
    assert 'nothing here to compare' in str(refusal.value)
    log.close()


def test_the_vocabulary_grew_three_types_and_changed_none():
    """The versioned governance act: three new types with three validators, three verb scopes and
    three `BLOB_FIELDS` rows, and every earlier type validating exactly what it validated before.

    The verb scopes are the interesting half. An attestation and a quote are BOOK; a promotion is
    APPROVE, because giving standing to a tuple this hub never witnessed is a second pair of eyes on
    somebody else's claim.
    """
    from derivus_spine.vocabulary import (
        BLOB_FIELDS, EVENT_TYPES, EVENT_VERB, FACT_TYPES, PROVENANCE_TYPES, SUBMITTABLE, validate)

    assert set(PROVENANCE_TYPES) == {'run_completed', 'result_pinned', 'quote_filed'}
    assert set(PROVENANCE_TYPES).isdisjoint(FACT_TYPES), 'a fourth mouth, not a fourth fact'
    assert set(PROVENANCE_TYPES) <= set(EVENT_TYPES) and set(PROVENANCE_TYPES) <= set(SUBMITTABLE)

    assert EVENT_VERB['run_completed'] == 'book' and EVENT_VERB['quote_filed'] == 'book'
    assert EVENT_VERB['result_pinned'] == 'approve'
    assert BLOB_FIELDS['run_completed'] == ('job', 'result', 'values_hash')
    assert BLOB_FIELDS['result_pinned'] == ('job', 'result', 'values_hash', 'tolerance_policy')
    assert BLOB_FIELDS['quote_filed'] == ('values_hash',)
    assert 'plan_hash' not in BLOB_FIELDS['run_completed'], 'a plan is re-derived, never trusted'

    # the increment-1 validators, unmoved: one required field missing, one surplus key, each
    # refused in the wording they were refused in before this vocabulary grew
    with pytest.raises(MalformedEvent) as refusal:
        validate('fill', {'instrument': 'a' * 64, 'quantity': 1.0, 'counterparty': 'x',
                          'netting_set': 'y'})
    assert 'the body has no execution_reference' in str(refusal.value)
    with pytest.raises(MalformedEvent) as refusal:
        validate('approval', {'plan_hash': 'a' * 64, 'why': 'because'})
    assert 'carries why beyond plan_hash' in str(refusal.value)
    with pytest.raises(UnknownEventType):
        validate('knocked_out', {})


def test_the_optional_field_is_not_an_extension_point():
    """`quote_filed` is the only type declaring optional fields, and `_validator` grew an
    `optional` arm for them. The arm's defence is that an optional field is still NAMED, so the body
    stays as closed as it ever was. A validator that stopped checking surplus the moment a type
    gained an optional field would open the hole on precisely the type that carries a client's own
    words, and an undeclared field sits outside every hash, seal and signature here.

    Three claims: a field may be ABSENT, it is CHECKED when present exactly as a declared field is
    and at its OWN kind, and its presence buys no other field a way in.
    """
    from derivus_spine.vocabulary import validate

    body = {'quote_id': 'Q-1', 'structure': 'Straddle', 'plan_hash': 'a' * 64,
            'values_hash': 'b' * 64, 'solved': {'floor': 17.25}, 'edge': 4200.0}
    validate('quote_filed', body)
    validate('quote_filed', dict(body, request='three months of downside at zero cost'))
    validate('quote_filed', dict(body, ticket='c' * 64))

    # the surplus rule, unmoved: the closure names the optional fields and still refuses the rest
    with pytest.raises(MalformedEvent) as refusal:
        validate('quote_filed', dict(body, surprise='outside every hash the closure names'))
    said = str(refusal.value)
    assert 'carries surprise beyond' in said, said
    for optional in ('request', 'ticket'):
        assert optional in said, 'the refusal does not name the optional fields among the known'

    # and present, each is held to its OWN kind - an optional field is not an unchecked one, and
    # two of them on one type are not one kind
    for broken in (17, '', None, {'said': 'it'}):
        with pytest.raises(MalformedEvent) as refusal:
            validate('quote_filed', dict(body, request=broken))
        assert 'request is' in str(refusal.value), (broken, str(refusal.value))
    for broken in ('not-a-hash', 'C' * 64, 17, None):
        with pytest.raises(MalformedEvent) as refusal:
            validate('quote_filed', dict(body, ticket=broken))
        assert 'ticket is' in str(refusal.value) and 'content hash' in str(refusal.value), broken

    # the types that declare NO optional field are validated by the identical code, and a gate that
    # did not say so would pass on a validator that had quietly stopped closing those too
    with pytest.raises(MalformedEvent) as refusal:
        validate('run_completed', {'plan_hash': 'a' * 64, 'values_hash': 'b' * 64,
                                   'engine_version': '0.1.0', 'seed': 1, 'lane': 'standing',
                                   'job': 'c' * 64, 'result': 'd' * 64, 'surprise': 1})
    assert 'carries surprise beyond' in str(refusal.value)


def test_a_v1_event_still_reads_under_the_vocabulary_that_grew(tmp_path):
    """Version tolerance from the writing side: a home minted before this increment carries frames
    whose bodies this vocabulary never validated, and they still verify - `verify_home` checks that
    a body can be READ rather than re-validating it against today's shapes. The proof is a home
    written with the increment-1 types alone, verified after the increment-3 types exist."""
    home, log = minted(tmp_path)
    verbs.book(log, MINT, INSTRUMENT, 1.0, COUNTERPARTY, CLIENT, 'EXEC-1', book=BOOK)
    log.append('approval', {'plan_hash': 'a' * 64}, actor=MINT, book=BOOK)
    log.close()

    report = verify_home(home)
    assert report == {'mode': 'entitled', 'events': 6, 'checkpoints_verified': 1,
                      'head_lsn': 6, 'head_hash': SpineLog(home).head()[1]}
    assert verify_home(home, entitled=False)['events'] == 6


def test_a_declared_policy_is_stored_canonically_so_one_policy_is_one_blob(tmp_path):
    """One policy must be one blob or the record holds two histories of one decision, so a document
    spelled two ways is stored once - and declaring it twice coalesces on the ordinary tag rule."""
    home, log = minted(tmp_path)

    first = policy.declare(log, MINT, policy.TOLERANCE_POLICY, {'tolerances': {'cva': 1e-6,
                                                                               'mtm': 1e-9}})
    second = policy.declare(log, MINT, policy.TOLERANCE_POLICY, {'tolerances': {'mtm': 1e-9,
                                                                                'cva': 1e-6}})
    assert first['blob'] == second['blob']
    assert second['coalesced'] is True and second['lsn'] == first['lsn']
    assert json.loads(log.store.get(first['blob']).decode('utf-8')) == {
        'tolerances': {'cva': 1e-6, 'mtm': 1e-9}}
    log.close()


def test_a_tiers_policy_declared_twice_leaves_the_last_one_in_force(tmp_path):
    """The fourth reserved name resolves the way the other three do: by the ordinary fold, the LAST
    declaration at or before a position standing. A workflow is governance, so replacing one is a
    whole new document rather than an edit of the one in force, and what routed a ticket in March is
    a fold like every other question.
    """
    home, log = minted(tmp_path)
    assert policy.tiers_in_force(log) is None, 'a home declaring no tiers routes nothing'

    first = policy.declare(log, MINT, policy.TIERS_POLICY,
                           {'tiers': [{'name': 'desk', 'four_eyes': True}]})
    second = policy.declare(log, MINT, policy.TIERS_POLICY, {
        'tiers': [{'name': 'auto', 'seat': 'policy/tiers/auto'}, {'name': 'desk'}],
        'designations': {'settlement_export': 'official'}})

    assert [tier['name'] for tier in policy.tiers_in_force(log)['tiers']] == ['auto', 'desk']
    assert policy.tiers_in_force(log)['designations'] == {'settlement_export': 'official'}
    assert [tier['name'] for tier in policy.tiers_in_force(log, second['lsn'] - 1)['tiers']] == \
        ['desk']
    assert first['blob'] != second['blob'] and first['lsn'] < second['lsn']
    # the completed document is what stands, so the catch-all says what it means about four eyes
    assert policy.tiers_in_force(log)['tiers'][1] == {'name': 'desk', 'four_eyes': False}
    log.close()
