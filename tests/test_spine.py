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

"""The spine's own acceptance bar - real homes on disk, and every wire format recomputed here.

Nothing in this file asks the log what it thinks. The semantic tuple, the HMAC that blinds it, the
nine-field AAD, the event hash over (ciphertext_hash, idempotency_tag, prev_hash, record_time) and
the exact spelling of a frame line are all rebuilt from `hashlib`, `hmac` and the canonicaliser, so
an implementation that agreed with itself but not with the wire format turns this file red.

Faults are injected the only way the house allows: by DOCTORING BYTES ON DISK. Not one line here
patches library code, and the three tamper detections run on three separate copies of one home,
because a tamper that a previous tamper already broke proves nothing. The three are chosen to
separate the mechanisms rather than to repeat one:

  * a body byte says the ciphertext is inside the chain hash;
  * an actor field says the plaintext ENVELOPE is inside the seal - the chain alone cannot see it,
    and the gate asserts chain-only stays green over exactly that edit;
  * a record time says the chain hash covers the envelope's timestamps, and is caught by a replica
    holding no key at all.

Two tampers are the insider's, and they are the sharpest ones here, because each disables the
other's detector. A tail frame re-sealed with this home's own key and its chain hash recomputed
checks out as a chain; what catches it is the interior binding - and, on a copy with no blind key,
ONLY the interior binding, which is what makes that check load-bearing rather than shadowed. The
dual keeps the binding honest and leaves the idempotency tag stale, and only the blinded
recomputation notices. Without both, half the entitled check could be decoration and every other
gate would still be green.

The rest of the file is the brief's law, one bullet at a time: a retry that is the same fact by
construction with nothing said about when it is true, collision that refuses, blinding that leaves
an unentitled holder no computable check, a crypto-shred that empties the bodies and leaves the
chain standing, referential closure asked of the writer AND of the whole history against the
manifest, a closed vocabulary and a closed frame, break-glass declared at genesis, a synthetic book
whose late booking makes as-of and as-at disagree across a restated close and a republished fixing,
checkpoint authenticity under the key in force at each checkpoint's own LSN, one writer and a named
refusal for the second, the one torn line the writer is allowed to remove and the terminated one it
must not, and a replica that holds the log and the blobs and no key at all.
"""
import base64
import hashlib
import hmac
import json
import os
import shutil
import sys

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from derivus_spine import (
    ChainBroken, CheckpointInvalid, CollisionRefusal, HomeExists, HomeMissing, MalformedEvent,
    MissingBlobRefusal, SealedBodyUnreadable, SpineLog, UnknownEventType, WriterBusy,
    canonical_bytes, init_home, verify_home, write_checkpoint)
from derivus_spine.log import as_of_key
from derivus_spine.seal import Keys
from derivus_spine.vocabulary import FACT_TYPES

ACTOR = 'subject-desk-one'
BOOK = 'FX-VANILLA'
INSTRUMENT = hashlib.sha256(b'EURUSD 1.0850 2026-12-18 call').hexdigest()
OTHER = hashlib.sha256(b'EURUSD 1.1000 2026-12-18 call').hexdigest()
#: One instant, named wherever a gate wants several facts to share a truth-time. A retry does NOT
#: need it: `effective_time` is null when the caller says nothing, and null is the same on the
#: second call as on the first, which IS the retry law rather than a way around it.
WHEN = '2026-08-29T09:15:00.000000Z'
#: The synthetic book's week. The Monday fill is recorded after the Wednesday one, which is what
#: makes as-of and as-at disagree on this fixture rather than merely differ in principle.
MON = '2026-08-24T08:00:00.000000Z'
TUE = '2026-08-25T08:00:00.000000Z'
WED = '2026-08-26T08:00:00.000000Z'
CLOSE = '2026-08-26T16:30:00.000000Z'
#: The nine pre-LSN envelope fields the body is sealed against, spelled out here rather than
#: imported - this gate exists to disagree with the writer if the writer changes its mind.
AAD_FIELDS = ('actor', 'book', 'effective_time', 'entitlement_class', 'event_type',
              'event_version', 'idempotency_tag', 'prev_hash', 'record_time')


# --------------------------------------------------------------------------------------------
# The wire format, rebuilt by hand.

def semantic(event_type, body, actor=ACTOR, book=BOOK, effective_time=WHEN):
    """The semantic tuple every hash of an event's MEANING is taken over."""
    return {'actor': actor, 'body': body, 'book': book, 'effective_time': effective_time,
            'type': event_type, 'version': 1}


def semantic_of(frame, body):
    """The semantic tuple a STORED frame's envelope implies, around a body of the caller's
    choosing - what an insider re-sealing a line has to get right for the binding to stay honest."""
    return semantic(frame['event_type'], body, actor=frame['actor'], book=frame['book'],
                    effective_time=frame['effective_time'])


def ed25519_pair():
    """A signing pair minted here rather than inside a home: what a logged key ROTATION publishes,
    and what an insider substituting a key would have to hand the verifier."""
    private = Ed25519PrivateKey.generate()
    return private, private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)


def sign(private, lsn, event_hash):
    """The checkpoint payload, signed - `{"event_hash", "lsn"}` canonicalised and nothing else."""
    return private.sign(canonical_bytes({'event_hash': event_hash, 'lsn': lsn})).hex()


def tag_of(home, tuple_):
    """The idempotency tag: HMAC-SHA256 of the canonical semantic bytes under the home's own
    blind key."""
    blind = (home / 'keys' / 'blind.key').read_bytes()
    return hmac.new(blind, canonical_bytes(tuple_), hashlib.sha256).hexdigest()


def aad_of(frame):
    """GCM's additional data: the nine pre-LSN envelope fields, canonicalised."""
    return canonical_bytes(dict((field, frame[field]) for field in AAD_FIELDS))


def chain_of(frame):
    """The event hash the frame's own bytes imply."""
    sealed = base64.b64decode(frame['body'])
    return hashlib.sha256(canonical_bytes({
        'ciphertext_hash': hashlib.sha256(sealed).hexdigest(),
        'idempotency_tag': frame['idempotency_tag'],
        'prev_hash': frame['prev_hash'],
        'record_time': frame['record_time']})).hexdigest()


def fill(reference, quantity=1000000.0, instrument=INSTRUMENT):
    """A fill body: signed quantity, counterparty and netting set on the row, and the execution
    reference that makes a retry the same fact and two clips two facts."""
    return {'instrument': instrument, 'quantity': quantity, 'netting_set': 'CSA-0007',
            'counterparty': 'LEI-5493001KJTIIGC8Y1R12', 'execution_reference': reference}


# --------------------------------------------------------------------------------------------
# Homes, and the doctoring of them.

def seeded(root, name='home', clips=('EXEC-1', 'EXEC-2', 'EXEC-3')):
    """A minted home with genesis plus one fill per clip - LSNs 1..4 are genesis, 5.. are trades.

    The writer is closed before the home is handed back: one deployment, one log, one writer, and a
    fixture that kept its claim would be the thing refusing the test's own next append.
    """
    home = root / name
    init_home(home, ACTOR)
    log = SpineLog(home)
    for reference in clips:
        log.append('fill', fill(reference), actor=ACTOR, book=BOOK, effective_time=WHEN)
    log.close()
    return home


def blob_path(home, digest):
    """Where the store files `digest` - the address a gate deletes at to make a blob vanish the way
    a silent expiry would."""
    return home / 'blobs' / digest[:2] / digest[2:4] / digest


def superseded(pairs, event_type, key):
    """The (effective_time, LSN) supersession rule as a three-line fold: rows of one type, read as
    of, the last one under each key winning.

    An administrator's republication is a new row under the same (index, date, source) key and a
    restated close is a new close - never an edit - so the winner is a fold's answer rather than a
    column somebody overwrote.
    """
    winner = {}
    for frame, body in sorted(pairs, key=lambda pair: as_of_key(pair[0])):
        if frame['event_type'] == event_type:
            winner[key(body)] = body
    return winner


def segments(home):
    return sorted((home / 'log').glob('segment-*.jsonl'))


def read_frames(home):
    """Every frame, read straight off the platter by this gate rather than by the log."""
    frames = []
    for path in segments(home):
        for raw in path.read_bytes().split(b'\n'):
            if raw.strip():
                frames.append(json.loads(raw.decode('utf-8')))
    return frames


def doctor(home, lsn, mutate):
    """Rewrite the frame at `lsn` in place - the fault injection this house permits.

    Every other line is re-emitted byte-identically (the writer's own `sort_keys`,
    `separators=(",", ":")` spelling), so exactly one line differs from what was fsynced.
    """
    for path in segments(home):
        lines = [raw for raw in path.read_bytes().split(b'\n') if raw.strip()]
        rebuilt, touched = [], False
        for raw in lines:
            frame = json.loads(raw.decode('utf-8'))
            if frame['lsn'] == lsn:
                mutate(frame)
                touched = True
            rebuilt.append(json.dumps(frame, sort_keys=True, separators=(',', ':')).encode('utf-8'))
        if touched:
            path.write_bytes(b'\n'.join(rebuilt) + b'\n')
            return
    raise AssertionError('no frame at LSN {} to doctor'.format(lsn))


def flip_base64(text):
    """One character of a base64 body, moved - the smallest edit that changes the ciphertext."""
    return ('B' if text[0] != 'B' else 'C') + text[1:]


# --------------------------------------------------------------------------------------------
# Genesis.

def test_genesis_writes_four_facts_and_verifies_in_both_modes(tmp_path):
    """Four events, and each one answers a question a replica would otherwise ask a human: who
    governs, how the last admin is recovered, which key signs the checkpoints, and where the head
    stood when the home was minted."""
    home = tmp_path / 'home'
    summary = init_home(home, ACTOR)
    assert summary['events'] == 4 and summary['head_lsn'] == 4

    frames = read_frames(home)
    assert [frame['lsn'] for frame in frames] == [1, 2, 3, 4]
    assert [frame['event_type'] for frame in frames] == \
        ['policy_declared', 'policy_declared', 'policy_declared', 'checkpoint']
    assert frames[0]['prev_hash'] == '0' * 64, 'genesis chains onto the edge, not onto a hash'
    assert all(frame['actor'] == ACTOR and frame['book'] is None for frame in frames)
    assert all(frame['entitlement_class'] == 'firm' for frame in frames)

    log = SpineLog(home)
    bodies = [log.open_body(frame) for frame in log.frames()]
    assert bodies[2]['policy'] == 'checkpoint_verifying_key'
    published = log.store.get(bodies[2]['blob'])
    assert published == (home / 'keys' / 'checkpoint_verify.key').read_bytes(), \
        'the blob a replica reads must be the key this home signs with'
    assert bodies[3] == {'lsn': 3, 'event_hash': frames[2]['event_hash'],
                         'signature': bodies[3]['signature']}, 'the first checkpoint covers LSN 3'

    entitled = verify_home(home)
    assert entitled == {'mode': 'entitled', 'events': 4, 'checkpoints_verified': 1,
                        'head_lsn': 4, 'head_hash': frames[3]['event_hash']}
    chain_only = verify_home(home, entitled=False)
    assert chain_only['mode'] == 'chain-only' and chain_only['events'] == 4
    assert chain_only['head_hash'] == entitled['head_hash']
    # Not a zero: a count here would be a lie a script believes.
    assert chain_only['checkpoints_verified'] == 'not assessed'


def test_a_second_genesis_over_a_home_is_refused_and_a_missing_home_is_named(tmp_path):
    """A home is minted once. A retyped `init` must not fork the record, and a verb pointed at a
    directory that is not a home says so rather than conjuring one."""
    home = seeded(tmp_path)
    with pytest.raises(HomeExists) as refusal:
        init_home(home, ACTOR)
    assert str(home) in str(refusal.value), refusal.value
    assert verify_home(home)['events'] == 7, 'the refusal wrote nothing'
    with pytest.raises(HomeMissing):
        SpineLog(tmp_path / 'not_a_home')
    with pytest.raises(HomeMissing):
        verify_home(tmp_path / 'not_a_home')


# --------------------------------------------------------------------------------------------
# The mandated mutation: three tampers, three detections, three copies.

def test_a_doctored_body_byte_breaks_the_chain_at_its_lsn(tmp_path):
    """The chain hash is taken over the CIPHERTEXT, so a body edited in place is caught by the
    recomputation - and named by the LSN it happened at, which is the whole point of a positional
    sequence."""
    home = seeded(tmp_path, 'body')
    doctor(home, 5, lambda frame: frame.__setitem__('body', flip_base64(frame['body'])))
    with pytest.raises(ChainBroken) as refusal:
        verify_home(home)
    assert 'LSN 5' in str(refusal.value), refusal.value
    # The gate's own arithmetic says the same thing: the stored hash no longer follows the bytes.
    doctored = [frame for frame in read_frames(home) if frame['lsn'] == 5][0]
    assert chain_of(doctored) != doctored['event_hash']


def test_a_doctored_envelope_field_breaks_the_seal_the_chain_alone_would_miss(tmp_path):
    """The envelope is plaintext but not unprotected: it is GCM's additional data, so moving an
    actor makes the body stop opening. This is the tamper the chain cannot see - the gate asserts
    chain-only stays GREEN over exactly this edit - and it is why the AAD carries nine fields
    rather than none."""
    home = seeded(tmp_path, 'envelope')
    doctor(home, 5, lambda frame: frame.__setitem__('actor', 'subject-someone-else'))
    doctored = [frame for frame in read_frames(home) if frame['lsn'] == 5][0]
    assert chain_of(doctored) == doctored['event_hash'], \
        'the actor is outside the chain hash by design - that is what makes this tamper the '\
        'seal\'s to catch'
    assert verify_home(home, entitled=False)['events'] == 7
    with pytest.raises(ChainBroken) as refusal:
        verify_home(home)
    assert 'LSN 5' in str(refusal.value), refusal.value


def test_a_doctored_record_time_is_caught_by_a_replica_holding_no_key(tmp_path):
    """`record_time` is inside the event hash, so an unentitled replica - bodies sealed, no key
    anywhere - still catches a rewritten writer clock. Verification is local and needs no
    entitlement to be worth something."""
    home = seeded(tmp_path, 'clock')
    doctor(home, 5, lambda frame: frame.__setitem__('record_time', '2020-01-01T00:00:00.000000Z'))
    with pytest.raises(ChainBroken) as refusal:
        verify_home(home, entitled=False)
    assert 'LSN 5' in str(refusal.value), refusal.value
    with pytest.raises(ChainBroken):
        verify_home(home)


def test_a_re_forged_tail_frame_is_caught_by_the_interior_binding(tmp_path):
    """The fourth tamper, and the only one an INSIDER could mount: someone holding the class key
    reseals the last event around a different payload and recomputes its chain hash, so the chain
    checks out - an unentitled replica sees nothing wrong, and the gate asserts that out loud. What
    catches it is the interior binding, re-derived from the envelope plus the decrypted payload,
    which is the reason the plaintext content hash lives INSIDE the sealed body rather than
    decoratively beside it.

    (Only a tail frame: re-forging a middle one means re-sealing every event after it too, because
    `prev_hash` is one of the nine fields their bodies are bound to - the seal chains the envelope
    into the ciphertext, which is worth stating as a fact about the design.)

    The second copy is where the binding is proven LOAD-BEARING rather than merely present. On the
    home that holds the blind key the tag check would catch this too, since a re-forged payload
    moves the semantic tuple; strip that key - the entitled replica's posture, the blind key never
    leaves the hub - and the interior binding is the only check left. Which kills the mutant that
    disables the `rebuilt != interior['content_hash']` comparison: without this copy that mutant
    survives everything except a substring assertion.
    """
    home = seeded(tmp_path, 'binding')
    frame = [f for f in read_frames(home) if f['lsn'] == 7][0]
    substitute = fill('EXEC-3', quantity=-42.0)
    # An honest seal, under the frame's own unchanged envelope, around a DISHONEST binding: the
    # content hash of some other fact entirely.
    interior = canonical_bytes({
        'content_hash': hashlib.sha256(
            canonical_bytes(semantic('fill', fill('EXEC-ELSEWHERE')))).hexdigest(),
        'payload': substitute})
    sealed = Keys(home).seal(interior, aad_of(frame))

    def reforge(target):
        target['body'] = base64.b64encode(sealed).decode('ascii')
        target['event_hash'] = chain_of(target)

    doctor(home, 7, reforge)

    assert verify_home(home, entitled=False)['events'] == 7, \
        'the chain is internally consistent again - an unentitled replica cannot see this one'
    with pytest.raises(ChainBroken) as refusal:
        verify_home(home)
    assert 'LSN 7' in str(refusal.value) and 'binding' in str(refusal.value), refusal.value
    log = SpineLog(home)
    assert log.open_body(log.frame_at(7)) == substitute, \
        'the body opens cleanly - it is the binding, not the seal, that catches this one'

    # The same forgery on an entitled replica that holds no blind key: the tag check cannot run
    # there, so the refusal below is the interior binding speaking on its own.
    replica = tmp_path / 'binding_no_blind'
    shutil.copytree(str(home), str(replica))
    os.unlink(str(replica / 'keys' / 'blind.key'))
    with pytest.raises(ChainBroken) as refusal:
        verify_home(replica)
    assert 'LSN 7' in str(refusal.value) and 'binding' in str(refusal.value), refusal.value


def test_a_stale_idempotency_tag_over_an_honest_binding_is_caught(tmp_path):
    """The dual of the forgery above, and the other half of what an entitled verification owes.

    Here the insider is careful: the payload is substituted, the interior content hash is recomputed
    HONESTLY over it, and the idempotency tag is left exactly as it was - so the nine-field AAD
    never moves, the seal stays valid, the binding agrees with itself, and the chain hash is
    recomputed to match. Every check but one passes. The tag is the field this attacker cannot
    forge, because forging it needs the blind key that never leaves the hub, and the recomputation
    is where that is cashed: on the hub, or on anyone verifying a restored backup.

    Kills the mutant that turns the blinded recomputation off (`if blinded:` -> `if False:`), under
    which this home verifies fully green and the fill reads -42.
    """
    home = seeded(tmp_path, 'stale')
    frame = [f for f in read_frames(home) if f['lsn'] == 7][0]
    substitute = fill('EXEC-3', quantity=-42.0)
    interior = canonical_bytes({
        'content_hash': hashlib.sha256(
            canonical_bytes(semantic_of(frame, substitute))).hexdigest(),
        'payload': substitute})
    sealed = Keys(home).seal(interior, aad_of(frame))

    def reforge(target):
        target['body'] = base64.b64encode(sealed).decode('ascii')
        target['event_hash'] = chain_of(target)

    doctor(home, 7, reforge)

    assert verify_home(home, entitled=False)['events'] == 7, 'the chain checks out, as intended'
    with pytest.raises(ChainBroken) as refusal:
        verify_home(home)
    assert 'LSN 7' in str(refusal.value) and 'tag' in str(refusal.value), refusal.value

    # and the replica that cannot recompute the tag cannot see it - which is the point of saying so
    # in the report rather than reporting a check that did not run
    replica = tmp_path / 'stale_no_blind'
    shutil.copytree(str(home), str(replica))
    os.unlink(str(replica / 'keys' / 'blind.key'))
    assert verify_home(replica)['events'] == 7
    log = SpineLog(replica)
    assert log.open_body(log.frame_at(7)) == substitute, \
        'the substitution is readable there, and only the hub\'s tag recomputation names it'


# --------------------------------------------------------------------------------------------
# Idempotency and collision.

def test_a_retried_append_coalesces_and_two_clips_are_two_facts(tmp_path):
    """The retry is safe by construction: the same fact meets its own tag and coalesces onto the
    event already written. Two legitimately identical clips differ in their execution reference and
    are therefore two facts - which is exactly why fills must carry one."""
    home = seeded(tmp_path, clips=())
    log = SpineLog(home)
    first = log.append('fill', fill('EXEC-77'), actor=ACTOR, book=BOOK, effective_time=WHEN)
    assert first['coalesced'] is False and first['lsn'] == 5

    retry = log.append('fill', fill('EXEC-77'), actor=ACTOR, book=BOOK, effective_time=WHEN)
    assert retry['coalesced'] is True
    assert retry['lsn'] == 5 and retry['event_hash'] == first['event_hash']
    assert log.head() == (5, first['event_hash'])
    assert len(read_frames(home)) == 5, 'a coalesced retry writes no line'

    # A second desk, a second clip, the same economics: two facts.
    twin = log.append('fill', fill('EXEC-78'), actor=ACTOR, book=BOOK, effective_time=WHEN)
    assert twin['coalesced'] is False and twin['lsn'] == 6
    assert twin['idempotency_tag'] != first['idempotency_tag']
    assert verify_home(home)['events'] == 6

    # The tag this gate computes is the tag the envelope carries.
    assert first['idempotency_tag'] == tag_of(home, semantic('fill', fill('EXEC-77')))
    log.close()


def test_a_retry_that_says_nothing_about_when_is_still_the_same_fact(tmp_path):
    """The brief's sentence itself rather than a proxy for it: client retries of `book` are safe BY
    CONSTRUCTION, so the same fill submitted twice - with nothing said about when it is true, which
    is how a submitter that is merely retrying submits - meets its own tag and coalesces.

    That holds only because the writer refuses to stamp `effective_time` itself. Kills the mutant
    that defaults it to `record_time` before the semantic tuple is built: under it this gate finds
    two events on disk and two different tags, because the writer's clock moved between the calls
    and the moment of arrival ended up inside the meaning of the fact.
    """
    home = seeded(tmp_path, clips=())
    log = SpineLog(home)
    first = log.append('fill', fill('EXEC-RETRY'), actor=ACTOR, book=BOOK)
    assert first['coalesced'] is False and first['lsn'] == 5
    assert first['effective_time'] is None, \
        'a fact with no truth-time of its own carries null, not the writer\'s clock'

    retry = log.append('fill', fill('EXEC-RETRY'), actor=ACTOR, book=BOOK)
    assert retry['coalesced'] is True
    assert retry['lsn'] == 5 and retry['event_hash'] == first['event_hash']
    assert retry['idempotency_tag'] == first['idempotency_tag']
    assert len(read_frames(home)) == 5, 'a retry that mints a second event is not a retry'
    assert log.head() == (5, first['event_hash'])

    # and the fact still READS as of when it was recorded: the resolution is the reader's, which
    # is the whole reason the writer can leave it null
    frame = log.frame_at(5)
    assert as_of_key(frame) == (frame['record_time'], 5)
    assert verify_home(home)['events'] == 5


def test_a_duplicate_tag_over_different_bytes_is_refused_by_name(tmp_path):
    """Verify-then-dedup, end to end. A tag hit is proven identical by DECRYPTING the stored event
    and byte-comparing, so a stored body that says something else is a named refusal rather than a
    silent swap - and a stored body that will not open is refused too, because 'probably the same'
    is how a record acquires a second version of a fact."""
    home = seeded(tmp_path, 'resealed', clips=('EXEC-9',))
    frame = [f for f in read_frames(home) if f['lsn'] == 5][0]
    tag = frame['idempotency_tag']

    # Doctoring that stays legible: a different payload, sealed under the SAME envelope, so the
    # writer's byte-compare is what has to catch it rather than the decryption.
    substitute = fill('EXEC-9', quantity=-1000000.0)
    interior = canonical_bytes({'content_hash': hashlib.sha256(
        canonical_bytes(semantic('fill', substitute))).hexdigest(), 'payload': substitute})
    sealed = Keys(home).seal(interior, aad_of(frame))
    doctor(home, 5, lambda f: f.__setitem__('body', base64.b64encode(sealed).decode('ascii')))

    with pytest.raises(CollisionRefusal) as refusal:
        SpineLog(home).append('fill', fill('EXEC-9'), actor=ACTOR, book=BOOK, effective_time=WHEN)
    assert 'LSN 5' in str(refusal.value) and tag in str(refusal.value), refusal.value

    # The cruder doctoring on its own copy: a body that will not open at all is the same refusal,
    # because the duplicate still cannot be proven identical.
    torn = seeded(tmp_path, 'unopenable', clips=('EXEC-9',))
    doctor(torn, 5, lambda f: f.__setitem__('body', flip_base64(f['body'])))
    with pytest.raises(CollisionRefusal) as refusal:
        SpineLog(torn).append('fill', fill('EXEC-9'), actor=ACTOR, book=BOOK, effective_time=WHEN)
    assert 'LSN 5' in str(refusal.value), refusal.value


# --------------------------------------------------------------------------------------------
# Blinding, and what an unentitled holder cannot compute.

def test_the_envelope_tag_is_blinded_and_key_dependent(tmp_path):
    """A raw plaintext hash in a firm-visible envelope is a dictionary oracle - any low-entropy
    body could be confirmed by hashing candidates. The envelope therefore carries an HMAC under a
    key that never leaves the hub, so the same fact wears different tags in two homes and no
    keyless check against a candidate plaintext exists."""
    home = seeded(tmp_path, 'first', clips=('EXEC-5',))
    other = seeded(tmp_path, 'second', clips=('EXEC-5',))
    tuple_ = semantic('fill', fill('EXEC-5'))

    mine = [f for f in read_frames(home) if f['lsn'] == 5][0]['idempotency_tag']
    theirs = [f for f in read_frames(other) if f['lsn'] == 5][0]['idempotency_tag']

    plain = hashlib.sha256(canonical_bytes(tuple_)).hexdigest()
    assert mine != plain, 'the tag is not the plaintext hash - that hash is the oracle'
    assert mine != theirs, 'one fact, two homes, two tags: the tag is key-dependent'
    assert mine == tag_of(home, tuple_) and theirs == tag_of(other, tuple_), \
        'with the home\'s own blind key the tag recomputes exactly'


# --------------------------------------------------------------------------------------------
# Crypto-shred.

def test_destroying_the_class_key_leaves_the_chain_green_and_the_bodies_gone(tmp_path):
    """Erasure of the bodies inside an untouched chain. One file destroyed and this home can never
    read its own facts again - which is what crypto-shredding IS - while the chain over ciphertext
    verifies exactly as before, and the refusal names the key rather than pretending the log is
    broken."""
    home = seeded(tmp_path, 'live')
    shredded = tmp_path / 'shredded'
    shutil.copytree(str(home), str(shredded))
    before = verify_home(shredded, entitled=False)
    os.unlink(str(shredded / 'keys' / 'class_firm.key'))

    assert verify_home(shredded, entitled=False) == before, 'the chain does not notice'
    with pytest.raises(SealedBodyUnreadable) as refusal:
        verify_home(shredded)
    assert 'class_firm.key' in str(refusal.value), refusal.value

    log = SpineLog(shredded)
    with pytest.raises(SealedBodyUnreadable):
        log.open_body(next(log.frames()))
    assert log.head() == SpineLog(home).head(), 'the position is unchanged; only the bodies went'


# --------------------------------------------------------------------------------------------
# Referential closure and the closed vocabulary.

def test_an_event_citing_an_absent_blob_does_not_append(tmp_path):
    """Durability ordering is law: the blob is on the platter before the fact that speaks of it.
    The refusal is checked BEFORE anything is written, so the head is exactly where it was - a
    rejected append is not a partial one."""
    home = seeded(tmp_path)
    log = SpineLog(home)
    head = log.head()
    sizes = [path.stat().st_size for path in segments(home)]
    absent = hashlib.sha256(b'a surface nobody stored').hexdigest()

    with pytest.raises(MissingBlobRefusal) as refusal:
        log.append('snapshot_registered', {'blob': absent}, actor=ACTOR, book=BOOK,
                   blob_refs=(absent,))
    assert absent in str(refusal.value), refusal.value

    # And with nothing declared at all: what the BODY says lives in the store is checked too, so an
    # unfsynced blob preceding its event is unrepresentable in the writer path rather than merely
    # discouraged - a submitter cannot get past the law by forgetting to mention it.
    with pytest.raises(MissingBlobRefusal) as refusal:
        log.append('snapshot_registered', {'blob': absent}, actor=ACTOR, book=BOOK)
    assert absent in str(refusal.value) and 'blob' in str(refusal.value), refusal.value

    assert log.head() == head and SpineLog(home).head() == head
    assert [path.stat().st_size for path in segments(home)] == sizes, 'not one byte was written'

    # And the same event, once the blob is there, appends.
    stored = log.store.put(b'a surface nobody stored')
    assert stored == absent
    assert log.append('snapshot_registered', {'blob': absent}, actor=ACTOR, book=BOOK,
                      blob_refs=(absent,))['lsn'] == head[0] + 1
    assert verify_home(home)['events'] == head[0] + 1


def test_a_cited_blob_that_went_quietly_is_caught_by_the_manifest(tmp_path):
    """Referential closure over the whole history, not only over the moment of the append. The
    manifest is a walk of the blob store - a projection, rebuilt rather than kept - and every hash
    an event says lives there must be in it.

    This is where the retention law is cashed: any blob class reduces through an explicit logged
    retention event and NEVER silently, so a citation the manifest cannot resolve is exactly the
    silent expiry the law forbids, and it is named by the LSN that cites it and the hash it cites.
    Kills the mutant that drops the walk, under which a log speaking of a blob that has vanished
    verifies fully green.
    """
    home = seeded(tmp_path)
    log = SpineLog(home)
    surface = log.store.put(b'{"pillars":[0.0810,0.0790,0.0785]}')
    registered = log.append('snapshot_registered', {'blob': surface}, actor=ACTOR, book=BOOK)
    log.close()
    assert verify_home(home)['events'] == registered['lsn']
    assert surface in set(SpineLog(home).store.walk())

    os.unlink(str(blob_path(home, surface)))
    with pytest.raises(MissingBlobRefusal) as refusal:
        verify_home(home)
    assert surface in str(refusal.value), refusal.value
    assert 'LSN {}'.format(registered['lsn']) in str(refusal.value), refusal.value
    # the chain over ciphertext is untouched: the blob went, the history did not
    assert verify_home(home, entitled=False)['events'] == registered['lsn']


def test_the_vocabulary_is_closed_and_its_bodies_are_shaped(tmp_path):
    """A knock is a projection, not a fact: the type does not exist, and the refusal says the
    vocabulary is closed rather than offering an extension point. Inside a type, a missing field or
    a surplus one is named - a field no projector folds is a silence discovered in year three."""
    home = seeded(tmp_path)
    log = SpineLog(home)
    head = log.head()

    with pytest.raises(UnknownEventType) as refusal:
        log.append('knocked_out', {'instrument': INSTRUMENT}, actor=ACTOR, book=BOOK)
    assert 'vocabulary is closed' in str(refusal.value) and 'knocked_out' in str(refusal.value)

    incomplete = fill('EXEC-1')
    del incomplete['execution_reference']
    with pytest.raises(MalformedEvent) as refusal:
        log.append('fill', incomplete, actor=ACTOR, book=BOOK)
    assert 'execution_reference' in str(refusal.value), refusal.value

    surplus = dict(fill('EXEC-1'), knocked='yes')
    with pytest.raises(MalformedEvent) as refusal:
        log.append('fill', surplus, actor=ACTOR, book=BOOK)
    assert 'knocked' in str(refusal.value), refusal.value

    with pytest.raises(MalformedEvent):
        log.append('fill', dict(fill('EXEC-1'), instrument='not-a-hash'), actor=ACTOR, book=BOOK)
    with pytest.raises(MalformedEvent):
        log.append('fixing_observed', {'index': 'EURUSD-ECB', 'date': '2026-08-28',
                                       'source': 'ECB', 'value': 'one point one'},
                   actor=ACTOR, book=None)
    with pytest.raises(MalformedEvent):
        log.append('fill', fill('EXEC-1'), actor=ACTOR, book=BOOK, effective_time='2026-08-29')

    assert log.head() == head, 'every refusal above wrote nothing'
    # Policy bodies are open by design: a policy document's shape is the policy's own business.
    assert log.append('policy_declared',
                      {'policy': 'firmness', 'pillar_age_seconds': 30, 'participation': 0.5},
                      actor=ACTOR)['lsn'] == head[0] + 1
    assert verify_home(home)['events'] == head[0] + 1


# --------------------------------------------------------------------------------------------
# Break-glass, determinism, checkpoints.

def test_break_glass_is_declared_at_genesis_and_its_use_is_a_fact(tmp_path):
    """The recovery path exists before the accident. Both grants are read back off the DECRYPTED
    genesis events, a declaration that strands the last admin appends like any other fact, and the
    break-glass use that follows is itself an appended, chained fact rather than a story."""
    home = seeded(tmp_path)
    log = SpineLog(home)
    genesis = [log.open_body(frame) for frame in log.frames(start_lsn=1, end_lsn=2)]
    assert genesis[0] == {'policy': 'genesis',
                          'grants': [{'subject': ACTOR, 'scope': 'admin'}]}
    assert genesis[1] == {'policy': 'break_glass', 'grant': {'subject': ACTOR}}

    stranded = log.append('policy_declared',
                          {'policy': 'grants', 'grants': [],
                           'revokes': [{'subject': ACTOR, 'scope': 'admin'}]}, actor=ACTOR)
    used = log.append('break_glass_used',
                      {'reason': 'the grant declaration at LSN {} left no admin'.format(
                          stranded['lsn'])}, actor=ACTOR)
    assert used['prev_hash'] == stranded['event_hash'], 'the recovery is in the chain'
    assert verify_home(home)['head_lsn'] == used['lsn']


def test_events_sharing_an_effective_time_replay_in_lsn_order(tmp_path):
    """Fold determinism, stated once and gated: as-at order is LSN order, and as-of order breaks
    its ties by LSN - so a fold is a function of the log rather than of a sort's tie-breaking."""
    home = seeded(tmp_path, clips=())
    log = SpineLog(home)
    written = [log.append('fill', fill(reference), actor=ACTOR, book=BOOK, effective_time=WHEN)
               for reference in ('EXEC-A', 'EXEC-B', 'EXEC-C')]
    assert [envelope['lsn'] for envelope in written] == [5, 6, 7]

    frames = [frame for frame in SpineLog(home).frames(start_lsn=5)]
    assert len(set(frame['effective_time'] for frame in frames)) == 1
    assert [as_of_key(frame) for frame in frames] == [(WHEN, 5), (WHEN, 6), (WHEN, 7)]
    assert [frame['lsn'] for frame in sorted(frames, key=as_of_key)] == [5, 6, 7]
    assert [frame['lsn'] for frame in sorted(reversed(frames), key=as_of_key)] == [5, 6, 7]
    log.close()


# --------------------------------------------------------------------------------------------
# The synthetic book - the fixture every later increment's reconstruction gate folds.

def synthetic_book(tmp_path):
    """The brief's fixture, appended through the ordinary writer and nothing else.

    A late booking (Monday's fill, recorded after Wednesday's), a backdated amendment behind it, an
    exercise election, an approval and the rejection that answers it from a second seat, a
    determination and a status transition, an administrator's republished fixing under the same
    (index, date, source) key, and a backdated observation arriving after an official close - which
    is answered by a NEW close, never an edit. Every fact in the closed vocabulary reaches the
    writer here.

    Answers `(home, log, marks)`.
    """
    home = seeded(tmp_path, 'book', clips=())
    log = SpineLog(home)
    marks = {
        'plan': hashlib.sha256(b'the compiled job this desk approved').hexdigest(),
        'values': log.store.put(b'{"EURUSD":1.0851}'),
        'restated': log.store.put(b'{"EURUSD":1.0857}'),
        'policy': log.store.put(b'{"tape":"90 days, then the logged reduction"}'),
        'snapshot': log.store.put(b'{"surface":"the vol cube of 2026-08-26"}'),
    }
    log.append('fill', fill('EXEC-WED'), actor=ACTOR, book=BOOK, effective_time=WED)
    log.append('fill', fill('EXEC-MON'), actor=ACTOR, book=BOOK, effective_time=MON)
    log.append('amendment', {'instrument': INSTRUMENT, 'amended_to': OTHER},
               actor=ACTOR, book=BOOK, effective_time=MON)
    log.append('election', {'instrument': OTHER, 'choice': 'exercise'},
               actor=ACTOR, book=BOOK, effective_time=TUE)
    log.append('approval', {'plan_hash': marks['plan']},
               actor=ACTOR, book=BOOK, effective_time=TUE)
    log.append('rejection', {'plan_hash': marks['plan'],
                             'reason': 'the booker and the approver are one seat'},
               actor='subject-desk-two', book=BOOK, effective_time=TUE)
    log.append('determination', {'subject': INSTRUMENT, 'ruling': 'the barrier was touched at 14:02'},
               actor=ACTOR, book=BOOK, effective_time=TUE)
    log.append('status_transition', {'subject': INSTRUMENT, 'status': 'confirmed'},
               actor=ACTOR, book=BOOK, effective_time=TUE)
    log.append('market_declared', {'name': 'official', 'values_hash': marks['values']},
               actor=ACTOR, effective_time=WED)
    log.append('fixing_observed',
               {'index': 'EURUSD-ECB', 'date': '2026-08-26', 'source': 'ECB', 'value': 1.0851},
               actor=ACTOR, effective_time=WED)
    log.append('official_close_declared', {'market': 'official', 'values_hash': marks['values']},
               actor=ACTOR, effective_time=CLOSE)
    # After the close, and backdated behind it: the administrator republishes the same key, and the
    # close is restated by a second close event rather than corrected in place.
    log.append('fixing_observed',
               {'index': 'EURUSD-ECB', 'date': '2026-08-26', 'source': 'ECB', 'value': 1.0857},
               actor='subject-administrator', effective_time=WED)
    log.append('official_close_declared', {'market': 'official', 'values_hash': marks['restated']},
               actor=ACTOR, effective_time=CLOSE)
    log.append('retention_declared', {'blob_class': 'tape', 'policy_blob': marks['policy']},
               actor=ACTOR)
    log.append('rehash_declared', {'algorithm': 'sha256'}, actor=ACTOR)
    log.append('snapshot_registered', {'blob': marks['snapshot']}, actor=ACTOR, book=BOOK)
    log.append('break_glass_used', {'reason': 'the grant declaration stranded the last admin'},
               actor=ACTOR)
    return home, log, marks


def test_the_synthetic_book_reads_as_of_and_as_at_across_a_restatement(tmp_path):
    """Bitemporality on a book where the two orders actually DISAGREE.

    As-at is LSN order - what the record knew, and when it knew it. As-of is (effective_time, LSN) -
    what was true, and when it was true. A gate whose events all share one effective time exercises
    only the degenerate half of that distinction; here a Monday booking recorded after a Wednesday
    one sorts before it as of and after it as at, and a backdated amendment does the same.

    The restatement is the other half. A backdated observation arriving after an official close
    supersedes the close with a NEW close, so the book reads the first values hash as of everything
    up to that point and the restated one afterwards - both correct, neither an edit - and the
    republished fixing supersedes under its own key by exactly the same rule.
    """
    home, log, marks = synthetic_book(tmp_path)
    assert verify_home(home) == {'mode': 'entitled', 'events': 21, 'checkpoints_verified': 1,
                                 'head_lsn': 21, 'head_hash': log.head()[1]}
    assert set(frame['event_type'] for frame in log.frames()) == set(FACT_TYPES), \
        'every fact in the closed vocabulary reaches the writer in this fixture'

    pairs = [(frame, log.open_body(frame)) for frame in log.frames(start_lsn=5)]
    as_at = [frame['lsn'] for frame, _ in pairs]
    as_of = [frame['lsn'] for frame, _ in sorted(pairs, key=lambda pair: as_of_key(pair[0]))]
    assert as_at == list(range(5, 22)), 'as-at is LSN order and needs no key'
    assert as_of != as_at, 'a book holding a late booking must read differently as of'
    assert as_of.index(6) < as_of.index(5), 'the Monday fill is true before the Wednesday one'
    assert as_of.index(7) < as_of.index(5), 'and so is the amendment backdated behind it'

    # The fixing the administrator republished wins under its own (index, date, source) key, and
    # the print it replaced is still there to be read at the position it was recorded at.
    fixings = superseded(pairs, 'fixing_observed',
                         lambda body: (body['index'], body['date'], body['source']))
    assert fixings[('EURUSD-ECB', '2026-08-26', 'ECB')]['value'] == 1.0857
    assert log.open_body(log.frame_at(14))['value'] == 1.0851, 'nothing was edited'

    # The close, read on both sides of the restatement: as at LSN 16 the official close is the
    # first values hash; over the whole book it is the restated one.
    before = [pair for pair in pairs if pair[0]['lsn'] <= 16]
    market = lambda body: body['market']
    assert superseded(before, 'official_close_declared', market)['official']['values_hash'] \
        == marks['values']
    assert superseded(pairs, 'official_close_declared', market)['official']['values_hash'] \
        == marks['restated']

    # The rejected approval: one plan hash, two decisions, two seats - the four-eyes rule as data.
    decisions = [(frame['event_type'], frame['actor']) for frame, body in pairs
                 if body.get('plan_hash') == marks['plan']]
    assert decisions == [('approval', ACTOR), ('rejection', 'subject-desk-two')]
    log.close()


def test_a_checkpoint_verifies_and_a_forged_signature_is_named(tmp_path):
    """Authenticity, not merely integrity. Signatures are checked against the verifying key read
    out of the genesis policy BLOB - the assertion a replica makes - so a checkpoint whose body was
    hand-forged through the ordinary writer, which validates its shape and knows nothing of its
    meaning, is refused by name at its own LSN."""
    home = seeded(tmp_path)
    log = SpineLog(home)
    signed = write_checkpoint(log)
    body = log.open_body(log.frame_at(signed['lsn']))
    assert body['lsn'] == 7 and body['event_hash'] == log.frame_at(7)['event_hash']
    assert verify_home(home)['checkpoints_verified'] == 2

    head_lsn, head_hash = log.head()
    forged = log.append('checkpoint',
                        {'lsn': head_lsn, 'event_hash': head_hash, 'signature': '00' * 64},
                        actor=ACTOR)
    with pytest.raises(CheckpointInvalid) as refusal:
        verify_home(home)
    assert 'LSN {}'.format(forged['lsn']) in str(refusal.value), refusal.value
    # The chain itself is intact - the forgery is a body, and only the signature says so.
    assert verify_home(home, entitled=False)['head_lsn'] == forged['lsn']

    # And the other half of a checkpoint's claim, on its own home: a PERFECTLY signed pair - this
    # home's own key over it - that names a position this log never stood at. A checkpoint from
    # another history verifies as a signature and is still refused.
    elsewhere = seeded(tmp_path, 'elsewhere')
    other = SpineLog(elsewhere)
    claim = {'event_hash': hashlib.sha256(b'a head this log never reached').hexdigest(), 'lsn': 3}
    other.append('checkpoint',
                 dict(claim, signature=Keys(elsewhere).sign_checkpoint(canonical_bytes(claim))),
                 actor=ACTOR)
    with pytest.raises(CheckpointInvalid) as refusal:
        verify_home(elsewhere)
    assert 'LSN 8' in str(refusal.value) and 'lsn 3' in str(refusal.value), refusal.value


def test_the_verifying_key_comes_out_of_the_log_and_not_out_of_the_keys_directory(tmp_path):
    """Where the key is READ from is the whole replica claim, so it is asserted from both sides.

    Delete the published blob and authenticity can no longer be asserted at all - the refusal names
    the blob rather than reporting a count. Delete `keys/checkpoint_verify.key` instead and nothing
    changes, because that file was never what verification read: a replica has the log and does not
    have `keys/`.

    Between them these kill two mutations of the key's provenance - reading `keys/` instead of the
    blob, and letting a checkpoint body nominate its own - which otherwise survive every other gate
    in this file.
    """
    home = seeded(tmp_path, 'published')
    log = SpineLog(home)
    blob = log.open_body(log.frame_at(3))['blob']
    assert verify_home(home)['checkpoints_verified'] == 1

    without_the_key_file = tmp_path / 'no_verify_key_file'
    shutil.copytree(str(home), str(without_the_key_file))
    os.unlink(str(without_the_key_file / 'keys' / 'checkpoint_verify.key'))
    assert verify_home(without_the_key_file)['checkpoints_verified'] == 1, \
        'the file is the writer\'s convenience; the log is what a verifier reads'

    without_the_blob = tmp_path / 'no_published_blob'
    shutil.copytree(str(home), str(without_the_blob))
    os.unlink(str(blob_path(without_the_blob, blob)))
    with pytest.raises(CheckpointInvalid) as refusal:
        verify_home(without_the_blob)
    assert blob in str(refusal.value), refusal.value


def test_a_checkpoint_may_not_nominate_the_key_it_is_checked_against(tmp_path):
    """An insider holding the class key, re-sealing a checkpoint around a foreign verifying key and
    a signature under it, over a position this log genuinely reached.

    Everything an entitled replica checks about the frame passes: the seal opens under the frame's
    own unchanged envelope, the interior binding is recomputed honestly over the substituted body,
    the chain hash is recomputed to match, and the blind key is absent - the posture the design
    declares, since the blind key never leaves the hub - so the tag check does not run either. The
    only thing standing between that forgery and a green report is that the verifier resolves the
    key from the LOG's published blob and never from the body in front of it.
    """
    home = seeded(tmp_path, 'nominated')
    log = SpineLog(home)
    signed = write_checkpoint(log)
    log.close()
    os.unlink(str(home / 'keys' / 'blind.key'))

    frame = [f for f in read_frames(home) if f['lsn'] == signed['lsn']][0]
    private, public = ed25519_pair()
    body = dict(SpineLog(home).open_body(frame))
    body['signature'] = sign(private, body['lsn'], body['event_hash'])
    body['verify_key'] = public.hex()
    interior = canonical_bytes({
        'content_hash': hashlib.sha256(canonical_bytes(semantic_of(frame, body))).hexdigest(),
        'payload': body})
    sealed = Keys(home).seal(interior, aad_of(frame))

    def reforge(target):
        target['body'] = base64.b64encode(sealed).decode('ascii')
        target['event_hash'] = chain_of(target)

    doctor(home, signed['lsn'], reforge)

    assert verify_home(home, entitled=False)['head_lsn'] == signed['lsn'], \
        'the chain is consistent and the body opens - this one is the signature\'s to catch'
    with pytest.raises(CheckpointInvalid) as refusal:
        verify_home(home)
    assert 'LSN {}'.format(signed['lsn']) in str(refusal.value), refusal.value
    assert 'does not verify' in str(refusal.value), refusal.value


def test_the_verifying_key_is_pinned_by_lsn_so_a_rotation_reads_both_sides(tmp_path):
    """Key rotation is itself a logged event, which is only true if the verifying key is a LADDER:
    every checkpoint checked under the key in force AT ITS OWN LSN. Genesis's checkpoint verifies
    under genesis's key forever, and the checkpoint after the rotation verifies under the new one.

    Kills the mutant that reads the last declaration in the log. Under it this rotation - an
    ordinary append, through the public vocabulary, altering nothing - retro-invalidates LSN 4 and
    the home never verifies entitled again, with no repair path anywhere in the package.
    """
    home = seeded(tmp_path, 'rotation')
    log = SpineLog(home)
    assert verify_home(home)['checkpoints_verified'] == 1

    private, public = ed25519_pair()
    blob = log.store.put(public)
    rotation = log.append('policy_declared', {'policy': 'checkpoint_verifying_key', 'blob': blob},
                          actor=ACTOR, blob_refs=(blob,))
    head_lsn, head_hash = log.head()
    turned = log.append('checkpoint',
                        {'lsn': head_lsn, 'event_hash': head_hash,
                         'signature': sign(private, head_lsn, head_hash)}, actor=ACTOR)
    assert (rotation['lsn'], turned['lsn']) == (8, 9)

    report = verify_home(home)
    assert report['checkpoints_verified'] == 2, 'both sides of the rotation verify, on their own key'
    assert report['head_lsn'] == 9

    # A declaration naming no blob names no key: it is not a rung of the ladder, and it does not
    # cost the home the rungs already on it either.
    log.append('policy_declared', {'policy': 'checkpoint_verifying_key'}, actor=ACTOR)
    assert verify_home(home)['checkpoints_verified'] == 2
    log.close()


# --------------------------------------------------------------------------------------------
# The torn tail, and the restore.

def test_a_torn_final_line_is_truncated_and_the_next_append_chains_onto_the_head(tmp_path):
    """The one place bytes are removed, and it is not a repair: a final line that will not parse
    was interrupted mid-write and was never fsynced, so nothing ever chained onto it. Everything
    durable survives, and the next append continues from the head that was already there."""
    home = seeded(tmp_path)
    head_lsn, head_hash = SpineLog(home).head()
    segment = segments(home)[-1]
    intact = segment.stat().st_size
    with segment.open('ab') as handle:
        handle.write(b'{"lsn":8,"actor":"subject-desk-one","body":"AAAA')

    reopened = SpineLog(home)
    assert segment.stat().st_size == intact, 'the torn bytes are gone and nothing else is'
    assert reopened.head() == (head_lsn, head_hash)
    assert verify_home(home)['head_lsn'] == head_lsn

    landed = reopened.append('fill', fill('EXEC-AFTER'), actor=ACTOR, book=BOOK)
    assert landed['lsn'] == head_lsn + 1 and landed['prev_hash'] == head_hash
    assert verify_home(home)['head_lsn'] == head_lsn + 1
    assert len(read_frames(home)) == head_lsn + 1
    reopened.close()


def test_a_terminated_line_that_will_not_parse_is_a_broken_chain_not_a_torn_tail(tmp_path):
    """The negative case of the one repair, and the reason it is drawn where it is.

    A write puts the terminating newline down LAST, so a partial one always loses that byte first: a
    line carrying its newline is a line the writer finished and fsynced. Garbage inside one is
    therefore a DURABLE line that was altered, and truncating it would delete the evidence of the
    tampering, roll the head backwards past checkpoints, and report green - during a read-only
    verification, at that, since opening the log is what would do the deleting.

    So it is `ChainBroken` naming the position, and the bytes are exactly where they were
    afterwards. Kills the mutant that truncates on `end >= len(data)` rather than on the missing
    terminator, under which this home silently loses LSN 7 and verifies clean.
    """
    home = seeded(tmp_path, 'terminated')
    segment = segments(home)[-1]
    lines = [raw for raw in segment.read_bytes().split(b'\n') if raw.strip()]
    doctored = b'\n'.join(lines[:-1] + [b'{ THIS IS NOT JSON }']) + b'\n'
    segment.write_bytes(doctored)

    with pytest.raises(ChainBroken) as refusal:
        SpineLog(home)
    assert 'LSN 7' in str(refusal.value), refusal.value
    with pytest.raises(ChainBroken):
        verify_home(home, entitled=False)
    with pytest.raises(ChainBroken):
        verify_home(home)
    assert segment.read_bytes() == doctored, \
        'a verification that deletes the evidence it was asked to look at is not a verification'


def test_a_surplus_frame_field_is_refused_in_both_modes(tmp_path):
    """The envelope is closed the way a body is. The nine-field AAD, the event hash and the interior
    binding cover exactly the twelve fields a frame has, so a thirteenth would ride into the record
    covered by no hash, no seal and no signature - and be read by every projector downstream. The
    field planted here is `display_name`, which is precisely the erasable attribute the design
    confines to a side table outside the log.

    Kills the mutant that checks only that the twelve fields are PRESENT: under it a frame carrying
    three extra fields verifies green in both modes.
    """
    home = seeded(tmp_path, 'surplus')
    doctor(home, 6, lambda frame: frame.__setitem__('display_name', 'Mallory'))

    with pytest.raises(ChainBroken) as refusal:
        SpineLog(home)
    assert 'LSN 6' in str(refusal.value) and 'display_name' in str(refusal.value), refusal.value
    for mode in (True, False):
        with pytest.raises(ChainBroken) as refusal:
            verify_home(home, entitled=mode)
        assert 'LSN 6' in str(refusal.value) and 'display_name' in str(refusal.value), refusal.value


def test_a_second_writer_is_refused_rather_than_left_to_corrupt_the_home(tmp_path):
    """One deployment, one log, one writer - claimed, not merely declared.

    Two handles on one home both believe the head is where they last read it, so both assign the
    same next LSN, and the result is a home holding two lines at one position - unverifiable
    forever, because nothing in this package may edit or remove either of them. The claim is taken
    at the first append (reading and verifying never take it, or a replica would be locked out of
    its own log) and the head is re-read under it, so the second writer is a named refusal and the
    book is exactly as it was.
    """
    home = seeded(tmp_path, 'writers', clips=())
    first, second = SpineLog(home), SpineLog(home)
    assert first.head() == second.head() == (4, read_frames(home)[3]['event_hash'])

    landed = first.append('fill', fill('EXEC-ONE'), actor=ACTOR, book=BOOK, effective_time=WHEN)
    with pytest.raises(WriterBusy) as refusal:
        second.append('fill', fill('EXEC-TWO'), actor=ACTOR, book=BOOK, effective_time=WHEN)
    assert '.writer.lock' in str(refusal.value), refusal.value

    assert [frame['lsn'] for frame in read_frames(home)] == [1, 2, 3, 4, 5]
    assert verify_home(home)['head_lsn'] == landed['lsn'] == 5

    # and the claim goes with the writer that held it - the second handle then re-reads the head
    # under its own claim rather than trusting the one it opened with
    first.close()
    assert second.append('fill', fill('EXEC-TWO'), actor=ACTOR, book=BOOK,
                         effective_time=WHEN)['lsn'] == 6
    assert verify_home(home)['head_lsn'] == 6
    second.close()


def test_the_chain_runs_across_a_segment_boundary(tmp_path):
    """Segments are bookkeeping - a 64 MiB roll, nothing more - and the chain does not know they
    exist. Rather than write 64 MiB to prove it, the split is done as data: the tail of the first
    segment is moved into a second one by hand, and everything must read, verify and append exactly
    as before across the seam."""
    home = seeded(tmp_path)
    first = segments(home)[0]
    lines = [raw for raw in first.read_bytes().split(b'\n') if raw.strip()]
    first.write_bytes(b'\n'.join(lines[:4]) + b'\n')
    (home / 'log' / 'segment-00000002.jsonl').write_bytes(b'\n'.join(lines[4:]) + b'\n')

    log = SpineLog(home)
    assert len(segments(home)) == 2
    assert [frame['lsn'] for frame in log.frames()] == [1, 2, 3, 4, 5, 6, 7]
    assert [frame['lsn'] for frame in log.frames(start_lsn=5)] == [5, 6, 7]
    assert [frame['lsn'] for frame in log.frames(start_lsn=3, end_lsn=5)] == [3, 4, 5]
    assert log.frame_at(1)['lsn'] == 1 and log.frame_at(7)['lsn'] == 7
    assert verify_home(home)['head_lsn'] == 7

    landed = log.append('fill', fill('EXEC-SEAM'), actor=ACTOR, book=BOOK)
    assert landed['lsn'] == 8
    assert verify_home(home)['head_lsn'] == 8
    assert len(segments(home)) == 2, 'a new line joins the last segment; the roll is by size only'


def test_a_copied_home_verifies_extends_and_still_catches_a_tamper(tmp_path):
    """The disaster posture, tested rather than asserted: three directories copied to a clean
    place, and everything the original could do the copy does - verify from genesis, catch a
    doctored line, and go on writing."""
    home = seeded(tmp_path)
    restored = tmp_path / 'restored'
    restored.mkdir()
    for part in ('log', 'blobs', 'keys'):
        shutil.copytree(str(home / part), str(restored / part))

    assert verify_home(restored) == verify_home(home)
    assert verify_home(restored, entitled=False)['checkpoints_verified'] == 'not assessed'

    log = SpineLog(restored)
    landed = log.append('fill', fill('EXEC-RESTORED'), actor=ACTOR, book=BOOK)
    assert landed['lsn'] == 8
    assert verify_home(restored)['head_lsn'] == 8
    assert verify_home(home)['head_lsn'] == 7, 'the copy went its own way, as a fork of files does'
    log.close()

    # All three detections again, on three copies of the copy: a restored home is not a weaker
    # witness than the one it came from.
    tampers = (
        ('body', lambda frame: frame.__setitem__('body', flip_base64(frame['body'])), True),
        ('actor', lambda frame: frame.__setitem__('actor', 'subject-someone-else'), False),
        ('clock', lambda frame: frame.__setitem__(
            'record_time', '2020-01-01T00:00:00.000000Z'), True),
    )
    for name, mutate, unentitled_sees_it in tampers:
        copy = tmp_path / ('restored_' + name)
        shutil.copytree(str(restored), str(copy))
        doctor(copy, 6, mutate)
        for mode in ((True, False) if unentitled_sees_it else (True,)):
            with pytest.raises(ChainBroken) as refusal:
                verify_home(copy, entitled=mode)
            assert 'LSN 6' in str(refusal.value), refusal.value
        if not unentitled_sees_it:
            assert verify_home(copy, entitled=False)['head_lsn'] == 8, \
                'the envelope edit is the seal\'s to catch, there as here'


def test_a_replica_holding_only_the_log_and_the_blobs_verifies_its_chain(tmp_path):
    """The brief's replica, in the two directories the brief names: the full log and the blob
    store, ciphertext where unentitled, no key of any kind.

    That posture is the whole reason the chain is taken over the CIPHERTEXT, and it has to be
    reachable to be true - a log that demanded a `keys/` directory before it would open would make
    the replica unrepresentable, and the remedy the refusal offered (`init`) would fork the record
    rather than fix it. So the keys' absence surfaces where a key is USED, by the file's own name,
    and everything a keyless holder can check it still checks.
    """
    home = seeded(tmp_path, 'hub')
    replica = tmp_path / 'replica'
    replica.mkdir()
    for part in ('log', 'blobs'):
        shutil.copytree(str(home / part), str(replica / part))
    assert not (replica / 'keys').exists()

    report = verify_home(replica, entitled=False)
    assert report['events'] == 7 and report['checkpoints_verified'] == 'not assessed'
    assert report['head_hash'] == verify_home(home, entitled=False)['head_hash']
    assert SpineLog(replica).head() == SpineLog(home).head()

    # the entitled read refuses by the KEY's name, not by the home's: this is an unentitled holder,
    # not a broken directory
    with pytest.raises(SealedBodyUnreadable) as refusal:
        verify_home(replica)
    assert 'class_firm.key' in str(refusal.value), refusal.value

    # and a home that really is not one still says so, with a remedy that fits the case
    with pytest.raises(HomeMissing) as refusal:
        SpineLog(tmp_path / 'nothing-here')
    assert 'log/' in str(refusal.value), refusal.value

    # what a keyless replica can catch, it catches: `record_time` is inside the event hash
    doctor(replica, 6, lambda frame: frame.__setitem__(
        'record_time', '2020-01-01T00:00:00.000000Z'))
    with pytest.raises(ChainBroken) as refusal:
        verify_home(replica, entitled=False)
    assert 'LSN 6' in str(refusal.value), refusal.value
