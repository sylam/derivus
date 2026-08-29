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

"""Key custody on real homes - and the one assertion that says a wrapped key is an entitlement.

The load-bearing gate in this file is the replica walk. A directory holding `log/` and `blobs/` and
an EMPTY `keys/` verifies its chain over ciphertext and cannot open one body; `materialize` puts the
class key into it out of that seat's own wrap; and the same directory then verifies ENTITLED, reads
its fills back and checks every checkpoint signature. Anything less than that would leave "the class
key is wrapped per entitled subject" as a sentence about a JSON blob rather than a property of the
system - the wrap has to be the thing that changes what a holder can do.

The escrow walk is the same claim run backwards through the increment-1 shred posture: on a COPY of
the home, `keys/class_firm.key` is deleted (which is what crypto-shredding is), the chain stays
green over ciphertext, the custodian's private half recovers the class key bit for bit, and the copy
comes back to entitled-green. Recovery is a key that was always addressed to escrow, not a back
door: escrow rides an ordinary `key_wrapped` row under a reserved subject, so there is one mechanism
here, read twice.

The AAD binding is gated from both sides, because it is what makes a wrap ADDRESSED rather than
merely encrypted. The wrong seat's private key does not open a wrap, and - the sharper half - the
RIGHT seat's private key does not open it either when the subject it is opened as is somebody else's.
That is the subject swap the contract asks for: a hub that files one subject's wrap on another's row
produces a refusal instead of a quiet mis-delivery.

Faults are the house's kind throughout: doctored bytes on disk (a flipped nibble in a stored wrap, a
blob from another history copied into a replica's store), never a patched function. Every key in
this file is a real X25519 key minted by `cryptography` inside the gate, and every event goes in
through the ordinary writer under a real capabilities document.
"""
import hashlib
import json
import os
import shutil
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding, NoEncryption, PrivateFormat, PublicFormat)

from derivus_spine import (
    CustodyRefusal, SealedBodyUnreadable, SpineLog, canonical_bytes, init_home, verify_home)
from derivus_spine import cli
from derivus_spine.capability import CAPABILITIES_POLICY, canonical_document
from derivus_spine.custody import (
    ALGORITHM, CLASS_KEY_FILE, ESCROW_POLICY, ESCROW_SUBJECT, SEAT_ALGORITHM, WRAP_FIELDS,
    X25519_BYTES, declare_escrow, enroll, materialize, read_class_key, recover_escrow, rewrap,
    seat_key_path, unwrap_key, wrap_drift, wrap_key)
from derivus_spine.seal import FIRM, KEY_BYTES, NONCE_BYTES
from derivus_spine.vocabulary import ADMIN, BOOK, FIRM_CLASS

MINT = 'subject-deployment'
DESK = 'subject-desk-one'
MARKER = 'subject-marks'
STRANGER = 'subject-nobody'
GHOST = 'subject-ghost'
BOOK_ONE = 'FX-VANILLA'
HASH_A = 'a' * 64
ANY = '*'


# --------------------------------------------------------------------------------------------
# Homes, documents and seats.

def document(grants=(), read=()):
    """A capabilities document out of `(subject, verb, book)` and `(subject, class)` rows."""
    return {'grants': [{'subject': s, 'verb': v, 'book': b} for s, v, b in grants],
            'read': [{'subject': s, 'class': c} for s, c in read]}


def declare(log, actor, doc):
    """Declare `doc` the way the writer's law requires: canonical bytes in the store, then the
    ordinary `policy_declared` naming the hash."""
    blob = log.store.put(canonical_document(doc))
    return log.append('policy_declared', {'policy': CAPABILITIES_POLICY, 'blob': blob},
                      actor=actor, blob_refs=(blob,))


def minted(tmp_path, name='hub'):
    """A home mid-genesis with the writer open on it."""
    home = tmp_path / name
    init_home(home, MINT)
    return home, SpineLog(home)


def fill(reference):
    return {'instrument': HASH_A, 'quantity': 2500000.0, 'netting_set': 'CSA-0007',
            'counterparty': 'LEI-5493001KJTIIGC8Y1R12', 'execution_reference': reference}


def seat_private(home, subject):
    """The seat's private key as bytes - what the seat itself carries, and what the lost laptop
    lost."""
    return seat_key_path(home, subject).read_bytes()


def custodian():
    """An escrow keypair, minted here: the public half is declared into the log, the private half
    is the custodian's and never touches the home."""
    private = X25519PrivateKey.generate()
    return (private.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption()),
            private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw))


def facts(log, event_type):
    """Every `event_type` fact on the log as `(lsn, body)`, read off the platter and decrypted."""
    return [(frame['lsn'], log.open_body(frame)) for frame in log.frames()
            if frame['event_type'] == event_type]


def seated(tmp_path, name='hub', read=(DESK,), enrolled=(DESK,)):
    """A hub with seats enrolled, a document in force and one fill on the book.

    Enrollment happens BEFORE the declaration, so the seats are minted while the home is still the
    single-user instrument it starts as, and everything after the declaration is under enforcement -
    which is the order a deployment actually runs in.
    """
    home, log = minted(tmp_path, name)
    for subject in enrolled:
        enroll(log, subject, actor=MINT)
    declare(log, MINT, document(
        grants=((MINT, ADMIN, ANY), (DESK, BOOK, BOOK_ONE)),
        read=tuple((subject, FIRM_CLASS) for subject in read)))
    log.append('fill', fill('EXEC-1'), actor=DESK, book=BOOK_ONE)
    return home, log


def replica_of(home, into):
    """The brief's replica: the full log and the blob store, an empty `keys/`, no key of any kind."""
    into.mkdir(parents=True)
    for part in ('log', 'blobs'):
        shutil.copytree(str(home / part), str(into / part))
    (into / 'keys').mkdir()
    return into


# --------------------------------------------------------------------------------------------
# Enrollment, wrapping, and the round trip.

def test_a_seat_is_enrolled_wrapped_and_opened_back_to_the_class_key_bit_for_bit(tmp_path):
    """The whole mechanism in one pass, and every step read back off the log rather than off the
    call that made it.

    A seat exists because `seat_enrolled` says its public key is at a hash; the key reached the
    subject because `key_wrapped` says the wrap is at a hash. So the assertions go through the
    record: the two facts are fetched from the platter, decrypted, and the blobs they name are what
    the round trip is run over. What comes out is the home's own class key, bit for bit - not a key
    that works, THE key.
    """
    home, log = minted(tmp_path)
    minting = enroll(log, DESK, actor=MINT)
    assert minting['event_type'] == 'seat_enrolled' and minting['coalesced'] is False
    assert minting['subject'] == DESK and minting['algorithm'] == SEAT_ALGORITHM

    declare(log, MINT, document(grants=((MINT, ADMIN, ANY),), read=((DESK, FIRM_CLASS),)))
    report = rewrap(log, actor=MINT)
    assert report['events'] == 1 and report['unenrolled'] == []
    assert [row['subject'] for row in report['wrapped']] == [DESK]

    # the two facts, verbatim: the vocabulary's shape, the hashes the bodies name, nothing else
    (_, enrolled), = facts(log, 'seat_enrolled')
    assert enrolled == {'subject': DESK, 'algorithm': SEAT_ALGORITHM,
                        'public_key': minting['public_key']}
    (_, wrapped), = facts(log, 'key_wrapped')
    assert wrapped == {'class': FIRM_CLASS, 'subject': DESK,
                       'wrap': report['wrapped'][0]['wrap']}

    # the published public key is the raw curve point, and the private half never went near the log
    assert len(log.store.get(enrolled['public_key'])) == X25519_BYTES
    opened = unwrap_key(log.store.get(wrapped['wrap']), seat_private(home, DESK), DESK)
    assert opened == read_class_key(home) == (home / 'keys' / FIRM).read_bytes()
    assert len(opened) == KEY_BYTES

    log.close()
    assert verify_home(home)['head_lsn'] == 7, 'custody events are ordinary chained facts'


def test_the_seat_key_file_names_the_subject_by_hash_and_never_in_the_clear(tmp_path):
    """A path is data too. The record is pseudonymous by rule, so a directory listing of a hub must
    not be a membership roster - the filename is sixteen hex characters of the subject's SHA-256,
    which is stable, unique at desk scale, and says nothing."""
    home, log = minted(tmp_path)
    enroll(log, DESK, actor=MINT)
    enroll(log, MARKER, actor=MINT)

    seats = sorted(path.name for path in (home / 'keys' / 'seats').iterdir())
    assert len(seats) == 2 and all(len(name) == len('0' * 16) + len('.key') for name in seats)
    assert DESK not in str(seat_key_path(home, DESK)), 'the path carries the raw identifier'
    assert seat_key_path(home, DESK).name == '{}.key'.format(
        hashlib.sha256(DESK.encode('utf-8')).hexdigest()[:16])
    assert len(seat_private(home, DESK)) == X25519_BYTES
    assert seat_private(home, DESK) != seat_private(home, MARKER)


def test_a_key_file_is_written_as_bytes_and_never_translated(tmp_path):
    """The platform trap, gated rather than remembered.

    A key is 32 bytes of noise, so about one key in eight carries a `0x0A`, and a text-mode handle
    on Windows writes that byte as two. The file is then a byte longer than a key and every wrap
    made from it opens onto nothing - which fails one run in eight and looks like bad luck. So seats
    are minted until one of them actually carries the byte, and the file is measured: what is
    asserted is the case that breaks, not the average one.
    """
    home, log = minted(tmp_path)
    carried = None
    for index in range(200):
        subject = 'subject-seat-{}'.format(index)
        enroll(log, subject, actor=MINT)
        material = seat_private(home, subject)
        assert len(material) == X25519_BYTES, subject
        if b'\n' in material:
            carried = subject
            break
    assert carried is not None, 'no minted seat key carried a newline in 200 tries'
    assert seat_key_path(home, carried).stat().st_size == X25519_BYTES


def test_the_wrap_blob_is_the_four_field_object_and_a_fresh_one_every_time(tmp_path):
    """The wire format, pinned. A wrap is canonical JSON carrying the construction's NAME, the
    ephemeral public key, the nonce and the ciphertext - four fields and no fifth, because a fifth
    would be bytes in the middle of the recovery path that the AAD does not cover.

    The freshness half is why the ephemeral is there at all: the same key wrapped to the same seat
    twice is two different blobs that both open, so two stored wraps never testify that they carry
    the same secret.
    """
    key = os.urandom(KEY_BYTES)
    private = X25519PrivateKey.generate()
    public = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    secret = private.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())

    raw = wrap_key(key, public, DESK)
    blob = json.loads(raw.decode('utf-8'))
    assert sorted(blob) == list(WRAP_FIELDS) == ['algorithm', 'ephemeral_public', 'nonce', 'wrapped']
    assert blob['algorithm'] == ALGORITHM == 'x25519-hkdf-sha256-aesgcm-v1'
    assert len(bytes.fromhex(blob['ephemeral_public'])) == X25519_BYTES
    assert len(bytes.fromhex(blob['nonce'])) == NONCE_BYTES
    assert canonical_bytes(blob) == raw, 'the blob is not stored in its canonical spelling'

    again = wrap_key(key, public, DESK)
    assert again != raw, 'two wraps of one key look identical'
    assert unwrap_key(raw, secret, DESK) == unwrap_key(again, secret, DESK) == key


# --------------------------------------------------------------------------------------------
# The wrap is ADDRESSED: the AAD binding, from both sides.

def test_another_seats_private_key_does_not_open_this_seats_wrap(tmp_path):
    """The obvious half of the binding, and the one a reader expects: a wrap is encrypted to one
    public key, so the seat next to it cannot open it however entitled that seat is in the
    document."""
    home, log = seated(tmp_path, read=(DESK, MARKER), enrolled=(DESK, MARKER))
    rewrap(log, actor=MINT)
    wraps = dict((body['subject'], body['wrap']) for _, body in facts(log, 'key_wrapped'))

    with pytest.raises(CustodyRefusal) as refusal:
        unwrap_key(log.store.get(wraps[DESK]), seat_private(home, MARKER), DESK)
    assert DESK in str(refusal.value) and 'rewrap' in str(refusal.value)

    # and each seat does open its own, so the refusal above is about the recipient and nothing else
    for subject in (DESK, MARKER):
        assert unwrap_key(log.store.get(wraps[subject]), seat_private(home, subject),
                          subject) == read_class_key(home)


def test_a_wrap_read_as_another_subject_refuses_even_under_the_right_private_key(tmp_path):
    """The sharp half of the binding, and the contract's subject swap.

    GCM's additional data binds the `{class, subject}` pair the `key_wrapped` row carries in the
    clear, so a wrap is not "the class key encrypted to a public key" - it is the class key
    encrypted to THIS subject for THIS class. Swap two subjects' wrap blobs onto each other's rows
    and the ciphertext and the record stop agreeing about who the recipient is, so nothing opens:
    the seat's own private key is not enough, which is exactly what makes a mis-filed wrap a refusal
    rather than a quiet mis-delivery.
    """
    home, log = seated(tmp_path, read=(DESK, MARKER), enrolled=(DESK, MARKER))
    rewrap(log, actor=MINT)
    wraps = dict((body['subject'], body['wrap']) for _, body in facts(log, 'key_wrapped'))

    # the swap: each blob is offered on the other's row, with that row's own private key
    for holder, filed_as in ((DESK, MARKER), (MARKER, DESK)):
        with pytest.raises(CustodyRefusal) as refusal:
            unwrap_key(log.store.get(wraps[holder]), seat_private(home, holder), filed_as)
        assert filed_as in str(refusal.value), refusal.value

    # the class half of the pair binds too: one class key, but not under another class's name
    with pytest.raises(CustodyRefusal) as refusal:
        unwrap_key(log.store.get(wraps[DESK]), seat_private(home, DESK), DESK,
                   entitlement_class='desk-two')
    assert 'desk-two' in str(refusal.value), refusal.value


def test_a_doctored_wrap_blob_refuses_and_names_what_is_wrong_with_it(tmp_path):
    """Faults are doctored data, and a wrap blob is ordinary bytes on disk. Every way one can be
    edited lands on `CustodyRefusal` naming the field, because the alternative to a named refusal
    here is a caller that believes it recovered something."""
    key = os.urandom(KEY_BYTES)
    private = X25519PrivateKey.generate()
    public = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    secret = private.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
    blob = json.loads(wrap_key(key, public, DESK).decode('utf-8'))

    def refused(edited):
        with pytest.raises(CustodyRefusal) as refusal:
            unwrap_key(canonical_bytes(edited), secret, DESK)
        return str(refusal.value)

    flipped = dict(blob)
    flipped['wrapped'] = ('f' if blob['wrapped'][0] != 'f' else '0') + blob['wrapped'][1:]
    assert 'does not open' in refused(flipped)

    truncated = dict(blob, ephemeral_public=blob['ephemeral_public'][:-2])
    assert 'ephemeral_public' in refused(truncated)

    assert 'nonce' in refused(dict(blob, nonce='not hex at all'))
    assert 'aes-128' in refused(dict(blob, algorithm='aes-128'))

    # a fifth field is bytes no AAD covers sitting in the recovery path: the shape is closed, and
    # the refusal names both what a wrap is and what this one carried instead
    surplus = refused(dict(blob, note='an unauthenticated field'))
    assert ', '.join(WRAP_FIELDS) in surplus and 'note' in surplus

    with pytest.raises(CustodyRefusal) as refusal:
        unwrap_key(b'a seat public key, not a wrap', secret, DESK)
    assert 'JSON' in str(refusal.value), refusal.value


# --------------------------------------------------------------------------------------------
# The replica: chain-only, then entitled, off one wrap.

def test_a_replica_goes_from_chain_only_to_entitled_green_off_one_wrap(tmp_path):
    """The assertion that makes a wrapped key an ENTITLEMENT rather than decoration.

    A replica arrives holding `log/`, `blobs/` and an empty `keys/`. It re-derives the whole chain
    over ciphertext - which is the posture the two-hash scheme exists for - and it cannot open one
    body, so an entitled verification refuses by the KEY's name rather than the home's. Then the
    seat's own private key materializes the class key out of the wrap the hub filed for it, and the
    same directory verifies ENTITLED: every plaintext binding, every checkpoint signature, and the
    desk's fill readable off the platter.

    Which wrap is this subject's is answered by the AAD, and it has to be: the `key_wrapped` row
    naming the wrap has a sealed body, sealed under the very key being recovered. That is the
    bootstrap this design has to survive, and surviving it is what the walk below asserts.
    """
    home, log = seated(tmp_path)
    rewrap(log, actor=MINT)
    log.close()
    replica = replica_of(home, tmp_path / 'replica')
    assert not any((replica / 'keys').iterdir()), 'the replica arrived holding a key'

    chain_only = verify_home(replica, entitled=False)
    assert chain_only['checkpoints_verified'] == 'not assessed'
    assert chain_only['head_hash'] == verify_home(home, entitled=False)['head_hash']
    with pytest.raises(SealedBodyUnreadable) as refusal:
        verify_home(replica)
    assert FIRM in str(refusal.value), refusal.value

    recovered = materialize(replica, DESK, seat_private(home, DESK))
    assert recovered['subject'] == DESK and recovered['bodies_readable'] is True
    assert recovered['key_file'] == str(replica / 'keys' / FIRM)

    entitled = verify_home(replica)
    assert entitled['mode'] == 'entitled' and entitled['checkpoints_verified'] >= 1
    assert entitled['head_hash'] == chain_only['head_hash'], 'recovery moved the record'
    assert (replica / 'keys' / FIRM).read_bytes() == read_class_key(home)

    # the entitlement is the point: the replica reads the desk's own fill back
    holder = SpineLog(replica)
    booked = [holder.open_body(frame) for frame in holder.frames()
              if frame['event_type'] == 'fill']
    assert booked == [fill('EXEC-1')]


def test_materialize_refuses_to_write_over_a_class_key_that_is_already_there(tmp_path):
    """Overwriting would be crypto-shredding by accident: every body sealed under the key that is
    there becomes unreadable, in the one move that looks like it is granting access. So the second
    call refuses, names the file, and leaves the first recovery exactly as it was."""
    home, log = seated(tmp_path)
    rewrap(log, actor=MINT)
    log.close()
    replica = replica_of(home, tmp_path / 'replica')
    materialize(replica, DESK, seat_private(home, DESK))
    before = (replica / 'keys' / FIRM).read_bytes()

    with pytest.raises(CustodyRefusal) as refusal:
        materialize(replica, DESK, seat_private(home, DESK))
    assert FIRM in str(refusal.value) and 'strand' in str(refusal.value)
    assert (replica / 'keys' / FIRM).read_bytes() == before
    assert verify_home(replica)['mode'] == 'entitled'

    # and the hub itself is a home that already holds the key, so it refuses for the same reason
    with pytest.raises(CustodyRefusal):
        materialize(home, DESK, seat_private(home, DESK))


def test_a_replica_with_no_wrap_of_its_own_refuses_and_stays_unentitled(tmp_path):
    """A subject the document never admitted has no wrap in the store, so there is nothing to
    materialize and nothing is written: the refusal names the remedy (rewrap on the hub) rather than
    leaving a half-recovered home behind."""
    home, log = seated(tmp_path)
    rewrap(log, actor=MINT)
    log.close()
    replica = replica_of(home, tmp_path / 'replica')

    with pytest.raises(CustodyRefusal) as refusal:
        materialize(replica, MARKER, seat_private(home, DESK))
    assert MARKER in str(refusal.value) and 'rewrap' in str(refusal.value)
    assert not (replica / 'keys' / FIRM).exists()
    with pytest.raises(SealedBodyUnreadable):
        verify_home(replica)


def test_a_wrap_from_another_history_is_refused_rather_than_materialized(tmp_path):
    """Doctored data on disk: a wrap blob carrying somebody else's class key, copied into a
    replica's store and addressed to a seat that home never enrolled.

    It opens - the AAD is satisfied, because whoever wrote it addressed it correctly - and the key
    inside it does not seal this log. Materializing it would leave a home holding a key that opens
    nothing, so the guard is the record's own law read once more: never trust what you can
    re-derive. The refusal names both, and the home is left exactly as unentitled as it was.
    """
    home, log = seated(tmp_path)
    rewrap(log, actor=MINT)
    log.close()
    replica = replica_of(home, tmp_path / 'replica')

    ghost = X25519PrivateKey.generate()
    ghost_secret = ghost.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
    stranger_key = os.urandom(KEY_BYTES)
    SpineLog(replica).store.put(wrap_key(
        stranger_key, ghost.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw), GHOST))

    with pytest.raises(CustodyRefusal) as refusal:
        materialize(replica, GHOST, ghost_secret)
    assert 'different deployments' in str(refusal.value), refusal.value
    assert not (replica / 'keys' / FIRM).exists(), 'a refused recovery left a key behind'
    assert verify_home(replica, entitled=False)['head_lsn'] == SpineLog(home).head()[0]

    # two wraps that open for one subject and DISAGREE is a guess this verb does not make
    SpineLog(replica).store.put(wrap_key(
        os.urandom(KEY_BYTES), ghost.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw),
        GHOST))
    with pytest.raises(CustodyRefusal) as refusal:
        materialize(replica, GHOST, ghost_secret)
    assert 'two histories' in str(refusal.value), refusal.value


# --------------------------------------------------------------------------------------------
# Rewrap: who is missing, and the second call.

def test_rewrap_is_idempotent_and_a_second_call_appends_nothing(tmp_path):
    """Safe to put in a runbook after every grant, which is the whole reason the report and the fold
    agree on what MISSING means: a wrap is current when it was appended after the enrollment it is
    addressed to. So the second call finds every wrap younger than its seat and writes not one
    byte."""
    home, log = seated(tmp_path, read=(DESK, MARKER), enrolled=(DESK, MARKER))
    first = rewrap(log, actor=MINT)
    assert first['events'] == 2 and sorted(first['current']) == []
    head = log.head()

    second = rewrap(log, actor=MINT)
    assert second['events'] == 0 and second['wrapped'] == []
    assert sorted(second['current']) == sorted([DESK, MARKER])
    assert log.head() == head, 'an idempotent call moved the head'
    assert len(facts(log, 'key_wrapped')) == 2

    log.close()
    assert verify_home(home)['head_lsn'] == head[0]


def test_a_subject_with_read_and_no_seat_gets_no_wrap_and_the_report_names_it(tmp_path):
    """The document is allowed to describe somebody who has not sat down yet, so this is not an
    error - but a silent omission is how an entitlement becomes a rumour, so the report says the
    sentence out loud. Enrolling the seat afterwards and rewrapping issues exactly the wrap that was
    missing, and nothing else."""
    home, log = seated(tmp_path, read=(DESK, STRANGER), enrolled=(DESK,))

    report = rewrap(log, actor=MINT)
    assert report['unenrolled'] == [STRANGER]
    assert [row['subject'] for row in report['wrapped']] == [DESK]
    assert sorted(report['entitled']) == sorted([DESK, STRANGER])

    enroll(log, STRANGER, actor=MINT)
    later = rewrap(log, actor=MINT)
    assert later['unenrolled'] == [] and later['events'] == 1
    assert [row['subject'] for row in later['wrapped']] == [STRANGER]
    assert later['current'] == [DESK]
    assert unwrap_key(log.store.get(later['wrapped'][0]['wrap']),
                      seat_private(home, STRANGER), STRANGER) == read_class_key(home)


def test_a_key_wrapped_row_citing_something_that_is_not_a_wrap_is_not_a_wrap(tmp_path):
    """A fact says an entitlement was delivered; the store does not bear it out. Which wins.

    Doctored data on disk, in its most ordinary form: a `key_wrapped` row whose `wrap` names a real
    blob that is not a wrap. Referential closure is satisfied - the bytes ARE on the platter, fsynced
    before the event, exactly as the durability law demands - so the writer accepts it and
    `verify_home` stays green. A fold that counted the citation would then report the subject
    `current` forever, `rewrap` would emit nothing forever, and the seat would discover it when a
    body would not open, with the refusal recommending the very rewrap that does nothing. That is the
    silent omission this module exists to refuse, wearing the report's own word for delivered.

    So the fold OPENS what the row cites. The row does not count, the subject is named `unresolved`
    rather than `current`, the next rewrap issues a real wrap, and it opens to the class key.
    """
    home, log = seated(tmp_path)
    not_a_wrap = log.store.put(b'not a wrap at all - a note, a public key, an operator\'s mistake')
    log.append('key_wrapped', {'class': FIRM_CLASS, 'subject': DESK, 'wrap': not_a_wrap},
               actor=MINT, blob_refs=(not_a_wrap,))

    drift = wrap_drift(log)
    assert drift['unresolved'] == [DESK] and drift['current'] == []
    assert [subject for subject, _ in drift['pending']] == [DESK]

    report = rewrap(log, actor=MINT)
    assert report['events'] == 1 and report['current'] == [] and report['unresolved'] == [DESK]
    assert [row['subject'] for row in report['wrapped']] == [DESK]
    issued = log.store.get(report['wrapped'][0]['wrap'])
    assert unwrap_key(issued, seat_private(home, DESK), DESK) == read_class_key(home)

    # and the reissue is now the current one, so the runbook's second call is quiet again
    second = rewrap(log, actor=MINT)
    assert second['events'] == 0 and second['current'] == [DESK] and second['unresolved'] == []

    log.close()
    assert verify_home(home)['mode'] == 'entitled', 'the doctored row is still a chained fact'


def test_a_grant_names_the_wrap_drift_it_creates_rather_than_leaving_it_to_a_runbook(
        tmp_path, capsys):
    """"Rewrap on grant change" is a property of the system or it is a line in a runbook.

    The brief lists it beside per-seat keypairs and escrow, and nothing couples the two verbs: a
    document that adds a READ row leaves the store owing that subject a wrap the moment it lands, and
    if the operator forgets the second command the failure is SILENT - the subject is entitled, holds
    no key, and finds out when a body will not open. So `grant` computes the drift against the
    document it just put in force and says what is owed, while the operator is still at the keyboard.
    """
    home = tmp_path / 'hub'
    init_home(home, MINT)
    log = SpineLog(home)
    enroll(log, DESK, actor=MINT)
    log.close()

    policy = tmp_path / 'policy.json'
    policy.write_text(json.dumps(document(grants=((MINT, ADMIN, ANY),),
                                          read=((DESK, FIRM_CLASS), (STRANGER, FIRM_CLASS)))),
                      encoding='utf-8')
    named = ['--home', str(home), '--actor', MINT]
    assert cli.main(['grant'] + named + ['--file', str(policy)]) == 0
    granted = json.loads(capsys.readouterr().out)

    assert sorted(granted['entitled']) == sorted([DESK, STRANGER])
    assert granted['rewrap_owed'] == [DESK], 'the grant said nothing about the key it just owed'
    assert granted['unenrolled'] == [STRANGER] and granted['wrapped'] == []

    # ...and rewrap is what clears it, which is the other half of the property
    assert cli.main(['rewrap'] + named) == 0
    assert json.loads(capsys.readouterr().out)['events'] == 1

    # a second grant of the SAME document owes nothing, because the wrap is already there
    again = tmp_path / 'again.json'
    again.write_text(json.dumps(document(grants=((MINT, ADMIN, ANY), (DESK, BOOK, BOOK_ONE)),
                                         read=((DESK, FIRM_CLASS),))), encoding='utf-8')
    assert cli.main(['grant'] + named + ['--file', str(again)]) == 0
    settled = json.loads(capsys.readouterr().out)
    assert settled['rewrap_owed'] == [] and settled['wrapped'] == [DESK]
    assert settled['unenrolled'] == [] and settled['unresolved'] == []


def test_a_seat_may_bring_its_own_public_key_and_the_hub_then_holds_no_private_half(tmp_path):
    """The third residual, bounded rather than accepted.

    `enroll` will mint a keypair on the hub, and then one filesystem read yields every seat's private
    key - per-seat wrapping is a boundary between seats and not one against the hub or its backups,
    and revocation being forward-only means there is no remedy for such a read short of a class-key
    rotation this increment does not have. It is declared in custody's own header for what it is, and
    this is the way out of it: the seat generates its keypair on its own machine and hands over the
    PUBLIC half, so the private one never existed here. The enrollment fact is byte-identical either
    way - the record cannot tell, and should not be able to - and the wrap opens for the seat that
    holds the key.
    """
    home, log = minted(tmp_path)
    private = X25519PrivateKey.generate()
    public = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)

    minting = enroll(log, DESK, actor=MINT, public_key=public)
    assert minting['hub_holds_private_key'] is False and minting['seat_key'] is None
    assert not seat_key_path(home, DESK).exists(), 'the hub kept a key it was never given'
    assert log.store.get(minting['public_key']) == public

    (_, enrolled), = facts(log, 'seat_enrolled')
    assert enrolled == {'subject': DESK, 'algorithm': SEAT_ALGORITHM,
                        'public_key': minting['public_key']}

    declare(log, MINT, document(grants=((MINT, ADMIN, ANY),), read=((DESK, FIRM_CLASS),)))
    report = rewrap(log, actor=MINT)
    opened = unwrap_key(log.store.get(report['wrapped'][0]['wrap']),
                        private.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption()),
                        DESK)
    assert opened == read_class_key(home), 'the seat that holds the key opens its own wrap'

    # the hub-minted form is the bootstrap and says so in its own report, so a runbook can tell
    # which seats have a file on the hub still to be handed over and shredded
    assert enroll(log, MARKER, actor=MINT)['hub_holds_private_key'] is True
    assert seat_key_path(home, MARKER).exists()

    # and a public key that is not one is a refusal rather than a seat nobody can wrap to
    with pytest.raises(CustodyRefusal) as refusal:
        enroll(log, STRANGER, actor=MINT, public_key=b'far too short')
    assert 'public key' in str(refusal.value)
    assert not seat_key_path(home, STRANGER).exists()

    log.close()
    assert verify_home(home)['mode'] == 'entitled'


def test_rewrap_on_a_held_handle_reads_the_document_that_is_in_force_now(tmp_path):
    """The custody half of the stale-fold question, and the sharpest form of it.

    Reading never claims the home, so an open `SpineLog` outliving somebody else's append is
    ordinary. If the capability fold answered off an index derived when the handle opened, `rewrap`
    on that handle would read a REVOKED document and wrap the firm's class key to a subject the
    record no longer admits - and the writer would not stop it, because the actor doing the wrapping
    is a scoped admin and the stale input is the document rather than the actor. The wrap would land
    as a chained, verify-green fact, and revocation being forward-only, there would be no taking it
    back.
    """
    home, log = seated(tmp_path, read=(DESK, MARKER), enrolled=(DESK, MARKER))
    rewrap(log, actor=MINT)
    log.close()

    stale = SpineLog(home)
    assert wrap_drift(stale)['current'] == [DESK, MARKER]

    # the desk is enrolled but revoked, and a fresh seat is admitted, through another handle
    writer = SpineLog(home)
    enroll(writer, STRANGER, actor=MINT)
    declare(writer, MINT, document(grants=((MINT, ADMIN, ANY),),
                                   read=((MARKER, FIRM_CLASS), (STRANGER, FIRM_CLASS))))
    writer.close()

    report = rewrap(stale, actor=MINT)
    assert [row['subject'] for row in report['wrapped']] == [STRANGER], \
        'the held handle wrapped the class key to whoever the OLD document admitted'
    assert DESK not in report['entitled'] and report['current'] == [MARKER]

    stale.close()
    assert verify_home(home)['mode'] == 'entitled'


def test_a_re_enrolled_seat_is_rewrapped_and_revocation_stays_forward_only(tmp_path):
    """The lost laptop. Re-enrolling replaces the seat key, so the enrollment is younger than the
    wrap and `rewrap` issues a new one without anybody having to remember to ask.

    And the declared residual, gated rather than asserted in a docstring: the OLD wrap still opens
    under the OLD private key. Nothing here un-wraps, because nothing can - a seat that once held
    the key holds the history it already pulled, and the honest remedy is a class-key rotation,
    which is a later increment's logged event.
    """
    home, log = seated(tmp_path)
    rewrap(log, actor=MINT)
    (_, first), = facts(log, 'key_wrapped')
    lost = seat_private(home, DESK)

    enroll(log, DESK, actor=MINT)
    replaced = seat_private(home, DESK)
    assert replaced != lost, 're-enrollment kept the old keypair'
    report = rewrap(log, actor=MINT)
    assert report['events'] == 1 and report['current'] == []

    issued = log.store.get(report['wrapped'][0]['wrap'])
    assert unwrap_key(issued, replaced, DESK) == read_class_key(home)
    with pytest.raises(CustodyRefusal):
        unwrap_key(issued, lost, DESK)
    # forward-only: the old wrap is still openable by whoever already had it
    assert unwrap_key(log.store.get(first['wrap']), lost, DESK) == read_class_key(home)

    log.close()
    assert verify_home(home)['events'] == SpineLog(home).head()[0]


def test_rewrap_on_a_home_that_cannot_read_its_own_bodies_refuses_by_the_keys_name(tmp_path):
    """A crypto-shredded home has nothing to hand out, and the refusal says which file and which
    remedy - the escrow key - rather than reporting a fold it could not run."""
    home, log = seated(tmp_path)
    rewrap(log, actor=MINT)
    log.close()
    shredded = tmp_path / 'shredded'
    shutil.copytree(str(home), str(shredded))
    os.unlink(str(shredded / 'keys' / FIRM))

    with pytest.raises(CustodyRefusal) as refusal:
        rewrap(SpineLog(shredded), actor=MINT)
    assert FIRM in str(refusal.value) and 'escrow' in str(refusal.value)
    with pytest.raises(CustodyRefusal):
        read_class_key(shredded)


# --------------------------------------------------------------------------------------------
# Escrow: the declaration, the shred, and the way back.

def test_escrow_recovers_the_class_key_after_a_shred_on_a_copy_of_the_home(tmp_path):
    """The lost hub, walked end to end on a COPY, in the increment-1 shred posture.

    The escrow public key is declared as an ordinary policy, `rewrap` wraps to it beside the seats,
    and then one file is deleted - which is what crypto-shredding IS. The chain stays green over
    ciphertext because it was never taken over plaintext; the bodies are gone; and the custodian's
    private half gives the class key back BIT FOR BIT. The same private half materializes it into
    the copy, and the copy verifies entitled again with its checkpoints checked and its fills
    readable.

    Escrow is not a back door: it rides an ordinary `key_wrapped` row under a reserved subject, so
    what is exercised here is the seat mechanism read a second time.
    """
    home, log = seated(tmp_path)
    escrow_private, escrow_public = custodian()
    declared = declare_escrow(log, escrow_public, actor=MINT)
    report = rewrap(log, actor=MINT)
    assert report['escrow'] == declared['blob']
    assert sorted(row['subject'] for row in report['wrapped']) == sorted([DESK, ESCROW_SUBJECT])
    original = read_class_key(home)
    log.close()

    # the declaration is the ordinary open-bodied policy, verbatim, and the store holds the raw key
    (_, body), = [(lsn, fact) for lsn, fact in facts(SpineLog(home), 'policy_declared')
                  if fact.get('policy') == ESCROW_POLICY]
    assert body == {'policy': ESCROW_POLICY, 'blob': declared['blob']}
    assert SpineLog(home).store.get(declared['blob']) == escrow_public

    shredded = tmp_path / 'shredded'
    shutil.copytree(str(home), str(shredded))
    before = verify_home(shredded, entitled=False)
    os.unlink(str(shredded / 'keys' / FIRM))
    assert verify_home(shredded, entitled=False) == before, 'the chain noticed the shred'
    with pytest.raises(SealedBodyUnreadable):
        verify_home(shredded)

    assert recover_escrow(SpineLog(shredded), escrow_private) == original
    recovered = materialize(shredded, ESCROW_SUBJECT, escrow_private)
    assert recovered['class'] == FIRM_CLASS
    assert (shredded / 'keys' / FIRM).read_bytes() == original
    assert verify_home(shredded)['checkpoints_verified'] >= 1
    assert facts(SpineLog(shredded), 'fill') == [(SpineLog(home).head()[0] - 3, fill('EXEC-1'))]


def test_escrow_recovery_without_a_declaration_or_with_the_wrong_key_refuses_by_name(tmp_path):
    """Two causes, one refusal, and that is honest rather than lazy: from a home that cannot read
    its own bodies, "no escrow was ever declared" and "this is not the custodian's key" are the same
    observable fact - no wrap in this store opens for escrow. So the sentence names both and points
    at the declaration that fixes the first."""
    home, log = seated(tmp_path)
    rewrap(log, actor=MINT)
    log.close()
    escrow_private, escrow_public = custodian()

    with pytest.raises(CustodyRefusal) as refusal:
        recover_escrow(SpineLog(home), escrow_private)
    assert ESCROW_POLICY in str(refusal.value), refusal.value

    # declared and wrapped, but the private half offered is somebody else's
    log = SpineLog(home)
    declare_escrow(log, escrow_public, actor=MINT)
    rewrap(log, actor=MINT)
    assert recover_escrow(log, escrow_private) == read_class_key(home)
    log.close()
    with pytest.raises(CustodyRefusal):
        recover_escrow(SpineLog(home), custodian()[0])


def test_no_seat_may_be_enrolled_as_escrow_and_the_refusal_writes_nothing(tmp_path):
    """A seat wearing the reserved name would receive the custodian's wrap, so the name is refused
    at the one place it could be taken - and the refusal is a refusal: no key file, no blob cited by
    anything, and the head exactly where it was."""
    home, log = minted(tmp_path)
    head = log.head()

    with pytest.raises(CustodyRefusal) as refusal:
        enroll(log, ESCROW_SUBJECT, actor=MINT)
    assert ESCROW_SUBJECT in str(refusal.value) and ESCROW_POLICY in str(refusal.value)
    assert log.head() == head and not seat_key_path(home, ESCROW_SUBJECT).exists()

    for empty in ('', None):
        with pytest.raises(CustodyRefusal):
            enroll(log, empty, actor=MINT)
    assert log.head() == head


def test_the_class_key_file_this_module_names_is_the_one_genesis_mints(tmp_path):
    """Two modules, one file, and no way for the spellings to drift: `seal.py` mints
    `class_firm.key` and custody reads `class_<class>.key`, so the identity is pinned here rather
    than discovered the day a recovery writes a key nothing opens."""
    assert CLASS_KEY_FILE.format(FIRM_CLASS) == FIRM == 'class_firm.key'

    home, _ = minted(tmp_path)
    assert read_class_key(home) == (home / 'keys' / FIRM).read_bytes()
    with pytest.raises(CustodyRefusal) as refusal:
        read_class_key(home, 'desk-two')
    assert 'class_desk-two.key' in str(refusal.value), refusal.value


# --------------------------------------------------------------------------------------------
# The mouth.

def test_the_cli_enrolls_and_rewraps_and_reports_what_it_did(tmp_path, capsys):
    """The runbook as an operator types it: enroll a seat, declare the document that admits it,
    rewrap. Every answer is JSON on stdout because a script is the caller, and the second rewrap
    reports zero events - which is what makes it safe to leave in the runbook after every grant."""
    home = tmp_path / 'hub'
    init_home(home, MINT)
    named = ['--home', str(home)]

    assert cli.main(['enroll'] + named + ['--subject', DESK, '--actor', MINT]) == 0
    seat = json.loads(capsys.readouterr().out)
    assert seat['subject'] == DESK and seat['event_type'] == 'seat_enrolled'
    assert seat['seat_key'] == str(seat_key_path(home, DESK))

    policy = tmp_path / 'policy.json'
    policy.write_text(json.dumps(document(grants=((MINT, ADMIN, ANY),),
                                          read=((DESK, FIRM_CLASS), (STRANGER, FIRM_CLASS)))),
                      encoding='utf-8')
    assert cli.main(['grant'] + named + ['--file', str(policy), '--actor', MINT]) == 0
    capsys.readouterr()

    assert cli.main(['rewrap'] + named + ['--actor', MINT]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['events'] == 1 and report['unenrolled'] == [STRANGER]
    assert unwrap_key(SpineLog(home).store.get(report['wrapped'][0]['wrap']),
                      seat_private(home, DESK), DESK) == read_class_key(home)

    assert cli.main(['rewrap'] + named + ['--actor', MINT]) == 0
    assert json.loads(capsys.readouterr().out)['events'] == 0
    assert verify_home(home)['mode'] == 'entitled'


def test_the_cli_turns_a_custody_refusal_into_the_librarys_own_sentence(tmp_path, capsys):
    """A refusal reaches the terminal as the library's wording and exit 1 - naming the file and the
    remedy - never as a traceback. A CLI that reworded a refusal would be a second source of truth
    about what went wrong."""
    home = tmp_path / 'hub'
    init_home(home, MINT)
    os.unlink(str(home / 'keys' / FIRM))

    assert cli.main(['rewrap', '--home', str(home), '--actor', MINT]) == 1
    said = capsys.readouterr().err
    assert FIRM in said and 'Traceback' not in said

    assert cli.main(['enroll', '--home', str(home), '--subject', ESCROW_SUBJECT,
                     '--actor', MINT]) == 1
    assert ESCROW_SUBJECT in capsys.readouterr().err
