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

"""Who holds the class key - a seat per subject, the key wrapped to each of them, and the way back.

Bodies are sealed under a class key, which is an entitlement rather than a shared password only if
the key reaches the entitled and nobody else. The construction is one composition and nothing
invented: ephemeral X25519 to the recipient's published public key, HKDF-SHA256 over the shared
secret, AES-256-GCM over the class key.

Every step is a fact through the ordinary writer, under the ordinary durability law. A seat exists
because `seat_enrolled` publishes its public key at a hash; the key reached a subject because
`key_wrapped` names the wrap at a hash. So "who could read the firm's bodies in March" is a fold
over the log, with no side table holding a second opinion.

The wrap is addressed: GCM's additional data binds the `{"class", "subject"}` pair the `key_wrapped`
event carries in the clear, so moving one subject's wrap blob onto another subject's row stops it
opening. That binding is also what makes recovery possible on a replica that cannot read a single
body - see `materialize`.

Revocation is forward-only. There is no unwrap verb, no deletion and no re-sealing of history:
`rewrap` adds recipients, a subject dropped from the document stops receiving new wraps and keeps
what it already pulled, and the remedy for a compromised seat is a class-key rotation.

Two private keys never enter the log. A seat's lives at `keys/seats/<sha256(subject)[:16]>.key`,
hashed so a directory listing is not a membership roster. The escrow private half is not in this
package at all: escrow publishes a public key through policy and the custodian keeps the rest.

Declared limitation: `enroll` will mint a seat's keypair on the hub, and the hub then holds every
such seat's private half on one filesystem - so per-seat wrapping is an access boundary between
seats, not against the hub or its backups, with no remedy short of a class-key rotation. It is
bounded rather than accepted: hand `enroll` a `public_key` the seat generated on its own machine and
the private half is never on the hub. The hub-minted form is the bootstrap case, and the seat key it
writes should be handed to its seat and shredded off the hub.

Phase 1 has one class, so `FIRM_CLASS` is the default everywhere; the class is carried as a
parameter rather than assumed, so a second class is a second key file and a second wrap.
"""
import hashlib
import json
import os
from pathlib import Path

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.serialization import (
    Encoding, NoEncryption, PrivateFormat, PublicFormat)

from .canon import canonical_bytes
from .capability import read_subjects, state_at
from .errors import CustodyRefusal, SealedBodyUnreadable, SpineRefusal
from .log import SpineLog
from .seal import KEY_BYTES, NONCE_BYTES
from .vocabulary import FIRM_CLASS, is_hash

#: The wrap construction, named in the blob itself and versioned in the name, so an old wrap keeps
#: opening under the rule it was written by.
ALGORITHM = 'x25519-hkdf-sha256-aesgcm-v1'
#: HKDF's `info`, pinned for domain separation: a secret derived here is not the same 32 bytes as
#: one derived by another protocol over the same curve.
WRAP_INFO = b'derivus-spine-wrap-v1'
#: The four fields of a wrap blob, closed as an event body is: a fifth would be a channel into the
#: recovery path that no AAD covers.
WRAP_FIELDS = ('algorithm', 'ephemeral_public', 'nonce', 'wrapped')
#: Size bound on a wrap blob, so scanning a store for one never parses a vol cube - see
#: `_open_wrap`, which reads blobs it was not handed the address of.
WRAP_BLOB_LIMIT = 4096

#: What a seat's keypair is, said in the enrollment so the record carries its own algorithm agility.
SEAT_ALGORITHM = 'x25519'
#: Raw X25519 keys, private and public alike.
X25519_BYTES = 32
#: `keys/seats/` under the home. A seat file is named by the subject HASHED - sixteen hex characters
#: of SHA-256 - so a directory listing is not a membership roster of the deployment.
SEATS = 'seats'
SEAT_NAME_HEX = 16
#: The class key file, spelled the way `seal.py` spells it: `CLASS_KEY_FILE.format(FIRM_CLASS)` is
#: `seal.FIRM`.
CLASS_KEY_FILE = 'class_{}.key'

#: The escrow declaration: an ordinary `policy_declared` naming the escrow PUBLIC key's blob. The
#: private half is the custodian's and is not in this package, this home, or this log.
ESCROW_POLICY = 'escrow_key'
#: The reserved subject reference escrow's `key_wrapped` row rides under, so the lost laptop and the
#: custodian's copy are one mechanism read twice. `enroll` refuses this name.
ESCROW_SUBJECT = 'escrow'


def seat_key_path(home, subject) -> Path:
    """Where `subject`'s private seat key lives under `home`. Public so an operator, a gate and a
    runbook name the file one way rather than recomputing the hash three times."""
    digest = hashlib.sha256(_subject(subject).encode('utf-8')).hexdigest()
    return Path(home) / 'keys' / SEATS / '{}.key'.format(digest[:SEAT_NAME_HEX])


def wrap_aad(subject, entitlement_class=FIRM_CLASS) -> bytes:
    """The additional data a wrap is bound to: `{"class", "subject"}`, canonical.

    Exactly the pair the `key_wrapped` body carries in the clear, so the ciphertext and the record
    must agree about the recipient or nothing opens.
    """
    return canonical_bytes({'class': _text(entitlement_class, 'the entitlement class'),
                            'subject': _subject(subject)})


def wrap_key(class_key: bytes, recipient_public: bytes, subject: str,
             entitlement_class: str = FIRM_CLASS) -> bytes:
    """Wrap `class_key` to `recipient_public`, addressed to `subject`. Returns the wrap blob's bytes.

    Ephemeral X25519 to the recipient, HKDF-SHA256 over the shared secret under the pinned info
    string, AES-256-GCM under a fresh nonce and the `{class, subject}` AAD. The ephemeral half is
    fresh per wrap, so the same key wrapped twice to one seat yields two unrelated blobs.

    `subject` is a parameter rather than read out of the blob: an AAD taken from the ciphertext it
    checks would authenticate nothing. The caller supplies it from the `key_wrapped` row.
    """
    material = _secret(class_key, 'the class key')
    public = _recipient(recipient_public, 'the recipient public key')
    aad = wrap_aad(subject, entitlement_class)

    ephemeral = X25519PrivateKey.generate()
    derived = _derive(ephemeral.exchange(public))
    nonce = os.urandom(NONCE_BYTES)
    return canonical_bytes({
        'algorithm': ALGORITHM,
        'ephemeral_public': ephemeral.public_key().public_bytes(
            Encoding.Raw, PublicFormat.Raw).hex(),
        'nonce': nonce.hex(),
        'wrapped': AESGCM(derived).encrypt(nonce, material, aad).hex()})


def unwrap_key(wrap_blob: bytes, seat_private: bytes, subject: str,
               entitlement_class: str = FIRM_CLASS) -> bytes:
    """Open `wrap_blob` with `seat_private` as `subject` and return the class key.

    Every failure raises `CustodyRefusal` - a wrong shape, an algorithm this build does not speak, a
    private key that is not the recipient's, a wrap addressed elsewhere, a flipped byte. GCM cannot
    tell the last three apart, and what they share is that these bytes are not this seat's class key.
    """
    document = _wrap_document(wrap_blob)
    private = _seat_private(seat_private)
    aad = wrap_aad(subject, entitlement_class)

    ephemeral = _recipient(_hex(document['ephemeral_public'], X25519_BYTES, 'ephemeral_public'),
                           'the wrap\'s ephemeral public key')
    nonce = _hex(document['nonce'], NONCE_BYTES, 'nonce')
    wrapped = _hex(document['wrapped'], None, 'wrapped')
    try:
        return AESGCM(_derive(private.exchange(ephemeral))).decrypt(nonce, wrapped, aad)
    except Exception:
        raise CustodyRefusal(
            'this wrap does not open for {!r} under the {} class with the private key supplied: '
            'either it is addressed to another subject or class - the wrap is bound to the '
            '{{class, subject}} pair its key_wrapped row carries, so a blob moved onto another row '
            'stops opening - or this is not the seat it was wrapped to, or the blob has been '
            'altered. Re-enroll the seat and run `DV_Spine rewrap`, or recover through the escrow '
            'key.'.format(subject, entitlement_class))


def enroll(log, subject, actor, public_key: bytes = None) -> dict:
    """Mint `subject`'s seat: the public half published as a blob, `seat_enrolled` in the log.
    Returns the envelope plus what was minted.

    Two forms. With `public_key` - 32 raw bytes the seat generated on its own machine - the hub
    publishes it and writes no private key. Without, the hub mints the keypair and keeps the private
    half at `seat_key_path`; see this module's declared limitation.

    The public key is fsynced into the store first. A hub-minted private half is staged to scratch
    and moved into place only after the enrollment appends, so a refused append leaves an existing
    seat key exactly as it was.

    Re-enrollment replaces the seat key - the lost laptop's remedy. The old wrap stops being openable
    by anyone, and `rewrap` sees the enrollment is younger than the wrap and issues a new one.
    `ESCROW_SUBJECT` is refused as a subject.
    """
    subject = _subject(subject)
    if subject == ESCROW_SUBJECT:
        raise CustodyRefusal(
            '{!r} is the reserved subject the escrow wrap rides under, so no seat may be enrolled '
            'as it: a seat wearing that name would receive the custodian\'s wrap. Enroll the person '
            'under their own subject reference, and declare the escrow PUBLIC key through the '
            '{} policy instead.'.format(ESCROW_SUBJECT, ESCROW_POLICY))

    if public_key is None:
        private = X25519PrivateKey.generate()
        material = private.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
        published = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    else:
        material = None
        published = _recipient_bytes(public_key, 'the seat public key handed in')
    blob = log.store.put(published)

    path = seat_key_path(log.home, subject)
    scratch = None if material is None else _stage_secret(path, material)
    try:
        envelope = log.append(
            'seat_enrolled',
            {'subject': subject, 'algorithm': SEAT_ALGORITHM, 'public_key': blob},
            actor=actor, blob_refs=(blob,))
    except BaseException:
        # A key nobody published is not a seat, so clearing the scratch leaves the existing seat
        # working.
        if scratch is not None:
            _discard(scratch)
        raise
    if scratch is not None:
        os.replace(str(scratch), str(path))
    return dict(envelope, subject=subject, algorithm=SEAT_ALGORITHM, public_key=blob,
                seat_key=str(path) if scratch is not None else None,
                hub_holds_private_key=scratch is not None)


def declare_escrow(log, escrow_public, actor) -> dict:
    """Declare the deployment's escrow public key: the blob, then `policy_declared`. Returns the
    envelope plus the blob.

    An ordinary open-bodied policy declaration, so a changed custodian key is one more declaration
    and `rewrap` wraps to it because the declaration is younger than the wrap.
    """
    material = _recipient_bytes(escrow_public, 'the escrow public key')
    blob = log.store.put(material)
    envelope = log.append('policy_declared', {'policy': ESCROW_POLICY, 'blob': blob},
                          actor=actor, blob_refs=(blob,))
    return dict(envelope, policy=ESCROW_POLICY, blob=blob)


def rewrap(log, actor, entitlement_class=FIRM_CLASS) -> dict:
    """Wrap the class key to every subject the document in force admits, and to escrow. Returns a
    report of what was written and what could not be.

    Recipients are the subjects granted read over `entitlement_class`, plus the declared escrow key.
    A wrap counts as current when it was appended after the enrollment (or escrow declaration) it is
    addressed to, so a second call emits nothing while a re-enrolled seat's stale wrap is reissued.

    A subject with read and no enrollment gets no wrap and is named under `unenrolled`; one whose
    latest `key_wrapped` cites a blob that is not a wrap is named under `unresolved` and reissued.
    Nothing here removes a wrap: revocation is forward-only.
    """
    class_key = read_class_key(log.home, entitlement_class)
    drift = wrap_drift(log, entitlement_class)

    wrapped = []
    for subject, blob in drift['pending']:
        raw = wrap_key(class_key, log.store.get(blob), subject, entitlement_class)
        address = log.store.put(raw)
        envelope = log.append(
            'key_wrapped', {'class': entitlement_class, 'subject': subject, 'wrap': address},
            actor=actor, blob_refs=(address,))
        wrapped.append({'subject': subject, 'wrap': address, 'lsn': envelope['lsn']})

    return {'home': str(log.home), 'actor': actor, 'class': entitlement_class,
            'entitled': drift['entitled'], 'wrapped': wrapped, 'current': drift['current'],
            'unenrolled': drift['unenrolled'], 'unresolved': drift['unresolved'],
            'escrow': drift['escrow'], 'events': len(wrapped)}


def wrap_drift(log, entitlement_class=FIRM_CLASS) -> dict:
    """What the document in force says the store owes: who is entitled and has no current wrap.

    Read-only, and separate from `rewrap` because two callers need it - `rewrap` acts on it and the
    `grant` verb reports it, so a subject granted read is never left to find out when a body will
    not open.

    Three ways to be owed a wrap, named separately since the remedies differ. `pending` is what a
    rewrap can issue now; `unenrolled` is subjects with no seat, who must be enrolled first;
    `unresolved` is subjects whose newest `key_wrapped` cites a blob that is not a wrap, which does
    not count as current and is reissued.
    """
    document, _ = state_at(log)
    entitled = read_subjects(document, entitlement_class)
    seats, wraps, escrow, unresolved = _custody_state(log, entitlement_class)

    pending, current, missing, broken = [], [], [], []
    for subject in entitled:
        seat = seats.get(subject)
        if seat is None:
            missing.append(subject)
        elif wraps.get(subject, 0) > seat['lsn']:
            current.append(subject)
        else:
            pending.append((subject, seat['public_key']))
            if subject in unresolved:
                broken.append(subject)
    if escrow is not None:
        if wraps.get(ESCROW_SUBJECT, 0) > escrow['lsn']:
            current.append(ESCROW_SUBJECT)
        else:
            pending.append((ESCROW_SUBJECT, escrow['blob']))
            if ESCROW_SUBJECT in unresolved:
                broken.append(ESCROW_SUBJECT)
    return {'class': entitlement_class, 'entitled': list(entitled), 'pending': pending,
            'current': current, 'unenrolled': missing, 'unresolved': broken,
            'escrow': escrow['blob'] if escrow is not None else None}


def materialize(home, subject, seat_private: bytes, entitlement_class=FIRM_CLASS) -> dict:
    """Put the class key into a replica `home` out of `subject`'s wrap, and report what was
    recovered.

    A replica arriving with `log/` and `blobs/` alone verifies its chain over ciphertext; after this
    it verifies entitled, reads its bodies, and folds its own projections.

    It refuses to overwrite an existing class key, the exclusive file creation being the refusal, so
    a race loses the way a retype does - overwriting would be crypto-shredding by accident. Which
    wrap belongs to `subject` is answered by the AAD rather than by the log, since the `key_wrapped`
    row naming it is sealed under the very key being recovered: the store's wraps are trial-opened
    under `{class, subject}`. Two wraps opening to different keys is refused, not chosen.
    """
    log = SpineLog(home)
    target = Path(home) / 'keys' / CLASS_KEY_FILE.format(
        _text(entitlement_class, 'the entitlement class'))
    if target.exists():
        raise CustodyRefusal(
            '{} already holds a {} class key, so nothing is written: overwriting it would strand '
            'every body sealed under the key that is there, which is crypto-shredding by accident. '
            'Materialize into a home that lacks the key - a replica holding log/ and blobs/ - or '
            'move the existing key aside deliberately if you mean to replace it.'.format(
                target, entitlement_class))

    address, class_key = _open_wrap(
        log.store, subject, seat_private, entitlement_class,
        'no wrap in the store at {} opens for {!r} under the {} class with this private key'.format(
            log.store.blobs, subject, entitlement_class),
        'Run `DV_Spine rewrap --actor <admin>` on the hub so a wrap for this subject exists, copy '
        'blobs/ across again, and check that the private key is the one `DV_Spine enroll` minted '
        'for this subject')

    _write_class_key(target, class_key)
    try:
        frame = next(log.frames(), None)
        if frame is not None:
            log.open_body(frame)
    except SealedBodyUnreadable:
        # The wrap opened but the key does not seal this log, so leave no half-materialized home.
        _discard(target)
        raise CustodyRefusal(
            'the wrap {} opened for {!r} but the key inside it does not open this log\'s bodies, so '
            'nothing was written: the wrap and the home at {} are from different deployments. '
            'Re-copy log/ and blobs/ from the hub that issued this wrap.'.format(
                address, subject, log.home))
    return {'home': str(log.home), 'subject': subject, 'class': entitlement_class,
            'wrap': address, 'key_file': str(target), 'bodies_readable': True}


def recover_escrow(log, escrow_private: bytes, entitlement_class=FIRM_CLASS) -> bytes:
    """The class key recovered from the escrow wrap, as raw bytes.

    Nothing is written: the caller decides where the key goes, and
    `materialize(home, ESCROW_SUBJECT, escrow_private)` is the same recovery run as a write. A home
    that declared no escrow key and a wrong private key raise the same refusal, since from here they
    are one fact - no wrap in this store opens for escrow with what was handed in.
    """
    return _open_wrap(
        log.store, ESCROW_SUBJECT, escrow_private, entitlement_class,
        'no escrow wrap in the store at {} opens under the private key supplied'.format(
            log.store.blobs),
        'Declare the escrow public key (`policy_declared` {{"policy": "{}"}}) and run `DV_Spine '
        'rewrap` on a home that still holds the class key, or check that this is the custodian\'s '
        'private half of the key that was declared'.format(ESCROW_POLICY))[1]


def read_class_key(home, entitlement_class=FIRM_CLASS) -> bytes:
    """The class key of `home`, read off `keys/class_<class>.key`.

    Read at every use rather than held, so a key destroyed at 14:02 is destroyed at 14:02. An absent
    or wrong-width file raises `CustodyRefusal`: the caller is giving the key away, not opening a
    body.
    """
    path = Path(home) / 'keys' / CLASS_KEY_FILE.format(
        _text(entitlement_class, 'the entitlement class'))
    try:
        material = path.read_bytes()
    except (IOError, OSError):
        raise CustodyRefusal(
            '{} is not in this home, so there is no {} class key to wrap to anybody: this home '
            'either never held it or was crypto-shredded. Run this on the hub that holds the key, '
            'or recover it through the escrow private key first '
            '(`custody.recover_escrow`).'.format(path, entitlement_class))
    if len(material) != KEY_BYTES:
        raise CustodyRefusal(
            '{} is {} bytes, not {}: the class key file is truncated or is not a key, and wrapping '
            'it would hand out something no body was ever sealed under - restore it from the '
            'custodian\'s copy.'.format(path, len(material), KEY_BYTES))
    return material


# ------------------------------------------------------------------------------------------------
# The fold, and the pieces every verb above is made of.

def _custody_state(log, entitlement_class):
    """`(seats, wraps, escrow, unresolved)` folded out of the log - who has a seat, who has a current
    wrap, which escrow key is declared, and whose wrap row does not bear itself out.

    Located by envelope and read by body: only the three types that can move custody are opened, so
    the fold costs the length of the custody history. Latest wins in LSN order, so a re-enrollment
    supersedes and a redeclared escrow key takes over.

    A `key_wrapped` row counts only once the blob it cites opens as a wrap. Durability closure proves
    bytes are at that address, not that they are a wrap, and a row citing anything else would report
    the subject current while suppressing the reissue forever. The AAD cannot be checked here - that
    needs a private key - so the test is the blob's shape, declared construction and decodable fields.
    """
    seats, wraps, escrow, unresolved = {}, {}, None, {}
    for frame in log.frames():
        kind = frame['event_type']
        if kind not in ('seat_enrolled', 'key_wrapped', 'policy_declared'):
            continue
        body = log.open_body(frame)
        if not isinstance(body, dict):
            continue
        if kind == 'seat_enrolled' and is_hash(body.get('public_key')):
            seats[body.get('subject')] = {'lsn': frame['lsn'], 'public_key': body['public_key']}
        elif kind == 'key_wrapped' and body.get('class') == entitlement_class:
            if _is_wrap(log.store, body.get('wrap')):
                wraps[body.get('subject')] = frame['lsn']
            else:
                unresolved[body.get('subject')] = frame['lsn']
        elif kind == 'policy_declared' and body.get('policy') == ESCROW_POLICY \
                and is_hash(body.get('blob')):
            escrow = {'lsn': frame['lsn'], 'blob': body['blob']}
    return seats, wraps, escrow, unresolved


def _is_wrap(store, address):
    """Whether `address` resolves to a wrap blob.

    Every way of failing - absent, altered, oversized, not the four-field object, another
    construction, a field that will not decode - answers False: the caller's one question is whether
    this row counts as a key delivered.
    """
    if not is_hash(address):
        return False
    try:
        raw = store.get(address)
    except SpineRefusal:
        return False
    if len(raw) > WRAP_BLOB_LIMIT:
        return False
    try:
        document = _wrap_document(raw)
        _hex(document['ephemeral_public'], X25519_BYTES, 'ephemeral_public')
        _hex(document['nonce'], NONCE_BYTES, 'nonce')
        _hex(document['wrapped'], None, 'wrapped')
    except CustodyRefusal:
        return False
    return True


def _open_wrap(store, subject, private, entitlement_class, absence, remedy):
    """`(blob id, class key)` for the one wrap in `store` that opens for `subject`, or
    `CustodyRefusal` carrying `absence` and `remedy`.

    A scan rather than a lookup, because the caller cannot read the bodies naming the wraps - see
    `materialize`. The AAD does the selecting, so trial-opening is exact rather than heuristic and
    the store's walk order makes it deterministic. Two wraps opening to different keys is refused:
    the home is holding blobs from two histories.
    """
    material = _secret_or_none(private)
    if material is None:
        raise CustodyRefusal(
            'the private key handed in is {} bytes, not the {} of a raw X25519 private key: read it '
            'out of the seat key file (`custody.seat_key_path`) as bytes rather than passing a path '
            'or a PEM.'.format(
                len(private) if isinstance(private, (bytes, bytearray)) else 0, X25519_BYTES))
    found = None
    for digest in store.walk():
        raw = store.get(digest)
        if len(raw) > WRAP_BLOB_LIMIT:
            continue
        try:
            opened = unwrap_key(raw, material, subject, entitlement_class)
        except CustodyRefusal:
            continue
        if found is not None and found[1] != opened:
            raise CustodyRefusal(
                'two wraps in the store at {} open for {!r} under the {} class and carry DIFFERENT '
                'keys ({} and {}), so nothing is chosen: this home holds blobs from two histories, '
                'and materializing either would be a guess about which record it belongs to. Re-copy '
                'blobs/ from the hub this log came from.'.format(
                    store.blobs, subject, entitlement_class, found[0], digest))
        if found is None:
            found = (digest, opened)
    if found is None:
        raise CustodyRefusal('{}: {}.'.format(absence, remedy))
    return found


def _derive(shared):
    """HKDF-SHA256 over the ECDH shared secret - 32 bytes for AES-256, under the pinned info string.

    The raw X25519 output is never a key directly: it is a curve point's coordinate rather than a
    uniform string.
    """
    return HKDF(algorithm=hashes.SHA256(), length=KEY_BYTES, salt=None,
                info=WRAP_INFO).derive(shared)


def _wrap_document(wrap_blob):
    """`wrap_blob` parsed as the four-field wrap document, or `CustodyRefusal` naming what it is
    instead. Closed at the field level: a fifth key would be bytes the AAD does not cover."""
    if not isinstance(wrap_blob, (bytes, bytearray)):
        raise CustodyRefusal(
            'the wrap is {}, not bytes: hand in the blob\'s own bytes (`log.store.get(wrap)`) - '
            'this reads the stored object, not a path and not a parsed dict.'.format(
                type(wrap_blob).__name__))
    try:
        document = json.loads(bytes(wrap_blob).decode('utf-8'))
    except (UnicodeDecodeError, ValueError):
        raise CustodyRefusal(
            'the wrap is not UTF-8 JSON, so it is not a wrap blob: a wrap is the canonical {{{}}} '
            'object - pull the blob again from a replica that holds it, since a blob is '
            'self-verifying by hash.'.format(', '.join(WRAP_FIELDS)))
    if not isinstance(document, dict) or sorted(document) != list(WRAP_FIELDS):
        raise CustodyRefusal(
            'the wrap is not the {{{}}} object a wrap blob is (it carries {}): the shape is closed, '
            'so this is either another kind of blob or one that was edited - pull it again from a '
            'replica that holds it.'.format(
                ', '.join(WRAP_FIELDS),
                ', '.join(sorted(document)) if isinstance(document, dict)
                else type(document).__name__))
    if document['algorithm'] != ALGORITHM:
        raise CustodyRefusal(
            'the wrap declares algorithm {!r} and this build speaks {!r}: a construction is named in '
            'the blob so an old wrap keeps opening under the rule it was written by - open this one '
            'with the build that wrote it, or rewrap under the current construction.'.format(
                document['algorithm'], ALGORITHM))
    return document


def _hex(value, length, field):
    """The wrap field `field` decoded from hex, checked against `length` where one is fixed, and
    required to be non-empty."""
    try:
        material = bytes.fromhex(value)
    except (AttributeError, TypeError, ValueError):
        raise CustodyRefusal(
            'the wrap\'s {} is {!r}, which is not hex: the blob has been altered - pull it again '
            'from a replica that holds it, since a blob is self-verifying by hash.'.format(
                field, value))
    if length is not None and len(material) != length:
        raise CustodyRefusal(
            'the wrap\'s {} is {} bytes, not {}: the blob has been truncated or edited - pull it '
            'again from a replica that holds it.'.format(field, len(material), length))
    if not material:
        raise CustodyRefusal(
            'the wrap\'s {} is empty: there is nothing here to open - pull the blob again from a '
            'replica that holds it.'.format(field))
    return material


def _secret(material, what):
    """`material` asserted to be `KEY_BYTES` of raw key material, the width every key file here
    has."""
    if not isinstance(material, (bytes, bytearray)) or len(material) != KEY_BYTES:
        raise CustodyRefusal(
            '{} is {} bytes, not the {} a class key is: pass the raw key material '
            '(`custody.read_class_key`) rather than a path, a hex string or a file '
            'object.'.format(what, len(material) if isinstance(material, (bytes, bytearray)) else 0,
                             KEY_BYTES))
    return bytes(material)


def _secret_or_none(material):
    """`material` as a raw X25519 private key's bytes, or None. The scan's own check, which must not
    raise before it can say which wrap it was looking for."""
    if not isinstance(material, (bytes, bytearray)) or len(material) != X25519_BYTES:
        return None
    return bytes(material)


def _seat_private(material):
    """The seat's private key as the `X25519PrivateKey` the exchange needs."""
    raw = _secret_or_none(material)
    if raw is None:
        raise CustodyRefusal(
            'the seat private key is {} bytes, not the {} of a raw X25519 private key: read the '
            'seat key file (`custody.seat_key_path`) as bytes - nothing here parses PEM, and a '
            'truncated key is not a wrong key, it is a broken file.'.format(
                len(material) if isinstance(material, (bytes, bytearray)) else 0, X25519_BYTES))
    try:
        return X25519PrivateKey.from_private_bytes(raw)
    except Exception:
        raise CustodyRefusal(
            'the seat private key is {} bytes but is not an X25519 private key: restore the seat '
            'key file from the seat that holds it, or re-enroll the subject and '
            'rewrap.'.format(X25519_BYTES))


def _recipient_bytes(material, what):
    """`material` asserted to be a raw X25519 public key's bytes. `what` names it in refusals."""
    if not isinstance(material, (bytes, bytearray)) or len(material) != X25519_BYTES:
        raise CustodyRefusal(
            '{} is {} bytes, not the {} of a raw X25519 public key: publish the key as raw bytes '
            '(`public_bytes(Encoding.Raw, PublicFormat.Raw)`) - the store holds the key itself, not '
            'an encoding of it.'.format(
                what, len(material) if isinstance(material, (bytes, bytearray)) else 0,
                X25519_BYTES))
    material = bytes(material)
    try:
        X25519PublicKey.from_public_bytes(material)
    except Exception:
        raise CustodyRefusal(
            '{} is {} bytes but is not an X25519 public key: republish the seat\'s public key, or '
            'declare an escrow key the deployment actually holds the private half of.'.format(
                what, X25519_BYTES))
    return material


def _recipient(material, what):
    """The recipient's public key as the `X25519PublicKey` the exchange needs."""
    return X25519PublicKey.from_public_bytes(_recipient_bytes(material, what))


def _subject(subject):
    """`subject` asserted to be a non-empty subject reference."""
    return _text(subject, 'the subject reference')


def _text(value, what):
    """`value` asserted to be a non-empty string. `what` names it in the refusal."""
    if not isinstance(value, str) or value == '':
        raise CustodyRefusal(
            '{} is {!r}, and a name that names nothing is not a name: pass the pseudonymous '
            'reference the log carries.'.format(what, value))
    return value


def _stage_secret(path, material):
    """Write `material` beside `path` as scratch, fsynced and 0600 where the platform has an
    opinion, and return the scratch path.

    Scratch first, so the move into place is the last thing that happens and a refused append leaves
    no private key the record never published.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.parent / '{}.{}'.format(path.name, os.urandom(8).hex())
    _write_secret(scratch, material)
    return scratch


def _write_class_key(path, material):
    """Write the class key into a home that lacks one. The exclusive create is the no-overwrite
    rule, so a race loses the same way a retype does."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _write_secret(path, material)
    except (IOError, OSError) as blocked:
        raise CustodyRefusal(
            '{} could not be written ({}): if the file appeared between the check and the write, '
            'another recovery got there first and this one changes nothing - read the home\'s '
            'status and verify it entitled.'.format(path, blocked))


def _write_secret(path, material):
    """Write one key file, fsynced and as private as the platform admits.

    `xb` and not `wb`: exclusive creation is how this module refuses to overwrite, and the mode must
    be binary - a text handle on Windows would rewrite the `0x0A` bytes a random key contains.
    """
    with open(str(path), 'xb') as handle:
        handle.write(material)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.chmod(str(path), 0o600)
    except (IOError, OSError):
        # Best effort, as in seal.py: Windows ACLs are not this mode.
        pass


def _discard(path):
    """Remove a key file that never became one, ignoring a failure to. Scratch has no address and
    nothing can cite it, so this is not deletion of anything the record names."""
    try:
        os.unlink(str(path))
    except (IOError, OSError):
        pass
