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

The log is classified: envelopes are firm-visible and bodies are sealed under a class key. That is
only an entitlement rather than a shared password if the key REACHES the entitled and no one else,
and this module is the whole of how it travels. Nothing here invents cryptography; it composes the
one construction the deployment can afford to be boring about - ephemeral X25519 to the recipient's
published public key, HKDF-SHA256 over the shared secret, AES-256-GCM over the class key - and
spends its care on what the record says about each step.

Every step is a FACT, because custody is exactly the kind of thing that becomes a rumour otherwise.
A seat exists because `seat_enrolled` says its public key is at a hash; the key reached a subject
because `key_wrapped` says the wrap is at a hash. Both go through the ordinary writer, under the
ordinary durability law - the blob is fsynced before the event that speaks of it - so "who could
read the firm's bodies in March" is a fold over the log like every other authorization question,
and no side table anywhere holds a second opinion about it.

The WRAP IS ADDRESSED, and that is the load-bearing detail. GCM's additional data binds the
`{"class", "subject"}` pair the `key_wrapped` event carries in the clear, so a wrap is not merely
"a class key encrypted to a public key" - it is a class key encrypted to THIS subject for THIS
class. Move one subject's wrap blob onto another subject's row and it stops opening: the record and
the ciphertext have to agree about who the recipient is, so a hub that mislabels a wrap produces a
refusal rather than a quiet mis-delivery. It is also what makes recovery possible on a replica that
cannot read a single body - see `materialize`.

Revocation is FORWARD-ONLY, and this module does not pretend otherwise. There is no unwrap verb, no
deletion, no re-sealing of history: a seat that once held the key holds the history it already
pulled, which is the brief's declared residual rather than a gap. `rewrap` ADDS recipients and
nothing else; a subject dropped from the document simply stops receiving new wraps, and the honest
remedy for a compromised seat is a class-key rotation, which is a later increment's logged event.

Two keys never enter the log and never leave the machine that holds them. A seat's private half
lives at `keys/seats/<sha256(subject)[:16]>.key`, hashed so that a directory listing is not a
membership roster - a path is data too, and the record is pseudonymous by rule. The escrow private
half is not here at all: escrow publishes a PUBLIC key through policy and the private half lives
wherever the deployment's custodian keeps it, which is the entire point of an escrow.

A THIRD RESIDUAL is declared here rather than left to be discovered, in the voice the brief declares
its other two in. `enroll` will MINT a seat's keypair on the hub, and the hub then holds every seat's
private half on one filesystem - so per-seat wrapping is an access boundary between seats and not one
against the hub or its backups, and because revocation is forward-only there is no remedy for a hub
read short of a class-key rotation this increment does not have. It is declared rather than solved
because the honest fix is a key-management posture the non-goals rule out ("no password or secret
storage"), and it is BOUNDED rather than accepted: hand `enroll` a `public_key` the seat generated on
its own machine and the private half is never on the hub at all, which is the posture a deployment
that cares should run. The hub-minted form is the BOOTSTRAP case - convenient for a desk being stood
up, and the seat key it writes should be handed to its seat and shredded off the hub afterwards.

Phase 1 has one class, so `firm` is the default everywhere and the class is carried as a parameter
rather than assumed - the day desk two arrives, a second class key is a second file and a second
wrap, not a redesign.
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

#: The wrap construction, named in the blob itself so a reader never has to guess which one it is
#: holding. Versioned in the name: a second construction is a second string and an old wrap keeps
#: opening under the rule it was written by.
ALGORITHM = 'x25519-hkdf-sha256-aesgcm-v1'
#: HKDF's `info`, pinned. Domain separation is what stops a shared secret derived here from being
#: the same 32 bytes as one derived by some other protocol that happens to use the same curve.
WRAP_INFO = b'derivus-spine-wrap-v1'
#: The four fields of a wrap blob, closed exactly as an event body is: a fifth would be a channel
#: into the recovery path that no AAD covers.
WRAP_FIELDS = ('algorithm', 'ephemeral_public', 'nonce', 'wrapped')
#: A wrap blob is a few hundred bytes. The bound exists so that scanning a store for one never
#: parses a vol cube - see `_open_wrap`, which reads blobs it was not handed the address of.
WRAP_BLOB_LIMIT = 4096

#: What a seat's keypair is, said in the enrollment so the record carries its own algorithm agility.
SEAT_ALGORITHM = 'x25519'
#: Raw X25519 keys, private and public alike.
X25519_BYTES = 32
#: `keys/seats/` under the home, and the filename is the SUBJECT HASHED - sixteen hex characters of
#: SHA-256. A directory listing of a hub would otherwise be a membership roster of the deployment,
#: and the record is pseudonymous by rule; the hash is a name that is stable, collision-free at desk
#: scale, and says nothing.
SEATS = 'seats'
SEAT_NAME_HEX = 16
#: The class key file, spelled the way `seal.py` spells it. `CLASS_KEY_FILE.format(FIRM_CLASS)` is
#: `seal.FIRM`, and a gate pins the two together so the one class phase 1 has cannot drift into two.
CLASS_KEY_FILE = 'class_{}.key'

#: The escrow declaration: an ordinary `policy_declared` naming the escrow PUBLIC key's blob. The
#: private half is the custodian's and is not in this package, this home, or this log.
ESCROW_POLICY = 'escrow_key'
#: Escrow rides the ordinary `key_wrapped` row under a reserved subject reference, so the lost
#: laptop and the custodian's copy are the same mechanism read twice rather than two mechanisms.
#: `enroll` refuses the name, which is what keeps a seat from wearing it.
ESCROW_SUBJECT = 'escrow'


def seat_key_path(home, subject) -> Path:
    """Where `subject`'s private seat key lives under `home`.

    Public because an operator, a gate and the lost-laptop runbook all need to name the file, and
    recomputing the hash in three places is how three spellings of one path are born.
    """
    digest = hashlib.sha256(_subject(subject).encode('utf-8')).hexdigest()
    return Path(home) / 'keys' / SEATS / '{}.key'.format(digest[:SEAT_NAME_HEX])


def wrap_aad(subject, entitlement_class=FIRM_CLASS) -> bytes:
    """The additional data a wrap is bound to: `{"class", "subject"}`, canonical.

    Exactly the pair the `key_wrapped` envelope's own body carries in the clear, so the ciphertext
    and the record have to agree about who the recipient is or nothing opens.
    """
    return canonical_bytes({'class': _text(entitlement_class, 'the entitlement class'),
                            'subject': _subject(subject)})


def wrap_key(class_key: bytes, recipient_public: bytes, subject: str,
             entitlement_class: str = FIRM_CLASS) -> bytes:
    """Wrap `class_key` to `recipient_public`, addressed to `subject`. Answers the wrap blob's bytes.

    Ephemeral X25519 to the recipient, HKDF-SHA256 over the shared secret with the pinned info
    string, AES-256-GCM under a fresh 12-byte nonce and the `{class, subject}` AAD. The ephemeral
    half is fresh per wrap, so wrapping the same key to the same seat twice yields two different
    blobs and neither leaks that they carry the same secret.

    The recipient's subject is a PARAMETER rather than something read out of the blob, and it has
    to be: an AAD a reader could take from the ciphertext it is checking would authenticate nothing.
    The caller supplies it from the `key_wrapped` row, which is where the record says who this is
    for.
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
    """Open `wrap_blob` with `seat_private` as `subject`. Answers the class key.

    Every failure is one `CustodyRefusal`, and deliberately one: a blob that is not the shape, an
    algorithm this version does not speak, a private key that is not the recipient's, a wrap
    addressed to somebody else, a flipped byte. GCM cannot tell the last three apart and neither
    should a caller - what they have in common is that these bytes are not this seat's class key,
    and reporting anything finer would be describing a key nobody here holds.
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
    Answers the envelope plus what was minted.

    TWO FORMS, and the difference between them is the residual this module's header declares. Hand
    in `public_key` - 32 raw bytes the seat generated on its own machine - and the hub publishes it
    and writes NO private key: the private half never existed here, so a hub filesystem read yields
    nothing about that seat. Omit it and the hub mints the keypair and keeps the private half at
    `seat_key_path`, which is the BOOTSTRAP convenience: fine for standing a desk up, and the file
    should be handed to its seat and shredded off the hub once it has arrived.

    The order is the durability law and one deliberate refinement of it. The public key is fsynced
    into the store first, because no event may cite a blob that is not on the platter. A hub-minted
    private half is written to scratch NEXT and moved into place only after the enrollment appends:
    a refused append - an actor without admin, most likely - therefore leaves an existing seat key
    exactly as it was, rather than clobbering a working seat on the way to being told no.

    Re-enrollment REPLACES the seat key, and that is the lost laptop's remedy rather than an
    oversight: the old wrap simply stops being openable by anyone, the new public key is published,
    and `rewrap` notices the enrollment is younger than the wrap and issues a new one. Nothing is
    stranded by it, which is exactly why this refuses to overwrite nothing while `Keys.generate`
    refuses to overwrite everything - a class key seals history, a seat key only receives it.
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
        # A key nobody published is not a seat and cannot be cited: clearing it is hygiene, and the
        # seat that was already there keeps working.
        if scratch is not None:
            _discard(scratch)
        raise
    if scratch is not None:
        os.replace(str(scratch), str(path))
    return dict(envelope, subject=subject, algorithm=SEAT_ALGORITHM, public_key=blob,
                seat_key=str(path) if scratch is not None else None,
                hub_holds_private_key=scratch is not None)


def declare_escrow(log, escrow_public, actor) -> dict:
    """Declare the deployment's escrow public key: the blob, then `policy_declared`. Answers the
    envelope plus the blob.

    The escrow is a PUBLIC key in the record and a private key in a safe. Declaring it is an
    ordinary policy declaration - the same open-bodied `policy_declared` every other policy rides -
    so the day the custodian's key changes, that is one more declaration and `rewrap` wraps to the
    new one because the declaration is younger than the wrap.
    """
    material = _recipient_bytes(escrow_public, 'the escrow public key')
    blob = log.store.put(material)
    envelope = log.append('policy_declared', {'policy': ESCROW_POLICY, 'blob': blob},
                          actor=actor, blob_refs=(blob,))
    return dict(envelope, policy=ESCROW_POLICY, blob=blob)


def rewrap(log, actor, entitlement_class=FIRM_CLASS) -> dict:
    """Wrap the class key to everyone the document in force now admits, and to escrow. Answers a
    report of what it did and what it could not.

    Who is "everyone": the subjects the capabilities document grants READ over this class, plus the
    declared escrow key. What counts as MISSING is answered by LSN order and nothing else - a wrap
    is current when it was appended AFTER the enrollment (or the escrow declaration) it is addressed
    to. That single rule buys both halves of the contract: a second call finds every wrap younger
    than its enrollment and emits nothing, while a re-enrolled seat's old wrap is older than the new
    public key and is reissued without anybody having to remember to ask.

    A subject with READ and no enrollment gets no wrap and is NAMED in the report. It is not an
    error - the document is allowed to describe a person who has not sat down yet - but a silent
    omission is how an entitlement becomes a rumour, so the report says the sentence out loud. A
    subject whose latest `key_wrapped` cites a blob that is not a wrap is named too, under
    `unresolved`, and is reissued: see `wrap_drift`.

    Revocation is forward-only: nothing here removes a wrap, and nothing can. A subject dropped from
    the document stops receiving new ones and keeps what it already pulled, which is the brief's
    declared residual.
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
    """What the document in force says the store OWES: who is entitled and has no current wrap.

    Read-only, and separated from `rewrap` because the answer is owed to two callers. `rewrap` acts
    on it; the `grant` verb REPORTS it, because "rewrap on grant change" is a property of the system
    or it is a line in a runbook, and a runbook line is the failure mode where a subject is granted
    READ, is told nothing, and finds out when a body will not open.

    Three ways to be owed a wrap and each is named separately, because the remedies differ. `pending`
    holds the ones a rewrap can issue now. `unenrolled` holds subjects the document admits who have
    no seat: enroll them first. `unresolved` holds subjects whose newest `key_wrapped` cites a blob
    that is NOT a wrap - the fact says an entitlement was delivered and the store does not bear it
    out, so the row does not count as current and the wrap is reissued, which is the whole reason the
    fold opens the blob instead of trusting the citation.
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
    """Put the class key into a replica `home` out of `subject`'s wrap. Answers what was recovered.

    This is what makes a wrapped key an ENTITLEMENT rather than decoration: a replica that arrived
    holding `log/` and `blobs/` and nothing else verifies its chain over ciphertext, and after this
    call it verifies entitled, reads its own bodies, and folds its own projections.

    It refuses to overwrite an existing class key, and the refusal is the file creation itself
    (`O_EXCL`), so a race loses the same way a retype does. Overwriting would be crypto-shredding by
    accident - every body sealed under the key that was there would become unreadable in the one
    move that looks like it is granting access.

    Which wrap is `subject`'s is answered by the AAD rather than by the log, and that is forced
    rather than chosen: the `key_wrapped` row naming the wrap has a SEALED body, sealed under the
    very key this call exists to recover. So the wraps the store holds are trial-opened under
    `{class, subject}`, and the binding is what makes the answer unambiguous - only the blob
    addressed to this subject and openable by this private key comes apart. Two wraps that open to
    DIFFERENT keys is not a choice this verb makes: it refuses and names both, because materializing
    either would be a guess about which history the home belongs to.
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
        # The wrap opened but what came out does not seal this log: the home and the wrap belong to
        # two histories. Leave no half-materialized home behind.
        _discard(target)
        raise CustodyRefusal(
            'the wrap {} opened for {!r} but the key inside it does not open this log\'s bodies, so '
            'nothing was written: the wrap and the home at {} are from different deployments. '
            'Re-copy log/ and blobs/ from the hub that issued this wrap.'.format(
                address, subject, log.home))
    return {'home': str(log.home), 'subject': subject, 'class': entitlement_class,
            'wrap': address, 'key_file': str(target), 'bodies_readable': True}


def recover_escrow(log, escrow_private: bytes, entitlement_class=FIRM_CLASS) -> bytes:
    """The class key, off the escrow wrap. Answers the raw key bytes.

    The lost laptop, and the lost hub: with the custodian's private half an admin gets the class key
    back in hand, re-enrolls whoever needs a seat and rewraps. Nothing is written here - the caller
    decides where the key goes, and `materialize(home, ESCROW_SUBJECT, escrow_private)` is the same
    recovery run as a write, since escrow rides an ordinary `key_wrapped` row under its reserved
    subject.

    A home that declared no escrow key and a wrong private key are ONE refusal, because from here
    they are one fact: no wrap in this store opens for escrow with what was handed in. Naming both
    is honest; picking one would require reading the policy declaration out of a body that, in the
    situation this verb exists for, is sealed under the key being recovered.
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

    Read at every use rather than held: a key destroyed at 14:02 is destroyed at 14:02, which is
    `seal.py`'s rule and is what crypto-shredding means. The refusal is custody's own, because the
    caller here is not trying to open a body - it is trying to give the key away, and a home that
    cannot read its own bodies has nothing to hand out.
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

    Located by ENVELOPE and read by body: only the three types that can move custody are opened, so
    the fold costs the length of the custody history rather than of the book. Latest wins in LSN
    order, which is what makes a re-enrollment supersede and a redeclared escrow key take over.

    A `key_wrapped` row is COUNTED only once the blob it cites is opened and found to be a wrap. The
    citation alone is not the entitlement: durability closure proves some bytes are on the platter at
    that address, not that they are a wrap, and a row citing anything else would suppress the reissue
    for that seat forever while this fold reported the subject current - the silent omission this
    module exists to refuse, wearing the report's own word for delivered. The AAD cannot be checked
    here (that needs a private key nobody in this process holds), so the check is the wrap's shape,
    its declared construction and its fields decoding, which is what distinguishes a wrap from a
    surface, a public key, or an operator's mistake.
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
    """Whether `address` resolves to something that is actually a wrap blob.

    Every refusal is one answer - absent, altered under its own name, oversized, not the four-field
    object, another construction, a field that will not decode - because the caller has one question:
    may this row be counted as a key delivered. Anything short of yes is no.
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
    """`(blob id, class key)` for the one wrap in `store` that opens for `subject`. Or refuses.

    A scan rather than a lookup, because the caller is by definition a holder that cannot read the
    bodies naming the wraps - see `materialize`. The AAD does the selecting: a blob addressed to
    another subject or another class does not come apart under this key, so trial-opening is exact
    rather than a heuristic, and the store's own walk order makes the answer deterministic.

    Two wraps opening to different keys is a refusal, not a choice. In this increment the class key
    does not rotate, so two openable wraps carry the same secret and the second is redundant; if
    they ever disagree, the home is holding blobs from two histories and picking one silently is how
    a replica ends up unable to say which record it is a replica of.
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

    The raw X25519 output is never used as a key directly: it is a curve point's coordinate, not a
    uniform string, and the KDF is what makes "32 bytes out of the exchange" mean the same thing to
    every implementation that ever reads these blobs.
    """
    return HKDF(algorithm=hashes.SHA256(), length=KEY_BYTES, salt=None,
                info=WRAP_INFO).derive(shared)


def _wrap_document(wrap_blob):
    """A wrap blob's four fields, or `CustodyRefusal` naming what it is instead.

    Closed at the field level like an event body: a fifth key would be bytes the AAD does not cover
    sitting in the middle of the recovery path.
    """
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
    """One hex field of a wrap blob, decoded and length-checked where the length is fixed."""
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
    """Key material of the width every key file in this home has."""
    if not isinstance(material, (bytes, bytearray)) or len(material) != KEY_BYTES:
        raise CustodyRefusal(
            '{} is {} bytes, not the {} a class key is: pass the raw key material '
            '(`custody.read_class_key`) rather than a path, a hex string or a file '
            'object.'.format(what, len(material) if isinstance(material, (bytes, bytearray)) else 0,
                             KEY_BYTES))
    return bytes(material)


def _secret_or_none(material):
    """A raw X25519 private key, or None - the scan's own check, which must not raise before it has
    a chance to say which wrap it was looking for."""
    if not isinstance(material, (bytes, bytearray)) or len(material) != X25519_BYTES:
        return None
    return bytes(material)


def _seat_private(material):
    """The seat's private key, as the object the exchange needs."""
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
    """A raw X25519 public key's bytes, checked."""
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
    """The recipient's public key, as the object the exchange needs."""
    return X25519PublicKey.from_public_bytes(_recipient_bytes(material, what))


def _subject(subject):
    """A subject reference: a non-empty string, and nothing else is a name."""
    return _text(subject, 'the subject reference')


def _text(value, what):
    if not isinstance(value, str) or value == '':
        raise CustodyRefusal(
            '{} is {!r}, and a name that names nothing is not a name: pass the pseudonymous '
            'reference the log carries.'.format(what, value))
    return value


def _stage_secret(path, material):
    """Write `material` beside `path` as scratch and answer where. Fsynced, and 0600 where the
    platform has an opinion.

    Scratch first so that the move into place is the last thing that happens: a refused append must
    not leave a seat holding a private key the record never published.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.parent / '{}.{}'.format(path.name, os.urandom(8).hex())
    _write_secret(scratch, material)
    return scratch


def _write_class_key(path, material):
    """The class key into a home that lacks one. The exclusive create IS the no-overwrite rule - a
    race loses the same way a retype does."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _write_secret(path, material)
    except (IOError, OSError) as blocked:
        raise CustodyRefusal(
            '{} could not be written ({}): if the file appeared between the check and the write, '
            'another recovery got there first and this one changes nothing - read the home\'s '
            'status and verify it entitled.'.format(path, blocked))


def _write_secret(path, material):
    """One key file on the platter, durable, and as private as the platform admits.

    `xb` and not `wb`: exclusive creation is how this module refuses to overwrite, and BINARY is not
    optional - a key is 32 bytes of noise, about one in eight of them carries a `0x0A`, and a text
    handle on Windows would write that byte as two. A key file that is a byte longer than a key is
    a recovery that hands back something no body was ever sealed under.
    """
    with open(str(path), 'xb') as handle:
        handle.write(material)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.chmod(str(path), 0o600)
    except (IOError, OSError):
        # Best effort, as in seal.py: Windows ACLs are not this mode, and a key the owner can read
        # is the posture either way.
        pass


def _discard(path):
    """Remove a key file that never became one. Scratch has no address and nothing can cite it, so
    this is hygiene rather than retention - the record is untouched either way."""
    try:
        os.unlink(str(path))
    except (IOError, OSError):
        pass
