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

"""Re-deriving a home from its own bytes - the replica's posture, run locally, trusting nothing.

The record never trusts what it can re-derive, and this is where that law is cashed. Nothing here
reads an index, a manifest or a cached head: every hash is recomputed from the line on the platter,
and the head this reports is the one the recomputation arrived at rather than the one the log
claims.

Two modes, because there are two kinds of holder, and the difference is stated in the report rather
than hidden in it.

CHAIN-ONLY is the unentitled replica: bodies stay sealed and are never opened. It still recomputes
every link - the ciphertext hash off the base64, the event hash over (ciphertext_hash,
idempotency_tag, prev_hash, record_time), the prev linkage and the dense LSN sequence from
genesis - so a replica that will never hold a key still proves it holds the same history. What it
CANNOT do is check a checkpoint: the signature lives in a body it cannot open. So the report says
`not assessed` in that field, in words, rather than a zero a script would read as a count. A
verification that quietly skipped something is worse than one that refused.

ENTITLED adds the plaintext half: every body opened under the AAD rebuilt from its own envelope -
which is what catches an edited actor or effective_time, since the envelope is authenticated by the
seal even though it is not encrypted - the interior `content_hash` recomputed over the envelope
plus the decrypted payload, every idempotency tag recomputed where the blind key is present, and
every checkpoint signature checked against the verifying key read out of a policy BLOB in the log.
Out of the blob, not out of `keys/`: that is the assertion a replica makes, and running the same
code here is what keeps it exercised.

The verifying key is a LADDER, not a single value, and this is the difference between a design that
admits rotation and one that only says it does. Key rotation is itself a logged event, so every
`checkpoint_verifying_key` declaration is recorded with the LSN it landed at and each checkpoint is
checked under the key in force AT OR BEFORE its own position - genesis's key for genesis's
checkpoint, forever. Reading the newest declaration instead would let one ordinary appended row
retro-invalidate every checkpoint written before it, which is a book of record destroyed by a legal
event; and a declaration naming no blob names no key, so it neither enters the ladder nor empties
it.

Referential closure is asked of the whole history rather than only of the moment each event was
written: the manifest is a walk of the blob store, and any blob a body says lives there must be in
it. That is where the retention law is cashed - a class reduces through a logged retention event
and never silently, so a citation that no longer resolves is a blob that went quietly.

A home whose class key was destroyed is not broken and does not verify entitled: it refuses with
`SealedBodyUnreadable` naming the key, and its chain still verifies with `entitled=False`. That is
crypto-shredding working, and it is a different sentence from tampering.
"""
import base64
import binascii
import hashlib

from .canon import canonical_bytes, content_hash
from .errors import (
    ChainBroken, CheckpointInvalid, MissingBlobRefusal, SealedBodyUnreadable)
from .genesis import VERIFYING_KEY_POLICY
from .log import GENESIS_PREV, SpineLog, check_frame, event_hash, semantic_tuple
from .seal import Keys
from .vocabulary import cited_blobs, is_hash

#: What `checkpoint_verified` carries when the bodies are sealed. A sentence rather than a zero:
#: zero is a number a caller would compare against and believe.
NOT_ASSESSED = 'not assessed'


def verify_home(home, entitled=True):
    """Verify every hash, link and signature in `home` and answer the report.

    Raises `ChainBroken` or `CheckpointInvalid` naming the LSN where the recomputation parts
    company with the bytes; `SealedBodyUnreadable` when an entitled verification is asked of a home
    that no longer holds its class key.
    """
    log = SpineLog(home)
    keys = log.keys
    if entitled and not keys.has_firm():
        raise SealedBodyUnreadable(
            'keys/class_firm.key is not in {}, so this home cannot open its own bodies: verify it '
            'with entitled=False (`DV_Spine verify --chain-only`) to re-derive the chain over '
            'ciphertext, and read the bodies on a replica that still holds the key'.format(
                log.home))
    blinded = entitled and keys.has_blind()

    previous = GENESIS_PREV
    events = 0
    heads = {}
    checkpoints = []
    declarations = []
    citations = []

    for frame in log.frames():
        check_frame(frame, str(log.log))
        lsn = frame['lsn']
        if lsn != events + 1:
            raise ChainBroken(
                'LSN {} follows LSN {}: the sequence is dense from genesis, so this is a lost or '
                'reordered line - restore the segment from a verified replica'.format(lsn, events))
        if frame['prev_hash'] != previous:
            raise ChainBroken(
                'LSN {}: prev_hash {} does not link to LSN {} ({}) - the chain parts company here; '
                'restore from a verified replica rather than repairing in place'.format(
                    lsn, frame['prev_hash'], events, previous))
        try:
            sealed = base64.b64decode(frame['body'])
        except (binascii.Error, TypeError, ValueError):
            raise ChainBroken(
                'LSN {}: the body is not base64 - the line has been altered; restore the frame '
                'from a verified replica'.format(lsn))
        recomputed = event_hash(hashlib.sha256(sealed).hexdigest(), frame['idempotency_tag'],
                                frame['prev_hash'], frame['record_time'])
        if recomputed != frame['event_hash']:
            raise ChainBroken(
                'LSN {}: the event hash recomputes to {}, not the {} the frame carries - one of '
                'the ciphertext, the tag, the link or the record time has been altered; restore '
                'the frame from a verified replica'.format(
                    lsn, recomputed, frame['event_hash']))

        if entitled:
            payload = _interior(log, frame, blinded)
            # The two body shapes verification itself reads. Bodies are NOT re-validated against
            # today's vocabulary - a v1 event must stay verifiable under a v(n+1) writer - so what
            # is checked here is only that these two can be read at all.
            if frame['event_type'] == 'checkpoint':
                checkpoints.append((lsn, payload))
            elif (frame['event_type'] == 'policy_declared' and isinstance(payload, dict)
                    and payload.get('policy') == VERIFYING_KEY_POLICY
                    and is_hash(payload.get('blob'))):
                # A rung of the ladder, at the position it was declared. A declaration naming no
                # blob names no key: it is not a rung, and it does not cost the ladder the rungs
                # already on it - one open-bodied policy row must never be able to unverify a home.
                declarations.append((lsn, payload['blob']))
            citations.extend((lsn, frame['event_type'], field, digest)
                             for field, digest in cited_blobs(frame['event_type'], payload))

        previous = frame['event_hash']
        events = lsn
        heads[lsn] = frame['event_hash']

    verified = NOT_ASSESSED
    if entitled:
        # Checkpoints before the manifest: the published verifying key is itself a citation, and a
        # home that has lost it should hear which assertion it can no longer make rather than a
        # sentence about a file.
        verified = _checkpoints(log, checkpoints, declarations, heads)
        _closure(log, citations)
    return {'mode': 'entitled' if entitled else 'chain-only', 'events': events,
            'checkpoints_verified': verified, 'head_lsn': events, 'head_hash': previous}


def _interior(log, frame, blinded):
    """Open one body and check what it binds. Answers the payload.

    The seal covers the envelope as additional data, so a failure here is an altered frame in
    EITHER half - and it is `ChainBroken` naming the LSN rather than `SealedBodyUnreadable`,
    because the key is known to be present: this is tampering, not entitlement.
    """
    lsn = frame['lsn']
    try:
        interior = log.open_interior(frame)
    except SealedBodyUnreadable as unreadable:
        raise ChainBroken(
            'LSN {}: {} - the body or one of the nine envelope fields it is sealed against has '
            'been altered'.format(lsn, unreadable))
    payload = interior['payload']
    semantic = semantic_tuple(frame['event_type'], payload, frame['actor'], frame['book'],
                              frame['effective_time'])
    rebuilt = content_hash(semantic)
    if rebuilt != interior['content_hash']:
        raise ChainBroken(
            'LSN {}: the interior binding is {} but the envelope and payload canonicalise to {} - '
            'the plaintext does not match what was sealed with it; restore the frame from a '
            'verified replica'.format(lsn, interior['content_hash'], rebuilt))
    if blinded:
        tag = log.keys.blind_tag(canonical_bytes(semantic))
        if tag != frame['idempotency_tag']:
            raise ChainBroken(
                'LSN {}: the idempotency tag recomputes to {}, not the {} the envelope carries - '
                'the tag, the body or an envelope field has been altered; restore the frame from '
                'a verified replica'.format(lsn, tag, frame['idempotency_tag']))
    return payload


def _checkpoints(log, checkpoints, declarations, heads):
    """Every checkpoint signature, against the key the log published for its position. Answers the
    count.

    Three things are asserted, and the last two matter as much as the first: the signature verifies,
    it covers a position this log actually has - a valid signature over a head nobody here ever
    reached is a checkpoint from another history - and it is checked under the key IN FORCE AT ITS
    OWN LSN, so a later rotation moves the checkpoints after it and leaves every earlier one
    verifying under the key that actually signed it.
    """
    if not checkpoints:
        return 0
    if not declarations:
        raise CheckpointInvalid(
            'this log carries {} checkpoint(s) but no {} policy declaration: the verifying key is '
            'published at genesis so that authenticity can be asserted locally - a log without it '
            'cannot be checked against anything'.format(len(checkpoints), VERIFYING_KEY_POLICY))
    published = {}
    for lsn, body in checkpoints:
        if not isinstance(body, dict) or not isinstance(body.get('lsn'), int) \
                or not isinstance(body.get('event_hash'), str) \
                or not isinstance(body.get('signature'), str):
            raise CheckpointInvalid(
                'the checkpoint at LSN {} does not carry an (lsn, event_hash, signature) body: '
                'there is nothing here to verify authenticity against - compare this home against '
                'a replica'.format(lsn))
        covered, claimed = body['lsn'], body['event_hash']
        actual = GENESIS_PREV if covered == 0 else heads.get(covered)
        if covered >= lsn or actual is None or actual != claimed:
            raise CheckpointInvalid(
                'the checkpoint at LSN {} covers (lsn {}, {}), which this log does not have at '
                'that position ({}): the checkpoint belongs to another history - compare this '
                'home against a replica'.format(
                    lsn, covered, claimed, actual if actual is not None else 'no such LSN'))
        blob = _key_in_force(declarations, lsn)
        if blob is None:
            raise CheckpointInvalid(
                'the checkpoint at LSN {} stands before any {} declaration in this log (the first '
                'is at LSN {}): a checkpoint signed before its key was published cannot be checked '
                'against anything - compare this home against a replica'.format(
                    lsn, VERIFYING_KEY_POLICY, declarations[0][0]))
        if blob not in published:
            try:
                published[blob] = log.store.get(blob)
            except MissingBlobRefusal:
                raise CheckpointInvalid(
                    'the verifying key blob {} this log publishes for LSN {} is not in the store '
                    'at {}: checkpoint authenticity cannot be asserted here - pull the blob from '
                    'any replica that holds it, it is self-verifying by hash'.format(
                        blob, lsn, log.store.blobs))
        Keys.verify_checkpoint(
            canonical_bytes({'event_hash': claimed, 'lsn': covered}), body['signature'],
            published[blob], where='the checkpoint at LSN {}'.format(lsn))
    return len(checkpoints)


def _key_in_force(declarations, lsn):
    """The verifying key blob standing at or before `lsn`, or None if none does.

    A fold, spelled out: the ladder is short and in LSN order, so the last rung not above the
    checkpoint is the key that signed it.
    """
    blob = None
    for declared_at, candidate in declarations:
        if declared_at > lsn:
            break
        blob = candidate
    return blob


def _closure(log, citations):
    """Every blob the log says lives in the store, resolved against a walk of the store itself.

    Referential closure is a property of the WHOLE history, not only of the moment an event was
    written. The writer refuses an event whose blob is not yet on the platter; this asks the same
    question of a log nobody here wrote, and it is also where the retention law is cashed - any
    blob class reduces through an explicit logged retention event and NEVER silently, so a citation
    that no longer resolves is exactly the silent expiry the law forbids.
    """
    if not citations:
        return
    manifest = frozenset(log.store.walk())
    for lsn, event_type, field, digest in citations:
        if digest not in manifest:
            raise MissingBlobRefusal(
                'LSN {}: the {} cites blob {} as its {}, and the manifest under {} does not hold '
                'it - either the blob went without the logged retention event the record requires, '
                'or this replica has not pulled it yet; a blob is self-verifying by hash, so pull '
                'it from any replica that holds it'.format(
                    lsn, event_type, digest, field, log.store.blobs))
