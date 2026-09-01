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

"""Re-deriving a home from its own bytes - the replica's posture, run locally.

Nothing here reads an index, a manifest or a cached head: every hash is recomputed from the line on
the platter, and the head reported is the one the recomputation arrived at. There are two modes,
and the report states which one ran.

CHAIN-ONLY is the unentitled replica: bodies stay sealed. It recomputes the ciphertext hash off the
base64, the event hash over (ciphertext_hash, idempotency_tag, prev_hash, record_time), the prev
linkage and the dense LSN sequence from genesis. It cannot check a checkpoint, whose signature
lives in a body it cannot open, so that field reports `not assessed` in words rather than a zero a
caller would read as a count.

ENTITLED adds the plaintext half: every body opened under the AAD rebuilt from its own envelope
(which catches an edited actor or effective_time, since the seal authenticates the envelope it does
not encrypt), the interior `content_hash` recomputed, every idempotency tag recomputed where the
blind key is present, and every checkpoint signature checked against the verifying key read out of
a policy blob in the log rather than out of `keys/`.

The verifying key is a ladder, not a single value. Rotation is a logged event, so each
`checkpoint_verifying_key` declaration is recorded with the LSN it landed at and each checkpoint is
checked under the key in force at or before its own position - reading the newest declaration
instead would let one appended row retro-invalidate every checkpoint before it. A declaration
naming no blob is not a rung and does not empty the ladder.

Referential closure is asked of the whole history: the manifest is a walk of the blob store, and
any blob a body cites must be in it. A blob class reduces only through a logged retention event, so
a citation that no longer resolves is a blob that went silently.

A home whose class key was destroyed raises `SealedBodyUnreadable` under `entitled=True` and still
verifies its chain under `entitled=False`. That is crypto-shredding, not tampering.
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

#: What `checkpoints_verified` carries when the bodies are sealed - a sentence rather than a zero a
#: caller would read as a count.
NOT_ASSESSED = 'not assessed'


def verify_home(home, entitled=True):
    """Verify every hash, link and signature in `home` and return the report.

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
            # Bodies are not re-validated against today's vocabulary - a v1 event must stay
            # verifiable under a v(n+1) writer - so only these two shapes are read.
            if frame['event_type'] == 'checkpoint':
                checkpoints.append((lsn, payload))
            elif (frame['event_type'] == 'policy_declared' and isinstance(payload, dict)
                    and payload.get('policy') == VERIFYING_KEY_POLICY
                    and is_hash(payload.get('blob'))):
                # A rung of the ladder at the position it was declared. A declaration naming no
                # blob is not a rung and does not remove the rungs already on it.
                declarations.append((lsn, payload['blob']))
            citations.extend((lsn, frame['event_type'], field, digest)
                             for field, digest in cited_blobs(frame['event_type'], payload))

        previous = frame['event_hash']
        events = lsn
        heads[lsn] = frame['event_hash']

    verified = NOT_ASSESSED
    if entitled:
        # Checkpoints before the manifest: the published verifying key is itself a citation, so a
        # home that lost it hears which assertion it can no longer make.
        verified = _checkpoints(log, checkpoints, declarations, heads)
        _closure(log, citations)
    return {'mode': 'entitled' if entitled else 'chain-only', 'events': events,
            'checkpoints_verified': verified, 'head_lsn': events, 'head_hash': previous}


def _interior(log, frame, blinded):
    """Open one body, check the interior binding and (when `blinded`) the tag, and return the
    payload.

    The seal covers the envelope as additional data, so a failure here is an altered frame in
    either half. It raises `ChainBroken` naming the LSN, not `SealedBodyUnreadable`: the key is
    known present, so this is tampering rather than entitlement.
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
    """Verify every checkpoint signature and return the count.

    Three assertions per checkpoint: the signature verifies, it covers a position this log actually
    reached, and it is checked under the key in force at its own LSN, so a later rotation moves only
    the checkpoints after it.
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

    `declarations` is in LSN order, so the last rung not above `lsn` is the key that signed it.
    """
    blob = None
    for declared_at, candidate in declarations:
        if declared_at > lsn:
            break
        blob = candidate
    return blob


def _closure(log, citations):
    """Resolve every blob `citations` names against a walk of the store, raising
    `MissingBlobRefusal` on the first that does not resolve.

    Referential closure is a property of the whole history, not only of the moment each event was
    written, and a blob class reduces only through a logged retention event.
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
