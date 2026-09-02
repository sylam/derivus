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

"""The single writer - one chained, fsynced, append-only sequence of facts, none of them editable.

A line is a frame: a firm-visible envelope over a sealed body, carrying exactly the twelve
`FRAME_FIELDS` (`check_frame` refuses a thirteenth) and bound to the body as GCM additional data
over `AAD_FIELDS`, so no envelope field moves without the body refusing to open. Uniqueness is
enforced on `idempotency_tag`, never on a plaintext hash: a retry coalesces onto the event already
written, and a tag hit whose decrypted bytes differ refuses rather than dedups. One writer, claimed
rather than declared on `WRITER_LOCK`; reading and verifying never claim.

The three hashes, the append order and the capability activation rule are in
docs_src/developer/spine.md. `_authorize` and `_scan` carry the two rules local to this file.
"""
import base64
import binascii
import datetime
import hashlib
import json
import logging
import os
from pathlib import Path

from .canon import canonical_bytes
from .capability import (
    CAPABILITIES_POLICY, CAPABILITY_EVENTS, UNREADABLE, apply_event, build_state, denial_body,
    evaluate, parse_document, verb_for)
from .errors import (
    CapabilityDenied, ChainBroken, CollisionRefusal, HomeMissing, MalformedEvent,
    MissingBlobRefusal, SealedBodyUnreadable, WriterBusy)
from .seal import Keys
from .store import BlobStore
from .vocabulary import (
    FIRM_CLASS, RECOVERY, WRITER, WRITER_TYPES, cited_blobs, classify, is_hash, validate)

LOG = logging.getLogger(__name__)

#: UTC, microseconds, and the `Z` said out loud. Client clocks lie, so `record_time` is the
#: writer's own stamp; `effective_time` is when the fact is TRUE and may be older or newer.
TIME_FORMAT = '%Y-%m-%dT%H:%M:%S.%fZ'
#: What the first event chains onto. Sixty-four zeros is not a hash of anything - it is the edge.
GENESIS_PREV = '0' * 64
#: The one class phase 1 has. The envelope asks `vocabulary.classify` for a frame's class rather
#: than stamping this in: the class is derived from provenance, and this names what that derivation
#: can answer so far.
ENTITLEMENT_CLASS = FIRM_CLASS
EVENT_VERSION = 1
#: Roll past 64 MiB. Segments are an operational convenience - the chain does not know they exist.
SEGMENT_LIMIT = 64 * 1024 * 1024
SEGMENT_GLOB = 'segment-*.jsonl'
#: The writer's claim, at the home's root rather than inside `log/` - the log directory is what a
#: replica file-copies, and a lock is about this machine's processes, never about the history.
WRITER_LOCK = '.writer.lock'

#: The nine pre-LSN envelope fields the body is sealed against. Listed once so the writer and the
#: verifier build the same AAD; the order here is immaterial, since JCS sorts them.
AAD_FIELDS = ('actor', 'book', 'effective_time', 'entitlement_class', 'event_type',
              'event_version', 'idempotency_tag', 'prev_hash', 'record_time')
#: What a line must carry to be a frame at all.
FRAME_FIELDS = AAD_FIELDS + ('body', 'event_hash', 'lsn')


def now_stamp():
    """The writer's clock, in the one format the record spells time in."""
    return datetime.datetime.now(datetime.timezone.utc).strftime(TIME_FORMAT)


def check_time(value, field='effective_time'):
    """`value` asserted to be exactly `YYYY-MM-DDTHH:MM:SS.ffffffZ`. `field` names it in refusals.

    The parse must round-trip its own text: `strptime` would take one digit of fraction where the
    format says six, and two spellings of one instant are two hashes of one fact.
    """
    parsed = None
    if isinstance(value, str):
        try:
            parsed = datetime.datetime.strptime(value, TIME_FORMAT)
        except ValueError:
            parsed = None
    if parsed is None or parsed.strftime(TIME_FORMAT) != value:
        raise MalformedEvent(
            '{}: {!r} is not a timestamp of the form {} - stamp it in UTC with six digits of '
            'fraction, or pass None where the fact carries no truth-time of its own'.format(
                field, value, '2026-08-29T14:02:11.004917Z'))
    return value


def semantic_tuple(event_type, body, actor, book, effective_time):
    """What the event means, as the one object every hash of its meaning is taken over.

    Neither `record_time` nor `lsn` is here, so the same fact submitted twice is the same tuple and
    idempotency is a property of the fact rather than of when it arrived. `effective_time` is
    carried as null where the caller named none, never defaulted to the writer's clock.
    """
    return {'actor': actor, 'body': body, 'book': book, 'effective_time': effective_time,
            'type': event_type, 'version': EVENT_VERSION}


def check_frame(frame, where):
    """Assert `frame` carries all twelve `FRAME_FIELDS` and nothing else, raising `ChainBroken`
    naming `where`.

    The surplus half is the quieter law: the nine-field AAD, the event hash and the interior binding
    cover exactly these fields, so a thirteenth would be an unauthenticated channel into the record.
    """
    missing = [field for field in FRAME_FIELDS if field not in frame]
    if missing:
        raise ChainBroken(
            '{}: the frame has no {} - it is not a line this writer wrote; restore the segment '
            'from a verified replica'.format(where, ', '.join(missing)))
    surplus = sorted(set(frame) - set(FRAME_FIELDS))
    if surplus:
        raise ChainBroken(
            'LSN {}: the frame carries {} beyond the twelve fields a frame has ({}) - a field '
            'outside them is covered by no hash, no seal and no signature, so it is not a field '
            'this record admits; restore the line from a verified replica'.format(
                frame['lsn'], ', '.join(surplus), where))
    return frame


def aad_bytes(envelope):
    """`envelope`'s nine pre-LSN fields, canonicalised - GCM's additional data.

    One spelling, used by the writer that seals and the verifier that opens; two would be two
    systems disagreeing about what a body was bound to.
    """
    try:
        return canonical_bytes(dict((field, envelope[field]) for field in AAD_FIELDS))
    except KeyError as missing:
        raise ChainBroken(
            'the frame has no {} - a frame carries the nine envelope fields the body is sealed '
            'against; the line is not one this writer wrote'.format(missing))


def event_hash(ciphertext_hash, idempotency_tag, prev_hash, record_time):
    """The chain link: SHA-256 over the canonical (ciphertext_hash, idempotency_tag, prev_hash,
    record_time).

    Over the CIPHERTEXT hash, so a replica that will never hold a key still re-derives every link.
    """
    return hashlib.sha256(canonical_bytes({
        'ciphertext_hash': ciphertext_hash, 'idempotency_tag': idempotency_tag,
        'prev_hash': prev_hash, 'record_time': record_time})).hexdigest()


def as_of_key(frame):
    """`(effective_time, lsn)` - the as-of sort key. As-at order is LSN order and needs no key.

    A frame whose `effective_time` is null reads as of when it was recorded, resolved here at the
    reader rather than at the writer, which would put arrival inside the semantic tuple. LSN breaks
    the tie, so two facts true at one instant replay in the order they were written.
    """
    return (frame['effective_time'] or frame['record_time'], frame['lsn'])


def take_claim(handle):
    """Take an exclusive, non-blocking byte-range lock on `handle`, per platform.

    A lock the operating system releases when the process dies, rather than an `O_EXCL` sentinel
    file a crash would leave behind for an operator to clear by hand.
    """
    if os.name == 'nt':
        import msvcrt
        msvcrt.locking(handle, msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)


def drop_claim(handle):
    """Release what `take_claim` took. Best effort - closing the descriptor releases it anyway."""
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(handle, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_UN)
    except (IOError, OSError):
        pass


def parse_number(text):
    """`json.loads`' `parse_int` hook for canonical bytes: an integer literal past 2**53 comes back
    as a `float`, per canon.py's declared limitation."""
    value = int(text)
    return value if abs(value) <= 2 ** 53 else float(value)


class SpineLog:
    """One home's log, opened.

    Opening streams every segment, checks what the stored fields claim about themselves - LSN
    continuity and the prev-hash linkage - and builds the three indexes the writer needs: the head,
    the tag index idempotency is enforced on, and a byte offset per LSN. It does not recompute
    hashes; that is `verify_home`'s job, so verification stays something a replica does to a log
    rather than something a log says about itself.
    """

    def __init__(self, home):
        self._lock = None
        # The writer's own voice, off. Set only around the internal denial path, which is the one
        # place the reserved type may be spoken and the one append that is never gated.
        self._reserved = False
        self.home = Path(home)
        self.log = self.home / 'log'
        # log/ and blobs/, and NOT keys/: a replica holding no key still opens the log and verifies
        # its chain over ciphertext, and a missing key surfaces where a key is used.
        for part in ('log', 'blobs'):
            if not (self.home / part).is_dir():
                raise HomeMissing(
                    'there is no {}/ under {}: a spine home is log/ and blobs/ - run `DV_Spine init '
                    '--home {}` to mint one, or point at the home that already exists. keys/ is '
                    'what makes a home WRITABLE and is not needed to verify one'.format(
                        part, self.home, self.home))
        self.store = BlobStore(self.home)
        self.keys = Keys(self.home)
        self._scan()

    def __repr__(self):
        return 'SpineLog({!r}) at lsn {}'.format(str(self.home), self._head_lsn)

    def __del__(self):
        # A writer dropped rather than closed still lets go of the home. Interpreter shutdown can
        # have taken `os` away, and process exit releases the claim regardless.
        try:
            self.close()
        except Exception:
            pass

    def close(self):
        """Release this home's writer claim. Idempotent, and a no-op on a handle that only ever
        read, since reading never claims."""
        if self._lock is None:
            return
        drop_claim(self._lock)
        os.close(self._lock)
        self._lock = None

    @property
    def genesis_actor(self):
        """Who minted this home, read off LSN 1's envelope - what a checkpoint is attributed to
        when nobody says otherwise."""
        return self._genesis_actor

    def head(self):
        """`(lsn, event_hash)` - where the log stands. `(0, GENESIS_PREV)` when nothing is written
        yet, which is what the first event chains onto."""
        return (self._head_lsn, self._head_hash)

    def frames(self, start_lsn=1, end_lsn=None):
        """Every frame from `start_lsn` to `end_lsn`, in LSN order, read from disk.

        The segment listing is re-read too, because reading never claims the home and a handle held
        across someone else's append would otherwise walk the segments that existed when it opened.
        The only remembered value is a seen segment's lower bound, which cannot change once the
        segment holds a line.
        """
        seen = dict((path, index) for index, path in enumerate(self._segments))
        for path in sorted(self.log.glob(SEGMENT_GLOB)):
            index = seen.get(path)
            if index is not None and end_lsn is not None \
                    and self._segment_first[index] is not None \
                    and self._segment_first[index] > end_lsn:
                return
            with path.open('rb') as handle:
                for raw in handle:
                    if not raw.strip():
                        continue
                    frame = self._parse(raw, path)
                    if frame['lsn'] < start_lsn:
                        continue
                    if end_lsn is not None and frame['lsn'] > end_lsn:
                        return
                    yield frame

    def frame_at(self, lsn):
        """The one frame at `lsn`, read by byte offset. Raises `ChainBroken` if this log has no
        such position."""
        located = self._at.get(lsn)
        if located is None:
            raise ChainBroken(
                'this log has no LSN {} (it stands at {}): ask a replica that is further ahead, or '
                'read a position this log holds'.format(lsn, self._head_lsn))
        index, offset = located
        path = self._segments[index]
        with path.open('rb') as handle:
            handle.seek(offset)
            return self._parse(handle.readline(), path)

    def open_interior(self, frame):
        """The sealed interior of `frame`: `{"content_hash", "payload"}`.

        The AAD is rebuilt from the frame's own envelope, so a body that moved and an envelope that
        moved both raise `SealedBodyUnreadable` - GCM does not distinguish them.
        """
        try:
            sealed = base64.b64decode(frame['body'])
        except (binascii.Error, TypeError, ValueError):
            raise SealedBodyUnreadable(
                'LSN {}: the body is not base64 - the line has been altered; restore the frame '
                'from a verified replica'.format(frame.get('lsn')))
        plaintext = self.keys.open(sealed, aad_bytes(frame))
        try:
            interior = json.loads(plaintext.decode('utf-8'), parse_int=parse_number)
        except (ValueError, UnicodeDecodeError):
            raise SealedBodyUnreadable(
                'LSN {}: the body opened but is not JSON - it was sealed by something that is not '
                'this writer'.format(frame.get('lsn')))
        if not isinstance(interior, dict) or 'payload' not in interior \
                or 'content_hash' not in interior:
            raise SealedBodyUnreadable(
                'LSN {}: the opened body is not the {{content_hash, payload}} binding this writer '
                'seals - it was written by another implementation'.format(frame.get('lsn')))
        return interior

    def open_body(self, frame):
        """The event's body - the fact itself. Raises `SealedBodyUnreadable` where the class key is
        gone, which is the crypto-shredded state."""
        return self.open_interior(frame)['payload']

    def append(self, event_type, body, actor, book=None, effective_time=None, blob_refs=()):
        """Append one fact and return its envelope.

        The order is the design: the writer's claim on the home, validation, authorization, the tag,
        the duplicate check (which decrypts the stored event and byte-compares), referential closure
        over every blob cited, and only then a byte on the platter. Every refusal therefore leaves
        the head where it found it.

        `effective_time` is the caller's or null, never defaulted to `record_time` - the semantic
        tuple contains it, so a writer-stamped default would make the second submission of one
        booking a second fact. A null reads as of when it was recorded (`as_of_key`).
        """
        self._claim()
        validate(event_type, body)
        if not isinstance(actor, str) or actor == '':
            raise MalformedEvent(
                'actor is {!r}: every event carries the pseudonymous subject reference that '
                'submitted it - pass the actor rather than letting the record forget who '
                'spoke'.format(actor))
        if book is not None and (not isinstance(book, str) or book == ''):
            raise MalformedEvent(
                'book is {!r}: a book is a non-empty name or None for the firm-level facts - '
                'policy, checkpoints, official market declarations'.format(book))
        if effective_time is not None:
            check_time(effective_time)
        self._authorize(event_type, body, actor, book)
        record_time = now_stamp()

        canonical = canonical_bytes(semantic_tuple(event_type, body, actor, book, effective_time))
        content_hash = hashlib.sha256(canonical).hexdigest()
        tag = self.keys.blind_tag(canonical)

        held = self._tags.get(tag)
        if held is not None:
            return self._coalesce(held, tag, canonical)

        if isinstance(blob_refs, str):
            # A bare hash iterates as characters, so the refusal would otherwise name a letter.
            raise MalformedEvent(
                'blob_refs is the string {!r}: pass the hashes as a sequence - `blob_refs=(h,)` '
                'for one - so the writer checks addresses rather than characters'.format(blob_refs))
        # What the caller declares AND what the body itself cites. The second half cannot be
        # forgotten, so a submitter that omits `blob_refs` still cannot cite an absent blob.
        citations = [('blob_refs', reference) for reference in tuple(blob_refs)]
        citations.extend(cited_blobs(event_type, body))
        for field, reference in citations:
            if not self.store.has(reference):
                raise MissingBlobRefusal(
                    'blob {} is not in the store at {}, so the {} citing it as {} does not append: '
                    'durability ordering is law - put and fsync the blob first, then record the '
                    'fact that speaks of it'.format(
                        reference, self.store.blobs, event_type, field))

        envelope = {'actor': actor, 'book': book, 'effective_time': effective_time,
                    'entitlement_class': classify(event_type, book), 'event_type': event_type,
                    'event_version': EVENT_VERSION, 'idempotency_tag': tag,
                    'prev_hash': self._head_hash, 'record_time': record_time}
        sealed = self.keys.seal(
            canonical_bytes({'content_hash': content_hash, 'payload': body}), aad_bytes(envelope))
        frame = dict(envelope)
        frame['body'] = base64.b64encode(sealed).decode('ascii')
        frame['event_hash'] = event_hash(
            hashlib.sha256(sealed).hexdigest(), tag, envelope['prev_hash'], record_time)
        frame['lsn'] = self._head_lsn + 1

        line = json.dumps(frame, sort_keys=True, separators=(',', ':')).encode('utf-8') + b'\n'
        index, offset = self._write(line)

        self._at[frame['lsn']] = (index, offset)
        self._tags[tag] = frame['lsn']
        self._segment_last[index] = frame['lsn']
        if self._segment_first[index] is None:
            self._segment_first[index] = frame['lsn']
        self._head_lsn = frame['lsn']
        self._head_hash = frame['event_hash']
        if frame['lsn'] == 1:
            self._genesis_actor = actor
        if event_type in CAPABILITY_EVENTS and self._capability is not None:
            # The fold moves with the log rather than being re-read from it. Safe only because this
            # handle holds the home's claim, so nothing else can have appended since.
            apply_event(self._capability, event_type, actor, body, self.store)

        envelope['event_hash'] = frame['event_hash']
        envelope['lsn'] = frame['lsn']
        envelope['coalesced'] = False
        return envelope

    def _coalesce(self, lsn, tag, canonical):
        """The duplicate-tag path: return the stored event's envelope once it is proven the same
        fact, otherwise raise `CollisionRefusal`.

        Proof is a byte comparison against the stored event's canonical semantic bytes, rebuilt by
        decrypting it. A duplicate that cannot be proven identical - a stored body that will not
        open - is refused too.
        """
        stored = self.frame_at(lsn)
        try:
            payload = self.open_body(stored)
        except SealedBodyUnreadable as unreadable:
            raise CollisionRefusal(
                'idempotency tag {} is already held by LSN {}, whose stored body does not open '
                '({}): a duplicate that cannot be proven identical is refused rather than '
                'deduplicated - verify this home against a replica before writing to it '
                'again'.format(tag, lsn, unreadable))
        rebuilt = canonical_bytes(semantic_tuple(
            stored['event_type'], payload, stored['actor'], stored['book'],
            stored['effective_time']))
        if rebuilt != canonical:
            raise CollisionRefusal(
                'idempotency tag {} is already held by LSN {} over DIFFERENT canonical bytes ({} '
                'stored, {} offered): the writer byte-compares before it deduplicates, so this is '
                'a refusal and not a swap - keep both payloads out of band and raise the '
                'collision'.format(tag, lsn, len(rebuilt), len(canonical)))
        envelope = dict((key, value) for key, value in stored.items() if key != 'body')
        envelope['coalesced'] = True
        return envelope

    def _authorize(self, event_type, body, actor, book):
        """The enforcement hook: may this actor say this, and is what they are saying evaluable?

        Enforcement activates by declaration for the six document verbs; break-glass is gated from
        event one instead, no declaration granting it and so no declaration's absence opening it. A
        refusal is itself a fact - `capability_denied` under the writer's own name, appended before
        the raise.

        Three checks, in order: the reserved type, which must be refused whether or not a document
        exists, then scope, then the document itself, so a malformed policy is met at the moment it
        is declared.
        """
        if event_type in WRITER_TYPES and not self._reserved:
            raise CapabilityDenied(
                '{0} is the writer\'s own voice and no submitter appends one: a denial is a fact '
                'ABOUT a refusal, emitted by the writer under the actor {1!r} on the path that '
                'refuses - read the denials back off the log (they are ordinary chained events), '
                'and do not write one'.format(event_type, WRITER))
        if self._reserved:
            return
        doc, genesis = self._capability_state()
        verb = verb_for(event_type)
        # By declaration for the six document verbs, from event one for the RECOVERY handle: the
        # handle is not in a document, so "no document yet" cannot mean anybody may pull it.
        if (doc is not None or verb == RECOVERY) \
                and not evaluate(doc, genesis, actor, verb, book):
            scope = book if book is not None else '*'
            denial = self._deny(actor, verb, book, event_type)
            raise CapabilityDenied(
                'actor {0!r} holds no {1} scope over {2!r}, so the {3} does not append: {4}. The '
                'refusal is itself recorded at LSN {5}'.format(
                    actor, verb, scope, event_type,
                    self._why(doc, genesis, actor, verb, scope), denial['lsn']))
        if event_type == 'policy_declared' and isinstance(body, dict) \
                and body.get('policy') == CAPABILITIES_POLICY:
            blob = body.get('blob')
            if not is_hash(blob):
                raise CapabilityDenied(
                    'a {} declaration names {!r} as its blob: the document is bulk and lives in '
                    'the store, so the body carries its 64-hex address and nothing else - put the '
                    'canonical document with `BlobStore.put` and declare the hash it '
                    'answers'.format(CAPABILITIES_POLICY, blob))
            parse_document(self.store.get(blob),
                           'the capabilities document {}'.format(blob))

    def _why(self, doc, genesis, actor, verb, scope):
        """The middle of a denial message: what the record says, and what would change it.

        Three cases have three sentences - a stranger at the break-glass handle, a document that
        will not read, and a document that simply grants nothing here.
        """
        if verb == RECOVERY:
            return (
                'the break-glass seat this home named at genesis is {!r}, and that grant is the '
                'whole of who may reach for the handle - it lives outside every document, so no '
                'document widens it either. Have the seat genesis named append the use'.format(
                    genesis.get('break_glass')))
        if doc is UNREADABLE:
            return (
                'the capabilities document in force here will not READ - its blob was altered under '
                'its own address, or is gone from the store - so it grants nothing to anybody. '
                'Restore blobs/ from a verified replica, or have the genesis break-glass seat append '
                '`break_glass_used` and declare a replacement through `DV_Spine grant --file`')
        return (
            'the capabilities document in force here grants it nothing that reaches this event. '
            'Declare a document granting ({!r}, {}, {!r}) through `DV_Spine grant --file`, or - if '
            'a declaration stranded the last admin - recover through the break-glass seat genesis '
            'named'.format(actor, verb, scope))

    def _capability_state(self):
        """`(document, genesis)` in force at this head, folded once and then kept current.

        Built the first time an append needs it and updated as later ones land, so a writer does not
        pay the length of the book per write. Safe to cache because an append holds the home's
        exclusive claim and `_claim` re-scans.

        It cannot fail: a declared document whose blob no longer answers for it folds to `UNREADABLE`
        rather than raising, since an exception here would take `break_glass_used` and the
        replacement declaration with it.
        """
        if self._capability is None:
            self._capability = build_state(self)
        return (self._capability['doc'], self._capability['genesis'])

    def _deny(self, subject, verb, book, attempted_type):
        """Append the refusal as a fact under the writer's own name, and return its envelope.

        The one path that may speak the reserved type, and the one append that is never gated. A
        second identical denial coalesces onto the first by the ordinary tag rule.
        """
        self._reserved = True
        try:
            return self.append('capability_denied',
                               denial_body(subject, verb, book, attempted_type), actor=WRITER)
        finally:
            self._reserved = False

    def _claim(self):
        """Take this home's writer claim and re-read the head under it. Idempotent; raises
        `WriterBusy` when another writer holds the home.

        Taken at the first append rather than at construction, since reading and verifying are not
        writing. The re-read matters because another writer may have appended between construction
        and the claim, and an LSN assigned off a stale head is the one defect with no repair.
        """
        if self._lock is not None:
            return
        path = self.home / WRITER_LOCK
        handle = os.open(str(path), os.O_CREAT | os.O_RDWR)
        try:
            take_claim(handle)
        except (IOError, OSError):
            os.close(handle)
            raise WriterBusy(
                '{} is held by another writer: one deployment, one log, one writer - a second one '
                'assigns an LSN that is already taken, and nothing in this package may edit or '
                'remove the line that results; close the other SpineLog (`log.close()`) or stop '
                'the other process, then append again'.format(path))
        self._lock = handle
        self._scan()

    def _scan(self):
        """Open the log: stream the segments, check what the stored fields claim, index them.

        The torn tail is the one repair in the package. An unterminated final line of the last
        segment was interrupted mid-write - the newline is the last byte a write puts down - and was
        never fsynced, so it is truncated. Garbage inside a terminated line is a durable line that
        was altered, so that raises `ChainBroken` rather than being healed away.
        """
        self._segments = sorted(self.log.glob(SEGMENT_GLOB))
        self._segment_first = [None] * len(self._segments)
        self._segment_last = [0] * len(self._segments)
        self._at = {}
        self._tags = {}
        # The capability fold is dropped rather than carried across a scan that replaced its
        # inputs; the next append that needs it rebuilds it from the platter.
        self._capability = None
        self._head_lsn = 0
        self._head_hash = GENESIS_PREV
        self._genesis_actor = None
        for index, path in enumerate(self._segments):
            last_segment = index == len(self._segments) - 1
            data = path.read_bytes()
            position = 0
            truncate_at = None
            while position < len(data):
                newline = data.find(b'\n', position)
                chunk = data[position:] if newline == -1 else data[position:newline]
                end = len(data) if newline == -1 else newline + 1
                if chunk.strip():
                    frame = self._loads(chunk)
                    if frame is None:
                        if last_segment and newline == -1:
                            truncate_at = position
                            break
                        raise ChainBroken(
                            '{} at byte {} is not JSON, where LSN {} should be: a line inside the '
                            'log was altered or lost - restore the segment from a verified '
                            'replica'.format(path.name, position, self._head_lsn + 1))
                    self._accept(frame, index, position, path)
                position = end
            if truncate_at is not None:
                LOG.warning('%s: truncating %d torn byte(s) at %d - a line interrupted mid-write, '
                            'never fsynced, never chained onto', path.name,
                            len(data) - truncate_at, truncate_at)
                os.truncate(str(path), truncate_at)
            self._segment_last[index] = self._head_lsn
            if self._segment_first[index] is None:
                self._segment_first[index] = self._head_lsn + 1

    def _accept(self, frame, index, offset, path):
        """Check one frame's stored fields against the position the log has reached, then index it
        and advance the head."""
        check_frame(frame, '{} at byte {}'.format(path.name, offset))
        expected = self._head_lsn + 1
        if frame['lsn'] != expected:
            raise ChainBroken(
                '{} at byte {}: LSN {} follows LSN {} - the sequence is positional and dense, so a '
                'gap means a lost or reordered line; restore the segment from a verified '
                'replica'.format(path.name, offset, frame['lsn'], self._head_lsn))
        if frame['prev_hash'] != self._head_hash:
            raise ChainBroken(
                'LSN {}: prev_hash {} does not link to LSN {} ({}) - the chain is broken here; '
                'restore from a verified replica rather than repairing in place'.format(
                    frame['lsn'], frame['prev_hash'], self._head_lsn, self._head_hash))
        self._at[frame['lsn']] = (index, offset)
        # First writer wins, so a coalesce points at the event that was actually first.
        self._tags.setdefault(frame['idempotency_tag'], frame['lsn'])
        if self._segment_first[index] is None:
            self._segment_first[index] = frame['lsn']
        self._head_lsn = frame['lsn']
        self._head_hash = frame['event_hash']
        if frame['lsn'] == 1:
            self._genesis_actor = frame['actor']

    def _parse(self, raw, path):
        """One line read back as a frame, or `ChainBroken` naming the segment. Reading is where a
        doctored file is met, so this refuses rather than skipping."""
        frame = self._loads(raw)
        if frame is None:
            raise ChainBroken(
                '{}: a line will not parse as JSON - the segment was altered after it was opened; '
                'restore it from a verified replica'.format(path.name))
        return frame

    @staticmethod
    def _loads(chunk):
        """`chunk` parsed as a frame, or None where the bytes are not a JSON object."""
        try:
            frame = json.loads(chunk.decode('utf-8'))
        except (ValueError, UnicodeDecodeError):
            return None
        return frame if isinstance(frame, dict) else None

    def _write(self, line):
        """Put one line on the platter and make it durable. Returns `(segment index, offset)`.

        `flush` then `fsync`: the event is not appended until the bytes are on the device, since
        durability ordering, the chain and the checkpoint are all worth what this call is worth.
        """
        index, path = self._segment_for(len(line))
        size = path.stat().st_size if path.exists() else 0
        payload = line
        offset = size
        if size and not self._ends_with_newline(path):
            # A durable frame that never got its terminator: start a new line rather than glue
            # this one onto it.
            payload = b'\n' + line
            offset = size + 1
        with path.open('ab') as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        return (index, offset)

    def _segment_for(self, length):
        """`(index, path)` of the segment a line of `length` bytes lands in, rolling a new one past
        `SEGMENT_LIMIT`. Rolling is bookkeeping: the chain runs across segments untouched."""
        if not self._segments:
            return self._roll()
        index = len(self._segments) - 1
        path = self._segments[index]
        size = path.stat().st_size if path.exists() else 0
        if size and size + length > SEGMENT_LIMIT:
            return self._roll()
        return (index, path)

    def _roll(self):
        """Start the next `segment-XXXXXXXX.jsonl`, numbered from one, and return `(index, path)`."""
        path = self.log / 'segment-{:08d}.jsonl'.format(len(self._segments) + 1)
        self._segments.append(path)
        self._segment_first.append(None)
        self._segment_last.append(self._head_lsn)
        return (len(self._segments) - 1, path)

    @staticmethod
    def _ends_with_newline(path):
        """Whether the last byte of `path` is a newline - whether its final line was terminated."""
        with path.open('rb') as handle:
            handle.seek(-1, os.SEEK_END)
            return handle.read(1) == b'\n'
