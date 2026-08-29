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

"""The single writer - one chained, fsynced, append-only sequence of facts, and no way to edit one.

A line is a FRAME: a firm-visible envelope and a sealed body, and the whole design is in how the
three hashes divide the work.

`content_hash` is SHA-256 over the canonical semantic tuple (type, version, effective_time, actor,
book, body) - what the event MEANS, independent of when it was written or what it chained onto. It
lives inside the sealed body, because in the envelope it would be a dictionary oracle.

`idempotency_tag` is that same canonical plaintext under HMAC, and it is what the envelope carries
and what uniqueness is enforced on. A retry of a booking meets its own tag and coalesces onto the
event already written; a tag hit whose stored bytes DIFFER is a named refusal, never a dedup, so a
cryptanalytic surprise degrades into a loud stop rather than a silent swap. The writer proves
identity by decrypting the stored body and byte-comparing - it holds every key, so it can - and a
duplicate it cannot prove identical is refused rather than assumed.

`event_hash` is SHA-256 over (ciphertext_hash, idempotency_tag, prev_hash, record_time): the chain,
each event committing to its predecessor from genesis, computed over the CIPHERTEXT so that an
unentitled replica verifies the entire history holding no plaintext at all. LSN is positional and
sits outside both hashes - it is where the event is, not what it is.

The envelope is bound to the body by GCM's additional data: nine pre-LSN fields, canonicalised. So
the plaintext half of a frame is tamper-evident too - edit an actor and the body stops opening -
and there is no field of a frame that can move quietly. That sentence is only true if a frame
carries the twelve fields and NO THIRTEENTH: a surplus field would sit outside every hash, seal and
signature while every projector downstream read it, so `check_frame` refuses one exactly as the
vocabulary refuses a surplus key in a body.

The writer also ENFORCES, and it enforces by declaration. Until a capabilities document is in the
log there is nothing to consult and every append lands as it always did; once one is in force, an
append whose actor lacks the event's verb is refused AND the refusal is appended as an ordinary
fact, because a decision is a fact and a denial recorded only in someone's log file is a decision
nobody can replay. Two authorizations sit outside that rule in opposite directions: the writer's own
denial is never gated, or a refusal could itself be refused; and the break-glass handle is gated from
event one rather than by declaration, because no document grants it and so no document's absence can
open it. The evaluation itself is not here - it is `capability.evaluate`, one pure function over a
fold this writer keeps current - so what lives in this file is the hook, and the one path allowed to
speak the writer's own reserved type.

One writer, and it is claimed rather than declared. A second `SpineLog` appending to the same home
would assign an LSN the first has already taken, and nothing in this package may edit or remove the
line that results - so the first append takes an exclusive claim on the home, re-reads the head
under it, and a second writer is `WriterBusy` rather than a corrupted book. Reading and verifying
never claim: a replica must not be locked out of its own log by whoever is writing to it.

Nothing is edited in place. There is exactly one exception in this file, and it is not a repair: a
final line with no terminating newline was interrupted mid-write and was never durable, so it is
truncated at open. Only UNTERMINATED trailing bytes, only in the last segment - a line that carries
its newline is a line the writer finished and fsynced, so garbage inside one is a durable line that
was altered, which is `ChainBroken` like every other defect. A log that heals itself is a log that
can be made to lie, and a log that heals itself during a read-only verification destroys the
evidence on its way past.
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
#: One class in phase 1. The mechanism ships dormant; the day desk two arrives this is a
#: classification decision recorded in policy, not a redesign - which is why the envelope asks
#: `vocabulary.classify` for the class rather than stamping this constant into the frame: the class
#: is DERIVED from provenance, and this is the name of the one class that derivation can answer yet.
ENTITLEMENT_CLASS = FIRM_CLASS
EVENT_VERSION = 1
#: Roll past 64 MiB. Segments are an operational convenience - the chain does not know they exist.
SEGMENT_LIMIT = 64 * 1024 * 1024
SEGMENT_GLOB = 'segment-*.jsonl'
#: The writer's claim, at the home's root rather than inside `log/` - the log directory is what a
#: replica file-copies, and a lock is about this machine's processes, never about the history.
WRITER_LOCK = '.writer.lock'

#: The nine pre-LSN envelope fields, in the order they are named nowhere - JCS sorts them - but
#: listed once so the writer and the verifier build the same AAD from the same tuple.
AAD_FIELDS = ('actor', 'book', 'effective_time', 'entitlement_class', 'event_type',
              'event_version', 'idempotency_tag', 'prev_hash', 'record_time')
#: What a line must carry to be a frame at all.
FRAME_FIELDS = AAD_FIELDS + ('body', 'event_hash', 'lsn')


def now_stamp():
    """The writer's clock, in the one format the record spells time in."""
    return datetime.datetime.now(datetime.timezone.utc).strftime(TIME_FORMAT)


def check_time(value, field='effective_time'):
    """Refuse anything that is not exactly `YYYY-MM-DDTHH:MM:SS.ffffffZ`.

    Exactly: `strptime` would take one digit of fraction where the format says six, and two
    spellings of one instant are two hashes of one fact, so the parse must round-trip its own text.
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
    """What the event MEANS, as the one object every hash of its meaning is taken over.

    Neither `record_time` nor `lsn` is here: the same fact submitted twice is the same tuple, which
    is what makes idempotency a property of the fact rather than of the moment it arrived. Which is
    also why `effective_time` is carried as null when the caller named none rather than defaulted
    to the writer's clock - a default would smuggle the moment of arrival back inside the meaning.
    """
    return {'actor': actor, 'body': body, 'book': book, 'effective_time': effective_time,
            'type': event_type, 'version': EVENT_VERSION}


def check_frame(frame, where):
    """Every field a frame carries is one of the twelve, and all twelve are there.

    The presence half is what makes a line a frame at all. The SURPLUS half is the quieter law: the
    nine-field AAD, the event hash and the interior binding cover exactly the fields named here, so
    a thirteenth would be an unauthenticated channel into the record - read by every projector
    downstream, covered by no hash, no seal and no signature, and precisely where the erasable
    attributes the brief confines to the side table would reappear. The vocabulary refuses a surplus
    key in a body for the same reason; the envelope does not get to be the exception.
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
    """The nine pre-LSN envelope fields, canonicalised - GCM's additional data.

    One spelling, used by the writer that seals and by the verifier that opens: two spellings of
    this tuple would be two systems disagreeing about what a body was bound to, and the gate is
    where the independent recomputation lives.
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
    """`(effective_time, lsn)` - the as-of order. As-at order is LSN order and needs no key.

    A frame whose `effective_time` is null carries no truth-time of its own, and it reads as of when
    it was RECORDED. The resolution happens here, at the reader, rather than at the writer: a writer
    that stamped its own clock would put the moment of arrival inside the semantic tuple and make
    the second submission of one booking a second fact.

    Stated once and gated: two facts true at the same instant replay in the order they were
    written, so a fold is a function of the log rather than of a sort's tie-breaking.
    """
    return (frame['effective_time'] or frame['record_time'], frame['lsn'])


def take_claim(handle):
    """An exclusive, non-blocking byte-range lock on `handle`, on whichever platform this is.

    A lock the operating system releases when the process dies, rather than an `O_EXCL` sentinel
    file: a crash under a sentinel leaves a claim nobody holds and an operator clearing it by hand,
    which is a runbook step invented to work around a design decision.
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
    """`json.loads`' integer hook for canonical bytes.

    Canonical bytes are not a fixpoint of `json.loads`: a magnitude past 2**53 canonicalises to a
    plain integer literal that comes back as an `int`, which the canonicaliser then refuses.
    Anything re-parsing canonical bytes reads them through this, per canon.py's declared asymmetry.
    """
    value = int(text)
    return value if abs(value) <= 2 ** 53 else float(value)


class SpineLog:
    """One home's log, opened.

    Opening streams every segment and checks what the stored bytes say about themselves - LSN
    continuity and the prev-hash linkage - then builds the three indexes the writer needs: the head,
    the tag index idempotency is enforced on, and a byte offset per LSN. It does NOT recompute
    hashes: that is `verify_home`'s job, and keeping the two apart is what lets verification be a
    thing a replica does to a log rather than a thing a log says about itself.
    """

    def __init__(self, home):
        self._lock = None
        # The writer's own voice, off. Set only around the internal denial path, so that the one
        # type no submitter may speak is still a type the writer can, and so that the denial itself
        # is never gated - a refusal that could be refused is a regress, not a record.
        self._reserved = False
        self.home = Path(home)
        self.log = self.home / 'log'
        # log/ and blobs/, and NOT keys/. The brief's replica holds the full log and the blob store
        # and no key at all: it verifies its chain over ciphertext, and the absence of a key surfaces
        # where a key is USED (seal.py refuses there, by the file's own name) rather than as a
        # refusal to open a log it is entitled to read.
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
        # A writer that was dropped rather than closed still lets go of the home. Interpreter
        # shutdown can have taken `os` away already, and a claim released by process exit is
        # released either way, so nothing here is worth an error at the end of a run.
        try:
            self.close()
        except Exception:
            pass

    def close(self):
        """Release this home's writer claim. Idempotent, and nothing at all on a handle that only
        ever read - reading never claims, so a replica is never in anyone's way."""
        if self._lock is None:
            return
        drop_claim(self._lock)
        os.close(self._lock)
        self._lock = None

    @property
    def genesis_actor(self):
        """Who minted this home, read off LSN 1's envelope. The deployment's own seat, and what a
        checkpoint is attributed to when nobody says otherwise."""
        return self._genesis_actor

    def head(self):
        """`(lsn, event_hash)` - where the log stands. `(0, GENESIS_PREV)` when nothing is written
        yet, which is what the first event chains onto."""
        return (self._head_lsn, self._head_hash)

    def frames(self, start_lsn=1, end_lsn=None):
        """Every frame from `start_lsn` to `end_lsn`, in LSN order, read from disk.

        Read rather than remembered, and the SEGMENT LISTING is re-read too. A handle held across
        somebody else's append - reading never claims the home, so this is ordinary - would otherwise
        walk the segments that existed when it opened, and a fold over that stream is a fold over a
        log that has moved on. The only remembered thing left is the lower bound of a segment already
        seen, which cannot change once the segment holds a line.
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
        """The one frame at `lsn`, by byte offset."""
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

        The AAD is rebuilt from the frame's own envelope, so this fails on a body that moved AND on
        an envelope that moved - one refusal, `SealedBodyUnreadable`, for both, because GCM does
        not distinguish them and neither should a reader.
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
        """The event's body - the fact itself. The entitled read; `SealedBodyUnreadable` where the
        class key is gone, which is exactly the crypto-shredded state."""
        return self.open_interior(frame)['payload']

    def append(self, event_type, body, actor, book=None, effective_time=None, blob_refs=()):
        """Append one fact and answer its envelope.

        The order is the design. The writer's claim on the home, then validation, then the tag, then
        the DUPLICATE check - which decrypts the stored event and byte-compares before anything is
        written - then referential closure over every blob this event cites, and only then a byte
        reaches the platter. Every refusal therefore leaves the head exactly where it found it: a
        rejected append is not a partial one.

        `effective_time` is the caller's, or NULL. It is not defaulted to `record_time`: the
        semantic tuple contains it, so a writer-stamped default would put this call's clock inside
        the fact's own identity and make the second submission of one booking a second fact - the
        exact opposite of the law that a retry is the same fact by construction. A fact with no
        truth-time of its own carries null and reads as of when it was recorded (`as_of_key`).
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
            # A bare hash iterates as characters, and the refusal would then name a letter: say
            # what happened instead.
            raise MalformedEvent(
                'blob_refs is the string {!r}: pass the hashes as a sequence - `blob_refs=(h,)` '
                'for one - so the writer checks addresses rather than characters'.format(blob_refs))
        # What the caller declares AND what the body itself says lives in the store. The second
        # half is the one that cannot be forgotten: a `snapshot_registered` naming a blob nobody
        # fsynced is unrepresentable here whether or not the submitter remembered `blob_refs`.
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
            # The fold moves with the log rather than being re-read from it: the same step
            # `build_state` repeats, applied to the one event that just landed. Safe only because
            # this handle holds the home's claim - nothing else can have appended since.
            apply_event(self._capability, event_type, actor, body, self.store)

        envelope['event_hash'] = frame['event_hash']
        envelope['lsn'] = frame['lsn']
        envelope['coalesced'] = False
        return envelope

    def _coalesce(self, lsn, tag, canonical):
        """The duplicate-tag path: prove the stored event is the same fact, or refuse by name.

        Proof is a byte comparison against the stored event's own canonical semantic bytes,
        reconstructed by decrypting it. A duplicate that cannot be PROVEN identical - because the
        stored body will not open - is refused too: 'probably the same' is how a record acquires a
        second version of a fact.
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

        Enforcement activates BY DECLARATION for the six document verbs. Until a capabilities
        document is in the log there is no document to consult, `evaluate` is not called about them,
        and the home runs as the single-user instrument it is - which is not a compatibility shim but
        the honest reading of a record that has never been told who may do what. It is also why every
        increment-1 home still writes.

        The break-glass handle is the one exception and is gated from event one, because it is the
        one authorization that does not come FROM a document: no declaration grants it, so no
        declaration's absence can mean "anybody may", and the alternative is an unscoped actor
        appending free text into a sealed body of the record on a home nobody has governed yet.

        Once a document IS in force, a refusal is a fact. The writer appends `capability_denied`
        under its own name and then raises, so the decision is in the record where a fold, an
        auditor and a surveillance projection can all read it - rather than in a log line on a
        machine nobody replays. Two appends therefore leave this method: the denial, and nothing.

        Three things are checked and the order is the design. The reserved type first, because it
        must be refused whether or not any document exists. Then scope, because that is the question
        the brief asks. Then the DOCUMENT ITSELF, last, so that a malformed policy is met at the
        moment it is declared - by the seat that has the right to declare it, holding the file that
        needs fixing - rather than discovered at the next append by a home that can no longer write.

        The fold this consults cannot fail. A capabilities blob that has been doctored under its own
        address folds to `UNREADABLE` rather than raising, so a home in that state refuses every verb
        by name and is still rescued through the break-glass seat and the admin it recovers - the
        bricked writer being exactly what the brief's recovery grant exists to prevent.
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
        # Enforcement activates by declaration for the six document verbs, and from event one for
        # the RECOVERY handle. The handle is the exception because it is not IN a document - that is
        # the whole point of it - so "no document yet" is not a reason to let anybody at all speak as
        # the seat genesis named, and an ungated `break_glass_used` is free text into a sealed body
        # of the record under any name that asks.
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
        """The middle of a denial: what the record actually says, and what would change it.

        Three different refusals wear one sentence otherwise, and the operator reading it needs the
        most specific true thing - a home whose policy blob was doctored is not a home that forgot to
        grant somebody a scope, and neither is a stranger reaching for the break-glass handle.
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

        The capability state is one more thing opening a home derives: it is built the first time an
        append needs it and updated as later ones land, so a writer does not pay the length of the
        book to answer a question about a handful of declarations. The cache is safe because this
        handle holds the home's exclusive claim before it may append, and `_claim` re-scans - a fold
        nobody else can move underneath is a fold worth remembering.

        It cannot fail. A declared document whose blob no longer answers for it folds to `UNREADABLE`
        rather than raising, because the alternative is an exception thrown from inside the writer's
        own authorization hook - which brings down `break_glass_used` and the replacement declaration
        with it, and that is a bricked home rather than a refused append.
        """
        if self._capability is None:
            self._capability = build_state(self)
        return (self._capability['doc'], self._capability['genesis'])

    def _deny(self, subject, verb, book, attempted_type):
        """Append the refusal as a fact, under the writer's own name. Answers its envelope.

        The one path that may speak the reserved type, and the one append that is never gated. A
        second identical denial coalesces onto the first by the ordinary tag rule, because "this
        subject was refused this verb over this book for this type" is one fact however many times
        it is attempted - the tempo of the attempts is serving-layer telemetry, not the record.
        """
        self._reserved = True
        try:
            return self.append('capability_denied',
                               denial_body(subject, verb, book, attempted_type), actor=WRITER)
        finally:
            self._reserved = False

    def _claim(self):
        """Take this home's writer claim, and re-read the head under it.

        Taken at the first APPEND rather than at construction, because verifying and reading are not
        writing: a replica that locked its own log would be a replica nobody else could write to,
        and `verify_home` opens one of these every time it runs.

        The re-read is the other half and is not decoration. Between constructing this handle and
        claiming the home another writer may have appended, and an LSN assigned off a stale head is
        the one defect this package has no repair for - two lines at one position, and nothing here
        may edit or remove either.
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

        The torn tail is the one repair in the package. A final line of the LAST segment with NO
        TERMINATING NEWLINE was interrupted mid-write - the terminator is the last byte a write puts
        down, so its absence is the signature of an interrupted one - and it was never fsynced, so
        nothing ever depended on it. A line that carries its newline is a line this writer finished:
        garbage inside one is a durable line that was ALTERED, and truncating it would delete the
        evidence of the tampering (during a read-only verification, at that). So it is `ChainBroken`
        naming the position, like every other defect.
        """
        self._segments = sorted(self.log.glob(SEGMENT_GLOB))
        self._segment_first = [None] * len(self._segments)
        self._segment_last = [0] * len(self._segments)
        self._at = {}
        self._tags = {}
        # The capability fold, dropped: a reopen may have found history this handle had not seen, so
        # the cached answer is thrown away rather than carried across the scan that replaced its
        # inputs. It is rebuilt from the platter the next time an append needs it.
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
        """One frame's stored fields, checked against the position the log has reached."""
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
        # First writer wins: a repeated tag is a defect verify will name, and the coalesce must
        # point at the event that was actually first.
        self._tags.setdefault(frame['idempotency_tag'], frame['lsn'])
        if self._segment_first[index] is None:
            self._segment_first[index] = frame['lsn']
        self._head_lsn = frame['lsn']
        self._head_hash = frame['event_hash']
        if frame['lsn'] == 1:
            self._genesis_actor = frame['actor']

    def _parse(self, raw, path):
        """A line read back, or `ChainBroken`. Reading is where a doctored file is met, so it
        refuses rather than skipping."""
        frame = self._loads(raw)
        if frame is None:
            raise ChainBroken(
                '{}: a line will not parse as JSON - the segment was altered after it was opened; '
                'restore it from a verified replica'.format(path.name))
        return frame

    @staticmethod
    def _loads(chunk):
        """A frame, or None where the bytes are not one."""
        try:
            frame = json.loads(chunk.decode('utf-8'))
        except (ValueError, UnicodeDecodeError):
            return None
        return frame if isinstance(frame, dict) else None

    def _write(self, line):
        """Put one line on the platter and make it durable. Answers `(segment index, offset)`.

        `flush` then `fsync`: the event is not appended until the bytes are on the device, because
        every promise above this - durability ordering, the chain, the checkpoint - is worth what
        this call is worth.
        """
        index, path = self._segment_for(len(line))
        size = path.stat().st_size if path.exists() else 0
        payload = line
        offset = size
        if size and not self._ends_with_newline(path):
            # A durable frame that never got its terminator (a crash between the write and the
            # sync of the byte after it): start the new line rather than gluing onto that one.
            payload = b'\n' + line
            offset = size + 1
        with path.open('ab') as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        return (index, offset)

    def _segment_for(self, length):
        """The segment this line lands in, rolling a new one past 64 MiB.

        Rolling is bookkeeping: the chain runs across segments untouched, so a segment boundary is
        never a place a verifier has to know about.
        """
        if not self._segments:
            return self._roll()
        index = len(self._segments) - 1
        path = self._segments[index]
        size = path.stat().st_size if path.exists() else 0
        if size and size + length > SEGMENT_LIMIT:
            return self._roll()
        return (index, path)

    def _roll(self):
        """Start `segment-XXXXXXXX.jsonl`, numbered from one."""
        path = self.log / 'segment-{:08d}.jsonl'.format(len(self._segments) + 1)
        self._segments.append(path)
        self._segment_first.append(None)
        self._segment_last.append(self._head_lsn)
        return (len(self._segments) - 1, path)

    @staticmethod
    def _ends_with_newline(path):
        with path.open('rb') as handle:
            handle.seek(-1, os.SEEK_END)
            return handle.read(1) == b'\n'
