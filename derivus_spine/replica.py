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

"""The follower - a read-only copy of a hub's home, pulled frame by frame and verified here.

A replica is `log/` and `blobs/`, written only through `SpineLog.accept`, which takes what the hub
already chained and changes no byte of it. Nothing here submits: writes are never peer to peer, in
any phase, so there is no path from this module back to the hub.

Two seams and no third. `Hub` is the hub as this package reaches it - two reads and a stream over
`urllib.request`, because the import budget is stdlib plus `cryptography` and a follower is not
where to spend it. `follow` takes an ITERABLE of beats, so the doorbell's stream, a metronome and a
list a gate wrote are one loop over one `catch_up`: A BEAT IS A NOTIFICATION AND NEVER A DELIVERY,
read for nothing but "ask again" at the position this replica stands at, so a beat dropped is
covered by the next, one repeated is an empty pull, and one behind the head is a pull that answers
nothing. Catching up after an hour asleep is that same loop with a bigger page.

What a follower CHECKS is the range it just landed, `verify_chain`'s posture: the chain behind it
was re-derived when it landed, so re-deriving the history on every beat would be O(the record) per
event. The whole chain is checked on the first catch-up of a session and on demand.

The authenticity boundary is honest and is the page's to state: a checkpoint signs the head BEFORE
it, so a replica proves authenticity up to its last pulled checkpoint and the frames past it are
chained but unsigned.
"""
import json
import logging
import time
import urllib.parse
import urllib.request

from .errors import CollisionRefusal, CustodyRefusal, HubUnreachable, SealedBodyUnreadable
from .verify import verify_chain, verify_home
from .vocabulary import BLOB_FIELDS, cited_blobs

LOG = logging.getLogger(__name__)

#: Frames per pull. A page rather than the whole gap, so a follower an hour behind holds one page
#: in memory at a time and lands each one before it asks for the next.
PAGE = 500
#: Seconds between beats where no doorbell answers - the fallback cadence, and the first pull of a
#: follower that has just started.
INTERVAL = 2.0
#: Seconds a read of the hub waits. The doorbell's own idle stream is held open by its heartbeat,
#: so this bounds a hub that went away rather than one that has nothing to say.
TIMEOUT = 30.0

#: Where the beats come from, and the prefix a server-sent event carries its payload under.
DOORBELL = '/spine/doorbell'
DATA = b'data:'


class Hub:
    """The hub as a replica reaches it: two reads and a stream, over `urllib.request`.

    `actor` is the seat the blob read is served under, which a chain-only follower never needs -
    the chain is re-derived over ciphertext and a frames read opens no body.
    """

    def __init__(self, url, actor=None, timeout=TIMEOUT):
        self.url = url.rstrip('/')
        self.actor = actor
        self.timeout = timeout

    def __repr__(self):
        return 'Hub({!r})'.format(self.url)

    def frames(self, since, limit):
        """`{head, frames}` - the hub's frames after `since`, at most `limit` of them."""
        return json.loads(self._read('/spine/frames?since={:d}&limit={:d}'.format(since, limit)))

    def blob(self, digest):
        """The bytes the hub holds at `digest`, asked for under this follower's seat."""
        return self._read('/spine/blobs/{}{}'.format(digest, '?actor={}'.format(
            urllib.parse.quote(self.actor)) if self.actor else ''))

    def beats(self):
        """One beat per line the doorbell sends, until the stream drops: the position where a line
        carries one, and None where it is a comment.

        A COMMENT IS A BEAT TOO, and so is the one that opens the stream. What a beat carries is
        read for nothing - the pull asks from the position this replica stands at - so a heartbeat
        costs one empty pull, and the opening tells a follower it is listening before anything has
        happened to hear about. A hub serving no doorbell ends this at once, which is the fallback
        rather than a failure.
        """
        with self._open(DOORBELL) as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line[len(DATA):]) if line.startswith(DATA) else None

    def _read(self, path):
        with self._open(path) as answer:
            return answer.read()

    def _open(self, path):
        """The hub's answer at `path`, or `HubUnreachable` naming what happened to the request."""
        try:
            return urllib.request.urlopen(self.url + path, timeout=self.timeout)
        except (IOError, OSError, ValueError) as unreachable:
            raise HubUnreachable(
                '{}{} did not answer ({}): a replica pulls what the hub serves and writes nothing '
                'back, so there is nothing here to replay by hand - check the hub is up and that '
                'this seat may read it. What this home already holds verifies without it'.format(
                    self.url, path, unreachable))


def catch_up(log, hub, blobs=False, page=PAGE, bounded=False, whole=False):
    """Pull frames until this replica stands where the hub does, and answer what it took.

    ONE LOOP, whatever woke it - a beat, a metronome, or an operator running the verb once - so an
    hour asleep costs more pages and no second path. `bounded` stops at the head THE FIRST PULL
    REPORTED, which is what a one-shot sync means: a desk in session is a hub that keeps writing,
    and a loop that waits to observe an empty page against one never returns. A follower on a beat
    takes the unbounded loop, because its next bound is its next beat.

    `blobs` pulls the bytes cited by every frame THIS REPLICA HOLDS whose store lacks them, not
    merely by the frames this pull brought: a replica becomes entitled after the fact, and what it
    then owes is the whole chain's citations. The walk opens a body only for the types that cite
    bytes at all, and a store that already holds one answers without asking the hub.

    THE REPLICA VERIFIES WHAT IT LANDED, off the platter and over ciphertext it may not be able to
    read, because a copy that took bytes off a network and did not check them is a copy of nothing.
    It verifies the RANGE and not the history: what is behind was re-derived when it landed, and a
    from-genesis pass on every beat is O(the record) per EVENT, which is the cost 6a just took out
    of the reader. `whole` asks for the from-genesis pass - what a first catch-up and `DV_Spine
    verify` do.
    """
    stood, taken, ceiling = log.head()[0], 0, None
    while True:
        answer = hub.frames(log.head()[0], page)
        ceiling = answer['head'] if ceiling is None else ceiling
        if not answer['frames']:
            break
        for frame in answer['frames']:
            log.accept(frame)
        taken += len(answer['frames'])
        if bounded and log.head()[0] >= ceiling:
            break
    addresses = _cited(log, hub) if blobs else []
    head_lsn, head_hash = log.head()
    checked = None if not taken else (verify_home(log.home, entitled=False) if whole
                                      else verify_chain(log.home, stood))
    LOG.info('%s: %d frame(s) and %d blob(s) pulled from %s, standing at LSN %d (%s), %s',
             log.home, taken, len(addresses), hub, head_lsn, head_hash,
             'chain verified' if whole and checked else
             'LSN {} to {} verified'.format(stood + 1, head_lsn) if checked else 'unmoved')
    return {'frames': taken, 'blobs': addresses, 'head_lsn': head_lsn, 'head_hash': head_hash,
            'verified': checked['head_hash'] if checked else None}


def follow(log, hub, beats, blobs=False, page=PAGE, whole=False):
    """Catch up once per beat of `beats`, answering every catch-up in turn.

    THE FIRST CATCH-UP VERIFIES THE WHOLE CHAIN and every later one the range it landed: a follower
    checks what arrives once, and what it has already checked it does not check again. `whole`
    keeps the from-genesis pass on every beat, for an operator who wants it.

    A hub that stops answering is logged and beaten at again rather than raised: there is no
    failover here, so a follower's whole recovery is asking once more.
    """
    first = True
    for _ in beats:
        try:
            yield catch_up(log, hub, blobs=blobs, page=page, whole=whole or first)
            first = False
        except HubUnreachable as dropped:
            LOG.info('%s: %s', log.home, dropped)


def beating(hub, interval=INTERVAL):
    """Beats from `hub`'s doorbell where it has one, and from a metronome where it does not.

    The stream is an OPTIMISATION and never a dependency: it removes latency, never a step. A hub
    serving no doorbell, a proxy closing an idle stream and a connection that resets all fall back
    to `interval` and are asked again, and the beat that opens each round is what makes a follower
    that has just started catch up before it waits for anything.
    """
    while True:
        yield time.monotonic()
        try:
            for beat in hub.beats():
                yield beat
        except (HubUnreachable, IOError, OSError) as dropped:
            LOG.info('%s: the doorbell dropped (%s) - beating every %.1fs until it answers again',
                     hub, dropped, interval)
        time.sleep(interval)


def _cited(log, hub):
    """Pull every blob this replica's chain cites that its store lacks, and answer what was taken.

    A citation lives in the SEALED BODY, so pulling blobs is the entitled posture by construction:
    a replica holding no class key cannot know what to ask for and is told so rather than quietly
    pulling nothing. Which frames to open is an ENVELOPE question - only some types cite bytes at
    all - so the walk costs a body per citing frame and a parse per other. What arrives is checked
    against the address it was asked for, a blob being self-verifying by hash however it travelled.
    """
    taken = []
    for frame in log.frames():
        if frame['event_type'] not in BLOB_FIELDS:
            continue
        try:
            body = log.open_body(frame)
        except SealedBodyUnreadable as sealed:
            raise CustodyRefusal(
                'this replica cannot open LSN {} ({}), so it cannot know which blobs that frame '
                'cites: a citation lives in the sealed body. Materialize the class key here '
                '(`custody.materialize`) and follow again, or follow chain-only - the chain '
                'verifies over ciphertext and wants no blob at all'.format(frame['lsn'], sealed))
        for _, digest in cited_blobs(frame['event_type'], body):
            if log.store.has(digest):
                continue
            landed = log.store.put(hub.blob(digest))
            if landed != digest:
                raise CollisionRefusal(
                    '{} served bytes for the blob {} that hash to {}, so they are filed under the '
                    'address they answer for and the citation is still unresolved: a blob is '
                    'self-verifying by hash, so pull it from any replica that really holds '
                    'it'.format(hub, digest, landed))
            taken.append(digest)
    return taken
