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

"""The replica: what a follower may write, what a stream may do to it, and what it still proves.

Real homes in temp directories, every frame through the ordinary writer, and every fault injected
as DATA. The doorbell's stream is a LIST this file builds and mutilates before handing it to the
follower, which is the only honest way to drop, duplicate and reorder a notification: what is under
test is that none of it matters, because a beat carries a position and the pull asks from the
position the replica stands at.

What the gates hold, in order:

  * THREE REPLICAS OFF ONE HUB CONVERGE. One follower hears every beat, one loses every third and
    the whole of the last round - so it really is behind before the next beat rings - and one hears
    them twice and out of order. After the beat that covers them, the three chain-only
    verifications carry ONE head hash and all NINE projectors fold to byte-equal rows.
  * THE PULL LOOP IS ONE FUNCTION, read off the module's own source, which is what makes the
    doorbell an optimisation: a stream that drops changes when a replica pulls and never how.
  * A BEAT VERIFIES THE RANGE IT LANDED, off the platter and never the history: a record time
    doctored inside the range leaves the scan's linkage intact and the event hash refusing.
  * AN HOUR ASLEEP IS THE SAME PATH. A hundred events land with the follower stopped, one
    resumption reaches the head, and the replica's segment bytes are the hub's byte for byte -
    which is the only place `accept` writing VERBATIM can be observed, a re-serialised frame being
    a different line under the same fields.
  * A REPLICA WRITES THE HUB'S NEXT LINE AND NOTHING ELSE: a thirteenth field, a position that is
    not next, a link to another history and an event hash that does not recompute, each refused by
    name with the head where it was found, and a second follower on one home refused the claim.
  * A TYPE THIS BUILD HAS NEVER HEARD OF STILL CHAINS - the version-tolerance case, built by hand
    against the hub's own key because no writer here will author one, and `accept`ed on the same
    handle whose `append` refuses that very type by name.
  * A CHAIN-ONLY REPLICA HOLDS NO BLOB and verifies anyway; the entitled reading refuses naming the
    blob it lacks, the seat's own wrap makes the directory entitled, pulling blobs fills what the
    chain cites, and only then is the entitled verification the hub's own report.
  * FIFTY OPEN STREAMS COST NO THREAD AND LEAVE NO LISTENER. The doorbell awaits rather than
    blocking, so an idle stream parks none of the worker tokens every other verb on this service
    shares, and a read behind fifty of them answers at once; a client that goes away is unregistered
    by the generator's own close rather than by a collection.
  * A ONE-SHOT SYNC RETURNS AGAINST A HUB THAT NEVER STOPS WRITING - bounded by the head the first
    pull reported, since a desk in session never shows an empty page.
  * AND ALL OF IT OVER A SOCKET, once: a service on an EPHEMERAL PORT, the CLI verb catching up
    against it, and the doorbell's beat timed from the append that rang it. A `TestClient` collects
    a response before it answers and a doorbell never completes, so this is where the stream is
    driven for real.
"""
import base64
import hashlib
import json
import os
import shutil
import socket
import sys
import threading
import time
from itertools import islice

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from derivus_spine import (
    ChainBroken, CapabilityDenied, SpineLog, SpineRefusal, UnknownEventType, WriterBusy,
    canonical_bytes, cli, init_home, replica, verify_home)
from derivus_spine.capability import CAPABILITIES_POLICY, canonical_document
from derivus_spine.custody import CLASS_KEY_FILE, enroll, materialize, rewrap, seat_key_path
from derivus_spine.log import EVENT_VERSION, aad_bytes, event_hash, now_stamp
from derivus_spine.projections import PROJECTORS, fold
from derivus_spine.verbs import file_quote
from derivus_spine.verify import verify_chain
from derivus_spine.vocabulary import FIRM_CLASS, VERBS

from test_spine import ACTOR, BOOK, doctor, fill, synthetic_book
from test_spine_custody import ANY, document

STRANGER = 'subject-nobody'
#: Where the design's synthetic book ends - every later head in this file counts from there.
BOOK_HEAD = 21
#: How long a gate waits on a socket or a beat before it says so instead of hanging the suite.
WIRE_SECONDS = 20.0
#: Open streams the load gate holds, and what a read behind them may cost. The failure this bounds
#: is a fifteen-second one - a reader waiting for somebody else's heartbeat to hand a thread back -
#: so the bound is loose enough never to flake and tight enough to leave that no room at all.
STREAMS = 50
BUSY_MS = 250.0


def _timed(call):
    """`call()`'s wall time in milliseconds."""
    began = time.monotonic()
    call()
    return (time.monotonic() - began) * 1000.0


def served(home):
    """The hub as a follower reaches it, IN PROCESS: `Hub`'s two reads answered off the hub's own
    log rather than over a socket, so what a gate mutilates is the stream and never the transport.
    """
    class Reader:
        def frames(self, since, limit):
            log = SpineLog(home)
            try:
                return {'head': log.head()[0],
                        'frames': list(islice(log.frames(start_lsn=since + 1), limit))}
            finally:
                log.close()

        def blob(self, digest):
            log = SpineLog(home)
            try:
                return log.store.get(digest)
            finally:
                log.close()

    return Reader()


def granted(log, actor, verbs=VERBS, read=(ACTOR,)):
    """Declare a capabilities document giving `actor` every verb over every book, and the named
    seats the firm class to read."""
    blob = log.store.put(canonical_document(document(
        grants=tuple((actor, verb, ANY) for verb in verbs),
        read=tuple((subject, FIRM_CLASS) for subject in read))))
    return log.append('policy_declared', {'policy': CAPABILITIES_POLICY, 'blob': blob},
                      actor=actor, blob_refs=(blob,))


def hub_of(tmp_path):
    """The design's synthetic book, then 5b's quote flow, then a document in force and a stranger
    refused under it - so every one of the nine projectors has rows to disagree about.

    Answers `(home, log)` with the writer still open.
    """
    home, log = synthetic_book(tmp_path)[:2]
    values = b'{"USDZAR":18.5}'
    file_quote(log, ACTOR, 'Q-1', 'ZeroCostCollar', 'a' * 64, values, {'floor': 17.1}, 4100.0,
               ticket='b' * 64, book=BOOK)
    file_quote(log, ACTOR, 'Q-2', 'Accumulator', 'a' * 64, values, {'strike': 16.5}, 900.0,
               book=BOOK)
    granted(log, ACTOR)
    with pytest.raises(CapabilityDenied):
        log.append('fill', fill('EXEC-STRANGER'), actor=STRANGER, book=BOOK)
    return home, log


def replica_home(tmp_path, name):
    """An empty replica: `log/` and `blobs/` and no key of any kind."""
    home = tmp_path / name
    for part in ('log', 'blobs'):
        (home / part).mkdir(parents=True)
    return home


def entitled_by_copy(hub, home):
    """The class key put into `home` as a file copy - what a materialized replica looks like on
    disk, used where a gate wants to FOLD rather than to gate custody, which the chain-only gate
    below does off a real wrap."""
    (home / 'keys').mkdir(exist_ok=True)
    shutil.copy(str(hub / 'keys' / CLASS_KEY_FILE.format(FIRM_CLASS)),
                str(home / 'keys' / CLASS_KEY_FILE.format(FIRM_CLASS)))
    return home


def rung(log, since):
    """One beat per frame the hub landed after `since` - the stream a doorbell would have sent."""
    return [{'lsn': frame['lsn'], 'head': frame['event_hash']}
            for frame in log.frames(start_lsn=since + 1)]


def forged(keys, onto, event_type, body, tag=None):
    """A frame sealed under the hub's own key onto `onto`'s head - what a hub a build ahead of this
    one would have written, and the only way to hand a replica a line no writer here will author.

    `tag` takes an idempotency tag already on that platter, which is the only way to ask whether a
    replica coalesces: the hub's own writer folds a repeat onto the LSN it already holds, and a
    replica must take the order it was given instead.
    """
    envelope = {'actor': ACTOR, 'book': BOOK, 'effective_time': None,
                'entitlement_class': FIRM_CLASS, 'event_type': event_type,
                'event_version': EVENT_VERSION,
                'idempotency_tag': tag or hashlib.sha256(canonical_bytes(body)).hexdigest(),
                'prev_hash': onto.head()[1], 'record_time': now_stamp()}
    sealed = keys.seal(canonical_bytes({'content_hash': 'c' * 64, 'payload': body}),
                       aad_bytes(envelope))
    return dict(envelope, body=base64.b64encode(sealed).decode('ascii'),
                lsn=onto.head()[0] + 1,
                event_hash=event_hash(hashlib.sha256(sealed).hexdigest(),
                                      envelope['idempotency_tag'], envelope['prev_hash'],
                                      envelope['record_time']))


def folded(home):
    """Every projector's rows on `home`, in canonical bytes - what two replicas must agree on."""
    log = SpineLog(home)
    try:
        return dict((name, canonical_bytes(PROJECTORS[name].rows(fold(log, PROJECTORS[name]))))
                    for name in sorted(PROJECTORS))
    finally:
        log.close()


# --------------------------------------------------------------------------------------------
# The stream, mutilated.

def test_a_dropped_duplicated_or_reordered_doorbell_converges_every_replica(tmp_path):
    """THE CLAIM THE DOORBELL RESTS ON. A beat carries a position and the pull asks from the
    position the replica stands at, so what a stream does to a beat cannot change where a replica
    ends up - only how long it takes to get there.

    Three followers off one hub, each handed its own view of one stream as a list. The lossy one is
    behind when the rounds end, which is what makes the mutilation real rather than decorative; one
    more beat - the next head move, or the metronome the stream falls back to - covers it, and then
    the three chain-only verifications carry one head hash and all nine projectors fold to
    byte-equal rows.
    """
    hub, writer = hub_of(tmp_path)
    reader = served(hub)
    streams = {'every': lambda beats: list(beats),
               'dropped': lambda beats: [beat for at, beat in enumerate(beats) if at % 3 != 2],
               'shuffled': lambda beats: beats[::-1] + beats + beats[::2]}
    mirrors = dict((name, SpineLog(replica_home(tmp_path, name))) for name in streams)

    for round_ in range(4):
        since = writer.head()[0]
        for clip in range(round_ + 1):
            writer.append('fill', fill('EXEC-{}-{}'.format(round_, clip)), actor=ACTOR, book=BOOK)
        beats = rung(writer, since)
        for name, mutilate in streams.items():
            # the LAST round is lost whole for the lossy stream, so it ends behind on purpose
            list(replica.follow(mirrors[name], reader,
                                [] if round_ == 3 and name == 'dropped' else mutilate(beats)))

    assert mirrors['dropped'].head()[0] < writer.head()[0], 'the lost beats cost it nothing at all'
    assert mirrors['every'].head() == mirrors['shuffled'].head() == writer.head()

    covering = rung(writer, writer.head()[0] - 1)
    for name in sorted(streams):
        list(replica.follow(mirrors[name], reader, covering))
        mirrors[name].close()

    reports = [verify_home(tmp_path / name, entitled=False) for name in sorted(streams)]
    assert set(report['head_hash'] for report in reports) == {writer.head()[1]}, reports
    assert set(report['events'] for report in reports) == {writer.head()[0]}

    rows = [folded(entitled_by_copy(hub, tmp_path / name)) for name in sorted(streams)]
    assert len(rows[0]) == 9, sorted(rows[0])
    for name in sorted(rows[0]):
        assert rows[0][name] == rows[1][name] == rows[2][name], name
    assert json.loads(rows[0]['denials'])[0]['subject'] == STRANGER
    assert [row['quote_id'] for row in json.loads(rows[0]['quotes'])] == ['Q-1', 'Q-2']
    writer.close()


def test_the_pull_loop_is_one_function():
    """The doorbell removes LATENCY and never a step, which is only true while the beat-driven
    follower and the one-shot verb run the same code. Read off the module's own source: exactly one
    function in it asks a hub for frames, and it is the one `follow` calls, so a stream that drops
    changes when a replica pulls and never how."""
    import ast

    def asks_the_hub(node):
        return any(isinstance(call.func, ast.Attribute) and call.func.attr == 'frames'
                   and getattr(call.func.value, 'id', None) == 'hub'
                   for call in ast.walk(node) if isinstance(call, ast.Call))

    module = ast.parse(open(replica.__file__, encoding='utf-8').read())
    pulling = set(node.name for node in ast.walk(module)
                  if isinstance(node, ast.FunctionDef) and asks_the_hub(node))
    assert pulling == {'catch_up'}, pulling
    assert 'catch_up' in replica.follow.__code__.co_names


def test_a_follower_asleep_for_an_hour_takes_the_same_path_as_a_live_one(tmp_path):
    """Catching up is the same loop with more pages in it, and what it writes is what the hub
    wrote. A hundred events land while the follower is stopped; one resumption reaches the head,
    the follower's check over THE RANGE IT LANDED agrees with the hub's head hash, and the
    replica's segment bytes are the hub's byte for byte.

    A hundred rather than a thousand: the assertion is that one resumption reaches the head and
    writes the hub's own bytes, and an event costs an fsync at both ends, so a bigger number buys
    minutes and no sentence.
    """
    hub, writer = synthetic_book(tmp_path)[:2]
    mirror = SpineLog(replica_home(tmp_path, 'asleep'))
    assert replica.catch_up(mirror, served(hub), whole=True)['frames'] == BOOK_HEAD

    for clip in range(100):
        writer.append('fill', fill('EXEC-SLEPT-{}'.format(clip)), actor=ACTOR, book=BOOK)
    assert writer.head()[0] == BOOK_HEAD + 100

    caught = replica.catch_up(mirror, served(hub))
    assert caught['frames'] == 100 and caught['head_lsn'] == writer.head()[0]
    assert caught['verified'] == caught['head_hash'] == writer.head()[1], 'the follower checked'
    assert (tmp_path / 'asleep' / 'log' / 'segment-00000001.jsonl').read_bytes() \
        == (hub / 'log' / 'segment-00000001.jsonl').read_bytes()
    mirror.close()
    writer.close()


def test_a_beat_verifies_the_range_it_landed_off_the_platter(tmp_path):
    """WHAT A FOLLOWER CHECKS ON A BEAT, and that it is a check. Re-deriving the history on every
    beat is O(the record) per event - the cost 6a took out of the reader, put back on the follower -
    and what is behind the range was re-derived when it landed. So a beat re-reads its own frames
    and the link they hang from, OFF THE PLATTER, which is where a write that did not land is met
    rather than believed.

    A record time doctored inside the range is what says this is more than trusting `accept`'s own
    arithmetic: the linkage the scan checks is intact and the event hash refuses.
    """
    hub, writer = synthetic_book(tmp_path)[:2]
    mirror = SpineLog(replica_home(tmp_path, 'ranged'))
    replica.catch_up(mirror, served(hub), whole=True)
    stood = mirror.head()[0]
    for clip in range(3):
        writer.append('fill', fill('EXEC-RANGE-{}'.format(clip)), actor=ACTOR, book=BOOK)

    ranged = tmp_path / 'ranged'
    caught = replica.catch_up(mirror, served(hub))
    assert caught['frames'] == 3
    assert caught['verified'] == caught['head_hash'] == writer.head()[1], 'the beat checked'
    assert verify_chain(ranged, stood)['head_lsn'] == writer.head()[0]
    with pytest.raises(ChainBroken) as absent:
        verify_chain(ranged, writer.head()[0] + 5)
    assert 'nothing for the frames after it to link to' in str(absent.value)

    # a record time doctored inside the range leaves the SCAN'S linkage intact, so what refuses is
    # the recomputation and nothing else would have
    doctor(ranged, stood + 2, lambda frame: frame.__setitem__(
        'record_time', '2020-01-01T00:00:00.000000Z'))
    with pytest.raises(ChainBroken) as altered:
        verify_chain(ranged, stood)
    assert 'LSN {}'.format(stood + 2) in str(altered.value) and 'recomputes to' in str(altered.value)

    # AND THE BEAT IS WHAT RUNS IT: the link the next range hangs from, doctored under an open
    # handle, is met by the catch-up that lands the next frame rather than by whoever opens the
    # home next
    doctor(ranged, mirror.head()[0], lambda frame: frame.__setitem__('event_hash', '0' * 64))
    writer.append('fill', fill('EXEC-RANGE-3'), actor=ACTOR, book=BOOK)
    with pytest.raises(ChainBroken):
        replica.catch_up(mirror, served(hub))
    mirror.close()
    writer.close()


def test_a_replica_writes_the_frame_the_hub_wrote_and_nothing_else(tmp_path):
    """The four refusals, each by name and each leaving the head where it found it, and the claim
    that makes two followers on one home impossible rather than merely unwise.

    A replica exercises no judgment: it does not ask what a frame MEANS, only whether it is the
    next line of this chain and whether its own bytes say so.
    """
    hub, writer = synthetic_book(tmp_path)[:2]
    mirror = SpineLog(replica_home(tmp_path, 'refusing'))
    frames = list(SpineLog(hub).frames())
    for frame in frames[:5]:
        mirror.accept(frame)
    stood = mirror.head()

    def refused(frame):
        with pytest.raises(ChainBroken) as refusal:
            mirror.accept(frame)
        assert mirror.head() == stood, 'the head moved under a refusal'
        return str(refusal.value)

    surplus = refused(dict(frames[5], note='an unauthenticated field'))
    assert 'note' in surplus and 'twelve fields' in surplus
    assert 'NEXT line' in refused(frames[6]) and 'LSN 5' in refused(frames[4])
    assert 'another history' in refused(dict(frames[5], prev_hash='f' * 64))
    assert 'recomputes to' in refused(dict(frames[5], event_hash='0' * 64))
    assert 'is not base64' in refused(dict(frames[5], body='not base64 at all!'))

    assert mirror.accept(frames[5])['lsn'] == 6
    with pytest.raises(WriterBusy) as busy:
        SpineLog(tmp_path / 'refusing').accept(frames[6])
    assert '.writer.lock' in str(busy.value), 'two followers on one replica both write LSN n+1'

    # and NO COALESCING: a tag already on this platter is written again rather than folded onto the
    # LSN that holds it, because a replica writes the order the hub gave it and nothing else
    twin = forged(writer.keys, mirror, 'rehash_declared', {'algorithm': 'sha256'},
                  tag=frames[5]['idempotency_tag'])
    assert mirror.accept(twin)['lsn'] == 7 and mirror.head()[0] == 7
    assert [frame['idempotency_tag'] for frame in SpineLog(tmp_path / 'refusing').frames()].count(
        frames[5]['idempotency_tag']) == 2, 'the replica folded a repeat the hub had ordered twice'
    mirror.close()
    writer.close()


def test_a_replica_chains_an_event_type_it_has_never_heard_of(tmp_path):
    """VERSION TOLERANCE, which is why `accept` does not consult the vocabulary: a replica of a hub
    running a newer one must still chain, or the first type a hub learns strands every copy of the
    record.

    The frame is built by hand against the hub's own key, because no writer in this build will
    author one - and that is itself the assertion, since `append` of the same type refuses by name
    on the very handle that has just `accept`ed it. The strip renders the type's own name rather
    than dropping the LSN out of the sequence.
    """
    hub, writer = synthetic_book(tmp_path)[:2]
    mirror = SpineLog(entitled_by_copy(hub, replica_home(tmp_path, 'newer')))
    for frame in SpineLog(hub).frames():
        mirror.accept(frame)

    body = {'what': 'a fact this build has never heard of'}
    assert mirror.accept(forged(writer.keys, mirror, 'future_fact', body))['lsn'] == BOOK_HEAD + 1
    with pytest.raises(UnknownEventType) as refusal:
        mirror.append('future_fact', body, actor=ACTOR, book=BOOK)
    assert 'the vocabulary is closed' in str(refusal.value), \
        'the writer refuses the very type the replica just chained, so accept never asked it'

    assert verify_home(tmp_path / 'newer', entitled=False)['head_lsn'] == BOOK_HEAD + 1
    strip = PROJECTORS['activity'].rows(fold(mirror, PROJECTORS['activity']))
    assert [row['lsn'] for row in strip] == list(range(1, BOOK_HEAD + 2))
    assert strip[-1]['summary'] == 'future_fact', 'a type with no sentence renders its own name'
    mirror.close()
    writer.close()


def test_a_chain_only_replica_holds_no_blob_and_still_verifies(tmp_path):
    """THE SPLIT THAT MAKES A REPLICA CHEAP. The chain is taken over the ciphertext, so a follower
    pulling frames alone re-derives every link holding no key and no blob; the ENTITLED reading is
    what wants referential closure, and it refuses naming what is missing rather than passing
    quietly.

    Pulling blobs is the entitled posture by construction - a citation lives in the sealed body -
    so the order is the deployment's own, and the FIRST wrap is the one blob that cannot be
    discovered: which blob makes this seat entitled is a question only an entitled reader can ask
    of the chain, so it arrives BY ADDRESS, which is what the blob read is addressed for. Frames,
    that one blob, the class key out of it, the bytes the chain cites, and only then does this
    directory answer the hub's own entitled report.
    """
    hub = tmp_path / 'hub'
    init_home(hub, ACTOR)
    writer = SpineLog(hub)
    enroll(writer, ACTOR, actor=ACTOR)
    granted(writer, ACTOR)
    rewrap(writer, actor=ACTOR)
    values = writer.store.put(b'{"USDZAR":18.5}')
    writer.append('market_declared', {'name': 'official', 'values_hash': values}, actor=ACTOR)
    private = seat_key_path(hub, ACTOR).read_bytes()
    wrapped = [writer.open_body(frame)['wrap'] for frame in writer.frames()
               if frame['event_type'] == 'key_wrapped']
    writer.close()

    mirror = SpineLog(replica_home(tmp_path, 'chain-only'))
    reader = served(hub)
    assert replica.catch_up(mirror, reader)['frames'] == SpineLog(hub).head()[0]
    assert not list((tmp_path / 'chain-only' / 'blobs').glob('*'))
    assert verify_home(tmp_path / 'chain-only', entitled=False)['head_hash'] \
        == verify_home(hub, entitled=False)['head_hash']

    with pytest.raises(SpineRefusal) as unopened:
        replica.catch_up(mirror, reader, blobs=True)
    assert 'cannot open LSN' in str(unopened.value) and 'chain-only' in str(unopened.value)
    mirror.store.put(reader.blob(wrapped[0]))
    mirror.close()

    materialize(tmp_path / 'chain-only', ACTOR, private)
    with pytest.raises(SpineRefusal) as refusal:
        verify_home(tmp_path / 'chain-only')
    assert 'is not in the store' in str(refusal.value), refusal.value

    mirror = SpineLog(tmp_path / 'chain-only')
    caught = replica.catch_up(mirror, reader, blobs=True)
    assert caught['frames'] == 0, 'the frames were already here; only the blobs were owed'
    assert values in caught['blobs'] and len(caught['blobs']) > 1
    assert replica.catch_up(mirror, reader, blobs=True)['blobs'] == [], 'a blob is pulled once'

    assert verify_home(tmp_path / 'chain-only') == verify_home(hub)
    assert mirror.open_body(mirror.frame_at(mirror.head()[0]))['values_hash'] == values
    mirror.close()


# --------------------------------------------------------------------------------------------
# And once, over a socket.

def serving(app):
    """A real service on an EPHEMERAL PORT, started and stopped by this gate. Never a fixed port:
    a suite that bound one would fight whatever the box is already running."""
    import uvicorn

    held = socket.socket()
    held.bind(('127.0.0.1', 0))
    server = uvicorn.Server(uvicorn.Config(app, log_level='error'))
    running = threading.Thread(target=lambda: server.run([held]), daemon=True)
    running.start()
    deadline = time.monotonic() + WIRE_SECONDS
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)
    assert server.started, 'the gate\'s own service did not come up'
    return 'http://127.0.0.1:{}'.format(held.getsockname()[1]), server, running, held


def test_fifty_open_doorbells_park_no_worker_thread_and_leave_no_listener(tmp_path, monkeypatch):
    """THE CEILING A DESK WOULD HAVE MET. A sync generator handed to `StreamingResponse` is driven
    through the thread pool, so every idle stream parks one of the forty tokens EVERY sync verb on
    this service shares: past forty tabs a booking, a price or a replica's pull waits for somebody
    else's heartbeat. Awaiting costs a task and no thread, and this is where that is a measurement
    rather than a claim.

    And a stream that goes away takes its listener with it. The registration is dropped by the
    generator's own `finally`, on the close starlette performs when a client disconnects, rather
    than whenever a cyclic collection happens to run - counted here through the writer's own
    registry, which is the list a doorbell registers on.
    """
    from derivus import service
    from derivus_spine.log import WATCHERS

    hub, writer = synthetic_book(tmp_path)[:2]
    monkeypatch.setenv('DV_SPINE_HOME', str(hub))
    url, server, running, held = serving(service.app)
    streams = []
    try:
        threads = threading.active_count()
        for _ in range(STREAMS):
            beats = replica.Hub(url).beats()
            assert next(beats) is None, 'the opening comment is what registers the listener'
            streams.append(beats)
        assert len(WATCHERS) == STREAMS, len(WATCHERS)
        assert threading.active_count() - threads <= 2, \
            '{} open streams took {} threads with them'.format(
                STREAMS, threading.active_count() - threads)

        reader = replica.Hub(url)
        reader.frames(0, 5)
        taken = min(_timed(lambda: reader.frames(0, 5)) for _ in range(3))
        assert taken < BUSY_MS, \
            'a read behind {} open streams took {:.0f} ms'.format(STREAMS, taken)

        for beats in streams:
            beats.close()
        streams = []
        deadline = time.monotonic() + WIRE_SECONDS
        while WATCHERS and time.monotonic() < deadline:
            time.sleep(0.05)
        assert WATCHERS == [], '{} listeners outlived the clients that opened them'.format(
            len(WATCHERS))
    finally:
        for beats in streams:
            beats.close()
        server.should_exit = True
        running.join(timeout=WIRE_SECONDS)
        held.close()
        writer.close()


def test_a_one_shot_sync_returns_against_a_hub_that_never_stops_writing(tmp_path):
    """`--once` IS THE CRON'S VERB and a desk in session is a hub that keeps writing, so a loop
    that waits to OBSERVE an empty page has no reason to end against one. The bound is the head the
    FIRST pull reported, which is what that field is on the wire for: a one-shot sync is a copy of
    the record as it stood when the copy began.
    """
    hub, writer = synthetic_book(tmp_path)[:2]
    mirror = SpineLog(replica_home(tmp_path, 'one-shot'))
    stop, writing, answer = threading.Event(), threading.Event(), {}

    def keep_writing():
        clip = 0
        while not stop.is_set():
            writer.append('fill', fill('EXEC-BUSY-{}'.format(clip)), actor=ACTOR, book=BOOK)
            writing.set()
            clip += 1

    # the one-shot runs on a thread of its own, so a loop that would never return is a RED gate
    # rather than a suite that hangs
    hand = threading.Thread(target=keep_writing, daemon=True)
    once = threading.Thread(
        target=lambda: answer.update(
            replica.catch_up(mirror, served(hub), bounded=True, whole=True)), daemon=True)
    hand.start()
    try:
        assert writing.wait(WIRE_SECONDS), 'the gate\'s own writer never started'
        once.start()
        once.join(timeout=WIRE_SECONDS)
    finally:
        stop.set()
        hand.join(timeout=WIRE_SECONDS)

    assert not once.is_alive(), \
        'the one-shot was still pulling after {:.0f} s against a hub that keeps writing'.format(
            WIRE_SECONDS)
    caught = answer
    assert caught['frames'] >= BOOK_HEAD and caught['verified'] == caught['head_hash']
    assert writer.head()[0] > caught['head_lsn'], \
        'the hub was not still writing, so this gate asked nothing'
    assert verify_home(tmp_path / 'one-shot', entitled=False)['head_lsn'] == caught['head_lsn']
    mirror.close()
    writer.close()


def test_the_doorbell_rings_over_the_wire_and_the_follow_verb_catches_up(tmp_path, monkeypatch):
    """The whole of 6 over a socket, once: the two reads, the CLI verb and the stream.

    A `TestClient` collects a response before it answers and a doorbell never completes, so the
    stream is driven against a real service here or nowhere. The beat is timed from the append that
    rang it, which is the number the page quotes; what the beat CARRIES is a position and nothing
    else, and the follower reads it for nothing but "ask again".
    """
    from derivus import service

    hub, writer = synthetic_book(tmp_path)[:2]
    monkeypatch.setenv('DV_SPINE_HOME', str(hub))
    url, server, running, held = serving(service.app)
    try:
        mirror = replica_home(tmp_path, 'over-the-wire')
        assert cli.main(['follow', url, '--home', str(mirror), '--once']) == 0
        assert SpineLog(mirror).head() == writer.head()

        beats = replica.Hub(url).beats()
        assert next(beats) is None, 'the stream opens with a comment, which is a beat like any'
        struck = time.monotonic()
        landed = writer.append('fill', fill('EXEC-RUNG'), actor=ACTOR, book=BOOK)
        beat = next(beats)
        rang = (time.monotonic() - struck) * 1000.0
        assert beat == {'head': landed['event_hash'], 'lsn': landed['lsn']}, beat
        assert rang < WIRE_SECONDS * 1000.0, 'the beat took {:.0f} ms'.format(rang)

        caught = replica.catch_up(SpineLog(mirror), replica.Hub(url))
        assert caught['frames'] == 1 and caught['head_hash'] == writer.head()[1]
    finally:
        server.should_exit = True
        running.join(timeout=WIRE_SECONDS)
        held.close()
        writer.close()
