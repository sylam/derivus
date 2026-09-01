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

"""The acts a desk performs on the record - booking, amending, lifecycle, marks, quotes and runs.

The logic lives here and the engine holds thin delegators, so no module under `derivus/` learns
about users, workflow or storage. Everything below takes plain data and injected callables - bytes,
strings, numbers, a function - and holds no idea that an engine exists; the engine side reaches this
module lazily and refuses by name when no home is configured.

That is why `pin_result` takes an executor rather than importing one: re-executing a claim would
make the truth layer depend on the thing it records. The caller hands in a real function, and the
record checks what came back rather than trusting who ran it - including the engine version, which
the executor reports and this module compares against the claim.

No verb here carries an authorization check: the writer evaluates capability and lands the refusal
as `capability_denied`, and a second check would be a second place to get it wrong. The writer's
flow is likewise untouched - blind tag, duplicate-tag coalescing on identical canonical bytes, blob
fsynced before the event citing it, one exclusive claim on the home.

Three verbs restate a refusal the writer would give anyway, each naming the verb: `apply_lifecycle`
refuses anything but an election, an observation or a determination (the writer would accept a
`fill` submitted through that arm), `complete_run` refuses a lane that mints nothing, and
`declare_market` refuses a name that is not a name.

Declared limitation: the desk's `book.json` is still the edge's own file, written by the edge after
the event lands. Its rehoming as an LSN-pinned projection is not built here, so until then the log
is what is true and the file is an interim stand-in.
"""
import json

from .errors import MalformedEvent, ReplayRefused, UnknownEventType
from .policy import compare, tolerances_in_force
from .vocabulary import is_hash, is_integer, is_number, is_text

#: The three attestation lanes, which are the three answers to "will this output be cited by a
#: fact". `telemetry` is a repaint, superseded before anything could cite it; `curiosity` is a
#: what-if nobody will cite either; `standing` is a run a fact is about to name.
TELEMETRY = 'telemetry'
CURIOSITY = 'curiosity'
STANDING = 'standing'
LANES = (TELEMETRY, CURIOSITY, STANDING)
#: The lanes that mint an event. A tuple rather than an equality test, so admitting a second lane
#: moves nothing else.
MINTING = (STANDING,)

#: What `apply_lifecycle` may file: the holder's act, the world's observation, and a ruling a
#: contract vests in an agent. A knock, an expiry or an accrual is a consequence of terms plus one
#: of these, so it is a projection rather than a fact.
LIFECYCLE_TYPES = ('election', 'fixing_observed', 'determination')

#: The four coordinates a reported number replays from, named here so the claim a verb is handed
#: and the body it writes cannot part company.
REPLAY_FIELDS = ('plan_hash', 'values_hash', 'engine_version', 'seed')


def check_lane(lane):
    """`lane` if it is one of `LANES`, otherwise `MalformedEvent` naming them."""
    if lane not in LANES:
        raise MalformedEvent(
            'lane {!r} is not one of {} - a run is recorded IFF its output will be cited by a '
            'fact, so a lane is that decision written down: {} for a reading that is superseded '
            'before anything could cite it, {} for a what-if, {} for a run a fact is about to name'
            .format(lane, ', '.join(LANES), TELEMETRY, CURIOSITY, STANDING))
    return lane


def mints(lane):
    """Whether a run in `lane` appends anything at all. Telemetry and curiosity mint nothing, since
    an event about a number no fact will cite is a row every later fold must read."""
    return check_lane(lane) in MINTING


def book(log, actor, instrument, quantity, counterparty, netting_set, execution_reference,
         book=None, effective_time=None):
    """Book a fill, returning the envelope plus the instrument's address.

    `instrument` is the canonical JSON of the deal's terms as bytes - the caller canonicalises,
    since the spelling of an instrument is the engine's own - and its hash is the instrument id, so
    booking the same strike twice files two events against one store row.

    `execution_reference` is required and has no default. It is what makes a retry the same fact by
    construction, coalescing onto the LSN it already has, and two legitimately identical clips two
    facts. `quantity` is a signed quantity, never a position: position is a fold. The instrument
    blob is fsynced before the event citing it appends.
    """
    address = _blob(log, instrument, 'the canonical instrument')
    if not is_number(quantity):
        raise MalformedEvent(
            'book: quantity is {!r} - a fill carries a SIGNED quantity of the instrument, as a '
            'finite number; position is a fold over these and is never written'.format(quantity))
    body = {'instrument': address, 'quantity': quantity,
            'counterparty': _name(counterparty, 'counterparty', 'book'),
            'netting_set': _name(netting_set, 'netting_set', 'book'),
            'execution_reference': _name(execution_reference, 'execution_reference', 'book')}
    envelope = log.append('fill', body, actor=actor, book=book, effective_time=effective_time,
                          blob_refs=(address,))
    return dict(envelope, instrument=address)


def amend(log, actor, instrument, amended_to, book=None, effective_time=None):
    """Amend a booked deal: a new instrument hash linked to the old one. Returns the envelope plus
    both addresses.

    Economics are never edited, so this is a second row saying these terms became those. Both
    instruments are registered because both are cited; the old one dedups to the address it already
    has, so a deal booked before this home existed still closes referentially. Terms that
    canonicalise to the same hash raise `MalformedEvent`.
    """
    was = _blob(log, instrument, 'the canonical instrument as it was')
    now = _blob(log, amended_to, 'the canonical instrument as amended')
    if was == now:
        raise MalformedEvent(
            'amend: the amended terms canonicalise to the same instrument {} - an amendment is a '
            'NEW instrument hash linked to the old one, so terms that did not move are not an '
            'amendment; file the operational fact as a status_transition instead'.format(was))
    envelope = log.append('amendment', {'instrument': was, 'amended_to': now},
                          actor=actor, book=book, effective_time=effective_time,
                          blob_refs=(was, now))
    return dict(envelope, instrument=was, amended_to=now)


def apply_lifecycle(log, actor, event_type, body, book=None, effective_time=None):
    """File one lifecycle fact - an election, a fixing observation, or a determination.

    Anything outside `LIFECYCLE_TYPES` raises `UnknownEventType`, restating the closure on this
    verb's own arm: the writer would refuse a knock as an unknown type but would accept a `fill`
    submitted here. A knock, an expiry or an accrual is a consequence of terms plus one of the three
    and is read off a projection. A determination is a fact about a ruling, not about the touch it
    rules on.
    """
    if event_type not in LIFECYCLE_TYPES:
        raise UnknownEventType(
            'apply_lifecycle does not file {!r}: it files {} and nothing else, because those are '
            'the three lifecycle FACTS - the holder\'s act, the world\'s observation, and a ruling '
            'a contract vests in an agent. Everything else a lifecycle produces - a knock, an '
            'expiry, an accrual, a state that changed - is a CONSEQUENCE of terms plus one of '
            'those three, so it is derived by a fold over what is already here and never stored; '
            'storing it would be a second source of truth about whether the barrier fired. File '
            'the observation the consequence follows from, and read the consequence off the '
            'projection'.format(event_type, ', '.join(LIFECYCLE_TYPES)))
    return log.append(event_type, body, actor=actor, book=book, effective_time=effective_time)


def declare_market(log, actor, name, values, effective_time=None):
    """Point a market name at a values vector, returning the envelope plus the vector's address.

    Officialness is a property of the name, never of the data: every values vector lives identically
    in the store, and `official` moves onto one only by a declaration from a `mark`-scoped actor. A
    `private/<subject>/<name>` scratch market is the same call under a different name. Firm-level,
    so it carries no book.
    """
    address = _blob(log, values, 'the values vector')
    envelope = log.append('market_declared',
                          {'name': _name(name, 'name', 'declare_market'), 'values_hash': address},
                          actor=actor, effective_time=effective_time, blob_refs=(address,))
    return dict(envelope, name=name, values_hash=address)


def file_quote(log, actor, quote_id, structure, plan_hash, values, solved, edge,
               request=None, book=None, effective_time=None):
    """File a quote: two hashes pinned, what was solved, and what the desk took for it.

    Two hashes because a quote goes stale in two unrelated ways (see `firmness`). `values` is the
    vector the quote was struck on, as bytes, and its address becomes the pinned `values_hash`.
    `plan_hash` is the book plan the marginal charge was solved against and is not stored, since a
    plan recompiles.

    `request` is the relayed client utterance - free text, optional and erasable: the body is sealed
    under its class key, so shredding that key erases the utterance while the chain still verifies.
    The same string in the envelope would be permanent.
    """
    address = _blob(log, values, 'the values vector this quote was struck on')
    body = {'quote_id': _name(quote_id, 'quote_id', 'file_quote'),
            'structure': _name(structure, 'structure', 'file_quote'),
            'plan_hash': _pinned(plan_hash, 'plan_hash'), 'values_hash': address,
            'solved': solved, 'edge': edge}
    if request is not None:
        body['request'] = request
    envelope = log.append('quote_filed', body, actor=actor, book=book,
                          effective_time=effective_time, blob_refs=(address,))
    return dict(envelope, quote_id=quote_id, plan_hash=body['plan_hash'], values_hash=address)


def complete_run(log, actor, lane, claim, job, values, result, book=None, effective_time=None):
    """The standing lane's attestation at birth, by the executor that produced the numbers. Returns
    the envelope plus the three addresses.

    `claim` is the replay tuple as the engine reports it; `job`, `values` and `result` are the three
    objects that make it checkable, as bytes. The values address is checked against the claimed
    `values_hash` rather than believed - the engine's values hash is the SHA-256 of that vector's
    canonical bytes, so a disagreement means the vector handed in is not the one the run read.
    `plan_hash` is not checked here, since checking it means compiling; an auditor recompiles it
    from the `job` blob at this LSN.

    A lane that mints nothing raises `MalformedEvent` rather than being dropped silently.
    """
    if not mints(lane):
        raise MalformedEvent(
            'complete_run was asked to attest a {} run: only the {} lane mints, because a run is '
            'recorded IFF its output will be cited by a fact - a repaint and a what-if are '
            'superseded before anything cites them. Run it and report it; do not record it'.format(
                lane, STANDING))
    body = dict(_claim(claim), lane=lane)
    body['job'] = _blob(log, job, 'the job document')
    body['result'] = _blob(log, result, 'the result')
    body['values_hash'] = _values(log, values, body['values_hash'])
    envelope = log.append('run_completed', body, actor=actor, book=book,
                          effective_time=effective_time,
                          blob_refs=(body['job'], body['result'], body['values_hash']))
    return dict(envelope, **body)


def pin_result(log, actor, claim, job, values, result, executor, book=None, effective_time=None):
    """Promote a tuple this hub did not witness, returning the envelope, the addresses, and the
    `resolution` that reached it.

    A cache hit is a `run_completed` at or before this head carrying the same four coordinates, so
    there is nothing to reproduce. If that attestation names a different result the claim is
    refused: one replay tuple cannot have two results.

    Otherwise the injected `executor` is called as `executor(job, values, engine_version)` and must
    answer `(version it ran at, result bytes)`. A version mismatch refuses by name, since a replay
    claim is a claim at the recorded version. Bit-equality of the bytes is the fast path; anything
    else falls to `policy.compare` against the declared tolerance policy, which is read before
    anything runs, so a home that declares none refuses on every path including the cache hit.

    Nothing is appended on a refusal. A pin that succeeds twice coalesces on its idempotency tag.
    """
    tolerance_blob, tolerances = tolerances_in_force(log)
    body = _claim(claim)
    body['job'] = _blob(log, job, 'the job document')
    body['result'] = _blob(log, result, 'the claimed result')
    body['values_hash'] = _values(log, values, body['values_hash'])
    body['tolerance_policy'] = tolerance_blob

    attested = attestation(log, claim)
    if attested is not None:
        if attested['result'] != body['result']:
            raise ReplayRefused(
                'this hub already attested {} at LSN {} with result {}, and the claim names {}: '
                'one replay tuple cannot have two results, so this is a claim about another '
                'history rather than a promotion of this one. Compare the two results out of band '
                'and pin the tuple the run actually produced'.format(
                    _coordinates(claim), attested['lsn'], attested['result'], body['result']))
        resolution = 'cache hit'
    else:
        version, produced = _executed(executor, job, values, claim['engine_version'])
        if version != claim['engine_version']:
            raise ReplayRefused(
                'the claim is at engine version {!r} and the executor ran at {!r}: a replay claim '
                'is a claim AT the recorded version, and agreement at another version is a '
                'statement about different software. Re-execute on a build of {!r}, or file the '
                'run this build produced as its own attestation'.format(
                    claim['engine_version'], version, claim['engine_version']))
        if produced != bytes(result):
            departures = compare(_document(result, 'the claimed result'),
                                 _document(produced, 'the replayed result'), tolerances)
            if departures:
                raise ReplayRefused(
                    'the replay of {} does not reproduce the claimed result within the tolerance '
                    'policy {} in force here: {}. Nothing is pinned - the record attests what '
                    'reproduces, and this does not'.format(
                        _coordinates(claim), tolerance_blob, '; '.join(departures)))
        resolution = 'reproduced'

    envelope = log.append('result_pinned', body, actor=actor, book=book,
                          effective_time=effective_time,
                          blob_refs=(body['job'], body['result'], body['values_hash'],
                                     body['tolerance_policy']))
    return dict(envelope, resolution=resolution, **body)


def attestation(log, claim, lsn=None):
    """The `run_completed` body carrying `claim`'s four coordinates at or before `lsn`, with the LSN
    it sits at, or None.

    Located by envelope - only `run_completed` rows are opened - so this costs the length of the
    attestation history. `result_pinned` rows are not read: a pin is by definition a claim the hub
    did not witness, and reading them would let one promotion become evidence for the next.
    """
    coordinates = _coordinates(claim)
    for frame in log.frames(end_lsn=lsn):
        if frame['event_type'] != 'run_completed':
            continue
        body = log.open_body(frame)
        if isinstance(body, dict) and _coordinates(body) == coordinates:
            return dict(body, lsn=frame['lsn'])
    return None


# ------------------------------------------------------------------------------------------------
# The pieces every verb above is made of.

def _claim(claim):
    """`claim` checked and returned as a body's four `REPLAY_FIELDS`. A claim is four coordinates
    and no fifth."""
    if not isinstance(claim, dict):
        raise MalformedEvent(
            'the replay claim is {}, not the {} tuple every reported number replays from'.format(
                type(claim).__name__, ', '.join(REPLAY_FIELDS)))
    surplus = sorted(set(claim) - set(REPLAY_FIELDS))
    if surplus:
        raise MalformedEvent(
            'the replay claim carries {} beyond {}: a tuple with a fifth coordinate is not the '
            'tuple a result was filed under'.format(', '.join(surplus), ', '.join(REPLAY_FIELDS)))
    body = {}
    for field in REPLAY_FIELDS:
        if field not in claim:
            raise MalformedEvent(
                'the replay claim has no {}: the four coordinates are {}, and a tuple missing one '
                'of them names no result at all'.format(field, ', '.join(REPLAY_FIELDS)))
        body[field] = claim[field]
    for field in ('plan_hash', 'values_hash'):
        _pinned(body[field], field)
    if not is_text(body['engine_version']):
        raise MalformedEvent(
            'the replay claim names engine_version {!r}: a replay is at a VERSION, and an unnamed '
            'one cannot be re-executed at the version it was recorded at'.format(
                body['engine_version']))
    if body['seed'] is not None and not is_integer(body['seed']):
        raise MalformedEvent(
            'the replay claim names seed {!r}: a seed is a whole number, or null where the job '
            'declared none - a substituted zero would record a tuple no result was filed '
            'under'.format(body['seed']))
    return body


def _coordinates(claim):
    """The four coordinates as a comparable, printable tuple - what a cache hit matches on."""
    return tuple(claim.get(field) for field in REPLAY_FIELDS)


def _executed(executor, job, values, engine_version):
    """Call the injected executor and return `(version, result bytes)`, checking the shape of what
    came back so a broken executor raises `ReplayRefused` rather than a TypeError."""
    if not callable(executor):
        raise ReplayRefused(
            'pin_result was handed {} as its executor: re-execution is the one thing this package '
            'cannot do - the truth layer does not import the engine it records - so the caller '
            'passes a function taking (job, values, engine_version) and answering (version, result '
            'bytes)'.format(type(executor).__name__))
    answered = executor(bytes(job), bytes(values), engine_version)
    if not (isinstance(answered, tuple) and len(answered) == 2):
        raise ReplayRefused(
            'the executor answered {!r} rather than (version, result bytes): the version is what '
            'the claim is checked against, so an executor that does not report the version it ran '
            'at cannot be told apart from one that ran at the wrong one'.format(answered))
    version, produced = answered
    if not isinstance(produced, (bytes, bytearray)):
        raise ReplayRefused(
            'the executor answered a {} result rather than bytes: the fast path is a BYTE '
            'comparison against the claimed result, so the replay has to arrive as the bytes it '
            'would have been stored as'.format(type(produced).__name__))
    return version, bytes(produced)


def _document(raw, what):
    """One result blob read back as JSON, for the tolerance comparison. Bytes that will not read
    raise `ReplayRefused` rather than counting as a departure - there is nothing to compare."""
    try:
        return json.loads(bytes(raw).decode('utf-8'))
    except (TypeError, UnicodeDecodeError, ValueError):
        raise ReplayRefused(
            '{} is not UTF-8 JSON, so there is nothing here to compare: a result is a document of '
            'result classes, and bytes that are not one cannot be held to a per-class '
            'tolerance'.format(what))


def _blob(log, data, what):
    """Fsync `data` into the store and return its address, so no verb above can append an event
    citing bytes that are not yet on the platter. `what` names it in refusals."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise MalformedEvent(
            '{} is {}, not bytes: the record addresses what it is handed, so canonicalise the '
            'object at the caller - the spelling of an engine object is the engine\'s own and this '
            'package does not have an opinion about it'.format(what, type(data).__name__))
    return log.store.put(bytes(data))


def _values(log, values, claimed):
    """File the values vector and return its address, asserting it equals `claimed`.

    The engine's `values_hash` is the SHA-256 of this vector's canonical bytes, so citation and
    store address are one number; a disagreement means the vector handed in is not the one the run
    read.
    """
    address = _blob(log, values, 'the values vector')
    if address != claimed:
        raise MalformedEvent(
            'the claim names values_hash {} and the vector handed in addresses {}: the engine\'s '
            'values hash IS the hash of that vector\'s canonical bytes, so these are two spellings '
            'of one number and they disagree - hand in the vector the run actually read'.format(
                claimed, address))
    return address


def _pinned(value, field):
    """`value` asserted to be a content hash. A pin that is not a content address is not a pin."""
    if not is_hash(value):
        raise MalformedEvent(
            '{} is {!r}, which is not a content hash: every reference this record makes to another '
            'object is 64 lowercase hex'.format(field, value))
    return value


def _name(value, field, verb):
    """`value` asserted to be a non-empty string, checked at the verb rather than only at the
    validator so the refusal names `verb`."""
    if not is_text(value):
        raise MalformedEvent(
            '{}: {} is {!r}, and a name that names nothing is not a name'.format(verb, field, value))
    return value
