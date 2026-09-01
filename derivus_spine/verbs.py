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

The brief's build order says "booking verbs on Context", and the house's first law says no module
under `derivus/` learns about users, workflow or storage. Both are kept by putting the LOGIC here
and leaving the engine a set of thin delegators: everything below takes plain data and injected
callables - bytes, strings, numbers, a function - and holds no idea that an engine exists. The
engine side reaches this module lazily, the way `service.py` reaches fastapi, and refuses by name
when the extra is not installed or no home is configured.

That is also why `pin_result` takes an EXECUTOR rather than importing one. Re-executing a claim is
the one thing this package cannot do and must not learn to do: it would mean the truth layer
depending on the thing it records, which is the dependency the whole workstream exists to refuse.
So the caller hands in a function, the function is a real function (never a patch of library code,
which is what makes the gates honest), and the record checks what came back rather than trusting
who ran it - including the engine version, which the executor REPORTS and this module compares
against the claim rather than taking on faith.

Every verb rides what increments 1 and 2 already built and adds nothing beside it. The writer's
flow is untouched: blind tag, duplicate-tag coalescing on identical canonical bytes, blob fsynced
before the event that cites it, one exclusive claim on the home. Capability evaluation is
untouched: an unscoped actor is refused by the writer and the refusal lands as `capability_denied`,
so none of these verbs carries an authorization check of its own - a second one would be a second
place to get it wrong.

Three of them restate a refusal the writer would give anyway, and each restatement earns its place
by naming the VERB. `apply_lifecycle` refuses anything that is not an election, an observation or a
determination - the writer refuses a knock because `knocked_out` is not in the vocabulary, but it
would happily accept a `fill` submitted through this arm, and a lifecycle verb that booked a trade
is a bug the closed vocabulary cannot see. `complete_run` refuses a lane that mints nothing.
`declare_market` refuses a name that is not a name. Everything else is the writer's job.

WHAT THIS INCREMENT DOES NOT BUILD, said here because the dual write is where somebody will look
for it: the desk's `book.json` is still the edge's own file and is still written by the edge. The
event goes first and the file follows, which is the fsync law applied to the pair, but the file's
formal rehoming as an LSN-pinned PROJECTION - hydrated from the centre, rebuilt by replaying rather
than edited - is increment 4's business. Until then the file is the interim stand-in the brief
calls it, and the log is what is true.
"""
import json

from .errors import MalformedEvent, ReplayRefused, UnknownEventType
from .policy import compare, tolerances_in_force
from .vocabulary import is_hash, is_integer, is_number, is_text

#: The three attestation lanes, and the whole of them. The rule compresses to one sentence - A RUN
#: IS RECORDED IFF ITS OUTPUT WILL BE CITED BY A FACT - and these are the three answers a caller can
#: give to it. `telemetry` is the blotter's thirty-second repaint: superseded before it could be
#: cited, a reading rather than a record. `curiosity` is a local what-if or an adjudicated
#: exploration: nobody will cite it either. `standing` is the quote's solve, the close's batch, the
#: recommendation acted on - a citation is coming, and fast, so the executor attests at birth.
TELEMETRY = 'telemetry'
CURIOSITY = 'curiosity'
STANDING = 'standing'
LANES = (TELEMETRY, CURIOSITY, STANDING)
#: The lanes that mint. Exactly one, and the tuple rather than an equality test because the day
#: hub-executed exploration earns standing it is a second entry here and nothing else moves.
MINTING = (STANDING,)

#: What `apply_lifecycle` may file, per the facts law: an election is the holder's act, an
#: observation is the world's, a determination is a ruling where a contract vests the call in an
#: agent. A knock, an expiry or an accrual is none of these - it is a consequence of terms plus one
#: of these - so it is a projection and this verb says so by name.
LIFECYCLE_TYPES = ('election', 'fixing_observed', 'determination')

#: The four coordinates a reported number replays from. Named here so the claim a verb is handed
#: and the body it writes cannot part company.
REPLAY_FIELDS = ('plan_hash', 'values_hash', 'engine_version', 'seed')


def check_lane(lane):
    """`lane` if it is one of the three, or a refusal naming them. An unknown lane is not a lane
    with unknown rules - it is a caller who has not decided whether their output will be cited."""
    if lane not in LANES:
        raise MalformedEvent(
            'lane {!r} is not one of {} - a run is recorded IFF its output will be cited by a '
            'fact, so a lane is that decision written down: {} for a reading that is superseded '
            'before anything could cite it, {} for a what-if, {} for a run a fact is about to name'
            .format(lane, ', '.join(LANES), TELEMETRY, CURIOSITY, STANDING))
    return lane


def mints(lane):
    """Whether a run in this lane appends anything at all. Telemetry and curiosity mint NOTHING,
    and that is the rule rather than an optimisation: an event about a number nobody will cite is
    a row every later fold has to read and no later fact ever refers to."""
    return check_lane(lane) in MINTING


def book(log, actor, instrument, quantity, counterparty, netting_set, execution_reference,
         book=None, effective_time=None):
    """Book a fill: the canonical instrument registered, the signed quantity recorded. Answers the
    envelope plus the instrument's own address.

    `instrument` is the canonical JSON of the deal's terms, as BYTES - the caller canonicalises,
    because the spelling of an instrument is the engine's own and the record addresses whatever it
    is handed. Its hash IS the instrument id, so booking the same strike twice finds one row in the
    store and files two events against it, which is the brief's sentence about `create_time`
    degrading to first-seen.

    The EXECUTION REFERENCE is required and there is no form of this verb without one. That is what
    makes a retry the same fact by construction - the same clip retried canonicalises to the same
    semantic tuple, meets its own idempotency tag and coalesces onto the LSN it already has - and
    two legitimately identical clips two facts by construction, because the venue gave them
    different ids. A booking verb that defaulted this would be a booking verb whose retry story was
    a hope.

    The order is the durability law: the instrument blob is fsynced, then the event citing it
    appends. Quantity is SIGNED and it is a quantity, never a position - position is a fold.
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
    """Amend a booked deal: a NEW instrument hash, linked to the old one. Answers the envelope plus
    both addresses.

    Economics are never edited, so an amendment is not a change to a row - it is a second row
    saying that these terms became those terms. Both instruments are registered because both are
    cited: the old one is re-put and dedups to the address it already has, which costs nothing and
    means a deal booked before this home existed still closes referentially when it is amended.
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

    THE CLOSURE, restated on this verb's own arm. The writer refuses `knocked_out` because it is
    not in the vocabulary; it would not refuse a `fill` submitted here, and a lifecycle verb that
    quietly booked a trade is a hole the closed vocabulary cannot see. So this arm names the three
    it files and says why there is no fourth: a knock, an expiry, an accrual, a barrier that fired,
    a position that rolled - every one of them is a CONSEQUENCE of terms plus one of the three, so
    every one of them is a projection, derived by a fold from what is already here. Storing one
    would create the second source of truth this design exists to forbid, and it would let a
    reconstruction gate pass while the book and the pricers disagreed about whether the barrier
    fired.

    A determination is the one that looks like a consequence and is not: it is a fact about a
    RULING - an agent called the touch, a dispute was settled - attributed to the actor whose
    judgment the contract vests it in, which is a different kind of thing from the touch itself.
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
    """Point a market NAME at a values vector. Answers the envelope plus the vector's address.

    Officialness is a property of the NAME, never of the data: all values vectors live identically
    in the store, and `official` moves onto one only by a declaration from a `mark`-scoped actor -
    which this verb does not check, because the writer already does and a second check is a second
    place to get it wrong. A `private/<subject>/<name>` scratch market is the same call under a
    different name, and it is private from colleagues rather than from the record.

    Firm-level, so it carries no book: a market is the deployment's reference, not a desk's, and a
    grant over one book is not a licence to move what `official` means.
    """
    address = _blob(log, values, 'the values vector')
    envelope = log.append('market_declared',
                          {'name': _name(name, 'name', 'declare_market'), 'values_hash': address},
                          actor=actor, effective_time=effective_time, blob_refs=(address,))
    return dict(envelope, name=name, values_hash=address)


def file_quote(log, actor, quote_id, structure, plan_hash, values, solved, edge,
               request=None, book=None, effective_time=None):
    """File a quote: TWO hashes pinned, what was solved, and what the desk took for it.

    The two hashes are the whole point and they are two because a quote goes stale in two unrelated
    ways - see `firmness`. `values` is the vector the quote was struck on, as bytes, and its
    address is the pinned `values_hash`; `plan_hash` is the book plan the marginal charge was
    solved against and is NOT stored, because a plan recompiles and the record never trusts what it
    can re-derive.

    `request` is what the client asked for, relayed - free text, optional, and ERASABLE. It needs no
    mechanism: every body in this log is sealed under its class key, so destroying that key erases
    the utterance while the chain over it still verifies, which is what crypto-shredding is and is
    this system's whole answer to an erasure regime. It is why the field may live in the body at
    all; the same string in the ENVELOPE would be permanent.
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
    """The standing lane's attestation at birth. Answers the envelope plus the three addresses.

    The executor is the only first-hand witness of what it produced, so it says so at the moment it
    produced it rather than leaving somebody to assert it afterwards - which is the difference
    between this verb and `pin_result`, and the reason they carry different capability scopes.

    `claim` is the replay tuple as the engine reports it; `job`, `values` and `result` are the
    three objects that make it checkable, as bytes. The VALUES ADDRESS IS CHECKED against the
    claimed `values_hash` rather than believed: the engine's values hash is the SHA-256 of that
    vector's canonical bytes, so the two are the same number and a disagreement means the caller
    handed in a vector that is not the one it ran on. `plan_hash` cannot be checked here and is not
    - checking it means compiling, which is the engine's job; the auditor recompiles it from the
    `job` blob at this LSN, which is what makes storing that blob worth the bytes.

    A lane that mints nothing is refused rather than quietly dropped: a caller asking to attest a
    telemetry repaint has misunderstood which lane it is in, and silence would leave them thinking
    the record holds something it does not.
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
    """Promote a tuple this hub did not witness - after re-executing it, or after finding it
    already attested. Answers the envelope, the addresses, and HOW it resolved.

    Three paths and only one of them runs anything.

    A CACHE HIT is content addressing already knowing the answer: a `run_completed` at or before
    this head carries the same four coordinates, so the hub witnessed this run itself and there is
    nothing to reproduce. If that attestation names a DIFFERENT result, the claim contradicts the
    hub's own record and is refused by name - two results under one replay tuple is the one thing
    the tuple exists to make impossible.

    A RE-EXECUTION is the injected `executor`, called as `executor(job, values, engine_version)`
    and answering `(the version it ran at, the result bytes)`. The version it reports is compared
    against the claim and a mismatch refuses BY NAME, because a replay claim is a claim AT the
    recorded version: agreeing at some other version is a different statement about different
    software. Bit-equality of the bytes is the fast path and is where the honest case lands.

    A TOLERANCE COMPARISON is what remains, and the epsilon is never this module's. The declared
    tolerance policy is read at the top - before anything runs, so a home that has declared none
    refuses on every path including the cache hit, which is right: a pin is an attestation made
    under a standard, and the record must not carry one made under no standard at all. Per-result
    class, absolute, declared; a class the policy does not name is a refusal rather than a pass.

    Nothing is appended on any refusal, so a claim that will not reproduce leaves the head where it
    found it. A pin that succeeds twice coalesces on its own idempotency tag, because pinning the
    same tuple under the same policy is one fact however many times it is proposed.
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
    it sits at - or None.

    A fold over the ENVELOPE first: only `run_completed` rows are opened, so this costs the length
    of the attestation history rather than of the book. `result_pinned` rows are deliberately NOT
    read: a cache hit means the hub itself witnessed the run, and a pin is by definition a claim it
    did not witness. Reading pins here would let one unverified promotion become the evidence for
    the next, which is a claim bootstrapping itself into a fact.
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
    """The replay tuple as a body's own four fields, checked. A claim is four coordinates and no
    fifth: anything else the caller knows about the run is not what the numbers replay from."""
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
    """The four coordinates as a comparable, printable tuple - what a cache hit matches on and what
    a refusal names the run by."""
    return tuple(claim.get(field) for field in REPLAY_FIELDS)


def _executed(executor, job, values, engine_version):
    """Call the injected executor and check the SHAPE of what came back.

    A callable somebody handed in is not trusted to answer the contract, because the alternative to
    checking is a TypeError raised from inside an attestation path with a traceback where a refusal
    should be.
    """
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
    """One result blob read back as JSON, for the tolerance comparison. A result that will not read
    is a refusal rather than a departure: there is nothing here to compare, which is a different
    fact from two numbers disagreeing."""
    try:
        return json.loads(bytes(raw).decode('utf-8'))
    except (TypeError, UnicodeDecodeError, ValueError):
        raise ReplayRefused(
            '{} is not UTF-8 JSON, so there is nothing here to compare: a result is a document of '
            'result classes, and bytes that are not one cannot be held to a per-class '
            'tolerance'.format(what))


def _blob(log, data, what):
    """Fsync `data` into the store and answer its address. The durability law in one call, so that
    no verb above can append an event citing bytes that are not yet on the platter."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise MalformedEvent(
            '{} is {}, not bytes: the record addresses what it is handed, so canonicalise the '
            'object at the caller - the spelling of an engine object is the engine\'s own and this '
            'package does not have an opinion about it'.format(what, type(data).__name__))
    return log.store.put(bytes(data))


def _values(log, values, claimed):
    """The values vector filed, and the claim about its address CHECKED rather than believed.

    The engine's `values_hash` is the SHA-256 of this vector's own canonical bytes, so the citation
    and the store address are one number. A disagreement is the caller handing in a vector that is
    not the one the run read, and attesting it would put a tuple in the record pointing at bytes
    that never produced it.
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
    """One pinned hash, checked. A pin that is not a content address is not a pin."""
    if not is_hash(value):
        raise MalformedEvent(
            '{} is {!r}, which is not a content hash: every reference this record makes to another '
            'object is 64 lowercase hex'.format(field, value))
    return value


def _name(value, field, verb):
    """One named string, checked at the verb rather than only at the validator, so the refusal says
    which verb was being asked for."""
    if not is_text(value):
        raise MalformedEvent(
            '{}: {} is {!r}, and a name that names nothing is not a name'.format(verb, field, value))
    return value
