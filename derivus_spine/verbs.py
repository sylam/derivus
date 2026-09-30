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

"""The acts a desk performs on the record - booking, amending, lifecycle, decisions, marks, quotes
and runs - and the paper it trades under: its counterparties and their agreements.

The logic lives here and the engine holds thin delegators, so no module under `derivus/` learns
about users, workflow or storage: everything below takes plain data and injected callables.
`pin_result` therefore takes an EXECUTOR rather than importing one, and the record checks what came
back rather than trusting who ran it, the engine version included. No verb here carries an
authorization check - the writer evaluates capability and lands the refusal as `capability_denied`,
and a second check is a second place to get it wrong. Three verbs do restate a refusal the writer
would give anyway, each on its own arm and each saying so in its own docstring: `apply_lifecycle`,
`complete_run` and `declare_market`.

DECLARED LIMITATION: the desk's `book.json` is still the edge's own file, written after the event
lands. Its rehoming as an LSN-pinned projection is not built, so the log is what is true and the
file is an interim stand-in.
"""
import json

from .canon import content_hash
from .errors import MalformedEvent, ReplayRefused, UnknownEventType
from .policy import PRIVATE_MARKET, compare, tolerances_in_force
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

#: What a settlement may move, and the one status money moves under. A `payment` settles the diary
#: row its subject keys, as a bare `settled` does; a `fee` is money a trade cost that no row
#: announced, filed on its instrument; `collateral` and `margin` move a balance held under an
#: agreement. Only a payment moves a state - the other three move money and nothing else. A trade
#: matched with its counterparty is `confirmed`, filed against its fill key.
PAYMENT, FEE, COLLATERAL, MARGIN = 'payment', 'fee', 'collateral', 'margin'
MOVEMENTS = (PAYMENT, FEE, COLLATERAL, MARGIN)
HELD = (COLLATERAL, MARGIN)
SETTLED, CONFIRMED = 'settled', 'confirmed'

#: The book a marks job is filed under - `marks:<book>@<lsn>`, a name no book carries so a compile
#: writes no position into it, the position its cut read the book at following the LAST '@'.
MARKS = 'marks:'


def marks_name(book, cut=None):
    """The name a marks job of `book` is filed under, cut at the position `cut` where one is
    named."""
    return MARKS + book + ('' if cut is None else '@{}'.format(cut))


def marks_of(reference):
    """`(book, cut)` for the name a marks job is filed under - `cut` None where it names no
    position - or None where `reference` names no marks job."""
    if not isinstance(reference, str) or not reference.startswith(MARKS):
        return None
    book, at, cut = reference[len(MARKS):].rpartition('@')
    return (book, int(cut)) if at and cut.isdigit() else (reference[len(MARKS):], None)


def fill_key(instrument, execution_reference):
    """The stable id of one fill, which a confirmation is filed against: the content hash of its
    instrument and the reference it was executed under."""
    return content_hash({'instrument': instrument, 'execution_reference': execution_reference})


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
         book=None, effective_time=None, price=None, agreement=None, portfolio=None,
         currency=None, rate=None, ticket=None):
    """Book a fill, returning the envelope plus the instrument's address.

    `instrument` is the canonical JSON of the deal's terms as bytes - the caller canonicalises,
    since the spelling of an instrument is the engine's own - and its hash is the instrument id, so
    booking the same strike twice files two events against one store row.

    `execution_reference` is required and has no default. It is what makes a retry the same fact by
    construction, coalescing onto the LSN it already has, and two legitimately identical clips two
    facts. `quantity` is the signed position CHANGE in units of the instrument - 1 is the instrument
    as written, -1 closes it, -0.5 unwinds half - never a position: position is a fold. `agreement`
    and `portfolio` say where the position sits and `price` what it was done at, each filed only
    where stated. A price stated in another currency than the book's carries that `currency` and
    the `rate` the booking crossed it at - units of the book's currency per unit of it - so the
    consideration is on the record as it was agreed and as it was booked. `ticket` is what an
    approval of this booking signs. The instrument blob is fsynced before the event citing it
    appends.
    """
    body = fill(log, instrument, quantity, counterparty, netting_set, execution_reference,
                price=price, agreement=agreement, portfolio=portfolio, currency=currency,
                rate=rate)
    if ticket is not None:
        body['ticket'] = _pinned(ticket, 'ticket')
    envelope = log.append('fill', body, actor=actor, book=book, effective_time=effective_time)
    return dict(envelope, instrument=body['instrument'])


def fill(log, instrument, quantity, counterparty, netting_set, execution_reference, price=None,
         agreement=None, portfolio=None, currency=None, rate=None):
    """The body `book` files, every field it asserts checked and the instrument blob fsynced -
    what a caller routing the fill first asks the writer about before anything appends."""
    address = _blob(log, instrument, 'the canonical instrument')
    if not is_number(quantity):
        raise MalformedEvent(
            'book: quantity is {!r} - a fill carries a SIGNED quantity of the instrument, as a '
            'finite number; position is a fold over these and is never written'.format(quantity))
    body = {'instrument': address, 'quantity': quantity,
            'counterparty': _name(counterparty, 'counterparty', 'book'),
            'netting_set': _name(netting_set, 'netting_set', 'book'),
            'execution_reference': _name(execution_reference, 'execution_reference', 'book')}
    if price is not None:
        if not is_number(price):
            raise MalformedEvent(
                'book: price is {!r} - what a fill was done at is a finite number in the '
                'instrument\'s own quote, or it is left out'.format(price))
        body['price'] = price
    if currency is not None or rate is not None:
        if price is None or not is_number(rate) or rate <= 0:
            raise MalformedEvent(
                'book: a price in another currency is the price, its `currency` and the `rate` it '
                'was crossed at, a positive number - this states price {!r}, currency {!r} and '
                'rate {!r}'.format(price, currency, rate))
        body['currency'], body['rate'] = _name(currency, 'currency', 'book'), rate
    return _placed(body, 'book', agreement=agreement, portfolio=portfolio)


def amend(log, actor, instrument, amended_to, book=None, effective_time=None, portfolio=None):
    """Amend a booked deal: a new instrument hash linked to the old one. Returns the envelope plus
    both addresses.

    Economics are never edited, so this is a second row saying these terms became those. Both
    instruments are registered because both are cited; the old one dedups to the address it already
    has, so a deal booked before this home existed still closes referentially. Terms that
    canonicalise to the same hash raise `MalformedEvent`. `portfolio` is the deepest node holding
    every position in the terms, which is where it is judged.
    """
    body = amendment(log, instrument, amended_to, portfolio=portfolio)
    envelope = log.append('amendment', body, actor=actor, book=book,
                          effective_time=effective_time)
    return dict(envelope, instrument=body['instrument'], amended_to=body['amended_to'])


def amendment(log, instrument, amended_to, portfolio=None):
    """The body `amend` files, both instruments' blobs fsynced and terms that did not move
    refused - what a caller routing the restrike first asks the writer about."""
    was = _blob(log, instrument, 'the canonical instrument as it was')
    now = _blob(log, amended_to, 'the canonical instrument as amended')
    if was == now:
        raise MalformedEvent(
            'amend: the amended terms canonicalise to the same instrument {} - an amendment is a '
            'NEW instrument hash linked to the old one, so terms that did not move are not an '
            'amendment; file the operational fact as a status_transition instead'.format(was))
    return _placed({'instrument': was, 'amended_to': now}, 'amend', portfolio=portfolio)


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
            'expiry, an accrual - is a CONSEQUENCE of terms plus one of those three, so it is '
            'derived by a fold over what is already here and never stored; storing it would be a '
            'second source of truth about whether the barrier fired. File the observation the '
            'consequence follows from, and read the consequence off the projection. An operational '
            'state a party MOVED - a settlement paid, a confirmation matched - is a fact rather '
            'than a consequence, and `transition` files it'.format(
                event_type, ', '.join(LIFECYCLE_TYPES)))
    return log.append(event_type, body, actor=actor, book=book, effective_time=effective_time)


def transition(log, actor, subject, status, book=None, effective_time=None, amount=None,
               asset=None, kind=None, reference=None):
    """Move the operational state a party put a subject in - a settlement paid, a confirmation
    matched - and return the envelope.

    `subject` is an ADDRESS: the derived cashflow key a diary row carries, or the instrument a
    trade books under. The vocabulary takes any name there, because a later subject may be keyed
    another way, and this verb is the narrower one - a state filed against something nobody can
    resolve is a state nobody can read back.

    WHETHER THE BOOK ANNOUNCES A ROW under that key is not asked here. The record holds what it was
    told and a fold says what answers for it, so a transition against a key the diary has dropped
    is a FACT this verb files and an invariant the oracle reads, never a refusal at the writer.

    A SETTLEMENT THAT MOVED MONEY SAYS HOW MUCH: the signed `amount` from the bank's side -
    received positive, paid or posted negative - the `asset` it is an amount of, its `kind`, and
    the settlement system's own `reference`, under which a retry is the same fact, two identical
    movements are two, and a corrected amount restates the one it corrects. See `_movement`.
    """
    body = {'subject': subject, 'status': _name(status, 'status', 'transition')}
    if amount is None:
        stated = [field for field, value in (('asset', asset), ('kind', kind),
                                             ('reference', reference)) if value is not None]
        if stated:
            raise MalformedEvent(
                'transition: {} describe money that moved and no amount was stated - file the '
                'amount with them, or none of them for a state that moved no money'.format(
                    ', '.join(stated)))
        body['subject'] = _pinned(subject, 'subject')
    else:
        body.update(_movement(log, subject, status, amount, asset, kind, reference,
                              effective_time))
    return log.append('status_transition', body, actor=actor, book=book,
                      effective_time=effective_time)


def approve(log, actor, plan_hash, book=None, effective_time=None, portfolio=None):
    """Sign a plan: an approval over the hash that identifies the ticket, judged at the `portfolio`
    the ticket books into where one is named.

    Retried by the same seat it COALESCES onto the LSN it already has, since the semantic tuple
    carries no clock of the writer's own - a second signature of one plan by one seat is one fact.
    An amended plan is a different hash and so is a different signature.
    """
    return log.append('approval', _placed({'plan_hash': _pinned(plan_hash, 'plan_hash')},
                                          'approve', portfolio=portfolio),
                      actor=actor, book=book, effective_time=effective_time)


def reject(log, actor, plan_hash, reason, book=None, effective_time=None, portfolio=None):
    """Refuse a plan, with the reason on the row, judged where `approve` is.

    The reason is required and has no default: a verdict is never withdrawn, so a rejection nobody
    can read the grounds of is one nothing can be filed against later.
    """
    return log.append('rejection', _placed({'plan_hash': _pinned(plan_hash, 'plan_hash'),
                                            'reason': _name(reason, 'reason', 'reject')},
                                           'reject', portfolio=portfolio),
                      actor=actor, book=book, effective_time=effective_time)


def declare_portfolio(log, actor, path, effective_time=None):
    """Declare a node of the desk's tree - a path whose top node is the book it is filed under -
    and return the envelope. Judged at its parent, so `admin` at `BANK/FX` declares
    `BANK/FX/Options` and a book itself is the firm's to declare. One path declared twice by one
    seat is one fact."""
    return log.append('portfolio_declared', {'path': path}, actor=actor,
                      book=_name(path, 'path', 'declare_portfolio').split('/')[0],
                      effective_time=effective_time)


def declare_market(log, actor, name, values, effective_time=None):
    """Point a market name at a values vector, returning the envelope plus the vector's address.

    Officialness is a property of the name, never of the data: every values vector lives identically
    in the store, and `official` moves onto one only by a declaration from a `mark`-scoped actor. A
    `private/<subject>/<name>` scratch market is the same call under a different name, with one
    rule: SELF-DECLARED MEANS SELF-DECLARED, so the subject in the name must be the seat declaring
    it. Otherwise a seat could mint a market inside another's namespace and own it while the named
    seat was refused their own prefix, and a reader would have two answers to who owns one board.
    Firm-level, so it carries no book.
    """
    name = _own(_name(name, 'name', 'declare_market'), actor, 'declare_market')
    address = _blob(log, values, 'the values vector')
    envelope = log.append('market_declared', {'name': name, 'values_hash': address},
                          actor=actor, effective_time=effective_time, blob_refs=(address,))
    return dict(envelope, name=name, values_hash=address)


def declare_close(log, actor, market, values, effective_time=None, date=None):
    """Declare the official close on `market` over a values vector, returning the envelope plus the
    vector's address.

    The values are blobbed exactly as `declare_market` blobs them, so a close and the mark it stands
    on are one address. A SECOND close on one market supersedes the first rather than correcting it
    - the `markets` fold names the LSN it stands over - so a day restated is two facts and an as-at
    read taken before the restatement still reads what it read. Firm-level, so it carries no book.

    The owner rule is `declare_market`'s and is asked here for the same reason: a close is one way
    of declaring what a name stands on, and one landing inside another seat's `private/` namespace
    would be a board its owner never declared and the only reader who can resolve it.

    `date` is the calendar day the close is declared FOR (`YYYY-MM-DD`), which is what a cash
    movement on that day is converted at - a close struck the next morning is still that day's.
    """
    address = _blob(log, values, 'the values vector this close stands on')
    body = {'market': _own(_name(market, 'market', 'declare_close'), actor, 'declare_close'),
            'values_hash': address}
    if date is not None:
        body['date'] = _day(date, 'declare_close')
    envelope = log.append('official_close_declared', body, actor=actor,
                          effective_time=effective_time, blob_refs=(address,))
    return dict(envelope, market=market, values_hash=address)


def declare_entity(log, actor, entity, name, parent=None, effective_time=None):
    """Declare a legal entity the book may trade with, and return the envelope.

    `entity` is its id as the deployment's legal system knows it and `name` what a reader is shown;
    `parent` groups it under another entity and says nothing else. A second declaration under one
    id restates it. Firm-level, so it carries no book.
    """
    body = {'entity': _name(entity, 'entity', 'declare_entity'),
            'name': _name(name, 'name', 'declare_entity')}
    if parent is not None:
        body['parent'] = _name(parent, 'parent', 'declare_entity')
    return log.append('entity_declared', body, actor=actor, effective_time=effective_time)


def declare_agreement(log, actor, agreement, entity, kind, terms, effective_time=None):
    """Declare an agreement with a legal entity, returning the envelope plus the terms' address.

    `terms` is the canonical JSON of the netting set the agreement's positions are compiled into,
    as bytes - the caller canonicalises and validates, the engine knowing what a netting set may
    say - and `kind` the document's label as its declarer states it. A second declaration under one
    id restates the agreement. Firm-level, so it carries no book.
    """
    address = _blob(log, terms, 'the canonical agreement terms')
    envelope = log.append('agreement_declared',
                          {'agreement': _name(agreement, 'agreement', 'declare_agreement'),
                           'entity': _name(entity, 'entity', 'declare_agreement'),
                           'kind': _name(kind, 'kind', 'declare_agreement'), 'terms': address},
                          actor=actor, effective_time=effective_time, blob_refs=(address,))
    return dict(envelope, agreement=agreement, terms=address)


def file_quote(log, actor, quote_id, structure, plan_hash, values, solved, edge,
               request=None, ticket=None, book=None, effective_time=None, portfolio=None):
    """File a quote: two hashes pinned, what was solved, and what the desk took for it.

    Two hashes because a quote goes stale in two unrelated ways (see `firmness`). `values` is the
    vector the quote was struck on, as bytes, and its address becomes the pinned `values_hash`.
    `plan_hash` is the book plan the marginal charge was solved against and is not stored, since a
    plan recompiles.

    `request` is the relayed client utterance - free text, optional and erasable: the body is sealed
    under its class key, so shredding that key erases the utterance while the chain still verifies.
    The same string in the envelope would be permanent.

    `ticket` is the plan an approval of this quote would sign, a different hash for every quote,
    and `portfolio` the node it books into. Optional, because a body filed before them validates
    exactly as it did.
    """
    address = _blob(log, values, 'the values vector this quote was struck on')
    body = {'quote_id': _name(quote_id, 'quote_id', 'file_quote'),
            'structure': _name(structure, 'structure', 'file_quote'),
            'plan_hash': _pinned(plan_hash, 'plan_hash'), 'values_hash': address,
            'solved': solved, 'edge': edge}
    if request is not None:
        body['request'] = request
    if ticket is not None:
        body['ticket'] = _pinned(ticket, 'ticket')
    envelope = log.append('quote_filed', _placed(body, 'file_quote', portfolio=portfolio),
                          actor=actor, book=book, effective_time=effective_time,
                          blob_refs=(address,))
    return dict(envelope, quote_id=quote_id, plan_hash=body['plan_hash'], values_hash=address)


def complete_run(log, lane, claim, job, values, result, book=None, effective_time=None):
    """The standing lane's attestation at birth, in the writer's own voice: the hub that executed
    the run is what says it ran. Returns the envelope plus the three addresses.

    `claim` is the replay tuple as the engine reports it; `job`, `values` and `result` are the three
    objects that make it checkable, as bytes. The values address is checked against the claimed
    `values_hash` rather than believed. `plan_hash` is not checked here, since checking it means
    compiling; an auditor recompiles it from the `job` blob at this LSN. A lane that mints nothing
    raises `MalformedEvent` rather than being dropped silently.
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
    envelope = log.own('run_completed', body, book=book, effective_time=effective_time,
                       blob_refs=(body['job'], body['result'], body['values_hash']))
    return dict(envelope, **body)


def pin_result(log, actor, claim, job, values, result, executor, book=None, effective_time=None):
    """Promote a tuple this hub did not witness, returning the envelope, the addresses, and the
    `resolution` that reached it.

    A cache hit is a `run_completed` at or before this head carrying the same four coordinates; if
    that attestation names a different result the claim is refused, since one replay tuple cannot
    have two. Otherwise the injected `executor` is called as `executor(job, values, engine_version)`
    and must answer `(version it ran at, result bytes)`, a version mismatch refusing by name.
    Bit-equality is the fast path; anything else falls to `policy.compare` against the declared
    tolerance policy, read before anything runs, so a home that declares none refuses on every path
    including the cache hit.

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


def _placed(body, verb, **named):
    """`body` with every name in `named` that is stated - where a fact sits, filed only where the
    caller said, each asserted to name something."""
    body.update((field, _name(value, field, verb)) for field, value in named.items()
                if value is not None)
    return body


def _movement(log, subject, status, amount, asset, kind, reference, effective_time):
    """The four fields a settlement that moved money adds to its body, each asserted.

    Money moves under `settled` alone, on a value date the settlement states and never on the day
    the record heard about it. A `payment` or a `fee` names an address - the diary row it settles,
    or the instrument a fee was paid on - and `collateral` or `margin` names an agreement THE RECORD
    DECLARES, the one place such a balance is held: an id nothing declared is a balance nobody can
    read back under the paper it moved under.
    """
    if not is_number(amount) or amount == 0:
        raise MalformedEvent(
            'transition: amount is {!r} - money that moved is a finite, non-zero number, signed from '
            "the bank's side: received positive, paid or posted negative".format(amount))
    if status != SETTLED:
        raise MalformedEvent(
            'transition: money moves under {!r} alone and this says {!r} - file the state without '
            'an amount, or the amount once it settled'.format(SETTLED, status))
    if kind not in MOVEMENTS:
        raise MalformedEvent(
            'transition: kind is {!r}, not one of {} - a {} settles a diary row, a {} is money a '
            'trade cost on its instrument, and {} and {} move a balance held under an '
            'agreement'.format(kind, ', '.join(MOVEMENTS), PAYMENT, FEE, COLLATERAL, MARGIN))
    if effective_time is None:
        raise MalformedEvent(
            'transition: money moves on a value date, and this movement states none - file it with '
            'the effective time it settled on')
    if kind in HELD:
        from .projections import PROJECTORS, fold

        _name(subject, 'subject', 'transition')
        if subject not in fold(log, PROJECTORS['agreements']):
            raise MalformedEvent(
                'transition: {} is held under an agreement and {!r} is none the record declares - '
                'declare the agreement first, since a balance is read back under the paper it '
                'moved under'.format(kind, subject))
    else:
        _pinned(subject, 'subject')
    return {'amount': amount, 'asset': _name(asset, 'asset', 'transition'), 'kind': kind,
            'reference': _name(reference, 'reference', 'transition')}


def _day(value, verb):
    """`value` asserted to be a calendar day, exactly `YYYY-MM-DD` - two spellings of one day would
    be two facts."""
    import datetime

    try:
        if isinstance(value, str) and datetime.date.fromisoformat(value).isoformat() == value:
            return value
    except ValueError:
        pass
    raise MalformedEvent('{}: date is {!r}, and a day is exactly YYYY-MM-DD'.format(verb, value))


def _own(name, actor, verb):
    """A `private/` market name asserted to name its own declarer. `verb` names the caller.

    The name carries the owner and the fold carries the declarer; binding them here makes them one
    seat, so a reader may resolve ownership off either and get the same answer. EVERY verb that
    moves what a name stands on asks it - a close is one way of declaring a market, so a close
    inside another seat's namespace would be the second answer the rule exists to prevent.
    """
    parts = name.split('/')
    if parts[0] + '/' == PRIVATE_MARKET and (len(parts) < 2 or parts[1] != actor):
        raise MalformedEvent(
            '{}: {!r} is a {} market of {!r} and {!r} is declaring it - a private '
            'market is one seat\'s own board, so the subject in the name IS the seat that declares '
            'it. Declare {}{}/... , or drop the prefix and name a market the firm holds'.format(
                verb, name, PRIVATE_MARKET, parts[1] if len(parts) > 1 else '(nobody)', actor,
                PRIVATE_MARKET, actor))
    return name
