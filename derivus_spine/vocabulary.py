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

"""What the log is allowed to say, and what each saying of it demands of the sayer.

The vocabulary is CLOSED, and the closure is the facts-only law with teeth. A knock is not a fact:
it is a consequence of terms plus a recorded observation, and a log that stored it would hold a
second source of truth about whether the barrier fired - so `knocked_out` is not a type the writer
declines to validate, it is a type that does not exist, and the refusal says so. Expiries, accruals,
positions and lifecycle state are the same argument: every one of them is a fold over what is here.

What IS here divides in two. Most types are observations of the world - a fill with its execution
reference, a fixing under its (index, date, source) key, an election, a status transition. The rest
are acts of judgment or of governance, which are facts ABOUT a ruling rather than the ruling's
consequences: a `determination` where a contract vests the call in an agent, an `approval` over a
plan hash, a `market_declared` moving a name onto a values vector, a `retention_declared` without
which no blob class may ever reduce.

Three validation rules, each with a reason.

Bulk never inlines: a field naming a surface, a plan, a values vector or a policy document is 64
lowercase hex and the object lives once in the blob store. The log inlines a value only when the
value IS the fact and tiny - a fixing print rides in its body; a cube never does.

Surplus keys are refused everywhere except `policy_declared`. A body that carries a field this
vocabulary does not know is a field no projector will fold and no auditor can rely on, so it is a
mistake caught at the writer rather than a silence discovered in year three. `policy_declared` is
open because policy documents are DATA whose shape is versioned by the policy itself.

A string field is a name, and the empty string names nothing: every declared string must be
non-empty, so a counterparty, a market or a reason is present or the event is refused.

And one thing beyond validation: `BLOB_FIELDS` says which of a body's hashes name BYTES IN THE
STORE, so referential closure can be asked the same way by the writer before an append and by a
verifier over a log it did not write. Not every 64-hex field is a blob, and guessing would be
wrong in both directions - a checkpoint's `event_hash` is a POSITION in this log, an `instrument`
is an identity whose canonical JSON the booking verb registers (increment 3), and a `plan_hash` is
re-derivable by recompiling at the recorded LSN, which is the brief's auditor recompiling rather
than trusting stored bytes. What is declared here is what the event itself puts on the platter.

The closed set has four parts, and they are named apart because they are said by four different
mouths. `FACT_TYPES` is the TRADING vocabulary - what a desk, an administrator or a deployment
submits, and what a synthetic book is made of. `CUSTODY_TYPES` is what key custody says about
seats and wrapped class keys: submitted through the ordinary writer like any other fact, but facts
about the machinery rather than about the market. `PROVENANCE_TYPES` is what the record says about
WHICH NUMBERS - the three types that pin replay coordinates, said by the thing that produced them
rather than by the desk that books against them. `WRITER_TYPES` is the writer's own voice - one
type, `capability_denied`, which no submitter may speak: the public append refuses it by name and
only the writer's internal denial path emits it. `EVENT_TYPES` is the union and is what `validate`
consults, so the closure stays total while the four generations stay legible.

`EVENT_VERB` is the other half of every type's declaration: which of the six capability verbs an
append of it demands. It is a CLOSED map over the closed vocabulary - a type without a verb would
be a write nobody could be scoped for or refused for - and two entries are not verbs at all but
the two places authorization lives outside a policy document: `break_glass_used` is the recovery,
answerable to the genesis break-glass seat alone and to no declaration, so that a declaration which
stranded every admin cannot also brick the path out of itself and no stranger can walk through it
either, and `capability_denied` is the writer speaking, never gated, because a denial that could
itself be denied is a regress rather than a record.

`classify` is the SEAM the classified log is built on: an event's entitlement class is DERIVED from
its provenance, never assigned at the call site, so a reclassification is one declaration whose
inputs changed rather than ten thousand per-object ACLs.

INCREMENT 3 GROWS THE VOCABULARY BY THREE TYPES AND CHANGES NONE. That is what a versioned
governance act on a closed vocabulary looks like: `run_completed`, `result_pinned` and
`quote_filed` are new rows in `PROVENANCE_TYPES` - the part of the closed set named above for
saying WHICH NUMBERS, grown whole rather than mixed into the trading vocabulary, which is why
`FACT_TYPES` is byte-identical to what it was - new rows in `EVENT_VERB` and new rows in
`BLOB_FIELDS`, and every validator that existed before them validates exactly what it validated
before them. A v1 event stays readable under this vocabulary, which is the version-tolerance law
met from the writing side.

Their three verb scopes are three different authorities and the difference is the point.
`run_completed` takes `book`, beside `snapshot_registered`, for the same reason that one does: it
pins an object to the book that the book's own facts will cite, and the executor attesting to what
it just produced is the desk putting something on the record rather than governing the deployment.
`quote_filed` takes `book` too - the brief's `draft` and `validate` name work that happens BEFORE
the log, and a quote that has been GIVEN is past that line; it is an act with a counterparty on the
other end of it, which is exactly what booking scope is for, and it is also why the brief's "no
agent `book` scope in this workstream" already forbids an agent from filing one. `result_pinned`
takes `approve`, and it is the only one of the three that does: it gives standing to a tuple this
hub never witnessed, which is a second pair of eyes on somebody else's claim - the brief's
"outputs acquiring standing are gated a third time at declaration" - and the re-execution the verb
performs is precisely that second look, made mechanical. An attestation and a promotion answering
to one verb would say the two are the same authority, and they are not.
"""
import math

from .errors import MalformedEvent, UnknownEventType

HEX_DIGITS = frozenset('0123456789abcdef')


def is_hash(value):
    """A content address: 64 lowercase hex. Everything the log says about another object is one of
    these."""
    return isinstance(value, str) and len(value) == 64 and HEX_DIGITS.issuperset(value)


def is_text(value):
    """A non-empty string. A name that names nothing is not a name."""
    return isinstance(value, str) and value != ''


def is_number(value):
    """A finite JSON number, and `bool` is not one - `True` as a quantity is a bug wearing an
    integer's clothes."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return isinstance(value, float) and math.isfinite(value)


def is_integer(value):
    """A whole number - an LSN, a position. Floats are refused rather than rounded."""
    return isinstance(value, int) and not isinstance(value, bool)


def is_maybe_integer(value):
    """A whole number, or null where the job declared none.

    The one nullable kind, and it exists for exactly one field: a replay tuple's `seed`. A base
    valuation that names no `Random_Seed` is hashed with a null there, so the tuple the executor
    witnessed genuinely carries one - substituting a zero would record a tuple no result was ever
    filed under, which is the one thing an attestation may not do. Null IS the fact here (this job
    declared no seed) and it is tiny, which is the inlining boundary read the usual way.
    """
    return value is None or is_integer(value)


def is_coordinates(value):
    """An object of name -> finite number - a solved coordinate set.

    Also one field's kind and deliberately so: a quote's solved strike or barrier IS the fact and
    it is tiny, so it rides in the body rather than in the store. The values are closed to finite
    numbers because a coordinate is a number a desk dealt at; anything richer is a document and
    belongs in the store under its hash.
    """
    return (isinstance(value, dict)
            and all(is_text(name) and is_number(number) for name, number in value.items()))


#: The field vocabulary, spelled once so every refusal names the same thing the same way. Appended
#: to, never reordered: the constants below are indexes into this tuple, and a validator declared in
#: increment 1 must keep meaning exactly what it meant.
KINDS = (
    ('a content hash (64 lowercase hex)', is_hash),
    ('a non-empty string', is_text),
    ('a finite number', is_number),
    ('a whole number', is_integer),
    ('a whole number or null', is_maybe_integer),
    ('an object of name -> finite number', is_coordinates),
)
HASH, TEXT, NUMBER, INTEGER, MAYBE_INTEGER, COORDINATES = range(6)


def _validator(event_type, fields, open_body=False, optional=()):
    """The validator for one type: every declared field present and of its kind, and - unless the
    type is open - nothing else.

    `optional` is the third form, and it is narrower than it looks: a field that may be ABSENT but,
    when present, is checked exactly as a declared one. One type uses it - `quote_filed`, whose
    relayed client utterance is there when a salesperson relayed one and is not there when nobody
    said anything. It is not an extension point: an optional field is still named in this
    vocabulary, so the surplus rule is as closed as it ever was, and a type that declares none is
    validated by identical code producing identical wording.
    """
    declared = tuple(name for name, _ in fields)
    known = declared + tuple(name for name, _ in optional)

    def validate(body):
        if not isinstance(body, dict):
            raise MalformedEvent(
                '{}: the body is {}, not a JSON object - an event body is an object whose fields '
                'are {}'.format(event_type, type(body).__name__, ', '.join(known)))
        for name, kind in fields:
            if name not in body:
                raise MalformedEvent(
                    '{0}: the body has no {1} - a {0} declares {2}; supply {1} or file the fact '
                    'under a type that does not need it'.format(
                        event_type, name, ', '.join(declared)))
            description, check = KINDS[kind]
            if not check(body[name]):
                raise MalformedEvent(
                    '{}: {} is {!r}, which is not {} - correct the field at the submitter; the '
                    'record does not coerce'.format(event_type, name, body[name], description))
        for name, kind in optional:
            if name not in body:
                continue
            description, check = KINDS[kind]
            if not check(body[name]):
                raise MalformedEvent(
                    '{}: {} is {!r}, which is not {} - correct the field at the submitter; the '
                    'record does not coerce'.format(event_type, name, body[name], description))
        if not open_body:
            surplus = sorted(set(body) - set(known))
            if surplus:
                raise MalformedEvent(
                    '{}: the body carries {} beyond {} - the vocabulary is closed at the field '
                    'level too; drop the key, or add it to the type in a versioned amendment of '
                    'this vocabulary'.format(
                        event_type, ', '.join(surplus), ', '.join(known)))

    validate.__name__ = 'validate_' + event_type
    validate.__doc__ = 'Validate a {} body: {}.'.format(event_type, ', '.join(known))
    return validate


#: type -> validate(body). The closed vocabulary v1, in the order the brief names it: observations
#: first, then judgment, then governance.
FACT_TYPES = {
    # Trading facts. A fill carries a SIGNED quantity, never a position, and an execution reference
    # so that a retry is the same fact by construction and two identical clips are two facts.
    'fill': _validator('fill', (
        ('instrument', HASH), ('quantity', NUMBER), ('counterparty', TEXT),
        ('netting_set', TEXT), ('execution_reference', TEXT))),
    # Economics are never edited: an amendment is a NEW instrument hash linked to the old one.
    'amendment': _validator('amendment', (
        ('instrument', HASH), ('amended_to', HASH))),
    'election': _validator('election', (
        ('instrument', HASH), ('choice', TEXT))),
    # Market-level and keyed by (index, date, source): the administrator's print and a vendor snap
    # are different facts, and a republication is a new row under the same key, never an edit.
    'fixing_observed': _validator('fixing_observed', (
        ('index', TEXT), ('date', TEXT), ('source', TEXT), ('value', NUMBER))),
    # Judgment. A determination is a fact about a RULING - the touch was called, the dispute was
    # settled - attributed to the actor whose judgment the contract vests it in.
    'determination': _validator('determination', (
        ('subject', TEXT), ('ruling', TEXT))),
    'status_transition': _validator('status_transition', (
        ('subject', TEXT), ('status', TEXT))),
    # Governance. A market is a NAME resolved by a fold over these declarations; officialness is a
    # property of the name, never of the data.
    'market_declared': _validator('market_declared', (
        ('name', TEXT), ('values_hash', HASH))),
    'official_close_declared': _validator('official_close_declared', (
        ('market', TEXT), ('values_hash', HASH))),
    # An approval is a decision over a plan HASH: an amended plan is a new hash needing new
    # approval, by construction rather than by procedure.
    'approval': _validator('approval', (('plan_hash', HASH),)),
    'rejection': _validator('rejection', (('plan_hash', HASH), ('reason', TEXT))),
    'snapshot_registered': _validator('snapshot_registered', (('blob', HASH),)),
    # No blob class reduces or expires except through one of these.
    'retention_declared': _validator('retention_declared', (
        ('blob_class', TEXT), ('policy_blob', HASH))),
    'rehash_declared': _validator('rehash_declared', (('algorithm', TEXT),)),
    'break_glass_used': _validator('break_glass_used', (('reason', TEXT),)),
    # Open by design: a policy document's shape is the policy's own business, versioned by it.
    'policy_declared': _validator('policy_declared', (('policy', TEXT),), open_body=True),
    # The signature is the point; the append is ordinary.
    'checkpoint': _validator('checkpoint', (
        ('lsn', INTEGER), ('event_hash', HASH), ('signature', TEXT))),
}


#: What the record says about WHICH NUMBERS - the three types that pin replay coordinates, and a
#: fourth mouth beside the trading vocabulary, custody and the writer's own voice. They are named
#: apart for the reason the other three are: these are said by the thing that PRODUCED the numbers
#: (or, for a promotion, by the hub that re-produced them), which is a different speaker from the
#: desk that books a trade and a different question from what the market did. A synthetic book is
#: made of `FACT_TYPES`; a provenance chain is made of these.
PROVENANCE_TYPES = {
    # A run is recorded IFF its output will be cited by a fact - referential closure
    # read backwards - and these two are the only two ways a number acquires standing in the
    # record. `run_completed` is the standing lane's attestation AT BIRTH: the executor is the only
    # first-hand witness of what it produced, so it says so under its own name at the moment it
    # produced it, carrying the whole replay tuple and the three objects that make the claim
    # checkable - the job the plan recompiles from, the values vector it ran against, and the
    # result itself. `lane` is on the row because the rule turns on it and a record that had to be
    # told which lane it was reading would not be self-describing; the verb refuses anything but
    # the standing lane, since telemetry and curiosity mint nothing by construction.
    'run_completed': _validator('run_completed', (
        ('plan_hash', HASH), ('values_hash', HASH), ('engine_version', TEXT),
        ('seed', MAYBE_INTEGER), ('lane', TEXT), ('job', HASH), ('result', HASH))),
    # Promotion across lanes, AFTER the fact: a tuple the hub did not witness, attested only once
    # it re-executed and the bytes reproduced within a declared tolerance. `tolerance_policy` names
    # the hashed policy blob the comparison was held to, because the standard a claim was admitted
    # under is a fact about the pinning and the policy in force will move. What is NOT here is
    # whether the pin was a cache hit or a re-execution: that is a fold over the `run_completed`
    # rows at or before this LSN, and a computable consequence does not go in a body.
    'result_pinned': _validator('result_pinned', (
        ('plan_hash', HASH), ('values_hash', HASH), ('engine_version', TEXT),
        ('seed', MAYBE_INTEGER), ('job', HASH), ('result', HASH),
        ('tolerance_policy', HASH))),
    # The quoting act, and it pins TWO hashes because a quote is firm in two dimensions: the values
    # vector it was struck on and the book plan its marginal charge was solved against. The solved
    # coordinates and the edge ride in the body because each IS the fact and is tiny - what was
    # dealt, and what the desk took for it, under its own name rather than as arithmetic somebody
    # re-derives. `request` is the relayed client utterance and is OPTIONAL: it is there when a
    # salesperson relayed one. It is also the erasure case, and it needs no mechanism of its own -
    # every body in this log is sealed under its class key, so destroying that key erases what was
    # asked while the chain over it stays verifiable, which is what crypto-shredding is.
    'quote_filed': _validator('quote_filed', (
        ('quote_id', TEXT), ('structure', TEXT), ('plan_hash', HASH), ('values_hash', HASH),
        ('solved', COORDINATES), ('edge', NUMBER)),
        optional=(('request', TEXT),)),
}


#: The custody vocabulary - what per-seat keys and class-key wrapping put in the record. Both are
#: ordinary submitted facts; both speak of BYTES rather than carrying them, because a public key and
#: a wrap blob are objects the store holds once and the log names by hash forever.
CUSTODY_TYPES = {
    # A seat exists because its public key was published. The private half never appears here, and
    # the subject is the pseudonymous reference identity yields - never a name.
    'seat_enrolled': _validator('seat_enrolled', (
        ('subject', TEXT), ('algorithm', TEXT), ('public_key', HASH))),
    # The class key, wrapped to one seat. Rewrap on grant change ADDS one of these; revocation is
    # forward-only by the brief's declared residual, so nothing here ever unwraps.
    'key_wrapped': _validator('key_wrapped', (
        ('class', TEXT), ('subject', TEXT), ('wrap', HASH))),
}

#: The writer's own voice, and the whole of it. A denial is a decision and a decision is a fact, so
#: the refusal is appended rather than logged to a file nobody replays - but it is said BY the
#: writer ABOUT a submitter, so the public append refuses this type by name (log.py) and only the
#: internal denial path emits it, under the actor `writer`.
WRITER_TYPES = {
    'capability_denied': _validator('capability_denied', (
        ('subject', TEXT), ('verb', TEXT), ('book', TEXT), ('attempted_type', TEXT))),
}

#: The closure, total: every type this writer will validate. The three parts stay separately named
#: because they are said by three different mouths, and code that means "a trading fact" must not
#: have to mean "anything the writer can hold".
EVENT_TYPES = dict(FACT_TYPES)
EVENT_TYPES.update(CUSTODY_TYPES)
EVENT_TYPES.update(PROVENANCE_TYPES)
EVENT_TYPES.update(WRITER_TYPES)

#: What a submitter may name. `capability_denied` is absent on purpose: an unknown-type refusal
#: that offered it would be advertising a door that is bolted.
SUBMITTABLE = tuple(sorted(set(FACT_TYPES) | set(CUSTODY_TYPES) | set(PROVENANCE_TYPES)))

#: The six capability verbs a policy document grants, spelled once. `draft` and `validate` name work
#: that happens BEFORE the log - a quote drafted, a ticket validated - so no event type demands
#: them; they are here because the document that grants them is evaluated by the same function.
DRAFT, VALIDATE, BOOK, APPROVE, MARK, ADMIN = (
    'draft', 'validate', 'book', 'approve', 'mark', 'admin')
VERBS = (DRAFT, VALIDATE, BOOK, APPROVE, MARK, ADMIN)

#: The two authorizations that do NOT come from a policy document, named as pseudo-verbs so that
#: one evaluator answers every append rather than the writer growing a second, quieter one.
#: `RECOVERY` is the break-glass path - it must survive a document that stranded every admin, which
#: is the entire reason it is declared at genesis, and it is answerable to the seat genesis named
#: rather than to nobody, because a handle anyone may pull is a write channel and not a recovery.
#: `WRITER` is both the pseudo-verb of the writer's own denial and the actor it appends that denial
#: under: one name, because they are one thing.
RECOVERY = 'break_glass'
WRITER = 'writer'

#: The class every event carries in phase 1. One trading unit, one class; the mechanism ships
#: dormant, and `classify` is where it wakes up.
FIRM_CLASS = 'firm'

#: type -> the verb an append of it demands. Closed over the closed vocabulary: a type with no verb
#: here is a write nobody could be scoped for and nobody could be refused for.
EVENT_VERB = {
    # book: putting paper on the record, and everything that moves a position or its terms.
    'fill': BOOK,                       # a signed quantity lands: this IS booking
    'amendment': BOOK,                  # new economics under a new instrument hash - a booking act
    'election': BOOK,                   # exercising is the holder's act, exercised by the desk
    'status_transition': BOOK,          # an operational acknowledgment about a trade on the book
    'snapshot_registered': BOOK,        # a surface pinned to the book that will cite it
    'run_completed': BOOK,              # numbers pinned to the book whose facts will cite them
    'quote_filed': BOOK,                # a quote is an ACT, and filing paper on the record is this
    # approve: the second pair of eyes, and the only verb whose refusal is also a decision.
    'approval': APPROVE,                # a signature over a plan hash
    'rejection': APPROVE,               # refusing to sign is the same authority exercised
    'result_pinned': APPROVE,           # a tuple the hub did not witness, given standing
    # mark: what the market is said to be - the verb mismark monitoring watches.
    'market_declared': MARK,            # moving a NAME onto a values vector
    'official_close_declared': MARK,    # the close is exactly such a declaration
    'fixing_observed': MARK,            # an observation under its (index, date, source) key
    'determination': MARK,              # a ruling where the contract vests the call in an agent
    # admin: governance, custody and the deployment attesting to its own position.
    'policy_declared': ADMIN,           # every policy, capabilities included - see the strand rule
    'retention_declared': ADMIN,        # without which no blob class may ever reduce
    'rehash_declared': ADMIN,           # changing the algorithm is a governance act
    'checkpoint': ADMIN,                # signing the head is the deployment governing itself
    'seat_enrolled': ADMIN,             # who holds a seat is custody, and custody is governance
    'key_wrapped': ADMIN,               # and so is who the class key is wrapped to
    # Outside the six, and outside any document.
    'break_glass_used': RECOVERY,       # the genesis seat's, doc or no doc; anyone else's is denied
    'capability_denied': WRITER,        # the writer's own; never gated, or a denial could be denied
}


def classify(event_type, book):
    """The entitlement class of one event, DERIVED from where it came from.

    Phase 1 is one trading unit and one class, so this answers `firm` for everything and the
    classified-log mechanism ships dormant - which is the brief's declared posture rather than a
    stub with a note on it. What desk two activates is a rule this signature already has the inputs
    for: a FILL TAKES ITS BOOK'S CLASS, read from the book-class declaration in force, while policy,
    checkpoints and official market declarations keep the firm class everyone holds.

    The seam is here rather than at the call site because that is the difference between a
    reclassification and a migration: the class is one function's answer, so changing the function's
    inputs moves ten thousand objects with one declaration, and per-object ACLs never exist.
    """
    return FIRM_CLASS


#: type -> the fields whose value is an address in the blob store. `policy_declared` is listed for
#: its `blob` alone - the genesis verifying key and every later rotation ride that field - and the
#: rest of an open policy body is the policy's own business, as it is everywhere else here.
BLOB_FIELDS = {
    'snapshot_registered': ('blob',),
    'market_declared': ('values_hash',),
    'official_close_declared': ('values_hash',),
    'retention_declared': ('policy_blob',),
    'policy_declared': ('blob',),
    # Custody speaks of bytes like everyone else: a seat's public key and a wrapped class key are
    # objects the store holds once, so closure covers them and durability ordering binds them - no
    # enrollment appends before the key it publishes is fsynced.
    'seat_enrolled': ('public_key',),
    'key_wrapped': ('wrap',),
    # Attestation speaks of three objects and declares two-and-a-half of them. The `job` document
    # and the `result` are bulk and live in the store; `values_hash` is there because a values
    # vector is bulk too AND because the engine's own values hash IS the SHA-256 of that vector's
    # canonical bytes, so the citation and the address are one number rather than two that could
    # disagree. `plan_hash` is deliberately ABSENT for the reason this file's header gives: a plan
    # is RE-DERIVABLE, so the auditor recompiles it from the `job` blob at the recorded LSN and
    # requires the same hash rather than trusting bytes somebody stored under it.
    'run_completed': ('job', 'result', 'values_hash'),
    'result_pinned': ('job', 'result', 'values_hash', 'tolerance_policy'),
    # A quote cites the board it was struck on and nothing else it does not itself carry: the
    # solved coordinates and the edge are in the body, and the plan is re-derivable as always.
    'quote_filed': ('values_hash',),
}


def cited_blobs(event_type, body):
    """`(field, blob id)` for every store address `body` names.

    A field that is absent or is not an address yields nothing rather than raising: this answers
    "what does the log say lives in the store", and the shape of a body is the validator's
    question, asked before this one at the writer and never re-asked at a verifier - a v1 event
    must stay verifiable under a v(n+1) vocabulary.
    """
    if not isinstance(body, dict):
        return ()
    return tuple((name, body[name]) for name in BLOB_FIELDS.get(event_type, ())
                 if is_hash(body.get(name)))


def validate(event_type, body):
    """Refuse `event_type` if it is not a fact this log can hold, then refuse `body` if it is not
    that fact's shape.

    The type check comes first and is the harder line: an unknown type is not an extension point.
    """
    check = EVENT_TYPES.get(event_type)
    if check is None:
        raise UnknownEventType(
            'the vocabulary is closed and {!r} is not in it - the log records facts, never '
            'computable consequences, so a knock, an expiry or an accrual is a projection over '
            'what is already here; the types are {}'.format(
                event_type, ', '.join(SUBMITTABLE)))
    check(body)
