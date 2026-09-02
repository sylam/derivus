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

The vocabulary is CLOSED, which is the facts-only law with teeth: `knocked_out` is not a type the
writer declines to validate, it is a type that does not exist. Knocks, expiries, accruals, positions
and lifecycle state are folds over what is here - observations of the world and acts of judgment or
governance.

Three validation rules. BULK NEVER INLINES: a field naming a surface, a plan, a values vector or a
policy document is 64 lowercase hex and the object lives once in the blob store. Surplus keys are
refused everywhere except `policy_declared`, whose shape is versioned by the policy itself. Every
declared string must be non-empty.

`BLOB_FIELDS` declares which of a body's hashes name bytes in the store, so referential closure is
asked the same way by the writer and by a verifier over a log it did not write - not every 64-hex
field is a blob. The closed set is in four parts because four different mouths speak it
(`FACT_TYPES`, `CUSTODY_TYPES`, `PROVENANCE_TYPES`, `WRITER_TYPES`), `EVENT_TYPES` is the union
`validate` consults, and `EVENT_VERB` names the capability verb each type demands - with
`break_glass_used` answering to the genesis seat alone and `capability_denied` never gated.
`classify` derives the entitlement class from provenance, so a reclassification is one declaration.
"""
import math

from .errors import MalformedEvent, UnknownEventType

HEX_DIGITS = frozenset('0123456789abcdef')


def is_hash(value):
    """Whether `value` is a content address: 64 lowercase hex."""
    return isinstance(value, str) and len(value) == 64 and HEX_DIGITS.issuperset(value)


def is_text(value):
    """Whether `value` is a non-empty string - a name that names nothing is not a name."""
    return isinstance(value, str) and value != ''


def is_number(value):
    """Whether `value` is a finite JSON number. `bool` is not one, though it is an `int`."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return isinstance(value, float) and math.isfinite(value)


def is_integer(value):
    """Whether `value` is a whole number - an LSN, a position. Floats are refused, not rounded."""
    return isinstance(value, int) and not isinstance(value, bool)


def is_maybe_integer(value):
    """Whether `value` is a whole number or null.

    The one nullable kind, for the replay tuple's `seed`: a job naming no `Random_Seed` is hashed
    with a null, so substituting a zero would record a tuple no result was filed under.
    """
    return value is None or is_integer(value)


def is_coordinates(value):
    """Whether `value` is an object of name -> finite number - a solved coordinate set.

    Closed to finite numbers: a coordinate is a number a desk dealt at, and anything richer is a
    document belonging in the store under its hash.
    """
    return (isinstance(value, dict)
            and all(is_text(name) and is_number(number) for name, number in value.items()))


#: `(description, predicate)` per field kind, spelled once so every refusal words it the same way.
#: Appended to, never reordered - the constants below are indexes into this tuple.
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
    """Build the validator for one event type: every field in `fields` present and of its kind,
    and - unless `open_body` - nothing beyond `fields` and `optional`.

    An `optional` field may be absent but is checked exactly as a declared one when present. It is
    still named in this vocabulary, so the surplus rule stays closed.
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


#: type -> validate(body). The trading vocabulary: observations, then judgment, then governance.
FACT_TYPES = {
    # A fill carries a SIGNED quantity, never a position, and an execution reference so a retry is
    # the same fact by construction while two identical clips are two facts.
    'fill': _validator('fill', (
        ('instrument', HASH), ('quantity', NUMBER), ('counterparty', TEXT),
        ('netting_set', TEXT), ('execution_reference', TEXT))),
    # Economics are never edited: an amendment is a new instrument hash linked to the old one.
    'amendment': _validator('amendment', (
        ('instrument', HASH), ('amended_to', HASH))),
    'election': _validator('election', (
        ('instrument', HASH), ('choice', TEXT))),
    # Keyed by (index, date, source), so an administrator's print and a vendor snap are different
    # facts and a republication is a new row under the same key rather than an edit.
    'fixing_observed': _validator('fixing_observed', (
        ('index', TEXT), ('date', TEXT), ('source', TEXT), ('value', NUMBER))),
    # A determination is a fact about a ruling, attributed to the actor the contract vests it in.
    'determination': _validator('determination', (
        ('subject', TEXT), ('ruling', TEXT))),
    'status_transition': _validator('status_transition', (
        ('subject', TEXT), ('status', TEXT))),
    # A market is a name resolved by a fold over these; officialness is a property of the name.
    'market_declared': _validator('market_declared', (
        ('name', TEXT), ('values_hash', HASH))),
    'official_close_declared': _validator('official_close_declared', (
        ('market', TEXT), ('values_hash', HASH))),
    # A decision over a plan HASH, so an amended plan is a new hash needing new approval.
    'approval': _validator('approval', (('plan_hash', HASH),)),
    'rejection': _validator('rejection', (('plan_hash', HASH), ('reason', TEXT))),
    'snapshot_registered': _validator('snapshot_registered', (('blob', HASH),)),
    # No blob class reduces or expires except through one of these.
    'retention_declared': _validator('retention_declared', (
        ('blob_class', TEXT), ('policy_blob', HASH))),
    'rehash_declared': _validator('rehash_declared', (('algorithm', TEXT),)),
    'break_glass_used': _validator('break_glass_used', (('reason', TEXT),)),
    # Open-bodied: a policy document's shape is versioned by the policy itself.
    'policy_declared': _validator('policy_declared', (('policy', TEXT),), open_body=True),
    'checkpoint': _validator('checkpoint', (
        ('lsn', INTEGER), ('event_hash', HASH), ('signature', TEXT))),
}


#: What the record says about WHICH NUMBERS - the types that pin replay coordinates, said by the
#: thing that produced them rather than by the desk that books against them. A synthetic book is
#: made of `FACT_TYPES`; a provenance chain is made of these.
PROVENANCE_TYPES = {
    # The standing lane's attestation at birth: the replay tuple plus the three objects that make
    # it checkable - the job the plan recompiles from, the values vector, and the result. `lane` is
    # on the row so the record is self-describing, and the verb refuses any lane but standing.
    'run_completed': _validator('run_completed', (
        ('plan_hash', HASH), ('values_hash', HASH), ('engine_version', TEXT),
        ('seed', MAYBE_INTEGER), ('lane', TEXT), ('job', HASH), ('result', HASH))),
    # Promotion across lanes after the fact: a tuple the hub did not witness, attested only once it
    # re-executed and the bytes reproduced within `tolerance_policy`, which names the hashed policy
    # the comparison was held to. Whether the pin was a cache hit is a fold, not a body field.
    'result_pinned': _validator('result_pinned', (
        ('plan_hash', HASH), ('values_hash', HASH), ('engine_version', TEXT),
        ('seed', MAYBE_INTEGER), ('job', HASH), ('result', HASH),
        ('tolerance_policy', HASH))),
    # The quoting act. It pins two hashes because a quote is firm in two dimensions: the values
    # vector it was struck on and the book plan its marginal charge was solved against. `request` is
    # the relayed client utterance, optional and erased by shredding the class key.
    'quote_filed': _validator('quote_filed', (
        ('quote_id', TEXT), ('structure', TEXT), ('plan_hash', HASH), ('values_hash', HASH),
        ('solved', COORDINATES), ('edge', NUMBER)),
        optional=(('request', TEXT),)),
}


#: The custody vocabulary - what per-seat keys and class-key wrapping put in the record. Ordinary
#: submitted facts, both speaking of bytes by hash rather than carrying them.
CUSTODY_TYPES = {
    # A seat exists because its public key was published. The private half never appears here, and
    # the subject is the pseudonymous reference identity yields, never a name.
    'seat_enrolled': _validator('seat_enrolled', (
        ('subject', TEXT), ('algorithm', TEXT), ('public_key', HASH))),
    # The class key wrapped to one seat. Rewrap on grant change adds one of these; revocation is
    # forward-only, so nothing here ever unwraps.
    'key_wrapped': _validator('key_wrapped', (
        ('class', TEXT), ('subject', TEXT), ('wrap', HASH))),
}

#: The writer's own voice, and the whole of it: a denial is a decision and so is appended rather
#: than logged. Said by the writer about a submitter, so the public append refuses this type by name
#: (log.py) and only the internal denial path emits it, under the actor `writer`.
WRITER_TYPES = {
    'capability_denied': _validator('capability_denied', (
        ('subject', TEXT), ('verb', TEXT), ('book', TEXT), ('attempted_type', TEXT))),
}

#: The closure, total: every type this writer will validate. The parts stay separately named so
#: code meaning "a trading fact" need not mean "anything the writer can hold".
EVENT_TYPES = dict(FACT_TYPES)
EVENT_TYPES.update(CUSTODY_TYPES)
EVENT_TYPES.update(PROVENANCE_TYPES)
EVENT_TYPES.update(WRITER_TYPES)

#: What a submitter may name, and what an unknown-type refusal lists. `capability_denied` is
#: excluded: it would advertise a door that is bolted.
SUBMITTABLE = tuple(sorted(set(FACT_TYPES) | set(CUSTODY_TYPES) | set(PROVENANCE_TYPES)))

#: The six capability verbs a policy document grants, spelled once. `draft` and `validate` name work
#: that happens before the log, so no event type demands them; they are here because the document
#: granting them is evaluated by the same function.
DRAFT, VALIDATE, BOOK, APPROVE, MARK, ADMIN = (
    'draft', 'validate', 'book', 'approve', 'mark', 'admin')
VERBS = (DRAFT, VALIDATE, BOOK, APPROVE, MARK, ADMIN)

#: The two authorizations that come from no policy document, named as pseudo-verbs so one evaluator
#: answers every append. `RECOVERY` is the break-glass path, declared at genesis and answerable to
#: the seat genesis named; `WRITER` is both the writer's denial pseudo-verb and the actor it uses.
RECOVERY = 'break_glass'
WRITER = 'writer'

#: The class every event carries in phase 1. One trading unit, one class; the mechanism ships
#: dormant, and `classify` is where it wakes up.
FIRM_CLASS = 'firm'

#: type -> the verb an append of it demands. Closed over the closed vocabulary: a type with no verb
#: here would be a write nobody could be scoped for or refused for.
EVENT_VERB = {
    # book: putting paper on the record, and everything that moves a position or its terms.
    'fill': BOOK,
    'amendment': BOOK,
    'election': BOOK,
    'status_transition': BOOK,
    'snapshot_registered': BOOK,
    'run_completed': BOOK,
    'quote_filed': BOOK,
    # approve: the second pair of eyes, and the only verb whose refusal is also a decision.
    'approval': APPROVE,
    'rejection': APPROVE,
    'result_pinned': APPROVE,           # a tuple the hub did not witness, given standing
    # mark: what the market is said to be - the verb mismark monitoring watches.
    'market_declared': MARK,
    'official_close_declared': MARK,
    'fixing_observed': MARK,
    'determination': MARK,              # a ruling where the contract vests the call in an agent
    # admin: governance, custody and the deployment attesting to its own position.
    'policy_declared': ADMIN,           # every policy, capabilities included - see the strand rule
    'retention_declared': ADMIN,
    'rehash_declared': ADMIN,
    'checkpoint': ADMIN,
    'seat_enrolled': ADMIN,
    'key_wrapped': ADMIN,
    # Outside the six, and outside any document.
    'break_glass_used': RECOVERY,       # the genesis seat's, doc or no doc; anyone else's is denied
    'capability_denied': WRITER,        # never gated, or a denial could itself be denied
}


def classify(event_type, book):
    """The entitlement class of one event, derived from its provenance.

    Declared limitation: phase 1 is one trading unit and one class, so this answers `FIRM_CLASS` for
    every event and the classified-log mechanism ships dormant. The seam lives here rather than at
    the call site, so a second desk is a reclassification through this function's inputs rather than
    a per-object ACL migration.
    """
    return FIRM_CLASS


#: type -> the fields whose value is an address in the blob store. `policy_declared` is listed for
#: its `blob` alone; the rest of an open policy body is the policy's own business.
BLOB_FIELDS = {
    'snapshot_registered': ('blob',),
    'market_declared': ('values_hash',),
    'official_close_declared': ('values_hash',),
    'retention_declared': ('policy_blob',),
    'policy_declared': ('blob',),
    # Durability ordering binds these like any other citation: no enrollment appends before the key
    # it publishes is fsynced.
    'seat_enrolled': ('public_key',),
    'key_wrapped': ('wrap',),
    # `job` and `result` are bulk. `values_hash` is here because the engine's values hash IS the
    # SHA-256 of that vector's canonical bytes, so citation and address are one number. `plan_hash`
    # is absent because a plan is re-derivable by recompiling `job` at the recorded LSN.
    'run_completed': ('job', 'result', 'values_hash'),
    'result_pinned': ('job', 'result', 'values_hash', 'tolerance_policy'),
    # A quote cites the board it was struck on; the solved coordinates and the edge ride in the body
    # and the plan is re-derivable.
    'quote_filed': ('values_hash',),
}


def cited_blobs(event_type, body):
    """`(field, blob id)` for every store address `body` names.

    A field that is absent or is not an address yields nothing rather than raising: body shape is
    the validator's question, and a v1 event must stay verifiable under a later vocabulary.
    """
    if not isinstance(body, dict):
        return ()
    return tuple((name, body[name]) for name in BLOB_FIELDS.get(event_type, ())
                 if is_hash(body.get(name)))


def validate(event_type, body):
    """Raise `UnknownEventType` if `event_type` is outside the closed vocabulary, then
    `MalformedEvent` if `body` is not that type's shape. The type check comes first."""
    check = EVENT_TYPES.get(event_type)
    if check is None:
        raise UnknownEventType(
            'the vocabulary is closed and {!r} is not in it - the log records facts, never '
            'computable consequences, so a knock, an expiry or an accrual is a projection over '
            'what is already here; the types are {}'.format(
                event_type, ', '.join(SUBMITTABLE)))
    check(body)
