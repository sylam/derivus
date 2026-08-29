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

"""What the log is allowed to say - seventeen facts, and no eighteenth.

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


#: The field vocabulary, spelled once so every refusal names the same thing the same way.
KINDS = (
    ('a content hash (64 lowercase hex)', is_hash),
    ('a non-empty string', is_text),
    ('a finite number', is_number),
    ('a whole number', is_integer),
)
HASH, TEXT, NUMBER, INTEGER = range(4)


def _validator(event_type, fields, open_body=False):
    """The validator for one type: every declared field present and of its kind, and - unless the
    type is open - nothing else."""
    declared = tuple(name for name, _ in fields)

    def validate(body):
        if not isinstance(body, dict):
            raise MalformedEvent(
                '{}: the body is {}, not a JSON object - an event body is an object whose fields '
                'are {}'.format(event_type, type(body).__name__, ', '.join(declared)))
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
        if not open_body:
            surplus = sorted(set(body) - set(declared))
            if surplus:
                raise MalformedEvent(
                    '{}: the body carries {} beyond {} - the vocabulary is closed at the field '
                    'level too; drop the key, or add it to the type in a versioned amendment of '
                    'this vocabulary'.format(
                        event_type, ', '.join(surplus), ', '.join(declared)))

    validate.__name__ = 'validate_' + event_type
    validate.__doc__ = 'Validate a {} body: {}.'.format(event_type, ', '.join(declared))
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


#: type -> the fields whose value is an address in the blob store. `policy_declared` is listed for
#: its `blob` alone - the genesis verifying key and every later rotation ride that field - and the
#: rest of an open policy body is the policy's own business, as it is everywhere else here.
BLOB_FIELDS = {
    'snapshot_registered': ('blob',),
    'market_declared': ('values_hash',),
    'official_close_declared': ('values_hash',),
    'retention_declared': ('policy_blob',),
    'policy_declared': ('blob',),
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
    check = FACT_TYPES.get(event_type)
    if check is None:
        raise UnknownEventType(
            'the vocabulary is closed and {!r} is not in it - the log records facts, never '
            'computable consequences, so a knock, an expiry or an accrual is a projection over '
            'what is already here; the types are {}'.format(
                event_type, ', '.join(sorted(FACT_TYPES))))
    check(body)
