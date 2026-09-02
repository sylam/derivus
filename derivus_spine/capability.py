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

"""Who may say what - a fold over declarations, and one pure function that answers the question.

The document is a hashed blob declared through the ordinary writer and is a complete replacement
rather than a patch, so the state at a position is the last declaration at or before it:

    {"grants": [{"subject": s, "verb": v, "book": b}], "read": [{"subject": s, "class": c}]}

Nothing here holds state that outlives the call. Three authorizations live outside the document -
genesis grants (read once and not surviving a document, so a declaration can strand the last admin,
deliberately), break-glass (the genesis seat alone, anyone else refused by name and the reach
recorded as a `capability_denied`), and the writer's own denial verb, never gated. A capabilities
blob doctored under its own address, or gone from the store, folds to `UNREADABLE` - in force and
granting nothing - rather than raising, which inside the writer's hook would take every append with
it. Enforcement activates BY DECLARATION: with no document in the log, `evaluate` is not consulted.
"""
import json

from .canon import canonical_bytes
from .errors import CapabilityDenied, CollisionRefusal, MissingBlobRefusal
from .vocabulary import ADMIN, EVENT_VERB, FIRM_CLASS, RECOVERY, VERBS, WRITER

#: The policy names this fold reads. The first two are genesis's own, respelled here rather than
#: imported: `genesis` imports the writer, and the writer imports this module.
GENESIS_POLICY = 'genesis'
BREAK_GLASS_POLICY = 'break_glass'
CAPABILITIES_POLICY = 'capabilities'

#: The event types that can move the capability state. Selected by envelope alone - no body is
#: decrypted to find them - so a replica holding no key still knows where the fold's inputs are.
CAPABILITY_EVENTS = ('policy_declared', 'break_glass_used')

#: A grant's book wildcard. `*` matches every book AND the firm-level facts that carry no book at
#: all, which is the only scope that reaches policy, checkpoints and official market declarations.
ANY_BOOK = '*'

#: The document's two sections and the fields of each row, closed exactly as an event body is.
GRANT_FIELDS = ('book', 'subject', 'verb')
READ_FIELDS = ('class', 'subject')
DOCUMENT_SECTIONS = ('grants', 'read')


class _Unreadable(object):
    """A capabilities document that is in force and cannot be read - the fail-closed document.

    A value rather than an exception: the fold runs inside the writer's authorization hook, so
    raising would take every append with it, `break_glass_used` and a replacement declaration
    included. `evaluate` refuses all six verbs against this, while break-glass and the admin it
    recovers still answer.
    """

    __slots__ = ()

    def __repr__(self):
        return 'UNREADABLE'


#: The one instance, compared by identity: a document is `None`, a dict, or this.
UNREADABLE = _Unreadable()


def verb_for(event_type):
    """The verb an append of `event_type` demands.

    Fails closed: a type absent from `EVENT_VERB` demands `admin`, so a forgotten entry costs a
    write that needs governance rather than one nobody is scoped for.
    """
    return EVENT_VERB.get(event_type, ADMIN)


def parse_document(raw, where):
    """The capabilities document in `raw`, parsed, shape-checked and required to be canonical.

    `where` names the document in refusals, which are `CapabilityDenied`. The canonical requirement
    keeps one policy to one blob, so the same decision spelled two ways cannot become two histories.
    """
    try:
        document = json.loads(bytes(raw).decode('utf-8'))
    except (TypeError, UnicodeDecodeError, ValueError):
        raise CapabilityDenied(
            '{}: the capabilities blob is not JSON - a document the evaluator cannot read '
            'authorizes nothing; declare a well-formed replacement through `DV_Spine grant '
            '--file`, or recover admin through break-glass if this one is already in force'.format(
                where))
    _shape(document, where)
    if canonical_bytes(document) != bytes(raw):
        raise CapabilityDenied(
            '{}: the capabilities blob is not the canonical spelling of what it says - one policy '
            'must be one blob or the record holds two histories of one decision; store it as '
            '`canonical_bytes(document)`, which is what `DV_Spine grant --file` does'.format(where))
    return document


def _shape(document, where):
    """Check every section, row and field of `document`, raising `CapabilityDenied` naming the
    first that is wrong.

    Closed at the field level, like an event body: a key no evaluator reads would be a grant nobody
    enforces.
    """
    def refuse(sentence):
        raise CapabilityDenied('{}: {}'.format(where, sentence))

    if not isinstance(document, dict):
        refuse('the capabilities document is {}, not a JSON object - it is {{"grants": [...], '
               '"read": [...]}}'.format(type(document).__name__))
    surplus = sorted(set(document) - set(DOCUMENT_SECTIONS))
    if surplus:
        refuse('the capabilities document carries {} beyond grants, read - the document is closed '
               'at the field level; drop the key or version the document shape'.format(
                   ', '.join(surplus)))
    for section, fields in (('grants', GRANT_FIELDS), ('read', READ_FIELDS)):
        if section not in document:
            refuse('the capabilities document has no {} - a document that grants nothing says so '
                   'with an empty list, so that silence is never mistaken for absence'.format(
                       section))
        rows = document[section]
        if not isinstance(rows, list):
            refuse('{} is {}, not a list of rows'.format(section, type(rows).__name__))
        for position, row in enumerate(rows):
            if not isinstance(row, dict) or sorted(row) != list(fields):
                refuse('{}[{}] is not a {{{}}} row - every row carries exactly those fields'.format(
                    section, position, ', '.join(fields)))
            for name in fields:
                if not isinstance(row[name], str) or row[name] == '':
                    refuse('{}[{}].{} is {!r}, and a name that names nothing is not a name'.format(
                        section, position, name, row[name]))
            if section == 'grants' and row['verb'] not in VERBS:
                refuse('grants[{}].verb is {!r}, which is not one of the six scopes ({}) - the '
                       'verbs are closed, so a document cannot invent authority'.format(
                           position, row['verb'], ', '.join(VERBS)))


def canonical_document(document, where='this capabilities document'):
    """`document` checked and canonicalised - the bytes a declaration puts in the store.

    One function, so what the writer accepts and what the `grant` verb writes cannot part company.
    """
    _shape(document, where)
    return canonical_bytes(document)


def initial_state():
    """The state before any declaration: no document, no genesis grants read, nothing recovered."""
    return {'doc': None, 'genesis': {'admin': (), 'break_glass': None, 'recovered': ()}}


def apply_event(state, event_type, actor, body, store):
    """Fold one event into `state` and return it - the single step `build_state` repeats.

    Public because the writer folds incrementally, applying each capability-bearing append as it
    lands rather than re-reading the history. Sharing the step keeps the cached answer and the
    replayed answer identical.
    """
    genesis = state['genesis']
    if event_type == 'break_glass_used':
        # Credited only to the seat genesis named. The writer refuses anybody else's, but this fold
        # also reads homes the writer did not write - a replica, a restored copy.
        if actor is not None and actor == genesis['break_glass'] \
                and actor not in genesis['recovered']:
            genesis['recovered'] = genesis['recovered'] + (actor,)
        return state
    if event_type != 'policy_declared' or not isinstance(body, dict):
        return state

    policy = body.get('policy')
    if policy == GENESIS_POLICY:
        # Read ONCE - the mint's grant, not the latest thing calling itself genesis.
        if not genesis['admin']:
            grants = body.get('grants')
            if isinstance(grants, list):
                genesis['admin'] = tuple(
                    row['subject'] for row in grants
                    if isinstance(row, dict) and row.get('scope') == ADMIN
                    and isinstance(row.get('subject'), str))
    elif policy == BREAK_GLASS_POLICY:
        if genesis['break_glass'] is None:
            grant = body.get('grant')
            if isinstance(grant, dict) and isinstance(grant.get('subject'), str):
                genesis['break_glass'] = grant['subject']
    elif policy == CAPABILITIES_POLICY:
        # A complete replacement, and where a recovered admin stops being one: carrying that grant
        # past the next declaration would be the one admin no policy could revoke.
        state['doc'] = _declared_document(body.get('blob'), store)
        if state['doc'] is not UNREADABLE:
            genesis['recovered'] = ()
    return state


def _declared_document(blob, store):
    """The document `blob` names, or `UNREADABLE` where the bytes no longer answer for it.

    The writer checked this blob when it was declared, so reaching here with something unparseable
    means the platter changed underneath a landed document. Raising would leave a home nobody could
    rescue, so it folds to a value instead.
    """
    try:
        return parse_document(store.get(blob), 'the capabilities document {}'.format(blob))
    except (CapabilityDenied, CollisionRefusal, MissingBlobRefusal):
        return UNREADABLE


def build_state(log, lsn=None):
    """The whole capability state folded out of `log` at or before `lsn` (default: the head).

    Read off the platter over `log.frames` rather than any index, which is a correctness property:
    reading never claims the home, so a handle routinely outlives someone else's append and an index
    built when it opened would answer today's question out of yesterday's policy. Frames are
    filtered by envelope type, which costs no key.
    """
    state = initial_state()
    for frame in log.frames(end_lsn=lsn):
        if frame['event_type'] not in CAPABILITY_EVENTS:
            continue
        apply_event(state, frame['event_type'], frame['actor'], log.open_body(frame), log.store)
    return state


def state_at(log, lsn=None):
    """`(document, genesis)` as of `lsn` - the fold every authorization question starts at.

    The document is `None` where none has been declared, the parsed object where one has, and
    `UNREADABLE` where one is in force and its blob no longer answers for it. A declaration applies
    to the appends after it, so passing `lsn` answers as of that position.
    """
    state = build_state(log, lsn)
    return (state['doc'], state['genesis'])


def evaluate(doc, genesis, subject, verb, book):
    """Whether `subject` may exercise `verb` over `book`. The one authorization function.

    Pure: no log, store, clock or home, so the same inputs answer the same way on the hub, on a
    replica and in a gate; `doc` and `genesis` come from `state_at`. Precedence, highest first: the
    writer's own verb, the genesis recovery seat, a recovered admin (until the next declaration
    lands, the only grant not held in a document), then the document - or, with no document in the
    log, yes. An unreadable document grants nothing, and the first two rules are what rescue the
    home.

    Book matching: `*` matches every book and the firm-level facts that carry no book at all; a
    named grant matches only its own book and never a book-less event.
    """
    seats = genesis or {}
    if verb == WRITER:
        return True
    if verb == RECOVERY:
        seat = seats.get('break_glass')
        return seat is None or subject == seat
    if doc is None:
        return True
    if verb == ADMIN and subject in seats.get('recovered', ()):
        return True
    if doc is UNREADABLE:
        return False
    for grant in doc.get('grants', ()):
        if grant.get('subject') != subject or grant.get('verb') != verb:
            continue
        if grant.get('book') == ANY_BOOK:
            return True
        if book is not None and grant.get('book') == book:
            return True
    return False


def read_subjects(doc, entitlement_class=FIRM_CLASS):
    """Every subject the document admits to read `entitlement_class`, in declaration order.

    Enumeration rather than evaluation: the question custody asks when deciding who a class key is
    wrapped to. A document that is `None` or `UNREADABLE` admits nobody.
    """
    if doc is None or doc is UNREADABLE:
        return ()
    seen = []
    for row in doc.get('read', ()):
        subject = row.get('subject')
        if row.get('class') == entitlement_class and subject not in seen:
            seen.append(subject)
    return tuple(seen)


def denial_body(subject, verb, book, attempted_type):
    """The body of the `capability_denied` fact the writer appends when it refuses.

    `book` is recorded as the scope the append needed, not the book it carried - a firm-level event
    needs `*` - so the fact names the grant that would have let it through.
    """
    return {'subject': subject, 'verb': verb,
            'book': book if book is not None else ANY_BOOK,
            'attempted_type': attempted_type}
