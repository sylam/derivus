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

"""Who may say what - a fold over declarations, and ONE pure function that answers the question.

Authorization here is not a table somebody edits. It is a DOCUMENT, hashed into the blob store and
declared through the ordinary writer, and every question about it is a fold over the log at an LSN.
That is what makes "could X book on that desk in March" answerable at all: the answer is replayed
rather than remembered, exactly as positions and lifecycle state are, and nothing in this module
holds state that outlives the call.

The document is a complete REPLACEMENT, never a patch:

    {"grants": [{"subject": s, "verb": v, "book": b}], "read": [{"subject": s, "class": c}]}

so a reclassification is one declaration and history answers as-of by LSN. Patches would make the
state a function of the order somebody applied them in and of nothing you can hold in your hand; a
replacement makes the state a function of the last declaration at or before the position asked
about, which is a fold with one line in it.

Three things live OUTSIDE the document, and each is outside it for a reason the brief names.

Genesis grants are read once and never expire FOR BREAK-GLASS PURPOSES, and they do not survive a
document: once a capabilities declaration is in force it is the whole truth about the six verbs, so
a declaration CAN strand the last admin. That is deliberate. A design where the mint's grant
silently outranks every later policy is a design where the policy is advisory, and an admin who
cannot be revoked is a finding.

Break-glass is therefore the recovery, and it is declared at genesis rather than invented after the
accident. The genesis break-glass seat's use is authorized whatever a document says - a recovery
path a stranding declaration could also brick is not a recovery path - and the fold credits it with
an admin grant on top of whatever document is in force. Anybody ELSE reaching for the handle is
refused by the ordinary path and the reach is RECORDED, as a `capability_denied` naming the
`break_glass` verb: a better fact than an authority-free use nobody was ever scoped for, and the
difference between a gated write path and an open one, which is what this increment is for. The
attempt is in the record either way, which was always the point.

A document the fold cannot READ is the other half of that rule. A capabilities blob doctored under
its own address, or gone from the store, is not an exception the writer dies of: a home that cannot
answer "who may write" would be a home nobody could rescue, which is exactly the state break-glass
exists to leave. It folds to `UNREADABLE` - a document in force that grants nothing - so every verb
is refused BY NAME, while the genesis break-glass seat still appends its use and the admin that use
recovers still declares the replacement that ends it. Fail-closed and recoverable, rather than
fail-closed and bricked.

A RECOVERED admin does not outlive the document that follows it. The grant break-glass mints is the
one that lives outside every document, so leaving it there forever would mint the single admin no
declaration could revoke - the finding the paragraph above names. The next capabilities declaration
to LAND clears it: authorization is evaluated before the event is folded, so the recovered seat's own
restoring document still lands, and admin is governable again the moment a document is back in force.

And the writer's own denial is never gated, because a refusal that could itself be refused is a
regress rather than a record.

Enforcement activates BY DECLARATION. With no capabilities document in the log, `evaluate` is not
consulted at all and the home runs as the single-user instrument it is - which is not a special
case bolted on for compatibility but the honest reading of a log that has never been told who may
do what. It is also why every increment-1 home and every increment-1 gate stays green.
"""
import json

from .canon import canonical_bytes
from .errors import CapabilityDenied, CollisionRefusal, MissingBlobRefusal
from .vocabulary import ADMIN, EVENT_VERB, FIRM_CLASS, RECOVERY, VERBS, WRITER

#: The policy names this fold reads. The first two are genesis's own, spelled here rather than
#: imported from `genesis` because that module imports the writer and the writer imports this one -
#: a gate in tests/test_spine_capability.py pins the two spellings together so they cannot drift.
GENESIS_POLICY = 'genesis'
BREAK_GLASS_POLICY = 'break_glass'
CAPABILITIES_POLICY = 'capabilities'

#: The event types that can move the capability state. The fold selects them by ENVELOPE alone - no
#: body is decrypted to find them - so a replica holding no key still knows where the fold's inputs
#: are, and only a holder of the class key can read what they say.
CAPABILITY_EVENTS = ('policy_declared', 'break_glass_used')

#: A grant's book wildcard. `*` matches every book AND the firm-level facts that carry no book at
#: all, which is the only scope that reaches policy, checkpoints and official market declarations.
ANY_BOOK = '*'

#: The document's two sections and the fields of each row, closed exactly as an event body is.
GRANT_FIELDS = ('book', 'subject', 'verb')
READ_FIELDS = ('class', 'subject')
DOCUMENT_SECTIONS = ('grants', 'read')


class _Unreadable(object):
    """A capabilities document that is IN FORCE and cannot be read - the fail-closed document.

    It is a value rather than an exception because of where the alternative leaves a home. The fold
    runs inside the writer's own authorization hook, so a declared blob that has been doctored under
    its own address, or lost out of the store, would raise there and take every append with it -
    `break_glass_used` and a replacement declaration included. That is precisely the bricked state
    the brief says break-glass exists to recover from, reached BY the recovery path being unreachable.

    So it folds to this instead: `evaluate` refuses every one of the six verbs against it by name,
    while the two authorizations that live outside every document still answer - the genesis
    break-glass seat's use, and the admin that use recovers, whose next declaration replaces this
    with a document that reads.
    """

    __slots__ = ()

    def __repr__(self):
        return 'UNREADABLE'


#: The one instance, compared by identity: a document is `None`, a dict, or this.
UNREADABLE = _Unreadable()


def verb_for(event_type):
    """The verb an append of `event_type` demands.

    Fails CLOSED: a type this map has not been taught demands `admin`, so the cost of forgetting an
    entry is a write that needs governance rather than a write nobody is scoped for.
    """
    return EVENT_VERB.get(event_type, ADMIN)


def parse_document(raw, where):
    """A capabilities document, read out of its blob bytes and checked to the last key.

    Strict, and canonical-or-nothing. A document the evaluator cannot read authorizes NOTHING, so
    the refusal is the same one an unscoped actor meets rather than a quieter kind - and it is
    raised where the document is declared, so a malformed policy cannot land and be discovered at
    the next append by a home that can no longer write.

    The canonical check is the other half: the same policy spelled two ways would be two blobs and
    two histories of one decision, and the whole spine addresses meaning by canonical bytes. The
    `grant` verb canonicalises the operator's file on the way in, so this is a law a human meets at
    the moment they can still fix it.
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
    """Every section, row and field of a document, named on the way out.

    Closed at the field level like an event body: a key no evaluator reads is a grant nobody
    enforces, and discovering that in year three is how an entitlement becomes a rumour.
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

    One function so that what the writer accepts and what the `grant` verb writes cannot part
    company: the editor and the enforcer read the same law out of the same place.
    """
    _shape(document, where)
    return canonical_bytes(document)


def initial_state():
    """The state before any declaration: no document, no genesis grants read, nothing recovered."""
    return {'doc': None, 'genesis': {'admin': (), 'break_glass': None, 'recovered': ()}}


def apply_event(state, event_type, actor, body, store):
    """Fold one event into `state` and answer it - the single step `build_state` repeats.

    Exposed because the writer folds INCREMENTALLY: it built this state when it opened the home and
    applies each capability-bearing append as it lands, rather than re-reading the history on every
    write. The step is the same one either way, which is what keeps the cached answer and the
    replayed answer the same answer.
    """
    genesis = state['genesis']
    if event_type == 'break_glass_used':
        # Credited only to the seat genesis named. The writer refuses anybody else's, so this is
        # belt and braces on purpose: the fold reads homes the writer did not write - a replica, a
        # restored copy, a log whose genesis named no break-glass seat at all - and a recovery
        # credited to whoever appended it is the whole authorization model handed to the platter.
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
        # A complete replacement, and the point at which a recovered admin stops being one: the
        # grant break-glass minted lives outside every document, so carrying it past the next
        # declaration would be the one admin no policy could revoke.
        state['doc'] = _declared_document(body.get('blob'), store)
        if state['doc'] is not UNREADABLE:
            genesis['recovered'] = ()
    return state


def _declared_document(blob, store):
    """The document a declaration names, or `UNREADABLE` where the bytes will not answer for it.

    The writer checked this blob at the moment it was declared - `parse_document` runs in the
    authorization hook - so reaching here with something that will not parse means the platter
    changed underneath a document that has already landed: doctored under its own address, or gone.
    Neither is a question the fold can answer, and neither may stop it: this fold is what tells the
    writer who may write, so raising here would leave a home that cannot even be rescued.
    """
    try:
        return parse_document(store.get(blob), 'the capabilities document {}'.format(blob))
    except (CapabilityDenied, CollisionRefusal, MissingBlobRefusal):
        return UNREADABLE


def build_state(log, lsn=None):
    """The whole capability state folded out of `log` at or before `lsn` (default: the head).

    Read off the PLATTER, over `log.frames`, and that is a correctness property rather than a
    performance note. An index of where the fold's inputs were, derived when a handle opened, says
    nothing about where they are once somebody else has appended - and a handle outlives an append
    routinely, because reading never claims the home. A fold over such an index answers today's
    authorization question out of yesterday's policy: a revocation that does not reach the custody
    path, an as-of replay that is wrong AT THE HEAD, two folds on one handle looking at two logs.
    The record never trusts what it can re-derive, and this is the cheapest case of the law: read
    the stream, and filter it by the envelope's own type, which costs no key.
    """
    state = initial_state()
    for frame in log.frames(end_lsn=lsn):
        if frame['event_type'] not in CAPABILITY_EVENTS:
            continue
        apply_event(state, frame['event_type'], frame['actor'], log.open_body(frame), log.store)
    return state


def state_at(log, lsn=None):
    """`(document, genesis)` as of `lsn` - the fold every authorization question starts at.

    The document is `None` where none has been declared yet, the parsed object where one has, and
    `UNREADABLE` where one is in force and its blob no longer answers for it.

    `lsn` is what makes "could X do Y in March" a question with an answer: pass the position and the
    document that was in force there is the one that answers, because a declaration applies to the
    appends AFTER it and to no others.
    """
    state = build_state(log, lsn)
    return (state['doc'], state['genesis'])


def evaluate(doc, genesis, subject, verb, book):
    """May `subject` exercise `verb` over `book`? The one authorization function, and it is pure.

    Pure by construction: no log, no store, no clock, no home. Everything it needs was folded out of
    the record before it was called, so the same inputs answer the same way on the hub, on a replica
    and in a gate - which is what lets a replica evaluate an entitlement locally and reach the same
    verdict the hub would.

    The rules, and the reason each one is a rule:

      * THE WRITER'S OWN VERB is always yes: a denial that could itself be denied is a regress
        rather than a record.
      * THE RECOVERY is yes for the seat GENESIS NAMED and no for everybody else. It must survive a
        document that stranded every admin, which is why it is not read out of one - and it must
        still be a gated write path, because an unscoped actor appending free text into sealed
        bodies of the record is a write channel however it is spelled. A stranger's reach is refused
        and the reach recorded, which is a better fact than a use that granted nothing.
      * NO DOCUMENT means enforcement is off and the answer is yes. The writer does not even call
        this then; it is answered here so the function is total and so the fresh-home posture is
        stated in the same place as everything else.
      * A DOCUMENT IS THE WHOLE TRUTH about the six verbs. Genesis admin does not survive it, so a
        declaration can strand the last admin - deliberately, because an admin who cannot be revoked
        is not governed.
      * A RECOVERED ADMIN outranks the document, from the LSN the break-glass use landed at until
        the next declaration lands. This is the only grant that is not in a document, and it is in
        the log as a fact rather than in anybody's memory of a weekend.
      * AN UNREADABLE DOCUMENT grants nothing at all - it is in force and it cannot be read, so the
        six verbs are refused and the two above it are what is left to rescue the home with.
      * BOOK MATCHING: `*` matches every book and the firm-level facts that carry no book; a named
        grant matches only its own book, and never a book-less event - policy, checkpoints and
        official market declarations are firm-level acts, and a desk's book scope is not a licence
        to govern the deployment.
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
    """Every subject the document admits to READ `entitlement_class`, in declaration order.

    Enumeration, not evaluation - the question custody asks when it decides who a class key is
    wrapped to, which is "who", where `evaluate` answers "may this one". They read the same document
    and there is still exactly one evaluator.

    A document that will not read admits NOBODY, for the same reason it grants nothing: custody hands
    out a class key on the strength of this answer, and guessing at a policy is not a thing to hand a
    key on.
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

    `book` is the SCOPE the append needed rather than the book it carried: a firm-level event needs
    `*`, so that is what the denial names, and reading the fact back tells you what grant would have
    let it through instead of what field happened to be null.
    """
    return {'subject': subject, 'verb': verb,
            'book': book if book is not None else ANY_BOOK,
            'attempted_type': attempted_type}
