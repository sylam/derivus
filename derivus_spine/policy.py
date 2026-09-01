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

"""The two policy documents increment 3 declares - tolerance and firmness - and the fold that finds
the one in force.

Policy is DATA, hashed into the blob store and declared through the ordinary writer, exactly as the
capabilities document is. Nothing here is a constant somebody edits in a release: a deployment that
wants a tighter PV tolerance or a shorter market window declares one, the declaration is a fact, and
"what standard was this claim held to in March" is a fold over the log like every other question.
This module is `capability.py`'s shape applied to two smaller documents, and it deliberately shares
none of its code: capabilities decide who may write and must fail closed WITHOUT raising, because
the fold runs inside the writer's own authorization hook. These two are read by VERBS instead, so a
document that will not read raises where the verb stands and the verb refuses by name - fail closed
and loud, which is the right posture when the caller is a booking rather than the writer itself.

THE SPINE ADMITS NO TOLERANCE OF ITS OWN, and this is the file that has to say so out loud because
it is the one place an epsilon appears anywhere in the package. Hashes, folds, chains and plan
recompilation are bit-exact and always will be; the only floating-point comparison in the spine is
`compare` below, it runs on numbers that came out of the ENGINE, and every epsilon it uses was
declared by a deployment in a document the log carries. A result class the policy does not name is
therefore REFUSED rather than compared at some default this module picked - a default epsilon is an
undeclared policy, and an undeclared policy is exactly what a hashed policy blob exists to prevent.

The FIRMNESS defaults are the other side of that rule and they are stated rather than hidden,
because a firmness policy is optional in a way a tolerance policy is not: a home that declares none
is not making a claim about somebody else's numbers, it is quoting off its own book, and the two
windows below are the desk conventions the edge already runs on - one tick of the market cadence,
and the ten minutes `Quote Policy.firm_seconds` defaults to.
"""
import json

from .canon import canonical_bytes
from .errors import MalformedEvent, ReplayRefused, SpineRefusal
from .vocabulary import is_hash, is_number, is_text

#: The policy names this module reads, and the whole of them. They are reserved the way
#: `capabilities` is: a declaration under one of these names is read by a verb, so its shape is this
#: module's business rather than the declaring operator's.
TOLERANCE_POLICY = 'tolerance'
FIRMNESS_POLICY = 'firmness'

#: The tolerance document's one section: result class -> the absolute epsilon a replay of that class
#: may differ by. Closed at the field level like an event body, because a key no comparison reads is
#: a tolerance nobody enforces.
TOLERANCE_SECTION = 'tolerances'

#: The firmness document's two windows, in seconds, and what a home that declares none runs on.
#:
#: `values_seconds` is one beat of `DV_Service --tick`: the market pin a quote was struck on is
#: refreshed on that cadence, so a pin older than one beat is a pin nobody refreshed and the quote
#: is priced against a board that has stopped being watched. `plan_seconds` is the ten minutes the
#: desk's own `Quote Policy.firm_seconds` defaults to - the book moves by BOOKING rather than by
#: clock, so on this dimension the hash comparison is the real test and the window is the backstop
#: under it.
FIRMNESS_DEFAULTS = {'values_seconds': 30.0, 'plan_seconds': 600.0}


def parse_tolerance(document, where):
    """A tolerance policy, read and checked to the last key. Answers the parsed document.

    One section, and every entry an absolute epsilon on one result class: `{"tolerances": {"mtm":
    1e-9, "cva": 1e-6}}`. Tighter for a PV, looser for a Monte Carlo greek, same-path by recorded
    seed - the brief's own sentence, spelled by the deployment rather than by this module.
    """
    def refuse(sentence):
        raise MalformedEvent('{}: {}'.format(where, sentence))

    if not isinstance(document, dict):
        refuse('a tolerance policy is {}, not a JSON object - it is {{"{}": {{class: epsilon}}}}'
               .format(type(document).__name__, TOLERANCE_SECTION))
    surplus = sorted(set(document) - {TOLERANCE_SECTION})
    if surplus:
        refuse('a tolerance policy carries {} beyond {} - the document is closed at the field '
               'level; drop the key or version the document shape'.format(
                   ', '.join(surplus), TOLERANCE_SECTION))
    entries = document.get(TOLERANCE_SECTION)
    if not isinstance(entries, dict):
        refuse('{} is {}, not an object of result class -> epsilon - a policy that tolerates '
               'nothing says so with an empty object, so that silence is never mistaken for '
               'absence'.format(TOLERANCE_SECTION, type(entries).__name__))
    for name, epsilon in sorted(entries.items()):
        if not is_text(name):
            refuse('{} is keyed by {!r}, and a class that names nothing is not a class'.format(
                TOLERANCE_SECTION, name))
        if not is_number(epsilon) or epsilon < 0:
            refuse('the tolerance on {!r} is {!r}: an epsilon is a finite number of the result\'s '
                   'own units and is never negative'.format(name, epsilon))
    return {TOLERANCE_SECTION: dict(entries)}


def parse_firmness(document, where):
    """A firmness policy, read and checked. Answers the parsed document with the defaults filled in.

    Two windows and no third: `{"values_seconds": 30, "plan_seconds": 600}`, each optional, each a
    finite non-negative number of seconds. A window that will not READ refuses here, where the
    document is declared and the operator still has the file open - never at the approval, where it
    would refuse a quote for a reason the salesperson cannot act on.
    """
    def refuse(sentence):
        raise MalformedEvent('{}: {}'.format(where, sentence))

    if not isinstance(document, dict):
        refuse('a firmness policy is {}, not a JSON object - it is {{{}}}'.format(
            type(document).__name__,
            ', '.join('"{}": seconds'.format(name) for name in sorted(FIRMNESS_DEFAULTS))))
    surplus = sorted(set(document) - set(FIRMNESS_DEFAULTS))
    if surplus:
        refuse('a firmness policy carries {} beyond {} - the document is closed at the field level; '
               'a window nobody reads is a staleness rule nobody enforces'.format(
                   ', '.join(surplus), ', '.join(sorted(FIRMNESS_DEFAULTS))))
    read = dict(FIRMNESS_DEFAULTS)
    for name in sorted(FIRMNESS_DEFAULTS):
        if name not in document:
            continue
        window = document[name]
        if not is_number(window) or window < 0:
            refuse('{} is {!r} - a staleness window is a NUMBER of seconds and is never negative; '
                   'a window that cannot be read is one no approval could be measured against'
                   .format(name, window))
        read[name] = float(window)
    return read


#: policy name -> the parser that reads it. The map is what makes `declare` refuse a name this
#: module does not own, rather than storing a document nobody will ever be able to read back.
PARSERS = {TOLERANCE_POLICY: parse_tolerance, FIRMNESS_POLICY: parse_firmness}


def canonical_policy(policy, document, where=None):
    """`document` checked and canonicalised - the bytes a declaration puts in the store.

    One function so that what a verb accepts and what an operator declares cannot part company,
    which is `capability.canonical_document`'s reason spelled again: the same policy spelled two
    ways would be two blobs and two histories of one decision.
    """
    parse = PARSERS.get(policy)
    if parse is None:
        raise MalformedEvent(
            '{!r} is not a policy this module reads - it owns {}, and a document declared under a '
            'name no reader knows is a decision nobody can ever apply. Declare it under one of '
            'those names, or through the ordinary open-bodied `policy_declared` if it is a policy '
            'this increment does not implement'.format(policy, ' and '.join(sorted(PARSERS))))
    return canonical_bytes(parse(document, where or 'this {} policy'.format(policy)))


def declare(log, actor, policy, document, effective_time=None):
    """Put a policy document in the store and declare it. Answers the envelope plus the blob.

    The durability law, obeyed the way every other declaration obeys it: the blob is fsynced first,
    then the `policy_declared` naming its address appends. `policy_declared` demands `admin`, so a
    seat that may not govern the deployment cannot move the standard a replay claim is held to.
    """
    raw = canonical_policy(policy, document)
    blob = log.store.put(raw)
    envelope = log.append('policy_declared', {'policy': policy, 'blob': blob},
                          actor=actor, effective_time=effective_time, blob_refs=(blob,))
    return dict(envelope, policy=policy, blob=blob)


def in_force(log, policy, lsn=None):
    """`(blob, document)` for the declaration of `policy` standing at or before `lsn`, or
    `(None, None)` where the log carries none.

    Read off the PLATTER over `log.frames`, for `capability.build_state`'s reason: an index of where
    the fold's inputs were says nothing about where they are once somebody else has appended, and a
    handle outlives an append routinely because reading never claims the home. Located by ENVELOPE
    (only `policy_declared` rows are opened) so the fold costs the length of the policy history
    rather than of the book.

    A declaration whose blob no longer answers for it RAISES here rather than folding to a
    sentinel. That is the opposite of the capabilities fold and deliberately so: this one is called
    by a verb, not by the writer's authorization hook, so nothing is bricked by a refusal and the
    honest answer to "what tolerance governs this pin" is that the record no longer says.
    """
    blob = None
    for frame in log.frames(end_lsn=lsn):
        if frame['event_type'] != 'policy_declared':
            continue
        body = log.open_body(frame)
        if not isinstance(body, dict) or body.get('policy') != policy:
            continue
        if not is_hash(body.get('blob')):
            continue
        blob = body['blob']
    if blob is None:
        return (None, None)
    where = 'the {} policy {}'.format(policy, blob)
    try:
        raw = log.store.get(blob)
    except SpineRefusal as missing:
        raise MalformedEvent(
            '{} is declared and its blob does not answer for it ({}): a policy the record cannot '
            'read is a standard nobody can be held to - restore blobs/ from a verified replica, or '
            'declare a replacement'.format(where, missing))
    try:
        document = json.loads(bytes(raw).decode('utf-8'))
    except (TypeError, UnicodeDecodeError, ValueError):
        raise MalformedEvent(
            '{} is not JSON - the blob was altered under its own address; restore blobs/ from a '
            'verified replica, or declare a replacement'.format(where))
    return (blob, PARSERS[policy](document, where))


def compare(claimed, produced, tolerances):
    """Every way `produced` departs from `claimed`, named. An empty list is agreement.

    The comparison is STRUCTURAL first and numeric second, and only the numbers get an epsilon. Two
    results whose shapes differ - a table one of them does not have, a row count that moved, a
    column relabelled - are not two readings of one number, they are two different answers, and no
    tolerance admits that. Numbers inside a class compare against that class's own declared epsilon.

    A class the policy does not name is a departure, not a pass. The spine has no epsilon of its
    own to fall back on (see this module's header), and quietly admitting an unnamed class would
    attest a claim against a standard nobody declared - which is the failure this whole mechanism
    exists to make impossible.
    """
    departures = []
    for name in sorted(set(claimed) | set(produced)):
        if name not in produced:
            departures.append('the replay produced no {!r}, which the claim carries'.format(name))
            continue
        if name not in claimed:
            departures.append('the replay produced {!r}, which the claim does not carry'.format(
                name))
            continue
        if claimed[name] == produced[name]:
            continue
        if name not in tolerances:
            departures.append(
                '{!r} differs and the tolerance policy declares no epsilon for it - the spine '
                'admits no tolerance of its own, so an unnamed result class is refused rather than '
                'compared at a default nobody declared'.format(name))
            continue
        _departures(claimed[name], produced[name], float(tolerances[name]), name, departures)
    return departures


def _departures(claimed, produced, epsilon, path, out):
    """Walk one result class in lockstep and append every departure. Numbers get the epsilon;
    everything else is compared for equality, because a label, a date or a row count is a statement
    about the shape rather than a measurement inside it."""
    if isinstance(claimed, dict) or isinstance(produced, dict):
        if not (isinstance(claimed, dict) and isinstance(produced, dict)):
            out.append('{}: the claim holds {} where the replay holds {}'.format(
                path, type(claimed).__name__, type(produced).__name__))
            return
        for key in sorted(set(claimed) | set(produced)):
            if key not in claimed or key not in produced:
                out.append('{}.{}: present on one side only - a shape that moved is a different '
                           'answer, not a number inside a tolerance'.format(path, key))
                continue
            _departures(claimed[key], produced[key], epsilon, '{}.{}'.format(path, key), out)
        return
    if isinstance(claimed, list) or isinstance(produced, list):
        if not (isinstance(claimed, list) and isinstance(produced, list)):
            out.append('{}: the claim holds {} where the replay holds {}'.format(
                path, type(claimed).__name__, type(produced).__name__))
            return
        if len(claimed) != len(produced):
            out.append('{}: the claim holds {} row(s) and the replay {} - a row count that moved '
                       'is a different answer'.format(path, len(claimed), len(produced)))
            return
        for position, (one, other) in enumerate(zip(claimed, produced)):
            _departures(one, other, epsilon, '{}[{}]'.format(path, position), out)
        return
    if is_number(claimed) and is_number(produced):
        if abs(float(claimed) - float(produced)) > epsilon:
            out.append('{}: the claim says {!r} and the replay {!r}, which is {:g} apart against a '
                       'declared tolerance of {:g}'.format(
                           path, claimed, produced, abs(float(claimed) - float(produced)), epsilon))
        return
    if claimed != produced:
        out.append('{}: the claim says {!r} and the replay {!r} - neither is a number, so there is '
                   'no epsilon that makes them one answer'.format(path, claimed, produced))


def tolerances_in_force(log, lsn=None):
    """`(blob, tolerances)` a replay claim is held to here, or `ReplayRefused` naming the remedy.

    Separate from `in_force` because the ABSENCE means different things to the two documents: a home
    with no firmness policy runs on the stated defaults, while a home with no tolerance policy has
    never said what "reproduces" means and so cannot attest anything at all. The brief puts the
    tolerance policy in the acceptance criteria as a written artifact for exactly that reason.
    """
    blob, document = in_force(log, TOLERANCE_POLICY, lsn)
    if blob is None:
        raise ReplayRefused(
            'no {} policy is declared in this log, so there is no standard a replay claim could be '
            'held to and nothing is pinned: declare one ({{"{}": {{result class: epsilon}}}}) '
            'through `policy.declare` under an admin seat, then pin the result again'.format(
                TOLERANCE_POLICY, TOLERANCE_SECTION))
    return (blob, document[TOLERANCE_SECTION])


def firmness_in_force(log, lsn=None):
    """The firmness windows standing here - the declared ones, or `FIRMNESS_DEFAULTS`.

    A home that has declared none is not making a claim about anybody else's numbers, so the stated
    desk conventions are the honest answer rather than a refusal. Which windows were used is
    reported by the check itself, so a quote refused on a default says which default it met.
    """
    blob, document = in_force(log, FIRMNESS_POLICY, lsn)
    return dict(FIRMNESS_DEFAULTS) if blob is None else document
