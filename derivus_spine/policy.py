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

"""The tolerance, firmness, fixings and tiers policy documents, and the fold that finds one in force.

Policy is data: hashed into the blob store and declared through the ordinary writer, as the
capabilities document is. Nothing here is a constant edited in a release - a deployment that wants a
tighter PV tolerance or a shorter market window declares one, and "what standard was this claim held
to in March" is a fold over the log. This shares no code with `capability.py`, which must fail
closed WITHOUT raising because its fold runs inside the writer's authorization hook; these two
documents are read by verbs instead, so a document that will not read raises where the verb stands.

The spine admits no tolerance of its own. `compare` below is the only floating-point comparison in
the package, it runs on numbers that came out of the engine, and every epsilon it uses was declared
by a deployment. A result class the policy does not name is refused rather than compared at a
default this module picked.

A firmness policy is optional where a tolerance policy is not, and it has no defaults: a home that
declares none has not said how stale a board may be, so nothing is refused on staleness and the age
measured is reported instead. A fixings policy is the authority an observation is read under, and it
names one source order per index: a home declaring none has said which administrator it believes
about nothing, so every index it is asked for refuses by name. A tiers policy is the workflow a
ticket falls through, and it has no default either: a home declaring none has declared no tiers, so
what an undeclared workflow means is the caller's decision rather than this module's.
"""
import json

from .canon import canonical_bytes
from .errors import MalformedEvent, ReplayRefused, SpineRefusal
from .vocabulary import is_hash, is_number, is_text

#: The policy names this module reads, and the whole of them. Reserved the way `capabilities` is: a
#: declaration under one of these names is read by a verb, so this module owns its shape.
TOLERANCE_POLICY = 'tolerance'
FIRMNESS_POLICY = 'firmness'
FIXINGS_POLICY = 'fixings'
TIERS_POLICY = 'tiers'

#: The tolerance document's one section: result class -> the absolute epsilon a replay of that
#: class may differ by. Closed at the field level, like an event body.
TOLERANCE_SECTION = 'tolerances'

#: The fixings document's one section: index -> the ordered administrators its prints are resolved
#: across, the first one holding a print winning.
FIXINGS_SECTION = 'sources'

#: The tiers document's two sections: the ORDERED list a ticket is matched against, and the market
#: each designated process resolves by name.
TIERS_SECTION = 'tiers'
DESIGNATIONS_SECTION = 'designations'

#: What a tier may declare. The first three name it and say who signs; the last three are CHECKS,
#: and a key a tier omits is a check it does not make, so a tier declaring none is the catch-all.
TIER_FIELDS = ('name', 'seat', 'four_eyes', 'max_notional', 'max_tenor_years', 'market')

#: The processes a designation binds, and the whole of them: a name nothing resolves by is a rule
#: nobody enforces.
DESIGNATED_PROCESSES = ('settlement_export',)

#: The prefix of a market one seat owns. A designated process never resolves one.
PRIVATE_MARKET = 'private/'

#: The firmness document's one field, and the whole of it: how old, in seconds, the oldest stamped
#: pillar of the board a quote was struck on may be. There is NO DEFAULT - a home declaring no
#: firmness policy has not said how stale is too stale, so it refuses nothing and reports the age.
PILLAR_SECONDS = 'pillar_seconds'

#: The two windows this one replaced, refused BY NAME rather than ignored: a document declaring one
#: would be a desk believing it had a rule that nothing enforces.
FIRMNESS_RETIRED = ('plan_seconds', 'values_seconds')

#: What a tiers document may not restate: the window and the verdict it answers. Pillar age is
#: checked on every booking before a tier is read.
FIRMNESS_FIELDS = (PILLAR_SECONDS, 'firm')


def parse_tolerance(document, where):
    """Check `document` as a tolerance policy and return it parsed. `where` names it in refusals.

    One section, every entry an absolute epsilon on one result class: `{"tolerances": {"mtm": 1e-9,
    "cva": 1e-6}}`. Anything else raises `MalformedEvent`.
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
    """Check `document` as a firmness policy and return it parsed.

    ONE WINDOW AND NO SECOND: `{"pillar_seconds": 900}`, a finite non-negative number of seconds,
    and the document may be empty. An unreadable window raises here, at the declaration, rather
    than later at a booking that could not act on it.

    The two windows this replaced are refused by name. `values_seconds` asked whether the board had
    moved since the quote, which is now REPORTED on the booking rather than refused - the market
    moves between a quote and the client's word, and the desk's own `firm_seconds` bounds it;
    `plan_seconds` aged a book that moves by booking rather than by clock, and the equality that
    catches a moved book needs no window at all.
    """
    def refuse(sentence):
        raise MalformedEvent('{}: {}'.format(where, sentence))

    if not isinstance(document, dict):
        refuse('a firmness policy is {}, not a JSON object - it is {{"{}": seconds}}'.format(
            type(document).__name__, PILLAR_SECONDS))
    retired = sorted(set(document) & set(FIRMNESS_RETIRED))
    if retired:
        refuse('a firmness policy carries {} - the board moving between a quote and its acceptance '
               'is REPORTED on the booking and never refused, and a book that moved is caught by '
               'the plan hash rather than by a clock. Declare {} instead, which is how old the '
               'board a quote was struck on may have been'.format(
                   ', '.join(retired), PILLAR_SECONDS))
    surplus = sorted(set(document) - {PILLAR_SECONDS})
    if surplus:
        refuse('a firmness policy carries {} beyond {} - the document is closed at the field level; '
               'a window nobody reads is a staleness rule nobody enforces'.format(
                   ', '.join(surplus), PILLAR_SECONDS))
    if PILLAR_SECONDS not in document:
        return {}
    window = document[PILLAR_SECONDS]
    if not is_number(window) or window < 0:
        refuse('{} is {!r} - a staleness window is a NUMBER of seconds and is never negative; a '
               'window that cannot be read is one no booking could be measured against'.format(
                   PILLAR_SECONDS, window))
    return {PILLAR_SECONDS: float(window)}


def parse_fixings(document, where):
    """Check `document` as a fixings policy and return it parsed. `where` names it in refusals.

    One section, an index and the ordered administrators its prints are resolved across:
    `{"sources": {"FxRate.ZAR": ["ECB", "BFIX"]}}`. The order IS the authority - the first named
    source holding a print is the fixing in force - so it is a list and never a set.
    """
    def refuse(sentence):
        raise MalformedEvent('{}: {}'.format(where, sentence))

    if not isinstance(document, dict):
        refuse('a fixings policy is {}, not a JSON object - it is {{"{}": {{index: [source, ...]}}}}'
               .format(type(document).__name__, FIXINGS_SECTION))
    surplus = sorted(set(document) - {FIXINGS_SECTION})
    if surplus:
        refuse('a fixings policy carries {} beyond {} - the document is closed at the field level; '
               'drop the key or version the document shape'.format(
                   ', '.join(surplus), FIXINGS_SECTION))
    entries = document.get(FIXINGS_SECTION)
    if not isinstance(entries, dict):
        refuse('{} is {}, not an object of index -> source order - a policy naming no index says so '
               'with an empty object, so that silence is never mistaken for absence'.format(
                   FIXINGS_SECTION, type(entries).__name__))
    for index, order in sorted(entries.items()):
        if not is_text(index):
            refuse('{} is keyed by {!r}, and an index that names nothing is not an index'.format(
                FIXINGS_SECTION, index))
        if not isinstance(order, list) or not order or not all(is_text(name) for name in order):
            refuse('the sources for {!r} are {!r}: an order is a non-empty LIST of administrator '
                   'names, and an index resolved across nobody is an index the policy should not '
                   'name'.format(index, order))
        elif len(set(order)) != len(order):
            refuse('the sources for {!r} name {} twice, so the order does not say which print '
                   'wins'.format(index, ', '.join(
                       sorted(name for name in set(order) if order.count(name) > 1))))
    return {FIXINGS_SECTION: dict((index, list(order)) for index, order in entries.items())}


def parse_tiers(document, where):
    """Check `document` as a tiers policy and return it COMPLETED. `where` names it in refusals.

    An ORDERED list of tiers, and the market each designated process resolves by name. A tier
    declares a unique name, optionally the seat an automatic approval signs under, and its checks -
    a key it omits is a check it does not make, so a tier declaring none is the catch-all the list
    ends at. Staleness is not among them: pillar age and book staleness are the firmness policy's
    two windows, checked on every booking before a tier is read, so a tier restating one is refused
    here rather than making one question two standards.

    WHAT IS STORED IS THE COMPLETED DOCUMENT, `parse_firmness`'s practice: the two defaults this
    shape has - `four_eyes` false on a tier that names no seat, and an empty `designations` - are
    written in before the bytes are hashed, so one workflow is ONE BLOB however the operator spelled
    it and two desks declaring the same rules do not get two governance histories of one decision.
    A tier that names a seat is completed with no `four_eyes` at all: the two keys together are
    refused, and an automatic seat never books.
    """
    def refuse(sentence):
        raise MalformedEvent('{}: {}'.format(where, sentence))

    if not isinstance(document, dict):
        refuse('a tiers policy is {}, not a JSON object - it is {{"{}": [tier, ...], "{}": '
               '{{process: market}}}}'.format(
                   type(document).__name__, TIERS_SECTION, DESIGNATIONS_SECTION))
    _unstaled(document, 'a tiers policy', refuse)
    surplus = sorted(set(document) - {TIERS_SECTION, DESIGNATIONS_SECTION})
    if surplus:
        refuse('a tiers policy carries {} beyond {} and {} - the document is closed at the field '
               'level; drop the key or version the document shape'.format(
                   ', '.join(surplus), TIERS_SECTION, DESIGNATIONS_SECTION))
    tiers = document.get(TIERS_SECTION)
    if not isinstance(tiers, list) or not tiers:
        refuse('{} is {!r}, not a non-empty ORDERED list - the FIRST tier whose every declared '
               'check passes is the one that applies, so the order IS the policy and a document '
               'naming no tier routes nothing at all'.format(TIERS_SECTION, tiers))
    named = set()
    for position, tier in enumerate(tiers):
        _tier(tier, position, named, refuse)
    return {TIERS_SECTION: [_completed(tier) for tier in tiers],
            DESIGNATIONS_SECTION: _designations(document, refuse)}


def _completed(tier):
    """`tier` with its one default written in: `four_eyes` false where the tier names no seat.

    A seated tier is left alone - it may not carry the key at all - so the completed document says
    of every tier exactly one thing about whether the approver may be the booker.
    """
    return dict(tier) if 'seat' in tier else dict({'four_eyes': False}, **tier)


def _tier(tier, position, named, refuse):
    """One tier of a tiers policy, checked. `named` collects the names as they are read, so a
    second tier spelled like the first is refused where it would otherwise shadow it."""
    at = 'tier {}'.format(position + 1)
    if not isinstance(tier, dict):
        refuse('{} is {}, not an object - a tier is {{"name": ..., check: bound, ...}}'.format(
            at, type(tier).__name__))
    name = tier.get('name')
    if not is_text(name):
        refuse('{} is named {!r}, and a tier that names nothing cannot be the answer to which tier '
               'a ticket falls in'.format(at, name))
    at = 'the {!r} tier'.format(name)
    if name in named:
        refuse('{} is declared twice, so the list does not say which of them applies - name them '
               'apart, or drop the one that says nothing'.format(at))
    named.add(name)
    _unstaled(tier, at, refuse)
    surplus = sorted(set(tier) - set(TIER_FIELDS))
    if surplus:
        refuse('{} carries {} beyond {} - the document is closed at the field level; a check nobody '
               'reads is a bound nobody enforces'.format(
                   at, ', '.join(surplus), ', '.join(TIER_FIELDS)))
    if 'seat' in tier and not is_text(tier['seat']):
        refuse('{} signs under seat {!r}, and a seat that names nothing signs nothing - name the '
               'subject the approval is attributed to, or leave the key out and let a human '
               'sign'.format(at, tier['seat']))
    if 'four_eyes' in tier and not isinstance(tier['four_eyes'], bool):
        refuse('{} declares four_eyes {!r}: it is true or false, and a value that is neither says '
               'nothing about whether the approver may be the booker'.format(
                   at, tier['four_eyes']))
    if 'seat' in tier and 'four_eyes' in tier:
        refuse('{} declares both a seat ({!r}) and four_eyes - an automatic seat never books, so '
               'the key would say nothing here; drop four_eyes, or drop the seat and let a human '
               'sign'.format(at, tier['seat']))
    if 'max_notional' in tier:
        _cap(tier['max_notional'], at, refuse)
    if 'max_tenor_years' in tier and (not is_number(tier['max_tenor_years'])
                                      or tier['max_tenor_years'] < 0):
        refuse('{} caps max_tenor_years at {!r} - a tenor bound is a finite number of years and is '
               'never negative'.format(at, tier['max_tenor_years']))
    if 'market' in tier:
        if not is_text(tier['market']):
            refuse('{} names market {!r}, and a market that names nothing is not one a quote\'s '
                   'values can be required to stand under'.format(at, tier['market']))
        _public(tier['market'], at, refuse)


def _cap(cap, at, refuse):
    """A tier's `max_notional`, checked: an amount and the currency it is an amount OF.

    Closed at those two. A cap that does not name its currency is a number nothing can compare a
    ticket against, and a policy check that crossed one currency into another would read a market.
    """
    if not isinstance(cap, dict) or sorted(cap) != ['amount', 'currency']:
        refuse('{} caps max_notional at {!r}: a cap is {{"amount": number, "currency": name}} and '
               'is closed at those two - a cap that does not name its currency is a number no '
               'ticket can be compared against'.format(at, cap))
    if not is_number(cap['amount']) or cap['amount'] < 0:
        refuse('{} caps max_notional at the amount {!r} - a cap is a finite number of its own '
               'currency and is never negative'.format(at, cap['amount']))
    if not is_text(cap['currency']):
        refuse('{} caps max_notional in the currency {!r}, and a currency that names nothing is not '
               'one a ticket can state its notional in'.format(at, cap['currency']))


def _designations(document, refuse):
    """The process -> market map of a tiers policy, checked and copied.

    Closed to `DESIGNATED_PROCESSES`, since a name nothing resolves by is a rule nobody enforces,
    and closed against a private market: a designated process resolves the market the firm declared
    and never one seat's own. A document designating nothing LEAVES THE SECTION OUT - a key stated
    and left empty of meaning is a document to fix rather than one to read past.
    """
    if DESIGNATIONS_SECTION not in document:
        return {}
    entries = document[DESIGNATIONS_SECTION]
    if not isinstance(entries, dict):
        refuse('{} is {}, not an object of process -> market name - a policy that designates '
               'nothing says so by leaving the section out'.format(
                   DESIGNATIONS_SECTION, type(entries).__name__))
    for process, market in sorted(entries.items()):
        if process not in DESIGNATED_PROCESSES:
            refuse('{} designates a market for {!r}, which no process resolves by name - the '
                   'designated processes are {}, and a name nothing reads is a rule nobody '
                   'enforces'.format(DESIGNATIONS_SECTION, process,
                                     ', '.join(DESIGNATED_PROCESSES)))
        if not is_text(market):
            refuse('{} designates {!r} for {}, and a market that names nothing is not one a process '
                   'can resolve'.format(DESIGNATIONS_SECTION, market, process))
        _public(market, '{} for {!r}'.format(DESIGNATIONS_SECTION, process), refuse)
    return dict(entries)


def _public(market, whose, refuse):
    """`market` asserted to be a name the firm declared rather than one seat's own.

    One check for the two places a policy points at a market. A designated process resolves the
    firm's board, and a tier's `market` decides whether an AUTOMATIC seat signs - the same character
    of decision - so neither may rest on a scratch board one seat declared for itself.
    """
    if market.startswith(PRIVATE_MARKET):
        refuse('{} names the market {!r}: a {} market is one seat\'s own, and neither a designated '
               'process nor a tier deciding whether an automatic seat signs prices on one - name '
               'the market the firm declared'.format(whose, market, PRIVATE_MARKET))


def _unstaled(carrying, whose, refuse):
    """Refuse a staleness window or its verdict declared in a tiers document.

    Pillar age IS the firmness policy's `pillar_seconds`, checked on every booking before any tier
    is read, so a second spelling of it would be two standards for one question.
    """
    restated = sorted(set(carrying) & (set(FIRMNESS_FIELDS) | set(FIRMNESS_RETIRED)))
    if restated:
        refuse('{} carries {} - how stale a board may be is the {} policy\'s {}, checked on EVERY '
               'booking before a tier is read, so a second spelling of it would be two standards '
               'for one question; declare the window under {} instead'.format(
                   whose, ', '.join(restated), FIRMNESS_POLICY, PILLAR_SECONDS, FIRMNESS_POLICY))


#: policy name -> the parser that reads it. `declare` refuses a name absent from this map rather
#: than store a document nobody could read back.
PARSERS = {TOLERANCE_POLICY: parse_tolerance, FIRMNESS_POLICY: parse_firmness,
           FIXINGS_POLICY: parse_fixings, TIERS_POLICY: parse_tiers}


def canonical_policy(policy, document, where=None):
    """`document` checked and canonicalised - the bytes a declaration puts in the store.

    One function, so what a verb accepts and what an operator declares cannot part company: the same
    policy spelled two ways would be two blobs and two histories of one decision.
    """
    parse = PARSERS.get(policy)
    if parse is None:
        raise MalformedEvent(
            '{!r} is not a policy this module reads - it owns {}, and a document declared under a '
            'name no reader knows is a decision nobody can ever apply. Declare it under one of '
            'those names, or through the ordinary open-bodied `policy_declared` if it is a policy '
            'this increment does not implement'.format(policy, ', '.join(sorted(PARSERS))))
    return canonical_bytes(parse(document, where or 'this {} policy'.format(policy)))


def declare(log, actor, policy, document, effective_time=None):
    """Put a policy document in the store and declare it, returning the envelope plus the blob.

    The blob is fsynced before the `policy_declared` naming its address appends. That event demands
    the `admin` scope, so only a governing seat can move the standard a claim is held to.
    """
    raw = canonical_policy(policy, document)
    blob = log.store.put(raw)
    envelope = log.append('policy_declared', {'policy': policy, 'blob': blob},
                          actor=actor, effective_time=effective_time, blob_refs=(blob,))
    return dict(envelope, policy=policy, blob=blob)


def in_force(log, policy, lsn=None):
    """`(blob, document, lsn)` for the declaration of `policy` standing at or before `lsn`, or
    `(None, None, None)` where the log carries none.

    Read off the platter over `log.frames` rather than from an index, since reading never claims the
    home and a handle routinely outlives someone else's append. Rows are located by envelope - only
    `policy_declared` bodies are opened - so the fold costs the length of the policy history.

    ONE WALK, ONE ANSWER: the position comes off the frame this fold CHOSE, so a reader that shows
    where a document was declared cannot show the blob of one declaration at the LSN of another.
    `policy_declared` is open-bodied, so a declaration carrying no blob is a fact about the name
    that this fold steps over, and joining a second fold to it would count it.

    A declaration whose blob no longer answers for it raises `MalformedEvent` rather than folding to
    a sentinel: unlike the capabilities fold, this one is called by a verb, so a refusal bricks
    nothing.
    """
    blob, at = None, None
    for frame in log.frames(end_lsn=lsn):
        if frame['event_type'] != 'policy_declared':
            continue
        body = log.open_body(frame)
        if not isinstance(body, dict) or body.get('policy') != policy:
            continue
        if not is_hash(body.get('blob')):
            continue
        blob, at = body['blob'], frame['lsn']
    if blob is None:
        return (None, None, None)
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
    return (blob, PARSERS[policy](document, where), at)


def compare(claimed, produced, tolerances):
    """Every way `produced` departs from `claimed`, named. An empty list is agreement.

    Structural first and numeric second: a shape that moved - a missing table, a row count that
    changed, a column relabelled - is a different answer no tolerance admits, while numbers inside a
    class compare against that class's declared epsilon. A class `tolerances` does not name is a
    departure, never a pass.
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
    """Walk one result class in lockstep and append every departure to `out`, under `path`.

    Numbers compare within `epsilon`; everything else compares for equality, a label or a date being
    a statement about the shape rather than a measurement inside it.
    """
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
    """`(blob, tolerances)` a replay claim is held to here.

    Raises `ReplayRefused` where no tolerance policy is declared: such a home has never said what
    "reproduces" means and so can attest nothing.
    """
    blob, document, _ = in_force(log, TOLERANCE_POLICY, lsn)
    if blob is None:
        raise ReplayRefused(
            'no {} policy is declared in this log, so there is no standard a replay claim could be '
            'held to and nothing is pinned: declare one ({{"{}": {{result class: epsilon}}}}) '
            'through `policy.declare` under an admin seat, then pin the result again'.format(
                TOLERANCE_POLICY, TOLERANCE_SECTION))
    return (blob, document[TOLERANCE_SECTION])


def firmness_in_force(log, lsn=None):
    """The firmness document standing at `lsn`, or the empty one where this home declared none.

    Absence is not a refusal and not a default: a home declaring no firmness policy refuses nothing
    on staleness, and the check reports the age it measured beside the window it had none of.
    """
    blob, document, _ = in_force(log, FIRMNESS_POLICY, lsn)
    return {} if blob is None else document


def tiers_in_force(log, lsn=None):
    """The tiers document standing at `lsn`, or None where this home declared none.

    Absence is neither a refusal nor a default: a home that declared no tiers has stated no
    workflow, and what a ticket falling through no tier means is the caller's decision rather than
    a document this module invented.
    """
    return in_force(log, TIERS_POLICY, lsn)[1]
